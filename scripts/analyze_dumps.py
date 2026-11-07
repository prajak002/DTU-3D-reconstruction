"""Selective prediction, fixed operating points and calibration from the per-pixel dumps of the vis protocol
(stage `dump` of scripts/run_h100.py, run family P6-dump). The benchmark's AUROC / CUR say whether a detector
*ranks* single-view pixels below supported ones; this script asks what a user gets from abstaining with it:

  risk-coverage   keep the pixels with confidence above a threshold; coverage = fraction of labelled pixels kept,
                  risk = fraction of kept pixels that are single-view (the hallucination rate of what survives),
                  and the AbsRel of the kept depth. AURC = area under risk(coverage) (lower is better; the oracle
                  keeps every supported pixel before any single-view one).
  frozen thresholds
                  CUR in the benchmark uses the test cell's own supported pixels to set the threshold (an oracle).
                  Two deployable alternatives: (a) leave-one-scene-out, the threshold that keeps 90% of the supported
                  pixels pooled over the OTHER scenes of the same dataset; (b) --tau-from DIR, the threshold frozen on
                  another dataset's dumps (DTU) and applied unchanged here. For each: fraction of single-view pixels
                  that pass (cur_*), fraction of supported pixels kept (keep_*) and AbsRel of the kept pixels.
  calibration     for the heads, 1 - conf = P(imagined): expected calibration error against the empirical single-view
                  rate in 10 bins; native confidence has no probability reading and is skipped.
  bands           the paper's fixed bands (observed > 0.7, imagined < 0.3, abstain between) for the heads: fraction of
                  pixels per band, single-view rate and AbsRel inside the observed band, recall of single-view pixels
                  in the imagined band.

Usage: python scripts/analyze_dumps.py results/P6-dump/dtu [results/P6-dump/co3d ...] [--tau-from results/P6-dump/dtu] [--out DIR]
Writes <out>/P6-dump-<dataset>_curves.csv (one row per detector) and <out>/P6-dump-<dataset>_rc.csv (the mean
risk-coverage curve on a 1% grid, for the figure). Pixels of U and S only; every cell (scene, seed, view count)
weighs the same."""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HEAD_VARIANTS = ("oiavis", "oiavisself", "oiavispatch", "oiavisselfpatch", "oiavislin", "oia", "oiapatch")
GRID = np.linspace(0.01, 1.0, 100)


def is_head(model):
    v = model.split("_", 1)[1] if "_" in model else ""
    v = v.split("_x")[0].split("_s")[0]
    return v in HEAD_VARIANTS


def load_cells(d, max_s=20000):
    """{model: [cell dicts]} with conf, is_u, err over the labelled pixels and a subsample of the supported confidences."""
    labels = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(d, "labels", "*.npz"))}
    out = {}
    rng = np.random.default_rng(0)
    for mdir in sorted(glob.glob(os.path.join(d, "*"))):
        model = os.path.basename(mdir)
        if model == "labels" or not os.path.isdir(mdir):
            continue
        cells = []
        for p in sorted(glob.glob(os.path.join(mdir, "*.npz"))):
            key = os.path.basename(p)[:-4]
            if key not in labels:
                continue
            L = np.load(labels[key]); M = np.load(p)
            U, S, gt = L["U"], L["S"], L["gt"]
            lab = U | S
            conf = M["conf"].astype(np.float32)[lab]
            depth = M["depth"].astype(np.float32)[lab]
            err = (np.abs(depth - gt[lab]) / np.maximum(gt[lab], 1e-9)).astype(np.float32)
            is_u = U[lab]
            s = conf[~is_u]
            if len(s) > max_s:
                s = rng.choice(s, max_s, replace=False)
            cells.append(dict(key=key, scene=key.split("__")[0], conf=conf, is_u=is_u, err=err, s_sample=s))
        if cells:
            out[model] = cells
    return out


def risk_coverage(conf, is_u, err):
    order = np.argsort(-conf, kind="stable")
    u, e = is_u[order].astype(np.float64), err[order].astype(np.float64)
    cum_u, cum_e = np.cumsum(u), np.cumsum(e)
    n = len(order)
    k = np.clip((GRID * n).astype(int), 1, n)
    return cum_u[k - 1] / k, cum_e[k - 1] / k


def ece(p_imagined, is_u, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_imagined, edges) - 1, 0, bins - 1)
    out, n = 0.0, len(p_imagined)
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.sum() / n * abs(p_imagined[m].mean() - is_u[m].mean())
    return float(out)


def at_tau(c, tau):
    """At threshold tau: cur = fraction of single-view pixels kept (the benchmark's CUR), risk = fraction of kept
    pixels that are single-view, keep = fraction of supported pixels kept, absrel of the kept pixels."""
    keep = c["conf"] >= tau
    u = c["is_u"]
    return dict(cur=float(keep[u].mean()) if u.any() else float("nan"), risk=float(u[keep].mean()) if keep.any() else float("nan"),
                keep=float(keep[~u].mean()), absrel=float(c["err"][keep].mean()) if keep.any() else float("nan"))


def analyse(name, cells_by_model, out, tau_source=None):
    rows, curves = [], []
    for model, cells in cells_by_model.items():
        pooled = {sc: np.concatenate([c["s_sample"] for c in cells if c["scene"] == sc]) for sc in {c["scene"] for c in cells}}
        tau_frozen = float(np.quantile(np.concatenate(list(tau_source[model].values())), 0.10)) if tau_source and model in tau_source else None
        per = []
        for c in cells:
            conf, is_u, err = c["conf"], c["is_u"], c["err"]
            r, e = risk_coverage(conf, is_u, err)
            cell = dict(aurc=float(np.trapezoid(r, GRID) / (GRID[-1] - GRID[0])), risk50=float(r[49]), risk80=float(r[79]),
                        risk90=float(r[89]), absrel_all=float(err.mean()), absrel50=float(e[49]), absrel90=float(e[89]),
                        frac_u=float(is_u.mean()))
            ro = np.maximum(0.0, 1 - (1 - is_u.mean()) / GRID)
            cell["aurc_oracle"] = float(np.trapezoid(ro, GRID) / (GRID[-1] - GRID[0]))
            tau_own = float(np.quantile(conf[~is_u], 0.10))               # the benchmark's oracle threshold
            cell.update({f"{k}_own": v for k, v in at_tau(c, tau_own).items()})
            others = np.concatenate([v for sc, v in pooled.items() if sc != c["scene"]]) if len(pooled) > 1 else None
            if others is not None:
                cell.update({f"{k}_loso": v for k, v in at_tau(c, float(np.quantile(others, 0.10))).items()})
            if tau_frozen is not None:
                cell.update({f"{k}_frozen": v for k, v in at_tau(c, tau_frozen).items()})
            if is_head(model):
                p_im = np.clip(1.0 - conf, 0, 1)
                cell["ece"] = ece(p_im, is_u)
                obs, im = conf > 0.7, conf < 0.3
                cell["frac_observed"], cell["frac_abstain"], cell["frac_imagined"] = float(obs.mean()), float((~obs & ~im).mean()), float(im.mean())
                cell["risk_observed"] = float(is_u[obs].mean()) if obs.any() else float("nan")
                cell["absrel_observed"] = float(err[obs].mean()) if obs.any() else float("nan")
                cell["u_recall_imagined"] = float((im & is_u).sum() / max(is_u.sum(), 1))
            per.append(cell)
            curves.append(pd.DataFrame(dict(model=model, key=c["key"], coverage=GRID, risk=r, absrel=e)))
        df = pd.DataFrame(per)
        row = dict(model=model, n_cells=len(df), tau_frozen=tau_frozen if tau_frozen is not None else float("nan"))
        row.update({col: float(df[col].mean()) for col in df.columns})
        rows.append(row)
        msg = (f"[{name}] {model:28s} cells {len(df):3d}  AURC {row['aurc']:.3f} (oracle {row['aurc_oracle']:.3f})  "
               f"AbsRel all {row['absrel_all']:.4f} / top-50% {row['absrel50']:.4f}  CUR own {row['cur_own']:.3f}")
        if "cur_loso" in row:
            msg += f"  loso {row['cur_loso']:.3f} (keep {row['keep_loso']:.2f})"
        if "cur_frozen" in row:
            msg += f"  frozen {row['cur_frozen']:.3f} (keep {row['keep_frozen']:.2f})"
        if "ece" in row:
            msg += (f"  ECE {row['ece']:.3f}  bands {row['frac_observed']:.2f}/{row['frac_abstain']:.2f}/{row['frac_imagined']:.2f}"
                    f"  risk in observed {row['risk_observed']:.3f}")
        print(msg)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, f"P6-dump-{name}_curves.csv"), index=False)
        rc = pd.concat(curves, ignore_index=True).groupby(["model", "coverage"], as_index=False)[["risk", "absrel"]].mean()
        rc.to_csv(os.path.join(out, f"P6-dump-{name}_rc.csv"), index=False)
        print(f"-> {out}/P6-dump-{name}_curves.csv, _rc.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tau-from", default=None, help="dump dir whose supported-pixel confidences freeze the threshold")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    loaded = {}
    tau_source = None
    if a.tau_from:
        src = os.path.normpath(a.tau_from)
        loaded[src] = load_cells(src)
        tau_source = {m: {c["scene"]: c["s_sample"] for c in cells} for m, cells in loaded[src].items()}
        # pooled per scene: concatenate the samples of all cells of a scene
        tau_source = {m: {sc: np.concatenate([c["s_sample"] for c in cells if c["scene"] == sc]) for sc in {c["scene"] for c in cells}}
                      for m, cells in loaded[src].items()}
    for d in a.dirs:
        d = os.path.normpath(d)
        name = os.path.basename(d)
        out = a.out or os.path.dirname(d)
        cells = loaded.get(d) or load_cells(d)
        if not cells:
            print(f"[{name}] no dumps under {d}"); continue
        analyse(name, cells, out, tau_source if (tau_source and d != os.path.normpath(a.tau_from)) else None)


if __name__ == "__main__":
    main()
