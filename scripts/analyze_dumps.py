"""Selective prediction and calibration from the per-pixel dumps of the vis protocol (stage `dump` of
scripts/run_h100.py, run family P6-dump). The benchmark's AUROC / CUR say whether a detector *ranks* single-view
pixels below supported ones; this script asks what a user gets from abstaining with it:

  risk-coverage   keep the pixels with confidence above a threshold; coverage = fraction of known pixels kept,
                  risk = fraction of kept pixels that are single-view (the hallucination rate of what survives),
                  and the AbsRel of the kept depth. AURC = area under risk(coverage) over coverage in [0, 1]
                  (lower is better; the oracle keeps all S before any U).
  calibration     for the heads, 1 - conf = P(imagined): expected calibration error against the empirical
                  single-view rate in 10 bins; native confidence has no probability reading and is skipped.
  operating point coverage and risk at the paper's thresholds (observed > 0.7, imagined < 0.3, abstain between)
                  for the heads, and at the threshold that keeps 90% of S (the CUR threshold) for every detector.

Usage: python scripts/analyze_dumps.py results/P6-dump/dtu [results/P6-dump/co3d ...] [--out results/P6-dump]
Writes <out>/P6-dump-<dataset>_curves.csv (per detector: AURC, risk at coverage 0.5 / 0.8 / 0.9, ECE, operating
points) and <out>/P6-dump-<dataset>_rc.csv (the mean risk-coverage curve on a 1% grid, for the figure).
Pixels of U and S only (the labelled set); every cell (scene, seed, view count) weighs the same."""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HEAD_RE = ("oiavis", "oiavisself", "oiavispatch", "oiavisselfpatch", "oiavislin", "oia", "oiapatch")


def is_head(model):
    v = model.split("_", 1)[1] if "_" in model else ""
    v = v.split("_x")[0].split("_s")[0]
    return v in HEAD_RE


def risk_coverage(conf, is_u, err, grid):
    """risk(c) and absrel(c) at coverage c in grid, ranking by conf (ties broken by position)."""
    order = np.argsort(-conf, kind="stable")
    u = is_u[order].astype(np.float64)
    e = err[order]
    cum_u = np.cumsum(u)
    cum_e = np.cumsum(np.where(np.isfinite(e), e, 0.0))
    cum_n = np.cumsum(np.isfinite(e))
    n = len(order)
    k = np.clip((grid * n).astype(int), 1, n)
    return cum_u[k - 1] / k, cum_e[k - 1] / np.maximum(cum_n[k - 1], 1)


def ece(p_imagined, is_u, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_imagined, edges) - 1, 0, bins - 1)
    out, n = 0.0, len(p_imagined)
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.sum() / n * abs(p_imagined[m].mean() - is_u[m].mean())
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grid = np.linspace(0.01, 1.0, 100)
    for d in a.dirs:
        name = os.path.basename(os.path.normpath(d))
        out = a.out or os.path.dirname(os.path.normpath(d))
        labels = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(d, "labels", "*.npz"))}
        rows, curves = [], []
        for mdir in sorted(glob.glob(os.path.join(d, "*"))):
            model = os.path.basename(mdir)
            if model == "labels" or not os.path.isdir(mdir):
                continue
            per_cell = []
            for p in sorted(glob.glob(os.path.join(mdir, "*.npz"))):
                key = os.path.basename(p)[:-4]
                if key not in labels:
                    continue
                L = np.load(labels[key]); M = np.load(p)
                U, S, gt = L["U"], L["S"], L["gt"]
                lab = U | S
                conf = M["conf"].astype(np.float64)[lab]
                depth = M["depth"].astype(np.float64)[lab]
                err = np.abs(depth - gt[lab]) / np.maximum(gt[lab], 1e-9)
                is_u = U[lab]
                r, e = risk_coverage(conf, is_u, err, grid)
                cell = dict(key=key, aurc=float(np.trapz(r, grid) / (grid[-1] - grid[0])),
                            risk50=float(r[49]), risk80=float(r[79]), risk90=float(r[89]),
                            absrel50=float(e[49]), absrel90=float(e[89]), frac_u=float(is_u.mean()))
                # oracle AURC for this cell (all S first): risk is 0 up to coverage 1 - frac_u, then rises
                ro = np.maximum(0.0, 1 - (1 - is_u.mean()) / grid)
                cell["aurc_oracle"] = float(np.trapz(ro, grid) / (grid[-1] - grid[0]))
                tau = np.quantile(conf[~is_u], 0.10)          # keeps 90% of S: the CUR operating point
                keep = conf >= tau
                cell["cov_at_s90"], cell["risk_at_s90"] = float(keep.mean()), float(is_u[keep].mean())
                if is_head(model):
                    p_im = np.clip(1.0 - conf, 0, 1)
                    cell["ece"] = ece(p_im, is_u)
                    obs, im = conf > 0.7, conf < 0.3
                    cell["frac_observed"], cell["frac_abstain"], cell["frac_imagined"] = float(obs.mean()), float((~obs & ~im).mean()), float(im.mean())
                    cell["risk_observed"] = float(is_u[obs].mean()) if obs.any() else float("nan")
                    cell["u_recall_imagined"] = float((im & is_u).sum() / max(is_u.sum(), 1))
                per_cell.append(cell)
                curves.append(pd.DataFrame(dict(model=model, key=key, coverage=grid, risk=r, absrel=e)))
            if per_cell:
                df = pd.DataFrame(per_cell)
                row = dict(model=model, n_cells=len(df))
                for c in df.columns:
                    if c != "key":
                        row[c] = float(df[c].mean())
                rows.append(row)
                print(f"[{name}] {model:28s} cells {len(df):3d}  AURC {row['aurc']:.3f} (oracle {row['aurc_oracle']:.3f})  "
                      f"risk@cov0.8 {row['risk80']:.3f}  risk@S90 {row['risk_at_s90']:.3f} cov {row['cov_at_s90']:.3f}"
                      + (f"  ECE {row['ece']:.3f}  observed {row['frac_observed']:.2f} / abstain {row['frac_abstain']:.2f} / "
                         f"imagined {row['frac_imagined']:.2f}, risk in observed {row['risk_observed']:.3f}" if "ece" in row else ""))
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out, f"P6-dump-{name}_curves.csv"), index=False)
            rc = pd.concat(curves, ignore_index=True).groupby(["model", "coverage"], as_index=False)[["risk", "absrel"]].mean()
            rc.to_csv(os.path.join(out, f"P6-dump-{name}_rc.csv"), index=False)
            print(f"-> {out}/P6-dump-{name}_curves.csv, _rc.csv")


if __name__ == "__main__":
    main()
