"""Statistics of the ninth pass (pre-registered Gates 15 to 17 in LOG.md, commit d4aebbd), generated from the CSVs.

  Gate 15  results/P16-nbv/{a,b}/P16-nbv.csv                  next-view selection, DTU benchmark, four backbones
  Gate 16  results/P16-oneside/{a,b}/P12-realocc.csv           real occlusion with one-sided captures
           results/P12-realocc/P12-realocc.csv                 the spread protocol at the same N (Gate 10 rows)
  Gate 17  results/P16-co3dadapt/{A,B}/P12-realocc-co3d.csv    CO3Dv2 real occlusion, heads adapted on the other fold
           results/P14-realocc/holes/P14-holes-co3d.csv        validity check (Gate 12)

  python scripts/make_stats_p16.py    writes results/P16-stats.md and docs/assets/p16.json"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
BASES = ["vggt", "pi3x", "pi3", "da3"]


def load(pattern):
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(R, pattern)))]
    return pd.concat(parts, ignore_index=True) if parts else None


def gate15(L, J):
    df = load("P16-nbv/*/P16-nbv.csv")
    L += ["## Gate 15: next-view selection (DTU benchmark)", "",
          "Gain: share of the surface no input view saw that the chosen next view sees. Pass per cell: head beats native "
          "and random at p < 0.05 (paired sign-flip over scans). Pass: at least 6 of 8 cells.", "",
          "| backbone | views | scans | unseen share | random | farthest | native | head | oracle | head - native (p) | head - random (p) | head - farthest (p) | cell passes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if df is None:
        L += ["no rows", ""]; return
    J["gate15"] = {}
    n_pass = n_cells = 0
    for b in BASES:
        for n in (3, 5):
            x = df[(df.backbone == b) & (df.n_views == n)]
            if x.empty:
                continue
            long = pd.concat([x.assign(model=r, v=x[f"gain_{r}"]) for r in
                              ("head", "native", "random", "farthest", "oracle")], ignore_index=True)
            m = {r: float(long[long.model == r].groupby("scan").v.mean().mean()) for r in
                 ("head", "native", "random", "farthest", "oracle")}
            tn = H.paired_permutation(long, "v", "head", "native")
            tr = H.paired_permutation(long, "v", "head", "random")
            tf = H.paired_permutation(long, "v", "head", "farthest")
            ok = bool(tn[0] > 0 and tn[1] < 0.05 and tr[0] > 0 and tr[1] < 0.05)
            n_cells += 1; n_pass += ok
            un = float(x.groupby("scan").frac_gt_unobserved.mean().mean())
            J["gate15"][f"{b}:{n}"] = dict(**{k: round(v, 4) for k, v in m.items()}, unseen=round(un, 4),
                                           head_native=[round(tn[0], 4), round(tn[1], 5)],
                                           head_random=[round(tr[0], 4), round(tr[1], 5)],
                                           head_farthest=[round(tf[0], 4), round(tf[1], 5)], passes=ok,
                                           n_scans=int(x.scan.nunique()))
            L.append(f"| {b} | {n} | {x.scan.nunique()} | {un:.3f} | {m['random']:.3f} | {m['farthest']:.3f} | "
                     f"{m['native']:.3f} | {m['head']:.3f} | {m['oracle']:.3f} | {tn[0]:+.3f} ({tn[1]:.3g}) | "
                     f"{tr[0]:+.3f} ({tr[1]:.3g}) | {tf[0]:+.3f} ({tf[1]:.3g}) | {'yes' if ok else 'no'} |")
    verdict = "passes" if n_pass >= 6 else "fails"
    J["gate15_total"] = dict(cells=n_cells, pass_cells=n_pass, verdict=verdict)
    L += ["", f"**Gate 15 {verdict}:** {n_pass} of {n_cells} cells.", ""]


def realocc_cells(rd, n, b, head_name):
    x = rd[rd.n_views == n]
    if b not in x.model.values or head_name not in x.model.values:
        return None
    y = x[x.model == b]
    fr = float(y.groupby("scan").frac_unobs.mean().mean())
    an = float(y.groupby("scan").auroc_obs_unobs.mean().mean())
    ah = float(x[x.model == head_name].groupby("scan").auroc_obs_unobs.mean().mean())
    t = H.paired_permutation(x, "auroc_obs_unobs", head_name, b)
    return dict(share=fr, native=an, head=ah, diff=t, n_scenes=int(y.scan.nunique()),
                onsurf=float(y.groupby("scan").onsurf_valid.mean().mean()) if "onsurf_valid" in y else float("nan"))


def gate16(L, J):
    rd = load("P16-oneside/*/P12-realocc.csv")
    sp = load("P12-realocc/P12-realocc.csv")
    L += ["## Gate 16: one-sided captures (DTU benchmark, real occlusion)", "",
          "Pass: (a) unobserved share at least twice the spread protocol's at the same N; (b) the head beats native "
          "confidence at p < 0.05 in at least 6 of 8 cells.", "",
          "| backbone | views | scans | unobserved share (one-sided) | spread share | ratio | native AUROC | head AUROC | head - native (p) | holds |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    if rd is None:
        L += ["no rows", ""]; return
    rd["scan"] = rd["scan"].astype(str)
    if sp is not None:
        sp["scan"] = sp["scan"].astype(str)
    J["gate16"] = {}
    holds_n = cells = 0; ratios = []
    for b in BASES:
        for n in (3, 5):
            c = realocc_cells(rd, n, b, f"{b}_oiavisself")
            if c is None:
                continue
            s = sp[(sp.model == b) & (sp.n_views == n)].groupby("scan").frac_unobs.mean().mean() if sp is not None else float("nan")
            ratio = c["share"] / s if s == s and s > 0 else float("nan")
            ratios.append(ratio)
            holds = bool(c["diff"][0] > 0 and c["diff"][1] < 0.05)
            cells += 1; holds_n += holds
            J["gate16"][f"{b}:{n}"] = dict(share=round(c["share"], 5), spread_share=round(float(s), 5),
                                           ratio=round(ratio, 2), native=round(c["native"], 4), head=round(c["head"], 4),
                                           diff=[round(c["diff"][0], 4), round(c["diff"][1], 5)], holds=holds)
            L.append(f"| {b} | {n} | {c['n_scenes']} | {100 * c['share']:.2f}% | {100 * s:.2f}% | {ratio:.1f}x | "
                     f"{c['native']:.3f} | {c['head']:.3f} | {c['diff'][0]:+.3f} ({c['diff'][1]:.3g}) | {'yes' if holds else 'no'} |")
    a_ok = bool(ratios) and min(ratios) >= 2
    verdict = "passes" if a_ok and holds_n >= 6 else ("partial" if a_ok or holds_n >= 6 else "fails")
    J["gate16_total"] = dict(cells=cells, holds=holds_n, min_ratio=round(min(ratios), 2) if ratios else None, verdict=verdict)
    L += ["", f"**Gate 16 {verdict}:** (a) smallest share ratio {min(ratios) if ratios else float('nan'):.1f}x "
              f"({'met' if a_ok else 'not met'}); (b) holds in {holds_n} of {cells} cells.", ""]


def gate17(L, J):
    rd = load("P16-co3dadapt/*/P12-realocc-co3d.csv")
    hd = load("P14-realocc/holes/P14-holes-co3d.csv")
    L += ["## Gate 17: CO3Dv2 real occlusion after label-free adaptation (other fold)", "",
          "Each capture is scored by the head adapted, without labels, on the other fold. Holds where the cell qualifies "
          "(share at least 0.5%, hole rate at most 0.10, alignment on surface at least 0.5) and the adapted head beats native "
          "confidence at p < 0.05 over the 18 captures. Pass: holds in at least 2 qualifying cells.", "",
          "| backbone | views | captures | hole rate | aligned on surface | unobserved share | qualifies | native AUROC | adapted head AUROC | head - native (p) | holds |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    if rd is None:
        L += ["no rows", ""]; return
    rd["scan"] = rd["scan"].astype(str)
    rd["model"] = rd["model"].str.replace(r"_x co3d[AB]$", "", regex=True).str.replace(r"_xco3d[AB]$", "", regex=True)
    J["gate17"] = {}
    holds_n = 0
    for b in ("vggt", "pi3x"):
        for n in sorted(rd.n_views.unique()):
            c = realocc_cells(rd, n, b, f"{b}_oiavisself")
            if c is None:
                continue
            hr = float(hd[hd.n_views == n].groupby("scan").hole_rate.mean().mean()) if hd is not None else float("nan")
            q = bool(hr <= 0.10 and c["onsurf"] >= 0.5 and c["share"] >= 0.005)
            holds = bool(q and c["diff"][0] > 0 and c["diff"][1] < 0.05)
            holds_n += holds
            J["gate17"][f"{b}:{n}"] = dict(hole_rate=round(hr, 4), onsurf=round(c["onsurf"], 4), share=round(c["share"], 5),
                                           qualifies=q, native=round(c["native"], 4), head=round(c["head"], 4),
                                           diff=[round(c["diff"][0], 4), round(c["diff"][1], 5)], holds=holds,
                                           n_scenes=c["n_scenes"])
            L.append(f"| {b} | {n} | {c['n_scenes']} | {hr:.3f} | {c['onsurf']:.3f} | {100 * c['share']:.2f}% | "
                     f"{'yes' if q else 'no'} | {c['native']:.3f} | {c['head']:.3f} | {c['diff'][0]:+.3f} ({c['diff'][1]:.3g}) | "
                     f"{'yes' if holds else ('no' if q else 'n/a')} |")
    verdict = "passes" if holds_n >= 2 else "fails"
    J["gate17_total"] = dict(holds=holds_n, verdict=verdict)
    L += ["", f"**Gate 17 {verdict}:** holds in {holds_n} qualifying cells.", ""]


def main():
    L = ["# Ninth pass (generated by scripts/make_stats_p16.py; pre-registered in LOG.md, commit d4aebbd)", ""]
    J = {}
    gate15(L, J); gate16(L, J); gate17(L, J)
    open(os.path.join(R, "P16-stats.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(ROOT, "docs", "assets", "p16.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
