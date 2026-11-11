"""Supplementary tables of the real-data experiments, generated from the run CSVs:
  tab_realocc.tex   real occlusion against dense references with hole-robust labels, every dataset, backbone and
                    view count (docs/assets/p17.json)
  tab_capacity.tex  linear probe, conv head, cross-view transformer head and per-image control on real occlusion
  tab_down3.tex     fused reconstruction at 3 views, native confidence, reprojection and the head at coverage 0.8 / 0.9

  python scripts/make_tab_realdata.py      writes paper/tex/tab_{realocc,capacity,down3}.tex"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tex")
NAMES = {"vggt": "VGGT-1B", "pi3x": "Pi3X", "pi3": "Pi3", "da3": "DA3"}
BASES = ["vggt", "pi3x", "pi3", "da3"]


def p_str(p):
    return "$<$0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def realocc():
    J = json.load(open(os.path.join(ROOT, "docs", "assets", "p17.json"), encoding="utf-8"))["cells"]
    L = [r"\begin{tabular}{llrcccccc}", r"\toprule",
         r"dataset & backbone & views & ambiguous & unobs.\ share & native & head & head $-$ native ($p$) & claim \\", r"\midrule"]
    for ds in ("DTU benchmark", "DTU test", "ETH3D", "CO3Dv2"):
        for b in BASES:
            for n in (3, 5, 10):
                c = J.get(f"{ds}:{n}:{b}")
                if not c:
                    continue
                claim = "yes" if c["holds"] else ("no" if c["qualifies"] else "--")
                L.append(f"{ds} & {NAMES[b]} & {n} & {100 * c['amb']:.2f}\\% & {100 * c['share']:.2f}\\% & {c['native']:.3f} & "
                         f"{c['head']:.3f} & ${c['diff']:+.3f}$ ({p_str(c['p'])}) & {claim} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(OUT, "tab_realocc.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def capacity():
    params = {"vggt": {"lin": 2.11, "conv": 3.29, "attn": 3.71}, "pi3x": {"lin": 0.53, "conv": 1.71, "attn": 2.12}}
    L = [r"\begin{tabular}{llrcccccc}", r"\toprule",
         r"backbone & dataset & views & native & per-image control & linear probe & conv head (ours) & transformer head & transformer $-$ conv ($p$) \\",
         r"\midrule"]
    for b in ("vggt", "pi3x"):
        for ds, f in (("DTU", f"P15-capacity/dtu_{b}/P12-realocc.csv"), ("ETH3D", f"P15-capacity/eth_{b}/P12-realocc-eth3d.csv")):
            d = pd.read_csv(os.path.join(R, f)); d["scan"] = d.scan.astype(str)
            for n in (3, 5, 10):
                x = d[d.n_views == n]
                m = lambda k: x[x.model == k].groupby("scan").auroc_obs_unobs.mean().mean()  # noqa: E731
                t = H.paired_permutation(x, "auroc_obs_unobs", f"{b}_oiavisselfattn", f"{b}_oiavisself")
                L.append(f"{NAMES[b]} & {ds} & {n} & {m(b):.3f} & {m(b + '_oiavisselfpatch'):.3f} & {m(b + '_oiavisselflin'):.3f} & "
                         f"{m(b + '_oiavisself'):.3f} & {m(b + '_oiavisselfattn'):.3f} & ${t[0]:+.3f}$ ({p_str(t[1])}) \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}"]
    pr = "; ".join(f"{NAMES[b]}: linear {v['lin']:.2f}\\,M, conv {v['conv']:.2f}\\,M, transformer {v['attn']:.2f}\\,M" for b, v in params.items())
    open(os.path.join(OUT, "tab_capacity.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    open(os.path.join(OUT, "tab_capacity_params.tex"), "w", encoding="utf-8", newline="\n").write(pr + "\n")


def down3():
    L = [r"\begin{tabular}{llrccccccc}", r"\toprule",
         r"dataset & backbone & coverage & acc.\ native & acc.\ reproj. & acc.\ head & head $-$ native ($p$) & F1 head $-$ native & single-view native & single-view head \\",
         r"\midrule"]
    for tag, ds in (("dtu", "DTU benchmark"), ("dtuext", "DTU test"), ("eth3d", "ETH3D")):
        for b in BASES:
            d = pd.read_csv(os.path.join(R, "P15-down3", b, f"P8-fuse-{tag}.csv")); d["scan"] = d.scan.astype(str)
            for cov in (0.8, 0.9):
                x = d[d.coverage == cov]
                a = x.groupby("model").accuracy.mean()
                t = H.paired_permutation(x, "accuracy", f"{b}_oiavisself", b)
                f = H.paired_permutation(x, "f1", f"{b}_oiavisself", b)
                sv = x.groupby("model").single_view_kept.mean()
                L.append(f"{ds} & {NAMES[b]} & {cov:.1f} & {a[b]:.3f} & {a[b + '_reproj']:.3f} & {a[b + '_oiavisself']:.3f} & "
                         f"${t[0]:+.3f}$ ({p_str(t[1])}) & ${f[0]:+.3f}$ & {sv[b]:.3f} & {sv[b + '_oiavisself']:.3f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(OUT, "tab_down3.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    realocc(); capacity(); down3()
    print("-> tab_realocc.tex, tab_capacity.tex, tab_down3.tex")
