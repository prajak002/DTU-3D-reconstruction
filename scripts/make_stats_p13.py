"""Statistics of the sixth pass (pre-registered Gates 9, 10 and 11 in EXPERIMENTS.md), generated from the run CSVs.

  Gate 9   results/P11-matched/P11-matched.csv        VGGT matched heads fine-tuned with one-view evidence (ev)
  Gate 10  results/P12-realocc/P12-realocc*.csv        real occlusion: four backbones on DTU, VGGT and Pi3X on ETH3D
  Gate 11  results/P13-downstream/P8-fuse-*.csv        fused reconstruction: native, head, rank-mean combination

  python scripts/make_stats_p13.py    writes results/P13-stats.md and docs/assets/p13.json"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASES = ["vggt", "pi3x", "pi3", "da3"]


def ci(df, col):
    m, lo, hi, n = H.cluster_bootstrap_ci(df, col)
    return [round(m, 4), round(lo, 4), round(hi, 4), n]


def gate9(L, J):
    p = os.path.join(ROOT, "results", "P11-matched", "P11-matched.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[~d.skipped.astype(bool)] if "skipped" in d else d
    d = d.drop_duplicates(subset=["model", "scan", "seed", "radius_frac", "fill"])
    P = d[d.fill_kind.isin(("blur", "noise"))]
    L += ["## Gate 9: VGGT matched heads fine-tuned with one-view evidence (DTU, blur and noise pooled)", "",
          "| label source | control AUROC_m (<= 0.60) | head AUROC_m | 1500-step AUROC_m | change (< -0.02 fails) | "
          "one view, low parallax (>= 0.75) | 1500-step one view, low | head - native (p) | head - control (p) | pass |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    J["gate9"] = {}
    for lab, h, c, old in (("GT placement", "vggt_oiamatchev", "vggt_oiamatchevpatch", "vggt_oiamatch"),
                           ("label-free", "vggt_oiamatchselfev", "vggt_oiamatchselfevpatch", "vggt_oiamatchself")):
        if h not in P.model.values:
            continue
        Hd, Cd, Od = P[P.model == h], P[P.model == c], P[P.model == old]
        hm, om = ci(Hd, "auroc_m"), ci(Od, "auroc_m")
        cm = ci(Cd, "auroc_m") if len(Cd) else None
        e1, e1o = ci(Hd, "auroc_m_ev1lo"), ci(Od, "auroc_m_ev1lo")
        hn = H.paired_permutation(P, "auroc_m", h, "vggt")
        hc = H.paired_permutation(P, "auroc_m", h, c) if cm else (float("nan"), float("nan"))
        dlt = hm[0] - om[0]
        ok = dict(p=bool(cm and cm[0] <= 0.60), a=bool(e1[0] >= 0.75), b=bool(dlt >= -0.02),
                  c=bool(hn[0] >= 0.10 and hn[1] < 0.05 and hc[0] >= 0.10 and hc[1] < 0.05))
        J["gate9"][lab] = dict(control=cm, head=hm, old=om, ev1lo=e1, ev1lo_old=e1o,
                               vs_native=[round(hn[0], 4), round(hn[1], 5)], vs_control=[round(hc[0], 4), round(hc[1], 5)],
                               clauses=ok, passed=all(ok.values()))
        fails = ", ".join(k for k, v in ok.items() if not v)
        L.append(f"| {lab} | {cm[0] if cm else '--'} | {hm[0]:.3f} [{hm[1]:.3f}, {hm[2]:.3f}] | {om[0]:.3f} | {dlt:+.3f} | "
                 f"{e1[0]:.3f} [{e1[1]:.3f}, {e1[2]:.3f}] | {e1o[0]:.3f} | {hn[0]:+.3f} ({hn[1]:.4g}) | {hc[0]:+.3f} ({hc[1]:.4g}) | "
                 f"{'yes' if all(ok.values()) else 'no: ' + fails} |")
    L.append("")


def gate10(L, J):
    J["gate10"] = {}
    for tag, name in (("P12-realocc.csv", "DTU"), ("P12-realocc-eth3d.csv", "ETH3D")):
        p = os.path.join(ROOT, "results", "P12-realocc", tag)
        if not os.path.exists(p):
            continue
        rd = pd.read_csv(p)
        L += [f"## Gate 10: real occlusion, {name}", "",
              "| views | backbone | scenes | unobserved share | qualifies (>= 0.5%) | native AUROC | head AUROC | head - native (p) | claim holds |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n in sorted(rd.n_views.unique()):
            x = rd[rd.n_views == n]
            for b in BASES:
                if b not in x.model.values or f"{b}_oiavisself" not in x.model.values:
                    continue
                y = x[x.model == b]
                fr = float(y.groupby("scan").frac_unobs.mean().mean())
                an = float(y.groupby("scan").auroc_obs_unobs.mean().mean())
                ah = float(x[x.model == f"{b}_oiavisself"].groupby("scan").auroc_obs_unobs.mean().mean())
                t = H.paired_permutation(x, "auroc_obs_unobs", f"{b}_oiavisself", b)
                q = fr >= 0.005
                holds = bool(q and t[0] > 0 and t[1] < 0.05)
                J["gate10"][f"{name}:{n}:{b}"] = dict(share=round(fr, 5), qualifies=bool(q), native=round(an, 4),
                                                       head=round(ah, 4), diff=[round(t[0], 4), round(t[1], 5)],
                                                       holds=holds, n_scenes=int(y.scan.nunique()))
                L.append(f"| {n} | {b} | {y.scan.nunique()} | {100 * fr:.2f}% | {'yes' if q else 'no'} | {an:.3f} | {ah:.3f} | "
                         f"{t[0]:+.3f} ({t[1]:.4g}) | {'yes' if holds else ('no' if q else 'n/a')} |")
        L.append("")
    cells = [v for v in J["gate10"].values() if v["qualifies"]]
    L.append(f"Qualifying cells: {len(cells)}; claim holds in {sum(v['holds'] for v in cells)}.")
    L.append("")


def gate11(L, J):
    J["gate11"] = {}
    for tag, name in (("dtu", "DTU"), ("eth3d", "ETH3D")):
        p = os.path.join(ROOT, "results", "P13-downstream", f"P8-fuse-{tag}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["scan"] = d.scan.astype(str)
        L += [f"## Gate 11: fused reconstruction, {name} (F1 at tau = 1% of median depth; views 5 and 10, seeds 0 and 1)", "",
              "| backbone | coverage | scenes | native F1 | head F1 | combined F1 | combined - native (p) | all points F1 |",
              "|---|---|---|---|---|---|---|---|"]
        for b in BASES:
            h = f"{b}_oiavisself"
            if f"{h}_comb" not in d.model.values:
                continue
            full = float(d[np.isclose(d.coverage, 1.0) & (d.model == h)].groupby("scan").f1.mean().mean())
            for cov in (0.6, 0.7, 0.8, 0.9):
                x = d[np.isclose(d.coverage, cov)]
                f = lambda m: float(x[x.model == m].groupby("scan").f1.mean().mean())
                t = H.paired_permutation(x, "f1", f"{h}_comb", f"{h}_native")
                ns = int(x[x.model == f"{h}_comb"].scan.nunique())
                J["gate11"][f"{name}:{b}:{cov}"] = dict(native=round(f(f"{h}_native"), 4), head=round(f(h), 4),
                                                         comb=round(f(f"{h}_comb"), 4), diff=[round(t[0], 4), round(t[1], 5)],
                                                         all_points=round(full, 4), n_scenes=ns)
                L.append(f"| {b} | {cov} | {ns} | {f(f'{h}_native'):.3f} | {f(h):.3f} | {f(f'{h}_comb'):.3f} | "
                         f"{t[0]:+.4f} ({t[1]:.4g}) | {full:.3f} |")
        L.append("")
    wins = {}
    for k, v in J["gate11"].items():
        name, b, cov = k.split(":")
        if cov in ("0.7", "0.8"):
            wins.setdefault((name, b), []).append(v["diff"][0] > 0 and v["diff"][1] < 0.05)
    per = {k: all(v) for k, v in wins.items()}
    ok = {name: sum(per.get((name, b), False) for b in BASES) for name in ("DTU", "ETH3D")}
    J["gate11_pass"] = dict(backbones_passing=ok, passed=bool(ok["DTU"] >= 2 and ok["ETH3D"] >= 2))
    L.append(f"Backbones where the combined detector beats native at 0.7 and 0.8 (p < 0.05): DTU {ok['DTU']}, "
             f"ETH3D {ok['ETH3D']}; Gate 11 {'passes' if J['gate11_pass']['passed'] else 'fails'}.")
    L.append("")


def main():
    L, J = ["# Sixth pass (generated by scripts/make_stats_p13.py)", ""], {}
    gate9(L, J)
    gate10(L, J)
    gate11(L, J)
    open(os.path.join(ROOT, "results", "P13-stats.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(ROOT, "docs", "assets", "p13.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
