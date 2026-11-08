"""Numbers and charts for the project page and the video, computed from the result CSVs (nothing typed by hand).

  python scripts/make_assets.py [--theme light|dark|both] [--out docs/assets/charts]

Writes docs/assets/numbers.json (every number the page and the video quote, with bootstrap intervals) and the charts:
  leak_<theme>.png        the finding: share of unseen geometry that passes a 90%-kept confidence filter, per model, per dataset
  compare_<theme>.png     detectors on the filled patches (DTU) and on real single-view geometry (DTU): native, perturbation
                          variance (K=4, K=8), reprojection, MapAnything multi-view, per-image head, support heads
  vis_<theme>.png         AUROC of every detector family on the four datasets, one panel per backbone
  layers_<theme>.png      where support is decodable: linear probe per VGGT layer, four datasets
  rc_<theme>.png          risk-coverage curves (single-view rate of the kept pixels against coverage), four datasets
  adapt_<theme>.png       label-free adaptation on the target capture against the DTU-trained head
  onfill_<theme>.png      the limit: visibility detectors on the benchmark's filled patches
  method_<theme>.png      schematic of the label-free support head
  xmodel_<theme>.png      cross-model disagreement as a detector
  fuse_<theme>.png        fused point cloud at matched coverage
  refill_<theme>.png      fill sensitivity of the invented surface (P9-refill)
  prov_<theme>.png        provenance: which view saw it (P10-prov), head against the reprojection test, per dataset
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
NAME = {"vggt": "VGGT-1B", "pi3": "Pi3", "pi3x": "Pi3X", "mapanything": "MapAnything", "da3": "Depth Anything 3"}
DATASETS = [("dtu", "DTU", "P4-vis", None), ("co3d", "CO3Dv2", "P4-vis-co3d", "P6-adapt-co3d"),
            ("eth3d", "ETH3D", "P4-vis-eth3d", "P6-adapt-eth3d"), ("imc", "Phototourism", "P7-vis-imc", "P7-adapt-imc")]
THEMES = {
    "light": dict(bg="#F7F8F6", fg="#17201F", muted="#4C5A58", rule="#D9DEDC", c=["#17201F", "#9aa4b2", "#B86E00", "#18B8E0", "#1E8A7A", "#0B5F55", "#C8102E"]),
    "dark": dict(bg="#101418", fg="#f2f4f7", muted="#9aa4b2", rule="#2a3139", c=["#f2f4f7", "#9aa4b2", "#f4a261", "#4cc9f0", "#2ec4b6", "#90be6d", "#f94144"]),
}


# ----------------------------------------------------------------------------- numbers
def _csv(run, name=None):
    p = os.path.join(RES, run, f"{name or run}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _vis(run, adapt=None):
    d = _csv(run)
    if d is None:
        return None
    if adapt:
        a = _csv(adapt)
        if a is not None:
            a = a.copy(); a["model"] = a.model.str.replace(r"_x(co3d|eth3d|imc)[AB]$", "_x", regex=True)
            d = pd.concat([d, a], ignore_index=True)
    return d[~d.skipped.astype(bool)] if "skipped" in d else d


def ci(d, model, col="auroc", nb=3000):
    x = d[d.model == model]
    if x.empty:
        return None
    m, lo, hi, n = H.cluster_bootstrap_ci(x, col, n_boot=nb)
    return [round(float(m), 4), round(float(lo), 4), round(float(hi), 4), int(n)]


def numbers():
    N = {"leak": {}, "vis": {}, "layers": {}, "adapt": {}, "onfill": {}, "compare": {}, "ext": {}, "bands": {}, "rc": {}}
    # the finding: full configuration pooled over blur and noise
    for key, run, name in [("dtu", None, "table1_all_models.csv"), ("co3d", None, "table3_all_models.csv"),
                           ("eth3d", "P3-eth3d", None), ("dtuext", "P3-dtuext", None)]:
        d = pd.read_csv(os.path.join(RES, name)) if name else _csv(run)
        if d is None:
            continue
        d = d[d.fill.isin(["blur", "noise"])]
        N["leak"][key] = {m: {"auroc": ci(d, m, "auroc"), "cur": ci(d, m, "cur"), "cur_err": ci(d, m, "cur_err")}
                          for m in NAME if m in d.model.values}
    # the vis protocol
    for key, label, run, adapt in DATASETS:
        d = _vis(run, adapt)
        if d is None:
            continue
        N["vis"][key] = {}
        for base in ("vggt", "pi3x", "pi3", "da3"):
            for suf in ("", "reproj", "border", "dedge", "oia", "oiapatch", "oiavispatch", "oiavislin", "oiavis", "oiavisselfpatch",
                        "oiavisselflin", "oiavisself", "oiavisself_x", "oiavisselfpatch_x", "oiavis_s1", "oiavis_s2", "oiavisfuse", "oiavisselffuse"):
                m = base if suf == "" else f"{base}_{suf}"
                c = ci(d, m)
                if c:
                    N["vis"][key][m] = {"auroc": c, "cur": ci(d, m, "cur")}
        for m in ("da3", "mapanything"):
            c = ci(d, m)
            if c:
                N["vis"][key][m] = {"auroc": c, "cur": ci(d, m, "cur")}
        N["layers"][key] = {l: ci(d, f"vggt_oiavislin{l}") for l in ("4", "11", "17", "23") if f"vggt_oiavislin{l}" in d.model.values}
        if "vggt_oiavispatchlin" in d.model.values:
            N["layers"][key]["patch"] = ci(d, "vggt_oiavispatchlin")
            N["layers"][key]["all"] = ci(d, "vggt_oiavislin")
    # 20 views
    d = _vis("P4-vis-ext")
    if d is not None:
        N["ext"] = {m: ci(d, m) for m in d.model.unique()}
    # on the filled patches: the visibility detectors against native on the same rows
    nat = _csv("P3-native"); onf = _csv("P6-vis-on-fill")
    if nat is not None and onf is not None:
        d = pd.concat([nat, onf], ignore_index=True)
        d = d[(d.seed == 0) & (d.fill == "blur") & (d.radius_frac == 0.18)]
        N["onfill"]["dtu"] = {m: ci(d, m) for m in d.model.unique()}
    t3 = pd.read_csv(os.path.join(RES, "table3_all_models.csv")); onc = _csv("P6-vis-on-fill-co3d")
    if onc is not None:
        d = pd.concat([t3, onc], ignore_index=True)
        d = d[(d.seed == 0) & (d.fill.isin(["blur", "noise"])) & (d.radius_frac == 0.18)]
        N["onfill"]["co3d"] = {m: ci(d, m) for m in d.model.unique() if m in ("vggt", "pi3x") or "oiavis" in m or "border" in m or "dedge" in m}
    # the cheap alternatives and the fill-trained heads on the DTU Table-4 keys
    frames = [x for x in (_csv("P3-native"), _csv("P3-pv"), _csv("P3-pv-full"), _csv("P3-reproj"), _csv("P6-mapanything-mv"),
                          _csv("P3-oia-dtu"), _csv("P3-oia-abl"), pd.read_csv(os.path.join(RES, "table1_all_models.csv"))) if x is not None]
    d = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model", "scan", "seed", "radius_frac", "fill"])
    bench = [str(x) for x in H.DEFAULT_CFG["scans"]]
    d = d[(d.seed == 0) & (d.fill == "blur") & (d.radius_frac == 0.18) & d.scan.astype(str).isin(bench)]
    N["compare"]["fill"] = {m: {"auroc": ci(d, m), "cur": ci(d, m, "cur")} for m in d.model.unique()}
    # cross-model disagreement
    N["xmodel"] = {}
    for key, run in (("fill_dtu", "P8-xmodel"), ("fill_co3d", "P8-xmodel-co3d"), ("vis_dtu", "P8-xmodel-vis")):
        d = _csv(run)
        if d is not None:
            d = d[~d.skipped.astype(bool)] if "skipped" in d else d
            N["xmodel"][key] = {m: {"auroc": ci(d, m), "cur": ci(d, m, "cur")} for m in d.model.unique()}
    # fused point clouds at matched coverage
    N["fuse"] = {}
    for key in ("dtu", "eth3d"):
        p = os.path.join(RES, "P8-fuse", f"P8-fuse-{key}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            g = d.groupby(["model", "coverage"])[["accuracy", "completeness", "f1", "single_view_kept"]].mean().round(4)
            N["fuse"][key] = {m: {str(c): g.loc[(m, c)].to_dict() for c in sorted(d.coverage.unique())} for m in d.model.unique()}
    # fill sensitivity of the invented surface
    N["refill"] = {}
    for key, fn in (("dtu", "P9-refill.csv"), ("co3d", "P9-refill-co3d.csv")):
        p = os.path.join(RES, "P9-refill", fn)
        if os.path.exists(p):
            d = pd.read_csv(p)
            N["refill"][key] = {}
            for m in d.model.unique():
                g = d[d.model == m].groupby("scan")[["auroc_blur_noise", "auroc_gray_blur", "auroc_gray_noise", "spread_U", "spread_S", "auroc_native_blur"]].mean().mean()
                N["refill"][key][m] = {"auroc": ci(d, m), "cur": ci(d, m, "cur"), **{k: round(float(v), 4) for k, v in g.items()}}
        p = os.path.join(RES, "P9-refill", fn.replace("P9-refill", "P9-refill-single"))
        if os.path.exists(p):
            d = pd.read_csv(p)
            N.setdefault("refill_single", {})[key] = {}
            for m in d.model.unique():
                cols = [c for c in d.columns if c.startswith(("cur_", "spread_", "auroc_native"))]
                g = d[d.model == m].groupby("scan")[cols].mean().mean()
                N["refill_single"][key][m] = {"auroc_own": ci(d, m, "auroc_own"), "auroc_cross": ci(d, m, "auroc_cross") if "auroc_cross" in d else None,
                                              **{k: (None if pd.isna(v) else round(float(v), 4)) for k, v in g.items()}}
    # provenance heads (P10-prov): support score and attribution per dataset and detector
    N["prov"] = {}
    for key, *_ in DATASETS:
        p = os.path.join(RES, f"P10-prov-{key}", f"P10-prov-{key}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[~d.skipped.astype(bool)] if "skipped" in d else d
        N["prov"][key] = {}
        for m in d.model.unique():
            x = d[d.model == m].dropna(subset=["auroc"])
            e = {"auroc": ci(x, m), "cur": ci(x, m, "cur")}
            for c in ("prov_auroc", "prov_auroc_inframe", "prov_top1", "prov_count_mae"):
                if c in x and x[c].notna().any():
                    e[c] = ci(x.dropna(subset=[c]), m, c)
            N["prov"][key][m] = e
    # matched-patch split (P11-matched): the same patch under two labels; blur and noise pooled, as the leak tables
    N["matched"] = {}
    for key, run in (("dtu", "P11-matched"), ("co3d", "P11-matched-co3d")):
        p = os.path.join(RES, run, f"{run}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[~d.skipped.astype(bool)] if "skipped" in d else d
        d = d[d.fill_kind.isin(["blur", "noise"])]
        N["matched"][key] = {}
        for m in d.model.unique():
            x = d[d.model == m].dropna(subset=["auroc_m"])
            e = {c: ci(x, m, c) for c in ("auroc", "cur", "auroc_m", "cur_m", "auroc_patch")}
            for c in ("auroc_m_ev1lo", "auroc_m_ev1hi", "auroc_m_ev2", "auroc_m_ev3p"):
                if c in x and x[c].notna().any():
                    e[c] = ci(x.dropna(subset=[c]), m, c)
            N["matched"][key][m] = e
    # operating points
    for key, *_ in DATASETS:
        p = os.path.join(RES, "P6-dump", f"P6-dump-{key}_curves.csv")
        if os.path.exists(p):
            c = pd.read_csv(p).set_index("model")
            N["bands"][key] = {m: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in c.loc[m].items() if k != "n_cells"}
                               for m in c.index}
        p = os.path.join(RES, "P6-dump", f"P6-dump-{key}_rc.csv")
        if os.path.exists(p):
            r = pd.read_csv(p)
            N["rc"][key] = {m: {"coverage": g.coverage.round(3).tolist(), "risk": g.risk.round(4).tolist(), "absrel": g.absrel.round(4).tolist()}
                            for m, g in r.groupby("model")}
    return N


# ----------------------------------------------------------------------------- charts
def _style(theme):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = THEMES[theme]
    plt.rcParams.update({"figure.facecolor": T["bg"], "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"], "text.color": T["fg"],
                         "axes.edgecolor": T["muted"], "axes.labelcolor": T["fg"], "xtick.color": T["fg"], "ytick.color": T["fg"],
                         "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "legend.frameon": False})
    return plt, T


def _clean(ax, T):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=T["rule"], lw=0.8); ax.set_axisbelow(True)


def chart_leak(N, theme, out):
    plt, T = _style(theme)
    keys = [("dtu", "DTU, 18 scans"), ("dtuext", "DTU, 22 more scans"), ("co3d", "CO3Dv2 phone orbits"), ("eth3d", "ETH3D buildings")]
    keys = [k for k in keys if k[0] in N["leak"]]
    models = ["vggt", "pi3", "pi3x", "mapanything", "da3"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(models)); w = 0.8 / len(keys)
    for j, (k, lab) in enumerate(keys):
        vals = [N["leak"][k].get(m, {}).get("cur") for m in models]
        m_ = [v[0] if v else 0 for v in vals]; lo = [v[0] - v[1] if v else 0 for v in vals]; hi = [v[2] - v[0] if v else 0 for v in vals]
        ax.bar(x + (j - (len(keys) - 1) / 2) * w, m_, w * 0.9, color=T["c"][2 + j], label=lab, yerr=[lo, hi], ecolor=T["muted"], capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in models])
    ax.set_ylim(0, 1); ax.set_ylabel("share of unseen geometry that passes\na filter keeping 90% of observed geometry")
    ax.set_title("The leak: native confidence does not know what the model has seen")
    ax.legend(loc="upper left", ncol=2); _clean(ax, T)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_compare(N, theme, out):
    plt, T = _style(theme)
    fill = N["compare"]["fill"]; vis = N["vis"].get("dtu", {})
    rows = [("native confidence", "vggt", "vggt"), ("perturbation variance, K=4", "vggt_pv4", None), ("perturbation, K=8 + geometric", "vggt_pvf8", None),
            ("reprojection consistency", "vggt_reproj", "vggt_reproj"), ("per-image head (control)", "vggt_oiapatch", "vggt_oiavispatch"),
            ("linear probe, cross-view tokens", "vggt_oialin", "vggt_oiavislin"), ("support head, fill labels", "vggt_oia", "vggt_oia"),
            ("support head, GT visibility labels", None, "vggt_oiavis"), ("support head, label-free", None, "vggt_oiavisself")]
    onfill = N["onfill"].get("dtu", {})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(rows))
    for i, (ax, (src, title, keyi)) in enumerate(zip(axes, [(fill, "benchmark: a filled patch no view shows", 1), (vis, "real single-view geometry (clean views)", 2)])):
        for yi, (lab, mf, mv) in zip(y, rows):
            m = mf if keyi == 1 else mv
            v = src.get(m, {}).get("auroc") if m else (onfill.get(mv) if keyi == 1 else None)
            col = T["c"][4] if "label-free" in lab else (T["c"][3] if "support" in lab or "probe" in lab else T["c"][1])
            if v:
                ax.barh(yi, v[0], color=col, xerr=[[v[0] - v[1]], [v[2] - v[0]]], ecolor=T["muted"], capsize=3)
                ax.text(min(v[2] + 0.01, 0.985), yi, f"{v[0]:.2f}", va="center", fontsize=11)
            else:
                ax.text(0.41, yi, "not applicable" if keyi == 1 else "not run", va="center", fontsize=10, color=T["muted"])
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows] if i == 0 else [""] * len(rows)); ax.invert_yaxis()
        ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color=T["muted"], lw=0.8, ls=":")
        ax.set_xlabel("AUROC (VGGT-1B, 18 DTU scans)"); ax.set_title(title, fontsize=13)
        _clean(ax, T); ax.grid(axis="x", color=T["rule"], lw=0.8); ax.grid(axis="y", visible=False)
    fig.suptitle("Two questions, two answers: nothing finds a consistent invention; the cross-view tokens know what lacks corroboration", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_vis(N, theme, out):
    plt, T = _style(theme)
    dets = [("", "native confidence", T["c"][1]), ("border", "image border", T["c"][6]), ("oiavispatch", "per-image head", T["c"][2]),
            ("reproj", "reprojection", T["c"][3]), ("oiavis", "head, GT labels", T["c"][5]), ("oiavisself", "head, label-free", T["c"][4])]
    keys = [k for k in DATASETS if k[0] in N["vis"]]
    if any(f"{b}_oiavisselffuse" in N["vis"][k[0]] for b in ("vggt", "pi3x", "pi3", "da3") for k in keys):
        dets.append(("oiavisselffuse", "label-free head x native", T["c"][0]))
    bases = [b for b in ("vggt", "pi3x", "pi3", "da3") if any(f"{b}_oiavisself" in N["vis"][k[0]] for k in keys)]
    fig, axes = plt.subplots(1, len(bases), figsize=(4.7 * len(bases), 5.0), sharey=True)
    for ax, base in zip(np.atleast_1d(axes), bases):
        x = np.arange(len(keys)); w = 0.8 / len(dets)
        for j, (suf, lab, col) in enumerate(dets):
            m = base if suf == "" else f"{base}_{suf}"
            vals = [N["vis"][k[0]].get(m, {}).get("auroc") for k in keys]
            ax.bar(x + (j - (len(dets) - 1) / 2) * w, [v[0] if v else 0 for v in vals], w * 0.9, color=col, label=lab if base == "vggt" else None,
                   yerr=[[v[0] - v[1] if v else 0 for v in vals], [v[2] - v[0] if v else 0 for v in vals]], ecolor=T["muted"], capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([k[1] for k in keys]); ax.set_ylim(0.3, 1.0); ax.axhline(0.5, color=T["muted"], lw=0.8, ls=":")
        ax.set_title(NAME[base]); _clean(ax, T)
    axes[0].set_ylabel("AUROC: single-view against supported pixels")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Detecting geometry that only one view supports, on four datasets the heads never saw", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=160); plt.close(fig)


def chart_layers(N, theme, out):
    plt, T = _style(theme)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = ["patch", "4", "11", "17", "23", "all"]; xl = ["per-image\ntokens", "layer 4", "layer 11", "layer 17", "layer 23", "all four\nlayers"]
    for j, (k, lab, *_r) in enumerate([d for d in DATASETS if d[0] in N["layers"]]):
        L = N["layers"][k]
        ys = [L.get(x) for x in xs]
        ax.plot(range(len(xs)), [y[0] if y else np.nan for y in ys], marker="o", color=T["c"][2 + j], label=lab, lw=2)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xl); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=T["muted"], lw=0.8, ls=":")
    ax.set_ylabel("AUROC of a linear probe"); ax.set_title("Where support becomes readable: one linear probe per VGGT layer")
    ax.legend(loc="lower right"); _clean(ax, T)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_rc(N, theme, out):
    plt, T = _style(theme)
    keys = [d for d in DATASETS if d[0] in N["rc"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, lab, *_r) in zip(axes, keys):
        R = N["rc"][k]
        for m, name, col, ls in [("vggt", "native confidence", T["c"][1], "-"), ("vggt_border", "image border", T["c"][2], ":"),
                                 ("vggt_reproj", "reprojection", T["c"][3], "-"), ("vggt_oiavisself", "head, label-free", T["c"][4], "-")]:
            if m in R:
                ax.plot(R[m]["coverage"], R[m]["risk"], color=col, ls=ls, lw=2, label=name)
        ax.set_title(lab, fontsize=12); ax.set_xlabel("coverage (fraction of pixels kept)"); ax.set_xlim(0, 1); ax.set_ylim(0, None); _clean(ax, T)
    axes[0].set_ylabel("single-view rate of the kept pixels"); axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Keep the pixels the detector trusts most: how much unsupported geometry survives (VGGT-1B)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_adapt(N, theme, out):
    plt, T = _style(theme)
    keys = [d for d in DATASETS if d[0] != "dtu" and d[0] in N["vis"] and any("_x" in m for m in N["vis"][d[0]])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(keys)); w = 0.2
    for j, (base, suf, lab, col) in enumerate([("vggt", "oiavisself", "VGGT, trained on DTU", T["c"][3]), ("vggt", "oiavisself_x", "VGGT, trained on the target capture", T["c"][4]),
                                               ("pi3x", "oiavisself", "Pi3X, trained on DTU", T["c"][2]), ("pi3x", "oiavisself_x", "Pi3X, trained on the target capture", T["c"][5])]):
        vals = [N["vis"][k[0]].get(f"{base}_{suf}", {}).get("auroc") for k in keys]
        ax.bar(x + (j - 1.5) * w, [v[0] if v else 0 for v in vals], w * 0.9, color=col, label=lab,
               yerr=[[v[0] - v[1] if v else 0 for v in vals], [v[2] - v[0] if v else 0 for v in vals]], ecolor=T["muted"], capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([k[1] for k in keys]); ax.set_ylim(0.6, 1.0); ax.set_ylabel("AUROC")
    ax.set_title("Label-free adaptation: training on the target capture itself, no ground truth"); ax.legend(fontsize=10, loc="lower right"); _clean(ax, T)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_onfill(N, theme, out):
    plt, T = _style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (k, title) in zip(axes, [("dtu", "DTU, blur fill, radius 0.18"), ("co3d", "CO3Dv2, blur and noise fills")]):
        src = N["onfill"].get(k, {})
        rows = [("native confidence", "vggt"), ("image border", "vggt_border"), ("depth edges", "vggt_dedge"), ("per-image head", "vggt_oiavispatch"),
                ("head, GT labels", "vggt_oiavis"), ("head, label-free", "vggt_oiavisself")]
        labs = [r[0] for r in rows if r[1] in src]; vals = [src[r[1]] for r in rows if r[1] in src]
        y = np.arange(len(labs))
        ax.barh(y, [v[0] for v in vals], color=[T["c"][1]] + [T["c"][2]] * 3 + [T["c"][4]] * 2, xerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], ecolor=T["muted"], capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_xlim(0.3, 1.0); ax.axvline(0.5, color=T["muted"], lw=0.8, ls=":")
        ax.set_title(title, fontsize=12); ax.set_xlabel("AUROC on the filled patch (VGGT-1B)"); _clean(ax, T); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=T["rule"], lw=0.8)
    fig.suptitle("The limit: a detector of single-view support does not fire on a patch that every view shows", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_xmodel(N, theme, out):
    plt, T = _style(theme)
    X = N.get("xmodel", {})
    if not X:
        return
    pairs = sorted({m for v in X.values() for m in v})
    lab = lambda m: m.replace("_x", " vs ").replace("vggt", "VGGT").replace("pi3x", "Pi3X").replace("pi3", "Pi3").replace("da3", "DA3").replace("mapanything", "MapAny")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(pairs)); w = 0.26
    for j, (k, name, col) in enumerate([("fill_dtu", "filled patch, DTU", T["c"][2]), ("fill_co3d", "filled patch, CO3Dv2", T["c"][3]), ("vis_dtu", "real single-view, DTU", T["c"][4])]):
        vals = [X.get(k, {}).get(m, {}).get("auroc") for m in pairs]
        ax.bar(x + (j - 1) * w, [v[0] if v else 0 for v in vals], w * 0.92, color=col, label=name,
               yerr=[[v[0] - v[1] if v else 0 for v in vals], [v[2] - v[0] if v else 0 for v in vals]], ecolor=T["muted"], capsize=2)
    ax.axhline(0.5, color=T["muted"], lw=0.8, ls=":"); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x); ax.set_xticklabels([lab(m) for m in pairs], rotation=15); ax.set_ylabel("AUROC of cross-model disagreement")
    ax.set_title("Do two models invent the same surface? Yes: disagreement between models is at chance"); ax.legend(loc="upper left", fontsize=10); _clean(ax, T)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_fuse(N, theme, out):
    plt, T = _style(theme)
    Fz = N.get("fuse", {})
    if not Fz:
        return
    keys = [k for k in ("dtu", "eth3d") if k in Fz]
    fig, axes = plt.subplots(1, 2 * len(keys), figsize=(4.4 * 2 * len(keys), 3.9))
    axes = np.atleast_1d(axes)
    series = [("vggt", "native confidence", T["c"][1], "-"), ("vggt_reproj", "reprojection", T["c"][3], "-"), ("vggt_oiavispatch", "per-image head", T["c"][2], "-"),
              ("vggt_oiavisself", "head, label-free", T["c"][4], "-"), ("vggt_oiavisselffuse", "head x native confidence", T["c"][0], "-"),
              ("vggt_random", "random dropping", T["c"][1], ":"), ("vggt_oracle", "oracle (drop single-view first)", T["c"][5], "--")]
    i = 0
    for k in keys:
        for metric, ylabel in (("accuracy", "accuracy of the fused points"), ("completeness", "completeness")):
            ax = axes[i]; i += 1
            for m, name, col, ls in series:
                if m in Fz[k]:
                    cov = sorted(float(c) for c in Fz[k][m]); ys = [Fz[k][m][str(c) if str(c) in Fz[k][m] else c][metric] for c in cov]
                    ax.plot(cov, ys, marker="o", color=col, ls=ls, lw=2, label=name)
            ax.set_title(f"{'DTU' if k == 'dtu' else 'ETH3D'}: {ylabel}", fontsize=11); ax.set_xlabel("coverage (fraction of predicted points kept)"); _clean(ax, T)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Fused point cloud (VGGT-1B, its own cameras, similarity-aligned to the laser scan): keeping what the detector trusts", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out, dpi=160); plt.close(fig)


def chart_refill(N, theme, out):
    """Fill sensitivity: AUROC of the depth spread over three fills of the same hidden sphere, against native confidence."""
    plt, T = _style(theme)
    R = N.get("refill", {})
    if not R:
        return
    keys = [k for k in ("dtu", "co3d") if k in R]
    models = [m for m in ("vggt", "pi3x", "pi3", "da3", "mapanything") if any(m in R[k] for k in keys)]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        x = np.arange(len(models)); w = 0.38
        nat = [R[k].get(m, {}).get("auroc_native_blur") for m in models]
        sp = [R[k].get(m, {}).get("auroc") for m in models]
        ax.bar(x - w / 2, [v or 0 for v in nat], w * 0.92, color=T["c"][1], label="native confidence (blur fill)")
        ax.bar(x + w / 2, [v[0] if v else 0 for v in sp], w * 0.92, color=T["c"][2], label="depth spread over gray / blur / noise fills",
               yerr=[[v[0] - v[1] if v else 0 for v in sp], [v[2] - v[0] if v else 0 for v in sp]], ecolor=T["muted"], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in models]); ax.set_ylim(0.3, 1.0); ax.axhline(0.5, color=T["muted"], lw=0.8, ls=":")
        ax.set_title({"dtu": "DTU, 18 scans", "co3d": "CO3Dv2, 18 scenes"}[k], fontsize=12); _clean(ax, T)
    axes[0].set_ylabel("AUROC on the filled patch"); axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Does the invented surface depend on the pixels it was filled with? The same sphere filled three ways (run P9-refill)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_matched(N, theme, out):
    """The same patch under two labels: AUROC of native confidence against clean supported pixels (the benchmark's
    original number) and against filled-but-observed pixels (the matched number), per model and dataset; with the
    matched heads where they exist."""
    plt, T = _style(theme)
    M = N.get("matched", {})
    keys = [k for k in ("dtu", "co3d") if k in M]
    if not keys:
        return
    models = [b for b in ("vggt", "pi3", "pi3x", "mapanything", "da3") if any(b in M[k] for k in keys)]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        x = np.arange(len(models)); w = 0.2
        series = [("auroc", "native confidence vs clean pixels (the original benchmark)", T["muted"]),
                  ("auroc_m", "native confidence vs filled-but-observed pixels (matched)", T["c"][1])]
        for i, (col, lab, colr) in enumerate(series):
            vals = [M[k].get(b, {}).get(col) for b in models]
            ax.bar(x + (i - 1.5) * w, [v[0] if v else 0 for v in vals], w * 0.92, color=colr, label=lab,
                   yerr=[[v[0] - v[1] if v else 0 for v in vals], [v[2] - v[0] if v else 0 for v in vals]], ecolor=T["muted"], capsize=2)
        for i, (suf, lab, colr) in enumerate((("_oiamatchselfpatch", "matched per-image control", T["c"][3]), ("_oiamatchself", "matched head, label-free", T["c"][2]))):
            vals = [M[k].get(f"{b}{suf}", {}).get("auroc_m") for b in models]
            if any(vals):
                ax.bar(x + (i + 0.5) * w, [v[0] if v else 0 for v in vals], w * 0.92, color=colr, label=lab,
                       yerr=[[v[0] - v[1] if v else 0 for v in vals], [v[2] - v[0] if v else 0 for v in vals]], ecolor=T["muted"], capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([NAME[b] for b in models], fontsize=9); ax.set_ylim(0.2, 1.0)
        ax.axhline(0.5, color=T["muted"], lw=0.8, ls=":")
        ax.set_title(dict(DATASETS_LABEL)[k], fontsize=12); _clean(ax, T)
    axes[0].set_ylabel("AUROC"); axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("The same patch under two labels: native confidence reads the patch, not the missing support (run P11-matched)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_prov(N, theme, out):
    """Provenance: attribution AUROC (does the per-view score rank the views that see a point above those that do not)
    of the label-free provenance head against the reprojection test with the model's own cameras, per backbone and dataset."""
    plt, T = _style(theme)
    P = N.get("prov", {})
    keys = [k for k, *_ in DATASETS if k in P]
    if not keys:
        return
    bases = [b for b in ("vggt", "pi3x", "pi3", "da3") if any(f"{b}_oiaprovself" in P[k] for k in keys)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        x = np.arange(len(bases)); w = 0.27
        for i, (suf, lab, col) in enumerate((("reproj", "reprojection, own cameras", T["c"][1]), ("oiaprov", "provenance head, GT labels", T["c"][4]),
                                             ("oiaprovself", "provenance head, label-free", T["c"][2]))):
            vals = [P[k].get(f"{b}_{suf}", {}).get("prov_auroc") for b in bases]
            ax.bar(x + (i - 1) * w, [v[0] if v else 0 for v in vals], w * 0.92, color=col, label=lab,
                   yerr=[[v[0] - v[1] if v else 0 for v in vals], [v[2] - v[0] if v else 0 for v in vals]], ecolor=T["muted"], capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([NAME[b] for b in bases], fontsize=9); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=T["muted"], lw=0.8, ls=":")
        ax.set_title(dict(DATASETS_LABEL)[k], fontsize=12); _clean(ax, T)
    axes[0].set_ylabel("attribution AUROC (which views see the point)"); axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Which view saw it: per-view attribution of the provenance head against the model's own reprojection test (run P10-prov)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


DATASETS_LABEL = [(k, f"{lab}" ) for k, lab, *_ in [("dtu", "DTU, 18 scans"), ("co3d", "CO3Dv2, 18 scenes"), ("eth3d", "ETH3D, 9 scenes"), ("imc", "Phototourism, 9 landmarks")]]


def chart_method(theme, out):
    plt, T = _style(theme)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig = plt.figure(figsize=(12, 5.8)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 5.8); ax.axis("off")

    def box(x, y, w, h, title, sub=None, col=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col or T["bg"], ec=T["fg"], lw=1.4))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color=T["fg"])
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center", fontsize=10.5, color=T["muted"], linespacing=1.35)

    def arrow(x0, y0, x1, y1, txt=None, col=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=col or T["fg"], lw=1.4))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.16, txt, ha="center", fontsize=10, color=T["muted"])

    box(0.3, 3.3, 2.2, 1.6, "N unposed views", "3 to 20 photographs\nno cameras, no labels")
    box(3.2, 3.3, 2.6, 1.6, "frozen 3D backbone", "VGGT, Pi3X or Pi3\nnot modified, not re-run")
    box(6.5, 4.1, 2.3, 0.85, "depth, cameras, confidence", None)
    box(6.5, 2.85, 2.3, 0.9, "cross-view tokens", "2048-d at 14 x 14 patches")
    box(9.5, 2.85, 2.2, 0.9, "support head", "3.3 M params, one logit\nper patch")
    box(9.2, 1.15, 2.6, 0.95, "observed / abstain / imagined", "1 - P(imagined), bands 0.7 / 0.3")
    box(3.2, 0.3, 5.6, 1.5, "label from the backbone's own output (training only)",
        "each predicted point is projected into the other views with the predicted cameras:\nsupported by two views = negative, supported by none = positive.\nThe reprojection test is the teacher; no ground truth anywhere.")
    arrow(2.5, 4.1, 3.2, 4.1); arrow(5.8, 4.3, 6.5, 4.5); arrow(5.8, 3.9, 6.5, 3.3); arrow(8.8, 3.3, 9.5, 3.3)
    arrow(10.6, 2.85, 10.6, 2.1); arrow(7.6, 4.1, 7.6, 1.8, col=T["c"][2]); arrow(8.8, 1.0, 9.5, 1.4, "loss", col=T["c"][2])
    ax.text(0.3, 1.0, "Inference: one backbone forward\nplus one head forward.\n\nTraining: 1500 steps on the\nbackbone's own predictions,\nten minutes on one H200.",
            fontsize=10.5, color=T["muted"], va="center")
    fig.savefig(out, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="both")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets", "charts"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    N = numbers()
    with open(os.path.join(ROOT, "docs", "assets", "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(N, f, indent=1)
    print("numbers.json written:", {k: len(v) for k, v in N.items()})
    for theme in (["light", "dark"] if a.theme == "both" else [a.theme]):
        chart_leak(N, theme, os.path.join(a.out, f"leak_{theme}.png"))
        chart_compare(N, theme, os.path.join(a.out, f"compare_{theme}.png"))
        chart_vis(N, theme, os.path.join(a.out, f"vis_{theme}.png"))
        chart_layers(N, theme, os.path.join(a.out, f"layers_{theme}.png"))
        chart_rc(N, theme, os.path.join(a.out, f"rc_{theme}.png"))
        chart_adapt(N, theme, os.path.join(a.out, f"adapt_{theme}.png"))
        chart_onfill(N, theme, os.path.join(a.out, f"onfill_{theme}.png"))
        chart_method(theme, os.path.join(a.out, f"method_{theme}.png"))
        chart_xmodel(N, theme, os.path.join(a.out, f"xmodel_{theme}.png"))
        chart_fuse(N, theme, os.path.join(a.out, f"fuse_{theme}.png"))
        chart_refill(N, theme, os.path.join(a.out, f"refill_{theme}.png"))
        chart_prov(N, theme, os.path.join(a.out, f"prov_{theme}.png"))
        chart_matched(N, theme, os.path.join(a.out, f"matched_{theme}.png"))
        print("charts written for", theme)


if __name__ == "__main__":
    main()
