"""Paper figures, all from logged results. Run from the project root:
  .venv/bin/python scripts/make_figures.py
Writes paper/figures/figN_*.pdf and .png. No model is run; model panels come from saved composites."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import hallu3d as H
import co3d

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
SCR = "/private/tmp/claude-502/-Users-prajaksen-Desktop-cvpr/01a3d1e9-f90c-4716-bf61-f00f183c7737/scratchpad"

# palette (validated with the dataviz skill validator, light surface)
C = {"vggt": "#2a78d6", "pi3": "#1baf7a", "pi3x": "#6250d6", "mapanything": "#eb6834", "da3": "#e87ba4"}
NAME = {"vggt": "VGGT-1B", "pi3": "Pi3", "pi3x": "Pi3X", "mapanything": "MapAnything", "da3": "Depth Anything 3"}
ORDER = ["vggt", "pi3", "pi3x", "mapanything", "da3"]
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
COLW, PAGEW = 3.3, 6.9

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name)


def despine(ax, left=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def ci(df, col):
    m, lo, hi, n = H.cluster_bootstrap_ci(df, col)
    return m, lo, hi


# ----------------------------------------------------------------------------- data
t1 = pd.read_csv("results/table1_all_models.csv")
t3 = pd.read_csv("results/table3_all_models.csv")
dtu_root = H.find_dtu_root([os.path.join(SCR, "dtu_probe")])
co3d_root = os.path.join(SCR, "co3d_prep")


def dtu_example(scan=3, seed=0, rf=0.18):
    d = H.load_scan(dtu_root, scan, H.select_views(10), 3)
    m = H.build_masks(d, seed, rf, H.DEFAULT_CFG)
    return d, m


def co3d_example(scene="toytrain_399_51323_100753", seed=0, rf=0.18):
    d = co3d.load_co3d_scene(co3d_root, scene, H.select_views(10, total=co3d.n_frames(co3d_root, scene)))
    m = H.build_masks(d, seed, rf, H.DEFAULT_CFG)
    return d, m


def crop_panels(path, n=4):
    """Saved composites: 1x4 equal-width panels (matplotlib tight layout at 16x3.4 in, dpi 110).
    Geometry measured on the file: first panel starts at x=17, last (conf) spans x 1324..1743, y 69..304."""
    im = np.asarray(Image.open(path).convert("RGB"))
    x0, xl, w, y0, y1 = 17, 1324, 420, 69, 305
    pitch = (xl - x0) / 3
    return [im[y0:y1, int(x0 + k * pitch):int(x0 + k * pitch) + w] for k in range(n)]


def overlay(img, mask, color=(1.0, 0.35, 0.1), alpha=0.45):
    out = img.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return out


# ----------------------------------------------------------------------------- fig 1 teaser
def fig1():
    d, m = dtu_example()
    cd, cm = co3d_example()
    panels = crop_panels("results/P2-co3d/ex_P2-co3d/vggt_scantoytrain_399_51323_100753_r0.18.png")
    fig, ax = plt.subplots(2, 4, figsize=(PAGEW, 3.1))
    i = 0
    # row 1: DTU
    ax[0, 0].imshow(d["imgs"][i]); ax[0, 0].set_title("DTU scan 3, view 1 (clean)")
    inp = H.apply_fill(d["imgs"], m["hide"], "inpaint", 0)
    ax[0, 1].imshow(inp[i]); ax[0, 1].set_title("input after inpaint fill")
    ax[0, 2].imshow(overlay(d["imgs"][i], m["hide"][i])); ax[0, 2].set_title("hidden region (same 3D sphere\nin all 10 views)")
    j = 5
    ax[0, 3].imshow(overlay(d["imgs"][j], m["hide"][j])); ax[0, 3].set_title(f"same sphere, view {d['views'][j] + 1}")
    # row 2: CO3D
    ax[1, 0].imshow(cd["imgs"][i]); ax[1, 0].set_title("CO3D toytrain, view 1 (clean)")
    ax[1, 1].imshow(overlay(cd["imgs"][i], cm["hide"][i])); ax[1, 1].set_title("hidden region (gray fill in run)")
    ax[1, 2].imshow(panels[2]); ax[1, 2].set_title("VGGT depth (cyan: hidden region)")
    ax[1, 3].imshow(panels[3]); ax[1, 3].set_title("VGGT log-confidence")
    for a in ax.ravel():
        a.axis("off")
    for a in ax.ravel():
        a.title.set_fontsize(7)
    fig.subplots_adjust(wspace=0.04, hspace=0.25, left=0.01, right=0.99, top=0.92, bottom=0.01)
    save(fig, "fig1_teaser")


# ----------------------------------------------------------------------------- fig 2 pipeline
def box(ax, x, y, w, h, text, fc="#f3f2ee", ec=INK2, fs=7, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.8)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color,
                                 shrinkA=0, shrinkB=0))


def fig2():
    d, m = dtu_example()
    i = 0
    fig = plt.figure(figsize=(PAGEW, 2.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0.6, 4.6); ax.axis("off")
    y, h = 3.5, 0.95
    box(ax, 0.15, y, 1.45, h, "GT surface\n(DTU / CO3D depth)")
    box(ax, 1.95, y, 1.45, h, "3D sphere on\nthe surface\n(seed, radius)")
    box(ax, 3.75, y, 1.55, h, "per-view hide masks\n(3D-consistent:\nhidden in all views)")
    box(ax, 5.65, y, 1.35, h, "fill\ngray / blur /\nnoise / inpaint")
    box(ax, 7.35, y, 1.45, h, "frozen 3D model\n(10 views in,\ndepth + conf out)", fc="#e3edfa", ec=BLUE)
    box(ax, 9.1, y, 0.8, h, "depth,\nconf.", fc="#e3edfa", ec=BLUE)
    for x0, x1 in [(1.6, 1.95), (3.4, 3.75), (5.3, 5.65), (7.0, 7.35), (8.8, 9.1)]:
        arrow(ax, x0, y + h / 2, x1, y + h / 2)
    # labels (from the sphere) and metrics (from labels + model output)
    box(ax, 5.65, 1.55, 1.75, 1.2, "pixel labels\nU: unsupported\n(inside the sphere)\nS: supported (outside,\nbeyond a dilated ring)", fc="#fff5ee", ec=ORANGE, fs=6.5)
    arrow(ax, 4.525, y, 4.525, 2.15); arrow(ax, 4.525, 2.15, 5.65, 2.15)
    box(ax, 7.75, 1.2, 2.15, 1.9, "metrics\nAUROC: conf(S) > conf(U)\nCUR: share of U above the\nthreshold that keeps 90% of S\nCUR-err: confident and\nAbsRel > 0.1\nAbsRel_U / AbsRel_S", fc="#fff5ee", ec=ORANGE, fs=6.5)
    arrow(ax, 9.5, y, 9.5, 3.1); arrow(ax, 7.4, 2.15, 7.75, 2.15)
    # thumbnails
    thumbs = [("clean view", d["imgs"][i]), ("hide mask", overlay(d["imgs"][i], m["hide"][i]))]
    for name in ["gray", "blur", "noise", "inpaint"]:
        thumbs.append((name, H.apply_fill(d["imgs"], m["hide"], name, 0)[i]))
    for k, (name, im) in enumerate(thumbs):
        a = fig.add_axes([0.015 + k * 0.088, 0.02, 0.082, 0.40])
        a.imshow(im); a.axis("off"); a.set_title(name, fontsize=6.5, pad=2)
    ax.text(0.15, 2.85, "example: DTU scan 3, seed 0, radius 0.18 of the object diagonal, view 1", fontsize=6.5, color=INK2)
    save(fig, "fig2_pipeline")


# ----------------------------------------------------------------------------- fig 3 per-run distributions
def fig3():
    df = t1[t1.fill.isin(["blur", "noise"])]
    fig, ax = plt.subplots(1, 2, figsize=(PAGEW, 2.3))
    rng = np.random.default_rng(0)
    for k, (col, lab) in enumerate([("auroc", "AUROC (per masked run)"), ("cur", "CUR (per masked run)")]):
        for j, mname in enumerate(ORDER):
            v = df[df.model == mname][col].values
            x = j + rng.uniform(-0.22, 0.22, len(v))
            ax[k].scatter(x, v, s=4, color=C[mname], alpha=0.35, lw=0)
            mu, lo, hi = ci(df[df.model == mname], col)
            ax[k].plot([j - 0.3, j + 0.3], [mu, mu], color=INK, lw=1.4)
            ax[k].plot([j, j], [lo, hi], color=INK, lw=0.8)
            ax[k].text(j, 1.03 if col == "auroc" else 1.03, f"{mu:.2f}", ha="center", fontsize=6.5, color=INK)
        ax[k].set_xticks(range(5)); ax[k].set_xticklabels([NAME[m] for m in ORDER], rotation=20, ha="right")
        ax[k].set_ylabel(lab); ax[k].set_ylim(0, 1.1); despine(ax[k])
        ax[k].text(-0.02, 1.06, "(a)" if k == 0 else "(b)", transform=ax[k].transAxes, fontweight="bold")
    ax[0].axhline(0.5, color=INK2, lw=0.6, ls=":"); ax[0].text(4.45, 0.52, "chance", fontsize=6, color=INK2, ha="right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_metrics")


# ----------------------------------------------------------------------------- fig 4 DTU vs CO3D
def fig4():
    a = t1[t1.fill.isin(["blur", "noise"])]; b = t3[t3.fill.isin(["blur", "noise"])]
    fig, ax = plt.subplots(1, 2, figsize=(PAGEW, 2.2))
    nums = {}
    for k, col in enumerate(["cur", "auroc"]):
        for j, mname in enumerate(ORDER):
            ma, la, ha_ = ci(a[a.model == mname], col); mb, lb, hb = ci(b[b.model == mname], col)
            nums[(col, mname)] = (ma, la, ha_, mb, lb, hb)
            yy = 4 - j
            ax[k].plot([ma, mb], [yy, yy], color=GRID, lw=2.5, zorder=1)
            ax[k].errorbar(ma, yy + 0.08, xerr=[[ma - la], [ha_ - ma]], fmt="o", color=BLUE, ms=4.5, capsize=0, lw=0.9, zorder=3)
            ax[k].errorbar(mb, yy - 0.08, xerr=[[mb - lb], [hb - mb]], fmt="o", color=ORANGE, ms=4.5, capsize=0, lw=0.9, zorder=3)
            ax[k].text(max(ha_, hb) + 0.02, yy, f"{ma:.2f} → {mb:.2f}", va="center", fontsize=6.5, color=INK2)
        ax[k].set_yticks(range(5)); ax[k].set_yticklabels([NAME[m] for m in ORDER][::-1])
        ax[k].set_xlim(0, 1.18); ax[k].set_xlabel({"cur": "CUR (lower is better)", "auroc": "AUROC (higher is better)"}[col])
        ax[k].grid(axis="x", color=GRID, lw=0.5); ax[k].set_axisbelow(True)
        for s in ["top", "right"]:
            ax[k].spines[s].set_visible(False)
        ax[k].text(-0.02, 1.04, "(a)" if k == 0 else "(b)", transform=ax[k].transAxes, fontweight="bold")
    ax[1].axvline(0.5, color=INK2, lw=0.6, ls=":")
    h1, = ax[0].plot([], [], "o", color=BLUE, label="DTU (18 scans)"); h2, = ax[0].plot([], [], "o", color=ORANGE, label="CO3Dv2 (18 scenes)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), handletextpad=0.3)
    fig.tight_layout(w_pad=2)
    save(fig, "fig4_dtu_vs_co3d")
    return nums


# ----------------------------------------------------------------------------- fig 5 fills
def fig5():
    inp = pd.read_csv("results/P2-stress/P2-stress_inpaint10.csv")
    fills = ["gray", "blur", "noise"]
    fig, ax = plt.subplots(1, 2, figsize=(PAGEW, 2.2), sharey=True)
    hatch = {"gray": "", "blur": "", "noise": "", "inpaint": "////"}
    alpha = {"gray": 0.35, "blur": 0.65, "noise": 1.0, "inpaint": 0.5}
    for k, (df, title, fl) in enumerate([(t1, "DTU", fills + ["inpaint"]), (t3, "CO3Dv2", fills)]):
        w = 0.8 / len(fl)
        for j, mname in enumerate(ORDER):
            for q, f in enumerate(fl):
                src = inp if f == "inpaint" else df
                d = src[(src.model == mname) & (src.fill == f)]
                if len(d) == 0:
                    continue
                mu, lo, hi = ci(d, "cur")
                x = j - 0.4 + w * (q + 0.5)
                ax[k].bar(x, mu, w * 0.92, color=C[mname], alpha=alpha[f], hatch=hatch[f], ec="white", lw=0.5)
                ax[k].plot([x, x], [lo, hi], color=INK, lw=0.7)
        ax[k].set_xticks(range(5)); ax[k].set_xticklabels([NAME[m] for m in ORDER], rotation=20, ha="right")
        ax[k].set_ylim(0, 1); despine(ax[k])
        ax[k].text(-0.02, 1.04, ("(a) " if k == 0 else "(b) ") + title, transform=ax[k].transAxes, fontweight="bold")
    ax[0].set_ylabel("CUR")
    from matplotlib.patches import Patch
    leg = [Patch(fc=INK2, alpha=alpha["gray"], label="gray"), Patch(fc=INK2, alpha=alpha["blur"], label="blur"),
           Patch(fc=INK2, alpha=alpha["noise"], label="noise"), Patch(fc=INK2, alpha=alpha["inpaint"], hatch="////", ec="white", label="inpaint (DTU only, seeds 0–1, VGGT/Pi3X/DA3)")]
    fig.legend(handles=leg, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), columnspacing=1.2, handlelength=1.2)
    fig.tight_layout(w_pad=1)
    save(fig, "fig5_fills")


# ----------------------------------------------------------------------------- fig 6 stress
def parse(s):
    s = str(s); m = float(s.split()[0]); lo, hi = s[s.index("[") + 1:s.index("]")].split(","); return m, float(lo), float(hi)


def fig6():
    df = pd.read_csv("results/P2-stress/stress_vs_reference.csv")
    tests = [("reference", "reference\n(blur, 10 views)"), ("inpaint10", "inpaint fill"), ("blur5", "5 views"),
             ("blur3", "3 views"), ("adjacent10", "low parallax\n(10 adjacent)"), ("distractor10", "+ distractor\nimage")]
    models = ["vggt", "pi3x", "da3"]
    fig, ax = plt.subplots(figsize=(PAGEW, 2.2))
    w = 0.26; nums = {}
    for j, (key, lab) in enumerate(tests):
        for q, mname in enumerate(models):
            row = df[(df.model == mname) & (df.test.str.startswith(key))].iloc[0]
            mu, lo, hi = parse(row.cur); nums[(mname, key)] = (mu, lo, hi, row.cur_diff_vs_ref)
            x = j - 0.26 + q * w
            ax.bar(x, mu, w * 0.9, color=C[mname], ec="white", lw=0.5, alpha=0.5 if key == "reference" else 1)
            ax.plot([x, x], [lo, hi], color=INK, lw=0.7)
            if key != "reference" and isinstance(row.cur_diff_vs_ref, str):
                p = float(row.cur_diff_vs_ref.split("p=")[1].rstrip(")"))
                if p < 0.05:
                    ax.text(x, hi + 0.02, "*", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(tests))); ax.set_xticklabels([t[1] for t in tests])
    ax.set_ylabel("CUR"); ax.set_ylim(0, 1.0); despine(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C[m], label=NAME[m]) for m in models], loc="upper left", ncol=3)
    ax.text(0.01, 0.84, "* paired sign-flip p < 0.05 vs reference (18 scans × 2 seeds)", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig6_stress")
    return nums


# ----------------------------------------------------------------------------- fig 7 per-scene heatmap
def fig7():
    def pivot(df):
        d = df[(df.fill == "blur") & (df.radius_frac == 0.18)]
        return d.pivot_table(index="scan", columns="model", values="cur")[ORDER]
    pa, pb = pivot(t1), pivot(t3)
    pb.index = [s.split("_")[0] + " " + s.split("_")[1] for s in pb.index]
    pa.index = [f"scan {s}" for s in pa.index]
    fig, ax = plt.subplots(1, 2, figsize=(PAGEW, 3.4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45})
    for k, (p, title) in enumerate([(pa, "DTU"), (pb, "CO3Dv2")]):
        im = ax[k].imshow(p.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax[k].set_xticks(range(5)); ax[k].set_xticklabels([NAME[m] for m in ORDER], rotation=30, ha="right")
        ax[k].set_yticks(range(len(p))); ax[k].set_yticklabels(p.index, fontsize=6)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax[k].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else INK)
        ax[k].text(-0.02, 1.02, ("(a) " if k == 0 else "(b) ") + title, transform=ax[k].transAxes, fontweight="bold")
        ax[k].tick_params(length=0)
        for s in ax[k].spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.set_label("CUR (blur, r = 0.18, mean of 5 seeds)")
    cb.outline.set_visible(False)
    save(fig, "fig7_perscene")


# ----------------------------------------------------------------------------- fig 8 cheap detectors
def fig8():
    pv = pd.read_csv("results/P3-pv-smoke/P3-pv-smoke.csv")
    sb = pd.read_csv("results/P2-smoke-b/P2-smoke-b.csv")
    nat = t1[(t1.fill == "blur") & (t1.seed == 0) & (t1.radius_frac == 0.18)]
    pairs = [("native", nat[nat.model == "vggt"], C["vggt"], 1.0),
             ("perturb.\nvariance", pv[pv.model == "vggt_pv4"], C["vggt"], 0.45),
             ("native", nat[nat.model == "pi3x"], C["pi3x"], 1.0),
             ("perturb.\nvariance", pv[pv.model == "pi3x_pv4"], C["pi3x"], 0.45),
             ("learned", sb[sb.model == "mapanything"], C["mapanything"], 1.0),
             ("multi-view\nconsistency", sb[sb.model == "mapanything_mv"], C["mapanything"], 0.45)]
    fig, ax = plt.subplots(1, 2, figsize=(PAGEW, 2.2))
    for k, col in enumerate(["auroc", "cur"]):
        for j, (lab, d, c, al) in enumerate(pairs):
            mu, lo, hi = ci(d, col)
            x = j + (0.18 if j % 2 else -0.18) * 0 + (j // 2) * 0.35
            ax[k].bar(x, mu, 0.8, color=c, alpha=al, ec="white", lw=0.5)
            ax[k].plot([x, x], [lo, hi], color=INK, lw=0.7)
            ax[k].text(x, hi + 0.02, f"{mu:.2f}", ha="center", fontsize=6)
        xs = [j + (j // 2) * 0.35 for j in range(6)]
        ax[k].set_xticks(xs); ax[k].set_xticklabels([p[0] for p in pairs], fontsize=6)
        for g, lab in enumerate(["VGGT-1B", "Pi3X", "MapAnything"]):
            ax[k].text(xs[2 * g] + 0.5, -0.33, lab, ha="center", va="top", fontsize=6.5, fontweight="bold", color=INK)
        ax[k].set_ylim(0, 1.12); despine(ax[k])
        ax[k].set_ylabel({"auroc": "AUROC", "cur": "CUR"}[col])
        ax[k].text(-0.02, 1.04, "(a)" if k == 0 else "(b)", transform=ax[k].transAxes, fontweight="bold")
    ax[0].axhline(0.5, color=INK2, lw=0.6, ls=":")
    fig.text(0.5, -0.1, "DTU, 18 scans, seed 0. Native vs perturbation variance: blur, r 0.18. MapAnything learned vs multi-view: gray, r 0.15.",
             ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_cheap_detectors")


# ----------------------------------------------------------------------------- fig 9 method (proposed)
def fig9():
    fig = plt.figure(figsize=(PAGEW, 2.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    y, h = 2.0, 1.0
    box(ax, 0.15, y, 1.5, h, "unposed photos\n(N views)")
    box(ax, 1.95, y, 1.6, h, "frozen 3D backbone\n(VGGT / Pi3 / ...)\ntokens, depth, conf", fc="#e3edfa", ec=BLUE, fs=6.5)
    box(ax, 3.85, y, 2.0, h, "discrete diffusion head over\nquantized scene tokens (trained\nby masking + corrupting\nclean outputs)", fc="#eceafa", ec=C["pi3x"], fs=6.5)
    box(ax, 6.15, y, 1.1, h, "K samples of\nthe recon-\nstruction")
    box(ax, 7.55, y, 1.2, h, "disagreement\nmap (per pixel)", fc="#fff5ee", ec=ORANGE)
    box(ax, 9.0, y, 0.9, h, "abstain /\nkeep", fc="#fff5ee", ec=ORANGE)
    for x0, x1 in [(1.65, 1.95), (3.55, 3.85), (5.85, 6.15), (7.25, 7.55), (8.75, 9.0)]:
        arrow(ax, x0, y + h / 2, x1, y + h / 2)
    ax.text(4.75, y + h + 0.18, "observed tokens are kept fixed; only unobserved regions are sampled", ha="center", fontsize=6.5, color=INK2)
    ax.text(8.1, y - 0.25, "samples agree: observed\nsamples disagree: imagined", ha="center", va="top", fontsize=6.5, color=INK2)
    # continual loop
    box(ax, 1.95, 0.25, 3.7, 0.95, "continual adaptation on a site stream (proposed)\nLoRA on the head + replay buffer of past sites;\nthe backbone stays frozen", fc="#f3f2ee")
    arrow(ax, 1.0, 0.72, 1.95, 0.72)
    ax.text(0.15, 0.72, "new site\nphotos", fontsize=6.5, color=INK2, va="center")
    arrow(ax, 4.75, 1.2, 4.75, y)
    ax.text(4.85, 1.55, "updates the head only", fontsize=6.5, color=INK2)
    ax.text(0.15, 3.35, "Proposed detector (design figure; numbers TBD until the head is trained)", fontsize=7, color=INK, fontweight="bold")
    ax.text(0.15, 1.75, "no hallucination labels\nare used in training", fontsize=6.5, color=INK2, va="center")
    save(fig, "fig9_method")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    n4 = fig4(); fig5(); n6 = fig6(); fig7(); fig8(); fig9()
    print("\nfig4 numbers (DTU mean [lo, hi] -> CO3D mean [lo, hi]):")
    for k, v in n4.items():
        print(k, "DTU %.3f [%.3f, %.3f]  CO3D %.3f [%.3f, %.3f]" % v)
    print("\nfig6 numbers (CUR mean [lo, hi], diff vs ref):")
    for k, v in n6.items():
        print(k, v)
