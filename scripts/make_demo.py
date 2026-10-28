"""Build the HalluBench-3D demo video (paper/demo/demo.mp4).

Part A runs the real masking protocol on real DTU and CO3D data on the CPU (hallu3d.build_masks,
apply_fill, consistency_check) and shows the numbers the code returns.
Part B replays model outputs recorded on Kaggle (composites under results/*/ex_*, metrics from the CSVs).

Usage (from the project root):
  python scripts/make_demo.py --dtu <dtu_root_search_dir> --co3d <co3d_prep_root> [--out paper/demo/demo.mp4]
Frames are rendered with matplotlib at 1920x1080 into a scratch directory and assembled with ffmpeg
(concat demuxer, per-frame durations, encoded at 30 fps H.264).
"""
import argparse
import glob
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H
import co3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FFMPEG = "/opt/homebrew/bin/ffmpeg"

BG, FG, MUTED, ACC, ACC2 = "#101418", "#f2f4f7", "#9aa4b2", "#4cc9f0", "#f4a261"
MODEL_LABEL = {"vggt": "VGGT-1B", "pi3": "Pi3", "pi3x": "Pi3X", "mapanything": "MapAnything", "da3": "Depth Anything 3"}
ORDER = ["vggt", "pi3", "pi3x", "mapanything", "da3"]
REPLAY_TAG = "model outputs recorded on Kaggle T4, 2026-10-28"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "font.family": "Helvetica",
    "font.size": 22, "axes.titlesize": 26, "axes.titleweight": "bold",
})


class Reel:
    """Collects (png, seconds) pairs and writes the ffmpeg concat file."""

    def __init__(self, frame_dir):
        self.frame_dir = frame_dir
        os.makedirs(frame_dir, exist_ok=True)
        self.items = []
        self.t = 0.0
        self.marks = []

    def fig(self):
        return plt.figure(figsize=(19.2, 10.8), dpi=100)

    def add(self, fig, seconds):
        p = os.path.join(self.frame_dir, f"f{len(self.items):05d}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        self.items.append((p, seconds))
        self.t += seconds

    def mark(self, name):
        self.marks.append((self.t, name))

    def write(self, out_mp4):
        lst = os.path.join(self.frame_dir, "list.txt")
        with open(lst, "w") as f:
            for p, s in self.items:
                f.write(f"file '{p}'\nduration {s:.3f}\n")
            f.write(f"file '{self.items[-1][0]}'\n")
        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "fps=30,format=yuv420p",
               "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart", out_mp4]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ----------------------------------------------------------------------------- frame helpers
def header(fig, title, sub=None, tag=None):
    fig.text(0.04, 0.945, title, fontsize=34, fontweight="bold", va="center")
    if sub:
        fig.text(0.04, 0.895, sub, fontsize=24, color=MUTED, va="center")
    if tag:
        fig.text(0.96, 0.945, tag, fontsize=20, color=ACC2, va="center", ha="right")


def card(reel, title, lines, seconds, sub=None, tag=None, fs=30):
    fig = reel.fig()
    header(fig, title, sub, tag)
    y = 0.78
    for ln in lines:
        fig.text(0.07, y, ln, fontsize=fs, va="top")
        y -= 0.085 * (fs / 30)
    reel.add(fig, seconds)


def show_img(ax, img, title=None):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=22, pad=6)


def views_grid(reel, data, title, sub, seconds, hide=None, highlight=None):
    fig = reel.fig()
    header(fig, title, sub)
    N = len(data["imgs"])
    cols = 5
    rows = int(np.ceil(N / cols))
    for i in range(N):
        ax = fig.add_axes([0.04 + (i % cols) * 0.187, 0.50 - (i // cols) * 0.41, 0.175, 0.36])
        img = data["imgs"][i].copy()
        if hide is not None:
            m = hide[i]
            img[m] = img[m] * 0.25 + np.array([0.3, 0.8, 0.95]) * 0.75
        show_img(ax, img, f"view {data['views'][i]}")
    reel.add(fig, seconds)


def sphere_frames(reel, data, masks, title, n=14, seconds_each=0.28):
    """3D scatter of back-projected GT points, sphere drawn, rotating camera."""
    depths, Kd, E = data["depths"], data["K_dep"], data["E"]
    pts, cols = [], []
    for i in range(len(depths)):
        P = H.backproject(depths[i], Kd[i], E[i])
        v = depths[i] > 0
        P = P[v]
        img = data["imgs"][i]
        h, w = depths.shape[1:]
        Hi, Wi = img.shape[:2]
        im_small = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h)))
        c = im_small[v] / 255.0
        step = max(1, len(P) // 6000)
        pts.append(P[::step]); cols.append(c[::step])
    P = np.concatenate(pts); C = np.concatenate(cols)
    inside = np.linalg.norm(P - masks["center"], axis=1) < masks["radius"]
    ctr = np.median(P, axis=0)
    span = np.percentile(np.linalg.norm(P - ctr, axis=1), 97)
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:16j]
    r = masks["radius"]; c0 = masks["center"]
    sx = c0[0] + r * np.cos(u) * np.sin(v); sy = c0[1] + r * np.sin(u) * np.sin(v); sz = c0[2] + r * np.cos(v)
    for k in range(n):
        fig = reel.fig()
        header(fig, title, "one sphere on the ground-truth surface, hidden in every view that sees it")
        ax = fig.add_axes([0.08, 0.04, 0.84, 0.82], projection="3d")
        ax.set_facecolor(BG)
        ax.scatter(P[~inside, 0], P[~inside, 1], P[~inside, 2], c=C[~inside], s=1.2, depthshade=False)
        if k >= 3:
            ax.scatter(P[inside, 0], P[inside, 1], P[inside, 2], c=ACC, s=4, depthshade=False)
            ax.plot_wireframe(sx, sy, sz, color=ACC, linewidth=0.6, alpha=0.6)
        ax.set_xlim(ctr[0] - span, ctr[0] + span); ax.set_ylim(ctr[1] - span, ctr[1] + span); ax.set_zlim(ctr[2] - span, ctr[2] + span)
        ax.view_init(elev=-60 + 4 * np.sin(k / 4), azim=-90 + 2.2 * k)
        ax.set_axis_off()
        txt = f"radius = {masks['radius']:.2f} ({masks['radius'] / masks['diag']:.2f} x object diagonal)"
        if k >= 3:
            fig.text(0.5, 0.06, txt, ha="center", fontsize=26, color=ACC)
        reel.add(fig, seconds_each)


def fills_frame(reel, data, masks, title, seconds):
    fig = reel.fig()
    header(fig, title, "the same hidden region, four fills; the models never see the gray patch alone")
    i = 0
    for j, mode in enumerate(["gray", "blur", "noise", "inpaint"]):
        f = H.apply_fill(data["imgs"][i:i + 1], masks["hide"][i:i + 1], mode, 0)[0]
        ax = fig.add_axes([0.03 + j * 0.24, 0.10, 0.225, 0.72])
        show_img(ax, f, mode)
    reel.add(fig, seconds)


def labels_frame(reel, data, masks, title, seconds):
    fig = reel.fig()
    header(fig, title, "labels at depth resolution: U unsupported (inside, hidden), S supported (outside a dilated ring)")
    i = 0
    h, w = masks["U"].shape[1:]
    lab = np.zeros((h, w, 3), np.float32) + 0.12
    valid = data["depths"][i] > 0
    lab[valid] = [0.35, 0.35, 0.35]
    lab[masks["S"][i]] = [0.3, 0.75, 0.4]
    ring = valid & ~masks["S"][i] & ~masks["U"][i]
    lab[ring] = [0.95, 0.65, 0.2]
    lab[masks["U"][i]] = [0.3, 0.8, 0.95]
    ax = fig.add_axes([0.04, 0.08, 0.44, 0.76]); show_img(ax, data["imgs"][i], "input view 0")
    ax = fig.add_axes([0.52, 0.08, 0.44, 0.76]); show_img(ax, lab, "labels")
    fig.text(0.74, 0.03, "U  unsupported", color=[0.3, 0.8, 0.95], ha="center", fontsize=24)
    fig.text(0.58, 0.03, "S  supported", color=[0.3, 0.75, 0.4], ha="center", fontsize=24)
    fig.text(0.90, 0.03, "ring, ignored", color=[0.95, 0.65, 0.2], ha="center", fontsize=24)
    nU, nS = int(masks["U"].sum()), int(masks["S"].sum())
    fig.text(0.26, 0.03, f"{nU} U pixels, {nS} S pixels over {len(data['imgs'])} views", ha="center", fontsize=22, color=MUTED)
    reel.add(fig, seconds)


def consistency_frame(reel, data, masks, title, seconds):
    chk = H.consistency_check(data, masks)
    lines = [
        "consistency_check(data, masks) returned:",
        f"   reproj_agree (1% tolerance)              = {chk['reproj_agree']:.3f}",
        f"   reproj_agree_3pct_vis (occlusion-aware)  = {chk['reproj_agree_3pct_vis']:.3f}",
        f"   U_transfer (U points land in other masks) = {chk['U_transfer']:.3f}",
        "",
        "The first two check cameras and depth agree across views.",
        "The third checks the hidden region is the same 3D region in every view.",
    ]
    card(reel, title, lines, seconds, sub="computed now, on this machine", fs=28)
    return chk


# ----------------------------------------------------------------------------- part B helpers
def composite_frame(reel, png, model, scene, row, title, seconds, dataset):
    fig = reel.fig()
    header(fig, title, f"{MODEL_LABEL[model]}   |   {dataset} scene {scene}", tag=REPLAY_TAG)
    im = np.asarray(Image.open(png).convert("RGB"))
    ax = fig.add_axes([0.02, 0.36, 0.96, 0.50]); show_img(ax, im)
    fig.text(0.02, 0.31, "left to right: masked input, GT depth, model depth, model log-confidence. Cyan outline = hidden region.",
             fontsize=21, color=MUTED)
    if row is not None:
        fig.text(0.04, 0.20, f"fill {row['fill']}, seed {int(row['seed'])}, radius {row['radius_frac']:.2f} x diagonal", fontsize=24, color=MUTED)
        fig.text(0.04, 0.12, f"AUROC {row['auroc']:.3f}", fontsize=30)
        fig.text(0.30, 0.12, f"CUR {row['cur']:.3f}", fontsize=30, color=ACC)
        fig.text(0.52, 0.12, f"CUR-err {row['cur_err']:.3f}", fontsize=30)
        fig.text(0.76, 0.12, f"AbsRel U {row['absrel_U']:.3f}  S {row['absrel_S']:.3f}", fontsize=30)
        fig.text(0.04, 0.05, "CUR: share of unsupported pixels kept by a threshold that keeps 90% of supported pixels. Lower is better.",
                 fontsize=20, color=MUTED)
    reel.add(fig, seconds)


def bar_chart_frames(reel, title, sub, groups, series, n_steps=10, hold=5.0, ylim=(0, 1), ylabel="CUR"):
    """groups: list of labels; series: dict name -> list of (mean, lo, hi). Bars grow over n_steps frames."""
    names = list(series)
    x = np.arange(len(groups))
    wdt = 0.8 / len(names)
    colors = [ACC, ACC2, "#90be6d", "#f94144"]
    for k in range(1, n_steps + 1):
        fr = k / n_steps
        fig = reel.fig()
        header(fig, title, sub, tag=REPLAY_TAG)
        ax = fig.add_axes([0.08, 0.14, 0.88, 0.68])
        for j, nm in enumerate(names):
            vals = series[nm]
            m = np.array([v[0] for v in vals]) * fr
            lo = np.array([v[0] - v[1] for v in vals]) * fr
            hi = np.array([v[2] - v[0] for v in vals]) * fr
            ax.bar(x + (j - (len(names) - 1) / 2) * wdt, m, wdt * 0.92, color=colors[j], label=nm)
            if k == n_steps:
                ax.errorbar(x + (j - (len(names) - 1) / 2) * wdt, m, yerr=[lo, hi], fmt="none", ecolor=FG, capsize=6, lw=1.5)
                for xi, mi in zip(x + (j - (len(names) - 1) / 2) * wdt, m):
                    ax.text(xi, mi + 0.02, f"{mi:.2f}", ha="center", fontsize=20)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=24)
        ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=24)
        ax.grid(axis="y", color="#2a3139"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=22, frameon=False, loc="upper left")
        reel.add(fig, hold if k == n_steps else 0.08)


def pooled(df, model, col):
    d = df[(df.model == model) & (df.fill.isin(["blur", "noise"]))]
    m, lo, hi, n = H.cluster_bootstrap_ci(d, col, n_boot=3000)
    return (m, lo, hi)


def parse_ci(s):
    s = str(s)
    m = float(s.split("[")[0]); lo, hi = [float(v) for v in s.split("[")[1].rstrip("]").split(",")]
    return (m, lo, hi)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True, help="directory searched for a DTU root (Cameras/train, Rectified, Depths)")
    ap.add_argument("--co3d", required=True, help="preprocessed CO3D root (folders with meta.npz)")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "demo", "demo.mp4"))
    ap.add_argument("--frames", default=os.path.join(ROOT, "paper", "demo", "_frames"))
    args = ap.parse_args()
    reel = Reel(args.frames)

    # ------------------------------------------------------------------ title
    reel.mark("title")
    card(reel, "Observe, Imagine, Abstain: HalluBench-3D demo",
         ["A benchmark for geometric hallucination in feed-forward 3D reconstruction.",
          "",
          "Part A  runs the masking protocol live on real DTU and CO3Dv2 data (CPU, this machine).",
          "Part B  replays model outputs recorded on a Kaggle T4 on 2026-10-28.",
          "",
          "Every number shown is read from the code or from a logged run."], 6.0)

    # ------------------------------------------------------------------ part A, DTU
    reel.mark("part A, DTU")
    card(reel, "Part A: the masking protocol (live)",
         ["1. Load 10 views of a real scan with ground-truth depth and cameras.",
          "2. Place one sphere on the ground-truth surface.",
          "3. Hide everything inside the sphere in every view that sees it.",
          "4. Fill the hole so the models see no obvious patch.",
          "5. Label pixels: unsupported (U), supported (S), ring (ignored).",
          "6. Check the geometry before any model runs."], 7.0)
    dtu_root = H.find_dtu_root([args.dtu])
    cfg = H.DEFAULT_CFG
    data = H.load_scan(dtu_root, 3, H.select_views(10), cfg["light"])
    masks = H.build_masks(data, 0, 0.18, cfg)
    views_grid(reel, data, "DTU scan 3: the 10 input views", "MVSNet split, light 3, 512x640, cameras from Cameras/train", 4.5)
    sphere_frames(reel, data, masks, "DTU scan 3: sphere on the GT surface")
    views_grid(reel, data, "DTU scan 3: the hidden region in all 10 views", "one 3D sphere, projected with the GT depth, so the hole is consistent across views", 5.0, hide=masks["hide"])
    fills_frame(reel, data, masks, "DTU scan 3: fills", 6.0)
    labels_frame(reel, data, masks, "DTU scan 3: labels", 5.0)
    chk_dtu = consistency_frame(reel, data, masks, "DTU scan 3: geometry check", 7.0)

    # ------------------------------------------------------------------ part A, CO3D
    reel.mark("part A, CO3D")
    scenes = co3d.list_co3d_scenes(args.co3d)
    scene = "hydrant_167_18184_34441" if "hydrant_167_18184_34441" in scenes else scenes[0]
    d2 = co3d.load_co3d_scene(args.co3d, scene, H.select_views(10, total=co3d.n_frames(args.co3d, scene)))
    m2 = H.build_masks(d2, 0, 0.18, cfg)
    views_grid(reel, d2, f"CO3Dv2 {scene}: the 10 input views", "phone capture, frames spread over the full orbit, COLMAP depth as ground truth", 4.5)
    sphere_frames(reel, d2, m2, f"CO3Dv2 {scene}: sphere on the GT surface", n=12)
    views_grid(reel, d2, f"CO3Dv2 {scene}: the hidden region in all 10 views", "the sphere is seen from the front and from the back", 5.0, hide=m2["hide"])
    chk_co3d = consistency_frame(reel, d2, m2, f"CO3Dv2 {scene}: geometry check", 7.0)

    # ------------------------------------------------------------------ part B, replay
    reel.mark("part B, DTU replay")
    t1 = pd.read_csv(os.path.join(RES, "table1_all_models.csv"))
    t3 = pd.read_csv(os.path.join(RES, "table3_all_models.csv"))
    card(reel, "Part B: what the models did", [
        "Five models, each given the masked views, each asked for depth and confidence.",
        "",
        "AUROC: can the confidence tell supported from unsupported pixels?",
        "CUR: share of unsupported pixels that pass a threshold keeping 90% of supported ones.",
        "CUR-err: confident and more than 10% wrong.",
        "",
        "Composites and numbers below are read from the logged Kaggle runs."], 8.0, tag=REPLAY_TAG)

    def find_png(dirs, model, scan, rf):
        for d in dirs:
            p = os.path.join(d, f"{model}_scan{scan}_r{rf}.png")
            if os.path.exists(p):
                return p
        return None

    dtu_dirs = [os.path.join(RES, "P1-full", "ex_P1-full"), os.path.join(RES, "P2-full", "ex_P2-full")]
    for model in ORDER:
        for scan in [17, 106]:
            png = find_png(dtu_dirs, model, scan, 0.18)
            if png is None:
                continue
            r = t1[(t1.model == model) & (t1.scan == scan) & (t1.seed == 0) & (t1.radius_frac == 0.18) & (t1.fill == "gray")]
            composite_frame(reel, png, model, f"scan {scan}", r.iloc[0] if len(r) else None, "DTU: recorded model output", 3.2, "DTU")

    reel.mark("part B, inpaint stress")
    sdir = os.path.join(RES, "P2-stress", "ex_P2-stress_inpaint10")
    sdf = pd.read_csv(os.path.join(RES, "P2-stress", "P2-stress_inpaint10.csv"))
    for model in ["vggt", "pi3x", "da3"]:
        png = find_png([sdir], model, 17, 0.18)
        if png is None:
            continue
        r = sdf[(sdf.model == model) & (sdf.scan == 17) & (sdf.seed == 0) & (sdf.radius_frac == 0.18)]
        composite_frame(reel, png, model, "scan 17, inpaint fill (no visible seam)", r.iloc[0] if len(r) else None,
                        "DTU stress test: inpainted fill", 3.5, "DTU")

    reel.mark("part B, CO3D replay")
    co_dirs = [os.path.join(RES, "P2-co3d", "ex_P2-co3d"), os.path.join(RES, "P2-co3d-b", "ex_P2-co3d-b")]
    for model in ORDER:
        for sc in ["toytrain_399_51323_100753", "apple_189_20393_38136"]:
            png = find_png(co_dirs, model, sc, 0.18)
            if png is None:
                continue
            r = t3[(t3.model == model) & (t3.scan == sc) & (t3.seed == 0) & (t3.radius_frac == 0.18) & (t3.fill == "gray")]
            composite_frame(reel, png, model, sc, r.iloc[0] if len(r) else None, "CO3Dv2: recorded model output", 3.2, "CO3Dv2")

    # ------------------------------------------------------------------ charts
    reel.mark("charts")
    labels = [MODEL_LABEL[m] for m in ORDER]
    bar_chart_frames(reel, "Leakage: CUR pooled over blur and noise fills", "mean with 95% cluster bootstrap CI over 18 scenes; lower is better",
                     labels, {"DTU (18 scans)": [pooled(t1, m, "cur") for m in ORDER], "CO3Dv2 (18 scenes)": [pooled(t3, m, "cur") for m in ORDER]})
    bar_chart_frames(reel, "Separability: AUROC pooled over blur and noise fills", "0.5 is chance; higher is better",
                     labels, {"DTU (18 scans)": [pooled(t1, m, "auroc") for m in ORDER], "CO3Dv2 (18 scenes)": [pooled(t3, m, "auroc") for m in ORDER]},
                     ylim=(0.4, 1.0), ylabel="AUROC")
    st = pd.read_csv(os.path.join(RES, "P2-stress", "stress_vs_reference.csv"))
    tests = ["reference(blur,10 spread)", "inpaint10", "blur5", "blur3", "adjacent10", "distractor10"]
    tlabels = ["reference", "inpaint", "5 views", "3 views", "low parallax", "distractor"]
    series = {}
    for m in ["vggt", "pi3x", "da3"]:
        vals = []
        for t in tests:
            r = st[(st.model == m) & (st.test == t)]
            vals.append(parse_ci(r.iloc[0]["cur"]) if len(r) else (0, 0, 0))
        series[MODEL_LABEL[m]] = vals
    bar_chart_frames(reel, "Stress tests on DTU: CUR under changed evidence", "18 scans x 2 seeds, radius 0.18; reference = blur fill, 10 spread views",
                     tlabels, series, hold=7.0)

    # ------------------------------------------------------------------ closing
    reel.mark("closing")
    card(reel, "What is measured, and what is not yet", [
        "Measured (logged runs, CSV + config in results/):",
        "   HalluBench-3D on DTU (18 scans) and CO3Dv2 (18 scenes), five models, three fills, 5 seeds, 2 radii.",
        "   Stress tests: inpainted fill, 5 and 3 views, low parallax, distractor image.",
        "   Two training-free detectors (perturbation variance, multi-view consistency).",
        "",
        "Not yet run (reported as TBD in the paper):",
        "   The discrete diffusion head and its K-sample disagreement detector (needs an H200).",
        "   Continual adaptation and the heritage photo study.",
        "   VGGT-Omega (gated checkpoint).",
        "",
        f"Geometry checks shown live: DTU reproj {chk_dtu['reproj_agree']:.2f}, CO3D occlusion-aware {chk_co3d['reproj_agree_3pct_vis']:.2f}."],
        9.0, fs=26)

    reel.write(args.out)
    for t, name in reel.marks:
        print(f"{int(t // 60):d}:{int(t % 60):02d}  {name}")
    print(f"total {reel.t:.1f}s, {len(reel.items)} frames -> {args.out}")


if __name__ == "__main__":
    main()
