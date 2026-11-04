"""Build the end-to-end walkthrough video (paper/demo/walkthrough.mp4).

Unlike make_demo.py this needs no DTU or CO3D data, so it runs on any machine with the Python
requirements and ffmpeg. It follows the whole pipeline in order:

  1. the protocol, live: scripts/test_synthetic.py is run as a subprocess; its synthetic DTU-format scene is
     loaded with the real loaders and masked, filled, labelled and checked with the real benchmark code;
  2. models and metrics, live: the two reference models of that test (oracle, blind) go through
     hallu3d.run_benchmark and the metric table and the reprojection detector are shown as computed;
  3. the recorded runs: composites and rows from the Kaggle runs under results/ (tagged as replay);
  4. the numbers: CUR / AUROC charts with cluster-bootstrap intervals recomputed from the CSVs, the stress
     tests and the training-free detectors;
  5. the paper figures, the proposed method and what is still pending;
  6. the artefacts the repository produces (PDFs, site, archive), read from disk at build time.

Usage (from the project root):
  python scripts/make_walkthrough.py [--out paper/demo/walkthrough.mp4] [--frames <scratch dir>]
Frames are 1920x1080 PNGs rendered with matplotlib and assembled with ffmpeg (30 fps, H.264). The MP4
carries chapter markers. Every number on screen is read from the code or from a logged run."""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H
from make_demo import (Reel, card, header, show_img, views_grid, sphere_frames, fills_frame, labels_frame,
                       consistency_frame, bar_chart_frames, pooled, parse_ci,
                       MODEL_LABEL, ORDER, REPLAY_TAG, ACC, ACC2, MUTED, FG, RES, ROOT, FFMPEG)

LIVE_TAG = "computed now, on this machine"
SYNTH_TAG = "live, synthetic scene (no DTU on this machine)"
FIG = os.path.join(ROOT, "paper", "figures")


# ----------------------------------------------------------------------------- extra frame helpers
def console_frame(reel, title, sub, lines, seconds, tag=LIVE_TAG, fs=20):
    """Monospace block, for real program output."""
    fig = reel.fig()
    header(fig, title, sub, tag)
    y = 0.80
    for ln in lines:
        fig.text(0.05, y, ln[:118], fontsize=fs, family="monospace", va="top", color=FG)
        y -= 0.052 * (fs / 20)
    reel.add(fig, seconds)


def figure_frame(reel, png, title, sub, seconds, caption=None, tag=None, box=(0.06, 0.12, 0.88, 0.72)):
    """One paper figure (white image) on the dark slide."""
    fig = reel.fig()
    header(fig, title, sub, tag)
    im = np.asarray(Image.open(png).convert("RGB"))
    ax = fig.add_axes(list(box))
    show_img(ax, im)
    if caption:
        fig.text(0.5, 0.05, caption, ha="center", fontsize=22, color=MUTED)
    reel.add(fig, seconds)


def composite(reel, png, label, scene, row, title, seconds, tag):
    """Model composite (masked input, GT depth, predicted depth, log-confidence) with its CSV row."""
    fig = reel.fig()
    header(fig, title, f"{label}   |   {scene}", tag=tag)
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


def thumbs_frame(reel, pngs, title, sub, seconds, cols=3):
    fig = reel.fig()
    header(fig, title, sub)
    n = len(pngs); rows = int(np.ceil(n / cols))
    cw, ch = 0.92 / cols, 0.78 / rows
    for i, p in enumerate(pngs):
        ax = fig.add_axes([0.04 + (i % cols) * cw, 0.86 - ch * (i // cols + 1), cw * 0.95, ch * 0.88])
        show_img(ax, np.asarray(Image.open(p).convert("RGB")), os.path.basename(p).replace(".png", ""))
    reel.add(fig, seconds)


def ci_of(df, col, **sel):
    d = df
    for k, v in sel.items():
        d = d[d[k] == v]
    m, lo, hi, n = H.cluster_bootstrap_ci(d, col, n_boot=3000)
    return (m, lo, hi)


def find_png(dirs, model, scan, rf):
    for d in dirs:
        p = os.path.join(d, f"{model}_scan{scan}_r{rf}.png")
        if os.path.exists(p):
            return p
    return None


def write_with_chapters(reel, out_mp4, frames_dir):
    """Reel.write, then a second ffmpeg pass that only adds chapter metadata (stream copy)."""
    tmp = os.path.join(frames_dir, "nochapters.mp4")
    reel.write(tmp)
    meta = os.path.join(frames_dir, "chapters.txt")
    marks = reel.marks + [(reel.t, None)]
    with open(meta, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\ntitle=HalluBench-3D end-to-end walkthrough\n")
        for (t0, name), (t1, _) in zip(marks[:-1], marks[1:]):
            f.write(f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(t0 * 1000)}\nEND={int(t1 * 1000)}\ntitle={name}\n")
    cmd = [FFMPEG, "-y", "-i", tmp, "-i", meta, "-map_metadata", "1", "-map_chapters", "1", "-codec", "copy",
           "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "demo", "walkthrough.mp4"))
    ap.add_argument("--frames", default=None, help="scratch dir for PNG frames (default: a fresh temp dir)")
    args = ap.parse_args()
    frames = args.frames or tempfile.mkdtemp(prefix="hallu3d_walkthrough_")
    reel = Reel(frames)

    # ------------------------------------------------------------------ 0. title
    reel.mark("title")
    card(reel, "HalluBench-3D, end to end",
         ["Observe, Imagine, Abstain: hallucination-aware 3D reconstruction.",
          "",
          "This video follows the repository from input views to the paper:",
          "   1. the masking protocol, run live by the benchmark code",
          "   2. models and metrics, run live on two reference models",
          "   3. the recorded runs of five real models (replayed, tagged)",
          "   4. the numbers, recomputed from the logged CSVs",
          "   5. the paper, the method and what is still pending",
          "Every number on screen is read from the code or from a logged run."], 9.0, fs=28)

    # ------------------------------------------------------------------ 1. problem
    reel.mark("the problem")
    figure_frame(reel, os.path.join(FIG, "fig1_teaser.png"), "The problem: confident geometry no camera saw",
                 "a 3D region of the surface is hidden in every input view; the model fills it and stays confident", 9.0,
                 caption="paper figure 1; model panels are recorded run outputs (VGGT). Cyan contour: pixels no input view supports.")
    card(reel, "Definition",
         ["A predicted point is hallucinated when it lies on a surface that no input view observes.",
          "",
          "The benchmark makes such labels for free:",
          "   hide one 3D region of the ground-truth surface in every view,",
          "   then every pixel inside is unsupported (U), every pixel far outside is supported (S).",
          "",
          "Two questions per model:",
          "   AUROC   does the confidence separate U from S?",
          "   CUR     how much of U survives a threshold that keeps 90% of S?  (lower is better)"], 10.0, fs=27)

    # ------------------------------------------------------------------ 2. protocol, live
    reel.mark("protocol, live")
    card(reel, "Stage 1: the protocol, live",
         ["No DTU or CO3D data is on this machine, so the benchmark's own CPU test is run instead:",
          "scripts/test_synthetic.py writes a synthetic scan in DTU format (cameras, PFM depth, images)",
          "and the real loaders, mask builder, fills, labels and checks run on it unchanged.",
          "",
          "Running it now ..."], 7.0, tag=LIVE_TAG, fs=27)
    synth = os.path.join(frames, "synth")
    os.makedirs(synth, exist_ok=True)
    env = {**os.environ, "HALLU3D_TMP": synth, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "test_synthetic.py")], env=env,
                          capture_output=True, text=True)
    out_lines = (proc.stdout or "").strip().splitlines()
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr)
        raise SystemExit("test_synthetic.py failed; see output above")
    console_frame(reel, "test_synthetic.py output", "the data / mask / metric pipeline on the synthetic scan, exit code 0",
                  [ln for ln in out_lines if not ln.startswith("scratch dir")][:14], 9.0, fs=17)

    root = H.find_dtu_root((os.path.join(synth, "fake_dtu"),))
    data = H.load_scan(root, 3, H.select_views(10), 3)
    cfg = {**H.DEFAULT_CFG, "scans": [3], "seeds": [0, 1], "radius_fracs": [0.15], "fills": ["gray", "noise"]}
    masks = H.build_masks(data, 0, 0.15, cfg)
    views_grid(reel, data, "Synthetic scan 3: the 10 input views", "select_views(10) over a 49-camera arc, load_scan reads cameras, depth and images", 4.5)
    sphere_frames(reel, data, masks, "Synthetic scan 3: sphere on the GT surface")
    views_grid(reel, data, "Synthetic scan 3: the hidden region in all 10 views",
               "build_masks projects one 3D sphere with the GT depth, so the hole is the same region in every view", 5.0, hide=masks["hide"])
    fills_frame(reel, data, masks, "Synthetic scan 3: fills", 6.0)
    labels_frame(reel, data, masks, "Synthetic scan 3: labels", 5.0)
    consistency_frame(reel, data, masks, "Synthetic scan 3: geometry check", 7.0)

    # ------------------------------------------------------------------ 3. models + metrics, live
    reel.mark("models and metrics, live")
    card(reel, "Stage 2: models and metrics, live",
         ["run_benchmark() gives every model the filled views and asks for depth and confidence,",
          "then compute_metrics() scores them against the U and S labels.",
          "",
          "Two reference models stand in for a network here:",
          "   oracle   correct depth, confidence drops exactly where the input was hidden",
          "   blind    correct depth where visible, a smooth guess inside, confidence flat everywhere",
          "",
          "A benchmark that cannot tell these two apart is broken. This one must give",
          "oracle AUROC 1 / CUR 0 and blind AUROC 0.5 / CUR about 0.9."], 10.0, tag=LIVE_TAG, fs=27)
    fr = pd.read_csv(os.path.join(synth, "fake_results.csv"))
    for m, lab in [("oracle", "oracle (reference model)"), ("blind", "blind (reference model)")]:
        png = os.path.join(synth, "fake_examples", f"{m}_scan3_r0.15.png")
        r = fr[(fr.model == m) & (fr.seed == 0) & (fr.fill == "gray")]
        composite(reel, png, lab, "synthetic scan 3", r.iloc[0] if len(r) else None, "Live: reference model output", 6.0, SYNTH_TAG)
    summ = H.summarize(os.path.join(synth, "fake_results.csv"))
    lines = ["summarize(fake_results.csv):", "",
             f"{'model':8s} {'fill':6s} {'AUROC':>22s} {'CUR':>22s} {'AbsRel S':>22s}"]
    for _, r in summ.iterrows():
        lines.append(f"{r['model']:8s} {r['fill']:6s} {r['auroc']:>22s} {r['cur']:>22s} {r['absrel_S']:>22s}")
    lines += ["", "AbsRel S is 0 after median scaling: the metric lives in the model's own scale.",
              "Intervals collapse to a point because the synthetic test has a single scan."]
    console_frame(reel, "Live: the metric table", "oracle separates perfectly, blind is at chance, as it must", lines, 10.0, fs=19)

    # reprojection-consistency detector, live, GT cameras
    h, w = data["depths"].shape[1:]
    score_gt = H.reproj_support_score(data["depths"], data["K_dep"], data["E"])
    hid_dep = masks["hide"].reshape(len(data["imgs"]), h, 4, w, 4).mean((2, 4)) > 0.5
    import cv2
    guess = cv2.GaussianBlur(np.where(data["depths"] > 0, data["depths"], 3.0).transpose(1, 2, 0), (0, 0), 6).transpose(2, 0, 1)
    blind_dep = np.where(hid_dep, guess * 1.15, data["depths"]).astype(np.float32)
    score_blind = H.reproj_support_score(blind_dep, data["K_dep"], data["E"])
    rs, ra = float(score_gt[masks["S"]].mean()), H.auroc(score_blind[masks["S"]], score_blind[masks["U"]])
    fig = reel.fig()
    header(fig, "Live: the reprojection-consistency detector", "reproj_support_score: share of other views whose own depth explains each point (occlusion-aware, 3% tolerance)", LIVE_TAG)
    for j, (img, t) in enumerate([(score_gt[0], "score on the true depth"), (score_blind[0], "score on the blind guess")]):
        ax = fig.add_axes([0.04 + j * 0.47, 0.30, 0.44, 0.52])
        ax.imshow(img, cmap="magma", vmin=0, vmax=1)
        ax.contour(masks["U"][0], levels=[0.5], colors=[ACC], linewidths=1.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=22, pad=6)
    fig.text(0.04, 0.20, f"mean score on S, true depth = {rs:.3f}", fontsize=28)
    fig.text(0.52, 0.20, f"AUROC S vs U, blind guess = {ra:.3f}", fontsize=28, color=ACC2)
    fig.text(0.04, 0.10, "A smooth, cross-view coherent imagination is barely flagged by geometry alone. That is the expected weakness of", fontsize=21, color=MUTED)
    fig.text(0.04, 0.05, "this training-free baseline and the reason the paper's method draws disagreement from a generative head. GPU run pending (P3-reproj).", fontsize=21, color=MUTED)
    reel.add(fig, 10.0)

    # ------------------------------------------------------------------ 4. recorded runs
    reel.mark("recorded runs, DTU")
    figure_frame(reel, os.path.join(FIG, "fig2_pipeline.png"), "Stage 3: the same protocol on real data",
                 "paper figure 2: DTU (18 scans of the MVSNet split) and CO3Dv2 (18 phone-captured scenes), 10 views each", 8.0,
                 caption="runs P1-full and P2-full (DTU), P2-co3d and P2-co3d-b (CO3Dv2), P2-stress, P3-pv-smoke, P2-smoke-b: Kaggle T4, fp16, 2026-10-28")
    card(reel, "What a real run produces", [
        "For every (model, scene, seed, radius, fill): one CSV row with AUROC, CUR, CUR-err, AbsRel on U and S,",
        "the same pixels under clean input as a control, and the confidence ratio inside the hidden region.",
        "",
        "Plus a composite PNG for seed 0, the configuration JSON, the consistency check and the kernel log.",
        "All of it is in results/ and nothing below is typed in."], 8.0, tag=REPLAY_TAG, fs=26)
    t1 = pd.read_csv(os.path.join(RES, "table1_all_models.csv"))
    t3 = pd.read_csv(os.path.join(RES, "table3_all_models.csv"))
    dtu_dirs = [os.path.join(RES, "P1-full", "ex_P1-full"), os.path.join(RES, "P2-full", "ex_P2-full")]
    for model in ORDER:
        for scan in [17, 106]:
            png = find_png(dtu_dirs, model, scan, 0.18)
            if png is None:
                continue
            r = t1[(t1.model == model) & (t1.scan == scan) & (t1.seed == 0) & (t1.radius_frac == 0.18) & (t1.fill == "gray")]
            composite(reel, png, MODEL_LABEL[model], f"DTU scan {scan}", r.iloc[0] if len(r) else None, "DTU: recorded model output", 3.2, REPLAY_TAG)

    reel.mark("recorded runs, inpainted fill")
    sdir = os.path.join(RES, "P2-stress", "ex_P2-stress_inpaint10")
    sdf = pd.read_csv(os.path.join(RES, "P2-stress", "P2-stress_inpaint10.csv"))
    for model in ["vggt", "pi3x", "da3"]:
        png = find_png([sdir], model, 17, 0.18)
        if png is None:
            continue
        r = sdf[(sdf.model == model) & (sdf.scan == 17) & (sdf.seed == 0) & (sdf.radius_frac == 0.18)]
        composite(reel, png, MODEL_LABEL[model], "DTU scan 17, inpainted fill (no visible seam)", r.iloc[0] if len(r) else None,
                  "DTU stress test: inpainted fill", 3.5, REPLAY_TAG)

    reel.mark("recorded runs, CO3Dv2")
    co_dirs = [os.path.join(RES, "P2-co3d", "ex_P2-co3d"), os.path.join(RES, "P2-co3d-b", "ex_P2-co3d-b")]
    for model in ORDER:
        for sc in ["toytrain_399_51323_100753", "apple_189_20393_38136"]:
            png = find_png(co_dirs, model, sc, 0.18)
            if png is None:
                continue
            r = t3[(t3.model == model) & (t3.scan == sc) & (t3.seed == 0) & (t3.radius_frac == 0.18) & (t3.fill == "gray")]
            composite(reel, png, MODEL_LABEL[model], f"CO3Dv2 {sc}", r.iloc[0] if len(r) else None, "CO3Dv2: recorded model output", 3.2, REPLAY_TAG)

    # ------------------------------------------------------------------ 5. numbers
    reel.mark("the numbers")
    labels = [MODEL_LABEL[m] for m in ORDER]
    bar_chart_frames(reel, "Stage 4: leakage (CUR, blur + noise fills)", "mean with 95% cluster bootstrap CI over 18 scenes, recomputed from the CSVs now; lower is better",
                     labels, {"DTU (18 scans)": [pooled(t1, m, "cur") for m in ORDER], "CO3Dv2 (18 scenes)": [pooled(t3, m, "cur") for m in ORDER]}, hold=7.0)
    bar_chart_frames(reel, "Separability (AUROC, blur + noise fills)", "0.5 is chance; higher is better. Only Pi3 keeps its place between the lab rig and phone captures",
                     labels, {"DTU (18 scans)": [pooled(t1, m, "auroc") for m in ORDER], "CO3Dv2 (18 scenes)": [pooled(t3, m, "auroc") for m in ORDER]},
                     ylim=(0.4, 1.0), ylabel="AUROC", hold=6.0)
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
    bar_chart_frames(reel, "Stress tests on DTU (CUR)", "18 scans x 2 seeds, r 0.18; reference = blur, 10 spread views. Inpainting leaks more; fewer views do not help",
                     tlabels, series, hold=7.0)
    # training-free detectors against native confidence
    pv = pd.read_csv(os.path.join(RES, "P3-pv-smoke", "P3-pv-smoke.csv"))
    sb = pd.read_csv(os.path.join(RES, "P2-smoke-b", "P2-smoke-b.csv"))
    det = {"native confidence": [ci_of(t1, "cur", model="vggt", seed=0, fill="blur", radius_frac=0.18),
                                 ci_of(t1, "cur", model="pi3x", seed=0, fill="blur", radius_frac=0.18),
                                 ci_of(sb, "cur", model="mapanything")],
           "alternative": [ci_of(pv, "cur", model="vggt_pv4"), ci_of(pv, "cur", model="pi3x_pv4"), ci_of(sb, "cur", model="mapanything_mv")]}
    bar_chart_frames(reel, "Training-free detectors leak more (CUR)",
                     "alternative: variance over K=4 perturbed runs (VGGT, Pi3X; 18 scans) or multi-view consistency confidence (MapAnything)",
                     ["VGGT-1B", "Pi3X", "MapAnything"], det, hold=8.0)
    print("detector CUR (native vs alternative):", {k: [tuple(round(x, 3) for x in v) for v in vs] for k, vs in det.items()})

    # ------------------------------------------------------------------ 6. paper, method, pending
    reel.mark("paper and method")
    figure_frame(reel, os.path.join(FIG, "fig4_dtu_vs_co3d.png"), "Stage 5: the paper", "figure 4: the same metrics, DTU against CO3Dv2, per model", 7.0,
                 caption="make_figures.py rebuilds every chart figure from results/*.csv; figures 1 and 2 additionally need the raw data")
    figure_frame(reel, os.path.join(FIG, "fig8_cheap_detectors.png"), "The paper: cheap detectors", "figure 8: native confidence against perturbation variance and consistency confidence", 6.0)
    figure_frame(reel, os.path.join(FIG, "fig9_method.png"), "Proposed method: observe, imagine, abstain",
                 "a small support head on the frozen multi-view features predicts P(imagined) per pixel; trained label-free on the backbone's own prediction", 10.0,
                 caption="design figure; its numbers are pending (run P3-oia, scripts/run_h100.py) and the pre-registered pass condition is in the paper")
    card(reel, "What is measured, and what is not yet", [
        "Measured (logged runs, CSV + config in results/):",
        "   five models x DTU (18 scans) x CO3Dv2 (18 scenes), three fills, 5 seeds, 2 radii;",
        "   stress tests: inpainted fill, 5 and 3 views, low parallax, distractor image;",
        "   two training-free detectors: perturbation variance, multi-view consistency.",
        "",
        "Written and CPU-tested, not yet run on a GPU: the reprojection detector (<model>_reproj) and the support head (<model>_oia),",
        "   with the GPU plan in scripts/run_h100.py (train, DTU, CO3Dv2, stress, baselines, ablations, stats).",
        "",
        "Pending and marked so in the paper: every support-head cell; VGGT-Omega (gated checkpoint).",
        "Future work: a generative head whose sample disagreement marks imagined regions; continual adaptation."], 11.0, fs=25)

    # ------------------------------------------------------------------ 7. artefacts
    reel.mark("artefacts")
    chain = [("python scripts/test_synthetic.py", "CPU self-test (what Stages 1 and 2 showed)"),
             ("kaggle_driver*.py + hallu3d.py", "GPU runs: one CSV row per (model, scene, seed, radius, fill)"),
             ("python scripts/make_figures.py", "paper/figures/*.png|pdf from the CSVs"),
             ("cd paper/tex && tectonic main.tex supp.tex", "paper and supplementary PDFs"),
             ("python scripts/make_site.py", "docs/explorer.html + CSV bundle; checks the site numbers"),
             ("python scripts/make_supp_archive.py", "anonymised code + results archive"),
             ("python scripts/make_demo.py / make_walkthrough.py", "the videos")]
    console_frame(reel, "Stage 6: from results to artefacts, the command chain", "the order in which the repository is rebuilt",
                  [f"{c:52s}{d}" for c, d in chain], 11.0, tag=None, fs=19)
    thumbs_frame(reel, sorted(glob.glob(os.path.join(FIG, "fig*.png"))), "The nine paper figures, as built by make_figures.py",
                 "all from results/*.csv except figures 1 and 2, which also need the raw views", 7.0)
    # read what exists on disk now
    def npages(p):
        try:
            from pypdf import PdfReader
            return f"{len(PdfReader(p).pages)} pages"
        except Exception:
            return f"{os.path.getsize(p) / 1e6:.1f} MB"
    arte = []
    for rel, kind in [("paper/observe_imagine_abstain_cvpr27.pdf", "paper"), ("paper/observe_imagine_abstain_cvpr27_supp.pdf", "supplementary"),
                      ("docs/index.html", "project page"), ("docs/explorer.html", "per-scene explorer"),
                      ("docs/assets/hallubench3d_results.zip", "CSV bundle"), ("paper/demo/demo.mp4", "demo video")]:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            arte.append(f"   {kind:20s} {rel}   (missing)"); continue
        if p.endswith(".pdf"):
            info = npages(p)
        elif p.endswith(".zip"):
            info = f"{len(zipfile.ZipFile(p).namelist())} files"
        else:
            b = os.path.getsize(p)
            info = f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"
        arte.append(f"   {kind:20s} {rel}   ({info})")
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "?"
    console_frame(reel, "Artefacts present in the repository at build time", f"read from disk now; git HEAD {head}", arte, 9.0, fs=19)
    card(reel, "Where to look", [
        "README.md          layout, reproduce commands, rules the work followed",
        "EXPERIMENTS.md     run registry: done, queued, known code notes",
        "LOG.md             dated log of every run, failure and correction",
        "results/           every CSV, config, consistency check, kernel log and composite",
        "paper/tex          main.tex, supp.tex, refs.bib (arXiv-verified), figures",
        "docs/              project page and per-scene explorer",
        "",
        "Code, benchmark and the full run log are released with the paper."], 10.0, fs=26)

    write_with_chapters(reel, args.out, frames)
    for t, name in reel.marks:
        print(f"{int(t // 60):d}:{int(t % 60):02d}  {name}")
    print(f"total {reel.t:.1f}s, {len(reel.items)} frames -> {args.out}")
    if args.frames is None:
        shutil.rmtree(frames, ignore_errors=True)


if __name__ == "__main__":
    main()
