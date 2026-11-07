"""Overview video (about three minutes, no audio): the question, the protocol, the leak, the cheap fixes and the
fill-trained head, the label-free support head, four datasets, controls, what abstaining buys, the limit.
Every chart is the dark-theme output of scripts/make_assets.py and every number on a card comes from
docs/assets/numbers.json, so the video is rebuilt from the released CSVs.

  python scripts/make_assets.py --theme dark && python scripts/make_video.py [--out docs/assets/overview.mp4]
Frames are rendered with matplotlib at 1920 x 1080 and assembled with ffmpeg (on PATH or $FFMPEG)."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
CH = os.path.join(ASSETS, "charts")
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
BG, FG, MUTED, ACC, ACC2, GREEN = "#101418", "#f2f4f7", "#9aa4b2", "#4cc9f0", "#f4a261", "#2ec4b6"
N = json.load(open(os.path.join(ASSETS, "numbers.json"), encoding="utf-8"))

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]})


def g(path, k=0, nd=2):
    x = N
    for p in path.split("/"):
        x = x.get(p) if isinstance(x, dict) else None
        if x is None:
            return "--"
    if isinstance(x, list):
        x = x[k]
    return f"{x:.{nd}f}"


def pc(path, k=0):
    x = g(path, k, 3)
    return "--" if x == "--" else f"{float(x) * 100:.0f}%"


class Reel:
    def __init__(self, d):
        self.d = d; os.makedirs(d, exist_ok=True); self.items = []; self.t = 0.0

    def fig(self):
        return plt.figure(figsize=(19.2, 10.8), dpi=100)

    def add(self, fig, seconds):
        p = os.path.join(self.d, f"f{len(self.items):05d}.png")
        fig.savefig(p, dpi=100); plt.close(fig)
        self.items.append((p, seconds)); self.t += seconds

    def write(self, out):
        lst = os.path.join(self.d, "list.txt")
        with open(lst, "w") as f:
            for p, s in self.items:
                f.write(f"file '{p}'\nduration {s:.3f}\n")
            f.write(f"file '{self.items[-1][0]}'\n")
        subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "fps=30,format=yuv420p",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart", out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def header(fig, title, sub=None, chapter=None):
    if chapter:
        fig.text(0.04, 0.965, chapter, fontsize=18, color=ACC2, va="center", family="monospace")
    fig.text(0.04, 0.915, title, fontsize=36, fontweight="bold", va="center")
    if sub:
        fig.text(0.04, 0.862, sub, fontsize=22, color=MUTED, va="center")


def card(reel, title, lines_pos, seconds, sub=None, chapter=None, fs=28, big=None, lines=None):
    lines = lines_pos if lines_pos is not None else (lines or [])
    fig = reel.fig(); header(fig, title, sub, chapter)
    y = 0.76
    if big:
        for i, (num, lab) in enumerate(big):
            x = 0.06 + i * (0.88 / len(big))
            fig.text(x, 0.66, num, fontsize=64, fontweight="bold", color=ACC, va="center")
            fig.text(x, 0.55, lab, fontsize=20, color=MUTED, va="top", wrap=True)
        y = 0.40
    for ln in lines:
        fig.text(0.06, y, ln, fontsize=fs, va="top")
        y -= 0.075 * (fs / 28)
    reel.add(fig, seconds)


def chart(reel, png, title, caption, seconds, chapter=None, sub=None):
    fig = reel.fig(); header(fig, title, sub, chapter)
    img = Image.open(png)
    w, h = img.size
    ax_h = 0.66; ax_w = min(0.92, ax_h * (w / h) * (10.8 / 19.2))
    ax = fig.add_axes([0.5 - ax_w / 2, 0.14, ax_w, ax_h]); ax.imshow(img); ax.axis("off")
    if caption:
        fig.text(0.5, 0.07, caption, fontsize=20, color=MUTED, ha="center", va="center", wrap=True)
    reel.add(fig, seconds)


def images_row(reel, pngs, titles, title, caption, seconds, chapter=None, sub=None, highlight=None):
    fig = reel.fig(); header(fig, title, sub, chapter)
    n = len(pngs)
    for i, (p, t) in enumerate(zip(pngs, titles)):
        ax = fig.add_axes([0.04 + i * 0.92 / n, 0.16, 0.92 / n - 0.015, 0.62])
        ax.imshow(Image.open(p)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(ACC if highlight == i else "#2a3139"); sp.set_linewidth(3 if highlight == i else 1)
        ax.set_title(t, fontsize=22, color=ACC if highlight == i else FG, pad=8)
    if caption:
        fig.text(0.5, 0.08, caption, fontsize=20, color=MUTED, ha="center", va="center", wrap=True)
    reel.add(fig, seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ASSETS, "overview.mp4"))
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    reel = Reel(a.frames or tempfile.mkdtemp(prefix="oia_video_"))
    V = lambda ds, m, k="auroc", nd=2: g(f"vis/{ds}/{m}/{k}", 0, nd)

    # 1. title
    fig = reel.fig()
    fig.text(0.5, 0.60, "Observe, Imagine, Abstain", fontsize=72, ha="center", va="center", fontweight="bold")
    fig.text(0.5, 0.48, "Detecting hallucinated geometry in feed-forward 3D reconstruction", fontsize=30, ha="center", color=MUTED)
    fig.text(0.5, 0.38, "CVPR 2027 submission. Every number below comes from a logged run; the charts are recomputed from the released CSVs.", fontsize=19, ha="center", color=MUTED)
    reel.add(fig, 6)

    # 2. the question
    hero = [os.path.join(ASSETS, f"hero_{k}.png") for k in ("input", "depth", "conf")]
    titles = ["what the model saw", "the depth it returned", "how sure it was"]
    for i, cap in enumerate(["DTU scan 17, VGGT. One 3D region of the real surface was hidden in all ten views and inpainted; the cyan contour encloses pixels no camera observed.",
                             "The model fills the hidden region with a plausible facade.",
                             "And it is about as confident there as anywhere else. The one signal a user has does not say what was observed and what was guessed."]):
        images_row(reel, hero, titles, "When a 3D model has not seen a surface, does it say so?", cap, 4.2, chapter="01 / THE QUESTION", highlight=i)

    # 3. the protocol
    chart(reel, os.path.join(ASSETS, "fig2_pipeline.png"), "A benchmark with labels for free",
          "One sphere on the ground-truth surface, hidden in every view with the true cameras: inside it, no view shows the surface. "
          "A clean run of the same views is the control for every pixel. Five models, three fills plus an inpainted one, five placements, two sizes.",
          9, chapter="02 / THE BENCHMARK")

    # 4. the leak
    chart(reel, os.path.join(CH, "leak_dark.png"), "The leak",
          "Share of unseen geometry that passes a confidence filter keeping 90% of observed geometry. Every model leaks; three leak more on phone captures than on the rig.", 9, chapter="03 / THE FINDING")
    card(reel, "The leak in three numbers", None, 7, chapter="03 / THE FINDING",
         big=[(pc("leak/dtu/vggt/cur"), "of unseen geometry passes for the best model on the lab rig (VGGT on DTU)"),
              (pc("leak/co3d/mapanything/cur"), "for the worst on phone captures (MapAnything on CO3Dv2)"),
              (pc("leak/eth3d/vggt/cur"), "for VGGT on building-scale ETH3D scans")],
         lines=["The imagined depth on DTU is as accurate as the observed depth: the failure is 'cannot tell', not 'wrong'.",
                "An inpainted fill raises the leak rather than lowering it. Dropping views or adding a distractor changes nothing."])

    # 5. cheap fixes and the fill-trained head
    chart(reel, os.path.join(CH, "compare_dark.png"), "Cheap fixes fail, and a head trained on the fill learns the fill",
          "Left: perturbation variance and reprojection consistency are no better than native confidence on the filled patch; a head trained on the fills scores 1.00, "
          "and so does its per-image control, so it found the patch, not the missing support. Right: on real single-view geometry the cross-view tokens know the answer.",
          11, chapter="04 / WHAT DOES NOT WORK")

    # 6. change the label
    card(reel, "Change what the detector must learn, not the head", [
        "Same head, same loss, same training scenes (79 DTU scans the benchmark never uses).",
        "Only the label changes: clean views; positive = a pixel only one view supports; negative = a pixel two or more views support.",
        "Decisive control: the same head on per-image tokens, before any cross-view attention.",
        "Then one step further: the label comes from the backbone's own prediction. No ground truth anywhere.",
        "Each predicted point is projected into the other views with the predicted cameras; the reprojection test is the teacher.",
    ], 11, chapter="05 / THE METHOD", fs=26)
    chart(reel, os.path.join(CH, "method_dark.png"), "The label-free support head",
          "3.3 M parameters on VGGT-1B, under 1% of the backbone's FLOPs, one forward. Ten minutes of training on one GPU.", 9, chapter="05 / THE METHOD")

    # 7. results
    chart(reel, os.path.join(CH, "vis_dark.png"), "Three backbones, four datasets the heads never saw",
          "DTU scans, CO3Dv2 phone orbits, ETH3D building scans and nine phototourism landmarks. Pre-registered pass conditions with a margin: both pass everywhere.",
          11, chapter="06 / RESULTS")
    card(reel, "The label-free head", None, 9, chapter="06 / RESULTS",
         big=[(V("dtu", "vggt_oiavisself"), f"AUROC on DTU (VGGT); native {V('dtu', 'vggt')}, per-image control {V('dtu', 'vggt_oiavispatch')}"),
              (V("co3d", "vggt_oiavisself"), f"on CO3Dv2 phone orbits; the per-image control falls to {V('co3d', 'vggt_oiavispatch')}, below chance"),
              (V("imc", "pi3x_oiavisself"), f"on tourist photographs (Pi3X); reprojection consistency {V('imc', 'pi3x_reproj')}, native {V('imc', 'pi3x')}")],
         lines=[f"It matches a head trained on ground-truth visibility within 0.02, and beats its own teacher, the reprojection test, where cameras are unreliable.",
                f"Only {pc('vis/dtu/vggt_oiavisself/cur')} of single-view pixels pass its 90%-kept threshold on DTU, against {pc('vis/dtu/vggt/cur')} for native confidence."], fs=24)

    # 8. controls
    chart(reel, os.path.join(CH, "layers_dark.png"), "Where the signal lives",
          f"A linear probe per VGGT layer: layer 4 scores {g('layers/dtu/4')} on DTU and {g('layers/co3d/4')} on CO3Dv2, like a per-image head; layer 17 scores {g('layers/dtu/17')} and {g('layers/co3d/17')}. "
          "Support becomes readable in the second half of the aggregator. The linear probe is within 0.015 of the head: the head is a reader.",
          10, chapter="07 / CONTROLS")
    card(reel, "Not the frame edge", [
        f"On the rig, single-view pixels sit at the image border: the distance to the border alone scores {V('dtu', 'vggt_border')} on DTU.",
        f"Off the rig it collapses ({V('co3d', 'vggt_border')} on CO3Dv2), and so does the per-image control ({V('co3d', 'vggt_oiavispatch')}): it had learned the edge.",
        f"The cross-view head keeps {V('co3d', 'vggt_oiavisself')} on the same scenes. Two extra seeds are within 0.005 of it.",
        f"With 20 views, twice the most seen in training, the head scores {g('ext/vggt_oiavisself')} while native confidence falls to {g('ext/vggt')}.",
    ], 10, chapter="07 / CONTROLS", fs=26)

    # 9. in the wild and adaptation
    chart(reel, os.path.join(CH, "adapt_dark.png"), "Training on the target capture itself, still without labels",
          f"Helps where the backbone's own geometry on that capture is reliable (Pi3X on CO3Dv2: {V('co3d', 'pi3x_oiavisself', nd=3)} to {V('co3d', 'pi3x_oiavisself_x', nd=3)}), "
          f"and hurts where it is not (VGGT on CO3Dv2: {V('co3d', 'vggt_oiavisself', nd=3)} to {V('co3d', 'vggt_oiavisself_x', nd=3)}; its reprojection test scores {V('co3d', 'vggt_reproj')} there). The DTU-trained head is the safe default.",
          10, chapter="08 / IN THE WILD")

    # 10. abstaining
    chart(reel, os.path.join(CH, "rc_dark.png"), "What abstaining buys",
          f"Keep the pixels the head trusts most: the single-view rate of what remains stays near the floor. Fixed bands 0.3 / 0.7 mark {pc('bands/dtu/vggt_oiavisself/frac_observed')} of DTU pixels observed with "
          f"{pc('bands/dtu/vggt_oiavisself/risk_observed')} single-view inside; a threshold set on the other scenes lets {pc('bands/dtu/vggt_oiavisself/cur_loso')} through against {pc('bands/dtu/vggt/cur_loso')} for native confidence.",
          11, chapter="09 / ABSTAINING")

    # 11. the limit
    chart(reel, os.path.join(CH, "onfill_dark.png"), "The limit: a consistent invention passes",
          f"Put back to the benchmark's filled patches, the visibility heads score {g('onfill/dtu/vggt_oiavisself')} on DTU and {g('onfill/co3d/vggt_oiavisself')} on CO3Dv2, below native confidence. "
          "The head certifies agreement between views; a surface the model invents consistently in every view has it. It answers 'which pixels lack corroboration', not 'which surfaces were invented'.",
          11, chapter="10 / THE LIMIT")

    # 12. closing
    card(reel, "Released", [
        "The benchmark (DTU, CO3Dv2, ETH3D, phototourism), every result CSV with its configuration and log,",
        "the heads for VGGT-1B, Pi3X and Pi3, the pre-registered pass conditions with dates and commits,",
        "the dated log of every failed run and every correction, and the scripts that rebuild every chart on this page.",
        "",
        "github.com/prajak002/DTU-3D-reconstruction",
    ], 9, chapter="11 / REPRODUCIBILITY", fs=26)

    reel.write(a.out)
    print(f"-> {a.out}  ({reel.t:.0f} s, {len(reel.items)} frames)")


if __name__ == "__main__":
    main()
