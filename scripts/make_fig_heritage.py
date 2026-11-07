"""Qualitative figure on photo collections without ground truth (P4-heritage): a folder of photographs of one
site (tourist photos of a monument, a museum object from a few angles), a backbone and a detector; for every input
view the figure shows the photograph, the backbone's native confidence, the detector's score and the three-way
observed / abstain / imagined map (thresholds 0.7 / 0.3 as in the paper). No labels are needed, which is the point:
the label-free head runs on any capture.

Usage: python scripts/make_fig_heritage.py <image dir or glob> --model vggt_oiavisself [--views 5] [--out paper/tex/figs/fig_heritage.png]
Images are taken in sorted order (every k-th to reach --views), resized to the benchmark resolution (longer side
518, divisible by 14). Prints the fraction of pixels in each band per view."""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402


def load_images(pattern, n):
    files = sorted(glob.glob(os.path.join(pattern, "*")) if os.path.isdir(pattern) else glob.glob(pattern))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(files) > n:
        files = [files[int(i)] for i in np.linspace(0, len(files) - 1, n)]
    imgs = []
    for f in files:
        im = cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        imgs.append(im)
    return files, imgs


def model_hw_for(im):
    h, w = im.shape[:2]
    if w >= h:
        return (int(round(518 * h / w / 14)) * 14, 518)
    return (518, int(round(518 * w / h / 14)) * 14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--model", default="vggt_oiavisself")
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("paper", "tex", "figs", "fig_heritage.png"))
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    files, raw = load_images(a.images, a.views)
    if not raw:
        raise SystemExit(f"no images under {a.images}")
    hw = model_hw_for(raw[0])
    imgs = np.stack([cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for im in raw]).astype(np.float32) / 255.0
    out_hw = (hw[0] // 2, hw[1] // 2)
    base = H.make_model(a.model.split("_")[0])
    d0, c0 = base.predict(imgs, hw, out_hw)
    det = H.make_model(a.model)
    d, s = det.predict(imgs, hw, out_hw)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = len(imgs)
    fig, ax = plt.subplots(N, 4, figsize=(4 * 3.2, N * 3.2 * out_hw[0] / out_hw[1] + 0.6), squeeze=False)
    cols = ["photograph", "native confidence", f"{a.model}: 1 - P(imagined)", "observed / abstain / imagined"]
    for i in range(N):
        ax[i, 0].imshow(cv2.resize(imgs[i], (out_hw[1], out_hw[0])))
        cn = (c0[i] - np.percentile(c0[i], 2)) / (np.percentile(c0[i], 98) - np.percentile(c0[i], 2) + 1e-9)
        ax[i, 1].imshow(np.clip(cn, 0, 1), cmap="viridis", vmin=0, vmax=1)
        ax[i, 2].imshow(s[i], cmap="viridis", vmin=0, vmax=1)
        band = np.zeros((*out_hw, 3), np.float32)
        band[s[i] > 0.7] = (0.20, 0.65, 0.30)
        band[(s[i] <= 0.7) & (s[i] >= 0.3)] = (0.95, 0.75, 0.15)
        band[s[i] < 0.3] = (0.85, 0.20, 0.20)
        ax[i, 3].imshow(0.4 * cv2.resize(imgs[i], (out_hw[1], out_hw[0])) + 0.6 * band)
        fo, fi = float((s[i] > 0.7).mean()), float((s[i] < 0.3).mean())
        print(f"{os.path.basename(files[i])}: observed {fo:.2f}, abstain {1 - fo - fi:.2f}, imagined {fi:.2f}")
        ax[i, 3].set_ylabel(os.path.basename(files[i])[:18], fontsize=7)
        for j in range(4):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=9)
    if a.title:
        fig.suptitle(a.title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("->", a.out)


if __name__ == "__main__":
    main()
