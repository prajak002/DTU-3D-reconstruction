"""More visuals of the sixteenth pass (moving scenes, full video), all from logged rows and the per-pixel dump:

  fig_dynamic_models   one window of the walking video on all four backbones: native confidence and the support head
                       side by side, including the backbones where the head shows no gain. The window is the one
                       with the largest share of moving pixels (fixed rule), middle frame.
  fig_dynamic_stats    (a) AUROC per backbone and detector on the dynamic sequences, static part and moving surfaces,
                       with the static-control level of the head; (b) the full video as a timeline: AUROC per window
                       of the walking sequence and the share of moving pixels.
  dynamic_walking.mp4  the whole walking video, every frame of every window: frame, labels, native confidence, head.

  python scripts/make_fig_dynamic_extra.py [--dump results/P23-dynamic/dump] [--out paper/figures]"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAG, OBS, MOV, BASE, REPRO, INK, MUTED = "#D1495B", "#2A9D8F", "#8E44AD", "#8D99AE", "#5B8DB8", "#1D2B36", "#5C6B77"
NAMES = {"vggt": "VGGT", "pi3x": "Pi3X", "pi3": "Pi3", "da3": "DA3"}
SCENE = "freiburg3_walking_xyz"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def rows(base):
    return pd.read_csv(os.path.join(ROOT, "results", "P23-dynamic", base, "P23-dynamic.csv"))


def norm(c, f):
    lo, hi = np.quantile(c, 0.02), np.quantile(c, 0.98)
    return np.clip((c[f] - lo) / (hi - lo + 1e-9), 0, 1)


def frame_style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#C9D1D9"); sp.set_linewidth(0.6)


def outline(ax, M):
    if M.any():
        ax.contour(M.astype(float), levels=[0.5], colors="white", linewidths=1.4)
        ax.contour(M.astype(float), levels=[0.5], colors=MOV, linewidths=0.75)


def fig_models(dump, out, bases):
    x = rows(bases[0]); x = x[(x.scan == SCENE) & (x.model == bases[0])].set_index("window")
    w = int(x.frac_M.idxmax())
    L = np.load(os.path.join(dump, "labels", f"{SCENE}__w{w:03d}.npz"))
    f = L["rgb"].shape[0] // 2
    fig, ax = plt.subplots(2, len(bases) + 1, figsize=(6.875, 2.42))
    fig.subplots_adjust(left=0.075, right=0.997, top=0.92, bottom=0.01, wspace=0.03, hspace=0.05)
    ax[0, 0].imshow(L["rgb"][f]); ax[0, 0].set_title("video frame", fontsize=7, weight="bold", pad=2.5, color=INK)
    lab = (0.45 * L["rgb"][f] + 0.55 * 255).astype(np.float32)
    for k, col in (("S", OBS), ("U", IMAG), ("M", MOV)):
        c = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
        lab[L[k][f]] = 0.25 * L["rgb"][f][L[k][f]] + 0.75 * c
    ax[1, 0].imshow(lab.astype(np.uint8))
    ax[1, 0].text(0.03, 0.04, "labels: supported / single-view / moving", transform=ax[1, 0].transAxes, fontsize=5.0, color="white",
                  weight="bold", va="bottom", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.2))
    ax[0, 0].set_ylabel("native confidence", fontsize=6.6, color=IMAG, weight="bold")
    ax[1, 0].set_ylabel("support head (ours)", fontsize=6.6, color=OBS, weight="bold")
    for j, b in enumerate(bases, 1):
        d = rows(b); d = d[d.scan == SCENE]
        for r, m in enumerate((b, b + "_oiavisself")):
            c = np.load(os.path.join(dump, m, f"{SCENE}__w{w:03d}.npz"))["conf"].astype(np.float32)
            ax[r, j].imshow(norm(c, f), cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
            outline(ax[r, j], L["M"][f])
            t = d[(d.model == m) & (d.window == w)].iloc[0]
            ax[r, j].text(0.03, 0.04, f"static {t.auroc_SU:.2f}  moving {t.auroc_SM:.2f}", transform=ax[r, j].transAxes,
                          fontsize=5.4, color="white", weight="bold", va="bottom", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.2))
        ax[0, j].set_title(NAMES[b], fontsize=7, weight="bold", pad=2.5, color=INK)
    for a_ in ax.ravel():
        frame_style(a_)
    fig.savefig(out + ".pdf", dpi=400); fig.savefig(out + ".png", dpi=400)
    plt.close(fig)
    print("->", out, "window", w, "frame", f)


def fig_stats(out, bases):
    fig = plt.figure(figsize=(6.875, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], left=0.06, right=0.955, top=0.87, bottom=0.3, wspace=0.3)
    dets = (("", "native confidence", BASE), ("_reproj", "reprojection", REPRO), ("_oiavisself", "support head", OBS))
    for k, (col, title) in enumerate((("auroc_SU", "(a) static part: supported vs single-view"),
                                      ("auroc_SM", "(b) moving surfaces: supported vs moving"))):
        ax = fig.add_subplot(gs[k])
        for i, b in enumerate(bases):
            d = rows(b); dyn = d[d.kind == "dynamic"]
            for j, (suf, name, c) in enumerate(dets):
                per = dyn[dyn.model == b + suf].groupby("scan")[col].mean().dropna()
                xx = i + (j - 1) * 0.25
                ax.bar(xx, per.mean(), 0.23, color=c, label=name if i == 0 else None, zorder=2)
                ax.plot(np.full(len(per), xx), per.values, ".", ms=2.2, color=INK, alpha=0.55, zorder=3)
            if col == "auroc_SU":
                st = d[(d.kind == "static") & (d.model == b + "_oiavisself")].groupby("scan")[col].mean().mean()
                ax.plot([i + 0.13, i + 0.37], [st, st], color=INK, lw=1.0, zorder=4,
                        label="head, static videos" if i == 0 else None)
        ax.axhline(0.5, color=MUTED, lw=0.5, ls=":")
        ax.set_xticks(range(len(bases))); ax.set_xticklabels([NAMES[b] for b in bases], fontsize=6.4)
        ax.set_ylim(0.3, 1.0); ax.set_title(title, fontsize=6.6, weight="bold", pad=3)
        ax.tick_params(axis="y", labelsize=6); ax.tick_params(axis="x", length=0)
        if k == 0:
            ax.set_ylabel("AUROC (sequence mean)", fontsize=6.2)
            ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(1.15, -0.16), ncol=4, columnspacing=1.2,
                      handlelength=1.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax = fig.add_subplot(gs[2])
    d = rows("vggt"); d = d[d.scan == SCENE]
    nat, head = d[d.model == "vggt"].set_index("window"), d[d.model == "vggt_oiavisself"].set_index("window")
    ax2 = ax.twinx()
    ax2.bar(nat.index, 100 * nat.frac_M, 0.8, color=MOV, alpha=0.22, zorder=1)
    ax2.set_ylim(0, 60); ax2.set_yticks([0, 10, 20]); ax2.tick_params(axis="y", labelsize=5.6, colors=MOV, length=2)
    ax2.set_ylabel("moving pixels (%)", fontsize=5.8, color=MOV, y=0.2)
    ax.plot(nat.index, nat.auroc_SU, "-o", ms=2.2, lw=0.9, color=BASE, label="native confidence")
    ax.plot(head.index, head.auroc_SU, "-o", ms=2.2, lw=0.9, color=OBS, label="support head")
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=":")
    ax.set_ylim(0.2, 1.0); ax.set_xlim(-0.7, 19.7); ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_xlabel("window of the video (10 frames each)", fontsize=6, labelpad=1.5)
    ax.set_title("(c) the whole walking video, VGGT, static part", fontsize=6.6, weight="bold", pad=3)
    ax.tick_params(labelsize=6); ax.legend(fontsize=5.4, frameon=False, loc="lower right", borderaxespad=0.1)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    for a_ in (ax, ax2):
        a_.spines["top"].set_visible(False)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=400)
    plt.close(fig)
    print("->", out)


def video(dump, out, base="vggt", fps=6):
    """Every frame of every window, in order: frame | labels | native confidence | support head (2 x 2, labelled)."""
    cm = matplotlib.colormaps["viridis"]
    vw = None
    n = 0
    for w in range(100):
        lp = os.path.join(dump, "labels", f"{SCENE}__w{w:03d}.npz")
        if not os.path.exists(lp):
            break
        L = np.load(lp)
        cs = [np.load(os.path.join(dump, m, f"{SCENE}__w{w:03d}.npz"))["conf"].astype(np.float32) for m in (base, base + "_oiavisself")]
        for f in range(L["rgb"].shape[0]):
            rgb = L["rgb"][f]
            lab = (0.45 * rgb + 0.55 * 255).astype(np.float32)
            for k, col in (("S", OBS), ("U", IMAG), ("M", MOV)):
                c = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
                lab[L[k][f]] = 0.25 * rgb[L[k][f]] + 0.75 * c
            tiles = [rgb, lab.astype(np.uint8)] + [(cm(norm(c, f))[..., :3] * 255).astype(np.uint8) for c in cs]
            names = ["video frame", "labels: supported / single-view / moving", "native confidence", "support head (ours)"]
            big = []
            for t, name in zip(tiles, names):
                t = cv2.resize(np.ascontiguousarray(t), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cnt, _ = cv2.findContours(cv2.resize(L["M"][f].astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if name != names[1]:
                    cv2.drawContours(t, cnt, -1, (255, 255, 255), 3, cv2.LINE_AA); cv2.drawContours(t, cnt, -1, (142, 68, 173), 1, cv2.LINE_AA)
                cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(t, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                big.append(t)
            img = np.concatenate([np.concatenate(big[:2], 1), np.concatenate(big[2:], 1)], 0)
            if vw is None:
                vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); n += 1
    if vw is not None:
        vw.release()
    print("->", out, n, "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(ROOT, "results", "P23-dynamic", "dump"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures"))
    ap.add_argument("--video", default=os.path.join(ROOT, "docs", "assets", "dynamic_walking.mp4"))
    a = ap.parse_args()
    bases = [b for b in ("vggt", "pi3x", "pi3", "da3") if os.path.isdir(os.path.join(a.dump, b + "_oiavisself"))]
    fig_models(a.dump, os.path.join(a.out, "fig_dynamic_models"), bases)
    fig_stats(os.path.join(a.out, "fig_dynamic_stats"), ("vggt", "pi3x", "pi3", "da3"))
    video(a.dump, a.video)


if __name__ == "__main__":
    main()
