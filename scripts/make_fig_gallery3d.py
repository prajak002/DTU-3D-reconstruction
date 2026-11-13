"""3D gallery (supplement): fused point clouds of static DTU captures and of a video with a walking person, as the
backbone predicts them, with the ground-truth class of every point and what the support head does.

Rows, chosen by fixed rules and not by eye:
  DTU      the --dtu-rows benchmark scans with the largest share of single-view pixels in the five-view dump
           (results/P20-dump, seed 0), VGGT
  video    the --video-rows ten-frame windows of TUM RGB-D walking_xyz with the largest share of moving pixels, VGGT
Columns: one input view; all predicted points in their colours with the input cameras; the points no other view
supports by the ground truth (DTU: single-view, red; video: moving, purple) over the rest in gray; the points
coloured by the support head; the points the head keeps at the benchmark's operating point (the threshold that keeps
90% of the supported pixels of that capture).

Geometry: the backbone's predicted depth (one median scale per capture) back-projected with the dataset's cameras
(DTU rig calibration, TUM motion capture), not the backbone's own cameras. Rendering: z-buffer point splat from one
virtual camera per row, placed by the same rule in every row.

  python scripts/make_fig_gallery3d.py --dtu-cams <dir with 000000NN_cam.txt> --tum-root <preprocessed TUM video root>"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
from make_fig_dynamic3d import draw_cams, look_at, splat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAG, OBS, MOV, INK, MUTED = "#D1495B", "#2A9D8F", "#8E44AD", "#1D2B36", "#5C6B77"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42})
HW = (760, 900)


def hexrgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def render_row(d, score, rgb, S, bad, Kd, Es, bad_col, r=2):
    """d (N,h,w) predicted depth in the cameras' scale, score (N,h,w), rgb (N,h,w,3), S supported, bad the ground-truth
    class shown in column 3. Returns the four renders and the row's numbers."""
    n = len(d)
    tau = float(np.quantile(score[S], 0.10))
    far = np.quantile(d[d > 0], 0.97)
    P, C, Sc, B = [], [], [], []
    for i in range(n):
        ok = (d[i] > 0) & (d[i] < far)
        P.append(H.backproject(d[i], Kd[i], Es[i])[ok]); C.append(rgb[i][ok]); Sc.append(score[i][ok]); B.append(bad[i][ok])
    P, C, Sc, B = np.concatenate(P), np.concatenate(C), np.concatenate(Sc), np.concatenate(B)
    Rcw = Es[n // 2][:3, :3].T; Cc = -Rcw @ Es[n // 2][:3, 3]
    centre = np.median(P, 0)
    dist = np.linalg.norm(centre - Cc)
    right, down, fwd = Rcw[:, 0], Rcw[:, 1], Rcw[:, 2]
    Ev = look_at(Cc - 0.30 * dist * fwd - 0.22 * dist * down + 0.28 * dist * right, centre, -down)
    # focal length from the cloud's extent, so that every row fills its panel in the same way
    Pc = P @ Ev[:3, :3].T + Ev[:3, 3]
    ok = Pc[:, 2] > 0.05
    ext = max(np.quantile(np.abs(Pc[ok, 0] / Pc[ok, 2]), 0.985) / (HW[1] / 2), np.quantile(np.abs(Pc[ok, 1] / Pc[ok, 2]), 0.985) / (HW[0] / 2))
    f = 0.94 / ext
    Kv = np.array([[f, 0, HW[1] / 2], [0, f, HW[0] / 2], [0, 0, 1]])
    lo, hi = np.quantile(score, 0.02), np.quantile(score, 0.98)
    Cs = (matplotlib.colormaps["viridis"](np.clip((Sc - lo) / (hi - lo + 1e-9), 0, 1))[:, :3] * 255).astype(np.uint8)
    gray = (0.35 * C.mean(1, keepdims=True) + 0.65 * 235).astype(np.uint8).repeat(3, 1)
    Cb = np.where(B[:, None], hexrgb(bad_col)[None], gray)
    keep = Sc >= tau
    hw_d = d.shape[1:]
    cams = lambda im: draw_cams(im, Es, Kd[0], hw_d, Ev, Kv, size=0.06 * dist)
    ims = [cams(splat(P, C, Ev, Kv, HW, r)), splat(P, Cb, Ev, Kv, HW, r), splat(P, Cs, Ev, Kv, HW, r),
           cams(splat(P[keep], C[keep], Ev, Kv, HW, r))]
    return ims, dict(points=len(P), bad_share=float(B.mean()), kept=float(keep.mean()),
                     bad_kept=float(keep[B].mean()) if B.any() else float("nan"))


def dtu_rows(a):
    lab_dir = os.path.join(a.dtu_dump, "labels")
    scans = sorted(((float(np.load(p)["U"].mean()), os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(lab_dir, "*__s0__v5.npz"))),
                   reverse=True)[:a.dtu_rows]
    views = H.vis_views(5, 0)
    out = []
    for _, key in scans:
        L = np.load(os.path.join(lab_dir, key + ".npz"))
        hw = L["gt"].shape[1:]
        Ks, Es = [], []
        for v in views:
            E, K = H.read_cam(os.path.join(a.dtu_cams, f"{v:08d}_cam.txt"))
            Ks.append(H._scale_K_to(K, hw)[0]); Es.append(E)
        Ks, Es = np.stack(Ks), np.stack(Es)
        # the cameras must reproduce the dump's own support labels: supported pixels reproject onto ground-truth depth
        seen, _, _ = H.support_matrix(L["gt"], Ks, Es)
        agree = float((seen.sum(1) >= 2)[L["S"]].mean())
        assert agree > 0.9, f"cameras do not match the dump ({key}: {agree:.2f})"
        D = np.load(os.path.join(a.dtu_dump, a.base, key + ".npz"))["depth"].astype(np.float64)      # already median-scaled
        sc = np.load(os.path.join(a.dtu_dump, a.base + "_oiavisself", key + ".npz"))["conf"].astype(np.float32)
        ims, st = render_row(D, sc, L["rgb"], L["S"], L["U"], Ks, Es, IMAG, r=3)
        out.append((f"DTU scan {key.split('__')[0]}", L["rgb"][len(views) // 2], ims, st, "single-view", agree))
    return out


def video_rows(a):
    scene = "freiburg3_walking_xyz"
    df = pd.read_csv(os.path.join(ROOT, "results", "P23-dynamic", a.base, "P23-dynamic.csv"))
    x = df[(df.scan == scene) & (df.model == a.base)].set_index("window")
    n = int(x.n_views.iloc[0])
    m = np.load(os.path.join(a.tum_root, scene, "meta.npz"))
    out = []
    for w in sorted(x.frac_M.sort_values(ascending=False).index[:a.video_rows]):
        views = list(range(w * n, (w + 1) * n))
        L = np.load(os.path.join(a.video_dump, "labels", f"{scene}__w{w:03d}.npz"))
        assert np.allclose(L["gt"], m["depth"][views], atol=1e-3)
        d = np.load(os.path.join(a.video_dump, a.base, f"{scene}__w{w:03d}.npz"))["depth"].astype(np.float64)
        d = d * H.median_scale(d, L["gt"].astype(np.float64), L["S"])
        sc = np.load(os.path.join(a.video_dump, a.base + "_oiavisself", f"{scene}__w{w:03d}.npz"))["conf"].astype(np.float32)
        ims, st = render_row(d, sc, L["rgb"], L["S"], L["M"], m["K_dep"][views].astype(np.float64), m["E"][views].astype(np.float64), MOV)
        out.append((f"video, window {w}", L["rgb"][n // 2], ims, st, "moving", 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu-cams", required=True)
    ap.add_argument("--tum-root", required=True)
    ap.add_argument("--dtu-dump", default=os.path.join(ROOT, "results", "P20-dump", "dtu"))
    ap.add_argument("--video-dump", default=os.path.join(ROOT, "results", "P23-dynamic", "dump"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--dtu-rows", type=int, default=3)
    ap.add_argument("--video-rows", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig_gallery3d"))
    a = ap.parse_args()
    rows = dtu_rows(a) + video_rows(a)
    wr = [0.62, 1, 1, 1, 1]
    fig = plt.figure(figsize=(6.875, 1.33 * len(rows) + 0.22))
    gs = fig.add_gridspec(len(rows), 5, width_ratios=wr, left=0.03, right=0.997, top=1 - 0.22 / (1.33 * len(rows) + 0.22),
                          bottom=0.004, wspace=0.03, hspace=0.05)
    titles = [("one input view", INK), ("predicted points, all views", INK), ("unsupported by ground truth", IMAG),
              ("coloured by the support head", OBS), ("points the head keeps", OBS)]
    for r, (name, frame, ims, st, kind, _) in enumerate(rows):
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(frame if c == 0 else ims[c - 1]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#C9D1D9"); sp.set_linewidth(0.5)
            if r == 0:
                ax.set_title(titles[c][0], fontsize=6.4, weight="bold", pad=2.5, color=titles[c][1])
            if c == 0:
                ax.set_ylabel(name, fontsize=6.0, color=INK, weight="bold", labelpad=2)
            note = {2: f"{kind}: {100 * st['bad_share']:.0f}% of the points",
                    4: f"keeps {100 * st['kept']:.0f}% of all, {100 * st['bad_kept']:.0f}% of {kind}"}.get(c)
            if note:
                ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=5.2, color="white", weight="bold", va="bottom",
                        bbox=dict(fc="black", ec="none", alpha=0.6, pad=1.2))
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()})
    fig.savefig(a.out + ".pdf", dpi=330); fig.savefig(a.out + ".png", dpi=330)
    print("->", a.out)


if __name__ == "__main__":
    main()
