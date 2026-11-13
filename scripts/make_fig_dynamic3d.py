"""3D visual of a moving scene (sixteenth pass): the fused point cloud of one ten-frame window of a TUM RGB-D video with
a walking person, as the backbone predicts it, and what the support head does to it.

Geometry: the backbone's predicted depth of every frame of the window (per-pixel dump of dynamic_eval.py, one median
scale per window against the sensor depth on supported pixels), back-projected with the dataset's motion-capture
cameras. The cameras are the dataset's, not the backbone's own: the dump stores depth and scores only. A person who
walks during the window is reconstructed once per frame, so the fused cloud shows several copies.

Panels: input frames; (a) all predicted points in their colours, with the ten cameras; (b) the same points coloured by
the support head; (c) the points the head keeps, at the benchmark's operating point (the threshold that keeps 90% of
the supported static pixels of the window). The window is the one with the largest share of moving pixels (fixed
rule). Rendering is a plain z-buffer point splat from one fixed virtual camera.

  python scripts/make_fig_dynamic3d.py --root <preprocessed TUM video root> [--base vggt] [--out paper/figures/fig_dynamic3d]"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAG, OBS, MOV, INK = "#D1495B", "#2A9D8F", "#8E44AD", "#1D2B36"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42})


def look_at(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)          # camera axes: x right, y down, z forward
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ eye
    return E


def splat(P, C, E, K, hw, r=2, bg=255):
    """Painter's z-buffer splat: far points first, each point a (2r+1)^2 square."""
    h, w = hw
    Pc = P @ E[:3, :3].T + E[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.05
    u = np.round(K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
    ok &= (u >= r) & (u < w - r) & (v >= r) & (v < h - r)
    o = np.argsort(-z[ok])
    u, v, c = u[ok][o], v[ok][o], C[ok][o]
    img = np.full((h, w, 3), bg, np.uint8)
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            img[v + dv, u + du] = c
    return img


def draw_cams(img, Es, Kd, hw_d, E, K, size=0.12, color=(29, 43, 54)):
    """Frustums of the input cameras, drawn over the render."""
    h, w = hw_d
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    for Ec in Es:
        Rcw = Ec[:3, :3].T; C = -Rcw @ Ec[:3, 3]
        dirs = np.stack([(corners[:, 0] - Kd[0, 2]) / Kd[0, 0], (corners[:, 1] - Kd[1, 2]) / Kd[1, 1], np.ones(4)], 1) * size
        pts = np.concatenate([C[None], dirs @ Rcw.T + C])
        pc = pts @ E[:3, :3].T + E[:3, 3]
        if (pc[:, 2] <= 0.05).any():
            continue
        uv = np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1).astype(int)
        for a_, b_ in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
            cv2.line(img, tuple(uv[a_]), tuple(uv[b_]), (255, 255, 255), 5, cv2.LINE_AA)
        for a_, b_ in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
            cv2.line(img, tuple(uv[a_]), tuple(uv[b_]), color, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="preprocessed TUM video root (scripts/tum.py --n-frames 200)")
    ap.add_argument("--dump", default=os.path.join(ROOT, "results", "P23-dynamic", "dump"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--scene", default="freiburg3_walking_xyz")
    ap.add_argument("--window", type=int, default=-1, help="-1: the window with the largest share of moving pixels")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig_dynamic3d"))
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "results", "P23-dynamic", a.base, "P23-dynamic.csv"))
    x = df[(df.scan == a.scene) & (df.model == a.base)].set_index("window")
    w = int(x.frac_M.idxmax()) if a.window < 0 else a.window
    n = int(x.n_views.iloc[0])
    m = np.load(os.path.join(a.root, a.scene, "meta.npz"))
    views = list(range(w * n, (w + 1) * n))
    Kd, Es = m["K_dep"][views].astype(np.float64), m["E"][views].astype(np.float64)
    L = np.load(os.path.join(a.dump, "labels", f"{a.scene}__w{w:03d}.npz"))
    assert np.allclose(L["gt"], m["depth"][views], atol=1e-3), "the local preprocessing does not match the dumped window"
    Dn = np.load(os.path.join(a.dump, a.base, f"{a.scene}__w{w:03d}.npz"))
    Dh = np.load(os.path.join(a.dump, a.base + "_oiavisself", f"{a.scene}__w{w:03d}.npz"))
    d = Dn["depth"].astype(np.float64)
    d = d * H.median_scale(d, L["gt"].astype(np.float64), L["S"])
    score = Dh["conf"].astype(np.float32)
    tau = float(np.quantile(score[L["S"]], 0.10))          # keeps 90% of the supported static pixels (the benchmark's tau)
    far = np.quantile(d, 0.97)
    P, C, Sc, Mv = [], [], [], []
    for i in range(n):
        ok = (d[i] > 0) & (d[i] < far)
        P.append(H.backproject(d[i], Kd[i], Es[i])[ok]); C.append(L["rgb"][i][ok]); Sc.append(score[i][ok]); Mv.append(L["M"][i][ok])
    P, C, Sc, Mv = np.concatenate(P), np.concatenate(C), np.concatenate(Sc), np.concatenate(Mv)
    # one fixed virtual camera: behind, above and to the side of the middle input camera, looking at the scene's centre
    Rcw = Es[n // 2][:3, :3].T; Cc = -Rcw @ Es[n // 2][:3, 3]
    centre = np.median(P, 0)
    dist = np.linalg.norm(centre - Cc)
    right, down, fwd = Rcw[:, 0], Rcw[:, 1], Rcw[:, 2]
    eye = Cc - 0.30 * dist * fwd - 0.22 * dist * down + 0.28 * dist * right
    Ev = look_at(eye, centre, -down)
    hw = (1000, 1060)
    Kv = np.array([[760.0, 0, hw[1] / 2], [0, 760.0, hw[0] / 2], [0, 0, 1]])
    lo, hi = np.quantile(score, 0.02), np.quantile(score, 0.98)
    cm = matplotlib.colormaps["viridis"]
    Cs = (cm(np.clip((Sc - lo) / (hi - lo + 1e-9), 0, 1))[:, :3] * 255).astype(np.uint8)
    keep = Sc >= tau
    r_all = draw_cams(splat(P, C, Ev, Kv, hw), Es, Kd[0], L["gt"].shape[1:], Ev, Kv)
    r_sc = splat(P, Cs, Ev, Kv, hw)
    r_keep = draw_cams(splat(P[keep], C[keep], Ev, Kv, hw), Es, Kd[0], L["gt"].shape[1:], Ev, Kv)
    share_mv = float(Mv.mean()); kept_mv = float(keep[Mv].mean()) if Mv.any() else float("nan")
    kept_all = float(keep.mean())

    nz = np.argwhere((r_all < 250).any(2))               # one crop for the three renders: the extent of the full cloud
    (y0, x0), (y1, x1) = np.maximum(nz.min(0) - 25, 0), nz.max(0) + 25
    r_all, r_sc, r_keep = (r[y0:y1, x0:x1] for r in (r_all, r_sc, r_keep))
    pw = 6.875 * 0.992 / 3.36 * 0.98                       # width of one render panel in inches
    fig = plt.figure(figsize=(6.875, pw * (y1 - y0) / (x1 - x0) / 0.90 + 0.02))
    gs = fig.add_gridspec(4, 4, width_ratios=[0.36, 1, 1, 1], left=0.004, right=0.996, top=0.91, bottom=0.01, wspace=0.03, hspace=0.06)
    for k, i in enumerate(np.round(np.linspace(0, n - 1, 4)).astype(int)):
        ax = fig.add_subplot(gs[k, 0]); ax.imshow(L["rgb"][i]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#C9D1D9"); sp.set_linewidth(0.5)
        if k == 0:
            ax.set_title(f"{n} video frames", fontsize=6.6, weight="bold", pad=2.5, color=INK)
    titles = [("(a) predicted points, all frames", INK), ("(b) coloured by the support head", OBS),
              ("(c) points the head keeps", OBS)]
    for k, (im, (t, col)) in enumerate(zip((r_all, r_sc, r_keep), titles)):
        ax = fig.add_subplot(gs[:, k + 1]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#C9D1D9"); sp.set_linewidth(0.6)
        ax.set_title(t, fontsize=6.8, weight="bold", pad=2.5, color=col)
    fig.savefig(a.out + ".pdf", dpi=450); fig.savefig(a.out + ".png", dpi=450)
    print(f"-> {a.out} window {w}; points {len(P)}; moving share {share_mv:.3f}; kept overall {kept_all:.3f}; "
          f"moving points kept {kept_mv:.3f}; tau {tau:.4f}")


if __name__ == "__main__":
    main()
