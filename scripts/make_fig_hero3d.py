"""Renders for the hero figure (Figure 1), at photograph resolution.

The same two captures, numbers and rule as scripts/make_fig_gallery3d.py (the DTU benchmark scan with the largest
single-view share in the five-view dump; the walking_xyz window with the largest moving share; VGGT's logged depth,
dataset cameras, the head's filter at the benchmark's tau). The only difference is display density: each predicted
depth value is drawn once for every photograph pixel it covers (nearest-neighbour, no depth is interpolated, so no
surface is added), which carries the photograph's detail into the cloud. Scores and labels are repeated the same way,
so every share printed on the figure is unchanged.

  python scripts/make_fig_hero3d.py --dtu-cams <dir> --tum-root <root>
Writes paper/figures/teaser3d_{all,keep}.png, teaser3d_video_{all,keep}.png and docs/assets/teaser3d.json."""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
import make_fig_gallery3d as G  # noqa: E402

ROOT = G.ROOT
VIEW = (0.06, 0.04, 0.05)      # the virtual camera sits next to the input cameras: the cloud reads like the photograph


def up(x, f):
    return np.repeat(np.repeat(x, f, axis=1), f, axis=2)


def up_K(K, f):
    K = np.array(K, np.float64).copy()
    K[..., 0, 0] *= f; K[..., 1, 1] *= f
    K[..., 0, 2] = f * K[..., 0, 2] + (f - 1) / 2.0
    K[..., 1, 2] = f * K[..., 1, 2] + (f - 1) / 2.0
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu-cams", required=True)
    ap.add_argument("--tum-root", required=True)
    ap.add_argument("--dtu-dump", default=os.path.join(ROOT, "results", "P20-dump", "dtu"))
    ap.add_argument("--dtu-hd", default=os.path.join(ROOT, "results", "P20-dump-hd", "dtu", "hd", "rgb"))
    ap.add_argument("--video-dump", default=os.path.join(ROOT, "results", "P23-dynamic", "dump"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures"))
    a = ap.parse_args()
    G.HW = (1180, 1400)
    T = {}
    # static capture: the benchmark scan with the largest single-view share
    lab_dir = os.path.join(a.dtu_dump, "labels")
    key = max(glob.glob(os.path.join(lab_dir, "*__s0__v5.npz")), key=lambda p: float(np.load(p)["U"].mean()))
    key = os.path.basename(key)[:-4]
    L = np.load(os.path.join(lab_dir, key + ".npz"))
    rgb = np.load(os.path.join(a.dtu_hd, key + ".npz"))["rgb"]
    f = rgb.shape[1] // L["gt"].shape[1]
    assert rgb.shape[1:3] == (L["gt"].shape[1] * f, L["gt"].shape[2] * f)
    views = H.vis_views(5, 0)
    Ks, Es = [], []
    for v in views:
        E, K = H.read_cam(os.path.join(a.dtu_cams, f"{v:08d}_cam.txt"))
        Ks.append(H._scale_K_to(K, L["gt"].shape[1:])[0]); Es.append(E)
    Ks, Es = np.stack(Ks), np.stack(Es)
    D = np.load(os.path.join(a.dtu_dump, a.base, key + ".npz"))["depth"].astype(np.float64)
    sc = np.load(os.path.join(a.dtu_dump, a.base + "_oiavisself", key + ".npz"))["conf"].astype(np.float32)
    ims, st = G.render_row(up(D, f), up(sc, f), rgb, up(L["S"], f), up(L["U"], f), up_K(Ks, f), Es, G.IMAG, r=1, view=VIEW)
    # in the left render the points without reference depth (the black backdrop of the rig) are drawn light gray, so
    # that they do not hide the object; they are the model's output all the same and are counted in every share
    gray = np.where(up(L["gt"] > 0, f)[..., None], rgb, np.uint8(205))
    ims_g, _ = G.render_row(up(D, f), up(sc, f), gray, up(L["S"], f), up(L["U"], f), up_K(Ks, f), Es, G.IMAG, r=1, view=VIEW)
    ims[4] = ims_g[4]
    st["no_reference"] = float((L["gt"] == 0).mean())
    Image.fromarray(ims[4]).save(os.path.join(a.out, "teaser3d_all.png"))
    for j, v in enumerate((len(views) // 2, 0, len(views) - 1)):        # the photographs themselves: middle view first
        Image.fromarray(rgb[v]).save(os.path.join(a.out, f"teaser3d_in_{j}.png"))
    Image.fromarray(ims[5]).save(os.path.join(a.out, "teaser3d_keep.png"))
    T.update(name=f"DTU scan {key.split('__')[0]}", base=a.base, **st)
    # video: the window with the largest share of moving pixels
    scene = "freiburg3_walking_xyz"
    df = pd.read_csv(os.path.join(ROOT, "results", "P23-dynamic", a.base, "P23-dynamic.csv"))
    x = df[(df.scan == scene) & (df.model == a.base)].set_index("window")
    w, n = int(x.frac_M.idxmax()), int(x.n_views.iloc[0])
    m = np.load(os.path.join(a.tum_root, scene, "meta.npz"))
    vw = list(range(w * n, (w + 1) * n))
    L = np.load(os.path.join(a.video_dump, "labels", f"{scene}__w{w:03d}.npz"))
    assert np.allclose(L["gt"], m["depth"][vw], atol=1e-3)
    f = int(m["depth_div"])
    rgb = np.stack([cv2.cvtColor(cv2.imread(os.path.join(a.tum_root, scene, "images", f"{int(m['frame_numbers'][v]):06d}.jpg")),
                                 cv2.COLOR_BGR2RGB)[:L["gt"].shape[1] * f, :L["gt"].shape[2] * f] for v in vw])
    d = np.load(os.path.join(a.video_dump, a.base, f"{scene}__w{w:03d}.npz"))["depth"].astype(np.float64)
    d = d * H.median_scale(d, L["gt"].astype(np.float64), L["S"])
    sc = np.load(os.path.join(a.video_dump, a.base + "_oiavisself", f"{scene}__w{w:03d}.npz"))["conf"].astype(np.float32)
    ims, st = G.render_row(up(d, f), up(sc, f), rgb, up(L["S"], f), up(L["M"], f), up_K(m["K_dep"][vw], f),
                           m["E"][vw].astype(np.float64), G.MOV, r=1, view=VIEW)
    Image.fromarray(ims[4]).save(os.path.join(a.out, "teaser3d_video_all.png"))
    for j, v in enumerate((n // 2, 0, n - 1)):
        Image.fromarray(rgb[v]).save(os.path.join(a.out, f"teaser3d_video_in_{j}.png"))
    Image.fromarray(ims[5]).save(os.path.join(a.out, "teaser3d_video_keep.png"))
    T["video"] = dict(name=f"video, window {w}", **st)
    json.dump(T, open(os.path.join(ROOT, "docs", "assets", "teaser3d.json"), "w"), indent=1)
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
