"""Phototourism (Image Matching Challenge 2020) heritage scenes for HalluBench-3D: in-the-wild photo collections of
landmarks (British Museum, Florence cathedral, Lincoln Memorial, London Bridge, Milan cathedral, Mount Rushmore,
Piazza San Marco, Sagrada Familia, St Paul's) with COLMAP multi-view-stereo depth and calibration, as redistributed
on Kaggle (ayatnabil/yfcc-phototourism: <scene>/set_100/{images, calibration, depth_maps, visibility}).

Per image: images/<id>.jpg (undistorted, varying size), calibration/calibration_<id>.h5 with K (3x3 pinhole, principal
point at the image centre), R, T (world -> camera, x_cam = R x + T), depth_maps/<id>.h5 ("depth", float16, image
resolution, 0 where MVS gave no depth; about 20 to 40% valid), visibility/vis_<id>.txt (one co-visibility value per
image of the set). Tourist photos: people, vehicles and scaffolding in front of the monument, unordered viewpoints,
different focal lengths. This is the capture the detector is meant for.

preprocess_imc() writes the CO3D-format folder (scripts/co3d.py: images/*.jpg, meta.npz) so that every benchmark
and vis-protocol code path runs unchanged with cfg["dataset"] = "co3d". Landscape images only; each is resized
anisotropically to IMG_HW (the intrinsics are scaled with it; the aspect change is at most 10%); frames are stored
in co-visibility order from the best-connected image, so view_mode "adjacent" takes the reference and its most
co-visible neighbours (as for ETH3D).

Usage: python scripts/imc.py <raw root containing phototourism/> <out root> [--scenes a,b] [--frames 20]
"""
import argparse
import glob
import os

import cv2
import numpy as np

IMG_HW = (448, 640)       # 7:10, between the 3:2 and 4:3 frames of the collection; both divisible by 2 and by 14 at model res
MODEL_HW = (364, 518)     # 364 = 26 x 14, 518 = 37 x 14; aspect 1.42 against 1.43
DEPTH_DIV = 2
SCENES = ["british_museum", "florence_cathedral_side", "lincoln_memorial_statue", "london_bridge", "milan_cathedral",
          "mount_rushmore", "piazza_san_marco", "sagrada_familia", "st_pauls_cathedral"]


def _ids(scene_dir):
    return [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(scene_dir, "images", "*.jpg")))]


def _vis_matrix(scene_dir, ids):
    """Co-visibility v[i, j] from visibility/vis_<id>.txt (one value per image in the order of images.txt)."""
    order = [os.path.basename(l.strip())[:-4] for l in open(os.path.join(scene_dir, "images.txt")) if l.strip()]
    pos = {k: i for i, k in enumerate(order)}
    V = np.zeros((len(ids), len(ids)))
    for a, ia in enumerate(ids):
        p = os.path.join(scene_dir, "visibility", f"vis_{ia}.txt")
        vals = np.array([float(x) for x in open(p).read().split()])
        for b, ib in enumerate(ids):
            V[a, b] = vals[pos[ib]] if ib in pos and pos[ib] < len(vals) else 0.0
    return 0.5 * (V + V.T)


def preprocess_imc(raw_root, out_root, scenes=None, n_frames=20, img_hw=IMG_HW, depth_div=DEPTH_DIV):
    import h5py
    for scene in scenes or SCENES:
        sd = os.path.join(raw_root, "phototourism", scene, "set_100")
        if not os.path.isdir(sd):
            print(f"[imc] {scene}: no set_100 folder, skipped"); continue
        d_out = os.path.join(out_root, scene)
        if os.path.exists(os.path.join(d_out, "meta.npz")):
            print(f"[imc] {scene}: already preprocessed"); continue
        ids = _ids(sd)
        land = []
        for i in ids:                         # landscape frames with depth and calibration only
            im = cv2.imread(os.path.join(sd, "images", i + ".jpg"), cv2.IMREAD_COLOR)
            if im is None or im.shape[0] > im.shape[1]:
                continue
            if not (os.path.exists(os.path.join(sd, "depth_maps", i + ".h5")) and os.path.exists(os.path.join(sd, "calibration", f"calibration_{i}.h5"))):
                continue
            land.append(i)
        V = _vis_matrix(sd, land)
        ref = int(np.argmax(V.sum(1)))
        order = [ref] + [int(j) for j in np.argsort(-V[ref]) if j != ref][: n_frames - 1]
        H, W = img_hw
        os.makedirs(os.path.join(d_out, "images"), exist_ok=True)
        deps, fgs, Ki, Kd, Es, names = [], [], [], [], [], []
        for k, j in enumerate(order):
            i = land[j]
            im = cv2.imread(os.path.join(sd, "images", i + ".jpg"), cv2.IMREAD_COLOR)
            h0, w0 = im.shape[:2]
            with h5py.File(os.path.join(sd, "calibration", f"calibration_{i}.h5")) as f:
                K0, R, T = f["K"][()].astype(np.float64), f["R"][()].astype(np.float64), f["T"][()].astype(np.float64)
            with h5py.File(os.path.join(sd, "depth_maps", i + ".h5")) as f:
                dep = f["depth"][()].astype(np.float32)
            dep = np.where(np.isfinite(dep) & (dep > 0), dep, 0.0).astype(np.float32)
            sy, sx = H / h0, W / w0
            cv2.imwrite(os.path.join(d_out, "images", f"{k:06d}.jpg"), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            K_img = K0.copy(); K_img[0] *= sx; K_img[1] *= sy
            hd, wd = H // depth_div, W // depth_div
            dep_d = cv2.resize(dep, (wd, hd), interpolation=cv2.INTER_NEAREST)
            K_dep = K_img.copy(); K_dep[:2] /= depth_div
            E = np.eye(4); E[:3, :3] = R; E[:3, 3] = T
            deps.append(dep_d); fgs.append(dep_d > 0); Ki.append(K_img); Kd.append(K_dep); Es.append(E); names.append(i)
        np.savez_compressed(os.path.join(d_out, "meta.npz"), depth=np.stack(deps), fg=np.stack(fgs),
                            K_img=np.stack(Ki), K_dep=np.stack(Kd), E=np.stack(Es), frame_numbers=np.arange(len(order)),
                            img_hw=np.array((H, W)), model_hw=np.array(MODEL_HW), depth_div=depth_div,
                            source_ids=np.array(names), covis_to_ref=V[ref][order])
        print(f"[imc] {scene}: {len(land)} landscape frames, kept {len(order)} (reference {land[ref]}, co-visibility to it "
              f"{V[ref][order][1:].min():.2f} to {V[ref][order][1:].max():.2f}), valid depth {np.mean([f.mean() for f in fgs]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root")
    ap.add_argument("out_root")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--frames", type=int, default=20)
    a = ap.parse_args()
    preprocess_imc(a.raw_root, a.out_root, a.scenes.split(",") if a.scenes else None, a.frames)
