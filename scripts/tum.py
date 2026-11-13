"""TUM RGB-D sequences as a real-occlusion dataset (indoor scenes, Kinect depth of every frame, motion-capture poses).

Raw data: https://cvg.cit.tum.de/data/datasets/rgbd-dataset (CC BY 4.0), one archive per sequence,
rgbd_dataset_<name>.tgz with rgb/*.png (640 x 480), depth/*.png (16 bit, 5000 units per metre, registered to the
colour frame, 0 where the sensor gives no depth), rgb.txt, depth.txt and groundtruth.txt (timestamp tx ty tz qx qy qz
qw: pose of the colour camera in the world, camera axes x right, y down, z forward).

preprocess_tum() writes the same folder format as co3d.preprocess_co3d and eth3d.preprocess_eth3d (images/*.jpg,
meta.npz with depth at half resolution, fg = valid depth, K_img, K_dep, E world -> camera OpenCV, frame_numbers,
img_hw, model_hw), so realocc_eval.py --co3d-format runs unchanged on it. N_FRAMES frames are taken evenly over the
sequence, each with a depth map and a pose within MAX_DT seconds of the colour frame; depth beyond MAX_DEPTH metres is
dropped (the sensor's error grows with the square of the distance). Order the frames by co-visibility afterwards:

  python scripts/tum.py <raw root> <out root> [--scenes freiburg1_desk,...]
  python scripts/eth3d.py <out root> --reorder
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eth3d import _downsample_depth, _quat_to_R  # noqa: E402

N_FRAMES = 80
MAX_DT = 0.02
MAX_DEPTH = 4.0
DEPTH_DIV = 2
MODEL_HW = (392, 518)   # both divisible by 14; aspect 1.32 against 1.33 of the frames
# colour-camera intrinsics published with the dataset (fx, fy, cx, cy)
INTR = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
        "freiburg3": (535.4, 539.2, 320.1, 247.6)}
SCENES = ["freiburg1_desk", "freiburg1_desk2", "freiburg1_room", "freiburg1_plant", "freiburg1_teddy", "freiburg1_360",
          "freiburg2_desk", "freiburg3_long_office_household", "freiburg3_teddy", "freiburg3_cabinet",
          "freiburg3_large_cabinet", "freiburg3_structure_texture_far"]


def _read_list(path):
    """timestamp -> rest of the line, comment lines skipped."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        out.append((float(p[0]), p[1:]))
    return out


def _nearest(ts, t):
    k = int(np.argmin(np.abs(ts - t)))
    return k, abs(float(ts[k]) - t)


def preprocess_tum(raw_root, out_root, scenes=None, n_frames=N_FRAMES, verbose=True):
    done = []
    for scene in scenes or SCENES:
        src = os.path.join(raw_root, "rgbd_dataset_" + scene)
        if not os.path.isdir(src):
            if verbose:
                print(f"[tum] missing {scene}")
            continue
        d_out = os.path.join(out_root, scene)
        if os.path.exists(os.path.join(d_out, "meta.npz")):
            done.append(scene); continue
        rgb, dep, gt = (_read_list(os.path.join(src, f)) for f in ("rgb.txt", "depth.txt", "groundtruth.txt"))
        t_dep, t_gt = np.array([t for t, _ in dep]), np.array([t for t, _ in gt])
        trip = []
        for t, (rp,) in rgb:
            kd, dd = _nearest(t_dep, t)
            kg, dg = _nearest(t_gt, t)
            if dd <= MAX_DT and dg <= MAX_DT:
                trip.append((rp, dep[kd][1][0], gt[kg][1]))
        if len(trip) < 10:
            if verbose:
                print(f"[tum] skip {scene}: only {len(trip)} frames with depth and pose")
            continue
        pick = np.unique(np.round(np.linspace(0, len(trip) - 1, min(n_frames, len(trip)))).astype(int))
        fx, fy, cx, cy = INTR[scene.split("_")[0]]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        os.makedirs(os.path.join(d_out, "images"), exist_ok=True)
        deps, fgs, Ki, Kd, Es, fnums, paths = [], [], [], [], [], [], []
        for k, i in enumerate(pick):
            rp, dp, pose = trip[i]
            img = cv2.imread(os.path.join(src, rp), cv2.IMREAD_COLOR)
            d16 = cv2.imread(os.path.join(src, dp), cv2.IMREAD_UNCHANGED)
            if img is None or d16 is None:
                raise FileNotFoundError(os.path.join(src, rp))
            d = d16.astype(np.float32) / 5000.0
            d[d > MAX_DEPTH] = 0
            tx, ty, tz, qx, qy, qz, qw = (float(x) for x in pose)
            Rcw = _quat_to_R((qw, qx, qy, qz))          # camera -> world
            E = np.eye(4); E[:3, :3] = Rcw.T; E[:3, 3] = -Rcw.T @ np.array([tx, ty, tz])
            cv2.imwrite(os.path.join(d_out, "images", f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            d2 = _downsample_depth(d, d > 0, DEPTH_DIV)
            K_dep = K.copy(); K_dep[:2] /= DEPTH_DIV
            deps.append(d2); fgs.append(d2 > 0); Ki.append(K.copy()); Kd.append(K_dep); Es.append(E)
            fnums.append(k); paths.append(rp)
        np.savez_compressed(os.path.join(d_out, "meta.npz"), depth=np.stack(deps), fg=np.stack(fgs),
                            K_img=np.stack(Ki), K_dep=np.stack(Kd), E=np.stack(Es), frame_numbers=np.array(fnums),
                            src_paths=np.array(paths), img_hw=np.array(img.shape[:2]), depth_div=DEPTH_DIV,
                            model_hw=np.array(MODEL_HW))
        done.append(scene)
        if verbose:
            print(f"[tum] {scene}: {len(deps)} of {len(trip)} frames, depth valid "
                  f"{np.mean([(d > 0).mean() for d in deps]):.3f}", flush=True)
    return done


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_root")
    ap.add_argument("out_root")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES,
                    help="frames taken evenly over the sequence, in temporal order (dynamic_eval.py: 200, not reordered)")
    a = ap.parse_args()
    preprocess_tum(a.raw_root, a.out_root, a.scenes.split(",") if a.scenes else None, n_frames=a.n_frames)
