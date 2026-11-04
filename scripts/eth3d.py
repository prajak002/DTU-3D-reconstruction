"""ETH3D high-resolution multi-view training scenes for HalluBench-3D (the third dataset: buildings, courtyards,
reliefs and statues at metre to tens-of-metre scale, with laser-scanner ground truth).

Raw data: https://www.eth3d.net/datasets, archive multi_view_training_dslr_undistorted.7z (5.9 GB), 13 scenes:
courtyard, delivery_area, electro, facade, kicker, meadow, office, pipes, playground, relief, relief_2, terrace,
terrains. Layout per scene: images/dslr_images_undistorted/*.JPG (undistorted DSLR frames, about 6000 x 4000),
dslr_calibration_undistorted/{cameras.txt, images.txt} (COLMAP text format, PINHOLE cameras, world -> camera
pose as quaternion + translation), ground_truth_depth/dslr_images_undistorted/<same name> (raw float32 depth,
row-major, one value per pixel of the undistorted image, infinity or 0 where the laser scan gives no depth).

preprocess_eth3d() writes the same folder format as co3d.preprocess_co3d (images/*.jpg at IMG_W width, meta.npz
with depth at half resolution, fg = valid depth, K_img, K_dep, E world -> camera OpenCV, frame_numbers,
img_hw, model_hw), so the CO3D code path of the benchmark (cfg["dataset"] = "co3d") runs unchanged on it.
"""
import os

import cv2
import numpy as np

IMG_W = 640           # landscape frames, 3:2 -> 640 x 427
DEPTH_DIV = 2
MODEL_HW = (350, 518)   # both divisible by 14; aspect 1.48 against 1.50 of the frames
SCENES = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office", "pipes", "playground",
          "relief", "relief_2", "terrace", "terrains"]


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)


def read_colmap_text(calib_dir):
    cams = {}
    for line in open(os.path.join(calib_dir, "cameras.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cid, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
        p = [float(x) for x in t[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:
            raise ValueError(f"camera model {model} is not pinhole; use the undistorted archive")
        cams[cid] = dict(w=w, h=h, K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64))
    images = []
    lines = [l for l in open(os.path.join(calib_dir, "images.txt")) if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        q = [float(x) for x in t[1:5]]
        tr = np.array([float(x) for x in t[5:8]])
        E = np.eye(4)
        E[:3, :3] = _quat_to_R(q)
        E[:3, 3] = tr
        images.append(dict(id=int(t[0]), E=E, cam=int(t[8]), name=t[9]))
    images.sort(key=lambda x: x["name"])
    return cams, images


def read_depth(path, hw):
    d = np.fromfile(path, dtype=np.float32)
    if d.size != hw[0] * hw[1]:
        raise ValueError(f"{path}: {d.size} values for {hw}")
    d = d.reshape(hw)
    d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
    return d


def _downsample_depth(d, valid, div):
    h, w = d.shape[0] // div, d.shape[1] // div
    d = d[: h * div, : w * div].reshape(h, div, w, div)
    v = valid[: h * div, : w * div].reshape(h, div, w, div)
    cnt = v.sum(axis=(1, 3))
    out = (d * v).sum(axis=(1, 3)) / np.maximum(cnt, 1)
    out[cnt < (div * div + 1) // 2] = 0.0
    return out.astype(np.float32)


def preprocess_eth3d(raw_root, out_root, scenes=None, img_w=IMG_W, depth_div=DEPTH_DIV, verbose=True):
    done = []
    for scene in scenes or SCENES:
        src = os.path.join(raw_root, scene)
        if not os.path.isdir(src):
            if verbose:
                print(f"[eth3d] missing {scene}")
            continue
        d_out = os.path.join(out_root, scene)
        if os.path.exists(os.path.join(d_out, "meta.npz")):
            done.append(scene); continue
        cams, images = read_colmap_text(os.path.join(src, "dslr_calibration_undistorted"))
        os.makedirs(os.path.join(d_out, "images"), exist_ok=True)
        deps, fgs, Ki, Kd, Es, fnums, paths, hw_out = [], [], [], [], [], [], [], None
        for k, im in enumerate(images):
            ip = os.path.join(src, "images", im["name"])
            dp = os.path.join(src, "ground_truth_depth", im["name"])
            if not os.path.exists(dp):
                continue
            cam = cams[im["cam"]]
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(ip)
            h0, w0 = img.shape[:2]
            if (h0, w0) != (cam["h"], cam["w"]):
                raise ValueError(f"{ip}: image {img.shape[:2]} vs camera {(cam['h'], cam['w'])}")
            dep = read_depth(dp, (h0, w0))
            if (dep > 0).mean() < 0.05:
                continue
            W = img_w
            H = int(round(W * h0 / w0))
            if hw_out is None:
                hw_out = (H, W)
            elif hw_out != (H, W):
                raise ValueError(f"{scene}: mixed frame sizes {hw_out} vs {(H, W)}")
            img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(d_out, "images", f"{k:06d}.jpg"), img_r, [cv2.IMWRITE_JPEG_QUALITY, 95])
            sx, sy = W / w0, H / h0
            K_img = cam["K"].copy(); K_img[0] *= sx; K_img[1] *= sy
            dep_r = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
            val_r = dep_r > 0
            d2 = _downsample_depth(dep_r, val_r, depth_div)
            deps.append(d2); fgs.append(d2 > 0)
            K_dep = K_img.copy(); K_dep[:2] /= depth_div
            Ki.append(K_img); Kd.append(K_dep); Es.append(im["E"]); fnums.append(k); paths.append(im["name"])
        if len(deps) < 10:
            if verbose:
                print(f"[eth3d] skip {scene}: only {len(deps)} frames with depth")
            continue
        np.savez_compressed(os.path.join(d_out, "meta.npz"), depth=np.stack(deps), fg=np.stack(fgs),
                            K_img=np.stack(Ki), K_dep=np.stack(Kd), E=np.stack(Es), frame_numbers=np.array(fnums),
                            src_paths=np.array(paths), img_hw=np.array(hw_out), depth_div=depth_div,
                            model_hw=np.array(MODEL_HW))
        done.append(scene)
        if verbose:
            print(f"[eth3d] {scene}: {len(deps)} frames at {hw_out}, depth valid {np.mean([(d > 0).mean() for d in deps]):.3f}")
    return done


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_root")
    ap.add_argument("out_root")
    ap.add_argument("--scenes", default=None)
    a = ap.parse_args()
    print(preprocess_eth3d(a.raw_root, a.out_root, a.scenes.split(",") if a.scenes else None))
