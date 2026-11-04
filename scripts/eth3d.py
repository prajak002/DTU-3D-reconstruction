"""ETH3D high-resolution multi-view training scenes for HalluBench-3D (the third dataset: buildings, courtyards,
reliefs and statues at metre to tens-of-metre scale, with laser-scanner ground truth).

Raw data: https://www.eth3d.net/datasets, archive multi_view_training_dslr_undistorted.7z (5.9 GB), 13 scenes:
courtyard, delivery_area, electro, facade, kicker, meadow, office, pipes, playground, relief, relief_2, terrace,
terrains. Layout per scene: images/dslr_images_undistorted/*.JPG (undistorted DSLR frames, about 6000 x 4000),
dslr_calibration_undistorted/{cameras.txt, images.txt} (COLMAP text format, PINHOLE cameras, world -> camera
pose as quaternion + translation), ground_truth_depth/dslr_images/<same name> (per-scene archives <scene>_dslr_depth.7z: raw float32 depth,
row-major, one value per pixel of the ORIGINAL distorted frame, infinity or 0 where the laser scan gives no depth)
and dslr_calibration_jpg/ (from multi_view_training_dslr_jpg.7z: the THIN_PRISM_FISHEYE calibration of those
frames). The depth maps are resampled onto the undistorted pinhole frame with COLMAP's camera model
(undistort_depth), so every scene is pinhole afterwards.

preprocess_eth3d() writes the same folder format as co3d.preprocess_co3d (images/*.jpg at IMG_W width, meta.npz
with depth at half resolution, fg = valid depth, K_img, K_dep, E world -> camera OpenCV, frame_numbers,
img_hw, model_hw), so the CO3D code path of the benchmark (cfg["dataset"] = "co3d") runs unchanged on it.
"""
import os
import sys

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


def read_jpg_cameras(calib_dir):
    """The calibration of the original (distorted) DSLR frames: COLMAP THIN_PRISM_FISHEYE cameras
    (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1). The ground-truth depth maps are stored at this resolution."""
    cams = {}
    for line in open(os.path.join(calib_dir, "cameras.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = dict(model=t[1], w=int(t[2]), h=int(t[3]), p=np.array([float(x) for x in t[4:]]))
    names = {}
    lines = [l for l in open(os.path.join(calib_dir, "images.txt")) if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        names[os.path.basename(t[9])] = dict(cam=int(t[8]), q=[float(x) for x in t[1:5]])
    return cams, names


def thin_prism_fisheye_project(xn, yn, p):
    """COLMAP ThinPrismFisheyeCameraModel.ImgFromCam for normalised coordinates (x/z, y/z) -> pixels."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = p[:12]
    r = np.sqrt(xn * xn + yn * yn)
    theta = np.arctan(r)
    s = np.where(r > 1e-8, theta / np.maximum(r, 1e-12), 1.0)
    u, v = xn * s, yn * s
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) + sx1 * r2
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) + sy1 * r2
    return fx * (u + du) + cx, fy * (v + dv) + cy


def undistort_depth(dep_d, cam_d, K_u, hw_u):
    """Resample a depth map given at the distorted frame (camera cam_d) onto the undistorted pinhole frame
    (intrinsics K_u, size hw_u). Depth is the z coordinate in the camera frame, which the undistortion does not
    change, so a nearest-neighbour remap of the values is exact up to sampling."""
    H, W = hw_u
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xn = (us - K_u[0, 2]) / K_u[0, 0]
    yn = (vs - K_u[1, 2]) / K_u[1, 1]
    if cam_d["model"] == "THIN_PRISM_FISHEYE":
        xd, yd = thin_prism_fisheye_project(xn, yn, cam_d["p"])
    elif cam_d["model"] == "PINHOLE":
        fx, fy, cx, cy = cam_d["p"][:4]
        xd, yd = fx * xn + cx, fy * yn + cy
    else:
        raise ValueError(cam_d["model"])
    out = cv2.remap(dep_d, xd.astype(np.float32), yd.astype(np.float32), cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out.astype(np.float32)


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
        cams_d, names_d = read_jpg_cameras(os.path.join(src, "dslr_calibration_jpg"))
        os.makedirs(os.path.join(d_out, "images"), exist_ok=True)
        deps, fgs, Ki, Kd, Es, fnums, paths, hw_out = [], [], [], [], [], [], [], None
        for k, im in enumerate(images):
            ip = os.path.join(src, "images", im["name"])
            base = os.path.basename(im["name"])
            dp = os.path.join(src, "ground_truth_depth", "dslr_images", base)
            if not os.path.exists(dp) or base not in names_d:
                continue
            cam = cams[im["cam"]]
            cam_d = cams_d[names_d[base]["cam"]]
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(ip)
            h0, w0 = img.shape[:2]
            if (h0, w0) != (cam["h"], cam["w"]):
                raise ValueError(f"{ip}: image {img.shape[:2]} vs camera {(cam['h'], cam['w'])}")
            dep = undistort_depth(read_depth(dp, (cam_d["h"], cam_d["w"])), cam_d, cam["K"], (h0, w0))
            if (dep > 0).mean() < 0.05:
                continue
            if hw_out is None:                       # frame size fixed per scene from the first frame (undistorted
                hw_out = (int(round(img_w * h0 / w0)), img_w)   # frames of one scene differ by a few pixels)
            H, W = hw_out
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


def covisibility(depth, K, E, tol=0.03):
    """overlap[i, j] = share of valid pixels of frame i whose point lands inside frame j, in front of it, and within
    tol of frame j's own depth (occlusion-aware). Uses hallu3d.backproject / project on the stored depth grid."""
    import hallu3d as H
    N, h, w = depth.shape
    ov = np.zeros((N, N))
    for i in range(N):
        Pw = H.backproject(depth[i], K[i], E[i])
        vi = depth[i] > 0
        if vi.sum() == 0:
            continue
        for j in range(N):
            if i == j:
                continue
            u, v, z = H.project(Pw, K[j], E[j])
            ui, vj = np.round(u).astype(int), np.round(v).astype(int)
            ok = vi & (z > 0) & (ui >= 0) & (ui < w) & (vj >= 0) & (vj < h)
            dj = np.zeros((h, w)); dj[ok] = depth[j][vj[ok], ui[ok]]
            ok &= dj > 0
            ov[i, j] = (ok & (np.abs(dj - z) / np.maximum(dj, 1e-9) < tol)).sum() / vi.sum()
    return ov


def reorder_by_covisibility(prep_root, n_views=10, scenes=None, verbose=True):
    """Store the frames of every scene so that the first n_views are a reference frame and the frames that overlap
    with it most; the benchmark then takes them with view_mode "adjacent". ETH3D captures walk around large scenes, so
    frames spread evenly over the capture share little surface and a hidden region would be seen by one or two of
    them only. The reference is the frame whose top n_views-1 overlaps sum highest. The images on disk keep their
    names; only the per-frame arrays in meta.npz are permuted. Idempotent (stores covis_order)."""
    out = {}
    for scene in scenes or sorted(os.listdir(prep_root)):
        mp = os.path.join(prep_root, scene, "meta.npz")
        if not os.path.exists(mp):
            continue
        m = dict(np.load(mp))
        if "covis_order" in m:
            out[scene] = m["covis_order"].tolist(); continue
        ov = covisibility(m["depth"], m["K_dep"], m["E"])
        sym = 0.5 * (ov + ov.T)
        score = np.sort(sym, axis=1)[:, ::-1][:, : n_views - 1].sum(1)
        ref = int(np.argmax(score))
        order = [ref] + [int(j) for j in np.argsort(-sym[ref]) if j != ref]
        for k in ("depth", "fg", "K_img", "K_dep", "E", "frame_numbers", "src_paths"):
            m[k] = m[k][order]
        m["covis_order"] = np.array(order)
        m["covis_top"] = sym[ref][order[1:n_views]]
        np.savez_compressed(mp, **m)
        out[scene] = order
        if verbose:
            print(f"[eth3d] {scene}: reference frame {ref}, overlap of the next {n_views - 1} frames "
                  f"{np.round(sym[ref][order[1:n_views]], 2).tolist()}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_root", help="raw ETH3D root, or the preprocessed root with --reorder")
    ap.add_argument("out_root", nargs="?", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--reorder", action="store_true", help="order the frames of each preprocessed scene by co-visibility")
    a = ap.parse_args()
    sc = a.scenes.split(",") if a.scenes else None
    if a.reorder:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        reorder_by_covisibility(a.raw_root, scenes=sc)
    else:
        print(preprocess_eth3d(a.raw_root, a.out_root, sc))
