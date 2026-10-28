"""CO3Dv2 single-sequence subset for HalluBench-3D.

Raw data: https://github.com/facebookresearch/co3d, links in co3d/links.json ("singlesequence"),
zips <category>_000_singlesequence.zip (annotations) and <category>_001_singlesequence.zip (frames).
Only the two "dev" sequences per category carry depth; the "test" sequence is redacted (all zeros),
and within a dev sequence only the "*_known" frames carry depth. Depth is COLMAP MVS depth, valid
only where depth_masks/ is set (object foreground, roughly 7 to 10% of pixels).

Conventions (from pytorch3d implicitron frame_data._get_pytorch3d_camera and
renderer.cameras.get_ndc_to_screen_transform, intrinsics_format "ndc_isotropic"):
  X_cam_p3d = X_world @ R + T   (row vectors), p3d camera: +x left, +y up, +z forward
  x_ndc = f * x_cam / z_cam + p ;  x_pix = -s * x_ndc + W/2, s = min(H, W) / 2  (same for y)
  => OpenCV: fx_px = s f_x, cx = W/2 - s p_x,  E_cv = diag(-1,-1,1) [R^T | T]

preprocess_co3d() writes one folder per scene: images/*.jpg (resized to IMG_HW) and meta.npz with
depth (float32, 0 = invalid, at IMG_HW / DEPTH_DIV), fg (bool, object mask at depth res),
K_img, K_dep, E (world -> camera, OpenCV), frame_numbers, and the source paths.
"""
import glob
import gzip
import json
import os

import cv2
import numpy as np

IMG_HW = (640, 360)       # portrait sequences (phone captures ~713x1268 W x H); landscape ones use (360, 640)
DEPTH_DIV = 2             # depth / label resolution = IMG_HW / DEPTH_DIV
MODEL_HW = (518, 294)     # model input for portrait scenes, both divisible by 14, aspect 1.76 vs 1.78; swapped for landscape


def _load_depth_png(path, scale_adjustment):
    """pytorch3d.implicitron.dataset.utils.load_16big_png_depth: uint16 PNG reinterpreted as float16."""
    from PIL import Image
    with Image.open(path) as im:
        d = np.frombuffer(np.array(im, dtype=np.uint16), dtype=np.float16).astype(np.float32)
        d = d.reshape(im.size[1], im.size[0])
    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d


def _load_mask_png(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("L")) > 0


def _camera_cv(vp, hw):
    H, W = hw
    f = np.array(vp["focal_length"], np.float64)
    p = np.array(vp["principal_point"], np.float64)
    fmt = vp.get("intrinsics_format", "ndc_norm_image_bounds")
    if fmt == "ndc_norm_image_bounds":
        scale_wh = np.array([W, H], np.float64) / min(H, W)
        f, p = f * scale_wh, p * scale_wh
    elif fmt != "ndc_isotropic":
        raise ValueError(f"unknown intrinsics_format {fmt}")
    s = min(H, W) / 2.0
    K = np.array([[s * f[0], 0, W / 2.0 - s * p[0]],
                  [0, s * f[1], H / 2.0 - s * p[1]],
                  [0, 0, 1]], np.float64)
    R = np.array(vp["R"], np.float64)      # X_cam = X_world @ R + T
    T = np.array(vp["T"], np.float64)
    flip = np.diag([-1.0, -1.0, 1.0])
    E = np.eye(4)
    E[:3, :3] = flip @ R.T
    E[:3, 3] = flip @ T
    return K, E


def _downsample_depth(d, valid, div):
    """Valid-aware block mean; a block is valid if at least half of its pixels are."""
    h, w = d.shape[0] // div, d.shape[1] // div
    d = d[: h * div, : w * div].reshape(h, div, w, div)
    v = valid[: h * div, : w * div].reshape(h, div, w, div)
    cnt = v.sum(axis=(1, 3))
    out = (d * v).sum(axis=(1, 3)) / np.maximum(cnt, 1)
    out[cnt < (div * div + 1) // 2] = 0.0
    return out.astype(np.float32)


def _downsample_mask(m, div):
    h, w = m.shape[0] // div, m.shape[1] // div
    return m[: h * div, : w * div].reshape(h, div, w, div).mean(axis=(1, 3)) > 0.5


def preprocess_co3d(raw_root, out_root, categories=None, img_hw=IMG_HW, depth_div=DEPTH_DIV,
                    quality_min=None, verbose=True):
    """Convert every dev sequence of every category under raw_root into out_root/<cat>_<seq>/."""
    cats = sorted(categories or [os.path.basename(os.path.dirname(f)) for f in
                                 glob.glob(os.path.join(raw_root, "*", "frame_annotations.jgz"))])
    scenes = []
    for cat in cats:
        fa = json.load(gzip.open(os.path.join(raw_root, cat, "frame_annotations.jgz")))
        seqs = sorted({x["sequence_name"] for x in fa if x["meta"]["frame_type"].endswith("_known")
                       and not x["meta"]["frame_type"].startswith("test")})
        for seq in seqs:
            frames = sorted([x for x in fa if x["sequence_name"] == seq
                             and x["meta"]["frame_type"].endswith("_known")], key=lambda x: x["frame_number"])
            scene = f"{cat}_{seq}"
            d_out = os.path.join(out_root, scene)
            if os.path.exists(os.path.join(d_out, "meta.npz")):
                scenes.append(scene); continue
            os.makedirs(os.path.join(d_out, "images"), exist_ok=True)
            portrait = frames[0]["image"]["size"][0] >= frames[0]["image"]["size"][1]
            H, W = img_hw if portrait else img_hw[::-1]
            imgs_ok, deps, fgs, Ki, Kd, Es, fnums, paths = 0, [], [], [], [], [], [], []
            for x in frames:
                ip = os.path.join(raw_root, x["image"]["path"])
                img = cv2.imread(ip, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(ip)
                h0, w0 = img.shape[:2]
                if (h0, w0) != tuple(x["image"]["size"]):
                    raise ValueError(f"{ip}: size {img.shape[:2]} != annotation {x['image']['size']}")
                if (h0 >= w0) != portrait:
                    raise ValueError(f"{ip}: mixed orientations within one sequence")
                dep = _load_depth_png(os.path.join(raw_root, x["depth"]["path"]), x["depth"]["scale_adjustment"])
                dmask = _load_mask_png(os.path.join(raw_root, x["depth"]["mask_path"]))
                fg = _load_mask_png(os.path.join(raw_root, x["mask"]["path"]))
                if dep.shape != (h0, w0) or dmask.shape != (h0, w0) or fg.shape != (h0, w0):
                    raise ValueError(f"{ip}: depth/mask shape mismatch")
                valid = (dep > 0) & dmask
                if valid.mean() < 0.005:
                    continue                     # frame without usable depth
                K0, E = _camera_cv(x["viewpoint"], (h0, w0))
                # resize image (anisotropic, tiny aspect change) and scale intrinsics accordingly
                sx, sy = W / w0, H / h0
                img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(d_out, "images", f"{x['frame_number']:06d}.jpg"), img_r,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                K_img = K0.copy(); K_img[0] *= sx; K_img[1] *= sy
                dep_r = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
                val_r = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                fg_r = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                deps.append(_downsample_depth(dep_r, val_r, depth_div))
                fgs.append(_downsample_mask(fg_r, depth_div))
                K_dep = K_img.copy(); K_dep[:2] /= depth_div
                Ki.append(K_img); Kd.append(K_dep); Es.append(E); fnums.append(x["frame_number"])
                paths.append(x["image"]["path"]); imgs_ok += 1
            if imgs_ok < 20:
                if verbose:
                    print(f"[co3d] skip {scene}: only {imgs_ok} frames with depth")
                continue
            np.savez_compressed(os.path.join(d_out, "meta.npz"), depth=np.stack(deps), fg=np.stack(fgs),
                                K_img=np.stack(Ki), K_dep=np.stack(Kd), E=np.stack(Es),
                                frame_numbers=np.array(fnums), src_paths=np.array(paths),
                                img_hw=np.array((H, W)), depth_div=depth_div)
            scenes.append(scene)
            if verbose:
                print(f"[co3d] {scene}: {imgs_ok} frames, depth valid {np.mean([ (d>0).mean() for d in deps]):.3f}")
    return scenes


def find_co3d_root(search_dirs=("/kaggle/input", ".", "/kaggle/working", "/kaggle/tmp", "data")):
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, "**", "meta.npz"), recursive=True))
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("Could not find a preprocessed CO3D root (folders with meta.npz)")


def list_co3d_scenes(root):
    return sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(root, "*", "meta.npz")))


def n_frames(root, scene):
    return int(np.load(os.path.join(root, scene, "meta.npz"))["frame_numbers"].shape[0])


def load_co3d_scene(root, scene, views):
    """Same dict as hallu3d.load_scan plus center_mask (object foreground at depth res) and model_hw."""
    m = np.load(os.path.join(root, scene, "meta.npz"))
    fn = m["frame_numbers"]
    views = [int(v) for v in views]
    if max(views) >= len(fn):
        raise IndexError(f"{scene}: view index {max(views)} >= {len(fn)} frames")
    imgs = []
    for v in views:
        ip = os.path.join(root, scene, "images", f"{fn[v]:06d}.jpg")
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(img)
    return dict(imgs=np.stack(imgs), depths=m["depth"][views], K_img=m["K_img"][views], K_dep=m["K_dep"][views],
                E=m["E"][views], views=views, scan=scene, center_mask=m["fg"][views],
                model_hw=MODEL_HW if m["img_hw"][0] >= m["img_hw"][1] else MODEL_HW[::-1])


def scene_hw(root, scene):
    return tuple(int(v) for v in np.load(os.path.join(root, scene, "meta.npz"))["img_hw"])


def load_any_image(root, scene, view):
    fn = np.load(os.path.join(root, scene, "meta.npz"))["frame_numbers"]
    ip = os.path.join(root, scene, "images", f"{fn[view]:06d}.jpg")
    return cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
