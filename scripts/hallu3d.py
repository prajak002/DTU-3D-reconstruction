"""
hallu3d.py  -  Geometric hallucination benchmark for feed-forward 3D reconstruction.

Test M1 (3D-consistent region masking) on DTU:
  * Pick a 3D sphere on the object surface.
  * Hide every pixel whose GT surface point lies inside that sphere, in EVERY input view.
    Those surface points are now unsupported: no input view shows them.
  * Run the model on masked views and on the clean views (control).
  * Ask: does the model still predict geometry there with high confidence?

Labels are free by construction:
  U = unsupported pixels (hidden in all views)
  S = supported pixels (valid GT, far from the hidden region)
  ring = pixels near the mask border, ignored

Metrics (per model, scan, seed, radius, fill):
  auroc     P(conf_S > conf_U). 1.0 = model knows what it cannot see. 0.5 = blind.
  cur       Confident-Unsupported Rate. Threshold tau keeps 90% of S; cur = frac of U above tau.
            This is the hallucination rate.
  cur_err   frac of U that is above tau AND has AbsRel > 0.1 (confident and wrong).
  absrel_S, absrel_U, d125_U        depth error after per-sequence median scale on S only.
  absrel_U_ctrl, conf_ratio_U       same pixels in the clean control run.

Nothing here touches GT inside U when aligning scale. Scale uses S only.
"""

import os, re, glob, json, math, hashlib, time
import numpy as np
import cv2

# ----------------------------------------------------------------------------- config
DEFAULT_CFG = dict(
    scans=[3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117],  # MVSNet validation split
    n_views=10,
    light=3,
    seeds=[0, 1, 2, 3, 4],
    radius_fracs=[0.10, 0.18],      # sphere radius as fraction of object bbox diagonal
    fills=["gray", "blur", "noise"],
    model_hw=(420, 518),            # model input (H, W), both divisible by 14
    dilate_px=6,                    # mask dilation at image res (640x512), border goes to ring
    min_U_frac=0.02, max_U_frac=0.35,  # U pixels as fraction of valid pixels, ref view
    max_tries=50,
    tau_keep=0.90,
    err_thresh=0.10,
    view_mode="spread",             # "spread" | "adjacent" (low-parallax stress test)
    distractor=False,               # append one image from another scan to the model input (dropped from outputs)
    distractor_view=0,
    dataset="dtu",                  # "dtu" | "co3d" (scans are then CO3D scene names, see co3d.py)
)

DTU_IMG_HW = (512, 640)


# ----------------------------------------------------------------------------- io
def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        color = header == "PF"
        dims = f.readline().decode("utf-8")
        m = re.match(r"^(\d+)\s(\d+)\s*$", dims)
        if not m:
            raise ValueError(f"Malformed PFM header: {path}")
        w, h = map(int, m.groups())
        scale = float(f.readline().decode("utf-8").rstrip())
        dt = "<f" if scale < 0 else ">f"
        data = np.frombuffer(f.read(), dtype=dt)
    shape = (h, w, 3) if color else (h, w)
    data = np.reshape(data, shape)
    data = np.flipud(data).astype(np.float32).copy()
    return data


def read_cam(path):
    words = open(path).read().split()
    E = np.array(words[1:17], dtype=np.float64).reshape(4, 4)   # world -> camera
    K = np.array(words[18:27], dtype=np.float64).reshape(3, 3)
    return E, K


def find_dtu_root(search_dirs=("/kaggle/input", ".", "/kaggle/working", "/kaggle/tmp")):
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        hits = glob.glob(os.path.join(d, "**", "Cameras", "train"), recursive=True)
        for h in hits:
            root = os.path.dirname(os.path.dirname(h))
            if os.path.isdir(os.path.join(root, "Rectified")) and os.path.isdir(os.path.join(root, "Depths")):
                return root
    raise FileNotFoundError("Could not find a DTU root containing Cameras/train, Rectified, Depths")


def _scale_K_to(K, hw):
    """Cameras/train intrinsics are at a lower resolution. Find the power-of-2 factor
    that puts the principal point near the image center of resolution hw."""
    H, W = hw
    best, best_err = None, 1e9
    for k in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        cx, cy = K[0, 2] * k, K[1, 2] * k
        err = abs(cx - W / 2) / W + abs(cy - H / 2) / H
        if err < best_err:
            best, best_err = k, err
    if best_err > 0.15:
        raise ValueError(f"Cannot match intrinsics to {hw}: K={K}, best err {best_err:.3f}")
    K2 = K.copy()
    K2[:2] *= best
    return K2, best


def select_views(n_views, total=49, mode="spread"):
    """spread: evenly over the rig (default). adjacent: consecutive camera indices starting at 0 (low parallax)."""
    if mode == "adjacent":
        return list(range(n_views))
    return [int(v) for v in np.round(np.linspace(0, total - 1, n_views))]


def load_scan(root, scan, views, light=3):
    imgs, depths, Ks_img, Ks_dep, Es = [], [], [], [], []
    for v in views:
        ip = os.path.join(root, "Rectified", f"scan{scan}_train", f"rect_{v + 1:03d}_{light}_r5000.png")
        dp = os.path.join(root, "Depths", f"scan{scan}_train", f"depth_map_{v:04d}.pfm")
        cp = os.path.join(root, "Cameras", "train", f"{v:08d}_cam.txt")
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if img.shape[:2] != DTU_IMG_HW:
            raise ValueError(f"Unexpected image size {img.shape} at {ip}")
        dep = read_pfm(dp)
        E, K = read_cam(cp)
        Ki, _ = _scale_K_to(K, img.shape[:2])
        Kd, _ = _scale_K_to(K, dep.shape[:2])
        imgs.append(img); depths.append(dep); Ks_img.append(Ki); Ks_dep.append(Kd); Es.append(E)
    return dict(imgs=np.stack(imgs), depths=np.stack(depths), K_img=np.stack(Ks_img),
                K_dep=np.stack(Ks_dep), E=np.stack(Es), views=list(views), scan=scan)


# ----------------------------------------------------------------------------- geometry
def backproject(depth, K, E):
    """depth HxW (0 = invalid), K at depth res, E world->cam. Returns HxWx3 world points."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    Pc = np.stack([x, y, depth.astype(np.float64), np.ones_like(x)], -1)
    Ew2c_inv = np.linalg.inv(E)
    Pw = Pc @ Ew2c_inv.T
    return Pw[..., :3]


def project(Pw, K, E):
    Ph = np.concatenate([Pw, np.ones(Pw.shape[:-1] + (1,))], -1)
    Pc = Ph @ E.T
    z = Pc[..., 2]
    u = K[0, 0] * Pc[..., 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * Pc[..., 1] / np.maximum(z, 1e-9) + K[1, 2]
    return u, v, z


def _stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def build_masks(data, seed, radius_frac, cfg):
    """Returns dict with image-res hide mask (N,512,640) bool and depth-res labels
    U (N,h,w) bool, S (N,h,w) bool, plus sphere center/radius. Deterministic in (scan, seed, radius)."""
    rng = np.random.default_rng(_stable_seed("mask", data["scan"], seed, radius_frac))
    depths, Kd, E = data["depths"], data["K_dep"], data["E"]
    N, h, w = depths.shape
    H, W = data["imgs"].shape[1:3]
    Pws = [backproject(depths[i], Kd[i], E[i]) for i in range(N)]
    valid = depths > 0

    ref = 0
    cm = data.get("center_mask")
    pts_ref = Pws[ref][valid[ref] & (cm[ref] if cm is not None else True)]
    if len(pts_ref) < 100:
        raise ValueError(f"scan {data['scan']}: too few valid GT pixels in ref view")
    lo, hi = np.percentile(pts_ref, 2, axis=0), np.percentile(pts_ref, 98, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    radius = radius_frac * diag

    for _ in range(cfg["max_tries"]):
        c = pts_ref[rng.integers(len(pts_ref))]
        inside = np.stack([(np.linalg.norm(Pws[i] - c, axis=-1) < radius) & valid[i] for i in range(N)])
        frac_ref = inside[ref].sum() / max(valid[ref].sum(), 1)
        if cfg["min_U_frac"] <= frac_ref <= cfg["max_U_frac"]:
            break
    else:
        raise RuntimeError(f"scan {data['scan']} seed {seed} r{radius_frac}: no valid sphere after {cfg['max_tries']} tries")

    sy, sx = H // h, W // w
    k = cfg["dilate_px"] * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    hide = np.zeros((N, H, W), bool)
    U = np.zeros((N, h, w), bool)
    S = np.zeros((N, h, w), bool)
    for i in range(N):
        up = cv2.resize(inside[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        up = cv2.GaussianBlur(up.astype(np.float32), (0, 0), sigmaX=max(sx, sy) * 0.6) > 0.5   # soften block edges
        hide[i] = up
        dil = cv2.dilate(up.astype(np.uint8), kernel) > 0
        # block-level labels at depth resolution
        hide_blk = up.reshape(h, sy, w, sx).mean(axis=(1, 3))
        dil_blk = dil.reshape(h, sy, w, sx).mean(axis=(1, 3))
        U[i] = (hide_blk >= 0.999) & valid[i] & inside[i]
        S[i] = (dil_blk <= 0.0) & valid[i]
    return dict(hide=hide, U=U, S=S, center=c, radius=radius, diag=diag)


def apply_fill(imgs, hide, mode, seed):
    out = imgs.copy()
    if mode == "none":
        return out
    rng = np.random.default_rng(_stable_seed("fill", mode, seed))
    for i in range(len(imgs)):
        m = hide[i]
        if not m.any():
            continue
        if mode == "gray":
            out[i][m] = 0.5
        elif mode == "blur":
            b = cv2.GaussianBlur(imgs[i], (0, 0), sigmaX=25)
            out[i][m] = b[m]
        elif mode == "noise":
            out[i][m] = rng.random((int(m.sum()), 3), dtype=np.float32)
        elif mode == "inpaint":
            # natural-looking fill: Telea inpainting from the surrounding pixels (no learned prior)
            src = (np.clip(imgs[i], 0, 1) * 255).astype(np.uint8)
            inp = cv2.inpaint(src, m.astype(np.uint8), 7, cv2.INPAINT_TELEA)
            out[i][m] = inp[m].astype(np.float32) / 255.0
        else:
            raise ValueError(mode)
    return out


def consistency_check(data, masks, max_pts=4000, seed=0):
    """Correctness guard for cameras + depth + masks.
    1) GT depth reprojection agreement between view pairs (should be high).
    2) U points from view i that are visible in view j should land in view j's hide mask."""
    rng = np.random.default_rng(seed)
    depths, Kd, E = data["depths"], data["K_dep"], data["E"]
    N, h, w = depths.shape
    H, W = data["imgs"].shape[1:3]
    agree, agree_vis, inmask = [], [], []
    for i in range(N):
        Pw = backproject(depths[i], Kd[i], E[i])
        vi = depths[i] > 0
        for j in range(N):
            if i == j:
                continue
            idx = np.argwhere(vi)
            if len(idx) == 0:
                continue
            idx = idx[rng.choice(len(idx), min(max_pts, len(idx)), replace=False)]
            P = Pw[idx[:, 0], idx[:, 1]]
            u, v, z = project(P, Kd[j], E[j])
            ui, vi_ = np.round(u).astype(int), np.round(v).astype(int)
            ok = (ui >= 0) & (ui < w) & (vi_ >= 0) & (vi_ < h) & (z > 0)
            dj = depths[j][vi_[ok], ui[ok]]
            rel = (dj - z[ok]) / np.maximum(dj, 1e-6)
            vis = (dj > 0) & (np.abs(rel) < 0.01)
            # looser diagnostic for MVS ground truth (CO3D): 3% tolerance, points occluded in view j
            # (a closer surface, rel < -0.03) are not counted as disagreement
            cand = (dj > 0) & (rel > -0.03)
            if ok.sum() > 50:
                agree.append(vis.mean())
                if cand.sum() > 50:
                    agree_vis.append((np.abs(rel[cand]) < 0.03).mean())
            # U transfer
            Ui = masks["U"][i][idx[:, 0], idx[:, 1]]
            sel = ok.copy(); sel[ok] = vis & Ui[ok]
            if sel.sum() > 20:
                inmask.append(masks["hide"][j][(vi_[sel] * (H // h)), (ui[sel] * (W // w))].mean())
    return dict(reproj_agree=float(np.mean(agree)) if agree else float("nan"),
                reproj_agree_3pct_vis=float(np.mean(agree_vis)) if agree_vis else float("nan"),
                U_transfer=float(np.mean(inmask)) if inmask else float("nan"))


# ----------------------------------------------------------------------------- metrics
def auroc(pos, neg):
    """P(pos > neg) with ties counted half. pos = conf on S, neg = conf on U."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def median_scale(pred, gt, S):
    r = gt[S] / np.maximum(pred[S], 1e-9)
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r))


def depth_metrics(pred, gt, M):
    if M.sum() == 0:
        return float("nan"), float("nan")
    p, g = pred[M], gt[M]
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / np.maximum(p, 1e-9))
    return absrel, float(np.mean(ratio < 1.25))


def compute_metrics(pred_depth, conf, gt, U, S, cfg, ctrl_depth=None, ctrl_conf=None):
    s = median_scale(pred_depth, gt, S)
    pd = pred_depth * s
    out = dict(scale=s, n_U=int(U.sum()), n_S=int(S.sum()))
    out["absrel_S"], out["d125_S"] = depth_metrics(pd, gt, S)
    out["absrel_U"], out["d125_U"] = depth_metrics(pd, gt, U)
    cS, cU = conf[S], conf[U]
    out["auroc"] = auroc(cS, cU)
    tau = np.quantile(cS, 1 - cfg["tau_keep"])
    out["cur"] = float(np.mean(cU >= tau))
    errU = np.abs(pd[U] - gt[U]) / gt[U]
    out["cur_err"] = float(np.mean((cU >= tau) & (errU > cfg["err_thresh"])))
    ok_corr = len(cU) > 2 and np.std(cU) > 0 and np.std(errU) > 0
    out["err_U_mean_conf_corr"] = float(np.corrcoef(cU, errU)[0, 1]) if ok_corr else float("nan")
    if ctrl_depth is not None:
        sc = median_scale(ctrl_depth, gt, S)
        pc = ctrl_depth * sc
        out["absrel_U_ctrl"], out["d125_U_ctrl"] = depth_metrics(pc, gt, U)
        out["absrel_S_ctrl"], _ = depth_metrics(pc, gt, S)
        mc = np.median(ctrl_conf[U])
        out["conf_ratio_U"] = float(np.median(cU) / mc) if mc > 0 else float("nan")
        # does the model rank U lower under masking than it did when U was visible?
        out["auroc_ctrl"] = auroc(ctrl_conf[S], ctrl_conf[U])
    return out


# ----------------------------------------------------------------------------- models
def _resize_t(x, hw, mode):
    import torch.nn.functional as F
    if mode == "area":
        return F.interpolate(x, size=hw, mode="area")
    return F.interpolate(x, size=hw, mode=mode, align_corners=False, antialias=True)


class ModelWrap:
    name = "base"

    def __init__(self, device="cuda"):
        import torch
        self.torch = torch
        self.device = device
        cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        self.dtype = torch.bfloat16 if cap >= 8 else torch.float16

    def _prep(self, imgs, hw):
        t = self.torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
        return _resize_t(t, hw, "bilinear").clamp(0, 1).to(self.device)

    def predict(self, imgs, model_hw, out_hw):
        """imgs: (N,H,W,3) float [0,1]. Returns depth (N,h,w), conf (N,h,w) numpy float32 at out_hw."""
        torch = self.torch
        x = self._prep(imgs, model_hw)
        with torch.no_grad():
            d, c = self._forward(x, self.dtype)
            if not (torch.isfinite(d).all() and torch.isfinite(c).all()):
                print(f"[{self.name}] non-finite output in {self.dtype}, retrying fp32")
                d, c = self._forward(x, None)
        d = _resize_t(d[:, None].float(), out_hw, "area")[:, 0]
        c = _resize_t(c[:, None].float(), out_hw, "area")[:, 0]
        return d.cpu().numpy().astype(np.float32), c.cpu().numpy().astype(np.float32)

    def free(self):
        del self.model
        self.torch.cuda.empty_cache()


class VGGTWrap(ModelWrap):
    name = "vggt"

    def __init__(self, device="cuda"):
        super().__init__(device)
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    def _forward(self, x, dtype):
        torch = self.torch
        if dtype is None:
            out = self.model(x)
        else:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = self.model(x)
        return out["depth"][0, ..., 0], out["depth_conf"][0]


class Pi3XWrap(ModelWrap):
    name = "pi3x"

    def __init__(self, device="cuda"):
        super().__init__(device)
        from pi3.models.pi3x import Pi3X
        self.model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()

    def _forward(self, x, dtype):
        torch = self.torch
        if dtype is None:
            out = self.model(x[None], with_prior=False)
        else:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = self.model(x[None], with_prior=False)
        return out["local_points"][0, ..., 2], torch.sigmoid(out["conf"][0, ..., 0])


class Pi3Wrap(ModelWrap):
    name = "pi3"

    def __init__(self, device="cuda"):
        super().__init__(device)
        from pi3.models.pi3 import Pi3
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    def _forward(self, x, dtype):
        torch = self.torch
        if dtype is None:
            out = self.model(x[None])
        else:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = self.model(x[None])
        return out["local_points"][0, ..., 2], torch.sigmoid(out["conf"][0, ..., 0])


class DA3Wrap(ModelWrap):
    """Depth Anything 3 (ByteDance-Seed/Depth-Anything-3). API verified from src/depth_anything_3/api.py:
    inference(list of HxWx3 uint8) -> Prediction with .depth (N,h,w) and .conf (N,h,w) numpy at processed res.
    Autocast bf16/fp16 is applied inside the model's forward. Use the -1.1 checkpoints (originals deprecated)."""
    name = "da3"
    ckpt = "depth-anything/DA3-LARGE-1.1"

    def __init__(self, device="cuda"):
        super().__init__(device)
        from depth_anything_3.api import DepthAnything3
        torch = self.torch
        if self.dtype == torch.float16:
            # DA3 picks bf16 whenever torch reports bf16 support, which recent torch does on T4 via emulation.
            # The memory-efficient SDPA kernel cannot run bf16 on sm75, so attention falls back to the math path
            # and materialises a 7 GiB fp32 attention matrix for 10 views. Force fp16 on pre-Ampere GPUs.
            torch.cuda.is_bf16_supported = lambda *a, **k: False
        self.model = DepthAnything3.from_pretrained(self.ckpt).to(device).eval()

    def predict(self, imgs, model_hw, out_hw):
        # longest side -> max(model_hw), then rounded to a multiple of 14 inside DA3 (640x512 -> 518x420)
        ims = [(np.clip(im, 0, 1) * 255).astype(np.uint8) for im in imgs]
        pred = self.model.inference(ims, process_res=int(max(model_hw)), process_res_method="upper_bound_resize")
        d = np.asarray(pred.depth, np.float32)
        c = np.asarray(pred.conf, np.float32) if pred.conf is not None else np.ones_like(d)
        if not (np.isfinite(d).all() and np.isfinite(c).all()):
            print(f"[{self.name}] non-finite output; replacing with 0")
            d, c = np.nan_to_num(d), np.nan_to_num(c)
        d = np.stack([cv2.resize(x, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA) for x in d])
        c = np.stack([cv2.resize(x, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA) for x in c])
        return d.astype(np.float32), c.astype(np.float32)


class MapAnythingWrap(ModelWrap):
    """MapAnything (facebookresearch/map-anything). API verified from mapanything/models/mapanything/model.py:
    infer(views=[{"img": (B,3,H,W) normalized, "data_norm_type": ["dinov2"]}, ...]) -> list of per-view dicts with
    "depth_z" (B,H,W,1) and "conf" (B,H,W). use_multiview_confidence=True swaps in a multi-view consistency conf."""
    name = "mapanything"
    ckpt = "facebook/map-anything"
    multiview_conf = False

    def __init__(self, device="cuda"):
        super().__init__(device)
        from mapanything.models import MapAnything
        self.model = MapAnything.from_pretrained(self.ckpt).to(device).eval()

    def _forward(self, x, dtype):
        torch = self.torch
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        xn = (x - mean) / std
        views = [{"img": xn[i:i + 1], "data_norm_type": ["dinov2"]} for i in range(x.shape[0])]
        amp = dtype is not None
        preds = self.model.infer(views, memory_efficient_inference=False, use_amp=amp,
                                 amp_dtype="bf16" if dtype == torch.bfloat16 else "fp16",
                                 apply_mask=False, mask_edges=False, apply_confidence_mask=False,
                                 use_multiview_confidence=self.multiview_conf)
        d = torch.cat([p["depth_z"][..., 0] for p in preds], 0)
        c = torch.cat([p["conf"] for p in preds], 0)
        return d.float(), c.float()


class MapAnythingMVWrap(MapAnythingWrap):
    """Same model, multi-view consistency confidence instead of the learned one (training-free baseline)."""
    name = "mapanything_mv"
    multiview_conf = True


MODEL_REGISTRY = {"vggt": VGGTWrap, "pi3x": Pi3XWrap, "pi3": Pi3Wrap, "da3": DA3Wrap,
                  "mapanything": MapAnythingWrap, "mapanything_mv": MapAnythingMVWrap}


class PerturbWrap:
    """Training-free detector (ReVar3R style): K forwards of the base model under photometric perturbations,
    each sample aligned by its global median depth (scale ambiguity), confidence = 1 / (eps + relative std
    across samples). Depth = per-pixel median of the aligned samples, re-scaled to the first sample.
    Costs K base forwards; report this when comparing at equal compute."""

    def __init__(self, base, K=4, strength=0.15):
        self.base, self.K, self.strength = base, K, strength
        self.name = f"{base.name}_pv{K}"

    def predict(self, imgs, model_hw, out_hw):
        rng = np.random.default_rng(_stable_seed("perturb", self.name))
        ds, s0 = [], None
        for k in range(self.K):
            x = imgs
            if k > 0:
                s = self.strength
                gain = rng.uniform(1 - s, 1 + s, size=(1, 1, 1, 3)).astype(np.float32)
                bias = rng.uniform(-s / 2, s / 2, size=(1, 1, 1, 3)).astype(np.float32)
                gamma = float(rng.uniform(1 - s, 1 + s))
                x = np.clip(np.clip(imgs * gain + bias, 0, 1) ** gamma, 0, 1).astype(np.float32)
            d, _ = self.base.predict(x, model_hw, out_hw)
            sk = max(float(np.median(d[d > 0])), 1e-9)
            s0 = s0 if s0 is not None else sk
            ds.append(d / sk)
        ds = np.stack(ds)                      # K,N,h,w
        med = np.median(ds, 0)
        rel_std = ds.std(0) / np.maximum(np.abs(med), 1e-6)
        conf = 1.0 / (1e-3 + rel_std)
        return (med * s0).astype(np.float32), conf.astype(np.float32)

    def free(self):
        self.base.free()


def make_model(name):
    """'vggt' -> VGGTWrap(); 'vggt_pv4' -> PerturbWrap(VGGTWrap(), K=4)."""
    m = re.fullmatch(r"(.+)_pv(\d+)", name)
    if m:
        return PerturbWrap(MODEL_REGISTRY[m.group(1)](), K=int(m.group(2)))
    return MODEL_REGISTRY[name]()


# ----------------------------------------------------------------------------- runner
def _key(r):
    sc = r["scan"]
    sc = int(sc) if str(sc).lstrip("-").isdigit() else str(sc)     # DTU scans are ints, CO3D scans are names
    return (r["model"], sc, int(r["seed"]), float(r["radius_frac"]), r["fill"])


def run_benchmark(root, model_names, cfg=None, out_csv="m1_results.csv", model_factory=None,
                  save_examples_dir=None, verbose=True):
    import pandas as pd
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    model_factory = model_factory or make_model
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = {_key(r) for r in prev.to_dict("records")}
        print(f"resuming: {len(done)} rows already in {out_csv}")
    json.dump(cfg, open(os.path.splitext(out_csv)[0] + "_cfg.json", "w"), indent=2)
    is_co3d = cfg.get("dataset", "dtu") == "co3d"
    if is_co3d:
        try:
            import co3d
        except ImportError:          # single-file Kaggle build: co3d.py is concatenated into this module
            import sys
            co3d = sys.modules[__name__]

    def _load(scan):
        if is_co3d:
            views = select_views(cfg["n_views"], total=co3d.n_frames(root, scan), mode=cfg.get("view_mode", "spread"))
            return co3d.load_co3d_scene(root, scan, views)
        views = select_views(cfg["n_views"], mode=cfg.get("view_mode", "spread"))
        return load_scan(root, scan, views, cfg["light"])

    def _distractor_for(scan):
        if not cfg.get("distractor"):
            return None
        pool = cfg["scans"] if is_co3d else DEFAULT_CFG["scans"]
        others = [s for s in pool if s != scan] or [scan]
        if is_co3d:   # CO3D sequences are portrait or landscape; the foreign image must share the orientation
            hw = co3d.scene_hw(root, scan)
            others = [s for s in others if co3d.scene_hw(root, s) == hw] or [scan]
        dscan = others[_stable_seed("distractor", scan) % len(others)]
        v = cfg.get("distractor_view", 0)
        if is_co3d:
            return co3d.load_any_image(root, dscan, v)
        ip = os.path.join(root, "Rectified", f"scan{dscan}_train", f"rect_{v + 1:03d}_{cfg['light']}_r5000.png")
        return cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _predict(model, imgs, out_hw, extra, model_hw):
        if extra is None:
            return model.predict(imgs, model_hw, out_hw)
        d, c = model.predict(np.concatenate([imgs, extra[None]], 0), model_hw, out_hw)
        return d[: len(imgs)], c[: len(imgs)]

    for mname in model_names:
        todo = [(sc, sd, rf, fl) for sc in cfg["scans"] for sd in cfg["seeds"]
                for rf in cfg["radius_fracs"] for fl in cfg["fills"]
                if (mname, sc, sd, rf, fl) not in done]
        if not todo:
            print(f"[{mname}] all done"); continue
        model = model_factory(mname)
        t0 = time.time()
        for scan in cfg["scans"]:
            scan_todo = [t for t in todo if t[0] == scan]
            if not scan_todo:
                continue
            data = _load(scan)
            out_hw = data["depths"].shape[1:]
            model_hw = tuple(data.get("model_hw", cfg["model_hw"]))
            extra = _distractor_for(scan)
            ctrl_d, ctrl_c = _predict(model, data["imgs"], out_hw, extra, model_hw)
            rows = []
            for (_, seed, rf, fill) in scan_todo:
                masks = build_masks(data, seed, rf, cfg)
                imgs_m = apply_fill(data["imgs"], masks["hide"], fill, seed)
                d, c = _predict(model, imgs_m, out_hw, extra, model_hw)
                m = compute_metrics(d, c, data["depths"], masks["U"], masks["S"], cfg, ctrl_d, ctrl_c)
                rows.append(dict(model=mname, scan=scan, seed=seed, radius_frac=rf, fill=fill, **m))
                if save_examples_dir and seed == 0 and fill in ("gray", "inpaint"):
                    _save_example(save_examples_dir, mname, scan, rf, imgs_m, d, c, masks, data["depths"])
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            if verbose:
                g = df[df.fill == "gray"]
                print(f"[{mname}] scan {scan}  auroc={g.auroc.mean():.3f}  cur={g.cur.mean():.3f}  "
                      f"absrel_U={g.absrel_U.mean():.3f} (ctrl {g.absrel_U_ctrl.mean():.3f})  "
                      f"{time.time() - t0:.0f}s")
        if hasattr(model, "free") and not cfg.get("keep_model", False):
            model.free()
    return out_csv


def _save_example(d_out, mname, scan, rf, imgs_m, depth, conf, masks, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(d_out, exist_ok=True)
    i = 0
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    ax[0].imshow(imgs_m[i]); ax[0].set_title(f"input (masked) scan{scan}")
    vm = gt[i][gt[i] > 0]
    ax[1].imshow(np.where(gt[i] > 0, gt[i], np.nan), vmin=vm.min(), vmax=vm.max()); ax[1].set_title("GT depth")
    s = median_scale(depth, gt, masks["S"])
    ax[2].imshow(np.where(gt[i] > 0, depth[i] * s, np.nan), vmin=vm.min(), vmax=vm.max()); ax[2].set_title(f"{mname} depth")
    ax[3].imshow(np.log(conf[i] + 1e-6), cmap="magma"); ax[3].set_title(f"{mname} log conf")
    for a in ax[1:]:
        a.contour(masks["U"][i].astype(float), levels=[0.5], colors="cyan", linewidths=1)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(d_out, f"{mname}_scan{scan}_r{rf}.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- statistics
def cluster_bootstrap_ci(df, col, n_boot=10000, seed=0, alpha=0.05):
    """Resample scans with replacement; each scan contributes its mean over seeds."""
    per_scan = df.groupby("scan")[col].mean().dropna().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_scan), size=(n_boot, len(per_scan)))
    boots = per_scan[idx].mean(1)
    return float(per_scan.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)), len(per_scan)


def paired_permutation(df, col, model_a, model_b, n_perm=20000, seed=0):
    """Sign-flip test on per-scan mean differences (A - B). Returns mean diff and two-sided p."""
    a = df[df.model == model_a].groupby("scan")[col].mean()
    b = df[df.model == model_b].groupby("scan")[col].mean()
    d = (a - b).dropna().values
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    null = (signs * d).mean(1)
    p = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n_perm + 1)
    return float(d.mean()), float(p)


def one_sample_signflip(values, null_value, n_perm=20000, seed=0):
    d = np.asarray(values, float) - null_value
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    null = (signs * d).mean(1)
    return float((np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n_perm + 1))


def summarize(csv_path, cols=("auroc", "auroc_ctrl", "cur", "cur_err", "absrel_S", "absrel_U",
                              "absrel_U_ctrl", "conf_ratio_U")):
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows = []
    for (m, rf, fl), g in df.groupby(["model", "radius_frac", "fill"]):
        r = dict(model=m, radius_frac=rf, fill=fl, n_runs=len(g))
        for c in cols:
            if c in g:
                mean, lo, hi, n = cluster_bootstrap_ci(g, c)
                r[c] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        r["n_scans"] = g.scan.nunique()
        # is AUROC distinguishable from chance (0.5)?
        r["p_auroc_vs_0.5"] = one_sample_signflip(g.groupby("scan")["auroc"].mean().values, 0.5)
        rows.append(r)
    return pd.DataFrame(rows)
