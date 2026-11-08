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

import os, re, glob, json, hashlib, time
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
# MVSNet test split of DTU (22 scans). Disjoint from the 18 benchmark scans and from the 79 training scans of the
# support head; used as the extension set (run family P3-dtuext).
DTU_TEST_SCANS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]


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
        raw = cv2.imread(ip, cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"scan {scan}: cannot read image {ip}")
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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

    def _forward_cams(self, x, dtype):
        """Depth plus the model's own cameras (vggt.utils.pose_enc.pose_encoding_to_extri_intri: OpenCV
        world->camera extrinsics (N,3,4) with the first camera as world, intrinsics (N,3,3) in pixels)."""
        torch = self.torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        if dtype is None:
            out = self.model(x)
        else:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = self.model(x)
        extr, intr = pose_encoding_to_extri_intri(out["pose_enc"].float(), x.shape[-2:])
        E = torch.eye(4, device=x.device).repeat(extr.shape[1], 1, 1)
        E[:, :3, :4] = extr[0].float()
        return out["depth"][0, ..., 0].float(), intr[0].float(), E


class _Pi3CamsMixin:
    """Pi3 and Pi3X output camera_poses (N,4,4, camera->world) and local_points (N,H,W,3, camera frame);
    intrinsics are not output and are fitted per view from local_points (pinhole least squares)."""

    def _forward_cams(self, x, dtype):
        torch = self.torch
        kw = {"with_prior": False} if self.name == "pi3x" else {}
        if dtype is None:
            out = self.model(x[None], **kw)
        else:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = self.model(x[None], **kw)
        lp = out["local_points"][0].float()
        c2w = out["camera_poses"][0].float()
        E = torch.linalg.inv(c2w)
        lp_np = lp.cpu().numpy()
        K = torch.from_numpy(np.stack([_fit_intrinsics(lp_np[i], lp_np.shape[1:3]) for i in range(len(lp_np))]))
        return lp[..., 2], K.float().to(x.device), E


class Pi3XWrap(_Pi3CamsMixin, ModelWrap):
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


class Pi3Wrap(_Pi3CamsMixin, ModelWrap):
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

    # direct forward (scripts/oia_head.py da3_forward_feats and the reprojection detector): x (N,3,H,W) in [0,1] at a
    # multiple of 14; ImageNet normalisation as in DA3's InputProcessor; DepthAnything3.forward takes (B,N,3,H,W) and
    # applies its own autocast. Returns the output Dict with depth (B,N,H,W), depth_conf, extrinsics, intrinsics.
    def _forward_raw(self, x):
        torch = self.torch
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        with torch.no_grad():
            return self.model(((x - mean) / std)[None], export_feat_layers=[])

    def _forward_cams(self, x, dtype):
        out = self._forward_raw(x)
        d = out["depth"][0].float()
        E = out["extrinsics"][0].float()
        if E.shape[-2] == 3:                       # (N,3,4) -> (N,4,4)
            E4 = self.torch.eye(4, device=E.device).repeat(E.shape[0], 1, 1); E4[:, :3, :4] = E; E = E4
        return d, out["intrinsics"][0].float(), E


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


class PerturbFullWrap:
    """Fuller training-free perturbation detector (P3-pv-full, closer to ReVar3R): K forwards of the base model
    under photometric jitter, gaussian blur, a horizontal flip (undone on the output) and a random permutation of
    the view order (undone on the output; it changes the reference frame of models that anchor the world to the
    first view). Each sample is aligned to the first by a least-squares scale on the pixels valid in both (the
    gauge of a depth map is its scale), then the per-pixel spread is the median absolute deviation across samples
    divided by the median depth, confidence = 1 / (eps + spread). Costs K base forwards."""

    def __init__(self, base, K=8, strength=0.15):
        self.base, self.K, self.strength = base, K, strength
        self.name = f"{base.name}_pvf{K}"

    def predict(self, imgs, model_hw, out_hw):
        rng = np.random.default_rng(_stable_seed("perturb-full", self.name))
        N = len(imgs)
        ds = []
        for k in range(self.K):
            x, order, flip = imgs, np.arange(N), False
            if k > 0:
                s = self.strength
                gain = rng.uniform(1 - s, 1 + s, size=(1, 1, 1, 3)).astype(np.float32)
                bias = rng.uniform(-s / 2, s / 2, size=(1, 1, 1, 3)).astype(np.float32)
                gamma = float(rng.uniform(1 - s, 1 + s))
                x = np.clip(np.clip(imgs * gain + bias, 0, 1) ** gamma, 0, 1).astype(np.float32)
                if rng.random() < 0.5:
                    sig = float(rng.uniform(0.5, 1.5))
                    x = np.stack([cv2.GaussianBlur(im, (0, 0), sig) for im in x])
                flip = bool(rng.random() < 0.5)
                if flip:
                    x = x[:, :, ::-1].copy()
                order = rng.permutation(N)
                x = x[order]
            d, _ = self.base.predict(x, model_hw, out_hw)
            inv = np.argsort(order)
            d = d[inv]
            if flip:
                d = d[:, :, ::-1].copy()
            ds.append(d.astype(np.float64))
        ref = ds[0]
        al = [ref]
        for d in ds[1:]:
            ok = (ref > 0) & (d > 0)
            sc = float((ref[ok] * d[ok]).sum() / max((d[ok] ** 2).sum(), 1e-9)) if ok.any() else 1.0
            al.append(d * sc)
        al = np.stack(al)                      # K,N,h,w
        med = np.median(al, 0)
        mad = np.median(np.abs(al - med[None]), 0)
        spread = mad / np.maximum(np.abs(med), 1e-6)
        conf = 1.0 / (1e-3 + spread)
        return med.astype(np.float32), conf.astype(np.float32)

    def free(self):
        self.base.free()


def reproj_support_pairs(depths, K, E, tol=0.03):
    """Per-pair reprojection test of a multi-view depth prediction: sees[i, j] (N,N,h,w) view j sees the predicted
    point of pixel p of view i (in frame, in front of camera j, not behind j's own predicted surface by more than
    tol), agree[i, j] (N,N,h,w) float in [0, 1]: 1 - |z - d_j| / (tol d_j) clipped, 0 where j does not see it. The
    training-free attribution baseline of the provenance heads (P10-prov); reproj_support_score is hits / sees."""
    depths = np.asarray(depths, np.float64)
    N, h, w = depths.shape
    sees = np.zeros((N, N, h, w), bool)
    agree = np.zeros((N, N, h, w), np.float32)
    for i in range(N):
        Pw = backproject(depths[i], K[i], E[i])
        valid_i = depths[i] > 0
        for j in range(N):
            if i == j:
                continue
            u, v, z = project(Pw, K[j], E[j])
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            ok = valid_i & (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            dj = np.zeros((h, w))
            dj[ok] = depths[j][vi[ok], ui[ok]]
            ok &= dj > 0
            rel = (dj - z) / np.maximum(dj, 1e-9)          # < 0: point behind view j's surface (occluded)
            sj = ok & (rel > -tol)
            sees[i, j] = sj
            agree[i, j] = np.where(sj, np.clip(1.0 - np.abs(rel) / tol, 0.0, 1.0), 0.0)
    return sees, agree


def reproj_support_score(depths, K, E, tol=0.03):
    """Training-free support score from cross-view reprojection agreement of a multi-view depth prediction
    (P3-reproj). depths (N,h,w) predicted depth of every view in one common frame (0 = invalid), K (N,3,3)
    intrinsics at that resolution, E (N,4,4) world -> camera.

    For each pixel of view i the predicted point is back-projected and projected into every other view j.
    View j *sees* the point when it lands in frame in front of the camera and is not behind view j's own
    predicted surface (points farther than view j's depth by more than tol are occluded and do not count).
    View j *explains* the point when |z_ij - d_j| / d_j < tol. The score is explains / sees, or 0 when no
    other view sees the point. A point the model places on a surface that its own other views do not contain
    gets a low score; a coherent imagination gets a high one, which is exactly what this baseline is meant
    to test. One forward pass, no learning."""
    sees, agree = reproj_support_pairs(depths, K, E, tol)
    seen = sees.sum(1).astype(np.float64)
    hits = (sees & (agree > 0)).sum(1).astype(np.float64)
    return np.where(seen > 0, hits / np.maximum(seen, 1), 0.0).astype(np.float32)


# ----------------------------------------------------------------------------- true-visibility support (P4-vis)
def support_matrix(depths, K, E, tol=0.03):
    """Per-view support of every pixel (the provenance labels, run family P10-prov). depths (N,h,w) with 0 = unknown,
    K (N,3,3) at that resolution, E (N,4,4) world -> camera. Returns three (N,N,h,w) bool arrays indexed [i, j]:
    seen: view j sees the 3D point of pixel p of view i on its own known surface (|z - d_j| / d_j < tol);
    unknown: the point lands on an unknown depth of view j (hole in the ground truth; the test is undecided);
    inframe: the point lands in frame in front of camera j. The diagonal is False. support_count sums over j."""
    depths = np.asarray(depths, np.float64)
    N, h, w = depths.shape
    seen = np.zeros((N, N, h, w), bool)
    unknown = np.zeros((N, N, h, w), bool)
    inframe = np.zeros((N, N, h, w), bool)
    for i in range(N):
        valid_i = depths[i] > 0
        Pw = backproject(depths[i], K[i], E[i])
        for j in range(N):
            if i == j:
                continue
            u, v, z = project(Pw, K[j], E[j])
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inside = valid_i & (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            dj = np.zeros((h, w))
            dj[inside] = depths[j][vi[inside], ui[inside]]
            known = inside & (dj > 0)
            inframe[i, j] = inside
            unknown[i, j] = inside & ~known
            seen[i, j] = known & (np.abs(dj - z) / np.maximum(dj, 1e-9) < tol)
    return seen, unknown, inframe


def support_count(depths, K, E, tol=0.03):
    """Ground-truth multi-view support of every pixel, the natural-occlusion counterpart of the filled patch.
    depths (N,h,w) ground-truth depth (0 = unknown), K (N,3,3) at that resolution, E (N,4,4) world -> camera.

    For each pixel of view i with known depth, its ground-truth 3D point is projected into every other input view j:
    j *sees* it when it lands in frame in front of camera j on a known depth d_j with |z - d_j| / d_j < tol (j's
    own surface is this point); the test is *unknown* when it lands on an unknown depth of view j (hole in the
    ground truth); otherwise j does not see it (out of frame, behind the camera, occluded by a nearer surface of j,
    or in front of j's surface, which is ground-truth noise). Returns seen (N,h,w) and unknown (N,h,w), int counts
    over the other N-1 views. A pixel seen by no other view has no parallax in the input: its geometry can only come
    from the single-image prior, which is what the benchmark's masking imitates with a patch. No patch is involved.
    The per-view bits behind the counts are support_matrix."""
    seen, unknown, _ = support_matrix(depths, K, E, tol)
    return seen.sum(1).astype(np.int32), unknown.sum(1).astype(np.int32)


def provenance_metrics(P, seen, unknown, inframe, valid, S=None, max_pixels=300000, seed=0):
    """Attribution quality of per-view support scores P (N,N,h,w), P[i, j] = predicted probability that view j
    corroborates pixel p of view i (diagonal ignored), against the ground-truth matrix of support_matrix.
    prov_auroc: over every ordered pair (i, j) and every pixel with known depth in view i whose test in view j is
    decided, does the score rank the views that see the point above the views that do not (out of frame, occluded,
    or on a different surface)? prov_auroc_inframe: the same restricted to views in whose frame the point lands, so
    that frame geometry alone cannot answer it. prov_top1: on the supported pixels S, the fraction whose highest
    scoring view does see the point. prov_count_mae: mean |sum_j P - number of seeing views| over valid pixels.
    Pixels are subsampled to max_pixels per class before the rank statistic."""
    P = np.asarray(P, np.float32)
    N = P.shape[0]
    off = ~np.eye(N, dtype=bool)
    base = off[:, :, None, None] & valid[:, None] & ~unknown
    rng = np.random.default_rng(seed)

    def _sub(x):
        return rng.choice(x, max_pixels, replace=False) if len(x) > max_pixels else x

    out = {"prov_auroc": auroc(_sub(P[base & seen]), _sub(P[base & ~seen])),
           "prov_auroc_inframe": auroc(_sub(P[base & seen]), _sub(P[base & inframe & ~seen]))}
    Pm = np.where(off[:, :, None, None], P, 0.0)
    cnt = seen.sum(1)
    out["prov_count_mae"] = float(np.mean(np.abs(Pm.sum(1) - cnt)[valid])) if valid.any() else float("nan")
    if S is not None and S.any():
        top = np.argmax(np.where(off[:, :, None, None], P, -1.0), axis=1)          # (N,h,w): best other view
        hit = np.take_along_axis(seen, top[:, None], 1)[:, 0]
        out["prov_top1"] = float(hit[S].mean())
    return out


def vis_labels(data, cfg=None):
    """U = pixels with known depth that no other input view sees and no other view leaves unknown (strict: every
    other view either looks away from the point or is occluded from it); S = pixels seen by at least vis_min_seen
    other views, outside a dilation of U. Labels at depth resolution, same shapes and meaning as build_masks."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    depths, Kd, E = data["depths"], data["K_dep"], data["E"]
    seen_m, unk_m, inframe_m = support_matrix(depths, Kd, E, cfg.get("vis_tol", 0.03))
    seen, unknown = seen_m.sum(1).astype(np.int32), unk_m.sum(1).astype(np.int32)
    valid = depths > 0
    U = valid & (seen == 0) & (unknown == 0)
    S = valid & (seen >= cfg.get("vis_min_seen", 2))
    k = max(1, int(cfg.get("vis_dilate_px", 3))) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for i in range(len(U)):
        S[i] &= ~(cv2.dilate(U[i].astype(np.uint8), kernel) > 0)
    out = dict(U=U, S=S, seen=seen, unknown=unknown)
    if cfg.get("vis_pairs"):          # the per-view bits behind the counts (provenance labels, P10-prov)
        out.update(seen_mat=seen_m, unknown_mat=unk_m, inframe_mat=inframe_m, valid=valid)
    return out


def vis_views(n_views, seed, total=49, mode="spread"):
    """The view set of a vis row: the spread selection rotated by the seed, so that seeds give different camera
    subsets and no fixed image region is always the single-view one."""
    base = select_views(n_views, total=total, mode=mode)
    return sorted((v + 7 * int(seed)) % total for v in base)


def _dump_vis(dump_dir, mname, scan, seed, n, depth, conf, gt, lab, imgs=None):
    """Per-pixel record of one vis row (scripts/analyze_dumps.py reads them): the labels of the (scan, seed, views)
    cell once under labels/, the detector's confidence and median-scaled depth under <model>/, both float16."""
    key = f"{scan}__s{seed}__v{n}"
    lp = os.path.join(dump_dir, "labels", key + ".npz")
    if not os.path.exists(lp):
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        h, w = gt.shape[1:]
        rgb = np.stack([cv2.resize((im * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) for im in imgs]) \
            if imgs is not None else np.zeros((len(gt), h, w, 3), np.uint8)
        np.savez_compressed(lp, gt=gt.astype(np.float32), U=lab["U"], S=lab["S"], seen=lab["seen"].astype(np.int8), rgb=rgb)
    mp = os.path.join(dump_dir, mname, key + ".npz")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    np.savez_compressed(mp, conf=conf.astype(np.float32), depth=depth.astype(np.float16))


def run_vis(root, model_names, cfg=None, out_csv="vis_results.csv", model_factory=None, verbose=True):
    """Natural-occlusion protocol (run family P4-vis): clean images, no fill, no patch. For each scan, each number
    of input views in cfg["vis_views"] and each seed (view-set rotation), the model runs once and every detector's
    score is tested on separating S (multi-view-supported pixels) from U (pixels no other view sees), with the same
    metrics as the masking benchmark (AUROC, CUR at 90% kept S, CUR-err, AbsRel on S and U). Rows are keyed like
    benchmark rows with radius_frac 0 and fill "vis<n_views>", so summarize / stats work unchanged."""
    import pandas as pd
    cfg = {**DEFAULT_CFG, "vis_views": [3, 5, 10], "seeds": [0, 1, 2], **(cfg or {})}
    model_factory = model_factory or make_model
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = {_key(r) for r in prev.to_dict("records")}
        print(f"resuming: {len(done)} rows already in {out_csv}")
    json.dump({k: v for k, v in cfg.items()}, open(os.path.splitext(out_csv)[0] + "_cfg.json", "w"), indent=2)
    is_co3d = cfg.get("dataset", "dtu") == "co3d"
    if is_co3d:
        try:
            import co3d
        except ImportError:
            import sys
            co3d = sys.modules[__name__]

    def _load(scan, n, seed):
        if is_co3d:
            views = vis_views(n, seed, total=co3d.n_frames(root, scan), mode=cfg.get("view_mode", "spread"))
            return co3d.load_co3d_scene(root, scan, views)
        return load_scan(root, scan, vis_views(n, seed, mode=cfg.get("view_mode", "spread")), cfg["light"])

    # the ground-truth support labels of a cell do not depend on the detector: computed once per (scene, seed, views)
    # and cached on disk next to the CSV (the support count is the slow part of a vis row, not the backbone forward)
    tag = f"{cfg.get('vis_tol', 0.03)}_{cfg.get('vis_min_seen', 2)}_{cfg.get('vis_dilate_px', 3)}_{cfg.get('view_mode', 'spread')}_{cfg.get('light', 3)}"
    cache_dir = cfg.get("label_cache") or os.path.join(os.path.dirname(os.path.abspath(out_csv)), "labels_cache")

    def _labels(scan, n, seed, data):
        p = os.path.join(cache_dir, f"{scan}__s{seed}__v{n}__{tag}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return dict(U=z["U"], S=z["S"], seen=z["seen"].astype(np.int32), unknown=z["unknown"].astype(np.int32))
        lab = vis_labels(data, cfg)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(p, U=lab["U"], S=lab["S"], seen=lab["seen"].astype(np.int16), unknown=lab["unknown"].astype(np.int16))
        return lab

    for mname in model_names:
        todo = [(sc, sd, n) for sc in cfg["scans"] for sd in cfg["seeds"] for n in cfg["vis_views"]
                if (mname, sc, sd, 0.0, f"vis{n}") not in done]
        if not todo:
            print(f"[{mname}] all done"); continue
        model = model_factory(mname)
        t0 = time.time()
        for scan in cfg["scans"]:
            rows = []
            for (_, seed, n) in [t for t in todo if t[0] == scan]:
                data = _load(scan, n, seed)
                lab = _labels(scan, n, seed, data)
                U, S = lab["U"], lab["S"]
                nU, nS = int(U.sum()), int(S.sum())
                if nU < cfg.get("vis_min_pixels", 50) or nS < cfg.get("vis_min_pixels", 50):
                    rows.append(dict(model=mname, scan=scan, seed=seed, radius_frac=0.0, fill=f"vis{n}", n_views=n,
                                     n_U=nU, n_S=nS, skipped=True))
                    continue
                out_hw = data["depths"].shape[1:]
                model_hw = tuple(data.get("model_hw", cfg["model_hw"]))
                pm = {}
                if cfg.get("prov") and getattr(model, "has_pairs", False):
                    # provenance (P10-prov): per-view support scores against the ground-truth per-view matrix
                    d, c, P = model.predict_pairs(data["imgs"], model_hw, out_hw)
                    sm, um, fm = support_matrix(data["depths"], data["K_dep"], data["E"], cfg.get("vis_tol", 0.03))
                    pm = provenance_metrics(P, sm, um, fm, data["depths"] > 0, S)
                    if cfg.get("dump_pairs"):
                        pp = os.path.join(cfg["dump_pairs"], mname, f"{scan}__s{seed}__v{n}.npz")
                        os.makedirs(os.path.dirname(pp), exist_ok=True)
                        np.savez_compressed(pp, P=P.astype(np.float16), seen=sm, unknown=um, inframe=fm,
                                            rgb=np.stack([cv2.resize((im * 255).astype(np.uint8), (out_hw[1], out_hw[0]),
                                                                     interpolation=cv2.INTER_AREA) for im in data["imgs"]]),
                                            gt=data["depths"].astype(np.float32), U=U, S=S, conf=c.astype(np.float32),
                                            depth=d.astype(np.float16))
                else:
                    d, c = model.predict(data["imgs"], model_hw, out_hw)
                m = compute_metrics(d, c, data["depths"], U, S, cfg)
                m.update(pm)
                valid = int((data["depths"] > 0).sum())
                if cfg.get("dump_dir"):          # per-pixel scores for the risk-coverage / calibration analysis
                    _dump_vis(cfg["dump_dir"], mname, scan, seed, n, d * m["scale"], c, data["depths"], lab, data["imgs"])
                rows.append(dict(model=mname, scan=scan, seed=seed, radius_frac=0.0, fill=f"vis{n}", n_views=n,
                                 frac_U=nU / max(valid, 1), frac_S=nS / max(valid, 1), skipped=False, **m))
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            if verbose:
                g = df[~df.skipped.astype(bool)]
                prov = f"prov_auroc={g.prov_auroc.mean():.3f} top1={g.prov_top1.mean():.3f}  " if "prov_auroc" in g and len(g) else ""
                print(f"[{mname}] scan {scan}  auroc={g.auroc.mean():.3f}  cur={g.cur.mean():.3f}  {prov}"
                      f"absrel_U={g.absrel_U.mean():.3f} (S {g.absrel_S.mean():.3f})  fracU={g.frac_U.mean():.3f}  "
                      f"{time.time() - t0:.0f}s" if len(g) else f"[{mname}] scan {scan}: too few U or S pixels")
        if hasattr(model, "free") and not cfg.get("keep_model", False):
            model.free()
    return out_csv


# ----------------------------------------------------------------------------- matched-patch protocol (P11-matched)
def _cam_dir(E):
    """Viewing direction of a world->camera extrinsic, in world coordinates."""
    return E[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def _view_angles(E):
    """(N,N) angle in degrees between the viewing directions of every pair of cameras."""
    d = np.stack([_cam_dir(e) for e in E])
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(d @ d.T, -1.0, 1.0)))


def _hide_from_inside(inside_i, H, W, sx, sy):
    up = cv2.resize(inside_i.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(up.astype(np.float32), (0, 0), sigmaX=max(sx, sy) * 0.6) > 0.5


def build_masks_matched(data, seed, radius_frac, cfg, evidence="1"):
    """The matched-patch protocol (run family P11-matched; Gates 6 and 7). Three spheres of the same radius on the
    ground-truth surface, filled the same way, so that from a single image their patches are identical:
      A  hidden in every input view: its pixels are imagined (U), as in build_masks;
      B  hidden in every view except an evidence set Ev of `evidence` views (1, 2 or half of them, drawn among the
         views that see it): its filled pixels in the hidden views stay visible in a clean view, so they are
         observed (Sm) wherever a view of Ev actually sees the pixel's ground-truth point (support test);
      D  hidden only in the views of Ev: observed with many-view evidence, and it keeps the number of patches per
         view at two, so counting patches across views cannot tell A from B.
    B and D match A's reference-view area within `match_area` and its reference depth within `match_depth`.
    Returns hide (N,H,W) at image resolution, U, S (clean supported, outside a dilation of every patch), Sm, the
    per-pixel evidence count ev_count (clean views that see the pixel) and ev_par (angle to the nearest of them,
    degrees), the evidence views and the sphere parameters. Deterministic in (scan, seed, radius, evidence).
    Raises RuntimeError when no matched placement exists for this (scene, seed)."""
    rng = np.random.default_rng(_stable_seed("matched", data["scan"], seed, radius_frac, evidence))
    depths, Kd, E = data["depths"], data["K_dep"], data["E"]
    N, h, w = depths.shape
    H, W = data["imgs"].shape[1:3]
    sy, sx = H // h, W // w
    Pws = [backproject(depths[i], Kd[i], E[i]) for i in range(N)]
    valid = depths > 0
    ref = 0
    cm = data.get("center_mask")
    sel = valid[ref] & (cm[ref] if cm is not None else True)
    pts_ref, dep_ref = Pws[ref][sel], depths[ref][sel]
    if len(pts_ref) < 100:
        raise ValueError(f"scan {data['scan']}: too few valid GT pixels in ref view")
    lo, hi = np.percentile(pts_ref, 2, axis=0), np.percentile(pts_ref, 98, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    radius = radius_frac * diag
    n_ev = {"1": 1, "2": 2, "half": max(1, N // 2)}[str(evidence)]
    match_area, match_depth = cfg.get("match_area", 0.3), cfg.get("match_depth", 0.15)
    min_sep = cfg.get("match_min_sep", 2.5) * radius
    seen_m, _, _ = support_matrix(depths, Kd, E, cfg.get("vis_tol", 0.03))        # (N,N,h,w): j sees pixel of i
    ang = _view_angles(E)

    def inside_of(c):
        return np.stack([(np.linalg.norm(Pws[i] - c, axis=-1) < radius) & valid[i] for i in range(N)])

    P_all = np.stack(Pws).reshape(-1, 3)                      # (N*h*w, 3): one distance pass per candidate sphere
    valid_flat = valid.reshape(-1)

    def inside_of(c):
        return ((np.linalg.norm(P_all - c, axis=-1) < radius) & valid_flat).reshape(N, h, w)

    def inside_frac_ok(ins):
        return cfg["min_U_frac"] <= ins[ref].sum() / max(valid[ref].sum(), 1) <= cfg["max_U_frac"]

    pool_size = int(cfg.get("match_pool", 240))

    def candidate_pool(dA):
        """Depth-matched reference points, shuffled and capped; their inside masks are computed once each."""
        idx = np.where(np.abs(dep_ref - dA) / dA < match_depth)[0]
        rng.shuffle(idx)
        return list(idx[:pool_size]), {}

    def matched_candidate(pool, cache, exclude_centres, sep, tol_area):
        """A sphere with A's reference area and depth, at least `sep` from the given centres (centre distance, so
        2 radii means tangent patches). Returns (centre, inside) or None; inside masks are cached per candidate."""
        for k in pool:
            c = pts_ref[k]
            if any(np.linalg.norm(c - e) < sep for e in exclude_centres):
                continue
            if k not in cache:
                cache[k] = inside_of(c)
            ins = cache[k]
            a = int(ins[ref].sum())
            if a == 0 or abs(a - areaA) / max(areaA, 1) > tol_area:
                continue
            return k, c, ins
        return None

    # sphere A as in build_masks, then B with its evidence set from one bounded candidate pool; when A leaves no
    # room for a matched B (small objects), A is drawn again a few times, with the separation and the area
    # tolerance relaxing one step before giving up
    found = False
    for _a in range(int(cfg.get("match_A_tries", 6))):
        inA = None
        for _ in range(cfg["max_tries"]):
            k = rng.integers(len(pts_ref))
            cA, dA = pts_ref[k], float(dep_ref[k])
            ins = inside_of(cA)
            if inside_frac_ok(ins):
                inA = ins
                break
        if inA is None:
            raise RuntimeError(f"scan {data['scan']} seed {seed} r{radius_frac}: no valid sphere A after {cfg['max_tries']} tries")
        areaA = int(inA[ref].sum())
        pool, cache = candidate_pool(dA)
        for sep_f, tol_area in ((2.5, match_area), (2.2, match_area), (2.2, match_area * 1.5)):
            used = []
            while True:
                cand = matched_candidate([k for k in pool if k not in used], cache, [cA], sep_f * radius, tol_area)
                if cand is None:
                    break
                kB, cB, inB = cand
                used.append(kB)
                sees_B = [i for i in range(N) if i != ref and inB[i].sum() >= 0.2 * areaA]
                if len(sees_B) >= n_ev + 1:
                    Ev = sorted(int(x) for x in rng.choice(sees_B, n_ev, replace=False))
                    found = True
                    break
                if len(used) >= 8:
                    break
            if found:
                break
        if found:
            break
    if not found:
        raise RuntimeError(f"scan {data['scan']} seed {seed} r{radius_frac} ev{evidence}: no matched spheres A and B")
    # sphere D: hidden only in the evidence views
    candD = (matched_candidate(pool, cache, [cA, cB], 2.5 * radius, match_area)
             or matched_candidate(pool, cache, [cA, cB], 2.2 * radius, match_area * 1.5))
    has_D = candD is not None
    if has_D:
        _, cD, inD = candD
    notEv = [i for i in range(N) if i not in Ev]

    hide = np.zeros((N, H, W), bool)
    U = np.zeros((N, h, w), bool)
    S = np.zeros((N, h, w), bool)
    Sm = np.zeros((N, h, w), bool)
    ev_count = np.zeros((N, h, w), np.int16)
    ev_par = np.full((N, h, w), np.nan, np.float32)
    kd = cfg["dilate_px"] * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))

    def core(up, ins):
        blk = up.reshape(h, sy, w, sx).mean(axis=(1, 3))
        return (blk >= 0.999) & valid[i] & ins

    for i in range(N):
        upA = _hide_from_inside(inA[i], H, W, sx, sy)
        upB = _hide_from_inside(inB[i], H, W, sx, sy) if i not in Ev else np.zeros((H, W), bool)
        upD = _hide_from_inside(inD[i], H, W, sx, sy) if (has_D and i in Ev) else np.zeros((H, W), bool)
        hide[i] = upA | upB | upD
        U[i] = core(upA, inA[i])
        # observed-but-filled pixels: B in the hidden views (evidence = Ev), D in the evidence views (evidence = the rest)
        if i not in Ev:
            cB_i, clean = core(upB, inB[i]), Ev
        elif has_D:
            cB_i, clean = core(upD, inD[i]), notEv
        else:
            cB_i, clean = np.zeros((h, w), bool), []
        if cB_i.any() and clean:
            seen_by = np.stack([seen_m[i, j] for j in clean])                      # (n_clean, h, w)
            cnt = seen_by.sum(0)
            Sm[i] = cB_i & (cnt > 0)
            ev_count[i][Sm[i]] = cnt[Sm[i]]
            a = np.where(seen_by, np.array([ang[i, j] for j in clean])[:, None, None], np.inf).min(0)
            ev_par[i][Sm[i]] = a[Sm[i]]
        dil = cv2.dilate(hide[i].astype(np.uint8), kernel) > 0
        dil_blk = dil.reshape(h, sy, w, sx).mean(axis=(1, 3))
        S[i] = (dil_blk <= 0.0) & valid[i]
    return dict(hide=hide, U=U, S=S, Sm=Sm, ev_count=ev_count, ev_par=ev_par, evidence_views=Ev, has_D=has_D,
                center=cA, center_B=cB, center_D=(cD if has_D else None), radius=radius, diag=diag,
                area_A=areaA, area_B=int(inB[ref].sum()), area_D=(int(inD[ref].sum()) if has_D else 0),
                depth_A=dA)


def compute_metrics_matched(pred_depth, conf, gt, masks, cfg, ctrl_depth=None, ctrl_conf=None, min_bucket=30):
    """The usual metrics (U against clean S) plus the matched columns: auroc_m / cur_m score U against the
    filled-but-observed pixels Sm (same patch, same fill), auroc_patch scores clean S against Sm (how much the
    score reacts to a patch that hides nothing), and auroc_m per evidence bucket: Sm pixels seen by exactly one
    clean view at low parallax (below the cell's median angle), one view at high parallax, two views, three or more."""
    U, S, Sm = masks["U"], masks["S"], masks["Sm"]
    out = compute_metrics(pred_depth, conf, gt, U, S, cfg, ctrl_depth, ctrl_conf)
    cU, cS, cM = conf[U], conf[S], conf[Sm]
    out["n_Sm"] = int(Sm.sum())
    out["auroc_m"] = auroc(cM, cU)
    if len(cM):
        tau_m = np.quantile(cM, 1 - cfg["tau_keep"])
        out["cur_m"] = float(np.mean(cU >= tau_m))
        out["keep_S_at_tau_m"] = float(np.mean(cS >= tau_m)) if len(cS) else float("nan")
    else:
        out["cur_m"] = out["keep_S_at_tau_m"] = float("nan")
    out["auroc_patch"] = auroc(cS, cM)
    s = out["scale"]
    out["absrel_Sm"], _ = depth_metrics(pred_depth * s, gt, Sm)
    ev, par = masks["ev_count"], masks["ev_par"]
    one = Sm & (ev == 1)
    med = float(np.nanmedian(par[one])) if one.any() else float("nan")
    buckets = {"ev1lo": one & (par <= med), "ev1hi": one & (par > med), "ev2": Sm & (ev == 2), "ev3p": Sm & (ev >= 3)}
    for k, m in buckets.items():
        n = int(m.sum())
        out[f"n_{k}"] = n
        out[f"auroc_m_{k}"] = auroc(conf[m], cU) if n >= min_bucket and len(cU) else float("nan")
    out["par_median_ev1"] = med
    out["n_evidence_views"] = len(masks["evidence_views"])
    out["has_D"] = bool(masks["has_D"])
    if ctrl_conf is not None:
        out["auroc_m_ctrl"] = auroc(ctrl_conf[Sm], ctrl_conf[U])
    return out


def run_matched(root, model_names, cfg=None, out_csv="matched_results.csv", model_factory=None, verbose=True):
    """The matched-patch protocol for a list of detectors (P11-matched). One row per (model, scene, seed, radius,
    fill, evidence); rows are keyed like benchmark rows with fill "m<evidence>_<fill>" so that summarize and the
    statistics work unchanged. Masks are cached per (scene, seed, radius, evidence) next to the CSV."""
    import pandas as pd
    cfg = {**DEFAULT_CFG, "seeds": [0, 1], "radius_fracs": [0.18], "fills": ["gray", "blur", "noise"],
           "evidence": ["1", "2", "half"], **(cfg or {})}
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
        except ImportError:
            import sys
            co3d = sys.modules[__name__]
    cache_dir = cfg.get("label_cache") or os.path.join(os.path.dirname(os.path.abspath(out_csv)), "masks_cache")

    def _load(scan):
        if is_co3d:
            views = select_views(cfg["n_views"], total=co3d.n_frames(root, scan), mode=cfg.get("view_mode", "spread"))
            return co3d.load_co3d_scene(root, scan, views)
        return load_scan(root, scan, select_views(cfg["n_views"], mode=cfg.get("view_mode", "spread")), cfg["light"])

    def _masks(scan, data, seed, rf, ev):
        p = os.path.join(cache_dir, f"{scan}__s{seed}__r{rf}__e{ev}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            if bool(z["failed"]):
                return None
            return dict(hide=z["hide"], U=z["U"], S=z["S"], Sm=z["Sm"], ev_count=z["ev_count"], ev_par=z["ev_par"],
                        evidence_views=list(z["evidence_views"]), has_D=bool(z["has_D"]))
        os.makedirs(cache_dir, exist_ok=True)
        try:
            m = build_masks_matched(data, seed, rf, cfg, ev)
        except RuntimeError as e:
            print(f"[masks] {scan} seed {seed} ev {ev}: {e}")
            np.savez_compressed(p, failed=True)
            return None
        np.savez_compressed(p, failed=False, hide=m["hide"], U=m["U"], S=m["S"], Sm=m["Sm"], ev_count=m["ev_count"],
                            ev_par=m["ev_par"], evidence_views=np.array(m["evidence_views"]), has_D=m["has_D"])
        return m

    for mname in model_names:
        todo = [(sc, sd, rf, ev, fl) for sc in cfg["scans"] for sd in cfg["seeds"] for rf in cfg["radius_fracs"]
                for ev in cfg["evidence"] for fl in cfg["fills"] if (mname, sc, sd, rf, f"m{ev}_{fl}") not in done]
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
            ctrl_d, ctrl_c = model.predict(data["imgs"], model_hw, out_hw)
            rows = []
            for (_, seed, rf, ev, fill) in scan_todo:
                masks = _masks(scan, data, seed, rf, ev)
                key = dict(model=mname, scan=scan, seed=seed, radius_frac=rf, fill=f"m{ev}_{fill}", evidence=ev, fill_kind=fill)
                if masks is None or masks["Sm"].sum() < cfg.get("min_Sm", 30) or masks["U"].sum() < cfg.get("min_Sm", 30):
                    rows.append(dict(**key, skipped=True)); continue
                imgs_m = apply_fill(data["imgs"], masks["hide"], fill, seed)
                d, c = model.predict(imgs_m, model_hw, out_hw)
                m = compute_metrics_matched(d, c, data["depths"], masks, cfg, ctrl_d, ctrl_c)
                rows.append(dict(**key, skipped=False, **m))
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            if verbose:
                g = df[~df.skipped.astype(bool)]
                if len(g):
                    print(f"[{mname}] {scan}  auroc={g.auroc.mean():.3f} auroc_m={g.auroc_m.mean():.3f} "
                          f"cur={g.cur.mean():.3f} cur_m={g.cur_m.mean():.3f} patch={g.auroc_patch.mean():.3f}  "
                          f"ev1lo={g.auroc_m_ev1lo.mean():.3f}  {time.time() - t0:.0f}s", flush=True)
                else:
                    print(f"[{mname}] {scan}: every cell skipped", flush=True)
        if hasattr(model, "free") and not cfg.get("keep_model", False):
            model.free()
    return out_csv


def _fit_intrinsics(local_points, hw):
    """Pinhole intrinsics from camera-frame points (h,w,3) by least squares: u = fx * x/z + cx, v = fy * y/z + cy."""
    h, w = hw
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    ok = z > 1e-6
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    A = np.stack([(x / np.maximum(z, 1e-6))[ok], np.ones(ok.sum())], 1)
    fx, cx = np.linalg.lstsq(A, u[ok], rcond=None)[0]
    B = np.stack([(y / np.maximum(z, 1e-6))[ok], np.ones(ok.sum())], 1)
    fy, cy = np.linalg.lstsq(B, v[ok], rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


class ReprojWrap:
    """Training-free detector (P3-reproj): confidence = cross-view reprojection agreement of the model's own
    multi-view prediction under the model's own predicted cameras (reproj_support_score). One backbone forward,
    no learning. The depth is the base model's depth; only the confidence changes. Needs a base wrapper with
    `_forward_cams(x, dtype) -> depth (N,h,w), K (N,3,3), E (N,4,4)` at model resolution (VGGT, Pi3, Pi3X).
    Written against the public model APIs; not yet run on a GPU (see EXPERIMENTS.md, P3-reproj)."""

    def __init__(self, base, tol=0.03):
        if not hasattr(base, "_forward_cams"):
            raise ValueError(f"{base.name} does not expose predicted cameras; reprojection detector unavailable")
        self.base, self.tol = base, tol
        self.name = f"{base.name}_reproj"

    def predict(self, imgs, model_hw, out_hw):
        torch = self.base.torch
        x = self.base._prep(imgs, model_hw)
        with torch.no_grad():
            d, K, E = self.base._forward_cams(x, self.base.dtype)
            if not torch.isfinite(d).all():
                print(f"[{self.name}] non-finite depth in {self.base.dtype}, retrying fp32")
                d, K, E = self.base._forward_cams(x, None)
        d = d.float().cpu().numpy().astype(np.float32)
        K = np.asarray(K.float().cpu().numpy(), np.float64).copy()
        E = np.asarray(E.float().cpu().numpy(), np.float64)
        # score at the label resolution: cheaper, and the metrics live there anyway
        d_s = np.stack([cv2.resize(x_, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA) for x_ in d])
        K[:, 0] *= out_hw[1] / model_hw[1]
        K[:, 1] *= out_hw[0] / model_hw[0]
        conf = reproj_support_score(d_s, K, E, self.tol)
        return d_s.astype(np.float32), conf.astype(np.float32)

    has_pairs = True

    def predict_pairs(self, imgs, model_hw, out_hw):
        """Depth, support score and the per-view agreement (N,N,h,w) of the model's own reprojection test: the
        training-free attribution baseline of the provenance heads."""
        torch = self.base.torch
        x = self.base._prep(imgs, model_hw)
        with torch.no_grad():
            d, K, E = self.base._forward_cams(x, self.base.dtype)
            if not torch.isfinite(d).all():
                d, K, E = self.base._forward_cams(x, None)
        d = d.float().cpu().numpy().astype(np.float32)
        K = np.asarray(K.float().cpu().numpy(), np.float64).copy()
        E = np.asarray(E.float().cpu().numpy(), np.float64)
        d_s = np.stack([cv2.resize(x_, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA) for x_ in d])
        K[:, 0] *= out_hw[1] / model_hw[1]
        K[:, 1] *= out_hw[0] / model_hw[0]
        sees, agree = reproj_support_pairs(d_s, K, E, self.tol)
        seen = sees.sum(1).astype(np.float64)
        hits = (sees & (agree > 0)).sum(1).astype(np.float64)
        conf = np.where(seen > 0, hits / np.maximum(seen, 1), 0.0).astype(np.float32)
        return d_s.astype(np.float32), conf, agree

    def free(self):
        self.base.free()


class HeuristicWrap:
    """Hand-crafted controls for the true-visibility protocol (P6-heur): single-view pixels concentrate at the image
    border and at depth discontinuities, so a reviewer may ask whether a head that finds them has learned anything
    beyond those two cues. Two training-free scores on the base model's own output, no learning, no cameras:
      <base>_border   confidence = distance of the pixel to the nearest image border (normalised by the shorter side)
      <base>_dedge    confidence = minus the relative depth gradient of the predicted depth, smoothed (Gaussian, 5 px):
                      low at occlusion boundaries, high on smooth surfaces
    The depth is the base model's depth; only the confidence changes."""

    def __init__(self, base, kind):
        assert kind in ("border", "dedge"), kind
        self.base, self.kind = base, kind
        self.name = f"{base.name}_{kind}"

    def predict(self, imgs, model_hw, out_hw):
        d, _ = self.base.predict(imgs, model_hw, out_hw)
        N, h, w = d.shape
        if self.kind == "border":
            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx)).astype(np.float32) / min(h, w)
            conf = np.repeat(dist[None], N, 0)
        else:
            conf = np.zeros_like(d)
            for i in range(N):
                di = cv2.GaussianBlur(d[i], (5, 5), 0)
                gx = cv2.Sobel(di, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(di, cv2.CV_32F, 0, 1, ksize=3)
                g = np.sqrt(gx * gx + gy * gy) / np.maximum(np.abs(di), 1e-6)
                conf[i] = -cv2.GaussianBlur(g, (5, 5), 0)
        return d.astype(np.float32), conf.astype(np.float32)

    def free(self):
        self.base.free()


class CrossModelWrap:
    """Cross-model disagreement (P8-xmodel): the one test the benchmark's filled patch has not had. Two backbones run
    on the same views; view by view, B's depth is aligned to A's by the median ratio over all valid pixels (no label
    is used) and the confidence is minus the relative disagreement |d_A - s d_B| / d_A. If the models invent
    different surfaces where no view shows one but agree where views do, this separates the two; if they invent the
    same surface, inventions are consistent across models as well as across views, which is the finding either way.
    Depth is A's. Model name '<a>_x<b>', e.g. vggt_xpi3x."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.name = f"{a.name}_x{b.name}"

    def predict(self, imgs, model_hw, out_hw):
        da, _ = self.a.predict(imgs, model_hw, out_hw)
        db, _ = self.b.predict(imgs, model_hw, out_hw)
        conf = np.zeros_like(da)
        for i in range(len(da)):
            ok = (da[i] > 0) & (db[i] > 0) & np.isfinite(da[i]) & np.isfinite(db[i])
            s = float(np.median(da[i][ok] / db[i][ok])) if ok.sum() > 10 else 1.0
            conf[i] = -np.abs(da[i] - s * db[i]) / np.maximum(np.abs(da[i]), 1e-6)
        return da.astype(np.float32), conf.astype(np.float32)

    def free(self):
        self.a.free(); self.b.free()


def make_model(name):
    """'vggt' -> VGGTWrap(); 'vggt_pv4' -> PerturbWrap(VGGTWrap(), K=4); 'vggt_reproj' -> ReprojWrap(VGGTWrap());
    'vggt_border' / 'vggt_dedge' -> HeuristicWrap(VGGTWrap(), kind); 'vggt_xpi3x' -> CrossModelWrap(VGGTWrap(), Pi3XWrap())."""
    m = re.fullmatch(r"(vggt|pi3x|pi3|da3|mapanything)_x(vggt|pi3x|pi3|da3|mapanything)", name)
    if m:
        return CrossModelWrap(MODEL_REGISTRY[m.group(1)](), MODEL_REGISTRY[m.group(2)]())
    m = re.fullmatch(r"(.+)_pvf(\d+)", name)
    if m:
        return PerturbFullWrap(MODEL_REGISTRY[m.group(1)](), K=int(m.group(2)))
    m = re.fullmatch(r"(.+)_pv(\d+)", name)
    if m:
        return PerturbWrap(MODEL_REGISTRY[m.group(1)](), K=int(m.group(2)))
    m = re.fullmatch(r"(.+)_reproj", name)
    if m:
        return ReprojWrap(MODEL_REGISTRY[m.group(1)]())
    m = re.fullmatch(r"(.+)_(border|dedge)", name)
    if m:
        return HeuristicWrap(MODEL_REGISTRY[m.group(1)](), m.group(2))
    if re.fullmatch(r".+_oia\w*", name):          # support head variants (scripts/oia_head.py)
        try:
            from oia_head import make_oia
        except ImportError:                        # single-file build: oia_head.py is concatenated below
            make_oia = globals()["make_oia"]
        w = make_oia(name, MODEL_REGISTRY)
        if w is not None:
            return w
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
