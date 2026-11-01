"""Observe, Imagine, Abstain: the support head (run family P3-oia).

A small head on the *frozen* features of a feed-forward 3D model that predicts, per pixel, whether the geometry
the model outputs there is supported by the input views (observed) or filled in from the prior (imagined).
The confidence the benchmark scores is 1 - P(imagined); a middle band of that probability is "abstain".

Training is self-supervised and needs no ground truth: the backbone's own clean prediction (depth + cameras) is
used to place a 3D-consistent sphere, the sphere is hidden in every view (hallu3d.build_masks / apply_fill, the
benchmark mechanism applied to predictions instead of ground truth), and the head learns to find the pixels that
were hidden from the features of the masked forward pass. Clean inputs with an all-observed target are mixed in
so the head is quiet when nothing is hidden. Training scenes are the DTU *training* split (disjoint from the 18
benchmark scans) and the benchmark is never touched during training; CO3Dv2 is evaluated zero-shot.

Variants (all share this file, selected by the model name passed to hallu3d.make_model):
  <base>_oia        conv head on the multi-view (aggregated) features, self-supervised labels      (main)
  <base>_oiafuse    same head, score fused with the native confidence (geometric mean of ranks)      (ablation)
  <base>_oialin     linear probe on the same features                                               (ablation)
  <base>_oiapatch   conv head on the per-image encoder features only: a patch detector, no multi-view (control)
  <base>_oiagt      conv head trained with ground-truth depth for mask placement instead of self      (ablation)
Checkpoints: $OIA_CKPT_<NAME> (e.g. OIA_CKPT_VGGT_OIA) or $OIA_CKPT_DIR/<name>.pt (default checkpoints/).
Backbones with feature hooks: vggt (aggregator tokens, layers 4/11/17/23, 2048-d), pi3 and pi3x (decoder tokens
fed to the confidence head, 2048-d). Written against the public model code; see scripts/test_oia.py for the
CPU test with a fake backbone and scripts/run_h100.py for the GPU plan.

Train:  python scripts/oia_head.py train --base vggt --dtu <dtu_root> [--variant oia|oialin|oiapatch|oiagt]
                                          [--steps 3000] [--seed 0] [--out checkpoints/vggt_oia.pt]
"""
import argparse
import contextlib
import json
import os
import re
import time

import cv2
import numpy as np

try:
    import hallu3d as H
except ImportError:                 # single-file build: hallu3d.py is concatenated above this file
    import sys
    H = sys.modules[__name__]

PATCH = 14
# MVSNet training split of DTU (79 scans). Disjoint from hallu3d.DEFAULT_CFG["scans"] (the 18 validation scans).
DTU_TRAIN_SCANS = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44, 45, 46, 47, 50, 51, 52, 53, 55, 57,
                   58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72, 74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                   96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                   121, 122, 123, 124, 125, 126, 127, 128]
assert not set(DTU_TRAIN_SCANS) & set(H.DEFAULT_CFG["scans"]), "training scans overlap the benchmark scans"

VARIANTS = {
    #           feats     arch      labels
    "oia":     ("agg",   "conv",   "self"),
    "oiafuse": ("agg",   "conv",   "self"),     # evaluation-time fusion; shares the _oia checkpoint
    "oialin":  ("agg",   "linear", "self"),
    "oiapatch": ("patch", "conv",  "self"),
    "oiagt":   ("agg",   "conv",   "gt"),
}
TRAIN_DEFAULTS = dict(steps=3000, lr=1e-3, wd=1e-2, warmup=100, n_views=10, radius_range=(0.08, 0.24),
                      fills=["gray", "blur", "noise"], p_clean=0.15, width=256, conf_floor_q=0.2,
                      val_every=250, log_every=25, val_scans=[2, 6], val_seeds=[0, 1], random_light=True)


# ----------------------------------------------------------------------------- feature hooks per backbone
def _grid(t, N, ph, pw):
    """(N, P, C) tokens -> (N, C, ph, pw) float map."""
    return t.float().reshape(N, ph, pw, -1).permute(0, 3, 1, 2).contiguous()


def _autocast(torch, dtype):
    return torch.amp.autocast("cuda", dtype=dtype) if (dtype is not None and torch.cuda.is_available()) else contextlib.nullcontext()


def vggt_forward_feats(wrap, x, dtype, feats="agg", layers=(4, 11, 17, 23)):
    """x (N,3,H,W) in [0,1]. Returns depth (N,H,W), native conf (N,H,W), feature maps [(N,C,ph,pw)...],
    intrinsics (N,3,3) and world->camera extrinsics (N,4,4), all torch on the model device."""
    torch, model = wrap.torch, wrap.model
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    N, ph, pw = x.shape[0], x.shape[-2] // PATCH, x.shape[-1] // PATCH
    with _autocast(torch, dtype):
        tokens, ps = model.aggregator(x[None])
    with torch.amp.autocast("cuda", enabled=False):
        depth, conf = model.depth_head(tokens, images=x[None], patch_start_idx=ps)
        pose_enc = model.camera_head(tokens)[-1]
    extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), x.shape[-2:])
    E = torch.eye(4, device=x.device).repeat(N, 1, 1)
    E[:, :3, :4] = extr[0].float()
    if feats == "agg":
        F = [_grid(tokens[l][0][:, ps:], N, ph, pw) for l in layers]
    elif feats == "patch":           # per-image DINOv2 tokens before any cross-view attention (the frame-only control)
        agg = model.aggregator
        xn = (x - agg._resnet_mean.reshape(1, 3, 1, 1)) / agg._resnet_std.reshape(1, 3, 1, 1)
        with _autocast(torch, dtype):
            pt = agg.patch_embed(xn)
        if isinstance(pt, dict):
            pt = pt["x_norm_patchtokens"]
        F = [_grid(pt[:, -ph * pw:], N, ph, pw)]
    else:
        raise ValueError(feats)
    return depth[0, ..., 0].float(), conf[0].float(), F, intr[0].float(), E


def _find_module(model, name):
    if hasattr(model, name):
        return getattr(model, name)
    for n, m in model.named_modules():
        if n.endswith(name):
            return m
    raise AttributeError(f"{type(model).__name__} has no module {name!r}; run with --list-modules")


def pi3_forward_feats(wrap, x, dtype, feats="agg"):
    """Pi3 / Pi3X: the 2048-d decoder tokens that feed the confidence head (captured with a pre-hook on
    conf_decoder), or the DINOv2 encoder tokens for the frame-only control."""
    torch, model = wrap.torch, wrap.model
    N, ph, pw = x.shape[0], x.shape[-2] // PATCH, x.shape[-1] // PATCH
    kw = {"with_prior": False} if wrap.name == "pi3x" else {}
    cap, hooks = {}, []
    hooks.append(_find_module(model, "conf_decoder").register_forward_pre_hook(lambda m, a: cap.__setitem__("hidden", a[0])))
    if feats == "patch":
        hooks.append(_find_module(model, "encoder").register_forward_hook(
            lambda m, a, o: cap.__setitem__("enc", o["x_norm_patchtokens"] if isinstance(o, dict) else o)))
    try:
        with _autocast(torch, dtype):
            out = model(x[None], **kw)
    finally:
        for h in hooks:
            h.remove()
    if feats == "agg":
        F = [_grid(cap["hidden"][:, -ph * pw:], N, ph, pw)]
    elif feats == "patch":
        F = [_grid(cap["enc"][:, -ph * pw:], N, ph, pw)]
    else:
        raise ValueError(feats)
    lp = out["local_points"][0].float()
    depth, conf = lp[..., 2], torch.sigmoid(out["conf"][0, ..., 0]).float()
    E = torch.linalg.inv(out["camera_poses"][0].float())
    lp_np = lp.detach().cpu().numpy()
    K = torch.from_numpy(np.stack([H._fit_intrinsics(lp_np[i], lp_np.shape[1:3]) for i in range(N)])).float().to(x.device)
    return depth, conf, F, K, E


FORWARD_FEATS = {"vggt": vggt_forward_feats, "pi3": pi3_forward_feats, "pi3x": pi3_forward_feats}


# ----------------------------------------------------------------------------- the head
def build_head(torch, in_dims, arch="conv", width=256):
    nn = torch.nn

    class ChannelLN(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.ln = nn.LayerNorm(c)

        def forward(self, x):
            return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    class SupportHead(nn.Module):
        """Feature maps (list of (N,Ci,ph,pw)) -> logit (N,ph,pw); positive = imagined (unsupported)."""

        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList([nn.Sequential(ChannelLN(c), nn.Conv2d(c, width, 1)) for c in in_dims])
            if arch == "linear":
                self.body = nn.Identity()
            elif arch == "conv":
                self.body = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(width, width, 3, padding=1), nn.GELU())
            else:
                raise ValueError(arch)
            self.out = nn.Conv2d(width, 1, 1)

        def forward(self, F):
            h = sum(p(f) for p, f in zip(self.proj, F))
            return self.out(self.body(h))[:, 0]

    return SupportHead()


def upsample_logit(torch, logit, hw):
    return torch.nn.functional.interpolate(logit[:, None], size=hw, mode="bilinear", align_corners=False)[:, 0]


def three_way(score, lo=0.3, hi=0.7):
    """score = 1 - P(imagined). Returns int map: 0 observed (score >= hi), 1 imagined (score <= lo), 2 abstain."""
    out = np.full(score.shape, 2, np.int8)
    out[score >= hi] = 0
    out[score <= lo] = 1
    return out


# ----------------------------------------------------------------------------- training samples
def _scale_K(K, sy, sx):
    K = np.array(K, np.float64, copy=True)
    K[:, 0] *= sx
    K[:, 1] *= sy
    return K


def model_res_data(data, model_hw, pred=None, conf_floor_q=0.2):
    """A hallu3d scene dict at model resolution (images) and half model resolution (depth labels), so that
    build_masks / apply_fill work unchanged. Depth and cameras for sphere placement come from the ground truth
    (pred=None, variant oiagt) or from the backbone's clean prediction pred=(depth, conf, K, E) at model res."""
    Hm, Wm = model_hw
    hm, wm = Hm // 2, Wm // 2
    imgs = np.stack([cv2.resize(im, (Wm, Hm), interpolation=cv2.INTER_AREA) for im in data["imgs"]]).astype(np.float32)
    if pred is None:
        h, w = data["depths"].shape[1:]
        depths = np.stack([cv2.resize(d, (wm, hm), interpolation=cv2.INTER_NEAREST) for d in data["depths"]])
        K, E = _scale_K(data["K_dep"], hm / h, wm / w), np.asarray(data["E"], np.float64)
    else:
        d, c, Km, E = pred
        depths = np.stack([cv2.resize(x, (wm, hm), interpolation=cv2.INTER_AREA) for x in d]).astype(np.float32)
        cs = np.stack([cv2.resize(x, (wm, hm), interpolation=cv2.INTER_AREA) for x in c])
        floor = np.quantile(cs, conf_floor_q)
        depths = np.where(cs >= floor, depths, 0.0).astype(np.float32)   # drop the least confident pixels (sky, black)
        K, E = _scale_K(Km, hm / Hm, wm / Wm), np.asarray(E, np.float64)
    out = dict(imgs=imgs, depths=depths.astype(np.float32), K_dep=K, E=E, scan=data["scan"], views=data.get("views"))
    if data.get("center_mask") is not None:
        out["center_mask"] = np.stack([cv2.resize(m.astype(np.uint8), (wm, hm), interpolation=cv2.INTER_NEAREST) > 0
                                       for m in data["center_mask"]])
    return out


def make_sample(mdata, seed, radius_frac, fill, cfg):
    """Masked images at model res, U and S labels at half model res. Raises RuntimeError when no sphere fits."""
    masks = H.build_masks(mdata, seed, radius_frac, cfg)
    imgs_in = H.apply_fill(mdata["imgs"], masks["hide"], fill, seed)
    return imgs_in, masks


def _dtu_lights(root, scan):
    d = os.path.join(root, "Rectified", f"scan{scan}_train")
    ls = sorted({int(f.split("_")[2]) for f in os.listdir(d) if f.startswith("rect_") and f.endswith(".png")})
    return ls or [3]


# ----------------------------------------------------------------------------- training
def train(base, dtu_root, variant="oia", out=None, steps=None, seed=0, device="cuda", scans=None, wrap=None,
          forward_feats=None, model_hw=None, **over):
    import torch
    feats, arch, labels = VARIANTS[variant]
    tc = {**TRAIN_DEFAULTS, **over}
    if steps:
        tc["steps"] = steps
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    wrap = wrap or H.MODEL_REGISTRY[base](device)
    ff = forward_feats or FORWARD_FEATS[base]
    model_hw = tuple(model_hw or H.DEFAULT_CFG["model_hw"])
    cfg = {**H.DEFAULT_CFG, "dilate_px": 3}          # dilation in pixels at model res (half the 640x512 value)
    scans = [s for s in (scans or DTU_TRAIN_SCANS)
             if os.path.isdir(os.path.join(dtu_root, "Rectified", f"scan{s}_train"))]
    if not scans:
        raise FileNotFoundError(f"no training scans under {dtu_root}")
    val_scans = [s for s in tc["val_scans"] if s in scans][:2]
    train_scans = [s for s in scans if s not in val_scans] or scans
    out = out or os.path.join("checkpoints", f"{base}_{variant}.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"[oia] base={base} variant={variant} feats={feats} arch={arch} labels={labels} "
          f"train scans={len(train_scans)} val scans={val_scans} steps={tc['steps']} -> {out}", flush=True)

    def forward(imgs_np):
        x = wrap._prep(imgs_np, model_hw)
        with torch.no_grad():
            d, c, F, K, E = ff(wrap, x, wrap.dtype, feats)
        return d, c, [f.detach() for f in F], K, E

    def scene(scan, views, light):
        data = H.load_scan(dtu_root, scan, views, light)
        pred = None
        if labels == "self":
            d, c, _, K, E = forward(data["imgs"])
            pred = (d.cpu().numpy(), c.cpu().numpy(), K.cpu().numpy(), E.cpu().numpy())
        return model_res_data(data, model_hw, pred, tc["conf_floor_q"])

    head, opt, sched = None, None, None
    hw_lab = (model_hw[0] // 2, model_hw[1] // 2)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    hist, best, t0 = [], -1.0, time.time()

    def step_loss(F, U, S):
        nonlocal head, opt, sched
        if head is None:
            head = build_head(torch, [f.shape[1] for f in F], arch, tc["width"]).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
            lam = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(s, tc["steps"]) / tc["steps"]))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
            print(f"[oia] head params: {sum(p.numel() for p in head.parameters()) / 1e6:.2f} M, feature dims {[f.shape[1] for f in F]}", flush=True)
        logit = upsample_logit(torch, head(F), hw_lab)
        U_t, S_t = torch.from_numpy(U).to(device), torch.from_numpy(S).to(device)
        tgt = U_t.float()
        w = torch.zeros_like(tgt)
        nU, nS = max(int(U_t.sum()), 1), max(int(S_t.sum()), 1)
        w[U_t] = 0.5 / nU
        w[S_t] = 0.5 / nS
        return (bce(logit, tgt) * w).sum(), logit

    def val_auroc():
        if not val_scans:
            return float("nan")
        head.eval()
        aucs = []
        with torch.no_grad():
            for s in val_scans:
                md = scene(s, H.select_views(tc["n_views"]), 3)
                for sd in tc["val_seeds"]:
                    try:
                        imgs_in, m = make_sample(md, 1000 + sd, 0.18, "blur", cfg)
                    except RuntimeError:
                        continue
                    _, _, F, _, _ = forward(imgs_in)
                    lg = upsample_logit(torch, head(F), hw_lab).cpu().numpy()
                    aucs.append(H.auroc(-lg[m["S"]], -lg[m["U"]]))
        head.train()
        return float(np.mean(aucs)) if aucs else float("nan")

    s = 0
    while s < tc["steps"]:
        scan = int(rng.choice(train_scans))
        views = sorted(rng.choice(49, tc["n_views"], replace=False).tolist())
        light = int(rng.choice(_dtu_lights(dtu_root, scan))) if tc["random_light"] else 3
        try:
            md = scene(scan, views, light)
        except Exception as e:            # unreadable scan: skip it
            print(f"[oia] skip scan {scan}: {e}", flush=True)
            train_scans = [t for t in train_scans if t != scan] or train_scans
            continue
        if rng.random() < tc["p_clean"]:
            imgs_in = md["imgs"]
            U = np.zeros(md["depths"].shape, bool)
            S = md["depths"] > 0
        else:
            try:
                imgs_in, m = make_sample(md, s, float(rng.uniform(*tc["radius_range"])), str(rng.choice(tc["fills"])), cfg)
            except RuntimeError as e:
                print(f"[oia] skip sample: {e}", flush=True)
                continue
            U, S = m["U"], m["S"]
        _, _, F, _, _ = forward(imgs_in)
        loss, logit = step_loss(F, U, S)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        s += 1
        if s % tc["log_every"] == 0 or s == 1:
            lg = logit.detach().cpu().numpy()
            a = H.auroc(-lg[S], -lg[U]) if U.any() else float("nan")
            print(f"[oia] step {s:5d} loss {loss.item():.4f} train-auroc {a:.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            hist.append(dict(step=s, loss=float(loss.item()), auroc=a))
        if s % tc["val_every"] == 0 or s == tc["steps"]:
            va = val_auroc()
            print(f"[oia] step {s:5d} val-auroc {va:.3f}", flush=True)
            hist.append(dict(step=s, val_auroc=va))
            ck = dict(state=head.state_dict(), in_dims=[f.shape[1] for f in F], arch=arch, feats=feats, labels=labels,
                      width=tc["width"], base=base, variant=variant, model_hw=list(model_hw), step=s, val_auroc=va,
                      seed=seed, train_cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in tc.items()},
                      train_scans=train_scans, val_scans=val_scans, time=time.strftime("%Y-%m-%d %H:%M:%S"))
            torch.save(ck, out)
            if va >= best:
                best = va
                torch.save(ck, out.replace(".pt", "_best.pt"))
    json.dump(hist, open(out.replace(".pt", "_hist.json"), "w"), indent=1)
    print(f"[oia] done: {out} (last), best val-auroc {best:.3f}, {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------- evaluation wrapper
def resolve_ckpt(name):
    env = os.environ.get("OIA_CKPT_" + re.sub(r"\W", "_", name).upper())
    if env:
        return env
    base, var = name.rsplit("_", 1)
    if var == "oiafuse":
        return resolve_ckpt(f"{base}_oia")
    return os.path.join(os.environ.get("OIA_CKPT_DIR", "checkpoints"), f"{name}.pt")


class OIAWrap:
    """hallu3d model: depth from the base model, confidence = 1 - P(imagined) from the support head."""

    def __init__(self, base, variant="oia", ckpt=None, forward_feats=None):
        torch = base.torch
        self.base, self.variant = base, variant
        self.name = f"{base.name}_{variant}"
        self.fuse = variant == "oiafuse"
        path = ckpt or resolve_ckpt(self.name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{self.name}: checkpoint {path} not found (train with scripts/oia_head.py train)")
        ck = torch.load(path, map_location=base.device, weights_only=False)
        self.feats, self.ckpt = ck["feats"], path
        self.head = build_head(torch, ck["in_dims"], ck["arch"], ck["width"]).to(base.device).eval()
        self.head.load_state_dict(ck["state"])
        self.ff = forward_feats or FORWARD_FEATS[base.name]
        print(f"[{self.name}] head from {path} (step {ck.get('step')}, val-auroc {ck.get('val_auroc', float('nan')):.3f})")

    def predict(self, imgs, model_hw, out_hw):
        torch = self.base.torch
        x = self.base._prep(imgs, model_hw)
        with torch.no_grad():
            d, c, F, _, _ = self.ff(self.base, x, self.base.dtype, self.feats)
            if not torch.isfinite(d).all():
                d, c, F, _, _ = self.ff(self.base, x, None, self.feats)
            logit = upsample_logit(torch, self.head([f.float() for f in F]), out_hw)
            score = 1.0 - torch.sigmoid(logit.float())
            d = H._resize_t(d[:, None].float(), out_hw, "area")[:, 0]
            if self.fuse:
                cn = H._resize_t(c[:, None].float(), out_hw, "area")[:, 0]
                cn = (cn - cn.min()) / (cn.max() - cn.min() + 1e-9)
                score = torch.sqrt(score.clamp_min(1e-6) * cn.clamp_min(1e-6))
        return d.cpu().numpy().astype(np.float32), score.cpu().numpy().astype(np.float32)

    def free(self):
        self.base.free()


def make_oia(name, registry=None):
    """'vggt_oia' -> OIAWrap(VGGTWrap(), 'oia'); used by hallu3d.make_model."""
    m = re.fullmatch(r"(.+?)_(" + "|".join(VARIANTS) + r")", name)
    if not m:
        return None
    registry = registry or H.MODEL_REGISTRY
    return OIAWrap(registry[m.group(1)](), m.group(2))


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", default="vggt", choices=sorted(FORWARD_FEATS))
    t.add_argument("--variant", default="oia", choices=[v for v in VARIANTS if v != "oiafuse"])
    t.add_argument("--dtu", required=True, help="DTU root (Cameras/train, Rectified, Depths) holding the training scans")
    t.add_argument("--out", default=None)
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--scans", default=None, help="comma-separated subset of training scans (smoke tests)")
    lm = sub.add_parser("list-modules", help="print module names of a backbone (to adapt the feature hooks)")
    lm.add_argument("--base", default="pi3x")
    a = ap.parse_args()
    if a.cmd == "list-modules":
        w = H.MODEL_REGISTRY[a.base]()
        for n, m in w.model.named_modules():
            if n.count(".") <= 1:
                print(n, type(m).__name__)
        return
    over = {k: v for k, v in dict(lr=a.lr).items() if v is not None}
    root = H.find_dtu_root((a.dtu,)) if os.path.isdir(a.dtu) else a.dtu
    scans = [int(s) for s in a.scans.split(",")] if a.scans else None
    train(a.base, root, a.variant, a.out, a.steps, a.seed, scans=scans, **over)


if __name__ == "__main__":
    main()
