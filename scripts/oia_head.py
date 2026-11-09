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
  <base>_oiaattn    two cross-view transformer blocks over the tokens of all views, same features       (ablation)
  <base>_oiavis     conv head trained on TRUE visibility labels (ground-truth multi-view support, no patch)  (P4-vis)
  <base>_oiavisself the same head trained on the backbone's OWN predicted support: no ground truth anywhere  (P5)
  <base>_oiavislin<L> linear probe on one aggregator layer, true-visibility labels (where visibility lives)
  <base>_oiaprov    provenance head: per-view evidence attention, says WHICH input views corroborate each pixel;
                    trained on ground-truth per-view visibility (hallu3d.support_matrix)                      (P10-prov)
  <base>_oiaprovself the same head trained on the per-view support of the backbone's own prediction, label-free (P10-prov)
Several heads with the same label source train jointly on the same backbone forwards (train_multi, --variant a,b,c).
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
    "oiaattn": ("agg",   "attn",   "self"),    # two cross-view transformer blocks over the tokens of all views
    # true-visibility labels (P4-vis): clean images, U = pixels no other input view sees (ground-truth support count
    # 0, hallu3d.vis_labels), S = pixels seen by at least two other views. No patch, no fill: the head has to learn
    # which pixels have parallax in the input. The same per-image control and linear probe are trained alongside.
    "oiavis":      ("agg",   "conv",   "vis"),
    "oiavispatch": ("patch", "conv",   "vis"),
    "oiavislin":   ("agg",   "linear", "vis"),
    # label-free visibility (P5-visself): the same clean-image protocol, but the labels come from the backbone's OWN
    # predicted depth and cameras (hallu3d.support_count on the prediction, the quantity reproj_support_score tests
    # at evaluation time). No ground truth at any point: the head distils the cross-view reprojection test into one
    # forward on the frozen features. Its per-image control and linear probe train alongside.
    "oiavisself":      ("agg",   "conv",   "visself"),
    "oiavisselfpatch": ("patch", "conv",   "visself"),
    "oiavisselflin":   ("agg",   "linear", "visself"),
    # capacity (P15-capacity, Gate 13): the same label-free labels and features, two cross-view transformer blocks
    # over the tokens of all views instead of the conv head (the larger end of linear probe < conv head < transformer)
    "oiavisselfattn":  ("agg",   "attn",   "visself"),
    # where visibility lives (mechanism): a linear probe on ONE aggregator layer each (VGGT layers 4 / 11 / 17 / 23,
    # LAYER_PICK indexes the hook's layer list) and on the per-image encoder tokens, true-visibility labels.
    "oiavislin4":     ("agg",   "linear", "vis"),
    "oiavislin11":    ("agg",   "linear", "vis"),
    "oiavislin17":    ("agg",   "linear", "vis"),
    "oiavislin23":    ("agg",   "linear", "vis"),
    "oiavispatchlin": ("patch", "linear", "vis"),
    # evaluation-time fusion with native confidence (geometric mean of the two scores); share the head's checkpoint
    "oiavisfuse":     ("agg",   "conv",   "vis"),
    "oiavisselffuse": ("agg",   "conv",   "visself"),
    # provenance (P10-prov): one logit per (target view, source view, patch) from cross-view evidence attention over
    # the same features; the support score is the log-odds that no source view corroborates the patch. Trained on
    # the per-view visibility bits of the ground truth (oiaprov) or of the backbone's own prediction (oiaprovself).
    "oiaprov":     ("agg",   "prov",   "vis"),
    "oiaprovself": ("agg",   "prov",   "visself"),
    # matched-patch labels (P11-matched, Gate 7): every training scene carries three identical patches (same fill,
    # size and depth), one hidden in every view (imagined) and two left visible in some views (observed, filled):
    # from one image the patches cannot be told apart, so the head must read cross-view evidence. Negatives are the
    # filled-but-observed pixels together with the clean supported pixels (equal weight, so a patch cue is
    # penalised). `match`: placement and visibility from the ground truth; `matchself`: from the backbone's own
    # prediction, no ground truth. The per-image control is the pipeline check (it should sit near chance).
    "oiamatch":          ("agg",   "conv",   "match"),
    "oiamatchpatch":     ("patch", "conv",   "match"),
    "oiamatchlin":       ("agg",   "linear", "match"),
    "oiamatchself":      ("agg",   "conv",   "matchself"),
    "oiamatchselfpatch": ("patch", "conv",   "matchself"),
    "oiamatchselflin":   ("agg",   "linear", "matchself"),
    # P13 (Gate 9): the same triples fine-tuned from the 1500-step matched checkpoints with single-view evidence
    # oversampled (evidence "1" with probability 0.7, "2" and "half" 0.15 each), aimed at the one-view bucket
    "oiamatchev":          ("agg",   "conv",   "match"),
    "oiamatchevpatch":     ("patch", "conv",   "match"),
    "oiamatchevlin":       ("agg",   "linear", "match"),
    "oiamatchselfev":      ("agg",   "conv",   "matchself"),
    "oiamatchselfevpatch": ("patch", "conv",   "matchself"),
    "oiamatchselfevlin":   ("agg",   "linear", "matchself"),
}
EV_FOCUS = {"oiamatchev", "oiamatchevpatch", "oiamatchevlin", "oiamatchselfev", "oiamatchselfevpatch", "oiamatchselfevlin"}
LAYER_PICK = {"oiavislin4": 0, "oiavislin11": 1, "oiavislin17": 2, "oiavislin23": 3}


def _pick(variant, F):
    """The feature maps a variant reads: all hooked layers, or one of them for the per-layer probes."""
    i = LAYER_PICK.get(variant)
    if i is None:
        return F
    if i >= len(F):
        raise ValueError(f"{variant} needs hooked layer index {i}, the backbone exposes {len(F)} feature map(s)")
    return [F[i]]
SELF_VARIANTS = [v for v, (_, _, lab) in VARIANTS.items() if lab == "self" and v != "oiafuse"]
VIS_VARIANTS = [v for v, (_, _, lab) in VARIANTS.items() if lab == "vis"]
TRAIN_DEFAULTS = dict(steps=3000, lr=1e-3, wd=1e-2, warmup=100, n_views=10, radius_range=(0.08, 0.24),
                      fills=["gray", "blur", "noise"], p_clean=0.15, width=256, conf_floor_q=0.2,
                      val_every=250, log_every=25, val_scans=[2, 6], val_seeds=[0, 1], random_light=True,
                      vis_views=(3, 5, 10), vis_tol=0.03, vis_min_seen=2, vis_min_pixels=50)


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
    F = {}
    if feats in ("agg", "both"):
        F["agg"] = [_grid(tokens[l][0][:, ps:], N, ph, pw) for l in layers]
    if feats in ("patch", "both"):   # per-image DINOv2 tokens before any cross-view attention (the frame-only control)
        agg = model.aggregator
        xn = (x - agg._resnet_mean.reshape(1, 3, 1, 1)) / agg._resnet_std.reshape(1, 3, 1, 1)
        with _autocast(torch, dtype):
            pt = agg.patch_embed(xn)
        if isinstance(pt, dict):
            pt = pt["x_norm_patchtokens"]
        F["patch"] = [_grid(pt[:, -ph * pw:], N, ph, pw)]
    if not F:
        raise ValueError(feats)
    return depth[0, ..., 0].float(), conf[0].float(), (F if feats == "both" else F[feats]), intr[0].float(), E


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
    if feats in ("patch", "both"):
        hooks.append(_find_module(model, "encoder").register_forward_hook(
            lambda m, a, o: cap.__setitem__("enc", o["x_norm_patchtokens"] if isinstance(o, dict) else o)))
    try:
        with _autocast(torch, dtype):
            out = model(x[None], **kw)
    finally:
        for h in hooks:
            h.remove()
    F = {}
    if feats in ("agg", "both"):
        F["agg"] = [_grid(cap["hidden"][:, -ph * pw:], N, ph, pw)]
    if feats in ("patch", "both"):
        F["patch"] = [_grid(cap["enc"][:, -ph * pw:], N, ph, pw)]
    if not F:
        raise ValueError(feats)
    F = F if feats == "both" else F[feats]
    lp = out["local_points"][0].float()
    depth, conf = lp[..., 2], torch.sigmoid(out["conf"][0, ..., 0]).float()
    E = torch.linalg.inv(out["camera_poses"][0].float())
    lp_np = lp.detach().cpu().numpy()
    K = torch.from_numpy(np.stack([H._fit_intrinsics(lp_np[i], lp_np.shape[1:3]) for i in range(N)])).float().to(x.device)
    return depth, conf, F, K, E


def da3_forward_feats(wrap, x, dtype, feats="agg"):
    """Depth Anything 3 (Large 1.1). Cross-view attention lives inside its DINOv2: blocks 0 to 7 attend within a
    frame, from block 8 on odd blocks attend across all views; the depth head reads blocks 11, 15, 19, 23, each as
    the frame-wise stream concatenated with the global stream (2048-d). "agg": those four maps, captured on the
    backbone's output (feats = four (patch tokens (B,S,Np,2048), camera token) pairs); "patch": the output of
    block 7, the last block before any cross-view attention (1024-d), captured with a hook on that block. Depth,
    confidence and cameras come from the same forward (hallu3d.DA3Wrap._forward_raw)."""
    torch, model = wrap.torch, wrap.model
    N, ph, pw = x.shape[0], x.shape[-2] // PATCH, x.shape[-1] // PATCH
    cap, hooks = {}, []
    net = model.model
    hooks.append(net.backbone.register_forward_hook(lambda m, a, o: cap.__setitem__("feats", o[0])))
    if feats in ("patch", "both"):
        hooks.append(net.backbone.pretrained.blocks[7].register_forward_hook(lambda m, a, o: cap.__setitem__("local", o)))
    try:
        out = wrap._forward_raw(x)
    finally:
        for h in hooks:
            h.remove()
    F = {}
    if feats in ("agg", "both"):
        F["agg"] = [_grid(f[0][0][:, -ph * pw:], N, ph, pw) for f in cap["feats"]]
    if feats in ("patch", "both"):
        loc = cap["local"]
        loc = loc.reshape(N, -1, loc.shape[-1]) if loc.dim() == 3 else loc[0]
        F["patch"] = [_grid(loc[:, -ph * pw:], N, ph, pw)]
    if not F:
        raise ValueError(feats)
    F = F if feats == "both" else F[feats]
    depth = out["depth"][0].float()
    conf = out["depth_conf"][0].float() if "depth_conf" in out else torch.ones_like(depth)
    E = out["extrinsics"][0].float()
    if E.shape[-2] == 3:
        E4 = torch.eye(4, device=E.device).repeat(N, 1, 1); E4[:, :3, :4] = E; E = E4
    return depth, conf, F, out["intrinsics"][0].float(), E


FORWARD_FEATS = {"vggt": vggt_forward_feats, "pi3": pi3_forward_feats, "pi3x": pi3_forward_feats, "da3": da3_forward_feats}


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
            elif arch == "attn":
                self.body = CrossViewBlocks(width, depth=2, heads=4)
            else:
                raise ValueError(arch)
            self.out = nn.Conv2d(width, 1, 1)

        def forward(self, F):
            h = sum(p(f) for p, f in zip(self.proj, F))
            return self.out(self.body(h))[:, 0]

    class ProvenanceHead(nn.Module):
        """Feature maps (list of (N,Ci,ph,pw)) -> per-view evidence logits (N,N,ph,pw): logit[i, j, p] says whether
        source view j corroborates the surface at patch p of target view i (the diagonal is unused). Every target
        token queries the tokens of each other view with multi-head attention; the attended evidence, the query,
        their product and the attention's peak weight per head feed a shared MLP. No cameras and no positions are
        given: the backbone's cross-view tokens already carry the geometry, and the head only has to read where a
        patch's evidence sits in another view. forward() returns the support logit of the other heads (positive =
        imagined): the log-odds that no source view corroborates the patch, sum_j log(1 - sigmoid(logit_ij))."""

        def __init__(self, heads=4):
            super().__init__()
            self.proj = nn.ModuleList([nn.Sequential(ChannelLN(c), nn.Conv2d(c, width, 1)) for c in in_dims])
            self.local = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, padding=1), nn.GELU())
            self.ln = nn.LayerNorm(width)
            self.q, self.k, self.v = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
            self.heads = heads
            self.mlp = nn.Sequential(nn.Linear(3 * width + heads, width), nn.GELU(), nn.Linear(width, width // 2),
                                     nn.GELU(), nn.Linear(width // 2, 1))

        def tokens(self, F):
            h = sum(p(f) for p, f in zip(self.proj, F))
            h = h + self.local(h)
            N, C, ph, pw = h.shape
            return h.flatten(2).transpose(1, 2), (ph, pw)                    # (N, L, C)

        def pairs(self, F):
            x, (ph, pw) = self.tokens(F)
            N, L, C = x.shape
            Hh, dh = self.heads, C // self.heads
            xn = self.ln(x)
            q = self.q(xn).reshape(N, L, Hh, dh)
            k = self.k(xn).reshape(N, L, Hh, dh)
            v = self.v(xn).reshape(N, L, Hh, dh)
            att = torch.einsum("ilhd,jmhd->ijhlm", q, k) / dh ** 0.5         # target (i, l) over source (j, m)
            a = att.softmax(-1)
            e = torch.einsum("ijhlm,jmhd->ijlhd", a, v).reshape(N, N, L, C)  # evidence read from view j
            peak = a.amax(-1).permute(0, 1, 3, 2)                             # (i, j, l, Hh)
            xi = x[:, None].expand(N, N, L, C)
            logit = self.mlp(torch.cat([xi, e, xi * e, peak], -1))[..., 0]
            return logit.reshape(N, N, ph, pw)

        @staticmethod
        def aggregate(pair_logit):
            N = pair_logit.shape[0]
            off = ~torch.eye(N, dtype=torch.bool, device=pair_logit.device)
            lognot = torch.nn.functional.logsigmoid(-pair_logit) * off[:, :, None, None]
            log_p = lognot.sum(1).clamp(max=-1e-6)                            # log P(no view corroborates)
            return log_p - torch.log(-torch.expm1(log_p))                     # logit, positive = imagined

        def forward(self, F):
            return self.aggregate(self.pairs(F))

    class CrossViewBlocks(nn.Module):
        """Transformer blocks with global attention over the tokens of every view (pre-LN, SDPA). Each token
        carries a 2D sinusoidal position and a learned view embedding, so the head can relate a patch to the
        same surface seen from the other cameras. Input / output (N, C, ph, pw)."""

        def __init__(self, c, depth=2, heads=4, max_views=64):
            super().__init__()
            self.view_emb = nn.Embedding(max_views, c)
            self.blocks = nn.ModuleList([nn.ModuleDict(dict(
                ln1=nn.LayerNorm(c), qkv=nn.Linear(c, 3 * c), proj=nn.Linear(c, c),
                ln2=nn.LayerNorm(c), mlp=nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))))
                for _ in range(depth)])
            self.heads, self.c = heads, c

        @staticmethod
        def _pos2d(c, ph, pw, device):
            import math
            q = c // 4
            om = torch.exp(-math.log(1e4) * torch.arange(q, device=device) / q)
            ys = torch.arange(ph, device=device)[:, None] * om[None]
            xs = torch.arange(pw, device=device)[:, None] * om[None]
            py = torch.cat([ys.sin(), ys.cos()], 1)            # ph, c/2
            px = torch.cat([xs.sin(), xs.cos()], 1)            # pw, c/2
            pos = torch.cat([py[:, None].expand(ph, pw, c // 2), px[None].expand(ph, pw, c // 2)], -1)
            return pos.reshape(ph * pw, c)

        def forward(self, h):
            N, C, ph, pw = h.shape
            x = h.flatten(2).transpose(1, 2)                                       # N, L, C
            x = x + self._pos2d(C, ph, pw, h.device)[None] + self.view_emb(torch.arange(N, device=h.device))[:, None]
            x = x.reshape(1, N * ph * pw, C)
            for b in self.blocks:
                y = b["ln1"](x)
                q, k, v = b["qkv"](y).reshape(1, -1, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                x = x + b["proj"](y.transpose(1, 2).reshape(1, -1, C))
                x = x + b["mlp"](b["ln2"](x))
            return x.reshape(N, ph * pw, C).transpose(1, 2).reshape(N, C, ph, pw)

    return ProvenanceHead() if arch == "prov" else SupportHead()


def upsample_logit(torch, logit, hw):
    return torch.nn.functional.interpolate(logit[:, None], size=hw, mode="bilinear", align_corners=False)[:, 0]


def upsample_pairs(torch, pair_logit, hw):
    """(N,N,ph,pw) -> (N,N,h,w), bilinear."""
    N = pair_logit.shape[0]
    up = torch.nn.functional.interpolate(pair_logit.reshape(N * N, 1, *pair_logit.shape[2:]), size=hw, mode="bilinear",
                                         align_corners=False)
    return up.reshape(N, N, *hw)


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
    """One head. See train_multi for several heads on the same backbone forwards."""
    out = out or os.path.join("checkpoints", f"{base}_{variant}.pt")
    return train_multi(base, dtu_root, [(variant, seed, out)], steps, device, scans, wrap, forward_feats, model_hw,
                       **over)[0]


def train_multi(base, dtu_root, specs, steps=None, device="cuda", scans=None, wrap=None, forward_feats=None,
                model_hw=None, resume=False, **over):
    """Train several support heads at once on the same stream of backbone forwards. specs: list of
    (variant, init_seed, out_path). All heads see the same samples (scenes, views, lights, masks, fills, drawn
    with the seed of the first spec); they differ in architecture, in the feature group they read, and in the
    initialisation seed. The backbone forward is the whole cost of a step, so this trains the main head, the
    linear probe, the per-image control and the transformer head for the price of one. Heads with labels "gt"
    (oiagt) need a different masked input and must be trained in a run of their own.
    resume=True continues from the checkpoints at the out paths (head weights and step; the optimiser state is not
    stored, so AdamW restarts at the schedule's learning rate for that step, and the data stream is re-seeded with
    data_seed + step so the samples after the resume differ from the ones before it)."""
    import torch
    tc = {**TRAIN_DEFAULTS, **over}
    if steps:
        tc["steps"] = steps
    specs = [(v, int(sd), o) for v, sd, o in specs]
    labels = {VARIANTS[v][2] for v, _, _ in specs}
    assert len(labels) == 1, f"heads with different label sources cannot share a run: {labels}"
    labels = labels.pop()
    data_seed = specs[0][1]
    start = 0
    if resume:
        saved = [torch.load(o, map_location="cpu", weights_only=False) for _, _, o in specs if os.path.exists(o)]
        if len(saved) == len(specs):
            start = min(int(ck.get("step", 0)) for ck in saved)
            if start >= (steps or TRAIN_DEFAULTS["steps"]):
                print(f"[oia] resume: every checkpoint is at step {start} >= {steps}; nothing to do", flush=True)
                return [o for _, _, o in specs]
            print(f"[oia] resume from step {start} ({len(saved)} checkpoints)", flush=True)
        else:
            print(f"[oia] resume requested but only {len(saved)} of {len(specs)} checkpoints exist; training from scratch", flush=True)
            start = 0
    torch.manual_seed(data_seed + start)
    rng = np.random.default_rng(data_seed + start)
    wrap = wrap or H.MODEL_REGISTRY[base](device)
    ff = forward_feats or FORWARD_FEATS[base]
    model_hw = tuple(model_hw or H.DEFAULT_CFG["model_hw"])
    cfg = {**H.DEFAULT_CFG, "dilate_px": 3}          # dilation in pixels at model res (half the 640x512 value)
    # training root: a DTU root (Rectified/, 49 views per scan, lights) or a preprocessed CO3D-format root
    # (<scene>/meta.npz, scripts/co3d.py and scripts/eth3d.py): label-free adaptation to a new capture domain
    # (P6-adapt) trains "visself" heads on scenes of that domain without any ground truth
    is_co3d = not os.path.isdir(os.path.join(dtu_root, "Rectified"))
    if is_co3d:
        import co3d as C
        all_scenes = C.list_co3d_scenes(dtu_root)
        scans = [s for s in (scans or all_scenes) if s in all_scenes]
        assert labels == "visself" or labels == "vis", f"only visibility labels can be trained on a CO3D-format root ({labels})"
    else:
        scans = [s for s in (scans or DTU_TRAIN_SCANS)
                 if os.path.isdir(os.path.join(dtu_root, "Rectified", f"scan{s}_train"))]
    if not scans:
        raise FileNotFoundError(f"no training scans under {dtu_root}")
    val_scans = [s for s in tc["val_scans"] if s in scans][:2]
    train_scans = [s for s in scans if s not in val_scans] or scans
    need = sorted({VARIANTS[v][0] for v, _, _ in specs})
    feats_mode = need[0] if len(need) == 1 else "both"
    for v, sd, o in specs:
        os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
        print(f"[oia] base={base} variant={v} feats={VARIANTS[v][0]} arch={VARIANTS[v][1]} labels={labels} seed={sd} "
              f"train scans={len(train_scans)} val scans={val_scans} steps={tc['steps']} -> {o}", flush=True)

    def forward(imgs_np, mode, hw=None):
        x = wrap._prep(imgs_np, tuple(hw or model_hw))
        with torch.no_grad():
            d, c, F, K, E = ff(wrap, x, wrap.dtype, mode)
            if mode == "both" and not isinstance(F, dict):      # hook without a joint mode: one call per group
                F = {m: ff(wrap, x, wrap.dtype, m)[2] for m in need}
        if isinstance(F, dict):
            F = {k: [f.detach() for f in v] for k, v in F.items()}
        else:
            F = {mode: [f.detach() for f in F]}
        return d, c, F, K, E

    def scene(scan, views, light, gt=False):
        """Scene dict at model resolution. Labels "gt" and "vis" place / label from the ground truth; "self" and
        "visself" from the backbone's clean prediction (gt=True forces the ground truth, for validation). For
        "visself" the clean forward is also the training forward, so its features ride along under "_F"."""
        data = C.load_co3d_scene(dtu_root, scan, views) if is_co3d else H.load_scan(dtu_root, scan, views, light)
        hw = tuple(data.get("model_hw", model_hw))     # CO3D-format scenes carry their own (portrait / landscape)
        pred, F = None, None
        if labels in ("self", "visself", "matchself") and not gt:
            d, c, F, K, E = forward(data["imgs"], feats_mode if labels == "visself" else "agg", hw)
            pred = (d.cpu().numpy(), c.cpu().numpy(), K.cpu().numpy(), E.cpu().numpy())
        md = model_res_data(data, hw, pred, tc["conf_floor_q"])
        md["model_hw"] = hw
        if labels == "visself" and F is not None:
            md["_F"] = F
        return md

    hw_lab = (model_hw[0] // 2, model_hw[1] // 2)
    any_prov = any(VARIANTS[v][1] == "prov" for v, _, _ in specs)
    vis_cfg = {"vis_tol": tc["vis_tol"], "vis_min_seen": tc["vis_min_seen"], "vis_dilate_px": 3, "vis_pairs": any_prov}
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    heads = [dict(variant=v, seed=sd, out=o, feats=VARIANTS[v][0], arch=VARIANTS[v][1], head=None, opt=None,
                  sched=None, hist=[], best=-1.0) for v, sd, o in specs]
    t0 = time.time()

    def build(hd, F):
        F = _pick(hd["variant"], F)
        torch.manual_seed(hd["seed"])
        hd["head"] = build_head(torch, [f.shape[1] for f in F], hd["arch"], tc["width"]).to(device)
        hd["opt"] = torch.optim.AdamW(hd["head"].parameters(), lr=tc["lr"], weight_decay=tc["wd"])
        lam = lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(s, tc["steps"]) / tc["steps"]))
        hd["sched"] = torch.optim.lr_scheduler.LambdaLR(hd["opt"], lam)
        if start > 0:
            ck = torch.load(hd["out"], map_location=device, weights_only=False)
            hd["head"].load_state_dict(ck["state"])
            hd["best"] = float(ck.get("val_auroc", -1.0)) if np.isfinite(ck.get("val_auroc", float("nan"))) else -1.0
            for _ in range(start):
                hd["sched"].step()
        print(f"[oia] {hd['variant']} head params: {sum(p.numel() for p in hd['head'].parameters()) / 1e6:.2f} M, "
              f"feature dims {[f.shape[1] for f in F]}", flush=True)

    def step_loss(hd, F, U, S, pairs=None, S2=None):
        """S2: a second negative group (the filled-but-observed pixels of the matched protocol); it shares the
        negative weight equally with S, so the head is penalised for calling a patch imagined on appearance."""
        if hd["head"] is None:
            build(hd, F)
        hw = tuple(U.shape[1:])
        pair_logit = None
        if hd["arch"] == "prov":
            pair_logit = hd["head"].pairs(_pick(hd["variant"], F))
            logit = upsample_logit(torch, hd["head"].aggregate(pair_logit), hw)
        else:
            logit = upsample_logit(torch, hd["head"](_pick(hd["variant"], F)), hw)
        U_t, S_t = torch.from_numpy(U).to(device), torch.from_numpy(S).to(device)
        tgt = U_t.float()
        w = torch.zeros_like(tgt)
        nU, nS = max(int(U_t.sum()), 1), max(int(S_t.sum()), 1)
        w[U_t] = 0.5 / nU
        if S2 is not None and S2.any():
            S2_t = torch.from_numpy(S2).to(device)
            w[S_t] = 0.25 / nS
            w[S2_t] = 0.25 / max(int(S2_t.sum()), 1)
        else:
            w[S_t] = 0.5 / nS
        loss = (bce(logit, tgt) * w).sum()
        if pair_logit is not None and pairs is not None:
            # per-view term: does view j corroborate pixel p of view i? Decided pixels only (known depth in view i,
            # no hole in view j), classes balanced, the diagonal excluded. Same weight as the support term.
            seen_m, unk_m, valid = pairs
            N = pair_logit.shape[0]
            pl = upsample_pairs(torch, pair_logit, hw)
            off = ~torch.eye(N, dtype=torch.bool, device=device)
            mask = off[:, :, None, None] & torch.from_numpy(valid).to(device)[:, None] & ~torch.from_numpy(unk_m).to(device)
            pos = mask & torch.from_numpy(seen_m).to(device)
            neg = mask & ~pos
            wp = torch.zeros_like(pl)
            wp[pos] = 0.5 / max(int(pos.sum()), 1)
            wp[neg] = 0.5 / max(int(neg.sum()), 1)
            loss = loss + (bce(pl, pos.float()) * wp).sum()
        return loss, logit

    def val_auroc():
        if not val_scans:
            return {hd["out"]: float("nan") for hd in heads}
        for hd in heads:
            hd["head"].eval()
        aucs = {hd["out"]: [] for hd in heads}
        with torch.no_grad():
            for s in val_scans:
                if labels in ("vis", "visself"):     # five views, view set rotated by the seed, clean images;
                    for sd in tc["val_seeds"]:         # validated on ground-truth labels for both label sources
                        total = C.n_frames(dtu_root, s) if is_co3d else 49
                        md = scene(s, H.vis_views(5, sd, total=total), 3, gt=True)
                        m = H.vis_labels(md, vis_cfg)
                        if m["U"].sum() < tc["vis_min_pixels"] or m["S"].sum() < tc["vis_min_pixels"]:
                            continue
                        _, _, F, _, _ = forward(md["imgs"], feats_mode, md["model_hw"])
                        for hd in heads:
                            lg = upsample_logit(torch, hd["head"](_pick(hd["variant"], F[hd["feats"]])), tuple(m["U"].shape[1:])).cpu().numpy()
                            aucs[hd["out"]].append(H.auroc(-lg[m["S"]], -lg[m["U"]]))
                    continue
                if labels in ("match", "matchself"):   # validated on ground-truth placement: AUROC of U against the
                    md = scene(s, H.select_views(tc["n_views"]), 3, gt=True)   # filled-but-observed pixels (auroc_m)
                    for sd in tc["val_seeds"]:
                        try:
                            m = H.build_masks_matched(md, 1000 + sd, 0.18, cfg, evidence="1")
                        except (RuntimeError, ValueError):
                            continue
                        imgs_in = H.apply_fill(md["imgs"], m["hide"], "blur", 1000 + sd)
                        _, _, F, _, _ = forward(imgs_in, feats_mode)
                        for hd in heads:
                            lg = upsample_logit(torch, hd["head"](_pick(hd["variant"], F[hd["feats"]])), tuple(m["U"].shape[1:])).cpu().numpy()
                            aucs[hd["out"]].append(H.auroc(-lg[m["Sm"]], -lg[m["U"]]))
                    continue
                md = scene(s, H.select_views(tc["n_views"]), 3)
                for sd in tc["val_seeds"]:
                    try:
                        imgs_in, m = make_sample(md, 1000 + sd, 0.18, "blur", cfg)
                    except RuntimeError:
                        continue
                    _, _, F, _, _ = forward(imgs_in, feats_mode)
                    for hd in heads:
                        lg = upsample_logit(torch, hd["head"](_pick(hd["variant"], F[hd["feats"]])), tuple(m["U"].shape[1:])).cpu().numpy()
                        aucs[hd["out"]].append(H.auroc(-lg[m["S"]], -lg[m["U"]]))
        for hd in heads:
            hd["head"].train()
        return {hd["out"]: (float(np.mean(aucs[hd["out"]])) if aucs[hd["out"]] else float("nan")) for hd in heads}

    def save(hd, s, va, F):
        ck = dict(state=hd["head"].state_dict(), in_dims=[f.shape[1] for f in _pick(hd["variant"], F)], arch=hd["arch"],
                  feats=hd["feats"], layer_pick=LAYER_PICK.get(hd["variant"]),
                  labels=labels, width=tc["width"], base=base, variant=hd["variant"], model_hw=list(model_hw), step=s,
                  val_auroc=va, seed=hd["seed"], data_seed=data_seed,
                  train_cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in tc.items()},
                  train_scans=train_scans, val_scans=val_scans, joint_with=[h["variant"] for h in heads],
                  time=time.strftime("%Y-%m-%d %H:%M:%S"))
        torch.save(ck, hd["out"])
        if va >= hd["best"]:
            hd["best"] = va
            torch.save(ck, hd["out"].replace(".pt", "_best.pt"))

    s, fails = start, 0
    while s < tc["steps"]:
        scan = str(rng.choice(train_scans)) if is_co3d else int(rng.choice(train_scans))
        n_views = int(rng.choice(tc["vis_views"])) if labels in ("vis", "visself") else tc["n_views"]
        total = C.n_frames(dtu_root, scan) if is_co3d else 49
        views = sorted(rng.choice(total, min(n_views, total), replace=False).tolist())
        light = int(rng.choice(_dtu_lights(dtu_root, scan))) if tc["random_light"] and not is_co3d else 3
        try:
            md = scene(scan, views, light)
            fails = 0
        except Exception as e:            # unreadable scan: skip it (and stop when nothing can be read at all)
            print(f"[oia] skip scan {scan}: {e}", flush=True)
            train_scans = [t for t in train_scans if t != scan] or train_scans
            fails += 1
            if fails >= 20:
                raise RuntimeError(f"20 consecutive scene failures, last: {e}")
            continue
        pairs, S2 = None, None
        if labels in ("vis", "visself"):  # clean images; labels from the ground-truth support count (vis) or from
            imgs_in = md["imgs"]          # the support count of the backbone's own prediction (visself, no GT)
            m = H.vis_labels(md, vis_cfg)
            U, S = m["U"], m["S"]
            if U.sum() < tc["vis_min_pixels"] or S.sum() < tc["vis_min_pixels"]:
                continue
            if any_prov:
                pairs = (m["seen_mat"], m["unknown_mat"], m["valid"])
        elif rng.random() < tc["p_clean"]:
            imgs_in = md["imgs"]
            U = np.zeros(md["depths"].shape, bool)
            S = md["depths"] > 0
        elif labels in ("match", "matchself"):
            try:                          # three identical patches: imagined (U), observed-filled (Sm), clean S
                ev_p = [0.7, 0.15, 0.15] if all(v in EV_FOCUS for v, _, _ in specs) else None
                m = H.build_masks_matched(md, s, float(rng.uniform(*tc["radius_range"])), cfg,
                                          evidence=str(rng.choice(["1", "2", "half"], p=ev_p)))
            except (RuntimeError, ValueError) as e:      # no placement, or too few confident pixels in the prediction
                print(f"[oia] skip matched sample: {e}", flush=True)
                continue
            imgs_in = H.apply_fill(md["imgs"], m["hide"], str(rng.choice(tc["fills"])), s)
            U, S, S2 = m["U"], m["S"], m["Sm"]
            if S2.sum() < tc["vis_min_pixels"]:
                continue
        else:
            try:
                imgs_in, m = make_sample(md, s, float(rng.uniform(*tc["radius_range"])), str(rng.choice(tc["fills"])), cfg)
            except RuntimeError as e:
                print(f"[oia] skip sample: {e}", flush=True)
                continue
            U, S = m["U"], m["S"]
        F = md.pop("_F", None)            # visself: the labelling forward already produced the features
        if F is None:
            _, _, F, _, _ = forward(imgs_in, feats_mode)
        s += 1
        for hd in heads:
            loss, logit = step_loss(hd, F[hd["feats"]], U, S, pairs if hd["arch"] == "prov" else None, S2)
            hd["opt"].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hd["head"].parameters(), 1.0)
            hd["opt"].step(); hd["sched"].step()
            if s % tc["log_every"] == 0 or s == 1:
                lg = logit.detach().cpu().numpy()
                a = H.auroc(-lg[S], -lg[U]) if U.any() else float("nan")
                am = H.auroc(-lg[S2], -lg[U]) if (S2 is not None and S2.any() and U.any()) else float("nan")
                extra = f" train-auroc-m {am:.3f}" if S2 is not None else ""
                print(f"[oia] step {s:5d} {hd['variant']:8s} loss {loss.item():.4f} train-auroc {a:.3f}{extra} "
                      f"lr {hd['sched'].get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                hd["hist"].append(dict(step=s, loss=float(loss.item()), auroc=a, auroc_m=am))
        if s % tc["val_every"] == 0 or s == tc["steps"]:
            va = val_auroc()
            for hd in heads:
                print(f"[oia] step {s:5d} {hd['variant']:8s} val-auroc {va[hd['out']]:.3f}", flush=True)
                hd["hist"].append(dict(step=s, val_auroc=va[hd["out"]]))
                save(hd, s, va[hd["out"]], F[hd["feats"]])
    for hd in heads:
        json.dump(hd["hist"], open(hd["out"].replace(".pt", "_hist.json"), "w"), indent=1)
        print(f"[oia] done: {hd['out']} (last), best val-auroc {hd['best']:.3f}, {time.time() - t0:.0f}s", flush=True)
    return [hd["out"] for hd in heads]


# ----------------------------------------------------------------------------- evaluation wrapper
def resolve_ckpt(name):
    env = os.environ.get("OIA_CKPT_" + re.sub(r"\W", "_", name).upper())
    if env:
        return env
    base, var = name.rsplit("_", 1)
    if var.endswith("fuse"):                       # oiafuse, oiavisfuse, oiavisselffuse share the head's checkpoint
        return resolve_ckpt(f"{base}_{var[:-4]}")
    return os.path.join(os.environ.get("OIA_CKPT_DIR", "checkpoints"), f"{name}.pt")


class OIAWrap:
    """hallu3d model: depth from the base model, confidence = 1 - P(imagined) from the support head."""

    def __init__(self, base, variant="oia", ckpt=None, forward_feats=None):
        torch = base.torch
        self.base, self.variant = base, variant          # variant may carry an extra-seed suffix, e.g. oiavis_s1
        self.name = f"{base.name}_{variant}"
        self.fuse = variant.endswith("fuse")
        path = ckpt or resolve_ckpt(self.name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{self.name}: checkpoint {path} not found (train with scripts/oia_head.py train)")
        ck = torch.load(path, map_location=base.device, weights_only=False)
        self.feats, self.ckpt, self.pick = ck["feats"], path, ck.get("layer_pick")
        self.head = build_head(torch, ck["in_dims"], ck["arch"], ck["width"]).to(base.device).eval()
        self.head.load_state_dict(ck["state"])
        self.has_pairs = ck["arch"] == "prov"
        self.ff = forward_feats or FORWARD_FEATS[base.name]
        print(f"[{self.name}] head from {path} (step {ck.get('step')}, val-auroc {ck.get('val_auroc', float('nan')):.3f})")

    def predict(self, imgs, model_hw, out_hw):
        torch = self.base.torch
        x = self.base._prep(imgs, model_hw)
        with torch.no_grad():
            d, c, F, _, _ = self.ff(self.base, x, self.base.dtype, self.feats)
            if not torch.isfinite(d).all():
                d, c, F, _, _ = self.ff(self.base, x, None, self.feats)
            F = [f.float() for f in F]
            if self.pick is not None:
                F = [F[self.pick]]
            logit = upsample_logit(torch, self.head(F), out_hw)
            score = 1.0 - torch.sigmoid(logit.float())
            d = H._resize_t(d[:, None].float(), out_hw, "area")[:, 0]
            if self.fuse:
                cn = H._resize_t(c[:, None].float(), out_hw, "area")[:, 0]
                cn = (cn - cn.min()) / (cn.max() - cn.min() + 1e-9)
                score = torch.sqrt(score.clamp_min(1e-6) * cn.clamp_min(1e-6))
        return d.cpu().numpy().astype(np.float32), score.cpu().numpy().astype(np.float32)

    def predict_pairs(self, imgs, model_hw, out_hw):
        """Provenance heads only: depth, support score and P (N,N,h,w), P[i, j] = probability that input view j
        corroborates pixel p of view i (diagonal 0)."""
        if not self.has_pairs:
            raise AttributeError(f"{self.name} is not a provenance head")
        torch = self.base.torch
        x = self.base._prep(imgs, model_hw)
        with torch.no_grad():
            d, c, F, _, _ = self.ff(self.base, x, self.base.dtype, self.feats)
            if not torch.isfinite(d).all():
                d, c, F, _, _ = self.ff(self.base, x, None, self.feats)
            F = [f.float() for f in F]
            pl = self.head.pairs(F)
            logit = upsample_logit(torch, self.head.aggregate(pl), out_hw)
            score = 1.0 - torch.sigmoid(logit.float())
            P = torch.sigmoid(upsample_pairs(torch, pl, tuple(out_hw)).float())
            N = P.shape[0]
            P = P * (~torch.eye(N, dtype=torch.bool, device=P.device))[:, :, None, None]
            d = H._resize_t(d[:, None].float(), out_hw, "area")[:, 0]
        return (d.cpu().numpy().astype(np.float32), score.cpu().numpy().astype(np.float32),
                P.cpu().numpy().astype(np.float32))

    def free(self):
        self.base.free()


def make_oia(name, registry=None):
    """'vggt_oia' -> OIAWrap(VGGTWrap(), 'oia'); 'vggt_oiavis_s1' -> the extra-seed checkpoint of the same variant
    (checkpoints/vggt_oiavis_s1.pt). Used by hallu3d.make_model."""
    # <base>_<variant>[_s<seed>][_x<tag>]: _s is an extra initialisation seed, _x a head of the same variant trained on
    # another root (label-free adaptation, e.g. vggt_oiavisself_xco3dA trained on CO3Dv2 fold A, scripts/run_h100.py)
    m = re.fullmatch(r"(.+?)_((?:" + "|".join(VARIANTS) + r")(?:_s\d+)?(?:_x[A-Za-z0-9]+)?)", name)
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
    t.add_argument("--variant", default="oia",
                   help="one variant, or a comma list of variants with the same label source trained jointly on the "
                        "same forwards (e.g. oia,oialin,oiapatch,oiaattn or oiavis,oiavispatch,oiavislin)")
    t.add_argument("--extra-seeds", default="", help="extra initialisation seeds for the first variant, e.g. 1,2 "
                                                     "(saved as <base>_<variant>_s<seed>.pt)")
    t.add_argument("--extra-seeds-of", default=None, help="the variant the extra seeds belong to (default: the first "
                                                          "of --variant; it may be a variant that is not otherwise trained)")
    t.add_argument("--resume", action="store_true", help="continue from the checkpoints at the output paths")
    t.add_argument("--dtu", required=True, help="DTU root (Cameras/train, Rectified, Depths) holding the training scans")
    t.add_argument("--out", default=None, help="checkpoint path (one variant) or directory (several)")
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--scans", default=None, help="comma-separated subset of training scans (DTU numbers or CO3D-format scene names)")
    t.add_argument("--val-scans", default=None, help="comma-separated validation scans (default: DTU 2, 6; none on a CO3D-format root)")
    t.add_argument("--tag", default=None, help="checkpoint name suffix _x<tag> for heads trained on another root "
                                               "(label-free adaptation; the model name carries the same suffix)")
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
    _scan = lambda s: int(s) if s.strip().lstrip("-").isdigit() else s.strip()
    try:
        root = H.find_dtu_root((a.dtu,)) if os.path.isdir(a.dtu) else a.dtu
    except FileNotFoundError:
        root = a.dtu                                   # a CO3D-format root (scenes with meta.npz)
    co3d_root = not os.path.isdir(os.path.join(root, "Rectified"))
    scans = [_scan(s) for s in a.scans.split(",")] if a.scans else None
    if a.val_scans is not None:
        over["val_scans"] = [_scan(s) for s in a.val_scans.split(",") if s.strip()]
    elif co3d_root:
        over["val_scans"] = []
    variants = [v for v in a.variant.split(",") if v]
    extra = [int(x) for x in a.extra_seeds.split(",") if x]
    esv = a.extra_seeds_of or variants[0]
    for v in variants + [esv]:
        assert v in VARIANTS and v != "oiafuse", v
    tag = f"_x{a.tag}" if a.tag else ""
    if len(variants) == 1 and not extra:
        out = a.out or os.path.join("checkpoints", f"{a.base}_{variants[0]}{tag}.pt")
        if os.path.isdir(out):                      # --out may be the checkpoint directory (run_h100 passes it so)
            out = os.path.join(out, f"{a.base}_{variants[0]}{tag}.pt")
        train_multi(a.base, root, [(variants[0], a.seed, out)], a.steps, scans=scans, resume=a.resume, **over)
        return
    d = a.out or "checkpoints"
    specs = [(v, a.seed, os.path.join(d, f"{a.base}_{v}{tag}.pt")) for v in variants]
    specs += [(esv, sd, os.path.join(d, f"{a.base}_{esv}_s{sd}{tag}.pt")) for sd in extra]
    train_multi(a.base, root, specs, a.steps, scans=scans, resume=a.resume, **over)


if __name__ == "__main__":
    main()
