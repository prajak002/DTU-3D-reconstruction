"""AnySplat (feed-forward 3D Gaussian splatting from unposed views) as hallu3d backbones.

AnySplat: Jiang et al., "AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views", SIGGRAPH Asia 2025.
Code https://github.com/InternRobotics/AnySplat, weights https://huggingface.co/lhjiang/anysplat (1.19 B params; it
embeds the VGGT-1B aggregator + camera head + depth head, plus a DPT Gaussian-parameter head).

Two wrappers (both implement the hallu3d.ModelWrap contract: predict(imgs, model_hw, out_hw) -> depth, conf):

  AnySplatWrap        name "anysplat"
      depth = z, in each input camera's own frame, of the *pixel-aligned* Gaussian centers predicted for that view
      conf  = the opacity of that pixel's Gaussian
  AnySplatRenderWrap  name "anysplat_render"   (needs the gsplat rasterizer; see install notes)
      depth = depth of the released (voxel-fused) Gaussian scene *rendered* into each input view with the model's own
              predicted camera (gsplat expected depth = alpha-weighted mean depth along the ray)
      conf  = rendered accumulated alpha of that pixel

Tensors used (all from EncoderAnySplat.forward, i.e. model.encoder(x[None]) with x (N,3,H,W) in [0,1]; B = 1):
  out.gaussians.means        (B, G, 3)   Gaussian centers in the model's world frame (= camera 0, OpenCV convention).
                                         With cfg.voxelize=False (what AnySplatWrap sets) G = N*H*W and the centers
                                         are ordered (view, row, col): they are exactly the VGGT depth-head depth
                                         unprojected with the predicted cameras (anysplat.py: pts_all =
                                         batchify_unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)),
                                         so means.view(N,H,W,3) is pixel aligned. With the released setting
                                         cfg.voxelize=True (what AnySplatRenderWrap keeps) G <= N*H*W and the centers
                                         are confidence-weighted voxel averages (voxel_size 0.002) - not pixel aligned.
  out.gaussians.opacities    (B, G)      opacity in (0,1): sigmoid of the first channel of the Gaussian-parameter head,
                                         passed through map_pdf_to_opacity which is the identity for the released
                                         opacity_mapping (initial = final = 0 -> exponent 1).
  out.gaussians.scales       (B, G, 3), .rotations (B, G, 4), .covariances (B, G, 3, 3), .harmonics (B, G, 3, 25):
                                         used only by the renderer.
  out.pred_context_pose["extrinsic"] (B, N, 4, 4)  camera -> world (c2w) of each input view, from the VGGT camera head.
  out.pred_context_pose["intrinsic"] (B, N, 3, 3)  pinhole intrinsics normalised by (W, H): fx/W, fy/H, cx/W, cy/H.
  out.depth_dict["depth"]    (B, N, H, W, 1)  the VGGT depth-head depth; AnySplatWrap's center depth equals it to ~1e-3
                                         (float round trip through the c2w inverse) and is only used as a cross-check.
The encoder runs its aggregator in bf16 and its camera/depth heads in fp32 regardless of the outer autocast; the
outer bf16 autocast requested by ModelWrap only affects the Gaussian-parameter head (opacity differs from fp32 by
< 0.02 on DTU scan 3; the depth is unchanged). fp32 fallback on non-finite output as in VGGTWrap.

Caveats
  * Input resolution: whatever model_hw hallu3d passes (420x518, multiples of 14); AnySplat was trained on 448x448
    style crops but its VGGT backbone handles any multiple of 14 and the DTU result is sane (AbsRel 0.016 on view 0
    of scan 3 after median scaling).
  * The depth of AnySplatWrap is the VGGT-style depth-head depth in the view's own camera frame, as the Gaussian centers
    are built from it; it is not a rendered depth. AnySplatRenderWrap gives the rendered one.
  * The AnySplat repo is a top-level package called `src`; it is put on sys.path by _import_anysplat() (set
    ANYSPLAT_ROOT if it is not at ~/ext/AnySplat). Nothing else on the hallu path is called `src`.
  * Three compiled dependencies of the repo are replaced by in-process shims (installed into sys.modules only when the
    real package is missing): xformers (imported at module level by heads/head_modules.py; the released checkpoint's
    head never calls it, the shim maps memory_efficient_attention to torch SDPA), torch_scatter (scatter_add /
    scatter_max, used by the voxel fusion; implemented with index_add_ / scatter_reduce_), and gsplat (stub so the
    model module imports; AnySplatRenderWrap refuses to construct when the real gsplat is unavailable).
  * The encoder prints statistics on every forward; they are swallowed (stdout redirected during the call).

Install (what worked on the L4 box, Ubuntu 24.04, /opt/pytorch venv with torch 2.12.1+cu130, CUDA 13.0 toolkit at
/opt/pytorch/cuda; nothing below touches torch / torchvision / numpy / opencv):
    mkdir -p ~/ext && cd ~/ext && git clone --depth 1 https://github.com/InternRobotics/AnySplat.git
    pip install ninja jaxtyping dacite opt_einsum colorspacious svg.py lightning wandb scikit-image
    pip install --no-deps e3nn opt_einsum_fx scikit-video lpips
    # gsplat (only for AnySplatRenderWrap): pure-python wheel, the CUDA kernels are JIT-built on first use (~15 min,
    # ninja + nvcc, cached under ~/.cache/torch_extensions/*/gsplat_cuda).
    pip install --no-build-isolation --no-deps gsplat              # 1.5.3
    export PATH=/opt/pytorch/cuda/bin:$PATH CUDA_HOME=/opt/pytorch/cuda TORCH_CUDA_ARCH_LIST="8.9"
    python -c "from gsplat import rasterization"                    # triggers / checks the JIT build
Usage:
    import hallu3d as H, anysplat_wrap as A; A.register(H.MODEL_REGISTRY); w = H.make_model("anysplat")
"""
import contextlib
import io
import os
import sys
import types

import numpy as np

import hallu3d as H

ANYSPLAT_ROOT = os.environ.get("ANYSPLAT_ROOT", os.path.expanduser("~/ext/AnySplat"))
HF_ID = "lhjiang/anysplat"


# ----------------------------------------------------------------------------- import shims
def _shim_xformers():
    import torch.nn.functional as F
    xf, ops = types.ModuleType("xformers"), types.ModuleType("xformers.ops")
    fmha, flash = types.ModuleType("xformers.ops.fmha"), types.ModuleType("xformers.ops.fmha.flash")
    flash.FwOp = flash.BwOp = None
    fmha.flash, ops.fmha, xf.ops = flash, fmha, ops

    def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None, op=None):
        # xformers layout (B, N, heads, dim) -> torch SDPA layout (B, heads, N, dim)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=attn_bias, dropout_p=p, scale=scale)
        return o.transpose(1, 2)

    ops.memory_efficient_attention = memory_efficient_attention
    sys.modules.update({"xformers": xf, "xformers.ops": ops, "xformers.ops.fmha": fmha,
                        "xformers.ops.fmha.flash": flash})


def _shim_torch_scatter():
    import torch
    ts = types.ModuleType("torch_scatter")

    def _n(index, dim_size):
        return dim_size if dim_size is not None else int(index.max()) + 1

    def scatter_add(src, index, dim=0, dim_size=None):
        assert dim == 0
        out = torch.zeros((_n(index, dim_size),) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        return out.index_add_(0, index, src)

    def scatter_max(src, index, dim=0, dim_size=None):
        assert dim == 0
        out = torch.full((_n(index, dim_size),) + tuple(src.shape[1:]), float("-inf"), dtype=src.dtype,
                         device=src.device)
        idx = index if src.dim() == 1 else index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        return out.scatter_reduce_(0, idx, src, "amax", include_self=False), None   # argmax not used by AnySplat

    ts.scatter_add, ts.scatter_max = scatter_add, scatter_max
    sys.modules["torch_scatter"] = ts


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _import_anysplat():
    if ANYSPLAT_ROOT not in sys.path:
        sys.path.insert(0, ANYSPLAT_ROOT)
    if not _have("xformers"):
        _shim_xformers()
    if not _have("torch_scatter"):
        _shim_torch_scatter()
    if not _have("gsplat"):
        g = types.ModuleType("gsplat")
        g.rasterization = None
        sys.modules["gsplat"] = g
    from src.model.model.anysplat import AnySplat
    return AnySplat


def _gsplat_rasterization():
    """The real gsplat rasterizer or None (the JIT build of the kernels is attempted here)."""
    try:
        import gsplat
        from gsplat import rasterization
        if rasterization is None or not hasattr(gsplat, "__version__"):
            return None
        return rasterization
    except Exception as e:                       # noqa: BLE001 - any build / import failure means "no renderer"
        print(f"[anysplat] gsplat unavailable: {e!r}")
        return None


# ----------------------------------------------------------------------------- wrappers
class AnySplatWrap(H.ModelWrap):
    """Depth of the pixel-aligned Gaussian centers in each input camera's frame; conf = Gaussian opacity.
    Uses out.gaussians.means (1, N*H*W, 3) and out.gaussians.opacities (1, N*H*W) with cfg.voxelize=False, and
    out.pred_context_pose["extrinsic"] (1, N, 4, 4) c2w to move the centers into their own camera. See module doc."""
    name = "anysplat"
    voxelize = False

    def __init__(self, device="cuda"):
        super().__init__(device)
        AnySplat = _import_anysplat()
        self.model = AnySplat.from_pretrained(HF_ID).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.encoder.distill = False
        self.model.encoder.cfg.voxelize = self.voxelize

    def _encode(self, x, dtype):
        """x (N,3,H,W) in [0,1] -> EncoderOutput. The outer autocast only touches the Gaussian-parameter head.
        The output is kept in self._last_out so _forward_cams does not need a second forward."""
        torch = self.torch
        with contextlib.redirect_stdout(io.StringIO()):          # the encoder prints stats every forward
            if dtype is None:
                out = self.model.encoder(x[None], global_step=0)
            else:
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = self.model.encoder(x[None], global_step=0)
        self._last_out = out
        return out

    def _cams(self, out, hw):
        """Predicted cameras: K (N,3,3) in pixels at hw, E (N,4,4) world -> camera (OpenCV, world = camera 0)."""
        torch = self.torch
        h, w = hw
        c2w = out.pred_context_pose["extrinsic"][0].float()
        E = torch.linalg.inv(c2w)
        K = out.pred_context_pose["intrinsic"][0].float().clone()
        K[:, 0] *= w
        K[:, 1] *= h
        return K, E

    def _forward(self, x, dtype):
        torch = self.torch
        N, _, h, w = x.shape
        out = self._encode(x, dtype)
        K, E = self._cams(out, (h, w))
        means = out.gaussians.means[0].float().view(N, h, w, 3)                     # world frame, pixel aligned
        pc = torch.einsum("nij,nhwj->nhwi", E[:, :3, :3], means) + E[:, None, None, :3, 3]
        depth = pc[..., 2].clamp_min(0)
        conf = out.gaussians.opacities[0].float().view(N, h, w)
        return depth, conf

    def _forward_cams(self, x, dtype):
        """Depth plus the model's own cameras, for hallu3d.ReprojWrap (same contract as VGGTWrap._forward_cams)."""
        d, _ = self._forward(x, dtype)
        K, E = self._cams(self._last_out, x.shape[-2:])
        return d, K, E

    def free(self):
        self._last_out = None
        super().free()


class AnySplatRenderWrap(AnySplatWrap):
    """Released (voxel-fused) Gaussians rendered back into every input view with the predicted cameras via gsplat:
    depth = expected depth (gsplat render_mode "RGB+ED": alpha-weighted mean depth along the ray, in the view's own
    camera frame), conf = accumulated alpha (1,H,W per view). Rasterizer arguments mirror the repo's
    DecoderSplattingCUDA.rendering_fn (covars from the adapter, radius_clip 0.1, near_plane 1e-10, classic mode,
    white background) except for the depth mode: the decoder's "RGB+D" is the alpha-weighted *sum*, which goes to 0
    where alpha is low and would confound depth error with coverage."""
    name = "anysplat_render"
    voxelize = True

    def __init__(self, device="cuda"):
        self.rasterization = _gsplat_rasterization()
        if self.rasterization is None:
            raise ImportError("anysplat_render needs a working gsplat (see anysplat_wrap module docstring)")
        super().__init__(device)

    def _forward(self, x, dtype):
        torch = self.torch
        N, _, h, w = x.shape
        out = self._encode(x, dtype)
        K, E = self._cams(out, (h, w))
        g = out.gaussians
        means, quats, scales = g.means[0].float(), g.rotations[0].float(), g.scales[0].float()
        opac, covars = g.opacities[0].float(), g.covariances[0].float()
        sh = g.harmonics[0].float().permute(0, 2, 1).contiguous()                  # (G, 25, 3) gsplat SH layout
        sh_degree = int(round(sh.shape[1] ** 0.5)) - 1
        bg = self.model.decoder.background_color.to(x.device)[None]
        depths, alphas = [], []
        for j in range(N):                                                         # one camera at a time (memory)
            rgbd, alpha, _ = self.rasterization(means, quats, scales, opac, sh, E[j:j + 1], K[j:j + 1], w, h,
                                                sh_degree=sh_degree, render_mode="RGB+ED", packed=False,
                                                near_plane=1e-10, backgrounds=bg, radius_clip=0.1, covars=covars,
                                                rasterize_mode="classic")
            depths.append(rgbd[0, ..., 3])
            alphas.append(alpha[0, ..., 0])
        depth = torch.stack(depths).clamp_min(0)
        conf = torch.stack(alphas)
        depth = torch.where(conf > 0, depth, torch.zeros_like(depth))               # nothing rendered -> invalid
        return depth, conf


def register(registry):
    """registry: hallu3d.MODEL_REGISTRY (name -> wrapper class). Adds both wrappers; the render one is registered
    even when gsplat is missing so that make_model gives a clear ImportError instead of a KeyError."""
    registry[AnySplatWrap.name] = AnySplatWrap
    registry[AnySplatRenderWrap.name] = AnySplatRenderWrap
    return registry
