"""HAD's hallucination scoring network as a detector of unobserved geometry (baseline, zero-shot).

HAD (Liu et al., CVPR 2026, github.com/xiliu8006/HAD) scores, per pixel, whether an image at a known pose is
consistent with a set of posed reference views; it was trained to find the hallucinations of diffusion-refined
renders. Here every input photograph is scored against the other input photographs: the references are the other
N - 1 views with their cameras, the scored image is the photograph of view i itself. A pixel no other view sees
cannot be predicted from the references, so a low score is the scorer's way of saying "unsupported".

Two things differ from the detectors it is compared with, and both favour HAD: it is given the ground-truth
cameras (the backbones and our head get none), and it runs one extra 24-layer transformer forward per view.
The depth returned is the base backbone's, as for the other training-free detectors; only the confidence changes.

Needs a checkout of the HAD repository and its released checkpoint:
  HAD_ROOT  directory of the checkout (contains LVSM/ and scripts/run_lvsm_hallucination.py)
  HAD_CKPT  the checkpoint file or its directory
Their code is imported, not copied (it is CC BY-NC 4.0)."""
import os
import sys
import types

import cv2
import numpy as np


def _stub(name, **attrs):
    """Their loss module imports training-only packages at import time; none is used at inference."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, m)


def had_hw(out_hw, long_side=512, patch=8):
    """Scorer resolution: long side 512 (the training size), the capture's aspect ratio, multiples of the patch."""
    h, w = out_hw
    s = long_side / max(h, w)
    return max(patch, int(round(h * s / patch)) * patch), max(patch, int(round(w * s / patch)) * patch)


class HADWrap:
    def __init__(self, base, max_refs=None):
        self.base = base
        self.name = f"{base.name}_had"
        self.max_refs = max_refs
        self.K = self.E = None
        self._model = None
        self.psnr = []          # render PSNR of every scored view (see _score_view)

    # the evaluators call this before predict (ground-truth intrinsics at the label resolution, world-to-camera E)
    def set_cams(self, K, E):
        self.K, self.E = np.asarray(K, np.float64), np.asarray(E, np.float64)

    def _load(self):
        if self._model is not None:
            return
        root = os.environ.get("HAD_ROOT")
        ckpt = os.environ.get("HAD_CKPT")
        if not root or not ckpt:
            raise RuntimeError("HAD_ROOT and HAD_CKPT must point at the HAD checkout and its checkpoint")
        for n in ("fused_ssim", "piq", "piq.functional"):
            _stub(n, fused_ssim=None, gaussian_filter=None)
        sys.path.insert(0, os.path.join(root, "scripts"))
        import run_lvsm_hallucination as R
        self.R = R
        self.torch = self.base.torch
        self.dev = "cuda" if self.torch.cuda.is_available() else "cpu"
        self._model = R.load_lvsm(os.path.join(root, "LVSM"), ckpt, self.dev)

    def _score_view(self, imgs_t, c2w, intr, i, refs, hw):
        torch, R = self.torch, self.R
        from easydict import EasyDict as edict
        h, w = hw
        poses = R.preprocess_poses(torch.cat([c2w[refs], c2w[i:i + 1]], 0).clone())
        ref_c2w, tgt_c2w = poses[:-1][None], poses[-1:][None]
        ref_intr, tgt_intr = intr[refs][None], intr[i:i + 1][None]
        ro, rd = R.compute_rays(ref_c2w, ref_intr, h, w, self.dev)
        to, td = R.compute_rays(tgt_c2w, tgt_intr, h, w, self.dev)
        target_img = imgs_t[i:i + 1][None]
        inp = edict(image=imgs_t[refs][None], c2w=ref_c2w, fxfycxcy=ref_intr, ray_o=ro.float(), ray_d=rd.float(),
                    image_h_w=[h, w], difix3D_image=target_img, pred_image=target_img)
        tgt = edict(c2w=tgt_c2w, fxfycxcy=tgt_intr, ray_o=to.float(), ray_d=td.float(), image_h_w=[h, w],
                    difix3D_image=target_img, pred_image=target_img)
        with torch.no_grad(), torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
            out = self._model.forward_direct(inp, tgt, has_target_image=False)
        # validity of the camera convention, independent of any label: the scorer's own rendering of view i from
        # the references must resemble the photograph of view i
        mse = float(((out["render"][0, 0].float().clamp(0, 1) - target_img[0, 0]) ** 2).mean())
        self.psnr.append(-10.0 * np.log10(max(mse, 1e-10)))
        return out["difix3D_conf"][0, 0, 0].float().cpu().numpy()

    def predict(self, imgs, model_hw, out_hw):
        if self.K is None:
            raise RuntimeError("HADWrap.predict called before set_cams")
        self._load()
        torch = self.torch
        d, _ = self.base.predict(imgs, model_hw, out_hw)
        N = len(imgs)
        hw = had_hw(out_hw)
        ims = np.stack([cv2.resize(np.asarray(im, np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for im in imgs])
        imgs_t = torch.from_numpy(ims).permute(0, 3, 1, 2).float().clamp(0, 1).to(self.dev)
        c2w = torch.from_numpy(np.linalg.inv(self.E).astype(np.float32)).to(self.dev)
        K = self.K.copy()
        K[:, 0] *= hw[1] / out_hw[1]
        K[:, 1] *= hw[0] / out_hw[0]
        intr = torch.from_numpy(np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], -1).astype(np.float32)).to(self.dev)
        centres = c2w[:, :3, 3].cpu().numpy()
        conf = np.zeros((N, *out_hw), np.float32)
        for i in range(N):
            order = [j for j in np.argsort(np.linalg.norm(centres - centres[i], axis=1)) if j != i]
            k = len(order) if self.max_refs is None else min(len(order), self.max_refs)
            while True:
                try:
                    c = self._score_view(imgs_t, c2w, intr, i, order[:k], hw)
                    break
                except RuntimeError as e:           # out of memory with many references: keep the nearest ones
                    if "out of memory" not in str(e).lower() or k <= 2:
                        raise
                    torch.cuda.empty_cache()
                    k = max(2, k // 2)
                    print(f"[{self.name}] out of memory, view {i}: scoring against the {k} nearest references", flush=True)
            conf[i] = cv2.resize(c, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
        self.K = self.E = None
        print(f"[{self.name}] {N} views at {hw[0]}x{hw[1]}: render PSNR {np.mean(self.psnr[-N:]):.2f} dB "
              f"(all captures so far {np.mean(self.psnr):.2f} dB, {len(self.psnr)} views)", flush=True)
        return d, conf

    def free(self):
        self._model = None
        self.base.free()
