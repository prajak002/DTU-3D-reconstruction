"""Diagnosis of the HAD scorer wrapper (scripts/had_wrap.py), no labels involved (LOG.md, fifteenth pass).

On one DTU capture: (1) do the checkpoint's keys match the model; (2) render PSNR of each view from the others at
several resolutions, in half and in full precision; (3) identity test: a reference view rendered at its own pose.
Writes a PNG per configuration (photograph | rendering) under --out.

  HAD_ROOT=... HAD_CKPT=... python scripts/had_diag.py --dtu <root> --out results/P22-had-diag"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
from had_wrap import HADWrap  # noqa: E402


class _Base:
    name, torch = "diag", __import__("torch")

    def free(self):
        pass


def main():
    import torch
    from easydict import EasyDict as edict
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--scan", type=int, default=37)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("results", "P22-had-diag"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = H.find_dtu_root((a.dtu,))
    cfg = dict(H.DEFAULT_CFG)
    data = H.load_scan(root, a.scan, H.vis_views(a.views, 0), cfg["light"])
    w = HADWrap(_Base())
    w._load()
    R, model, dev = w.R, w._model, w.dev

    # (1) checkpoint keys
    ck = torch.load(os.environ["HAD_CKPT"], map_location="cpu", weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    shape = [k for k in own if k in sd and tuple(own[k].shape) != tuple(sd[k].shape)]
    print(f"[keys] model {len(own)}, checkpoint {len(sd)}, missing {len(missing)}, unexpected {len(unexpected)}, shape mismatch {len(shape)}")
    for name, lst in (("missing", missing), ("unexpected", unexpected), ("shape", shape)):
        groups = sorted({k.split(".")[0] for k in lst})
        print(f"[keys] {name}: top-level modules {groups}; first {lst[:6]}")
    if isinstance(ck, dict):
        print("[keys] checkpoint top-level entries:", [k for k in ck.keys()][:12])

    N = len(data["imgs"])
    Kimg, E = data["K_img"].astype(np.float64), data["E"].astype(np.float64)
    c2w_all = torch.from_numpy(np.linalg.inv(E).astype(np.float32)).to(dev)
    H0, W0 = data["imgs"].shape[1:3]

    def run(hw, i, refs, half, tag, scale=1.35):
        h, wd = hw
        ims = np.stack([cv2.resize(im, (wd, h), interpolation=cv2.INTER_AREA) for im in data["imgs"]])
        t = torch.from_numpy(ims).permute(0, 3, 1, 2).float().clamp(0, 1).to(dev)
        K = Kimg.copy()
        K[:, 0] *= wd / W0
        K[:, 1] *= h / H0
        intr = torch.from_numpy(np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], -1).astype(np.float32)).to(dev)
        poses = R.preprocess_poses(torch.cat([c2w_all[refs], c2w_all[i:i + 1]], 0).clone(), scale)
        rc, tc = poses[:-1][None], poses[-1:][None]
        ri, ti = intr[refs][None], intr[i:i + 1][None]
        ro, rd = R.compute_rays(rc, ri, h, wd, dev)
        to, td = R.compute_rays(tc, ti, h, wd, dev)
        tgt_img = t[i:i + 1][None]
        inp = edict(image=t[refs][None], c2w=rc, fxfycxcy=ri, ray_o=ro.float(), ray_d=rd.float(), image_h_w=[h, wd],
                    difix3D_image=tgt_img, pred_image=tgt_img)
        tg = edict(c2w=tc, fxfycxcy=ti, ray_o=to.float(), ray_d=td.float(), image_h_w=[h, wd],
                   difix3D_image=tgt_img, pred_image=tgt_img)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=half and dev == "cuda"):
            out = model.forward_direct(inp, tg, has_target_image=False)
        r = out["render"][0, 0].float()
        print(f"[{tag}] view {i} refs {refs}: render range [{float(r.min()):.2f}, {float(r.max()):.2f}] mean {float(r.mean()):.3f}", end="  ")
        r = r.clamp(0, 1)
        ps = -10 * np.log10(max(float(((r - t[i]) ** 2).mean()), 1e-10))
        print(f"PSNR {ps:.2f} dB, mean scorer conf {float(out['difix3D_conf'][0, 0, 0].float().mean()):.3f}", flush=True)
        both = np.concatenate([t[i].permute(1, 2, 0).cpu().numpy(), r.permute(1, 2, 0).cpu().numpy()], 1)
        cv2.imwrite(os.path.join(a.out, f"{tag}_v{i}.png"), cv2.cvtColor((both * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return ps

    others = lambda i: [j for j in range(N) if j != i]
    for hw in ((408, 512), (512, 512), (288, 512), (256, 320)):
        for half in (True, False):
            tag = f"{hw[0]}x{hw[1]}_{'fp16' if half else 'fp32'}"
            ps = [run(hw, i, others(i), half, tag) for i in (0, 2)]
            print(f"== {tag}: mean PSNR {np.mean(ps):.2f} dB", flush=True)
    # (3) identity: view 0 is among the references and is also the target
    ps = run((408, 512), 0, list(range(N)), False, "identity_408x512_fp32")
    print(f"== identity (target is one of the references): PSNR {ps:.2f} dB", flush=True)
    for sc in (1.0, 2.0, 4.0):
        ps = run((408, 512), 2, others(2), False, f"scale{sc}_408x512_fp32", sc)
        print(f"== scene scale factor {sc}: PSNR {ps:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
