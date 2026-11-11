"""Novel-view test: does dropping the points a detector trusts least, before Gaussian splatting is fitted to the
input views, improve the rendering of views the model never saw?

For every (DTU scan, seed, N input views) and backbone: predicted depth and cameras -> one point per pixel, aligned
to the rig with a similarity fitted on the input camera centres; four initial point sets
  all      every valid predicted point
  native   without the --drop share with the lowest native confidence
  head     without the --drop share with the lowest support score (label-free head <backbone>_oiavisself)
  random   without a random --drop share (control)
are each turned into isotropic Gaussians (colour of the pixel, scale from the neighbour distance) and optimised with
identical settings on the N input images at their rig cameras (no densification, so the point set is the only
difference). Eight rig views that are not inputs are rendered and scored against the photographs: PSNR, SSIM and
LPIPS on the full image, and PSNR / LPIPS on the pixels that all four variants cover (accumulated alpha > 0.5), so
that a variant is not rewarded or punished for holes alone. train_psnr (input views) is the sanity check of the
alignment.

  python scripts/nvs_eval.py --dtu <root> --models vggt --scans 24,37,40 --views 3,6 --out results/P19-nvs/pilot
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
from fuse_eval import umeyama  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ("all", "native", "head", "random")


def held_out_views(views, n_held=8, total=49):
    rest = [v for v in range(total) if v not in set(views)]
    return [rest[int(i)] for i in np.round(np.linspace(0, len(rest) - 1, n_held))]


def scale_K(K, src_hw, dst_hw):
    K = np.array(K, np.float64).copy()
    K[0] *= dst_hw[1] / src_hw[1]; K[1] *= dst_hw[0] / src_hw[0]
    return K


def keep_mask(valid, score, drop, mode, rng):
    """valid points minus the drop share chosen by mode ("all": none; "random"; else lowest score)."""
    keep = valid.copy()
    idx = np.flatnonzero(valid)
    k = int(round(drop * len(idx)))
    if mode == "all" or k == 0:
        return keep
    out = rng.choice(idx, k, replace=False) if mode == "random" else idx[np.argsort(score[idx], kind="stable")[:k]]
    keep[out] = False
    return keep


def ssim_t(torch, a, b):
    """Mean SSIM of two (C,3,H,W) images in [0,1], 11x11 Gaussian window."""
    F = torch.nn.functional
    g = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-g ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None]).expand(3, 1, 11, 11).contiguous()
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=3), F.conv2d(b, w, padding=5, groups=3)
    va = F.conv2d(a * a, w, padding=5, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()


def fit_and_render(torch, pts, rgb, train, test, hw, steps, seed):
    """Isotropic Gaussians from (pts, rgb), optimised on the train views; returns renders and alphas of both sets.
    train / test: dict(img (C,H,W,3) float tensor, viewmat (C,4,4), K (C,3,3))."""
    from gsplat import rasterization
    from scipy.spatial import cKDTree
    dev = train["img"].device
    g = torch.Generator(device="cpu").manual_seed(seed)
    d3 = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(1)
    d3 = np.clip(d3, np.percentile(d3, 1), np.percentile(d3, 99))
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    P = dict(means=torch.tensor(pts, dtype=torch.float32, device=dev),
             log_s=torch.tensor(np.log(d3), dtype=torch.float32, device=dev)[:, None].repeat(1, 3),
             quat=torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(pts), 1),
             op=torch.full((len(pts),), 0.0, device=dev),                          # sigmoid -> 0.5
             col=torch.logit(torch.tensor(rgb, dtype=torch.float32, device=dev).clamp(0.01, 0.99)))
    for v in P.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam([dict(params=[P["means"]], lr=1.6e-4 * extent), dict(params=[P["log_s"]], lr=5e-3),
                            dict(params=[P["quat"]], lr=1e-3), dict(params=[P["op"]], lr=5e-2),
                            dict(params=[P["col"]], lr=2.5e-3)], eps=1e-15)

    def render(s):
        out, alpha, _ = rasterization(P["means"], torch.nn.functional.normalize(P["quat"], dim=-1), torch.exp(P["log_s"]),
                                      torch.sigmoid(P["op"]), torch.sigmoid(P["col"]), s["viewmat"], s["K"],
                                      hw[1], hw[0], packed=False)
        return out.clamp(0, 1), alpha[..., 0]

    C = len(train["img"])
    for it in range(steps):
        i = int(torch.randint(0, C, (1,), generator=g))
        one = dict(img=train["img"][i:i + 1], viewmat=train["viewmat"][i:i + 1], K=train["K"][i:i + 1])
        out, _ = render(one)
        a, b = out.permute(0, 3, 1, 2), one["img"].permute(0, 3, 1, 2)
        loss = 0.8 * (a - b).abs().mean() + 0.2 * (1 - ssim_t(torch, a, b))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        tr, _ = render(train)
        te, al = render(test)
    return tr, te, al


def psnr_t(torch, a, b, m=None):
    e = (a - b) ** 2
    mse = e.mean() if m is None else (e * m[..., None]).sum() / (3 * m.sum()).clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--models", default="vggt", help="backbones; each is scored with its <backbone>_oiavisself head")
    ap.add_argument("--views", default="3,6")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--hw", default="256,320", help="resolution of the points, the training images and the renders")
    ap.add_argument("--save-images", type=int, default=0, help="save the renders of the first k (scan, views) cells")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P19-nvs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").cuda().eval()
    except Exception as e:                      # reported as NaN, never silently replaced
        print("[nvs] LPIPS unavailable:", repr(e)); lp = None
    root = H.find_dtu_root((a.dtu,))
    scans = [int(x) for x in (a.scans.split(",") if a.scans else H.DEFAULT_CFG["scans"])]
    hw = tuple(int(x) for x in a.hw.split(","))
    csv = os.path.join(a.out, "P19-nvs.csv")
    rows = pd.read_csv(csv).to_dict("records") if os.path.exists(csv) else []
    done = {(r["model"], int(r["scan"]), int(r["seed"]), int(r["n_views"])) for r in rows}
    saved = 0
    for b in a.models.split(","):
        hm = H.make_model(f"{b}_oiavisself")
        base = hm.base
        for sc in scans:
            for sd in (int(x) for x in a.seeds.split(",")):
                for n in (int(x) for x in a.views.split(",")):
                    if (b, sc, sd, n) in done:
                        continue
                    t0 = time.time()
                    views = H.vis_views(n, sd)
                    held = held_out_views(views)
                    data, test = H.load_scan(root, sc, views), H.load_scan(root, sc, held)
                    model_hw = tuple(H.DEFAULT_CFG["model_hw"])
                    d, c_nat = base.predict(data["imgs"], model_hw, hw)
                    d2, c_head = hm.predict(data["imgs"], model_hw, hw)
                    x = base._prep(data["imgs"], model_hw)
                    with torch.no_grad():
                        _, Km, Em = base._forward_cams(x, base.dtype)
                    Km = np.asarray(Km.float().cpu().numpy(), np.float64).copy(); Em = np.asarray(Em.float().cpu().numpy(), np.float64)
                    Km[:, 0] *= hw[1] / model_hw[1]; Km[:, 1] *= hw[0] / model_hw[0]
                    pred = np.concatenate([H.backproject(d[i], Km[i], Em[i]).reshape(-1, 3) for i in range(n)])
                    C_pred = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Em])
                    C_gt = np.stack([-E[:3, :3].T @ E[:3, 3] for E in data["E"]])
                    s, R, t = umeyama(C_pred, C_gt)
                    pred = (s * (R @ pred.T)).T + t
                    rgb = np.concatenate([cv2.resize(im, hw[::-1], interpolation=cv2.INTER_AREA).reshape(-1, 3) for im in data["imgs"]])
                    valid = (d.reshape(-1) > 0) & np.isfinite(pred).all(1)
                    img_hw = data["imgs"].shape[1:3]

                    def pack(dd):
                        ims = np.stack([cv2.resize(im, hw[::-1], interpolation=cv2.INTER_AREA) for im in dd["imgs"]])
                        Ks = np.stack([scale_K(K, img_hw, hw) for K in dd["K_img"]])
                        return dict(img=torch.tensor(ims, dtype=torch.float32).cuda(),
                                    viewmat=torch.tensor(dd["E"], dtype=torch.float32).cuda(),
                                    K=torch.tensor(Ks, dtype=torch.float32).cuda())
                    tr_set, te_set = pack(data), pack(test)
                    rng = np.random.default_rng(1000 * sc + 10 * sd + n)
                    res = {}
                    for var in VARIANTS:
                        score = c_head if var == "head" else c_nat
                        keep = keep_mask(valid, score.reshape(-1).astype(np.float64), a.drop, var, rng)
                        tr, te, al = fit_and_render(torch, pred[keep], rgb[keep], tr_set, te_set, hw, a.steps, sd)
                        res[var] = dict(tr=tr, te=te, al=al, n=int(keep.sum()))
                    common = torch.stack([res[v]["al"] > 0.5 for v in VARIANTS]).all(0).float()
                    gt = te_set["img"]
                    for var in VARIANTS:
                        r = res[var]
                        row = dict(model=b, scan=sc, seed=sd, n_views=n, variant=var, n_points=r["n"], drop=a.drop,
                                   steps=a.steps, depth_maxdiff=float(np.abs(d - d2).max()),
                                   train_psnr=psnr_t(torch, r["tr"], tr_set["img"]),
                                   psnr=psnr_t(torch, r["te"], gt), psnr_common=psnr_t(torch, r["te"], gt, common),
                                   ssim=float(ssim_t(torch, r["te"].permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))),
                                   cover=float((r["al"] > 0.5).float().mean()), common=float(common.mean()),
                                   lpips=float("nan"), lpips_common=float("nan"))
                        if lp is not None:
                            with torch.no_grad():
                                A, B = r["te"].permute(0, 3, 1, 2) * 2 - 1, gt.permute(0, 3, 1, 2) * 2 - 1
                                m = common[:, None]
                                row["lpips"] = float(lp(A, B).mean())
                                row["lpips_common"] = float(lp(A * m, B * m).mean())
                        rows.append(row)
                    if saved < a.save_images:
                        saved += 1
                        strip = [np.concatenate([gt[k].cpu().numpy()] + [res[v]["te"][k].cpu().numpy() for v in VARIANTS], 1)
                                 for k in (0, 4)]
                        strip.append(np.concatenate([tr_set["img"][0].cpu().numpy()] +
                                                    [res[v]["tr"][0].cpu().numpy() for v in VARIANTS], 1))
                        cv2.imwrite(os.path.join(a.out, f"{b}_scan{sc}_s{sd}_v{n}.jpg"),
                                    (np.concatenate(strip, 0)[..., ::-1] * 255).astype(np.uint8))
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    last = {r["variant"]: r for r in rows[-4:]}
                    print(f"[{b}] scan {sc} s{sd} n{n} ({time.time() - t0:.0f}s) train {last['all']['train_psnr']:.1f} | " +
                          " ".join(f"{v} {last[v]['psnr']:.2f}/{last[v]['psnr_common']:.2f}/{last[v]['lpips']:.3f}" for v in VARIANTS),
                          flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        print(df.groupby(["model", "n_views", "variant"])[["train_psnr", "psnr", "psnr_common", "ssim", "lpips",
                                                           "lpips_common", "cover"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
