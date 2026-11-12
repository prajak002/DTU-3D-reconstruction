"""Causal test of the support signal: edit the backbone's cross-view tokens along the support direction and
measure what changes downstream (pre-registered in LOG.md, fourteenth pass).

The direction is the difference of the mean token of supported and of single-view patches, per hooked layer,
fitted on DTU *training* scans with the ground-truth support labels (no benchmark scan is used for the fit).
On the 18 benchmark scans the tokens are then edited before every output head of the backbone reads them:

  none      no edit
  ablate    the component along the direction is set to its mean over the capture (the direction carries no
            information any more; every other component is untouched)
  rand<k>   the same edit along a random direction (control for "any edit of this rank")
  pc<k>     the same edit along the k-th principal direction of the patch tokens, after the support direction is
            projected out (control for "any edit of this energy": these are the highest-variance directions left)
  plus      every patch token is moved by +alpha * (mean_supported - mean_single) along the direction
  minus     the opposite move

Per condition and scan: AUROC of the support head (the nonlinear `oiavisself` head, reading the edited tokens),
AUROC of native confidence, AbsRel on supported and on single-view pixels, mean native confidence on both sets,
and the mean relative change of depth against the unedited forward on both sets.

  python scripts/causal_eval.py --dtu /kaggle/input --bases vggt,pi3x,pi3 --out results/P21-causal"""
import argparse
import os
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
import oia_head as O  # noqa: E402

FIT_SCANS = [7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39]      # DTU training scans (oia_head.DTU_TRAIN_SCANS)
assert not set(FIT_SCANS) & set(H.DEFAULT_CFG["scans"]) and set(FIT_SCANS) <= set(O.DTU_TRAIN_SCANS)
N_VIEWS, N_RAND, N_PC, ALPHA = 5, 3, 3, 1.0


class Editor:
    """Applied to a (..., P, C) patch-token tensor of hooked layer `li`. mode None records nothing and edits nothing."""

    def __init__(self, torch):
        self.torch, self.mode, self.dirs, self.delta, self.rand, self.cap = torch, None, None, None, {}, {}

    def __call__(self, li, t):
        self.cap[li] = t.detach()
        if self.mode in (None, "none"):
            return t
        u = (self.rand[self.mode][li] if self.mode in self.rand else self.dirs[li]).to(t.dtype)
        proj = (t * u).sum(-1, keepdim=True)
        if self.mode == "ablate" or self.mode in self.rand:
            return t - (proj - proj.mean(dim=tuple(range(t.dim() - 1)), keepdim=True)) * u
        sign = 1.0 if self.mode == "plus" else -1.0
        return t + sign * ALPHA * float(self.delta[li]) * u


def vggt_forward(wrap, x, dtype, ed, layers=(4, 11, 17, 23)):
    """As oia_head.vggt_forward_feats, with the patch tokens of the four layers the depth head reads edited first."""
    torch, model = wrap.torch, wrap.model
    N, ph, pw = x.shape[0], x.shape[-2] // O.PATCH, x.shape[-1] // O.PATCH
    with O._autocast(torch, dtype):
        tokens, ps = model.aggregator(x[None])
    tokens = list(tokens)
    for li, l in enumerate(layers):
        t = tokens[l].clone()
        t[:, :, ps:] = ed(li, t[:, :, ps:].float()).to(t.dtype)
        tokens[l] = t
    with torch.amp.autocast("cuda", enabled=False):
        depth, conf = model.depth_head(tokens, images=x[None], patch_start_idx=ps)
    F = [O._grid(tokens[l][0][:, ps:], N, ph, pw) for l in layers]
    return depth[0, ..., 0].float(), conf[0].float(), F


def pi3_forward(wrap, x, dtype, ed):
    """Pi3 / Pi3X: the decoder tokens are edited where the point decoder and the confidence decoder read them."""
    torch, model = wrap.torch, wrap.model
    N, ph, pw = x.shape[0], x.shape[-2] // O.PATCH, x.shape[-1] // O.PATCH
    kw = {"with_prior": False} if wrap.name == "pi3x" else {}
    cap, n = {}, ph * pw

    def pre(m, a):
        h = a[0]
        if "edited" not in cap or cap["src"] is not h:
            e = h.clone()
            e[:, -n:] = ed(0, h[:, -n:].float()).to(h.dtype)
            cap["src"], cap["edited"] = h, e
        return (cap["edited"],) + tuple(a[1:])

    hooks = [O._find_module(model, nm).register_forward_pre_hook(pre) for nm in ("point_decoder", "conf_decoder")]
    try:
        with O._autocast(torch, dtype):
            out = model(x[None], **kw)
    finally:
        for h in hooks:
            h.remove()
    lp = out["local_points"][0].float()
    return lp[..., 2], torch.sigmoid(out["conf"][0, ..., 0]).float(), [O._grid(cap["edited"][:, -n:], N, ph, pw)]


FORWARD = {"vggt": vggt_forward, "pi3x": pi3_forward, "pi3": pi3_forward}


def patch_labels(lab, ph, pw):
    """Support labels at the patch grid: a patch counts when more than half of it carries the label."""
    f = lambda m: np.stack([cv2.resize(v.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) for v in m]) > 0.5
    return f(lab["U"]), f(lab["S"])


def run_base(base, root, out, scans, cfg, eval_root=None, view_mode="adjacent"):
    import pandas as pd
    torch = __import__("torch")
    head = H.make_model(f"{base}_oiavisself")           # OIAWrap: .base (backbone wrapper), .head, .pick
    wrap, fwd, ed = head.base, FORWARD[base], Editor(torch)
    model_hw = tuple(cfg["model_hw"])
    ph, pw = model_hw[0] // O.PATCH, model_hw[1] // O.PATCH

    def forward(imgs, mode, mhw=None):
        ed.mode, ed.cap = mode, {}
        x = wrap._prep(imgs, mhw or model_hw)
        with torch.no_grad():
            d, c, F = fwd(wrap, x, wrap.dtype, ed)
            if not torch.isfinite(d).all():
                d, c, F = fwd(wrap, x, None, ed)
            F = [f.float() for f in F]
            if head.pick is not None:
                F = [F[head.pick]]
            return d, c, F

    def load(scan):
        data = H.load_scan(root, scan, H.vis_views(N_VIEWS, 0, mode=cfg.get("view_mode", "spread")), cfg["light"])
        return data, H.vis_labels(data, cfg)

    def load_eval(scan):
        # a second dataset in the repository's CO3D format (ETH3D): the direction stays the one fitted on DTU
        if eval_root is None:
            return load(scan) + (model_hw,)
        import co3d
        views = H.vis_views(N_VIEWS, 0, total=co3d.n_frames(eval_root, scan), mode=view_mode)
        data = co3d.load_co3d_scene(eval_root, scan, views)
        return data, H.vis_labels(data, cfg), tuple(data["model_hw"])

    # ---- fit the direction on training scans
    t0, sums = time.time(), None
    for scan in FIT_SCANS:
        try:
            data, lab = load(scan)
        except Exception as e:
            print(f"[fit] scan {scan} skipped: {e!r}", flush=True)
            continue
        forward(data["imgs"], None)
        Up, Sp = patch_labels(lab, ph, pw)
        Um, Sm = torch.from_numpy(Up.reshape(-1)).to(wrap.device), torch.from_numpy(Sp.reshape(-1)).to(wrap.device)
        if sums is None:
            sums = {li: [0, 0, 0, 0, 0, 0, 0] for li in ed.cap}
        for li, t in ed.cap.items():
            t = t.reshape(-1, t.shape[-1]).double()
            s = sums[li]
            s[0], s[1], s[2], s[3] = s[0] + t[Sm].sum(0), s[1] + int(Sm.sum()), s[2] + t[Um].sum(0), s[3] + int(Um.sum())
            s[4], s[5], s[6] = s[4] + t.sum(0), s[5] + t.T @ t, s[6] + t.shape[0]
    ed.dirs, ed.delta = {}, {}
    g = torch.Generator(device="cpu").manual_seed(0)
    var_rows = []
    for k in range(N_PC):
        ed.rand[f"pc{k}"] = {}
    for li, (ss, ns, su, nu, st, stt, nt) in sums.items():
        diff = (ss / max(ns, 1) - su / max(nu, 1)).float()
        ed.delta[li] = diff.norm().item()
        ed.dirs[li] = diff / diff.norm()
        print(f"[fit] {base} layer {li}: |mean_S - mean_U| = {ed.delta[li]:.3f}, supported {ns}, single-view {nu}", flush=True)
        mu = st / nt
        cov = stt / nt - torch.outer(mu, mu)                      # covariance of all patch tokens of the fit scans
        u = ed.dirs[li].double()
        Pm = torch.eye(len(u), dtype=torch.float64, device=u.device) - torch.outer(u, u)
        w, V = torch.linalg.eigh(Pm @ cov @ Pm)                   # ascending
        v_sup = float(u @ cov @ u)
        row = dict(base=base, layer=li, var_support=v_sup, var_total=float(torch.trace(cov)))
        for k in range(N_PC):
            ed.rand[f"pc{k}"][li] = V[:, -1 - k].float()
            row[f"var_pc{k}"] = float(w[-1 - k])
        var_rows.append(row)
        print(f"[fit] {base} layer {li}: token variance along support direction {v_sup:.3f}, along pc0..{N_PC - 1} "
              + " ".join(f"{row[f'var_pc{k}']:.3f}" for k in range(N_PC)) + f", total {row['var_total']:.1f}", flush=True)
    for k in range(N_RAND):
        ed.rand[f"rand{k}"] = {}
        for li, u in ed.dirs.items():
            r = torch.randn(u.shape, generator=g).to(u.device)
            ed.rand[f"rand{k}"][li] = r / r.norm()
    pd.DataFrame(var_rows).to_csv(os.path.join(out, f"P21-causal-{base}-variance.csv"), index=False)
    print(f"[fit] {base}: {time.time() - t0:.0f}s", flush=True)

    # ---- evaluate on the benchmark scans
    rows, modes = [], ["none", "ablate"] + [f"rand{k}" for k in range(N_RAND)] + [f"pc{k}" for k in range(N_PC)] + ["plus", "minus"]
    for scan in scans:
        data, lab, mhw = load_eval(scan)
        U, S, gt = lab["U"], lab["S"], data["depths"]
        if U.sum() < 50 or S.sum() < 50:
            continue
        hw, ref = gt.shape[1:], None
        for mode in modes:
            d, c, F = forward(data["imgs"], mode, mhw)
            with torch.no_grad():
                score = 1.0 - torch.sigmoid(O.upsample_logit(torch, head.head(F), hw).float())
            d = H._resize_t(d[:, None].float(), hw, "area")[:, 0].cpu().numpy()
            c = H._resize_t(c[:, None].float(), hw, "area")[:, 0].cpu().numpy()
            score = score.cpu().numpy()
            m = H.compute_metrics(d, c, gt, U, S, cfg)
            ds = d * m["scale"]
            if mode == "none":
                ref = ds
            rel = np.abs(ds - ref) / np.maximum(ref, 1e-9)
            rows.append(dict(base=base, scan=scan, mode=mode, auroc_head=H.auroc(score[S], score[U]),
                             auroc_native=m["auroc"], absrel_S=m["absrel_S"], absrel_U=m["absrel_U"],
                             conf_S=float(c[S].mean()), conf_U=float(c[U].mean()),
                             head_S=float(score[S].mean()), head_U=float(score[U].mean()),
                             dchange_S=float(rel[S].mean()), dchange_U=float(rel[U].mean())))
        r = {x["mode"]: x for x in rows[-len(modes):]}
        print(f"[{base}] scan {scan}: head AUROC none {r['none']['auroc_head']:.3f} ablate {r['ablate']['auroc_head']:.3f} "
              f"rand0 {r['rand0']['auroc_head']:.3f} pc0 {r['pc0']['auroc_head']:.3f} | AbsRel_S none {r['none']['absrel_S']:.4f} ablate {r['ablate']['absrel_S']:.4f}",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, f"P21-causal-{base}.csv"), index=False)
    head.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--bases", default="vggt,pi3x,pi3")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--eval-co3d", default=None, help="evaluate on scenes of this CO3D-format root (ETH3D) instead of DTU")
    ap.add_argument("--view-mode", default="adjacent")
    ap.add_argument("--out", default=os.path.join("results", "P21-causal"))
    a = ap.parse_args()
    root = H.find_dtu_root((a.dtu,))
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(H.DEFAULT_CFG)
    if a.eval_co3d:
        scans = a.scans.split(",")
    else:
        scans = [int(s) for s in a.scans.split(",")] if a.scans else cfg["scans"]
    for b in a.bases.split(","):
        try:
            run_base(b, root, a.out, scans, cfg, a.eval_co3d, a.view_mode)
        except Exception:
            print(f"[causal] {b} failed:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
