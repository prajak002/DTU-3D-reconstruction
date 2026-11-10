"""Next-view selection from the support head (run family P16-nbv, Gate 15).

Question: does knowing which predicted geometry is imagined tell a user where to take the next photograph?
On each DTU benchmark scan the backbone runs on N input views (3 or 5, the vis protocol's view sets, seeds 0 and 1).
Its predicted points (every input pixel, back-projected with its own depth and cameras, placed in the ground-truth
frame by a similarity fitted on the camera centres, as in realocc_eval.py) are scored by two detectors on the same
geometry: native confidence and the label-free support head. Each detector flags the 10% of predicted points it
trusts least. Every other rig view is a candidate next view (its pose is known: a capture plan or a robot arm knows
where it can go; its image is not used). A selection rule picks one candidate:
  head      the candidate that sees the most points the head flags (z-buffer of the predicted points in that view)
  native    the same rule with native confidence
  farthest  the candidate whose viewing direction is farthest from every input view's (a geometric heuristic)
  random    the mean over all candidates (the expected gain of a random choice)
  oracle    the candidate with the largest gain (upper bound)
Gain of a candidate: the share of the ground-truth surface that no input view saw (the 49-view reference of
realocc_eval.py) and that the candidate view does see (projects onto its own ground-truth depth within 3%). The
gain is computed from the ground truth only, so no extra forward pass is needed per candidate.

  python scripts/nbv_eval.py --dtu <root> --pairs vggt:vggt_oiavisself,pi3x:pi3x_oiavisself
Writes results/P16-nbv/P16-nbv.csv: one row per (backbone, scan, seed, N) with the gain of every rule."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
from fuse_eval import umeyama  # noqa: E402
from realocc_eval import load_gt_all, observed_flags  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG_SHARE = 0.10   # each detector flags the 10% of predicted points it trusts least (fixed before the run)


def predict_aligned(model, data):
    """Depth, score and the predicted points in the ground-truth frame (camera-centre similarity)."""
    base = getattr(model, "base", model)
    out_hw = data["depths"].shape[1:]
    model_hw = tuple(data.get("model_hw", H.DEFAULT_CFG["model_hw"]))
    d, c = model.predict(data["imgs"], model_hw, out_hw)
    torch = base.torch
    x = base._prep(data["imgs"], model_hw)
    with torch.no_grad():
        _, Km, Em = base._forward_cams(x, base.dtype)
    Km = np.asarray(Km.float().cpu().numpy(), np.float64).copy(); Em = np.asarray(Em.float().cpu().numpy(), np.float64)
    Km[:, 0] *= out_hw[1] / model_hw[1]; Km[:, 1] *= out_hw[0] / model_hw[0]
    C_pred = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Em]); C_gt = np.stack([-E[:3, :3].T @ E[:3, 3] for E in data["E"]])
    pred = np.concatenate([H.backproject(d[i], Km[i], Em[i]).reshape(-1, 3) for i in range(len(d))])
    s, R, t = umeyama(C_pred, C_gt)
    pred = (s * (R @ pred.T)).T + t
    return pred, c.reshape(-1).astype(np.float64)


def visible_in(pts, K, E, hw, tol=0.03):
    """Which points a camera would see: in frame and on the front surface of their own z-buffer (3x3 min filter)."""
    from scipy.ndimage import minimum_filter
    u, v, z = H.project(pts, K, E)
    h, w = hw
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    zb = np.full((h, w), np.inf)
    np.minimum.at(zb, (vi[inside], ui[inside]), z[inside])
    zb = minimum_filter(zb, size=3)
    ok = np.zeros(len(pts), bool)
    ok[inside] = z[inside] <= zb[vi[inside], ui[inside]] * (1 + tol)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--pairs", default="vggt:vggt_oiavisself,pi3x:pi3x_oiavisself,pi3:pi3_oiavisself,da3:da3_oiavisself")
    ap.add_argument("--views", default="3,5")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P16-nbv"))
    ap.add_argument("--max-gt", type=int, default=600000)
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = H.find_dtu_root((a.dtu,))
    scans = [int(x) for x in (a.scans.split(",") if a.scans else H.DEFAULT_CFG["scans"])]
    csv = os.path.join(a.out, "P16-nbv.csv")
    done = set()
    if os.path.exists(csv):
        done = {(r.backbone, int(r.scan), int(r.seed), int(r.n_views)) for r in pd.read_csv(csv).itertuples()}
    rng = np.random.default_rng(0)
    gt = {}
    for pair in a.pairs.split(","):
        native_name, head_name = pair.split(":")
        todo = [(sc, int(sd), int(n)) for sc in scans for sd in a.seeds.split(",") for n in a.views.split(",")
                if (native_name, sc, int(sd), int(n)) not in done]
        if not todo:
            print(f"[{native_name}] all done"); continue
        head = H.make_model(head_name)
        native = H.make_model(native_name)
        t0 = time.time()
        for sc, sd, n in todo:
            if sc not in gt:
                D, Ks, Es = load_gt_all(root, sc)
                pts, src = [], []
                for v in range(len(D)):
                    m = D[v] > 0
                    pts.append(H.backproject(D[v], Ks[v], Es[v])[m]); src.append(np.full(int(m.sum()), v))
                pts, src = np.concatenate(pts), np.concatenate(src)
                if len(pts) > a.max_gt:
                    k = rng.choice(len(pts), a.max_gt, replace=False); pts, src = pts[k], src[k]
                gt = {sc: (D, Ks, Es, pts, src)}
            D, Ks, Es, gt_pts, src = gt[sc]
            views = H.vis_views(n, sd)
            data = H.load_scan(root, sc, views, 3)
            pred, s_head = predict_aligned(head, data)
            _, s_nat = predict_aligned(native, data)
            obs = observed_flags(gt_pts, src, views, D, Ks, Es, a.tol)
            un_pts, un_src = gt_pts[~obs], src[~obs]
            cands = [j for j in range(len(D)) if j not in views]
            gain = np.array([observed_flags(un_pts, un_src, [j], D, Ks, Es, a.tol).mean() if len(un_pts) else 0.0
                             for j in cands])
            flags = {}
            for name, s in (("head", s_head), ("native", s_nat)):
                thr = np.quantile(s, FLAG_SHARE)
                flags[name] = s <= thr
            vis = [visible_in(pred, Ks[j], Es[j], D[j].shape[:2]) for j in cands]   # occlusion by all predicted points
            hits = {name: np.array([int((vj & f).sum()) for vj in vis]) for name, f in flags.items()}
            C = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Es]); ctr = gt_pts.mean(0)
            dirs = (C - ctr) / np.linalg.norm(C - ctr, axis=1, keepdims=True)
            far = np.array([min(np.arccos(np.clip(dirs[j] @ dirs[i], -1, 1)) for i in views) for j in cands])
            pick = {"head": int(np.argmax(hits["head"])), "native": int(np.argmax(hits["native"])),
                    "farthest": int(np.argmax(far)), "oracle": int(np.argmax(gain))}
            row = dict(backbone=native_name, head=head_name, scan=sc, seed=sd, n_views=n,
                       frac_gt_unobserved=float((~obs).mean()), n_candidates=len(cands),
                       gain_random=float(gain.mean()),
                       **{f"gain_{k}": float(gain[i]) for k, i in pick.items()},
                       **{f"view_{k}": int(cands[i]) for k, i in pick.items()})
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(f"[{native_name}] scan {sc} views {n} seed {sd}: unseen {row['frac_gt_unobserved']:.3f}  gain head "
                  f"{row['gain_head']:.3f} native {row['gain_native']:.3f} far {row['gain_farthest']:.3f} rand "
                  f"{row['gain_random']:.3f} oracle {row['gain_oracle']:.3f}  {time.time() - t0:.0f}s", flush=True)
        for m in (head, native):
            if hasattr(m, "free"):
                m.free()
    df = pd.read_csv(csv)
    cols = ["frac_gt_unobserved", "gain_head", "gain_native", "gain_farthest", "gain_random", "gain_oracle"]
    summ = df.groupby(["backbone", "n_views"])[cols].mean().round(4)
    summ.to_csv(csv.replace(".csv", "_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
