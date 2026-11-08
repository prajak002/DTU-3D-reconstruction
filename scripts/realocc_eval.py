"""Real occlusion against the dense scan (run family P12-realocc, Gate 8).

The benchmark creates unobserved geometry by hiding a region from every camera; this script asks how often a
backbone places geometry on a surface that none of its input views showed, on real captures and without any
patch. On each DTU benchmark scan the backbone runs on N input views (3, 5 or 10, the vis protocol's view sets).
Its predicted points (every input pixel, back-projected with its own depth and cameras, placed in the ground-truth
frame by a similarity fitted on the camera centres, as in fuse_eval.py) are matched to the nearest ground-truth
point of ALL 49 rig views. A ground-truth point is *observed* when at least one input view sees it (projects in
frame onto that view's own ground-truth depth within 3%) and *unobserved* when only the 39 held-out views do.
Classes of predicted points, within tau = 1% of the median camera-to-surface distance of a ground-truth point:
  observed    the point sits on a surface some input view showed
  unobserved  the point sits on a surface no input view showed (verified by the denser scan)
  off         no ground-truth surface within tau (wrong depth, floater, or a part of the scene the scan missed)
Reported per (detector, scan, seed, N): the share of each class, the mean score per class, and the AUROC of the
detector's score for observed against unobserved points and for observed against off points.

  python scripts/realocc_eval.py --dtu <root> --models vggt,vggt_reproj,vggt_oiavisself,pi3x,...
Writes results/P12-realocc/P12-realocc.csv and a summary."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
from fuse_eval import umeyama, nn_dist  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gt_all(root, scan, n_total=49):
    """Depth maps and cameras of every rig view (no images)."""
    depths, Ks, Es = [], [], []
    for v in range(n_total):
        dp = os.path.join(root, "Depths", f"scan{scan}_train", f"depth_map_{v:04d}.pfm")
        cp = os.path.join(root, "Cameras", "train", f"{v:08d}_cam.txt")
        dep = H.read_pfm(dp)
        E, K = H.read_cam(cp)
        Kd, _ = H._scale_K_to(K, dep.shape[:2])
        depths.append(dep); Ks.append(Kd); Es.append(E)
    return np.stack(depths), np.stack(Ks), np.stack(Es)


def observed_flags(gt_pts, src_view, in_views, depths_all, Ks, Es, tol=0.03):
    """For every ground-truth point (from rig view src_view[k]): does at least one input view see it? A point from
    an input view is seen by that view itself; otherwise it must project in frame onto the input view's own known
    depth within tol."""
    obs = np.isin(src_view, in_views)
    for i in in_views:
        u, v, z = H.project(gt_pts, Ks[i], Es[i])
        h, w = depths_all[i].shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        dj = np.zeros(len(gt_pts))
        dj[inside] = depths_all[i][vi[inside], ui[inside]]
        seen = inside & (dj > 0) & (np.abs(dj - z) / np.maximum(dj, 1e-9) < tol)
        obs |= seen
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--models", default="vggt,vggt_reproj,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself")
    ap.add_argument("--views", default="3,5,10")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P12-realocc"))
    ap.add_argument("--max-gt", type=int, default=600000)
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = H.find_dtu_root((a.dtu,))
    scans = [int(x) for x in (a.scans.split(",") if a.scans else H.DEFAULT_CFG["scans"])]
    csv = os.path.join(a.out, "P12-realocc.csv")
    done = set()
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        done = {(r.model, int(r.scan), int(r.seed), int(r.n_views)) for r in prev.itertuples()}
    rng = np.random.default_rng(0)
    gt_cache = {}
    for mname in a.models.split(","):
        todo = [(sc, int(sd), int(n)) for sc in scans for sd in a.seeds.split(",") for n in a.views.split(",")
                if (mname, sc, int(sd), int(n)) not in done]
        if not todo:
            print(f"[{mname}] all done"); continue
        model = H.make_model(mname)
        base = getattr(model, "base", model)
        t0 = time.time()
        for sc, sd, n in todo:
            if sc not in gt_cache:
                D, Ks, Es = load_gt_all(root, sc)
                pts, src = [], []
                for v in range(len(D)):
                    m = D[v] > 0
                    pts.append(H.backproject(D[v], Ks[v], Es[v])[m]); src.append(np.full(int(m.sum()), v))
                pts, src = np.concatenate(pts), np.concatenate(src)
                if len(pts) > a.max_gt:
                    k = rng.choice(len(pts), a.max_gt, replace=False); pts, src = pts[k], src[k]
                gt_cache = {sc: (D, Ks, Es, pts, src)}
            D, Ks, Es, gt_pts, src = gt_cache[sc]
            views = H.vis_views(n, sd)
            data = H.load_scan(root, sc, views, 3)
            out_hw = data["depths"].shape[1:]
            model_hw = tuple(H.DEFAULT_CFG["model_hw"])
            d, c = model.predict(data["imgs"], model_hw, out_hw)
            torch = base.torch
            x = base._prep(data["imgs"], model_hw)
            with torch.no_grad():
                dm, Km, Em = base._forward_cams(x, base.dtype)
            Km = np.asarray(Km.float().cpu().numpy(), np.float64).copy(); Em = np.asarray(Em.float().cpu().numpy(), np.float64)
            Km[:, 0] *= out_hw[1] / model_hw[1]; Km[:, 1] *= out_hw[0] / model_hw[0]
            C_pred = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Em]); C_gt = np.stack([-E[:3, :3].T @ E[:3, 3] for E in data["E"]])
            s, R, t = umeyama(C_pred, C_gt)
            pred = np.concatenate([H.backproject(d[i], Km[i], Em[i]).reshape(-1, 3) for i in range(len(d))])
            pred = (s * (R @ pred.T)).T + t
            score = c.reshape(-1).astype(np.float64)
            valid_in = data["depths"] > 0
            tau = 0.01 * float(np.median(data["depths"][valid_in]))
            obs = observed_flags(gt_pts, src, views, D, Ks, Es, a.tol)
            from scipy.spatial import cKDTree
            dist, idx = cKDTree(gt_pts).query(pred, k=1, workers=-1)
            on = dist < tau
            cls_obs = on & obs[idx]
            cls_unobs = on & ~obs[idx]
            cls_off = ~on
            row = dict(model=mname, scan=sc, seed=sd, n_views=n, n_pred=len(pred), n_gt=len(gt_pts), tau=tau,
                       frac_gt_unobserved=float((~obs).mean()),
                       frac_obs=float(cls_obs.mean()), frac_unobs=float(cls_unobs.mean()), frac_off=float(cls_off.mean()),
                       n_unobs=int(cls_unobs.sum()),
                       score_obs=float(score[cls_obs].mean()) if cls_obs.any() else float("nan"),
                       score_unobs=float(score[cls_unobs].mean()) if cls_unobs.any() else float("nan"),
                       score_off=float(score[cls_off].mean()) if cls_off.any() else float("nan"),
                       auroc_obs_unobs=H.auroc(score[cls_obs], score[cls_unobs]) if cls_unobs.sum() >= 30 else float("nan"),
                       auroc_obs_off=H.auroc(score[cls_obs], score[cls_off]) if cls_off.sum() >= 30 else float("nan"))
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(f"[{mname}] scan {sc} views {n} seed {sd}: obs {row['frac_obs']:.3f} unobs {row['frac_unobs']:.4f} "
                  f"off {row['frac_off']:.3f}  auroc obs/unobs {row['auroc_obs_unobs']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if hasattr(model, "free"):
            model.free()
    df = pd.read_csv(csv)
    summ = df.groupby(["model", "n_views"])[["frac_obs", "frac_unobs", "frac_off", "frac_gt_unobserved",
                                             "auroc_obs_unobs", "auroc_obs_off"]].mean().round(4)
    summ.to_csv(os.path.join(a.out, "P12-realocc_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
