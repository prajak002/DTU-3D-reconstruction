"""Downstream: does abstaining improve the fused reconstruction? (P8-fuse)

For every scene the backbone runs once on N clean views. Its per-view depths are back-projected with its OWN
predicted cameras into one point cloud, aligned to the ground-truth frame by a similarity transform fitted on the
camera centres (Umeyama, predicted against ground-truth cameras of the same views), and compared with the
ground-truth cloud (the ground-truth depths of the same views back-projected with the true cameras):

  accuracy      fraction of predicted points within tau of a ground-truth point
  completeness  fraction of ground-truth points within tau of a predicted point
  tau           1% of the scene's median camera-to-surface distance (scale-free; DTU's 2 mm is about 0.3%)

A detector keeps the fraction `coverage` of the predicted points with the highest score (0.6, 0.7, 0.8, 0.9, 1.0),
so detectors are compared at matched coverage: native confidence, reprojection consistency, the per-image control,
the label-free head, and, as references, random dropping and the oracle that drops single-view pixels first.

  python scripts/fuse_eval.py --dtu <root> [--co3d-format <root> --scenes a,b] --models vggt,vggt_oiavisself,...
Writes results/P8-fuse/P8-fuse.csv (one row per scene, view count, seed, detector, coverage) and a summary."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
import co3d  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def umeyama(src, dst):
    """Similarity s, R, t with dst ~ s R src + t (Umeyama 1991), src/dst (n,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.ones(3); d[-1] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ np.diag(d) @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = (sig * d).sum() / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def cloud(depths, K, E, mask):
    pts = []
    for i in range(len(depths)):
        P = H.backproject(depths[i], K[i], E[i])
        pts.append(P.reshape(-1, 3)[mask[i].reshape(-1)])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def nn_dist(a, b):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf)
    return cKDTree(b).query(a, k=1, workers=-1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", default=None)
    ap.add_argument("--co3d-format", default=None, help="ETH3D or phototourism root (scripts/co3d.py layout)")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--models", default="vggt,vggt_reproj,vggt_oiavispatch,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself")
    ap.add_argument("--views", default="5,10")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--coverages", default="0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P8-fuse"))
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--combine", action="store_true", help="for a head model also score native confidence and the "
                    "rank mean of the two (rows <model>_native and <model>_comb; P13, Gate 11)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or ("dtu" if a.dtu else os.path.basename(os.path.normpath(a.co3d_format)))
    csv = os.path.join(a.out, f"P8-fuse-{tag}.csv")
    done = set()
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        done = {(r.model, str(r.scan), int(r.seed), int(r.n_views)) for r in prev.itertuples()}
    if a.dtu:
        root = H.find_dtu_root((a.dtu,)); scenes = [int(s) for s in (a.scenes.split(",") if a.scenes else H.DEFAULT_CFG["scans"])]
        load = lambda sc, n, sd: H.load_scan(root, sc, H.vis_views(n, sd), 3)
        mode = "spread"
    else:
        root = co3d.find_co3d_root((a.co3d_format,)); scenes = a.scenes.split(",") if a.scenes else co3d.list_co3d_scenes(root)
        load = lambda sc, n, sd: co3d.load_co3d_scene(root, sc, H.vis_views(n, sd, total=co3d.n_frames(root, sc), mode="adjacent"))
        mode = "adjacent"
    covs = [float(c) for c in a.coverages.split(",")]
    rng = np.random.default_rng(0)
    for mname in a.models.split(","):
        todo = [(sc, int(sd), int(n)) for sc in scenes for sd in a.seeds.split(",") for n in a.views.split(",") if (mname, str(sc), int(sd), int(n)) not in done]
        if not todo:
            print(f"[{mname}] all done"); continue
        model = H.make_model(mname)
        base = getattr(model, "base", getattr(model, "a", model))
        t0 = time.time()
        for sc, sd, n in todo:
            data = load(sc, n, sd)
            lab = H.vis_labels(data, H.DEFAULT_CFG)
            out_hw = data["depths"].shape[1:]
            model_hw = tuple(data.get("model_hw", H.DEFAULT_CFG["model_hw"]))
            d, c = model.predict(data["imgs"], model_hw, out_hw)
            # the backbone's own cameras for the fusion (same forward family as the reprojection detector)
            torch = base.torch
            x = base._prep(data["imgs"], model_hw)
            with torch.no_grad():
                dm, Km, Em = base._forward_cams(x, base.dtype)
            Km = np.asarray(Km.float().cpu().numpy(), np.float64).copy(); Em = np.asarray(Em.float().cpu().numpy(), np.float64)
            Km[:, 0] *= out_hw[1] / model_hw[1]; Km[:, 1] *= out_hw[0] / model_hw[0]
            # similarity from predicted to ground-truth frame, fitted on camera centres
            C_pred = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Em]); C_gt = np.stack([-E[:3, :3].T @ E[:3, 3] for E in data["E"]])
            s, R, t = umeyama(C_pred, C_gt)
            valid = data["depths"] > 0
            gt_pts = cloud(data["depths"], data["K_dep"], data["E"], valid)
            if len(gt_pts) > a.max_points:
                gt_pts = gt_pts[rng.choice(len(gt_pts), a.max_points, replace=False)]
            scale_ref = float(np.median(data["depths"][valid]))
            tau = 0.01 * scale_ref
            pred_all = cloud(d, Km, Em, np.ones_like(valid))
            pred_all = (s * (R @ pred_all.T)).T + t
            score = c.reshape(-1)
            is_u = lab["U"].reshape(-1); is_s = lab["S"].reshape(-1)
            rows = []
            orders = [("detector", np.argsort(-score, kind="stable")), ("random", rng.permutation(len(score))),
                      ("oracle", np.argsort(is_u.astype(np.int8), kind="stable"))]
            if a.combine and base is not model:
                _, c_nat = base.predict(data["imgs"], model_hw, out_hw)
                nat = np.asarray(c_nat, np.float64).reshape(-1)
                rk = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable") / max(len(v) - 1, 1)
                comb = 0.5 * (rk(nat) + rk(score.astype(np.float64)))
                orders += [("native", np.argsort(-nat, kind="stable")), ("comb", np.argsort(-comb, kind="stable"))]
            for name, order in orders:
                for cov in covs:
                    k = max(1, int(round(cov * len(score))))
                    keep = order[:k]
                    P = pred_all[keep]
                    if len(P) > a.max_points:
                        P = P[rng.choice(len(P), a.max_points, replace=False)]
                    acc = float((nn_dist(P, gt_pts) < tau).mean())
                    comp = float((nn_dist(gt_pts, P) < tau).mean())
                    ku = is_u[keep]
                    rows.append(dict(model=mname if name == "detector" else f"{mname}_{name}", scan=sc, seed=sd, n_views=n, coverage=cov,
                                     accuracy=acc, completeness=comp, f1=2 * acc * comp / max(acc + comp, 1e-9),
                                     single_view_kept=float(ku.mean()) if len(ku) else 0.0, tau=tau, n_pred=len(keep), n_gt=len(gt_pts)))
            pd.DataFrame(rows).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            r = [x for x in rows if x["model"] == mname and x["coverage"] == 0.8][0]
            print(f"[{mname}] {sc} views {n} seed {sd}: acc@0.8 {r['accuracy']:.3f} comp {r['completeness']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if hasattr(model, "free"):
            model.free()
    df = pd.read_csv(csv)
    summ = df.groupby(["model", "coverage"])[["accuracy", "completeness", "f1", "single_view_kept"]].mean().round(4)
    summ.to_csv(os.path.join(a.out, f"P8-fuse-{tag}_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
