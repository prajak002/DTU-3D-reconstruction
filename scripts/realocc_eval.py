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
  python scripts/realocc_eval.py --co3d-format <eth3d root> --models ...   (P13: second real dataset)
With --co3d-format the reference is every frame of the scene (ETH3D: laser-scan depth of all images of the scene)
and the input views are the scene's adjacent-view selection, as in fuse_eval.py.
Writes results/P12-realocc/P12-realocc.csv (or P12-realocc-<root name>.csv) and a summary."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
import co3d  # noqa: E402
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


def load_gt_all_co3d(root, scene):
    m = np.load(os.path.join(root, scene, "meta.npz"))
    return m["depth"].astype(np.float32), m["K_dep"].astype(np.float64), m["E"].astype(np.float64)


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


def oneside_views(Es, n, seed, pool=12):
    """One-sided capture (P16, Gate 16): the n input views are spread over the `pool` rig views at one end of the
    rig's main axis (first principal axis of the 49 camera centres); seed 0 takes one end, seed 1 the other. Much
    more of the object faces away from every input view than with the spread selection."""
    C = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Es])
    Cc = C - C.mean(0)
    axis = np.linalg.svd(Cc, full_matrices=False)[2][0]
    order = np.argsort(Cc @ axis)
    block = order[:pool] if int(seed) % 2 == 0 else order[::-1][:pool]
    block = np.sort(block)
    pos = np.round(np.linspace(0, pool - 1, n)).astype(int)
    return sorted(int(v) for v in block[pos])


def zvisible_flags(gt_pts, zb_pts, in_views, depths_all, Ks, Es, tol=0.03):
    """Independent visibility for the hole-robust labels (P17, Gate 18): render zb_pts (every ground-truth point of
    the scene) into each input view as a z-buffer (3x3 minimum filter, as in hole_rate) and call a point of gt_pts
    z-visible when it lies within tol of the front surface in at least one input view."""
    from scipy.ndimage import minimum_filter
    zvis = np.zeros(len(gt_pts), bool)
    for i in in_views:
        h, w = depths_all[i].shape
        u, v, z = H.project(zb_pts, Ks[i], Es[i])
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        zb = np.full((h, w), np.inf)
        np.minimum.at(zb, (vi[inside], ui[inside]), z[inside])
        zb = minimum_filter(zb, size=3)
        u, v, z = H.project(gt_pts, Ks[i], Es[i])
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ok = np.zeros(len(gt_pts), bool)
        ok[inside] = z[inside] <= zb[vi[inside], ui[inside]] * (1 + tol)
        zvis |= ok
    return zvis


def hole_rate(gt_pts, src_view, in_views, depths_all, Ks, Es, tol=0.03):
    """Validity check of the observed test (P14, Gate 12). The depth test calls a ground-truth point observed only
    where an input view's own ground-truth depth exists; a hole in that depth (sparse MVS on CO3Dv2) would label a
    surface the camera does see as unobserved. Independent visibility: render every ground-truth point into each
    input view as a z-buffer (3x3 minimum filter) and call a point z-visible when it lies within tol of the front
    surface. Returns the share of z-visible points (not from an input view) that the depth test calls unobserved."""
    from scipy.ndimage import minimum_filter
    cand = ~np.isin(src_view, in_views)
    zvis = np.zeros(len(gt_pts), bool)
    for i in in_views:
        u, v, z = H.project(gt_pts, Ks[i], Es[i])
        h, w = depths_all[i].shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        zb = np.full((h, w), np.inf)
        np.minimum.at(zb, (vi[inside], ui[inside]), z[inside])
        zb = minimum_filter(zb, size=3)
        ok = np.zeros(len(gt_pts), bool)
        ok[inside] = z[inside] <= zb[vi[inside], ui[inside]] * (1 + tol)
        zvis |= ok
    obs = observed_flags(gt_pts, src_view, in_views, depths_all, Ks, Es, tol)
    sel = cand & zvis
    return float((~obs[sel]).mean()) if sel.any() else float("nan"), int(sel.sum())


def holes_main(a, root, scans):
    """--holes: the validity check alone (no model), one row per (scan, seed, views)."""
    tag = "dtu" if a.dtu else os.path.basename(os.path.normpath(a.co3d_format))
    csv = os.path.join(a.out, f"P14-holes-{tag}.csv")
    rng = np.random.default_rng(0)
    rows = []
    for sc in scans:
        D, Ks, Es = load_gt_all(root, sc) if a.dtu else load_gt_all_co3d(root, sc)
        pts, src = [], []
        for v in range(len(D)):
            m = D[v] > 0
            pts.append(H.backproject(D[v], Ks[v], Es[v])[m]); src.append(np.full(int(m.sum()), v))
        pts, src = np.concatenate(pts), np.concatenate(src)
        if len(pts) > a.max_gt:
            k = rng.choice(len(pts), a.max_gt, replace=False); pts, src = pts[k], src[k]
        for sd in a.seeds.split(","):
            for n in a.views.split(","):
                n, sd = int(n), int(sd)
                if not a.dtu and n >= len(D):
                    continue
                views = H.vis_views(n, sd) if a.dtu else H.vis_views(n, sd, total=len(D), mode="adjacent")
                hr, nz = hole_rate(pts, src, views, D, Ks, Es, a.tol)
                rows.append(dict(dataset=tag, scan=sc, seed=sd, n_views=n, hole_rate=hr, n_zvisible=nz))
                print(f"[holes {tag}] {sc} views {n} seed {sd}: {hr:.4f} of {nz}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(csv, index=False)
    print(df.groupby("n_views")["hole_rate"].mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes", action="store_true", help="validity check of the observed test only (no model)")
    ap.add_argument("--dtu", default=None)
    ap.add_argument("--co3d-format", default=None)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--models", default="vggt,vggt_reproj,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself")
    ap.add_argument("--views", default="3,5,10")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P12-realocc"))
    ap.add_argument("--max-gt", type=int, default=600000)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--align", default="cams", choices=["cams", "points"],
                    help="cams: similarity on camera centres (P12/P13 protocol); points: similarity on the input pixels "
                         "with ground-truth depth (Gate 12 amendment for CO3Dv2, whose adjacent orbit frames make the "
                         "camera-centre fit degenerate)")
    ap.add_argument("--view-mode", default="spread", choices=["spread", "oneside"],
                    help="spread: the vis protocol's view sets; oneside: views from one end of the DTU rig (Gate 16)")
    ap.add_argument("--label-mode", default="depth", choices=["depth", "zagree"],
                    help="depth: unobserved = no input view passes the depth test (P12 to P16); zagree (Gate 18): a "
                         "ground-truth point the depth test calls unobserved but a z-buffer of the whole scan calls "
                         "visible in an input view is ambiguous (a hole in that view's reference depth) and is left out "
                         "of both classes; unobserved = depth test and z-buffer agree")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.dtu:
        root = H.find_dtu_root((a.dtu,))
        scans = [int(x) for x in (a.scans.split(",") if a.scans else H.DEFAULT_CFG["scans"])]
        csv = os.path.join(a.out, "P12-realocc.csv")
    else:
        root = co3d.find_co3d_root((a.co3d_format,))
        scans = a.scans.split(",") if a.scans else co3d.list_co3d_scenes(root)
        csv = os.path.join(a.out, f"P12-realocc-{os.path.basename(os.path.normpath(a.co3d_format))}.csv")
    if a.holes:
        return holes_main(a, root, scans)
    done = set()
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        done = {(r.model, str(r.scan), int(r.seed), int(r.n_views)) for r in prev.itertuples()}
    rng = np.random.default_rng(0)
    gt_cache = {}
    lab_cache = {}   # (scan, seed, views) -> labels of the ground-truth points; the same for every detector
    sub_idx = {}     # scan -> subsample of its ground-truth points, drawn once so that every detector of a job is
                     # scored against the same points and the cached labels stay aligned with them
    for mname in a.models.split(","):
        todo = [(sc, int(sd), int(n)) for sc in scans for sd in a.seeds.split(",") for n in a.views.split(",")
                if (mname, str(sc), int(sd), int(n)) not in done]
        if not todo:
            print(f"[{mname}] all done"); continue
        model = H.make_model(mname)
        base = getattr(model, "base", model)
        t0 = time.time()
        for sc, sd, n in todo:
            if sc not in gt_cache:
                D, Ks, Es = load_gt_all(root, sc) if a.dtu else load_gt_all_co3d(root, sc)
                pts, src = [], []
                for v in range(len(D)):
                    m = D[v] > 0
                    pts.append(H.backproject(D[v], Ks[v], Es[v])[m]); src.append(np.full(int(m.sum()), v))
                pts, src = np.concatenate(pts), np.concatenate(src)
                pts_all = pts
                if len(pts) > a.max_gt:
                    if sc not in sub_idx:
                        sub_idx[sc] = rng.choice(len(pts), a.max_gt, replace=False)
                    k = sub_idx[sc]; pts, src = pts[k], src[k]
                gt_cache = {sc: (D, Ks, Es, pts, src, pts_all)}
            D, Ks, Es, gt_pts, src, gt_all = gt_cache[sc]
            if a.dtu:
                views = oneside_views(Es, n, sd) if a.view_mode == "oneside" else H.vis_views(n, sd)
                data = H.load_scan(root, sc, views, 3)
            else:
                if n >= len(D):
                    continue
                views = H.vis_views(n, sd, total=len(D), mode="adjacent")
                data = co3d.load_co3d_scene(root, sc, views)
            out_hw = data["depths"].shape[1:]
            model_hw = tuple(data.get("model_hw", H.DEFAULT_CFG["model_hw"]))
            if hasattr(model, "set_cams"):   # detectors that are given the ground-truth cameras
                model.set_cams(data["K_dep"] if "K_dep" in data else data["K"], data["E"])
            d, c = model.predict(data["imgs"], model_hw, out_hw)
            torch = base.torch
            x = base._prep(data["imgs"], model_hw)
            with torch.no_grad():
                dm, Km, Em = base._forward_cams(x, base.dtype)
            Km = np.asarray(Km.float().cpu().numpy(), np.float64).copy(); Em = np.asarray(Em.float().cpu().numpy(), np.float64)
            Km[:, 0] *= out_hw[1] / model_hw[1]; Km[:, 1] *= out_hw[0] / model_hw[0]
            C_pred = np.stack([-E[:3, :3].T @ E[:3, 3] for E in Em]); C_gt = np.stack([-E[:3, :3].T @ E[:3, 3] for E in data["E"]])
            pred = np.concatenate([H.backproject(d[i], Km[i], Em[i]).reshape(-1, 3) for i in range(len(d))])
            Kg = data["K_dep"] if "K_dep" in data else data["K"]
            gt_pix = np.concatenate([H.backproject(data["depths"][i], Kg[i], data["E"][i]).reshape(-1, 3)
                                     for i in range(len(d))])
            gvalid = (data["depths"] > 0).reshape(-1)
            if a.align == "points":
                k = np.flatnonzero(gvalid)
                k = k if len(k) <= 50000 else np.random.default_rng(sd).choice(k, 50000, replace=False)
                s, R, t = umeyama(pred[k], gt_pix[k])
            else:
                s, R, t = umeyama(C_pred, C_gt)
            pred = (s * (R @ pred.T)).T + t
            score = c.reshape(-1).astype(np.float64)
            valid_in = data["depths"] > 0
            tau = 0.01 * float(np.median(data["depths"][valid_in]))
            if (sc, sd, n) not in lab_cache:
                o_ = observed_flags(gt_pts, src, views, D, Ks, Es, a.tol)
                a_ = np.zeros(len(gt_pts), bool)
                if a.label_mode == "zagree":
                    a_ = ~o_ & zvisible_flags(gt_pts, gt_all, views, D, Ks, Es, a.tol)
                lab_cache[(sc, sd, n)] = (o_, a_, float(gt_pts[::9973].sum()))
            obs, amb, fp = lab_cache[(sc, sd, n)]
            assert len(obs) == len(gt_pts) and fp == float(gt_pts[::9973].sum()), "cached labels do not match the points"
            from scipy.spatial import cKDTree
            dist, idx = cKDTree(gt_pts).query(pred, k=1, workers=-1)
            on = dist < tau
            cls_obs = on & obs[idx]
            cls_unobs = on & ~obs[idx] & ~amb[idx]
            cls_amb = on & amb[idx]
            cls_off = ~on
            onsurf_valid = float(on[gvalid].mean()) if gvalid.any() else float("nan")
            row = dict(model=mname, scan=sc, seed=sd, n_views=n, align=a.align, view_mode=a.view_mode,
                       label_mode=a.label_mode, frac_amb=float(cls_amb.mean()),
                       frac_gt_ambiguous=float(amb.mean()),
                       onsurf_valid=onsurf_valid,
                       n_pred=len(pred), n_gt=len(gt_pts), tau=tau,
                       frac_gt_unobserved=float((~obs).mean()),
                       frac_obs=float(cls_obs.mean()), frac_unobs=float(cls_unobs.mean()), frac_off=float(cls_off.mean()),
                       n_unobs=int(cls_unobs.sum()),
                       score_obs=float(score[cls_obs].mean()) if cls_obs.any() else float("nan"),
                       score_unobs=float(score[cls_unobs].mean()) if cls_unobs.any() else float("nan"),
                       score_off=float(score[cls_off].mean()) if cls_off.any() else float("nan"),
                       auroc_obs_unobs=H.auroc(score[cls_obs], score[cls_unobs]) if cls_unobs.sum() >= 30 else float("nan"),
                       auroc_obs_off=H.auroc(score[cls_obs], score[cls_off]) if cls_off.sum() >= 30 else float("nan"))
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(f"[{mname}] scan {sc} views {n} seed {sd}: obs {row['frac_obs']:.3f} unobs {row['frac_unobs']:.4f} amb {row['frac_amb']:.4f} "
                  f"off {row['frac_off']:.3f}  auroc obs/unobs {row['auroc_obs_unobs']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if hasattr(model, "free"):
            model.free()
    df = pd.read_csv(csv)
    summ = df.groupby(["model", "n_views"])[["frac_obs", "frac_unobs", "frac_off", "frac_gt_unobserved",
                                             "auroc_obs_unobs", "auroc_obs_off"]].mean().round(4)
    summ.to_csv(csv.replace(".csv", "_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
