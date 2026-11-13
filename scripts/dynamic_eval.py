"""Moving scenes, full video (run family P23-dynamic, sixteenth pass).

Every protocol so far assumes a rigid scene: a surface that one view shows is the same surface in every other view.
In a video of a scene with moving people that fails for the moving surface, and the single-view label of the vis
protocol silently mixes two different things: a static surface no other frame shows, and a surface every frame
shows but that is somewhere else in the other frames. This script separates them, on real video with sensor depth
and motion-capture camera poses (TUM RGB-D, the "dynamic objects" sequences), with no patch and no fill.

A sequence is preprocessed in temporal order (scripts/tum.py --n-frames F, no co-visibility reordering) and cut
into consecutive windows of N frames that together cover the whole video. In a window, the sensor point of every
pixel of frame i is projected into every other frame j of the window (known camera poses). Frame j then says one of:
  seen      j's own sensor depth is this point (within tol)
  free      j looks through the point: j's sensor depth is farther than the point by more than tol_free everywhere in
            the 3 x 3 neighbourhood of the landing pixel. In a rigid scene this cannot happen; the surface moved.
  occluded  j's sensor depth is nearer; unknown: j has no depth there; or the point is out of j's frame.
Classes of the pixels of frame i (sensor depth known):
  S  static, supported   seen by at least min_seen other frames, free in none
  U  static, single-view seen by no other frame, unknown in none, free in none (the vis protocol's U without motion)
  M  moving              free in at least min_free other frames (eroded by one pixel)
S and U stay clear of a dilation of every pixel with a free-space violation. On a static sequence every M pixel is
label noise (pose, depth and timing error), so the share of M on static sequences of the same camera is the label's
false-positive floor; it is measured by the same script and reported next to every result.

Per (detector, sequence, window): the shares of S, U and M, the AUROC of the detector's score for S against U
(static pixels only), S against M and U against M, the mean score per class, and AbsRel of the predicted depth
(median-scaled on S) per class.

  python scripts/tum.py <raw> <root> --scenes <...> --n-frames 200
  python scripts/dynamic_eval.py --root <root> --models vggt,vggt_reproj,vggt_oiavisself --window 10
  python scripts/dynamic_eval.py --selftest          (no GPU, no data: the labels on a synthetic moving scene)
Writes results/P23-dynamic/P23-dynamic.csv (one row per detector, sequence and window) and a summary."""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TUM RGB-D, category "dynamic objects" (all freiburg3, motion-capture poses): two people at a desk, sitting (little
# motion) or walking (large motion), four camera motions each
DYNAMIC = ["freiburg3_sitting_static", "freiburg3_sitting_xyz", "freiburg3_sitting_halfsphere", "freiburg3_sitting_rpy",
           "freiburg3_walking_static", "freiburg3_walking_xyz", "freiburg3_walking_halfsphere", "freiburg3_walking_rpy"]
# static sequences of the same camera: the false-positive floor of the M label and the static reference of every metric
STATIC = ["freiburg3_long_office_household", "freiburg3_teddy", "freiburg3_cabinet", "freiburg3_large_cabinet",
          "freiburg3_structure_texture_far"]
CFG = dict(tol=0.03, tol_free=0.10, min_seen=2, min_free=2, dilate_px=3, min_pixels=50)


def motion_matrix(depths, K, E, tol=0.03, tol_free=0.10):
    """(N,N,h,w) bool arrays indexed [i, j] for the sensor point of pixel p of frame i tested in frame j: seen, unknown
    (as hallu3d.support_matrix) and free (frame j's depth is known and farther than the point by more than tol_free
    over the whole 3 x 3 neighbourhood of the landing pixel: j looks through the point)."""
    from scipy.ndimage import minimum_filter
    depths = np.asarray(depths, np.float64)
    N, h, w = depths.shape
    seen = np.zeros((N, N, h, w), bool)
    unknown = np.zeros((N, N, h, w), bool)
    free = np.zeros((N, N, h, w), bool)
    dmin = np.stack([minimum_filter(d, size=3, mode="nearest") for d in depths])   # 0 wherever a neighbour is unknown
    for i in range(N):
        valid_i = depths[i] > 0
        Pw = H.backproject(depths[i], K[i], E[i])
        for j in range(N):
            if i == j:
                continue
            u, v, z = H.project(Pw, K[j], E[j])
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inside = valid_i & (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            dj = np.zeros((h, w)); dn = np.zeros((h, w))
            dj[inside] = depths[j][vi[inside], ui[inside]]
            dn[inside] = dmin[j][vi[inside], ui[inside]]
            known = inside & (dj > 0)
            unknown[i, j] = inside & ~known
            seen[i, j] = known & (np.abs(dj - z) / np.maximum(dj, 1e-9) < tol)
            free[i, j] = known & (dn > z * (1.0 + tol_free))
    return seen, unknown, free


def motion_labels(depths, K, E, cfg=None):
    """S, U, M (N,h,w) bool and the per-pixel counts behind them; see the module docstring."""
    cfg = {**CFG, **(cfg or {})}
    seen_m, unk_m, free_m = motion_matrix(depths, K, E, cfg["tol"], cfg["tol_free"])
    seen, unknown, free = seen_m.sum(1), unk_m.sum(1), free_m.sum(1)
    valid = np.asarray(depths) > 0
    M = valid & (free >= cfg["min_free"])
    touched = valid & (free >= 1)
    U_all = valid & (seen == 0) & (unknown == 0)           # the vis protocol's U: static single-view and moved, mixed
    U = U_all & (free == 0)
    S = valid & (seen >= cfg["min_seen"]) & (free == 0)
    k = max(1, int(cfg["dilate_px"])) * 2 + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    k3 = np.ones((3, 3), np.uint8)
    for i in range(len(M)):
        M[i] = cv2.erode(M[i].astype(np.uint8), k3) > 0
        near_motion = cv2.dilate(touched[i].astype(np.uint8), ker) > 0
        U[i] &= ~near_motion
        S[i] &= ~near_motion & ~(cv2.dilate(U_all[i].astype(np.uint8), ker) > 0)
    return dict(S=S, U=U, M=M, U_all=U_all, seen=seen.astype(np.int16), unknown=unknown.astype(np.int16),
                free=free.astype(np.int16))


def window_views(n_frames, window):
    """Consecutive, non-overlapping windows that cover the video; a short tail is dropped."""
    return [list(range(s, s + window)) for s in range(0, n_frames - window + 1, window)]


def score_row(d, c, gt, lab, cfg=None):
    cfg = {**CFG, **(cfg or {})}
    S, U, M = lab["S"], lab["U"], lab["M"]
    valid = max(int((gt > 0).sum()), 1)
    mp = cfg["min_pixels"]
    nS, nU, nM = int(S.sum()), int(U.sum()), int(M.sum())
    row = dict(n_S=nS, n_U=nU, n_M=nM, frac_S=nS / valid, frac_U=nU / valid, frac_M=nM / valid,
               frac_U_mixed=float(lab["U_all"].sum()) / valid)
    nan = float("nan")
    row.update(score_S=float(c[S].mean()) if nS else nan, score_U=float(c[U].mean()) if nU else nan,
               score_M=float(c[M].mean()) if nM else nan,
               auroc_SU=H.auroc(c[S], c[U]) if min(nS, nU) >= mp else nan,
               auroc_SM=H.auroc(c[S], c[M]) if min(nS, nM) >= mp else nan,
               auroc_UM=H.auroc(c[U], c[M]) if min(nU, nM) >= mp else nan,
               auroc_S_Umixed=H.auroc(c[S], c[lab["U_all"]]) if min(nS, int(lab["U_all"].sum())) >= mp else nan)
    if nS >= mp:
        pdep = d * H.median_scale(d, gt, S)
        for name, m in (("S", S), ("U", U), ("M", M)):
            row[f"absrel_{name}"] = H.depth_metrics(pdep, gt, m)[0] if m.sum() >= mp else nan
    return row


# ----------------------------------------------------------------------------- self-test (no GPU, no data)
def _render(cams_x, rects, hw=(120, 160), f=140.0):
    """Depth maps of fronto-parallel rectangles (x0, x1, y0, y1, z) seen by cameras at (cx, 0, 0) looking along +z.
    rects is one list per camera, so a rectangle can sit elsewhere in each frame."""
    h, w = hw
    K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    dx, dy = (uu - K[0, 2]) / f, (vv - K[1, 2]) / f
    D, ids, Es = [], [], []
    for cx, rs in zip(cams_x, rects):
        dep = np.zeros((h, w)); idm = -np.ones((h, w), int)
        for r, (x0, x1, y0, y1, z) in enumerate(rs):
            X, Y = cx + dx * z, dy * z
            hit = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & ((dep == 0) | (z < dep))
            dep[hit] = z; idm[hit] = r
        E = np.eye(4); E[0, 3] = -cx
        D.append(dep); ids.append(idm); Es.append(E)
    return np.stack(D).astype(np.float32), np.stack([K] * len(cams_x)), np.stack(Es), np.stack(ids)


def selftest():
    n = 8
    cams = np.linspace(-0.35, 0.35, n)
    wall, box = (-6, 6, -6, 6, 3.0), (-0.25, 0.25, -0.3, 0.3, 2.0)       # the box hides a strip of the wall per camera
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}: {detail}")

    # 1. rigid scene: no pixel may be called moving
    D, K, E, ids = _render(cams, [[wall, box]] * n)
    lab = motion_labels(D, K, E)
    check("rigid scene has no moving pixel", lab["M"].sum() == 0, f"M pixels {int(lab['M'].sum())}")
    check("rigid scene has supported pixels", lab["S"].mean() > 0.3, f"S share {lab['S'].mean():.3f}")
    # 2. a second box walks across the scene, 0.25 per frame: every frame sees it, at a different place
    mov = [(-0.9 + 0.25 * t, -0.6 + 0.25 * t, -0.25, 0.25, 1.6) for t in range(n)]
    D, K, E, ids = _render(cams, [[wall, box, m] for m in mov])
    lab = motion_labels(D, K, E)
    truth = ids == 2
    prec = (lab["M"] & truth).sum() / max(lab["M"].sum(), 1)
    rec = (lab["M"] & truth).sum() / max(truth.sum(), 1)
    check("moving box: precision of M", prec > 0.99, f"{prec:.3f}")
    check("moving box: recall of M", rec > 0.6, f"{rec:.3f} (the rest is the eroded rim)")
    check("moving box is not in S or U", not ((lab["S"] | lab["U"]) & truth).any(),
          f"S {int((lab['S'] & truth).sum())} U {int((lab['U'] & truth).sum())}")
    check("the vis protocol's U would contain the moving box", (lab["U_all"] & truth).sum() > 0.5 * truth.sum(),
          f"{(lab['U_all'] & truth).sum() / max(truth.sum(), 1):.3f} of the box")
    check("static pixels remain", lab["S"].mean() > 0.2, f"S share {lab['S'].mean():.3f}")
    # 3. a box that stands still in every frame is static, whatever its depth
    D, K, E, ids = _render(cams, [[wall, box, mov[3]]] * n)
    lab = motion_labels(D, K, E)
    check("standing box has no moving pixel", lab["M"].sum() == 0, f"M pixels {int(lab['M'].sum())}")
    # 4. depth noise of 1% and holes do not create motion
    rng = np.random.default_rng(0)
    D, K, E, ids = _render(cams, [[wall, box]] * n)
    Dn = D * (1 + 0.01 * rng.standard_normal(D.shape)).astype(np.float32)
    Dn[rng.random(D.shape) < 0.05] = 0
    lab = motion_labels(Dn, K, E)
    check("noise and holes: share of M", lab["M"].mean() < 1e-3, f"{lab['M'].mean():.5f}")
    # 5. scoring: a score that is low exactly on the moving box separates S from M
    D, K, E, ids = _render(cams, [[wall, box, m] for m in mov])
    lab = motion_labels(D, K, E)
    r = score_row(D.astype(np.float64), np.where(ids == 2, 0.1, 0.9), D, lab)
    check("score row", r["auroc_SM"] == 1.0 and abs(r["absrel_S"]) < 1e-6, f"auroc_SM {r['auroc_SM']:.3f}")
    print("self-test", "passed" if ok else "FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None, help="preprocessed sequences in temporal order (scripts/tum.py --n-frames)")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--models", default="vggt,vggt_reproj,vggt_oiavisself")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P23-dynamic"))
    ap.add_argument("--labels-only", action="store_true", help="shares of S, U and M per window, no model")
    ap.add_argument("--dump-scenes", default="", help="sequences whose per-pixel scores and labels are saved (video)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    import pandas as pd
    import co3d
    root = co3d.find_co3d_root((a.root,))
    scenes = a.scenes.split(",") if a.scenes else [s for s in DYNAMIC + STATIC if os.path.isdir(os.path.join(root, s))]
    os.makedirs(a.out, exist_ok=True)
    csv = os.path.join(a.out, "P23-labels.csv" if a.labels_only else "P23-dynamic.csv")
    done = set()
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        done = {(r.model, str(r.scan), int(r.window)) for r in prev.itertuples()}
    cache = os.path.join(a.out, "labels_cache")
    os.makedirs(cache, exist_ok=True)
    dump = set(filter(None, a.dump_scenes.split(",")))

    def labels(sc, w, data):
        p = os.path.join(cache, f"{sc}__w{w}__n{a.window}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return {k: z[k] for k in z.files}
        lab = motion_labels(data["depths"], data["K_dep"], data["E"])
        np.savez_compressed(p, **lab)
        return lab

    for mname in (["labels"] if a.labels_only else a.models.split(",")):
        todo = [(sc, w, v) for sc in scenes for w, v in enumerate(window_views(co3d.n_frames(root, sc), a.window))
                if (mname, sc, w) not in done]
        if not todo:
            print(f"[{mname}] all done"); continue
        model = None if a.labels_only else H.make_model(mname)
        t0 = time.time()
        for sc, w, views in todo:
            data = co3d.load_co3d_scene(root, sc, views)
            lab = labels(sc, w, data)
            gt = data["depths"]
            if a.labels_only:
                d, c = gt.astype(np.float64), np.zeros(gt.shape)
            else:
                if hasattr(model, "set_cams"):
                    model.set_cams(data["K_dep"], data["E"])
                d, c = model.predict(data["imgs"], tuple(data["model_hw"]), gt.shape[1:])
            row = dict(model=mname, scan=sc, window=w, n_views=a.window, first_frame=views[0],
                       kind="dynamic" if sc in DYNAMIC else "static", **score_row(d, c, gt, lab))
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            if sc in dump and not a.labels_only:
                dp = os.path.join(a.out, "dump", mname, f"{sc}__w{w:03d}.npz")
                os.makedirs(os.path.dirname(dp), exist_ok=True)
                np.savez_compressed(dp, conf=c.astype(np.float16), depth=d.astype(np.float16))
                lp = os.path.join(a.out, "dump", "labels", f"{sc}__w{w:03d}.npz")
                if not os.path.exists(lp):
                    os.makedirs(os.path.dirname(lp), exist_ok=True)
                    h, wd = gt.shape[1:]
                    rgb = np.stack([cv2.resize((im * 255).astype(np.uint8), (wd, h), interpolation=cv2.INTER_AREA)
                                    for im in data["imgs"]])
                    np.savez_compressed(lp, S=lab["S"], U=lab["U"], M=lab["M"], gt=gt.astype(np.float32), rgb=rgb)
            print(f"[{mname}] {sc} w{w}: S {row['frac_S']:.3f} U {row['frac_U']:.4f} M {row['frac_M']:.4f}  auroc S/U "
                  f"{row['auroc_SU']:.3f} S/M {row['auroc_SM']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if model is not None and hasattr(model, "free"):
            model.free()
    df = pd.read_csv(csv)
    cols = [c for c in ("frac_S", "frac_U", "frac_M", "auroc_SU", "auroc_SM", "auroc_UM", "absrel_S", "absrel_U",
                        "absrel_M") if c in df]
    summ = df.groupby(["model", "kind", "scan"])[cols].mean().groupby(["model", "kind"]).mean().round(4)
    summ.to_csv(csv.replace(".csv", "_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
