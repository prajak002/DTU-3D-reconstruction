"""Foreign-view test (pre-registered in LOG.md, fourteenth pass): does a per-pixel support score, averaged over a
view, single out a photograph that does not belong to the scene?

The setting is the single-outlier level (L1) of SysCON3D (Paul et al., 2026), on DTU: K - 1 views of one scan
plus one view of another scan, placed at a seeded position. Every detector gives one number per view, the mean
of its per-pixel score over the image; the foreign view should get the lowest. Per capture: `sep`, the share of
genuine views that score above the foreign one (1 = perfectly singled out, 0.5 = chance), and `top1`, whether
the foreign view has the lowest score (chance 1 / K). A second setting replaces the foreign view with Gaussian
noise, N(0.5, 0.2^2) per pixel, clipped (their SysCON3D-N).

  python scripts/foreign_eval.py --dtu /kaggle/input --bases vggt,pi3x --out results/P21-foreign"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

OUT_HW = (128, 160)


def capture(root, scans, i, K, seed, kind, cfg):
    scan = scans[i]
    views = H.vis_views(K, seed, mode=cfg.get("view_mode", "spread"))
    imgs = H.load_scan(root, scan, views, cfg["light"])["imgs"].copy()
    rng = np.random.default_rng(H._stable_seed("foreign", scan, K, seed))
    pos = int(rng.integers(K))
    if kind == "scan":
        other = scans[(i + 1 + int(rng.integers(len(scans) - 1))) % len(scans)]
        imgs[pos] = H.load_scan(root, other, [views[pos]], cfg["light"])["imgs"][0]
    else:
        other = -1
        imgs[pos] = np.clip(rng.normal(0.5, 0.2, imgs[pos].shape), 0, 1).astype(imgs.dtype)
    return imgs, pos, other


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--bases", default="vggt,pi3x,pi3,da3")
    ap.add_argument("--detectors", default=",oiavisself,reproj")
    ap.add_argument("--views", default="5,10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=os.path.join("results", "P21-foreign"))
    a = ap.parse_args()
    root = H.find_dtu_root((a.dtu,))
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(H.DEFAULT_CFG)
    scans, model_hw = cfg["scans"], tuple(cfg["model_hw"])
    for b in a.bases.split(","):
        rows, csv = [], os.path.join(a.out, f"P21-foreign-{b}.csv")
        for mname in [f"{b}_{s}" if s else b for s in a.detectors.split(",")]:
            t0 = time.time()
            try:
                model = H.make_model(mname)
                for kind in ("scan", "noise"):
                    for K in [int(v) for v in a.views.split(",")]:
                        for seed in [int(s) for s in a.seeds.split(",")]:
                            for i, scan in enumerate(scans):
                                imgs, pos, other = capture(root, scans, i, K, seed, kind, cfg)
                                _, c = model.predict(imgs, model_hw, OUT_HW)
                                v = np.nan_to_num(c.reshape(K, -1), nan=0.0, posinf=0.0, neginf=0.0).mean(1)
                                rest = np.delete(v, pos)
                                rows.append(dict(model=mname, base=b, kind=kind, n_views=K, seed=seed, scan=scan,
                                                 other=other, pos=pos, sep=float((rest > v[pos]).mean() + 0.5 * (rest == v[pos]).mean()),
                                                 top1=bool(v[pos] < rest.min()), score_foreign=float(v[pos]),
                                                 score_rest=float(rest.mean())))
                    d = pd.DataFrame([r for r in rows if r["model"] == mname and r["kind"] == kind])
                    print(f"[foreign] {mname} {kind}: sep {d.sep.mean():.3f} top1 {d.top1.mean():.3f}", flush=True)
                    pd.DataFrame(rows).to_csv(csv, index=False)
                model.free() if hasattr(model, "free") else None
                print(f"[foreign] {mname}: {time.time() - t0:.0f}s", flush=True)
            except Exception:
                print(f"[foreign] {mname} failed:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
