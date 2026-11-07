"""Fill sensitivity of the invented surface (P9-refill): does the geometry a model returns inside the hidden region
depend on the pixels it was filled with?

For every benchmark cell (seed 0, radius 0.18) the same hidden sphere is filled three ways (gray, blur, noise); the
backbone runs on each; after per-view median-scale alignment on the pixels outside the hidden region, every pixel's
confidence is minus the relative spread of its depth over the three fills (max minus min over the depth). A surface
other views support should not move when one region's pixels change; a surface drawn from the prior may. Scored with
the benchmark's U / S labels of the cell (compute_metrics), so the rows join the Table-4 keys. Also reports the same
spread restricted to pairs of fills, and the AbsRel of the invented depth under each fill.

  python scripts/refill_eval.py --dtu <root> [--co3d <root>] --models vggt,pi3x,pi3,da3
Writes results/P9-refill/P9-refill.csv (DTU) and P9-refill-co3d.csv."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402
import co3d  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILLS = ["gray", "blur", "noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", default=None)
    ap.add_argument("--co3d", default=None)
    ap.add_argument("--models", default="vggt,pi3x,pi3,da3")
    ap.add_argument("--radius", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P9-refill"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    if a.dtu:
        root = H.find_dtu_root((a.dtu,))
        jobs.append(("dtu", root, [int(s) for s in H.DEFAULT_CFG["scans"]], lambda sc: H.load_scan(root, sc, H.select_views(10), 3), dict(H.DEFAULT_CFG)))
    if a.co3d:
        croot = co3d.find_co3d_root((a.co3d,))
        jobs.append(("co3d", croot, co3d.list_co3d_scenes(croot), lambda sc: co3d.load_co3d_scene(croot, sc, H.select_views(10, total=co3d.n_frames(croot, sc))), {**H.DEFAULT_CFG, "dataset": "co3d"}))
    for tag, root, scenes, load, cfg in jobs:
        csv = os.path.join(a.out, f"P9-refill{'' if tag == 'dtu' else '-' + tag}.csv")
        done = set()
        if os.path.exists(csv):
            prev = pd.read_csv(csv); done = {(r.model, str(r.scan)) for r in prev.itertuples()}
        for mname in a.models.split(","):
            todo = [sc for sc in scenes if (mname, str(sc)) not in done]
            if not todo:
                print(f"[{mname}] {tag}: all done"); continue
            model = H.make_model(mname); t0 = time.time()
            for sc in todo:
                data = load(sc)
                try:
                    masks = H.build_masks(data, a.seed, a.radius, cfg)
                except RuntimeError as e:
                    print(f"[{mname}] {sc}: {e}"); continue
                out_hw = data["depths"].shape[1:]; model_hw = tuple(data.get("model_hw", cfg["model_hw"]))
                U, S = masks["U"], masks["S"]
                deps, confs = {}, {}
                for f in FILLS:
                    imgs_in = H.apply_fill(data["imgs"], masks["hide"], f, a.seed)
                    d, c = model.predict(imgs_in, model_hw, out_hw)
                    deps[f], confs[f] = d, c
                # align every fill's depth to the blur run on pixels outside the hidden region (no label used: hide is the input mask)
                outside = ~U
                ref = deps["blur"]
                al = {}
                for f in FILLS:
                    s = np.array([np.median(ref[i][outside[i] & (deps[f][i] > 0)] / np.maximum(deps[f][i][outside[i] & (deps[f][i] > 0)], 1e-9)) for i in range(len(ref))])
                    al[f] = deps[f] * s[:, None, None]
                stack = np.stack([al[f] for f in FILLS])
                spread = (stack.max(0) - stack.min(0)) / np.maximum(np.abs(ref), 1e-6)
                row = dict(model=mname, scan=sc, seed=a.seed, radius_frac=a.radius, fill="refill3", n_views=10)
                m = H.compute_metrics(ref, -spread, data["depths"], U, S, cfg)
                row.update(m)
                for f1, f2 in (("blur", "noise"), ("gray", "blur"), ("gray", "noise")):
                    sp = np.abs(al[f1] - al[f2]) / np.maximum(np.abs(ref), 1e-6)
                    row[f"auroc_{f1}_{f2}"] = H.auroc(-sp[S], -sp[U])
                    tau = np.quantile(-sp[S], 1 - cfg["tau_keep"]); row[f"cur_{f1}_{f2}"] = float(np.mean(-sp[U] >= tau))
                row["spread_U"], row["spread_S"] = float(np.median(spread[U])), float(np.median(spread[S]))
                row["auroc_native_blur"] = H.auroc(confs["blur"][S], confs["blur"][U])
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
                print(f"[{mname}] {tag} {sc}: spread AUROC {row['auroc']:.3f} CUR {row['cur']:.3f}  median spread U {row['spread_U']:.4f} S {row['spread_S']:.4f}  native {row['auroc_native_blur']:.3f}  {time.time() - t0:.0f}s", flush=True)
            if hasattr(model, "free"):
                model.free()
        df = pd.read_csv(csv)
        print(df.groupby("model")[["auroc", "cur", "auroc_blur_noise", "auroc_gray_blur", "auroc_gray_noise", "spread_U", "spread_S", "auroc_native_blur"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
