"""Per-pixel dumps for the detector comparison figure (scripts/make_fig_compare.py), and the same rows as a CSV.

Runs the clean-view single-view test (hallu3d.run_vis) on DTU scans for one or more backbones with four detectors
each: native confidence, the aligned perturbation variance (`_pvf8`), reprojection with the model's own cameras
(`_reproj`) and the label-free support head (`_oiavisself`). Writes <out>/P20-dump.csv and the per-pixel maps under
<out>/dtu/{labels,<detector>}/<scan>__s<seed>__v<views>.npz.

  python scripts/dump_compare.py --dtu /kaggle/input --bases vggt,pi3x --out results/P20-dump"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--bases", default="vggt,pi3x,pi3,da3")
    ap.add_argument("--scans", default=None, help="comma-separated DTU scans; default: the 18 benchmark scans")
    ap.add_argument("--views", default="5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--detectors", default=",pvf8,reproj,oiavisself",
                    help="suffixes after the backbone name; the empty one is native confidence")
    ap.add_argument("--out", default=os.path.join("results", "P20-dump"))
    a = ap.parse_args()
    root = H.find_dtu_root((a.dtu,))
    os.makedirs(a.out, exist_ok=True)
    cfg = {"vis_views": [int(v) for v in a.views.split(",")], "seeds": [int(s) for s in a.seeds.split(",")],
           "dump_dir": os.path.join(a.out, "dtu")}
    if a.scans:
        cfg["scans"] = [int(s) for s in a.scans.split(",")]
    csv = os.path.join(a.out, "P20-dump.csv")
    for b in a.bases.split(","):
        for m in [f"{b}_{s}" if s else b for s in a.detectors.split(",")]:
            t0 = time.time()
            try:
                H.run_vis(root, [m], cfg=cfg, out_csv=csv)
                print(f"[dump] {m}: {time.time() - t0:.0f}s", flush=True)
            except Exception:
                print(f"[dump] {m} failed:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
