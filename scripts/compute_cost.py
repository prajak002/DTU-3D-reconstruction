"""Compute accounting for every detector (P3-compute, issue #4): backbone forwards, wall-clock per scene of ten
views, peak GPU memory, head parameters and head FLOPs relative to the backbone.

Usage: python scripts/compute_cost.py --dtu <root> [--models vggt,vggt_oia,vggt_oiapatch,vggt_oiaattn,vggt_reproj,vggt_pv4,vggt_pvf8]
Writes results/P3-compute/P3-compute.csv and prints a markdown table. Timing is the median of --reps forwards after
one warm-up, on DTU scan 3 at the benchmark resolution (420 x 518). FLOPs come from torch.utils.flop_counter on the
head alone and on the backbone alone; detectors with K forwards count K backbone passes.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H
import oia_head as O

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def forwards_of(name):
    m = re.fullmatch(r".+_pvf?(\d+)", name)
    return int(m.group(1)) if m else 1


def head_flops(torch, wrap, F):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            wrap.head([f.float() for f in F])
    return fc.get_total_flops()


def backbone_flops(torch, base, x):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            base._forward(x, base.dtype)
    return fc.get_total_flops()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--models", default="vggt,vggt_oia,vggt_oialin,vggt_oiapatch,vggt_oiaattn,vggt_reproj,vggt_pv4,vggt_pvf8,"
                                        "pi3x,pi3x_oia,pi3x_reproj,pi3x_pv4,pi3x_pvf8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "P3-compute"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints"))
    a = ap.parse_args()
    os.environ.setdefault("OIA_CKPT_DIR", a.ckpt)
    import torch
    root = H.find_dtu_root((a.dtu,))
    data = H.load_scan(root, 3, H.select_views(10), 3)
    out_hw = data["depths"].shape[1:]
    model_hw = H.DEFAULT_CFG["model_hw"]
    rows = []
    for name in a.models.split(","):
        try:
            torch.cuda.reset_peak_memory_stats()
            w = H.make_model(name)
            w.predict(data["imgs"], model_hw, out_hw)                     # warm-up
            ts = []
            for _ in range(a.reps):
                torch.cuda.synchronize(); t0 = time.time()
                w.predict(data["imgs"], model_hw, out_hw)
                torch.cuda.synchronize(); ts.append(time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 1e9
            base = getattr(w, "base", w)
            x = base._prep(data["imgs"], model_hw)
            bflops = backbone_flops(torch, base, x)
            hparams = hflops = 0
            if isinstance(w, O.OIAWrap):
                hparams = sum(p.numel() for p in w.head.parameters())
                _, _, F, _, _ = w.ff(base, x, base.dtype, w.feats)
                hflops = head_flops(torch, w, F)
            k = forwards_of(name)
            rows.append(dict(model=name, forwards=k, sec_per_scene=float(np.median(ts)), peak_gb=peak,
                             backbone_gflops=bflops / 1e9 * k, head_params_M=hparams / 1e6, head_gflops=hflops / 1e9,
                             head_over_backbone_pct=(100.0 * hflops / bflops) if bflops else 0.0))
            print(rows[-1], flush=True)
            w.free(); del w; torch.cuda.empty_cache()
        except Exception as e:
            print(f"!! {name}: {type(e).__name__}: {e}", flush=True)
    import pandas as pd
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "P3-compute.csv"), index=False)
    json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, reps=a.reps, scan=3, views=10,
                   model_hw=list(model_hw)), open(os.path.join(a.out, "P3-compute_cfg.json"), "w"), indent=2)
    lines = ["| detector | forwards | s / scene | peak GB | backbone GFLOPs | head params (M) | head GFLOPs | head / backbone |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['forwards']} | {r['sec_per_scene']:.2f} | {r['peak_gb']:.1f} | {r['backbone_gflops']:.0f} | "
                     f"{r['head_params_M']:.2f} | {r['head_gflops']:.2f} | {r['head_over_backbone_pct']:.2f}% |")
    open(os.path.join(a.out, "P3-compute.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
