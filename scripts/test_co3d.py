"""CPU test of the CO3D path: consistency check on every preprocessed scene, then the benchmark
loop with the fake oracle / blind models from test_synthetic.py. Usage: test_co3d.py <prep_root> <out_dir>"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import hallu3d as H
import co3d

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
scenes = co3d.list_co3d_scenes(root)
cfg = {**H.DEFAULT_CFG, "dataset": "co3d", "scans": scenes, "seeds": [0, 1], "radius_fracs": [0.18], "fills": ["gray"]}

print("consistency check (reproj_agree should be high, U_transfer should be high):")
checks = {}
for sc in scenes:
    d = co3d.load_co3d_scene(root, sc, H.select_views(10, total=co3d.n_frames(root, sc)))
    m = H.build_masks(d, 0, 0.18, cfg)
    checks[sc] = H.consistency_check(d, m)
    print(f"  {sc:30s} {checks[sc]}  U={m['U'].mean():.4f} S={m['S'].mean():.4f}")
import json
json.dump(checks, open(os.path.join(out, "consistency_co3d.json"), "w"), indent=1)

rng = np.random.default_rng(0)
_data = {}


class Fake:
    def __init__(self, kind): self.kind = kind

    def predict(self, imgs, model_hw, out_hw):
        d_ = _data["d"]
        gt = d_["depths"]
        r = imgs.shape[1] // out_hw[0]
        hidden = (np.abs(imgs - d_["imgs"]).sum(-1) > 1e-6)
        hid = hidden.reshape(len(imgs), out_hw[0], r, out_hw[1], r).mean((2, 4)) > 0.5
        d = np.where(gt > 0, gt, 3.0) * 2.0
        if self.kind == "oracle":
            c = np.where(hid, 0.1, 5.0) + rng.random(d.shape) * 0.01
        else:
            guess = cv2.GaussianBlur(d.transpose(1, 2, 0), (0, 0), 6).transpose(2, 0, 1)
            d = np.where(hid, guess * 1.15, d)
            c = 5.0 + rng.random(d.shape) * 0.01
        return d.astype(np.float32), c.astype(np.float32)


_orig = co3d.load_co3d_scene
def _spy(*a, **k):
    _data["d"] = _orig(*a, **k); return _data["d"]
co3d.load_co3d_scene = _spy

csv = os.path.join(out, "fake_co3d.csv")
if os.path.exists(csv): os.remove(csv)
H.run_benchmark(root, ["oracle", "blind"], cfg, out_csv=csv, model_factory=Fake, save_examples_dir=os.path.join(out, "ex"))
import pandas as pd
df = pd.read_csv(csv)
print(df.groupby("model")[["auroc", "cur", "cur_err", "absrel_S", "absrel_U", "absrel_U_ctrl"]].mean().round(3))
