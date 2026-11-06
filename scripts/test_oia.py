"""CPU test of the support head (scripts/oia_head.py) with a fake backbone on the synthetic DTU-format scan.

The fake backbone returns the ground-truth depth and cameras as its "prediction" and, as features, per-patch
colour statistics of the input images (mean, std, gradient energy). That is enough for the head to learn where
a region was hidden, so the test checks the plumbing end to end: self-supervised sample generation from the
backbone's own prediction, head construction from feature dims, training, checkpointing, OIAWrap through
hallu3d.run_benchmark, and the model-name routing in hallu3d.make_model. It does not test the real feature hooks
(those need the GPU and the model packages; see scripts/run_h100.py, stage check).
Exit code 0 when all checks pass."""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H
import oia_head as O

WORK = os.environ.get("HALLU3D_TMP") or tempfile.mkdtemp(prefix="hallu3d_oia_")
os.makedirs(WORK, exist_ok=True)
env = {**os.environ, "HALLU3D_TMP": WORK, "PYTHONIOENCODING": "utf-8"}
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_synthetic.py")],
                   env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
root = H.find_dtu_root((os.path.join(WORK, "fake_dtu"),))
import torch

MODEL_HW = (224, 280)       # small, divisible by 14 and 2, to keep the CPU test short (benchmark uses 420x518)
GT = {}                     # the fake backbone "predicts" the GT of the scene currently loaded
GT_BY_N = {}                # ... or, keyed by the number of input views, the GT of the vis protocol's view sets


class FakeWrap:
    name = "fake"

    def __init__(self, device="cpu"):
        self.torch, self.device, self.dtype = torch, device, None
        self.model = None

    _prep = H.ModelWrap._prep

    def free(self):
        pass


def fake_forward_feats(wrap, x, dtype, feats="agg"):
    N, Hm, Wm = x.shape[0], x.shape[-2], x.shape[-1]
    ph, pw = Hm // O.PATCH, Wm // O.PATCH
    xp = x[:, :, :ph * O.PATCH, :pw * O.PATCH].reshape(N, 3, ph, O.PATCH, pw, O.PATCH)
    mean = xp.mean((3, 5))
    std = xp.std((3, 5))
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()[:, :, :ph * O.PATCH - 1, :pw * O.PATCH]
    gy = torch.nn.functional.adaptive_avg_pool2d(gy, (ph, pw))
    F = [torch.cat([mean, std], 1), gy] if feats == "agg" else [mean]
    g = GT_BY_N.get(N, GT)
    d = torch.from_numpy(g["depths"]).float()
    d = torch.nn.functional.interpolate(d[:, None], size=(Hm, Wm), mode="nearest")[:, 0]
    conf = torch.ones_like(d)
    K = torch.from_numpy(O._scale_K(g["K_dep"], Hm / g["depths"].shape[1], Wm / g["depths"].shape[2])).float()
    E = torch.from_numpy(np.asarray(g["E"])).float()
    return d, conf, F, K, E


O.FORWARD_FEATS["fake"] = fake_forward_feats
H.MODEL_REGISTRY["fake"] = FakeWrap

# the synthetic root has a single scan (3); train and validate on it with different seeds / fills
data = H.load_scan(root, 3, H.select_views(10), 3)
GT.update(data)
ck = O.train("fake", root, "oia", out=os.path.join(WORK, "ckpt", "fake_oia.pt"), steps=60, seed=0, device="cpu",
             scans=[3], wrap=FakeWrap(), model_hw=MODEL_HW, lr=2e-3, warmup=5, log_every=10, val_every=30,
             val_scans=[], fills=["gray", "noise"], random_light=False, p_clean=0.1)
assert os.path.exists(ck) and os.path.exists(ck.replace(".pt", "_hist.json"))
hist = __import__("json").load(open(ck.replace(".pt", "_hist.json")))
losses = [h["loss"] for h in hist if "loss" in h]
print("loss first/last:", losses[0], losses[-1])

# evaluation through the benchmark, on a fill the head never saw (blur) and on a seen one (gray)
os.environ["OIA_CKPT_FAKE_OIA"] = ck
cfg = {**H.DEFAULT_CFG, "scans": [3], "seeds": [7], "radius_fracs": [0.15], "fills": ["gray", "blur"], "model_hw": MODEL_HW}
csv = os.path.join(WORK, "oia_results.csv")
if os.path.exists(csv):
    os.remove(csv)
H.run_benchmark(root, ["fake_oia", "fake_oiafuse"], cfg, out_csv=csv, verbose=True)
import pandas as pd
df = pd.read_csv(csv)
print(df[["model", "fill", "auroc", "cur", "absrel_S", "absrel_U"]].to_string())
g = df[(df.model == "fake_oia") & (df.fill == "gray")].iloc[0]
b = df[(df.model == "fake_oia") & (df.fill == "blur")].iloc[0]

# routing + three-way map
m = O.make_oia("fake_oia")
score = np.linspace(0, 1, 11)
tw = O.three_way(score)
checks = {
    "loss decreased": losses[-1] < losses[0],
    "head beats chance on the seen fill (auroc > 0.8)": g["auroc"] > 0.8,
    "cur below 0.5 on the seen fill": g["cur"] < 0.5,
    "unseen fill still above chance (auroc > 0.6)": b["auroc"] > 0.6,
    "depth passed through by the wrapper (absrel_S < 0.01, resampling only)": g["absrel_S"] < 0.01,
    "make_model routes <base>_oia": isinstance(m, O.OIAWrap) and m.name == "fake_oia",
    "three_way labels": tw[0] == 1 and tw[-1] == 0 and tw[5] == 2,
}

# ---------------------------------------------------------------- true-visibility labels and heads (P4-vis)
# support_count: a view projected into an identical copy of itself sees every known pixel exactly once
d3 = H.load_scan(root, 3, H.vis_views(3, 1), 3)
dup = dict(depths=np.stack([d3["depths"][0]] * 2), K_dep=np.stack([d3["K_dep"][0]] * 2), E=np.stack([d3["E"][0]] * 2))
seen_dup, unk_dup = H.support_count(dup["depths"], dup["K_dep"], dup["E"])
valid0 = d3["depths"][0] > 0
lab3 = H.vis_labels(d3)
lab10 = H.vis_labels(H.load_scan(root, 3, H.vis_views(10, 0), 3))
print("vis labels, 3 views: U", int(lab3["U"].sum()), "S", int(lab3["S"].sum()),
      "| 10 views: U", int(lab10["U"].sum()), "S", int(lab10["S"].sum()))
GT.update(d3)
cks = O.train_multi("fake", root, [("oiavis", 0, os.path.join(WORK, "ckpt", "fake_oiavis.pt")),
                                   ("oiavispatch", 0, os.path.join(WORK, "ckpt", "fake_oiavispatch.pt"))],
                    steps=30, device="cpu", scans=[3], wrap=FakeWrap(), model_hw=MODEL_HW, lr=2e-3, warmup=5,
                    log_every=10, val_every=15, val_scans=[], random_light=False, vis_views=(3, 5))
os.environ["OIA_CKPT_FAKE_OIAVIS"] = cks[0]
os.environ["OIA_CKPT_FAKE_OIAVISPATCH"] = cks[1]
for n in (3, 5):            # the fake backbone must return depth for the view set run_vis loads
    GT_BY_N[n] = H.load_scan(root, 3, H.vis_views(n, 0), 3)
vcsv = os.path.join(WORK, "vis_results.csv")
if os.path.exists(vcsv):
    os.remove(vcsv)
H.run_vis(root, ["fake_oiavis", "fake_oiavispatch"], {"scans": [3], "vis_views": [3, 5], "seeds": [0], "model_hw": MODEL_HW},
          out_csv=vcsv)
H.run_vis(root, ["fake_oiavis"], {"scans": [3], "vis_views": [3, 5], "seeds": [0], "model_hw": MODEL_HW}, out_csv=vcsv)
dv = pd.read_csv(vcsv)
print(dv[["model", "fill", "n_U", "n_S", "auroc", "cur", "absrel_S", "absrel_U"]].to_string())
mv = O.make_oia("fake_oiavispatch")
checks.update({
    "support_count: duplicated view sees every known pixel once": bool((seen_dup[0][valid0] == 1).all()) and int(unk_dup.sum()) == 0,
    "vis labels: U and S disjoint, both non-empty with 3 views": bool(lab3["U"].sum() > 0 and lab3["S"].sum() > 0 and not (lab3["U"] & lab3["S"]).any()),
    "vis labels: fewer single-view pixels with 10 views than with 3": int(lab10["U"].sum()) <= int(lab3["U"].sum()),
    "vis heads train jointly and checkpoint": all(os.path.exists(c) for c in cks),
    "run_vis writes one row per (model, views) and resumes": len(dv) == 4 and dv.auroc.notna().all(),
    "make_model routes <base>_oiavispatch to the per-image features": isinstance(mv, O.OIAWrap) and mv.feats == "patch",
})
failed = [k for k, ok in checks.items() if not ok]
print("checks:", "all passed" if not failed else "FAILED " + ", ".join(failed))
print("scratch dir:", WORK)
sys.exit(1 if failed else 0)
