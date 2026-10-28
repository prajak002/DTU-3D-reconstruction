# ---------------------------------------------------------------- kaggle stress-test driver (appended after hallu3d.py)
# Each test is a cfg override run with the same models; one CSV per test in /kaggle/working/out/.
# Reference rows (10 spread views, blur, r 0.18, seeds 0 and 1) already exist in P1-full / P2-full.
import sys, os, json, traceback, time
RUN_NAME = os.environ.get("RUN_NAME", "P2-stress")
MODELS = os.environ.get("MODELS", "vggt,pi3x,da3").split(",")
COMMON = {"seeds": [0, 1], "radius_fracs": [0.18], "keep_model": True}
TESTS = {
    "inpaint10":    {"fills": ["inpaint"], "n_views": 10},
    "blur5":        {"fills": ["blur"], "n_views": 5},
    "blur3":        {"fills": ["blur"], "n_views": 3},
    "adjacent10":   {"fills": ["blur"], "n_views": 10, "view_mode": "adjacent"},
    "distractor10": {"fills": ["blur"], "n_views": 10, "distractor": True},
}
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/{RUN_NAME}.log", "a")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
t0 = time.time()
import torch
P("== setup", time.strftime("%Y-%m-%d %H:%M:%S"), "torch", torch.__version__, torch.cuda.get_device_name(0))
root = find_dtu_root(); P("dtu root:", root)
for mname in MODELS:
    try:
        model = make_model(mname)
    except Exception:
        P("!! load", mname, "failed:\n" + traceback.format_exc()); continue
    for tname, over in TESTS.items():
        cfg = {**COMMON, **over}
        try:
            P("== model", mname, "test", tname, time.strftime("%H:%M:%S"))
            run_benchmark(root, [mname], cfg=cfg, out_csv=f"{OUT}/{RUN_NAME}_{tname}.csv",
                          model_factory=lambda n: model, save_examples_dir=f"{OUT}/ex_{RUN_NAME}_{tname}" if tname == "inpaint10" else None)
        except Exception:
            P("!! model", mname, "test", tname, "failed:\n" + traceback.format_exc())
    model.free()
import pandas as pd
for tname in TESTS:
    f = f"{OUT}/{RUN_NAME}_{tname}.csv"
    if os.path.exists(f):
        P("==", tname); P(summarize(f).to_string())
P("== done, total %.0fs" % (time.time() - t0))
