# ---------------------------------------------------------------- kaggle driver, CO3D (appended after hallu3d.py + co3d.py)
# Runs Table 3: HalluBench-3D on the preprocessed CO3Dv2 single-sequence subset (dataset popsensharma/co3d-hallubench).
import sys, os, json, traceback, time
RUN_NAME = os.environ.get("RUN_NAME", "P2-co3d")
MODELS = os.environ.get("MODELS", "vggt,pi3x").split(",")
CFG = json.loads(os.environ.get("CFG", '{"seeds":[0],"fills":["gray"],"radius_fracs":[0.18]}'))
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/{RUN_NAME}.log", "a")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

t0 = time.time()
P("== setup", time.strftime("%Y-%m-%d %H:%M:%S"))
import torch
P("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
root = find_co3d_root()
scenes = list_co3d_scenes(root)
CFG = {"dataset": "co3d", "scans": scenes, **CFG}
P("co3d root:", root, len(scenes), "scenes:", scenes)

# geometry / mask correctness on every scene (cameras converted from PyTorch3D NDC, depth from COLMAP MVS)
chk = {}
for sc in scenes:
    d = load_co3d_scene(root, sc, select_views(CFG.get("n_views", DEFAULT_CFG["n_views"]), total=n_frames(root, sc),
                                               mode=CFG.get("view_mode", "spread")))
    m = build_masks(d, 0, 0.18, DEFAULT_CFG)
    chk[sc] = consistency_check(d, m)
    chk[sc]["U_frac_ref"] = float(m["U"][0].sum() / max((d["depths"][0] > 0).sum(), 1))
    P("consistency", sc, {k: round(v, 3) for k, v in chk[sc].items()})
json.dump(chk, open(f"{OUT}/{RUN_NAME}_consistency.json", "w"), indent=2)
bad = [sc for sc, c in chk.items() if not (c["reproj_agree_3pct_vis"] > 0.9 and c["U_transfer"] > 0.85)]
if bad:
    P("!! consistency gate FAILED on scenes", bad, "- continuing so the run is diagnosable, but do not use their metrics")

csv = f"{OUT}/{RUN_NAME}.csv"
for mname in MODELS:
    try:
        P("== model", mname, time.strftime("%H:%M:%S"))
        run_benchmark(root, [mname], cfg=CFG, out_csv=csv, save_examples_dir=f"{OUT}/ex_{RUN_NAME}")
    except Exception:
        P("!! model", mname, "failed:\n" + traceback.format_exc())
if os.path.exists(csv):
    import pandas as pd
    s = summarize(csv)
    P(s.to_string())
    s.to_csv(f"{OUT}/{RUN_NAME}_summary.csv", index=False)
P("== done, total %.0fs" % (time.time() - t0))
