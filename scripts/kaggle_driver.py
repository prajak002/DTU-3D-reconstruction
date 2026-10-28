# ---------------------------------------------------------------- kaggle driver (appended after hallu3d.py)
# RUN_NAME and CFG are the only knobs. Output lands in /kaggle/working/out/.
import subprocess, sys, os, json, traceback, time
RUN_NAME = os.environ.get("RUN_NAME", "P1-smoke")
MODELS = os.environ.get("MODELS", "vggt,pi3x").split(",")
CFG = json.loads(os.environ.get("CFG", '{"seeds":[0],"fills":["gray"],"radius_fracs":[0.15]}'))
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/{RUN_NAME}.log", "a")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

t0 = time.time()
P("== setup", time.strftime("%Y-%m-%d %H:%M:%S"))
import torch
P("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
root = find_dtu_root()
P("dtu root:", root, "setup %.0fs" % (time.time() - t0))

# P1-check: geometry / mask correctness on real DTU
views = select_views(DEFAULT_CFG["n_views"])
chk = {}
for sc in [3, 5]:
    d = load_scan(root, sc, views, DEFAULT_CFG["light"])
    m = build_masks(d, 0, 0.15, DEFAULT_CFG)
    chk[sc] = consistency_check(d, m)
    chk[sc]["U_frac_ref"] = float(m["U"][0].sum() / max((d["depths"][0] > 0).sum(), 1))
    P("consistency scan", sc, chk[sc])
json.dump(chk, open(f"{OUT}/{RUN_NAME}_consistency.json", "w"), indent=2)
bad = [sc for sc, c in chk.items() if not (c["reproj_agree"] > 0.8 and c["U_transfer"] > 0.95)]
if bad:
    P("!! consistency gate FAILED on scans", bad, "- continuing anyway so the run is diagnosable, but do not use metrics")

# benchmark, one model at a time so one failure does not kill the others
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
    df = pd.read_csv(csv)
    if df.model.nunique() >= 2:
        ms = sorted(df.model.unique())
        for col in ["auroc", "cur", "cur_err"]:
            P("paired perm", ms[0], "vs", ms[1], col, paired_permutation(df, col, ms[0], ms[1]))
P("== done, total %.0fs" % (time.time() - t0))
