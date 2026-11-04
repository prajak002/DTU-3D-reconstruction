# Kaggle T4 kernel, run P3-check: the support-head feature hooks on the real backbones (run_h100.py check),
# a 200-step vggt_oia training smoke on six DTU training scans, and a 3-scan evaluation of the smoke head
# against native confidence on the Table-4 keys. Pulls the public repository at kernel start, so the kernel
# file itself never changes; bump the pinned commit below when the code moves.
import os, subprocess, sys, time

T0 = time.time()
REPO = "/kaggle/tmp/repo"
COMMIT = os.environ.get("HALLU3D_COMMIT", "ca2977b")   # 2026-11-01: support head + H100 plan
OUT, CK = "/kaggle/working/results", "/kaggle/working/checkpoints"
os.makedirs("/kaggle/tmp", exist_ok=True); os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)


def banner(s):
    print(f"\n{'=' * 100}\n== {s}  [{time.time() - T0:.0f}s]\n{'=' * 100}", flush=True)


banner("clone")
subprocess.run(["git", "clone", "-q", "https://github.com/prajak002/DTU-3D-reconstruction.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "-q", COMMIT], check=True)
print("commit", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())

banner("install backbones (kaggle_prelude.py)")
os.environ["MODELS"] = "vggt,pi3x"
exec(compile(open(f"{REPO}/scripts/kaggle_prelude.py").read(), "kaggle_prelude.py", "exec"))

ENV = dict(os.environ, PYTHONPATH=f"/kaggle/tmp/Pi3:{REPO}/scripts", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           OIA_CKPT_DIR=CK)


def sh(*args, cwd=REPO):
    print("\n$", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=cwd, env=ENV)
    print(f"[exit {r.returncode}, {time.time() - T0:.0f}s]", flush=True)
    return r.returncode


banner("stage check: hook shapes, timing, self-consistency of predicted depth + cameras")
sh(sys.executable, "scripts/run_h100.py", "check", "--dtu", "/kaggle/input", "--co3d", "/kaggle/input",
   "--out", OUT, "--ckpt", CK, "--bases", "vggt,pi3x")

banner("training smoke: vggt_oia, 200 steps, training scans 7,8,14,16 (2 and 6 are the validation scans)")
root = subprocess.run([sys.executable, "-c", "import hallu3d as H; print(H.find_dtu_root())"], cwd=f"{REPO}/scripts",
                      capture_output=True, text=True, env=ENV).stdout.strip().splitlines()[-1]
print("dtu root:", root)
sh(sys.executable, "scripts/oia_head.py", "train", "--base", "vggt", "--dtu", root, "--steps", "200",
   "--out", f"{CK}/vggt_oia.pt", "--scans", "2,6,7,8,14,16")

banner("evaluation smoke: vggt vs vggt_oia vs vggt_oiafuse, 3 benchmark scans, Table-4 keys (seed 0, blur, r 0.18)")
code = f"""
import os, hallu3d as H, pandas as pd
root = H.find_dtu_root()
cfg = {{"seeds": [0], "fills": ["blur"], "radius_fracs": [0.18], "scans": H.DEFAULT_CFG["scans"][:3]}}
d = "{OUT}/P3-smoke"; os.makedirs(d, exist_ok=True); csv = d + "/P3-smoke.csv"
for m in ["vggt", "vggt_oia", "vggt_oiafuse"]:
    try:
        H.run_benchmark(root, [m], cfg=cfg, out_csv=csv, save_examples_dir=d + "/ex_P3-smoke")
    except Exception as e:
        import traceback; print("!!", m, "failed:"); traceback.print_exc()
if os.path.exists(csv):
    s = H.summarize(csv); s.to_csv(d + "/P3-smoke_summary.csv", index=False); print(s.to_string())
    df = pd.read_csv(csv)
    print(df[["model", "scan", "auroc", "cur", "absrel_U"]].to_string())
"""
sh(sys.executable, "-c", code, cwd=f"{REPO}/scripts")

banner("done")
for dp, dn, fn in os.walk("/kaggle/working"):
    for f in fn:
        p = os.path.join(dp, f)
        if not f.endswith((".png", ".jpg")):
            print(f"{os.path.getsize(p):>12,d}  {p}")
print(f"total {time.time() - T0:.0f}s")
