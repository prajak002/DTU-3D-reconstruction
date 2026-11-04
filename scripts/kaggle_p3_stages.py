# Kaggle T4 kernel for the P3 stages: clones the public repository at a pinned commit, installs the backbones,
# trains the support heads of one backbone and runs the evaluation stages in order, stopping before the 12-hour
# session limit so the output (checkpoints and results/P3-*) is saved. Resumable: a second kernel that lists this
# one's output under kernel_sources starts from its checkpoints and CSVs (copied into /kaggle/working first).
# Configure with the constants below (BASES, STAGES, VARIANTS) or the environment.
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
REPO = "/kaggle/tmp/repo"
COMMIT = os.environ.get("HALLU3D_COMMIT", "74a13c7")       # 2026-11-04: joint heads, resume, controls on every row
BASES = os.environ.get("HALLU3D_BASES", "vggt")
STAGES = os.environ.get("HALLU3D_STAGES", "train,baselines,dtu,stress,ablate,co3d,dtuext").split(",")
VARIANTS = os.environ.get("HALLU3D_VARIANTS", "oia,oialin,oiapatch")   # oiaattn is left to the 24 GB GPU
STEPS = os.environ.get("HALLU3D_STEPS", "1500")
BUDGET_S = float(os.environ.get("HALLU3D_BUDGET_H", "11.2")) * 3600   # start no new stage after this
OUT, CK = "/kaggle/working/results", "/kaggle/working/checkpoints"
for d in ("/kaggle/tmp", OUT, CK):
    os.makedirs(d, exist_ok=True)


def banner(s):
    print(f"\n{'=' * 100}\n== {s}  [{time.time() - T0:.0f}s]\n{'=' * 100}", flush=True)


# previous kernel output (checkpoints and partial CSVs) -> working dir, so every stage resumes
for src in sorted(os.listdir("/kaggle/input")):
    p = os.path.join("/kaggle/input", src)
    for sub, dst in (("checkpoints", CK), ("results", OUT)):
        q = os.path.join(p, sub)
        if os.path.isdir(q):
            for root, _, files in os.walk(q):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), q)
                    t = os.path.join(dst, rel)
                    os.makedirs(os.path.dirname(t), exist_ok=True)
                    if not os.path.exists(t):
                        shutil.copy2(os.path.join(root, f), t)
            print(f"restored {sub} from {p}", flush=True)

banner("clone")
subprocess.run(["git", "clone", "-q", "https://github.com/prajak002/DTU-3D-reconstruction.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "-q", COMMIT], check=True)
print("commit", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())

banner("install backbones (kaggle_prelude.py)")
os.environ["MODELS"] = "vggt,pi3x,pi3,da3,mapanything" if "dtuext" in STAGES else BASES
exec(compile(open(f"{REPO}/scripts/kaggle_prelude.py").read(), "kaggle_prelude.py", "exec"))

ENV = dict(os.environ, PYTHONPATH=f"/kaggle/tmp/Pi3:{REPO}/scripts", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           OIA_CKPT_DIR=CK, PYTHONUNBUFFERED="1")


def sh(*args):
    print("\n$", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=REPO, env=ENV)
    print(f"[exit {r.returncode}, {time.time() - T0:.0f}s]", flush=True)
    return r.returncode


common = ["--dtu", "/kaggle/input", "--co3d", "/kaggle/input", "--out", OUT, "--ckpt", CK, "--bases", BASES,
          "--steps", STEPS, "--skip-gt", "--variants", VARIANTS]
for stage in STAGES:
    if time.time() - T0 > BUDGET_S:
        banner(f"budget reached before stage {stage}: stopping so the output is saved"); break
    banner(f"stage {stage}")
    sh(sys.executable, "scripts/run_h100.py", stage, *common)

banner("done")
for dp, dn, fn in os.walk("/kaggle/working"):
    for f in fn:
        p = os.path.join(dp, f)
        if not f.endswith((".png", ".jpg")):
            print(f"{os.path.getsize(p):>12,d}  {p}")
print(f"total {time.time() - T0:.0f}s")
