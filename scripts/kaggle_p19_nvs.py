# Kaggle GPU kernel for the novel-view test (scripts/nvs_eval.py): clones the public repository at a pinned commit,
# installs the backbones, gsplat and lpips, and runs the test for the backbones in BASES on the DTU scans in SCANS.
# The heads come from the attached dataset popsensharma/hallu3d-oiavisself-heads, DTU from tobyzhou/dtu-training.
# Output: /kaggle/working/results/P19-nvs/<TAG>/P19-nvs.csv (and a few render strips).
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
REPO = "/kaggle/tmp/repo"
COMMIT = os.environ.get("HALLU3D_COMMIT", "PINNED_COMMIT")
BASES = os.environ.get("HALLU3D_BASES", "vggt").split(",")
SCANS = os.environ.get("HALLU3D_SCANS", "PINNED_SCANS")
VIEWS = os.environ.get("HALLU3D_VIEWS", "3,6")
TAG = os.environ.get("HALLU3D_TAG", "pilot")
OUT, CK = f"/kaggle/working/results/P19-nvs/{TAG}", "/kaggle/tmp/checkpoints"
for d in (OUT, CK):
    os.makedirs(d, exist_ok=True)


def banner(s):
    print(f"\n{'=' * 100}\n== {s}  [{time.time() - T0:.0f}s]\n{'=' * 100}", flush=True)


banner("clone")
subprocess.run(["git", "clone", "-q", "https://github.com/prajak002/DTU-3D-reconstruction.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "-q", COMMIT], check=True)
print("commit", subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())

banner("heads")
for b in BASES:
    hits = glob.glob(f"/kaggle/input/**/{b}_oiavisself.pt", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{b}_oiavisself.pt not found under /kaggle/input")
    shutil.copy2(hits[0], CK)
    print(b, hashlib.md5(open(hits[0], "rb").read()).hexdigest(), hits[0], flush=True)

banner("install backbones (kaggle_prelude.py), gsplat, lpips")
os.environ["MODELS"] = ",".join(BASES)
exec(compile(open(f"{REPO}/scripts/kaggle_prelude.py").read(), "kaggle_prelude.py", "exec"))
_pip("lpips", "ninja", "jaxtyping", "rich")
_pip("--no-deps", "gsplat")
ENV = dict(os.environ, PYTHONPATH=f"/kaggle/tmp/Pi3:/kaggle/tmp/da3/src:{REPO}/scripts",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OIA_CKPT_DIR=CK, PYTHONUNBUFFERED="1")


def sh(*args):
    print("\n$", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=REPO, env=ENV)
    print(f"[exit {r.returncode}, {time.time() - T0:.0f}s]", flush=True)
    return r.returncode


banner("gsplat rasterizer (builds its CUDA kernels on first use)")
sh(sys.executable, "-c", "import torch, gsplat; from gsplat import rasterization; "
   "m = torch.rand(100, 3).cuda(); m[:, 2] += 2; q = torch.tensor([1., 0, 0, 0]).repeat(100, 1).cuda(); "
   "o = rasterization(m, q, torch.full((100, 3), 0.05).cuda(), torch.full((100,), 0.5).cuda(), torch.rand(100, 3).cuda(), "
   "torch.eye(4)[None].cuda(), torch.tensor([[[100., 0, 32], [0, 100., 32], [0, 0, 1]]]).cuda(), 64, 64); "
   "print('gsplat', gsplat.__version__, o[0].shape, float(o[1].mean()))")

banner(f"novel-view test: {BASES} scans {SCANS} views {VIEWS}")
sh(sys.executable, "scripts/nvs_eval.py", "--dtu", "/kaggle/input", "--models", ",".join(BASES), "--scans", SCANS,
   "--views", VIEWS, "--save-images", "6", "--out", OUT)

banner("done")
for p in sorted(glob.glob(f"{OUT}/*.csv")):
    print(p, os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest())
