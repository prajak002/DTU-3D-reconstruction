# Kaggle GPU kernel for the sixteenth pass (declared in LOG.md before any run): moving scenes, full video. TUM RGB-D
# "dynamic objects" sequences and static sequences of the same camera, in temporal order, cut into consecutive
# windows of 10 frames; scored with the DTU-trained label-free heads of record. Clones the public repository at a
# pinned commit, installs the backbones, downloads and preprocesses the sequences (scripts/tum.py --n-frames 200, no
# reordering) and runs scripts/dynamic_eval.py for the backbones in BASES. The heads come from the attached dataset
# popsensharma/hallu3d-oiavisself-heads (md5 printed below).
# Output: /kaggle/working/results/P23-dynamic/<backbone>/P23-dynamic.csv (+ per-pixel dump of one sequence).
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
BASES = os.environ.get("HALLU3D_BASES", "vggt,pi3x").split(",")
OUT, CK = "/kaggle/working/results/P23-dynamic", "/kaggle/tmp/checkpoints"
RAW, TUM = "/kaggle/tmp/tum_raw", "/kaggle/tmp/tum_video"
SC = ("freiburg3_sitting_static,freiburg3_sitting_xyz,freiburg3_sitting_halfsphere,freiburg3_sitting_rpy,"
      "freiburg3_walking_static,freiburg3_walking_xyz,freiburg3_walking_halfsphere,freiburg3_walking_rpy,"
      "freiburg3_long_office_household,freiburg3_teddy,freiburg3_cabinet,freiburg3_large_cabinet,"
      "freiburg3_structure_texture_far")
N_FRAMES, WINDOW, DUMP = "200", "10", "freiburg3_walking_xyz"
for d in (OUT, CK, RAW, TUM):
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

banner("install backbones (kaggle_prelude.py)")
os.environ["MODELS"] = ",".join(BASES)
exec(compile(open(f"{REPO}/scripts/kaggle_prelude.py").read(), "kaggle_prelude.py", "exec"))
ENV = dict(os.environ, PYTHONPATH=f"/kaggle/tmp/Pi3:/kaggle/tmp/da3/src:{REPO}/scripts",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OIA_CKPT_DIR=CK, PYTHONUNBUFFERED="1")


def sh(*args):
    print("\n$", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=REPO, env=ENV)
    print(f"[exit {r.returncode}, {time.time() - T0:.0f}s]", flush=True)
    return r.returncode


banner("label self-test (no data)")
if sh(sys.executable, "scripts/dynamic_eval.py", "--selftest") != 0:
    raise SystemExit("label self-test failed")

banner("TUM RGB-D: download, preprocess in temporal order")
for s in SC.split(","):
    url = f"https://cvg.cit.tum.de/rgbd/dataset/{s.split('_')[0]}/rgbd_dataset_{s}.tgz"
    for attempt in range(3):
        r = subprocess.run(f"curl -sS -L --retry 3 -o {s}.tgz {url} && tar xzf {s}.tgz && rm -f {s}.tgz", shell=True, cwd=RAW)
        if r.returncode == 0:
            break
    print(s, "ok" if r.returncode == 0 else "FAILED", f"{time.time() - T0:.0f}s", flush=True)
    sh(sys.executable, "scripts/tum.py", RAW, TUM, "--scenes", s, "--n-frames", N_FRAMES)
    shutil.rmtree(os.path.join(RAW, "rgbd_dataset_" + s), ignore_errors=True)
print(sorted(os.listdir(TUM)), flush=True)

banner("labels only (no model): shares of S, U and M per window")
sh(sys.executable, "scripts/dynamic_eval.py", "--root", TUM, "--window", WINDOW, "--labels-only", "--out", f"{OUT}/labels")

for b in BASES:
    banner(f"moving scenes: {b}")
    os.makedirs(f"{OUT}/{b}", exist_ok=True)
    shutil.copytree(f"{OUT}/labels/labels_cache", f"{OUT}/{b}/labels_cache", dirs_exist_ok=True)
    sh(sys.executable, "scripts/dynamic_eval.py", "--root", TUM, "--window", WINDOW,
       "--models", f"{b},{b}_reproj,{b}_oiavisself", "--dump-scenes", DUMP, "--out", f"{OUT}/{b}")
    shutil.rmtree(f"{OUT}/{b}/labels_cache", ignore_errors=True)
shutil.rmtree(f"{OUT}/labels/labels_cache", ignore_errors=True)

banner("done")
for p in sorted(glob.glob(f"{OUT}/*/*.csv")):
    print(p, os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest())
