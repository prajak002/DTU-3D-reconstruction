# Kaggle GPU kernel for the eleventh pass (pre-registered Gate 19 in LOG.md): real occlusion with the hole-robust
# labels on TUM RGB-D, scored with the DTU-trained label-free heads. Clones the public repository at a pinned commit,
# installs the backbones, downloads and preprocesses the 12 sequences (scripts/tum.py, co-visibility order), and runs
# scripts/realocc_eval.py for the backbones in BASES. The heads come from the attached dataset
# popsensharma/hallu3d-oiavisself-heads (the four *_oiavisself.pt of record, md5 printed below).
# Output: /kaggle/working/results/P18-tum/<backbone>/P12-realocc-tum.csv.
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
OUT, CK = "/kaggle/working/results/P18-tum", "/kaggle/tmp/checkpoints"
RAW, TUM = "/kaggle/tmp/tum_raw", "/kaggle/tmp/tum"
SC = ("freiburg1_desk,freiburg1_desk2,freiburg1_room,freiburg1_plant,freiburg1_teddy,freiburg1_360,freiburg2_desk,"
      "freiburg3_long_office_household,freiburg3_teddy,freiburg3_cabinet,freiburg3_large_cabinet,"
      "freiburg3_structure_texture_far")
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


banner("TUM RGB-D: download, preprocess, order by co-visibility")
jobs = []
for s in SC.split(","):
    url = f"https://cvg.cit.tum.de/rgbd/dataset/{s.split('_')[0]}/rgbd_dataset_{s}.tgz"
    for attempt in range(3):
        r = subprocess.run(f"curl -sS -L --retry 3 -o {s}.tgz {url} && tar xzf {s}.tgz && rm -f {s}.tgz", shell=True, cwd=RAW)
        if r.returncode == 0:
            break
    print(s, "ok" if r.returncode == 0 else "FAILED", f"{time.time() - T0:.0f}s", flush=True)
    sh(sys.executable, "scripts/tum.py", RAW, TUM, "--scenes", s)
    shutil.rmtree(os.path.join(RAW, "rgbd_dataset_" + s), ignore_errors=True)
    jobs.append(subprocess.Popen([sys.executable, "scripts/eth3d.py", TUM, "--reorder", "--scenes", s, "--stride", "4"],
                                 cwd=REPO, env=ENV))   # co-visibility order, one process per sequence
print("reorder exit codes", [j.wait() for j in jobs], f"{time.time() - T0:.0f}s", flush=True)
print(sorted(os.listdir(TUM)), flush=True)

for b in BASES:
    banner(f"real occlusion: {b}")
    sh(sys.executable, "scripts/realocc_eval.py", "--co3d-format", TUM, "--scans", SC, "--models", f"{b},{b}_oiavisself",
       "--align", "points", "--label-mode", "zagree", "--out", f"{OUT}/{b}")

banner("done")
for p in sorted(glob.glob(f"{OUT}/*/*.csv")):
    print(p, os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest())
