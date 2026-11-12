# Kaggle GPU kernel for the thirteenth pass (pre-registered Gate 21 in LOG.md): the real-occlusion test with the
# hole-robust labels, scored for four detectors per backbone in one job so that every comparison is paired on the
# same points: native confidence, the DTU-trained label-free head, the aligned perturbation variance (K = 8,
# `<backbone>_pvf8`, the training-free detector in the style of ReVar3R) and reprojection with the model's own
# cameras (`<backbone>_reproj`, the head's teacher). Stage 1: TUM RGB-D, 12 sequences, 3, 5 and 10 views (protocol
# of Gate 19). Stage 2: the 18 DTU benchmark scans at 3 views (protocol of Gate 18). Clones the public repository at
# a pinned commit; the heads come from the attached dataset popsensharma/hallu3d-oiavisself-heads (md5 printed
# below), DTU from tobyzhou/dtu-training.
# Output: /kaggle/working/results/P20-baselines/<backbone>/P12-realocc-tum.csv and <backbone>_dtu/P12-realocc.csv.
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
STAGES = os.environ.get("HALLU3D_STAGES", "1,2,3,4").split(",")      # "hd": full-resolution dumps for the figure
TUM_BASES = [b for b in os.environ.get("HALLU3D_TUM_BASES", ",".join(BASES)).split(",") if b]
HD_SCANS = "37,82,28"   # the three benchmark scans with the largest single-view share (rule fixed in LOG.md)
OUT, CK = "/kaggle/working/results/P20-baselines", "/kaggle/tmp/checkpoints"
RAW, TUM = "/kaggle/tmp/tum_raw", "/kaggle/tmp/tum"
SC = ("freiburg1_desk,freiburg1_desk2,freiburg1_room,freiburg1_plant,freiburg1_teddy,freiburg1_360,freiburg2_desk,"
      "freiburg3_long_office_household,freiburg3_teddy,freiburg3_cabinet,freiburg3_large_cabinet,"
      "freiburg3_structure_texture_far")
for d in (OUT, CK, RAW, TUM):
    os.makedirs(d, exist_ok=True)


def banner(s):
    print(f"\n{'=' * 100}\n== {s}  [{time.time() - T0:.0f}s]\n{'=' * 100}", flush=True)


def detectors(b):
    return f"{b},{b}_oiavisself,{b}_pvf8,{b}_reproj"


HAD_ROOT, HAD_COMMIT = "/kaggle/tmp/HAD", "484afb4ef24fb62de56c2233123e8030cb82c604"
HAD_CKPT = "/kaggle/tmp/had_ckpt/ckpt_0000000000010000.pt"
HAD_SHA256 = "dd272986eaded1eee8058af4cc619ec94c0dab320942e2549db6d1ea84a2acce"   # from the HAD README
HAD_OK = False


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


banner("HAD scoring network (optional baseline; the rest of the job does not depend on it)")
try:
    subprocess.run(["git", "clone", "-q", "https://github.com/xiliu8006/HAD.git", HAD_ROOT], check=True)
    subprocess.run(["git", "-C", HAD_ROOT, "checkout", "-q", HAD_COMMIT], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easydict", "omegaconf", "einops", "lpips",
                    "torchmetrics", "jaxtyping", "imageio", "gdown"], check=True)
    os.makedirs(os.path.dirname(HAD_CKPT), exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", "1004LRTsUr0k1D42Ivcg2Q2Zbzjx2mVCg", "-O", HAD_CKPT], check=True)
    sha = hashlib.sha256(open(HAD_CKPT, "rb").read()).hexdigest()
    print("HAD checkpoint sha256", sha, "matches README" if sha == HAD_SHA256 else "DOES NOT MATCH README", flush=True)
    HAD_OK = sha == HAD_SHA256
    ENV.update(HAD_ROOT=HAD_ROOT, HAD_CKPT=HAD_CKPT)
except Exception as e:
    print("HAD setup failed, its rows are skipped:", repr(e), flush=True)

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

if "hd" in STAGES:
    banner("full-resolution dumps for the comparison figure")
    sh(sys.executable, "scripts/dump_compare.py", "--dtu", "/kaggle/input", "--bases", ",".join(BASES),
       "--scans", HD_SCANS, "--hd", "--out", "/kaggle/working/results/P20-dump-hd")

for b in (TUM_BASES if "1" in STAGES else []):
    banner(f"stage 1, TUM RGB-D: {b}")
    sh(sys.executable, "scripts/realocc_eval.py", "--co3d-format", TUM, "--scans", SC, "--models", detectors(b),
       "--align", "points", "--label-mode", "zagree", "--out", f"{OUT}/{b}")

for b in (BASES if "2" in STAGES else []):
    banner(f"stage 2, DTU benchmark scans at 3 views: {b}")
    sh(sys.executable, "scripts/realocc_eval.py", "--dtu", "/kaggle/input", "--models", detectors(b),
       "--views", "3", "--label-mode", "zagree", "--out", f"{OUT}/{b}_dtu")

if "3" in STAGES:
    banner("stage 3, per-pixel dumps for the comparison figure: 18 DTU benchmark scans, 5 views, seed 0")
    sh(sys.executable, "scripts/dump_compare.py", "--dtu", "/kaggle/input", "--bases", ",".join(BASES),
       "--out", "/kaggle/working/results/P20-dump")

if HAD_OK and "4" in STAGES:   # separate invocations appended to the same CSVs: a failure here cannot touch the rows above
    banner("stage 4, HAD scorer: dumps, then real occlusion at 3 and 5 views")
    sh(sys.executable, "scripts/dump_compare.py", "--dtu", "/kaggle/input", "--bases", ",".join(BASES),
       "--detectors", "had", "--out", "/kaggle/working/results/P20-dump")
    if "hd" in STAGES:
        sh(sys.executable, "scripts/dump_compare.py", "--dtu", "/kaggle/input", "--bases", ",".join(BASES),
           "--scans", HD_SCANS, "--hd", "--detectors", "had", "--out", "/kaggle/working/results/P20-dump-hd")
    for b in BASES:
        sh(sys.executable, "scripts/realocc_eval.py", "--dtu", "/kaggle/input", "--models", f"{b}_had",
           "--views", "3", "--label-mode", "zagree", "--out", f"{OUT}/{b}_dtu")
        sh(sys.executable, "scripts/realocc_eval.py", "--co3d-format", TUM, "--scans", SC, "--models", f"{b}_had",
           "--views", "3,5", "--align", "points", "--label-mode", "zagree", "--out", f"{OUT}/{b}")

banner("done")
for p in sorted(glob.glob(f"{OUT}/*/*.csv")):
    print(p, os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest())
