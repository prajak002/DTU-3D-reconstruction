# ---------------------------------------------------------------- kaggle prelude (runs BEFORE hallu3d.py imports numpy)
# Pi3 pins numpy==1.26.4 / torch==2.5.1, DA3 pins numpy<2, MapAnything pins opencv 4.10. Installing any of them
# with deps downgrades numpy under a running numpy 2.x and breaks np.random. So: --no-deps everywhere, then
# add only light pure-python extras. Repos are cloned to /kaggle/tmp so they do not bloat the kernel output.
import subprocess, sys, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _pip(*a): subprocess.run([sys.executable, "-m", "pip", "-q", "install", *a], check=False)
def _clone(url, d):
    if not os.path.isdir(d):
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, d], check=False)
import re as _re
NEED = {_re.sub(r"_pv\d+$", "", m) for m in os.environ.get("MODELS", "vggt,pi3x").split(",")}  # vggt_pv4 -> vggt
_pip("einops", "safetensors", "huggingface_hub", "plyfile")
if NEED & {"vggt"}:
    _pip("--no-deps", "git+https://github.com/facebookresearch/vggt.git")
if NEED & {"pi3", "pi3x"}:
    _clone("https://github.com/yyfz/Pi3.git", "/kaggle/tmp/Pi3"); sys.path.append("/kaggle/tmp/Pi3")
if NEED & {"da3"}:
    _clone("https://github.com/ByteDance-Seed/Depth-Anything-3.git", "/kaggle/tmp/da3")
    _pip("addict", "omegaconf", "evo", "imageio", "trimesh", "pillow_heif", "typer", "moviepy==1.0.3", "pycolmap")
    _pip("--no-deps", "-e", "/kaggle/tmp/da3"); sys.path.append("/kaggle/tmp/da3/src")
if NEED & {"mapanything", "mapanything_mv"}:
    _clone("https://github.com/facebookresearch/map-anything.git", "/kaggle/tmp/mapanything")
    _pip("hydra-core", "python-box", "natsort", "orjson", "pillow-heif", "trimesh",
         "jaxtyping", "timm", "termcolor", "torchmetrics", "minio", "scikit-learn", "rerun-sdk")  # uniception deps
    _pip("--no-deps", "uniception==0.1.7")
    _pip("--no-deps", "-e", "/kaggle/tmp/mapanything"); sys.path.append("/kaggle/tmp/mapanything")
