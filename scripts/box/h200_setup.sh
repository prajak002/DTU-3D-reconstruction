#!/bin/bash
# One-time setup of a fresh GPU container (written for the Jarvislabs H200, 2026-11-07; root user, PyTorch image):
# repository, the five backbones (same recipe as the Kaggle kernels: scripts/kaggle_prelude.py, --no-deps so the
# container's torch stays), the DTU training set from Kaggle, and the preprocessed CO3Dv2 / ETH3D subsets from a
# tarball uploaded next to this script. Idempotent: every step checks what is already there.
#
#   ROOT=/home KAGGLE_API_TOKEN=... bash scripts/box/h200_setup.sh
#
# Afterwards `source $ROOT/env.sh` and run scripts/box/h200_all.sh in tmux.
set -u
export ROOT=${ROOT:-/home}
export REPO=$ROOT/hallu
export DATA=$ROOT/data
mkdir -p $DATA /kaggle/tmp
cd $ROOT
echo "== repo"
if [ ! -d $REPO ]; then git clone -q https://github.com/prajak002/DTU-3D-reconstruction.git $REPO; fi
cd $REPO && git fetch -q && git checkout -q ${HALLU3D_COMMIT:-origin/main} && git log --oneline -1
echo "== python packages"
python -m pip -q install -r requirements.txt 2>&1 | grep -v "already satisfied" | tail -3
python -m pip -q install kaggle truststore tensorboard 2>&1 | tail -1
echo "== backbones (kaggle_prelude.py recipe; clones under /kaggle/tmp)"
MODELS=vggt,pi3x,pi3,da3,mapanything python -c "exec(open('scripts/kaggle_prelude.py').read())"
cat > $ROOT/env.sh <<EOF
export PYTHONPATH=/kaggle/tmp/Pi3:$REPO/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OIA_CKPT_DIR=$REPO/checkpoints
export DTU=$DATA/dtu
export CO3D=$DATA/co3d
export ETH3D=$DATA/eth3d
cd $REPO
EOF
source $ROOT/env.sh
echo "== check imports"
python - <<'EOF'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for m in ("vggt", "pi3", "depth_anything_3", "mapanything"):
    try:
        importlib.import_module(m); print("ok  ", m)
    except Exception as e:
        print("FAIL", m, type(e).__name__, e)
EOF
echo "== data: CO3Dv2 and ETH3D (preprocessed tarball uploaded to $ROOT/prep_data.tgz)"
if [ -f $ROOT/prep_data.tgz ] && [ ! -d $DATA/co3d ]; then tar xzf $ROOT/prep_data.tgz -C $DATA && ls $DATA; fi
echo "== data: DTU training set (Kaggle tobyzhou/dtu-training; needs KAGGLE_API_TOKEN)"
if [ ! -d $DATA/dtu ]; then
  mkdir -p $DATA/dtu_zip && python - <<'EOF'
import os, truststore; truststore.inject_into_ssl()
from kaggle.cli import main
import sys
sys.argv = ["kaggle", "datasets", "download", "-d", "tobyzhou/dtu-training", "-p", os.path.expandvars("$DATA/dtu_zip"), "--unzip"]
main()
EOF
  mv $DATA/dtu_zip $DATA/dtu
fi
python -c "import hallu3d as H; print('DTU root:', H.find_dtu_root(('$DATA/dtu',)))"
df -h $ROOT | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== SETUP_DONE"
