#!/bin/bash
# Fourth H200 pass (2026-11-08): the provenance heads (run family P10-prov). Waits for the third pass (P9_DONE in its
# log), trains <base>_oiaprov (ground-truth per-view visibility) and <base>_oiaprovself (per-view support of the
# backbone's own prediction, label-free) for the four backbones, then scores them and the reprojection attribution
# baseline on the vis protocol of DTU, CO3Dv2, ETH3D and the landmark photographs with the attribution metrics.
#   tmux new -d -s p10 "bash scripts/box/h200_p10.sh > /home/logs/h200_p10.out 2>&1"
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
export ETH3D=${ETH3D:-$ETH}
cd ${ROOT:-/home}/hallu
while ! grep -q P9_DONE /home/logs/h200_p9.out 2>/dev/null; do sleep 60; done
echo "== P10 start $(date -u)"
python scripts/run_h100.py trainprov --dtu $DTU --bases vggt,pi3x,pi3,da3
echo "== trainprov end $(date -u) rc $?"
python scripts/run_h100.py prov --dtu $DTU --co3d $CO3D --eth3d $ETH3D --imc /home/data/imc --bases vggt,pi3x,pi3,da3
echo "== prov end $(date -u) rc $?"
md5sum checkpoints/*oiaprov*.pt > /home/logs/p10_ckpt_md5.txt
echo "== P10_DONE $(date -u)"
