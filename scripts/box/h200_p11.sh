#!/bin/bash
# Fifth H200 pass (2026-11-08): the matched-patch protocol (P11-matched, Gates 6 and 7) and real occlusion against the
# dense scan (P12-realocc, Gate 8). Two tmux sessions share the card:
#   tmux new -d -s p11t "bash scripts/box/h200_p11.sh train > /home/logs/h200_p11t.out 2>&1"  # matched heads
#   tmux new -d -s p11  "bash scripts/box/h200_p11.sh score > /home/logs/h200_p11.out 2>&1"   # natives, reproj, heads
# `score` first runs the five native models and the reprojection detectors on DTU and CO3Dv2 (Gate 6 arrives first),
# then waits for the VGGT and Pi3X matched heads, scores them on both datasets, then real occlusion; the Pi3 and DA3
# heads are scored at the end when their training is done.
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
mode=${1:-score}
if [ "$mode" = train ]; then
  echo "== P11 train start $(date -u)"
  python scripts/run_h100.py trainmatch --dtu $DTU --bases vggt,pi3x
  echo "== P11_TRAIN_MAIN_DONE $(date -u) rc $?"
  python scripts/run_h100.py trainmatch --dtu $DTU --bases pi3,da3
  echo "== P11_TRAIN_DONE $(date -u) rc $?"
  exit 0
fi
echo "== P11 score start $(date -u)"
python scripts/run_h100.py matched --dtu $DTU --bases none
echo "== matched natives DTU end $(date -u) rc $?"
python scripts/run_h100.py matched_co3d --dtu $DTU --co3d $CO3D --bases none
echo "== matched natives CO3D end $(date -u) rc $?"
while ! grep -q P11_TRAIN_MAIN_DONE /home/logs/h200_p11t.out 2>/dev/null; do sleep 60; done
python scripts/run_h100.py matched --dtu $DTU --bases vggt,pi3x
echo "== matched heads DTU end $(date -u) rc $?"
python scripts/run_h100.py matched_co3d --dtu $DTU --co3d $CO3D --bases vggt,pi3x
echo "== matched heads CO3D end $(date -u) rc $?"
python scripts/run_h100.py realocc --dtu $DTU --bases vggt,pi3x
echo "== realocc end $(date -u) rc $?"
echo "== P11_MAIN_DONE $(date -u)"
while ! grep -q P11_TRAIN_DONE /home/logs/h200_p11t.out 2>/dev/null; do sleep 60; done
python scripts/run_h100.py matched --dtu $DTU --bases vggt,pi3x,pi3,da3
python scripts/run_h100.py matched_co3d --dtu $DTU --co3d $CO3D --bases vggt,pi3x,pi3,da3
echo "== P11_DONE $(date -u)"
