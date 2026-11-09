#!/bin/bash
# Eighth H200 pass (2026-11-09, pre-registered Gates 13 and 14 in EXPERIMENTS.md).
#   bash scripts/box/h200_p15.sh launch
# cap_vggt / cap_pi3x   Gate 13: train the transformer label-free head, then score linear / conv / transformer / control
# down3_<base>          Gate 14: fused reconstruction at 3 views on DTU benchmark, DTU test and ETH3D
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
job=${1:-launch}
CK=$OIA_CKPT_DIR
DEXT=1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118
ETH_SC=courtyard,delivery_area,electro,facade,kicker,meadow,pipes,playground,relief,relief_2,terrace,terrains
case $job in
launch)
  for j in cap_vggt cap_pi3x down3_vggt down3_pi3x down3_pi3 down3_da3; do
    tmux new -d -s p15_$j "bash scripts/box/h200_p15.sh $j > /home/logs/p15_$j.out 2>&1"
  done; tmux ls ;;
cap_vggt|cap_pi3x)
  b=${job#cap_}
  python scripts/oia_head.py train --base $b --variant oiavisselfattn --dtu $DTU --out $CK --steps 1500 --seed 0 --resume
  echo "== P15_${b}_TRAIN_DONE $(date -u)"
  M=${b}_oiavisselflin,${b}_oiavisself,${b}_oiavisselfattn,${b}_oiavisselfpatch,$b
  python scripts/realocc_eval.py --dtu $DTU --models $M --out results/P15-capacity/dtu_$b
  python scripts/realocc_eval.py --co3d-format $ETH3D --scans $ETH_SC --models $M --out results/P15-capacity/eth_$b
  echo "== P15_${job}_DONE $(date -u)" ;;
down3_*)
  b=${job#down3_}
  M=$b,${b}_reproj,${b}_oiavisself
  python scripts/fuse_eval.py --dtu $DTU --models $M --views 3 --tag dtu --out results/P15-down3/$b
  python scripts/fuse_eval.py --dtu $DTU --scenes $DEXT --models $M --views 3 --tag dtuext --out results/P15-down3/$b
  python scripts/fuse_eval.py --co3d-format $ETH3D --scenes $ETH_SC --models $M --views 3 --tag eth3d --out results/P15-down3/$b
  echo "== P15_${job}_DONE $(date -u)" ;;
esac
