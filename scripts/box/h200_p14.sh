#!/bin/bash
# Seventh H200 pass (2026-11-09, pre-registered Gate 12 in EXPERIMENTS.md): real occlusion on three real datasets.
#   bash scripts/box/h200_p14.sh launch     # one tmux session per job, logs under /home/logs/p14_*.out
# holes        validity check of the observed test (no model): DTU benchmark + test scans, ETH3D, CO3Dv2
# dext_a/b     DTU test scans (22), VGGT + Pi3X / Pi3 + DA3
# eth          ETH3D, Pi3 + DA3
# co3d_a/b     CO3Dv2 benchmark scenes (18), VGGT + Pi3X / Pi3 + DA3
# Every job writes its own folder under results/P14-realocc/ (no two jobs append to one CSV).
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
job=${1:-launch}
O=results/P14-realocc
DEXT=1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118
CO3D_SC=apple_110_13051_23361,apple_189_20393_38136,ball_123_14363_28981,ball_375_42693_85518,bench_415_57112_110099,bench_415_57121_110109,hydrant_167_18184_34441,hydrant_411_56064_108483,mouse_107_12753_23606,mouse_377_43416_86289,teddybear_187_20215_38541,teddybear_34_1479_4753,toytrain_240_25394_51994,toytrain_399_51323_100753,toytruck_190_20494_39385,toytruck_346_36113_66551,vase_374_41862_83720,vase_380_44863_89631
A=vggt,vggt_reproj,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself
B=pi3,pi3_reproj,pi3_oiavisself,da3,da3_reproj,da3_oiavisself
case $job in
launch)
  mkdir -p /home/logs
  for j in holes dext_a dext_b eth co3d_a co3d_b; do
    tmux new -d -s p14_$j "bash scripts/box/h200_p14.sh $j > /home/logs/p14_$j.out 2>&1"
  done; tmux ls ;;
holes)
  python scripts/realocc_eval.py --holes --dtu $DTU --out $O/holes
  python scripts/realocc_eval.py --holes --dtu $DTU --scans $DEXT --out $O/holes_dext
  python scripts/realocc_eval.py --holes --co3d-format $ETH3D --out $O/holes
  python scripts/realocc_eval.py --holes --co3d-format $CO3D --scans $CO3D_SC --out $O/holes
  echo "== P14_holes_DONE $(date -u)" ;;
dext_a) python scripts/realocc_eval.py --dtu $DTU --scans $DEXT --models $A --out $O/dext_a; echo "== P14_dext_a_DONE $(date -u)" ;;
dext_b) python scripts/realocc_eval.py --dtu $DTU --scans $DEXT --models $B --out $O/dext_b; echo "== P14_dext_b_DONE $(date -u)" ;;
eth)    python scripts/realocc_eval.py --co3d-format $ETH3D --models $B --out $O/eth; echo "== P14_eth_DONE $(date -u)" ;;
co3d_a) python scripts/realocc_eval.py --co3d-format $CO3D --scans $CO3D_SC --models $A --out $O/co3d_a; echo "== P14_co3d_a_DONE $(date -u)" ;;
co3d_b) python scripts/realocc_eval.py --co3d-format $CO3D --scans $CO3D_SC --models $B --out $O/co3d_b; echo "== P14_co3d_b_DONE $(date -u)" ;;
esac
