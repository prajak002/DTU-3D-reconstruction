#!/bin/bash
# Ninth H200 pass (2026-11-11, pre-registered Gates 15 to 17 in LOG.md before any run).
#   bash scripts/box/h200_p16.sh launch     # one tmux session per job, logs under /home/logs/p16_*.out
# nbv_a/b     Gate 15: next-view selection from the head vs native confidence, farthest view, random, oracle (DTU benchmark)
# side_a/b    Gate 16: real occlusion with one-sided captures (views from one end of the DTU rig)
# co3d_A/B    Gate 17: real occlusion on CO3Dv2 with the label-free heads adapted on the other fold (no labels)
# Every job writes its own folder under results/P16-*/ (no two jobs append to one CSV).
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
job=${1:-launch}
A=vggt,vggt_reproj,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself
B=pi3,pi3_reproj,pi3_oiavisself,da3,da3_reproj,da3_oiavisself
FOLD_A=apple_110_13051_23361,ball_123_14363_28981,bench_415_57112_110099,hydrant_167_18184_34441,mouse_107_12753_23606,teddybear_187_20215_38541,toytrain_240_25394_51994,toytruck_190_20494_39385,vase_374_41862_83720
FOLD_B=apple_189_20393_38136,ball_375_42693_85518,bench_415_57121_110109,hydrant_411_56064_108483,mouse_377_43416_86289,teddybear_34_1479_4753,toytrain_399_51323_100753,toytruck_346_36113_66551,vase_380_44863_89631
case $job in
launch)
  mkdir -p /home/logs
  for j in nbv_a nbv_b side_a side_b co3d_A co3d_B; do
    tmux new -d -s p16_$j "bash scripts/box/h200_p16.sh $j > /home/logs/p16_$j.out 2>&1"
  done; tmux ls ;;
nbv_a)  python scripts/nbv_eval.py --dtu $DTU --pairs vggt:vggt_oiavisself,pi3x:pi3x_oiavisself --out results/P16-nbv/a; echo "== P16_nbv_a_DONE $(date -u)" ;;
nbv_b)  python scripts/nbv_eval.py --dtu $DTU --pairs pi3:pi3_oiavisself,da3:da3_oiavisself --out results/P16-nbv/b; echo "== P16_nbv_b_DONE $(date -u)" ;;
side_a) python scripts/realocc_eval.py --dtu $DTU --view-mode oneside --views 3,5 --models $A --out results/P16-oneside/a; echo "== P16_side_a_DONE $(date -u)" ;;
side_b) python scripts/realocc_eval.py --dtu $DTU --view-mode oneside --views 3,5 --models $B --out results/P16-oneside/b; echo "== P16_side_b_DONE $(date -u)" ;;
co3d_A) python scripts/realocc_eval.py --co3d-format $CO3D --scans $FOLD_A --align points --models vggt,vggt_oiavisself_xco3dB,pi3x,pi3x_oiavisself_xco3dB --out results/P16-co3dadapt/A; echo "== P16_co3d_A_DONE $(date -u)" ;;
co3d_B) python scripts/realocc_eval.py --co3d-format $CO3D --scans $FOLD_B --align points --models vggt,vggt_oiavisself_xco3dA,pi3x,pi3x_oiavisself_xco3dA --out results/P16-co3dadapt/B; echo "== P16_co3d_B_DONE $(date -u)" ;;
esac
