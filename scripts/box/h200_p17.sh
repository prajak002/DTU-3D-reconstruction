#!/bin/bash
# Tenth H200 pass (2026-11-11, pre-registered Gate 18 in LOG.md before any run): real occlusion with hole-robust
# labels (realocc_eval.py --label-mode zagree) on ETH3D, the DTU benchmark and test scans and CO3Dv2.
#   bash scripts/box/h200_p17.sh eth3d      # download and preprocess ETH3D (12 scenes of Gate 10), reorder by co-visibility
#   bash scripts/box/h200_p17.sh launch     # one tmux session per job, logs under /home/logs/p17_*.out
# Alignment as registered for each dataset: camera centres on DTU and ETH3D, input pixels on CO3Dv2 (Gate 12 amendment).
# Every job writes its own folder under results/P17-zagree/ (no two jobs append to one CSV).
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
job=${1:-launch}
O=results/P17-zagree
DEXT=1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118
CO3D_SC=apple_110_13051_23361,apple_189_20393_38136,ball_123_14363_28981,ball_375_42693_85518,bench_415_57112_110099,bench_415_57121_110109,hydrant_167_18184_34441,hydrant_411_56064_108483,mouse_107_12753_23606,mouse_377_43416_86289,teddybear_187_20215_38541,teddybear_34_1479_4753,toytrain_240_25394_51994,toytrain_399_51323_100753,toytruck_190_20494_39385,toytruck_346_36113_66551,vase_374_41862_83720,vase_380_44863_89631
ETH_SC=courtyard,delivery_area,electro,facade,kicker,meadow,pipes,playground,relief,relief_2,terrace,terrains   # the 12 scenes of Gate 10
A=vggt,vggt_reproj,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself
B=pi3,pi3_reproj,pi3_oiavisself,da3,da3_reproj,da3_oiavisself
Z="--label-mode zagree"
case $job in
eth3d)
  RAW=/home/data/eth3d_raw; mkdir -p $RAW /home/data/eth3d; cd $RAW
  which 7z > /dev/null || (apt-get -qq update > /dev/null; apt-get -qq install -y p7zip-full > /dev/null)
  for f in multi_view_training_dslr_undistorted multi_view_training_dslr_jpg; do
    (curl -sS -L -o $f.7z https://www.eth3d.net/data/$f.7z && 7z x -y $f.7z > /dev/null && rm -f $f.7z; echo "== $f") &
  done
  for s in ${ETH_SC//,/ }; do
    (curl -sS -L -o ${s}_depth.7z https://www.eth3d.net/data/${s}_dslr_depth.7z && 7z x -y ${s}_depth.7z > /dev/null && rm -f ${s}_depth.7z) &
  done; wait
  cd /home/hallu
  for s in ${ETH_SC//,/ }; do
    (python scripts/eth3d.py $RAW /home/data/eth3d --scenes $s 2>&1 | grep -v pynvml | tail -2) &
  done; wait
  python scripts/eth3d.py /home/data/eth3d --reorder
  ls /home/data/eth3d | wc -l; rm -rf $RAW; df -h /home | tail -1
  echo "== P17_eth3d_DONE $(date -u)" ;;
launch)
  mkdir -p /home/logs
  for j in eth_a eth_b dtu_a dtu_b dext_a dext_b co3d_a co3d_b; do
    tmux new -d -s p17_$j "bash scripts/box/h200_p17.sh $j > /home/logs/p17_$j.out 2>&1"
  done; tmux ls ;;
eth_a)  python scripts/realocc_eval.py --co3d-format $ETH3D --scans $ETH_SC --models $A,vggt_oiavisself_xeth3d,pi3x_oiavisself_xeth3d $Z --out $O/eth_a; echo "== P17_eth_a_DONE $(date -u)" ;;
eth_b)  python scripts/realocc_eval.py --co3d-format $ETH3D --scans $ETH_SC --models $B $Z --out $O/eth_b; echo "== P17_eth_b_DONE $(date -u)" ;;
dtu_a)  python scripts/realocc_eval.py --dtu $DTU --models $A $Z --out $O/dtu_a; echo "== P17_dtu_a_DONE $(date -u)" ;;
dtu_b)  python scripts/realocc_eval.py --dtu $DTU --models $B $Z --out $O/dtu_b; echo "== P17_dtu_b_DONE $(date -u)" ;;
dext_a) python scripts/realocc_eval.py --dtu $DTU --scans $DEXT --models $A $Z --out $O/dext_a; echo "== P17_dext_a_DONE $(date -u)" ;;
dext_b) python scripts/realocc_eval.py --dtu $DTU --scans $DEXT --models $B $Z --out $O/dext_b; echo "== P17_dext_b_DONE $(date -u)" ;;
co3d_a) python scripts/realocc_eval.py --co3d-format $CO3D --scans $CO3D_SC --models $A --align points $Z --out $O/co3d_a; echo "== P17_co3d_a_DONE $(date -u)" ;;
co3d_b) python scripts/realocc_eval.py --co3d-format $CO3D --scans $CO3D_SC --models $B --align points $Z --out $O/co3d_b; echo "== P17_co3d_b_DONE $(date -u)" ;;
esac
