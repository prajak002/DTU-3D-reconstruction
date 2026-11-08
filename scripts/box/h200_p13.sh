#!/bin/bash
# Sixth H200 pass (2026-11-08, pre-registered Gates 7 for Pi3 / DA3, 9, 10, 11 in EXPERIMENTS.md). Jobs share the card:
#   bash scripts/box/h200_p13.sh launch     # starts one tmux session per job, logs under /home/logs/p13_*.out
# ev      Gate 9: fine-tune the VGGT matched triples with one-view evidence oversampled (two runs), then score them
# m_pi3 / m_da3 / m_co3d   Gate 7: Pi3 and DA3 matched heads and controls on DTU and CO3Dv2
# occ     Gate 10: real occlusion, Pi3 and DA3 on DTU, then VGGT and Pi3X on ETH3D
# fuse    Gate 11: fused reconstruction with the combined detector, DTU and ETH3D
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
job=${1:-launch}
CK=$OIA_CKPT_DIR
case $job in
launch)
  for j in ev m_pi3 m_da3 m_co3d occ fuse; do
    tmux new -d -s p13_$j "bash scripts/box/h200_p13.sh $j > /home/logs/p13_$j.out 2>&1"
  done; tmux ls ;;
ev)
  for p in "oiamatch oiamatchev" "oiamatchpatch oiamatchevpatch" "oiamatchlin oiamatchevlin" \
           "oiamatchself oiamatchselfev" "oiamatchselfpatch oiamatchselfevpatch" "oiamatchselflin oiamatchselfevlin"; do
    set -- $p; [ -f $CK/vggt_$2.pt ] || cp $CK/vggt_$1.pt $CK/vggt_$2.pt
  done
  python scripts/oia_head.py train --base vggt --variant oiamatchev,oiamatchevpatch,oiamatchevlin --dtu $DTU --out $CK \
    --steps 2500 --seed 0 --resume > /home/logs/p13_ev_gt.out 2>&1 &
  python scripts/oia_head.py train --base vggt --variant oiamatchselfev,oiamatchselfevpatch,oiamatchselfevlin --dtu $DTU \
    --out $CK --steps 2500 --seed 0 --resume > /home/logs/p13_ev_self.out 2>&1 &
  wait
  echo "== P13_EV_TRAIN_DONE $(date -u)"
  python scripts/run_h100.py matched --dtu $DTU --bases vggt \
    --only vggt_oiamatchev,vggt_oiamatchevpatch,vggt_oiamatchselfev,vggt_oiamatchselfevpatch
  echo "== P13_EV_DONE $(date -u)" ;;
m_pi3|m_da3)
  b=${job#m_}
  python scripts/run_h100.py matched --dtu $DTU --bases $b \
    --only ${b}_oiamatch,${b}_oiamatchpatch,${b}_oiamatchself,${b}_oiamatchselfpatch
  echo "== P13_${job}_DONE $(date -u)" ;;
m_co3d)
  python scripts/run_h100.py matched_co3d --dtu $DTU --co3d $CO3D --bases pi3,da3 \
    --only pi3_oiamatch,pi3_oiamatchpatch,pi3_oiamatchself,pi3_oiamatchselfpatch,da3_oiamatch,da3_oiamatchpatch,da3_oiamatchself,da3_oiamatchselfpatch
  echo "== P13_m_co3d_DONE $(date -u)" ;;
occ)
  python scripts/realocc_eval.py --dtu $DTU --models pi3,pi3_reproj,pi3_oiavisself,da3,da3_reproj,da3_oiavisself \
    --out results/P12-realocc
  python scripts/realocc_eval.py --co3d-format $ETH3D --models vggt,vggt_reproj,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself \
    --out results/P12-realocc
  echo "== P13_occ_DONE $(date -u)" ;;
fuse)
  for m in vggt_oiavisself pi3x_oiavisself pi3_oiavisself da3_oiavisself; do
    python scripts/fuse_eval.py --dtu $DTU --models $m --combine --tag dtu --out results/P13-downstream
    python scripts/fuse_eval.py --co3d-format $ETH3D --models $m --combine --tag eth3d --out results/P13-downstream
  done
  echo "== P13_fuse_DONE $(date -u)" ;;
esac
