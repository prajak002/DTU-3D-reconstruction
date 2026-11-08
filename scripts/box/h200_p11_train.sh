#!/bin/bash
# Matched heads in parallel (the mask construction is CPU work, so four trainings share the card well):
#   tmux new -d -s p11t "bash scripts/box/h200_p11_train.sh > /home/logs/h200_p11t.out 2>&1"
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
cd ${ROOT:-/home}/hallu
train() {  # base, label source (match | matchself)
  python scripts/oia_head.py train --base $1 --variant oia$2,oia$2patch,oia$2lin --dtu $DTU --out checkpoints --steps 1500 --seed 0 \
    > /home/logs/train_$1_$2.out 2>&1
  echo "== $1 $2 done $(date -u) rc $?"
}
echo "== P11 train start $(date -u)"
for b in vggt pi3x; do for l in match matchself; do train $b $l & done; done
wait
echo "== P11_TRAIN_MAIN_DONE $(date -u)"
for b in pi3 da3; do for l in match matchself; do train $b $l & done; done
wait
echo "== P11_TRAIN_DONE $(date -u)"
