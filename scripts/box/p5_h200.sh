#!/bin/bash
# H200 (JarvisLabs jl-565870), 2026-11-07: the P5 evaluation stages, taken over from the L4 box. The remaining Pi3X
# CO3Dv2 control rows first (needs only the P3 heads), then, once the probe heads trained on the L4 have been copied
# in (marker /home/PROBES_SYNCED), every vis stage and the stats. Every stage resumes by row, so relaunch at will:
#   tmux new -d -s p5 "/home/p5_h200.sh > /home/logs/p5_h200.out 2>&1"
source /home/env.sh
cd /home/hallu
COMMON="--dtu $DTU --co3d $CO3D --eth3d $ETH --out results --ckpt checkpoints --bases vggt,pi3x --ext-bases vggt,pi3x,pi3"
stage() {
  echo "== $1 start $(date -u) ($2)"
  python scripts/run_h100.py $1 $COMMON $2 2>&1 | tee -a /home/logs/$1.log
  echo "== $1 end $(date -u) rc ${PIPESTATUS[0]}"
}
stage co3d "--bases pi3x"
n=0; while [ ! -f /home/PROBES_SYNCED ]; do [ $n -eq 0 ] && echo "== waiting for /home/PROBES_SYNCED $(date -u)"; sleep 60; n=$((n+1)); done
stage vis_co3d ""
stage vis_eth3d ""
stage vis ""
stage stats ""
echo "== ALL_DONE $(date -u)"
