#!/bin/bash
# Everything that is still open for the paper, on one H200 (2026-11-07), in order of value. Every stage resumes by
# row or by checkpoint step, so the script can be relaunched at any time:
#   tmux new -d -s h200 "bash scripts/box/h200_all.sh > ~/logs/h200_all.out 2>&1"
# Stage markers: grep "^==" ~/logs/h200_all.out
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0                  # DA3 and MapAnything weights are not in the offline cache
export ETH3D=${ETH3D:-$ETH}
mkdir -p ~/logs
COMMON="--dtu $DTU --co3d $CO3D --eth3d $ETH3D --out results --ckpt checkpoints"
ALL="--bases vggt,pi3x,pi3 --ext-bases vggt,pi3x,pi3,da3,mapanything"
stage() {
  echo "== $1 start $(date -u) ($2)"
  python scripts/run_h100.py $1 $COMMON $2
  echo "== $1 end $(date -u) rc $?"
}
# 1. heads that do not exist yet: Pi3X label-free (P5) and probes, the third backbone Pi3 (GT-visibility and label-free)
stage trainvisself "--bases pi3x"
stage trainvisprobe "--bases pi3x"
stage trainvis "--bases pi3"
stage trainvisself "--bases pi3"
# 2. the vis protocol on the three datasets with every detector (hand-crafted controls, DA3 and MapAnything native)
stage vis_eth3d "$ALL"
stage vis "$ALL"
stage vis_co3d "$ALL"
# 3. what a reviewer asks next: label-free adaptation, view counts outside training, per-pixel dumps
stage adapt_co3d "--bases vggt,pi3x"
stage adapt_eth3d "--bases vggt,pi3x"
stage vis_ext "--bases vggt,pi3x"
stage dump "--bases vggt,pi3x"
python scripts/analyze_dumps.py results/P6-dump/dtu results/P6-dump/co3d results/P6-dump/eth3d --tau-from results/P6-dump/dtu
stage vis_on_fill "--bases vggt,pi3x"
# 4. the remaining fill-protocol rows (pending markers of the paper) and the compute table
stage co3d "--bases pi3x"
stage dtuext "--bases vggt,pi3x"
stage eth3d "--bases vggt,pi3x"
stage pvfull "--bases vggt,pi3x"
python scripts/compute_cost.py --dtu $DTU --models vggt,vggt_oiavis,vggt_oiavisself,vggt_oiavispatch,vggt_reproj,vggt_border,vggt_dedge,vggt_pv4,pi3x,pi3x_oiavis,pi3x_oiavisself,pi3x_reproj
# 5. statistics over everything
stage stats "--bases vggt,pi3x,pi3"
echo "== ALL_DONE $(date -u)"
