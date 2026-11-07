#!/bin/bash
# Second H200 pass (the six fixes, 2026-11-08): a head on DA3, cross-model disagreement on the filled patches,
# the fused point-cloud metric at matched coverage, the qualitative panels from the dumps. Every stage resumes.
#   tmux new -d -s p8 "bash scripts/box/h200_p8.sh > /home/logs/h200_p8.out 2>&1"
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
export ETH3D=${ETH3D:-$ETH}
mkdir -p ~/logs
COMMON="--dtu $DTU --co3d $CO3D --eth3d $ETH3D --imc ${IMC:-/home/data/imc} --out results --ckpt checkpoints"
stage() {
  echo "== $1 start $(date -u) ($2)"
  python scripts/run_h100.py $1 $COMMON $2
  echo "== $1 end $(date -u) rc $?"
}
# 0. the DA3 hook on the real model: shapes, a reprojection self-consistency check of its cameras
echo "== da3 check start $(date -u)"
python - <<'EOF'
import sys, numpy as np, torch
sys.path.insert(0, "scripts")
import hallu3d as H, oia_head as O
w = H.MODEL_REGISTRY["da3"]()
root = H.find_dtu_root((__import__("os").environ["DTU"],))
d = H.load_scan(root, 3, H.vis_views(5, 0), 3)
x = w._prep(d["imgs"], (420, 518))
dep, conf, F, K, E = O.da3_forward_feats(w, x, w.dtype, "both")
print("depth", tuple(dep.shape), "conf", tuple(conf.shape), "agg", [tuple(f.shape) for f in F["agg"]], "patch", [tuple(f.shape) for f in F["patch"]])
print("K[0]", K[0].cpu().numpy().round(1).tolist(), "E[0] row0", E[0, 0].cpu().numpy().round(3).tolist())
dn = dep.float().cpu().numpy(); Kn = K.float().cpu().numpy().astype(np.float64); En = E.float().cpu().numpy().astype(np.float64)
sc = H.reproj_support_score(dn, Kn, En)
print("reprojection self-consistency of DA3's own prediction (mean over pixels, should be high):", float(sc.mean()))
EOF
echo "== da3 check end $(date -u) rc $?"
# 1. heads on DA3 (GT-visibility and label-free, with controls and probe), then every vis stage for da3
stage trainvis "--bases da3"
stage trainvisself "--bases da3"
stage vis "--bases da3 --ext-bases da3"
stage vis_co3d "--bases da3 --ext-bases da3"
stage vis_eth3d "--bases da3 --ext-bases da3"
stage vis_imc "--bases da3 --ext-bases da3"
# 2. cross-model disagreement on the filled patches and on real single-view geometry
stage xmodel "--bases vggt,pi3x"
# 3. the fused point cloud at matched coverage, DTU and ETH3D
echo "== fuse start $(date -u)"
python scripts/fuse_eval.py --dtu $DTU --models vggt,vggt_reproj,vggt_oiavispatch,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself
python scripts/fuse_eval.py --co3d-format $ETH3D --scenes courtyard,electro,facade,kicker,meadow,relief,relief_2,terrace,terrains --models vggt,vggt_reproj,vggt_oiavispatch,vggt_oiavisself,pi3x,pi3x_reproj,pi3x_oiavisself
echo "== fuse end $(date -u) rc $?"
# 4. qualitative panels from the dumps (figures for the paper, the page and the video)
echo "== figures start $(date -u)"
for spec in "dtu 3 5 0 vggt 0,2" "dtu 17 5 0 vggt 1,3" "co3d hydrant_167_18184_34441 5 0 vggt 0,2" "eth3d facade 5 0 vggt 0,1" "eth3d relief 5 0 pi3x 0,1" "imc sagrada_familia 5 0 vggt 0,1" "imc british_museum 5 0 pi3x 0,2" "imc lincoln_memorial_statue 5 0 vggt 0,1"; do
  set -- $spec
  python scripts/make_fig_vis.py results/P6-dump/$1 --scene $2 --views $3 --seed $4 --base $5 --rows $6 --detectors $5,$5_reproj,$5_oiavispatch,$5_oiavisself --out paper/figures/qual_$1_$2.pdf || true
done
echo "== figures end $(date -u)"
stage stats "--bases vggt,pi3x,pi3,da3"
echo "== P8_DONE $(date -u)"
