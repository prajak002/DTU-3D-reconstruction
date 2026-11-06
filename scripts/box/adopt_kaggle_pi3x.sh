#!/bin/bash
# Adopt the Kaggle Pi3X run (kernel hallu3d-p3-pi3x, commit 74a13c7, heads at step 1500) on the box, so that the
# resumable P3 stages skip what Kaggle already produced. oia_head.py and hallu3d.py are unchanged between 74a13c7
# and the box commit, so the rows are produced by identical code (T4 fp16 instead of L4; noted in the log).
#   adopt_kaggle_pi3x.sh STAGING checkpoints   # Kaggle heads replace the box's step-750 Pi3X checkpoints (kept as *_box750.pt)
#   adopt_kaggle_pi3x.sh STAGING rows          # every stage CSV except the ablation
#   adopt_kaggle_pi3x.sh STAGING abl           # ablation CSV: drops the box's step-750 pi3x_oiapatch rows, adds Kaggle's
set -e
source /home/ubuntu/env.sh
K=$1; what=$2; cd ~/hallu
case $what in
checkpoints)
  for n in pi3x_oia pi3x_oiapatch; do
    [ -f checkpoints/$n.pt ] && [ ! -f checkpoints/${n}_box750.pt ] && mv checkpoints/$n.pt checkpoints/${n}_box750.pt
    [ -f checkpoints/${n}_best.pt ] && [ ! -f checkpoints/${n}_best_box750.pt ] && mv checkpoints/${n}_best.pt checkpoints/${n}_best_box750.pt
    cp -p $K/checkpoints/$n.pt $K/checkpoints/${n}_best.pt $K/checkpoints/${n}_hist.json checkpoints/
  done
  python - <<'PY'
import torch, glob
for p in sorted(glob.glob("checkpoints/pi3x_*.pt")):
    print(p, "step", torch.load(p, map_location="cpu", weights_only=False).get("step"))
PY
  ;;
rows)
  for r in P3-native P3-pv P3-reproj P3-oia-dtu P3-oia-co3d P3-oia-stress_inpaint10 P3-oia-stress_blur5 P3-oia-stress_blur3 P3-oia-stress_adjacent10 P3-oia-stress_distractor10; do
    python scripts/merge_rows.py $K/results/$r/$r.csv results/$r/$r.csv
    mkdir -p results/$r/ex_$r; cp -pn $K/results/$r/ex_$r/*.png results/$r/ex_$r/ 2>/dev/null || true
    cp -pn $K/results/$r/$r.log results/$r/kaggle_$r.log
  done
  ;;
abl)
  python scripts/merge_rows.py $K/results/P3-oia-abl/P3-oia-abl.csv results/P3-oia-abl/P3-oia-abl.csv --drop-models pi3x_oiapatch
  cp -pn $K/results/P3-oia-abl/ex_P3-oia-abl/*.png results/P3-oia-abl/ex_P3-oia-abl/ 2>/dev/null || true
  cp -pn $K/results/P3-oia-abl/P3-oia-abl.log results/P3-oia-abl/kaggle_P3-oia-abl.log
  ;;
esac
