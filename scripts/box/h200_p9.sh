#!/bin/bash
# Third H200 pass (2026-11-07 evening): fill sensitivity of the invented surface (P9-refill) and the fused
# support-times-confidence score on the vis protocol, the dumps and the fused point cloud.
#   tmux new -d -s p9 "bash scripts/box/h200_p9.sh > /home/logs/h200_p9.out 2>&1"
source ${ROOT:-/home}/env.sh
export HF_HUB_OFFLINE=0
export ETH3D=${ETH3D:-$ETH}
cd ${ROOT:-/home}/hallu
echo "== refill start $(date -u)"
python scripts/refill_eval.py --dtu $DTU --co3d $CO3D --models vggt,pi3x,pi3,da3
echo "== refill end $(date -u) rc $?"
echo "== fused vis start $(date -u)"
python - <<'EOF'
import os, sys
sys.path.insert(0, "scripts")
import hallu3d as H, co3d, run_h100 as R
os.environ["OIA_CKPT_DIR"] = "checkpoints"
dtu = H.find_dtu_root((os.environ["DTU"],)); c3 = co3d.find_co3d_root((os.environ["CO3D"],))
eth = co3d.find_co3d_root((os.environ["ETH3D"],)); imc = co3d.find_co3d_root(("/home/data/imc",))
models = [f"{b}_oiavisselffuse" for b in ("vggt", "pi3x", "pi3", "da3")] + [f"{b}_oiavisfuse" for b in ("vggt", "pi3x")]
jobs = [("P4-vis", dtu, {**R.VIS_CFG, "dump_dir": "results/P6-dump/dtu"}),
        ("P4-vis-co3d", c3, {**R.VIS_CFG, "dataset": "co3d", "scans": co3d.list_co3d_scenes(c3), "dump_dir": "results/P6-dump/co3d"}),
        ("P4-vis-eth3d", eth, {**R.VIS_CFG, "dataset": "co3d", "scans": [s for s in R.ETH3D_SCENES if s in co3d.list_co3d_scenes(eth)], "view_mode": "adjacent", "dump_dir": "results/P6-dump/eth3d"}),
        ("P7-vis-imc", imc, {**R.VIS_CFG, "dataset": "co3d", "scans": [s for s in R.IMC_SCENES if s in co3d.list_co3d_scenes(imc)], "view_mode": "adjacent", "dump_dir": "results/P6-dump/imc"})]
for run, root, cfg in jobs:
    for m in models:
        try:
            H.run_vis(root, [m], cfg=cfg, out_csv=f"results/{run}/{run}.csv")
        except Exception as e:
            print("!!", run, m, type(e).__name__, e)
EOF
echo "== fused vis end $(date -u) rc $?"
echo "== fused fuse start $(date -u)"
python scripts/fuse_eval.py --dtu $DTU --models vggt_oiavisselffuse,pi3x_oiavisselffuse
python scripts/fuse_eval.py --co3d-format $ETH3D --scenes courtyard,electro,facade,kicker,meadow,relief,relief_2,terrace,terrains --models vggt_oiavisselffuse,pi3x_oiavisselffuse
echo "== fused fuse end $(date -u) rc $?"
python scripts/analyze_dumps.py results/P6-dump/dtu results/P6-dump/co3d results/P6-dump/eth3d results/P6-dump/imc --tau-from results/P6-dump/dtu > /home/logs/analyze2.out 2>&1
echo "== analyze end $(date -u) rc $?"
python scripts/run_h100.py stats --out results --ckpt checkpoints --bases vggt,pi3x,pi3,da3 > /dev/null 2>&1
echo "== P9_DONE $(date -u)"
