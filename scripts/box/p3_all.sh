#!/bin/bash
# One-shot relaunch of the whole P3 plan on the L4 box. Every stage is resumable (rows already in a CSV are skipped,
# training resumes from its last checkpoint), so this script can be run again after any interruption.
#   tmux new -d -s p3 "~/p3_all.sh > ~/logs/p3_all.out 2>&1"
source /home/ubuntu/env.sh
export PATH=/opt/pytorch/cuda/bin:$PATH CUDA_HOME=/opt/pytorch/cuda TORCH_CUDA_ARCH_LIST="8.9"
mkdir -p ~/logs
ETH=/home/ubuntu/data/eth3d
echo "== start $(date -u)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 1. finish the VGGT heads (resumes from the last checkpoint; no-op when complete)
python scripts/run_h100.py train --dtu $DTU --bases vggt --skip-gt --steps 1500 2>&1 | tee -a ~/logs/train_vggt.log

# 2. VGGT evaluations in the background (about 8 GB) ...
(
  python scripts/run_h100.py baselines --dtu $DTU --bases vggt 2>&1 | tee -a ~/logs/baselines_vggt.log
  python scripts/run_h100.py dtu --dtu $DTU --bases vggt 2>&1 | tee -a ~/logs/dtu_vggt.log
  python scripts/run_h100.py stress --dtu $DTU --bases vggt 2>&1 | tee -a ~/logs/stress_vggt.log
  python scripts/run_h100.py ablate --dtu $DTU --bases vggt 2>&1 | tee -a ~/logs/ablate_vggt.log
  python scripts/run_h100.py co3d --dtu $DTU --co3d $CO3D --bases vggt 2>&1 | tee -a ~/logs/co3d_vggt.log
  python scripts/run_h100.py dtuext --dtu $DTU --bases vggt 2>&1 | tee -a ~/logs/dtuext.log
  python scripts/run_h100.py eth3d --dtu $DTU --eth3d $ETH --bases vggt 2>&1 | tee -a ~/logs/eth3d.log
  echo PIPELINE_A_DONE
) > ~/logs/p3a.out 2>&1 &
A_PID=$!

# 3. ... while the Pi3X heads train and evaluate (about 10 GB), then the splatting backbone and the perturbation baseline
python scripts/run_h100.py train --dtu $DTU --bases pi3x --skip-gt --steps 1500 2>&1 | tee -a ~/logs/train_pi3x.log
python scripts/run_h100.py baselines --dtu $DTU --bases pi3x 2>&1 | tee -a ~/logs/baselines_pi3x.log
python scripts/run_h100.py dtu --dtu $DTU --bases pi3x 2>&1 | tee -a ~/logs/dtu_pi3x.log
python scripts/run_h100.py stress --dtu $DTU --bases pi3x 2>&1 | tee -a ~/logs/stress_pi3x.log
python scripts/run_h100.py co3d --dtu $DTU --co3d $CO3D --bases pi3x 2>&1 | tee -a ~/logs/co3d_pi3x.log
python scripts/run_h100.py ablate --dtu $DTU --bases pi3x 2>&1 | tee -a ~/logs/ablate_pi3x.log
python ~/ext/verify_anysplat.py anysplat_render 2>&1 | tail -20 | tee -a ~/logs/anysplat_render_check.log
if grep -q "AbsRel" ~/logs/anysplat_render_check.log 2>/dev/null; then GSR="--gs-render"; else GSR=""; fi
python scripts/run_h100.py gs --dtu $DTU --co3d $CO3D $GSR 2>&1 | tee -a ~/logs/gs.log
python scripts/run_h100.py pvfull --dtu $DTU 2>&1 | tee -a ~/logs/pvfull.log
echo PIPELINE_B_DONE
wait $A_PID

# 4. accounting, statistics, qualitative figure
python scripts/compute_cost.py --dtu $DTU 2>&1 | tee -a ~/logs/compute.log
python scripts/run_h100.py stats --bases vggt,pi3x 2>&1 | tee -a ~/logs/stats.log
python scripts/make_fig_threeway.py --dtu $DTU --eth3d $ETH 2>&1 | tee -a ~/logs/fig_threeway.log
echo "== ALL_DONE $(date -u)"
