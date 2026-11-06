#!/bin/bash
# One-shot relaunch of the whole P3 plan on the L4 box. Every stage is resumable (rows already in a CSV are skipped,
# training resumes from its last checkpoint), so this script can be run again after any interruption.
#   tmux new -d -s p3 "~/p3_all.sh > ~/logs/p3_all.out 2>&1"
# The card is shared with another user's jobs (2 to 6 GB, variable), so every stage waits until enough memory is free
# before it starts; the two evaluation pipelines therefore overlap when the card allows it and serialise when it does not.
# Head training ran alone on 2026-11-05 because the trainer (about 7.5 GB resident) had been killed by the kernel's OOM
# killer next to an evaluation stage on a 15 GB host; since then a 12 GB swap file is enabled before launch (the script
# does it), so from 2026-11-06 the VGGT evaluation pipeline starts at once and the Pi3X trainer only waits for GPU memory.
# A stage never starts into insufficient memory: the wait has no practical cap (15 h), the sweep fills any gap.
source /home/ubuntu/env.sh
export PATH=/opt/pytorch/cuda/bin:$PATH CUDA_HOME=/opt/pytorch/cuda TORCH_CUDA_ARCH_LIST="8.9"
mkdir -p ~/logs
sudo -n swapon /swapfile 2>/dev/null; swapon --show | grep -q swapfile || echo "WARNING: swap file not active"
ETH=/home/ubuntu/data/eth3d
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# wait_free GB [max_minutes]: block until that much GPU memory is free; give up waiting after max_minutes (default 900)
wait_free() {
  local need=$1 max=${2:-900} n=0 used free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    free=$(( (TOTAL - used) / 1024 ))
    [ "$free" -ge "$need" ] && return 0
    if [ "$n" -ge "$max" ]; then echo "wait_free: $max min elapsed, ${free} GB free (need $need), proceeding anyway"; return 0; fi
    [ "$n" -eq 0 ] && echo "wait_free: ${free} GB free, need ${need}; waiting since $(date -u)"
    sleep 60; n=$((n + 1))
  done
}
# run NAME GB CMD...: wait for GB free, run CMD, append its output to ~/logs/NAME.log
run() {
  local name=$1 need=$2; shift 2
  wait_free "$need"
  echo "== $name start $(date -u)"
  "$@" 2>&1 | tee -a ~/logs/$name.log
  echo "== $name end $(date -u) rc ${PIPESTATUS[0]}"
}
echo "== start $(date -u), ${TOTAL} MiB card, $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used by others"

# 1. pipeline A, VGGT evaluations, in the background (about 8 GB each; baselines, dtu and stress finished on 2026-11-04,
#    ablate was 162 rows in when the box stopped on 2026-11-05)
(
  run baselines_vggt 9 python scripts/run_h100.py baselines --dtu $DTU --bases vggt
  run dtu_vggt 9 python scripts/run_h100.py dtu --dtu $DTU --bases vggt
  run stress_vggt 9 python scripts/run_h100.py stress --dtu $DTU --bases vggt
  run ablate_vggt 9 python scripts/run_h100.py ablate --dtu $DTU --bases vggt
  run co3d_vggt 9 python scripts/run_h100.py co3d --dtu $DTU --co3d $CO3D --bases vggt
  run dtuext 10 python scripts/run_h100.py dtuext --dtu $DTU --bases vggt
  run eth3d 9 python scripts/run_h100.py eth3d --dtu $DTU --eth3d $ETH --bases vggt
  echo PIPELINE_A_DONE
) > ~/logs/p3a.out 2>&1 &
A_PID=$!

# 2. VGGT heads (complete at step 1500 on 2026-11-04; the stage checks the checkpoint and is a no-op when complete)
run train_vggt 14 python scripts/run_h100.py train --dtu $DTU --bases vggt --skip-gt --steps 1500

# 3. Pi3X heads (9.8 GB of GPU memory measured on 2026-11-05, 7.5 GB of host RAM; checkpoint at step 750 of 1500); a second
#    call resumes if the first was interrupted and is a no-op once the checkpoint is at 1500 steps
run train_pi3x 12 python scripts/run_h100.py train --dtu $DTU --bases pi3x --skip-gt --steps 1500
run train_pi3x 12 python scripts/run_h100.py train --dtu $DTU --bases pi3x --skip-gt --steps 1500

# 4. pipeline B: Pi3X evaluations (about 10 GB), the splatting backbone, the perturbation baseline
run baselines_pi3x 10 python scripts/run_h100.py baselines --dtu $DTU --bases pi3x
run dtu_pi3x 10 python scripts/run_h100.py dtu --dtu $DTU --bases pi3x
run stress_pi3x 10 python scripts/run_h100.py stress --dtu $DTU --bases pi3x
run co3d_pi3x 10 python scripts/run_h100.py co3d --dtu $DTU --co3d $CO3D --bases pi3x
run ablate_pi3x 10 python scripts/run_h100.py ablate --dtu $DTU --bases pi3x
wait_free 8
python ~/ext/verify_anysplat.py anysplat_render 2>&1 | tail -20 | tee -a ~/logs/anysplat_render_check.log
if grep -q "AbsRel" ~/logs/anysplat_render_check.log 2>/dev/null; then GSR="--gs-render"; else GSR=""; fi
run gs 8 python scripts/run_h100.py gs --dtu $DTU --co3d $CO3D $GSR
run pvfull 12 python scripts/run_h100.py pvfull --dtu $DTU
echo PIPELINE_B_DONE
wait $A_PID

# 5. sweep: training once more (a no-op when both checkpoints are at 1500 steps), then every evaluation stage once more,
#    sequentially; a no-op where the rows exist, fills any gap left by an out-of-memory stop while the pipelines overlapped
run sweep_train 12 python scripts/run_h100.py train --dtu $DTU --bases vggt,pi3x --skip-gt --steps 1500
for b in vggt pi3x; do
  run sweep_$b 10 python scripts/run_h100.py baselines --dtu $DTU --bases $b
  run sweep_$b 10 python scripts/run_h100.py dtu --dtu $DTU --bases $b
  run sweep_$b 10 python scripts/run_h100.py stress --dtu $DTU --bases $b
  run sweep_$b 10 python scripts/run_h100.py co3d --dtu $DTU --co3d $CO3D --bases $b
  run sweep_$b 10 python scripts/run_h100.py ablate --dtu $DTU --bases $b
done
run sweep_vggt 10 python scripts/run_h100.py dtuext --dtu $DTU --bases vggt
run sweep_vggt 10 python scripts/run_h100.py eth3d --dtu $DTU --eth3d $ETH --bases vggt

# 6. accounting, statistics, qualitative figure
run compute 8 python scripts/compute_cost.py --dtu $DTU
run stats 1 python scripts/run_h100.py stats --bases vggt,pi3x
run fig_threeway 8 python scripts/make_fig_threeway.py --dtu $DTU --eth3d $ETH
echo "== ALL_DONE $(date -u)"
