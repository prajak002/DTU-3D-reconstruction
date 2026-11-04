# Scripts for the GPU box

Driver scripts used on the AWS L4 instance (one `g6.xlarge`, Ubuntu 24.04, PyTorch venv at `/opt/pytorch`). Paths are the box's.

- `env.sh`: environment (PATH, PYTHONPATH with the Pi3 clone and `~/hallu/scripts`, HF token from `~/.hf_token`, `$DTU`, `$CO3D`). `source ~/env.sh` first.
- `p3_all.sh`: the whole P3 plan in one resumable run. Finishes the VGGT heads (resume from the last checkpoint), then runs the VGGT evaluations in the background while the Pi3X heads train and evaluate, then the Gaussian-splatting backbone, the fuller perturbation baseline, the compute accounting, the statistics and the qualitative figure. Launch with `tmux new -d -s p3 "~/p3_all.sh > ~/logs/p3_all.out 2>&1"`; re-run after any interruption, finished rows are skipped.
- `eth3d_prep.sh`: ETH3D download, depth undistortion and preprocessing scene by scene, deleting the raw data as it goes (already done once; the preprocessed scenes live in `~/data/eth3d`).

Results land in `~/hallu/results/P3-*` and are copied into this repository with scp; nothing is committed from the box.
