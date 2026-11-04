export PATH=/opt/pytorch/bin:$PATH
export PYTHONPATH=/home/ubuntu/Pi3:/home/ubuntu/hallu/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/ubuntu/.hf_token)
export PYTHONUNBUFFERED=1
export DTU=/home/ubuntu/data/dtu
export CO3D=/home/ubuntu/data/co3d
cd /home/ubuntu/hallu
