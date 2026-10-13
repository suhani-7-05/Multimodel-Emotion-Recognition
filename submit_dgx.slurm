#!/bin/bash
#SBATCH --job-name=prj4_meld
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err

echo "Running on host: $(hostname)"
echo "Assigned GPU info:"
nvidia-smi

# Load environment modules if your cluster uses environment modules (adjust version if needed):
# module load cuda/12.1
# module load python/3.10

# Activate conda environment (adjust path if needed):
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate prj4

DATA_PATH="/workspace/data"

python3 new.py \
    --data_dir "$DATA_PATH" \
    --batch_size 64 \
    --epochs 40 \
    --lr 2e-5 \
    --save_path "best3.pt"
