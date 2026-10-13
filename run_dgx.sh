#!/bin/bash
# ==========================================================
# PRJ4: DGX Training Runner Script
# ==========================================================

# Choose GPU (e.g., 0, 1, or leave unset to use all available)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Path to MELD data directory (defaults to /workspace/data, or pass as $1)
DATA_PATH="${1:-/workspace/data}"

# Timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_${TIMESTAMP}.log"

echo "=========================================================="
echo " Starting PRJ4 Multimodal Emotion Training on DGX"
echo " Target GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo " Dataset Path    : $DATA_PATH"
echo " Log File        : $LOG_FILE"
echo "=========================================================="

# Print GPU info
nvidia-smi

# Run training (output printed to console and saved to log)
python3 new.py \
    --data_dir "$DATA_PATH" \
    --batch_size 64 \
    --epochs 40 \
    --lr 2e-5 \
    --save_path "best3.pt" 2>&1 | tee "$LOG_FILE"

echo "Training finished. Checkpoint saved to best3.pt, log saved to $LOG_FILE."
