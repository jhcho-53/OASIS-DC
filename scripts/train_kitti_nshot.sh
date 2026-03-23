#!/bin/bash

# Train KITTI n-shot model with configurable shots
# Usage: ./scripts/train_kitti_nshot.sh [SHOTS] [SEED] [SAVE_DIR]
# Example: ./scripts/train_kitti_nshot.sh 10 0 runs/kitti_10shot

SHOTS=${1:-1}          # Default to 1-shot
SEED=${2:-0}           # Default seed 0  
SAVE_DIR=${3:-"runs/kitti_${SHOTS}shot"}

echo "Training KITTI ${SHOTS}-shot model (seed: ${SEED})"
echo "Results will be saved to: ${SAVE_DIR}"

python train.py \
    --dataset kitti \
    --config configs/kitti_nshot.yaml \
    --shots ${SHOTS} \
    --seed ${SEED} \
    --device cuda \
    --save-dir ${SAVE_DIR}

echo "Training completed. Checkpoints saved in ${SAVE_DIR}/"