#!/bin/bash

# Train NYU n-shot model with configurable shots
# Usage: ./scripts/train_nyu_nshot.sh [SHOTS] [SEED] [SAVE_DIR]
# Example: ./scripts/train_nyu_nshot.sh 10 0 runs/nyu_10shot

SHOTS=${1:-1}          # Default to 1-shot
SEED=${2:-0}           # Default seed 0  
SAVE_DIR=${3:-"runs/nyu_${SHOTS}shot"}

echo "Training NYU ${SHOTS}-shot model (seed: ${SEED})"
echo "Results will be saved to: ${SAVE_DIR}"

python train.py \
    --dataset nyu \
    --config configs/nyu_nshot.yaml \
    --shots ${SHOTS} \
    --seed ${SEED} \
    --device cuda \
    --save-dir ${SAVE_DIR}

echo "Training completed. Checkpoints saved in ${SAVE_DIR}/"