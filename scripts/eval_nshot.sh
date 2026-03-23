#!/bin/bash

# Evaluate n-shot models with configurable parameters
# Usage: ./scripts/eval_nshot.sh [DATASET] [SHOTS] [CHECKPOINT] [MODE] [OUTPUT]
# Example: ./scripts/eval_nshot.sh nyu 10 runs/nyu_10shot/checkpoint_best.pth all results/nyu_10shot.json

DATASET=${1:-nyu}      # Default to NYU
SHOTS=${2:-1}          # Default to 1-shot
CHECKPOINT=${3}        # Checkpoint path (required for full/residual_off modes)
MODE=${4:-all}         # Default to all modes
OUTPUT=${5:-"results/${DATASET}_${SHOTS}shot_results.json"}

echo "Evaluating ${DATASET} ${SHOTS}-shot model"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT}"

if [[ ${DATASET} == "nyu" ]]; then
    CONFIG="configs/nyu_eval.yaml"
elif [[ ${DATASET} == "kitti" ]]; then
    CONFIG="configs/kitti_eval.yaml"
else
    echo "Error: Unknown dataset ${DATASET}"
    exit 1
fi

# Build command
CMD="python eval.py --dataset ${DATASET} --config ${CONFIG} --mode ${MODE} --output ${OUTPUT}"

# Add checkpoint if provided
if [[ -n ${CHECKPOINT} ]]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT}"
    echo "Using checkpoint: ${CHECKPOINT}"
fi

# Add shots parameter if using n-shot config
if [[ ${SHOTS} != "1" ]]; then
    CMD="${CMD} --shots ${SHOTS}"
fi

echo "Running: ${CMD}"
${CMD}

echo "Evaluation completed. Results saved to ${OUTPUT}"