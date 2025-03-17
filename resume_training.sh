#!/bin/bash

# First make sure Stockfish is available
if ! which stockfish > /dev/null; then
    echo "Stockfish not found. Installing..."
    ./install_stockfish.sh
fi

# Default values
CHECKPOINT_PATH=""
NUM_NODES=1
NUM_PROCESSES=1
MIXED_PRECISION="bf16"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --num_nodes)
      NUM_NODES="$2"
      shift 2
      ;;
    --num_processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --mixed_precision)
      MIXED_PRECISION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set the environment variable for the checkpoint if specified
if [[ -n "$CHECKPOINT_PATH" ]]; then
    export RESUME_CHECKPOINT="$CHECKPOINT_PATH"
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

# Set up accelerate configuration
accelerate config --config_file=accelerate_config.yaml

# Launch the training
echo "Starting training with $NUM_NODES node(s) and $NUM_PROCESSES process(es) per node..."
accelerate launch \
    --config_file=accelerate_config.yaml \
    --num_processes=$NUM_PROCESSES \
    --num_machines=$NUM_NODES \
    --machine_rank=0 \
    --mixed_precision=$MIXED_PRECISION \
    chess_gym.py

echo "Training completed" 