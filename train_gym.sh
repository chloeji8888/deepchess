export TOKENIZERS_PARALLELISM=false
# Use 3 GPUs for training, leaving 1 GPU for vLLM inference
accelerate launch --num_processes 3 chess_gym.py