export TOKENIZERS_PARALLELISM=false

# Using configuration with 3 GPUs (0,1,2) for training, leaving GPU 3 for vLLM
accelerate launch --config_file cfg/4xgpu.yml chess_gym.py