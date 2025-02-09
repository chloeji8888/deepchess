
export TOKENIZERS_PARALLELISM=false
accelerate launch --config_file cfg/4xgpu.yml chess_gym.py