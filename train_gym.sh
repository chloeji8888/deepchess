export TOKENIZERS_PARALLELISM=false
# Enable detailed error reporting for distributed training
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Try running with explicit GPU selection instead of config file
accelerate launch --multi_gpu --num_processes 3 --gpu_ids 0,1,2 chess_gym.py