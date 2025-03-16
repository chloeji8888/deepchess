export TOKENIZERS_PARALLELISM=false
# Enable detailed error reporting for distributed training
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Now that we're using bfloat16 instead of 8-bit quantization, we can use multiple GPUs
accelerate launch --multi_gpu --num_processes 3 --gpu_ids 0,1,2 chess_gym.py