import chess
import chess.engine
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
import re
import wandb
from datasets import Dataset
from accelerate import Accelerator
from multiprocessing import Pool, cpu_count
import os

accelerator = Accelerator()
print(f"Available CUDA devices: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")

# Try different Stockfish paths based on environment
STOCKFISH_PATHS = [
    "/usr/local/bin/stockfish",
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    # Add user's home directory
    os.path.expanduser("~/stockfish")
]

def find_stockfish():
    """Find the first valid stockfish executable"""
    for path in STOCKFISH_PATHS:
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"Found stockfish at: {path}")
            return path
    print("WARNING: Stockfish not found! Please install it or specify the correct path.")
    return STOCKFISH_PATHS[0]  # Return first path as default

STOCKFISH_PATH = find_stockfish()

def load_positions(file_path):
    """Load chess positions from a text file containing FEN strings"""
    try:
        with open(file_path, 'r') as f:
            positions = [line.strip() for line in f if line.strip()]
        return positions
    except FileNotFoundError:
        print(f"Warning: Position file {file_path} not found")
        return []

# Load sampled positions
SAMPLED_POSITIONS = load_positions("sampled_positions.txt")


class ChessEnv(gym.Env):
    def __init__(self, skill_level=20, agent_color=chess.WHITE, stockfish_path="/usr/games/stockfish"):
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4677)  # Max number of legal moves in chess
        self.observation_space = spaces.Text(max_length=512)
        self.stockfish_path = stockfish_path
        self.skill_level = skill_level
        self.agent_color = agent_color
        self.previous_score = 0.0  # Track previous evaluation score
        self.move_history = []
        self.engine = None  # Initialize as None
        
    def _init_engine(self):
        """Initialize engine instance for current process"""
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({"Skill Level": self.skill_level})
            
    def init_engine(self):
        """Public method to initialize engine"""
        self._init_engine()

    def _get_observation(self):
        return self.board.fen()
    
    def _get_board_score(self):
        """Get position evaluation from Stockfish in centipawns"""
        self.init_engine()  # Ensure engine exists
        analysis = self.engine.analyse(self.board, chess.engine.Limit(depth=12))
        score = analysis['score'].white().score(mate_score=10000)
        return score if self.board.turn == chess.WHITE else -score

    def reset(self, seed=None, random_moves=False):
        super().reset(seed=seed)
        self.init_engine()  # Ensure engine exists
        self.board.reset()
        if random_moves:
            for _ in range(np.random.randint(10, 30)):
                self._make_random_move()
        return self._get_prompt(), {}

    def _make_random_move(self):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            move = np.random.choice(legal_moves)
            self.board.push(move)
            
    def _get_prompt(self):
        return f"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., 
<think> reasoning process here </think>
<answer> answer here </answer>.
User: Analyze this chess position and suggest the best move.
Current position (FEN): {self.board.fen()}
Legal moves: {', '.join([self.board.san(m) for m in self.board.legal_moves])}
Please think and respond with your move in SAN format. Assistant:
"""

    def configure_from_prompt(self, prompt):
        fen = prompt.split("Current position (FEN):")[1].split("Legal moves:")[0].strip()
        self.board.set_fen(fen)
        self.init_engine()  # Initialize after setting position
    
    def step(self, action):
        try:
            self.init_engine()  # Initialize before use
            # Store pre-move evaluation
            current_score = self._get_board_score()
            
            # Apply player move
            move = self.board.parse_san(action)
            self.board.push(move)
            
            # Get Stockfish response
            result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(result.move)
            
            # Calculate score delta
            new_score = self._get_board_score()
            score_delta = new_score - current_score
            
            # Adjust reward based on player color perspective
            reward = score_delta/100  # Convert centipawns to pawn units
            if self.agent_color == chess.BLACK:
                reward = -reward
                
            terminated = self.board.is_game_over()
            self.previous_score = new_score
            
            return self._get_observation(), reward, terminated, False, {}
            
        except chess.IllegalMoveError:
            return self._get_observation(), -1.0, True, False, {"illegal_move": True}

    def render(self):
        print(self.board)
        
    def close(self):
        """Properly close engine with timeout"""
        if self.engine:
            try:
                self.engine.quit()
                self.engine.close()
            except Exception as _:
                print("Error closing engine")
            finally:
                self.engine = None

def reward_function(prompts, completions):
    rewards = []
    samples_to_log = []
    local_env = ChessEnv()  
    
    try:
        for prompt, completion in zip(prompts, completions):
            try:
                local_env.init_engine()  # Use public method instead
                local_env.configure_from_prompt(prompt)
                # Extract SAN move from completion using regex
                move = re.search(r'([KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](=[QRNB])?|O-O|O-O-O)[+#]?', completion).group(0)
                # Apply player move
                obs, reward, terminated, _, _ = local_env.step(move)
                # if positive, clip to 1, else goto 0
                if reward > 0:
                    reward = 1
                else:
                    reward = 0
                rewards.append(reward)
                
                samples_to_log.append({
                    "prompt": prompt,
                    "completion": completion,
                    "move": move,
                    "reward": reward
                })
                
            except Exception as e:
                print(e)
                rewards.append(-10)  # Penalize invalid moves
                
                samples_to_log.append({
                    "prompt": prompt,
                    "completion": completion,
                    "move": "INVALID",
                    "reward": -1
                })
                
    finally:
        local_env.close()
        
    if wandb.run is not None and accelerator.is_main_process:  # Add is_main_process check
        wandb.log({
            "samples": wandb.Table(
                data=[[s["prompt"], s["completion"], s["move"], s["reward"]] 
                      for s in samples_to_log[:5]],
                columns=["prompt", "completion", "move", "reward"]
            )
        })
    
    return rewards

# Initialize environment and model
env = ChessEnv(
    agent_color=chess.WHITE,
    stockfish_path=STOCKFISH_PATH
)
model_id = "Qwen/Qwen2.5-1.5B-Instruct" # 
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GRPO Configuration

LLM_MAX_LENGTH = 256

# Check if we should use vLLM (need at least one GPU free)
total_gpus = torch.cuda.device_count()
vllm_device = None
use_vllm = False

if total_gpus > 3:  # If we have more than 3 GPUs, we can use vLLM
    use_vllm = True
    vllm_device = f"cuda:{total_gpus-1}"  # Use the last GPU
    print(f"Enabling vLLM on device: {vllm_device}")
else:
    print("Not enough GPUs for vLLM, disabling it")

grpo_config = GRPOConfig( # TODO: upload model to HF hub (push to hub, etc.)
    output_dir="chess-grpo",
    learning_rate=1e-5,
    logging_steps=10,
    gradient_accumulation_steps=8,
    max_completion_length=LLM_MAX_LENGTH,
    dataloader_pin_memory=True,
    report_to="wandb",
    disable_tqdm=False,
    use_vllm=use_vllm,
    vllm_device=vllm_device,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Move dataset generation outside the training loop
def generate_random_prompt():
    """Generate prompts without capturing environment instance"""
    local_env = ChessEnv()  # This will use the passed stockfish_path
    prompt, _ = local_env.reset(random_moves=True)
    print(prompt)
    local_env.close()
    return prompt

# Pre-generate all prompts before training
num_iterations = 10
num_samples = -1
num_test_samples = 5


if accelerator.is_main_process:
    print("Initializing WandB...")
    wandb.login()  # This will use API key from environment variable if set
    wandb.init(
        project="chess-rl",
        config={
            "model": model_id,
            "learning_rate": grpo_config.learning_rate,
            "num_iterations": num_iterations,
            "num_samples": num_samples,
            "num_test_samples": num_test_samples,
        }
    )
    
def generate_prompts_from_positions(positions):
    prompts = []
    for fen in positions:
        env = ChessEnv()
        env.board.set_fen(fen)
        prompts.append(env._get_prompt())
    return prompts

def evaluate_model(model, tokenizer, num_test_samples=5):
    """Evaluate model on fixed set of positions"""
    model = model.to(accelerator.device).eval()
    print("evaluating on device:", model.device)
    print("model precision:", model.dtype)
    test_prompts = generate_prompts_from_positions(SAMPLED_POSITIONS[:num_test_samples])
    results = []
    
    for prompt in test_prompts:
        try:
            env = ChessEnv()  # Fix path here - use the global variable
            env.init_engine()  # Initialize before configuration
            env.configure_from_prompt(prompt)
            # Generate model completion
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=LLM_MAX_LENGTH)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("completion:", completion)
            
            # Get model's move and reward
            move = re.search(r'([KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](=[QRNB])?|O-O|O-O-O)[+#]?', completion).group(0)
            obs, reward, terminated, _, _ = env.step(move)
            new_score = env._get_board_score()
            score_delta = new_score - env._get_board_score()
            
            # Get Stockfish's recommended move
            result = env.engine.play(env.board, chess.engine.Limit(time=0.1))
            best_move = env.board.san(result.move)
            
            results.append({
                "prompt": prompt,
                "completion": completion,
                "model_move": move,
                "best_move": best_move,
                "score_delta": score_delta/100,
                "reward": reward,
                "valid": True
            })
        except Exception as e:
            print("error encountered in evaluation:", e)
            results.append({
                "prompt": prompt,
                "completion": "ERROR",
                "model_move": "INVALID",
                "best_move": "",
                "score_delta": 0,
                "reward": -0.5,
                "valid": False
            })
    
    # Calculate metrics
    accuracy = sum(1 for r in results if r['valid'] and r['model_move'] == r['best_move'])/len(results)
    avg_score_delta = sum(r['score_delta'] for r in results)/len(results)
    avg_reward = sum(r['reward'] for r in results)/len(results)
    illegal_rate = sum(1 for r in results if not r['valid'])/len(results)
    
    # Log to WandB
    if wandb.run is not None and accelerator.is_main_process: 
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/avg_score_delta": avg_score_delta,
            "eval/avg_reward": avg_reward,
            "eval/illegal_rate": illegal_rate,
            "eval/examples": wandb.Table(
                columns=["Position", "Model Move", "Best Move", "Score Δ", "Reward", "Raw Completion"],
                data=[[
                    r['prompt'],
                    r['model_move'],
                    r['best_move'],
                    r['score_delta'],
                    r['reward'],
                    r['completion']
                ] for r in results[:5]]  # Log first 5 examples
            )
        })
    
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Avg Score Δ: {avg_score_delta:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Illegal Moves: {illegal_rate:.2%}\n")


prompts = generate_prompts_from_positions(SAMPLED_POSITIONS[:num_samples])
dataset = Dataset.from_dict({"prompt": prompts})
print("[DEBUG] Dataset prepared with prompts:", len(prompts))

print("[DEBUG] About to create trainer...")
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=reward_function,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)
if accelerator.is_main_process:
    print("Evaluating model...")
    evaluate_model(trainer.model, tokenizer, num_test_samples)
# Wait for all processes to finish evaluation
accelerator.wait_for_everyone()
print("[DEBUG] Trainer created successfully")

print("[DEBUG] Starting training...")
trainer.train()
print("[DEBUG] Training complete")

trainer.save_model("chess-grpo")
trainer.push_to_hub(dataset_name="chess-grpo")
trainer.generate_completions()
if accelerator.is_main_process:
    evaluate_model(trainer.model, tokenizer, num_test_samples)

if accelerator.is_main_process:
    wandb.finish()
