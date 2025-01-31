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

accelerator = Accelerator()


class ChessEnv(gym.Env):
    def __init__(self, stockfish_path, skill_level=20, agent_color=chess.WHITE):
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4677)  # Max number of legal moves in chess
        self.observation_space = spaces.Text(max_length=128)
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
            
    def _get_observation(self):
        return self.board.fen()
    
    def _get_reward(self):
        # Basic reward function - modify based on training needs
        if self.board.is_checkmate():
            return 1.0  # Win
        if self.board.is_game_over():
            return -1.0  # Loss or draw
        return 0.0  # Intermediate state
    
    def _get_board_score(self):
        """Get position evaluation from Stockfish in centipawns"""
        self._init_engine()  # Ensure engine exists
        analysis = self.engine.analyse(self.board, chess.engine.Limit(depth=12))
        score = analysis['score'].white().score(mate_score=10000)
        return score if self.board.turn == chess.WHITE else -score

    def reset(self, seed=None, random_moves=False):
        super().reset(seed=seed)
        self._init_engine()  # Ensure engine exists
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
        return f"""Analyze this chess position and suggest the best move.
Current position (FEN): {self.board.fen()}
Legal moves: {', '.join([self.board.san(m) for m in self.board.legal_moves])}
Please respond with your move in SAN format. Answer:"""

    def configure_from_prompt(self, prompt):
        fen = prompt.split("Current position (FEN):")[1].split("Legal moves:")[0].strip()
        self.board.set_fen(fen)
        self._init_engine()  # Initialize after setting position
    
    def step(self, action):
        try:
            self._init_engine()  # Initialize before use
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
            except Exception as e:
                print(f"Error closing engine: {e}")
            finally:
                self.engine = None

def reward_function(prompts, completions):
    rewards = []
    samples_to_log = []
    # Create per-process environment
    env = ChessEnv(stockfish_path="/usr/local/bin/stockfish")
    
    try:
        for prompt, completion in zip(prompts, completions):
            try:
                env._init_engine()  # Re-init if needed
                env.configure_from_prompt(prompt)
                # Extract SAN move from completion using regex
                move = re.search(r'([KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](=[QRNB])?|O-O|O-O-O)[+#]?', completion).group(0)
                # Apply player move
                obs, reward, terminated, _, _ = env.step(move)
                rewards.append(reward)
                
                samples_to_log.append({
                    "prompt": prompt,
                    "completion": completion,
                    "move": move,
                    "reward": reward
                })
                
            except Exception as e:
                print(e)
                rewards.append(-0.5)  # Penalize invalid moves
                
                samples_to_log.append({
                    "prompt": prompt,
                    "completion": completion,
                    "move": "INVALID",
                    "reward": -0.5
                })
                
    finally:
        env.close()
        
    if wandb.run is not None:
        wandb.log({
            "samples": wandb.Table(
                data=[[s["prompt"], s["completion"], s["move"], s["reward"]] 
                      for s in samples_to_log[:5]],  # Log first 5 samples
                columns=["prompt", "completion", "move", "reward"]
            )
        })
        
    return rewards

# Initialize environment and model
env = ChessEnv(
    stockfish_path="/usr/local/bin/stockfish",
    agent_color=chess.WHITE  # or chess.BLACK
)
model_id = "Qwen/Qwen2-0.5B-Instruct" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GRPO Configuration
grpo_config = GRPOConfig(
    output_dir="chess-grpo",
    learning_rate=1e-5,
    logging_steps=10,
    gradient_accumulation_steps=16,
    max_completion_length=128,
    use_mps_device=True,
    report_to="wandb",
    run_name="chess-grpo-run",
    dataloader_pin_memory=False
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Move dataset generation outside the training loop
def generate_random_prompt(stockfish_path):
    """Generate prompts without capturing environment instance"""
    local_env = ChessEnv(stockfish_path)
    prompt, _ = local_env.reset(random_moves=True)
    print(prompt)
    local_env.close()
    return prompt

# Pre-generate all prompts before training
num_iterations = 100
num_samples = 10
wandb.init(
    project="chess-rl",
    config={
        "model": model_id,
        "learning_rate": grpo_config.learning_rate,
        "num_iterations": num_iterations,
        "num_samples": num_samples,
    }
)


stockfish_path = "/usr/local/bin/stockfish"

def evaluate_model(model, tokenizer, num_samples=5):
    """Evaluate model on fixed set of positions"""
    test_prompts = [generate_random_prompt(stockfish_path) for _ in range(num_samples)]
    results = []
    
    for prompt in test_prompts:
        try:
            env = ChessEnv(stockfish_path=stockfish_path)
            env._init_engine()  # Initialize before configuration
            env.configure_from_prompt(prompt)
            # Generate model completion
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=128)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
            results.append({
                "prompt": prompt,
                "completion": completion,
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
    if wandb.run:
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/avg_score_delta": avg_score_delta,
            "eval/avg_reward": avg_reward,
            "eval/illegal_rate": illegal_rate,
            "eval/examples": wandb.Table(
                columns=["Position", "Model Move", "Best Move", "Score Δ", "Reward"],
                data=[[
                    r['prompt'].split('\n')[1].split(': ')[1],  # FEN
                    r['model_move'],
                    r['best_move'],
                    r['score_delta'],
                    r['reward']
                ] for r in results[:5]]  # Log first 5 examples
            )
        })
    
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Avg Score Δ: {avg_score_delta:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Illegal Moves: {illegal_rate:.2%}\n")

for i in range(num_iterations):
    print(f"Iteration {i+1} of {num_iterations}")
    prompts = [generate_random_prompt(stockfish_path) for _ in range(num_samples)]
    dataset = Dataset.from_dict({"prompt": prompts})
    print("Prepared dataset")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_function,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    print("Prepare Training...")
    trainer.model, trainer.optimizer = accelerator.prepare(
        trainer.model, trainer.optimizer
    )
    print("Training...")
    trainer.train()
    print("Saving model...")
    
    # Run evaluation every 5 epochs
    if (i + 1) % 5 == 0:
        print(f"\nRunning evaluation after epoch {i+1}")
        evaluate_model(trainer.model, tokenizer)
        
    # Save model checkpoints
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer.save_model(f"chess-grpo-epoch-{i}")
        
    accelerator.wait_for_everyone()
    
wandb.finish()