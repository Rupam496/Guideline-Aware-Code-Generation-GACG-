"""
train.py
========
Entry point — runs on Machine 1.

Before running:
  1. Start judge_server.py on Machine 2:
         JUDGE_API_KEY=your_secret python judge_server.py

  2. Set environment variables on Machine 1:
         export JUDGE_SERVER_URL=http://<machine2_ip>:8100
         export JUDGE_API_KEY=your_secret

  3. Build the vector index (once):
         python build_index.py guidelines.txt

  4. Run training:
         python train.py prompts.txt
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train")

from vector_store    import VectorStore
from judge           import LLMJudge
from reward_pipeline import RewardPipeline
from grpo_trainer    import GRPOCodeTrainer, TrainerConfig


# =====================================================
# Helpers
# =====================================================
def load_prompts(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]

    cleaned = []
    for line in lines:
        # Strip leading numbering: "1. Write a function …"
        if line.split(".")[0].isdigit():
            line = line.split(". ", 1)[-1].strip()
        cleaned.append(line)

    return cleaned


def check_env() -> tuple:
    url = os.environ.get("JUDGE_SERVER_URL", "")
    key = os.environ.get("JUDGE_API_KEY",    "")

    if not url:
        logger.error(
            "JUDGE_SERVER_URL is not set.\n"
            "  export JUDGE_SERVER_URL=http://<machine2_ip>:8100"
        )
        sys.exit(1)

    if not key:
        logger.error(
            "JUDGE_API_KEY is not set.\n"
            "  export JUDGE_API_KEY=your_shared_secret"
        )
        sys.exit(1)

    return url, key


# =====================================================
# Main
# =====================================================
def main():

    prompts_path = sys.argv[1] if len(sys.argv) > 1 else "prompts.txt"

    # 1. Validate environment
    judge_url, api_key = check_env()
    logger.info(f"Judge server: {judge_url}")

    # 2. Load vector store (CPU)
    logger.info("Loading vector store…")
    store = VectorStore()
    store.load("vector_store.npz")

    # 3. Init judge HTTP client
    judge = LLMJudge(server_url=judge_url, api_key=api_key)

    logger.info("Pinging judge server…")
    if not judge.ping():
        logger.error(
            "Judge server not reachable. "
            f"Make sure judge_server.py is running at {judge_url}"
        )
        sys.exit(1)
    logger.info("Judge server reachable ✓")

    # 4. Build reward pipeline
    pipeline = RewardPipeline(
        vector_store=store,
        judge=judge,
        top_k=6,
    )

    # 5. Load prompts
    prompts = load_prompts(prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

    if not prompts:
        logger.error("No prompts found. Exiting.")
        sys.exit(1)

    # 6. Trainer config  (3B model, tuned hyperparameters)
    config = TrainerConfig()

    logger.info(
        f"TrainerConfig | "
        f"model={config.model_name} | "
        f"epochs={config.num_train_epochs} | "
        f"group_size={config.group_size} | "
        f"max_completion_length={config.max_completion_length} | "
        f"lr={config.learning_rate} | "
        f"lora_r={config.lora_r} | "
        f"beta(kl)={config.kl_beta}"
    )

    # 7. Init trainer (downloads + loads 3B generator)
    trainer = GRPOCodeTrainer(
        config=config,
        reward_pipeline=pipeline,
    )

    # 8. Train
    trainer.train(prompts)


if __name__ == "__main__":
    main()
