"""
reward_pipeline.py
==================
Glue layer between GRPOCodeTrainer and LLMJudge.

Updated Logic:
  1. Retrieve ONE shared guideline set using:
        prompt + code A + code B
  2. Use same retrieved guidelines for both candidates.
  3. Judge compares both codes under same rules.
"""

import logging
from typing import List

logger = logging.getLogger("reward_pipeline")


class RewardPipeline:
    def __init__(self, vector_store, judge, top_k: int = 5):
        self.vector_store = vector_store
        self.judge = judge
        self.top_k = top_k

    def score_batch(self, codes: List[str], prompt: str = "") -> List[float]:

        if not codes:
            return []

        # ------------------------------------------
        # 1. Shared retrieval for all candidates
        # ------------------------------------------
        try:
            query_parts = []

            if prompt.strip():
                query_parts.append(f"Task:\n{prompt}")

            for i, code in enumerate(codes, 1):
                if code.strip():
                    query_parts.append(f"Code {i}:\n{code}")

            query = "\n\n".join(query_parts)[:4000]

            shared_chunks = self.vector_store.retrieve(
                query,
                top_k=self.top_k
            )

        except Exception as e:
            logger.warning(f"Vector store retrieval error: {e}")
            shared_chunks = []

        # Same chunks for every code candidate
        all_chunks = [shared_chunks for _ in codes]

        # ------------------------------------------
        # 2. Judge
        # ------------------------------------------
        try:
            results = self.judge.judge_batch(
                codes=codes,
                chunks=all_chunks,
                prompt=prompt,
            )
        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            return [0.5] * len(codes)

        # ------------------------------------------
        # 3. Parse rewards
        # ------------------------------------------
        rewards = []

        for r in results:
            if isinstance(r, dict):
                rewards.append(float(r.get("reward", 0.5)))
            else:
                try:
                    rewards.append(float(r))
                except Exception:
                    rewards.append(0.5)

        while len(rewards) < len(codes):
            rewards.append(0.5)

        logger.info(f"Rewards: {[round(r, 3) for r in rewards[:len(codes)]]}")

        return rewards[:len(codes)]