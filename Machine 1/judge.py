# judge.py
# Only fix added: clean metadata from c.description

import os
import time
import logging
import requests

from typing import List
from guideline_chunker import RuleChunk

logger = logging.getLogger("judge_client")

JUDGE_URL   = os.environ.get("JUDGE_SERVER_URL", "http://localhost:8100")
API_KEY     = os.environ.get("JUDGE_API_KEY", "")
TIMEOUT     = int(os.environ.get("JUDGE_TIMEOUT_SEC", 120))
MAX_RETRIES = 3


class LLMJudge:

    def __init__(self, server_url: str = JUDGE_URL, api_key: str = API_KEY):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def load(self):
        pass

    def unload(self):
        pass

    def ping(self):
        try:
            r = requests.get(
                f"{self.server_url}/health",
                headers=self.headers,
                timeout=10
            )
            return r.status_code == 200
        except Exception:
            return False

    def _clean_advice(self, text: str) -> str:
        lines = []

        for line in text.splitlines():
            s = line.strip()

            if not s:
                continue

            low = s.lower()

            if low.startswith("priority:"):
                continue
            if low.startswith("category:"):
                continue
            if low.startswith("ast:"):
                continue

            lines.append(s)

        return " ".join(lines)

    def judge_batch(
        self,
        codes: List[str],
        chunks: List[List[RuleChunk]],
        prompt: str = ""
    ):

        entries = []

        for code, code_chunks in zip(codes, chunks):

            guidelines = []

            for c in code_chunks[:3]:
                text = f"Rule: {c.rule_title}\n"

                advice = self._clean_advice(c.description)

                if advice:
                    text += f"Advice: {advice[:220]}\n"

                if c.correct_code.strip():
                    text += f"Good Example:\n{c.correct_code[:300]}\n"

                if c.wrong_code.strip():
                    text += f"Bad Example:\n{c.wrong_code[:300]}\n"

                guidelines.append(text.strip())

            entries.append({
                "code": code,
                "guidelines": guidelines
            })

        payload = {
            "prompt": prompt,
            "entries": entries
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.post(
                    f"{self.server_url}/judge",
                    json=payload,
                    headers=self.headers,
                    timeout=TIMEOUT
                )

                if r.status_code == 200:
                    data = r.json()
                    rewards = data.get("rewards", [])

                    while len(rewards) < len(codes):
                        rewards.append(0.5)

                    rewards = rewards[:len(codes)]

                    return [{"reward": float(x)} for x in rewards]

            except Exception as e:
                logger.warning(f"Judge attempt {attempt} failed: {e}")

            time.sleep(attempt * 2)

        return [{"reward": 0.5} for _ in codes]