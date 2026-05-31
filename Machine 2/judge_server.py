# judge_server.py
# Compatible with your installed stack
# Stable reward generation with guideline-aware judging

import os
import re
import gc
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("judge_server")

API_KEY = os.environ.get("JUDGE_API_KEY", "")
MODEL_NAME = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
HOST = os.environ.get("JUDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("JUDGE_PORT", 8100))

if not API_KEY:
    raise RuntimeError("Set JUDGE_API_KEY")

tokenizer = None
model = None


def load_model():
    global tokenizer, model

    logger.info(f"Loading judge model: {MODEL_NAME}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map={"": 0},
    )

    model.eval()
    logger.info("Judge model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    global tokenizer, model
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


class CodeEntry(BaseModel):
    code: str
    guidelines: List[str]


class JudgeRequest(BaseModel):
    prompt: str
    entries: List[CodeEntry]


class JudgeResponse(BaseModel):
    rewards: List[float]


def verify_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def build_prompt(prompt: str, entries: List[CodeEntry]):

    blocks = []

    for i, entry in enumerate(entries, 1):

        guideline_text = "\n".join(
            f"- {g[:220]}" for g in entry.guidelines[:2]
        )

        code_text = entry.code[:700]

        block = f"""
Candidate {i}

Guidelines:
{guideline_text}

Code:
{code_text}
"""
        blocks.append(block.strip())

    joined = "\n\n-------------------\n\n".join(blocks)

    n = len(entries)

    lines = "\n".join([f"{i}:0.50" for i in range(1, n + 1)])

    return f"""
You are a strict Python code evaluator.

Task:
{prompt}

For each candidate score from 0.00 to 1.00 using:

- Correctness = 0.50
- Follows guidelines = 0.30
- Readability / quality = 0.20

Return ONLY {n} lines exactly like:

{lines}

No explanation.
No markdown.
No extra text.

{joined}
"""


def parse_rewards(text: str, n: int):

    logger.info(f"Judge raw output:\n{text}")

    nums = re.findall(r'^\s*\d+\s*:\s*([01](?:\.\d+)?)', text, re.M)

    vals = []

    for x in nums[:n]:
        try:
            v = float(x)
            v = max(0.0, min(1.0, v))
            vals.append(v)
        except Exception:
            vals.append(0.5)

    while len(vals) < n:
        vals.append(0.5)

    return vals[:n]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.post("/judge", response_model=JudgeResponse)
def judge(
    request: JudgeRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    verify_key(x_api_key)

    if not request.entries:
        return JudgeResponse(rewards=[])

    prompt_text = build_prompt(request.prompt, request.entries)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    rewards = parse_rewards(text, len(request.entries))

    logger.info(f"Rewards: {rewards}")

    return JudgeResponse(rewards=rewards)


if __name__ == "__main__":
    uvicorn.run(
        "judge_server:app",
        host=HOST,
        port=PORT,
        workers=1,
        log_level="info",
    )