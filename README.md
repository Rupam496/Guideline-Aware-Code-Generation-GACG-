# Guideline-Aware Code Generation (GACG)

Guideline-Aware Code Generation (GACG) is a reinforcement learning-based framework designed to fine-tune Large Language Models (LLMs) for generating codes that follows coding guidelines. The framework combines embedding-based guideline retrieval, LLM-based judging, and Group Relative Policy Optimization (GRPO) to improve guideline adherence during code generation.

---

# Project Architecture

```text
Prompt
   ↓
Generator LLM
   ↓
Generate K Candidate Codes
   ↓
Combine:
(Prompt + All Generated Codes)
   ↓
Combined Embedding Generation
   ↓
Retrieve Relevant Guidelines
   ↓
Judge LLM Evaluation
   ↓
Reward Scores
   ↓
GRPO Loss Calculation
   ↓
Generator Model Update
```

---

# Models Used

## Generator Model

Model:

```text
Qwen/Qwen2.5-Coder-1.5B
```

Purpose:

* Generates multiple candidate codes
* Fine-tuned using LoRA
* Optimized through GRPO-based reinforcement learning

---

## Judge Model

Model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Purpose:

* Evaluates generated candidate codes
* Scores guideline adherence
* Provides reward signals

Scoring Criteria:

* Correctness
* Guideline adherence
* Code quality

Reward Range:

```text
0.0 → Poor Quality
1.0 → Excellent Quality
```

---

# Guideline Retrieval System

The framework uses semantic embedding-based retrieval to dynamically select relevant coding guidelines during training.

## Guideline Database

Coding guidelines are stored in:

```text
guidelines.txt
```

The guideline file contains:

* rule descriptions
* correct code examples
* incorrect code examples

These guidelines are processed and converted into vector embeddings during indexing. The generated embeddings are stored inside:

```text
vector_store.npz
```

## Query Construction

A single query representation is created using:

```text
Prompt + Generated Code 1 + Generated Code 2 + ... + Generated Code K
```

## Retrieval Process

1. Combine prompt and generated codes
2. Create a single embedding vector representation
3. Perform similarity search over the embedded guideline database (`vector_store.npz`)
4. Retrieve top-k relevant guidelines
5. Send retrieved guidelines to the Judge LLM for evaluation

The retrieved guidelines contain:

* rule descriptions
* correct code examples
* incorrect code examples

---

# Distributed Architecture

## Machine 1 (Training Machine)

Responsibilities:

* Generator model
* Reward pipeline
* Embedding retrieval
* GRPO training
* LoRA fine-tuning

Files:

```text
train.py
grpo_trainer.py
reward_pipeline.py
judge.py
vector_store.py
```

---

## Machine 2 (Judge Server)

Responsibilities:

* Judge LLM inference
* Reward computation
* HTTP API server

Files:

```text
judge_server.py
```

Communication:

```text
Machine 1
    ↓ HTTP Request
Machine 2
    ↓ Reward Response
Machine 1
```

Used Technologies:

* FastAPI
* Uvicorn
* Requests

---

# Build Guideline Index

```bash
python build_index.py guidelines.txt
```

Generates:

```text
vector_store.npz
```

---

# Start Judge Server (Machine 2)

Set environment variables:

```bash
export JUDGE_API_KEY=secret_key
export JUDGE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
```

Run:

```bash
python judge_server.py
```

---

# Start Training (Machine 1)

Set environment variables:

```bash
export JUDGE_SERVER_URL=http://<machine2_ip>:8100
export JUDGE_API_KEY=secret_key
```

Run:

```bash
python train.py prompts.txt
```

---

# Implementation and Results

The proposed Guideline-Aware Code Generation (GACG) framework has been fully implemented and evaluated in a distributed training environment. Experimental observations indicate that the fine-tuned model demonstrates improved adherence to coding guidelines compared to the base model while maintaining code quality and correctness.

The framework successfully learns guideline-aware generation behavior through reinforcement learning using retrieved coding rules and LLM-based reward signals.

Generated outputs from both the base model and the fine-tuned model are stored in:

```text
results/
```

This directory contains generated outputs used for qualitative comparison between the base model and the fine-tuned model.

---

# Output

The final model aims to generate:

* guideline-compliant code
* improved readability
* better software engineering practices
* higher-quality generated implementations

---

# Author

```text
Rupam Das
```
