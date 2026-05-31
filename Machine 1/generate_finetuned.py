import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
LORA_PATH = "grpo_output"   # your folder from screenshot
OUTPUT_DIR = "finetuned_model_output"


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,   # change to float32 if GPU issues
        device_map="auto"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    model.eval()

    print("Model ready ✅")
    return tokenizer, model


# -----------------------------
# GENERATE
# -----------------------------
def generate_code(tokenizer, model, prompt, max_new_tokens=212):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------------
# SAVE OUTPUT
# -----------------------------
def save_output(prompt, output):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"finetuned_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n")
        f.write(prompt + "\n\n")
        f.write("=== OUTPUT ===\n")
        f.write(output + "\n")

    print(f"Saved to: {filepath}")


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    tokenizer, model = load_model()

    print("\nEnter prompt (Ctrl+C to exit):\n")

    while True:
        prompt = input(">>> ").strip()

        if not prompt:
            print("Empty prompt, skipping...\n")
            continue

        full_prompt = f"""# Task:
{prompt}

# Write Python code:
"""

        output = generate_code(tokenizer, model, full_prompt)

        print("\n--- Generated Code (Finetuned) ---\n")
        print(output)
        print("\n----------------------------------\n")

        save_output(prompt, output)


if __name__ == "__main__":
    main()