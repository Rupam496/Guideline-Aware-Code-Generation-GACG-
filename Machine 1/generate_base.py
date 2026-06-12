import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
OUTPUT_DIR = "base_model_output"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,   # change to float32 if needed
        device_map="auto"
    )

    return tokenizer, model


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


def save_output(prompt, output):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"output_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n")
        f.write(prompt + "\n\n")
        f.write("=== OUTPUT ===\n")
        f.write(output + "\n")

    print(f"Saved to: {filepath}")


def main():
    tokenizer, model = load_model()

    print("Model loaded. Enter prompt (Ctrl+C to exit):\n")

    while True:
        prompt = input(">>> ")

        full_prompt = f"""# Task:
{prompt}

# Write Python code:
"""

        output = generate_code(tokenizer, model, full_prompt)

        print("\n--- Generated Code ---\n")
        print(output)
        print("\n----------------------\n")

        save_output(prompt, output)


if __name__ == "__main__":
    main()