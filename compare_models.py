"""Compare v2 and v3 finetuned models using simple benchmark"""
import sys
sys.stdout = open("benchmark_output.txt", "w")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=" * 60)
print("Model Generation Test - Compare v2 vs v3")
print("=" * 60)

model_path = "Qwen/Qwen2___5-3B"

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Test prompts
prompts = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is artificial intelligence?",
]

# v2 model
print("\n" + "=" * 60)
print("Testing v2 Model (old synthetic data)")
print("=" * 60)

print("\nLoading v2 model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)
v2_model = PeftModel.from_pretrained(base_model, "Qwen2.5-3B-Instruct-Finetuned-v2/final")
v2_model.eval()
print("v2 model loaded!")

for prompt in prompts:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(v2_model.device)

    with torch.no_grad():
        outputs = v2_model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"v2 Response: {response[:150]}...")

del base_model, v2_model
torch.cuda.empty_cache()

# v3 model
print("\n" + "=" * 60)
print("Testing v3 Model (Alpaca real data)")
print("=" * 60)

print("\nLoading v3 model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)
v3_model = PeftModel.from_pretrained(base_model, "Qwen2.5-3B-Instruct-Finetuned-v3/final")
v3_model.eval()
print("v3 model loaded!")

for prompt in prompts:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(v3_model.device)

    with torch.no_grad():
        outputs = v3_model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"v3 Response: {response[:150]}...")

print("\n" + "=" * 60)
print("Benchmark complete!")
print("=" * 60)
