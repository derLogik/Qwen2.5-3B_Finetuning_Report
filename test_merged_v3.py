"""Test merged v3 model"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Testing Merged v3 Model")
print("=" * 60)

model_path = "Qwen2.5-3B-Instruct-v3-Merged"

print("\nLoading merged model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)
print("Model loaded!")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Test prompts
prompts = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is artificial intelligence?",
    "Explain the theory of relativity.",
]

print("\n" + "-" * 60)
print("Generation Results:")
print("-" * 60)

for prompt in prompts:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response[:200]}...")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
