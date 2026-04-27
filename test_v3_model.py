"""Test new v3 finetuned model - simpler generation"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=" * 60)
print("Testing Finetuned v3 Model (Alpaca Data)")
print("=" * 60)

model_path = "Qwen/Qwen2___5-3B"
adapter_path = "Qwen2.5-3B-Instruct-Finetuned-v3/final"

print("\n[1] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)
print("Base model loaded!")

print("\n[2] Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Adapter loaded!")

print("\n[3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Simple test
prompt = "What is the capital of France?"
text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\n[4] Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
