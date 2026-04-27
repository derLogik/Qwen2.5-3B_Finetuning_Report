"""Test v2 finetuned model generation"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=" * 60)
print("Testing Finetuned v2 Model")
print("=" * 60)

model_path = "C:/Users/lians/qwen_model/Qwen/Qwen2___5-3B"
adapter_path = "C:/Users/lians/qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final"

print("\n[1] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
print("Base model loaded!")

print("\n[2] Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Adapter loaded!")

print("\n[3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Test prompts
prompts = [
    "What is the capital of France?",
    "How does photosynthesis work?",
]

print("\n[4] Generation test:")
for prompt in prompts:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response[:150]}...")

print("\n" + "=" * 60)
print("Generation test complete!")
print("=" * 60)
