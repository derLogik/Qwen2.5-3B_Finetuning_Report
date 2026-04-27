"""
直接用 transformers 加载 PEFT 模型进行生成测试
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

def test_generation():
    print("="*60)
    print("Testing Finetuned Model Generation")
    print("="*60)

    # 加载 base 和 adapter
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
    )

    # 测试 prompts
    test_prompts = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is artificial intelligence?",
        "Explain the theory of relativity.",
        "What causes earthquakes?",
    ]

    print("\n" + "-"*60)
    print("Generation Results:")
    print("-"*60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")

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
        print(f"    Response: {response[:150]}...")

    print("\n" + "="*60)
    print("Generation test complete!")
    print("="*60)

if __name__ == "__main__":
    test_generation()
