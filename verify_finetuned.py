"""
验证微调模型效果 - 生成对比测试
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_generation():
    print("="*70)
    print("FINETUNED MODEL GENERATION TEST")
    print("="*70)

    # Load base model
    print("\n[1] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.eval()

    base_tokenizer = AutoTokenizer.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
    )

    # Load finetuned model
    print("[2] Loading finetuned model (base + LoRA adapter)...")
    ft_model = PeftModel.from_pretrained(base_model, "qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final")
    ft_model.eval()

    ft_tokenizer = base_tokenizer

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is artificial intelligence?",
        "Explain the theory of relativity.",
        "What causes earthquakes?",
    ]

    print("\n" + "-"*70)
    print("GENERATION COMPARISON (Base vs Finetuned)")
    print("-"*70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] PROMPT: {prompt}")

        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Base model generation
        inputs = base_tokenizer(text, return_tensors="pt").to(base_model.device)
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=base_tokenizer.pad_token_id or base_tokenizer.eos_token_id,
            )
        base_response = base_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    BASE:     {base_response[:120]}...")

        # Finetuned model generation
        inputs = ft_tokenizer(text, return_tensors="pt").to(ft_model.device)
        with torch.no_grad():
            outputs = ft_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=ft_tokenizer.pad_token_id or ft_tokenizer.eos_token_id,
            )
        ft_response = ft_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    FINETUNED: {ft_response[:120]}...")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

if __name__ == "__main__":
    test_generation()
