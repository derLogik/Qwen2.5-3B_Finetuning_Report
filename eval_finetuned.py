"""
评估微调后的 Qwen2.5-3B 模型 (LoRA adapter)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "qwen_model/Qwen/Qwen2___5-3B"
ADAPTER_PATH = "qwen_model/Qwen2.5-3B-Instruct-Finetuned/final"

def load_finetuned_model():
    """加载带有 LoRA adapter 的模型"""
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )

    return model, tokenizer

def test_generation():
    """测试生成效果"""
    model, tokenizer = load_finetuned_model()

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]

    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print('-'*50)

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
        print(f"Response: {response[:500]}")

    print(f"\n{'='*50}")
    print("Test completed!")

if __name__ == "__main__":
    test_generation()
