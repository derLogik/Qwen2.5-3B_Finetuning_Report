"""
测试微调后的 Qwen2.5-3B-Instruct 模型
"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "qwen_model/Qwen2.5-3B-Instruct-Finetuned/final"

def test_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 测试 prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about cherry blossoms.",
    ]

    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print('-'*50)

        # 构建输入
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # 解码响应
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # 清理特殊字符避免编码问题
        response_clean = response.replace('\U0001f3b9', '[instrument]').replace('\U0001f3b5', '[music]')
        print(f"Response: {response_clean}")

    print(f"\n{'='*50}")
    print("All tests completed!")


if __name__ == "__main__":
    # 修复 Windows UTF-8 输出
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    test_model()
