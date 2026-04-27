"""
合并 LoRA adapter 到 base 模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "qwen_model/Qwen/Qwen2___5-3B"
ADAPTER_PATH = "qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final"
OUTPUT_PATH = "qwen_model/Qwen2.5-3B-Instruct-Merged"

def merge_and_save():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",  # 先放 CPU
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print("Saving merged model...")
    merged_model.save_pretrained(OUTPUT_PATH)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"\nMerged model saved to: {OUTPUT_PATH}")

    # 清理
    del model, merged_model, base_model
    del base_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merge_and_save()
