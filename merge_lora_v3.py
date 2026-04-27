"""
合并 LoRA adapter 到 base 模型 (v3)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2___5-3B"
ADAPTER_PATH = "Qwen2.5-3B-Instruct-Finetuned-v3/final"
OUTPUT_PATH = "Qwen2.5-3B-Instruct-v3-Merged"

def merge_and_save():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
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

    del model, merged_model, base_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    merge_and_save()
