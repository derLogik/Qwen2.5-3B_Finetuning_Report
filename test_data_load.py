"""Test dataset loading"""
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
print(f"Script dir: {script_dir}")

# Qwen model is at: Qwen/Qwen2___5-3B relative to C:/Users/lians/qwen_model/
# But script is in C:/Users/lians/qwen_model/qwen_model/ so go up one level
parent_dir = script_dir.parent
print(f"Parent dir: {parent_dir}")

model_path = str(parent_dir / "Qwen" / "Qwen2___5-3B")
print(f"Model path: {model_path}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading data...")
data_path = str(script_dir / "alpaca_data" / "alpaca_gpt4_data.jsonl")
print(f"Data path: {data_path}")
samples = []
with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            samples.append(json.loads(line))
        if i >= 9:
            break

print(f"Loaded {len(samples)} samples")
print("Sample text:", samples[0]["text"][:200])
