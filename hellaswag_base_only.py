"""HellaSwag Quick Test - Base Model Only"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print("=" * 60, file=sys.stdout)
print("HellaSwag Quick Test - Base Model Only", file=sys.stdout)
print("=" * 60, file=sys.stdout)

model_path = "Qwen/Qwen2___5-3B"
print("\nLoading base model...", file=sys.stdout)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model.eval()
print("Model loaded!", file=sys.stdout)

print("\nLoading dataset...", file=sys.stdout)
dataset = load_dataset("Rowan/hellaswag", split="test")
dataset = dataset.select(range(10))

correct = 0
for i, item in enumerate(dataset):
    ctx = item["ctx"]
    label = item["label"]
    endings = item["endings"]

    scores = []
    for ending in endings:
        text = f"{ctx} {ending}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if inputs["input_ids"].shape[1] == 0:
            scores.append(float("-inf"))
            continue
        inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = base_model(**inputs)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
        ending_tokens = tokenizer.encode(ending, add_special_tokens=False)
        if ending_tokens:
            score = log_probs[0, ending_tokens[0]].item()
        else:
            score = float("-inf")
        scores.append(score)

    pred = scores.index(max(scores))
    if pred == label:
        correct += 1
    print(f"{i+1}/10 - pred={pred}, label={label}, correct={correct}", file=sys.stdout)

acc = correct / 10
print(f"\nAccuracy: {acc:.4f}", file=sys.stdout)
print("=" * 60, file=sys.stdout)

with open("hellaswag_result.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Correct: {correct}/10\n")
