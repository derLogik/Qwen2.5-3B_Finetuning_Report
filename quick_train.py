"""Quick training test with file output"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2___5-3B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2___5-3B",
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)

print("Setting up LoRA...", flush=True)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Simple dataset with 100 samples
print("Loading data...", flush=True)
samples = []
with open("qwen_model/alpaca_data/alpaca_gpt4_data.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples", flush=True)

class SimpleDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        enc = self.tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze(),
        }

dataset = SimpleDataset(samples, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

print("Starting training...", flush=True)
model.train()
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to("cuda:0")
    attention_mask = batch["attention_mask"].to("cuda:0")
    labels = batch["labels"].to("cuda:0")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}", flush=True)

    if step >= 50:
        break

print("Training complete!", flush=True)
model.save_pretrained("Qwen2.5-3B-Instruct-Finetuned-v3/final")
print("Saved model!", flush=True)
