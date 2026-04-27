"""Minimal training test"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2___5-3B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2___5-3B",
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="cuda:0",
)
print("Model loaded!")

# Setup LoRA
print("Setting up LoRA...")
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

# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        text = f"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"
        enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze(),
        }

dataset = SimpleDataset(10)

# Training args
training_args = TrainingArguments(
    output_dir="test_output",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    fp16=True,
    save_steps=10,
    logging_steps=5,
    report_to="none",
)

print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Training...")
trainer.train()

print("Done!")
