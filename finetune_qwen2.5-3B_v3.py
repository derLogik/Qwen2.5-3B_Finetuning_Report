"""
Qwen2.5-3B Base -> Instruct 微调 (v3)
使用真实 Alpaca GPT4 指令数据 + LoRA
简单 PyTorch 训练循环
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
import json
from pathlib import Path
import gc
import os


@dataclass
class FinetuneConfig:
    base_model_path: str = "Qwen/Qwen2___5-3B"
    data_path: str = "qwen_model/alpaca_data/alpaca_gpt4_data.jsonl"
    output_dir: str = "Qwen2.5-3B-Instruct-Finetuned-v3"
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 512
    lora_rank: int = 16
    num_train_epochs: int = 1
    learning_rate: float = 2e-4


class InstructionDataset(Dataset):
    """指令微调数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading data from {data_path}...")
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def setup_lora_config(config: FinetuneConfig):
    """配置 LoRA"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )


def train():
    config = FinetuneConfig()

    print("=" * 60)
    print("Qwen2.5-3B Base -> Instruct Fine-tuning (v3)")
    print("Using Alpaca GPT4 Instruction Data")
    print("=" * 60)

    # 加载 tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载模型
    print("\n[2] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cuda:0",
    )

    # 配置 LoRA
    print("\n[3] Setting up LoRA...")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 准备数据集
    print("\n[4] Preparing dataset...")
    dataset = InstructionDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    # 训练循环
    print("\n[5] Starting training...")
    model.train()

    global_step = 0
    for epoch in range(config.num_train_epochs):
        epoch_loss = 0
        num_batches = 0

        accumulated_loss = 0
        accumulated_steps = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item() * config.gradient_accumulation_steps
            accumulated_steps += 1

            if accumulated_steps >= config.gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"  Step {global_step} | Loss: {accumulated_loss/accumulated_steps:.4f}")

                epoch_loss += accumulated_loss
                num_batches += 1
                accumulated_loss = 0
                accumulated_steps = 0

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint_dir = Path(config.output_dir) / f"checkpoint-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            print(f"  Saved checkpoint to {checkpoint_dir}")

    # 保存最终模型
    print("\n[6] Saving LoRA adapter...")
    final_dir = Path(config.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print(f"Training complete! Adapter saved to: {final_dir}")
    print("=" * 60)

    # 清理
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
