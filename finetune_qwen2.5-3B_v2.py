"""
Qwen2.5-3B Base → Instruct 微调脚本
优化配置: batch=2, seq=512, 适应 8GB VRAM
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")

# ============ 配置 ============
@dataclass
class FinetuneConfig:
    # 模型路径
    base_model_path: str = "qwen_model/Qwen/Qwen2___5-3B"
    output_dir: str = "qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2"

    # 训练参数 - 针对 8GB VRAM 优化
    seed: int = 42
    batch_size: int = 2              # 增大 batch size
    gradient_accumulation_steps: int = 8  # 有效 batch = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = True

    # LoRA 配置
    lora_rank: int = 32              # 维持 32 平衡效果/显存
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # 数据集配置
    train_dataset_path: str = "qwen_model/data/train.jsonl"
    max_seq_length: int = 512        # 增长序列长度

    # 日志与保存
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3


def setup_model_and_tokenizer(config: FinetuneConfig):
    """加载模型和分词器"""
    print(f"Loading tokenizer from {config.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {config.base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer


def preprocess_function(examples, tokenizer, max_seq_length: int):
    """预处理数据集"""
    texts = []
    for prompt, response in zip(examples["prompt"], examples["response"]):
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        texts.append(text)

    model_inputs = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        return_tensors=None,
    )

    return model_inputs


def setup_dataset(config: FinetuneConfig, tokenizer):
    """加载并预处理数据集"""
    print(f"Loading dataset from {config.train_dataset_path}...")

    dataset = load_dataset("json", data_files=config.train_dataset_path, split="train")

    print(f"Dataset size: {len(dataset)}")

    def preprocess(examples):
        return preprocess_function(examples, tokenizer, config.max_seq_length)

    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset",
    )

    split_dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)

    return split_dataset["train"], split_dataset["test"]


def setup_lora_config(config: FinetuneConfig) -> LoraConfig:
    """配置 LoRA"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        inference_mode=False,
    )


def main():
    config = FinetuneConfig()

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    model, tokenizer = setup_model_and_tokenizer(config)
    train_dataset, eval_dataset = setup_dataset(config, tokenizer)

    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Total training steps: ~{len(train_dataset) // config.batch_size // config.gradient_accumulation_steps * config.num_train_epochs}")

    trainer.train()

    print(f"Saving final model to {config.output_dir}/final")
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")

    print("Training complete!")


if __name__ == "__main__":
    main()
