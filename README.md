# Qwen2.5-3B Finetuning Report

将 Qwen2.5-3B Base 模型微调为 Instruct 模型的实验报告。

## 项目概述

| 项目 | 值 |
|------|-----|
| Base 模型 | Qwen/Qwen2.5-3B (3B 参数) |
| 目标 | 指令微调 (Instruction Tuning) |
| 方法 | LoRA (Low-Rank Adaptation) |
| 训练环境 | RTX 3070 Laptop GPU (8GB VRAM) |

## 实验结果

### 关键发现

1. **数据质量至关重要** - 使用真实 GPT4 生成的 Alpaca 指令数据，远优于模板合成的数据
2. **训练配置** - batch_size=1, gradient_accumulation=16, lora_rank=8, max_seq_length=128
3. **损失函数** - 损失值不是唯一指标，v3 损失较高但生成效果更好

### 生成效果对比

| Prompt | Base | v2 (合成) | v3 (Alpaca) |
|--------|------|----------|-------------|
| "What is the capital of France?" | The capital of... | The Capital Of France can be explained... | **The capital of France is Paris.** |
| "How does photosynthesis work?" | Photosynthesis is... | Photosynthesis Work works through... | **Photosynthesis is the process by which plants...** |

## 文件结构

```
Qwen2.5-3B_Finetuning_Report/
├── REPORT.md                          # 完整实验报告
├── finetune_qwen2.5-3B.py             # v1 基础训练脚本
├── finetune_qwen2.5-3B_v2.py          # v2 训练脚本 (失败)
├── finetune_qwen2.5-3B_v3.py          # v3 训练脚本 (成功)
├── merge_lora.py                      # LoRA 权重合并
├── merge_lora_v3.py                   # v3 版本合并
├── eval_*.py                           # 评估脚本
├── test_*.py                           # 测试脚本
├── hellaswag_*.py                      # HellaSwag 基准测试
├── generate_dataset.py                 # 数据集生成
├── prepare_alpaca_data.py              # Alpaca 数据预处理
├── quick_*.py                          # 快速测试脚本
├── training_monitor.py                 # 训练监控
├── run_benchmark.py                     # 基准测试运行
├── train.jsonl                         # 训练数据
├── train_part_*                        # 分片训练日志
└── monitoring/                          # 训练监控数据
```

## 训练配置 (v3 成功配置)

```python
TrainingArguments:
  output_dir="./output"
  per_device_train_batch_size=1
  gradient_accumulation_steps=16
  learning_rate=2e-4
  num_train_epochs=1
  fp16=True
  logging_steps=10
  save_steps=50

LoraConfig:
  r=8
  lora_alpha=16
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout=0.05
  bias="none"
  task_type="CAUSAL_LM"
```

## 结论

- ✅ 使用真实指令数据微调成功，模型能正确回答问题
- ⚠️ 仅用 100 样本演示，需完整训练达到最佳效果
- 📝 建议：使用全部 52K 样本训练 1-3 epoch

## 参考资料

- [Qwen2.5-3B Base](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
