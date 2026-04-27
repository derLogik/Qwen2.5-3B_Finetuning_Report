# Qwen2.5-3B Base → Instruct 微调报告

## 任务目标

将 Qwen2.5-3B Base 模型微调为 Instruct 模型，提升指令遵循能力。

## 项目概述

| 项目 | 值 |
|------|-----|
| Base 模型 | Qwen/Qwen2.5-3B (3B 参数) |
| 目标 | 指令微调 (Instruction Tuning) |
| 方法 | LoRA (Low-Rank Adaptation) |
| 训练环境 | RTX 3070 Laptop GPU (8GB VRAM) |

## 数据集

| 数据集 | 来源 | 样本数 |
|--------|------|--------|
| Alpaca GPT4 | `tatsu-lab/alpaca` (HuggingFace) | 52,002 |
| 实际训练样本 | 100 | 50 steps |

### 数据格式 (Alpaca)

```json
{
  "text": "<|im_start|>user\nGive three tips for staying healthy.<|im_end|>\n<|im_start|>assistant\n1.Eat a balanced diet..."
}
```

## 训练配置

| 参数 | v2 (失败) | v3 (成功) |
|------|-----------|-----------|
| 数据类型 | 合成模板数据 | 真实 Alpaca 指令 |
| batch_size | 2 | 1 |
| gradient_accumulation | 8 | 16 |
| lora_rank | 32 | 8 |
| max_seq_length | 512 | 128 |
| 训练损失 (final) | 0.76 | 1.23 → ~2.0 |

## 训练结果

### Loss 曲线

```
Step 0:  loss=4.3252
Step 10: loss=2.6930
Step 20: loss=3.1715
Step 30: loss=2.2613
Step 40: loss=1.2316
Step 50: loss=2.0377
```

### 生成效果对比

| Prompt | Base | v2 (合成) | v3 (Alpaca) |
|--------|------|----------|-------------|
| "What is the capital of France?" | The capital of... | The Capital Of France can be explained... | **The capital of France is Paris.** |
| "How does photosynthesis work?" | Photosynthesis is... | Photosynthesis Work works through... | **Photosynthesis is the process by which plants...** |

## 关键发现

### 1. 数据质量至关重要
- **v2 失败原因**: 使用模板生成的合成数据，模型学习到的是"模板格式"而非"指令遵循"
- **v3 成功原因**: 使用真实的 GPT4 生成指令数据

### 2. 训练配置调整
- 减小 batch_size (2 → 1) 避免 OOM
- 增加 gradient_accumulation (8 → 16) 保持有效 batch
- 减小 lora_rank (32 → 8) 减少可训练参数

### 3. 损失函数观察
- v3 损失较高 (~2.0) 但生成效果更好，说明损失不是唯一指标
- v2 损失低 (0.76) 但过拟合到模板格式

## 模型文件

```
qwen_model/
├── Qwen2.5-3B-Instruct-Finetuned-v3/final/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── Qwen2.5-3B-Instruct-v3-Merged/           # 合并后完整模型
│   ├── config.json
│   ├── model-00001-of-00001.safetensors
│   └── tokenizer.json
└── alpaca_data/
    └── alpaca_gpt4_data.jsonl               # 52K 训练数据
```

## 下一步建议

1. **完整训练**: 使用全部 52K 样本训练 1-3 epoch
2. **Benchmark**: 完整 HellaSwag (100 samples) 对比
3. **超参数调优**: 调整 learning_rate, lora_rank, num_epochs
4. **DPO/RLHF**: 使用强化学习进一步对齐

## 结论

✅ 使用真实指令数据微调成功，模型能正确回答问题
⚠️ 仅用 100 样本演示，需完整训练达到最佳效果
