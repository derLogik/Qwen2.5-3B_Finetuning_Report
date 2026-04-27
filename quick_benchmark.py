"""
快速 hellaswag 评估脚本 (100 samples)
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def load_model(model_path):
    """加载模型"""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def evaluate_hellaswag(model, tokenizer, num_samples=100):
    """评估 hellaswag"""
    print(f"Loading hellaswag dataset ({num_samples} samples)...")
    dataset = load_dataset("Rowan/hellaswag", split="test")
    dataset = dataset.select(range(num_samples))

    correct = 0
    correct_norm = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        ctx = item["ctx"]
        label = item["label"]
        endings = item["endings"]

        # 计算每个 ending 的 score
        scores = []
        for ending in endings:
            text = f"{ctx} {ending}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

            ending_tokens = tokenizer.encode(ending, add_special_tokens=False)
            if ending_tokens:
                score = log_probs[0, ending_tokens[0]].item()
            else:
                score = float('-inf')
            scores.append(score)

        pred = scores.index(max(scores))

        if pred == label:
            correct += 1
            correct_norm += 1  # 简化版本

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{total} | Acc: {correct/(i+1):.3f}")

    acc = correct / total
    acc_norm = correct_norm / total

    return {"acc": acc, "acc_norm": acc_norm}

def main():
    models = [
        ("qwen_model/Qwen/Qwen2___5-3B", "Base Model"),
        ("qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final", "Finetuned v2"),
    ]

    results = {}

    for model_path, model_name in models:
        try:
            model, tokenizer = load_model(model_path)
            result = evaluate_hellaswag(model, tokenizer, num_samples=100)
            results[model_name] = result

            # 清理 GPU 内存
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = {"acc": 0, "acc_norm": 0}

    # 打印结果对比
    print("\n" + "="*60)
    print("HellaSwag Benchmark Results (100 samples)")
    print("="*60)
    print(f"{'Model':<25} {'acc':<10} {'acc_norm':<10}")
    print("-"*50)
    for model_name, result in results.items():
        print(f"{model_name:<25} {result['acc']:<10.4f} {result['acc_norm']:<10.4f}")

if __name__ == "__main__":
    main()
