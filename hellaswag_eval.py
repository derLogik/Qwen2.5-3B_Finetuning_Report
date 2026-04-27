"""
独立的 HellaSwag 评估脚本
手动实现 loglikelihood 评估
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gc

def evaluate_hellaswag(model, tokenizer, num_samples=100):
    """评估 HellaSwag"""
    print(f"Loading hellaswag ({num_samples} samples)...")
    dataset = load_dataset("Rowan/hellaswag", split="test")
    if num_samples:
        dataset = dataset.select(range(num_samples))

    correct = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        ctx = item["ctx"]
        label = item["label"]
        endings = item["endings"]

        # 计算每个 ending 的 score (loglikelihood)
        scores = []
        for ending in endings:
            text = f"{ctx} {ending}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            if inputs["input_ids"].shape[1] == 0:
                scores.append(float('-inf'))
                continue

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} | acc={correct/(i+1):.3f}")

    return correct / total

def main():
    print("="*60)
    print("HellaSwag Benchmark")
    print("="*60)

    # Base model
    print("\n[1/2] Evaluating Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
    )
    model.eval()

    base_acc = evaluate_hellaswag(model, tokenizer, 100)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Finetuned model
    print("\n[2/2] Evaluating Finetuned Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "qwen_model/Qwen/Qwen2___5-3B",
        trust_remote_code=True,
    )

    ft_acc = evaluate_hellaswag(model, tokenizer, 100)

    # Results
    print("\n" + "="*60)
    print("RESULTS (HellaSwag 100 samples)")
    print("="*60)
    print(f"Base Model:     acc={base_acc:.4f}")
    print(f"Finetuned v2:   acc={ft_acc:.4f}")
    print(f"Delta:         {ft_acc - base_acc:+.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
