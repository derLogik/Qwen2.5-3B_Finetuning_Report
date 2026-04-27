"""
评估单个模型 (避免显存不足)
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def load_and_eval(model_path, name, num_samples=100):
    """加载模型并评估"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    try:
        print("Loading model...")
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
        print("Model loaded!")

        print("Loading hellaswag...")
        dataset = load_dataset("Rowan/hellaswag", split="test")
        dataset = dataset.select(range(num_samples))

        correct = 0
        total = len(dataset)

        for i, item in enumerate(dataset):
            ctx = item["ctx"]
            label = item["label"]
            endings = item["endings"]

            scores = []
            for ending in endings:
                text = f"{ctx} {ending}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)

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
                print(f"  {i+1}/{total} | Acc: {correct/(i+1):.3f}")

        acc = correct / total
        print(f"\nResult: acc = {acc:.4f}")

        # 清理
        del model, tokenizer
        del dataset
        gc.collect()
        torch.cuda.empty_cache()

        return acc

    except Exception as e:
        print(f"Error: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return 0

def main():
    print("HellaSwag Benchmark (100 samples)")

    # Base model
    base_acc = load_and_eval("qwen_model/Qwen/Qwen2___5-3B", "Base Model (Qwen2.5-3B)", 100)

    gc.collect()
    torch.cuda.empty_cache()

    # Finetuned model
    ft_acc = load_and_eval("qwen_model/Qwen2.5-3B-Instruct-Finetuned-v2/final", "Finetuned v2 (LoRA)", 100)

    # 结果对比
    print("\n" + "="*60)
    print("BENCHMARK RESULTS (HellaSwag 100 samples)")
    print("="*60)
    print(f"Base Model:     {base_acc:.4f}")
    print(f"Finetuned v2:   {ft_acc:.4f}")
    print(f"Delta:          {ft_acc - base_acc:+.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
