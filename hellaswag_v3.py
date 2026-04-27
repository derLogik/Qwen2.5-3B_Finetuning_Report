"""
HellaSwag Benchmark - 测试 Base vs v3 Finetuned 模型
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

def evaluate_hellaswag(model, tokenizer, num_samples=100):
    """评估 HellaSwag"""
    print(f"Evaluating on {num_samples} samples...")
    dataset = load_dataset("Rowan/hellaswag", split="test")
    if num_samples:
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
    print("=" * 60)
    print("HellaSwag Benchmark - Base vs Finetuned v3")
    print("=" * 60)

    model_path = "Qwen/Qwen2___5-3B"
    adapter_path = "Qwen2.5-3B-Instruct-Finetuned-v3/final"

    # Base model
    print("\n[1/2] Evaluating Base Model...")
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model.eval()

    base_acc = evaluate_hellaswag(base_model, tokenizer, 100)
    print(f"\nBase Model Accuracy: {base_acc:.4f}")

    del base_model
    torch.cuda.empty_cache()

    # Finetuned model
    print("\n[2/2] Evaluating Finetuned v3 Model...")
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ft_acc = evaluate_hellaswag(model, tokenizer, 100)
    print(f"\nFinetuned v3 Accuracy: {ft_acc:.4f}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS (HellaSwag 100 samples)")
    print("=" * 60)
    print(f"Base Model:     acc={base_acc:.4f}")
    print(f"Finetuned v3:   acc={ft_acc:.4f}")
    print(f"Delta:         {ft_acc - base_acc:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
