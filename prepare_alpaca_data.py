"""
准备 Alpaca GPT4 指令数据用于微调
从 HuggingFace 下载并转换为 Qwen 格式
"""

from datasets import load_dataset
import json
from pathlib import Path


def download_and_prepare_alpaca_data():
    print("=" * 60)
    print("Downloading Alpaca GPT4 Data")
    print("=" * 60)

    # 加载数据集 - 尝试多个可能的数据集名称
    print("\n[1] Loading dataset from HuggingFace...")
    dataset = None

    dataset_names = [
        "yahi-alpaca/Alpaca-data",
        "yahi-alpaca/alpaca_data",
        "llama-org/alpaca-data",
        "tatsu-lab/alpaca",
        "V掌教育/Alpaca中文数据",
    ]

    for name in dataset_names:
        try:
            print(f"    Trying: {name}")
            dataset = load_dataset(name, split="train")
            print(f"    Successfully loaded: {name}")
            break
        except Exception as e:
            print(f"    Failed: {str(e)[:60]}")

    if dataset is None:
        print("\n[!] Could not load any dataset from HuggingFace.")
        print("    Please check your network connection or dataset name.")
        return None

    print(f"    Total samples: {len(dataset)}")

    # 查看数据格式
    print("\n[2] Checking data format...")
    print(f"    Features: {dataset.features}")
    sample = dataset[0]
    print(f"    Sample keys: {list(sample.keys())}")

    # 转换数据格式
    print("\n[3] Converting to Qwen format...")

    def convert_to_qwen_format(sample):
        """转换为 Qwen 指令微调格式"""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        # 构建对话格式
        if input_text:
            text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

        return {"text": text}

    # 转换所有数据
    converted_data = []
    for i, sample in enumerate(dataset):
        try:
            converted = convert_to_qwen_format(sample)
            converted_data.append(converted)
        except Exception as e:
            print(f"    Error at sample {i}: {e}")

        if (i + 1) % 10000 == 0:
            print(f"    Processed {i+1}/{len(dataset)} samples...")

    print(f"    Converted {len(converted_data)} samples")

    # 保存为 JSONL 格式
    output_dir = Path("qwen_model/alpaca_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "alpaca_gpt4_data.jsonl"

    print(f"\n[4] Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"    Saved {len(converted_data)} samples")

    # 保存一份用于训练日志
    print("\n[5] Sample data preview:")
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line)
            print(f"\n    Sample {i+1}:")
            print(f"    Text: {sample['text'][:200]}...")

    print("\n" + "=" * 60)
    print(f"Data preparation complete!")
    print(f"Output: {output_file}")
    print("=" * 60)

    return str(output_file)


if __name__ == "__main__":
    download_and_prepare_alpaca_data()
