"""
使用 lm_eval 评估模型
"""

import subprocess
import sys
import os

# 模型路径配置
BASE_MODEL = "qwen_model/Qwen/Qwen2___5-3B"
FINETUNED_MODEL = "qwen_model/Qwen2.5-3B-Instruct-Finetuned/final"

def run_eval(model_path, model_name):
    """运行评估"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}\n")

    # 常用 benchmark 任务
    tasks = "hellaswag,arc_challenge"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", "1",
        "--limit", "10",  # 限制样本数快速测试
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def main():
    print("Running benchmark on base model...")
    run_eval(BASE_MODEL, "Base Model (Qwen2.5-3B)")

    print("\n" + "="*80 + "\n")

    print("Running benchmark on finetuned model...")
    run_eval(FINETUNED_MODEL, "Finetuned Model")

if __name__ == "__main__":
    main()
