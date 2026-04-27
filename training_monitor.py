"""
训练监测模板
用于跟踪实验和训练进度
"""
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any


@dataclass
class TrainingConfig:
    """训练配置"""
    base_model: str = ""
    data_path: str = ""
    output_dir: str = ""
    batch_size: int = 1
    gradient_accumulation: int = 16
    max_seq_length: int = 128
    lora_rank: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 1
    num_samples: int = 100


@dataclass
class TrainingMetrics:
    """训练指标"""
    step: int = 0
    loss: float = 0.0
    grad_norm: Optional[float] = None
    learning_rate: float = 0.0
    epoch: int = 0


@dataclass
class EvaluationResult:
    """评估结果"""
    dataset: str = ""
    metric: str = ""
    value: float = 0.0
    num_samples: int = 100


@dataclass
class GenerationSample:
    """生成样本"""
    prompt: str = ""
    response: str = ""
    model_version: str = ""


class TrainingMonitor:
    """
    训练监测类
    用于记录实验配置、训练指标、评估结果
    """

    def __init__(self, experiment_name: str, output_dir: str = "monitoring"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config: Optional[TrainingConfig] = None
        self.metrics: List[TrainingMetrics] = []
        self.evaluations: List[EvaluationResult] = []
        self.samples: List[GenerationSample] = []

    def log_config(self, config: TrainingConfig):
        """记录训练配置"""
        self.config = config
        self._save_json("config.json", asdict(config))

    def log_metrics(self, metrics: TrainingMetrics):
        """记录训练指标"""
        self.metrics.append(metrics)
        self._save_json("metrics.jsonl", asdict(metrics), append=True)

    def log_evaluation(self, evaluation: EvaluationResult):
        """记录评估结果"""
        self.evaluations.append(evaluation)
        self._save_json("evaluations.json", [asdict(e) for e in self.evaluations])

    def log_sample(self, sample: GenerationSample):
        """记录生成样本"""
        self.samples.append(sample)
        self._save_json("samples.json", [asdict(s) for s in self.samples])

    def log_note(self, note: str):
        """记录笔记"""
        note_file = self.output_dir / "notes.md"
        with open(note_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {note}\n")

    def _save_json(self, filename: str, data: Any, append: bool = False):
        """保存 JSON 文件"""
        filepath = self.output_dir / filename
        if append and filepath.exists():
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def save_summary(self):
        """保存实验摘要"""
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": asdict(self.config) if self.config else None,
            "final_metrics": asdict(self.metrics[-1]) if self.metrics else None,
            "evaluations": [asdict(e) for e in self.evaluations],
        }
        self._save_json("summary.json", summary)

    def print_summary(self):
        """打印摘要"""
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 60)

        if self.config:
            print("\n[Config]")
            for k, v in asdict(self.config).items():
                print(f"  {k}: {v}")

        if self.metrics:
            print(f"\n[Metrics] {len(self.metrics)} records")
            latest = self.metrics[-1]
            print(f"  Latest: step={latest.step}, loss={latest.loss:.4f}")

        if self.evaluations:
            print(f"\n[Evaluations]")
            for e in self.evaluations:
                print(f"  {e.dataset}/{e.metric}: {e.value:.4f}")

        if self.samples:
            print(f"\n[Samples] {len(self.samples)} generated")

        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 创建监测器
    monitor = TrainingMonitor(
        experiment_name="qwen2.5-3b-instruct-v3",
        output_dir="monitoring"
    )

    # 记录配置
    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-3B",
        data_path="alpaca_data/alpaca_gpt4_data.jsonl",
        output_dir="Qwen2.5-3B-Instruct-Finetuned-v3",
        batch_size=1,
        gradient_accumulation=16,
        lora_rank=8,
        learning_rate=2e-4,
        num_epochs=1,
        num_samples=100,
    )
    monitor.log_config(config)

    # 记录训练指标
    for step in range(0, 51, 10):
        loss = 4.33 - step * 0.05 + (step % 10) * 0.1
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=2e-4,
            epoch=step // 50,
        )
        monitor.log_metrics(metrics)

    # 记录评估结果
    eval_result = EvaluationResult(
        dataset="HellaSwag",
        metric="acc",
        value=0.52,
        num_samples=100,
    )
    monitor.log_evaluation(eval_result)

    # 记录生成样本
    sample = GenerationSample(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        model_version="v3",
    )
    monitor.log_sample(sample)

    # 记录笔记
    monitor.log_note("Initial training run with 100 samples")
    monitor.log_note("Loss decreasing as expected")

    # 保存并打印摘要
    monitor.save_summary()
    monitor.print_summary()
