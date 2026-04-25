from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from training.dataset import build_training_prompt_rows, summarize_training_prompt_rows, write_training_prompt_jsonl
from training.grpo_env import GRPOAegisEnvironment, aegis_reward_func


@dataclass(slots=True)
class GRPOTrainingConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "artifacts/grpo"
    episodes_per_attack: int = 2
    seed: int = 0
    learning_rate: float = 1e-6
    num_train_epochs: float = 1.0
    max_steps: int = 32
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_completion_length: int = 256
    max_tool_calling_iterations: int = 7
    logging_steps: int = 10
    save_steps: int = 50
    warmup_steps: int = 0
    report_to: str = "none"
    run_name: str = "aegis-grpo"
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    log_completions: bool = True
    manifest_path: str | None = None
    resume_from_checkpoint: str | bool | None = None


def build_training_plan(config: GRPOTrainingConfig) -> dict[str, object]:
    rows = build_training_prompt_rows(
        episodes_per_attack=config.episodes_per_attack,
        seed=config.seed,
        manifest_path=config.manifest_path,
    )
    summary = summarize_training_prompt_rows(rows)
    return {
        "model_name": config.model_name,
        "output_dir": config.output_dir,
        "episodes_per_attack": config.episodes_per_attack,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "num_generations": config.num_generations,
        "max_completion_length": config.max_completion_length,
        "max_tool_calling_iterations": config.max_tool_calling_iterations,
        "dataset": summary,
        "use_vllm": config.use_vllm,
        "vllm_mode": config.vllm_mode,
        "report_to": config.report_to,
    }


def export_training_prompt_dataset(
    output_path: str | Path,
    *,
    episodes_per_attack: int,
    seed: int,
    manifest_path: str | None = None,
) -> dict[str, object]:
    rows = build_training_prompt_rows(
        episodes_per_attack=episodes_per_attack,
        seed=seed,
        manifest_path=manifest_path,
    )
    destination = write_training_prompt_jsonl(rows, output_path)
    return {
        "dataset_path": str(destination),
        **summarize_training_prompt_rows(rows),
    }


def run_grpo_training(config: GRPOTrainingConfig) -> dict[str, object]:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GRPO training requires the training extra. Install it with `pip install -e .[training]`."
        ) from exc

    rows = build_training_prompt_rows(
        episodes_per_attack=config.episodes_per_attack,
        seed=config.seed,
        manifest_path=config.manifest_path,
    )
    dataset = Dataset.from_list([row.as_dict() for row in rows])

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        report_to=config.report_to,
        run_name=config.run_name,
        remove_unused_columns=False,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        max_tool_calling_iterations=config.max_tool_calling_iterations,
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        log_completions=config.log_completions,
    )

    trainer = GRPOTrainer(
        model=config.model_name,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=aegis_reward_func,
        environment_factory=lambda: GRPOAegisEnvironment(manifest_path=config.manifest_path),
    )
    train_output = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))

    metrics = dict(train_output.metrics)
    metrics["global_step"] = train_output.global_step
    metrics["training_loss"] = train_output.training_loss
    metrics["output_dir"] = str(output_dir)
    metrics["episodes"] = len(rows)
    return metrics