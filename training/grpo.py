from __future__ import annotations

import locale
import os
from dataclasses import dataclass
from pathlib import Path

from training.dataset import build_training_prompt_rows, summarize_training_prompt_rows, write_training_prompt_jsonl
from training.grpo_env import GRPOAegisEnvironment, aegis_completion_reward_func, aegis_reward_func
from training.reporting import save_training_artifacts

_DEFAULT_TOOL_TEMPLATE_SOURCE = "Qwen/Qwen3-0.6B"


@dataclass(slots=True)
class GRPOTrainingConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "artifacts/grpo"
    evidence_dir: str | None = None
    episodes_per_attack: int = 2
    seed: int = 0
    learning_rate: float = 1e-6
    num_train_epochs: float = 1.0
    max_steps: int = 32
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_generations: int = 2
    max_completion_length: int = 256
    max_tool_calling_iterations: int = 7
    logging_steps: int = 10
    save_steps: int = 50
    warmup_steps: int = 0
    report_to: str = "none"
    run_name: str = "aegis-grpo"
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    # TRL's console completion samples can break on Windows terminals with non-UTF8 codepages.
    log_completions: bool = False
    manifest_path: str | None = None
    resume_from_checkpoint: str | bool | None = None
    use_curriculum: bool = False
    force_clone_tool_template: bool = True
    tool_template_source: str = _DEFAULT_TOOL_TEMPLATE_SOURCE


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
        "evidence_dir": config.evidence_dir,
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
    _force_utf8_locale_for_trl_import()
    try:
        from datasets import Dataset
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from trl.chat_template_utils import clone_chat_template, supports_tool_calling
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GRPO training requires the training extra. Install it with `pip install -e .[training]`."
        ) from exc

    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())

    rows = build_training_prompt_rows(
        episodes_per_attack=config.episodes_per_attack,
        seed=config.seed,
        manifest_path=config.manifest_path,
    )
    dataset = Dataset.from_list([row.as_dict() for row in rows])

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    cloned_template = False
    should_clone = config.force_clone_tool_template or (not supports_tool_calling(tokenizer))
    if should_clone:
        model, tokenizer, _ = clone_chat_template(model, tokenizer, config.tool_template_source)
        cloned_template = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

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
        use_cpu=not has_cuda,
        bf16=use_bf16,
        fp16=False,
    )

    def _env_factory() -> GRPOAegisEnvironment:
        env = GRPOAegisEnvironment(
            manifest_path=config.manifest_path,
            evidence_dir=config.evidence_dir or config.output_dir,
            run_id=config.run_name,
        )
        if config.use_curriculum:
            env._enable_curriculum(seed=config.seed)
        return env

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[aegis_reward_func, aegis_completion_reward_func],
        processing_class=tokenizer,
        environment_factory=_env_factory,
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
    metrics["log_history_entries"] = len(trainer.state.log_history)
    metrics["tool_template_source"] = config.tool_template_source if cloned_template else config.model_name

    artifact_paths = save_training_artifacts(
        metrics=metrics,
        log_history=list(trainer.state.log_history),
        output_dir=config.evidence_dir or output_dir,
    )
    metrics.update(artifact_paths)
    return metrics


def _force_utf8_locale_for_trl_import() -> None:
    if os.name != "nt":
        return

    if hasattr(locale, "getencoding"):
        locale.getencoding = lambda: "utf-8"  # type: ignore[assignment]
    if hasattr(locale, "getpreferredencoding"):
        locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"  # type: ignore[assignment]
    if getattr(Path.read_text, "__name__", "") != "_aegis_read_text_utf8":
        original_read_text = Path.read_text

        def _aegis_read_text_utf8(self, encoding=None, errors=None, newline=None):
            resolved_encoding = "utf-8" if encoding is None else encoding
            # Python 3.10's Path.read_text does not accept newline=.
            return original_read_text(self, encoding=resolved_encoding, errors=errors)

        Path.read_text = _aegis_read_text_utf8  # type: ignore[assignment]