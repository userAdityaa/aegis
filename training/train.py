from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.baseline import HeuristicBaselinePolicy
from training.env_client import AegisEnvClient
from training.grpo import GRPOTrainingConfig, build_training_plan, export_training_prompt_dataset, run_grpo_training
from training.rollout import build_rollout_sample
from training.stack import StackStatus, check_training_stack



def run_smoke_training_sample(seed: int = 0) -> dict[str, object]:
    client = AegisEnvClient(seed=seed)
    sample = build_rollout_sample(client, HeuristicBaselinePolicy(), seed=seed)
    return {
        "episode_id": sample.trace.episode_id,
        "target_pkg": sample.trace.target_pkg,
        "actual_attack": sample.trace.actual_attack.value,
        "decision": sample.trace.decision.value,
        "reward": sample.reward,
        "tool_names": sample.trace.tool_names,
        "prompt_preview": sample.prompt[:240],
        "completion": sample.completion,
    }


def build_phase7_plan(config: GRPOTrainingConfig, stack_status: StackStatus) -> dict[str, object]:
    return {
        "stack": asdict(stack_status),
        "ready_for_training": stack_status.ready and (config.report_to != "wandb" or "wandb" not in stack_status.missing_optional),
        "plan": build_training_plan(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 training entrypoint for Aegis-Env.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-stack", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--prepare-dataset", type=Path)
    parser.add_argument("--episodes-per-attack", type=int, default=2)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="artifacts/grpo")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=7)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--run-name", default="aegis-grpo")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--manifest-path")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--force-clone-tool-template", action="store_true")
    parser.add_argument("--tool-template-source", default=None)
    args = parser.parse_args()

    if args.check_stack:
        print(json.dumps(asdict(check_training_stack()), indent=2, sort_keys=True))
        return

    config = GRPOTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        evidence_dir=args.evidence_dir,
        episodes_per_attack=args.episodes_per_attack,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        manifest_path=args.manifest_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_curriculum=bool(args.curriculum),
        force_clone_tool_template=bool(args.force_clone_tool_template),
        tool_template_source=(args.tool_template_source or "Qwen/Qwen3-0.6B"),
    )

    if config.per_device_train_batch_size % config.num_generations != 0:
        raise SystemExit(
            "per_device_train_batch_size must be divisible by num_generations for GRPO. "
            f"Received batch_size={config.per_device_train_batch_size} and num_generations={config.num_generations}."
        )

    if args.prepare_dataset is not None:
        payload = export_training_prompt_dataset(
            args.prepare_dataset,
            episodes_per_attack=config.episodes_per_attack,
            seed=config.seed,
            manifest_path=config.manifest_path,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.dry_run:
        print(json.dumps(build_phase7_plan(config, check_training_stack()), indent=2, sort_keys=True))
        return

    if args.train:
        stack_status = check_training_stack()
        if not stack_status.ready:
            missing = ", ".join(stack_status.missing_required)
            raise SystemExit(f"Missing required training packages: {missing}. Install them with `pip install -e .[training]`.")
        if config.report_to == "wandb" and "wandb" in stack_status.missing_optional:
            raise SystemExit("wandb logging was requested, but wandb is not installed. Install it with `pip install -e .[training]`.")
        print(json.dumps(run_grpo_training(config), indent=2, sort_keys=True))
        return

    print(json.dumps(run_smoke_training_sample(seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()