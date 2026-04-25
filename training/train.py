from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

from training.baseline import HeuristicBaselinePolicy
from training.env_client import AegisEnvClient
from training.rollout import build_rollout_sample


@dataclass(slots=True)
class StackStatus:
    installed: dict[str, str]
    missing: list[str]


def check_training_stack() -> StackStatus:
    installed: dict[str, str] = {}
    missing: list[str] = []
    for package_name in ("trl", "transformers", "torch", "wandb", "unsloth"):
        try:
            installed[package_name] = version(package_name)
        except PackageNotFoundError:
            missing.append(package_name)
    return StackStatus(installed=installed, missing=missing)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 training entrypoint for Aegis-Env.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-stack", action="store_true")
    args = parser.parse_args()

    if args.check_stack:
        print(json.dumps(asdict(check_training_stack()), indent=2, sort_keys=True))
        return

    print(json.dumps(run_smoke_training_sample(seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()