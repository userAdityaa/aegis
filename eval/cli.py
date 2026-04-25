from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.classifier_policy import NearestNeighborForensicPolicy
from training.baseline import HeuristicBaselinePolicy, RandomBaselinePolicy

from .plotting import save_evaluation_figure
from .runner import evaluate_policy, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation entrypoint for Aegis-Env.")
    parser.add_argument("--episodes-per-attack", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("heuristic", "random", "checkpoint", "classifier"), default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()

    label = args.label

    if args.policy == "checkpoint":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when --policy=checkpoint")
        from training.model_policy import TransformerTranscriptPolicy

        policy = TransformerTranscriptPolicy(
            args.checkpoint,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        if label is None:
            label = f"TransformerTranscriptPolicy[{args.checkpoint.as_posix()}]"
    elif args.policy == "classifier":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when --policy=classifier")
        policy = NearestNeighborForensicPolicy(args.checkpoint)
        if label is None:
            label = f"NearestNeighborForensicPolicy[{args.checkpoint.as_posix()}]"
    elif args.policy == "random":
        policy = RandomBaselinePolicy(seed=args.seed)
    else:
        policy = HeuristicBaselinePolicy()

    summary, episodes = evaluate_policy(
        policy,
        episodes_per_attack=args.episodes_per_attack,
        seed=args.seed,
    )
    payload = summary.as_dict()
    payload["policy"] = args.policy
    if args.checkpoint is not None:
        payload["checkpoint"] = args.checkpoint.as_posix()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.plot is not None:
        try:
            plot_path = save_evaluation_figure(summary, args.plot)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        payload["plot_path"] = str(plot_path)

    if args.report is not None:
        report_path = write_evaluation_report(summary, episodes, args.report, label=label)
        payload["report_path"] = str(report_path)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()