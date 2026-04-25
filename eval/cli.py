from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plotting import save_evaluation_figure
from .runner import evaluate_baseline, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation entrypoint for Aegis-Env.")
    parser.add_argument("--episodes-per-attack", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()

    summary, episodes = evaluate_baseline(
        episodes_per_attack=args.episodes_per_attack,
        seed=args.seed,
    )
    payload = summary.as_dict()

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
        report_path = write_evaluation_report(summary, episodes, args.report, label=args.label)
        payload["report_path"] = str(report_path)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()