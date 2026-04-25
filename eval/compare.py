from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .plot import save_comparison_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and trained Aegis evaluation runs.")
    baseline_source = parser.add_mutually_exclusive_group(required=True)
    baseline_source.add_argument("--baseline-report", type=Path)
    baseline_source.add_argument("--baseline-run")

    trained_source = parser.add_mutually_exclusive_group(required=True)
    trained_source.add_argument("--trained-report", type=Path)
    trained_source.add_argument("--trained-run")

    parser.add_argument("--baseline-label")
    parser.add_argument("--trained-label")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_report = _load_report(
        report_path=args.baseline_report,
        run_path=args.baseline_run,
        label_override=args.baseline_label,
    )
    trained_report = _load_report(
        report_path=args.trained_report,
        run_path=args.trained_run,
        label_override=args.trained_label,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot or output_dir / "comparison.png"
    json_path = args.output or output_dir / "comparison.json"

    try:
        rendered_plot = save_comparison_figure(baseline_report, trained_report, plot_path)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    payload = build_comparison_payload(baseline_report, trained_report)
    payload["artifacts"] = {
        "plot_path": str(rendered_plot),
        "comparison_path": str(json_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_comparison_payload(
    baseline_report: Mapping[str, object],
    trained_report: Mapping[str, object],
) -> dict[str, object]:
    baseline_summary = _summary(baseline_report)
    trained_summary = _summary(trained_report)
    return {
        "baseline": {
            "label": _label(baseline_report),
            "summary": baseline_summary,
            "episodes_with_trace": len(_episodes(baseline_report)),
        },
        "trained": {
            "label": _label(trained_report),
            "summary": trained_summary,
            "episodes_with_trace": len(_episodes(trained_report)),
        },
        "delta": {
            "accuracy": _metric(trained_summary, "accuracy") - _metric(baseline_summary, "accuracy"),
            "average_reward": _metric(trained_summary, "average_reward") - _metric(baseline_summary, "average_reward"),
            "average_steps": _metric(trained_summary, "average_steps") - _metric(baseline_summary, "average_steps"),
        },
    }


def _load_report(
    *,
    report_path: Path | None,
    run_path: str | None,
    label_override: str | None,
) -> dict[str, object]:
    if report_path is not None:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        return _normalize_report(payload, label_override=label_override)
    if run_path is None:
        raise ValueError("Expected either a report path or a Weights & Biases run path.")
    payload = _load_wandb_run(run_path)
    return _normalize_report(payload, label_override=label_override)


def _normalize_report(payload: Mapping[str, object], *, label_override: str | None = None) -> dict[str, object]:
    summary_payload = payload.get("summary")
    if isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
        episodes = _coerce_episode_list(payload.get("episodes"))
        reward_curve = _coerce_float_list(payload.get("reward_curve"))
    else:
        summary = dict(payload)
        episodes = []
        reward_curve = []

    if not reward_curve and episodes:
        reward_curve = [float(episode.get("reward", 0.0)) for episode in episodes]

    label = label_override or payload.get("label") or summary.get("policy_name") or "run"
    normalized: dict[str, object] = {
        "schema_version": payload.get("schema_version", 1),
        "label": str(label),
        "summary": summary,
        "episodes": episodes,
        "reward_curve": reward_curve,
    }
    source = payload.get("source")
    if isinstance(source, Mapping):
        normalized["source"] = dict(source)
    return normalized


def _load_wandb_run(run_path: str) -> dict[str, object]:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb is required for run-based comparison. Install it with `pip install -e .[training]`."
        ) from exc

    run = wandb.Api().run(run_path)
    report = _download_report_artifact(run)
    if report is not None:
        report.setdefault("source", {"type": "wandb", "run": run_path, "partial": False})
        return report

    summary = dict(run.summary)
    history = list(run.scan_history(page_size=500))
    reward_curve = _extract_reward_curve(history)
    return {
        "schema_version": 1,
        "label": run.name or run.id,
        "summary": {
            "policy_name": run.name or run.id,
            "episodes": int(summary.get("episodes", len(reward_curve))),
            "accuracy": _coerce_float(summary.get("accuracy")),
            "average_reward": _coerce_float(summary.get("average_reward", summary.get("aegis/reward_mean"))),
            "average_steps": _coerce_float(summary.get("average_steps")),
            "decision_counts": _extract_prefixed_counts(summary, "decision_counts/"),
            "tool_usage_counts": _extract_prefixed_counts(summary, "tool_usage_counts/"),
            "confusion_matrix": {},
            "per_attack": _extract_per_attack(summary),
        },
        "episodes": [],
        "reward_curve": reward_curve,
        "source": {"type": "wandb", "run": run_path, "partial": True},
    }


def _download_report_artifact(run: Any) -> dict[str, object] | None:
    candidate_names = {
        "evaluation_report.json",
        "baseline_report.json",
        "trained_report.json",
        "aegis_evaluation_report.json",
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="aegis-wandb-"))

    for remote_file in run.files():
        if Path(remote_file.name).name not in candidate_names:
            continue
        remote_file.download(root=str(temp_dir), replace=True)
        local_path = temp_dir / remote_file.name
        if local_path.exists():
            return json.loads(local_path.read_text(encoding="utf-8"))
    return None


def _extract_reward_curve(history: list[Mapping[str, object]]) -> list[float]:
    metric_names = (
        "aegis/reward_mean",
        "average_reward",
        "reward",
        "episode_reward",
        "eval/reward",
    )
    reward_curve: list[float] = []
    for row in history:
        for metric_name in metric_names:
            value = row.get(metric_name)
            if isinstance(value, (int, float)):
                reward_curve.append(float(value))
                break
    return reward_curve


def _extract_per_attack(summary: Mapping[str, object]) -> dict[str, dict[str, float]]:
    per_attack: dict[str, dict[str, float]] = {}
    prefixes = ("per_attack/", "evaluation/per_attack/")
    for key, value in summary.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            continue
        for prefix in prefixes:
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            attack_name, _, metric_name = remainder.partition("/")
            if not attack_name or not metric_name:
                continue
            per_attack.setdefault(attack_name, {})[metric_name] = float(value)
    return per_attack


def _extract_prefixed_counts(summary: Mapping[str, object], prefix: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in summary.items():
        if isinstance(key, str) and key.startswith(prefix) and isinstance(value, (int, float)):
            counts[key[len(prefix):]] = int(value)
    return counts


def _summary(report: Mapping[str, object]) -> dict[str, object]:
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return {}


def _episodes(report: Mapping[str, object]) -> list[dict[str, object]]:
    return _coerce_episode_list(report.get("episodes"))


def _label(report: Mapping[str, object]) -> str:
    label = report.get("label")
    if isinstance(label, str) and label.strip():
        return label
    summary = _summary(report)
    policy_name = summary.get("policy_name")
    if isinstance(policy_name, str) and policy_name.strip():
        return policy_name
    return "run"


def _metric(summary: Mapping[str, object], key: str) -> float:
    return _coerce_float(summary.get(key))


def _coerce_episode_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _coerce_float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, (int, float))]


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


if __name__ == "__main__":
    main()