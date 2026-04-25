from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def save_training_artifacts(
    *,
    metrics: dict[str, object],
    log_history: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_path = destination / "training_summary.json"
    history_path = destination / "training_log_history.json"
    history_jsonl_path = destination / "training_log_history.jsonl"
    history_path.write_text(json.dumps(list(log_history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with history_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in log_history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    artifact_paths = {
        "summary_path": str(summary_path),
        "history_path": str(history_path),
        "history_jsonl_path": str(history_jsonl_path),
    }

    plot_path = destination / "training_curves.png"
    saved_plot = _save_training_curves(log_history, plot_path)
    if saved_plot is not None:
        artifact_paths["plot_path"] = str(saved_plot)

    summary_payload = dict(metrics)
    summary_payload.update(artifact_paths)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return artifact_paths


def _save_training_curves(
    log_history: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    loss_points = _metric_points(log_history, "loss")
    reward_points = _metric_points(log_history, "aegis/reward_mean")

    if not loss_points and not reward_points:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_series(axes[0], loss_points, title="Training Loss", ylabel="Loss", color="#b54708")
    _plot_series(axes[1], reward_points, title="Mean Reward", ylabel="Reward", color="#175cd3")
    figure.suptitle("Aegis GRPO Training Curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _metric_points(
    log_history: Sequence[dict[str, Any]],
    metric_name: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in log_history:
        value = row.get(metric_name)
        if not isinstance(value, (int, float)):
            continue
        step = row.get("step", len(points) + 1)
        if not isinstance(step, (int, float)):
            step = len(points) + 1
        points.append((float(step), float(value)))
    return points


def _plot_series(axis, points: Sequence[tuple[float, float]], *, title: str, ylabel: str, color: str) -> None:
    if points:
        axis.plot([point[0] for point in points], [point[1] for point in points], color=color, linewidth=2)
        axis.scatter([point[0] for point in points], [point[1] for point in points], color=color, s=18)
    else:
        axis.text(0.5, 0.5, "No data logged", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25, linestyle=":")