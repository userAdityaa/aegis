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

    per_episode_path = destination / "per_episode_events.jsonl"
    if per_episode_path.exists():
        rubric_plot = destination / "rubric_components.png"
        per_class_plot = destination / "per_class_accuracy.png"
        cm_plot = destination / "confusion_matrix.png"
        viewer_path = destination / "transcript_viewer.html"
        saved = _save_rubric_components(per_episode_path, rubric_plot)
        if saved is not None:
            artifact_paths["rubric_plot_path"] = str(saved)
        saved = _save_per_class_accuracy(per_episode_path, per_class_plot)
        if saved is not None:
            artifact_paths["per_class_plot_path"] = str(saved)
        saved = _save_confusion_matrix(per_episode_path, cm_plot)
        if saved is not None:
            artifact_paths["confusion_matrix_path"] = str(saved)
        saved = _save_transcript_viewer(per_episode_path, viewer_path)
        if saved is not None:
            artifact_paths["transcript_viewer_path"] = str(saved)

    summary_payload = dict(metrics)
    summary_payload.update(artifact_paths)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return artifact_paths


def _load_episode_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _save_rubric_components(events_path: Path, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    events = _load_episode_events(events_path)
    if not events:
        return None

    components = ["verdict", "speed", "specificity", "evidence"]
    series: dict[str, list[float]] = {name: [] for name in components}
    steps: list[int] = []
    for idx, event in enumerate(events, start=1):
        breakdown = event.get("reward_breakdown", {}) or {}
        for name in components:
            series[name].append(float(breakdown.get(name, 0.0)))
        steps.append(idx)

    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.2))
    for name in components:
        axis.plot(steps, series[name], label=name, linewidth=1.8)
    axis.set_title("Composable Rubrics: Component Scores Over Episodes")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(ncol=4, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_per_class_accuracy(events_path: Path, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    events = _load_episode_events(events_path)
    if not events:
        return None

    by_class: dict[str, list[int]] = {}
    for event in events:
        atk = str(event.get("actual_attack", "unknown"))
        by_class.setdefault(atk, []).append(int(bool(event.get("correct", False))))

    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.2))
    for atk, corrects in sorted(by_class.items()):
        if not corrects:
            continue
        running = []
        total = 0
        for idx, c in enumerate(corrects, start=1):
            total += c
            running.append(total / idx)
        axis.plot(range(1, len(running) + 1), running, label=atk, linewidth=1.6)
    axis.set_title("Per-Attack Running Accuracy (Training Episodes)")
    axis.set_xlabel("Episode index within each attack class")
    axis.set_ylabel("Running accuracy")
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_confusion_matrix(events_path: Path, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    events = _load_episode_events(events_path)
    if not events:
        return None

    attacks = sorted({str(e.get("actual_attack")) for e in events if e.get("actual_attack")})
    if not attacks:
        return None
    index = {name: i for i, name in enumerate(attacks)}
    matrix = [[0 for _ in attacks] for _ in attacks]
    for event in events:
        actual = str(event.get("actual_attack"))
        pred = str(event.get("decision"))
        if actual in index and pred in index:
            matrix[index[actual]][index[pred]] += 1

    figure, axis = plt.subplots(1, 1, figsize=(8.2, 7.2))
    im = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Training Confusion Matrix (from per-episode logs)")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(len(attacks)), attacks, rotation=45, ha="right")
    axis.set_yticks(range(len(attacks)), attacks)
    figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_transcript_viewer(events_path: Path, output_path: Path) -> Path | None:
    events = _load_episode_events(events_path)
    if len(events) < 2:
        return None
    left = events[0]
    tail = events[-10:] if len(events) >= 10 else events[1:]
    right = max(tail, key=lambda e: float(e.get("reward_total", 0.0)))

    html = _render_transcript_viewer(left=left, right=right)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _render_transcript_viewer(*, left: dict[str, Any], right: dict[str, Any]) -> str:
    def panel(event: dict[str, Any], title: str) -> str:
        steps = event.get("observations", []) or []
        steps_html = []
        for step in steps:
            steps_html.append(
                f"<div class='step'><div class='tool'>Tool: <code>{step.get('tool_name')}</code></div>"
                f"<pre class='json'>{json.dumps(step.get('result', {}), indent=2, sort_keys=True)}</pre></div>"
            )
        correct = bool(event.get("correct", False))
        verdict_class = "good" if correct else "bad"
        return (
            f"<div class='panel'>"
            f"<h2>{title}</h2>"
            f"<div class='meta'>attack=<code>{event.get('actual_attack')}</code> "
            f"decision=<code class='{verdict_class}'>{event.get('decision')}</code> "
            f"reward=<code>{event.get('reward_total')}</code></div>"
            f"<h3>Reasoning</h3><div class='reasoning'>{event.get('reasoning','')}</div>"
            f"<h3>Tool transcript</h3>{''.join(steps_html)}"
            f"</div>"
        )

    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Aegis-Env Transcript: Episode 1 vs Final</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, Arial; margin: 0; background: #0b1220; color: #e6edf7; }}
      .wrap {{ padding: 18px; }}
      .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
      .panel {{ background: #101a33; border: 1px solid #223054; border-radius: 12px; padding: 14px; }}
      h1 {{ margin: 0 0 12px 0; font-size: 18px; }}
      h2 {{ margin: 0 0 8px 0; font-size: 16px; }}
      h3 {{ margin: 14px 0 6px 0; font-size: 13px; color: #b9c7e8; }}
      .meta {{ font-size: 12px; color: #b9c7e8; margin-bottom: 10px; }}
      code {{ background: #0b1220; padding: 2px 6px; border-radius: 6px; }}
      code.good {{ color: #1fe38d; }}
      code.bad {{ color: #ff6b6b; }}
      .reasoning {{ background: #0b1220; border-radius: 10px; padding: 10px; white-space: pre-wrap; }}
      .step {{ margin: 10px 0; }}
      .tool {{ font-size: 12px; color: #b9c7e8; margin-bottom: 6px; }}
      pre.json {{ background: #0b1220; border-radius: 10px; padding: 10px; overflow: auto; max-height: 280px; }}
      @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>Episode transcript: Episode 1 vs later episode</h1>
      <div class="grid">
        {panel(left, "Episode 1 (early)") }
        {panel(right, "Later episode (best of last 10)") }
      </div>
    </div>
  </body>
</html>"""


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