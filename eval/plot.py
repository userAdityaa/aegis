from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from environment.models import AttackClass

_TOOL_NAMES = (
    "check_maintainer_history",
    "diff_versions",
    "inspect_install_script",
    "get_reputation_score",
    "trace_dependencies",
    "run_sandbox_test",
)


def save_comparison_figure(
    baseline_report: Mapping[str, object],
    trained_report: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for comparison plotting. Install the eval extra with `pip install -e .[eval]`."
        ) from exc

    figure, axes = plt.subplots(3, 1, figsize=(14, 18))

    _plot_reward_curve(axes[0], baseline_report, trained_report)
    _plot_accuracy_bars(axes[1], baseline_report, trained_report)
    _plot_tool_heatmap(axes[2], trained_report)

    figure.tight_layout(h_pad=2.5)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def _plot_reward_curve(axis, baseline_report: Mapping[str, object], trained_report: Mapping[str, object]) -> None:
    baseline_curve = _reward_curve(baseline_report)
    trained_curve = _reward_curve(trained_report)

    if baseline_curve:
        axis.plot(range(1, len(baseline_curve) + 1), baseline_curve, linestyle="--", color="#c44e52", linewidth=2, label=_label(baseline_report))
    if trained_curve:
        axis.plot(range(1, len(trained_curve) + 1), trained_curve, linestyle="-", color="#2c7fb8", linewidth=2, label=_label(trained_report))

    axis.set_title("Reward Curve Over Episodes")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.axhline(0.0, color="#999999", linewidth=1, alpha=0.5)
    if baseline_curve or trained_curve:
        axis.legend(loc="best")
    else:
        axis.text(0.5, 0.5, "No reward history available.", ha="center", va="center", transform=axis.transAxes)


def _plot_accuracy_bars(axis, baseline_report: Mapping[str, object], trained_report: Mapping[str, object]) -> None:
    attack_names = [attack.value for attack in AttackClass]
    baseline_accuracy = [_per_attack_accuracy(baseline_report, attack_name) for attack_name in attack_names]
    trained_accuracy = [_per_attack_accuracy(trained_report, attack_name) for attack_name in attack_names]
    positions = list(range(len(attack_names)))
    width = 0.38

    axis.bar([position - width / 2 for position in positions], baseline_accuracy, width=width, color="#f28e2b", label=_label(baseline_report))
    axis.bar([position + width / 2 for position in positions], trained_accuracy, width=width, color="#4e79a7", label=_label(trained_report))
    axis.set_title("Per-Attack Accuracy")
    axis.set_xticks(positions, attack_names, rotation=45, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.legend(loc="best")


def _plot_tool_heatmap(axis, trained_report: Mapping[str, object]) -> None:
    matrix = _tool_usage_matrix(trained_report)
    attack_names = [attack.value for attack in AttackClass]
    image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto")
    axis.set_title(f"Tool Usage Frequency Heatmap ({_label(trained_report)})")
    axis.set_xticks(range(len(_TOOL_NAMES)), _TOOL_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(len(attack_names)), attack_names)
    axis.set_xlabel("Tool")
    axis.set_ylabel("Actual Attack")

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="#111111")

    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if not _episodes(trained_report):
        axis.text(0.5, -0.18, "Episode-level report data was not available; heatmap cells default to 0.", ha="center", va="top", transform=axis.transAxes)


def _reward_curve(report: Mapping[str, object]) -> list[float]:
    curve = report.get("reward_curve")
    if isinstance(curve, Sequence) and not isinstance(curve, (str, bytes)):
        return [float(value) for value in curve]

    rewards: list[float] = []
    for episode in _episodes(report):
        value = episode.get("reward")
        if isinstance(value, (int, float)):
            rewards.append(float(value))
    return rewards


def _per_attack_accuracy(report: Mapping[str, object], attack_name: str) -> float:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return 0.0
    per_attack = summary.get("per_attack")
    if not isinstance(per_attack, Mapping):
        return 0.0
    metrics = per_attack.get(attack_name)
    if not isinstance(metrics, Mapping):
        return 0.0
    value = metrics.get("accuracy")
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _tool_usage_matrix(report: Mapping[str, object]) -> list[list[float]]:
    attack_names = [attack.value for attack in AttackClass]
    episodes = _episodes(report)
    if not episodes:
        return [[0.0 for _ in _TOOL_NAMES] for _ in attack_names]

    matrix: list[list[float]] = []
    for attack_name in attack_names:
        matching = [episode for episode in episodes if episode.get("actual_attack") == attack_name]
        row: list[float] = []
        for tool_name in _TOOL_NAMES:
            count = sum(sum(1 for observed in episode.get("tool_names", []) if observed == tool_name) for episode in matching)
            row.append(count / max(1, len(matching)))
        matrix.append(row)
    return matrix


def _episodes(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    episodes = report.get("episodes")
    if isinstance(episodes, Sequence) and not isinstance(episodes, (str, bytes)):
        return [episode for episode in episodes if isinstance(episode, Mapping)]
    return []


def _label(report: Mapping[str, object]) -> str:
    label = report.get("label")
    if isinstance(label, str) and label.strip():
        return label
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        policy_name = summary.get("policy_name")
        if isinstance(policy_name, str) and policy_name.strip():
            return policy_name
    return "run"