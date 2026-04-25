from __future__ import annotations

from pathlib import Path

from environment.models import AttackClass

from .runner import EvaluationSummary


def save_evaluation_figure(summary: EvaluationSummary, output_path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install the eval extra with `pip install -e .[eval]`."
        ) from exc

    attack_names = [attack.value for attack in AttackClass]
    confusion = [
        [summary.confusion_matrix.get(actual_name, {}).get(predicted_name, 0) for predicted_name in attack_names]
        for actual_name in attack_names
    ]
    per_attack_accuracy = [summary.per_attack.get(attack_name).accuracy if attack_name in summary.per_attack else 0.0 for attack_name in attack_names]

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    bar_axis = axes[0]
    bar_axis.bar(range(len(attack_names)), per_attack_accuracy, color="#31688e")
    bar_axis.set_title("Per-Attack Accuracy")
    bar_axis.set_xticks(range(len(attack_names)), attack_names, rotation=45, ha="right")
    bar_axis.set_ylim(0.0, 1.0)
    bar_axis.set_ylabel("Accuracy")

    heatmap_axis = axes[1]
    image = heatmap_axis.imshow(confusion, cmap="Blues")
    heatmap_axis.set_title("Confusion Matrix")
    heatmap_axis.set_xticks(range(len(attack_names)), attack_names, rotation=45, ha="right")
    heatmap_axis.set_yticks(range(len(attack_names)), attack_names)
    heatmap_axis.set_xlabel("Predicted")
    heatmap_axis.set_ylabel("Actual")

    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            heatmap_axis.text(column_index, row_index, str(value), ha="center", va="center", color="#111111")

    figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{summary.policy_name}: accuracy={summary.accuracy:.3f}, avg_reward={summary.average_reward:.3f}, avg_steps={summary.average_steps:.2f}"
    )
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination