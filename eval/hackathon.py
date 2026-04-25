from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

from training.classifier_policy import NearestNeighborForensicPolicy, train_classifier_artifact
from training.baseline import HeuristicBaselinePolicy, RandomBaselinePolicy
from training.model_policy import TransformerTranscriptPolicy

from .compare import build_comparison_payload
from .plot import save_comparison_figure
from .plotting import save_evaluation_figure
from .runner import evaluate_policy, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hackathon-ready evaluation evidence for Aegis-Env.")
    parser.add_argument("--episodes-per-attack", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hackathon"))
    parser.add_argument("--classifier-artifact", type=Path, default=Path("artifacts/classifier-smoke/policy.json"))
    parser.add_argument("--classifier-train-episodes-per-attack", type=int, default=12)
    parser.add_argument("--classifier-seed", type=int, default=100)
    parser.add_argument("--trained-model", type=Path)
    parser.add_argument("--trained-label")
    parser.add_argument("--trained-report", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random_summary, random_episodes = evaluate_policy(
        RandomBaselinePolicy(seed=args.seed),
        episodes_per_attack=args.episodes_per_attack,
        seed=args.seed,
    )
    heuristic_summary, heuristic_episodes = evaluate_policy(
        HeuristicBaselinePolicy(),
        episodes_per_attack=args.episodes_per_attack,
        seed=args.seed,
    )

    random_report_path = write_evaluation_report(
        random_summary,
        random_episodes,
        output_dir / "random_report.json",
        label="random-baseline",
    )
    heuristic_report_path = write_evaluation_report(
        heuristic_summary,
        heuristic_episodes,
        output_dir / "heuristic_report.json",
        label="heuristic-baseline",
    )

    random_plot_path = save_evaluation_figure(random_summary, output_dir / "random_summary.png")
    heuristic_plot_path = save_evaluation_figure(heuristic_summary, output_dir / "heuristic_summary.png")
    random_summary_path = _write_summary(random_summary.as_dict(), output_dir / "random_summary.json")
    heuristic_summary_path = _write_summary(heuristic_summary.as_dict(), output_dir / "heuristic_summary.json")

    random_report = json.loads(random_report_path.read_text(encoding="utf-8"))
    heuristic_report = json.loads(heuristic_report_path.read_text(encoding="utf-8"))
    random_vs_heuristic = _write_comparison(
        baseline_report=random_report,
        trained_report=heuristic_report,
        output_dir=output_dir,
        stem="random_vs_heuristic",
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "episodes_per_attack": args.episodes_per_attack,
        "artifacts": {
            "random_report": random_report_path.as_posix(),
            "random_plot": random_plot_path.as_posix(),
            "random_summary": random_summary_path.as_posix(),
            "heuristic_report": heuristic_report_path.as_posix(),
            "heuristic_plot": heuristic_plot_path.as_posix(),
            "heuristic_summary": heuristic_summary_path.as_posix(),
            "random_vs_heuristic": random_vs_heuristic,
        },
        "metrics": {
            "random": random_summary.as_dict(),
            "heuristic": heuristic_summary.as_dict(),
            "delta": random_vs_heuristic["delta"],
        },
    }

    trained_payload: dict[str, object] | None = None
    if args.trained_report is not None:
        trained_payload = _normalize_report(json.loads(args.trained_report.read_text(encoding="utf-8")))
        payload["metrics"]["trained"] = dict(trained_payload.get("summary", {}))
        trained_summary_path = _write_summary(dict(trained_payload.get("summary", {})), output_dir / "trained_summary.json")
        payload["artifacts"]["trained_summary"] = trained_summary_path.as_posix()
    elif args.trained_model is not None:
        trained_policy = TransformerTranscriptPolicy(args.trained_model)
        trained_summary, trained_episodes = evaluate_policy(
            trained_policy,
            episodes_per_attack=args.episodes_per_attack,
            seed=args.seed,
        )
        trained_label = args.trained_label or f"TransformerTranscriptPolicy[{args.trained_model.as_posix()}]"
        trained_report_path = write_evaluation_report(
            trained_summary,
            trained_episodes,
            output_dir / "trained_report.json",
            label=trained_label,
        )
        trained_plot_path = save_evaluation_figure(trained_summary, output_dir / "trained_summary.png")
        trained_summary_path = _write_summary(trained_summary.as_dict(), output_dir / "trained_summary.json")
        trained_payload = json.loads(trained_report_path.read_text(encoding="utf-8"))
        payload["artifacts"]["trained_report"] = trained_report_path.as_posix()
        payload["artifacts"]["trained_plot"] = trained_plot_path.as_posix()
        payload["artifacts"]["trained_summary"] = trained_summary_path.as_posix()
        payload["metrics"]["trained"] = trained_summary.as_dict()
    else:
        if not args.classifier_artifact.exists():
            train_classifier_artifact(
                args.classifier_artifact,
                episodes_per_attack=args.classifier_train_episodes_per_attack,
                seed=args.classifier_seed,
            )
        trained_summary, trained_episodes = evaluate_policy(
            NearestNeighborForensicPolicy(args.classifier_artifact),
            episodes_per_attack=args.episodes_per_attack,
            seed=args.seed,
        )
        trained_report_path = write_evaluation_report(
            trained_summary,
            trained_episodes,
            output_dir / "trained_report.json",
            label=f"NearestNeighborForensicPolicy[{args.classifier_artifact.as_posix()}]",
        )
        trained_plot_path = save_evaluation_figure(trained_summary, output_dir / "trained_summary.png")
        trained_summary_path = _write_summary(trained_summary.as_dict(), output_dir / "trained_summary.json")
        trained_payload = json.loads(trained_report_path.read_text(encoding="utf-8"))
        payload["artifacts"]["trained_report"] = trained_report_path.as_posix()
        payload["artifacts"]["trained_plot"] = trained_plot_path.as_posix()
        payload["artifacts"]["trained_summary"] = trained_summary_path.as_posix()
        payload["metrics"]["trained"] = trained_summary.as_dict()

    if trained_payload is not None:
        heuristic_vs_trained = _write_comparison(
            baseline_report=heuristic_report,
            trained_report=trained_payload,
            output_dir=output_dir,
            stem="heuristic_vs_trained",
        )
        random_vs_trained = _write_comparison(
            baseline_report=random_report,
            trained_report=trained_payload,
            output_dir=output_dir,
            stem="random_vs_trained",
        )
        payload["artifacts"]["heuristic_vs_trained"] = heuristic_vs_trained
        payload["artifacts"]["random_vs_trained"] = random_vs_trained
        payload["metrics"]["trained_delta"] = heuristic_vs_trained["delta"]
        payload["metrics"]["random_to_trained_delta"] = random_vs_trained["delta"]

    submission_checks = _submission_checks(output_dir)
    payload["submission_checks"] = submission_checks
    payload["submission_blockers"] = _submission_blockers(submission_checks)

    summary_path = output_dir / "submission_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_comparison(
    *,
    baseline_report: Mapping[str, object],
    trained_report: Mapping[str, object],
    output_dir: Path,
    stem: str,
) -> dict[str, object]:
    comparison_plot = save_comparison_figure(
        baseline_report,
        trained_report,
        output_dir / f"{stem}.png",
    )
    comparison_payload = build_comparison_payload(baseline_report, trained_report)
    comparison_path = output_dir / f"{stem}.json"
    comparison_payload["artifacts"] = {
        "plot_path": comparison_plot.as_posix(),
        "comparison_path": comparison_path.as_posix(),
    }
    comparison_path.write_text(json.dumps(comparison_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "comparison_path": comparison_path.as_posix(),
        "plot_path": comparison_plot.as_posix(),
        "delta": comparison_payload["delta"],
    }


def _normalize_report(payload: Mapping[str, object]) -> dict[str, object]:
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

    label = payload.get("label") or summary.get("policy_name") or "run"
    return {
        "schema_version": payload.get("schema_version", 1),
        "label": str(label),
        "summary": summary,
        "episodes": episodes,
        "reward_curve": reward_curve,
    }


def _coerce_episode_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _coerce_float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, (int, float))]


def _write_summary(summary: Mapping[str, object], output_path: Path) -> Path:
    output_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _submission_checks(output_dir: Path) -> dict[str, bool]:
    repo_root = Path(__file__).resolve().parents[1]
    readme_path = repo_root / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    checks = {
        "openenv_manifest_exists": (repo_root / "openenv.yaml").exists(),
        "training_notebook_exists": (repo_root / "notebooks" / "aegis_grpo_colab.ipynb").exists(),
        "dockerfile_exists": (repo_root / "docker" / "Dockerfile").exists(),
        "training_evidence_summary_exists": (repo_root / "reports" / "training_evidence" / "training_summary.json").exists(),
        "training_evidence_history_exists": (repo_root / "reports" / "training_evidence" / "training_log_history.json").exists(),
        "training_evidence_plot_exists": (repo_root / "reports" / "training_evidence" / "training_curves.png").exists(),
        "slide_deck_exists": (repo_root / "docs" / "hackathon_slide_deck.md").exists(),
        "random_report_exists": (output_dir / "random_report.json").exists(),
        "heuristic_report_exists": (output_dir / "heuristic_report.json").exists(),
        "trained_report_exists": (output_dir / "trained_report.json").exists(),
        "comparison_exists": (output_dir / "random_vs_heuristic.json").exists(),
        "trained_comparison_exists": (output_dir / "heuristic_vs_trained.json").exists(),
        "readme_has_theme_statement": "Theme 3.1 (World Modeling - Professional Tasks)" in readme_text,
        "readme_has_minimum_requirements_section": "## Minimum Requirement Status" in readme_text,
        "readme_has_results_section": "## Results Snapshot" in readme_text,
        "readme_has_submission_assets_section": "## Submission Assets" in readme_text,
        "readme_links_notebook": "notebooks/aegis_grpo_colab.ipynb" in readme_text,
        "readme_links_training_evidence": (
            "reports/training_evidence/training_summary.json" in readme_text
            and "reports/training_evidence/training_log_history.json" in readme_text
            and "reports/training_evidence/training_curves.png" in readme_text
        ),
        "readme_links_hackathon_summary": "reports/hackathon/submission_summary.json" in readme_text,
        "readme_has_presentation_asset": _readme_has_presentation_asset(readme_text),
        "readme_has_live_hf_space_url": _has_live_hf_space_url(readme_text),
        "readme_has_no_todo_placeholders": "TODO" not in readme_text,
    }
    checks["submission_ready"] = not _submission_blockers(checks)
    return checks


def _submission_blockers(checks: Mapping[str, bool]) -> list[str]:
    blockers: list[str] = []
    if not checks.get("openenv_manifest_exists", False):
        blockers.append("Missing openenv.yaml.")
    if not checks.get("training_notebook_exists", False):
        blockers.append("Missing the Colab notebook at notebooks/aegis_grpo_colab.ipynb.")
    if not checks.get("dockerfile_exists", False):
        blockers.append("Missing the Docker Space entrypoint at docker/Dockerfile.")
    if not checks.get("training_evidence_summary_exists", False):
        blockers.append("Missing reports/training_evidence/training_summary.json.")
    if not checks.get("training_evidence_history_exists", False):
        blockers.append("Missing reports/training_evidence/training_log_history.json.")
    if not checks.get("training_evidence_plot_exists", False):
        blockers.append("Missing reports/training_evidence/training_curves.png.")
    if not checks.get("trained_report_exists", False):
        blockers.append("Missing reports/hackathon/trained_report.json.")
    if not checks.get("readme_has_results_section", False):
        blockers.append("README.md is missing a judge-facing results section.")
    if not checks.get("readme_has_presentation_asset", False):
        blockers.append("README.md must link to a slide deck, mini blog, or short video.")
    if not checks.get("readme_has_live_hf_space_url", False):
        blockers.append("Publish the Docker app to a Hugging Face Space and add the live Space URL to README.md.")
    return blockers


def _readme_has_presentation_asset(readme_text: str) -> bool:
    if "docs/hackathon_slide_deck.md" in readme_text:
        return True
    return re.search(r"https://(www\.)?(youtube\.com|youtu\.be|huggingface\.co/(blog|posts?))/", readme_text) is not None


def _has_live_hf_space_url(readme_text: str) -> bool:
    return re.search(r"https://huggingface\.co/spaces/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", readme_text) is not None


if __name__ == "__main__":
    main()