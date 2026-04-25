from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from environment.models import AttackClass
from training.env_client import AegisEnvClient
from training.parsing import render_tool_call, render_verdict
from training.types import ToolObservation

TOOL_ORDER: tuple[str, ...] = (
    "check_maintainer_history",
    "diff_versions",
    "inspect_install_script",
    "get_reputation_score",
    "trace_dependencies",
    "run_sandbox_test",
)

FEATURE_NAMES: tuple[str, ...] = (
    "ip_shift",
    "off_hours",
    "diff_base64",
    "diff_alias",
    "diff_socket",
    "install_exec",
    "install_net",
    "pattern_exec",
    "pattern_base64",
    "pattern_fetch",
    "rep_suspicious",
    "low_downloads",
    "public_internal_dep",
    "sandbox_net",
    "maintainer_count",
    "dep_count",
)

_NUMERIC_FEATURES = {"maintainer_count", "dep_count"}
_FEATURE_REASONING = {
    "ip_shift": "maintainer IP history changed",
    "off_hours": "recent off-hours activity appeared",
    "diff_alias": "the latest release looks like a compatibility alias",
    "diff_socket": "release changes contain network-oriented behavior",
    "install_exec": "install metadata executes code",
    "install_net": "install metadata makes network calls",
    "pattern_exec": "setup metadata includes exec",
    "pattern_base64": "setup metadata includes encoded payloads",
    "pattern_fetch": "build metadata references external fetches",
    "rep_suspicious": "package popularity does not match engagement signals",
    "low_downloads": "package reputation is still low",
    "public_internal_dep": "public dependencies look like private internal names",
    "sandbox_net": "sandbox execution produced outbound network traffic",
}


@dataclass(slots=True)
class FeatureExample:
    attack_class: str
    seed: int
    target_package: str
    features: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class NearestNeighborForensicPolicy:
    def __init__(self, artifact_path: str | Path, *, top_k: int | None = None) -> None:
        self.artifact_path = Path(artifact_path)
        payload = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.top_k = int(top_k or payload.get("top_k", 5))
        self.examples = [
            FeatureExample(
                attack_class=str(example["attack_class"]),
                seed=int(example["seed"]),
                target_package=str(example["target_package"]),
                features={name: int(value) for name, value in dict(example["features"]).items()},
            )
            for example in payload.get("examples", [])
        ]
        if not self.examples:
            raise ValueError(f"No training examples were found in {self.artifact_path}.")

    def __call__(self, state: dict[str, object], observations: list[ToolObservation]) -> str:
        del state

        seen_tools = {observation.call.name for observation in observations}
        for tool_name in TOOL_ORDER:
            if tool_name not in seen_tools:
                return render_tool_call(tool_name)

        features = extract_features(observations)
        predicted, neighbors = self._predict(features)
        reasoning = build_reasoning(predicted, features, neighbors)
        return render_verdict(predicted, reasoning)

    def _predict(self, features: dict[str, int]) -> tuple[AttackClass, list[tuple[float, FeatureExample]]]:
        ranked = sorted(
            ((feature_distance(features, example.features), example) for example in self.examples),
            key=lambda item: (item[0], item[1].attack_class, item[1].seed),
        )
        neighbors = ranked[: self.top_k]
        counts = Counter(example.attack_class for _, example in neighbors)
        predicted = counts.most_common(1)[0][0]
        return AttackClass(predicted), neighbors


def collect_feature_example(
    *,
    attack_class: AttackClass,
    seed: int,
    manifest_path: str | Path | None = None,
) -> FeatureExample:
    client = AegisEnvClient(seed=seed, manifest_path=manifest_path)
    state = client.reset(attack_class=attack_class.value, seed=seed)
    for tool_name in TOOL_ORDER:
        client.call_tool(tool_name)
    return FeatureExample(
        attack_class=attack_class.value,
        seed=seed,
        target_package=str(state["target_pkg"]),
        features=extract_features(client.observations),
    )


def train_classifier_artifact(
    output_path: str | Path,
    *,
    episodes_per_attack: int = 12,
    seed: int = 100,
    manifest_path: str | Path | None = None,
    top_k: int = 5,
    attack_schedule: Sequence[AttackClass | str] | None = None,
) -> dict[str, object]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    schedule = _normalize_schedule(attack_schedule, episodes_per_attack=episodes_per_attack)
    examples: list[FeatureExample] = []
    for index, attack_class in enumerate(schedule):
        examples.append(
            collect_feature_example(
                attack_class=attack_class,
                seed=seed + index,
                manifest_path=manifest_path,
            )
        )

    payload = {
        "schema_version": 1,
        "policy_name": "NearestNeighborForensicPolicy",
        "tool_order": list(TOOL_ORDER),
        "feature_names": list(FEATURE_NAMES),
        "top_k": top_k,
        "training_summary": {
            "episodes": len(examples),
            "episodes_per_attack": episodes_per_attack,
            "seed": seed,
            "attack_counts": dict(sorted(Counter(example.attack_class for example in examples).items())),
        },
        "examples": [example.as_dict() for example in examples],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact_path": destination.as_posix(),
        "examples": len(examples),
        "top_k": top_k,
        "attack_counts": payload["training_summary"]["attack_counts"],
    }


def extract_features(observations: Sequence[ToolObservation]) -> dict[str, int]:
    by_tool = {observation.call.name: observation.result for observation in observations}
    missing = [tool_name for tool_name in TOOL_ORDER if tool_name not in by_tool]
    if missing:
        raise ValueError(f"Cannot extract features before all forensic tools have been called: {missing}")

    maintainer = by_tool["check_maintainer_history"]
    version_diff = by_tool["diff_versions"]
    install_script = by_tool["inspect_install_script"]
    reputation = by_tool["get_reputation_score"]
    dependencies = by_tool["trace_dependencies"]
    sandbox = by_tool["run_sandbox_test"]
    diff_text = str(version_diff.get("diff", "")).lower()
    suspicious_patterns = set(install_script.get("suspicious_patterns", []))

    return {
        "ip_shift": int(bool(maintainer.get("suspicious_ip_shift"))),
        "off_hours": int(bool(maintainer.get("off_hours_recent_commits"))),
        "diff_base64": int("base64" in diff_text),
        "diff_alias": int("alias" in diff_text),
        "diff_socket": int(any(token in diff_text for token in ("socket", "urllib", "fetch("))),
        "install_exec": int(bool(install_script.get("has_exec"))),
        "install_net": int(bool(install_script.get("has_network_call"))),
        "pattern_exec": int("exec(" in suspicious_patterns),
        "pattern_base64": int("base64" in suspicious_patterns),
        "pattern_fetch": int(any(token in suspicious_patterns for token in ("fetch(", "urllib.request"))),
        "rep_suspicious": int(bool(reputation.get("suspicious_popularity"))),
        "low_downloads": int(float(reputation.get("average_weekly_downloads", 0.0)) < 500.0),
        "public_internal_dep": int(
            any(
                dependency.get("source") == "public"
                and str(dependency.get("name", "")).startswith(("corp", "internal", "private"))
                for dependency in dependencies.get("flat_dependencies", [])
            )
        ),
        "sandbox_net": int(bool(sandbox.get("network_calls"))),
        "maintainer_count": int(reputation.get("maintainer_count", 0)),
        "dep_count": int(dependencies.get("dependency_count", 0)),
    }


def feature_distance(left: dict[str, int], right: dict[str, int]) -> float:
    total = 0.0
    for name in FEATURE_NAMES:
        left_value = int(left[name])
        right_value = int(right[name])
        if name in _NUMERIC_FEATURES:
            total += abs(left_value - right_value) / 25.0
        else:
            total += 0.0 if left_value == right_value else 1.0
    return total


def build_reasoning(
    decision: AttackClass,
    features: dict[str, int],
    neighbors: Sequence[tuple[float, FeatureExample]],
) -> str:
    active_signals = [message for feature_name, message in _FEATURE_REASONING.items() if features.get(feature_name)]
    if decision is AttackClass.SAFE:
        return "nearest learned forensic traces did not surface a strong malicious signal"
    if active_signals:
        summary = "; ".join(active_signals[:3])
        return f"nearest learned forensic traces matched: {summary}"

    nearest_labels = ", ".join(example.attack_class for _, example in neighbors[:3])
    return f"nearest learned forensic traces most often matched {nearest_labels}"


def _normalize_schedule(
    attack_schedule: Sequence[AttackClass | str] | None,
    *,
    episodes_per_attack: int,
) -> list[AttackClass]:
    if attack_schedule is not None:
        return [attack if isinstance(attack, AttackClass) else AttackClass(attack) for attack in attack_schedule]

    schedule: list[AttackClass] = []
    for attack_class in AttackClass:
        schedule.extend([attack_class] * episodes_per_attack)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a nearest-neighbor forensic policy artifact.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/classifier-smoke/policy.json"))
    parser.add_argument("--episodes-per-attack", type=int, default=12)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest-path")
    args = parser.parse_args()

    payload = train_classifier_artifact(
        args.output,
        episodes_per_attack=args.episodes_per_attack,
        seed=args.seed,
        manifest_path=args.manifest_path,
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
