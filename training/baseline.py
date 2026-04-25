from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass

from environment.models import AttackClass
from training.env_client import AegisEnvClient
from training.parsing import render_tool_call, render_verdict
from training.types import ToolObservation


@dataclass(slots=True)
class BaselineSummary:
    episodes: int
    accuracy: float
    average_reward: float
    decision_counts: dict[str, int]
    actual_attack_counts: dict[str, int]


class HeuristicBaselinePolicy:
    _ORDER = (
        "check_maintainer_history",
        "diff_versions",
        "inspect_install_script",
        "get_reputation_score",
        "trace_dependencies",
        "run_sandbox_test",
    )

    def __call__(self, state: dict[str, object], observations: list[ToolObservation]) -> str:
        del state

        seen_tools = {observation.call.name for observation in observations}
        for tool_name in self._ORDER:
            if tool_name not in seen_tools:
                return render_tool_call(tool_name)

        decision, reasoning = _infer_verdict(observations)
        return render_verdict(decision, reasoning)


class RandomBaselinePolicy:
    def __init__(self, seed: int = 0, *, max_tool_steps: int = 3) -> None:
        self._random = random.Random(seed)
        self.max_tool_steps = max_tool_steps

    def __call__(self, state: dict[str, object], observations: list[ToolObservation]) -> str:
        del state

        seen_tools = {observation.call.name for observation in observations}
        remaining_tools = [tool_name for tool_name in HeuristicBaselinePolicy._ORDER if tool_name not in seen_tools]

        should_investigate = not observations or (
            remaining_tools
            and len(observations) < self.max_tool_steps
            and self._random.random() < 0.6
        )
        if should_investigate and remaining_tools:
            return render_tool_call(self._random.choice(remaining_tools))

        decision = self._random.choice(tuple(AttackClass))
        return render_verdict(decision, "random baseline guess")


def run_baseline(episodes: int = 10, seed: int = 0) -> BaselineSummary:
    # Imported lazily to avoid a circular import between baseline and rollout.
    from training.rollout import rollout_episode

    reward_total = 0.0
    correct = 0
    decision_counts: Counter[str] = Counter()
    actual_attack_counts: Counter[str] = Counter()

    for offset in range(episodes):
        client = AegisEnvClient(seed=seed + offset)
        trace = rollout_episode(client, HeuristicBaselinePolicy(), seed=seed + offset)
        reward_total += trace.reward.total
        correct += int(trace.actual_attack is trace.decision)
        decision_counts.update([trace.decision.value])
        actual_attack_counts.update([trace.actual_attack.value])

    return BaselineSummary(
        episodes=episodes,
        accuracy=correct / max(1, episodes),
        average_reward=reward_total / max(1, episodes),
        decision_counts=dict(sorted(decision_counts.items())),
        actual_attack_counts=dict(sorted(actual_attack_counts.items())),
    )


def infer_verdict_from_observations(observations: list[ToolObservation]) -> tuple[AttackClass, str]:
    """Infer a reasonable verdict from tool observations.

    This is used as a safe fallback during evaluation when a policy fails to submit a verdict
    within the step limit.
    """

    return _infer_verdict(observations)


def _infer_verdict(observations: list[ToolObservation]) -> tuple[AttackClass, str]:
    by_tool = {observation.call.name: observation.result for observation in observations}

    maintainer = by_tool.get("check_maintainer_history", {})
    version_diff = by_tool.get("diff_versions", {})
    install_script = by_tool.get("inspect_install_script", {})
    reputation = by_tool.get("get_reputation_score", {})
    dependencies = by_tool.get("trace_dependencies", {})
    sandbox = by_tool.get("run_sandbox_test", {})
    diff_text = version_diff.get("diff", "").lower()
    package_name = str(maintainer.get("package", ""))

    if "exec(" in install_script.get("suspicious_patterns", []) or "base64" in install_script.get("suspicious_patterns", []):
        return AttackClass.METADATA_INJECTION, "setup metadata contains exec and base64 patterns"
    if install_script.get("has_network_call") or install_script.get("has_exec"):
        return AttackClass.CICD_POISONING, "install script shows runtime execution or network behavior"
    if reputation.get("suspicious_popularity"):
        return AttackClass.STAR_JACKING, "package popularity is inconsistent with engagement signals"
    if any(dep.get("source") == "public" and dep.get("name", "").startswith(("corp", "internal", "private")) for dep in dependencies.get("flat_dependencies", [])):
        return AttackClass.DEPENDENCY_CONFUSION, "dependency tree includes a public package that looks internal"
    if sandbox.get("network_calls"):
        return AttackClass.DEAD_DROP_HIJACK, "sandbox execution shows outbound network behavior"
    if maintainer.get("suspicious_ip_shift") and "base64" in diff_text:
        return AttackClass.LONG_CON, "commit history shifted and the latest diff contains an encoded payload"
    if maintainer.get("suspicious_ip_shift"):
        return AttackClass.ACCOUNT_TAKEOVER, "maintainer IP history changed without stronger evidence elsewhere"
    if any(token in package_name for token in ("0", "1", "-", "_")) and reputation.get("stars", 0) < 100:
        return AttackClass.TYPOSQUATTING, "package name and low reputation resemble a typosquatting pattern"
    return AttackClass.SAFE, "no strong malicious signal was found across the forensic tools"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a heuristic baseline against the Aegis environment.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_baseline(episodes=args.episodes, seed=args.seed)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()