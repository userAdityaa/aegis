from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.env_client import AegisEnvClient
from training.types import EpisodeTrace
from training.curriculum import CurriculumScheduler


class GRPOAegisEnvironment:
    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        evidence_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self._manifest_path = manifest_path
        self.client = AegisEnvClient(manifest_path=manifest_path)
        self.last_trace: EpisodeTrace | None = None
        self.last_reward: float = 0.0
        self._evidence_dir = Path(evidence_dir) if evidence_dir else None
        self._run_id = run_id or "run"
        self._last_episode_meta: dict[str, object] = {}
        self._use_curriculum = False
        self._curriculum: CurriculumScheduler | None = None

    def _enable_curriculum(self, *, seed: int = 0) -> None:
        """Enable curriculum sampling for subsequent resets.

        This is a training-time convenience hook (not a forensic tool). When enabled, calls
        to `reset()` will select the next attack class via the curriculum scheduler.

        Args:
            seed: RNG seed for the curriculum scheduler.
        """
        if self._evidence_dir is None:
            return
        self._use_curriculum = True
        self._curriculum = CurriculumScheduler(evidence_dir=self._evidence_dir, seed=seed)

    def reset(
        self,
        attack_class: str = "random",
        seed: int | None = None,
        package_name: str | None = None,
        prompt: object | None = None,
        **_: object,
    ) -> str:
        """Start a new episode and return the initial user prompt.

        Args:
            attack_class: Attack class label to inject, or "random".
            seed: Optional episode RNG seed.
            package_name: Optional package name to investigate.
            prompt: Reserved for compatibility with TRL environments (unused).
            **_: Ignored extra keyword arguments for compatibility.
        """
        del prompt
        if self._use_curriculum and self._curriculum is not None:
            attack_class = self._curriculum.select_attack().value
        state = self.client.reset(
            attack_class=attack_class,
            package_name=package_name,
            seed=seed,
        )
        self.last_trace = None
        self.last_reward = 0.0
        self._last_episode_meta = {
            "episode_id": str(state.get("episode_id") or ""),
            "target_pkg": str(state.get("target_pkg") or ""),
            "actual_attack": self.client.current_attack_class.value,
        }
        return (
            f"Target package: {state['target_pkg']}\n"
            "Use the available forensic tools to investigate the package. "
            "When you are confident, call final_verdict exactly once."
        )

    def check_maintainer_history(self, pkg_name: str | None = None) -> str:
        """Return maintainer metadata plus IP and commit-timing history.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
        """

        return self._call_tool("check_maintainer_history", pkg_name=pkg_name)

    def diff_versions(
        self,
        pkg_name: str | None = None,
        v1: str | None = None,
        v2: str | None = None,
    ) -> str:
        """Return a diff between two package versions.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
            v1: Optional earlier version identifier. If omitted, the environment selects one.
            v2: Optional later version identifier. If omitted, the environment selects one.
        """

        return self._call_tool("diff_versions", pkg_name=pkg_name, v1=v1, v2=v2)

    def trace_dependencies(self, pkg_name: str | None = None) -> str:
        """Return the dependency tree and flattened dependency summary.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
        """

        return self._call_tool("trace_dependencies", pkg_name=pkg_name)

    def inspect_install_script(self, pkg_name: str | None = None) -> str:
        """Inspect install/build scripts for suspicious patterns.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
        """

        return self._call_tool("inspect_install_script", pkg_name=pkg_name)

    def get_reputation_score(self, pkg_name: str | None = None) -> str:
        """Return reputation and popularity signals for a package.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
        """

        return self._call_tool("get_reputation_score", pkg_name=pkg_name)

    def run_sandbox_test(self, pkg_name: str | None = None) -> str:
        """Run a simulated sandbox test and return runtime observations.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
        """

        return self._call_tool("run_sandbox_test", pkg_name=pkg_name)

    def final_verdict(self, decision: str, reasoning: str) -> str:
        """Submit the final verdict and end the episode.

        Args:
            decision: Predicted attack class label (or "safe").
            reasoning: Concise evidence-based rationale for the decision.
        """

        trace = self.client.submit_verdict(decision, reasoning)
        self.last_trace = trace
        self.last_reward = trace.reward.total
        self._append_training_event(trace)
        return json.dumps(trace.verdict_response, sort_keys=True)

    def _call_tool(self, tool_name: str, **arguments: Any) -> str:
        normalized = {key: value for key, value in arguments.items() if value is not None}
        result = self.client.call_tool(tool_name, normalized)
        return json.dumps(result, sort_keys=True)

    def _append_training_event(self, trace: EpisodeTrace) -> None:
        if self._evidence_dir is None:
            return
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self._evidence_dir / "per_episode_events.jsonl"
        payload = {
            "run_id": self._run_id,
            "episode_id": trace.episode_id,
            "target_pkg": trace.target_pkg,
            "actual_attack": trace.actual_attack.value,
            "decision": trace.decision.value,
            "correct": bool(trace.actual_attack is trace.decision),
            "reward_total": trace.reward.total,
            "reward_breakdown": trace.reward.as_dict(),
            "step_count": len(trace.observations),
            "tool_names": trace.tool_names,
            "reasoning": trace.reasoning,
            "observations": [
                {
                    "step_index": obs.step_index,
                    "tool_name": obs.call.name,
                    "arguments": obs.call.arguments,
                    "result": obs.result,
                }
                for obs in trace.observations
            ],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _append_failed_event(self, *, reward: float = -1.0) -> None:
        if self._evidence_dir is None:
            return
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self._evidence_dir / "per_episode_events.jsonl"
        payload = {
            "run_id": self._run_id,
            "episode_id": self._last_episode_meta.get("episode_id", ""),
            "target_pkg": self._last_episode_meta.get("target_pkg", ""),
            "actual_attack": self._last_episode_meta.get("actual_attack", "safe"),
            "decision": "no_verdict",
            "correct": False,
            "reward_total": float(reward),
            "reward_breakdown": {"verdict": 0.0, "speed": 0.0, "specificity": 0.0, "evidence": 0.0, "total": float(reward)},
            "step_count": len(self.client.observations),
            "tool_names": [obs.call.name for obs in self.client.observations],
            "reasoning": "",
            "observations": [
                {
                    "step_index": obs.step_index,
                    "tool_name": obs.call.name,
                    "arguments": obs.call.arguments,
                    "result": obs.result,
                }
                for obs in self.client.observations
            ],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def aegis_reward_func(
    environments: list[GRPOAegisEnvironment],
    log_metric=None,
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    verdict_completion_count = 0
    per_class_rewards: dict[str, list[float]] = {}
    per_class_correct: dict[str, list[int]] = {}
    per_component: dict[str, list[float]] = {}

    for environment in environments:
        if environment.last_trace is None:
            environment._append_failed_event(reward=-1.0)
            rewards.append(-1.0)
            continue
        verdict_completion_count += 1
        rewards.append(environment.last_reward)
        trace = environment.last_trace
        attack = trace.actual_attack.value
        per_class_rewards.setdefault(attack, []).append(trace.reward.total)
        per_class_correct.setdefault(attack, []).append(int(trace.actual_attack is trace.decision))
        for key, value in trace.reward.as_dict().items():
            if key == "total":
                continue
            per_component.setdefault(key, []).append(float(value))

    if log_metric and rewards:
        log_metric("aegis/reward_mean", sum(rewards) / len(rewards))
        log_metric("aegis/verdict_completion_rate", verdict_completion_count / len(rewards))
        for attack, values in per_class_rewards.items():
            log_metric(f"aegis/per_class/{attack}/reward_mean", sum(values) / len(values))
        for attack, values in per_class_correct.items():
            log_metric(f"aegis/per_class/{attack}/accuracy", sum(values) / len(values))
        for component, values in per_component.items():
            log_metric(f"aegis/rubric/{component}_mean", sum(values) / len(values))

    return rewards


def _completion_to_text(completion: object) -> str:
    """Normalize TRL completion payloads to a single string for pattern matching."""

    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (bytes, bytearray)):
        return completion.decode("utf-8", errors="replace")
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        return str(completion)
    return str(completion)


def aegis_completion_reward_func(
    completions: list[object] | None = None,
    log_metric=None,
    **_: object,
) -> list[float]:
    """A lightweight shaping reward based on the raw completion text.

    This is a safety net for tiny smoke runs where the model may fail to execute tools
    (and thus produce a constant environment reward). It rewards producing the expected
    "verdict" / "final_verdict" structure so gradients are non-degenerate.
    """

    if not completions:
        return []

    rewards: list[float] = []
    verdict_like = 0
    tool_like = 0
    for raw in completions:
        normalized = _completion_to_text(raw).lower()
        has_verdict = ("<verdict>" in normalized and "</verdict>" in normalized) or ("final_verdict" in normalized)
        has_tool = "<tool>" in normalized or "check_maintainer_history" in normalized or "diff_versions" in normalized
        verdict_like += int(has_verdict)
        tool_like += int(has_tool)

        # Small, bounded reward to avoid dominating the true environment rubric.
        reward = 0.0
        if has_tool:
            reward += 0.05
        if has_verdict:
            reward += 0.15
        rewards.append(reward)

    if log_metric and rewards:
        log_metric("aegis/completion_shaping_mean", sum(rewards) / len(rewards))
        log_metric("aegis/completion_verdict_like_rate", verdict_like / len(rewards))
        log_metric("aegis/completion_tool_like_rate", tool_like / len(rewards))

    return rewards
