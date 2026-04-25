from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.env_client import AegisEnvClient
from training.types import EpisodeTrace


class GRPOAegisEnvironment:
    def __init__(self, manifest_path: str | Path | None = None) -> None:
        self._manifest_path = manifest_path
        self.client = AegisEnvClient(manifest_path=manifest_path)
        self.last_trace: EpisodeTrace | None = None
        self.last_reward: float = 0.0

    def reset(
        self,
        attack_class: str = "random",
        seed: int | None = None,
        package_name: str | None = None,
        prompt: object | None = None,
        **_: object,
    ) -> str:
        del prompt
        state = self.client.reset(
            attack_class=attack_class,
            package_name=package_name,
            seed=seed,
        )
        self.last_trace = None
        self.last_reward = 0.0
        return (
            f"Target package: {state['target_pkg']}\n"
            "Use the available forensic tools to investigate the package. "
            "When you are confident, call final_verdict exactly once."
        )

    def check_maintainer_history(self, pkg_name: str | None = None) -> str:
        """Inspect maintainer identity, IP history, and commit timing.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.

        Returns:
            A JSON string containing maintainer investigation results.
        """

        return self._call_tool("check_maintainer_history", pkg_name=pkg_name)

    def diff_versions(
        self,
        pkg_name: str | None = None,
        v1: str | None = None,
        v2: str | None = None,
    ) -> str:
        """Compare two versions of a package.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.
            v1: Optional earlier version. Omit to compare the latest two versions.
            v2: Optional later version. Omit to compare the latest two versions.

        Returns:
            A JSON string containing the version diff and metadata.
        """

        return self._call_tool("diff_versions", pkg_name=pkg_name, v1=v1, v2=v2)

    def trace_dependencies(self, pkg_name: str | None = None) -> str:
        """Trace dependency relationships for a package.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.

        Returns:
            A JSON string containing the dependency tree and flattened summary.
        """

        return self._call_tool("trace_dependencies", pkg_name=pkg_name)

    def inspect_install_script(self, pkg_name: str | None = None) -> str:
        """Inspect installation metadata and setup logic for suspicious behavior.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.

        Returns:
            A JSON string describing suspicious install-script patterns.
        """

        return self._call_tool("inspect_install_script", pkg_name=pkg_name)

    def get_reputation_score(self, pkg_name: str | None = None) -> str:
        """Inspect popularity and maintainer reputation signals.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.

        Returns:
            A JSON string containing reputation metrics and anomaly flags.
        """

        return self._call_tool("get_reputation_score", pkg_name=pkg_name)

    def run_sandbox_test(self, pkg_name: str | None = None) -> str:
        """Return simulated runtime activity observed during installation.

        Args:
            pkg_name: Optional package name. Defaults to the active episode target.

        Returns:
            A JSON string containing sandbox observations.
        """

        return self._call_tool("run_sandbox_test", pkg_name=pkg_name)

    def final_verdict(self, decision: str, reasoning: str) -> str:
        """Submit the final attack classification for the current episode.

        Args:
            decision: The final predicted attack class or safe.
            reasoning: Concise evidence-based rationale for the decision.

        Returns:
            A JSON string containing the terminal verdict response.
        """

        trace = self.client.submit_verdict(decision, reasoning)
        self.last_trace = trace
        self.last_reward = trace.reward.total
        return json.dumps(trace.verdict_response, sort_keys=True)

    def _call_tool(self, tool_name: str, **arguments: Any) -> str:
        normalized = {key: value for key, value in arguments.items() if value is not None}
        result = self.client.call_tool(tool_name, normalized)
        return json.dumps(result, sort_keys=True)


def aegis_reward_func(
    environments: list[GRPOAegisEnvironment],
    log_metric=None,
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    verdict_completion_count = 0

    for environment in environments:
        if environment.last_trace is None:
            rewards.append(-1.0)
            continue
        verdict_completion_count += 1
        rewards.append(environment.last_reward)

    if log_metric and rewards:
        log_metric("aegis/reward_mean", sum(rewards) / len(rewards))
        log_metric("aegis/verdict_completion_rate", verdict_completion_count / len(rewards))

    return rewards