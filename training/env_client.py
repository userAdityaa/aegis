from __future__ import annotations

from pathlib import Path
from typing import Callable

from environment.runtime import AegisRuntime
from environment.models import AttackClass
from rewards import score_episode
from training.types import EpisodeTrace, ToolCall, ToolObservation


class AegisEnvClient:
    def __init__(
        self,
        seed: int | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.runtime = AegisRuntime(seed=seed)
        self.manifest_path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parents[1] / "openenv.yaml"
        self._observations: list[ToolObservation] = []
        self._tool_dispatch: dict[str, Callable[..., dict[str, object]]] = {
            "check_maintainer_history": self.runtime.check_maintainer_history,
            "diff_versions": self.runtime.diff_versions,
            "trace_dependencies": self.runtime.trace_dependencies,
            "inspect_install_script": self.runtime.inspect_install_script,
            "get_reputation_score": self.runtime.get_reputation_score,
            "run_sandbox_test": self.runtime.run_sandbox_test,
            "append_case_note": self.runtime.append_case_note,
            "list_incident_inbox": self.runtime.list_incident_inbox,
            "read_incident_message": self.runtime.read_incident_message,
            "draft_incident_reply": self.runtime.draft_incident_reply,
            "send_incident_reply": self.runtime.send_incident_reply,
        }

    def reset(
        self,
        *,
        attack_class: str = "random",
        package_name: str | None = None,
        seed: int | None = None,
    ) -> dict[str, object]:
        self._observations = []
        return self.runtime.start_episode(
            attack_class=attack_class,
            package_name=package_name,
            seed=seed,
        )

    def call_tool(self, tool_name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        dispatch = self._tool_dispatch.get(tool_name)
        if dispatch is None:
            choices = sorted(self._tool_dispatch)
            raise ValueError(f"Unsupported tool '{tool_name}'. Expected one of: {choices}")

        call = ToolCall(name=tool_name, arguments=arguments or {})
        try:
            result = dispatch(**call.arguments)
        except Exception as exc:
            result = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "tool": tool_name,
            }
        self._observations.append(
            ToolObservation(
                step_index=len(self._observations) + 1,
                call=call,
                result=result,
            )
        )
        return result

    def submit_verdict(self, decision: str | AttackClass, reasoning: str) -> EpisodeTrace:
        normalized_decision = AttackClass(decision)
        verdict_response = self.runtime.final_verdict(normalized_decision, reasoning)
        current_episode = self.runtime.registry.current_episode
        if current_episode is None:
            raise RuntimeError("No active episode is available for scoring.")

        actual_attack = self.current_attack_class
        reward = score_episode(
            actual_attack=actual_attack,
            decision=normalized_decision,
            reasoning=reasoning,
            step_count=self.runtime.audit.step_count,
            tools_used=self.runtime.audit.tools_used,
        )
        return EpisodeTrace(
            episode_id=current_episode.episode_id,
            target_pkg=current_episode.target_pkg,
            actual_attack=actual_attack,
            decision=normalized_decision,
            reasoning=reasoning,
            observations=list(self._observations),
            reward=reward,
            verdict_response=verdict_response,
        )

    @property
    def current_attack_class(self) -> AttackClass:
        attack_label = self.runtime.registry.attack_label
        if attack_label is None:
            return AttackClass.SAFE
        return attack_label.attack_class

    @property
    def current_target(self) -> str | None:
        if self.runtime.registry.current_episode is None:
            return None
        return self.runtime.registry.current_episode.target_pkg

    def available_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._tool_dispatch))

    def manifest_text(self) -> str:
        return self.manifest_path.read_text(encoding="utf-8")

    @property
    def observations(self) -> list[ToolObservation]:
        return list(self._observations)