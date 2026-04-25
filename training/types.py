from __future__ import annotations

from dataclasses import dataclass, field

from environment.models import AttackClass
from rewards import RewardBreakdown


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ToolObservation:
    step_index: int
    call: ToolCall
    result: dict[str, object]


@dataclass(slots=True)
class EpisodeTrace:
    episode_id: str
    target_pkg: str
    actual_attack: AttackClass
    decision: AttackClass
    reasoning: str
    observations: list[ToolObservation]
    reward: RewardBreakdown
    verdict_response: dict[str, object]

    @property
    def tool_names(self) -> list[str]:
        return [observation.call.name for observation in self.observations]