from __future__ import annotations

from typing import TYPE_CHECKING

from environment.models import AttackClass

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


def final_verdict(
    decision: str,
    reasoning: str,
    registry: "ShadowRegistry" | None = None,
) -> dict[str, object]:
    normalized = decision.strip().lower()
    allowed = {attack.value for attack in AttackClass}
    if normalized not in allowed:
        raise ValueError(f"Unsupported verdict '{decision}'")

    return {
        "episode_id": registry.current_episode.episode_id if registry and registry.current_episode else None,
        "decision_received": normalized,
        "reasoning": reasoning.strip(),
        "is_terminal": True,
    }