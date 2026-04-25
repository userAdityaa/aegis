from __future__ import annotations

from environment.models import AttackClass


def final_verdict(decision: str, reasoning: str) -> dict[str, object]:
    normalized = decision.strip().lower()
    allowed = {attack.value for attack in AttackClass}
    if normalized not in allowed:
        raise ValueError(f"Unsupported verdict '{decision}'")

    return {
        "decision_received": normalized,
        "reasoning": reasoning.strip(),
        "is_terminal": True,
    }