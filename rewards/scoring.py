from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from environment.attacks import ALL_ATTACKS
from environment.models import AttackClass

_REQUIRED_TOOLS = {
    attack_cls.ATTACK_CLASS: attack_cls.REQUIRED_TOOLS
    for attack_cls in ALL_ATTACKS
}

_REASONING_HINTS = {
    AttackClass.TYPOSQUATTING: ("name", "similar", "maintainer", "reputation"),
    AttackClass.LONG_CON: ("commit", "ip", "base64", "diff"),
    AttackClass.ACCOUNT_TAKEOVER: ("maintainer", "ip", "timezone", "off-hours"),
    AttackClass.CICD_POISONING: ("install", "script", "curl", "subprocess"),
    AttackClass.DEPENDENCY_CONFUSION: ("dependency", "public", "private", "version"),
    AttackClass.METADATA_INJECTION: ("setup", "exec", "base64", "metadata"),
    AttackClass.STAR_JACKING: ("stars", "issues", "downloads", "popularity"),
    AttackClass.DEAD_DROP_HIJACK: ("maintainer", "socket", "abandoned", "gap"),
    AttackClass.SAFE: ("safe", "clean", "benign", "no evidence"),
}


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    accuracy: float
    parsimony: float
    reasoning: float
    false_alarm: float

    @property
    def total(self) -> float:
        return self.accuracy + self.parsimony + self.reasoning + self.false_alarm

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "parsimony": self.parsimony,
            "reasoning": self.reasoning,
            "false_alarm": self.false_alarm,
            "total": self.total,
        }


def score_episode(
    *,
    actual_attack: AttackClass,
    decision: AttackClass,
    reasoning: str,
    step_count: int,
    tools_used: Sequence[str],
) -> RewardBreakdown:
    del tools_used

    accuracy = _accuracy_reward(actual_attack=actual_attack, decision=decision)
    false_alarm = _false_alarm_penalty(actual_attack=actual_attack, decision=decision)
    parsimony = _parsimony_penalty(actual_attack=actual_attack, step_count=step_count)
    reasoning_reward = _reasoning_reward(actual_attack=actual_attack, reasoning=reasoning)
    return RewardBreakdown(
        accuracy=accuracy,
        parsimony=parsimony,
        reasoning=reasoning_reward,
        false_alarm=false_alarm,
    )


def _accuracy_reward(*, actual_attack: AttackClass, decision: AttackClass) -> float:
    if actual_attack is AttackClass.SAFE:
        return 0.5 if decision is AttackClass.SAFE else 0.0
    return 1.0 if decision is actual_attack else -1.0


def _false_alarm_penalty(*, actual_attack: AttackClass, decision: AttackClass) -> float:
    if actual_attack is AttackClass.SAFE and decision is not AttackClass.SAFE:
        return -0.6
    return 0.0


def _parsimony_penalty(*, actual_attack: AttackClass, step_count: int) -> float:
    required_tools = _REQUIRED_TOOLS.get(actual_attack, ())
    unnecessary_steps = max(0, step_count - len(required_tools))
    return -0.05 * unnecessary_steps


def _reasoning_reward(*, actual_attack: AttackClass, reasoning: str) -> float:
    normalized = reasoning.strip().lower()
    if not normalized:
        return 0.0
    matched_hints = sum(1 for hint in _REASONING_HINTS[actual_attack] if hint in normalized)
    return 0.2 if matched_hints >= 2 else 0.0