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

from rewards.rubrics import (
    ComposedRubric,
    EvidenceRubric,
    RubricInput,
    SpeedRubric,
    SpecificityRubric,
    VerdictRubric,
)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    verdict: float
    speed: float
    specificity: float
    evidence: float

    @property
    def total(self) -> float:
        return self.verdict + self.speed + self.specificity + self.evidence

    def as_dict(self) -> dict[str, float]:
        return {
            "verdict": self.verdict,
            "speed": self.speed,
            "specificity": self.specificity,
            "evidence": self.evidence,
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
    composed = build_default_rubric(actual_attack=actual_attack)
    inputs = RubricInput(
        actual_attack=actual_attack,
        decision=decision,
        reasoning=reasoning,
        step_count=step_count,
        tools_used=tuple(tools_used),
    )
    parts = composed.score_components(inputs)
    return RewardBreakdown(
        verdict=float(parts["verdict"]),
        speed=float(parts["speed"]),
        specificity=float(parts["specificity"]),
        evidence=float(parts["evidence"]),
    )

def build_default_rubric(*, actual_attack: AttackClass) -> ComposedRubric:
    hints = _REASONING_HINTS.get(actual_attack, ())
    required_tools = tuple(_REQUIRED_TOOLS.get(actual_attack, ()))
    evidence_signals = tuple(dict.fromkeys((*hints, *required_tools, "customer", "stakeholder", "mitigation", "remediation", "impact")))
    return ComposedRubric(
        verdict=VerdictRubric(),
        speed=SpeedRubric(max_steps_for_bonus=max(4, len(required_tools) + 1), bonus=0.3),
        specificity=SpecificityRubric(false_quarantine_penalty=-0.6),
        evidence=EvidenceRubric(signals=evidence_signals, per_signal=0.1, max_total=0.8),
        weights={"verdict": 1.0, "speed": 1.0, "specificity": 1.0, "evidence": 1.0},
    )