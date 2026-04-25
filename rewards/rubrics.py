from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from openenv.core.rubrics import Rubric

from environment.models import AttackClass


@dataclass(frozen=True, slots=True)
class RubricInput:
    actual_attack: AttackClass
    decision: AttackClass
    reasoning: str
    step_count: int
    tools_used: tuple[str, ...]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class VerdictRubric(Rubric):
    """+1.0 if correct, -1.0 if incorrect (safe included)."""

    def forward(self, x: RubricInput) -> float:
        return 1.0 if x.actual_attack is x.decision else -1.0


class SpeedRubric(Rubric):
    """Small bonus for concise investigations."""

    def __init__(self, *, max_steps_for_bonus: int = 6, bonus: float = 0.3):
        super().__init__()
        self.max_steps_for_bonus = int(max_steps_for_bonus)
        self.bonus = float(bonus)

    def forward(self, x: RubricInput) -> float:
        return self.bonus if x.step_count <= self.max_steps_for_bonus else 0.0


class SpecificityRubric(Rubric):
    """Penalize false quarantines (safe → non-safe)."""

    def __init__(self, *, false_quarantine_penalty: float = -0.6):
        super().__init__()
        self.false_quarantine_penalty = float(false_quarantine_penalty)

    def forward(self, x: RubricInput) -> float:
        if x.actual_attack is AttackClass.SAFE and x.decision is not AttackClass.SAFE:
            return self.false_quarantine_penalty
        return 0.0


class EvidenceRubric(Rubric):
    """Reward unique evidence signals cited in reasoning (+0.1 each)."""

    def __init__(self, *, signals: Iterable[str], per_signal: float = 0.1, max_total: float = 0.8):
        super().__init__()
        self._signals = tuple(dict.fromkeys(signal.strip().lower() for signal in signals if signal.strip()))
        self.per_signal = float(per_signal)
        self.max_total = float(max_total)

    def forward(self, x: RubricInput) -> float:
        text = _normalize_text(x.reasoning)
        if not text:
            return 0.0
        matched = sum(1 for signal in self._signals if signal in text)
        return min(self.max_total, self.per_signal * matched)


class ComposedRubric(Rubric):
    """A transparent, introspectable composition of child rubrics with weights."""

    def __init__(self, *, verdict: Rubric, speed: Rubric, specificity: Rubric, evidence: Rubric, weights: dict[str, float] | None = None):
        super().__init__()
        self.verdict = verdict
        self.speed = speed
        self.specificity = specificity
        self.evidence = evidence
        self.weights = weights or {"verdict": 1.0, "speed": 1.0, "specificity": 1.0, "evidence": 1.0}

    def forward(self, x: RubricInput) -> float:
        scores = self.score_components(x)
        return sum(scores.values())

    def score_components(self, x: RubricInput) -> dict[str, float]:
        verdict = float(self.verdict(x)) * float(self.weights.get("verdict", 1.0))
        speed = float(self.speed(x)) * float(self.weights.get("speed", 1.0))
        specificity = float(self.specificity(x)) * float(self.weights.get("specificity", 1.0))
        evidence = float(self.evidence(x)) * float(self.weights.get("evidence", 1.0))
        return {
            "verdict": verdict,
            "speed": speed,
            "specificity": specificity,
            "evidence": evidence,
        }

