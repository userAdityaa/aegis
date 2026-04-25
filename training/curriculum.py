from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from environment.models import AttackClass


@dataclass(slots=True)
class CurriculumState:
    unlocked_tier: int = 1
    tier1_accuracy: float = 0.0
    tier2_accuracy: float = 0.0


class CurriculumScheduler:
    """Gate harder attacks behind tier accuracy thresholds.

    This is deliberately lightweight: it derives tier accuracies from the
    per-episode event log written during training.
    """

    TIER_1 = (
        AttackClass.ACCOUNT_TAKEOVER,
        AttackClass.TYPOSQUATTING,
        AttackClass.STAR_JACKING,
        AttackClass.SAFE,
    )
    TIER_2 = (
        AttackClass.CICD_POISONING,
        AttackClass.DEPENDENCY_CONFUSION,
        AttackClass.METADATA_INJECTION,
    )
    TIER_3 = (
        AttackClass.LONG_CON,
        AttackClass.DEAD_DROP_HIJACK,
    )

    def __init__(
        self,
        *,
        evidence_dir: str | Path,
        seed: int = 0,
        tier1_unlock_threshold: float = 0.75,
        tier2_unlock_threshold: float = 0.65,
    ) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.rng = random.Random(seed)
        self.tier1_unlock_threshold = float(tier1_unlock_threshold)
        self.tier2_unlock_threshold = float(tier2_unlock_threshold)

    def state(self) -> CurriculumState:
        return self._compute_state()

    def select_attack(self) -> AttackClass:
        state = self._compute_state()
        if state.unlocked_tier <= 1:
            pool = self.TIER_1
        elif state.unlocked_tier == 2:
            pool = (*self.TIER_1, *self.TIER_2)
        else:
            pool = (*self.TIER_1, *self.TIER_2, *self.TIER_3)
        return self.rng.choice(pool)

    def _compute_state(self) -> CurriculumState:
        path = self.evidence_dir / "per_episode_events.jsonl"
        if not path.exists():
            return CurriculumState()

        tier1_total = tier1_correct = 0
        tier2_total = tier2_correct = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            actual = event.get("actual_attack")
            if not actual:
                continue
            try:
                attack = AttackClass(str(actual))
            except ValueError:
                continue
            correct = bool(event.get("correct", False))
            if attack in self.TIER_1:
                tier1_total += 1
                tier1_correct += int(correct)
            elif attack in self.TIER_2:
                tier2_total += 1
                tier2_correct += int(correct)

        tier1_acc = tier1_correct / tier1_total if tier1_total else 0.0
        tier2_acc = tier2_correct / tier2_total if tier2_total else 0.0
        unlocked = 1
        if tier1_acc >= self.tier1_unlock_threshold:
            unlocked = 2
        if unlocked == 2 and tier2_acc >= self.tier2_unlock_threshold:
            unlocked = 3
        return CurriculumState(unlocked_tier=unlocked, tier1_accuracy=tier1_acc, tier2_accuracy=tier2_acc)

