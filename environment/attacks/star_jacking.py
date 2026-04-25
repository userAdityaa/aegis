from __future__ import annotations

from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class StarJackingAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.STAR_JACKING
    REQUIRED_TOOLS = ("get_reputation_score",)

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        del registry

        pkg.stars = max(pkg.stars, self._rng.randint(8_000, 15_000))
        pkg.open_issues = 0
        pkg.closed_issues = self._rng.randint(0, 3)
        pkg.weekly_download_trend = [self._rng.randint(80, 140) for _ in range(12)]
        return self._label(
            injected_version=self._latest_version(pkg).version,
            evidence_field="stars",
        )
