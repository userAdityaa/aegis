from __future__ import annotations

from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class TyposquattingAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.TYPOSQUATTING
    REQUIRED_TOOLS = ("get_reputation_score", "check_maintainer_history")

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        reference = max(
            (candidate for name, candidate in registry.packages.items() if name != pkg.name),
            key=lambda candidate: candidate.stars,
            default=pkg,
        )
        original_name = pkg.name
        squatted_name = self._build_typosquat_name(reference.name, registry.packages, original_name)
        imposter = self._create_maintainer(squatted_name)

        latest = self._fork_version(pkg, days_delta=4)
        latest.files["README.md"] = (
            f"# {squatted_name}\n\n"
            f"Compatibility helpers for {reference.name} consumers.\n"
        )
        latest.files["core.py"] = latest.files["core.py"].rstrip() + f"\n\nALIAS_TARGET = \"{reference.name}\"\n"
        self._append_commit(
            latest,
            author=imposter.email,
            ip=imposter.ip_history[0],
            diff_summary="publish compatibility alias",
        )

        pkg.maintainers = [imposter]
        pkg.stars = self._rng.randint(9, 80)
        pkg.open_issues = self._rng.randint(0, 2)
        pkg.closed_issues = self._rng.randint(0, 5)
        pkg.weekly_download_trend = [self._rng.randint(4, 160) for _ in range(12)]

        registry.rename_package(original_name, squatted_name)
        return self._label(injected_version=latest.version, evidence_field="name")

    def _build_typosquat_name(self, reference_name: str, packages: dict[str, Package], original_name: str) -> str:
        candidates: list[str] = []
        if "-" in reference_name:
            candidates.append(reference_name.replace("-", "", 1))
        if "_" in reference_name:
            candidates.append(reference_name.replace("_", "", 1))
        if "-" not in reference_name and len(reference_name) > 4:
            candidates.append(reference_name[:3] + "-" + reference_name[3:])
        if "_" not in reference_name and len(reference_name) > 4:
            candidates.append(reference_name[:3] + "_" + reference_name[3:])
        if len(reference_name) > 3:
            candidates.append(reference_name[:2] + reference_name[3] + reference_name[2] + reference_name[4:])
        if len(reference_name) > 2:
            candidates.append(reference_name[:2] + reference_name[1] + reference_name[2:])
        if "o" in reference_name:
            candidates.append(reference_name.replace("o", "0", 1))
        if "l" in reference_name:
            candidates.append(reference_name.replace("l", "1", 1))

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen or candidate == original_name:
                continue
            seen.add(candidate)
            if candidate not in packages:
                return candidate

        fallback = reference_name + reference_name[-1]
        while fallback in packages or fallback == original_name:
            fallback += str(self._rng.randint(1, 9))
        return fallback
