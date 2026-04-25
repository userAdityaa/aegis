from __future__ import annotations

from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class AccountTakeoverAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.ACCOUNT_TAKEOVER
    REQUIRED_TOOLS = ("check_maintainer_history",)

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        del registry

        latest = self._fork_version(pkg, days_delta=6)
        suspicious_ip = self._external_ip(pkg)
        latest.files["core.py"] = latest.files["core.py"].replace(
            "    return ",
            "    audit_value = value.strip()\n    return ",
            1,
        ).replace(" + value", " + audit_value", 1)
        self._append_commit(
            latest,
            author=self._primary_author(pkg),
            ip=suspicious_ip,
            diff_summary="normalize release path",
            hour=self._rng.choice((1, 2, 3, 4)),
        )
        self._append_commit(
            latest,
            author=self._primary_author(pkg),
            ip=suspicious_ip,
            diff_summary="fix release headers",
            days_delta=0,
            hour=self._rng.choice((1, 2, 3, 4)),
        )
        return self._label(injected_version=latest.version, evidence_field="version_history[-1].commits")
