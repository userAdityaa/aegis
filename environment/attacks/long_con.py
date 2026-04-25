from __future__ import annotations

from base64 import b64encode
from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class LongConAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.LONG_CON
    REQUIRED_TOOLS = ("check_maintainer_history", "diff_versions")

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        del registry

        latest = self._fork_version(pkg, days_delta=self._rng.randint(30, 90))
        payload = b64encode(
            b"import socket;socket.getaddrinfo('cache.shadow-registry.io', 443)"
        ).decode()
        latest.files["core.py"] = latest.files["core.py"].rstrip() + f"\n\n_CACHE_TAG = \"{payload}\"\n"
        self._append_commit(
            latest,
            author=self._primary_author(pkg),
            ip=self._external_ip(pkg),
            diff_summary="trim cache fallback",
            hour=self._rng.choice((0, 1, 2, 3, 23)),
        )
        return self._label(
            injected_version=latest.version,
            evidence_field="version_history[-1].files['core.py']",
        )
