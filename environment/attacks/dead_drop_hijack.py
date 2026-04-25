from __future__ import annotations

from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, NetworkCall, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class DeadDropHijackAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.DEAD_DROP_HIJACK
    REQUIRED_TOOLS = ("check_maintainer_history", "diff_versions", "run_sandbox_test")

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        del registry

        latest = self._fork_version(pkg, days_delta=730)
        hijacker = self._create_maintainer(pkg.name, off_hours=True)
        pkg.maintainers = [*pkg.maintainers, hijacker]
        latest.files["__init__.py"] = (
            f'__version__ = "{latest.version}"\n'
            'import socket\n'
            'socket.getaddrinfo("fix.shadow-registry.io", 443)\n'
            'from .core import run\n'
        )
        pkg.network_calls = [
            *pkg.network_calls,
            NetworkCall(host="fix.shadow-registry.io", port=443, protocol="https"),
        ]
        self._append_commit(
            latest,
            author=hijacker.email,
            ip=hijacker.ip_history[0],
            diff_summary="fix import ordering",
            days_delta=0,
            hour=self._rng.choice((0, 1, 2, 3, 23)),
        )
        return self._label(injected_version=latest.version, evidence_field="maintainers")
