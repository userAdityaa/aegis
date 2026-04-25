from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from environment.attacks.base import BaseAttack
from environment.models import AttackClass, AttackLabel, Package, PackageVersion

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class DependencyConfusionAttack(BaseAttack):
    ATTACK_CLASS = AttackClass.DEPENDENCY_CONFUSION
    REQUIRED_TOOLS = ("trace_dependencies", "get_reputation_score")

    def inject(self, pkg: Package, registry: ShadowRegistry) -> AttackLabel:
        dependency_name = self._dependency_name(registry.packages)
        maintainer = self._create_maintainer(dependency_name)
        latest = self._latest_version(pkg)
        dependency_version = PackageVersion(
            version="99.0.0",
            timestamp=latest.timestamp + timedelta(days=1),
            files={
                "__init__.py": '__version__ = "99.0.0"\nfrom .core import bootstrap\n',
                "core.py": "def bootstrap() -> str:\n    return \"corp shim\"\n",
                "README.md": f"# {dependency_name}\n\nShared internal auth helpers.\n",
            },
            commits=[],
        )
        self._append_commit(
            dependency_version,
            author=maintainer.email,
            ip=maintainer.ip_history[0],
            diff_summary="publish compatibility build",
            days_delta=0,
        )
        dependency_pkg = Package(
            name=dependency_name,
            source="public",
            version_history=[dependency_version],
            maintainers=[maintainer],
            dependencies=[],
            install_script="python -m pip install .",
            stars=self._rng.randint(0, 12),
            open_issues=0,
            closed_issues=0,
            weekly_download_trend=[self._rng.randint(0, 8) for _ in range(12)],
            network_calls=[],
        )

        registry.add_package(dependency_pkg)
        registry.set_dependencies(pkg.name, [*pkg.dependencies, dependency_name])
        return self._label(injected_version=dependency_version.version, evidence_field="dependencies")

    def _dependency_name(self, packages: dict[str, Package]) -> str:
        for candidate in ("corp-auth", "corp-build", "internal-token", "team-config"):
            if candidate not in packages:
                return candidate

        suffix = self._rng.randint(10, 99)
        while f"corp-auth-{suffix}" in packages:
            suffix += 1
        return f"corp-auth-{suffix}"
