from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone

from environment.models import CommitRecord, Maintainer, Package, PackageVersion


class PackageFactory:
    _DIFF_SUMMARIES: tuple[str, ...] = (
        "refactor parser helper",
        "harden input validation",
        "update packaging metadata",
        "optimize cache lookup",
        "improve retry behavior",
        "fix edge case in serializer",
        "bump lockfile",
    )

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def build_package(self, package_name: str, maintainers: list[Maintainer], dependencies: list[str]) -> Package:
        versions = self._build_versions(package_name=package_name, maintainers=maintainers)
        base_downloads = self._rng.randint(120, 30_000)
        weekly_download_trend = [max(0, int(base_downloads * self._rng.uniform(0.7, 1.3))) for _ in range(12)]

        return Package(
            name=package_name,
            source=self._rng.choices(("public", "private"), weights=(0.8, 0.2), k=1)[0],
            version_history=versions,
            maintainers=maintainers,
            dependencies=sorted(dependencies),
            install_script="python -m pip install .",
            stars=self._rng.randint(10, 12_000),
            open_issues=self._rng.randint(0, 180),
            closed_issues=self._rng.randint(5, 650),
            weekly_download_trend=weekly_download_trend,
            network_calls=[],
        )

    def _build_versions(self, package_name: str, maintainers: list[Maintainer]) -> list[PackageVersion]:
        version_count = self._rng.randint(1, 4)
        minor = self._rng.randint(0, 4)
        base_time = datetime.now(timezone.utc) - timedelta(days=self._rng.randint(320, 1500))

        versions: list[PackageVersion] = []
        current_time = base_time

        for patch in range(version_count):
            version = f"1.{minor}.{patch}"
            commit_count = self._rng.randint(5, 30)
            commits: list[CommitRecord] = []

            for _ in range(commit_count):
                maintainer = self._rng.choice(maintainers)
                current_time += timedelta(days=self._rng.randint(1, 4))
                hour = self._rng.choice(maintainer.commit_times or [10])
                timestamp = current_time.replace(
                    hour=hour,
                    minute=self._rng.randint(0, 59),
                    second=self._rng.randint(0, 59),
                    microsecond=0,
                )

                commits.append(
                    CommitRecord(
                        hash=uuid.uuid4().hex[:12],
                        timestamp=timestamp,
                        ip=self._rng.choice(maintainer.ip_history or ["127.0.0.1"]),
                        diff_summary=self._rng.choice(self._DIFF_SUMMARIES),
                        author=maintainer.email,
                    )
                )

            versions.append(
                PackageVersion(
                    version=version,
                    timestamp=commits[-1].timestamp,
                    files=self._build_version_files(package_name=package_name, version=version),
                    commits=commits,
                )
            )

        return versions

    def _build_version_files(self, package_name: str, version: str) -> dict[str, str]:
        return {
            "__init__.py": f"__version__ = \"{version}\"\nfrom .core import run\n",
            "core.py": (
                "def run(value: str) -> str:\n"
                f"    return \"{package_name}:{version}:\" + value\n"
            ),
            "README.md": f"# {package_name}\n\nStable utilities package.\n",
        }