from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

from environment.models import AttackClass, AttackLabel, CommitRecord, Maintainer, Package, PackageVersion

if TYPE_CHECKING:
    from environment.registry import ShadowRegistry


class AttackInjector(Protocol):
    def inject(self, pkg: Package, registry: "ShadowRegistry") -> AttackLabel:
        ...


class BaseAttack(ABC):
    ATTACK_CLASS: AttackClass
    REQUIRED_TOOLS: tuple[str, ...] = ()

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def inject(self, pkg: Package, registry: "ShadowRegistry") -> AttackLabel:
        raise NotImplementedError

    def _label(self, *, injected_version: str | None = None, evidence_field: str | None = None) -> AttackLabel:
        return AttackLabel(
            attack_class=self.ATTACK_CLASS,
            injected_version=injected_version,
            evidence_field=evidence_field,
        )

    def _latest_version(self, pkg: Package) -> PackageVersion:
        return pkg.version_history[-1]

    def _fork_version(self, pkg: Package, *, days_delta: int = 2) -> PackageVersion:
        previous = self._latest_version(pkg)
        version = previous.model_copy(deep=True)
        version.version = self._next_version(previous.version)
        version.timestamp = previous.timestamp + timedelta(days=days_delta)
        init_file = version.files.get("__init__.py")
        if init_file is not None:
            version.files["__init__.py"] = init_file.replace(previous.version, version.version)
        pkg.version_history.append(version)
        return version

    def _append_commit(
        self,
        version: PackageVersion,
        *,
        author: str,
        ip: str,
        diff_summary: str,
        days_delta: int = 1,
        hour: int | None = None,
    ) -> CommitRecord:
        timestamp = version.timestamp + timedelta(days=days_delta)
        if hour is not None:
            timestamp = timestamp.replace(
                hour=hour,
                minute=self._rng.randint(0, 59),
                second=self._rng.randint(0, 59),
                microsecond=0,
            )
        commit = CommitRecord(
            hash=f"{self._rng.getrandbits(48):012x}",
            timestamp=timestamp,
            ip=ip,
            diff_summary=diff_summary,
            author=author,
        )
        version.commits.append(commit)
        version.timestamp = commit.timestamp
        return commit

    def _next_version(self, version: str) -> str:
        parts = version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            return f"{version}.1"
        major, minor, patch = (int(part) for part in parts)
        return f"{major}.{minor}.{patch + 1}"

    def _primary_author(self, pkg: Package) -> str:
        return pkg.maintainers[0].email

    def _known_ips(self, pkg: Package) -> set[str]:
        known_ips = {commit.ip for version in pkg.version_history for commit in version.commits}
        for maintainer in pkg.maintainers:
            known_ips.update(maintainer.ip_history)
        return known_ips

    def _external_ip(self, pkg: Package) -> str:
        known_ips = self._known_ips(pkg)
        while True:
            candidate = ".".join(str(self._rng.randint(2, 254)) for _ in range(4))
            if candidate not in known_ips:
                return candidate

    def _create_maintainer(self, package_name: str, *, off_hours: bool = False) -> Maintainer:
        name_root = package_name.replace("-", " ").replace("_", " ").title()[:18] or "Package"
        email_root = package_name.replace("-", "").replace("_", "")[:10] or "package"
        commit_hours = [0, 1, 2, 3, 22, 23] if off_hours else [9, 10, 11, 13, 14, 15]
        ip_prefix = [self._rng.randint(11, 197), self._rng.randint(0, 255), self._rng.randint(0, 255)]
        ip_history = [f"{ip_prefix[0]}.{ip_prefix[1]}.{ip_prefix[2]}.{self._rng.randint(2, 254)}" for _ in range(3)]

        return Maintainer(
            name=f"{name_root} Maintainer",
            email=f"{email_root}{self._rng.randint(10, 99)}@patchmail.dev",
            ip_history=sorted(set(ip_history)),
            commit_times=commit_hours,
            commit_style_fingerprint={
                "avg_line_length": round(self._rng.uniform(74, 106), 2),
                "avg_commit_message_length": round(self._rng.uniform(24, 72), 2),
                "comment_ratio": round(self._rng.uniform(0.04, 0.22), 3),
                "punctuation_ratio": round(self._rng.uniform(0.02, 0.09), 3),
                "emoji_ratio": 0.0,
            },
        )