from __future__ import annotations

from environment.models import CommitRecord, Package, PackageVersion


def latest_version(pkg: Package) -> PackageVersion:
    return pkg.version_history[-1]


def find_version(pkg: Package, version: str) -> PackageVersion:
    for package_version in pkg.version_history:
        if package_version.version == version:
            return package_version
    raise ValueError(f"Unknown version '{version}' for package '{pkg.name}'")


def commit_to_dict(commit: CommitRecord) -> dict[str, object]:
    return {
        "hash": commit.hash,
        "timestamp": commit.timestamp.isoformat(),
        "ip": commit.ip,
        "diff_summary": commit.diff_summary,
        "author": commit.author,
    }