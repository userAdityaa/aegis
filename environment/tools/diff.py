from __future__ import annotations

from difflib import unified_diff

from environment.registry import ShadowRegistry
from environment.tools._helpers import find_version


def diff_versions(pkg_name: str, v1: str, v2: str, registry: ShadowRegistry) -> dict[str, object]:
    pkg = registry.get_package(pkg_name)
    left_version = find_version(pkg, v1)
    right_version = find_version(pkg, v2)

    changed_files: list[str] = []
    diff_chunks: list[str] = []
    file_names = sorted(set(left_version.files) | set(right_version.files))

    for file_name in file_names:
        left_text = left_version.files.get(file_name, "")
        right_text = right_version.files.get(file_name, "")
        if left_text == right_text:
            continue

        changed_files.append(file_name)
        diff_chunks.extend(
            unified_diff(
                left_text.splitlines(),
                right_text.splitlines(),
                fromfile=f"{pkg.name}:{v1}:{file_name}",
                tofile=f"{pkg.name}:{v2}:{file_name}",
                lineterm="",
            )
        )

    return {
        "package": pkg.name,
        "from_version": v1,
        "to_version": v2,
        "changed_files": changed_files,
        "diff": "\n".join(diff_chunks),
    }