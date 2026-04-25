from __future__ import annotations

from environment.registry import ShadowRegistry
from environment.tools._helpers import latest_version


def trace_dependencies(pkg_name: str, registry: ShadowRegistry) -> dict[str, object]:
    registry.get_package(pkg_name)
    tree = registry.get_full_tree(pkg_name)
    flat_dependencies: list[dict[str, object]] = []
    seen: set[str] = set()

    def walk(current_name: str) -> None:
        package = registry.get_package(current_name)
        for dependency_name in package.dependencies:
            if dependency_name in seen:
                continue
            seen.add(dependency_name)
            dependency_pkg = registry.get_package(dependency_name)
            flat_dependencies.append(
                {
                    "name": dependency_name,
                    "source": dependency_pkg.source,
                    "latest_version": latest_version(dependency_pkg).version,
                    "maintainer_count": len(dependency_pkg.maintainers),
                }
            )
            walk(dependency_name)

    walk(pkg_name)
    return {
        "package": pkg_name,
        "tree": tree,
        "flat_dependencies": flat_dependencies,
        "dependency_count": len(flat_dependencies),
    }