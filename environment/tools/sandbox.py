from __future__ import annotations

from environment.registry import ShadowRegistry


def run_sandbox_test(pkg_name: str, registry: ShadowRegistry) -> dict[str, object]:
    pkg = registry.get_package(pkg_name)
    process_markers: list[str] = []
    if "python -c" in pkg.install_script:
        process_markers.append("python -c")
    if "python setup.py install" in pkg.install_script:
        process_markers.append("python setup.py install")
    if "pip install" in pkg.install_script:
        process_markers.append("pip install")

    return {
        "package": pkg.name,
        "network_calls": [call.model_dump(mode="json") for call in pkg.network_calls],
        "files_written": [],
        "processes_spawned": process_markers,
        "exit_code": 0,
    }