from __future__ import annotations

from environment.registry import ShadowRegistry
from environment.tools._helpers import latest_version


SUSPICIOUS_PATTERNS = (
    "curl",
    "wget",
    "urllib.request",
    "subprocess",
    "os.system",
    "exec(",
    "eval(",
    "base64",
    "socket.getaddrinfo",
    "fetch(",
)


def inspect_install_script(pkg_name: str, registry: ShadowRegistry) -> dict[str, object]:
    pkg = registry.get_package(pkg_name)
    version = latest_version(pkg)
    setup_text = version.files.get("setup.py", "")
    readme_text = version.files.get("README.md", "")
    combined = "\n".join(part for part in (pkg.install_script, setup_text, readme_text) if part)
    matches = [pattern for pattern in SUSPICIOUS_PATTERNS if pattern in combined]

    return {
        "package": pkg.name,
        "install_script": pkg.install_script,
        "has_network_call": any(token in combined for token in ("curl", "wget", "urllib.request", "socket.getaddrinfo", "fetch(")),
        "has_exec": any(token in combined for token in ("exec(", "eval(", "os.system", "subprocess")),
        "suspicious_patterns": matches,
        "setup_py_excerpt": setup_text[:400],
        "readme_excerpt": readme_text[:400],
    }