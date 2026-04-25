from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from environment.models import AttackClass
from environment.runtime import AegisRuntime, CURRENT_EPISODE_URI


def create_app(seed: int | None = None) -> tuple[FastMCP, AegisRuntime]:
    runtime = AegisRuntime(seed=seed)
    mcp = FastMCP(
        name="Aegis-Env",
        instructions=(
            "Investigate one package per episode. Read aegis://episode/current to get the target package, "
            "use the forensic tools to gather evidence, and finish with final_verdict."
        ),
    )

    @mcp.resource(
        CURRENT_EPISODE_URI,
        name="current_episode",
        description="Observable episode state for the active investigation.",
        mime_type="application/json",
    )
    def current_episode() -> dict[str, object]:
        return runtime.read_current_episode()

    @mcp.tool(
        name="start_episode",
        description="Reset the registry and begin a new investigation episode.",
    )
    def start_episode(
        attack_class: str = "random",
        package_name: str | None = None,
        seed: int | None = None,
    ) -> dict[str, object]:
        return runtime.start_episode(attack_class=attack_class, package_name=package_name, seed=seed)

    @mcp.tool(
        name="check_maintainer_history",
        description="Inspect maintainer identity, IP history, and commit timing for a package.",
    )
    def check_maintainer_history(pkg_name: str | None = None) -> dict[str, object]:
        return runtime.check_maintainer_history(pkg_name=pkg_name)

    @mcp.tool(
        name="diff_versions",
        description="Compare two versions of a package. Omitting versions defaults to the latest two.",
    )
    def diff_versions(
        pkg_name: str | None = None,
        v1: str | None = None,
        v2: str | None = None,
    ) -> dict[str, object]:
        return runtime.diff_versions(pkg_name=pkg_name, v1=v1, v2=v2)

    @mcp.tool(
        name="trace_dependencies",
        description="Return the dependency tree and flattened dependency summary for a package.",
    )
    def trace_dependencies(pkg_name: str | None = None) -> dict[str, object]:
        return runtime.trace_dependencies(pkg_name=pkg_name)

    @mcp.tool(
        name="inspect_install_script",
        description="Inspect package installation metadata for suspicious execution or network patterns.",
    )
    def inspect_install_script(pkg_name: str | None = None) -> dict[str, object]:
        return runtime.inspect_install_script(pkg_name=pkg_name)

    @mcp.tool(
        name="get_reputation_score",
        description="Inspect popularity and maintainer signals for suspicious reputation anomalies.",
    )
    def get_reputation_score(pkg_name: str | None = None) -> dict[str, object]:
        return runtime.get_reputation_score(pkg_name=pkg_name)

    @mcp.tool(
        name="run_sandbox_test",
        description="Return simulated runtime activity observed during package installation.",
    )
    def run_sandbox_test(pkg_name: str | None = None) -> dict[str, object]:
        return runtime.run_sandbox_test(pkg_name=pkg_name)

    @mcp.tool(
        name="final_verdict",
        description="Submit the final attack classification for the current episode.",
    )
    def final_verdict(decision: AttackClass, reasoning: str) -> dict[str, object]:
        return runtime.final_verdict(decision=decision, reasoning=reasoning)

    return mcp, runtime


mcp, runtime = create_app()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()