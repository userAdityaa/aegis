from __future__ import annotations

import random
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP

from environment import ShadowRegistry
from environment.attacks import ALL_ATTACKS
from environment.models import AttackClass, Episode, Package
from environment.tools import (
    check_maintainer_history as _check_maintainer_history,
    diff_versions as _diff_versions,
    final_verdict as _final_verdict,
    get_reputation_score as _get_reputation_score,
    inspect_install_script as _inspect_install_script,
    run_sandbox_test as _run_sandbox_test,
    trace_dependencies as _trace_dependencies,
)

CURRENT_EPISODE_URI = "aegis://episode/current"


@dataclass
class EpisodeAudit:
    episode_id: str | None = None
    target_pkg: str | None = None
    step_count: int = 0
    tools_used: list[str] = field(default_factory=list)
    verdict_submitted: bool = False


class AegisMCPRuntime:
    def __init__(self, seed: int | None = None) -> None:
        self.registry = ShadowRegistry(seed=seed)
        self.audit = EpisodeAudit()
        self._attack_types = {
            attack_cls.ATTACK_CLASS.value: attack_cls
            for attack_cls in ALL_ATTACKS
        }

    def start_episode(
        self,
        attack_class: str = "random",
        package_name: str | None = None,
        seed: int | None = None,
    ) -> dict[str, object]:
        episode = self.registry.reset(seed=seed)
        target_pkg = package_name or episode.target_pkg
        self.registry.get_package(target_pkg)
        episode.target_pkg = target_pkg

        normalized_attack = attack_class.strip().lower()
        if normalized_attack == "random":
            attack_cls = self._choose_random_attack(seed=seed)
            self.registry.inject_attack(attack_cls(seed=seed), package_name=target_pkg)
        elif normalized_attack != AttackClass.SAFE.value:
            attack_cls = self._attack_types.get(normalized_attack)
            if attack_cls is None:
                choices = [AttackClass.SAFE.value, "random", *sorted(self._attack_types)]
                raise ValueError(f"Unsupported attack_class '{attack_class}'. Expected one of: {choices}")
            self.registry.inject_attack(attack_cls(seed=seed), package_name=target_pkg)

        current_episode = self._require_current_episode()
        self.audit = EpisodeAudit(
            episode_id=current_episode.episode_id,
            target_pkg=current_episode.target_pkg,
        )
        return self.registry.get_observable_state(current_episode.target_pkg)

    def read_current_episode(self) -> dict[str, object]:
        if self.registry.current_episode is None:
            return self.start_episode()
        return self.registry.get_observable_state(self.registry.current_episode.target_pkg)

    def check_maintainer_history(self, pkg_name: str | None = None) -> dict[str, object]:
        resolved_pkg = self._resolve_package_name(pkg_name)
        self._record_tool_call("check_maintainer_history")
        return _check_maintainer_history(pkg_name=resolved_pkg, registry=self.registry)

    def diff_versions(
        self,
        pkg_name: str | None = None,
        v1: str | None = None,
        v2: str | None = None,
    ) -> dict[str, object]:
        resolved_pkg = self._resolve_package_name(pkg_name)
        left_version, right_version = self._resolve_diff_versions(resolved_pkg, v1=v1, v2=v2)
        self._record_tool_call("diff_versions")
        return _diff_versions(pkg_name=resolved_pkg, v1=left_version, v2=right_version, registry=self.registry)

    def trace_dependencies(self, pkg_name: str | None = None) -> dict[str, object]:
        resolved_pkg = self._resolve_package_name(pkg_name)
        self._record_tool_call("trace_dependencies")
        return _trace_dependencies(pkg_name=resolved_pkg, registry=self.registry)

    def inspect_install_script(self, pkg_name: str | None = None) -> dict[str, object]:
        resolved_pkg = self._resolve_package_name(pkg_name)
        self._record_tool_call("inspect_install_script")
        return _inspect_install_script(pkg_name=resolved_pkg, registry=self.registry)

    def get_reputation_score(self, pkg_name: str | None = None) -> dict[str, object]:
        resolved_pkg = self._resolve_package_name(pkg_name)
        self._record_tool_call("get_reputation_score")
        return _get_reputation_score(pkg_name=resolved_pkg, registry=self.registry)

    def run_sandbox_test(self, pkg_name: str | None = None) -> dict[str, object]:
        resolved_pkg = self._resolve_package_name(pkg_name)
        self._record_tool_call("run_sandbox_test")
        return _run_sandbox_test(pkg_name=resolved_pkg, registry=self.registry)

    def final_verdict(self, decision: AttackClass, reasoning: str) -> dict[str, object]:
        self._record_verdict_submission()
        return _final_verdict(decision=decision.value, reasoning=reasoning, registry=self.registry)

    def _choose_random_attack(self, seed: int | None = None):
        attack_classes = tuple(self._attack_types.values())
        chooser = random.Random(seed)
        return chooser.choice(attack_classes)

    def _resolve_package_name(self, pkg_name: str | None) -> str:
        if pkg_name is not None:
            self.registry.get_package(pkg_name)
            return pkg_name
        current_episode = self._require_current_episode()
        return current_episode.target_pkg

    def _resolve_diff_versions(
        self,
        pkg_name: str,
        *,
        v1: str | None,
        v2: str | None,
    ) -> tuple[str, str]:
        if (v1 is None) != (v2 is None):
            raise ValueError("Provide both v1 and v2, or omit both to diff the latest two versions.")
        if v1 is not None and v2 is not None:
            return v1, v2

        package = self.registry.get_package(pkg_name)
        if len(package.version_history) < 2:
            raise ValueError(f"Package '{pkg_name}' does not have two versions to diff.")
        return package.version_history[-2].version, package.version_history[-1].version

    def _require_current_episode(self) -> Episode:
        current_episode = self.registry.current_episode
        if current_episode is None:
            raise RuntimeError(
                f"No active episode. Read resource '{CURRENT_EPISODE_URI}' or call start_episode first."
            )
        return current_episode

    def _ensure_live_episode(self) -> Episode:
        current_episode = self._require_current_episode()
        if self.audit.verdict_submitted:
            raise RuntimeError("The current episode is terminal. Call start_episode before invoking more tools.")
        return current_episode

    def _record_tool_call(self, tool_name: str) -> None:
        current_episode = self._ensure_live_episode()
        if self.audit.episode_id != current_episode.episode_id:
            self.audit = EpisodeAudit(
                episode_id=current_episode.episode_id,
                target_pkg=current_episode.target_pkg,
            )
        self.audit.step_count += 1
        self.audit.target_pkg = current_episode.target_pkg
        self.audit.tools_used.append(tool_name)

    def _record_verdict_submission(self) -> None:
        current_episode = self._ensure_live_episode()
        if self.audit.episode_id != current_episode.episode_id:
            self.audit = EpisodeAudit(
                episode_id=current_episode.episode_id,
                target_pkg=current_episode.target_pkg,
            )
        self.audit.verdict_submitted = True


def create_app(seed: int | None = None) -> tuple[FastMCP, AegisMCPRuntime]:
    runtime = AegisMCPRuntime(seed=seed)
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