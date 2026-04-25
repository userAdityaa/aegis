from __future__ import annotations

import random
import uuid

from environment.attacks import AttackInjector
from environment.models import AttackLabel, Episode, Package
from environment.world import DependencyGraph


class ShadowRegistry:
    """Stateful synthetic package registry used by the forensic environment."""

    def __init__(
        self,
        seed: int | None = None,
        min_packages: int = 50,
        max_packages: int = 200,
    ) -> None:
        self._seed = seed
        self._min_packages = min_packages
        self._max_packages = max_packages

        self._rng = random.Random(seed)
        self._graph_builder = DependencyGraph(
            seed=seed,
            min_packages=min_packages,
            max_packages=max_packages,
        )

        self._packages: dict[str, Package] = {}
        self._current_episode: Episode | None = None
        self._attack_label: AttackLabel | None = None

    @property
    def packages(self) -> dict[str, Package]:
        return self._packages

    @property
    def attack_label(self) -> AttackLabel | None:
        """Internal ground truth used only by reward logic."""

        return self._attack_label

    @property
    def current_episode(self) -> Episode | None:
        return self._current_episode

    def reset(self, seed: int | None = None) -> Episode:
        """Regenerate the entire registry and start a fresh episode."""

        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
            self._graph_builder = DependencyGraph(
                seed=seed,
                min_packages=self._min_packages,
                max_packages=self._max_packages,
            )

        self._packages = self._graph_builder.generate()
        target_pkg = self.get_episode_target()

        episode = Episode(
            episode_id=uuid.uuid4().hex,
            target_pkg=target_pkg,
            package_count=len(self._packages),
            attack_label=None,
        )

        self._current_episode = episode
        self._attack_label = None
        return episode

    def list_packages(self) -> list[str]:
        return sorted(self._packages.keys())

    def get_episode_target(self) -> str:
        if not self._packages:
            raise RuntimeError("Registry has not been initialized. Call reset() first.")
        return self._rng.choice(list(self._packages.keys()))

    def get_package(self, package_name: str) -> Package:
        if package_name not in self._packages:
            raise KeyError(f"Unknown package: {package_name}")
        return self._packages[package_name]

    def add_package(self, package: Package) -> None:
        if package.name in self._packages:
            raise ValueError(f"Package already exists: {package.name}")

        self._packages[package.name] = package
        self._graph_builder.graph.add_node(package.name)
        self.set_dependencies(package.name, package.dependencies)
        self._sync_episode_package_count()

    def rename_package(self, old_name: str, new_name: str) -> None:
        if old_name == new_name:
            return
        if new_name in self._packages:
            raise ValueError(f"Package already exists: {new_name}")

        package = self.get_package(old_name)
        package.name = new_name
        self._packages[new_name] = package
        del self._packages[old_name]

        graph = self._graph_builder.graph
        predecessors = list(graph.predecessors(old_name)) if old_name in graph else []
        successors = list(graph.successors(old_name)) if old_name in graph else []
        graph.add_node(new_name)
        graph.add_edges_from((parent, new_name) for parent in predecessors)
        graph.add_edges_from((new_name, child) for child in successors)
        if old_name in graph:
            graph.remove_node(old_name)

        if self._current_episode is not None and self._current_episode.target_pkg == old_name:
            self._current_episode.target_pkg = new_name

    def set_dependencies(self, package_name: str, dependencies: list[str]) -> None:
        package = self.get_package(package_name)
        graph = self._graph_builder.graph
        if package_name not in graph:
            graph.add_node(package_name)

        normalized = sorted(dict.fromkeys(dependencies))
        graph.remove_edges_from((package_name, dep_name) for dep_name in list(graph.successors(package_name)))
        for dep_name in normalized:
            graph.add_node(dep_name)
            graph.add_edge(package_name, dep_name)

        package.dependencies = normalized

    def get_dependents(self, package_name: str) -> list[str]:
        self.get_package(package_name)
        return self._graph_builder.get_dependents(package_name)

    def get_full_tree(self, package_name: str) -> dict[str, dict]:
        self.get_package(package_name)
        return self._graph_builder.get_full_tree(package_name)

    def inject_attack(self, attack: AttackInjector, package_name: str | None = None) -> AttackLabel:
        """Mutate one package with an attack and keep hidden ground truth."""

        target = package_name
        if target is None:
            if self._current_episode is None:
                raise RuntimeError("No active episode. Call reset() first.")
            target = self._current_episode.target_pkg

        package = self.get_package(target)
        label = attack.inject(package, self)
        self._attack_label = label
        return label

    def _sync_episode_package_count(self) -> None:
        if self._current_episode is not None:
            self._current_episode.package_count = len(self._packages)

    def get_observable_state(self, package_name: str) -> dict[str, object]:
        """Return only agent-visible fields without the attack ground truth."""

        self.get_package(package_name)
        return {
            "episode_id": self._current_episode.episode_id if self._current_episode else None,
            "target_pkg": package_name,
        }
