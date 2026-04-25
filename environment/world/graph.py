from __future__ import annotations

import random

import networkx as nx

from environment.models import Package
from environment.world.package_factory import PackageFactory
from environment.world.personas import PersonaFactory


class DependencyGraph:
    """Builds a synthetic package registry with a DAG dependency structure."""

    _PACKAGE_TOKENS: tuple[str, ...] = (
        "alpha",
        "beta",
        "core",
        "data",
        "delta",
        "edge",
        "flux",
        "guard",
        "hash",
        "index",
        "kappa",
        "logic",
        "mesh",
        "nova",
        "orbit",
        "proxy",
        "quark",
        "relay",
        "stack",
        "trace",
        "utils",
        "vault",
        "watch",
        "xeno",
        "yield",
        "zen",
    )

    def __init__(
        self,
        seed: int | None = None,
        min_packages: int = 50,
        max_packages: int = 200,
        persona_factory: PersonaFactory | None = None,
    ) -> None:
        if min_packages < 1:
            raise ValueError("min_packages must be at least 1")
        if max_packages < min_packages:
            raise ValueError("max_packages must be greater than or equal to min_packages")

        self._rng = random.Random(seed)
        self._min_packages = min_packages
        self._max_packages = max_packages
        self._persona_factory = persona_factory or PersonaFactory(seed=seed)
        self._package_factory = PackageFactory(self._rng)

        self._graph: nx.DiGraph = nx.DiGraph()
        self._packages: dict[str, Package] = {}

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def packages(self) -> dict[str, Package]:
        return self._packages

    def generate(self, num_packages: int | None = None) -> dict[str, Package]:
        """Generate a new synthetic registry snapshot."""

        package_count = num_packages or self._rng.randint(self._min_packages, self._max_packages)
        names = self._generate_package_names(package_count)

        graph = nx.DiGraph()
        graph.add_nodes_from(names)
        for idx, pkg_name in enumerate(names):
            if idx == 0:
                continue
            max_deps = min(4, idx)
            dep_count = self._rng.randint(0, max_deps)
            for dep_name in self._rng.sample(names[:idx], dep_count):
                graph.add_edge(pkg_name, dep_name)

        package_index: dict[str, Package] = {}
        for pkg_name in names:
            maintainers = self._persona_factory.create_maintainers(
                count=self._rng.randint(1, 3),
                package_hint=pkg_name,
            )
            package_index[pkg_name] = self._package_factory.build_package(
                package_name=pkg_name,
                maintainers=maintainers,
                dependencies=list(graph.successors(pkg_name)),
            )

        self._graph = graph
        self._packages = package_index
        return package_index

    def get_package(self, package_name: str) -> Package:
        return self._packages[package_name]

    def get_dependents(self, package_name: str) -> list[str]:
        return sorted(self._graph.predecessors(package_name))

    def get_full_tree(self, package_name: str) -> dict[str, dict]:
        return {package_name: self._build_tree(package_name, visited=set())}

    def _build_tree(self, package_name: str, visited: set[str]) -> dict[str, dict]:
        if package_name in visited:
            return {}

        visited.add(package_name)
        children = sorted(self._graph.successors(package_name))
        return {child: self._build_tree(child, visited.copy()) for child in children}

    def _generate_package_names(self, count: int) -> list[str]:
        names: set[str] = set()
        while len(names) < count:
            left = self._rng.choice(self._PACKAGE_TOKENS)
            right = self._rng.choice(self._PACKAGE_TOKENS)
            sep = self._rng.choice(("-", "_", ""))
            suffix = self._rng.randint(1, 999)
            names.add(f"{left}{sep}{right}{suffix}")
        return sorted(names)
