from __future__ import annotations

import random

from faker import Faker

from environment.models import Maintainer


class PersonaFactory:
    """Generates synthetic maintainer personas with stable behavior traits."""

    _EMAIL_DOMAINS: tuple[str, ...] = (
        "devmail.io",
        "shipit.dev",
        "codeforge.ai",
        "maintainers.net",
        "patchlab.org",
    )

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._faker = Faker()
        if seed is not None:
            self._faker.seed_instance(seed)

    def create_maintainer(self, package_hint: str | None = None) -> Maintainer:
        """Build one maintainer persona."""

        name = self._faker.name()
        email = self._build_email(name, package_hint)

        return Maintainer(
            name=name,
            email=email,
            ip_history=self._generate_ip_cluster(),
            commit_times=self._generate_commit_time_profile(),
            commit_style_fingerprint=self._generate_style_fingerprint(),
        )

    def create_maintainers(self, count: int, package_hint: str | None = None) -> list[Maintainer]:
        """Build multiple maintainer personas."""

        if count < 1:
            return []
        return [self.create_maintainer(package_hint=package_hint) for _ in range(count)]

    def _build_email(self, name: str, package_hint: str | None) -> str:
        local = "".join(ch.lower() for ch in name if ch.isalnum())
        token = str(self._rng.randint(10, 99))
        if package_hint:
            local = f"{local[:8]}{package_hint.replace('-', '')[:4]}{token}"
        domain = self._rng.choice(self._EMAIL_DOMAINS)
        return f"{local}@{domain}"

    def _generate_ip_cluster(self) -> list[str]:
        base_a = self._rng.randint(11, 197)
        base_b = self._rng.randint(0, 255)
        base_c = self._rng.randint(0, 255)
        count = self._rng.randint(2, 4)

        ips: set[str] = set()
        while len(ips) < count:
            ips.add(f"{base_a}.{base_b}.{base_c}.{self._rng.randint(2, 254)}")
        return sorted(ips)

    def _generate_commit_time_profile(self) -> list[int]:
        center = self._rng.choice((9, 10, 11, 13, 14, 15))
        samples = [int(round(self._rng.gauss(center, 1.75))) for _ in range(32)]
        return [max(0, min(23, hour)) for hour in samples]

    def _generate_style_fingerprint(self) -> dict[str, float]:
        return {
            "avg_line_length": round(self._rng.uniform(72, 110), 2),
            "avg_commit_message_length": round(self._rng.uniform(28, 85), 2),
            "comment_ratio": round(self._rng.uniform(0.05, 0.28), 3),
            "punctuation_ratio": round(self._rng.uniform(0.02, 0.1), 3),
            "emoji_ratio": round(self._rng.uniform(0.0, 0.02), 3),
        }
