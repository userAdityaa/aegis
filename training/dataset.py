from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from environment.models import AttackClass
from training.env_client import AegisEnvClient
from training.prompting import build_agent_training_prompt, load_manifest_text


@dataclass(slots=True)
class TrainingPromptRow:
    prompt: list[dict[str, str]]
    attack_class: str
    seed: int
    package_name: str
    episode_index: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_training_prompt_rows(
    *,
    episodes_per_attack: int = 2,
    seed: int = 0,
    manifest_path: str | Path | None = None,
    attack_schedule: Sequence[AttackClass | str] | None = None,
) -> list[TrainingPromptRow]:
    manifest_text = load_manifest_text(manifest_path)
    prompt_text = build_agent_training_prompt(manifest_text=manifest_text)
    rows: list[TrainingPromptRow] = []

    for episode_index, attack_class in enumerate(
        _normalize_attack_schedule(attack_schedule, episodes_per_attack=episodes_per_attack)
    ):
        episode_seed = seed + episode_index
        client = AegisEnvClient(seed=episode_seed, manifest_path=manifest_path)
        state = client.reset(attack_class=attack_class.value, seed=episode_seed)
        rows.append(
            TrainingPromptRow(
                prompt=[{"role": "user", "content": prompt_text}],
                attack_class=attack_class.value,
                seed=episode_seed,
                package_name=str(state["target_pkg"]),
                episode_index=episode_index,
            )
        )

    return rows


def summarize_training_prompt_rows(rows: Sequence[TrainingPromptRow]) -> dict[str, object]:
    attack_counts = Counter(row.attack_class for row in rows)
    return {
        "episodes": len(rows),
        "attack_counts": dict(sorted(attack_counts.items())),
        "sample_packages": [row.package_name for row in rows[:5]],
        "sample_seeds": [row.seed for row in rows[:5]],
    }


def write_training_prompt_jsonl(rows: Sequence[TrainingPromptRow], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), sort_keys=True) + "\n")
    return destination


def _normalize_attack_schedule(
    attack_schedule: Sequence[AttackClass | str] | None,
    *,
    episodes_per_attack: int,
) -> list[AttackClass]:
    if attack_schedule is not None:
        return [attack if isinstance(attack, AttackClass) else AttackClass(attack) for attack in attack_schedule]

    schedule: list[AttackClass] = []
    for attack_class in AttackClass:
        schedule.extend([attack_class] * episodes_per_attack)
    return schedule