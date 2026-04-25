from __future__ import annotations

from pydantic import BaseModel, Field

from .attack import AttackLabel


class Episode(BaseModel):
    episode_id: str
    target_pkg: str
    package_count: int = Field(ge=1)
    attack_label: AttackLabel | None = None