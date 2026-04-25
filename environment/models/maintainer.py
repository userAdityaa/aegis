from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Maintainer(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    email: str
    ip_history: list[str] = Field(default_factory=list)
    commit_times: list[int] = Field(default_factory=list)
    commit_style_fingerprint: dict[str, float] = Field(default_factory=dict)