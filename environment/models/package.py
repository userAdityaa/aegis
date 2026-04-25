from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .maintainer import Maintainer


class NetworkCall(BaseModel):
    host: str
    port: int
    protocol: str


class CommitRecord(BaseModel):
    hash: str
    timestamp: datetime
    ip: str
    diff_summary: str
    author: str


class PackageVersion(BaseModel):
    version: str
    timestamp: datetime
    files: dict[str, str] = Field(default_factory=dict)
    commits: list[CommitRecord] = Field(default_factory=list)


class Package(BaseModel):
    name: str
    source: str = "public"
    version_history: list[PackageVersion] = Field(default_factory=list)
    maintainers: list[Maintainer] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    install_script: str
    stars: int = Field(default=0, ge=0)
    open_issues: int = Field(default=0, ge=0)
    closed_issues: int = Field(default=0, ge=0)
    weekly_download_trend: list[int] = Field(default_factory=list)
    network_calls: list[NetworkCall] = Field(default_factory=list)