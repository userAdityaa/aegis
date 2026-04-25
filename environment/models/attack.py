from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class AttackClass(str, Enum):
    TYPOSQUATTING = "typosquatting"
    LONG_CON = "long_con"
    ACCOUNT_TAKEOVER = "account_takeover"
    CICD_POISONING = "cicd_poisoning"
    DEPENDENCY_CONFUSION = "dependency_confusion"
    METADATA_INJECTION = "metadata_injection"
    STAR_JACKING = "star_jacking"
    DEAD_DROP_HIJACK = "dead_drop_hijack"
    SAFE = "safe"


class AttackLabel(BaseModel):
    attack_class: AttackClass
    injected_version: str | None = None
    evidence_field: str | None = None