from __future__ import annotations

import random
from dataclasses import dataclass, field

from environment import ShadowRegistry
from environment.attacks import ALL_ATTACKS
from environment.models import AttackClass, Episode
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
CURRENT_CASEFILE_URI = "aegis://casefile/current"
CURRENT_INBOX_URI = "aegis://inbox/current"


@dataclass
class EpisodeAudit:
    episode_id: str | None = None
    target_pkg: str | None = None
    step_count: int = 0
    tools_used: list[str] = field(default_factory=list)
    verdict_submitted: bool = False
    case_notes: list[str] = field(default_factory=list)
    incident_updates_sent: int = 0


class AegisRuntime:
    def __init__(self, seed: int | None = None) -> None:
        self.registry = ShadowRegistry(seed=seed)
        self.audit = EpisodeAudit()
        self._casefile_notes: list[str] = []
        self._incident_inbox: list[dict[str, object]] = []
        self._draft_replies: dict[str, str] = {}
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
        self._casefile_notes = []
        self._incident_inbox = self._build_incident_inbox(current_episode.target_pkg)
        self._draft_replies = {}
        return self.registry.get_observable_state(current_episode.target_pkg)

    def read_current_episode(self) -> dict[str, object]:
        if self.registry.current_episode is None:
            return self.start_episode()
        return self.registry.get_observable_state(self.registry.current_episode.target_pkg)

    def read_casefile(self) -> dict[str, object]:
        current_episode = self._require_current_episode()
        return {
            "episode_id": current_episode.episode_id,
            "target_pkg": current_episode.target_pkg,
            "notes": list(self._casefile_notes),
            "notes_count": len(self._casefile_notes),
        }

    def append_case_note(self, note: str) -> dict[str, object]:
        self._record_tool_call("append_case_note")
        trimmed = note.strip()
        if not trimmed:
            raise ValueError("note must be non-empty")
        if len(trimmed) > 800:
            raise ValueError("note is too long (max 800 chars)")
        self._casefile_notes.append(trimmed)
        self.audit.case_notes = list(self._casefile_notes)
        return {"ok": True, "notes_count": len(self._casefile_notes)}

    def list_incident_inbox(self) -> dict[str, object]:
        self._record_tool_call("list_incident_inbox")
        return {
            "messages": [
                {k: v for k, v in message.items() if k != "body"}
                for message in self._incident_inbox
            ],
            "count": len(self._incident_inbox),
        }

    def read_incident_message(self, message_id: str) -> dict[str, object]:
        self._record_tool_call("read_incident_message")
        message = self._get_incident_message(message_id)
        return dict(message)

    def draft_incident_reply(self, message_id: str, intent: str) -> dict[str, object]:
        self._record_tool_call("draft_incident_reply")
        message = self._get_incident_message(message_id)
        reply = self._render_reply(message=message, intent=intent)
        self._draft_replies[message_id] = reply
        return {"message_id": message_id, "draft": reply}

    def send_incident_reply(self, message_id: str) -> dict[str, object]:
        self._record_tool_call("send_incident_reply")
        if message_id not in self._draft_replies:
            raise ValueError("No draft found for message_id. Call draft_incident_reply first.")
        self.audit.incident_updates_sent += 1
        return {"ok": True, "message_id": message_id, "sent": True}

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

    def _build_incident_inbox(self, target_pkg: str) -> list[dict[str, object]]:
        """Theme 3.2: a lightweight, personalized 'stakeholder inbox' tied to the current episode."""
        return [
            {
                "id": "msg-customer",
                "from": "customer-success@vercel.example",
                "to": "security@claude.example",
                "subject": f"URGENT: Customer reports suspicious install for {target_pkg}",
                "priority": "high",
                "body": (
                    f"We got an escalation: a customer claims installing `{target_pkg}` triggered unexpected network calls. "
                    "Can we confirm impact and advise next steps? Please reply with a short update they can forward."
                ),
            },
            {
                "id": "msg-internal",
                "from": "oncall@claude.example",
                "to": "security@claude.example",
                "subject": f"On-call handoff: investigate {target_pkg}",
                "priority": "normal",
                "body": (
                    "Please keep a running case file with what you checked and why. "
                    "Before you close the incident, send a 3-bullet executive update."
                ),
            },
        ]

    def _get_incident_message(self, message_id: str) -> dict[str, object]:
        for message in self._incident_inbox:
            if str(message.get("id")) == message_id:
                return message
        raise KeyError(f"Unknown message_id: {message_id}")

    def _render_reply(self, *, message: dict[str, object], intent: str) -> str:
        normalized_intent = intent.strip().lower()
        if normalized_intent not in {"status_update", "customer_guidance", "exec_summary"}:
            raise ValueError("intent must be one of: status_update, customer_guidance, exec_summary")

        target_pkg = str(self._require_current_episode().target_pkg)
        notes_preview = "; ".join(self._casefile_notes[-3:]) if self._casefile_notes else "no case notes yet"
        if normalized_intent == "exec_summary":
            return (
                f"Executive update on `{target_pkg}`:\n"
                f"- Current status: investigating, evidence gathered via env tools.\n"
                f"- Key findings: {notes_preview}.\n"
                "- Next steps: validate indicators, confirm attack class, publish guidance and mitigation.\n"
            )
        if normalized_intent == "customer_guidance":
            return (
                f"Customer guidance for `{target_pkg}`:\n"
                "- We are investigating a potential supply-chain issue.\n"
                "- As a precaution: pin versions, audit install scripts, and monitor outbound network calls during install.\n"
                "- We will follow up with confirmed indicators and remediation steps.\n"
            )
        return (
            f"Status update: investigating `{target_pkg}`. Latest notes: {notes_preview}. "
            "Will share a confirmed classification and remediation shortly."
        )

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


AegisMCPRuntime = AegisRuntime