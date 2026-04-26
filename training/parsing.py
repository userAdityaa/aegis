from __future__ import annotations

import json
import re
from dataclasses import dataclass

from environment.models import AttackClass
from training.types import ToolCall

_TOOL_CALL_PATTERN = re.compile(
    r"<tool>\s*(?P<tool>[a-zA-Z0-9_:-]+)\s*</tool>\s*<args>\s*(?P<args>\{.*?\})\s*</args>",
    re.DOTALL,
)
_TOOL_CALL_JSON_PATTERN = re.compile(
    r'(?P<blob>\{[\s\S]*?"name"\s*:\s*"(?P<tool>[a-zA-Z0-9_:-]+)"[\s\S]*?\})',
    re.DOTALL,
)
_VERDICT_PATTERN = re.compile(
    r"<verdict>\s*(?P<decision>[a-z_]+)\s*</verdict>\s*<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>",
    re.DOTALL,
)
_FINAL_VERDICT_PATTERN = re.compile(
    r"final_verdict\s*\(\s*decision\s*=\s*(?P<decision>(?:\"[^\"]+\"|'[^']+'|[a-zA-Z_][a-zA-Z0-9_]*))\s*,\s*reasoning\s*=\s*(?P<reasoning>(?:\"[\s\S]*?\"|'[\s\S]*?'))\s*\)",
    re.DOTALL,
)
_FINAL_VERDICT_JSON_PATTERN = re.compile(
    r"final_verdict\s*\(\s*(?P<args>\{[\s\S]*?\})\s*\)",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ParsedVerdict:
    decision: AttackClass
    reasoning: str


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        arguments = json.loads(match.group("args"))
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        calls.append(ToolCall(name=match.group("tool"), arguments=arguments))
    if calls:
        return calls

    # Fallback for models that emit JSON-ish tool call blobs (common with tool-aware chat templates).
    # We keep this permissive because training smoke runs should not hard-fail on formatting noise.
    for match in _TOOL_CALL_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group("blob"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = payload.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            continue
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def parse_verdict(text: str) -> ParsedVerdict | None:
    match = _VERDICT_PATTERN.search(text)
    if match is not None:
        decision_text = match.group("decision").strip()
        # Models sometimes echo the prompt's placeholder instead of a real class.
        if decision_text in {"attack_class_or_safe", "attack_class", "class"}:
            return None
        try:
            decision = AttackClass(decision_text)
        except ValueError:
            return None
        return ParsedVerdict(
            decision=decision,
            reasoning=match.group("reasoning").strip(),
        )

    # Fallback for prompt text that asks for `final_verdict(decision=..., reasoning=...)`.
    match = _FINAL_VERDICT_PATTERN.search(text)
    if match is not None:
        raw_decision = match.group("decision").strip()
        if (raw_decision.startswith('"') and raw_decision.endswith('"')) or (
            raw_decision.startswith("'") and raw_decision.endswith("'")
        ):
            raw_decision = raw_decision[1:-1].strip()
        if raw_decision in {"attack_class_or_safe", "attack_class", "class"}:
            return None

        raw_reasoning = match.group("reasoning").strip()
        if (raw_reasoning.startswith('"') and raw_reasoning.endswith('"')) or (
            raw_reasoning.startswith("'") and raw_reasoning.endswith("'")
        ):
            raw_reasoning = raw_reasoning[1:-1]
        try:
            decision = AttackClass(raw_decision)
        except ValueError:
            return None
        return ParsedVerdict(decision=decision, reasoning=raw_reasoning.strip())

    # Fallback for `final_verdict({ "decision": "...", "reasoning": "..." })`.
    match = _FINAL_VERDICT_JSON_PATTERN.search(text)
    if match is not None:
        try:
            args = json.loads(match.group("args"))
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        raw_decision = args.get("decision")
        raw_reasoning = args.get("reasoning", "")
        if not isinstance(raw_decision, str):
            return None
        if not isinstance(raw_reasoning, str):
            raw_reasoning = str(raw_reasoning)
        if raw_decision in {"attack_class_or_safe", "attack_class", "class"}:
            return None
        try:
            decision = AttackClass(raw_decision.strip())
        except ValueError:
            return None
        return ParsedVerdict(decision=decision, reasoning=raw_reasoning.strip())

    return None


def render_tool_call(name: str, arguments: dict[str, object] | None = None) -> str:
    encoded = json.dumps(arguments or {}, sort_keys=True)
    return f"<tool>{name}</tool><args>{encoded}</args>"


def render_verdict(decision: str | AttackClass, reasoning: str) -> str:
    normalized = decision.value if isinstance(decision, AttackClass) else decision
    return f"<verdict>{normalized}</verdict><reasoning>{reasoning.strip()}</reasoning>"