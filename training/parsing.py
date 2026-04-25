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
_VERDICT_PATTERN = re.compile(
    r"<verdict>\s*(?P<decision>[a-z_]+)\s*</verdict>\s*<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>",
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
    return calls


def parse_verdict(text: str) -> ParsedVerdict | None:
    match = _VERDICT_PATTERN.search(text)
    if match is None:
        return None
    return ParsedVerdict(
        decision=AttackClass(match.group("decision").strip()),
        reasoning=match.group("reasoning").strip(),
    )


def render_tool_call(name: str, arguments: dict[str, object] | None = None) -> str:
    encoded = json.dumps(arguments or {}, sort_keys=True)
    return f"<tool>{name}</tool><args>{encoded}</args>"


def render_verdict(decision: str | AttackClass, reasoning: str) -> str:
    normalized = decision.value if isinstance(decision, AttackClass) else decision
    return f"<verdict>{normalized}</verdict><reasoning>{reasoning.strip()}</reasoning>"