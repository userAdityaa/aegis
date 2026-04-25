from __future__ import annotations

from pathlib import Path


def load_manifest_text(manifest_path: str | Path | None = None) -> str:
    path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parents[1] / "openenv.yaml"
    return path.read_text(encoding="utf-8")


def build_system_prompt(manifest_text: str | None = None, manifest_path: str | Path | None = None) -> str:
    resolved_manifest = manifest_text if manifest_text is not None else load_manifest_text(manifest_path)
    return (
        "You are a software supply-chain forensic investigator.\n"
        "Investigate exactly one package per episode, call tools only when needed, and finish with a final verdict.\n"
        "Use this tool-call format when you need evidence:\n"
        "<tool>tool_name</tool><args>{\"optional\": \"json args\"}</args>\n"
        "Use this verdict format when you are ready to stop:\n"
        "<verdict>attack_class_or_safe</verdict><reasoning>concise evidence-based rationale</reasoning>\n\n"
        "Environment manifest:\n"
        f"{resolved_manifest}"
    )


def build_agent_training_prompt(manifest_text: str | None = None, manifest_path: str | Path | None = None) -> str:
    del manifest_text, manifest_path
    return (
        "You are a software supply-chain forensic investigator.\n"
        "Investigate exactly one package per episode.\n"
        "Use the available forensic tools to gather evidence before deciding.\n"
        "IMPORTANT: During training, tools are called via the model's native tool-calling interface.\n"
        "Do NOT output XML tags like <tool>...</tool> or <verdict>...</verdict>.\n"
        "Instead:\n"
        "- Call investigation tools (e.g., check_maintainer_history, diff_versions, inspect_install_script, trace_dependencies, get_reputation_score, run_sandbox_test).\n"
        "- When ready to stop, call final_verdict(decision=..., reasoning=...) exactly once.\n"
        "Keep reasoning concise, evidence-based, and consistent with tool outputs."
    )
