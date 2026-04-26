from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment.models import AttackClass
from training.parsing import parse_tool_calls, parse_verdict, render_tool_call, render_verdict
from training.prompting import build_agent_training_prompt, load_manifest_text
from training.types import ToolObservation


class TransformerTranscriptPolicy:
    _FORENSIC_TOOLS = (
        "check_maintainer_history",
        "diff_versions",
        "inspect_install_script",
        "get_reputation_score",
        "trace_dependencies",
        "run_sandbox_test",
    )

    _AVAILABLE_TOOLS = (
        *_FORENSIC_TOOLS,
        "append_case_note",
        "list_incident_inbox",
        "read_incident_message",
        "draft_incident_reply",
        "send_incident_reply",
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        device: str | None = None,
        max_new_tokens: int = 96,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        resolved_device = _resolve_device(device)
        self.max_new_tokens = int(os.environ.get("AEGIS_EVAL_MAX_NEW_TOKENS", str(max_new_tokens)))

        self.prompt_prefix = self._build_prompt_prefix()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path.as_posix())
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, object] = {}
        if resolved_device.type == "cuda":
            # Keep eval memory low enough for typical Colab GPUs.
            load_kwargs.update(
                {
                    "dtype": torch.float16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                }
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path.as_posix(), **load_kwargs)
        if resolved_device.type != "cuda":
            self.model.to(resolved_device)
        self.model.eval()
        # When device_map="auto" is used, inputs must be placed on the model's first device.
        self.device = next(self.model.parameters()).device
        self.context_window = int(
            getattr(self.model.config, "n_positions", 0)
            or getattr(self.model.config, "max_position_embeddings", 0)
            or 0
        )

    def __call__(self, state: dict[str, object], observations: list[ToolObservation]) -> str:
        prompt = self.render_transcript(state, observations)
        tokenizer_kwargs: dict[str, object] = {"return_tensors": "pt"}
        # Cap prompt length aggressively for Colab-sized GPUs. Qwen models can have very
        # large context windows, and letting prompts grow to that size will OOM during eval.
        max_prompt_tokens = int(os.environ.get("AEGIS_EVAL_MAX_PROMPT_TOKENS", "2048"))
        max_length = max_prompt_tokens
        if self.context_window > self.max_new_tokens:
            max_length = min(max_length, self.context_window - self.max_new_tokens)
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_length

        encoded = self.tokenizer(prompt, **tokenizer_kwargs)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion = self.tokenizer.decode(
            generated[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        action_text = _extract_action_text(completion)
        if parse_verdict(action_text) is not None or parse_tool_calls(action_text):
            return action_text
        return self._fallback_action(observations)

    def render_transcript(self, state: dict[str, object], observations: list[ToolObservation]) -> str:
        target_package = str(state.get("target_pkg", ""))
        max_obs = int(os.environ.get("AEGIS_EVAL_MAX_OBSERVATIONS", "4"))
        recent_observations = observations[-max_obs:] if max_obs > 0 else []
        transcript: list[str] = [
            self.prompt_prefix,
            "",
            f"User: Target package: {target_package}",
            "Use the available forensic tools to gather evidence before deciding.",
            "When you are confident, emit exactly one final verdict in the required verdict format.",
            "",
        ]

        for observation in recent_observations:
            transcript.append(
                f"Assistant: {render_tool_call(observation.call.name, observation.call.arguments)}"
            )
            transcript.append("")
            transcript.append(f"User: Tool result for {observation.call.name}:")
            transcript.append(json.dumps(observation.result, sort_keys=True))
            transcript.append("")

        transcript.append("Assistant:")
        return "\n".join(transcript)

    def _fallback_action(self, observations: list[ToolObservation]) -> str:
        seen_tools = {observation.call.name for observation in observations}
        for tool_name in self._FORENSIC_TOOLS:
            if tool_name not in seen_tools:
                return render_tool_call(tool_name)
        return render_verdict("safe", "model output was unparsable after collecting forensic evidence")

    def _build_prompt_prefix(self) -> str:
        available_tools = ", ".join(self._AVAILABLE_TOOLS)
        return (
            build_agent_training_prompt()
            + "\n"
            + "When you are ready to stop, choose one of these classes for <verdict>: "
            + ", ".join([c.value for c in AttackClass])
            + ".\n"
            + f"Available forensic tools: {available_tools}."
        )


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_action_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("Assistant:"):
        cleaned = cleaned.split("Assistant:", 1)[1].lstrip()

    for marker in ("\n\nUser:", "\nUser:", "\n\nAssistant:", "\nAssistant:"):
        marker_index = cleaned.find(marker)
        if marker_index != -1:
            cleaned = cleaned[:marker_index]
            break

    return cleaned.strip()