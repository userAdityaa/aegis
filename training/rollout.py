from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from training.env_client import AegisEnvClient
from training.parsing import parse_tool_calls, parse_verdict
from training.prompting import build_system_prompt
from training.types import EpisodeTrace, ToolObservation

Policy = Callable[[dict[str, object], list[ToolObservation]], str]


@dataclass(slots=True)
class RolloutSample:
    prompt: str
    completion: str
    reward: float
    trace: EpisodeTrace


def rollout_episode(
    client: AegisEnvClient,
    policy: Policy,
    *,
    attack_class: str = "random",
    package_name: str | None = None,
    seed: int | None = None,
    max_steps: int = 7,
    force_verdict_on_timeout: bool = False,
) -> EpisodeTrace:
    state = client.reset(attack_class=attack_class, package_name=package_name, seed=seed)
    for _ in range(max_steps):
        response = policy(state, client.observations)
        parsed_verdict = parse_verdict(response)
        if parsed_verdict is not None:
            return client.submit_verdict(parsed_verdict.decision, parsed_verdict.reasoning)

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            raise ValueError("Policy output must contain either a tool call or a final verdict.")

        supported_tools = set(client.available_tools())
        for tool_call in tool_calls:
            # Some model checkpoints hallucinate tool names. Ignore unsupported tool calls
            # instead of hard-failing the entire evaluation run.
            if tool_call.name not in supported_tools:
                continue
            client.call_tool(tool_call.name, tool_call.arguments)

    if not force_verdict_on_timeout:
        raise RuntimeError(f"Policy exceeded max_steps={max_steps} without submitting a verdict.")

    # Imported lazily to avoid circular imports (baseline imports rollout for its CLI helpers).
    from training.baseline import infer_verdict_from_observations

    decision, reasoning = infer_verdict_from_observations(client.observations)
    timeout_reasoning = f"[timeout after {max_steps} steps] {reasoning}"
    return client.submit_verdict(decision, timeout_reasoning)


def build_rollout_sample(
    client: AegisEnvClient,
    policy: Policy,
    *,
    attack_class: str = "random",
    package_name: str | None = None,
    seed: int | None = None,
    max_steps: int = 7,
) -> RolloutSample:
    trace = rollout_episode(
        client,
        policy,
        attack_class=attack_class,
        package_name=package_name,
        seed=seed,
        max_steps=max_steps,
    )
    prompt = build_system_prompt(client.manifest_text()) + f"\n\nTarget package: {trace.target_pkg}"
    completion = trace.verdict_response["decision_received"]
    return RolloutSample(
        prompt=prompt,
        completion=completion,
        reward=trace.reward.total,
        trace=trace,
    )