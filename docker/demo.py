from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import gradio as gr

from environment import ShadowRegistry
from environment.models import AttackClass
from training.baseline import HeuristicBaselinePolicy
from training.env_client import AegisEnvClient
from training.parsing import parse_tool_calls, parse_verdict

DEFAULT_SEED = 17
MAX_STEPS = 7


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Aegis-Env Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Aegis-Env Forensic Demo\n"
            "Pick a synthetic package from the current registry, then watch the simulated agent call forensic tools step by step before submitting a verdict."
        )

        with gr.Row():
            seed_box = gr.Textbox(label="Registry seed", value=str(DEFAULT_SEED))
            attack_box = gr.Dropdown(
                label="Injected attack",
                choices=["random", *[attack.value for attack in AttackClass]],
                value="random",
            )
            refresh_button = gr.Button("Refresh package catalog", variant="secondary")

        with gr.Row():
            package_box = gr.Textbox(label="Package name", placeholder="Select from the catalog or type a known package")
            package_picker = gr.Dropdown(label="Synthetic package catalog", choices=[])
            run_button = gr.Button("Run demo", variant="primary")

        summary_output = gr.Markdown(label="Episode summary")
        transcript_output = gr.JSON(label="Tool transcript")
        transcript_viewer = gr.HTML(label="Side-by-side transcript viewer")

        refresh_button.click(
            fn=refresh_catalog,
            inputs=seed_box,
            outputs=[seed_box, package_picker, package_box],
        )
        package_picker.change(fn=lambda package_name: package_name, inputs=package_picker, outputs=package_box)
        run_button.click(
            fn=run_demo,
            inputs=[seed_box, package_box, package_picker, attack_box],
            outputs=[summary_output, transcript_output, transcript_viewer],
        )
        demo.load(fn=refresh_catalog, inputs=seed_box, outputs=[seed_box, package_picker, package_box])
        demo.load(fn=load_transcript_viewer, outputs=transcript_viewer)

    return demo


def refresh_catalog(seed_text: str) -> tuple[str, gr.Dropdown, str]:
    seed = _resolve_seed(seed_text)
    packages = _list_packages(seed)
    default_package = packages[0] if packages else ""
    return str(seed), gr.Dropdown(choices=packages, value=default_package), default_package


def run_demo(
    seed_text: str,
    package_name: str,
    selected_package: str | None,
    attack_class: str,
) -> tuple[str, list[dict[str, Any]], str]:
    seed = _resolve_seed(seed_text)
    candidate_package = (package_name or "").strip() or (selected_package or "").strip()
    available_packages = _list_packages(seed)
    if not candidate_package:
        candidate_package = available_packages[0] if available_packages else ""
    if candidate_package not in available_packages:
        suggestions = ", ".join(available_packages[:8]) if available_packages else "none"
        message = (
            f"### Unknown package\n"
            f"`{candidate_package}` is not present in the synthetic registry for seed `{seed}`.\n\n"
            f"Try one of: {suggestions}"
        )
        return message, [], load_transcript_viewer()

    trace = _simulate_episode(seed=seed, package_name=candidate_package, attack_class=attack_class)
    summary = _render_summary(trace)
    return summary, trace["steps"], load_transcript_viewer()


def load_transcript_viewer() -> str:
    """Load a generated transcript_viewer.html if present."""
    candidates = [
        "reports/training_evidence/transcript_viewer.html",
        "reports/rubric_smoke2/transcript_viewer.html",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return (
        "<div style='padding:10px'>"
        "<h3>Transcript viewer not generated yet</h3>"
        "<p>Run a training job that writes <code>per_episode_events.jsonl</code> and the app will render "
        "<code>transcript_viewer.html</code> automatically.</p>"
        "</div>"
    )


def _simulate_episode(*, seed: int, package_name: str, attack_class: str) -> dict[str, Any]:
    client = AegisEnvClient(seed=seed)
    policy = HeuristicBaselinePolicy()
    state = client.reset(attack_class=attack_class, package_name=package_name, seed=seed)
    steps: list[dict[str, Any]] = []

    for _ in range(MAX_STEPS):
        response = policy(state, client.observations)
        parsed_verdict = parse_verdict(response)
        if parsed_verdict is not None:
            trace = client.submit_verdict(parsed_verdict.decision, parsed_verdict.reasoning)
            return {
                "episode_id": trace.episode_id,
                "target_pkg": trace.target_pkg,
                "actual_attack": trace.actual_attack.value,
                "decision": trace.decision.value,
                "reasoning": trace.reasoning,
                "reward": trace.reward.as_dict(),
                "tool_names": trace.tool_names,
                "steps": steps,
            }

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            raise RuntimeError("The baseline policy returned neither tool calls nor a final verdict.")

        for tool_call in tool_calls:
            result = client.call_tool(tool_call.name, tool_call.arguments)
            steps.append(
                {
                    "step": len(steps) + 1,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result,
                }
            )

    raise RuntimeError(f"The demo policy exceeded max_steps={MAX_STEPS} without producing a verdict.")


def _render_summary(trace: dict[str, Any]) -> str:
    reward = trace["reward"]
    verdict_line = "correct" if trace["actual_attack"] == trace["decision"] else "incorrect"
    tools_used = ", ".join(trace["tool_names"]) if trace["tool_names"] else "none"
    return (
        f"### Episode `{trace['episode_id']}`\n"
        f"- Target package: `{trace['target_pkg']}`\n"
        f"- Hidden attack: `{trace['actual_attack']}`\n"
        f"- Final verdict: `{trace['decision']}` ({verdict_line})\n"
        f"- Total reward: `{reward['total']:.2f}`\n"
        f"- Tools used: `{tools_used}`\n\n"
        f"**Reasoning**\n\n{trace['reasoning']}"
    )


def _list_packages(seed: int) -> list[str]:
    registry = ShadowRegistry(seed=seed)
    registry.reset(seed=seed)
    return registry.list_packages()


def _resolve_seed(seed_text: str) -> int:
    normalized = (seed_text or "").strip()
    if not normalized:
        return random.randint(0, 1_000_000)
    return int(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Aegis-Env Gradio demo.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()