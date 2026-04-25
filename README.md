# Aegis-Env

Aegis-Env is an OpenEnv MCP environment for training LLM agents on realistic package supply-chain forensics. Each episode hides one attack in a synthetic package ecosystem; the agent must gather evidence through tools, maintain state across steps, and submit a final attack verdict.

This repository is aligned to **Theme 3.1 (World Modeling - Professional Tasks)** of the OpenEnv Hackathon. The core task is a professional security workflow in a partially observable world: the agent must interrogate tool outputs, update its beliefs, and end with a single high-stakes classification.

## Problem Statement

Modern coding agents fail on real-world supply-chain incidents because they:

1. Underuse available tools.
2. Jump to verdicts without complete evidence.
3. Lose consistency between observed signals and final reasoning.

Aegis-Env trains those capabilities in a partially observable world with delayed terminal decisions and compositional reward signals.

## Minimum Requirement Status

- OpenEnv environment: yes. Manifest in [openenv.yaml](openenv.yaml), server in [environment/mcp_server.py](environment/mcp_server.py), and shared runtime in [environment/runtime.py](environment/runtime.py).
- Working TRL training script: yes. Entrypoint [training/train.py](training/train.py), GRPO implementation in [training/grpo.py](training/grpo.py), and a re-runnable Colab notebook in [notebooks/aegis_grpo_colab.ipynb](notebooks/aegis_grpo_colab.ipynb).
- Real training evidence with loss and reward plots: yes. See [reports/training_evidence/training_summary.json](reports/training_evidence/training_summary.json), [reports/training_evidence/training_log_history.json](reports/training_evidence/training_log_history.json), and [reports/training_evidence/training_curves.png](reports/training_evidence/training_curves.png).
- Short presentation asset linked from the README: yes. See [docs/hackathon_slide_deck.md](docs/hackathon_slide_deck.md).
- Hugging Face Space packaging: yes. The app is Docker Space-ready via [docker/Dockerfile](docker/Dockerfile) and [docker/demo.py](docker/demo.py).
- Live Hugging Face Space URL: pending deployment. Add the deployed `https://huggingface.co/spaces/...` URL in **Submission Assets** before the final submission.

## Judge Quickstart

1. Install the repo:

```bash
pip install -e .[server,eval,demo]
pip install -e .[training]
```

2. Run the local demo:

```bash
python docker/demo.py --server-name 127.0.0.1 --server-port 7860
```

3. Regenerate the hackathon evidence bundle:

```bash
python -m eval.hackathon --episodes-per-attack 1 --seed 0 --output-dir reports/hackathon
```

## Why This Environment Is Novel

1. **Tool-grounded forensic world**: agents must use realistic investigation tools, not just emit labels.
2. **Partially observable dynamics**: the ground-truth attack is hidden; only traces and metadata are exposed.
3. **Attack diversity**: 8 malicious classes plus safe episodes (typosquatting, dependency confusion, CI/CD poisoning, and more).
4. **Anti-gaming reward shaping**: reward combines accuracy, false-alarm control, parsimony, evidence-tool coverage, and reasoning hints.

## Results Snapshot

Current source of truth: [reports/hackathon/submission_summary.json](reports/hackathon/submission_summary.json)

- Random baseline: 22.2% accuracy, -0.48 average reward, 2.0 average steps.
- Heuristic baseline: 77.8% accuracy, 0.63 average reward, 6.0 average steps.
- Trained classifier policy: 88.9% accuracy, 0.86 average reward, 6.0 average steps.
- Heuristic -> trained delta: +11.1 accuracy points and +0.22 average reward.
- Random -> trained delta: +66.7 accuracy points and +1.33 average reward.

The default **trained** line in [reports/hackathon/submission_summary.json](reports/hackathon/submission_summary.json) comes from the lightweight nearest-neighbor forensic policy artifact in [artifacts/classifier-smoke/policy.json](artifacts/classifier-smoke/policy.json). The optional transformer checkpoint evaluation path remains available through `--trained-model artifacts/sft-smoke`.

![Aegis GRPO training curves](reports/training_evidence/training_curves.png)

![Aegis heuristic versus trained comparison](reports/hackathon/heuristic_vs_trained.png)

## OpenEnv Compliance Snapshot

1. Manifest: [openenv.yaml](openenv.yaml)
2. MCP server runtime: [environment/mcp_server.py](environment/mcp_server.py)
3. Shared non-server runtime (client/server separation): [environment/runtime.py](environment/runtime.py)
4. Gym-like episode flow: reset/start, step via tools, terminal verdict.
5. Reserved tool names avoided (`reset`, `step`, `state`, and `close` are not used as tool names).

## Reward Design

[rewards/scoring.py](rewards/scoring.py) combines:

1. `accuracy`
2. `false_alarm`
3. `parsimony`
4. `evidence_coverage`
5. `reasoning`

This makes it harder for agents to exploit shallow strategies such as guessing without collecting attack-specific evidence.

## Training Pipeline (TRL + Colab)

Minimal training entrypoint:

- [training/train.py](training/train.py)

Re-runnable Colab notebook:

- [notebooks/aegis_grpo_colab.ipynb](notebooks/aegis_grpo_colab.ipynb)

Compact TRL / GRPO smoke run that writes training evidence:

```bash
python -m training.train \
	--train \
	--model-name sshleifer/tiny-gpt2 \
	--episodes-per-attack 1 \
	--max-steps 3 \
	--per-device-train-batch-size 2 \
	--gradient-accumulation-steps 1 \
	--num-generations 2 \
	--max-completion-length 48 \
	--max-tool-calling-iterations 4 \
	--logging-steps 1 \
	--save-steps 10 \
	--output-dir artifacts/grpo-evidence \
	--evidence-dir reports/training_evidence \
	--run-name aegis-grpo-evidence
```

Committed training evidence:

1. [reports/training_evidence/training_summary.json](reports/training_evidence/training_summary.json)
2. [reports/training_evidence/training_log_history.json](reports/training_evidence/training_log_history.json)
3. [reports/training_evidence/training_curves.png](reports/training_evidence/training_curves.png)

Additional transformer smoke artifacts:

1. [reports/sft_smoke/training_summary.json](reports/sft_smoke/training_summary.json)
2. [reports/sft_smoke/trained_report.json](reports/sft_smoke/trained_report.json)
3. [reports/sft_smoke_compact/training_summary.json](reports/sft_smoke_compact/training_summary.json)

## Evaluation And Improvement Evidence

Single-command hackathon evidence bundle:

```bash
python -m eval.hackathon --episodes-per-attack 1 --seed 0 --output-dir reports/hackathon
```

By default, this command trains or reuses [artifacts/classifier-smoke/policy.json](artifacts/classifier-smoke/policy.json) and evaluates it as the trained policy.

Optional transformer-checkpoint evaluation path:

```bash
python -m eval.hackathon \
	--episodes-per-attack 1 \
	--seed 0 \
	--output-dir reports/hackathon \
	--trained-model artifacts/sft-smoke \
	--trained-label sft-smoke
```

Optional comparison against the committed transformer smoke report:

```bash
python -m eval.hackathon \
	--episodes-per-attack 1 \
	--seed 0 \
	--output-dir reports/hackathon \
	--trained-report reports/sft_smoke/trained_report.json
```

Generated bundle includes:

1. [reports/hackathon/random_report.json](reports/hackathon/random_report.json)
2. [reports/hackathon/heuristic_report.json](reports/hackathon/heuristic_report.json)
3. [reports/hackathon/trained_report.json](reports/hackathon/trained_report.json)
4. [reports/hackathon/random_vs_heuristic.json](reports/hackathon/random_vs_heuristic.json)
5. [reports/hackathon/heuristic_vs_trained.json](reports/hackathon/heuristic_vs_trained.json)
6. [reports/hackathon/random_vs_trained.json](reports/hackathon/random_vs_trained.json)
7. [reports/hackathon/submission_summary.json](reports/hackathon/submission_summary.json)

Use [reports/hackathon/submission_summary.json](reports/hackathon/submission_summary.json) as the source of truth for current metrics and compliance checks.

## Install

```bash
pip install -e .[server,eval,demo]
pip install -e .[training]
```

CLI entrypoints:

1. `aegis-env-mcp`
2. `aegis-env-fit-classifier`
3. `aegis-env-train`
4. `aegis-env-eval`
5. `aegis-env-compare`
6. `aegis-env-hackathon`

## Run Demo

```bash
python docker/demo.py --server-name 127.0.0.1 --server-port 7860
```

## Docker / Hugging Face Space

Build locally:

```bash
docker build -f docker/Dockerfile -t aegis-env .
```

Run demo mode:

```bash
docker run --rm -p 7860:7860 aegis-env
```

Run MCP mode:

```bash
docker run --rm -e AEGIS_APP_MODE=mcp aegis-env
```

For Hugging Face Spaces, create a Docker Space and point it at [docker/Dockerfile](docker/Dockerfile). The container entrypoint already supports both demo mode and MCP mode.

## Submission Assets

- Environment manifest: [openenv.yaml](openenv.yaml)
- MCP server: [environment/mcp_server.py](environment/mcp_server.py)
- Colab notebook: [notebooks/aegis_grpo_colab.ipynb](notebooks/aegis_grpo_colab.ipynb)
- Training evidence bundle: [reports/training_evidence/training_summary.json](reports/training_evidence/training_summary.json), [reports/training_evidence/training_log_history.json](reports/training_evidence/training_log_history.json), [reports/training_evidence/training_curves.png](reports/training_evidence/training_curves.png)
- Hackathon evaluation bundle: [reports/hackathon/submission_summary.json](reports/hackathon/submission_summary.json), [reports/hackathon/trained_summary.json](reports/hackathon/trained_summary.json), [reports/hackathon/heuristic_vs_trained.png](reports/hackathon/heuristic_vs_trained.png)
- Slide deck: [docs/hackathon_slide_deck.md](docs/hackathon_slide_deck.md)
- Submission playbook: [docs/hackathon_submission.md](docs/hackathon_submission.md)
- Live HF Space URL: replace this line with the deployed `https://huggingface.co/spaces/<org-or-user>/aegis-env` URL before final submission.

## Final Checklist

1. Keep [README.md](README.md) as the single landing page for judges.
2. Publish the Docker app to a Hugging Face Space and replace the placeholder line above with the live URL.
3. Re-run [eval/hackathon.py](eval/hackathon.py) after updating the live Space URL so the `submission_checks` and `submission_blockers` fields capture the final state.
4. Do not commit large media binaries; link to external assets when needed.

Detailed handoff checklist: [docs/hackathon_submission.md](docs/hackathon_submission.md).
