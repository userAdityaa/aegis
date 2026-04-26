# Aegis-Env Hackathon Slide Deck

## Slide 1: Problem

- Package supply-chain incidents require more than a label.
- A useful agent has to inspect metadata, compare releases, reason over dependencies, and commit to a final verdict.
- Most current coding agents fail because they skip tools, overfit to shallow cues, or lose consistency across the investigation.

## Slide 2: Environment

- Aegis-Env is an OpenEnv MCP environment for package supply-chain forensics.
- Each episode hides one attack in a synthetic package ecosystem.
- The agent starts with a target package, gathers evidence through tools, and ends the episode with `final_verdict`.
- Bonus realism: a stakeholder inbox + long-horizon case file (Themes 3.2 + 2).
- Bonus multi-agent interaction: `consult_peer_analyst` simulates asking a teammate for a second opinion (Theme 1-style).

## Slide 3: Tool Surface

- `check_maintainer_history`
- `diff_versions`
- `inspect_install_script`
- `get_reputation_score`
- `trace_dependencies`
- `run_sandbox_test`

These tools force grounded investigation instead of direct label prediction.

## Slide 4: Reward And Training

- Reward combines accuracy, false-alarm control, parsimony, evidence coverage, and reasoning quality.
- TRL / GRPO training entrypoint: [training/train.py](../training/train.py)
- Re-runnable Colab notebook: [notebooks/aegis_grpo_colab.ipynb](../notebooks/aegis_grpo_colab.ipynb)
- Logged training evidence: [reports/training_evidence/training_curves.png](../reports/training_evidence/training_curves.png)
- Fast preset (100-step evidence run): `python -m training.train --train ... --fast-evidence-100`

## Slide 5: Results

**Important:** The headline **“trained”** row is **not** a neural LLM. It is a **nearest-neighbor classifier** on handcrafted forensic features (`artifacts/classifier-smoke/policy.json`). Random and heuristic rows are explicit baselines. For **TRL / GRPO transformer** evaluation, run `eval.hackathon` with `--trained-model` pointing at a checkpoint (see README).

If the transformer GRPO run is compute-limited, we provide a short **100-step evidence run** that still logs tool-call frequency, verdict completion, and reward curves.

- Random baseline: 22.2% accuracy, -0.32 average reward.
- Heuristic baseline: 77.8% accuracy, 0.70 average reward.
- **Trained (kNN classifier, not LLM):** 88.9% accuracy, 0.92 average reward.
- Heuristic -> trained delta: +11.1 accuracy points, +0.22 average reward.

Source of truth: [reports/hackathon/submission_summary.json](../reports/hackathon/submission_summary.json)

## Slide 6: Submission Assets

- README landing page: [README.md](../README.md)
- Submission playbook: [docs/hackathon_submission.md](hackathon_submission.md)
- MCP server: [environment/mcp_server.py](../environment/mcp_server.py)
- Docker Space packaging: [docker/Dockerfile](../docker/Dockerfile)

Final manual step before submission: publish the Docker app to a Hugging Face Space and add the live URL back into [README.md](../README.md).