# Aegis-Env Hackathon Slide Deck

## Slide 1: Problem

- Package supply-chain incidents require more than a label.
- A useful agent has to inspect metadata, compare releases, reason over dependencies, and commit to a final verdict.
- Most current coding agents fail because they skip tools, overfit to shallow cues, or lose consistency across the investigation.

## Slide 2: Environment

- Aegis-Env is an OpenEnv MCP environment for package supply-chain forensics.
- Each episode hides one attack in a synthetic package ecosystem.
- The agent starts with a target package, gathers evidence through tools, and ends the episode with `final_verdict`.

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

## Slide 5: Results

- Random baseline: 22.2% accuracy, -0.48 average reward.
- Heuristic baseline: 77.8% accuracy, 0.63 average reward.
- Trained classifier policy: 88.9% accuracy, 0.86 average reward.
- Heuristic -> trained delta: +11.1 accuracy points, +0.22 average reward.

Source of truth: [reports/hackathon/submission_summary.json](../reports/hackathon/submission_summary.json)

## Slide 6: Submission Assets

- README landing page: [README.md](../README.md)
- Submission playbook: [docs/hackathon_submission.md](hackathon_submission.md)
- MCP server: [environment/mcp_server.py](../environment/mcp_server.py)
- Docker Space packaging: [docker/Dockerfile](../docker/Dockerfile)

Final manual step before submission: publish the Docker app to a Hugging Face Space and add the live URL back into [README.md](../README.md).