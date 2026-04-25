# Hackathon Submission Playbook

Use this checklist before the final OpenEnv Hackathon submission.

## 1. Repo Artifacts That Must Exist

1. [openenv.yaml](../openenv.yaml)
2. [notebooks/aegis_grpo_colab.ipynb](../notebooks/aegis_grpo_colab.ipynb)
3. [docker/Dockerfile](../docker/Dockerfile)
4. [artifacts/classifier-smoke/policy.json](../artifacts/classifier-smoke/policy.json)
5. [reports/training_evidence/training_summary.json](../reports/training_evidence/training_summary.json)
6. [reports/training_evidence/training_log_history.json](../reports/training_evidence/training_log_history.json)
7. [reports/training_evidence/training_curves.png](../reports/training_evidence/training_curves.png)
8. [reports/hackathon/submission_summary.json](../reports/hackathon/submission_summary.json)
9. [docs/hackathon_slide_deck.md](hackathon_slide_deck.md)

## 2. Hugging Face Space + submission_ready

1. Create a Docker Space and point it at [docker/Dockerfile](../docker/Dockerfile).
2. Verify the demo boots successfully.
3. Ensure the live `https://huggingface.co/spaces/...` URL appears in [README.md](../README.md) (Submission Assets or body text — the automated checker scans the whole file).
4. Re-run the hackathon command below so `reports/hackathon/submission_summary.json` picks up the URL and `submission_checks.submission_ready` becomes **true**.

## 3. Regenerate Evidence

```bash
python -m eval.hackathon --episodes-per-attack 1 --seed 0 --output-dir reports/hackathon
```

Optional transformer-checkpoint evaluation:

```bash
python -m eval.hackathon --episodes-per-attack 1 --seed 0 --output-dir reports/hackathon --trained-model artifacts/grpo-evidence --trained-label grpo-evidence
```

Optional comparison against the committed transformer smoke report:

```bash
python -m eval.hackathon --episodes-per-attack 1 --seed 0 --output-dir reports/hackathon --trained-report reports/sft_smoke/trained_report.json
```

## 4. Judge-Facing Narrative

Use this flow for a short pitch, post, or deck:

1. Problem: why package supply-chain forensics needs tool-grounded LLM training.
2. Environment: what the agent observes, which tools it can call, and how episodes end.
3. Reward: why the scoring is difficult to game.
4. Training: show the TRL notebook, the logged loss / reward curves, and the evidence files.
5. Results: show the improvement from random -> heuristic -> trained policy.
6. Importance: why this matters for real agent reliability.

## 5. Final Local Smoke

```bash
python -m training.train --check-stack
python -m training.train --smoke --seed 0
python -m eval.cli --episodes-per-attack 1 --seed 0 --policy heuristic --report reports/hackathon/heuristic_report.json
python -m eval.cli --episodes-per-attack 1 --seed 0 --policy classifier --checkpoint artifacts/classifier-smoke/policy.json --report reports/hackathon/trained_report.json
```

## 6. Submission Summary Audit

The generated [reports/hackathon/submission_summary.json](../reports/hackathon/submission_summary.json) now contains:

1. `submission_checks`: file and README coverage checks.
2. `submission_blockers`: human-readable blockers that still prevent a fully complete submission.

Before submitting, make sure `submission_blockers` is an empty list.
