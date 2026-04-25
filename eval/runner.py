from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from environment.models import AttackClass
from training.baseline import HeuristicBaselinePolicy
from training.env_client import AegisEnvClient
from training.rollout import rollout_episode
from training.types import EpisodeTrace, ToolObservation

Policy = Callable[[dict[str, object], list[ToolObservation]], str]


@dataclass(slots=True)
class EvaluationEpisode:
    seed: int
    trace: EpisodeTrace

    @property
    def step_count(self) -> int:
        return len(self.trace.observations)


@dataclass(slots=True)
class AttackMetrics:
    episodes: int
    accuracy: float
    average_reward: float
    average_steps: float


@dataclass(slots=True)
class EvaluationSummary:
    policy_name: str
    episodes: int
    accuracy: float
    average_reward: float
    average_steps: float
    decision_counts: dict[str, int]
    tool_usage_counts: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]]
    per_attack: dict[str, AttackMetrics]

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "episodes": self.episodes,
            "accuracy": self.accuracy,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "decision_counts": self.decision_counts,
            "tool_usage_counts": self.tool_usage_counts,
            "confusion_matrix": self.confusion_matrix,
            "per_attack": {name: asdict(metrics) for name, metrics in self.per_attack.items()},
        }


def evaluate_policy(
    policy: Policy,
    *,
    episodes_per_attack: int = 2,
    seed: int = 0,
    attack_schedule: Sequence[AttackClass | str] | None = None,
) -> tuple[EvaluationSummary, list[EvaluationEpisode]]:
    schedule = _normalize_schedule(attack_schedule, episodes_per_attack=episodes_per_attack)
    evaluation_episodes: list[EvaluationEpisode] = []

    for offset, attack_class in enumerate(schedule):
        episode_seed = seed + offset
        client = AegisEnvClient(seed=episode_seed)
        trace = rollout_episode(
            client,
            policy,
            attack_class=attack_class.value,
            seed=episode_seed,
        )
        evaluation_episodes.append(EvaluationEpisode(seed=episode_seed, trace=trace))

    summary = _summarize(policy_name=type(policy).__name__, episodes=evaluation_episodes)
    return summary, evaluation_episodes


def evaluate_baseline(*, episodes_per_attack: int = 2, seed: int = 0) -> tuple[EvaluationSummary, list[EvaluationEpisode]]:
    return evaluate_policy(
        HeuristicBaselinePolicy(),
        episodes_per_attack=episodes_per_attack,
        seed=seed,
    )


def _normalize_schedule(
    attack_schedule: Sequence[AttackClass | str] | None,
    *,
    episodes_per_attack: int,
) -> list[AttackClass]:
    if attack_schedule is not None:
        return [attack if isinstance(attack, AttackClass) else AttackClass(attack) for attack in attack_schedule]

    schedule: list[AttackClass] = []
    for attack_class in AttackClass:
        schedule.extend([attack_class] * episodes_per_attack)
    return schedule


def _summarize(*, policy_name: str, episodes: Sequence[EvaluationEpisode]) -> EvaluationSummary:
    total_episodes = len(episodes)
    correct = 0
    reward_total = 0.0
    step_total = 0
    decision_counts: Counter[str] = Counter()
    tool_usage_counts: Counter[str] = Counter()
    confusion_counts: dict[str, Counter[str]] = defaultdict(Counter)
    traces_by_attack: dict[str, list[EpisodeTrace]] = defaultdict(list)

    for episode in episodes:
        trace = episode.trace
        actual_name = trace.actual_attack.value
        decision_name = trace.decision.value
        traces_by_attack[actual_name].append(trace)
        correct += int(trace.actual_attack is trace.decision)
        reward_total += trace.reward.total
        step_total += len(trace.observations)
        decision_counts.update([decision_name])
        tool_usage_counts.update(trace.tool_names)
        confusion_counts[actual_name].update([decision_name])

    confusion_matrix = {
        actual_name: {decision: counts.get(decision, 0) for decision in _attack_names()}
        for actual_name, counts in sorted(confusion_counts.items())
    }
    per_attack = {
        attack_name: _attack_metrics(traces)
        for attack_name, traces in sorted(traces_by_attack.items())
    }

    return EvaluationSummary(
        policy_name=policy_name,
        episodes=total_episodes,
        accuracy=correct / max(1, total_episodes),
        average_reward=reward_total / max(1, total_episodes),
        average_steps=step_total / max(1, total_episodes),
        decision_counts=dict(sorted(decision_counts.items())),
        tool_usage_counts=dict(sorted(tool_usage_counts.items())),
        confusion_matrix=confusion_matrix,
        per_attack=per_attack,
    )


def _attack_metrics(traces: Sequence[EpisodeTrace]) -> AttackMetrics:
    episodes = len(traces)
    correct = sum(1 for trace in traces if trace.actual_attack is trace.decision)
    reward_total = sum(trace.reward.total for trace in traces)
    step_total = sum(len(trace.observations) for trace in traces)
    return AttackMetrics(
        episodes=episodes,
        accuracy=correct / max(1, episodes),
        average_reward=reward_total / max(1, episodes),
        average_steps=step_total / max(1, episodes),
    )


def _attack_names() -> tuple[str, ...]:
    return tuple(attack_class.value for attack_class in AttackClass)