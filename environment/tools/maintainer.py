from __future__ import annotations

from collections import Counter

from environment.registry import ShadowRegistry
from environment.tools._helpers import commit_to_dict


def check_maintainer_history(pkg_name: str, registry: ShadowRegistry) -> dict[str, object]:
    pkg = registry.get_package(pkg_name)
    commits = sorted(
        (commit for version in pkg.version_history for commit in version.commits),
        key=lambda record: record.timestamp,
    )
    recent_commits = commits[-5:]
    prior_ips = {commit.ip for commit in commits[:-5]}
    unseen_recent_ips = sorted({commit.ip for commit in recent_commits if commit.ip not in prior_ips})
    off_hours_recent_commits = [
        commit_to_dict(commit)
        for commit in recent_commits
        if commit.timestamp.hour < 6 or commit.timestamp.hour > 21
    ]

    maintainer_summaries = []
    for maintainer in pkg.maintainers:
        author_commits = [commit for commit in commits if commit.author == maintainer.email]
        hour_distribution = Counter(commit.timestamp.hour for commit in author_commits)
        maintainer_summaries.append(
            {
                "name": maintainer.name,
                "email": maintainer.email,
                "known_ips": sorted(set(maintainer.ip_history)),
                "commit_count": len(author_commits),
                "first_seen": author_commits[0].timestamp.isoformat() if author_commits else None,
                "last_seen": author_commits[-1].timestamp.isoformat() if author_commits else None,
                "commit_hour_distribution": dict(sorted(hour_distribution.items())),
            }
        )

    return {
        "package": pkg.name,
        "maintainers": maintainer_summaries,
        "commit_count": len(commits),
        "ip_history": [commit_to_dict(commit) for commit in commits],
        "recent_commits": [commit_to_dict(commit) for commit in recent_commits],
        "unseen_recent_ips": unseen_recent_ips,
        "off_hours_recent_commits": off_hours_recent_commits,
        "suspicious_ip_shift": bool(unseen_recent_ips),
    }