from __future__ import annotations

from environment.registry import ShadowRegistry


def get_reputation_score(pkg_name: str, registry: ShadowRegistry) -> dict[str, object]:
    pkg = registry.get_package(pkg_name)
    weekly_trend = pkg.weekly_download_trend
    average_downloads = sum(weekly_trend) / len(weekly_trend) if weekly_trend else 0.0
    issue_total = pkg.open_issues + pkg.closed_issues
    star_issue_ratio = pkg.stars / max(1, issue_total)
    star_download_ratio = pkg.stars / max(1.0, average_downloads)

    return {
        "package": pkg.name,
        "stars": pkg.stars,
        "open_issues": pkg.open_issues,
        "closed_issues": pkg.closed_issues,
        "weekly_download_trend": weekly_trend,
        "average_weekly_downloads": round(average_downloads, 2),
        "star_issue_ratio": round(star_issue_ratio, 2),
        "star_download_ratio": round(star_download_ratio, 2),
        "maintainer_count": len(pkg.maintainers),
        "suspicious_popularity": pkg.stars >= 5000 and issue_total <= 3,
    }