from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

REQUIRED_TRAINING_PACKAGES = (
    "trl",
    "transformers",
    "torch",
    "datasets",
    "accelerate",
)

OPTIONAL_TRAINING_PACKAGES = (
    "wandb",
    "unsloth",
)


@dataclass(slots=True)
class StackStatus:
    installed: dict[str, str]
    missing_required: list[str]
    missing_optional: list[str]

    @property
    def ready(self) -> bool:
        return not self.missing_required


def check_training_stack() -> StackStatus:
    installed: dict[str, str] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for package_name in REQUIRED_TRAINING_PACKAGES:
        try:
            installed[package_name] = version(package_name)
        except PackageNotFoundError:
            missing_required.append(package_name)

    for package_name in OPTIONAL_TRAINING_PACKAGES:
        try:
            installed[package_name] = version(package_name)
        except PackageNotFoundError:
            missing_optional.append(package_name)

    return StackStatus(
        installed=installed,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )