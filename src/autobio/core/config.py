"""Global configuration with env-var and runtime override support."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AutobioConfig:
    """Configuration for the autobio runtime.

    Resolved via a three-level precedence chain:
    runtime kwargs > environment variables > defaults.
    """

    docker_host: str | None = None
    image_prefix: str = "ghcr.io/briney/autobio-"
    log_level: str = "INFO"

    @classmethod
    def resolve(cls, **runtime_overrides: str | None) -> AutobioConfig:
        """Build a config using runtime args > env vars > defaults.

        Args:
            **runtime_overrides: Any of ``docker_host``, ``image_prefix``,
                or ``log_level``.
        """
        return cls(
            docker_host=(
                runtime_overrides.get("docker_host") or os.environ.get("AUTOBIO_DOCKER_HOST")
            ),
            image_prefix=(
                runtime_overrides.get("image_prefix")
                or os.environ.get("AUTOBIO_IMAGE_PREFIX", cls.image_prefix)
            ),
            log_level=(
                runtime_overrides.get("log_level")
                or os.environ.get("AUTOBIO_LOG_LEVEL", cls.log_level)
            ),
        )
