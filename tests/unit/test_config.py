"""Tests for autobio.core.config."""

from __future__ import annotations

from autobio.core.config import AutobioConfig


class TestAutoConfigDefaults:
    def test_default_docker_host_is_none(self) -> None:
        cfg = AutobioConfig.resolve()
        assert cfg.docker_host is None

    def test_default_image_prefix(self) -> None:
        cfg = AutobioConfig.resolve()
        assert cfg.image_prefix == "ghcr.io/briney/autobio-"

    def test_default_log_level(self) -> None:
        cfg = AutobioConfig.resolve()
        assert cfg.log_level == "INFO"


class TestEnvVarOverrides:
    def test_docker_host_from_env(self, monkeypatch: object) -> None:
        import pytest as _pytest

        mp = _pytest.MonkeyPatch()
        mp.setenv("AUTOBIO_DOCKER_HOST", "tcp://remote:2375")
        try:
            cfg = AutobioConfig.resolve()
            assert cfg.docker_host == "tcp://remote:2375"
        finally:
            mp.undo()

    def test_image_prefix_from_env(self, monkeypatch: object) -> None:
        import pytest as _pytest

        mp = _pytest.MonkeyPatch()
        mp.setenv("AUTOBIO_IMAGE_PREFIX", "my-registry/")
        try:
            cfg = AutobioConfig.resolve()
            assert cfg.image_prefix == "my-registry/"
        finally:
            mp.undo()

    def test_log_level_from_env(self, monkeypatch: object) -> None:
        import pytest as _pytest

        mp = _pytest.MonkeyPatch()
        mp.setenv("AUTOBIO_LOG_LEVEL", "DEBUG")
        try:
            cfg = AutobioConfig.resolve()
            assert cfg.log_level == "DEBUG"
        finally:
            mp.undo()


class TestRuntimeOverridePrecedence:
    def test_runtime_overrides_beat_env(self) -> None:
        import os

        old = os.environ.get("AUTOBIO_LOG_LEVEL")
        os.environ["AUTOBIO_LOG_LEVEL"] = "WARNING"
        try:
            cfg = AutobioConfig.resolve(log_level="ERROR")
            assert cfg.log_level == "ERROR"
        finally:
            if old is None:
                os.environ.pop("AUTOBIO_LOG_LEVEL", None)
            else:
                os.environ["AUTOBIO_LOG_LEVEL"] = old

    def test_runtime_override_image_prefix(self) -> None:
        cfg = AutobioConfig.resolve(image_prefix="custom/")
        assert cfg.image_prefix == "custom/"

    def test_runtime_override_docker_host(self) -> None:
        cfg = AutobioConfig.resolve(docker_host="unix:///custom.sock")
        assert cfg.docker_host == "unix:///custom.sock"
