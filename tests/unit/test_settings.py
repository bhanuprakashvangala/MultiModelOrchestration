"""Tests for mmorch.settings and the errors of mmorch.errors."""

from __future__ import annotations

import dataclasses
import os
import pickle
import re
from pathlib import Path

import pytest

from mmorch.errors import ConfigError, DataNotFoundError, MissingDependencyError, MmorchError
from mmorch.settings import DEFAULT_CHART_DIR, EndpointSettings, MatrixSettings, Paths, require_file

# ---------------------------------------------------------------- Paths and require_file


def test_paths_default_to_the_current_directory() -> None:
    paths = Paths()
    assert paths.root == Path(".")
    assert paths.prompts == Path("data") / "prompts.jsonl.gz"
    assert paths.traces == Path("results") / "traces"
    assert paths.results == Path("results")
    assert paths.live == Path("results") / "live"


def test_paths_rebase_on_the_root(tmp_path: Path) -> None:
    paths = Paths(Path("x"))
    assert paths.prompts == Path("x") / "data" / "prompts.jsonl.gz"
    assert paths.traces == Path("x") / "results" / "traces"
    assert paths.results == Path("x") / "results"
    assert paths.live == Path("x") / "results" / "live"
    assert Paths(tmp_path).traces == tmp_path / "results" / "traces"


def test_paths_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Paths().root = Path("y")


def test_require_file_returns_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl.gz"
    path.write_bytes(b"")
    assert require_file(path, "prompts file") is path


@pytest.mark.parametrize("is_dir", [False, True], ids=["missing", "directory"])
def test_require_file_raises_data_not_found(tmp_path: Path, is_dir: bool) -> None:
    path = tmp_path / "routing_keyword.csv.gz"
    if is_dir:
        path.mkdir()
    with pytest.raises(DataNotFoundError) as excinfo:
        require_file(path, "keyword trace")
    message = str(excinfo.value)
    assert message == (
        f"keyword trace not found: {path} (run from the repository root, pass --root DIR, or give the path explicitly)"
    )
    assert str(path) in message
    assert "--root" in message


# ---------------------------------------------------------------- EndpointSettings


def test_endpoint_settings_are_none_when_unset() -> None:
    settings = EndpointSettings.from_env({})
    assert settings.api_base is None
    assert settings.api_key is None
    assert settings == EndpointSettings()


def test_endpoint_settings_keep_raw_values() -> None:
    settings = EndpointSettings.from_env({"LLM_API_BASE": " https://host/v1/ ", "LLM_API_KEY": "k"})
    assert settings.api_base == " https://host/v1/ "
    assert settings.api_key == "k"


def test_endpoint_settings_keep_empty_strings() -> None:
    settings = EndpointSettings.from_env({"LLM_API_BASE": "", "LLM_API_KEY": ""})
    assert settings.api_base == ""
    assert settings.api_key == ""


def test_endpoint_settings_read_only_the_llm_variables() -> None:
    settings = EndpointSettings.from_env({"OPENAI_BASE_URL": "https://other/v1", "OPENAI_API_KEY": "other"})
    assert settings == EndpointSettings(None, None)


def test_endpoint_settings_never_show_the_key() -> None:
    settings = EndpointSettings.from_env({"LLM_API_BASE": "https://host/v1", "LLM_API_KEY": "sk-secret-123"})
    assert "sk-secret-123" not in repr(settings)
    assert "sk-secret-123" not in str(settings)
    assert "https://host/v1" in repr(settings)


# ---------------------------------------------------------------- MatrixSettings


def test_default_chart_dir_is_the_legacy_string() -> None:
    assert DEFAULT_CHART_DIR == "./deploy/helm/pick-and-spin-umbrella"
    assert isinstance(DEFAULT_CHART_DIR, str)


def test_matrix_settings_defaults() -> None:
    settings = MatrixSettings.from_env({})
    assert settings == MatrixSettings()
    assert settings.namespace == "default"
    assert settings.chart_dir == "./deploy/helm/pick-and-spin-umbrella"
    assert settings.host == "localhost"
    assert settings.port == 8080


def test_matrix_settings_read_the_variables() -> None:
    environ = {"KUBERNETES_NAMESPACE": "ns", "MATRIX_CHART_DIR": "./charts", "API_HOST": "0.0.0.0", "API_PORT": "9000"}
    settings = MatrixSettings.from_env(environ)
    assert settings == MatrixSettings(namespace="ns", chart_dir="./charts", host="0.0.0.0", port=9000)
    assert type(settings.port) is int


def test_matrix_settings_keep_empty_strings() -> None:
    settings = MatrixSettings.from_env({"KUBERNETES_NAMESPACE": "", "MATRIX_CHART_DIR": "", "API_HOST": ""})
    assert settings.namespace == ""
    assert settings.chart_dir == ""
    assert settings.host == ""


@pytest.mark.parametrize("value", ["abc", "", "80.5"])
def test_matrix_settings_reject_a_non_integer_port(value: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(f"API_PORT must be an integer, got {value!r}")):
        MatrixSettings.from_env({"API_PORT": value})


# ---------------------------------------------------------------- the environment is read only through from_env


def test_from_env_ignores_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://env/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("KUBERNETES_NAMESPACE", "env-ns")
    monkeypatch.setenv("API_PORT", "1234")
    assert EndpointSettings.from_env({}) == EndpointSettings()
    assert MatrixSettings.from_env({}) == MatrixSettings()


def test_from_env_accepts_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://env/v1")
    monkeypatch.setenv("API_PORT", "1234")
    assert EndpointSettings.from_env(os.environ) == EndpointSettings("https://env/v1", None)
    assert MatrixSettings.from_env(os.environ).port == 1234


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("error", "builtin"),
    [(ConfigError, ValueError), (DataNotFoundError, FileNotFoundError), (MissingDependencyError, ImportError)],
)
def test_errors_are_mmorch_errors_and_builtin_errors(error: type[MmorchError], builtin: type[Exception]) -> None:
    assert issubclass(error, MmorchError)
    assert issubclass(error, builtin)


def test_missing_dependency_error_names_the_extra() -> None:
    error = MissingDependencyError("pyyaml", "live")
    assert str(error) == 'pyyaml is required for this command: pip install -e ".[live]"'
    assert error.package == "pyyaml"
    assert error.extra == "live"


def test_missing_dependency_error_survives_pickling() -> None:
    error = pickle.loads(pickle.dumps(MissingDependencyError("fastapi", "matrix")))
    assert isinstance(error, MissingDependencyError)
    assert str(error) == 'fastapi is required for this command: pip install -e ".[matrix]"'
    assert (error.package, error.extra) == ("fastapi", "matrix")
