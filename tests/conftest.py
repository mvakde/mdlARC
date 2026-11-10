import pytest


@pytest.fixture(autouse=True)
def isolated_runs(monkeypatch):
    # Never inherit the caller's run location or remote-machine settings.
    monkeypatch.delenv("MDLARC_RUNS_DIR", raising=False)
    monkeypatch.setenv("MDLARC_REMOTE", "0")
