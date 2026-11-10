import pytest


@pytest.fixture(autouse=True)
def isolated_run_branch(monkeypatch):
    # Temporary test repositories have no Git HEAD. Never use the user's branch.
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "test-branch")
    monkeypatch.setenv("MDLARC_REMOTE", "0")
