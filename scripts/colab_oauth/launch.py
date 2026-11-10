#!/usr/bin/env python3
"""Launch Colab with cached Drive authentication by default; manual auth is a fallback."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.colab_oauth.authorize import authorize_session
from scripts.colab_oauth.credentials import AuthError, DEFAULT_TOKEN, ROOT
from scripts import colab_accounts


def delegate(arguments):
    return subprocess.call([sys.executable, str(ROOT / "scripts/run_on_colab.py"), *arguments], cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="All other arguments pass through to scripts/run_on_colab.py.")
    parser.add_argument("--auth-mode", choices=("cached", "manual"), default="cached")
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--account", choices=colab_accounts.ACCOUNTS,
                        help="Colab compute account: 1=your-colab-account@example.com, 2=your-drive-account@example.com")
    args, remaining = parser.parse_known_args(argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--session")
    probe.add_argument("--watch-session")
    probe.add_argument("--resume-unsubmitted")
    probe.add_argument("--reuse-session")
    probe.add_argument("--dry-run", action="store_true")
    probe.add_argument("--local-artifacts", action="store_true")
    probe.add_argument("--branch")
    launch, _ = probe.parse_known_args(remaining)
    if args.account is not None:
        remaining += ["--account", args.account]
    if launch.dry_run or launch.watch_session or launch.resume_unsubmitted or launch.reuse_session:
        # Recovery never reinstalls credentials. Explicit unsubmitted recovery
        # performs the first submission only after the launcher's safety checks.
        return delegate(remaining)
    os.environ[colab_accounts.ACCOUNT_ENV] = colab_accounts.selected_account(args.account)
    if args.auth_mode == "cached":
        if launch.local_artifacts or not launch.session:
            parser.error("Cached OAuth requires --session for a Drive-backed VM you already created.")
        # Validate the actual spec/flags using the existing launcher before touching any VM.
        if delegate([*remaining, "--dry-run"]):
            return 1
        from scripts.run_layout import branch_folder, results_root
        base = (ROOT / "results" / branch_folder(launch.branch)) if launch.branch else results_root(ROOT)
        if any(path.exists() for path in (
                base / "colab_runs" / launch.session,
                base / "colab_runs" / "_active" / f"{launch.session}.json",
                base / "experiments" / "notes" / f"{launch.session}.md")):
            raise AuthError("Session already has run records. Use the existing --watch-session command or a fresh session.")
        authorize_session(launch.session, args.token)
    return delegate(remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuthError) else "Experiment launch failed; no credential details were logged.", file=sys.stderr)
        raise SystemExit(1) from None
