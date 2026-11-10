#!/usr/bin/env python3
"""Choose Vast, Runpod, SF Compute, or cached-auth Colab."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from workflows.vast.policy import GPU_TEMPLATES
from scripts.colab_accounts import ACCOUNTS
BACKENDS = {
    "vast": "workflows/vast/run.py",
    "runpod": "workflows/runpod/run.py",
    "sfcompute": "workflows/sfcompute/run.py",
    "colab": "scripts/colab_oauth/launch.py",
    "colab-oauth": "scripts/colab_oauth/launch.py",  # Existing cached-auth alias.
    "colab-manual": "scripts/run_on_colab.py",
}


def dialog(script, *args):
    result = subprocess.run(["osascript", "-e", script, *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("Run cancelled or native chooser unavailable")
    return result.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Arguments after -- are passed unchanged to the selected launcher.")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--account", choices=ACCOUNTS, help="Colab compute account: 1=your-colab-account@example.com, 2=your-drive-account@example.com")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    backend = args.backend
    account = args.account
    forwarded = args.arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if backend is None:
        if sys.platform != "darwin":
            parser.error("Use --backend on non-macOS systems")
        chosen = dialog('choose from list {"Vast.ai", "Runpod", "SF Compute", "Colab", "Colab (manual auth, legacy)"} with prompt "Choose experiment workflow" default items {"Vast.ai"}')
        backend = {"Vast.ai": "vast", "Runpod": "runpod", "SF Compute": "sfcompute", "Colab": "colab", "Colab (manual auth, legacy)": "colab-manual"}.get(chosen)
        if backend is None:
            return 0
        has_account = any(value == "--account" or value.startswith("--account=") for value in forwarded)
        watching = any(value == "--watch-session" or value.startswith("--watch-session=") for value in forwarded)
        if backend.startswith("colab") and account is None and not has_account and not watching and "--dry-run" not in forwarded:
            labels = {f"Account {number}: {email}": number for number, email in ACCOUNTS.items()}
            options = ", ".join(f'"{label}"' for label in labels)
            selected = dialog(f'choose from list {{{options}}} with prompt "Choose Colab compute account" default items {{"Account 1: {ACCOUNTS["1"]}"}}')
            if selected not in labels:
                return 0
            account = labels[selected]
        if not forwarded:
            spec = dialog('POSIX path of (choose file with prompt "Choose an mdlARC experiment JSON spec")')
            forwarded = ["--config", spec]
            if backend in {"vast", "runpod", "sfcompute"}:
                provider = {"vast": "Vast", "runpod": "Runpod", "sfcompute": "SF Compute"}[backend]
                user_policy = Path.home() / f".config/mdlarc-{backend}/policy.json"
                if not user_policy.is_file():
                    policy = Path(dialog(f'POSIX path of (choose file with prompt "Choose your {provider} infrastructure policy JSON")'))
                    forwarded += ["--policy", str(policy)]
                labels = ({"H100 80 GB (1 GPU)": "h100-1", "H100 80 GB (8 GPUs)": "h100-8"}
                          if backend == "sfcompute" else {label: name for name, label in GPU_TEMPLATES.items()})
                options = ", ".join(f'"{label}"' for label in [*labels, "Saved policy"])
                default = "H100 80 GB (1 GPU)" if backend == "sfcompute" else "A100 80 GB"
                selected = dialog(f'choose from list {{{options}}} with prompt "Choose {provider} GPU template" default items {{"{default}"}}')
                if selected in labels:
                    forwarded += ["--gpu-template", labels[selected]]
                elif selected != "Saved policy":
                    return 0
            else:
                prompt = ("Enter your manually authenticated Colab session name" if backend == "colab-manual"
                          else "Enter your fresh, unused Colab session name")
                if account:
                    prompt += f" for account {account} ({ACCOUNTS[account]})"
                session = dialog(f'text returned of (display dialog "{prompt}" default answer "")')
                forwarded += ["--session", session]
    path = ROOT / BACKENDS[backend]
    if not path.is_file():
        parser.error(f"Selected workflow entry point is unavailable: {path}")
    if backend in {"vast", "runpod", "sfcompute"}:
        if account is not None:
            parser.error("--account selects a Colab compute account; choose a Colab backend")
        forwarded = ["launch", *forwarded]
    elif account is not None:
        if any(value == "--account" or value.startswith("--account=") for value in forwarded):
            parser.error("Provide --account once, before or after --")
        forwarded += ["--account", account]
    return subprocess.call([sys.executable, str(path), *forwarded], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
