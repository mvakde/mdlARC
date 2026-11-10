#!/usr/bin/env python3
"""Authorize the experiment account once, or renew its seven-day Testing grant."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.colab_oauth.credentials import (
    ACCOUNT, AuthError, DEFAULT_TOKEN, authorize_client, outside_repository, save_token,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-secrets", required=True, type=Path)
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--reauthorize", action="store_true")
    args = parser.parse_args(argv)
    path = outside_repository(args.token)
    if path.exists() and not args.reauthorize:
        raise AuthError("Token already exists. Use --reauthorize for the next Testing grant.")
    print(f"Choose {ACCOUNT} in the browser. Leave your Google OAuth app in Testing.", flush=True)
    data = authorize_client(args.client_secrets)
    save_token(path, data, replace=args.reauthorize)
    print(f"Drive account verified. Private token saved outside Git: {path}")
    print("Testing grants normally need browser reauthorization after seven days.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuthError) else "Setup failed; no credential details were logged.", file=sys.stderr)
        raise SystemExit(1) from None
