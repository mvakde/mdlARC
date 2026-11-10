#!/usr/bin/env python3
"""Stop saved Runpod pods at their deadline from an independent always-on machine."""

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.runpod.api import Runpod, api_key
from workflows.vast.common import atomic_json, lock, now
from workflows.vast.watchdog import enforce


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leases", type=Path, required=True, help="Runpod-only exported lease directory")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("Poll interval must be at least five seconds")
    args.leases.mkdir(parents=True, exist_ok=True)
    client = Runpod(api_key(args.api_key_file))
    with lock(args.leases / ".watchdog.lock"):
        while True:
            for path in sorted(args.leases.rglob("lease.json")):
                if path.is_symlink():
                    continue
                try:
                    lease = json.loads(path.read_text())
                    if not lease["label"].startswith("mdlarc:runpod:"):
                        raise ValueError("Not a Runpod lease")
                    result = enforce(client, lease, time.time())
                except Exception as exc:
                    result = "unknown_" + type(exc).__name__
                atomic_json(path.with_name("watchdog-status.json"), {"at": now(), "state": result})
                if result != "within_allowance":
                    print(f"{path.parent.name}: {result}", flush=True)
            if args.once:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
