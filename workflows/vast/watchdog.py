#!/usr/bin/env python3
"""Run on an always-on machine to stop only saved Vast rentals at their deadline."""

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.vast.common import atomic_json, keep_b300_running, lock, now
from workflows.vast.vast_api import Vast, api_key


def enforce(client, lease, timestamp):
    if timestamp < lease["deadline_unix"]:
        return "within_allowance"
    # Two authenticated listings confirm absence. An API error remains unknown.
    for _ in range(2):
        matches = [item for item in client.instances() if (item.get("id") == lease["instance_id"]
                   if lease["instance_id"] is not None else item.get("label") == lease["label"])]
        if len(matches) > 1:
            raise ValueError("Ambiguous rental label; watchdog will not guess")
        if matches:
            instance = matches[0]
            if instance.get("label") != lease["label"]:
                raise ValueError("Watchdog refuses an instance whose identity changed")
            if keep_b300_running(lease, instance):
                return "keep_b300_running"
            if instance.get("actual_status") == "stopped":
                return "stopped"
            client.stop(instance["id"])
            return "stop_requested"
    return "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leases", type=Path, required=True, help="Directory containing exported lease.json files")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("Poll interval must be at least five seconds")
    args.leases.mkdir(parents=True, exist_ok=True)
    client = Vast(api_key(args.api_key_file))
    with lock(args.leases / ".watchdog.lock"):
        while True:
            for path in sorted(args.leases.rglob("lease.json")):
                if path.is_symlink():
                    continue
                try:
                    lease = json.loads(path.read_text())
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
