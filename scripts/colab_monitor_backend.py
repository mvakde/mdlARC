"""Read-only Colab account worker. Run only in the installed Colab CLI Python.

    One account per process; credentials and runtime proxy tokens never leave it.
    No CLI session pruning, kernel execution, login prompts or credential writes.
"""

import base64
from collections import deque
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
import select
import sys
import time
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.colab_accounts import ACCOUNTS, profile_dir
from scripts.colab_monitor_data import compact_score, parse_console

MAX_FILE = 4 * 1024 * 1024
TAIL = 128 * 1024
TIMEOUT = 12


class MonitorError(Exception):
    pass


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise MonitorError("Invalid run name.")
    return value


def read_json(path):
    try:
        if path.stat().st_size > MAX_FILE:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class Account:
    def __init__(self, account):
        # Imported lazily in the short-lived worker, never in the UI server.
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        from google.auth.transport.requests import AuthorizedSession, Request
        from google.oauth2.credentials import Credentials
        from colab_cli.client import Client, Prod

        self.account = account
        directory = profile_dir(account)
        try:
            credentials = Credentials.from_authorized_user_file(str(directory / "token.json"))
            if not credentials.valid:
                credentials.refresh(lambda *args, **kwargs: Request()(*args, **{**kwargs, "timeout": TIMEOUT}))
            self.authenticated = AuthorizedSession(credentials, refresh_timeout=TIMEOUT)
            original_request = self.authenticated.request

            def bounded(*args, **kwargs):
                kwargs.setdefault("timeout", TIMEOUT)
                return original_request(*args, **kwargs)

            self.authenticated.request = bounded
            response = self.authenticated.get("https://openidconnect.googleapis.com/v1/userinfo")
            response.raise_for_status()
            if response.json().get("email", "").lower() != ACCOUNTS[account]:
                raise MonitorError(f"Saved login is not {ACCOUNTS[account]}.")
        except MonitorError:
            raise
        except Exception:
            raise MonitorError(f"Account {account}: cached Colab login unavailable; renew it in your terminal.") from None
        self.client = Client(Prod(), self.authenticated)
        self.http = requests.Session()
        self.http.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=2,
                        max_retries=Retry(total=1, connect=1, read=1, status=1,
                                          status_forcelist=(502, 503, 504),
                                          allowed_methods={"GET"}, backoff_factor=0.15)))
        self.assignments = {}
        self.listed_at = 0
        self.aliases = read_json(directory / "sessions.json")
        self.resolved = {}

    def refresh(self, force=False):
        if force or time.monotonic() - self.listed_at > 15:
            self.assignments = {a.endpoint: a for a in self.client.list_assignments()}
            self.listed_at = time.monotonic()
        return self.assignments

    def listing(self):
        names = {v.get("endpoint"): name for name, v in self.aliases.items() if isinstance(v, dict)}
        return [{"endpoint": a.endpoint, "session": names.get(a.endpoint, a.endpoint),
                 "gpu": str(getattr(a.accelerator, "name", a.accelerator)),
                 "shape": str(getattr(a.machine_shape, "name", a.machine_shape))}
                for a in self.refresh(force=True).values()]

    def request(self, endpoint, path, *, optional=False, tail=False):
        assignment = self.refresh().get(endpoint)
        if assignment is None:
            raise MonitorError("VM is absent from the authenticated assignment list. Refresh the VM list.")
        proxy = assignment.runtime_proxy_info
        if urlsplit(proxy.url).scheme != "https":
            raise MonitorError("Colab returned an invalid runtime address.")
        headers = {"X-Colab-Runtime-Proxy-Token": proxy.token, "X-Colab-Client-Agent": "colab-cli"}
        if tail:
            headers["Range"] = f"bytes=-{TAIL}"
        with self.http.get(proxy.url.rstrip("/") + path, headers=headers,
                           params={"colab-runtime-proxy-token": proxy.token},
                           timeout=TIMEOUT, stream=True) as response:
            if optional and response.status_code == 404:
                return None
            if response.status_code in (401, 403):
                self.listed_at = 0
                raise MonitorError("Runtime authorization expired; retry to refresh the assignment.")
            response.raise_for_status()
            # Some Colab file handlers ignore Range. Stream a bounded file and
            # retain only its tail rather than keeping full logs in RAM.
            size = 0
            chunks = deque()
            kept = 0
            for chunk in response.iter_content(32768):
                size += len(chunk)
                if size > MAX_FILE:
                    raise MonitorError("Remote file exceeds the 4 MiB read limit.")
                chunks.append(chunk)
                kept += len(chunk)
                if tail:
                    while len(chunks) > 1 and kept - len(chunks[0]) >= TAIL:
                        kept -= len(chunks.popleft())
            content = b"".join(chunks)
        if tail:
            return content[-TAIL:].decode("utf-8", errors="replace")
        text = content.decode("utf-8")
        if text.startswith(")]}'"):
            text = text.split("\n", 1)[1]
        return json.loads(text)

    def contents(self, endpoint, path, optional=False):
        return self.request(endpoint, "/api/contents/" + quote(path.strip("/"), safe="/") + "?content=1", optional=optional)

    def file_json(self, endpoint, path):
        data = self.contents(endpoint, path, optional=True)
        if data is None:
            return None
        content = data.get("content", "")
        if data.get("format") == "base64":
            content = base64.b64decode(content).decode()
        return json.loads(content)

    def resolve_run(self, endpoint, hint):
        key = (endpoint, hint.get("run_name"))
        if key in self.resolved:
            return self.resolved[key]
        if hint.get("endpoint") == endpoint and hint.get("run_name") and hint.get("spec"):
            # The launcher saved these exact uploaded spec bytes with the
            # original VM identity. No remote directory/spec round trips are
            # needed; the next live console GET still checks this assignment.
            result = {"run_name": safe_name(hint["run_name"]), "branch": hint.get("branch"),
                      "spec": hint["spec"], "run_count": None}
            self.resolved[key] = result
            return result
        directory = self.contents(endpoint, "/content/mdlARC/runs", optional=True)
        if directory is None:
            raise MonitorError("This VM has no mdlARC run folder yet.")
        runs = [r for r in directory.get("content", []) if r.get("type") == "directory"]
        if not runs:
            raise MonitorError("This VM has no mdlARC run folder yet.")
        preferred = next((r for r in runs if r["name"] == hint.get("run_name")), None)
        # Normal launcher VMs contain one run. For a manually reused VM, show
        # the most recently modified run and expose its actual name in the UI.
        chosen = preferred or max(runs, key=lambda r: r.get("last_modified", ""))
        name = safe_name(chosen["name"])
        metadata = self.file_json(endpoint, f"/content/mdlARC/runs/{name}/experiment.json") or {}
        result = {"run_name": name, "branch": metadata.get("branch", hint.get("branch")),
                  "spec": metadata.get("spec") or hint.get("spec") or {}, "run_count": len(runs)}
        self.resolved[key] = result
        return result

    def progress(self, endpoint, run):
        path = f"/files/content/mdlARC/runs/{run['run_name']}/console.log"
        console = self.request(endpoint, path, optional=True, tail=True)
        return parse_console(console or "")

    def check(self, request):
        endpoint = request["endpoint"]
        if not isinstance(endpoint, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", endpoint):
            raise MonitorError("Invalid VM identity.")
        result = {"observed_at": datetime.now(UTC).isoformat()}
        action = request["action"]
        if action == "performance":
            resources = self.request(endpoint, "/api/colab/resources")
            result["gpus"] = [{k: g.get(k) for k in ("name", "memoryUsedBytes", "memoryTotalBytes", "gpuUtilization", "memoryUtilization")}
                              for g in resources.get("gpus", [])]
        try:
            run = self.resolve_run(endpoint, request.get("hint", {}))
            result.update({k: run[k] for k in ("run_name", "branch", "run_count")})
            started = time.monotonic()
            result["progress"] = self.progress(endpoint, run)
            result["sample_time"] = (started + time.monotonic()) / 2
            result["observed_at"] = datetime.now(UTC).isoformat()
        except MonitorError as exc:
            if action == "performance":
                result["note"] = str(exc)
                return result
            raise
        if action == "metrics":
            result["scores"] = []
            result["evaluation_configured"] = bool(run["spec"].get("evals")) and not run["spec"].get("skip_evals", False)
            progress = result["progress"]
            # During ordinary pretraining, no scored artifacts or completion
            # manifest exist yet. Avoid slow negative remote file lookups.
            # Later finetuning stages may already have earlier eval results.
            active_training = (progress.get("kind") == "training" and progress.get("epoch") is not None
                               and (progress.get("step") is not None
                                    or progress["epoch"] < (progress.get("epoch_total") or progress["epoch"])))
            pretraining = progress.get("stage") == "pretrain" or not run["spec"].get("enable_finetune", False)
            probe_results = progress.get("error_observed") or not (active_training and pretraining)
            run_path = f"/content/mdlARC/runs/{run['run_name']}"
            evaluations = list(run["spec"].get("evals", {}).items())[:12] if probe_results else []
            for name, settings in evaluations:
                safe_name(name)
                mode = settings.get("mode", run["spec"].get("eval_defaults", {}).get("mode"))
                filename = "accuracy.json" if mode == "accuracy-only" else "score.json"
                value = self.file_json(endpoint, f"{run_path}/evals/{name}/{filename}")
                if value:
                    if filename == "score.json":
                        result["scores"].append({"eval": name, **compact_score(value)})
                    else:
                        # Accuracy reports have split metrics; keep them distinct
                        # from solved-task scores and avoid shipping predictions.
                        result.setdefault("evaluation_accuracy", []).append({
                            "eval": name, "splits": {
                                split: {k: v for k, v in data.get("summary", {}).items()
                                        if k.endswith("_accuracy") or k in {"total_sequences", "total_cases"}}
                                for split, data in value.get("splits", {}).items()
                            }})
            manifest = self.file_json(endpoint, "/content/mdlARC-results-manifest.json") if probe_results else None
            if manifest and manifest.get("run_name") == run["run_name"]:
                status = self.file_json(endpoint, "/content/mdlARC-run-status.json") or {}
                result["completion"] = {"finalized": manifest.get("finalized"),
                                        **{k: status[k] for k in ("ok", "backups_complete") if k in status}}
        return result


def main(account, idle_seconds=20):
    logging.disable(logging.CRITICAL)
    backend = None
    while select.select([sys.stdin], [], [], idle_seconds)[0]:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            request = json.loads(line)
            if request.get("action") not in {"list", "performance", "progress", "metrics"}:
                raise MonitorError("Unsupported monitor operation.")
            if backend is None:
                backend = Account(account)
            data = backend.listing() if request["action"] == "list" else backend.check(request)
            response = {"ok": True, "data": data}
        except MonitorError as exc:
            response = {"ok": False, "error": str(exc)}
        except Exception as exc:
            # HTTP exceptions may contain proxy credentials in URLs. Only the
            # exception class crosses this process boundary.
            response = {"ok": False, "error": f"{type(exc).__name__}: Colab read failed; retry or check the saved login."}
        print(json.dumps(response, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ACCOUNTS:
        raise SystemExit("Select account 1 or 2.")
    main(sys.argv[1])
