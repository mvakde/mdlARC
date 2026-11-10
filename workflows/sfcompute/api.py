"""Use autor's refreshed local login; never send the account credential to a node."""

import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from workflows.vast.common import safe_name
from workflows.vast.vast_api import AmbiguousSubmission, ApiError


class Autor:
    def __init__(self, executable=None, *, workspace="default", api_url="https://givemeanode.com", identities=None):
        self.executable = str(executable or shutil.which("autor") or Path.home() / ".local/bin/autor")
        self.workspace = safe_name(workspace)
        if api_url not in {"https://givemeanode.com", "https://autoresearch.sfcompute.com"}:
            raise ValueError("Use an official SF Compute API URL")
        self.api_url = api_url
        self.identities = dict(identities or {})

    def cli(self, arguments, *, body=None, timeout=90, reading=False):
        command = [self.executable, "--api-url", self.api_url, "--workspace", self.workspace,
                   "--json", *arguments]
        try:
            result = subprocess.run(command, input=json.dumps(body) if body is not None else None,
                                    text=True, capture_output=True, timeout=timeout, check=False)
        except FileNotFoundError:
            raise ValueError("Install the official autor CLI and run autor login before launching") from None
        except (OSError, subprocess.TimeoutExpired):
            error = ApiError if reading else AmbiguousSubmission
            raise error("SF Compute response is unconfirmed; inspect the saved identity before retrying") from None
        if result.returncode:
            if result.returncode == 3:
                raise ApiError("SF Compute login is unavailable; run autor login (no credentials were logged)")
            # Exit 1 also covers server refusals after a request reached the API.
            # Treat it conservatively on writes; reconciliation is cheaper than a duplicate run.
            error = AmbiguousSubmission if not reading and result.returncode not in {2, 4} else ApiError
            raise error(f"SF Compute CLI failed (exit {result.returncode}); provider output withheld")
        try:
            value = json.loads(result.stdout)
            if not isinstance(value, dict) or value.get("error"):
                raise ValueError()
            return value
        except (ValueError, TypeError):
            error = ApiError if reading else AmbiguousSubmission
            raise error("SF Compute returned an unconfirmed response; inspect the saved run") from None

    def request(self, method, path, body=None, *, reading=False, timeout=90):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Expected a relative SF Compute API path")
        args = ["api", method.lower(), path]
        if body is not None:
            args += ["--data", "-"]  # Secrets and command env values never appear in process arguments.
        return self.cli(args, body=body, reading=reading, timeout=timeout)

    def nodes(self):
        nodes, cursors, identities, cursor = [], set(), set(), None
        for _ in range(1000):
            query = "?" + urlencode({"cursor": cursor}) if cursor else ""
            page = self.request("get", "/nodes" + query, reading=True)
            if not isinstance(page.get("items"), list):
                raise ApiError("Invalid SF Compute node listing")
            for node in page["items"]:
                if (not isinstance(node, dict) or not isinstance(node.get("id"), str)
                        or not node["id"] or node["id"] in identities):
                    raise ApiError("Invalid or duplicate SF Compute node identity")
                identities.add(node["id"])
                self.identities[node["id"]] = safe_name(node["name"])
                nodes.append(node)
            cursor = page.get("next_cursor")
            if cursor is None:
                return nodes
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise ApiError("Invalid SF Compute pagination cursor")
            cursors.add(cursor)
        raise ApiError("SF Compute listing exceeded the pagination limit")

    def name(self, identity):
        if identity not in self.identities:
            self.nodes()
        if identity not in self.identities:
            raise ApiError("Saved SF Compute node is not visible; no replacement was created")
        return self.identities[identity]

    def raw_node(self, identity):
        node = self.request("get", "/nodes/" + quote(self.name(identity), safe=""), reading=True)
        if node.get("id") != identity or node.get("name") != self.name(identity):
            raise ValueError("SF Compute node identity changed")
        return node

    def instances(self):
        return [normalize(node) for node in self.nodes()]

    def instance(self, identity):
        return normalize(self.raw_node(identity))

    def create(self, policy, name):
        # The official CLI normalizes public shape aliases and duration syntax.
        # Its creation answer can be sparse while queued; read the full node
        # before image/price checks. Neither operation retries the allocation.
        arguments = ["node", "create", "--name", safe_name(name), "--chip", policy["chip"],
                     "--image", policy["image"], "--scratch-gib", str(policy["scratch_gib"]),
                     "--max-wait", str(policy["startup_timeout_seconds"]) + "s"]
        if any(c in policy["image"] for c in "/:@"):
            arguments += ["--home", "/home/dev"]
        node = self.cli(arguments)
        if not isinstance(node.get("id"), str) or not node["id"] or node.get("name") != name:
            raise AmbiguousSubmission("SF Compute creation identity is unconfirmed; reconcile the saved name")
        self.identities[node["id"]] = name
        try:
            return self.raw_node(node["id"])
        except (ApiError, ValueError):
            raise AmbiguousSubmission("SF Compute created-node detail is unconfirmed; reconcile the saved name") from None

    def stop(self, identity):
        self.raw_node(identity)
        return self.cli(["node", "stop", self.name(identity)])

    def start(self, identity):
        self.raw_node(identity)
        return self.cli(["run", self.name(identity), "--detach", "--", "true"])


def normalize(node):
    # Listings use state; detail responses use status. Both append explanatory text.
    status = str(node.get("status", node.get("state", "unknown")))
    status = ("running" if status.startswith("running") else "stopped"
              if status.startswith(("stopped", "stopping")) else status)
    chip = node.get("chip")
    count = 8 if chip in {"8xh100", "h100-8"} else 1
    micros = (node.get("rate") or {}).get("node_per_min_usd_micros")
    return {"id": node["id"], "label": node["name"], "actual_status": status,
            "gpu_name": "NVIDIA H100 80GB HBM3", "num_gpus": count,
            "image_uuid": node.get("image"), "gpu_ram": 80 * 1000,
            "dph_total": micros * 60 / 1e6 if type(micros) is int else None}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Unexpected metadata redirect")


class MetadataClient:
    """Inside a node, lifecycle control is restricted to that node, without a token."""
    def __init__(self, identity, name, *, opener=None):
        self.identity, self.node_name = identity, name
        self.opener = opener or build_opener(ProxyHandler({}), NoRedirect())
        self.verify()

    def request(self, path, *, method="GET"):
        request = Request("http://169.254.42.1/v1/" + path, method=method,
                          headers={"Metadata-Flavor": "givemeanode"})
        try:
            with self.opener.open(request, timeout=20) as response:
                return json.load(response)
        except Exception:
            raise RuntimeError("SF Compute node metadata is unavailable; no account credential is used") from None

    def verify(self):
        value = self.request("node")
        node = value.get("node", value)
        if node.get("id", node.get("node_id")) != self.identity or node.get("name") != self.node_name:
            raise ValueError("SF Compute metadata differs from the saved node identity")

    def stop(self, identity):
        if identity != self.identity:
            raise ValueError("SF Compute metadata cannot control another node")
        self.verify()
        return self.request("stop", method="POST")
