"""Vast REST client. Rental submissions are never automatically retried."""

import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from workflows.vast.common import PROTECTED_B300_INSTANCES

API_URL = "https://console.vast.ai/api/v0"
API_V1_URL = "https://console.vast.ai/api/v1"


class ApiError(RuntimeError):
    pass


class AmbiguousSubmission(ApiError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ApiError("Unexpected redirect from Vast API")


def api_key(path=None):
    key = os.environ.get("VAST_API_KEY")
    if not key:
        path = Path(path or Path.home() / ".config/vastai/vast_api_key").expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Set VAST_API_KEY or configure an API key outside the repository")
        key = path.read_text().strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Invalid Vast API key")
    return key


class Vast:
    def __init__(self, key, *, opener=None, sleep=time.sleep):
        self.key = key
        self.opener = opener or build_opener(NoRedirect())
        self.sleep = sleep

    def request(self, method, endpoint, body=None, *, reading=False, api_version="v0"):
        target = re.fullmatch(r"/instances/([0-9]+)/?", endpoint.split("?", 1)[0])
        if (target and int(target[1]) in PROTECTED_B300_INSTANCES
                and (method == "DELETE" or (method == "PUT" and isinstance(body, dict) and body.get("state") == "stopped"))):
            raise ApiError("B300 instance 57431449 is protected by the user's no-stop rule; shutdown is forbidden")
        if api_version not in {"v0", "v1"}:
            raise ValueError("Unsupported Vast API version")
        base_url = API_URL if api_version == "v0" else API_V1_URL
        data = json.dumps(body).encode() if body is not None else None
        request = Request(base_url + endpoint, data=data, method=method,
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        attempts = 3 if reading else 1
        for attempt in range(attempts):
            try:
                with self.opener.open(request, timeout=30) as response:
                    result = json.load(response)
                if not isinstance(result, dict) or result.get("success") is False:
                    raise ApiError("Vast API rejected the request")
                return result
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if reading and code in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    self.sleep(2 ** attempt)
                    continue
                if not reading and (code >= 500 or code in {408, 429}):
                    raise AmbiguousSubmission("Vast request outcome is unknown; inspect the saved run before retrying") from None
                raise ApiError(f"Vast API returned HTTP {code}") from None
            except (URLError, TimeoutError, OSError, ValueError):
                if reading and attempt + 1 < attempts:
                    self.sleep(2 ** attempt)
                    continue
                kind = ApiError if reading else AmbiguousSubmission
                raise kind("Vast request could not be confirmed; credentials and provider response were withheld") from None
        raise ApiError("Vast API unavailable")

    def offers(self, query):
        return self.request("POST", "/bundles/", query, reading=True).get("offers", [])

    def instances(self):
        # The old v0 list endpoint returns HTTP 410. Reconciliation must inspect
        # every v1 page before treating a saved rental label as absent.
        query = {"limit": 25, "order_by": json.dumps([{"col": "id", "dir": "asc"}])}
        instances, cursors, identities = [], set(), set()
        for _ in range(1000):
            page = self.request("GET", "/instances/?" + urlencode(query), reading=True, api_version="v1")
            rows = page.get("instances")
            if not isinstance(rows, list):
                raise ApiError("Unexpected instance listing")
            for row in rows:
                if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in identities:
                    raise ApiError("Unexpected instance identity in listing")
                identities.add(row["id"])
                instances.append(row)
            cursor = page.get("next_token")
            if cursor is None:
                return instances
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise ApiError("Unexpected instance pagination cursor")
            cursors.add(cursor)
            query["after_token"] = cursor
        raise ApiError("Instance listing exceeded pagination limit")

    def instance(self, instance_id):
        result = self.request("GET", f"/instances/{int(instance_id)}/", reading=True).get("instances")
        if not isinstance(result, dict) or result.get("id") != int(instance_id):
            raise ApiError("Unexpected instance identity")
        return result

    def create(self, offer_id, policy, label):
        result = self.request("PUT", f"/asks/{int(offer_id)}/", {
            "client_id": "me", "image": policy["image"], "disk": policy["disk_gb"],
            "runtype": "ssh", "label": label, "target_state": "running", "cancel_unavail": True,
            # Preserve per-instance credentials for the detached supervisor without putting them in a snapshot.
            "onstart": "umask 077; env | sed -n '/^CONTAINER_ID=/p; /^CONTAINER_API_KEY=/p' > /root/.mdlarc-vast-instance.env",
        })
        identity = result.get("new_contract")
        if type(identity) is not int or identity <= 0:
            raise AmbiguousSubmission("Rental response lacks an instance ID; reconcile by the saved label")
        return identity

    def attach_ssh_key(self, instance_id, public_key):
        return self.request("POST", f"/instances/{int(instance_id)}/ssh/", {"ssh_key": public_key})

    def label(self, instance_id, label):
        return self.request("PUT", f"/instances/{int(instance_id)}/", {"label": label})

    def stop(self, instance_id):
        return self.request("PUT", f"/instances/{int(instance_id)}/", {"state": "stopped"})

    def start(self, instance_id):
        return self.request("PUT", f"/instances/{int(instance_id)}/", {"state": "running"})

    def destroy(self, instance_id):
        return self.request("DELETE", f"/instances/{int(instance_id)}/")


def public_instance(instance):
    """Never persist Jupyter tokens, SSH keys, environment values, or arbitrary provider fields."""
    fields = ("id", "label", "machine_id", "actual_status", "intended_status", "ssh_host", "ssh_port",
              "gpu_name", "gpu_ram", "num_gpus", "image_uuid", "start_date", "end_date", "dph_total")
    return {key: instance[key] for key in fields if key in instance}
