"""Bounded Runpod API requests; mutations are submitted exactly once."""

import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from workflows.vast.vast_api import ApiError, AmbiguousSubmission

API_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
USER_AGENT = "mdlARC-Runpod/1.0"
CONTROL_FILE = "/root/.mdlarc-runpod-instance.env"

# Only provider-injected, pod-scoped credentials are kept on the pod. The
# account API key and private SSH key never leave the local launcher.
STARTUP = r'''set -eu
umask 077
env | sed -n '/^RUNPOD_POD_ID=/p; /^RUNPOD_API_KEY=/p' > /root/.mdlarc-runpod-instance.env
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
printf '%s\n' "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
if ! test -x /usr/sbin/sshd; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server
fi
mkdir -p /workspace/.mdlarc-runpod-ssh
chmod 700 /workspace/.mdlarc-runpod-ssh
if ! test -f /workspace/.mdlarc-runpod-ssh/host_ed25519_key; then
    ssh-keygen -q -t ed25519 -N '' -f /workspace/.mdlarc-runpod-ssh/host_ed25519_key
fi
exec /usr/sbin/sshd -D -e -o HostKey=/workspace/.mdlarc-runpod-ssh/host_ed25519_key -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no
'''

CONTROL_CHECK = '''import os,pathlib,sys
d=dict(os.environ); p=pathlib.Path('/root/.mdlarc-runpod-instance.env')
if p.is_file(): d.update(dict(line.split('=',1) for line in p.read_text().splitlines() if '=' in line))
assert d.get('RUNPOD_API_KEY') and d.get('RUNPOD_POD_ID') == sys.argv[1], 'Pod-scoped control credentials unavailable'
'''


def pod_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("Invalid Runpod pod ID")
    return value


def api_key(path=None):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        path = Path(path or Path.home() / ".config/mdlarc-runpod/api-key").expanduser()
        if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("Set RUNPOD_API_KEY or configure ~/.config/mdlarc-runpod/api-key outside the repository")
        key = path.read_text().strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Invalid Runpod API key")
    return key


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ApiError("Unexpected redirect from Runpod API")


def normalize(pod):
    """Expose only the fields consumed by the shared lifecycle and observer."""
    identity = pod_id(pod.get("id"))
    desired = pod.get("desiredStatus")
    return {"id": identity, "label": pod.get("name"),
            "actual_status": "stopped" if desired in {"EXITED", "STOPPED"} else "loading",
            "intended_status": "running" if desired == "RUNNING" else "stopped",
            "ssh_host": pod.get("publicIp") or "", "ssh_port": (pod.get("portMappings") or {}).get("22"),
            "gpu_name": (pod.get("gpu") or {}).get("id") or (pod.get("machine") or {}).get("gpuTypeId"),
            "num_gpus": pod.get("gpuCount", (pod.get("gpu") or {}).get("count")),
            "image_uuid": pod.get("imageName") or pod.get("image"), "dph_total": pod.get("costPerHr")}


class Runpod:
    def __init__(self, key, *, opener=None, sleep=time.sleep, own_pod=None):
        self.key, self.opener, self.sleep = key, opener or build_opener(NoRedirect()), sleep
        self.own_pod = pod_id(own_pod) if own_pod else None

    def request(self, method, endpoint, body=None, *, reading=False, graphql=False):
        url = GRAPHQL_URL if graphql else API_URL + endpoint
        data = json.dumps(body, allow_nan=False).encode() if body is not None else None
        request = Request(url, data=data, method=method,
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                                   "User-Agent": USER_AGENT})
        attempts = 3 if reading else 1
        for attempt in range(attempts):
            try:
                with self.opener.open(request, timeout=30) as response:
                    raw = response.read()
                result = json.loads(raw) if raw else None
                if graphql:
                    if not isinstance(result, dict) or result.get("errors") or not isinstance(result.get("data"), dict):
                        raise ApiError("Runpod rejected the catalog query; provider details withheld")
                    return result["data"]
                return result
            except HTTPError as exc:
                code = exc.code
                reason = ""
                try:
                    problem = json.loads(exc.read(16384))
                    if isinstance(problem, dict) and str(problem.get("error_code")) == "1010" and problem.get("cloudflare_error") is True:
                        reason = " (Cloudflare rejected the client signature)"
                except (ValueError, OSError):
                    pass
                exc.close()
                if reading and code in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    self.sleep(2 ** attempt)
                    continue
                kind = AmbiguousSubmission if not reading and (code >= 500 or code in {408, 429}) else ApiError
                raise kind(f"Runpod API returned HTTP {code}{reason}; inspect the saved run before retrying") from None
            except (URLError, TimeoutError, OSError, ValueError):
                if reading and attempt + 1 < attempts:
                    self.sleep(2 ** attempt)
                    continue
                kind = ApiError if reading else AmbiguousSubmission
                raise kind("Runpod request could not be confirmed; credentials and provider response withheld") from None

    def gpu_types(self, policy):
        secure = "true" if policy["cloud_type"] == "SECURE" else "false"
        query = ("query { gpuTypes { id memoryInGb lowestPrice(input: { gpuCount: "
                 + str(policy["num_gpus"]) + ", secureCloud: " + secure
                 + " }) { uninterruptablePrice stockStatus } } }")
        rows = self.request("POST", "", {"query": query}, reading=True, graphql=True).get("gpuTypes")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ApiError("Unexpected Runpod GPU catalog")
        return rows

    def supported_cuda_versions(self):
        """Read the live create contract before submitting a billable request."""
        schema = self.request("GET", "/openapi.json", reading=True)
        try:
            body = schema["paths"]["/pods"]["post"]["requestBody"]["content"]["application/json"]["schema"]
            if "$ref" in body:
                prefix = "#/components/schemas/"
                if not body["$ref"].startswith(prefix):
                    raise ValueError
                body = schema["components"]["schemas"][body["$ref"][len(prefix):]]
            versions = body["properties"]["allowedCudaVersions"]["items"]["enum"]
        except (KeyError, TypeError, ValueError):
            raise ApiError("Runpod create schema does not describe supported CUDA versions") from None
        if not isinstance(versions, list) or not versions or not all(isinstance(v, str) and v for v in versions):
            raise ApiError("Runpod create schema has invalid CUDA versions")
        return versions

    def instances(self):
        rows = self.request("GET", "/pods", reading=True)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ApiError("Unexpected Runpod pod listing")
        instances = [normalize(row) for row in rows]
        if len({row["id"] for row in instances}) != len(instances):
            raise ApiError("Duplicate Runpod pod identity")
        return instances

    def raw_pod(self, identity):
        identity = pod_id(identity)
        result = self.request("GET", f"/pods/{identity}?includeMachine=true", reading=True)
        if not isinstance(result, dict) or result.get("id") != identity:
            raise ApiError("Unexpected Runpod pod identity")
        return result

    def instance(self, identity):
        return normalize(self.raw_pod(identity))

    def create(self, body):
        result = self.request("POST", "/pods", body)
        try:
            return pod_id(result.get("id"))
        except (AttributeError, ValueError):
            raise AmbiguousSubmission("Pod creation response lacks an ID; reconcile by the saved name") from None

    def _control(self, identity, method, suffix=""):
        identity = pod_id(identity)
        if self.own_pod is not None and identity != self.own_pod:
            raise ValueError("Pod-scoped client refuses another pod")
        return self.request(method, f"/pods/{identity}{suffix}")

    def stop(self, identity):
        return self._control(identity, "POST", "/stop")

    def start(self, identity):
        return self._control(identity, "POST", "/start")

    def destroy(self, identity):
        return self._control(identity, "DELETE")
