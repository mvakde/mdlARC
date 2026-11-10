"""Bounded SSH operations with separately streamed, private credential transfers."""

import base64
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


def public_key(identity):
    """Load only the matching .pub file; private key material never reaches Vast."""
    path = Path(str(Path(identity).expanduser()) + ".pub")
    try:
        parts = path.read_text().split()
        if len(parts) < 2 or not re.fullmatch(r"(?:ssh-|ecdsa-sha2-|sk-)[A-Za-z0-9@._-]+", parts[0]):
            raise ValueError
        blob = base64.b64decode(parts[1], validate=True)
        length = int.from_bytes(blob[:4], "big")
        if len(blob) < 4 or not length or blob[4:4 + length] != parts[0].encode():
            raise ValueError
    except (OSError, UnicodeError, ValueError):
        raise ValueError("Configure a matching OpenSSH public key in <private-key-path>.pub before launching") from None
    # Comments can contain personal metadata and are unnecessary for access.
    return " ".join(parts[:2])


class SSH:
    def __init__(self, instance, identity, known_hosts):
        self.host = instance.get("ssh_host", "")
        self.port = instance.get("ssh_port")
        if os.environ.get("MDLARC_SSH_DIRECT") == "1":
            mappings = (instance.get("ports") or {}).get("22/tcp") or []
            if instance.get("public_ipaddr") and mappings:
                self.host = instance["public_ipaddr"]
                self.port = int(mappings[0]["HostPort"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", self.host) or not isinstance(self.port, int) or not 0 < self.port < 65536:
            raise ValueError("Instance does not yet have a valid SSH endpoint")
        self.identity = Path(identity).expanduser().resolve()
        self.known_hosts = Path(known_hosts).resolve()

    def command(self, args):
        # OpenSSH expands percent tokens even in an absolute known-hosts path.
        known_hosts = str(self.known_hosts).replace("%", "%%")
        return ["ssh", "-T", "-p", str(self.port), "-i", str(self.identity),
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}",
                "root@" + self.host, shlex.join([str(x) for x in args])]

    def run(self, args, *, timeout=60, stdin=None):
        try:
            result = subprocess.run(self.command(args), stdin=stdin, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError("SSH operation timed out or could not connect") from None
        if result.returncode:
            # stderr might contain a remote exception with credentials. Do not expose it.
            raise RuntimeError(f"SSH operation failed (exit {result.returncode}); inspect saved remote status")
        return result.stdout

    def python(self, code, *args, python="python3", timeout=60, stdin=None):
        return self.run([python, "-c", code, *args], timeout=timeout, stdin=stdin)

    def upload(self, local_path, remote_path, *, python="python3", timeout=900):
        from .common import hashes
        digest = hashes(local_path)
        code = """import hashlib, os, pathlib, shutil, sys
p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda:src.read(8388608), b''): h.update(chunk)
    return h.hexdigest()
if p.exists():
    if p.is_symlink() or sha(p) != sys.argv[2]: raise ValueError('Existing transfer differs')
    sys.exit(0)
tmp=p.with_name(p.name+'.incoming')
tmp.unlink(missing_ok=True)
fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    with os.fdopen(fd, 'wb') as out: shutil.copyfileobj(sys.stdin.buffer, out)
    if sha(tmp) != sys.argv[2]: raise ValueError('Transfer checksum failed')
    os.link(tmp,p)
finally: tmp.unlink(missing_ok=True)
"""
        with Path(local_path).open("rb") as source:
            self.python(code, remote_path, digest["sha256"], python=python, timeout=timeout, stdin=source)

    def json_file(self, path, *, python="python3"):
        data = self.python("import pathlib, sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.is_file() else 'null')", path, python=python)
        return json.loads(data)

    def cache_input(self, local_path, remote_path, expected, *, python="python3"):
        """Transfer a pinned dataset once per pod, independently of code snapshots."""
        probe = """import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1])
if not p.exists(): print('missing'); sys.exit(0)
if p.is_symlink() or not p.is_file() or p.stat().st_size != int(sys.argv[3]): raise ValueError('Invalid input cache')
h=hashlib.sha256()
with p.open('rb') as src:
    for chunk in iter(lambda:src.read(8388608),b''):h.update(chunk)
if h.hexdigest()!=sys.argv[2]:raise ValueError('Input cache checksum changed')
print('present')
"""
        if self.python(probe, remote_path, expected["sha256"], str(expected["size"]), python=python).strip() == b"present":
            return False
        from .common import hashes
        actual = hashes(local_path)
        if any(actual[key] != expected[key] for key in ("size", "sha256")):
            raise ValueError("Local family archive differs from saved input identity")
        self.upload(local_path, remote_path, python=python)
        return True

    def launch(self, directory, python):
        # The remote launcher has an exclusive lock and a permanent started marker.
        # Even if SSH loses this response, a repeated observation does not submit another job.
        code = """import os, pathlib, subprocess, sys
p=pathlib.Path(sys.argv[1]); env=dict(os.environ)
plan=__import__('json').loads((p/'launch.json').read_text())
runpod=plan.get('backend') == 'runpod'
creds=pathlib.Path('/root/.mdlarc-runpod-instance.env' if runpod else '/root/.mdlarc-vast-instance.env')
if creds.is_file():
    for line in creds.read_text().splitlines():
        k, _, v=line.partition('=')
        if k in (('RUNPOD_POD_ID','RUNPOD_API_KEY') if runpod else ('CONTAINER_ID','CONTAINER_API_KEY')): env[k]=v
with (p/'supervisor.log').open('ab', buffering=0) as log:
    subprocess.Popen([sys.executable, '-u', str(p/'code/workflows/vast/worker.py'), '--workspace', str(p)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=p)
print('worker-dispatched')
"""
        self.python(code, directory, python=python)

    def logs(self, path, offset, *, python="python3"):
        code = """import base64,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); offset=int(sys.argv[2]); data=b''
if p.is_file():
    with p.open('rb') as src: src.seek(offset); data=src.read(131072)
print(json.dumps({'offset':offset+len(data),'data':base64.b64encode(data).decode()}))
"""
        return json.loads(self.python(code, path, str(offset), python=python))
