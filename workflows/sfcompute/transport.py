"""The shared supervisor's transport interface, implemented using Autoresearch exec/files."""

import shlex
from urllib.parse import quote

from workflows.vast.common import hashes
from workflows.vast.ssh import SSH


class Transport(SSH):
    def __init__(self, client, instance):
        if instance.get("actual_status") != "running":
            raise RuntimeError("SF Compute node is not running; observing it never wakes it")
        self.client, self.identity = client, instance["id"]
        self.name = instance["label"]

    def run(self, args, *, timeout=60, stdin=None):
        if stdin is not None:
            raise ValueError("Upload private files through the file API, never through an exec command")
        if self.client.instance(self.identity)["actual_status"] != "running":
            raise RuntimeError("SF Compute node stopped; observation will not wake it")
        result = self.client.cli(["run", self.name, "--timeout", str(min(timeout, 300)),
                                  "--", *[str(x) for x in args]], timeout=min(timeout, 300) + 30)
        if result.get("exit_code") != 0 or result.get("truncated"):
            raise RuntimeError("SF Compute exec failed or truncated output; inspect the saved remote log")
        return result.get("stdout", "").encode()

    def upload(self, local_path, remote_path, *, python="python3", timeout=900):
        expected = hashes(local_path)
        probe = """import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if not p.exists():print('missing')
else:
 if p.is_symlink() or not p.is_file():raise ValueError('Invalid transfer destination')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 if h.hexdigest()!=expected:raise ValueError('Existing transfer differs')
 print('present')
"""
        if self.python(probe, remote_path, expected["sha256"], python=python).strip() == b"present":
            return
        result = self.client.cli(["cp", str(local_path), self.name + ":" + remote_path], timeout=timeout)
        if (result.get("sha256") != expected["sha256"] or result.get("bytes_written") != expected["size"]
                or result.get("sha256_verified") is not True or result.get("written") is not True):
            raise ValueError("SF Compute upload did not verify the exact file identity")
        self.python("import os,sys; os.chmod(sys.argv[1],0o600)", remote_path, python=python)

    def launch(self, directory, python):
        # Mark dispatch before starting bootstrap. A lost response is observed, never resubmitted.
        code = """import json,os,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]); marker=p/'private/dispatched.json'
try:fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
except FileExistsError:print('already-dispatched');sys.exit(0)
with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid()},f)
with (p/'supervisor.log').open('ab',buffering=0) as log:
 subprocess.Popen([sys.executable,'-u',str(p/'code/workflows/sfcompute/bootstrap.py'),'--workspace',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=p)
print('worker-dispatched')
"""
        self.python(code, directory, python=python)
