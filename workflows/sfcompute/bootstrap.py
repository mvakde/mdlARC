#!/usr/bin/env python3
"""Prepare a persistent venv, retaining the image's Torch, then run the shared supervisor."""

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.vast.common import atomic_json, hashes, relative_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    private, code = workspace / "private", workspace / "code"
    plan = json.loads((workspace / "launch.json").read_text())
    spec_path = relative_file(code, plan["spec_relative"])
    if hashes(workspace / "source.tar.gz")["sha256"] != plan["source"]["sha256"] or hashes(spec_path)["sha256"] != plan["spec_sha256"]:
        raise ValueError("Bootstrap source or spec identity differs")
    from workflows.sfcompute.api import MetadataClient
    MetadataClient(plan["instance_id"], plan["label"])
    # Start the existing independent deadline guard before installing anything.
    with (workspace / "deadline.log").open("ab") as log:
        subprocess.Popen([sys.executable, "-u", str(code / "workflows/vast/worker.py"),
                          "--workspace", str(workspace), "--deadline-only"], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        prepare_environment(workspace, plan)
        interpreter = workspace / "venv/bin/python"
        os.execv(str(interpreter), [str(interpreter), "-u", str(code / "workflows/vast/worker.py"),
                                   "--workspace", str(workspace)])
    except BaseException as exc:
        atomic_json(workspace / "status.json", {"state": "needs_recovery", "ok": False,
            "finalized": False, "backups_complete": False, "error": type(exc).__name__,
            **{k: plan[k] for k in ("branch", "session", "instance_id")}})
        raise


def prepare_environment(workspace, plan):
    import torch
    from src.attention_runtime import selected_versions
    code, private = workspace / "code", workspace / "private"
    spec = json.loads(relative_file(code, plan["spec_relative"]).read_text())
    selected = selected_versions(spec)
    for version, module_name, function in ((2, "flash_attn", "flash_attn_varlen_func"),
                                           (3, "flash_attn_interface", "flash_attn_varlen_func")):
        if version in selected:
            try:
                module = importlib.import_module(module_name)
                if not callable(getattr(module, function, None)):
                    raise ImportError()
            except ImportError:
                raise ValueError(f"FA{version} needs a compatible preinstalled attention package; no paid source build is attempted") from None
    constraint = private / "torch-constraints.txt"
    constraint.write_text("torch==" + torch.__version__ + "\n")
    # Explicitly target the image interpreter. uv's interpreter discovery must not
    # silently select a different Python or pull a replacement CUDA/Torch stack.
    command = (["uv", "venv", "--python", sys.executable, "--system-site-packages", "--seed", str(workspace / "venv")]
               if shutil.which("uv") else [sys.executable, "-m", "venv", "--system-site-packages", str(workspace / "venv")])
    subprocess.run(command, check=True, timeout=60)
    interpreter = workspace / "venv/bin/python"
    env = {**os.environ, "PIP_CONSTRAINT": str(constraint)}
    subprocess.run([str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                    "-r", str(code / "workflows/vast/requirements.txt"), "numpy", "numba", "matplotlib"],
                   env=env, check=True, timeout=300)
    actual = subprocess.check_output([str(interpreter), "-c", "import torch; print(torch.__version__)"], text=True).strip()
    if actual != torch.__version__:
        raise ValueError("Bootstrap changed the image's Torch installation")
    os.environ["PIP_CONSTRAINT"] = str(constraint)
    # Do not edit the immutable launch plan: worker subprocesses already use
    # sys.executable, and exec retains the original source/spec provenance.


if __name__ == "__main__":
    main()
