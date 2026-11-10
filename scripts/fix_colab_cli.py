#!/usr/bin/env python3
"""Repair the installed Colab CLI's missing Jupyter stdin reply, without contacting a VM."""

import argparse
import ast
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


BROKEN_HOOK = '''        def wrapped_stdin_hook(prompt):
            if self.history and self.session_name:
                self.history.log_event(
                    self.session_name, "stdin_request", {"prompt": prompt}
                )

            res = original_stdin_hook(prompt) if original_stdin_hook else input(prompt)

            if self.history and self.session_name:
                self.history.log_event(self.session_name, "input_reply", {"value": res})
            return res
'''

FIXED_HOOK = '''        def wrapped_stdin_hook(prompt):
            # mdlARC stdin repair: a Jupyter hook must send input_reply itself.
            content = prompt.get("content", {})
            prompt_text = content.get("prompt", "")
            secret = bool(content.get("password", False)) or any(
                phrase in prompt_text.lower()
                for phrase in ("verification code", "authorization code")
            )
            if self.history and self.session_name:
                self.history.log_event(
                    self.session_name, "stdin_request", {"password": secret}
                )

            if original_stdin_hook is not None:
                # Custom hooks already send their reply; never send it twice.
                res = original_stdin_hook(prompt)
            else:
                if secret:
                    from getpass import getpass
                    value = getpass(prompt_text)
                else:
                    value = input(prompt_text)
                self.kernel_client._manager.client.input(value)
                res = None

            if self.history and self.session_name:
                # Never persist passwords or OAuth verification codes in history.
                self.history.log_event(self.session_name, "input_reply", {"sent": True})
            return res
'''


def repaired_source(source: str) -> str:
    if FIXED_HOOK in source:
        ast.parse(source)
        return source
    if source.count(BROKEN_HOOK) != 1:
        raise ValueError("Installed CLI has an unrecognized stdin handler; left it unchanged")
    patched = source.replace(BROKEN_HOOK, FIXED_HOOK)
    ast.parse(patched)
    return patched


def installed_runtime_path() -> Path:
    executable = shutil.which("colab")
    if not executable:
        raise RuntimeError("Colab CLI is not installed or is missing from PATH")
    launcher = Path(executable).resolve()
    first_line = launcher.read_text(encoding="utf-8").splitlines()[0]
    interpreter = Path(first_line.removeprefix("#!"))
    if not first_line.startswith("#!/") or not interpreter.is_file():
        raise RuntimeError("Colab CLI does not use a recognized Python launcher")
    result = subprocess.run(
        [str(interpreter), "-c", "import importlib.util; print(importlib.util.find_spec('colab_cli.runtime').origin)"],
        capture_output=True, text=True, check=True,
    )
    runtime = Path(result.stdout.strip())
    if not runtime.is_absolute() or runtime.name != "runtime.py" or runtime.parent.name != "colab_cli":
        raise RuntimeError("Colab CLI returned an unexpected runtime path")
    return runtime


def repair_runtime(path: Path, *, check: bool = False) -> bool:
    """Return whether a repair is needed/applied; keep an original-file backup."""
    source = path.read_text(encoding="utf-8")
    patched = repaired_source(source)
    if source == patched:
        return False
    if check:
        return True
    backup = path.with_name(path.name + ".mdlarc-stdin-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".mdlarc-stdin-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(patched)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check for the bug without changing installed files")
    args = parser.parse_args()
    path = installed_runtime_path()
    changed = repair_runtime(path, check=args.check)
    if args.check and changed:
        print("Colab CLI stdin reply is broken. Run: venv/bin/python scripts/fix_colab_cli.py")
        return 1
    print(f"{'Repaired' if changed else 'Verified'} Colab CLI stdin replies: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Colab CLI repair failed: {exc}") from None
