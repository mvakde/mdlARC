"""Unpack the source code embedded in a submitted Kaggle notebook.

Each notebook carries its whole runtime (src/ and kaggle/ modules) as one base64 zip, so it runs
with Internet off. This script finds that zip, checks the SHA-256 the notebook itself pins, and
writes the files out so you can read or diff exactly the code that produced the score.

    python kaggle_submissions/extract_runtime.py \
        kaggle_submissions/notebooks/submit_d1024_e440_ddp4_ft180.ipynb out/d1024_runtime
"""

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

SHA_PATTERN = re.compile(r"RUNTIME_SHA256 = '([0-9a-f]{64})'")
B64_PATTERN = re.compile(r"RUNTIME_B64 = '([A-Za-z0-9+/=]+)'")


def embedded_runtime(notebook: Path) -> tuple[bytes, str]:
    cells = json.loads(notebook.read_text())["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        sha = SHA_PATTERN.search(source)
        payload = B64_PATTERN.search(source)
        if sha and payload:
            return base64.b64decode(payload.group(1)), sha.group(1)
    raise SystemExit(f"{notebook}: no embedded runtime found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data, expected = embedded_runtime(args.notebook)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise SystemExit(f"checksum mismatch: notebook pins {expected}, payload is {actual}")

    root = args.output.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            target = (root / entry.filename).resolve()
            target.relative_to(root)  # refuse paths outside the output folder
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
        names = archive.namelist()
    print(f"runtime sha256 {actual}: {len(names)} files -> {root}")


if __name__ == "__main__":
    main()
