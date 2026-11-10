"""Pack non-checkpoint run records and publish immutable, verified Drive copies.

The VM publisher uses the VM's dedicated account. Local pack operations need no
Drive credentials. After a VM stops, Codex can upload the same bundle through the
connected Drive plugin.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

if os.environ.get("MDLARC_PUBLISH_RECORDS") == "1":
    # An older training snapshot may not contain run_layout.py.
    from urllib.parse import quote
    branch_name = lambda root: os.environ["MDLARC_RUN_BRANCH"]
    branch_folder = lambda name: quote(name, safe="-_.")
    results_root = lambda root: root / 'results' / branch_folder(branch_name(root))
    sys.path.insert(0, '/content/mdlARC/scripts')
elif __package__:
    from .run_layout import branch_folder, branch_name, results_root
else:
    from run_layout import branch_folder, branch_name, results_root

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
BINARY_DIRS = {"checkpoints", "pretrain_parts", "wheel_parts"}


def is_record(path: Path) -> bool:
    return (path.is_file() and not path.is_symlink()
            and path.suffix not in CHECKPOINT_SUFFIXES | {".lock", ".whl"}
            and not BINARY_DIRS.intersection(path.parts))


def record_files(root: Path, session: str | None = None) -> list[Path]:
    base = results_root(root)
    if session is None:
        candidates = list(base.rglob("*"))
    else:
        if not session or Path(session).name != session or session in {".", ".."}:
            raise ValueError("Invalid session name")
        candidates = list((base / "colab_runs" / session).rglob("*"))
        candidates += list((base / "colab_runs" / "_active").glob(f"{session}.*"))
        candidates += [base / "experiments" / "notes" / f"{session}.md",
                       base / "experiments" / "generated" / f"{session}.md",
                       base / "experiments" / "INDEX.md"]
    return sorted(set(path for path in candidates if is_record(path)
                      and 'drive_sync' not in path.parts))


def pack_records(root: Path, destination: Path, session: str | None = None) -> dict:
    files = record_files(root, session)
    files = [path for path in files if path.resolve() != destination.resolve()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = results_root(root)
    with destination.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in files:
                info = archive.gettarinfo(str(path), arcname=path.relative_to(base).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
    data = destination.read_bytes()
    return {"branch": branch_name(root), "session": session, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "md5": hashlib.md5(data).hexdigest(),
            "files": [path.relative_to(base).as_posix() for path in files]}


def publish_bundle(store, folder: str, bundle: Path) -> dict:
    """A repeated sync reuses identical content; never overwrite historical records."""
    store.accept_run_folder(folder)
    data = bundle.read_bytes()
    md5, sha256 = hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()
    name = f"local-records-{sha256}.tar.gz"
    response = store.service.files().list(
        q=f"'{folder}' in parents and trashed = false and name = '{name}'",
        fields="files(id,name,size,md5Checksum,parents)", pageSize=100,
    ).execute(num_retries=3)
    if response.get("nextPageToken") or len(response["files"]) > 1:
        raise ValueError("Duplicate record backups require inspection")
    if response["files"]:
        item = response["files"][0]
        if (item.get("parents") != [folder] or int(item.get("size", -1)) != len(data)
                or item.get("md5Checksum") != md5):
            raise ValueError("Existing record backup failed checksum verification")
        return {"id": item["id"], "size": len(data), "sha256": sha256, "md5": md5}
    return store.upload_new(bundle, folder, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch")
    parser.add_argument("--session")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-vm", action="store_true")
    publishing = os.environ.get("MDLARC_PUBLISH_RECORDS") == "1"
    args = parser.parse_args([] if publishing else None)
    args.publish_vm = args.publish_vm or publishing
    if args.branch:
        os.environ["MDLARC_RUN_BRANCH"] = args.branch
    if args.publish_vm:
        # This is a bounded publishing operation, separate from training.
        # It also works on an older source snapshot after its run folder moved.
        from colab_drive import SharedDriveStore, build_shared_service, load_config, ROOT_FOLDER_ID, FOLDER_MIME
        root = Path('/content/mdlARC')
        store = SharedDriveStore(load_config(root / 'scripts/colab_drive.json'), build_shared_service())
        store.preflight()
        folder = os.environ['MDLARC_RECORDS_FOLDER']
        store.allowed.add(folder)
        metadata = store.metadata(folder)
        parents = metadata.get('parents', [])
        if len(parents) != 1:
            raise ValueError('Run folder has an ambiguous parent')
        parent_id = parents[0]
        store.allowed.add(parent_id)
        parent = store.metadata(parent_id)
        if (parent.get('name') != branch_folder(branch_name(root))
                or parent.get('parents') != [store.config['runs_folder_id']]
                or parent.get('mimeType') != FOLDER_MIME):
            raise ValueError('Run folder is outside the selected branch')
        store.run_parent_id = parent_id
        result = publish_bundle(store, folder, Path('/content/mdlARC-local-records.tar.gz'))
        Path('/content/mdlARC-local-records-status.json').write_text(json.dumps(result))
        print('Run records verified on Drive.', flush=True)
    else:
        root = Path(__file__).resolve().parents[1]
        if args.output is None:
            parser.error('--output is required for a local pack')
        print(json.dumps(pack_records(root, args.output, args.session), indent=2))


if __name__ == '__main__':
    main()
