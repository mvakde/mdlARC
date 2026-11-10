import hashlib
import json
from pathlib import Path
import sys
import tarfile
import threading
import zipfile
from types import SimpleNamespace

import pytest

from scripts import colab_drive, colab_remote, run_on_colab

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpoint_io import save_checkpoint_atomically, save_rolling_checkpoint


def config():
    return {"root_folder_id": colab_drive.ROOT_FOLDER_ID, "runs_folder_id": "shared-runs", "wheel_file_id": "wheel",
            "environment_file_id": "environment", "wheel_ancestor_ids": ["wheel_folder", colab_drive.ROOT_FOLDER_ID],
            "dataset_folder_id": "datasets", "dataset_file_id": "task-variations",
            "dataset_name": "task-variations-v1.zip", "dataset_size": 0, "dataset_sha256": "0" * 64}


class Call:
    def __init__(self, result):
        self.result = result

    def execute(self, **kwargs):
        return self.result


class FakeFiles:
    def __init__(self):
        root = colab_drive.ROOT_FOLDER_ID
        self.items = {
            root: {"id": root, "mimeType": colab_drive.FOLDER_MIME},
            "wheel_folder": {"id": "wheel_folder", "mimeType": colab_drive.FOLDER_MIME, "parents": [root]},
            "wheel": {"id": "wheel", "mimeType": "application/octet-stream", "parents": ["wheel_folder"]},
            "environment": {"id": "environment", "mimeType": "application/json", "parents": ["wheel_folder"]},
            "datasets": {"id": "datasets", "mimeType": colab_drive.FOLDER_MIME, "parents": [root]},
            "task-variations": {"id": "task-variations", "name": "task-variations-v1.zip",
                                "mimeType": "application/zip", "size": "0", "parents": ["datasets"]},
        }
        self.reads, self.creates = [], []
        self.corrupt_upload = False
        self.upload_raises = False

    def get(self, fileId, fields):
        self.reads.append(fileId)
        return Call(self.items[fileId])

    def create(self, body, fields, media_body=None):
        self.creates.append(body)
        file_id = f"new-{len(self.creates)}"
        result = {"id": file_id, **body}
        if media_body is None:
            self.items[file_id] = result
            return Call(result)
        data = Path(media_body._filename).read_bytes()
        result.update(size=str(len(data)), md5Checksum=hashlib.md5(data).hexdigest())
        if self.corrupt_upload:
            result["md5Checksum"] = "wrong"
        self.items[file_id] = result

        def next_chunk(**kwargs):
            if self.upload_raises:
                raise RuntimeError("provider details that should not be logged")
            return None, result
        return SimpleNamespace(next_chunk=next_chunk)


def store():
    files = FakeFiles()
    return colab_drive.DriveStore(config(), SimpleNamespace(files=lambda: files)), files


def test_cross_branch_checkpoint_requires_explicit_matching_source(monkeypatch, tmp_path):
    files=FakeFiles();content=b'checkpoint';root=colab_drive.ROOT_FOLDER_ID
    files.items.update({
        'source-branch':{'id':'source-branch','name':'codex%2Fhistorical','mimeType':colab_drive.FOLDER_MIME,'parents':['shared-runs']},
        'source-run':{'id':'source-run','name':'old','mimeType':colab_drive.FOLDER_MIME,'parents':['source-branch']},
        'checkpoint':{'id':'checkpoint','name':'pretrain__tiny.pt','mimeType':'application/octet-stream','parents':['source-run'],
            'size':str(len(content)),'md5Checksum':hashlib.md5(content).hexdigest()}})
    files.get_media=lambda fileId:content
    target=colab_drive.SharedDriveStore(config(),SimpleNamespace(files=lambda:files))
    target.run_parent_id='current-branch'
    monkeypatch.setattr(target, '_download_checkpoint_ranges', lambda file_id, output, size: output.write(content))
    args=dict(expected_name='pretrain__tiny.pt',expected_size=len(content),expected_sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError,match='selected branch'):
        target.download_input_checkpoint('checkpoint',tmp_path/'default.pt',**args)
    with pytest.raises(ValueError,match='ancestry'):
        target.download_input_checkpoint('checkpoint',tmp_path/'wrong.pt',source_branch='codex/other',**args)
    result=target.download_input_checkpoint('checkpoint',tmp_path/'valid.pt',source_branch='codex/historical',**args)
    assert result['source_run_folder_id']=='source-run' and (tmp_path/'valid.pt').read_bytes()==content
    assert not files.creates


@pytest.fixture
def mounted_checkpoint(tmp_path):
    content = b"verified checkpoint bytes"
    files = FakeFiles()
    files.items.update({
        "source-branch": {"id": "source-branch", "name": "codex%2Fhistorical",
                          "mimeType": colab_drive.FOLDER_MIME, "parents": ["shared-runs"]},
        "source-run": {"id": "source-run", "name": "old-run",
                       "mimeType": colab_drive.FOLDER_MIME, "parents": ["source-branch"]},
        "checkpoint": {"id": "checkpoint", "name": "pretrain__tiny.pt",
                       "mimeType": "application/octet-stream", "parents": ["source-run"],
                       "size": str(len(content)), "md5Checksum": hashlib.md5(content).hexdigest(),
                       "sha256Checksum": hashlib.sha256(content).hexdigest()},
    })
    files.get_media = lambda **kwargs: pytest.fail("Mounted copy must not issue API media requests")
    target = colab_drive.SharedDriveStore(config(), SimpleNamespace(files=lambda: files))
    target.run_parent_id = "current-branch"
    runs_root = tmp_path / "mounted-runs"
    source = runs_root / "codex%2Fhistorical/old-run/pretrain__tiny.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    destination = tmp_path / "local/inputs/pretrain.pt"
    args = dict(expected_name=source.name, expected_size=len(content),
                expected_sha256=hashlib.sha256(content).hexdigest(),
                source_branch="codex/historical", mounted_runs_root=runs_root)
    return target, files, source, destination, args


@pytest.mark.parametrize("checksum_from_drive", [False, True])
def test_mounted_checkpoint_copies_verified_bytes_without_api_media(mounted_checkpoint, checksum_from_drive):
    target, files, source, destination, args = mounted_checkpoint
    if checksum_from_drive:
        args["expected_sha256"] = None
    result = target.download_input_checkpoint("checkpoint", destination, **args)
    assert destination.read_bytes() == source.read_bytes()
    assert result["transfer_method"] == "mounted_copy"
    assert result["transfer_seconds"] >= 0 and result["verification_seconds"] >= 0
    assert result["sha256_source"] == ("drive_metadata" if checksum_from_drive else "supplied")
    assert result["source_run_folder_id"] == "source-run"
    assert not destination.with_name(destination.name + ".partial").exists()
    assert not files.creates


@pytest.mark.parametrize("failure", ["size", "md5", "sha256", "missing"])
def test_mounted_checkpoint_rejects_missing_or_corrupt_bytes(mounted_checkpoint, failure):
    target, files, source, destination, args = mounted_checkpoint
    if failure == "size":
        source.write_bytes(b"short")
    elif failure == "md5":
        files.items["checkpoint"]["md5Checksum"] = "0" * 32
    elif failure == "sha256":
        args["expected_sha256"] = "0" * 64
    else:
        source.unlink()
    with pytest.raises(ValueError):
        target.download_input_checkpoint("checkpoint", destination, **args)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


@pytest.mark.parametrize("failure", ["traversal", "symlink", "wrong_branch", "existing_partial", "mounted_destination"])
def test_mounted_checkpoint_preserves_path_and_staging_guards(mounted_checkpoint, failure, tmp_path):
    target, files, source, destination, args = mounted_checkpoint
    error = ValueError
    if failure == "traversal":
        files.items["source-run"]["name"] = "../old-run"
    elif failure == "symlink":
        outside = tmp_path / "outside.pt"
        source.rename(outside)
        source.symlink_to(outside)
    elif failure == "wrong_branch":
        args["source_branch"] = "codex/wrong"
    elif failure == "existing_partial":
        error = FileExistsError
        destination.parent.mkdir(parents=True)
        destination.with_name(destination.name + ".partial").write_bytes(b"in-progress")
    else:
        destination = source.parent / "local-copy.pt"
    with pytest.raises(error):
        target.download_input_checkpoint("checkpoint", destination, **args)
    assert not destination.exists()
    assert source.read_bytes() == b"verified checkpoint bytes"
    if failure == "existing_partial":
        assert destination.with_name(destination.name + ".partial").read_bytes() == b"in-progress"


def test_mounted_copy_failure_removes_only_its_partial_file(monkeypatch, mounted_checkpoint):
    target, files, source, destination, args = mounted_checkpoint
    def failed_copy(input_file, output, **kwargs):
        output.write(b"partial")
        raise OSError("copy interrupted")
    monkeypatch.setattr(colab_drive.shutil, "copyfileobj", failed_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        target.download_input_checkpoint("checkpoint", destination, **args)
    assert source.read_bytes() == b"verified checkpoint bytes"
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_mounted_checkpoint_rejects_destination_elsewhere_on_same_mount(monkeypatch, mounted_checkpoint):
    target, files, source, destination, args = mounted_checkpoint
    mounted_root = args["mounted_runs_root"]
    original = Path.is_mount
    monkeypatch.setattr(Path, "is_mount", lambda path: path == mounted_root.parent or original(path))
    with pytest.raises(ValueError, match="outside the Drive mount"):
        target.download_input_checkpoint("checkpoint", mounted_root.parent / "other-folder/local.pt", **args)
    assert not (mounted_root.parent / "other-folder/local.pt").exists()


def test_mounted_checkpoint_supports_historical_runs_without_branch_directory(mounted_checkpoint):
    target, files, source, destination, args = mounted_checkpoint
    target.run_parent_id = "shared-runs"
    files.items["source-run"]["parents"] = ["shared-runs"]
    args.pop("source_branch")
    new_source = args["mounted_runs_root"] / "old-run" / source.name
    new_source.parent.mkdir()
    source.rename(new_source)
    result = target.download_input_checkpoint("checkpoint", destination, **args)
    assert destination.read_bytes() == new_source.read_bytes()
    assert result["transfer_method"] == "mounted_copy"


@pytest.fixture
def checkpoint_api(monkeypatch, mounted_checkpoint):
    from google.auth.transport import requests as google_requests
    import requests

    target, files, source, destination, args = mounted_checkpoint
    args.pop("mounted_runs_root")
    content = source.read_bytes()
    credential = object()
    uri = "https://www.googleapis.com/drive/v3/files/checkpoint?alt=media"
    files.get_media = lambda fileId: SimpleNamespace(uri=uri, http=SimpleNamespace(credentials=credential))
    state = {"calls": [], "sleeps": [], "closed": False,
             "default_range_bytes": colab_drive.CHECKPOINT_API_RANGE_BYTES,
             "default_stream_bytes": colab_drive.CHECKPOINT_API_STREAM_BYTES}
    monkeypatch.setattr(colab_drive, "CHECKPOINT_API_RANGE_BYTES", 8)
    monkeypatch.setattr(colab_drive, "CHECKPOINT_API_STREAM_BYTES", 2)
    monkeypatch.setattr(colab_drive.time, "sleep", state["sleeps"].append)

    class Response:
        def __init__(self, first, last):
            self.status_code = 206
            self.headers = {"Content-Range": f"bytes {first}-{last}/{len(content)}"}
            self.data = content[first:last + 1]
            failure = state.get("failure")
            if failure == "ignored_range":
                self.status_code = 200
            elif failure == "wrong_range":
                self.headers["Content-Range"] = f"bytes {first + 1}-{last}/{len(content)}"
            elif failure == "wrong_total":
                self.headers["Content-Range"] = f"bytes {first}-{last}/{len(content) + 1}"
            elif failure == "compressed":
                self.headers["Content-Encoding"] = "gzip"
            elif failure == "truncated":
                self.data = self.data[:-1]
            elif failure == "overlong":
                self.data += b"x"
            elif failure == "corrupt":
                self.data = bytes([self.data[0] ^ 1]) + self.data[1:]
            if state.get("transient_http") and len(state["calls"]) == 1:
                self.status_code = 503

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            raise requests.HTTPError("provider credential details must not be logged")

        def iter_content(self, chunk_size):
            assert chunk_size == 2
            for offset in range(0, len(self.data), chunk_size):
                yield self.data[offset:offset + chunk_size]
                if state.get("interrupt_once") and len(state["calls"]) == 1:
                    raise requests.ConnectionError("provider credential details must not be logged")

    class Session:
        def __init__(self, credentials, refresh_timeout):
            assert credentials is credential and refresh_timeout == 30

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

        def get(self, url, *, headers, stream, timeout):
            assert url == uri and stream is True and timeout == (15, 120)
            assert headers["Accept-Encoding"] == "identity"
            first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            state["calls"].append((first, last))
            if state.get("always_disconnect"):
                raise requests.ConnectionError("provider credential details must not be logged")
            return Response(first, last)

    monkeypatch.setattr(google_requests, "AuthorizedSession", Session)
    return target, destination, args, content, state


def test_default_checkpoint_api_streams_uncompressed_ranges_and_verifies_bytes(checkpoint_api):
    target, destination, args, content, state = checkpoint_api
    result = target.download_input_checkpoint("checkpoint", destination, **args)
    assert state["default_range_bytes"] == 256 * 1024 * 1024
    assert state["default_stream_bytes"] == 1024 * 1024
    assert state["calls"] == [(0, 7), (8, 15), (16, 23), (24, len(content) - 1)]
    assert destination.read_bytes() == content and state["closed"]
    assert result["transfer_method"] == "drive_api" and result["accept_encoding"] == "identity"
    assert result["api_range_bytes"] == 8 and result["api_stream_bytes"] == 2
    assert not destination.with_name(destination.name + ".partial").exists()


@pytest.mark.parametrize("failure", ["ignored_range", "wrong_range", "wrong_total", "compressed", "truncated", "overlong", "corrupt"])
def test_checkpoint_api_never_publishes_invalid_response_bytes(checkpoint_api, failure):
    target, destination, args, content, state = checkpoint_api
    state["failure"] = failure
    with pytest.raises(ValueError):
        target.download_input_checkpoint("checkpoint", destination, **args)
    assert not destination.exists() and state["closed"]
    assert not destination.with_name(destination.name + ".partial").exists()


@pytest.mark.parametrize("failure", ["interrupt_once", "transient_http"])
def test_checkpoint_api_retries_from_last_written_byte(checkpoint_api, failure):
    target, destination, args, content, state = checkpoint_api
    state[failure] = True
    target.download_input_checkpoint("checkpoint", destination, **args)
    resumed_at = 2 if failure == "interrupt_once" else 0
    assert state["calls"][:2] == [(0, 7), (resumed_at, 7)]
    assert state["sleeps"] == [1]
    assert destination.read_bytes() == content and state["closed"]


def test_checkpoint_api_bounds_retries_and_redacts_provider_errors(checkpoint_api):
    target, destination, args, content, state = checkpoint_api
    state["always_disconnect"] = True
    with pytest.raises(RuntimeError, match="bounded retries") as failure:
        target.download_input_checkpoint("checkpoint", destination, **args)
    assert "provider credential" not in str(failure.value)
    assert state["calls"] == [(0, 7)] * 4 and state["sleeps"] == [1, 2, 4]
    assert not destination.exists() and state["closed"]
    assert not destination.with_name(destination.name + ".partial").exists()


def test_scope_and_destination_cannot_expand(tmp_path):
    broad = {"type": "authorized_user", "scopes": ["https://www.googleapis.com/auth/drive"]}
    with pytest.raises(ValueError, match="drive.file"):
        colab_drive.validate_token_data(broad)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**config(), "root_folder_id": "personal-folder"}))
    with pytest.raises(ValueError, match="restricted"):
        colab_drive.load_config(path)
    target, files = store()
    with pytest.raises(ValueError, match="unapproved"):
        target.metadata("personal-file")
    assert files.reads == []


def test_wheel_moved_outside_root_fails_without_following_personal_parent():
    target, files = store()
    files.items["wheel_folder"]["parents"] = ["personal-folder"]
    with pytest.raises(ValueError, match="inside"):
        target.preflight()
    assert "personal-folder" not in files.reads


def test_dataset_download_is_opt_in_and_rejects_a_moved_file(monkeypatch, tmp_path):
    from googleapiclient import http
    data = b"reviewed dataset archive"
    dataset_config = {**config(), "dataset_size": len(data),
                      "dataset_sha256": hashlib.sha256(data).hexdigest()}
    files = FakeFiles()
    files.items["task-variations"]["size"] = str(len(data))
    target = colab_drive.DriveStore(dataset_config, SimpleNamespace(files=lambda: files))
    target.preflight()
    assert "task-variations" not in files.reads
    files.items["task-variations"]["parents"] = ["personal-folder"]
    with pytest.raises(ValueError, match="approved Drive file"):
        target.preflight(include_dataset=True)
    assert "personal-folder" not in files.reads
    files.items["task-variations"]["parents"] = ["datasets"]
    files.get_media = lambda fileId: data

    class Download:
        def __init__(self, output, content, **kwargs):
            self.output, self.content = output, content

        def next_chunk(self, **kwargs):
            self.output.write(self.content)
            return None, True

    monkeypatch.setattr(http, "MediaIoBaseDownload", Download)
    destination = tmp_path / "task-variations-v1.zip"
    result = target.download_dataset_zip(destination)
    assert destination.read_bytes() == data
    assert result["sha256"] == dataset_config["dataset_sha256"]
    with pytest.raises(ValueError, match="new"):
        target.download_dataset_zip(destination)


def test_dataset_download_rejects_hash_mismatch(monkeypatch, tmp_path):
    from googleapiclient import http
    target, files = store()
    target.config["dataset_size"] = 5
    target.config["dataset_sha256"] = hashlib.sha256(b"right").hexdigest()
    files.items["task-variations"]["size"] = "5"
    files.get_media = lambda fileId: b"wrong"

    class Download:
        def __init__(self, output, content, **kwargs):
            self.output, self.content = output, content

        def next_chunk(self, **kwargs):
            self.output.write(self.content)
            return None, True

    monkeypatch.setattr(http, "MediaIoBaseDownload", Download)
    destination = tmp_path / "task-variations-v1.zip"
    with pytest.raises(ValueError, match="SHA-256"):
        target.download_dataset_zip(destination)
    assert not destination.exists()
    assert not (tmp_path / "task-variations-v1.zip.partial").exists()


def test_dataset_extraction_skips_macos_metadata_and_rejects_traversal(tmp_path):
    archive = tmp_path / "task-variations-v1.zip"
    destination = tmp_path / "assets_tmp/exported-datasets"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("exported-datasets/task-variation/arc2 train/example.json", "{}")
        output.writestr("__MACOSX/exported-datasets/._example", "metadata")
        output.writestr("exported-datasets/.DS_Store", "metadata")
    assert colab_remote.unpack_task_variations(archive, destination) == 1
    assert (destination / "task-variation/arc2 train/example.json").read_text() == "{}"
    assert not (destination / ".DS_Store").exists()
    destination.rename(tmp_path / "previous")
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("exported-datasets/../../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        colab_remote.unpack_task_variations(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_upload_is_new_verified_file_under_run_folder(tmp_path):
    target, files = store()
    target.preflight()
    folder = target.create_run_folder("session")
    checkpoint = tmp_path / "epoch-01.pt"
    checkpoint.write_bytes(b"saved checkpoint")
    result = target.upload_new(checkpoint, folder, checkpoint.name)
    assert files.creates[0]["parents"] == [colab_drive.ROOT_FOLDER_ID]
    assert files.creates[1] == {"name": checkpoint.name, "parents": [folder]}
    assert result["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["size"] == checkpoint.stat().st_size
    with pytest.raises(ValueError, match="verified"):
        target.upload_new(checkpoint, colab_drive.ROOT_FOLDER_ID, checkpoint.name)
    files.items[folder]["parents"] = ["personal-folder"]
    with pytest.raises(ValueError, match="inside"):
        target.upload_new(checkpoint, folder, checkpoint.name)
    assert len(files.creates) == 2


@pytest.mark.parametrize("checksum_from_drive", [False, True])
def test_shared_account_uses_runs_folder_and_verifies_checkpoint(monkeypatch, tmp_path, checksum_from_drive):
    monkeypatch.delenv("MDLARC_RUN_BRANCH", raising=False)
    files = FakeFiles()
    root = colab_drive.ROOT_FOLDER_ID
    runs = "shared-runs"
    old_run = "old-run"
    checkpoint_id = "saved-checkpoint"
    content = b"verified pretrain checkpoint"
    files.items[runs] = {"id": runs, "name": "runs", "mimeType": colab_drive.FOLDER_MIME,
                         "parents": [root], "capabilities": {"canAddChildren": True}}
    files.items[old_run] = {"id": old_run, "name": "older-session", "mimeType": colab_drive.FOLDER_MIME,
                            "parents": [runs]}
    files.items[checkpoint_id] = {
        "id": checkpoint_id, "name": "pretrain__tiny.pt", "mimeType": "application/octet-stream",
        "parents": [old_run], "size": str(len(content)),
        "md5Checksum": hashlib.md5(content).hexdigest(),
        "sha256Checksum": hashlib.sha256(content).hexdigest(),
    }
    files.get_media = lambda fileId: content
    about = SimpleNamespace(get=lambda **kwargs: Call({"user": {"emailAddress": "your-drive-account@example.com"}}))
    target = colab_drive.SharedDriveStore(config(), SimpleNamespace(files=lambda: files, about=lambda: about))

    monkeypatch.setattr(target, "_download_checkpoint_ranges", lambda file_id, output, size: output.write(content))
    target.preflight()
    assert target.run_parent_id == runs
    new_run = target.create_run_folder("new-session")
    assert files.creates[0]["parents"] == [runs]
    destination = tmp_path / "pretrain/tiny.pt"
    result = target.download_input_checkpoint(
        checkpoint_id, destination, expected_name="pretrain__tiny.pt",
        expected_size=len(content), expected_sha256=None if checksum_from_drive else hashlib.sha256(content).hexdigest())
    assert destination.read_bytes() == content
    assert result["source_run_folder_id"] == old_run
    assert result["sha256_source"] == ("drive_metadata" if checksum_from_drive else "supplied")
    if checksum_from_drive:
        files.items[checkpoint_id].pop("sha256Checksum")
        with pytest.raises(ValueError, match="no SHA-256"):
            target.download_input_checkpoint(checkpoint_id, tmp_path / "missing-hash.pt",
                                             expected_name="pretrain__tiny.pt", expected_size=len(content),
                                             expected_sha256=None)
        files.items[checkpoint_id]["sha256Checksum"] = "0" * 64
        with pytest.raises(ValueError, match="SHA-256 verification"):
            target.download_input_checkpoint(checkpoint_id, tmp_path / "wrong-hash.pt",
                                             expected_name="pretrain__tiny.pt", expected_size=len(content),
                                             expected_sha256=None)
        assert not (tmp_path / "wrong-hash.pt").exists()
        assert not (tmp_path / "wrong-hash.pt.partial").exists()
    files.items[old_run]["parents"] = [root]
    with pytest.raises(ValueError, match="under mdlARC/runs"):
        target.download_input_checkpoint(checkpoint_id, tmp_path / "other.pt",
                                         expected_name="pretrain__tiny.pt", expected_size=len(content),
                                         expected_sha256=hashlib.sha256(content).hexdigest())
    assert new_run in target.run_folders


def test_shared_account_rejects_wrong_identity():
    files = FakeFiles()
    about = SimpleNamespace(get=lambda **kwargs: Call({"user": {"emailAddress": "personal@example.com"}}))
    target = colab_drive.SharedDriveStore(config(), SimpleNamespace(files=lambda: files, about=lambda: about))
    with pytest.raises(ValueError, match="your-drive-account@example.com"):
        target.preflight()
    assert files.reads == []


def test_corrupt_or_ambiguous_upload_does_not_restart_create(tmp_path):
    target, files = store()
    folder = target.create_run_folder("session")
    checkpoint = tmp_path / "epoch.pt"
    checkpoint.write_bytes(b"complete")
    files.corrupt_upload = True
    with pytest.raises(RuntimeError, match="verification"):
        target.upload_new(checkpoint, folder, checkpoint.name)
    files.upload_raises = True
    with pytest.raises(RuntimeError, match="completion") as error:
        target.upload_new(checkpoint, folder, checkpoint.name)
    assert "provider details" not in str(error.value)
    assert len(files.creates) == 3  # folder + two separate calls, no blind retry


def prepare_queue(monkeypatch, tmp_path):
    run_dir, queue = tmp_path / "run", tmp_path / "queue"
    run_dir.mkdir()
    queue.mkdir()
    monkeypatch.setenv("MDLARC_CHECKPOINT_QUEUE", str(queue))
    monkeypatch.setenv("MDLARC_RUN_DIR", str(run_dir))
    return run_dir, queue


def test_half_written_checkpoint_is_never_published(monkeypatch, tmp_path):
    run_dir, queue = prepare_queue(monkeypatch, tmp_path)
    path = run_dir / "epoch.pt"

    def fail(temporary):
        temporary.write_bytes(b"half written")
        assert not path.exists()
        assert not list(queue.glob("*.json"))
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint_atomically(path, fail)
    assert list(run_dir.iterdir()) == []
    assert list(queue.iterdir()) == []
    save_checkpoint_atomically(path, lambda temporary: temporary.write_bytes(b"complete"))
    assert path.read_bytes() == b"complete"
    assert len(list(queue.glob("*.json"))) == 1
    with pytest.raises(FileExistsError):
        save_checkpoint_atomically(path, lambda temporary: temporary.write_bytes(b"replacement"))
    assert path.read_bytes() == b"complete"


def test_rolling_checkpoint_overwrites_in_place_and_is_never_queued(monkeypatch, tmp_path):
    run_dir, queue = prepare_queue(monkeypatch, tmp_path)
    path = run_dir / "pretrain/tiny.rolling.pt"
    save_rolling_checkpoint(path, lambda temporary: temporary.write_bytes(b"epoch 10"))
    save_rolling_checkpoint(path, lambda temporary: temporary.write_bytes(b"epoch 20"))
    assert path.read_bytes() == b"epoch 20"
    assert list(queue.iterdir()) == []
    assert [item.name for item in path.parent.iterdir()] == ["tiny.rolling.pt"]
    with pytest.raises(ValueError, match="rolling"):
        save_rolling_checkpoint(run_dir / "pretrain/tiny.epoch050.pt", lambda temporary: temporary.write_bytes(b"x"))


def test_backup_uploads_before_training_stops_and_drains(monkeypatch, tmp_path):
    run_dir, queue = prepare_queue(monkeypatch, tmp_path)
    stop = tmp_path / "stop"
    uploading, finish = threading.Event(), threading.Event()

    class BackgroundStore:
        def upload_new(self, path, folder, name):
            uploading.set()
            assert finish.wait(5)
            assert path.read_bytes() == b"complete"
            return {"id": "checkpoint", "size": path.stat().st_size}

    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(colab_drive.worker(BackgroundStore(), "folder", run_dir, queue, stop)))
    thread.start()
    try:
        save_checkpoint_atomically(run_dir / "epoch.pt", lambda temporary: temporary.write_bytes(b"complete"))
        assert uploading.wait(5)
        assert not stop.exists()
        assert thread.is_alive()  # saving returned while the upload is still in progress
    finally:
        stop.touch()
        finish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcome == [0]
    ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
    assert ledger["complete"] and ledger["files"][0]["path"] == "epoch.pt"


def test_worker_records_failure_and_rejects_escaped_entries(tmp_path):
    run_dir, queue = tmp_path / "run", tmp_path / "queue"
    run_dir.mkdir()
    queue.mkdir()
    (queue / "bad.json").write_text(json.dumps({"path": "../personal.pt", "size": 1}))
    stop = tmp_path / "stop"
    stop.touch()
    forbidden = SimpleNamespace(upload_new=lambda *args: pytest.fail("Should never upload"))
    assert colab_drive.worker(forbidden, "folder", run_dir, queue, stop) == 1
    ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
    assert not ledger["complete"] and len(ledger["errors"]) == 1


@pytest.mark.parametrize("publish", [False, True])
def test_drive_result_archive_contains_only_small_records(monkeypatch, tmp_path, publish):
    root = tmp_path / "source"
    run_dir = root / "runs" / "experiment"
    run_dir.mkdir(parents=True)
    (run_dir / "tiny.pt").write_bytes(b"checkpoint must stay remote")
    (run_dir / "console.log").write_text("training output")
    (run_dir / "accuracy.json").write_text('{"accuracy": 0.5}')
    status_file = tmp_path / "status.json"
    monkeypatch.setattr(colab_remote, "ROOT", root)
    monkeypatch.setattr(colab_remote, "RESULT_ARCHIVE", tmp_path / "results.tar.gz")
    monkeypatch.setattr(colab_remote, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setenv("MDLARC_DRIVE", "1")
    # Package paths normally live in /content; redirect only those temporary files.
    original_path = colab_remote.Path

    def redirected_path(*args):
        if args == ("/content/mdlARC-run-status.json",):
            return status_file
        if len(args) == 2 and args[0] == "/content":
            return tmp_path / args[1]
        return original_path(*args)

    monkeypatch.setattr(colab_remote, "Path", redirected_path)
    manifest = colab_remote.package_results("experiment", {"ok": True}, publish=publish)
    assert colab_remote.MANIFEST.exists() is publish
    colab_remote.publish_results_manifest(manifest)
    assert json.loads(colab_remote.MANIFEST.read_text())["finalized"] is True
    assert not colab_remote.MANIFEST.with_suffix(".json.tmp").exists()
    with tarfile.open(colab_remote.RESULT_ARCHIVE) as archive:
        assert sorted(archive.getnames()) == ["run_status.json", "runs/experiment/accuracy.json", "runs/experiment/console.log"]


def test_drive_snapshot_excludes_wheel_and_credentials(monkeypatch, tmp_path):
    for name in ("launch_run.py", "requirements.txt", "specs/test.json", "scripts/colab_drive.py",
                 "scripts/colab_drive.json", "scripts/requirements_colab_drive.txt", "src/model.py",
                 "scripts/colab_remote.py", "scripts/run_on_colab.py", "scripts/colab_run_docs.py",
                 "scripts/materialize_sweep_spec.py", "scripts/colab_connection.py",
                 "scripts/run_layout.py", "scripts/colab_accounts.py", "scripts/run_records.py", "scripts/colab_reuse.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "environment.json").write_text(json.dumps({"wheel": "test.whl", "sha256": "0" * 64}))
    (wheels / "test.whl").write_bytes(b"do not upload")
    (tmp_path / ".token.json").write_text("private")
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    archive = tmp_path / "source.tar.gz"
    run_on_colab.create_snapshot(tmp_path / "specs/test.json", archive, drive=True)
    with tarfile.open(archive) as source:
        names = source.getnames()
        assert "wheels/environment.json" in names
        assert "wheels/test.whl" not in names
        assert ".token.json" not in names


def test_picker_is_limited_to_the_configured_folder_and_assets(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from setup_colab_drive import picker_parameters
    parameters = picker_parameters(config())
    assert parameters["trigger_onepick"] == "true"
    assert parameters["include_granted_scopes"] == "false"
    assert set(parameters["file_ids"].split(",")) == {colab_drive.ROOT_FOLDER_ID, "wheel", "environment", "wheel_folder", "datasets", "task-variations"}


def test_actual_broad_google_token_is_rejected_even_with_narrow_json(monkeypatch, tmp_path):
    from io import BytesIO
    from google.oauth2.credentials import Credentials
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"type": "authorized_user", "scopes": [colab_drive.DRIVE_SCOPE],
                                 "client_id": "client", "client_secret": "secret", "refresh_token": "refresh"}))
    credentials = SimpleNamespace(token="access", refresh=lambda request: None)
    monkeypatch.setattr(Credentials, "from_authorized_user_info", lambda *args, **kwargs: credentials)
    monkeypatch.setattr(colab_drive, "urlopen", lambda *args, **kwargs: BytesIO(json.dumps({
        "scope": colab_drive.DRIVE_SCOPE + " https://www.googleapis.com/auth/drive",
    }).encode()))
    with pytest.raises(ValueError, match="broader"):
        colab_drive.build_service(token)


@pytest.mark.parametrize("corrupt", [False, True])
def test_wheel_download_checks_environment_and_hash(monkeypatch, tmp_path, corrupt):
    from googleapiclient import http
    target, files = store()
    wheel = tmp_path / "test.whl"
    wheel_bytes = b"reviewed wheel"
    manifest = {"wheel": wheel.name, "sha256": hashlib.sha256(wheel_bytes).hexdigest()}
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps(manifest))
    files.items["wheel"]["name"] = wheel.name
    media = {"environment": json.dumps(manifest).encode(), "wheel": b"corrupt" if corrupt else wheel_bytes}
    files.get_media = lambda fileId: media[fileId]

    class Download:
        def __init__(self, output, content, **kwargs):
            self.output, self.content = output, content

        def next_chunk(self, **kwargs):
            self.output.write(self.content)
            return None, True

    monkeypatch.setattr(http, "MediaIoBaseDownload", Download)
    target.preflight()
    if corrupt:
        with pytest.raises(ValueError, match="SHA-256"):
            target.download_wheel(wheel, environment)
        assert not wheel.exists()
        assert not list(tmp_path.glob("*.partial"))
    else:
        target.download_wheel(wheel, environment)
        assert wheel.read_bytes() == wheel_bytes


@pytest.fixture
def shared_launcher(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    spec = root / "specs/test.json"
    spec.parent.mkdir(parents=True)
    spec.write_text('{"run_name":"test"}')
    commands = []
    monkeypatch.setattr(run_on_colab, "ROOT", root)
    monkeypatch.setattr(run_on_colab, "run_command", lambda command, **kwargs: commands.append(command))
    def watch(colab, session, run_name, *, command):
        run_on_colab.run_command(command)
        return {"finalized": True}
    monkeypatch.setattr(run_on_colab, "watch_remote_run", watch)
    monkeypatch.setattr(run_on_colab.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(run_on_colab, "require_unused_remote_workspace", lambda *args: None)
    monkeypatch.setattr(run_on_colab, "create_snapshot", lambda *args, **kwargs: (1, 100))
    monkeypatch.setattr(run_on_colab, "preserve_launch_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_on_colab.shutil, "which", lambda executable: "colab")
    monkeypatch.setattr(run_on_colab.colab_connection, "cli_python", lambda _: sys.executable)
    monkeypatch.setattr(sys, "argv", ["run_on_colab.py", "--config", "specs/test.json",
                                   "--task-variations", "--session", "new-session",
                                   "--intent", "Test run", "--changed-from-previous", "No changes"])
    return root, commands


def test_standalone_temp_spec_launches_fresh_training(monkeypatch, shared_launcher):
    root, commands = shared_launcher
    selected = root / "specs/temp specs/fresh.json"
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps({"run_name": "test", "pretrain": {"epochs": 672, "checkpoint_epochs": [336]}}))
    command_line = list(sys.argv)
    command_line[command_line.index("--config") + 1] = str(selected)
    monkeypatch.setattr(sys, "argv", [*command_line, "--standalone-spec"])
    monkeypatch.setattr(run_on_colab, "download_results", lambda *args: {"ok": True, "backups_complete": True})
    assert run_on_colab.main() == 0
    execution = next(command for command in commands if "--file" in command)
    assert "MDLARC_PRETRAIN_DRIVE_ID=" in execution
    assert not (root / "results" / "test-branch" / "colab_runs/new-session/trial.json").exists()


@pytest.mark.parametrize("spec_path", ["specs/test.json", "specs/temp specs/continuation.json"])
@pytest.mark.parametrize("mount", [None, "/content/drive/MyDrive/mdlARC/runs"])
def test_drive_checkpoint_launcher_uses_spec_without_resume_flags(monkeypatch, shared_launcher, spec_path, mount):
    root, commands = shared_launcher
    selected = root / spec_path
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(json.dumps({
        "run_name": "test", "checkpoint": {
            "drive_id": "checkpoint", "name": "pretrain__tiny.epoch067.pt",
            "size": 609386904, "sha256": "drive",
        },
    }))
    command_line = list(sys.argv)
    command_line[command_line.index("--config") + 1] = spec_path
    if mount:
        command_line += ["--checkpoint-mount", mount]
    monkeypatch.setattr(sys, "argv", command_line)
    monkeypatch.setattr(run_on_colab, "download_results", lambda *args: {"ok": True, "backups_complete": True})
    assert run_on_colab.main() == 0
    execution = next(command for command in commands if "--file" in command)
    assert not any("RESUME_PRETRAIN" in argument for argument in execution)
    assert "MDLARC_PRETRAIN_DRIVE_ID=" in execution
    assert "MDLARC_FINETUNE_TARGET=" in execution
    assert f"MDLARC_CHECKPOINT_MOUNT={mount or ''}" in execution


@pytest.mark.parametrize("mount", ["relative/runs", "/content/../runs", "/content/runs\nother"])
def test_checkpoint_mount_rejects_unsafe_remote_paths(monkeypatch, shared_launcher, mount):
    root, commands = shared_launcher
    (root / "specs/test.json").write_text(json.dumps({"checkpoint": {
        "drive_id": "checkpoint", "name": "pretrain__tiny.pt", "size": 123, "sha256": "drive"}}))
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--checkpoint-mount", mount])
    with pytest.raises(SystemExit):
        run_on_colab.main()
    assert commands == []


def test_checkpoint_mount_requires_drive_input(monkeypatch, shared_launcher):
    root, commands = shared_launcher
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--checkpoint-mount", "/content/drive/MyDrive/mdlARC/runs"])
    with pytest.raises(SystemExit):
        run_on_colab.main()
    assert commands == []


@pytest.mark.parametrize("flags", [
    ["--pretrain-drive-id", "another-file"], ["--finetune-target", "heldout"],
    ["--pretrain-size", "123"], ["--local-artifacts"],
])
def test_drive_checkpoint_spec_rejects_conflicting_cli_flags(monkeypatch, shared_launcher, flags):
    root, commands = shared_launcher
    (root / "specs/test.json").write_text(json.dumps({
        "run_name": "test", "checkpoint": {
            "drive_id": "checkpoint", "name": "saved.pt", "size": 123, "sha256": "drive",
        },
    }))
    monkeypatch.setattr(sys, "argv", [*sys.argv, *flags])
    with pytest.raises(SystemExit):
        run_on_colab.main()
    assert commands == []


def test_legacy_finetune_checkpoint_flags_still_work(monkeypatch, shared_launcher):
    root, commands = shared_launcher
    (root / "specs/test.json").write_text(json.dumps({"run_name": "test", "finetunes": {"heldout": {}}}))
    monkeypatch.setattr(sys, "argv", [*sys.argv,
        "--pretrain-drive-id", "checkpoint", "--pretrain-name", "saved.pt",
        "--pretrain-size", "123", "--pretrain-sha256", "a" * 64, "--finetune-target", "heldout"])
    monkeypatch.setattr(run_on_colab, "download_results", lambda *args: {"ok": True, "backups_complete": True})
    assert run_on_colab.main() == 0
    execution = next(command for command in commands if "--file" in command)
    assert "MDLARC_PRETRAIN_DRIVE_ID=checkpoint" in execution
    assert "MDLARC_FINETUNE_TARGET=heldout" in execution


@pytest.mark.parametrize("final_backed_up", [False, True])
@pytest.mark.parametrize("summary_upload_failed", [False, True])
@pytest.mark.parametrize("mount", [None, "/content/drive/MyDrive/mdlARC/runs"])
def test_remote_resume_stages_separate_input_and_requires_final_backup(
        monkeypatch, tmp_path, final_backed_up, summary_upload_failed, mount):
    import torch
    root = tmp_path / "repo"
    run_dir = root / "runs/resume"
    (root / "specs").mkdir(parents=True)
    selected_spec = {"run_name": "resume", "checkpoint": {
        "drive_id": "checkpoint", "name": "pretrain__tiny.epoch067.pt",
        "size": 609386904, "sha256": "drive",
    }}
    (root / "specs/resume.json").write_text(json.dumps(selected_spec))
    (root / "wheels").mkdir()
    (root / "wheels/environment.json").write_text('{"wheel":"test.whl"}')
    staged = []
    uploaded = []

    class Store:
        run_parent_id = "shared-runs"

        def preflight(self, **kwargs):
            pass

        def download_input_checkpoint(self, file_id, path, **kwargs):
            assert file_id == selected_spec["checkpoint"]["drive_id"]
            assert kwargs["expected_name"] == selected_spec["checkpoint"]["name"]
            assert kwargs["expected_size"] == selected_spec["checkpoint"]["size"]
            assert kwargs["expected_sha256"] is None
            assert kwargs.get("mounted_runs_root") == (Path(mount) if mount else None)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"old checkpoint")
            staged.append(path)
            return {"file_id": file_id}

        def create_run_folder(self, session):
            return "new-folder"

        def download_wheel(self, *args):
            pass

        def upload_new(self, path, folder, name):
            uploaded.append(name)
            if name == "results-without-checkpoints.tar.gz" and summary_upload_failed:
                raise RuntimeError("summary upload did not verify")
            return {"id": name}

    class Worker:
        def __init__(self, *args, **kwargs):
            (run_dir / "checkpoint_backups.json").write_text('{"complete":false,"files":[]}')

        def poll(self):
            return None

        def wait(self, **kwargs):
            (run_dir / "checkpoint_backups.json").write_text(json.dumps({
                "complete": True, "files": [{"path": "pretrain/tiny.pt"}] if final_backed_up else [],
            }))
            return 0

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "launch_run.py" in command:
            assert staged == [run_dir / "inputs/pretrain.pt"]
            assert not (run_dir / "pretrain/tiny.pt").exists()
            output = run_dir / "pretrain/tiny.pt"
            output.parent.mkdir()
            output.write_bytes(b"new final checkpoint")

    monkeypatch.setattr(colab_remote, "ROOT", root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(colab_remote, "Path", lambda path: tmp_path / "control" if path == "/content/.mdlarc-drive-control" else Path(path))
    monkeypatch.setattr(colab_remote, "unpack_source", lambda: None)
    from scripts import run_records
    monkeypatch.setattr(run_records, "publish_bundle", lambda *args: {"id": "launch-records"})
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"source")
    monkeypatch.setattr(colab_remote, "ARCHIVE", archive)
    monkeypatch.setattr(colab_remote, "run", run)
    monkeypatch.setattr(colab_remote, "flash_attn_wheel", lambda torch: root / "wheels/test.whl")
    monkeypatch.setattr(colab_remote.subprocess, "Popen", Worker)
    monkeypatch.setitem(sys.modules, "colab_drive", colab_drive)
    monkeypatch.setattr(colab_drive, "load_config", lambda path: config())
    monkeypatch.setattr(colab_drive, "build_shared_service", lambda: None)
    monkeypatch.setattr(colab_drive, "SharedDriveStore", lambda *args: Store())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "A100")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=80 * 1024**3))
    for key, value in {
        "MDLARC_SPEC": "specs/resume.json", "MDLARC_RUN_NAME": "resume",
        "MDLARC_DRIVE": "1", "MDLARC_DRIVE_AUTH": "shared", "MDLARC_SESSION": "new-session",
        "MDLARC_PRETRAIN_DRIVE_ID": "", "MDLARC_TASK_VARIATIONS": "0",
        "MDLARC_FINETUNE_TARGET": "", "MDLARC_MIN_GPU_MEMORY_GB": "75",
        "MDLARC_CHECKPOINT_MOUNT": mount or "",
    }.items():
        monkeypatch.setenv(key, value)
    statuses = []
    def package(name, status, *, publish):
        assert publish is False
        statuses.append(dict(status))
        return {"status": dict(status)}
    def publish(manifest):
        assert "results-without-checkpoints.tar.gz" in uploaded
        assert manifest["status"] == statuses[-1]
    monkeypatch.setattr(colab_remote, "package_results", package)
    monkeypatch.setattr(colab_remote, "publish_results_manifest", publish)
    colab_remote.main()
    assert statuses[-1]["ok"] is (final_backed_up and not summary_upload_failed)
    assert statuses[-1]["backups_complete"] is (final_backed_up and not summary_upload_failed)
    details = json.loads((run_dir / "experiment.json").read_text())
    assert details["spec"] == selected_spec
    assert details["input_checkpoint"]["staged_path"] == "inputs/pretrain.pt"
    assert [command for command in commands if "launch_run.py" in command] == [
        [sys.executable, "-u", "launch_run.py", "--config", "specs/resume.json"],
    ]
    assert "pretrain__tiny.epoch067.pt" not in uploaded


@pytest.mark.parametrize("stop_after_success", [False, True])
@pytest.mark.parametrize("ok,backups_complete,expected_exit", [
    (True, True, 0), (False, True, 1), (False, False, 1),
    (True, False, 1), (True, None, 1),
])
def test_shared_launcher_shutdown_requires_opt_in_and_verified_success(
        monkeypatch, shared_launcher, ok, backups_complete, expected_exit, stop_after_success):
    root, commands = shared_launcher
    status = {"ok": ok, "error": "run failed" if not ok else None}
    if backups_complete is not None:
        status["backups_complete"] = backups_complete

    def recover(*args):
        commands.append(["recovered-results"])
        return status

    monkeypatch.setattr(run_on_colab, "download_results", recover)
    if stop_after_success:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--stop-after-success"])
    assert run_on_colab.main() == expected_exit
    assert any("MDLARC_TASK_VARIATIONS=1" in command for command in commands)
    stops = [command for command in commands if command[1:2] == ["stop"]]
    should_stop = stop_after_success and ok and backups_complete is True
    assert stops == ([["colab", "stop", "-s", "new-session"]] if should_stop else [])
    if should_stop:
        assert commands.index(["recovered-results"]) < commands.index(stops[0])
    assert not any(command[1:2] == ["new"] for command in commands)
    assert len([command for command in commands if command[1:2] == ["upload"]]) == 1
    assert any("MDLARC_DRIVE_AUTH=shared" in command for command in commands)
    journal = json.loads((root / "results" / "test-branch" / "colab_runs/_active/new-session.json").read_text())
    assert journal["stop_after_success"] is stop_after_success
    assert journal["session_disposition"] == (
        "stopped" if should_stop else "retained for recovery" if stop_after_success else "user-managed"
    )


@pytest.mark.parametrize("failure_step", ["exec", "download"])
def test_shared_auto_shutdown_keeps_vm_after_execution_or_recovery_error(
        monkeypatch, shared_launcher, failure_step):
    root, commands = shared_launcher
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--stop-after-success"])

    def command(command, **kwargs):
        commands.append(command)
        if command[1:2] == [failure_step] or (failure_step == "exec" and "--file" in command):
            raise RuntimeError("connection lost")

    def recover(*args):
        command(["colab", "download"])
        return {"ok": True, "backups_complete": True}

    monkeypatch.setattr(run_on_colab, "run_command", command)
    monkeypatch.setattr(run_on_colab, "download_results", recover)
    with pytest.raises(RuntimeError, match="connection lost"):
        run_on_colab.main()
    assert not any(command[1:2] == ["stop"] for command in commands)
    journal = json.loads((root / "results" / "test-branch" / "colab_runs/_active/new-session.json").read_text())
    assert journal["session_disposition"] == "retained for recovery"


def test_shared_auto_shutdown_reports_stop_failure(monkeypatch, shared_launcher):
    root, commands = shared_launcher
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--stop-after-success"])
    monkeypatch.setattr(run_on_colab, "download_results", lambda *args: {
        "ok": True, "backups_complete": True,
    })

    def command(command, **kwargs):
        commands.append(command)
        if command[1:2] == ["stop"]:
            raise RuntimeError("stop unavailable")

    monkeypatch.setattr(run_on_colab, "run_command", command)
    with pytest.raises(RuntimeError, match="Automatic shutdown or record backup failed"):
        run_on_colab.main()
    assert commands[-1] == ["colab", "stop", "-s", "new-session"]
    journal = json.loads((root / "results" / "test-branch" / "colab_runs/_active/new-session.json").read_text())
    assert journal["state"] == "completed"
    assert journal["session_disposition"] == "stop_failed"


def test_record_backup_failure_prevents_shutdown(monkeypatch, shared_launcher):
    root, commands = shared_launcher
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--stop-after-success"])
    monkeypatch.setattr(run_on_colab, "download_results", lambda *args: {
        "ok": True, "backups_complete": True, "drive_folder_id": "run-folder",
    })
    def failed_backup(*args):
        raise RuntimeError("records did not verify")
    monkeypatch.setattr(run_on_colab, "sync_records_to_drive", failed_backup)
    with pytest.raises(RuntimeError, match="record backup failed"):
        run_on_colab.main()
    assert not any(command[1:2] == ["stop"] for command in commands)
    journal = json.loads((root / "results/test-branch/colab_runs/_active/new-session.json").read_text())
    assert journal["state"] == "needs_recovery"
    assert "records did not verify" in journal["records_backup_error"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_shared_store_routes_runs_into_exact_branch(monkeypatch, duplicate):
    files = FakeFiles()
    root = colab_drive.ROOT_FOLDER_ID
    files.items["shared-runs"] = {"id": "shared-runs", "name": "runs",
        "mimeType": colab_drive.FOLDER_MIME, "parents": [root]}
    files.items["branch"] = {"id": "branch", "name": "test-branch",
        "mimeType": colab_drive.FOLDER_MIME, "parents": ["shared-runs"],
        "capabilities": {"canAddChildren": True}}
    listed = [{"id": "branch", "name": "test-branch"}]
    if duplicate:
        listed.append({"id": "duplicate", "name": "test-branch"})
    queries = []
    def listing(**kwargs):
        queries.append(kwargs["q"])
        return Call({"files": listed})
    files.list = listing
    about = SimpleNamespace(get=lambda **kwargs: Call({"user": {"emailAddress": "your-drive-account@example.com"}}))
    target = colab_drive.SharedDriveStore(config(), SimpleNamespace(files=lambda: files, about=lambda: about))
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate Drive branch"):
            target.preflight()
    else:
        target.preflight()
        target.create_run_folder("new-session")
        assert target.run_parent_id == "branch"
        assert files.creates[-1]["parents"] == ["branch"]
    assert "'shared-runs' in parents" in queries[0]


def test_shared_launcher_rejects_a_vm_with_existing_run_files(monkeypatch):
    calls = []

    def list_content(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="sample_data/\nmdlARC-source.tar.gz\n", stderr="")

    monkeypatch.setattr(run_on_colab.subprocess, "run", list_content)
    with pytest.raises(RuntimeError, match="already contains mdlARC run files"):
        run_on_colab.require_unused_remote_workspace("colab", "existing-session")
    assert calls == [["colab", "ls", "-s", "existing-session", "/content"]]


def test_console_capture_preserves_successful_subprocess_exit(tmp_path):
    log = tmp_path / "console.log"
    colab_remote.run([sys.executable, "-c", "print('completed')"], log_path=log)
    assert log.read_text() == "completed\n"


def test_silent_training_is_stopped_if_backup_process_dies():
    import time
    started = time.monotonic()

    def failed_uploader():
        if time.monotonic() - started > 0.2:
            raise RuntimeError("uploader died")

    with pytest.raises(RuntimeError, match="uploader died"):
        colab_remote.run([sys.executable, "-c", "import time; time.sleep(60)"], health_check=failed_uploader)
    assert time.monotonic() - started < 5
