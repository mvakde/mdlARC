"""Cross-laptop behavior and preservation of research records during failures."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from scripts import sync_records as sync

NOTE = "results/codex%2Fexample/experiments/notes/run.md"
SPEC = "specs/temp specs/next run.json"


class SharedDrive:
    def __init__(self):
        self.bundles = []

    def read_snapshots(self, *, dry_run=False):
        return [sync.unpack(data) for data in self.bundles]

    def publish(self, revision, blob):
        assert sync.unpack(blob)[0]["revision"] == revision
        self.bundles.append(blob)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_new_laptop_restores_records_and_offline_independent_edits_merge(tmp_path):
    air, pro = tmp_path / "air", tmp_path / "pro"
    drive = SharedDrive()
    write(air, NOTE, "initial intent")
    write(air, SPEC, '{"run_name":"next"}')
    assert sync.synchronize(air, drive)["upload"] == 2
    assert sync.synchronize(pro, drive)["download"] == 2
    write(air, NOTE, "updated intent")
    write(pro, SPEC, '{"run_name":"next","seed":43}')
    sync.synchronize(air, drive)
    result = sync.synchronize(pro, drive)
    assert (result["upload"], result["download"]) == (1, 1)
    sync.synchronize(air, drive)
    assert (pro / NOTE).read_text() == "updated intent"
    assert (air / SPEC).read_text() == (pro / SPEC).read_text()
    assert sync.synchronize(air, drive)["upload"] == 0
    assert len(drive.bundles) == 3


def test_conflicting_offline_edits_preserve_both_and_require_review(tmp_path):
    air, pro = tmp_path / "air", tmp_path / "pro"
    drive = SharedDrive()
    write(air, NOTE, "initial")
    sync.synchronize(air, drive)
    sync.synchronize(pro, drive)
    write(air, NOTE, "air interpretation")
    write(pro, NOTE, "pro interpretation")
    sync.synchronize(air, drive)
    assert sync.synchronize(pro, drive, dry_run=True)["conflicts"] == [NOTE]
    with pytest.raises(sync.SyncError, match="nothing was uploaded or replaced"):
        sync.synchronize(pro, drive)
    assert (pro / NOTE).read_text() == "pro interpretation"
    assert len(drive.bundles) == 2
    copies = list((pro / sync.CONTROL / "conflicts").rglob("*.md"))
    assert [p.read_text() for p in copies] == ["air interpretation"]
    write(pro, NOTE, "combined interpretation")
    sync.synchronize(pro, drive, resolutions=[NOTE])
    sync.synchronize(air, drive)
    assert (air / NOTE).read_text() == "combined interpretation"


def test_simultaneous_uploads_never_replace_a_remote_revision(tmp_path):
    air, pro = tmp_path / "air", tmp_path / "pro"
    drive = SharedDrive()
    write(air, NOTE, "base")
    sync.synchronize(air, drive)
    sync.synchronize(pro, drive)
    parent = sync.unpack(drive.bundles[0])[0]["revision"]
    # Both writers observed the same parent before either upload became visible.
    for text in (b"from air", b"from pro"):
        _, blob = sync.pack({NOTE: text}, {NOTE: [parent]})
        drive.bundles.append(blob)
    with pytest.raises(sync.SyncError, match="Conflicting edits"):
        sync.synchronize(air, drive)
    assert len(drive.bundles) == 3
    write(air, NOTE, "reviewed merge")
    sync.synchronize(air, drive, resolutions=[NOTE])
    sync.synchronize(pro, drive)
    assert (pro / NOTE).read_text() == "reviewed merge"


def test_first_sync_will_not_overwrite_an_unrecognized_local_note(tmp_path):
    drive = SharedDrive()
    write(tmp_path / "air", NOTE, "on Drive")
    sync.synchronize(tmp_path / "air", drive)
    write(tmp_path / "pro", NOTE, "local unsynced draft")
    with pytest.raises(sync.SyncError, match="Conflicting edits"):
        sync.synchronize(tmp_path / "pro", drive)
    assert (tmp_path / "pro" / NOTE).read_text() == "local unsynced draft"


def test_identical_existing_copies_adopt_drive_history_and_missing_files_restore(tmp_path):
    drive = SharedDrive()
    for name in ("air", "pro"):
        write(tmp_path / name, NOTE, "same")
    sync.synchronize(tmp_path / "air", drive)
    assert sync.synchronize(tmp_path / "pro", drive)["upload"] == 0
    (tmp_path / "pro" / NOTE).unlink()
    assert sync.synchronize(tmp_path / "pro", drive)["download"] == 1
    assert (tmp_path / "pro" / NOTE).read_text() == "same"
    assert len(drive.bundles) == 1


def test_dry_run_changes_no_notes_or_cloud_history(tmp_path):
    drive = SharedDrive()
    write(tmp_path / "air", NOTE, "on Drive")
    sync.synchronize(tmp_path / "air", drive)
    pro = tmp_path / "pro"
    write(pro, SPEC, '{"run_name":"draft"}')
    result = sync.synchronize(pro, drive, dry_run=True)
    assert (result["upload"], result["download"]) == (1, 1)
    assert not (pro / NOTE).exists()
    assert not (pro / sync.CONTROL / "baseline.json").exists()
    assert len(drive.bundles) == 1


def test_all_branches_share_storage_but_keep_original_paths(monkeypatch, tmp_path):
    other = NOTE.replace("codex%2Fexample", "newreason")
    write(tmp_path, NOTE, "branch one")
    write(tmp_path, other, "branch two")
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "unrelated-branch")
    records = sync.collect(tmp_path)
    assert records[NOTE] == b"branch one"
    assert records[other] == b"branch two"


def test_bulk_data_baselines_tokens_and_live_identities_are_not_selected(tmp_path):
    keep = [NOTE, SPEC,
            "results/codex/old-layout/experiments/JOURNAL.md",
            "results/main/colab_runs/session/spec.json",
            "results/main/colab_runs/session/runs/job/evals/final/score.json"]
    omit = ["results/baselines.md", "results/main/colab_runs/_active/session.connection.json",
            "results/main/experiments/data-audits/challenges.json",
            "results/main/experiments/trials/workspace/spec.json",
            "results/main/colab_runs/session/source.tar.gz",
            "results/main/colab_runs/session/checkpoint.pt",
            "results/main/colab_runs/session/console.log",
            "results/_shared/record_sync/baseline.json",
            "scripts/colab_oauth/private/token.json"]
    for name in keep + omit:
        write(tmp_path, name, "{}" if name.endswith(".json") else "record")
    assert set(sync.collect(tmp_path)) == set(keep)


def test_credentials_in_an_allowed_record_abort_without_exposing_the_value(tmp_path):
    secret = "a-sensitive-refresh-token"
    write(tmp_path, SPEC, json.dumps({"nested": {"refresh_token": secret}}))
    drive = SharedDrive()
    with pytest.raises(sync.SyncError) as error:
        sync.synchronize(tmp_path, drive)
    assert secret not in str(error.value)
    assert drive.bundles == []


def test_credential_assignment_in_markdown_is_rejected(tmp_path):
    write(tmp_path, NOTE, 'refresh_token="a-sensitive-refresh-token"')
    with pytest.raises(sync.SyncError, match="Credential-like"):
        sync.collect(tmp_path)


def test_symlink_destination_is_never_followed(tmp_path):
    drive = SharedDrive()
    write(tmp_path / "air", NOTE, "on Drive")
    sync.synchronize(tmp_path / "air", drive)
    outside = write(tmp_path, "outside.txt", "keep me")
    dest = tmp_path / "pro" / NOTE
    dest.parent.mkdir(parents=True)
    dest.symlink_to(outside)
    with pytest.raises(sync.SyncError, match="linked"):
        sync.synchronize(tmp_path / "pro", drive)
    assert outside.read_text() == "keep me"


def test_damaged_payload_and_unsafe_archive_entries_are_rejected():
    _, blob = sync.pack({NOTE: b"original"}, {})
    expanded = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as old:
        with tarfile.open(fileobj=expanded, mode="w:gz") as new:
            for member in old.getmembers():
                data = old.extractfile(member).read()
                if member.name.startswith("files/"):
                    data = b"modified"
                    member.size = len(data)
                new.addfile(member, io.BytesIO(data))
    with pytest.raises(sync.SyncError, match="damaged"):
        sync.unpack(expanded.getvalue())
    assert not sync.selected("specs/temp specs/../../token.json")
    assert not sync.selected("/specs/temp specs/run.json")


def test_missing_history_is_unknown_and_does_not_touch_local_notes(tmp_path):
    drive = SharedDrive()
    _, blob = sync.pack({NOTE: b"new"}, {NOTE: ["a" * 32]})
    drive.bundles.append(blob)
    write(tmp_path, NOTE, "local")
    with pytest.raises(sync.SyncError, match="incomplete"):
        sync.synchronize(tmp_path, drive)
    assert (tmp_path / NOTE).read_text() == "local"


def test_local_changes_during_a_cloud_read_are_preserved(tmp_path):
    drive = SharedDrive()
    write(tmp_path / "air", NOTE, "on Drive")
    sync.synchronize(tmp_path / "air", drive)
    pro = tmp_path / "pro"
    write(pro, SPEC, '{"run_name":"before"}')
    original = drive.read_snapshots
    def changing(**kwargs):
        write(pro, SPEC, '{"run_name":"after"}')
        return original(**kwargs)
    drive.read_snapshots = changing
    with pytest.raises(sync.SyncError, match="changed during sync"):
        sync.synchronize(pro, drive)
    assert (pro / SPEC).read_text() == '{"run_name":"after"}'
    assert not (pro / NOTE).exists()


def test_installed_copy_operates_without_source_scripts(tmp_path, monkeypatch):
    environment = tmp_path / "venv"
    executable = environment / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "prefix", str(environment))
    sync.install()
    write(tmp_path, NOTE, "research note")
    result = subprocess.run([str(environment / "bin/mdlarc-records"), "--root", str(tmp_path), "inventory"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["records"] == 1


def test_upload_with_lost_response_reconciles_same_id_and_checks_readback(tmp_path):
    from googleapiclient.errors import HttpError
    calls, saved = [], {}
    def get(fileId, **kwargs):
        def execute(**kwargs):
            if fileId not in saved:
                raise HttpError(SimpleNamespace(status=404, reason="missing"), b"{}")
            return saved[fileId]
        return SimpleNamespace(execute=execute)
    def create(body, media_body, **kwargs):
        def execute(**kwargs):
            calls.append(body["id"])
            data = media_body.getbytes(0, media_body.size())
            saved[body["id"]] = {**body, "size": str(len(data)),
                                  "md5Checksum": sync.hashlib.md5(data).hexdigest(), "data": data}
            raise OSError("response was lost after storage")
        return SimpleNamespace(execute=execute)
    api = SimpleNamespace(files=lambda: SimpleNamespace(get=get, create=create,
        get_media=lambda fileId: SimpleNamespace(execute=lambda **kw: saved[fileId]["data"])))
    store = object.__new__(sync.DriveStore)
    store.api, store.control = api, tmp_path
    store.get_folder = lambda **kw: "shared-folder"
    store.new_id = lambda: "fixed-file-id"
    revision, blob = sync.pack({NOTE: b"important"}, {})
    store.publish(revision, blob)
    assert calls == ["fixed-file-id"]
    assert not (tmp_path / "pending.json").exists()
    assert (tmp_path / "cache/fixed-file-id.tar.gz").read_bytes() == blob
