"""Streamed Drive ranges retain exact boundaries and bounded read recovery."""
import io
import hashlib
from types import SimpleNamespace

import pytest
import requests

from workflows.vast import drive


@pytest.mark.parametrize("corrupt", [False, True])
def test_large_download_checks_hash_before_publish(tmp_path, monkeypatch, corrupt):
    content = b"verified checkpoint"
    store = object.__new__(drive.Drive)
    store.metadata = lambda identity: {
        "id": identity, "name": "input.pt", "size": str(len(content)),
        "mimeType": "application/octet-stream", "md5Checksum": hashlib.md5(content).hexdigest(),
    }
    store.charge = lambda direction, size: None
    store._download_checkpoint_ranges = lambda identity, target, size: target.write(
        b"x" * len(content) if corrupt else content)
    monkeypatch.setattr(drive, "CHECKPOINT_API_RANGE_BYTES", 1)
    destination = tmp_path / "checkpoint.pt"
    if corrupt:
        with pytest.raises(ValueError, match="verification failed"):
            store.download("id", destination, expected={"sha256": hashlib.sha256(content).hexdigest()})
        assert not destination.exists()
        assert not destination.with_name("checkpoint.pt.partial").exists()
    else:
        store.download("id", destination, expected={"sha256": hashlib.sha256(content).hexdigest()})
        assert destination.read_bytes() == content


@pytest.mark.parametrize("failure", [None, "disconnect", "http503", "ignored", "wrong_range", "compressed", "truncated", "overlong"])
def test_checkpoint_range_protocol(monkeypatch, failure):
    content = b"0123456789abcdefg"
    credential = object()
    calls = []
    sleeps = []
    monkeypatch.setattr(drive, "CHECKPOINT_API_RANGE_BYTES", 8)
    monkeypatch.setattr(drive, "CHECKPOINT_API_STREAM_BYTES", 2)
    monkeypatch.setattr(drive.time, "sleep", sleeps.append)

    class Response:
        def __init__(self, first, last):
            self.status_code = 503 if failure == "http503" and len(calls) == 1 else 206
            self.headers = {"Content-Range": f"bytes {first}-{last}/{len(content)}"}
            self.data = content[first:last + 1]
            if failure == "ignored":
                self.status_code = 200
            elif failure == "wrong_range":
                self.headers["Content-Range"] = "bytes 1-8/17"
            elif failure == "compressed":
                self.headers["Content-Encoding"] = "gzip"
            elif failure == "truncated":
                self.data = self.data[:-1]
            elif failure == "overlong":
                self.data += b"x"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            raise requests.HTTPError("private provider response")

        def iter_content(self, chunk_size):
            assert chunk_size == 2
            for pos in range(0, len(self.data), chunk_size):
                yield self.data[pos:pos + chunk_size]
                if failure == "disconnect" and len(calls) == 1:
                    raise requests.ConnectionError("private provider response")

    class Session:
        def __init__(self, auth, refresh_timeout):
            assert auth is credential and refresh_timeout == 30

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, *, headers, stream, timeout):
            assert url == "https://www.googleapis.com/drive/v3/files/id?alt=media"
            assert stream and timeout == (15, 120)
            assert headers["Accept-Encoding"] == "identity"
            first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            calls.append((first, last))
            return Response(first, last)

    monkeypatch.setattr("google.auth.transport.requests.AuthorizedSession", Session)
    media = SimpleNamespace(uri="https://www.googleapis.com/drive/v3/files/id?alt=media",
                            http=SimpleNamespace(credentials=credential))
    store = SimpleNamespace(api=SimpleNamespace(files=lambda: SimpleNamespace(get_media=lambda **kwargs: media)))
    target = io.BytesIO()
    if failure in {"ignored", "wrong_range", "compressed", "truncated", "overlong"}:
        with pytest.raises(ValueError):
            drive.Drive._download_checkpoint_ranges(store, "id", target, len(content))
    else:
        drive.Drive._download_checkpoint_ranges(store, "id", target, len(content))
        assert target.getvalue() == content
        if failure == "disconnect":
            assert calls[:2] == [(0, 7), (2, 7)]
        if failure == "http503":
            assert calls[:2] == [(0, 7), (0, 7)]
