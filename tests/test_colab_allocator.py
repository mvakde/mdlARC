"""CUDA allocator defaults are operational and honor both override names."""
import pytest

from scripts.colab_remote import configure_cuda_allocator


@pytest.mark.parametrize("key", [None, "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_colab_allocator_default_preserves_explicit_override(monkeypatch, key):
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.setenv(name, "")
    if key:
        monkeypatch.setenv(key, "max_split_size_mb:128")
    configure_cuda_allocator()
    if key:
        import os
        assert os.environ[key] == "max_split_size_mb:128"
        if key == "PYTORCH_CUDA_ALLOC_CONF":
            assert not os.environ.get("PYTORCH_ALLOC_CONF")
    else:
        import os
        assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
