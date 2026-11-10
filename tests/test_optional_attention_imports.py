"""An unused optional attention backend must not block another backend."""
import builtins
from pathlib import Path
import runpy


def test_incompatible_cutedsl_does_not_block_fa2_import(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.syspath_prepend(str(source))
    original_import = builtins.__import__

    def import_with_broken_cutedsl(name, *args, **kwargs):
        if name == "flash_attn.cute":
            raise AttributeError("module 'cutlass.cute.core' has no attribute 'ThrMma'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_cutedsl)
    module = runpy.run_path(str(source / "tinytransformer.py"))
    assert module["flash_attn4_varlen_func"] is None
    assert module["TinyTransformerConfig"](FA=2).FA == 2
    assert module["normalize_flash_attention_version"](4) == 4
