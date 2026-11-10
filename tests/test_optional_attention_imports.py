"""An unused optional attention backend must not block another backend."""
import builtins
from pathlib import Path
import runpy
import sys


def test_fa4_is_imported_only_when_selected(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.syspath_prepend(str(source))
    original_import = builtins.__import__
    attempts = []

    def import_with_broken_cutedsl(name, *args, **kwargs):
        if name.startswith("flash_attn.cute"):
            attempts.append(name)
            raise AttributeError("module 'cutlass.cute.core' has no attribute 'ThrMma'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_cutedsl)
    for name in ("attention_runtime", "tinytransformer"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = runpy.run_path(str(source / "tinytransformer.py"))
    assert attempts == []  # importing the model never touches FA4
    config = module["TinyTransformerConfig"](FA=2, d_model=16, n_heads=2, d_ff=32, n_layers=1,
                                             num_examples=1, num_task_embedding_latents=1)
    assert config.FA == 2 and module["normalize_flash_attention_version"](4) == 4
    attention = module["TinyTransformer"](config).blocks[0].attention
    assert attention._has_flash_attn4_varlen_func is False and attempts == []
    import attention_runtime
    monkeypatch.setattr(attention_runtime, "_fa4_varlen_func", attention_runtime._UNSET)
    attention.flash_attention_version = 4
    # An incompatible CuTe DSL disables FA4 instead of raising at import.
    assert attention._has_flash_attn4_varlen_func is False and attempts
