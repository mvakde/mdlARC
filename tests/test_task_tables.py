"""Without task embeddings the model has no task tables, and old checkpoints still load."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import build
from tinytransformer import TASK_TABLE_MODULES, TinyTransformer, TinyTransformerConfig
from test_mtp import _batch

PREFIXES = tuple(f"{name}." for name in TASK_TABLE_MODULES)
SHAPES = [{}, dict(n_layers=3, loop_iterations=3, loop_start_layer=1, loop_end_layer=2)]


def _model(task_embeddings, **options):
    settings = dict(d_model=24, n_heads=4, d_ff=48, n_layers=2, num_examples=3,
                    num_task_embedding_latents=3, n_predict=2, dropout=0, attention_dropout=0,
                    embedding_dropout=0, task_embeddings=task_embeddings)
    torch.manual_seed(11)
    return TinyTransformer(TinyTransformerConfig(**{**settings, **options})).eval()


@pytest.mark.parametrize("options", SHAPES)
def test_only_the_task_tables_are_missing_and_shared_weights_are_unchanged(options):
    # Up to v3.1 both settings built the same weights (task embeddings off only
    # froze the tables), and the task-embeddings model is unchanged since.
    with_tables, without = _model(True, **options), _model(False, **options)
    full, stripped = with_tables.state_dict(), without.state_dict()
    assert set(full) - set(stripped) == {key for key in full if key.startswith(PREFIXES)}
    assert set(stripped) < set(full)
    for key, value in stripped.items():
        assert torch.equal(value, full[key]), key
    assert not any(hasattr(without, name) for name in TASK_TABLE_MODULES)
    assert not without.example_gate_logit.requires_grad
    assert without.conditioning_dtype == without.token_embedding.weight.dtype


@pytest.mark.parametrize("options", SHAPES)
def test_old_and_stripped_checkpoints_load_to_the_same_logits(options):
    source = _model(False, **options)
    with torch.no_grad():
        for weight in source.parameters():
            weight.add_(0.1 * torch.randn_like(weight))
    tables = {key: value for key, value in _model(True, **options).state_dict().items()
              if key.startswith(PREFIXES)}
    task_ids = ["a", "b", "c"]
    stripped = dict(config=asdict(source.config), model_state=source.state_dict(),
                    embedding_task_ids=task_ids)
    old = dict(stripped, model_state={**source.state_dict(), **tables})
    old_keys = set(old["model_state"])
    batch = _batch()
    expected = source(**batch)["logits"]
    for checkpoint in (old, stripped):
        loaded = TinyTransformer(build.build_checkpoint_config(checkpoint)).eval()
        build._load_model_state(loaded, checkpoint, task_ids)
        assert torch.equal(loaded(**batch)["logits"], expected)
    assert set(old["model_state"]) == old_keys
