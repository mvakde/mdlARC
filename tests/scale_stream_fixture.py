"""Shared fixture for the scale-augmentation tests: a small multi-task ARC file and a
digest of the exact training stream the launcher builds (context_palette, pair_group,
half_or_more demos, balanced dihedrals, drop_last finalized as in train.py)."""
import hashlib
import json
import random
from types import SimpleNamespace

import torch


def fixture_tasks():
    """Deterministic tasks of varied shapes: tiny, thin, tall, wide, mid and 30x30 grids."""
    rng = random.Random(0)

    def grid(rows, cols, colors):
        return [[rng.choice(colors) for _ in range(cols)] for _ in range(rows)]

    shapes = {
        "t_tiny": [((1, 1), (1, 1)), ((1, 2), (2, 2)), ((2, 1), (2, 2)), ((1, 1), (2, 2))],
        "t_thin": [((1, 9), (1, 9)), ((2, 7), (2, 7)), ((1, 5), (1, 5))],
        "t_tall": [((12, 2), (12, 2)), ((14, 3), (14, 3)), ((10, 1), (10, 1)), ((9, 2), (9, 2))],
        "t_mid": [((5, 5), (10, 10)), ((4, 6), (8, 12)), ((3, 3), (6, 6)), ((6, 4), (12, 8)), ((5, 7), (10, 14))],
        "t_big": [((30, 30), (30, 30)), ((28, 25), (28, 25)), ((20, 30), (20, 30))],
        "t_mixed": [((3, 3), (3, 3)), ((15, 15), (15, 15)), ((4, 2), (4, 2))],
    }
    tasks = {}
    for task_id, pairs in shapes.items():
        palette = rng.sample(range(10), 4)
        train = [{"input": grid(*shape_in, palette), "output": grid(*shape_out, palette)}
                 for shape_in, shape_out in pairs[:-1]]
        shape_in, shape_out = pairs[-1]
        tasks[task_id] = {"train": train, "test": [{"input": grid(*shape_in, palette),
                                                    "output": grid(*shape_out, palette)}]}
    return tasks


def write_fixture(path):
    path.write_text(json.dumps(fixture_tasks()))
    return path


def training_args(data_path, **overrides):
    """The baseline recipe's data/augmentation settings on a tiny model."""
    values = dict(
        seed=42, data_path=data_path, checkpoint_path=None, enable_aug=True, max_augments=80,
        augmentation_sampling_mode="context_palette", attention_pattern="pair_group",
        training_demo_schedule="half_or_more", d_model=24, n_heads=4, d_ff=48, n_layers=1,
        num_task_embedding_latents=2, dropout=0, embedding_dropout=0, attention_dropout=0,
        n_predict=1, batch_size=2, epochs=6, num_workers=0, gradient_accumulation_steps=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def training_loader(build_module, args):
    _, dataset, loader, _, _ = build_module.build_model_and_data(args)
    sampler = loader.batch_sampler
    sampler.drop_last = True  # exactly what train._train_single_model finalizes
    loader.augmentor.prepare_training_schedule(sampler)
    return dataset, loader


def stream_digest(loader, epochs):
    """SHA-256 over every collated training batch the loader yields for ``epochs``."""
    digest = hashlib.sha256()
    for epoch in range(epochs):
        loader.augmentor.set_epoch(epoch)
        loader.batch_sampler.set_epoch(epoch)
        for batch in loader:
            for key in ("input_ids", "positions_3d", "cu_seqlens", "sep_indices", "dihedral_ids", "example_ids"):
                tensor = batch[key]
                digest.update(key.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.to(torch.int64).contiguous().numpy().tobytes())
            digest.update(str(batch["max_seqlen"]).encode())
            digest.update("|".join(batch["task_ids"]).encode())
    return digest.hexdigest()
