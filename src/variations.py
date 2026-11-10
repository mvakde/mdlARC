"""Lazy training for exported task variations and their original parents."""

from __future__ import annotations

import functools
import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Sampler
import distributed_training as distributed
from demo_schedule import MAX_CONTEXT_DEMOS

from common import (
    ARCExampleDataset,
    _derive_order_seed,
    build_augmentor,
    build_unique_input_augments,
    Augmentor,
    collate_examples,
    extract_pairs_colors,
    extract_task_output_only_colors,
    color_augmentation_domain,
    _color_mapping_tensor,
)


ROOT = Path(__file__).resolve().parents[1]


class VariationTrainingDataset(ARCExampleDataset):
    def __init__(self, *, json_path: Path, index_path: Path, seed: int, max_augments: int,
                 augmentation_sampling_mode: Optional[str] = None,
                 attention_pattern: str = "dense", training_demo_schedule: str = "all",
                 pair_identity_vertices: int = 0, max_training_demos: Optional[int] = None):
        with Path(index_path).open(encoding="utf-8") as handle:
            self.variation_index = json.load(handle)
        self.variation_root = ROOT
        if self.variation_index.get("snapshot") != "task-variations-v1":
            raise ValueError("Unsupported task-variation snapshot")
        self.variation_parents = self.variation_index["parents"]
        if self.variation_index.get("sampling", {}).get("unit") == "family":
            catalog_path = Path(index_path).with_name("family_catalog.json")
            catalog = json.loads(catalog_path.read_text())
            canonical = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(canonical).hexdigest() != self.variation_index.get("catalog_sha256"):
                raise ValueError("Run family catalog changed")
            members = lambda entries: [(v["sha256"], v["kind"], v["seed"], v["pair_count"]) for v in entries]
            if ({k: members(v) for k, v in self.variation_parents.items()} !=
                    {k: members(v["instances"]) for k, v in catalog["families"].items()}):
                raise ValueError("Family index membership differs from immutable catalog")
            self.family_sample_hashes = {v["path"]: v["sha256"] for vs in self.variation_parents.values() for v in vs}
        self.eval_support_task_ids = set(self.variation_index["eval_support_task_ids"])
        self.variation_cache: OrderedDict[str, list[dict]] = OrderedDict()
        family_mode = self.variation_index.get("sampling", {}).get("unit") == "family"
        if family_mode and max_training_demos is None:
            # Family-sampled data always trains on balanced capped contexts (at most six demos).
            max_training_demos = MAX_CONTEXT_DEMOS
        capped = max_training_demos is not None or training_demo_schedule == "two"
        if not (family_mode or capped):
            # Uncapped task-variation training shows every demo, as before the capped schedules.
            training_demo_schedule = "all"
        super().__init__(
            json_path=json_path,
            splits=("train",),
            seed=seed,
            max_augments=max_augments,
            augmentation_sampling_mode=augmentation_sampling_mode,
            attention_pattern=attention_pattern,
            training_demo_schedule=training_demo_schedule,
            max_training_demos=max_training_demos,
            pair_identity_vertices=pair_identity_vertices if family_mode else 0,
            max_seq_len=None,
        )
        missing = set(self.variation_parents) - set(self.task_ids)
        if missing:
            raise ValueError(f"Variation parents missing from training task index: {sorted(missing)[:3]}")
        self.variation_task_ids: dict[str, list[str]] = {}
        self.variation_embedding_families: dict[int, list[int]] = {}
        scope = self.variation_index.get("sampling", {}).get("embedding_scope", "parent")
        if scope not in ("parent", "variation"):
            raise ValueError("Variation embedding_scope must be 'parent' or 'variation'")
        for parent_id, variants in sorted(self.variation_parents.items()):
            task_ids = []
            rows = []
            for variant in variants:
                task_id = parent_id
                if scope == "variation":
                    task_id = f"{parent_id}/variation/seed{variant['seed']}"
                    if task_id in self.task_id_to_embedding_ids:
                        raise ValueError(f"Duplicate variation task identity: {task_id}")
                    row = len(self.embedding_task_ids)
                    self.embedding_task_ids.append(task_id)
                    self.embedding_task_id_to_index[task_id] = row
                    self.task_id_to_embedding_ids[task_id] = (row, -1)
                    self.task_id_to_example_id[task_id] = len(self.task_ids)
                    self.task_ids.append(task_id)
                    rows.append(row)
                task_ids.append(task_id)
            self.variation_task_ids[parent_id] = task_ids
            if rows:
                parent_row = self.task_id_to_embedding_ids[parent_id][0]
                self.variation_embedding_families[parent_row] = rows
        self.num_examples = len(self.embedding_task_ids)
        self.has_variation_sampling = True
        maximum_pairs = max((int(item["pair_count"]) for variants in self.variation_parents.values()
                             for item in variants), default=0)
        self.max_seq_len = max(self.max_seq_len, self.context_capacity(maximum_pairs))

    @torch.no_grad()
    def initialize_task_factors(self, embedding: torch.nn.Embedding) -> None:
        """Copy each normally initialized parent row to its independent variant rows."""
        for parent_row, variant_rows in self.variation_embedding_families.items():
            rows = torch.tensor(variant_rows, device=embedding.weight.device)
            embedding.weight[rows] = embedding.weight[parent_row].clone()

    def checkpoint_metadata(self, *, enable_aug: bool) -> dict:
        index = self.variation_index
        metadata = {
            key: index[key]
            for key in ("snapshot", "archive_sha256", "sources", "original_sources", "sampling", "eval_support_repeats_per_epoch")
        }
        for key in ("parent_split", "evaluation_source", "public_pretraining", "composition_archive_sha256",
                    "family_catalog", "catalog_sha256", "data_mode"):
            if key in index:
                metadata[key] = index[key]
        metadata.update(seed=self.seed, max_augments=self.max_augments, enable_aug=bool(enable_aug))
        if self.max_training_demos not in (None, MAX_CONTEXT_DEMOS):
            metadata['max_training_demos'] = self.max_training_demos
        if index.get("sampling", {}).get("unit") == "family":
            metadata.update(training_demo_schedule=self.training_demo_schedule,
                            attention_pattern=self.attention_pattern,
                            pair_identity_vertices=self.pair_identity_vertices)
        return metadata

    def _pairs_for_path(self, relative_path: str) -> list[dict]:
        cached = self.variation_cache.pop(relative_path, None)
        if cached is None:
            path = self.variation_root / relative_path
            if hasattr(self, "family_sample_hashes"):
                if hashlib.sha256(path.read_bytes()).hexdigest() != self.family_sample_hashes[relative_path]:
                    raise ValueError("Family payload changed")
            with path.open(encoding="utf-8") as handle:
                task = json.load(handle)
            cached = list(task.get("train", [])) + list(task.get("test", []))
        self.variation_cache[relative_path] = cached
        while len(self.variation_cache) > 8:
            self.variation_cache.popitem(last=False)
        return cached

    def __getitem__(self, idx: object):
        if isinstance(idx, tuple) and idx and idx[0] == "regular" and len(idx) == 4:
            return replace(super().__getitem__(idx[:3]), demo_visit_override=int(idx[3]))
        if isinstance(idx, tuple) and idx and idx[0] == "variation":
            _, parent_id, variant_index, pair_index, epoch = idx[:5]
            parent_id = str(parent_id)
            item = self.variation_parents[parent_id][int(variant_index)]
            task_id = self.variation_task_ids[parent_id][int(variant_index)]
            path = str(item["path"])
            pairs = self._pairs_for_path(path)
            pair_index = int(pair_index)
            example = self._build_sequence_example(
                example_id=self.task_id_to_example_id[task_id],
                task_id=task_id,
                task_embedding_ids=self.task_id_to_embedding_ids[task_id],
                split="train",
                pair_index=pair_index,
                pair=pairs[pair_index],
                is_rearc=False,
                is_nvarc=False,
            )
            if example is None:
                raise ValueError(f"Variation target exceeds sequence limit: {path}:{pair_index}")
            return replace(
                example,
                variation_path=path,
                variation_colors=tuple(extract_pairs_colors(pairs)),
                variation_output_only_colors=tuple(extract_task_output_only_colors({"train": pairs})),
                augment_epoch_override=int(epoch),
                demo_visit_override=int(idx[5]) if len(idx) > 5 else int(epoch),
            )
        return super().__getitem__(idx)

    def demonstration_pairs(self, example):
        if example.variation_path is not None:
            return self._pairs_for_path(example.variation_path)
        return super().demonstration_pairs(example)


class VariationBatchSampler(Sampler[list[object]]):
    def __init__(self, dataset: VariationTrainingDataset, batch_size: int, *, seed: int,
                 variations_per_parent_per_epoch: int, eval_support_repeats_per_epoch: int,
                 minibatch_multiple: int = 1, num_epochs: int = 1):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.variations_per_parent_per_epoch = int(variations_per_parent_per_epoch)
        self.eval_support_repeats_per_epoch = int(eval_support_repeats_per_epoch)
        self.minibatch_multiple = max(1, int(minibatch_multiple))
        self.num_epochs = max(1, int(num_epochs))
        self.drop_last = False
        self._epoch = 0
        self._orders: dict[str, list[int]] = {}
        self._cycle_orders: dict[str, tuple[int, list[int]]] = {}
        self.composition_parents = set(dataset.variation_index.get("composition_parents", []))
        self.family_sampling = dataset.variation_index.get("sampling", {}).get("unit") == "family"
        self.preserve_all_pairs = self.family_sampling
        if self.family_sampling and (self.variations_per_parent_per_epoch != 1 or self.eval_support_repeats_per_epoch):
            raise ValueError("Family sampling requires one instance and zero evaluation support repeats")
        for parent_id, variants in sorted(dataset.variation_parents.items()):
            order = list(range(len(variants)))
            random.Random(_derive_order_seed(self.seed, (parent_id, "variation", 0))).shuffle(order)
            self._orders[parent_id] = order

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def _sample_count(self, parent_id):
        if self.family_sampling:
            return 1
        return (len(self.dataset.variation_parents[parent_id]) if parent_id in self.composition_parents
                else self.variations_per_parent_per_epoch)

    def _order_for_cycle(self, parent_id: str, cycle: int) -> list[int]:
        first_order = self._orders[parent_id]
        if not self.family_sampling or cycle == 0 or len(first_order) < 2:
            return first_order
        previous_cycle, order = self._cycle_orders.get(parent_id, (0, first_order))
        if previous_cycle > cycle:
            previous_cycle, order = 0, first_order
        # Rebuild deterministically on backward/random access; retain only one
        # cycle per family, rather than every permutation across the horizon.
        for current_cycle in range(previous_cycle + 1, cycle + 1):
            next_order = list(range(len(first_order)))
            random.Random(_derive_order_seed(
                self.seed, (parent_id, "variation", current_cycle),
            )).shuffle(next_order)
            if next_order == order:
                next_order = next_order[1:] + next_order[:1]
            order = next_order
        self._cycle_orders[parent_id] = (cycle, order)
        return order

    def _variants_for_epoch(self, parent_id: str, epoch: int):
        count = self._sample_count(parent_id)
        for offset in range(count):
            visit, position = divmod(epoch * count + offset, len(self._orders[parent_id]))
            yield self._order_for_cycle(parent_id, visit)[position], visit

    def _indices_for_epoch(self, epoch: int) -> list[object]:
        indices: list[object] = []
        for parent_id, variants in sorted(self.dataset.variation_parents.items()):
            for variant_index, visit in self._variants_for_epoch(parent_id, epoch):
                for pair_index in range(int(variants[variant_index]["pair_count"])):
                    indices.append(("variation", parent_id, variant_index, pair_index, epoch, visit))
        for example_id, example_indices in self.dataset.regular_train_indices_by_example_id.items():
            if self.family_sampling:
                raise ValueError("Family instances must all be in the sampling pool; static pairs are forbidden")
            task_id = self.dataset.task_ids[example_id]
            repeats = (
                self.eval_support_repeats_per_epoch
                if task_id in self.dataset.eval_support_task_ids else 1
            )
            for repeat in range(repeats):
                indices.extend(("regular", index, epoch, epoch * repeats + repeat)
                               for index in example_indices)
        random.Random(self.seed + epoch).shuffle(indices)
        if self.drop_last and not self.preserve_all_pairs:
            block = self.batch_size * self.minibatch_multiple
            indices = indices[:len(indices) // block * block]
        return indices

    def __len__(self) -> int:
        # Count selected pairs without allocating and shuffling a full epoch.
        count = 0
        for parent_id, variants in self.dataset.variation_parents.items():
            for variant_index, _ in self._variants_for_epoch(parent_id, self._epoch):
                count += int(variants[variant_index]["pair_count"])
        for example_id, indices in self.dataset.regular_train_indices_by_example_id.items():
            repeats = (self.eval_support_repeats_per_epoch
                       if self.dataset.task_ids[example_id] in self.dataset.eval_support_task_ids else 1)
            count += len(indices) * repeats
        if self.drop_last and not self.preserve_all_pairs:
            block = self.batch_size * self.minibatch_multiple
            count = count // block * block
        return count // self.batch_size if self.drop_last and not self.preserve_all_pairs else (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = self._indices_for_epoch(self._epoch)
        return iter([indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)])


class VariationAugmentor:
    def __init__(self, dataset: VariationTrainingDataset, *, seed: int, max_augments: int):
        self.dataset = dataset
        self.unique_pools = OrderedDict()
        self.seed = int(seed)
        self.augmentation_sampling_mode = dataset.augmentation_sampling_mode
        self.deduplicate_variation_inputs = getattr(dataset, "variation_index", {}).get(
            "sampling", {}
        ).get("deduplicate_variation_inputs", True)
        self.max_augments = max(1, int(max_augments))
        self.regular = (
            build_augmentor(dataset.examples, dataset.task_colors,
                            max_augments=max_augments, seed=seed,
                            augmentation_sampling_mode=dataset.augmentation_sampling_mode,
                            task_output_only_colors=dataset.task_output_only_colors,
                            task_inputs={task_id: [pair["input"] for pair in
                                         dataset.pairs_by_task.get(task_id, []) + dataset.test_pairs_by_task.get(task_id, [])]
                                         for task_id in {example.task_id for example in dataset.examples}}
                                        if dataset.augmentation_sampling_mode == "single_grid_prefix" else None)
            if dataset.examples else None
        )

    def set_epoch(self, epoch: int) -> None:
        if self.regular is not None:
            self.regular.set_epoch(epoch)

    def select_for_example(self, example):
        if example.variation_path is None:
            return (self.regular.select_for_example(example)
                    if self.regular is not None else (None, None))
        if self.augmentation_sampling_mode == "single_grid_prefix" and self.deduplicate_variation_inputs:
            path = example.variation_path
            augmentor = self.unique_pools.pop(path, None)
            if augmentor is None:
                pairs = self.dataset._pairs_for_path(path)
                augmentor = Augmentor(build_unique_input_augments(
                    {(example.task_id, "train", index): pair["input"] for index, pair in enumerate(pairs)},
                    example.variation_colors, protected_colors=example.variation_output_only_colors,
                    max_augments=self.max_augments, seed=self.seed, mapping_dtype=torch.uint8,
                ), max_augments=self.max_augments)
            self.unique_pools[path] = augmentor
            while len(self.unique_pools) > 128:
                self.unique_pools.popitem(last=False)
            return augmentor.select_for_example(example)
        epoch = int(example.augment_epoch_override or 0) % self.max_augments
        order_seed = _derive_order_seed(
            self.seed,
            (example.task_id, example.variation_path, example.pair_index),
        )
        rng = random.Random(order_seed + epoch)
        colors, destinations = color_augmentation_domain(
            example.variation_colors, example.variation_output_only_colors,
            self.augmentation_sampling_mode,
        )
        mapping = _color_mapping_tensor(
            colors, rng.sample(range(10) if destinations is None else destinations, len(colors)),
            destinations,
        )
        return mapping, rng.randrange(8)


def create_variation_dataloader(dataset: VariationTrainingDataset, *, batch_size: int,
                                seed: int, enable_aug: bool, max_augments: int,
                                num_workers: int, pin_memory: Optional[bool],
                                persistent_workers: Optional[bool], prefetch_factor: Optional[int],
                                gradient_accumulation_steps: int, num_epochs: int = 1):
    sampling = dataset.variation_index["sampling"]
    dataset.variation_augmentation_enabled = bool(enable_aug)
    batch_sampler = VariationBatchSampler(
        dataset,
        batch_size,
        seed=seed,
        variations_per_parent_per_epoch=sampling["variations_per_parent_per_epoch"],
        eval_support_repeats_per_epoch=dataset.variation_index["eval_support_repeats_per_epoch"],
        minibatch_multiple=gradient_accumulation_steps,
        num_epochs=num_epochs,
    )
    if enable_aug and dataset.augmentation_sampling_mode == "context_palette":
        from context_palette import ContextPaletteAugmentor
        augmentor = ContextPaletteAugmentor(dataset, seed=seed, max_augments=max_augments)
    else:
        augmentor = VariationAugmentor(dataset, seed=seed, max_augments=max_augments) if enable_aug else None
    collate = functools.partial(
        collate_examples,
        dataset=dataset,
        augment_selector=augmentor.select_for_example if augmentor else None,
    )
    num_workers = max(0, int(num_workers))
    kwargs = {
        "batch_sampler": distributed.shard_batch_sampler(batch_sampler),
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
    }
    if num_workers:
        if distributed.world_size() > 1:
            kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True if persistent_workers is None else bool(persistent_workers)
        kwargs["prefetch_factor"] = 2 if prefetch_factor is None else max(1, int(prefetch_factor))
    loader = DataLoader(dataset, **kwargs)
    if augmentor is not None:
        loader.augmentor = augmentor
        dataset.augmentor = augmentor
    return loader
