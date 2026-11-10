"""Context-observable color augmentation, independent of model/sampler RNG.

K is the known task palette; V contains only the selected demos and target
input. Fix K-V, reserve those destinations, and permute every other label.
Unused labels remain legal destinations. Neither kind of input deduplication
is imposed. Geometry and demo selection remain owned by the existing schedules.
"""
from collections import defaultdict, OrderedDict
import random

import torch

from pair_identity import schedule_seed
from intermediate_sampling import intermediate_candidates


def grid_colors(grid):
    return {int(color) for row in grid for color in row if 0 <= int(color) < 10}


def protection_schedule(dataset, example, count, *, visits=None):
    """Test answers and gold target output never contribute to V."""
    pairs = dataset.demonstration_pairs(example)
    pair_colors = [grid_colors(p['input']) | grid_colors(p['output']) for p in pairs]
    candidate_colors = [[grid_colors(g) for g in intermediate_candidates(p)] for p in pairs]
    targets = dataset.test_pairs_by_task[example.task_id] if example.split == 'test' else pairs
    target_colors = grid_colors(targets[example.pair_index]['input'])
    known = set(example.variation_colors if getattr(example, 'variation_path', None)
                else dataset.task_colors[example.task_id])
    protected = []
    for visit in (range(count) if visits is None else visits):
        visible = set(target_colors)
        for index in dataset.demonstration_indices(example, visit):
            visible.update(pair_colors[index])
            if candidate_colors[index]:
                selected = dataset.intermediate_grid_indices(example, index, visit)
                for candidate in selected:
                    visible.update(candidate_colors[index][candidate])
        protected.append(tuple(sorted(known - visible)))
    return protected


def color_schedule(dataset, example, dihedrals, *, seed, max_augments, visits=None):
    """Balance actual visits conditional on protection, then pair with fixed D4.

    Reuse complete Latin blocks from a pool capped at max_augments (rounded
    down to complete blocks). Every source visits every legal destination once
    per block. Whole-horizon destination counts differ by at most one within
    each protection set. The existing minimum-multiplicity color/D4 assignment
    preserves those histograms and avoids joint repeats when feasible.
    """
    from evaluate import _build_balanced_inference_color_schedule, _pair_color_mappings_with_dihedrals
    groups = defaultdict(list)
    for visit, protected in enumerate(protection_schedule(dataset, example, len(dihedrals), visits=visits)):
        groups[protected].append(visit)
    result = [None] * len(dihedrals)
    for protected, visits in groups.items():
        legal = [color for color in range(10) if color not in protected]
        period = max(1, len(legal))
        pool_count = period * max(1, int(max_augments) // period)
        identity = (example.task_id, example.split, example.pair_index)
        if getattr(example, 'variation_path', None):
            identity += (example.variation_path,)
        group_seed = schedule_seed(seed, *identity,
                                   'context-palette-v1', protected)
        pool = _build_balanced_inference_color_schedule(
            legal, pool_count, group_seed, destination_colors=legal)
        mappings = pool[torch.arange(len(visits)) % len(pool)]
        geometry = [dihedrals[visit] for visit in visits]
        anchor = next((i for i, value in enumerate(geometry) if value == 0), 0)
        paired = _pair_color_mappings_with_dihedrals(
            mappings, geometry, legal, anchor_index=anchor,
            seed=schedule_seed(group_seed, 'geometry-pairing'))
        for visit, mapping in zip(visits, paired):
            result[visit] = mapping
    return result


class ContextPaletteAugmentor:
    """Replayable regular and family training plans using actual sampler visits.

    The parent prepares the small geometry table before workers start. Workers
    lazily cache color plans, keyed by physical target, for the complete planned
    horizon. Selection has no mutable visit counter; resume, worker assignment,
    out-of-order access, and epoch changes cannot change a plan.
    """
    def __init__(self, dataset, *, seed, max_augments):
        self._family_sampling = (getattr(dataset, 'variation_index', {}).get('sampling', {}).get('unit') == 'family')
        if (dataset.has_rearc_train_sampling or dataset.has_nvarc_train_sampling
                or (getattr(dataset, 'has_variation_sampling', False) and not self._family_sampling)):
            raise ValueError('context_palette currently supports regular ARC tasks only')
        self.dataset = dataset
        self.seed = int(seed)
        self.max_augments = int(max_augments)
        self._enabled = True
        self._epoch = 0
        self._shared_epoch = None
        self._geometry = {}
        self._plans = OrderedDict()
        self._family_visits = None

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        if self._shared_epoch is not None:
            self._shared_epoch.fill_(self._epoch)

    def set_shared_epoch_buffer(self, shared_epoch):
        self._shared_epoch = shared_epoch
        if shared_epoch is not None:
            shared_epoch.fill_(self._epoch)

    def prepare_training_schedule(self, sampler):
        """Read the sampler plan; never change its order, seeds or geometry."""
        if self._family_sampling:
            if not getattr(sampler, 'family_sampling', False):
                raise ValueError('context_palette requires the actual family sampler')
            # Membership/order metadata only. Payloads and color plans remain lazy;
            # no full epoch or corpus is materialized in the parent process.
            self._family_visits = {}
            for parent, variants in self.dataset.variation_parents.items():
                cycles, remainder = divmod(sampler.num_epochs, len(variants))
                extra = set(sampler._order_for_cycle(parent, cycles)[:remainder]) if remainder else set()
                for variant, item in enumerate(variants):
                    task_id = self.dataset.variation_task_ids[parent][variant]
                    self._family_visits[(task_id, item['path'])] = cycles + int(variant in extra)
            self._plans.clear()
            return
        if not hasattr(sampler, '_balanced_indices'):
            raise ValueError('context_palette requires the balanced regular training sampler')
        geometry = defaultdict(dict)
        for epoch in sampler._balanced_indices():
            for source, index, visit, dihedral, occurrence in epoch:
                if source != 'regular' or visit in geometry[index]:
                    raise ValueError('Unexpected regular training visit schedule')
                # drop_last can skip different physical targets each epoch.
                # Keep their original demo-visit IDs; balance colors over only
                # the visits actually yielded by the finalized sampler.
                geometry[index][int(visit)] = int(dihedral)
        self._geometry = {
            (self.dataset.examples[index].task_id, self.dataset.examples[index].split,
             self.dataset.examples[index].pair_index): dict(values)
            for index, values in geometry.items()
        }
        self._plans.clear()

    def select_for_example(self, example):
        if not self._enabled:
            return None, None
        if self._family_sampling:
            return self._select_family_example(example)
        key = (example.task_id, example.split, example.pair_index)
        geometry = self._geometry.get(key)
        if geometry is None:
            raise ValueError('Prepare context_palette with the actual training sampler before selecting')
        visit = example.demo_visit_override
        if visit is None:
            visit = example.augment_epoch_override
        if visit is None:
            visit = self._epoch if self._shared_epoch is None else int(self._shared_epoch.item())
        visit = int(visit)
        if visit not in geometry:
            raise ValueError('Context palette visit exceeds the prepared training horizon')
        if example.preselected_dihedral_index is not None and int(example.preselected_dihedral_index) != geometry[visit]:
            raise ValueError('Context palette geometry disagrees with the balanced sampler')
        if key not in self._plans:
            visits = list(geometry)
            maps = color_schedule(self.dataset, example, list(geometry.values()),
                seed=self.seed, max_augments=self.max_augments, visits=visits)
            self._plans[key] = dict(zip(visits, maps))
        return self._plans[key][visit], geometry[visit]

    def _select_family_example(self, example):
        if self._family_visits is None:
            raise ValueError('Prepare context_palette with the actual family sampler before selecting')
        key = (example.task_id, example.variation_path, example.pair_index)
        count = self._family_visits[(example.task_id, example.variation_path)]
        visit = int(example.demo_visit_override)
        if not 0 <= visit < count:
            raise ValueError('Context palette visit exceeds the prepared training horizon')
        plan = self._plans.pop(key, None)
        if plan is None:
            order = list(range(8))
            random.Random(schedule_seed(self.seed, *key, 'family-context-geometry-v1')).shuffle(order)
            geometry = [order[i % 8] for i in range(count)]
            maps = color_schedule(self.dataset, example, geometry,
                seed=self.seed, max_augments=self.max_augments, visits=range(count))
            plan = maps, geometry
        self._plans[key] = plan
        while len(self._plans) > 128:
            self._plans.popitem(last=False)
        maps, geometry = plan
        return maps[visit], geometry[visit]
