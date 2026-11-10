"""Stateless schedules and a fixed, isometrically embedded regular simplex."""
import hashlib
import random

import torch
from demo_schedule import balanced_demonstrations


def schedule_seed(seed, *identity):
    return int.from_bytes(hashlib.sha256(repr((int(seed), *identity)).encode()).digest()[:8], 'little')


def leave_one_out(eligible, *, visit, seed):
    """Match a training target's context, with at most six complete demos."""
    eligible = tuple(eligible)
    return balanced_demonstrations(eligible, count=len(eligible) - 1, visit=visit, seed=seed)


def balanced_pair_vertices(physical_pairs, *, visit, seed, vertices):
    """Randomized Latin cycles: each physical pair visits every vertex once.

    Within a sequence the assignment is injective. Across each N-occurrence
    cycle every vertex is used M times, and every physical pair receives each
    vertex once. The caller counts occurrences of the same visible context.
    No global augmentation, torch, or initialization RNG is consumed.
    """
    pairs = tuple(physical_pairs)
    if len(pairs) > vertices or len(set(pairs)) != len(pairs):
        raise ValueError('Simplex needs distinct physical pairs and enough vertices')
    cycle, slot = divmod(int(visit), vertices)
    rng = random.Random(schedule_seed(seed, 'simplex', pairs, cycle))
    order, shifts, pair_order = list(range(vertices)), list(range(vertices)), list(pairs)
    rng.shuffle(order); rng.shuffle(shifts); rng.shuffle(pair_order)
    return {pair: 1 + order[(i + shifts[slot]) % vertices] for i, pair in enumerate(pair_order)}


def simplex_table(vertices, width):
    """Row zero is neutral. N unit vertices occupy a fixed N-1 dimensional span.

    Helmert's orthonormal basis embeds the centered simplex in the first N-1
    model coordinates. No learned parameter or random projection is introduced.
    """
    if vertices < 2 or width < vertices - 1:
        raise ValueError('Simplex requires N >= 2 and model width >= N-1')
    basis = torch.zeros(vertices, vertices - 1, dtype=torch.float64)
    for j in range(vertices - 1):
        scale = ((j + 1) * (j + 2)) ** -0.5
        basis[:j + 1, j] = scale
        basis[j + 1, j] = -(j + 1) * scale
    table = torch.zeros(vertices + 1, width)
    table[1:, :vertices - 1] = (basis * (vertices / (vertices - 1)) ** 0.5).float()
    return table
