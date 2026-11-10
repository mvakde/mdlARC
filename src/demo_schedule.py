"""Seeded, stateless demo cycles with balanced inclusion and bounded contexts."""
from functools import lru_cache
import hashlib
import random


MAX_CONTEXT_DEMOS = 6


@lru_cache(maxsize=4096)
def _permutation(size, seed, label, cycle):
    digest = hashlib.sha256(f"{seed}|{label}|{cycle}".encode()).digest()
    order = list(range(size))
    random.Random(int.from_bytes(digest[:8], "little")).shuffle(order)
    return tuple(order)


@lru_cache(maxsize=2048)
def _layout(pairs, seed, block, exclude_target):
    order = tuple(pairs[i] for i in _permutation(len(pairs), seed, "pairs", block))
    # Nonzero offsets on one shared ring exclude each target. Translating the
    # same offsets to every target uses each pair exactly k times per epoch.
    first = int(exclude_target)
    offsets = tuple(first + i for i in _permutation(len(pairs) - first, seed, "offsets", block))
    return order, {pair: i for i, pair in enumerate(order)}, offsets


def _select(pairs, *, target, lower, upper, visit, seed):
    exclude_target = target is not None
    count = len(pairs) - int(exclude_target)
    if not upper:
        return ()
    if lower == count:
        return tuple(pair for pair in pairs if pair != target)
    visit = max(0, int(visit))
    sizes = upper - lower + 1
    cycle, slot = divmod(visit, sizes)
    counts = _permutation(sizes, seed, "count", cycle)
    selected_count = lower + counts[slot]
    # A block contains count complete count cycles, so it consumes a multiple
    # of count demo slots. Reshuffling only there preserves prefix balance and
    # cannot duplicate a demo inside a context at a shuffle boundary.
    block, local_visit = divmod(visit, count * sizes)
    complete_cycles = local_visit // sizes
    start = complete_cycles * ((lower + upper) * sizes // 2)
    start += slot * lower + sum(counts[:slot])
    order, positions, offsets = _layout(pairs, seed, block, exclude_target)
    origin = positions[target] if exclude_target else 0
    selected = (order[(origin + offsets[(start + i) % count]) % len(order)]
                for i in range(selected_count))
    # Group attention is permutation invariant; retain physical source order.
    return tuple(sorted(selected))


def training_demonstrations(solved, *, target, visit, seed, half_or_more=False,
                            max_demos=MAX_CONTEXT_DEMOS):
    """Cap training context and balance both targets and repeated visits.

    All targets of a puzzle share the count and offsets on a seeded ring for
    each visit. A complete pass over its solved targets therefore uses every
    pair equally often as a demo. For each target, eligible pair frequencies
    differ by at most one at every visit prefix. No global RNG is consumed.
    """
    solved = tuple(solved)
    if target not in solved:
        target = None
    count = len(solved) - int(target is not None)
    upper = min(MAX_CONTEXT_DEMOS, max_demos, count)
    lower = min(upper, max(2, (count + 1) // 2)) if half_or_more else upper
    return _select(solved, target=target, lower=lower, upper=upper, visit=visit, seed=seed)


def half_or_more_demonstrations(eligible, *, visit, seed):
    """Half or more available demos, capped at six, balanced at every prefix."""
    return training_demonstrations(eligible, target=None, visit=visit, seed=seed,
                                   half_or_more=True)


def balanced_demonstrations(eligible, *, count, visit, seed):
    """Fixed-size evaluation subsets; every demo's use differs by at most one."""
    eligible = tuple(eligible)
    count = max(0, min(int(count), MAX_CONTEXT_DEMOS, len(eligible)))
    return _select(eligible, target=None, lower=count, upper=count, visit=visit, seed=seed)
