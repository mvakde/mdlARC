"""Grid-only candidate pools and replayable, ordered context augmentation."""
import random

from pair_identity import schedule_seed


SAMPLING = "ordered-stratified-per-visit-v1"


def intermediate_candidates(pair):
    """Old immutable packages retain their fixed zero/one/two-grid selection."""
    return pair.get("intermediate_grid_candidates", pair.get("intermediate_grids", []))


def intermediate_indices(pair, *, seed, task, split, target, visit):
    candidates = intermediate_candidates(pair)
    count = len(candidates)
    if "intermediate_grid_candidates" not in pair or count <= 2:
        return tuple(range(count))
    # Share the random quantiles across demos in this instance. This aligns
    # relative early/late progress; it does not assert semantic phase alignment
    # between independently written programs. Never consume global model RNG.
    rng = random.Random(schedule_seed(seed, SAMPLING, task, split, target, int(visit)))
    boundary = count // 2
    return (int(rng.random() * boundary),
            boundary + int(rng.random() * (count - boundary)))
