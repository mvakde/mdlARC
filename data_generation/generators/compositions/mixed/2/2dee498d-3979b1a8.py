"""Recover a square by stable column deduplication, then grow southeast Ls.

Only the input determines the result. The seed's corner/cross/filler roles
define the three-cycle; its asymmetric texture makes whole columns unique.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json

import numpy as np


SIDES = (5, 7, 9, 11, 13, 15)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid


def _signature(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _columns(grid):
    return [tuple(int(v) for v in grid[:, j]) for j in range(grid.shape[1])]


def _deduplicate(grid):
    retained = list(dict.fromkeys(_columns(grid)))
    return np.array(retained, dtype=np.int8).T.copy()


def _cross_masks(n, reach, half):
    mid = n // 2
    cross = np.zeros((n, n), dtype=bool)
    cross[mid - half:mid + half + 1, mid - reach:mid + reach + 1] = True
    cross[mid - reach:mid + reach + 1, mid - half:mid + half + 1] = True
    moat = np.zeros_like(cross)
    moat[1:] |= cross[:-1]
    moat[:-1] |= cross[1:]
    moat[:, 1:] |= cross[:, :-1]
    moat[:, :-1] |= cross[:, 1:]
    moat &= ~cross
    return cross, moat


def _corner_mask(n, patch):
    mask = np.zeros((n, n), dtype=bool)
    for rows in (slice(0, patch), slice(n - patch, n)):
        for cols in (slice(0, patch), slice(n - patch, n)):
            mask[rows, cols] = True
    return mask


def _seed_roles(seed):
    """Read fixed spatial cues, including the isolated center component."""
    n = seed.shape[0]
    if seed.shape != (n, n) or n not in SIDES:
        raise ValueError("the recovered seed must be an odd square of side 5..15")
    colors = {int(v) for v in np.unique(seed)}
    corner = int(seed[0, 0])
    mid = n // 2
    cross_color = int(seed[mid, mid])
    if (len(colors) != 3 or corner == cross_color
            or any(int(seed[r, c]) != corner
                   for r, c in ((0, n - 1), (n - 1, 0), (n - 1, n - 1)))):
        raise ValueError("three roles and four equal corners are required")
    filler = next(iter(colors - {corner, cross_color}))
    component = np.zeros((n, n), dtype=bool)
    component[mid, mid] = True
    pending = [(mid, mid)]
    while pending:
        r, c = pending.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= rr < n and 0 <= cc < n and not component[rr, cc]
                    and int(seed[rr, cc]) == cross_color):
                component[rr, cc] = True
                pending.append((rr, cc))
    matches = []
    for reach in range(1, mid):
        for half in range(min(2, reach - 1) + 1):
            mask, moat = _cross_masks(n, reach, half)
            if np.array_equal(mask, component) and np.all(seed[moat] == filler):
                matches.append((reach, half))
    if len(matches) != 1:
        raise ValueError("the center must be an isolated, filler-bordered cross")
    return (corner, cross_color, filler), matches[0]


def _grow(seed, roles):
    n = seed.shape[0]
    result = np.empty((2 * n, 2 * n), dtype=np.int8)
    result[:n, :n] = seed
    successor = dict(zip(roles, roles[1:] + roles[:1]))
    state = int(seed[-1, -1])
    for edge in range(n, 2 * n):
        result[:edge, edge] = state
        result[edge, :edge] = state
        state = successor[state]
        result[edge, edge] = state
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    source = _grid(input_grid)
    n, width = source.shape
    if n not in SIDES or not 2 * n <= width <= 30:
        raise ValueError("input must be a horizontal redundant strip")
    seed = _deduplicate(source)
    if seed.shape != (n, n):
        raise ValueError("exactly n distinct full-height columns are required")
    if any(not 2 <= count <= 5 for count in Counter(_columns(source)).values()):
        raise ValueError("each seed column must appear two to five times")
    roles, _ = _seed_roles(seed)
    return _grow(seed, roles)


def _locked_masks(n, profile):
    patch, reach, half = profile
    corners = _corner_mask(n, patch)
    cross, moat = _cross_masks(n, reach, half)
    locked = corners | cross | moat
    # One filler cell makes a raw-strip-corner shortcut visibly falsifiable.
    locked[-1, n // 2] = True
    return corners, cross, moat, locked


@lru_cache(maxsize=6)
def _profiles(n):
    profiles = []
    for patch in (1, 2):
        for reach in range(1, max(1, n // 2 - 2) + 1):
            for half in range(min(1, reach - 1) + 1):
                profile = (patch, reach, half)
                corners, cross, moat, locked = _locked_masks(n, profile)
                if np.any(corners & (cross | moat)):
                    continue
                free_counts = sorted(int(np.sum(~locked[:, j])) for j in range(n))
                # Sufficient capacity for greedy distinct-vector assignment:
                # each column has more candidates than all earlier columns.
                if all(3 ** free > rank for rank, free in enumerate(free_counts)):
                    profiles.append(profile)
    return tuple(profiles)


def _make_seed(rng, n, roles, profile, style, near_match=False):
    corner, cross_color, filler = roles
    corners, cross, moat, locked = _locked_masks(n, profile)
    palette = np.array(roles, dtype=np.int8)
    if style == "sparse":
        seed = rng.choice(palette, size=(n, n), p=(0.13, 0.13, 0.74))
    elif style == "dense":
        seed = rng.choice(palette, size=(n, n))
    else:
        rows, cols = np.indices((n, n))
        period = int(rng.integers(3, 7))
        phase = int(rng.integers(period))
        step = int(rng.choice((1, 2, -1, -2)))
        bands = (rows + step * cols + phase) % period
        seed = np.full((n, n), filler, dtype=np.int8)
        seed[bands == 0] = corner
        seed[bands == 1] = cross_color
    seed[corners] = corner
    seed[cross] = cross_color
    seed[moat] = filler
    seed[-1, n // 2] = filler
    free_rows = [np.flatnonzero(~locked[:, j]).tolist() for j in range(n)]
    order = sorted(range(n), key=lambda j: len(free_rows[j]))
    if near_match:
        # The first, simple motif has interchangeable constraints on its two
        # outer columns. Make them differ at exactly one unprotected cell.
        constrained = [j for j in order if len(free_rows[j]) < 2]
        order = constrained + [0, n - 1] + [j for j in order
                                          if j not in constrained + [0, n - 1]]
    seen = set()
    for col in order:
        positions = free_rows[col]
        if near_match and col == n - 1:
            candidates = []
            for row in rng.permutation(positions):
                for color in rng.permutation(palette):
                    vector = seed[:, 0].copy()
                    if int(vector[row]) == int(color):
                        continue
                    vector[row] = color
                    if tuple(int(v) for v in vector) not in seen:
                        candidates.append(vector)
            if not candidates:
                raise RuntimeError("no one-cell near match available")
            seed[:, col] = candidates[int(rng.integers(len(candidates)))]
        key = tuple(int(v) for v in seed[:, col])
        if key in seen:
            positions = [int(v) for v in rng.permutation(positions)]
            capacity = 3 ** len(positions)
            start = int(rng.integers(capacity))
            # At most n existing vectors can exclude candidates. No payloads
            # are retained across retries, and no full search tree is built.
            for offset in range(min(capacity, n + 1)):
                code = (start + offset) % capacity
                vector = seed[:, col].copy()
                for row in positions:
                    vector[row] = palette[code % 3]
                    code //= 3
                key = tuple(int(v) for v in vector)
                if key not in seen:
                    seed[:, col] = vector
                    break
            else:
                raise RuntimeError("column capacity invariant failed")
        seen.add(tuple(int(v) for v in seed[:, col]))
    _require(len(seen) == n, "canonical columns are not distinct")
    _require(_seed_roles(seed)[0] == roles, "role markers were damaged")
    return seed.astype(np.int8, copy=False)


def _indices(rng, n, kind, force_wrong_corner=False):
    counts = [2] * n
    if kind != "blocks":
        spare = min(30 - 2 * n, 3 * n)
        extra = int(rng.integers(1 if spare else 0, spare + 1))
        for _ in range(extra):
            eligible = [j for j in range(n) if counts[j] < 5]
            counts[int(rng.choice(eligible))] += 1
    if kind == "blocks":
        prefix = list(range(n))
    elif kind == "adjacent":
        prefix = [0, 0] + list(range(1, n))
    elif kind == "separated":
        prefix = [0, 1, 0] + list(range(2, n))
    else:
        prefix = [0, 1, 0, 1]
        used = Counter(prefix)
        for new in range(2, n):
            prefix.append(new)
            used[new] += 1
            available = [j for j in range(new + 1) if used[j] < counts[j]]
            if available and bool(rng.integers(2)):
                chosen = int(rng.choice(available))
                prefix.append(chosen)
                used[chosen] += 1
    remaining = counts.copy()
    for j in prefix:
        remaining[j] -= 1
    tail = []
    if force_wrong_corner:
        chosen = n // 2
        # The separated prefix uses the middle column just once.
        _require(remaining[chosen] >= 1, "cannot reserve final middle column")
        remaining[chosen] -= 1
        tail = [chosen]
    suffix = [j for j in range(n) for _ in range(remaining[j])]
    if kind == "blocks":
        if bool(rng.integers(2)):
            suffix.reverse()
    elif bool(rng.integers(2)):
        rng.shuffle(suffix)
    else:
        if bool(rng.integers(2)):
            suffix.reverse()
    return prefix + suffix + tail


def _new_pair(rng, n, roles, profile, style, kind, seen, near_match=False,
              wrong_corner=False):
    for _ in range(32):
        seed = _make_seed(rng, n, roles, profile, style, near_match)
        if near_match and _columns(seed) == sorted(_columns(seed)):
            continue
        indices = _indices(rng, n, kind, wrong_corner)
        source = seed[:, indices].copy()
        key = _signature(source)
        if key not in seen:
            seen.add(key)
            # Labels are built from the constructed seed, without calling
            # solve; the validator re-extracts it from the returned input.
            return {"input": source, "output": _grow(seed, roles)}
    raise RuntimeError("bounded distinct-input attempts exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_n = int(rng.choice((5, 7)))
    second_n = int(rng.choice((7, 9)))
    if first_n == second_n:
        second_n = 9
    sizes = [first_n, second_n] + [int(rng.choice((5, 7, 9)))
                                  for _ in range(num_train - 2)]
    first_roles = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    seen = set()
    puzzle = {"train": [], "test": []}
    for i, n in enumerate(sizes):
        roles = (first_roles if i == 0 else (first_roles[1], first_roles[0], first_roles[2])
                 if i == 1 else tuple(int(c) for c in rng.choice(10, size=3, replace=False)))
        profile = ((1, 1, 0) if i == 0 else
                   _profiles(n)[int(rng.integers(len(_profiles(n))))])
        style = "sparse" if i == 0 else str(rng.choice(("sparse", "dense", "bands")))
        kind = ("adjacent" if i == 0 else "separated" if i == 1 else
                str(rng.choice(("blocks", "adjacent", "separated", "delayed"))))
        puzzle["train"].append(_new_pair(
            rng, n, roles, profile, style, kind, seen,
            near_match=i == 0, wrong_corner=i == 1))
    for _ in range(num_test):
        n = int(rng.choice((11, 13, 15)))
        roles = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
        # Use familiar cross geometry: extrapolate length, texture and column
        # order without asking the test to teach a new kind of role marker.
        taught = {_seed_roles(_deduplicate(p["input"]))[1] for p in puzzle["train"]}
        eligible = [p for p in _profiles(n) if p[1:] in taught]
        profile = eligible[int(rng.integers(len(eligible)))]
        style = str(rng.choice(("sparse", "dense", "bands")))
        kind = str(rng.choice(("separated", "delayed")))
        puzzle["test"].append(_new_pair(rng, n, roles, profile, style, kind, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(source, seed):
    vectors = _columns(source)
    canonical = _columns(seed)
    ids = [canonical.index(v) for v in vectors]
    last_new = max(ids.index(j) for j in range(len(canonical)))
    return {
        "crop_fails": not np.array_equal(source[:, :seed.shape[1]], seed),
        "separated_early": any(ids[j] in ids[:j - 1] and ids[j] != ids[j - 1]
                               for j in range(2, last_new + 1)),
        "adjacent_early": any(ids[j] == ids[j - 1] for j in range(1, last_new + 1)),
        "sort_fails": canonical != sorted(canonical),
        "raw_corner_fails": int(source[-1, -1]) != int(seed[-1, -1]),
        "near_match": any(sum(x != y for x, y in zip(a, b)) == 1
                          for i, a in enumerate(canonical) for b in canonical[i + 1:]),
        "unequal_multiplicities": len(set(Counter(vectors).values())) > 1,
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     label + " needs exactly input and output")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(source)
            _require(key not in seen, label + " repeats an input")
            seen.add(key)
            n, width = source.shape
            _require(n in SIDES and 2 * n <= width <= 30, label + " has invalid strip shape")
            # Equality comparisons provide a second extraction implementation.
            retained = []
            for col in source.T:
                if not any(np.array_equal(col, old) for old in retained):
                    retained.append(col.copy())
            _require(len(retained) == n, label + " does not recover a square")
            seed = np.stack(retained, axis=1)
            roles, cross_profile = _seed_roles(seed)
            _require(all(2 <= c <= 5 for c in Counter(_columns(source)).values()),
                     label + " has unsupported multiplicities")
            _require(target.shape == (2 * n, 2 * n), label + " does not double side")
            _require(np.array_equal(target[:n, :n], seed), label + " corrupts canonical seed")
            _require(set(np.unique(source)) == set(np.unique(target)), label + " changes palette")
            successor = dict(zip(roles, roles[1:] + roles[:1]))
            observed = set()
            for edge in range(n, 2 * n):
                old = int(target[edge - 1, edge - 1])
                new = int(target[edge, edge])
                _require(np.all(target[:edge, edge] == old)
                         and np.all(target[edge, :edge] == old), label + " violates inherited arms")
                _require(new == successor[old], label + " violates corner succession")
                observed.add((old, new))
                # Closed-form phase check as well as the local recurrence.
                _require(old == roles[(edge - n) % 3]
                         and new == roles[(edge - n + 1) % 3], label + " has a phase error")
            _require(observed == set(successor.items()), label + " omits a cycle transition")
            _require(np.array_equal(solve(source), target), label + " disagrees with solve")
            info[split].append((n, roles, cross_profile, _features(source, seed)))
    train = info["train"]
    _require(len({v[0] for v in train}) >= 2, "training must show size-dependent doubling")
    for feature in ("crop_fails", "adjacent_early", "separated_early", "sort_fails",
                    "raw_corner_fails", "near_match"):
        _require(any(v[3][feature] for v in train), "missing training witness: " + feature)
    _require(any(c in right[1] and right[1].index(c) != pos
                 for i, left in enumerate(train) for right in train[i + 1:]
                 for pos, c in enumerate(left[1])), "literal colors never change roles")
    cycles = [dict(zip(v[1], v[1][1:] + v[1][:1])) for v in train]
    _require(any(left[c] != right[c] for i, left in enumerate(cycles)
                 for right in cycles[i + 1:] for c in left.keys() & right.keys()),
             "training permits a fixed literal-color successor map")
    _require(all(v[0] > max(t[0] for t in train) for v in info["test"]),
             "generated tests must extrapolate seed size")
    _require({v[2] for v in info["test"]} <= {v[2] for v in train},
             "test cross geometry has not been demonstrated")


def _digest(puzzle):
    payload = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 seeds, all explicit counts, decoder generality and damaged labels.

    Puzzles are checked one at a time; only bounded sets of hashes/metrics
    survive the loop. No examples, logs or execution traces are written.
    """
    digests, sizes, shapes, cross_shapes, palettes = set(), set(), set(), set(), set()
    structural, multiplicities, phases = set(), set(), set()
    features = Counter()
    role_colors = [set(), set(), set()]
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "not reproducible")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                recovered = _deduplicate(source)
                roles, profile = _seed_roles(recovered)
                n = recovered.shape[0]
                sizes.add(n)
                shapes.add(source.shape)
                cross_shapes.add(profile)
                palettes.add(tuple(sorted(roles)))
                structural.add((n, tuple(roles.index(int(v)) for v in recovered.flat)))
                multiplicities.update(Counter(_columns(source)).values())
                phases.add(roles.index(int(target[-1, -1])))
                for position, color in enumerate(roles):
                    role_colors[position].add(color)
                features.update(k for k, yes in _features(source, recovered).items() if yes)
                pairs += 1
    _require(len(digests) == 42, "duplicate puzzles across 42 seeds")
    _require(sizes == set(SIDES) and len(shapes) >= 25, "sizes/widths lack variation")
    _require(len(structural) >= 140 and len(palettes) >= 50, "texture/palette diversity too narrow")
    _require(len(cross_shapes) >= 3 and phases == {0, 1, 2}, "cross/phase variation missing")
    _require({2, 3, 4, 5} <= multiplicities, "repeat counts lack variation")
    _require(all(colors == set(range(10)) for colors in role_colors), "color roles lack coverage")
    _require(all(features[k] >= 42 for k in ("crop_fails", "near_match", "separated_early",
                                            "raw_corner_fails")), "missing causal witnesses")
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts ignored")
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    # Valid configurations outside the sampled train/test geometry check that
    # solving is not a seed lookup or dependent on the generation-only aid.
    rng = np.random.default_rng(20261023)
    for profile in ((1, 4, 1), (3, 3, 2)):
        seed = _make_seed(rng, 15, (0, 8, 4), profile, "bands")
        seed[-1, 15 // 2] = 0  # not the generation-only filler reservation
        _require(len(set(_columns(seed))) == 15, "hand-configured columns collided")
        source = seed[:, _indices(rng, 15, "separated")]
        _require(np.array_equal(solve(source.tolist()), _grow(seed, (0, 8, 4))),
                 "input-only decoder failed a hand-configured family member")
    puzzle = generate_puzzle(0)
    target = puzzle["train"][0]["output"]
    n = puzzle["train"][0]["input"].shape[0]
    for row, col in ((0, 0), (0, n), (n, n)):
        before = int(target[row, col])
        target[row, col] = (before + 1) % 10
        try:
            validate_generated_puzzle(puzzle)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("damaged seed/arm/corner label was accepted")
        finally:
            target[row, col] = before
    return {"seeds": 42, "pairs": pairs, "legal_count_combinations": count_calls,
            "distinct_puzzles": len(digests), "seed_sides": sorted(sizes),
            "input_shapes": len(shapes), "seed_structures": len(structural),
            "cross_shapes": [list(p) for p in sorted(cross_shapes)],
            "palettes": len(palettes), "multiplicities": sorted(multiplicities),
            "final_corner_phases": sorted(phases), "witness_pairs": dict(features),
            "semantic_verification": "generator-owned checks only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
