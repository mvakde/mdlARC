"""L-key strip assembly followed by global adjacent-vector run collapse.

Family convention: border color 1 is the fast guide and 2 is the slow guide.
Parent B's stripe compression is extended to complete row/column vectors of
the assembled, generally two-dimensional payload. Only adjacent copies vanish.

Only standard library and NumPy are used. No parent modules, files, labels,
random state, or generation metadata are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numbers
import numpy as np


FAST, SLOW = 1, 2
HOST_SEEDS = (0, 875307780, 181658488, 318608021, 990961231, 628399707,
              793230956, 370265095, 173825869, 932418280, 817362485,
              55440202, 197854725, 908583740, 161800089, 701806770,
              148069414, 518121700, 121917792, 487514106, 265881339,
              348733392, 601361424, 200681914, 862553109, 232434686,
              363233709, 809142332, 312080632, 20430398, 633131336,
              23817844, 838557833, 54519756, 616216965, 594844422,
              949079373, 490339757, 874380718, 18220102, 344668348, 26758217)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _identity(a):
    return a.shape, a.tobytes()


def _kept_vectors(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _collapse(a):
    # Both masks refer to full vectors of the SAME expanded grid. Removing
    # copies of equal vectors cannot change equality on the perpendicular axis.
    return a[np.ix_(_kept_vectors(a, 0), _kept_vectors(a, 1))].copy()


def _stack(blocks, background):
    width = max(b.shape[1] for b in blocks)
    a = np.full((2 * len(blocks), width), background, dtype=np.int8)
    for i, b in enumerate(blocks):
        pad = (width - b.shape[1]) // 2
        a[2*i:2*i+2, pad:pad+b.shape[1]] = b
    return a


def _read_key(a):
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            aligned = a[::-1] if bottom else a
            aligned = aligned[:, ::-1] if right else aligned
            origin, horizontal, vertical, bg = map(int, (
                aligned[0, 0], aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if {horizontal, vertical} != {FAST, SLOW}:
                continue
            if (np.all(aligned[0, 1:] == horizontal)
                    and np.all(aligned[1:, 0] == vertical)
                    and np.all(aligned[-1, 1:] == bg)
                    and np.all(aligned[1:, -1] == bg)):
                candidates.append((int(vertical == FAST), bottom, right, bg, origin))
    if len(candidates) != 1:
        raise ValueError("A unique L-key with fast=1 and slow=2 is required")
    return candidates[0]


def _boxes(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, points = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = (int(v)+1 for v in np.max(points, axis=0))
        if len(points) != (r1-r0)*(c1-c0):
            raise ValueError("Payload components must be filled rectangles")
        result.append((r0, r1, c0, c1))
    return result


def _parse(value):
    a = _grid(value)
    if min(a.shape) < 5:
        raise ValueError("Canvas is too small for separated strips and a key")
    axis, bottom, right, bg, origin = _read_key(a)
    mask = a != bg
    mask[-1 if bottom else 0] = False
    mask[:, -1 if right else 0] = False
    boxes = _boxes(mask)
    if not 1 <= len(boxes) <= 15:
        raise ValueError("Expected 1..15 strips")
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if not (1 <= r0 < r1 < a.shape[0] and 1 <= c0 < c1 < a.shape[1]):
            raise ValueError("Strips need clear outer margins")
        thickness = c1-c0 if axis else r1-r0
        length = r1-r0 if axis else c1-c0
        if thickness != 2 or length < 2:
            raise ValueError("Strips must be two cells thick and length >= 2")
        lanes.setdefault(c0 if axis else r0, []).append(box)
    coordinates = sorted(lanes)
    if any(y-x < 3 for x, y in zip(coordinates, coordinates[1:])):
        raise ValueError("Lane bands must be aligned and separated")
    slow_reverse = right if axis else bottom
    fast_reverse = bottom if axis else right
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks, ordered, counts = [], [], []
    for v in sorted(lanes, reverse=slow_reverse):
        group = sorted(lanes[v], key=lambda b: b[0] if axis else b[2],
                       reverse=fast_reverse)
        counts.append(len(group))
        for box in group:
            r0, r1, c0, c1 = box
            blocks.append(np.rot90(a[r0:r1, c0:c1], rotation).copy())
            ordered.append(box)
    lengths = [b.shape[1] for b in blocks]
    if len({length % 2 for length in lengths}) != 1:
        raise ValueError("Lengths must share parity for exact centering")
    return {"blocks": blocks, "boxes": ordered, "lengths": lengths,
            "counts": counts, "config": (axis, bottom, right),
            "background": bg, "origin": origin}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the input; assemble before comparing complete row/column vectors."""
    d = _parse(input_grid)
    return _collapse(_stack(d["blocks"], d["background"]))


def _nonlocal_repeat(a, axis):
    vectors = a if axis == 0 else a.T
    compressed = vectors[_kept_vectors(a, axis)]
    return len({_identity(v) for v in compressed}) < len(compressed)


def _wrong_outputs(d):
    blocks, bg = d["blocks"], d["background"]
    axis, bottom, right = d["config"]
    groups, start = [], 0
    for count in d["counts"]:
        groups.append(list(range(start, start+count)))
        start += count
    orders = {
        "reverse_fast": [i for group in groups for i in group[::-1]],
        "reverse_slow": [i for group in groups[::-1] for i in group],
        "fast_major": sorted(range(len(blocks)),
            key=lambda i: d["boxes"][i][0 if axis else 2],
            reverse=bottom if axis else right),
    }
    alternatives = {name: _collapse(_stack([blocks[i] for i in order], bg))
                    for name, order in orders.items()}
    expanded = _stack(blocks, bg)
    alternatives["no_compression"] = expanded
    alternatives["rows_only"] = expanded[_kept_vectors(expanded, 0)]
    alternatives["columns_only"] = expanded[:, _kept_vectors(expanded, 1)]
    alternatives["sample_one_row_per_strip"] = _collapse(expanded[::2])
    alternatives["reflect_payload"] = _collapse(_stack([b[::-1] for b in blocks], bg))
    # Global deduplication preserves only the first occurrence, unlike run collapse.
    row_indices, col_indices, seen = [], [], set()
    for i, row in enumerate(expanded):
        key = row.tobytes()
        if key not in seen:
            row_indices.append(i)
            seen.add(key)
    seen = set()
    for i, col in enumerate(expanded.T):
        key = col.tobytes()
        if key not in seen:
            col_indices.append(i)
            seen.add(key)
    alternatives["deduplicate_all"] = expanded[np.ix_(row_indices, col_indices)]
    # Compression within individual strips followed by assembly: this cannot
    # remove equal rows spanning a strip boundary and may destroy alignment.
    local = [_collapse(b) for b in blocks]
    width = max(b.shape[1] for b in local)
    alternatives["compress_strips_first"] = np.concatenate([
        np.pad(b, ((0, 0), ((width-b.shape[1])//2,
                            (width-b.shape[1]+1)//2)), constant_values=bg)
        for b in local], axis=0)
    return alternatives


def _features(d):
    a = _stack(d["blocks"], d["background"])
    out = _collapse(a)
    return {
        "row_removed": a.shape[0]-out.shape[0],
        "column_removed": a.shape[1]-out.shape[1],
        "row_recurrence": _nonlocal_repeat(a, 0),
        "column_recurrence": _nonlocal_repeat(a, 1),
        "boundary_merge": any(np.array_equal(a[i-1], a[i])
                              for i in range(2, len(a), 2)),
        "centering": len(set(d["lengths"])) > 1,
        "two_dimensional": min(out.shape) > 1,
        "distinguished": {name for name, alt in _wrong_outputs(d).items()
                          if not np.array_equal(alt, out)},
    }


def _weights(rng, width, limit):
    # Palindromic expansion factors make centered logical spans centered after
    # expansion too. They encode real adjacent full-column runs, not labels.
    half = rng.integers(1, 4, size=(width+1)//2).tolist()
    factors = half + (half[-2::-1] if width % 2 else half[::-1])
    while sum(factors) > limit:
        eligible = [i for i in range((width+1)//2) if factors[i] > 1]
        i = int(rng.choice(eligible))
        factors[i] -= 1
        factors[-1-i] = factors[i]
    if max(factors) == 1:
        i = int(rng.integers((width+1)//2))
        factors[i] += 1
        factors[-1-i] = factors[i]
    return np.asarray(factors, dtype=int)


def _payloads(rng, witness, bg):
    n = int(rng.integers(4, 8) if witness else rng.integers(1, 10))
    logical_width = int(rng.integers(6, 9) if witness else rng.integers(4, 13))
    factors = _weights(rng, logical_width, 12 if witness else 25)
    pool = np.array([c for c in range(10) if c != bg], dtype=np.int8)
    palette = rng.choice(pool, size=int(rng.integers(3, 10)), replace=False)
    # Four central columns fit every witness strip. Columns j and j+2 are
    # identical across ALL rows; j+1 differs. The fourth column breaks symmetry.
    j = (logical_width-4)//2

    def row():
        a = rng.choice(palette, size=logical_width).astype(np.int8)
        if witness:
            a[j+2] = a[j]
        return a

    raw = []
    widths = []
    allowed = list(range(2 + logical_width % 2, logical_width+1, 2))
    for _ in range(n):
        width = int(rng.choice(allowed))
        a = np.stack([row(), row()])
        if rng.random() < .40:
            a[1] = a[0]
        raw.append(a)
        widths.append(width)
    if witness:
        a, b, c, e = row(), row(), row(), row()
        # These inequalities survive removal of adjacent repeated columns.
        a[j:j+4] = palette[[0, 1, 0, 2]]
        b[j:j+4] = palette[[1, 2, 1, 0]]
        c[j:j+4] = palette[[2, 0, 2, 1]]
        e[j+2] = e[j]
        start = int(rng.integers(n-3))
        raw[start:start+4] = [np.stack([a, b]), np.stack([b, c]),
                              np.stack([e, e]), np.stack([a, b])]
        widths[start:start+4] = [logical_width, logical_width,
                                  logical_width-2, logical_width]
        # Extras also occupy the recurrence columns so the equal full-column
        # vectors do not get broken by a shorter support interval.
        widths = [max(w, logical_width-2) for w in widths]
    blocks = []
    for a, width in zip(raw, widths):
        left = (logical_width-width)//2
        b = np.repeat(a[:, left:left+width], factors[left:left+width], axis=1)
        blocks.append(b.astype(np.int8))
    return blocks


def _gaps(rng, occupied, count, extent=None):
    minimum = occupied+count+1
    if extent is None:
        extent = int(rng.integers(minimum, 30))
    spare = extent-minimum
    if spare < 0:
        raise ValueError("Layout exceeds available space")
    # A single bounded multinomial, not repeated placement retries.
    gaps = 1+rng.multinomial(spare, rng.dirichlet(np.ones(count+1)))
    return gaps.tolist(), extent


def _layout(rng, blocks):
    groups, current, used = [], [], 0
    for i, b in enumerate(blocks):
        length = b.shape[1]
        if current and (used+length+1 > 28 or rng.random() < .30):
            groups.append(current)
            current, used = [], 0
        current.append(i)
        used += length+1
    groups.append(current)
    minimum_fast = max(sum(blocks[i].shape[1] for i in group)+len(group)+1
                       for group in groups)
    fast_extent = int(rng.integers(minimum_fast, 30))
    slow_gaps, slow_extent = _gaps(rng, 2*len(groups), len(groups))
    placements, v = [], slow_gaps[0]
    for lane, group in enumerate(groups):
        gaps, _ = _gaps(rng, sum(blocks[i].shape[1] for i in group),
                         len(group), fast_extent)
        u = gaps[0]
        for k, i in enumerate(group):
            placements.append((v, u))
            u += blocks[i].shape[1]+gaps[k+1]
        v += 2+slow_gaps[lane+1]
    return placements, (slow_extent, fast_extent)


def _paint(blocks, placements, shape, config, bg, origin):
    axis, bottom, right = config
    a = np.full((shape[0]+1, shape[1]+1), bg, dtype=np.int8)
    reverse_thickness = (bottom == right) if axis else (bottom != right)
    for b, (v, u) in zip(blocks, placements):
        a[v+1:v+3, u+1:u+1+b.shape[1]] = b[::-1] if reverse_thickness else b
    a[0, 0], a[0, 1:], a[1:, 0] = origin, FAST, SLOW
    if axis:
        a = a.T
    if bottom:
        a = a[::-1]
    if right:
        a = a[:, ::-1]
    return a.copy()


_REQUIRED_ALTERNATIVES = {"reverse_fast", "reverse_slow", "fast_major",
                         "no_compression", "rows_only", "columns_only",
                         "sample_one_row_per_strip", "reflect_payload",
                         "deduplicate_all", "compress_strips_first"}


def _pair(rng, config, witness, seen):
    for _ in range(96):
        bg, origin = map(int, rng.choice([0, 3, 4, 5, 6, 7, 8, 9], 2, replace=False))
        blocks = _payloads(rng, witness, bg)
        placements, shape = _layout(rng, blocks)
        inp = _paint(blocks, placements, shape, config, bg, origin)
        if _identity(inp) in seen:
            continue
        # Ground truth is constructed from the intended normalized strips,
        # independently of the input-only component reader.
        out = _collapse(_stack(blocks, bg))
        d = _parse(inp)
        assert all(np.array_equal(a, b) for a, b in zip(blocks, d["blocks"]))
        f = _features(d)
        if witness:
            if not (f["row_recurrence"] and f["column_recurrence"]
                    and f["boundary_merge"] and f["centering"]
                    and _REQUIRED_ALTERNATIVES <= f["distinguished"]):
                continue
        elif not (f["column_removed"] > 0 and
                  any(name in f["distinguished"] for name in
                      ("reverse_fast", "reverse_slow", "reflect_payload"))):
            continue
        seen.add(_identity(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("Bounded sampling could not obtain an informative pair")


def _integer(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral)
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    first = (int(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2)))
    second = (1-first[0], not first[1], not first[2])
    configurations = [(a, b, c) for a in (0, 1) for b in (False, True)
                      for c in (False, True)]
    held = [c for c in configurations if c not in (first, second)]
    seen = set()
    train = [_pair(rng, first, True, seen), _pair(rng, second, True, seen)]
    for _ in range(num_train-2):
        train.append(_pair(rng, (first, second)[int(rng.integers(2))], False, seen))
    rng.shuffle(train)
    test = [_pair(rng, held[int(rng.integers(len(held)))], bool(rng.integers(2)), seen)
            for _ in range(num_test)]
    return {"train": train, "test": test}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training, configs = set(), [], {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _identity(inp) not in seen
            seen.add(_identity(inp))
            d = _parse(inp)
            expected = _collapse(_stack(d["blocks"], d["background"]))
            assert np.array_equal(out, expected), "Incorrect assembly or run collapse"
            assert np.all(_kept_vectors(out, 0)) and np.all(_kept_vectors(out, 1))
            configs[split].add(d["config"])
            if split == "train":
                training.append((d, _features(d)))
    assert {d["config"][0] for d, _ in training} == {0, 1}
    assert {d["config"][1] for d, _ in training} == {False, True}
    assert {d["config"][2] for d, _ in training} == {False, True}
    assert configs["train"].isdisjoint(configs["test"]), "Expected a configuration shift"
    witnesses = [(d, f) for d, f in training
                 if f["row_recurrence"] and f["column_recurrence"]
                 and f["boundary_merge"] and f["centering"]
                 and _REQUIRED_ALTERNATIVES <= f["distinguished"]]
    assert {d["config"][0] for d, _ in witnesses} == {0, 1}, "Missing two-axis evidence"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    # Streaming 42-seed checks: retain signatures and counters, never all grids.
    keys, shapes, structures, populations, lengths, configs, palettes = (set() for _ in range(7))
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        keys.add(_fingerprint(p))
        assert _fingerprint(p) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                d = _parse(inp)
                configs.add(d["config"])
                shapes.add((inp.shape, out.shape))
                populations.add(tuple(d["counts"]))
                lengths.add(tuple(d["lengths"]))
                palettes.add((d["background"], d["origin"], len(np.unique(inp))))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape)+normalized).hexdigest())
    assert len(keys) == 42 and len(configs) == 8
    assert len(shapes) >= 80 and len(structures) >= 150
    assert len(populations) >= 25 and len(lengths) >= 80 and len(palettes) >= 40
    # Explicit global-vector counterexample: separated repetitions survive.
    raw = np.array([[3,3,4,3,3],[3,3,4,3,3],[5,5,6,5,5],
                    [3,3,4,3,3]], dtype=np.int8)
    assert np.array_equal(_collapse(raw), [[3,4,3],[5,6,5],[3,4,3]])
    assert np.array_equal(_collapse(raw), _collapse(raw.T).T)
    # A's order changes B's run boundaries, including the answer's height.
    # These are complete vector comparisons, not per-row color compression.
    strips = [np.array([[3,3,4,4,3,3], [5,5,6,6,5,5]], dtype=np.int8),
              np.array([[5,5,6,6,5,5], [7,7,8,8,7,7]], dtype=np.int8),
              np.array([[9,9,1,1,9,9], [9,9,1,1,9,9]], dtype=np.int8)]
    strips.append(strips[0].copy())
    expected = np.array([[3,4,3], [5,6,5], [7,8,7], [9,1,9],
                         [3,4,3], [5,6,5]], dtype=np.int8)
    assert np.array_equal(_collapse(_stack(strips, 0)), expected)
    changed = _collapse(_stack([strips[i] for i in (1,0,2,3)], 0))
    assert changed.shape == (7, 3) and not np.array_equal(changed, expected)
    # Equal cells in one row do not make equal FULL columns. No columns
    # vanish here, although each individual row contains a repeated color.
    assert np.array_equal(_collapse(np.array([[3,3,4], [5,6,6]])),
                          [[3,3,4], [5,6,6]])
    # The input-only reader also supports the full stated 15-strip budget,
    # odd lengths and width 27, beyond the sampler's usual 1..9 strips.
    wide = [np.tile(np.array([[3], [4]], dtype=np.int8), (1, 27))]
    wide += [np.full((2, 3), 3+i % 7, dtype=np.int8) for i in range(14)]
    placements = [(1, 1)] + [(v, 1+4*i) for v in (4, 7) for i in range(7)]
    target = _collapse(_stack(wide, 0))
    for axis in (0, 1):
        for bottom in (False, True):
            for right in (False, True):
                inp = _paint(wide, placements, (10, 29),
                             (axis, bottom, right), 0, 3)
                assert np.array_equal(solve(inp), target)
    # Every legal requested count, strict invalid types, and deterministic repeats.
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            p = generate_puzzle(204, a, b)
            validate_generated_puzzle(p)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4),
                 (True,1), (4,False), (4.0,1), (4,1.0), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Proper rotations of whole inputs leave normalized answers invariant.
    p = generate_puzzle(0, 2, 1)
    for pair in p["train"]+p["test"]:
        for k in range(4):
            assert np.array_equal(solve(np.rot90(pair["input"], k)), pair["output"])
        before = pair["input"].copy()
        solve(pair["input"])
        assert np.array_equal(before, pair["input"])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, global-vector invariants")
