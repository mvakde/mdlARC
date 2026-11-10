"""Cyclic overprint loom: middle-reference winding, parity, and exact stamping.

All computation is bounded by 30 x 30 grids. Generation and self tests do not
write files, cache puzzles, load parent code, or retain execution traces.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a.astype(np.int8, copy=False)


def _box(binary):
    cells = np.argwhere(binary)
    assert len(cells), "empty encoding"
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _crop(a, b):
    return a[b[0]:b[1], b[2]:b[3]]


def _separated(a, b):
    # Half-open bounds: strict inequality leaves at least one blank line.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read_input(value):
    """Recover every convention from cells, without training pairs or metadata."""
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max()) * 2 > grid.size, "background must be a strict majority"
    candidates = []
    for signal in map(int, colors):
        if signal == bg:
            continue
        other = (grid != bg) & (grid != signal)
        if len(set(map(int, grid[other]))) < 2:
            continue
        rb, sb = _box(grid == signal), _box(other)
        stamp = _crop(grid, sb).copy()
        if not _separated(rb, sb) or not all(2 <= n <= 5 for n in stamp.shape):
            continue
        if signal in stamp or not np.all(np.isin(_crop(grid, rb), (bg, signal))):
            continue
        if not all(np.any(side != bg) for side in
                   (stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1])):
            continue
        coords = np.argwhere(grid == signal)
        for horizontal in (True, False):
            transverse, axial = (0, 1) if horizontal else (1, 0)
            keys = sorted(set(map(int, coords[:, transverse])))
            if len(keys) not in (3, 5, 7) or min(np.diff(keys)) < 2:
                continue
            groups = [np.sort(coords[coords[:, transverse] == key, axial]) for key in keys]
            if any(np.any(np.diff(g) != 1) for g in groups):
                continue
            common_min = len({int(g[0]) for g in groups}) == 1
            common_max = len({int(g[-1]) for g in groups}) == 1
            if common_min == common_max:
                continue
            lengths = tuple(len(g) for g in groups)
            period = lengths[len(keys) // 2]
            span = keys[-1] - keys[0] + 1
            coarse_shape = (span, period) if horizontal else (period, span)
            if not 2 <= period <= 6 or any(a * b > 30 for a, b in zip(coarse_shape, stamp.shape)):
                continue
            candidates.append({
                "background": bg, "signal": signal, "stamp": stamp,
                "rail_box": rb, "stamp_box": sb, "horizontal": horizontal,
                "positive": common_min, "keys": tuple(keys),
                "positions": tuple(k - keys[0] for k in keys),
                "lengths": lengths, "period": period, "span": span,
                "anchor": int(groups[0][0] if common_min else groups[0][-1]),
            })
    assert len(candidates) == 1, "need one uniquely readable rail array and stamp"
    return candidates[0]


def _mask(info, period=None, mode="parity"):
    """Closed-form occupancy, separate from generation's per-cell toggles."""
    period = info["period"] if period is None else int(period)
    assert period >= 1
    lengths = np.asarray(info["lengths"], dtype=int)
    k = np.arange(period)[None, :]
    q, r = np.divmod(lengths[:, None], period)
    if mode == "parity":
        strips = (q + (k < r)) % 2 == 1
    elif mode == "union":
        strips = (q > 0) | (k < r)
    elif mode == "clip":
        strips = k < np.minimum(lengths[:, None], period)
    elif mode == "remainder":
        strips = k < r
    elif mode == "normalize":
        strips = np.ones((len(lengths), period), dtype=bool)
    elif mode == "fold":
        strips = np.zeros((len(lengths), period), dtype=bool)
        for i, length in enumerate(lengths):
            for step in range(int(length)):
                lap, slot = divmod(step, period)
                strips[i, slot if lap % 2 == 0 else period - 1 - slot] ^= True
    else:
        raise ValueError("unknown occupancy convention")
    if not info["positive"]:
        strips = strips[:, ::-1]
    mask = np.zeros((info["span"], period), dtype=bool)
    mask[list(info["positions"])] = strips
    return mask if info["horizontal"] else mask.T


def _expand(mask, stamp, bg):
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    return np.where(mask[rr // sh, cc // sw], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Wind each rail in the middle rail's period; print odd-covered slots."""
    info = _read_input(input_grid)
    return _expand(_mask(info), info["stamp"], info["background"])


def _draw_output(positions, lengths, direction, stamp, bg):
    """Construct ground truth by visiting and toggling every source rail cell."""
    horizontal = direction[0] == 0
    positive = max(direction) > 0
    period, span = lengths[len(lengths) // 2], positions[-1] + 1
    shape = (span, period) if horizontal else (period, span)
    mask = np.zeros(shape, dtype=bool)
    for position, length in zip(positions, lengths):
        for step in range(length):
            slot = step % period
            if not positive:
                slot = period - 1 - slot
            cell = (position, slot) if horizontal else (slot, position)
            mask[cell] ^= True
    sh, sw = stamp.shape
    output = np.full((shape[0] * sh, shape[1] * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = stamp
    return output


def _stamp(rng, shape, bg, payload, witness):
    h, w = shape
    rr, cc = np.indices(shape)
    family = str(rng.choice(("random", "bands", "rings", "diagonal", "quarters")))
    if family == "bands":
        labels = rr if rng.integers(2) else cc
    elif family == "rings":
        labels = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc))
    elif family == "diagonal":
        labels = rr + cc * int(rng.integers(1, 4))
    elif family == "quarters":
        labels = (rr >= h // 2) + 2 * (cc >= w // 2)
    else:
        labels = rng.integers(len(payload), size=shape)
    stamp = np.asarray(payload, dtype=np.int8)[labels % len(payload)].copy()
    if witness == 0:
        # A single edge hole fixes the exact crop and micro-orientation.
        stamp[0, 0] = bg
        stamp[-1, -1] = payload[0]
        stamp[0, -1] = payload[1]
    elif witness == 1:
        stamp[1, 1] = bg
        stamp[0, 0], stamp[-1, -1] = payload[0], payload[1]
    else:
        holes = rng.random(shape) < float(rng.choice((0, .08, .2, .35)))
        # Two opposing corners make every crop side observable.
        holes[0, 0] = holes[-1, -1] = False
        stamp[holes] = bg
        stamp[0, 0], stamp[-1, -1] = payload[0], payload[1]
    return stamp


def _gaps(rng, count, witness):
    if witness == 0:
        return [5, 2, 2, 2] if rng.integers(2) else [2, 2, 2, 5]
    if witness == 1:
        return [int(rng.integers(2, 5))] * 2
    if rng.random() < .4:
        return [int(rng.integers(2, 14 // (count - 1) + 1))] * (count - 1)
    gaps = [2] * (count - 1)
    for _ in range(int(rng.integers(15 - 2 * (count - 1)))):
        gaps[int(rng.integers(count - 1))] += 1
    return gaps


def _lengths(rng, count, period, witness, test):
    others = [i for i in range(count) if i != count // 2]
    rng.shuffle(others)
    if witness == 0:
        lengths = [int(rng.integers(1, period)) for _ in range(count)]
        lengths[others[0]] = period + int(rng.integers(1, period))
    elif witness == 1:
        lengths = [period] * count
        lengths[others[0]] = 2 * period
        lengths[others[1]] = 2 * period + int(rng.integers(1, period))
    else:
        ceiling = min(29, (5 if test else 3) * period - 1)
        lengths = [int(rng.integers(1, ceiling + 1)) for _ in range(count)]
        if test:
            choices = [q * period + r for q in (3, 4) for r in range(1, period)
                       if q * period + r <= 29]
            lengths[others[0]] = int(rng.choice(choices))
            lengths[others[1]] = int(rng.integers(1, period))
        elif all(length == period for length in lengths):
            lengths[others[0]] = 1
    lengths[count // 2] = period
    return lengths


def _canvas(rng, rails, stamp, bg):
    rh, rw = rails.shape
    sh, sw = stamp.shape
    fg = int(np.count_nonzero(rails != bg) + np.count_nonzero(stamp != bg))
    options = []
    for side in range(4):  # stamp right, below, left, above
        for gap in range(1, 6):
            h, w = ((max(rh, sh), rw + sw + gap) if side % 2 == 0
                    else (rh + sh + gap, max(rw, sw)))
            if h <= 30 and w <= 30:
                options.append((side, gap, h, w))
    assert options
    side, gap, base_h, base_w = options[int(rng.integers(len(options)))]
    pad_cap = int(rng.choice((2, 6, 30)))
    h = base_h + int(rng.integers(min(pad_cap, 30 - base_h) + 1))
    w = base_w + int(rng.integers(min(pad_cap, 30 - base_w) + 1))
    while h * w <= 2 * fg:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
        assert h <= 30 and w <= 30
    if side % 2 == 0:
        start = int(rng.integers(w - base_w + 1))
        rc, sc = (start, start + rw + gap) if side == 0 else (start + sw + gap, start)
        rr, sr = int(rng.integers(h - rh + 1)), int(rng.integers(h - sh + 1))
    else:
        start = int(rng.integers(h - base_h + 1))
        rr, sr = (start, start + rh + gap) if side == 1 else (start + sh + gap, start)
        rc, sc = int(rng.integers(w - rw + 1)), int(rng.integers(w - sw + 1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[rr:rr + rh, rc:rc + rw] = rails
    grid[sr:sr + sh, sc:sc + sw] = stamp
    return grid


def _pair(rng, period, count, direction, palette, witness, test):
    bg, signal = int(palette[0]), int(palette[1])
    gaps = _gaps(rng, count, witness)
    positions = [0] + list(map(int, np.cumsum(gaps)))
    span = positions[-1] + 1
    horizontal = direction[0] == 0
    trans_max = min(5, 30 // span)
    trans_size = 2 if witness == 0 else 3 if witness == 1 else int(rng.integers(2, trans_max + 1))
    axial_min = 3 if witness in (0, 1) else 2
    axial_size = int(rng.integers(axial_min, min(5, 30 // period) + 1))
    shape = (trans_size, axial_size) if horizontal else (axial_size, trans_size)
    payload_count = int(rng.integers(2, min(6, shape[0] * shape[1] - 1) + 1))
    stamp = _stamp(rng, shape, bg, palette[2:2 + payload_count], witness)
    lengths = _lengths(rng, count, period, witness, test)
    longest = max(lengths)
    rails = np.full((span, longest) if horizontal else (longest, span), bg, dtype=np.int8)
    positive = max(direction) > 0
    for position, length in zip(positions, lengths):
        for step in range(length):
            axial = step if positive else longest - 1 - step
            rails[(position, axial) if horizontal else (axial, position)] = signal
    return {"input": _canvas(rng, rails, stamp, bg),
            "output": _draw_output(positions, lengths, direction, stamp, bg)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_direction = int(rng.integers(4))
    mixed = bool(rng.random() < .7)
    stable_palette = rng.permutation(10)
    stable = bool(rng.random() < .2)
    first_periods = list(map(int, rng.choice(np.arange(2, 7), size=2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        previous = None
        for index in range(count):
            witness = index if split == "train" and index < 2 else -1
            period = first_periods[index] if witness >= 0 else int(rng.integers(2, 7))
            rails = 5 if witness == 0 else 3 if witness == 1 else int(rng.choice((3, 5, 7)))
            if split == "test" and index == 1:
                period = int(rng.choice([p for p in range(2, 7) if p != previous[0]]))
                rails = int(rng.choice([n for n in (3, 5, 7) if n != previous[1]]))
            d = base_direction
            if mixed:
                d = (base_direction + 1) % 4 if witness == 1 else int(rng.integers(4))
            palette = stable_palette if stable else rng.permutation(10)
            for _ in range(32):
                pair = _pair(rng, period, rails, _DIRECTIONS[d], palette, witness, split == "test")
                key = _key(pair["input"])
                if key in seen:
                    continue
                # Assert uniqueness of the visible encoding before accepting it.
                _read_input(pair["input"])
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("failed to generate a distinct input")
            previous = period, rails
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _selection_period(info, selector):
    lengths = info["lengths"]
    if selector == "median":
        return sorted(lengths)[len(lengths) // 2]
    if selector == "shortest":
        return min(lengths)
    if selector == "longest":
        return max(lengths)
    if selector == "first":
        return lengths[0]
    if selector == "last":
        return lengths[-1]
    if selector == "geometric":
        keys = info["keys"]
        distances = [abs(2 * k - keys[0] - keys[-1]) for k in keys]
        return lengths[distances.index(min(distances))]
    return info["period"]


def _geometric_witness(info):
    keys = info["keys"]
    ds = [abs(2 * k - keys[0] - keys[-1]) for k in keys]
    return ds.count(min(ds)) == 1 and _selection_period(info, "geometric") != info["period"]


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(inp)
            mask = _mask(info)
            expected = _expand(mask, info["stamp"], info["background"])
            assert np.array_equal(out, expected), "incorrect cyclic overprint"
            sh, sw = info["stamp"].shape
            bg, signal = info["background"], info["signal"]
            assert signal not in out, "instruction ink survives"
            assert np.count_nonzero(out != bg) == int(mask.sum()) * np.count_nonzero(info["stamp"] != bg)
            # Every block is an exact tile or entirely blank, including edge holes.
            for r, c in np.ndindex(mask.shape):
                block = out[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                assert np.array_equal(block, info["stamp"]) if mask[r, c] else np.all(block == bg)
            assert not np.array_equal(inp, out)
            infos[split].append(info)
    train = infos["train"]
    assert len({i["period"] for i in train}) >= 2, "fixed-period shortcut"
    assert len({len(i["lengths"]) for i in train}) >= 2, "fixed-count shortcut"
    assert len({i["stamp"].shape for i in train}) >= 2, "fixed-tile-size shortcut"
    assert any(_geometric_witness(i) for i in train), "ordinal vs geometric evidence"
    assert any(_selection_period(i, "median") != i["period"] for i in train)
    assert any(np.any(i["stamp"] == i["background"]) for i in train)
    assert any(2 * i["period"] in i["lengths"] for i in train), "whole-lap cancellation evidence"
    assert any(any(2 * i["period"] < n < 3 * i["period"] for n in i["lengths"]) for i in train)
    assert all(max(i["lengths"]) < 3 * i["period"] for i in train)
    assert all(max(i["lengths"]) > 3 * i["period"] for i in infos["test"]), "missing lap extrapolation"
    # These are specific alternative rules, not claims of independent inference.
    for selector in ("median", "shortest", "longest", "first", "last", "geometric"):
        assert any(not np.array_equal(_mask(i), _mask(i, _selection_period(i, selector))) for i in train), selector
    for mode in ("union", "clip", "remainder", "normalize", "fold"):
        assert any(not np.array_equal(_mask(i), _mask(i, mode=mode)) for i in train), mode
    # Omitting A loses the actual tile; omitting B leaves the original long mask.
    assert any(len(set(map(int, i["stamp"].ravel())) - {i["background"]}) >= 2 for i in train)
    assert any(max(i["lengths"]) != i["period"] for i in train)


def _canonical_hash(grid, bg):
    mapping = {int(bg): 0}
    canonical = bytes(mapping.setdefault(int(v), len(mapping)) for v in grid.ravel())
    return (grid.shape, hashlib.sha256(canonical).hexdigest())


def self_test():
    """Stream 42 seeds and small metamorphic/count checks; no filesystem writes."""
    counts = Counter()
    features = {k: set() for k in ("directions", "periods", "counts", "backgrounds",
                                  "input_shapes", "output_shapes", "stamp_shapes",
                                  "masks", "stamps", "puzzles", "test_laps")}
    changed_reference = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], again[s]) for k in ("input", "output"))
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                i = _read_input(inp)
                mask = _mask(i)
                bg, period = i["background"], i["period"]
                counts[split] += 1
                features["directions"].add((i["horizontal"], i["positive"]))
                features["periods"].add(period)
                features["counts"].add(len(i["lengths"]))
                features["backgrounds"].add(bg)
                features["input_shapes"].add(inp.shape)
                features["output_shapes"].add(out.shape)
                features["stamp_shapes"].add(i["stamp"].shape)
                features["masks"].add(_canonical_hash(mask.astype(np.int8), 0))
                features["stamps"].add(_canonical_hash(i["stamp"], bg))
                digest.update(repr(inp.shape).encode())
                digest.update(inp.tobytes())
                counts["unequal_gaps"] += len(set(np.diff(i["keys"]))) > 1
                counts["empty_rail"] += any(n % (2 * period) == 0 for n in i["lengths"])
                counts["nonmedian"] += _selection_period(i, "median") != period
                if split == "test":
                    features["test_laps"].update(n // period for n in i["lengths"] if n // period >= 3)
                    # The period is visible: trimming only its terminal cell changes the answer.
                    if period > 2:
                        altered = inp.copy()
                        key = i["keys"][len(i["keys"]) // 2]
                        end = i["anchor"] + (period - 1) * (1 if i["positive"] else -1)
                        altered[(key, end) if i["horizontal"] else (end, key)] = bg
                        assert _read_input(altered)["period"] == period - 1
                        assert not np.array_equal(solve(altered), out)
                        changed_reference += 1
        features["puzzles"].add(digest.hexdigest())
        # A copied tile cell affects every active placement, and no other pixels.
        pair = puzzle["train"][0]
        info = _read_input(pair["input"])
        changed = pair["input"].copy()
        sb = info["stamp_box"]
        p = next(tuple(map(int, p)) for p in np.argwhere(info["stamp"] != info["background"]))
        new = next(c for c in range(10) if c not in set(map(int, np.unique(changed))))
        changed[sb[0] + p[0], sb[2] + p[1]] = new
        assert np.count_nonzero(solve(changed) != pair["output"]) == int(_mask(info).sum())
        # Palette roles and full geometry commute with all eight square symmetries.
        permutation = np.random.default_rng(1000 + seed).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        for turns in range(4):
            for flip in (False, True):
                a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                if flip:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(a), b)
    assert len(features["puzzles"]) == 42
    assert len(features["directions"]) == 4
    assert features["periods"] == set(range(2, 7))
    assert features["counts"] == {3, 5, 7}
    assert features["backgrounds"] == set(range(10))
    assert features["test_laps"] == {3, 4}
    assert len(features["input_shapes"]) >= 35 and len(features["output_shapes"]) >= 35
    assert len(features["stamp_shapes"]) >= 10 and len(features["masks"]) >= 100
    assert len(features["stamps"]) >= 100 and changed_reference >= 20
    # Exercise every input-length/period pair, including many more laps than
    # the sampler uses and a full 30-cell source axis. Keep only one at a time.
    boundary_cases = 0
    stamp = np.array([[2, 3, 2], [2, 0, 3], [3, 2, 0]], dtype=np.int8)
    for period in range(2, 7):
        for length in range(1, 31):
            for direction in _DIRECTIONS:
                lengths, positions = [length, period, period + 1], [0, 2, 4]
                horizontal, positive = direction[0] == 0, max(direction) > 0
                longest = max(lengths)
                rails = np.zeros((5, longest) if horizontal else (longest, 5), dtype=np.int8)
                for position, n in zip(positions, lengths):
                    for step in range(n):
                        axial = step if positive else longest - 1 - step
                        rails[(position, axial) if horizontal else (axial, position)] = 1
                inp = _canvas(np.random.default_rng(boundary_cases), rails, stamp, 0)
                expected = _draw_output(positions, lengths, direction, stamp, 0)
                assert np.array_equal(solve(inp), expected)
                boundary_cases += 1
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(271828, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: a.copy() for k, a in p.items()} for p in puzzle[s]] for s in puzzle}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "pairs": dict(counts), "diversity": {k: len(v) for k, v in features.items()},
            "reference_mutations": changed_reference, "palette_checks": 42, "symmetry_checks": 336,
            "boundary_cases": boundary_cases, "independent_solver": False}
