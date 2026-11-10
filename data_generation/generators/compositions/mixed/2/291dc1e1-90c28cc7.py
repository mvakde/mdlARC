"""L-key assembly of enlarged mosaic fragments, then global two-axis collapse.

No parent generator is imported. Labels are sampled latent mosaics, while solve
recovers fragments and maximal runs solely from the supplied input pixels.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


# Fixed, input-readable roles; background is any color outside these three.
ORIGIN, FAST, SLOW = 0, 1, 2


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(train, test):
    if not _integer(train) or not 2 <= train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(test) or not 1 <= test <= 4 or train <= test:
        raise ValueError("num_test must be an integer in 1..4 smaller than num_train")
    return int(train), int(test)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30 and colors 0..9")
    return array.astype(np.int8, copy=False)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _runs(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def _collapse(mosaic):
    rows = np.r_[0, 1 + np.flatnonzero(np.any(mosaic[1:] != mosaic[:-1], axis=1))]
    cols = np.r_[0, 1 + np.flatnonzero(np.any(mosaic[:, 1:] != mosaic[:, :-1], axis=0))]
    return (mosaic[np.ix_(rows, cols)].copy(),
            np.diff(np.r_[rows, mosaic.shape[0]]),
            np.diff(np.r_[cols, mosaic.shape[1]]))


def _boundaries(grid):
    return (bool(np.all(np.any(grid[1:] != grid[:-1], axis=1)))
            and bool(np.all(np.any(grid[:, 1:] != grid[:, :-1], axis=0))))


def _key(grid):
    candidates = []
    for bottom, right in itertools.product((False, True), repeat=2):
        aligned = grid[::-1] if bottom else grid
        aligned = aligned[:, ::-1] if right else aligned
        if min(aligned.shape) < 5 or aligned[0, 0] != ORIGIN:
            continue
        h, v, bg = map(int, (aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
        if {h, v} != {FAST, SLOW} or bg in (ORIGIN, FAST, SLOW):
            continue
        if (np.all(aligned[0, 1:] == h) and np.all(aligned[1:, 0] == v)
                and np.all(aligned[-1, 1:] == bg) and np.all(aligned[1:, -1] == bg)):
            candidates.append((int(v == FAST), bottom, right, bg))
    if len(candidates) != 1:
        raise ValueError("Expected one L key: origin 0, fast arm 1, slow arm 2")
    return candidates[0]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    boxes = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, points = [start], [start]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = (np.max(points, axis=0) + 1).astype(int)
        if len(points) != (r1 - r0) * (c1 - c0):
            raise ValueError("Every payload component must be a filled rectangle")
        boxes.append((r0, int(r1), c0, int(c1)))
    return boxes


def _read(input_grid):
    grid = _grid(input_grid)
    axis, bottom, right, background = _key(grid)
    mask = grid != background
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _components(mask)
    if not boxes:
        raise ValueError("Missing mosaic fragments")
    height, width = grid.shape
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if not (0 < r0 < r1 < height and 0 < c0 < c1 < width):
            raise ValueError("Fragments need background margins outside the key")
        if axis:
            slow = width - c1 if right else c0
            fast = height - r1 if bottom else r0
            thickness = c1 - c0
        else:
            slow = height - r1 if bottom else r0
            fast = width - c1 if right else c0
            thickness = r1 - r0
        block = np.rot90(grid[r0:r1, c0:c1], rotation).copy()
        lanes.setdefault(slow, []).append((fast, thickness, block, box))
    ordered, groups, coordinates, ordered_boxes = [], [], [], []
    previous_end = None
    for slow in sorted(lanes):
        lane = sorted(lanes[slow], key=lambda item: item[0])
        if previous_end is not None and slow <= previous_end:
            raise ValueError("Lane bands must have a background gap and aligned near edges")
        previous_end = slow + max(item[1] for item in lane)
        group, last_end = [], None
        for fast, _, block, box in lane:
            if last_end is not None and fast <= last_end:
                raise ValueError("Fragments in a lane need background gaps")
            last_end = fast + block.shape[1]
            group.append(len(ordered))
            ordered.append(block)
            coordinates.append(fast)
            ordered_boxes.append(box)
        groups.append(group)
    if len({b.shape[1] for b in ordered}) != 1:
        raise ValueError("All normalized fragments must span the same mosaic width")
    mosaic = np.concatenate(ordered, axis=0)
    if max(mosaic.shape) > 30:
        raise ValueError("Assembled mosaic exceeds the family's 30-cell budget")
    return {"blocks": ordered, "groups": groups, "fast_positions": coordinates,
            "boxes": ordered_boxes, "mosaic": mosaic, "background": background,
            "configuration": (axis, bottom, right)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the key, rotate and stack full-width fragments, collapse global runs."""
    return _collapse(_read(input_grid)["mosaic"])[0]


def _unequal(first, second):
    return second is None or not np.array_equal(first, second)


def _stack_collapse(blocks):
    if not blocks or len({b.shape[1] for b in blocks}) != 1:
        return None
    return _collapse(np.concatenate(blocks, axis=0))[0]


def _alternatives(d):
    blocks, groups = d["blocks"], d["groups"]
    orders = {
        "reverse_fast": [i for group in groups for i in group[::-1]],
        "reverse_slow": [i for group in groups[::-1] for i in group],
        "fast_major": sorted(range(len(blocks)), key=lambda i: (d["fast_positions"][i], i)),
    }
    results = {name: _stack_collapse([blocks[i] for i in order])
               for name, order in orders.items()}
    results["reflect_rows"] = _stack_collapse([b[::-1] for b in blocks])
    results["reflect_columns"] = _stack_collapse([b[:, ::-1] for b in blocks])
    results["compress_fragments_first"] = _stack_collapse([_collapse(b)[0] for b in blocks])
    # This alternative compresses rows in each fragment but retains duplicates
    # across fragment joins; unlike _stack_collapse it must not collapse again.
    locally = [_collapse(b)[0] for b in blocks]
    results["local_then_concat"] = (np.concatenate(locally, axis=0)
                                    if len({b.shape[1] for b in locally}) == 1 else None)
    m = d["mosaic"]
    row_indices = np.r_[0, 1 + np.flatnonzero(np.any(m[1:] != m[:-1], axis=1))]
    col_indices = np.r_[0, 1 + np.flatnonzero(np.any(m[:, 1:] != m[:, :-1], axis=0))]
    results["row_only"] = m[row_indices, :]
    results["column_only"] = m[:, col_indices]
    results["no_collapse"] = m
    return results


def _hidden_seam(d):
    """A full-column boundary is visible in some fragments and invisible in others."""
    blocks = d["blocks"]
    for j in range(1, blocks[0].shape[1]):
        witnesses = [bool(np.any(b[:, j - 1] != b[:, j])) for b in blocks]
        if any(witnesses) and not all(witnesses):
            return True
    return False


def _cross_join(d):
    return any(np.array_equal(a[-1], b[0]) for a, b in zip(d["blocks"], d["blocks"][1:]))


def _separated_repeat(grid):
    return (len({_signature(row[None, :]) for row in grid}) < grid.shape[0]
            or len({_signature(col[:, None]) for col in grid.T}) < grid.shape[1])


def _witness(d, output):
    _, rows, cols = _collapse(d["mosaic"])
    alternatives = _alternatives(d)
    return (len(d["groups"]) >= 2 and max(map(len, d["groups"])) >= 2
            and _hidden_seam(d) and _cross_join(d) and _separated_repeat(output)
            and len(set(rows)) > 1 and len(set(cols)) > 1
            and all(_unequal(output, alternatives[name]) for name in
                    ("reverse_fast", "reverse_slow", "fast_major", "reflect_rows",
                     "reflect_columns", "local_then_concat")))


def _latent(rng, shape, background, witness):
    h, w = shape
    pool = np.array([c for c in range(10) if c != background], dtype=np.int8)
    for _ in range(300):
        size = int(rng.integers(2, min(9, h * w) + 1))
        palette = rng.choice(pool, size=size, replace=False)
        latent = rng.choice(palette, size=shape).astype(np.int8)
        if witness:
            # A separated repeated full row defeats global deduplication.
            a = int(rng.integers(h - 2))
            latent[a + 2] = latent[a]
            # One logical row reveals a boundary hidden throughout the others.
            j = int(rng.integers(w - 1))
            latent[:, j + 1] = latent[:, j]
            eligible = [r for r in range(h) if r not in (a, a + 2)]
            r = int(rng.choice(eligible))
            latent[r, j + 1] = rng.choice(palette[palette != latent[r, j]])
        if _boundaries(latent):
            return latent
    raise RuntimeError("Bounded latent sampling exhausted")


def _profile(rng, n, budget, uneven):
    for _ in range(100):
        mode = int(rng.integers(4))
        if mode == 0 and not uneven:
            values = np.full(n, int(rng.integers(1, max(2, budget // n + 1))))
        elif mode == 1:
            values = rng.integers(1, 3, size=n)
            values[int(rng.integers(n))] += int(rng.integers(2, 6))
        else:
            values = rng.integers(1, 5, size=n)
        if values.sum() <= budget and values.max() > 1 and (not uneven or len(set(values)) > 1):
            return values.astype(int)
    # Deterministic feasible fallback, not a semantic exception.
    values = np.ones(n, dtype=int)
    values[int(rng.integers(n))] = 2
    return values


def _partitions(total, cap):
    if total == 0:
        yield ()
    else:
        for first in range(1, min(cap, total) + 1):
            for rest in _partitions(total - first, cap):
                yield (first,) + rest


def _gaps(rng, occupied, count, extent):
    extra = extent - occupied - count - 1
    return 1 + rng.multinomial(extra, rng.dirichlet(np.ones(count + 1)))


def _paint(rng, blocks, counts, config, background):
    widths = [b.shape[1] for b in blocks]
    heights, start = [], 0
    for count in counts:
        heights.append(max(b.shape[0] for b in blocks[start:start + count]))
        start += count
    fast_min = max(count * (widths[0] + 1) + 1 for count in counts)
    slow_min = sum(heights) + len(counts) + 1
    fast_extent = int(rng.integers(fast_min, min(29, fast_min + 7) + 1))
    slow_extent = int(rng.integers(slow_min, min(29, slow_min + 7) + 1))
    slow_gaps = _gaps(rng, sum(heights), len(counts), slow_extent)
    interior = np.full((slow_extent, fast_extent), background, dtype=np.int8)
    axis, bottom, right = config
    reverse = (bottom != right) if axis == 0 else (bottom == right)
    start, v = 0, int(slow_gaps[0])
    for lane, count in enumerate(counts):
        fast_gaps = _gaps(rng, count * widths[0], count, fast_extent)
        u = int(fast_gaps[0])
        for k in range(count):
            block = blocks[start + k]
            h, w = block.shape
            interior[v:v + h, u:u + w] = block[::-1] if reverse else block
            u += w + int(fast_gaps[k + 1])
        v += heights[lane] + int(slow_gaps[lane + 1])
        start += count
    grid = np.full((slow_extent + 1, fast_extent + 1), background, dtype=np.int8)
    grid[1:, 1:] = interior
    grid[0, 0], grid[0, 1:], grid[1:, 0] = ORIGIN, FAST, SLOW
    if axis:
        grid = grid.T
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    return grid.copy()


def _sample_pair(rng, config, witness=False, exclude_shape=None):
    for _ in range(200):
        background = int(rng.integers(3, 10))
        h = int(rng.integers(3, 7) if witness else rng.integers(2, 8))
        w = int(rng.integers(3, 6) if witness else rng.integers(2, 8))
        if (h, w) == exclude_shape:
            continue
        latent = _latent(rng, (h, w), background, witness)
        rows = _profile(rng, h, 22, witness)
        cols = _profile(rng, w, 12 if witness else 26, witness)
        mosaic = np.repeat(np.repeat(latent, rows, axis=0), cols, axis=1)
        n = 4 if witness else int(rng.integers(2, min(7, mosaic.shape[0]) + 1))
        possible = np.arange(1, mosaic.shape[0])
        cuts = sorted(map(int, rng.choice(possible, size=n - 1, replace=False)))
        if witness and not any(np.array_equal(mosaic[c - 1], mosaic[c]) for c in cuts):
            continue
        blocks = list(np.split(mosaic, cuts, axis=0))
        cap = 28 // (mosaic.shape[1] + 1)
        layouts = []
        candidates = [(2, 2)] if witness else _partitions(n, cap)
        for counts in candidates:
            if max(counts) > cap:
                continue
            start, used_height = 0, len(counts) + 1
            for count in counts:
                used_height += max(b.shape[0] for b in blocks[start:start + count])
                start += count
            if used_height <= 29:
                layouts.append(counts)
        if not layouts:
            continue
        counts = layouts[int(rng.integers(len(layouts)))]
        grid = _paint(rng, blocks, counts, config, background)
        if witness and not _witness(_read(grid), latent):
            continue
        return {"input": grid, "output": latent.copy()}
    raise RuntimeError("Bounded fragment construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    first = (int(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2)))
    second = (1 - first[0], not first[1], not first[2])
    familiar = [first, second]
    held = [c for c in itertools.product((0, 1), (False, True), (False, True)) if c not in familiar]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            configs = familiar if split == "train" else held
            config = familiar[i] if split == "train" and i < 2 else configs[int(rng.integers(len(configs)))]
            for _ in range(32):
                exclude = puzzle["train"][0]["output"].shape if split == "train" and i == 1 else None
                pair = _sample_pair(rng, config, witness=(split == "train" and i < 2), exclude_shape=exclude)
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not obtain a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_data, test_configs = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = _grid(pair["input"]), _grid(pair["output"])
            signature = _signature(source)
            assert signature not in seen, "Duplicate input"
            seen.add(signature)
            d = _read(source)
            recovered, rows, cols = _collapse(d["mosaic"])
            assert np.array_equal(target, recovered), "Incorrect assembled and compressed label"
            assert d["background"] not in target and _boundaries(target)
            reconstructed = np.repeat(np.repeat(target, rows, axis=0), cols, axis=1)
            assert np.array_equal(reconstructed, d["mosaic"]), "Expansion is not lossless"
            assert max(rows) > 1 and max(cols) > 1, "Both axes must need collapse"
            assert 2 <= len(d["blocks"]) <= 7
            assert 2 <= min(target.shape) and max(target.shape) <= 7
            if split == "train":
                train_data.append((d, target))
            else:
                test_configs.append(d["configuration"])
    configs = {d["configuration"] for d, _ in train_data}
    assert {c[0] for c in configs} == {0, 1}
    assert {c[1] for c in configs} == {False, True}
    assert {c[2] for c in configs} == {False, True}
    assert all(c not in configs for c in test_configs), "Queries should recombine known direction factors"
    assert len({t.shape for _, t in train_data}) >= 2
    witnesses = [(d, t) for d, t in train_data if _witness(d, t)]
    assert {d["configuration"][0] for d, _ in witnesses} == {0, 1}, "Need both orientation demonstrations"
    for name in ("reverse_fast", "reverse_slow", "fast_major", "reflect_rows", "reflect_columns",
                 "local_then_concat", "row_only", "column_only", "no_collapse"):
        assert any(_unequal(t, _alternatives(d)[name]) for d, t in train_data), name
    # Ignoring the key and reading physical top-to-bottom/left-to-right rectangles.
    distinguished = False
    for d, target in train_data:
        order = sorted(range(len(d["blocks"])), key=lambda i: (d["boxes"][i][0], d["boxes"][i][2]))
        # Retain proper rotation here, making this an even stronger ordering-only shortcut.
        distinguished |= _unequal(target, _stack_collapse([d["blocks"][i] for i in order]))
    assert distinguished, "Physical raster order accidentally agrees throughout training"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: 42 puzzles held one at a time; no files or execution traces."""
    hashes, configs, layouts, input_shapes, output_shapes, palettes = (set() for _ in range(6))
    backgrounds, profiles, counts, max_pops = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_digest(puzzle))
        if seed == 0:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                assert np.array_equal(solve(source.copy()), target)
                d = _read(source)
                _, rows, cols = _collapse(d["mosaic"])
                configs.add(d["configuration"])
                counts.add(len(d["blocks"]))
                max_pops.add(max(map(len, d["groups"])))
                backgrounds.add(d["background"])
                profiles.add((tuple(rows), tuple(cols)))
                input_shapes.add(source.shape)
                output_shapes.add(target.shape)
                palettes.add(len(np.unique(target)))
                layouts.add((d["configuration"], source.shape, tuple(d["boxes"])))
    assert len(hashes) == 42 and len(configs) == 8
    assert len(input_shapes) >= 60 and len(output_shapes) >= 20
    assert len(profiles) >= 150 and len(layouts) >= 200
    assert len(backgrounds) == 7 and len(palettes) >= 6
    assert min(counts) == 2 and max(counts) == 7 and max(max_pops) >= 3
    # Every legal explicit count combination, plus noninteger and range failures.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(900 + 10 * train + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Proper rotation covariance across ALL frames, with asymmetric payloads.
    rng = np.random.default_rng(9921)
    pair = _sample_pair(rng, (0, False, False), witness=True)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], rotation)), pair["output"])
    # Separated repeats survive; adjacent duplicates alone collapse.
    m = np.array([[3, 3, 4, 4, 3], [3, 3, 4, 4, 3], [4, 4, 3, 3, 4],
                  [3, 3, 4, 4, 3]], dtype=np.int8)
    assert np.array_equal(_collapse(m)[0], np.array([[3, 4, 3], [4, 3, 4], [3, 4, 3]]))
    damaged = generate_puzzle(1024, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged label accepted")
