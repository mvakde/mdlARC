"""Correlate scattered, incomplete stratigraphic samples into a shared atlas.

Only Python's standard library and NumPy are required.  The solver sees no
generation metadata: background, rectangles, local orders, and consensus are
all recovered from the input.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _signature(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _puzzle_signature(puzzle):
    return tuple(_signature(pair[role]) for split in ("train", "test")
                 for pair in puzzle[split] for role in ("input", "output"))


def _collapse(line):
    return tuple(int(line[i]) for i in range(len(line))
                 if i == 0 or line[i] != line[i - 1])


def _components(mask):
    """Four-connected components of all non-background cells, color ignored."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, points = [(y, x)], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(points)
    return result


def _local_order(tile):
    candidates = []
    for axis in (0, 1):
        lines = tile.T if axis == 0 else tile
        sequences = [_collapse(line) for line in lines]
        sequence = sequences[0]
        if (len(sequence) >= 2 and len(sequence) == len(set(sequence))
                and set(sequence) == set(int(c) for c in np.unique(tile))
                and all(s == sequence for s in sequences)):
            candidates.append((axis, sequence))
    if len(candidates) != 1:
        raise ValueError("Sample must have exactly one common-transect axis")
    return candidates[0]


def _consensus(sequences):
    colors = set().union(*(set(s) for s in sequences))
    edges = {c: set() for c in colors}
    indegree = {c: 0 for c in colors}
    for sequence in sequences:
        for a, b in zip(sequence, sequence[1:]):
            if b not in edges[a]:
                edges[a].add(b)
                indegree[b] += 1
    result = []
    while indegree:
        available = [c for c, degree in indegree.items() if degree == 0]
        if len(available) != 1:
            raise ValueError("Sample orders are contradictory or ambiguous")
        c = available[0]
        result.append(c)
        del indegree[c]
        for following in edges[c]:
            indegree[following] -= 1
    return tuple(result)


def _analyze(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (np.all(grid[0] == background) and np.all(grid[-1] == background)
            and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)):
        raise ValueError("The outer border must expose the common background")
    components = _components(grid != background)
    if not 2 <= len(components) <= 5:
        raise ValueError("Expected two to five isolated rectangular samples")
    samples = []
    for points in components:
        ys, xs = zip(*points)
        y, x, bottom, right = min(ys), min(xs), max(ys) + 1, max(xs) + 1
        tile = grid[y:bottom, x:right]
        if len(points) != tile.size or min(tile.shape) < 2:
            raise ValueError("Samples must be filled rectangles at least 2 by 2")
        axis, sequence = _local_order(tile)
        # Adjacent transects must overlap in every color, so strata are connected.
        lines = tile.T if axis == 0 else tile
        intervals = []
        for line in lines:
            cuts = np.r_[0, np.flatnonzero(line[1:] != line[:-1]) + 1, len(line)]
            intervals.append(cuts)
        for before, after in zip(intervals, intervals[1:]):
            if np.any(np.maximum(before[:-1], after[:-1])
                      >= np.minimum(before[1:], after[1:])):
                raise ValueError("Each stratum must overlap across neighboring transects")
        samples.append({"box": (y, x, bottom, right), "tile": tile.copy(),
                        "axis": axis, "sequence": sequence,
                        "cuts": np.asarray(intervals)})
    axes = {s["axis"] for s in samples}
    if len(axes) != 1:
        raise ValueError("All samples must share the same stacking axis")
    axis = axes.pop()
    transverse = 1 - axis
    samples.sort(key=lambda s: s["box"][transverse])
    for left, right in zip(samples, samples[1:]):
        if left["box"][transverse + 2] >= right["box"][transverse]:
            raise ValueError("Sample lanes must have at least one background cell between them")
    order = _consensus([s["sequence"] for s in samples])
    if not 3 <= len(order) <= 9 or any(len(s["sequence"]) == len(order) for s in samples):
        raise ValueError("Expected 3..9 strata with every sample incomplete")
    return {"background": background, "axis": axis, "order": order, "samples": samples}


def _chart(analysis):
    """Canonical vertical-stack chart, transposed for horizontal strata."""
    samples, order = analysis["samples"], analysis["order"]
    chart = np.full((len(order), 2 * len(samples) - 1),
                    analysis["background"], dtype=np.int8)
    index = {color: i for i, color in enumerate(order)}
    for lane, sample in enumerate(samples):
        for color in sample["sequence"]:
            chart[index[color], 2 * lane] = color
    return chart if analysis["axis"] == 0 else chart.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the atlas entirely from the supplied grid."""
    return _chart(_analyze(input_grid))


def _sample_records(rng, layers, count, skipping):
    """Cover every consecutive global pair without a complete reference core."""
    for _ in range(1500):
        records = []
        for _ in range(count):
            if skipping:
                flags = rng.random(layers) < float(rng.uniform(.48, .84))
                record = tuple(int(i) for i in np.flatnonzero(flags))
            else:
                start = int(rng.integers(0, layers - 1))
                stop = int(rng.integers(start + 2, layers + 1))
                record = tuple(range(start, stop))
            if not 2 <= len(record) < layers:
                break
            records.append(record)
        if len(records) != count:
            continue
        edges = {(a, b) for r in records for a, b in zip(r, r[1:])}
        if not all((i, i + 1) in edges for i in range(layers - 1)):
            continue
        if skipping and not any(any(b > a + 1 for a, b in zip(r, r[1:])) for r in records):
            continue
        # Sample reflection must be observable, even if some records repeat.
        if records == records[::-1]:
            continue
        rng.shuffle(records)
        if records != records[::-1]:
            return records
    raise RuntimeError("Could not sample a uniquely ordered partial-record cover")


def _stratified_tile(rng, sequence, force_jagged=False):
    count = len(sequence)
    transverse = int(rng.integers(2, 6))
    length = count + int(rng.integers(2, 8))
    style = int(rng.integers(4))  # straight, plateau, directional, rough
    if force_jagged:
        style = 3
    # Start with a positive width for every band; spare thickness is arbitrary.
    widths = np.ones(count, dtype=int) + rng.multinomial(
        length - count, rng.dirichlet(np.ones(count)))
    cuts = np.cumsum(widths)[:-1]
    profiles = [cuts.copy()]
    direction = int(rng.choice((-1, 1)))
    for _ in range(1, transverse):
        old = cuts.copy()
        if style and (force_jagged or rng.random() < (.3 if style == 1 else .85)):
            for k in rng.permutation(count - 1):
                directions = (direction, -direction) if style == 2 else tuple(rng.permutation((-1, 1)))
                for delta in directions:
                    candidate = cuts.copy()
                    candidate[k] += delta
                    before, after = np.r_[0, old, length], np.r_[0, candidate, length]
                    if (np.all(np.diff(after) > 0)
                            and np.all(np.maximum(before[:-1], after[:-1])
                                       < np.minimum(before[1:], after[1:]))):
                        cuts = candidate
                        break
                if not np.array_equal(cuts, old):
                    break
        profiles.append(cuts.copy())
    tile = np.empty((length, transverse), dtype=np.int8)
    for column, profile in enumerate(profiles):
        ends = [0, *profile, length]
        for i, color in enumerate(sequence):
            tile[ends[i]:ends[i + 1], column] = color
    return tile


def _layout(rng, tiles, background):
    """Place cores in disjoint transverse lanes, independently staggered."""
    for _ in range(100):
        gaps = rng.integers(1, 4, size=len(tiles) - 1)
        left, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        width = left + right + sum(t.shape[1] for t in tiles) + int(gaps.sum())
        if width <= 30:
            break
    else:
        # A compact two-cell transverse footprint always fits five samples.
        tiles = [tile[:, :2].copy() for tile in tiles]
        gaps = np.ones(len(tiles) - 1, dtype=int)
        left = right = 1
        width = sum(t.shape[1] for t in tiles) + len(tiles) + 1
    offsets = rng.integers(0, 9, size=len(tiles))
    offsets -= offsets.min()
    top, bottom = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    height = top + max(int(offset) + tile.shape[0] for offset, tile in zip(offsets, tiles)) + bottom
    # Free exterior padding varies shape independently of the layer axis.
    width += int(rng.integers(0, min(4, 30 - width) + 1))
    height += int(rng.integers(0, min(4, 30 - height) + 1))
    grid = np.full((height, width), background, dtype=np.int8)
    column = left
    for i, (tile, offset) in enumerate(zip(tiles, offsets)):
        row = top + int(offset)
        grid[row:row + tile.shape[0], column:column + tile.shape[1]] = tile
        column += tile.shape[1] + (int(gaps[i]) if i < len(gaps) else 0)
    return grid


def _pair(rng, layers, count, axis, skipping):
    for _ in range(100):
        background = int(rng.integers(10))
        pool = np.array([c for c in range(10) if c != background])
        order = tuple(int(c) for c in rng.choice(pool, size=layers, replace=False))
        if order == tuple(sorted(order)) or order == tuple(sorted(order, reverse=True)):
            continue
        records = _sample_records(rng, layers, count, skipping)
        sequences = [tuple(order[i] for i in record) for record in records]
        tiles = [_stratified_tile(rng, s, force_jagged=(i == 0))
                 for i, s in enumerate(sequences)]
        grid = _layout(rng, tiles, background)
        if axis == 1:
            grid = grid.T.copy()
        analysis = _analyze(grid)
        # Reject a misleading accidental raster-order solution in the demos
        # and tests alike.  This does not affect the input-only rule.
        raster = tuple(dict.fromkeys(int(c) for c in grid.flat if c != background))
        if raster == order:
            continue
        if not any(np.any(np.ptp(s["cuts"], axis=0)) for s in analysis["samples"]):
            continue
        # Construct the target from the sampled abstract records, separately
        # from the solver's rectangle parsing and topological reconstruction.
        expected = np.full((layers, 2 * count - 1), background, dtype=np.int8)
        for lane, record in enumerate(records):
            for i in record:
                expected[i, 2 * lane] = order[i]
        if axis == 1:
            expected = expected.T.copy()
        _require(np.array_equal(expected, _chart(analysis)), "Construction disagrees with visible strata")
        return {"input": grid, "output": expected}
    raise RuntimeError("Could not construct an informative sample atlas")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"Illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_layers = int(rng.integers(3, 7))
    second_layers = first_layers + 1
    first_axis = int(rng.integers(2))
    first_count = int(rng.integers(2, 5))
    second_count = int(rng.choice([n for n in range(3, 6) if n != first_count]))
    specs = [(first_layers, first_count, first_axis, False),
             (second_layers, second_count, 1 - first_axis, True)]
    while len(specs) < num_train:
        skipping = bool(rng.integers(2))
        layers = int(rng.integers(4 if skipping else 3, second_layers + 1))
        count = int(rng.integers(3 if skipping else 2, 6))
        specs.append((layers, count, int(rng.integers(2)), skipping))
    rng.shuffle(specs)
    test_specs = [(int(rng.integers(second_layers + 1, min(9, second_layers + 2) + 1)),
                   int(rng.integers(3, 6)), int(rng.integers(2)), True)
                  for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, split_specs in (("train", specs), ("test", test_specs)):
        for spec in split_specs:
            for _ in range(40):
                pair = _pair(rng, *spec)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_interior_gap(analysis):
    positions = {c: i for i, c in enumerate(analysis["order"])}
    return any(any(positions[b] > positions[a] + 1
                   for a, b in zip(s["sequence"], s["sequence"][1:]))
               for s in analysis["samples"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Invalid example counts")
    analyses, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(grid)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            info = _analyze(grid)
            expected = _chart(info)
            _require(np.array_equal(output, expected), "Incorrect correlated atlas")
            canonical = output if info["axis"] == 0 else output.T
            _require(np.all(canonical[:, 1::2] == info["background"]), "Separators must be background")
            _require(set(int(c) for c in np.unique(output)) == set(int(c) for c in np.unique(grid)),
                     "Atlas must retain all strata and the common background")
            # A: transverse reflection changes lanes, not stratigraphic order.
            flipped_input = np.flip(grid, axis=1 - info["axis"])
            _require(np.array_equal(solve(flipped_input), np.flip(output, axis=1 - info["axis"])),
                     "Sample spatial order was lost")
            _require(not np.array_equal(output, np.flip(output, axis=1 - info["axis"])),
                     "Spatial-order witness must be observable")
            # B: even integer stretching cannot change the atlas.
            if 2 * grid.shape[info["axis"]] <= 30:
                _require(np.array_equal(solve(np.repeat(grid, 2, axis=info["axis"])), output),
                         "Layer thickness must not affect the atlas")
            # Local compression plus top alignment must be insufficient.
            naive = np.full_like(canonical, info["background"])
            for lane, sample in enumerate(info["samples"]):
                naive[:len(sample["sequence"]), 2 * lane] = sample["sequence"]
            _require(not np.array_equal(naive, canonical), "Registration witness missing")
            raster = tuple(dict.fromkeys(int(c) for c in grid.flat if c != info["background"]))
            _require(raster != info["order"], "Raster shortcut witness missing")
            analyses[split].append(info)
    train = analyses["train"]
    _require({i["axis"] for i in train} == {0, 1}, "Training must expose both axes")
    _require(len({len(i["order"]) for i in train}) >= 2, "Training must expose union layer count")
    _require(len({len(i["samples"]) for i in train}) >= 2, "Training must expose sample multiplicity")
    _require(any(_has_interior_gap(i) for i in train), "Training must expose internal missing horizons")
    _require(any(not _has_interior_gap(i) for i in train), "Training needs an accessible contiguous-window example")
    _require(all(len(i["order"]) > max(len(t["order"]) for t in train)
                 for i in analyses["test"]), "Tests should extend the demonstrated consensus length")
    _require(all(_has_interior_gap(i) for i in analyses["test"]), "Tests must use distributed skipping records")


def self_test():
    """Bounded 42-seed family, count, and metamorphic checks; returns metrics."""
    shapes, outputs, palettes, records, geometries, normalized, signatures = (set() for _ in range(7))
    axes, layers, sample_counts, backgrounds = set(), set(), set(), set()
    repeated = jagged = straight = internal = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                info = _analyze(grid)
                _require(np.array_equal(solve(grid), output), "Solver mismatch")
                _require(np.array_equal(solve(grid.T), output.T), "Transpose equivariance failed")
                # Arbitrary one-to-one color reassignment must preserve structure.
                mapping = np.random.default_rng(seed + 700).permutation(10)
                _require(np.array_equal(solve(mapping[grid]), mapping[output]), "Palette equivariance failed")
                # Replace every irregular, thick core by its thinnest straight
                # version.  Reposition it too: only its lane order survives.
                thin = np.full((max(len(s["sequence"]) for s in info["samples"]) + 2,
                                3 * len(info["samples"]) + 1), info["background"], dtype=np.int8)
                for lane, sample in enumerate(info["samples"]):
                    sequence = np.asarray(sample["sequence"], dtype=np.int8)
                    thin[1:len(sequence) + 1, 1 + 3 * lane:3 + 3 * lane] = sequence[:, None]
                if info["axis"] == 1:
                    thin = thin.T.copy()
                _require(np.array_equal(solve(thin), output), "Thickness/roughness/spacing invariance failed")
                shapes.add(grid.shape)
                outputs.add(output.shape)
                palettes.add((info["background"], info["order"]))
                axes.add((split, info["axis"]))
                layers.add(len(info["order"]))
                sample_counts.add(len(info["samples"]))
                backgrounds.add(info["background"])
                seqs = [s["sequence"] for s in info["samples"]]
                records.add(tuple(tuple(info["order"].index(c) for c in s) for s in seqs))
                repeated += len(set(seqs)) < len(seqs)
                internal += _has_interior_gap(info)
                geometries.add(tuple(s["box"] for s in info["samples"]))
                role_map = {info["background"]: 0, **{c: i + 1 for i, c in enumerate(info["order"])}}
                normalized.add((grid.shape, bytes(role_map[int(c)] for c in grid.flat)))
                for sample in info["samples"]:
                    if np.any(np.ptp(sample["cuts"], axis=0)):
                        jagged += 1
                    else:
                        straight += 1
    _require(len(signatures) == 42 and len(shapes) >= 45 and len(outputs) >= 15,
             "Insufficient puzzle/dimension variation")
    _require(len(palettes) >= 180 and len(records) >= 120 and len(geometries) >= 180
             and len(normalized) >= 200, "Insufficient structural variation")
    _require(axes == {("train", 0), ("train", 1), ("test", 0), ("test", 1)}, "Missing axis/split coverage")
    _require(layers == set(range(3, 10)) and sample_counts == {2, 3, 4, 5}
             and backgrounds == set(range(10)), "Missing count/palette roles")
    _require(repeated >= 3 and jagged >= 200 and straight >= 30 and internal >= 100,
             "Missing repeated, jagged, straight, or skipping records")
    _require(_puzzle_signature(generate_puzzle(73, 10, 4))
             == _puzzle_signature(generate_puzzle(73, 10, 4)), "Nondeterministic puzzle")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            generate_puzzle(0, train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # An underspecified/cyclic partial order must not be resolved by color ID.
    for sequences in (((1, 3), (2, 3)), ((1, 2), (2, 1))):
        try:
            _consensus(sequences)
        except ValueError:
            continue
        raise AssertionError("Ambiguous or cyclic strata accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes), "output_shapes": len(outputs),
            "palettes": len(palettes), "partial_record_patterns": len(records),
            "layouts": len(geometries), "normalized_inputs": len(normalized),
            "repeated_record_pairs": repeated, "jagged_samples": jagged,
            "straight_samples": straight, "internal_gap_pairs": internal}
