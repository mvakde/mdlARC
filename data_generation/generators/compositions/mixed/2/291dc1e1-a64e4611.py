"""Keyed strip assembly followed by clipped empty-neighborhood highlighting.

The corner marker is the output ink. No training data or sampled state is used
by solve. All generation and diagnostics remain in memory; no files are written.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected an integer ARC grid of dimensions 1..30")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    """Find the geometrically unique L, including background and output ink."""
    candidates = []
    if min(grid.shape) < 5:
        raise ValueError("No room for separated strips and an L key")
    for bottom in (False, True):
        for right in (False, True):
            a = grid[::-1] if bottom else grid
            a = a[:, ::-1] if right else a
            ink, horizontal, vertical, bg = map(int, (a[0, 0], a[0, 1], a[1, 0], a[-1, -1]))
            if (len({ink, horizontal, vertical, bg}) == 4
                    and np.all(a[0, 1:] == horizontal)
                    and np.all(a[1:, 0] == vertical)
                    and np.all(a[-1, 1:] == bg)
                    and np.all(a[1:, -1] == bg)):
                candidates.append((bottom, right, ink, horizontal, vertical, bg))
    if len(candidates) != 1:
        raise ValueError("The input must determine exactly one L key")
    return candidates[0]


def _components(mask):
    """Four-connected components, independent of foreground literal colors."""
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, todo = [start], [start]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.append(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _read(input_grid):
    grid = _grid(input_grid)
    bottom, right, ink, horizontal, vertical, bg = _key(grid)
    foreground = grid != bg
    foreground[-1 if bottom else 0, :] = False
    foreground[:, -1 if right else 0] = False
    if np.any((grid == ink) & foreground):
        raise ValueError("The corner ink is reserved, not a payload color")
    boxes, axes = [], set()
    for group in _components(foreground):
        r0, r1 = min(r for r, c in group), max(r for r, c in group) + 1
        c0, c1 = min(c for r, c in group), max(c for r, c in group) + 1
        height, width = r1 - r0, c1 - c0
        if len(group) != height * width or min(height, width) != 2:
            raise ValueError("Each payload must be a full two-cell-thick rectangle")
        if not (r0 >= 1 + int(not bottom) and c0 >= 1 + int(not right)
                and r1 <= grid.shape[0] - 1 - int(bottom)
                and c1 <= grid.shape[1] - 1 - int(right)):
            raise ValueError("Payload must have a background margin inside the outer edges")
        if width > 2:
            axes.add(0)
        if height > 2:
            axes.add(1)
        boxes.append(tuple(map(int, (r0, r1, c0, c1))))
    if not 1 <= len(boxes) <= 15 or len(axes) != 1:
        raise ValueError("There must be 1..15 strips with one visible common long axis")
    axis = axes.pop()  # 0: horizontal fast axis; 1: vertical fast axis.
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if (c1 - c0 if axis else r1 - r0) != 2:
            raise ValueError("Strip thickness disagrees with the common fast axis")
        lanes.setdefault(c0 if axis else r0, []).append(box)
    positions = sorted(lanes)
    if any(b - a < 3 for a, b in zip(positions, positions[1:])):
        raise ValueError("Lane bands must align and be separated by background")
    slow_reverse, fast_reverse = (right, bottom) if axis else (bottom, right)
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks, ordered_boxes, counts, fast_positions = [], [], [], []
    for lane in sorted(lanes, reverse=slow_reverse):
        ordered = sorted(lanes[lane], key=lambda b: b[0] if axis else b[2], reverse=fast_reverse)
        intervals = sorted((b[0], b[1]) if axis else (b[2], b[3]) for b in ordered)
        if any(b[0] - a[1] < 1 for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Strips within a lane must be separated")
        counts.append(len(ordered))
        for r0, r1, c0, c1 in ordered:
            blocks.append(np.rot90(grid[r0:r1, c0:c1], rotation).copy())
            ordered_boxes.append((r0, r1, c0, c1))
            fast_positions.append(-(r1 if axis else c1) if fast_reverse else (r0 if axis else c0))
    lengths = [b.shape[1] for b in blocks]
    if len({n % 2 for n in lengths}) != 1:
        raise ValueError("All strip lengths must share parity for exact centering")
    return {"blocks": blocks, "lengths": lengths, "counts": counts, "boxes": ordered_boxes,
            "fast_positions": fast_positions, "axis": axis, "bottom": bottom, "right": right,
            "ink": ink, "bg": bg, "fast": vertical if axis else horizontal,
            "slow": horizontal if axis else vertical}


def _stack(blocks, bg):
    width = max(b.shape[1] for b in blocks)
    canvas = np.full((2 * len(blocks), width), bg, dtype=np.int8)
    for i, block in enumerate(blocks):
        left = (width - block.shape[1]) // 2
        canvas[2 * i:2 * i + 2, left:left + block.shape[1]] = block
    return canvas


def _clear(canvas, bg, *, diagonal=True, outside_foreground=False):
    height, width = canvas.shape
    padded = np.pad(canvas != bg, 1, constant_values=outside_foreground)
    near = np.zeros(canvas.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if diagonal or dy == 1 or dx == 1:
                near |= padded[dy:dy + height, dx:dx + width]
    return ~near


def _highlight(canvas, bg, ink):
    output = canvas.copy()
    output[_clear(canvas, bg)] = ink
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the input alone; first establish the canvas, then inspect its cells."""
    d = _read(input_grid)
    return _highlight(_stack(d["blocks"], d["bg"]), d["bg"], d["ink"])


def _ground_truth(blocks, bg, ink):
    """Construction labels from sampled blocks, using explicit clipped slices."""
    canvas = _stack(blocks, bg)
    output = canvas.copy()
    height, width = canvas.shape
    for r in range(height):
        for c in range(width):
            if np.all(canvas[max(0, r - 1):min(height, r + 2), max(0, c - 1):min(width, c + 2)] == bg):
                output[r, c] = ink
    return output


def _gaps(rng, occupied, count, extent, aligned=False):
    spare = extent - occupied - count - 1
    assert spare >= 0
    gaps = np.ones(count + 1, dtype=int)
    start = 1 if aligned else 0
    weights = rng.dirichlet(np.full(count + 1 - start, float(rng.choice([.4, 1., 3.]))))
    gaps[start:] += rng.multinomial(spare, weights)
    return gaps.tolist()


def _partition(rng, lengths, witness):
    """Partition a sequence into feasible lanes, without changing its order."""
    n = len(lengths)
    ends = [[] for _ in range(n)]
    for i in range(n):
        occupied = 0
        for j in range(i + 1, n + 1):
            occupied += lengths[j - 1]
            if occupied + (j - i) + 1 <= 29:
                ends[i].append(j)
    minimum = [n + 1] * n + [0]
    for i in range(n - 1, -1, -1):
        minimum[i] = 1 + min(minimum[j] for j in ends[i])
    possibilities = []
    for k in range(2 if witness else 1, min(9, n - int(witness)) + 1):
        if any((not witness or j >= 2) and minimum[j] <= k - 1 <= n - j for j in ends[0]):
            possibilities.append(k)
    if not possibilities:
        raise ValueError("Sampled widths cannot fit into nine input lanes")
    remaining = int(rng.choice(possibilities))
    counts, i = [], 0
    while i < n:
        eligible = [j for j in ends[i] if minimum[j] <= remaining - 1 <= n - j
                    and (not witness or i != 0 or j >= 2)]
        j = int(rng.choice(eligible))
        counts.append(j - i)
        i, remaining = j, remaining - 1
    return counts


def _layout(rng, lengths, witness):
    counts = _partition(rng, lengths, witness)
    groups, start = [], 0
    for count in counts:
        groups.append(lengths[start:start + count])
        start += count
    minimum_fast = max(sum(g) + len(g) + 1 for g in groups)
    fast_extent = int(rng.integers(minimum_fast, 30))
    minimum_slow = 3 * len(counts) + 1
    slow_extent = int(rng.integers(minimum_slow, 30))
    slow_gaps = _gaps(rng, 2 * len(counts), len(counts), slow_extent)
    placements, v = [], slow_gaps[0]
    for lane, group in enumerate(groups):
        gaps = _gaps(rng, sum(group), len(group), fast_extent, aligned=witness)
        u = gaps[0]
        for index, length in enumerate(group):
            placements.append((v, u, length))
            u += length + gaps[index + 1]
        v += 2 + slow_gaps[lane + 1]
    return placements, (slow_extent, fast_extent)


def _paint(blocks, placements, shape, config, roles):
    axis, bottom, right = config
    fast, slow, bg, ink = roles
    slow_extent, fast_extent = shape
    height, width = (fast_extent + 1, slow_extent + 1) if axis else (slow_extent + 1, fast_extent + 1)
    grid = np.full((height, width), bg, dtype=np.int8)
    row, col = (height - 1 if bottom else 0), (width - 1 if right else 0)
    grid[row, :] = slow if axis else fast
    grid[:, col] = fast if axis else slow
    grid[row, col] = ink
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    for block, (v, u, length) in zip(blocks, placements):
        physical = np.rot90(block, -rotation)
        r0, c0 = (1 + u, 1 + v) if axis else (1 + v, 1 + u)
        h, w = physical.shape
        r0 = height - r0 - h if bottom else r0
        c0 = width - c0 - w if right else c0
        grid[r0:r0 + h, c0:c0 + w] = physical
    return grid


def _lengths(rng, kind):
    base = int(rng.choice([2, 3]))
    if kind == "edge":
        n = int(rng.integers(5, 9))
        maximum = base + 2 * int(rng.integers(3, 5))
        lengths = rng.choice(np.arange(base, maximum + 1, 2), size=n).astype(int).tolist()
        lengths[:5] = [base, maximum, base + 2, base, base]
        if rng.random() < .5:
            lengths.reverse()
    elif kind == "bay":
        n = int(rng.integers(6, 11))
        maximum = base + 2 * int(rng.integers(4, 8))
        lengths = rng.choice(np.arange(base, base + 5, 2), size=n).astype(int).tolist()
        start = int(rng.integers(n - 4))
        lengths[start:start + 5] = [maximum, base, base + 2, base, maximum]
        if rng.random() < .5:
            lengths.reverse()
    else:
        n = int(rng.integers(6 if kind == "test" else 3, 16))
        maximum = base + 2 * int(rng.integers(3 if kind == "test" else 1, 13))
        ceiling = min(maximum, base + 8)
        lengths = rng.choice(np.arange(base, ceiling + 1, 2), size=n).astype(int).tolist()
        start = int(rng.integers(n - 1))
        lengths[start:start + 2] = [base, base]
        pivot = int(rng.choice([i for i in range(n) if i not in (start, start + 1)]))
        lengths[pivot] = maximum
    return lengths


def _asymmetric(block):
    return all(not np.array_equal(block, other) for other in (block[::-1], block[:, ::-1], block[::-1, ::-1]))


def _payloads(rng, lengths, pool, witness):
    palette = rng.choice(pool, size=int(rng.integers(3 if witness else 1, len(pool) + 1)), replace=False)
    blocks = []
    simple = not witness and rng.random() < .25
    for length in lengths:
        repeated = [b for b in blocks if b.shape[1] == length]
        if repeated and rng.random() < .25:
            block = repeated[int(rng.integers(len(repeated)))].copy()
        else:
            mode = 0 if simple else int(rng.integers(4))
            if mode == 0:
                block = np.full((2, length), rng.choice(palette), dtype=np.int8)
            elif mode == 1:
                block = np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
            elif mode == 2:
                motif = rng.choice(palette, size=(2, int(rng.integers(1, min(4, length) + 1))))
                block = np.tile(motif, (1, length))[:, :length].astype(np.int8)
            else:
                block = rng.choice(palette, size=(2, length)).astype(np.int8)
        blocks.append(block)
    if witness:
        chosen = int(rng.integers(len(blocks)))
        for _ in range(32):
            block = rng.choice(palette, size=blocks[chosen].shape).astype(np.int8)
            if _asymmetric(block) and not any(np.array_equal(block, b) for i, b in enumerate(blocks) if i != chosen):
                break
        else:
            block = np.full(blocks[chosen].shape, palette[0], dtype=np.int8)
            block[0, 0], block[1, -1] = palette[1:3]
        blocks[chosen] = block
    return blocks


def _grid_key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _alternatives(d):
    """Concrete plausible wrong rules, used only to audit demonstrations."""
    blocks, bg, ink = d["blocks"], d["bg"], d["ink"]
    canvas = _stack(blocks, bg)
    expected = _highlight(canvas, bg, ink)
    masks = {"fill_all_padding": canvas == bg,
             "orthogonal_only": _clear(canvas, bg, diagonal=False),
             "outside_is_obstacle": _clear(canvas, bg, outside_foreground=True)}
    clear = _clear(canvas, bg)
    filtered = np.zeros(canvas.shape, dtype=bool)
    for group in _components(clear):
        if len(group) >= 5:
            for r, c in group:
                filtered[r, c] = True
    masks["five_cell_cutoff"] = filtered
    union = np.zeros(canvas.shape, dtype=bool)
    for r in range(canvas.shape[0] - 2):
        for c in range(canvas.shape[1] - 2):
            if np.all(canvas[r:r + 3, c:c + 3] == bg):
                union[r:r + 3, c:c + 3] = True
    masks["union_of_empty_windows"] = union
    flags = {"omit_B": not np.array_equal(canvas, expected)}
    for name, mask in masks.items():
        flags[name] = not np.array_equal(mask, clear)
    independently = np.concatenate([_highlight(canvas[i:i + 2], bg, ink) for i in range(0, len(canvas), 2)])
    flags["test_each_block_separately"] = not np.array_equal(independently, expected)
    groups, start = [], 0
    for count in d["counts"]:
        groups.append(list(range(start, start + count)))
        start += count
    orders = {"reverse_fast": [i for group in groups for i in group[::-1]],
              "reverse_slow": [i for group in groups[::-1] for i in group],
              "fast_major": sorted(range(len(blocks)), key=lambda i: (d["fast_positions"][i], i)),
              "sort_by_length": sorted(range(len(blocks)), key=lambda i: d["lengths"][i])}
    flags["key_changes_neighborhoods"] = False
    for name, order in orders.items():
        alternative_canvas = _stack([blocks[i] for i in order], bg)
        flags[name] = not np.array_equal(_highlight(alternative_canvas, bg, ink), expected)
        if name != "sort_by_length":
            flags["key_changes_neighborhoods"] |= not np.array_equal(_clear(alternative_canvas, bg), clear)
    return flags


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    fast, slow, ink = map(int, rng.choice(10, size=3, replace=False))
    backgrounds = [c for c in range(10) if c not in (fast, slow, ink)]
    rng.shuffle(backgrounds)
    configs = [(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)]
    first = configs[int(rng.integers(8))]
    second = (1 - first[0], not first[1], not first[2])
    held = [c for c in configs if c not in (first, second)]
    test_config = held[int(rng.integers(len(held)))]
    training_configs = [c for c in configs if c != test_config]
    plan = {"train": [first, second] + [training_configs[int(rng.integers(7))] for _ in range(num_train - 2)],
            "test": [test_config] * num_test}
    puzzle, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for index, config in enumerate(plan[split]):
            kind = ("edge" if index == 0 else "bay") if split == "train" and index < 2 else ("test" if split == "test" else "ordinary")
            bg = backgrounds[index] if split == "train" and index < 2 else int(rng.choice(backgrounds))
            witness = kind in ("edge", "bay")
            for _ in range(32):
                lengths = _lengths(rng, kind)
                placements, shape = _layout(rng, lengths, witness)
                pool = [c for c in range(10) if c not in (bg, ink)]
                blocks = _payloads(rng, lengths, pool, witness)
                inp = _paint(blocks, placements, shape, config, (fast, slow, bg, ink))
                if _grid_key(inp) in seen:
                    continue
                if kind == "edge" and not all(_alternatives(_read(inp)).values()):
                    continue
                out = _ground_truth(blocks, bg, ink)
                if kind == "test" and not np.any(out == ink):
                    continue
                seen.add(_grid_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("Bounded pair construction exhausted")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, roles, descriptors = set(), set(), {"train": [], "test": []}
    flags = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen, "Repeated input"
            seen.add(key)
            d = _read(inp)
            descriptors[split].append(d)
            roles.add((d["fast"], d["slow"], d["ink"]))
            canvas = _stack(d["blocks"], d["bg"])
            assert out.shape == (2 * len(d["blocks"]), max(d["lengths"]))
            foreground = canvas != d["bg"]
            assert np.array_equal(canvas[foreground], out[foreground]), "Foreground was changed"
            assert int(np.sum(foreground)) == sum(b.size for b in d["blocks"]), "Lost a payload occurrence"
            # Explicit per-cell biconditional checks clipping, diagonals and fringe.
            for r, c in zip(*np.nonzero(~foreground)):
                neighborhood = canvas[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                expected = d["ink"] if np.all(neighborhood == d["bg"]) else d["bg"]
                assert out[r, c] == expected, "Wrong empty-neighborhood selection or fringe"
            assert np.array_equal(solve(inp.copy()), out), "Input-only solve disagrees"
            if split == "train":
                for name, value in _alternatives(d).items():
                    flags[name] = flags.get(name, False) or value
    assert len(roles) == 1, "Guide identities and ink must be coherent within a puzzle"
    train = descriptors["train"]
    assert {d["axis"] for d in train} == {0, 1}
    assert {d["bottom"] for d in train} == {False, True}
    assert {d["right"] for d in train} == {False, True}
    assert len({d["bg"] for d in train}) >= 2
    assert all(flags.values()), f"Undistinguished shortcuts: {[k for k, v in flags.items() if not v]}"
    assert any(any(_asymmetric(b) for b in d["blocks"]) for d in train)
    # The test recombines visible axis/corner factors in an unseen combination.
    familiar = {(d["axis"], d["bottom"], d["right"]) for d in train}
    assert all((d["axis"], d["bottom"], d["right"]) not in familiar for d in descriptors["test"])


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _grid_key(pair[role])
                h.update(bytes(shape))
                h.update(payload)
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; keep digests and small feature sets, never a corpus."""
    signatures, normalized, shapes, output_shapes, configs = set(), set(), set(), set(), set()
    counts, maxima, palettes, inks, lane_counts, parities = set(), set(), set(), set(), set(), set()
    highlights, small_components, diagonal_cases, joined_cases = 0, 0, 0, 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                d = _read(inp)
                canvas = _stack(d["blocks"], d["bg"])
                shapes.add(inp.shape)
                output_shapes.add(out.shape)
                counts.add(len(d["blocks"]))
                lane_counts.add(tuple(d["counts"]))
                maxima.add(max(d["lengths"]))
                parities.add(max(d["lengths"]) % 2)
                inks.add(d["ink"])
                palettes.add(len(set(map(int, canvas[canvas != d["bg"]]))))
                configs.add((d["axis"], d["bottom"], d["right"]))
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                normalized.add(hashlib.sha256(bytes(inp.shape) + norm).hexdigest())
                highlights += int(np.sum(out == d["ink"]))
                small_components += any(len(g) < 5 for g in _components(out == d["ink"]))
                flags = _alternatives(d)
                diagonal_cases += flags["orthogonal_only"]
                joined_cases += flags["test_each_block_separately"]
                # Proper whole-input rotations and arbitrary palette permutations.
                for rotation in range(4):
                    assert np.array_equal(solve(np.rot90(inp, rotation)), out)
                permutation = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
                assert np.array_equal(solve(permutation[inp]), permutation[out])
                unchanged = inp.copy()
                solve(inp)
                assert np.array_equal(inp, unchanged)
                pairs += 1
    assert len(signatures) == 42 and len(normalized) >= 200
    assert len(shapes) >= 60 and len(output_shapes) >= 40
    assert len(configs) == 8 and len(lane_counts) >= 30
    assert min(counts) <= 4 and max(counts) == 15
    assert max(maxima) >= 24 and parities == {0, 1}
    assert len(inks) >= 8 and len(palettes) >= 6
    assert small_components >= 42 and diagonal_cases >= 42 and joined_cases >= 42
    # All explicit legal pair counts, plus noninteger/boolean/null rejection.
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            puzzle = generate_puzzle(61942879, tr, te)
            assert len(puzzle["train"]) == tr and len(puzzle["test"]) == te
            validate_generated_puzzle(puzzle)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for invalid_seed in (-1, True, 1.5, None):
        try:
            generate_puzzle(invalid_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal seed accepted")
    # Boundary clipping and diagonal-only obstruction, including tiny arrays.
    assert np.all(_highlight(np.zeros((2, 3), dtype=np.int8), 0, 7) == 7)
    tiny = np.zeros((4, 4), dtype=np.int8)
    tiny[0, 0] = 1
    result = _highlight(tiny, 0, 7)
    assert result[1, 1] == 0 and result[0, 2] == 7 and result[3, 3] == 7
    # Supported small populations, tied maxima, odd lengths and full-width
    # payloads, including solver members not emphasized by the random sampler.
    for lengths in ([3], [27], [4, 4], [3, 7, 3], [2, 6, 2], [3, 27, 3]):
        blocks = [np.resize(np.array([[4, 5, 4], [5, 4, 4]], dtype=np.int8), (2, length))
                  for length in lengths]
        placements = [(1 + 3 * i, 1, length) for i, length in enumerate(lengths)]
        shape = (3 * len(lengths) + 1, max(lengths) + 2)
        expected = _ground_truth(blocks, 0, 7)
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    inp = _paint(blocks, placements, shape, (axis, bottom, right), (1, 2, 0, 7))
                    assert np.array_equal(solve(inp), expected)
        if len(set(lengths)) == 1:
            assert np.array_equal(expected, _stack(blocks, 0))
    # A strip may not touch either guide: one background cell is mandatory.
    touching = _paint([np.full((2, 3), 4, dtype=np.int8)], [(0, 1, 3)],
                      (4, 5), (0, False, False), (1, 2, 0, 7))
    try:
        _read(touching)
    except ValueError:
        pass
    else:
        raise AssertionError("Payload touching the key was accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(normalized), "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "key_configurations": len(configs),
            "lane_population_vectors": len(lane_counts), "strip_counts": sorted(counts),
            "maximum_widths": sorted(maxima), "payload_palette_sizes": sorted(palettes),
            "ink_colors": sorted(inks), "marked_cells": highlights,
            "small_highlight_components_pairs": small_components,
            "diagonal_diagnostic_pairs": diagonal_cases, "cross_strip_diagnostic_pairs": joined_cases,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
