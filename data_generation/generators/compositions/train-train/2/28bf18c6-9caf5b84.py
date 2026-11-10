"""Frequency-selected sockets register two copies of one cropped link.

Only Python's standard library and NumPy are used.  No parent code or data is
loaded.  The solver uses the supplied grid alone.
"""

from collections import Counter

import numpy as np


WELD = 7


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _check_counts(num_train, num_test):
    a = _integer(num_train, "num_train")
    b = _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer 1..30 by 1..30 ARC grid")
    return grid.astype(np.int64, copy=False)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(map(int, grid.flat))


def _components(mask):
    """Eight-connected components, deterministically sorted by first cell."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        reached = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        reached.add(p)
                        stack.append(p)
        result.append(reached)
    return result


def _box(points):
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _normalized(points):
    r0, c0, r1, c1 = _box(points)
    return (r1 - r0, c1 - c0), frozenset((r - r0, c - c0) for r, c in points)


def _asymmetric(tile):
    variants = [np.flipud(tile), np.fliplr(tile), np.rot90(tile, 2)]
    if tile.shape[0] == tile.shape[1]:
        variants += [tile.T, np.rot90(tile), np.rot90(tile, 3), np.rot90(tile.T, 2)]
    return all(not np.array_equal(tile, other) for other in variants)


def _analyze(input_grid):
    """Parse visible roles and enforce the general family, not sampler bounds."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 6:
        raise ValueError("expected background and five object inks")
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= grid.size or WELD in colors:
        raise ValueError("background must be a strict majority; 7 is output-only")
    inks = colors[colors != bg]
    ink_counts = np.array([np.count_nonzero(grid == color) for color in inks])
    order = np.argsort(-ink_counts, kind="stable")
    if ink_counts[order[1]] <= ink_counts[order[2]]:
        raise ValueError("ambiguous frequency boundary")
    keep = tuple(map(int, inks[order[:2]]))
    rare = tuple(map(int, inks[order[2:]]))
    occupied = set(map(tuple, np.argwhere(grid != bg)))
    if len(_components(grid != bg)) != 1:
        raise ValueError("the complete link must be eight-connected")
    ports = _components(np.isin(grid, rare))
    if len(ports) != 2 or _normalized(ports[0]) != _normalized(ports[1]):
        raise ValueError("expected two translation-congruent sockets")
    a = _box(ports[0])
    b = _box(ports[1])
    delta = (b[0] - a[0], b[1] - a[1])
    dr, dc = delta
    shifted = {(r + dr, c + dc) for r, c in occupied}
    if occupied & shifted != ports[1]:
        raise ValueError("docking must overlap only the matching socket")
    socket_cells = sorted(ports[0])
    if all(grid[r, c] == grid[r + dr, c + dc] for r, c in socket_cells):
        raise ValueError("socket color patterns must differ")
    r0, c0, r1, c1 = _box(occupied)
    tile = grid[r0:r1, c0:c1].copy()
    output_shape = (tile.shape[0] + abs(dr), tile.shape[1] + abs(dc))
    if max(output_shape) > 30:
        raise ValueError("docked output exceeds ARC bounds")
    clean = tile.copy()
    clean[np.isin(tile, rare)] = WELD
    port_shape, port_mask = _normalized(ports[0])
    major_axis = int(abs(dc) > abs(dr))
    gap = abs(delta[major_axis]) - port_shape[major_axis]
    return {"grid": grid, "background": bg, "keep": keep, "rare": rare,
            "counts": tuple(map(int, ink_counts[order])), "ports": ports,
            "port_shape": port_shape, "port_mask": port_mask,
            "delta": delta, "tile": tile, "clean": clean,
            "position": (r0, c0), "output_shape": output_shape,
            "gap": gap}


def _assemble(tile, background, delta):
    dr, dc = delta
    result = np.full((tile.shape[0] + abs(dr), tile.shape[1] + abs(dc)),
                     background, dtype=np.int64)
    occupied = tile != background
    for r, c in ((max(0, -dr), max(0, -dc)), (max(0, dr), max(0, dc))):
        window = result[r:r + tile.shape[0], c:c + tile.shape[1]]
        window[occupied] = tile[occupied]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    return _assemble(info["clean"], info["background"], info["delta"])


def _socket(rng, cells, max_side):
    options = [(h, w) for h in range(1, max_side + 1)
               for w in range(1, max_side + 1) if h * w >= cells]
    h, w = options[int(rng.integers(len(options)))]
    mask = np.zeros((h, w), dtype=bool)
    mask[int(rng.integers(h)), int(rng.integers(w))] = True
    while int(mask.sum()) < cells:
        candidates = [(r, c) for r in range(h) for c in range(w)
                      if not mask[r, c] and
                      np.any(mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2])]
        r, c = candidates[int(rng.integers(len(candidates)))]
        mask[r, c] = True
    r0, c0, r1, c1 = _box(set(map(tuple, np.argwhere(mask))))
    return mask[r0:r1, c0:c1]


def _minority_counts(rng, k, profile):
    if profile == "equal":
        assert (2 * k) % 3 == 0
        return [2 * k // 3] * 3
    if profile == "skew":
        return [k, k - 1, 1]
    if rng.random() < .5:
        counts = [2 * k // 3] * 3
        for i in range((2 * k) % 3):
            counts[i] += 1
        return counts
    return list(map(int, rng.multinomial(2 * k - 3, [.36, .34, .30]) + 1))


def _body(rng, h, gap, row_a, row_b, minimum, tied):
    mask = np.zeros((h, gap), dtype=bool)
    r, c = row_a, 0
    end = (row_b, gap - 1)
    mask[r, c] = True
    while (r, c) != end:
        dr = int(np.sign(end[0] - r))
        dc = int(np.sign(end[1] - c))
        if dr and dc and rng.random() < .55:
            if rng.integers(2):
                dr = 0
            else:
                dc = 0
        r += dr
        c += dc
        mask[r, c] = True
    density = float(rng.choice([.27, .42, .58, .76, .92, 1.0]))
    target = max(minimum, int(mask.sum()), int(round(mask.size * density)))
    if tied and target % 2:
        target = target + 1 if target < mask.size else target - 1
    while int(mask.sum()) < target:
        candidates = [(r, c) for r in range(h) for c in range(gap)
                      if not mask[r, c] and
                      np.any(mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2])]
        r, c = candidates[int(rng.integers(len(candidates)))]
        mask[r, c] = True
    return mask


def _paint_body(rng, tile, body_mask, colors, small_count, disperse=False):
    points = np.argwhere(body_mask)
    mode = int(rng.integers(3))
    if disperse:
        # Farthest-point placement makes a globally retained ink locally
        # fragmented, opposing the compact minority patch in train example 2.
        chosen = [int(rng.integers(len(points)))]
        distances = np.full(len(points), np.inf)
        while len(chosen) < small_count:
            distances = np.minimum(distances, np.max(np.abs(points - points[chosen[-1]]), axis=1))
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances + rng.random(len(points)) * .15)))
        order = np.array(chosen, dtype=int)
    elif mode == 0:
        order = rng.permutation(len(points))
    elif mode == 1:
        center = points[int(rng.integers(len(points)))]
        scores = np.sum(np.abs(points - center), axis=1) + rng.random(len(points)) * .8
        order = np.argsort(scores)
    else:
        direction = rng.choice([-1, 1], size=2)
        scores = points @ direction + rng.random(len(points)) * .7
        order = np.argsort(scores)
    tile[body_mask] = colors[0]
    selected = points[order[:small_count]]
    tile[selected[:, 0], selected[:, 1]] = colors[1]


def _make_link(rng, bg, inks, gap, stage, index):
    if stage == "train" and index == 0:
        k = int(rng.choice([3, 6]))
        profile, tied = "equal", True
    elif stage == "train" and index == 1:
        k = int(rng.integers(3, 8))
        profile, tied = "skew", False
    else:
        k = int(rng.integers(3, 8 if stage == "train" else 9))
        profile = "varied"
        tied = bool(rng.random() < .3)
    socket = _socket(rng, k, 3 if stage == "train" else 4)
    ph, pw = socket.shape
    offset_y = int(rng.integers(-3, 4) if stage == "train" else rng.integers(-5, 6))
    distance = pw + gap
    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    ra = top + max(0, -offset_y)
    rb = ra + offset_y
    h = ph + abs(offset_y) + top + bottom
    minor_counts = _minority_counts(rng, k, profile)
    largest_rare = max(minor_counts)
    minimum = 2 * (largest_rare + 1) + (0 if tied else 1)
    h = max(h, (minimum + gap - 1) // gap)
    if tied and h * gap % 2 and h * gap == minimum:
        h += 1
    # All main cells lie strictly between port column ranges.  Therefore the
    # shifted occupied masks can meet only on a socket, even for diagonal links.
    a_candidates = np.flatnonzero(socket[:, -1])
    b_candidates = np.flatnonzero(socket[:, 0])
    row_a = ra + int(rng.choice(a_candidates))
    row_b = rb + int(rng.choice(b_candidates))
    body = _body(rng, h, gap, row_a, row_b, minimum, tied)
    n = int(body.sum())
    if tied:
        small_count = n // 2
    elif index == 1 and stage == "train":
        small_count = largest_rare + 1
    else:
        high = (n - 1) // 2
        small_count = int(rng.integers(largest_rare + 1, high + 1))
    tile = np.full((h, distance + pw), bg, dtype=np.int64)
    body_mask = np.zeros(tile.shape, dtype=bool)
    body_mask[:, pw:distance] = body
    _paint_body(rng, tile, body_mask, inks[:2], small_count,
                disperse=(stage == "train" and index == 1))
    points_a = [(ra + int(r), int(c)) for r, c in np.argwhere(socket)]
    points_b = [(rb + int(r), distance + int(c)) for r, c in np.argwhere(socket)]
    port_colors = np.repeat(np.array(inks[2:], dtype=int), minor_counts)
    # Deterministic repair ensures geometry, not matching literal colors,
    # must be used to identify corresponding sockets.
    if stage == "train" and index == 1:
        # The most common minority ink occupies an entire connected socket.
        # Its global count is still below BOTH body inks.
        rng.shuffle(port_colors[k:])
        if rng.integers(2):
            port_colors = np.concatenate((port_colors[k:], port_colors[:k]))
    else:
        rng.shuffle(port_colors)
    if np.array_equal(port_colors[:k], port_colors[k:]):
        unequal = next(j for j in range(1, k) if port_colors[j] != port_colors[0])
        port_colors[0], port_colors[unequal] = port_colors[unequal], port_colors[0]
    for point, color in zip(points_a + points_b, port_colors):
        tile[point] = color
    if stage == "train" and index < 2:
        turns = 2 * int(rng.integers(2)) + index
    else:
        turns = int(rng.integers(4))
    tile = np.rot90(tile, turns).copy()
    if rng.random() < .5:
        tile = np.fliplr(tile).copy()
    r0, c0, r1, c1 = _box(set(map(tuple, np.argwhere(tile != bg))))
    return tile[r0:r1, c0:c1].copy()


def _place(rng, tile, bg):
    h, w = tile.shape
    rows = min(30, h + int(rng.integers(0, 10)))
    cols = min(30, w + int(rng.integers(0, 10)))
    occupied = int(np.count_nonzero(tile != bg))
    while rows * cols <= 2 * occupied:
        options = [axis for axis, value in enumerate((rows, cols)) if value < 30]
        axis = options[int(rng.integers(len(options)))]
        if axis == 0:
            rows += 1
        else:
            cols += 1
    if rng.random() < .45:
        r = int(rng.choice([0, rows - h]))
        c = int(rng.choice([0, cols - w]))
    else:
        r = int(rng.integers(rows - h + 1))
        c = int(rng.integers(cols - w + 1))
    grid = np.full((rows, cols), bg, dtype=np.int64)
    grid[r:r + h, c:c + w] = tile
    return grid


def _palette(rng, anchor=None, previous=None):
    if previous is not None and rng.random() < .45:
        bg, inks = previous
        return bg, list(map(int, rng.permutation(inks)))
    bg_choices = [c for c in range(10) if c != WELD and c != anchor]
    bg = int(rng.choice(bg_choices))
    choices = [c for c in range(10) if c not in (WELD, bg, anchor)]
    if anchor is None:
        inks = list(map(int, rng.choice(choices, size=5, replace=False)))
    else:
        inks = [anchor] + list(map(int, rng.choice(choices, size=4, replace=False)))
        rng.shuffle(inks)
    return bg, inks


def _local_component_witness(info):
    grid = info["grid"]
    largest = {color: max(map(len, _components(grid == color)))
               for color in info["keep"] + info["rare"]}
    return max(largest[c] for c in info["rare"]) > min(largest[c] for c in info["keep"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor = int(rng.choice([c for c in range(10) if c != WELD]))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_palette = None
    first_shape = None
    for stage, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(100):
                use_anchor = anchor if stage == "train" and index < 2 else None
                bg, inks = _palette(rng, use_anchor, first_palette if index else None)
                if use_anchor is not None:
                    destination = 0 if index == 0 else 2
                    position = inks.index(anchor)
                    inks[position], inks[destination] = inks[destination], inks[position]
                gap = ((3, 5)[index] if stage == "train" and index < 2 else
                       int(rng.integers(3, 6) if stage == "train" else rng.integers(6, 9)))
                tile = _make_link(rng, bg, inks, gap, stage, index)
                clean = tile.copy()
                clean[np.isin(tile, inks[2:])] = WELD
                if stage == "train" and index == 0 and not _asymmetric(clean):
                    continue
                if stage == "train" and index == 1 and tile.shape == first_shape:
                    continue
                inp = _place(rng, tile, bg)
                key = _key(inp)
                if key in seen:
                    continue
                info = _analyze(inp)
                if stage == "train" and index == 1 and not _local_component_witness(info):
                    continue
                # Label construction and solve share the geometric convention,
                # but neither uses the seed or any stored example.
                out = _assemble(clean, bg, info["delta"])
                break
            else:
                raise RuntimeError("bounded link construction exhausted")
            seen.add(key)
            puzzle[stage].append({"input": inp, "output": out})
            if stage == "train" and index == 0:
                first_palette = (bg, inks.copy())
                first_shape = tile.shape
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze(inp)
    bg, tile, delta = info["background"], info["tile"], info["delta"]
    assert out.shape == info["output_shape"]
    assert np.array_equal(out, solve(inp))
    assert len(_components(out != bg)) == 1
    welds = _components(out == WELD)
    assert len(welds) == 3
    assert all(_normalized(c) == (info["port_shape"], info["port_mask"]) for c in welds)
    origins = sorted(_box(c)[:2] for c in welds)
    assert tuple(np.subtract(origins[1], origins[0])) == delta
    assert tuple(np.subtract(origins[2], origins[1])) == delta
    k = len(info["ports"][0])
    assert np.count_nonzero(out == WELD) == 3 * k
    assert set(map(int, np.unique(out))) <= {bg, WELD, *info["keep"]}
    for color in info["keep"]:
        assert np.count_nonzero(out == color) == 2 * np.count_nonzero(inp == color)
    # Audit every output cell from its possible source coordinates, separately
    # from _assemble's window writes.  Background is transparent.
    dr, dc = delta
    offsets = [(max(0, -dr), max(0, -dc)), (max(0, dr), max(0, dc))]
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            contributions = []
            for ro, co in offsets:
                tr, tc = r - ro, c - co
                if 0 <= tr < tile.shape[0] and 0 <= tc < tile.shape[1]:
                    color = int(tile[tr, tc])
                    if color != bg:
                        contributions.append(WELD if color in info["rare"] else color)
            assert len(set(contributions)) <= 1
            expected = contributions[0] if contributions else bg
            assert out[r, c] == expected
    # Explicit conceptual ablations, all observably wrong on every example.
    assert not np.array_equal(out, info["clean"])
    assert not np.array_equal(out, np.concatenate([info["clean"]] * 2, axis=1))
    assert not np.array_equal(out, _assemble(tile, bg, delta))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for stage in ("train", "test"):
        for pair in puzzle[stage]:
            info = _validate_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            analyses[stage].append(info)
    train = analyses["train"]
    assert train[0]["counts"][0] == train[0]["counts"][1]
    assert len(set(train[0]["counts"][2:])) == 1
    assert train[1]["counts"][0] > train[1]["counts"][1]
    assert len(set(train[1]["counts"][2:])) > 1
    assert set(train[0]["keep"]) & set(train[1]["rare"])
    assert _local_component_witness(train[1])
    assert _asymmetric(train[0]["clean"])
    assert train[0]["tile"].shape != train[1]["tile"].shape
    assert train[0]["gap"] == 3 and train[1]["gap"] == 5
    assert all(3 <= a["gap"] <= 5 for a in train)
    assert all(6 <= a["gap"] <= 8 for a in analyses["test"])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _variation_metrics(puzzles):
    shapes, canvases, ports, bodies, palettes, vectors, backgrounds = [set() for _ in range(7)]
    measures = Counter()
    for puzzle in puzzles:
        for stage in ("train", "test"):
            for pair in puzzle[stage]:
                a = _analyze(pair["input"])
                inp = a["grid"]
                shapes.add(a["tile"].shape)
                canvases.add(inp.shape)
                ports.add((a["port_shape"], a["port_mask"]))
                bodies.add((a["tile"].shape, tuple(np.isin(a["tile"], a["keep"]).flat)))
                palettes.add((a["background"], frozenset(a["keep"]), frozenset(a["rare"])))
                vectors.add(a["delta"])
                backgrounds.add(a["background"])
                dr, dc = a["delta"]
                measures["horizontal"] += dr == 0
                measures["vertical"] += dc == 0
                measures["diagonal"] += dr != 0 and dc != 0
                measures["negative_column"] += dc < 0
                measures["zero_ink"] += 0 in a["keep"] or 0 in a["rare"]
                measures["zero_background"] += a["background"] == 0
                measures["top_tie"] += a["counts"][0] == a["counts"][1]
                measures["minority_tie"] += len(set(a["counts"][2:])) == 1
                measures["close_cutoff"] += a["counts"][1] - a["counts"][2] == 1
                measures["local_component_counterexamples"] += _local_component_witness(a)
                r, c = a["position"]
                h, w = a["tile"].shape
                edge = r == 0 or c == 0 or r + h == inp.shape[0] or c + w == inp.shape[1]
                measures["edge"] += edge
                measures["interior"] += not edge
                measures["pairs"] += 1
    return {"puzzles": len(puzzles), "crop_shapes": len(shapes), "canvas_shapes": len(canvases),
            "socket_shapes": len(ports), "body_masks": len(bodies), "role_palettes": len(palettes),
            "vectors": len(vectors), "backgrounds": len(backgrounds),
            **{name: int(value) for name, value in measures.items()}}


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    metrics = _variation_metrics(puzzles)
    assert metrics["crop_shapes"] >= 30 and metrics["canvas_shapes"] >= 50
    assert metrics["socket_shapes"] >= 20 and metrics["body_masks"] >= 150
    assert metrics["vectors"] >= 35 and metrics["role_palettes"] >= 100
    assert metrics["backgrounds"] == 9
    assert metrics["local_component_counterexamples"] >= 42
    for name in ("horizontal", "vertical", "diagonal", "negative_column", "zero_ink",
                 "zero_background", "top_tie", "minority_tie", "close_cutoff", "edge", "interior"):
        assert metrics[name] >= 4, (name, metrics)
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(4200 + train * 10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (None, 1), (4, None), (True, 1), (4, False), (4.0, 1), (4, 1.0)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Exchange the two body inks and translate the object to another canvas edge.
    pair = puzzles[0]["train"][0]
    a = _analyze(pair["input"])
    swapped = a["tile"].copy()
    first, second = a["keep"]
    swapped[a["tile"] == first] = second
    swapped[a["tile"] == second] = first
    old_clean = a["clean"].copy()
    old_clean[a["clean"] == first] = second
    old_clean[a["clean"] == second] = first
    expanded = np.full((30, 30), a["background"], dtype=int)
    expanded[-swapped.shape[0]:, -swapped.shape[1]:] = swapped
    assert np.array_equal(solve(expanded), _assemble(old_clean, a["background"], a["delta"]))
    # A valid link outside the sampler's strict body corridor.  Its extra body
    # cell at (0, 1) lies inside the left socket's column range, yet still docks
    # without any additional occupied collision.  Expected rows are explicit.
    rows = ("9112299", "3412153", "5921249")
    tile = np.array([[int(c) for c in row] for row in rows], dtype=int)
    witness = np.full((9, 14), 9, dtype=int)
    witness[3:6, 4:11] = tile
    expected_rows = ("911229112299", "771217712177", "792127921279")
    expected = np.array([[int(c) for c in row] for row in expected_rows], dtype=int)
    _validate_pair({"input": witness, "output": expected})
    return metrics
