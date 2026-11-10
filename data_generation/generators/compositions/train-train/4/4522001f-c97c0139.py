"""Diagonal diamond looms: marked bars become two tapered opposite wings.

Only NumPy and the standard library are used.  All solving conventions are
visible in one input; in particular no scale or output color is latent.
"""

import numpy as np


CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=True)


def _diamond(s):
    """The union of two adjacent even-diameter diamonds, centered on 2x2."""
    r, c = np.indices((2 * s, 2 * s))
    return np.abs(2 * r - (2 * s - 1)) + np.abs(2 * c - (2 * s - 1)) <= 2 * s


def _wings(s, corner):
    r, c = np.indices((2 * s, 2 * s))
    same_quadrant = (r < s) == (c < s)
    return _diamond(s) & (same_quadrant == (corner[0] == corner[1]))


def _bar(s, axis):
    mask = np.zeros((2 * s, 2 * s), dtype=bool)
    if axis == "H":
        mask[s - 1:s + 1, :] = True
    else:
        mask[:, s - 1:s + 1] = True
    return mask


def _touch4(mask):
    result = mask.copy()
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    result = []
    for sr, sc in zip(*np.nonzero(mask)):
        sr, sc = int(sr), int(sc)
        if visited[sr, sc]:
            continue
        stack, cells = [(sr, sc)], []
        visited[sr, sc] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not visited[rr, cc]):
                    visited[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _parse(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("the background must occupy more than half the input")
    background = int(colors[int(np.argmax(counts))])
    objects = []
    occupied = np.zeros(grid.shape, dtype=bool)
    for cells in _components(grid != background):
        rows, cols = zip(*cells)
        top, left, bottom, right = min(rows), min(cols), max(rows), max(cols)
        height, width = bottom - top + 1, right - left + 1
        if height == 2 and width >= 4 and width % 2 == 0:
            axis, s = "H", width // 2
            square_top, square_left = top - s + 1, left
        elif width == 2 and height >= 4 and height % 2 == 0:
            axis, s = "V", height // 2
            square_top, square_left = top, left - s + 1
        else:
            raise ValueError("each component must be a two-cell-wide even bar")
        if len(cells) != height * width:
            raise ValueError("bars must be solid rectangles")
        values, frequencies = np.unique(grid[top:bottom + 1, left:right + 1], return_counts=True)
        if len(values) != 2 or sorted(map(int, frequencies)) != [1, 4 * s - 1]:
            raise ValueError("each bar needs one marker and one payload color")
        marker = int(values[int(np.argmin(frequencies))])
        payload = int(values[int(np.argmax(frequencies))])
        mr, mc = map(int, np.argwhere(grid[top:bottom + 1, left:right + 1] == marker)[0])
        mr += top
        mc += left
        corner = (mr - square_top - s + 1, mc - square_left - s + 1)
        if corner not in CORNERS:
            raise ValueError("marker must lie in the central 2x2 knot")
        r, c = square_top, square_left
        if r < 0 or c < 0 or r + 2 * s > grid.shape[0] or c + 2 * s > grid.shape[1]:
            raise ValueError("the entire implied diamond must fit")
        region = occupied[r:r + 2 * s, c:c + 2 * s]
        diamond = _diamond(s)
        if np.any(region & diamond):
            raise ValueError("implied full diamonds must be disjoint")
        region |= diamond
        objects.append(dict(top=r, left=c, size=s, axis=axis, corner=corner,
                            payload=payload, marker=marker))
    if not objects:
        raise ValueError("at least one bar is required")
    return background, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read bars and their knots, then construct selected taper cross-sections."""
    grid = _grid(input_grid)
    background, objects = _parse(grid)
    output = np.full(grid.shape, background, dtype=np.int64)
    for obj in objects:
        s, r, c = obj["size"], obj["top"], obj["left"]
        main = obj["corner"][0] == obj["corner"][1]
        # Use the inherited parallel-cross-section construction, rather than
        # the Manhattan-distance formula used by rendering and validation.
        for d in range(s):
            if obj["axis"] == "H":
                sections = (((r + s - 1 - d, cc) for cc in range(c + d, c + 2 * s - d)),
                            ((r + s + d, cc) for cc in range(c + d, c + 2 * s - d)))
            else:
                sections = (((rr, c + s - 1 - d) for rr in range(r + d, r + 2 * s - d)),
                            ((rr, c + s + d) for rr in range(r + d, r + 2 * s - d)))
            for section in sections:
                for rr, cc in section:
                    if (((rr < r + s) == (cc < c + s)) == main):
                        output[rr, cc] = obj["payload"]
    return output


def _render(shape, background, objects):
    inp = np.full(shape, background, dtype=np.int64)
    out = inp.copy()
    for obj in objects:
        r, c, s = obj["top"], obj["left"], obj["size"]
        inp[r:r + 2 * s, c:c + 2 * s][_bar(s, obj["axis"])] = obj["payload"]
        mr, mc = obj["corner"]
        inp[r + s - 1 + mr, c + s - 1 + mc] = obj["marker"]
        out[r:r + 2 * s, c:c + 2 * s][_wings(s, obj["corner"])] = obj["payload"]
    return {"input": inp, "output": out}


def _layout(rng, specs):
    """Pack disjoint full diamonds; their bounding squares may overlap."""
    largest = 2 * max(s for s, _ in specs)
    for attempt in range(60):
        if attempt >= 40:
            shape = (30, 30)
        else:
            low = max(largest, 10 if len(specs) > 1 else 4)
            if attempt >= 15 and len(specs) > 1:
                low = max(low, 24)
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        if shape[0] * shape[1] <= 8 * sum(s for s, _ in specs):
            continue
        occupied = np.zeros(shape, dtype=bool)
        seeds = occupied.copy()
        objects = [None] * len(specs)
        edge_hint = rng.random() < 0.35
        for index in sorted(range(len(specs)), key=lambda i: -specs[i][0]):
            s, axis = specs[index]
            diamond, bar = _diamond(s), _bar(s, axis)
            positions = [(r, c) for r in range(shape[0] - 2 * s + 1)
                         for c in range(shape[1] - 2 * s + 1)]
            rng.shuffle(positions)
            if edge_hint and not np.any(occupied):
                positions.sort(key=lambda rc: not (rc[0] in (0, shape[0] - 2 * s)
                                                   or rc[1] in (0, shape[1] - 2 * s)))
            forbidden_seed = _touch4(seeds)
            for r, c in positions:
                region = np.s_[r:r + 2 * s, c:c + 2 * s]
                if np.any(occupied[region] & diamond) or np.any(forbidden_seed[region] & bar):
                    continue
                occupied[region] |= diamond
                seeds[region] |= bar
                objects[index] = dict(top=r, left=c, size=s, axis=axis)
                break
            else:
                break
        if all(obj is not None for obj in objects):
            return shape, objects
    # A guaranteed final placement for the sampler's <=4 objects of side <=7.
    # Fifteen-cell slots leave a background gap around every <=14-cell square.
    # This affects placement only; it never alters lengths or the rule.
    if len(specs) <= 4 and largest <= 14:
        slots = [(0, 0), (0, 15), (15, 0), (15, 15)]
        rng.shuffle(slots)
        return (30, 30), [dict(top=r, left=c, size=s, axis=axis)
                          for (s, axis), (r, c) in zip(specs, slots)]
    raise RuntimeError("could not pack the requested looms")


def _palette(rng):
    return tuple(map(int, rng.choice(10, 5, replace=False)))


def _decorate(rng, objects, palette, corners=None):
    _, p1, p2, m1, m2 = palette
    result = []
    for i, obj in enumerate(objects):
        corner = CORNERS[int(rng.integers(4))] if corners is None else corners[i]
        result.append(dict(obj, corner=corner, payload=(p1, p2)[i % 2],
                           marker=(m1, m2)[int(rng.integers(2))]))
    return result


def _key(grid):
    grid = np.asarray(grid, dtype=np.int64)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(2, 5))
    sizes = [base, base + 1, base + 2]
    rng.shuffle(sizes)
    axes = ["H", "V", ("H", "V")[int(rng.integers(2))]]
    rng.shuffle(axes)
    shape, layout = _layout(rng, list(zip(sizes, axes)))
    palette = _palette(rng)
    changing_palette = bool(rng.integers(2))
    first_corner = CORNERS[int(rng.integers(4))]
    main_corner = (0, 0) if rng.integers(2) else (1, 1)
    anti_corner = (0, 1) if rng.integers(2) else (1, 0)
    corners = [first_corner, main_corner, anti_corner]
    objects = _decorate(rng, layout, palette, corners)
    first = _render(shape, palette[0], objects)
    second_objects = [dict(obj) for obj in objects]
    # A controlled adjacent-corner change and both opposite-corner identities.
    if rng.integers(2):
        second_objects[0]["corner"] = (1 - first_corner[0], first_corner[1])
    else:
        second_objects[0]["corner"] = (first_corner[0], 1 - first_corner[1])
    for obj in second_objects[1:]:
        obj["corner"] = tuple(1 - x for x in obj["corner"])
    second = _render(shape, palette[0], second_objects)
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for index in range(count):
            for _ in range(40):
                n = int(rng.integers(1, 5))
                chosen_sizes = [int(rng.choice(sizes)) for _ in range(n)]
                if split == "test":
                    chosen_sizes[int(rng.integers(n))] = base + 3
                specs = [(s, ("H", "V")[int(rng.integers(2))]) for s in chosen_sizes]
                shape, raw = _layout(rng, specs)
                colors = _palette(rng) if changing_palette and not (split == "test" and index == 0) else palette
                decorated = _decorate(rng, raw, colors)
                pair = _render(shape, colors[0], decorated)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain a distinct input")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometry(obj):
    return obj["top"], obj["left"], obj["size"], obj["axis"], obj["payload"]


def validate_generated_puzzle(puzzle):
    """Check returned grids, local geometry, ablations, teaching, and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            background, objects = _parse(inp)
            assert 1 <= len(objects) <= 4
            expected = np.full(inp.shape, background, dtype=np.int64)
            count = 0
            for obj in objects:
                s, r, c = obj["size"], obj["top"], obj["left"]
                assert 2 <= s <= 7
                mask = _wings(s, obj["corner"])
                # Area and both ablations are checked for EVERY object.
                assert int(mask.sum()) == s * (s + 1)
                assert int(mask[:s, :s].sum() + mask[s:, s:].sum()) in (0, s * (s + 1))
                assert np.any(_diamond(s) & ~mask)  # Omitting A fills extra quarters.
                square_quarters = np.indices(mask.shape)
                square_quarters = ((square_quarters[0] < s) == (square_quarters[1] < s)) == (obj["corner"][0] == obj["corner"][1])
                assert np.any(square_quarters & ~mask)  # Omitting B gives solid squares.
                expected[r:r + 2 * s, c:c + 2 * s][mask] = obj["payload"]
                count += s * (s + 1)
                # Knot marker is recolored and its two adjacent corners disappear.
                knot = out[r + s - 1:r + s + 1, c + s - 1:c + s + 1]
                mr, mc = obj["corner"]
                assert knot[mr, mc] == knot[1 - mr, 1 - mc] == obj["payload"]
                assert knot[mr, 1 - mc] == knot[1 - mr, mc] == background
            assert np.array_equal(expected, out)
            assert np.count_nonzero(out != background) == count
            assert not np.array_equal(inp, out)
            assert np.array_equal(solve(inp), out)
            records[split].append((inp.shape, background, objects))
    train_objects = [o for _, _, objs in records["train"] for o in objs]
    assert {o["axis"] for o in train_objects} == {"H", "V"}
    assert len({o["size"] for o in train_objects}) == 3
    assert {o["corner"] for o in train_objects} == set(CORNERS)
    max_train = max(o["size"] for o in train_objects)
    assert all(max(o["size"] for o in objs) == max_train + 1 for _, _, objs in records["test"])
    # Find the matched training layouts rather than relying on example order.
    contrasts = []
    for i, (shape, bg, objects) in enumerate(records["train"]):
        for shape2, bg2, other in records["train"][i + 1:]:
            if shape != shape2 or bg != bg2:
                continue
            left = {_geometry(o): o for o in objects}
            right = {_geometry(o): o for o in other}
            if left.keys() != right.keys():
                continue
            adjacent, opposite_slopes = 0, set()
            for key, obj in left.items():
                a, b = obj["corner"], right[key]["corner"]
                if sum(x != y for x, y in zip(a, b)) == 1:
                    adjacent += 1
                elif all(x != y for x, y in zip(a, b)):
                    opposite_slopes.add(a[0] == a[1])
            contrasts.append(adjacent >= 1 and opposite_slopes == {True, False})
    assert any(contrasts), "training must teach adjacency and both opposite-corner identities"


def self_test():
    """Bounded 42-seed structural audit, symmetry, errors, and count coverage."""
    shapes, layouts, puzzles, palettes = set(), set(), set(), set()
    sizes, axes, corners, counts = set(), set(), set(), set()
    edge_pairs = bbox_overlap_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for split in ("train", "test") for p in puzzle[split])
        puzzles.add(signature)
        again = generate_puzzle(seed)
        assert signature == tuple((_key(p["input"]), _key(p["output"]))
                                  for split in ("train", "test") for p in again[split])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, objects = _parse(inp)
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(sorted(_geometry(o)[:4] + (o["corner"],) for o in objects))))
                palettes.add((bg, tuple(sorted({o["payload"] for o in objects}))))
                counts.add(len(objects))
                boxes = np.zeros(inp.shape, dtype=int)
                edge = False
                for o in objects:
                    s, r, c = o["size"], o["top"], o["left"]
                    sizes.add(s)
                    axes.add(o["axis"])
                    corners.add(o["corner"])
                    boxes[r:r + 2 * s, c:c + 2 * s] += 1
                    edge |= r == 0 or c == 0 or r + 2 * s == inp.shape[0] or c + 2 * s == inp.shape[1]
                edge_pairs += edge
                bbox_overlap_pairs += bool(np.any(boxes > 1))
                # Rotation/reflection are input-only metamorphic checks.
                assert np.array_equal(solve(np.rot90(inp)), np.rot90(pair["output"]))
                assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(pair["output"]))
    assert len(puzzles) == 42 and len(shapes) >= 40 and len(layouts) >= 160
    assert len(palettes) >= 35 and sizes == set(range(2, 8))
    assert axes == {"H", "V"} and corners == set(CORNERS) and counts == {1, 2, 3, 4}
    assert edge_pairs >= 20 and bbox_overlap_pairs >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The solver supports valid family sizes beyond the generator's packing range.
    for s in (2, 3, 8, 11, 15):
        for axis in ("H", "V"):
            obj = dict(top=0, left=0, size=s, axis=axis, corner=(0, 0), payload=3, marker=2)
            shape = (2 * s, 2 * s + (1 if s == 2 else 0))
            pair = _render(shape, 0, [obj])
            assert np.array_equal(solve(pair["input"]), pair["output"])
            opposite = _render(shape, 0, [dict(obj, corner=(1, 1))])
            adjacent = _render(shape, 0, [dict(obj, corner=(0, 1))])
            assert np.array_equal(pair["output"], solve(opposite["input"]))
            assert not np.array_equal(pair["output"], solve(adjacent["input"]))
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "shapes": len(shapes),
            "layouts": len(layouts), "palettes": len(palettes), "sizes": sorted(sizes),
            "object_counts": sorted(counts), "edge_pairs": edge_pairs,
            "overlapping_bounding_box_pairs": bbox_overlap_pairs}
