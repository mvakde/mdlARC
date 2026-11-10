"""Cross-laminated onion: boundary-depth grain controls a deletion/support puzzle.

Only the standard library and NumPy are used.  A line is (row, col, h, w, color),
where h == 1 or w == 1.  The solver decodes geometry solely from its input.
"""

import numpy as np


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is outside its legal range")
    return int(value)


def _choice(rng, values):
    values = list(values)
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _cells(p):
    r, c, h, w, _ = p
    return tuple((y, x) for y in range(r, r + h) for x in range(c, c + w))


def _shift(p, d):
    return (p[0] + d,) + p[1:]


def _laminate(top, left, h, w, code):
    pieces, depths = [], []
    layers = min(h, w) // 2
    for d in range(layers):
        width, height = w - 2 * d, h - 2 * d
        pieces.extend([(top + d, left + d, 1, width, int(code[d])),
                       (top + h - 1 - d, left + d, 1, width, int(code[d]))])
        depths.extend([d, d])
        if height > 2:
            color = int(code[layers - 1 - d])
            pieces.extend([(top + d + 1, left + d, height - 2, 1, color),
                           (top + d + 1, left + w - 1 - d, height - 2, 1, color)])
            depths.extend([d, d])
    return pieces, depths


def _render(shape, background, band, pieces):
    result = np.full(shape, background, dtype=np.int8)
    result[0] = band
    occupied = set()
    for p in pieces:
        cells = _cells(p)
        assert not occupied.intersection(cells), "pieces overlap"
        occupied.update(cells)
        r, c, h, w, color = p
        assert r >= 2 and r + h <= shape[0] and c >= 0 and c + w <= shape[1]
        result[r:r + h, c:c + w] = color
    return result


def _settle(pieces, height):
    """Event-distance line gravity. Lowest endpoints are processed first."""
    result = list(pieces)
    occupied = {cell: i for i, p in enumerate(result) for cell in _cells(p)}
    for _ in range(height + 1):
        changed = False
        order = sorted(range(len(result)), key=lambda i: result[i][0] + result[i][2], reverse=True)
        for i in order:
            p = result[i]
            for cell in _cells(p):
                del occupied[cell]
            distance = height - p[0] - p[2]
            # Only the bottom face of a solid line can encounter a new obstacle.
            bottom = p[0] + p[2] - 1
            for col in range(p[1], p[1] + p[3]):
                for row in range(bottom + 1, bottom + distance + 1):
                    if (row, col) in occupied:
                        distance = min(distance, row - bottom - 1)
                        break
            result[i] = _shift(p, distance)
            for cell in _cells(result[i]):
                assert cell not in occupied
                occupied[cell] = i
            changed |= distance > 0
        if not changed:
            return result
    raise AssertionError("line gravity did not converge")


def _settle_objects(objects, height):
    """Separate synchronous reference, also usable for competing bonded models.

    Each object is a tuple of (row, column, color) cells. Colors have no physics.
    """
    result = list(objects)
    for _ in range(height + 1):
        sets = [{(r, c) for r, c, _ in obj} for obj in result]
        targets = [{(r + 1, c) for r, c in cells} for cells in sets]
        moving = {i for i, cells in enumerate(targets) if all(r < height for r, c in cells)}
        while True:
            fixed = set(range(len(result))) - moving
            excluded = {i for i in moving if any(targets[i] & sets[j] for j in fixed)}
            if not excluded:
                break
            moving -= excluded
        if not moving:
            return result
        result = [tuple((r + 1, c, color) for r, c, color in obj) if i in moving else obj
                  for i, obj in enumerate(result)]
    raise AssertionError("synchronous gravity did not converge")


def _as_objects(pieces):
    return [tuple((r, c, p[4]) for r, c in _cells(p)) for p in pieces]


def _render_objects(scene, objects):
    out = np.full(scene["grid"].shape, scene["bg"], dtype=np.int8)
    out[0] = scene["grid"][0]
    used = set()
    for obj in objects:
        for r, c, color in obj:
            assert (r, c) not in used
            used.add((r, c))
            out[r, c] = color
    return out


def _decode(value):
    grid = _grid(value)
    rows, cols = grid.shape
    assert rows >= 6
    bg = int(grid[1, 0])
    assert np.all(grid[1] == bg)
    commands = tuple(int(c) for c in grid[0] if c != bg)
    assert 1 <= len(commands) <= 3 and len(set(commands)) == len(commands)
    occupied_rows = np.flatnonzero(np.any(grid[2:] != bg, axis=1))
    assert len(occupied_rows), "a laminate must be present"
    top = int(occupied_rows[0]) + 2
    xs = np.flatnonzero(grid[top] != bg)
    left, right = int(xs[0]), int(xs[-1]) + 1
    assert np.array_equal(xs, np.arange(left, right))
    mask = np.zeros(cols, dtype=bool)
    mask[left:right] = True
    bottom = top
    while bottom < rows and np.array_equal(grid[bottom] != bg, mask):
        bottom += 1
    h, w = bottom - top, right - left
    assert h >= 4 and w >= 4 and h % 2 == w % 2 == 0
    layers = min(h, w) // 2
    code = tuple(int(grid[top + d, left + d]) for d in range(layers))
    assert bg not in code
    pieces, depths = _laminate(top, left, h, w, code)
    slab_count = len(pieces)
    pier_columns = []
    for col in range(cols):
        ys = np.flatnonzero(grid[bottom:, col] != bg) + bottom
        if not len(ys):
            continue
        assert left <= col < right
        assert np.array_equal(ys, np.arange(int(ys[0]), rows)), "pier contains a gap"
        pier_columns.append(col)
        start = int(ys[0])
        while start < rows:
            end = start + 1
            while end < rows and grid[end, col] == grid[start, col]:
                end += 1
            assert end - start >= 2, "pier segments must be at least two cells"
            pieces.append((start, col, end - start, 1, int(grid[start, col])))
            start = end
    assert all(b - a >= 2 for a, b in zip(pier_columns, pier_columns[1:]))
    if bottom < rows:
        assert pier_columns and any(p[0] == bottom for p in pieces[slab_count:])
    assert np.array_equal(_render(grid.shape, bg, grid[0], pieces), grid), "invalid grain or extra cells"
    assert _settle(pieces, rows) == pieces, "initial scene is not stable"
    return {"grid": grid, "bg": bg, "commands": commands, "tile": (top, left, h, w),
            "code": code, "pieces": pieces, "slab_count": slab_count, "depths": depths,
            "pier_columns": pier_columns}


def _remaining(scene, region="both"):
    selected = set(scene["commands"])
    n = scene["slab_count"]
    return [(i, p) for i, p in enumerate(scene["pieces"])
            if not (p[4] in selected and (region == "both" or (i < n) == (region == "slab")))]


def _outcome(scene, region="both"):
    remaining = _remaining(scene, region)
    settled = _settle([p for _, p in remaining], scene["grid"].shape[0])
    positions = {i: p[0] for (i, _), p in zip(remaining, settled)}
    out = _render(scene["grid"].shape, scene["bg"], scene["grid"][0], settled)
    return out, remaining, settled, positions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a family member without training labels, seeds, or generation state."""
    return _outcome(_decode(input_grid))[0]


def _coupling(scene):
    """Each region's cuts must affect common surviving LAMINATE pieces."""
    full = _outcome(scene)[3]
    slab_only = _outcome(scene, "slab")[3]
    pier_only = _outcome(scene, "piers")[3]
    laminate_ids = [i for i in full if i < scene["slab_count"]]
    return (any(full[i] != slab_only[i] for i in laminate_ids),
            any(full[i] != pier_only[i] for i in laminate_ids))


def _grouped_alternative(scene, mode):
    grid, bg = scene["grid"], scene["bg"]
    selected = set(scene["commands"])
    groups = []
    if mode == "components":
        seen = set()
        for row in range(2, grid.shape[0]):
            for col in range(grid.shape[1]):
                color = int(grid[row, col])
                if color == bg or color in selected or (row, col) in seen:
                    continue
                seen.add((row, col))
                stack, group = [(row, col)], []
                while stack:
                    r, c = stack.pop()
                    group.append((r, c, color))
                    for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 2 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and (y, x) not in seen and grid[y, x] == color:
                            seen.add((y, x))
                            stack.append((y, x))
                groups.append(tuple(group))
    else:
        top, left, h, w = scene["tile"]
        for d in range(len(scene["code"])):
            if mode == "frames":
                cells = [(r, c) for r in range(top + d, top + h - d)
                         for c in range(left + d, left + w - d)
                         if min(r - top, c - left, top + h - 1 - r, left + w - 1 - c) == d]
                partitions = [cells]
            else:
                assert mode == "vertical_corners"
                a, b, y, z = top + d, top + h - 1 - d, left + d, left + w - 1 - d
                partitions = [[(r, y) for r in range(a, b + 1)],
                              [(r, z) for r in range(a, b + 1)],
                              [(a, c) for c in range(y + 1, z)],
                              [(b, c) for c in range(y + 1, z)]]
            for cells in partitions:
                obj = tuple((r, c, int(grid[r, c])) for r, c in cells if grid[r, c] not in selected)
                if obj:
                    groups.append(obj)
        groups.extend(_as_objects([p for i, p in _remaining(scene) if i >= scene["slab_count"]]))
    return _render_objects(scene, _settle_objects(groups, grid.shape[0]))


def _alternatives(scene):
    intended = _outcome(scene)[0]
    results = {name: not np.array_equal(intended, _grouped_alternative(scene, name))
               for name in ("components", "frames", "vertical_corners")}
    deleted = scene["grid"].copy()
    deleted[2:][np.isin(deleted[2:], scene["commands"])] = scene["bg"]
    results["no_gravity"] = not np.array_equal(intended, deleted)
    pixels = np.full(deleted.shape, scene["bg"], dtype=np.int8)
    pixels[0] = deleted[0]
    for col in range(deleted.shape[1]):
        values = deleted[2:, col][deleted[2:, col] != scene["bg"]]
        if len(values):
            pixels[-len(values):, col] = values
    results["pixels"] = not np.array_equal(intended, pixels)
    return results


def _code(rng, length, palette):
    values = [int(c) for c in rng.permutation(palette)]
    while len(values) < length:
        values.insert(int(rng.integers(len(values) + 1)), int(_choice(rng, palette)))
    return values


def _segmented_pier(rng, row, col, length, colors, split=True):
    if split and length >= 4 and len(colors) >= 2 and rng.random() < .5:
        count = int(rng.integers(2, min(3, length // 2) + 1))
        sizes = np.full(count, 2, dtype=int)
        sizes += rng.multinomial(length - 2 * count, np.full(count, 1 / count))
    else:
        sizes = [length]
    pieces, previous = [], None
    for size in sizes:
        color = int(_choice(rng, [c for c in colors if c != previous]))
        pieces.append((row, col, int(size), 1, color))
        row += int(size)
        previous = color
    return pieces


def _sample_pair(rng, layers, kind, max_commands):
    """Construct one shared slab/support scene, then accept observable effects."""
    for _ in range(120):
        palette = [int(c) for c in rng.permutation(10)]
        bg, available = palette[0], palette[1:]
        if kind == "coupled":
            richness = int(rng.integers(4, min(layers, 7) + 1))
        else:
            richness = int(rng.integers(3 if kind == "internal" else 2, min(layers, 8) + 1))
        code = _code(rng, layers, available[:richness])
        if kind == "internal":
            commands = [int(_choice(rng, sorted(set(code) - {code[0]})))]
        elif kind == "coupled":
            commands = [code[0], int(_choice(rng, sorted(set(code) - {code[0], code[-1]})))]
        else:
            count = int(rng.integers(1, max_commands + 1))
            pool = list(dict.fromkeys(code + available[richness:richness + 2]))
            commands = [int(c) for c in rng.choice(pool, size=min(count, len(pool)), replace=False)]

        h = w = 2 * layers
        slack = 2 * int(rng.integers(0, 4))
        if rng.random() < .5:
            h += min(slack, 24 - h)
        else:
            w += min(slack, 28 - w)
        top = int(rng.integers(2, min(5, 30 - h - 2) + 1))
        max_pier = min(9, 30 - top - h)
        min_pier = 3 if kind == "coupled" else 2
        pier_height = int(rng.integers(min_pier, max_pier + 1))
        if kind == "general" and rng.random() < .13:
            pier_height = 0
        height = top + h + pier_height
        margin = min(4, 30 - w)
        left = int(rng.integers(margin + 1))
        width = w + left + int(rng.integers(min(4, 30 - w - left) + 1))
        pieces, _ = _laminate(top, left, h, w, code)
        n_slab = len(pieces)
        if pier_height:
            parity = int(rng.integers(2))
            choices = list(range(left + parity, left + w, 2))
            count = int(rng.integers(2 if kind == "coupled" else 1, min(6, len(choices)) + 1))
            columns = sorted(int(c) for c in rng.choice(choices, size=count, replace=False))
            tall_count = int(rng.integers(1, count if kind == "coupled" else count + 1))
            # Keep a short landing close to the middle in the coupled case.
            short = min(columns, key=lambda c: abs(c - (left + w / 2)))
            tall_options = [c for c in columns if kind != "coupled" or c != short]
            tall = set(int(c) for c in rng.choice(tall_options, size=tall_count, replace=False))
            for col in columns:
                length = pier_height if col in tall else int(rng.integers(2, pier_height)) if pier_height > 2 else 2
                if kind == "internal":
                    colors = [c for c in available[:min(9, richness + 2)] if c not in commands]
                elif kind == "coupled":
                    colors = commands if col in tall else [c for c in available if c not in commands]
                else:
                    colors = available[:min(9, richness + 2)]
                pieces.extend(_segmented_pier(rng, height - length, col, length, colors,
                                              split=not (kind == "coupled" and col in tall)))
        band = np.full(width, bg, dtype=np.int8)
        for col, color in zip(rng.choice(width, size=len(commands), replace=False), rng.permutation(commands)):
            band[int(col)] = int(color)
        inp = _render((height, width), bg, band, pieces)
        scene = _decode(inp)
        assert scene["pieces"] == pieces[:n_slab] + sorted(pieces[n_slab:], key=lambda p: (p[1], p[0]))
        out, before, after, _ = _outcome(scene)
        moved = [q[0] - p[0] for (_, p), q in zip(before, after)]
        if kind == "internal":
            contrasts = _alternatives(scene)
            if not any(moved) or not (contrasts["frames"] or contrasts["components"]):
                continue
        if kind == "coupled" and not all(_coupling(scene)):
            continue
        return {"input": inp, "output": out}
    raise AssertionError("bounded scene sampling exhausted for " + kind)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    a = _integer(num_train, 2, 10, "num_train")
    b = _integer(num_test, 1, 4, "num_test")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # The minimum training set must teach the complete fixed convention.
    for _ in range(30):
        train = [_sample_pair(rng, int(rng.integers(4, 7)), "internal", 1),
                 _sample_pair(rng, int(rng.integers(5, 7)), "coupled", 2)]
        evidence = [_alternatives(_decode(p["input"])) for p in train]
        if all(any(e[name] for e in evidence) for name in evidence[0]):
            break
    else:
        raise AssertionError("bounded teaching construction exhausted")
    seen = {_key(p["input"]) for p in train}
    for _ in range(a - 2):
        for _ in range(30):
            pair = _sample_pair(rng, int(rng.integers(2, 7)), "general", 3)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("duplicate training inputs")
        seen.add(_key(pair["input"]))
        train.append(pair)
    command_max = max(len(_decode(p["input"])["commands"]) for p in train)
    tests = []
    for _ in range(b):
        for _ in range(40):
            pair = _sample_pair(rng, int(rng.integers(7, 11)), "general", command_max)
            scene = _decode(pair["input"])
            before, after = _outcome(scene)[1:3]
            if _key(pair["input"]) not in seen and any(q[0] > p[0] for (_, p), q in zip(before, after)):
                break
        else:
            raise AssertionError("bounded test sampling exhausted")
        seen.add(_key(pair["input"]))
        tests.append(pair)
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and np.array_equal(inp[:2], out[:2])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _decode(inp)
            expected, before, after, _ = _outcome(scene)
            assert np.array_equal(out, expected)
            assert all(p[1:] == q[1:] and q[0] >= p[0] for (_, p), q in zip(before, after))
            assert _settle(after, inp.shape[0]) == after
            (training if split == "train" else testing).append(scene)
    evidence = [_alternatives(s) for s in training]
    assert all(any(e[name] for e in evidence) for name in evidence[0]), "insufficient teaching contrasts"
    assert any(all(_coupling(s)) for s in training), "must teach coupled internal/external release"
    assert any(s["code"][0] not in s["commands"] and
               not any(p[4] in s["commands"] for p in s["pieces"][s["slab_count"]:]) and
               _alternatives(s)["no_gravity"] for s in training), "must teach an internal rupture"
    assert max(len(s["code"]) for s in training) <= 6
    assert all(7 <= len(s["code"]) <= 10 for s in testing)
    max_commands = max(len(s["commands"]) for s in training)
    assert all(len(s["commands"]) <= max_commands for s in testing)


def _features(pair):
    s = _decode(pair["input"])
    _, before, after, _ = _outcome(s)
    distances = [q[0] - p[0] for (_, p), q in zip(before, after)]
    role = {}
    code = tuple(role.setdefault(c, len(role)) for c in s["code"])
    piers = s["pieces"][s["slab_count"]:]
    return {"shape": s["grid"].shape, "tile": s["tile"][2:], "layers": len(code),
            "code": code, "background": s["bg"], "commands": len(s["commands"]),
            "pier_count": len(s["pier_columns"]), "segments": len(piers),
            "segmented": len(piers) > len(s["pier_columns"]),
            "recurrent": len(set(code)) < len(code),
            "adjacent_repeat": any(a == b for a, b in zip(code, code[1:])),
            "fall_distances": tuple(sorted(set(distances) - {0})),
            "structure": (s["tile"][2:], code,
                          tuple((p[0] - s["tile"][0] - s["tile"][2],
                                 p[1] - s["tile"][1], p[2]) for p in piers))}


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    records, signatures, colors = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        signatures.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            scene = _decode(pair["input"])
            survivors = [p for _, p in _remaining(scene)]
            reference = _settle_objects(_as_objects(survivors), pair["input"].shape[0])
            assert np.array_equal(_render_objects(scene, reference), pair["output"])
            assert np.array_equal(solve(pair["input"].copy()), pair["output"])
            colors.update(int(c) for c in np.unique(pair["input"]))
            records.append(_features(pair))
    assert len(signatures) == 42 and colors == set(range(10))
    assert len({f["structure"] for f in records}) >= 180
    assert len({f["shape"] for f in records}) >= 60
    assert len({f["tile"] for f in records}) >= 25
    assert {f["layers"] for f in records} >= set(range(2, 11))
    assert {f["background"] for f in records} == set(range(10))
    assert {f["commands"] for f in records} == {1, 2, 3}
    assert len({f["pier_count"] for f in records}) >= 5
    assert sum(f["segmented"] for f in records) >= 15
    assert sum(f["adjacent_repeat"] for f in records) >= 25
    assert sum(f["recurrent"] for f in records) >= 60
    assert sum(len(f["fall_distances"]) >= 2 for f in records) >= 70

    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4200 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    # A transparent internal rupture with exact, hand-derived coordinates.
    pieces, _ = _laminate(2, 0, 8, 8, [1, 2, 3, 4])
    inp = _render((10, 8), 0, np.array([3] + [0] * 7), pieces)
    scene = _decode(inp)
    _, before, after, _ = _outcome(scene)
    mapping = {p: q for (_, p), q in zip(before, after)}
    assert mapping[(3, 1, 1, 6, 2)][0] == 5  # local row 1 -> 3
    assert mapping[(8, 1, 1, 6, 2)][0] == 8  # local bottom row 6 stays
    assert mapping[(5, 2, 2, 1, 2)][0] == 6  # an upright falls one
    contrast = _alternatives(scene)
    assert all(contrast[name] for name in ("frames", "vertical_corners", "no_gravity", "pixels"))

    # Equal adjacent grain colors do not glue neighboring depth uprights.
    pieces, _ = _laminate(2, 0, 10, 10, [1, 2, 3, 4, 4])
    inp = _render((12, 10), 0, np.array([2] + [0] * 9), pieces)
    assert _alternatives(_decode(inp))["components"]

    # The rule also handles an absent command and complete construction removal.
    pieces, _ = _laminate(2, 0, 4, 6, [1, 1])
    for command in (1, 9):
        inp = _render((6, 6), 0, np.array([command] + [0] * 5), pieces)
        out = solve(inp)
        assert np.array_equal(out[:2], inp[:2])
        assert (np.all(out[2:] == 0) if command == 1 else np.array_equal(out, inp))

    # Solver scope is wider than the sampled train/test lengths and dimensions.
    for h, w in ((4, 4), (4, 30), (28, 4), (28, 30)):
        length = min(h, w) // 2
        code = [1 + d % 4 for d in range(length)]
        pieces, _ = _laminate(2, 0, h, w, code)
        inp = _render((h + 2, w), 0, np.array([2] + [0] * (w - 1)), pieces)
        scene = _decode(inp)
        ref = _settle_objects(_as_objects([p for _, p in _remaining(scene)]), h + 2)
        assert np.array_equal(solve(inp), _render_objects(scene, ref))

    broken = generate_puzzle(42, 2, 1)
    broken["test"][0]["output"][-1, -1] = (int(broken["test"][0]["output"][-1, -1]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")
