"""Synthetic instances of the repeated-tile, collinear-mark ARC rule."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    "8789cf4b49202567ba07c9e6173a0d6d10c5b298ad47f687644fc10e1bc4bdb8",
    "9b822bfc5b5dbcd05f0b5f73ce2dd8c38384d405772fdc69d3d32a32d7eca980",
    "98ed3d9b6a49974ba02ec0757211d1a87da9b5b8646f840c4f82ff82e5b206d3",
    "756461526491dca6ee5efc72b45d054246db7809854d1707c0a28b094410ad80",
}
_TILE_SHAPES = ((4, 4), (4, 4), (4, 4), (4, 4), (4, 5), (5, 4), (5, 5), (3, 4), (4, 3))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def _counts(rng, num_train, num_test):
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        choices = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return choices[int(rng.integers(len(choices)))]
    if nt is None:
        return int(rng.integers(max(2, ns + 1), 11)), ns
    if ns is None:
        return nt, int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _layout(rng, regime):
    th, tw = _TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]
    max_rows, max_cols = 29 // (th + 1), 29 // (tw + 1)
    # Test instances lean toward larger lattices, but the overlap with
    # training remains broad, including smaller test examples.
    large = int(rng.integers(5 if regime == "target_test" else 12)) < 2
    rows = max_rows if large else int(rng.integers(3, max_rows + 1))
    cols = max_cols if large else int(rng.integers(3, max_cols + 1))
    return rows, cols, th, tw


def _palette(rng, group_count):
    colors = [int(c) for c in rng.permutation(10)]
    sep, structural, pattern = colors[:3]
    # The roles are legible within this pair; no literal color has a fixed role.
    distinct = int(rng.integers(1, min(group_count, 4) + 1))
    signals = colors[3:3 + distinct]
    while len(signals) < group_count:
        signals.append(signals[int(rng.integers(distinct))])
    rng.shuffle(signals)
    return sep, structural, pattern, signals


def _template(rng, th, tw, structural, pattern):
    tile = np.full((th, tw), structural, dtype=np.int8)
    inner = [(r, c) for r in range(1, th - 1) for c in range(1, tw - 1)]
    if (th, tw) == (4, 4):
        amount = int(rng.choice((2, 3, 3, 4, 4)))
    else:
        amount = int(rng.integers(max(1, len(inner) // 3), len(inner)))
    for index in rng.choice(len(inner), size=amount, replace=False):
        tile[inner[int(index)]] = pattern
    return tile


def _base(rows, cols, template, separator):
    th, tw = template.shape
    grid = np.full((rows * (th + 1) + 1, cols * (tw + 1) + 1), separator, dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            grid[1 + row * (th + 1):1 + row * (th + 1) + th,
                 1 + col * (tw + 1):1 + col * (tw + 1) + tw] = template
    return grid


def _put(grid, tile, coord, color, th, tw):
    grid[1 + tile[0] * (th + 1) + coord[0],
         1 + tile[1] * (tw + 1) + coord[1]] = color


def _closure(points):
    points = set(points)
    result = set(points)
    for row in {p[0] for p in points}:
        columns = [c for r, c in points if r == row]
        if len(columns) >= 2:
            result.update((row, c) for c in range(min(columns), max(columns) + 1))
    for col in {p[1] for p in points}:
        rows = [r for r, c in points if c == col]
        if len(rows) >= 2:
            result.update((r, col) for r in range(min(rows), max(rows) + 1))
    return result


def _shape(rng, rows, cols, kind):
    # Sampling endpoints rather than a template bank allows interior starts and
    # varying short and long gaps. Invalid tiny arms get a bounded retry.
    for _ in range(40):
        if kind == "H":
            r = int(rng.integers(rows))
            c0, c1 = sorted(int(x) for x in rng.choice(cols, 2, replace=False))
            points = {(r, c0), (r, c1)}
        elif kind == "V":
            c = int(rng.integers(cols))
            r0, r1 = sorted(int(x) for x in rng.choice(rows, 2, replace=False))
            points = {(r0, c), (r1, c)}
        elif kind == "L":
            r0, r1 = sorted(int(x) for x in rng.choice(rows, 2, replace=False))
            c0, c1 = sorted(int(x) for x in rng.choice(cols, 2, replace=False))
            corner = (int(rng.choice((r0, r1))), int(rng.choice((c0, c1))))
            points = {corner, (r0 + r1 - corner[0], corner[1]),
                      (corner[0], c0 + c1 - corner[1])}
        elif kind == "T":
            if cols >= 4 and (rows < 4 or int(rng.integers(2))):
                c0, cm, c1 = sorted(int(x) for x in rng.choice(cols, 3, replace=False))
                r0, r1 = [int(x) for x in rng.choice(rows, 2, replace=False)]
                points = {(r0, c0), (r0, cm), (r0, c1), (r1, cm)}
            else:
                r0, rm, r1 = sorted(int(x) for x in rng.choice(rows, 3, replace=False))
                c0, c1 = [int(x) for x in rng.choice(cols, 2, replace=False)]
                points = {(r0, c0), (rm, c0), (r1, c0), (rm, c1)}
        else:  # plus
            if rows < 3 or cols < 3:
                continue
            r = int(rng.integers(1, rows - 1))
            c = int(rng.integers(1, cols - 1))
            points = {(r, c), (int(rng.integers(0, r)), c),
                      (int(rng.integers(r + 1, rows)), c),
                      (r, int(rng.integers(0, c))),
                      (r, int(rng.integers(c + 1, cols)))}
        if _closure(points) - points and len(points) * 2 < rows * cols:
            return points
    if kind == "H":
        return {(0, 0), (0, cols - 1)}
    if kind == "V":
        return {(0, 0), (rows - 1, 0)}
    if kind == "L":
        return {(0, 0), (rows - 1, 0), (0, cols - 1)}
    if kind == "T":
        if cols >= 4:
            return {(0, 0), (0, 1), (0, cols - 1), (rows - 1, 1)}
        return {(0, 0), (1, 0), (rows - 1, 0), (1, cols - 1)}
    r, c = rows // 2, cols // 2
    return {(r, c), (0, c), (rows - 1, c), (r, 0), (r, cols - 1)}


def _parse(grid):
    assert grid.ndim == 2
    sep = int(grid[0, 0])
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    assert len(sep_rows) >= 4 and len(sep_cols) >= 4
    assert sep_rows[0] == 0 and sep_rows[-1] == grid.shape[0] - 1
    assert sep_cols[0] == 0 and sep_cols[-1] == grid.shape[1] - 1
    th = sep_rows[1] - sep_rows[0] - 1
    tw = sep_cols[1] - sep_cols[0] - 1
    assert th >= 3 and tw >= 3
    assert all(b - a == th + 1 for a, b in zip(sep_rows, sep_rows[1:]))
    assert all(b - a == tw + 1 for a, b in zip(sep_cols, sep_cols[1:]))
    rows, cols = len(sep_rows) - 1, len(sep_cols) - 1
    tiles = np.stack([grid[sep_rows[r] + 1:sep_rows[r + 1],
                           sep_cols[c] + 1:sep_cols[c + 1]]
                      for r in range(rows) for c in range(cols)]).reshape(rows, cols, th, tw)
    assert not np.any(tiles == sep)
    template = np.empty((th, tw), dtype=np.int8)
    for r in range(th):
        for c in range(tw):
            counts = np.bincount(tiles[:, :, r, c].reshape(-1).astype(int), minlength=10)
            winners = np.flatnonzero(counts == counts.max())
            assert len(winners) == 1
            assert counts[winners[0]] > rows * cols // 2
            template[r, c] = winners[0]
    groups = {}
    for tr in range(rows):
        for tc in range(cols):
            for lr in range(th):
                for lc in range(tw):
                    color = int(tiles[tr, tc, lr, lc])
                    if color != int(template[lr, lc]):
                        groups.setdefault((color, lr, lc), set()).add((tr, tc))
    return rows, cols, template, sep, groups


def _predict(input_grid):
    rows, cols, template, sep, groups = _parse(input_grid)
    th, tw = template.shape
    output = input_grid.copy()
    for (color, lr, lc), points in groups.items():
        for tile in _closure(points):
            _put(output, tile, (lr, lc), color, th, tw)
    return output


def _coordinates(rng, template, paths, colors, force_roles):
    th, tw = template.shape
    structural = int(template[0, 0])
    # Pattern is the distinct interior role, regardless of numeric ordering.
    pattern = next(int(v) for v in np.unique(template) if int(v) != structural)
    coords = []
    for index, path in enumerate(paths):
        preferred = None
        if force_roles and index < 2:
            preferred = (pattern, structural)[index]
        candidates = []
        for r in range(th):
            for c in range(tw):
                coord = (r, c)
                if preferred is not None and int(template[coord]) != preferred:
                    continue
                valid = True
                for previous, other_path, other_color in zip(coords, paths[:index], colors[:index]):
                    if coord == previous and (_closure(path) & _closure(other_path) or colors[index] == other_color):
                        valid = False
                        break
                if valid:
                    candidates.append(coord)
        if not candidates:
            return None
        # A strong interior bias preserves the original tile vocabulary, while
        # border signals remain an occasional extension.
        weights = np.array([8 if 0 < r < th - 1 and 0 < c < tw - 1 else 1
                            for r, c in candidates], dtype=float)
        # Deliberately reuse a coordinate when disjoint completed paths permit it.
        reusable = [i for i, coord in enumerate(candidates) if coord in coords]
        if reusable and int(rng.integers(3)) == 0:
            weights[reusable] *= 15
        weights /= weights.sum()
        coords.append(candidates[int(rng.choice(len(candidates), p=weights))])
    return coords


def _fallback_pair(rng, first, serial):
    # A legal, shape-distinct fallback for each of the at most 14 pair slots.
    rows = 3 + serial % 3
    cols = 3 + (serial // 3) % 3
    th, tw = ((4, 4) if serial < 9 else (3, 4))
    colors = [int(c) for c in rng.permutation(10)]
    sep, structural, pattern, signal, other = colors[:5]
    template = np.full((th, tw), structural, dtype=np.int8)
    template[1, 1:3] = pattern
    inp = _base(rows, cols, template, sep)
    out = inp.copy()
    groups = [(signal, (1, 1), {(0, 0), (rows - 1, 0), (0, cols - 1)})]
    if first:
        groups.append((other, (0, 1), {(rows - 1, 0), (rows - 1, cols - 1)}))
    for color, coord, points in groups:
        for tile in points:
            _put(inp, tile, coord, color, th, tw)
        for tile in _closure(points):
            _put(out, tile, coord, color, th, tw)
    return {"input": inp, "output": out}


def _make_pair(rng, regime, first=False, serial=0):
    for _ in range(100):
        rows, cols, th, tw = _layout(rng, regime)
        area = rows * cols
        if first:
            n = int(rng.integers(2, min(4, area // 3) + 1))
        elif regime == "target_test":
            choices = np.arange(1, min(4, area // 3) + 1)
            weights = choices.astype(float)
            weights /= weights.sum()
            n = int(rng.choice(choices, p=weights))
        else:
            n = int(rng.integers(1, min(4, area // 3) + 1))
        sep, structural, pattern, colors = _palette(rng, n)
        template = _template(rng, th, tw, structural, pattern)
        kinds = []
        for index in range(n):
            options = ["H", "V", "L", "T", "plus"] if area >= 16 else ["H", "V", "L"]
            if first and index == 0:
                kind = "L"
            elif first and index == 1:
                kind = "H" if int(rng.integers(2)) else "V"
            else:
                kind = options[int(rng.integers(len(options)))]
            kinds.append(kind)
        paths = [_shape(rng, rows, cols, kind) for kind in kinds]
        coords = _coordinates(rng, template, paths, colors, first)
        if coords is None:
            continue
        inp = _base(rows, cols, template, sep)
        out = inp.copy()
        valid = True
        occupied = {}
        for color, coord, path in zip(colors, coords, paths):
            for tile in _closure(path):
                key = (tile, coord)
                if key in occupied and occupied[key] != color:
                    valid = False
                occupied[key] = color
            for tile in path:
                _put(inp, tile, coord, color, th, tw)
            for tile in _closure(path):
                _put(out, tile, coord, color, th, tw)
        if not valid:
            continue
        # Single marks and diagonal pairs can be inert for geometric reasons,
        # including when their literal color happens to be orange.
        if int(rng.integers(4)) == 0:
            available = [int(c) for c in range(10) if c not in (sep, structural, pattern) and c not in colors]
            free = [(r, c) for r in range(th) for c in range(tw) if (r, c) not in coords]
            if available and free:
                color = available[int(rng.integers(len(available)))]
                coord = free[int(rng.integers(len(free)))]
                tile = (int(rng.integers(rows)), int(rng.integers(cols)))
                _put(inp, tile, coord, color, th, tw)
                _put(out, tile, coord, color, th, tw)
        if np.array_equal(inp, out):
            continue
        try:
            parsed = _parse(inp)
            if not np.array_equal(_predict(inp), out):
                continue
            if parsed[2].shape != template.shape or not np.array_equal(parsed[2], template):
                continue
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    return _fallback_pair(rng, first, serial)


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_hash(inp, out):
    return hashlib.sha256(str(inp.shape).encode() + inp.astype(np.int8).tobytes()
                          + out.astype(np.int8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    # The geometric interpolation rule is the only puzzle-wide convention.
    # Pair roles and geometry are sampled independently, as in the source task.
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=nt + ns, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, offset in (("train", nt, 0), ("test", ns, nt)):
        for i in range(count):
            pair_rng = np.random.default_rng(pair_seeds[offset + i])
            for _ in range(20):
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  first=(split == "train" and i == 0), serial=offset + i)
                key = _key(pair["input"])
                if key not in seen and _source_hash(pair["input"], pair["output"]) not in _SOURCE_HASHES:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, out = pair["input"], pair["output"]
    rows, cols, template, sep, groups = _parse(inp)
    structural = int(template[0, 0])
    colors = set(int(v) for v in np.unique(template))
    assert len(colors) == 2 and sep not in colors
    assert np.all(template[0, :] == structural) and np.all(template[-1, :] == structural)
    assert np.all(template[:, 0] == structural) and np.all(template[:, -1] == structural)
    signal_colors = {key[0] for key in groups}
    assert not signal_colors & (colors | {sep})
    productive = [key for key, points in groups.items() if _closure(points) - points]
    assert productive
    # Audit every output tile cell directly from its input marks. This uses
    # row/column brackets instead of the construction-time path sets.
    th, tw = template.shape
    for tr in range(rows):
        for tc in range(cols):
            for lr in range(th):
                for lc in range(tw):
                    y, x = 1 + tr * (th + 1) + lr, 1 + tc * (tw + 1) + lc
                    expected = int(inp[y, x])
                    matches = set()
                    for (color, gr, gc), points in groups.items():
                        if (gr, gc) != (lr, lc):
                            continue
                        row_cols = [c for r, c in points if r == tr]
                        col_rows = [r for r, c in points if c == tc]
                        if ((len(row_cols) >= 2 and min(row_cols) <= tc <= max(row_cols))
                                or (len(col_rows) >= 2 and min(col_rows) <= tr <= max(col_rows))):
                            matches.add(color)
                    assert len(matches) <= 1
                    if matches:
                        expected = next(iter(matches))
                    assert int(out[y, x]) == expected
    assert np.all(inp[::th + 1, :] == out[::th + 1, :])
    assert np.all(inp[:, ::tw + 1] == out[:, ::tw + 1])
    assert not np.array_equal(inp, out)
    roles = set()
    horizontal = vertical = False
    for (color, lr, lc), points in groups.items():
        roles.add(int(template[lr, lc]))
        for r in range(rows):
            if sum(p[0] == r for p in points) >= 2:
                horizontal = True
        for c in range(cols):
            if sum(p[1] == c for p in points) >= 2:
                vertical = True
    return {"layout": (rows, cols), "tile": template.shape,
            "roles": roles, "horizontal": horizontal, "vertical": vertical,
            "palette": (sep, structural, next(iter(colors - {structural}))),
            "groups": groups, "added": int(np.count_nonzero(inp != out))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_horizontal = train_vertical = False
    train_roles = set()
    palettes = set()
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            assert _source_hash(inp, out) not in _SOURCE_HASHES
            features = _features(pair)
            palettes.add(features["palette"])
            if split == "train":
                train_horizontal |= features["horizontal"]
                train_vertical |= features["vertical"]
                train_roles.update(features["roles"])
                if i == 0:
                    assert len([g for g, p in features["groups"].items() if _closure(p) - p]) >= 2
                    assert features["roles"] == set(features["palette"][1:])
                    assert any(
                        any(sum(r == rr for rr, _ in points) >= 2 for r in range(features["layout"][0]))
                        and any(sum(c == cc for _, cc in points) >= 2 for c in range(features["layout"][1]))
                        for points in features["groups"].values()
                    )
    assert train_horizontal and train_vertical
    # The first demonstration already shows both template roles and both axes.
    assert len(train_roles) >= 2
    assert len(palettes) >= 2


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    for args in ((None, None), (2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1317, *args)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(1317, *args))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(args)
    # Reconstruct the orange alignment in source train 3 at tile (0,1),(2,1).
    tile = np.full((4, 4), 8, dtype=np.int8)
    tile[1:3, 1:3] = 4
    demo = _base(3, 3, tile, 0)
    _put(demo, (0, 1), (1, 1), 7, 4, 4)
    _put(demo, (2, 1), (1, 1), 7, 4, 4)
    assert _predict(demo)[1 + 1 * 5 + 1, 1 + 1 * 5 + 1] == 7
    for serial in range(14):
        _features(_fallback_pair(np.random.default_rng(serial), serial == 0, serial))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    counts = ({len(p["train"]) for p in puzzles}, {len(p["test"]) for p in puzzles})
    assert len(counts[0]) >= 4 and len(counts[1]) >= 3
    assert not _equal(puzzles[0], puzzles[1])
    layouts, tiles, palettes, additions, masks, all_inputs = set(), set(), set(), set(), set(), set()
    reused = interior = border = both_roles = 0
    split_counts = {s: {"simple": 0, "dense": 0, "junction": 0, "orange": 0,
                        "rectangular": 0, "pattern": 0, "structure": 0}
                    for s in ("train", "test")}
    split_layouts = {"train": set(), "test": set()}
    split_patterns = {"train": set(), "test": set()}
    changing_palettes = changing_layouts = changing_additions = 0
    literal_role_changes = test_not_larger = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        within = {"palette": set(), "layout": set(), "added": set()}
        literal_roles = {}
        train_areas = []
        test_areas = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                within["palette"].add(f["palette"])
                within["layout"].add(f["layout"])
                within["added"].add(f["added"])
                for role, color in zip(("separator", "structure", "pattern"), f["palette"]):
                    literal_roles.setdefault(color, set()).add(role)
                for color, _, _ in f["groups"]:
                    literal_roles.setdefault(color, set()).add("signal")
                layouts.add(f["layout"])
                split_layouts[split].add(f["layout"])
                tiles.add(f["tile"])
                palettes.add(f["palette"])
                additions.add(f["added"])
                template = _parse(pair["input"])[2]
                masks.add(tuple(int(v) for v in template.reshape(-1)))
                pattern = f["palette"][2]
                split_patterns[split].add(int(np.sum(template == pattern)))
                area = f["layout"][0] * f["layout"][1]
                (train_areas if split == "train" else test_areas).append(area)
                key = _key(pair["input"])
                assert key not in all_inputs
                all_inputs.add(key)
                coord_colors = {}
                th, tw = f["tile"]
                productive = []
                for (color, lr, lc), points in f["groups"].items():
                    coord_colors.setdefault((lr, lc), set()).add(color)
                    interior += int(0 < lr < th - 1 and 0 < lc < tw - 1)
                    border += int(lr in (0, th - 1) or lc in (0, tw - 1))
                    if _closure(points) - points:
                        productive.append((color, points))
                        split_counts[split]["orange"] += color == 7
                        has_h = any(sum(r == rr for rr, _ in points) >= 2
                                    for r in range(f["layout"][0]))
                        has_v = any(sum(c == cc for _, cc in points) >= 2
                                    for c in range(f["layout"][1]))
                        split_counts[split]["junction"] += has_h and has_v and len(points) >= 4
                    split_counts[split]["pattern" if int(template[lr, lc]) == pattern
                                        else "structure"] += len(points)
                split_counts[split]["simple"] += len(productive) == 1
                split_counts[split]["dense"] += len(productive) >= 4
                split_counts[split]["rectangular"] += th != tw
                reused += any(len(v) > 1 for v in coord_colors.values())
                both_roles += len(f["roles"]) == 2
        changing_palettes += len(within["palette"]) > 1
        changing_layouts += len(within["layout"]) > 1
        changing_additions += len(within["added"]) > 1
        literal_role_changes += any(len(roles) > 1 for roles in literal_roles.values())
        test_not_larger += any(area <= max(train_areas) for area in test_areas)
    assert len(layouts) >= 8 and len(tiles) >= 4 and len(palettes) >= 20
    assert len(additions) >= 8 and len(masks) >= 20
    assert interior > border and border > 0 and both_roles > 0 and reused > 0
    assert changing_palettes == changing_layouts == literal_role_changes == 42
    assert changing_additions >= 35 and test_not_larger > 0
    assert {(3, 3), (4, 4), (5, 4)} <= split_layouts["train"]
    assert len(split_layouts["test"]) >= 8
    assert {2, 3, 4} <= split_patterns["train"] & split_patterns["test"]
    for split in ("train", "test"):
        assert all(split_counts[split][name] > 0 for name in split_counts[split])
