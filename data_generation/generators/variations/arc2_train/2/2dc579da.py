"""Generate puzzles that crop the sole marked compartment of a divided grid."""

import numpy as np


def _rows(*lines):
    return np.array([[int(c) for c in line] for line in lines], dtype=np.int8)


_SOURCE_EXAMPLES = (
    (_rows("4442444", "4442414", "4442444", "2222222", "4442444", "4442444", "4442444"),
     _rows("444", "414", "444")),
    (_rows("88388", "88388", "33333", "88388", "48388"), _rows("88", "48")),
    (_rows("33333133333", "33333133333", "38333133333", "33333133333", "33333133333",
           "11111111111", "33333133333", "33333133333", "33333133333", "33333133333",
           "33333133333"),
     _rows("33333", "33333", "38333", "33333", "33333")),
    (_rows("1111110111111", "1111110111111", "1111110111111", "1111110121111",
           "1111110111111", "1111110111111", "0000000000000", "1111110111111",
           "1111110111111", "1111110111111", "1111110111111", "1111110111111",
           "1111110111111"),
     _rows("111111", "111111", "111111", "121111", "111111", "111111")),
)

_STYLES = ("single", "bar", "cluster", "hollow", "diagonal", "disconnected", "multi")
_KINDS = ("classic", "rectangular", "unequal", "expanded")
_REGIMES = ("larger", "overlap", "equal", "smaller", "mixed")


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        options = [(a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)]
        return options[int(rng.integers(len(options)))]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _sizes(rng, train_count, test_count, regime):
    if regime == "larger":
        train = [int(rng.integers(2, 6)) for _ in range(train_count)]
        test = [int(rng.integers(6, 13)) for _ in range(test_count)]
    elif regime == "smaller":
        train = [int(rng.integers(6, 13)) for _ in range(train_count)]
        test = [int(rng.integers(2, 6)) for _ in range(test_count)]
    elif regime == "equal":
        q = int(rng.integers(3, 11))
        train, test = [q] * train_count, [q] * test_count
    elif regime == "overlap":
        train = [int(rng.integers(3, 11)) for _ in range(train_count)]
        test = [int(rng.integers(3, 11)) for _ in range(test_count)]
        test[0] = train[int(rng.integers(train_count))]
    else:
        train = [int(rng.integers(2, 13)) for _ in range(train_count)]
        test = [int(rng.integers(2, 13)) for _ in range(test_count)]
        train[0], train[1] = int(rng.integers(2, 6)), int(rng.integers(7, 13))
        if test_count > 1:
            test[0], test[1] = int(rng.integers(2, 6)), int(rng.integers(7, 13))
    return train, test


def _palette(rng, multi):
    colors = [int(v) for v in rng.choice(10, size=5 if multi else 3, replace=False)]
    return colors


def _geometry(rng, q, kind, position):
    """Return compartment sizes, divider widths, and marked compartment index."""
    if kind == "classic":
        heights, widths, rh, cw = [q, q], [q, q], 1, 1
    elif kind == "rectangular":
        w = int(np.clip(q + int(rng.integers(-3, 4)), 2, 14))
        if w == q:
            w = q + 1 if q < 14 else q - 1
        heights, widths, rh, cw = [q, q], [w, w], 1, 1
    elif kind == "unequal":
        rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        w = int(np.clip(q + int(rng.integers(-3, 4)), 2, 14))
        heights = [q, q]
        widths = [w, w]
        other_h = int(rng.integers(2, min(10, 30 - q - rh) + 1))
        other_w = int(rng.integers(2, min(10, 30 - w - cw) + 1))
        heights[1 - position[0]] = other_h
        widths[1 - position[1]] = other_w
        if heights[0] == heights[1] and widths[0] == widths[1]:
            heights[1 - position[0]] = 2 if q != 2 else 3
    elif kind == "expanded":
        rows, cols = ((2, 3), (3, 2), (3, 3))[int(rng.integers(3))]
        rh, cw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        if bool(rng.integers(2)):
            position = (2, int(rng.integers(cols))) if rows == 3 else (int(rng.integers(rows)), 2)
        w = int(np.clip(q + int(rng.integers(-3, 4)), 2, 14))
        heights = [int(rng.integers(2, 6)) for _ in range(rows)]
        widths = [int(rng.integers(2, 6)) for _ in range(cols)]
        heights[position[0]], widths[position[1]] = q, w
    else:
        raise AssertionError("unknown geometry")
    assert sum(heights) + rh * (len(heights) - 1) <= 30
    assert sum(widths) + cw * (len(widths) - 1) <= 30
    return heights, widths, rh, cw, position


def _mask(rng, height, width, style):
    mask = np.zeros((height, width), dtype=bool)
    if style == "single":
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    elif style == "bar":
        horizontal = bool(rng.integers(2))
        limit = width if horizontal else height
        length = int(rng.integers(2, limit + 1))
        if horizontal:
            r, c = int(rng.integers(height)), int(rng.integers(width - length + 1))
            mask[r, c:c + length] = True
        else:
            r, c = int(rng.integers(height - length + 1)), int(rng.integers(width))
            mask[r:r + length, c] = True
    elif style == "diagonal":
        length = int(rng.integers(2, min(height, width) + 1))
        r, c = int(rng.integers(height - length + 1)), int(rng.integers(width - length + 1))
        reverse = bool(rng.integers(2))
        for k in range(length):
            mask[r + k, c + (length - 1 - k if reverse else k)] = True
    elif style == "hollow":
        h, w = int(rng.integers(3, height + 1)), int(rng.integers(3, width + 1))
        r, c = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))
        mask[r, c:c + w] = True
        mask[r + h - 1, c:c + w] = True
        mask[r:r + h, c] = True
        mask[r:r + h, c + w - 1] = True
    elif style == "disconnected":
        cells = [(r, c) for r in range(height) for c in range(width)]
        order = rng.permutation(len(cells))
        wanted = int(rng.integers(2, min(14, max(3, height * width // 3)) + 1))
        for index in order:
            r, c = cells[int(index)]
            if not any(abs(r - a) + abs(c - b) == 1 for a, b in np.argwhere(mask)):
                mask[r, c] = True
                if int(mask.sum()) >= wanted:
                    break
        if int(mask.sum()) < 2:
            mask[:] = False
            mask[0, 0] = mask[height - 1, width - 1] = True
    else:  # cluster, also used as the base of a multicolor motif
        wanted = int(rng.integers(3, min(height * width - 1, max(4, int(height * width * .6))) + 1))
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
        while int(mask.sum()) < wanted:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            choices = sorted(frontier)
            r, c = choices[int(rng.integers(len(choices)))]
            mask[r, c] = True
    return mask


def _motif(rng, height, width, fill, accents, style):
    if style == "hollow" and min(height, width) < 3:
        style = "diagonal"
    if style == "multi" and height * width <= 4:
        base = "diagonal"
    elif style == "multi":
        base = ("bar", "cluster", "hollow", "disconnected")[int(rng.integers(4))]
        if base == "hollow" and min(height, width) < 3:
            base = "cluster"
    else:
        base = style
    mask = _mask(rng, height, width, base)
    assert 0 < int(mask.sum()) < height * width
    out = np.full((height, width), fill, dtype=np.int8)
    if style == "multi":
        coords = np.argwhere(mask)
        out[tuple(coords[0])] = accents[0]
        out[tuple(coords[1])] = accents[1]
        for r, c in coords[2:]:
            out[r, c] = accents[int(rng.integers(len(accents)))]
    else:
        out[mask] = accents[0]
    return out


def _segments(sizes, band):
    starts = []
    cursor = 0
    for size in sizes:
        starts.append(slice(cursor, cursor + size))
        cursor += size + band
    return starts


def _pair(rng, q, kind, position, style, colors):
    heights, widths, rh, cw, position = _geometry(rng, q, kind, position)
    fill, divider = colors[:2]
    grid = np.full((sum(heights) + rh * (len(heights) - 1),
                    sum(widths) + cw * (len(widths) - 1)), fill, dtype=np.int8)
    rows, cols = _segments(heights, rh), _segments(widths, cw)
    for row in rows[:-1]:
        grid[row.stop:row.stop + rh, :] = divider
    for col in cols[:-1]:
        grid[:, col.stop:col.stop + cw] = divider
    output = _motif(rng, heights[position[0]], widths[position[1]], fill, colors[2:], style)
    grid[rows[position[0]], cols[position[1]]] = output
    return {"input": grid, "output": output}


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _source_copy(pair):
    return any(np.array_equal(pair["input"], a) and np.array_equal(pair["output"], b)
               for a, b in _SOURCE_EXAMPLES)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    regime = _REGIMES[int(rng.integers(len(_REGIMES)))]
    simple = bool(rng.integers(7) == 0)
    rich = not simple and bool(rng.integers(2))
    train_sizes, test_sizes = _sizes(rng, train_count, test_count, regime)
    positions = [(r, c) for r in range(2) for c in range(2)]
    rng.shuffle(positions)
    train_positions = [positions[0], positions[1]] + [positions[int(rng.integers(4))]
                                                  for _ in range(train_count - 2)]
    overlap = bool(rng.integers(2))
    test_positions = []
    for i in range(test_count):
        options = train_positions if overlap and i == 0 else positions
        if not overlap and i == 0:
            options = [p for p in positions if p not in train_positions] or positions
        test_positions.append(options[int(rng.integers(len(options)))])
    train, test = [], []
    seen = set()
    palette_history = []
    for split, sizes, chosen_positions in ((train, train_sizes, train_positions),
                                            (test, test_sizes, test_positions)):
        for i, (q, position) in enumerate(zip(sizes, chosen_positions)):
            if simple or (split is train and i == 0):
                kind = "classic"
            elif split is train and i == 1 and rich:
                kind = "expanded"
            else:
                options = _KINDS if rich else _KINDS[:3]
                kind = options[int(rng.integers(len(options)))]
            if simple or (split is train and i == 0):
                style = "single"
            else:
                style = _STYLES[int(rng.integers(len(_STYLES)))]
                if split is train and i == 1 and style == "single":
                    style = "bar"
            if style != "multi" and palette_history and int(rng.integers(4)) == 0:
                colors = palette_history[int(rng.integers(len(palette_history)))]
            else:
                colors = _palette(rng, style == "multi")
            for _ in range(30):
                pair = _pair(rng, q, kind, position, style, colors)
                key = _key(pair["input"])
                if key not in seen and not _source_copy(pair):
                    seen.add(key)
                    split.append(pair)
                    palette_history.append(colors[:3])
                    break
            else:
                raise AssertionError("could not construct a unique pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(indices):
    groups = []
    for index in indices:
        index = int(index)
        if not groups or index > groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _inspect(pair):
    grid, output = pair["input"], pair["output"]
    full_rows = np.flatnonzero(np.all(grid == grid[:, :1], axis=1))
    full_cols = np.flatnonzero(np.all(grid == grid[:1, :], axis=0))
    assert len(full_rows) and len(full_cols), "missing complete divider"
    divider = int(grid[full_rows[0], 0])
    assert all(int(grid[r, 0]) == divider for r in full_rows)
    assert all(int(grid[0, c]) == divider for c in full_cols)
    row_bands, col_bands = _runs(full_rows), _runs(full_cols)
    assert 1 <= len(row_bands) <= 2 and 1 <= len(col_bands) <= 2
    assert all(1 <= len(b) <= 3 for b in row_bands + col_bands)
    assert np.all(grid[full_rows, :] == divider)
    assert np.all(grid[:, full_cols] == divider)
    outside = np.ones(grid.shape, dtype=bool)
    outside[full_rows, :] = False
    outside[:, full_cols] = False
    assert not np.any(grid[outside] == divider)
    def gaps(bands, total):
        bounds = [0] + [b[-1] + 1 for b in bands]
        ends = [b[0] for b in bands] + [total]
        result = [slice(a, b) for a, b in zip(bounds, ends)]
        assert all(s.stop - s.start >= 2 for s in result)
        return result
    rows, cols = gaps(row_bands, grid.shape[0]), gaps(col_bands, grid.shape[1])
    cells = [(r, c, grid[rs, cs]) for r, rs in enumerate(rows) for c, cs in enumerate(cols)]
    marked = [(r, c, part) for r, c, part in cells if np.unique(part).size > 1]
    assert len(marked) == 1, "exactly one compartment must be nonuniform"
    uniform = [part for _, _, part in cells if np.unique(part).size == 1]
    assert len({int(part[0, 0]) for part in uniform}) == 1
    fill = int(uniform[0][0, 0])
    assert fill != divider
    r, c, target = marked[0]
    values = {int(v) for v in np.unique(target)}
    assert fill in values and divider not in values and len(values) >= 2
    assert np.array_equal(output, target), "output is not the full marked compartment"
    assert {int(v) for v in np.unique(grid)} == values | {divider}
    return {"rows": len(rows), "cols": len(cols), "position": (r, c),
            "size": output.shape, "bands": (len(row_bands[0]), len(col_bands[0])),
            "colors": len(values) + 1, "accent_count": int(np.count_nonzero(target != fill)),
            "edge": bool(np.any(target[0] != fill) or np.any(target[-1] != fill)
                         or np.any(target[:, 0] != fill) or np.any(target[:, -1] != fill)),
            "fill": fill, "divider": divider}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _key(pair["input"]), _key(pair["output"])
            assert key[0] not in seen_inputs and key not in seen_pairs
            seen_inputs.add(key[0]); seen_pairs.add(key)
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _source_copy(pair)
            records[split].append(_inspect(pair))
    assert len({r["position"] for r in records["train"]}) >= 2
    if any(r["rows"] > 2 or r["cols"] > 2 for r in records["test"]):
        assert any(r["rows"] > 2 or r["cols"] > 2 for r in records["train"])


def _predict_from_input(train_pairs, test_input):
    # This uses observable demonstrations and input cells only; no test output or metadata.
    for pair in train_pairs:
        assert np.array_equal(_visible_crop(pair["input"]), pair["output"])
    return _visible_crop(test_input)


def _visible_crop(grid):
    divider_rows = np.flatnonzero(np.all(grid == grid[:, :1], axis=1))
    divider_cols = np.flatnonzero(np.all(grid == grid[:1, :], axis=0))
    row_groups, col_groups = _runs(divider_rows), _runs(divider_cols)
    row_bounds = [0] + [g[-1] + 1 for g in row_groups]
    row_ends = [g[0] for g in row_groups] + [grid.shape[0]]
    col_bounds = [0] + [g[-1] + 1 for g in col_groups]
    col_ends = [g[0] for g in col_groups] + [grid.shape[1]]
    crops = [grid[a:b, c:d] for a, b in zip(row_bounds, row_ends)
             for c, d in zip(col_bounds, col_ends)]
    selected = [crop for crop in crops if np.unique(crop).size > 1]
    assert len(selected) == 1
    return selected[0].copy()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _motif_traits(pair):
    info = _inspect(pair)
    mask = pair["output"] != info["fill"]
    points = {tuple(map(int, point)) for point in np.argwhere(mask)}
    remaining = set(points)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    rows = {r for r, _ in points}
    cols = {c for _, c in points}
    bar = len(rows) == 1 or len(cols) == 1
    ordered = sorted(points)
    diagonal = (len(points) >= 3 and len(rows) == len(cols) == len(points)
                and all(ordered[i + 1][0] - ordered[i][0] == 1 and
                        abs(ordered[i + 1][1] - ordered[i][1]) == 1
                        for i in range(len(ordered) - 1)))
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    sub = mask[r0:r1 + 1, c0:c1 + 1]
    hollow = (sub.shape[0] >= 3 and sub.shape[1] >= 3
              and bool(np.all(sub[0]) and np.all(sub[-1])
                       and np.all(sub[:, 0]) and np.all(sub[:, -1]))
              and not bool(np.all(sub[1:-1, 1:-1])))
    return {"components": components, "diagonal": diagonal, "hollow": hollow,
            "long_bar": bar and len(points) >= 5,
            "large_irregular": len(points) >= 10 and components == 1 and not bar and not hollow}


def self_test() -> None:
    samples = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in samples:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_input(puzzle["train"], pair["input"]), pair["output"])
    assert len({_puzzle_key(p) for p in samples}) == 42
    assert len({len(p["train"]) for p in samples}) >= 4
    assert len({len(p["test"]) for p in samples}) >= 3
    records = {s: [_inspect(pair) for p in samples for pair in p[s]] for s in ("train", "test")}
    all_records = records["train"] + records["test"]
    assert {(r["rows"], r["cols"]) for r in all_records} >= {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert {r["position"] for r in records["test"]} >= {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert {r["position"] for r in records["test"]
            if r["rows"] == r["cols"] == 2} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert any(r["colors"] >= 4 for r in all_records)
    assert any(r["accent_count"] >= 15 for r in all_records)
    assert any(r["bands"] != (1, 1) for r in all_records)
    assert any(r["size"][0] != r["size"][1] for r in all_records)
    assert any(r["edge"] for r in all_records)
    assert any(not r["edge"] for r in all_records)
    assert sum(any(_inspect(x)["accent_count"] == 1 for x in p["train"])
               and any(_inspect(x)["accent_count"] > 1 for x in p["train"])
               for p in samples) >= 25
    assert sum(any(_inspect(x)["rows"] > 2 or _inspect(x)["cols"] > 2
                   for x in p["train"]) for p in samples) >= 10
    assert sum(any(_inspect(x)["rows"] > 2 or _inspect(x)["cols"] > 2
                   for x in p["test"]) for p in samples) >= 4
    assert any(not ({_inspect(x)["position"] for x in p["train"]} &
                    {_inspect(x)["position"] for x in p["test"]}) for p in samples)
    assert any({_inspect(x)["position"] for x in p["train"]} &
               {_inspect(x)["position"] for x in p["test"]} for p in samples)
    assert any(all(_inspect(x)["accent_count"] == 1 for x in p["train"] + p["test"])
               for p in samples)
    traits = [_motif_traits(pair) for p in samples for pair in p["train"] + p["test"]]
    assert sum(t["long_bar"] for t in traits) >= 8
    assert sum(t["hollow"] for t in traits) >= 8
    assert sum(t["diagonal"] for t in traits) >= 8
    assert sum(t["components"] >= 3 and not t["diagonal"] for t in traits) >= 8
    assert sum(t["large_irregular"] for t in traits) >= 8
    role_colors = [set(), set(), set()]
    palettes = set()
    for p in samples:
        for pair in p["train"] + p["test"]:
            info = _inspect(pair)
            accents = {int(v) for v in np.unique(pair["output"])} - {info["fill"]}
            role_colors[0].add(info["fill"])
            role_colors[1].add(info["divider"])
            role_colors[2].update(accents)
            palettes.add((info["fill"], info["divider"], tuple(sorted(accents))))
    assert all(colors == set(range(10)) for colors in role_colors)
    assert len(palettes) < len(all_records)  # Palette reuse is permitted.
    assert sum(
        len({(info["fill"], info["divider"],
              tuple(sorted({int(v) for v in np.unique(pair["output"])} - {info["fill"]})))
             for pair in p["train"] + p["test"] for info in [_inspect(pair)]})
        < len(p["train"]) + len(p["test"])
        for p in samples
    ) >= 15
    assert any(max(r["size"][0] for r in [_inspect(x) for x in p["test"]]) >
               max(r["size"][0] for r in [_inspect(x) for x in p["train"]]) for p in samples)
    assert any(max(r["size"][0] for r in [_inspect(x) for x in p["test"]]) <
               min(r["size"][0] for r in [_inspect(x) for x in p["train"]]) for p in samples)
    assert any(set(_inspect(x)["size"][0] for x in p["test"]) &
               set(_inspect(x)["size"][0] for x in p["train"]) for p in samples)
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(1000 + i, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(1000 + i, *counts))
    for i, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        validate_generated_puzzle(generate_puzzle(1100 + i, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1200, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
