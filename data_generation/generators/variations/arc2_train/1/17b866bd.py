import numpy as np


_OUTLINE_OFFSETS = (
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 4), (1, 5),
    (2, 0), (2, 5),
    (3, 0), (3, 5),
    (4, 0), (4, 1), (4, 4), (4, 5),
    (5, 1), (5, 2), (5, 3), (5, 4),
)

_CAVITY_OFFSETS = (
    (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 2), (4, 3),
)

_SOURCE_INPUT_SIGNATURES = frozenset(
    tuple(encoded.split("/"))
    for encoded in (
        (
            "088880888808888088880/"
            "880088800888008880088/"
            "800008000080000800008/"
            "800008000080000800008/"
            "880088800888008880088/"
            "088880888888888088880/"
            "880088800888008880088/"
            "800008000080000800008/"
            "800008000080000800008/"
            "880088800888008880088/"
            "488880888808888088880/"
            "880088800888008880088/"
            "800008000080000800008/"
            "800008000080000800008/"
            "880088800888008880088/"
            "088880888808888088880"
        ),
        (
            "7888808888088880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "0888808888188880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "0888848888088880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "0888808888088880"
        ),
        (
            "8888888888888880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "8888818888888880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "8888888888888880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "0888808888088880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "0888808888088880"
        ),
        (
            "1888818888188880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "1888808888488880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "4888848888488880/"
            "8800888008880088/"
            "8000080000800008/"
            "8000080000800008/"
            "8800888008880088/"
            "0888808888088880"
        ),
    )
)


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be nonnegative")
    return result


def _count(name: str, value: int | None, lo: int, hi: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not lo <= result <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return result


def _resolve_counts(rng: np.random.Generator, seed: int,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + (seed // 4) % 9)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_signature(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _is_source_input(grid: np.ndarray) -> bool:
    rows = tuple("".join(str(int(v)) for v in row) for row in grid)
    return rows in _SOURCE_INPUT_SIGNATURES


def _bevel_options(height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple((br, bc) for br in (1, 2) for bc in (1, 2)
                 if height - 2 * br - 2 >= 1 and width - 2 * bc - 2 >= 1)


def _template(height: int, width: int, bevel_rows: int,
              bevel_columns: int, background: int, outline: int) -> tuple[np.ndarray, tuple]:
    tile = np.full((height, width), outline, dtype=np.int8)
    tile[0, 0] = tile[0, -1] = tile[-1, 0] = tile[-1, -1] = background
    cavity = []
    for row in range(1, height - 1):
        bevel = bevel_columns if row <= bevel_rows or row >= height - 1 - bevel_rows else 0
        for column in range(1 + bevel, width - 1 - bevel):
            tile[row, column] = background
            cavity.append((row, column))
    return tile, tuple(cavity)


def _scaffold(rows: int, columns: int, height: int, width: int,
              bevel_rows: int, bevel_columns: int,
              background: int, outline: int) -> tuple[np.ndarray, tuple]:
    tile, cavity = _template(height, width, bevel_rows, bevel_columns,
                             background, outline)
    result = np.full((rows * (height - 1) + 1, columns * (width - 1) + 1),
                     background, dtype=np.int8)
    for mr in range(rows):
        for mc in range(columns):
            r, c = mr * (height - 1), mc * (width - 1)
            result[r:r + height, c:c + width] = tile
    return result, cavity


def _choose_geometry(rng: np.random.Generator, first: bool,
                     previous: tuple | None) -> tuple[int, int, int, int]:
    for _ in range(20):
        if first and rng.random() < 0.35:
            choice = (6, 6, 1, 1)
        else:
            h = int(rng.integers(5, 9))
            w = int(rng.integers(5, 9))
            br, bc = _bevel_options(h, w)[int(rng.integers(len(_bevel_options(h, w))))]
            choice = (h, w, br, bc)
        if choice != previous:
            return choice
    return (5, 6, 1, 1) if previous != (5, 6, 1, 1) else (6, 5, 1, 1)


def _choose_layout(rng: np.random.Generator, geometry: tuple,
                   first: bool, rotation: int, split: str) -> tuple[int, int, tuple[int, int, int, int]]:
    h, w, _, _ = geometry
    for _ in range(20):
        margins = tuple(int(v) for v in rng.integers(0, 3, size=4))
        top, bottom, left, right = margins
        if rotation % 2:
            row_limit = min(6, (29 - left - right) // (h - 1))
            col_limit = min(6, (29 - top - bottom) // (w - 1))
        else:
            row_limit = min(6, (29 - top - bottom) // (h - 1))
            col_limit = min(6, (29 - left - right) // (w - 1))
        minimum = 2 if first else 1
        if row_limit >= minimum and col_limit >= minimum:
            if split == "train" and (first or rng.random() < 0.7):
                row_limit = min(row_limit, 4)
                col_limit = min(col_limit, 4)
            if split == "test" and rng.random() < 0.42:
                rows = int(rng.integers(min(4, row_limit), row_limit + 1))
                cols = int(rng.integers(min(4, col_limit), col_limit + 1))
            else:
                rows = int(rng.integers(minimum, row_limit + 1))
                cols = int(rng.integers(minimum, col_limit + 1))
            return rows, cols, margins
    return 2, 2, (0, 0, 0, 0)


def _choose_positions(rng: np.random.Generator, rows: int, columns: int,
                      split: str, first: bool) -> tuple[list[tuple[int, int]], str]:
    cells = [(r, c) for r in range(rows) for c in range(columns)]
    total = len(cells)
    modes = ("scattered", "scattered", "row", "column", "block", "cluster",
             "stripe", "single", "full", "empty")
    if first:
        interior = [(r, c) for r, c in cells if r > 0 and c > 0]
        anchor = interior[int(rng.integers(len(interior)))]
        remaining = [cell for cell in cells if cell != anchor]
        rng.shuffle(remaining)
        count = min(3, total - 1)
        return [anchor] + remaining[:count - 1], "demonstration"
    mode = modes[int(rng.integers(len(modes)))]
    if split == "test" and rng.random() < 0.28:
        mode = "dense"
    if mode == "empty":
        return [], mode
    if mode == "full":
        return cells, mode
    if mode == "single":
        return [cells[int(rng.integers(total))]], mode
    if mode == "row":
        row = int(rng.integers(rows))
        selected = [(r, c) for r, c in cells if r == row]
        if rows > 1 and rng.random() < 0.5:
            selected += [(r, c) for r, c in cells if r == (row + 1) % rows]
        return selected, mode
    if mode == "column":
        col = int(rng.integers(columns))
        selected = [(r, c) for r, c in cells if c == col]
        if columns > 1 and rng.random() < 0.5:
            selected += [(r, c) for r, c in cells if c == (col + 1) % columns]
        return selected, mode
    if mode == "stripe":
        if rng.random() < 0.5:
            phase = int(rng.integers(2))
            return [(r, c) for r, c in cells if (r + phase) % 2 == 0], mode
        phase = int(rng.integers(2))
        return [(r, c) for r, c in cells if (c + phase) % 2 == 0], mode
    if mode == "block":
        r0, r1 = sorted(int(v) for v in rng.integers(0, rows, size=2))
        c0, c1 = sorted(int(v) for v in rng.integers(0, columns, size=2))
        return [(r, c) for r, c in cells if r0 <= r <= r1 and c0 <= c <= c1], mode
    if mode == "cluster":
        center = cells[int(rng.integers(total))]
        ranked = sorted(cells, key=lambda x: (abs(x[0] - center[0]) + abs(x[1] - center[1]), rng.random()))
        count = int(rng.integers(1, total + 1))
        return ranked[:count], mode
    if mode == "dense":
        count = int(rng.integers(max(1, (2 * total) // 3), total + 1))
    else:
        count = int(rng.integers(1, total + 1))
    selected = rng.choice(total, size=count, replace=False)
    return [cells[int(i)] for i in selected], mode


def _choose_colors(rng: np.random.Generator, positions: list,
                   background: int, outline: int, first: bool) -> list[int]:
    count = len(positions)
    if count == 0:
        return []
    accents = [digit for digit in range(10) if digit not in (background, outline)]
    if first:
        chosen = [outline] + [int(v) for v in rng.choice(accents, size=count - 1, replace=False)]
        rng.shuffle(chosen)
        return chosen
    mode = int(rng.integers(7))
    if mode == 0:
        palette = [outline]
    elif mode == 1:
        palette = [int(rng.choice(accents))]
    elif mode == 2:
        palette = [int(v) for v in rng.choice(accents, size=int(rng.integers(1, min(6, count, len(accents)) + 1)), replace=False)]
    elif mode == 3:
        palette = [outline] + [int(v) for v in rng.choice(accents, size=int(rng.integers(1, min(5, count, len(accents)) + 1)), replace=False)]
    elif mode == 4:
        palette = [int(v) for v in rng.choice([digit for digit in range(10) if digit != background],
                                               size=int(rng.integers(1, min(8, count, 9) + 1)), replace=False)]
    elif mode == 5:
        chosen = [outline] * count
        chosen[int(rng.integers(count))] = int(rng.choice(accents))
        return chosen
    else:
        chosen = [outline] * count
        for index in rng.choice(count, size=int(rng.integers(1, min(3, count) + 1)), replace=False):
            chosen[int(index)] = int(rng.choice(accents))
        return chosen
    if count >= len(palette):
        chosen = palette + [int(rng.choice(palette)) for _ in range(count - len(palette))]
    else:
        chosen = [int(rng.choice(palette)) for _ in range(count)]
    rng.shuffle(chosen)
    return chosen


def _make_pair(rng: np.random.Generator, rotation: int, split: str,
               first: bool, previous_geometry: tuple | None) -> tuple[dict, tuple]:
    geometry = _choose_geometry(rng, first, previous_geometry)
    h, w, br, bc = geometry
    rows, columns, margins = _choose_layout(rng, geometry, first, rotation, split)
    top, bottom, left, right = margins
    background, outline = [int(v) for v in rng.choice(10, size=2, replace=False)]
    scaffold, cavity = _scaffold(rows, columns, h, w, br, bc, background, outline)
    inp, out = scaffold.copy(), scaffold.copy()
    positions, _ = _choose_positions(rng, rows, columns, split, first)
    colors = _choose_colors(rng, positions, background, outline, first)
    for (mr, mc), color in zip(positions, colors):
        r, c = mr * (h - 1), mc * (w - 1)
        inp[r, c] = color
        for dr, dc in cavity:
            out[r + dr, c + dc] = color
    # The same transformation of all pairs fixes the puzzle's corner convention.
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    inp = np.pad(inp, ((top, bottom), (left, right)), constant_values=background)
    out = np.pad(out, ((top, bottom), (left, right)), constant_values=background)
    return {"input": inp.astype(np.int8), "output": out.astype(np.int8)}, geometry


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    rotation = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_geometry = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            first = split == "train" and index == 0
            for attempt in range(100):
                pair, geometry = _make_pair(rng, rotation, split, first,
                                            previous_geometry if split == "train" and index == 1 else None)
                signature = _grid_signature(pair["input"])
                if signature not in seen and not _is_source_input(pair["input"]):
                    break
            else:
                raise RuntimeError("unable to construct a unique pair")
            seen.add(signature)
            puzzle[split].append(pair)
            if first:
                previous_geometry = geometry
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _parse_pair(inp: np.ndarray, out: np.ndarray) -> list[dict]:
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(values) >= 2
    roles = [int(values[i]) for i in order[:2]]
    found = []
    H, W = inp.shape
    for background, outline in ((roles[0], roles[1]), (roles[1], roles[0])):
        for top in range(3):
            for bottom in range(3):
                core_h = H - top - bottom
                if core_h < 5:
                    continue
                for left in range(3):
                    for right in range(3):
                        core_w = W - left - right
                        if core_w < 5:
                            continue
                        core_in = inp[top:H-bottom if bottom else H, left:W-right if right else W]
                        core_out = out[top:H-bottom if bottom else H, left:W-right if right else W]
                        if (top and (np.any(inp[:top] != background) or np.any(out[:top] != background))) or \
                           (bottom and (np.any(inp[H-bottom:] != background) or np.any(out[H-bottom:] != background))) or \
                           (left and (np.any(inp[:, :left] != background) or np.any(out[:, :left] != background))) or \
                           (right and (np.any(inp[:, W-right:] != background) or np.any(out[:, W-right:] != background))):
                            continue
                        for rotation in range(4):
                            canonical_in = np.rot90(core_in, -rotation)
                            canonical_out = np.rot90(core_out, -rotation)
                            ch, cw = canonical_in.shape
                            for h in range(5, 9):
                                if (ch - 1) % (h - 1):
                                    continue
                                rows = (ch - 1) // (h - 1)
                                if not 1 <= rows <= 6:
                                    continue
                                for w in range(5, 9):
                                    if (cw - 1) % (w - 1):
                                        continue
                                    columns = (cw - 1) // (w - 1)
                                    if not 1 <= columns <= 6:
                                        continue
                                    clean = canonical_in.copy()
                                    positions, colors = [], []
                                    for mr in range(rows):
                                        for mc in range(columns):
                                            r, c = mr * (h - 1), mc * (w - 1)
                                            color = int(clean[r, c])
                                            if color != background:
                                                positions.append((mr, mc))
                                                colors.append(color)
                                                clean[r, c] = background
                                    for br, bc in _bevel_options(h, w):
                                        scaffold, cavity = _scaffold(rows, columns, h, w, br, bc,
                                                                     background, outline)
                                        if not np.array_equal(clean, scaffold):
                                            continue
                                        expected = scaffold.copy()
                                        for (mr, mc), color in zip(positions, colors):
                                            r, c = mr * (h - 1), mc * (w - 1)
                                            for dr, dc in cavity:
                                                expected[r + dr, c + dc] = color
                                        if np.array_equal(canonical_out, expected):
                                            found.append({"rotation": rotation, "rows": rows,
                                                          "columns": columns, "geometry": (h, w, br, bc),
                                                          "markers": positions, "colors": colors,
                                                          "background": background, "outline": outline,
                                                          "margins": (top, bottom, left, right)})
    return found


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape
            signature = _grid_signature(inp)
            assert signature not in seen and not _is_source_input(inp)
            seen.add(signature)
            descriptions = _parse_pair(inp, out)
            assert descriptions
            analyses[split].append(descriptions)
    train_directions = set(range(4))
    for descriptions in analyses["train"]:
        train_directions &= {item["rotation"] for item in descriptions}
    assert len(train_directions) == 1
    rotation = next(iter(train_directions))
    assert all(any(item["rotation"] == rotation for item in descriptions)
               for descriptions in analyses["test"])
    train = [next(item for item in d if item["rotation"] == rotation) for d in analyses["train"]]
    tests = [next(item for item in d if item["rotation"] == rotation) for d in analyses["test"]]
    # One training pair establishes direction at a junction, marker erasure,
    # an unmarked cavity, outline-colored paint, and two ordinary paints.
    assert any(info["rows"] >= 2 and info["columns"] >= 2 and
               any(r > 0 and c > 0 for r, c in info["markers"]) and
               0 < len(info["markers"]) < info["rows"] * info["columns"] and
               info["outline"] in info["colors"] and
               len(set(info["colors"]) - {info["outline"]}) >= 2
               for info in train)
    assert len({info["geometry"] for info in train}) >= 2
    # Even a single-marker or unchanged test follows the demonstrated rule.
    assert all(info["background"] not in info["colors"] for info in train + tests)


def _puzzles_equal(left: dict, right: dict) -> bool:
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(left[s], right[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert _puzzles_equal(generate_puzzle(23), generate_puzzle(23))
    assert not _puzzles_equal(generate_puzzle(0), generate_puzzle(1))
    for seed, ntrain, ntest in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                                (1101, None, 1), (1102, None, 4),
                                (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, shapes, geometries, markers = set(), set(), set(), set(), set()
    rotations, palettes, layouts = set(), set(), set()
    cavity_sizes, axes = set(), set()
    within = {"roles": 0, "geometry": 0, "marker_count": 0,
              "size": 0, "margins": 0}
    regimes = {"height_only": 0, "width_only": 0,
               "both_larger": 0, "not_larger": 0}
    unseen_test_colors = 0
    combined = 0
    split_modes = {split: {key: False for key in
                   ("empty", "single", "four", "full", "sparse", "dense",
                    "outline_zero", "outline_all", "one_accent", "repeated_accent",
                    "single_paint", "many_paints", "one_row", "one_column")}
                   for split in ("train", "test")}
    modes = {"empty": False, "single": False, "full": False,
             "no_outline": False, "all_outline": False, "adjacent": False,
             "multirow": False, "multicol": False, "margin": False}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        descriptions = [_parse_pair(pair["input"], pair["output"])[0]
                        for split in ("train", "test") for pair in puzzle[split]]
        ntrain = len(puzzle["train"])
        tests = descriptions[ntrain:]
        train_paint = {color for info in descriptions[:ntrain] for color in info["colors"]}
        unseen_test_colors += any(color not in train_paint for info in tests for color in info["colors"])
        within["roles"] += len({(x["background"], x["outline"]) for x in descriptions}) > 1
        within["geometry"] += len({x["geometry"] for x in descriptions}) > 1
        within["marker_count"] += len({len(x["markers"]) for x in descriptions}) > 1
        within["size"] += len({(x["rows"], x["columns"]) for x in descriptions}) > 1
        within["margins"] += len({x["margins"] for x in descriptions}) > 1
        max_train_height = max(x["input"].shape[0] for x in puzzle["train"])
        max_train_width = max(x["input"].shape[1] for x in puzzle["train"])
        for pair in puzzle["test"]:
            h, w = pair["input"].shape
            key = ("both_larger" if h > max_train_height and w > max_train_width else
                   "height_only" if h > max_train_height else
                   "width_only" if w > max_train_width else "not_larger")
            regimes[key] += 1
        for pair_index, (info, pair) in enumerate(zip(descriptions, puzzle["train"] + puzzle["test"])):
            shapes.add(pair["input"].shape)
            geometries.add(info["geometry"])
            rotations.add(info["rotation"])
            palettes.add((info["background"], info["outline"]))
            layouts.add((info["rows"], info["columns"], tuple(info["markers"])))
            count = len(info["markers"])
            total = info["rows"] * info["columns"]
            markers.add(count)
            axes.add((info["rows"], info["columns"]))
            cavity_sizes.add(len(_template(*info["geometry"], info["background"],
                                       info["outline"])[1]))
            split = "train" if pair_index < ntrain else "test"
            sm = split_modes[split]
            accent = [color for color in info["colors"] if color != info["outline"]]
            sm["empty"] |= count == 0
            sm["single"] |= count == 1
            sm["four"] |= count == 4
            sm["full"] |= count == total
            sm["sparse"] |= 0 < count < total / 3
            sm["dense"] |= count >= 2 * total / 3
            sm["outline_zero"] |= count > 0 and info["outline"] not in info["colors"]
            sm["outline_all"] |= count > 0 and set(info["colors"]) == {info["outline"]}
            sm["one_accent"] |= len(accent) == 1
            sm["repeated_accent"] |= len(accent) > len(set(accent))
            sm["single_paint"] |= count > 0 and len(set(info["colors"])) == 1
            sm["many_paints"] |= len(set(info["colors"])) >= 3
            sm["one_row"] |= info["rows"] == 1
            sm["one_column"] |= info["columns"] == 1
            combined += (info["geometry"] != (6, 6, 1, 1) and
                         any(info["margins"]) and count >= 2 * total / 3)
            modes["empty"] |= count == 0
            modes["single"] |= count == 1
            modes["full"] |= count == total
            modes["no_outline"] |= count > 0 and info["outline"] not in info["colors"]
            modes["all_outline"] |= count > 0 and set(info["colors"]) == {info["outline"]}
            modes["adjacent"] |= any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                                     for a in info["markers"] for b in info["markers"] if a != b)
            modes["multirow"] |= any(sum(r == row for r, _ in info["markers"]) >= 2
                                     for row in range(info["rows"]))
            modes["multicol"] |= any(sum(c == col for _, c in info["markers"]) >= 2
                                     for col in range(info["columns"]))
            modes["margin"] |= any(info["margins"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(geometries) >= 12
    assert len(markers) >= 10 and len(layouts) >= 100
    assert len(palettes) >= 20 and len(rotations) == 4
    assert all(modes.values())
    assert len(cavity_sizes) >= 10 and len(axes) >= 12
    assert all(value >= 30 for value in within.values())
    assert all(value > 0 for value in regimes.values())
    assert unseen_test_colors >= 10 and combined >= 20
    assert all(all(flags.values()) for flags in split_modes.values())
