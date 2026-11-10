"""Compositional examples of orthogonal marker span completion."""

import numpy as np


_SOURCE_INPUTS = (
    ("000000100001", "000000000000", "000000000010", "000000000000",
     "000001000000", "010000000100", "000000100010", "001000000000",
     "000000000000", "000010001000", "000000000000", "000010001000"),
    ("00001000010", "00000000000", "00001000100", "00100000000",
     "00000000000", "00000000000", "00000000000", "10000001000",
     "00000000000", "00000010000"),
    ("010010000", "000000000", "000010100", "100000000",
     "000001000", "001000010", "000000000", "100000000"),
    ("000100000", "000000001", "000000000", "000000000",
     "010000010", "000000000", "010000000", "000000100",
     "000100000"),
    ("000000010010001000000", "001000000000000000000",
     "000000000010000000000", "000000000000000000010",
     "000000000100000000000", "010000000000000000000",
     "000000100000000000000", "010000000000000000000",
     "000000000010000000000", "000000000001000000000",
     "000000000000010000000", "000000000000000000000",
     "000000000100000000000", "000000000000000000000",
     "000000100000001000000", "000000000000000000000",
     "010000000000000000000", "000000000000000000000",
     "010100000000000000000"),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


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
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _span_masks(markers):
    height, width = markers.shape
    horizontal = np.zeros_like(markers)
    vertical = np.zeros_like(markers)
    for row in range(height):
        cols = np.flatnonzero(markers[row])
        if len(cols) >= 2:
            horizontal[row, cols[0]:cols[-1] + 1] = True
    for col in range(width):
        rows = np.flatnonzero(markers[:, col])
        if len(rows) >= 2:
            vertical[rows[0]:rows[-1] + 1, col] = True
    return horizontal, vertical


def _apply_connections(input_grid, background, marker, connector):
    markers = input_grid == marker
    horizontal, vertical = _span_masks(markers)
    output = input_grid.copy()
    output[(horizontal | vertical) & ~markers] = connector
    return output


def _features(markers):
    rows = markers.sum(axis=1)
    cols = markers.sum(axis=0)
    horizontal, vertical = _span_masks(markers)
    row_gaps = []
    col_gaps = []
    unequal_multi = False
    short_multi = False
    interior_filled = False
    for row in np.flatnonzero(rows >= 2):
        positions = np.flatnonzero(markers[row])
        gaps = np.diff(positions).tolist()
        row_gaps.extend(gaps)
        if len(positions) >= 3:
            unequal_multi |= len(set(gaps)) > 1
            short_multi |= any(g <= 2 for g in gaps)
            interior_filled |= positions[-1] - positions[0] + 1 > len(positions)
    for col in np.flatnonzero(cols >= 2):
        positions = np.flatnonzero(markers[:, col])
        gaps = np.diff(positions).tolist()
        col_gaps.extend(gaps)
        if len(positions) >= 3:
            unequal_multi |= len(set(gaps)) > 1
            short_multi |= any(g <= 2 for g in gaps)
            interior_filled |= positions[-1] - positions[0] + 1 > len(positions)
    rectangles = 0
    multi_rows = np.flatnonzero(rows >= 2)
    for index, row in enumerate(multi_rows):
        for other in multi_rows[index + 1:]:
            shared_cols = np.count_nonzero(markers[row] & markers[other])
            rectangles += shared_cols * (shared_cols - 1) // 2
    return {
        "markers": int(markers.sum()),
        "rows": int(np.count_nonzero(rows >= 2)),
        "cols": int(np.count_nonzero(cols >= 2)),
        "max_row": int(rows.max()),
        "max_col": int(cols.max()),
        "connector_cells": int(np.count_nonzero((horizontal | vertical) & ~markers)),
        "empty_crossing": bool(np.any(horizontal & vertical & ~markers)),
        "shared_marker": bool(np.any(horizontal & vertical & markers)),
        "rectangles": rectangles,
        "isolated": int(sum(rows[r] == 1 and cols[c] == 1 for r, c in np.argwhere(markers))),
        "short_gap": any(gap <= 2 for gap in row_gaps + col_gaps),
        "unequal_gaps": len(set(row_gaps + col_gaps)) >= 2,
        "unequal_multi": unequal_multi,
        "short_multi": short_multi,
        "interior_filled": interior_filled,
    }


def _shape(rng, regime, teaching=False):
    # All bands occur in both splits. Tests are only statistically tilted larger.
    bands = ((8, 14), (15, 22), (23, 30))
    probabilities = (0.36, 0.34, 0.30) if regime == "source_train" else (0.28, 0.34, 0.38)
    band_h = int(rng.choice(3, p=probabilities))
    band_w = int(rng.choice(3, p=probabilities))
    h = int(rng.integers(bands[band_h][0], bands[band_h][1] + 1))
    w = int(rng.integers(bands[band_w][0], bands[band_w][1] + 1))
    if teaching:
        h, w = max(8, h), max(8, w)
    return h, w


def _coordinates(rng, size, count):
    return sorted(int(v) for v in rng.choice(size, size=count, replace=False))


def _add_random(rng, cells, height, width, count, excluded=()):
    vacant = [(r, c) for r in range(height) for c in range(width)
              if (r, c) not in cells and (r, c) not in excluded]
    for index in rng.choice(len(vacant), size=min(count, len(vacant)), replace=False):
        cells.add(vacant[int(index)])


def _layout(rng, height, width, mode):
    cells = set()
    if mode == "teaching":
        r0, r1, r2 = _coordinates(rng, height, 3)
        c0, c1, c2, c3 = _coordinates(rng, width, 4)
        # A three-marker row has an occupied junction and an empty crossing.
        cells.update(((r1, c0), (r1, c1), (r1, c3),
                      (r0, c1), (r2, c1), (r0, c2), (r2, c2)))
        # A singleton shows that unaligned markers are retained.
        free_rows = [r for r in range(height) if r not in (r0, r1, r2)]
        free_cols = [c for c in range(width) if c not in (c0, c1, c2, c3)]
        cells.add((int(rng.choice(free_rows)), int(rng.choice(free_cols))))
        if rng.random() < 0.35:
            cells.update(((r0, c0), (r2, c0)))
        if rng.random() < 0.45:
            _add_random(rng, cells, height, width, int(rng.integers(1, 6)),
                        excluded=((r1, c2),))
    elif mode == "rectangle":
        r0, r1 = _coordinates(rng, height, 2)
        c0, c1 = _coordinates(rng, width, 2)
        cells.update(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))
        if c1 - c0 >= 2 and rng.random() < 0.6:
            cells.add((r0, int(rng.integers(c0 + 1, c1))))
        _add_random(rng, cells, height, width, int(rng.integers(0, 11)))
    elif mode == "crossing":
        r0, rm, r1 = _coordinates(rng, height, 3)
        c0, cm, c1 = _coordinates(rng, width, 3)
        cells.update(((rm, c0), (rm, c1), (r0, cm), (r1, cm)))
        _add_random(rng, cells, height, width, int(rng.integers(0, 9)))
    elif mode == "junction":
        r0, rm, r1 = _coordinates(rng, height, 3)
        c0, cm, c1 = _coordinates(rng, width, 3)
        cells.update(((rm, c0), (rm, cm), (rm, c1), (r0, cm), (r1, cm)))
        _add_random(rng, cells, height, width, int(rng.integers(0, 9)))
    elif mode == "network":
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        for c in _coordinates(rng, width, int(rng.integers(2, min(6, width) + 1))):
            cells.add((row, c))
        for r in _coordinates(rng, height, int(rng.integers(2, min(6, height) + 1))):
            cells.add((r, col))
        _add_random(rng, cells, height, width,
                    int(rng.integers(3, min(36, height * width // 7) + 1)))
    elif mode in ("horizontal", "vertical"):
        # Extra singleton rows and columns are unique, so the other axis stays inactive.
        if mode == "horizontal":
            row = int(rng.integers(height))
            line_cols = _coordinates(rng, width, int(rng.integers(2, min(5, width) + 1)))
            cells.update((row, c) for c in line_cols)
            other_rows = [r for r in range(height) if r != row]
            other_cols = [c for c in range(width) if c not in line_cols]
            k = int(rng.integers(0, min(5, len(other_rows), len(other_cols)) + 1))
            cells.update(zip(rng.choice(other_rows, k, replace=False).tolist(),
                             rng.choice(other_cols, k, replace=False).tolist()))
        else:
            col = int(rng.integers(width))
            line_rows = _coordinates(rng, height, int(rng.integers(2, min(5, height) + 1)))
            cells.update((r, col) for r in line_rows)
            other_rows = [r for r in range(height) if r not in line_rows]
            other_cols = [c for c in range(width) if c != col]
            k = int(rng.integers(0, min(5, len(other_rows), len(other_cols)) + 1))
            cells.update(zip(rng.choice(other_rows, k, replace=False).tolist(),
                             rng.choice(other_cols, k, replace=False).tolist()))
    elif mode == "isolated":
        count = int(rng.integers(2, min(height, width, 7) + 1))
        cells.update(zip(rng.choice(height, count, replace=False).tolist(),
                         rng.choice(width, count, replace=False).tolist()))
    elif mode == "adjacent":
        if rng.random() < 0.5:
            row = int(rng.integers(height))
            col = int(rng.integers(width - 1))
            cells.update(((row, col), (row, col + 1)))
        else:
            row = int(rng.integers(height - 1))
            col = int(rng.integers(width))
            cells.update(((row, col), (row + 1, col)))
    elif mode == "short":
        if rng.random() < 0.5:
            row = int(rng.integers(height))
            col = int(rng.integers(width - 2))
            cells.update(((row, col), (row, col + 2)))
        else:
            row = int(rng.integers(height - 2))
            col = int(rng.integers(width))
            cells.update(((row, col), (row + 2, col)))
    elif mode == "scatter":
        count = int(rng.integers(4, min(38, height * width // 5) + 1))
        _add_random(rng, cells, height, width, count)
    elif mode == "separate":
        r0, r1 = _coordinates(rng, height, 2)
        four_cols = _coordinates(rng, width, 4)
        cells.update(((r0, four_cols[0]), (r0, four_cols[1]),
                      (r1, four_cols[2]), (r1, four_cols[3])))
    else:
        raise ValueError("unknown layout mode")
    return cells


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_input(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def _palette(rng, convention, pair_index):
    background, marker, connector, alternate = convention
    if alternate is None or pair_index == 0:
        return background, marker, connector
    if pair_index == 1 or rng.random() < 0.5:
        return alternate[0], alternate[1], connector
    return background, marker, connector


_MODES = ("rectangle", "crossing", "junction", "network", "horizontal",
          "vertical", "isolated", "adjacent", "short", "scatter", "separate")
_TRAIN_WEIGHTS = (0.12, 0.10, 0.12, 0.13, 0.10, 0.10, 0.05, 0.04, 0.04, 0.14, 0.06)
_TEST_WEIGHTS = (0.13, 0.11, 0.11, 0.17, 0.09, 0.09, 0.07, 0.05, 0.05, 0.09, 0.04)


def _make_pair(rng, regime, convention, pair_index, teaching):
    height, width = _shape(rng, regime, teaching)
    if teaching:
        mode = "teaching"
    elif pair_index == 1 and convention[3] is not None:
        mode = str(rng.choice(("crossing", "junction")))
    else:
        weights = _TRAIN_WEIGHTS if regime == "source_train" else _TEST_WEIGHTS
        mode = str(rng.choice(_MODES, p=weights))
    cells = _layout(rng, height, width, mode)
    background, marker, connector = _palette(rng, convention, pair_index)
    inp = np.full((height, width), background, dtype=np.int8)
    for row, col in cells:
        inp[row, col] = marker
    out = _apply_connections(inp, background, marker, connector)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    colors = rng.choice(10, size=5, replace=False)
    alternate = (int(colors[3]), int(colors[4])) if rng.random() < 0.45 else None
    convention = (int(colors[0]), int(colors[1]), int(colors[2]), alternate)
    puzzle = {"train": [], "test": []}
    seen = set()
    child_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                               size=num_train + num_test, dtype=np.uint64)
    for index, child_seed in enumerate(child_seeds):
        regime = "source_train" if index < num_train else "target_test"
        child_rng = np.random.default_rng(child_seed)
        for _ in range(30):
            pair = _make_pair(child_rng, regime, convention, index, index == 0)
            key = _key(pair["input"])
            positive_needed = index == 1 and alternate is not None
            if (key not in seen and not _source_input(pair["input"])
                    and (not positive_needed or not np.array_equal(pair["input"], pair["output"]))):
                break
        else:
            raise RuntimeError("unable to make a unique pair")
        seen.add(key)
        puzzle["train" if index < num_train else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(inp):
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 2, "input must have background and markers")
    _require(counts[0] != counts[1], "background must be a clear majority")
    background = int(colors[np.argmax(counts)])
    marker = int(colors[np.argmin(counts)])
    return background, marker


def _check_grid(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
    h, w = grid.shape
    _require(1 <= h <= 30 and 1 <= w <= 30, "illegal grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal ARC color")


def _validate_impl(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "puzzle splits")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen = set()
    connector = None
    palettes = []
    train_features = []
    positive_train_palettes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            _require(inp.shape == out.shape, "dimensions changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            _require(not _source_input(inp), "source input copied")
            seen.add(key)
            background, marker = _roles(inp)
            palettes.append((background, marker))
            _require(connector not in (background, marker), "connector in input palette")
            changed = inp != out
            if np.any(changed):
                new_colors = set(np.unique(out[changed]).tolist())
                _require(len(new_colors) == 1, "multiple connector colors")
                new_color = int(next(iter(new_colors)))
                _require(new_color not in (background, marker), "reused input color")
                if connector is None:
                    connector = new_color
                _require(new_color == connector, "connector convention changed")
                _require(bool(np.all(inp[changed] == background)), "marker was changed")
                if split == "train":
                    positive_train_palettes.add((background, marker))
            _require(connector is None or connector not in (background, marker), "ambiguous palette")
            markers = inp == marker
            features = _features(markers)
            _require(features["markers"] < inp.size // 2, "markers not sparse")
            if split == "train":
                train_features.append(features)
            if connector is not None:
                expected = _apply_connections(inp, background, marker, connector)
                _require(np.array_equal(out, expected), "incorrect spans")
            else:
                _require(np.array_equal(inp, out), "unexplained output change")
    _require(connector is not None, "training omitted connector convention")
    _require(any(f["connector_cells"] > 0 for f in train_features), "training lacks positive example")
    _require(any(f["rows"] and f["cols"] and f["max_row"] >= 3
                 and f["empty_crossing"] and f["shared_marker"]
                 for f in train_features), "training omits rule evidence")
    # Changed training palettes, if any, show that the connector remains fixed.
    _require(set(palettes).issubset(positive_train_palettes),
             "a palette was not demonstrated with connector in training")
    _require(len(set(palettes)) <= 2, "palette variation is too broad")


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid puzzle: {exc}") from exc


def _predict(train, inp):
    # This uses only copied demonstrations and the query input.
    learned = set()
    for example in train:
        before, after = example["input"], example["output"]
        learned.update(int(v) for v in np.unique(after[before != after]))
    _require(len(learned) == 1, "cannot infer connector")
    connector = next(iter(learned))
    background, marker = _roles(inp)
    result = inp.copy()
    for r in range(inp.shape[0]):
        positions = np.flatnonzero(inp[r] == marker)
        if len(positions) > 1:
            for c in range(int(positions[0]), int(positions[-1]) + 1):
                if inp[r, c] == background:
                    result[r, c] = connector
    for c in range(inp.shape[1]):
        positions = np.flatnonzero(inp[:, c] == marker)
        if len(positions) > 1:
            for r in range(int(positions[0]), int(positions[-1]) + 1):
                if inp[r, c] == background:
                    result[r, c] = connector
    return result


def self_test() -> None:
    for seed, nt, nq in ((3, None, None), (37, None, None), (91, 6, 3),
                         (200, 2, 1), (201, 10, 1), (202, 10, 4),
                         (301, None, 1), (302, None, 4),
                         (303, 2, None), (304, 10, None)):
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "train count not honored")
        if nq is not None:
            _require(len(puzzle["test"]) == nq, "test count not honored")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(400, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    first = generate_puzzle(123456, 7, 3)
    second = generate_puzzle(123456, 7, 3)
    _require(all(np.array_equal(a[role], b[role]) for split in ("train", "test")
                 for a, b in zip(first[split], second[split]) for role in ("input", "output")),
             "generation is not deterministic")
    other = generate_puzzle(123457, 7, 3)
    _require(not np.array_equal(first["train"][0]["input"], other["train"][0]["input"]),
             "nearby seeds identical")
    counts_train, counts_test, shapes, layouts, structural = set(), set(), set(), set(), set()
    roles, axes, multi_train, multi_test, rectangles = set(), set(), 0, 0, 0
    crossings, junctions, quiet, short, variable_palettes, comparable, big_train, small_test = 0, 0, 0, 0, 0, 0, 0, 0
    observed = {}
    constant_palettes = 0
    source_shift = 0
    reversed_shift = 0
    combined_variation = 0
    varied_shapes = 0
    for index in range(42):
        puzzle = generate_puzzle(10000 + 7919 * index)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        connector = int(np.unique(puzzle["train"][0]["output"][
            puzzle["train"][0]["input"] != puzzle["train"][0]["output"]])[0])
        bg, marker = _roles(puzzle["train"][0]["input"])
        roles.add((bg, marker, connector))
        palettes = set()
        train_areas = []
        test_areas = []
        puzzle_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, marker = _roles(inp)
                palettes.add((bg, marker))
                mask = inp == marker
                f = _features(mask)
                puzzle_shapes.add(inp.shape)
                _require(np.array_equal(_predict(puzzle["train"], inp), pair["output"]),
                         "training-only prediction failed")
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(map(tuple, np.argwhere(mask)))))
                structural.add((split, inp.shape, f["markers"], f["rows"], f["cols"],
                                f["max_row"], f["max_col"], f["connector_cells"]))
                axes.add((f["rows"] > 0, f["cols"] > 0))
                rectangles += f["rectangles"] > 0
                crossings += f["empty_crossing"]
                junctions += f["shared_marker"]
                quiet += f["connector_cells"] == 0
                short += f["short_gap"]
                checks = {
                    "multi": max(f["max_row"], f["max_col"]) >= 3,
                    "two_marker_lines": max(f["max_row"], f["max_col"]) <= 2
                    and (f["rows"] > 0 or f["cols"] > 0),
                    "rectangle": f["rectangles"] > 0,
                    "crossing": f["empty_crossing"],
                    "junction": f["shared_marker"],
                    "isolated_only": f["rows"] == 0 and f["cols"] == 0,
                    "adjacent_unchanged": f["markers"] == 2
                    and f["rows"] + f["cols"] == 1 and f["connector_cells"] == 0,
                    "short_positive": 0 < f["connector_cells"] <= 2,
                    "short_multi": f["short_multi"],
                    "unequal_multi": f["unequal_multi"],
                    "interior_filled": f["interior_filled"],
                    "separate_parallel": (f["rows"] >= 2 and f["cols"] == 0)
                    or (f["cols"] >= 2 and f["rows"] == 0),
                    "large_dense": inp.size >= 300 and f["markers"] >= 17,
                    "small_sparse": inp.size <= 196 and f["markers"] <= 8,
                }
                for name, present in checks.items():
                    if present:
                        key = (split, name)
                        observed[key] = observed.get(key, 0) + 1
                if split == "train":
                    train_areas.append(inp.size)
                    multi_train += f["max_row"] >= 3 or f["max_col"] >= 3
                    big_train += inp.size >= 300
                else:
                    test_areas.append(inp.size)
                    multi_test += f["max_row"] >= 3 or f["max_col"] >= 3
                    small_test += inp.size <= 196
        variable_palettes += len(palettes) > 1
        constant_palettes += len(palettes) == 1
        comparable += any(abs(a - b) <= 80 for a in train_areas for b in test_areas)
        source_shift += min(train_areas) <= 196 and max(test_areas) >= 300
        reversed_shift += max(train_areas) >= 300 and min(test_areas) <= 196
        combined_variation += (len(palettes) > 1 and max(train_areas) >= 300
                               and min(test_areas) <= 196)
        varied_shapes += len(puzzle_shapes) > 1
    _require(len(counts_train) >= 4 and len(counts_test) >= 3, "count diversity")
    _require(len(roles) >= 25 and len(shapes) >= 45, "palette or shape diversity")
    _require(len(layouts) >= 200 and len(structural) >= 150, "layout diversity")
    _require(axes == {(False, False), (True, False), (False, True), (True, True)},
             "axis cases absent")
    _require(min(multi_train, multi_test, rectangles, crossings, junctions,
                 quiet, short, variable_palettes, comparable, big_train, small_test) > 0,
             "review variation absent")
    _require(min(constant_palettes, source_shift, reversed_shift,
                 combined_variation, varied_shapes) > 0,
             "puzzle-level variation absent")
    for split in ("train", "test"):
        for name in ("multi", "two_marker_lines", "rectangle", "crossing",
                     "junction", "isolated_only", "adjacent_unchanged",
                     "short_positive", "short_multi", "unequal_multi",
                     "interior_filled", "separate_parallel", "large_dense",
                     "small_sparse"):
            _require(observed.get((split, name), 0) > 0,
                     f"review category absent: {split} {name}")
