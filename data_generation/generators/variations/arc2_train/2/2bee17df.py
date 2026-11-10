import numpy as np


# Exact original inputs are excluded; dimensions and colors remain available.
_SOURCE_INPUTS = {
    ("888888888888", "200000888888", "220000088008", "200000080008",
     "200000000008", "222000000008", "220000000008", "220000000008",
     "200000000008", "200000000022", "220200200222", "222222222222"),
    ("222222222222", "800022022222", "800002002202", "800000000002",
     "800000000002", "800000000002", "800000000002", "800000000008",
     "800000000008", "880000880008", "888008880088", "888888888888"),
    ("8888888882", "8880880802", "8800800002", "8800000022",
     "8000000022", "8000000002", "8000000002", "8000002202",
     "8200022222", "8222222222"),
    ("88888888888888", "88880088800822", "88800000000002",
     "80000000000002", "88800000000022", "88000000000222",
     "80000000000002", "88000000000002", "88000000000002",
     "88000000000002", "80000000000002", "88000000000022",
     "88020220000222", "82222222222222"),
}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _perimeter(height, width):
    return ([(0, c) for c in range(width)]
            + [(r, width - 1) for r in range(1, height)]
            + [(height - 1, c) for c in range(width - 2, -1, -1)]
            + [(r, 0) for r in range(height - 2, 0, -1)])


def _border(rng, grid, colors, rows, cols, evidence):
    coords = _perimeter(*grid.shape)
    length = len(coords)
    for _ in range(80):
        # Alternating cyclic patches give two, four, six, or eight transitions.
        patches = int(rng.choice((2, 2, 2, 4, 4, 6, 8)))
        concentration = float(rng.choice((0.35, 0.7, 1.5, 3.0)))
        sizes = rng.multinomial(
            length - patches,
            rng.dirichlet(np.full(patches, concentration)),
        ) + 1
        sequence = []
        for index, size in enumerate(sizes):
            sequence.extend([colors[index % 2]] * int(size))
        offset = int(rng.integers(length))
        for index, (row, col) in enumerate(coords):
            grid[row, col] = sequence[(index + offset) % length]
        endpoint_equal = ([grid[r, 0] == grid[r, -1] for r in rows]
                          + [grid[0, c] == grid[-1, c] for c in cols])
        if evidence == "same" and not any(endpoint_equal):
            continue
        if evidence == "different" and not any(not x for x in endpoint_equal):
            continue
        return
    raise AssertionError("could not establish endpoint evidence")


def _run_lengths(indices):
    if not indices:
        return ()
    lengths = []
    current = 1
    for a, b in zip(indices, indices[1:]):
        if b == a + 1:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return tuple(lengths)


def _lines(rng, size, style, central=False):
    interior = np.arange(1, size - 1, dtype=int)
    if central:
        interior = np.arange(3, size - 3, dtype=int)
    if len(interior) == 0:
        return ()
    if style == "empty":
        return ()
    if style == "single":
        return (int(rng.choice(interior)),)
    if style == "edge":
        edge = int(rng.choice((1, size - 2)))
        extras = list(int(x) for x in rng.choice(
            np.arange(1, size - 1), size=min(2, size - 3), replace=False
        ))
        return tuple(sorted(set([edge] + extras)))
    if style == "band":
        maximum = min(len(interior), max(2, (size - 2) // 2))
        band_size = int(rng.integers(2, maximum + 1))
        start = int(rng.integers(int(interior[0]), int(interior[-1]) - band_size + 2))
        return tuple(range(start, start + band_size))
    if style == "separated":
        candidates = [int(x) for x in rng.permutation(interior)]
        chosen = []
        target = int(rng.integers(2, min(5, (len(interior) + 1) // 2) + 1))
        for value in candidates:
            if all(abs(value - prior) > 1 for prior in chosen):
                chosen.append(value)
            if len(chosen) == target:
                break
        if not chosen:
            return (int(rng.choice(interior)),)
        return tuple(sorted(chosen))
    if style == "mixed":
        start_probability = float(rng.uniform(0.18, 0.55))
        continuation = float(rng.uniform(0.35, 0.85))
        selected = []
        previous = False
        for value in interior:
            previous = bool(rng.random() < (continuation if previous else start_probability))
            if previous:
                selected.append(int(value))
        if not selected:
            selected = [int(rng.choice(interior))]
        if len(selected) == size - 2:
            selected.pop(int(rng.integers(len(selected))))
        return tuple(selected)
    raise ValueError("unknown line style")


def _dimensions(rng, regime, index):
    if regime == "source_train" and index < 2:
        return int(rng.integers(10, 20)), int(rng.integers(10, 20))
    category = int(rng.integers(5))
    if category == 0:
        return int(rng.integers(7, 13)), int(rng.integers(7, 14))
    if category == 1:
        return int(rng.integers(21, 31)), int(rng.integers(21, 31))
    if category == 2:
        if rng.random() < 0.5:
            return int(rng.integers(7, 14)), int(rng.integers(20, 31))
        return int(rng.integers(20, 31)), int(rng.integers(7, 14))
    if category == 3 and regime == "target_test":
        return int(rng.integers(14, 31)), int(rng.integers(14, 31))
    return int(rng.integers(9, 25)), int(rng.integers(9, 25))


def _layout(rng, height, width, regime, index):
    if regime == "source_train" and index < 2:
        styles = ("single", "band", "separated", "mixed")
        row_style = str(rng.choice(styles))
        col_style = str(rng.choice(styles))
        rows = _lines(rng, height, row_style, central=True)
        cols = _lines(rng, width, col_style, central=True)
        return rows, cols, "row" if index == 0 else "col"

    for _ in range(30):
        regime_style = str(rng.choice(
            ("both", "both", "both", "both", "horizontal", "vertical",
             "full", "unchanged", "source_mix")
        ))
        if regime_style == "full":
            return tuple(range(1, height - 1)), tuple(range(1, width - 1)), None
        if regime_style == "unchanged":
            return (), (), None
        if regime_style == "source_mix":
            if height >= 11 and width >= 11:
                if rng.random() < 0.5:
                    rows = _lines(rng, height, "separated")
                    cols = _lines(rng, width, "mixed")
                else:
                    rows = _lines(rng, height, "mixed")
                    cols = _lines(rng, width, "separated")
            else:
                regime_style = "both"
        if regime_style != "source_mix":
            styles = ("single", "band", "separated", "mixed", "edge")
            rows = _lines(rng, height, str(rng.choice(styles))) if regime_style != "vertical" else ()
            cols = _lines(rng, width, str(rng.choice(styles))) if regime_style != "horizontal" else ()
        # A side of each axis remains accessible to border-connected fingers.
        if (not (1 in rows and height - 2 in rows)
                and not (1 in cols and width - 2 in cols)
                and len(rows) < height - 2 and len(cols) < width - 2):
            return rows, cols, None
    return (height // 2,), (width // 2,), None


def _choose_witness(rows, cols, height, width, axis):
    row_set, col_set = set(rows), set(cols)
    if axis == "row":
        supports = [c for c in range(3, width - 3) if c not in col_set]
        targets = [
            r for r in range(2, height - 2) if r not in row_set
            and (all(x not in row_set for x in range(1, r + 1))
                 or all(x not in row_set for x in range(r, height - 1)))
        ]
        return [(r, c) for r in targets for c in supports]
    if axis == "col":
        supports = [r for r in range(3, height - 3) if r not in row_set]
        targets = [
            c for c in range(2, width - 2) if c not in col_set
            and (all(x not in col_set for x in range(1, c + 1))
                 or all(x not in col_set for x in range(c, width - 1)))
        ]
        return [(r, c) for r in supports for c in targets]
    return []


def _finger(rng, grid, allowed, forbidden, axis, position, side, depth=None):
    height, width = grid.shape
    if axis == "row":
        coords = ([(position, c) for c in range(1, width - 1)]
                  if side == "left" else
                  [(position, c) for c in range(width - 2, 0, -1)])
        color = int(grid[position, 0 if side == "left" else width - 1])
    else:
        coords = ([(r, position) for r in range(1, height - 1)]
                  if side == "top" else
                  [(r, position) for r in range(height - 2, 0, -1)])
        color = int(grid[0 if side == "top" else height - 1, position])
    available = []
    for row, col in coords:
        if not allowed[row, col] or forbidden[row, col]:
            break
        available.append((row, col))
    if not available:
        return False
    if depth is None:
        if rng.random() < 0.30:
            depth = int(rng.integers(max(1, len(available) // 2), len(available) + 1))
        else:
            depth = int(rng.integers(1, len(available) + 1))
    for row, col in available[:depth]:
        grid[row, col] = color
    return True


def _grow_branches(rng, grid, allowed, forbidden, background):
    height, width = grid.shape
    branches = int(rng.integers(1, max(3, (height + width) // 3)))
    for _ in range(branches):
        parents = np.argwhere(
            allowed & (grid != background) & ~forbidden
        )
        if len(parents) == 0:
            return
        row, col = (int(x) for x in parents[int(rng.integers(len(parents)))])
        steps = int(rng.integers(1, max(3, max(height, width) // 2 + 1)))
        for _ in range(steps):
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if (0 < rr < height - 1 and 0 < cc < width - 1
                        and allowed[rr, cc] and not forbidden[rr, cc]
                        and grid[rr, cc] == background):
                    neighbors.append((rr, cc))
            if not neighbors:
                break
            rr, cc = neighbors[int(rng.integers(len(neighbors)))]
            grid[rr, cc] = grid[row, col]
            row, col = rr, cc


def _build_input(rng, height, width, rows, cols, palette, evidence, witness_axis):
    background, structure_a, structure_b = palette
    grid = np.full((height, width), background, dtype=np.int8)
    _border(rng, grid, (structure_a, structure_b), rows, cols, evidence)
    allowed = np.ones((height, width), dtype=bool)
    allowed[0, :] = allowed[-1, :] = False
    allowed[:, 0] = allowed[:, -1] = False
    allowed[list(rows), :] = False
    allowed[:, list(cols)] = False
    forbidden = np.zeros_like(allowed)
    candidates = _choose_witness(rows, cols, height, width, witness_axis)
    if witness_axis is not None and not candidates:
        raise ValueError("required inward obstruction does not fit")
    witness = (candidates[int(rng.integers(len(candidates)))]
               if candidates else None)
    if witness is not None:
        wr, wc = witness
        if witness_axis == "row":
            forbidden[wr, 1:-1] = True
            forbidden[wr, wc] = False
            side = ("top" if all(r not in rows for r in range(1, wr + 1))
                    else "bottom")
            distance = wr if side == "top" else height - 1 - wr
            assert _finger(rng, grid, allowed, forbidden, "col", wc, side, distance)
        else:
            forbidden[1:-1, wc] = True
            forbidden[wr, wc] = False
            side = ("left" if all(c not in cols for c in range(1, wc + 1))
                    else "right")
            distance = wc if side == "left" else width - 1 - wc
            assert _finger(rng, grid, allowed, forbidden, "row", wr, side, distance)

    nonclear_rows = [r for r in range(1, height - 1) if r not in rows]
    nonclear_cols = [c for c in range(1, width - 1) if c not in cols]
    for row in nonclear_rows:
        if witness_axis == "row" and row == witness[0]:
            continue
        if rng.random() < 0.76:
            sides = ("left", "right") if rng.random() < 0.5 else ("right", "left")
            if not _finger(rng, grid, allowed, forbidden, "row", row, sides[0]):
                _finger(rng, grid, allowed, forbidden, "row", row, sides[1])
            if rng.random() < 0.24:
                _finger(rng, grid, allowed, forbidden, "row", row, sides[1])
    for col in nonclear_cols:
        if witness_axis == "col" and col == witness[1]:
            continue
        if rng.random() < 0.76:
            sides = ("top", "bottom") if rng.random() < 0.5 else ("bottom", "top")
            if not _finger(rng, grid, allowed, forbidden, "col", col, sides[0]):
                _finger(rng, grid, allowed, forbidden, "col", col, sides[1])
            if rng.random() < 0.24:
                _finger(rng, grid, allowed, forbidden, "col", col, sides[1])

    _grow_branches(rng, grid, allowed, forbidden, background)
    for row in nonclear_rows:
        if np.all(grid[row, 1:-1] == background):
            sides = ("left", "right") if rng.random() < 0.5 else ("right", "left")
            if not _finger(rng, grid, allowed, forbidden, "row", row, sides[0]):
                assert _finger(rng, grid, allowed, forbidden, "row", row, sides[1])
    for col in nonclear_cols:
        if np.all(grid[1:-1, col] == background):
            sides = ("top", "bottom") if rng.random() < 0.5 else ("bottom", "top")
            if not _finger(rng, grid, allowed, forbidden, "col", col, sides[0]):
                assert _finger(rng, grid, allowed, forbidden, "col", col, sides[1])
    assert _selected(grid, background) == (rows, cols)
    return grid


def _selected(grid, background):
    height, width = grid.shape
    rows = tuple(r for r in range(1, height - 1)
                 if np.all(grid[r, 1:-1] == background))
    cols = tuple(c for c in range(1, width - 1)
                 if np.all(grid[1:-1, c] == background))
    return rows, cols


def _transform(grid, background, highlight):
    rows, cols = _selected(grid, background)
    result = grid.copy()
    for row in rows:
        result[row, 1:-1] = highlight
    for col in cols:
        result[1:-1, col] = highlight
    return result


def _palette(rng, highlight, previous=None, require_role_change=False):
    choices = [color for color in range(10) if color != highlight]
    for _ in range(30):
        palette = tuple(int(x) for x in rng.choice(choices, size=3, replace=False))
        if (palette != previous
                and (not require_role_change or (
                    palette[0] != previous[0]
                    and set(palette[1:]) != set(previous[1:])
                ))):
            return palette
    raise AssertionError("palette sampling failed")


def _signature(grid):
    return grid.shape, grid.tobytes()


def _row_signature(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _make_pair(rng, regime, index, highlight, previous_palette):
    for attempt in range(40):
        height, width = _dimensions(rng, regime, index)
        rows, cols, witness_axis = _layout(rng, height, width, regime, index)
        if witness_axis is None and rows != tuple(range(1, height - 1)):
            if rng.random() < 0.55:
                candidate_axis = "row" if rng.random() < 0.5 else "col"
                if _choose_witness(rows, cols, height, width, candidate_axis):
                    witness_axis = candidate_axis
        palette = _palette(
            rng, highlight, previous_palette,
            require_role_change=(regime == "source_train" and index == 1),
        )
        evidence = "same" if regime == "source_train" and index == 0 else (
            "different" if regime == "source_train" and index == 1 else None
        )
        try:
            input_grid = _build_input(
                rng, height, width, rows, cols, palette, evidence, witness_axis
            )
        except (ValueError, AssertionError):
            continue
        if _row_signature(input_grid) in _SOURCE_INPUTS:
            continue
        return {"input": input_grid,
                "output": _transform(input_grid, palette[0], highlight)}, palette
    raise AssertionError("could not construct a valid pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    highlight = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_palette = None
    for regime, count, split in (("source_train", n_train, "train"),
                                 ("target_test", n_test, "test")):
        for index in range(count):
            for _ in range(30):
                pair, palette = _make_pair(rng, regime, index, highlight, previous_palette)
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    previous_palette = palette
                    break
            else:
                raise AssertionError("duplicate pair generation")
    validate_generated_puzzle(puzzle)
    return puzzle


def _connected_to_border(grid, structural):
    mask = np.isin(grid, tuple(structural))
    height, width = grid.shape
    seen = np.zeros_like(mask)
    stack = []
    for row, col in _perimeter(height, width):
        if mask[row, col] and not seen[row, col]:
            seen[row, col] = True
            stack.append((row, col))
    while stack:
        row, col = stack.pop()
        for rr, cc in ((row - 1, col), (row + 1, col),
                       (row, col - 1), (row, col + 1)):
            if (0 <= rr < height and 0 <= cc < width
                    and mask[rr, cc] and not seen[rr, cc]):
                seen[rr, cc] = True
                stack.append((rr, cc))
    return bool(np.all(seen[mask]))


def _features(pair, highlight):
    inp, out = pair["input"], pair["output"]
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    for grid in (inp, out):
        assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    height, width = inp.shape
    assert height >= 7 and width >= 7
    border = [int(inp[r, c]) for r, c in _perimeter(height, width)]
    structural = frozenset(border)
    assert len(structural) == 2
    inp_colors = frozenset(int(x) for x in np.unique(inp))
    assert len(inp_colors) == 3 and highlight not in inp_colors
    background_set = inp_colors - structural
    assert len(background_set) == 1
    background = next(iter(background_set))
    assert _connected_to_border(inp, structural)
    rows, cols = _selected(inp, background)
    assert np.array_equal(out, _transform(inp, background, highlight))
    assert _row_signature(inp) not in _SOURCE_INPUTS
    edge_only_rows = tuple(r for r in range(1, height - 1)
                           if inp[r, 1] == background and inp[r, -2] == background)
    edge_only_cols = tuple(c for c in range(1, width - 1)
                           if inp[1, c] == background and inp[-2, c] == background)
    single_hidden_rows = tuple(
        r not in rows and np.count_nonzero(inp[r, 1:-1] != background) == 1
        for r in edge_only_rows
    )
    single_hidden_cols = tuple(
        c not in cols and np.count_nonzero(inp[1:-1, c] != background) == 1
        for c in edge_only_cols
    )
    single_hidden_row_positions = tuple(
        r for r, single in zip(edge_only_rows, single_hidden_rows) if single
    )
    single_hidden_col_positions = tuple(
        c for c, single in zip(edge_only_cols, single_hidden_cols) if single
    )
    endpoint_equal = tuple(
        [bool(inp[r, 0] == inp[r, -1]) for r in rows]
        + [bool(inp[0, c] == inp[-1, c]) for c in cols]
    )
    transitions = sum(a != b for a, b in zip(border, border[1:] + border[:1]))
    structural_positions = np.argwhere(np.isin(inp, tuple(structural)))
    max_structural_depth = max(
        min(int(r), int(c), height - 1 - int(r), width - 1 - int(c))
        for r, c in structural_positions
    )
    minority_share = min(border.count(color) for color in structural) / len(border)
    return {"shape": inp.shape, "palette": (background, structural),
            "rows": rows, "cols": cols,
            "row_runs": _run_lengths(rows), "col_runs": _run_lengths(cols),
            "hidden_row": bool(set(edge_only_rows) - set(rows)),
            "hidden_col": bool(set(edge_only_cols) - set(cols)),
            "single_hidden_row": bool(single_hidden_row_positions),
            "single_hidden_col": bool(single_hidden_col_positions),
            "single_hidden_row_positions": single_hidden_row_positions,
            "single_hidden_col_positions": single_hidden_col_positions,
            "endpoint_equal": endpoint_equal, "transitions": transitions,
            "max_structural_depth": max_structural_depth,
            "border_minority_share": minority_share,
            "changed": not np.array_equal(inp, out)}


def _predict_from_training(train, test_input):
    highlights = set()
    for pair in train:
        before = set(int(x) for x in np.unique(pair["input"]))
        after = set(int(x) for x in np.unique(pair["output"]))
        highlights.update(after - before)
    assert len(highlights) == 1
    highlight = next(iter(highlights))
    height, width = test_input.shape
    border = set(int(test_input[r, c]) for r, c in _perimeter(height, width))
    background = set(int(x) for x in np.unique(test_input)) - border
    assert len(background) == 1
    background = next(iter(background))
    result = test_input.copy()
    row_flags = [all(int(test_input[r, c]) == background for c in range(1, width - 1))
                 for r in range(1, height - 1)]
    col_flags = [all(int(test_input[r, c]) == background for r in range(1, height - 1))
                 for c in range(1, width - 1)]
    for r, flag in enumerate(row_flags, 1):
        if flag:
            result[r, 1:-1] = highlight
    for c, flag in enumerate(col_flags, 1):
        if flag:
            result[1:-1, c] = highlight
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    highlight_candidates = set()
    for pair in puzzle["train"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        highlight_candidates |= (
            set(int(x) for x in np.unique(pair["output"]))
            - set(int(x) for x in np.unique(pair["input"]))
        )
    assert len(highlight_candidates) == 1
    highlight = next(iter(highlight_candidates))
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            analysis = _features(pair, highlight)
            analyses[split].append(analysis)
            inp_sig = _signature(pair["input"])
            pair_sig = inp_sig, _signature(pair["output"])
            assert inp_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(inp_sig)
            seen_pairs.add(pair_sig)
    # Both axes and the fixed new color are demonstrated even at the count minimum.
    assert analyses["train"][0]["rows"] and analyses["train"][0]["cols"]
    assert analyses["train"][1]["rows"] and analyses["train"][1]["cols"]
    assert analyses["train"][0]["hidden_row"]
    assert analyses["train"][1]["hidden_col"]
    assert analyses["train"][0]["single_hidden_row"]
    assert analyses["train"][1]["single_hidden_col"]
    assert len({a["palette"] for a in analyses["train"]}) >= 2
    assert analyses["train"][0]["palette"][0] != analyses["train"][1]["palette"][0]
    assert analyses["train"][0]["palette"][1] != analyses["train"][1]["palette"][1]
    equalities = [x for a in analyses["train"] for x in a["endpoint_equal"]]
    assert any(equalities) and any(not x for x in equalities)
    for pair in puzzle["test"]:
        assert np.array_equal(
            _predict_from_training(puzzle["train"], pair["input"]),
            pair["output"],
        )


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split])
               and all(np.array_equal(x[role], y[role])
                       for x, y in zip(a[split], b[split])
                       for role in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(7001)
    assert _equal_puzzles(automatic, generate_puzzle(7001))
    for offset, (train, test) in enumerate(
        ((2, 1), (10, 1), (10, 4), (None, 1),
         (None, 4), (2, None), (10, None), (6, 3))
    ):
        puzzle = generate_puzzle(7100 + offset, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 2.0, "num_test": 1},
    ):
        try:
            generate_puzzle(7300, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    broken = generate_puzzle(7400, 3, 1)
    broken["train"][0]["output"][1, 1] = (int(broken["train"][0]["output"][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")

    train_counts, test_counts, shapes, palettes = set(), set(), set(), set()
    layouts, row_runs, col_runs, transitions = set(), set(), set(), set()
    endpoint_evidence = set()
    row_counts, col_counts = set(), set()
    single_row_positions, single_col_positions = set(), set()
    witnessed = {"row_train": 0, "col_train": 0, "row_test": 0, "col_test": 0,
                 "single_row": 0, "single_col": 0,
                 "horizontal": 0, "vertical": 0, "full": 0, "unchanged": 0,
                 "edge": 0, "wide_row": 0, "wide_col": 0,
                 "row_mixed": 0, "col_mixed": 0, "small": 0, "thirty": 0,
                 "wide_shape": 0, "tall_shape": 0, "multi_patch": 0,
                 "larger_test": 0, "overlapping_test_shape": 0,
                 "deep_5": 0, "deep_8": 0, "uneven_border": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        train_palettes = [
            _features(pair, _predict_highlight(puzzle["train"]))["palette"]
            for pair in puzzle["train"]
        ]
        assert len({palette[0] for palette in train_palettes}) >= 2
        assert len({palette[1] for palette in train_palettes}) >= 2
        for split in ("train", "test"):
            for pair in puzzle[split]:
                feature = _features(pair, _predict_highlight(puzzle["train"]))
                h, w = feature["shape"]
                shapes.add((h, w))
                palettes.add(feature["palette"])
                row_counts.add(len(feature["rows"]))
                col_counts.add(len(feature["cols"]))
                endpoint_evidence.update(feature["endpoint_equal"])
                single_row_positions.update(feature["single_hidden_row_positions"])
                single_col_positions.update(feature["single_hidden_col_positions"])
                layouts.add((feature["shape"], feature["rows"], feature["cols"]))
                row_runs.add(feature["row_runs"])
                col_runs.add(feature["col_runs"])
                transitions.add(feature["transitions"])
                witnessed["row_" + split] += int(feature["hidden_row"])
                witnessed["col_" + split] += int(feature["hidden_col"])
                witnessed["single_row"] += int(feature["single_hidden_row"])
                witnessed["single_col"] += int(feature["single_hidden_col"])
                witnessed["horizontal"] += int(bool(feature["rows"]) and not feature["cols"])
                witnessed["vertical"] += int(bool(feature["cols"]) and not feature["rows"])
                witnessed["full"] += int(len(feature["rows"]) == h - 2)
                witnessed["unchanged"] += int(not feature["changed"])
                witnessed["edge"] += int(1 in feature["rows"] or h - 2 in feature["rows"]
                                         or 1 in feature["cols"] or w - 2 in feature["cols"])
                witnessed["wide_row"] += int(bool(feature["row_runs"]) and max(feature["row_runs"]) >= 4)
                witnessed["wide_col"] += int(bool(feature["col_runs"]) and max(feature["col_runs"]) >= 4)
                witnessed["row_mixed"] += int(len(feature["row_runs"]) >= 2
                                              and max(feature["row_runs"]) >= 2)
                witnessed["col_mixed"] += int(len(feature["col_runs"]) >= 2
                                              and max(feature["col_runs"]) >= 2)
                witnessed["small"] += int(min(h, w) <= 9)
                witnessed["thirty"] += int(h == 30 or w == 30)
                witnessed["wide_shape"] += int(w >= 2 * h)
                witnessed["tall_shape"] += int(h >= 2 * w)
                witnessed["multi_patch"] += int(feature["transitions"] >= 4)
                witnessed["deep_5"] += int(feature["max_structural_depth"] >= 5)
                witnessed["deep_8"] += int(feature["max_structural_depth"] >= 8)
                witnessed["uneven_border"] += int(feature["border_minority_share"] < 0.2)
                if split == "test":
                    witnessed["larger_test"] += int(
                        h > max(shape[0] for shape in train_shapes)
                        and w > max(shape[1] for shape in train_shapes)
                    )
                    witnessed["overlapping_test_shape"] += int(
                        (h, w) in train_shapes
                    )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(palettes) >= 40 and len(layouts) >= 150
    assert len(row_runs) >= 12 and len(col_runs) >= 12
    assert min(row_counts) == 0 and max(row_counts) >= 6
    assert min(col_counts) == 0 and max(col_counts) >= 6
    assert endpoint_evidence == {False, True}
    assert len(single_row_positions) >= 5 and len(single_col_positions) >= 5
    assert 2 in transitions and any(x >= 4 for x in transitions)
    assert all(value > 0 for value in witnessed.values())


def _predict_highlight(train):
    colors = set()
    for pair in train:
        colors |= (set(int(x) for x in np.unique(pair["output"]))
                   - set(int(x) for x in np.unique(pair["input"])))
    assert len(colors) == 1
    return next(iter(colors))
