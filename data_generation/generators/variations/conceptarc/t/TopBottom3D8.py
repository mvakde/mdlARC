import numpy as np


_SOURCE_INPUTS = {
    ("000010", "000010", "000010", "222222", "000010", "000010"),
    ("030000", "030000", "030000", "434444", "030000", "030000"),
    ("000080", "000080", "777777", "000080", "000080", "000080"),
    ("080000", "585555", "080000", "555555", "080000", "080000"),
    ("090000", "090000", "090000", "333333", "090000", "090000"),
    ("040000", "888888", "040000", "040000", "747777", "040000"),
    ("050000", "350000", "030000", "053000", "050300", "050030"),
}


def _counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(num_train), int(num_test)


def _paths(height, width, column):
    """Maximal straight transverse paths, each visible on both sides."""
    paths = []
    for row in range(height):
        paths.append(("h", tuple((row, c) for c in range(width))))
    for slope in (-1, 1):
        intercepts = (range(0, height + width - 1) if slope == -1
                      else range(-width + 1, height))
        for intercept in intercepts:
            cells = tuple((slope * c + intercept, c) for c in range(width)
                          if 0 <= slope * c + intercept < height)
            if (len(cells) >= 5 and sum(c < column for _, c in cells) >= 2
                    and sum(c > column for _, c in cells) >= 2):
                paths.append(("d+" if slope == 1 else "d-", cells))
    return paths


def _compatible(first, second, column):
    left = first[1]
    right = second[1]
    a_cross = next(r for r, c in left if c == column)
    b_cross = next(r for r, c in right if c == column)
    if abs(a_cross - b_cross) < 3:
        return False
    # The transverse strokes have a blank one-cell moat. This also prevents
    # touching same-color strokes from appearing to be one broad object.
    for ar, ac in left:
        if ac == column:
            continue
        for br, bc in right:
            if bc == column:
                continue
            if max(abs(ar - br), abs(ac - bc)) <= 1:
                return False
    return True


def _choose_layout(rng, height, width, column, count, composition):
    all_paths = _paths(height, width, column)
    horizontal = [p for p in all_paths if p[0] == "h"]
    diagonal = [p for p in all_paths if p[0] != "h"]
    for _ in range(80):
        chosen = []
        if composition == "mixed":
            kinds = ["h", "d"] + ["h" if rng.integers(2) else "d"
                                     for _ in range(count - 2)]
        elif composition == "diagonal":
            kinds = ["d"] * count
        else:
            kinds = ["h"] * count
        rng.shuffle(kinds)
        for kind in kinds:
            pool = horizontal if kind == "h" else diagonal
            eligible = [p for p in pool if all(_compatible(p, q, column)
                                                for q in chosen)]
            if not eligible:
                break
            chosen.append(eligible[int(rng.integers(len(eligible)))])
        if len(chosen) == count:
            return sorted(chosen, key=lambda p: next(r for r, c in p[1]
                                                       if c == column))
    return None


def _pick_colors(rng, count, shared):
    if shared is None:
        background = int(rng.choice(np.arange(1, 10))) if rng.random() < .35 else 0
        available = [c for c in range(10) if c != background]
        backbone = int(rng.choice(available))
        remaining = [c for c in available if c != backbone]
    else:
        background, backbone, remaining = shared
    # A nonzero background often makes black one of the line colors.
    if background != 0 and 0 in remaining and rng.random() < .5:
        first = 0
    else:
        first = int(rng.choice(remaining))
    if count == 1:
        strokes = [first]
    else:
        distinct = bool(rng.integers(2))
        if distinct:
            colors = [first] + [int(c) for c in rng.choice(
                [c for c in remaining if c != first],
                size=min(count - 1, len(remaining) - 1), replace=False)]
            strokes = [colors[int(rng.integers(len(colors)))] for _ in range(count)]
            strokes[0], strokes[1] = colors[0], colors[1]
        else:
            strokes = [first] * count
    return background, backbone, strokes


def _build_pair(rng, spec, shared, shape, orientation):
    height, width = shape
    count, composition, states = spec
    for _ in range(45):
        column = int(rng.integers(2, width - 2))
        layout = _choose_layout(rng, height, width, column, count, composition)
        if layout is not None:
            break
    else:
        raise RuntimeError("No separated line layout")
    background, backbone, strokes = _pick_colors(rng, count, shared)
    grid = np.full(shape, background, dtype=np.int8)
    grid[:, column] = backbone
    for (_, cells), stroke in zip(layout, strokes):
        for row, col in cells:
            grid[row, col] = stroke
    output = grid.copy()
    for (_, cells), stroke, state in zip(layout, strokes, states):
        row = next(row for row, col in cells if col == column)
        grid[row, column] = backbone if state == "b" else stroke
        output[row, column] = stroke if state == "b" else backbone
    if orientation:
        grid = np.rot90(grid)
        output = np.rot90(output)
    return grid.copy(), output.copy()


def _shape(rng, count, fixed):
    if fixed is not None:
        return fixed
    minimum = max(7, 3 * count + 2)
    height = int(rng.integers(minimum, max(minimum + 1, 18)))
    width = int(rng.integers(7, 18))
    return height, width


def _spec(rng, section, index, regime, theme):
    if section == "train" and index < 2:
        if index == 0:
            count = 1 if rng.random() < .7 else 2
            states = ["b"] * count if rng.integers(2) else ["s"] * count
        else:
            count = 2
            states = [theme, "s" if theme == "b" else "b"]
        if index == 0:
            # The second demonstration always supplies the absent direction.
            theme = states[0]
    else:
        weights = ([.35, .37, .20, .08] if section == "train"
                   else [.22, .35, .29, .14])
        count = int(rng.choice(np.arange(1, 5), p=weights))
        states = ["b" if rng.integers(2) else "s" for _ in range(count)]
    if regime == "reverse" and section == "test" and index == 0:
        count = 1
        states = ["b" if rng.integers(2) else "s"]
    if count == 1:
        composition = "horizontal" if rng.random() < .55 else "diagonal"
    else:
        composition = str(rng.choice(["horizontal", "diagonal", "mixed"],
                                     p=[.56, .22, .22]))
    if regime == "hold_diagonal":
        if section == "train":
            composition = "horizontal"
        elif index == 0:
            composition = "diagonal"
    elif regime == "reverse":
        if section == "train" and index == 0:
            composition = "diagonal"
        elif section == "test" and index == 0:
            composition = "horizontal"
    return (count, composition, states), theme


def _grid_rows(grid):
    return tuple("".join(str(int(c)) for c in row) for row in grid)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    regime = str(rng.choice(["hold_diagonal", "reverse", "free"],
                            p=[.20, .20, .60]))
    same_shape = bool(rng.random() < .30)
    same_orientation = same_shape or bool(rng.random() < .30)
    fixed_orientation = int(rng.integers(2))
    shared_palette = None
    if rng.random() < .18:
        background = int(rng.integers(10)) if rng.random() < .4 else 0
        backbone = int(rng.choice([c for c in range(10) if c != background]))
        stroke_palette = [int(c) for c in rng.choice(
            [c for c in range(10) if c not in (background, backbone)],
            size=4, replace=False)]
        shared_palette = (background, backbone,
                          stroke_palette)
    fixed_shape = _shape(rng, 4, None) if same_shape else None
    puzzle = {"train": [], "test": []}
    seen = set()
    theme = "b"
    for section, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            spec, theme = _spec(rng, section, index, regime, theme)
            for _ in range(90):
                shape = _shape(rng, spec[0], fixed_shape)
                orientation = fixed_orientation if same_orientation else int(rng.integers(2))
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    grid, output = _build_pair(pair_rng, spec, shared_palette,
                                               shape, orientation)
                except RuntimeError:
                    continue
                key = (grid.shape, grid.tobytes())
                if key in seen or (grid.shape == (6, 6)
                                   and _grid_rows(grid) in _SOURCE_INPUTS):
                    continue
                seen.add(key)
                puzzle[section].append({"input": grid, "output": output})
                break
            else:
                raise RuntimeError("Could not construct a unique pair")
    return puzzle


def _decode(grid):
    """Infer every valid crossing prediction from input pixels alone."""
    predictions = {}
    for turned in (False, True):
        view = grid.T if turned else grid
        height, width = view.shape
        for background in np.unique(view):
            for column in range(2, width - 2):
                column_values = view[:, column]
                if np.any(column_values == background):
                    continue
                for backbone in np.unique(column_values):
                    if backbone == background or np.count_nonzero(
                            column_values == backbone) < height - 4:
                        continue
                    residual = {(r, c) for r in range(height) for c in range(width)
                                if c != column and view[r, c] != background}
                    if len(residual) < 4:
                        continue
                    candidates = []
                    for kind, cells in _paths(height, width, column):
                        off = frozenset((r, c) for r, c in cells if c != column)
                        if not off <= residual:
                            continue
                        colors = {int(view[r, c]) for r, c in off}
                        if len(colors) != 1:
                            continue
                        stroke = next(iter(colors))
                        if stroke in (background, backbone):
                            continue
                        row = next(r for r, c in cells if c == column)
                        if column_values[row] not in (backbone, stroke):
                            continue
                        candidates.append((kind, cells, off, stroke, row))
                    by_cell = {cell: [] for cell in residual}
                    for candidate in candidates:
                        for cell in candidate[2]:
                            by_cell[cell].append(candidate)

                    def search(uncovered, chosen):
                        if not uncovered:
                            if not 1 <= len(chosen) <= 4:
                                return
                            rows = [entry[4] for entry in chosen]
                            if len(set(rows)) != len(rows):
                                return
                            if any(not _compatible((a[0], a[1]), (b[0], b[1]),
                                                   column)
                                   for i, a in enumerate(chosen) for b in chosen[i+1:]):
                                return
                            if any(column_values[r] != backbone
                                   for r in range(height) if r not in rows):
                                return
                            predicted = view.copy()
                            for _, _, _, stroke, row in chosen:
                                predicted[row, column] = (stroke if column_values[row]
                                                          == backbone else backbone)
                            predicted = predicted.T if turned else predicted
                            key = (predicted.shape, predicted.tobytes())
                            predictions[key] = {
                                "output": predicted.copy(), "count": len(chosen),
                                "kinds": tuple(entry[0] for entry in chosen),
                                "states": tuple("b" if column_values[entry[4]]
                                                == backbone else "s" for entry in chosen),
                                "colors": tuple(entry[3] for entry in chosen),
                                "background": int(background),
                                "backbone": int(backbone),
                                "orientation": int(turned),
                                "shape": grid.shape,
                                "lengths": tuple(len(entry[1]) for entry in chosen),
                                "endpoints": tuple((entry[1][0], entry[1][-1])
                                                   for entry in chosen),
                                "crossing_rows": tuple(entry[4] for entry in chosen),
                                "column": column,
                            }
                            return
                        cell = min(uncovered, key=lambda p: len(by_cell[p]))
                        for candidate in by_cell[cell]:
                            if candidate[2] <= uncovered:
                                search(uncovered - candidate[2], chosen + [candidate])

                    search(frozenset(residual), [])
                    if len(predictions) > 1:
                        return list(predictions.values())
    return list(predictions.values())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs = set()
    seen_pairs = set()
    train_states = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert not (inp.shape == (6, 6) and _grid_rows(inp) in _SOURCE_INPUTS)
            solutions = _decode(inp)
            assert len(solutions) == 1, "Input has no unique line-switch prediction"
            report = solutions[0]
            assert np.array_equal(out, report["output"])
            assert np.count_nonzero(inp != out) == report["count"]
            assert set(np.unique(out)) <= set(np.unique(inp))
            if section == "train":
                train_states.update(report["states"])
    assert train_states == {"b", "s"}


def _same_puzzle(a, b):
    return all(len(a[part]) == len(b[part]) and all(
        np.array_equal(x[key], y[key])
        for x, y in zip(a[part], b[part]) for key in ("input", "output"))
        for part in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, backgrounds = set(), set(), set(), set()
    kinds_train, kinds_test, counts, states = set(), set(), set(), set()
    orientations, lengths, layouts, signatures = set(), set(), set(), set()
    paired_shapes = paired_orientations = paired_counts = 0
    same_shapes = same_orientations = repeated_palettes = 0
    crossing_columns, boundary_pairs = set(), set()
    two_stroke_cases = set()
    nonzero_with_zero_line = mixed_geometry = parallel_diagonals = 0
    complex_train = simple_test = diagonal_train = horizontal_test = 0
    reverse_shift = distinct_to_shared = diagonal_holdout = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for section in ("train", "test") for p in puzzle[section]))
        reports = {section: [_decode(p["input"])[0] for p in puzzle[section]]
                   for section in ("train", "test")}
        all_reports = reports["train"] + reports["test"]
        paired_shapes += len({r["shape"] for r in all_reports}) > 1
        paired_orientations += len({r["orientation"] for r in all_reports}) > 1
        paired_counts += len({r["count"] for r in all_reports}) > 1
        same_shapes += len({r["shape"] for r in all_reports}) == 1
        same_orientations += len({r["orientation"] for r in all_reports}) == 1
        palettes = [tuple(np.unique(pair["input"]))
                    for section in ("train", "test") for pair in puzzle[section]]
        repeated_palettes += len(palettes) > len(set(palettes))
        complex_train += any(r["count"] >= 2 for r in reports["train"])
        simple_test += any(r["count"] == 1 for r in reports["test"])
        diagonal_train += any(any(k != "h" for k in r["kinds"])
                              for r in reports["train"])
        horizontal_test += any(all(k == "h" for k in r["kinds"])
                               for r in reports["test"])
        reverse_shift += (any(any(k != "h" for k in r["kinds"])
                              for r in reports["train"])
                          and any(all(k == "h" for k in r["kinds"])
                                  for r in reports["test"]))
        diagonal_holdout += (all(all(k == "h" for k in r["kinds"])
                                 for r in reports["train"])
                             and any(any(k != "h" for k in r["kinds"])
                                     for r in reports["test"]))
        distinct_to_shared += (any(r["count"] > 1 and len(set(r["colors"])) > 1
                                   for r in reports["train"])
                               and any(r["count"] > 1 and len(set(r["colors"])) == 1
                                       for r in reports["test"]))
        for section in ("train", "test"):
            for pair, report in zip(puzzle[section], reports[section]):
                (kinds_train if section == "train" else kinds_test).update(report["kinds"])
                shapes.add(report["shape"])
                backgrounds.add(report["background"])
                orientations.add(report["orientation"])
                lengths.update(report["lengths"])
                crossing_columns.add((report["orientation"], report["column"]))
                counts.add(report["count"])
                states.add(report["states"])
                layouts.add((report["shape"], report["count"], report["kinds"],
                             report["lengths"]))
                nonzero_with_zero_line += (report["background"] != 0
                                           and 0 in np.unique(pair["input"]))
                mixed_geometry += ("h" in report["kinds"]
                                   and any(k != "h" for k in report["kinds"]))
                parallel_diagonals += (report["kinds"].count("d+") >= 2
                                       or report["kinds"].count("d-") >= 2)
                if report["count"] == 2:
                    palette = "shared" if len(set(report["colors"])) == 1 else "distinct"
                    visibility = ("all_b" if set(report["states"]) == {"b"}
                                  else "all_s" if set(report["states"]) == {"s"}
                                  else "mixed")
                    two_stroke_cases.add((section, palette, visibility))
                for kind, endpoints in zip(report["kinds"], report["endpoints"]):
                    if kind != "h":
                        first, last = endpoints
                        height, width = (report["shape"][::-1]
                                         if report["orientation"] else report["shape"])
                        edge = lambda point: ("L" if point[1] == 0 else
                                              "R" if point[1] == width - 1 else
                                              "T" if point[0] == 0 else "B")
                        boundary_pairs.add((kind, edge(first), edge(last)))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(shapes) >= 25
    assert paired_shapes >= 15 and paired_orientations >= 10
    assert paired_counts >= 25
    assert same_shapes >= 5 and same_orientations >= 5
    assert repeated_palettes >= 5
    assert len(backgrounds) >= 4 and 0 in backgrounds
    assert nonzero_with_zero_line >= 10
    assert {1, 2, 3, 4} <= counts
    assert {"h", "d+", "d-"} <= kinds_train and {"h", "d+", "d-"} <= kinds_test
    assert {0, 1} <= orientations and len(lengths) >= 8
    assert len(crossing_columns) >= 10 and len(boundary_pairs) >= 6
    assert len(layouts) >= 100 and len(states) >= 8
    assert mixed_geometry >= 10 and parallel_diagonals >= 5
    assert complex_train >= 20 and simple_test >= 10
    assert diagonal_train >= 20 and horizontal_test >= 10
    assert diagonal_holdout >= 2 and reverse_shift >= 10
    assert distinct_to_shared >= 10
    assert all((section, palette, visibility) in two_stroke_cases
               for section in ("train", "test")
               for palette in ("shared", "distinct")
               for visibility in ("all_b", "all_s", "mixed"))
    for n_train, n_test in ((2, 1), (10, 1), (10, 4), (6, 3),
                            (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(100 + (n_train or 0) + (n_test or 0), n_train, n_test)
        validate_generated_puzzle(puzzle)
        if n_train is not None:
            assert len(puzzle["train"]) == n_train
        if n_test is not None:
            assert len(puzzle["test"]) == n_test
    for n_train, n_test in ((1, None), (11, None), (None, 0), (None, 5),
                            (2, 2), (3, 4), (4, 4), (True, 1)):
        try:
            generate_puzzle(900, n_train, n_test)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid pair counts accepted")
