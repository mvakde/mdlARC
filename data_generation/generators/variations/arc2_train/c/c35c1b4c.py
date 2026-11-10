from __future__ import annotations

import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "9000077090",
            "0090009990",
            "7703333797",
            "0373333937",
            "0393300039",
            "9333300930",
            "3333390037",
            "3333309930",
            "0903339999",
            "7979007700",
        ),
        (
            "9000077090",
            "0090009990",
            "7703333797",
            "0373333937",
            "0393333039",
            "9333333330",
            "3333333333",
            "3333333333",
            "0903333999",
            "7979007700",
        ),
    ),
    (
        (
            "6688808060",
            "0800666680",
            "6601111066",
            "0011111100",
            "8111111100",
            "6111111160",
            "6111111168",
            "0811186800",
            "6860608068",
            "8606066808",
        ),
        (
            "6688808060",
            "0800666680",
            "6601111066",
            "0011111100",
            "8111111110",
            "6111111110",
            "6111111118",
            "0811111100",
            "6860608068",
            "8606066808",
        ),
    ),
    (
        (
            "1101100041",
            "4404221444",
            "4022222210",
            "0422200110",
            "0012221010",
            "0402202000",
            "2222222222",
            "4141224414",
            "0444211441",
            "4044041140",
        ),
        (
            "1101100041",
            "4404221444",
            "4022222210",
            "0422222210",
            "0012222010",
            "0402222000",
            "2222222222",
            "4141224414",
            "0444221441",
            "4044041140",
        ),
    ),
    (
        (
            "0066660660",
            "2606906026",
            "2669999066",
            "2009909602",
            "9999996000",
            "9999999900",
            "0099996600",
            "2999999626",
            "0029069026",
            "6002060662",
        ),
        (
            "0066660660",
            "2606996026",
            "2669999066",
            "2009999602",
            "9999999999",
            "9999999999",
            "0099999900",
            "2999999996",
            "0029069026",
            "6002060662",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _is_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    for input_rows, output_rows in _SOURCE_PAIR_ROWS:
        source_input = _rows_to_grid(input_rows)
        source_output = _rows_to_grid(output_rows)
        if (
            np.array_equal(inp, source_input)
            and np.array_equal(out, source_output)
        ):
            return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    start = (int(points[0, 0]), int(points[0, 1]))
    stack = [start]
    seen = {start}
    height, width = mask.shape

    while stack:
        row, column = stack.pop()

        for delta_row, delta_column in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            new_row = row + delta_row
            new_column = column + delta_column

            if (
                0 <= new_row < height
                and 0 <= new_column < width
                and bool(mask[new_row, new_column])
                and (new_row, new_column) not in seen
            ):
                seen.add((new_row, new_column))
                stack.append((new_row, new_column))

    return len(seen) == int(np.count_nonzero(mask))


def _validate_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)

    if not low <= result <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_count(
        "num_train",
        num_train,
        2,
        10,
    )
    test_count = _validate_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        low = max(2, test_count + 1)
        span = 10 - low + 1
        train_count = low + (seed // 4) % span

    elif train_count is None:
        low = max(2, int(test_count) + 1)
        train_count = int(rng.integers(low, 11))

    elif test_count is None:
        high = min(4, train_count - 1)

        if high < 1:
            raise ValueError(
                "no legal num_test exists for num_train"
            )

        test_count = int(rng.integers(1, high + 1))

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return int(train_count), int(test_count)


def _largest_component(mask: np.ndarray) -> int:
    remaining = mask.copy()
    height, width = mask.shape
    largest = 0
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if not remaining[row, column]:
            continue
        remaining[row, column] = False
        stack = [(row, column)]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and remaining[rr, cc]:
                    remaining[rr, cc] = False
                    stack.append((rr, cc))
        largest = max(largest, size)
    return largest


def _background_from_input(inp: np.ndarray) -> int:
    corners = (int(inp[0, 0]), int(inp[0, -1]), int(inp[-1, 0]), int(inp[-1, -1]))
    return max(set(corners), key=lambda color: corners.count(color))


def _figure_from_input(inp: np.ndarray) -> int:
    background = _background_from_input(inp)
    half = inp.shape[1] // 2
    choices = [int(color) for color in np.unique(inp) if int(color) != background]
    if not choices:
        raise AssertionError("figure color is not visible")
    return max(choices, key=lambda color: _largest_component(inp[:, :half] == color))


def _solve_from_visible_input(inp: np.ndarray) -> np.ndarray:
    figure = _figure_from_input(inp)
    half = inp.shape[1] // 2
    result = inp.copy()
    for row, column in np.argwhere(inp[:, :half] == figure):
        result[int(row), inp.shape[1] - 1 - int(column)] = figure
    return result


def _make_shape(rng: np.random.Generator, height: int, half: int, regime: str) -> np.ndarray:
    width = 2 * half
    for _ in range(30):
        top = int(rng.integers(1, max(2, height // 3 + 1)))
        bottom_margin = int(rng.integers(1, max(2, height // 3 + 1)))
        active = height - top - bottom_margin
        if active < 4:
            continue
        widths = []
        current = int(rng.integers(1, half + 1))
        for _row in range(active):
            if rng.random() < 0.28:
                current = int(rng.integers(1, half + 1))
            else:
                current = min(half, max(1, current + int(rng.integers(-2, 3))))
            widths.append(current)
        if rng.random() < 0.38:
            widths[int(rng.integers(active))] = half
        if max(widths) < 2:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for offset, row_width in enumerate(widths):
            row = top + offset
            mask[row, half - row_width:half + row_width] = True
        # Carving matched cells creates forks, center gaps, notches and enclosed holes.
        complexity = int(rng.choice(4, p=(0.25, 0.25, 0.27, 0.23)))
        if regime == "target_test" and rng.random() < 0.15:
            complexity = max(complexity, 2)
        requested = (0, 1, 2, 3)[complexity]
        for _carve in range(requested):
            for _attempt in range(30):
                row = int(rng.integers(top, top + active))
                row_width = widths[row - top]
                if row_width < 2:
                    continue
                kind = int(rng.integers(3))
                if kind == 0:
                    column = half - 1  # an axis notch or split row
                elif kind == 1 and row_width >= 3:
                    column = int(rng.integers(half - row_width + 1, half - 1))
                else:
                    column = half - row_width  # an outer notch
                if not mask[row, column]:
                    continue
                trial = mask.copy()
                trial[row, column] = False
                trial[row, width - 1 - column] = False
                if (
                    np.any(trial[row])
                    and _is_connected(trial)
                    and _is_connected(trial[:, :half])
                ):
                    mask = trial
                    break
        left_count = int(np.count_nonzero(mask[:, :half]))
        if left_count >= max(10, int(0.045 * height * width)):
            return mask
    # A bounded fallback still varies with the sampled dimensions.
    mask = np.zeros((height, width), dtype=bool)
    top, bottom = max(1, height // 5), min(height - 1, height - height // 5)
    for row in range(top, bottom):
        row_width = min(half, max(2, half // 2 + (row - top) % 3 - 1))
        mask[row, half - row_width:half + row_width] = True
    return mask


def _missing_cells(rng: np.random.Generator, mask: np.ndarray, center_required: bool) -> list[tuple[int, int]]:
    height, width = mask.shape
    half = width // 2
    points = [(int(r), int(c)) for r, c in np.argwhere(mask[:, half:])]
    points = [(r, c + half) for r, c in points]
    ratio = float(rng.uniform(0.10, 0.68))
    target = min(len(points) - 2, max(2, int(round(len(points) * ratio))))
    if rng.random() < 0.38:
        # A second damage regime peels cells without breaking the visible core.
        # It balances the bridge-erasure regime below; the completed shape stays
        # connected in both cases.
        order = points.copy()
        rng.shuffle(order)
        center_candidates = [point for point in order if point[1] == half]
        if center_candidates and (center_required or rng.random() < 0.58):
            order.remove(center_candidates[0])
            order.insert(0, center_candidates[0])
        connected_choice: set[tuple[int, int]] = set()
        for point in order:
            if len(connected_choice) >= target:
                break
            trial = connected_choice | {point}
            visible = mask.copy()
            for row, column in trial:
                visible[row, column] = False
            if _is_connected(visible):
                connected_choice = trial
        if len(connected_choice) == target and (
            not center_required or any(column == half for _, column in connected_choice)
        ):
            return sorted(connected_choice)
    chosen: set[tuple[int, int]] = set()
    center = [(r, c) for r, c in points if c == half]
    if center and (center_required or rng.random() < 0.58):
        chosen.add(center[int(rng.integers(len(center)))])
    # Sometimes erase the bridge to an outer right arm, leaving a visible remnant.
    if rng.random() < 0.32:
        rows = [int(r) for r in np.flatnonzero(mask[:, half:].any(axis=1))]
        rng.shuffle(rows)
        detached = False
        for row in rows:
            occupied = [c for c in range(half, width) if mask[row, c]]
            if len(occupied) < 3:
                continue
            for cut in range(1, len(occupied) - 1):
                trial = chosen | set((row, c) for c in occupied[:cut])
                if len(trial) > target:
                    continue
                visible = mask.copy()
                for r, c in trial:
                    visible[r, c] = False
                if not _is_connected(visible):
                    chosen = trial
                    detached = True
                    break
            if detached:
                break
    style = int(rng.integers(3))  # specks, connected erasures, or both
    cluster_goal = 0 if style == 0 else int(round(target * (0.75 if style == 1 else 0.45)))
    if cluster_goal > len(chosen):
        anchor = points[int(rng.integers(len(points)))]
        chosen.add(anchor)
    point_set = set(points)
    for _ in range(target * 3):
        if len(chosen) >= min(target, cluster_goal):
            break
        neighbors = set()
        for row, column in chosen:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                candidate = (row + dr, column + dc)
                if candidate in point_set and candidate not in chosen:
                    neighbors.add(candidate)
        if not neighbors:
            break
        candidates = sorted(neighbors)
        chosen.add(candidates[int(rng.integers(len(candidates)))])
    remaining = sorted(point_set - chosen)
    if len(chosen) < target:
        indices = rng.choice(len(remaining), size=target - len(chosen), replace=False)
        chosen.update(remaining[int(index)] for index in np.atleast_1d(indices))
    return sorted(chosen)


def _add_clutter(rng: np.random.Generator, out: np.ndarray, mask: np.ndarray,
                 background: int, distractors: list[int], visible_count: int) -> None:
    height, width = out.shape
    eligible = ~mask.copy()
    for row, column in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        eligible[row, column] = False
    available = [(int(r), int(c)) for r, c in np.argwhere(eligible)]
    if not available:
        return
    figure = int(out[mask][0])
    clutter_style = int(rng.choice(3, p=(0.24, 0.46, 0.30)))
    density_ranges = ((0.10, 0.26), (0.28, 0.65), (0.65, 1.05))
    low, high = density_ranges[clutter_style]
    budget = min(len(available), max(len(distractors) + 2,
                                    int(round(visible_count * float(rng.uniform(low, high))))))
    budget = min(budget, max(5, visible_count - 2))
    placed = 0
    # Each distractor is visible, and the remaining marks mix short runs and specks.
    rng.shuffle(available)
    for color, (row, column) in zip(distractors, available):
        out[row, column] = color
        placed += 1
    run_count = 0 if clutter_style == 0 else int(rng.integers(0, 4))
    for _ in range(run_count):
        if placed + 2 > budget:
            break
        color = distractors[int(rng.integers(len(distractors)))]
        row, column = available[int(rng.integers(len(available)))]
        dr, dc = ((0, 1), (1, 0))[int(rng.integers(2))]
        length = int(rng.integers(2, 5))
        segment = [(row + dr * i, column + dc * i) for i in range(length)]
        segment = [(r, c) for r, c in segment
                   if 0 <= r < height and 0 <= c < width and eligible[r, c] and out[r, c] == background]
        if len(segment) >= 2 and placed + len(segment) <= budget:
            for r, c in segment:
                out[r, c] = color
            placed += len(segment)
    if clutter_style != 0 and rng.random() < 0.38 and placed + 3 <= budget:
        row, column = available[int(rng.integers(len(available)))]
        patch = [(row + dr, column + dc) for dr in (0, 1) for dc in (0, 1)]
        patch = [(r, c) for r, c in patch
                 if 0 <= r < height and 0 <= c < width and eligible[r, c] and out[r, c] == background]
        if len(patch) >= 3 and placed + len(patch) <= budget:
            color = distractors[int(rng.integers(len(distractors)))]
            for r, c in patch:
                out[r, c] = color
            placed += len(patch)
    rng.shuffle(available)
    for row, column in available:
        if placed >= budget:
            break
        if out[row, column] == background:
            if clutter_style == 0 and any(
                0 <= row + dr < height and 0 <= column + dc < width
                and out[row + dr, column + dc] not in (background, figure)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ):
                continue
            out[row, column] = distractors[int(rng.integers(len(distractors)))]
            placed += 1


def _make_pair(rng: np.random.Generator, regime: str, mode: str,
               center_required: bool = False, forced_figure: int | None = None) -> dict:
    for _ in range(32):
        height = int(rng.integers(8, 25))
        half = int(rng.integers(4, 13))
        if rng.random() < 0.08:
            half = 15
        mask = _make_shape(rng, height, half, regime)
        if mask[:, half].sum() < 2:
            continue
        if rng.random() < 0.66:
            background = 0
        else:
            background = int(rng.integers(1, 10))
        if forced_figure is None:
            figure = int(rng.choice([c for c in range(10) if c != background]))
        else:
            figure = forced_figure
            if figure == background:
                background = int(rng.choice([c for c in range(10) if c != figure]))
        remaining_colors = [c for c in range(10) if c not in (background, figure)]
        number_distractors = int(rng.integers(1, 4))
        distractors = [int(c) for c in rng.choice(remaining_colors, size=number_distractors, replace=False)]
        missing = _missing_cells(rng, mask, center_required)
        if center_required and not any(c == half for _, c in missing):
            continue
        out = np.full(mask.shape, background, dtype=np.int8)
        out[mask] = figure
        visible_count = int(np.count_nonzero(mask)) - len(missing)
        _add_clutter(rng, out, mask, background, distractors, visible_count)
        inp = out.copy()
        if mode == "background":
            replacements = [background] * len(missing)
        elif mode == "distractor":
            replacements = [distractors[int(rng.integers(len(distractors)))] for _ in missing]
        else:
            replacements = [background] + [distractors[int(rng.integers(len(distractors)))]]
            replacements += [background if rng.random() < 0.5 else distractors[int(rng.integers(len(distractors)))]
                             for _ in range(len(missing) - 2)]
            rng.shuffle(replacements)
        for (row, column), old_color in zip(missing, replacements):
            inp[row, column] = old_color
        if _figure_from_input(inp) != figure:
            continue
        if any(_largest_component(inp[:, :half] == distractor) >=
               _largest_component(inp[:, :half] == figure) for distractor in distractors):
            continue
        return {"input": inp, "output": out}
    raise AssertionError("bounded pair construction failed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    # The shared convention is the source rule: a vertical midline and an intact
    # left template. Colors and geometry carry no hidden puzzle-wide meanings.
    modes = [str(rng.choice(("background", "distractor", "mixed"))) for _ in range(train_count)]
    if not any(mode in ("background", "mixed") for mode in modes):
        modes[int(rng.integers(train_count))] = "mixed"
    if not any(mode in ("distractor", "mixed") for mode in modes):
        modes[int(rng.integers(train_count))] = "mixed"
    center_index = int(rng.integers(train_count))
    for index in range(train_count):
        for _ in range(12):
            pair = _make_pair(rng, "source_train", modes[index], index == center_index)
            key = _pair_key(pair)
            if key not in seen and not _is_source_pair(pair["input"], pair["output"]):
                puzzle["train"].append(pair)
                seen.add(key)
                break
        else:
            raise AssertionError("could not make distinct training pair")
    train_clutter = sorted({int(c) for pair in puzzle["train"]
                            for c in np.unique(pair["input"])
                            if int(c) not in (_figure_from_input(pair["input"]),
                                               _background_from_input(pair["input"]))})
    for _index in range(test_count):
        forced = None
        if train_clutter and rng.random() < 0.56:
            forced = train_clutter[int(rng.integers(len(train_clutter)))]
        mode = str(rng.choice(("background", "distractor", "mixed")))
        for _ in range(12):
            pair = _make_pair(rng, "target_test", mode, False, forced)
            key = _pair_key(pair)
            if key not in seen and not _is_source_pair(pair["input"], pair["output"]):
                puzzle["test"].append(pair)
                seen.add(key)
                break
        else:
            raise AssertionError("could not make distinct test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "split lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    training_old_colors = set()
    training_center_repair = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "array grid")
                _require(np.issubdtype(grid.dtype, np.integer), "integer grid")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "ARC colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and inp.shape[1] % 2 == 0, "same even dimensions")
            _require(_grid_key(inp) not in seen_inputs and _pair_key(pair) not in seen_pairs, "unique pair")
            seen_inputs.add(_grid_key(inp))
            seen_pairs.add(_pair_key(pair))
            _require(not _is_source_pair(inp, out), "source pair copied")
            _require(set(np.unique(inp)) == set(np.unique(out)), "palette changed")
            half = inp.shape[1] // 2
            changed = inp != out
            _require(int(changed.sum()) >= 2, "no meaningful repair")
            _require(not np.any(changed[:, :half]), "intact left half")
            _require(np.all(inp[changed] != out[changed]), "changed cells")
            new_colors = np.unique(out[changed])
            _require(len(new_colors) == 1, "repairs have one figure color")
            figure = int(new_colors[0])
            _require(_background_from_input(inp) != figure, "background mistaken for figure")
            figure_mask = out == figure
            _require(np.array_equal(figure_mask, np.fliplr(figure_mask)), "figure not symmetric")
            _require(_is_connected(figure_mask) and _is_connected(figure_mask[:, :half]), "figure not connected")
            _require(np.all(out[:, :half] == inp[:, :half]), "left half altered")
            _require(np.array_equal(_solve_from_visible_input(inp), out), "visible rule fails")
            _require(_figure_from_input(inp) == figure, "figure is not identifiable")
            figure_component = _largest_component(inp[:, :half] == figure)
            _require(all(_largest_component(inp[:, :half] == int(color)) < figure_component
                         for color in np.unique(inp) if int(color) not in (figure, _background_from_input(inp))),
                     "a distractor competes with the figure")
            _require(sum(inp[0, c] == _background_from_input(inp) for c in (0, -1)) +
                     sum(inp[-1, c] == _background_from_input(inp) for c in (0, -1)) >= 3,
                     "background is unclear at corners")
            _require(np.any(inp[:, :half] == figure), "no left template")
            if split == "train":
                bg = _background_from_input(inp)
                training_old_colors.update("background" if int(color) == bg else "distractor"
                                           for color in np.unique(inp[changed]))
                training_center_repair |= bool(np.any(changed[:, half]))
    _require(training_old_colors == {"background", "distractor"}, "training omits a replacement kind")
    _require(training_center_repair, "training omits center-adjacent repair")
    # This prediction uses only training input/output pairs and each test input.
    for pair in puzzle["test"]:
        _require(np.array_equal(_solve_from_visible_input(pair["input"]), pair["output"]),
                 "test not recoverable from visible rule")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name]) for a, b in zip(first[split], second[split])
                   for name in ("input", "output")) for split in ("train", "test"))


def _expect_bad_counts(train: int | None, test: int | None) -> None:
    try:
        generate_puzzle(501, train, test)
    except ValueError:
        return
    raise AssertionError("invalid pair counts accepted")


def self_test() -> None:
    for train, test in ((None, None), (2, 1), (10, 1), (10, 4),
                        (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(314159, train, test)
        second = generate_puzzle(314159, train, test)
        _require(_puzzles_equal(first, second), "not deterministic")
        if train is not None:
            _require(len(first["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(first["test"]) == test, "test count not honored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        _expect_bad_counts(train, test)
    train_counts, test_counts = set(), set()
    sizes, shapes, colors, ratios = set(), set(), set(), set()
    center_repairs = disconnected = split_rows = full_rows = 0
    background_only = distractor_only = mixed = 0
    split_stats = {split: {name: 0 for name in (
        "full", "margin", "split", "center", "disconnected", "connected",
        "light", "severe", "nonzero_background", "sparse", "streak",
        "three_colors", "five_colors", "combined")}
        for split in ("train", "test")}
    varied_train_palettes = varied_train_backgrounds = varied_train_figures = 0
    reused_clutter_role = 0
    total = 0
    keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(7000 + seed)
        keys.add(tuple(_pair_key(pair) for split in ("train", "test") for pair in puzzle[split]))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        training_palettes = set()
        training_backgrounds = set()
        training_figures = set()
        training_clutter = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                inp, out = pair["input"], pair["output"]
                half = inp.shape[1] // 2
                figure = _figure_from_input(inp)
                mask = out == figure
                changed = inp != out
                sizes.add(inp.shape)
                shapes.add((inp.shape, tuple(tuple(bool(x) for x in row) for row in mask[:, :half])))
                colors.add((_background_from_input(inp), figure, tuple(sorted(int(x) for x in np.unique(inp)))))
                ratios.add(round(float(changed.sum() / mask[:, half:].sum()), 1))
                center_repairs += bool(np.any(changed[:, half]))
                disconnected += not _is_connected(inp == figure)
                split_rows += any(np.any(row) and np.any(~row) and sum(row) >= 2 and
                                  np.any(row[np.flatnonzero(row).min():np.flatnonzero(row).max() + 1] == 0)
                                  for row in mask)
                full_rows += bool(np.any(np.all(mask, axis=1)))
                old = set(int(x) for x in np.unique(inp[changed]))
                bg = _background_from_input(inp)
                background_only += old == {bg}
                distractor_only += bg not in old
                mixed += bg in old and len(old) > 1
                stats = split_stats[split]
                stats["full" if np.any(np.all(mask, axis=1)) else "margin"] += 1
                has_split = any(len(xs := np.flatnonzero(row)) > 1 and
                                xs[-1] - xs[0] + 1 > len(xs) for row in mask)
                stats["split"] += has_split
                stats["center"] += bool(np.any(changed[:, half]))
                stats["disconnected" if not _is_connected(inp == figure) else "connected"] += 1
                repair_ratio = float(changed.sum() / mask[:, half:].sum())
                stats["light"] += repair_ratio <= 0.25
                stats["severe"] += repair_ratio >= 0.50
                stats["nonzero_background"] += bg != 0
                palette = set(int(x) for x in np.unique(inp))
                stats["three_colors"] += len(palette) == 3
                stats["five_colors"] += len(palette) == 5
                clutter_run = any(_largest_component(out == color) >= 2
                                  for color in palette - {bg, figure})
                stats["streak" if clutter_run else "sparse"] += 1
                stats["combined"] += has_split and repair_ratio >= 0.50 and bg != 0
                if split == "train":
                    training_palettes.add(tuple(sorted(palette)))
                    training_backgrounds.add(bg)
                    training_figures.add(figure)
                    training_clutter.update(palette - {bg, figure})
                else:
                    reused_clutter_role += figure in training_clutter
        varied_train_palettes += len(training_palettes) > 1
        varied_train_backgrounds += len(training_backgrounds) > 1
        varied_train_figures += len(training_figures) > 1
    _require(len(keys) == 42, "seed diversity")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(sizes) >= 30 and len(shapes) >= total // 2 and len(colors) >= 40, "structural diversity")
    _require(len(ratios) >= 6 and center_repairs > total // 4, "damage diversity")
    _require(split_rows > 0 and full_rows > 0 and disconnected > 0, "shape and remnant diversity")
    _require(background_only > 0 and distractor_only > 0 and mixed > 0, "replacement diversity")
    _require(varied_train_palettes >= 35 and varied_train_backgrounds >= 25
             and varied_train_figures >= 30, "within-puzzle color diversity")
    _require(reused_clutter_role > 0, "clutter to figure role changes absent")
    for split, stats in split_stats.items():
        _require(all(stats[name] > 0 for name in stats),
                 f"{split} omits a review variation")
