import numpy as np


_COLORS = np.arange(1, 10, dtype=np.int8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _pair_key(inp: np.ndarray, out: np.ndarray) -> tuple:
    return (_grid_key(inp), _grid_key(out))


def _grid_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIR_ROWS = (
    (
        (
            "0000000000",
            "0110550440",
            "0110330440",
            "0330550480",
            "0000000000",
            "0220710990",
            "0220770190",
            "0220710990",
            "0000000000",
        ),
        (
            "00000",
            "01540",
            "02790",
            "00000",
        ),
    ),
    (
        (
            "0000000000",
            "0330660970",
            "0830630970",
            "0380360770",
            "0000000000",
            "0330220610",
            "0230550110",
            "0230550160",
            "0000000000",
        ),
        (
            "00000",
            "03670",
            "03510",
            "00000",
        ),
    ),
    (
        (
            "0000000000",
            "0350840770",
            "0530880760",
            "0330840670",
            "0000000000",
            "0330220130",
            "0430220110",
            "0330120130",
            "0000000000",
        ),
        (
            "00000",
            "03870",
            "03210",
            "00000",
        ),
    ),
    (
        (
            "0000000000",
            "0110330440",
            "0310830440",
            "0110380840",
            "0000000000",
            "0220350220",
            "0660550220",
            "0220530220",
            "0000000000",
        ),
        (
            "00000",
            "01340",
            "02520",
            "00000",
        ),
    ),
)


_SOURCE_PAIR_KEYS = frozenset(
    _pair_key(_grid_from_rows(inp), _grid_from_rows(out))
    for inp, out in _SOURCE_PAIR_ROWS
)


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        extra_range = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % extra_range
        )
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError(
            "counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return train_count, test_count


def _geometry(rng: np.random.Generator, training_example: bool) -> tuple:
    kind = int(rng.choice(3, p=[0.13, 0.13, 0.74]))
    if kind == 0:
        heights, widths = [3, 3], [2, 2, 2]
    elif kind == 1:
        heights, widths = [2, 2, 2], [3, 3]
    else:
        rows = int(rng.integers(2, 5))
        cols = int(rng.integers(2, 6))
        heights = [int(x) for x in rng.integers(2, 6, size=rows)]
        widths = [int(x) for x in rng.integers(2, 6, size=cols)]
        while sum(widths) + cols + 1 > 30:
            col = int(rng.integers(0, cols))
            if widths[col] > 2:
                widths[col] -= 1
    if training_example:
        if max(heights) < 3:
            heights[int(rng.integers(len(heights)))] = 3
        if max(widths) < 3:
            widths[int(rng.integers(len(widths)))] = 3
    return tuple(heights), tuple(widths)


def _representatives(
    rng: np.random.Generator, background: int, rows: int, cols: int,
    first_training: bool,
) -> np.ndarray:
    available = np.asarray([v for v in range(10) if v != background], dtype=np.int8)
    size = rows * cols
    styles = ["adjacent", "separated", "cross"]
    if size <= 8:
        styles.append("none")
    style = str(rng.choice(styles))
    if style == "none":
        values = rng.choice(available, size=size, replace=False)
        if first_training and background != 0 and 0 not in values:
            values[0] = 0
        return values.reshape(rows, cols).astype(np.int8)

    palette_size = min(8, max(3, size // 2 + 2))
    palette = rng.choice(available, size=palette_size, replace=False)
    if first_training and background != 0 and 0 not in palette:
        palette[0] = 0
    values = rng.choice(palette, size=(rows, cols)).astype(np.int8)
    if style == "adjacent":
        row = int(rng.integers(rows))
        left = int(rng.integers(cols - 1))
        first, second = (row, left), (row, left + 1)
    elif style == "separated":
        if cols < 3:
            row = int(rng.integers(rows))
            first, second = (row, 0), (row, 1)
        else:
            row = int(rng.integers(rows))
            first, second = (row, 0), (row, int(rng.integers(2, cols)))
    else:
        first = (0, int(rng.integers(cols)))
        second = (int(rng.integers(1, rows)), int(rng.integers(cols)))
    values[second] = values[first]
    if first_training and background != 0:
        values[0, 0] = 0
    if len(np.unique(values)) == 1:
        values[-1, -1] = next(int(v) for v in available if v != values[0, 0])
    return values


def _minority_positions(
    rng: np.random.Generator, height: int, width: int, count: int,
) -> np.ndarray:
    style = str(rng.choice(["scattered", "cluster", "edge", "stripe"]))
    cells = [(r, c) for r in range(height) for c in range(width)]
    order = rng.permutation(len(cells))
    if style == "cluster":
        anchor_r, anchor_c = cells[int(rng.integers(len(cells)))]
        ordered = sorted(order, key=lambda i: abs(cells[i][0] - anchor_r) + abs(cells[i][1] - anchor_c))
    elif style == "edge":
        ordered = sorted(order, key=lambda i: int(not (
            cells[i][0] in (0, height - 1) or cells[i][1] in (0, width - 1)
        )))
    elif style == "stripe":
        horizontal = bool(rng.integers(2))
        index = int(rng.integers(height if horizontal else width))
        ordered = sorted(order, key=lambda i: int(
            cells[i][0 if horizontal else 1] != index
        ))
    else:
        ordered = list(order)
    return np.asarray([cells[int(i)] for i in ordered[:count]], dtype=np.int16)


def _noise_counts(
    rng: np.random.Generator, area: int, force_plurality: bool,
    regime: str,
) -> tuple[int, ...]:
    if force_plurality:
        mode = (area + 4) // 3
        other = area - mode
        return (mode, (other + 1) // 2, other // 2)
    two_colors = area >= 6 and bool(rng.random() < (
        0.33 if regime == "target_test" else 0.25
    ))
    if two_colors:
        candidates = [
            (mode, first, area - mode - first)
            for mode in range(2, area - 1)
            for first in range(1, area - mode)
            if mode > first and mode > area - mode - first
        ]
        return candidates[int(rng.integers(len(candidates)))]
    maximum_noise = (area - 1) // 2
    difficulty = str(rng.choice(
        ["easy", "moderate", "near"],
        p=[0.34, 0.36, 0.30] if regime == "source_train"
        else [0.25, 0.35, 0.40],
    ))
    if difficulty == "easy":
        noise = 1
    elif difficulty == "moderate":
        noise = max(1, maximum_noise // 2)
    else:
        noise = maximum_noise
    return (area - noise, noise)


def generate_pair(
    rng: np.random.Generator, regime: str, puzzle_parameters: dict,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    background = int(puzzle_parameters["background"])
    first_training = regime == "source_train" and pair_index == 0
    heights, widths = puzzle_parameters["geometry"]
    rows, cols = len(heights), len(widths)
    modes = _representatives(rng, background, rows, cols, first_training)
    output = np.full((rows + 2, cols + 2), background, dtype=np.int8)
    output[1:-1, 1:-1] = modes
    inp = np.full(
        (sum(heights) + rows + 1, sum(widths) + cols + 1),
        background, dtype=np.int8,
    )
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    special = max(positions, key=lambda p: heights[p[0]] * widths[p[1]])
    if first_training:
        eligible = [p for p in positions if heights[p[0]] * widths[p[1]] >= 9]
        special = eligible[int(rng.integers(len(eligible)))]

    maximum_uniform = max(1, min(4, len(positions) // 2))
    if first_training:
        uniform_count = int(rng.integers(1, maximum_uniform + 1))
    elif regime == "source_train" and pair_index == 1:
        uniform_count = 0
    else:
        uniform_count = int(rng.integers(0, maximum_uniform + 1))
    eligible_uniform = [p for p in positions if p != special or not first_training]
    chosen_uniform = rng.choice(len(eligible_uniform), size=uniform_count, replace=False)
    uniforms = {eligible_uniform[int(i)] for i in chosen_uniform}

    modal_colors = set(int(x) for x in modes.flat)
    input_only = [v for v in range(10) if v != background and v not in modal_colors]
    allow_external_noise = first_training or bool(rng.random() < 0.65)
    row_starts = [1]
    col_starts = [1]
    for h in heights[:-1]:
        row_starts.append(row_starts[-1] + h + 1)
    for w in widths[:-1]:
        col_starts.append(col_starts[-1] + w + 1)

    for row, col in positions:
        height, width = heights[row], widths[col]
        dominant = int(modes[row, col])
        patch = np.full((height, width), dominant, dtype=np.int8)
        if (row, col) not in uniforms:
            special_patch = first_training and (row, col) == special
            counts = _noise_counts(rng, height * width, special_patch, regime)
            other_modes = sorted(modal_colors - {dominant})
            if special_patch:
                minorities = [int(rng.choice(input_only)), int(rng.choice(other_modes))]
            else:
                internal = sorted(modal_colors - {dominant})
                choices = ([v for v in range(10) if v not in (background, dominant)]
                           if allow_external_noise else internal)
                if len(counts) == 3:
                    if len(choices) < 2:
                        choices = [v for v in range(10) if v not in (background, dominant)]
                    minorities = [int(x) for x in rng.choice(choices, size=2, replace=False)]
                else:
                    source = (input_only if allow_external_noise and input_only
                              and rng.random() < 0.45 else choices)
                    minorities = [int(rng.choice(source))]
            noise_total = sum(counts[1:])
            selected = _minority_positions(rng, height, width, noise_total)
            offset = 0
            for color, count in zip(minorities, counts[1:]):
                for r, c in selected[offset:offset + count]:
                    patch[int(r), int(c)] = color
                offset += count
        rs, cs = row_starts[row], col_starts[col]
        inp[rs:rs + height, cs:cs + width] = patch
    return inp, output


def _decode_input(inp: np.ndarray) -> tuple[np.ndarray, dict]:
    background = int(inp[0, 0])
    _require(np.all(inp[0] == background) and np.all(inp[-1] == background),
             "outer horizontal background border is incomplete")
    _require(np.all(inp[:, 0] == background) and np.all(inp[:, -1] == background),
             "outer vertical background border is incomplete")
    gutter_rows = np.flatnonzero(np.all(inp == background, axis=1))
    gutter_cols = np.flatnonzero(np.all(inp == background, axis=0))
    heights = tuple(int(v) for v in np.diff(gutter_rows) - 1)
    widths = tuple(int(v) for v in np.diff(gutter_cols) - 1)
    rows, cols = len(heights), len(widths)
    _require(gutter_rows[0] == 0 and gutter_rows[-1] == inp.shape[0] - 1,
             "bad horizontal gutters")
    _require(gutter_cols[0] == 0 and gutter_cols[-1] == inp.shape[1] - 1,
             "bad vertical gutters")
    _require(2 <= rows <= 4 and 2 <= cols <= 5, "bad patch array")
    _require(all(2 <= v <= 5 for v in heights + widths), "bad patch dimensions")
    modes = np.empty((rows, cols), dtype=np.int8)
    uniform_count = 0
    noisy_count = 0
    plurality_count = 0
    patch_color_counts = []
    minority_set = set()
    margin_one = 0
    margin_many = 0
    noise_geometry = {name: 0 for name in ("edge", "stripe", "clustered", "scattered")}
    for row in range(rows):
        for col in range(cols):
            patch = inp[
                gutter_rows[row] + 1:gutter_rows[row + 1],
                gutter_cols[col] + 1:gutter_cols[col + 1],
            ]
            _require(np.all(patch != background), "background inside a patch")
            values, counts = np.unique(patch, return_counts=True)
            _require(1 <= len(values) <= 3, "too many patch colors")
            best = int(np.argmax(counts))
            _require(int(np.count_nonzero(counts == counts[best])) == 1,
                     "tied patch mode")
            modes[row, col] = values[best]
            patch_color_counts.append(len(values))
            if len(values) == 1:
                uniform_count += 1
            else:
                noisy_count += 1
                plurality_count += int(2 * counts[best] < patch.size)
                minority_set.update(int(v) for i, v in enumerate(values) if i != best)
                margin = int(counts[best] - max(
                    count for i, count in enumerate(counts) if i != best
                ))
                margin_one += int(margin == 1)
                margin_many += int(margin >= 2)
                noise = np.argwhere(patch != values[best])
                if len(noise) >= 2:
                    edge = np.all(
                        (noise[:, 0] == 0) | (noise[:, 0] == patch.shape[0] - 1)
                        | (noise[:, 1] == 0) | (noise[:, 1] == patch.shape[1] - 1)
                    )
                    noise_geometry["edge"] += int(edge)
                    noise_geometry["stripe"] += int(
                        len(np.unique(noise[:, 0])) == 1
                        or len(np.unique(noise[:, 1])) == 1
                    )
                    remaining = {tuple(int(v) for v in point) for point in noise}
                    queue = [remaining.pop()]
                    while queue:
                        y, x = queue.pop()
                        for neighbor in ((y - 1, x), (y + 1, x),
                                         (y, x - 1), (y, x + 1)):
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                queue.append(neighbor)
                    noise_geometry["clustered"] += int(not remaining)
                    noise_geometry["scattered"] += int(bool(remaining))
    output = np.full((rows + 2, cols + 2), background, dtype=np.int8)
    output[1:-1, 1:-1] = modes
    modal_set = {int(v) for v in modes.flat}
    flat = [int(v) for v in modes.flat]
    adjacent = any(int(modes[r, c]) == int(modes[r, c + 1])
                   for r in range(rows) for c in range(cols - 1))
    separated = any(int(modes[r, c]) == int(modes[r, d])
                    for r in range(rows) for c in range(cols)
                    for d in range(c + 2, cols))
    cross = any(int(modes[r, c]) == int(modes[s, d])
                for r in range(rows) for s in range(r + 1, rows)
                for c in range(cols) for d in range(cols))
    return output, {
        "background": background, "heights": heights, "widths": widths,
        "rows": rows, "cols": cols, "modes": modes,
        "uniform": uniform_count, "noisy": noisy_count,
        "plurality": plurality_count, "patch_colors": patch_color_counts,
        "margin_one": margin_one, "margin_many": margin_many,
        "noise_geometry": noise_geometry,
        "input_only": bool(minority_set - modal_set),
        "role_overlap": bool(minority_set & modal_set),
        "adjacent": adjacent, "separated": separated, "cross": cross,
        "no_repeats": len(set(flat)) == len(flat),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "invalid pair")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid too large")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
            input_key, pair_key = _grid_key(inp), _pair_key(inp, out)
            _require(input_key not in seen_inputs, "duplicate input")
            _require(pair_key not in seen_pairs, "duplicate pair")
            _require(pair_key not in _SOURCE_PAIR_KEYS, "source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            decoded, analysis = _decode_input(inp)
            _require(np.array_equal(decoded, out), "output is not the local modes")
            analyses[split].append(analysis)
    training = analyses["train"]
    _require(len({a["background"] for a in training}) >= 2,
             "training does not show pair-specific backgrounds")
    _require(len({a["rows"] for a in training}) >= 2,
             "training does not show variable array row counts")
    _require(len({a["cols"] for a in training}) >= 2,
             "training does not show variable array column counts")
    _require(len({(a["heights"], a["widths"]) for a in training}) >= 2,
             "training does not show variable geometry")
    _require(any(a["uniform"] for a in training), "no uniform training patch")
    _require(any(a["noisy"] for a in training), "no noisy training patch")
    _require(any(a["plurality"] for a in training), "no plurality training patch")
    _require(any(a["input_only"] for a in training), "no input-only training color")
    _require(any(a["role_overlap"] for a in training), "no local role overlap in training")


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_geometry = None
    first_background = int(rng.integers(10))
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            accepted = False
            for _ in range(128):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                background = first_background if split == "train" and index == 0 else (
                    int(pair_rng.integers(10)) if split != "train" or index != 1
                    else int(pair_rng.choice([v for v in range(10) if v != first_background]))
                )
                geometry = _geometry(pair_rng, split == "train" and index == 0)
                if (split == "train" and index == 1
                        and (len(geometry[0]) == len(first_geometry[0])
                             or len(geometry[1]) == len(first_geometry[1]))):
                    continue
                parameters = {"background": background, "geometry": geometry}
                inp, out = generate_pair(pair_rng, regime, parameters, index)
                if _grid_key(inp) in seen_inputs or _pair_key(inp, out) in _SOURCE_PAIR_KEYS:
                    continue
                if split == "train" and index == 0:
                    first_geometry = geometry
                puzzle[split].append({"input": inp, "output": out})
                seen_inputs.add(_grid_key(inp))
                accepted = True
                break
            if not accepted:
                raise ValueError("could not construct distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    larger_tests = 0
    transposed_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training = [_decode_input(pair["input"])[1] for pair in puzzle["train"]]
        testing = [_decode_input(pair["input"])[1] for pair in puzzle["test"]]
        _require(len({a["background"] for a in training}) >= 2,
                 "training backgrounds do not vary within a puzzle")
        _require(len({a["rows"] for a in training}) >= 2
                 and len({a["cols"] for a in training}) >= 2,
                 "training array dimensions do not vary within a puzzle")
        larger_tests += int(any(
            b["rows"] * b["cols"] > min(a["rows"] * a["cols"] for a in training)
            for b in testing
        ))
        transposed_tests += int(any(
            b["rows"] == a["cols"] and b["cols"] == a["rows"]
            for a in training for b in testing
        ))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                prediction, _ = _decode_input(pair["input"].copy())
                _require(np.array_equal(prediction, pair["output"]),
                         "input-only prediction failed")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts collapsed")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts collapsed")
    _require(len({_grid_key(p["train"][0]["input"]) for p in puzzles}) >= 40,
             "seed diversity collapsed")
    _require(larger_tests >= 10 and transposed_tests >= 10,
             "test geometry extensions collapsed")
    analyses = {split: [_decode_input(pair["input"])[1] for puzzle in puzzles
                        for pair in puzzle[split]] for split in ("train", "test")}
    all_pairs = analyses["train"] + analyses["test"]
    _require(len({a["background"] for a in all_pairs}) >= 8,
             "background colors collapsed")
    _require(any(a["background"] != 0 and 0 in a["modes"] for a in all_pairs),
             "zero is never an ordinary patch color")
    _require(len({(a["heights"], a["widths"]) for a in all_pairs}) >= 50,
             "geometry diversity collapsed")
    _require(any(a["rows"] == 2 and a["cols"] == 3
                 and a["heights"] == (3, 3) and a["widths"] == (2, 2, 2)
                 for a in all_pairs), "source geometry absent")
    _require(any(a["rows"] == 3 and a["cols"] == 2 for a in all_pairs),
             "transposed layout absent")
    _require(any(len(set(a["heights"])) > 1 or len(set(a["widths"])) > 1
                 for a in all_pairs), "mixed patch dimensions absent")
    _require({v for a in all_pairs for v in a["heights"]} == {2, 3, 4, 5},
             "patch height range collapsed")
    _require({v for a in all_pairs for v in a["widths"]} == {2, 3, 4, 5},
             "patch width range collapsed")
    for split in ("train", "test"):
        part = analyses[split]
        _require(all(any(a["uniform"] == k for a in part)
                     for k in (0, 1)), f"{split} uniform regimes absent")
        _require(any(a["uniform"] >= 2 for a in part),
                 f"{split} several-uniform regime absent")
        for feature in ("input_only", "role_overlap", "adjacent", "separated",
                        "cross", "no_repeats", "plurality"):
            _require(any(a[feature] for a in part), f"{split} lacks {feature}")
        _require(any(not a["input_only"] for a in part),
                 f"{split} lacks a representative-only noise palette")
        _require(any(3 in a["patch_colors"] for a in part),
                 f"{split} lacks three-color patches")
        _require(any(a["margin_one"] for a in part)
                 and any(a["margin_many"] for a in part),
                 f"{split} lacks corruption difficulty range")
        for shape in ("edge", "stripe", "clustered", "scattered"):
            _require(any(a["noise_geometry"][shape] for a in part),
                     f"{split} lacks {shape} minority placement")
    for seed, counts in ((314159, (7, 3)), (5000, (2, 1)),
                         (5001, (10, 1)), (5002, (10, 4))):
        first = generate_puzzle(seed, *counts)
        second = generate_puzzle(seed, *counts)
        _require(all(np.array_equal(a[name], b[name])
                     for split in ("train", "test")
                     for a, b in zip(first[split], second[split])
                     for name in ("input", "output")), "nondeterministic puzzle")
        _require((len(first["train"]), len(first["test"])) == counts,
                 "explicit counts ignored")
    for seed, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                   {"num_train": 2}, {"num_train": 10}), 6000):
        puzzle = generate_puzzle(seed, **kwargs)
        validate_generated_puzzle(puzzle)
        for name, value in kwargs.items():
            _require(len(puzzle[name[4:]]) == value, "partial count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_test": 1.5}):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
