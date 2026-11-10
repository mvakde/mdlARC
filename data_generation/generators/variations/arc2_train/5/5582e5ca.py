import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    ((3, 3), (6, 8, 9, 1, 8, 1, 9, 4, 9)),
    ((3, 3), (4, 4, 8, 6, 4, 3, 6, 3, 0)),
    ((3, 3), (4, 6, 9, 6, 4, 1, 8, 8, 6)),
    ((3, 3), (8, 8, 6, 4, 6, 9, 8, 3, 0)),
}

_SOURCE_PROFILE = (3, 2, 2, 1, 1)
_SHAPES = ((2, 3), (3, 3), (3, 4), (4, 4))
_PROFILES = {
    (2, 3): (
        (2, 1, 1, 1, 1), (3, 2, 1), (3, 1, 1, 1),
        (4, 2), (4, 1, 1), (5, 1),
    ),
    (3, 3): (
        _SOURCE_PROFILE, (3, 2, 1, 1, 1, 1),
        (2, 1, 1, 1, 1, 1, 1, 1), (4, 3, 2),
        (4, 2, 2, 1), (5, 4), (5, 2, 1, 1),
        (6, 2, 1),
    ),
    (3, 4): (
        (3, 2, 2, 1, 1, 1, 1, 1),
        (4, 3, 2, 2, 1), (4, 2, 2, 2, 1, 1),
        (5, 4, 3), (5, 3, 2, 1, 1), (6, 4, 2),
        (7, 3, 2), (8, 4),
    ),
    (4, 4): (
        (4, 3, 2, 2, 2, 1, 1, 1),
        (5, 4, 3, 2, 2), (5, 3, 2, 2, 2, 1, 1),
        (6, 4, 3, 2, 1), (6, 3, 2, 2, 1, 1, 1),
        (7, 5, 4), (8, 5, 3), (9, 5, 2),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _content_signature(grid: np.ndarray) -> tuple:
    return tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.ravel())


def _coerce_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(
    rng: np.random.Generator, num_train: int | None, num_test: int | None
) -> tuple[int, int]:
    train = _coerce_count(num_train, "num_train", 2, 10)
    test = _coerce_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None:
        choices = np.arange(max(2, (test or 0) + 1), 11)
        weights = np.asarray((28, 29, 22, 10, 5, 3, 2, 1, 1), dtype=float)
        selected = weights[choices - 2]
        train = int(rng.choice(choices, p=selected / selected.sum()))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _derive_pair_rngs(rng: np.random.Generator, count: int) -> list[np.random.Generator]:
    entropy = rng.integers(0, 2**32, size=8, dtype=np.uint32)
    return [np.random.default_rng(s) for s in np.random.SeedSequence(entropy).spawn(count)]


def _unique_mode(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    winners = colors[counts == counts.max()]
    _require(len(winners) == 1, "Input needs a unique most frequent color")
    return int(winners[0])


def _solve_visible_rule(grid: np.ndarray) -> np.ndarray:
    return np.full(grid.shape, _unique_mode(grid), dtype=np.int8)


def _sample_shape(rng: np.random.Generator, test: bool = False) -> tuple[int, int]:
    weights = (0.23, 0.35, 0.27, 0.15) if test else (0.17, 0.45, 0.25, 0.13)
    return _SHAPES[int(rng.choice(4, p=weights))]


def _sample_profile(
    rng: np.random.Generator, shape: tuple[int, int], nonthree: bool = False
) -> tuple[int, ...]:
    choices = [p for p in _PROFILES[shape] if not nonthree or p[0] != 3]
    return choices[int(rng.integers(len(choices)))]


def _adjacent_edges(grid: np.ndarray, color: int) -> int:
    mask = grid == color
    return int(np.count_nonzero(mask[:, :-1] & mask[:, 1:])) + int(
        np.count_nonzero(mask[:-1, :] & mask[1:, :])
    )


def _arrange(
    rng: np.random.Generator,
    values: np.ndarray,
    shape: tuple[int, int],
    mode: int,
    runner: int,
    style: str,
) -> np.ndarray:
    if style == "line":
        grid = np.empty(shape, dtype=np.int8)
        positions = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
        if shape[1] >= 3 and (shape[0] < 3 or rng.random() < 0.5):
            row = int(rng.integers(shape[0]))
            line = {(row, c) for c in range(3)}
        else:
            col = int(rng.integers(shape[1]))
            line = {(r, col) for r in range(3)}
        remaining = values[values != mode].copy()
        rng.shuffle(remaining)
        for (r, c), value in zip((p for p in positions if p not in line), remaining):
            grid[r, c] = value
        for r, c in line:
            grid[r, c] = mode
        return grid
    if style == "dispersed" and np.count_nonzero(values == runner) >= 2:
        best = None
        best_score = -10**9
        for _ in range(64):
            candidate = rng.permutation(values).reshape(shape)
            score = 2 * _adjacent_edges(candidate, runner) - _adjacent_edges(candidate, mode)
            if score > best_score:
                best, best_score = candidate, score
        return best
    return rng.permutation(values).reshape(shape)


def _make_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    profile: tuple[int, ...],
    mode: int,
    allowed_colors: tuple[int, ...],
    mandatory_distractor: int | None,
    style: str,
    seen_inputs: set,
) -> dict:
    available = [c for c in allowed_colors if c != mode and c != mandatory_distractor]
    needed = len(profile) - 1 - int(mandatory_distractor is not None)
    _require(needed <= len(available), "Insufficient colors for profile")
    for _ in range(128):
        sampled = [int(c) for c in rng.choice(available, size=needed, replace=False)]
        distractors = ([mandatory_distractor] if mandatory_distractor is not None else []) + sampled
        rng.shuffle(distractors)
        values = np.asarray(
            [mode] * profile[0]
            + [color for color, count in zip(distractors, profile[1:]) for _ in range(count)],
            dtype=np.int8,
        )
        grid = _arrange(rng, values, shape, mode, distractors[0], style)
        signature = _content_signature(grid)
        if signature in seen_inputs or signature in _SOURCE_INPUT_SIGNATURES:
            continue
        seen_inputs.add(signature)
        return {"input": grid, "output": _solve_visible_rule(grid)}
    # A deterministic fallback avoids depending on a lucky final rejection draw.
    for start in range(len(available)):
        sampled = [available[(start + j) % len(available)] for j in range(needed)]
        distractors = ([mandatory_distractor] if mandatory_distractor is not None else []) + sampled
        values = np.asarray(
            [mode] * profile[0]
            + [color for color, count in zip(distractors, profile[1:]) for _ in range(count)],
            dtype=np.int8,
        )
        for reverse in (False, True):
            ordered = values[::-1] if reverse else values
            for shift in range(values.size):
                grid = np.roll(ordered, shift).reshape(shape)
                signature = _content_signature(grid)
                if signature not in seen_inputs and signature not in _SOURCE_INPUT_SIGNATURES:
                    seen_inputs.add(signature)
                    return {"input": grid, "output": _solve_visible_rule(grid)}
    raise RuntimeError("Could not construct a distinct input")


def _pick_mode(
    rng: np.random.Generator, candidates: list[int], used: set[int]
) -> int:
    fresh = [c for c in candidates if c not in used]
    pool = fresh if fresh else candidates
    return int(rng.choice(pool))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # Puzzle-level plans affect which color roles and sizes are demonstrated.
    # They never change the rule used for any pair.
    transition = ("distractor", "reuse", "new")[int(rng.integers(3))]
    shuffled_colors = [int(c) for c in rng.permutation(10)]
    reserved_new = shuffled_colors[0] if transition == "new" else None
    train_colors = tuple(c for c in shuffled_colors if c != reserved_new)
    winner_colors = train_colors[: int(rng.integers(2, 5))]
    train_modes = [winner_colors[i % len(winner_colors)] for i in range(train_count)]
    size_extension = bool(rng.random() < 0.52)

    train_shapes = [(3, 3)]
    if size_extension:
        nonstandard = ((2, 3), (3, 4), (4, 4))
        train_shapes.append(nonstandard[int(rng.integers(3))])
    else:
        train_shapes.append((3, 3))
    train_shapes.extend(
        _sample_shape(rng) if size_extension else (3, 3)
        for _ in range(train_count - 2)
    )
    test_shapes = [
        _sample_shape(rng, test=True) if size_extension else (3, 3)
        for _ in range(test_count)
    ]
    if size_extension and all(shape == (3, 3) for shape in test_shapes):
        test_shapes[0] = ((2, 3), (3, 4), (4, 4))[int(rng.integers(3))]

    train_profiles = [
        _SOURCE_PROFILE if rng.random() < 0.68 else (3, 2, 1, 1, 1, 1)
    ]
    train_profiles.append(_sample_profile(rng, train_shapes[1], nonthree=True))
    for index in range(2, train_count):
        if index == 2 and train_shapes[index] == (3, 3) and rng.random() < 0.6:
            train_profiles.append(_SOURCE_PROFILE)
        else:
            train_profiles.append(_sample_profile(rng, train_shapes[index]))
    test_profiles = [_sample_profile(rng, shape) for shape in test_shapes]

    pair_rngs = _derive_pair_rngs(rng, train_count + test_count)
    seen_inputs: set = set()
    planned_distractor = (
        train_colors[len(winner_colors)] if transition == "distractor" else None
    )
    train_pairs = []
    for index in range(train_count):
        style = (
            ("line" if rng.random() < 0.55 else "random") if index == 0
            else "dispersed" if index == 1 else "random"
        )
        train_pairs.append(
            _make_pair(
                pair_rngs[index], train_shapes[index], train_profiles[index],
                train_modes[index], train_colors,
                planned_distractor if index == 0 else None, style, seen_inputs,
            )
        )

    observed_colors = {
        int(v) for pair in train_pairs for v in pair["input"].ravel()
    }
    train_mode_set = set(train_modes)
    pure_distractors = sorted(observed_colors - train_mode_set)
    _require(bool(pure_distractors), "Training needs a distractor color")
    roles = [transition] + [
        ("distractor", "reuse")[int(rng.integers(2))]
        for _ in range(test_count - 1)
    ]
    used_test_modes: set[int] = set()
    test_pairs = []
    for index, (shape, profile, role) in enumerate(zip(test_shapes, test_profiles, roles)):
        if role == "new":
            mode = reserved_new
        elif role == "reuse":
            mode = _pick_mode(rng, sorted(train_mode_set), used_test_modes)
        else:
            options = [planned_distractor] if index == 0 and planned_distractor is not None else pure_distractors
            mode = _pick_mode(rng, options, used_test_modes)
        used_test_modes.add(mode)
        test_pairs.append(
            _make_pair(
                pair_rngs[train_count + index], shape, profile, mode,
                tuple(range(10)), None, "random", seen_inputs,
            )
        )
    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[name], list) for name in ("train", "test")),
             "Train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "Illegal pair counts")
    seen_inputs: set = set()
    seen_pairs: set = set()
    train_modes: set[int] = set()
    train_mode_counts: set[int] = set()
    train_shapes: set[tuple[int, int]] = set()
    test_shapes: set[tuple[int, int]] = set()
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            label = f"{section}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must have exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}.{name} must be a 2D array")
                _require(all(1 <= d <= 30 for d in grid.shape),
                         f"{label}.{name} has invalid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer),
                         f"{label}.{name} must have integer dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))),
                         f"{label}.{name} has an invalid color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape in _SHAPES, f"{label} has an unsupported shape")
            _require(out.shape == inp.shape, f"{label} changes dimensions")
            colors, counts = np.unique(inp, return_counts=True)
            _require(len(colors) >= 2, f"{label} needs multiple colors")
            _require(int(np.count_nonzero(counts == counts.max())) == 1,
                     f"{label} has a tied frequency maximum")
            mode = int(colors[np.argmax(counts)])
            _require(bool(np.all(out == mode)), f"{label} has an incorrect output")
            _require(not np.array_equal(inp, out), f"{label} is unchanged")
            signature = _content_signature(inp)
            pair_signature = (signature, _content_signature(out))
            _require(signature not in seen_inputs and pair_signature not in seen_pairs,
                     f"{label} duplicates another pair")
            _require(signature not in _SOURCE_INPUT_SIGNATURES,
                     f"{label} copies a source input")
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            if section == "train":
                train_modes.add(mode)
                train_mode_counts.add(int(counts.max()))
                train_shapes.add(inp.shape)
            else:
                test_shapes.add(inp.shape)
    _require(len(train_modes) >= 2, "Training must show different winner colors")
    _require(len(train_mode_counts) >= 2, "Training must show different winner counts")
    _require((3, 3) in train_shapes, "Training must include a source-sized example")
    if any(shape != (3, 3) for shape in test_shapes):
        _require(len(train_shapes) >= 2,
                 "A size-changing test needs varied training sizes")


def _puzzles_equal(left: dict, right: dict) -> bool:
    return all(
        len(left[section]) == len(right[section])
        and all(
            np.array_equal(a[name], b[name]) and a[name].dtype == b[name].dtype
            for a, b in zip(left[section], right[section])
            for name in ("input", "output")
        )
        for section in ("train", "test")
    )


def _visible_prediction(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    # This prediction uses only visible grids, without a seed or generated test output.
    for pair in train_pairs:
        frequencies: dict[int, int] = {}
        for value in pair["input"].flat:
            color = int(value)
            frequencies[color] = frequencies.get(color, 0) + 1
        largest = max(frequencies.values())
        winners = [color for color, count in frequencies.items() if count == largest]
        _require(len(winners) == 1, "Training has no unique mode")
        _require(pair["output"].shape == pair["input"].shape
                 and bool(np.all(pair["output"] == winners[0])),
                 "Training contradicts the frequency rule")
    frequencies = {}
    for value in test_input.flat:
        color = int(value)
        frequencies[color] = frequencies.get(color, 0) + 1
    largest = max(frequencies.values())
    winners = [color for color, count in frequencies.items() if count == largest]
    _require(len(winners) == 1, "Test has no unique mode")
    return np.full(test_input.shape, winners[0], dtype=np.int8)


def self_test() -> None:
    train_counts: set[int] = set()
    test_counts: set[int] = set()
    shapes: set[tuple[int, int]] = set()
    profiles: set[tuple[int, ...]] = set()
    winner_counts: set[int] = set()
    palette_sizes: set[int] = set()
    winners: set[int] = set()
    structures: set[tuple] = set()
    roles: set[str] = set()
    puzzle_signatures: set[tuple] = set()
    short_puzzles = 0
    size_extensions = 0
    mixed_train_palettes = 0
    size_shifted_tests = 0
    size_and_new_color_tests = 0
    source_like_pairs = 0
    line_winners = 0
    dispersed_winner_connected_runner = 0
    mixed_test_roles = 0
    repeated_profiles_with_layout_change = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzles_equal(puzzle, generate_puzzle(seed)),
                 "Automatic generation is not deterministic")
        puzzle_signatures.add(tuple(
            tuple(
                (_content_signature(pair["input"]), _content_signature(pair["output"]))
                for pair in puzzle[section]
            )
            for section in ("train", "test")
        ))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        short_puzzles += len(puzzle["train"]) <= 4
        train_inputs = [pair["input"] for pair in puzzle["train"]]
        train_colors = {int(v) for grid in train_inputs for v in grid.flat}
        train_winners = {_unique_mode(grid) for grid in train_inputs}
        mixed_train_palettes += len({len(np.unique(grid)) for grid in train_inputs}) >= 2
        train_signatures = []
        masks_by_profile: dict[tuple, set] = {}
        for grid in train_inputs:
            counts = tuple(sorted((int(n) for n in np.unique(grid, return_counts=True)[1]), reverse=True))
            signature = (grid.shape, counts)
            train_signatures.append(signature)
            masks_by_profile.setdefault(signature, set()).add(
                tuple(bool(x) for x in (grid == _unique_mode(grid)).flat)
            )
        repeated_profiles_with_layout_change += any(
            len(masks) >= 2 for masks in masks_by_profile.values()
        )
        size_extensions += len({grid.shape for grid in train_inputs}) >= 2
        puzzle_test_roles: set[str] = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                colors, counts = np.unique(inp, return_counts=True)
                _require(int(np.count_nonzero(counts == counts.max())) == 1,
                         "Self-test found tied modes")
                mode = int(colors[np.argmax(counts)])
                _require(out.shape == inp.shape and bool(np.all(out == mode)),
                         "Self-test found an incorrect transformation")
                shapes.add(inp.shape)
                profiles.add(tuple(sorted((int(n) for n in counts), reverse=True)))
                winner_counts.add(int(counts.max()))
                palette_sizes.add(len(colors))
                winners.add(mode)
                profile = tuple(sorted((int(n) for n in counts), reverse=True))
                if section == "train":
                    source_like_pairs += inp.shape == (3, 3) and profile == _SOURCE_PROFILE
                    if int(counts.max()) == 3:
                        line_winners += (
                            (inp.shape[1] == 3 and any(bool(np.all(inp[r, :] == mode)) for r in range(inp.shape[0])))
                            or (inp.shape[0] == 3 and any(bool(np.all(inp[:, c] == mode)) for c in range(inp.shape[1])))
                        )
                    if len(profile) >= 2 and profile[1] >= 2:
                        runner_edges = max(
                            _adjacent_edges(inp, int(color))
                            for color, frequency in zip(colors, counts)
                            if int(frequency) == profile[1]
                        )
                        dispersed_winner_connected_runner += (
                            runner_edges > 0 and _adjacent_edges(inp, mode) < runner_edges
                        )
                # Normalize literal colors in first-occurrence order.
                mapping = {mode: 0}
                normalized = []
                for value in inp.flat:
                    color = int(value)
                    if color not in mapping:
                        mapping[color] = len(mapping)
                    normalized.append(mapping[color])
                structures.add((inp.shape, tuple(normalized)))
                if section == "test":
                    size_shifted_tests += inp.shape != (3, 3)
                    predicted = _visible_prediction(
                        [{"input": p["input"].copy(), "output": p["output"].copy()}
                         for p in puzzle["train"]], inp.copy()
                    )
                    _require(np.array_equal(predicted, out),
                             "Visible-rule prediction failed")
                    if mode in train_winners:
                        role = "reuse"
                    elif mode in train_colors:
                        role = "distractor"
                    else:
                        role = "new"
                    roles.add(role)
                    puzzle_test_roles.add(role)
                    size_and_new_color_tests += inp.shape != (3, 3) and role == "new"
        mixed_test_roles += len(puzzle_test_roles) >= 2
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "Automatic pair counts lack diversity")
    _require(short_puzzles >= 20, "Too few short demonstrations")
    _require(len(puzzle_signatures) >= 40, "Different seeds repeat whole puzzles")
    _require(shapes == set(_SHAPES), "Size variation is missing")
    _require(size_extensions >= 8, "Too few puzzles demonstrate size independence")
    _require(size_shifted_tests >= 8 and size_and_new_color_tests >= 2,
             "Size and color shifts do not combine often enough")
    _require(len(profiles) >= 15 and len(winner_counts) >= 5,
             "Frequency profiles lack diversity")
    _require(
        {(3, 2, 2, 1, 1), (2, 1, 1, 1, 1, 1, 1, 1),
         (4, 3, 2), (4, 2, 2, 1), (5, 4)} <= profiles,
        "Requested 3-by-3 profiles are missing",
    )
    _require(len(palette_sizes) >= 6 and len(winners) >= 9,
             "Palette and literal winner roles lack diversity")
    _require(roles == {"reuse", "distractor", "new"},
             "Train/test color-role transitions are incomplete")
    _require(mixed_test_roles >= 3, "Too few puzzles mix test color roles")
    _require(mixed_train_palettes >= 20, "Training palette sizes lack variation")
    _require(len(structures) >= 100, "Layouts lack diversity")
    _require(repeated_profiles_with_layout_change >= 3,
             "Too few within-puzzle rearrangements")
    _require(source_like_pairs >= 15 and line_winners >= 8
             and dispersed_winner_connected_runner >= 5,
             "Source-like and contrasting layouts are too rare")
    for seed, train, test in (
        (1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
        (1004, None, 1), (1005, None, 4),
        (1006, 2, None), (1007, 10, None),
        (314159, 7, 3),
    ):
        a = generate_puzzle(seed, train, test)
        b = generate_puzzle(seed, train, test)
        validate_generated_puzzle(a)
        _require(_puzzles_equal(a, b), "Requested-count generation is not deterministic")
        _require(train is None or len(a["train"]) == train, "Ignored train count")
        _require(test is None or len(a["test"]) == test, "Ignored test count")
    for train, test in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (4, 4), (True, 1), (3, 1.5),
    ):
        try:
            generate_puzzle(9000, train, test)
        except ValueError:
            continue
        raise AssertionError("An illegal count request was accepted")
