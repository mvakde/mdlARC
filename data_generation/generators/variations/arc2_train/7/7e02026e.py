import numpy as np


_ALLOWED_DIMS = np.array(
    [9, 10, 11, 13, 14, 15, 16, 17, 18],
    dtype=np.int64,
)

# Rejecting these inputs is stronger than merely rejecting their corresponding
# source outputs, because no generated pair can then duplicate a source pair.
_SOURCE_INPUT_ROWS = {
    (
        "808888080888",
        "080008800088",
        "808088888888",
        "080088000800",
        "808800880088",
        "888088008888",
        "808000800000",
        "080808000880",
        "080800088088",
        "888800008080",
        "088000880000",
        "800808888888",
    ),
    (
        "800800088080",
        "808000800800",
        "000808888808",
        "080800808800",
        "800800088800",
        "880808888880",
        "080008080880",
        "088880080088",
        "080888800880",
        "088880008008",
        "808000008800",
        "080808080080",
    ),
    (
        "880008000088",
        "888888880008",
        "888008800088",
        "080888800888",
        "000888880000",
        "000880808800",
        "008808808880",
        "888088880808",
        "880008880888",
        "880008088888",
        "800008888888",
        "808888808808",
    ),
    (
        "808888808080",
        "088800808000",
        "888800088888",
        "800080880080",
        "088808088088",
        "008880000000",
        "808808808000",
        "080800888888",
        "000880080800",
        "000080880880",
        "000880880888",
        "888080000888",
    ),
}


def _check_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _check_optional_count("num_train", num_train, 2, 10)
    num_test = _check_optional_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees count diversity over
        # consecutive seeds while always satisfying num_train > num_test.
        num_test = 1 + seed % 4
        num_train = (
            num_test
            + 1
            + ((seed // 4) % (10 - num_test))
        )
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _plus_centers(grid: np.ndarray, color: int) -> np.ndarray:
    same = grid == color
    mask = np.zeros(grid.shape, dtype=bool)

    mask[1:-1, 1:-1] = (
        same[1:-1, 1:-1]
        & same[:-2, 1:-1]
        & same[2:, 1:-1]
        & same[1:-1, :-2]
        & same[1:-1, 2:]
    )

    return np.argwhere(mask)


def _stencil_union(
    shape: tuple[int, int],
    centers,
) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)

    for row, col in centers:
        row = int(row)
        col = int(col)

        union[row, col] = True
        union[row - 1, col] = True
        union[row + 1, col] = True
        union[row, col - 1] = True
        union[row, col + 1] = True

    return union


def _has_horizontal_adjacent(centers: np.ndarray) -> bool:
    points = {tuple(map(int, point)) for point in centers}
    return any((row, col + 1) in points for row, col in points)


def _has_vertical_adjacent(centers: np.ndarray) -> bool:
    points = {tuple(map(int, point)) for point in centers}
    return any((row + 1, col) in points for row, col in points)


def _has_diagonal_adjacent(centers: np.ndarray) -> bool:
    points = {tuple(map(int, point)) for point in centers}
    return any(
        (row + 1, col - 1) in points
        or (row + 1, col + 1) in points
        for row, col in points
    )


def _has_separated_pair(centers: np.ndarray) -> bool:
    points = [tuple(map(int, point)) for point in centers]

    for index, (row_a, col_a) in enumerate(points):
        for row_b, col_b in points[index + 1:]:
            if max(
                abs(row_a - row_b),
                abs(col_a - col_b),
            ) >= 4:
                return True

    return False


def _has_isolated_center(centers: np.ndarray) -> bool:
    points = [tuple(map(int, point)) for point in centers]

    for index, (row_a, col_a) in enumerate(points):
        if all(
            max(
                abs(row_a - row_b),
                abs(col_a - col_b),
            ) >= 3
            for other_index, (row_b, col_b) in enumerate(points)
            if index != other_index
        ):
            return True

    return False


# These are center arrangements, not output templates. The final input is
# rescanned after its surrounding field and counterexamples are constructed.
_MOTIFS = {
    "single": ((0, 0),),
    "pair_h": ((0, 0), (0, 1)),
    "pair_v": ((0, 0), (1, 0)),
    "pair_d": ((0, 0), (1, 1)),
    "arm_h": ((0, 0), (0, 2)),
    "arm_v": ((0, 0), (2, 0)),
    "run_h": ((0, 0), (0, 1), (0, 2)),
    "run_v": ((0, 0), (1, 0), (2, 0)),
    "bend": ((0, 0), (1, 0), (1, 1)),
    "stair": ((0, 0), (1, 1), (2, 2)),
    "block": ((0, 0), (0, 1), (1, 0), (1, 1)),
}
_OVERLAPS = ("pair_h", "pair_v", "pair_d", "run_h", "run_v", "bend", "stair", "block")
_EXTRA = ("single", "pair_h", "pair_v", "pair_d", "arm_h", "arm_v",
          "run_h", "run_v", "bend", "stair", "block", "separated2",
          "separated3", "mixed")
_HOLDOUT_TRAIN = {
    "h": ("pair_v", "pair_d", "arm_h", "arm_v", "run_v",
          "single", "separated2", "separated3"),
    "v": ("pair_h", "pair_d", "arm_h", "arm_v", "run_h",
          "single", "separated2", "separated3"),
    "d": ("pair_h", "pair_v", "arm_h", "arm_v", "run_h", "run_v",
          "single", "separated2", "separated3"),
}


def _motif_clusters(profile: str, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    if profile == "separated2":
        return [[(0, 0)], [(0, 0)]]
    if profile == "separated3":
        return [[(0, 0)] for _ in range(3)]
    if profile == "mixed":
        main = str(rng.choice(_OVERLAPS))
        return _motif_clusters(main, rng) + [[(0, 0)]]
    if profile not in _MOTIFS:
        raise ValueError("unknown center arrangement")
    points = list(_MOTIFS[profile])
    if profile in ("pair_d", "bend", "stair", "block"):
        rotation = int(rng.integers(0, 4))
        mirror = bool(rng.integers(0, 2))
        for _ in range(rotation):
            points = [(-c, r) for r, c in points]
        if mirror:
            points = [(r, -c) for r, c in points]
        low_r = min(r for r, _ in points)
        low_c = min(c for _, c in points)
        points = [(r - low_r, c - low_c) for r, c in points]
    return [points]


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    # The independent axis draws include source size 12 and rectangular grids.
    if rng.random() < 0.04:
        return (12, 12)
    bands = ((6, 9), (10, 14), (15, 21), (22, 30))
    weights = (0.19, 0.39, 0.27, 0.15)
    h_band = int(rng.choice(4, p=weights))
    w_band = int(rng.choice(4, p=weights))
    return (int(rng.integers(bands[h_band][0], bands[h_band][1] + 1)),
            int(rng.integers(bands[w_band][0], bands[w_band][1] + 1)))


def _place_clusters(rng: np.random.Generator, shape: tuple[int, int],
                    clusters: list[list[tuple[int, int]]]) -> list[tuple[int, int]] | None:
    height, width = shape
    chosen: list[tuple[int, int]] = []
    for offsets in clusters:
        row_max = max(r for r, _ in offsets)
        col_max = max(c for _, c in offsets)
        anchors = [(r, c) for r in range(1, height - row_max - 1)
                   for c in range(1, width - col_max - 1)]
        if not anchors:
            return None
        for index in rng.permutation(len(anchors)):
            r0, c0 = anchors[int(index)]
            proposed = [(r0 + r, c0 + c) for r, c in offsets]
            if all(max(abs(r - pr), abs(c - pc)) >= 4
                   for r, c in proposed for pr, pc in chosen):
                chosen.extend(proposed)
                break
        else:
            return None
    return chosen


def _field(rng: np.random.Generator, shape: tuple[int, int],
           signal: int, distractor: int) -> np.ndarray:
    style = int(rng.choice(3, p=(0.47, 0.25, 0.28)))
    if style == 0:
        density = float(rng.uniform(0.24, 0.52))
    elif style == 1:
        density = float(rng.uniform(0.08, 0.28))
    else:
        density = float(rng.uniform(0.10, 0.31))
    grid = np.where(rng.random(shape) < density, signal, distractor).astype(np.int8)
    if style == 2:
        for _ in range(int(rng.integers(2, 6))):
            h = int(rng.integers(2, 5))
            w = int(rng.integers(2, 6))
            r = int(rng.integers(0, max(1, shape[0] - h + 1)))
            c = int(rng.integers(0, max(1, shape[1] - w + 1)))
            patch = grid[r:r+h, c:c+w]
            patch[rng.random(patch.shape) < float(rng.uniform(0.65, 0.95))] = signal
    return grid


def _plus_cells(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return ((row, col), (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1))


def _add_distractor_pluses(rng: np.random.Generator, grid: np.ndarray,
                            forbidden: np.ndarray, distractor: int,
                            requested: int) -> list[tuple[int, int]] | None:
    selected: list[tuple[int, int]] = []
    for _ in range(requested):
        choices = []
        for r in range(1, grid.shape[0] - 1):
            for c in range(1, grid.shape[1] - 1):
                cells = _plus_cells(r, c)
                if all(not forbidden[rr, cc] for rr, cc in cells) and all(
                    max(abs(r - pr), abs(c - pc)) >= 3 for pr, pc in selected
                ):
                    choices.append((r, c))
        if not choices:
            return None
        center = choices[int(rng.integers(0, len(choices)))]
        selected.append(center)
        for r, c in _plus_cells(*center):
            grid[r, c] = distractor
            forbidden[r, c] = True
    return selected


def _add_near_misses(rng: np.random.Generator, grid: np.ndarray,
                     protected: np.ndarray, signal: int, distractor: int,
                     requested: int) -> int:
    made = 0
    for _ in range(requested):
        choices = [(r, c) for r in range(1, grid.shape[0] - 1)
                   for c in range(1, grid.shape[1] - 1)
                   if all(not protected[rr, cc] for rr, cc in _plus_cells(r, c))]
        if not choices:
            break
        row, col = choices[int(rng.integers(0, len(choices)))]
        cells = _plus_cells(row, col)
        missing = int(rng.integers(0, 5))
        for index, (r, c) in enumerate(cells):
            grid[r, c] = distractor if index == missing else signal
            protected[r, c] = True
        made += 1
    return made


def _clear_distractor_pluses(grid: np.ndarray, signal: int,
                             distractor: int) -> None:
    # Flipping a distractor center cannot create another distractor plus.
    for r, c in _plus_centers(grid, distractor):
        grid[int(r), int(c)] = signal


def _has_unmatched_extension(grid: np.ndarray, signal: int,
                             union: np.ndarray) -> bool:
    height, width = grid.shape
    return any(
        grid[row, col] == signal and not union[row, col]
        and any(0 <= rr < height and 0 <= cc < width and union[rr, cc]
                for rr, cc in ((row - 1, col), (row + 1, col),
                               (row, col - 1), (row, col + 1)))
        for row in range(height) for col in range(width)
    )


def _keep_forced_distractor_pluses(grid: np.ndarray, protected: np.ndarray,
                                  selected: list[tuple[int, int]],
                                  signal: int, distractor: int) -> bool:
    keep = set(selected)
    for row, col in _plus_centers(grid, distractor):
        row, col = int(row), int(col)
        if (row, col) in keep:
            continue
        cells = _plus_cells(row, col)
        choices = [(r, c) for r, c in cells if not protected[r, c]]
        if not choices:
            return False
        r, c = choices[0]
        grid[r, c] = signal
    return len(_plus_centers(grid, distractor)) == len(selected)


def _generate_pair(rng: np.random.Generator, profile: str, signal: int,
                   distractor: int, highlight: int,
                   distractor_mode: int,
                   forbidden_adjacency: str | None = None,
                   require_extension: bool = False) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(650):
        shape = _sample_shape(rng)
        clusters = _motif_clusters(profile, rng)
        desired = _place_clusters(rng, shape, clusters)
        if desired is None:
            continue
        forced = _stencil_union(shape, desired)
        grid = _field(rng, shape, signal, distractor)
        grid[forced] = signal
        protected = forced.copy()

        # Extend some arms into a larger component while leaving other
        # same-color cells outside every detected stencil.
        edge = []
        for r, c in np.argwhere(forced):
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < shape[0] and 0 <= cc < shape[1] and not forced[rr, cc]:
                    edge.append((rr, cc))
        if edge and rng.random() < 0.65:
            for index in rng.choice(len(edge), size=min(len(edge), int(rng.integers(1, 4))), replace=False):
                rr, cc = edge[int(index)]
                grid[rr, cc] = signal

        selected_other = []
        if distractor_mode:
            selected_other = _add_distractor_pluses(
                rng, grid, protected, distractor, distractor_mode
            )
            if selected_other is None:
                continue
        _add_near_misses(rng, grid, protected, signal, distractor,
                         int(rng.integers(0, 4)))
        if distractor_mode == 0:
            _clear_distractor_pluses(grid, signal, distractor)
        elif distractor_mode == 1 and not _keep_forced_distractor_pluses(
            grid, protected, selected_other, signal, distractor
        ):
            continue

        centers = _plus_centers(grid, signal)
        actual = {tuple(map(int, point)) for point in centers}
        if not set(desired).issubset(actual):
            continue
        if not 1 <= len(centers) <= max(35, grid.size // 12):
            continue
        if profile in ("single", "pair_h", "pair_v", "pair_d", "arm_h", "arm_v",
                       "separated2", "separated3") and len(centers) != len(desired):
            continue
        if forbidden_adjacency is not None and {
            "h": _has_horizontal_adjacent,
            "v": _has_vertical_adjacent,
            "d": _has_diagonal_adjacent,
        }[forbidden_adjacency](centers):
            continue
        if distractor_mode == 1 and len(_plus_centers(grid, distractor)) != 1:
            continue
        if distractor_mode == 2 and len(_plus_centers(grid, distractor)) < 2:
            continue
        if distractor_mode == 0 and len(_plus_centers(grid, distractor)) != 0:
            continue

        union = _stencil_union(shape, centers)
        if require_extension and not _has_unmatched_extension(grid, signal, union):
            continue
        unmatched = int(np.count_nonzero((grid == signal) & ~union))
        if unmatched < 2:
            continue
        if min(np.count_nonzero(grid == signal), np.count_nonzero(grid == distractor)) < max(3, int(0.06 * grid.size)):
            continue
        output = grid.copy()
        output[union] = highlight
        if set(map(int, np.unique(output))) != {signal, distractor, highlight}:
            continue
        return grid, output
    raise RuntimeError(f"could not construct arrangement {profile}")


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(value) for value in grid.flat))


def _is_source_input(grid: np.ndarray) -> bool:
    if grid.shape != (12, 12):
        return False
    rows = tuple("".join(str(int(value)) for value in row) for row in grid)
    return rows in _SOURCE_INPUT_ROWS


def _infer_parameters_from_training(train: list[dict]) -> dict:
    signal = None
    highlight = None
    for pair in train:
        inp, out = pair["input"], pair["output"]
        assert len(np.unique(inp)) == 2
        changed = inp != out
        assert np.any(changed)
        old = np.unique(inp[changed])
        new = np.unique(out[changed])
        assert len(old) == len(new) == 1
        old, new = int(old[0]), int(new[0])
        if signal is None:
            signal, highlight = old, new
        else:
            assert (old, new) == (signal, highlight)
        assert highlight not in set(map(int, np.unique(inp)))
    return {"signal": signal, "highlight": highlight}


def _has_adjacent_centers(centers: np.ndarray) -> bool:
    points = [tuple(map(int, point)) for point in centers]
    return any(max(abs(r - rr), abs(c - cc)) == 1
               for i, (r, c) in enumerate(points) for rr, cc in points[i+1:])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    signal, highlight = (int(value) for value in rng.choice(10, size=2, replace=False))
    available = [color for color in range(10) if color not in (signal, highlight)]
    varying_distractor = bool(rng.integers(0, 2))
    count = int(rng.integers(2, 5)) if varying_distractor else 1
    distractors = [int(value) for value in rng.choice(available, size=count, replace=False)]

    # The minimum two training examples teach overlap and separation. Their
    # order and overlap direction vary; further pairs can be simple or crowded.
    holdout = str(rng.choice(("shared", "shared", "h", "v", "d")))
    if holdout == "shared":
        overlap_choices = _OVERLAPS
        extra_choices = _EXTRA
        separated_choices = ("separated2", "separated3", "mixed")
    else:
        extra_choices = _HOLDOUT_TRAIN[holdout]
        overlap_choices = tuple(profile for profile in extra_choices
                                if profile in _OVERLAPS)
        separated_choices = ("separated2", "separated3")
    train_profiles = [str(rng.choice(overlap_choices)),
                      str(rng.choice(separated_choices))]
    while len(train_profiles) < num_train:
        train_profiles.append(str(rng.choice(extra_choices)))
    rng.shuffle(train_profiles)
    test_profiles = [str(rng.choice(_EXTRA)) for _ in range(num_test)]
    if holdout != "shared":
        test_profiles[int(rng.integers(0, num_test))] = {
            "h": "pair_h", "v": "pair_v", "d": "pair_d"
        }[holdout]
    elif num_test and rng.random() < 0.55:
        test_profiles[int(rng.integers(0, num_test))] = str(rng.choice(
            ("run_h", "run_v", "bend", "stair", "block", "mixed", "arm_h", "arm_v")
        ))

    puzzle = {"train": [], "test": []}
    used = set()
    for section, profiles in (("train", train_profiles), ("test", test_profiles)):
        for index, profile in enumerate(profiles):
            for _ in range(30):
                if varying_distractor and section == "train" and index < 2:
                    distractor = distractors[index]
                else:
                    distractor = int(rng.choice(distractors))
                if section == "train" and index == 0:
                    mode = int(rng.integers(1, 3))
                else:
                    mode = int(rng.choice(3, p=(0.28, 0.35, 0.37)))
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                inp, out = _generate_pair(
                    pair_rng, profile, signal, distractor, highlight, mode,
                    holdout if section == "train" and holdout != "shared" else None,
                    section == "train" and index == 0,
                )
                key = _grid_key(inp)
                if key not in used and not _is_source_input(inp):
                    puzzle[section].append({"input": inp, "output": out})
                    used.add(key)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= dim <= 30 for dim in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        assert inp.shape == out.shape and not _is_source_input(inp)
        key = _grid_key(inp)
        pair_key = (key, _grid_key(out))
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
    params = _infer_parameters_from_training(train)
    signal, highlight = params["signal"], params["highlight"]
    assert signal != highlight
    train_centers = []
    unchanged_other_plus = False
    has_component_counterexample = False
    for index, pair in enumerate(train + test):
        inp, out = pair["input"], pair["output"]
        colors = set(map(int, np.unique(inp)))
        assert len(colors) == 2 and signal in colors and highlight not in colors
        distractor = next(iter(colors - {signal}))
        centers = _plus_centers(inp, signal)
        assert len(centers) > 0
        union = _stencil_union(inp.shape, centers)
        expected = inp.copy()
        expected[union] = highlight
        assert np.array_equal(out, expected)
        assert set(map(int, np.unique(out))) == {signal, distractor, highlight}
        assert np.count_nonzero((inp == signal) & ~union) >= 2
        assert min(np.count_nonzero(inp == signal), np.count_nonzero(inp == distractor)) >= max(3, int(0.06 * inp.size))
        if index < len(train):
            train_centers.append(centers)
            if len(_plus_centers(inp, distractor)):
                unchanged_other_plus = True
            if _has_unmatched_extension(inp, signal, union):
                has_component_counterexample = True
    assert unchanged_other_plus
    assert has_component_counterexample
    assert any(_has_adjacent_centers(centers) for centers in train_centers)
    assert any(_has_separated_pair(centers) for centers in train_centers)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[section]) == len(second[section]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[section], second[section])
    ) for section in ("train", "test"))


def _solve_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    changed = train[0]["input"] != train[0]["output"]
    signal = int(np.unique(train[0]["input"][changed])[0])
    highlight = int(np.unique(train[0]["output"][changed])[0])
    answer = test_input.copy()
    for r in range(1, test_input.shape[0] - 1):
        for c in range(1, test_input.shape[1] - 1):
            if all(test_input[rr, cc] == signal for rr, cc in _plus_cells(r, c)):
                for rr, cc in _plus_cells(r, c):
                    answer[rr, cc] = highlight
    return answer


def _union_component_count(union: np.ndarray) -> int:
    remaining = {tuple(map(int, point)) for point in np.argwhere(union)}
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            row, col = pending.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
    return count


def self_test() -> None:
    automatic = generate_puzzle(12345)
    explicit = generate_puzzle(12345, 7, 3)
    assert _puzzles_equal(automatic, generate_puzzle(12345))
    assert _puzzles_equal(explicit, generate_puzzle(12345, 7, 3))
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(20000 + index, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
    for index, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(21000 + index, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(22000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    train_counts, test_counts = set(), set()
    shapes, palettes, layouts, center_counts = set(), set(), set(), set()
    profile_features = set()
    sizes = set()
    varying_palettes = 0
    train_complex_simple = 0
    all_inputs = set()
    total_pairs = 0
    measured = {name: 0 for name in (
        "one", "two_overlap", "crowded", "run", "block", "bend",
        "stair", "disconnected3", "arm_share",
        "small", "large", "source_size", "edge", "near_center", "near_arm",
        "extension", "sparse", "dense", "other_zero", "other_one", "other_many",
        "train_run", "train_block", "test_run", "test_block",
        "same_orientation", "different_orientation", "density_range",
    )}
    test_only_orientations = {name: 0 for name in ("h", "v", "d")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signal = _infer_parameters_from_training(puzzle["train"])["signal"]
        distractors = set()
        puzzle_density = []
        split_orientations = {}
        train_simple = train_crowded = False
        for section in ("train", "test"):
            section_orientations = set()
            for pair in puzzle[section]:
                inp = pair["input"]
                h, w = inp.shape
                assert not _is_source_input(inp)
                assert np.array_equal(_solve_from_training(puzzle["train"], inp), pair["output"])
                distractors.update(set(map(int, np.unique(inp))) - {signal})
                centers = _plus_centers(inp, signal)
                points = {tuple(map(int, point)) for point in centers}
                union = _stencil_union(inp.shape, centers)
                distractor = next(iter(set(map(int, np.unique(inp))) - {signal}))
                other_count = len(_plus_centers(inp, distractor))
                density = float(np.mean(inp == signal))
                puzzle_density.append(density)
                orientations = {
                    name for name, predicate in (
                        ("h", _has_horizontal_adjacent),
                        ("v", _has_vertical_adjacent),
                        ("d", _has_diagonal_adjacent),
                    ) if predicate(centers)
                }
                section_orientations.update(orientations)
                run = any(
                    ((r, c + 1) in points and (r, c + 2) in points)
                    or ((r + 1, c) in points and (r + 2, c) in points)
                    for r, c in points
                )
                block = any(
                    (r, c + 1) in points and (r + 1, c) in points
                    and (r + 1, c + 1) in points for r, c in points
                )
                bend = any(
                    (r, c + 1) in points and (r + 1, c) in points
                    for r, c in points
                )
                stair = any(
                    ((r + 1, c + 1) in points and (r + 2, c + 2) in points)
                    or ((r + 1, c - 1) in points and (r + 2, c - 2) in points)
                    for r, c in points
                )
                disconnected3 = _union_component_count(union) >= 3
                arm_share = any(
                    (r, c + 2) in points or (r + 2, c) in points
                    for r, c in points
                )
                two_overlap = len(points) == 2 and _has_adjacent_centers(centers)
                near_center = near_arm = False
                for r in range(1, h - 1):
                    for c in range(1, w - 1):
                        cells = _plus_cells(r, c)
                        mask = [inp[rr, cc] == signal for rr, cc in cells]
                        if sum(mask) == 4:
                            near_center |= not mask[0]
                            near_arm |= bool(mask[0])
                extension = any(
                    inp[r, c] == signal and not union[r, c]
                    and any(0 <= rr < h and 0 <= cc < w and union[rr, cc]
                            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))
                    for r in range(h) for c in range(w)
                )
                observations = {
                    "one": len(points) == 1, "two_overlap": two_overlap,
                    "crowded": len(points) >= 5, "run": run,
                    "block": block, "bend": bend, "stair": stair,
                    "disconnected3": disconnected3, "arm_share": arm_share,
                    "small": min(h, w) <= 9, "large": max(h, w) >= 22,
                    "source_size": (h, w) == (12, 12),
                    "edge": bool(np.any(union[0]) or np.any(union[-1])
                                 or np.any(union[:, 0]) or np.any(union[:, -1])),
                    "near_center": near_center, "near_arm": near_arm,
                    "extension": extension, "sparse": density < 0.30,
                    "dense": density >= 0.45, "other_zero": other_count == 0,
                    "other_one": other_count == 1,
                    "other_many": other_count >= 2,
                    "train_run": section == "train" and run,
                    "train_block": section == "train" and block,
                    "test_run": section == "test" and run,
                    "test_block": section == "test" and block,
                }
                for name, present in observations.items():
                    measured[name] += bool(present)
                if section == "train":
                    train_simple |= len(points) <= 2
                    train_crowded |= len(points) >= 5
                shapes.add(tuple(inp.shape))
                sizes.add(inp.size)
                center_counts.add(len(centers))
                palettes.add((signal, tuple(sorted(map(int, np.unique(inp))))))
                layouts.add((tuple(inp.shape), tuple(map(tuple, centers.tolist()))))
                profile_features.add((_has_horizontal_adjacent(centers),
                                      _has_vertical_adjacent(centers),
                                      _has_diagonal_adjacent(centers),
                                      _has_isolated_center(centers)))
                all_inputs.add(_grid_key(inp))
                total_pairs += 1
            split_orientations[section] = section_orientations
        varying_palettes += len(distractors) > 1
        train_complex_simple += train_simple and train_crowded
        common_orientations = split_orientations["train"] & split_orientations["test"]
        measured["same_orientation"] += bool(common_orientations)
        measured["different_orientation"] += bool(
            split_orientations["test"] - split_orientations["train"]
        )
        measured["density_range"] += max(puzzle_density) - min(puzzle_density) >= 0.15
        for name in test_only_orientations:
            test_only_orientations[name] += (
                name in split_orientations["test"]
                and name not in split_orientations["train"]
            )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(sizes) >= 30
    assert len(layouts) >= 80 and len(center_counts) >= 5
    assert len(palettes) >= 35 and varying_palettes >= 8
    assert 8 <= varying_palettes <= 34 and train_complex_simple >= 8
    assert len(profile_features) >= 8 and len(all_inputs) >= 0.95 * total_pairs
    assert all(measured[name] >= minimum for name, minimum in {
        "one": 5, "two_overlap": 20, "crowded": 20, "run": 20,
        "block": 8, "bend": 10, "stair": 5,
        "disconnected3": 15, "arm_share": 25,
        "small": 30, "large": 20,
        "source_size": 1, "edge": 40, "near_center": 35,
        "near_arm": 50, "extension": 50, "sparse": 20, "dense": 15,
        "other_zero": 20, "other_one": 20, "other_many": 30,
        "train_run": 10, "train_block": 5,
        "test_run": 5, "test_block": 3,
        "same_orientation": 10, "different_orientation": 10,
        "density_range": 25,
    }.items())
    assert all(value >= 3 for value in test_only_orientations.values())
    assert not _puzzles_equal(generate_puzzle(31001), generate_puzzle(31002))
