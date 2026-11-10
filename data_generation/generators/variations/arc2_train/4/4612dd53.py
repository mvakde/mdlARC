import numpy as np


_SOURCE_PAIRS_TEXT = (
    (
        (
            "0000000000000",
            "0000000000000",
            "0011100110000",
            "0010000010000",
            "0000100000000",
            "0010100010000",
            "0010000010000",
            "0000100010000",
            "0011110100000",
            "0000000000000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0000000000000",
            "0011122110000",
            "0010200010000",
            "0020100020000",
            "0010100010000",
            "0010200010000",
            "0020100010000",
            "0011112120000",
            "0000000000000",
            "0000000000000",
        ),
    ),
    (
        (
            "0000000000000",
            "0010100110100",
            "0010000000000",
            "0000000000100",
            "0000000000000",
            "0000000000100",
            "0010000000100",
            "0011001101100",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0012122112100",
            "0010000000200",
            "0020000000100",
            "0020000000200",
            "0020000000100",
            "0010000000100",
            "0011221121100",
            "0000000000000",
        ),
    ),
    (
        (
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0011011011100",
            "0010000000100",
            "0000000000000",
            "0010000000100",
            "0011010110000",
            "0010000000100",
            "0000000000100",
            "0011011001100",
            "0000000000000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0011211211100",
            "0010000000100",
            "0020000000200",
            "0010000000100",
            "0011212112200",
            "0010000000100",
            "0020000000100",
            "0011211221100",
            "0000000000000",
            "0000000000000",
        ),
    ),
    (
        (
            "0000000000000",
            "0000000000000",
            "0010110101100",
            "0010000000100",
            "0000000000000",
            "0000000000000",
            "0010000000100",
            "0010101001100",
            "0000000000000",
            "0010000000100",
            "0010110101100",
            "0000000000000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0000000000000",
            "0012112121100",
            "0010000000100",
            "0020000000200",
            "0020000000200",
            "0010000000100",
            "0012121221100",
            "0020000000200",
            "0010000000100",
            "0012112121100",
            "0000000000000",
            "0000000000000",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _text_grid_key(rows: tuple[str, ...]) -> tuple:
    width = len(rows[0])
    _require(all(len(row) == width for row in rows), "Bad source grid")
    return (
        (len(rows), width),
        tuple(int(char) for row in rows for char in row),
    )


_SOURCE_PAIR_KEYS = frozenset(
    (_text_grid_key(input_rows), _text_grid_key(output_rows))
    for input_rows, output_rows in _SOURCE_PAIRS_TEXT
)


def _normalize_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % number_of_choices
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + seed % number_of_choices
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {"repair": int(rng.integers(0, 10))}


def _cuts(rng: np.random.Generator, first: int, last: int, count: int) -> list[int]:
    if not count:
        return []
    spare = last - first - 5 * (count + 1)
    _require(spare >= 0, "Insufficient room for compartments")
    separators = sorted(int(x) for x in rng.choice(spare + count, count, replace=False))
    extras = []
    previous = -1
    for position in separators:
        extras.append(position - previous - 1)
        previous = position
    extras.append(spare + count - previous - 1)
    result = []
    position = first
    for extra in extras[:-1]:
        position += 5 + extra
        result.append(position)
    return result


def _profile_counts(profile: str) -> tuple[int, int]:
    if profile == "none":
        return 0, 0
    if profile[0] in "hv" and profile[1:].isdigit():
        n = int(profile[1:])
        return (n, 0) if profile[0] == "h" else (0, n)
    if profile in ("grid", "t"):
        return 1, 1
    if profile == "grid2":
        return 2, 1
    raise ValueError("Unknown framework profile")


def _make_segments(
    top: int, bottom: int, left: int, right: int,
    horizontal: list[int], vertical: list[int], profile: str,
    rng: np.random.Generator,
) -> list[tuple[str, list[tuple[int, int]]]]:
    ys = [top] + horizontal + [bottom]
    xs = [left] + vertical + [right]
    segments = []
    for y in (top, bottom):
        for x0, x1 in zip(xs, xs[1:]):
            segments.append(("outer", [(y, x) for x in range(x0, x1 + 1)]))
    for x in (left, right):
        for y0, y1 in zip(ys, ys[1:]):
            segments.append(("outer", [(y, x) for y in range(y0, y1 + 1)]))
    if profile == "t":
        if int(rng.integers(0, 2)):
            chosen = int(rng.integers(0, 2))
            active_h = {(horizontal[0], chosen)}
            active_v = {(vertical[0], 0), (vertical[0], 1)}
        else:
            chosen = int(rng.integers(0, 2))
            active_h = {(horizontal[0], 0), (horizontal[0], 1)}
            active_v = {(vertical[0], chosen)}
    else:
        active_h = {(y, j) for y in horizontal for j in range(len(xs) - 1)}
        active_v = {(x, i) for x in vertical for i in range(len(ys) - 1)}
    for y, j in sorted(active_h):
        segments.append(("divider", [(y, x) for x in range(xs[j], xs[j + 1] + 1)]))
    for x, i in sorted(active_v):
        segments.append(("divider", [(y, x) for y in range(ys[i], ys[i + 1] + 1)]))
    return segments


def _mask_from_segments(shape: tuple[int, int], segments: list) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for _, cells in segments:
        for y, x in cells:
            mask[y, x] = True
    return mask


def _damage(
    rng: np.random.Generator, segments: list, mask: np.ndarray, mode: str,
) -> set[tuple[int, int]]:
    protected = set()
    for _, cells in segments:
        # Three consecutive interior cells make each elementary line visible.
        start = int(rng.integers(1, len(cells) - 3))
        protected.update(cells[start:start + 3])
    all_cells = {tuple(int(z) for z in cell) for cell in np.argwhere(mask)}
    eligible = all_cells - protected
    _require(bool(eligible), "No repairable cells")
    outer = {cell for kind, cells in segments if kind == "outer" for cell in cells}
    divider = all_cells - outer
    selected = set()
    if mode == "sparse":
        target = int(rng.integers(1, min(4, len(eligible) + 1)))
        pool = sorted(eligible)
    elif mode == "uniform":
        target = max(1, int(round(float(rng.uniform(.09, .56)) * len(all_cells))))
        pool = sorted(eligible)
    elif mode in ("outer", "divider"):
        pool = sorted(eligible & (outer if mode == "outer" or not divider else divider))
        target = max(1, int(round(float(rng.uniform(.13, .65)) * len(pool))))
    elif mode == "corner":
        outer_segments = [cells for kind, cells in segments if kind == "outer"]
        outer_cells = [cell for cells in outer_segments for cell in cells]
        ys, xs = [cell[0] for cell in outer_cells], [cell[1] for cell in outer_cells]
        corners = [(min(ys), min(xs)), (min(ys), max(xs)),
                   (max(ys), min(xs)), (max(ys), max(xs))]
        corner = corners[int(rng.integers(0, 4))]
        selected.add(corner)
        near = [cell for cell in eligible if abs(cell[0] - corner[0]) + abs(cell[1] - corner[1]) <= 4]
        selected.update(near)
        pool = sorted(eligible - selected)
        target = len(selected) + int(rng.integers(0, min(4, len(pool) + 1)))
    else:  # localized run on one line, possibly leaving other sides intact
        chosen = segments[int(rng.integers(0, len(segments)))][1]
        runs = []
        current = []
        for cell in chosen:
            if cell in eligible:
                current.append(cell)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        runs.sort(key=len, reverse=True)
        if runs:
            selected.update(runs[0][:int(rng.integers(1, len(runs[0]) + 1))])
        pool = sorted(eligible - selected)
        target = len(selected) + int(rng.integers(0, min(3, len(pool) + 1)))
    target = min(target, len(eligible))
    if len(selected) < target:
        order = rng.permutation(len(pool))
        selected.update(pool[int(i)] for i in order[:target - len(selected)])
    return selected


def generate_pair(
    rng: np.random.Generator, regime: str, puzzle_parameters: dict,
    profile: str, colors: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("Unknown regime")
    hcount, vcount = _profile_counts(profile)
    minimum_h = 6 + 5 * hcount
    minimum_w = 6 + 5 * vcount
    # Both splits span small through large frames. Tests favor the large tier
    # slightly, without imposing an ordering against their training examples.
    def extent(minimum: int) -> int:
        roll = float(rng.random())
        small_probability = .40 if regime == "source_train" else .28
        large_probability = .20 if regime == "source_train" else .35
        if roll < small_probability:
            high = min(30, max(12, minimum + 3))
            return int(rng.integers(minimum, high + 1))
        if roll >= 1 - large_probability:
            low = max(minimum, 23)
            return int(rng.integers(low, 31))
        return int(rng.integers(minimum, 31))
    height = extent(minimum_h)
    width = extent(minimum_w)
    canvas_h = int(rng.integers(height, 31))
    canvas_w = int(rng.integers(width, 31))
    top = int(rng.integers(0, canvas_h - height + 1))
    left = int(rng.integers(0, canvas_w - width + 1))
    bottom, right = top + height - 1, left + width - 1
    horizontal = _cuts(rng, top, bottom, hcount)
    vertical = _cuts(rng, left, right, vcount)
    segments = _make_segments(top, bottom, left, right, horizontal, vertical, profile, rng)
    framework = _mask_from_segments((canvas_h, canvas_w), segments)
    modes = ("sparse", "uniform", "localized", "outer", "divider", "corner")
    mode = modes[int(rng.integers(0, len(modes)))]
    missing = _damage(rng, segments, framework, mode)
    background, stroke = colors
    repair = puzzle_parameters["repair"]
    input_grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
    input_grid[framework] = stroke
    for y, x in missing:
        input_grid[y, x] = background
    output_grid = input_grid.copy()
    output_grid[framework & (input_grid == background)] = repair
    return input_grid, output_grid


def _infer_framework_from_input(input_grid: np.ndarray, stroke: int) -> dict:
    positions = np.argwhere(input_grid == stroke)
    _require(len(positions) > 0, "No visible framework")
    top, bottom = int(positions[:, 0].min()), int(positions[:, 0].max())
    left, right = int(positions[:, 1].min()), int(positions[:, 1].max())
    _require(bottom - top >= 5 and right - left >= 5, "Frame is too small")
    # Adjacent survivors on an internal line cannot be caused by orthogonal
    # lines: their spacing is at least five cells.
    hlines = [y for y in range(top + 1, bottom) if np.any(
        (input_grid[y, left:right] == stroke)
        & (input_grid[y, left + 1:right + 1] == stroke))]
    vlines = [x for x in range(left + 1, right) if np.any(
        (input_grid[top:bottom, x] == stroke)
        & (input_grid[top + 1:bottom + 1, x] == stroke))]
    ys, xs = [top] + hlines + [bottom], [left] + vlines + [right]
    _require(all(b - a >= 2 for a, b in zip(ys, ys[1:])), "Ambiguous horizontal spacing")
    _require(all(b - a >= 2 for a, b in zip(xs, xs[1:])), "Ambiguous vertical spacing")
    inferred = np.zeros(input_grid.shape, dtype=bool)
    active_h, active_v = [], []
    for y in (top, bottom):
        inferred[y, left:right + 1] = True
    for x in (left, right):
        inferred[top:bottom + 1, x] = True
    for y in hlines:
        for j, (x0, x1) in enumerate(zip(xs, xs[1:])):
            values = input_grid[y, x0:x1 + 1]
            if np.any((values[:-1] == stroke) & (values[1:] == stroke)):
                inferred[y, x0:x1 + 1] = True
                active_h.append((y, j))
    for x in vlines:
        for i, (y0, y1) in enumerate(zip(ys, ys[1:])):
            values = input_grid[y0:y1 + 1, x]
            if np.any((values[:-1] == stroke) & (values[1:] == stroke)):
                inferred[y0:y1 + 1, x] = True
                active_v.append((x, i))
    _require(np.all(inferred[input_grid == stroke]), "Stroke outside inferred framework")
    _require(all(any(row == y for row, _ in active_h) for y in hlines), "Invisible horizontal line")
    _require(all(any(col == x for col, _ in active_v) for x in vlines), "Invisible vertical line")
    return {"mask": inferred, "top": top, "bottom": bottom, "left": left,
            "right": right, "hlines": hlines, "vlines": vlines,
            "active_h": active_h, "active_v": active_v,
            "outer_height": bottom - top + 1, "outer_width": right - left + 1,
            "margins": (top, input_grid.shape[0] - bottom - 1,
                        left, input_grid.shape[1] - right - 1)}


def _analyze_pair(pair: dict) -> dict:
    input_grid, output_grid = pair["input"], pair["output"]
    _require(input_grid.shape == output_grid.shape, "Dimensions changed")
    changed = input_grid != output_grid
    _require(np.any(changed), "No repairs")
    background_values = np.unique(input_grid[changed])
    repair_values = np.unique(output_grid[changed])
    _require(len(background_values) == 1 and len(repair_values) == 1,
             "Inconsistent repair colors")
    background, repair = int(background_values[0]), int(repair_values[0])
    input_colors = set(int(x) for x in np.unique(input_grid))
    _require(len(input_colors) == 2 and background in input_colors and repair not in input_colors,
             "Invalid visible colors")
    stroke = next(iter(input_colors - {background}))
    info = _infer_framework_from_input(input_grid, stroke)
    mask = info["mask"]
    expected = input_grid.copy()
    expected[mask & (input_grid == background)] = repair
    _require(np.array_equal(output_grid, expected), "Incorrect framework repair")
    _require(np.all(input_grid[~mask] == background), "Clue outside framework")
    _require(np.all(output_grid[~mask] == background), "Changed outside framework")
    top, bottom, left, right = (info[k] for k in ("top", "bottom", "left", "right"))
    for side in (input_grid[top, left + 1:right], input_grid[bottom, left + 1:right],
                 input_grid[top + 1:bottom, left], input_grid[top + 1:bottom, right]):
        _require(np.any(side == stroke), "Outer side lacks a noncorner clue")
    info.update({"roles": (background, stroke, repair),
                 "missing_count": int(np.count_nonzero(changed)),
                 "repair_fraction": float(np.count_nonzero(changed) / np.count_nonzero(mask)),
                 "corners_missing": sum(input_grid[y, x] == background for y in (info["top"], info["bottom"])
                                        for x in (info["left"], info["right"])),
                 "intact_sides": sum(not np.any(changed[y, info["left"]:info["right"] + 1])
                                      for y in (info["top"], info["bottom"]))
                 + sum(not np.any(changed[info["top"]:info["bottom"] + 1, x])
                       for x in (info["left"], info["right"]))})
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle schema")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")), "Pair lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Pair counts")
    seen_inputs, seen_pairs, infos = set(), set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair schema")
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid type")
            _require(np.issubdtype(grid.dtype, np.integer), "Grid dtype")
            _require(all(1 <= d <= 30 for d in grid.shape), "Grid bounds")
            _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "Grid colors")
        ik, pk = _grid_key(pair["input"]), _pair_key(pair)
        _require(ik not in seen_inputs and pk not in seen_pairs, "Duplicate pair")
        _require(pk not in _SOURCE_PAIR_KEYS, "Source copy")
        seen_inputs.add(ik)
        seen_pairs.add(pk)
        infos.append(_analyze_pair(pair))
    repairs = {info["roles"][2] for info in infos}
    _require(len(repairs) == 1, "Puzzle repair color varies")
    _require(all(info["roles"][2] in repairs for info in infos[:nt]), "Unlearned repair color")
    _require(len({info["roles"][:2] for info in infos[:nt]}) >= 2,
             "Training must demonstrate pair-level visible colors")
    train_types = {(bool(i["hlines"]), bool(i["vlines"])) for i in infos[:nt]}
    def crossing(info: dict) -> bool:
        return bool(info["hlines"] and info["vlines"]
                    and len(info["active_h"]) == len(info["hlines"]) * (len(info["vlines"]) + 1)
                    and len(info["active_v"]) == len(info["vlines"]) * (len(info["hlines"]) + 1))
    for info in infos[nt:]:
        _require((bool(info["hlines"]), bool(info["vlines"])) in train_types,
                 "Test line orientation absent from training")
        if len(info["hlines"]) >= 2:
            _require(any(len(t["hlines"]) >= 2 for t in infos[:nt]),
                     "Multiple horizontal dividers absent from training")
        if len(info["vlines"]) >= 2:
            _require(any(len(t["vlines"]) >= 2 for t in infos[:nt]),
                     "Multiple vertical dividers absent from training")
        if info["hlines"] and info["vlines"]:
            _require(any(t["hlines"] and t["vlines"] and crossing(t) == crossing(info)
                         for t in infos[:nt]), "Mixed junction type absent from training")
    for pair in puzzle["test"]:
        _require(np.array_equal(_predict_from_visible(puzzle["train"], pair["input"]),
                                pair["output"]), "Test output is not recoverable from visible clues")


def _pick_profiles(rng: np.random.Generator, train_count: int, test_count: int) -> tuple[list[str], list[str]]:
    level = int(rng.integers(0, 4))
    simple = ["none", "h1", "v1"]
    advanced = ["h2", "v2"] if level == 1 else (["grid", "t"] if level == 2 else ["h2", "v2", "grid", "grid2", "t"])
    if level == 0:
        train = ["none", str(rng.choice(["h1", "v1"]))]
    else:
        train = [str(rng.choice(simple)), str(rng.choice(advanced))]
    choices = simple + advanced
    while len(train) < train_count:
        train.append(str(rng.choice(choices)))
    train = [train[int(i)] for i in rng.permutation(len(train))]
    # A test may increase an already demonstrated divider count, but crossings
    # and terminating dividers are each demonstrated before appearing in test.
    witnessed = {(bool(_profile_counts(p)[0]), bool(_profile_counts(p)[1])) for p in train}
    compatible = [p for p in simple if
                  (bool(_profile_counts(p)[0]), bool(_profile_counts(p)[1])) in witnessed]
    compatible.extend(p for p in train if p not in compatible)
    if "h2" in train:
        compatible.append("h3")
    if "v2" in train:
        compatible.append("v3")
    if "grid" in train and "h2" in train and level == 3:
        compatible.append("grid2")
    test = [str(rng.choice(compatible)) for _ in range(test_count)]
    return train, test


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    parameters = sample_puzzle_parameters(rng)
    train_profiles, test_profiles = _pick_profiles(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used = set()
    prior_colors = None
    for split, profiles in (("train", train_profiles), ("test", test_profiles)):
        for profile in profiles:
            accepted = None
            for _ in range(80):
                available = [x for x in range(10) if x != parameters["repair"]]
                pick = rng.choice(available, 2, replace=False)
                colors = (int(pick[0]), int(pick[1]))
                if split == "train" and prior_colors is not None and len(puzzle["train"]) == 1 and colors == prior_colors:
                    continue
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                input_grid, output_grid = generate_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                                         parameters, profile, colors)
                pair = {"input": input_grid, "output": output_grid}
                if _grid_key(input_grid) in used or _pair_key(pair) in _SOURCE_PAIR_KEYS:
                    continue
                try:
                    _analyze_pair(pair)
                except AssertionError:
                    continue
                accepted = pair
                prior_colors = colors
                used.add(_grid_key(input_grid))
                break
            if accepted is None:
                raise ValueError("Could not construct an identifiable framework")
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(_pair_key(p) for split in ("train", "test") for p in puzzle[split])


def _predict_from_visible(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    repairs = {int(p["output"][p["input"] != p["output"]][0]) for p in train}
    _require(len(repairs) == 1, "Training repair convention unclear")
    repair = next(iter(repairs))
    candidates = []
    for stroke in np.unique(test_input):
        try:
            info = _infer_framework_from_input(test_input, int(stroke))
            background = next(iter(set(int(x) for x in np.unique(test_input)) - {int(stroke)}))
            if not np.all(test_input[~info["mask"]] == background):
                continue
            predicted = test_input.copy()
            predicted[info["mask"] & (test_input == background)] = repair
            candidates.append(predicted)
        except AssertionError:
            pass
    _require(len(candidates) == 1, "Test input has ambiguous color roles")
    return candidates[0]


def self_test() -> None:
    source_pairs = [{"input": np.array([[int(c) for c in row] for row in a], dtype=np.int8),
                     "output": np.array([[int(c) for c in row] for row in b], dtype=np.int8)}
                    for a, b in _SOURCE_PAIRS_TEXT]
    source_input, source_output = source_pairs[0]["input"], source_pairs[0]["output"]
    _require(source_input[8, 8] == 0 and source_output[8, 8] == 2,
             "Source corner repair evidence")
    _require(np.array_equal(_infer_framework_from_input(source_input, 1)["mask"],
                            (source_input == 1) | (source_output == 2)),
             "Source framework inference")
    for pair in source_pairs:
        _require(np.array_equal(_infer_framework_from_input(pair["input"], 1)["mask"],
                                (pair["input"] == 1) | (pair["output"] == 2)),
                 "Source framework inference")
    _require(np.array_equal(_predict_from_visible(source_pairs[:3], source_pairs[3]["input"]),
                            source_pairs[3]["output"]), "Source test prediction")
    for counts in ((2, 1), (10, 1), (10, 4), (4, 2), (7, 3),
                   (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(801 + len(str(counts)), *counts)
        validate_generated_puzzle(p)
        if counts[0] is not None:
            _require(len(p["train"]) == counts[0], "Train count mismatch")
        if counts[1] is not None:
            _require(len(p["test"]) == counts[1], "Test count mismatch")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(900, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    signature = _puzzle_signature(generate_puzzle(777, 6, 3))
    _require(signature == _puzzle_signature(generate_puzzle(777, 6, 3)), "Nondeterministic")
    summaries = []
    signatures = set()
    counts = [set(), set()]
    repair_colors = set()
    observed = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        counts[0].add(len(puzzle["train"]))
        counts[1].add(len(puzzle["test"]))
        train_infos = [_analyze_pair(p) for p in puzzle["train"]]
        test_infos = [_analyze_pair(p) for p in puzzle["test"]]
        repair_colors.add(train_infos[0]["roles"][2])
        _require(len({x["roles"][:2] for x in train_infos}) > 1,
                 "Visible colors do not vary within training")
        for info in train_infos + test_infos:
            if info["hlines"] and info["vlines"]:
                if (len(info["active_h"]) == len(info["hlines"]) * (len(info["vlines"]) + 1)
                        and len(info["active_v"]) == len(info["vlines"]) * (len(info["hlines"]) + 1)):
                    observed.add("crossing")
                else:
                    observed.add("termination")
            if 30 in (info["outer_height"], info["outer_width"]):
                observed.add("frame_30")
        for split, infos in (("train", train_infos), ("test", test_infos)):
            for pair, info in zip(puzzle[split], infos):
                if 30 in pair["input"].shape:
                    observed.add("canvas_30")
                if info["outer_height"] <= 12 and info["outer_width"] <= 12:
                    observed.add(split + "_small")
                if max(info["outer_height"], info["outer_width"]) >= 25:
                    observed.add(split + "_large")
                if max(info["margins"]) > 4:
                    observed.add("wide_margin")
                if 0 in info["margins"]:
                    observed.add("edge_touch")
                if max(info["outer_height"], info["outer_width"]) >= 2 * min(info["outer_height"], info["outer_width"]):
                    observed.add("wide_aspect")
                changed = pair["input"] != pair["output"]
                top, bottom, left, right = (info[k] for k in ("top", "bottom", "left", "right"))
                if any(changed[y, x] and changed[y, x + dx] and changed[y + dy, x]
                       for y, dy in ((top, 1), (bottom, -1))
                       for x, dx in ((left, 1), (right, -1))):
                    observed.add("corner_wrap")
                outer = np.zeros(changed.shape, dtype=bool)
                outer[top, left:right + 1] = outer[bottom, left:right + 1] = True
                outer[top:bottom + 1, left] = outer[top:bottom + 1, right] = True
                divider = info["mask"] & ~outer
                if np.any(divider) and np.any(changed & outer) and not np.any(changed & divider):
                    observed.add("intact_divider")
                if np.any(divider) and np.any(changed & divider) and not np.any(changed & outer):
                    observed.add("divider_only_damage")
                longest = 0
                for lines in (changed, changed.T):
                    for line in lines:
                        run = 0
                        for cell in line:
                            run = run + 1 if cell else 0
                            longest = max(longest, run)
                if longest >= 10:
                    observed.add("long_gap")
        if any(q["repair_fraction"] < t["repair_fraction"] for q in test_infos for t in train_infos):
            observed.add("lighter_test")
        if any(len(q["hlines"]) + len(q["vlines"]) >
               max(len(t["hlines"]) + len(t["vlines"]) for t in train_infos)
               for q in test_infos):
            observed.add("more_dividers_test")
        if any(q["hlines"] and not q["vlines"] for q in test_infos) and any(t["vlines"] and not t["hlines"] for t in train_infos):
            observed.add("vertical_to_horizontal")
        if any(q["vlines"] and not q["hlines"] for q in test_infos) and any(t["hlines"] and not t["vlines"] for t in train_infos):
            observed.add("horizontal_to_vertical")
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict_from_visible(puzzle["train"], pair["input"]), pair["output"]),
                     "Visible-clue prediction failed")
        summaries.extend(train_infos + test_infos)
    _require(len(signatures) == 42 and len(counts[0]) >= 4 and len(counts[1]) >= 3,
             "Insufficient puzzle/count diversity")
    _require(any(x["corners_missing"] for x in summaries), "No missing corners")
    _require(any(x["intact_sides"] for x in summaries), "No intact outer sides")
    _require(any(len(x["hlines"]) >= 2 for x in summaries), "No multiple horizontal dividers")
    _require(any(len(x["vlines"]) >= 2 for x in summaries), "No multiple vertical dividers")
    _require(any(x["hlines"] and x["vlines"] for x in summaries), "No mixed partitions")
    _require(any(x["missing_count"] <= 3 for x in summaries), "No sparse repairs")
    _require(any(max(x["margins"]) >= 5 for x in summaries), "No wide margin")
    _require(any(0 in x["margins"] for x in summaries), "No edge-touching frames")
    _require(any(x["outer_height"] >= 25 for x in summaries), "No large frames")
    _require(any(x["outer_height"] <= 10 for x in summaries), "No small frames")
    _require(len(repair_colors) == 10, "Repair colors lack cross-seed variety")
    _require({"crossing", "termination", "canvas_30", "frame_30",
              "train_small", "test_small", "train_large", "test_large",
              "wide_margin", "edge_touch", "wide_aspect", "corner_wrap",
              "intact_divider", "divider_only_damage", "long_gap",
              "lighter_test", "more_dividers_test",
              "vertical_to_horizontal", "horizontal_to_vertical"} <= observed,
             "A reviewed variation axis was not observed")
