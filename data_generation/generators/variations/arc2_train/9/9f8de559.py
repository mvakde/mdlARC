import numpy as np


_AUTO_COUNT_PAIRS = (
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 4),
    (6, 1),
    (7, 2),
    (8, 3),
    (9, 4),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 4),
)


# Rejecting all five original inputs is sufficient to ensure that no complete
# generated input/output pair exactly duplicates a source-task pair.
_SOURCE_INPUT_FLATS = frozenset(
    "".join(rows)
    for rows in (
        (
            "99999999999955",
            "55777777777755",
            "55777777767755",
            "99777777767755",
            "99777777767755",
            "55777777767755",
            "55777777727755",
            "55777777777755",
            "55777777777755",
            "55777777777755",
            "55777777777755",
            "55777777777755",
            "55777777777755",
            "55855888888888",
        ),
        (
            "55555555995555",
            "55555555995555",
            "88777777777999",
            "88777277777999",
            "88777767777999",
            "88777776777999",
            "88777777677999",
            "88777777767999",
            "88777777777999",
            "88777777777999",
            "88777777777999",
            "88777777777888",
            "88777777777888",
            "55995555555888",
        ),
        (
            "99999955999999",
            "99999955999999",
            "99977777777755",
            "99977777777755",
            "99977777777755",
            "99977777777755",
            "99977726666755",
            "99977777777755",
            "99977777777755",
            "99977777777755",
            "99977777777755",
            "99977777777755",
            "99988888988888",
            "99988888988888",
        ),
        (
            "88888888888888",
            "97777777777888",
            "97777777777888",
            "97777777777888",
            "97777777777888",
            "97766662777888",
            "97777777777888",
            "97777777777888",
            "97777777777999",
            "97777777777888",
            "97777777777888",
            "97777777777888",
            "55555555555888",
            "55555555555888",
        ),
        (
            "88998888888888",
            "88998888888888",
            "88998888888888",
            "88777777777779",
            "88777777777679",
            "88777777776779",
            "88777777767779",
            "88777777677779",
            "88777772777779",
            "88777777777779",
            "88777777777779",
            "88777777777779",
            "55555555559559",
            "55555555559559",
        ),
    )
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")

    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        return _AUTO_COUNT_PAIRS[seed % len(_AUTO_COUNT_PAIRS)]

    if train_count is None:
        minimum_train = max(2, int(test_count) + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train exists for num_test")
        train_count = int(rng.integers(minimum_train, 11))
        return train_count, int(test_count)

    maximum_test = min(4, train_count - 1)
    if maximum_test < 1:
        raise ValueError("no legal num_test exists for num_train")
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


_DIRECTIONS = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
    if dr or dc
)
_CARDINAL = tuple(d for d in _DIRECTIONS if 0 in d)
_DIAGONAL = tuple(d for d in _DIRECTIONS if 0 not in d)
_LENGTHS = (2, 3, 4, 5, 6)
_SHAPES = ("tall", "square", "wide")


def _pick(rng: np.random.Generator, values):
    return values[int(rng.integers(len(values)))]


def _grid_signature(grid: np.ndarray) -> tuple:
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _pair_signature(pair: dict) -> tuple:
    return _grid_signature(pair["input"]), _grid_signature(pair["output"])


def _is_source_input(grid: np.ndarray) -> bool:
    return grid.shape == (14, 14) and "".join(
        str(int(v)) for v in grid.flat
    ) in _SOURCE_INPUT_FLATS


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The operation has one puzzle-wide convention: the distinct terminal cell
    # is the arrow tip. Literal colors can either remain fixed or vary locally.
    constant_palette = bool(rng.integers(0, 2))
    colors = [int(c) for c in rng.choice(10, size=6, replace=False)]
    return {
        "constant_palette": constant_palette,
        "roles": tuple(colors[:3]),
        "exterior": tuple(colors[3:]),
        "variable_lengths": bool(rng.integers(0, 4)),
    }


def _sample_directions(
    rng: np.random.Generator, train_count: int, test_count: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    # Even the minimum training set contains a cardinal and a diagonal ray.
    train = [_pick(rng, _CARDINAL), _pick(rng, _DIAGONAL)]
    remaining = [d for d in _DIRECTIONS if d not in train]
    remaining = [remaining[int(i)] for i in rng.permutation(len(remaining))]
    for index in range(train_count - 2):
        train.append(remaining[index] if index < len(remaining)
                     else _pick(rng, _DIRECTIONS))
    train = [train[int(i)] for i in rng.permutation(len(train))]

    mode = int(rng.integers(0, 4))
    if mode == 0 and (1, -1) not in train:
        first = (1, -1)  # occasional source-style withheld direction
    elif mode == 1:
        first = _pick(rng, train)  # a familiar direction can be tested
    else:
        unseen = [d for d in _DIRECTIONS if d not in train]
        first = _pick(rng, unseen if unseen else _DIRECTIONS)
    test = [first]
    while len(test) < test_count:
        options = [d for d in _DIRECTIONS if d not in test]
        if mode in (0, 2):
            preferred = [d for d in options if d not in train]
        elif mode == 1:
            preferred = [d for d in options if d in train]
        else:
            preferred = options
        test.append(_pick(rng, preferred if preferred else options))
    return train, test


def _sample_shape(rng: np.random.Generator, category: str,
                  shaft_length: int) -> tuple[int, int]:
    minimum = max(7, shaft_length + 2)
    if category == "square":
        height = int(rng.integers(minimum, 20))
        widths = [w for w in range(minimum, 22) if abs(w - height) <= 2]
        width = _pick(rng, widths)
    elif category == "tall":
        width = int(rng.integers(minimum, 17))
        height = int(rng.integers(width + 3, min(22, width + 10)))
    else:
        height = int(rng.integers(minimum, 17))
        width = int(rng.integers(height + 3, min(22, height + 10)))
    return height, width


def _sample_geometry(rng: np.random.Generator, category: str,
                     shaft_length: int) -> tuple[int, int, int, int, int, int]:
    ph, pw = _sample_shape(rng, category, shaft_length)
    extra_h = int(rng.integers(2, min(10, 30 - ph) + 1))
    extra_w = int(rng.integers(2, min(10, 30 - pw) + 1))
    top = int(rng.integers(1, extra_h))
    left = int(rng.integers(1, extra_w))
    return ph + extra_h, pw + extra_w, top, left, ph, pw


def _first_exit(tip: tuple[int, int], direction: tuple[int, int],
                height: int, width: int) -> tuple[int, int, int]:
    row, col = tip
    run = 0
    while True:
        row += direction[0]
        col += direction[1]
        if not (0 <= row < height and 0 <= col < width):
            return row, col, run
        run += 1


def _exit_side(row: int, col: int, height: int, width: int) -> str:
    sides = []
    if row < 0:
        sides.append("top")
    elif row >= height:
        sides.append("bottom")
    if col < 0:
        sides.append("left")
    elif col >= width:
        sides.append("right")
    return "+".join(sides)


def _choose_tip(rng: np.random.Generator, height: int, width: int,
                direction: tuple[int, int], shaft_length: int,
                regime: str) -> tuple[int, int, int, int]:
    candidates = {}
    dr, dc = direction
    for row in range(height):
        for col in range(width):
            if not (0 <= row - shaft_length * dr < height and
                    0 <= col - shaft_length * dc < width and
                    0 <= row + dr < height and 0 <= col + dc < width):
                continue
            er, ec, run = _first_exit((row, col), direction, height, width)
            side = _exit_side(er, ec, height, width)
            candidates.setdefault(side, []).append((row, col, run))
    _require(bool(candidates), "geometry has no legal pointer placement")
    sides = list(candidates)
    # Side exits remain more common than the proposed corner extension.
    # This also gives a diagonal ray a useful mix of its two side exits.
    weights = np.asarray([1 if "+" in value else 2 for value in sides],
                         dtype=np.int64)
    side = sides[int(rng.choice(len(sides), p=weights / weights.sum()))]
    positions = candidates[side]
    # Select across the full short-to-long range; test pairs favor harder runs.
    runs = sorted({item[2] for item in positions})
    quantile = int(rng.integers(0, 3))
    if regime == "target_test" and int(rng.integers(0, 3)):
        quantile = int(rng.integers(1, 3))
    lo = min(runs) + (max(runs) - min(runs)) * quantile // 3
    hi = min(runs) + (max(runs) - min(runs)) * (quantile + 1) // 3
    eligible = [item for item in positions if lo <= item[2] <= hi]
    if not eligible:
        eligible = positions
    row, col, _ = _pick(rng, eligible)
    er, ec, _ = _first_exit((row, col), direction, height, width)
    return row, col, er, ec


def _make_exterior_canvas(rng: np.random.Generator, height: int,
                          width: int, top: int, left: int,
                          ph: int, pw: int,
                          colors: tuple[int, ...]) -> np.ndarray:
    rows, cols = np.indices((height, width))
    anchors = [(int(rng.integers(height)), int(rng.integers(width)))
               for _ in colors]
    scores = np.stack([
        abs(rows - ar) + abs(cols - ac) + int(rng.integers(0, 5))
        for ar, ac in anchors
    ])
    canvas = np.asarray(colors, dtype=np.int8)[np.argmin(scores, axis=0)]
    for _ in range(int(rng.integers(1, 7))):
        color = _pick(rng, colors)
        if bool(rng.integers(2)):
            r0 = int(rng.integers(height))
            thickness = int(rng.integers(1, max(2, height // 4 + 1)))
            canvas[r0:min(height, r0 + thickness), :] = color
        else:
            c0 = int(rng.integers(width))
            thickness = int(rng.integers(1, max(2, width // 4 + 1)))
            canvas[:, c0:min(width, c0 + thickness)] = color
    for _ in range(int(rng.integers(1, 5))):
        r0 = int(rng.integers(height))
        c0 = int(rng.integers(width))
        bh = int(rng.integers(1, max(2, height // 3 + 1)))
        bw = int(rng.integers(1, max(2, width // 3 + 1)))
        canvas[r0:min(height, r0 + bh), c0:min(width, c0 + bw)] = _pick(rng, colors)

    exterior = [(r, c) for r in range(height) for c in range(width)
                if not (top <= r < top + ph and left <= c < left + pw)]
    order = rng.permutation(len(exterior))
    # Make each selected exterior role visible without fixing corner colors.
    for color_index, color in enumerate(colors):
        for offset in (0, len(colors)):
            r, c = exterior[int(order[color_index + offset])]
            canvas[r, c] = color
    return canvas.astype(np.int8, copy=False)


def _sample_colors(rng: np.random.Generator, params: dict,
                   pair_index: int) -> tuple[tuple[int, int, int], tuple[int, ...]]:
    if params["constant_palette"]:
        return params["roles"], params["exterior"]
    colors = [int(c) for c in rng.permutation(10)]
    roles = tuple(colors[:3])
    if pair_index == 1 and roles == params["first_roles"]:
        roles = (colors[1], colors[0], colors[2])
    exterior_count = int(rng.integers(1, 5))
    if pair_index == 1 and exterior_count == params["first_exterior_count"]:
        exterior_count = exterior_count % 4 + 1
    return roles, tuple(colors[3:3 + exterior_count])


def generate_pair(rng: np.random.Generator, regime: str,
                  params: dict, direction: tuple[int, int],
                  category: str, shaft_length: int,
                  pair_index: int) -> dict:
    gh, gw, top, left, ph, pw = _sample_geometry(rng, category, shaft_length)
    tr, tc, er, ec = _choose_tip(rng, ph, pw, direction, shaft_length, regime)
    roles, exterior = _sample_colors(rng, params, pair_index)
    if pair_index == 0:
        params["first_roles"] = roles
        params["first_exterior_count"] = len(exterior)
    grid = _make_exterior_canvas(rng, gh, gw, top, left, ph, pw, exterior)
    fill, shaft, tip = roles
    grid[top:top + ph, left:left + pw] = fill
    for distance in range(1, shaft_length + 1):
        grid[top + tr - distance * direction[0],
             left + tc - distance * direction[1]] = shaft
    grid[top + tr, left + tc] = tip
    output = grid.copy()
    output[top + er, left + ec] = fill
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    train_directions, test_directions = _sample_directions(
        rng, train_count, test_count)
    total = train_count + test_count
    categories = [_SHAPES[int(i)] for i in rng.permutation(3)]
    categories += [_pick(rng, _SHAPES) for _ in range(total - 3)]
    lengths = []
    if params["variable_lengths"]:
        lengths = [4, _pick(rng, (2, 3, 5, 6))]
        lengths += [_pick(rng, _LENGTHS) for _ in range(total - 2)]
    else:
        lengths = [4] * total
    seen = set()
    result = {"train": [], "test": []}
    for index, direction in enumerate(train_directions + test_directions):
        regime = "source_train" if index < train_count else "target_test"
        for attempt in range(30):
            child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = generate_pair(child_rng, regime, params, direction,
                                 categories[index], lengths[index], index)
            signature = _grid_signature(pair["input"])
            if signature not in seen and not _is_source_input(pair["input"]):
                break
        else:
            raise ValueError("could not generate a distinct pair")
        seen.add(signature)
        result["train" if regime == "source_train" else "test"].append(pair)
    validate_generated_puzzle(result)
    return result


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(input_grid.shape == output_grid.shape, "grid shape changed")
    changed = np.argwhere(input_grid != output_grid)
    _require(len(changed) == 1, "the output must change exactly one cell")
    cr, cc = map(int, changed[0])
    fill = int(output_grid[cr, cc])
    _require(int(input_grid[cr, cc]) != fill, "the change is invisible")
    locations = np.argwhere(input_grid == fill)
    _require(len(locations) > 0, "panel fill is absent")
    top, left = map(int, locations.min(axis=0))
    bottom, right = map(int, locations.max(axis=0))
    ph, pw = bottom - top + 1, right - left + 1
    _require(ph >= 7 and pw >= 7, "panel is too small")
    panel = input_grid[top:bottom + 1, left:right + 1]
    markers = panel[panel != fill]
    colors, counts = np.unique(markers, return_counts=True)
    _require(len(colors) == 2, "pointer must have shaft and tip colors")
    _require(1 in counts and 2 <= max(counts) <= 6,
             "pointer length is invalid")
    shaft_length = int(max(counts))
    tip = int(colors[np.argmin(counts)])
    shaft = int(colors[np.argmax(counts)])
    _require(len(markers) == shaft_length + 1, "panel is not nearly solid")
    _require(np.count_nonzero(input_grid == tip) == 1,
             "tip occurs outside the panel")
    _require(np.count_nonzero(input_grid == shaft) == shaft_length,
             "shaft occurs outside the panel")
    _require(np.count_nonzero(input_grid == fill) == ph * pw - shaft_length - 1,
             "fill occurs outside the panel")
    tr, tc = map(int, np.argwhere(input_grid == tip)[0])
    shafts = set(map(tuple, np.argwhere(input_grid == shaft)))
    directions = [d for d in _DIRECTIONS if shafts == {
        (tr - step * d[0], tc - step * d[1])
        for step in range(1, shaft_length + 1)
    }]
    _require(len(directions) == 1, "pointer direction is ambiguous")
    direction = directions[0]
    row, col = tr, tc
    run = 0
    while True:
        row += direction[0]
        col += direction[1]
        if top <= row <= bottom and left <= col <= right:
            _require(int(input_grid[row, col]) == fill,
                     "pointer ray meets a marker")
            run += 1
        else:
            break
    _require(run >= 1, "no panel fill ahead of the tip")
    gh, gw = input_grid.shape
    _require(0 <= row < gh and 0 <= col < gw,
             "first exterior cell is outside the grid")
    _require((row, col) == (cr, cc), "changed cell is not the first exit")
    mask = np.ones(input_grid.shape, dtype=bool)
    mask[top:bottom + 1, left:right + 1] = False
    exterior = tuple(int(x) for x in np.unique(input_grid[mask]))
    _require(1 <= len(exterior) <= 4, "exterior palette size is invalid")
    _require(np.array_equal(np.unique(input_grid), np.unique(output_grid)),
             "output lost a color role")
    _require({fill, shaft, tip}.isdisjoint(exterior),
             "panel roles overlap exterior colors")
    _require(int(input_grid[cr, cc]) in exterior,
             "changed cell is not exterior material")
    return {
        "roles": (fill, shaft, tip), "exterior": exterior,
        "direction": direction, "shaft_length": shaft_length,
        "shape": (ph, pw), "grid_shape": (gh, gw),
        "bounds": (top, bottom, left, right),
        "exit_side": _exit_side(row - top, col - left, ph, pw),
        "run": run,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(type(puzzle) is dict and set(puzzle) == {"train", "test"},
                 "puzzle must contain train and test")
        train, test = puzzle["train"], puzzle["test"]
        _require(type(train) is list and type(test) is list,
                 "sections must be lists")
        _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
                 and len(train) > len(test), "illegal pair counts")
        seen_inputs, seen_pairs = set(), set()
        metadata = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                _require(type(pair) is dict and set(pair) == {"input", "output"},
                         "pair must contain input and output")
                for grid in (pair["input"], pair["output"]):
                    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                             "grid must be a 2D NumPy array")
                    h, w = grid.shape
                    _require(1 <= h <= 30 and 1 <= w <= 30,
                             "grid dimensions exceed ARC limits")
                    _require(np.issubdtype(grid.dtype, np.integer),
                             "grid values must be integers")
                    _require(0 <= int(grid.min()) <= int(grid.max()) <= 9,
                             "grid contains an invalid color")
                key = _grid_signature(pair["input"])
                pair_key = _pair_signature(pair)
                _require(key not in seen_inputs and pair_key not in seen_pairs,
                         "duplicate pair")
                seen_inputs.add(key)
                seen_pairs.add(pair_key)
                _require(not _is_source_input(pair["input"]),
                         "pair copies a source input")
                metadata[section].append(_analyze_pair(
                    pair["input"], pair["output"]))
        train_directions = [m["direction"] for m in metadata["train"]]
        _require(any(d in _CARDINAL for d in train_directions)
                 and any(d in _DIAGONAL for d in train_directions),
                 "training must show cardinal and diagonal motion")
        # All choices that change the output are visible in each input:
        # the panel fill, the terminal tip, and the line direction.
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(np.array_equal(pa[key], pb[key])
               for section in ("train", "test")
               for pa, pb in zip(a[section], b[section])
               for key in ("input", "output")) and all(
        len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    corpus = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in corpus}) >= 4,
             "automatic train counts lack diversity")
    _require(len({len(p["test"]) for p in corpus}) >= 3,
             "automatic test counts lack diversity")
    for seed, nt, ne in ((7101, 2, 1), (7102, 10, 1), (7103, 10, 4),
                         (7201, None, 1), (7202, None, 4),
                         (7203, 2, None), (7204, 10, None)):
        p = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(p)
        if nt is not None:
            _require(len(p["train"]) == nt, "train count was ignored")
        if ne is not None:
            _require(len(p["test"]) == ne, "test count was ignored")
    a = generate_puzzle(7001, 7, 3)
    _require(_puzzles_equal(a, generate_puzzle(7001, 7, 3)),
             "explicit generation is nondeterministic")
    _require(_puzzles_equal(generate_puzzle(7010), generate_puzzle(7010)),
             "automatic generation is nondeterministic")
    _require(not _puzzles_equal(generate_puzzle(7020, 6, 2),
                                generate_puzzle(7021, 6, 2)),
             "different seeds produced identical puzzles")
    for index, kwargs in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    )):
        try:
            generate_puzzle(7300 + index, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")

    train, test = [], []
    mixed_lengths = mixed_roles = mixed_shapes = multi_test = 0
    inputs, layouts = set(), set()
    for puzzle in corpus:
        local = {s: [_analyze_pair(x["input"], x["output"])
                     for x in puzzle[s]] for s in ("train", "test")}
        all_meta = local["train"] + local["test"]
        train += local["train"]
        test += local["test"]
        mixed_lengths += len({m["shaft_length"] for m in all_meta}) > 1
        mixed_roles += len({m["roles"] for m in local["train"]}) > 1
        mixed_shapes += len({"tall" if m["shape"][0] > m["shape"][1] + 2
                             else "wide" if m["shape"][1] > m["shape"][0] + 2
                             else "square" for m in all_meta}) > 1
        multi_test += len(local["test"]) > 1 and len({m["direction"]
                     for m in local["test"]}) > 1
        for pair, m in zip(puzzle["train"] + puzzle["test"], all_meta):
            inputs.add(_grid_signature(pair["input"]))
            layouts.add((m["grid_shape"], m["bounds"], m["direction"],
                         m["shaft_length"], m["exit_side"]))
    all_meta = train + test
    _require(len(inputs) >= 270 and len(layouts) >= 250,
             "layouts or inputs lack diversity")
    _require(len({m["grid_shape"] for m in all_meta}) >= 45,
             "grid dimensions lack diversity")
    _require(len({m["shape"] for m in all_meta}) >= 35,
             "panel dimensions lack diversity")
    _require(set(m["direction"] for m in train) == set(_DIRECTIONS)
             and set(m["direction"] for m in test) == set(_DIRECTIONS),
             "direction coverage is incomplete")
    _require(any(m["direction"] == (1, -1) for m in train),
             "down-left never appears in training")
    _require({m["exit_side"] for m in test if m["direction"] == (1, -1)}
             >= {"left", "bottom"},
             "down-left test exits do not include left and bottom")
    _require(any("+" in m["exit_side"] for m in all_meta),
             "no diagonal corner exit")
    _require(set(m["shaft_length"] for m in all_meta) == set(_LENGTHS),
             "shaft length coverage is incomplete")
    _require({len(m["exterior"]) for m in all_meta} == {1, 2, 3, 4},
             "exterior palette sizes lack coverage")
    _require(max(m["run"] for m in all_meta) >= 9 and
             min(m["run"] for m in all_meta) == 1,
             "ray travel lacks short or long examples")
    _require(mixed_lengths >= 20 and mixed_roles >= 12 and
             mixed_shapes >= 32 and multi_test >= 20,
             "within-puzzle variation is insufficient")
