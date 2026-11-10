import numpy as np


_SOURCE_PAIR_ROWS = frozenset({
    (
        (
            "01111", "01100", "01110", "00111", "01110",
            "08800", "80080", "08008", "00800", "80808",
            "06066", "00666", "06000", "06606", "00000",
        ),
        ("06166", "81666", "06118", "06616", "81118"),
    ),
    (
        (
            "10101", "01001", "01000", "10011", "10011",
            "00000", "08880", "08000", "80008", "80880",
            "00606", "60600", "60006", "60006", "06666",
        ),
        ("10606", "61681", "61006", "60016", "16666"),
    ),
    (
        (
            "00110", "11100", "01110", "01001", "10011",
            "80880", "80888", "88808", "08088", "80888",
            "60606", "00006", "66666", "00600", "06060",
        ),
        ("60616", "11186", "66666", "01681", "16861"),
    ),
    (
        (
            "11100", "00110", "11001", "01111", "00001",
            "08008", "88800", "00000", "00080", "00888",
            "66000", "06660", "00606", "00666", "66666",
        ),
        ("66108", "86660", "11606", "01666", "66666"),
    ),
    (
        (
            "10000", "11111", "01010", "11101", "00010",
            "80800", "80080", "80008", "88000", "88000",
            "06006", "60066", "06660", "66066", "00606",
        ),
        ("16806", "61166", "86668", "66166", "88616"),
    ),
    (
        (
            "10111", "10100", "01100", "01110", "10101",
            "00800", "00888", "88088", "00800", "88080",
            "06060", "00606", "06006", "00606", "60660",
        ),
        ("16161", "10686", "86186", "01616", "68661"),
    ),
})


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("seed must be a non-negative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _count_value(
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
    train_count = _count_value("num_train", num_train, 2, 10)
    test_count = _count_value("num_test", num_test, 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        smallest_train = max(2, test_count + 1)
        number_of_choices = 11 - smallest_train
        train_count = (
            smallest_train
            + (seed // 4) % number_of_choices
        )
        return train_count, test_count

    if train_count is None:
        smallest_train = max(2, test_count + 1)
        train_count = int(rng.integers(smallest_train, 11))
    else:
        largest_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, largest_test + 1))

    if train_count <= test_count:
        raise ValueError("no legal count can be selected")

    return train_count, test_count


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    height = int(rng.integers(3, 11))
    if rng.random() < 0.28:
        width = int(rng.integers(11, 19))
    else:
        width = int(rng.integers(3, 11))
    return height, width


def _sample_colors(rng: np.random.Generator) -> tuple[int, int, int, int]:
    return tuple(int(x) for x in rng.choice(10, size=4, replace=False))


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {
        "shared_shape": bool(rng.random() < 0.44),
        "shared_colors": bool(rng.random() < 0.44),
        "withhold_triple": bool(rng.random() < 0.26),
        "shape": _sample_shape(rng),
        "colors": _sample_colors(rng),
    }


def _field(
    rng: np.random.Generator,
    height: int,
    width: int,
    style: str,
) -> np.ndarray:
    rows, cols = np.indices((height, width))
    noise = rng.normal(size=(height, width))
    if style == "dots":
        field = noise
    elif style in ("patch", "hole"):
        center_r = rng.uniform(0, height - 1)
        center_c = rng.uniform(0, width - 1)
        radius_r = rng.uniform(0.23, 0.72) * height
        radius_c = rng.uniform(0.23, 0.72) * width
        distance = ((rows - center_r) / radius_r) ** 2 + ((cols - center_c) / radius_c) ** 2
        field = (-distance if style == "patch" else distance) + 0.32 * noise
    elif style == "stripe":
        if rng.random() < 0.5:
            coordinate, limit = rows, height
        else:
            coordinate, limit = cols, width
        center = rng.uniform(0, limit - 1)
        field = -np.abs(coordinate - center) + 0.34 * noise
    elif style == "runs":
        field = 0.38 * noise
        for _ in range(int(rng.integers(2, 7))):
            if rng.random() < 0.5:
                row = int(rng.integers(height))
                start = int(rng.integers(width))
                length = int(rng.integers(2, max(3, width + 1)))
                field[row, start:min(width, start + length)] += rng.uniform(1.5, 3.2)
            else:
                col = int(rng.integers(width))
                start = int(rng.integers(height))
                length = int(rng.integers(2, max(3, height + 1)))
                field[start:min(height, start + length), col] += rng.uniform(1.5, 3.2)
    else:
        raise ValueError("unknown field style")
    return (field - field.mean()) / max(float(field.std()), 0.01)


def _sample_density(rng: np.random.Generator, high_overlap: bool) -> float:
    choice = rng.random()
    if choice < 0.14:
        return float(rng.uniform(0.07, 0.23))
    if choice < 0.28:
        return float(rng.uniform(0.78, 0.94))
    if high_overlap and choice < 0.70:
        return float(rng.uniform(0.55, 0.83))
    return float(rng.uniform(0.27, 0.73))


def _codes_from_masks(
    rng: np.random.Generator,
    height: int,
    width: int,
    high_overlap: bool,
) -> np.ndarray:
    styles = ("dots", "patch", "hole", "stripe", "runs")
    source_like = not high_overlap and rng.random() < 0.18
    shared = _field(rng, height, width, styles[int(rng.integers(len(styles)))])
    correlation = float(rng.choice(
        (0.0, 0.1, 0.25, 0.5, 0.75),
        p=(0.28, 0.25, 0.25, 0.15, 0.07),
    ))
    if high_overlap:
        correlation = max(correlation, float(rng.choice((0.4, 0.65, 0.82))))
    if source_like:
        correlation = 0.0
    masks = []
    area = height * width
    for _ in range(3):
        style = "dots" if source_like else styles[int(rng.integers(len(styles)))]
        personal = _field(rng, height, width, style)
        score = (1.0 - correlation) * personal + correlation * shared
        score += 0.08 * rng.normal(size=(height, width))
        density = float(rng.uniform(0.36, 0.72)) if source_like else _sample_density(rng, high_overlap)
        count = int(round(density * area))
        count = max(1, min(area - 1, count))
        selected = np.argpartition(score.reshape(-1), -count)[-count:]
        mask = np.zeros(area, dtype=bool)
        mask[selected] = True
        masks.append(mask.reshape(height, width))

    # Occasionally let a lower-priority role disappear completely in the
    # composite while remaining plainly visible in its own input panel.
    if rng.random() < 0.06 and int(masks[2].sum()) >= 2:
        role = int(rng.choice((0, 1)))
        covered = np.flatnonzero(masks[2].reshape(-1))
        subset_size = int(rng.integers(1, len(covered) + 1))
        subset = rng.choice(covered, size=subset_size, replace=False)
        replacement = np.zeros(area, dtype=bool)
        replacement[subset] = True
        masks[role] = replacement.reshape(height, width)

    return (
        4 * masks[0].astype(np.int8)
        + 2 * masks[1].astype(np.int8)
        + masks[2].astype(np.int8)
    )


def _has_triple_block(codes: np.ndarray) -> bool:
    triple = codes == 7
    return bool(
        np.any(
            triple[:-1, :-1]
            & triple[1:, :-1]
            & triple[:-1, 1:]
            & triple[1:, 1:]
        )
    )


def _has_block(mask: np.ndarray) -> bool:
    return bool(np.any(
        mask[:-1, :-1] & mask[1:, :-1]
        & mask[:-1, 1:] & mask[1:, 1:]
    ))


def _has_run(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:])
        or np.any(mask[:-2, :] & mask[1:-1, :] & mask[2:, :])
    )


def _isolated(mask: np.ndarray) -> np.ndarray:
    return (
        mask
        & ~np.pad(mask[1:, :], ((0, 1), (0, 0)))
        & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)))
        & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)))
        & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)))
    )


def _masks_nontrivial(codes: np.ndarray) -> bool:
    return all(
        bool(np.any((codes & bit) != 0))
        and bool(np.any((codes & bit) == 0))
        for bit in (4, 2, 1)
    )


def _add_required_codes(
    rng: np.random.Generator,
    codes: np.ndarray,
    required: set[int],
) -> np.ndarray:
    missing = sorted(required - set(int(x) for x in np.unique(codes)))
    if not missing:
        return codes
    result = codes.copy()
    flat = result.reshape(-1)
    counts = np.bincount(flat, minlength=8)
    candidates = []
    for position in rng.permutation(result.size):
        value = int(flat[position])
        if value not in required or counts[value] > 1:
            candidates.append(int(position))
            counts[value] -= 1
    for position, code in zip(candidates, missing):
        flat[int(position)] = code
    return result


def _sample_codes(
    rng: np.random.Generator,
    height: int,
    width: int,
    regime: str,
    withhold_triple: bool,
    required: set[int],
    force_triple: bool,
) -> np.ndarray:
    for _ in range(80):
        high_overlap = regime == "target_test" and rng.random() < 0.48
        codes = _codes_from_masks(rng, height, width, high_overlap)
        if regime == "source_train" and withhold_triple:
            triple = codes == 7
            if bool(np.any(triple)):
                codes[triple] = rng.choice((3, 5, 6), size=int(triple.sum()))
        if force_triple:
            required = required | {7}
        codes = _add_required_codes(rng, codes, required)
        if not _masks_nontrivial(codes):
            continue
        if not bool(np.any(np.isin(codes, (3, 5, 6, 7)))):
            continue
        if regime == "source_train" and withhold_triple and bool(np.any(codes == 7)):
            continue
        return codes

    # A bounded fallback preserves the required evidence for tiny panels.
    values = np.array(sorted(required | {0, 1, 2, 3, 4, 5, 6}), dtype=np.int8)
    filler = rng.integers(0, 7, size=height * width - len(values), dtype=np.int8)
    all_values = np.concatenate((values, filler))
    rng.shuffle(all_values)
    return all_values.reshape(height, width)


def _pair_from_codes(
    codes: np.ndarray,
    colors: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    background, top_color, middle_color, bottom_color = colors
    top = np.where((codes & 4) != 0, top_color, background)
    middle = np.where((codes & 2) != 0, middle_color, background)
    bottom = np.where((codes & 1) != 0, bottom_color, background)
    input_grid = np.vstack((top, middle, bottom)).astype(np.int8)
    output_grid = np.full(codes.shape, background, dtype=np.int8)
    output_grid[(codes & 2) != 0] = middle_color
    output_grid[(codes & 4) != 0] = top_color
    output_grid[(codes & 1) != 0] = bottom_color
    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _row_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return (_row_key(input_grid), _row_key(output_grid)) in _SOURCE_PAIR_ROWS


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    presentation = _sample_puzzle_parameters(rng)
    puzzle: dict = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    demonstrated: set[int] = set()

    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(96):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if presentation["shared_shape"]:
                    height, width = presentation["shape"]
                else:
                    height, width = _sample_shape(pair_rng)
                    if section == "train" and index == 1 and (height, width) == (
                        puzzle["train"][0]["input"].shape[0] // 3,
                        puzzle["train"][0]["input"].shape[1],
                    ):
                        continue
                if presentation["shared_colors"]:
                    colors = presentation["colors"]
                else:
                    colors = _sample_colors(pair_rng)

                required = set(range(7)) - demonstrated if section == "train" and index == count - 1 else set()
                force_triple = (
                    section == "test"
                    and index == 0
                    and presentation["withhold_triple"]
                )
                regime = "source_train" if section == "train" else "target_test"
                codes = _sample_codes(
                    pair_rng, height, width, regime,
                    presentation["withhold_triple"], required, force_triple,
                )
                input_grid, output_grid = _pair_from_codes(codes, colors)
                key = _grid_key(input_grid)
                if key in seen_inputs or _is_source_pair(input_grid, output_grid):
                    continue
                if section == "train" and index == 1 and not presentation["shared_colors"]:
                    prior = _analyze_pair(
                        puzzle["train"][0]["input"],
                        puzzle["train"][0]["output"],
                    )["roles"]
                    if colors == prior:
                        continue
                seen_inputs.add(key)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                if section == "train":
                    demonstrated.update(int(x) for x in np.unique(codes))
                break
            else:
                raise AssertionError("could not construct a unique pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    total_height, width = input_grid.shape
    _check(total_height % 3 == 0, "input must split into three panels")
    height = total_height // 3
    _check(output_grid.shape == (height, width), "output must have one-panel shape")
    panels = (
        input_grid[:height], input_grid[height:2 * height], input_grid[2 * height:],
    )
    panel_sets = [set(int(x) for x in np.unique(panel)) for panel in panels]
    _check(all(len(values) == 2 for values in panel_sets), "each panel needs background and foreground")
    shared = panel_sets[0] & panel_sets[1] & panel_sets[2]
    _check(len(shared) == 1, "panels need one common background")
    background = next(iter(shared))
    foregrounds = [next(iter(values - {background})) for values in panel_sets]
    _check(len(set(foregrounds)) == 3, "foreground colors must be distinct")
    top, middle, bottom = foregrounds
    top_mask, middle_mask, bottom_mask = (panel != background for panel in panels)
    codes = 4 * top_mask.astype(np.int8) + 2 * middle_mask.astype(np.int8) + bottom_mask.astype(np.int8)
    expected = np.full((height, width), background, dtype=np.int8)
    expected[middle_mask] = middle
    expected[top_mask] = top
    expected[bottom_mask] = bottom
    _check(np.array_equal(output_grid, expected), "incorrect bottom > top > middle composite")
    _check(
        set(int(x) for x in np.unique(input_grid)) == {background, top, middle, bottom},
        "input must contain four distinct roles",
    )
    _check(bool(np.any(np.isin(codes, (3, 5, 6, 7)))), "pair must include an overlap")
    return {"shape": (height, width), "roles": (background, top, middle, bottom), "codes": codes}


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _check(all(isinstance(puzzle[part], list) for part in ("train", "test")), "pair lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "pair counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    analyses: dict[str, list[dict]] = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for grid in (pair["input"], pair["output"]):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grids must be 2D arrays")
                _check(all(1 <= side <= 30 for side in grid.shape), "grid dimensions")
                _check(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _check(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _check(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _check(not _is_source_pair(pair["input"], pair["output"]), "source pair copied")
            analyses[section].append(_analyze_pair(pair["input"], pair["output"]))

    # The order is the only learned convention. The seven discriminating
    # occupancy cases identify all four output roles and all three priorities.
    demonstrated = set().union(*(set(int(x) for x in np.unique(a["codes"])) for a in analyses["train"]))
    _check(set(range(7)) <= demonstrated, "training omits an exclusive or pairwise case")


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (section, tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[section]))
        for section in ("train", "test")
    )


def _predict_with_order(input_grid: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    height = input_grid.shape[0] // 3
    panels = (input_grid[:height], input_grid[height:2 * height], input_grid[2 * height:])
    colors = [set(int(x) for x in np.unique(panel)) for panel in panels]
    background = next(iter(colors[0] & colors[1] & colors[2]))
    foregrounds = [next(iter(values - {background})) for values in colors]
    output = np.full(panels[0].shape, background, dtype=np.int8)
    for role in reversed(order):
        output[panels[role] != background] = foregrounds[role]
    return output


def self_test() -> None:
    explicit = ((101, 2, 1), (102, 6, 2), (103, 8, 4), (104, 10, 1), (105, 10, 4))
    for seed, train_count, test_count in explicit:
        puzzle = generate_puzzle(seed, train_count, test_count)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count), "explicit counts")
    partial = ((201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None))
    for seed, train_count, test_count in partial:
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _check(len(puzzle["train"]) == train_count, "partial train count")
        if test_count is not None:
            _check(len(puzzle["test"]) == test_count, "partial test count")
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": True}, {"num_train": 4.0},
    ):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    first = generate_puzzle(123456, 7, 3)
    _check(_puzzle_fingerprint(first) == _puzzle_fingerprint(generate_puzzle(123456, 7, 3)), "nondeterministic")
    batch = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_puzzle_fingerprint(p) for p in batch}) == 42, "whole-puzzle duplicates")
    _check(len({len(p["train"]) for p in batch}) >= 4, "train count diversity")
    _check(len({len(p["test"]) for p in batch}) >= 3, "test count diversity")
    shapes, palettes, layouts = set(), set(), set()
    varied_shapes = varied_palettes = shared_shapes = shared_palettes = both_varied = 0
    sparse = dense = unbalanced = occluded = train_without_all = 0
    train_blocks = test_blocks = tests_without_blocks = withheld = 0
    mask_isolates = mask_blocks = mask_runs = holes = source_like = 0
    collision_isolates = collision_blocks = collision_runs = 0
    for puzzle in batch:
        orders = (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        )
        train_copies = [
            (pair["input"].copy(), pair["output"].copy())
            for pair in puzzle["train"]
        ]
        matching = [
            order for order in orders
            if all(np.array_equal(_predict_with_order(inp, order), out) for inp, out in train_copies)
        ]
        _check(matching == [(2, 0, 1)], "training does not identify the precedence")
        for pair in puzzle["test"]:
            prediction = _predict_with_order(pair["input"].copy(), matching[0])
            _check(np.array_equal(prediction, pair["output"]), "train-only prediction failed")
        train_codes = []
        local_shapes, local_palettes = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                codes = info["codes"]
                shapes.add(info["shape"])
                palettes.add(info["roles"])
                local_shapes.add(info["shape"])
                local_palettes.add(info["roles"])
                layouts.add((info["shape"], tuple(int(x) for x in codes.flat)))
                densities = [float(np.mean((codes & bit) != 0)) for bit in (4, 2, 1)]
                masks = [(codes & bit) != 0 for bit in (4, 2, 1)]
                sparse += any(value <= 0.23 for value in densities)
                dense += any(value >= 0.77 for value in densities)
                unbalanced += max(densities) - min(densities) >= 0.35
                occluded += any(color not in pair["output"] for color in info["roles"][1:])
                collision = np.isin(codes, (3, 5, 6, 7))
                collision_isolates += bool(np.any(_isolated(collision)))
                collision_blocks += _has_block(collision)
                collision_runs += _has_run(collision)
                isolated_counts = [int(_isolated(mask).sum()) for mask in masks]
                source_like += all(0.36 <= density <= 0.72 for density in densities) and all(
                    count >= 2 for count in isolated_counts
                )
                for mask, count in zip(masks, isolated_counts):
                    mask_isolates += count >= 3
                    mask_blocks += _has_block(mask)
                    mask_runs += _has_run(mask)
                    holes += bool(np.any(
                        ~mask[1:-1, 1:-1]
                        & mask[:-2, 1:-1] & mask[2:, 1:-1]
                        & mask[1:-1, :-2] & mask[1:-1, 2:]
                    ))
                if section == "train":
                    train_codes.append(codes)
                    train_without_all += len(set(int(x) for x in np.unique(codes))) < 8
                    train_blocks += _has_triple_block(codes)
                else:
                    test_blocks += _has_triple_block(codes)
                    tests_without_blocks += not _has_triple_block(codes)
        varied_shapes += len(local_shapes) > 1
        varied_palettes += len(local_palettes) > 1
        shared_shapes += len(local_shapes) == 1
        shared_palettes += len(local_palettes) == 1
        both_varied += len(local_shapes) > 1 and len(local_palettes) > 1
        if all(not bool(np.any(codes == 7)) for codes in train_codes):
            withheld += any(bool(np.any(_analyze_pair(p["input"], p["output"])["codes"] == 7)) for p in puzzle["test"])
    _check(len(shapes) >= 25 and len(palettes) >= 40 and len(layouts) >= 200, "structural diversity")
    _check(min(varied_shapes, varied_palettes, shared_shapes, shared_palettes) >= 4, "presentation diversity")
    _check(both_varied >= 4, "combined size and palette variation absent")
    _check(min(sparse, dense, unbalanced, occluded) >= 3, "mask diversity")
    _check(min(mask_isolates, mask_blocks, mask_runs, holes) >= 10, "mask geometries collapsed")
    _check(min(collision_isolates, collision_blocks, collision_runs) >= 10, "collision arrangements collapsed")
    _check(source_like >= 5, "source-like scattered pairs absent")
    _check(train_without_all >= 30, "training still balanced per pair")
    _check(test_blocks >= 1 and tests_without_blocks >= 10, "triple-block regime collapsed")
    _check(withheld >= 1, "pairwise-to-triple shift absent")
