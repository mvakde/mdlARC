"""Generate directed sequences of rectangular bands for the ARC recoloring rule."""

import numpy as np


_SOURCE_INPUT_ROWS = frozenset({
    ("1110", "1110", "8888", "0011", "0011", "8888", "8888",
     "0110", "8800", "1111", "0880", "0880", "1111", "8880",
     "8880", "0111", "0111", "8880", "0110", "8888"),
    ("08880", "08880", "11100", "08880", "08880", "11111",
     "08888", "08888", "11110", "11110", "08880", "01111",
     "01111", "88800", "00011", "88800", "88800", "00110",
     "00110"),
    ("11100", "08888", "11110", "08800", "00111", "00111",
     "88880", "01100", "01100", "88888", "88888", "11100",
     "08880", "00011"),
    ("110", "110", "888", "888", "011", "011", "880", "111",
     "880", "880", "111", "111", "111", "888", "888"),
    ("8880", "0111", "0880", "1111", "8880", "8880", "0111",
     "0880", "0880", "1111", "1111", "8888", "1100", "0888",
     "1110", "8888", "0110", "0110", "8888", "8888", "0111",
     "0111"),
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _signature(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.flat)


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_count(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _oriented(grid: np.ndarray, axis: str) -> np.ndarray:
    return grid if axis == "vertical" else grid.T


def _parse_bands(oriented: np.ndarray, background: int) -> list[tuple] | None:
    """Return maximal identical-row runs in reading-axis coordinates."""
    height, width = oriented.shape
    descriptions: list[tuple] = []
    for row in oriented:
        places = np.flatnonzero(row != background)
        if places.size == 0:
            return None
        left = int(places[0])
        right = int(places[-1]) + 1
        if right - left != places.size:
            return None
        color = int(row[left])
        if not bool(np.all(row[left:right] == color)):
            return None
        descriptions.append((tuple(int(x) for x in row), color, left, right))

    bands = []
    pos = 0
    while pos < height:
        end = pos + 1
        while end < height and descriptions[end][0] == descriptions[pos][0]:
            end += 1
        _, color, left, right = descriptions[pos]
        bands.append((pos, end, color, left, right))
        pos = end
    if not 5 <= len(bands) <= 27:
        return None
    foregrounds = {band[2] for band in bands}
    if not 2 <= len(foregrounds) <= 4:
        return None
    if foregrounds != set(int(x) for x in np.unique(oriented)) - {background}:
        return None
    if any(bands[i][2] == bands[i - 1][2] for i in range(1, len(bands))):
        return None
    return bands


def _input_structure(grid: np.ndarray) -> tuple[str, int, list[tuple]] | None:
    """The input alone must identify the reading axis and background."""
    candidates = []
    for axis in ("vertical", "horizontal"):
        oriented = _oriented(grid, axis)
        for background in (int(x) for x in np.unique(grid)):
            bands = _parse_bands(oriented, background)
            if bands is not None:
                candidates.append((axis, background, bands))
    return candidates[0] if len(candidates) == 1 else None


def _transform(
    grid: np.ndarray,
    axis: str,
    background: int,
    highlight: int,
    bands: list[tuple],
    reverse: bool,
) -> np.ndarray:
    result = grid.copy()
    oriented = _oriented(result, axis)
    count = len(bands)
    for index, (start, end, _color, left, right) in enumerate(bands):
        ordinal = count - 1 - index if reverse else index
        if ordinal % 3 == 0:
            oriented[start:end, left:right] = highlight
    return result


def _composition(
    rng: np.random.Generator,
    total: int,
    count: int,
) -> list[int]:
    if total == count:
        return [1] * count
    cuts = np.sort(rng.choice(np.arange(1, total), size=count - 1, replace=False))
    return [int(x) for x in np.diff(np.concatenate(([0], cuts, [total])))]


def _thicknesses(
    rng: np.random.Generator,
    count: int,
    force_first_thick: bool,
) -> list[int]:
    draw = float(rng.random())
    if draw < 0.11 and not force_first_thick:
        total = count
        thickness = [1] * count
    elif draw < 0.23 and count <= 15:
        factor = int(rng.integers(2, 30 // count + 1))
        thickness = [factor] * count
    else:
        if draw < 0.55:
            total = int(rng.integers(count + int(force_first_thick),
                                     min(30, count + 6) + 1))
        elif draw < 0.88:
            total = int(rng.integers(count + int(force_first_thick), 31))
        else:
            total = 30
        thickness = _composition(rng, total, count)
    if force_first_thick and thickness[0] == 1:
        donor = next(i for i, value in enumerate(thickness) if value > 1)
        thickness[donor] -= 1
        thickness[0] += 1
    return thickness


def _cross_size(rng: np.random.Generator) -> int:
    draw = float(rng.random())
    if draw < 0.25:
        return int(rng.integers(3, 6))
    if draw < 0.49:
        return int(rng.integers(6, 13))
    if draw < 0.76:
        return int(rng.integers(13, 23))
    if draw < 0.92:
        return int(rng.integers(23, 30))
    return 30


def _bar(
    rng: np.random.Generator,
    cross_size: int,
) -> tuple[int, int]:
    draw = float(rng.random())
    if draw < 0.07:
        width = 1
    elif draw < 0.17:
        width = cross_size
    elif draw < 0.45:
        width = int(rng.integers(2, min(cross_size, 5) + 1))
    else:
        width = int(rng.integers(2, cross_size + 1))
    left = int(rng.integers(0, cross_size - width + 1))
    return left, width


def _pair_colors(
    rng: np.random.Generator,
    highlight: int,
    avoid_background: int | None,
) -> tuple[int, list[int]]:
    pool = [color for color in range(10) if color != highlight]
    backgrounds = [color for color in pool if color != avoid_background]
    background = int(rng.choice(backgrounds))
    pool.remove(background)
    color_count = int(rng.choice((2, 3, 4), p=(0.57, 0.28, 0.15)))
    foregrounds = [int(x) for x in rng.choice(pool, size=color_count, replace=False)]
    return background, foregrounds


def _make_pair(
    rng: np.random.Generator,
    axis: str,
    reverse: bool,
    highlight: int,
    band_count: int,
    cross_size: int,
    avoid_background: int | None,
    force_first_thick: bool,
) -> dict:
    for _ in range(400):
        background, foregrounds = _pair_colors(rng, highlight, avoid_background)
        thicknesses = _thicknesses(rng, band_count, force_first_thick)
        colors = list(int(x) for x in rng.permutation(foregrounds))
        while len(colors) < band_count:
            choices = [x for x in foregrounds if x != colors[-1]]
            colors.append(int(rng.choice(choices)))
        bars = []
        for index in range(band_count):
            if index and rng.random() < 0.16:
                bars.append(bars[int(rng.integers(0, index))])
            else:
                bars.append(_bar(rng, cross_size))
        oriented = np.full((sum(thicknesses), cross_size), background, dtype=np.int8)
        pos = 0
        for thickness, color, (left, width) in zip(thicknesses, colors, bars):
            oriented[pos:pos + thickness, left:left + width] = color
            pos += thickness
        grid = oriented if axis == "vertical" else oriented.T.copy()
        structure = _input_structure(grid)
        if structure is None or structure[0] != axis or structure[1] != background:
            continue
        bands = structure[2]
        if len(bands) != band_count:
            continue
        output = _transform(grid, axis, background, highlight, bands, reverse)
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct an unambiguous band input")


def _train_band_counts(rng: np.random.Generator, count: int) -> list[int]:
    first = int(rng.choice((8, 11, 12, 14, 17, 20)))
    result = [first]
    while len(result) < count:
        candidate = int(rng.integers(7, 21))
        if candidate != result[0] and (candidate not in result or rng.random() < 0.4):
            result.append(candidate)
    return result


def _test_band_count(rng: np.random.Generator, maximum_train: int) -> int:
    draw = float(rng.random())
    if draw < 0.16:
        return 16
    if draw < 0.51:
        return int(rng.integers(maximum_train + 1, 28))
    if draw < 0.68:
        return maximum_train
    return int(rng.integers(5, maximum_train))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    axis = str(rng.choice(("vertical", "horizontal")))
    reverse = bool(rng.integers(0, 2))
    highlight = int(rng.integers(0, 10))
    train_bands = _train_band_counts(rng, train_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = None
    first_cross = None

    for split in ("train", "test"):
        number = train_count if split == "train" else test_count
        for index in range(number):
            band_count = (train_bands[index] if split == "train" else
                          _test_band_count(rng, max(train_bands)))
            for _ in range(400):
                cross = _cross_size(rng)
                if split == "train" and index == 1 and cross == first_cross:
                    continue
                pair = _make_pair(
                    rng, axis, reverse, highlight, band_count, cross,
                    first_background if split == "train" and index == 1 else None,
                    split == "train" and index == 0,
                )
                signature = _signature(pair["input"])
                if signature in seen or _rows(pair["input"]) in _SOURCE_INPUT_ROWS:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_background = _input_structure(pair["input"])[1]
                    first_cross = cross
                break
            else:
                raise RuntimeError("could not produce a unique non-source pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair: dict) -> dict:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must contain exactly input and output")
    for name in ("input", "output"):
        grid = pair[name]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 "grid must be a two-dimensional NumPy array")
        _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
        _require(all(1 <= length <= 30 for length in grid.shape),
                 "grid dimensions must be within 1 through 30")
        _require(bool(np.all((grid >= 0) & (grid <= 9))),
                 "grid has a non-ARC color")
    grid, output = pair["input"], pair["output"]
    _require(grid.shape == output.shape, "output shape changed")
    structure = _input_structure(grid)
    _require(structure is not None, "input axis or background is ambiguous")
    axis, background, bands = structure
    input_colors = set(int(x) for x in np.unique(grid))
    output_colors = set(int(x) for x in np.unique(output))
    introduced = output_colors - input_colors
    _require(len(introduced) == 1, "output must introduce one highlight")
    highlight = next(iter(introduced))
    directions = {
        reverse for reverse in (False, True)
        if np.array_equal(
            _transform(grid, axis, background, highlight, bands, reverse),
            output,
        )
    }
    _require(bool(directions), "output violates every-third-band recoloring")
    return {
        "axis": axis, "background": background, "bands": bands,
        "highlight": highlight, "directions": directions,
        "foregrounds": input_colors - {background},
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            signature = _signature(pair["input"])
            pair_signature = (signature, _signature(pair["output"]))
            _require(signature not in seen_inputs, "duplicate input")
            _require(pair_signature not in seen_pairs, "duplicate pair")
            _require(_rows(pair["input"]) not in _SOURCE_INPUT_ROWS,
                     "source input copied")
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            count = len(info["bands"])
            if split == "train":
                _require(7 <= count <= 20, "training band count outside regime")
            else:
                _require(5 <= count <= 27, "test band count outside regime")
            infos[split].append(info)

    highlights = {info["highlight"] for split in infos for info in infos[split]}
    _require(len(highlights) == 1, "highlight changes within puzzle")
    axes = {info["axis"] for split in infos for info in infos[split]}
    _require(len(axes) == 1, "reading axis changes within puzzle")
    directions = set.intersection(*(info["directions"] for info in infos["train"]))
    _require(len(directions) == 1, "training does not establish reading direction")
    direction = next(iter(directions))
    for info in infos["test"]:
        _require(direction in info["directions"],
                 "test uses a different reading direction")
    _require(len({len(info["bands"]) for info in infos["train"]}) >= 2,
             "training lacks sequence-length variation")
    _require(len({info["background"] for info in infos["train"]}) >= 2,
             "training lacks local background variation")
    axis = next(iter(axes))
    cross_sizes = {
        _oriented(pair["input"], axis).shape[1] for pair in puzzle["train"]
    }
    _require(len(cross_sizes) >= 2, "training lacks transverse-size variation")
    _require(
        any(info["bands"][0][1] - info["bands"][0][0] > 1
            for info in infos["train"]),
        "training never demonstrates that thickness is counted once",
    )


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a[name], b[name])
            for a, b in zip(first[split], second[split])
            for name in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(_puzzles_equal(generate_puzzle(123456, 7, 3),
                           generate_puzzle(123456, 7, 3)),
             "explicit generation is nondeterministic")
    _require(_puzzles_equal(generate_puzzle(654321), generate_puzzle(654321)),
             "automatic generation is nondeterministic")
    _require(not _puzzles_equal(generate_puzzle(9001, 6, 2),
                               generate_puzzle(9002, 6, 2)),
             "different seeds gave the same puzzle")
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "too few training pair counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "too few test pair counts")
    directions = set()
    train_remainders, test_remainders = set(), set()
    train_counts, test_counts = set(), set()
    shapes, layouts, grids = set(), set(), set()
    local_palette_changes = 0
    singleton_free = full_free = one_row = thick = multicolor = 0
    compact = large = very_thick = rich_combination = 0
    source_like = source_continuations = 0
    left_edge = right_edge = interior = shifted_bars = 0
    relative_lengths = set()
    total_pairs = 0
    for puzzle in puzzles:
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        orientation = train_info[0]["axis"]
        reverse = next(iter(set.intersection(
            *(info["directions"] for info in train_info)
        )))
        directions.add((orientation, reverse))
        local_palette_changes += len({
            (info["background"], tuple(sorted(info["foregrounds"])))
            for info in train_info
        }) > 1
        max_train = max(len(info["bands"]) for info in train_info)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                count = len(info["bands"])
                (train_remainders if split == "train" else test_remainders).add(count % 3)
                (train_counts if split == "train" else test_counts).add(count)
                if split == "test":
                    relative_lengths.add((count > max_train) - (count < max_train))
                total_pairs += 1
                oriented = _oriented(pair["input"], info["axis"])
                shape = pair["input"].shape
                shapes.add(shape)
                grids.add(_signature(pair["input"]))
                layouts.add((shape, tuple(
                    (end - start, left, right - left)
                    for start, end, _color, left, right in info["bands"]
                )))
                widths = [right - left for _, _, _, left, right in info["bands"]]
                lefts = [left for _, _, _, left, right in info["bands"]]
                singleton_free += 1 not in widths
                full_free += oriented.shape[1] not in widths
                one_row += all(end - start == 1 for start, end, *_ in info["bands"])
                thick += any(end - start >= 5 for start, end, *_ in info["bands"])
                multicolor += len(info["foregrounds"]) >= 3
                compact += max(shape) <= 12
                large += 30 in shape
                very_thick += any(end - start >= 9 for start, end, *_ in info["bands"])
                rich_combination += (
                    len(info["foregrounds"]) >= 3
                    and oriented.shape[1] >= 10
                    and any(end - start > 1 for start, end, *_ in info["bands"])
                    and reverse
                )
                source_like += (
                    len(info["foregrounds"]) == 2
                    and oriented.shape[1] in (3, 4, 5)
                    and 14 <= oriented.shape[0] <= 22
                    and info["axis"] == "vertical"
                )
                if split == "test":
                    source_continuations += count == 16
                left_edge += 0 in lefts
                right_edge += any(right == oriented.shape[1]
                                  for _, _, _, left, right in info["bands"])
                interior += any(left > 0 and right < oriented.shape[1]
                                for _, _, _, left, right in info["bands"])
                shifted_bars += len(set(lefts)) > 1
    _require(len(directions) == 4, "directed axes lack coverage")
    _require(train_remainders == {0, 1, 2} and test_remainders == {0, 1, 2},
             "sequence endings lack coverage")
    _require(min(train_counts) <= 8 and max(train_counts) >= 18
             and min(test_counts) <= 8 and max(test_counts) > 16,
             "sequence lengths are too narrow")
    _require(relative_lengths == {-1, 0, 1}, "test length regimes lack coverage")
    _require(local_palette_changes >= 40, "local palette does not vary within puzzles")
    _require(singleton_free >= 10 and full_free >= 10,
             "singleton or full-width bars are effectively compulsory")
    _require(one_row >= 5 and thick >= 5,
             "band thickness modes lack coverage")
    _require(multicolor >= 10, "irregular multicolor sequences are too rare")
    _require(compact >= 5 and large >= 10 and very_thick >= 3,
             "canvas or thickness extremes lack coverage")
    _require(rich_combination >= 5,
             "compatible geometry, palette, and direction axes do not combine")
    _require(source_like >= 5 and source_continuations >= 3,
             "source-like cases or sixteen-band continuations are too rare")
    _require(min(left_edge, right_edge, interior, shifted_bars) >= 10,
             "bar offsets and edge contact lack coverage")
    _require(len(shapes) >= 60 and len(layouts) >= 120
             and len(grids) >= int(0.9 * total_pairs),
             "structural variety is too low")
    for seed, train_count, test_count in (
        (700, 2, 1), (701, 10, 1), (702, 10, 4),
        (710, None, 1), (711, None, 4),
        (712, 2, None), (713, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count ignored")
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 2.0, "num_test": 1},
    ):
        try:
            generate_puzzle(800, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal pair counts accepted: {kwargs}")
