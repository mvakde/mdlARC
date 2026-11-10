import numpy as np


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(_pair_key(pair) for pair in puzzle[split]),
        )
        for split in ("train", "test")
    )


def _raw_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_RAW = (
    (
        (
            "08000000000",
            "00800080008",
            "00000000088",
            "00800000800",
            "80500000500",
            "00500880500",
            "00500000500",
            "00808000800",
            "08000000880",
        ),
        (
            "8000008",
            "5000005",
            "5008805",
            "5000005",
            "8080008",
        ),
    ),
    (
        (
            "000000000",
            "080008008",
            "050005000",
            "050805080",
            "050005000",
            "080008000",
            "000000000",
            "000800080",
            "080000000",
        ),
        (
            "80008",
            "50005",
            "50805",
            "50005",
            "80008",
        ),
    ),
    (
        (
            "0000000008800",
            "0000000000000",
            "0008000800000",
            "0005000508000",
            "0085080500000",
            "0005000580000",
            "0085080500000",
            "0008000800800",
            "0000000000800",
            "0088000800000",
            "0000000080000",
        ),
        (
            "80008",
            "50005",
            "50805",
            "50005",
            "50805",
            "80008",
        ),
    ),
    (
        (
            "8000000000000",
            "0000000000000",
            "0000000080000",
            "8008000000000",
            "5805000000008",
            "5005000000000",
            "5005000000000",
            "5005008008000",
            "5085800000008",
            "5005080000000",
            "5805000000808",
            "5005000800000",
            "8008000000000",
            "0000000000008",
        ),
        (
            "8008",
            "5805",
            "5005",
            "5005",
            "5005",
            "5085",
            "5005",
            "5805",
            "5005",
            "8008",
        ),
    ),
)


_SOURCE_PAIR_KEYS = frozenset(
    (
        _grid_key(_raw_grid(input_rows)),
        _grid_key(_raw_grid(output_rows)),
    )
    for input_rows, output_rows in _SOURCE_RAW
)


def _optional_count(
    name: str,
    value: int | None,
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
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _optional_count("num_train", num_train, 2, 10)
    test_count = _optional_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % number_of_train_choices
        )
    elif train_count is None:
        assert test_count is not None
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        largest_test_count = min(4, train_count - 1)
        test_count = int(rng.integers(1, largest_test_count + 1))

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Only the crop rule is shared; visual roles are visible in each pair."""
    return {"base_colors": tuple(int(c) for c in rng.choice(10, 3, replace=False))}


def _palette_plan(rng, base, count, varying):
    if not varying:
        return [base] * count
    palettes = [base]
    for _ in range(1, count):
        for _attempt in range(100):
            colors = tuple(int(c) for c in rng.choice(10, 3, replace=False))
            if all(a != b for a, b in zip(colors, palettes[-1])):
                palettes.append(colors)
                break
        else:
            raise AssertionError("palette construction failed")
    return palettes


def _orientation_plan(seed, train_count, test_count):
    mode = (seed // 3) % 5
    if mode == 0:
        return ["vertical"] * (train_count + test_count)
    if mode == 1:
        return ["horizontal"] * (train_count + test_count)
    if mode == 2:
        return ["vertical" if i % 2 == 0 else "horizontal"
                for i in range(train_count)] + [
                    "horizontal" if i % 2 == 0 else "vertical"
                    for i in range(test_count)]
    if mode == 3:
        return ["vertical"] * train_count + ["horizontal"] * test_count
    return ["horizontal"] * train_count + ["vertical"] * test_count


def _window_size(rng, regime):
    categories = ("small", "square", "long", "wide", "large", "free")
    probabilities = ((.15, .19, .17, .17, .12, .20) if regime == "source_train"
                     else (.12, .13, .24, .17, .14, .20))
    category = str(rng.choice(categories, p=probabilities))
    if category == "small":
        axis, separation = int(rng.integers(4, 8)), int(rng.integers(3, 8))
    elif category == "square":
        axis = int(rng.integers(5, 17))
        separation = max(3, min(24, axis + int(rng.integers(-2, 3))))
    elif category == "long":
        axis, separation = int(rng.integers(9, 25)), int(rng.integers(3, 9))
    elif category == "wide":
        axis, separation = int(rng.integers(4, 10)), int(rng.integers(9, 24))
    elif category == "large":
        axis, separation = int(rng.integers(14, 25)), int(rng.integers(12, 25))
    else:
        axis, separation = int(rng.integers(4, 25)), int(rng.integers(3, 25))
    return axis, separation


def _slack(rng, dimension, minimum):
    maximum = min(30 - dimension, 12)
    if maximum < minimum:
        return None
    if maximum == minimum:
        return minimum
    if rng.random() < .22:
        return maximum
    return int(rng.integers(minimum, maximum + 1))


def _placement(rng, height, width, orientation, source_like):
    if source_like:
        choices = ("left", "right") if orientation == "vertical" else ("top", "bottom")
        mode = str(rng.choice(choices))
    else:
        mode = str(rng.choice(("interior", "top", "bottom", "left", "right",
                               "corner", "full_height", "full_width", "free"),
                              p=(.16, .09, .09, .09, .09, .15, .10, .10, .13)))
    if mode == "interior":
        min_h, min_w = 2, 2
    elif mode in ("top", "bottom"):
        min_h, min_w = 1, 2
    elif mode in ("left", "right"):
        min_h, min_w = 2, 1
    elif mode == "corner":
        min_h, min_w = 1, 1
    elif mode == "full_height":
        min_h, min_w = 0, 1
    elif mode == "full_width":
        min_h, min_w = 1, 0
    else:
        min_h, min_w = 0, 0
    extra_h = 0 if mode == "full_height" else _slack(rng, height, min_h)
    extra_w = 0 if mode == "full_width" else _slack(rng, width, min_w)
    if extra_h is None or extra_w is None:
        # This only affects a near-maximum crop and preserves a real crop.
        mode = "free"
        extra_h = _slack(rng, height, 0)
        extra_w = _slack(rng, width, 0)
    if extra_h == extra_w == 0:
        if width < 30:
            extra_w = 1
        else:
            extra_h = 1
    top = int(rng.integers(0, extra_h + 1))
    left = int(rng.integers(0, extra_w + 1))
    if mode == "interior":
        top = int(rng.integers(1, extra_h))
        left = int(rng.integers(1, extra_w))
    elif mode == "top":
        top = 0
        left = int(rng.integers(1, extra_w))
    elif mode == "bottom":
        top = extra_h
        left = int(rng.integers(1, extra_w))
    elif mode == "left":
        left = 0
        top = int(rng.integers(1, extra_h))
    elif mode == "right":
        left = extra_w
        top = int(rng.integers(1, extra_h))
    elif mode == "corner":
        top = 0 if rng.random() < .5 else extra_h
        left = 0 if rng.random() < .5 else extra_w
    return height + extra_h, width + extra_w, top, left


def _marks(rng, positions, shape, density, arrangement, anchor=None):
    n = len(positions)
    if n == 0 or density == "empty":
        return np.empty((0, 2), dtype=int)
    if density == "sparse":
        count = int(rng.integers(1, min(n, 12 if anchor is not None else 7) + 1))
    elif density == "medium":
        low, high = (.05, .20) if anchor is not None else (.10, .35)
        count = max(1, int(round(n * rng.uniform(low, high))))
    else:
        low, high = (.18, .42) if anchor is not None else (.30, .65)
        count = max(1, int(round(n * rng.uniform(low, high))))
    count = min(count, n - 1) if anchor is None else min(count, n)
    if count <= 0:
        return np.empty((0, 2), dtype=int)
    if arrangement == "scatter":
        selected = rng.choice(n, count, replace=False)
    else:
        if arrangement == "adjacent" and anchor is not None:
            top, left, bottom, right = anchor
            distances = (np.maximum.reduce((top - positions[:, 0],
                                            positions[:, 0] - bottom,
                                            left - positions[:, 1],
                                            positions[:, 1] - right,
                                            np.zeros(n, dtype=int))))
            scores = distances + rng.random(n) * 2.0
        elif arrangement == "line":
            row, col = positions[int(rng.integers(n))]
            if rng.random() < .5:
                scores = abs(positions[:, 0] - row) * 5 + abs(positions[:, 1] - col) * .1
            else:
                scores = abs(positions[:, 1] - col) * 5 + abs(positions[:, 0] - row) * .1
            scores = scores + rng.random(n) * .05
        else:
            center = positions[int(rng.integers(n))]
            scores = (abs(positions[:, 0] - center[0])
                      + abs(positions[:, 1] - center[1]) + rng.random(n) * 1.5)
            if arrangement == "mixed":
                scores[rng.choice(n, max(1, n // 3), replace=False)] *= .25
        selected = np.argpartition(scores, count - 1)[:count]
    return positions[selected]


def _runs_with_caps(work, rail):
    height, width = work.shape
    grouped = {}
    for col in range(width):
        row = 0
        while row < height:
            if int(work[row, col]) != rail:
                row += 1
                continue
            start = row
            while row < height and int(work[row, col]) == rail:
                row += 1
            stop = row
            if stop - start < 2 or start == 0 or stop == height:
                continue
            marker = int(work[start - 1, col])
            if marker == rail or int(work[stop, col]) != marker:
                continue
            grouped.setdefault((start, stop, marker), []).append(col)
    return grouped


def _window_candidates(input_grid):
    colors = {int(c) for c in np.unique(input_grid)}
    if len(colors) != 3:
        return []
    candidates = []
    for orientation in ("vertical", "horizontal"):
        work = input_grid if orientation == "vertical" else input_grid.T
        for rail in colors:
            for (start, stop, marker), columns in _runs_with_caps(work, rail).items():
                background = next(iter(colors - {rail, marker}))
                for i, left in enumerate(columns):
                    for right in columns[i + 1:]:
                        if right - left < 2:
                            continue
                        crop = work[start - 1:stop + 1, left:right + 1]
                        if np.count_nonzero(crop == rail) != 2 * (stop - start):
                            continue
                        if not np.any(crop == background):
                            continue
                        if orientation == "vertical":
                            bounds = (start - 1, left, stop, right)
                        else:
                            bounds = (left, start - 1, right, stop)
                        candidates.append((bounds, orientation, (background, rail, marker)))
    return candidates


def _locate_window(input_grid):
    candidates = _window_candidates(input_grid)
    assert len(candidates) == 1, f"expected one capped rail pair, got {len(candidates)}"
    return candidates[0]


def _generate_pair(rng, regime, palette, orientation, source_like=False, rail_decoy=False):
    background, rail, marker = palette
    axis, separation = _window_size(rng, regime)
    if source_like:
        axis = int(rng.integers(9, 18))
        separation = int(rng.integers(3, 6))
    height, width = (axis, separation) if orientation == "vertical" else (separation, axis)
    canvas_h, canvas_w, top, left = _placement(rng, height, width, orientation, source_like)
    frame = np.full((height, width), background, dtype=np.int8)
    frame[0, 0] = frame[0, -1] = frame[-1, 0] = frame[-1, -1] = marker
    if orientation == "vertical":
        frame[1:-1, 0] = frame[1:-1, -1] = rail
    else:
        frame[0, 1:-1] = frame[-1, 1:-1] = rail
    payload_positions = np.argwhere(frame == background)
    density = str(rng.choice(("empty", "sparse", "medium", "dense"),
                             p=(.12, .40, .30, .18)))
    arrangement = str(rng.choice(("scatter", "cluster", "line", "mixed")))
    payload = _marks(rng, payload_positions, frame.shape, density, arrangement)
    if len(payload):
        frame[payload[:, 0], payload[:, 1]] = marker
    grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
    grid[top:top + height, left:left + width] = frame
    outside = np.ones(grid.shape, dtype=bool)
    outside[top:top + height, left:left + width] = False
    outside_positions = np.argwhere(outside)
    noise_density = str(rng.choice(("empty", "sparse", "medium", "dense"),
                                   p=(.13, .40, .32, .15)))
    noise_arrangement = str(rng.choice(("scatter", "cluster", "line", "adjacent", "mixed")))
    noise = _marks(rng, outside_positions, grid.shape, noise_density,
                   noise_arrangement, (top, left, top + height - 1, left + width - 1))
    if len(noise):
        grid[noise[:, 0], noise[:, 1]] = marker
    if rail_decoy:
        available = np.argwhere(outside & (grid == background))
        if len(available):
            count = min(len(available), int(rng.integers(1, 4)))
            chosen = available[rng.choice(len(available), count, replace=False)]
            grid[chosen[:, 0], chosen[:, 1]] = rail
    output = grid[top:top + height, left:left + width].copy()
    return {"input": grid, "output": output}


def _pair_analysis(pair):
    grid, output = pair["input"], pair["output"]
    (top, left, bottom, right), orientation, palette = _locate_window(grid)
    assert np.array_equal(output, grid[top:bottom + 1, left:right + 1])
    background, rail, marker = palette
    height, width = output.shape
    payload_mask = output == marker
    payload_mask[0, 0] = payload_mask[0, -1] = False
    payload_mask[-1, 0] = payload_mask[-1, -1] = False
    outside = np.ones(grid.shape, dtype=bool)
    outside[top:bottom + 1, left:right + 1] = False
    return {"bounds": (top, left, bottom, right), "orientation": orientation,
            "palette": palette, "axis": height if orientation == "vertical" else width,
            "separation": width if orientation == "vertical" else height,
            "payload": int(payload_mask.sum()),
            "noise": int(np.count_nonzero((grid == marker) & outside)),
            "rail_decoys": int(np.count_nonzero((grid == rail) & outside))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert pair["input"].shape != pair["output"].shape
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_pair_analysis(pair))
    # Cropping is the sole convention. Each training example shows the exact crop,
    # and each test input has exactly one locally capped rail pair.
    assert len({tuple(pair["output"].shape) for pair in train}) >= 2


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    count = train_count + test_count
    varying_palette = seed % 4 != 0
    palettes = _palette_plan(rng, parameters["base_colors"], count, varying_palette)
    orientations = _orientation_plan(seed, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_pairs, train_shapes = set(), set(), set()
    for index in range(count):
        split = "train" if index < train_count else "test"
        regime = "source_train" if split == "train" else "target_test"
        source_like = split == "test" and rng.random() < .22
        rail_decoy = (seed % 7 == 2 and index == 0) or rng.random() < .04
        for attempt in range(96):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _generate_pair(pair_rng, regime, palettes[index], orientations[index],
                                  source_like, rail_decoy if attempt < 64 else False)
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            if input_key in seen_inputs or pair_key in seen_pairs or pair_key in _SOURCE_PAIR_KEYS:
                continue
            if split == "train" and index == train_count - 1 and len(train_shapes) == 1 \
                    and tuple(pair["output"].shape) in train_shapes:
                continue
            if len(_window_candidates(pair["input"])) != 1:
                continue
            puzzle[split].append(pair)
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_shapes.add(tuple(pair["output"].shape))
            break
        else:
            raise RuntimeError("could not construct an unambiguous unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts, palettes, orientations = set(), set(), set(), set()
    within_palette, within_orientation = 0, 0
    payloads, noises, decoys, placements, axes, separations = set(), set(), set(), set(), set(), set()
    shapes, canvas_shapes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        info = [_pair_analysis(p) for split in ("train", "test") for p in puzzle[split]]
        within_palette += len({x["palette"] for x in info}) > 1
        within_orientation += len({x["orientation"] for x in info}) > 1
        for x, pair in zip(info, puzzle["train"] + puzzle["test"]):
            palettes.add(x["palette"])
            orientations.add(x["orientation"])
            payloads.add(x["payload"])
            noises.add(x["noise"])
            decoys.add(x["rail_decoys"])
            placements.add(x["bounds"][0:2])
            axes.add(x["axis"])
            separations.add(x["separation"])
            shapes.add(tuple(pair["output"].shape))
            canvas_shapes.add(tuple(pair["input"].shape))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 40 and orientations == {"vertical", "horizontal"}
    assert within_palette >= 20 and within_orientation >= 15
    assert 0 in payloads and max(payloads) >= 20
    assert 0 in noises and max(noises) >= 20
    assert max(decoys) > 0
    assert min(axes) == 4 and max(axes) >= 20
    assert min(separations) == 3 and max(separations) >= 20
    assert len(shapes) >= 40 and len(canvas_shapes) >= 40
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        p = generate_puzzle(10000 + index, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(20000 + index, train, test)
        assert (train is None or len(p["train"]) == train)
        assert (test is None or len(p["test"]) == test)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(30000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair count accepted")
