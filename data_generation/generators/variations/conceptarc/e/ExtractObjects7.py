"""Construct divider, crop, and concatenate ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    "000000030000000\n011111030444440\n011111030400040\n"
    "011111030400040\n011111030400040\n011111030444440\n"
    "000000030000000\n000000030000000",
    "0000000\n0003000\n0030300\n0300030\n0030300\n0003000\n"
    "6666666\n6666666\n0000000\n0055555\n0055555\n0055555\n"
    "0055555\n0055555\n0000000",
    "07700\n07700\n00000\n88888\n00000\n00000\n00220\n"
    "00220\n00000\n00000",
    "000005503333\n111105503003\n100105503003\n100105503003\n"
    "111105500000\n000005500000",
    "000000\n066660\n066660\n066660\n000000\n000000\n"
    "999999\n000000\n066660\n066660\n066660\n000000\n"
    "000000\n000000\n000000",
}

_STYLES = (
    "solid", "frame", "diamond", "corners", "checker", "x",
    "plus", "sparse", "holes", "irregular",
)


def _grid_text(grid: np.ndarray) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def _count(value: int, low: int, high: int, label: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}")
    return value


def _pattern(rng: np.random.Generator, height: int, width: int,
             style: str, color: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if style == "solid":
        mask[:] = True
    elif style == "frame":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif style == "corners":
        mask[0, 0] = mask[0, -1] = True
        mask[-1, 0] = mask[-1, -1] = True
        if height > 3 and width > 3 and rng.random() < 0.45:
            mask[int(rng.integers(1, height - 1)),
                 int(rng.integers(1, width - 1))] = True
    elif style == "x":
        for row in range(height):
            col = int(round(row * (width - 1) / max(1, height - 1)))
            mask[row, col] = mask[row, width - 1 - col] = True
    elif style == "plus":
        mask[int(rng.integers(height)), :] = True
        mask[:, int(rng.integers(width))] = True
    elif style == "checker":
        rows, cols = np.indices((height, width))
        mask = (rows + cols + int(rng.integers(2))) % 2 == 0
    elif style == "diamond":
        for row in range(height):
            distance = abs(2 * row / max(1, height - 1) - 1)
            col = int(round(distance * (width - 1) / 2))
            mask[row, col] = mask[row, width - 1 - col] = True
    elif style == "sparse":
        mask = rng.random((height, width)) < float(rng.uniform(0.16, 0.36))
        if height >= 3 and width >= 3 and mask.all():
            mask[height // 2, width // 2] = False
    elif style == "holes":
        mask[:] = True
        if height >= 3 and width >= 3:
            interior = rng.random((height - 2, width - 2)) < float(rng.uniform(0.35, 0.8))
            if not interior.any():
                interior[int(rng.integers(height - 2)),
                         int(rng.integers(width - 2))] = True
            mask[1:-1, 1:-1][interior] = False
        else:
            mask[int(rng.integers(height)), int(rng.integers(width))] = False
    elif style == "irregular":
        mask = rng.random((height, width)) < float(rng.uniform(0.38, 0.72))
    else:
        raise ValueError("unknown pattern style")

    # Force a tight box while leaving all interior cells, including holes, intact.
    if not mask[0].any():
        mask[0, int(rng.integers(width))] = True
    if not mask[-1].any():
        mask[-1, int(rng.integers(width))] = True
    if not mask[:, 0].any():
        mask[int(rng.integers(height)), 0] = True
    if not mask[:, -1].any():
        mask[int(rng.integers(height)), -1] = True
    if style != "solid" and height >= 3 and width >= 3 and mask.all():
        mask[height // 2, width // 2] = False
    return np.where(mask, color, 0).astype(np.int8)


def _size(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    # Both splits share sizes. Tests have a modest extra chance of size 8.
    upper = 9 if regime == "target_test" else 8
    height = int(rng.integers(2, upper))
    width = int(rng.integers(2, upper))
    return height, width


def _compose(rng: np.random.Generator, first: np.ndarray,
             second: np.ndarray, orientation: str, thickness: int,
             divider: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = first.shape
    # Padding parallel to the divider is shared in total, but each offset is
    # independent. One spare row/column prevents a motif becoming a full line.
    extra = int(rng.integers(1, 8))
    offset_a = int(rng.integers(0, extra + 1))
    offset_b = int(rng.integers(0, extra + 1))
    margins = [int(v) for v in rng.integers(0, 4, size=4)]
    across = width if orientation == "V" else height
    while 2 * across + thickness + sum(margins) > 30:
        positive = [i for i, value in enumerate(margins) if value]
        margins[int(rng.choice(positive))] -= 1
    outer_a, inner_a, inner_b, outer_b = margins

    if orientation == "V":
        split = outer_a + width + inner_a
        grid = np.zeros((height + extra,
                         split + thickness + inner_b + width + outer_b), dtype=np.int8)
        grid[:, split:split + thickness] = divider
        grid[offset_a:offset_a + height, outer_a:outer_a + width] = first
        start_b = split + thickness + inner_b
        grid[offset_b:offset_b + height, start_b:start_b + width] = second
        answer = np.concatenate((first, second), axis=1)
    else:
        split = outer_a + height + inner_a
        grid = np.zeros((split + thickness + inner_b + height + outer_b,
                         width + extra), dtype=np.int8)
        grid[split:split + thickness, :] = divider
        grid[outer_a:outer_a + height, offset_a:offset_a + width] = first
        start_b = split + thickness + inner_b
        grid[start_b:start_b + height, offset_b:offset_b + width] = second
        answer = np.concatenate((first, second), axis=0)
    return grid, answer


def _pair(rng: np.random.Generator, regime: str, index: int,
          orientation: str, source_like: bool,
          first_train_divider: int | None,
          first_train_same: bool) -> tuple[np.ndarray, np.ndarray]:
    if source_like and index < 2 and regime == "source_train":
        height, width = 5, 5
    elif source_like and regime == "target_test" and index < 3:
        height, width = ((2, 2), (4, 4), (3, 4))[index]
    else:
        height, width = _size(rng, regime)

    if regime == "source_train" and index == 0:
        # The first example always proves that interior zeroes are retained.
        height = max(3, height)
        width = max(3, width)
        hole_style = ("frame" if source_like else
                      str(rng.choice(("frame", "corners", "diamond", "sparse", "holes"))))
        styles = ["solid", hole_style]
    elif regime == "source_train" and index == 1:
        height = max(3, height)
        width = max(3, width)
        styles = [str(rng.choice(("diamond", "sparse", "irregular"))),
                  str(rng.choice(("solid", "holes", "corners")))]
    else:
        styles = [str(rng.choice(_STYLES)), str(rng.choice(_STYLES))]
    rng.shuffle(styles)

    divider = int(rng.integers(1, 10))
    if regime == "source_train" and index == 1 and first_train_divider is not None:
        if divider == first_train_divider:
            divider = divider % 9 + 1
    available = [c for c in range(1, 10) if c != divider]
    if regime == "source_train" and index < 2:
        same_color = first_train_same if index == 0 else not first_train_same
    else:
        same_color = bool(rng.random() < 0.43)
    if same_color:
        color_a = color_b = int(rng.choice(available))
    else:
        chosen = rng.choice(available, size=2, replace=False)
        color_a, color_b = int(chosen[0]), int(chosen[1])

    first = _pattern(rng, height, width, styles[0], color_a)
    if regime == "target_test" and rng.random() < 0.13:
        second = first.copy()
    else:
        second = _pattern(rng, height, width, styles[1], color_b)
    thickness = int(rng.integers(1, 4))
    if source_like and index < 2 and regime == "source_train":
        thickness = 1 if orientation == "V" else 2
    elif source_like and index == 0 and regime == "target_test":
        thickness = 2 if orientation == "V" else 1
    return _compose(rng, first, second, orientation, thickness, divider)


def _analyze(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    rows = [r for r in range(input_grid.shape[0])
            if input_grid[r, 0] != 0 and np.all(input_grid[r] == input_grid[r, 0])]
    cols = [c for c in range(input_grid.shape[1])
            if input_grid[0, c] != 0 and np.all(input_grid[:, c] == input_grid[0, c])]
    assert bool(rows) != bool(cols)
    orientation = "H" if rows else "V"
    line = rows if rows else cols
    divider = int(input_grid[line[0], 0] if rows else input_grid[0, line[0]])
    assert line == list(range(line[0], line[-1] + 1))
    assert 1 <= len(line) <= 3
    if rows:
        assert np.all(input_grid[line, :] == divider)
        regions = (input_grid[:line[0]], input_grid[line[-1] + 1:])
        axis = 0
    else:
        assert np.all(input_grid[:, line] == divider)
        regions = (input_grid[:, :line[0]], input_grid[:, line[-1] + 1:])
        axis = 1
    assert all(region.size for region in regions)

    crops, offsets, margins, colors = [], [], [], []
    for region in regions:
        rr, cc = np.nonzero(region)
        assert rr.size
        top, bottom = int(rr.min()), int(rr.max()) + 1
        left, right = int(cc.min()), int(cc.max()) + 1
        crop = region[top:bottom, left:right]
        unique = set(int(v) for v in np.unique(crop) if v != 0)
        assert len(unique) == 1 and divider not in unique
        assert np.any(crop[0] != 0) and np.any(crop[-1] != 0)
        assert np.any(crop[:, 0] != 0) and np.any(crop[:, -1] != 0)
        crops.append(crop)
        offsets.append(top if orientation == "V" else left)
        margins.append((top, region.shape[0] - bottom, left, region.shape[1] - right))
        colors.append(next(iter(unique)))
    assert crops[0].shape == crops[1].shape
    assert np.array_equal(output_grid, np.concatenate(crops, axis=axis))
    assert not np.any(output_grid == divider)
    return {"orientation": orientation, "thickness": len(line),
            "divider": divider, "shape": crops[0].shape,
            "crops": crops, "colors": tuple(colors),
            "offsets": tuple(offsets), "margins": tuple(margins)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _count(num_train, 2, 10, "num_train")
    if num_test is not None:
        num_test = _count(num_test, 1, 4, "num_test")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # Zero is the sole puzzle-wide convention. Divider and motif colors are
    # locally visible, and the spatial order is fixed for every pair.
    source_like = bool(rng.random() < 0.12)
    first_train_same = bool(rng.integers(2))
    pair_seeds = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.int64)
    train_first = str(rng.choice(("H", "V")))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_divider = None

    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[index if split == "train"
                                                          else num_train + index]))
            if split == "train" and index < 2:
                orientation = train_first if index == 0 else ("V" if train_first == "H" else "H")
            else:
                orientation = str(pair_rng.choice(("H", "V")))
            # The fallback changes the local seed; all legal counts terminate
            # without imposing a narrow canonical layout.
            for attempt in range(20):
                local_rng = np.random.default_rng(int(pair_seeds[index if split == "train"
                                                                else num_train + index]) + attempt)
                inp, out = _pair(local_rng, "source_train" if split == "train" else "target_test",
                                 index, orientation, source_like, first_divider,
                                 first_train_same)
                signature = (inp.shape, inp.tobytes())
                if signature not in seen and _grid_text(inp) not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[split].append({"input": inp, "output": out})
                    if split == "train" and index == 0:
                        first_divider = _analyze(inp, out)["divider"]
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu" and all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _grid_text(inp) not in _SOURCE_INPUTS
            assert not np.array_equal(inp, out)
            analyses[split].append(_analyze(inp, out))
    train = analyses["train"]
    assert {info["orientation"] for info in train} == {"H", "V"}
    assert any(np.all(crop != 0) for info in train for crop in info["crops"])
    assert any(np.any(crop == 0) for info in train for crop in info["crops"])
    assert any(info["colors"][0] == info["colors"][1] for info in train)
    assert any(info["colors"][0] != info["colors"][1] for info in train)
    assert len({info["divider"] for info in train}) >= 2


def _same(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(first[split], second[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, input_shapes, output_shapes = set(), set(), set(), set()
    masks, signatures, same_offsets, zero_margins, identical = set(), set(), 0, 0, 0
    train_sizes, test_sizes, thick, orientations, dividers = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((pair["input"].shape, pair["input"].tobytes())
                             for split in ("train", "test") for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze(inp, out)
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                (train_sizes if split == "train" else test_sizes).add(info["shape"])
                thick.add(info["thickness"])
                orientations.add(info["orientation"])
                dividers.add(info["divider"])
                same_offsets += info["offsets"][0] == info["offsets"][1]
                zero_margins += any(0 in side for side in info["margins"])
                identical += np.array_equal(*info["crops"])
                for crop in info["crops"]:
                    masks.add((crop.shape, (crop != 0).tobytes()))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(input_shapes) >= 25
    assert len(output_shapes) >= 20 and len(masks) >= 100
    assert len(train_sizes & test_sizes) >= 5 and max(h for h, w in test_sizes) >= 7
    assert thick == {1, 2, 3} and orientations == {"H", "V"} and len(dividers) == 9
    assert same_offsets >= 10 and zero_margins >= 30 and identical >= 2

    cases = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
             (2, None), (10, None))
    for index, (n_train, n_test) in enumerate(cases):
        puzzle = generate_puzzle(1000 + index, n_train, n_test)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(1000 + index, n_train, n_test))
        assert n_train is None or len(puzzle["train"]) == n_train
        assert n_test is None or len(puzzle["test"]) == n_test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
