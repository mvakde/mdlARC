"""ARC dense-sector compression puzzle generator."""

import numpy as np


def _source_grid(rows):
    return np.array([[int(cell) for cell in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_source_grid(("000222000", "050222000", "000222000", "500000000",
                   "000500050", "000000000", "050777000", "000775000",
                   "000777000")), _source_grid(("020", "000", "070"))),
    (_source_grid(("333000888", "333000858", "333000888", "000757000",
                   "000777000", "000777000", "666005999", "666000999",
                   "656050995")), _source_grid(("308", "070", "609"))),
    (_source_grid(("444000050", "544000000", "444050000", "000333050",
                   "000333000", "000333000", "005999000", "000959000",
                   "000999000")), _source_grid(("400", "030", "090"))),
)
_SHAPES = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4))
_SOURCE_TEST_LAYOUT = ((0, 0), (1, 1), (2, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + (seed // 4 + test) % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(array):
    return (array.shape, tuple(int(v) for v in array.flat))


def _layout(rng, shape, count, suggested=None):
    if suggested is not None and len(suggested) == count and rng.random() < 0.28:
        return tuple(suggested)
    indexes = rng.choice(shape[0] * shape[1], size=count, replace=False)
    return tuple(sorted(divmod(int(i), shape[1]) for i in indexes))


def _tile_colors(rng, pool, count, min_distinct=1):
    """At most three uses per color keep background identifiable by sector modes."""
    pool = [int(color) for color in pool]
    chosen = [int(x) for x in rng.choice(
        pool, size=max((count + 2) // 3, min_distinct), replace=False)]
    colors = chosen.copy()
    while len(colors) < count:
        options = [c for c in chosen if colors.count(c) < 3]
        colors.append(int(rng.choice(options)))
    rng.shuffle(colors)
    return colors


def _make_pair(rng, shape, stride, background, nuisance, layout, pool,
               corruption=None, noise=None, multi_noise=False, min_distinct=1):
    rows, columns = shape
    area = stride * stride
    grid = np.full((rows * stride, columns * stride), background, dtype=np.int8)
    output = np.full(shape, background, dtype=np.int8)
    colors = _tile_colors(rng, [c for c in pool if c not in (background, nuisance)],
                          len(layout), min_distinct)
    if corruption is None:
        corruption = int(rng.integers(0, len(layout) + 1))
    corrupted = set(int(i) for i in rng.choice(len(layout), size=corruption, replace=False))
    for index, ((row, column), color) in enumerate(zip(layout, colors)):
        output[row, column] = color
        grid[row * stride:(row + 1) * stride,
             column * stride:(column + 1) * stride] = color
        if index in corrupted:
            dy, dx = divmod(int(rng.integers(area)), stride)
            grid[row * stride + dy, column * stride + dx] = nuisance
    empty = [(r, c) for r in range(rows) for c in range(columns)
             if (r, c) not in layout]
    if noise is None:
        noise = int(rng.integers(0, min(len(empty), 6) + 1))
    for i in rng.choice(len(empty), size=noise, replace=False):
        row, column = empty[int(i)]
        mark_count = 1
        if multi_noise and stride >= 3:
            mark_count = int(rng.integers(2, 4))
        elif stride >= 3 and rng.random() < 0.38:
            mark_count = int(rng.integers(2, 4))
        for cell in rng.choice(area, size=mark_count, replace=False):
            dy, dx = divmod(int(cell), stride)
            grid[row * stride + dy, column * stride + dx] = nuisance
    return {"input": grid, "output": output}


def _source_orbit(rng):
    turns = int(rng.integers(4))
    reflect = bool(rng.integers(2))
    result = []
    for row, column in _SOURCE_TEST_LAYOUT:
        if reflect:
            column = 2 - column
        for _ in range(turns):
            row, column = column, 2 - row
        result.append((row, column))
    return tuple(sorted(result))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer(seed, "seed", 0, None)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    stride = (3, 3, 3, 2, 4)[seed % 5]
    variable_roles = train_count >= 3 and bool(rng.integers(2))
    base_background = 0 if rng.random() < 0.22 else int(rng.integers(1, 10))
    base_nuisance = int(rng.choice([c for c in range(10) if c != base_background]))
    palette = [c for c in range(10) if c not in (base_background, base_nuisance)]
    if base_background != 0 and 0 in palette and seed % 2 == 0:
        primary = [0] + [int(c) for c in rng.choice(
            [c for c in palette if c != 0], size=4, replace=False)]
    else:
        primary = [int(c) for c in rng.choice(palette, size=5, replace=False)]
    puzzle = {"train": [], "test": []}
    used = set()
    train_layouts = []
    train_shapes = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(64):
                if split == "train" and index == 0 and stride == 3:
                    shape = (3, 3)
                elif split == "train" and index == 1:
                    shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
                    if shape == train_shapes[0]:
                        shape = _SHAPES[(_SHAPES.index(shape) + 1) % len(_SHAPES)]
                elif split == "test" and index == 0 and (seed % 11 == 0 or seed % 13 == 0):
                    shape = (3, 3)
                elif split == "test" and rng.random() < 0.48:
                    shape = train_shapes[int(rng.integers(len(train_shapes)))]
                else:
                    shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
                maximum = shape[0] * shape[1] - 4
                tile_count = int(rng.integers(1, maximum + 1))
                if split == "test" and index == 0 and (seed % 11 == 0 or seed % 13 == 0):
                    tile_count = 3
                    layout = _SOURCE_TEST_LAYOUT if seed % 11 == 0 else _source_orbit(rng)
                else:
                    suggested = None
                    if split == "test":
                        matches = [old for old_shape, old in zip(train_shapes, train_layouts)
                                   if old_shape == shape and len(old) == tile_count]
                        if matches:
                            suggested = matches[int(rng.integers(len(matches)))]
                    layout = _layout(rng, shape, tile_count, suggested)
                if variable_roles:
                    if split == "train" and index == 0:
                        background, nuisance = base_background, base_nuisance
                    elif split == "train" and index == 1:
                        background = int(rng.choice([c for c in range(10)
                                                     if c != base_background and c not in primary]))
                        first_colors = [int(c) for c in np.unique(puzzle["train"][0]["output"])
                                        if c != base_background]
                        nuisance = int(rng.choice(first_colors))
                    else:
                        background = int(rng.choice([c for c in range(10) if c not in primary]))
                        nuisance = int(rng.choice([c for c in range(10) if c != background]))
                else:
                    background, nuisance = base_background, base_nuisance
                if split == "train":
                    pool = primary
                    min_distinct = 1
                    if index == 0:
                        corruption = max(1, int(rng.integers(1, tile_count + 1)))
                        noise = max(1, int(rng.integers(1, min(5, shape[0] * shape[1] - tile_count) + 1)))
                    elif index == 1 and variable_roles:
                        corruption, noise = 1, 1
                    elif index == 1 and seed % 4 == 0:
                        corruption, noise = 0, 0
                    else:
                        corruption, noise = None, None
                else:
                    trained = {int(v) for pair in puzzle["train"] for v in pair["output"].flat
                               if int(v) != _pair_background(pair)}
                    available_novel = [c for c in range(10) if c not in trained
                                       and c not in (background, nuisance)]
                    familiar = [c for c in trained if c not in (background, nuisance)]
                    novelty_mode = (seed + index) % 3
                    if novelty_mode == 0 or not available_novel:
                        pool = familiar
                    elif novelty_mode == 1 and len(available_novel) >= 2 and tile_count >= 2:
                        pool = available_novel
                    else:
                        pool = available_novel + familiar
                    if not pool:
                        pool = [c for c in range(10) if c not in (background, nuisance)]
                    min_distinct = 2 if novelty_mode == 1 and pool == available_novel and tile_count >= 2 else 1
                    corruption, noise = None, None
                allowed = [c for c in pool if c not in (background, nuisance)]
                if len(allowed) < max((tile_count + 2) // 3, min_distinct):
                    continue
                pair = _make_pair(rng, shape, stride, background, nuisance, layout,
                                  pool, corruption, noise,
                                  multi_noise=(split == "train" and index == 0 and stride >= 3),
                                  min_distinct=min_distinct)
                pair_key = _key(pair["input"])
                if pair_key in used or any(np.array_equal(pair["input"], src_in)
                                           and np.array_equal(pair["output"], src_out)
                                           for src_in, src_out in _SOURCE_PAIRS):
                    continue
                used.add(pair_key)
                puzzle[split].append(pair)
                if split == "train":
                    train_layouts.append(layout)
                    train_shapes.append(shape)
                break
            else:
                raise RuntimeError(f"failed to build unique {split} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_background(pair):
    grid, output = pair["input"], pair["output"]
    _require(grid.shape[0] % output.shape[0] == 0 and
             grid.shape[1] % output.shape[1] == 0, "input must divide into output sectors")
    sy, sx = grid.shape[0] // output.shape[0], grid.shape[1] // output.shape[1]
    _require(sy == sx and sy in (2, 3, 4), "square tile stride must be 2, 3, or 4")
    modes = []
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            block = grid[row * sy:(row + 1) * sy, column * sy:(column + 1) * sy]
            modes.append(int(np.argmax(np.bincount(block.flat, minlength=10))))
    counts = np.bincount(modes, minlength=10)
    _require(int(counts.max()) >= 4 and int(np.count_nonzero(counts == counts.max())) == 1,
             "background must be the unique sector-mode plurality")
    return int(np.argmax(counts))


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    background = _pair_background(pair)
    stride = grid.shape[0] // output.shape[0]
    area = stride * stride
    _require(output.shape in _SHAPES, "unsupported lattice shape")
    layout, anomalies, payload_counts, noise_sizes = [], [], {}, []
    corrupted = noisy = 0
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            block = grid[row * stride:(row + 1) * stride,
                         column * stride:(column + 1) * stride]
            counts = np.bincount(block.flat, minlength=10)
            dominant = int(np.argmax(counts))
            value = int(output[row, column])
            if dominant == background:
                _require(value == background, "sparse sector must output background")
                other = block[block != background]
                _require(other.size <= (1 if stride == 2 else 3), "empty sector noise too dense")
                if other.size:
                    noisy += 1
                    noise_sizes.append(int(other.size))
                    anomalies.extend(int(x) for x in other)
            else:
                _require(counts[dominant] >= area - 1, "tile is not dense")
                _require(value == dominant, "tile must output its dominant color")
                payload_counts[dominant] = payload_counts.get(dominant, 0) + 1
                layout.append((row, column))
                others = block[block != dominant]
                _require(others.size <= 1, "tile has too many defects")
                if others.size:
                    corrupted += 1
                    anomalies.append(int(others[0]))
    _require(1 <= len(layout) <= output.size - 4, "invalid tile count")
    _require(max(payload_counts.values()) <= 3, "payload color repeats too often")
    _require(len(set(anomalies)) <= 1, "multiple nuisance colors in pair")
    nuisance = anomalies[0] if anomalies else None
    if nuisance is not None:
        _require(nuisance != background and nuisance not in payload_counts,
                 "nuisance conflicts with local color roles")
    return {"background": background, "nuisance": nuisance, "stride": stride,
            "shape": output.shape, "layout": tuple(layout), "tiles": len(layout),
            "corrupted": corrupted, "noisy": noisy, "noise_sizes": noise_sizes,
            "payload": set(payload_counts)}


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= n <= 30 for n in grid.shape), f"{label} dimensions out of bounds")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} colors out of bounds")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    inputs, pairs, train_info, test_info = set(), set(), [], []
    for split, infos in (("train", train_info), ("test", test_info)):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            _validate_grid(pair["input"], f"{split}[{index}].input")
            _validate_grid(pair["output"], f"{split}[{index}].output")
            ikey = _key(pair["input"])
            pkey = (ikey, _key(pair["output"]))
            _require(ikey not in inputs and pkey not in pairs, "duplicate pair")
            inputs.add(ikey)
            pairs.add(pkey)
            _require(not any(np.array_equal(pair["input"], si)
                             and np.array_equal(pair["output"], so)
                             for si, so in _SOURCE_PAIRS), "source pair copied")
            infos.append(_inspect_pair(pair))
    all_info = train_info + test_info
    _require(len({info["stride"] for info in all_info}) == 1, "stride must be puzzle-level")
    _require(len({info["shape"] for info in train_info}) >= 2,
             "training must demonstrate variable lattice shapes")
    _require(any(info["corrupted"] and info["noisy"] for info in train_info),
             "training must show tile defects and empty-sector noise")
    train_backgrounds = {info["background"] for info in train_info}
    if len(train_backgrounds) == 1:
        _require(all(info["background"] in train_backgrounds for info in test_info),
                 "unexplained test background change")
    train_nuisances = {info["nuisance"] for info in train_info if info["nuisance"] is not None}
    if len(train_nuisances) == 1:
        _require(all(info["nuisance"] in train_nuisances or info["nuisance"] is None
                     for info in test_info), "unexplained test nuisance change")


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _predict_from_training(train_pairs, test_input):
    # This receives only demonstrations and an input; no generation state or label.
    strides = {(pair["input"].shape[0] // pair["output"].shape[0],
                pair["input"].shape[1] // pair["output"].shape[1])
               for pair in train_pairs}
    _require(len(strides) == 1, "training has no single tile stride")
    sy, sx = next(iter(strides))
    _require(sy == sx and test_input.shape[0] % sy == 0
             and test_input.shape[1] % sx == 0, "test cannot use training stride")
    height, width = test_input.shape[0] // sy, test_input.shape[1] // sx
    sector_modes = np.empty((height, width), dtype=np.int8)
    sector_counts = np.empty((height, width), dtype=np.int8)
    for row in range(height):
        for column in range(width):
            block = test_input[row * sy:(row + 1) * sy,
                               column * sx:(column + 1) * sx]
            counts = np.bincount(block.flat, minlength=10)
            sector_modes[row, column] = int(np.argmax(counts))
            sector_counts[row, column] = int(counts[sector_modes[row, column]])
    background = int(np.argmax(np.bincount(sector_modes.flat, minlength=10)))
    prediction = np.full((height, width), background, dtype=np.int8)
    dense = (sector_modes != background) & (sector_counts >= sy * sx - 1)
    prediction[dense] = sector_modes[dense]
    return prediction


def self_test() -> None:
    signatures = set()
    shapes, strides, backgrounds, counts, layouts = set(), set(), set(), set(), set()
    clean = all_corrupt = multi_noise = variable_bg = zero_tile = 0
    train_counts, test_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)),
                 "generation is not deterministic")
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = [_inspect_pair(pair) for pair in puzzle["train"] + puzzle["test"]]
        for pair in puzzle["train"] + puzzle["test"]:
            _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                    pair["output"]),
                     "output not recoverable from demonstrations and input")
        variable_bg += len({x["background"] for x in infos}) > 1
        for info in infos:
            shapes.add(info["shape"])
            strides.add(info["stride"])
            backgrounds.add(info["background"])
            counts.add(info["tiles"])
            layouts.add((info["shape"], info["layout"]))
            clean += info["corrupted"] == 0 and info["noisy"] == 0
            all_corrupt += info["corrupted"] == info["tiles"]
            multi_noise += any(n >= 2 for n in info["noise_sizes"])
            zero_tile += 0 in info["payload"]
    _require(len(signatures) == 42, "cross-seed puzzle repetition")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack range")
    _require(len(shapes) == len(_SHAPES) and strides == {2, 3, 4}, "scale coverage lacking")
    _require(len(backgrounds) >= 7 and len(counts) >= 8 and len(layouts) >= 60,
             "structural or color-role diversity lacking")
    _require(min(clean, all_corrupt, multi_noise, variable_bg, zero_tile) > 0,
             "noise or color-role regime missing")
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(113, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts ignored")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(116, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
