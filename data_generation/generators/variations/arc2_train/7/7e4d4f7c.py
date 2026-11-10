"""Generate variations of the three-row header projection ARC task."""

import numpy as np


_COLORS = tuple(range(10))
_MODES = ("separate", "shared_signal_guide", "guide_is_highlight")
_SOURCE_ROWS = (
    (("07070707", "70000000", "00000000", "70000000", "00000000", "70000000"),
     ("07070707", "70000000", "06060606")),
    (("2121221211", "7222222222", "2222222222", "7222222222", "2222222222",
      "7222222222", "2222222222", "7222222222", "2222222222", "7222222222"),
     ("2121221211", "7222222222", "2626226266")),
    (("1114111441441", "6444444444444", "4444444444444", "6444444444444",
      "4444444444444", "6444444444444", "4444444444444", "6444444444444",
      "4444444444444", "6444444444444", "4444444444444", "6444444444444"),
     ("1114111441441", "6444444444444", "6664666446446")),
    (("49499449994", "79999999999", "99999999999", "79999999999",
      "99999999999", "79999999999", "99999999999", "79999999999"),
     ("49499449994", "79999999999", "69699669996")),
    (("81818", "41111", "11111"), ("81818", "41111", "61616")),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.flat)


_SOURCE_KEYS = frozenset(
    (_key(np.array([[int(c) for c in row] for row in inputs], dtype=np.int8)),
     _key(np.array([[int(c) for c in row] for row in outputs], dtype=np.int8)))
    for inputs, outputs in _SOURCE_ROWS
)


def _checked_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(rng: np.random.Generator, num_train: int | None,
            num_test: int | None) -> tuple[int, int]:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _mode_schedule(rng: np.random.Generator, num_train: int,
                   num_test: int) -> tuple[list[str], list[str]]:
    # Select a visible set of relationships, then use each at least once.
    if num_train >= 3:
        subset_size = int(rng.choice((1, 2, 3), p=(0.17, 0.33, 0.50)))
    else:
        subset_size = int(rng.choice((1, 2), p=(0.25, 0.75)))
    if subset_size == 1:
        allowed = (_MODES[int(rng.integers(0, 2))],)
    elif subset_size == 2:
        pairs = ((_MODES[0], _MODES[1]), (_MODES[0], _MODES[2]),
                 (_MODES[1], _MODES[2]))
        allowed = pairs[int(rng.integers(0, len(pairs)))]
    else:
        allowed = _MODES
    train = list(allowed)
    while len(train) < num_train:
        train.append(allowed[int(rng.integers(0, len(allowed)))])
    rng.shuffle(train)
    test = [allowed[int(rng.integers(0, len(allowed)))]
            for _ in range(num_test)]
    return train, test


def _new_roles(rng: np.random.Generator, mode: str,
               highlight: int, force_recolor: bool) -> tuple[int, int, int]:
    filler = int(rng.choice([c for c in _COLORS if c != highlight]))
    if mode == "separate" and not force_recolor and rng.random() < 0.24:
        signal = highlight
    else:
        signal = int(rng.choice([c for c in _COLORS if c not in (filler, highlight)]))
    if mode == "shared_signal_guide":
        guide = signal
    elif mode == "guide_is_highlight":
        guide = highlight
    elif mode == "separate":
        guide = int(rng.choice([c for c in _COLORS if c not in
                                (filler, signal, highlight)]))
    else:
        raise ValueError("unknown role mode")
    return filler, signal, guide


def _roles(rng: np.random.Generator, mode: str, highlight: int,
           prior: list[tuple[int, int, int, str]],
           force_recolor: bool) -> tuple[int, int, int]:
    reusable = [entry[:3] for entry in prior if entry[3] == mode and
                (not force_recolor or entry[1] != highlight)]
    if reusable and rng.random() < 0.48:
        return reusable[int(rng.integers(0, len(reusable)))]
    return _new_roles(rng, mode, highlight, force_recolor)


def _dimensions(rng: np.random.Generator, regime: str,
                source_like: bool, must_crop: bool) -> tuple[int, int]:
    if source_like:
        if regime == "source_train":
            return int(rng.choice((6, 8, 10, 12))), int(rng.integers(8, 14))
        return 3, int(rng.integers(4, 8))

    if regime == "source_train":
        h_category = int(rng.choice(3, p=(0.20, 0.45, 0.35)))
        w_category = int(rng.choice(3, p=(0.20, 0.45, 0.35)))
    elif regime == "target_test":
        h_category = int(rng.choice(3, p=(0.35, 0.35, 0.30)))
        w_category = int(rng.choice(3, p=(0.35, 0.30, 0.35)))
    else:
        raise ValueError("unknown regime")

    h_ranges = ((3, 3), (4, 12), (13, 30))
    w_ranges = ((4, 7), (8, 13), (14, 30))
    h_low, h_high = h_ranges[h_category]
    w_low, w_high = w_ranges[w_category]
    if must_crop and h_high == 3:
        h_low, h_high = 4, 12
    height = int(rng.integers(h_low, h_high + 1))
    width = int(rng.integers(w_low, w_high + 1))
    return height, width


def _mask(rng: np.random.Generator, width: int) -> np.ndarray:
    style = int(rng.integers(0, 6))
    mask = np.zeros(width, dtype=bool)
    if style == 1:
        mask[int(rng.integers(width))] = True
    elif style == 2:
        mask[:] = True
        mask[int(rng.integers(width))] = False
    elif style == 3:
        length = int(rng.integers(1, width))
        start = int(rng.integers(0, width - length + 1))
        mask[start:start + length] = True
    elif style == 4:
        step = int(rng.integers(2, min(6, width + 1)))
        phase = int(rng.integers(step))
        mask = np.arange(width) % step == phase
    elif style == 5:
        density = float(rng.uniform(0.18, 0.82))
        mask = rng.random(width) < density
    else:
        count = int(rng.integers(1, width))
        mask[rng.choice(width, size=count, replace=False)] = True
    if not mask.any() or mask.all():
        mask[:] = False
        mask[int(rng.integers(width))] = True
    return mask


def _make_pair(height: int, width: int, roles: tuple[int, int, int],
               mask: np.ndarray, highlight: int) -> dict:
    filler, signal, guide = roles
    inputs = np.full((height, width), filler, dtype=np.int8)
    inputs[0, mask] = signal
    inputs[1::2, 0] = guide
    outputs = inputs[:3].copy()
    outputs[2, mask] = highlight
    return {"input": inputs, "output": outputs}


def _pair_info(pair: dict) -> dict:
    inputs, outputs = pair["input"], pair["output"]
    height, width = inputs.shape
    _require(3 <= height <= 30 and 4 <= width <= 30,
             "input scaffold dimensions are invalid")
    _require(outputs.shape == (3, width), "output must have three rows")
    filler = int(inputs[2, 0])
    _require(np.all(inputs[2] == filler), "third input row must be blank")
    header_colors = set(int(x) for x in np.unique(inputs[0]))
    _require(len(header_colors) == 2 and filler in header_colors,
             "header must contain filler and one signal")
    signal = next(iter(header_colors - {filler}))
    guide = int(inputs[1, 0])
    _require(guide != filler, "guide must contrast with filler")
    for row in range(1, height):
        expected = np.full(width, filler, dtype=np.int8)
        if row % 2:
            expected[0] = guide
        _require(np.array_equal(inputs[row], expected),
                 "body must alternate guide and blank rows")
    mask = inputs[0] != filler
    _require(mask.any() and (~mask).any(), "mask must be nontrivial")
    _require(np.array_equal(outputs[:2], inputs[:2]),
             "first two rows must be copied")
    _require(np.all(outputs[2, ~mask] == filler),
             "inactive positions must retain filler")
    projection = np.unique(outputs[2, mask])
    _require(len(projection) == 1, "active positions need one highlight")
    highlight = int(projection[0])
    _require(highlight != filler, "highlight cannot equal filler")
    if guide == highlight:
        _require(signal != highlight, "ambiguous simultaneous role equality")
        mode = "guide_is_highlight"
    elif guide == signal:
        mode = "shared_signal_guide"
    else:
        mode = "separate"
    return {"height": height, "width": width, "filler": filler,
            "signal": signal, "guide": guide, "highlight": highlight,
            "mode": mode, "mask": tuple(bool(x) for x in mask)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
             "illegal pair counts")

    infos = {"train": [], "test": []}
    inputs_seen, pairs_seen, layouts_seen = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= size <= 30 for size in grid.shape),
                         "grid dimension exceeds ARC bounds")
                _require(np.issubdtype(grid.dtype, np.integer),
                         "grid must have integer dtype")
                _require(np.all((grid >= 0) & (grid <= 9)),
                         "grid colors must be in [0, 9]")
            input_key = _key(pair["input"])
            pair_key = input_key, _key(pair["output"])
            _require(input_key not in inputs_seen and pair_key not in pairs_seen,
                     "duplicate input or pair")
            _require(pair_key not in _SOURCE_KEYS, "copied source pair")
            inputs_seen.add(input_key)
            pairs_seen.add(pair_key)
            info = _pair_info(pair)
            layout = info["width"], info["mask"]
            _require(layout not in layouts_seen, "duplicate header layout")
            layouts_seen.add(layout)
            infos[split].append(info)

    highlights = {info["highlight"] for split in infos.values() for info in split}
    _require(len(highlights) == 1, "highlight must be puzzle-wide")
    highlight = next(iter(highlights))
    train = infos["train"]
    tests = infos["test"]
    _require(any(info["height"] > 3 for info in train),
             "training must demonstrate cropping")
    _require(any(info["guide"] != highlight for info in train),
             "training must disprove copying the guide color")
    _require(any(info["signal"] != highlight for info in train),
             "training must demonstrate signal recoloring")
    train_modes = {info["mode"] for info in train}
    _require({info["mode"] for info in tests} <= train_modes,
             "test guide relationship must occur in training")
    if any(info["height"] > 3 for info in tests):
        _require(any(info["height"] > 3 for info in train),
                 "test cropping must be demonstrated in training")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    highlight = 6 if rng.random() < 0.50 else int(rng.choice([c for c in _COLORS if c != 6]))
    source_like = bool(rng.random() < 0.20)
    train_modes, test_modes = _mode_schedule(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    prior: list[tuple[int, int, int, str]] = []
    seen_inputs, seen_layouts = set(), set()
    has_recolor = False
    for split, regime, modes in (("train", "source_train", train_modes),
                                  ("test", "target_test", test_modes)):
        for index, mode in enumerate(modes):
            force_recolor = split == "train" and not has_recolor and index == num_train - 1
            roles = _roles(rng, mode, highlight, prior, force_recolor)
            if split == "train" and roles[1] != highlight:
                has_recolor = True
            global_index = len(puzzle["train"]) + len(puzzle["test"])
            for attempt in range(300):
                height, width = _dimensions(rng, regime, source_like,
                                            split == "train" and index == 0)
                mask = _mask(rng, width)
                pair = _make_pair(height, width, roles, mask, highlight)
                layout = width, tuple(bool(x) for x in mask)
                key = _key(pair["input"])
                if (layout not in seen_layouts and key not in seen_inputs and
                        (key, _key(pair["output"])) not in _SOURCE_KEYS):
                    break
            else:
                # A deterministic unique layout if a short-width draw collided repeatedly.
                height, width = 5 + global_index, 30
                for column in range(width):
                    mask = np.zeros(width, dtype=bool)
                    mask[column] = True
                    layout = width, tuple(bool(x) for x in mask)
                    if layout not in seen_layouts:
                        break
                pair = _make_pair(height, width, roles, mask, highlight)
                key = _key(pair["input"])
            seen_inputs.add(key)
            seen_layouts.add(layout)
            puzzle[split].append(pair)
            prior.append((*roles, mode))
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle: dict) -> tuple:
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # The inference reads only visible demonstrations and the test input.
    colors = set()
    for pair in train_pairs:
        filler = int(pair["input"][2, 0])
        mask = pair["input"][0] != filler
        colors.update(int(x) for x in pair["output"][2, mask])
    _require(len(colors) == 1, "training does not establish a unique highlight")
    highlight = next(iter(colors))
    filler = int(test_input[2, 0])
    predicted = test_input[:3].copy()
    predicted[2, test_input[0] != filler] = highlight
    return predicted


def self_test() -> None:
    examples = [generate_puzzle(314159), generate_puzzle(271828, 7, 3)]
    _require(_signature(examples[0]) == _signature(generate_puzzle(314159)),
             "automatic generation is not deterministic")
    _require(_signature(examples[1]) == _signature(generate_puzzle(271828, 7, 3)),
             "explicit generation is not deterministic")
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + offset, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "train count not honored")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "test count not honored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")

    train_counts, test_counts, highlights = set(), set(), set()
    signatures, shapes, layouts, palettes = set(), set(), set(), set()
    counts = {name: 0 for name in ("train_minimal", "test_crop", "test_long",
                                   "odd_train", "odd_test", "train_single",
                                   "train_dense", "test_single", "test_dense",
                                   "source_like", "wider_test", "same_width",
                                   "all_modes", "shared_and_guide", "role_reuse",
                                   "test_familiar", "test_novel", "signal_highlight")}
    for seed in range(100000, 100042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [_pair_info(pair) for pair in puzzle["train"]]
        tests = [_pair_info(pair) for pair in puzzle["test"]]
        highlights.add(train[0]["highlight"])
        all_infos = train + tests
        for info in all_infos:
            shapes.add((info["height"], info["width"]))
            layouts.add((info["width"], info["mask"]))
            palettes.add((info["filler"], info["signal"], info["guide"]))
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict_from_training(puzzle["train"],
                                                        pair["input"]), pair["output"]),
                     "training-only prediction differs from test output")
        train_palettes = [(i["filler"], i["signal"], i["guide"]) for i in train]
        test_palettes = [(i["filler"], i["signal"], i["guide"]) for i in tests]
        counts["train_minimal"] += sum(i["height"] == 3 for i in train)
        counts["test_crop"] += sum(i["height"] > 3 for i in tests)
        counts["test_long"] += sum(i["height"] >= 13 for i in tests)
        counts["odd_train"] += sum(i["height"] > 3 and i["height"] % 2 for i in train)
        counts["odd_test"] += sum(i["height"] > 3 and i["height"] % 2 for i in tests)
        counts["train_single"] += sum(sum(i["mask"]) == 1 for i in train)
        counts["train_dense"] += sum(sum(i["mask"]) == i["width"] - 1 for i in train)
        counts["test_single"] += sum(sum(i["mask"]) == 1 for i in tests)
        counts["test_dense"] += sum(sum(i["mask"]) == i["width"] - 1 for i in tests)
        counts["source_like"] += int(all(i["height"] in (6, 8, 10, 12) and
                                         8 <= i["width"] <= 13 for i in train) and
                                     all(i["height"] == 3 and 4 <= i["width"] <= 7
                                         for i in tests))
        counts["wider_test"] += int(max(i["width"] for i in tests) >
                                     min(i["width"] for i in train))
        counts["same_width"] += int(bool({i["width"] for i in tests} &
                                         {i["width"] for i in train}))
        train_modes = {i["mode"] for i in train}
        counts["all_modes"] += int(train_modes == set(_MODES))
        counts["shared_and_guide"] += int({"shared_signal_guide", "guide_is_highlight"}
                                           <= train_modes)
        counts["role_reuse"] += int(len(set(train_palettes + test_palettes)) <
                                    len(train_palettes + test_palettes))
        counts["test_familiar"] += sum(p in train_palettes for p in test_palettes)
        counts["test_novel"] += sum(p not in train_palettes for p in test_palettes)
        counts["signal_highlight"] += sum(i["signal"] == i["highlight"] for i in train)

    _require(len(signatures) == 42 and len(train_counts) >= 4 and
             len(test_counts) >= 3, "puzzle or count diversity is too narrow")
    _require(len(highlights) >= 5 and 6 in highlights and len(shapes) >= 70 and
             len(layouts) >= 100 and len(palettes) >= 75,
             "color, shape, layout, or palette diversity is too narrow")
    thresholds = {"train_minimal": 5, "test_crop": 10, "test_long": 5,
                  "odd_train": 10, "odd_test": 5, "train_single": 5,
                  "train_dense": 5, "test_single": 3, "test_dense": 3,
                  "source_like": 2, "wider_test": 10, "same_width": 5,
                  "all_modes": 5, "shared_and_guide": 5, "role_reuse": 5,
                  "test_familiar": 5, "test_novel": 10, "signal_highlight": 2}
    for name, threshold in thresholds.items():
        _require(counts[name] >= threshold,
                 f"insufficient diversity for {name}: {counts[name]}")

    damaged = {split: [{name: grid.copy() for name, grid in pair.items()}
                       for pair in examples[0][split]] for split in ("train", "test")}
    test_pair = damaged["test"][0]
    filler = int(test_pair["input"][2, 0])
    column = int(np.flatnonzero(test_pair["input"][0] != filler)[0])
    test_pair["output"][2, column] = filler
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged projection")
