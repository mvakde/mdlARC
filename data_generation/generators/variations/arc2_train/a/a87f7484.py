import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(array.shape),
        tuple(int(value) for value in array.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _array_key(input_grid), _array_key(output_grid)


_SOURCE_PAIRS = (
    (
        _grid((
            "300404200800100",
            "033444022088011",
            "030404020080010",
        )),
        _grid((
            "404",
            "444",
            "404",
        )),
    ),
    (
        _grid((
            "200300707100",
            "200300070100",
            "022033707011",
        )),
        _grid((
            "707",
            "070",
            "707",
        )),
    ),
    (
        _grid((
            "077",
            "770",
            "707",
            "300",
            "033",
            "300",
            "200",
            "022",
            "200",
            "800",
            "088",
            "800",
        )),
        _grid((
            "077",
            "770",
            "707",
        )),
    ),
    (
        _grid((
            "606",
            "066",
            "606",
            "404",
            "044",
            "404",
            "888",
            "808",
            "888",
        )),
        _grid((
            "888",
            "808",
            "888",
        )),
    ),
    (
        _grid((
            "050",
            "505",
            "050",
            "030",
            "303",
            "030",
            "606",
            "660",
            "606",
            "040",
            "404",
            "040",
            "080",
            "808",
            "080",
        )),
        _grid((
            "606",
            "660",
            "606",
        )),
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    _pair_key(input_grid, output_grid)
    for input_grid, output_grid in _SOURCE_PAIRS
)

_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))


def _is_plain_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        if not _is_plain_integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_plain_integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _mask_key(mask):
    return tuple(int(v) for v in mask.ravel())


def _new_mask(rng, side):
    area = side * side
    weight = int(rng.integers(1, area))
    positions = rng.choice(area, size=weight, replace=False)
    mask = np.zeros(area, dtype=bool)
    mask[positions] = True
    return mask.reshape(side, side)


def _transforms(mask):
    for turns in (1, 2, 3):
        yield np.rot90(mask, turns)
    reflected = np.fliplr(mask)
    for turns in (0, 1, 2, 3):
        yield np.rot90(reflected, turns)


def _outlier_mask(rng, repeated, mode):
    area = repeated.size
    if mode == "transform":
        options = [m for m in _transforms(repeated) if not np.array_equal(m, repeated)]
        if options:
            return options[int(rng.integers(len(options)))].copy()
        mode = "random"
    if mode == "one_flip":
        choices = []
        weight = int(repeated.sum())
        for index in range(area):
            if (weight == 1 and repeated.flat[index]) or (weight == area - 1 and not repeated.flat[index]):
                continue
            choices.append(index)
        result = repeated.copy()
        index = choices[int(rng.integers(len(choices)))]
        result.flat[index] = not result.flat[index]
        return result
    if mode == "near_complement":
        result = ~repeated
        changes = int(rng.integers(0, min(3, area - 1)))
        if changes:
            indexes = rng.choice(area, size=changes, replace=False)
            result.flat[indexes] = ~result.flat[indexes]
        if 0 < int(result.sum()) < area and not np.array_equal(result, repeated):
            return result
    for _ in range(24):
        result = _new_mask(rng, repeated.shape[0])
        if not np.array_equal(result, repeated):
            return result
    return _outlier_mask(rng, repeated, "one_flip")


def _layout(rng, regime, split, ordinal, side):
    maximum = 30 // side
    if regime == "source_like":
        if split == "train" and ordinal < 2:
            return (("H", min(5, maximum)) if ordinal == 0 else ("V", 3))
        if split == "test":
            return "V", min(5, maximum)
    if regime == "reverse":
        if split == "train" and ordinal < 2:
            return (("V", min(5, maximum)) if ordinal == 0 else ("H", 3))
        if split == "test" and ordinal == 0:
            return "H", min(5, maximum)
    if regime == "horizontal":
        orientation = "H" if split == "train" or rng.random() < .7 else "V"
    elif regime == "vertical":
        orientation = "V" if split == "train" or rng.random() < .7 else "H"
    else:
        orientation = ("H" if ordinal % 2 == 0 else "V") if split == "train" and ordinal < 2 else ("H" if rng.random() < .5 else "V")
    return orientation, int(rng.integers(3, maximum + 1))


def _colors(rng, panel_count, background, outlier_index, witness):
    available = [c for c in range(10) if c != background]
    if witness:
        a, b = (int(v) for v in rng.choice(available, size=2, replace=False))
        colors = [a] * panel_count
        other_index = next(i for i in range(panel_count) if i != outlier_index)
        colors[other_index] = b
        return colors
    if panel_count <= 9 and rng.random() < .3:
        return [int(v) for v in rng.choice(available, size=panel_count, replace=False)]
    colors = [int(v) for v in rng.choice(available, size=panel_count, replace=True)]
    if len(set(colors)) == 1:
        replacement = next(c for c in available if c != colors[0])
        colors[int(rng.integers(panel_count))] = replacement
    return colors


def _construct_pair(rng, side, orientation, panel_count, background,
                    repeated, outlier, outlier_index, witness):
    colors = _colors(rng, panel_count, background, outlier_index, witness)
    panels = [np.where(outlier if i == outlier_index else repeated,
                       colors[i], background).astype(np.int8)
              for i in range(panel_count)]
    return {"input": np.concatenate(panels, axis=1 if orientation == "H" else 0),
            "output": panels[outlier_index].copy()}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_plain_integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    side = 3 if rng.random() < .78 else 4
    background = 0 if rng.random() < .42 else int(rng.integers(1, 10))
    vary_background = bool(rng.random() < .55)
    regime = ("source_like", "reverse", "horizontal", "vertical", "mixed")[int(rng.integers(5))]
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    roles = []
    first_outlier_index = None
    global_index = 0
    for split, count in (("train", train_count), ("test", test_count)):
        for ordinal in range(count):
            orientation, panel_count = _layout(rng, regime, split, ordinal, side)
            if vary_background and global_index == 1:
                choices = [c for c in range(10) if c != background]
                pair_background = int(rng.choice(choices))
            elif vary_background and rng.random() < .32:
                pair_background = int(rng.integers(0, 10))
            else:
                pair_background = background
            if split == "train" and ordinal == 0:
                repeated = _new_mask(rng, side)
                for _ in range(24):
                    if any(not np.array_equal(m, repeated) for m in _transforms(repeated)):
                        break
                    repeated = _new_mask(rng, side)
                if not any(not np.array_equal(m, repeated) for m in _transforms(repeated)):
                    repeated = np.zeros((side, side), dtype=bool)
                    repeated[0, 0] = True
                mode = "transform"
            elif split == "train" and ordinal == 1:
                repeated = roles[0][1].copy()
                mode = "one_flip"
            elif split == "test" and ordinal == 0:
                repeated = roles[1][1].copy()
                mode = "near_complement"
            else:
                if roles and rng.random() < .32:
                    prior = roles[int(rng.integers(len(roles)))]
                    repeated = prior[int(rng.integers(2))].copy()
                else:
                    repeated = _new_mask(rng, side)
                mode = ("one_flip", "near_complement", "transform", "random")[int(rng.integers(4))]
            outlier = _outlier_mask(rng, repeated, mode)
            outlier_index = int(rng.integers(panel_count))
            if split == "train" and ordinal == 1 and outlier_index == first_outlier_index:
                outlier_index = (outlier_index + 1) % panel_count
            witness = split == "train" and ordinal == 1
            for attempt in range(64):
                pair = _construct_pair(rng, side, orientation, panel_count,
                                       pair_background, repeated, outlier,
                                       outlier_index, witness)
                input_key = _array_key(pair["input"])
                if input_key not in used_inputs and _pair_key(pair["input"], pair["output"]) not in _SOURCE_PAIR_KEYS:
                    break
                if split == "train" and ordinal == 1:
                    outlier = _outlier_mask(rng, repeated, "one_flip")
                elif split == "test" and ordinal == 0:
                    outlier = _outlier_mask(rng, repeated, "near_complement")
                else:
                    repeated = _new_mask(rng, side)
                    outlier = _outlier_mask(rng, repeated, mode)
                outlier_index = int(rng.integers(panel_count))
                if split == "train" and ordinal == 1 and outlier_index == first_outlier_index:
                    outlier_index = (outlier_index + 1) % panel_count
            else:
                raise RuntimeError("unable to construct a unique pair")
            puzzle[split].append(pair)
            if split == "train" and ordinal == 0:
                first_outlier_index = outlier_index
            used_inputs.add(input_key)
            roles.append((repeated.copy(), outlier.copy()))
            global_index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, name):
    _require(isinstance(grid, np.ndarray), f"{name} must be a NumPy array")
    _require(grid.ndim == 2, f"{name} must have two dimensions")
    _require(all(1 <= dimension <= 30 for dimension in grid.shape), f"{name} dimensions are illegal")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must have integer cells")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{name} has an illegal color")


def _analyze_pair(pair, expected_side=None):
    inp, out = pair["input"], pair["output"]
    side = out.shape[0]
    _require(side in (3, 4) and out.shape == (side, side), "output must be one permitted square panel")
    if expected_side is not None:
        _require(side == expected_side, "panel size changed within puzzle")
    height, width = inp.shape
    if height == side and width % side == 0:
        orientation, panel_count, axis = "H", width // side, 1
    elif width == side and height % side == 0:
        orientation, panel_count, axis = "V", height // side, 0
    else:
        raise AssertionError("input is not a one-panel-wide strip")
    _require(3 <= panel_count <= 30 // side, "illegal panel count")
    panels = list(np.split(inp, panel_count, axis=axis))
    common = set(int(v) for v in np.unique(panels[0]))
    for panel in panels:
        common &= set(int(v) for v in np.unique(panel))
        _require(len(np.unique(panel)) == 2, "panel must have background and one foreground")
    _require(len(common) == 1, "shared background is not identifiable")
    background = common.pop()
    masks = [panel != background for panel in panels]
    keys = [_mask_key(mask) for mask in masks]
    groups = {key: [i for i, k in enumerate(keys) if k == key] for key in keys}
    _require(sorted(len(group) for group in groups.values()) == [1, panel_count - 1],
             "input lacks exactly one odd mask and one repeated mask")
    odd_index = next(group[0] for group in groups.values() if len(group) == 1)
    repeated_index = next(i for i in range(panel_count) if i != odd_index)
    _require(np.array_equal(out, panels[odd_index]), "output is not the exact odd panel")
    colors = tuple(next(int(v) for v in np.unique(panel) if v != background) for panel in panels)
    repeated, odd = masks[repeated_index], masks[odd_index]
    return {"side": side, "orientation": orientation, "panel_count": panel_count,
            "background": background, "colors": colors, "outlier_index": odd_index,
            "repeated_mask": _mask_key(repeated), "outlier_mask": _mask_key(odd),
            "weights": (int(repeated.sum()), int(odd.sum())),
            "distance": int(np.count_nonzero(repeated != odd)),
            "transform": any(np.array_equal(m, odd) for m in _transforms(repeated)),
            "color_collision": colors[odd_index] in [colors[i] for i in range(panel_count) if i != odd_index]}


def _infer_from_input(inp, side):
    if inp.shape[0] == side and inp.shape[1] % side == 0:
        panels = list(np.split(inp, inp.shape[1] // side, axis=1))
    elif inp.shape[1] == side and inp.shape[0] % side == 0:
        panels = list(np.split(inp, inp.shape[0] // side, axis=0))
    else:
        raise AssertionError("test strip cannot be segmented using training panel size")
    common = set(int(v) for v in np.unique(panels[0]))
    for panel in panels:
        common &= set(int(v) for v in np.unique(panel))
    _require(len(common) == 1, "test background is not inferable")
    background = common.pop()
    keys = [_mask_key(panel != background) for panel in panels]
    counts = {key: keys.count(key) for key in set(keys)}
    odd = [i for i, key in enumerate(keys) if counts[key] == 1]
    _require(len(odd) == 1, "test has no unique odd panel")
    return panels[odd[0]].copy()


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys are invalid")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts are illegal")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    side = None
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys are invalid")
            _validate_grid(pair["input"], f"{split}[{index}].input")
            _validate_grid(pair["output"], f"{split}[{index}].output")
            input_key = _array_key(pair["input"])
            pair_key = _pair_key(pair["input"], pair["output"])
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate input or pair")
            _require(pair_key not in _SOURCE_PAIR_KEYS, "source pair copied exactly")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _analyze_pair(pair, side)
            if side is None:
                side = analysis["side"]
            analyses[split].append(analysis)
    train = analyses["train"]
    _require(any(a["distance"] == 1 for a in train), "training lacks a one-cell difference")
    _require(any(a["transform"] and a["weights"][0] == a["weights"][1] for a in train),
             "training lacks an orientation-sensitive equal-density example")
    _require(any(a["color_collision"] for a in train), "training lacks a color-collision example")
    _require(len({a["outlier_index"] for a in train}) >= 2,
             "training always selects the same position")
    _require(any(a["outlier_mask"] == b["repeated_mask"]
                 for i, a in enumerate(train) for j, b in enumerate(train) if i != j),
             "training lacks a local motif role switch")
    _require(any(a["outlier_mask"] == b["repeated_mask"] or
                 a["repeated_mask"] == b["outlier_mask"]
                 for a in train for b in analyses["test"]),
             "test lacks a train/test motif role switch")
    for pair in puzzle["test"]:
        _require(np.array_equal(_infer_from_input(pair["input"].copy(), side), pair["output"]),
                 "test output cannot be inferred from the learned panel size and input")
    return None


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p["input"], p["output"]) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    for seed, counts in ((314159, (None, None)), (271828, (7, 3)),
                         (4000, (2, 1)), (4001, (10, 1)), (4002, (10, 4)),
                         (5000, (None, 1)), (5001, (None, 4)),
                         (5002, (2, None)), (5003, (10, None))):
        puzzle = generate_puzzle(seed, *counts)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, *counts)), "generation is not deterministic")
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "num_train not honored")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "num_test not honored")
    for bad in ((1, None), (11, None), (None, 0), (None, 5),
                (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(123, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "cross-seed puzzle diversity is low")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "automatic training counts do not vary")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "automatic test counts do not vary")
    analyses = [(p, {split: [_analyze_pair(pair) for pair in p[split]]
                     for split in ("train", "test")}) for p in puzzles]
    all_pairs = [a for _, entries in analyses for split in ("train", "test") for a in entries[split]]
    test_pairs = [a for _, entries in analyses for a in entries["test"]]
    _require({a["side"] for a in all_pairs} == {3, 4}, "panel size does not vary")
    _require({a["orientation"] for a in test_pairs} == {"H", "V"}, "test orientation does not vary")
    _require({a["background"] for a in all_pairs} == set(range(10)), "background palette coverage is low")
    _require(any(len({a["background"] for a in entries["train"] + entries["test"]}) > 1
                 for _, entries in analyses), "background never changes within a puzzle")
    _require(any(a["distance"] >= 8 and a["side"] == 3 for a in all_pairs), "large mask differences absent")
    _require({1, 8}.issubset({weight for a in all_pairs if a["side"] == 3 for weight in a["weights"]}),
             "extreme 3-by-3 mask weights absent")
    _require(any(a["panel_count"] >= 6 for a in all_pairs), "long strips absent")
    _require(any(len(set(a["colors"])) == a["panel_count"] for a in all_pairs), "source-like distinct colors absent")
    _require(any(a["color_collision"] for a in test_pairs), "test color collisions absent")
    _require(any(0 in a["colors"] for a in all_pairs), "zero never serves as foreground")
    _require(any(len({a["colors"][a["outlier_index"]] for a in entries["train"] + entries["test"]})
                 < len(entries["train"] + entries["test"])
                 for _, entries in analyses), "output colors never recur within puzzles")
    _require(any({a["orientation"] for a in entries["train"]} ==
                 {a["orientation"] for a in entries["test"]} and
                 len({a["orientation"] for a in entries["train"]}) == 1
                 for _, entries in analyses), "same-orientation train/test puzzles absent")
    _require(any({a["orientation"] for a in entries["train"]} == {"V"} and
                 any(a["orientation"] == "H" for a in entries["test"])
                 for _, entries in analyses), "reverse orientation shift absent")
    _require(any(any(t["repeated_mask"] == q["outlier_mask"] or t["outlier_mask"] == q["repeated_mask"]
                     for q in entries["train"] for t in entries["test"])
                 for _, entries in analyses), "train/test motif role switches absent")
    _require(any({a["orientation"] for a in entries["train"]} == {"H"}
                 for _, entries in analyses), "horizontal-only training absent")
    _require(any({a["orientation"] for a in entries["train"]} == {"V"}
                 for _, entries in analyses), "vertical-only training absent")
    _require(any({a["orientation"] for a in entries["train"]} == {"H", "V"}
                 for _, entries in analyses), "mixed training absent")
    damaged = {split: [{"input": pair["input"].copy(), "output": pair["output"].copy()}
                       for pair in puzzles[0][split]] for split in ("train", "test")}
    damaged["test"][0]["output"] = np.full_like(damaged["test"][0]["output"], 9)
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incorrect output")
