import numpy as np


_SOURCE_INPUT_ROWS = {
    ("77777777777", "77777777777", "27272727272", "57575757585", "78777777787", "78777877787", "78777878717", "71787878717", "71787178717", "71787171717", "71717171717"),
    ("777777777", "777777777", "777777777", "777777777", "272827272", "575858585", "787178787", "787171717", "717171717"),
    ("7777777777777", "7777777777777", "7777777777777", "7777777777777", "7777777777777", "7777777777777", "2727272728272", "5758575758575", "7878777778787", "7878787871787", "7871787871787", "7171787171717", "7171717171717"),
    ("777777777777777", "777777777777777", "777777777777777", "777777777777777", "777777777777777", "272727272727272", "585757575757575", "787777777777787", "787777787777787", "787778787877787", "717878787878787", "717878787878717", "717878717878717", "717871717178717", "717171717171717"),
}


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + (seed // 4) % 9)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _sample_profile(rng, lanes, maximum):
    style = int(rng.integers(0, 5))
    interior = maximum >= 4 and rng.random() < 0.30
    low, high = (2, maximum - 1) if interior else (1, maximum)
    if style == 0:
        values = rng.integers(low, high + 1, size=lanes)
    elif style == 1:
        values = np.full(lanes, int(rng.integers(low, high + 1)))
    elif style == 2:
        a, b = rng.integers(low, high + 1, size=2)
        if a == b and low < high:
            b = low if a != low else high
        split = int(rng.integers(1, lanes))
        values = np.array([a] * split + [b] * (lanes - split))
        if rng.random() < 0.5:
            values = values[::-1]
    elif style == 3:
        a, b = sorted(int(x) for x in rng.integers(low, high + 1, size=2))
        if a == b and low < high:
            a, b = low, high
        values = np.rint(np.linspace(a, b, lanes)).astype(int)
        if rng.random() < 0.5:
            values = values[::-1]
    else:
        half = rng.integers(low, high + 1, size=(lanes + 1) // 2)
        values = np.concatenate((half, half[:lanes // 2][::-1]))
    if not interior and style != 1 and rng.random() < 0.22:
        values[int(rng.integers(0, lanes))] = maximum
    return tuple(int(v) for v in values)


def _sample_geometry(rng, regime):
    if regime == "source_train":
        if rng.random() < 0.32:
            height = int(rng.choice((9, 11, 13)))
            lanes = (height - 1) // 2
        else:
            height = int(rng.integers(7, 25))
            lanes = int(rng.integers(3, 11))
        thickness = int(rng.integers(1, min(5, height - 2) + 1))
    else:
        if rng.random() < 0.18:
            height, lanes, thickness = 15, 7, 4
        else:
            height = int(rng.integers(7, 31))
            lanes = int(rng.integers(3, 13))
            thickness = int(rng.integers(1, min(7, height - 2) + 1))
    guide = int(rng.integers(thickness - 1, height - 2))
    return height, lanes, thickness, guide


def _build_pair(height, lanes, thickness, guide, lengths, dent, colors, rotation):
    background, band_marker, terminal_marker, band, terminal, complement = colors
    width = 2 * lanes + 1
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[guide, 0::2] = band_marker
    inp[guide + 1, 0::2] = terminal_marker
    out[:, 0::2] = inp[:, 0::2]
    band_start = guide - thickness + 1
    for lane, length in enumerate(lengths):
        col = 2 * lane + 1
        input_end = height - length
        inp[input_end - thickness:input_end, col] = band
        inp[input_end:, col] = terminal
        out[band_start:guide + 1, col] = band
        out[guide + 1:guide + 1 + length, col] = terminal
        out[guide + 1 + length:, col] = complement
    if dent is not None:
        col = 2 * dent + 1
        inp[height - lengths[dent] - thickness, col] = background
    return np.rot90(inp, -rotation).copy(), np.rot90(out, -rotation).copy()


def _inspect_canonical(grid):
    h, w = grid.shape
    _check(7 <= h <= 30 and 7 <= w <= 25 and w % 2 == 1,
           "canonical dimensions are invalid")
    lanes = w // 2
    _check(lanes >= 3, "too few data lanes")
    separators = grid[:, 0::2]
    spine = separators[:, 0]
    _check(np.all(separators == spine[:, None]), "separator lanes differ")
    vals, counts = np.unique(spine, return_counts=True)
    background = int(vals[np.argmax(counts)])
    guide_rows = np.flatnonzero(spine != background)
    _check(len(guide_rows) == 2 and guide_rows[1] == guide_rows[0] + 1,
           "two adjacent guides are required")
    guide = int(guide_rows[0])
    _check(0 <= guide <= h - 3, "guide leaves no terminal space")
    band_marker, terminal_marker = int(spine[guide]), int(spine[guide + 1])
    _check(len({background, band_marker, terminal_marker}) == 3,
           "guide roles are not distinct")
    data = grid[:, 1::2]
    terminal = int(data[-1, 0])
    _check(np.all(data[-1] == terminal), "terminal suffixes miss border")
    all_colors = {int(c) for c in np.unique(grid)}
    rest = all_colors - {background, band_marker, terminal_marker, terminal}
    _check(len(all_colors) == 5 and len(rest) == 1,
           "five distinct visible roles are required")
    band = next(iter(rest))
    _check({int(c) for c in np.unique(data)} <= {background, band, terminal},
           "marker color leaks into data lanes")
    lengths, visible = [], []
    for lane in range(lanes):
        col = data[:, lane]
        end = h
        while end > 0 and col[end - 1] == terminal:
            end -= 1
        length = h - end
        start = end
        while start > 0 and col[start - 1] == band:
            start -= 1
        seen = end - start
        _check(length >= 1 and seen >= 1 and np.all(col[:start] == background),
               "lane is not a background/band/terminal stack")
        lengths.append(length)
        visible.append(seen)
    thickness = max(visible)
    dents = [i for i, size in enumerate(visible) if size < thickness]
    _check(sum(size == thickness for size in visible) >= 2,
           "common thickness has too little evidence")
    _check(len(dents) <= 1 and all(size in (thickness, thickness - 1) for size in visible),
           "only one one-cell boundary dent is supported")
    _check(thickness >= 2 or not dents, "a one-cell band cannot be dented")
    _check(guide + 1 >= thickness, "band cannot reach first guide")
    _check(max(lengths) <= h - guide - 1, "terminal run exceeds output tail")
    _check(max(lengths) + thickness <= h, "input stack exceeds canvas")
    return dict(height=h, width=w, lanes=lanes, thickness=thickness,
                guide=guide, lengths=tuple(lengths), visible=tuple(visible),
                dents=tuple(dents), roles=(background, band_marker,
                terminal_marker, band, terminal))


def _inspect_input(grid):
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(grid, rotation)
        try:
            info = _inspect_canonical(canonical)
        except AssertionError:
            continue
        info["rotation"] = rotation
        info["canonical"] = canonical
        candidates.append(info)
    _check(len(candidates) == 1, "stack direction is missing or ambiguous")
    return candidates[0]


def _predict_from_input(grid, info, complement):
    h, w = info["height"], info["width"]
    background, band_marker, terminal_marker, band, terminal = info["roles"]
    _check(complement not in info["roles"], "complement is an input role")
    result = np.full((h, w), background, dtype=np.int8)
    result[:, 0::2] = info["canonical"][:, 0::2]
    guide, thickness = info["guide"], info["thickness"]
    for lane, length in enumerate(info["lengths"]):
        col = 2 * lane + 1
        result[guide - thickness + 1:guide + 1, col] = band
        result[guide + 1:guide + 1 + length, col] = terminal
        result[guide + 1 + length:, col] = complement
    _check(result[guide, 0] == band_marker and
           result[guide + 1, 0] == terminal_marker, "markers changed")
    return np.rot90(result, -info["rotation"]).copy()


def _solve_from_training(train_pairs, test_input):
    new_colors = set()
    for pair in train_pairs:
        new_colors |= set(np.unique(pair["output"])) - set(np.unique(pair["input"]))
    _check(len(new_colors) == 1, "training does not teach the complement color")
    complement = int(next(iter(new_colors)))
    info = _inspect_input(test_input)
    return _predict_from_input(test_input, info, complement)


def _generate_pair(rng, regime, complement, rotation, geometry=None):
    for _ in range(32):
        if geometry is None:
            height, lanes, thickness, guide = _sample_geometry(rng, regime)
        else:
            height, lanes, thickness, guide = geometry
        maximum = height - guide - 1
        lengths = _sample_profile(rng, lanes, maximum)
        dent = (int(rng.integers(0, lanes))
                if thickness >= 2 and rng.random() < 0.19 else None)
        visible = [int(c) for c in rng.choice(
            [c for c in range(10) if c != complement], size=5, replace=False)]
        inp, out = _build_pair(height, lanes, thickness, guide,
                               lengths, dent, (*visible, complement), rotation)
        if np.array_equal(inp, out) or _rows(inp) in _SOURCE_INPUT_ROWS:
            continue
        return {"input": inp, "output": out}
    raise ValueError("could not construct a nondegenerate pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    complement = int(rng.integers(0, 10))
    constant_orientation = bool(rng.random() < 0.28)
    common_rotation = int(rng.integers(0, 4))
    train_rotations = ([common_rotation] * train_count if constant_orientation
                       else [int(rng.integers(0, 4)) for _ in range(train_count)])
    test_rotations = ([common_rotation] * test_count if constant_orientation
                      else [int(rng.integers(0, 4)) for _ in range(test_count)])
    if not constant_orientation and rng.random() < 0.28:
        unseen = [r for r in range(4) if r not in train_rotations]
        if unseen:
            test_rotations[0] = int(rng.choice(unseen))
    puzzle = {"train": [], "test": []}
    used = set()
    for section, rotations in (("train", train_rotations), ("test", test_rotations)):
        for rotation in rotations:
            geometry = None
            prior_lengths = set()
            if section == "test" and rng.random() < 0.24:
                exemplar = puzzle["train"][int(rng.integers(0, train_count))]
                example_info = _inspect_input(exemplar["input"])
                geometry = (example_info["height"], example_info["lanes"],
                            example_info["thickness"], example_info["guide"])
                for train_pair in puzzle["train"]:
                    train_info = _inspect_input(train_pair["input"])
                    train_geometry = (train_info["height"], train_info["lanes"],
                                      train_info["thickness"], train_info["guide"])
                    if train_geometry == geometry:
                        prior_lengths.add(train_info["lengths"])
            for _ in range(64):
                pair = _generate_pair(rng, "source_train" if section == "train"
                                      else "target_test", complement, rotation, geometry)
                if geometry is not None:
                    candidate_lengths = _inspect_input(pair["input"])["lengths"]
                    if candidate_lengths in prior_lengths:
                        continue
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("could not construct a unique pair")
    train_infos = [_inspect_input(p["input"]) for p in puzzle["train"]]
    taught = (any(complement in np.unique(p["output"]) for p in puzzle["train"])
              and any(len(set(i["lengths"])) > 1 for i in train_infos)
              and any(i["thickness"] >= 2 for i in train_infos))
    if not taught:
        replace_index = int(rng.integers(0, train_count))
        for _ in range(64):
            replacement = _generate_pair(rng, "source_train", complement,
                                         train_rotations[replace_index])
            candidate_info = _inspect_input(replacement["input"])
            other = (puzzle["train"][:replace_index] +
                     puzzle["train"][replace_index + 1:] + puzzle["test"])
            if (candidate_info["thickness"] >= 2 and
                    len(set(candidate_info["lengths"])) > 1 and
                    complement in np.unique(replacement["output"]) and
                    _key(replacement["input"]) not in {_key(p["input"]) for p in other}):
                puzzle["train"][replace_index] = replacement
                break
        else:
            raise ValueError("training did not reveal the full rule")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle requires train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    new_colors = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair requires input and output")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a 2-D NumPy array")
                _check(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _check(all(1 <= dim <= 30 for dim in grid.shape),
                       "grid dimensions exceed ARC bounds")
                _check(bool(np.all((grid >= 0) & (grid <= 9))),
                       "grid has an invalid color")
            inp, out = pair["input"], pair["output"]
            _check(inp.shape == out.shape and not np.array_equal(inp, out),
                   "transformation must preserve dimensions and change the grid")
            key, pkey = _key(inp), (_key(inp), _key(out))
            _check(key not in inputs and pkey not in pairs, "duplicate pair or input")
            inputs.add(key)
            pairs.add(pkey)
            _check(_rows(inp) not in _SOURCE_INPUT_ROWS, "source input copied")
            info = _inspect_input(inp)
            analyses[section].append(info)
            added = {int(c) for c in np.unique(out)} - {int(c) for c in np.unique(inp)}
            _check(len(added) <= 1, "multiple output-only colors")
            if section == "train":
                new_colors |= added
    _check(len(new_colors) == 1, "training must expose one complement color")
    _check(any(len(set(info["lengths"])) > 1 for info in analyses["train"]),
           "training must demonstrate unequal terminal lengths")
    _check(any(info["thickness"] >= 2 for info in analyses["train"]),
           "training must demonstrate a multi-cell band")
    complement = next(iter(new_colors))
    for section in ("train", "test"):
        for pair, info in zip(puzzle[section], analyses[section]):
            _check(complement not in set(np.unique(pair["input"])),
                   "complement cannot appear in an input")
            predicted = _predict_from_input(pair["input"], info, complement)
            _check(np.array_equal(predicted, pair["output"]),
                   "pair does not follow guide re-anchoring")
    for pair in puzzle["test"]:
        _check(np.array_equal(_solve_from_training(puzzle["train"], pair["input"]),
                              pair["output"]),
               "test output cannot be reconstructed from training and test input")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    a, b = generate_puzzle(19), generate_puzzle(19)
    _check(_puzzle_key(a) == _puzzle_key(b), "automatic counts are not deterministic")
    a, b = generate_puzzle(2027, 7, 3), generate_puzzle(2027, 7, 3)
    _check(_puzzle_key(a) == _puzzle_key(b) and
           (len(a["train"]), len(a["test"])) == (7, 3),
           "explicit counts are not honored deterministically")
    for i, (t, s) in enumerate(((2, 1), (10, 1), (10, 4),
                                (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(4100 + i, t, s)
        _check((t is None or len(p["train"]) == t) and
               (s is None or len(p["test"]) == s), "partial count not honored")
    for i, args in enumerate(((1, None), (11, None), (None, 0),
                              (None, 5), (2, 2), (3, 4))):
        try:
            generate_puzzle(4300 + i, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, thicknesses, guides = set(), set(), set(), set(), set()
    orientations, profiles, colors, backgrounds, signatures = set(), set(), set(), set(), set()
    style_hits = set()
    clean_puzzles = varying_palettes = held_out_orientations = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        local = []
        for section in ("train", "test"):
            for pair in p[section]:
                info = _inspect_input(pair["input"])
                _check(np.array_equal(_solve_from_training(p["train"], pair["input"]),
                                      pair["output"]), "reconstruction failed")
                shapes.add(pair["input"].shape)
                thicknesses.add(info["thickness"])
                guides.add(info["guide"])
                orientations.add(info["rotation"])
                profiles.add(info["lengths"])
                colors.add(info["roles"])
                backgrounds.add(info["roles"][0])
                local.append((pair["input"].shape, info["thickness"], info["guide"],
                              info["rotation"], info["lengths"], info["dents"]))
                x = info["lengths"]
                if len(set(x)) == 1:
                    style_hits.add("flat")
                if x == x[::-1]:
                    style_hits.add("symmetric")
                if list(x) == sorted(x) or list(x) == sorted(x, reverse=True):
                    style_hits.add("ramp")
                if min(x) > 1 and max(x) < info["height"] - info["guide"] - 1:
                    style_hits.add("interior")
                if max(x) == info["height"] - info["guide"] - 1:
                    style_hits.add("exact_fit")
                if info["dents"]:
                    style_hits.add("dent")
        signatures.add(tuple(local))
        train_infos = [_inspect_input(pair["input"]) for pair in p["train"]]
        test_infos = [_inspect_input(pair["input"]) for pair in p["test"]]
        if not any(info["dents"] for info in train_infos + test_infos):
            clean_puzzles += 1
        if len({info["roles"] for info in train_infos + test_infos}) > 1:
            varying_palettes += 1
        if {i["rotation"] for i in test_infos} - {i["rotation"] for i in train_infos}:
            held_out_orientations += 1
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _check(len(shapes) >= 40 and len(thicknesses) >= 6 and len(guides) >= 15,
           "geometry lacks diversity")
    _check(len(orientations) == 4 and len(profiles) >= 100,
           "orientation or profiles lack diversity")
    _check(len(colors) >= 100 and len(backgrounds) >= 8,
           "visible palettes lack diversity")
    _check(len(signatures) >= 40 and clean_puzzles >= 4 and
           varying_palettes >= 35 and held_out_orientations >= 4,
           "puzzle-level combinations lack diversity")
    _check({"flat", "symmetric", "ramp", "interior", "exact_fit", "dent"}
           <= style_hits, "a profile or boundary mode was not observed")
