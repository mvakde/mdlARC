import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))
_SOURCE = (
    (("90099", "00000", "00909", "00440", "44400", "40404"), ("60606", "66600", "60000")),
    (("00990", "99099", "09000", "44000", "40444", "04004"), ("66660", "06600", "00006")),
    (("09900", "90009", "90000", "00404", "44040", "40440"), ("06006", "06066", "00660")),
    (("09000", "09909", "90009", "44040", "04440", "44000"), ("60060", "00066", "06006")),
    (("99090", "90090", "09999", "40040", "44044", "44404"), ("06000", "06006", "60060")),
    (("00909", "09000", "90900", "40444", "04404", "40000"), ("60060", "00606", "00600")),
    (("99090", "00909", "00099", "44404", "40444", "44000"), ("00666", "60060", "66066")),
)
_STATES = ((False, False), (True, False), (False, True), (True, True))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    if a is None and b is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if a is None:
        a = max(2, b + 1) + seed % (11 - max(2, b + 1))
    if b is None:
        b = 1 + seed % min(4, a - 1)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_signatures():
    def rows_signature(rows):
        return ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))
    return {(rows_signature(a), rows_signature(b)) for a, b in _SOURCE}


def _pick_shape(rng, orientation, need_cue, test):
    # The output is always one unrotated panel. The long side may approach
    # the ARC canvas limit in the direction not used to join panels.
    max_h = 15 if orientation == "vertical" else 30
    max_w = 30 if orientation == "vertical" else 15
    category = int(rng.choice(6, p=(.25, .17, .15, .20, .13, .10)))
    if category == 0:  # small and source-like
        h, w = int(rng.integers(2, 7)), int(rng.integers(3, 8))
    elif category == 1:  # wide
        h = int(rng.integers(2, min(max_h, 7) + 1))
        w = int(rng.integers(min(max_w, 8), max_w + 1))
    elif category == 2:  # tall
        h = int(rng.integers(min(max_h, 7), max_h + 1))
        w = int(rng.integers(2, min(max_w, 7) + 1))
    elif category == 3:  # roughly square
        side = int(rng.integers(3, min(max_h, max_w, 14) + 1))
        h = min(max_h, max(2, side + int(rng.integers(-2, 3))))
        w = min(max_w, max(2, side + int(rng.integers(-2, 3))))
    elif category == 4:  # large or full canvas
        if int(rng.integers(0, 6)) == 0 and not need_cue:
            h, w = max_h, max_w
        else:
            h = int(rng.integers(max(7, max_h // 2), max_h + 1))
            w = int(rng.integers(max(7, max_w // 2), max_w + 1))
    else:
        h, w = 3, 5
    if need_cue:
        # A mixed-layout test must identify its split from input dimensions.
        if orientation == "vertical" and w % 2 == 0:
            w = w - 1 if w > 2 else 3
        if orientation == "horizontal" and h % 2 == 0:
            h = h - 1 if h > 2 else 3
    return h, w


def _ranked_mask(rng, h, w, count):
    rows, cols = np.indices((h, w))
    style = int(rng.integers(0, 5))
    noise = rng.random((h, w))
    if style == 0:
        score = noise
    elif style == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        score = abs(rows - r) + abs(cols - c) + .3 * noise
    elif style == 2:
        score = np.full((h, w), float(h + w))
        for _ in range(int(rng.integers(2, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            score = np.minimum(score, abs(rows - r) + abs(cols - c))
        score += .3 * noise
    elif style == 3:
        if int(rng.integers(2)):
            score = abs(rows - int(rng.integers(h))) + .3 * noise
        else:
            score = abs(cols - int(rng.integers(w))) + .3 * noise
    else:
        score = (rows + cols + int(rng.integers(2))) % 2 + .3 * noise
    mask = np.zeros(h * w, bool)
    mask[np.argsort(score, axis=None)[:count]] = True
    return mask.reshape(h, w)


def _primitive(rng, h, w):
    style = int(rng.integers(0, 7))
    mask = np.zeros((h, w), bool)
    if style in (0, 1):
        fraction = float(rng.uniform(.10, .75))
        return _ranked_mask(rng, h, w, min(h * w - 1, max(1, round(fraction * h * w))))
    if style in (2, 3):  # filled or outlined rectangle
        r0 = int(rng.integers(h))
        c0 = int(rng.integers(w))
        r1 = int(rng.integers(r0 + 1, h + 1))
        c1 = int(rng.integers(c0 + 1, w + 1))
        mask[r0:r1, c0:c1] = True
        if style == 3 and r1 - r0 >= 3 and c1 - c0 >= 3:
            mask[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = False
    elif style == 4:  # one or several bars
        for _ in range(int(rng.integers(1, 4))):
            if int(rng.integers(2)):
                mask[int(rng.integers(h)), :] = True
            else:
                mask[:, int(rng.integers(w))] = True
    elif style == 5:  # separated components
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[r:min(h, r + int(rng.integers(1, 4))),
                 c:min(w, c + int(rng.integers(1, 4)))] = True
    else:  # checker or diagonal motif
        rows, cols = np.indices((h, w))
        mask = (rows + cols + int(rng.integers(2))) % int(rng.integers(2, 4)) == 0
    if not mask.any() or mask.all():
        return _ranked_mask(rng, h, w, max(1, min(h * w - 1, h * w // 3)))
    return mask


def _masks(rng, h, w, mode):
    n = h * w
    if mode == "compact_xor":
        count = int(rng.integers(2, n))
        xor = _ranked_mask(rng, h, w, count)
        a = np.zeros(n, bool)
        b = np.zeros(n, bool)
        exclusive = np.flatnonzero(xor.ravel())
        rng.shuffle(exclusive)
        cut = int(rng.integers(1, len(exclusive)))
        a[exclusive[:cut]] = True
        b[exclusive[cut:]] = True
        shared = np.flatnonzero(~xor.ravel())
        if len(shared):
            rng.shuffle(shared)
            overlap = int(rng.integers(len(shared) + 1))
            a[shared[:overlap]] = True
            b[shared[:overlap]] = True
        return a.reshape(h, w), b.reshape(h, w)
    if mode == "identical":
        a = _primitive(rng, h, w)
        return a, a.copy()
    if mode == "near":
        a = _primitive(rng, h, w)
        b = a.copy()
        positions = rng.choice(n, size=int(rng.integers(1, min(4, n) + 1)), replace=False)
        b.flat[positions] = ~b.flat[positions]
        if b.all():
            b.flat[int(rng.choice(np.flatnonzero(a)))] = False
        elif not b.any():
            b.flat[int(rng.choice(np.flatnonzero(~a)))] = True
        return a, b
    if mode == "complement":
        a = _primitive(rng, h, w)
        return a, ~a
    if mode == "contained":
        a = _primitive(rng, h, w)
        if a.sum() < 2:
            a = _ranked_mask(rng, h, w, max(2, n // 2))
        b = np.zeros((h, w), bool)
        cells = np.flatnonzero(a)
        chosen = rng.choice(cells, size=int(rng.integers(1, len(cells))), replace=False)
        b.flat[chosen] = True
        return a, b
    if mode == "disjoint":
        a = _primitive(rng, h, w)
        available = np.flatnonzero(~a)
        b = np.zeros((h, w), bool)
        chosen = rng.choice(available, size=int(rng.integers(1, len(available) + 1)), replace=False)
        b.flat[chosen] = True
        return a, b
    if mode == "same_disjoint":
        # A repeated small motif is placed at two different local positions.
        mh, mw = min(h, max(1, h // 3)), min(w, max(1, w // 3))
        motif = rng.random((mh, mw)) < .65
        motif[0, 0] = True
        a = np.zeros((h, w), bool)
        b = np.zeros((h, w), bool)
        a[:mh, :mw] = motif
        b[h - mh:, w - mw:] = motif
        return a, b
    if mode == "unequal":
        small = _ranked_mask(rng, h, w, max(1, n // 12))
        large = _ranked_mask(rng, h, w, min(n - 1, max(2, 3 * n // 4)))
        return (small, large) if int(rng.integers(2)) else (large, small)
    if mode == "empty":
        return np.zeros((h, w), bool), _primitive(rng, h, w)
    if mode == "full":
        return np.ones((h, w), bool), _primitive(rng, h, w)
    if mode == "informative":
        for _ in range(40):
            a, b = _primitive(rng, h, w), _primitive(rng, h, w)
            if all(np.any((a == x) & (b == y)) for x, y in _STATES):
                return a, b
        states = rng.permutation(n) % 4
        return (states.reshape(h, w) & 1) != 0, (states.reshape(h, w) & 2) != 0
    return _primitive(rng, h, w), _primitive(rng, h, w)


def _orientation(layout, index, split, rng):
    if layout == "vertical" or layout == "horizontal":
        return layout
    if layout == "switch_vh":
        return "vertical" if split == "train" else "horizontal"
    if layout == "switch_hv":
        return "horizontal" if split == "train" else "vertical"
    if split == "train" and index < 2:
        return ("vertical", "horizontal")[index]
    return ("vertical", "horizontal")[int(rng.integers(2))]


def _mode(rng, density_regime, split, fixed_palette, index):
    if split == "train" and (index == 0 or (index == 1 and not fixed_palette)):
        return "informative"
    names = ["free", "compact_xor", "near", "identical", "complement",
             "contained", "disjoint", "same_disjoint", "unequal"]
    weights = np.array([20, 14, 10, 8, 8, 9, 9, 8, 14], dtype=float)
    if split == "test" and density_regime == "denser":
        weights *= np.array([1, 2, .4, .3, 2, .5, 1.8, 1.3, 1.2])
    if split == "test" and density_regime == "sparser":
        weights *= np.array([1, .6, 2, 2, .3, 1.5, .6, .6, .7])
    if split == "test" and density_regime == "steady":
        weights *= np.array([1.4, 1.3, .7, .7, .7, 1, 1, 1, 1])
    if fixed_palette:
        names += ["empty", "full"]
        weights = np.append(weights, [5., 5.])
    return str(rng.choice(names, p=weights / weights.sum()))


def _make_pair(rng, orientation, shape, palette, mode):
    h, w = shape
    background, first, second, result = palette
    a, b = _masks(rng, h, w, mode)
    if int(rng.integers(2)):
        a, b = b, a
    p = np.where(a, first, background).astype(np.int8)
    q = np.where(b, second, background).astype(np.int8)
    inp = np.vstack((p, q)) if orientation == "vertical" else np.hstack((p, q))
    out = np.where(a ^ b, result, background).astype(np.int8)
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    result = int(rng.integers(10))
    fixed_palette = bool(rng.integers(0, 3) == 0)
    layout = str(rng.choice(["vertical", "horizontal", "mixed", "switch_vh", "switch_hv"],
                            p=[.24, .20, .22, .17, .17]))
    density_regime = str(rng.choice(["mixed", "denser", "sparser", "steady"]))
    base_colors = [int(x) for x in rng.permutation([c for c in range(10) if c != result])[:3]]
    puzzle = {"train": [], "test": []}
    used = set()
    source = _source_signatures()
    first_shape = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            orientation = _orientation(layout, index, split, rng)
            need_cue = layout not in ("vertical", "horizontal")
            for attempt in range(100):
                shape = _pick_shape(rng, orientation, need_cue, split == "test")
                if first_shape is not None and index == 1 and split == "train" and shape == first_shape:
                    continue
                if fixed_palette:
                    colors = base_colors.copy()
                else:
                    colors = [int(x) for x in rng.permutation([c for c in range(10) if c != result])[:3]]
                    if split == "train" and index == 1 and colors == base_colors:
                        continue
                if int(rng.integers(2)):
                    colors[1], colors[2] = colors[2], colors[1]
                palette = (*colors, result)
                mode = _mode(rng, density_regime, split, fixed_palette, index)
                pair = _make_pair(rng, orientation, shape, palette, mode)
                sig = (_signature(pair["input"]), _signature(pair["output"]))
                if sig in source or sig[0] in used:
                    continue
                used.add(sig[0])
                puzzle[split].append(pair)
                if first_shape is None:
                    first_shape = shape
                break
            else:
                raise ValueError("could not produce a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= d <= 30 for d in grid.shape), f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must contain integers")
    _require(grid.min() >= 0 and grid.max() <= 9, f"{label} has illegal colors")


def _panels(inp, orientation):
    if orientation == "vertical":
        _require(inp.shape[0] % 2 == 0, "vertical split is not even")
        return np.split(inp, 2, axis=0)
    _require(inp.shape[1] % 2 == 0, "horizontal split is not even")
    return np.split(inp, 2, axis=1)


def _train_orientation(pair):
    inp, out = pair["input"], pair["output"]
    options = []
    if inp.shape[0] == 2 * out.shape[0] and inp.shape[1] == out.shape[1]:
        options.append("vertical")
    if inp.shape[1] == 2 * out.shape[1] and inp.shape[0] == out.shape[0]:
        options.append("horizontal")
    _require(len(options) == 1, "training output does not define one panel split")
    return options[0]


def _test_orientation(inp, learned_orientations):
    rows, cols = inp.shape
    if rows % 2 == 0 and cols % 2 == 1:
        return "vertical"
    if rows % 2 == 1 and cols % 2 == 0:
        return "horizontal"
    _require(rows % 2 == 0 and cols % 2 == 0, "test input has no equal split")
    _require(len(learned_orientations) == 1, "test input split is ambiguous")
    return next(iter(learned_orientations))


def _background(p, q, known_backgrounds):
    common = set(int(x) for x in np.unique(p)) & set(int(x) for x in np.unique(q))
    if len(common) == 1:
        return next(iter(common))
    _require(not common and len(known_backgrounds) == 1, "background is ambiguous")
    return next(iter(known_backgrounds))


def _analyze(pair, orientation, known_backgrounds, result=None):
    inp, out = pair["input"], pair["output"]
    p, q = _panels(inp, orientation)
    _require(out.shape == p.shape, "output does not have one-panel dimensions")
    bg = _background(p, q, known_backgrounds)
    pv, qv = set(int(x) for x in np.unique(p)), set(int(x) for x in np.unique(q))
    _require(len(pv - {bg}) <= 1 and len(qv - {bg}) <= 1, "panel has multiple foreground colors")
    _require(pv - {bg} != qv - {bg} or not (pv - {bg}), "input foreground roles coincide")
    a, b = p != bg, q != bg
    if result is None:
        result_candidates = set(int(x) for x in np.unique(out)) - {bg}
        _require(len(result_candidates) == 1, "first training pair must show the result color")
        result = next(iter(result_candidates))
    _require(result != bg and result not in pv and result not in qv,
             "result color must be distinct from this input palette")
    expected = np.where(a ^ b, result, bg)
    _require(np.array_equal(out, expected), "output does not show coordinatewise XOR")
    return {"orientation": orientation, "shape": p.shape, "background": bg,
            "result": result, "a": a, "b": b, "density": float((a ^ b).mean()),
            "states": {(x, y): int(np.count_nonzero((a == x) & (b == y)))
                       for x, y in _STATES},
            "palette": (bg, next(iter(pv - {bg}), None), next(iter(qv - {bg}), None))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            _grid(pair["input"], "input")
            _grid(pair["output"], "output")
            key = (_signature(pair["input"]), _signature(pair["output"]))
            _require(key[0] not in seen_inputs and key not in seen_pairs, "duplicate input or pair")
            _require(key not in _source_signatures(), "source pair copied")
            seen_inputs.add(key[0])
            seen_pairs.add(key)
    first = puzzle["train"][0]
    first_analysis = _analyze(first, _train_orientation(first), set())
    result = first_analysis["result"]
    backgrounds = {first_analysis["background"]}
    orientations = {first_analysis["orientation"]}
    analyses = [first_analysis]
    for pair in puzzle["train"][1:]:
        analysis = _analyze(pair, _train_orientation(pair), backgrounds, result)
        analyses.append(analysis)
        backgrounds.add(analysis["background"])
        orientations.add(analysis["orientation"])
    _require(all(any(x["states"][state] for x in analyses) for state in _STATES),
             "training does not expose all four occupancy states")
    _require(all(any(x["states"][state] for x in analyses)
                 for state in ((True, False), (False, True))),
             "training does not show both exclusive contributions")
    # Decode tests using only training-derived result color and split convention,
    # plus colors and dimensions visible in each test input.
    for pair in puzzle["test"]:
        orientation = _test_orientation(pair["input"], orientations)
        _analyze(pair, orientation, backgrounds, result)


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variation")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variation")
    shapes, orientations, palettes, densities, state_patterns = set(), set(), set(), set(), set()
    varied_shapes = varied_palettes = vertical_only = horizontal_only = mixed = 0
    identical = complementary = disjoint = contained = one_state_missing = 0
    near = almost_overlap = unequal = empty_or_full = 0
    thin = square = wide = tall = source_size = full_canvas = 0
    sparse_tests = dense_trains = size_larger = size_smaller = 0
    reversed_roles = fixed_palettes = changing_backgrounds = 0
    switched_vh = switched_hv = overlapping_density = 0
    density_relations = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_shapes, local_palettes, local_orientations = set(), set(), set()
        local_backgrounds, local_color_sets, local_roles = set(), set(), {}
        train_orientations, test_orientations = set(), set()
        train_areas, test_areas = [], []
        tr_densities, te_densities = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                ori = _train_orientation(pair)
                # This recomputation uses the visible split and input colors.
                p, q = _panels(pair["input"], ori)
                common = set(np.unique(p).tolist()) & set(np.unique(q).tolist())
                if len(common) != 1:
                    common = set(np.unique(pair["output"]).tolist()) & set(np.unique(p).tolist() + np.unique(q).tolist())
                bg = next(iter(common))
                a, b = p != bg, q != bg
                expected = a ^ b
                _require(np.array_equal(expected, pair["output"] != bg), "independent XOR failure")
                local_shapes.add(p.shape)
                local_orientations.add(ori)
                (train_orientations if split == "train" else test_orientations).add(ori)
                (train_areas if split == "train" else test_areas).append(p.size)
                local_backgrounds.add(bg)
                first_color = next(iter(set(np.unique(p).tolist()) - {bg}), None)
                second_color = next(iter(set(np.unique(q).tolist()) - {bg}), None)
                local_color_sets.add((bg, tuple(sorted(
                    c for c in (first_color, second_color) if c is not None))))
                if first_color is not None:
                    local_roles.setdefault(first_color, set()).add(0)
                if second_color is not None:
                    local_roles.setdefault(second_color, set()).add(1)
                local_palettes.add((bg, tuple(sorted(set(np.unique(p).tolist()) - {bg})),
                                    tuple(sorted(set(np.unique(q).tolist()) - {bg}))))
                shapes.add(p.shape)
                orientations.add(ori)
                palettes.add((bg, tuple(sorted(set(np.unique(p).tolist()) - {bg})),
                              tuple(sorted(set(np.unique(q).tolist()) - {bg}))))
                densities.add((int(expected.sum()), expected.size))
                state_patterns.add((p.shape, tuple((a.astype(np.uint8) + 2 * b.astype(np.uint8)).flat)))
                identical += int(np.array_equal(a, b))
                complementary += int(np.array_equal(a, ~b))
                disjoint += int(not np.any(a & b))
                contained += int(np.all(~a | b) or np.all(~b | a))
                one_state_missing += int(any(not np.any((a == x) & (b == y)) for x, y in _STATES))
                xor_count = int(expected.sum())
                overlap_count = int(np.count_nonzero(a & b))
                union_count = int(np.count_nonzero(a | b))
                first_only = int(np.count_nonzero(a & ~b))
                second_only = int(np.count_nonzero(~a & b))
                near += int(1 <= xor_count <= 3)
                almost_overlap += int(overlap_count > 0 and
                                      0 < xor_count <= max(2, union_count // 10))
                unequal += int(xor_count > 0 and
                               min(first_only, second_only) / xor_count < .2)
                empty_or_full += int(a.all() or b.all() or not a.any() or not b.any())
                h, w = p.shape
                thin += int(min(h, w) <= 3)
                square += int(.8 <= h / w <= 1.25)
                wide += int(w >= 2 * h)
                tall += int(h >= 2 * w)
                source_size += int((h, w) == (3, 5))
                full_canvas += int(pair["input"].shape == (30, 30))
                (tr_densities if split == "train" else te_densities).append(float(expected.mean()))
        varied_shapes += len(local_shapes) > 1
        varied_palettes += len(local_palettes) > 1
        fixed_palettes += len(local_color_sets) == 1
        changing_backgrounds += len(local_backgrounds) > 1
        reversed_roles += any(len(roles) > 1 for roles in local_roles.values())
        vertical_only += local_orientations == {"vertical"}
        horizontal_only += local_orientations == {"horizontal"}
        mixed += len(local_orientations) == 2
        switched_vh += train_orientations == {"vertical"} and test_orientations == {"horizontal"}
        switched_hv += train_orientations == {"horizontal"} and test_orientations == {"vertical"}
        sparse_tests += any(value < .25 for value in te_densities)
        dense_trains += any(value > .70 for value in tr_densities)
        size_larger += any(area > max(train_areas) for area in test_areas)
        size_smaller += any(area < min(train_areas) for area in test_areas)
        overlapping_density += any(min(tr_densities) <= value <= max(tr_densities)
                                   for value in te_densities)
        delta = np.mean(te_densities) - np.mean(tr_densities)
        density_relations.add("denser" if delta > .12 else "sparser" if delta < -.12 else "overlap")
    _require(len(shapes) >= 25 and len(state_patterns) >= 180, "structure diversity is too small")
    _require(len(palettes) >= 30 and len(densities) >= 30, "palette or density diversity is too small")
    _require(varied_shapes >= 30 and varied_palettes >= 15, "within-puzzle variation is too small")
    _require(vertical_only >= 2 and horizontal_only >= 2 and mixed >= 5, "layout modes missing")
    _require(identical > 0 and complementary > 0 and disjoint > 0 and contained > 0
             and one_state_missing > 0, "mask relationships missing")
    _require(near > 0 and almost_overlap > 0 and unequal > 0 and empty_or_full > 0,
             "edge-case mask relationships missing")
    _require(all(value > 0 for value in (thin, square, wide, tall, source_size, full_canvas)),
             "panel shape categories missing")
    _require(all(value > 0 for value in (sparse_tests, dense_trains, size_larger,
                                        size_smaller, overlapping_density)),
             "density or size regimes missing")
    _require(all(value > 0 for value in (reversed_roles, fixed_palettes,
                                        changing_backgrounds, switched_vh, switched_hv)),
             "palette or orientation regimes missing")
    _require(density_relations == {"denser", "sparser", "overlap"}, "density regimes missing")
    for seed, a, b in ((8001, 2, 1), (8002, 10, 1), (8003, 10, 4), (8004, 7, 3)):
        p = generate_puzzle(seed, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    for seed, a, b in ((9001, None, 1), (9002, None, 4), (9003, 2, None), (9004, 10, None)):
        p = generate_puzzle(seed, a, b)
        _require((a is None or len(p["train"]) == a) and (b is None or len(p["test"]) == b),
                 "partial counts ignored")
    _require(_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3)),
             "generation is nondeterministic")
    _require(not _equal(generate_puzzle(123456, 6, 2), generate_puzzle(123457, 6, 2)),
             "different seeds match")
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(77, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count request accepted")
