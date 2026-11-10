import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 1),
                (7, 2), (8, 3), (9, 4), (10, 4), (10, 1),
                (6, 4), (5, 1))
_SOURCE_INPUTS = (
    ("000000000000", "111111111111", "111111111111", "111111111111",
     "111111111111", "111111116111", "111111116611", "611111166661",
     "661111666666", "666616666666", "666666666666", "666666666666"),
    ("000000000000", "111111111111", "111111111111", "111111111111",
     "111111111111", "111111111111", "111111111111", "111111161116",
     "111111666166", "116116666666", "166666666666", "666666666666"),
    ("000000000000", "111111111111", "111111111111", "111111111111",
     "111111111111", "111111111111", "111111611111", "111116661111",
     "166166666611", "666666666661", "666666666666", "666666666666"),
)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _source_keys():
    result = set()
    for rows in _SOURCE_INPUTS:
        inp = np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)
        heights = np.count_nonzero(inp == 6, axis=0)
        peak = int(np.flatnonzero(heights == heights.max())[-1])
        valley = int(np.flatnonzero(heights == heights.min())[0])
        out = inp.copy()
        out[1:len(inp) - heights[peak], peak] = 4
        out[1:len(inp) - heights[valley], valley] = 9
        result.add((_signature(inp), _signature(out)))
    return result


def _count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, rng, num_train, num_test):
    _count(num_train, "num_train", 2, 10)
    _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        train, test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(test + 1, 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    dimension_mode = ("fixed", "height", "width", "both")[int(rng.integers(4))]
    return {
        "guides": (colors[0], colors[1]),
        "base_roles": tuple(colors[2:5]),
        "free_colors": tuple(colors[5:]),
        "variable_roles": int(rng.integers(1, 8)) if rng.random() < 0.62 else 0,
        "dimension_mode": dimension_mode,
        "base_shape": (int(rng.integers(9, 27)), int(rng.integers(8, 27))),
    }


def _roles(rng, parameters, index):
    roles = list(parameters["base_roles"])
    mask = parameters["variable_roles"]
    if index > 0 and mask:
        positions = [i for i in range(3) if mask & (1 << i)]
        replacements = rng.choice(parameters["free_colors"], size=len(positions), replace=False)
        for position, color in zip(positions, replacements):
            roles[position] = int(color)
    return tuple(roles)


def _shape(rng, parameters):
    base_h, base_w = parameters["base_shape"]
    mode = parameters["dimension_mode"]
    height = base_h if mode in ("fixed", "width") else int(rng.integers(9, 27))
    width = base_w if mode in ("fixed", "height") else int(rng.integers(8, 27))
    return height, width


def _positions(rng, width, order, tied, edge_mode, tie_style):
    candidates = []
    for peak in range(width):
        for valley in range(width):
            if (valley < peak) != (order == "valley_left") or valley == peak:
                continue
            if tied:
                left = [c for c in range(peak) if c != valley]
                if tie_style == "adjacent":
                    left = [c for c in left if c == peak - 1]
                else:
                    left = [c for c in left if c <= peak - 2]
                if not left:
                    continue
            peak_edge = peak in (0, width - 1)
            valley_edge = valley in (0, width - 1)
            if edge_mode == "interior" and (peak_edge or valley_edge):
                continue
            if edge_mode == "valley" and not valley_edge:
                continue
            if edge_mode == "peak" and not peak_edge:
                continue
            if edge_mode == "both" and not (peak_edge and valley_edge):
                continue
            candidates.append((peak, valley))
    if not candidates:
        return _positions(rng, width, order, tied, "any", tie_style)
    peak, valley = candidates[int(rng.integers(len(candidates)))]
    ties = []
    if tied:
        allowed = [c for c in range(peak) if c != valley and
                   ((c == peak - 1) if tie_style == "adjacent" else (c <= peak - 2))]
        first = int(rng.choice(allowed))
        ties.append(first)
        if tie_style == "separated" and len(allowed) > 1 and rng.random() < 0.24:
            extra = int(rng.choice([c for c in allowed if c != first]))
            ties.append(extra)
    return peak, valley, ties


def _profile(rng, width, low, high, peak, valley, ties, style):
    inner_low, inner_high = low + 1, high - 1
    if style == "flat":
        middle = int(rng.integers(inner_low, inner_high + 1))
        values = np.full(width, middle, dtype=np.int64)
        changes = rng.random(width) < 0.18
        values[changes] += rng.integers(-1, 2, size=int(changes.sum()))
    elif style == "jagged":
        values = rng.integers(inner_low, inner_high + 1, size=width)
    elif style == "ridge":
        x = np.arange(width)
        values = np.full(width, float(inner_low))
        for _ in range(int(rng.integers(2, 5))):
            center = int(rng.integers(width))
            radius = int(rng.integers(2, max(4, width // 2 + 1)))
            amplitude = float(rng.uniform(1.5, max(2.5, high - low)))
            values += amplitude * np.maximum(0, 1 - np.abs(x - center) / radius)
        values = np.rint(values).astype(np.int64)
    elif style == "notches":
        x = np.arange(width)
        values = np.full(width, high - 1.0)
        for _ in range(int(rng.integers(2, 5))):
            center = int(rng.integers(width))
            radius = int(rng.integers(1, max(3, width // 3 + 1)))
            depth = float(rng.uniform(1.5, max(2.5, high - low)))
            values -= depth * np.maximum(0, 1 - np.abs(x - center) / radius)
        values = np.rint(values).astype(np.int64)
    else:
        knots = {0, width - 1, peak, valley, *ties}
        extra_count = int(rng.integers(2, min(7, width)))
        knots.update(int(x) for x in rng.choice(width, size=extra_count, replace=False))
        xs = np.array(sorted(knots))
        ys = rng.integers(inner_low, inner_high + 1, size=len(xs))
        for i, x in enumerate(xs):
            if x == peak or x in ties:
                ys[i] = high
            elif x == valley:
                ys[i] = low
        values = np.rint(np.interp(np.arange(width), xs, ys)).astype(np.int64)
        if style == "terraces":
            step = int(rng.integers(2, 5))
            values = low + np.rint((values - low) / step).astype(np.int64) * step
    values = np.clip(values, inner_low, inner_high)
    values[valley] = low
    values[peak] = high
    for column in ties:
        values[column] = high
    return values


def _pair(rng, parameters, index, plan):
    height, width = _shape(rng, parameters)
    roles = _roles(rng, parameters, index)
    peak, valley, ties = _positions(rng, width, plan["order"], plan["tie"],
                                    plan["edge"], plan["tie_style"])
    min_guide = plan["min_guide"]
    max_air = height - 4
    if min_guide:
        peak_air = int(rng.integers(min_guide, max_air + 1))
    else:
        tier = int(rng.integers(3))
        ranges = ((1, min(2, max_air)), (3, min(6, max_air)),
                  (7, max_air))
        lo, hi = ranges[tier]
        if lo > hi:
            lo, hi = 1, max_air
        peak_air = int(rng.integers(lo, hi + 1))
    high = height - 1 - peak_air
    max_span = min(12, high - 1)
    if plan["style"] == "flat" and rng.random() < 0.65:
        max_span = min(max_span, 4)
    span = int(rng.integers(2, max_span + 1))
    low = high - span
    heights = _profile(rng, width, low, high, peak, valley, ties, plan["style"])
    border, air, terrain = roles
    inp = np.full((height, width), air, dtype=np.int8)
    inp[0] = border
    for column, stack in enumerate(heights):
        inp[height - stack:, column] = terrain
    out = inp.copy()
    out[1:height - high, peak] = parameters["guides"][0]
    out[1:height - low, valley] = parameters["guides"][1]
    return {"input": inp, "output": out}


def _plans(rng, train_count, test_count):
    first_order = ("valley_left", "peak_left")[int(rng.integers(2))]
    other_order = "peak_left" if first_order == "valley_left" else "valley_left"
    plans = []
    total = train_count + test_count
    for i in range(total):
        training = i < train_count
        if i == 0:
            order, tie, min_guide = first_order, False, 4
        elif i == 1:
            order, tie, min_guide = other_order, True, 3
        else:
            order = ("valley_left", "peak_left")[int(rng.integers(2))]
            tie = bool(rng.random() < (0.38 if training else 0.48))
            min_guide = 0
        edge = ("interior", "valley", "peak", "both", "any")[
            int(rng.integers(5))]
        if not training and rng.random() < 0.24:
            edge = "valley"
        style = ("smooth", "terraces", "ridge", "notches", "flat", "jagged")[
            int(rng.integers(6))]
        tie_style = ("adjacent", "separated")[int(rng.integers(2))]
        plans.append({"order": order, "tie": tie, "edge": edge,
                      "style": style, "tie_style": tie_style,
                      "min_guide": min_guide})
    return plans


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    h, w = inp.shape
    _require(inp.shape == out.shape and h >= 8 and w >= 8, "invalid terrain dimensions")
    colors = {int(x) for x in np.unique(inp)}
    _require(len(colors) == 3, "input needs three distinct roles")
    border, terrain = int(inp[0, 0]), int(inp[-1, 0])
    _require(np.all(inp[0] == border), "top border is not uniform")
    _require(np.all(inp[-1] == terrain), "bottom row is not terrain")
    _require(border != terrain, "border and terrain coincide")
    air = (colors - {border, terrain}).pop()
    heights = []
    for c in range(w):
        stack = int(np.count_nonzero(inp[1:, c] == terrain))
        _require(1 <= stack <= h - 2, "terrain stack cannot be measured")
        surface = h - stack
        _require(np.all(inp[1:surface, c] == air) and
                 np.all(inp[surface:, c] == terrain), "terrain is not bottom supported")
        heights.append(stack)
    heights = np.array(heights)
    peaks = np.flatnonzero(heights == heights.max())
    valleys = np.flatnonzero(heights == heights.min())
    _require(len(valleys) == 1 and heights.max() > heights.min(),
             "minimum must be unique and lower than maximum")
    peak, valley = int(peaks[-1]), int(valleys[0])
    peak_segment = out[1:h - heights[peak], peak]
    valley_segment = out[1:h - heights[valley], valley]
    _require(len(peak_segment) > 0 and len(valley_segment) > 0, "missing guide")
    guide_peak, guide_valley = int(peak_segment[0]), int(valley_segment[0])
    _require(guide_peak != guide_valley and
             guide_peak not in colors and guide_valley not in colors,
             "guide colors are not distinct from input roles")
    expected = inp.copy()
    expected[1:h - heights[peak], peak] = guide_peak
    expected[1:h - heights[valley], valley] = guide_valley
    _require(np.array_equal(out, expected), "incorrect extrema transformation")
    return {"roles": (border, air, terrain), "guides": (guide_peak, guide_valley),
            "shape": (h, w), "heights": tuple(int(x) for x in heights),
            "peak": peak, "valley": valley, "tie": len(peaks),
            "peak_air": len(peak_segment), "valley_air": len(valley_segment)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    pairs = puzzle["train"] + puzzle["test"]
    seen_inputs, seen_pairs = set(), set()
    source_keys = _source_keys()
    for pair in pairs:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must have input and output")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grid must be a 2D ndarray")
            h, w = grid.shape
            _require(1 <= h <= 30 and 1 <= w <= 30 and
                     np.issubdtype(grid.dtype, np.integer) and
                     int(grid.min()) >= 0 and int(grid.max()) <= 9,
                     "invalid ARC grid")
        key = (_signature(pair["input"]), _signature(pair["output"]))
        _require(key[0] not in seen_inputs and key not in seen_pairs,
                 "duplicate generated pair")
        _require(key not in source_keys, "generated pair copies source")
        seen_inputs.add(key[0])
        seen_pairs.add(key)
    infos = [_analyze(p) for p in pairs]
    train = infos[:nt]
    _require(len({info["guides"] for info in infos}) == 1,
             "puzzle guide meanings changed")
    guides = set(infos[0]["guides"])
    _require(all(guides.isdisjoint(info["roles"]) for info in infos),
             "a guide color also denotes an input role")
    _require({info["valley"] < info["peak"] for info in train} == {True, False},
             "training must show both extrema orders")
    _require(any(info["tie"] == 1 for info in train) and
             any(info["tie"] >= 2 for info in train),
             "training must show unique and tied peaks")
    _require(any(info["peak_air"] >= 4 and info["valley_air"] >= 6
                 for info in train), "training needs substantial guides")
    if nt == 2:
        _require(all(info["peak_air"] >= 3 for info in train),
                 "two-pair training cannot have a one-cell peak guide")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _parameters(rng)
    plans = _plans(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    source_keys = _source_keys()
    for index, plan in enumerate(plans):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for _ in range(32):
            pair = _pair(pair_rng, parameters, index, plan)
            key = (_signature(pair["input"]), _signature(pair["output"]))
            if key[0] not in seen and key not in source_keys:
                break
        else:
            raise AssertionError("could not construct unique pair")
        seen.add(key[0])
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic counts lack variety")
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42,
             "seeded puzzles repeat")
    shapes, profiles, palettes = set(), set(), set()
    stable_palettes = varied_palettes = fixed_sizes = varied_sizes = 0
    interior_ties = adjacent_ties = separated_ties = 0
    short_peak = substantial_peak = flat_span = broad_span = 0
    test_orders = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = [_analyze(pair) for pair in puzzle["train"] + puzzle["test"]]
        roles = {info["roles"] for info in infos}
        sizes = {info["shape"] for info in infos}
        stable_palettes += len(roles) == 1
        varied_palettes += len(roles) > 1
        fixed_sizes += len(sizes) == 1
        varied_sizes += len(sizes) > 1
        for info in infos:
            shapes.add(info["shape"])
            profiles.add((info["shape"], info["heights"]))
            palettes.add(info["roles"] + info["guides"])
            short_peak += info["peak_air"] <= 2
            substantial_peak += info["peak_air"] >= 4
            span = max(info["heights"]) - min(info["heights"])
            flat_span += span <= 3
            broad_span += span >= 7
            if info["tie"] > 1:
                maxima = [i for i, h in enumerate(info["heights"]) if h == max(info["heights"])]
                interior_ties += info["peak"] < info["shape"][1] - 1
                adjacent_ties += any(b - a == 1 for a, b in zip(maxima, maxima[1:]))
                separated_ties += any(b - a > 1 for a, b in zip(maxima, maxima[1:]))
        test_orders.update(info["valley"] < info["peak"] for info in infos[len(puzzle["train"]):])
    _require(len(shapes) >= 90 and len(profiles) >= 250 and len(palettes) >= 100,
             "structural or color diversity is too low")
    _require(stable_palettes >= 8 and varied_palettes >= 12 and
             fixed_sizes >= 7 and varied_sizes >= 20,
             "within-puzzle palette or size regimes are missing")
    _require(interior_ties >= 25 and adjacent_ties >= 10 and separated_ties >= 10,
             "maximum ties lack positional diversity")
    _require(short_peak >= 10 and substantial_peak >= 100 and
             flat_span >= 15 and broad_span >= 30 and test_orders == {True, False},
             "guide lengths, terrain spans, or test orders lack diversity")
    a = generate_puzzle(777, 7, 3)
    b = generate_puzzle(777, 7, 3)
    _require(_puzzle_signature(a) == _puzzle_signature(b), "generation is nondeterministic")
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(2000 + i, *counts)
        _require((len(p["train"]), len(p["test"])) == counts, "explicit counts ignored")
    for i, (nt, nq) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(3000 + i, nt, nq)
        _require((nt is None or len(p["train"]) == nt) and
                 (nq is None or len(p["test"]) == nq), "partial count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(12345, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
