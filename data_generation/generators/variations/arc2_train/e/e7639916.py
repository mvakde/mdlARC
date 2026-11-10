"""Generate three-landmark, tight-bounding-rectangle ARC puzzles."""

import itertools
import numpy as np


_SOURCE = frozenset((shape, tuple(sorted(points))) for shape, points in (
    ((8, 12), ((1, 3), (3, 9), (6, 2))),
    ((10, 20), ((1, 3), (2, 17), (8, 10))),
    ((11, 13), ((1, 8), (6, 2), (9, 11))),
    ((14, 13), ((3, 3), (5, 11), (12, 8))),
))
_COUNTS = tuple((a, b) for b in range(1, 5) for a in range(max(2, b + 1), 11))

# A corner subset plus the sides holding any noncorner landmarks. The twenty
# descriptors compose all valid one-, two-, and three-corner arrangements.
_ARRANGEMENTS = []
for i, sides in enumerate((("right", "bottom"), ("left", "bottom"),
                           ("top", "right"), ("top", "left"))):
    _ARRANGEMENTS.append(((i,), sides))
for pair in itertools.combinations(range(4), 2):
    if pair == (0, 1):
        sides = ("bottom",)
    elif pair == (2, 3):
        sides = ("top",)
    elif pair == (0, 2):
        sides = ("right",)
    elif pair == (1, 3):
        sides = ("left",)
    else:
        sides = ("top", "bottom", "left", "right")
    _ARRANGEMENTS.extend((pair, (s,)) for s in sides)
_ARRANGEMENTS.extend((corners, ()) for corners in itertools.combinations(range(4), 3))


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt is not None and ns is not None:
        if nt <= ns:
            raise ValueError("num_train must exceed num_test")
        return nt, ns
    if nt is None and ns is None:
        return _COUNTS[seed % len(_COUNTS)]
    if nt is None:
        choices = list(range(max(2, ns + 1), 11))
        return choices[seed % len(choices)], ns
    choices = list(range(1, min(4, nt - 1) + 1))
    return nt, choices[seed % len(choices)]


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _geometry_key(shape, points):
    return tuple(shape), tuple(sorted((int(r), int(c)) for r, c in points))


def _draw(input_grid, marker, outline):
    points = np.argwhere(input_grid == marker)
    top, bottom = int(points[:, 0].min()), int(points[:, 0].max())
    left, right = int(points[:, 1].min()), int(points[:, 1].max())
    result = input_grid.copy()
    result[top, left:right + 1] = outline
    result[bottom, left:right + 1] = outline
    result[top:bottom + 1, left] = outline
    result[top:bottom + 1, right] = outline
    result[input_grid == marker] = marker
    return result


def _length(rng, regime):
    weights = (0.27, 0.53, 0.20) if regime == "source_train" else (0.22, 0.48, 0.30)
    lo, hi = ((3, 6), (7, 16), (17, 29))[int(rng.choice(3, p=weights))]
    return int(rng.integers(lo, hi + 1))


def _slack(rng, maximum):
    if maximum == 0:
        return 0
    draw = rng.random()
    if draw < 0.20:
        return 0
    if draw < 0.70 or maximum < 5:
        return int(rng.integers(1, min(4, maximum) + 1))
    return int(rng.integers(5, maximum + 1))


def _geometry(rng, regime):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    rh, rw = _length(rng, regime), _length(rng, regime)
    if rng.random() < 0.12:  # Promote the square case present in source train 3.
        rw = rh
    extra_h, extra_w = _slack(rng, 30 - rh), _slack(rng, 30 - rw)
    top, left = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    return rh + extra_h, rw + extra_w, top, left, rh, rw


def _local_landmarks(rng, rh, rw, descriptor):
    corners = ((0, 0), (0, rw - 1), (rh - 1, 0), (rh - 1, rw - 1))
    selected, sides = descriptor
    fixed = tuple(corners[i] for i in selected)
    options = {
        "top": [(0, c) for c in range(1, rw - 1)],
        "bottom": [(rh - 1, c) for c in range(1, rw - 1)],
        "left": [(r, 0) for r in range(1, rh - 1)],
        "right": [(r, rw - 1) for r in range(1, rh - 1)],
    }
    candidates = []
    for added in itertools.product(*(options[s] for s in sides)):
        points = fixed + added
        if len(set(points)) != 3:
            continue
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
               for a, b in itertools.combinations(points, 2)):
            continue
        candidates.append(points)
    if not candidates:
        return None
    if rng.random() < 0.30:
        near = [points for points in candidates if any(
            (r in (0, rh - 1) and c in (1, rw - 2)) or
            (c in (0, rw - 1) and r in (1, rh - 2))
            for r, c in points if (r, c) not in corners)]
        if near:
            candidates = near
    return candidates[int(rng.integers(len(candidates)))]


def _palette(rng, outline, previous):
    colors = [c for c in range(10) if c != outline]
    background = int(rng.choice([c for c in colors if previous is None or c != previous[0]]))
    marker = int(rng.choice([c for c in colors if c != background and
                             (previous is None or c != previous[1])]))
    return background, marker


def _pair(rng, regime, descriptor, outline, previous, used, required_aspects,
          top_bounds=None):
    required_box, required_canvas = required_aspects
    for attempt in range(100):
        if attempt < 96:
            h, w, top, left, rh, rw = _geometry(rng, regime)
        else:
            rh, rw = ((6, 7), (7, 6), (8, 8), (9, 10))[attempt - 96]
            top = 0 if top_bounds is None else top_bounds[0]
            h, w, left = rh + top + 1, rw + 1, 0
        box_aspect = "square" if rh == rw else "wide" if rw > rh else "tall"
        canvas_aspect = "square" if h == w else "wide" if w > h else "tall"
        if required_box is not None and box_aspect != required_box:
            continue
        if required_canvas is not None and canvas_aspect != required_canvas:
            continue
        if top_bounds is not None and not top_bounds[0] <= top <= top_bounds[1]:
            continue
        local = _local_landmarks(rng, rh, rw, descriptor)
        if local is None:
            continue
        points = tuple((top + r, left + c) for r, c in local)
        if _geometry_key((h, w), points) in _SOURCE:
            continue
        background, marker = _palette(rng, outline, previous)
        inp = np.full((h, w), background, dtype=np.int8)
        for r, c in points:
            inp[r, c] = marker
        key = _grid_key(inp)
        if key in used:
            continue
        used.add(key)
        return {"input": inp, "output": _draw(inp, marker, outline)}, (background, marker)
    raise RuntimeError("unable to construct a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, ns = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    outline = int(rng.integers(10))  # One convention learned from every train output.
    first = _ARRANGEMENTS[int(rng.integers(4))]
    other = [a for a in _ARRANGEMENTS if a != first]
    train_arrangements = [first] + [other[int(i)] for i in rng.choice(len(other), size=nt - 1, replace=False)]
    rng.shuffle(train_arrangements)
    # Most puzzles freely mix geometry. A few deliberately hold the same
    # aspect on both splits or reverse the source's wide-to-tall shift.
    mode = seed % 12
    if mode == 0:
        train_aspects = test_aspects = ("wide", "wide")
        train_top_bounds = test_top_bounds = None
    elif mode == 1:
        train_aspects = test_aspects = ("tall", "tall")
        train_top_bounds = test_top_bounds = None
    elif mode == 2:
        train_aspects, test_aspects = ("tall", None), ("wide", None)
        train_top_bounds = test_top_bounds = None
    elif mode == 3:
        train_aspects, test_aspects = ("wide", None), ("tall", None)
        train_top_bounds, test_top_bounds = (0, 2), (4, 30)
    else:
        train_aspects = test_aspects = (None, None)
        train_top_bounds = test_top_bounds = None
    test_arrangements = []
    for _ in range(ns):
        novel = [a for a in _ARRANGEMENTS if a not in train_arrangements]
        pool = novel if novel and rng.random() < 0.62 else _ARRANGEMENTS
        test_arrangements.append(pool[int(rng.integers(len(pool)))])
    puzzle = {"train": [], "test": []}
    used, previous = set(), None
    for split, regime, arrangements, aspects, top_bounds in (
        ("train", "source_train", train_arrangements, train_aspects, train_top_bounds),
        ("test", "target_test", test_arrangements, test_aspects, test_top_bounds),
    ):
        for descriptor in arrangements:
            pair, previous = _pair(rng, regime, descriptor, outline, previous, used,
                                   aspects, top_bounds)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be an ndarray")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= n <= 30 for n in grid.shape), f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must contain integers")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has invalid colors")


def _analyze_pair(pair, regime):
    _require(regime in ("source_train", "target_test"), "invalid regime")
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
    inp, out = pair["input"], pair["output"]
    _validate_grid(inp, f"{regime} input")
    _validate_grid(out, f"{regime} output")
    _require(inp.shape == out.shape, "dimensions changed")
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 2, "input must use background and marker colors")
    indices = np.flatnonzero(counts == 3)
    _require(len(indices) == 1, "input must have three same-color landmarks")
    marker = int(values[int(indices[0])])
    background = int(values[1 - int(indices[0])])
    points = np.argwhere(inp == marker)
    _require(all(abs(int(a[0])-int(b[0])) + abs(int(a[1])-int(b[1])) > 1
                 for a, b in itertools.combinations(points, 2)), "landmarks touch")
    top, bottom = int(points[:, 0].min()), int(points[:, 0].max())
    left, right = int(points[:, 1].min()), int(points[:, 1].max())
    rh, rw = bottom - top + 1, right - left + 1
    _require(rh >= 3 and rw >= 3, "rectangle must have an interior")
    _require(all(r in (top, bottom) or c in (left, right) for r, c in points),
             "landmarks must lie on the perimeter")
    _require(_geometry_key(inp.shape, points) not in _SOURCE, "source geometry copied")
    output_colors = set(int(v) for v in np.unique(out))
    new = output_colors - {background, marker}
    _require(len(new) == 1, "output must introduce exactly one outline color")
    outline = next(iter(new))
    _require(output_colors == {background, marker, outline}, "output palette is invalid")
    _require(np.array_equal(out, _draw(inp, marker, outline)),
             "output differs from tight bounding perimeter")
    _require(np.count_nonzero(inp != out) >= 1, "perimeter made no change")
    corners = sum(int(r in (top, bottom) and c in (left, right)) for r, c in points)
    arrangement = tuple(sorted(
        (1 if r == top else 0) | (2 if r == bottom else 0) |
        (4 if c == left else 0) | (8 if c == right else 0)
        for r, c in points))
    return {"background": background, "marker": marker, "outline": outline,
            "bbox": (top, bottom, left, right), "rectangle_shape": (rh, rw),
            "coordinates": points.copy(), "arrangement": arrangement,
            "corner_count": corners,
            "margins": (top, inp.shape[0]-1-bottom, left, inp.shape[1]-1-right)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    records = {"train": [], "test": []}
    for split, regime in (("train", "source_train"), ("test", "target_test")):
        for pair in puzzle[split]:
            record = _analyze_pair(pair, regime)
            records[split].append(record)
            ik, ok = _grid_key(pair["input"]), _grid_key(pair["output"])
            _require(ik not in seen_inputs and (ik, ok) not in seen_pairs, "duplicate generated pair")
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
    _require(len({r["outline"] for rows in records.values() for r in rows}) == 1,
             "outline color must be learned once per puzzle")
    _require(any(r["corner_count"] < 3 for r in records["train"]),
             "training must show a landmark partway along a side")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _predict_from_training(train, test_input):
    learned = set()
    for pair in train:
        learned.update(set(int(v) for v in np.unique(pair["output"])) -
                       set(int(v) for v in np.unique(pair["input"])))
    _require(len(learned) == 1, "training does not reveal a unique outline color")
    outline = next(iter(learned))
    values, counts = np.unique(test_input, return_counts=True)
    marker = int(values[int(np.flatnonzero(counts == 3)[0])])
    points = np.argwhere(test_input == marker)
    top, bottom = int(points[:, 0].min()), int(points[:, 0].max())
    left, right = int(points[:, 1].min()), int(points[:, 1].max())
    predicted = test_input.copy()
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r == top or r == bottom or c == left or c == right) and predicted[r, c] != marker:
                predicted[r, c] = outline
    return predicted


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds do not vary")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    records = []
    regime_counts = {"same": 0, "mixed": 0, "reverse": 0,
                     "small_test_top": 0, "large_test_top": 0,
                     "novel_test_layout": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        analyzed = {
            split: [_analyze_pair(p, "source_train" if split == "train" else "target_test")
                    for p in puzzle[split]]
            for split in ("train", "test")
        }
        combined = analyzed["train"] + analyzed["test"]
        _require(len({r["background"] for r in combined}) > 1 and
                 len({r["marker"] for r in combined}) > 1,
                 "pair-level palette does not vary within puzzle")
        _require(len({r["outline"] for r in combined}) == 1,
                 "outline convention changes within puzzle")
        _require(len({r["rectangle_shape"] for r in combined}) > 1,
                 "box shape does not vary within puzzle")
        aspects = [("square" if h == w else "wide" if w > h else "tall")
                   for h, w in (r["rectangle_shape"] for r in combined)]
        regime_counts["same" if len(set(aspects)) == 1 else "mixed"] += 1
        train_aspects, test_aspects = aspects[:len(analyzed["train"])], aspects[len(analyzed["train"]):]
        regime_counts["reverse"] += int(all(a == "tall" for a in train_aspects) and
                                        all(a == "wide" for a in test_aspects))
        regime_counts["small_test_top"] += int(
            min(r["margins"][0] for r in analyzed["test"]) <=
            max(r["margins"][0] for r in analyzed["train"]))
        regime_counts["large_test_top"] += int(
            min(r["margins"][0] for r in analyzed["test"]) >
            max(r["margins"][0] for r in analyzed["train"]))
        regime_counts["novel_test_layout"] += int(
            any(r["arrangement"] not in {q["arrangement"] for q in analyzed["train"]}
                for r in analyzed["test"]))
        for pair in puzzle["test"]:
            examples = [{"input": p["input"].copy(), "output": p["output"].copy()}
                        for p in puzzle["train"]]
            _require(np.array_equal(_predict_from_training(examples, pair["input"].copy()),
                                    pair["output"]), "training-only prediction failed")
        for split in ("train", "test"):
            records.extend(_analyze_pair(p, "source_train" if split == "train" else "target_test")
                           for p in puzzle[split])
    shapes = {r["rectangle_shape"] for r in records}
    _require(len(shapes) >= 35 and len({r["arrangement"] for r in records}) >= 15,
             "structural diversity too narrow")
    _require({1, 2, 3}.issubset({r["corner_count"] for r in records}), "corner diversity missing")
    _require(any(h == w for h, w in shapes) and any(h < w for h, w in shapes)
             and any(h > w for h, w in shapes), "box aspect diversity missing")
    _require(any(0 in r["margins"] for r in records), "boundary cases missing")
    _require(any(max(r["margins"]) >= 8 for r in records), "large margins missing")
    _require(any(max(r["rectangle_shape"]) >= 25 for r in records), "large boxes missing")
    _require(any(min(r["rectangle_shape"]) <= 4 for r in records), "small boxes missing")
    _require(regime_counts["same"] >= 4 and regime_counts["mixed"] >= 20,
             "same- and mixed-aspect puzzles are not both represented")
    _require(regime_counts["reverse"] >= 2, "reversed aspect shifts missing")
    _require(regime_counts["small_test_top"] >= 20, "test top margins remain predictive")
    _require(regime_counts["large_test_top"] >= 2, "source-style upper-margin shift missing")
    _require(regime_counts["novel_test_layout"] >= 20, "test layouts merely copy training")
    _require(_puzzle_key(generate_puzzle(123)) == _puzzle_key(generate_puzzle(123)),
             "automatic generation is nondeterministic")
    for nt, ns in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(31415 + nt + ns, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                 "explicit counts were ignored")
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(31415 + nt + ns, nt, ns)),
                 "explicit generation is nondeterministic")
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(27182, nt, ns)
        _require(nt is None or len(puzzle["train"]) == nt, "partial train count ignored")
        _require(ns is None or len(puzzle["test"]) == ns, "partial test count ignored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(5, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
