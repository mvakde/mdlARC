"""Synthetic examples for the solid-neutral-rectangle ARC rule."""

import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = frozenset({
    "3ae258952f9e52475d75bced28c97cef928d21ca0f5f9ae9e9678f69fa06c6bc",
    "d92fedd46f9a39941c7239858320ab8b840fb56013123f5ad01b2730250ff856",
    "173e54f2e508a60b053427d3a47d030f803316cd9b7aca0ad41ad5cfa8175a06",
    "2d3d09c5dfce055539c297e2d5760f1cce1fb13535f2ef5c7209695a94237ec6",
})


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (not _integer(value) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(array):
    return (array.shape, array.tobytes())


def _source_digest(pair):
    a = pair["input"].astype(np.uint8)
    b = pair["output"].astype(np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _largest_rectangles(grid, color):
    """Largest all-color rectangles of at least 3 by 3; bounds are exclusive."""
    height, width = grid.shape
    best_area = 0
    best = []
    for top in range(height):
        columns = np.ones(width, dtype=bool)
        for bottom in range(top, height):
            columns &= grid[bottom] == color
            rectangle_height = bottom - top + 1
            if rectangle_height < 3:
                continue
            x = 0
            while x < width:
                if not columns[x]:
                    x += 1
                    continue
                left = x
                while x < width and columns[x]:
                    x += 1
                area = rectangle_height * (x - left)
                if x - left >= 3 and area >= best_area:
                    rectangle = (top, bottom + 1, left, x)
                    if area > best_area:
                        best_area, best = area, [rectangle]
                    else:
                        best.append(rectangle)
    return best_area, best


def _largest_any_color(grid):
    winner = None
    best_area = 0
    tied = False
    for color in np.unique(grid):
        area, rectangles = _largest_rectangles(grid, int(color))
        if area > best_area:
            winner = (int(color), rectangles[0]) if len(rectangles) == 1 else None
            best_area, tied = area, len(rectangles) != 1
        elif area == best_area and area > 0:
            tied = True
    return None if tied else winner


def _masks(shape, rectangle):
    top, bottom, left, right = rectangle
    whole = np.zeros(shape, dtype=bool)
    inside = np.zeros(shape, dtype=bool)
    whole[top:bottom, left:right] = True
    inside[top + 1:bottom - 1, left + 1:right - 1] = True
    return whole, inside


def _ring_mask(shape, rectangle):
    top, bottom, left, right = rectangle
    height, width = shape
    ring = np.zeros(shape, dtype=bool)
    ring[max(0, top - 1):min(height, bottom + 1),
         max(0, left - 1):min(width, right + 1)] = True
    whole, _ = _masks(shape, rectangle)
    ring[whole] = False
    return ring


def _solve_from_training(train_pairs, test_input):
    """Predict using only displayed demonstrations and the new input."""
    highlights = set()
    for pair in train_pairs:
        changed = pair["input"] != pair["output"]
        _require(bool(changed.any()), "training must demonstrate a change")
        _require(len(np.unique(pair["output"][changed])) == 1,
                 "training has inconsistent highlight cells")
        highlights.add(int(pair["output"][changed][0]))
    _require(len(highlights) == 1, "training must establish one highlight color")
    highlight = next(iter(highlights))
    _require(not bool(np.any(test_input == highlight)),
             "test input must not already contain the highlight")
    winner = _largest_any_color(test_input)
    _require(winner is not None, "test input lacks a unique largest rectangle")
    _, rectangle = winner
    _, interior = _masks(test_input.shape, rectangle)
    result = test_input.copy()
    result[interior] = highlight
    return result


def _analyze_pair(pair):
    source, output = pair["input"], pair["output"]
    _require(source.shape == output.shape, "input/output shapes differ")
    changed = source != output
    _require(bool(changed.any()), "output must differ from input")
    _require(len(np.unique(source[changed])) == 1, "changed input role varies")
    _require(len(np.unique(output[changed])) == 1, "highlight role varies")
    neutral = int(source[changed][0])
    highlight = int(output[changed][0])
    _require(neutral != highlight and not bool(np.any(source == highlight)),
             "highlight must be a new, distinct output color")
    area, rectangles = _largest_rectangles(source, neutral)
    _require(len(rectangles) == 1, "neutral target is not uniquely largest")
    rectangle = rectangles[0]
    _require(_largest_any_color(source) == (neutral, rectangle),
             "target must be the unique largest solid rectangle across colors")
    whole, interior = _masks(source.shape, rectangle)
    _require(np.array_equal(changed, interior), "only the strict interior may change")
    _require(bool(np.all(output[interior] == highlight)),
             "interior must become highlight")
    _require(bool(np.all(output[~interior] == source[~interior])),
             "all border and texture cells must remain unchanged")
    ring = _ring_mask(source.shape, rectangle)
    ring_neutral = int(np.count_nonzero((source == neutral) & ring))
    _require(ring_neutral >= 1, "rectangle exterior needs incidental neutral cells")
    outside = ~whole
    density = float(np.count_nonzero((source == neutral) & outside) /
                    np.count_nonzero(outside))
    _require(0.08 <= density <= 0.78, "surrounding neutral density is implausible")
    other_colors = [int(c) for c in np.unique(source) if c != neutral]
    _require(other_colors, "input needs texture")
    outside_only = source.copy()
    outside_only[whole] = -1
    competitor_area, _ = _largest_rectangles(outside_only, neutral)
    _require(competitor_area <= area * 0.8,
             "another neutral rectangle is too close in size")
    top, bottom, left, right = rectangle
    return {
        "neutral": neutral, "highlight": highlight,
        "rectangle": rectangle, "height": bottom - top, "width": right - left,
        "ring_neutral": ring_neutral, "density": density,
        "texture_counts": tuple(int(np.count_nonzero(source == c)) for c in other_colors),
        "competitor_area": competitor_area,
    }


def _parameters(rng):
    colors = [int(c) for c in rng.permutation(10)]
    neutral_count = 2 if rng.random() < 0.32 else 1
    neutrals = tuple(colors[1:1 + neutral_count])
    textures = colors[1 + neutral_count:]
    return {
        "highlight": colors[0],
        "neutrals": neutrals,
        "train_textures": tuple(textures[:3]),
        "fresh_textures": tuple(textures[3:]),
        "texture_regime": int(rng.integers(0, 3)),  # shared, new, mixed
        "train_base": (int(rng.integers(14, 25)), int(rng.integers(14, 25))),
        "test_shift": (int(rng.integers(-5, 6)), int(rng.integers(-5, 6))),
    }


def _canvas(rng, parameters, regime):
    base_h, base_w = parameters["train_base"]
    if regime == "target_test":
        shift_h, shift_w = parameters["test_shift"]
        base_h += shift_h
        base_w += shift_w
    height = int(np.clip(base_h + rng.integers(-4, 5), 11, 29))
    width = int(np.clip(base_w + rng.integers(-4, 5), 11, 29))
    if rng.random() < 0.31:
        height = width = int(np.clip((height + width + 1) // 2, 11, 29))
    return height, width


def _position(rng, canvas, extent):
    last = canvas - extent - 1
    choice = rng.random()
    if choice < 0.21:
        return 1
    if choice < 0.42:
        return last
    return int(rng.integers(1, last + 1))


def _texture_colors(rng, parameters, regime, index, seen_train):
    common = parameters["train_textures"]
    if regime == "source_train":
        count = 1 if index == 0 else (int(rng.integers(2, 4)) if index == 1
                                          else int(rng.integers(1, 4)))
        first = common[index % len(common)]
        rest = [c for c in common if c != first]
        rng.shuffle(rest)
        return (first, *rest[:count - 1])
    mode = parameters["texture_regime"]
    observed = tuple(sorted(seen_train))
    fresh = parameters["fresh_textures"]
    if mode == 0:
        count = int(rng.integers(1, min(3, len(observed)) + 1))
        return tuple(int(c) for c in rng.choice(observed, size=count, replace=False))
    if mode == 1:
        count = int(rng.integers(1, min(3, len(fresh)) + 1))
        return tuple(int(c) for c in rng.choice(fresh, size=count, replace=False))
    return (int(rng.choice(observed)), int(rng.choice(fresh)))


def _field(rng, shape, neutral, textures):
    height, width = shape
    probability = float(rng.uniform(0.16, 0.64))
    neutral_mask = rng.random(shape) < probability
    style = int(rng.integers(0, 4))
    if style == 1:
        # A few horizontal or vertical runs of background cells.
        for _ in range(int(rng.integers(2, 9))):
            row = int(rng.integers(0, height))
            col = int(rng.integers(0, width))
            length = int(rng.integers(2, 6))
            if rng.random() < 0.5:
                neutral_mask[row, col:min(width, col + length)] = True
            else:
                neutral_mask[row:min(height, row + length), col] = True
    elif style == 2:
        # Local patches can create partial rectangular distractors.
        for _ in range(int(rng.integers(3, 11))):
            row = int(rng.integers(0, height - 1))
            col = int(rng.integers(0, width - 1))
            neutral_mask[row:row + 2, col:col + 2] = bool(rng.integers(0, 2))
    elif style == 3:
        # Weakly correlated texture, without imposing a repeated template.
        near = np.roll(neutral_mask, 1, axis=int(rng.integers(0, 2)))
        use_near = rng.random(shape) < float(rng.uniform(0.10, 0.28))
        neutral_mask[use_near] = near[use_near]
    grid = np.where(neutral_mask, neutral, textures[0]).astype(np.int16)
    for color in textures[1:]:
        candidates = np.argwhere(grid != neutral)
        fraction = float(rng.uniform(0.04, 0.30))
        count = max(3, int(round(len(candidates) * fraction / (len(textures) - 1))))
        chosen = candidates[rng.choice(len(candidates), size=min(count, len(candidates)),
                                       replace=False)]
        grid[chosen[:, 0], chosen[:, 1]] = color
        if rng.random() < 0.58:
            row = int(rng.integers(0, height - 2))
            col = int(rng.integers(0, width - 2))
            patch = grid[row:row + int(rng.integers(2, 4)),
                         col:col + int(rng.integers(2, 4))]
            patch[patch != neutral] = color
    return grid


def _plant_distractors(rng, grid, rectangle, neutral, dominant):
    top, bottom, left, right = rectangle
    target_area = (bottom - top) * (right - left)
    mode = int(rng.integers(0, 4))  # none, smaller rectangle, broken patch, both
    for kind in (("solid",) if mode == 1 else
                 ("broken",) if mode == 2 else
                 ("solid", "broken") if mode == 3 else ()):
        for _ in range(16):
            h = int(rng.integers(3, min(7, grid.shape[0] - 2) + 1))
            w = int(rng.integers(3, min(7, grid.shape[1] - 2) + 1))
            if h * w > target_area * (0.62 if kind == "solid" else 0.95):
                continue
            row = int(rng.integers(1, grid.shape[0] - h))
            col = int(rng.integers(1, grid.shape[1] - w))
            if row < bottom + 1 and row + h > top - 1 and \
                    col < right + 1 and col + w > left - 1:
                continue
            grid[row:row + h, col:col + w] = neutral
            if kind == "broken":
                holes = int(rng.integers(1, min(4, h * w - 1)))
                for _ in range(holes):
                    grid[row + int(rng.integers(0, h)),
                         col + int(rng.integers(0, w))] = dominant
            break


def _generate_pair(rng, regime, parameters, index, seen_train, prior_train):
    neutral = (parameters["neutrals"][index % len(parameters["neutrals"])]
               if regime == "source_train" else int(rng.choice(parameters["neutrals"])))
    textures = _texture_colors(rng, parameters, regime, index, seen_train)
    for _ in range(180):
        height, width = _canvas(rng, parameters, regime)
        minimum = 4 if regime == "source_train" and index == 0 else 3
        rect_h = int(rng.integers(minimum, min(10, height - 2) + 1))
        rect_w = int(rng.integers(minimum, min(10, width - 2) + 1))
        if regime == "source_train" and index == 1 and \
                ((rect_h, rect_w) == prior_train[0][1] or
                 (height, width) == prior_train[0][0]):
            continue
        top = _position(rng, height, rect_h)
        left = _position(rng, width, rect_w)
        rectangle = (top, top + rect_h, left, left + rect_w)
        grid = _field(rng, (height, width), neutral, textures)
        _plant_distractors(rng, grid, rectangle, neutral, textures[0])
        whole, interior = _masks(grid.shape, rectangle)
        grid[whole] = neutral
        ring = _ring_mask(grid.shape, rectangle)
        ring_positions = np.argwhere(ring)
        if not np.any(grid[ring] == neutral):
            row, col = ring_positions[int(rng.integers(0, len(ring_positions)))]
            grid[row, col] = neutral
        output = grid.copy()
        output[interior] = parameters["highlight"]
        pair = {"input": grid, "output": output}
        try:
            analysis = _analyze_pair(pair)
        except AssertionError:
            continue
        if analysis["rectangle"] == rectangle:
            return pair, analysis
    raise ValueError(f"could not construct an unambiguous {regime} pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(int(seed), rng, num_train, num_test)
    parameters = _parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_keys = set()
    seen_train_colors = set()
    prior_train = []
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair, analysis = _generate_pair(pair_rng, regime, parameters, index,
                                                seen_train_colors, prior_train)
                key = _key(pair["input"])
                if key not in seen_keys and _source_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                    break
            else:
                raise ValueError("could not make a unique pair")
            seen_keys.add(key)
            puzzle[split].append(pair)
            if split == "train":
                prior_train.append((pair["input"].shape,
                                    (analysis["height"], analysis["width"])))
                seen_train_colors.update(int(c) for c in np.unique(pair["input"])
                                         if int(c) != analysis["neutral"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "illegal train/test counts")
    seen = set()
    analyses = []
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must have exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                     np.issubdtype(grid.dtype, np.integer),
                     f"{name} must be a two-dimensional integer array")
            _require(all(1 <= d <= 30 for d in grid.shape) and
                     0 <= int(grid.min()) <= int(grid.max()) <= 9,
                     f"{name} has invalid dimensions or colors")
        key = _key(pair["input"])
        _require(key not in seen, "duplicate puzzle input")
        seen.add(key)
        _require(_source_digest(pair) not in _SOURCE_PAIR_DIGESTS,
                 "generated pair exactly copies a source pair")
        analyses.append(_analyze_pair(pair))
    highlights = {a["highlight"] for a in analyses}
    _require(len(highlights) == 1, "highlight convention must be shared")
    train = analyses[:ntrain]
    tests = analyses[ntrain:]
    _require(len({(a["height"], a["width"]) for a in train}) >= 2,
             "training must vary target dimensions")
    _require(len({p["input"].shape for p in puzzle["train"]}) >= 2,
             "training must vary canvas dimensions")
    _require(any(a["height"] >= 4 and a["width"] >= 4 for a in train),
             "training must show a substantial two-dimensional interior")
    _require(len({len(a["texture_counts"]) for a in train}) >= 2,
             "training must vary texture palette complexity")
    _require({a["neutral"] for a in tests}.issubset({a["neutral"] for a in train}),
             "test neutral roles must be demonstrated by training")
    for pair in puzzle["test"]:
        predicted = _solve_from_training(puzzle["train"], pair["input"])
        _require(np.array_equal(predicted, pair["output"]),
                 "test answer is not inferable from demonstrations")


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[s], second[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    a, b = generate_puzzle(123456, 6, 3), generate_puzzle(123456, 6, 3)
    _require(_puzzles_equal(a, b), "generation must be deterministic")
    _require(not _puzzles_equal(a, generate_puzzle(123457, 6, 3)),
             "different seeds should produce different puzzles")
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4), (2, None),
                                      (10, None), (5, 2))):
        puzzle = generate_puzzle(9000 + index, nt, ns)
        validate_generated_puzzle(puzzle)
        _require((nt is None or len(puzzle["train"]) == nt) and
                 (ns is None or len(puzzle["test"]) == ns),
                 "requested counts were not honored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(9900, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count did not raise ValueError")

    metrics = {name: set() for name in ("train_counts", "test_counts", "shapes",
              "rectangles", "positions", "neutral_highlight", "ring",
              "density_bands", "secondary_sizes", "competitors", "split_colors",
              "relative_area", "neutral_modes")}
    combinations = set()
    input_keys = set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        metrics["train_counts"].add(len(puzzle["train"]))
        metrics["test_counts"].add(len(puzzle["test"]))
        train_analyses = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_analyses = [_analyze_pair(pair) for pair in puzzle["test"]]
        neutrals = {x["neutral"] for x in train_analyses}
        metrics["neutral_modes"].add(len(neutrals))
        train_colors = {int(c) for pair, a in zip(puzzle["train"], train_analyses)
                        for c in np.unique(pair["input"]) if int(c) != a["neutral"]}
        test_colors = {int(c) for pair, a in zip(puzzle["test"], test_analyses)
                       for c in np.unique(pair["input"]) if int(c) != a["neutral"]}
        metrics["split_colors"].add((bool(train_colors & test_colors),
                                      bool(test_colors - train_colors)))
        train_area = np.median([p["input"].size for p in puzzle["train"]])
        for pair, analysis in zip(puzzle["test"], test_analyses):
            metrics["relative_area"].add(int(np.sign(pair["input"].size - train_area)))
            _require(np.array_equal(_solve_from_training(puzzle["train"],
                                                         pair["input"]), pair["output"]),
                     "training-only prediction failed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair)
                grid = pair["input"]
                key = _key(grid)
                input_keys.add(key)
                total += 1
                h, w = grid.shape
                top, bottom, left, right = analysis["rectangle"]
                rh, rw = analysis["height"], analysis["width"]
                metrics["shapes"].add((h, w))
                metrics["rectangles"].add((rh, rw))
                metrics["positions"].add((top < (h - rh) / 2,
                                           left < (w - rw) / 2))
                metrics["neutral_highlight"].add((analysis["neutral"],
                                                   analysis["highlight"]))
                metrics["ring"].add(min(analysis["ring_neutral"], 10))
                metrics["density_bands"].add(int(analysis["density"] * 10))
                metrics["secondary_sizes"].update(analysis["texture_counts"])
                metrics["competitors"].add(analysis["competitor_area"])
                combinations.add((h == w, rh == rw, rh % 2, rw % 2,
                                  analysis["ring_neutral"] >= 4,
                                  len(analysis["texture_counts"]) >= 2))
    for name, minimum in (("train_counts", 4), ("test_counts", 3),
                          ("shapes", 40), ("rectangles", 20),
                          ("positions", 4), ("neutral_highlight", 10),
                          ("ring", 5), ("density_bands", 4),
                          ("secondary_sizes", 8), ("competitors", 5),
                          ("split_colors", 3), ("relative_area", 3),
                          ("neutral_modes", 2)):
        _require(len(metrics[name]) >= minimum,
                 f"insufficient {name} diversity")
    _require(len(combinations) >= 12, "compatible variation axes seldom combine")
    _require(len(input_keys) >= int(total * 0.98), "too many repeated grids")
    _require(any(h == w for h, w in metrics["shapes"]) and
             any(h != w for h, w in metrics["shapes"]),
             "need square and nonsquare canvases")
    _require(any(h == 3 or w == 3 for h, w in metrics["rectangles"]) and
             any(h >= w + 3 for h, w in metrics["rectangles"]) and
             any(w >= h + 3 for h, w in metrics["rectangles"]),
             "need small, tall, and wide rectangles")
