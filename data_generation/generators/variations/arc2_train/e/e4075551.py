"""Generate isolated-marker rectangle and interior-cross ARC puzzles."""

import itertools

import numpy as np


# Exact sparse source inputs, used only to rule out literal source copies.
_SOURCE_INPUT_SIGNATURES = {
    ((15, 14), ((2, 6, 4), (5, 10, 7), (8, 3, 8), (8, 7, 2), (12, 6, 6))),
    ((14, 15), ((1, 5, 8), (4, 2, 4), (5, 9, 3), (7, 5, 2), (11, 3, 6))),
    ((15, 15), ((3, 4, 3), (6, 6, 2), (7, 12, 6), (9, 2, 1), (13, 5, 9))),
    ((16, 15), ((1, 7, 1), (7, 4, 2), (7, 11, 3), (9, 2, 7), (13, 6, 4))),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer_argument(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        if high is None:
            raise ValueError(f"{name} must be at least {low}")
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer_argument("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _integer_argument("num_test", num_test, 1, 4)
    legal = [(t, s) for t in range(2, 11) for s in range(1, 5) if t > s]
    if num_train is None and num_test is None:
        return legal[seed % len(legal)]
    choices = [(t, s) for t, s in legal
               if (num_train is None or t == num_train)
               and (num_test is None or s == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[seed % len(choices)]


def _key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _isolated(positions):
    return all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
               for a, b in itertools.combinations(positions, 2))


def _source_signature(grid, background):
    if background != 0:
        return None
    return (grid.shape, tuple((int(r), int(c), int(grid[r, c]))
                              for r, c in np.argwhere(grid != 0)))


def _analyze(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             "input must be a two-dimensional array")
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(int(np.max(counts)) == grid.size - 5,
             "input must have a uniform background and five marker cells")
    positions = [tuple(map(int, p)) for p in np.argwhere(grid != background)]
    _require(len(positions) == 5 and _isolated(positions),
             "input must have five isolated markers")
    roles = {}
    for name, coordinate, extreme in (("top", 0, min), ("bottom", 0, max),
                                      ("left", 1, min), ("right", 1, max)):
        value = extreme(p[coordinate] for p in positions)
        matches = [p for p in positions if p[coordinate] == value]
        _require(len(matches) == 1, f"{name} extreme must be unique")
        roles[name] = matches[0]
    _require(len(set(roles.values())) == 4, "extrema must be different markers")
    junction = next(p for p in positions if p not in roles.values())
    top, bottom = roles["top"][0], roles["bottom"][0]
    left, right = roles["left"][1], roles["right"][1]
    _require(bottom - top + 1 >= 7 and right - left + 1 >= 7,
             "rectangle is too small for separated cross arms")
    _require(left < roles["top"][1] < right and left < roles["bottom"][1] < right,
             "horizontal markers must lie between the side columns")
    _require(top < roles["left"][0] < bottom and top < roles["right"][0] < bottom,
             "side markers must lie between the edge rows")
    _require(top + 2 <= junction[0] <= bottom - 2
             and left + 2 <= junction[1] <= right - 2,
             "junction must leave a connector cell on all four arms")
    colors = {name: int(grid[p]) for name, p in roles.items()}
    junction_color = int(grid[junction])
    _require(junction_color not in colors.values(),
             "junction must be identified by a different color")
    return {"background": background, "positions": roles, "junction": junction,
            "junction_color": junction_color, "colors": colors,
            "bounds": (top, bottom, left, right),
            "size": (bottom - top + 1, right - left + 1),
            "alignment": tuple(name for name in ("top", "bottom", "left", "right")
                               if (roles[name][1] == junction[1] if name in ("top", "bottom")
                                   else roles[name][0] == junction[0]))}


def _paint(input_grid, info, connector):
    top, bottom, left, right = info["bounds"]
    row, col = info["junction"]
    colors = info["colors"]
    output = input_grid.copy()
    output[top, left:right + 1] = colors["top"]
    output[bottom, left:right + 1] = colors["bottom"]
    output[top + 1:bottom, left] = colors["left"]
    output[top + 1:bottom, right] = colors["right"]
    output[row, left + 1:right] = connector
    output[top + 1:bottom, col] = connector
    output[row, col] = info["junction_color"]
    return output


def _mask_for_pair(rng):
    mask = set()
    for name in ("top", "bottom", "left", "right"):
        if name not in mask and rng.random() < 0.27:
            mask.add(name)
    return mask


def _choice_list(rng, values):
    return [int(x) for x in rng.permutation(np.asarray(values))]


def _marker_positions(rng, top, bottom, left, right, junction, mask):
    jr, jc = junction
    def candidates(name):
        if name in ("top", "bottom"):
            values = [c for c in range(left + 1, right) if (c == jc) == (name in mask)]
        else:
            values = [r for r in range(top + 1, bottom) if (r == jr) == (name in mask)]
        return _choice_list(rng, values)
    tc, bc, lr, rr = (candidates(name) for name in ("top", "bottom", "left", "right"))
    for a in tc:
        for b in bc:
            for c in lr:
                for d in rr:
                    positions = {"top": (top, a), "bottom": (bottom, b),
                                 "left": (c, left), "right": (d, right)}
                    if _isolated([*positions.values(), junction]):
                        return positions
    return None


def _sample_pair(rng, connector, split, mask, seen):
    for attempt in range(96):
        # Both splits overlap broadly. Tests have a somewhat longer upper tail.
        max_side = 20 if split == "train" else 23
        rh = int(rng.integers(7, max_side + 1))
        rw = int(rng.integers(7, max_side + 1))
        top_margin = int(rng.integers(0, min(5, 30 - rh) + 1))
        bottom_margin = int(rng.integers(0, min(5, 30 - rh - top_margin) + 1))
        left_margin = int(rng.integers(0, min(5, 30 - rw) + 1))
        right_margin = int(rng.integers(0, min(5, 30 - rw - left_margin) + 1))
        height, width = rh + top_margin + bottom_margin, rw + left_margin + right_margin
        top, left = top_margin, left_margin
        bottom, right = top + rh - 1, left + rw - 1
        jr = int(rng.integers(top + 2, bottom - 1))
        jc = int(rng.integers(left + 2, right - 1))
        junction = (jr, jc)
        positions = _marker_positions(rng, top, bottom, left, right, junction, mask)
        if positions is None:
            continue
        background = int(rng.choice([c for c in range(10) if c != connector]))
        junction_color = int(rng.choice([c for c in range(10)
                                         if c not in (connector, background)]))
        palette = [c for c in range(10) if c not in (connector, background, junction_color)]
        selected = [int(c) for c in rng.choice(palette, size=4, replace=False)]
        if rng.random() < 0.16:
            chosen = int(rng.integers(0, 4))
            selected[chosen] = selected[(chosen + int(rng.integers(1, 4))) % 4]
        grid = np.full((height, width), background, dtype=np.int8)
        for name, color in zip(("top", "bottom", "left", "right"), selected):
            grid[positions[name]] = color
        grid[junction] = junction_color
        key = _key(grid)
        if key in seen or _source_signature(grid, background) in _SOURCE_INPUT_SIGNATURES:
            continue
        info = _analyze(grid)
        pair = {"input": grid, "output": _paint(grid, info, connector)}
        return pair
    raise RuntimeError("could not construct a unique isolated-marker pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer_argument("seed", seed, 0, None)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    connector = int(rng.integers(0, 10))
    child_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    masks = [_mask_for_pair(rng) for _ in range(train_count + test_count)]
    for axis in (("left", "right"), ("top", "bottom")):
        if not any(mask.intersection(axis) for mask in masks[:train_count]):
            index = int(rng.integers(0, train_count))
            masks[index].add(axis[int(rng.integers(0, 2))])
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, offset in (("train", train_count, 0), ("test", test_count, train_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(child_seeds[offset + index]))
            pair = _sample_pair(pair_rng, connector, split,
                                masks[offset + index], seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             f"{label} must be a two-dimensional NumPy array")
    _require(all(1 <= n <= 30 for n in grid.shape), f"{label} size out of range")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
             f"{label} colors out of range")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_infos, test_infos = [], []
    connectors = set()
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain input and output")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, label + " input")
            _validate_grid(out, label + " output")
            _require(inp.shape == out.shape and inp.dtype == out.dtype,
                     f"{label} must preserve shape and dtype")
            ik, ok = _key(inp), _key(out)
            _require(ik not in seen_inputs and (ik, ok) not in seen_pairs,
                     f"{label} duplicates a pair")
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
            info = _analyze(inp)
            _require(_source_signature(inp, info["background"]) not in _SOURCE_INPUT_SIGNATURES,
                     f"{label} copies a source input")
            new_colors = set(map(int, np.unique(out))) - set(map(int, np.unique(inp)))
            _require(len(new_colors) == 1, f"{label} must introduce one connector color")
            connector = next(iter(new_colors))
            _require(not np.any(inp == connector), f"{label} contains connector in input")
            _require(np.array_equal(out, _paint(inp, info, connector)),
                     f"{label} violates rectangle and cross rule")
            _require(not np.array_equal(inp, out), f"{label} has no transformation")
            connectors.add(connector)
            infos.append(info)
    _require(len(connectors) == 1, "all pairs must share one demonstrated connector")
    # Test colors need no codebook: each of the five positions states its own color.
    # Endpoint behavior is demonstrated on both axes in training.
    _require(any("left" in a["alignment"] or "right" in a["alignment"]
                 for a in train_infos), "training lacks a horizontal endpoint example")
    _require(any("top" in a["alignment"] or "bottom" in a["alignment"]
                 for a in train_infos), "training lacks a vertical endpoint example")
    train_features = {(a["size"], a["alignment"], tuple(a["colors"].values()))
                      for a in train_infos}
    _require(all((a["size"], a["alignment"], tuple(a["colors"].values()))
                 not in train_features for a in test_infos),
             "test must use new geometry, alignment, or role colors")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _independent_cell_check(pair, connector):
    inp, out = pair["input"], pair["output"]
    info = _analyze(inp)
    top, bottom, left, right = info["bounds"]
    jr, jc = info["junction"]
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            expected = int(inp[r, c])
            if r == top and left <= c <= right:
                expected = info["colors"]["top"]
            elif r == bottom and left <= c <= right:
                expected = info["colors"]["bottom"]
            elif top < r < bottom and c == left:
                expected = info["colors"]["left"]
            elif top < r < bottom and c == right:
                expected = info["colors"]["right"]
            elif top < r < bottom and left < c < right and (r == jr or c == jc):
                expected = connector if (r, c) != (jr, jc) else info["junction_color"]
            _require(int(out[r, c]) == expected, "independent cell rule mismatch")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds repeat puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    sizes = {"train": set(), "test": set()}
    aligns = {"train": set(), "test": set()}
    aspects = {"train": set(), "test": set()}
    backgrounds, junctions, connectors = set(), set(), set()
    layouts = set()
    edge = centered = repeated = outside_first = repeated_width = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first_colors = set(_analyze(puzzle["train"][0]["input"])["colors"].values())
        train_widths = [_analyze(p["input"])["size"][1] for p in puzzle["train"]]
        repeated_width += len(train_widths) != len(set(train_widths))
        connector = next(iter(set(map(int, np.unique(puzzle["train"][0]["output"])))
                              - set(map(int, np.unique(puzzle["train"][0]["input"])))))
        connectors.add(connector)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                _independent_cell_check(pair, connector)
                h, w = info["size"]
                sizes[split].add((h, w))
                aligns[split].add(info["alignment"])
                aspects[split].add("square" if h == w else "wide" if w > h else "tall")
                backgrounds.add(info["background"])
                junctions.add(info["junction_color"])
                repeated += len(set(info["colors"].values())) < 4
                t, b, l, r = info["bounds"]
                gh, gw = pair["input"].shape
                edge += t == 0 or l == 0 or b == gh - 1 or r == gw - 1
                centered += 2 * info["junction"][0] == t + b or 2 * info["junction"][1] == l + r
                layouts.add((split, pair["input"].shape, info["size"], info["junction"],
                             tuple(info["positions"].values())))
                if split == "test":
                    outside_first += bool(set(info["colors"].values()) - first_colors)
    _require(len(connectors) >= 7 and len(backgrounds) >= 8 and len(junctions) >= 8,
             "color roles lack diversity")
    _require(all(aspects[s] == {"square", "wide", "tall"} for s in aspects),
             "both splits need all aspect ratios")
    _require(all(len(sizes[s]) >= 40 for s in sizes), "rectangle sizes lack diversity")
    _require(all(len(aligns[s]) >= 8 for s in aligns), "alignment combinations lack diversity")
    _require(len(layouts) >= 175, "layouts lack diversity")
    _require(edge >= 20 and centered >= 10 and repeated >= 10 and outside_first >= 10
             and repeated_width >= 4, "review dimensions or colors lack coverage")
    for seed, t, s in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 3)):
        p = generate_puzzle(seed, t, s)
        _require((len(p["train"]), len(p["test"])) == (t, s), "explicit counts ignored")
    for seed, t, s in ((200, None, 1), (201, None, 4), (202, 2, None), (203, 10, None)):
        p = generate_puzzle(seed, t, s)
        _require((t is None or len(p["train"]) == t)
                 and (s is None or len(p["test"]) == s), "partial count ignored")
    _require(_puzzle_key(generate_puzzle(987654, 7, 3))
             == _puzzle_key(generate_puzzle(987654, 7, 3)), "not deterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.5},
                   {"num_test": True}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
