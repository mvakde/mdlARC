import numpy as np


_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_KINDS = ("line", "elbow", "tee", "cross", "block", "winding", "hollow", "bridge")
_SOURCE_INPUTS = frozenset((
    "1110011101011" "1101111100101" "1011111111011" "1100101101111"
    "1011111111101" "1110111010111" "1001110110011" "1001101111110"
    "1010111110111" "0011111011111" "0111011101111" "1111101010101"
    "1011011111001",
    "1101111011111" "1011111111111" "1111101111111" "1111000110110"
    "1100110110111" "0111111111111" "1111110111111" "0110111101111"
    "1111111111111" "1111111110101" "1011110011111" "0111111011111"
    "1111111110011",
    "1111111101101" "1110111101110" "1111111111111" "1011011101111"
    "1111111111111" "0111111110011" "0100011101101" "1100111111110"
    "1110111111010" "0111111001111" "1111111110101" "1111110111101"
    "0110111111011",
    "1110011111001" "0111011111100" "1100011011010" "0110111110101"
    "1111101111111" "1010110111010" "1001010111110" "1010111100010"
    "1111110111010" "1010011110011" "0110111101011" "1111011111010"
    "1111110101111",
))


def _ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test, rng):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = seed % 4 + 1
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in _CARDINAL:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append(component)
    return result


def _shape(kind, rng):
    if kind == "line":
        points = [(0, c) for c in range(int(rng.integers(2, 9)))]
    elif kind == "elbow":
        a, b = (int(rng.integers(2, 6)), int(rng.integers(2, 6)))
        points = [(r, 0) for r in range(a)] + [(a - 1, c) for c in range(1, b)]
    elif kind == "tee":
        stem = int(rng.integers(3, 7))
        arm = int(rng.integers(1, 4))
        mid = int(rng.integers(1, stem - 1))
        points = [(r, arm) for r in range(stem)] + [(mid, c) for c in range(2 * arm + 1)]
    elif kind == "cross":
        arm = int(rng.integers(1, 4))
        points = [(r, arm) for r in range(2 * arm + 1)] + [(arm, c) for c in range(2 * arm + 1)]
    elif kind == "block":
        h, w = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        points = [(r, c) for r in range(h) for c in range(w)]
    elif kind == "winding":
        w = int(rng.integers(3, 7))
        points = ([(0, c) for c in range(w)] + [(1, w - 1)] +
                  [(2, c) for c in range(w)] + [(3, 0)] +
                  [(4, c) for c in range(w)])
    elif kind == "hollow":
        h, w = (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
        points = [(r, c) for r in range(h) for c in range(w)
                  if r in (0, h - 1) or c in (0, w - 1)]
    elif kind == "bridge":
        gap = int(rng.integers(2, 6))
        points = ([(r, c) for r in range(2) for c in (0, 1, gap + 2, gap + 3)] +
                  [(0, c) for c in range(2, gap + 2)])
    else:
        raise ValueError("unknown shape")
    if rng.integers(2):
        points = [(r, -c) for r, c in points]
    for _ in range(int(rng.integers(4))):
        points = [(c, -r) for r, c in points]
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted({(r - min_r, c - min_c) for r, c in points}))


def _can_place(mask, points):
    height, width = mask.shape
    cells = set(points)
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width) or mask[r, c]:
            return False
        for dr, dc in _CARDINAL:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and (rr, cc) not in cells:
                return False
    return True


def _place(mask, shape, rng, diagonal=False):
    height, width = mask.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    if shape_h > height or shape_w > width:
        return False
    existing = np.argwhere(mask)
    for attempt in range(110):
        if diagonal and len(existing) and attempt < 55:
            old_r, old_c = existing[int(rng.integers(len(existing)))]
            dr, dc = _DIAGONAL[int(rng.integers(4))]
            sr, sc = shape[int(rng.integers(len(shape)))]
            top, left = int(old_r + dr - sr), int(old_c + dc - sc)
        else:
            top = int(rng.integers(height - shape_h + 1))
            left = int(rng.integers(width - shape_w + 1))
        points = [(top + r, left + c) for r, c in shape]
        if _can_place(mask, points):
            for r, c in points:
                mask[r, c] = True
            return True
    return False


def _grow(mask, target, rng, protect_small=False):
    components = _components(mask)
    labels = np.full(mask.shape, -1, dtype=np.int16)
    for index, component in enumerate(components):
        for r, c in component:
            labels[r, c] = index
    eligible = [i for i, comp in enumerate(components) if len(comp) >= (3 if protect_small else 2)]
    if not eligible:
        return
    height, width = mask.shape
    count = int(mask.sum())
    for _ in range(mask.size * 30):
        if count >= target:
            return
        index = eligible[int(rng.integers(len(eligible)))]
        r, c = components[index][int(rng.integers(len(components[index])))]
        dr, dc = _CARDINAL[int(rng.integers(4))]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < height and 0 <= cc < width) or mask[rr, cc]:
            continue
        neighbors = []
        for adr, adc in _CARDINAL:
            nr, nc = rr + adr, cc + adc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
                neighbors.append(int(labels[nr, nc]))
        if neighbors and set(neighbors) == {index}:
            mask[rr, cc] = True
            labels[rr, cc] = index
            components[index].append((rr, cc))
            count += 1


def _density_target(area, category, rng):
    bounds = ((0.045, 0.125), (0.13, 0.235), (0.24, 0.39))[category]
    return min(int(area * 0.42), max(2, int(round(area * float(rng.uniform(*bounds))))))


def _make_mask(shape, mode, category, featured, rng):
    mask = np.zeros(shape, dtype=bool)
    area = mask.size
    target = _density_target(area, category, rng)
    if mode == "evidence":
        # A small rotated witness contains both dimer directions, a branch,
        # and a singleton touching a dimer diagonally.
        anchor = ((0, 0), (0, 1), (1, 2), (3, 0), (4, 0),
                  (3, 4), (4, 3), (4, 4), (4, 5), (5, 4))
        _ensure(_place(mask, _transform_anchor(anchor, rng), rng), "evidence anchor did not fit")
        for _ in range(int(rng.integers(1, 5))):
            extra = _shape(_KINDS[int(rng.integers(len(_KINDS)))], rng)
            if int(mask.sum()) + len(extra) <= int(area * 0.40):
                _place(mask, extra, rng)
        for _ in range(int(rng.integers(1, 6))):
            if int(mask.sum()) < int(area * 0.42):
                _place(mask, ((0, 0),), rng, diagonal=bool(rng.integers(2)))
        return mask
    if mode == "simple":
        for index in range(int(rng.integers(1, 4))):
            _place(mask, _shape("line", rng)[:2], rng, diagonal=index > 0)
        for _ in range(int(rng.integers(1, 5))):
            _place(mask, ((0, 0),), rng, diagonal=bool(rng.integers(2)))
        return mask
    if mode == "isolated":
        for _ in range(target):
            if not _place(mask, ((0, 0),), rng, diagonal=bool(rng.integers(2))):
                break
        return mask
    # Rich and grouped modes deliberately seed a named geometry, then compose
    # additional shapes. Growth attaches only to one existing component.
    _place(mask, _shape(featured, rng), rng)
    target = max(target, int(mask.sum()) + 2)
    single_target = 0 if mode == "grouped" else max(1, int(target * float(rng.uniform(0.08, 0.28))))
    attempts = 0
    while int(mask.sum()) < target - single_target and attempts < 32:
        kind = _KINDS[int(rng.integers(len(_KINDS)))]
        shape_points = _shape(kind, rng)
        if int(mask.sum()) + len(shape_points) <= int(area * 0.42) - single_target:
            _place(mask, shape_points, rng, diagonal=bool(rng.integers(3) == 0))
        attempts += 1
    _grow(mask, min(target - single_target, int(area * 0.43)), rng)
    for _ in range(single_target):
        if int(mask.sum()) < int(area * 0.42):
            _place(mask, ((0, 0),), rng, diagonal=bool(rng.integers(2)))
    if not mask.any():
        _place(mask, ((0, 0), (0, 1)), rng)
    return mask


def _transform_anchor(points, rng):
    if rng.integers(2):
        points = [(r, -c) for r, c in points]
    for _ in range(int(rng.integers(4))):
        points = [(c, -r) for r, c in points]
    low_r = min(r for r, _ in points)
    low_c = min(c for _, c in points)
    return tuple((r - low_r, c - low_c) for r, c in points)


def _solve(input_grid, marker, highlight):
    mask = input_grid == marker
    adjacent = np.zeros(mask.shape, dtype=bool)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    output = input_grid.copy()
    output[mask & adjacent] = highlight
    return output


def _grid_key(grid):
    return (grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes())


def _source_copy(grid):
    return (grid.shape == (13, 13) and
            set(int(v) for v in np.unique(grid)) == {0, 1} and
            "".join(str(int(v)) for v in grid.flat) in _SOURCE_INPUTS)


def _pair(rng, shape, field, marker, highlight, mode, category, featured):
    mask = _make_mask(shape, mode, category, featured, rng)
    grid = np.full(shape, field, dtype=np.int8)
    grid[mask] = marker
    return {"input": grid, "output": _solve(grid, marker, highlight)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _checked_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test, rng)
    source_style = seed % 7 == 0
    highlight = 8 if source_style else int(rng.integers(10))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)
    puzzle = {"train": [], "test": []}
    used = set()
    shapes = set()
    palettes = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            offset = index if split == "train" else train_count + index
            pair_rng = np.random.default_rng(int(pair_seeds[offset]))
            if split == "train" and index == 0:
                mode = "evidence"
            elif split == "train":
                mode = ("simple", "isolated", "grouped", "rich", "crowded")[(seed + index - 1) % 5]
            else:
                mode = ("rich", "crowded", "rich", "grouped")[(seed + index) % 4]
            if source_style and split == "train" and index == 0:
                shape = (13, 13)
                field, marker = 1, 0
            else:
                low = 12 if mode == "evidence" else 8
                shape = (int(pair_rng.integers(low, 31)), int(pair_rng.integers(low, 31)))
                while shape in shapes:
                    shape = (shape[0], 8 + (shape[1] - 7) % 23)
                available = [color for color in range(10) if color != highlight]
                options = [(a, b) for a in available for b in available
                           if a < b and (a, b) not in palettes]
                a, b = options[int(pair_rng.integers(len(options)))]
                field, marker = (a, b) if pair_rng.integers(2) else (b, a)
            shapes.add(shape)
            palettes.add(tuple(sorted((field, marker))))
            category = int(pair_rng.choice(3, p=(0.28, 0.35, 0.37) if split == "train" else (0.25, 0.30, 0.45)))
            featured = _KINDS[(seed + offset * 3) % len(_KINDS)]
            for _ in range(5):
                pair = _pair(pair_rng, shape, field, marker, highlight, mode, category, featured)
                key = _grid_key(pair["input"])
                if key not in used and not _source_copy(pair["input"]):
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            used.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(mask):
    components = _components(mask)
    sizes = []
    horizontal_dimer = vertical_dimer = diagonal_singleton = False
    branch = False
    for component in components:
        sizes.append(len(component))
        cells = set(component)
        if len(component) == 2:
            horizontal_dimer |= component[0][0] == component[1][0]
            vertical_dimer |= component[0][1] == component[1][1]
        for r, c in component:
            degree = sum((r + dr, c + dc) in cells for dr, dc in _CARDINAL)
            branch |= degree >= 3
        if len(component) == 1:
            r, c = component[0]
            diagonal_singleton |= any(
                0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1] and mask[r + dr, c + dc]
                for dr, dc in _DIAGONAL
            )
    block = bool(np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))
    long_run = False
    for row in mask:
        long_run |= "1111" in "".join("1" if v else "0" for v in row)
    for col in mask.T:
        long_run |= "1111" in "".join("1" if v else "0" for v in col)
    return {
        "sizes": sizes, "horizontal_dimer": horizontal_dimer, "vertical_dimer": vertical_dimer,
        "diagonal_singleton": diagonal_singleton, "branch": branch, "block": block,
        "long_run": long_run, "density": float(mask.mean()),
        "singleton_count": sizes.count(1), "multi_count": sum(size >= 2 for size in sizes),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _ensure(all(isinstance(puzzle[k], list) for k in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    observed_highlights = set()
    training = []
    test = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _ensure(all(1 <= d <= 30 for d in grid.shape), "grid dimensions")
                _ensure(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _ensure(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            _ensure(inp.shape == out.shape and inp.dtype == out.dtype, "shape and dtype preserved")
            _ensure(not _source_copy(inp), "source input copied")
            ik, ok = _grid_key(inp), _grid_key(out)
            _ensure(ik not in seen_inputs and (ik, ok) not in seen_pairs, "duplicate pair")
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
            colors, counts = np.unique(inp, return_counts=True)
            _ensure(len(colors) == 2 and counts[0] != counts[1], "two input roles")
            field = int(colors[int(np.argmax(counts))])
            marker = int(colors[int(np.argmin(counts))])
            mask = inp == marker
            _ensure(float(mask.mean()) <= 0.43, "field must remain clearly dominant")
            extras = set(int(v) for v in np.unique(out)) - {field, marker}
            _ensure(len(extras) <= 1, "at most one output-only color")
            if extras:
                observed_highlights.update(extras)
            metrics = _analyze(mask)
            (training if split == "train" else test).append((inp, out, marker, metrics))
    _ensure(len(observed_highlights) == 1, "one learned highlight color")
    highlight = next(iter(observed_highlights))
    for inp, out, marker, metrics in training + test:
        _ensure(highlight not in inp, "highlight appears in an input")
        mask = inp == marker
        expected = _solve(inp, marker, highlight)
        _ensure(np.array_equal(out, expected), "four-neighbor recoloring error")
        _ensure((metrics["multi_count"] > 0) == (highlight in out), "output highlight mismatch")
    _ensure(any(m["singleton_count"] for _, _, _, m in training), "training lacks unchanged singletons")
    _ensure(any(m["horizontal_dimer"] for _, _, _, m in training), "training lacks horizontal dimers")
    _ensure(any(m["vertical_dimer"] for _, _, _, m in training), "training lacks vertical dimers")
    _ensure(any(m["diagonal_singleton"] for _, _, _, m in training), "training lacks diagonal-only contact")
    _ensure(any(any(size >= 3 for size in m["sizes"]) for _, _, _, m in training),
            "training lacks larger components")
    _ensure(all(m["multi_count"] for _, _, _, m in test), "each test must exercise recoloring")


def _puzzles_equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[k], b[k]) for role in ("input", "output")
    ) for k in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    all_shapes, palettes = set(), set()
    densities = {"train": set(), "test": set()}
    identity = grouped = simple = branch = block = run = 0
    within_shapes = within_palettes = sparse_tests = crowded_tests = 0
    source_style = 0
    first_puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 2:
            first_puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        shapes, colors = set(), set()
        train_density = []
        test_density = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                palette, counts = np.unique(inp, return_counts=True)
                marker = int(palette[int(np.argmin(counts))])
                metrics = _analyze(inp == marker)
                # Independent component walk, separate from the adjacency solver.
                changed = {(r, c) for comp in _components(inp == marker) if len(comp) >= 2 for r, c in comp}
                actual = {(r, c) for r, c in zip(*np.nonzero(inp != out))}
                assert changed == actual
                assert all(out[r, c] == inp[r, c] for r, c in zip(*np.nonzero(inp == int(palette[int(np.argmax(counts))]))))
                shapes.add(inp.shape)
                colors.add(tuple(int(v) for v in palette))
                all_shapes.add(inp.shape)
                palettes.add(tuple(int(v) for v in palette))
                densities[split].add(round(metrics["density"], 2))
                (train_density if split == "train" else test_density).append(metrics["density"])
                identity += split == "train" and metrics["multi_count"] == 0
                grouped += split == "train" and metrics["singleton_count"] == 0
                simple += split == "train" and max(metrics["sizes"]) <= 2
                branch += split == "train" and metrics["branch"]
                block += split == "train" and metrics["block"]
                run += split == "train" and metrics["long_run"]
        within_shapes += len(shapes) >= 2
        within_palettes += len(colors) >= 2
        sparse_tests += min(test_density) < max(train_density)
        crowded_tests += max(test_density) > max(train_density)
        source_style += (puzzle["train"][0]["input"].shape == (13, 13) and
                         set(np.unique(puzzle["train"][0]["input"])) == {0, 1})
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(all_shapes) >= 100 and min(min(s) for s in all_shapes) <= 10
    assert max(max(s) for s in all_shapes) >= 28
    assert len(palettes) >= 30 and within_shapes >= 35 and within_palettes >= 35
    assert len(densities["train"]) >= 20 and len(densities["test"]) >= 15
    assert identity >= 3 and grouped >= 3 and simple >= 5
    assert branch >= 25 and block >= 8 and run >= 20
    assert sparse_tests >= 5 and crowded_tests >= 5 and source_style >= 4
    assert _puzzles_equal(first_puzzles[0], generate_puzzle(0))
    assert not _puzzles_equal(first_puzzles[0], first_puzzles[1])
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(100 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for index, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, *counts)
        assert counts[0] is None or len(puzzle["train"]) == counts[0]
        assert counts[1] is None or len(puzzle["test"]) == counts[1]
    explicit = generate_puzzle(400, 6, 3)
    assert _puzzles_equal(explicit, generate_puzzle(400, 6, 3))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(500, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {counts}")
