"""Four-neighbor component recoloring puzzles."""

import numpy as np


_SOURCE_PAIR_ROWS = {
    (("3303", "3300", "3003", "0033"), ("8803", "8800", "8008", "0088")),
    (("030003", "033300", "000030", "030000"), ("080003", "088800", "000030", "030000")),
    (("330000", "030030", "300000", "033000", "033003"), ("880000", "080030", "300000", "088000", "088003")),
    (("330", "030", "303"), ("880", "080", "303")),
    (("30303", "33300", "00003", "03300", "03300"), ("80803", "88800", "00003", "08800", "08800")),
}
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _is_source_pair(inp, out):
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(inp), rows(out)) in _SOURCE_PAIR_ROWS


def _components(mask):
    seen = set()
    found = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or (r, c) in seen:
                continue
            group = []
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                point = stack.pop()
                group.append(point)
                for dr, dc in _NEIGHBORS:
                    other = (point[0] + dr, point[1] + dc)
                    if (0 <= other[0] < height and 0 <= other[1] < width
                            and mask[other] and other not in seen):
                        seen.add(other)
                        stack.append(other)
            found.append(group)
    return found


def _normalize(cells):
    low_r = min(r for r, _ in cells)
    low_c = min(c for _, c in cells)
    return tuple(sorted((r - low_r, c - low_c) for r, c in cells))


def _orient(cells, rng):
    cells = tuple(cells)
    if rng.random() < 0.5:
        cells = tuple((c, r) for r, c in cells)
    if rng.random() < 0.5:
        cells = tuple((-r, c) for r, c in cells)
    if rng.random() < 0.5:
        cells = tuple((r, -c) for r, c in cells)
    return _normalize(cells)


def _growth(rng, target, compact):
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in _NEIGHBORS:
                point = (r + dr, c + dc)
                if point not in cells:
                    frontier.add(point)
        choices = sorted(frontier)
        weights = []
        for r, c in choices:
            touching = sum((r + dr, c + dc) in cells for dr, dc in _NEIGHBORS)
            if compact:
                weights.append(float(touching ** 3))
            else:
                weights.append(float(1 if touching == 1 else 0.12))
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        cells.add(choices[int(rng.choice(len(choices), p=weights))])
    return _normalize(cells)


def _path(rng, target):
    for _ in range(25):
        cells = [(0, 0)]
        used = {(0, 0)}
        while len(cells) < target:
            r, c = cells[-1]
            choices = []
            for dr, dc in _NEIGHBORS:
                point = (r + dr, c + dc)
                if point in used:
                    continue
                # Prevent the path from closing into a compact region.
                if sum((point[0] + x, point[1] + y) in used for x, y in _NEIGHBORS) != 1:
                    continue
                choices.append(point)
            if not choices:
                break
            point = choices[int(rng.integers(len(choices)))]
            cells.append(point)
            used.add(point)
        if len(cells) == target:
            return _normalize(cells)
    return _growth(rng, target, False)


def _shape(rng, target):
    families = ["line", "branch", "path", "blob", "growth"]
    if target >= 4:
        families.append("rectangle")
    if target >= 8:
        families.extend(("ring", "bridge"))
    family = families[int(rng.integers(len(families)))]
    if family == "line":
        cells = [(0, c) for c in range(target)]
    elif family == "branch":
        trunk = max(3, (target + 1) // 2)
        cells = [(0, c) for c in range(trunk)]
        anchor = int(rng.integers(1, trunk - 1))
        cells += [(r, anchor) for r in range(1, target - trunk + 1)]
    elif family == "path":
        cells = _path(rng, target)
    elif family in ("blob", "growth"):
        cells = _growth(rng, target, family == "blob")
    elif family == "rectangle":
        options = [(a, b) for a in range(2, 6) for b in range(2, 7)
                   if abs(a * b - target) <= 3]
        if not options:
            options = [(2, 2)]
        a, b = options[int(rng.integers(len(options)))]
        cells = [(r, c) for r in range(a) for c in range(b)]
    elif family == "ring":
        options = [(a, b) for a in range(3, 7) for b in range(3, 7)
                   if abs((2 * a + 2 * b - 4) - target) <= 4]
        if not options:
            options = [(3, 3)]
        a, b = options[int(rng.integers(len(options)))]
        cells = [(r, c) for r in range(a) for c in range(b)
                 if r in (0, a - 1) or c in (0, b - 1)]
    else:
        # Two lobes connected by a one-cell-wide orthogonal bridge.
        length = max(1, min(5, target - 8))
        right = length + 2
        cells = [(r, c) for r in (0, 1) for c in (0, 1, right, right + 1)]
        cells += [(1, c) for c in range(2, right)]
    return _orient(cells, rng)


def _relation(cells, existing):
    diagonal = any(abs(r - rr) == 1 and abs(c - cc) == 1
                   for r, c in cells for rr, cc in existing)
    gap = any((r == rr and abs(c - cc) == 2) or
              (c == cc and abs(r - rr) == 2)
              for r, c in cells for rr, cc in existing)
    far = all(abs(r - rr) + abs(c - cc) >= 4
              for r, c in cells for rr, cc in existing)
    return diagonal, gap, far


def _place(rng, shape, height, width, occupied, related, preference):
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    if sh > height or sw > width:
        return None
    choices = []
    preferred = []
    for top in range(height - sh + 1):
        for left in range(width - sw + 1):
            cells = {(top + r, left + c) for r, c in shape}
            if cells & occupied:
                continue
            if any((r + dr, c + dc) in occupied
                   for r, c in cells for dr, dc in _NEIGHBORS):
                continue
            choices.append(cells)
            if related:
                diag, gap, far = _relation(cells, related)
                if ((preference == "diagonal" and diag) or
                    (preference == "gap" and gap) or
                    (preference == "far" and far)):
                    preferred.append(cells)
    if not choices:
        return None
    pool = preferred if preferred else choices
    return pool[int(rng.integers(len(pool)))]


def _dimensions(rng, regime, mode):
    if mode == "dense":
        options = ("medium", "large")
    elif mode == "compact":
        options = ("small",)
    else:
        options = ("small", "small", "medium", "medium", "large", "narrow")
        if regime == "target_test":
            options = ("small", "medium", "medium", "large", "large", "narrow")
    choice = options[int(rng.integers(len(options)))]
    if choice == "small":
        height, width = int(rng.integers(3, 8)), int(rng.integers(3, 9))
    elif choice == "medium":
        height, width = int(rng.integers(6, 13)), int(rng.integers(6, 14))
    elif choice == "large":
        height, width = int(rng.integers(10, 18)), int(rng.integers(10, 19))
    else:
        height, width = int(rng.integers(3, 6)), int(rng.integers(10, 21))
        if rng.random() < 0.5:
            height, width = width, height
    return height, width


def _composition(rng, regime, first_train):
    if first_train:
        return "mixed", int(rng.integers(1, 4)), int(rng.integers(1, 5))
    modes = ("mixed", "mixed", "mixed", "compact", "one_large", "several", "dense",
             "all_singleton", "all_group")
    mode = modes[int(rng.integers(len(modes)))]
    if regime == "source_train" and mode == "all_singleton":
        mode = "mixed"
    if mode == "mixed":
        return mode, int(rng.integers(1, 4)), int(rng.integers(1, 5))
    if mode == "compact":
        return mode, 1, int(rng.integers(1, 4))
    if mode == "one_large":
        return mode, 1, int(rng.integers(4, 9))
    if mode == "several":
        return mode, int(rng.integers(3, 6)), int(rng.integers(0, 3))
    if mode == "dense":
        return mode, int(rng.integers(4, 8)), int(rng.integers(0, 5))
    if mode == "all_singleton":
        return mode, 0, int(rng.integers(3, 9))
    return mode, int(rng.integers(1, 6)), 0


def _make_mask(rng, regime, first_train, preferred_shape=None):
    mode, n_groups, n_singletons = _composition(rng, regime, first_train)
    for attempt in range(110):
        height, width = (preferred_shape if preferred_shape is not None and attempt < 45
                         else _dimensions(rng, regime, mode))
        occupied = set()
        groups = set()
        singles = set()
        ok = True
        for index in range(n_groups):
            placed = None
            for _ in range(28):
                if mode == "one_large":
                    target = int(rng.integers(9, 20))
                elif mode == "dense":
                    target = int(rng.integers(2, 8))
                elif mode == "compact":
                    target = int(rng.integers(2, 6))
                else:
                    upper = 17 if regime == "target_test" else 14
                    upper = min(upper, max(4, height * width // (n_groups + 1)))
                    target = int(rng.integers(2, upper + 1))
                shape = _shape(rng, target)
                preference = ("diagonal", "gap", "far", "any")[
                    int(rng.integers(4))]
                placed = _place(rng, shape, height, width, occupied, groups, preference)
                if placed is not None:
                    break
            if placed is None:
                ok = False
                break
            occupied.update(placed)
            groups.update(placed)
        if not ok:
            continue
        for index in range(n_singletons):
            preferences = ("diagonal", "diagonal", "gap", "far", "any")
            preference = preferences[int(rng.integers(len(preferences)))]
            related = singles if (singles and rng.random() < 0.3) else groups
            placed = _place(rng, ((0, 0),), height, width,
                            occupied, related, preference)
            if placed is None:
                ok = False
                break
            occupied.update(placed)
            singles.update(placed)
        if not ok or len(occupied) * 2 >= height * width:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for point in occupied:
            mask[point] = True
        components = _components(mask)
        if sum(len(x) >= 2 for x in components) != n_groups:
            continue
        if sum(len(x) == 1 for x in components) != n_singletons:
            continue
        if first_train and not _singleton_group_diagonal(components):
            continue
        return mask
    # A deterministic fallback for very unlucky layouts, translated by the
    # pair RNG.
    height, width = 7, 9
    mask = np.zeros((height, width), dtype=bool)
    cells = [(1, 1), (1, 2), (2, 2), (3, 3), (5, 6)]
    shift = int(rng.integers(3))
    for r, c in cells:
        mask[r, c + shift] = True
    return mask


def _singleton_group_diagonal(components):
    singles = [group[0] for group in components if len(group) == 1]
    groups = [point for group in components if len(group) >= 2 for point in group]
    return any(abs(r - rr) == 1 and abs(c - cc) == 1
               for r, c in singles for rr, cc in groups)


def _transform_grid(inp, foreground, highlight):
    out = inp.copy()
    for group in _components(inp == foreground):
        if len(group) >= 2:
            for point in group:
                out[point] = highlight
    return out


def _cellwise_expected(inp, foreground, highlight):
    """Independent local check: a cell changes iff it has a 4-neighbor."""
    height, width = inp.shape
    expected = inp.copy()
    for r in range(height):
        for c in range(width):
            if inp[r, c] != foreground:
                continue
            if any(0 <= r + dr < height and 0 <= c + dc < width and
                   inp[r + dr, c + dc] == foreground
                   for dr, dc in _NEIGHBORS):
                expected[r, c] = highlight
    return expected


def _validate_optional_count(name, value, minimum, maximum):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _resolve_counts(rng, seed, num_train, num_test):
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        minimum_train = test_count + 1
        train_count = minimum_train + (seed // 4) % (11 - minimum_train)
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    colors = rng.choice(10, size=3, replace=False)
    base_bg, base_fg, highlight = map(int, colors)
    color_modes = ("stable", "stable", "vary_bg", "vary_fg", "vary_both")
    color_mode = color_modes[int(rng.integers(len(color_modes)))]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for regime, count, destination in (("source_train", train_count, "train"),
                                       ("target_test", test_count, "test")):
        for index in range(count):
            accepted = None
            preferred_shape = None
            if regime == "target_test" and index == 0 and rng.random() < 0.45:
                preferred_shape = puzzle["train"][int(rng.integers(train_count))]["input"].shape
            for _ in range(60):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                mask = _make_mask(child, regime, regime == "source_train" and index == 0,
                                  preferred_shape)
                bg, fg = base_bg, base_fg
                if color_mode in ("vary_bg", "vary_both"):
                    bg = int(child.choice([x for x in range(10) if x not in (fg, highlight)]))
                if color_mode in ("vary_fg", "vary_both"):
                    fg = int(child.choice([x for x in range(10) if x not in (bg, highlight)]))
                if index == 1 and regime == "source_train" and color_mode != "stable":
                    previous = puzzle["train"][0]["input"]
                    previous_colors, counts = np.unique(previous, return_counts=True)
                    old_bg = int(previous_colors[np.argmax(counts)])
                    old_fg = int(previous_colors[np.argmin(counts)])
                    if color_mode in ("vary_bg", "vary_both") and bg == old_bg:
                        bg = int(child.choice([x for x in range(10) if x not in (fg, highlight, old_bg)]))
                    if color_mode in ("vary_fg", "vary_both") and fg == old_fg:
                        fg = int(child.choice([x for x in range(10) if x not in (bg, highlight, old_fg)]))
                inp = np.full(mask.shape, bg, dtype=np.int8)
                inp[mask] = fg
                out = _transform_grid(inp, fg, highlight)
                if _grid_key(inp) in seen_inputs or _is_source_pair(inp, out):
                    continue
                accepted = {"input": inp, "output": out}
                seen_inputs.add(_grid_key(inp))
                break
            if accepted is None:
                raise RuntimeError("Unable to generate a unique puzzle pair")
            puzzle[destination].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair, highlight):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    bg = int(colors[np.argmax(counts)])
    fg = int(colors[np.argmin(counts)])
    assert int(np.max(counts)) > int(np.min(counts))
    assert highlight not in (bg, fg)
    assert np.array_equal(out, _cellwise_expected(inp, fg, highlight))
    groups = _components(inp == fg)
    assert groups
    return {"bg": bg, "fg": fg, "groups": groups}


def _validate_train_regime(pair, info):
    height, width = pair["input"].shape
    assert 3 <= height <= 20 and 3 <= width <= 20
    assert len(info["groups"]) >= 1


def _validate_test_regime(pair, info):
    height, width = pair["input"].shape
    assert 3 <= height <= 20 and 3 <= width <= 20
    assert len(info["groups"]) >= 1


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs = set()
    changed_train = set()
    for pair in puzzle["train"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        changed = pair["input"] != pair["output"]
        if np.any(changed):
            changed_train.update(int(x) for x in np.unique(pair["output"][changed]))
    assert len(changed_train) == 1
    highlight = next(iter(changed_train))
    train_info = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert not _is_source_pair(pair["input"], pair["output"])
            info = _pair_info(pair, highlight)
            if regime == "train":
                _validate_train_regime(pair, info)
                train_info.append(info)
            else:
                _validate_test_regime(pair, info)
    # The first demonstration visibly establishes both outcomes and diagonal
    # separation. Every later test uses the highlight learned from training.
    first = train_info[0]["groups"]
    assert any(len(g) >= 2 for g in first) and any(len(g) == 1 for g in first)
    assert _singleton_group_diagonal(first)


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[split], b[split]) for role in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    def multi_sizes(pair):
        values, counts = np.unique(pair["input"], return_counts=True)
        foreground = int(values[np.argmin(counts)])
        return {len(group) for group in _components(pair["input"] == foreground)
                if len(group) >= 2}

    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    assert _puzzles_equal(generate_puzzle(271828, 7, 3), generate_puzzle(271828, 7, 3))
    assert not _puzzles_equal(generate_puzzle(100, 5, 2), generate_puzzle(101, 5, 2))
    for seed, a, b in ((5000, 2, 1), (5001, 10, 1), (5002, 10, 4),
                       (6000, None, 1), (6001, None, 4),
                       (6002, 2, None), (6003, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        assert (a is None or len(puzzle["train"]) == a)
        assert (b is None or len(puzzle["test"]) == b)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(99173, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, profiles, palettes = set(), set(), set()
    kinds = {"mixed": 0, "all_singleton": 0, "all_group": 0,
             "large": 0, "narrow": 0, "small": 0, "diagonal": 0,
             "group_corners": 0, "singleton_chain": 0, "gap": 0,
             "many_groups": 0, "long_line": 0, "rectangle": 0,
             "branched": 0}
    varying_palette = 0
    smaller_test = 0
    same_dimension = 0
    novel_test_size = 0
    total_pairs = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pairs = puzzle["train"] + puzzle["test"]
        local_palettes = set()
        train_sizes = set().union(*(multi_sizes(pair) for pair in puzzle["train"]))
        test_sizes = set().union(*(multi_sizes(pair) for pair in puzzle["test"]))
        novel_test_size += bool(test_sizes - train_sizes)
        same_dimension += bool({pair["input"].shape for pair in puzzle["train"]} &
                               {pair["input"].shape for pair in puzzle["test"]})
        if min(x["input"].size for x in puzzle["test"]) < max(x["input"].size for x in puzzle["train"]):
            smaller_test += 1
        train_input_colors = set(int(v) for v in np.unique(puzzle["train"][0]["input"]))
        train_output_colors = set(int(v) for v in np.unique(puzzle["train"][0]["output"]))
        highlight = next(iter(train_output_colors - train_input_colors))
        for pair in pairs:
            total_pairs += 1
            inp, out = pair["input"], pair["output"]
            colors, counts = np.unique(inp, return_counts=True)
            bg = int(colors[np.argmax(counts)])
            fg = int(colors[np.argmin(counts)])
            local_palettes.add((bg, fg))
            palettes.add((bg, fg))
            groups = _components(inp == fg)
            sizes = tuple(sorted(len(x) for x in groups))
            profiles.add(sizes)
            shapes.add(_grid_key(inp == fg))
            assert np.array_equal(out, _cellwise_expected(inp, fg, highlight))
            n_multi = sum(size >= 2 for size in sizes)
            n_single = sum(size == 1 for size in sizes)
            kinds["mixed"] += n_multi > 0 and n_single > 0
            kinds["all_singleton"] += n_multi == 0
            kinds["all_group"] += n_single == 0
            kinds["large"] += max(sizes) >= 9
            kinds["narrow"] += min(inp.shape) <= 5 and max(inp.shape) >= 10
            kinds["small"] += inp.shape[0] <= 6 and inp.shape[1] <= 6
            kinds["diagonal"] += _singleton_group_diagonal(groups)
            multis = [group for group in groups if len(group) >= 2]
            singles = [group[0] for group in groups if len(group) == 1]
            kinds["many_groups"] += n_multi >= 4
            kinds["group_corners"] += any(
                abs(r - rr) == 1 and abs(c - cc) == 1
                for i, group in enumerate(multis) for other in multis[i + 1:]
                for r, c in group for rr, cc in other)
            kinds["singleton_chain"] += any(
                sum(abs(r - rr) == 1 and abs(c - cc) == 1
                    for rr, cc in singles if (rr, cc) != (r, c)) >= 2
                for r, c in singles)
            kinds["gap"] += any(
                (r == rr and abs(c - cc) == 2) or
                (c == cc and abs(r - rr) == 2)
                for i, group in enumerate(groups) for other in groups[i + 1:]
                for r, c in group for rr, cc in other)
            for group in multis:
                rows = {r for r, _ in group}
                cols = {c for _, c in group}
                kinds["long_line"] += len(group) >= 6 and (len(rows) == 1 or len(cols) == 1)
                kinds["rectangle"] += (len(rows) >= 2 and len(cols) >= 2 and
                                        len(group) == len(rows) * len(cols))
                group_set = set(group)
                kinds["branched"] += any(
                    sum((r + dr, c + dc) in group_set for dr, dc in _NEIGHBORS) >= 3
                    for r, c in group)
        varying_palette += len(local_palettes) > 1
    assert len(shapes) >= 100 and len(profiles) >= 25 and len(palettes) >= 20
    assert kinds["mixed"] >= 40 and kinds["all_singleton"] >= 2
    assert kinds["all_group"] >= 2 and kinds["large"] >= 10
    assert kinds["narrow"] >= 5 and kinds["diagonal"] >= 42
    assert kinds["diagonal"] < total_pairs and kinds["small"] >= 10
    assert kinds["group_corners"] >= 10 and kinds["singleton_chain"] >= 2
    assert kinds["gap"] >= 20 and kinds["many_groups"] >= 10
    assert kinds["long_line"] >= 5 and kinds["rectangle"] >= 5
    assert kinds["branched"] >= 5
    assert varying_palette >= 10 and smaller_test >= 3
    assert same_dimension >= 3 and novel_test_size >= 5
