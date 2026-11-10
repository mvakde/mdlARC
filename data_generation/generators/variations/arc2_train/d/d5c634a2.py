"""Generate isolated horizontal T glyphs and their two unary tally registers."""

import numpy as np


_ABOVE = frozenset({(0, 1), (1, 0), (1, 1), (1, 2)})
_BELOW = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
_SLOTS = ((0, 0), (2, 0), (0, 2), (2, 2))
_COUNTS = tuple((a, b) for a in range(1, 5) for b in range(1, 5))
_SOURCE_PAIRS = {
    ("0200222/2220020/0000000/0000222/0020020/0222000", "300100/000000/300100"),
    ("0222000/0020000/0000020/2220222/0200000/0000000/0022200/0002000", "300101/000000/000100"),
    ("0200000/2220020/0000222/0020000/0222000/0000222/0200020/2220000", "303100/000000/303000"),
    ("2220/0200/0000/0200/2220", "300100/000000/000000"),
    ("22200/02000/00000/00222/02020/22200", "300100/000000/000100"),
    ("0000000/2220000/0200020/0000222/0000000/0020000/0222020/0000222", "303100/000000/300000"),
    ("000000000000000/000200000000000/002220000000000/000000000000000/000000022200000/222000002000000/020000000000000/000000000000000/000002220000000/002000200000020/022200000000222/000000000000000/000000222000000/000000020000000/000000000000000", "303101/000000/300101"),
    ("00000000000/00200000000/02220002220/00000000200/00222000000/00020000000/00000022200/00200002000/02220000000/00000000000", "300101/000000/300100"),
    ("000000222/020000020/222002000/000022200/000200000/002220222/000000020/222000000/020022200/000002000", "303101/000000/300101"),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _pair_numbers(seed, num_train, num_test):
    train = _integer(num_train, "num_train", 2, 10)
    test = _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _count_plan(rng, ntrain, ntest):
    # A same-total contrast forces orientation, rather than total count, to matter.
    totals = list(range(3, 8))
    rng.shuffle(totals)
    for total in totals:
        contrasts = [p for p in _COUNTS if sum(p) == total]
        if len(contrasts) < 2:
            continue
        chosen = [contrasts[int(i)] for i in rng.choice(len(contrasts), 2, replace=False)]
        rest = [p for p in _COUNTS if p not in chosen]
        rng.shuffle(rest)
        train = chosen + rest[:ntrain - 2]
        rng.shuffle(train)
        left = {a for a, _ in train}
        right = {b for _, b in train}
        possible = [p for p in _COUNTS if p[0] in left and p[1] in right]
        familiar = [p for p in possible if p in train]
        novel = [p for p in possible if p not in train]
        rng.shuffle(familiar)
        rng.shuffle(novel)
        if len(possible) < ntest:
            continue
        # Both a repeated result and a new combination appear across seeds.
        lead = familiar if int(rng.integers(0, 2)) else novel
        tail = novel if lead is familiar else familiar
        test = (lead + tail)[:ntest]
        rng.shuffle(test)
        return train, test
    raise AssertionError("unable to choose count plan")


def _dimensions(rng, count, mode):
    if mode == "compact":
        area_per = rng.uniform(9.0, 13.0)
    elif mode == "roomy":
        area_per = rng.uniform(21.0, 36.0)
    else:
        area_per = rng.uniform(13.0, 22.0)
    area = int(round(count * area_per))
    aspect = rng.uniform(0.55, 1.9)
    height = max(2, int(round(np.sqrt(area * aspect))))
    width = max(3, int(round(area / height)))
    height = min(24, height)
    width = min(24, width)
    if count == 2 and mode == "compact" and rng.random() < 0.28:
        height, width = (5, 4) if rng.random() < 0.5 else (4, 5)
    # Four cells per glyph leave an identifiable majority background.
    # Rounding the aspect and area can otherwise produce a 4-by-4 tie.
    while height * width <= 8 * count:
        expansions = []
        if height < 24:
            expansions.append((height + 1, width))
        if width < 24:
            expansions.append((height, width + 1))
        _require(bool(expansions), "no room for a majority background")
        height, width = expansions[int(rng.integers(0, len(expansions)))]
    return height, width


def _cells(row, col, orientation):
    shape = _ABOVE if orientation == 0 else _BELOW
    return tuple((row + dr, col + dc) for dr, dc in shape)


def _place(rng, height, width, orientations, arrangement):
    for _ in range(18):
        occupied = set()
        placements = []
        order = [int(i) for i in rng.permutation(len(orientations))]
        failed = False
        for index in order:
            options = []
            for row in range(height - 1):
                for col in range(width - 2):
                    cells = _cells(row, col, orientations[index])
                    if any((r + dr, c + dc) in occupied
                           for r, c in cells
                           for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
                        continue
                    if occupied:
                        nearest = min(abs(r - rr) + abs(c - cc)
                                      for r, c in cells for rr, cc in occupied)
                    else:
                        nearest = 0
                    edge = min(row, col, height - row - 2, width - col - 3)
                    options.append((row, col, nearest, edge))
            if not options:
                failed = True
                break
            if arrangement == "cluster" and occupied:
                options.sort(key=lambda x: x[2])
                options = options[:max(1, len(options) // 4)]
            elif arrangement == "spread" and occupied:
                options.sort(key=lambda x: -x[2])
                options = options[:max(1, len(options) // 3)]
            elif arrangement == "edge":
                options.sort(key=lambda x: x[3])
                options = options[:max(1, len(options) // 3)]
            row, col, _, _ = options[int(rng.integers(0, len(options)))]
            cells = _cells(row, col, orientations[index])
            placements.append((index, cells))
            occupied.update(cells)
        if not failed:
            return placements
    return None


def _layout(rng, orientations, mode, arrangement):
    height, width = _dimensions(rng, len(orientations), mode)
    for growth in range(12):
        placed = _place(rng, height, width, orientations, arrangement)
        if placed is not None:
            return height, width, placed
        if (height <= width and height < 24) or width >= 24:
            if height >= 24:
                break
            height += 1
        else:
            width += 1
    raise AssertionError("unable to place glyphs")


def _pair(rng, counts, tally, mode, arrangement, background, glyph_colors):
    above, below = counts
    orientations = [0] * above + [1] * below
    height, width, placements = _layout(rng, orientations, mode, arrangement)
    grid = np.full((height, width), background, dtype=np.int8)
    if len(glyph_colors) == 1:
        colors = [glyph_colors[0]] * len(orientations)
    else:
        # One hue spans both orientations; remaining whole-glyph hues are independent.
        colors = [int(rng.choice(glyph_colors)) for _ in orientations]
        colors[0] = glyph_colors[0]
        colors[above] = glyph_colors[0]
        if len(orientations) > 2:
            alternate = 1 if above > 1 else above + 1
            colors[alternate] = glyph_colors[1]
    for index, cells in placements:
        for row, col in cells:
            grid[row, col] = colors[index]
    output = np.full((3, 6), background, dtype=np.int8)
    for i in range(above):
        row, col = _SLOTS[i]
        output[row, col] = tally[0]
    for i in range(below):
        row, col = _SLOTS[i]
        output[row, col + 3] = tally[1]
    return {"input": grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed(seed)
    train_number, test_number = _pair_numbers(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    if seed % 9 == 0:
        tally = (3, 1)
    else:
        tally = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    train_counts, test_counts = _count_plan(rng, train_number, test_number)
    colors = [c for c in range(10) if c not in tally]
    train_modes = ["compact", "roomy"] + [
        ("compact", "medium", "roomy")[int(rng.integers(0, 3))]
        for _ in range(train_number - 2)
    ]
    rng.shuffle(train_modes)
    test_modes = [
        ("compact", "compact", "medium", "roomy")[int(rng.integers(0, 4))]
        for _ in range(test_number)
    ]
    arrangements = ("scatter", "cluster", "spread", "edge")
    pair_backgrounds = [int(x) for x in rng.choice(colors, train_number, replace=True)]
    if seed % 9 == 0:
        pair_backgrounds[0] = 0
    if len(set(pair_backgrounds)) == 1:
        pair_backgrounds[1] = colors[(colors.index(pair_backgrounds[0]) + 1) % len(colors)]
    pair_backgrounds += [int(rng.choice(colors)) for _ in range(test_number)]
    monochrome_train = int(rng.integers(0, train_number))
    if seed % 9 == 0:
        monochrome_train = 0
    multi_candidates = [i for i, counts in enumerate(train_counts)
                        if sum(counts) >= 3 and i != monochrome_train]
    if not multi_candidates:
        monochrome_train = next(i for i, counts in enumerate(train_counts)
                                if sum(counts) == 2)
        multi_candidates = [i for i, counts in enumerate(train_counts)
                            if sum(counts) >= 3]
    multicolor_train = int(rng.choice(multi_candidates))
    used_monochrome = set()
    used_inputs = set()
    pairs = {"train": [], "test": []}
    for split, counts, modes in (("train", train_counts, train_modes),
                                  ("test", test_counts, test_modes)):
        for index, (count_pair, mode) in enumerate(zip(counts, modes)):
            global_index = index if split == "train" else train_number + index
            background = pair_backgrounds[global_index]
            available = [c for c in colors if c != background]
            if seed % 9 == 0 and split == "train" and index == 0:
                background = 0
                available = [c for c in colors if c != background]
                glyph_colors = [2]
            elif split == "train" and index == monochrome_train:
                alternatives = [c for c in available if c not in used_monochrome]
                glyph_colors = [int(rng.choice(alternatives or available))]
            elif split == "train" and index == multicolor_train:
                glyph_colors = [int(c) for c in rng.choice(available, 2, replace=False)]
            elif rng.random() < 0.38:
                glyph_colors = [int(c) for c in rng.choice(available, 2 + int(rng.integers(0, 3)), replace=False)]
            else:
                glyph_colors = [int(rng.choice(available))]
            if split == "train" and len(glyph_colors) == 1:
                used_monochrome.add(glyph_colors[0])
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _pair(pair_rng, count_pair, tally, mode,
                         arrangements[int(rng.integers(0, 4))],
                         background, glyph_colors)
            grid = pair["input"]
            # Preserve the sampled scene whenever possible. Extra background
            # margin is a deterministic fallback for an exact source copy or
            # an input collision in a small puzzle.
            selected = False
            for extra_rows in range(31 - grid.shape[0]):
                for extra_cols in range(31 - grid.shape[1]):
                    candidate = np.pad(grid, ((0, extra_rows), (0, extra_cols)),
                                       constant_values=background)
                    if (_key(candidate) not in used_inputs and
                            (_text(candidate), _text(pair["output"])) not in _SOURCE_PAIRS):
                        pair["input"] = candidate
                        used_inputs.add(_key(candidate))
                        selected = True
                        break
                else:
                    continue
                break
            _require(selected, "no distinct canvas remains for this pair")
            pairs[split].append(pair)
    validate_generated_puzzle(pairs)
    return pairs


def _text(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _components(grid, background):
    foreground = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    groups = []
    for point in np.argwhere(foreground):
        start = tuple(int(x) for x in point)
        if seen[start]:
            continue
        queue = [start]
        seen[start] = True
        group = []
        while queue:
            row, col = queue.pop()
            group.append((row, col))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and foreground[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        groups.append(group)
    return groups


def _inspect(pair):
    grid, output = pair["input"], pair["output"]
    _require(output.shape == (3, 6), "output must be 3 by 6")
    background = int(output[1, 1])
    _require(background in grid, "output background absent from input")
    _require(int(np.count_nonzero(grid == background)) > grid.size // 2,
             "background must be identifiable as majority")
    above = below = 0
    above_hues, below_hues = set(), set()
    groups = _components(grid, background)
    _require(2 <= len(groups) <= 8, "glyph count outside 2 through 8")
    glyph_colors = []
    anchors = []
    for group in groups:
        _require(len(group) == 4, "glyph is not four cells")
        color = int(grid[group[0]])
        _require(all(int(grid[p]) == color for p in group), "glyph has mixed colors")
        glyph_colors.append(color)
        r0 = min(r for r, _ in group)
        c0 = min(c for _, c in group)
        shape = frozenset((r-r0, c-c0) for r, c in group)
        _require(shape in (_ABOVE, _BELOW), "not a horizontal T")
        anchors.append((r0, c0))
        if shape == _ABOVE:
            above += 1
            above_hues.add(color)
        else:
            below += 1
            below_hues.add(color)
    _require(1 <= above <= 4 and 1 <= below <= 4,
             "each orientation must appear one through four times")
    tally = (int(output[0, 0]), int(output[0, 3]))
    _require(len(set((background,) + tally + tuple(glyph_colors))) >= 4,
             "color roles collide")
    _require(background not in tally and tally[0] != tally[1]
             and all(c not in tally for c in glyph_colors), "color roles collide")
    expected = np.full((3, 6), background, dtype=output.dtype)
    for i in range(above):
        r, c = _SLOTS[i]
        expected[r, c] = tally[0]
    for i in range(below):
        r, c = _SLOTS[i]
        expected[r, c+3] = tally[1]
    _require(np.array_equal(output, expected), "incorrect unary tally")
    return {"above": above, "below": below, "background": background,
            "tally": tally, "glyph_colors": tuple(glyph_colors),
            "dimensions": grid.shape, "density": 4*len(groups)/grid.size,
            "anchors": tuple(anchors), "shared_hues": above_hues & below_hues}


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
        _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
        nt, ns = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
        keys = set()
        facts = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
                for grid in pair.values():
                    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                    _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                    _require(all(1 <= x <= 30 for x in grid.shape), "grid dimensions")
                    _require(0 <= int(grid.min()) and int(grid.max()) <= 9, "ARC colors")
                key = _key(pair["input"])
                _require(key not in keys, "duplicate input")
                keys.add(key)
                _require((_text(pair["input"]), _text(pair["output"])) not in _SOURCE_PAIRS,
                         "source pair copied")
                facts[split].append(_inspect(pair))
        tallies = {f["tally"] for s in facts for f in facts[s]}
        _require(len(tallies) == 1, "register colors differ between pairs")
        train = facts["train"]
        _require(len({f["background"] for f in train}) >= 2,
                 "training backgrounds need variation")
        _require(any(len(set(f["glyph_colors"])) == 1 for f in train),
                 "training needs monochrome example")
        _require(any(len(set(f["glyph_colors"])) > 1 for f in train),
                 "training needs multi-hue example")
        _require(any(len(set(f["glyph_colors"])) > 1 and f["shared_hues"]
                     for f in train), "training does not disprove hue-based orientation")
        _require(any(f["above"] != f["below"] for f in train), "orientation mapping unclear")
        _require(any(a["above"] + a["below"] == b["above"] + b["below"]
                     and a["above"] != b["above"]
                     for i, a in enumerate(train) for b in train[i+1:]),
                 "training lacks same-total contrast")
        seen_above = {f["above"] for f in train}
        seen_below = {f["below"] for f in train}
        _require(all(f["above"] in seen_above and f["below"] in seen_below
                     for f in facts["test"]), "test tally state not demonstrated")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def self_test() -> None:
    puzzles = []
    dimensions, layouts, count_plans, tally_palettes = set(), set(), set(), set()
    compact_train = roomy_test = diagonal = multi = low_test = familiar = novel = 0
    source_palette = source_size = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts = {(f["above"], f["below"]) for f in map(_inspect, puzzle["train"])}
        test_counts = {(f["above"], f["below"]) for f in map(_inspect, puzzle["test"])}
        count_plans.add((tuple(sorted(train_counts)), tuple(sorted(test_counts))))
        familiar += bool(train_counts & test_counts)
        novel += bool(test_counts - train_counts)
        tally_palettes.add(_inspect(puzzle["train"][0])["tally"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _inspect(pair)
                dimensions.add(f["dimensions"])
                layouts.add((f["dimensions"], f["anchors"], f["above"], f["below"]))
                multi += len(set(f["glyph_colors"])) > 1
                source_palette += (f["background"] == 0 and
                                   set(f["glyph_colors"]) == {2} and
                                   f["tally"] == (3, 1))
                if split == "train":
                    compact_train += f["density"] >= .3
                    source_size += f["dimensions"] == (5, 4)
                else:
                    roomy_test += f["density"] <= .2
                    low_test += f["above"] + f["below"] <= 4
                groups = _components(pair["input"], f["background"])
                diagonal += any(abs(r-rr) == 1 and abs(c-cc) == 1
                                for i, g in enumerate(groups) for h in groups[i+1:]
                                for r, c in g for rr, cc in h)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len(dimensions) >= 15 and len(layouts) >= 100, "layout diversity")
    _require(len(count_plans) >= 12 and len(tally_palettes) >= 15, "plan or color diversity")
    _require(compact_train >= 20 and roomy_test >= 10 and diagonal >= 30,
             "density or staggered-layout diversity is insufficient")
    _require(multi >= 20 and low_test >= 10 and familiar >= 5 and novel >= 5,
             "color or test-count diversity is insufficient")
    _require(source_palette >= 1 and source_size >= 1,
             "source palette or small source-like grid is missing")
    for seed, nt, ns in ((900, 2, 1), (901, 10, 1), (902, 10, 4),
                         (903, None, 1), (904, None, 4),
                         (905, 2, None), (906, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            _require(len(p["train"]) == nt, "explicit train count")
        if ns is not None:
            _require(len(p["test"]) == ns, "explicit test count")
    regression = generate_puzzle(5532021848150028142)
    for split in ("train", "test"):
        for pair in regression[split]:
            background = int(pair["output"][1, 1])
            _require(int(np.count_nonzero(pair["input"] == background))
                     > pair["input"].size // 2,
                     "regression: tied background and foreground")
    a, b = generate_puzzle(701, 7, 3), generate_puzzle(701, 7, 3)
    _require(all(np.array_equal(x[k], y[k]) for split in ("train", "test")
                 for x, y in zip(a[split], b[split]) for k in ("input", "output")),
             "not deterministic")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2),
                 (3, 4), (4.0, 1), (3, True)):
        try:
            generate_puzzle(950, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corrupted = generate_puzzle(22)
    corrupted["test"][0]["output"][1, 1] = corrupted["train"][0]["output"][0, 0]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
