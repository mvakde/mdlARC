"""Four identical 2x2 tiles are repacked into a 5x5 separated square."""

from itertools import combinations

import numpy as np


_MOTIFS = (np.ones((2, 2), dtype=np.uint8),) + tuple(
    np.array([[int((r, c) != missing) for c in range(2)] for r in range(2)],
             dtype=np.uint8)
    for missing in ((0, 0), (0, 1), (1, 0), (1, 1))
)
_SOURCE_INPUTS = {
    ("888800", "888888", "088088", "088000", "000000", "000000"),
    ("008800", "888800", "888888", "008888", "000000", "000000"),
    ("000800", "088800", "888080", "088880", "000000", "000000"),
    ("000800", "088880", "888880", "088000", "000000", "000000"),
    ("000800", "008880", "080880", "888000", "088000", "000000"),
    ("080800", "888880", "000888", "000088", "000000", "000000"),
    ("008800", "888800", "880880", "088880", "088000", "000000"),
}
_CATALOG = None


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _canonical(anchors):
    r0 = min(r for r, _ in anchors)
    c0 = min(c for _, c in anchors)
    return tuple(sorted((r - r0, c - c0) for r, c in anchors))


def _cells(anchor, motif):
    ar, ac = anchor
    return frozenset((ar + r, ac + c) for r in range(2) for c in range(2)
                     if motif[r, c])


def _mask(anchors, motif, shape):
    result = np.zeros(shape, dtype=np.uint8)
    for anchor in anchors:
        for r, c in _cells(anchor, motif):
            result[r, c] = 1
    return result


def _decompositions(mask, motif, limit=2):
    target = frozenset(map(tuple, np.argwhere(mask)))
    size = int(np.sum(motif))
    if len(target) != 4 * size:
        return []
    h, w = mask.shape
    options = []
    for r in range(h - 1):
        for c in range(w - 1):
            cells = _cells((r, c), motif)
            if cells <= target:
                options.append(((r, c), cells))
    solutions = []

    def search(remaining, chosen):
        if len(solutions) >= limit:
            return
        if not remaining:
            if len(chosen) == 4:
                solution = tuple(sorted(chosen))
                if solution not in solutions:
                    solutions.append(solution)
            return
        if len(chosen) >= 4 or len(remaining) != (4 - len(chosen)) * size:
            return
        pivot = min(remaining)
        for anchor, cells in options:
            if pivot in cells and cells <= remaining:
                search(remaining - cells, chosen + (anchor,))
                if len(solutions) >= limit:
                    return

    search(target, ())
    return solutions


def _unique_decomposition(mask):
    indices = (0,) if int(np.count_nonzero(mask)) == 16 else range(1, 5)
    found = []
    for index in indices:
        for anchors in _decompositions(mask, _MOTIFS[index], 2):
            found.append((index, anchors))
            if len(found) > 1:
                return []
    return found


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
    return count


def _contact_edges(anchors, motif):
    owner = {cell: index for index, anchor in enumerate(anchors)
             for cell in _cells(anchor, motif)}
    return sum(owner[(r, c)] != owner[(r + dr, c + dc)]
               for r, c in owner for dr, dc in ((1, 0), (0, 1))
               if (r + dr, c + dc) in owner)


def _category(mask, anchors):
    rows = {r for r, _ in anchors}
    cols = {c for _, c in anchors}
    regular = len(rows) == len(cols) == 2 and all(
        (r, c) in anchors for r in rows for c in cols
    )
    if regular or _components(mask) > 1:
        return "simple"
    if (len(rows) == 4 or len(cols) == 4 or
            max(rows) - min(rows) >= 4 or max(cols) - min(cols) >= 4):
        return "extended"
    return "compact"


def _catalog():
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    catalog = {i: {name: [] for name in ("simple", "compact", "extended")}
               for i in range(5)}
    positions = tuple((r, c) for r in range(5) for c in range(5))
    for index, motif in enumerate(_MOTIFS):
        seen = set()
        for anchors in combinations(positions, 4):
            canonical = _canonical(anchors)
            if canonical in seen:
                continue
            seen.add(canonical)
            occupied = set()
            for anchor in anchors:
                cells = _cells(anchor, motif)
                if occupied & cells:
                    break
                occupied.update(cells)
            else:
                mask = _mask(anchors, motif, (6, 6))
                if _unique_decomposition(mask) == [(index, tuple(anchors))]:
                    catalog[index][_category(mask, anchors)].append(tuple(anchors))
    for groups in catalog.values():
        for name, layouts in groups.items():
            if not layouts:
                raise RuntimeError(f"empty {name} layout catalog")
    _CATALOG = catalog
    return catalog


def _output(motif, background, foreground):
    result = np.full((5, 5), background, dtype=np.int8)
    patch = np.where(motif, foreground, background)
    for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)):
        result[r:r + 2, c:c + 2] = patch
    return result


def _counts(rng, num_train, num_test):
    def check(value, name, low, high):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    train = check(num_train, "num_train", 2, 10)
    test = check(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _palette(rng, excluded=None):
    colors = rng.choice(10, size=2, replace=False)
    answer = (int(colors[0]), int(colors[1]))
    return (answer[1], answer[0]) if answer == excluded else answer


def _source_copy(input_grid):
    if input_grid.shape != (6, 6) or not set(np.unique(input_grid)) <= {0, 8}:
        return False
    return tuple("".join(str(int(x)) for x in row) for row in input_grid) in _SOURCE_INPUTS


def _generate_pair(rng, index, category, colors, shape, used):
    candidates = _catalog()[index][category]
    motif = _MOTIFS[index]
    h, w = shape
    ordered = [candidates[int(raw)] for raw in rng.permutation(len(candidates))]
    if category == "simple":
        block = ((0, 0), (0, 2), (2, 0), (2, 2))
        corners = ((0, 0), (0, 4), (4, 0), (4, 4))
        style = int(rng.choice(4, p=(0.18, 0.18, 0.14, 0.50)))

        def preferred(anchors):
            if style == 0:
                return anchors == block
            if style == 1:
                return anchors == corners
            rows = {r for r, _ in anchors}
            cols = {c for _, c in anchors}
            regular = len(rows) == len(cols) == 2 and all(
                (r, c) in anchors for r in rows for c in cols
            )
            return regular if style == 2 else not regular

        ordered.sort(key=lambda anchors: not preferred(anchors))
    for anchors in ordered:
        key = (index, _canonical(anchors))
        if key in used:
            continue
        bbox_h = max(r for r, _ in anchors) + 2
        bbox_w = max(c for _, c in anchors) + 2
        if bbox_h > h or bbox_w > w:
            continue
        dr = int(rng.integers(h - bbox_h + 1))
        dc = int(rng.integers(w - bbox_w + 1))
        shifted = tuple((r + dr, c + dc) for r, c in anchors)
        mask = _mask(shifted, motif, shape)
        background, foreground = colors
        input_grid = np.where(mask, foreground, background).astype(np.int8)
        if _source_copy(input_grid):
            continue
        used.add(key)
        return {"input": input_grid, "output": _output(motif, *colors)}
    if category == "compact":
        return _generate_pair(rng, index, "extended", colors, shape, used)
    raise RuntimeError("layout catalog exhausted")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    _catalog()

    train_motifs = [0, int(rng.integers(1, 5))]
    if train_count >= 3:
        train_motifs.append(int(rng.choice([i for i in range(1, 5)
                                             if i != train_motifs[1]])))
    while len(train_motifs) < train_count:
        train_motifs.append(int(rng.integers(0, 5)))
    rng.shuffle(train_motifs)
    test_motifs = [int(rng.choice(train_motifs)) for _ in range(test_count)]

    categories = ("compact", "extended", "simple")
    test_categories = [str(rng.choice(categories, p=(0.25, 0.45, 0.30)))
                       for _ in range(test_count)]
    train_categories = ["compact"]
    if "simple" in test_categories:
        train_categories.append("simple")
    else:
        train_categories.append(str(rng.choice(("extended", "simple"))))
    if train_count >= 3:
        train_categories.append("extended" if "extended" not in train_categories
                                else "simple")
    while len(train_categories) < train_count:
        train_categories.append(str(rng.choice(categories)))
    rng.shuffle(train_categories)

    train_shapes = [(int(rng.integers(6, 11)), int(rng.integers(6, 11)))
                    for _ in range(train_count)]
    if len(set(train_shapes)) == 1:
        h, w = train_shapes[-1]
        train_shapes[-1] = (6 if h != 6 else 7, w)
    test_shapes = [(int(rng.integers(6, 11)), int(rng.integers(6, 11)))
                   for _ in range(test_count)]
    palettes = [_palette(rng)]
    for _ in range(1, train_count):
        palettes.append(_palette(rng, palettes[0] if len(palettes) == 1 else None))
    test_palettes = [_palette(rng) for _ in range(test_count)]

    used = set()
    train = []
    test = []
    for motifs, regimes, shapes, colors_list, result in (
        (train_motifs, train_categories, train_shapes, palettes, train),
        (test_motifs, test_categories, test_shapes, test_palettes, test),
    ):
        for index, category, shape, colors in zip(motifs, regimes, shapes, colors_list):
            pair_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            pair_rng = np.random.default_rng(pair_seed)
            result.append(_generate_pair(pair_rng, index, category, colors, shape, used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    for name in ("input", "output"):
        grid = pair[name]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
        _require(all(1 <= x <= 30 for x in grid.shape), "grid dimensions")
        _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
        _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid colors")
    inp, out = pair["input"], pair["output"]
    _require(6 <= inp.shape[0] <= 10 and 6 <= inp.shape[1] <= 10,
             "input canvas range")
    _require(out.shape == (5, 5), "output shape")
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 2 and counts[0] != counts[1], "color roles unclear")
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    mask = inp == foreground
    solutions = _unique_decomposition(mask)
    _require(len(solutions) == 1, "ambiguous four-copy decomposition")
    index, anchors = solutions[0]
    _require(np.array_equal(out, _output(_MOTIFS[index], background, foreground)),
             "incorrect normalized output")
    _require(not _source_copy(inp), "original source pair copied")
    return index, _canonical(anchors), _category(mask, anchors), (background, foreground)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "puzzle splits")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "pair counts")
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            ik = _grid_key(pair["input"])
            pk = (ik, _grid_key(pair["output"]))
            _require(ik not in seen_inputs and pk not in seen_pairs, "duplicate pair")
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            layout = info[:2]
            _require(layout not in seen_layouts, "repeated layout")
            seen_layouts.add(layout)
            information[split].append(info)
    train = information["train"]
    test = information["test"]
    demonstrated = {item[0] for item in train}
    _require(0 in demonstrated and bool(demonstrated & {1, 2, 3, 4}),
             "training must demonstrate solid and L tiles")
    _require({item[0] for item in test} <= demonstrated,
             "test orientation absent from training")
    _require("compact" in {item[2] for item in train}, "no compact demonstration")
    if "simple" in {item[2] for item in test}:
        _require("simple" in {item[2] for item in train},
                 "simple test placement lacks demonstration")
    _require(len({pair["input"].shape for pair in puzzle["train"]}) >= 2,
             "training canvas size does not vary")
    _require(len({item[3] for item in train}) >= 2,
             "training palette does not vary")


def _fingerprint(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    auto = generate_puzzle(12345)
    _require(_fingerprint(auto) == _fingerprint(generate_puzzle(12345)),
             "automatic determinism")
    explicit = generate_puzzle(98765, 7, 3)
    _require(_fingerprint(explicit) == _fingerprint(generate_puzzle(98765, 7, 3)),
             "explicit determinism")
    for seed, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                          (None, 1), (None, 4), (2, None),
                                          (10, None))):
        p = generate_puzzle(20000 + seed, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            _require(len(p["train"]) == train, "explicit train count")
        if test is not None:
            _require(len(p["test"]) == test, "explicit test count")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(22000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts narrow")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts narrow")
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "puzzles repeat")
    motifs, categories, sizes, palettes, layouts, combinations_seen = (
        set(), set(), set(), set(), set(), set()
    )
    within_categories = set()
    component_counts = set()
    connected_contacts = set()
    tall_solid_train = wide_l_train = 0
    compact_tests = extended_tests = 0
    solid_blocks = solid_corners = new_test_colors = 0
    all_category_puzzles = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        local_categories = set()
        train_colors = {int(color) for pair in p["train"]
                        for color in np.unique(pair["input"])}
        for split in ("train", "test"):
            for pair in p[split]:
                index, layout, category, palette = _inspect_pair(pair)
                motifs.add(index)
                categories.add(category)
                sizes.add(pair["input"].shape)
                palettes.add(palette)
                layouts.add((index, layout))
                combinations_seen.add((index, category, pair["input"].shape, palette))
                local_categories.add(category)
                component_counts.add(_components(pair["input"] == palette[1]))
                if _components(pair["input"] == palette[1]) == 1:
                    connected_contacts.add(_contact_edges(layout, _MOTIFS[index]))
                rows = {r for r, _ in layout}
                cols = {c for _, c in layout}
                if split == "train":
                    tall_solid_train += int(index == 0 and len(rows) == 4)
                    wide_l_train += int(index > 0 and len(cols) == 4)
                else:
                    compact_tests += int(category == "compact")
                    extended_tests += int(category == "extended")
                    new_test_colors += int(any(color not in train_colors
                                               for color in palette))
                solid_blocks += int(index == 0 and layout ==
                                    ((0, 0), (0, 2), (2, 0), (2, 2)))
                solid_corners += int(index == 0 and layout ==
                                     ((0, 0), (0, 4), (4, 0), (4, 4)))
        within_categories.add(len(local_categories))
        all_category_puzzles += int(len(local_categories) == 3)
    _require(motifs == set(range(5)), "missing tile orientation")
    _require(categories == {"simple", "compact", "extended"}, "packing range")
    _require(len(sizes) >= 15, "canvas sizes narrow")
    _require(any(h != w for h, w in sizes), "rectangular canvases absent")
    _require(len(palettes) >= 40, "pair palettes narrow")
    _require(len(layouts) >= 100, "layout range narrow")
    _require(len(combinations_seen) >= 150, "variation axes collapse")
    _require(all_category_puzzles >= 20, "packing families not mixed within puzzles")
    _require(component_counts == {1, 2, 3, 4}, "contact range narrow")
    _require(min(connected_contacts) <= 3 and max(connected_contacts) >= 7,
             "connected tile contacts narrow")
    _require(tall_solid_train >= 5 and wide_l_train >= 5,
             "extended packings absent from training")
    _require(compact_tests >= 5 and extended_tests >= 5,
             "test packing difficulty fixed")
    _require(solid_blocks >= 1 and solid_corners >= 1,
             "simple solid examples absent")
    _require(new_test_colors >= 5, "test-only literal colors absent")
