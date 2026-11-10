"""Shape-label ARC puzzle generator."""

import numpy as np
from collections import Counter


BASE = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
)
ALTERNATIVES = (
    (),
    (((0, 0), (1, 0), (1, 1)),),
    (((0, 0), (0, 1), (0, 2)),),
    (((0, 0), (1, 0), (2, 0), (2, 1)),
     ((0, 0), (0, 1), (0, 2), (1, 1))),
    (((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
     ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))),
    (((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
     ((0, 0), (0, 1), (1, 0), (1, 1))),
)


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _transform(cells, operation):
    result = []
    for r, c in cells:
        if operation & 4:
            r, c = c, r
        if operation & 1:
            r = -r
        if operation & 2:
            c = -c
        result.append((r, c))
    return _normalize(result)


def _vocabulary(rng):
    # Only a coordinated transform of the vocabulary is used in one puzzle.
    # Each oriented template has one stable label, not a rotation-invariant label.
    for _ in range(30):
        kind = int(rng.integers(3))
        operation = 0 if kind == 0 else int(rng.integers(8))
        shapes = list(BASE)
        if kind == 2:
            chosen = rng.permutation(np.arange(1, 6))[:int(rng.integers(2, 5))]
            for raw in chosen:
                index = int(raw)
                options = ALTERNATIVES[index]
                shapes[index] = options[int(rng.integers(len(options)))]
        shapes = tuple(_transform(shape, operation) for shape in shapes)
        if len(set(shapes)) == 6:
            return shapes
    raise ValueError("no distinct vocabulary")


def _digit_grid(*rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


SOURCE = (
    (_digit_grid("667766777", "767776677", "777677776", "776667676",
                 "767677676", "767777777", "767777766", "777766676", "777766677"),
     _digit_grid("447788777", "747778877", "777377772", "773337972",
                 "727377972", "727777777", "727777744", "777755574", "777755577")),
    (_digit_grid("777777777", "767767777", "666767667", "767777767",
                 "777777777", "777766777", "777776676", "766677776", "766677776"),
     _digit_grid("777777777", "737797777", "333797447", "737777747",
                 "777777777", "777788777", "777778872", "755577772", "755577772")),
    (_digit_grid("767766777", "666776677", "767777776", "777666776",
                 "777666776", "777777777", "667777766", "767776776", "777776777"),
     _digit_grid("737788777", "333778877", "737777772", "777555772",
                 "777555772", "777777777", "447777744", "747779774", "777779777")),
)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
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


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    found = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        cells = {start}
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.add(neighbor)
                    stack.append(neighbor)
        found.append(cells)
    return found


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _contact_pairs(components):
    owner = {cell: i for i, component in enumerate(components) for cell in component}
    pairs = set()
    for (r, c), i in owner.items():
        for cell in ((r - 1, c - 1), (r - 1, c + 1),
                     (r + 1, c - 1), (r + 1, c + 1)):
            j = owner.get(cell)
            if j is not None and j != i:
                pairs.add(tuple(sorted((i, j))))
    return len(pairs)


def _dimensions(rng, names, shapes, mode, margin):
    band = int(rng.choice(3, p=(0.28, 0.48, 0.24)))
    low, high = ((6, 10), (9, 14), (13, 20))[band]
    low = max(low, 6 + max(0, len(names) - 5) // 2)
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    if mode == "crowded":
        height = min(height, max(low, 9 + len(names) // 3))
        width = min(width, max(low, 9 + len(names) // 3))
    area = sum(len(shapes[i]) for i in names)
    while height * width < 2.5 * area:
        if height <= width:
            height += 1
        else:
            width += 1
    return height + 2 * margin, width + 2 * margin


def _place(rng, names, shapes, dimensions, mode, margin):
    height, width = dimensions
    placements = {}
    for name in set(names):
        shape = shapes[name]
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        placements[name] = [tuple((r + top, c + left) for r, c in shape)
                            for top in range(margin, height - sh - margin + 1)
                            for left in range(margin, width - sw - margin + 1)]
    target = 0 if mode == "separated" else (int(rng.integers(1, 3)) if mode == "few" else max(3, len(names) // 2))
    for _ in range(16):
        occupied = {}
        placed = []
        order = list(range(len(names)))
        order.sort(key=lambda i: (-len(shapes[names[i]]), float(rng.random())))
        contacts = 0
        for instance in order:
            options = []
            for cells in placements[names[instance]]:
                if any((r, c) in occupied or (r - 1, c) in occupied or
                       (r + 1, c) in occupied or (r, c - 1) in occupied or
                       (r, c + 1) in occupied for r, c in cells):
                    continue
                neighbors = {occupied[point] for r, c in cells
                             for point in ((r - 1, c - 1), (r - 1, c + 1),
                                           (r + 1, c - 1), (r + 1, c + 1))
                             if point in occupied}
                new = len(neighbors)
                if mode == "separated" and new:
                    continue
                if mode == "few" and contacts + new > target:
                    continue
                options.append((cells, new))
            if not options:
                break
            if mode == "crowded":
                best = max(item[1] for item in options)
                pool = [item for item in options if item[1] >= best - 1]
            elif mode == "few" and placed and contacts < target:
                best = max(item[1] for item in options)
                pool = [item for item in options if item[1] == best]
            else:
                pool = options
            cells, new = pool[int(rng.integers(len(pool)))]
            for point in cells:
                occupied[point] = instance
            placed.append((instance, cells))
            contacts += new
        if len(placed) == len(names):
            result = [None] * len(names)
            for instance, cells in placed:
                result[instance] = cells
            contact = _contact_pairs([set(cells) for cells in result])
            if mode == "separated" or (mode == "few" and 1 <= contact <= 2) or (mode == "crowded" and contact >= 3):
                return result
    return None


def _pair_names(rng, forced, test=False):
    allowed = set(int(v) for v in forced)
    desired_classes = int(rng.integers(3, 7))
    while len(allowed) < desired_classes:
        allowed.add(int(rng.integers(6)))
    minimum = max(3, len(allowed))
    maximum = 11 if test else 10
    total = int(rng.integers(minimum, maximum + 1))
    names = list(allowed)
    while len(names) < total:
        names.append(int(rng.choice(tuple(sorted(allowed)))))
    rng.shuffle(names)
    return names


def _make_pair(rng, names, shapes, labels, backgrounds, mode, color_mode,
               previous_background=None, witness=False):
    background_options = [c for c in backgrounds if c != previous_background]
    background = int(rng.choice(background_options))
    margin = int(rng.choice((0, 1, 2), p=(0.55, 0.30, 0.15)))
    dimensions = _dimensions(rng, names, shapes, mode, margin)
    result = None
    for expansion in range(7):
        dims = (dimensions[0] + expansion, dimensions[1] + expansion)
        result = _place(rng, names, shapes, dims, mode, margin)
        if result is not None:
            break
    if result is None:
        # A requested contact band can be infeasible for a small geometry.
        # Widen the canvas and relax that local difficulty target deterministically.
        for fallback_mode in ("few", "separated"):
            dims = (min(30, dimensions[0] + 7), min(30, dimensions[1] + 7))
            result = _place(rng, names, shapes, dims, fallback_mode, margin)
            if result is not None:
                break
    if result is None:
        raise ValueError("could not place shapes")
    inp = np.full(dims, background, dtype=np.int8)
    out = np.full(dims, background, dtype=np.int8)
    candidates = ([c for c in backgrounds if c != background] if witness else
                  [c for c in range(10) if c != background])
    if color_mode == "mono":
        preferred = [c for c in candidates if c not in labels]
        colors = [int(rng.choice(preferred or candidates))] * len(names)
    else:
        color_count = int(rng.integers(2, min(4, len(candidates)) + 1))
        palette = list(int(v) for v in rng.choice(candidates, size=color_count, replace=False))
        colors = [int(rng.choice(palette)) for _ in names]
        if color_mode == "echo":
            i = int(rng.integers(len(names)))
            colors[i] = labels[names[i]]
        if len(set(colors)) == 1:
            colors[1] = next(c for c in candidates if c != colors[0])
    if all(colors[i] == labels[names[i]] for i in range(len(names))):
        colors[0] = next(c for c in candidates if c != labels[names[0]])
    for name, cells, color in zip(names, result, colors):
        for r, c in cells:
            inp[r, c] = color
            out[r, c] = labels[name]
    return {"input": inp, "output": out}


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _source_copy(pair):
    return any(np.array_equal(pair["input"], a) and np.array_equal(pair["output"], b)
               for a, b in SOURCE)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(rng, num_train, num_test)
    shapes = _vocabulary(rng)
    palette = [int(v) for v in rng.permutation(10)]
    labels = palette[:6]
    backgrounds = palette[6:]
    source_like = bool(rng.random() < 0.12)
    easy = bool(rng.random() < 0.25)
    forced = [set() for _ in range(train_count)]
    if source_like or easy:
        forced[0] = set(range(6))
    else:
        permutation = [int(v) for v in rng.permutation(6)]
        for index, name in enumerate(permutation):
            forced[index % min(train_count, 3)].add(name)
    plans = []
    for i in range(train_count):
        if source_like and i == 0:
            names = list(range(6))
        elif source_like and i == 1:
            names = list(range(6)) + [1, 2]
        else:
            names = _pair_names(rng, forced[i])
        plans.append(("train", names))
    for i in range(test_count):
        names = list(range(6)) + [2] if source_like and i == 0 else _pair_names(rng, (), True)
        plans.append(("test", names))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_background = None
    for index, (section, names) in enumerate(plans):
        mode = str(rng.choice(("separated", "few", "crowded"), p=(0.30, 0.38, 0.32)))
        color_mode = "mono" if index == 0 else ("multi" if index == 1 else
                     str(rng.choice(("mono", "multi", "echo"), p=(0.30, 0.38, 0.32))))
        for _ in range(20):
            pair = _make_pair(rng, names, shapes, labels, backgrounds, mode, color_mode,
                              previous_background if index < 2 else None,
                              section == "train" and index < min(3, train_count))
            key = _grid_key(pair["input"])
            if key not in seen and not _source_copy(pair):
                break
        else:
            raise ValueError("could not make a distinct pair")
        seen.add(key)
        previous_background = _background(pair["input"])
        puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    background = _background(inp)
    mask = inp != background
    assert 2 * int(np.count_nonzero(mask)) < inp.size
    assert np.array_equal(mask, out != background)
    assert np.all(out[~mask] == background)
    assert not np.array_equal(inp, out)
    components = _components(mask)
    assert 2 <= len(components) <= 11
    mapping = {}
    for component in components:
        shape = _normalize(component)
        assert 2 <= len(shape) <= 6
        input_colors = {int(inp[r, c]) for r, c in component}
        colors = {int(out[r, c]) for r, c in component}
        assert len(input_colors) == len(colors) == 1
        color = colors.pop()
        assert shape not in mapping or mapping[shape] == color
        mapping[shape] = color
    assert len(set(mapping.values())) == len(mapping)
    return background, mapping, components


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen_inputs = set()
    learned = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu"
                assert all(1 <= length <= 30 for length in grid.shape)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            key = _grid_key(pair["input"])
            assert key not in seen_inputs and not _source_copy(pair)
            seen_inputs.add(key)
            _, mapping, _ = _inspect_pair(pair)
            if split == "train":
                for shape, color in mapping.items():
                    assert shape not in learned or learned[shape] == color
                    learned[shape] = color
            else:
                assert all(shape in learned and learned[shape] == color
                           for shape, color in mapping.items())
    assert len(learned) == 6 and len(set(learned.values())) == 6
    for split in ("train", "test"):
        for pair in puzzle[split]:
            background, _, components = _inspect_pair(pair)
            assert background not in set(learned.values())
            prediction = np.full(pair["input"].shape, background, dtype=np.int8)
            for component in components:
                color = learned[_normalize(component)]
                for r, c in component:
                    prediction[r, c] = color
            assert np.array_equal(prediction, pair["output"])


def _solve_from_training(train, test_input):
    mapping = {}
    for pair in train:
        inp, out = pair["input"], pair["output"]
        for component in _components(inp != _background(inp)):
            shape = _normalize(component)
            colors = {int(out[r, c]) for r, c in component}
            assert len(colors) == 1
            color = colors.pop()
            assert shape not in mapping or mapping[shape] == color
            mapping[shape] = color
    background = _background(test_input)
    result = np.full(test_input.shape, background, dtype=np.int8)
    for component in _components(test_input != background):
        color = mapping[_normalize(component)]
        for r, c in component:
            result[r, c] = color
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    coverage = Counter()
    vocabularies = set()
    transformed_base = {frozenset(_transform(s, op) for s in BASE)
                        for op in range(8)}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        examples = [{role: pair[role].copy() for role in ("input", "output")}
                    for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(_solve_from_training(examples, pair["input"].copy()),
                                  pair["output"])
        observations = []
        training_colors = {}
        shapes_by_input_color = {}
        witnessed = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                components = _components(inp != bg)
                shapes = [_normalize(c) for c in components]
                colors = [int(inp[next(iter(c))]) for c in components]
                labels = [int(out[next(iter(c))]) for c in components]
                contact = _contact_pairs(components)
                h, w = inp.shape
                rows = [r for component in components for r, _ in component]
                cols = [c for component in components for _, c in component]
                margin = min(min(rows), min(cols), h - 1 - max(rows), w - 1 - max(cols))
                observations.append((split, h, w, len(components), bg))
                for condition, name in (
                    (contact == 0, "no_contact"), (1 <= contact <= 2, "few_contact"),
                    (contact >= 3, "many_contact"), (h != w, "rectangular"),
                    (h == w, "square"), (h <= 9 and w <= 9, "small"),
                    (h >= 14 or w >= 14, "large"), (margin == 0, "edge"),
                    (margin >= 2, "wide_margin"), (len(set(shapes)) < 6, "class_absent"),
                    (len(set(shapes)) < len(shapes), "class_repeated"),
                    (sum(v > 1 for v in Counter(shapes).values()) >= 2, "two_repeats"),
                    (len(set(colors)) > 1, "multicolor"),
                    (any(a == b for a, b in zip(colors, labels)), "echo"),
                    (h != w and len(set(colors)) > 1 and contact > 0, "combined"),
                ):
                    coverage[name] += int(condition)
                if split == "train":
                    for shape, color, label in zip(shapes, colors, labels):
                        training_colors.setdefault(shape, set()).add(color)
                        shapes_by_input_color.setdefault(color, set()).add(shape)
                        if color != label:
                            witnessed.add(shape)
                coverage["equal_area_geometry"] += len({shape for shape in shapes
                                                         if len(shape) == 4}) >= 2
        assert set(training_colors) == witnessed
        vocabulary = frozenset(training_colors)
        vocabularies.add(vocabulary)
        coverage["base_vocabulary" if vocabulary == frozenset(BASE) else
                 "transformed_vocabulary" if vocabulary in transformed_base else
                 "replacement_vocabulary"] += 1
        coverage["background_shift"] += len({x[4] for x in observations}) > 1
        coverage["size_shift"] += len({(x[1], x[2]) for x in observations}) > 1
        coverage["count_shift"] += len({x[3] for x in observations}) > 1
        coverage["same_shape_different_input"] += any(len(v) > 1 for v in training_colors.values())
        coverage["different_shapes_same_input"] += any(len(v) > 1 for v in shapes_by_input_color.values())
        train_counts = [x[3] for x in observations if x[0] == "train"]
        test_counts = [x[3] for x in observations if x[0] == "test"]
        coverage["test_more"] += any(n > max(train_counts) for n in test_counts)
        coverage["test_fewer"] += any(n < min(train_counts) for n in test_counts)
        coverage["source_like"] += (train_counts[:2] == [6, 8] and test_counts[0] == 7)
    assert len(vocabularies) >= 15
    for name, minimum in (("no_contact", 50), ("few_contact", 50),
                          ("many_contact", 50), ("rectangular", 100),
                          ("square", 50), ("small", 15), ("large", 50),
                          ("edge", 100), ("wide_margin", 20),
                          ("class_absent", 100), ("class_repeated", 100),
                          ("two_repeats", 80), ("multicolor", 100),
                          ("echo", 50), ("combined", 50),
                          ("base_vocabulary", 8), ("transformed_vocabulary", 8),
                          ("replacement_vocabulary", 8),
                          ("background_shift", 42), ("size_shift", 42),
                          ("count_shift", 42), ("same_shape_different_input", 30),
                          ("different_shapes_same_input", 30),
                          ("equal_area_geometry", 20),
                          ("test_more", 8), ("test_fewer", 3),
                          ("source_like", 2)):
        assert coverage[name] >= minimum, (name, coverage[name])
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_grid_key(pair["input"]) for p in puzzles
                for split in ("train", "test") for pair in p[split]}) > 200
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    assert _puzzle_key(generate_puzzle(123456, 7, 3)) == _puzzle_key(generate_puzzle(123456, 7, 3))
    for seed, train, test in ((701, 2, 1), (702, 10, 1), (703, 10, 4),
                              (801, None, 1), (802, None, 4),
                              (803, 2, None), (804, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 2.5}, {"num_train": True}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
