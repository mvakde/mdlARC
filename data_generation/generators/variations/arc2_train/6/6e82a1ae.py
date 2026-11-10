"""Generate recoloring puzzles where an object's edge-connected cell count is its class."""

import numpy as np


_SOURCE_INPUT_ROWS = frozenset({
    ("0000000000", "0000000000", "0555000500", "0000000500", "0000000000",
     "0005500000", "0005000000", "0000000500", "0550005550", "0550000000"),
    ("0000000000", "0000000550", "0550000550", "0055000000", "0000000000",
     "0000000005", "0000055005", "0500000005", "0500500000", "0005500000"),
    ("0000000000", "0050005500", "0050000500", "0000000000", "0000000000",
     "0000550000", "5000000000", "5500000000", "5000000000", "0000000000"),
    ("0000000005", "0000000005", "0055055005", "0550005005", "0000000000",
     "0000000000", "5000000000", "5005500500", "5000000500", "0000000000"),
})


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _variants(cells):
    answer = set()
    for reflected in (False, True):
        for turns in range(4):
            transformed = []
            for row, col in cells:
                r, c = row, -col if reflected else col
                for _ in range(turns):
                    r, c = c, -r
                transformed.append((r, c))
            answer.add(_normalize(transformed))
    return tuple(sorted(answer))


_FAMILIES = {
    2: (_variants(((0, 0), (0, 1))),),
    3: (_variants(((0, 0), (0, 1), (0, 2))),
        _variants(((0, 0), (1, 0), (1, 1)))),
    4: (_variants(((0, 0), (0, 1), (0, 2), (0, 3))),
        _variants(((0, 0), (0, 1), (1, 0), (1, 1))),
        _variants(((0, 0), (1, 0), (2, 0), (2, 1))),
        _variants(((0, 0), (0, 1), (0, 2), (1, 1))),
        _variants(((0, 1), (0, 2), (1, 0), (1, 1)))),
}


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _shape(rng, size, straight_allowed, force_straight=False):
    families = _FAMILIES[size]
    if size == 4:
        family = families[0] if force_straight else _pick(
            rng, families if straight_allowed else families[1:]
        )
    else:
        family = _pick(rng, families)
    return _pick(rng, family)


def _integer(name, value, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} outside legal range")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer("num_train", num_train, 2, 10)
    test = None if num_test is None else _integer("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _canvas(rng, canvas_mode, shared_size, split):
    if canvas_mode == "fixed":
        return shared_size
    if canvas_mode == "shift":
        bounds = (7, 12) if split == "train" else (16, 24)
    else:
        category = int(rng.choice(3, p=(0.22, 0.56, 0.22)))
        bounds = ((6, 10), (10, 18), (18, 31))[category]
    return (int(rng.integers(*bounds)), int(rng.integers(*bounds)))


def _pair_colors(role_pool, mode, index):
    if mode == "stable":
        return role_pool[0], role_pool[1]
    if mode == "source":
        return role_pool[0], role_pool[1 + index % 6]
    if mode == "background":
        return role_pool[1 + index % 6], role_pool[0]
    return role_pool[index % 7], role_pool[(index + 1) % 7]


def _size_list(rng, rows, cols, first_training):
    area = rows * cols
    if area >= 300:
        density = float(rng.uniform(0.105, 0.19))
    else:
        density = float(rng.uniform(0.11, 0.30))
    target = max(3, min(50, int(round(area * density / 3))))
    subset = [2, 3, 4]
    if first_training:
        sizes = [2, 3, 4]
    else:
        if rng.random() < 0.43:
            how_many = int(rng.integers(1, 3))
            subset = [int(x) for x in rng.choice((2, 3, 4), size=how_many, replace=False)]
        sizes = []
    dominant = int(rng.integers(2, 5))
    while len(sizes) < target:
        if dominant in subset and rng.random() < 0.57:
            sizes.append(dominant)
        else:
            sizes.append(_pick(rng, subset))
    rng.shuffle(sizes)
    return sizes


def _fits(candidate, occupied, allow_diagonal):
    for row, col in candidate:
        if (row, col) in occupied:
            return False
        if any((row + dr, col + dc) in occupied for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            return False
        if not allow_diagonal and any(
            (row + dr, col + dc) in occupied for dr in (-1, 1) for dc in (-1, 1)
        ):
            return False
    return True


def _diagonal_touch(first, second):
    first_set = set(first)
    return any((r + dr, c + dc) in first_set
               for r, c in second for dr in (-1, 1) for dc in (-1, 1))


def _locations(shape, rows, cols):
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    return rows - height + 1, cols - width + 1


def _place(rng, specs, rows, cols, allow_diagonal, require_diagonal):
    count = len(specs)
    for _ in range(18):
        occupied = set()
        placed = [None] * count
        remaining = list(range(count))
        if require_diagonal:
            different = [(i, j) for i in remaining for j in remaining
                         if i < j and specs[i][0] != specs[j][0]]
            if not different:
                return None
            first, second = _pick(rng, different)
            first_shape = specs[first][1]
            nrow, ncol = _locations(first_shape, rows, cols)
            if nrow <= 0 or ncol <= 0:
                return None
            top = int(rng.integers(nrow))
            left = int(rng.integers(ncol))
            one = tuple((top + r, left + c) for r, c in first_shape)
            placed[first] = one
            occupied.update(one)
            second_shape = specs[second][1]
            nrow, ncol = _locations(second_shape, rows, cols)
            candidates = []
            for top in range(max(0, nrow)):
                for left in range(max(0, ncol)):
                    two = tuple((top + r, left + c) for r, c in second_shape)
                    if _diagonal_touch(one, two) and _fits(two, occupied, True):
                        candidates.append(two)
            if not candidates:
                continue
            two = _pick(rng, candidates)
            placed[second] = two
            occupied.update(two)
            remaining.remove(first)
            remaining.remove(second)
        rng.shuffle(remaining)
        remaining.sort(key=lambda i: len(specs[i][1]), reverse=True)
        failed = False
        for index in remaining:
            shape = specs[index][1]
            nrow, ncol = _locations(shape, rows, cols)
            if nrow <= 0 or ncol <= 0:
                failed = True
                break
            found = False
            for flat in rng.permutation(nrow * ncol):
                top, left = divmod(int(flat), ncol)
                cells = tuple((top + r, left + c) for r, c in shape)
                if _fits(cells, occupied, allow_diagonal):
                    placed[index] = cells
                    occupied.update(cells)
                    found = True
                    break
            if not found:
                failed = True
                break
        if not failed:
            return placed
    return None


def _make_pair(rng, rows, cols, colors, mapping, first_training,
               straight_allowed, force_straight, allow_diagonal, require_diagonal):
    background, source = colors
    for _ in range(14):
        sizes = _size_list(rng, rows, cols, first_training)
        if force_straight and 4 not in sizes:
            sizes[int(rng.integers(len(sizes)))] = 4
        if require_diagonal and len(set(sizes)) == 1:
            sizes[0] = 2 if sizes[0] != 2 else 3
        forced_index = None
        if force_straight:
            forced_index = _pick(rng, [i for i, size in enumerate(sizes) if size == 4])
        specs = [(size, _shape(rng, size, straight_allowed, i == forced_index))
                 for i, size in enumerate(sizes)]
        while len(specs) >= 3:
            placed = _place(rng, specs, rows, cols, allow_diagonal, require_diagonal)
            if placed is not None:
                inp = np.full((rows, cols), background, dtype=np.int8)
                out = np.full((rows, cols), background, dtype=np.int8)
                for (size, _), cells in zip(specs, placed):
                    for r, c in cells:
                        inp[r, c] = source
                        out[r, c] = mapping[size]
                if np.count_nonzero(inp == source) < inp.size // 2:
                    return {"input": inp, "output": out}
            # Preserve mapping coverage in the first demonstration and any
            # required straight object while easing a crowded small canvas.
            removable = [i for i, (size, shape) in enumerate(specs)
                         if (not first_training or sum(s == size for s, _ in specs) > 1)
                         and (not force_straight or shape not in _FAMILIES[4][0])]
            if not removable:
                break
            specs.pop(_pick(rng, removable))
    raise RuntimeError("bounded placement failed")


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _rows_key(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, low=0)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    labels = [int(v) for v in rng.choice(10, size=3, replace=False)]
    mapping = dict(zip((2, 3, 4), labels))
    role_pool = [int(v) for v in rng.permutation([v for v in range(10) if v not in labels])]
    color_mode = _pick(rng, ("stable", "source", "background", "both"))
    canvas_mode = _pick(rng, ("fixed", "mixed", "mixed", "shift"))
    fixed_category = int(rng.choice(3, p=(0.25, 0.65, 0.10)))
    fixed_bounds = ((7, 10), (10, 17), (18, 23))[fixed_category]
    shared_size = (int(rng.integers(*fixed_bounds)), int(rng.integers(*fixed_bounds)))
    train_straight = bool(rng.integers(2))
    test_straight = bool(rng.integers(2))
    test_straight_index = int(rng.integers(test_count))
    diagonal_mode = bool(rng.integers(2))
    test_diagonal_index = int(rng.integers(test_count))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            index = len(seen)
            colors = _pair_colors(role_pool, color_mode, index)
            size = _canvas(rng, canvas_mode, shared_size, split)
            force_straight = train_straight and split == "train" and pair_index == 0
            force_straight |= test_straight and split == "test" and pair_index == test_straight_index
            require_diagonal = diagonal_mode and (
                split == "train" and pair_index == 0 or
                split == "test" and pair_index == test_diagonal_index
            )
            for _ in range(15):
                pair = _make_pair(rng, *size, colors, mapping,
                                  split == "train" and pair_index == 0,
                                  train_straight if split == "train" else test_straight,
                                  force_straight, diagonal_mode, require_diagonal)
                key = _grid_key(pair["input"])
                if key not in seen and _rows_key(pair["input"]) not in _SOURCE_INPUT_ROWS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = [(row, col)]
            seen[row, col] = True
            for r, c in queue:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            groups.append(tuple(queue))
    return groups


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2
    bg_index = int(np.argmax(counts))
    bg = int(values[bg_index])
    source = int(values[1 - bg_index])
    assert counts[bg_index] > counts[1 - bg_index]
    mask = inp != bg
    assert np.all(out[~mask] == bg)
    groups = _components(mask)
    assert 3 <= len(groups) <= 50
    local_mapping = {}
    owner = {}
    for index, group in enumerate(groups):
        size = len(group)
        assert size in (2, 3, 4)
        labels = {int(out[r, c]) for r, c in group}
        assert len(labels) == 1
        label = labels.pop()
        assert label not in (bg, source)
        if size in local_mapping:
            assert local_mapping[size] == label
        local_mapping[size] = label
        for cell in group:
            owner[cell] = index
    assert len(set(local_mapping.values())) == len(local_mapping)
    diagonal = False
    different_size_diagonal = False
    for (r, c), i in owner.items():
        for dr in (-1, 1):
            for dc in (-1, 1):
                other = owner.get((r + dr, c + dc))
                if other is not None and other != i:
                    diagonal = True
                    if len(groups[other]) != len(groups[i]):
                        different_size_diagonal = True
    return {"background": bg, "source": source, "mapping": local_mapping,
            "groups": groups, "diagonal": diagonal,
            "different_size_diagonal": different_size_diagonal}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    learned = {}
    all_labels = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen and _rows_key(pair["input"]) not in _SOURCE_INPUT_ROWS
            seen.add(key)
            analysis = _analyze(pair)
            analyses[split].append(analysis)
            for size, label in analysis["mapping"].items():
                if size in learned:
                    assert learned[size] == label
                else:
                    learned[size] = label
                all_labels.add(label)
    assert set(learned) == {2, 3, 4} and len(all_labels) == 3
    assert set().union(*(set(a["mapping"]) for a in analyses["train"])) == {2, 3, 4}
    for split in ("train", "test"):
        for a in analyses[split]:
            assert a["background"] not in all_labels
            assert a["source"] not in all_labels
    if any(a["diagonal"] for a in analyses["test"]):
        assert any(a["different_size_diagonal"] for a in analyses["train"])


def _same(first, second):
    return all(len(first[k]) == len(second[k]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[k], second[k])) for k in ("train", "test"))


def _predict_from_training(train, test_input):
    mapping = {}
    for pair in train:
        a = _analyze(pair)
        mapping.update(a["mapping"])
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    answer = np.full_like(test_input, background)
    for group in _components(test_input != background):
        for r, c in group:
            answer[r, c] = mapping[len(group)]
    return answer


def self_test() -> None:
    cases = ((1234, None, None), (8675309, 7, 3), (2000, 2, 1),
             (2001, 10, 1), (2002, 10, 4), (3000, None, 1),
             (3001, None, 4), (3002, 2, None), (3003, 10, None))
    for seed, train, test in cases:
        a = generate_puzzle(seed, train, test)
        b = generate_puzzle(seed, train, test)
        assert _same(a, b)
        validate_generated_puzzle(a)
        for pair in a["test"]:
            assert np.array_equal(_predict_from_training(a["train"], pair["input"]), pair["output"])
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, signatures = set(), set(), set()
    backgrounds, color_pairs, shapes, component_counts, class_mixes = set(), set(), set(), set(), set()
    straight_train = straight_test = diagonal_train = diagonal_test = 0
    missing_class_train = missing_class_test = 0
    fixed_sizes = mixed_sizes = large_pairs = small_pairs = 0
    varied_background = varied_source = varied_count = upward_shift = 0
    straight_transfer = train_only_straight = combined_variation = 0
    low_density = high_density = 0
    companion_families = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        dimensions = set()
        seed_analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                seed_analyses[split].append(a)
                dimensions.add(pair["input"].shape)
                backgrounds.add(a["background"])
                color_pairs.add((a["background"], a["source"]))
                component_counts.add(len(a["groups"]))
                class_mixes.add(tuple(sorted(len(g) for g in a["groups"])))
                small_pairs += min(pair["input"].shape) <= 9
                large_pairs += max(pair["input"].shape) >= 18
                density = sum(len(g) for g in a["groups"]) / pair["input"].size
                low_density += density < 0.14
                high_density += density >= 0.20
                missing = len(a["mapping"]) < 3
                if split == "train":
                    missing_class_train += missing
                    diagonal_train += a["diagonal"]
                else:
                    missing_class_test += missing
                    diagonal_test += a["diagonal"]
                    assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])
                for group in a["groups"]:
                    shapes.add((len(group), _normalize(group)))
                    if len(group) == 4 and (len({r for r, _ in group}) == 1 or len({c for _, c in group}) == 1):
                        if split == "train":
                            straight_train += 1
                        else:
                            straight_test += 1
                if split == "test" and any(
                    len(g) == 4 and _normalize(g) in _FAMILIES[4][0]
                    for g in a["groups"]
                ):
                    for group in a["groups"]:
                        if len(group) == 4:
                            companion_families.update(
                                i for i, family in enumerate(_FAMILIES[4][1:], start=1)
                                if _normalize(group) in family
                            )
        fixed_sizes += len(dimensions) == 1
        mixed_sizes += len(dimensions) > 1
        all_analyses = seed_analyses["train"] + seed_analyses["test"]
        varied_background += len({a["background"] for a in all_analyses}) > 1
        varied_source += len({a["source"] for a in all_analyses}) > 1
        varied_count += len({len(a["groups"]) for a in all_analyses}) > 1
        train_has_straight = any(
            len(g) == 4 and _normalize(g) in _FAMILIES[4][0]
            for a in seed_analyses["train"] for g in a["groups"]
        )
        test_has_straight = any(
            len(g) == 4 and _normalize(g) in _FAMILIES[4][0]
            for a in seed_analyses["test"] for g in a["groups"]
        )
        straight_transfer += not train_has_straight and test_has_straight
        train_only_straight += train_has_straight and not test_has_straight
        train_areas = [p["input"].size for p in puzzle["train"]]
        test_areas = [p["input"].size for p in puzzle["test"]]
        upward_shift += max(train_areas) < min(test_areas)
        combined_variation += (
            len(dimensions) > 1 and
            len({a["background"] for a in all_analyses}) > 1 and
            len({len(a["groups"]) for a in all_analyses}) > 1
        )
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(signatures) == 42
    assert len(backgrounds) >= 5 and len(color_pairs) >= 15
    assert len(shapes) >= 20 and len(component_counts) >= 10
    assert len(class_mixes) >= 100 and len(companion_families) == 4
    assert all((straight_train, straight_test, diagonal_train, diagonal_test,
                missing_class_train, missing_class_test, fixed_sizes, mixed_sizes,
                large_pairs, small_pairs))
    assert min(varied_background, varied_source, varied_count) >= 15
    assert straight_transfer >= 4 and train_only_straight >= 4
    assert upward_shift >= 5 and combined_variation >= 10
    assert low_density >= 15 and high_density >= 15
