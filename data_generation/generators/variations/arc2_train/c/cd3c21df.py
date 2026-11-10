"""Generate ARC puzzles whose answer is the lone unpaired colored object."""

import numpy as np


_SOURCE_PAIRS = (
    (("000000000000", "008200006660", "002800000000", "000000000000", "000000000000", "000000008200", "666000002800", "000000000000", "000077000000", "000077000000", "000000000000", "000000000000"), ("77", "77")),
    (("00000000000", "00500000000", "00500004000", "00500004020", "00500004020", "00500000020", "00005000020", "00005000000", "40005000000", "40005000000", "40005000000"), ("2", "2", "2", "2")),
    (("0000000000000", "0066600000000", "0006000022200", "0000000023200", "0000000000000", "0000000002120", "0000000001110", "0222000000000", "0232000000000", "0000006660000", "0000000600000", "0000000000000", "0000000000000"), ("212", "111")),
    (("0000000000000", "0555000000000", "0000000480000", "0000000880000", "0440000840000", "0440000000000", "0000000000000", "0000000000000", "0044000000000", "0044000000000", "0000000005550", "0000000000000", "0000000000000"), ("48", "88", "84")),
)


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be between 1 and 4")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train")
    num_test = _checked_count(num_test, "num_test")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _key(array):
    return (array.shape, tuple(int(x) for x in array.flat))


def _mask_key(pattern, background):
    return _key(pattern != background)


def _crop_mask(mask):
    rows, cols = np.where(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _mask(rng, kind):
    if kind == "bar":
        length = int(rng.integers(2, 7))
        return np.ones((1, length) if int(rng.integers(2)) else (length, 1), dtype=bool)
    if kind == "rectangle":
        return np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
    if kind == "hollow":
        height, width = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        result = np.ones((height, width), dtype=bool)
        result[1:-1, 1:-1] = False
        return result
    if kind == "bend":
        height, width = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        result = np.zeros((height, width), dtype=bool)
        result[:, 0] = True
        result[-1, :] = True
        if int(rng.integers(2)):
            result = np.fliplr(result)
        if int(rng.integers(2)):
            result = np.flipud(result)
        return result
    if kind == "irregular":
        height, width = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        cells = {(int(rng.integers(height)), int(rng.integers(width)))}
        wanted = int(rng.integers(5, min(12, height * width) + 1))
        for _ in range(4 * wanted):
            if len(cells) >= wanted:
                break
            r, c = tuple(cells)[int(rng.integers(len(cells)))]
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            if 0 <= r + dr < height and 0 <= c + dc < width:
                cells.add((r + dr, c + dc))
        result = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            result[r, c] = True
        return _crop_mask(result)
    raise ValueError("unknown mask kind")


def _pattern(rng, mask, background, palette, color_count):
    color_count = min(color_count, int(mask.sum()), len(palette))
    colors = rng.choice(palette, size=color_count, replace=False)
    cells = int(mask.sum())
    values = np.empty(cells, dtype=np.int8)
    values[:color_count] = colors
    if cells > color_count:
        values[color_count:] = rng.choice(colors, size=cells - color_count)
    rng.shuffle(values)
    result = np.full(mask.shape, background, dtype=np.int8)
    result[mask] = values
    return result


def _twin(rng, target, background, palette):
    mask = target != background
    colors = np.unique(target[mask])
    for _ in range(24):
        if len(colors) > 1 and int(rng.integers(3)):
            values = target[mask].copy()
            rng.shuffle(values)
            if int(rng.integers(3)) == 0:
                values[int(rng.integers(len(values)))] = int(rng.choice(palette))
            result = target.copy()
            result[mask] = values
        else:
            count = int(rng.integers(1, min(4, len(palette), int(mask.sum())) + 1))
            result = _pattern(rng, mask, background, palette, count)
        if not np.array_equal(result, target):
            return result
    result = target.copy()
    first = int(np.flatnonzero(mask)[0])
    alternative = next(int(c) for c in palette if c != result.flat[first])
    result.flat[first] = alternative
    return result


def _new_decoy(rng, background, palette, forbidden, target_mask, mode, index):
    kinds = ("bar", "rectangle", "bend", "hollow", "irregular")
    for _ in range(160):
        if mode == "multiple" and index < 2:
            mask = target_mask.copy()
        else:
            mask = _mask(rng, kinds[int(rng.integers(len(kinds)))])
        if mode == "unique" and _key(mask) == _key(target_mask):
            continue
        count = int(rng.choice((1, 1, 1, 2, 2, 3, 4)))
        candidate = _pattern(rng, mask, background, palette, count)
        if _key(candidate) not in forbidden:
            return candidate
    raise RuntimeError("could not make a distinct decoy")


def _gap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 <= br0 or br1 + 1 <= ar0 or ac1 + 1 <= bc0 or bc1 + 1 <= ac0


def _place(rng, patterns, background, height, width):
    for _ in range(32):
        grid = np.full((height, width), background, dtype=np.int8)
        boxes = []
        for index in rng.permutation(len(patterns)):
            pattern = patterns[int(index)]
            ph, pw = pattern.shape
            if ph > height or pw > width:
                break
            positions_per_row = width - pw + 1
            candidates = (height - ph + 1) * positions_per_row
            chosen = None
            for location in rng.permutation(candidates):
                row, col = divmod(int(location), positions_per_row)
                box = (row, col, row + ph, col + pw)
                if all(_gap(box, other) for other in boxes):
                    chosen = box
                    break
            if chosen is None:
                break
            row, col, _, _ = chosen
            view = grid[row:row + ph, col:col + pw]
            mask = pattern != background
            view[mask] = pattern[mask]
            boxes.append(chosen)
        else:
            return grid
    return None


def _canvas(rng, families):
    regime = int(rng.choice((0, 0, 1, 1, 1, 2, 2, 3)))
    if regime == 0:
        height, width = int(rng.integers(11, 16)), int(rng.integers(11, 16))
    elif regime == 1:
        height, width = int(rng.integers(16, 23)), int(rng.integers(16, 23))
    elif regime == 2:
        height, width = int(rng.integers(23, 31)), int(rng.integers(23, 31))
    else:
        height, width = int(rng.integers(11, 17)), int(rng.integers(23, 31))
        if int(rng.integers(2)):
            height, width = width, height
    if families >= 3:
        height = max(height, 16)
        width = max(width, 16)
    return height, width


def _generate_pair(rng, background, families, target_type, mode):
    other_colors = np.array([c for c in range(10) if c != background], dtype=np.int8)
    palette = rng.choice(other_colors, size=int(rng.integers(2, 7)), replace=False)
    if target_type == "solid":
        target_kind = ("bar", "rectangle")[int(rng.integers(2))]
        colors = 1
    elif target_type == "colored":
        target_kind = ("rectangle", "bar")[int(rng.integers(2))]
        colors = int(rng.integers(2, 4))
    elif target_type in ("hollow", "colored_hollow"):
        target_kind = "hollow"
        colors = int(rng.integers(2 if target_type == "colored_hollow" else 1, 4))
    else:
        target_kind = ("bend", "irregular")[int(rng.integers(2))]
        colors = int(rng.integers(1, 4))
    target = _pattern(rng, _mask(rng, target_kind), background, palette, colors)
    forbidden = {_key(target)}
    decoys = []
    if mode in ("overlap", "multiple"):
        twin = _twin(rng, target, background, palette)
        decoys.append(twin)
        forbidden.add(_key(twin))
    while len(decoys) < families:
        decoy = _new_decoy(rng, background, palette, forbidden, target != background, mode, len(decoys))
        decoys.append(decoy)
        forbidden.add(_key(decoy))
    patterns = [target] + [item for decoy in decoys for item in (decoy, decoy.copy())]
    height, width = _canvas(rng, families)
    for growth in range(0, 19):
        grid = _place(rng, patterns, background, min(30, height + growth), min(30, width + growth))
        if grid is not None:
            return {"input": grid, "output": target.copy()}
    raise RuntimeError("could not place objects")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    anchor_background = int(rng.integers(10))
    shared_background = bool(rng.integers(3) == 0)
    # The uniform background is visible in each input. Its value may be stable
    # across a puzzle or vary by pair; it never selects the answer by itself.
    train_specs = [("solid", "overlap", 1), ("colored", "unique", 2)]
    test_types = [str(rng.choice(("solid", "colored", "colored", "hollow", "irregular"))) for _ in range(test_count)]
    if any(typ in ("hollow", "irregular") for typ in test_types) or int(rng.integers(4)) == 0:
        train_specs[1] = ("colored_hollow", "unique", 2)
    for _ in range(train_count - 2):
        typ = str(rng.choice(("solid", "colored", "hollow", "irregular")))
        mode = str(rng.choice(("overlap", "unique", "multiple")))
        family_count = int(rng.integers(1, 5))
        if mode == "multiple" and family_count < 2:
            mode = "overlap"
        train_specs.append((typ, mode, family_count))
    rng.shuffle(train_specs)
    train, test = [], []
    seen = set()
    for index, (typ, mode, families) in enumerate(train_specs):
        if shared_background or index == 0:
            background = anchor_background
        elif index == 1:
            background = int(rng.choice([color for color in range(10) if color != anchor_background]))
        else:
            background = int(rng.integers(10))
        for _ in range(12):
            pair = _generate_pair(rng, background, families, typ, mode)
            if _key(pair["input"]) not in seen:
                train.append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise RuntimeError("duplicate training input")
    for typ in test_types:
        background = anchor_background if shared_background else int(rng.integers(10))
        families = int(rng.integers(1, 5))
        mode = str(rng.choice(("overlap", "unique", "multiple")))
        if mode == "multiple" and families < 2:
            mode = "overlap"
        for _ in range(12):
            pair = _generate_pair(rng, background, families, typ, mode)
            if _key(pair["input"]) not in seen:
                test.append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            raise RuntimeError("duplicate test input")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(grid):
    counts = np.bincount(grid.flat, minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(int(counts[c]) for c in range(10) if c != background)
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    records = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or grid[row, col] == background:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and grid[nr, nc] != background:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs, cs = zip(*cells)
            box = (min(rs), min(cs), max(rs) + 1, max(cs) + 1)
            r0, c0, r1, c1 = box
            crop = grid[r0:r1, c0:c1].copy()
            assert int(np.count_nonzero(crop != background)) == len(cells)
            records.append((box, crop))
    for i, (box, _) in enumerate(records):
        for other, _ in records[i + 1:]:
            assert _gap(box, other)
    return background, records


def _features(pair):
    background, records = _analyze(pair["input"])
    groups = {}
    for _, crop in records:
        groups.setdefault(_key(crop), []).append(crop)
    sizes = sorted(map(len, groups.values()))
    assert sizes[0] == 1 and all(size == 2 for size in sizes[1:])
    assert 2 <= len(groups) <= 5
    singleton = next(crops[0] for crops in groups.values() if len(crops) == 1)
    assert np.array_equal(pair["output"], singleton)
    silhouette = _mask_key(singleton, background)
    overlap = sum(_mask_key(crops[0], background) == silhouette for crops in groups.values()) - 1
    enclosed = (min(singleton.shape) >= 3 and bool(np.any(singleton[1:-1, 1:-1] == background))
                and bool(np.all(singleton[0] != background)) and bool(np.all(singleton[-1] != background))
                and bool(np.all(singleton[:, 0] != background)) and bool(np.all(singleton[:, -1] != background)))
    return background, len(groups) - 1, overlap, len(np.unique(singleton[singleton != background])), bool(np.any(singleton == background)), singleton.shape, enclosed


def _strings(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    all_features = []
    for split in ("train", "test"):
        features = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert not (inp.shape == out.shape and np.array_equal(inp, out))
            assert (_strings(inp), _strings(out)) not in _SOURCE_PAIRS
            features.append(_features(pair))
        all_features.append(features)
    train, test = all_features
    assert all(0 <= feature[0] <= 9 for feature in train + test)
    assert {1, 2}.issubset({feature[1] for feature in train})
    assert any(feature[2] >= 1 for feature in train)
    assert any(feature[2] == 0 for feature in train)
    assert any(feature[3] == 1 and not feature[4] for feature in train)
    assert any(feature[3] >= 2 for feature in train)
    if any(feature[4] for feature in test):
        assert any(feature[4] for feature in train)
    if any(feature[6] for feature in test):
        assert any(feature[6] for feature in train)


def _equal_puzzles(first, second):
    return all(len(first[split]) == len(second[split]) and all(np.array_equal(a[key], b[key]) for a, b in zip(first[split], second[split]) for key in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(5000 + index, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.0, 1)):
        try:
            generate_puzzle(5000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    train_counts, test_counts, backgrounds, sizes = set(), set(), set(), set()
    family_counts, target_shapes, overlaps, color_counts = set(), set(), set(), set()
    hollow = 0
    compact = large = elongated = 0
    shared_backgrounds = varied_backgrounds = 0
    tests_add = tests_remove = 0
    reused_color = distinct_color = 0
    near_change = far_change = 0
    decoy_color_counts = set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        puzzle_backgrounds = set()
        train_families = []
        test_families = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                background, family, overlap, colors, has_hole, shape, enclosed = _features(pair)
                puzzle_backgrounds.add(background)
                (train_families if split == "train" else test_families).append(family)
                backgrounds.add(background)
                family_counts.add(family)
                target_shapes.add(shape)
                overlaps.add(overlap)
                color_counts.add(colors)
                hollow += int(enclosed)
                h, w = pair["input"].shape
                sizes.add((h, w))
                compact += int(max(h, w) <= 15)
                large += int(max(h, w) >= 24)
                elongated += int(max(h, w) - min(h, w) >= 8)
                _, records = _analyze(pair["input"])
                pattern_set = {_key(crop): crop for _, crop in records}
                target = pair["output"]
                mask = target != background
                target_colors = set(np.unique(target[mask]))
                decoys = [crop for crop in pattern_set.values() if not np.array_equal(crop, target)]
                if any(target_colors & set(np.unique(crop[crop != background])) for crop in decoys):
                    reused_color += 1
                else:
                    distinct_color += 1
                for crop in decoys:
                    decoy_color_counts.add(len(np.unique(crop[crop != background])))
                    if crop.shape == target.shape and np.array_equal(crop != background, mask) and colors > 1:
                        differences = int(np.count_nonzero(crop[mask] != target[mask]))
                        near_change += int(differences <= 2)
                        far_change += int(differences >= 4)
        shared_backgrounds += int(len(puzzle_backgrounds) == 1)
        varied_backgrounds += int(len(puzzle_backgrounds) > 1)
        assert len(set(train_families)) > 1
        tests_add += int(any(t > r for t in test_families for r in train_families))
        tests_remove += int(any(t < r for t in test_families for r in train_families))
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 6 and len(sizes) >= 40
    assert {1, 2, 3, 4}.issubset(family_counts)
    assert len(target_shapes) >= 12 and {0, 1, 2}.issubset(overlaps)
    assert {1, 2, 3}.issubset(color_counts)
    assert hollow >= 10 and compact >= 5 and large >= 20 and elongated >= 10
    assert shared_backgrounds >= 3 and varied_backgrounds >= 10
    assert tests_add >= 5 and tests_remove >= 5
    assert reused_color >= 100 and distinct_color >= 10
    assert {1, 2, 3, 4}.issubset(decoy_color_counts)
    assert near_change >= 10 and far_change >= 10
    assert _equal_puzzles(generate_puzzle(17), generate_puzzle(17))
    assert not _equal_puzzles(generate_puzzle(17), generate_puzzle(18))
