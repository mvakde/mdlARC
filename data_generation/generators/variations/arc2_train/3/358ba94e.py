"""Generate count-outlier tile puzzles in the source ARC family."""

import hashlib
import numpy as np


_SOURCE_INPUT_HASHES = frozenset((
    "3afff21bdb82a64e2dd68dfc8443706586bdeb3bb503e402ba9d5a234ec6babd",
    "ad85a02144f4806d96cf176b02af93a62118f31707d5f8328102bb3235b6f397",
    "f0154219cb233f715d04095906e2a8418ebdea9d2bba2052f0269ae77b9cd284",
    "328a1e85d6bf9dbb8a9349856cb61d663aece2baa3185720773d06073259d78e",
    "567b7eddf48059a86f157a0f95981b9d5d96e4233987203bdde25cf415d3e532",
))
_MAX_HOLES = {4: 2, 5: 5, 6: 6, 7: 8}
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check_request(name, value, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")


def _pair_counts(rng, num_train, num_test):
    _check_request("num_train", num_train, 2, 10)
    _check_request("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(num_train), int(num_test)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in _NEIGHBORS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append(cells)
    return result


def _valid_holes(size, holes):
    holes = set(holes)
    if any((r + dr, c + dc) in holes for r, c in holes for dr, dc in ((1, 0), (0, 1))):
        return False
    foreground = np.ones((size, size), dtype=bool)
    for r, c in holes:
        foreground[r, c] = False
    return len(_components(foreground)) == 1


def _holes(rng, size, count):
    if count == 0:
        return ()
    cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    # A shuffled greedy independent set gives separated single-cell cavities.
    for _ in range(120):
        order = rng.permutation(len(cells))
        chosen = []
        for index in order:
            cell = cells[int(index)]
            if all(abs(cell[0] - r) + abs(cell[1] - c) != 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    if _valid_holes(size, chosen):
                        return tuple(sorted(chosen))
                    break
    # The bounded fallback tries subsets of the two checkerboard parities.
    for parity in (0, 1):
        candidates = [cell for cell in cells if sum(cell) % 2 == parity]
        for _ in range(120):
            chosen = [candidates[int(i)] for i in rng.permutation(len(candidates))[:count]]
            if len(chosen) == count and _valid_holes(size, chosen):
                return tuple(sorted(chosen))
    raise AssertionError("no connected tile mask found")


def _layout(rng, size, tile_count, target_shape=None):
    configurations = [
        (rows, cols) for rows in range(1, 5) for cols in range(1, 5)
        if tile_count <= rows * cols <= tile_count + 2
        and rows * size + rows - 1 <= 30
        and cols * size + cols - 1 <= 30
        and (target_shape is None or (rows * size + rows - 1 <= target_shape[0]
                                      and cols * size + cols - 1 <= target_shape[1]))
    ]
    rows, cols = configurations[int(rng.integers(len(configurations)))]
    min_h = rows * size + rows - 1
    min_w = cols * size + cols - 1
    candidates = [(h, w) for h in range(min_h, 31) for w in range(min_w, 31)
                  if h * w > 2 * tile_count * size * size
                  and (target_shape is None or (h, w) == target_shape)]
    candidates.sort(key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))
    fraction = (0.0, 0.36, 0.72, 1.0)
    band = int(rng.integers(0, 3))
    lo = int(len(candidates) * fraction[band])
    hi = max(lo + 1, int(len(candidates) * fraction[band + 1]))
    height, width = candidates[int(rng.integers(lo, hi))]

    def bands(length, count):
        slack = length - count * size - count + 1
        extras = rng.multinomial(slack, np.full(count + 2, 1 / (count + 2)))
        starts = []
        cursor = int(extras[0])
        for i in range(count):
            starts.append((cursor, int(extras[i + 1])))
            cursor += size + 1 + int(extras[i + 1])
        return starts

    row_bands = bands(height, rows)
    col_bands = bands(width, cols)
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    slots = slots[:tile_count]
    orderly = bool(rng.integers(0, 4) == 0)
    row_shifts = [int(rng.integers(extra + 1)) for _, extra in row_bands]
    col_shifts = [int(rng.integers(extra + 1)) for _, extra in col_bands]
    positions = []
    for r, c in slots:
        top, re = row_bands[r]
        left, ce = col_bands[c]
        top += row_shifts[r] if orderly else int(rng.integers(re + 1))
        left += col_shifts[c] if orderly else int(rng.integers(ce + 1))
        positions.append((top, left))
    return height, width, positions


def _make_pair(rng, size, tile_count, common, exceptional, background, colors, multi,
               target_shape=None):
    height, width, positions = _layout(rng, size, tile_count, target_shape)
    exceptional_index = int(rng.integers(tile_count))
    color_by_tile = [colors[0]] * tile_count
    if multi:
        second_count = int(rng.integers(2, tile_count - 1))
        second_indices = rng.permutation(tile_count)[:second_count]
        for index in second_indices:
            color_by_tile[int(index)] = colors[1]
    grid = np.full((height, width), background, dtype=np.int8)
    selected = None
    ordinary_masks = []
    duplicate_style = int(rng.integers(3))  # independent, repeated, or mixed
    for index, (top, left) in enumerate(positions):
        count = exceptional if index == exceptional_index else common
        if index != exceptional_index and ordinary_masks and duplicate_style != 0 and rng.random() < (0.8 if duplicate_style == 1 else 0.4):
            holes = ordinary_masks[int(rng.integers(len(ordinary_masks)))]
            transform = int(rng.integers(3))
            if transform == 1:
                holes = tuple(sorted((size - 1 - c, r) for r, c in holes))
            elif transform == 2:
                holes = tuple(sorted((r, size - 1 - c) for r, c in holes))
        else:
            holes = _holes(rng, size, count)
        if index != exceptional_index:
            ordinary_masks.append(holes)
        tile = np.full((size, size), color_by_tile[index], dtype=np.int8)
        for r, c in holes:
            tile[r, c] = background
        grid[top:top + size, left:left + size] = tile
        if index == exceptional_index:
            selected = tile.copy()
    return {"input": grid, "output": selected}


def _grid_key(grid):
    return (grid.shape, grid.astype(np.uint8).tobytes())


def _source_hash(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest()


def _analyze(pair):
    grid, output = pair["input"], pair["output"]
    values, frequencies = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(frequencies))])
    assert frequencies.max() > sum(frequencies) / 2
    foreground = grid != background
    components = _components(foreground)
    assert 3 <= len(components) <= 8
    records = []
    sizes = set()
    for component in components:
        rows, cols = zip(*component)
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        tile = grid[top:bottom + 1, left:right + 1]
        size = tile.shape[0]
        sizes.add(size)
        assert tile.shape == (size, size) and size in _MAX_HOLES
        color = int(tile[0, 0])
        assert color != background
        assert np.all(tile[0] == color) and np.all(tile[-1] == color)
        assert np.all(tile[:, 0] == color) and np.all(tile[:, -1] == color)
        assert set(int(x) for x in np.unique(tile)) <= {background, color}
        assert int(np.count_nonzero(tile == color)) == len(component)
        holes = tuple((r, c) for r in range(1, size - 1) for c in range(1, size - 1)
                      if tile[r, c] == background)
        assert len(holes) <= _MAX_HOLES[size] and _valid_holes(size, holes)
        records.append((top, left, tile.copy(), holes, color))
    assert len(sizes) == 1
    counts = [len(record[3]) for record in records]
    unique, frequency = np.unique(counts, return_counts=True)
    assert len(unique) == 2 and sorted(int(n) for n in frequency) == [1, len(records) - 1]
    unusual = int(unique[int(np.argmin(frequency))])
    ordinary = int(unique[int(np.argmax(frequency))])
    selected = [record for record in records if len(record[3]) == unusual]
    assert len(selected) == 1 and np.array_equal(output, selected[0][2])
    assert _source_hash(grid) not in _SOURCE_INPUT_HASHES
    return {
        "background": background, "size": next(iter(sizes)), "tile_count": len(records),
        "common": ordinary, "exceptional": unusual,
        "multicolor": len({record[4] for record in records}) > 1,
        "colors": tuple(record[4] for record in records),
        "positions": tuple((record[0], record[1]) for record in records),
        "masks": tuple(record[3] for record in records),
    }


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= dimension <= 30 for dimension in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _assert_grid(pair["input"])
            _assert_grid(pair["output"])
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert not np.array_equal(pair["input"], pair["output"])
            analyses[split].append(_analyze(pair))
    train = analyses["train"]
    assert {int(np.sign(a["exceptional"] - a["common"])) for a in train} == {-1, 1}
    # A numerical role reversal in the demonstrations rules out an absolute count target.
    common_values = {a["common"] for a in train}
    exceptional_values = {a["exceptional"] for a in train}
    assert common_values & exceptional_values
    if any(a["multicolor"] for a in analyses["test"]):
        assert any(a["multicolor"] for a in train)
    if any(a["multicolor"] for a in train):
        assert any(not a["multicolor"] for a in train)
    for split in ("train", "test"):
        for a in analyses[split]:
            if a["multicolor"]:
                colors = a["colors"]
                selected_color = next(color for color, holes in zip(colors, a["masks"])
                                      if len(holes) == a["exceptional"])
                assert colors.count(selected_color) >= 2
                assert all(colors.count(color) >= 2 for color in set(colors))


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[x], q[x]) for p, q in zip(a[s], b[s]) for x in ("input", "output"))
        for s in ("train", "test"))


def _infer_crop_from_input(grid):
    values, frequencies = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(frequencies))])
    remaining = set(zip(*np.nonzero(grid != background)))
    candidates = []
    while remaining:
        origin = remaining.pop()
        cells = {origin}
        frontier = [origin]
        while frontier:
            row, col = frontier.pop()
            for dr, dc in _NEIGHBORS:
                neighbor = (row + dr, col + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.add(neighbor)
                    frontier.append(neighbor)
        rows, cols = zip(*cells)
        crop = grid[min(rows):max(rows) + 1, min(cols):max(cols) + 1]
        count = int(np.count_nonzero(crop[1:-1, 1:-1] == background))
        candidates.append((count, crop))
    counts = [count for count, _ in candidates]
    exceptional = next(count for count in counts if counts.count(count) == 1)
    return next(crop.copy() for count, crop in candidates if count == exceptional)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(rng, num_train, num_test)
    total = train_count + test_count
    # The geometry/count rule is the sole puzzle-level convention. A palette order and
    # the availability of mixed-color examples are sampled once, then visibly taught.
    palette = [int(x) for x in rng.permutation(10)]
    multi_allowed = bool(rng.integers(0, 2) == 0)
    forced_source = int(rng.integers(train_count))
    multi_train = int(rng.integers(train_count - 1))
    if multi_train >= forced_source:
        multi_train += 1
    sizes = []
    tile_counts = []
    for i in range(total):
        size_options = (4, 5, 5, 5, 6, 6, 7) if i < train_count else (4, 5, 5, 6, 6, 7, 7)
        size = int(size_options[int(rng.integers(len(size_options)))])
        if i == forced_source:
            size = 5
        sizes.append(size)
        tile_counts.append(int(rng.integers(3, 9)))
    if multi_allowed:
        tile_counts[multi_train] = max(4, tile_counts[multi_train])
    source_canvas = bool(rng.integers(0, 2) == 0)
    source_shape = (int(rng.integers(14, 18)), int(rng.integers(18, 20)))
    if source_canvas:
        tile_counts[forced_source] = int(rng.integers(4, 6))
    first, second = [int(x) for x in rng.choice(train_count, size=2, replace=False)]
    cap = min(_MAX_HOLES[sizes[first]], _MAX_HOLES[sizes[second]])
    anchor_counts = [int(x) for x in rng.choice(np.arange(cap + 1), size=2, replace=False)]
    counts = []
    for size in sizes:
        allowed = np.arange(_MAX_HOLES[size] + 1)
        c, e = [int(x) for x in rng.choice(allowed, size=2, replace=False)]
        counts.append((c, e))
    counts[first] = tuple(anchor_counts)
    counts[second] = tuple(reversed(anchor_counts))
    backgrounds = [palette[0], palette[1]] + [
        palette[int(rng.integers(10))] for _ in range(total - 2)
    ]
    train_colors = []
    puzzle = {"train": [], "test": []}
    used = set()
    for i in range(total):
        split = "train" if i < train_count else "test"
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        background = backgrounds[i]
        available = [c for c in range(10) if c != background]
        if split == "test":
            reused = [c for c in sorted(set(train_colors)) if c != background]
            novel = [c for c in available if c not in train_colors]
            pool = reused if pair_rng.random() < 0.5 else novel
            if not pool:
                pool = available
            base = pool[int(pair_rng.integers(len(pool)))]
        else:
            base = available[int(pair_rng.integers(len(available)))]
        if split == "train":
            train_colors.append(base)
        secondary = [c for c in available if c != base]
        second_color = secondary[int(pair_rng.integers(len(secondary)))]
        multi = (multi_allowed and i == multi_train) if split == "train" else (
            multi_allowed and i != forced_source and tile_counts[i] >= 4 and pair_rng.random() < 0.55)
        for _ in range(12):
            pair = _make_pair(pair_rng, sizes[i], tile_counts[i], *counts[i], background,
                              (base, second_color), multi,
                              source_shape if source_canvas and i == forced_source else None)
            key = _grid_key(pair["input"])
            if key not in used and _source_hash(pair["input"]) not in _SOURCE_INPUT_HASHES:
                break
        else:
            raise AssertionError("failed to construct unique input")
        used.add(key)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _same_puzzle(generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3))
    assert _same_puzzle(generate_puzzle(456), generate_puzzle(456))
    assert not _same_puzzle(generate_puzzle(1), generate_puzzle(2))
    analyses = []
    within_backgrounds = []
    observed = []
    height_shifts = set()
    test_color_reuse = set()
    test_color_new = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pa = []
        train_analyses = [_analyze(pair) for pair in puzzle["train"]]
        train_colors = {color for a in train_analyses for color in a["colors"]}
        train_median_height = float(np.median([pair["input"].shape[0] for pair in puzzle["train"]]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                assert np.array_equal(_infer_crop_from_input(pair["input"]), pair["output"])
                pa.append(a)
                observed.append((split, pair, a))
                if split == "test":
                    height_shifts.add(int(np.sign(pair["input"].shape[0] - train_median_height)))
                    test_color_reuse.add(any(color in train_colors for color in a["colors"]))
                    test_color_new.add(any(color not in train_colors for color in a["colors"]))
        analyses += pa
        within_backgrounds.append(len({a["background"] for a in pa}))
    assert min(within_backgrounds) >= 2
    assert {a["size"] for a in analyses} == {4, 5, 6, 7}
    assert {a["tile_count"] for a in analyses} == set(range(3, 9))
    assert {a["background"] for a in analyses} == set(range(10))
    assert {a["common"] for a in analyses} >= {1, 2, 3, 4}
    assert {a["exceptional"] for a in analyses} >= {1, 2, 3, 4}
    assert any(a["multicolor"] for a in analyses)
    assert any(len(set(a["masks"])) < a["tile_count"] for a in analyses)
    assert len({a["positions"] for a in analyses}) >= 100
    assert len({(a["size"], a["common"], a["exceptional"]) for a in analyses}) >= 35
    assert any(a["multicolor"] and split == "test" for split, _, a in observed)
    assert any(a["common"] == 1 and a["exceptional"] == 2 for a in analyses)
    assert any(a["common"] == 4 and a["exceptional"] == 2 for a in analyses)
    assert any(a["common"] == 2 and a["exceptional"] == 4 for a in analyses)
    assert any(a["common"] == 0 or a["exceptional"] == 0 for a in analyses)
    assert any(len({r for r, _ in a["positions"]}) > 2 and
               len({c for _, c in a["positions"]}) > 2 for a in analyses)
    assert any(a["multicolor"] and a["size"] >= 6 and
               len({r for r, _ in a["positions"]}) > 2 for a in analyses)
    assert any(a["size"] == 5 and not a["multicolor"] and a["tile_count"] in (4, 5)
               and 14 <= pair["input"].shape[0] <= 17
               and 18 <= pair["input"].shape[1] <= 19
               for _, pair, a in observed)
    assert height_shifts == {-1, 0, 1}
    assert test_color_reuse == {False, True} and test_color_new == {False, True}
    assert min(pair["input"].shape[0] for _, pair, _ in observed) < 14
    assert max(pair["input"].shape[0] for _, pair, _ in observed) == 30
    assert max(pair["input"].shape[1] for _, pair, _ in observed) == 30
    for seed, nt, ne in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0},
                   {"num_train": True}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
