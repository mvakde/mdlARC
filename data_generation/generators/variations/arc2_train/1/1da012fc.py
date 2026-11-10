"""Generators and grid-based checks for ordered palette recoloring puzzles."""

from itertools import combinations
import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = {
    "befc1d197ef80c07dd98fd0808eecad16c11113efc3ff0fcb7aef625f43fd100",
    "7545bd7940de4c814186186b4b0a6f6bf8dd410c5b2238b437f1f4579e914b6a",
    "5d42c36e7fb966a13666025ff7d24ecbe6f3b25a91a0c5e00f6c4f55a0412c50",
}
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))
_NEIGHBORS_8 = tuple((dr, dc) for dr in (-1, 0, 1)
                     for dc in (-1, 0, 1) if dr or dc)
_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _check_count(num_train, "num_train", 2, 10)
    num_test = _check_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        choices = list(range(num_test + 1, 11))
        num_train = choices[seed % len(choices)]
    if num_test is None:
        choices = list(range(1, min(4, num_train - 1) + 1))
        num_test = choices[seed % len(choices)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _compositions(total, maximum_part=3, maximum_bands=5):
    if total == 0:
        yield ()
        return
    if maximum_bands == 0:
        return
    for first in range(1, min(maximum_part, total) + 1):
        for rest in _compositions(total - first, maximum_part, maximum_bands - 1):
            yield (first,) + rest


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _shape(rng, size_class):
    """Mix four- and eight-connected growth with rings and branches."""
    mode = int(rng.integers(0, 8))
    if mode == 0:
        height = int(rng.integers(3, 5))
        width = int(rng.integers(3, 5))
        cells = [(r, c) for r in range(height) for c in range(width)
                 if r in (0, height - 1) or c in (0, width - 1)]
    elif mode == 1:
        # A source-like silhouette whose arms attach diagonally.
        cells = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 4), (2, 2)]
        if bool(rng.integers(0, 2)):
            cells.extend([(2, 1), (2, 3)])
    else:
        if size_class == "small":
            target = int(rng.integers(3, 8))
        elif size_class == "large":
            target = int(rng.integers(11, 19))
        else:
            target = int(rng.integers(5, 16))
        height = int(rng.integers(2, 6))
        width = int(rng.integers(2, 8))
        if height * width < target:
            height = min(6, max(height, (target + width - 1) // width))
        if height * width < target:
            width = min(7, max(width, (target + height - 1) // height))
        target = min(target, height * width)
        occupied = {(int(rng.integers(height)), int(rng.integers(width)))}
        neighbors = _NEIGHBORS_8 if mode in (2, 3, 4) else _NEIGHBORS_4
        while len(occupied) < target:
            frontier = {(r + dr, c + dc) for r, c in occupied
                        for dr, dc in neighbors
                        if 0 <= r + dr < height and 0 <= c + dc < width
                        and (r + dr, c + dc) not in occupied}
            chosen = sorted(frontier)[int(rng.integers(len(frontier)))]
            occupied.add(chosen)
        cells = list(occupied)
    cells = _normalize(cells)
    if bool(rng.integers(0, 2)):
        width = max(c for _, c in cells) + 1
        cells = [(r, width - 1 - c) for r, c in cells]
    rotations = int(rng.integers(0, 4))
    for _ in range(rotations):
        height = max(r for r, _ in cells) + 1
        cells = [(c, height - 1 - r) for r, c in cells]
    return _normalize(cells)


def _palette_origin(rng, height, width, ph, pw):
    row_max, col_max = height - ph, width - pw
    mode = int(rng.integers(0, 9))
    if mode < 4:
        return ((0 if mode < 2 else row_max),
                (0 if mode % 2 == 0 else col_max))
    if mode == 4:
        return 0, int(rng.integers(col_max + 1))
    if mode == 5:
        return row_max, int(rng.integers(col_max + 1))
    if mode == 6:
        return int(rng.integers(row_max + 1)), 0
    if mode == 7:
        return int(rng.integers(row_max + 1)), col_max
    return int(rng.integers(2, row_max - 1)), int(rng.integers(2, col_max - 1))


def _place_objects(grid, groups, band_tops, rng):
    height, width = grid.shape
    placed = []
    assigned_slots = []
    centers_by_slot = []
    for shapes, top in zip(groups, band_tops):
        count = len(shapes)
        options = [slots for slots in combinations(range(5), count)
                   if all(b - a >= 2 for a, b in zip(slots, slots[1:]))]
        slots = options[int(rng.integers(len(options)))]
        band = []
        for shape, slot in zip(shapes, slots):
            shape_width = max(c for _, c in shape) + 1
            anchor = (slot + 0.5) * width / 5
            candidates = list(range(width - shape_width + 1))
            rng.shuffle(candidates)
            candidates.sort(key=lambda left: abs(left + shape_width / 2 - anchor))
            selected = None
            for left in candidates:
                if abs(left + shape_width / 2 - anchor) > max(5, width / 5):
                    continue
                center = left + shape_width / 2
                if any((slot < other_slot and center >= other_center)
                       or (slot > other_slot and center <= other_center)
                       for other_slot, other_center in centers_by_slot):
                    continue
                cells = [(top + r, left + c) for r, c in shape]
                if any(r >= height or grid[r, c] for r, c in cells):
                    continue
                if any(np.any(grid[max(0, r - 1):min(height, r + 2),
                                   max(0, c - 1):min(width, c + 2)])
                       for r, c in cells):
                    continue
                selected = cells
                break
            if selected is None:
                return None
            for r, c in selected:
                grid[r, c] = 1
            band.append(selected)
            centers_by_slot.append((slot, left + shape_width / 2))
        if any(min(c for _, c in a) >= min(c for _, c in b)
               for a, b in zip(band, band[1:])):
            return None
        placed.extend(band)
        assigned_slots.append(slots)
    return placed, assigned_slots


def _label_locations(rng, band_counts, ph, pw, assigned_slots):
    possibilities = [rows for rows in combinations(range(ph), len(band_counts))
                     if all(b - a >= 2 for a, b in zip(rows, rows[1:]))]
    rows = possibilities[int(rng.integers(len(possibilities)))]
    border_columns = bool(rng.integers(0, 3) == 0)
    low = 0 if border_columns else 1
    high = pw - 1 if border_columns else pw - 2
    slots_to_columns = [round(low + slot * (high - low) / 4) for slot in range(5)]
    positions = []
    for row, slots in zip(rows, assigned_slots):
        columns = [slots_to_columns[slot] for slot in slots]
        if any(b - a < 2 for a, b in zip(columns, columns[1:])):
            return None
        positions.extend((row, c) for c in columns)
    return positions


def _draw_pair(rng, n, bands, roles):
    background, carrier, placeholder = roles
    size_classes = ["mix"] * n
    if n >= 2:
        small_index, large_index = (int(x) for x in rng.choice(n, 2, replace=False))
        size_classes[small_index] = "small"
        size_classes[large_index] = "large"
    shapes = []
    for index in range(n):
        shapes.append(_shape(rng, size_classes[index]))
    groups, at = [], 0
    for count in bands:
        groups.append(shapes[at:at + count])
        at += count
    band_heights = [max(max(r for r, _ in shape) + 1 for shape in group)
                    for group in groups]
    needed_height = sum(band_heights) + 2 * (len(bands) - 1)
    minimum_height = max(13 if n <= 2 else 20, needed_height + 1)
    if minimum_height > 30:
        return None
    height = int(rng.integers(minimum_height, 31 if n >= 4 else min(31, minimum_height + 10)))
    minimum_width = 17 if n <= 2 else 22 if max(bands) <= 2 else 27
    width = int(rng.integers(minimum_width, 31))
    ph_min = max(5, 2 * len(bands) + 1)
    pw_min = max(5, 2 * max(bands) + 2)
    if ph_min >= height or pw_min >= width:
        return None
    ph = int(rng.integers(ph_min, min(12, height - 1) + 1))
    pw = int(rng.integers(pw_min, min(11, width - 1) + 1))
    if height - ph < 4 or width - pw < 4:
        return None
    palette_r, palette_c = _palette_origin(rng, height, width, ph, pw)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    occupancy[palette_r:palette_r + ph, palette_c:palette_c + pw] = 1
    slack = height - needed_height
    allocations = rng.multinomial(slack, [1 / (len(bands) + 1)] * (len(bands) + 1))
    top = int(allocations[0])
    band_tops = []
    for index, band_height in enumerate(band_heights):
        band_tops.append(top)
        top += band_height + 2 + int(allocations[index + 1])
    result = _place_objects(occupancy, groups, band_tops, rng)
    if result is None:
        return None
    objects, slots = result
    locations = _label_locations(rng, bands, ph, pw, slots)
    if locations is None:
        return None
    eligible = [c for c in range(10) if c not in roles]
    labels = [int(c) for c in rng.choice(eligible, n, replace=n > len(eligible))]
    if n >= 2 and rng.integers(0, 4) == 0:
        labels[-1] = labels[0]
    rng.shuffle(labels)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[palette_r:palette_r + ph, palette_c:palette_c + pw] = carrier
    for (r, c), color in zip(locations, labels):
        inp[palette_r + r, palette_c + c] = color
    for cells in objects:
        for r, c in cells:
            inp[r, c] = placeholder
    out = inp.copy()
    for cells, color in zip(objects, labels):
        for r, c in cells:
            out[r, c] = color
    return inp, out


def _components(mask, neighbors=_NEIGHBORS_8):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = remaining.pop()
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    cells.append(point)
        result.append(sorted(cells))
    return result


def _band_counts(rows):
    counts = []
    last = None
    for row in rows:
        if row != last:
            counts.append(1)
            last = row
        else:
            counts[-1] += 1
    return tuple(counts)


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    colors, amounts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(amounts))])
    candidates = []
    for carrier in (int(c) for c in colors if c != background):
        positions = np.argwhere(inp == carrier)
        r0, c0 = positions.min(axis=0)
        r1, c1 = positions.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        ph, pw = r1 - r0 + 1, c1 - c0 + 1
        if ph < 5 or pw < 5 or ph * pw < 25:
            continue
        rectangle = inp[r0:r1 + 1, c0:c1 + 1]
        if np.any(rectangle == background):
            continue
        if np.count_nonzero(rectangle == carrier) / rectangle.size < 0.70:
            continue
        local_labels = np.argwhere(rectangle != carrier)
        n = len(local_labels)
        if not 1 <= n <= 6:
            continue
        label_entries = sorted((r0 + int(r), c0 + int(c), int(rectangle[r, c]))
                               for r, c in local_labels)
        if any(abs(ra - rb) <= 1 and abs(ca - cb) <= 1
               for (ra, ca, _), (rb, cb, _) in combinations(label_entries, 2)):
            continue
        palette_mask = np.zeros(inp.shape, dtype=bool)
        palette_mask[r0:r1 + 1, c0:c1 + 1] = True
        outside = inp[~palette_mask]
        foreign = set(map(int, np.unique(outside))) - {background}
        if len(foreign) != 1:
            continue
        placeholder = foreign.pop()
        if placeholder in {carrier, background} or placeholder in {c for _, _, c in label_entries}:
            continue
        components = _components((inp == placeholder) & ~palette_mask)
        if len(components) != n or any(len(comp) < 3 for comp in components):
            continue
        components.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))
        object_rows = [min(r for r, _ in comp) for comp in components]
        label_rows = [r for r, _, _ in label_entries]
        object_bands = _band_counts(object_rows)
        label_bands = _band_counts(label_rows)
        if object_bands != label_bands or max(object_bands) > 3:
            continue
        # The key is a compressed spatial map: labels in different rows
        # cannot reverse a clear left-to-right relationship of objects.
        centers = [(min(c for _, c in comp) + max(c for _, c in comp)) / 2
                   for comp in components]
        if any((label_entries[i][1] - label_entries[j][1])
               * (centers[i] - centers[j]) < 0
               for i, j in combinations(range(n), 2)):
            continue
        if any(b - a < 2 for a, b in zip(sorted(set(label_rows)), sorted(set(label_rows))[1:])):
            continue
        expected = inp.copy()
        for comp, (_, _, color) in zip(components, label_entries):
            for r, c in comp:
                expected[r, c] = color
        if not np.array_equal(expected, out) or np.array_equal(inp, out):
            continue
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1
               for comp in components for r, c in comp
               for rr in range(max(0, r0 - 1), min(inp.shape[0], r1 + 2))
               for cc in range(max(0, c0 - 1), min(inp.shape[1], c1 + 2))
               if palette_mask[rr, cc]):
            continue
        candidates.append({
            "background": background, "carrier": carrier, "placeholder": placeholder,
            "n": n, "bands": object_bands, "labels": tuple(c for _, _, c in label_entries),
            "palette_bbox": (r0, c0, r1, c1), "palette_shape": (ph, pw),
            "edge_label": any(r in (r0, r1) or c in (c0, c1)
                              for r, c, _ in label_entries),
            "object_sizes": tuple(len(comp) for comp in components),
            "object_boxes": tuple((max(r for r, _ in comp) - min(r for r, _ in comp) + 1,
                                   max(c for _, c in comp) - min(c for _, c in comp) + 1)
                                  for comp in components),
            "diagonal_only": any(len(_components(
                np.array([[((r, c) in set(comp)) for c in range(inp.shape[1])]
                          for r in range(inp.shape[0])], dtype=bool), _NEIGHBORS_4)) > 1
                                 for comp in components),
        })
    assert len(candidates) == 1
    return candidates[0]


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_hash(inp, out):
    a = np.asarray(inp, dtype=np.uint8)
    b = np.asarray(out, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_key = _grid_key(inp)
            pair_key = input_key, _grid_key(out)
            assert input_key not in inputs and pair_key not in pairs
            assert _source_hash(inp, out) not in _SOURCE_PAIR_HASHES
            inputs.add(input_key)
            pairs.add(pair_key)
            analyses[split].append(_analyze_pair(inp, out))
    # Every puzzle demonstrates the direction and within-row convention before test.
    assert any(max(a["bands"]) >= 2 for a in analyses["train"])
    assert all(a["n"] <= 6 for a in analyses["test"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Spatial reading order and an optional color-role policy are puzzle-wide.
    base_roles = tuple(int(c) for c in rng.permutation(10)[:3])
    consistent_colors = bool(rng.integers(0, 5) < 2)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for pair_index in range(count):
            for _ in range(240):
                child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                if split == "train" and pair_index == 0:
                    n = int(child_rng.integers(3, 6))
                    options = [c for c in _compositions(n) if max(c) >= 2 and len(c) >= 2]
                else:
                    n = int(child_rng.integers(1, 7))
                    if split == "test" and child_rng.integers(0, 5) < 3:
                        n = int(child_rng.integers(4, 7))
                    options = list(_compositions(n))
                bands = options[int(child_rng.integers(len(options)))]
                roles = base_roles if consistent_colors else tuple(
                    int(c) for c in child_rng.permutation(10)[:3])
                drawn = _draw_pair(child_rng, n, bands, roles)
                if drawn is None:
                    continue
                inp, out = drawn
                key = _grid_key(inp)
                if key in seen or _source_hash(inp, out) in _SOURCE_PAIR_HASHES:
                    continue
                pair = {"input": inp, "output": out}
                try:
                    _analyze_pair(inp, out)
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f"unable to place {split} pair {pair_index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, bands, palette_positions, roles = set(), set(), set(), set()
    object_counts, object_sizes, object_boxes = set(), set(), set()
    edge_labels = diagonal = repeated = 0
    within_size_variation = within_roles_variation = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        per_sizes, per_roles = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair["input"], pair["output"])
                shapes.add(pair["input"].shape)
                bands.add(a["bands"])
                object_counts.add(a["n"])
                object_sizes.update(a["object_sizes"])
                object_boxes.update(a["object_boxes"])
                edge_labels += a["edge_label"]
                diagonal += a["diagonal_only"]
                repeated += len(set(a["labels"])) < len(a["labels"])
                r0, c0, r1, c1 = a["palette_bbox"]
                palette_positions.add((r0 == 0, c0 == 0,
                                       r1 == pair["input"].shape[0] - 1,
                                       c1 == pair["input"].shape[1] - 1))
                roles.add((a["background"], a["carrier"], a["placeholder"]))
                per_sizes.add(pair["input"].shape)
                per_roles.add((a["background"], a["carrier"], a["placeholder"]))
        within_size_variation += len(per_sizes) > 1
        within_roles_variation += len(per_roles) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert object_counts == set(range(1, 7))
    assert len(shapes) >= 40 and len(bands) >= 12
    assert any(len(pattern) >= 5 for pattern in bands)
    assert max(object_sizes) >= 14 and min(object_sizes) <= 5
    assert any(h >= 4 for h, _ in object_boxes)
    assert any(w >= 6 for _, w in object_boxes)
    assert edge_labels >= 20 and diagonal >= 10 and repeated >= 10
    assert len(palette_positions) >= 6 and len(roles) >= 30
    assert within_size_variation >= 35 and within_roles_variation >= 15
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 3),
                         (201, None, 1), (202, None, 4),
                         (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    a = generate_puzzle(999, 7, 3)
    b = generate_puzzle(999, 7, 3)
    c = generate_puzzle(1000, 7, 3)
    def signature(p):
        return tuple((_grid_key(q["input"]), _grid_key(q["output"]))
                     for split in ("train", "test") for q in p[split])
    assert signature(a) == signature(b) and signature(a) != signature(c)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
