import hashlib
import itertools
import numpy as np


_SOURCE_INPUT_HASHES = frozenset({
    "188fa1444dd23d8358d36ceb9007d3757179024ab8183de89428287da27e89df",
    "febd8a48c6c472ddbc2050cf9623a81dd5b7318e17748021204f30d8db9cecf1",
    "84f149d2f229ff40d8d8fc8fe4a2b2f4305e68788cc3d50877c38c201ce89458",
    "c31819c1095559db4d5d7aa92d6de2909671ace50ae8605820ebf6abc1f9b609",
    "fae956bfe77e808b624af8cbf58476929ff311e564b628ea74e4cc1afedd8385",
})
_NONSQUARE = ((4, 5), (5, 4), (4, 6), (6, 4), (5, 6), (6, 5))
_GENERAL_DIMS = ((4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (5, 6), (6, 5), (6, 6))


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _checked_count(num_train, "num_train")
    test = _checked_count(num_test, "num_test")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _variants(height, width, motif):
    base = np.zeros((height, width), dtype=np.uint8)
    for row, col in motif:
        base[row, col] = 1
    records, seen = [], set()
    for reflected in (False, True):
        reflected_grid = np.fliplr(base) if reflected else base
        for turns in range(4):
            transformed = np.rot90(reflected_grid, turns)
            coords = tuple(map(tuple, np.argwhere(transformed == 1).tolist()))
            signature = (int(transformed.shape[0]), int(transformed.shape[1]), coords)
            if signature not in seen:
                seen.add(signature)
                records.append({"h": signature[0], "w": signature[1],
                                "coords": coords, "signature": signature,
                                "reflected": reflected, "turns": turns})
    return records


def _motif(rng, category):
    if category == "line2":
        height, width = ((3, 4), (4, 3))[int(rng.integers(2))]
        motif = ((1, 1), (1, 2)) if height == 3 else ((1, 1), (2, 1))
        return height, width, motif
    if category == "diag2":
        height = width = 4
        motif = ((1, 1), (2, 2)) if rng.random() < .5 else ((1, 2), (2, 1))
        return height, width, motif
    if category == "symmetric3":
        height = width = 5
        cells = list(itertools.product(range(1, 4), repeat=2))
        choices = [m for m in itertools.combinations(cells, 3)
                   if len(_variants(5, 5, m)) == 4]
        motif = choices[int(rng.integers(len(choices)))]
        return height, width, motif
    for _ in range(100):
        height, width = _GENERAL_DIMS[int(rng.integers(len(_GENERAL_DIMS)))]
        count = 3 if category == "generic3" else 4
        cells = list(itertools.product(range(1, height - 1), range(1, width - 1)))
        indices = rng.choice(len(cells), size=count, replace=False)
        motif = tuple(sorted(cells[int(i)] for i in indices))
        if len(_variants(height, width, motif)) == 8:
            return height, width, motif
    raise ValueError("could not construct a motif")


def _extract(grid):
    values, counts = np.unique(grid, return_counts=True)
    if len(values) != 3:
        raise AssertionError("input needs three role colors")
    background = int(values[np.argmax(counts)])
    candidates = []
    for value in values:
        carrier = int(value)
        if carrier == background:
            continue
        points = np.argwhere(grid == carrier)
        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0) + 1
        height, width = int(bottom - top), int(right - left)
        if not 3 <= height <= 6 or not 3 <= width <= 6:
            continue
        sub = grid[top:bottom, left:right]
        if np.any(sub == background):
            continue
        if not (np.all(sub[0] == carrier) and np.all(sub[-1] == carrier)
                and np.all(sub[:, 0] == carrier) and np.all(sub[:, -1] == carrier)):
            continue
        marker = next(int(v) for v in values if v not in (background, carrier))
        motif = tuple(map(tuple, np.argwhere(sub == marker).tolist()))
        if not 2 <= len(motif) <= 4 or len(_variants(height, width, motif)) < 2:
            continue
        candidates.append({"bg": background, "carrier": carrier, "marker": marker,
                           "top": int(top), "left": int(left), "h": height,
                           "w": width, "motif": motif})
    if len(candidates) != 1:
        raise AssertionError("prototype is not uniquely identifiable")
    return candidates[0]


def _matches(grid, info):
    markers = set(map(tuple, np.argwhere(grid == info["marker"]).tolist()))
    matches = []
    for variant in _variants(info["h"], info["w"], info["motif"]):
        ar, ac = variant["coords"][0]
        for row, col in markers:
            top, left = row - ar, col - ac
            if all((top + r, left + c) in markers for r, c in variant["coords"]):
                matches.append({**variant, "top": top, "left": left})
    return matches


def _rect(match):
    return match["top"], match["left"], match["h"], match["w"]


def _intersection(a, b):
    at, al, ah, aw = a
    bt, bl, bh, bw = b
    return max(0, min(at + ah, bt + bh) - max(at, bt)) * max(0, min(al + aw, bl + bw) - max(al, bl))


def _touch(a, b):
    at, al, ah, aw = a
    bt, bl, bh, bw = b
    row_overlap = max(0, min(at + ah, bt + bh) - max(at, bt))
    col_overlap = max(0, min(al + aw, bl + bw) - max(al, bl))
    return (row_overlap > 0 and (al + aw == bl or bl + bw == al)) or (col_overlap > 0 and (at + ah == bt or bt + bh == at))


def _active(grid, info=None):
    if info is None:
        info = _extract(grid)
    active = []
    for match in _matches(grid, info):
        top, left, height, width = _rect(match)
        r0, r1 = max(0, top), min(grid.shape[0], top + height)
        c0, c1 = max(0, left), min(grid.shape[1], left + width)
        if r0 < r1 and c0 < c1 and np.any(grid[r0:r1, c0:c1] == info["bg"]):
            active.append(match)
    return active


def _solve(grid):
    info = _extract(grid)
    output = grid.copy()
    for match in _matches(grid, info):
        top, left, height, width = _rect(match)
        r0, r1 = max(0, top), min(grid.shape[0], top + height)
        c0, c1 = max(0, left), min(grid.shape[1], left + width)
        if r0 < r1 and c0 < c1:
            region = output[r0:r1, c0:c1]
            region[region == info["bg"]] = info["carrier"]
    return output


def _bruteforce_expected(grid):
    """A separate exhaustive construction used by the self-test."""
    info = _extract(grid)
    height, width = info["h"], info["w"]
    mask = np.zeros((height, width), dtype=bool)
    for row, col in info["motif"]:
        mask[row, col] = True
    output = grid.copy()
    seen = set()
    for flip in (False, True):
        start = np.fliplr(mask) if flip else mask
        for turns in range(4):
            transformed = np.rot90(start, turns)
            offsets = tuple(map(tuple, np.argwhere(transformed).tolist()))
            signature = (transformed.shape, offsets)
            if signature in seen:
                continue
            seen.add(signature)
            first_row = min(r for r, _ in offsets)
            last_row = max(r for r, _ in offsets)
            first_col = min(c for _, c in offsets)
            last_col = max(c for _, c in offsets)
            for top in range(-first_row, grid.shape[0] - last_row):
                for left in range(-first_col, grid.shape[1] - last_col):
                    if not all(grid[top + r, left + c] == info["marker"]
                               for r, c in offsets):
                        continue
                    r0 = max(0, top)
                    r1 = min(grid.shape[0], top + transformed.shape[0])
                    c0 = max(0, left)
                    c1 = min(grid.shape[1], left + transformed.shape[1])
                    region = output[r0:r1, c0:c1]
                    region[region == info["bg"]] = info["carrier"]
    return output


def _positions(variant, mode, grid_shape):
    height, width = grid_shape
    vh, vw = variant["h"], variant["w"]
    rows = range(0, height - vh + 1)
    cols = range(0, width - vw + 1)
    if mode == "top":
        rows = (-1,)
    elif mode == "bottom":
        rows = (height - vh + 1,)
    elif mode == "left":
        cols = (-1,)
    elif mode == "right":
        cols = (width - vw + 1,)
    return list(itertools.product(rows, cols))


def _place(rng, variant, mode, shape, prototype, placed, occupied, relation):
    positions = _positions(variant, mode, shape)
    for index in rng.permutation(len(positions)):
        top, left = positions[int(index)]
        rect = (top, left, variant["h"], variant["w"])
        markers = {(top + r, left + c) for r, c in variant["coords"]}
        if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in markers):
            continue
        if _intersection(rect, prototype) or markers & occupied:
            continue
        if relation == "touch" and placed and not any(_touch(rect, old) for old in placed):
            continue
        if relation == "overlap" and placed:
            intersections = [_intersection(rect, old) for old in placed]
            if not any(0 < area <= min(rect[2] * rect[3], old[2] * old[3]) // 3
                       for area, old in zip(intersections, placed)):
                continue
        elif relation != "overlap" and any(_intersection(rect, old) for old in placed):
            continue
        return rect, markers
    return None


def _spec(rng, kind):
    if kind == "anchor":
        category = "generic3"
        height, width, motif = _motif(rng, category)
        while height == width:
            height, width, motif = _motif(rng, category)
        variants = _variants(height, width, motif)
        quarter = [v for v in variants if not v["reflected"] and v["turns"] in (1, 3)]
        reflections = [v for v in variants if v["reflected"]]
        targets = [(quarter[int(rng.integers(len(quarter)))],
                    ("top", "bottom", "left", "right")[int(rng.integers(4))]),
                   (reflections[int(rng.integers(len(reflections)))], "interior")]
        return height, width, motif, variants, targets
    category = kind if kind in ("line2", "diag2") else (
        "line2", "diag2", "symmetric3", "generic3", "generic4")[int(rng.integers(5))]
    height, width, motif = _motif(rng, category)
    variants = _variants(height, width, motif)
    counts = (1, 2, 2, 3, 3, 4) if kind == "test" else (1, 1, 2, 2, 3, 4)
    target_count = int(rng.choice(counts))
    chosen = rng.choice(len(variants), size=target_count, replace=target_count > len(variants))
    targets = []
    for i in chosen:
        modes = (("interior",) * 4 if kind == "test" else ("interior",) * 6) + (
            "top", "bottom", "left", "right")
        mode = modes[int(rng.integers(len(modes)))]
        targets.append((variants[int(i)], mode))
    return height, width, motif, variants, targets


def _add_distractors(rng, grid, info, desired, expected_footprints):
    background, marker = info["bg"], info["marker"]
    count = 0
    attempts = 0
    while count < desired and attempts < max(150, desired * 30):
        attempts += 1
        available = np.argwhere(grid == background)
        if len(available) == 0:
            break
        row, col = available[int(rng.integers(len(available)))]
        row, col = int(row), int(col)
        grid[row, col] = marker
        current = {_rect(m) for m in _active(grid, info)}
        if current == expected_footprints:
            count += 1
        else:
            grid[row, col] = background
    return count


def _build_pair(rng, kind, colors):
    background, carrier, marker = colors
    for attempt in range(250):
        if rng.random() < .12:
            grid_height = grid_width = 23
        else:
            grid_height = int(rng.integers(11, 31))
            grid_width = int(rng.integers(11, 31))
        height, width, motif, variants, targets = _spec(rng, kind)
        top = int(rng.integers(grid_height - height + 1))
        left = int(rng.integers(grid_width - width + 1))
        prototype = (top, left, height, width)
        occupied = {(top + r, left + c) for r, c in motif}
        placed, target_markers = [], []
        success = True
        relation_choice = rng.choice(("free", "free", "free", "touch", "overlap"))
        for index, (variant, mode) in enumerate(targets):
            relation = relation_choice if index == 1 and mode == "interior" else "free"
            placement = _place(rng, variant, mode, (grid_height, grid_width),
                               prototype, placed, occupied, relation)
            if placement is None:
                success = False
                break
            rect, cells = placement
            placed.append(rect)
            target_markers.append(cells)
            occupied |= cells
        if not success:
            continue
        grid = np.full((grid_height, grid_width), background, dtype=np.int8)
        grid[top:top + height, left:left + width] = carrier
        for row, col in occupied:
            grid[row, col] = marker
        try:
            info = _extract(grid)
            inferred = {_rect(m) for m in _active(grid, info)}
        except AssertionError:
            continue
        if inferred != set(placed):
            continue
        outside_before = sum(len(cells) for cells in target_markers)
        area = grid_height * grid_width
        density_probs = (.18, .30, .52) if kind == "test" else (.25, .35, .40)
        density = rng.choice(("sparse", "medium", "dense"), p=density_probs)
        if density == "sparse":
            desired = int(rng.integers(0, 5))
        elif density == "medium":
            desired = int(rng.integers(6, 15))
        else:
            desired = int(rng.integers(max(15, area // 18), min(51, max(24, area // 9)) + 1))
        added = _add_distractors(rng, grid, info, desired, inferred)
        if added < min(desired, 2) or outside_before + added > 65:
            continue
        output = _solve(grid)
        if np.array_equal(grid, output):
            continue
        return grid, output
    raise ValueError("could not construct a controlled pair")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_hash(grid):
    small = grid.astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(grid.shape) + small.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    shared_palette = rng.random() < .45
    base_colors = tuple(map(int, rng.choice(10, size=3, replace=False)))
    two_marker_kinds = ["line2", "diag2"]
    rng.shuffle(two_marker_kinds)
    train_kinds = ["anchor", two_marker_kinds[0]]
    if train_count >= 3:
        train_kinds.append(two_marker_kinds[1])
    train_kinds.extend(["free"] * (train_count - len(train_kinds)))
    rng.shuffle(train_kinds)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, kinds in (("train", train_kinds), ("test", ["test"] * test_count)):
        for kind in kinds:
            colors = base_colors if shared_palette else tuple(map(int, rng.choice(10, size=3, replace=False)))
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(8):
                pair_input, pair_output = _build_pair(pair_rng, kind, colors)
                key = _grid_key(pair_input)
                if key not in seen and _source_hash(pair_input) not in _SOURCE_INPUT_HASHES:
                    break
            else:
                raise ValueError("could not construct a unique pair")
            seen.add(key)
            puzzle[split].append({"input": pair_input, "output": pair_output})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_features = set()
    test_features = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = pair["input"], pair["output"]
            for grid in (x, y):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert x.shape == y.shape and not np.array_equal(x, y)
            key = _grid_key(x)
            pair_key = (key, _grid_key(y))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert _source_hash(x) not in _SOURCE_INPUT_HASHES
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            info = _extract(x)
            assert np.array_equal(y, _solve(x))
            changed = x != y
            assert np.all(x[changed] == info["bg"])
            assert np.all(y[changed] == info["carrier"])
            assert np.array_equal(x == info["marker"], y == info["marker"])
            active = _active(x, info)
            footprints = {_rect(m) for m in active}
            assert 1 <= len(footprints) <= 4
            features = train_features if split == "train" else test_features
            if any(m["reflected"] for m in active):
                features.add("reflection")
            if any(m["turns"] in (1, 3) and not m["reflected"] for m in active):
                features.add("quarter_turn")
            if any(t < 0 or l < 0 or t + h > x.shape[0] or l + w > x.shape[1]
                   for t, l, h, w in footprints):
                features.add("clip")
            if len(footprints) >= 2:
                features.add("multiple")
    assert {"reflection", "quarter_turn", "clip", "multiple"} <= train_features
    assert test_features <= train_features
    train_motifs = [_extract(pair["input"])["motif"] for pair in puzzle["train"]]
    assert any(len(motif) == 2 for motif in train_motifs)
    assert any(len(motif) == 3 for motif in train_motifs)


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((11, 2, 1), (12, 10, 1), (13, 10, 4),
                              (14, None, 1), (15, None, 4), (16, 2, None),
                              (17, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _equal_puzzles(puzzle, generate_puzzle(seed, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts = set(), set()
    sizes, motifs, variants, outside, footprints, palettes = set(), set(), set(), [], set(), set()
    clipped_sides, prototype_positions = set(), set()
    overlaps = touches = pairs_with_marker_neighbors = clusters = combinations = 0
    within_motifs = within_counts = within_shapes = 0
    train_outside, test_outside = [], []
    inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = []
        puzzle_motifs, puzzle_counts, puzzle_shapes = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = pair["input"]
                assert np.array_equal(pair["output"], _bruteforce_expected(x))
                info = _extract(x)
                roles.append((info["bg"], info["carrier"], info["marker"]))
                sizes.add(x.shape)
                puzzle_shapes.add(x.shape)
                motifs.add((info["h"], info["w"], len(info["motif"])))
                puzzle_motifs.add(len(info["motif"]))
                variants.add(len(_variants(info["h"], info["w"], info["motif"])))
                field_count = int(np.count_nonzero(x == info["marker"]) - len(info["motif"]))
                outside.append(field_count)
                (train_outside if split == "train" else test_outside).append(field_count)
                active = _active(x, info)
                rects = list({_rect(m) for m in active})
                footprints.add(len(rects))
                puzzle_counts.add(len(rects))
                prototype_positions.add((info["top"] == 0 or info["left"] == 0 or
                                         info["top"] + info["h"] == x.shape[0] or
                                         info["left"] + info["w"] == x.shape[1]))
                sides = set()
                for top, left, height, width in rects:
                    if top < 0:
                        sides.add("top")
                    if left < 0:
                        sides.add("left")
                    if top + height > x.shape[0]:
                        sides.add("bottom")
                    if left + width > x.shape[1]:
                        sides.add("right")
                clipped_sides |= sides
                overlaps += any(_intersection(a, b) for i, a in enumerate(rects)
                                for b in rects[i + 1:])
                touches += any(_touch(a, b) for i, a in enumerate(rects)
                               for b in rects[i + 1:])
                template_markers = {(info["top"] + r, info["left"] + c)
                                    for r, c in info["motif"]}
                matched_markers = {(m["top"] + r, m["left"] + c)
                                   for m in active for r, c in m["coords"]}
                distractors = set(map(tuple, np.argwhere(x == info["marker"]).tolist()))
                distractors -= template_markers | matched_markers
                while distractors:
                    stack = [distractors.pop()]
                    component = 0
                    while stack:
                        row, col = stack.pop()
                        component += 1
                        neighbors = {(row + 1, col), (row - 1, col),
                                     (row, col + 1), (row, col - 1)} & distractors
                        distractors -= neighbors
                        stack.extend(neighbors)
                    pairs_with_marker_neighbors += component == 2
                    clusters += component >= 3
                combinations += (len(info["motif"]) == 2 and field_count >= 24
                                 and bool(sides) and len(rects) >= 2)
                inputs.add(_grid_key(x))
        palettes.add(len(set(roles)) > 1)
        within_motifs += len(puzzle_motifs) > 1
        within_counts += len(puzzle_counts) > 1
        within_shapes += len(puzzle_shapes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 35 and len(motifs) >= 10
    assert {2, 4, 8} <= variants
    assert min(outside) <= 6 and max(outside) >= 25
    assert {1, 2, 3} <= footprints
    assert palettes == {False, True}
    assert any(min(s) <= 15 for s in sizes) and any(max(s) >= 28 for s in sizes)
    assert (23, 23) in sizes
    assert len(inputs) >= 150
    assert {"top", "bottom", "left", "right"} <= clipped_sides
    assert prototype_positions == {False, True}
    assert overlaps >= 3 and touches >= 3
    assert pairs_with_marker_neighbors >= 10 and clusters >= 2
    assert combinations >= 3
    assert within_motifs >= 30 and within_counts >= 25 and within_shapes == 42
    assert float(np.median(test_outside)) > float(np.median(train_outside))
