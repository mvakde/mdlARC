import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_SOURCE_INPUTS = {
    """0000000000000000
0070000000000000
0770000000000000
1111000000000000
0001000077000000
1010000070000000
0111000101100000
0000000111100000
0000000100100000
0000000000000000
0000000000000000
0007700000000000
0000700000000000
0001011000000000
0000111000000000
0001001000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000""",
    """000000
007000
077000
101100
011100
110000
000000
000000""",
    """00000000000
00000077000
00000070000
07700111000
00700011100
10100010100
01110000000
01010000000
00000000000""",
    """000000000000
070000000000
077000000000
101100070000
010000770000
111001101000
000000111000
000000101000""",
    """000000000000
077000000000
070000007000
111000077000
011000111100
100000010100
000000001000
000000000000
000077000000
000007000000
000110100700
000101000770
000011001101
000000000111
000000001100""",
}


def _choose_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} is out of range")
    choices = [
        (a, b)
        for a in range(2, 11)
        for b in range(1, 5)
        if a > b
        and (num_train is None or a == num_train)
        and (num_test is None or b == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _train_schedule(rng, count):
    sizes = [int(rng.integers(1, 4)) for _ in range(count)]
    while sum(sizes) < 4:
        choices = [i for i, size in enumerate(sizes) if size < 3]
        sizes[int(rng.choice(choices))] += 1
    slots = [(i, j) for i, size in enumerate(sizes) for j in range(size)]
    assignments = {}
    corners = list(_CORNERS)
    rng.shuffle(corners)
    if count >= 4 and rng.random() < 0.65:
        late = (count - 1, int(rng.integers(sizes[-1])))
        assignments[late] = corners.pop()
        earlier = [slot for slot in slots if slot[0] < count - 1]
        chosen = rng.choice(len(earlier), size=3, replace=False)
        for j, corner in zip(chosen, corners):
            assignments[earlier[int(j)]] = corner
    else:
        chosen = rng.choice(len(slots), size=4, replace=False)
        for j, corner in zip(chosen, corners):
            assignments[slots[int(j)]] = corner
    return [
        [
            assignments.get((i, j), _CORNERS[int(rng.integers(4))])
            for j in range(size)
        ]
        for i, size in enumerate(sizes)
    ]


def _test_schedule(rng, count):
    result = []
    for _ in range(count):
        size = int(rng.choice((1, 2, 3, 4), p=(0.17, 0.25, 0.27, 0.31)))
        if size == 4 and rng.random() < 0.6:
            corners = list(_CORNERS)
            rng.shuffle(corners)
        else:
            corners = [_CORNERS[int(rng.integers(4))] for _ in range(size)]
        result.append(corners)
    return result


def _sample_pattern(rng, width, height):
    density = float(rng.uniform(0.28, 0.88))
    mask = rng.random((height, width)) < density
    for row in range(height):
        if not mask[row].any():
            mask[row, int(rng.integers(width))] = True
    for col in range(width):
        if not mask[:, col].any():
            mask[int(rng.integers(height)), col] = True
    while int(mask.sum()) < 4:
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        mask[row, col] = True
    return mask


def _make_object(rng, missing, roles, regime, shared=None, compact=False):
    background, source, marker, mapping = roles
    if shared is None:
        limit = 5 if compact else 8 if regime == "target_test" else 7
        width = int(rng.integers(2, limit + 1))
        height = int(rng.integers(2, limit + 1))
        mask = _sample_pattern(rng, width, height)
    else:
        mask = shared.copy()
        height, width = mask.shape
    marker_x = int(rng.integers(width - 1))
    bottom = [marker_x + dx for dx in (0, 1) if (1, dx) != missing]
    detached = False
    if shared is None and rng.random() < (0.36 if regime == "target_test" else 0.27):
        allowed = [col for col in range(width) if col not in bottom]
        if allowed and height * width - len(bottom) >= 4:
            detached = True
            mask[0, bottom] = False
            if not mask[0].any():
                mask[0, int(rng.choice(allowed))] = True
            for col in range(width):
                if not mask[:, col].any():
                    mask[int(rng.integers(1, height)), col] = True
    while int(mask.sum()) < 4:
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        if not detached or row != 0 or col not in bottom:
            mask[row, col] = True
    inp = np.full((height + 2, width), background, dtype=np.int8)
    out = inp.copy()
    for row in (0, 1):
        for col in (0, 1):
            if (row, col) != missing:
                inp[row, marker_x + col] = marker
    inp[2:][mask] = source
    out[2:][mask] = mapping[missing]
    return inp, out


def _join(rng, left, right, axis, background):
    a_in, a_out = left
    b_in, b_out = right
    gap = int(rng.integers(1, 5))
    slack = int(rng.integers(0, 4))
    if axis == 1:
        height = max(a_in.shape[0], b_in.shape[0]) + slack
        width = a_in.shape[1] + gap + b_in.shape[1]
        a_pos = (int(rng.integers(height - a_in.shape[0] + 1)), 0)
        b_pos = (int(rng.integers(height - b_in.shape[0] + 1)), a_in.shape[1] + gap)
    else:
        height = a_in.shape[0] + gap + b_in.shape[0]
        width = max(a_in.shape[1], b_in.shape[1]) + slack
        a_pos = (0, int(rng.integers(width - a_in.shape[1] + 1)))
        b_pos = (a_in.shape[0] + gap, int(rng.integers(width - b_in.shape[1] + 1)))
    if height > 26 or width > 26:
        return None
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for item, pos in ((left, a_pos), (right, b_pos)):
        row, col = pos
        h, w = item[0].shape
        inp[row:row + h, col:col + w] = item[0]
        out[row:row + h, col:col + w] = item[1]
    return inp, out


def _pack(rng, objects, background):
    if len(objects) == 1:
        return objects[0]
    split = int(rng.integers(1, len(objects)))
    left = _pack(rng, objects[:split], background)
    right = _pack(rng, objects[split:], background)
    if left is None or right is None:
        return None
    axes = [0, 1]
    rng.shuffle(axes)
    for axis in axes:
        joined = _join(rng, left, right, axis, background)
        if joined is not None:
            return joined
    return None


def _generate_pair(rng, regime, roles, corners, shared_slots):
    background = roles[0]
    for attempt in range(160):
        objects = [
            _make_object(
                rng, corner, roles, regime, shared_slots.get(index),
                compact=attempt >= 100,
            )
            for index, corner in enumerate(corners)
        ]
        rng.shuffle(objects)
        packed = _pack(rng, objects, background)
        if packed is None:
            continue
        small_in, small_out = packed
        h, w = small_in.shape
        max_margin = 8 if regime == "target_test" else 6
        top = int(rng.integers(1, min(max_margin, 30 - h - 1) + 1))
        bottom = int(rng.integers(1, min(max_margin, 30 - h - top) + 1))
        left = int(rng.integers(1, min(max_margin, 30 - w - 1) + 1))
        right = int(rng.integers(1, min(max_margin, 30 - w - left) + 1))
        inp = np.full((top + h + bottom, left + w + right), background, dtype=np.int8)
        out = inp.copy()
        inp[top:top + h, left:left + w] = small_in
        out[top:top + h, left:left + w] = small_out
        return inp, out
    raise RuntimeError("could not pack objects within 30 by 30")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    roles = (
        colors[0], colors[1], colors[2],
        {corner: colors[3 + i] for i, corner in enumerate(_CORNERS)},
    )
    train_schedule = _train_schedule(rng, train_count)
    test_schedule = _test_schedule(rng, test_count)

    # The same silhouette under two different markers makes the code explicit.
    flat_train = [(i, j, corner) for i, row in enumerate(train_schedule)
                  for j, corner in enumerate(row)]
    first = int(rng.integers(len(flat_train)))
    alternatives = [i for i, slot in enumerate(flat_train)
                    if slot[2] != flat_train[first][2]]
    second = int(rng.choice(alternatives))
    clone = _sample_pattern(rng, int(rng.integers(3, 6)), int(rng.integers(2, 6)))
    shared = {
        flat_train[first][:2]: clone,
        flat_train[second][:2]: clone,
    }

    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, schedule in (("source_train", train_schedule),
                             ("target_test", test_schedule)):
        destination = "train" if regime == "source_train" else "test"
        for pair_index, corners in enumerate(schedule):
            shared_slots = {
                j: mask for (i, j), mask in shared.items()
                if regime == "source_train" and i == pair_index
            }
            for _ in range(12):
                child_seed = int(rng.integers(0, 2**63, dtype=np.int64))
                inp, out = _generate_pair(
                    np.random.default_rng(child_seed), regime,
                    roles, corners, shared_slots,
                )
                key = (inp.shape, inp.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[destination].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("could not make distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _regions(inp, background):
    mask = inp != background
    rows, cols = np.nonzero(mask)
    assert len(rows) > 0

    def split(top, bottom, left, right):
        local = mask[top:bottom, left:right]
        rr, cc = np.nonzero(local)
        assert len(rr) > 0
        top += int(rr.min())
        bottom = top + int(rr.max() - rr.min()) + 1
        left += int(cc.min())
        right = left + int(cc.max() - cc.min()) + 1
        for row in range(top + 1, bottom - 1):
            if not mask[row, left:right].any():
                return split(top, row, left, right) + split(row + 1, bottom, left, right)
        for col in range(left + 1, right - 1):
            if not mask[top:bottom, col].any():
                return split(top, bottom, left, col) + split(top, bottom, col + 1, right)
        return [(top, bottom, left, right)]

    return split(0, inp.shape[0], 0, inp.shape[1])


def _components(mask):
    points = {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}
    result = []
    while points:
        start = points.pop()
        stack = [start]
        component = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in points:
                    points.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        result.append(component)
    return result


def _describe_input(inp, background, source, marker):
    assert len({background, source, marker}) == 3
    assert set(int(x) for x in np.unique(inp)) == {background, source, marker}
    descriptions = []
    for top, bottom, left, right in _regions(inp, background):
        region = inp[top:bottom, left:right]
        marker_points = [(top + int(r), left + int(c))
                         for r, c in zip(*np.nonzero(region == marker))]
        source_points = [(top + int(r), left + int(c))
                         for r, c in zip(*np.nonzero(region == source))]
        assert len(marker_points) == 3
        assert len(_components(region == marker)) == 1
        assert len(source_points) >= 4
        mr = min(r for r, _ in marker_points)
        mc = min(c for _, c in marker_points)
        occupied = {(r - mr, c - mc) for r, c in marker_points}
        assert occupied.issubset(set(_CORNERS))
        assert len(set(_CORNERS) - occupied) == 1
        missing = next(iter(set(_CORNERS) - occupied))
        sr = min(r for r, _ in source_points)
        sc = min(c for _, c in source_points)
        er = max(r for r, _ in source_points)
        ec = max(c for _, c in source_points)
        assert sr == mr + 2
        assert er > sr and ec > sc
        assert sc <= mc and mc + 1 <= ec
        source_set = set(source_points)
        assert all(any((r, c) in source_set for c in range(sc, ec + 1))
                   for r in range(sr, er + 1))
        assert all(any((r, c) in source_set for r in range(sr, er + 1))
                   for c in range(sc, ec + 1))
        silhouette = tuple(
            tuple((r, c) in source_set for c in range(sc, ec + 1))
            for r in range(sr, er + 1)
        )
        descriptions.append({
            "missing": missing, "source": source_points,
            "marker": marker_points, "silhouette": silhouette,
            "bounds": (top, bottom, left, right),
        })
    assert descriptions
    return descriptions


def _infer_training_pair(inp, out):
    candidates = []
    for value in np.unique(inp):
        background = int(value)
        if np.any(out[inp == background] != background):
            continue
        marker_values = np.unique(inp[(inp != background) & (out == background)])
        source_values = np.unique(inp[(inp != background) & (out != background)])
        if len(marker_values) != 1 or len(source_values) != 1:
            continue
        marker, source = int(marker_values[0]), int(source_values[0])
        try:
            descriptions = _describe_input(inp, background, source, marker)
            expected = np.full(inp.shape, background, dtype=np.int8)
            mapping = {}
            for item in descriptions:
                colors = {int(out[r, c]) for r, c in item["source"]}
                assert len(colors) == 1
                color = next(iter(colors))
                assert color not in (background, source, marker)
                missing = item["missing"]
                assert missing not in mapping or mapping[missing] == color
                mapping[missing] = color
                for r, c in item["source"]:
                    expected[r, c] = color
            assert np.array_equal(out, expected)
            candidates.append((background, source, marker, mapping, descriptions))
        except AssertionError:
            continue
    assert len(candidates) == 1
    return candidates[0]


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    roles = None
    learned = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert "\n".join("".join(map(str, row)) for row in inp) not in _SOURCE_INPUTS
            if split == "train":
                background, source, marker, mapping, descriptions = (
                    _infer_training_pair(inp, out)
                )
                current = (background, source, marker)
                assert roles is None or roles == current
                roles = current
                assert 1 <= len(descriptions) <= 3
                for missing, color in mapping.items():
                    assert missing not in learned or learned[missing] == color
                    learned[missing] = color
            else:
                assert roles is not None
                descriptions = _describe_input(inp, *roles)
                assert 1 <= len(descriptions) <= 4
                expected = np.full(inp.shape, roles[0], dtype=np.int8)
                for item in descriptions:
                    assert item["missing"] in learned
                    for r, c in item["source"]:
                        expected[r, c] = learned[item["missing"]]
                assert np.array_equal(out, expected)
    assert set(learned) == set(_CORNERS)
    assert len(set(learned.values())) == 4
    assert not (set(learned.values()) & set(roles))


def _same_puzzle(a, b):
    return all(
        len(a[split]) == len(b[split])
        and all(np.array_equal(x[key], y[key])
                for x, y in zip(a[split], b[split])
                for key in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    backgrounds, codes, dimensions, structures = set(), set(), set(), set()
    disconnected = detached = large = repeated_test = four_test = 0
    clone_puzzles = variable_density_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = None
        learned = {}
        silhouettes = {}
        occupancies = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                dimensions.add(inp.shape)
                if max(inp.shape) >= 23:
                    large += 1
                structures.add((inp.shape, tuple(tuple(int(v) for v in row)
                                                  for row in inp != int(inp[0, 0]))))
                if split == "train":
                    bg, src, mark, mapping, descriptions = _infer_training_pair(inp, out)
                    roles = (bg, src, mark)
                    learned.update(mapping)
                else:
                    descriptions = _describe_input(inp, *roles)
                    if len(descriptions) == 4:
                        four_test += 1
                    if len(set(item["missing"] for item in descriptions)) < len(descriptions):
                        repeated_test += 1
                    predicted = np.full(inp.shape, roles[0], dtype=np.int8)
                    for item in descriptions:
                        for r, c in item["source"]:
                            predicted[r, c] = learned[item["missing"]]
                    assert np.array_equal(predicted, out)
                for item in descriptions:
                    source_points = item["source"]
                    assert all(inp[r, c] == roles[1] for r, c in source_points)
                    source_mask = np.zeros(inp.shape, dtype=bool)
                    for r, c in source_points:
                        source_mask[r, c] = True
                    if len(_components(source_mask)) > 1:
                        disconnected += 1
                    if not any((r + 1, c) in source_points for r, c in item["marker"]):
                        detached += 1
                    silhouette = item["silhouette"]
                    silhouettes.setdefault(silhouette, set()).add(item["missing"])
                    occupancy = len(source_points) / (
                        len(silhouette) * len(silhouette[0])
                    )
                    occupancies.append(occupancy)
        backgrounds.add(roles[0])
        codes.add(tuple(learned[corner] for corner in _CORNERS))
        if any(len(corners) > 1 for corners in silhouettes.values()):
            clone_puzzles += 1
        if max(occupancies) - min(occupancies) >= 0.18:
            variable_density_puzzles += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 5 and len(codes) >= 30
    assert len(dimensions) >= 60 and len(structures) >= 150
    assert disconnected >= 50 and detached >= 20
    assert large >= 20 and repeated_test >= 8 and four_test >= 10
    assert clone_puzzles >= 35 and variable_density_puzzles >= 30
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        seed = 100 + train_count + test_count
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _same_puzzle(puzzle, generate_puzzle(seed, train_count, test_count))
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(211, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (4, 4),
    ):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count combination was accepted")
