"""Marker-guided completion of concealed 3 by 3 patches."""

import hashlib

import numpy as np


PERIMETER = ((0, 0), (0, 1), (0, 2), (1, 2),
             (2, 2), (2, 1), (2, 0), (1, 0))
MASKS = tuple(m for m in range(1, 256) if 1 <= m.bit_count() <= 6)
SOURCE_INPUT_HASHES = {
    "39152b766bc67677b8c25c54bccfe97b18a157748278a5c62258dd3d860ebdf2",
    "4d8cc39fa79da66a3fe8aad2a291a5e9a46676a68054f015143877737f8ed6b0",
    "c65ee8dbd274708d0c432d9f11618c69c0fba381d4261b07dfe924cd4553347a",
    "10e9d90b7198accceecc5d8db2b2c5e2b2ae036da47139d13b3035143cb5a7cc",
}
COUNT_PAIRS = tuple((a, b) for b in range(1, 5)
                    for a in range(2, 11) if a > b)


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        return COUNT_PAIRS[seed % len(COUNT_PAIRS)]
    if num_train is None:
        choices = tuple(range(max(2, num_test + 1), 11))
        return choices[seed % len(choices)], num_test
    if num_test is None:
        choices = tuple(range(1, min(4, num_train - 1) + 1))
        return num_train, choices[seed % len(choices)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _cells(row, col, width):
    return sum(1 << ((row + dr) * width + col + dc)
               for dr in range(3) for dc in range(3))


def _windows(grid, background, carrier, marker):
    """Return all full windows and the marker-bearing candidates."""
    height, width = grid.shape
    full = []
    candidates = []
    carrier_only = 0
    for row in range(height - 2):
        for col in range(width - 2):
            window = grid[row:row + 3, col:col + 3]
            if np.any(window == background):
                continue
            full.append((row, col))
            if np.all(window == carrier):
                carrier_only += 1
            if int(grid[row + 1, col + 1]) != carrier:
                continue
            marked = tuple((row + dr) * width + col + dc
                           for dr, dc in PERIMETER
                           if int(grid[row + dr, col + dc]) == marker)
            if marked:
                candidates.append((row, col, _cells(row, col, width),
                                   sum(1 << cell for cell in marked)))
    return full, carrier_only, candidates


def _interpret(grid, background, carrier, marker):
    """Solve the marker cover using only input pixels and learned roles.

    A legal interpretation uses nonoverlapping full windows, covers every
    marker exactly once, and may leave carrier-only full windows untouched.
    Search stops on the second interpretation.
    """
    full, carrier_only, candidates = _windows(grid, background, carrier, marker)
    height, width = grid.shape
    marker_cells = np.argwhere(grid == marker)
    remaining = sum(1 << (int(r) * width + int(c)) for r, c in marker_cells)
    if not remaining:
        return None
    by_marker = {int(r) * width + int(c): [] for r, c in marker_cells}
    for index, (_, _, _, marks) in enumerate(candidates):
        bits = marks
        while bits:
            bit = bits & -bits
            by_marker[bit.bit_length() - 1].append(index)
            bits -= bit
    solutions = []

    def search(left, occupied, chosen):
        if len(solutions) >= 2:
            return
        if not left:
            solutions.append(tuple(sorted(chosen)))
            return
        best = None
        options = None
        bits = left
        while bits:
            bit = bits & -bits
            choices = [i for i in by_marker[bit.bit_length() - 1]
                       if candidates[i][2] & occupied == 0
                       and candidates[i][3] & ~left == 0]
            if not choices:
                return
            if options is None or len(choices) < len(options):
                best, options = bit, choices
            bits -= bit
        for i in options:
            row, col, cells, marks = candidates[i]
            search(left & ~marks, occupied | cells, chosen + ((row, col),))
            if len(solutions) >= 2:
                return

    search(remaining, 0, ())
    if len(solutions) != 1:
        return None
    return {
        "positions": solutions[0],
        "full": len(full),
        "carrier_only": carrier_only,
        "candidates": len(candidates),
    }


def _position_options(height, width, edge):
    positions = [(r, c) for r in range(height - 2)
                 for c in range(width - 2)]
    if edge == "interior":
        return [(r, c) for r, c in positions
                if 0 < r < height - 3 and 0 < c < width - 3]
    if edge == "top":
        return [(r, c) for r, c in positions if r == 0]
    if edge == "bottom":
        return [(r, c) for r, c in positions if r == height - 3]
    if edge == "left":
        return [(r, c) for r, c in positions if c == 0]
    if edge == "right":
        return [(r, c) for r, c in positions if c == width - 3]
    if edge == "corner":
        return [(0, 0), (0, width - 3),
                (height - 3, 0), (height - 3, width - 3)]
    raise ValueError("unknown edge mode")


def _overlap(a, b):
    return abs(a[0] - b[0]) < 3 and abs(a[1] - b[1]) < 3


def _positions(rng, height, width, count, edge):
    interior = _position_options(height, width, "interior")
    all_places = [(r, c) for r in range(height - 2)
                  for c in range(width - 2)]
    chosen = []
    first_options = _position_options(height, width, edge)
    chosen.append(first_options[int(rng.integers(len(first_options)))])
    options = interior if edge == "interior" else all_places
    for index in rng.permutation(len(options)):
        candidate = options[int(index)]
        if all(not _overlap(candidate, existing) for existing in chosen):
            chosen.append(candidate)
            if len(chosen) == count:
                return tuple(chosen)
    return None


def _new_mask(rng, excluded=()):
    marker_count = int(rng.integers(1, 7))
    allowed = [m for m in MASKS
               if m.bit_count() == marker_count and m not in excluded]
    if not allowed:
        allowed = [m for m in MASKS if m not in excluded]
    return allowed[int(rng.integers(len(allowed)))]


def _make_pair(rng, roles, height, width, count, edge, richness, masks):
    background, carrier, marker, border, center = roles
    for _ in range(180):
        positions = _positions(rng, height, width, count, edge)
        if positions is None:
            continue
        density = float(rng.uniform(0.55, 0.70)) if richness else float(rng.uniform(0.43, 0.62))
        grid = np.where(rng.random((height, width)) < density,
                        carrier, background).astype(np.int8)
        occupied = np.zeros((height, width), dtype=bool)
        for (row, col), mask in zip(positions, masks):
            occupied[row:row + 3, col:col + 3] = True
            grid[row:row + 3, col:col + 3] = carrier
            for bit, (dr, dc) in enumerate(PERIMETER):
                if mask & (1 << bit):
                    grid[row + dr, col + dc] = marker
        if richness:
            # An explicit carrier-only window makes marker use necessary
            # even in small or unusually sparse noise fields.
            distractors = [(r, c) for r in range(height - 2)
                           for c in range(width - 2)
                           if not np.any(occupied[r:r + 3, c:c + 3])]
            if not distractors:
                continue
            r, c = distractors[int(rng.integers(len(distractors)))]
            grid[r:r + 3, c:c + 3] = carrier
        interpretation = _interpret(grid, background, carrier, marker)
        if interpretation is None or set(interpretation["positions"]) != set(positions):
            continue
        if richness and (interpretation["full"] < count + 2
                         or interpretation["carrier_only"] < 1
                         or interpretation["candidates"] <= count):
            continue
        if not np.any((~occupied) & (grid == carrier)):
            continue
        output = grid.copy()
        for row, col in positions:
            output[row + 1, col + 1] = center
            for dr, dc in PERIMETER:
                if grid[row + dr, col + dc] == carrier:
                    output[row + dr, col + dc] = border
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct an unambiguous pair")


def _dimensions(rng, count):
    if rng.random() < 0.24:
        return 19, 19
    minimum = 12 + count // 2
    return int(rng.integers(minimum, 29)), int(rng.integers(minimum, 29))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    anchor = _new_mask(rng)
    counts = [int(rng.integers(2, 7)) for _ in range(num_train + num_test)]
    if counts[1] == counts[0]:
        counts[1] = 2 + (counts[1] - 1) % 5
    puzzle = {"train": [], "test": []}
    training_masks = set()
    used_shapes = set()
    edge_choices = ("top", "bottom", "left", "right", "corner")
    pair_seeds = rng.integers(0, 2**63 - 1, size=len(counts), dtype=np.int64)
    for index, count in enumerate(counts):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        split = "train" if index < num_train else "test"
        if index == 0:
            edge, richness = "interior", False
        elif index == 1:
            edge = edge_choices[int(pair_rng.integers(len(edge_choices)))]
            richness = True
        else:
            edge = (edge_choices[int(pair_rng.integers(len(edge_choices)))]
                    if pair_rng.random() < (0.72 if split == "test" else 0.55)
                    else "interior")
            richness = split == "test" or bool(pair_rng.integers(2))
        masks = [_new_mask(pair_rng) for _ in range(count)]
        if index == 0 or index == num_train:
            masks[0] = anchor
        elif pair_rng.random() < 0.35:
            masks[0] = anchor
        if index == 0:
            dense = [m for m in MASKS if m.bit_count() >= 5]
            masks[1] = dense[int(pair_rng.integers(len(dense)))]
        if index == num_train:
            masks[1] = _new_mask(pair_rng, training_masks)
        if count >= 3 and pair_rng.random() < 0.32:
            masks[-1] = masks[0]
        if richness:
            excluded = training_masks if index == num_train else ()
            sparse = [m for m in MASKS if m.bit_count() == 2 and m not in excluded]
            masks[-1] = sparse[int(pair_rng.integers(len(sparse)))]
        height, width = _dimensions(pair_rng, count)
        if index == 1:
            while (height, width) in used_shapes:
                height, width = _dimensions(pair_rng, count)
        used_shapes.add((height, width))
        pair = _make_pair(pair_rng, roles, height, width, count,
                          edge, richness, masks)
        puzzle[split].append(pair)
        if split == "train":
            training_masks.update(masks)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.tobytes()


def _mask(grid, row, col, marker):
    return sum(1 << bit for bit, (dr, dc) in enumerate(PERIMETER)
               if int(grid[row + dr, col + dc]) == marker)


def _inspect_pair(pair, roles):
    grid, output = pair["input"], pair["output"]
    background, carrier, marker, border, center = roles
    assert grid.shape == output.shape
    assert set(int(x) for x in np.unique(grid)) == {background, carrier, marker}
    assert set(int(x) for x in np.unique(output)) == set(roles)
    interpretation = _interpret(grid, background, carrier, marker)
    assert interpretation is not None
    positions = interpretation["positions"]
    assert 2 <= len(positions) <= 6
    expected = grid.copy()
    occupied = np.zeros(grid.shape, dtype=bool)
    masks = []
    for row, col in positions:
        occupied[row:row + 3, col:col + 3] = True
        expected[row + 1, col + 1] = center
        masks.append(_mask(grid, row, col, marker))
        for dr, dc in PERIMETER:
            if grid[row + dr, col + dc] == carrier:
                expected[row + dr, col + dc] = border
    assert np.array_equal(output, expected)
    assert np.all(grid[occupied & (grid == marker)] == output[occupied & (grid == marker)])
    assert not np.any((~occupied) & (grid == marker))
    assert np.array_equal(grid[~occupied], output[~occupied])
    assert all(1 <= mask.bit_count() <= 6 for mask in masks)
    assert np.any((~occupied) & (grid == carrier))
    return {**interpretation, "masks": tuple(masks),
            "shape": grid.shape, "count": len(positions),
            "edges": tuple(sorted({name for r, c in positions
                                   for name, condition in (("top", r == 0),
                                                           ("bottom", r == grid.shape[0] - 3),
                                                           ("left", c == 0),
                                                           ("right", c == grid.shape[1] - 3))
                                   if condition}))}


def _infer_roles(pair):
    grid, output = pair["input"], pair["output"]
    changed = grid != output
    carriers = np.unique(grid[changed])
    assert len(carriers) == 1
    carrier = int(carriers[0])
    input_colors = {int(x) for x in np.unique(grid)}
    new_colors = {int(x) for x in np.unique(output)} - input_colors
    assert len(input_colors) == 3 and len(new_colors) == 2
    hypotheses = []
    other = tuple(input_colors - {carrier})
    new = tuple(new_colors)
    for background, marker in ((other[0], other[1]), (other[1], other[0])):
        for border, center in ((new[0], new[1]), (new[1], new[0])):
            roles = (background, carrier, marker, border, center)
            try:
                _inspect_pair(pair, roles)
            except AssertionError:
                continue
            hypotheses.append(roles)
    assert len(hypotheses) == 1
    return hypotheses[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    roles = None
    keys = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _key(pair["input"])
            assert key not in keys
            keys.add(key)
            source_hash = hashlib.sha256(bytes(int(x) for x in pair["input"].flat)).hexdigest()
            assert pair["input"].shape != (19, 19) or source_hash not in SOURCE_INPUT_HASHES
            if roles is None:
                roles = _infer_roles(pair)
            analyses[split].append(_inspect_pair(pair, roles))
    train, test = analyses["train"], analyses["test"]
    assert len({item["shape"] for item in train}) >= 2
    assert len({item["count"] for item in train}) >= 2
    assert not train[0]["edges"]
    assert train[1]["edges"]
    assert train[1]["full"] >= train[1]["count"] + 2
    assert train[1]["carrier_only"] >= 1
    assert train[1]["candidates"] > train[1]["count"]
    assert any(mask.bit_count() >= 5 for item in train for mask in item["masks"])
    train_masks = {mask for item in train for mask in item["masks"]}
    test_masks = {mask for item in test for mask in item["masks"]}
    assert train_masks & test_masks
    assert test_masks - train_masks
    assert test[0]["full"] >= test[0]["count"] + 2
    assert test[0]["carrier_only"] >= 1
    assert test[0]["candidates"] > test[0]["count"]


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[s], second[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    assert _same(generate_puzzle(314159), generate_puzzle(314159))
    assert _same(generate_puzzle(271828, 7, 3), generate_puzzle(271828, 7, 3))
    assert not _same(generate_puzzle(100, 6, 2), generate_puzzle(101, 6, 2))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(8000 + a * 10 + b, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for seed, a, b in ((9001, None, 1), (9002, None, 4),
                       (9003, 2, None), (9004, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(123, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    train_counts, test_counts, shapes, counts, edges = set(), set(), set(), set(), set()
    masks, marker_counts, dense, repeated, novel = set(), set(), 0, 0, 0
    rich_train, rich_test, carrier_distractors, combined = 0, 0, 0, 0
    corners = touching = distant = repeated_within = source_like = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = _infer_roles(puzzle["train"][0])
        train_masks = set()
        test_masks = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _inspect_pair(pair, roles)
                shapes.add(item["shape"])
                counts.add(item["count"])
                edges.update(item["edges"])
                masks.update(item["masks"])
                marker_counts.update(mask.bit_count() for mask in item["masks"])
                dense += sum(mask.bit_count() >= 5 for mask in item["masks"])
                carrier_distractors += item["carrier_only"] > 0
                height, width = item["shape"]
                corners += any(r in (0, height - 3) and c in (0, width - 3)
                               for r, c in item["positions"])
                repeated_within += len(set(item["masks"])) < len(item["masks"])
                source_like += item["shape"] == (19, 19) and item["count"] == 4
                distances = [max(0, abs(r - s) - 3) + max(0, abs(c - d) - 3)
                             for i, (r, c) in enumerate(item["positions"])
                             for s, d in item["positions"][i + 1:]]
                touching += min(distances) == 0
                distant += min(distances) >= 3
                rich = item["full"] > item["count"] and item["candidates"] > item["count"]
                if split == "train":
                    train_masks.update(item["masks"])
                    rich_train += rich
                else:
                    test_masks.update(item["masks"])
                    rich_test += rich
                combined += bool(item["edges"] and item["carrier_only"]
                                 and any(mask.bit_count() >= 5 for mask in item["masks"])
                                 and item["shape"] != (19, 19))
        repeated += bool(train_masks & test_masks)
        novel += bool(test_masks - train_masks)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(counts) == 5
    assert edges == {"top", "bottom", "left", "right"}
    assert len(masks) >= 80 and marker_counts == set(range(1, 7)) and dense >= 42
    assert corners >= 5 and touching >= 10 and distant >= 10
    assert repeated_within >= 5 and source_like >= 5
    assert repeated == novel == 42
    assert rich_train >= 42 and rich_test >= 42
    assert carrier_distractors >= 84 and combined >= 10
