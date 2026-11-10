"""Generate completions of a motif reflected around a solid 2-by-2 hub."""

import hashlib

import numpy as np


# Hashes of the four original inputs, including their dimensions.  A new
# output cannot copy a source pair when its input differs from all four.
_SOURCE_INPUTS = frozenset({
    "afe7bb871a62db7491211e83a8348da70d6f5302e84f7a0201999ab3eec46917",
    "b0c450d8aec9e3c4df252a598e45a29e698d3bb7fd9c22d93b177500f73180b0",
    "82c7f155bdffc3782b89a460049f2ab98c8f33b1a7de237b6e48de3fad9c5799",
    "8b6fe1a46b2d2209b06f12d8f996ac4578f79ed8fc917a3e72938aeacbfc2bbb",
})
_SOURCE_SHAPES = ((10, 10), (20, 30), (12, 14), (18, 14))


def _source_input_hash(grid: np.ndarray) -> str:
    height, width = grid.shape
    return hashlib.sha256(bytes((height, width)) + grid.astype(np.uint8).tobytes()).hexdigest()


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.uint8).tobytes()


def _validate_count_argument(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _validate_count_argument("num_train", num_train, 2, 10)
    test = _validate_count_argument("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4 + int(seed) ** 2 + 3) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _connected(motif):
    seen = {next(iter(motif))}
    pending = list(seen)
    while pending:
        row, col = pending.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = row + dr, col + dc
                if neighbor in motif and neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
    return len(seen) == len(motif)


def _asymmetric(motif, height, width):
    return (motif != {(height - 1 - r, c) for r, c in motif}
            and motif != {(r, width - 1 - c) for r, c in motif}
            and motif != {(height - 1 - r, width - 1 - c) for r, c in motif})


def _sample_motif(rng, height, width, require_asymmetric=False):
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(350):
        mode = int(rng.integers(0, 6))
        if mode <= 2:
            density = float(rng.uniform(0.30, 0.82))
            motif = {cell for cell in cells if rng.random() < density}
        elif mode == 3:
            motif = {(r, c) for r, c in cells
                     if r in (0, height - 1) or c in (0, width - 1)}
            if height > 2 and width > 2 and rng.random() < 0.5:
                motif.discard((height - 1, 0))
        elif mode == 4:
            # Thin bars and bent strokes are deliberately simple motifs.
            motif = {(0, c) for c in range(width)} | {(r, 0) for r in range(height)}
            if height > 2 and width > 2 and rng.random() < 0.5:
                motif |= {(r, min(r, width - 1)) for r in range(height)}
        else:
            density = float(rng.uniform(0.20, 0.55))
            motif = {cell for cell in cells if rng.random() < density}
            if rng.random() < 0.5:
                motif |= {(height - 1 - r, c) for r, c in tuple(motif)}
            else:
                motif |= {(r, width - 1 - c) for r, c in tuple(motif)}

        motif.add((0, 0))
        if not any(r == height - 1 for r, _ in motif):
            motif.add((height - 1, int(rng.integers(width))))
        if not any(c == width - 1 for _, c in motif):
            motif.add((int(rng.integers(height)), width - 1))
        # A lone solid 2-by-2 payload would look like another hub.
        if height == width == 2 and len(motif) == 4:
            continue
        if not _connected(motif) and rng.random() < 0.70:
            continue
        if require_asymmetric and not _asymmetric(motif, height, width):
            continue
        return frozenset(motif)
    # Bounded fallback for narrow dimensions or a rare run of rejected masks.
    motif = {(0, c) for c in range(width)} | {(r, 0) for r in range(height)}
    if height == width == 2 and len(motif) == 4:
        motif.remove((1, 1))
    if require_asymmetric and not _asymmetric(motif, height, width):
        motif.add((1, 1))
    return frozenset(motif)


def _sample_dimensions(rng, motif_height, motif_width):
    minimum_height, minimum_width = 2 * motif_height + 2, 2 * motif_width + 2
    for _ in range(200):
        if rng.random() < 0.22:
            shape = _SOURCE_SHAPES[int(rng.integers(len(_SOURCE_SHAPES)))]
            height, width = shape
        else:
            pad_mode = int(rng.integers(3))
            if pad_mode == 0:
                extra_h = int(rng.integers(0, 5))
                extra_w = int(rng.integers(0, 5))
            elif pad_mode == 1:
                extra_h = int(rng.integers(3, 13))
                extra_w = int(rng.integers(3, 13))
            else:
                extra_h = int(rng.integers(0, 31 - minimum_height))
                extra_w = int(rng.integers(0, 31 - minimum_width))
            height, width = minimum_height + extra_h, minimum_width + extra_w
            if rng.random() < 0.22:
                width = height
        if minimum_height <= height <= 30 and minimum_width <= width <= 30:
            return height, width
    return max(minimum_height, 15), max(minimum_width, 15)


def _position(marker_row, marker_col, quadrant, distance_row, distance_col):
    row = (marker_row + 2 + distance_row if quadrant >= 2
           else marker_row - 1 - distance_row)
    col = (marker_col + 2 + distance_col if quadrant % 2
           else marker_col - 1 - distance_col)
    return row, col


def _put_copy(grid, motif, marker_row, marker_col, quadrant, payload):
    for dr, dc in motif:
        row, col = _position(marker_row, marker_col, quadrant, dr, dc)
        grid[row, col] = payload


def _pair_palette(rng, base, variable, pair_index):
    if not variable:
        return base
    if pair_index == 0:
        return base
    if pair_index == 1:
        return base[0], base[2], base[1]
    colors = rng.choice(10, size=3, replace=False)
    return tuple(int(c) for c in colors)


def _make_pair(rng, regime, palette, copy_count, first=False):
    for _ in range(300):
        if first:
            motif_height = int(rng.integers(3, 6))
            motif_width = int(rng.integers(3, 6))
        elif regime == "target_test" and rng.random() < 0.45:
            motif_height = int(rng.integers(3, 7))
            motif_width = int(rng.integers(3, 7))
            if max(motif_height, motif_width) < 5:
                motif_height = int(rng.integers(5, 7))
        else:
            motif_height = int(rng.integers(1, 7))
            motif_width = int(rng.integers(1, 7))
        motif = _sample_motif(rng, motif_height, motif_width, first)
        height, width = _sample_dimensions(rng, motif_height, motif_width)
        if height * width <= 2 * (4 * len(motif) + 4):
            continue
        top_padding = int(rng.integers(height - 2 * motif_height - 1))
        left_padding = int(rng.integers(width - 2 * motif_width - 1))
        marker_row = top_padding + motif_height
        marker_col = left_padding + motif_width
        background, marker, payload = palette
        input_grid = np.full((height, width), background, dtype=np.int8)
        input_grid[marker_row:marker_row + 2, marker_col:marker_col + 2] = marker
        quadrants = rng.choice(4, size=copy_count, replace=False)
        for quadrant in quadrants:
            _put_copy(input_grid, motif, marker_row, marker_col, int(quadrant), payload)
        output_grid = input_grid.copy()
        for quadrant in range(4):
            _put_copy(output_grid, motif, marker_row, marker_col, quadrant, payload)
        if _source_input_hash(input_grid) in _SOURCE_INPUTS:
            continue
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("unable to construct a valid pair")


def _quadrant_sets(grid, payload, marker_row, marker_col):
    groups = [set() for _ in range(4)]
    for raw_row, raw_col in np.argwhere(grid == payload):
        row, col = int(raw_row), int(raw_col)
        assert row < marker_row or row > marker_row + 1
        assert col < marker_col or col > marker_col + 1
        bottom, right = row > marker_row + 1, col > marker_col + 1
        dr = row - marker_row - 2 if bottom else marker_row - 1 - row
        dc = col - marker_col - 2 if right else marker_col - 1 - col
        groups[2 * bottom + right].add((dr, dc))
    return groups


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    input_colors = set(int(c) for c in np.unique(input_grid))
    colors, counts = np.unique(output_grid, return_counts=True)
    output_colors = set(int(c) for c in colors)
    assert len(input_colors) == 3 and input_colors == output_colors
    background = int(colors[int(np.argmax(counts))])
    assert list(counts).count(max(counts)) == 1
    candidates = []
    for color in output_colors - {background}:
        cells = {(int(r), int(c)) for r, c in np.argwhere(output_grid == color)}
        if len(cells) == 4:
            row = min(r for r, _ in cells)
            col = min(c for _, c in cells)
            if cells == {(row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)}:
                candidates.append((color, row, col))
    assert len(candidates) == 1
    marker, marker_row, marker_col = candidates[0]
    payload = next(iter(output_colors - {background, marker}))
    assert np.array_equal(input_grid == marker, output_grid == marker)
    source_groups = _quadrant_sets(input_grid, payload, marker_row, marker_col)
    result_groups = _quadrant_sets(output_grid, payload, marker_row, marker_col)
    occupied = tuple(i for i, group in enumerate(source_groups) if group)
    assert occupied
    motif = frozenset(source_groups[occupied[0]])
    motif_height = max(r for r, _ in motif) + 1
    motif_width = max(c for _, c in motif) + 1
    assert (0, 0) in motif
    assert 1 <= motif_height <= 6 and 1 <= motif_width <= 6
    assert not (motif_height == motif_width == 2 and len(motif) == 4)
    assert all(source_groups[i] == set(motif) for i in occupied)
    assert all(group == set(motif) for group in result_groups)
    for quadrant in range(4):
        for dr, dc in motif:
            row, col = _position(marker_row, marker_col, quadrant, dr, dc)
            assert 0 <= row < input_grid.shape[0] and 0 <= col < input_grid.shape[1]
    expected = input_grid.copy()
    for row, col in np.argwhere(input_grid == payload):
        for reflected_row in (int(row), 2 * marker_row + 1 - int(row)):
            for reflected_col in (int(col), 2 * marker_col + 1 - int(col)):
                assert expected[reflected_row, reflected_col] in (background, payload)
                expected[reflected_row, reflected_col] = payload
    assert np.array_equal(expected, output_grid)
    return {"palette": (background, marker, payload), "motif": motif,
            "motif_shape": (motif_height, motif_width), "quadrants": occupied,
            "marker": (marker_row, marker_col), "changed": not np.array_equal(input_grid, output_grid)}


def _valid_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= dim <= 30 for dim in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) and int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            _valid_grid(source)
            _valid_grid(target)
            input_key, pair_key = _grid_key(source), (_grid_key(source), _grid_key(target))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _source_input_hash(source) not in _SOURCE_INPUTS
            analyses[split].append(_analyze_pair(source, target))
    train = analyses["train"]
    test = analyses["test"]
    assert any(item["changed"] and len(item["quadrants"]) == 1
               and _asymmetric(item["motif"], *item["motif_shape"]) for item in train)
    assert any(not item["changed"] and len(item["quadrants"]) == 4 for item in train)
    train_palettes = {item["palette"] for item in train}
    all_palettes = {item["palette"] for item in train + test}
    assert len(all_palettes) == 1 or len(train_palettes) > 1
    train_motifs = {(item["motif_shape"], item["motif"]) for item in train}
    test_motifs = [(item["motif_shape"], item["motif"]) for item in test]
    assert len(set(test_motifs)) == len(test_motifs)
    assert not train_motifs.intersection(test_motifs)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    base = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    variable_palette = bool(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_motifs = set(), set()
    pair_index = 0

    def add(split, copy_count, first=False):
        nonlocal pair_index
        regime = "source_train" if split == "train" else "target_test"
        palette = _pair_palette(rng, base, variable_palette, pair_index)
        for _ in range(300):
            pair = _make_pair(rng, regime, palette, copy_count, first)
            key = _grid_key(pair["input"])
            analysis = _analyze_pair(pair["input"], pair["output"])
            motif_key = analysis["motif_shape"], analysis["motif"]
            if key in seen_inputs or (split == "test" and motif_key in seen_motifs):
                continue
            seen_inputs.add(key)
            seen_motifs.add(motif_key)
            puzzle[split].append(pair)
            pair_index += 1
            return
        raise RuntimeError("unable to find a distinct pair")

    # One clear asymmetric completion and one idempotent example establish
    # the rule even for the minimum legal training count.
    add("train", 1, True)
    add("train", 4)
    for _ in range(train_count - 2):
        add("train", int(rng.integers(1, 5)))
    rng.shuffle(puzzle["train"])
    for _ in range(test_count):
        add("test", int(rng.integers(1, 5)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(len(left[split]) == len(right[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(left[split], right[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, motifs, palettes, orientations = set(), set(), set(), set()
    copy_counts = {"train": set(), "test": set()}
    variable_puzzles = 0
    far_corner_missing = 0
    solid_patches = 0
    symmetric = 0
    disconnected = 0
    rectangular = 0
    short_bars = 0
    tight = 0
    roomy = 0
    roomy_off_center = 0
    combined = 0
    test_noops = 0
    role_swaps = 0
    fixed_puzzles = 0
    within_shapes = []
    within_motif_shapes = []
    source_shapes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_palettes = set()
        local_shapes = set()
        local_motif_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                analysis = _analyze_pair(source, target)
                assert source.shape == target.shape
                assert _source_input_hash(source) not in _SOURCE_INPUTS
                local_palettes.add(analysis["palette"])
                palettes.add(analysis["palette"])
                shapes.add(source.shape)
                local_shapes.add(source.shape)
                motifs.add((analysis["motif_shape"], analysis["motif"]))
                local_motif_shapes.add(analysis["motif_shape"])
                orientations.add((split, int(source.shape[0] > source.shape[1]) -
                                  int(source.shape[0] < source.shape[1])))
                copy_counts[split].add(len(analysis["quadrants"]))
                if source.shape in _SOURCE_SHAPES:
                    source_shapes.add(source.shape)
                h, w = analysis["motif_shape"]
                motif = analysis["motif"]
                rectangular += h != w
                short_bars += min(h, w) == 1
                disconnected += not _connected(motif)
                far_corner_missing += (h > 1 and (h - 1, 0) not in motif)
                far_corner_missing += (w > 1 and (0, w - 1) not in motif)
                symmetric += not _asymmetric(motif, h, w)
                solid_patches += any({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= motif
                                     for r in range(h - 1) for c in range(w - 1))
                marker_row, marker_col = analysis["marker"]
                edge_spaces = (marker_row - h, source.shape[0] - marker_row - h - 2,
                               marker_col - w, source.shape[1] - marker_col - w - 2)
                tight += min(edge_spaces) == 0
                roomy += max(edge_spaces) >= 6
                roomy_off_center += (max(edge_spaces) >= 6 and
                                     (abs(edge_spaces[0] - edge_spaces[1]) >= 3 or
                                      abs(edge_spaces[2] - edge_spaces[3]) >= 3))
                combined += (h != w and len(analysis["quadrants"]) in (2, 3)
                             and source.shape[0] > source.shape[1])
                test_noops += split == "test" and not analysis["changed"]
        variable_puzzles += len(local_palettes) > 1
        fixed_puzzles += len(local_palettes) == 1
        ordered_palettes = list(local_palettes)
        role_swaps += any(a[0] == b[0] and a[1] == b[2] and a[2] == b[1]
                          for a in ordered_palettes for b in ordered_palettes if a != b)
        within_shapes.append(len(local_shapes))
        within_motif_shapes.append(len(local_motif_shapes))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(motifs) >= 80 and len(palettes) >= 25
    assert len(orientations) == 6
    assert copy_counts["train"] == copy_counts["test"] == {1, 2, 3, 4}
    assert variable_puzzles >= 10
    assert fixed_puzzles >= 10 and role_swaps >= 10
    assert far_corner_missing >= 20 and symmetric >= 20 and solid_patches >= 10
    assert disconnected >= 5 and rectangular >= 80 and short_bars >= 10
    assert tight >= 30 and roomy >= 30 and roomy_off_center >= 20
    assert combined >= 5 and test_noops >= 5
    assert min(within_shapes) >= 2 and min(within_motif_shapes) >= 2
    assert len(source_shapes) == 4

    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 7, 3),
                              (2001, None, 1), (2002, None, 4), (2003, 2, None),
                              (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _puzzles_equal(generate_puzzle(3001, 8, 3), generate_puzzle(3001, 8, 3))
    assert not _puzzles_equal(generate_puzzle(3002, 8, 3), generate_puzzle(3003, 8, 3))
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4},
                    {"num_train": True, "num_test": 1},
                    {"num_train": 3, "num_test": False}):
        try:
            generate_puzzle(4001, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal count request: {request}")
