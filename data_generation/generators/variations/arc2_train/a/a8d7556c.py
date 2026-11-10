"""Generate dense two-colour grids with local 2-by-2 recolouring."""

import hashlib

import numpy as np


_AUTOMATIC_COUNTS = (
    (2, 1), (3, 1), (4, 2), (5, 3), (6, 4), (7, 1),
    (8, 2), (9, 3), (10, 4), (4, 1), (6, 2), (8, 3),
)
_SOURCE_PAIR_DIGESTS = {
    "632d55f4653f40b0b7983bfe5f4c1f9698197bd7017e5d6166758d40fd31277a",
    "2acd9a02c70a78436215f6fc614834325478531b4079eb03e5639ce8c15d94ce",
    "92cbeb550523cbc487e7d2b3cc2073d5534216b043569ca0da2a187be0e772d1",
    "6613a5743441d2d6f3543e737291be6c4f5c2693b69bc213e6b5eb4bd7c0121f",
}


def _optional_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _optional_count(num_train, "num_train")
    num_test = _optional_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be 2..10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be 1..4")
    if num_train is None and num_test is None:
        return _AUTOMATIC_COUNTS[seed % len(_AUTOMATIC_COUNTS)]
    if num_train is None:
        choices = range(max(2, num_test + 1), 11)
        num_train = choices[(seed + 3 * num_test) % len(choices)]
    if num_test is None:
        choices = range(1, min(4, num_train - 1) + 1)
        num_test = choices[(seed + 5 * num_train) % len(choices)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _window_tops(mask):
    return mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]


def _eligible_mask(mask):
    eligible = np.zeros_like(mask, dtype=bool)
    tops = _window_tops(mask)
    eligible[:-1, :-1] |= tops
    eligible[1:, :-1] |= tops
    eligible[:-1, 1:] |= tops
    eligible[1:, 1:] |= tops
    return eligible


def _apply_rule(grid, searched, highlight):
    output = grid.copy()
    output[_eligible_mask(grid == searched)] = highlight
    return output


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(component)
    return components


def _shape_of_component(component):
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    if len(component) != height * width:
        return None
    return height, width


def _adjacent(mask):
    result = np.zeros_like(mask, dtype=bool)
    result[:-1] |= mask[1:]
    result[1:] |= mask[:-1]
    result[:, :-1] |= mask[:, 1:]
    result[:, 1:] |= mask[:, :-1]
    return result


def _near_miss_next_to_change(searched, eligible):
    counts = (searched[:-1, :-1].astype(np.int8) + searched[1:, :-1]
              + searched[:-1, 1:] + searched[1:, 1:])
    adjacent_window = (eligible[:-1, :-1] | eligible[1:, :-1]
                       | eligible[:-1, 1:] | eligible[1:, 1:])
    unchanged = searched & ~eligible
    unchanged_window = (unchanged[:-1, :-1] | unchanged[1:, :-1]
                        | unchanged[:-1, 1:] | unchanged[1:, 1:])
    return bool(np.any((counts == 3) & adjacent_window & unchanged_window))


def _pattern(rng, kind=None):
    if kind is None:
        kind = int(rng.choice(5, p=[0.36, 0.29, 0.20, 0.10, 0.05]))
    if kind == 0:
        return np.ones((2, 2), dtype=bool)
    if kind == 1:
        length = int(rng.integers(3, 8))
        pattern = np.ones((2, length), dtype=bool)
    elif kind == 2:
        pattern = np.ones((4, 3), dtype=bool)
        pattern[0, 2] = False
        pattern[3, 2] = False  # A 4x2 bar with a 2x2 side overlap.
    elif kind == 3:
        length = int(rng.integers(3, 6))
        pattern = np.ones((3, length), dtype=bool)
    else:
        pattern = np.ones((2, int(rng.integers(4, 9))), dtype=bool)
    return np.rot90(pattern, int(rng.integers(4))).copy()


def _sample_shape(rng):
    if rng.random() < 0.16:
        return 18, 18
    return int(rng.integers(12, 28)), int(rng.integers(12, 28))


def _placement_candidates(mask, pattern, mode):
    height, width = mask.shape
    ph, pw = pattern.shape
    candidates = []
    for row in range(height - ph + 1):
        for col in range(width - pw + 1):
            if mode == "interior" and (row == 0 or col == 0 or row + ph == height or col + pw == width):
                continue
            if mode == "corner" and not ((row in (0, height - ph)) and (col in (0, width - pw))):
                continue
            if mode == "edge" and not (row == 0 or col == 0 or row + ph == height or col + pw == width):
                continue
            # Keep deliberately planted motifs distinct; the texture may still join them.
            if np.any(mask[max(0, row - 1):min(height, row + ph + 1),
                           max(0, col - 1):min(width, col + pw + 1)]):
                continue
            candidates.append((row, col))
    return candidates


def _add_tail(searched, rng):
    eligible = _eligible_mask(searched)
    unchanged = searched & ~eligible
    if np.any(unchanged & _adjacent(eligible)):
        return searched
    candidates = np.argwhere(~searched & _adjacent(eligible))
    if len(candidates):
        for index in rng.permutation(len(candidates)):
            row, col = map(int, candidates[int(index)])
            trial = searched.copy()
            trial[row, col] = True
            if not _eligible_mask(trial)[row, col]:
                return trial
    return searched


def _pair_features(searched):
    eligible = _eligible_mask(searched)
    components = _components(eligible)
    shapes = [_shape_of_component(c) for c in components]
    boundary = bool(np.any(eligible[0]) or np.any(eligible[-1])
                    or np.any(eligible[:, 0]) or np.any(eligible[:, -1]))
    corners = sum(bool(np.all(eligible[r:r+2, c:c+2]))
                  for r in (0, searched.shape[0]-2)
                  for c in (0, searched.shape[1]-2))
    side_pattern = np.ones((4, 3), dtype=bool)
    side_pattern[0, 2] = side_pattern[3, 2] = False
    side_overlap = False
    for turns in range(4):
        pattern = np.rot90(side_pattern, turns)
        ph, pw = pattern.shape
        matches = np.ones((searched.shape[0] - ph + 1,
                           searched.shape[1] - pw + 1), dtype=bool)
        for r in range(ph):
            for c in range(pw):
                matches &= searched[r:r+matches.shape[0], c:c+matches.shape[1]] == pattern[r, c]
        if np.any(matches):
            side_overlap = True
            break
    return {
        "regions": len(components),
        "isolated": (2, 2) in shapes,
        "bar": any(s is not None and ((s[0] == 2 and s[1] >= 3)
                                        or (s[1] == 2 and s[0] >= 3)) for s in shapes),
        "branch": any(s is None for s in shapes),
        "side_overlap": side_overlap,
        "boundary": boundary,
        "corner": corners > 0,
        "adjacent_unchanged": bool(np.any(searched & ~eligible & _adjacent(eligible))),
        "near_miss": _near_miss_next_to_change(searched, eligible),
    }


def _generate_pair(rng, roles, regime, required=None):
    searched_color, contrast_color, highlight_color = roles
    required = required or {}
    for _ in range(180):
        height, width = _sample_shape(rng)
        density = float(rng.uniform(0.25, 0.47))
        searched = rng.random((height, width)) < density
        # A free boundary is common; forcing an empty boundary creates true
        # interior-only examples without imposing a fixed train/test template.
        boundary_mode = str(rng.choice(
            ["free", "interior", "edge", "corner"],
            p=([0.26, 0.27, 0.25, 0.22] if regime == "source_train"
               else [0.18, 0.22, 0.30, 0.30]),
        ))
        if boundary_mode == "interior":
            searched[0] = False
            searched[-1] = False
            searched[:, 0] = False
            searched[:, -1] = False
        # For sparse pairs, zero motifs lets qualifying squares arise solely
        # from the random texture. Other pairs compose one to five motifs.
        motif_count = int(rng.choice(6, p=[0.17, 0.20, 0.22, 0.19, 0.14, 0.08]))
        if required.get("branch") or required.get("isolated") or boundary_mode in ("edge", "corner"):
            motif_count = max(1, motif_count)
        occupied = np.zeros_like(searched, dtype=bool)
        for motif_index in range(motif_count):
            kind = 2 if required.get("branch") and motif_index == 0 else (
                0 if required.get("isolated") and motif_index == 0 else None)
            pattern = _pattern(rng, kind)
            placement_mode = boundary_mode if motif_index == 0 and boundary_mode in ("corner", "edge") else "free"
            candidates = _placement_candidates(occupied, pattern, placement_mode)
            if not candidates:
                continue
            row, col = candidates[int(rng.integers(len(candidates)))]
            ph, pw = pattern.shape
            searched[row:row+ph, col:col+pw] = pattern
            occupied[row:row+ph, col:col+pw] |= pattern
        searched = _add_tail(searched, rng)
        eligible = _eligible_mask(searched)
        if not np.any(eligible):
            continue
        area = height * width
        if np.count_nonzero(searched) < area // 8 or np.count_nonzero(~searched) < area // 8:
            continue
        if np.count_nonzero(searched & ~eligible) < 8:
            continue
        if not np.any(_window_tops(~searched)):
            continue
        features = _pair_features(searched)
        if not features["adjacent_unchanged"] or not features["near_miss"]:
            continue
        if required.get("branch") and not (features["branch"] and features["side_overlap"]):
            continue
        if required.get("isolated") and not features["isolated"]:
            continue
        if boundary_mode == "interior" and features["boundary"]:
            continue
        if boundary_mode == "corner" and not features["corner"]:
            continue
        if boundary_mode == "edge" and not features["boundary"]:
            continue
        input_grid = np.where(searched, searched_color, contrast_color).astype(np.int8)
        output_grid = _apply_rule(input_grid, searched_color, highlight_color)
        return input_grid, output_grid
    raise RuntimeError("could not construct pair")


def _grid_key(grid):
    return (grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes())


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_pair_digest(pair):
    a, b = pair["input"], pair["output"]
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int8).tobytes()
                          + np.ascontiguousarray(b, dtype=np.int8).tobytes()).hexdigest()


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, b = pair["input"], pair["output"]
    for grid in (a, b):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 2 <= grid.shape[0] <= 30 and 2 <= grid.shape[1] <= 30
        assert np.all((grid >= 0) & (grid <= 9))
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.shape != (18, 18) or _source_pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
    changed = a != b
    assert np.any(changed)
    from_colors = np.unique(a[changed])
    to_colors = np.unique(b[changed])
    assert len(from_colors) == len(to_colors) == 1
    searched, highlight = int(from_colors[0]), int(to_colors[0])
    colors = set(map(int, np.unique(a)))
    assert len(colors) == 2 and searched in colors and highlight not in colors
    contrast = next(c for c in colors if c != searched)
    searched_mask = a == searched
    eligible = _eligible_mask(searched_mask)
    assert np.array_equal(changed, eligible)
    assert np.array_equal(b, _apply_rule(a, searched, highlight))
    assert np.count_nonzero(searched_mask & ~eligible) >= 8
    area = a.size
    assert np.count_nonzero(searched_mask) >= area // 8
    assert np.count_nonzero(~searched_mask) >= area // 8
    assert np.any(_window_tops(~searched_mask))
    features = _pair_features(searched_mask)
    assert features["adjacent_unchanged"] and features["near_miss"]
    return (searched, contrast, highlight), features


def _prediction_from_demonstrations(train_pairs, test_input):
    learned_roles = set()
    for pair in train_pairs:
        changed = pair["input"] != pair["output"]
        from_colors = np.unique(pair["input"][changed])
        to_colors = np.unique(pair["output"][changed])
        assert len(from_colors) == len(to_colors) == 1
        learned_roles.add((int(from_colors[0]), int(to_colors[0])))
    assert len(learned_roles) == 1
    searched, highlight = next(iter(learned_roles))
    prediction = test_input.copy()
    height, width = test_input.shape
    for row in range(height - 1):
        for col in range(width - 1):
            if (test_input[row, col] == searched
                    and test_input[row + 1, col] == searched
                    and test_input[row, col + 1] == searched
                    and test_input[row + 1, col + 1] == searched):
                prediction[row:row+2, col:col+2] = highlight
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys, pair_keys = set(), set()
    roles = None
    train_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            pair_roles, features = _inspect_pair(pair)
            if roles is None:
                roles = pair_roles
            assert pair_roles == roles
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            if split == "train":
                train_features.append(features)
    # The examples identify the changed input color, new highlight color,
    # local overlap convention, and unchanged searched-color tails.
    assert any(f["branch"] and f["side_overlap"] for f in train_features)
    assert any(f["isolated"] for f in train_features)
    for pair in puzzle["test"]:
        assert np.array_equal(
            _prediction_from_demonstrations(puzzle["train"], pair["input"]),
            pair["output"],
        )


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = rng.choice(np.arange(10), size=3, replace=False)
    roles = tuple(map(int, palette))
    child_seeds = rng.integers(0, 2**63 - 1, size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for index, child_seed in enumerate(child_seeds):
        split = "train" if index < num_train else "test"
        regime = "source_train" if split == "train" else "target_test"
        required = {"branch": True} if index == 0 else ({"isolated": True} if index == 1 else {})
        pair_rng = np.random.default_rng(int(child_seed))
        for _ in range(30):
            a, b = _generate_pair(pair_rng, roles, regime, required)
            key = _grid_key(a)
            if key not in used_inputs:
                used_inputs.add(key)
                puzzle[split].append({"input": a, "output": b})
                break
        else:
            raise RuntimeError("could not construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[s], second[s])) for s in ("train", "test"))


def self_test() -> None:
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(7301 + (counts[0] or 0) + (counts[1] or 0), *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, palettes, shapes = set(), set(), set(), set()
    region_counts, boundary_states, corner_states = set(), set(), set()
    fingerprints = set()
    component_vocabulary = set()
    interior_train = corner_train = interior_test = corner_test = 0
    five_regions = one_or_two_regions = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_pair_key(p) for s in ("train", "test") for p in puzzle[s]))
        puzzle_shapes = set()
        puzzle_regions = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                roles, features = _inspect_pair(pair)
                palettes.add(roles)
                shape = pair["input"].shape
                shapes.add(shape)
                puzzle_shapes.add(shape)
                puzzle_regions.add(features["regions"])
                region_counts.add(features["regions"])
                eligible = pair["input"] != pair["output"]
                for component in _components(eligible):
                    min_row = min(r for r, _ in component)
                    min_col = min(c for _, c in component)
                    component_vocabulary.add(tuple(sorted(
                        (r - min_row, c - min_col) for r, c in component
                    )))
                boundary_states.add(features["boundary"])
                corner_states.add(features["corner"])
                five_regions += features["regions"] >= 5
                one_or_two_regions += features["regions"] <= 2
                if split == "train":
                    interior_train += not features["boundary"]
                    corner_train += features["corner"]
                else:
                    interior_test += not features["boundary"]
                    corner_test += features["corner"]
        assert len(puzzle_shapes) >= 2
        assert len(puzzle_regions) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(palettes) >= 24
    assert len(shapes) >= 55 and len(region_counts) >= 5
    assert len(component_vocabulary) >= 100
    assert boundary_states == {False, True} and corner_states == {False, True}
    assert interior_train > 0 and corner_train > 0 and interior_test > 0 and corner_test > 0
    assert five_regions > 0 and one_or_two_regions > 0
