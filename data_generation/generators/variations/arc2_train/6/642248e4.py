"""Synthetic examples for the nearest-colored-boundary ARC rule."""

import numpy as np


_SOURCE_INPUT_TEXTS = frozenset({
    "2222222222/0000000000/0100000010/0001000000/0000000000/0010000000/0000001000/0001000000/0000000000/0100000000/0000001000/0000000010/0000000000/4444444444",
    "8888888888888/0000000000000/0010000010000/0000100000000/0000000000000/0000000001000/0000010000000/0000000000000/0010000000001/0000000010000/0000000000000/3333333333333",
    "3000000004/3000001004/3010000004/3000000004/3001000004/3000010004/3000000004/3000000004/3000001004/3000000004/3010000004/3000000004",
    "200000000000000108/201000000010000008/200000000000000008/200010000000000008/200000000000000008/200000000001000008/200000100000000008/200000000000000008/200000001000000008/200000000000000008/200010000000000008/200000000000001008/200000000010000008/200000000000000008/200100000000000108/200000000000000008",
})


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _grid_text(grid):
    return "/".join("".join(map(str, row)) for row in grid)


def _orientation_plan(rng, num_train, num_test):
    plan = int(rng.integers(0, 6))
    locked_tests = False
    if plan == 0:  # retain some source-like H, H, V -> V sequences
        train = ["horizontal"] * (num_train - 1) + ["vertical"]
        test = ["vertical"] * num_test
    elif plan == 1:
        train = ["horizontal"] * num_train
        locked_tests = rng.random() < 0.55
        test = (["horizontal"] * num_test if locked_tests else
                list(rng.choice(["horizontal", "vertical"], size=num_test)))
    elif plan == 2:
        train = ["vertical"] * num_train
        locked_tests = rng.random() < 0.55
        test = (["vertical"] * num_test if locked_tests else
                list(rng.choice(["horizontal", "vertical"], size=num_test)))
    else:
        train = list(rng.choice(["horizontal", "vertical"], size=num_train))
        first = "vertical" if plan == 4 else "horizontal"
        train[0] = first
        train[-1] = "horizontal" if first == "vertical" else "vertical"
        test = list(rng.choice(["horizontal", "vertical"], size=num_test))
    if num_test >= 2 and plan != 0 and not locked_tests and rng.random() < 0.45:
        test[0], test[1] = "horizontal", "vertical"
    return train, test


def _length(rng, regime):
    probabilities = (0.38, 0.40, 0.22) if regime == "source_train" else (0.24, 0.34, 0.42)
    band = int(rng.choice(3, p=probabilities))
    low, high = ((6, 10), (11, 18), (19, 30))[band]
    return int(rng.integers(low, high + 1))


def _side_axes(axis_length):
    positions = [p for p in range(2, axis_length - 2) if p != axis_length - 1 - p]
    return (
        [p for p in positions if p < axis_length - 1 - p],
        [p for p in positions if p > axis_length - 1 - p],
    )


def _side_capacity(positions, cross_length):
    return ((len(positions) + 1) // 2) * cross_length


def _sample_load(rng, axis_length, cross_length, regime, first_train):
    left, right = _side_axes(axis_length)
    capacities = [_side_capacity(left, cross_length), _side_capacity(right, cross_length)]
    area = (axis_length - 2) * cross_length
    density_band = int(rng.choice(3, p=(0.34, 0.39, 0.27)))
    low, high = ((0.025, 0.075), (0.075, 0.16), (0.16, 0.29))[density_band]
    count = max(1, round(area * float(rng.uniform(low, high))))
    count = min(count, 90, (area - 1) // 2, sum(capacities))
    if first_train:
        count = max(4, min(12, count))
        balance = "balanced"
    else:
        choices = ["balanced", "imbalanced", "one_side"]
        balance = str(rng.choice(choices, p=(0.46, 0.27, 0.27)))
        if regime == "target_test" and rng.random() < 0.13:
            count, balance = 1, "one_side"
    if balance == "one_side":
        chosen = int(rng.integers(0, 2))
        count = min(count, capacities[chosen])
        return count, (count, 0) if chosen == 0 else (0, count)
    if balance == "imbalanced":
        count = max(3, count)
        small = max(1, round(count * float(rng.uniform(0.08, 0.23))))
        counts = [small, count - small]
        if rng.random() < 0.5:
            counts.reverse()
    else:
        count = max(4 if first_train else 2, count)
        share = float(rng.uniform(0.35, 0.65))
        start = max(2 if first_train else 1, min(count - (2 if first_train else 1), round(count * share)))
        counts = [start, count - start]
    # A narrow strip can have less capacity on one side than the sampled load.
    counts = [min(counts[i], capacities[i]) for i in (0, 1)]
    return sum(counts), tuple(counts)


def _place_side(rng, positions, cross_length, count, style, axis_length):
    if count == 0:
        return []
    anchor_cross = int(rng.integers(0, cross_length))
    anchor_axis = int(rng.choice(positions))
    candidates = [(p, c) for p in positions for c in range(cross_length)]
    for attempt in range(5):
        scored = []
        for p, c in candidates:
            score = float(rng.random())
            if attempt < 3:
                if style == "cluster":
                    score += 0.8 * abs(c - anchor_cross) / max(1, cross_length - 1)
                elif style == "aligned":
                    score += 0.85 * abs(p - anchor_axis) / max(1, axis_length - 1)
                elif style == "middle":
                    score += 1.2 * abs(p - (axis_length - 1) / 2) / axis_length
            scored.append((score, p, c))
        scored.sort()
        occupied = set()
        chosen = []
        for _, p, c in scored:
            target = p - 1 if p < (axis_length - 1) / 2 else p + 1
            if (p, c) in occupied or (target, c) in occupied:
                continue
            chosen.append((p, c, target))
            occupied.add((p, c))
            occupied.add((target, c))
            if len(chosen) == count:
                return chosen
    # Alternating axis positions attain the side's maximum packing capacity.
    maximal = [(p, c) for p in positions[::2] for c in range(cross_length)]
    selected = rng.choice(len(maximal), size=count, replace=False)
    return [(maximal[int(i)][0], maximal[int(i)][1],
             maximal[int(i)][0] - 1 if maximal[int(i)][0] < (axis_length - 1) / 2
             else maximal[int(i)][0] + 1) for i in selected]


def _sample_placements(rng, axis_length, cross_length, side_counts):
    sides = _side_axes(axis_length)
    style = str(rng.choice(["scattered", "cluster", "aligned", "middle"],
                           p=(0.40, 0.20, 0.20, 0.20)))
    placements = []
    for side in (0, 1):
        for p, c, target in _place_side(rng, sides[side], cross_length,
                                        side_counts[side], style, axis_length):
            placements.append((p, c, target, side))
    return placements


def _palette(rng, previous, fixed_roles, force_role_change=False, mode=None):
    all_colors = list(range(10))
    prior_edges = [pair[2] for pair in previous]
    used_edges = {color for edge in prior_edges for color in edge}
    if mode is None:
        mode = str(rng.choice(["exact", "reverse", "recombine", "fresh"]))
    edge = None
    if previous and mode in ("exact", "reverse"):
        edge = prior_edges[int(rng.integers(0, len(prior_edges)))]
        if mode == "reverse":
            edge = edge[::-1]
    elif len(previous) >= 2 and mode == "recombine":
        options = [(a, b) for a in used_edges for b in used_edges
                   if a != b and frozenset((a, b)) not in
                   {frozenset(e) for e in prior_edges}]
        if fixed_roles is not None:
            options = [e for e in options if not set(e) & set(fixed_roles)]
        if options:
            edge = options[int(rng.integers(0, len(options)))]
    if edge is None:
        eligible = [c for c in all_colors if fixed_roles is None or c not in fixed_roles]
        unseen = [c for c in eligible if c not in used_edges]
        first = int(rng.choice(unseen if unseen else eligible))
        second = int(rng.choice([c for c in eligible if c != first]))
        edge = (first, second)
    available = [c for c in all_colors if c not in edge]
    if fixed_roles is not None:
        roles = fixed_roles
    elif not previous:
        roles = tuple(int(c) for c in rng.choice(available, size=2, replace=False))
    else:
        prior_roles = previous[-1][:2]
        favored = [c for c in prior_edges[-1] if c in available]
        if favored and rng.random() < 0.62:
            first = int(rng.choice(favored))
        else:
            first = int(rng.choice(available))
        second = int(rng.choice([c for c in available if c != first]))
        roles = (first, second)
        if force_role_change and roles == prior_roles:
            roles = (second, first)
    assert len(set((*roles, *edge))) == 4
    return (*roles, tuple(int(c) for c in edge))


def _make_pair(rng, regime, orientation, palette, first_train):
    axis_length = _length(rng, regime)
    cross_length = _length(rng, regime)
    height, width = ((axis_length, cross_length) if orientation == "horizontal"
                     else (cross_length, axis_length))
    _, side_counts = _sample_load(rng, axis_length, cross_length, regime, first_train)
    placements = _sample_placements(rng, axis_length, cross_length, side_counts)
    background, marker, edges = palette
    input_grid = np.full((height, width), background, dtype=np.int8)
    if orientation == "horizontal":
        input_grid[0, :] = edges[0]
        input_grid[-1, :] = edges[1]
    else:
        input_grid[:, 0] = edges[0]
        input_grid[:, -1] = edges[1]
    for p, c, _, _ in placements:
        if orientation == "horizontal":
            input_grid[p, c] = marker
        else:
            input_grid[c, p] = marker
    output_grid = input_grid.copy()
    for _, c, target, side in placements:
        if orientation == "horizontal":
            output_grid[target, c] = edges[side]
        else:
            output_grid[c, target] = edges[side]
    return {"input": input_grid, "output": output_grid}


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= n <= 30 for n in grid.shape)
        assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
    assert inp.shape == out.shape
    h, w = inp.shape
    horizontal = bool(np.all(inp[0] == inp[0, 0]) and np.all(inp[-1] == inp[-1, 0])
                      and inp[0, 0] != inp[-1, 0])
    vertical = bool(np.all(inp[:, 0] == inp[0, 0]) and np.all(inp[:, -1] == inp[0, -1])
                    and inp[0, 0] != inp[0, -1])
    assert horizontal != vertical
    orientation = "horizontal" if horizontal else "vertical"
    edges = (int(inp[0, 0]), int(inp[-1, 0]) if horizontal else int(inp[0, -1]))
    interior = inp[1:-1, :] if horizontal else inp[:, 1:-1]
    counts = sorted(((int(np.count_nonzero(interior == color)), int(color))
                     for color in np.unique(interior)), reverse=True)
    assert len(counts) == 2 and counts[0][0] > counts[1][0] > 0
    background, marker = counts[0][1], counts[1][1]
    assert len({background, marker, *edges}) == 4
    assert set(int(x) for x in np.unique(inp)) == {background, marker, *edges}
    positions = [tuple(int(x) for x in rc) for rc in np.argwhere(inp == marker)]
    expected = inp.copy()
    targets = set()
    sides = [0, 0]
    axis_length = h if horizontal else w
    for r, c in positions:
        p = r if horizontal else c
        assert 1 < p < axis_length - 2
        assert p != axis_length - 1 - p
        side = 0 if p < axis_length - 1 - p else 1
        target = (r + (-1 if side == 0 else 1), c) if horizontal else (r, c + (-1 if side == 0 else 1))
        assert target not in targets and int(inp[target]) == background
        targets.add(target)
        sides[side] += 1
        expected[target] = edges[side]
    assert np.array_equal(out, expected)
    assert len(targets) == int(np.count_nonzero(inp != out))
    return {"orientation": orientation, "size": (h, w), "roles": (background, marker),
            "edges": edges, "sides": tuple(sides), "markers": len(positions),
            "positions": tuple(positions)}


def _predict_from_training(train, test_input):
    # Infer the transformation from examples, then use only the test input.
    for pair in train:
        _inspect_pair(pair)
    inp = test_input
    h, w = inp.shape
    horizontal = bool(np.all(inp[0] == inp[0, 0]) and np.all(inp[-1] == inp[-1, 0])
                      and inp[0, 0] != inp[-1, 0])
    edges = (int(inp[0, 0]), int(inp[-1, 0]) if horizontal else int(inp[0, -1]))
    interior = inp[1:-1] if horizontal else inp[:, 1:-1]
    counts = sorted(((int(np.count_nonzero(interior == color)), int(color))
                     for color in np.unique(interior)), reverse=True)
    marker = counts[1][1]
    predicted = inp.copy()
    for r, c in np.argwhere(inp == marker):
        p = int(r if horizontal else c)
        side = 0 if p < (h if horizontal else w) - 1 - p else 1
        if horizontal:
            predicted[r + (-1 if side == 0 else 1), c] = edges[side]
        else:
            predicted[r, c + (-1 if side == 0 else 1)] = edges[side]
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos = {split: [_inspect_pair(pair) for pair in puzzle[split]] for split in ("train", "test")}
    inputs = set()
    pairs = set()
    for pair in puzzle["train"] + puzzle["test"]:
        key = _grid_key(pair["input"])
        both = (key, _grid_key(pair["output"]))
        assert key not in inputs and both not in pairs
        inputs.add(key)
        pairs.add(both)
        assert _grid_text(pair["input"]) not in _SOURCE_INPUT_TEXTS
    assert all(sum(info["sides"]) == info["markers"] for split in infos for info in infos[split])
    assert all(min(n for n in info["size"]) >= 6 for split in infos for info in infos[split])
    assert min(infos["train"][0]["sides"]) >= 2
    train_sides = [sum(info["sides"][side] for info in infos["train"]) for side in (0, 1)]
    assert min(train_sides) >= 2
    train_roles = {info["roles"] for info in infos["train"]}
    if len(train_roles) == 1:
        assert all(info["roles"] in train_roles for info in infos["test"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    train_orientations, test_orientations = _orientation_plan(rng, num_train, num_test)
    fixed_roles = tuple(int(c) for c in rng.choice(10, size=2, replace=False)) if rng.random() < 0.35 else None
    previous = []
    train, test = [], []
    test_modes = list(rng.permutation(["exact", "reverse", "recombine", "fresh"]))
    input_keys = set()
    for split, orientations in (("train", train_orientations), ("test", test_orientations)):
        for index, orientation in enumerate(orientations):
            mode = ("fresh" if index == 1 else None) if split == "train" else test_modes[index]
            palette = _palette(rng, previous if split == "train" else previous[:num_train],
                               fixed_roles,
                               force_role_change=(split == "train" and index == 1 and fixed_roles is None),
                               mode=mode)
            previous.append(palette)
            for _ in range(8):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  orientation, palette, split == "train" and index == 0)
                key = _grid_key(pair["input"])
                if key not in input_keys and _grid_text(pair["input"]) not in _SOURCE_INPUT_TEXTS:
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
            input_keys.add(key)
            (train if split == "train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[split], second[split])
                   for k in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(generate_puzzle(7001), generate_puzzle(7001))
    assert _puzzles_equal(generate_puzzle(7002, 7, 3), generate_puzzle(7002, 7, 3))
    assert not _puzzles_equal(generate_puzzle(7002, 7, 3), generate_puzzle(7003, 7, 3))
    for seed, nt, ne in ((90, 2, 1), (91, 10, 1), (92, 10, 4),
                         (80, 5, 2), (81, 8, 3)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
    for nt, ne in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(100, nt, ne)
        assert (nt is None or len(puzzle["train"]) == nt)
        assert (ne is None or len(puzzle["test"]) == ne)
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(200, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    infos = [(split, _inspect_pair(pair)) for puzzle in automatic
             for split in ("train", "test") for pair in puzzle[split]]
    assert {info["orientation"] for _, info in infos} == {"horizontal", "vertical"}
    assert len({(info["orientation"], info["size"]) for _, info in infos}) >= 60
    assert len({info["markers"] for _, info in infos}) >= 20
    assert len({info["positions"] for _, info in infos}) >= 120
    assert len({info["roles"] for _, info in infos}) >= 20
    assert min(info["markers"] for _, info in infos) == 1
    assert max(info["markers"] for _, info in infos) >= 40
    assert any(info["size"] in ((14, 10), (12, 13), (12, 10), (16, 18))
               for _, info in infos)

    all_horizontal = all_vertical = complete_horizontal = complete_vertical = 0
    vertical_first = mixed_tests = 0
    horizontal_tests = one_sided_tests = single_tests = 0
    smaller_tests = dense_pairs = sparse_pairs = aligned_pairs = clustered_pairs = 0
    varied_sizes = varied_loads = 0
    varied_roles = role_switches = exact = reverse = recombined = fresh = 0
    for puzzle in automatic:
        train = [_inspect_pair(pair) for pair in puzzle["train"]]
        tests = [_inspect_pair(pair) for pair in puzzle["test"]]
        orientations = {item["orientation"] for item in train}
        all_horizontal += orientations == {"horizontal"}
        all_vertical += orientations == {"vertical"}
        complete = {item["orientation"] for item in train + tests}
        complete_horizontal += complete == {"horizontal"}
        complete_vertical += complete == {"vertical"}
        vertical_first += train[0]["orientation"] == "vertical"
        mixed_tests += len({item["orientation"] for item in tests}) == 2
        varied_roles += len({item["roles"] for item in train}) > 1
        all_infos = train + tests
        role_switches += any(set(item["roles"]) &
                             {color for earlier in all_infos[:index] for color in earlier["edges"]}
                             for index, item in enumerate(all_infos) if index > 0)
        varied_sizes += len({item["size"] for item in train}) > 1
        varied_loads += len({item["markers"] for item in train}) > 1
        train_edges = {item["edges"] for item in train}
        train_edge_colors = {color for edge in train_edges for color in edge}
        max_train_size = tuple(max(item["size"][dimension] for item in train)
                               for dimension in (0, 1))
        for item in train + tests:
            h, w = item["size"]
            axis = h if item["orientation"] == "horizontal" else w
            cross = w if item["orientation"] == "horizontal" else h
            density = item["markers"] / ((axis - 2) * cross)
            dense_pairs += density >= 0.16
            sparse_pairs += density <= 0.075
            axis_positions = [r if item["orientation"] == "horizontal" else c
                              for r, c in item["positions"]]
            cross_positions = [c if item["orientation"] == "horizontal" else r
                               for r, c in item["positions"]]
            aligned_pairs += (max(axis_positions.count(p) for p in set(axis_positions))
                              >= max(2, round(item["markers"] * 0.35)))
            clustered_pairs += (item["markers"] >= 4 and
                                len(set(cross_positions)) <= min(cross, round(item["markers"] * 0.5)))
        for item in tests:
            horizontal_tests += item["orientation"] == "horizontal"
            one_sided_tests += min(item["sides"]) == 0
            single_tests += item["markers"] == 1
            smaller_tests += all(item["size"][i] < max_train_size[i] for i in (0, 1))
            edge = item["edges"]
            exact += edge in train_edges
            reverse += edge[::-1] in train_edges
            recombined += (set(edge) <= train_edge_colors and
                           frozenset(edge) not in {frozenset(e) for e in train_edges})
            fresh += bool(set(edge) - train_edge_colors)
    assert all_horizontal >= 3 and all_vertical >= 3 and vertical_first >= 8
    assert complete_horizontal >= 2 and complete_vertical >= 2
    assert mixed_tests >= 8 and horizontal_tests >= 10
    assert one_sided_tests >= 10 and single_tests >= 3 and smaller_tests >= 10
    assert dense_pairs >= 30 and sparse_pairs >= 30 and varied_roles >= 10
    assert role_switches >= 10
    assert aligned_pairs >= 40 and clustered_pairs >= 25
    assert varied_sizes >= 30 and varied_loads >= 30
    assert min(exact, reverse, recombined, fresh) >= 5
