import numpy as np


_SOURCE_INPUT_ROWS = {
    (
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000003000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0002000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
    ),
    (
        "000000",
        "000300",
        "000000",
        "000000",
        "000000",
        "000000",
    ),
    (
        "000000",
        "000000",
        "008000",
        "000000",
        "000000",
        "000000",
    ),
    (
        "000000000000",
        "000000000000",
        "000000000000",
        "000000000000",
        "000008000000",
        "030000000000",
        "000000002000",
        "000000000000",
        "000000000000",
        "000000000000",
        "000000000000",
        "000000000000",
    ),
    (
        "0000000000000000",
        "0300000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000200",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0080000000000000",
        "0000000000000000",
    ),
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4) % (10 - test)
        if test == 1 and (int(seed) // 4) % 3 == 0:
            train = 2
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _edge(p, h, w):
    return p[0] in (1, h - 2) or p[1] in (1, w - 2)


def _corner(p, h, w):
    return p[0] in (1, h - 2) and p[1] in (1, w - 2)


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _mode_dimensions(rng, mode, regime, minimum=6):
    if mode == "dense":
        return int(rng.integers(18, 29)), int(rng.integers(18, 29))
    if mode == "multi_edge":
        return int(rng.integers(12, 25)), int(rng.integers(12, 25))
    if mode == "cluster":
        return int(rng.integers(10, 23)), int(rng.integers(10, 23))
    shape = int(rng.integers(0, 4 if regime == "target_test" else 3))
    if shape == 0:
        h, w = int(rng.integers(6, 11)), int(rng.integers(6, 11))
    elif shape == 1:
        h, w = int(rng.integers(9, 18)), int(rng.integers(9, 18))
    elif shape == 2:
        short = int(rng.integers(6, 12))
        long = int(rng.integers(16, 26 if regime == "target_test" else 20))
        h, w = (short, long) if rng.integers(2) else (long, short)
    else:
        h, w = int(rng.integers(17, 29)), int(rng.integers(17, 29))
    return max(h, minimum), max(w, minimum)


def _candidates(h, w, mode):
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if mode == "interior":
        return [p for p in cells if not _edge(p, h, w)]
    return cells


def _place(rng, h, w, count, mode, separation=3):
    cells = _candidates(h, w, mode)
    if mode in ("corner", "multi_edge"):
        first_pool = [p for p in cells if _corner(p, h, w)]
    elif mode == "edge":
        first_pool = [p for p in cells if _edge(p, h, w) and not _corner(p, h, w)]
    else:
        first_pool = cells
    for _ in range(40):
        positions = [first_pool[int(rng.integers(len(first_pool)))]]
        while len(positions) < count:
            choices = [p for p in cells if all(_distance(p, q) >= separation for q in positions)]
            first = positions[0]
            if mode == "multi_edge" and len(positions) == 1:
                choices = [p for p in choices if _edge(p, h, w) and _distance(p, first) >= max(h, w) // 2]
            elif mode == "touch_edge" and len(positions) == 1:
                choices = [p for p in choices if (abs(p[0] - first[0]), abs(p[1] - first[1])) in ((3, 0), (0, 3))]
            elif mode == "touch_corner" and len(positions) == 1:
                choices = [p for p in choices if abs(p[0] - first[0]) == 3 and abs(p[1] - first[1]) == 3]
            elif mode == "cluster":
                choices = [p for p in choices if min(_distance(p, q) for q in positions) == 3]
            if not choices:
                break
            if mode in ("multi_edge", "dense", "free"):
                scores = [min(_distance(p, q) for q in positions) for p in choices]
                if mode == "multi_edge":
                    threshold = max(scores) - 2
                    choices = [p for p, score in zip(choices, scores) if score >= threshold]
                elif mode == "dense":
                    threshold = min(scores) + 2
                    choices = [p for p, score in zip(choices, scores) if score <= threshold]
            positions.append(choices[int(rng.integers(len(choices)))])
        if len(positions) == count:
            return tuple(positions)
    lattice = [(r, c) for r in range(1, h - 1, separation) for c in range(1, w - 1, separation)]
    if mode == "interior":
        lattice = [p for p in lattice if not _edge(p, h, w)]
    if len(lattice) >= count and mode in ("free", "interior", "dense"):
        indices = rng.permutation(len(lattice))[:count]
        return tuple(lattice[int(i)] for i in indices)
    raise RuntimeError(f"placement failed: {mode}, {h}x{w}, {count}")


def _fallback_geometry(mode, count, index):
    # Large, pair-unique canvases make every legal count constructible.
    h, w = 20 + index % 9, 20 + index // 9
    lattice = [(r, c) for r in (4, 7, 10, 13, 16)
               for c in (4, 7, 10, 13, 16)]
    if mode == "free":
        lattice = [(r, c) for r in (3, 7, 11, 15)
                   for c in (3, 7, 11, 15)]
    if mode == "corner":
        initial = [(1, 1)]
    elif mode == "edge":
        initial = [(1, 4)]
    elif mode == "multi_edge":
        initial = [(1, 1), (h - 2, w - 2)]
    elif mode == "touch_edge":
        initial = [(4, 4), (4, 7)]
    elif mode == "touch_corner":
        initial = [(4, 4), (7, 7)]
    elif mode == "cluster":
        connected = [(4, 4), (4, 7), (7, 4), (7, 7), (4, 10),
                     (10, 4), (7, 10), (10, 7), (10, 10)]
        return h, w, tuple(connected[:count])
    else:
        initial = []
    positions = list(initial)
    for position in lattice:
        if len(positions) == count:
            break
        if all(_distance(position, other) >= 3 for other in positions):
            positions.append(position)
    assert len(positions) == count
    return h, w, tuple(positions)


def _render(h, w, positions, roles, background, seeds, halos):
    input_grid = np.full((h, w), background, dtype=np.int8)
    output_grid = input_grid.copy()
    for (r, c), role in zip(positions, roles):
        input_grid[r, c] = seeds[role]
        output_grid[r - 1:r + 2, c - 1:c + 2] = halos[role]
        output_grid[r, c] = seeds[role]
    return {"input": input_grid, "output": output_grid}


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _source_rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(rng, seed, num_train, num_test)
    n_roles = 2 + int(rng.integers(3))
    palette = [int(v) for v in rng.permutation(10)]
    seeds = tuple(palette[:n_roles])
    halos = tuple(palette[n_roles:2 * n_roles])
    backgrounds = tuple(palette[2 * n_roles:])

    # Randomly ordered clean demonstrations cover the complete shared map.
    plans = [None] * n_train
    anchor_indices = [int(i) for i in rng.permutation(n_train)[:min(n_roles, n_train)]]
    shuffled_roles = [int(i) for i in rng.permutation(n_roles)]
    for j, role in enumerate(shuffled_roles):
        index = anchor_indices[j % len(anchor_indices)]
        if plans[index] is None:
            plans[index] = []
        plans[index].append(role)
    free_indices = [i for i in range(n_train) if plans[i] is None]
    mixed_index = int(rng.choice(free_indices if free_indices else anchor_indices))
    for i in range(n_train):
        if plans[i] is None:
            size = int(rng.integers(1, n_roles + 1))
            plans[i] = [int(v) for v in rng.choice(n_roles, size=size, replace=False)]
    plans[mixed_index] = list(dict.fromkeys(plans[mixed_index] + shuffled_roles))

    total = n_train + n_test
    pair_seeds = rng.integers(0, 2**63 - 1, size=total, dtype=np.int64)
    bg_choices = [int(rng.choice(backgrounds)) for _ in range(total)]
    if len(set(bg_choices)) == 1:
        bg_choices[-1] = next(v for v in backgrounds if v != bg_choices[0])

    puzzle = {"train": [], "test": []}
    seen = set()
    modes = ("interior", "edge", "corner", "touch_edge", "touch_corner", "cluster", "multi_edge", "dense")
    for index in range(total):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        training = index < n_train
        if training:
            anchor = index in anchor_indices
            base_roles = list(plans[index])
            if not anchor:
                extra = int(pair_rng.integers(0, 5))
                base_roles.extend(int(v) for v in pair_rng.choice(n_roles, size=extra))
            mode = "free" if anchor else modes[int(pair_rng.integers(len(modes)))]
            regime = "source_train"
        else:
            anchor = False
            base_roles = []
            mode = modes[(int(seed) + index - n_train) % len(modes)] if index == n_train else modes[int(pair_rng.integers(len(modes)))]
            regime = "target_test"

        for attempt in range(48):
            minimum = max(6, 4 + 2 * len(base_roles)) if anchor else 6
            h, w = _mode_dimensions(pair_rng, mode, regime, minimum)
            if training:
                count = len(base_roles)
                if not anchor:
                    count = max(count, 2 if mode.startswith("touch") else 1,
                                3 if mode in ("cluster", "multi_edge") else 1,
                                5 if mode == "dense" else 1)
            else:
                floor = 5 if mode == "dense" else 3 if mode in ("cluster", "multi_edge") else 2 if mode.startswith("touch") else 1
                count = int(pair_rng.integers(floor, 10 if mode == "dense" else 7))
            if mode == "interior":
                capacity = max(1, ((h - 4 + 2) // 3) * ((w - 4 + 2) // 3))
                count = min(count, capacity)
            if mode != "dense" and h * w < 100:
                count = min(count, 3 if min(h, w) >= 8 else 2)
            if training and len(base_roles) > count:
                continue
            try:
                positions = _place(pair_rng, h, w, count, mode, 4 if anchor else 3)
            except RuntimeError:
                continue
            if training:
                roles = base_roles + [int(v) for v in pair_rng.choice(n_roles, size=count - len(base_roles))]
            else:
                subset_size = int(pair_rng.integers(1, min(n_roles, count) + 1))
                if count >= n_roles and pair_rng.integers(3) == 0:
                    subset_size = n_roles
                subset = [int(v) for v in pair_rng.choice(n_roles, size=subset_size, replace=False)]
                roles = subset + [int(v) for v in pair_rng.choice(subset, size=count - subset_size)]
            pair_rng.shuffle(roles)
            pair = _render(h, w, positions, roles, bg_choices[index], seeds, halos)
            sig = _signature(pair["input"])
            if sig in seen or _source_rows(pair["input"]) in _SOURCE_INPUT_ROWS:
                continue
            seen.add(sig)
            puzzle["train" if training else "test"].append(pair)
            break
        else:
            h, w, positions = _fallback_geometry(mode, count, index)
            if training:
                roles = base_roles + [int(v) for v in pair_rng.choice(n_roles, size=count - len(base_roles))]
            else:
                subset_size = min(n_roles, count)
                subset = [int(v) for v in pair_rng.choice(n_roles, size=subset_size, replace=False)]
                roles = subset + [int(v) for v in pair_rng.choice(subset, size=count - subset_size)]
            for extra_width in range(9):
                pair = _render(h, w + extra_width, positions, roles, bg_choices[index], seeds, halos)
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle["train" if training else "test"].append(pair)
                    break
            else:
                raise RuntimeError(f"unable to construct unique pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, b = pair["input"], pair["output"]
    for grid in (a, b):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
    assert a.shape == b.shape and not np.array_equal(a, b)
    values, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == max(counts)) == 1
    background = int(values[int(np.argmax(counts))])
    locations = [(int(r), int(c)) for r, c in np.argwhere(a != background)]
    assert 1 <= len(locations) <= 10
    h, w = a.shape
    assert all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in locations)
    assert all(_distance(p, q) >= 3 for i, p in enumerate(locations) for q in locations[i + 1:])
    expected = np.full_like(a, background)
    mapping = {}
    for r, c in locations:
        seed = int(a[r, c])
        patch = b[r - 1:r + 2, c - 1:c + 2]
        assert int(patch[1, 1]) == seed
        ring = patch.copy()
        ring[1, 1] = patch[0, 0]
        assert np.all(ring == ring[0, 0])
        halo = int(ring[0, 0])
        assert halo != seed and halo != background
        if seed in mapping:
            assert mapping[seed] == halo
        mapping[seed] = halo
        expected[r - 1:r + 2, c - 1:c + 2] = halo
        expected[r, c] = seed
    assert np.array_equal(b, expected)
    return background, mapping, locations


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen = set()
        learned = {}
        infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                sig = _signature(pair["input"])
                assert sig not in seen
                seen.add(sig)
                assert _source_rows(pair["input"]) not in _SOURCE_INPUT_ROWS
                background, mapping, locations = _inspect(pair)
                infos[split].append((background, mapping, locations, pair))
                if split == "train":
                    for seed, halo in mapping.items():
                        if seed in learned:
                            assert learned[seed] == halo
                        learned[seed] = halo
        assert 2 <= len(learned) <= 4
        assert len(set(learned.values())) == len(learned)
        assert set(learned).isdisjoint(learned.values())
        assert len({item[0] for split in infos.values() for item in split}) >= 2
        clean_demonstrations = set()
        for _, _, locations, pair in infos["train"]:
            for position in locations:
                if all(_distance(position, other) > 3 for other in locations if other != position):
                    clean_demonstrations.add(int(pair["input"][position]))
        assert clean_demonstrations == set(learned)
        for split in ("train", "test"):
            for background, mapping, locations, pair in infos[split]:
                assert background not in learned and background not in learned.values()
                assert all(seed in learned and learned[seed] == halo for seed, halo in mapping.items())
                reconstruction = np.full_like(pair["input"], background)
                for r, c in locations:
                    seed = int(pair["input"][r, c])
                    reconstruction[r - 1:r + 2, c - 1:c + 2] = learned[seed]
                    reconstruction[r, c] = seed
                assert np.array_equal(reconstruction, pair["output"])
        assert any(len(info[1]) >= 2 for info in infos["train"])
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid puzzle: {exc}") from exc


def _fingerprint(puzzle):
    return tuple((split, tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _fingerprint(generate_puzzle(17)) == _fingerprint(puzzles[17])
    assert _fingerprint(generate_puzzle(123, 7, 3)) == _fingerprint(generate_puzzle(123, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1000 + (train or 0) * 10 + (test or 0), train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(900, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    roles, backgrounds, shapes, layouts, counts, categories = set(), set(), set(), set(), set(), set()
    observations = set()
    first_training_counts = set()
    varied_training_counts = 0
    varied_training_shapes = 0
    source_like = 0
    corner_train_interior_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        learned = {}
        for pair in puzzle["train"]:
            _, mapping, _ = _inspect(pair)
            learned.update(mapping)
        roles.add(len(learned))
        train_info = [_inspect(pair) for pair in puzzle["train"]]
        first_training_counts.add(len(train_info[0][2]))
        varied_training_counts += len({len(item[2]) for item in train_info}) >= 2
        varied_training_shapes += len({pair["input"].shape for pair in puzzle["train"]}) >= 2
        source_like += (sum(len(item[2]) == 1 for item in train_info) >= 2
                        and any(len(item[1]) >= 2 for item in train_info))
        train_corner = any(
            any(_corner(pos, *pair["input"].shape) for pos in info[2])
            for pair, info in zip(puzzle["train"], train_info)
        )
        test_interior = any(
            all(not _edge(pos, *pair["input"].shape) for pos in _inspect(pair)[2])
            for pair in puzzle["test"]
        )
        corner_train_interior_test += train_corner and test_interior
        for split in ("train", "test"):
            for pair in puzzle[split]:
                background, mapping, loc = _inspect(pair)
                h, w = pair["input"].shape
                backgrounds.add(background)
                shapes.add((h, w))
                layouts.add((h, w, tuple(loc)))
                counts.add(len(loc))
                assert all(learned[s] == halo for s, halo in mapping.items())
                touching_pairs = [(a, b) for i, a in enumerate(loc) for b in loc[i + 1:]
                                  if _distance(a, b) == 3]
                same_touch = [(a, b) for a, b in touching_pairs
                              if pair["input"][a] == pair["input"][b]]
                if split == "train" and any(_corner(p, h, w) for p in loc):
                    observations.add("train_corner")
                if split == "test":
                    edge = any(_edge(p, h, w) for p in loc)
                    corner = any(_corner(p, h, w) for p in loc)
                    touching = bool(touching_pairs)
                    repeated = len(loc) > len(mapping)
                    categories.add((edge, corner, touching, repeated))
                    if not edge:
                        observations.add("test_interior")
                    if edge and not corner:
                        observations.add("test_edge_no_corner")
                    if corner:
                        observations.add("test_corner")
                    if touching:
                        observations.add("test_touch")
                    if len(loc) >= 3 and touching:
                        observations.add("test_cluster")
                    if sum(_edge(p, h, w) for p in loc) >= 2 and corner and len(loc) >= 3:
                        observations.add("test_multi_edge")
                    if len(mapping) < len(learned):
                        observations.add("test_role_subset")
                    if len(loc) > len(learned):
                        observations.add("test_more_seeds_than_roles")
                    if any((abs(a[0]-b[0]), abs(a[1]-b[1])) in ((3, 0), (0, 3))
                           for a, b in same_touch):
                        observations.add("same_role_edge_touch")
                    if any(abs(a[0]-b[0]) == 3 and abs(a[1]-b[1]) == 3
                           for a, b in same_touch):
                        observations.add("same_role_corner_touch")
                    if max(h, w) <= 10:
                        observations.add("small_test")
                    if max(h, w) >= 2 * min(h, w):
                        observations.add("elongated_test")
                    if min(h, w) >= 18 and len(loc) >= 5:
                        observations.add("large_dense_test")
                if background == 0:
                    observations.add("zero_background")
                if 0 in mapping:
                    observations.add("zero_seed")
                if 0 in mapping.values():
                    observations.add("zero_halo")
    assert roles == {2, 3, 4}
    assert len(backgrounds) >= 8 and len(shapes) >= 60 and len(layouts) >= 100
    assert max(counts) >= 7 and len(categories) >= 7
    assert 1 in first_training_counts and max(first_training_counts) >= 3
    assert varied_training_counts >= 35 and varied_training_shapes >= 35
    assert source_like >= 10 and corner_train_interior_test >= 5
    assert observations == {
        "train_corner", "test_interior", "test_edge_no_corner", "test_corner",
        "test_touch", "test_cluster", "test_multi_edge", "test_role_subset",
        "test_more_seeds_than_roles", "same_role_edge_touch",
        "same_role_corner_touch", "small_test", "elongated_test",
        "large_dense_test", "zero_background", "zero_seed", "zero_halo",
    }
