"""Generate ARC puzzles whose cells are enlarged by the foreground-cell count."""

import numpy as np


SOURCE_INPUTS = {
    (0, 4, 0, 0, 0, 8, 6, 0, 0),
    (2, 0, 0, 0, 0, 7, 0, 0, 0),
    (0, 6, 9, 3, 0, 2, 0, 7, 0),
    (1, 0, 0, 0, 9, 6, 8, 0, 0),
}
SHAPES = ((3, 3), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3))


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1 and int(counts[winners[0]]) >= 2
    return int(colors[winners[0]])


def _output(grid: np.ndarray, background: int) -> np.ndarray:
    k = int(np.count_nonzero(grid != background))
    return np.repeat(np.repeat(grid, k, axis=0), k, axis=1)


def _validate_count(value: int | None, name: str, lo: int, hi: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")


def _pair_counts(seed: int, train: int | None, test: int | None) -> tuple[int, int]:
    _validate_count(train, "num_train", 2, 10)
    _validate_count(test, "num_test", 1, 4)
    if test is None:
        choices = list(range(1, min(4, int(train) - 1) + 1)) if train is not None else [1, 2, 3, 4]
        test = choices[seed % len(choices)]
    if train is None:
        choices = list(range(max(2, int(test) + 1), 11))
        train = choices[(seed // 4) % len(choices)]
    if int(train) <= int(test):
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _sample_counts(rng: np.random.Generator, ntrain: int, ntest: int) -> tuple[list[int], list[int]]:
    source_like = bool(rng.random() < 0.22)
    if source_like:
        base = [2, 5] if ntrain == 2 else [2, 3, 5]
        train = base + [int(rng.choice(base)) for _ in range(ntrain - len(base))]
        test = [4] * ntest
    else:
        first = int(rng.integers(2, 7))  # a repeat-color witness fits 3 by 3
        second = int(rng.choice([k for k in range(2, 8) if k != first]))
        distinct = [first, second]
        extra_distinct = min(ntrain, int(rng.integers(2, 5)))
        remaining = [k for k in range(2, 8) if k not in distinct]
        if extra_distinct > 2:
            distinct.extend(int(x) for x in rng.choice(remaining, size=extra_distinct - 2, replace=False))
        train = distinct + [int(rng.choice(distinct)) for _ in range(ntrain - len(distinct))]
        seen = sorted(set(train))
        unseen = [k for k in range(2, 8) if k not in seen]
        if rng.random() < 0.13:
            unseen.append(1)
        test = []
        for i in range(ntest):
            use_unseen = bool(rng.random() < 0.55) if ntest == 1 else i % 2 == 1
            pool = unseen if use_unseen and unseen else seen
            test.append(int(rng.choice(pool)))
        if rng.random() < 0.08:
            test[-1] = 1
    return train, test


def _compatible_shapes(k: int) -> list[tuple[int, int]]:
    return [shape for shape in SHAPES if max(shape) * k <= 30 and shape[0] * shape[1] - k >= 2]


def _choose_shape(
    rng: np.random.Generator,
    k: int,
    extended: bool,
    force_extended: bool = False,
    familiar_count: int | None = None,
    novelty: int = 0,
) -> tuple[int, int]:
    choices = _compatible_shapes(k)
    if familiar_count is not None:
        choices = [
            shape for shape in choices
            if 1 + (k - 1) // (shape[0] * shape[1] - k - 1)
            <= familiar_count + novelty
        ]
        assert choices
    if not extended:
        return (3, 3) if (3, 3) in choices else choices[int(rng.integers(len(choices)))]
    nonstandard = [shape for shape in choices if shape != (3, 3)]
    if nonstandard and (force_extended or (3, 3) not in choices or rng.random() < 0.64):
        return nonstandard[int(rng.integers(len(nonstandard)))]
    return (3, 3)


def _make_colors(
    rng: np.random.Generator,
    k: int,
    bg_cells: int,
    familiar: list[int],
    novel: list[int],
    novel_count: int,
    force_repeat: bool,
) -> np.ndarray:
    cap = bg_cells - 1
    min_distinct = max((k + cap - 1) // cap, novel_count, 1)
    max_distinct = min(k, len(familiar) + novel_count)
    if force_repeat:
        max_distinct = min(max_distinct, k - 1)
    if max_distinct < min_distinct:
        raise ValueError("Palette cannot preserve a unique dominant background")
    if force_repeat:
        distinct = max_distinct
    elif max_distinct > min_distinct and rng.random() < 0.47:
        distinct = int(rng.integers(min_distinct, max_distinct))
    else:
        distinct = max_distinct
    chosen = [int(x) for x in rng.choice(novel, size=novel_count, replace=False)] if novel_count else []
    chosen.extend(int(x) for x in rng.choice(familiar, size=distinct - novel_count, replace=False))
    values = list(chosen)
    counts = {color: 1 for color in chosen}
    while len(values) < k:
        candidates = [color for color in chosen if counts[color] < cap]
        color = int(rng.choice(candidates))
        values.append(color)
        counts[color] += 1
    rng.shuffle(values)
    return np.asarray(values, dtype=np.int8)


def _make_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    k: int,
    train_pool: tuple[int, ...],
    split: str,
    seen_inputs: set[tuple],
    prior_masks: dict[tuple[tuple[int, int], int], list[tuple[int, ...]]],
    novelty: int = 0,
    witness: bool = False,
    avoid_background: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    area = shape[0] * shape[1]
    bg_cells = area - k
    group = (shape, k)
    for attempt in range(128):
        backgrounds = [c for c in range(10) if c != avoid_background]
        if shape == (3, 3) and k == 7:
            backgrounds = [c for c in backgrounds if c not in train_pool]
        background = int(rng.choice(backgrounds))
        familiar = [c for c in train_pool if c != background]
        novel = [c for c in range(10) if c not in train_pool and c != background]
        requested_novel = min(novelty, k, len(novel)) if split == "test" else 0
        if split == "train":
            novel = []
        if witness and bg_cells < 3:
            raise ValueError("A repeated-color witness needs three background cells")
        reuse = bool(prior_masks.get(group)) and rng.random() < 0.42
        if reuse:
            positions = prior_masks[group][int(rng.integers(len(prior_masks[group])))]
        else:
            positions = tuple(sorted(int(x) for x in rng.choice(area, size=k, replace=False)))
        try:
            colors = _make_colors(rng, k, bg_cells, familiar, novel, requested_novel, witness)
        except ValueError:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        for pos, color in zip(positions, colors):
            grid.flat[pos] = color
        if _key(grid) in seen_inputs:
            continue
        if shape == (3, 3) and tuple(int(x) for x in grid.flat) in SOURCE_INPUTS:
            continue
        prior_masks.setdefault(group, []).append(positions)
        return grid, _output(grid, background)
    # Exhaust the finite background/mask space in a fixed order if random draws collide.
    for background in range(10):
        if background == avoid_background:
            continue
        if shape == (3, 3) and k == 7 and background in train_pool:
            continue
        familiar = [c for c in train_pool if c != background]
        novel = [c for c in range(10) if c not in train_pool and c != background]
        if split == "train":
            novel = []
        requested_novel = min(novelty, k, len(novel)) if split == "test" else 0
        cap = bg_cells - 1
        distinct = max(1 + (k - 1) // cap, requested_novel)
        if witness and distinct == k:
            continue
        if distinct - requested_novel > len(familiar):
            continue
        chosen = novel[:requested_novel] + familiar[:distinct - requested_novel]
        colors = list(chosen)
        for color in chosen:
            colors.extend([color] * min(cap - 1, k - len(colors)))
        for code in range(1 << area):
            if code.bit_count() != k:
                continue
            positions = tuple(pos for pos in range(area) if code & (1 << pos))
            grid = np.full(shape, background, dtype=np.int8)
            for pos, color in zip(positions, colors):
                grid.flat[pos] = color
            if _key(grid) in seen_inputs:
                continue
            if shape == (3, 3) and tuple(int(x) for x in grid.flat) in SOURCE_INPUTS:
                continue
            prior_masks.setdefault(group, []).append(positions)
            return grid, _output(grid, background)
    raise RuntimeError("Could not construct a distinct valid pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _pair_counts(seed, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    train_pool = tuple(palette[:7])
    train_counts, test_counts = _sample_counts(rng, ntrain, ntest)
    extended = bool(rng.random() < 0.68)
    pair_rngs = [np.random.default_rng(int(x)) for x in rng.integers(0, np.iinfo(np.int64).max, size=ntrain + ntest)]
    seen_inputs: set[tuple] = set()
    prior_masks: dict[tuple[tuple[int, int], int], list[tuple[int, ...]]] = {}
    puzzle: dict[str, list[dict]] = {"train": [], "test": []}
    first_background: int | None = None
    for i, k in enumerate(train_counts):
        shape = (3, 3) if i == 0 else _choose_shape(pair_rngs[i], k, extended, force_extended=extended and i == 1)
        grid, output = _make_pair(
            pair_rngs[i], shape, k, train_pool, "train", seen_inputs, prior_masks,
            witness=i == 0, avoid_background=first_background if i == 1 else None,
        )
        if i == 0:
            first_background = _background(grid)
        seen_inputs.add(_key(grid))
        puzzle["train"].append({"input": grid, "output": output})
    demonstrated_colors = {
        int(color)
        for pair in puzzle["train"]
        for color in pair["input"][pair["input"] != _background(pair["input"])]
    }
    for i, k in enumerate(test_counts):
        j = ntrain + i
        if k == 1:
            novelty = int(pair_rngs[j].integers(0, 2))
        else:
            novelty = int(pair_rngs[j].choice([0, 1, 2, 3], p=[0.32, 0.32, 0.28, 0.08]))
        novelty = min(novelty, k)
        if not extended:
            background_cells = 9 - k
            minimum_colors = (k + background_cells - 2) // (background_cells - 1)
            novelty = max(novelty, minimum_colors - len(demonstrated_colors))
        shape = _choose_shape(
            pair_rngs[j], k, extended, familiar_count=len(demonstrated_colors), novelty=novelty
        )
        grid, output = _make_pair(
            pair_rngs[j], shape, k, tuple(sorted(demonstrated_colors)), "test", seen_inputs, prior_masks,
            novelty=novelty,
        )
        seen_inputs.add(_key(grid))
        puzzle["test"].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    records: dict[str, list[tuple[int, tuple[int, int], int, bool]]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert np.issubdtype(array.dtype, np.integer)
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert 0 <= int(array.min()) <= int(array.max()) <= 9
            input_key, output_key = _key(grid), _key(output)
            assert input_key not in seen_inputs and (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            assert grid.shape in SHAPES
            background = _background(grid)
            k = int(np.count_nonzero(grid != background))
            assert 1 <= k <= 7 and grid.size - k >= 2
            if split == "train":
                assert k >= 2
            assert grid.shape[0] * k <= 30 and grid.shape[1] * k <= 30
            assert output.shape == (grid.shape[0] * k, grid.shape[1] * k)
            assert np.array_equal(output, _output(grid, background))
            if grid.shape == (3, 3):
                assert tuple(int(x) for x in grid.flat) not in SOURCE_INPUTS
            distinct = len(set(int(x) for x in grid[grid != background]))
            records[split].append((k, grid.shape, background, distinct < k))
    train = records["train"]
    assert len({r[0] for r in train}) >= 2
    assert len({r[2] for r in train}) >= 2
    assert any(r[3] for r in train)
    if any(r[1] != (3, 3) for r in records["test"]):
        assert any(r[1] != (3, 3) for r in train)


def _equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[t], y[t]) for x, y in zip(a[s], b[s]) for t in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    assert _equal(generate_puzzle(12345), generate_puzzle(12345))
    for seed, nt, ns in ((54321, 7, 3), (20001, 2, 1), (20002, 10, 1), (20003, 10, 4),
                         (21001, None, 1), (21002, None, 4), (21003, 2, None), (21004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
        assert _equal(puzzle, generate_puzzle(seed, nt, ns))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    train_sizes, test_sizes, count_pairs = set(), set(), set()
    shapes, scales, backgrounds, masks, signatures = set(), set(), set(), set(), set()
    seen_tests, unseen_tests, repeated, extended_train, extended_test = 0, 0, 0, 0, 0
    familiar_tests, one_novel_tests, many_novel_tests = 0, 0, 0
    reused_masks = same_mask_new_colors = identity_tests = repeated_test = 0
    two_cell_same_color_witnesses = source_like = standard_only = combined = 0
    zero_foreground = 0
    contact_modes: set[bool] = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_sizes.add(len(puzzle["train"]))
        test_sizes.add(len(puzzle["test"]))
        count_pairs.add((len(puzzle["train"]), len(puzzle["test"])))
        signatures.add(tuple((s, _key(p["input"])) for s in ("train", "test") for p in puzzle[s]))
        training_counts, training_colors = set(), set()
        local_masks: dict[tuple[tuple[int, int], int, tuple[bool, ...]], tuple[int, ...]] = {}
        puzzle_extended_train = puzzle_extended_test = puzzle_unseen_many = False
        all_standard = True
        test_counts: list[int] = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                bg = _background(grid)
                mask = grid != bg
                k = int(mask.sum())
                assert np.array_equal(output, np.kron(grid, np.ones((k, k), dtype=np.int8)))
                assert set(int(x) for x in np.unique(grid)) == set(int(x) for x in np.unique(output))
                shapes.add(grid.shape)
                scales.add(k)
                backgrounds.add(bg)
                all_standard &= grid.shape == (3, 3)
                zero_foreground += 0 in grid[mask]
                contact_modes.add(bool(
                    np.any(mask[:, :-1] & mask[:, 1:])
                    or np.any(mask[:-1, :] & mask[1:, :])
                ))
                mask_key = (grid.shape, k, tuple(bool(x) for x in mask.flat))
                masks.add(mask_key)
                foreground_assignment = tuple(int(x) for x in grid[mask])
                if mask_key in local_masks:
                    reused_masks += 1
                    same_mask_new_colors += foreground_assignment != local_masks[mask_key]
                local_masks[mask_key] = foreground_assignment
                if len(set(int(x) for x in grid[mask])) < k:
                    repeated += 1
                    repeated_test += split == "test"
                if split == "train":
                    training_counts.add(k)
                    training_colors.update(int(x) for x in grid[mask])
                    extended_train += grid.shape != (3, 3)
                    puzzle_extended_train |= grid.shape != (3, 3)
                    two_cell_same_color_witnesses += k == 2 and len(set(foreground_assignment)) == 1
                else:
                    test_counts.append(k)
                    extended_test += grid.shape != (3, 3)
                    puzzle_extended_test |= grid.shape != (3, 3)
                    identity_tests += k == 1 and np.array_equal(grid, output)
                    if k in training_counts:
                        seen_tests += 1
                    else:
                        unseen_tests += 1
                    novel = len(set(int(x) for x in grid[mask]) - training_colors)
                    familiar_tests += novel == 0
                    one_novel_tests += novel == 1
                    many_novel_tests += novel >= 2
                    puzzle_unseen_many |= k not in training_counts and novel >= 2
        source_like += training_counts == {2, 3, 5} and set(test_counts) == {4}
        standard_only += all_standard
        combined += puzzle_extended_train and puzzle_extended_test and puzzle_unseen_many
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3 and len(count_pairs) >= 10
    assert len(signatures) == 42 and len(masks) >= 90
    assert len(shapes) >= 5 and scales.issuperset({2, 3, 4, 5, 6, 7})
    assert len(backgrounds) >= 8 and repeated >= 42
    assert seen_tests > 0 and unseen_tests > 0
    assert extended_train > 0 and extended_test > 0
    assert familiar_tests > 0 and one_novel_tests > 0 and many_novel_tests > 0
    assert reused_masks > 0 and same_mask_new_colors > 0
    assert identity_tests > 0 and repeated_test > 0 and two_cell_same_color_witnesses > 0
    assert source_like > 0 and standard_only > 0 and combined > 0
    assert zero_foreground > 0 and contact_modes == {False, True}
