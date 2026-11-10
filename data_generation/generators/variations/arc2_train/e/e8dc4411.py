"""Generate ARC puzzles about diagonal stamping of a visible template."""

import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SOURCE_INPUTS = {
    (13, 15, 1, 0, 3, ((5, 6), (6, 5), (6, 6), (6, 7), (7, 6)), (7, 4)),
    (12, 14, 8, 0, 2, ((3, 4), (4, 3), (4, 5), (5, 4)), (5, 6)),
    (12, 16, 4, 0, 8,
     tuple(sorted([(r, c) for r in (5, 6, 8, 9) for c in (6, 7, 9, 10)]
                  + [(7, 8)])), (4, 11)),
    (18, 16, 3, 0, 6, ((6, 4), (6, 6), (7, 5), (8, 4), (8, 6)), (5, 3)),
}


def _validate_count(name, value, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(seed, rng, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _coords(coords):
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _overlaps(coords, dr, dc):
    cells = set(_coords(coords))
    return any((r + dr, c + dc) in cells for r, c in cells)


def _step(coords, direction):
    for k in range(1, 7):
        if not _overlaps(coords, direction[0] * k, direction[1] * k):
            return k
    raise AssertionError("no non-overlapping step")


def _candidate_vectors(coords, seed):
    candidates = set()
    for row, col in coords:
        dr, dc = int(seed[0] - row), int(seed[1] - col)
        if not dr or abs(dr) != abs(dc) or _overlaps(coords, dr, dc):
            continue
        direction = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
        if all(_overlaps(coords, direction[0] * k, direction[1] * k)
               for k in range(1, abs(dr))):
            candidates.add((dr, dc))
    return sorted(candidates)


def _safe_repetitions(coords, vector):
    return all(not _overlaps(coords, vector[0] * k, vector[1] * k)
               for k in range(1, 7))


def _eligible_seeds(coords, vector):
    return [(int(r), int(c)) for r, c in coords
            if _candidate_vectors(coords, (int(r) + vector[0], int(c) + vector[1]))
            == [vector]]


def _sample_motif(rng, size, direction):
    outward = (size - 1 if direction[0] > 0 else 0,
               size - 1 if direction[1] > 0 else 0)
    for _ in range(3000):
        mode = int(rng.integers(6))
        if mode == 0 and size == 3:
            cells = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}
            if rng.random() < .5:
                cells.add((int(rng.integers(3)), int(rng.integers(3))))
        elif mode == 0 and size == 5:
            # The source's dense 17-cell block is one possible template.
            cells = {(r, c) for r in (0, 1, 3, 4)
                     for c in (0, 1, 3, 4)} | {(2, 2)}
        elif mode == 1:
            # An adjacent diagonal pair establishes a two-cell step. Place
            # opposite-edge cells to retain the full bounding box.
            cells = {outward,
                     (outward[0] - direction[0],
                      outward[1] - direction[1]),
                     (size - 1 - outward[0], outward[1]),
                     (outward[0], size - 1 - outward[1])}
            count = int(rng.integers(4, 7 if size == 3 else 10))
            count = max(count, 6 if size == 5 else 4)
            available = [(r, c) for r in range(size) for c in range(size)
                         if (r, c) not in cells]
            indices = rng.choice(len(available), size=count - len(cells),
                                 replace=False)
            cells.update(available[int(i)] for i in indices)
        else:
            if mode == 2:
                diagonal = int(rng.integers(2))
                cells = {(i, i if diagonal == 0 else size - 1 - i)
                         for i in range(size)}
                count = int(rng.integers(4 if size == 3 else 6,
                                         9 if size == 3 else 15))
            else:
                cells = {outward}
                count = int(rng.integers(4, 9 if size == 3 else 21))
                if size == 5:
                    count = max(6, count)
            available = [(r, c) for r in range(size) for c in range(size)
                         if (r, c) not in cells]
            extra = max(0, count - len(cells))
            if extra:
                indices = rng.choice(len(available), size=extra, replace=False)
                cells.update(available[int(i)] for i in indices)
        cells.add(outward)
        coords = np.array(sorted(cells), dtype=np.int64)
        if (coords[:, 0].min() != 0 or coords[:, 0].max() != size - 1
                or coords[:, 1].min() != 0 or coords[:, 1].max() != size - 1):
            continue
        step = _step(coords, direction)
        if step not in (2, 3, 4, 5):
            continue
        vector = (direction[0] * step, direction[1] * step)
        if not _safe_repetitions(coords, vector):
            continue
        eligible = _eligible_seeds(coords, vector)
        if eligible:
            return coords, vector, eligible
    raise RuntimeError("could not sample an unambiguous motif")


def _copies(coords, vector, height, width):
    copies = []
    for multiple in range(1, 2 * max(height, width) + 10):
        shifted = coords + np.array(vector) * multiple
        inside = ((0 <= shifted[:, 0]) & (shifted[:, 0] < height)
                  & (0 <= shifted[:, 1]) & (shifted[:, 1] < width))
        if not np.any(inside):
            return copies
        copies.append((shifted, inside))
    raise AssertionError("copies did not leave the grid")


def _clip_edges(copies, direction, height, width):
    shifted, inside = copies[-1]
    if np.all(inside):
        return 0
    row_exit = shifted[:, 0] < 0 if direction[0] < 0 else shifted[:, 0] >= height
    col_exit = shifted[:, 1] < 0 if direction[1] < 0 else shifted[:, 1] >= width
    return int(np.any(row_exit)) + int(np.any(col_exit))


def _output(inp, coords, vector, color):
    output = inp.copy()
    copies = _copies(coords, vector, *inp.shape)
    for shifted, inside in copies:
        visible = shifted[inside]
        output[visible[:, 0], visible[:, 1]] = color
    return output, copies


def _placement(rng, motif, vector, regime):
    size = int(max(motif.max(axis=0)) + 1)
    direction = (1 if vector[0] > 0 else -1,
                 1 if vector[1] > 0 else -1)
    desired_clip = int(rng.choice(3, p=(.25, .43, .32)))
    desired_full = int(rng.integers(1, 7))
    fallback = []
    for _ in range(14):
        height = int(rng.integers(10, 27 if regime == "target_test" else 25))
        width = int(rng.integers(11, 28 if regime == "target_test" else 26))
        low_r, high_r = max(0, -vector[0]), height - size - max(0, vector[0])
        low_c, high_c = max(0, -vector[1]), width - size - max(0, vector[1])
        if low_r > high_r or low_c > high_c:
            continue
        choices = []
        for top in range(low_r, high_r + 1):
            for left in range(low_c, high_c + 1):
                coords = motif + np.array((top, left))
                copies = _copies(coords, vector, height, width)
                if not copies or not np.all(copies[0][1]):
                    continue
                clip = _clip_edges(copies, direction, height, width)
                full = sum(bool(np.all(mask)) for _, mask in copies)
                fallback.append((height, width, top, left, clip, full))
                if clip == desired_clip:
                    choices.append((top, left, full))
        if choices:
            distance = min(abs(item[2] - desired_full) for item in choices)
            choices = [item for item in choices if abs(item[2] - desired_full) == distance]
            top, left, _ = choices[int(rng.integers(len(choices)))]
            return height, width, top, left
    if fallback:
        distance = min((abs(item[4] - desired_clip), abs(item[5] - desired_full))
                       for item in fallback)
        matching = [item for item in fallback
                    if (abs(item[4] - desired_clip), abs(item[5] - desired_full)) == distance]
        height, width, top, left, _, _ = matching[int(rng.integers(len(matching)))]
        return height, width, top, left
    raise RuntimeError("could not place motif and its first copy")


def _generate_pair(rng, regime, roles, direction, size, shared=None):
    if shared is None:
        motif, vector, eligible = _sample_motif(rng, size, direction)
    else:
        motif = shared.copy()
        step = _step(motif, direction)
        vector = (direction[0] * step, direction[1] * step)
        eligible = _eligible_seeds(motif, vector)
        if not eligible or not _safe_repetitions(motif, vector):
            raise RuntimeError("shared motif is ambiguous")
    height, width, top, left = _placement(rng, motif, vector, regime)
    coords = motif + np.array((top, left))
    phase = eligible[int(rng.integers(len(eligible)))]
    seed = (top + phase[0] + vector[0], left + phase[1] + vector[1])
    background, template, projection = roles
    inp = np.full((height, width), background, dtype=np.int8)
    inp[coords[:, 0], coords[:, 1]] = template
    assert 0 <= seed[0] < height and 0 <= seed[1] < width
    assert inp[seed] == background
    inp[seed] = projection
    output, copies = _output(inp, coords, vector, projection)
    assert copies and not np.array_equal(inp, output)
    return {"input": inp, "output": output}


def _analyze_pair(inp, output):
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    background = int(values[int(np.argmax(counts))])
    assert max(counts) > inp.size // 2
    others = [(int(v), int(n)) for v, n in zip(values, counts)
              if int(v) != background]
    singleton = [v for v, n in others if n == 1]
    assert len(singleton) == 1
    projection = singleton[0]
    template = next(v for v, _ in others if v != projection)
    coords = np.argwhere(inp == template).astype(np.int64)
    assert len(coords) >= 4
    seed = tuple(int(x) for x in np.argwhere(inp == projection)[0])
    vectors = _candidate_vectors(coords, seed)
    assert len(vectors) == 1
    vector = vectors[0]
    assert _safe_repetitions(coords, vector)
    expected, copies = _output(inp, coords, vector, projection)
    assert copies and np.all(copies[0][1])
    assert np.array_equal(output, expected) and not np.array_equal(inp, output)
    assert np.count_nonzero(inp != output) >= len(coords) - 1
    anchor = coords.min(axis=0)
    box = tuple(int(x) for x in np.ptp(coords, axis=0) + 1)
    direction = (1 if vector[0] > 0 else -1,
                 1 if vector[1] > 0 else -1)
    fingerprint = (inp.shape[0], inp.shape[1], background, template,
                   projection, _coords(coords), seed)
    assert fingerprint not in _SOURCE_INPUTS
    return {
        "roles": (background, template, projection),
        "direction": direction, "step": abs(vector[0]),
        "box": box, "cell_count": len(coords),
        "motif": _coords(coords - anchor),
        "anchor": tuple(int(x) for x in anchor),
        "copies": len(copies),
        "full_copies": sum(bool(np.all(mask)) for _, mask in copies),
        "clip_edges": _clip_edges(copies, direction, *inp.shape),
    }


def _grid_key(grid):
    return tuple(grid.shape), grid.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = pair["input"], pair["output"]
            for grid in (inp, output):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == output.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze_pair(inp, output)
            assert info["box"] in ((3, 3), (5, 5))
            assert info["step"] in (2, 3, 4, 5)


def _roles(rng, stable_template):
    if stable_template is None:
        return tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    remaining = [c for c in range(10) if c != stable_template]
    chosen = rng.choice(remaining, size=2, replace=False)
    return int(chosen[0]), stable_template, int(chosen[1])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    # The stamping rule is shared. Literal color roles are legible in each
    # input; sometimes the template color also persists across pairs.
    stable_template = int(rng.integers(10)) if rng.random() < .4 else None
    regime = int(rng.integers(3))
    train_pool = _DIRECTIONS[1:] if regime == 0 else _DIRECTIONS
    train_directions = [train_pool[int(rng.integers(len(train_pool)))]
                        for _ in range(num_train)]
    if num_train >= 2 and (regime == 0 or rng.random() < .75):
        first = train_pool.index(train_directions[0])
        train_directions[1] = train_pool[(first + int(rng.integers(1, len(train_pool))))
                                         % len(train_pool)]
    if regime == 2 and (-1, -1) not in train_directions:
        train_directions[int(rng.integers(num_train))] = (-1, -1)
    shared = (num_train >= 2 and (regime == 1 or num_train >= 3)
              and rng.random() < .28)
    if shared:
        train_directions[1] = train_directions[0]
        if num_train >= 3 and all(d == train_directions[0] for d in train_directions):
            pool = [d for d in train_pool if d != train_directions[0]]
            train_directions[2] = pool[int(rng.integers(len(pool)))]
    if regime == 2 and (-1, -1) not in train_directions:
        train_directions[-1] = (-1, -1)
    if regime == 0:
        test_directions = [(-1, -1)] * num_test
    elif regime == 1:
        test_directions = [train_directions[int(rng.integers(num_train))]
                           for _ in range(num_test)]
    else:
        test_directions = [_DIRECTIONS[int(rng.integers(4))]
                           for _ in range(num_test)]
    train_sizes = [3 if rng.random() < .5 else 5 for _ in range(num_train)]
    if shared:
        train_sizes[1] = train_sizes[0]
    test_sizes = [3 if rng.random() < .5 else 5 for _ in range(num_test)]
    train, test = [], []
    first_motif = None
    for i in range(num_train):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        motif = first_motif if shared and i == 1 else None
        pair = _generate_pair(pair_rng, "source_train", _roles(pair_rng, stable_template),
                              train_directions[i], train_sizes[i], motif)
        if shared and i == 0:
            info = _analyze_pair(pair["input"], pair["output"])
            first_motif = np.array(info["motif"], dtype=np.int64)
        train.append(pair)
    for i in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        test.append(_generate_pair(pair_rng, "target_test", _roles(pair_rng, stable_template),
                                   test_directions[i], test_sizes[i]))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    seen = set()
    features = {split: {"directions": set(), "sizes": set(), "steps": set(),
                        "clips": set(), "full": set(), "shapes": set()}
                for split in ("train", "test")}
    shared_templates = stable_colors = trained_up_left = repeated_test_direction = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        key = _puzzle_key(puzzle)
        assert key not in seen
        seen.add(key)
        infos = {split: [_analyze_pair(p["input"], p["output"])
                         for p in puzzle[split]] for split in ("train", "test")}
        train_dirs = {x["direction"] for x in infos["train"]}
        trained_up_left += (-1, -1) in train_dirs
        repeated_test_direction += any(x["direction"] in train_dirs for x in infos["test"])
        shared_templates += len({x["motif"] for x in infos["train"]}) < len(infos["train"])
        stable_colors += len({x["roles"][1] for x in infos["train"] + infos["test"]}) == 1
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                bucket = features[split]
                bucket["directions"].add(info["direction"])
                bucket["sizes"].add(info["box"])
                bucket["steps"].add(info["step"])
                bucket["clips"].add(info["clip_edges"])
                bucket["full"].add(info["full_copies"])
                bucket["shapes"].add(pair["input"].shape)
    assert len(seen) == 42
    for split in ("train", "test"):
        bucket = features[split]
        assert bucket["directions"] == set(_DIRECTIONS)
        assert bucket["sizes"] == {(3, 3), (5, 5)}
        assert bucket["steps"] == {2, 3, 4, 5}
        assert bucket["clips"] == {0, 1, 2}
        assert 1 in bucket["full"] and max(bucket["full"]) >= 4
        assert len(bucket["shapes"]) >= 30
    assert shared_templates >= 3 and stable_colors >= 3
    assert trained_up_left >= 3 and repeated_test_direction >= 3
    assert _puzzle_key(generate_puzzle(1000)) == _puzzle_key(generate_puzzle(1000))
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(222, train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(222, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
