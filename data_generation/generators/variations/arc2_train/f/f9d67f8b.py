import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = frozenset({
    "8678a61e10f34c06b64387f815b4420e240deb1dcc852aef037b240f56d9dae6",
    "34a5ffd68ca632842e8de1dd97d3ad8afe017db1b6cbfad617d4cf012afaeeb6",
    "6a9ff1214ae17159520df91e4f46763bccdd3c916f6c13554a51a552792186eb",
    "16a86cf1ad1945f89a2cd49470f45182ec6aed622fcbde755c458398d7ec29cf",
    "d9ca37b96c40015771eac3560c805dd022b7d5d1f7b1fa05574fdfa91c3f351a",
})


def _counts(seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} outside legal range")
    code = int(seed)
    if num_train is None and num_test is None:
        test = 1 + code % 4
        train = test + 1 + (code // 4) % (10 - test)
    elif num_train is None:
        test = int(num_test)
        train = test + 1 + code % (10 - test)
    elif num_test is None:
        train = int(num_train)
        test = 1 + code % min(4, train - 1)
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _geometry(n, shape, top, left, turn):
    full = 4 * n
    ids = np.arange(full * full, dtype=np.int16).reshape(full, full)
    ids = np.rot90(ids, turn % 4)
    if turn >= 4:
        ids = np.fliplr(ids)
    ids = ids[top:top + shape[0], left:left + shape[1]]
    row, col = ids // full, ids % full
    base_row = np.where(row < 2 * n, row, full - 1 - row)
    base_col = np.where(col < 2 * n, col, full - 1 - col)
    base_id = base_row * (2 * n) + base_col
    swap = (base_row >= n) & (base_col < n)
    key = np.where(swap, base_col * (2 * n) + base_row, base_id)
    return key.astype(np.int16), base_id.astype(np.int16)


def _shape(rng, n, source_size=False):
    if source_size:
        totals = (2, 2)
    else:
        low = max(0, 4 * n - 30)
        high = min(2 * (n - 2), 2 + n)
        totals = (int(rng.integers(low, high + 1)),
                  int(rng.integers(low, high + 1)))
    offsets = []
    for total in totals:
        lo = max(0, total - (n - 2))
        hi = min(n - 2, total)
        offsets.append(int(rng.integers(lo, hi + 1)))
    return (4 * n - totals[0], 4 * n - totals[1]), offsets[0], offsets[1]


def _block(rng, n, palette, texture):
    colors = np.asarray(palette, dtype=np.int8)
    concentration = float(rng.uniform(0.35, 2.0))
    weights = rng.dirichlet(np.full(len(colors), concentration))
    if texture == 0:
        result = rng.choice(colors, size=(n, n), p=weights).astype(np.int8)
    elif texture == 1:
        result = np.empty((n, n), dtype=np.int8)
        horizontal = bool(rng.integers(0, 2))
        cursor = 0
        while cursor < n:
            span = int(rng.integers(1, max(3, n // 2 + 2)))
            color = int(rng.choice(colors, p=weights))
            if horizontal:
                result[cursor:cursor + span, :] = color
            else:
                result[:, cursor:cursor + span] = color
            cursor += span
        noise = rng.random((n, n)) < rng.uniform(0.08, 0.27)
        result[noise] = rng.choice(colors, size=int(noise.sum()), p=weights)
    else:
        result = np.full((n, n), int(rng.choice(colors, p=weights)), dtype=np.int8)
        for _ in range(int(rng.integers(3, 8))):
            h = int(rng.integers(1, max(3, n - 1)))
            w = int(rng.integers(1, max(3, n - 1)))
            r = int(rng.integers(0, n - h + 1))
            c = int(rng.integers(0, n - w + 1))
            result[r:r + h, c:c + w] = int(rng.choice(colors, p=weights))
        if texture == 3:
            noise = rng.random((n, n)) < rng.uniform(0.09, 0.30)
            result[noise] = rng.choice(colors, size=int(noise.sum()), p=weights)
    return result


def _base(rng, n, palette):
    for _ in range(40):
        textures = rng.permutation(4)
        a = _block(rng, n, palette, int(textures[0]))
        b = _block(rng, n, palette, int(textures[1]))
        d = _block(rng, n, palette, int(textures[2]))
        # Put every clean color in a visible, redundant part of the fundamental square.
        locations = rng.choice(n * n, size=len(palette), replace=False)
        for loc, color in zip(locations, palette):
            a.flat[int(loc)] = int(color)
        if any(np.count_nonzero(x != x.T) < max(4, n) for x in (a, b, d)):
            continue
        if np.array_equal(a, d) or np.array_equal(a, d.T):
            continue
        return np.block([[a, b], [b.T, d]]).astype(np.int8)
    raise AssertionError("could not construct asymmetric blocks")


def _palette(rng, shared_palette, shared_marker):
    if shared_palette is not None:
        pool = [c for c in range(10) if c not in shared_palette]
        marker = shared_marker if shared_marker is not None else int(rng.choice(pool))
        return tuple(shared_palette), marker
    marker = shared_marker if shared_marker is not None else int(rng.integers(0, 10))
    pool = [c for c in range(10) if c != marker]
    size = int(rng.integers(4, 9))
    include_zero = bool(rng.integers(0, 2)) and marker != 0
    if include_zero:
        choice = [0] + list(map(int, rng.choice([c for c in pool if c != 0], size - 1, replace=False)))
    else:
        nonzero = [c for c in pool if c != 0]
        size = min(size, len(nonzero))
        choice = list(map(int, rng.choice(nonzero, size, replace=False)))
    return tuple(sorted(choice)), marker


def _direct_witness(mask, row, col, n, top, left):
    full = 4 * n
    mr = full - 1 - row - 2 * top
    mc = full - 1 - col - 2 * left
    return ((0 <= mr < mask.shape[0] and not mask[mr, col]) or
            (0 <= mc < mask.shape[1] and not mask[row, mc]))


def _critical_count(mask, key, base_id, n, top, left):
    count = 0
    for row, col in np.argwhere(mask):
        if not _direct_witness(mask, int(row), int(col), n, top, left):
            same = (key == key[row, col]) & (~mask) & (base_id != base_id[row, col])
            if np.any(same):
                count += 1
    return count


def _mask(rng, n, shape, top, left, key, base_id, regime, difficulty):
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    scale = min(height, width)
    if difficulty == "small":
        spans = [(1, max(2, scale // 5)), (1, max(2, scale // 6))]
    elif difficulty == "medium":
        spans = [(2, max(3, scale // 3)), (2, max(3, scale // 4))]
    else:
        spans = [(max(3, scale // 5), max(4, scale // 3)),
                 (max(3, scale // 6), max(4, scale // 3)),
                 (1, max(2, scale // 6))]
    if regime != "critical_only":
        for low, high in spans:
            h = min(height, int(rng.integers(low, high + 1)))
            w = min(width, int(rng.integers(low, high + 1)))
            r = int(rng.integers(0, height - h + 1))
            c = int(rng.integers(0, width - w + 1))
            shape_kind = int(rng.integers(0, 5))
            patch = np.ones((h, w), dtype=bool)
            if shape_kind == 1 and h > 2 and w > 2:
                patch[1:-1, 1:-1] = False
            elif shape_kind == 2 and h > 2 and w > 2:
                patch[h // 2:, w // 2:] = False
            mask[r:r + h, c:c + w] |= patch
    if regime != "critical_only" and difficulty == "large" and scale >= 22:
        # This source-scale rectangle makes 70-120 erased cells commonplace.
        h = int(rng.integers(5, min(10, height) + 1))
        w = int(rng.integers(5, min(9, width) + 1))
        r = int(rng.integers(0, height - h + 1))
        c = int(rng.integers(0, width - w + 1))
        mask[r:r + h, c:c + w] = True

    if regime != "direct":
        # Erase every visible axial copy of a few off-diagonal base cells.
        # Their transposed partners remain as clean witnesses.
        r0 = int(rng.integers(0, n))
        c0 = int(rng.integers(n, 2 * n))
        rh = int(rng.integers(1, min(4, n - r0 + 1)))
        cw = int(rng.integers(1, min(4, 2 * n - c0 + 1)))
        chosen = np.zeros(shape, dtype=bool)
        counterparts = np.zeros(shape, dtype=bool)
        for br in range(r0, min(n, r0 + rh)):
            for bc in range(c0, min(2 * n, c0 + cw)):
                forward = br * (2 * n) + bc
                transposed = bc * (2 * n) + br
                if np.any(base_id == forward) and np.any(base_id == transposed):
                    chosen |= base_id == forward
                    counterparts |= base_id == transposed
        mask |= chosen
        mask &= ~counterparts

    # Every equivalence class must retain a visible value.
    for group in np.unique(key[mask]):
        members = key == group
        if np.all(mask[members]):
            choices = np.argwhere(members)
            row, col = choices[int(rng.integers(0, len(choices)))]
            mask[row, col] = False
    if regime == "direct":
        for row, col in np.argwhere(mask):
            if not _direct_witness(mask, int(row), int(col), n, top, left):
                mask[row, col] = False
    if not np.any(mask):
        if regime != "direct":
            for group in np.unique(key):
                ids = np.unique(base_id[key == group])
                if len(ids) >= 2:
                    mask[(key == group) & (base_id == ids[0])] = True
                    break
        else:
            for flat in rng.permutation(height * width):
                row, col = divmod(int(flat), width)
                if _direct_witness(mask, row, col, n, top, left):
                    mask[row, col] = True
                    break
    return mask


def _candidate_maps(shape):
    height, width = shape
    for n in range(4, 9):
        full = 4 * n
        dh, dw = full - height, full - width
        if not (0 <= dh <= 2 * (n - 2) and 0 <= dw <= 2 * (n - 2)):
            continue
        for top in range(max(0, dh - n + 2), min(n - 2, dh) + 1):
            for left in range(max(0, dw - n + 2), min(n - 2, dw) + 1):
                for turn in range(8):
                    key, base_id = _geometry(n, shape, top, left, turn)
                    yield n, top, left, turn, key, base_id


def _consistent(values, key, selected):
    low = np.full(256, 10, dtype=np.int16)
    high = np.full(256, -1, dtype=np.int16)
    np.minimum.at(low, key[selected], values[selected])
    np.maximum.at(high, key[selected], values[selected])
    return bool(np.all((high < 0) | (low == high))), low


def _infer_input(input_grid):
    predictions = []
    possible_markers = [int(c) for c in np.unique(input_grid)]
    for n, top, left, turn, key, _ in _candidate_maps(input_grid.shape):
        for marker in possible_markers:
            unknown = input_grid == marker
            valid, known = _consistent(input_grid, key, ~unknown)
            if not valid or np.any(known[key[unknown]] == 10):
                continue
            filled = input_grid.copy()
            filled[unknown] = known[key[unknown]]
            if marker in filled:
                continue
            predictions.append(filled)
    assert predictions, "input has no complete structural interpretation"
    first = predictions[0]
    assert all(np.array_equal(first, other) for other in predictions), "ambiguous completion"
    return first


def _new_pair(rng, n, shape, top, left, turn, palette, marker, split, difficulty, regime):
    base = _base(rng, n, palette)
    full = base[np.ix_(np.where(np.arange(4 * n) < 2 * n, np.arange(4 * n), 4 * n - 1 - np.arange(4 * n)),
                       np.where(np.arange(4 * n) < 2 * n, np.arange(4 * n), 4 * n - 1 - np.arange(4 * n)))]
    full = np.rot90(full, turn % 4)
    if turn >= 4:
        full = np.fliplr(full)
    output = full[top:top + shape[0], left:left + shape[1]].copy()
    key, base_id = _geometry(n, shape, top, left, turn)
    mask = _mask(rng, n, shape, top, left, key, base_id, regime, difficulty)
    if regime != "direct" and _critical_count(mask, key, base_id, n, top, left) == 0:
        # A clipped off-diagonal cell still gives a deterministic fallback.
        for group in np.unique(key):
            ids = np.unique(base_id[key == group])
            if len(ids) < 2:
                continue
            first = (key == group) & (base_id == ids[0])
            other = (key == group) & (base_id == ids[1])
            if np.any(first) and np.any(other):
                mask[first] = True
                mask[other] = False
                break
    input_grid = output.copy()
    input_grid[mask] = marker
    return {"input": input_grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    constant_shape = bool(rng.random() < 0.26)
    shared_marker = int(rng.integers(0, 10)) if rng.random() < 0.52 else None
    shared_palette = None
    if rng.random() < 0.38:
        pool = [c for c in range(10) if c != shared_marker] if shared_marker is not None else list(range(10))
        size = int(rng.integers(4, min(8, len(pool) - 2) + 1))
        use_zero = 0 in pool and bool(rng.integers(0, 2))
        if use_zero:
            shared_palette = tuple(sorted([0] + list(map(int, rng.choice([c for c in pool if c != 0], size - 1, replace=False)))))
        else:
            shared_palette = tuple(sorted(map(int, rng.choice([c for c in pool if c != 0], size, replace=False))))
    fixed = None
    if constant_shape:
        n = int(rng.integers(4, 9))
        fixed = (n, *_shape(rng, n, n == 8 and rng.random() < 0.28))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(8):
                if fixed is None:
                    source_size = bool(rng.random() < 0.15)
                    n = 8 if source_size else int(rng.integers(4, 9))
                    shape, top, left = _shape(rng, n, source_size)
                else:
                    n, shape, top, left = fixed
                turn = int(rng.integers(0, 8))
                palette, marker = _palette(rng, shared_palette, shared_marker)
                if split == "train":
                    regime = "critical" if index == 0 and train_count >= 3 and seed % 4 == 0 else ("critical" if rng.random() < 0.19 else "direct")
                    difficulty = str(rng.choice(["small", "medium", "large"], p=[0.20, 0.53, 0.27]))
                else:
                    regime = str(rng.choice(["direct", "critical", "critical_only"], p=[0.30, 0.54, 0.16]))
                    difficulty = str(rng.choice(["small", "medium", "large"], p=[0.09, 0.40, 0.51]))
                pair = _new_pair(rng, n, shape, top, left, turn, palette, marker, split, difficulty, regime)
                token = (pair["input"].shape, pair["input"].tobytes())
                if token not in seen:
                    seen.add(token)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("duplicate construction")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert type(train) is list and type(test) is list
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_relation = 0
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            source_digest = hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()).hexdigest()
            assert source_digest not in _SOURCE_INPUT_HASHES
            it = (inp.shape, inp.tobytes())
            pt = (it, out.shape, out.tobytes())
            assert it not in seen_inputs and pt not in seen_pairs
            seen_inputs.add(it)
            seen_pairs.add(pt)
            changed = inp != out
            assert np.any(changed)
            markers = np.unique(inp[changed])
            assert len(markers) == 1
            marker = int(markers[0])
            assert marker not in out
            assert np.all(inp[~changed] == out[~changed])
            candidates = []
            for n, top, left, turn, key, base_id in _candidate_maps(out.shape):
                valid, _ = _consistent(out, key, np.ones(out.shape, dtype=bool))
                if not valid:
                    continue
                witnessed = True
                for group in np.unique(key[changed]):
                    if not np.any((key == group) & ~changed):
                        witnessed = False
                        break
                if not witnessed:
                    continue
                offdiag = (base_id // (2 * n) < n) & (base_id % (2 * n) >= n)
                train_relation += int(split == "train" and np.count_nonzero(offdiag & ~changed) >= n)
                candidates.append((n, top, left, turn, key, base_id))
                break
            assert candidates, "clean mosaic or erasure witnesses invalid"
            if split == "test":
                assert np.array_equal(_infer_input(inp), out), "input-only completion failed"
    assert train_relation >= len(train)


def self_test() -> None:
    counts_train, counts_test = set(), set()
    shapes, masks, markers, palette_sets, sizes = set(), set(), set(), set(), []
    within_shape, within_palette, within_scale, within_marker = 0, 0, 0, 0
    witness_regimes = {"train": set(), "test": set()}
    clipped_sides = set()
    full_fields = source_sized = seam_spanning = 0
    mask_quadrants = set()
    adjacent_rates = []
    scales = set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            first = puzzle
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        pairs = puzzle["train"] + puzzle["test"]
        within_shape += len({p["input"].shape for p in pairs}) > 1
        within_palette += len({tuple(np.unique(p["output"])) for p in pairs}) > 1
        within_marker += len({int(p["input"][p["input"] != p["output"]][0]) for p in pairs}) > 1
        local_scales = set()
        for split, pair in [(s, p) for s in ("train", "test") for p in puzzle[s]]:
            inp, out = pair["input"], pair["output"]
            diff = inp != out
            loss = int(diff.sum())
            sizes.append(loss)
            markers.add(int(inp[diff][0]))
            palette_sets.add(tuple(int(x) for x in np.unique(out)))
            shapes.add(out.shape)
            masks.add((out.shape, diff.tobytes()))
            n, top, left, _, key, base_id = next(
                candidate for candidate in _candidate_maps(out.shape)
                if _consistent(out, candidate[4], np.ones(out.shape, dtype=bool))[0]
            )
            scales.add(n)
            local_scales.add(n)
            critical = _critical_count(diff, key, base_id, n, top, left)
            witness_regimes[split].add("direct" if critical == 0 else
                                       "critical_only" if critical == loss else "mixed")
            source_sized += out.shape == (30, 30)
            full_fields += top == left == 0 and out.shape == (4 * n, 4 * n)
            for side, amount in (("top", top), ("bottom", 4 * n - out.shape[0] - top),
                                 ("left", left), ("right", 4 * n - out.shape[1] - left)):
                if amount:
                    clipped_sides.add(side)
            ys, xs = np.where(diff)
            mask_quadrants.add((int(ys.mean() >= out.shape[0] / 2),
                                int(xs.mean() >= out.shape[1] / 2)))
            seam_spanning += bool((ys.min() < 2 * n - top <= ys.max()) or
                                  (xs.min() < 2 * n - left <= xs.max()))
            adjacent_rates.append(((out[:, 1:] == out[:, :-1]).mean() +
                                   (out[1:, :] == out[:-1, :]).mean()) / 2)
        within_scale += len(local_scales) > 1
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(shapes) >= 20 and len(masks) >= 100 and len(palette_sets) >= 20
    assert len(scales) == 5 and len(markers) == 10 and max(sizes) >= 100
    assert sum(size >= 31 for size in sizes) >= 40
    assert within_shape >= 15 and within_palette >= 15 and within_scale >= 15
    assert within_marker >= 5
    assert witness_regimes["train"] >= {"direct", "mixed"}
    assert witness_regimes["test"] == {"direct", "mixed", "critical_only"}
    assert clipped_sides == {"top", "bottom", "left", "right"}
    assert full_fields >= 1 and source_sized >= 5 and seam_spanning >= 10
    assert len(mask_quadrants) == 4
    assert min(adjacent_rates) < 0.35 and max(adjacent_rates) > 0.70
    assert any(0 in p for p in palette_sets) and any(0 not in p for p in palette_sets)
    assert any(shape == (30, 30) for shape in shapes)
    again = generate_puzzle(0)
    for split in ("train", "test"):
        assert all(np.array_equal(a[k], b[k]) for a, b in zip(first[split], again[split]) for k in ("input", "output"))
    different = generate_puzzle(1)
    assert not np.array_equal(first["train"][0]["input"], different["train"][0]["input"])
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(1200 + (train or 0) * 10 + (test or 0), train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
