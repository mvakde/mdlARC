"""Periodic multiplication tables with recoverable erased cells."""

import hashlib
import numpy as np

_SOURCE_PAIR_DIGESTS = frozenset({
    "9aca3fdf0e330b269d4166415445231bbc8a4ce26f5e77aded03e62af40399ca",
    "de8c75980394d27b75927886f2ff9e211f3aac7c4ca6da1d2e8a321df3c84d82",
    "ec99e6880ad90073127dd91ec51304fe2604ad4610ff4ddd37cfa914cfb8044d",
    "42af1ad55308506c916706af27918c7299ced8b71b92bb1250e1c3f149a7d6d4",
    "6699875c2dc4d94eab716a8df328626c3651fa3450a53f7ca5805c2216bd88d4",
})


def _check_count(value, name, low, high):
    if value is not None and (isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _choose_counts(rng, num_train, num_test):
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _pair_digest(pair):
    a = pair["input"].astype(np.uint8, copy=False)
    b = pair["output"].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _table(height, width, period, cycle, row_phase, col_phase):
    rows = (np.arange(height, dtype=np.int16) + row_phase)[:, None]
    cols = (np.arange(width, dtype=np.int16) + col_phase)[None, :]
    return cycle[(rows * cols) % period].astype(np.int8)


def _recover(input_grid, base, erasure):
    """Find the unique completion using visible cells, without generation state."""
    height, width = input_grid.shape
    visible = input_grid != erasure
    vr, vc = np.nonzero(visible)
    colors = input_grid[visible].astype(np.int16)
    answer = None
    for period in range(5, 10):
        if height < 2 * period or width < 2 * period:
            continue
        both = visible[period:, :] & visible[:-period, :]
        if np.any((input_grid[period:, :] != input_grid[:-period, :]) & both):
            continue
        both = visible[:, period:] & visible[:, :-period]
        if np.any((input_grid[:, period:] != input_grid[:, :-period]) & both):
            continue
        for row_phase in range(period):
            rr = (vr + row_phase) % period
            for col_phase in range(period):
                phases = (rr * ((vc + col_phase) % period)) % period
                low = np.full(period, 10, dtype=np.int16)
                high = np.full(period, -1, dtype=np.int16)
                np.minimum.at(low, phases, colors)
                np.maximum.at(high, phases, colors)
                if (np.any(low != high) or low[0] != base
                        or len(set(map(int, low))) != period):
                    continue
                cycle = low.astype(np.int8)
                predicted = _table(height, width, period, cycle,
                                   row_phase, col_phase)
                if answer is not None and not np.array_equal(predicted, answer[0]):
                    return None
                answer = (predicted, period, cycle, row_phase, col_phase)
    return answer


def _phase_copy_missing(mask, period, row_phase, col_phase):
    h, w = mask.shape
    for r in range(period):
        rows = np.arange(h)[(np.arange(h) + row_phase) % period == r]
        for c in range(period):
            cols = np.arange(w)[(np.arange(w) + col_phase) % period == c]
            if len(rows) and len(cols) and np.all(mask[np.ix_(rows, cols)]):
                return True
    return False


def _mask_geometry(mask, period, row_phase, col_phase):
    h, w = mask.shape
    row_seams = [r for r in range(1, h) if (r + row_phase) % period == 0]
    col_seams = [c for c in range(1, w) if (c + col_phase) % period == 0]
    crossing = (any(np.any(mask[r - 1] & mask[r]) for r in row_seams)
                or any(np.any(mask[:, c - 1] & mask[:, c]) for c in col_seams))
    seen = np.zeros(mask.shape, dtype=bool)
    single = thin = ragged = large = False
    for rr, cc in np.argwhere(mask):
        rr, cc = int(rr), int(cc)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        n = len(cells)
        single |= n == 1
        thin |= n >= 2 and (min(rs) == max(rs) or min(cs) == max(cs))
        ragged |= n >= 5 and n < (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        large |= n >= 25
    edge = bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any())
    return crossing, single, thin, ragged, large, edge


def _add_shape(mask, rng, kind, period):
    h, w = mask.shape
    if kind == 0:
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    elif kind == 1:
        length = int(rng.integers(2, min(11, max(h, w)) + 1))
        if int(rng.integers(2)):
            length = min(length, w)
            r, c = int(rng.integers(h)), int(rng.integers(w - length + 1))
            mask[r, c:c + length] = True
        else:
            length = min(length, h)
            r, c = int(rng.integers(h - length + 1)), int(rng.integers(w))
            mask[r:r + length, c] = True
    elif kind == 2:
        rh = int(rng.integers(2, min(h, period + 4) + 1))
        rw = int(rng.integers(2, min(w, period + 4) + 1))
        r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        mask[r:r + rh, c:c + rw] = True
        for _ in range(int(rng.integers(1, max(3, rh + rw)))):
            rr = int(np.clip(r + rng.integers(-1, rh + 1), 0, h - 1))
            cc = int(np.clip(c + rng.integers(-1, rw + 1), 0, w - 1))
            mask[rr, cc] = True
    elif kind == 3:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(5, 3 * period + 5))):
            mask[r, c] = True
            step = int(rng.integers(4))
            r = int(np.clip(r + (step == 0) - (step == 1), 0, h - 1))
            c = int(np.clip(c + (step == 2) - (step == 3), 0, w - 1))
    else:
        length = int(rng.integers(2, min(9, h, w) + 1))
        side = int(rng.integers(4))
        if side == 0:
            mask[:length, int(rng.integers(w))] = True
        elif side == 1:
            mask[h - length:, int(rng.integers(w))] = True
        elif side == 2:
            mask[int(rng.integers(h)), :length] = True
        else:
            mask[int(rng.integers(h)), w - length:] = True


def _sample_mask(rng, shape, period, row_phase, col_phase,
                 regime, clean_key, key_erased, rule_required, no_seam):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if no_seam:
        # Place sparse breaks strictly inside repeated tiles. This gives
        # genuinely seam-free examples at any canvas aspect ratio.
        row_ok = (np.arange(h) + row_phase) % period
        col_ok = (np.arange(w) + col_phase) % period
        eligible = ((row_ok >= 2) & (row_ok <= period - 2))[:, None] & (
            (col_ok >= 1) & (col_ok <= period - 2))[None, :]
        mask = eligible & (rng.random(shape) < float(rng.uniform(.18, .38)))
        return mask
    if rule_required:
        # Erase every copy of one coordinate phase, while other phase
        # positions still display the same product color.
        choices = [(r, c) for r in range(1, period)
                   for c in range(1, period) if (r * c) % period]
        r, c = choices[int(rng.integers(len(choices)))]
        mask[((np.arange(h) + row_phase) % period == r)[:, None]
             & ((np.arange(w) + col_phase) % period == c)[None, :]] = True
    if key_erased:
        base_row = (-row_phase) % period
        cycle_row = (1 - row_phase) % period
        base_col = (-col_phase) % period
        for row in (base_row, cycle_row):
            start = int(rng.integers(1, min(w - 2, period) + 1))
            mask[row, start:start + int(rng.integers(1, 4))] = True
        mask[int(rng.integers(1, min(h, period + 1))), base_col] = True
    density = float(rng.choice((0.06, 0.12, 0.19) if regime == "source_train"
                               else (0.16, 0.24, 0.34)))
    if clean_key:
        density = min(density, 0.12)
    target = max(int(h * w * density), int(mask.sum()) + 1)
    for _ in range(80):
        if int(mask.sum()) >= target:
            break
        _add_shape(mask, rng, int(rng.integers(5)), period)
    if clean_key:
        mask[0, :period + 1] = False
        mask[1, :period + 1] = False
    return mask


def _make_pair(rng, regime, period, base, erasure, color_order, reorder,
               anchored, key_erased, rule_required, seen):
    for _ in range(100):
        h = int(rng.integers(2 * period + 2, 31))
        w = int(rng.integers(2 * period + 2, 31))
        if rng.random() < 0.12:
            h = 30
        if rng.random() < 0.12:
            w = 30
        if anchored:
            row_phase = col_phase = 0
        else:
            row_phase = int(rng.integers(period))
            col_phase = int(rng.integers(period))
            if row_phase == col_phase == 0:
                row_phase = 1
        order = color_order.copy()
        if reorder:
            rng.shuffle(order)
        cycle = np.array([base] + list(map(int, order[:period - 1])), dtype=np.int8)
        output = _table(h, w, period, cycle, row_phase, col_phase)
        mask = _sample_mask(rng, (h, w), period, row_phase, col_phase,
                            regime, regime == "source_train" and anchored and not key_erased,
                            key_erased, rule_required,
                            regime == "source_train" and anchored and rng.random() < .38)
        if not mask.any():
            continue
        input_grid = output.copy()
        input_grid[mask] = erasure
        recovered = _recover(input_grid, base, erasure)
        if recovered is None or not np.array_equal(recovered[0], output):
            continue
        pair = {"input": input_grid, "output": output}
        sig = _signature(input_grid)
        if sig in seen or _pair_digest(pair) in _SOURCE_PAIR_DIGESTS:
            continue
        seen.add(sig)
        return pair
    raise RuntimeError("Could not construct a uniquely recoverable pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    erasure = int(rng.integers(10))
    base = int(rng.choice([c for c in range(10) if c != erasure]))
    color_order = np.array([c for c in range(10) if c not in (base, erasure)],
                           dtype=np.int8)
    rng.shuffle(color_order)
    reorder_pairs = bool(rng.random() < .65)
    periods = list(map(int, rng.permutation(np.arange(5, 9))))
    train_periods = periods[:min(4, num_train)]
    while len(train_periods) < num_train:
        train_periods.append(int(rng.integers(5, 9)))
    test_periods = [9] if rng.random() < .60 else []
    familiar = list(dict.fromkeys(train_periods))
    rng.shuffle(familiar)
    for period in familiar:
        if len(test_periods) == num_test:
            break
        test_periods.append(period)
    rng.shuffle(test_periods)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, period in enumerate(train_periods):
        pair = _make_pair(rng, "source_train", period, base, erasure,
                          color_order, reorder_pairs, anchored=index == 0,
                          key_erased=index == 1 or (index > 1 and rng.random() < .35),
                          rule_required=index == 1 or (index > 1 and rng.random() < .4),
                          seen=seen)
        puzzle["train"].append(pair)
    for index, period in enumerate(test_periods):
        pair = _make_pair(rng, "target_test", period, base, erasure,
                          color_order, reorder_pairs, anchored=rng.random() < .24,
                          key_erased=rng.random() < .65,
                          rule_required=index == 0 or rng.random() < .65,
                          seen=seen)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    mask = inp != out
    assert mask.any()
    erasures = np.unique(inp[mask])
    assert len(erasures) == 1
    erasure = int(erasures[0])
    assert erasure not in out
    assert np.array_equal(inp[~mask], out[~mask])
    bases = []
    for color in np.unique(out):
        if np.any(np.all(out == color, axis=1)) and np.any(np.all(out == color, axis=0)):
            bases.append(int(color))
    assert len(bases) == 1
    base = bases[0]
    recovered = _recover(inp, base, erasure)
    assert recovered is not None
    predicted, period, cycle, row_phase, col_phase = recovered
    assert np.array_equal(predicted, out)
    assert len(np.unique(out)) == period
    return period, base, erasure, mask, row_phase, col_phase, cycle


def _predict_from_training(train_pairs, test_input):
    """Predict from demonstration grids and the test input alone."""
    demonstration = {key: value.copy() for key, value in train_pairs[0].items()}
    _, base, erasure, _, _, _, _ = _inspect_pair(demonstration)
    recovered = _recover(test_input.copy(), base, erasure)
    assert recovered is not None
    return recovered[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    facts = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            sig = _signature(pair["input"])
            pair_sig = (sig, _signature(pair["output"]))
            assert sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(sig)
            seen_pairs.add(pair_sig)
            assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
            facts[section].append(_inspect_pair(pair))
    bases = {f[1] for values in facts.values() for f in values}
    erasures = {f[2] for values in facts.values() for f in values}
    assert len(bases) == len(erasures) == 1
    assert all(5 <= f[0] <= 8 for f in facts["train"])
    assert all(5 <= f[0] <= 9 for f in facts["test"])
    assert len({f[0] for f in facts["train"]}) >= 2
    assert facts["train"][0][4:6] == (0, 0)
    assert any(f[4:6] != (0, 0) for f in facts["train"])
    assert any(_phase_copy_missing(f[3], f[0], f[4], f[5])
               for f in facts["train"])
    for f in facts["test"]:
        assert int(f[3].sum()) >= int(.12 * f[3].size)
    for pair in puzzle["test"]:
        assert np.array_equal(
            _predict_from_training(puzzle["train"], pair["input"]),
            pair["output"],
        )
    if len(facts["test"]) > 1:
        assert len({f[0] for f in facts["test"]}) == len(facts["test"])
        assert len({_signature(pair["output"])
                    for pair in puzzle["test"]}) == len(puzzle["test"])


def _same_puzzle(left, right):
    return all(
        len(left[section]) == len(right[section]) and
        all(np.array_equal(a[k], b[k]) for a, b in
            zip(left[section], right[section]) for k in ("input", "output"))
        for section in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, periods, bases, erasures, cycles, masks = set(), set(), set(), set(), set(), set()
    shifted = partial_key = phase_holes = rectangular = 0
    seam_free = seam_cross = isolated = thin = ragged = large = edge = 0
    sparse = heavy = combinations = familiar_tests = new_tests = cycle_vary = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        validate_generated_puzzle(puzzle)
        local_cycles = []
        train_periods = set()
        test_periods = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                period, base, erasure, mask, rp, cp, cycle = _inspect_pair(pair)
                assert np.array_equal(out, _table(*out.shape, period, cycle, rp, cp))
                sizes.add(out.shape)
                periods.add(period)
                bases.add(base)
                erasures.add(erasure)
                cycles.add(tuple(map(int, cycle)))
                masks.add(_signature(mask))
                local_cycles.append(tuple(map(int, cycle[:3])))
                (train_periods if section == "train" else test_periods).add(period)
                shifted += (rp, cp) != (0, 0)
                partial_key += bool(mask[(-rp) % period].any() or
                                    mask[(1 - rp) % period].any())
                phase_holes += _phase_copy_missing(mask, period, rp, cp)
                rectangular += out.shape[0] != out.shape[1]
                crossing, one, line, irregular, big, border = _mask_geometry(
                    mask, period, rp, cp)
                seam_cross += crossing
                seam_free += not crossing
                isolated += one
                thin += line
                ragged += irregular
                large += big
                edge += border
                sparse += mask.mean() < .10
                heavy += mask.mean() >= .25
                combinations += ((rp, cp) != (0, 0) and
                                 out.shape[0] != out.shape[1] and
                                 (mask[(-rp) % period].any() or
                                  mask[(1 - rp) % period].any()) and
                                 _phase_copy_missing(mask, period, rp, cp))
        familiar_tests += bool(train_periods & test_periods)
        new_tests += 9 in test_periods
        cycle_vary += len(set(local_cycles)) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 75 and len(periods) == 5
    assert len(bases) >= 7 and len(erasures) >= 7 and len(cycles) >= 60
    assert len(masks) >= 100 and shifted >= 40 and partial_key >= 30
    assert phase_holes >= 42 and rectangular >= 75
    assert 1 in bases and seam_free >= 5 and seam_cross >= 75
    assert isolated >= 50 and thin >= 30 and ragged >= 50
    assert large >= 30 and edge >= 50 and sparse >= 10 and heavy >= 10
    assert combinations >= 10 and familiar_tests >= 10 and new_tests >= 10
    assert cycle_vary >= 15
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1234, *counts)
        validate_generated_puzzle(p)
        assert _same_puzzle(p, generate_puzzle(1234, *counts))
        if counts[0] is not None:
            assert len(p["train"]) == counts[0]
        if counts[1] is not None:
            assert len(p["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0),
                   (None, 5), (2, 2), (2, 4), (4, 4)):
        try:
            generate_puzzle(1234, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid counts accepted: {counts}")
    assert not _same_puzzle(generate_puzzle(10), generate_puzzle(11))
    damaged = generate_puzzle(77, 2, 1)
    damaged["test"][0]["output"][2, 2] = (
        int(damaged["test"][0]["output"][2, 2]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a corrupted output")
