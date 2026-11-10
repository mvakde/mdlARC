import numpy as np


_SOURCE_UNITS = {
    (2, (1, 4)), (5, (1, 4, 2, 5, 3)),
    (7, (1, 4, 7, 3, 6, 2, 5)),
    (8, (1, 4, 7, 2, 5, 8, 3, 6)),
}


def _counts(rng, num_train, num_test):
    for name, value, lo, hi in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 5) if rng.random() < 0.45
                        else rng.integers(5, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _minimal_period(unit):
    p = len(unit)
    return next(
        q for q in range(1, p + 1)
        if all(unit[i] == unit[i % q] for i in range(p))
    )


def _unit(rng, structural, erased, period, repeat):
    colors = [c for c in range(10) if c not in (structural, erased)]
    if not repeat or period == 2:
        return (structural,) + tuple(
            int(v) for v in rng.choice(colors, size=period - 1, replace=False)
        )
    for _ in range(40):
        n = int(rng.integers(2, period))
        accents = rng.choice(colors, size=n - 1, replace=False)
        alphabet = [structural] + [int(v) for v in accents]
        unit = [structural] + [
            int(v) for v in rng.choice(alphabet, size=period - 1)
        ]
        if len(set(unit)) < period and len(set(unit)) >= 2 and _minimal_period(unit) == period:
            return tuple(unit)
    return tuple([structural] + [colors[0]] * (period - 1))


def _field(height, width, structural, unit, row_phase, column_phase):
    out = np.full((height, width), structural, dtype=np.int8)
    patterned = np.arange(height) % 2 == row_phase
    out[patterned] = np.asarray(unit, dtype=np.int8)[
        (np.arange(width) + column_phase) % len(unit)
    ]
    return out


def _mask(rng, height, width, row_phase, mode):
    mask = np.zeros((height, width), dtype=bool)

    def rect(y, x, hh, ww):
        mask[y:y + hh, x:x + ww] = True

    def random_rect(long=False):
        hh = int(rng.integers(1, min(height, 7) + 1))
        ww = int(rng.integers(2, min(width, 9) + 1))
        if long:
            if rng.integers(2):
                hh = int(rng.integers(max(2, height // 2), height + 1))
            else:
                ww = int(rng.integers(max(3, width // 2), width + 1))
        y = int(rng.integers(height - hh + 1))
        x = int(rng.integers(width - ww + 1))
        rect(y, x, hh, ww)

    if mode == 0:
        random_rect()
    elif mode == 1:
        for _ in range(int(rng.integers(2, 7))):
            random_rect()
    elif mode == 2:
        hh = int(rng.integers(1, min(height, 5) + 1))
        ww = int(rng.integers(max(3, width // 2), width + 1))
        rect(int(rng.integers(height - hh + 1)),
             int(rng.integers(width - ww + 1)), hh, ww)
    elif mode == 3:
        hh = int(rng.integers(max(2, height // 2), height + 1))
        ww = int(rng.integers(1, min(width, 5) + 1))
        rect(int(rng.integers(height - hh + 1)),
             int(rng.integers(width - ww + 1)), hh, ww)
    elif mode == 4:
        random_rect(long=True)
        for _ in range(int(rng.integers(2, 5))):
            yy, xx = np.where(mask)
            pick = int(rng.integers(len(yy)))
            hh = int(rng.integers(1, min(5, height) + 1))
            ww = int(rng.integers(1, min(7, width) + 1))
            y = int(np.clip(yy[pick] - rng.integers(hh), 0, height - hh))
            x = int(np.clip(xx[pick] - rng.integers(ww), 0, width - ww))
            rect(y, x, hh, ww)
    elif mode in (5, 6):
        parity = 1 - row_phase if mode == 5 else row_phase
        rows = np.arange(parity, height, 2)
        count = min(len(rows), int(rng.integers(1, 4)))
        for y in rng.choice(rows, size=count, replace=False):
            ww = int(rng.integers(2, max(3, width // 2) + 1))
            x = int(rng.integers(width - ww + 1))
            rect(int(y), x, 1, ww)
    elif mode == 7:
        rows = np.arange(1 - row_phase, height, 2)
        mask[int(rng.choice(rows))] = True
        random_rect()
    elif mode == 8:
        hh = int(rng.integers(2, min(height, 7) + 1))
        ww = int(rng.integers(2, min(width, 8) + 1))
        side = int(rng.integers(4))
        y = 0 if side == 0 else height - hh if side == 1 else int(rng.integers(height - hh + 1))
        x = 0 if side == 2 else width - ww if side == 3 else int(rng.integers(width - ww + 1))
        rect(y, x, hh, ww)
    else:
        for y in range(row_phase, height, 2):
            ww = int(rng.integers(1, min(width, 5) + 1))
            x = int(rng.integers(width - ww + 1))
            rect(y, x, 1, ww)
    return mask


def _solutions(inp):
    height, width = inp.shape
    answers = {}
    for erased in np.unique(inp):
        erased = int(erased)
        missing = inp == erased
        for row_phase in (0, 1):
            solid_rows = np.arange(height) % 2 != row_phase
            solid = inp[solid_rows]
            visible = solid != erased
            if np.count_nonzero(np.any(visible, axis=1)) < 2:
                continue
            values = np.unique(solid[visible])
            if len(values) != 1:
                continue
            structural = int(values[0])
            if structural == erased:
                continue
            patterned = inp[~solid_rows]
            for period in range(2, min(8, width) + 1):
                unit = []
                for phase in range(period):
                    observed = patterned[:, phase::period]
                    phase_values = np.unique(observed[observed != erased])
                    if len(phase_values) != 1:
                        break
                    unit.append(int(phase_values[0]))
                if len(unit) != period or len(set(unit)) < 2:
                    continue
                if _minimal_period(unit) != period:
                    continue
                restored = _field(height, width, structural, unit, row_phase, 0)
                if np.array_equal(restored[~missing], inp[~missing]):
                    answers[restored.tobytes()] = (restored, erased, row_phase, period)
    return list(answers.values())


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(inp, out, erased, row_phase, period):
    return (
        inp.shape == (30, 30) and erased == 0 and row_phase == 1
        and int(out[0, 0]) == 1
        and (period, tuple(int(v) for v in out[1, :period])) in _SOURCE_UNITS
    )


def _make_pair(rng, period, roles, row_phase, column_phase, mode, require_shift):
    structural, erased = roles
    for _ in range(80):
        height = 30 if rng.random() < 0.15 else int(rng.integers(12, 31))
        width = (30 if rng.random() < 0.15
                 else int(rng.integers(max(12, period + 3), 31)))
        unit = _unit(rng, structural, erased, period,
                     period >= 3 and rng.random() < 0.38)
        if require_shift and unit[column_phase] == structural:
            continue
        out = _field(height, width, structural, unit, row_phase, column_phase)
        mask = _mask(rng, height, width, row_phase, mode)
        if not 3 <= int(mask.sum()) <= int(0.43 * height * width):
            continue
        inp = out.copy()
        inp[mask] = erased
        solutions = _solutions(inp)
        if len(solutions) != 1 or not np.array_equal(solutions[0][0], out):
            continue
        if solutions[0][1:] != (erased, row_phase, period):
            continue
        if _source_copy(inp, out, erased, row_phase, period):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a uniquely recoverable pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    variable_roles = bool(rng.integers(2))
    base_roles = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    periods = [2, int(rng.integers(3, 9))]
    if num_train > 2:
        extra = [p for p in range(3, 9) if p != periods[1]]
        rng.shuffle(extra)
        periods.extend((extra * 2)[:num_train - 2])
    rng.shuffle(periods)
    train_roles = [base_roles]
    if variable_roles:
        alternatives = [(s, e) for s in range(10) for e in range(10)
                        if s != e and (s, e) != base_roles]
        train_roles.append(alternatives[int(rng.integers(len(alternatives)))])
    else:
        train_roles.append(base_roles)
    for _ in range(2, num_train):
        train_roles.append(
            tuple(int(v) for v in rng.choice(10, size=2, replace=False))
            if variable_roles else base_roles
        )
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(num_train + num_test):
        is_train = index < num_train
        period = periods[index] if is_train else int(rng.integers(2, 9))
        if is_train:
            roles = train_roles[index]
            row_phase = index % 2
            column_phase = 0 if index == 0 else int(rng.integers(1, period))
            mode = 0 if index == 0 else 3 if index == 1 else int(rng.integers(10))
        else:
            roles = (tuple(int(v) for v in rng.choice(10, size=2, replace=False))
                     if variable_roles else base_roles)
            row_phase = int(rng.integers(2))
            column_phase = int(rng.integers(period))
            mode = int(rng.integers(10))
        for _ in range(30):
            pair_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            pair = _make_pair(np.random.default_rng(pair_seed), period, roles,
                              row_phase, column_phase, mode,
                              is_train and index == 1)
            key = _signature(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if is_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_phases, train_roles, all_roles, train_periods = set(), set(), set(), set()
    train_shifted = False
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            assert inp.shape[0] >= 12 and inp.shape[1] >= 12
            ik = _signature(inp)
            pk = (ik, _signature(out))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            solutions = _solutions(inp)
            assert len(solutions) == 1
            restored, erased, row_phase, period = solutions[0]
            assert np.array_equal(restored, out)
            mask = inp == erased
            assert np.any(mask) and np.array_equal(inp[~mask], out[~mask])
            assert erased not in out
            assert not _source_copy(inp, out, erased, row_phase, period)
            role = (int(out[1 - row_phase, 0]), erased)
            all_roles.add(role)
            if regime == "train":
                train_phases.add(row_phase)
                train_roles.add(role)
                train_periods.add(period)
                train_shifted |= int(out[row_phase, 0]) != role[0]
    assert train_phases == {0, 1}
    assert len(train_periods) >= 2
    assert train_shifted
    if len(train_roles) == 1:
        assert all_roles == train_roles


def _same_puzzle(a, b):
    return all(
        len(a[key]) == len(b[key]) and all(
            np.array_equal(pa[name], pb[name])
            for pa, pb in zip(a[key], b[key]) for name in ("input", "output")
        ) for key in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, roles, periods, phases, masks = set(), set(), set(), set(), set()
    repeat_units = shifted = partial_widths = full_solid = train_hard = test_easy = 0
    role_change_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_roles = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                restored, erased, rp, p = _solutions(inp)[0]
                assert np.array_equal(restored, out)
                unit = tuple(int(v) for v in out[rp, :p])
                assert _minimal_period(unit) == p
                structural = int(out[1 - rp, 0])
                role = (structural, erased)
                local_roles.add(role)
                roles.add(role)
                shapes.add(out.shape)
                periods.add(p)
                phases.add(rp)
                mask = inp == erased
                masks.add((out.shape, mask.tobytes()))
                repeat_units += len(set(unit)) < p
                shifted += unit[0] != structural
                partial_widths += out.shape[1] % p != 0
                solid_rows = np.arange(out.shape[0]) % 2 != rp
                full_solid += bool(np.any(np.all(mask[solid_rows], axis=1)))
                if regime == "train":
                    train_hard += bool(np.any(mask[:, 0]) or np.any(mask[:, -1])
                                       or np.any(np.all(mask[solid_rows], axis=1)))
                else:
                    test_easy += int(mask.sum()) <= 15
        role_change_puzzles += len(local_roles) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(masks) >= 150
    assert len(roles) >= 15 and periods == set(range(2, 9)) and phases == {0, 1}
    assert repeat_units >= 20 and shifted >= 80 and partial_widths >= 70
    assert full_solid >= 3 and train_hard >= 10 and test_easy >= 3
    assert role_change_puzzles >= 10
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(503, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(503, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _same_puzzle(generate_puzzle(761), generate_puzzle(761))
    assert _same_puzzle(generate_puzzle(761, 7, 3), generate_puzzle(761, 7, 3))
    assert not _same_puzzle(generate_puzzle(761, 7, 3), generate_puzzle(762, 7, 3))
