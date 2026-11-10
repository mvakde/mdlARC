"""Solved puzzles whose two aligned panels are combined by inclusive OR."""

import numpy as np


_SOURCE_PAIRS = {
    (("6064000", "6604808", "6664080"), ("202", "222", "222")),
    (("0064880", "0604088", "0664800"), ("222", "022", "222")),
    (("0064808", "6064000", "0664808"), ("202", "202", "222")),
    (("0064808", "0604080", "0004800"), ("202", "020", "200")),
    (("6004008", "0604008", "0604880"), ("202", "022", "220")),
    (("0664008", "0604888", "6064000"), ("022", "222", "202")),
}

_MODES = (
    "scatter", "shape", "containment", "disjoint", "near_complement",
    "empty_panel", "full_panel", "identical", "blank_union", "full_union",
)
_SIZES = np.array((2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14))
_SIZE_WEIGHTS = np.array((9, 16, 16, 15, 12, 10, 8, 5, 4, 3, 2), dtype=float)
_SIZE_WEIGHTS /= _SIZE_WEIGHTS.sum()


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _source_copy(pair):
    return (_rows(pair["input"]), _rows(pair["output"])) in _SOURCE_PAIRS


def _geometry(rng, forced_orientation=None, avoid=None):
    for _ in range(30):
        orientation = forced_orientation or ("vertical" if rng.integers(2) == 0 else "horizontal")
        h = int(rng.choice(_SIZES, p=_SIZE_WEIGHTS))
        w = int(rng.choice(_SIZES, p=_SIZE_WEIGHTS))
        if orientation == "vertical" and h == 14:
            h = 12
        if orientation == "horizontal" and w == 14:
            w = 12
        value = (h, w, orientation)
        if avoid is None or value not in avoid:
            return value
    return (5, 7, forced_orientation or "vertical")


def _primitive(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    motif = int(rng.integers(7))
    r1, r2 = sorted(int(x) for x in rng.integers(0, h, size=2))
    c1, c2 = sorted(int(x) for x in rng.integers(0, w, size=2))
    if motif == 0:
        mask[int(rng.integers(h)), c1:c2 + 1] = True
    elif motif == 1:
        mask[r1:r2 + 1, int(rng.integers(w))] = True
    elif motif == 2:
        mask[r1:r2 + 1, c1:c2 + 1] = True
    elif motif == 3:
        mask[r1, c1:c2 + 1] = mask[r2, c1:c2 + 1] = True
        mask[r1:r2 + 1, c1] = mask[r1:r2 + 1, c2] = True
    elif motif == 4:
        offset = int(rng.integers(-h + 1, w))
        ascending = bool(rng.integers(2))
        for r in range(h):
            c = r + offset if ascending else w - 1 - r - offset
            if 0 <= c < w:
                mask[r, c] = True
    elif motif == 5:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, :] = True
        mask[:, c] = True
    else:
        mask[r1:r2 + 1, c1:c2 + 1] = True
        if r2 - r1 >= 2 and c2 - c1 >= 2:
            mask[r1 + 1:r2, c1 + 1:c2] = False
    return mask


def _scatter(rng, h, w):
    density = float(rng.uniform(0.08, 0.84))
    return rng.random((h, w)) < density


def _masks(rng, h, w, mode, previous_union=None):
    if mode == "same_union" and previous_union is not None:
        states = rng.choice(np.array((1, 2, 3), dtype=np.int8), size=(h, w))
        states = np.where(previous_union, states, 0)
        return (states & 1) != 0, (states & 2) != 0
    if mode == "scatter":
        return _scatter(rng, h, w), _scatter(rng, h, w)
    if mode == "shape":
        a, b = _primitive(rng, h, w), _primitive(rng, h, w)
        if rng.random() < 0.35:
            a |= _primitive(rng, h, w)
        if rng.random() < 0.35:
            b |= _primitive(rng, h, w)
        return a, b
    if mode == "containment":
        outer = _primitive(rng, h, w) if rng.random() < 0.5 else _scatter(rng, h, w)
        inner = outer & (rng.random((h, w)) < rng.uniform(0.3, 0.85))
        return (outer, inner) if rng.integers(2) else (inner, outer)
    if mode == "disjoint":
        a = _scatter(rng, h, w)
        return a, (~a) & _scatter(rng, h, w)
    if mode == "near_complement":
        a = _scatter(rng, h, w)
        b = ~a
        flips = int(rng.integers(1, max(2, h * w // 4 + 1)))
        indices = rng.choice(h * w, size=flips, replace=False)
        b.flat[indices] = ~b.flat[indices]
        return a, b
    if mode == "empty_panel":
        a = _primitive(rng, h, w) if rng.integers(2) else _scatter(rng, h, w)
        return (a, np.zeros((h, w), dtype=bool)) if rng.integers(2) else (np.zeros((h, w), dtype=bool), a)
    if mode == "full_panel":
        a = _scatter(rng, h, w)
        return (a, np.ones((h, w), dtype=bool)) if rng.integers(2) else (np.ones((h, w), dtype=bool), a)
    if mode == "identical":
        a = _primitive(rng, h, w) if rng.integers(2) else _scatter(rng, h, w)
        return a, a.copy()
    if mode == "blank_union":
        z = np.zeros((h, w), dtype=bool)
        return z, z.copy()
    if mode == "full_union":
        a = _scatter(rng, h, w)
        b = ~a | (rng.random((h, w)) < rng.uniform(0.1, 0.5))
        return a, b
    raise ValueError("unknown mask mode")


def _force_states(rng, left, right, required):
    positions = rng.choice(left.size, size=len(required), replace=False)
    for flat, state in zip(positions, required):
        left.flat[flat] = bool(state & 1)
        right.flat[flat] = bool(state & 2)
    return left, right


def _colors(rng, base, left, right, anchor=False):
    if anchor or not (left.any() and (~left).any() and right.any() and (~right).any()):
        roles = base.copy()
        if not anchor and rng.random() < 0.4:
            roles["left"], roles["right"] = roles["right"], roles["left"]
        return roles
    kind = int(rng.choice(3, p=(0.30, 0.25, 0.45)))
    if kind == 0:
        return base.copy()
    if kind == 1:
        roles = base.copy()
        roles["left"], roles["right"] = roles["right"], roles["left"]
        return roles
    pool = np.array([c for c in range(10) if c != base["result"]])
    values = rng.choice(pool, size=4, replace=False)
    return dict(zip(("background", "divider", "left", "right"), map(int, values))) | {"result": base["result"]}


def _build(left, right, orientation, roles):
    h, w = left.shape
    b, d = roles["background"], roles["divider"]
    l = np.where(left, roles["left"], b)
    r = np.where(right, roles["right"], b)
    if orientation == "vertical":
        inp = np.concatenate((l, np.full((h, 1), d), r), axis=1)
    else:
        inp = np.concatenate((l, np.full((1, w), d), r), axis=0)
    out = np.where(left | right, roles["result"], b)
    return {"input": inp.astype(np.int8), "output": out.astype(np.int8)}


def _split(inp, shape):
    h, w = shape
    if inp.shape == (h, 2 * w + 1):
        return inp[:, :w], inp[:, w + 1:], inp[:, w], "vertical"
    if inp.shape == (2 * h + 1, w):
        return inp[:h, :], inp[h + 1:, :], inp[h, :], "horizontal"
    raise AssertionError("input must contain two equal aligned panels and a divider")


def _inspect_pair(pair, base, result, anchor=False):
    inp, out = pair["input"], pair["output"]
    left, right, divider_line, orientation = _split(inp, out.shape)
    divider = int(divider_line.flat[0])
    assert np.all(divider_line == divider)
    lv, rv = set(map(int, np.unique(left))), set(map(int, np.unique(right)))
    assert divider not in lv | rv
    common = lv & rv
    if len(common) == 1:
        background = next(iter(common))
    else:
        assert base is not None
        background = base["background"]
    assert background != divider and background != result
    assert len(lv - {background}) <= 1 and len(rv - {background}) <= 1
    lf, rf = lv - {background}, rv - {background}
    assert not lf or not rf or lf != rf
    assert result not in lv | rv | {divider}
    mixed = len(lv) == 2 and len(rv) == 2
    if not mixed:
        assert base is not None
        assert background == base["background"] and divider == base["divider"]
        assert (lv | rv) - {background} <= {base["left"], base["right"]}
    if anchor:
        assert mixed and len(common) == 1
        assert lf and rf and lf != rf
        assert set(map(int, np.unique(out))) == {background, result}
    lm, rm = left != background, right != background
    expected = np.where(lm | rm, result, background)
    assert np.array_equal(out, expected)
    assert not _source_copy(pair)
    states = lm.astype(np.int8) + 2 * rm.astype(np.int8)
    return states, (out.shape[0], out.shape[1], orientation), background, divider, lf, rf


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_n, test_n = _counts(rng, int(seed), num_train, num_test)
    values = list(map(int, rng.permutation(10)[:5]))
    base = dict(zip(("background", "divider", "left", "right", "result"), values))
    puzzle = {"train": [], "test": []}
    used = set()
    train_geometries = []
    first_union = None
    first_geometry = None
    for split, count in (("train", train_n), ("test", test_n)):
        for index in range(count):
            for attempt in range(100):
                anchor = split == "train" and index == 0
                second = split == "train" and index == 1
                repeat_union = split == "train" and index == 2 and train_n >= 3 and int(seed) % 2 == 0
                if repeat_union:
                    h, w, orientation = first_geometry
                    mode = "same_union"
                else:
                    if anchor and rng.random() < 0.28:
                        h, w, orientation = 3, 3, "vertical"
                    elif split == "test" and rng.random() < 0.35:
                        h, w, orientation = train_geometries[int(rng.integers(len(train_geometries)))]
                    else:
                        orientation = "vertical" if anchor else None
                        if second:
                            orientation = "horizontal" if first_geometry[2] == "vertical" else "vertical"
                        avoid = set(train_geometries) if (second or (split == "test" and rng.random() < 0.65)) else None
                        h, w, orientation = _geometry(rng, orientation, avoid)
                    mode = ("scatter" if anchor else "shape") if (anchor or second) else _MODES[int(rng.integers(len(_MODES)))]
                left, right = _masks(rng, h, w, mode, first_union if repeat_union else None)
                if anchor:
                    left, right = _force_states(rng, left, right, (0, 1, 3))
                elif second:
                    left, right = _force_states(rng, left, right, (0, 2, 3))
                roles = base.copy() if repeat_union else _colors(rng, base, left, right, anchor)
                pair = _build(left, right, orientation, roles)
                key = _key(pair["input"])
                if key in used or _source_copy(pair):
                    continue
                used.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_geometries.append((h, w, orientation))
                    if anchor:
                        first_union = left | right
                        first_geometry = (h, w, orientation)
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
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
    first = puzzle["train"][0]
    left, right, divider_line, _ = _split(first["input"], first["output"].shape)
    common = set(map(int, np.unique(left))) & set(map(int, np.unique(right)))
    assert len(common) == 1
    background = next(iter(common))
    output_other = set(map(int, np.unique(first["output"]))) - {background}
    assert len(output_other) == 1
    result = next(iter(output_other))
    assert len(set(map(int, np.unique(divider_line)))) == 1
    base = {
        "background": background,
        "divider": int(divider_line.flat[0]),
        "left": next(iter(set(map(int, np.unique(left))) - {background})),
        "right": next(iter(set(map(int, np.unique(right))) - {background})),
        "result": result,
    }
    assert len(set(base.values())) == 5
    observed = set()
    for i, pair in enumerate(puzzle["train"]):
        states, _, _, _, _, _ = _inspect_pair(pair, base, result, anchor=i == 0)
        observed.update(map(int, np.unique(states)))
    assert observed == {0, 1, 2, 3}
    for pair in puzzle["test"]:
        _inspect_pair(pair, base, result)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_key(x["input"]) for x in p["train"] + p["test"]) for p in puzzles}) == 42
    for seed, train, test in ((700, 2, 1), (701, 10, 1), (702, 10, 4), (703, None, 1), (704, None, 4), (705, 2, None), (706, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True}, {"num_test": False}):
        try:
            generate_puzzle(800, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request was accepted")
    a, b = generate_puzzle(999, 8, 3), generate_puzzle(999, 8, 3)
    assert all(np.array_equal(x[field], y[field]) for split in ("train", "test") for x, y in zip(a[split], b[split]) for field in ("input", "output"))
    shapes, orientations, state_sets, densities, backgrounds, palettes = set(), set(), set(), set(), set(), set()
    empty_panel = full_panel = disjoint = contained = full_union = blank_union = 0
    same_union = full_rows_train = full_rows_test = new_test_shapes = shared_test_shapes = 0
    changed_backgrounds = swapped_foregrounds = small_original_size = 0
    mixed_train_orientations = varied_train_shapes = 0
    for p in puzzles:
        outputs = [_key(x["output"]) for x in p["train"]]
        same_union += len(outputs) - len(set(outputs))
        pair_backgrounds = set()
        train_shapes = set()
        train_orientations = set()
        first_left, first_right, _, _ = _split(
            p["train"][0]["input"], p["train"][0]["output"].shape
        )
        base_background = next(iter(
            set(map(int, np.unique(first_left)))
            & set(map(int, np.unique(first_right)))
        ))
        base_left = next(iter(set(map(int, np.unique(first_left))) - {base_background}))
        base_right = next(iter(set(map(int, np.unique(first_right))) - {base_background}))
        result = next(iter(set(map(int, np.unique(p["train"][0]["output"]))) - {base_background}))
        for split in ("train", "test"):
            for pair in p[split]:
                out = pair["output"]
                l, r, _, orientation = _split(pair["input"], out.shape)
                shapes.add(out.shape)
                orientations.add(orientation)
                shared = set(map(int, np.unique(l))) & set(map(int, np.unique(r)))
                bg = next(iter(shared)) if len(shared) == 1 else base_background
                pair_backgrounds.add(bg)
                lm, rm = l != bg, r != bg
                assert np.array_equal(out, np.where(lm | rm, result, bg))
                states = lm.astype(np.int8) + 2 * rm.astype(np.int8)
                state_sets.add(tuple(sorted(map(int, np.unique(states)))))
                densities.add(int(np.count_nonzero(out != bg) * 10 / out.size))
                empty_panel += int(not lm.any() or not rm.any())
                full_panel += int(lm.all() or rm.all())
                disjoint += int(not (lm & rm).any() and lm.any() and rm.any())
                contained += int(np.all(lm <= rm) or np.all(rm <= lm))
                full_union += int((lm | rm).all())
                blank_union += int(not (lm | rm).any())
                small_original_size += int(out.shape == (3, 3))
                full_row = bool(np.any(np.all(lm, axis=1)) or np.any(np.all(rm, axis=1)))
                if split == "train":
                    train_shapes.add(out.shape)
                    train_orientations.add(orientation)
                    full_rows_train += int(full_row)
                else:
                    full_rows_test += int(full_row)
                    new_test_shapes += int(out.shape not in train_shapes)
                    shared_test_shapes += int(out.shape in train_shapes)
                if lm.any() and rm.any():
                    left_fg = next(iter(set(map(int, np.unique(l))) - {bg}))
                    right_fg = next(iter(set(map(int, np.unique(r))) - {bg}))
                    swapped_foregrounds += int(left_fg == base_right and right_fg == base_left)
        mixed_train_orientations += int(len(train_orientations) == 2)
        varied_train_shapes += int(len(train_shapes) > 1)
        changed_backgrounds += int(len(pair_backgrounds) > 1)
        backgrounds.add(len(pair_backgrounds))
        palettes.add(tuple(sorted(set(map(int, p["train"][0]["input"].flat)))))
    assert len(shapes) >= 20 and orientations == {"vertical", "horizontal"}
    assert len(state_sets) >= 10 and len(densities) >= 8
    assert min(backgrounds) >= 1 and max(backgrounds) >= 2
    assert len(palettes) >= 24
    assert min(empty_panel, full_panel, disjoint, contained, full_union, blank_union, same_union) > 0
    assert mixed_train_orientations == 42 and varied_train_shapes >= 30
    assert changed_backgrounds >= 20 and swapped_foregrounds >= 10
    assert small_original_size >= 5
    assert full_rows_train > 0 and 0 < full_rows_test < sum(len(p["test"]) for p in puzzles)
    assert new_test_shapes > 0 and shared_test_shapes > 0
