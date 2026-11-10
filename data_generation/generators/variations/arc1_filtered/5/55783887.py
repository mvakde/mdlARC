import numpy as np


_SOURCE = (
    (9, 9, 4, ((2, 1), (7, 6)), ()),
    (8, 11, 8, ((1, 6), (6, 1)), ((3, 4),)),
    (9, 12, 4, ((1, 7), (7, 1)), ((6, 7),)),
    (15, 12, 5, ((3, 1), (9, 7)), ((2, 9), (5, 3), (8, 6), (13, 4))),
    (15, 16, 8, ((2, 2), (3, 12), (11, 11), (14, 3)),
     ((1, 8), (9, 9), (10, 1), (14, 13))),
    (19, 16, 4, ((2, 2), (4, 14), (10, 10), (17, 9)),
     ((4, 4), (6, 6), (7, 14), (15, 2))),
)


def _source_inputs():
    result = []
    for h, w, bg, paths, branches in _SOURCE:
        grid = np.full((h, w), bg, dtype=np.int8)
        for r, c in paths:
            grid[r, c] = 1
        for r, c in branches:
            grid[r, c] = 6
        result.append(grid)
    return result


def _counts(rng, nt, ns):
    for name, n, lo, hi in (("num_train", nt, 2, 10), ("num_test", ns, 1, 4)):
        if n is not None and (isinstance(n, (bool, np.bool_)) or
                              not isinstance(n, (int, np.integer)) or not lo <= n <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if nt is not None and ns is not None and nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if nt is None:
        nt = int(rng.integers(max(2, (ns or 0) + 1), 11))
    if ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    return int(nt), int(ns)


def _line(h, w, point, slope):
    r, c = point
    return {(rr, c + slope * (rr - r)) for rr in range(h)
            if 0 <= c + slope * (rr - r) < w}


def _shape(rng, regime):
    probs = (0.27, 0.42, 0.31) if regime == "source_train" else (0.30, 0.32, 0.38)
    size = int(rng.choice(3, p=probs))
    lo, hi = ((7, 11), (11, 18), (18, 27))[size]
    side = int(rng.integers(lo, hi + 1))
    aspect = int(rng.integers(3))
    if aspect == 0:
        return min(30, side + int(rng.integers(3, 9))), side
    if aspect == 1:
        return side, min(30, side + int(rng.integers(3, 9)))
    return side, min(30, max(7, side + int(rng.integers(-2, 3))))


def _length(rng, h, w, active):
    lo, hi = max(3, active + 1), min(h, w) - 1
    mode = int(rng.integers(3))
    if mode == 0:
        return int(rng.integers(lo, min(hi, max(lo, 5)) + 1))
    if mode == 1:
        low = min(hi, max(lo, 6))
        return int(rng.integers(low, min(hi, max(low, 12)) + 1))
    return int(rng.integers(max(lo, hi - 5), hi + 1))


def _position(rng, h, w, length, slope):
    positions = [(r, c) for r in range(h - length)
                 for c in (range(w - length) if slope == 1 else range(length, w))]
    edge = [p for p in positions if p[0] == 0 or p[0] + length == h - 1
            or p[1] in (0, w - 1) or p[1] + slope * length in (0, w - 1)]
    corner = [p for p in edge if sum((p[0] == 0, p[0] + length == h - 1,
                                    p[1] in (0, w - 1),
                                    p[1] + slope * length in (0, w - 1))) >= 2]
    mode = int(rng.choice(3, p=(0.52, 0.33, 0.15)))
    pool = (positions, edge or positions, corner or edge or positions)[mode]
    return pool[int(rng.integers(len(pool)))]


def _pick_cells(rng, n, pools, occupied, valid):
    picked = []
    for _ in range(n):
        kind = str(rng.choice(("extension", "near", "far", "all"),
                              p=(0.27, 0.31, 0.27, 0.15)))
        options = [p for p in pools[kind] if p not in occupied and valid(p, picked)]
        if not options:
            options = [p for p in pools["all"] if p not in occupied and valid(p, picked)]
        if not options:
            return None
        p = options[int(rng.integers(len(options)))]
        picked.append(p)
        occupied.add(p)
    return picked


def _make_pair(rng, regime, colors, active_count, slope, extra_path,
               inactive_count, required_inactive=()):
    bg, path, branch = colors
    for _ in range(180):
        h, w = _shape(rng, regime)
        length = _length(rng, h, w, active_count)
        r0, c0 = _position(rng, h, w, length, slope)
        segment = [(r0 + k, c0 + slope * k) for k in range(length + 1)]
        indices = rng.choice(np.arange(1, length), size=active_count, replace=False)
        active = {segment[int(k)] for k in indices}
        lines = set().union(*(_line(h, w, p, -slope) for p in active)) if active else set()
        ends = (segment[0], segment[-1])
        occupied = set(segment) | lines
        cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
        extension = [p for p in cells if p[1] - c0 == slope * (p[0] - r0)]
        near = [p for p in cells if p not in extension and
                min(abs(p[0] - r) + abs(p[1] - c) for r, c in segment) <= 2]
        far = [p for p in cells if p not in extension and p not in near]
        pools = {"all": cells, "extension": extension, "near": near, "far": far}

        def valid_path(p, already):
            return all(abs(p[0] - q[0]) != abs(p[1] - q[1])
                       for q in (*ends, *already))

        extras = _pick_cells(rng, extra_path, pools, occupied, valid_path)
        if extras is None:
            continue
        forced = []
        for kind in required_inactive:
            options = [p for p in pools[kind] if p not in occupied]
            if not options:
                break
            p = options[int(rng.integers(len(options)))]
            forced.append(p)
            occupied.add(p)
        if len(forced) != len(required_inactive):
            continue
        remainder = _pick_cells(rng, inactive_count - len(forced), pools, occupied,
                                lambda p, already: True)
        if remainder is None:
            continue
        inactive = forced + remainder
        inp = np.full((h, w), bg, dtype=np.int8)
        for r, c in (*ends, *extras):
            inp[r, c] = path
        for r, c in active | set(inactive):
            inp[r, c] = branch
        out = inp.copy()
        for r, c in segment:
            if out[r, c] == bg:
                out[r, c] = path
        for r, c in lines:
            out[r, c] = branch
        if not np.array_equal(inp, out):
            return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a diagonal pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    path, branch = (int(x) for x in rng.choice(10, size=2, replace=False))
    backgrounds = [x for x in range(10) if x not in (path, branch)]
    base = 1 if int(rng.integers(2)) else -1
    plans = [(1, base, 0, 0, ()),
             (2, -base, int(rng.integers(1, 3)), int(rng.integers(2, 4)),
              ("extension", "near"))]
    if nt >= 3:
        plans.append((0, 1 if int(rng.integers(2)) else -1, 0,
                      0 if int(rng.integers(2)) else int(rng.integers(1, 4)), ()))
    if nt >= 4:
        plans.append((3, 1 if int(rng.integers(2)) else -1,
                      int(rng.integers(4)), int(rng.integers(5)), ()))
    while len(plans) < nt:
        plans.append((int(rng.integers(4)), 1 if int(rng.integers(2)) else -1,
                      int(rng.integers(5)), int(rng.integers(6)), ()))
    rng.shuffle(plans)
    puzzle = {"train": [], "test": []}
    seen, sources = set(), _source_inputs()
    first_bg = None
    for split in ("train", "test"):
        for index in range(nt if split == "train" else ns):
            prng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
            pool = [x for x in backgrounds if x != first_bg] if split == "train" and index == 1 else backgrounds
            bg = int(prng.choice(pool))
            if split == "train" and index == 0:
                first_bg = bg
            if split == "train":
                active, slope, extras, inactive, required_inactive = plans[index]
            else:
                cap = 3 if nt >= 4 else 2
                active, slope = int(prng.integers(cap + 1)), 1 if int(prng.integers(2)) else -1
                extras, inactive = int(prng.integers(5)), int(prng.integers(6))
                if int(prng.integers(3)) == 0:
                    active = min(cap, int(prng.integers(2, 4)))
                    extras, inactive = int(prng.integers(1, 5)), int(prng.integers(1, 6))
                if int(prng.integers(5)) == 0:
                    extras = inactive = 0
                required_inactive = ()
            for _ in range(30):
                pair = _make_pair(prng, "source_train" if split == "train" else "target_test",
                                  (bg, path, branch), active, slope, extras,
                                  inactive, required_inactive)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and not any(np.array_equal(pair["input"], x) for x in sources):
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(inp, bg, path, branch):
    h, w = inp.shape
    colors = set(np.unique(inp).tolist())
    assert colors <= {bg, path, branch} and bg in colors and path in colors
    paths = [tuple(map(int, p)) for p in np.argwhere(inp == path)]
    pairs = [(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]
             if abs(a[0] - b[0]) == abs(a[1] - b[1])]
    assert len(pairs) == 1
    a, b = sorted(pairs[0])
    length = b[0] - a[0]
    assert length >= 3
    slope = 1 if b[1] > a[1] else -1
    segment = {(a[0] + k, a[1] + slope * k) for k in range(length + 1)}
    branches = {tuple(map(int, p)) for p in np.argwhere(inp == branch)}
    active = branches & segment
    lines = set().union(*(_line(h, w, p, -slope) for p in active)) if active else set()
    assert (set(paths) - {a, b}).isdisjoint(lines)
    assert (branches - active).isdisjoint(lines)
    expected = inp.copy()
    for r, c in segment:
        if expected[r, c] == bg:
            expected[r, c] = path
    for r, c in lines:
        expected[r, c] = branch
    extension = sum(p[1] - a[1] == slope * (p[0] - a[0]) for p in branches - active)
    near_branch = sum(p[1] - a[1] != slope * (p[0] - a[0]) and
                      min(abs(p[0] - r) + abs(p[1] - c) for r, c in segment) <= 2
                      for p in branches - active)
    far_branch = len(branches - active) - extension - near_branch
    near_path = sum(min(abs(p[0] - r) + abs(p[1] - c) for r, c in segment) <= 2
                    for p in set(paths) - {a, b})
    adjacent = any(abs(p[0] - q[0]) == 1 for p in active for q in active if p != q)
    edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in (a, b))
    corner = any(r in (0, h - 1) and c in (0, w - 1) for r, c in (a, b))
    active_near_edge = any(min(r, c, h - 1 - r, w - 1 - c) <= 2 for r, c in active)
    return expected, {"background": bg, "slope": slope, "length": length,
                      "active": len(active), "extra_path": len(paths) - 2,
                      "inactive": len(branches - active), "extension": extension,
                      "near_branch": near_branch, "far_branch": far_branch,
                      "near_path": near_path, "adjacent": adjacent,
                      "edge": edge, "corner": corner,
                      "active_near_edge": active_near_edge, "shape": inp.shape,
                      "colors": len(colors)}


def _roles_and_details(puzzle):
    pairs = puzzle["train"] + puzzle["test"]
    bgs = [int(np.bincount(p["input"].ravel().astype(int), minlength=10).argmax())
           for p in pairs]
    other = set().union(*(set(np.unique(p["input"]).tolist()) - {bg}
                          for p, bg in zip(pairs, bgs)))
    assert len(other) == 2
    colors = sorted(other)
    candidates = []
    for path, branch in ((colors[0], colors[1]), (colors[1], colors[0])):
        try:
            details = []
            for pair, bg in zip(puzzle["train"], bgs):
                predicted, detail = _analyze(pair["input"], bg, path, branch)
                assert np.array_equal(predicted, pair["output"])
                details.append(detail)
            candidates.append((path, branch, details))
        except AssertionError:
            pass
    assert len(candidates) == 1
    path, branch, train = candidates[0]
    test = []
    for pair, bg in zip(puzzle["test"], bgs[len(train):]):
        predicted, detail = _analyze(pair["input"], bg, path, branch)
        assert np.array_equal(predicted, pair["output"])
        test.append(detail)
    return (path, branch), train, test


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, seen_pairs, sources = set(), set(), _source_inputs()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        key = (inp.shape, inp.tobytes())
        pair_key = (key, out.tobytes())
        assert key not in seen and pair_key not in seen_pairs
        seen.add(key)
        seen_pairs.add(pair_key)
        assert not any(np.array_equal(inp, x) for x in sources)
    (path, branch), train, test = _roles_and_details(puzzle)
    assert path != branch and len({d["background"] for d in train}) >= 2
    assert {d["slope"] for d in train} == {-1, 1}
    assert any(d["active"] >= 2 and d["extra_path"] >= 1 and d["inactive"] >= 1
               for d in train)
    assert any(d["extension"] > 0 and d["near_branch"] > 0 for d in train)
    assert any(d["active"] == 1 and d["extra_path"] == d["inactive"] == 0
               for d in train)
    if nt >= 3:
        assert any(d["active"] == 0 for d in train)
    if nt >= 4:
        assert any(d["active"] >= 3 for d in train)
    assert max(d["active"] for d in test) <= max(d["active"] for d in train)


def _same(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x[g], y[g]) for x, y in zip(a[k], b[k])
        for g in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    tc, sc, shapes, roles, backgrounds, signatures = set(), set(), set(), set(), set(), set()
    actives, slopes, combos = set(), set(), set()
    orders, train_extras, test_extras = set(), set(), set()
    train_inactives, test_inactives, test_actives = set(), set(), set()
    edges = adjacent = extensions = bare = big_train = small_test = wide_test = 0
    zero_test = clean_test = test_three = square_test = inactive_only = 0
    two_color_test = bare_test = no_extension_test = 0
    near_branch = far_branch = near_path = corner = active_edge = combined = 0
    source_like_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        role, train, test = _roles_and_details(puzzle)
        roles.add(role)
        tc.add(len(train))
        sc.add(len(test))
        orders.add(tuple(d["active"] for d in train[:3]))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for split in ("train", "test") for p in puzzle[split]))
        for d in train + test:
            shapes.add(d["shape"])
            backgrounds.add(d["background"])
            actives.add(d["active"])
            slopes.add(d["slope"])
            edges += d["edge"]
            adjacent += d["adjacent"]
            extensions += bool(d["extension"])
            near_branch += d["near_branch"]
            far_branch += d["far_branch"]
            near_path += d["near_path"]
            corner += d["corner"]
            active_edge += d["active_near_edge"]
            combined += d["edge"] and d["active"] >= 2 and d["inactive"] > 0 and d["extra_path"] > 0
            combos.add((d["active"], d["extra_path"] > 0, d["inactive"] > 0,
                        d["edge"], d["slope"]))
        bare += sum(d["colors"] == 2 and d["active"] == d["inactive"] == 0
                    for d in train)
        train_extras.update(d["extra_path"] for d in train)
        train_inactives.update(d["inactive"] for d in train)
        inactive_only += sum(d["active"] == 0 and d["inactive"] > 0 for d in train)
        big_train += sum(max(d["shape"]) >= 23 for d in train)
        small_test += sum(max(d["shape"]) <= 14 for d in test)
        wide_test += sum(d["shape"][1] > d["shape"][0] for d in test)
        zero_test += sum(d["active"] == 0 for d in test)
        clean_test += sum(d["extra_path"] == d["inactive"] == 0 for d in test)
        test_extras.update(d["extra_path"] for d in test)
        test_inactives.update(d["inactive"] for d in test)
        test_actives.update(d["active"] for d in test)
        two_color_test += sum(d["colors"] == 2 for d in test)
        bare_test += sum(d["active"] == d["inactive"] == d["extra_path"] == 0
                         for d in test)
        no_extension_test += sum(d["extension"] == 0 for d in test)
        test_three += sum(d["active"] == 3 for d in test)
        square_test += sum(abs(d["shape"][0] - d["shape"][1]) <= 2 for d in test)
        source_like_test += sum(d["shape"][0] > d["shape"][1] and
                                d["shape"][0] >= 18 and d["active"] >= 2 and
                                d["extra_path"] > 0 and d["inactive"] > 0 for d in test)
    assert len(tc) >= 4 and len(sc) >= 3 and len(signatures) == 42
    assert len(roles) >= 20 and len(backgrounds) >= 8 and len(shapes) >= 60
    assert len(combos) >= 30 and actives == {0, 1, 2, 3} and slopes == {-1, 1}
    assert len(orders) >= 10 and train_extras == test_extras == set(range(5))
    assert train_inactives == test_inactives == set(range(6))
    assert test_actives == {0, 1, 2, 3}
    assert edges >= 20 and adjacent >= 10 and extensions >= 10
    assert bare >= 5 and big_train >= 10 and small_test >= 5 and wide_test >= 10
    assert zero_test >= 5 and clean_test >= 2
    assert two_color_test >= 1 and bare_test >= 1 and no_extension_test >= 10
    assert test_three >= 5 and square_test >= 5 and inactive_only >= 5
    assert near_branch >= 10 and far_branch >= 10 and near_path >= 10
    assert corner >= 5 and active_edge >= 10 and combined >= 10
    assert source_like_test >= 5
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        p = generate_puzzle(913, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
        assert _same(p, generate_puzzle(913, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("Illegal pair count was accepted")
