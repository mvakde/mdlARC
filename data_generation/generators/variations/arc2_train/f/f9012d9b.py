"""Generate missing-square, two-color periodic ARC puzzles."""

import numpy as np


_SOURCE_PAIRS = (
    (np.array([[8, 6, 0, 6], [6, 8, 6, 8], [8, 6, 8, 6],
               [6, 8, 6, 8]], dtype=np.int8),
     np.array([[8]], dtype=np.int8)),
    (np.array([[2, 1, 2, 1, 2], [1, 1, 1, 1, 1],
               [2, 1, 2, 1, 2], [0, 0, 1, 1, 1],
               [0, 0, 2, 1, 2]], dtype=np.int8),
     np.array([[1, 1], [2, 1]], dtype=np.int8)),
    (np.array([[2, 2, 5, 2, 2, 5, 2], [2, 2, 5, 2, 2, 5, 2],
               [5, 5, 5, 5, 5, 5, 5], [2, 2, 5, 2, 2, 5, 2],
               [2, 2, 5, 2, 2, 5, 2], [5, 5, 5, 5, 5, 0, 0],
               [2, 2, 5, 2, 2, 0, 0]], dtype=np.int8),
     np.array([[5, 5], [5, 2]], dtype=np.int8)),
    (np.array([[8, 1, 8, 8, 0, 0, 0], [1, 8, 8, 1, 0, 0, 0],
               [8, 8, 1, 8, 0, 0, 0], [8, 1, 8, 8, 1, 8, 8],
               [1, 8, 8, 1, 8, 8, 1], [8, 8, 1, 8, 8, 1, 8],
               [8, 1, 8, 8, 1, 8, 8]], dtype=np.int8),
     np.array([[1, 8, 8], [8, 8, 1], [8, 1, 8]], dtype=np.int8)),
)

_LOCATIONS = ("top_left", "top_right", "bottom_left", "bottom_right",
              "top_edge", "bottom_edge", "left_edge", "right_edge",
              "interior")


def _grid_signature(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _pair_signature(pair):
    return (_grid_signature(pair["input"]), _grid_signature(pair["output"]))


def _is_source_pair(inp, out):
    return any(inp.shape == a.shape and out.shape == b.shape
               and np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in _SOURCE_PAIRS)


def _periodic_candidates(grid, mask):
    """All fully witnessed square periods consistent with visible cells."""
    h, w = grid.shape
    rows, cols = np.nonzero(mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    candidates = []
    for period in range(1, min(h, w) + 1):
        motif = np.full((period, period), -1, dtype=np.int16)
        possible = True
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                value = int(grid[r, c])
                old = motif[r % period, c % period]
                if old != -1 and old != value:
                    possible = False
                    break
                motif[r % period, c % period] = value
            if not possible:
                break
        if not possible or np.any(motif == -1) or len(np.unique(motif)) != 2:
            continue
        fill = motif[np.arange(r0, r1)[:, None] % period,
                     np.arange(c0, c1)[None, :] % period].copy()
        candidates.append((period, motif, fill))
    return candidates


def _candidate_analyses(grid):
    analyses = []
    for color in np.unique(grid):
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        size = int(r1 - r0)
        if size != c1 - c0 or len(coords) != size * size:
            continue
        mask = grid == color
        if len(np.unique(grid[~mask])) != 2:
            continue
        possibilities = _periodic_candidates(grid, mask)
        if not possibilities:
            continue
        if any(not np.array_equal(item[2], possibilities[0][2])
               for item in possibilities[1:]):
            continue
        period, motif, fill = possibilities[0]
        analyses.append({"blank": int(color), "period": period,
                         "motif": motif, "fill": fill, "r0": int(r0),
                         "c0": int(c0), "size": size})
    return analyses


def _location(r0, c0, size, h, w):
    top, bottom = r0 == 0, r0 + size == h
    left, right = c0 == 0, c0 + size == w
    if top and left:
        return "top_left"
    if top and right:
        return "top_right"
    if bottom and left:
        return "bottom_left"
    if bottom and right:
        return "bottom_right"
    if top:
        return "top_edge"
    if bottom:
        return "bottom_edge"
    if left:
        return "left_edge"
    if right:
        return "right_edge"
    return "interior"


def _has_vertical_copy(inp, analysis):
    s, c0, r0 = analysis["size"], analysis["c0"], analysis["r0"]
    h = inp.shape[0]
    blank = analysis["blank"]
    target = analysis["fill"]
    for row in range(h - s + 1):
        if row == r0:
            continue
        patch = inp[row:row + s, c0:c0 + s]
        if not np.any(patch == blank) and np.array_equal(patch, target):
            return True
    return False


def _has_horizontal_witness(inp, analysis):
    p, s = analysis["period"], analysis["size"]
    r0, c0, blank = analysis["r0"], analysis["c0"], analysis["blank"]
    for row in range(r0, r0 + s):
        for col in range(c0, c0 + s):
            if not any(other != col and (other - col) % p == 0
                       and inp[row, other] != blank
                       for other in range(inp.shape[1])):
                return False
    return True


def _validate_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _validate_count(num_train, "num_train", 2, 10)
    ns = _validate_count(num_test, "num_test", 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _choose_period(rng):
    return int(rng.choice([2, 3, 4, 5], p=[0.25, 0.38, 0.27, 0.10]))


def _sample_specs(rng, num_train, num_test):
    # The first training pair simultaneously proves nonuniform reconstruction
    # and horizontal continuation without an intact same-column source patch.
    anchor_period = int(rng.integers(2, 5))
    anchor_size = int(rng.integers(max(3 if num_test == 4 else 2,
                                       (anchor_period + 2) // 2), 6))
    train = [{"period": anchor_period, "size": anchor_size,
              "mode": "horizontal", "mixed": True}]
    uniform_second = bool(anchor_size != 2 and rng.random() < 0.27)
    if uniform_second:
        second_period = int(rng.choice([p for p in (3, 4, 5) if p != anchor_period]))
        second_size = 2
    else:
        second_period = int(rng.choice([p for p in (2, 3, 4, 5) if p != anchor_period]))
        second_size = 1 if rng.random() < 0.6 else int(rng.integers(2, 5))
        if second_size == anchor_size:
            second_size = 1
    train.append({"period": second_period, "size": second_size,
                  "mode": "general", "mixed": False,
                  "uniform": uniform_second})
    while len(train) < num_train:
        p = _choose_period(rng)
        s = int(rng.integers(1, 6))
        horizontal = s >= max(2, (p + 2) // 2) and rng.random() < 0.25
        train.append({"period": p, "size": s,
                      "mode": "horizontal" if horizontal else "general",
                      "mixed": False})
    max_train_size = max(item["size"] for item in train)
    tests = []
    used_sizes = set()
    used_locations = set()
    for index in range(num_test):
        if 3 not in used_sizes and "top_right" not in used_locations and rng.random() < 0.28:
            p, s, mode, location = 3, 3, "source_shift", "top_right"
        else:
            p = _choose_period(rng)
            possible_sizes = [x for x in range(1, min(5, max_train_size + 1) + 1)
                              if x not in used_sizes]
            s = int(rng.choice(possible_sizes))
            available_sides = [x for x in ("left_edge", "right_edge")
                               if x not in used_locations]
            if s >= max(3, (p + 2) // 2) and available_sides and rng.random() < 0.45:
                mode = "horizontal"
                location = str(rng.choice(available_sides))
            else:
                mode = "general"
                location = str(rng.choice([x for x in _LOCATIONS
                                           if x not in used_locations]))
        tests.append({"period": p, "size": s, "mode": mode,
                      "mixed": False, "location": location})
        used_sizes.add(s)
        used_locations.add(location)
    return train, tests


def _choose_geometry(rng, spec):
    p, s, mode = spec["period"], spec["size"], spec["mode"]
    if mode == "horizontal":
        if spec.get("location") in ("left_edge", "right_edge"):
            h = int(rng.integers(max(p, s + 2), 2 * s))
        else:
            h = int(rng.integers(max(p, s + 1), 2 * s))
        w = int(rng.integers(p + s, min(19, p + s + 7)))
        r0 = int(rng.integers(1, h - s if spec.get("location") else h - s + 1))
        side = (0 if spec.get("location") == "left_edge" else
                1 if spec.get("location") == "right_edge" else
                int(rng.integers(0, 2)))
        c0 = 0 if side == 0 else w - s
        return h, w, r0, c0
    h = int(rng.integers(max(2 * p, s + 2), min(22, max(2 * p, s + 2) + 7)))
    w = int(rng.integers(max(2 * p, s + 2), min(22, max(2 * p, s + 2) + 7)))
    loc = spec.get("location", "top_right" if mode == "source_shift"
                   else str(rng.choice(_LOCATIONS)))
    if loc.startswith("top"):
        r0 = 0
    elif loc.startswith("bottom"):
        r0 = h - s
    elif loc == "left_edge" or loc == "right_edge":
        r0 = int(rng.integers(1, h - s))
    else:
        r0 = int(rng.integers(1, h - s))
    if loc.endswith("left") or loc == "left_edge":
        c0 = 0
    elif loc.endswith("right") or loc == "right_edge":
        c0 = w - s
    else:
        c0 = int(rng.integers(1, w - s))
    return h, w, r0, c0


def _generate_pair(rng, spec, blank, forbidden):
    p, s = spec["period"], spec["size"]
    available = np.array([x for x in range(10) if x != blank], dtype=np.int8)
    for _ in range(1200):
        h, w, r0, c0 = _choose_geometry(rng, spec)
        colors = rng.choice(available, 2, replace=False)
        motif = rng.integers(0, 2, size=(p, p), dtype=np.int8)
        if len(np.unique(motif)) != 2:
            continue
        row_phase = int(rng.integers(0, p))
        col_phase = int(rng.integers(0, p))
        complete = colors[motif[(np.arange(h)[:, None] + row_phase) % p,
                                (np.arange(w)[None, :] + col_phase) % p]].astype(np.int8)
        out = complete[r0:r0 + s, c0:c0 + s].copy()
        if spec["mixed"] and len(np.unique(out)) != 2:
            continue
        if spec.get("uniform") and len(np.unique(out)) != 1:
            continue
        inp = complete.copy()
        inp[r0:r0 + s, c0:c0 + s] = blank
        if _grid_signature(inp) in forbidden or _is_source_pair(inp, out):
            continue
        analyses = _candidate_analyses(inp)
        if len(analyses) != 1:
            continue
        a = analyses[0]
        if (a["blank"] != blank or a["period"] != p or a["size"] != s
                or a["r0"] != r0 or a["c0"] != c0
                or not np.array_equal(a["fill"], out)):
            continue
        if spec["mode"] == "horizontal" and _has_vertical_copy(inp, a):
            continue
        return {"input": inp, "output": out}
    raise ValueError("unable to construct an unambiguous periodic pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    varying_blank = bool(rng.random() < 0.4)
    fixed_blank = int(rng.integers(0, 10))
    train_specs, test_specs = _sample_specs(rng, nt, ns)
    if varying_blank:
        first_blanks = rng.choice(np.arange(10), size=2, replace=False)
        blanks = [int(first_blanks[0]), int(first_blanks[1])]
        blanks.extend(int(rng.integers(0, 10)) for _ in range(nt + ns - 2))
    else:
        blanks = [fixed_blank] * (nt + ns)
    child_seeds = rng.integers(0, 2**63, size=nt + ns, dtype=np.int64)
    forbidden = set()
    train, test = [], []
    for index, spec in enumerate(train_specs + test_specs):
        pair = _generate_pair(np.random.default_rng(int(child_seeds[index])),
                              spec, blanks[index], forbidden)
        forbidden.add(_grid_signature(pair["input"]))
        (train if index < nt else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    train_analyses, test_analyses = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert len(np.unique(inp)) == 3
            assert not _is_source_pair(inp, out)
            signature = _grid_signature(inp)
            pair_signature = _pair_signature(pair)
            assert signature not in inputs and pair_signature not in pairs
            inputs.add(signature)
            pairs.add(pair_signature)
            analyses = _candidate_analyses(inp)
            assert len(analyses) == 1
            a = analyses[0]
            assert out.shape == (a["size"], a["size"])
            assert np.array_equal(out, a["fill"])
            assert a["blank"] not in np.unique(out)
            assert a["period"] >= 2
            (train_analyses if split == "train" else test_analyses).append(a)
    # The learned rule is exhibited by a multi-cell mixed patch that can only
    # be reconstructed horizontally, so single-color and vertical-copy
    # shortcuts cannot account for the training set.
    assert any(a["size"] >= 2 and len(np.unique(pair["output"])) == 2
               and not _has_vertical_copy(pair["input"], a)
               and _has_horizontal_witness(pair["input"], a)
               for pair, a in zip(puzzle["train"], train_analyses))
    train_blanks = {a["blank"] for a in train_analyses}
    test_blanks = {a["blank"] for a in test_analyses}
    if len(train_blanks) == 1:
        assert test_blanks == train_blanks
    # With varying blank colors, two training examples establish that the
    # solid square's geometry, rather than a fixed literal color, marks it.
    if ns >= 2:
        assert len({a["size"] for a in test_analyses}) == ns
        assert len({_location(a["r0"], a["c0"], a["size"],
                              *pair["input"].shape)
                    for a, pair in zip(test_analyses, puzzle["test"])}) == ns
    assert max(a["size"] for a in test_analyses) <= 1 + max(
        a["size"] for a in train_analyses)
    assert len({a["size"] for a in train_analyses}) >= 2


def _puzzles_equal(left, right):
    return (all(len(left[k]) == len(right[k]) for k in ("train", "test"))
            and all(np.array_equal(a[t], b[t]) for k in ("train", "test")
                    for a, b in zip(left[k], right[k]) for t in ("input", "output")))


def _expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_pair_signature(p["train"][0]) for p in puzzles}) >= 35
    dimensions, locations, periods, sizes, palettes, layouts = (
        set(), set(), set(), set(), set(), set())
    horizontal_train = horizontal_test = mixed_train = rectangular = 0
    partial = uniform_train = single_train = combined = noncanonical = 0
    p2_s3 = p3_s1 = p3_s4 = long_large = 0
    multi_test = multi_test_diverse = 0
    variable_blank = fixed_blank = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        blanks = set()
        local_locations = {"train": set(), "test": set()}
        local_sizes = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _candidate_analyses(pair["input"])[0]
                h, w = pair["input"].shape
                loc = _location(a["r0"], a["c0"], a["size"], h, w)
                dimensions.add((h, w))
                locations.add(loc)
                local_locations[split].add(loc)
                local_sizes[split].add(a["size"])
                periods.add(a["period"])
                sizes.add(a["size"])
                palettes.add(tuple(int(c) for c in np.unique(pair["input"])))
                layouts.add((split, h, w, a["period"], a["size"],
                             a["r0"], a["c0"]))
                rectangular += h != w
                partial += h % a["period"] != 0 or w % a["period"] != 0
                blanks.add(a["blank"])
                no_vertical = not _has_vertical_copy(pair["input"], a)
                if no_vertical:
                    if split == "train":
                        horizontal_train += 1
                    else:
                        horizontal_test += 1
                mixed = len(np.unique(pair["output"])) == 2
                if split == "train" and len(np.unique(pair["output"])) == 2:
                    mixed_train += 1
                if split == "train":
                    uniform_train += a["size"] > 1 and not mixed
                    single_train += a["size"] == 1
                combined += no_vertical and h != w and mixed
                if not np.any(pair["input"][:a["size"], :a["size"]] == a["blank"]):
                    noncanonical += not np.array_equal(
                        pair["output"], pair["input"][:a["size"], :a["size"]])
                p2_s3 += a["period"] == 2 and a["size"] == 3
                p3_s1 += a["period"] == 3 and a["size"] == 1
                p3_s4 += a["period"] == 3 and a["size"] == 4
                long_large += a["period"] >= 4 and a["size"] >= 3
        assert len(local_locations["train"]) >= 2
        assert len(local_sizes["train"]) >= 2
        if len(puzzle["test"]) >= 2:
            multi_test += 1
            multi_test_diverse += (len(local_locations["test"]) == len(puzzle["test"])
                                   and len(local_sizes["test"]) == len(puzzle["test"]))
        variable_blank += len(blanks) > 1
        fixed_blank += len(blanks) == 1
    assert len(dimensions) >= 25 and len(locations) == 9
    assert periods == {2, 3, 4, 5} and sizes == {1, 2, 3, 4, 5}
    assert len(palettes) >= 35 and len(layouts) >= 100
    assert rectangular >= 40 and partial >= 40 and noncanonical >= 20
    assert horizontal_train >= 42 and horizontal_test >= 10
    assert mixed_train >= 42 and uniform_train >= 5 and single_train >= 15
    assert combined >= 20 and long_large >= 10
    assert p2_s3 >= 3 and p3_s1 >= 3 and p3_s4 >= 3
    assert variable_blank >= 5 and fixed_blank >= 5
    assert multi_test >= 10 and multi_test_diverse == multi_test
    a = generate_puzzle(1001, 6, 3)
    assert _puzzles_equal(a, generate_puzzle(1001, 6, 3))
    assert not _puzzles_equal(a, generate_puzzle(1002, 6, 3))
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(2000 + nt + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(3000 + (nt or 0) + (ns or 0), nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (3, 4), (True, 1)):
        _expect_value_error(lambda nt=nt, ns=ns: generate_puzzle(0, nt, ns))
