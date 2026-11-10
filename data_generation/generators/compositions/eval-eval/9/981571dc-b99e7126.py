"""Folded stencils: recover a reflected glyph and realize its ink at tile scale.

Only NumPy and the standard library are used. No parent code or examples are
loaded. All semantic roles except the fixed missing-value color 0 are visible.
"""
from collections import Counter

import numpy as np


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _key(array):
    a = np.asarray(array)
    return a.shape, a.astype(np.int8).tobytes()


def _normalized(array):
    labels = {}
    return array.shape, tuple(labels.setdefault(int(c), len(labels)) for c in array.flat)


def _flip(tile, upper, left):
    a = tile[::-1] if upper else tile
    return a[:, ::-1] if left else a


def _runs(indices):
    runs = []
    for index in map(int, indices):
        if runs and index == runs[-1][1] + 1:
            runs[-1] = runs[-1][0], index
        else:
            runs.append((index, index))
    return runs


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=True)


def _segment(grid):
    """The cross and border identify colors and fold positions structurally."""
    divider = int(grid[0, 0])
    axes = []
    for color in range(1, 10):
        if color == divider:
            continue
        rr = _runs(np.flatnonzero(np.all(grid == color, axis=1)))
        cc = _runs(np.flatnonzero(np.all(grid == color, axis=0)))
        if len(rr) == len(cc) == 1:
            axes.append((color, rr[0], cc[0]))
    _check(len(axes) == 1 and divider != 0, "one colored fold cross is required")
    axis, row_axis, col_axis = axes[0]
    separators = (grid == divider) | (grid == axis)
    rr = _runs(np.flatnonzero(np.all(separators, axis=1)))
    cc = _runs(np.flatnonzero(np.all(separators, axis=0)))
    _check(len(rr) >= 5 and len(cc) >= 5, "insufficient tile lattice")
    _check(rr[0][0] == cc[0][0] == 0 and
           rr[-1][1] == grid.shape[0] - 1 and
           cc[-1][1] == grid.shape[1] - 1, "lattice must have an outer border")
    thicknesses = {b - a + 1 for a, b in rr + cc}
    _check(len(thicknesses) == 1, "inconsistent divider thickness")
    t = thicknesses.pop()
    hs = {rr[i+1][0] - rr[i][1] - 1 for i in range(len(rr)-1)}
    ws = {cc[i+1][0] - cc[i][1] - 1 for i in range(len(cc)-1)}
    _check(len(hs) == len(ws) == 1, "unequal tile sizes")
    h, w = hs.pop(), ws.pop()
    _check(h in (2, 3) and w in (2, 3) and t in (1, 2), "unsupported tile geometry")
    _check(row_axis in rr[1:-1] and col_axis in cc[1:-1], "folds must be internal dividers")
    rows, cols = len(rr)-1, len(cc)-1
    k, l = rr.index(row_axis), cc.index(col_axis)
    _check(min(k, rows-k) >= h and min(l, cols-l) >= w,
           "the entire reflected stencil must fit")
    geometry = h, w, t, rows, cols, k, l
    expected = _frame(geometry, divider, axis)
    _check(np.array_equal(grid[separators], expected[separators]), "damaged divider")
    return geometry, divider, axis


def _slice(geometry, r, c):
    h, w, t = geometry[:3]
    y, x = t + r*(h+t), t + c*(w+t)
    return slice(y, y+h), slice(x, x+w)


def _distance(geometry, r, c):
    k, l = geometry[-2:]
    return (k-1-r if r < k else r-k,
            l-1-c if c < l else c-l)


def _locations(geometry, i, j):
    k, l = geometry[-2:]
    return ((k-1-i, l-1-j), (k-1-i, l+j),
            (k+i, l-1-j), (k+i, l+j))


def _frame(geometry, divider, axis):
    h, w, t, rows, cols, k, l = geometry
    grid = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    y, x = k*(h+t), l*(w+t)
    grid[y:y+t, :] = axis
    grid[:, x:x+t] = axis
    return grid


def _paint(geometry, divider, axis, ordinary, glyph, active_distances):
    grid = _frame(geometry, divider, axis)
    _, _, _, rows, cols, k, l = geometry
    for r in range(rows):
        for c in range(cols):
            tile = glyph if _distance(geometry, r, c) in active_distances else ordinary
            grid[_slice(geometry, r, c)] = _flip(tile, r < k, c < l)
    return grid


def _infer(value):
    grid = _grid(value)
    geometry, divider, axis = _segment(grid)
    h, w, t, rows, cols, k, l = geometry
    tiles = {}
    for r in range(rows):
        for c in range(cols):
            tile = grid[_slice(geometry, r, c)]
            _check(not np.any((tile == divider) | (tile == axis)), "divider color inside a tile")
            tiles[r, c] = _flip(tile, r < k, c < l).copy()
    frequencies = Counter(_key(a) for a in tiles.values())
    ordinary_key, ordinary_count = frequencies.most_common(1)[0]
    _check(ordinary_count > rows*cols/2, "ordinary pattern must have a strict majority")
    ordinary = next(a.copy() for a in tiles.values() if _key(a) == ordinary_key)
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _check(0 not in ordinary_colors and 2 <= len(ordinary_colors) <= 4,
           "ordinary tile needs two to four nonzero payload colors")
    supplied = {p: a for p, a in tiles.items() if _key(a) != ordinary_key}
    _check(len(supplied) >= 4, "too few supplied glyph fragments")
    glyph = np.zeros((h, w), dtype=np.int8)
    witnesses = np.zeros((h, w), dtype=int)
    for (r, c), a in supplied.items():
        i, j = _distance(geometry, r, c)
        _check(0 <= i < h and 0 <= j < w, "glyph outside the stencil window")
        _check(bool(np.any(a == 0)) and bool(np.any(a != 0)), "glyph fragments must be partial")
        known = a != 0
        _check(not np.any(known & (glyph != 0) & (glyph != a)), "contradictory reflected fragments")
        glyph[known] = a[known]
        witnesses += known
    _check(bool(np.all(glyph != 0)), "every canonical pixel needs a surviving witness")
    novel = set(map(int, np.unique(glyph))) - ordinary_colors
    _check(len(novel) == 1, "one ink color must be absent from the ordinary pattern")
    ink = novel.pop()
    mask = glyph == ink
    _check(3 <= int(mask.sum()) < h*w, "invalid stencil density")
    _check(len({_key(_flip(mask, u, v)) for u in (False, True) for v in (False, True)}) == 4,
           "stencil must distinguish both fold directions")
    _check(bool(np.any(glyph[~mask] != ordinary[~mask])), "whole-tile copying needs a background witness")
    active = set(map(tuple, np.argwhere(mask)))
    supplied_distances = {_distance(geometry, *p) for p in supplied}
    _check(supplied_distances < active, "supplied tiles must occupy a strict subset of active distances")
    expected = _paint(geometry, divider, axis, ordinary, glyph, active)
    return expected, {
        "geometry": geometry, "divider": divider, "axis": axis, "ink": ink,
        "ordinary": ordinary, "glyph": glyph, "mask": mask, "supplied": supplied,
        "active": active, "supplied_distances": supplied_distances,
        "missing_distances": active - supplied_distances, "witnesses": witnesses,
    }


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible geometry, reflected fragments, and their ink."""
    return _infer(input_grid)[0]


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return int(value)


def _geometry(rng, shape=None, fixed=None):
    if fixed is not None:
        return fixed
    h, w = shape if shape is not None else tuple(map(int, rng.integers(2, 4, size=2)))
    # A size-three tile cannot fit six tiles with thickness two within 30.
    t = int(rng.integers(1, 3)) if h == w == 2 else 1
    rows = int(rng.integers(2*h, (30-t)//(h+t)+1))
    cols = int(rng.integers(2*w, (30-t)//(w+t)+1))
    k = int(rng.integers(h, rows-h+1))
    l = int(rng.integers(w, cols-w+1))
    return h, w, t, rows, cols, k, l


def _design(rng, shape, forbidden=()):
    h, w = shape
    # A shared distribution in both splits, including disconnected designs.
    for _ in range(80):
        mask = np.zeros(shape, dtype=bool)
        n = int(rng.integers(3, h*w))
        mask.flat[rng.choice(h*w, n, replace=False)] = True
        if _key(mask) not in forbidden and len({
                _key(_flip(mask, u, v)) for u in (False, True) for v in (False, True)}) == 4:
            return mask
    # Finite constructive fallback, with every possible orientation available.
    for u, v in ((False, False), (False, True), (True, False), (True, True)):
        mask = np.zeros(shape, dtype=bool)
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = True
        mask = _flip(mask, u, v).copy()
        if _key(mask) not in forbidden:
            return mask
    raise RuntimeError("no permitted stencil design")


def _ordinary(rng, shape, colors, forbidden=None):
    n, count = int(np.prod(shape)), len(colors)
    for _ in range(80):
        labels = np.concatenate((np.arange(count), rng.integers(count, size=n-count)))
        rng.shuffle(labels)
        a = np.asarray(colors, dtype=np.int8)[labels.reshape(shape)]
        if _normalized(a) != forbidden:
            return a
    raise RuntimeError("ordinary motif sampling exhausted")


def _fragments(rng, glyph, positions, special, unique_pixel, survivor, sparse):
    """Build complementary damage; a corner pixel has exactly one witness."""
    n = glyph.size
    q = unique_pixel
    known = {p: np.zeros(glyph.shape, dtype=bool) for p in positions}
    if sparse:
        capacity = n//2
        known[survivor].flat[q] = True
        others = [p for p in special if p != survivor]
        rng.shuffle(others)
        order = others + [survivor]
        pixels = [i for i in range(n) if i != q]
        rng.shuffle(pixels)
        for index, pixel in enumerate(pixels):
            known[order[index % 4]].flat[pixel] = True
        for p in special:
            candidates = [i for i in range(n) if i != q and not known[p].flat[i]]
            rng.shuffle(candidates)
            extra = int(rng.integers(0, capacity-int(known[p].sum())+1))
            known[p].flat[candidates[:extra]] = True
        for p in positions:
            if p not in special:
                count = int(rng.integers(1, capacity+1))
                known[p].flat[rng.choice([i for i in range(n) if i != q], count, replace=False)] = True
    else:
        max_missing = n//2
        protected_missing = int(rng.choice([i for i in range(n) if i != q]))
        for p in positions:
            known[p][:] = True
            forced = protected_missing if p == survivor else q
            known[p].flat[forced] = False
            candidates = [i for i in range(n) if i not in (q, forced)]
            extra = int(rng.integers(0, max_missing))
            if extra:
                known[p].flat[rng.choice(candidates, extra, replace=False)] = False
        # Reveal any lost pixel in the special four-way orbit, retaining damage.
        for pixel in range(n):
            if not any(known[p].flat[pixel] for p in special):
                eligible = [p for p in special if p != survivor]
                known[eligible[int(rng.integers(len(eligible)))]].flat[pixel] = True
    return {p: np.where(a, glyph, 0).astype(np.int8) for p, a in known.items()}


def _make_pair(rng, geometry, palette, sparse, forbidden_mask=(), forbidden_motif=None):
    h, w, t, rows, cols, k, l = geometry
    divider, axis, ink, *colors = palette
    mask = _design(rng, (h, w), forbidden_mask)
    ordinary = _ordinary(rng, (h, w), colors, forbidden_motif)
    glyph = np.asarray(rng.choice(colors, size=(h, w)), dtype=np.int8)
    glyph[mask] = ink
    # Exact copying of backgrounds must change the output outside ink.
    p = tuple(np.argwhere(~mask)[int(rng.integers((~mask).sum()))])
    glyph[p] = int(rng.choice([c for c in colors if c != ordinary[p]]))
    active = list(map(tuple, np.argwhere(mask)))
    rng.shuffle(active)
    low = 1 if sparse else max(2, (len(active)+1)//2)
    high = max(low, (len(active)+1)//2) if sparse else len(active)-1
    visible_distances = active[:int(rng.integers(low, high+1))]
    special = _locations(geometry, *visible_distances[0])
    positions = list(special)
    limit = (rows*cols-1)//2
    for index, distance in enumerate(visible_distances[1:]):
        choices = list(_locations(geometry, *distance))
        rng.shuffle(choices)
        reserved = len(visible_distances)-index-2
        count = int(rng.integers(1, min(4, limit-len(positions)-reserved)+1))
        positions.extend(choices[:count])
    corners = (0, w-1, (h-1)*w, h*w-1)
    unique_pixel = int(rng.choice(corners))
    survivor = special[int(rng.integers(4))]
    fragments = _fragments(rng, glyph, positions, special, unique_pixel, survivor, sparse)
    inp = _paint(geometry, divider, axis, ordinary, glyph, set())
    for (r, c), fragment in fragments.items():
        inp[_slice(geometry, r, c)] = _flip(fragment, r < k, c < l)
    out = _paint(geometry, divider, axis, ordinary, glyph, set(active))
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    # Stability is a presentation preference, never a hidden solving parameter.
    stable_geometry = bool(rng.random() < .25)
    stable_palette = bool(rng.random() < .25)
    base_geometry = _geometry(rng)
    base_palette = tuple(map(int, rng.choice(np.arange(1, 10), 5, replace=False)))
    puzzle, seen, train_info = {"train": [], "test": []}, set(), []
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            for _ in range(80):
                geometry = _geometry(rng, fixed=base_geometry if stable_geometry else None)
                palette = base_palette if stable_palette else tuple(map(int, rng.choice(
                    np.arange(1, 10), 3 + int(rng.integers(2, min(4, geometry[0]*geometry[1])+1)),
                    replace=False)))
                forbidden_mask = {_key(train_info[0]["mask"])} if split == "train" and index == 1 else set()
                forbidden_motif = _normalized(train_info[0]["ordinary"]) if split == "train" and index == 1 else None
                # An all-distinct tile has only one color-normalized motif.
                # Reduce its payload count if it would repeat train[0].
                if (len(palette)-3 == geometry[0]*geometry[1] and
                        forbidden_motif == (geometry[:2], tuple(range(geometry[0]*geometry[1])))):
                    palette = palette[:-1]
                pair = _make_pair(rng, geometry, palette, split == "test", forbidden_mask, forbidden_motif)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                train_info.append(_infer(pair["input"])[1])
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation(info, kind):
    """Deliberately simpler rules, used to check demonstration witnesses."""
    g = info["geometry"]
    if kind == "no_stencil":
        return _paint(g, info["divider"], info["axis"], info["ordinary"],
                      info["glyph"], info["supplied_distances"])
    result = _paint(g, info["divider"], info["axis"], info["ordinary"], info["glyph"], info["active"])
    for i, j in info["active"]:
        for r, c in _locations(g, i, j):
            upper, left = r < g[-2], c < g[-1]
            if kind == "no_folds":
                tile = info["glyph"]
            elif kind == "row_only":
                tile = _flip(info["glyph"], upper, False)
            elif kind == "col_only":
                tile = _flip(info["glyph"], False, left)
            elif kind == "ink_only":
                tile = _flip(np.where(info["mask"], info["ink"], info["ordinary"]), upper, left)
            else:
                raise ValueError("unknown ablation")
            result[_slice(g, r, c)] = tile
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, f = _infer(inp)
            assert np.array_equal(expected, out), "incorrect folded stencil output"
            assert not np.any(out == 0) and np.any(inp == 0)
            g = f["geometry"]
            h, w, t, rows, cols, k, l = g
            # All surviving glyph cells are immutable; background tiles change
            # only when promoted to active stencil locations.
            for (r, c), fragment in f["supplied"].items():
                actual = _flip(out[_slice(g, r, c)], r < k, c < l)
                assert np.array_equal(actual[fragment != 0], fragment[fragment != 0])
            # Check actual full-field reflection, including one-fold fringes.
            for dimension, fold in ((0, 2*k*(h+t)+t-1), (1, 2*l*(w+t)+t-1)):
                a = out if dimension == 0 else out.T
                for x in range(a.shape[0]):
                    mate = fold-x
                    if 0 <= mate < a.shape[0]:
                        assert np.array_equal(a[x], a[mate]), "broken reflection symmetry"
            four_way = False
            for distance in f["supplied_distances"]:
                locations = _locations(g, *distance)
                if all(p in f["supplied"] for p in locations):
                    witness_counts = sum((f["supplied"][p] != 0).astype(int) for p in locations)
                    four_way |= bool(np.any(witness_counts == 1))
            assert four_way, "a four-way orbit must demonstrate complementary damage"
            assert np.any(f["witnesses"] == 1), "single-survivor evidence is required"
            for kind in ("no_stencil", "no_folds", "row_only", "col_only", "ink_only"):
                assert not np.array_equal(out, _ablation(f, kind)), f"missing witness for {kind}"
            sizes = [int(np.count_nonzero(a)) for a in f["supplied"].values()]
            if split == "train":
                assert min(sizes) >= (h*w+1)//2
                training.append(f)
            else:
                assert max(sizes) <= h*w//2
    assert len({_key(f["mask"]) for f in training}) >= 2
    assert len({_normalized(f["ordinary"]) for f in training}) >= 2


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed study plus count, preservation, and ambiguity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    records = [_infer(pair["input"])[1] for p in puzzles
               for s in ("train", "test") for pair in p[s]]
    assert len({_key(f["mask"]) for f in records}) >= 30
    assert len({_normalized(f["ordinary"]) for f in records}) >= 60
    assert len({f["geometry"] for f in records}) >= 70
    assert {f["geometry"][:2] for f in records} == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert {f["geometry"][2] for f in records} == {1, 2}
    assert len({(f["divider"], f["axis"], f["ink"]) for f in records}) >= 60
    assert len({len(f["active"]) for f in records}) >= 5
    assert len({len(f["supplied"]) for f in records}) >= 7
    assert len({len(f["missing_distances"]) for f in records}) >= 3
    offsets = {(np.sign(2*f["geometry"][-2]-f["geometry"][3]),
                np.sign(2*f["geometry"][-1]-f["geometry"][4])) for f in records}
    assert len(offsets) >= 7
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a damaged divider")
    # Erase the last canonical witness, then introduce a conflicting witness.
    inp = puzzles[0]["train"][0]["input"]
    _, f = _infer(inp)
    pixel = tuple(np.argwhere(f["witnesses"] == 1)[0])
    g = f["geometry"]
    missing = inp.copy()
    for (r, c), a in f["supplied"].items():
        fragment = a.copy()
        fragment[pixel] = 0
        missing[_slice(g, r, c)] = _flip(fragment, r < g[-2], c < g[-1])
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("unrecoverable glyph was accepted")
    conflict = inp.copy()
    for (r, c), a in f["supplied"].items():
        if a[pixel] == 0:
            fragment = a.copy()
            choices = [x for x in np.unique(f["glyph"]) if x != f["glyph"][pixel]]
            fragment[pixel] = choices[0]
            conflict[_slice(g, r, c)] = _flip(fragment, r < g[-2], c < g[-1])
            break
    try:
        solve(conflict)
    except ValueError:
        pass
    else:
        raise AssertionError("contradictory glyph was accepted")
