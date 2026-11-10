"""Mirror-address bridges: reflection classes are addresses on a coarse lattice.

Only original clues are endpoints. Matching brackets activate four-point mirror
orbits, in the local cross color; unbracketed clues remain single. No parent code
is imported, and solve uses only the supplied grid.
"""

from itertools import combinations
import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is out of range")
    return int(value)


def _counts(num_train, num_test):
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _array(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _parse(input_grid):
    g = _array(input_grid)
    sep = int(g[0, 0])
    sr = np.flatnonzero(np.all(g == sep, axis=1)).tolist()
    sc = np.flatnonzero(np.all(g == sep, axis=0)).tolist()
    if (len(sr) < 2 or len(sc) < 2 or sr[0] != 0 or sc[0] != 0
            or sr[-1] != g.shape[0] - 1 or sc[-1] != g.shape[1] - 1):
        raise ValueError("missing separator frame")
    th, tw = sr[1] - 1, sc[1] - 1
    if (th < 3 or tw < 3 or th % 2 != 1 or tw % 2 != 1
            or any(b - a != th + 1 for a, b in zip(sr, sr[1:]))
            or any(b - a != tw + 1 for a, b in zip(sc, sc[1:]))):
        raise ValueError("tiles must be equal odd rectangles")
    rows, cols, h, w = len(sr) - 1, len(sc) - 1, th // 2, tw // 2
    tiles = np.stack([g[y + 1:y + th + 1, x + 1:x + tw + 1]
                      for y in sr[:-1] for x in sc[:-1]]).reshape(rows, cols, th, tw)
    axis = int(tiles[0, 0, h, w])
    if (axis == sep or np.any(tiles == sep)
            or not np.all(tiles[:, :, h, :] == axis)
            or not np.all(tiles[:, :, :, w] == axis)):
        raise ValueError("invalid local crosses")
    off = np.ones((th, tw), dtype=bool)
    off[h, :] = False
    off[:, w] = False
    values = tiles[:, :, off].ravel().astype(int)
    freq = np.bincount(values, minlength=10)
    bg = int(np.argmax(freq))
    if bg in (sep, axis) or freq[bg] * 2 <= len(values) or np.any(values == axis):
        raise ValueError("background must dominate off-axis cells")
    records, groups, occupied = [], {}, set()
    for tr in range(rows):
        for tc in range(cols):
            for lr, lc in zip(*np.nonzero(off & (tiles[tr, tc] != bg))):
                lr, lc = int(lr), int(lc)
                color = int(tiles[tr, tc, lr, lc])
                a, b = min(lr, 2 * h - lr), min(lc, 2 * w - lc)
                if (tr, tc, a, b) in occupied:
                    raise ValueError("at most one clue per mirror orbit per tile")
                occupied.add((tr, tc, a, b))
                records.append((tr, tc, lr, lc, color))
                groups.setdefault((color, a, b), set()).add((tr, tc))
    return dict(grid=g, sep=sep, bg=bg, axis=axis, rows=rows, cols=cols,
                h=h, w=w, th=th, tw=tw, records=records, groups=groups)


def _activation(points):
    """Inclusive row/column brackets, with no activation for an isolated point."""
    active = set()
    for r in {r for r, _ in points}:
        cs = [c for rr, c in points if rr == r]
        if len(cs) >= 2:
            active.update((r, c) for c in range(min(cs), max(cs) + 1))
    for c in {c for _, c in points}:
        rs = [r for r, cc in points if cc == c]
        if len(rs) >= 2:
            active.update((r, c) for r in range(min(rs), max(rs) + 1))
    return active


def _compress(info):
    rr = [1 + r * (info["th"] + 1) + info["h"] for r in range(info["rows"])]
    cc = [1 + c * (info["tw"] + 1) + info["w"] for c in range(info["cols"])]
    return np.delete(np.delete(info["grid"], rr, axis=0), cc, axis=1).copy()


def _paint(out, info, tile, address):
    tr, tc = tile
    a, b = address
    h, w = info["h"], info["w"]
    y, x = 1 + tr * (2 * h + 1), 1 + tc * (2 * w + 1)
    for lr in (a, 2 * h - 1 - a):
        for lc in (b, 2 * w - 1 - b):
            out[y + lr, x + lc] = info["axis"]


def _predict(info, mode="normal"):
    if mode == "quarterturn" and info["h"] != info["w"]:
        return _predict(info)
    out = _compress(info)
    if mode == "mirror_all":
        for tr, tc, lr, lc, _ in info["records"]:
            _paint(out, info, (tr, tc), (min(lr, 2 * info["h"] - lr),
                                       min(lc, 2 * info["w"] - lc)))
        return out
    groups, addresses = {}, {}
    for tr, tc, lr, lc, color in info["records"]:
        address = (min(lr, 2 * info["h"] - lr), min(lc, 2 * info["w"] - lc))
        if mode == "raw":
            key = (color, lr, lc)
        elif mode == "colorblind":
            key = address
        elif mode == "addressblind":
            key = color
        else:
            key = (color,) + address
        groups.setdefault(key, set()).add((tr, tc))
        addresses.setdefault(key, set()).add(address)
    for key, points in groups.items():
        active = _activation(points)
        if mode == "no_interpolation":
            active &= points
        elif mode == "recursive":
            while True:
                larger = _activation(points | active)
                if larger <= active:
                    break
                active |= larger
        for tile in active:
            for address in addresses[key]:
                if mode == "quarterturn":
                    a, b = address
                    n = 2 * info["h"]
                    y, x = 1 + tile[0] * (n + 1), 1 + tile[1] * (n + 1)
                    for lr, lc in ((a, b), (b, n - 1 - a),
                                   (n - 1 - a, n - 1 - b), (n - 1 - b, a)):
                        out[y + lr, x + lc] = info["axis"]
                else:
                    _paint(out, info, tile, address)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode geometry, group original clues by color and reflection class."""
    return _predict(_parse(input_grid))


def _blank(rows, cols, h, w, palette):
    bg, sep, axis = palette[:3]
    th, tw = 2 * h + 1, 2 * w + 1
    g = np.full((rows * (th + 1) + 1, cols * (tw + 1) + 1), sep, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            tile = np.full((th, tw), bg, dtype=np.int8)
            tile[h, :] = axis
            tile[:, w] = axis
            g[1 + r * (th + 1):1 + r * (th + 1) + th,
              1 + c * (tw + 1):1 + c * (tw + 1) + tw] = tile
    return g


def _render(rows, cols, h, w, palette, clues):
    """Build labels from construction records, by enumerating endpoint pairs.

    clues are (tile_row, tile_col, orbit_a, orbit_b, color, source_quadrant).
    This does not call the input parser or solve.
    """
    inp = _blank(rows, cols, h, w, palette)
    out = np.full((rows * (2 * h + 1) + 1, cols * (2 * w + 1) + 1), palette[1], dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            out[1 + r * (2 * h + 1):1 + r * (2 * h + 1) + 2 * h,
                1 + c * (2 * w + 1):1 + c * (2 * w + 1) + 2 * w] = palette[0]
    claimed = set()
    for r, c, a, b, color, q in clues:
        assert (r, c, a, b) not in claimed
        claimed.add((r, c, a, b))
        lr, lc = (a if q < 2 else 2 * h - a), (b if q % 2 == 0 else 2 * w - b)
        inp[1 + r * (2 * h + 2) + lr, 1 + c * (2 * w + 2) + lc] = color
        out[1 + r * (2 * h + 1) + lr - (lr > h),
            1 + c * (2 * w + 1) + lc - (lc > w)] = color
    activated = set()
    for one, two in combinations(clues, 2):
        r1, c1, a, b, color, _ = one
        r2, c2, aa, bb, other, _ = two
        if (color, a, b) != (other, aa, bb):
            continue
        if r1 == r2 and c1 != c2:
            activated.update((r1, c, a, b) for c in range(min(c1, c2), max(c1, c2) + 1))
        if c1 == c2 and r1 != r2:
            activated.update((r, c1, a, b) for r in range(min(r1, r2), max(r1, r2) + 1))
    for r, c, a, b in activated:
        for lr in (a, 2 * h - 1 - a):
            for lc in (b, 2 * w - 1 - b):
                out[1 + r * (2 * h + 1) + lr, 1 + c * (2 * w + 1) + lc] = palette[2]
    return {"input": inp, "output": out}


def _far(rng, n, interior=False):
    choices = [(a, b) for a in range(n) for b in range(a + 2, n)
               if not interior or a > 0 or b < n - 1]
    return choices[int(rng.integers(len(choices)))]


def _shape(rng, rows, cols, kind):
    if kind == "H":
        a, b = _far(rng, cols)
        r = int(rng.integers(rows))
        return {(r, a), (r, b)}
    if kind == "V":
        a, b = _far(rng, rows)
        c = int(rng.integers(cols))
        return {(a, c), (b, c)}
    if kind == "L":
        r0, r1 = sorted(int(x) for x in rng.choice(rows, 2, replace=False))
        c0, c1 = sorted(int(x) for x in rng.choice(cols, 2, replace=False))
        if r1 - r0 == c1 - c0 == 1:
            if rows >= 3:
                r0, r1 = _far(rng, rows)
            else:
                c0, c1 = _far(rng, cols)
        rr, cc = int(rng.choice((r0, r1))), int(rng.choice((c0, c1)))
        return {(rr, cc), (r0 + r1 - rr, cc), (rr, c0 + c1 - cc)}
    r = int(rng.integers(1, rows - 1))
    c = int(rng.integers(1, cols - 1))
    points = {(r, 0), (r, cols - 1), (0, c), (rows - 1, c)}
    if kind == "T":
        points.remove((0, c) if int(rng.integers(2)) else (rows - 1, c))
        points.add((r, c))
    elif kind == "plus" and rng.random() < .5:
        points.add((r, c))
    if not (_activation(points) - points):
        return _shape(rng, rows, cols, "H")
    return points


def _attach(rng, points, address, color, quadrants, force_quadrant=None):
    points = sorted(points)
    qs = [int(rng.choice(quadrants)) for _ in points]
    if len(points) >= 2 and len(quadrants) >= 2:
        # Different source quadrants are common, rather than a rare accident.
        shift = int(rng.integers(len(quadrants)))
        qs = [quadrants[(i + shift) % len(quadrants)] for i in range(len(points))]
    if force_quadrant is not None:
        qs[0] = force_quadrant
    return [(r, c, *address, color, q) for (r, c), q in zip(points, qs)]


def _teaching_pair(rng, which, palette, quadrants):
    h = w = 2
    non_diag = [(0, 1), (1, 0)]
    a = non_diag[int(rng.integers(2))]
    rest = [x for x in ((0, 0), (0, 1), (1, 0), (1, 1)) if x != a]
    rng.shuffle(rest)
    if which == 0:
        rows = cols = 4
        middle = int(rng.integers(1, 3))
        points = {(0, 0), (0, 3), (3, middle), (3, 3)}
        # Symmetries of an open frame provide nonrecursive-closure evidence.
        if int(rng.integers(2)):
            points = {(c, r) for r, c in points}
        if int(rng.integers(2)):
            points = {(3 - r, c) for r, c in points}
        if int(rng.integers(2)):
            points = {(r, 3 - c) for r, c in points}
        clues = _attach(rng, points, a, palette[3], quadrants)
        clues += _attach(rng, {(int(rng.integers(4)), int(rng.integers(4)))},
                         rest[0], palette[3], quadrants)
    else:
        rows, cols = 3, 4
        r = int(rng.integers(rows))
        c0, c1 = _far(rng, cols, interior=True)
        clues = _attach(rng, {(r, c0), (r, c1)}, a, palette[3], quadrants)
        other_row = (r + int(rng.integers(1, rows))) % rows
        q = int(rng.choice(quadrants))
        # Same address/different color and same color/different address decoys.
        clues += [(other_row, 0, *rest[0], palette[3], q),
                  (other_row, 2, *rest[0], palette[4], q),
                  (other_row, 2, *rest[1], palette[3], q)]
        if int(rng.integers(2)):
            rows, cols = cols, rows
            clues = [(c, r, b, a, color, (q % 2) * 2 + q // 2)
                     for r, c, a, b, color, q in clues]
            # A geometric transpose must respect the held-out source quadrant.
            # Reassigning source quadrants leaves every mirror address intact.
            clues = [(r, c, a, b, color, int(rng.choice(quadrants)))
                     for r, c, a, b, color, _ in clues]
    return _render(rows, cols, h, w, palette, clues)


_HALF_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2),
                (3, 3), (1, 3), (3, 1), (2, 4), (4, 2))


def _random_pair(rng, palette, quadrants, held_out=None):
    h, w = _HALF_SHAPES[int(rng.integers(len(_HALF_SHAPES)))]
    max_r, max_c = 29 // (2 * h + 2), 29 // (2 * w + 2)
    layouts = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)
               if max(r, c) >= 3]
    rows, cols = layouts[int(rng.integers(len(layouts)))]
    kinds = (["H"] if cols >= 3 else []) + (["V"] if rows >= 3 else [])
    if min(rows, cols) >= 2:
        kinds += ["L"]
    if min(rows, cols) >= 3:
        kinds += ["T", "plus"]
    addresses = [(a, b) for a in range(h) for b in range(w)]
    rng.shuffle(addresses)
    n = int(rng.integers(1, min(4, len(addresses)) + 1))
    color_count = int(rng.integers(1, min(3, n) + 1))
    colors = list(palette[3:3 + color_count])
    clues, used = [], []
    for i in range(n):
        points = _shape(rng, rows, cols, str(rng.choice(kinds)))
        address = addresses[i]
        color = colors[int(rng.integers(len(colors)))]
        # Occasional coordinate reuse, only for disjoint footprints and a
        # different source color. Other crossings use distinct mirror orbits.
        if i and rng.random() < .2:
            candidates = [(addr, old_color) for addr, old_color, footprint in used
                          if color != old_color and not (_activation(points) & footprint)
                          and all(addr != aa or not (_activation(points) & ff)
                                  for aa, _, ff in used)]
            if candidates:
                address = candidates[int(rng.integers(len(candidates)))][0]
        clues += _attach(rng, points, address, color, quadrants,
                         force_quadrant=held_out if i == 0 else None)
        used.append((address, color, _activation(points)))
    free = [addr for addr in addresses if addr not in {a for a, _, _ in used}]
    if free and rng.random() < .6:
        address = free[int(rng.integers(len(free)))]
        r, c = int(rng.integers(rows)), int(rng.integers(cols))
        points = {(r, c)}
        if min(rows, cols) >= 2 and rng.random() < .4:
            points.add(((r + 1) % rows, (c + 1) % cols))
        clues += _attach(rng, points, address, colors[0], quadrants)
    return _render(rows, cols, h, w, palette, clues)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    held_out = int(rng.integers(4))
    train_quadrants = [q for q in range(4) if q != held_out]
    base = tuple(int(x) for x in rng.permutation(10))
    # Both demonstrations use local color roles; a former clue color becomes
    # the cross color in the second demonstration.
    second = (base[0], base[1], base[3], base[2], *base[4:])
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            for _ in range(100):
                if split == "train" and i < 2:
                    palette = base if i == 0 else second
                    pair = _teaching_pair(rng, i, palette, train_quadrants)
                else:
                    palette = (base if rng.random() < .2 else
                               tuple(int(x) for x in rng.permutation(10)))
                    pair = _random_pair(rng, palette,
                                        train_quadrants if split == "train" else list(range(4)),
                                        held_out=held_out if split == "test" else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not sample a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    """Audit every output pixel using original endpoint brackets, not solve."""
    info = _parse(pair["input"])
    out = _array(pair["output"])
    rows, cols, h, w = (info[k] for k in ("rows", "cols", "h", "w"))
    assert out.shape == (rows * (2 * h + 1) + 1, cols * (2 * w + 1) + 1)
    assert np.all(out[::2 * h + 1, :] == info["sep"])
    assert np.all(out[:, ::2 * w + 1] == info["sep"])
    old = {(r, c, lr - (lr > h), lc - (lc > w)): color
           for r, c, lr, lc, color in info["records"]}
    horizontal = vertical = False
    bridged, active_orbits, orphans = set(), set(), 0
    for r in range(rows):
        for c in range(cols):
            for a in range(h):
                for b in range(w):
                    active = False
                    for (color, aa, bb), points in info["groups"].items():
                        if (a, b) != (aa, bb):
                            continue
                        cs = [cc for rr, cc in points if rr == r]
                        rs = [rr for rr, cc in points if cc == c]
                        hh = len(cs) >= 2 and min(cs) <= c <= max(cs)
                        vv = len(rs) >= 2 and min(rs) <= r <= max(rs)
                        horizontal |= hh
                        vertical |= vv
                        active |= hh or vv
                    if active:
                        active_orbits.add((r, c, a, b))
                        if not any((r, c, lr, lc) in old
                                   for lr in (a, 2 * h - 1 - a)
                                   for lc in (b, 2 * w - 1 - b)):
                            bridged.add((r, c, a, b))
                    for lr in (a, 2 * h - 1 - a):
                        for lc in (b, 2 * w - 1 - b):
                            expected = info["axis"] if active else old.get((r, c, lr, lc), info["bg"])
                            assert int(out[1 + r * (2 * h + 1) + lr,
                                           1 + c * (2 * w + 1) + lc]) == expected
    for r, c, lr, lc, _ in info["records"]:
        a, b = min(lr, 2 * h - lr), min(lc, 2 * w - lc)
        orphans += (r, c, a, b) not in active_orbits
    assert bridged, "every generated pair must have a genuinely empty bridge orbit"
    info.update(horizontal=horizontal, vertical=vertical, bridged=len(bridged),
                orphans=orphans, active_orbits=active_orbits)
    return info


def _quadrants(info):
    return {(2 if lr > info["h"] else 0) + (1 if lc > info["w"] else 0)
            for _, _, lr, lc, _ in info["records"]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    rejected = {name: False for name in ("raw", "mirror_all", "no_interpolation",
                                       "colorblind", "addressblind", "recursive", "quarterturn")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            f = _audit_pair(pair)
            assert np.array_equal(solve(inp), out)
            features[split].append(f)
            if split == "train":
                for name in rejected:
                    rejected[name] |= not np.array_equal(_predict(f, name), out)
    train = features["train"]
    assert any(f["horizontal"] for f in train) and any(f["vertical"] for f in train)
    assert any(f["orphans"] for f in train)
    assert all(rejected.values()), rejected
    tq = set().union(*(_quadrants(f) for f in train))
    assert len(tq) == 3, "training must expose three source quadrants"
    for f in features["test"]:
        assert _quadrants(f) - tq, "test must use the held-out source quadrant"
        assert any((r, c, min(lr, 2 * f["h"] - lr), min(lc, 2 * f["w"] - lc))
                   in f["active_orbits"]
                   for r, c, lr, lc, _ in f["records"]
                   if ((2 if lr > f["h"] else 0) + (lc > f["w"])) not in tq)


def _signature(puzzle):
    return tuple(_key(pair[k]) for s in ("train", "test")
                 for pair in puzzle[s] for k in ("input", "output"))


def self_test():
    """Bounded 42-seed diversity suite, count contracts, and semantic ablations."""
    # A hand-written 1x3 lattice: differently reflected endpoints, one orphan.
    # Expected pixels are literal, independent of the label renderer.
    inp = np.array([[int(c) for c in row] for row in (
        "9999999999999999999", "9038009008009008009", "9008009408009008009",
        "9888889888889888889", "9008009008009008009", "9008009008009008309",
        "9999999999999999999")], dtype=int)
    expected = np.array([[int(c) for c in row] for row in (
        "9999999999999999", "9088090880908809", "9000094000900009",
        "9000090000900009", "9088090880908809", "9999999999999999")], dtype=int)
    untouched = inp.copy()
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(inp, untouched)
    for turn in range(4):
        for flip in (False, True):
            x, y = np.rot90(inp, turn), np.rot90(expected, turn)
            if flip:
                x, y = x[:, ::-1], y[:, ::-1]
            assert np.array_equal(solve(x), y)
    palette_map = np.array([7, 2, 1, 6, 0, 9, 4, 8, 3, 5], dtype=int)
    assert np.array_equal(solve(palette_map[inp]), palette_map[expected])
    first = generate_puzzle(0)
    assert _signature(first) == _signature(generate_puzzle(0))
    assert _signature(first) != _signature(generate_puzzle(1))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 100 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError((nt, ns))
    shapes, half_shapes, lattices, palettes, structures, signatures = (set() for _ in range(6))
    all_quadrants, spans, group_counts, held_quadrants = set(), set(), set(), set()
    rectangles = ribbons = reused = crosses = productive_orange = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        tq = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _parse(pair["input"])
                q = _quadrants(f)
                all_quadrants |= q
                if split == "train":
                    tq |= q
                else:
                    held_quadrants |= q - tq
                shapes.add(pair["input"].shape)
                half_shapes.add((f["h"], f["w"]))
                lattices.add((f["rows"], f["cols"]))
                palettes.add((f["bg"], f["sep"], f["axis"]))
                # Canonicalize the complete colored structure, not just density.
                mapping = {}
                structures.add((pair["input"].shape, tuple(mapping.setdefault(int(v), len(mapping))
                                                          for v in pair["input"].flat)))
                rectangles += f["h"] != f["w"]
                ribbons += min(f["rows"], f["cols"]) == 1
                group_counts.add(len(f["groups"]))
                by_address, tile_coverage = {}, {}
                for (color, a, b), points in f["groups"].items():
                    by_address.setdefault((a, b), set()).add(color)
                    productive_orange += color == 7 and bool(_activation(points) - points)
                    for p1, p2 in combinations(points, 2):
                        if p1[0] == p2[0] or p1[1] == p2[1]:
                            spans.add(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                    for tile in _activation(points):
                        tile_coverage.setdefault(tile, set()).add((a, b))
                reused += any(len(v) > 1 for v in by_address.values())
                crosses += any(len(v) > 1 for v in tile_coverage.values())
    assert len(signatures) == 42
    assert all_quadrants == held_quadrants == set(range(4))
    assert len(shapes) >= 25 and len(half_shapes) >= 9 and len(lattices) >= 15
    assert len(palettes) >= 50 and len(structures) >= 180
    assert len(spans) >= 4 and len(group_counts) >= 4
    assert min(rectangles, ribbons, reused, crosses, productive_orange) > 0
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "quadrant_shapes": len(half_shapes), "lattices": len(lattices),
            "role_palettes": len(palettes), "normalized_structures": len(structures),
            "span_lengths": sorted(spans), "group_counts": sorted(group_counts),
            "rectangular_tiles": rectangles, "ribbons": ribbons,
            "coordinate_reuse_pairs": reused, "crossing_pairs": crosses,
            "productive_orange_groups": productive_orange}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
