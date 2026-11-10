"""Nested phase gates: containment advances an edge-seeded three-beat clock.

Only standard-library modules and NumPy are used. No parent code or data is loaded.
"""

from collections import deque
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_TEACHING = ("clearing", "seed_overwrite", "highlight_overwrite", "remote_highlight",
             "remote_preserved", "background_preserved", "topology_matters",
             "nesting_matters", "strip_inside", "remote_phase_matters", "chamber_contrast",
             "first_advance_highlight", "second_advance_highlight")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _NEIGHBORS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _enclosed(wall):
    """Actual bounded complementary space, ignoring every other component."""
    blocked = np.pad(wall, 1)
    outside = np.zeros_like(blocked)
    outside[0, 0] = True
    queue = deque([(0, 0)])
    h, w = blocked.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in _NEIGHBORS:
            yy, xx = y + dy, x + dx
            if (0 <= yy < h and 0 <= xx < w and not blocked[yy, xx]
                    and not outside[yy, xx]):
                outside[yy, xx] = True
                queue.append((yy, xx))
    return (~blocked & ~outside)[1:-1, 1:-1]


def _drawing_info(mask):
    objects = []
    labels = np.zeros(mask.shape, dtype=int)
    for i, cells in enumerate(_components(mask)):
        wall = np.zeros_like(mask)
        wall[tuple(cells.T)] = True
        inside = _enclosed(wall)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        if inside.any():
            if len(_components(inside)) != 1:
                raise ValueError("a contour must enclose exactly one region")
        elif len(cells) < .35 * np.prod(hi - lo + 1):
            raise ValueError("a terminal blob must be compact")
        objects.append(dict(cells=cells, wall=wall, inside=inside, lo=lo, hi=hi))
        labels[tuple(cells.T)] = i + 1
    if not any(o["inside"].any() for o in objects):
        raise ValueError("at least one contour is required")
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        if np.any((a > 0) & (b > 0) & (a != b)):
            raise ValueError("drawing components may not touch")
    depth = np.zeros(mask.shape, dtype=int)
    contours = [o for o in objects if o["inside"].any()]
    for outer in contours:
        descendants = 0
        for inner in objects:
            if inner is outer:
                continue
            membership = outer["inside"][tuple(inner["cells"].T)]
            if membership.any() and not membership.all():
                raise ValueError("components may not cross a contour")
            descendants += bool(membership.all())
        if not descendants:
            raise ValueError("empty contours are outside this family")
        depth += outer["inside"]
    return dict(objects=objects, contours=contours, depth=depth)


def _analyze(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or counts.max() <= grid.size / 2:
        raise ValueError("exactly four roles and a strict-majority background are required")
    background = int(colors[np.argmax(counts)])
    candidates = []
    for side in range(4):
        g = np.rot90(grid, side)
        for seed in colors:
            seed = int(seed)
            if seed == background:
                continue
            cells = np.argwhere(g == seed)
            if len(cells) < 2 or np.any(cells[:, 0] != 0):
                continue
            start, end = int(cells[:, 1].min()), int(cells[:, 1].max()) + 1
            if end - start != len(cells):
                continue
            for cap in (start - 1, end):
                if min(start, cap) < 1 or max(end - 1, cap) >= g.shape[1] - 1:
                    continue
                highlight = int(g[0, cap])
                if highlight in (background, seed) or np.count_nonzero(g == highlight) != 1:
                    continue
                drawing = next(int(c) for c in colors if c not in (background, seed, highlight))
                candidates.append(dict(grid=g, background=background, drawing=drawing,
                                       seed=seed, highlight=highlight, side=side,
                                       start=start, end=end, cap=cap))
    if len(candidates) != 1:
        raise ValueError("the edge bar and its one-cell color cap must be unique")
    info = candidates[0]
    info.update(_drawing_info(info["grid"] == info["drawing"]))
    return info


def _phase_arrays(info, depth=None):
    g = info["grid"]
    if depth is None:
        depth = info["depth"]
    phase = (np.arange(g.shape[0])[:, None] + 2 * depth) % 6
    strip = np.zeros(g.shape, dtype=bool)
    strip[:, info["start"]:info["end"]] = True
    return phase, strip


def _paint(info, depth=None, remote=True):
    """Row-wise painter used by the public input-only solver."""
    g = info["grid"]
    phase, strip = _phase_arrays(info, depth)
    out = g.copy()
    out[strip] = info["background"]
    for r in range(len(g)):
        tick = (phase[r] % 2 == 0) & strip[r]
        out[r, tick] = info["seed"]
        out[r, strip[r] & (phase[r] == 4)] = info["highlight"]
        if remote:
            out[r, ~strip[r] & (g[r] == info["drawing"]) & (phase[r] == 4)] = info["highlight"]
    return np.rot90(out, -info["side"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, geometry and nesting from this input alone."""
    return _paint(_analyze(input_grid))


def _formula(info):
    """Vectorized cell-case expression for validation, separate from painting."""
    g = info["grid"]
    p, strip = _phase_arrays(info)
    corridor = np.where(p % 2, info["background"],
                        np.where(p == 4, info["highlight"], info["seed"]))
    elsewhere = np.where((g == info["drawing"]) & (p == 4), info["highlight"], g)
    return np.rot90(np.where(strip, corridor, elsewhere).astype(np.int8), -info["side"]).copy()


def _features(info):
    g = info["grid"]
    phase, strip = _phase_arrays(info)
    ink, high = g == info["drawing"], phase == 4
    actual = _paint(info)
    depth = info["depth"]
    flat_high = np.broadcast_to(np.arange(len(g))[:, None] % 6 == 4, g.shape)
    return dict(
        clearing=bool(np.any(ink & strip & (phase % 2 == 1))),
        seed_overwrite=bool(np.any(ink & strip & (phase % 2 == 0) & ~high)),
        highlight_overwrite=bool(np.any(ink & strip & high)),
        remote_highlight=bool(np.any(ink & ~strip & high)),
        remote_preserved=bool(np.any(ink & ~strip & ~high)),
        background_preserved=bool(np.any((g == info["background"]) & ~strip & high)),
        topology_matters=not np.array_equal(actual, _paint(info, np.zeros_like(depth))),
        nesting_matters=not np.array_equal(actual, _paint(info, np.minimum(depth, 1))),
        wrap_matters=not np.array_equal(actual, _paint(info, np.minimum(depth, 2))),
        strip_inside=bool(np.any(strip & (depth > 0))),
        remote_phase_matters=bool(np.any(ink & ~strip & (high != flat_high))),
        chamber_contrast=bool(any(np.any(ink[r] & ~strip[r] & high[r])
                                 and np.any(ink[r] & ~strip[r] & ~high[r])
                                 for r in range(len(g)))),
        first_advance_highlight=bool(np.any((depth == 1) & high & (strip | ink))),
        second_advance_highlight=bool(np.any((depth == 2) & high & (strip | ink))),
        exterior_cycle=bool(len(g) >= 13 and any(np.all(depth[:13, c] == 0)
                            for c in range(info["start"], info["end"]))),
        depth=int(depth.max()), contours=len(info["contours"]),
        blobs=len(info["objects"]) - len(info["contours"]))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _blob(rng, compact):
    cap = 2 if compact else 4
    h, w = (int(rng.integers(1, cap + 1)) for _ in range(2))
    a = np.ones((h, w), dtype=bool)
    if h > 1 and w > 1 and rng.random() < .65:
        # Corner trimming preserves connected, filled, hole-free clusters.
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rng.shuffle(corners)
        for r, c in corners[:int(rng.integers(1, min(3, h * w - 1)))]:
            trial = a.copy()
            trial[r, c] = False
            if len(_components(trial)) == 1 and not _enclosed(trial).any():
                a = trial
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _join(rng, parts, compact):
    if len(parts) == 1:
        return parts[0]
    # Recursive grouping allows mixed-axis spatial arrangements without extra loops.
    if len(parts) >= 3 and rng.random() < .6:
        cut = int(rng.integers(1, len(parts)))
        parts = [_join(rng, parts[:cut], compact), _join(rng, parts[cut:], compact)]
    axis = int(rng.integers(2))
    cross = 1 - axis
    gaps = [int(rng.integers(1, 2 if compact else 5)) for _ in parts[:-1]]
    extent = max(p.shape[cross] for p in parts) + (0 if compact else int(rng.integers(3)))
    shape = [0, 0]
    shape[axis] = sum(p.shape[axis] for p in parts) + sum(gaps)
    shape[cross] = extent
    out = np.zeros(shape, dtype=bool)
    pos = 0
    for i, p in enumerate(parts):
        start = [0, 0]
        start[axis] = pos
        start[cross] = int(rng.integers(extent - p.shape[cross] + 1))
        r, c = start
        out[r:r + p.shape[0], c:c + p.shape[1]] = p
        pos += p.shape[axis] + (gaps[i] if i < len(gaps) else 0)
    return out


def _side_profile(rng, total, begin, end, cap):
    values = np.zeros(total, dtype=int)
    n = end - begin
    if cap == 0 or rng.random() < .24:
        return values
    style = int(rng.integers(3))
    if style == 0:
        # An inward bay, not just a rectangular bounding-box perimeter.
        a = int(rng.integers(n))
        b = int(rng.integers(a + 1, n + 1))
        values[begin + a:begin + b] = int(rng.integers(1, cap + 1))
    elif style == 1:
        values[begin:end] = np.rint(np.linspace(0, cap, n)).astype(int)
    else:
        pos = begin
        while pos < end:
            stop = min(end, pos + int(rng.integers(1, 4)))
            values[pos:stop] = int(rng.integers(cap + 1))
            pos = stop
    return values


def _surround(rng, content, compact):
    pads = ([2 + int(rng.random() < .25) for _ in range(4)] if compact
            else [int(rng.integers(2, 5)) for _ in range(4)])
    top, bottom, left, right = pads
    ch, cw = content.shape
    h, w = ch + top + bottom, cw + left + right
    if max(h, w) > 28:
        raise ValueError("scene does not fit")
    t = _side_profile(rng, w, left, w - right, top - 2)
    b = h - 1 - _side_profile(rng, w, left, w - right, bottom - 2)
    l = _side_profile(rng, h, top, h - bottom, left - 2)
    r = w - 1 - _side_profile(rng, h, top, h - bottom, right - 2)
    path = ([(int(t[x]), x) for x in range(w)]
            + [(y, int(r[y])) for y in range(h)]
            + [(int(b[x]), x) for x in range(w - 1, -1, -1)]
            + [(y, int(l[y])) for y in range(h - 1, -1, -1)])
    wall = np.zeros((h, w), dtype=bool)
    for (y, x), (yy, xx) in zip(path, path[1:] + path[:1]):
        wall[min(y, yy):max(y, yy) + 1, x] = True
        wall[yy, min(x, xx):max(x, xx) + 1] = True
    inside = _enclosed(wall)
    if len(_components(wall)) != 1 or len(_components(inside)) != 1:
        raise ValueError("invalid stepped contour")
    if not np.all(inside[top:top + ch, left:left + cw][content]):
        raise ValueError("a child escaped its contour")
    out = wall.copy()
    out[top:top + ch, left:left + cw] |= content
    return out


def _scene(rng, target_depth, compact):
    def loop(level):
        parts = [loop(level + 1) if level < target_depth else _blob(rng, compact)]
        # Budget decreases at depth so rich outer scenes can still fit 30 cells.
        probability = .20 if compact else (.62 if level == 1 else .37)
        for _ in range(1 if compact else 3):
            if rng.random() < probability:
                parts.append(_blob(rng, compact))
        if level < target_depth and not compact and rng.random() < .12:
            parts.append(_surround(rng, _blob(rng, compact), compact))
        if target_depth == 1 and not compact and rng.random() < .13:
            parts.append(_surround(rng, _blob(rng, compact), compact))
        rng.shuffle(parts)
        return _surround(rng, _join(rng, parts, compact), compact)

    roots = [loop(1)]
    if rng.random() < (.25 if compact else .65):
        roots.append(_blob(rng, compact))
    if rng.random() < .16:
        roots.append(_surround(rng, _blob(rng, compact), compact))
    rng.shuffle(roots)
    return _join(rng, roots, compact)


def _candidate(rng, target_depth, side, palette, mode, compact):
    core = _scene(rng, target_depth, compact)
    ch, cw = core.shape
    extra_width = 6 if mode == "comparison" else 2
    low_h, low_w = max(14, ch + 2), max(14, cw + extra_width)
    if max(low_h, low_w) > 30:
        raise ValueError("scene does not fit the key canvas")
    h, w = int(rng.integers(low_h, 31)), int(rng.integers(low_w, 31))
    y, x = int(rng.integers(1, h - ch)), int(rng.integers(1, w - cw))
    mask = np.zeros((h, w), dtype=bool)
    mask[y:y + ch, x:x + cw] = core
    depth = _drawing_info(mask)["depth"]
    if mode != "comparison" and rng.random() < .22:
        length = int(rng.integers(max(2, w // 3), w - 2))
    else:
        length = int(rng.integers(2, min(8, w - 3)))
    choices = []
    for start in range(1, w - length):
        for cap in (start - 1, start + length):
            if not 1 <= cap < w - 1:
                continue
            strip_depth = depth[:, start:start + length]
            if mode == "comparison" and np.any(strip_depth):
                continue
            if mode in ("witness", "test") and not np.any(strip_depth == target_depth):
                continue
            choices.append((start, cap))
    if not choices:
        raise ValueError("no suitable bar span")
    start, cap = choices[int(rng.integers(len(choices)))]
    bg, ink, seed, highlight = palette
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[mask] = ink
    grid[0, start:start + length] = seed
    grid[0, cap] = highlight
    grid = np.rot90(grid, -side).copy()
    info = _analyze(grid)
    features = _features(info)
    if features["depth"] != target_depth:
        raise ValueError("unexpected maximum nesting")
    if mode == "witness" and not all(features[k] for k in _TEACHING):
        raise ValueError("nested witness lacks an interaction")
    if mode == "comparison" and not features["exterior_cycle"]:
        raise ValueError("comparison must display the exterior clock")
    if mode == "test" and not features["wrap_matters"]:
        raise ValueError("third level must affect the answer")
    return dict(input=grid, output=_formula(info))


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("illegal train/test counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.integers(2))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    base_side = int(rng.integers(4))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if not stable_palette:
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            mode = ("test" if split == "test" else
                    "witness" if i == 0 else "comparison" if i == 1 else "ordinary")
            target = 3 if split == "test" else 2 if i == 0 else 1 if i == 1 else int(rng.integers(1, 3))
            side = (base_side + i) % 4 if split == "train" and i < 2 else int(rng.integers(4))
            for attempt in range(240):
                try:
                    # Mix metric scales from the start: otherwise the first fit
                    # systematically favors a narrow class of rich large trees.
                    compact = attempt >= 36 or bool(rng.random() < .4)
                    pair = _candidate(rng, target, side, palette, mode, compact)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(out, _formula(info)), "wrong phase or overwrite pixels"
            assert np.array_equal(out, solve(inp)), "input-only solver disagrees"
            canonical_out = np.rot90(out, info["side"])
            assert canonical_out[0, info["cap"]] == info["highlight"]
            f = _features(info)
            if split == "train":
                assert 1 <= f["depth"] <= 2
                training.append(f)
            else:
                assert f["depth"] == 3 and f["wrap_matters"]
    assert any(all(f[k] for k in _TEACHING) for f in training), "missing nested teaching witness"
    assert any(f["depth"] == 1 and f["exterior_cycle"] for f in training), "missing shallow clock comparison"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _regression_grid():
    """Hand-made nested scene with a genuine inward bay and a visible key."""
    a = np.zeros((25, 27), dtype=np.int8)
    a[0, 10:13], a[0, 13] = 1, 3
    # Outer loop with a left bay: rows 8..12 start at column 6, not column 2.
    vertices = [(3, 2), (3, 23), (21, 23), (21, 2), (13, 2),
                (13, 6), (7, 6), (7, 2), (3, 2)]
    for (r, c), (rr, cc) in zip(vertices, vertices[1:]):
        a[min(r, rr):max(r, rr) + 1, c] = 2
        a[rr, min(c, cc):max(c, cc) + 1] = 2
    a[6, 9:20] = a[18, 9:20] = 2
    a[6:19, 9] = a[6:19, 19] = 2
    a[10:13, 14:16] = 2
    a[9:11, 3:5] = 2  # exterior blob, despite lying inside the outer bounding box
    return a


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, normalized, structures = [], set(), set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                f = _features(info)
                f.update(split=split, side=info["side"], shape=pair["input"].shape,
                         palette=tuple(info[k] for k in ("background", "drawing", "seed", "highlight")),
                         cap_end=info["cap"] < info["start"], bar=info["end"] - info["start"])
                f["nonrect"] = sum(not (o["wall"][o["lo"][0], o["lo"][1]:o["hi"][1] + 1].all()
                    and o["wall"][o["hi"][0], o["lo"][1]:o["hi"][1] + 1].all()
                    and o["wall"][o["lo"][0]:o["hi"][0] + 1, o["lo"][1]].all()
                    and o["wall"][o["lo"][0]:o["hi"][0] + 1, o["hi"][1]].all())
                    for o in info["contours"])
                records.append(f)
                normalized.add(_key(info["grid"] == info["drawing"]))
                structures.add((f["contours"], f["blobs"], f["depth"],
                                tuple(sorted(int(info["depth"][tuple(o["cells"][0])])
                                             for o in info["objects"]))))
        # Metamorphic tests use the input and output only.
        pair = puzzle["test"][0]
        inp, out = pair["input"], pair["output"]
        permutation = np.random.default_rng(seed + 900).permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
        assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(normalized) >= 180
    assert len({r["shape"] for r in records}) >= 45
    assert len(structures) >= 15
    assert sum(r["nonrect"] for r in records) >= 60
    assert {r["cap_end"] for r in records} == {False, True}
    assert len({r["bar"] for r in records}) >= 5
    assert all({r["palette"][i] for r in records} == set(range(10)) for i in range(4))
    for split in ("train", "test"):
        selected = [r for r in records if r["split"] == split]
        assert {r["side"] for r in selected} == set(range(4))
        assert len({r["contours"] for r in selected}) >= 2
        assert len({r["blobs"] for r in selected}) >= 3

    # Check every legal count combination, generate representative extremes.
    # The supplied host verifier separately generates every legal combination.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            _check_counts(nt, ns)
    for nt, ns in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(7600 + 10 * nt + ns, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")

    a = _regression_grid()
    info, b = _analyze(a), solve(a)
    # Walls belong to the parent; blobs never increment depth.
    assert info["depth"][6, 12] == 1
    assert info["depth"][10, 14] == 2
    assert info["depth"][10, 3] == 0
    assert b[10, 3] == 3 and b[10, 14] == 2
    assert b[8, 10] == 1  # depth-two corridor tick: (8+4) % 6 = 0
    assert b[8, 9] == 3
    assert b[7, 10] == 0  # odd corridor distance clears even inside two contours
    assert b[6, 10] == 1  # outer ancestor advances once on the inner contour's wall
    assert b[12, 10] == 3  # two enclosing contours advance twice
    assert not np.array_equal(b, _paint(info, np.zeros_like(info["depth"])))
    assert not np.array_equal(b, _paint(info, np.minimum(info["depth"], 1)))
    # Bounding-box approximation incorrectly declares the bay blob enclosed.
    boxes = np.zeros(a.shape, dtype=int)
    for o in info["contours"]:
        y, x = o["lo"]; yy, xx = o["hi"]
        boxes[y + 1:yy, x + 1:xx] += 1
    assert not np.array_equal(b, _paint(info, boxes))
    bay = a.copy()
    bay[0] = 0
    bay[0, 3:5], bay[0, 5] = 1, 3
    bay_info = _analyze(bay)
    assert solve(bay)[10, 3] == 3
    assert _paint(bay_info, boxes)[10, 3] == 1
    # The edge-origin clock does not restart at the near contour wall (row 3).
    assert b[4, 10] == 1  # (4+2) % 6 = 0; restarting at row 3 would make a gap
    # Deliberately wrong output must fail full-puzzle validation.
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output was accepted")
    return dict(seeds=42, pairs=len(records), distinct_drawing_structures=len(normalized),
                input_shapes=len({r["shape"] for r in records}), topological_signatures=len(structures),
                nonrectangular_contours=sum(r["nonrect"] for r in records),
                contours=sorted({r["contours"] for r in records}),
                blobs=sorted({r["blobs"] for r in records}),
                depths=sorted({r["depth"] for r in records}))


if __name__ == "__main__":
    print(self_test())
