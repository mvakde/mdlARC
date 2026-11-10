"""Border-directed transparent stamping, followed by full-vector run collapse.

Only the standard library and NumPy are used. Nothing is written by this module.
The input-only solver has no dependency on the sampler or a demonstration bank.
"""
from itertools import product

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in product((-1, 0, 1), repeat=2):
            q = (r + dr, c + dc)
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return seen == cells


def _parse(a):
    """Find the divider and empty framed field, then read visible templates."""
    h, w = a.shape
    layouts = []
    for d in range(1, w - 4):
        if not np.all(a[:, d] == a[0, d]):
            continue
        interior = a[1:-1, d + 2:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, div, frame = int(interior[0, 0]), int(a[0, d]), int(a[0, d + 1])
        if len({bg, div, frame}) != 3:
            continue
        if not all(a[r, c] == frame for r, c in
                   ((0, d + 1), (0, w - 1), (h - 1, d + 1), (h - 1, w - 1))):
            continue
        stage = a[:, :d]
        if (not np.any(stage != bg) or np.any((stage == div) | (stage == frame))
                or not np.array_equal(a[0, d + 2:-1], a[-1, d + 2:-1])
                or not np.array_equal(a[1:-1, d + 1], a[1:-1, -1])):
            continue
        edge = np.concatenate((a[0, d + 2:-1], a[1:-1, d + 1]))
        if np.any((edge == bg) | (edge == div)):
            continue
        layouts.append((d, bg, div, frame))
    if len(layouts) != 1:
        raise ValueError("input must have one unambiguous staging strip and empty framed field")
    d, bg, div, frame = layouts[0]
    ts = []
    for color in sorted(set(map(int, a[:, :d].flat)) - {bg}):
        cells = np.argwhere(a[:, :d] == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        bh, bw = map(int, hi - lo + 1)
        if bh % 2 != 1 or bw % 2 != 1:
            raise ValueError("template bounding boxes must have odd side lengths")
        mask = a[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] == color
        if not _connected(mask):
            raise ValueError("one 8-connected template per color is required")
        rows = [r for r in range(1, h - 1) if a[r, d + 1] == color]
        cols = [c for c in range(d + 2, w - 1) if a[0, c] == color]
        offsets = cells - (lo + hi) // 2
        if rows and cols:
            for r, c in product(rows, cols):
                dest = offsets + (r, c)
                if not np.all((dest[:, 0] >= 1) & (dest[:, 0] < h - 1)
                              & (dest[:, 1] >= d + 2) & (dest[:, 1] < w - 1)):
                    raise ValueError("every occupied stamp cell must fit inside the frame")
        ts.append(dict(color=color, cells=cells, lo=tuple(map(int, lo)),
                       shape=(bh, bw), mask=mask, offsets=offsets, rows=rows, cols=cols))
    ts.sort(key=lambda t: t["lo"])
    if len({t["lo"] for t in ts}) != len(ts):
        raise ValueError("source reading order must be unambiguous")
    return dict(d=d, bg=bg, div=div, frame=frame, templates=ts)


def _stamp(a, p=None, variant="source"):
    """Operate in original pixel coordinates; variants are diagnostic mutations."""
    p = _parse(a) if p is None else p
    out = a.copy()
    active = [t for t in p["templates"] if t["rows"] and t["cols"]]
    for t in active:
        out[t["cells"][:, 0], t["cells"][:, 1]] = p["bg"]
    if variant == "reverse":
        active = active[::-1]
    elif variant == "destination":
        active = sorted(active, key=lambda t: (min(t["rows"]), min(t["cols"])))
    elif variant == "size":
        active = sorted(active, key=lambda t: t["mask"].size)
    claimed = np.zeros(a.shape, dtype=bool)
    for t in active:
        centers = (zip(t["rows"], t["cols"]) if variant == "zip"
                   else product(t["rows"], t["cols"]))
        if variant == "opaque":
            bh, bw = t["shape"]
            offsets = [(r - bh // 2, c - bw // 2) for r, c in np.ndindex(bh, bw)]
        else:
            offsets = t["offsets"]
        for r, c in centers:
            for dr, dc in offsets:
                y, x = r + int(dr), c + int(dc)
                if not claimed[y, x]:
                    occupied = t["mask"][int(dr) + t["shape"][0] // 2,
                                           int(dc) + t["shape"][1] // 2]
                    out[y, x] = t["color"] if occupied else p["bg"]
                    claimed[y, x] = True
    return out


def _run_starts(a):
    # Both equivalence partitions are measured on the same complete scene.
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return np.flatnonzero(rows), np.flatnonzero(cols)


def _collapse(a):
    rows, cols = _run_starts(a)
    return a[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Stamp first; collapse maximal consecutive identical FULL vectors second."""
    a = _grid(input_grid)
    return _collapse(_stamp(a))


def _mask(rng, h, w):
    m = np.zeros((h, w), dtype=bool)
    mode = int(rng.integers(7))
    if min(h, w) == 1 or mode == 0:
        m[:] = True
    elif mode == 1:
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
    elif mode == 2 and h == w:
        yy, xx = np.indices((h, w))
        m = abs(yy - h // 2) + abs(xx - w // 2) == h // 2
    elif mode == 3:
        for r in range(h):
            m[r, round(r * (w - 1) / (h - 1))] = True
        for c in range(w):
            m[round(c * (h - 1) / (w - 1)), c] = True
    elif mode == 4:
        m[h // 2, :] = True
        m[:, w // 2] = True
    else:
        # Join four independently chosen extremes, then add a random density.
        points = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(points)
        r, c = points[0]
        m[r, c] = True
        for y, x in points[1:]:
            while (r, c) != (y, x):
                r += int(np.sign(y - r))
                c += int(np.sign(x - c))
                m[r, c] = True
        m |= rng.random((h, w)) < float(rng.uniform(.15, .85))
        # Join every addition to the anchor in at most h*w*(h+w) steps.
        for r, c in np.argwhere(m):
            r, c = int(r), int(c)
            y, x = points[0]
            while (r, c) != (y, x):
                r += int(np.sign(y - r))
                c += int(np.sign(x - c))
                m[r, c] = True
    if rng.random() < .5:
        m = m[::-1]
    if rng.random() < .5:
        m = m[:, ::-1]
    return m.copy()


def _template(mask, state="A", counts=(1, 1)):
    return dict(mask=mask, state=state, counts=counts, rows=[], cols=[])


def _source_positions(rng, ts, h, sw):
    slack = h - sum(t["mask"].shape[0] for t in ts) - len(ts) + 1
    if slack < 0:
        return False
    gaps = rng.multinomial(slack, np.full(len(ts) + 1, 1 / (len(ts) + 1)))
    top = int(gaps[0])
    for i, t in enumerate(ts):
        t["top"] = top
        t["left"] = int(rng.integers(sw - t["mask"].shape[1] + 1))
        top += t["mask"].shape[0] + 1 + int(gaps[i + 1])
    return True


def _coordinates(rng, ts, h, fw):
    for axis, (length, name) in enumerate(((h, "rows"), (fw, "cols"))):
        used = {v for t in ts for v in t[name]}
        # Large masks get their tighter coordinate ranges first.
        for t in sorted(ts, key=lambda t: -t["mask"].shape[axis]):
            if t[name] or t["state"] not in ("A", "R" if axis == 0 else "C"):
                continue
            rad = t["mask"].shape[axis] // 2 if t["state"] == "A" else 0
            options = [v for v in range(1 + rad, length - 1 - rad) if v not in used]
            n = t["counts"][axis] if t["state"] == "A" else 1
            if len(options) < n:
                return False
            t[name] = sorted(map(int, rng.choice(options, n, replace=False)))
            used.update(t[name])
    return True


def _assemble(rng, ts, h, sw, fw, unused=False, unused_row=None):
    """Labels are built from construction records, not from the input parser.

    Ground truth paints masks in reverse order, then uses explicit vector tuples
    to contract runs. This separately checks the solver's occupancy/array method.
    """
    palette = list(map(int, rng.permutation(10)))
    bg, div, frame = palette[:3]
    a = np.full((h, sw + 1 + fw), bg, dtype=np.int8)
    a[:, sw] = div
    a[[0, -1], sw + 1:] = frame
    a[:, [sw + 1, -1]] = frame
    for t, color in zip(ts, palette[3:]):
        t["color"] = color
        bh, bw = t["mask"].shape
        view = a[t["top"]:t["top"] + bh, t["left"]:t["left"] + bw]
        view[t["mask"]] = color
        for r in t["rows"]:
            a[r, sw + 1] = a[r, -1] = color
        for c in t["cols"]:
            a[0, sw + 1 + c] = a[-1, sw + 1 + c] = color
    if unused:
        free_r = [r for r in range(1, h - 1) if a[r, sw + 1] == frame]
        free_c = [c for c in range(sw + 2, a.shape[1] - 1) if a[0, c] == frame]
        color = palette[3 + len(ts)]
        r = int(rng.choice(free_r)) if unused_row is None else int(unused_row)
        assert r in free_r
        c = int(rng.choice(free_c))
        a[r, sw + 1] = a[r, -1] = color
        a[0, c] = a[-1, c] = color
    stamped = a.copy()
    for t in ts:
        if t["state"] == "A":
            bh, bw = t["mask"].shape
            view = stamped[t["top"]:t["top"] + bh, t["left"]:t["left"] + bw]
            view[t["mask"]] = bg
    for t in reversed(ts):
        if t["state"] != "A":
            continue
        bh, bw = t["mask"].shape
        for r, c in product(t["rows"], t["cols"]):
            view = stamped[r - bh // 2:r + bh // 2 + 1,
                           sw + 1 + c - bw // 2:sw + 1 + c + bw // 2 + 1]
            view[t["mask"]] = t["color"]
    rows = [i for i in range(h) if i == 0 or tuple(stamped[i]) != tuple(stamped[i - 1])]
    cols = [j for j in range(a.shape[1]) if j == 0
            or tuple(stamped[:, j]) != tuple(stamped[:, j - 1])]
    return {"input": a, "output": stamped[np.ix_(rows, cols)].copy()}


def _priority_pair(rng):
    h, sw = int(rng.integers(13, 27)), int(rng.integers(5, 9))
    fw = int(rng.integers(11, 30 - sw))
    ring = np.ones((5, 5), dtype=bool)
    ring[2, 2] = False
    small = _mask(rng, 3, 3)
    # All corners ensure a later pixel is exposed through the earlier hole.
    small[[0, 0, 2, 2], [0, 2, 0, 2]] = True
    small[1, 1] = True
    ts = [_template(ring), _template(small)]
    _source_positions(rng, ts, h, sw)
    r, c = int(rng.integers(3, h - 3)), int(rng.integers(3, fw - 3))
    ts[0]["rows"], ts[0]["cols"] = [r], [c]
    ts[1]["rows"], ts[1]["cols"] = [r - 1], [c + int(rng.choice((-1, 1))) ]
    return _assemble(rng, ts, h, sw, fw)


def _cartesian_pair(rng):
    h, sw = int(rng.integers(18, 28)), int(rng.integers(5, 8))
    fw = int(rng.integers(17, 30 - sw))
    diamond = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    block = _template(np.ones((5, 5), dtype=bool), counts=(2, 2))
    hollow = _template(diamond)
    ts = [block, hollow] + [_template(np.ones((1, 3), dtype=bool), state=s) for s in "NRC"]
    rng.shuffle(ts)
    # The field's rows 2 and 3 will be equal after stamping the solid block.
    # A retained source on row 2, with row 3 empty, makes the FULL rows differ.
    # This rules out compressing from target-only signatures even at nt=2.
    inactive = [t for t in ts if t["state"] != "A"]
    anchor = inactive[int(rng.integers(len(inactive)))]
    rest = [t for t in ts if t is not anchor]
    assert _source_positions(rng, rest, h - 4, sw)
    for t in rest:
        t["top"] += 4
    anchor["top"], anchor["left"] = 2, int(rng.integers(sw - 2))
    ts.sort(key=lambda t: (t["top"], t["left"]))
    block["rows"], block["cols"] = [4, h - 5], [4, fw - 5]
    hollow["rows"], hollow["cols"] = [(h - 1) // 2], [(fw - 1) // 2]
    next(t for t in ts if t["state"] == "R")["rows"] = [h - 2]
    assert _coordinates(rng, ts, h, fw)
    return _assemble(rng, ts, h, sw, fw, unused=True, unused_row=h - 3)


def _general_pair(rng, shift=None):
    # Only this bounded loop resamples candidates; no files or histories accrue.
    sizes = ((1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (5, 5))
    for _ in range(96):
        count = int(rng.integers(1, 6 if shift is None else 7))
        ts = []
        for i in range(count):
            bh, bw = sizes[int(rng.integers(len(sizes)))]
            if shift == "size" and i == 0:
                bh, bw = ((7, 7), (7, 5), (5, 7), (7, 3), (3, 7))[int(rng.integers(5))]
            state = "A" if i == 0 else str(rng.choice(["A", "N", "R", "C"], p=[.6, .14, .13, .13]))
            counts = tuple(map(int, rng.integers(1, 3, size=2)))
            if shift == "multiplicity" and i == 0:
                axis = int(rng.integers(2))
                counts = (3, counts[1]) if axis == 0 else (counts[0], 3)
            ts.append(_template(_mask(rng, bh, bw), state, counts))
        rng.shuffle(ts)
        minimum_h = sum(t["mask"].shape[0] for t in ts) + len(ts) - 1
        if minimum_h > 30:
            continue
        h = int(rng.integers(max(10, minimum_h), 31))
        sw = int(rng.integers(max(t["mask"].shape[1] for t in ts), 9))
        minimum_fw = max(9, max(t["mask"].shape[1] for t in ts) + 6)
        if minimum_fw > 29 - sw:
            continue
        fw = int(rng.integers(minimum_fw, 30 - sw))
        _source_positions(rng, ts, h, sw)
        if not _coordinates(rng, ts, h, fw):
            continue
        pair = _assemble(rng, ts, h, sw, fw, unused=count < 6 and rng.random() < .3)
        a, b = pair["input"], pair["output"]
        if b.shape != a.shape and not np.array_equal(_collapse(a), b):
            return pair
    # A guaranteed bounded fallback retains the requested train/test shift.
    size = 7 if shift == "size" else 5
    t = _template(np.ones((size, size), dtype=bool), counts=(3, 1) if shift == "multiplicity" else (1, 1))
    ts = [t]
    _source_positions(rng, ts, 24, size)
    t["rows"] = [5, 12, 18] if shift == "multiplicity" else [12]
    t["cols"] = [9]
    return _assemble(rng, ts, 24, size, 19)


def _count(n, name, low, high):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(n)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(("size", "multiplicity")))
    puzzle = {"train": [_priority_pair(rng), _cartesian_pair(rng)], "test": []}
    puzzle["train"].extend(_general_pair(rng) for _ in range(nt - 2))
    puzzle["test"].extend(_general_pair(rng, shift) for _ in range(ne))
    # At most 13 preceding inputs: 20 distinct bijections guarantee uniqueness.
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        a, b = pair["input"], pair["output"]
        geometry = _parse(a)
        for k in range(20):
            palette = (np.arange(10, dtype=np.int8) + k % 10) % 10
            if k >= 10:
                bg, div = geometry["bg"], geometry["div"]
                palette[[bg, div]] = palette[[div, bg]]
            candidate = palette[a]
            if _key(candidate) not in seen:
                pair["input"], pair["output"] = candidate, palette[b]
                seen.add(_key(candidate))
                break
        else:
            raise AssertionError("distinct recolorings exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    p = _parse(a)
    s = _stamp(a, p)
    out = _collapse(s)
    rows, cols = _run_starts(s)
    ir, ic = _run_starts(a)
    field_rows = _run_starts(s[:, p["d"] + 1:])[0]
    active = [t for t in p["templates"] if t["rows"] and t["cols"]]
    return dict(
        max_size=max(max(t["shape"]) for t in active),
        max_mult=max(max(len(t["rows"]), len(t["cols"])) for t in active),
        inactive={"R" if t["rows"] else "C" if t["cols"] else "N"
                  for t in p["templates"] if not (t["rows"] and t["cols"])},
        nonbg_row_repeat=any(np.array_equal(s[r], s[r - 1])
                            and np.any(s[r, p["d"] + 2:-1] != p["bg"]) for r in range(1, len(s))),
        nonbg_col_repeat=any(np.array_equal(s[:, c], s[:, c - 1])
                            and np.any(s[1:-1, c] != p["bg"])
                            for c in range(p["d"] + 3, s.shape[1] - 1)),
        separated_rows=len({tuple(row) for row in out}) < len(out),
        separated_cols=len({tuple(col) for col in out.T}) < out.shape[1],
        pre_partition=not np.array_equal(s[np.ix_(ir, ic)], out),
        field_only_rows=not np.array_equal(_collapse(s[field_rows]), out),
        **{v: not np.array_equal(_collapse(_stamp(a, p, v)), out)
           for v in ("reverse", "destination", "size", "opaque", "zip")},
        row_runs=tuple(map(int, np.diff(np.r_[rows, s.shape[0]]))),
        col_runs=tuple(map(int, np.diff(np.r_[cols, s.shape[1]]))))


def validate_generated_puzzle(puzzle):
    """Validate structure, exact reconstruction, causal witnesses and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, fs = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            p = _parse(a)
            s = _stamp(a, p)
            rows, cols = _run_starts(s)
            assert np.array_equal(out, s[np.ix_(rows, cols)]), "incorrect composed output"
            assert not np.array_equal(_collapse(a), out), "parent A is answer-irrelevant"
            assert out.shape != s.shape, "parent B is answer-irrelevant"
            # Every retained value, including collisions, reconstructs exactly.
            rr, cc = np.diff(np.r_[rows, s.shape[0]]), np.diff(np.r_[cols, s.shape[1]])
            assert np.array_equal(np.repeat(np.repeat(out, rr, axis=0), cc, axis=1), s)
            assert np.array_equal(_collapse(out), out)
            assert np.array_equal(_collapse(s.T).T, out), "axis order changed crossing values"
            d = p["d"]
            assert np.array_equal(s[[0, -1], d + 1:], a[[0, -1], d + 1:])
            assert np.array_equal(s[:, [d, d + 1, a.shape[1] - 1]],
                                  a[:, [d, d + 1, a.shape[1] - 1]])
            for t in p["templates"]:
                expected = p["bg"] if t["rows"] and t["cols"] else t["color"]
                assert np.all(s[t["cells"][:, 0], t["cells"][:, 1]] == expected)
            fs[split].append(_features(a))
    for name in ("reverse", "destination", "size", "opaque", "zip", "pre_partition",
                 "field_only_rows", "nonbg_row_repeat", "nonbg_col_repeat",
                 "separated_rows", "separated_cols"):
        assert any(f[name] for f in fs["train"]), f"missing training witness: {name}"
    assert set().union(*(f["inactive"] for f in fs["train"])) == {"N", "R", "C"}
    size_cap = max(f["max_size"] for f in fs["train"])
    mult_cap = max(f["max_mult"] for f in fs["train"])
    assert all(f["max_size"] > size_cap or f["max_mult"] > mult_cap for f in fs["test"])


def self_test():
    """42 in-memory seeds; no saved grids, trace files, or unbounded logs."""
    # Literal fixtures specify maximal, rather than global, deduplication.
    a = np.array([[1, 1, 2, 1, 1], [1, 1, 2, 1, 1],
                  [3, 3, 4, 3, 3], [1, 1, 2, 1, 1]])
    assert np.array_equal(_collapse(a), [[1, 2, 1], [3, 4, 3], [1, 2, 1]])
    shapes, motifs, layouts, backgrounds, puzzle_keys = set(), set(), set(), set(), set()
    both_axes = one_axis = size_shift = multiplicity_shift = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s])
        puzzle_keys.add(signature)
        if seed in (0, 41):
            other = generate_puzzle(seed)
            for split in ("train", "test"):
                for first, second in zip(puzzle[split], other[split]):
                    assert all(np.array_equal(first[role], second[role])
                               for role in ("input", "output"))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x, y = pair["input"], pair["output"]
                original = x.copy()
                assert np.array_equal(solve(x), y) and np.array_equal(x, original)
                p = _parse(x)
                f = _features(x)
                shapes.add(x.shape)
                backgrounds.add(p["bg"])
                motifs.update((t["shape"], t["mask"].tobytes()) for t in p["templates"])
                layouts.add((x.shape, p["d"], tuple((tuple(t["rows"]), tuple(t["cols"]))
                                                     for t in p["templates"])))
                both_axes += all(i > j for i, j in zip(x.shape, y.shape))
                one_axis += sum(i > j for i, j in zip(x.shape, y.shape)) == 1
                if split == "test":
                    size_shift += f["max_size"] == 7
                    multiplicity_shift += f["max_mult"] == 3
                # Bijection invariance also exercises black templates.
                mapping = (np.arange(10) + 3) % 10
                assert np.array_equal(solve(mapping[x]), mapping[y])
    assert len(puzzle_keys) == 42 and len(shapes) >= 50
    assert len(motifs) >= 40 and len(layouts) >= 150 and len(backgrounds) == 10
    assert both_axes >= 100 and size_shift >= 8 and multiplicity_shift >= 8
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1100 + nt * 10 + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject omission of either parent, wrong priority, and target-only equality.
    p = generate_puzzle(0, 2, 1)
    second = p["train"][1]["input"]
    info = _parse(second)
    stamped = _stamp(second, info)
    field_rows = _run_starts(stamped[:, info["d"] + 1:])[0]
    for index, wrong in ((0, _collapse(p["train"][0]["input"])),
                         (0, _stamp(p["train"][0]["input"])),
                         (0, _collapse(_stamp(p["train"][0]["input"], variant="reverse"))),
                         (1, _collapse(stamped[field_rows]))):
        bad = {s: [dict(pair) for pair in p[s]] for s in ("train", "test")}
        bad["train"][index]["output"] = wrong
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a parent omission or reviewed shortcut")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes), "masks": len(motifs),
            "layouts": len(layouts), "backgrounds": len(backgrounds), "both_axes_shrink": both_axes,
            "one_axis_shrinks": one_axis, "size_shift_tests": size_shift,
            "multiplicity_shift_tests": multiplicity_shift}
