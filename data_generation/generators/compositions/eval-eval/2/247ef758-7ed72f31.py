"""Addressed remote mirrors. Only NumPy and the Python standard library are used.

The solver reads a single grid. Generation records are used only to construct
labels, never by solve. A sparse set of stations gates Cartesian edge addresses.
"""
from __future__ import annotations

import copy
import numpy as np


KINDS = ("H", "V", "P")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        todo.append(q)
        result.append(component)
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a


def _read(value):
    a = _grid(value)
    h, w = a.shape
    assert h >= 7 and w >= 7
    frame = int(a[0, 0])
    assert all(int(a[r, c]) == frame for r, c in ((0, w-1), (h-1, 0), (h-1, w-1)))
    body = a[1:-1, 1:-1]
    counts = np.bincount(body.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert bg != frame and counts[bg] > body.size / 2, "visible majority background"
    rows, cols = {}, {}
    assert np.array_equal(a[1:-1, 0], a[1:-1, -1])
    assert np.array_equal(a[0, 1:-1], a[-1, 1:-1])
    for r in range(1, h-1):
        color = int(a[r, 0])
        if color != frame:
            assert color != bg
            rows[r] = color
    for c in range(1, w-1):
        color = int(a[0, c])
        if color != frame:
            assert color != bg
            cols[c] = color
    sources = {}
    for color in np.unique(body):
        color = int(color)
        if color in (bg, frame):
            continue
        cells = {tuple(map(int, q + 1)) for q in np.argwhere(body == color)}
        assert len(_components(cells)) == 1, "one 8-connected original per color"
        sources[color] = cells
    assert sources and set(rows.values()) | set(cols.values()) <= sources.keys()
    marker_cells = {tuple(map(int, q + 1)) for q in np.argwhere(body == frame)}
    stations = []
    for cells in _components(marker_cells):
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) == 1:
            kind = "P"
        elif r0 == r1 and len(cells) == c1-c0+1 and len(cells) % 2:
            kind = "H"
        elif c0 == c1 and len(cells) == r1-r0+1 and len(cells) % 2:
            kind = "V"
        else:
            raise AssertionError("stations must be dots or odd straight segments")
        r, c = (r0+r1)//2, (c0+c1)//2
        selected = rows.get(r) if rows.get(r) == cols.get(c) else None
        stations.append(dict(kind=kind, center=(r, c), cells=cells, color=selected))
    assert stations
    return dict(grid=a, bg=bg, frame=frame, rows=rows, cols=cols,
                sources=sources, stations=stations, shape=a.shape)


def _image(cells, kind, center):
    ar, ac = center
    if kind == "H":
        return {(2*ar-r, c) for r, c in cells}
    if kind == "V":
        return {(r, 2*ac-c) for r, c in cells}
    return {(2*ar-r, 2*ac-c) for r, c in cells}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply all and only addressed stations to the original input sources."""
    p = _read(input_grid)
    out = p["grid"].copy()
    for s in p["stations"]:
        if s["color"] is None:
            continue
        for r, c in _image(p["sources"][s["color"]], s["kind"], s["center"]):
            assert 1 <= r < out.shape[0]-1 and 1 <= c < out.shape[1]-1
            assert int(out[r, c]) in (p["bg"], s["color"])
            out[r, c] = s["color"]
    return out


def _silhouette(rng, size, witness=False):
    """Procedural tight masks, including simple, sparse and hollow shapes."""
    h, w = int(rng.integers(2, size+1)), int(rng.integers(2, size+1))
    mode = int(rng.integers(6))
    m = np.zeros((h, w), bool)
    if mode == 0:
        m[:] = True
    elif mode == 1:
        m[0, :] = m[-1, :] = True
        m[:, 0] = m[:, -1] = True
    elif mode == 2:
        for r in range(h):
            m[r, round(r*(w-1)/(h-1))] = True
        for c in range(w):
            m[round(c*(h-1)/(w-1)), c] = True
    elif mode == 3:
        m[:] = True
        m[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
    else:
        # Join four independently located extremes by king-neighbor paths.
        pts = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
               (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        rng.shuffle(pts)
        r, c = pts[0]
        m[r, c] = True
        for tr, tc in pts[1:]:
            while (r, c) != (tr, tc):
                dr, dc = int(np.sign(tr-r)), int(np.sign(tc-c))
                if dr and dc and rng.random() < .4:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r+dr, c+dc
                m[r, c] = True
        m |= rng.random((h, w)) < rng.uniform(0, .2)
        # Random additions are allowed only in the connected component of the path.
        cells = set(map(tuple, np.argwhere(m)))
        comp = max(_components(cells), key=len)
        m[:] = False
        for q in comp:
            m[q] = True
    if witness and any(np.array_equal(m, t) for t in (m[::-1], m[:, ::-1], m[::-1, ::-1])):
        m[:] = True
        m[0, 0] = False
    if rng.random() < .5:
        m = m[::-1]
    if rng.random() < .5:
        m = m[:, ::-1]
    return m.copy()


def _flipped_cells(source, mask, kind, center):
    """Construction path uses whole-mask flips, separate from solver mapping."""
    top, _, left, _ = _bbox(source)
    ar, ac = center
    h, w = mask.shape
    if kind in ("H", "P"):
        mask = np.flipud(mask)
        top = 2*ar-(top+h-1)
    if kind in ("V", "P"):
        mask = np.fliplr(mask)
        left = 2*ac-(left+w-1)
    return {(top+int(r), left+int(c)) for r, c in np.argwhere(mask)}


def _station_cells(kind, r, c, length):
    if kind == "P":
        return {(r, c)}
    rad = length//2
    if kind == "H":
        return {(r, c+d) for d in range(-rad, rad+1)}
    return {(r+d, c) for d in range(-rad, rad+1)}


def _one_side(source, kind, center):
    r0, r1, c0, c1 = _bbox(source)
    r, c = center
    return ((kind == "V" or r < r0 or r > r1)
            and (kind == "H" or c < c0 or c > c1))


def _try_pair(rng, plan, count, dimensions, witness, decoy_modes):
    h, w = dimensions
    palette = list(map(int, rng.permutation(10)))
    bg, frame = palette[:2]
    colors = palette[2:2+count]
    source_records, sources = [], set()
    for i, color in enumerate(colors):
        m = _silhouette(rng, int(rng.choice([3, 4, 5])), witness and i == 0)
        mh, mw = m.shape
        placed = False
        for _ in range(70):
            r = int(rng.integers(1, h-mh))
            c = int(rng.integers(1, w-mw))
            cells = {(r+int(y), c+int(x)) for y, x in np.argwhere(m)}
            if not _halo(cells) & sources:
                source_records.append(dict(color=color, mask=m, cells=cells))
                sources.update(cells)
                placed = True
                break
        if not placed:
            return None
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[[0, -1], :] = frame
    grid[:, [0, -1]] = frame
    for rec in source_records:
        for q in rec["cells"]:
            grid[q] = rec["color"]
    output = grid.copy()
    rows, cols, stations = {}, {}, []
    marker_cells, image_owner, ghost_cells = set(), {}, set()

    def place(rec, kind, active=True, mode=None):
        color, source = rec["color"], rec["cells"]
        length = 1 if kind == "P" else int(rng.choice([3, 5, 7]))
        candidates = [(r, c) for r in range(2, h-2) for c in range(2, w-2)]
        rng.shuffle(candidates)
        for r, c in candidates:
            if active:
                if rows.get(r, color) != color or cols.get(c, color) != color:
                    continue
            elif mode == "row":
                if rows.get(r) != color or c in cols:
                    continue
            elif mode == "col":
                if cols.get(c) != color or r in rows:
                    continue
            elif mode == "none":
                if r in rows or c in cols:
                    continue
            if not _one_side(source, kind, (r, c)):
                continue
            station = _station_cells(kind, r, c, length)
            if active and witness and rec is source_records[0]:
                if kind in ("H", "V"):
                    tangent = 1 if kind == "H" else 0
                    low = min(q[tangent] for q in station)
                    high = max(q[tangent] for q in station)
                    if all(low <= q[tangent] <= high for q in source):
                        continue  # teach that finite endpoints do not clip images
                elif any(s["color"] == color and s["kind"] == "P"
                         and (s["center"][0] == r or s["center"][1] == c)
                         for s in stations):
                    continue  # repeated dots must expose both coordinate axes
            if any(not (2 <= y < h-2 and 2 <= x < w-2) for y, x in station):
                continue
            if _halo(station) & (marker_cells | sources):
                continue
            if station & (image_owner.keys() | ghost_cells):
                continue
            image = _flipped_cells(source, rec["mask"], kind, (r, c))
            if any(not (1 <= y < h-1 and 1 <= x < w-1) for y, x in image):
                continue
            if image & (sources | marker_cells | station):
                continue
            if any(image_owner.get(q, color) != color for q in image):
                continue
            if not image - image_owner.keys():
                continue
            if active:
                rows[r] = cols[c] = color
                image_owner.update({q: color for q in image})
                for q in image:
                    output[q] = color
            else:
                # An idle station has a clear, visibly absent hypothetical image.
                if image & image_owner.keys():
                    continue
                ghost_cells.update(image)
            for q in station:
                grid[q] = output[q] = frame
            marker_cells.update(station)
            stations.append(dict(kind=kind, center=(r, c), cells=station,
                                 color=color if active else None))
            return True
        return False

    # First source carries the curriculum's interacting routing pattern.
    for kind in plan:
        if not place(source_records[0], kind):
            return None
    for rec in source_records[1:]:
        for kind in rng.choice(KINDS, size=int(rng.integers(1, 3)), replace=False):
            if not place(rec, str(kind)):
                return None
    for mode in decoy_modes:
        kinds = list(rng.permutation(KINDS))
        if not any(place(source_records[0], str(kind), False, mode) for kind in kinds):
            return None
    # Some otherwise valid Cartesian addresses contain no station at all.
    # Extra edge labels may be added without changing station eligibility.
    for mapping, length, axis in ((rows, h, 0), (cols, w, 1)):
        for _ in range(int(rng.integers(3))):
            options = [v for v in range(1, length-1) if v not in mapping
                       and all(s["center"][axis] != v for s in stations)]
            if options:
                mapping[int(rng.choice(options))] = int(rng.choice(colors))
    for r, color in rows.items():
        grid[r, 0] = grid[r, -1] = output[r, 0] = output[r, -1] = color
    for c, color in cols.items():
        grid[0, c] = grid[-1, c] = output[0, c] = output[-1, c] = color
    pair = {"input": grid, "output": output}
    # Rotate the complete coordinate language, sources and stations together.
    turns = int(rng.integers(4))
    pair = {k: np.rot90(v, turns).copy() for k, v in pair.items()}
    try:
        p = _read(pair["input"])
        _check_pair(p, pair["output"])
    except AssertionError:
        return None
    return pair


def _make_pair(rng, plan, count, witness=False, decoy_modes=("none",)):
    for attempt in range(150):
        # Overlapping split ranges: the shift is routing composition, not size.
        lo = 17 if attempt < 70 else 23
        dims = (int(rng.integers(lo, 31)), int(rng.integers(lo, 31)))
        pair = _try_pair(rng, plan, count, dims, witness, decoy_modes)
        if pair is not None:
            return pair
    raise RuntimeError("bounded scene packing exhausted")


def _orientation_difference(cells, kind, center):
    actual = _image(cells, kind, center)
    r0, _, c0, _ = _bbox(cells)
    t0, _, u0, _ = _bbox(actual)
    translated = {(r+t0-r0, c+u0-c0) for r, c in cells}
    return actual != translated


def _check_pair(p, out):
    a, bg = p["grid"], p["bg"]
    assert a.shape == out.shape and not np.array_equal(a, out)
    assert np.array_equal(a[a != bg], out[a != bg]), "only background can change"
    originals = set().union(*p["sources"].values())
    markers = set().union(*(s["cells"] for s in p["stations"]))
    owners, used = {}, set()
    for s in p["stations"]:
        if s["color"] is None:
            continue
        color, source = s["color"], p["sources"][s["color"]]
        assert _one_side(source, s["kind"], s["center"])
        image = _image(source, s["kind"], s["center"])
        assert not image & (originals | markers)
        for r, c in image:
            assert 1 <= r < a.shape[0]-1 and 1 <= c < a.shape[1]-1
            assert owners.get((r, c), color) == color, "different colors collide"
            owners[r, c] = color
        used.add(color)
    assert used == set(p["sources"]), "each source participates"
    changed = set(map(tuple, np.argwhere(a != out)))
    assert changed == owners.keys(), "exact union of addressed images"
    assert all(int(out[q]) == color for q, color in owners.items())
    assert np.array_equal(solve(a), out)


def _facts(pair):
    p = _read(pair["input"])
    active = [s for s in p["stations"] if s["color"] is not None]
    idle = [s for s in p["stations"] if s["color"] is None]
    witnesses = {s["kind"] for s in active if _orientation_difference(
        p["sources"][s["color"]], s["kind"], s["center"])}
    modes, endpoint_diff = set(), False
    for s in idle:
        r, c = s["center"]
        rc, cc = p["rows"].get(r), p["cols"].get(c)
        for color, source in p["sources"].items():
            if not _one_side(source, s["kind"], s["center"]):
                continue
            image = _image(source, s["kind"], s["center"])
            if all(1 <= y < p["shape"][0]-1 and 1 <= x < p["shape"][1]-1
                   and pair["input"][y, x] == p["bg"] for y, x in image):
                if any(pair["output"][q] == p["bg"] for q in image):
                    endpoint_diff |= any(p["rows"].get(y) == color and p["cols"].get(x) == color
                                         for y, x in s["cells"])
                    if rc == color and cc is None:
                        modes.add("row")
                    if cc == color and rc is None:
                        modes.add("col")
                    if rc is None and cc is None:
                        modes.add("none")
    # Compare full Cartesian routing with sorted zip of the two coordinate lists.
    zipped = set()
    for color in p["sources"]:
        rs = sorted(r for r, v in p["rows"].items() if v == color)
        cs = sorted(c for c, v in p["cols"].items() if v == color)
        zipped.update((r, c, color) for r, c in zip(rs, cs))
    zip_output = pair["input"].copy()
    for s in active:
        if (*s["center"], s["color"]) in zipped:
            for q in _image(p["sources"][s["color"]], s["kind"], s["center"]):
                zip_output[q] = s["color"]
    # A second operator must not consume the image created by a first operator.
    feedback = False
    for a in active:
        for b in active:
            if a["color"] != b["color"] or a is b:
                continue
            twice = _image(_image(p["sources"][a["color"]], a["kind"], a["center"]),
                           b["kind"], b["center"])
            feedback |= any(1 <= r < p["shape"][0]-1 and 1 <= c < p["shape"][1]-1
                            and pair["output"][r, c] == p["bg"] for r, c in twice)
    by_color = {color: {s["kind"] for s in active if s["color"] == color}
                for color in p["sources"]}
    matrix = any(len({s["center"][0] for s in active if s["color"] == color}) >= 2
                 and len({s["center"][1] for s in active if s["color"] == color}) >= 2
                 for color in p["sources"])
    overhang = set()
    for s in active:
        if s["kind"] == "P":
            continue
        axis = 1 if s["kind"] == "H" else 0
        lo, hi = min(q[axis] for q in s["cells"]), max(q[axis] for q in s["cells"])
        if any(not lo <= q[axis] <= hi for q in p["sources"][s["color"]]):
            overhang.add(s["kind"])
    return dict(parsed=p, active=active, idle=idle, witnesses=witnesses,
                modes=modes, zip_diff=not np.array_equal(zip_output, pair["output"]),
                feedback=feedback, matrix=matrix, overhang=overhang, endpoint_diff=endpoint_diff,
                max_kinds=max(map(len, by_color.values())))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # These are evidence plans, not hidden conventions. Each pair uses the same rule.
    plans = [("H", "V"), ("P", "P")]
    train = []
    for plan in plans:
        for _ in range(60):
            pair = _make_pair(rng, plan, int(rng.choice([1, 1, 2])), True, ("row", "col"))
            f = _facts(pair)
            needed = set(plan)
            if needed <= f["witnesses"] and f["zip_diff"] and f["feedback"] and {"row", "col"} <= f["modes"]:
                train.append(pair)
                break
        else:
            raise RuntimeError("could not expose routing and orientation")
    for _ in range(nt-2):
        kinds = list(map(str, rng.choice(KINDS, size=int(rng.integers(1, 3)), replace=False)))
        if rng.random() < .35:
            kinds.append(kinds[0])
        train.append(_make_pair(rng, kinds, int(rng.integers(1, 4)), False,
                                ("none",) if rng.random() < .5 else ("row", "col")))
    rng.shuffle(train)
    midpoint_taught = any(_facts(pair)["endpoint_diff"] for pair in train)
    test = []
    for _ in range(ne):
        for _ in range(60):
            pair = _make_pair(rng, ("H", "V", "P"), int(rng.choice([1, 2, 2, 3])), False,
                              ("row", "col", "none") if rng.random() < .4 else ("row", "col"))
            if midpoint_taught or not _facts(pair)["endpoint_diff"]:
                test.append(pair)
                break
        else:
            raise RuntimeError("could not obtain a test with demonstrated anchor behavior")
    seen = set()
    for pair in train + test:
        # Bijections preserve all routing and geometry; at most 13 prior inputs.
        original = {k: v.copy() for k, v in pair.items()}
        for trial in range(30):
            if trial:
                palette = rng.permutation(10)
                pair.update({k: palette[v].astype(np.int8) for k, v in original.items()})
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("duplicate-input repair exhausted")
        seen.add(_key(pair["input"]))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    facts, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            parsed = _read(inp)
            _check_pair(parsed, out)
            facts[split].append(_facts({"input": inp, "output": out}))
    train = facts["train"]
    assert set().union(*(f["witnesses"] for f in train)) == set(KINDS)
    assert {"row", "col"} <= set().union(*(f["modes"] for f in train))
    assert any(f["zip_diff"] for f in train)
    assert any(f["feedback"] for f in train)
    assert any(f["matrix"] for f in train), "both coordinate axes must independently vary"
    assert {"H", "V"} <= set().union(*(f["overhang"] for f in train))
    if any(f["endpoint_diff"] for f in facts["test"]):
        assert any(f["endpoint_diff"] for f in train), "midpoint addressing is underdemonstrated"
    assert all(f["max_kinds"] <= 2 for f in train)
    assert all(f["max_kinds"] == 3 for f in facts["test"]), "routing-combination test shift"


def self_test():
    """Bounded deterministic, mutation, geometric and 42-seed variation checks."""
    # Literal expected cells: a tiny asymmetric L, remote address (6,7),
    # and two idle stations matching only one axis. No generation is involved.
    for kind, expected in (("H", {(10,2),(10,3),(9,2)}),
                           ("V", {(2,12),(2,11),(3,12)}),
                           ("P", {(10,12),(10,11),(9,12)})):
        a = np.zeros((13,15), dtype=np.int8)
        a[[0,-1],:] = 8; a[:,[0,-1]] = 8
        a[6,0] = a[6,-1] = a[0,7] = a[-1,7] = 2
        a[2,2] = a[2,3] = a[3,2] = 2
        marker = ({(6,7)} if kind == "P" else
                  {(6,c) for c in (6,7,8)} if kind == "H" else
                  {(r,7) for r in (5,6,7)})
        marker |= {(6,c) for c in (10,11,12)} | {(r,7) for r in (9,10,11)}
        for q in marker:
            a[q] = 8
        b = a.copy()
        for q in expected:
            b[q] = 2
        assert np.array_equal(solve(a), b)
    signatures, shapes, silhouettes, station_counts, source_counts = set(), set(), set(), set(), set()
    backgrounds, frames, lengths, diagonal_masks, non_solid_masks = set(), set(), set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(x["input"]) for s in puzzle.values() for x in s))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _read(pair["input"])
                shapes.add(p["shape"])
                station_counts.add(len(p["stations"]))
                source_counts.add(len(p["sources"]))
                backgrounds.add(p["bg"])
                frames.add(p["frame"])
                lengths.update(len(s["cells"]) for s in p["stations"])
                for cells in p["sources"].values():
                    r0, r1, c0, c1 = _bbox(cells)
                    motif = tuple(sorted((r-r0, c-c0) for r, c in cells))
                    silhouettes.add((r1-r0+1, c1-c0+1, motif))
                    # Four-connectivity is not required: compare its explicit flood.
                    left = set(cells)
                    todo = [left.pop()]
                    while todo:
                        r, c = todo.pop()
                        for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                            if q in left:
                                left.remove(q); todo.append(q)
                    diagonal_masks += bool(left)
                    non_solid_masks += ((r1-r0+1)*(c1-c0+1) > len(cells)
                                     and (r1-r0+1) >= 3 and (c1-c0+1) >= 3)
                # Solving cannot depend on dtype, construction state, or mutation.
                a = pair["input"].copy()
                assert np.array_equal(solve(a.astype(np.int64)), pair["output"])
                assert np.array_equal(a, pair["input"])
                # Full palette equivariance includes black as a payload color.
                recolor = np.array([9, 4, 1, 7, 0, 6, 3, 8, 2, 5])
                assert np.array_equal(solve(recolor[a]), recolor[pair["output"]])
        # Removing B leaves unturned images in their correct destination boxes.
        if seed == 0:
            omitted_b_differs = False
            for pair in puzzle["train"]:
                f = _facts(pair)
                p = f["parsed"]
                wrong = pair["input"].copy()
                for s in f["active"]:
                    source = p["sources"][s["color"]]
                    image = _image(source, s["kind"], s["center"])
                    r0, _, c0, _ = _bbox(source)
                    t0, _, u0, _ = _bbox(image)
                    for r, c in source:
                        wrong[r+t0-r0, c+u0-c0] = s["color"]
                omitted_b_differs |= not np.array_equal(wrong, pair["output"])
            assert omitted_b_differs
            corrupted = copy.deepcopy(puzzle)
            a = corrupted["train"][0]["output"]
            changed = np.argwhere(a != corrupted["train"][0]["input"])[0]
            a[tuple(changed)] = corrupted["train"][0]["input"][tuple(changed)]
            try:
                validate_generated_puzzle(corrupted)
            except AssertionError:
                pass
            else:
                raise AssertionError("accepted a missing image pixel")
    assert len(signatures) == 42 and len(shapes) >= 70 and len(silhouettes) >= 90
    assert {1, 2, 3} <= source_counts and len(station_counts) >= 5
    assert len(backgrounds) == len(frames) == 10 and {1, 3, 5, 7} <= lengths
    assert diagonal_masks >= 5 and non_solid_masks >= 20
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91000+nt*10+ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(123456, 2, 1), generate_puzzle(123456, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    return {"seeds": 42, "puzzles": len(signatures), "grid_shapes": len(shapes),
            "silhouettes": len(silhouettes), "station_counts": sorted(station_counts),
            "source_counts": sorted(source_counts), "backgrounds": sorted(backgrounds),
            "frame_colors": sorted(frames), "station_lengths": sorted(lengths)}
