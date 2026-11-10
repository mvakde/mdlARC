"""Shape-family coordinate loom. All semantic information is in the input.

The constructor paints from its sampled records; solve reconstructs those records
from geometry and colors. No parent generator, filesystem, or seed is used by solve.
"""
from collections import Counter
import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 1..30 rectangles with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _mask_key(m):
    return m.shape, tuple(map(int, m.flat))


def _canonical(m, mode="rotation"):
    if mode == "area":
        return int(m.sum())
    if mode == "identity":
        return _mask_key(m)
    turns = (0, 2) if mode == "half_turn" else range(4)
    masks = [np.rot90(m, k) for k in turns]
    if mode == "reflection":
        masks += [np.rot90(m[:, ::-1], k) for k in range(4)]
    return min(_mask_key(x) for x in masks)


def _connected(m, directions=DIR8):
    cells = set(map(tuple, np.argwhere(m)))
    if not cells:
        return False
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == cells


def _parse(value):
    a = _grid(value)
    h, w = a.shape
    layouts = []
    for d in range(1, w - 4):
        divider = int(a[0, d])
        if not np.all(a[:, d] == divider):
            continue
        interior = a[1:-1, d+2:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, frame = int(interior[0, 0]), int(a[0, d+1])
        if len({bg, frame, divider}) != 3:
            continue
        if not all(int(a[r, c]) == frame for r, c in
                   ((0, d+1), (0, w-1), (h-1, d+1), (h-1, w-1))):
            continue
        if not np.array_equal(a[1:-1, d+1], a[1:-1, -1]):
            continue
        if not np.any(a[:, :d] != bg) or np.any(
                (a[:, :d] == divider) | (a[:, :d] == frame)):
            continue
        edges = np.concatenate((a[0, d+2:-1], a[-1, d+2:-1], a[1:-1, d+1]))
        if np.any((edges == bg) | (edges == divider)):
            continue
        layouts.append((d, bg, frame, divider))
    if len(layouts) != 1:
        raise ValueError("input must have one visible source/divider/framed-field layout")
    d, bg, frame, divider = layouts[0]
    records = []
    for color in sorted(set(map(int, a[:, :d].flat)) - {bg}):
        cells = np.argwhere(a[:, :d] == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        bh, bw = map(int, hi - lo + 1)
        m = a[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        if bh % 2 != 1 or bw % 2 != 1 or not _connected(m):
            raise ValueError("each source color must form one connected, odd-box template")
        records.append(dict(color=color, mask=m, cells=cells,
                            top=int(lo[0]), left=int(lo[1]),
                            rows=list(map(int, np.flatnonzero(a[:, d+1] == color))),
                            upper=list(map(int, np.flatnonzero(a[0] == color)[
                                np.flatnonzero(a[0] == color) > d+1])),
                            lower=list(map(int, np.flatnonzero(a[-1] == color)[
                                np.flatnonzero(a[-1] == color) > d+1]))))
    records.sort(key=lambda t: (t["top"], t["left"], t["color"]))
    return a, dict(d=d, bg=bg, frame=frame, divider=divider, records=records)


def _placements(p, match="rotation", share=True, rail="correct", threshold="any"):
    """Diagnostic alternatives are explicit; the public rule uses defaults."""
    ts = p["records"]
    keys = [_canonical(t["mask"], match) for t in ts]
    groups = {key: [j for j, k in enumerate(keys) if k == key] for key in keys}
    result = []
    for i, t in enumerate(ts):
        group = groups[keys[i]]
        repeated = len(group) > 1 if threshold == "any" else len(group) == 2
        names = ["upper" if repeated else "lower"]
        if rail == "both":
            names = ["upper", "lower"]
        elif rail == "top":
            names = ["upper"]
        elif rail == "bottom":
            names = ["lower"]
        elif rail == "reverse":
            names = ["lower" if repeated else "upper"]
        donors = group if repeated and share else [i]
        cols = sorted({c for j in donors for name in names for c in ts[j][name]})
        offsets = np.argwhere(t["mask"]) - np.array(t["mask"].shape) // 2
        stamps = [{(r+int(y), c+int(x)) for y, x in offsets}
                  for r in t["rows"] for c in cols]
        result.append(dict(template=t, repeated=repeated, group=group,
                           columns=cols, stamps=stamps,
                           cells=set().union(*stamps) if stamps else set()))
    return result


def _predict(value, **alternatives):
    a, p = _parse(value)
    out = a.copy()
    occupied = set()
    if not alternatives.pop("coordinates", True):
        # Parent-B-only ablation: sort the source objects to the strip's
        # top/bottom edges at their original columns, ignoring frame addresses.
        frequencies = Counter(_canonical(t["mask"]) for t in p["records"])
        out[:, :p["d"]] = p["bg"]
        for t in p["records"]:
            top = 0 if frequencies[_canonical(t["mask"])] > 1 else a.shape[0]-t["mask"].shape[0]
            for dr, dc in np.argwhere(t["mask"]):
                r, c = top+int(dr), t["left"]+int(dc)
                if (r, c) not in occupied:
                    out[r, c] = t["color"]
                    occupied.add((r, c))
        return out
    for item in _placements(p, **alternatives):
        t = item["template"]
        if not item["stamps"]:
            continue
        out[t["cells"][:, 0], t["cells"][:, 1]] = p["bg"]
        for r, c in sorted(item["cells"]):
            if not (1 <= r < a.shape[0]-1 and p["d"]+2 <= c < a.shape[1]-1):
                raise ValueError("a requested stamp would leave the field")
            # Intended inputs have no cross-owner intersections. Diagnostic
            # alternatives can collide, and use reading order for comparison.
            if (r, c) not in occupied:
                out[r, c] = t["color"]
                occupied.add((r, c))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _predict(input_grid)


def _sample_mask(rng, max_side=5):
    sides = [1, 3] + ([5] if max_side >= 5 else [])
    h, w = map(int, rng.choice(sides, size=2, p=
                            ([.12, .58, .30] if len(sides) == 3 else [.15, .85])))
    m = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(7))
    if min(h, w) == 1 or kind == 0:
        m[:] = True
    elif kind == 1:
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
    elif kind == 2:
        for r in range(h):
            m[r, round(r * (w-1) / (h-1))] = True
        for c in range(w):
            m[round(c * (h-1) / (w-1)), c] = True
    elif kind == 3:
        m[:, int(rng.integers(w))] = True
        m[int(rng.integers(h)), :] = True
    else:
        # Random paths between extremes keep an exact odd bounding box.
        points = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        rng.shuffle(points)
        r, c = points[0]
        m[r, c] = True
        for tr, tc in points[1:]:
            while (r, c) != (tr, tc):
                dr, dc = int(np.sign(tr-r)), int(np.sign(tc-c))
                if dr and dc and rng.random() < .5:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r+dr, c+dc
                m[r, c] = True
        target = int(rng.uniform(.20, .95) * h * w)
        while m.sum() < target:
            frontier = [(r, c) for r in range(h) for c in range(w)
                        if not m[r, c] and m[max(0,r-1):r+2,max(0,c-1):c+2].any()]
            r, c = frontier[int(rng.integers(len(frontier)))]
            m[r, c] = True
    return np.rot90(m, int(rng.integers(4))).copy()


def _new_shape(rng, forbidden=(), max_side=5, chiral=False):
    forbidden = set(forbidden)
    for _ in range(160):
        m = _sample_mask(rng, max_side)
        if (_canonical(m) not in forbidden and (not chiral or
                (m.shape == (3, 3) and _canonical(m) != _canonical(m[:, ::-1])
                 and len({_mask_key(np.rot90(m, k)) for k in range(4)}) == 4))):
            return m
    # Exhaustive 3x3 fallback is bounded and obeys the same mask constraints.
    for bits in range(1, 512):
        m = np.array([(bits >> i) & 1 for i in range(9)], bool).reshape(3, 3)
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()
                and _connected(m)):
            continue
        if (_canonical(m) not in forbidden and (not chiral or
                (_canonical(m) != _canonical(m[:, ::-1]) and
                 len({_mask_key(np.rot90(m, k)) for k in range(4)}) == 4))):
            return m
    raise AssertionError("mask vocabulary exhausted")


def _plan(rng, kind, multiplicity, bank=()):
    """Construct classes before assigning any coordinates or source positions."""
    ts, classes = [], []

    def add_class(base, n, active, row_counts=None):
        indices = []
        start = int(rng.integers(4))
        for j in range(n):
            turn = start if j == 0 else ((start+1) % 4 if j == 1 else int(rng.integers(4)))
            m = np.rot90(base, turn).copy()
            indices.append(len(ts))
            ts.append(dict(mask=m, row_count=(row_counts[j] if row_counts else
                           int(rng.integers(1, 4))) if j < active else 0,
                           rows=[], upper=[], lower=[]))
        classes.append(indices)
        return indices

    if kind == "rotation":
        base = _new_shape(rng, max_side=3, chiral=True)
        pair = add_class(base, 2, 2, [2, 1])
        mirror = add_class(base[:, ::-1], 1, 1, [1])
        columns = [(pair, 2, [pair[1]]), (mirror, 1, mirror)]
    elif kind == "donor":
        base = _new_shape(rng)
        group = add_class(base, multiplicity, 2, [3, 1] + [0]*(multiplicity-2))
        forbidden = {_canonical(base), _canonical(base[:, ::-1])}
        single = _new_shape(rng, forbidden)
        unique = add_class(single, 1, 1, [2])
        forbidden.update((_canonical(single), _canonical(single[:, ::-1])))
        inactive = add_class(_new_shape(rng, forbidden), 1, 1, [1])
        # The rowless member is the only top donor, so it remains at source
        # while its two companions can use every one of its coordinates.
        columns = [(group, int(rng.integers(1, 4)), [group[-1]]),
                   (unique, 2, unique), (inactive, 0, inactive)]
    else:
        # Tests combine two demonstrated classes in a new membership context.
        # Extras use the same space, sometimes with just one repeated class.
        bank = list(bank)
        if not bank:
            bank = [_new_shape(rng), _new_shape(rng)]
        rng.shuffle(bank)
        bases = []
        for m in bank:
            if _canonical(m) not in {_canonical(x) for x in bases}:
                bases.append(m)
        while len(bases) < 3:
            bases.append(_new_shape(rng, [_canonical(x) for x in bases]))
        sizes = [int(rng.integers(2, multiplicity+1)), 2] if kind == "test" or rng.random() < .5 else [int(rng.integers(2, multiplicity+1))]
        if sum(sizes)+1 > 7:
            sizes[0] = 4
        columns = []
        for base, n in zip(bases, sizes):
            active = int(rng.integers(1, min(n, 3)+1))
            group = add_class(base, n, active)
            donors = list(map(int, rng.choice(group, size=int(rng.integers(1, n+1)), replace=False)))
            columns.append((group, int(rng.integers(1, 4)), donors))
        unique = add_class(bases[len(sizes)], 1, 1)
        columns.append((unique, int(rng.integers(1, 4)), unique))
        if len(ts) < 7 and rng.random() < .45:
            extra = _new_shape(rng, [_canonical(t["mask"]) for t in ts])
            group = add_class(extra, 1, int(rng.integers(2)))
            columns.append((group, int(rng.integers(2)), group))
    # All axis positions are filled later; zero chosen columns leaves source
    # objects intact even when they have row marks on the frame.
    for group, count, donors in columns:
        for i in group:
            ts[i]["column_count"] = count
    for t in ts:
        t["row_spacing"] = int(rng.choice([1, 2, 3]))
    active = [i for i, t in enumerate(ts) if t["row_count"] and t["column_count"]]
    # Fit row bands by reducing replication first, then excess active members.
    def need():
        return sum(ts[i]["mask"].shape[0] + (ts[i]["row_count"]-1)*ts[i]["row_spacing"]
                   for i in active) + len(active)-1 + 2
    while need() > 30:
        spread = [i for i in active if ts[i]["row_count"] > 1 and ts[i]["row_spacing"] > 1]
        if spread:
            ts[int(rng.choice(spread))]["row_spacing"] -= 1
            continue
        extra = [i for i in active if ts[i]["row_count"] > 1]
        if extra:
            ts[int(rng.choice(extra))]["row_count"] -= 1
        else:
            removable = [i for g in classes if len(g) > 1 for i in g
                         if i in active and sum(j in active for j in g) > 1]
            i = int(rng.choice(removable))
            ts[i]["row_count"] = 0
            active.remove(i)
    return ts, classes, columns, active


def _source_positions(rng, ts, h, sw):
    occupied = set()
    order = sorted(range(len(ts)), key=lambda i: -int(ts[i]["mask"].sum()))
    for i in order:
        t = ts[i]
        bh, bw = t["mask"].shape
        local = list(map(tuple, np.argwhere(t["mask"])))
        halo = {(r+dr, c+dc) for r, c in occupied for dr, dc in ((0, 0),)+DIR8}
        choices = [(r, c) for r in range(h-bh+1) for c in range(sw-bw+1)
                   if not any((r+int(y), c+int(x)) in halo for y, x in local)]
        if not choices:
            return False
        r, c = choices[int(rng.integers(len(choices)))]
        t["top"], t["left"] = r, c
        occupied.update((r+int(y), c+int(x)) for y, x in local)
    return True


def _make_pair(rng, kind, multiplicity, bank=()):
    ts, classes, columns, active = _plan(rng, kind, multiplicity, bank)
    row_need = sum(ts[i]["mask"].shape[0] + (ts[i]["row_count"]-1)*ts[i]["row_spacing"]
                   for i in active) + len(active)-1
    # Inactive owners with row-only evidence also need isolated labels, though
    # their unused rows do not reserve stamp bands.
    row_only = [i for i, t in enumerate(ts) if t["row_count"] and not t["column_count"]]
    minimum = max(18 if len(ts) >= 5 else 13, row_need+2)
    h = int(rng.integers(minimum, 31))
    sw = int(rng.integers(7, 12))
    if max(t["mask"].shape[1] for t in ts) == 5 and len(ts) >= 5:
        sw = 11
    # Bounded placement; the two-column box layout always fits in 23x11.
    for _ in range(12):
        if _source_positions(rng, ts, h, sw):
            break
    else:
        h, sw = max(h, 23), 11
        size = max(max(t["mask"].shape) for t in ts)
        for slot, i in enumerate(map(int, rng.permutation(len(ts)))):
            ts[i]["top"], ts[i]["left"] = (slot//2)*(size+1), (slot%2)*(size+1)
    fw = int(rng.integers(14, 30-sw))
    w, left_edge = sw+1+fw, sw+1
    order = list(map(int, rng.permutation(active)))
    slack = h-2-row_need
    gaps = rng.multinomial(slack, np.full(len(order)+1, 1/(len(order)+1)))
    cursor = 1+int(gaps[0])
    for j, i in enumerate(order):
        t = ts[i]
        radius = t["mask"].shape[0]//2
        t["rows"] = [cursor+radius+k*t["row_spacing"] for k in range(t["row_count"])]
        cursor += t["mask"].shape[0]+(t["row_count"]-1)*t["row_spacing"]+1+int(gaps[j+1])
    used_rows = {r for t in ts for r in t["rows"]}
    for i in row_only:
        rad = ts[i]["mask"].shape[0]//2
        free = [r for r in range(1+rad, h-1-rad) if r not in used_rows]
        if free:
            r = int(rng.choice(free))
            ts[i]["rows"] = [r]
            used_rows.add(r)
    max_radius = max(t["mask"].shape[1]//2 for t in ts)
    rail_free = {name: list(range(left_edge+1+max_radius, w-1-max_radius))
                 for name in ("upper", "lower")}
    for name in rail_free:
        rng.shuffle(rail_free[name])
    for group, count, donors in columns:
        name = "upper" if len(group) > 1 else "lower"
        for k in range(count):
            ts[donors[k % len(donors)]][name].append(rail_free[name].pop())
        # Every class has a wrong-rail distractor. This is essential evidence
        # against pooling both edges and against using a fixed edge for all.
        wrong = "lower" if name == "upper" else "upper"
        if rail_free[wrong]:
            ts[int(rng.choice(group))][wrong].append(rail_free[wrong].pop())
    palette = list(map(int, rng.permutation(10)))
    bg, divider, frame = palette[:3]
    for t, color in zip(ts, palette[3:]):
        t["color"] = color
    inp = np.full((h, w), bg, dtype=np.int8)
    inp[:, sw] = divider
    inp[[0, -1], left_edge:] = frame
    inp[:, [left_edge, w-1]] = frame
    for t in ts:
        bh, bw = t["mask"].shape
        view = inp[t["top"]:t["top"]+bh, t["left"]:t["left"]+bw]
        assert np.all(view[t["mask"]] == bg)
        view[t["mask"]] = t["color"]
        for r in t["rows"]:
            inp[r, left_edge] = inp[r, -1] = t["color"]
        for c in t["upper"]:
            inp[0, c] = t["color"]
        for c in t["lower"]:
            inp[-1, c] = t["color"]
    # Paint labels from the sampled class records, without invoking the parser.
    out = inp.copy()
    for group, count, donors in columns:
        name = "upper" if len(group) > 1 else "lower"
        cs = [c for i in group for c in ts[i][name]]
        for i in group:
            t = ts[i]
            if not t["rows"] or not cs:
                continue
            bh, bw = t["mask"].shape
            view = out[t["top"]:t["top"]+bh, t["left"]:t["left"]+bw]
            view[t["mask"]] = bg
            offsets = np.argwhere(t["mask"])-np.array([bh//2, bw//2])
            for r in t["rows"]:
                for c in cs:
                    for dr, dc in offsets:
                        out[r+int(dr), c+int(dc)] = t["color"]
    return {"input": inp, "output": out}


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    multiplicity = int(rng.choice([3, 4]))
    train = [_make_pair(rng, "rotation", multiplicity), _make_pair(rng, "donor", multiplicity)]
    bank = [item["template"]["mask"].copy() for pair in train
            for item in _placements(_parse(pair["input"])[1]) if item["stamps"]]
    train += [_make_pair(rng, "extra", multiplicity, bank) for _ in range(nt-2)]
    tests = [_make_pair(rng, "test", multiplicity, bank) for _ in range(ne)]
    # Geometry has a large space. Regenerate actual inputs in the unlikely event
    # of duplication; colors alone are not used as a uniqueness fallback.
    seen = set()
    for position, pair in enumerate(train+tests):
        for attempt in range(20):
            if _key(pair["input"]) not in seen:
                break
            pair.update(_make_pair(rng, "test" if position >= nt else "extra", multiplicity, bank))
        else:
            raise AssertionError("distinct input budget exhausted")
        seen.add(_key(pair["input"]))
    train = [train[int(i)] for i in rng.permutation(nt)]
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


ALTERNATIVES = {
    "omit_coordinates": dict(coordinates=False),
    "omit_shape_sharing": dict(share=False),
    "ignore_top_bottom": dict(rail="both"),
    "identity_only": dict(match="identity"),
    "half_turn_only": dict(match="half_turn"),
    "include_reflections": dict(match="reflection"),
    "area_only": dict(match="area"),
    "exactly_two": dict(threshold="two"),
    "all_top": dict(rail="top"),
    "all_bottom": dict(rail="bottom"),
    "reversed_edges": dict(rail="reverse"),
}


def _features(inp):
    a, p = _parse(inp)
    items = _placements(p)
    correct = solve(a)
    groups = Counter(_canonical(t["mask"]) for t in p["records"])
    return dict(shape=a.shape, roles=(p["bg"], p["divider"], p["frame"]),
                count=len(items), classes=len(groups),
                repeated=tuple(sorted(v for v in groups.values() if v > 1)),
                masks=tuple(_mask_key(t["mask"]) for t in p["records"]),
                products=tuple((len(x["template"]["rows"]), len(x["columns"]))
                               for x in items if x["stamps"]),
                row_gaps=tuple(int(gap) for x in items if x["stamps"]
                               for gap in np.diff(x["template"]["rows"])),
                active=sum(bool(x["stamps"]) for x in items),
                borrowed=sum(bool(x["stamps"]) and x["repeated"] and
                             not x["template"]["upper"] for x in items),
                retained_donor=sum(not x["stamps"] and x["repeated"] and
                                   bool(x["template"]["upper"]) for x in items),
                diagonal=sum(not _connected(x["template"]["mask"], DIR4) for x in items),
                empty_center=sum(not t["mask"][t["mask"].shape[0]//2,
                                                   t["mask"].shape[1]//2] for t in p["records"]),
                failures={name for name, kwargs in ALTERNATIVES.items()
                          if not np.array_equal(_predict(a, **kwargs), correct)})


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and not np.array_equal(a, out)
            assert _key(a) not in seen
            seen.add(_key(a))
            _, p = _parse(a)
            items = _placements(p)
            occupied = set()
            for x in items:
                assert not occupied & x["cells"], "cross-owner stamps may not overlap"
                occupied.update(x["cells"])
                t = x["template"]
                old = out[t["cells"][:, 0], t["cells"][:, 1]]
                assert np.all(old == (p["bg"] if x["stamps"] else t["color"])), "source erasure"
                for r, c in x["cells"]:
                    assert 1 <= r < a.shape[0]-1 and p["d"]+2 <= c < a.shape[1]-1
                    assert int(out[r, c]) == t["color"], "orientation/mask/owner changed"
            # All dividers and border marks survive literally; source clearing
            # and exact union-of-stamps preserve transparent mask backgrounds.
            assert np.array_equal(a[:, p["d"]:p["d"]+2], out[:, p["d"]:p["d"]+2])
            assert np.array_equal(a[:, -1], out[:, -1])
            assert np.array_equal(a[[0, -1], p["d"]+1:], out[[0, -1], p["d"]+1:])
            assert np.array_equal(solve(a), out), "wrong output"
            features[split].append(_features(a))
    tr, te = features["train"], features["test"]
    assert set(ALTERNATIVES) <= set().union(*(f["failures"] for f in tr)), "missing distinguishing evidence"
    assert any(f["borrowed"] for f in tr) and any(f["retained_donor"] for f in tr)
    assert any(r > 1 and c > 1 for f in tr for r, c in f["products"]), "Cartesian rule not taught"
    assert max(n for f in tr for n in f["repeated"]) >= max(n for f in te for n in f["repeated"])


def _puzzle_key(p):
    return tuple((_key(x["input"]), _key(x["output"]))
                 for s in ("train", "test") for x in p[s])


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    fs = [_features(pair["input"]) for p in puzzles for s in p for pair in p[s]]
    dimensions = {f["shape"] for f in fs}
    masks = {m for f in fs for m in f["masks"]}
    products = {x for f in fs for x in f["products"]}
    profiles = {(f["count"], f["active"], f["repeated"]) for f in fs}
    assert len(dimensions) >= 30 and len(masks) >= 60 and len(profiles) >= 12
    assert {(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)} <= products
    assert {2, 3, 4} <= {n for f in fs for n in f["repeated"]}
    assert {f["roles"][0] for f in fs} == set(range(10))
    assert any(f["diagonal"] for f in fs) and any(f["empty_center"] for f in fs)
    assert any(5 in m[0] for m in masks) and any(min(m[0]) == 1 for m in masks)
    assert {1, 2, 3} <= {gap for f in fs for gap in f["row_gaps"]}
    assert all(len({_features(x["input"])["masks"] for s in p for x in p[s]}) > 1 for p in puzzles)
    # Color equivariance includes black as a source and as a structural color.
    pair = puzzles[0]["test"][0]
    for shift in range(10):
        assert np.array_equal(solve((pair["input"]+shift)%10), (pair["output"]+shift)%10)
    # A hand-built family member extends beyond sampled row/column layouts:
    # four donated columns, row gap eight, a rotated rectangular partner,
    # a retained singleton inside the active source's box, and an absent label.
    a = np.zeros((15, 30), dtype=np.int8)
    a[:, 8] = 8
    a[[0, -1], 9:] = 9
    a[:, [9, 29]] = 9
    ring = np.ones((3, 5), dtype=bool)
    ring[1, 1:-1] = False
    a[1:4, 1:6][ring] = 2
    a[6:11, 3:6][np.rot90(ring)] = 4
    a[2, 3] = 6
    for r in (3, 11):
        a[r, 9] = a[r, 29] = 2
    for c in (12, 16, 21, 26):
        a[0, c] = 4
    a[-1, 19] = 2
    a[0, 18] = 7  # No source of color 7: do not donate this column.
    expected = a.copy()
    expected[1:4, 1:6][ring] = 0
    for r in (3, 11):
        for c in (12, 16, 21, 26):
            expected[r-1:r+2, c-2:c+3][ring] = 2
    assert np.array_equal(solve(a), expected)
    assert expected[2, 3] == 6 and np.array_equal(expected[6:11, 3:6], a[6:11, 3:6])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(900+nt*5+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Every diagnostic alternative must be rejected as a replacement label.
    for name, kwargs in ALTERNATIVES.items():
        p = {s: [{k: v.copy() for k, v in x.items()} for x in puzzles[0][s]] for s in puzzles[0]}
        pair = next(x for x in p["train"] if name in _features(x["input"])["failures"])
        pair["output"] = _predict(pair["input"], **kwargs)
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepts {name}")
    return dict(seeds=42, pairs=len(fs), dimensions=len(dimensions),
                oriented_masks=len(masks), class_profiles=len(profiles),
                coordinate_products=sorted(products), shortcuts_rejected=sorted(ALTERNATIVES))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
