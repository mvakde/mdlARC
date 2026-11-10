"""Print-and-release scaffolds: anchored brick substitution with rigid-stick gravity.

Only NumPy and the standard library are used. All solving conventions are
recoverable from the input. Generation metadata never reaches solve().
"""
from collections import deque
import numpy as np


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(grid, bg, start_row=2):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)
                 if r >= start_row}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        color = int(grid[p])
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining and int(grid[q]) == color:
                    remaining.remove(q)
                    stack.append(q)
        result.append((color, frozenset(cells)))
    return result


def _cells(p):
    o, r, c, n = p[:4]
    return frozenset((r + (k if o == "V" else 0),
                      c + (k if o == "H" else 0)) for k in range(n))


def _shift(p, d):
    return (p[0], p[1] + d, *p[2:])


def _settle_cells(bodies, height):
    """Greatest simultaneously movable set; floor contact propagates upwards.

    A body may move through its own former cells and alongside moving bodies.
    Contact with any stationary body blocks it. No overlaps or sliding joints.
    """
    bodies = list(bodies)
    distances = [0] * len(bodies)
    for _ in range(height + 1):
        owner = {}
        for i, cells in enumerate(bodies):
            for cell in cells:
                if cell in owner:
                    raise ValueError("overlapping mechanical parts")
                owner[cell] = i
        dependents = [set() for _ in bodies]
        stopped = set()
        for i, cells in enumerate(bodies):
            for r, c in cells:
                if r + 1 >= height:
                    stopped.add(i)
                j = owner.get((r + 1, c))
                if j is not None and j != i:
                    dependents[j].add(i)
        queue = deque(sorted(stopped))
        while queue:
            for i in sorted(dependents[queue.popleft()]):
                if i not in stopped:
                    stopped.add(i)
                    queue.append(i)
        moving = set(range(len(bodies))) - stopped
        if not moving:
            return bodies, distances
        for i in moving:
            bodies[i] = frozenset((r + 1, c) for r, c in bodies[i])
            distances[i] += 1
    raise AssertionError("gravity did not converge")


def _settle(rods, height):
    _, distances = _settle_cells([_cells(p) for p in rods], height)
    return [_shift(p, d) for p, d in zip(rods, distances)], distances


def _decode(input_grid):
    a = _grid(input_grid)
    if a.shape[0] < 5 or not np.all(a[1] == a[1, 0]):
        raise ValueError("the second row must be a blank separator")
    bg = int(a[1, 0])
    controls = tuple(int(v) for v in a[0] if v != bg)
    if not 1 <= len(controls) <= 3 or len(set(controls)) != len(controls):
        raise ValueError("expected one to three distinct control swatches")
    components = _components(a, bg)
    dots = [(color, next(iter(cells))) for color, cells in components if len(cells) == 1]
    chunks = [(color, cells) for color, cells in components if len(cells) > 1]
    if len(dots) < 3 or len(chunks) < 2:
        raise ValueError("expected a prototype and a placement map")
    colors = {color for color, _ in chunks}
    if len(colors) != 2 or {color for color, _ in dots} != colors:
        raise ValueError("the prototype and map must use the same two materials")
    unique = [color for color in colors if sum(v == color for v, _ in dots) == 1]
    if len(unique) != 1:
        raise ValueError("the map needs exactly one exceptional point")
    primary = unique[0]
    secondary = next(v for v in colors if v != primary)
    anchor_pixel = next(p for color, p in dots if color == primary)
    prototype_cells = set().union(*(cells for _, cells in chunks))
    tr, tc = min(r for r, c in prototype_cells), min(c for r, c in prototype_cells)
    br, bc = max(r for r, c in prototype_cells), max(c for r, c in prototype_cells)
    h, w = br - tr + 1, bc - tc + 1
    if h < 2 or w < 2 or w % 2:
        raise ValueError("prototype width must be even and both dimensions at least two")
    rods = []
    for color, cells in chunks:
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        if len(rows) != 1 and len(cols) != 1:
            raise ValueError("each monochrome prototype component must be a straight stick")
        o = "H" if len(rows) == 1 else "V"
        p = (o, min(rows) - tr, min(cols) - tc, len(cells), color)
        if _cells(p) != frozenset((r-tr, c-tc) for r, c in cells):
            raise ValueError("prototype sticks must be continuous")
        rods.append(p)
    rods.sort()
    points = [p for _, p in dots]
    pr = min(r for r, c in points)
    if any((r-pr) % 2 for r, c in points):
        raise ValueError("map rows must lie on the pitch-two lattice")
    # Removing the visible alternate-row stagger leaves regular pitch two.
    pc = min(c - ((r-pr)//2 % 2) for r, c in points)
    sites = []
    for color, (r, c) in dots:
        row = (r-pr)//2
        x = c - pc - row % 2
        if x < 0 or x % 2:
            raise ValueError("invalid brick lattice")
        sites.append((row, x//2))
    sites = tuple(sorted(sites))
    ar = (anchor_pixel[0]-pr)//2
    ac = (anchor_pixel[1]-pc-ar % 2)//2
    map_h, map_w = max(r for r, c in sites)+1, max(c for r, c in sites)+1
    er, ec = max(r for r, c in points), max(c for r, c in points)
    pl = min(c for r, c in points)
    if not (br + 1 < pr or er + 1 < tr or bc + 1 < pl or ec + 1 < tc):
        raise ValueError("map and prototype boxes need a blank separating line")
    meta = dict(grid=a, bg=bg, controls=controls, primary=primary,
                secondary=secondary, rods=tuple(rods), h=h, w=w,
                tr=tr, tc=tc, sites=sites, anchor=(ar, ac),
                map_h=map_h, map_w=map_w)
    for p in _expand(meta):
        if any(not (2 <= r < a.shape[0] and 0 <= c < a.shape[1]) for r, c in _cells(p)):
            raise ValueError("printed scaffold must fit below the control band")
    return meta


def _expand(meta, swap=True, stagger=True):
    ar, ac = meta["anchor"]
    result = []
    for row, col in meta["sites"]:
        y = meta["tr"] + (row-ar)*meta["h"]
        x = meta["tc"] + (col-ac)*meta["w"]
        if stagger:
            x += (row % 2 - ar % 2)*(meta["w"]//2)
        for o, r, c, n, color in meta["rods"]:
            if swap and (row, col) == (ar, ac):
                color = meta["secondary"] if color == meta["primary"] else meta["primary"]
            result.append((o, y+r, x+c, n, color, (row, col)))
    return result


def _render(meta, rods, clip=False):
    out = np.full_like(meta["grid"], meta["bg"])
    out[0] = meta["grid"][0]
    occupied = set()
    for p in rods:
        for r, c in _cells(p):
            if clip and not (2 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                continue
            if not (2 <= r < out.shape[0] and 0 <= c < out.shape[1]) or (r, c) in occupied:
                raise AssertionError("invalid rendered scaffold")
            occupied.add((r, c))
            out[r, c] = p[4]
    return out


def _result(meta, swap=True, stagger=True, delete=True, fall=True):
    rods = [p for p in _expand(meta, swap, stagger)
            if not delete or p[4] not in meta["controls"]]
    settled, distances = _settle(rods, meta["grid"].shape[0]) if fall else (rods, [0]*len(rods))
    return _render(meta, settled, clip=not stagger), rods, settled, distances


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the prototype, map, material exchange and commands from pixels."""
    return _result(_decode(input_grid))[0]


def _grouped_answer(meta, by_site=False):
    rods = [p for p in _expand(meta) if p[4] not in meta["controls"]]
    if by_site:
        groups = []
        for site in meta["sites"]:
            group = [p for p in rods if p[5] == site]
            if group:
                groups.append(group)
    else:
        painted = _render(meta, rods)
        chunks = _components(painted, meta["bg"])
        groups = [[p for p in rods if _cells(p) <= cells] for _, cells in chunks]
    bodies = [frozenset().union(*(_cells(p) for p in group)) for group in groups]
    _, distances = _settle_cells(bodies, meta["grid"].shape[0])
    return _render(meta, [_shift(p, d) for group, d in zip(groups, distances) for p in group])


def _metrics(meta):
    answer, survivors, final, distances = _result(meta)
    owner = {q: i for i, p in enumerate(final) for q in _cells(p)}
    contacts = {(i, owner[(r+1, c)]) for i, p in enumerate(final)
                for r, c in _cells(p) if (r+1, c) in owner and owner[(r+1, c)] != i}
    cross = sum(final[i][5] != final[j][5] and final[i][2] != final[j][2]
                and distances[i] > 0 for i, j in contacts)
    occupied = set(meta["sites"])
    witnesses = (any((r, c+1) in occupied for r, c in occupied),
                 any((r+1, c) in occupied for r, c in occupied))
    return dict(moved=sum(d > 0 for d in distances),
                distances=tuple(sorted(set(distances))), cross_support=cross,
                swap_matters=not np.array_equal(answer, _result(meta, swap=False)[0]),
                deletion_matters=not np.array_equal(answer, _result(meta, delete=False)[0]),
                fall_matters=not np.array_equal(answer, _result(meta, fall=False)[0]),
                stagger_matters=not np.array_equal(answer, _result(meta, stagger=False)[0]),
                fusion_matters=not np.array_equal(answer, _grouped_answer(meta)),
                tile_rigidity_matters=not np.array_equal(answer, _grouped_answer(meta, True)),
                witnesses=witnesses, pieces=len(meta["rods"]), sites=len(meta["sites"]))


def _prototype(rng, h, w, primary, secondary, teaching=False):
    # Full-span shelf + independently placed/lengthened posts gives a tight
    # box. Interior bars may use either material: deletion is not an orientation
    # command. No same-color contacts are permitted within the prototype.
    xs = [0, w-1] if teaching and rng.random() < .55 else []
    if not xs:
        options = list(range(w))
        rng.shuffle(options)
        for x in options:
            if all(abs(x-y) >= 2 for y in xs):
                xs.append(x)
                if len(xs) >= int(rng.integers(1, min(4, w//2)+1)):
                    break
    rods = [("H", 0, 0, w, secondary)]
    for i, x in enumerate(xs):
        n = h-1 if i == 0 else int(rng.integers(2, h))
        rods.append(("V", 1, x, n, primary))

    def compatible(p):
        cells = _cells(p)
        near = cells | {(r+dr, c+dc) for r, c in cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
        return not any(cells & _cells(q) or
                       (p[4] == q[4] and near & _cells(q)) for q in rods)

    candidates = [("H", r, c, n, color) for r in range(1, h)
                  for n in range(2, w+1) for c in range(w-n+1)
                  for color in (primary, secondary)]
    candidates += [("V", r, c, n, color) for c in range(w)
                   for n in range(2, h) for r in range(1, h-n+1)
                   for color in (primary, secondary)]
    if teaching:
        extra = [p for p in candidates if p[0] == "H" and compatible(p)]
        if not extra:
            # Sparse end posts always leave room for a visibly separate shelf.
            rods = [("H", 0, 0, w, secondary), ("V", 1, 0, h-1, primary),
                    ("V", 1, w-1, h-1, primary)]
            extra = [p for p in candidates if p[0] == "H" and compatible(p)]
        rods.append(_pick(rng, extra))
    rng.shuffle(candidates)
    target = len(rods) + int(rng.integers(0, 4))
    for p in candidates:
        if len(rods) >= target:
            break
        if compatible(p):
            rods.append(p)
    if rng.integers(2):
        rods = [(o, r, c, n, secondary if color == primary else primary)
                for o, r, c, n, color in rods]
    if rng.integers(2):
        rods = [(o, h-1-r-(n-1 if o == "V" else 0), c, n, color)
                for o, r, c, n, color in rods]
    if rng.integers(2):
        rods = [(o, r, w-1-c-(n-1 if o == "H" else 0), n, color)
                for o, r, c, n, color in rods]
    return tuple(sorted(rods))


def _layout(rng, rows, cols, network=False, previous_anchor=None):
    density = float(rng.uniform(.35, .95))
    sites = {(r, c) for r in range(rows) for c in range(cols) if rng.random() < density}
    # Direct witnesses distinguish half-width staggering from rectangular tiles.
    r, c = int(rng.integers(rows)), int(rng.integers(cols-1))
    sites.update(((r, c), (r, c+1)))
    r, c = int(rng.integers(rows-1)), int(rng.integers(cols))
    sites.update(((r, c), (r+1, c)))
    # Empty sites are common, but no wholly empty internal band is introduced
    # only in a larger test. The semantic decoder also handles such gaps.
    sites.update((r, int(rng.integers(cols))) for r in range(rows))
    sites.update((int(rng.integers(rows)), c) for c in range(cols))
    if network:
        sites.update((0, c) for c in range(cols))
        anchor = (rows-1, int(_pick(rng, (0, cols-1))))
        sites.add(anchor)
    else:
        choices = sorted(p for p in sites if previous_anchor is None or
                         (p[0] != previous_anchor[0] and p[1] != previous_anchor[1]))
        if not choices:
            anchor = next((r, c) for r in range(rows) for c in range(cols)
                          if r != previous_anchor[0] and c != previous_anchor[1])
            sites.add(anchor)
        else:
            anchor = _pick(rng, choices)
    return tuple(sorted(sites)), anchor


def _place(rng, h, w, rods, sites, anchor, bg, primary, secondary, controls):
    xs = [c*w + (r % 2)*(w//2) for r, c in sites]
    minx, maxx = min(xs), max(xs)+w
    map_rows = max(r for r, c in sites)+1
    raw_xs = [2*c+r % 2 for r, c in sites]
    map_span_h = 2*map_rows-1
    map_span_w = max(raw_xs)-min(raw_xs)+1
    min_h = max(2 + h*map_rows, h+map_span_h+3)
    min_w = max(maxx-minx, w+map_span_w+1)
    if min_h > 30 or min_w > 30:
        raise ValueError("geometry does not fit")
    gh = int(rng.integers(min_h, min(30, min_h+5)+1))
    gw = int(rng.integers(min_w, min(30, min_w+6)+1))
    top = int(rng.integers(2, gh-h*map_rows+1))
    left = int(rng.integers(gw-(maxx-minx)+1))-minx
    tr = top+anchor[0]*h
    tc = left+anchor[1]*w+(anchor[0] % 2)*(w//2)
    positions = [(r, c) for r in range(2, gh-map_span_h+1)
                 for c in range(gw-map_span_w+1)
                 if (r+map_span_h < tr or tr+h < r
                     or c+map_span_w < tc or tc+w < c)]
    if not positions:
        raise ValueError("no separated map position")
    pr, pl = _pick(rng, positions)
    pc = pl-min(raw_xs)
    a = np.full((gh, gw), bg, dtype=np.int8)
    locations = rng.choice(gw, len(controls), replace=False)
    for c, color in zip(locations, controls):
        a[0, int(c)] = color
    for o, r, c, n, color in rods:
        for y, x in _cells((o, tr+r, tc+c, n, color)):
            a[y, x] = color
    for r, c in sites:
        a[pr+2*r, pc+2*c+r % 2] = primary if (r, c) == anchor else secondary
    return a


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    # A larger horizontal or vertical extent is the puzzle's sampled shift.
    shift_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    previous = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = split == "train" and index < 2
            for attempt in range(200):
                rows = int(rng.integers(2, 4))
                cols = 3 if teaching and index == 0 else int(rng.integers(2, 4))
                if split == "test":
                    if shift_axis == 0:
                        rows = int(rng.integers(4, 6))
                    else:
                        cols = int(rng.integers(4, 6))
                dims = [(h, w) for h in range(3, 8) for w in (4, 6, 8)
                        if h*rows+2 <= 30 and w*cols+w//2 <= 30
                        and (not teaching or index != 1 or
                             (h != previous["h"] and w != previous["w"]))]
                h, w = _pick(rng, dims)
                bg, primary, secondary, absent = map(int, rng.choice(10, 4, replace=False))
                rods = _prototype(rng, h, w, primary, secondary, teaching)
                sites, anchor = _layout(rng, rows, cols, teaching and index == 0,
                                        previous["anchor"] if teaching and index == 1 else None)
                if teaching:
                    controls = [primary if index == 0 else secondary]
                elif split == "test" and index == 0:
                    controls = [int(_pick(rng, (primary, secondary)))]
                else:
                    mode = int(rng.choice(5, p=(.35, .35, .12, .10, .08)))
                    controls = ([primary], [secondary], [absent], [primary, secondary],
                                [int(_pick(rng, (primary, secondary))), absent])[mode]
                try:
                    inp = _place(rng, h, w, rods, sites, anchor, bg, primary, secondary, controls)
                    meta = _decode(inp)
                except ValueError:
                    continue
                if _key(inp) in seen:
                    continue
                if teaching:
                    f = _metrics(meta)
                    if index == 1 and not any(len({p[0] for p in meta["rods"] if p[4] == color}) == 2
                                              for color in (primary, secondary)):
                        continue
                    if not all(f[k] for k in ("swap_matters", "deletion_matters", "fall_matters",
                                              "stagger_matters", "tile_rigidity_matters")):
                        continue
                    if index == 0 and not (f["cross_support"] and f["fusion_matters"]):
                        continue
                if split == "test" and index == 0:
                    f = _metrics(meta)
                    if not all(f[k] for k in ("cross_support", "swap_matters", "deletion_matters", "fall_matters")):
                        continue
                pair = {"input": inp, "output": solve(inp)}
                puzzle[split].append(pair)
                seen.add(_key(inp))
                if teaching and index == 0:
                    previous = meta
                break
            else:
                raise AssertionError("bounded blueprint construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train, test = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and np.array_equal(inp[:2], out[:2])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            meta = _decode(inp)
            answer, survivors, settled, distances = _result(meta)
            assert np.array_equal(answer, out)
            assert len(survivors) == sum(p[4] not in meta["controls"] for p in _expand(meta))
            assert all(d >= 0 for d in distances)
            assert all(a[0] == b[0] and a[2:] == b[2:] for a, b in zip(survivors, settled))
            assert _settle(settled, inp.shape[0])[1] == [0]*len(settled)
            assert np.count_nonzero(out[2:] != meta["bg"]) == sum(p[3] for p in survivors)
            assert len(meta["sites"]) == len(set(meta["sites"]))
            assert {"H", "V"} <= {p[0] for p in meta["rods"]}
            assert len(_expand(meta)) == len(meta["sites"])*len(meta["rods"])
            assert {r for r, c in meta["sites"]} == set(range(meta["map_h"]))
            assert {c for r, c in meta["sites"]} == set(range(meta["map_w"]))
            if split == "train":
                train.append((meta, _metrics(meta)))
            else:
                test.append(meta)
    witnesses = [m for m, f in train if all(f[k] for k in
                 ("swap_matters", "deletion_matters", "fall_matters", "stagger_matters", "tile_rigidity_matters"))]
    assert len(witnesses) >= 2, "need two informative complementary demonstrations"
    assert {m["primary"] in m["controls"] for m in witnesses} == {True, False}
    assert any(f["cross_support"] and f["fusion_matters"] for m, f in train)
    assert len({m["anchor"][0] for m in witnesses}) >= 2
    assert len({m["anchor"][1] for m in witnesses}) >= 2
    assert len({m["h"] for m in witnesses}) >= 2 and len({m["w"] for m in witnesses}) >= 2
    assert all(f["witnesses"] == (True, True) for m, f in train)
    assert any(any(len({p[0] for p in m["rods"] if p[4] == color}) == 2
                   for color in (m["primary"], m["secondary"])) for m, f in train), \
        "training must distinguish material deletion from orientation deletion"
    assert any(all(m[axis] > max(t[axis] for t, f in train) for m in test)
               for axis in ("map_h", "map_w")), "tests must extend a demonstrated map axis"
    test_features = [_metrics(m) for m in test]
    assert any(all(f[k] for k in
                   ("cross_support", "swap_matters", "deletion_matters", "fall_matters"))
               for f in test_features), "a test must exercise the interacting partial scaffold"


def self_test() -> None:
    # Hand-calculated four-site case. The unsupported right shelf falls below
    # its touching left neighbors, which land on posts from the swapped anchor.
    inp = np.zeros((9, 16), dtype=np.int8)
    inp[0, 15] = 1
    inp[5, 2:6] = 2
    inp[6:8, 2] = inp[6:8, 5] = 1
    inp[7, 3:5] = 2
    inp[2, (8, 10, 12)] = 2
    inp[4, 9] = 1
    m = _decode(inp)
    expected_rods = [("H", 5, 0, 4, 2), ("H", 6, 1, 2, 2),
                     ("H", 5, 4, 4, 2), ("H", 6, 5, 2, 2),
                     ("H", 7, 8, 4, 2), ("H", 8, 9, 2, 2),
                     ("V", 7, 2, 2, 2), ("V", 7, 5, 2, 2)]
    assert np.array_equal(solve(inp), _render(m, expected_rods))
    assert _metrics(m)["fusion_matters"] and _metrics(m)["cross_support"]
    extra_command = inp.copy()
    extra_command[0, 0] = 3  # Absent material joins the real command.
    assert np.array_equal(solve(extra_command)[2:], solve(inp)[2:])
    extra_command[0, 1] = 2  # Both prototype materials now disappear.
    assert np.all(solve(extra_command)[2:] == 0)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprint = lambda p: tuple((_key(q["input"]), _key(q["output"]))
                                 for s in ("train", "test") for q in p[s])
    assert len({fingerprint(p) for p in puzzles}) == 42
    assert fingerprint(puzzles[17]) == fingerprint(generate_puzzle(17))
    records, masks, layouts, shapes, roles = [], set(), set(), set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _decode(pair["input"])
                f = _metrics(m)
                records.append((split, m, f))
                masks.add((m["h"], m["w"], tuple(p[:4] for p in m["rods"])))
                layouts.add((m["sites"], m["anchor"]))
                shapes.add(pair["input"].shape)
                roles.add((m["bg"], m["primary"], m["secondary"]))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert len(masks) >= 50 and len(layouts) >= 90 and len(shapes) >= 45
    assert len(roles) >= 80
    assert {m["bg"] for _, m, _ in records} == set(range(10))
    assert {4, 6, 8} <= {m["w"] for _, m, _ in records}
    assert len({m["h"] for _, m, _ in records}) >= 5
    assert len({f["pieces"] for _, _, f in records}) >= 4
    assert sum(any(len({p[0] for p in m["rods"] if p[4] == color}) == 2
                   for color in (m["primary"], m["secondary"]))
               for _, m, _ in records) >= 40
    assert len({f["sites"] for _, _, f in records}) >= 10
    assert len({m["anchor"] for _, m, _ in records}) >= 12
    assert sum(f["cross_support"] > 0 for _, _, f in records) >= 42
    assert sum(len(f["distances"]) >= 3 for _, _, f in records) >= 15
    assert any(set(m["controls"]) == {m["primary"], m["secondary"]} for _, m, _ in records)
    assert any(not set(m["controls"]) & {m["primary"], m["secondary"]} for _, m, _ in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(1000+10*nt+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["train"][0]["output"][-1, 0] = (int(damaged["train"][0]["output"][-1, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus legal-count coverage)")
