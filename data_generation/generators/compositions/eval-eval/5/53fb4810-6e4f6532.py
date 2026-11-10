"""Phase-locked conveyor: registration + compass + periodic-field matching.

Only standard-library modules and NumPy are used. Construction paints a known
scene, while solve enumerates placements using only the given input pixels.
"""
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # north east south west
HOST_SEEDS = (0, 72630870, 839512917, 542178547, 639138116, 739480130,
              805297073, 931222748, 452208113, 372471359, 312095789,
              676908011, 624970828, 761082796, 884052366, 633822586,
              461690390, 862271376, 670552747, 859737286, 219945719,
              960879967, 588843106, 271076895, 699893134, 64507471,
              337128477, 940381233, 131370090, 782352984, 660742559,
              303649626, 698945510, 513989322, 80996919, 216163449,
              328058730, 976396793, 674742907, 95039820, 614987738, 464122400)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(points, diagonal=True):
    todo, result = set(points), []
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    while todo:
        p = min(todo)
        todo.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _transform(cells, t):
    moved = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        moved.append((r, c, v))
    r0, c0 = min(r for r, c, v in moved), min(c for r, c, v in moved)
    return {(r-r0, c-c0): int(v) for r, c, v in moved}


def _translate(cells, origin):
    return {(r+origin[0], c+origin[1]): v for (r, c), v in cells.items()}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _hub_mask(h, w, kind, thickness=None):
    mask = np.ones((h, w), dtype=bool)
    if kind == 1:
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    elif kind == 2:
        th, tw = thickness or (h-2, 1 if w % 2 else 2)
        mask[:] = False
        mask[(h-th)//2:(h+th)//2, :] = True
        mask[:, (w-tw)//2:(w+tw)//2] = True
    return mask


def _slice(box, direction, width, distance):
    r0, c0, r1, c1 = box
    dr, dc = DIRS[direction]
    if dr:
        row = r0-distance if dr < 0 else r1-1+distance
        start = c0 + (c1-c0-width)//2
        return [(row, start+j) for j in range(width)]
    col = c0-distance if dc < 0 else c1-1+distance
    start = r0 + (r1-r0-width)//2
    return [(start+j, col) for j in range(width)]


def _max_distance(shape, box, direction):
    r0, c0, r1, c1 = box
    return (r0-1, shape[1]-1-c1, shape[0]-1-r1, c0-1)[direction]


def _parse_emitter(cells, shape):
    """Recognize a uniform symmetric hub and its one centered rectangular word."""
    found = []
    for body in sorted(set(cells.values())):
        hubs = {p for p, v in cells.items() if v == body}
        r0, c0, r1, c1 = box = _box(hubs)
        h, w = r1-r0, c1-c0
        if not (3 <= h <= 7 and 3 <= w <= 7):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in hubs:
            mask[r-r0, c-c0] = True
        supported = [_hub_mask(h, w, k) for k in (0, 1)]
        supported.extend(_hub_mask(h, w, 2, (th, tw))
                         for th in range(1, h-1) for tw in range(1, w-1)
                         if (h-th) % 2 == 0 and (w-tw) % 2 == 0)
        if not any(np.array_equal(mask, possible) for possible in supported):
            continue
        rest = set(cells)-hubs
        if not rest:
            continue
        for d in range(4):
            transverse = w if d in (0, 2) else h
            maximum = _max_distance(shape, box, d)
            for width in range(1, transverse+1):
                if (transverse-width) % 2 or maximum < 2:
                    continue
                word, claimed = [], set()
                for distance in range(1, maximum+1):
                    ps = _slice(box, d, width, distance)
                    if all(p in rest for p in ps):
                        word.append(tuple(cells[p] for p in ps))
                        claimed.update(ps)
                    else:
                        break
                if claimed != rest or not (1 <= len(word) < maximum):
                    continue
                dr, dc = DIRS[d]
                if not all((r-dr, c-dc) in hubs for r, c in _slice(box, d, width, 1)):
                    continue
                field = {p: word[(distance-1) % len(word)][j]
                         for distance in range(1, maximum+1)
                         for j, p in enumerate(_slice(box, d, width, distance))}
                future = {p for distance in range(len(word)+1, maximum+1)
                          for p in _slice(box, d, width, distance)}
                found.append(dict(body=body, hub=hubs, box=box, direction=d,
                                  width=width, word=word, maximum=maximum,
                                  field=field, future=future, cells=cells))
    assert len(found) == 1, "emitter must have one geometric interpretation"
    return found[0]


def _labels(cells, bands):
    core = {p for p, v in cells.items() if v not in bands}
    assert core
    r0, c0, r1, c1 = _box(core)
    directions = {}
    for p, v in cells.items():
        if v not in bands:
            continue
        r, c = p
        ds = [d for d, ok in enumerate((r < r0, c >= c1, r >= r1, c < c0)) if ok]
        assert len(ds) == 1, "tabs must project from exactly one core side"
        directions[p] = ds[0]
    assert len(set(directions.values())) >= 2
    assert any((d+1) % 4 in directions.values() for d in directions.values())
    return directions


def _parse(inp):
    h, w = inp.shape
    assert h >= 9 and w >= 9
    bg = int(inp[0, 0])
    assert all(int(inp[p]) == bg for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))
    sides = [int(inp[0, 1]), int(inp[1, -1]), int(inp[-1, 1]), int(inp[1, 0])]
    assert len(set(sides+[bg])) == 5
    assert np.all(inp[0, 1:-1] == sides[0]) and np.all(inp[-1, 1:-1] == sides[2])
    assert np.all(inp[1:-1, 0] == sides[3]) and np.all(inp[1:-1, -1] == sides[1])
    interior = {(r, c) for r in range(1, h-1) for c in range(1, w-1) if inp[r, c] != bg}
    groups = _components(interior)
    targets, sources, emitters = [], [], []
    for group in groups:
        cells = {p: int(inp[p]) for p in group}
        colors = set(cells.values())
        if len(colors) == 1:
            targets.append(group)
        elif colors & set(sides):
            sources.append(cells)
        else:
            emitters.append(cells)
    assert len(emitters) == 1 and 1 <= len(sources) == len(targets) <= 3
    reg_colors = {int(inp[next(iter(t))]) for t in targets}
    assert len(reg_colors) == 1
    reg = reg_colors.pop()
    emitter = _parse_emitter(emitters[0], inp.shape)
    body = emitter["body"]
    payload = set(emitter["cells"].values())-{body}
    assert len({bg, body, reg, *sides}) == 7
    assert payload and payload.isdisjoint({bg, body, reg, *sides})
    for source in sources:
        values = set(source.values())
        assert body in values and reg in values and values & payload
        assert values <= {body, reg, *sides, *payload}
        _labels(source, set(sides))
        assert len(_components(source, diagonal=False)) == 1
        marks = {p for p, v in source.items() if v == reg}
        assert len(_components(marks)) == 1
    return dict(background=bg, sides=sides, bands=set(sides), body=body,
                registration=reg, payload=payload, targets=targets,
                sources=sources, emitter=emitter, shape=inp.shape)


def _candidates(source, info, use_phase=True, use_compass=True):
    result, seen = [], set()
    e, reg = info["emitter"], info["registration"]
    h, w = info["shape"]
    for t in range(8):
        moved = _transform(source, t)
        marks = {p for p, v in moved.items() if v == reg}
        labels = _labels(moved, info["bands"])
        if use_compass and any(moved[p] != info["sides"][d] for p, d in labels.items()):
            continue
        for index, target in enumerate(info["targets"]):
            if len(marks) != len(target):
                continue
            a, b = min(marks), min(target)
            delta = b[0]-a[0], b[1]-a[1]
            if {(r+delta[0], c+delta[1]) for r, c in marks} != target:
                continue
            placed = _translate(moved, delta)
            if any(not (1 <= r < h-1 and 1 <= c < w-1) for r, c in placed):
                continue
            if placed.keys() & e["cells"].keys():
                continue
            sensors = {p: v for p, v in placed.items() if v in info["payload"]}
            if any(p not in e["future"] for p in sensors):
                continue
            if use_phase and any(e["field"][p] != v for p, v in sensors.items()):
                continue
            signature = index, tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                result.append((index, placed, t))
    return result


def _arrangements(info, use_phase=True, use_compass=True, limit=2):
    options = [_candidates(s, info, use_phase, use_compass) for s in info["sources"]]
    order = sorted(range(len(options)), key=lambda i: len(options[i]))
    solutions = {}

    def search(depth, used, occupied, placed):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            sig = tuple(sorted(occupied.items()))
            solutions[sig] = list(placed)
            return
        i = order[depth]
        for target, cells, transform in options[i]:
            if target in used or occupied.keys() & cells.keys():
                continue
            search(depth+1, used | {target}, {**occupied, **cells},
                   placed+[(i, target, cells, transform)])
    search(0, set(), {}, [])
    return list(solutions.values())


def _render(inp, info, arrangement, extend=True):
    out = np.full_like(inp, info["background"])
    out[0, :] = inp[0, :]
    out[-1, :] = inp[-1, :]
    out[:, 0] = inp[:, 0]
    out[:, -1] = inp[:, -1]
    e = info["emitter"]
    for p, v in e["cells"].items():
        out[p] = v
    if extend:
        for p, v in e["field"].items():
            out[p] = v
    for i, target, cells, transform in arrangement:
        for p, v in cells.items():
            out[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    info = _parse(inp)
    arrangements = _arrangements(info)
    assert len(arrangements) == 1, "phase, stencil and compass must determine one pixel placement"
    return _render(inp, info, arrangements[0])


def _make_core(rng, height, width, body, reg, marks, sensors, sides, gap=None):
    core = {(r, c): body for r in range(height) for c in range(width)}
    # Random clipped corners/indentations, while retaining every edge extent
    # and all marks/sensors. This never fills gaps at solution time.
    for p in ((0, 0), (0, width-1), (height-1, 0), (height-1, width-1)):
        if p not in marks and p not in sensors and rng.random() < .65:
            core.pop(p)
    core.update({p: reg for p in marks})
    core.update(sensors)
    if gap is not None:
        assert core[gap] == body
        core.pop(gap)
    d = int(rng.integers(4))
    selected = {d, (d+1) % 4}
    for j in range(4):
        if rng.random() < .20:
            selected.add(j)
    for d in sorted(selected):
        extent = width if d in (0, 2) else height
        positions = [int(rng.integers(1, extent-1))]
        if extent >= 4 and rng.random() < .3:
            positions = sorted(set(positions+[int(rng.integers(1, extent-1))]))
        for x in positions:
            p = ((-1, x), (x, width), (height, x), (x, -1))[d]
            core[p] = sides[d]
    # Thin the body without destroying the visible envelope or connectivity.
    # This creates open notches, narrow supports and occasional internal holes;
    # the registration shape does not select a body template.
    removable = [p for p, v in core.items() if v == body]
    rng.shuffle(removable)
    desired = int(height*width*float(rng.uniform(.48, 1.0)))
    for p in removable:
        interior = {q for q, v in core.items() if v not in sides}
        if len(interior) <= desired or sum(v == body for v in core.values()) <= 3:
            break
        trial = dict(core)
        trial.pop(p)
        trial_core = {q for q, v in trial.items() if v not in sides}
        if (_box(trial_core) == (0, 0, height, width)
                and len(_components(trial, diagonal=False)) == 1):
            core = trial
    return core


def _construct(rng, width, period, count, witness=None, generous=False):
    palette = [int(v) for v in rng.permutation(10)]
    bg, body, reg = palette[:3]
    sides, payload = palette[3:7], palette[7:]
    # Sensor colors are read from the seed's projected field, so each occurs
    # in the visible word; there is no hidden payload palette dictionary.
    h = int(rng.integers(16 if count > 1 else 12, 31))
    w = 30 if generous else int(rng.integers(26 if count == 3 else 22 if count == 2 else 17, 31))
    base = np.full((h, w), bg, dtype=np.int8)
    base[0, 1:-1], base[1:-1, -1] = sides[0], sides[1]
    base[-1, 1:-1], base[1:-1, 0] = sides[2], sides[3]
    hub_h, hub_w = width+2, int(rng.choice([3, 3, 5]))
    center = int(rng.integers(4, h-4))
    hub_r, hub_c = center-hub_h//2, int(rng.integers(1, 3))
    box = hub_r, hub_c, hub_r+hub_h, hub_c+hub_w
    mask = _hub_mask(hub_h, hub_w, int(rng.integers(3)))
    hub = {(hub_r+int(r), hub_c+int(c)): body for r, c in zip(*np.nonzero(mask))}
    if witness == "word":
        a = np.array([payload[j % 3] for j in range(width)], dtype=np.int8)
        b = np.array([payload[(j+1) % 3] for j in range(width)], dtype=np.int8)
        tile = np.array([a, b, a], dtype=np.int8)
    else:
        tile = rng.choice(payload, size=(period, width)).astype(np.int8)
        if period > 1:
            # Force a primitive word without requiring every cell to differ.
            tile[:, 0] = payload[0]
            tile[-1, 0] = payload[1]
    copies = 1 if witness or count > 1 else int(rng.choice([1, 1, 2]))
    word = tile[np.arange(len(tile)*copies) % len(tile)]
    maximum = _max_distance(base.shape, box, 1)
    emitter = dict(hub=hub, cells=dict(hub), field={}, future=set())
    for distance in range(1, maximum+1):
        for j, p in enumerate(_slice(box, 1, width, distance)):
            emitter["field"][p] = int(word[(distance-1) % len(word), j])
            if distance <= len(word):
                emitter["cells"][p] = emitter["field"][p]
            else:
                emitter["future"].add(p)
    finals, blocked = [], _halo(emitter["cells"])
    same_marks = {(1, 1)} if witness == "phase" else None
    for i in range(count):
        found = False
        for attempt in range(80):
            ch = int(rng.choice([3, 3, 4, 5]))
            cw = 5 if witness == "word" else 3 if count >= 2 else int(rng.choice([3, 4, 5]))
            mr = ch//2
            marks = same_marks or {(mr, cw//2)}
            if witness == "word":
                marks = {(mr, 2), (mr, 3)}
            if not witness and rng.random() < .6:
                choices = ({(mr-1, 1), (mr, 1)}, {(mr-1, 0), (mr, 1)},
                           {(mr-1, 1), (mr-1, 2), (mr, 1)},
                           {(mr-1, 0), (mr-1, 1), (mr-1, 2), (mr, 1)})
                marks = choices[int(rng.integers(len(choices)))]
            sensor_row = 1 if witness == "phase" else mr
            sensor_cols = [cw-1] if witness == "phase" else [0, cw-1]
            if not witness and width >= 2 and rng.random() < .5:
                sensor_points = [(sensor_row, cw-1), (sensor_row-1, cw-1)]
            else:
                sensor_points = [(sensor_row, c) for c in sensor_cols]
            # Pose and station origins are sampled throughout the available ray.
            row = center-sensor_row + int(rng.integers(-(width//2), width-width//2))
            low = box[3]+len(word)+2
            high = w-cw-2
            if high < low:
                continue
            col = int(rng.integers(low, high+1))
            absolute = [(row+r, col+c) for r, c in sensor_points]
            if any(p not in emitter["future"] for p in absolute):
                continue
            sensors = {p: emitter["field"][(row+p[0], col+p[1])] for p in sensor_points}
            if set(sensors) & marks:
                continue
            cells = _make_core(rng, ch, cw, body, reg, marks, sensors, sides,
                               gap=(mr, 1) if witness == "word" else None)
            placed = _translate(cells, (row, col))
            if any(not (1 <= r < h-1 and 1 <= c < w-1) for r, c in placed):
                continue
            if placed.keys() & blocked:
                continue
            # The teaching witness must put congruent stations at different phases.
            if witness == "phase" and finals:
                old = {v for v in finals[0].values() if v in payload}
                if old == set(sensors.values()):
                    continue
            finals.append(placed)
            blocked.update(_halo(placed))
            found = True
            break
        if not found:
            return None
    targets = {p: reg for final in finals for p, v in final.items() if v == reg}
    inp = base.copy()
    for p, v in {**emitter["cells"], **targets}.items():
        inp[p] = v
    input_blocked = _halo(emitter["cells"].keys() | targets.keys())
    sources = []
    for i, final in enumerate(finals):
        t = int(rng.integers(1, 4) if witness == "phase" and i == 0
                else rng.integers(4, 8) if witness == "word" and i == 0
                else rng.integers(8))
        moved = _transform(final, t)
        sh, sw = _box(moved)[2:]
        origins = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
        rng.shuffle(origins)
        for origin in origins:
            placed = _translate(moved, origin)
            if placed.keys() & input_blocked:
                continue
            sources.append(placed)
            input_blocked.update(_halo(placed))
            for p, v in placed.items():
                inp[p] = v
            break
        else:
            return None
    out = base.copy()
    for p, v in {**emitter["cells"], **emitter["field"]}.items():
        out[p] = v
    for final in finals:
        for p, v in final.items():
            out[p] = v
    # A common D4 operation changes absolute direction without changing semantics.
    t = int(rng.integers(4))
    mirror = bool(rng.integers(2))
    def orient(a):
        a = np.rot90(a, t)
        return np.ascontiguousarray(np.fliplr(a) if mirror else a)
    pair = dict(input=orient(inp), output=orient(out))
    try:
        info = _parse(pair["input"])
        arrangements = _arrangements(info)
        if len(arrangements) != 1 or not np.array_equal(_render(pair["input"], info, arrangements[0]), pair["output"]):
            return None
        if witness == "phase" and (len(_arrangements(info, use_phase=False)) != 2
                                    or len(_arrangements(info, use_compass=False)) != 2):
            return None
    except AssertionError:
        return None
    return pair


def _fallback(rng, width, period, count, witness=None):
    """Spacious finite construction for every sampled parameter combination."""
    colors = [int(v) for v in rng.permutation(10)]
    bg, body, reg = colors[:3]
    sides, payload = colors[3:7], colors[7:]
    base = np.full((30, 30), bg, dtype=np.int8)
    base[0, 1:-1], base[1:-1, -1] = sides[0], sides[1]
    base[-1, 1:-1], base[1:-1, 0] = sides[2], sides[3]
    h, center = width+2, 16
    box = center-h//2, 1, center-h//2+h, 4
    hub = {(r, c): body for r in range(box[0], box[2]) for c in range(1, 4)}
    tile = np.array([[payload[j % 3] for j in range(width)] for _ in range(period)])
    if period > 1:
        tile[-1, 0] = payload[1]
    if witness == "word":
        tile = np.array([[payload[0], payload[1]],
                         [payload[1], payload[2]], [payload[0], payload[1]]])
    field = {p: int(tile[(distance-1) % len(tile), j])
             for distance in range(1, _max_distance(base.shape, box, 1)+1)
             for j, p in enumerate(_slice(box, 1, width, distance))}
    seed = {p: field[p] for distance in range(1, len(tile)+1)
            for p in _slice(box, 1, width, distance)}
    inp, out = base.copy(), base.copy()
    for p, v in {**hub, **seed}.items():
        inp[p] = v
    for p, v in {**hub, **field}.items():
        out[p] = v
    for i in range(count):
        cw = 5 if witness == "word" else 3
        cells = {(r, c): body for r in range(3) for c in range(cw)}
        cells[(-1, 1)], cells[(1, -1)] = sides[0], sides[3]
        marks = ({(1, 2), (1, 3)} if witness == "word" else
                 {(1, 1)} if witness == "phase" or i == 0 else
                 {(r, 1) for r in range(i+1)})
        origin = (center-1, 14 if witness == "word" else 9+7*i)
        if witness == "word":
            cells.pop((1, 1))
        for p in marks:
            cells[p] = reg
        sensor_points = [(1, 0), (1, 4)] if witness == "word" else [(1, 2)]
        for p in sensor_points:
            cells[p] = field[(origin[0]+p[0], origin[1]+p[1])]
        for p, v in _translate(cells, origin).items():
            out[p] = v
            if v == reg:
                inp[p] = v
        t = 1 if witness == "phase" and i == 0 else 4 if witness == "word" else int(rng.integers(8))
        for p, v in _translate(_transform(cells, t), (3, 2+8*i)).items():
            assert inp[p] == bg
            inp[p] = v
    t = int(rng.integers(4))
    return {"input": np.ascontiguousarray(np.rot90(inp, t)),
            "output": np.ascontiguousarray(np.rot90(out, t))}


def _distinct_fallback(pair, used):
    # At most 13 inputs are excluded. The asymmetric spacious arrangement
    # supplies 40 distinct rotated/cyclically recolored candidates.
    for t in range(4):
        for shift in range(10):
            candidate = {k: np.ascontiguousarray((np.rot90(a, t)+shift) % 10)
                         for k, a in pair.items()}
            if _key(candidate["input"]) not in used:
                return candidate
    raise AssertionError("finite fallback exclusion space exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for i in range(number):
            witness = "phase" if split == "train" and i == 0 else "word" if split == "train" and i == 1 else None
            if witness == "phase":
                width, period, count = 1, 2, 2
            elif witness == "word":
                width, period, count = 2, 3, 1
            elif split == "test":
                width = 3 if regime == 0 else int(rng.integers(2, 4))
                period = 4 if regime == 1 else 3
                count = 3 if regime == 2 else int(rng.integers(1, 3))
            else:
                width = int(rng.integers(1, 3 if regime == 0 else 4))
                period = int(rng.integers(1, 4 if regime == 1 else 5))
                count = int(rng.integers(1, 3 if regime == 2 else 4))
            for attempt in range(300):
                pair = _construct(rng, width, period, count, witness, generous=attempt >= 80)
                if pair is not None and _key(pair["input"]) not in used:
                    break
            else:
                pair = _distinct_fallback(_fallback(rng, width, period, count, witness), used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out):
    info = _parse(inp)
    arrangements = _arrangements(info)
    assert len(arrangements) == 1
    arrangement = arrangements[0]
    assert np.array_equal(_render(inp, info, arrangement), out), "incorrect field or installed piece"
    e = info["emitter"]
    final_cells = {p for _, _, cells, _ in arrangement for p in cells}
    assert all(inp[p] == out[p] for p in e["cells"])
    assert all(out[p] == v for p, v in e["field"].items() if p not in final_cells)
    for _, _, cells, _ in arrangement:
        assert all(out[p] == v for p, v in cells.items())
        assert all(e["field"][p] == v for p, v in cells.items() if v in info["payload"])
    old = {p for source in info["sources"] for p in source}
    assert all(out[p] == info["background"] for p in old-e["field"].keys()-final_cells)
    assert not np.array_equal(inp, out)
    word = e["word"]
    period = next(p for p in range(1, len(word)+1) if len(word) % p == 0
                  and all(row == word[i % p] for i, row in enumerate(word)))
    prefix_periods = [p for p in range(1, len(word)) if len(word) % p
                      and all(row == word[i % p] for i, row in enumerate(word))]
    whole_witness = any(out[q] == word[(d-1) % len(word)][j]
                        and word[(d-1) % len(word)][j] != word[(d-1) % p][j]
                        for p in prefix_periods
                        for d in range(len(word)+1, e["maximum"]+1)
                        for j, q in enumerate(_slice(e["box"], e["direction"], e["width"], d))
                        if q not in final_cells)
    sensors, shapes, patterns, moves = [], [], [], []
    transparent_ribbon = False
    for i, target, cells, t in arrangement:
        r0, c0, r1, c1 = _box(cells)
        shapes.append((r1-r0, c1-c0))
        core = {p for p, v in cells.items() if v not in info["bands"]}
        a, b, c, d = _box(core)
        transparent_ribbon |= any(p in e["future"] and p not in cells and out[p] == e["field"][p]
                                  for p in ((r, c0) for r in range(a, c) for c0 in range(b, d)))
        sensors.append(sum(v in info["payload"] for v in cells.values()))
        marks = {p: 1 for p, v in cells.items() if v == info["registration"]}
        patterns.append(min(_signature(_transform(marks, j)) for j in range(8)))
        source = info["sources"][i]
        transforms = [j for j in range(8) if _signature(_transform(source, j)) == _signature(cells)]
        moves.append("translation" if 0 in transforms else "rotation" if any(j < 4 for j in transforms) else "reflection")
    return dict(shape=inp.shape, direction=e["direction"], width=e["width"],
                word_length=len(word), period=period, count=len(arrangement), background=info["background"],
                pieces=shapes, sensors=sensors, patterns=patterns, moves=moves,
                phase_needed=any(not np.array_equal(_render(inp, info, a), out)
                                 for a in _arrangements(info, use_phase=False)),
                compass_needed=any(not np.array_equal(_render(inp, info, a), out)
                                   for a in _arrangements(info, use_compass=False)),
                whole_word_witness=whole_witness,
                transparent_ribbon=transparent_ribbon,
                exposed_extension=len(e["future"]-final_cells),
                clipped=e["maximum"] % len(word) != 0,
                old_field_overlap=len(old & e["future"]),
                old_new_overlap=len(old & final_cells),
                repeated_stencil=len(set(patterns)) < len(patterns))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            records[split].append(_features(inp, out))
    ts = records["train"]
    assert any(r["phase_needed"] and r["repeated_stencil"] for r in ts)
    assert any(r["compass_needed"] for r in ts)
    assert any(r["whole_word_witness"] for r in ts)
    assert any(r["transparent_ribbon"] for r in ts), "teach that absent piece cells reveal the field"
    assert any(r["width"] >= 2 and max(r["sensors"]) >= 2 for r in ts)
    assert any("rotation" in r["moves"] for r in ts)
    assert any("reflection" in r["moves"] for r in ts)
    assert all(r["exposed_extension"] > 0 for s in records.values() for r in s)
    assert any(min(r[axis] for r in records["test"]) > max(r[axis] for r in ts)
               for axis in ("width", "period", "count")), "test must extend a taught structural axis"


def self_test():
    """Bounded actual-pixel diversity audit on the 42 supplied verification seeds."""
    records, keys = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(tuple(tuple(_key(p["input"]) for p in puzzle[s]) for s in ("train", "test")))
        for split in ("train", "test"):
            records.extend(_features(_grid(p["input"]), _grid(p["output"])) for p in puzzle[split])
    assert len(keys) == 42
    assert {r["direction"] for r in records} == set(range(4))
    assert {r["width"] for r in records} == {1, 2, 3}
    assert {r["count"] for r in records} == {1, 2, 3}
    assert {r["background"] for r in records} == set(range(10))
    assert {r["word_length"] for r in records} >= {1, 2, 3, 4}
    assert len({r["shape"] for r in records}) >= 40
    assert len({shape for r in records for shape in r["pieces"]}) >= 8
    assert len({p for r in records for p in r["patterns"]}) >= 4
    assert {m for r in records for m in r["moves"]} == {"translation", "rotation", "reflection"}
    assert any(r["clipped"] for r in records) and any(not r["clipped"] for r in records)
    assert any(r["old_field_overlap"] for r in records)
    assert any(r["old_new_overlap"] for r in records)
    # Exercise the escape path even when random packing did not need it.
    for width in (1, 2, 3):
        for period in (1, 2, 3, 4):
            for count in (1, 2, 3):
                pair = _fallback(np.random.default_rng(width*100+period*10+count), width, period, count)
                _features(pair["input"], pair["output"])
    phase = _fallback(np.random.default_rng(73), 1, 2, 2, "phase")
    word = _fallback(np.random.default_rng(74), 2, 3, 1, "word")
    phase_f, word_f = (_features(p["input"], p["output"]) for p in (phase, word))
    assert phase_f["phase_needed"] and phase_f["compass_needed"]
    assert word_f["whole_word_witness"] and word_f["transparent_ribbon"]
    assert "rotation" in phase_f["moves"] and "reflection" in word_f["moves"]
    test = _fallback(np.random.default_rng(75), 3, 4, 3)
    validate_generated_puzzle({"train": [phase, word], "test": [test]})
    candidates = [{k: np.ascontiguousarray((np.rot90(a, t)+shift) % 10)
                   for k, a in phase.items()} for t in range(4) for shift in range(10)]
    assert len({_key(p["input"]) for p in candidates}) == 40
    used = {_key(p["input"]) for p in candidates[:13]}
    repaired = _distinct_fallback(phase, used)
    assert _key(repaired["input"]) not in used
    _features(repaired["input"], repaired["output"])
    p = generate_puzzle(1234, 2, 1)
    q = generate_puzzle(1234, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(321+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    damaged = generate_puzzle(37, 2, 1)
    damaged["test"][0]["output"][0, 1] = (int(damaged["test"][0]["output"][0, 1])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return {"seeds": 42, "pairs": len(records), "shapes": len({r["shape"] for r in records})}
    raise AssertionError("damaged output accepted")
