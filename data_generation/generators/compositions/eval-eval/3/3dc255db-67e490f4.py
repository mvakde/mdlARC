"""Elected cargo streams: component plurality authorizes unary queue export.

Only the input reader is used by solve. Construction paints its own labels
from the sampled cargo records; validation recovers all records from grids.
"""
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 64028396, 466684185, 110620780, 881609373, 233218776,
              391025008, 800198165, 123553778, 819432503, 278697081, 6568636,
              448582107, 494463805, 825649908, 159528745, 370464836, 647197047,
              613032203, 787664810, 116455957, 82592018, 548576597, 505776299,
              72150482, 528835340, 802788127, 584132689, 861820808, 278445143,
              828746701, 209275648, 951882305, 733717072, 963386692, 484265561,
              601987728, 533523143, 579261311, 684721807, 667539894, 516677092)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _normalize(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


@lru_cache(maxsize=4096)
def _variants(shape, reflect=True):
    result = set()
    for mirror in range(2 if reflect else 1):
        current = tuple((r, -c if mirror else c) for r, c in shape)
        for _ in range(4):
            current = _normalize(current)
            result.add(current)
            current = tuple((c, -r) for r, c in current)
    return tuple(sorted(result))


def _signature(cells, grouping="shape"):
    raw = _normalize(cells)
    if grouping == "global":
        return 0
    if grouping == "area":
        return len(raw)
    if grouping == "box":
        return tuple(sorted(_extent(raw)))
    if grouping == "translation":
        return raw
    return min(_variants(raw, grouping != "rotation"))


def _extent(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _components(cells, diagonal=False):
    todo, result = set(cells), []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr)+abs(dc) == 1)]
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r+dr, c+dc
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _turn(point, direction):
    """Map upward coordinates into direction 0=up, 1=right, 2=down, 3=left."""
    r, c = point
    return ((r, c), (c, -r), (-r, -c), (-c, r))[direction]


def _unturn(point, direction):
    return _turn(point, (-direction) % 4)


def _envelope(cells, direction):
    """Read thin, monotone pointed sides without knowing any sampled profile."""
    local = {_unturn(p, direction) for p in cells}
    lo_r, hi_r = min(r for r, c in local), max(r for r, c in local)
    rows = [sorted(c for r, c in local if r == y) for y in range(lo_r, hi_r+1)]
    if hi_r-lo_r < 3 or len(rows[0]) != 1 or len(rows[-1]) != 2:
        return None
    tip_x = rows[0][0]
    left = right = tip_x
    hollow = set()
    for offset, row in enumerate(rows):
        if not row or len(row) > 2:
            return None
        a, b = row[0], row[-1]
        if a > left or b < right or left-a > 1 or b-right > 1:
            return None
        left, right = a, b
        hollow.update(_turn((lo_r+offset, x), direction) for x in range(a+1, b))
    if not left < tip_x < right or right-left < 4 or not hollow:
        return None
    return _turn((lo_r, tip_x), direction), hollow


def _as_grid(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a


def _read(grid):
    a = _as_grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    i = int(np.argmax(counts))
    assert counts[i] > a.size/2, "background must be the unique majority color"
    bg = int(colors[i])
    candidates = []
    for color in map(int, colors):
        if color == bg:
            continue
        for cells in _components(map(tuple, np.argwhere(a == color)), True):
            if len(cells) < 7:
                continue
            for d in range(4):
                geometry = _envelope(cells, d)
                if geometry is None:
                    continue
                tip, cavity = geometry
                if not any(int(a[p]) not in (bg, color) for p in cavity):
                    continue
                candidates.append(dict(frame=cells, direction=d, tip=tip,
                                       cavity=cavity, color=color))
    frames = [f for f in candidates if not any(f is not g and f["frame"] <= g["cavity"]
                                               for g in candidates)]
    assert frames and len({frozenset(f["frame"]) for f in frames}) == len(frames), "ambiguous frames"
    assert len({f["color"] for f in frames}) == 1, "one visible frame color per scene"
    frames.sort(key=lambda f: min(f["frame"]))
    occupied, hollows, boxes = set(), set(), set()
    for f in frames:
        rs, cs = zip(*f["frame"])
        box = {(r, c) for r in range(min(rs), max(rs)+1) for c in range(min(cs), max(cs)+1)}
        assert boxes.isdisjoint(box), "frame bounding boxes overlap"
        boxes.update(box)
        assert occupied.isdisjoint(f["frame"]) and hollows.isdisjoint(f["cavity"])
        occupied.update(f["frame"])
        hollows.update(f["cavity"])
    assert occupied.isdisjoint(hollows)
    cargoes = []
    frame_color = frames[0]["color"]
    for color in map(int, colors):
        if color in (bg, frame_color):
            continue
        for cells in _components(map(tuple, np.argwhere(a == color))):
            owners = [j for j, f in enumerate(frames) if cells <= f["cavity"]]
            assert len(owners) == 1, "each cargo must lie wholly inside one frame"
            j = owners[0]
            f = frames[j]
            dr, dc = STEPS[f["direction"]]
            leading = min(-(r-f["tip"][0])*dr-(c-f["tip"][1])*dc for r, c in cells)
            cargoes.append(dict(cells=cells, color=color, owner=j, leading=leading,
                                shape=_signature(cells), raw=_normalize(cells), area=len(cells)))
    assert cargoes
    assert occupied | set().union(*(x["cells"] for x in cargoes)) == set(map(tuple, np.argwhere(a != bg)))
    for j, f in enumerate(frames):
        f["cargoes"] = sorted((x for x in cargoes if x["owner"] == j), key=lambda x: x["leading"])
        assert f["cargoes"] and len({x["leading"] for x in f["cargoes"]}) == len(f["cargoes"]), "queue-order tie"
        # Whole nonbackground components cannot merge two cargoes at an edge.
        inside = set().union(*(x["cells"] for x in f["cargoes"]))
        assert len(_components(inside, True)) == len(f["cargoes"]), "cargoes touch"
    return dict(background=bg, frames=frames, cargoes=cargoes)


def _votes(data, grouping="shape"):
    ballots = defaultdict(Counter)
    for x in data["cargoes"]:
        ballots[_signature(x["cells"], grouping)][x["color"]] += 1
    winners = {}
    for shape, by_color in ballots.items():
        high = max(by_color.values())
        best = [c for c, n in by_color.items() if n == high]
        assert len(best) == 1, "each shape must have a unique plurality"
        winners[shape] = best[0]
    return dict(ballots), winners


def _prediction(a, data, *, grouping="shape", local=False, all_cargo=False,
                one_per_component=False, reverse=False, edge=False, color_order=False):
    _, winners = _votes(data, grouping)
    out = a.copy()
    for x in data["cargoes"]:
        for p in x["cells"]:
            out[p] = data["background"]
    paths = set()
    for f in data["frames"]:
        current = winners
        if local:
            _, current = _votes(dict(cargoes=f["cargoes"]), grouping)
        chosen = [x for x in f["cargoes"] if all_cargo or x["color"] == current[_signature(x["cells"], grouping)]]
        if reverse:
            chosen.reverse()
        if color_order:
            chosen.sort(key=lambda x: x["color"])
        stream = [x["color"] for x in chosen for _ in range(1 if one_per_component else x["area"])]
        dr, dc = STEPS[f["direction"]]
        r, c = f["tip"]
        if edge and stream:
            stream = [stream[0]] * 30
        for color in stream:
            r, c = r+dr, c+dc
            if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
                break
            p = (r, c)
            assert int(a[p]) == data["background"] and p not in paths, "stream collision"
            paths.add(p)
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _as_grid(input_grid)
    return _prediction(a, _read(a))


@lru_cache(maxsize=1)
def _pool():
    """Enumerate free polyominoes through six cells, plus long/large forms."""
    shapes, layer = {((0, 0),)}, {((0, 0),)}
    for size in range(2, 7):
        next_layer = set()
        for s in sorted(layer):
            cells = set(s)
            frontier = {(r+dr, c+dc) for r, c in cells for dr, dc in STEPS} - cells
            for p in sorted(frontier):
                candidate = _signature(cells | {p})
                if max(_extent(candidate)) <= 5:
                    next_layer.add(candidate)
        shapes.update(next_layer)
        layer = next_layer
    for h, w in ((2, 5), (3, 4), (2, 6)):
        shapes.add(_signature({(r, 0) for r in range(h)} | {(r, w-1) for r in range(h)}
                              | {(h-1, c) for c in range(w)}))
    return tuple(sorted(s for s in shapes if len(s) <= 8))


def _record(shape, color, variants=None):
    return dict(shape=shape, color=int(color), variants=variants or _variants(shape))


def _teaching_cargoes(anchor, companion, small, palette, rich):
    x, y, z = map(int, palette[:3])
    if not rich:
        # Opposite handedness wins for the anchor. Rotation-only grouping
        # would authorize the lone losing cargo as a separate class.
        reflected = _normalize((r, -c) for r, c in anchor)
        aw = lambda: _record(anchor, x, (reflected,))
        al = lambda: _record(anchor, y, (anchor,))
        bw = lambda: _record(companion, y)
        bl = lambda: _record(companion, x)
        bundles = [[aw(), bw(), al()], [bl(), aw(), bw(), _record(small, z)]]
        # Opposite authorized A/B orders reject a numeric-color sort. Losing
        # cargoes still receive independently shuffled positions among them.
        for j, bundle in enumerate(bundles):
            for record in bundle:
                if record["color"] == (x if record["shape"] == anchor else y if record["shape"] == companion else z):
                    record["priority"] = (0 if record["shape"] == (anchor if j == 0 else companion)
                                          else 1 if record["shape"] in (anchor, companion) else 2)
        return bundles
    aw = lambda: _record(anchor, y)
    a1 = lambda: _record(anchor, x)
    a2 = lambda: _record(anchor, z)
    bw = lambda: _record(companion, x)
    bl = lambda: _record(companion, y)
    # Each of the first two arrows has a local 2-to-1 winner for A that
    # loses the global 3-to-2-to-2 contest. Global pooling is observable
    # even if an alternative is granted an arbitrary local tie-break.
    return [[aw(), bw(), a1(), a1()], [aw(), a2(), a2(), _record(small, z)],
            [aw(), bw(), bl()]]


def _ordinary_cargoes(rng, shapes, palette):
    all_records = []
    winner_colors = [int(_choice(rng, palette)) for _ in shapes]
    for shape, win in zip(shapes, winner_colors):
        pattern = _choice(rng, ((1,), (2,), (2, 1), (3, 1), (3, 2), (2, 1, 1),
                                (3, 2, 1), (4, 3), (3, 2, 2)))
        losers = [int(c) for c in rng.permutation([c for c in palette if c != win])]
        for color, count in zip([win]+losers, pattern):
            all_records.extend(_record(shape, color) for _ in range(count))
    if len(all_records) > 12:
        # A bounded physical-load convention, not a different election rule.
        all_records = [_record(shape, win) for shape, win in zip(shapes, winner_colors)]
        all_records += [_record(shapes[0], winner_colors[0])]
    nframes = int(rng.integers(max(1, (len(all_records)+3)//4), min(3, len(all_records))+1))
    bundles = [[] for _ in range(nframes)]
    for index, j in enumerate(rng.permutation(len(all_records))):
        bundles[index % nframes].append(all_records[int(j)])
    return bundles


def _build_frame(rng, records, winners, full=False, compact=False):
    records = [records[int(j)] for j in rng.permutation(len(records))]
    ordered_slots = [i for i, x in enumerate(records) if "priority" in x]
    ordered = sorted((records[i] for i in ordered_slots), key=lambda x: x["priority"])
    for i, record in zip(ordered_slots, ordered):
        records[i] = record
    stream = [x["color"] for x in records if x["color"] == winners[x["shape"]]
              for _ in x["shape"]]
    q = len(stream)
    # Full frames must leave a stopping cell beyond the emitted stream.
    max_height = 29-q if full else 29
    for attempt in range(12):
        conservative = compact or attempt == 11
        a = b = 3 if conservative else int(rng.integers(1, 6))
        if not conservative:
            b = int(rng.integers(1, 6))
            if a+b < 4:
                b = 4-a
        need_width = max(min(_extent(v)[1] for v in x["variants"]) for x in records)
        if a+b-1 < need_width:
            b = max(b, need_width+1-a)
        stem = 0 if conservative else int(rng.integers(0, 3))
        bounds = [(a, a)] * (stem+1)
        left = right = a
        while left > 0 or right < a+b:
            dl = left > 0 and (conservative or right == a+b or rng.random() < .8)
            dr = right < a+b and (conservative or left == 0 or rng.random() < .8)
            if not dl and not dr:
                dl, dr = left > 0, left == 0
            left, right = left-int(dl), right+int(dr)
            bounds.append((left, right))
        cap = len(bounds)
        y = cap + (0 if conservative else int(rng.integers(0, 3)))
        cargoes = []
        for x in records:
            options = [v for v in x["variants"] if _extent(v)[1] <= a+b-1]
            if conservative:
                options = [v for v in options if _extent(v)[0] == min(_extent(z)[0] for z in options)]
            v = _choice(rng, options)
            h, w = _extent(v)
            c = int(rng.integers(1, a+b-w+1))
            cells = {(y+r, c+col) for r, col in v}
            cargoes.append(dict(x, cells=cells))
            y += h + (1 if conservative else int(rng.integers(1, 3)))
        height = y - 1 if conservative else y + int(rng.integers(0, 2))
        if height > max_height:
            continue
        bounds.extend([(0, a+b)] * (height-cap))
        frame = {(r, c) for r, (l, u) in enumerate(bounds) for c in {l, u}}
        if full:
            clearance = q+1
        elif not q:
            clearance = int(rng.integers(0, 4))
        else:
            mode = int(rng.integers(5))
            clearance = q+1 if mode < 2 else q if mode == 2 else int(rng.integers(0, q))
            clearance = min(clearance, 30-height)
        return dict(frame=frame, cargoes=cargoes, tip=(0, a), height=height,
                    width=a+b+1, quantity=q, stream=stream, clearance=clearance)
    return None


def _orient_frame(spec, d):
    points = spec["frame"] | {p for x in spec["cargoes"] for p in x["cells"]}
    transformed = [_turn(p, d) for p in points]
    r0, c0 = min(r for r, c in transformed), min(c for r, c in transformed)
    def convert(p):
        r, c = _turn(p, d)
        return r-r0, c-c0
    return dict(spec, frame={convert(p) for p in spec["frame"]}, tip=convert(spec["tip"]),
                cargoes=[dict(x, cells={convert(p) for p in x["cells"]}) for x in spec["cargoes"]],
                height=max(r for r, c in transformed)-r0+1,
                width=max(c for r, c in transformed)-c0+1, direction=d)


def _try_pack(rng, specs, height, width, directions):
    boxes = np.zeros((height, width), dtype=bool)
    forbidden = boxes.copy()
    rays = set()
    placed = []
    for spec, d in zip(specs, directions):
        obj = _orient_frame(spec, d)
        h, w = obj["height"], obj["width"]
        dr, dc = STEPS[d]
        positions = []
        for r in range(height-h+1):
            for c in range(width-w+1):
                tr, tc = r+obj["tip"][0], c+obj["tip"][1]
                room = (tr, width-1-tc, height-1-tr, tc)[d]
                if room >= obj["clearance"] and (obj["clearance"] > obj["quantity"] or room == obj["clearance"]):
                    positions.append((r, c))
        accepted = None
        for index in rng.permutation(len(positions)):
            r, c = positions[int(index)]
            if np.any(forbidden[r:r+h, c:c+w]):
                continue
            tip = r+obj["tip"][0], c+obj["tip"][1]
            path = []
            # Reserve the actual stream plus its visible stopping cell.
            for k in range(1, obj["quantity"]+2):
                p = tip[0]+k*dr, tip[1]+k*dc
                if not (0 <= p[0] < height and 0 <= p[1] < width):
                    break
                path.append(p)
            if any(boxes[p] or p in rays for p in path):
                continue
            accepted = r, c, tip, path
            break
        if accepted is None:
            return None
        r, c, tip, path = accepted
        placed.append(dict(obj, origin=(r, c), world_tip=tip))
        boxes[r:r+h, c:c+w] = True
        forbidden[max(0, r-1):min(height, r+h+1), max(0, c-1):min(width, c+w+1)] = True
        for p in path:
            forbidden[p] = True
        rays.update(path)
    return placed


def _scene(rng, bundles, bg, frame_color, force_full=False, teaching=False):
    ballots = defaultdict(Counter)
    for records in bundles:
        for x in records:
            ballots[x["shape"]][x["color"]] += 1
    winners = {s: max(counts, key=counts.get) for s, counts in ballots.items()}
    quantities = [sum(len(x["shape"]) for x in b if x["color"] == winners[x["shape"]]) for b in bundles]
    full_index = min((j for j, q in enumerate(quantities) if q), key=lambda j: quantities[j])
    specs = [_build_frame(rng, b, winners, full=force_full and (teaching or j == full_index))
             for j, b in enumerate(bundles)]
    if any(s is None for s in specs):
        return None
    placed = None
    for attempt in range(18):
        height, width = int(rng.integers(16, 31)), int(rng.integers(14, 31))
        if attempt >= 9:
            height = width = 30
        common = int(rng.integers(4))
        directions = ([common]*len(specs) if attempt >= 12 or rng.random() < .45
                      else [int(rng.integers(4)) for _ in specs])
        placed = _try_pack(rng, specs, height, width, directions)
        if placed is not None:
            break
    if placed is None:
        # Guaranteed disjoint columns for at most three narrow frames.
        specs = [_build_frame(rng, b, winners, full=force_full and (teaching or j == full_index), compact=True)
                 for j, b in enumerate(bundles)]
        if any(s is None for s in specs) or sum(s["width"] for s in specs)+len(specs)-1 > 30:
            return None
        height = width = 30
        placed = []
        c = 0
        for spec in specs:
            obj = _orient_frame(spec, 0)
            placed.append(dict(obj, origin=(obj["clearance"], c),
                               world_tip=(obj["clearance"], c+obj["tip"][1])))
            c += obj["width"]+1
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for obj in placed:
        r, c = obj["origin"]
        for y, x in obj["frame"]:
            inp[r+y, c+x] = out[r+y, c+x] = frame_color
        for cargo in obj["cargoes"]:
            for y, x in cargo["cells"]:
                inp[r+y, c+x] = cargo["color"]
        dr, dc = STEPS[obj["direction"]]
        tr, tc = obj["world_tip"]
        for k, color in enumerate(obj["stream"], 1):
            y, x = tr+k*dr, tc+k*dc
            if not (0 <= y < height and 0 <= x < width):
                break
            out[y, x] = color
    if np.count_nonzero(inp == bg) <= inp.size/2:
        return None
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    pool = _pool()
    anchors = [s for s in pool if len(s) in (4, 5) and max(_extent(s)) <= 3
               and _normalize((r, -c) for r, c in s) not in _variants(s, False)
               and any(t != s and len(t) == len(s) and sorted(_extent(t)) == sorted(_extent(s)) for t in pool)]
    anchor = _choice(rng, anchors)
    companion = _choice(rng, [t for t in pool if t != anchor and len(t) == len(anchor)
                             and sorted(_extent(t)) == sorted(_extent(anchor))])
    small = _signature(((0, 0), (0, 1)))
    vote_palette = [int(c) for c in rng.choice(10, 3, replace=False)]
    role_palette = [c for c in range(10) if c not in vote_palette]
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    for split, n in (("train", num_train), ("test", num_test)):
        for i in range(n):
            teaching = split == "train" and i < 2
            for attempt in range(40):
                roles = role_palette if teaching else list(range(10))
                bg, fc = map(int, rng.choice(roles, 2, replace=False))
                if teaching and i == 1:
                    previous = _read(puzzle["train"][0]["input"])
                    bg = int(_choice(rng, [c for c in roles if c != previous["background"]]))
                    fc = int(_choice(rng, [c for c in roles if c not in (bg, previous["frames"][0]["color"])]))
                if teaching:
                    bundles = _teaching_cargoes(anchor, companion, small, vote_palette, rich=i == 1)
                else:
                    k = int(rng.integers(1, 4 if split == "train" else 5))
                    eligible = [s for s in pool if len(s) <= 8 and max(_extent(s)) <= 4]
                    shapes = [eligible[int(j)] for j in rng.choice(len(eligible), k, replace=False)]
                    if split == "test":
                        # A new visible shape with its own votes requires
                        # transferring the operation, not a learned dictionary.
                        novel = [s for s in eligible if s not in known]
                        shapes[0] = _choice(rng, novel)
                        shapes = list(dict.fromkeys(shapes))
                    palette = [int(c) for c in rng.choice([c for c in range(10) if c not in (bg, fc)],
                                                         int(rng.integers(3, 7)), replace=False)]
                    bundles = _ordinary_cargoes(rng, shapes, palette)
                pair = _scene(rng, bundles, bg, fc, force_full=teaching or split == "test", teaching=teaching)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction did not fit")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                known.update(x["shape"] for b in bundles for x in b)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fails(pair, **kwargs):
    try:
        a = np.asarray(pair["input"])
        guess = _prediction(a, _read(a), **kwargs)
        return not np.array_equal(guess, pair["output"])
    except AssertionError:
        return True


def _training_evidence(pairs):
    # Alternative failures include undefined tied alternatives. The intended
    # rule itself must always have unique elections, checked separately.
    alternatives = dict(no_election=dict(all_cargo=True), area_only=dict(grouping="area"),
                        box_only=dict(grouping="box"), rotation_only=dict(grouping="rotation"),
                        global_color=dict(grouping="global"), color_sort=dict(color_order=True),
                        local_election=dict(local=True), count_components=dict(one_per_component=True),
                        backwards_queue=dict(reverse=True), extend_to_edge=dict(edge=True))
    for name, kwargs in alternatives.items():
        assert any(_fails(p, **kwargs) for p in pairs), f"training admits {name}"
    history, lengths = defaultdict(set), set()
    multi_color = False
    for p in pairs:
        data = _read(p["input"])
        ballots, winners = _votes(data)
        for s, c in winners.items():
            history[s].add(c)
        for f in data["frames"]:
            selected = [x for x in f["cargoes"] if x["color"] == winners[x["shape"]]]
            q = sum(x["area"] for x in selected)
            dr, dc = STEPS[f["direction"]]
            tr, tc = f["tip"]
            room = (tr, p["input"].shape[1]-1-tc, p["input"].shape[0]-1-tr, tc)[f["direction"]]
            if q and room > q:
                lengths.add(q)
                multi_color |= len({x["color"] for x in selected}) >= 2
    assert len(lengths) >= 2 and multi_color, "need varied fully visible multicolor streams"
    assert any(len(colors) > 1 for colors in history.values()), "need changing shape winners"
    assert any(len(v) >= 3 and max(v.values())*2 <= sum(v.values())
               for p in pairs for v in _votes(_read(p["input"]))[0].values()), "need nonmajority plurality"


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, train_shapes = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _as_grid(pair["input"]), _as_grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            data = _read(a)
            ballots, winners = _votes(data)
            assert np.array_equal(_prediction(a, data), b), "wrong election/queue/count or changed frame"
            assert not np.array_equal(a, b)
            for f in data["frames"]:
                assert all(a[p] == b[p] for p in f["frame"])
            assert all(b[p] == data["background"] for x in data["cargoes"] for p in x["cells"])
            if split == "train":
                train_shapes.update(ballots)
    _training_evidence(puzzle["train"])
    for pair in puzzle["test"]:
        data = _read(pair["input"])
        _, winners = _votes(data)
        assert set(winners) - train_shapes, "test must transfer to a new shape"
        assert any(sum(x["area"] for x in f["cargoes"] if x["color"] == winners[x["shape"]]) <
                   (f["tip"][0], pair["input"].shape[1]-1-f["tip"][1],
                    pair["input"].shape[0]-1-f["tip"][0], f["tip"][1])[f["direction"]]
                   and any(x["color"] == winners[x["shape"]] for x in f["cargoes"])
                   for f in data["frames"]), "test must retain visible count-dependent stopping"


def _fingerprint(p):
    return tuple((_key(x["input"]), _key(x["output"])) for s in ("train", "test") for x in p[s])


def self_test() -> dict:
    fingerprints, shapes, dimensions, profiles = set(), set(), set(), set()
    directions, frame_counts, cargo_counts, ballots_seen, lengths, clipping = set(), set(), set(), set(), set(), set()
    backgrounds, mixed, empty, shared = set(), 0, 0, 0
    normalized_grids = set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                data = _read(a)
                ballots, winners = _votes(data)
                shapes.update(ballots)
                ballots_seen.update(tuple(sorted(v.values(), reverse=True)) for v in ballots.values())
                dimensions.add(a.shape)
                backgrounds.add(data["background"])
                frame_counts.add(len(data["frames"]))
                cargo_counts.add(len(data["cargoes"]))
                mixed += len({f["direction"] for f in data["frames"]}) > 1
                shared += len(set(winners.values())) < len(winners)
                roles = {}
                normalized_grids.add(tuple(roles.setdefault(int(c), len(roles)) for c in a.flat))
                for f in data["frames"]:
                    directions.add(f["direction"])
                    profiles.add(_normalize(_unturn(x, f["direction"]) for x in f["frame"]))
                    q = sum(x["area"] for x in f["cargoes"] if x["color"] == winners[x["shape"]])
                    lengths.add(q)
                    r, c = f["tip"]
                    room = (r, a.shape[1]-1-c, a.shape[0]-1-r, c)[f["direction"]]
                    clipping.add("empty" if not q else "free" if room > q else "exact" if room == q else "zero" if room == 0 else "clip")
                    empty += q == 0
                # Input-only rotation and arbitrary palette relabeling regressions.
                if split == "test":
                    perm = np.array([3, 7, 1, 9, 0, 6, 4, 8, 2, 5], dtype=np.int8)
                    for turn in range(4):
                        assert np.array_equal(solve(perm[np.rot90(a, turn)]), perm[np.rot90(pair["output"], turn)])
    assert len(fingerprints) == 42 and len(normalized_grids) >= 190
    assert len(dimensions) >= 25 and len(profiles) >= 70 and len(shapes) >= 30
    assert directions == set(range(4)) and frame_counts == {1, 2, 3}
    assert len(cargo_counts) >= 8 and len(ballots_seen) >= 8 and len(lengths) >= 15
    assert {"free", "exact", "clip", "zero", "empty"} <= clipping
    assert mixed >= 10 and empty >= 10 and shared >= 10 and len(backgrounds) >= 7
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(0, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), normalized_inputs=len(normalized_grids),
                dimensions=len(dimensions), frame_profiles=len(profiles), shape_classes=len(shapes),
                cargo_counts=sorted(cargo_counts), ballot_patterns=sorted(ballots_seen),
                stream_lengths=sorted(lengths), clearance_modes=sorted(clipping), mixed_scenes=mixed,
                empty_streams=empty, shared_winner_scenes=shared)


if __name__ == "__main__":
    print(self_test())
