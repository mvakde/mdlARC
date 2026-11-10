"""Hollow engines: enclosed cells pay for motion, using a local color key.

Only the standard library and NumPy are used.  The reader accepts the geometric
family described in task_spec.md, not just shapes emitted by this sampler.
"""
import numpy as np


DIRS = ("U", "D", "L", "R")
DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _need(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    steps = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else N4
    h, w = mask.shape
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(tuple(sorted(cells)))
    return result


def _holes(wall):
    """Four-connected cavities in a wall; diagonal ink contacts can seal them."""
    padded = np.pad(wall, 1, constant_values=False)
    h, w = padded.shape
    return [tuple((r - 1, c - 1) for r, c in cells)
            for cells in _components(~padded)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]


def _order(cells, direction):
    keys = {"U": lambda p: (p[0], p[1]), "D": lambda p: (-p[0], -p[1]),
            "L": lambda p: (p[1], -p[0]), "R": lambda p: (-p[1], p[0])}
    return sorted(cells, key=keys[direction])


def _bar(t, l, h, w, direction, distance=0):
    if direction == "U":
        return [(t - 1 - distance, c) for c in range(l, l + w)]
    if direction == "D":
        return [(t + h + distance, c) for c in range(l, l + w)]
    if direction == "L":
        return [(r, l - 1 - distance) for r in range(t, t + h)]
    return [(r, l + w + distance) for r in range(t, t + h)]


def _in_bounds(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _exact_bar(grid, cells, color):
    if not _in_bounds(cells, grid.shape) or any(grid[q] != color for q in cells):
        return False
    a, b = cells[0], cells[-1]
    dr, dc = (0, 1) if a[0] == b[0] else (1, 0)
    for q in ((a[0] - dr, a[1] - dc), (b[0] + dr, b[1] + dc)):
        if _in_bounds([q], grid.shape) and grid[q] == color:
            return False
    return True


def _key(grid, background):
    h, w = grid.shape
    candidates = []
    for cells in _components(grid != background, diagonal=True):
        rr, cc = zip(*cells)
        t, b, l, r = min(rr), max(rr), min(cc), max(cc)
        sh, sw = b - t + 1, r - l + 1
        if len(cells) != sh * sw:
            continue
        lanes = []
        # The strip never reaches a corner. Its short dimension is exactly two.
        if sh == 2 and 2 <= sw <= 4 and 0 < l and r < w - 1:
            if t == 0:
                lanes.append(("top", grid[t, l:r + 1], grid[b, l:r + 1]))
            if b == h - 1:
                lanes.append(("bottom", grid[b, l:r + 1], grid[t, l:r + 1]))
        if sw == 2 and 2 <= sh <= 4 and 0 < t and b < h - 1:
            if l == 0:
                lanes.append(("left", grid[t:b + 1, l], grid[t:b + 1, r]))
            if r == w - 1:
                lanes.append(("right", grid[t:b + 1, r], grid[t:b + 1, l]))
        for side, outer, inner in lanes:
            sources, fills = list(map(int, outer)), list(map(int, inner))
            if len(set(sources)) == len(sources) and set(sources).isdisjoint(fills):
                candidates.append(dict(side=side, mapping=dict(zip(sources, fills)),
                                       cells=set(cells)))
    _need(len(candidates) == 1, "missing or ambiguous local key")
    return candidates[0]


def _explain(grid, bg, key, marker):
    mapping = key["mapping"]
    fills = set(mapping.values())
    cover = np.zeros(grid.shape, dtype=np.int8)
    swept = np.zeros(grid.shape, dtype=np.int8)
    for q in key["cells"]:
        cover[q] = 1
        swept[q] = 1
    engines = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, marker} - fills):
        mask = grid == color
        for q in key["cells"]:
            mask[q] = False
        for ink in _components(mask, diagonal=True):
            rr, cc = zip(*ink)
            t, l = min(rr), min(cc)
            h, w = max(rr) - t + 1, max(cc) - l + 1
            _need(h >= 3 and w >= 3, "contour too small")
            block = grid[t:t + h, l:l + w]
            wall = block == color
            holes = _holes(wall)
            holes_flat = {q for hole in holes for q in hole}
            active = color in mapping
            allowed = {bg, color, mapping[color]} if active else {bg, color}
            _need(set(map(int, np.unique(block))) <= allowed, "mixed contour ownership")
            for r, c in np.argwhere(~wall):
                if (int(r), int(c)) not in holes_flat:
                    _need(block[r, c] == bg, "spent ink outside a sealed cavity")
            attached = [(d, _bar(t, l, h, w, d)) for d in DIRS
                        if _exact_bar(grid, _bar(t, l, h, w, d), marker)]
            _need(len(attached) == 1, "engine needs one leading bar")
            direction, lead = attached[0]
            destination = None
            for distance in range(1, 31):
                segment = _bar(t, l, h, w, direction, distance)
                if not _in_bounds(segment, grid.shape):
                    break
                if _exact_bar(grid, segment, marker):
                    destination = segment
                    break
                _need(all(grid[q] == bg for q in segment), "obstructed travel corridor")
            _need(destination is not None, "missing destination bar")
            ordered = _order(holes_flat, direction)
            advanced = sum(block[q] != bg for q in ordered)
            if active:
                values = [int(block[q]) for q in ordered]
                _need(values == [mapping[color]] * advanced + [bg] * (len(ordered) - advanced),
                      "spent cells are not a direction-relative prefix")
            else:
                _need(advanced == 0, "unlisted engine has spent ink")
            fuel = len(ordered) - advanced if active else 0
            movement = min(fuel, distance)
            bounds_r = [t, t + h - 1] + [q[0] for q in destination]
            bounds_c = [l, l + w - 1] + [q[1] for q in destination]
            bounds = (min(bounds_r), max(bounds_r), min(bounds_c), max(bounds_c))
            y0, y1, x0, x1 = bounds
            _need(y0 >= 1 and x0 >= 1 and y1 < grid.shape[0] - 1 and x1 < grid.shape[1] - 1,
                  "engine footprints must be inset")
            swept[y0:y1 + 1, x0:x1 + 1] += 1
            for q in ink:
                cover[q] += 1
            for r, c in ordered:
                if block[r, c] != bg:
                    cover[t + r, l + c] += 1
            for q in lead + destination:
                cover[q] += 1
            engines.append(dict(top=t, left=l, height=h, width=w, color=color,
                                direction=direction, distance=distance, holes=holes,
                                ordered=ordered, advanced=advanced, fuel=fuel,
                                movement=movement, active=active, lead=lead,
                                destination=destination, bounds=bounds, block=block.copy()))
    _need(engines, "no engines")
    _need(np.max(cover) == 1 and np.array_equal(cover == 1, grid != bg), "unexplained ink")
    _need(np.max(swept) == 1, "overlapping swept regions")
    return sorted(engines, key=lambda e: (e["top"], e["left"]))


def _read(input_grid):
    grid = _grid(input_grid)
    corners = [int(grid[q]) for q in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    _need(len(set(corners)) == 1, "corners must encode the background")
    bg = corners[0]
    key = _key(grid, bg)
    forbidden = {bg} | set(key["mapping"]) | set(key["mapping"].values())
    explanations = []
    for marker in sorted(set(map(int, np.unique(grid))) - forbidden):
        try:
            explanations.append((marker, _explain(grid, bg, key, marker)))
        except AssertionError:
            pass
    _need(len(explanations) == 1, "scene needs one complete marker/engine explanation")
    marker, engines = explanations[0]
    return dict(background=bg, marker=marker, key=key, engines=engines)


def _transform(grid, scene, unit_steps=False):
    out = grid.copy()
    bg, marker = scene["background"], scene["marker"]
    for e in scene["engines"]:
        t, l, h, w = (e[k] for k in ("top", "left", "height", "width"))
        out[t:t + h, l:l + w] = bg
        for q in e["lead"]:
            out[q] = bg
    for e in scene["engines"]:
        block = e["block"].copy()
        movement = e["movement"]
        if e["active"]:
            fill = scene["key"]["mapping"][e["color"]]
            if unit_steps:
                ticks = 0
                while ticks < e["distance"]:
                    pending = [q for q in e["ordered"] if block[q] == bg]
                    if not pending:
                        break
                    block[pending[0]] = fill
                    ticks += 1
                _need(ticks == movement, "unit-step movement disagrees")
            else:
                for q in e["ordered"][e["advanced"]:e["advanced"] + movement]:
                    block[q] = fill
        dr, dc = DELTA[e["direction"]]
        t, l = e["top"] + dr * movement, e["left"] + dc * movement
        h, w = block.shape
        out[t:t + h, l:l + w] = block
        for q in _bar(t, l, h, w, e["direction"]):
            out[q] = marker
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, cavities, states and translations from this input alone."""
    grid = _grid(input_grid)
    return _transform(grid, _read(grid))


def _dilate(mask, diagonal=False):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = mask.copy()
    steps = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else N4
    for dr, dc in steps:
        result |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _profile(rng, h, w):
    cavity = np.zeros((h, w), dtype=bool)
    lo, hi = 0, w - 1
    for r in range(h):
        if r:
            anchor = int(rng.integers(lo, hi + 1))
            lo, hi = int(rng.integers(anchor + 1)), int(rng.integers(anchor, w))
        cavity[r, lo:hi + 1] = True
    if h > 1 and w > 1 and np.all(cavity):
        cavity[-1, -1] = False
    return cavity


def _shape(rng, kind, count=1, small=False):
    if kind == "diamond":
        radius = 2 if small else int(rng.integers(1, 4))
        rr, cc = np.indices((2 * radius + 1, 2 * radius + 1))
        mask = abs(rr - radius) + abs(cc - radius) == radius
    elif kind == "multi":
        h, w = (5, 7) if small and count == 2 else (int(rng.integers(5, 10)), int(rng.integers(7, 13)))
        zones = [(1, h - 1, 1, w - 1)]
        while len(zones) < count:
            eligible = [i for i, (a, b, c, d) in enumerate(zones) if b - a >= 3 or d - c >= 3]
            _need(eligible, "cannot partition cavities")
            i = int(rng.choice(eligible))
            a, b, c, d = zones.pop(i)
            horizontal = b - a >= 3 and (d - c < 3 or rng.random() < .5)
            if horizontal:
                cut = int(rng.integers(a + 1, b - 1))
                zones.extend([(a, cut, c, d), (cut + 1, b, c, d)])
            else:
                cut = int(rng.integers(c + 1, d - 1))
                zones.extend([(a, b, c, cut), (a, b, cut + 1, d)])
        mask = np.ones((h, w), dtype=bool)
        for a, b, c, d in zones:
            cavity = _profile(rng, b - a, d - c) if rng.random() < .65 else np.ones((b - a, d - c), dtype=bool)
            mask[a:b, c:d][cavity] = False
        for q in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            if rng.random() < .6:
                mask[q] = False
    else:
        h, w = int(rng.integers(1, 4 if small else 6)), int(rng.integers(2, 4 if small else 7))
        if kind == "profile":
            h, w = max(h, 2), max(w, 2)
            cavity = _profile(rng, h, w)
            thickness = 1 if small else int(rng.choice([1, 1, 2]))
            cavity = np.pad(cavity, thickness)
            mask = cavity.copy()
            for _ in range(thickness):
                mask = _dilate(mask, diagonal=rng.random() < .4)
            mask &= ~cavity
        else:
            thickness = 1 if small or kind == "open" else int(rng.choice([1, 1, 2]))
            mask = np.ones((h + 2 * thickness, w + 2 * thickness), dtype=bool)
            mask[thickness:thickness + h, thickness:thickness + w] = False
            if kind == "open":
                mask[0, int(rng.integers(1, w + 1))] = False
    rr, cc = np.nonzero(mask)
    mask = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    _need(len(_components(mask, True)) == 1, "disconnected contour")
    return mask.copy()


def _engine_spec(rng, direction, role="random", kind=None, holes=1, small=False):
    for _ in range(40):
        chosen = kind or str(rng.choice(["rectangle", "profile", "diamond", "multi", "open"]))
        mask = _shape(rng, chosen, holes if chosen == "multi" else 1, small)
        cavities = _holes(mask)
        ordered = _order({q for hole in cavities for q in hole}, direction)
        capacity = len(ordered)
        max_distance = min(12, 26 - (mask.shape[0] if direction in ("U", "D") else mask.shape[1]))
        if role == "pool_lesson":
            ids = {q: i for i, hole in enumerate(cavities) for q in hole}
            options = [(a, m) for a in range(capacity) for m in range(2, min(5, max_distance + 1))
                       if a + m < capacity and len({ids[q] for q in ordered[a:a + m]}) >= 2]
            if not options:
                continue
            advanced, distance = options[int(rng.integers(len(options)))]
        elif role == "fuel_lesson":
            if capacity < 5:
                continue
            remaining = int(rng.integers(1, min(4, capacity - 1, max_distance - 1) + 1))
            advanced = capacity - remaining
            distance = int(rng.integers(remaining + 1, min(max_distance, remaining + 4) + 1))
        elif role == "exhausted":
            advanced, distance = capacity, int(rng.integers(1, max_distance + 1))
        elif role in ("open", "inert") or not capacity:
            advanced, distance = 0, int(rng.integers(1, min(7, max_distance) + 1))
        else:
            stop = str(rng.choice(["destination", "fuel", "exact", "zero"], p=[.35, .3, .25, .1]))
            if role == "target":
                stop = str(rng.choice(["destination", "fuel", "exact"]))
            if stop == "zero":
                advanced, distance = capacity, int(rng.integers(1, max_distance + 1))
            elif stop == "destination" and capacity >= 2:
                distance = int(rng.integers(1, min(capacity - 1, max_distance) + 1))
                remaining = capacity if rng.random() < .3 else int(rng.integers(distance + 1, capacity + 1))
                advanced = capacity - remaining
            elif stop == "fuel" and max_distance >= 2:
                remaining = int(rng.integers(1, min(capacity, max_distance - 1) + 1))
                advanced = capacity - remaining
                distance = int(rng.integers(remaining + 1, max_distance + 1))
            else:
                distance = int(rng.integers(1, min(capacity, max_distance) + 1))
                advanced = capacity - distance
        return dict(mask=mask, direction=direction, ordered=ordered, advanced=advanced,
                    distance=distance, role=role, kind=chosen)
    raise AssertionError("bounded engine construction failed")


def _footprint(spec):
    h, w = spec["mask"].shape
    d = spec["distance"] + 1
    return (h + d, w) if spec["direction"] in ("U", "D") else (h, w + d)


def _positions(occupied, h, w):
    blocked = _dilate(occupied, True)
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return []
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _put_key(grid, mapping, side, start):
    cells = []
    for i, (source, fill) in enumerate(mapping.items()):
        p = start + i
        coords = {"top": ((0, p), (1, p)), "bottom": ((grid.shape[0] - 1, p), (grid.shape[0] - 2, p)),
                  "left": ((p, 0), (p, 1)), "right": ((p, grid.shape[1] - 1), (p, grid.shape[1] - 2))}[side]
        for q, color in zip(coords, (source, fill)):
            grid[q] = color
            cells.append(q)
    return cells


def _render(rng, specs, mapping, bg, marker, side):
    for attempt in range(24):
        shapes = [_footprint(s) for s in specs]
        loh, low = max(h for h, w in shapes) + 2, max(w for h, w in shapes) + 2
        if attempt >= 16:
            H = W = 30
        else:
            H, W = int(rng.integers(max(14, loh), 31)), int(rng.integers(max(14, low), 31))
        inp = np.full((H, W), bg, dtype=np.int8)
        occupied = np.zeros((H, W), dtype=bool)
        axis = W if side in ("top", "bottom") else H
        key_cells = _put_key(inp, mapping, side, int(rng.integers(1, axis - len(mapping))))
        for q in key_cells:
            occupied[q] = True
        placements = [None] * len(specs)
        for i in sorted(range(len(specs)), key=lambda j: -shapes[j][0] * shapes[j][1]):
            h, w = shapes[i]
            positions = _positions(occupied, h, w)
            if not len(positions):
                break
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            placements[i] = (y, x)
            occupied[y:y + h, x:x + w] = True
        else:
            records = []
            for s, (y, x) in zip(specs, placements):
                mask, d, direction = s["mask"], s["distance"], s["direction"]
                h, w = mask.shape
                t = y + d + 1 if direction == "U" else y
                l = x + d + 1 if direction == "L" else x
                block = np.full(mask.shape, bg, dtype=np.int8)
                block[mask] = s["color"]
                active = s["color"] in mapping
                if active:
                    for q in s["ordered"][:s["advanced"]]:
                        block[q] = mapping[s["color"]]
                inp[t:t + h, l:l + w] = block
                lead, dest = _bar(t, l, h, w, direction), _bar(t, l, h, w, direction, d)
                for q in lead + dest:
                    inp[q] = marker
                movement = min(len(s["ordered"]) - s["advanced"], d) if active else 0
                records.append((s, t, l, block, lead, movement))
            # Render expected output from the sampled object records, not solve().
            out = inp.copy()
            for s, t, l, block, lead, movement in records:
                h, w = block.shape
                out[t:t + h, l:l + w] = bg
                for q in lead:
                    out[q] = bg
            for s, t, l, block, lead, movement in records:
                block = block.copy()
                for q in s["ordered"][s["advanced"]:s["advanced"] + movement]:
                    block[q] = mapping[s["color"]]
                dr, dc = DELTA[s["direction"]]
                t, l = t + dr * movement, l + dc * movement
                h, w = block.shape
                out[t:t + h, l:l + w] = block
                for q in _bar(t, l, h, w, s["direction"]):
                    out[q] = marker
            return dict(input=inp, output=out)
    return None


def _make_pair(rng, split, lesson, directions, side):
    for attempt in range(24):
        small = attempt >= 12
        n = int(rng.integers(2, 5))
        palette = list(map(int, rng.permutation(10)))
        bg, marker = palette[:2]
        sources, fills, inert = palette[2:2 + n], palette[2 + n:9], palette[9]
        mapping = {c: int(rng.choice(fills)) for c in sources}
        specs = []
        if lesson == 0:
            specs.append(_engine_spec(rng, directions[0], "pool_lesson", "multi", 2, small))
            specs.append(_engine_spec(rng, str(rng.choice(directions)), "open", "open", small=True))
            if len(directions) >= 3:
                specs.append(_engine_spec(rng, directions[2], "target", "profile", small=True))
        elif lesson == 1:
            specs.append(_engine_spec(rng, directions[1], "fuel_lesson", "diamond", small=small))
            specs.append(_engine_spec(rng, str(rng.choice(directions)), "inert", "rectangle", small=True))
            specs.append(_engine_spec(rng, str(rng.choice(directions)), "exhausted", "rectangle", small=True))
            if len(directions) == 4:
                specs.append(_engine_spec(rng, directions[3], "fuel_lesson", "diamond", small=True))
        elif split == "test":
            specs.append(_engine_spec(rng, str(rng.choice(directions)), "target", "multi", int(rng.integers(3, 5))))
        target = max(len(specs), int(rng.choice([1, 2, 3, 4, 5], p=[.12, .28, .28, .22, .1])))
        if small:
            target = len(specs) or 1
        while len(specs) < target:
            role = str(rng.choice(["random", "random", "inert", "open", "exhausted"]))
            kind = "open" if role == "open" else None
            specs.append(_engine_spec(rng, str(rng.choice(directions)), role, kind,
                                      int(rng.integers(1, 3)), small=True))
        for s in specs:
            s["color"] = inert if s["role"] == "inert" else int(rng.choice(sources))
        pair = _render(rng, specs, mapping, bg, marker, side)
        if pair is not None:
            scene = _read(pair["input"])
            _need(np.array_equal(_transform(pair["input"], scene), pair["output"]), "render/reader disagreement")
            return pair
    raise AssertionError("bounded scene construction failed")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _fingerprint(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    directions = list(map(str, rng.permutation(DIRS)))[:int(rng.integers(2, 5))]
    sides = list(map(str, rng.permutation(["top", "bottom", "left", "right"])))
    lessons = [0, 1] + [-1] * (num_train - 2)
    rng.shuffle(lessons)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            side = sides[i % 2] if split == "train" else str(rng.choice(sides))
            for _ in range(20):
                pair = _make_pair(rng, split, lessons[i] if split == "train" else -1, directions, side)
                signature = _fingerprint(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene):
    es = scene["engines"]
    def stop(e):
        if e["fuel"] == 0:
            return "zero"
        return "destination" if e["fuel"] > e["distance"] else "exact" if e["fuel"] == e["distance"] else "fuel"
    pooling = False
    diagonal = False
    irregular = False
    for e in es:
        if not e["active"] or not e["movement"]:
            continue
        ids = {q: i for i, hole in enumerate(e["holes"]) for q in hole}
        spent_now = e["ordered"][e["advanced"]:e["advanced"] + e["movement"]]
        pooling |= len({ids[q] for q in spent_now}) >= 2
        diagonal |= len(_components(e["block"] == e["color"])) > 1
        for hole in e["holes"]:
            rr, cc = zip(*hole)
            irregular |= len(hole) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
    return dict(stops={stop(e) for e in es if e["active"] and e["holes"]},
                directions={e["direction"] for e in es if e["movement"]}, pooling=pooling,
                diagonal=diagonal, irregular=irregular,
                open=any(e["active"] and not e["holes"] for e in es),
                inert=any(not e["active"] and e["holes"] for e in es),
                exhausted=any(e["active"] and e["holes"] and not e["fuel"] for e in es),
                max_holes=max(len(e["holes"]) for e in es))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions changed")
            signature = _fingerprint(inp)
            _need(signature not in seen, "duplicate input")
            seen.add(signature)
            scene = _read(inp)
            expected = _transform(inp, scene)
            _need(np.array_equal(expected, out), "incorrect output")
            _need(np.array_equal(_transform(inp, scene, True), out), "tick/bulk disagreement")
            for q in scene["key"]["cells"]:
                _need(inp[q] == out[q], "key altered")
            for e in scene["engines"]:
                dr, dc = DELTA[e["direction"]]
                t, l = e["top"] + dr * e["movement"], e["left"] + dc * e["movement"]
                block = out[t:t + e["height"], l:l + e["width"]]
                _need(np.array_equal(block == e["color"], e["block"] == e["color"]), "contour shape changed")
                spent = sum(block[q] != scene["background"] for q in e["ordered"])
                _need(spent - e["advanced"] == e["movement"], "one-cell/one-step conservation failed")
                _need(e["movement"] == e["distance"] or e["fuel"] == e["movement"], "wrong stopping condition")
                _need(all(out[q] == scene["marker"] for q in e["destination"]), "destination altered")
            features[split].append(_features(scene))
    train = features["train"]
    _need({"destination", "fuel", "zero"} <= set().union(*(f["stops"] for f in train)), "missing stopping lessons")
    for name in ("pooling", "diagonal", "irregular", "open", "inert", "exhausted"):
        _need(any(f[name] for f in train), "missing lesson: " + name)
    _need(len(set().union(*(f["directions"] for f in train))) >= 2, "rotation not demonstrated")
    _need(set().union(*(f["directions"] for f in features["test"])) <=
          set().union(*(f["directions"] for f in train)), "test travel direction lacks an active lesson")
    train_max = max(f["max_holes"] for f in train)
    _need(train_max == 2 and all(f["max_holes"] > train_max for f in features["test"]), "missing cavity-count shift")


def self_test():
    # A hand-specified scene with analytically specified poses and counter cells.
    # It combines a fuel-limited diamond, destination-limited pooled reservoirs,
    # an opened mapped contour, and a closed unlisted contour.
    hand = np.zeros((25, 27), dtype=np.int8)
    hand[:2, 3:5] = [[1, 2], [4, 5]]
    rr, cc = np.indices((5, 5))
    diamond = abs(rr - 2) + abs(cc - 2) == 2
    a = np.zeros((5, 5), dtype=np.int8)
    a[diamond] = 1
    a[2, 3] = a[1, 2] = 4  # first two cells in rightward order
    hand[8:13, 2:7] = a
    hand[8:13, 7] = hand[8:13, 12] = 9  # distance five, fuel three
    b = np.full((5, 5), 2, dtype=np.int8)
    b[1:4, 1] = b[1:4, 3] = 0
    b[1, 1] = 5
    hand[11:16, 19:24] = b
    hand[10, 19:24] = hand[8, 19:24] = 9  # distance two, fuel five
    opened = np.full((4, 4), 1, dtype=np.int8)
    opened[1:3, 1:3] = 0
    opened[0, 1] = 0
    hand[17:21, 8:12] = opened
    hand[17:21, 7] = hand[17:21, 4] = 9
    hand[17:20, 20:23] = [[8, 8, 8], [8, 0, 8], [8, 8, 8]]
    hand[20, 20:23] = hand[23, 20:23] = 9
    expected = hand.copy()
    expected[8:13, 2:8] = 0
    expected[11:16, 19:24] = expected[10, 19:24] = 0
    a[abs(rr - 2) + abs(cc - 2) < 2] = 4
    expected[8:13, 5:10] = a
    expected[8:13, 10] = 9
    b[1, 3] = b[2, 1] = 5
    expected[9:14, 19:24] = b
    expected[8, 19:24] = 9
    _need(np.array_equal(solve(hand), expected), "hand-counted pose/counter regression")
    hand_scene = _read(hand)
    b_only = hand.copy()
    for e in hand_scene["engines"]:
        if e["active"]:
            for r, c in e["ordered"]:
                b_only[e["top"] + r, e["left"] + c] = hand_scene["key"]["mapping"][e["color"]]
    _need(not np.array_equal(b_only, expected), "omitting A did not change the answer")
    no_enclosure = dict(hand_scene, engines=[dict(e) for e in hand_scene["engines"]])
    for e in no_enclosure["engines"]:
        if e["active"] and not e["holes"]:
            e["ordered"] = _order([tuple(map(int, q)) for q in np.argwhere(e["block"] == 0)], e["direction"])
            e["fuel"] = len(e["ordered"])
            e["movement"] = min(e["fuel"], e["distance"])
    _need(not np.array_equal(_transform(hand, no_enclosure), expected), "omitting B enclosure did not change the answer")
    no_activation = dict(hand_scene, key=dict(hand_scene["key"], mapping=dict(hand_scene["key"]["mapping"])),
                         engines=[dict(e) for e in hand_scene["engines"]])
    for e in no_activation["engines"]:
        if not e["active"]:
            no_activation["key"]["mapping"][e["color"]] = 4
            e["active"], e["fuel"] = True, len(e["ordered"])
            e["movement"] = min(e["fuel"], e["distance"])
    _need(not np.array_equal(_transform(hand, no_activation), expected), "omitting B activation did not change the answer")
    puzzles, shapes, normalized, engine_records, sides, palettes = [], set(), set(), [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                scene = _read(a)
                shapes.add(a.shape)
                roles = {}
                normalized.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                sides.add(scene["key"]["side"])
                palettes.add((scene["background"], scene["marker"], tuple(scene["key"]["mapping"].items())))
                engine_records.extend((split, e) for e in scene["engines"])
    _need(len({tuple(_fingerprint(p["input"]) for s in q.values() for p in s) for q in puzzles}) == 42,
          "seeds repeat puzzles")
    _need(len(shapes) >= 50 and len(normalized) >= 180, "insufficient structural variation")
    _need(sides == {"top", "bottom", "left", "right"} and len(palettes) >= 100, "palette/key variation collapsed")
    _need({e["direction"] for split, e in engine_records} == set(DIRS), "missing direction")
    _need(any(e["active"] and e["holes"] and e["advanced"] == 0 for split, e in engine_records), "no fresh counter")
    _need(max(len(e["holes"]) for split, e in engine_records) == 4, "missing four-cavity engine")
    _need(len({len(_read(p["input"])["engines"]) for q in puzzles for s in q.values() for p in s}) >= 4,
          "engine count variation collapsed")
    repeated = generate_puzzle(17)
    _need(all(np.array_equal(a[k], b[k]) for s in puzzles[17] for a, b in zip(puzzles[17][s], repeated[s])
              for k in ("input", "output")), "seed not reproducible")
    # Equivariance checks cover new literal palettes and rotations without seed access.
    pair = puzzles[0]["train"][0]
    permutation = np.array([3, 8, 1, 9, 0, 6, 2, 5, 7, 4], dtype=np.int8)
    for turn in range(4):
        inp, out = (permutation[np.rot90(pair[k], turn)] for k in ("input", "output"))
        _need(np.array_equal(solve(inp), out), "rotation/palette equivariance failed")
    # Concrete counterfactuals: ignoring motion leaves the pose wrong; ignoring
    # enclosure gives fuel to an open engine; reversing the key disables engines.
    scenes = [_read(p["input"]) for p in puzzles[0]["train"]]
    _need(any(e["movement"] and e["fuel"] > e["distance"] for s in scenes for e in s["engines"]),
          "no A stopping witness")
    _need(any(e["active"] and not e["holes"] and np.count_nonzero(e["block"] == s["background"]) > 0
              for s in scenes for e in s["engines"]), "no B enclosure witness")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(100 + train * 10 + test, train, test)
            _need(len(p["train"]) == train and len(p["test"]) == test, "counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: g.copy() for k, g in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    scene = _read(broken["test"][0]["input"])
    q = min(scene["key"]["cells"])
    broken["test"][0]["output"][q] = scene["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("output corruption accepted")
