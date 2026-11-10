"""Remote-push cargo: endpoint commands, collision locks, and stable packing.

Only the standard library and NumPy are used. No files are read or written.
Canonical construction is horizontal; a puzzle may be transposed to vertical.
The public solver reads its axis, objects, and commands entirely from its input.
"""
from __future__ import annotations

import hashlib

import numpy as np


DISTANCE = 5
SELF_TEST_SEEDS = tuple(range(42))


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _arrow(tip, vector):
    r, c = tip
    dr, dc = vector
    if dr == 0 and abs(dc) == 1:
        return {(r, c), (r - 1, c - dc), (r + 1, c - dc)}
    if dc == 0 and abs(dr) == 1:
        return {(r, c), (r - dr, c - 1), (r - dr, c + 1)}
    raise ValueError("This family uses cardinal arrows")


def _box(r, c, h, w, direction, local_row, local_col):
    return dict(r=int(r), c=int(c), h=int(h), w=int(w),
                tip=(int(r + local_row), int(c + local_col)),
                vector=(0, int(direction)))


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"])


def _overlap(a, b):
    return (_band(a, b)
            and max(a["c"], b["c"]) < min(a["c"] + a["w"], b["c"] + b["w"]))


def _inside(point, obj):
    r, c = point
    return (obj["r"] <= r < obj["r"] + obj["h"]
            and obj["c"] <= c < obj["c"] + obj["w"])


def _commands(objects, distance=DISTANCE, ignore=()):
    """Compute all shots before any motion. Count collisions per coordinate."""
    endpoints = {}
    for i, obj in enumerate(objects):
        if i in ignore:
            continue
        p = tuple(obj["tip"][a] + distance * obj["vector"][a] for a in (0, 1))
        endpoints.setdefault(p, []).append(i)
    commands = [0] * len(objects)
    locks, hits, target_sites = set(), {}, {}
    for point, shooters in endpoints.items():
        targets = [j for j, obj in enumerate(objects) if _inside(point, obj)]
        if len(targets) > 1:
            raise ValueError("Overlapping input crates")
        if not targets:
            continue
        j = targets[0]
        if j in shooters:
            raise ValueError("Self-addressed shots are outside this family")
        target_sites.setdefault(j, []).append((point, len(shooters)))
        if len(shooters) > 1:
            locks.add(j)
        else:
            hits.setdefault(j, set()).add(objects[shooters[0]]["vector"][1])
    for j, directions in hits.items():
        if j not in locks:
            if len(directions) != 1:
                raise ValueError("Opposed commands at separate endpoints")
            commands[j] = next(iter(directions))
    return commands, dict(endpoints=endpoints, locks=locks, hits=hits, sites=target_sites)


def _settle(objects, width, commands):
    """Ordered boundary recurrence; fixed crates are impassable local stops."""
    columns = [o["c"] for o in objects]
    fixed = [i for i, d in enumerate(commands) if d == 0]
    for direction in (-1, 1):
        group = [i for i, d in enumerate(commands) if d == direction]
        group.sort(key=lambda i: (objects[i]["c"] if direction == -1
                                 else -objects[i]["c"] - objects[i]["w"],
                                 objects[i]["r"]))
        settled = []
        for i in group:
            obj = objects[i]
            # A fixed crate behind the original position is irrelevant. An
            # earlier same-class crate is always ahead wherever spans overlap.
            ahead = [j for j in fixed if _band(obj, objects[j]) and
                     (objects[j]["c"] + objects[j]["w"] <= obj["c"]
                      if direction == -1 else
                      objects[j]["c"] >= obj["c"] + obj["w"])]
            blockers = ahead + [j for j in settled if _band(obj, objects[j])]
            if direction == -1:
                columns[i] = max((columns[j] + objects[j]["w"] for j in blockers), default=0)
            else:
                columns[i] = min((columns[j] for j in blockers), default=width) - obj["w"]
            if not 0 <= columns[i] <= width - obj["w"]:
                raise ValueError("Packing exceeds canvas")
            if direction * (columns[i] - obj["c"]) < 0:
                raise ValueError("Packing would move against a command")
            settled.append(i)
    final = [dict(o, c=c) for o, c in zip(objects, columns)]
    if any(_overlap(a, b) for i, a in enumerate(final) for b in final[i + 1:]):
        raise ValueError("Opposite groups have colliding final footprints")
    return columns


def _simulate(objects, width, commands):
    """Validation check using unit movement, rather than the recurrence."""
    columns = [o["c"] for o in objects]
    for direction in (-1, 1):
        group = [i for i, d in enumerate(commands) if d == direction]
        blockers = [i for i, d in enumerate(commands) if d in (0, direction)]
        for _ in range(width * max(1, len(group)) + 1):
            changed = False
            for i in group:
                candidate = dict(objects[i], c=columns[i] + direction)
                if not 0 <= candidate["c"] <= width - candidate["w"]:
                    continue
                if any(_overlap(candidate, dict(objects[j], c=columns[j]))
                       for j in blockers if j != i):
                    continue
                columns[i] += direction
                changed = True
            if not changed:
                break
        else:
            raise AssertionError("Packing did not converge")
    return columns


def _check_grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or any(n < 1 or n > 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30 and colors 0..9")
    return g.astype(np.int8, copy=False)


def _decode(grid):
    """Find complete frames by raster scan, even when frames touch."""
    g = _check_grid(grid)
    counts = np.bincount(g.ravel(), minlength=10)
    background = int(counts.argmax())
    if 2 * counts[background] <= g.size:
        raise ValueError("Background must occupy over half the canvas")
    covered = g == background
    objects = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if covered[r, c]:
                continue
            frame = int(g[r, c])
            if r + 2 >= g.shape[0] or c + 2 >= g.shape[1]:
                raise ValueError("Incomplete crate")
            bottom, right = r + 2, c + 2
            while bottom < g.shape[0] and g[bottom, c + 1] != frame:
                bottom += 1
            while right < g.shape[1] and g[r + 1, right] != frame:
                right += 1
            if bottom >= g.shape[0] or right >= g.shape[1]:
                raise ValueError("Open frame")
            patch = g[r:bottom + 1, c:right + 1]
            interior = patch[1:-1, 1:-1]
            if (not all(np.all(edge == frame) for edge in
                        (patch[0], patch[-1], patch[:, 0], patch[:, -1]))
                    or np.any(interior == frame) or np.any(interior == background)
                    or np.any(covered[r:bottom + 1, c:right + 1])):
                raise ValueError("Malformed or overlapping crate")
            colors, frequencies = np.unique(interior, return_counts=True)
            if len(colors) != 2 or interior.size <= 6 or sum(frequencies == 3) != 1:
                raise ValueError("Interior must contain one three-cell arrow on a plain fill")
            arrow_color = int(colors[np.flatnonzero(frequencies == 3)[0]])
            cells = {(int(rr + r + 1), int(cc + c + 1))
                     for rr, cc in np.argwhere(interior == arrow_color)}
            matches = [(p, v) for p in sorted(cells)
                       for v in ((0, -1), (0, 1), (-1, 0), (1, 0))
                       if _arrow(p, v) == cells]
            if len(matches) != 1:
                raise ValueError("Ambiguous arrow")
            tip, vector = matches[0]
            covered[r:bottom + 1, c:right + 1] = True
            objects.append(dict(r=r, c=c, h=bottom-r+1, w=right-c+1,
                                tip=tip, vector=vector, patch=patch.copy()))
    if not objects or not np.all(covered):
        raise ValueError("No complete cargo")
    axes = {0 if o["vector"][0] else 1 for o in objects}
    if len(axes) != 1:
        raise ValueError("All arrows in one scene must share an axis")
    return background, objects, axes.pop()


def _canonical(grid):
    g = _check_grid(grid)
    background, objects, axis = _decode(g)
    if axis == 0:
        background, objects, _ = _decode(g.T)
        g = g.T
    if any(not (5 <= o["h"] <= 9 and 5 <= o["w"] <= 8) for o in objects):
        raise ValueError("Crate dimensions outside the defined family")
    return g, background, objects, axis


def _paint(shape, background, objects, columns):
    out = np.full(shape, background, dtype=np.int8)
    for obj, c in zip(objects, columns):
        out[obj["r"]:obj["r"] + obj["h"], c:c + obj["w"]] = obj["patch"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Project from the input snapshot, then carry whole crates to their stops."""
    g, bg, objects, axis = _canonical(input_grid)
    commands, _ = _commands(objects)
    columns = _settle(objects, g.shape[1], commands)
    output = _paint(g.shape, bg, objects, columns)
    return output if axis == 1 else output.T.copy()


def _canvas(rng, objects, minimum=(0, 0), right_margin=0):
    objects = [dict(o) for o in objects]
    low = [min(o[k] for o in objects) for k in ("r", "c")]
    for o in objects:
        o["r"] -= low[0]
        o["c"] -= low[1]
        o["tip"] = (o["tip"][0] - low[0], o["tip"][1] - low[1])
    h = max(o["r"] + o["h"] for o in objects)
    w = max(o["c"] + o["w"] for o in objects)
    area = sum(o["h"] * o["w"] for o in objects)
    w = max(w + right_margin, minimum[1])
    if w > 30 or h > 30:
        raise ValueError("Construction does not fit")
    w = int(rng.integers(w, min(30, w + 4) + 1))
    h = max(h, minimum[0], 2 * area // w + 1)
    if h > 30:
        raise ValueError("Insufficient background")
    h = int(rng.integers(h, min(30, h + 5) + 1))
    offsets = (int(rng.integers(h - max(o["r"] + o["h"] for o in objects) + 1)),
               int(rng.integers(w - max(o["c"] + o["w"] for o in objects) - right_margin + 1)))
    for o in objects:
        o["r"] += offsets[0]
        o["c"] += offsets[1]
        o["tip"] = tuple(o["tip"][a] + offsets[a] for a in (0, 1))
    return (h, w), objects


def _convoy(rng, crossing=False):
    n = int(rng.choice((4, 4, 5)))
    h = int(rng.choice((6, 7)))
    step = (h + 1) // 2
    widths = [5] * n if n == 5 else list(map(int, rng.integers(5, 7, n)))
    # The rear active crate fits in the edge slot vacated by the disjoint
    # front active crate under the wrong repacking rule. This guarantees that
    # the demonstration visibly distinguishes stable sliding from overtaking.
    if n == 4:
        widths[-1] = max(widths[1:])
    gaps = [1] * (n - 1)
    if n == 4 and sum(widths) + sum(gaps) < 28:
        gaps[int(rng.integers(n - 1))] += int(rng.integers(2))
    objects, c = [], 0
    for i, w in enumerate(widths):
        # The first n-1 endpoints land on the next crate's top boundary.
        local_col = int(rng.integers(max(2, w + (gaps[i] if i < n-1 else 0) - 5), w - 1))
        objects.append(_box(i * step, c, h, w, 1, step, local_col))
        if i < n - 1:
            c += w + gaps[i]
    if crossing:
        base = max(o["r"] + o["h"] for o in objects) + int(rng.integers(2))
        ch = int(rng.integers(5, min(7, 30 - base) + 1))
        pitch = int(rng.integers(6, 9))
        # Four reciprocal addresses yield W,E,W,E incoming commands. The
        # middle two pass each other; the two same-class queues remain stable.
        for i, d in enumerate((1, -1, 1, -1)):
            objects.append(_box(base, i * pitch, ch, 5, d,
                                int(rng.integers(2, ch - 2)), 3 if d == 1 else 1))
        # All four arrows must hit their partners despite different heights.
        row = base + int(rng.integers(2, ch - 2))
        for o in objects[-4:]:
            o["tip"] = (row, o["tip"][1])
    return _canvas(rng, objects)


def _jam(rng, parallel=True):
    # Two opposite tips are exactly ten columns apart. Their common endpoint
    # lies in the middle crate, whose own shot still releases the right crate.
    we, wl, wr = (int(rng.integers(5, 7)), int(rng.integers(5, 8)), int(rng.integers(5, 8)))
    wm = int(rng.integers(5, 7))
    gap = int(rng.integers(1, 8 - wm))
    left = we + 1
    middle = left + wl + gap
    right = left + wl + 7
    hm = int(rng.integers(5, 7))
    hl, he = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    hr = int(rng.integers(5, 8))
    # Keep the collision on a shared row; middle/extra arrow rows may differ.
    qrow = 2
    objects = [_box(0, 0, he, we, 1, int(rng.integers(2, min(he-3, hl-1) + 1)), we-2),
               _box(0, left, hl, wl, 1, qrow, wl-2),
               _box(0, middle, hm, wm, 1, int(rng.integers(2, hm-2)), 2),
               _box(0, right, hr, wr, -1, qrow, 1)]
    if parallel:
        # A separate receiver gets two E endpoints at distinct coordinates.
        # Keeping this receiver separate makes the upper locked transmitter's
        # outgoing shot indispensable to the upper right crate's movement.
        base = max(he, hl, hm, hr) + int(rng.integers(2))
        step = int(rng.integers(5, 7))
        objects.append(_box(base, middle, step, 5, 1, 2, 2))
        objects.append(_box(base + step, middle, step, 5, 1, 2, 2))
        objects.append(_box(base, right, step + 3, wr, 1, 2, max(2, wr - 5)))
        # This arrow faces a real crate on its row, but that crate is farther
        # than five cells away. It teaches finite range rather than a ray to
        # the first object. It and the lower middle sender stay anchored.
        # Its endpoint is one cell before the receiver, while distances 6..9
        # would address that receiver. This also distinguishes nearby range
        # hypotheses that the wider upper crates alone could leave ambiguous.
        objects.append(_box(base + 1, middle - 9, 5, 5, 1, 2, 3))
    return _canvas(rng, objects, right_margin=1)


def _random_scene(rng, query=False):
    """Bounded proposals mix arbitrary rectangles with near-neighbor placement."""
    if query and rng.random() < .30:
        _, objects = _jam(rng, parallel=False)
        # Change the obstacle's span, keeping its original tip and all commands.
        # Unlike the aligned teaching scene, contact is through partial spans.
        objects[2]["r"] -= 1
        objects[2]["h"] += 1
        if objects[2]["h"] == objects[1]["h"]:
            objects[2]["h"] += 1
        return _canvas(rng, objects, right_margin=1)
    for _ in range(220):
        h, w = int(rng.integers(15, 31)), int(rng.integers(18, 31))
        requested = int(rng.integers(3 if query else 2, 9 if query else 8))
        objects, area = [], 0
        for _ in range(requested):
            for attempt in range(65):
                oh, ow = int(rng.integers(5, 9)), int(rng.integers(5, 9))
                if 2 * (area + oh * ow) >= h * w:
                    continue
                r, c = int(rng.integers(h - oh + 1)), int(rng.integers(w - ow + 1))
                if objects and rng.random() < .7:
                    base = _pick(rng, objects)
                    r = base["r"] + int(rng.integers(-4, 5))
                    c = (base["c"] + base["w"] + int(rng.integers(0, 4))
                         if rng.random() < .5 else base["c"] - ow - int(rng.integers(0, 4)))
                if not (0 <= r <= h - oh and 0 <= c <= w - ow):
                    continue
                d = int(rng.choice((-1, 1)))
                u = int(rng.integers(2 if d == 1 else 1, ow-1 if d == 1 else ow-2))
                obj = _box(r, c, oh, ow, d, int(rng.integers(2, oh - 2)), u)
                if any(_overlap(obj, p) for p in objects):
                    continue
                objects.append(obj)
                area += oh * ow
                break
        if len(objects) < (3 if query else 2):
            continue
        try:
            commands, _ = _commands(objects)
            columns = _settle(objects, w, commands)
        except ValueError:
            continue
        moving = [i for i, o in enumerate(objects) if o["c"] != columns[i]]
        if not moving:
            continue
        # Queries require an irregular, partially overlapping fixed stop. This
        # recombines the taught lock/unhit obstacles with new row relationships.
        if query:
            if len({(o["h"], o["w"]) for o in objects}) < 2:
                continue
            if not any(columns[i] not in (0, w - objects[i]["w"])
                       and any(commands[j] == 0 and _band(objects[i], objects[j])
                               and objects[i]["r"] != objects[j]["r"]
                               and (columns[i] + objects[i]["w"] == columns[j]
                                    or columns[j] + objects[j]["w"] == columns[i])
                               for j in range(len(objects))) for i in moving):
                continue
        return (h, w), objects
    # The fallback is the same endpoint/obstacle interaction, with an offset
    # middle crate to expose partial span overlap. It does not change the rule.
    objects = [_box(0, 0, 5, 5, 1, 2, 3),
               _box(0, 6, 5, 5, 1, 2, 3),
               _box(-1, 12, 6, 5, 1, 3, 2),
               _box(0, 18, 5, 5, -1, 2, 1)]
    return _canvas(rng, objects, right_margin=1)


def _appearance(rng, shape, objects, palette):
    bg = int(_pick(rng, palette["backgrounds"]))
    grid = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        frame, fill, arrow = (int(_pick(rng, palette[k])) for k in ("frames", "fills", "arrows"))
        patch = np.full((o["h"], o["w"]), frame, dtype=np.int8)
        patch[1:-1, 1:-1] = fill
        for r, c in _arrow(o["tip"], o["vector"]):
            patch[r - o["r"], c - o["c"]] = arrow
        o["patch"] = patch
        grid[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = patch
    return grid


def _pair(rng, palette, kind, axis):
    if kind == "teach_chain":
        shape, objects = _convoy(rng, crossing=True)
    elif kind == "teach_jam":
        shape, objects = _jam(rng, parallel=True)
    elif kind == "jam":
        shape, objects = _jam(rng, parallel=bool(rng.integers(2)))
    elif kind == "convoy":
        shape, objects = _convoy(rng)
    else:
        shape, objects = _random_scene(rng, query=kind == "query")
    commands, _ = _commands(objects)
    columns = _settle(objects, shape[1], commands)
    inp = _appearance(rng, shape, objects, palette)
    out = _paint(shape, int(np.bincount(inp.ravel()).argmax()), objects, columns)
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    if rng.random() < .5:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    if axis == 0:
        inp, out = inp.T.copy(), out.T.copy()
    return dict(input=inp, output=out)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _training_distances(train):
    """Check teaching identifiability inside this rule's distance hypothesis set."""
    parsed = []
    for pair in train:
        grid, bg, objects, axis = _canonical(pair["input"])
        out = np.asarray(pair["output"])
        parsed.append((grid, bg, objects, out if axis == 1 else out.T))
    candidates = []
    for distance in range(1, 30):
        for grid, bg, objects, out in parsed:
            try:
                commands, _ = _commands(objects, distance)
                columns = _settle(objects, grid.shape[1], commands)
            except ValueError:
                break
            if not np.array_equal(_paint(grid.shape, bg, objects, columns), out):
                break
        else:
            candidates.append(distance)
    return candidates


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    palette = dict(backgrounds=colors[:2] + colors[9:], frames=colors[2:4],
                   fills=colors[4:6], arrows=colors[6:9])
    axis = int(rng.integers(2))
    train_kinds = ["teach_chain", "teach_jam"] + [
        _pick(rng, ("random", "random", "jam", "convoy")) for _ in range(num_train - 2)]
    rng.shuffle(train_kinds)
    test_kinds = ["query"] + [_pick(rng, ("query", "jam", "random", "convoy"))
                               for _ in range(num_test - 1)]
    puzzle, seen = dict(train=[], test=[]), set()
    for split, kinds in (("train", train_kinds), ("test", test_kinds)):
        for kind in kinds:
            for _ in range(24):
                pair = _pair(rng, palette, kind, axis)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted its finite bound")
    # Changing the teaching geometry is rejection sampling of inputs, never a
    # change in the fixed rule. It is bounded and uses this puzzle's same RNG.
    if _training_distances(puzzle["train"]) != [DISTANCE]:
        j = train_kinds.index("teach_jam")
        for _ in range(12):
            pair = _pair(rng, palette, "teach_jam", axis)
            key = _key(pair["input"])
            if key in seen:
                continue
            seen.remove(_key(puzzle["train"][j]["input"]))
            puzzle["train"][j] = pair
            seen.add(key)
            if _training_distances(puzzle["train"]) == [DISTANCE]:
                break
        else:
            raise RuntimeError("Could not separate fixed-distance teaching hypotheses")
    validate_generated_puzzle(puzzle)
    return puzzle


def _edge_first(objects, width, commands):
    """A deliberately wrong repacker that can jump a same-class predecessor."""
    columns = [o["c"] for o in objects]
    for d in (-1, 1):
        group = [i for i, x in enumerate(commands) if x == d]
        group.sort(key=lambda i: objects[i]["c"] if d == -1 else -objects[i]["c"]-objects[i]["w"])
        fixed = [j for j, x in enumerate(commands) if x == 0]
        placed = []
        for i in group:
            positions = range(width - objects[i]["w"] + 1)
            if d == 1:
                positions = reversed(positions)
            for c in positions:
                candidate = dict(objects[i], c=c)
                if not any(_overlap(candidate, dict(objects[j], c=columns[j])) for j in fixed + placed):
                    columns[i] = c
                    break
            placed.append(i)
    return columns


def measure_pair(pair):
    g, bg, objects, axis = _canonical(pair["input"])
    commands, events = _commands(objects)
    columns = _settle(objects, g.shape[1], commands)
    moved = [i for i, o in enumerate(objects) if columns[i] != o["c"]]
    crossings = sum(commands[i] == 1 and commands[j] == -1 and _band(a, b)
                    and a["c"] < b["c"] and columns[i] >= columns[j] + b["w"]
                    for i, a in enumerate(objects) for j, b in enumerate(objects))
    anchored = sum(columns[i] not in (0, g.shape[1]-objects[i]["w"])
                   and any(commands[j] == 0 and _band(objects[i], b)
                           and (columns[i]+objects[i]["w"] == columns[j]
                                or columns[j]+b["w"] == columns[i])
                           for j, b in enumerate(objects)) for i in moved)
    long_misses = 0
    for o in objects:
        point = (o["tip"][0], o["tip"][1] + DISTANCE * o["vector"][1])
        if any(_inside(point, b) for b in objects):
            continue
        long_misses += any(b is not o and b["r"] <= o["tip"][0] < b["r"] + b["h"]
                           and (b["c"] > point[1] if o["vector"][1] == 1
                                else b["c"] + b["w"] - 1 < point[1]) for b in objects)
    return dict(shape=tuple(pair["input"].shape), axis=axis, background=bg,
                count=len(objects), footprints=tuple(sorted({(o["h"], o["w"]) for o in objects})),
                arrow_offsets=tuple(sorted({(o["tip"][0]-o["r"], o["tip"][1]-o["c"]) for o in objects})),
                directions=tuple(sorted({o["vector"][1] for o in objects})),
                moved=len(moved), fixed=commands.count(0), locks=len(events["locks"]),
                multi_single=sum(len(points) > 1 and j not in events["locks"]
                                 for j, points in events["sites"].items()),
                crossings=int(crossings), anchored_stops=int(anchored),
                stable_witness=columns != _edge_first(objects, g.shape[1], commands),
                travel=tuple(abs(columns[i] - o["c"]) for i, o in enumerate(objects)),
                misses=sum(not any(_inside(p, o) for o in objects) for p in events["endpoints"]),
                long_misses=int(long_misses))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, axes = set(), set()
    teaching = dict(locks=False, multi=False, stable=False, crossing=False, anchor=False,
                    remote=False, not_all=False, range=False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            g, bg, objects, axis = _canonical(inp)
            axes.add(axis)
            commands, events = _commands(objects)
            columns = _settle(objects, g.shape[1], commands)
            assert columns == _simulate(objects, g.shape[1], commands), "Stable packing disagreement"
            expected = _paint(g.shape, bg, objects, columns)
            actual = out if axis == 1 else out.T
            assert np.array_equal(expected, actual), "Incorrect translation or payload"
            assert not np.array_equal(inp, out), "Generated demonstrations must visibly move"
            # Fixed crates and every translated patch must be pixel-exact.
            for i, obj in enumerate(objects):
                assert np.array_equal(actual[obj["r"]:obj["r"]+obj["h"],
                                             columns[i]:columns[i]+obj["w"]], obj["patch"])
                if commands[i] == 0:
                    assert columns[i] == obj["c"]
            if split == "train":
                f = measure_pair(pair)
                teaching["locks"] |= bool(f["locks"])
                teaching["multi"] |= any(len(points) > 1 and j not in events["locks"]
                                          and columns[j] != objects[j]["c"]
                                          for j, points in events["sites"].items())
                teaching["stable"] |= bool(f["stable_witness"])
                teaching["crossing"] |= bool(f["crossings"])
                teaching["anchor"] |= bool(f["anchored_stops"])
                teaching["range"] |= bool(f["long_misses"])
                teaching["remote"] |= any(commands[i] and commands[i] != o["vector"][1]
                                          and columns[i] != o["c"] for i, o in enumerate(objects))
                teaching["not_all"] |= any(commands[i] == 0 and i not in events["locks"]
                                           for i in range(len(objects)))
    assert len(axes) == 1, "One flow axis per puzzle"
    assert all(teaching.values()), f"Missing training witness: {teaching}"
    assert _training_distances(puzzle["train"]) == [DISTANCE], "Training leaves another distance explanation"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _counterfactuals(pair):
    """Deliberately omit inherited mechanisms and compare the resulting grids."""
    g, bg, objects, _ = _canonical(pair["input"])
    commands, events = _commands(objects)
    columns = _settle(objects, g.shape[1], commands)
    expected = _paint(g.shape, bg, objects, columns)

    def different(new_commands):
        try:
            new_columns = _settle(objects, g.shape[1], new_commands)
        except ValueError:
            return True  # The shortcut cannot even produce a valid packing.
        return not np.array_equal(expected, _paint(g.shape, bg, objects, new_columns))

    total_hit_locks = commands.copy()
    for j, sites in events["sites"].items():
        if sum(count for _, count in sites) >= 2:
            total_hit_locks[j] = 0
    no_collision_locks = commands.copy()
    for j in events["locks"]:
        no_collision_locks[j] = objects[j]["vector"][1]
    nearest = commands.copy()
    for o in objects:
        p = (o["tip"][0], o["tip"][1] + DISTANCE * o["vector"][1])
        if any(_inside(p, b) for b in objects):
            continue
        candidates = [(b["c"]-o["tip"][1] if o["vector"][1] == 1
                       else o["tip"][1]-b["c"]-b["w"]+1, j)
                      for j, b in enumerate(objects)
                      if b is not o and b["r"] <= o["tip"][0] < b["r"]+b["h"]
                      and (b["c"] > p[1] if o["vector"][1] == 1
                           else b["c"]+b["w"]-1 < p[1])]
        if candidates:
            j = min(candidates)[1]
            if commands[j] == 0 and j not in events["locks"]:
                nearest[j] = o["vector"][1]
    short = [max(0, min(g.shape[1]-o["w"], o["c"]+DISTANCE*d))
             for o, d in zip(objects, commands)]
    edge = [o["c"] if d == 0 else 0 if d == -1 else g.shape[1]-o["w"]
            for o, d in zip(objects, commands)]
    return dict(
        own_arrows=different([o["vector"][1] for o in objects]),
        total_hits_lock=different(total_hit_locks),
        remove_locks=different(no_collision_locks),
        stop_locked_firing=different(_commands(objects, ignore=events["locks"])[0]),
        first_object_ray=different(nearest),
        fixed_five_motion=not np.array_equal(expected, _paint(g.shape, bg, objects, short)),
        edge_only=not np.array_equal(expected, _paint(g.shape, bg, objects, edge)),
        unstable_repacking=not np.array_equal(
            expected, _paint(g.shape, bg, objects, _edge_first(objects, g.shape[1], commands))))


def self_test():
    """Stream 42 puzzles, meaningful mutations, counts, and geometric coverage."""
    fingerprints, rows = set(), []
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        omissions = [_counterfactuals(pair) for pair in puzzle["train"]]
        assert all(any(row[k] for row in omissions) for k in omissions[0]), omissions
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
                rows.append(measure_pair(pair))
    assert len(fingerprints) == 42
    assert {r["axis"] for r in rows} == {0, 1}
    assert len({r["shape"] for r in rows}) >= 50
    assert len({r["count"] for r in rows}) >= 5
    assert len({r["footprints"] for r in rows}) >= 30
    assert len({r["arrow_offsets"] for r in rows}) >= 30
    assert any(r["count"] <= 3 for r in rows)
    assert any(r["count"] >= 8 for r in rows)
    assert any(r["locks"] == 0 for r in rows) and any(r["locks"] > 0 for r in rows)
    assert any(r["multi_single"] for r in rows)
    assert any(r["crossings"] for r in rows)
    assert any(r["anchored_stops"] for r in rows)
    assert any(r["stable_witness"] for r in rows)
    assert any(r["long_misses"] for r in rows)
    assert max(t for r in rows for t in r["travel"]) >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            generate_puzzle(93, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # Collision locations, not total object hits, select locks.
    for pair in p["train"]:
        g, bg, objects, _ = _canonical(pair["input"])
        commands, events = _commands(objects)
        columns = _settle(objects, g.shape[1], commands)
        if events["locks"]:
            assert any(len(points) >= 2 and j not in events["locks"]
                       and columns[j] != objects[j]["c"] for j, points in events["sites"].items())
        assert _simulate(objects, g.shape[1], commands) == columns
    # Reject the concrete same-class overtaking label, not just random damage.
    for i, pair in enumerate(p["train"]):
        if not measure_pair(pair)["stable_witness"]:
            continue
        g, bg, objects, axis = _canonical(pair["input"])
        commands, _ = _commands(objects)
        wrong = _paint(g.shape, bg, objects, _edge_first(objects, g.shape[1], commands))
        broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        broken["train"][i]["output"] = wrong if axis == 1 else wrong.T.copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("Same-class overtaking label accepted")
        break
    return dict(seeds=42, pairs=len(rows), distinct_puzzles=len(fingerprints),
                shapes=len({r["shape"] for r in rows}),
                counts=sorted({r["count"] for r in rows}),
                footprint_sets=len({r["footprints"] for r in rows}))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
