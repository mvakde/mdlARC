"""Parity-lit chambers: hollow diamond beams vote on repaired compartments.

All interpretation in solve() comes from the supplied pixels. Generation uses
clean walls to label pairs, then checks exact reconstruction of those walls.
Only Python's standard library and NumPy are needed.
"""
from collections import deque
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _outline(center, radius):
    y, x = center
    return [(y + a, x + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius]


def _cap(center, radius, direction):
    y, x = center
    dy, dx = direction
    return [(y + dy * (k + 1), x + dx * (radius - k + 1))
            for k in range(radius + 1)]


def _footprint(e):
    return set(_outline(e["center"], e["radius"])) | {
        p for d in e["directions"] for p in _cap(e["center"], e["radius"], d)}


def _in(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _background(a):
    parity = np.indices(a.shape).sum(axis=0) % 2
    colors = []
    for k in (0, 1):
        counts = np.bincount(a[parity == k], minlength=10)
        c = int(counts.argmax())
        if counts[c] * 2 <= np.count_nonzero(parity == k):
            raise ValueError("checkerboard parity lacks a majority background")
        colors.append(c)
    if colors[0] == colors[1]:
        raise ValueError("checkerboard needs two colors")
    return tuple(colors)


def _emitters(a, bg):
    h, w = a.shape
    emitters = []
    for radius in (2, 3):
        if min(h, w) < 2 * radius + 1:
            continue
        sy, sx = h - 2 * radius, w - 2 * radius
        offsets = _outline((0, 0), radius)
        dy, dx = offsets[0]
        colors = a[radius + dy:radius + dy + sy, radius + dx:radius + dx + sx]
        good = ~np.isin(colors, bg)
        for dy, dx in offsets[1:]:
            good &= a[radius + dy:radius + dy + sy, radius + dx:radius + dx + sx] == colors
        for yy, xx in np.argwhere(good):
            center = (int(yy) + radius, int(xx) + radius)
            color = int(colors[yy, xx])
            cells = _outline(center, radius)
            # Diamond outlines have diagonal adjacency, whereas walls retain
            # orthogonal connections. This excludes accidental wall diamonds.
            if any(_in(q, a.shape) and int(a[q]) == color for y, x in cells
                   for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))):
                continue
            emitters.append(dict(center=center, radius=radius, struct=color, directions=[]))
    if not 1 <= len(emitters) <= 3 or len({e["struct"] for e in emitters}) != 1:
        raise ValueError("expected 1..3 complete diamonds with a common outline color")
    covered, inks = set(), set()
    for e in emitters:
        cells = set(_outline(e["center"], e["radius"]))
        if covered & cells:
            raise ValueError("overlapping outlines")
        covered |= cells
    for e in emitters:
        for d in DIRECTIONS:
            cells = _cap(e["center"], e["radius"], d)
            if not all(_in(p, a.shape) for p in cells):
                continue
            values = {int(a[p]) for p in cells}
            if len(values) == 1 and not values & set((*bg, e["struct"])):
                if covered & set(cells):
                    raise ValueError("overlapping caps")
                covered |= set(cells)
                e["directions"].append(d)
                inks |= values
        if not e["directions"]:
            raise ValueError("diamond has no active cap")
    if len(inks) != 1:
        raise ValueError("all caps must share one local ink color")
    struct = emitters[0]["struct"]
    ink = next(iter(inks))
    actual = set(map(tuple, np.argwhere(np.isin(a, (struct, ink)))))
    if actual != covered:
        raise ValueError("unexplained diamond or cap pixels")
    emitters.sort(key=lambda e: e["center"])
    return emitters, struct, ink, covered


def _repair(a, wall, noise):
    mask = a == wall
    # Repair only one- or two-pixel runs of the visibly distinct damage marker,
    # bracketed by intact wall pixels on a straight axis. Never bridge background.
    for axis in (0, 1):
        field = a if axis == 0 else a.T
        repaired = mask if axis == 0 else mask.T
        for y in range(field.shape[0]):
            for x in range(field.shape[1] - 2):
                if field[y, x] != wall:
                    continue
                for gap in (1, 2):
                    end = x + gap + 1
                    if (end < field.shape[1] and field[y, end] == wall
                            and np.all(field[y, x+1:end] == noise)):
                        repaired[y, x+1:end] = True
    return mask


def _regions(walls, rectangular=True):
    """Label bounded four-connected interiors; symbols are transparent here."""
    free = np.pad(~walls, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    labels = np.full(walls.shape, -1, dtype=np.int16)
    rooms = []
    for yy, xx in np.argwhere(free):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        queue = deque([(yy, xx)])
        seen[yy, xx] = True
        cells, outside = [], False
        while queue:
            y, x = queue.popleft()
            cells.append((y-1, x-1))
            outside |= y in (0, free.shape[0]-1) or x in (0, free.shape[1]-1)
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                r, c = p
                if _in(p, free.shape) and free[p] and not seen[p]:
                    seen[p] = True
                    queue.append((r, c))
        if outside:
            continue
        rows, cols = zip(*cells)
        box = (min(rows), min(cols), max(rows), max(cols))
        if rectangular and len(cells) != (box[2]-box[0]+1) * (box[3]-box[1]+1):
            raise ValueError("nonrectangular compartment")
        index = len(rooms)
        for p in cells:
            labels[p] = index
        rooms.append(box)
    return labels, rooms


def _walls_from_boxes(shape, boxes):
    walls = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        walls[t, l:r+1] = walls[b, l:r+1] = True
        walls[t:b+1, l] = walls[t:b+1, r] = True
    return walls


def _analyze(a):
    bg = _background(a)
    bg_pixels = np.isin(a, bg)
    if np.any(a[bg_pixels] != _base(a.shape, bg)[bg_pixels]):
        raise ValueError("background pixels must follow the checkerboard phase")
    emitters, struct, ink, protected = _emitters(a, bg)
    remaining = sorted(set(map(int, np.unique(a))) - set((*bg, struct, ink)))
    if len(remaining) != 2:
        raise ValueError("expected a wall color and a visible damage color")
    choices = []
    for wall, noise in (remaining, remaining[::-1]):
        walls = _repair(a, wall, noise)
        try:
            labels, rooms = _regions(walls)
        except ValueError:
            continue
        if not rooms:
            continue
        boxes = [(t-1, l-1, b+1, r+1) for t, l, b, r in rooms]
        if not np.array_equal(walls, _walls_from_boxes(a.shape, boxes)):
            continue
        if any(walls[p] for p in protected):
            continue
        choices.append((wall, noise, walls, labels, rooms))
    if len(choices) != 1:
        raise ValueError("wall/damage roles are not uniquely recoverable")
    wall, noise, walls, labels, rooms = choices[0]
    return dict(background=bg, emitters=emitters, struct=struct, ink=ink,
                protected=protected, wall=wall, noise=noise, walls=walls,
                labels=labels, rooms=rooms)


def _votes(labels, emitters, mode="endpoints", walls=None):
    count = int(labels.max()) + 1
    per_emitter = np.zeros((len(emitters), count), dtype=np.int16)
    ray_hits = []
    for ei, e in enumerate(emitters):
        for d in e["directions"]:
            cap = _cap(e["center"], e["radius"], d)
            starts = cap if mode == "whole_cap" else (cap[0], cap[-1])
            for p in starts:
                visited = [] if mode == "pixels" else set()
                y, x = p[0] + d[0], p[1] + d[1]
                while _in((y, x), labels.shape):
                    if mode == "stop" and walls[y, x]:
                        break
                    room = int(labels[y, x])
                    if room >= 0:
                        if mode == "pixels":
                            visited.append(room)
                        else:
                            visited.add(room)
                    y, x = y+d[0], x+d[1]
                for room in visited:
                    per_emitter[ei, room] += 1
                ray_hits.append((d, tuple(sorted(visited))))
    return per_emitter.sum(axis=0), per_emitter, ray_hits


def _selection(labels, votes):
    selected = np.zeros(labels.shape, dtype=bool)
    valid = labels >= 0
    if len(votes):
        selected[valid] = (votes[labels[valid]] % 2) == 1
    return selected


def _paint(a, analysis, votes):
    out = _base(a.shape, analysis["background"])
    out[_selection(analysis["labels"], votes)] = analysis["ink"]
    out[analysis["walls"]] = analysis["wall"]
    for p in analysis["protected"]:
        out[p] = a[p]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    analysis = _analyze(a)
    votes, _, _ = _votes(analysis["labels"], analysis["emitters"])
    return _paint(a, analysis, votes)


def _split_frames(rng, shape, number):
    """Disjoint irregularly sized boxes, rather than a fixed object layout."""
    h, w = shape
    slots = [(0, 0, h-1, w-1)]
    for _ in range(number-1):
        options = []
        for i, (t, l, b, r) in enumerate(slots):
            for axis, lo, hi in ((0, t, b), (1, l, r)):
                for cut in range(lo+9, hi-8):
                    halves = ((t, l, cut-1, r), (cut+1, l, b, r)) if axis == 0 else (
                        (t, l, b, cut-1), (t, cut+1, b, r))
                    options.append((i, halves))
        if not options:
            return None
        i, halves = options[int(rng.integers(len(options)))]
        slots[i:i+1] = halves
    boxes = []
    for t, l, b, r in slots:
        # Margins vary independently; complete frames may meet the canvas edge.
        margins = list(map(int, rng.integers(0, 3, size=4)))
        box = (t+margins[0], l+margins[1], b-margins[2], r-margins[3])
        if min(box[2]-box[0], box[3]-box[1]) < 5:
            return None
        boxes.append(box)
    return boxes


def _line_cuts(rng, start, end, parts, allowed):
    if parts == 1:
        return [start, end]
    for _ in range(20):
        candidates = list(allowed)
        rng.shuffle(candidates)
        cuts = [start, end]
        for cut in candidates:
            if all(abs(cut-old) >= 2 for old in cuts):
                cuts.append(cut)
                if len(cuts) == parts+1:
                    return sorted(cuts)
    return None


def _partition(rng, shape, outer_boxes, target, protected, style="recursive"):
    if style != "recursive":
        counts = [1] * len(outer_boxes)
        for _ in range(target-len(counts)):
            counts[int(rng.integers(len(counts)))] += 1
        rooms = []
        for (t, l, b, r), count in zip(outer_boxes, counts):
            factors = [(ny, count//ny) for ny in range(1, count+1) if count % ny == 0
                       and (style == "grid" or ny == 1 or ny == count)]
            rng.shuffle(factors)
            allowed_y = [y for y in range(t+2, b-1)
                         if not any((y, x) in protected for x in range(l, r+1))]
            allowed_x = [x for x in range(l+2, r-1)
                         if not any((y, x) in protected for y in range(t, b+1))]
            for ny, nx in factors:
                ys = _line_cuts(rng, t, b, ny, allowed_y)
                xs = _line_cuts(rng, l, r, nx, allowed_x)
                if ys is not None and xs is not None:
                    rooms.extend((ys[i], xs[j], ys[i+1], xs[j+1])
                                 for i in range(ny) for j in range(nx))
                    break
            else:
                return None
        return _walls_from_boxes(shape, rooms)
    rooms = list(outer_boxes)
    while len(rooms) < target:
        options = []
        for i, (t, l, b, r) in enumerate(rooms):
            for y in range(t+2, b-1):
                if not any((y, x) in protected for x in range(l, r+1)):
                    options.append((i, ((t, l, y, r), (y, l, b, r))))
            for x in range(l+2, r-1):
                if not any((y, x) in protected for y in range(t, b+1)):
                    options.append((i, ((t, l, b, x), (t, x, b, r))))
        if not options:
            return None
        i, halves = options[int(rng.integers(len(options)))]
        rooms[i:i+1] = halves
    return _walls_from_boxes(shape, rooms)


def _wall_components(walls):
    pending = set(map(tuple, np.argwhere(walls)))
    components = []
    while pending:
        cells, stack = [], [pending.pop()]
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        components.append(cells)
    return components


def _euler_rooms(walls):
    edges = int(np.count_nonzero(walls[:-1] & walls[1:]))
    edges += int(np.count_nonzero(walls[:, :-1] & walls[:, 1:]))
    unit_faces = int(np.count_nonzero(walls[:-1, :-1] & walls[1:, :-1]
                                     & walls[:-1, 1:] & walls[1:, 1:]))
    return edges-int(walls.sum())+len(_wall_components(walls))-unit_faces


def _topologies(analysis):
    signatures = []
    for cells in _wall_components(analysis["walls"]):
        ys, xs = zip(*cells)
        top, left, bottom, right = min(ys), min(xs), max(ys), max(xs)
        boxes = [(t-1, l-1, b+1, r+1) for t, l, b, r in analysis["rooms"]
                 if top < t <= b < bottom and left < l <= r < right]
        yranks = {y: i for i, y in enumerate(sorted({y for t, _, b, _ in boxes for y in (t, b)}))}
        xranks = {x: i for i, x in enumerate(sorted({x for _, l, _, r in boxes for x in (l, r)}))}
        signatures.append(tuple(sorted((yranks[t], xranks[l], yranks[b], xranks[r]) for t, l, b, r in boxes)))
    return signatures


def _partial(walls):
    # A genuine T-junction has three orthogonal wall neighbors.
    degree = np.zeros(walls.shape, dtype=np.int8)
    degree[:-1] += walls[1:]
    degree[1:] += walls[:-1]
    degree[:, :-1] += walls[:, 1:]
    degree[:, 1:] += walls[:, :-1]
    for cells in _wall_components(walls):
        ys, xs = zip(*cells)
        t, l, b, r = min(ys), min(xs), max(ys), max(xs)
        if np.any((walls & (degree == 3))[t+1:b, l+1:r]):
            return True
    return False


def _lattice(walls):
    """Wrong alternative: continue every divider across its entire frame."""
    result = np.zeros_like(walls)
    for cells in _wall_components(walls):
        ys, xs = zip(*cells)
        t, l, b, r = min(ys), min(xs), max(ys), max(xs)
        local = walls[t:b+1, l:r+1]
        hlines = np.flatnonzero(np.any(local[:, :-1] & local[:, 1:], axis=1)) + t
        vlines = np.flatnonzero(np.any(local[:-1] & local[1:], axis=0)) + l
        for y in hlines:
            result[y, l:r+1] = True
        for x in vlines:
            result[t:b+1, x] = True
    return result


def _evidence(a, analysis):
    labels = analysis["labels"]
    votes, per, hits = _votes(labels, analysis["emitters"])
    actual = _selection(labels, votes)
    ordinary = labels >= 0
    for p in analysis["protected"]:
        ordinary[p] = False
    def differs(other_labels, other_votes):
        return bool(np.any(ordinary & (actual != _selection(other_labels, other_votes))))
    full, _, _ = _votes(labels, analysis["emitters"], "whole_cap")
    stopped, _, _ = _votes(labels, analysis["emitters"], "stop", analysis["walls"])
    pixels, _, _ = _votes(labels, analysis["emitters"], "pixels")
    raw_labels, _ = _regions(a == analysis["wall"], rectangular=False)
    raw, _, _ = _votes(raw_labels, analysis["emitters"])
    lattice_labels, _ = _regions(_lattice(analysis["walls"]))
    lattice_votes, _, _ = _votes(lattice_labels, analysis["emitters"])
    odd_by_emitter = per % 2
    influential = set()
    for d in DIRECTIONS:
        es = [dict(e, directions=[v for v in e["directions"] if v == d])
              for e in analysis["emitters"]]
        contribution, _, _ = _votes(labels, es)
        if np.any(_selection(labels, contribution) & ordinary):
            influential.add(d)
    ray_pixels = np.zeros(a.shape, dtype=bool)
    for e in analysis["emitters"]:
        for d in e["directions"]:
            cap = _cap(e["center"], e["radius"], d)
            for endpoint in (cap[0], cap[-1]):
                y, x = endpoint[0]+d[0], endpoint[1]+d[1]
                while _in((y, x), a.shape):
                    ray_pixels[y, x] = True
                    y, x = y+d[0], x+d[1]
    return dict(odd=bool(np.any(votes % 2)), even=bool(np.any((votes > 0) & (votes % 2 == 0))),
                higher_odd=bool(np.any(ordinary & _selection(labels, ((votes >= 3) & (votes % 2 == 1)).astype(int)))),
                unhit=bool(np.any(votes == 0)),
                endpoints=differs(labels, full),
                transparent=differs(labels, stopped),
                once=differs(labels, pixels),
                repair=differs(raw_labels, raw),
                compartments=differs(lattice_labels, lattice_votes),
                room_fill=bool(np.any(actual & ordinary & ~ray_pixels)),
                superposition=bool(np.any((odd_by_emitter.sum(axis=0) >= 2) & (votes % 2 == 0))),
                partial=_partial(analysis["walls"]),
                directions=influential,
                crossing=any(len(rooms) >= 2 for _, rooms in hits))


def _sample_pair(rng, parameters, n, target, witness=0):
    """Bounded rejection sampling of freely positioned probes and partitions."""
    for attempt in range(1200):
        low = 18 if n == 1 else 22
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        frames = int(rng.choice([1, 1, 1, 1, 2, 2, 3]))
        outer = _split_frames(rng, shape, min(frames, target))
        if outer is None:
            continue
        outer_walls = _walls_from_boxes(shape, outer)
        emitters, protected = [], set()
        for i in range(n):
            radius = parameters["radius"] if witness else int(rng.integers(2, parameters["radius"]+1))
            mask = 15 if witness == 1 else int(rng.integers(1, 16))
            directions = [d for j, d in enumerate(DIRECTIONS) if mask & (1 << j)]
            offsets = _footprint(dict(center=(0, 0), radius=radius, directions=directions))
            for _ in range(120):
                y = int(rng.integers(radius+1, shape[0]-radius-1))
                x = int(rng.integers(radius+1, shape[1]-radius-1))
                if any(abs(y-old["center"][0])+abs(x-old["center"][1])
                       < radius+old["radius"]+5 for old in emitters):
                    continue
                fp = {(y+dy, x+dx) for dy, dx in offsets}
                if fp & protected or any(outer_walls[p] for p in fp):
                    continue
                e = dict(center=(y, x), radius=radius, directions=directions)
                break
            else:
                break
            emitters.append(e)
            protected |= fp
        if len(emitters) != n:
            continue
        style = "recursive" if witness else str(rng.choice(["recursive", "recursive", "strip", "grid"]))
        walls = _partition(rng, shape, outer, target, protected, style)
        if walls is None:
            continue
        colors = parameters["colors"] if parameters["fixed"] else tuple(map(int, rng.choice(10, 6, replace=False)))
        bg, wall, struct, ink, noise = colors[:2], colors[2], colors[3], colors[4], colors[5]
        a = _base(shape, bg)
        a[walls] = wall
        for e in emitters:
            for p in _outline(e["center"], e["radius"]):
                a[p] = struct
            for d in e["directions"]:
                for p in _cap(e["center"], e["radius"], d):
                    a[p] = ink
        try:
            _background(a)
        except ValueError:
            continue
        # Damage straight segments only. Both supported gap lengths are taught.
        options = []
        for y, x in np.argwhere(walls):
            y, x = int(y), int(x)
            for d in ((0, 1), (1, 0)):
                for length in (1, 2):
                    cells = [(y+d[0]*k, x+d[1]*k) for k in range(length)]
                    before, after = (y-d[0], x-d[1]), (y+d[0]*length, x+d[1]*length)
                    if not _in(before, shape) or not _in(after, shape) or not walls[before] or not walls[after]:
                        continue
                    if not all(_in(p, shape) and walls[p] for p in cells):
                        continue
                    # Avoid all corners/junctions, including accidental adjacent
                    # walls. Leave a supported straight line on either side.
                    perp = (1, 0) if d == (0, 1) else (0, 1)
                    if any(_in(q, shape) and walls[q] for p in cells for q in (
                            (p[0]-perp[0], p[1]-perp[1]), (p[0]+perp[0], p[1]+perp[1]))):
                        continue
                    options.append(cells)
        if not options:
            continue
        rng.shuffle(options)
        chosen = 0
        gap_lengths = set()
        for cells in options:
            if any(a[p] != wall for p in cells):
                continue
            trial = a.copy()
            for p in cells:
                trial[p] = noise
            if not np.array_equal(_repair(trial, wall, noise), walls):
                continue
            a = trial
            chosen += 1
            gap_lengths.add(len(cells))
            if chosen >= int(parameters["damage"]):
                break
        if chosen == 0 or (witness and gap_lengths != {1, 2}):
            continue
        # Stray markers must not invent a divider across an intentionally open
        # interval. Acceptance checks the exact clean wall mask.
        sites = [tuple(map(int, p)) for p in np.argwhere(~walls)
                 if tuple(map(int, p)) not in protected]
        if sites:
            for j in rng.choice(len(sites), min(len(sites), int(rng.integers(1, 8))), replace=False):
                a[sites[int(j)]] = noise
        try:
            analysis = _analyze(a)
        except ValueError:
            continue
        if not np.array_equal(analysis["walls"], walls) or len(analysis["rooms"]) != target:
            continue
        intended = sorted([(e["center"], e["radius"], e["directions"]) for e in emitters])
        actual = [(e["center"], e["radius"], e["directions"]) for e in analysis["emitters"]]
        if intended != actual:
            continue
        labels, rooms = _regions(walls)
        clean_analysis = dict(analysis, labels=labels, rooms=rooms, walls=walls)
        votes, _, _ = _votes(labels, emitters)
        out = _paint(a, clean_analysis, votes)
        evidence = _evidence(a, analysis)
        if not evidence["odd"] or not evidence["even"]:
            continue
        if witness:
            required = ("endpoints", "transparent", "repair", "partial", "crossing", "once", "compartments", "room_fill")
            if not all(evidence[k] for k in required):
                continue
            if witness == 1 and evidence["directions"] != set(DIRECTIONS):
                continue
            if witness == 2 and not (evidence["superposition"] and evidence["higher_odd"]):
                continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded chamber/probe construction exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = dict(colors=tuple(map(int, rng.choice(10, 6, replace=False))),
                      fixed=bool(rng.random() < .35), radius=int(rng.integers(2, 4)),
                      damage=int(rng.integers(3, 7)))
    shift = str(rng.choice(["emitters", "rooms"]))
    train_ceiling = int(rng.integers(5, 8))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            witness = i+1 if split == "train" and i < 2 else 0
            n = witness if witness else int(rng.integers(1, 3))
            if split == "test" and shift == "emitters":
                n = 3
            target = int(rng.integers(4, train_ceiling+1))
            if split == "test" and shift == "rooms":
                target = int(rng.integers(train_ceiling+1, train_ceiling+4))
            for _ in range(12):
                pair = _sample_pair(rng, parameters, n, target, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _interval_votes(rooms, emitters, shape):
    """Closed-form ray/rectangle intersections, independent of pixel walking."""
    votes = np.zeros(len(rooms), dtype=np.int16)
    for e in emitters:
        y, x = e["center"]
        radius = e["radius"]
        for dy, dx in e["directions"]:
            for py, px in ((y+dy, x+dx*(radius+1)), (y+dy*(radius+1), x+dx)):
                for i, (t, l, b, r) in enumerate(rooms):
                    ylo, yhi = (t-py, b-py) if dy == 1 else (py-b, py-t)
                    xlo, xhi = (l-px, r-px) if dx == 1 else (px-r, px-l)
                    if max(1, ylo, xlo) <= min(yhi, xhi):
                        votes[i] += 1
    return votes


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_evidence, train_radius, train_gap_lengths = set(), [], 0, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and not np.array_equal(a, out)
            assert _key(a) not in seen
            seen.add(_key(a))
            analysis = _analyze(a)
            assert _euler_rooms(analysis["walls"]) == len(analysis["rooms"])
            votes, _, _ = _votes(analysis["labels"], analysis["emitters"])
            assert np.array_equal(votes, _interval_votes(analysis["rooms"], analysis["emitters"], a.shape))
            assert np.array_equal(out, _paint(a, analysis, votes))
            assert np.all(out[analysis["walls"]] == analysis["wall"])
            assert analysis["noise"] not in out
            damage = analysis["walls"] & (a == analysis["noise"])
            gaps = _wall_components(damage)
            assert gaps and all(len(cells) <= 2 for cells in gaps)
            for cells in gaps:
                assert len({p[0] for p in cells}) == 1 or len({p[1] for p in cells}) == 1
                for y, x in cells:
                    neighbors = [p for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                                 if _in(p, a.shape) and analysis["walls"][p]]
                    assert len(neighbors) == 2
                    assert neighbors[0][0] == neighbors[1][0] or neighbors[0][1] == neighbors[1][1]
            assert all(out[p] == a[p] for p in analysis["protected"])
            selected = _selection(analysis["labels"], votes)
            free = analysis["labels"] >= 0
            for p in analysis["protected"]:
                free[p] = False
            assert np.all(out[free & selected] == analysis["ink"])
            base = _base(a.shape, analysis["background"])
            assert np.array_equal(out[free & ~selected], base[free & ~selected])
            evidence = _evidence(a, analysis)
            assert evidence["odd"] and evidence["even"]
            if split == "train":
                train_evidence.append(evidence)
                train_gap_lengths.update(len(cells) for cells in gaps)
                train_radius = max(train_radius, max(e["radius"] for e in analysis["emitters"]))
            else:
                assert max(e["radius"] for e in analysis["emitters"]) <= train_radius
    for concept in ("endpoints", "transparent", "repair", "partial", "crossing", "superposition", "higher_odd", "once", "compartments", "room_fill"):
        assert any(e[concept] for e in train_evidence), "missing teaching witness: " + concept
    assert set().union(*(e["directions"] for e in train_evidence)) == set(DIRECTIONS)
    assert train_gap_lengths == {1, 2}


def self_test() -> dict:
    """Bounded semantic, equivariance, count, and 42-seed variation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, geometries, room_counts, emitter_counts, radii, masks, palettes = (set() for _ in range(7))
    odd_counts, frame_counts, partials, cross_cancel = set(), set(), 0, 0
    topologies, incidences, room_sizes = set(), set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _grid(pair["input"])
                analysis = _analyze(a)
                shapes.add(a.shape)
                rooms = tuple(analysis["rooms"])
                es = tuple((e["center"], e["radius"], tuple(e["directions"])) for e in analysis["emitters"])
                geometries.add((a.shape, rooms, es))
                room_counts.add(len(rooms))
                emitter_counts.add(len(es))
                radii.update(e["radius"] for e in analysis["emitters"])
                masks.update(sum(1 << DIRECTIONS.index(d) for d in e["directions"]) for e in analysis["emitters"])
                palettes.add((*analysis["background"], analysis["wall"], analysis["struct"], analysis["ink"], analysis["noise"]))
                votes, _, _ = _votes(analysis["labels"], analysis["emitters"])
                topologies.update(_topologies(analysis))
                _, per_emitter, _ = _votes(analysis["labels"], analysis["emitters"])
                incidences.add(tuple(map(tuple, per_emitter.tolist())))
                room_sizes.update((b-t+1, r-l+1) for t, l, b, r in rooms)
                odd_counts.add(int(np.count_nonzero(votes % 2)))
                partials += _partial(analysis["walls"])
                cross_cancel += _evidence(a, analysis)["superposition"]
                # Count connected wall networks without referring to sampler.
                frame_counts.add(len(_wall_components(analysis["walls"])))
    assert len(shapes) >= 35 and len(geometries) >= 180
    assert emitter_counts == {1, 2, 3} and radii == {2, 3}
    assert {1, 2}.issubset(frame_counts) and len(room_counts) >= 6
    assert len(masks) >= 12 and len(palettes) >= 40
    assert len(topologies) >= 20 and len(incidences) >= 100
    assert min(min(s) for s in room_sizes) == 1 and max(max(s) for s in room_sizes) >= 20
    assert len(odd_counts) >= 4 and 84 <= partials < 200 and cross_cancel >= 42, (odd_counts, partials, cross_cancel)
    repeated = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in puzzles[0]
               for a, b in zip(puzzles[0][s], repeated[s]) for k in a)
    # Geometry and role colors are intrinsic, not hidden compass/color codes.
    recolor = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    for puzzle in puzzles[:3]:
        for pair in puzzle["train"] + puzzle["test"]:
            a, out = pair["input"], pair["output"]
            assert np.array_equal(solve(recolor[a]), recolor[out])
            for transform in (np.rot90, np.fliplr, np.transpose):
                assert np.array_equal(solve(transform(a)), transform(out))
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(101+nt*5+ne, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    # Mutate a selected room's ordinary pixel: validation must reject the label.
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    pair = bad["test"][0]
    info = _analyze(pair["input"])
    candidates = [tuple(p) for p in np.argwhere((pair["output"] == info["ink"]) & (info["labels"] >= 0))
                  if tuple(p) not in info["protected"]]
    assert candidates
    p = candidates[0]
    pair["output"][p] = _base(pair["input"].shape, info["background"])[p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incompletely filled room")
    return dict(seeds=42, pairs=210, shapes=len(shapes), geometries=len(geometries),
                room_counts=sorted(room_counts), emitter_counts=sorted(emitter_counts),
                radii=sorted(radii), cap_subsets=len(masks), palettes=len(palettes),
                frame_counts=sorted(frame_counts), partial_pairs=partials,
                superposition_pairs=cross_cancel, partition_topologies=len(topologies),
                incidence_structures=len(incidences), room_sizes=len(room_sizes))
