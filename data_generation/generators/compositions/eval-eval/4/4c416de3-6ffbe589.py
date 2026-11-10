"""Phase-aligned union of distributed frame-corner stencils.

No parent code or data is imported. Generation uses NumPy flips/rotations;
the input-only solver uses coordinate arithmetic. All controls remain visible.
"""

from __future__ import annotations

import numpy as np


_SIGNS = ((1, 1), (1, -1), (-1, -1), (-1, 1))  # TL, TR, BR, BL
_RADIUS = 2


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _turn(point, count):
    u, v = point
    for _ in range(int(count) % 4):
        u, v = v, -u
    return u, v


def _forward(point, q, count, reflect=True):
    if reflect:
        sy, sx = _SIGNS[q]
        point = (sy * point[0], sx * point[1])
    else:
        point = _turn(point, q)
    return _turn(point, count)


def _inverse(point, q, count, reflect=True):
    point = _turn(point, -count)
    if reflect:
        sy, sx = _SIGNS[q]
        return sy * point[0], sx * point[1]
    return _turn(point, -q)


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, r, 2), (b, l, 3))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _nearest(point, corners):
    y, x = point
    distances = [(y - a) ** 2 + (x - b) ** 2 for a, b, _ in corners]
    minimum = min(distances)
    if distances.count(minimum) != 1:
        raise ValueError("ambiguous closest frame corner")
    return corners[distances.index(minimum)]


def _read_input(input_grid):
    """Segment all roles from the input, then recover the collective stencil."""
    inp = _grid(input_grid)
    frequencies = np.bincount(inp.ravel(), minlength=10)
    if np.count_nonzero(frequencies == frequencies.max()) != 1:
        raise ValueError("background must be uniquely dominant")
    bg = int(frequencies.argmax())
    possibilities = []
    for color in np.unique(inp):
        color = int(color)
        if color == bg:
            continue
        rects = []
        for component in _components(inp == color):
            rows, cols = zip(*component)
            rect = min(rows), min(cols), max(rows), max(cols)
            if (rect[2] - rect[0] < 4 or rect[3] - rect[1] < 4
                    or component != _outline(rect)):
                break
            rects.append(rect)
        else:
            possibilities.append((color, sorted(rects)))
    if len(possibilities) != 1:
        raise ValueError("expected one color of intact rectangular outlines")
    frame, rects = possibilities[0]
    if not 1 <= len(rects) <= 4:
        raise ValueError("expected 1..4 separated frames")
    for i, a in enumerate(rects):
        for b in rects[:i]:
            if not _separated(a, b, 2):
                raise ValueError("frame bounding boxes are not separated")
    corners = sorted(c for rect in rects for c in _corners(rect))
    assigned = {c: [] for c in corners}
    exterior = {}
    for y, x in np.argwhere((inp != bg) & (inp != frame)):
        p = int(y), int(x)
        color = int(inp[p])
        nearby = [c for c in corners if max(abs(p[0] - c[0]), abs(p[1] - c[1])) <= _RADIUS]
        if nearby:
            nearest = _nearest(p, corners)
            a, b, _ = nearest
            if p[0] == a or p[1] == b:
                raise ValueError("stencil points cannot lie on corner axes")
            assigned[nearest].append(p)
        else:
            exterior.setdefault(color, []).append(p)
    occurrences = []
    for corner, coords in assigned.items():
        if not coords:
            continue
        colors = {int(inp[p]) for p in coords}
        if len(colors) != 1:
            raise ValueError("a corner must have one accent color")
        a, b, _ = corner
        occurrences.append({"corner": corner, "color": colors.pop(),
                            "offsets": frozenset((y - a, x - b) for y, x in coords),
                            "coords": frozenset(coords)})
    if not 2 <= len(occurrences) <= 16:
        raise ValueError("expected at least two marked corners")
    accents = {o["color"] for o in occurrences}
    if not 1 <= len(accents) <= 4:
        raise ValueError("expected 1..4 accent colors")
    counts, bars = {}, {}
    for color, coords in exterior.items():
        rows, cols = zip(*coords)
        horizontal = max(rows) == min(rows) and max(cols) - min(cols) + 1 == len(coords)
        vertical = max(cols) == min(cols) and max(rows) - min(rows) + 1 == len(coords)
        if color not in accents or not 1 <= len(coords) <= 9 or not (horizontal or vertical):
            raise ValueError("exterior color must be a single matching unary bar")
        for y, x in coords:
            if any(t - 3 <= y <= b + 3 and l - 3 <= x <= r + 3 for t, l, b, r in rects):
                raise ValueError("bar is too close to a frame")
        counts[color] = len(coords)
        bars[color] = {"coords": frozenset(coords), "length": len(coords),
                       "orientation": "single" if len(coords) == 1 else
                       ("horizontal" if horizontal else "vertical")}
    if not bars:
        raise ValueError("expected at least one counter")
    bar_items = list(bars.items())
    for i, (_, bar) in enumerate(bar_items):
        for _, other in bar_items[:i]:
            if any(max(abs(y - a), abs(x - b)) <= 1
                   for y, x in bar["coords"] for a, b in other["coords"]):
                raise ValueError("bars must have a background gap")
    info = {"input": inp, "background": bg, "frame": frame, "rectangles": rects,
            "corners": corners, "occurrences": occurrences, "counts": counts, "bars": bars}
    normalized = [_normalized(o, counts) for o in occurrences]
    stencil = frozenset().union(*normalized)
    if not 2 <= len(stencil) <= 14:
        raise ValueError("collective stencil must have 2..14 points")
    if any(u == 0 or v == 0 or max(abs(u), abs(v)) > 2 for u, v in stencil):
        raise ValueError("invalid collective stencil offsets")
    info["stencil"], info["normalized"] = stencil, normalized
    return info


def _normalized(occurrence, counts, direction=1, reflect=True, absent=0):
    k = direction * counts.get(occurrence["color"], absent)
    return frozenset(_inverse(p, occurrence["corner"][2], k, reflect)
                     for p in occurrence["offsets"])


def _prediction(info, *, direction=1, reflect=True, absent=0, ignore_counts=False,
                per_color=False, largest=False, rank=False, check=True):
    """Coordinate renderer; optional countermodels are used only for validation."""
    counts = {} if ignore_counts else dict(info["counts"])
    if rank:
        ranks = {k: j + 1 for j, k in enumerate(sorted(set(counts.values())))}
        counts = {c: ranks[k] for c, k in counts.items()}
    occurrences = info["occurrences"]
    pieces = [_normalized(o, counts, direction, reflect, absent) for o in occurrences]
    stencil = max(pieces, key=lambda s: (len(s), tuple(sorted(s)))) if largest else frozenset().union(*pieces)
    out = info["input"].copy()
    h, w = out.shape
    occupied, targets = set(), []
    for occurrence in occurrences:
        a, b, q = occurrence["corner"]
        color = occurrence["color"]
        pattern = stencil
        if per_color:
            pattern = frozenset().union(*(s for o, s in zip(occurrences, pieces) if o["color"] == color))
        k = direction * counts.get(color, absent)
        positions = {(a + u, b + v) for u, v in (_forward(p, q, k, reflect) for p in pattern)}
        positions = {(y, x) for y, x in positions if 0 <= y < h and 0 <= x < w}
        if check:
            if positions & occupied:
                raise ValueError("completed corner constellations overlap")
            if not occurrence["coords"] <= positions:
                raise ValueError("completion lost clue pixels")
            for p in positions:
                if _nearest(p, info["corners"]) != occurrence["corner"]:
                    raise ValueError("completed cell has another closest corner")
                if int(out[p]) not in (info["background"], color):
                    raise ValueError("completion conflicts with another object")
        occupied.update(positions)
        targets.append(frozenset(positions))
        for p in positions:
            out[p] = color
    return out, targets


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover phases, local geometry and the union entirely from input cells."""
    return _prediction(_read_input(input_grid))[0]


def _sample_stencil(rng, radius, witness):
    domain = [(u, v) for u in range(-radius, radius + 1)
              for v in range(-radius, radius + 1) if u != 0 and v != 0]
    for _ in range(100):
        size = int(rng.integers(6 if witness else 2, min(12 if witness else 14, len(domain)) + 1))
        chosen = {domain[int(rng.integers(len(domain)))]}
        while len(chosen) < size:
            candidates = [p for p in domain if p not in chosen]
            # Grow small shards, with occasional jumps to another diagonal sector.
            weights = np.array([1 + 3 * sum(max(abs(p[0] - u), abs(p[1] - v)) == 1
                                           for u, v in chosen) for p in candidates], dtype=float)
            if rng.random() < .25:
                weights[:] = 1
            chosen.add(candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))])
        pattern = frozenset(chosen)
        if max(max(abs(u), abs(v)) for u, v in pattern) != radius:
            continue
        if witness and (pattern == frozenset((v, u) for u, v in pattern)
                        or any(pattern == frozenset(_turn(p, k) for p in pattern) for k in (1, 2, 3))):
            continue
        return pattern
    raise RuntimeError("failed to sample stencil")


def _separated(a, b, radius):
    gap = 2 * radius + 2
    return (a[2] + gap <= b[0] or b[2] + gap <= a[0]
            or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _sample_frames(rng, shape, radius, witness, rescue):
    if rescue:
        return [(5, 5, 18, 18)]
    h, w = shape
    desired = int(rng.choice((1, 1, 2, 2, 3, 4)))
    minimum = 2 * radius + 3
    margin = radius if witness else 0
    rects = []
    for _ in range(desired):
        for _ in range(60):
            cap = 10 if desired >= 3 else 18
            fh = int(rng.integers(minimum, min(cap, h - 2 * margin - 4) + 1))
            fw = int(rng.integers(minimum, min(cap, w - 2 * margin - 4) + 1))
            t = int(rng.integers(margin, h - margin - fh + 1))
            l = int(rng.integers(margin, w - margin - fw + 1))
            if not witness and rng.random() < .28:
                t = int(rng.choice((0, h - fh)))
            if not witness and rng.random() < .28:
                l = int(rng.choice((0, w - fw)))
            rect = t, l, t + fh - 1, l + fw - 1
            # Use radius two separation even for smaller constellations.
            if all(_separated(rect, other, 2) for other in rects):
                rects.append(rect)
                break
        else:
            break
    if not rects:
        raise ValueError("no frame fit")
    return rects


def _stamp_map(pattern, corner, count):
    """Construction uses an array transform, independent of solver coordinates."""
    labels = np.zeros((5, 5), dtype=np.int16)
    ordered = sorted(pattern)
    for label, (u, v) in enumerate(ordered, 1):
        labels[u + 2, v + 2] = label
    sy, sx = _SIGNS[corner[2]]
    if sy == -1:
        labels = np.flip(labels, 0)
    if sx == -1:
        labels = np.flip(labels, 1)
    labels = np.rot90(labels, -int(count) % 4)
    return {ordered[int(labels[y, x]) - 1]: (corner[0] + int(y) - 2, corner[1] + int(x) - 2)
            for y, x in np.argwhere(labels != 0)}


def _distributed_clues(rng, pattern, visible, witness):
    clues = [set() for _ in visible]
    private = {}
    if witness:
        points = sorted(pattern)
        order = rng.permutation(len(points))
        private = {points[int(j)]: i for i, j in enumerate(order[:3])}
        for p, i in private.items():
            clues[i].add(p)
    for p in sorted(pattern):
        choices = [i for i, view in enumerate(visible) if p in view]
        if not choices:
            raise ValueError("a stencil point has no visible donor")
        owner = private[p] if p in private else int(rng.choice(choices))
        clues[owner].add(p)
    for i, view in enumerate(visible):
        permitted = sorted(p for p in view if p not in private or private[p] == i)
        if not clues[i]:
            if not permitted:
                raise ValueError("corner has no retainable clue")
            clues[i].add(permitted[int(rng.integers(len(permitted)))])
        retain = float(rng.uniform(.12, .78))
        if not witness and rng.random() < .2:
            retain = 1.0
        clues[i].update(p for p in permitted if rng.random() < retain)
    return [frozenset(p) for p in clues]


def _put_bars(rng, inp, rects, counts, forced):
    h, w = inp.shape
    blocked = np.zeros(inp.shape, dtype=bool)
    for t, l, b, r in rects:
        blocked[max(0, t - 3):min(h, b + 4), max(0, l - 3):min(w, r + 4)] = True
    for color in sorted(counts, key=lambda c: (-counts[c], c)):
        length = counts[color]
        groups = []
        for orientation in ("horizontal", "vertical"):
            if color in forced and orientation != forced[color]:
                continue
            bh, bw = (1, length) if orientation == "horizontal" else (length, 1)
            rr, cc = np.indices((h - bh + 1, w - bw + 1))
            clear = np.ones(rr.shape, dtype=bool)
            for i in range(length):
                clear &= ~blocked[rr + (i if bh > 1 else 0), cc + (i if bw > 1 else 0)]
            starts = np.argwhere(clear)
            if len(starts):
                groups.append((orientation, starts))
        if not groups:
            raise ValueError("cannot fit detached counters")
        orientation, starts = groups[int(rng.integers(len(groups)))]
        y, x = map(int, starts[int(rng.integers(len(starts)))])
        for i in range(length):
            a = y + (i if orientation == "vertical" else 0)
            b = x + (i if orientation == "horizontal" else 0)
            inp[a, b] = color
            blocked[max(0, a - 1):min(h, a + 2), max(0, b - 1):min(w, b + 2)] = True


def _make_pair(rng, witness, testing, previous_pattern, rescue=False):
    radius = 2 if witness >= 0 or rng.random() < .65 else 1
    shape = (30, 30) if rescue else tuple(map(int, rng.integers(16, 31, size=2)))
    rects = _sample_frames(rng, shape, radius, witness >= 0, rescue)
    pattern = _sample_stencil(rng, radius, witness >= 0)
    if witness == 1 and pattern == previous_pattern:
        raise ValueError("initial demonstration stencils must differ")
    marked = []
    for i, rect in enumerate(rects):
        corners = _corners(rect)
        count = 3 if witness >= 0 and i == 0 else (0 if witness >= 0 else int(rng.integers(2 if i == 0 else 0, 5)))
        marked.extend(corners[int(j)] for j in rng.permutation(4)[:count])
    rng.shuffle(marked)
    color_count = 3 if witness >= 0 else int(rng.integers(1, min(4, len(marked)) + 1))
    palette = list(map(int, rng.choice(10, color_count + 2, replace=False)))
    bg, frame, colors = palette[0], palette[1], palette[2:]
    local_colors = colors.copy() + [int(rng.choice(colors)) for _ in range(len(marked) - color_count)]
    rng.shuffle(local_colors)
    counts = {color: int(rng.integers(0, 10 if testing else 5)) for color in colors}
    forced = {}
    if witness == 0:
        counts = dict(zip(local_colors, (1, 2, 2)))
        horizontal, vertical = ("horizontal", "vertical") if rng.random() < .5 else ("vertical", "horizontal")
        forced = {local_colors[1]: horizontal, local_colors[2]: vertical}
    elif witness == 1:
        counts = dict(zip(local_colors, (3, 4, 0)))
    elif testing:
        counts[int(rng.choice(colors))] = int(rng.integers(5, 10))
    if not any(counts.values()):
        counts[int(rng.choice(colors))] = int(rng.integers(1, 5))
    counts = {color: k for color, k in counts.items() if k}
    maps = [_stamp_map(pattern, corner, counts.get(color, 0))
            for corner, color in zip(marked, local_colors)]
    visible = [{p for p, (y, x) in mapping.items() if 0 <= y < shape[0] and 0 <= x < shape[1]}
               for mapping in maps]
    if any(not view for view in visible) or (witness >= 0 and any(view != pattern for view in visible)):
        raise ValueError("empty or truncated training witness")
    clues = _distributed_clues(rng, pattern, visible, witness >= 0)
    inp = np.full(shape, bg, dtype=np.int8)
    for rect in rects:
        for p in _outline(rect):
            inp[p] = frame
    out = inp.copy()
    occupied = set()
    for color, mapping, view, clue in zip(local_colors, maps, visible, clues):
        positions = {mapping[p] for p in view}
        if positions & occupied or any(out[p] != bg for p in positions):
            raise ValueError("conflicting constellation placements")
        occupied.update(positions)
        for p in view:
            out[mapping[p]] = color
        for p in clue:
            inp[mapping[p]] = color
    _put_bars(rng, inp, rects, counts, forced)
    # Controls are preserved in the output; they do not become donor cells.
    for color in counts:
        for p in map(tuple, np.argwhere(inp == color).tolist()):
            if p not in occupied:
                out[p] = color
    info = _read_input(inp)
    if info["stencil"] != pattern or not np.array_equal(_prediction(info)[0], out):
        raise ValueError("constructed scene is not recoverable")
    if witness >= 0:
        if len(pattern) <= max(map(len, info["normalized"])):
            raise ValueError("witness needs a distributed reference")
        for alternative in ({"ignore_counts": True}, {"direction": -1}, {"reflect": False}, {"per_color": True}):
            if np.array_equal(_prediction(info, check=False, **alternative)[0], out):
                raise ValueError("uninformative training witness")
    return {"input": inp, "output": out}, pattern


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen, previous_pattern = set(), None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = index if split == "train" and index < 2 else -1
            for attempt in range(160):
                try:
                    pair, pattern = _make_pair(rng, witness, split == "test", previous_pattern,
                                               rescue=attempt >= 100)
                    if _key(pair["input"]) in seen:
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded construction budget exhausted")
            seen.add(_key(pair["input"]))
            previous_pattern = pattern
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if _key(inp) in seen:
                raise ValueError("duplicate input")
            seen.add(_key(inp))
            info = _read_input(inp)
            prediction, targets = _prediction(info)
            if not np.array_equal(prediction, out):
                raise ValueError("wrong consensus completion")
            if inp.shape != out.shape or set(np.unique(inp)) != set(np.unique(out)):
                raise ValueError("canvas or palette changed")
            changed = inp != out
            if np.any(inp[changed] != info["background"]):
                raise ValueError("original content was changed")
            allowed = set().union(*targets)
            if not set(map(tuple, np.argwhere(changed).tolist())) <= allowed:
                raise ValueError("change outside a marked constellation")
            # Exact expected visible area, including clipped copies and controls.
            for color in {o["color"] for o in info["occurrences"]}:
                expected = sum(len(t) for o, t in zip(info["occurrences"], targets) if o["color"] == color)
                expected += info["counts"].get(color, 0)
                if int(np.count_nonzero(out == color)) != expected:
                    raise ValueError("wrong visible accent area")
            if not np.array_equal(solve(inp), out):
                raise ValueError("input-only solver disagrees")
            if split == "train":
                if max(info["counts"].values()) > 4:
                    raise ValueError("training counter exceeds four")
                training.append((info, out))
            elif max(info["counts"].values()) < 5:
                raise ValueError("test lacks the cyclic-count shift")
    for info, out in training[:2]:
        if (len(info["occurrences"]) != 3 or len({o["color"] for o in info["occurrences"]}) != 3
                or len(info["stencil"]) <= max(map(len, info["normalized"]))
                or not np.any(out != info["input"])):
            raise ValueError("initial training examples need incomplete cross-color witnesses")
        owners = [sum(p in piece for piece in info["normalized"]) for p in info["stencil"]]
        if owners.count(1) < 3:
            raise ValueError("initial evidence lacks private donor points")
    lengths = {k for info, _ in training for k in info["counts"].values()}
    if not {1, 2, 3, 4} <= lengths:
        raise ValueError("training must demonstrate counts 1..4")
    first = training[0][0]
    if {b["orientation"] for b in first["bars"].values() if b["length"] == 2} != {"horizontal", "vertical"}:
        raise ValueError("equal two-cell counts must have opposite bar orientations")
    alternatives = ({"ignore_counts": True}, {"direction": -1}, {"reflect": False},
                    {"per_color": True}, {"largest": True}, {"absent": 1}, {"rank": True})
    for alternative in alternatives:
        if all(np.array_equal(_prediction(info, check=False, **alternative)[0], out) for info, out in training):
            raise ValueError(f"training admits shortcut {alternative}")


def _features(pair):
    info = _read_input(pair["input"])
    _, targets = _prediction(info)
    pattern = info["stencil"]
    colors = [o["color"] for o in info["occurrences"]]
    return {"shape": info["input"].shape, "frames": len(info["rectangles"]),
            "frame_shapes": tuple((b - t + 1, r - l + 1) for t, l, b, r in info["rectangles"]),
            "positions": tuple((t, l) for t, l, _, _ in info["rectangles"]),
            "background": info["background"], "frame": info["frame"],
            "accent_colors": tuple(sorted(set(colors))), "marks": len(colors),
            "reused": len(set(colors)) < len(colors), "stencil": tuple(sorted(pattern)),
            "radius": max(max(abs(u), abs(v)) for u, v in pattern), "size": len(pattern),
            "intact": sum(piece == pattern for piece in info["normalized"]),
            "clipped": sum(len(t) < len(pattern) for t in targets),
            "unmarked": len(info["corners"]) - len(colors),
            "lengths": tuple(sorted(info["counts"].values())),
            "absent": len(set(colors)) - len(info["counts"]),
            "bar_orientations": tuple(b["orientation"] for b in info["bars"].values()),
            "symmetry": pattern == frozenset(_turn(p, 1) for p in pattern)}


def _boundary_regressions():
    """Explicit legal extremes beyond the randomly sampled 42-seed audit."""
    def fixture(shape, rects, pattern, colors, counts, bars, sparse):
        inp = np.full(shape, 0, dtype=np.int8)
        for rect in rects:
            for p in _outline(rect):
                inp[p] = 5
        out = inp.copy()
        for i, corner in enumerate(c for rect in rects for c in _corners(rect)):
            color = colors[i % len(colors)]
            mapping = _stamp_map(pattern, corner, counts.get(color, 0))
            visible = sorted(p for p, (y, x) in mapping.items() if 0 <= y < shape[0] and 0 <= x < shape[1])
            assert visible
            retained = [sorted(pattern)[i % len(pattern)]] if sparse else visible
            assert all(p in visible for p in retained)
            for p in visible:
                out[mapping[p]] = color
            for p in retained:
                inp[mapping[p]] = color
        for color, coords in bars.items():
            for p in coords:
                inp[p] = out[p] = color
        assert np.array_equal(solve(inp), out)
        assert np.array_equal(solve(out), out)
        return _read_input(inp)

    domain = {(u, v) for u in (-2, -1, 1, 2) for v in (-2, -1, 1, 2)}
    pattern = domain - {(-2, -2), (2, 1)}
    info = fixture((30, 30), [(3, 3, 9, 9), (3, 20, 9, 26),
                              (20, 3, 26, 9), (20, 20, 26, 26)],
                   pattern, [1, 2, 3, 4], {1: 1, 2: 4, 3: 8, 4: 9},
                   {1: [(25, 14)], 2: [(14, x) for x in range(20, 24)],
                    3: [(y, 14) for y in range(8)], 4: [(14, x) for x in range(9)]}, True)
    assert len(info["occurrences"]) == 16 and len(info["stencil"]) == 14
    assert max(map(len, info["normalized"])) == 1
    small = {(1, 1), (-1, 1), (1, -1)}
    info = fixture((17, 20), [(0, 0, 12, 12)], small, [1, 2], {1: 4},
                   {1: [(16, x) for x in range(1, 5)]}, False)
    assert any(len(piece) < len(small) for piece in info["normalized"])
    fixture((16, 16), [(3, 3, 7, 7)], small, [1], {1: 2},
            {1: [(y, 15) for y in range(2)]}, False)


def self_test():
    """Bounded 42-seed audit plus API, mutation and semantic regressions."""
    _boundary_regressions()
    signatures, features = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                features.append(_features(pair))
                assert np.array_equal(solve(pair["output"]), pair["output"]), "completion is not closed"
        assert _key(puzzle["train"][0]["input"]) == _key(generate_puzzle(seed, 2, 1)["train"][0]["input"])
    assert len(signatures) == 42
    assert len({f["shape"] for f in features}) >= 50
    assert len({f["stencil"] for f in features}) >= 90
    assert {f["radius"] for f in features} == {1, 2}
    assert len({f["size"] for f in features}) >= 10
    assert {f["frames"] for f in features} >= {1, 2, 3}
    assert max(f["marks"] for f in features) >= 6
    assert len({g for f in features for g in f["frame_shapes"]}) >= 40
    assert {f["background"] for f in features} == set(range(10))
    assert {f["frame"] for f in features} == set(range(10))
    assert set(k for f in features for k in f["lengths"]) == set(range(1, 10))
    assert any(f["reused"] for f in features)
    assert any(f["clipped"] for f in features)
    assert any(f["intact"] for f in features)
    assert any(f["symmetry"] for f in features)
    assert any(f["absent"] for f in features)
    assert any(f["unmarked"] for f in features)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(311, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(777, 6, 3), generate_puzzle(777, 6, 3)
    assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
               for p, q in zip(a[s], b[s]) for k in ("input", "output"))
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Change a visible count without consulting generation: other colors' repairs change.
    witnessed = False
    for pair in generate_puzzle(17)["train"][:2]:
        info = _read_input(pair["input"])
        for color, bar in info["bars"].items():
            if len(bar["coords"]) < 2:
                continue
            changed = pair["input"].copy()
            changed[min(bar["coords"])] = info["background"]
            try:
                changed_out = solve(changed)
            except ValueError:
                continue
            # Compare neighborhoods, including the cells those other occurrences can acquire.
            for occurrence in info["occurrences"]:
                if occurrence["color"] == color:
                    continue
                y, x, _ = occurrence["corner"]
                region = (slice(max(0, y - 2), min(changed.shape[0], y + 3)),
                          slice(max(0, x - 2), min(changed.shape[1], x + 3)))
                witnessed |= not np.array_equal(changed_out[region], pair["output"][region])
    assert witnessed, "counter changes never propagate to other colors"


if __name__ == "__main__":
    self_test()
    print("42-seed family audit passed")
