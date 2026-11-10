"""Sealed cross-rays: local junctions, occluded labels, simultaneous erasure.

Only NumPy and the Python standard library are used. No parent code or examples
are imported; all semantic roles are recovered from the visible corner key.
"""
from __future__ import annotations

import itertools
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SAMPLE = ((2, 2), (1, 2), (3, 2), (2, 1), (2, 3))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside the permitted range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _canonical(grid, corner):
    if corner & 2:
        grid = grid[::-1, :]
    if corner & 1:
        grid = grid[:, ::-1]
    return grid


def _key(grid):
    """The 17 padding cells and explicit carrier sample identify local roles."""
    if min(grid.shape) < 7:
        raise ValueError("The key needs a six-cell corner and a separating gap")
    possibilities = []
    for corner in range(4):
        view = _canonical(grid, corner)
        frame = int(view[5, 5])
        if not (np.all(view[5, :6] == frame) and np.all(view[:6, 5] == frame)
                and np.count_nonzero(grid == frame) == 11):
            continue
        carrier = int(view[2, 2])
        if not all(view[p] == carrier for p in _SAMPLE):
            continue
        values, counts = np.unique(view[:5, :5], return_counts=True)
        backgrounds = values[counts == 17]
        if len(backgrounds) != 1 or np.count_nonzero(view[:5, :5] == carrier) != 5:
            continue
        background = int(backgrounds[0])
        query = frozenset(int(c) for c in values if c not in (carrier, background))
        if len(query) != 3 or len({background, carrier, frame} | set(query)) != 6:
            continue
        if not all(np.count_nonzero(view[:5, :5] == q) == 1 for q in query):
            continue
        gap = np.ones((7, 7), dtype=bool)
        gap[:6, :6] = False
        if not np.all(view[:7, :7][gap] == background):
            continue
        possibilities.append((corner, background, frame, carrier, query))
    if len(possibilities) != 1:
        raise ValueError("Expected exactly one recognizable, separated L key")
    corner, background, frame, carrier, query = possibilities[0]
    return dict(corner=corner, background=background, frame=frame,
                carrier=carrier, query=query)


def _records(view, info, relaxed=False):
    """Maximal rays on the original input; records may share arbitrary cells."""
    carrier = info["carrier"]
    mask = view == carrier
    mask = mask.copy()
    mask[:6, :6] = False
    centers = np.zeros_like(mask)
    if relaxed:
        centers = mask
    else:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                              & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    records = []
    h, w = view.shape
    for rr, cc in np.argwhere(centers):
        r, c = int(rr), int(cc)
        cells, caps, lengths = {(r, c)}, {}, []
        for dr, dc in _DIRECTIONS:
            a, b, length = r + dr, c + dc, 0
            while 0 <= a < h and 0 <= b < w and mask[a, b]:
                cells.add((a, b))
                length += 1
                a, b = a + dr, b + dc
            lengths.append(length)
            if 0 <= a < h and 0 <= b < w and not (a < 6 and b < 6):
                color = int(view[a, b])
                if color not in (info["background"], info["frame"], carrier):
                    caps[(a, b)] = color
                    cells.add((a, b))
        labels = frozenset(caps.values())
        records.append(dict(center=(r, c), cells=cells, caps=caps, labels=labels,
                            lengths=tuple(lengths), positive=info["query"] <= labels))
    return records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the key, certify each complete plus by its ray caps, erase their union."""
    inp = _grid(input_grid)
    info = _key(inp)
    view = _canonical(inp, info["corner"])
    erase = set()
    for record in _records(view, info):
        if record["positive"]:
            erase.update(record["cells"])
    output = inp.copy()
    result = _canonical(output, info["corner"])
    for p in erase:
        result[p] = info["background"]
    return output


def _audit_records(view, info):
    """Direct coordinate audit, separate from vectorized center detection."""
    h, w = view.shape
    ink = {(r, c) for r in range(h) for c in range(w)
           if view[r, c] == info["carrier"] and not (r < 6 and c < 6)}
    answer = []
    for r, c in sorted(ink):
        if not all((r + dr, c + dc) in ink for dr, dc in _DIRECTIONS):
            continue
        footprint, caps, lengths = {(r, c)}, {}, []
        for dr, dc in _DIRECTIONS:
            ray = []
            for distance in range(1, 31):
                p = r + distance * dr, c + distance * dc
                if p in ink:
                    ray.append(p)
                else:
                    a, b = p
                    if (0 <= a < h and 0 <= b < w and not (a < 6 and b < 6)
                            and int(view[p]) not in (info["background"], info["frame"], info["carrier"])):
                        caps[p] = int(view[p])
                    break
            footprint.update(ray)
            lengths.append(len(ray))
        footprint.update(caps)
        labels = frozenset(caps.values())
        answer.append(dict(center=(r, c), cells=footprint, caps=caps, labels=labels,
                           lengths=tuple(lengths), positive=info["query"] <= labels))
    return answer


def _components(view, background):
    todo = set(zip(*np.nonzero(view != background)))
    todo = {(int(r), int(c)) for r, c in todo if not (r < 6 and c < 6)}
    components = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells = {start}
        queue = [start]
        for r, c in queue:
            for dr, dc in _DIRECTIONS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    cells.add(p)
                    queue.append(p)
        components.append(cells)
    return components


def _module(points, background):
    rows, cols = zip(*points)
    r0, c0 = min(rows), min(cols)
    a = np.full((max(rows) - r0 + 1, max(cols) - c0 + 1), background, dtype=np.int8)
    for (r, c), color in points.items():
        a[r - r0, c - c0] = color
    return a


def _star(rng, roles, query, extras, mode, length=(1, 3), missing=0,
          clutter=True, occluded=False, long=False):
    background, _, carrier = roles
    q = list(query)
    x = int(extras[0])
    if mode == "three":
        caps = q + [None]
    elif mode == "extra":
        caps = q + [x]
    elif mode == "repeat":
        caps = q + [q[int(rng.integers(3))]]
    elif mode == "missing":
        caps = [q[i] for i in range(3) if i != missing] + [x, None]
    elif mode == "empty":
        caps = [None] * 4
    else:
        raise ValueError("Unknown star mode")
    caps = [caps[int(i)] for i in rng.permutation(4)]
    lengths = [int(v) for v in rng.integers(length[0], length[1] + 1, size=4)]
    if long:
        lengths[int(rng.integers(4))] = 9
    points = {(0, 0): carrier}
    for (dr, dc), steps, label in zip(_DIRECTIONS, lengths, caps):
        for k in range(1, steps + 1):
            points[(dr * k, dc * k)] = carrier
        if label is not None:
            points[(dr * (steps + 1), dc * (steps + 1))] = int(label)
    if mode == "missing":
        # A missing identity is attached diagonally, but it is not a ray cap.
        points[(1, 1)] = q[missing]
        if occluded:
            direction = caps.index(x)
            dr, dc = _DIRECTIONS[direction]
            stop = lengths[direction] + 1
            for k in range(stop + 1, stop + 3):
                points[(dr * k, dc * k)] = carrier
            points[(dr * (stop + 3), dc * (stop + 3))] = q[missing]
    if clutter:
        for p in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if p not in points and rng.random() < .45:
                points[p] = carrier if rng.random() < .7 else int(rng.choice(extras))
        # A one-sided branch gives a T along an arm; its tip is not on the hub's rays.
        for i, ((dr, dc), steps) in enumerate(zip(_DIRECTIONS, lengths)):
            if steps >= 2 and rng.random() < .35:
                k = int(rng.integers(2, steps + 1))
                side = int(rng.choice((-1, 1)))
                br, bc = dc * side, -dr * side
                for distance, color in ((1, carrier), (2, int(rng.choice(extras)))):
                    p = dr * k + br * distance, dc * k + bc * distance
                    if p not in points:
                        points[p] = color
    return _module(points, background)


def _network(rng, roles, query, extras, width, teaching=False):
    """A block or branching spine, with shared rays and independently capped columns."""
    background, _, carrier = roles
    q = list(query)
    solid = teaching or rng.random() < .45
    points = {(r, c): carrier for r in (-1, 0, 1) for c in range(width)} if solid else {}
    left, right = int(rng.integers(3)), int(rng.integers(3))
    for c in range(-left, width + right):
        points[(0, c)] = carrier
    mode = 0 if teaching else int(rng.integers(4))
    shared = ((q[0], q[1]), (q[0], int(extras[0])),
              (q[0], q[0]), (int(extras[0]), int(extras[-1])))[mode]
    points[(0, -left - 1)], points[(0, width + right)] = shared
    active = rng.random(width - 2) < .55
    if teaching:
        active[:2] = True
    elif not active.any() and rng.random() < .8:
        active[int(rng.integers(len(active)))] = True
    for c, positive in enumerate(active, 1):
        needed = [v for v in q if v not in shared]
        if positive and len(needed) <= 2:
            labels = needed + [int(rng.choice(q + list(extras)))] * (2 - len(needed))
            if teaching:
                labels = [q[2], int(extras[0]) if c == 1 else q[0]]
        else:
            missing = int(rng.choice(needed))
            options = [v for v in q + list(extras) if v != missing]
            labels = [int(v) for v in rng.choice(options, 2)]
        if rng.random() < .5:
            labels.reverse()
        lengths = (1, 1) if solid else tuple(int(v) for v in rng.integers(1, 5, size=2))
        if not solid and rng.random() < .10:
            lengths = (0, lengths[1])  # A capped T, rather than a complete junction.
        for sign, length, label in zip((-1, 1), lengths, labels):
            for distance in range(1, length + 1):
                points[(sign * distance, c)] = carrier
            points[(sign * (length + 1), c)] = label
    # The outer corners stay attached to erased arms without belonging to them.
    return np.rot90(_module(points, background), int(rng.integers(4))).copy()


def _near_miss(rng, roles, query):
    background, _, carrier = roles
    q = [int(v) for v in rng.permutation(query)]
    # Three carrier arms, and a colored cell in place of the missing fourth arm.
    points = {(0, 0): carrier, (-1, 0): carrier, (0, -1): carrier, (0, 1): carrier,
              (-2, 0): q[0], (0, -2): q[1], (1, 0): q[2]}
    turns = int(rng.integers(4))
    for _ in range(turns):
        points = {(-c, r): color for (r, c), color in points.items()}
    return _module(points, background)


def _separated(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _pack(rng, modules, shape):
    """Random placement of uneven bounding boxes; one background cell separates them."""
    order = sorted(range(len(modules)), key=lambda i: -modules[i].size)
    for expansion in range(17):
        h, w = min(30, shape[0] + expansion), min(30, shape[1] + expansion)
        if any(a.shape[0] > h or a.shape[1] > w for a in modules):
            continue
        for _ in range(35):
            boxes = [(0, 6, 0, 6)]
            placements = []
            for i in order:
                mh, mw = modules[i].shape
                positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)
                             if all(_separated((r, r + mh, c, c + mw), b) for b in boxes)]
                if not positions:
                    break
                r, c = positions[int(rng.integers(len(positions)))]
                boxes.append((r, r + mh, c, c + mw))
                placements.append((i, r, c))
            if len(placements) == len(modules):
                return (h, w), placements
        if h == 30 and w == 30:
            break
    return None


def _paint_key(grid, roles, query, rng):
    _, frame, carrier = roles
    grid[5, :6] = frame
    grid[:6, 5] = frame
    for p in _SAMPLE:
        grid[p] = carrier
    options = [(r, c) for r in range(5) for c in range(5) if (r, c) not in _SAMPLE]
    for i, color in zip(rng.choice(len(options), 3, replace=False), rng.permutation(query)):
        grid[options[int(i)]] = int(color)


def _make_pair(rng, roles, query, extras, corner, package, seen):
    for attempt in range(25):
        if package == "membership":
            modules = [_star(rng, roles, query, extras, "three", clutter=False)]
            modules += [_star(rng, roles, query, extras, "missing", missing=i,
                              length=(1, 2), occluded=(i == 0)) for i in range(3)]
            shape = tuple(int(v) for v in rng.integers(21, 29, size=2))
        elif package == "overlap":
            modules = [_network(rng, roles, query, extras, int(rng.integers(4, 6)), teaching=True),
                       _star(rng, roles, query, extras, "extra"), _near_miss(rng, roles, query)]
            shape = tuple(int(v) for v in rng.integers(15, 27, size=2))
        elif package == "long":
            mode = str(rng.choice(("three", "extra", "repeat")))
            modules = [_star(rng, roles, query, extras, mode, length=(6, 9), long=True)]
            if rng.random() < .4:
                modules.append(_near_miss(rng, roles, query))
            shape = tuple(int(v) for v in rng.integers(24, 31, size=2))
        elif package == "junctions":
            modules = [_network(rng, roles, query, extras, int(rng.integers(8, 13)))]
            if rng.random() < .6:
                modules.append(_star(rng, roles, query, extras, "missing", missing=int(rng.integers(3))))
            shape = tuple(int(v) for v in rng.integers(13, 30, size=2))
        else:
            compact = package == "compact"
            count = int(rng.integers(1, 3 if compact else 5))
            modes = ("three", "extra", "repeat", "missing", "empty")
            mood = int(rng.integers(5))
            modules = []
            for _ in range(count):
                if not compact and rng.random() < .20:
                    modules.append(_network(rng, roles, query, extras, int(rng.integers(4, 7))))
                elif rng.random() < .12:
                    modules.append(_near_miss(rng, roles, query))
                else:
                    mode = "missing" if mood == 0 else "three" if mood == 1 else str(rng.choice(modes))
                    modules.append(_star(rng, roles, query, extras, mode, length=(1, 2 if compact else 4),
                                         missing=int(rng.integers(3)), occluded=(mode == "missing" and rng.random() < .3)))
            low, high = (12, 23) if compact else (13, 31)
            shape = tuple(int(v) for v in rng.integers(low, high, size=2))
        packed = _pack(rng, modules, shape)
        if packed is None:
            continue
        shape, placements = packed
        inp = np.full(shape, roles[0], dtype=np.int8)
        _paint_key(inp, roles, query, rng)
        for i, r, c in placements:
            module = modules[i]
            inp[r:r + module.shape[0], c:c + module.shape[1]] = module
        inp = _canonical(inp, corner).copy()
        key = _array_key(inp)
        if key not in seen:
            seen.add(key)
            return {"input": inp, "output": solve(inp)}
    return _fallback_pair(rng, roles, query, extras, corner, package, seen)


def _fallback_pair(rng, roles, query, extras, corner, package, seen):
    """Finite emergency packing; the same semantics and teaching witnesses apply."""
    if package == "membership":
        modules = [_star(rng, roles, query, extras, "three", length=(1, 1), clutter=False)]
        modules += [_star(rng, roles, query, extras, "missing", length=(1, 1), missing=i,
                          clutter=False, occluded=(i == 0)) for i in range(3)]
    elif package == "overlap":
        modules = [_network(rng, roles, query, extras, 4, teaching=True),
                   _star(rng, roles, query, extras, "extra", length=(1, 1), clutter=False),
                   _near_miss(rng, roles, query)]
    elif package == "long":
        modules = [_star(rng, roles, query, extras, "extra", length=(6, 6), clutter=False, long=True)]
    elif package == "junctions":
        modules = [_network(rng, roles, query, extras, 12)]
    else:
        modules = [_star(rng, roles, query, extras, str(rng.choice(("three", "extra", "missing", "empty"))),
                         length=(1, 1), missing=int(rng.integers(3)), clutter=False)
                   for _ in range(2 if package == "compact" else 4)]
    canvas = np.full((30, 30), roles[0], dtype=np.int8)
    positions = [(7, 0)] if package in ("long", "junctions") else [(8, 8), (8, 19), (19, 8), (19, 19)]
    for module, (r, c) in zip(modules, positions):
        assert r + module.shape[0] <= 30 and c + module.shape[1] <= 30
        if package not in ("long", "junctions"):
            assert max(module.shape) <= 10
        canvas[r:r + module.shape[0], c:c + module.shape[1]] = module
    options = [(r, c) for r in range(5) for c in range(5) if (r, c) not in _SAMPLE]
    combinations = list(itertools.combinations(options, 3))
    start = int(rng.integers(len(combinations)))
    # 1,140 key-position patterns; at most thirteen previous inputs can collide.
    for offset in range(len(seen) + 1):
        inp = canvas.copy()
        inp[5, :6], inp[:6, 5] = roles[1], roles[1]
        for p in _SAMPLE:
            inp[p] = roles[2]
        for p, color in zip(combinations[(start + offset) % len(combinations)], query):
            inp[p] = color
        inp = _canonical(inp, corner).copy()
        key = _array_key(inp)
        if key not in seen:
            seen.add(key)
            return {"input": inp, "output": solve(inp)}
    raise AssertionError("Finite key-pattern uniqueness bound exceeded")


def _array_key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("long", "junctions", "compact")))
    varied_palette = shift != "compact" and rng.random() < .65
    move_key = rng.random() < .75
    base = [int(c) for c in rng.permutation(10)]
    active_labels = int(rng.integers(4, 8))
    base = base[:3 + active_labels]
    first_corner = int(rng.integers(4))
    second_corner = (first_corner + int(rng.integers(1, 4))) % 4 if move_key else first_corner
    corners = (first_corner, second_corner)
    puzzle, seen = {"train": [], "test": []}, set()
    previous_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            colors = [int(c) for c in rng.permutation(10)][:3 + active_labels] if varied_palette else base
            if varied_palette and split == "train" and index == 1 and tuple(colors[:3]) == previous_roles:
                colors[0], colors[1] = colors[1], colors[0]
            roles, labels = tuple(colors[:3]), colors[3:]
            if shift == "compact":
                # Every test triple is new, but all its identities occur in training keys.
                choices = ((0, 1, 2), (0, 1, 3)) if split == "train" else ((0, 2, 3), (1, 2, 3))
                pick = choices[index % 2]
                query = [labels[i] for i in pick]
            else:
                query = [int(v) for v in rng.choice(labels, 3, replace=False)]
            extras = [c for c in labels if c not in query]
            previous_roles = roles
            if split == "train":
                corner = corners[index] if index < 2 else int(rng.choice(corners))
                package = "membership" if index == 0 else "overlap" if index == 1 else "free"
            else:
                corner = int(rng.choice(corners))
                package = shift
            puzzle[split].append(_make_pair(rng, roles, query, extras, corner, package, seen))
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _destructive(view, info):
    out = view.copy()
    for r in range(1, view.shape[0] - 1):
        for c in range(1, view.shape[1] - 1):
            if r < 6 and c < 6:
                continue
            if all(out[r + dr, c + dc] == info["carrier"] for dr, dc in ((0, 0),) + _DIRECTIONS):
                records = _records(out, info)
                record = next((v for v in records if v["center"] == (r, c)), None)
                if record and record["positive"]:
                    for p in record["cells"]:
                        out[p] = info["background"]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    coverage = dict(three=False, extra=False, missing=False, off_ray=False,
                    occluded=False, selective=False, overlap=False, relaxed=False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _array_key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _key(inp)
            view, result = _canonical(inp, info["corner"]), _canonical(out, info["corner"])
            records = _audit_records(view, info)
            production = _records(view, info)
            assert records == production
            erase = set().union(*(v["cells"] for v in records if v["positive"]))
            changed = set(zip(*np.nonzero(view != result)))
            assert changed == erase, "Erasure must equal the certified ray union"
            assert all(result[p] == info["background"] for p in erase)
            assert np.array_equal(view[:6, :6], result[:6, :6]), "Key must be copied"
            assert np.array_equal(solve(inp), out)
            infos[split].append(info)
            if split != "train":
                continue
            positives = [v for v in records if v["positive"]]
            negatives = [v for v in records if not v["positive"]]
            coverage["three"] |= any(len(v["caps"]) == 3 and v["labels"] == info["query"] for v in positives)
            coverage["extra"] |= any(v["labels"] - info["query"] for v in positives)
            coverage["missing"] |= all(any(v["center"] not in erase and v["labels"] & info["query"] == info["query"] - {q}
                                           and len(v["caps"]) == 3 for v in negatives) for q in info["query"])
            components = _components(view, info["background"])
            for v in negatives:
                if len(v["labels"] & info["query"]) == 2:
                    component = next(g for g in components if v["center"] in g)
                    coverage["off_ray"] |= info["query"] <= {int(view[p]) for p in component}
                for dr, dc in _DIRECTIONS:
                    r, c = v["center"]
                    k = 1
                    while 0 <= r + dr * k < view.shape[0] and 0 <= c + dc * k < view.shape[1]:
                        p = r + dr * k, c + dc * k
                        if view[p] != info["carrier"]:
                            if p in v["caps"]:
                                for j in range(k + 1, k + 5):
                                    a, b = r + dr * j, c + dc * j
                                    if (0 <= a < view.shape[0] and 0 <= b < view.shape[1]
                                            and int(view[a, b]) in info["query"] - v["labels"]):
                                        coverage["occluded"] = True
                            break
                        k += 1
            coverage["selective"] |= any(view[p] == info["carrier"] and p not in erase
                                          and any((p[0] + dr, p[1] + dc) in erase for dr, dc in _DIRECTIONS)
                                          for g in components for p in g)
            coverage["overlap"] |= any(a["cells"] & b["cells"] for a, b in itertools.combinations(positives, 2))
            relaxed = set().union(*(v["cells"] for v in _records(view, info, relaxed=True) if v["positive"]))
            coverage["relaxed"] |= relaxed != erase
    assert all(coverage.values()), f"Missing training evidence: {coverage}"
    train_corners = {i["corner"] for i in infos["train"]}
    assert all(i["corner"] in train_corners for i in infos["test"])
    # All variants use input-visible roles. A varied palette is demonstrated in training.
    train_roles = {tuple(i[k] for k in ("background", "frame", "carrier")) for i in infos["train"]}
    test_roles = {tuple(i[k] for k in ("background", "frame", "carrier")) for i in infos["test"]}
    assert len(train_roles) > 1 or test_roles <= train_roles


def _fingerprint(puzzle):
    return tuple((_array_key(p["input"]), _array_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _counterfactual(pair):
    inp = np.asarray(pair["input"])
    info = _key(inp)
    view = _canonical(inp, info["corner"])
    records = _audit_records(view, info)
    options = [v for v in records if not v["positive"] and len(v["labels"]) == 3]
    if not options:
        return False
    # Change only the visible query: a formerly incomplete record now qualifies.
    changed = inp.copy()
    target = _canonical(changed, info["corner"])
    positions = [tuple(int(v) for v in p) for p in np.argwhere(np.isin(target[:5, :5], list(info["query"])))]
    for p, q in zip(positions, sorted(options[0]["labels"])):
        target[p] = q
    solved = _canonical(solve(changed), info["corner"])
    assert all(solved[p] == info["background"] for p in options[0]["cells"])
    assert not np.array_equal(solved[6:, :], _canonical(np.asarray(pair["output"]), info["corner"])[6:, :]) or not np.array_equal(solved[:, 6:], _canonical(np.asarray(pair["output"]), info["corner"])[:, 6:])
    return True


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, structures, roles, corners, counts, lengths, cap_counts = set(), set(), set(), set(), set(), set(), set()
    unchanged, overlap_sensitive = 0, 0
    for puzzle in puzzles:
        assert any(_counterfactual(pair) for pair in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                info = _key(inp)
                view = _canonical(inp, info["corner"])
                records = _audit_records(view, info)
                shapes.add(inp.shape)
                mapping = {}
                structures.add((inp.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)))
                roles.add(tuple(info[k] for k in ("background", "frame", "carrier")))
                corners.add(info["corner"])
                counts.add((len(records), sum(v["positive"] for v in records)))
                lengths.update(k for v in records for k in v["lengths"])
                cap_counts.update(len(v["caps"]) for v in records)
                unchanged += np.array_equal(inp, pair["output"])
                if split == "train":
                    overlap_sensitive += not np.array_equal(_destructive(view, info), _canonical(pair["output"], info["corner"]))
                # Palette equivariance: roles never depend on literal digit values.
                permutation = np.arange(10)[::-1]
                assert np.array_equal(solve(permutation[inp]), permutation[pair["output"]])
                # All four reflections preserve the rule, including cap occlusion.
                for corner in range(4):
                    assert np.array_equal(solve(_canonical(inp, corner)), _canonical(pair["output"], corner))
                reordered = inp.copy()
                key_view = _canonical(reordered, info["corner"])
                query_positions = [tuple(p) for p in np.argwhere(np.isin(key_view[:5, :5], list(info["query"])))]
                old_labels = [int(key_view[p]) for p in query_positions]
                for p, color in zip(query_positions, old_labels[::-1]):
                    key_view[p] = color
                expected = np.asarray(pair["output"]).copy()
                _canonical(expected, info["corner"])[:6, :6] = key_view[:6, :6]
                assert np.array_equal(solve(reordered), expected)
    assert len(shapes) >= 35 and len(structures) >= 175
    assert len(roles) >= 30 and corners == {0, 1, 2, 3}
    assert min(lengths) == 1 and max(lengths) >= 9
    assert {0, 3, 4} <= cap_counts
    assert len(counts) >= 12 and max(n for n, _ in counts) >= 8
    assert unchanged >= 1 and overlap_sensitive >= 42
    # Every legal explicit count combination, including the minimum witnesses.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(911, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # An arm can end at the canvas edge: three other caps are sufficient.
    roles0, query0 = (0, 9, 8), (1, 2, 3)
    edge = np.zeros((12, 13), dtype=np.int8)
    _paint_key(edge, roles0, query0, np.random.default_rng(5))
    edge[8, 8:13] = 8
    edge[7:10, 9] = 8
    edge[6, 9], edge[10, 9], edge[8, 7] = query0
    answer = solve(edge)
    assert np.all(answer[8, 7:13] == 0) and answer[6, 9] == answer[10, 9] == 0
    # Boundary-centered clipped T is not a plus and must not wrap.
    clipped = np.zeros((12, 13), dtype=np.int8)
    _paint_key(clipped, roles0, query0, np.random.default_rng(5))
    clipped[11, 8:11] = 8
    clipped[10, 9] = 8
    clipped[9, 9], clipped[11, 7], clipped[11, 11] = query0
    assert np.array_equal(solve(clipped), clipped)
    # Exercise the guaranteed fallback directly, even when ordinary packing succeeds.
    fallback_rng, fallback_seen = np.random.default_rng(18), set()
    anchor = [_fallback_pair(fallback_rng, roles0, query0, [4, 5], 0, kind, fallback_seen)
              for kind in ("membership", "overlap")]
    for kind in ("long", "junctions", "compact", "free"):
        pair = _fallback_pair(fallback_rng, roles0, query0, [4, 5], 0, kind, fallback_seen)
        validate_generated_puzzle({"train": anchor, "test": [pair]})
    # Verify recognition independently on every 3x3 binary carrier neighborhood.
    for bits in range(512):
        patch = np.array([(bits >> i) & 1 for i in range(9)]).reshape(3, 3)
        probe = np.zeros((13, 13), dtype=np.int8)
        _paint_key(probe, roles0, query0, np.random.default_rng(5))
        probe[8:11, 8:11] = patch * 8
        info = _key(probe)
        centers = {v["center"] for v in _records(probe, info)}
        assert ((9, 9) in centers) == all(patch[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
        assert _records(probe, info) == _audit_records(probe, info)
    # Wrong changed pixels, deleted side branches, and changed key cells are rejected.
    for kind in ("extra", "partial", "key"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        pair = bad["train"][0]
        inp, out = pair["input"], pair["output"]
        if kind == "key":
            info = _key(inp)
            _canonical(out, info["corner"])[5, 5] = info["background"]
        elif kind == "partial":
            p = tuple(np.argwhere(inp != out)[0])
            out[p] = inp[p]
        else:
            info = _key(inp)
            p = tuple(np.argwhere((inp == out) & (inp == info["background"]))[-1])
            out[p] = info["carrier"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": 42, "shapes": len(shapes),
            "color_normalized_structures": len(structures), "role_assignments": len(roles),
            "junction_active_count_combinations": len(counts), "max_arm_length": max(lengths),
            "unchanged_pairs": int(unchanged), "overlap_sensitive_training_pairs": int(overlap_sensitive)}
