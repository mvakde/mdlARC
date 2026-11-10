"""Register a marked crop, recover hollow-channel ownership, export counts.

Only standard library and NumPy. No file access, stored templates, or RNG in
solve. Generation and tests retain one puzzle at a time and write no artifacts.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _d4(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _components(points, diagonal=True):
    todo = set(points)
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _local(p, direction):
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


_STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _ray(shape, tip, direction, quantity=None):
    dr, dc = _STEPS[direction]
    r, c = tip
    cells = []
    while quantity is None or len(cells) < quantity:
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        cells.append((r, c))
    return tuple(cells)


def _read_arrow(frame):
    """Recognize thin monotone pointed envelopes, not a list of stencils."""
    found = []
    for direction in range(4):
        local = {_local(p, direction) for p in frame}
        y0 = min(y for y, x in local)
        y1 = max(y for y, x in local)
        rows = [sorted(x for y, x in local if y == row)
                for row in range(y0, y1 + 1)]
        if len(rows) < 4 or len(rows[0]) != 1 or len(rows[-1]) != 2:
            continue
        center = rows[0][0]
        lo = hi = center
        split, valid, cavity = False, True, set()
        for offset, row in enumerate(rows[1:], 1):
            if len(row) not in (1, 2):
                valid = False
                break
            left, right = row[0], row[-1]
            if not (0 <= lo - left <= 1 and 0 <= right - hi <= 1):
                valid = False
                break
            if len(row) == 1 and (split or left != center):
                valid = False
                break
            split |= len(row) == 2
            cavity.update(_world((y0 + offset, x), direction)
                          for x in range(left + 1, right))
            lo, hi = left, right
        if valid and cavity and lo < center < hi:
            found.append(dict(frame=set(frame), cavity=cavity,
                              tip=_world((y0, center), direction), direction=direction))
    if len(found) > 1:
        raise ValueError("ambiguous channel direction")
    return found[0] if found else None


def _arrows(mask):
    objects = []
    for frame in _components(map(tuple, np.argwhere(mask))):
        obj = _read_arrow(frame)
        if obj is not None:
            objects.append(obj)
    if not objects:
        raise ValueError("blueprint has no channel")
    owned = set()
    for obj in objects:
        if owned & obj["cavity"] or any(mask[p] for p in obj["cavity"]):
            raise ValueError("overlapping channel hollows")
        owned.update(obj["cavity"])
    return objects


def _panels(grid):
    """Color 9 is packing material; the two non-9 components are rectangles."""
    grid = np.asarray(grid)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("expected a two-dimensional integer grid")
    if not all(1 <= n <= 30 for n in grid.shape) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("grid outside ARC bounds")
    parts = _components(map(tuple, np.argwhere(grid != 9)), diagonal=False)
    if len(parts) != 2:
        raise ValueError("expected two rectangular panels separated by 9")
    panels = []
    for part in parts:
        rr, cc = zip(*part)
        r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
        if len(part) != (r1 - r0) * (c1 - c0):
            raise ValueError("panel is not rectangular")
        panels.append(grid[r0:r1, c0:c1].astype(np.int8, copy=True))
    panels.sort(key=lambda a: len(np.unique(a)))
    reference, patch = panels
    if len(np.unique(reference)) != 2 or len(np.unique(patch)) != 3:
        raise ValueError("blueprint has two colors and observation has three")
    bg = int(reference[0, 0])
    if any(np.any(edge != bg) for edge in
           (reference[0], reference[-1], reference[:, 0], reference[:, -1])):
        raise ValueError("blueprint needs a background border")
    ink = int(next(c for c in np.unique(reference) if c != bg))
    signal = set(map(int, np.unique(patch))) - {bg, ink}
    if len(signal) != 1 or not {bg, ink} <= set(map(int, np.unique(patch))):
        raise ValueError("observation does not share the blueprint palette")
    return reference == ink, patch, bg, ink, signal.pop()


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rr, cc = np.nonzero(np.all(windows == patch, axis=(-1, -2)))
    return list(zip(map(int, rr), map(int, cc)))


def _matches(reference, patch):
    return [(t, r, c) for t in range(8)
            for r, c in _locations(_d4(reference, t), patch)]


def _decode(grid):
    reference, patch, bg, ink, signal = _panels(grid)
    matches = _matches(reference, patch == ink)
    if len(matches) != 1:
        raise ValueError("observation must have one exact dihedral registration")
    t, r0, c0 = matches[0]
    full = _d4(reference, t)
    if patch.size >= full.size:
        raise ValueError("observation must be a proper crop")
    marks = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(patch == signal)}
    objects = _arrows(full)
    for obj in objects:
        obj["payload"] = marks & obj["cavity"]
        obj["quantity"] = len(obj["payload"])
        obj["ray"] = _ray(full.shape, obj["tip"], obj["direction"], obj["quantity"])
        obj["room"] = len(_ray(full.shape, obj["tip"], obj["direction"]))
        obj["tip_visible"] = r0 <= obj["tip"][0] < r0 + patch.shape[0] and c0 <= obj["tip"][1] < c0 + patch.shape[1]
    cavities = set().union(*(o["cavity"] for o in objects))
    exterior = marks - cavities
    reserved = set()
    for obj in objects:
        if reserved.intersection(obj["ray"]) or any(full[p] or p in cavities or p in exterior for p in obj["ray"]):
            raise ValueError("an exported ray would collide")
        reserved.update(obj["ray"])
    return dict(reference=reference, full=full, patch=patch, bg=bg, ink=ink,
                signal=signal, transform=t, offset=(r0, c0), marks=marks,
                objects=objects, exterior=exterior)


def _answer(info):
    out = np.where(info["full"], info["ink"], info["bg"]).astype(np.int8)
    for p in info["exterior"]:
        out[p] = info["signal"]
    for obj in info["objects"]:
        for p in obj["ray"]:
            out[p] = info["signal"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct all geometry and ownership from this input alone."""
    return _answer(_decode(input_grid))


def _shape(rng, primary=False):
    left, right = (int(v) for v in rng.integers(2 if primary else 1, 4, size=2))
    tail = int(rng.integers(3, 5) if primary else rng.integers(2, 5))
    stem = int(rng.integers(0, 2))
    levels = [(0, 0)] * (stem + 1)
    a = b = 0
    while a < left or b < right:
        da = a < left and (b == right or rng.random() < .85)
        db = b < right and (a == left or rng.random() < .85)
        if not da and not db:
            da = a < left
            db = not da
        a, b = a + int(da), b + int(db)
        levels.append((-a, b))
    levels.extend([(-left, right)] * tail)
    frame, cavity = set(), set()
    for r, (a, b) in enumerate(levels):
        frame.update(((r, a + left), (r, b + left)))
        cavity.update((r, c + left) for c in range(a + 1, b))
    return dict(frame=frame, cavity=cavity, tip=(0, left), direction=0)


def _transform_object(obj, shape, t):
    ids = _d4(np.arange(shape[0] * shape[1]).reshape(shape), t)
    inverse = {int(ids[p]): tuple(map(int, p)) for p in np.ndindex(ids.shape)}
    def point(p):
        return inverse[p[0] * shape[1] + p[1]]
    tip = point(obj["tip"])
    # A virtual inward neighbor always lies inside the primary frame bounds.
    dr, dc = _STEPS[obj["direction"]]
    inside = point((obj["tip"][0] - dr, obj["tip"][1] - dc))
    direction = _STEPS.index((tip[0] - inside[0], tip[1] - inside[1]))
    return dict(frame={point(p) for p in obj["frame"]},
                cavity={point(p) for p in obj["cavity"]}, tip=tip, direction=direction)


def _shift_object(obj, r, c):
    return dict(frame={(r + y, c + x) for y, x in obj["frame"]},
                cavity={(r + y, c + x) for y, x in obj["cavity"]},
                tip=(r + obj["tip"][0], c + obj["tip"][1]), direction=obj["direction"])


def _box(points):
    rr, cc = zip(*points)
    return {(r, c) for r in range(min(rr), max(rr) + 1)
            for c in range(min(cc), max(cc) + 1)}


def _guard(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _blueprint(rng):
    """Compose a new structured reference, including a nearby registration bend."""
    for attempt in range(60):
        H, W = int(rng.integers(15, 19)), int(rng.integers(12, 19))
        primary = _shape(rng, primary=True)
        box = _box(primary["frame"])
        h = max(r for r, c in box) + 1
        w = max(c for r, c in box) + 1
        if h > H - 6:
            continue
        r = int(rng.integers(5, H - h))
        possible = [c for c in range(1, W - w) if c >= 4 or W - c - w >= 4]
        if not possible:
            continue
        c = int(rng.choice(possible))
        primary = _shift_object(primary, r, c)
        objects = [primary]
        mask = np.zeros((H, W), dtype=bool)
        for p in primary["frame"]:
            mask[p] = True
        boxes = _box(primary["frame"])
        corridors = set(_ray(mask.shape, primary["tip"], 0))
        # A small L alongside the deep hollow remains visible when the tip is
        # absent. Its exact placement, handedness and length vary.
        bends = []
        for side in (-1, 1):
            x = c - 3 if side < 0 else c + w + 1
            for y in range(max(1, r + h - 5), min(H - 3, r + h - 1)):
                cells = {(y, x), (y + 1, x), (y + 2, x),
                         (y + (0 if rng.random() < .5 else 2), x + 1)}
                if (all(1 <= a < H - 1 and 1 <= b < W - 1 for a, b in cells)
                        and not cells & (_guard(boxes) | corridors)):
                    bends.append(cells)
        if not bends:
            continue
        bend = bends[int(rng.integers(len(bends)))]
        for p in bend:
            mask[p] = True
        protected = _guard(boxes | bend)
        desired = int(rng.integers(1, 4))
        for _ in range(desired - 1):
            for trial in range(60):
                obj = _shape(rng)
                sh = (max(p[0] for p in obj["frame"]) + 1,
                      max(p[1] for p in obj["frame"]) + 1)
                t = int(rng.integers(4))
                obj = _transform_object(obj, sh, t)
                bh = max(p[0] for p in obj["frame"]) + 1
                bw = max(p[1] for p in obj["frame"]) + 1
                if bh > H - 2 or bw > W - 2:
                    continue
                obj = _shift_object(obj, int(rng.integers(1, H - bh)), int(rng.integers(1, W - bw)))
                bbox = _box(obj["frame"])
                ray = set(_ray(mask.shape, obj["tip"], obj["direction"]))
                if bbox & (protected | corridors) or ray & (boxes | set(map(tuple, np.argwhere(mask))) | corridors):
                    continue
                objects.append(obj)
                for p in obj["frame"]:
                    mask[p] = True
                boxes.update(bbox)
                corridors.update(ray)
                protected.update(_guard(bbox))
                break
        for _ in range(int(rng.integers(0, 4))):
            for trial in range(30):
                a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                kind = int(rng.integers(3))
                if kind == 0:
                    tile = {(y, 0) for y in range(a)} | {(a - 1, x) for x in range(b)}
                elif kind == 1:
                    tile = {(0, x) for x in range(b)} | {(a - 1, x) for x in range(b)} | {(y, x) for y in range(a) for x in (0, b - 1)}
                else:
                    tile = {(0, x) for x in range(b)}
                r0, c0 = int(rng.integers(1, H - a)), int(rng.integers(1, W - b))
                cells = {(r0 + y, c0 + x) for y, x in tile}
                if cells & (protected | corridors):
                    continue
                for p in cells:
                    mask[p] = True
                protected.update(_guard(cells))
                break
        # Compare the geometry reader with the construction's cell sets.
        read = _arrows(mask)
        if {frozenset(o["frame"]) for o in read} != {frozenset(o["frame"]) for o in objects}:
            continue
        t = int(rng.integers(8))
        transformed = [_transform_object(o, mask.shape, t) for o in objects]
        return _d4(mask, t), transformed, transformed[0]["tip"]
    raise RuntimeError("bounded blueprint construction exhausted")


def _window(rng, reference, full, primary, objects, hidden):
    H, W = full.shape
    cavities = set().union(*(o["cavity"] for o in objects))
    axes = set().union(*(set(_ray(full.shape, o["tip"], o["direction"])) for o in objects))
    # One bounded search, no saved bank of reconstructible windows.
    for attempt in range(900):
        h, w = int(rng.integers(3, H + 1)), int(rng.integers(3, W + 1))
        if h * w == H * W or not (reference.shape[1] + w + 1 <= 30 or reference.shape[0] + h + 1 <= 30):
            continue
        r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
        def inside(p):
            return r <= p[0] < r + h and c <= p[1] < c + w
        if hidden:
            if inside(primary["tip"]):
                continue
        elif not all(inside(p) for p in primary["frame"]):
            continue
        available = sorted(p for p in primary["cavity"] if inside(p))
        if len(available) < 4 or sum(inside(p) for p in primary["frame"]) < 4:
            continue
        exterior = [(y, x) for y in range(r, r + h) for x in range(c, c + w)
                    if not full[y, x] and (y, x) not in cavities | axes]
        if not exterior:
            continue
        patch = full[r:r + h, c:c + w]
        if len(_matches(reference, patch)) == 1:
            return r, c, h, w, available, exterior
    raise RuntimeError("bounded identifying-window search exhausted")


def _pick_payload(rng, cells, count, compact=False):
    cells = sorted(cells)
    if compact:
        chosen = {cells[int(rng.integers(len(cells)))]}
        while len(chosen) < count:
            choices = [p for p in cells if p not in chosen and
                       any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for q in chosen)]
            if not choices:
                choices = [p for p in cells if p not in chosen]
            chosen.add(choices[int(rng.integers(len(choices)))])
        return chosen
    return {cells[int(i)] for i in rng.choice(len(cells), count, replace=False)}


def _pack(rng, reference, patch):
    axes = []
    if reference.shape[1] + patch.shape[1] + 1 <= 30:
        axes.append(1)
    if reference.shape[0] + patch.shape[0] + 1 <= 30:
        axes.append(0)
    axis = int(rng.choice(axes))
    panels = [reference, patch]
    if rng.random() < .5:
        panels.reverse()
    if axis == 0:
        panels = [a.T for a in panels]
    height = max(a.shape[0] for a in panels)
    offsets = [int(rng.integers(height - a.shape[0] + 1)) for a in panels]
    out = np.full((height, sum(a.shape[1] for a in panels) + 1), 9, dtype=np.int8)
    col = 0
    for a, row in zip(panels, offsets):
        out[row:row + a.shape[0], col:col + a.shape[1]] = a
        col += a.shape[1] + 1
    return out.T.copy() if axis == 0 else out


def _make_pair(rng, reference, objects, primary_tip, t, hidden, palette, quantity=None):
    full = _d4(reference, t)
    transformed = [_transform_object(o, reference.shape, t) for o in objects]
    primary_index = next(i for i, o in enumerate(objects) if o["tip"] == primary_tip)
    primary = transformed[primary_index]
    r, c, h, w, available, exterior = _window(rng, reference, full, primary, transformed, hidden)
    room = len(_ray(full.shape, primary["tip"], primary["direction"]))
    q = quantity if quantity is not None else int(rng.integers(1, min(len(available), room - 1, 9) + 1))
    marks = _pick_payload(rng, available, q, compact=quantity == 2)
    for i, obj in enumerate(transformed):
        if i == primary_index:
            continue
        visible = sorted(p for p in obj["cavity"] if r <= p[0] < r + h and c <= p[1] < c + w)
        if visible and rng.random() < .65:
            count = int(rng.integers(1, min(len(visible), 9) + 1))
            marks.update(_pick_payload(rng, visible, count))
    # Prefer an exterior mark within a frame's rectangular bounding box when
    # available, so bounding-box counting is also demonstrably wrong.
    corners = [p for p in exterior if any(p in _box(o["frame"]) for o in transformed)]
    first = (corners or exterior)[int(rng.integers(len(corners or exterior)))]
    inert = {first}
    others = sorted(set(exterior) - inert)
    if others:
        extra = min(len(others), int(rng.integers(0, 3)))
        inert.update(others[int(i)] for i in rng.choice(len(others), extra, replace=False))
    marks.update(inert)
    bg, ink, signal = palette
    ref_grid = np.where(reference, ink, bg).astype(np.int8)
    observation = np.where(full[r:r + h, c:c + w], ink, bg).astype(np.int8)
    for y, x in marks:
        observation[y - r, x - c] = signal
    inp = _pack(rng, ref_grid, observation)
    # Expected output uses construction-owned geometry and positions. The
    # public solver independently segments the two panels and rereads arrows.
    out = np.where(full, ink, bg).astype(np.int8)
    for p in inert:
        out[p] = signal
    for obj in transformed:
        count = len(marks & obj["cavity"])
        for p in _ray(full.shape, obj["tip"], obj["direction"], count):
            out[p] = signal
    if not np.array_equal(solve(inp), out):
        raise AssertionError("construction and input reader disagree")
    return {"input": inp, "output": out}


def _count(value, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= int(value) <= upper:
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # All rules and palettes are input-readable. This schedule only controls
    # evidence and novelty, never the meaning of a test instance.
    for layout_attempt in range(18):
        reference, objects, primary_tip = _blueprint(rng)
        quarter = int(rng.choice([1, 3]))
        first_palette = tuple(map(int, rng.choice(9, 3, replace=False)))
        # Rotate the second supplied blueprint, then register its observation
        # back to the FIRST output orientation. Counts 2 and 4 therefore vary
        # without changing output direction, channel profile or role colors.
        train_t = [0, (-quarter) % 4] + ([int(rng.integers(4, 8))] if nt >= 3 else [])
        train_t += [int(rng.integers(8)) for _ in range(nt - len(train_t))]
        test_pool = [0, 1, 2, 3] if nt == 2 else list(range(8))
        puzzle, seen = {"train": [], "test": []}, set()
        try:
            for split, count in (("train", nt), ("test", nq)):
                for index in range(count):
                    t = train_t[index] if split == "train" else int(rng.choice(test_pool))
                    hidden = (index == 1 if split == "train" else True)
                    if split == "train" and index >= 2:
                        hidden = bool(rng.random() < .65)
                    quantity = (2 if index == 0 else 4) if split == "train" and index < 2 else None
                    palette = first_palette if split == "train" and index < 2 else tuple(map(int, rng.choice(9, 3, replace=False)))
                    pair_reference, pair_objects, pair_tip = reference, objects, primary_tip
                    if split == "train" and index == 1:
                        pair_reference = _d4(reference, quarter)
                        pair_objects = [_transform_object(o, reference.shape, quarter) for o in objects]
                        primary_index = next(i for i, o in enumerate(objects) if o["tip"] == primary_tip)
                        pair_tip = pair_objects[primary_index]["tip"]
                    for retry in range(8):
                        pair = _make_pair(rng, pair_reference, pair_objects, pair_tip, t, hidden, palette, quantity)
                        k = _key(pair["input"])
                        if k not in seen:
                            break
                    else:
                        raise RuntimeError("distinct-input retries exhausted")
                    seen.add(k)
                    puzzle[split].append(pair)
            validate_generated_puzzle(puzzle)
            return puzzle
        except RuntimeError:
            # Failed window searches leave no disk state or partially returned
            # puzzle. A new blueprint is tried, with an explicit finite bound.
            continue
    raise RuntimeError("bounded puzzle construction exhausted")


def _profile(obj):
    points = [_local(p, obj["direction"]) for p in obj["frame"]]
    y0, x0 = min(y for y, x in points), min(x for y, x in points)
    return tuple(sorted((y - y0, x - x0) for y, x in points))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, reference, witnesses = set(), None, {}
    hidden_train, hidden_test, full_train, connected = False, False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_answer(info), out)
            reference_class = min(_key(_d4(info["reference"], t)) for t in range(8))
            if reference is None:
                reference = reference_class
            assert reference_class == reference, "reference content changes within puzzle"
            assert info["exterior"], "ownership has no exterior counterexample"
            assert all(out[p] == info["signal"] for p in info["exterior"])
            assert np.all(out[info["full"]] == info["ink"]), "frame/landmark changed"
            active = [o for o in info["objects"] if o["quantity"]]
            assert active and any(o["room"] > o["quantity"] for o in active)
            assert all(out[p] == info["bg"] for o in active for p in o["payload"])
            hidden = any(not o["tip_visible"] for o in active)
            r0, c0 = info["offset"]
            h, w = info["patch"].shape
            if split == "train":
                hidden_train |= hidden
                for o in active:
                    full_train |= all(r0 <= r < r0 + h and c0 <= c < c0 + w for r, c in o["frame"])
                    if o["room"] > o["quantity"]:
                        signature = (_profile(o), o["direction"], info["bg"], info["ink"], info["signal"])
                        witnesses.setdefault(signature, set()).add(o["quantity"])
                    connected |= len(_components(o["payload"])) < o["quantity"]
            else:
                assert hidden, "each test must depend on a hidden active tip"
                hidden_test |= hidden
            # An A-only completion that transfers every observed mark but
            # performs no counting/export is an explicit competing answer.
            raw = np.where(info["full"], info["ink"], info["bg"])
            for p in info["marks"]:
                raw[p] = info["signal"]
            assert not np.array_equal(raw, out), "registered copying alone solves this pair"
            assert len(info["marks"]) > sum(o["quantity"] for o in active)
            if hidden:
                assert any(o["ray"] and all(not (r0 <= r < r0 + h and c0 <= c < c0 + w)
                                             for r, c in o["ray"])
                           for o in active if not o["tip_visible"]), "hidden tip has no consequential export"
    assert hidden_train and hidden_test and full_train and connected
    assert any({2, 4} <= values for values in witnesses.values()), "training lacks matched count witnesses"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """42 sequential seeds; bounded diagnostics, no retained sample corpus."""
    puzzles, blueprints, profiles, sizes, crops, offsets = set(), set(), set(), set(), set(), set()
    transforms, directions, counts, object_counts, palettes = set(), set(), set(), set(), set()
    clipping, multi_active, rectangle = 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_fingerprint(p))
        first = _decode(p["train"][0]["input"])["reference"]
        blueprints.add(min(_key(_d4(first, t)) for t in range(8)))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                sizes.add(info["full"].shape)
                crops.add(info["patch"].shape)
                offsets.add(info["offset"])
                transforms.add(info["transform"])
                palettes.add((info["bg"], info["ink"], info["signal"]))
                object_counts.add(len(info["objects"]))
                rectangle += info["full"].shape[0] != info["full"].shape[1]
                active = [o for o in info["objects"] if o["quantity"]]
                multi_active += len(active) > 1
                for o in info["objects"]:
                    profiles.add(_profile(o))
                    directions.add(o["direction"])
                    if o["quantity"]:
                        counts.add(o["quantity"])
                        clipping += o["quantity"] > o["room"]
    assert len(puzzles) == len(blueprints) == 42
    assert len(profiles) >= 30 and len(sizes) >= 12 and len(crops) >= 35
    assert len(offsets) >= 20 and transforms == set(range(8)) and directions == set(range(4))
    assert {1, 2, 4} <= counts and max(counts) >= 6 and len(palettes) >= 60
    assert len(object_counts) >= 2 and multi_active and clipping and rectangle
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42 + nt * 5 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in a.items()} for a in p[s]] for s in p}
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 9
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Solver must be equivariant under any permutation of the nine role
    # colors, with the packing color held fixed.
    permutation = np.array([4, 7, 1, 8, 0, 3, 6, 2, 5, 9], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "blueprint_classes": len(blueprints),
            "channel_profiles": len(profiles), "output_shapes": len(sizes), "crop_shapes": len(crops),
            "offsets": len(offsets), "transforms": sorted(transforms), "directions": sorted(directions),
            "quantities": sorted(counts), "object_counts": sorted(object_counts),
            "palettes": len(palettes), "multi_active_pairs": multi_active, "clipped_channels": clipping}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
