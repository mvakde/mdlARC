"""Boundary pulses: topology activates endpoint-coded, finite rim propagation.

Only NumPy and the standard library are used. All solving conventions are visible
or family-fixed: instruction color 9, dominant background, unique ink singleton.
"""
from collections import deque
import numbers

import numpy as np


INSTRUCTION = 9
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _flood(mask, start):
    reached = np.zeros(mask.shape, bool)
    if not mask[start]:
        return reached
    reached[start] = True
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and mask[x, y] and not reached[x, y]:
                reached[x, y] = True
                stack.append((x, y))
    return reached


def _components(mask):
    pending = mask.copy()
    result = []
    while pending.any():
        start = tuple(map(int, np.argwhere(pending)[0]))
        component = _flood(pending, start)
        pending[component] = False
        result.append([tuple(map(int, p)) for p in np.argwhere(component)])
    return result


def _dilate(mask, diagonal=False):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = mask.copy()
    offsets = ((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)) if diagonal else N4
    for dr, dc in offsets:
        out |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4 or INSTRUCTION not in colors or sum(counts == max(counts)) != 1:
        raise ValueError("input must have four roles and one dominant background")
    bg = int(colors[np.argmax(counts)])
    singletons = [int(c) for c, n in zip(colors, counts)
                  if n == 1 and c not in (bg, INSTRUCTION)]
    if bg == INSTRUCTION or len(singletons) != 1:
        raise ValueError("ink must be the unique non-instruction singleton")
    ink = singletons[0]
    wall = int(next(c for c in colors if c not in (bg, ink, INSTRUCTION)))
    return bg, wall, ink


def _read_bars(a, wall):
    bars = []
    for cells in _components((a == wall) | (a == INSTRUCTION)):
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        if len(rows) != 1 and len(cols) != 1:
            raise ValueError("every structural component must be a straight bar")
        ordered = sorted(cells)
        axis = 1 if len(rows) == 1 else 0
        coordinates = [p[axis] for p in ordered]
        if coordinates != list(range(coordinates[0], coordinates[-1] + 1)):
            raise ValueError("bars must be contiguous")
        offsets = [j for j, p in enumerate(ordered) if a[p] == INSTRUCTION]
        n, length = len(offsets), len(ordered)
        if not 1 <= n <= 4 or length < 2 * n + 1:
            raise ValueError("asymmetric codes need 1..4 markers and a plain tail")
        low = offsets == list(range(0, 2 * n, 2))
        high = offsets == sorted(length - 1 - 2 * j for j in range(n))
        if low == high:
            raise ValueError("markers must alternate inward from exactly one end")
        bars.append({"cells": ordered, "n": n, "marked_low": low,
                     "endpoint": ordered[-1] if low else ordered[0],
                     "other_endpoint": ordered[0] if low else ordered[-1],
                     "axis": axis})
    return bars


def _rim_from_complement(selected, complement):
    # complement is padded: edge-or-corner contact makes A's one-cell inner rim.
    return _dilate(complement, True)[1:-1, 1:-1] & selected


def _propagate(rim, sources, distance):
    """Multi-source graph radius; all equal-distance branches are included."""
    reached = np.zeros(rim.shape, bool)
    h, w = rim.shape
    queue = deque()
    for p in sources:
        if rim[p] and not reached[p]:
            reached[p] = True
            queue.append((*p, 0))
    while queue:
        r, c, d = queue.popleft()
        if d == distance:
            continue
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and rim[x, y] and not reached[x, y]:
                reached[x, y] = True
                queue.append((x, y, d + 1))
    return reached


def _pulse_masks(info, *, radius=None, reverse=False, holes=False):
    selected = info["selected"]
    rim = (_rim_from_complement(selected, np.pad(~selected, 1, constant_values=True))
           if holes else info["rim"])
    h, w = rim.shape
    masks = []
    for bar in info["bars"]:
        if not bar["keep"] or (not holes and not bar["external"]):
            continue
        r, c = bar["other_endpoint"] if reverse else bar["endpoint"]
        sources = [(r + dr, c + dc) for dr, dc in N4
                   if 0 <= r + dr < h and 0 <= c + dc < w and rim[r + dr, c + dc]]
        if not sources:
            raise ValueError("an active endpoint must have selected rim contact")
        masks.append(_propagate(rim, sources, bar["n"] + 1 if radius is None else radius))
    return masks


def _union(masks, shape):
    out = np.zeros(shape, bool)
    for mask in masks:
        out |= mask
    return out


def _plane_pulse(info):
    """Diagnostic shortcut: Manhattan radius clipped to the rim after measuring."""
    rim = info["rim"]
    rr, cc = np.indices(rim.shape)
    h, w = rim.shape
    pulse = np.zeros(rim.shape, bool)
    for bar in info["bars"]:
        if not (bar["keep"] and bar["external"]):
            continue
        r, c = bar["endpoint"]
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and rim[x, y]:
                pulse |= rim & (np.abs(rr - x) + np.abs(cc - y) <= bar["n"] + 1)
    return pulse


def _analyze(input_grid):
    a = _grid(input_grid)
    bg, wall, ink = _roles(a)
    seed = tuple(map(int, np.argwhere(a == ink)[0]))
    free = (a == bg) | (a == ink)
    selected = _flood(free, seed)
    outside = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    bars = _read_bars(a, wall)
    neighbor = _dilate(selected)
    for bar in bars:
        bar["keep"] = any(neighbor[p] for p in bar["cells"])
        bar["external"] = any(outside[r + 1, c + 1] for r, c in bar["cells"])
    info = {"input": a, "bg": bg, "wall": wall, "ink": ink, "seed": seed,
            "selected": selected, "outside": outside, "bars": bars,
            "rim": _rim_from_complement(selected, outside)}
    info["individual_pulses"] = _pulse_masks(info)
    info["pulse"] = _union(info["individual_pulses"], a.shape)
    return info


def _output(info, pulse=None):
    a = info["input"].copy()
    for bar in info["bars"]:
        for p in bar["cells"]:
            a[p] = info["wall"] if bar["keep"] else info["bg"]
    a[info["pulse"] if pulse is None else pulse] = info["ink"]
    a[info["seed"]] = info["ink"]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, topology, instructions and graph distances from input only."""
    return _output(_analyze(input_grid))


def _materialize(role, palette):
    bg, wall, ink = palette
    table = np.array([bg, wall, ink, INSTRUCTION], np.int8)
    return table[role]


def _add_bar(role, plan, cells, n, marked_low):
    cells = sorted(cells)
    mask = np.zeros(role.shape, bool)
    for p in cells:
        mask[p] = True
    if np.any(_dilate(_dilate(mask, True), True) & (role != 0)):
        return False
    role[mask] = 1
    plan.append((cells, n, marked_low))
    return True


def _place_edge(rng, role, plan, c0, c1, n, *, forced=None):
    h, w = role.shape
    minimum = 2 * n + 1
    if forced is not None:
        edge, col, marked_low, fixed_length = forced
        length = (fixed_length if fixed_length is not None else
                  int(rng.integers(minimum, min(h - 5, minimum + 5) + 1)))
        rows = range(length) if edge == "top" else range(h - length, h)
        return _add_bar(role, plan, [(r, col) for r in rows], n, marked_low)
    edges = ["top", "bottom"] + (["left"] if c0 == 0 else []) + (["right"] if c1 == w else [])
    for _ in range(80):
        edge = str(rng.choice(edges))
        if edge in ("top", "bottom"):
            col = int(rng.integers(c0 + 2, c1 - 2))
            length = int(rng.integers(minimum, min(h - 5, minimum + 6) + 1))
            rows = range(length) if edge == "top" else range(h - length, h)
            cells = [(r, col) for r in rows]
        else:
            if c1 - c0 < minimum + 3:
                continue
            row = int(rng.integers(2, h - 2))
            length = int(rng.integers(minimum, min(c1 - c0 - 3, minimum + 6) + 1))
            cols = range(length) if edge == "left" else range(w - length, w)
            cells = [(row, c) for c in cols]
        if _add_bar(role, plan, cells, n, bool(rng.integers(2))):
            return True
    return False


def _place_inside(rng, role, plan, c0, c1, n, *, allow_edge=False):
    h, w = role.shape
    for _ in range(100):
        vertical = bool(rng.integers(2))
        margin = 0 if allow_edge else 3
        available = h - 2 * margin if vertical else c1 - c0 - 2 * margin
        if available < 2 * n + 1:
            continue
        length = int(rng.integers(2 * n + 1, min(available, 2 * n + 6) + 1))
        if vertical:
            r = int(rng.integers(margin, h - margin - length + 1))
            c = int(rng.integers(c0 + 3, c1 - 2))
            cells = [(x, c) for x in range(r, r + length)]
        else:
            r = int(rng.integers(3, h - 2))
            c = int(rng.integers(c0 + margin, c1 - margin - length + 1))
            cells = [(r, x) for x in range(c, c + length)]
        if _add_bar(role, plan, cells, n, bool(rng.integers(2))):
            return True
    return False


def _scene(rng, maximum, ndiv, witness, requirements, palette, rotation, reflect,
           middle=False):
    h = int(rng.integers(22 if witness else 16, 31))
    w = int(rng.integers(24 if witness else 25 if ndiv == 2 else 17, 31))
    role = np.zeros((h, w), np.int8)
    plan = []
    if witness:
        cuts = [int(rng.integers(8, 11))]
    elif ndiv == 1:
        cuts = [int(rng.integers(7, w - 6))]
    elif ndiv == 2:
        widths = 7 + rng.multinomial(w - 23, [1 / 3] * 3)
        cuts = [int(widths[0]), int(widths[0] + widths[1] + 1)]
    else:
        cuts = []
    rooms = list(zip([0] + [c + 1 for c in cuts], cuts + [w]))
    room_index = (0 if witness == 1 else 1 if witness == 2 else
                  1 if middle else int(rng.integers(len(rooms))))
    c0, c1 = rooms[room_index]
    for j, col in enumerate(cuts):
        n = requirements[0] if witness and j == 0 else int(rng.integers(1, maximum + 1))
        # First witness launches its divider at the top, second at the bottom.
        marked_low = witness == 2 if witness else bool(rng.integers(2))
        if not _add_bar(role, plan, [(r, col) for r in range(h)], n, marked_low):
            return None
    nedge = 1 if witness else int(rng.integers(1 if not cuts else 0, 5))
    for j in range(nedge):
        n = requirements[1] if witness and j == 0 else int(rng.integers(1, maximum + 1))
        forced = (("top", 2, True, None) if witness == 1 else
                  ("bottom", cuts[0] + 3, False, 4) if witness == 2 else None)
        success = _place_edge(rng, role, plan, c0, c1, n, forced=forced)
        if witness and not success:
            return None
    for _ in range(int(rng.integers(1 if witness else 0, 3 if witness else 4))):
        n = 1 if witness else int(rng.integers(1, maximum + 1))
        success = _place_inside(rng, role, plan, c0, c1, n)
        if witness and not success:
            return None
    other_rooms = [room for i, room in enumerate(rooms) if i != room_index]
    for j in range(int(rng.integers(1 if witness else 0, 4)) if other_rooms else 0):
        lo, hi = other_rooms[int(rng.integers(len(other_rooms)))]
        n = 1 if witness and j == 0 else int(rng.integers(1, maximum + 1))
        success = _place_inside(rng, role, plan, lo, hi, n, allow_edge=True)
        if witness and j == 0 and not success:
            return None
    for cells, n, low in plan:
        for j in range(n):
            role[cells[2 * j if low else len(cells) - 1 - 2 * j]] = 3
    candidates = [(r, c) for r in range(h) for c in range(c0, c1) if role[r, c] == 0]
    if not candidates:
        return None
    start = candidates[0]
    role[start] = 2
    info = _analyze(_materialize(role, palette))
    if witness == 1:
        mask = info["selected"] & ~info["rim"]
    elif witness == 2:
        mask = info["pulse"]
    else:
        preference = int(rng.integers(3))
        mask = (info["pulse"] if preference == 0 else info["selected"] & ~info["rim"]
                if preference == 1 else info["selected"])
    candidates = np.argwhere(mask)
    if not len(candidates):
        return None
    role[start] = 0
    role[tuple(map(int, candidates[int(rng.integers(len(candidates)))]))] = 2
    role = np.rot90(role, rotation)
    if reflect:
        role = np.fliplr(role)
    a = _materialize(role, palette)
    info = _analyze(a)
    features = _features(info)
    if witness:
        wanted = "small" if witness == 1 else "large"
        if (features["selection"] != wanted or not features["islands"]
                or not features["erased"]
                or not np.any(info["rim"] & ~info["pulse"])):
            return None
        if witness == 1 and not features["diagonal_painted"]:
            return None
        if witness == 1 and np.array_equal(info["pulse"], _union(_pulse_masks(info, radius=2), a.shape)):
            return None
        if witness == 2 and not features["overlap"]:
            return None
        if witness == 2 and np.array_equal(info["pulse"], _plane_pulse(info)):
            return None
    if not np.any(info["rim"] & ~info["pulse"]):
        return None
    return {"input": a, "output": _output(info)}


def _features(info):
    a, target, rim = info["input"], info["selected"], info["rim"]
    free = (a == info["bg"]) | (a == info["ink"])
    sizes = [len(cells) for cells in _components(free)]
    area = int(target.sum())
    selection = ("only" if len(sizes) == 1 else "small" if area < max(sizes)
                 else "equal" if sizes.count(area) > 1 else "large")
    orth = _dilate(info["outside"])[1:-1, 1:-1] & target
    pulses = info["individual_pulses"]
    overlap = any(np.any(x & y) for i, x in enumerate(pulses) for y in pulses[i + 1:])
    active = [b for b in info["bars"] if b["keep"] and b["external"]]
    return {"shape": a.shape, "roles": (info["bg"], info["wall"], info["ink"]),
            "rooms": len(sizes), "selection": selection, "bars": len(info["bars"]),
            "active": len(active), "islands": sum(b["keep"] and not b["external"] for b in info["bars"]),
            "erased": sum(not b["keep"] for b in info["bars"]),
            "counts": {b["n"] for b in active}, "ends": {b["marked_low"] for b in active},
            "axes": {b["axis"] for b in active}, "overlap": overlap,
            "diagonal_painted": bool(np.any(info["pulse"] & rim & ~orth)),
            "seed_on_pulse": bool(info["pulse"][info["seed"]]),
            "seed_on_rim": bool(rim[info["seed"]])}


def _legal_integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _legal_integer(num_train, "num_train", 2, 10)
    num_test = _legal_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.integers(2, 5))
    palette = tuple(map(int, rng.choice(9, 3, replace=False)))
    recolor = bool(rng.integers(2))
    shift = int(rng.integers(2))  # held-out perpendicular orientation / three rooms
    base_rotation = int(rng.integers(4))
    base_reflect = bool(rng.integers(2))
    # The four active witness slots jointly demonstrate every queried unary count.
    obligations = list(map(int, rng.permutation(np.arange(2, maximum))))
    if not obligations:
        obligations.append(2)
    while len(obligations) < 2:
        obligations.append(int(rng.integers(1, maximum + 1)))
    # A long-range divider and a short free-tip intrusion teach graph distance.
    obligations.extend((maximum, 1))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index + 1 if split == "train" and index < 2 else 0
            requirements = obligations[2 * index:2 * index + 2] if witness else ()
            for attempt in range(160):
                roles = tuple(map(int, rng.choice(9, 3, replace=False))) if recolor else palette
                if witness:
                    rotation, reflect, ndiv = base_rotation, base_reflect, 1
                elif shift == 0:
                    rotation = (base_rotation + (int(rng.integers(2)) * 2) + (split == "test")) % 4
                    reflect, ndiv = bool(rng.integers(2)), int(rng.integers(3))
                else:
                    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
                    ndiv = 2 if split == "test" else int(rng.integers(2))
                pair = _scene(rng, maximum, ndiv, witness, requirements, roles, rotation, reflect,
                              middle=shift == 1 and split == "test" and index == 0)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a fresh boundary-pulse scene")
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, tests = set(), [], []
    count_witness = endpoint_witness = hole_witness = metric_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            original = a.copy()
            info = _analyze(a)
            assert np.array_equal(a, original) and np.array_equal(b, _output(info))
            assert not np.array_equal(a, b) and INSTRUCTION not in b
            assert b[info["seed"]] == info["ink"]
            assert np.all(info["pulse"] <= info["rim"])
            assert np.any(info["rim"] & ~info["pulse"]), "pulses must not saturate the rim"
            assert np.any((a == info["bg"]) & (b == info["bg"]))
            for bar in info["bars"]:
                expected = info["wall"] if bar["keep"] else info["bg"]
                assert all(b[p] == expected for p in bar["cells"])
                component = np.zeros(a.shape, bool)
                for p in bar["cells"]:
                    component[p] = True
                blocked = (a == info["wall"]) | (a == INSTRUCTION)
                assert not np.any(_dilate(_dilate(component, True), True) & blocked & ~component)
                if bar["keep"] and not bar["external"]:
                    assert not any(info["outside"][r + 1, c + 1] for r, c in bar["cells"])
            # No nested enclosed free rooms; every free region meets the canvas.
            free = (a == info["bg"]) | (a == info["ink"])
            h, w = a.shape
            assert all(any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
                       for cells in _components(free))
            f = _features(info)
            (training if split == "train" else tests).append(f)
            if split == "train":
                count_witness |= not np.array_equal(info["pulse"], _union(_pulse_masks(info, radius=2), a.shape))
                endpoint_witness |= not np.array_equal(info["pulse"], _union(_pulse_masks(info, reverse=True), a.shape))
                hole_witness |= not np.array_equal(info["pulse"], _union(_pulse_masks(info, holes=True), a.shape))
                metric_witness |= not np.array_equal(info["pulse"], _plane_pulse(info))
    counts = set.union(*(f["counts"] for f in training))
    assert len(counts) >= 2 and set.union(*(f["counts"] for f in tests)) <= counts
    assert set.union(*(f["ends"] for f in training)) == {False, True}
    assert any(f["selection"] == "small" for f in training)
    assert any(f["selection"] == "large" for f in training)
    assert any(f["erased"] for f in training) and any(f["islands"] for f in training)
    assert any(f["diagonal_painted"] for f in training)
    assert any(f["seed_on_pulse"] for f in training) and any(not f["seed_on_pulse"] for f in training)
    assert any(not f["seed_on_rim"] for f in training)
    assert any(f["overlap"] for f in training)
    assert count_witness and endpoint_witness and hole_witness and metric_witness


def _literal_fixtures():
    """Hand-calculated regression labels, unrelated to the random constructors."""
    a = np.zeros((12, 16), np.int8)
    a[:, 6] = 5
    a[[0, 2], 6] = 9
    a[4:9, 3] = 5
    a[[4, 6], 3] = 9
    a[5, 9:14] = 5
    a[5, [11, 13]] = 9
    a[2, 2] = 2
    b = a.copy()
    b[b == 9] = 5
    b[5, 9:14] = 0
    b[11, 2:6] = 2
    b[8:12, 5] = 2
    reverse = a.copy()
    reverse[:, 6] = 5
    reverse[[9, 11], 6] = 9
    reverse_out = a.copy()
    reverse_out[reverse_out == 9] = 5
    reverse_out[5, 9:14] = 0
    reverse_out[0, 2:6] = 2
    reverse_out[:4, 5] = 2
    intrusion = np.zeros((14, 18), np.int8)
    intrusion[:, 9] = 5
    intrusion[0, 9] = 9
    intrusion[:7, 4] = 5
    intrusion[[0, 2, 4], 4] = 9
    intrusion[9, 3] = 2
    intrusion_out = intrusion.copy()
    intrusion_out[intrusion_out == 9] = 5
    intrusion_out[2:7, 3] = intrusion_out[2:7, 5] = 2
    intrusion_out[7, 3:6] = 2
    intrusion_out[11:14, 8] = 2
    intrusion_out[13, 6:9] = 2
    blocked_wave = np.zeros((14, 18), np.int8)
    blocked_wave[:, 10] = 5
    blocked_wave[[0, 2], 10] = 9
    blocked_wave[10:14, 7] = 5
    blocked_wave[13, 7] = 9
    blocked_wave[5, 3] = 2
    blocked_out = blocked_wave.copy()
    blocked_out[blocked_out == 9] = 5
    blocked_out[10:14, 9] = 2
    blocked_out[11:14, 8] = 2
    blocked_out[10:13, 6] = 2
    blocked_out[10, 8] = 2
    blocked_out[9, 6:10] = 2
    return [(a, b), (reverse, reverse_out), (intrusion, intrusion_out),
            (blocked_wave, blocked_out)]


def self_test():
    fixtures = _literal_fixtures()
    for a, b in fixtures:
        for rotation in range(4):
            for reflect in (False, True):
                x, y = np.rot90(a, rotation), np.rot90(b, rotation)
                if reflect:
                    x, y = np.fliplr(x), np.fliplr(y)
                original = x.copy()
                assert np.array_equal(solve(x), y) and np.array_equal(x, original)
        # Palette changes preserve semantics, fixing the instruction role.
        permutation = np.array([7, 0, 4, 1, 8, 3, 5, 6, 2, 9], np.int8)
        assert np.array_equal(solve(permutation[a]), permutation[b])
    blocked = _analyze(fixtures[-1][0])
    assert _plane_pulse(blocked)[13, 6] and not blocked["pulse"][13, 6]
    diagonal = np.zeros(blocked["rim"].shape, bool)
    for bar in blocked["bars"]:
        if bar["keep"] and bar["external"]:
            pulse = np.zeros_like(diagonal)
            r, c = bar["endpoint"]
            for dr, dc in N4:
                x, y = r + dr, c + dc
                if 0 <= x < pulse.shape[0] and 0 <= y < pulse.shape[1] and blocked["rim"][x, y]:
                    pulse[x, y] = True
            for _ in range(bar["n"] + 1):
                pulse = _dilate(pulse, True) & blocked["rim"]
            diagonal |= pulse
    assert diagonal[8, 9] and not blocked["pulse"][8, 9]
    a, _ = fixtures[0]
    one_marker = a.copy()
    one_marker[:, 6] = 5
    one_marker[0, 6] = 9
    one_output = fixtures[0][1].copy()
    one_output[11, 2] = one_output[8, 5] = 0
    assert np.array_equal(solve(one_marker), one_output)
    moved = a.copy()
    moved[2, 2], moved[7, 1] = 0, 2
    assert np.array_equal(_analyze(a)["pulse"], _analyze(moved)["pulse"])
    across = a.copy()
    across[2, 2], across[2, 12] = 0, 2
    assert not np.array_equal(_analyze(a)["pulse"], _analyze(across)["pulse"])
    assert np.all(solve(across)[4:9, 3] == 0)
    # Move a coded interior island to the canvas: it becomes an active intrusion.
    attached = a.copy()
    attached[4:9, 3] = 0
    attached[:5, 3] = [9, 5, 9, 5, 5]
    assert _features(_analyze(a))["active"] == 1
    assert _features(_analyze(attached))["active"] == 2
    assert not np.array_equal(_analyze(a)["pulse"], _analyze(attached)["pulse"])
    records, signatures, shapes, palettes, normalized = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for s in ("train", "test"):
            for pair in puzzle[s]:
                info = _analyze(pair["input"])
                f = _features(info)
                records.append(f)
                shapes.add(f["shape"])
                palettes.add(f["roles"])
                semantic = np.zeros(f["shape"], np.int8)
                for role, color in enumerate((*f["roles"], INSTRUCTION)):
                    semantic[pair["input"] == color] = role
                normalized.add(_key(semantic))
                # Larger graph radii contain smaller ones, including around bends.
                small = _union(_pulse_masks(info, radius=2), f["shape"])
                large = _union(_pulse_masks(info, radius=5), f["shape"])
                assert np.all(small <= large)
    assert len(signatures) == 42 and len(normalized) >= 190
    assert len(shapes) >= 70 and len(palettes) >= 30
    assert {1, 2, 3} <= {f["rooms"] for f in records}
    assert {1, 2, 3, 4} == set.union(*(f["counts"] for f in records))
    assert {0, 1, 2} <= {f["islands"] for f in records}
    assert {0, 1, 2} <= {f["erased"] for f in records}
    assert len({f["bars"] for f in records}) >= 6
    assert {0, 1} == set.union(*(f["axes"] for f in records))
    assert max(max(shape) for shape in shapes) == 30
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * train + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    first, second = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first
               for p, q in zip(first[s], second[s]) for k in ("input", "output"))
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    pair = damaged["test"][0]
    info = _analyze(pair["input"])
    p = tuple(map(int, np.argwhere(info["pulse"] & (pair["input"] == info["bg"]))[0]))
    pair["output"][p] = info["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("a missing pulse cell was accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len(shapes),
            "palettes": len(palettes), "normalized_inputs": len(normalized),
            "room_counts": sorted({f["rooms"] for f in records}),
            "bar_counts": sorted({f["bars"] for f in records}),
            "active_counts": sorted({f["active"] for f in records}),
            "island_counts": sorted({f["islands"] for f in records}),
            "erased_counts": sorted({f["erased"] for f in records}),
            "overlapping_pairs": sum(f["overlap"] for f in records)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
