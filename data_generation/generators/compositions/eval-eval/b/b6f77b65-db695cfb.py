"""Rigid rods transported into finite diagonal latches; no parent imports.

Rod tuples are (axis, row, col, length, base_color, passenger_offsets).
Only the visible input is consulted by solve().
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _cell(p, k):
    o, r, c, _, _, _ = p
    return r + (k if o == "V" else 0), c + (k if o == "H" else 0)


def _cells(p):
    return frozenset(_cell(p, k) for k in range(p[3]))


def _shift(p, d):
    return p[0], p[1] + d, p[2], p[3], p[4], p[5]


def _full_line(point, slope, shape):
    h, w = shape
    r, c = point
    return tuple((q, c + slope * (q - r)) for q in range(2, h)
                 if 0 <= c + slope * (q - r) < w)


def _guides(points):
    points = sorted(points)
    degree = {p: 0 for p in points}
    guides = []
    occupied = set()
    interior = {}
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            dr, dc = b[0] - a[0], b[1] - a[1]
            if dr and abs(dr) == abs(dc):
                degree[a] += 1
                degree[b] += 1
                assert degree[a] <= 1 and degree[b] <= 1, "ambiguous endpoints"
                assert dr >= 2, "guide must have an interior"
                slope = 1 if dc > 0 else -1
                segment = tuple((a[0] + k, a[1] + slope * k) for k in range(dr + 1))
                assert not occupied.intersection(segment), "primary guides must be disjoint"
                occupied.update(segment)
                guides.append((segment, slope))
                interior.update({p: slope for p in segment[1:-1]})
    assert guides, "at least one guide is required"
    return guides, interior, sum(n == 0 for n in degree.values())


def _input(shape, bg, band, primary, secondary, rods, endpoints):
    a = np.full(shape, bg, dtype=np.int8)
    for p in rods:
        for pos in _cells(p):
            assert a[pos] == bg, "input rods overlap"
            a[pos] = p[4]
        for k in p[5]:
            a[_cell(p, k)] = secondary
    for pos in endpoints:
        assert a[pos] == bg, "an endpoint hides an input rod"
        a[pos] = primary
    a[0] = band
    return a


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    assert h >= 5 and w >= 3
    bg = int(a[1, 0])
    assert np.all(a[1] == bg), "row 1 must be the background separator"
    legend = [int(x) for x in a[0] if x != bg]
    assert 3 <= len(legend) <= 5 and len(set(legend)) == len(legend)
    primary, secondary = legend[:2]
    commands = set(legend[2:])
    endpoints = {tuple(map(int, x)) for x in np.argwhere(a[2:] == primary) + (2, 0)}
    guides, interior, unpaired = _guides(endpoints)
    restored = a.copy()
    passengers = {tuple(map(int, x)) for x in np.argwhere(a[2:] == secondary) + (2, 0)}
    membership = {}
    for r, c in passengers:
        candidates = []
        for axis, dr, dc in (("H", 0, 1), ("V", 1, 0)):
            if not (2 <= r - dr and r + dr < h and 0 <= c - dc and c + dc < w):
                continue
            color = int(a[r - dr, c - dc])
            if color == int(a[r + dr, c + dc]) and color not in (bg, primary, secondary):
                candidates.append((axis, color))
        assert len(candidates) == 1, "each dot must belong to one recoverable rod"
        membership[r, c] = candidates[0]
        restored[r, c] = candidates[0][1]
    restored[restored == primary] = bg
    seen = set()
    rods = []
    for r in range(2, h):
        for c in range(w):
            if (r, c) in seen or restored[r, c] == bg:
                continue
            color = int(restored[r, c])
            stack = [(r, c)]
            cells = set(stack)
            seen.add((r, c))
            while stack:
                x, y = stack.pop()
                for q in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if (2 <= q[0] < h and 0 <= q[1] < w and q not in seen
                            and restored[q] == color):
                        stack.append(q)
                        cells.add(q)
                        seen.add(q)
            assert len(cells) >= 2, "rods have length at least two"
            rows, cols = {x for x, y in cells}, {y for x, y in cells}
            assert len(rows) == 1 or len(cols) == 1, "a rod must be straight"
            axis = "H" if len(rows) == 1 else "V"
            rr, cc = min(rows), min(cols)
            offsets = tuple(sorted((q[1] - cc if axis == "H" else q[0] - rr)
                                   for q in cells if q in passengers))
            p = axis, rr, cc, len(cells), color, offsets
            assert _cells(p) == cells
            assert all(0 < k < p[3] - 1 and membership[_cell(p, k)] == (axis, color)
                       for k in offsets)
            rods.append(p)
    assert rods
    assert np.array_equal(_input(a.shape, bg, a[0], primary, secondary, rods, endpoints), a)
    return {"grid": a, "bg": bg, "primary": primary, "secondary": secondary,
            "commands": commands, "rods": rods, "endpoints": endpoints,
            "guides": guides, "interior": interior, "unpaired": unpaired}


def _run(rods, shape, interior, commands=(), *, latching=True, gravity=True):
    """Synchronous simulation on reconstructed rigid footprints.

    The event list is (survivor_index, offset, row, col, guide_slope, time).
    A maximal moving set handles contacts without bonding independent rods.
    """
    result = [p for p in rods if p[4] not in commands]
    pinned, fired = set(), set()
    events = []

    def activate(time):
        if not latching:
            return
        batch = []
        for i, p in enumerate(result):
            for k in p[5]:
                point = _cell(p, k)
                if (i, k) not in fired and point in interior:
                    batch.append((i, k, *point, interior[point], time))
        for e in batch:
            pinned.add(e[0])
            fired.add(e[:2])
        events.extend(batch)

    activate(0)
    if not gravity:
        return result, events, pinned
    for time in range(1, shape[0] * (len(result) + 1) + 1):
        footprints = [_cells(p) for p in result]
        targets = [{(r + 1, c) for r, c in s} for s in footprints]
        moving = {i for i, s in enumerate(targets)
                  if i not in pinned and all(r < shape[0] for r, c in s)}
        while True:
            blocked = {i for i in moving if any(targets[i] & footprints[j]
                       for j in range(len(result)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result, events, pinned
        result = [_shift(p, 1) if i in moving else p for i, p in enumerate(result)]
        activate(time)
    raise AssertionError("monotone downward motion exceeded its finite bound")


def _draw(parsed, rods, events):
    a = np.full(parsed["grid"].shape, parsed["bg"], dtype=np.int8)
    for segment, _ in parsed["guides"]:
        for point in segment:
            a[point] = parsed["primary"]
    for _, _, r, c, slope, _ in events:
        for point in _full_line((r, c), -slope, a.shape):
            a[point] = parsed["secondary"]
    for p in rods:
        for point in _cells(p):
            a[point] = p[4]
        for k in p[5]:
            a[_cell(p, k)] = parsed["secondary"]
    for point in parsed["endpoints"]:
        a[point] = parsed["primary"]
    a[:2] = parsed["grid"][:2]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _parse(input_grid)
    rods, events, _ = _run(parsed["rods"], parsed["grid"].shape,
                           parsed["interior"], parsed["commands"])
    return _draw(parsed, rods, events)


def _infinite_interiors(parsed):
    # Used only for the finite-span teaching counterfactual. Multiple candidate
    # slopes in this incorrect rule are resolved consistently for the check.
    interior = {}
    for segment, slope in parsed["guides"]:
        interior.update({point: slope for point in _full_line(segment[0], slope,
                                                            parsed["grid"].shape)})
    return interior


def _drop(rods, axis, col, length, h):
    p = axis, 2, col, length, -1, ()
    occupied = set().union(*(_cells(q) for q in rods)) if rods else set()
    if _cells(p) & occupied:
        return False
    for _ in range(h):
        target = _cells(_shift(p, 1))
        if any(r >= h for r, c in target) or target & occupied:
            rods.append(p)
            return True
        p = _shift(p, 1)
    return False


def _geometry(rng, shape, desired, finite, upper=True):
    h, w = shape
    span = int(rng.integers(8 if finite else 5, w + 1))
    left = int(rng.integers(w - span + 1))
    leg_n = int(rng.integers(8 if finite else 4, h - 6))
    row = h - leg_n - 1
    rods = [("H", row, left, span, -1, ())]
    xs = [int(rng.integers(left, left + span))]
    if rng.random() < .45:
        choices = [x for x in range(left, left + span) if abs(x - xs[0]) >= 2]
        xs.append(int(rng.choice(choices)))
    selected = set()
    for x in xs:
        selected.add(len(rods))
        rods.append(("V", row + 1, x, leg_n, -1, ()))
    if rng.random() < .55:
        choices = [x for x in range(left, left + span) if all(abs(x - y) >= 2 for y in xs)]
        if choices:
            n = int(rng.integers(2, leg_n - (5 if finite else 1)))
            rods.append(("V", h - n, int(rng.choice(choices)), n, -1, ()))
    upper_index = None
    if upper:
        x = int(rng.integers(left, left + span))
        n = int(rng.integers(3 if finite else 2, row - 2))
        upper_index = len(rods)
        rods.append(("V", row - n, x, n, -1, ()))
        cap_n = int(rng.integers(3, min(w, span + 2) + 1))
        cap_left = int(rng.integers(max(0, x - cap_n + 1), min(x, w - cap_n) + 1))
        rods.append(("H", row - n - 1, cap_left, cap_n, -1, ()))
    for _ in range(220):
        if len(rods) >= desired:
            break
        axis = "H" if rng.random() < .58 else "V"
        n = int(rng.integers(2, min(w if axis == "H" else h - 3, 12) + 1))
        col = int(rng.integers(w - n + 1)) if axis == "H" else int(rng.integers(w))
        _drop(rods, axis, col, n, h)
    assert len(rods) >= desired
    return rods, selected, upper_index


def _color(rng, rods, selected, palette):
    footprints = [_cells(p) for p in rods]
    adjacent = []
    for i, cells in enumerate(footprints):
        neighborhood = cells | {(r + dr, c + dc) for r, c in cells
                                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
        adjacent.append({j for j, other in enumerate(footprints) if j != i and neighborhood & other})
    assigned = {i: palette[0] for i in selected}
    assert all(not (adjacent[i] & selected) for i in selected)
    for i in sorted(set(range(len(rods))) - selected, key=lambda i: -len(adjacent[i])):
        forbidden = {assigned[j] for j in adjacent[i] if j in assigned} | {palette[0]}
        choices = [c for c in palette if c not in forbidden]
        if not choices:
            return None
        assigned[i] = int(rng.choice(choices))
    return [(*p[:4], assigned[i], ()) for i, p in enumerate(rods)]


def _band(rng, width, bg, primary, secondary, commands):
    band = np.full(width, bg, dtype=np.int8)
    cols = sorted(map(int, rng.choice(width, size=2 + len(commands), replace=False)))
    for col, value in zip(cols, [primary, secondary] + list(commands)):
        band[col] = value
    return band


def _extend_guides(rng, shape, rods, endpoints, count, *, exhaustive=False,
                   avoid_passenger_columns=False):
    h, w = shape
    occupied = set().union(*(_cells(p) for p in rods))
    marks = {_cell(p, k) for p in rods for k in p[5]}
    guides, _, _ = _guides(endpoints) if endpoints else ([], {}, 0)
    candidates = None
    if exhaustive:
        candidates = [(r, c, n, slope) for n in range(2, 7)
                      for r in range(2, h - n) for c in range(w)
                      for slope in (-1, 1) if 0 <= c + slope * n < w]
        rng.shuffle(candidates)
    for attempt in range(len(candidates) if exhaustive else 450):
        if len(guides) >= count:
            return endpoints
        if exhaustive:
            r, c, n, slope = candidates[attempt]
        else:
            n = int(rng.integers(2, min(h - 3, w - 1, 15) + 1))
            r = int(rng.integers(2, h - n))
            slope = int(rng.choice((-1, 1)))
            c = int(rng.integers(w - n)) + (n if slope == -1 else 0)
        a, b = (r, c), (r + n, c + slope * n)
        if a in occupied or b in occupied or a in endpoints or b in endpoints:
            continue
        try:
            new_guides, interior, _ = _guides(endpoints | {a, b})
            assert len(new_guides) == len(guides) + 1
            assert not (marks & interior.keys())
            if avoid_passenger_columns:
                old_interior = _guides(endpoints)[1]
                added = set(interior) - set(old_interior)
                assert not any(c in {q[1] for q in marks} for r, c in added)
        except AssertionError:
            continue
        endpoints = endpoints | {a, b}
        guides = new_guides
    raise AssertionError("guide placement exhausted")


def _candidate(rng, roles, mode, slope, guide_count, desired, large=False):
    bg, primary, secondary = roles
    finite, witness = mode == "finite", mode != "free"
    h = int(rng.integers(18 if finite else 14, 31))
    w = int(rng.integers(12 if finite else 9, 31))
    if large:
        h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
    shape = h, w
    geometry, selected, upper = _geometry(rng, shape, desired, finite,
                                           upper=witness or rng.random() < .65)
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    palette = palette[:int(rng.integers(3, 8))]
    rods = _color(rng, geometry, selected, palette)
    assert rods is not None
    commands = [palette[0]]
    if not witness and rng.random() < .35:
        absent = [c for c in range(10) if c not in roles and c not in {p[4] for p in rods}]
        commands = [int(rng.choice(absent))] if absent else [rods[-1][4]]
    extras = [c for c in range(10) if c not in roles and c not in commands
              and (not witness or c not in {rods[0][4], rods[upper][4], rods[upper + 1][4]})]
    if rng.random() < .35:
        commands.extend(map(int, rng.choice(extras, size=min(len(extras), int(rng.integers(1, 3))),
                                             replace=False)))
    survivors, _, _ = _run(rods, shape, {}, commands, latching=False)
    main = rods[0]
    if witness:
        ordinary_main = next(p for p in survivors if p[2:5] == main[2:5] and p[0] == "H")
        fall = ordinary_main[1] - main[1]
        assert fall >= (6 if finite else 2)
        if finite:
            diff = int(rng.integers(4, min(6, fall - 2, main[3] - 3) + 1))
            # The riser's penultimate passenger is two rows above the shelf.
            # Put its column two guide steps before the main passenger so both
            # reach this shared guide in the same movement snapshot.
            k = rods[upper][2] - main[2] + slope * 2
            assert 0 < k < main[3] - 1 and 0 < k - slope * diff < main[3] - 1
            marks = (k, k - slope * diff)
            distance = int(rng.integers(diff + 1, fall))
        else:
            k = int(rng.integers(1, main[3] - 1))
            marks = (k,)
            distance = int(rng.integers(1, fall))
        rods[0] = (*main[:5], tuple(sorted(marks)))
        # The finite witness also exposes vertical passenger carriage.
        if finite:
            p = rods[upper]
            rods[upper] = (*p[:5], (p[3] - 2,))
        point = main[1] + distance, main[2] + k
        pre_max = min(point[0] - 2, point[1] if slope == 1 else w - 1 - point[1],
                      diff - 1 if finite else 7)
        post_max = min(h - 1 - point[0], w - 1 - point[1] if slope == 1 else point[1], 7)
        assert pre_max >= (3 if finite else 1) and post_max >= 1
        pre = int(rng.integers(3 if finite else 1, pre_max + 1))
        post = int(rng.integers(1, post_max + 1))
        endpoints = {(point[0] - pre, point[1] - slope * pre),
                     (point[0] + post, point[1] + slope * post)}
    else:
        endpoints = set()
        if main[3] >= 3:
            rods[0] = (*main[:5], (int(rng.integers(1, main[3] - 1)),))
    # Extra rods may be marked, including commanded rods. The first upper stack
    # remains unmarked in a capture witness to show support without activation.
    for i, p in enumerate(rods[1:], 1):
        if witness and i in (upper, upper + 1):
            continue
        if p[3] >= 3 and rng.random() < .28:
            offsets = [int(rng.integers(1, p[3] - 1))]
            if p[3] >= 6 and rng.random() < .2:
                pool = [k for k in range(1, p[3] - 1) if abs(k - offsets[0]) >= 2]
                if pool:
                    offsets.append(int(rng.choice(pool)))
            rods[i] = (*p[:5], tuple(sorted(offsets)))
    endpoints = _extend_guides(rng, shape, rods, endpoints, guide_count)
    # Optional primary distractors retain degree zero, not an alternative pair.
    for _ in range(int(rng.integers(3))):
        occupied = set().union(*(_cells(p) for p in rods))
        for index in rng.permutation((h - 2) * w):
            p = int(index) // w + 2, int(index) % w
            if p not in occupied and p not in endpoints and not any(
                    abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in endpoints):
                endpoints.add(p)
                break
    band = _band(rng, w, bg, primary, secondary, commands)
    inp = _input(shape, bg, band, primary, secondary, rods, endpoints)
    parsed = _parse(inp)
    assert set(parsed["rods"]) == set(rods), "construction must round-trip exactly"
    assert not any(_cell(p, k) in parsed["interior"] for p in rods for k in p[5])
    final, events, _ = _run(rods, shape, parsed["interior"], commands)
    if witness:
        main_index = [p for p in rods if p[4] not in commands].index(rods[0])
        assert final[main_index][1] == main[1] + distance
        assert any(e[0] == main_index and e[1] == k and e[-1] == distance for e in events)
        assert final[main_index][1] < ordinary_main[1]
        if finite:
            alternate, _, _ = _run(rods, shape, _infinite_interiors(parsed), commands)
            assert alternate[main_index][1] < final[main_index][1]
            upper_index = [p for p in rods if p[4] not in commands].index(rods[upper])
            assert any(e[0] == upper_index and e[-1] == distance for e in events)
        output = _draw(parsed, final, events)
        # Direction and extent must have visible evidence behind foreground rods.
        assert np.sum(output[2:] == secondary) >= sum(len(p[5]) for p in final) + 3
    return {"input": inp, "output": _draw(parsed, final, events)}


def _fallback(rng, roles, slope, guide_count, desired, used):
    """Finite emergency construction, with both catch and finite-span evidence."""
    bg, primary, secondary = roles
    palette = [c for c in range(10) if c not in roles]
    # 63 distinct footprints exceed the maximum 14-pair request even with
    # identical palettes and legends. Extra rods occupy a separate solid stack
    # outside the moving shelf, so they cannot suppress the teaching event.
    for variant in rng.permutation(63):
        left = int(variant) % 9
        shape = (24 + int(variant) // 9, 30)
        c1, c2, support = ((left + 10, left + 6, left + 1) if slope == 1
                           else (left + 6, left + 10, left + 15))
        rods = [("H", 6, left, 17, palette[1], tuple(sorted((c1 - left, c2 - left)))),
                ("V", 7, support, shape[0] - 7, palette[0], ()),
                ("V", 3, left + 8, 3, palette[2], (1,)),
                ("H", 2, left + 6, 5, palette[3], ())]
        side_col = 0 if left >= 3 else left + 18
        for j in range(max(0, desired - 4)):
            rods.append(("H", shape[0] - 1 - j, side_col, 2, palette[2 + j % 2], ()))
        endpoints = {(10, c1 - 3 * slope), (15, c1 + 2 * slope)}
        try:
            endpoints = _extend_guides(rng, shape, rods, endpoints, guide_count, exhaustive=True,
                                       avoid_passenger_columns=True)
            band = _band(rng, 30, bg, primary, secondary, [palette[0]])
            inp = _input(shape, bg, band, primary, secondary, rods, endpoints)
            parsed = _parse(inp)
            assert set(parsed["rods"]) == set(rods)
            final, events, _ = _run(rods, shape, parsed["interior"], [palette[0]])
            main_index = [p for p in rods if p[4] != palette[0]].index(rods[0])
            assert final[main_index][1] == 13
            assert any(e[0] == main_index and e[4] == slope for e in events)
            assert not any(_cell(p, k) in parsed["interior"] for p in rods for k in p[5])
            infinite, _, _ = _run(rods, shape, _infinite_interiors(parsed), [palette[0]])
            assert infinite[main_index][1] < final[main_index][1]
            if _key(inp) not in used:
                return {"input": inp, "output": _draw(parsed, final, events)}
        except AssertionError:
            continue
    raise AssertionError("emergency construction exhausted")


def _make_pair(rng, roles, mode, slope, guide_count, desired, used, large=False):
    for _ in range(100):
        try:
            pair = _candidate(rng, roles, mode, slope, guide_count, desired, large)
            if _key(pair["input"]) not in used:
                return pair
        except (AssertionError, ValueError, StopIteration):
            continue
    return _fallback(rng, roles, slope, guide_count, desired, used)


def _integer(value, name, lower, upper=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < lower or (upper is not None and value > upper)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fixed_roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
    rekey = bool(rng.integers(2))
    shift_kind = int(rng.integers(2))
    first_slope = int(rng.choice((-1, 1)))
    used = set()
    puzzle = {"train": [], "test": []}

    def roles():
        return tuple(map(int, rng.choice(10, size=3, replace=False))) if rekey else fixed_roles

    for i in range(num_train):
        mode = "capture" if i == 0 else "finite" if i == 1 else str(rng.choice(("capture", "finite", "free")))
        slope = first_slope if i == 0 else -first_slope if i == 1 else int(rng.choice((-1, 1)))
        guide_count = 1 if shift_kind == 0 else int(rng.integers(1, 3))
        pair = _make_pair(rng, roles(), mode, slope, guide_count, int(rng.integers(4, 10)), used)
        puzzle["train"].append(pair)
        used.add(_key(pair["input"]))
    train_features = [features(p) for p in puzzle["train"]]
    maximum_guides = max(f["guides"] for f in train_features)
    maximum_rods = max(f["rods"] for f in train_features)
    for _ in range(num_test):
        guide_count = int(rng.integers(maximum_guides + 1, 4)) if shift_kind == 0 else int(rng.integers(1, 4))
        desired = int(rng.integers(4, 11)) if shift_kind == 0 else maximum_rods + int(rng.integers(1, 4))
        pair = _make_pair(rng, roles(), str(rng.choice(("capture", "finite", "free"))),
                          int(rng.choice((-1, 1))), guide_count, desired, used,
                          large=shift_kind == 1)
        puzzle["test"].append(pair)
        used.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def features(pair):
    p = _parse(pair["input"])
    shape, rods, commands = p["grid"].shape, p["rods"], p["commands"]
    survivors = [r for r in rods if r[4] not in commands]
    final, events, pinned = _run(rods, shape, p["interior"], commands)
    ordinary, _, _ = _run(rods, shape, {}, commands, latching=False)
    infinite, _, _ = _run(rods, shape, _infinite_interiors(p), commands)
    no_gravity, _, _ = _run(rods, shape, p["interior"], commands, gravity=False)
    footprints = [_cells(r) for r in final]
    supporting_pins = sum(any({(r + 1, c) for r, c in cells} & footprints[j]
                              for j in pinned if j != i) for i, cells in enumerate(footprints))
    return {"shape": shape, "roles": (p["bg"], p["primary"], p["secondary"]),
            "rods": len(rods), "guides": len(p["guides"]), "unpaired": p["unpaired"],
            "h_lengths": [r[3] for r in rods if r[0] == "H"],
            "v_lengths": [r[3] for r in rods if r[0] == "V"],
            "guide_lengths": [len(s) for s, _ in p["guides"]],
            "commands": len(commands), "deleted": len(rods) - len(survivors),
            "marked": sum(len(r[5]) for r in rods),
            "marked_v": any(r[0] == "V" and r[5] for r in rods),
            "inactive": sum(len(r[5]) for r in final) - len(events),
            "pins": len(pinned), "events": len(events), "supporting_pins": supporting_pins,
            "active_slopes": sorted({e[4] for e in events}),
            "active_vertical": any(final[e[0]][0] == "V" for e in events),
            "event_times": [e[-1] for e in events],
            "fall_distances": [q[1] - r[1] for r, q in zip(survivors, final)],
            "without_B_changes_rods": final != ordinary,
            "without_A_motion_changes_rods": final != no_gravity,
            "infinite_changes_rods": final != infinite,
            "simultaneous": len({e[-1] for e in events}) < len(events),
            "palette_size": len({r[4] for r in rods})}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and np.array_equal(inp[:2], out[:2])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            initially_settled, _, _ = _run(p["rods"], inp.shape, {}, latching=False)
            assert initially_settled == p["rods"], "input construction is unstable"
            assert not any(_cell(r, k) in p["interior"] for r in p["rods"] for k in r[5])
            survivors = [r for r in p["rods"] if r[4] not in p["commands"]]
            final, events, pinned = _run(p["rods"], inp.shape, p["interior"], p["commands"])
            assert len(final) == len(survivors)
            assert all(a[0] == b[0] and a[2:] == b[2:] and a[1] <= b[1]
                       for a, b in zip(survivors, final)), "rigidity or marker offsets changed"
            cells = [_cells(r) for r in final]
            assert sum(map(len, cells)) == len(set().union(*cells)) if cells else True
            for i, r in enumerate(final):
                target = {(x + 1, y) for x, y in cells[i]}
                assert i in pinned or any(x >= inp.shape[0] for x, y in target) or any(
                    target & cells[j] for j in range(len(final)) if i != j)
            assert all(e[0] in pinned and _cell(final[e[0]], e[1]) == e[2:4]
                       and p["interior"][e[2:4]] == e[4] for e in events)
            assert len({e[:2] for e in events}) == len(events)
            assert np.array_equal(out, _draw(p, final, events)), "incorrect rendered target"
            assert not np.array_equal(inp, out)
            records[split].append(features(pair))
    tr, te = records["train"], records["test"]
    assert set().union(*(set(f["active_slopes"]) for f in tr)) == {-1, 1}
    assert any(f["without_B_changes_rods"] for f in tr)
    assert any(f["without_A_motion_changes_rods"] for f in tr)
    assert any(f["infinite_changes_rods"] and f["inactive"] for f in tr)
    assert any(f["supporting_pins"] for f in tr)
    assert any(f["marked_v"] for f in tr)
    assert any(f["active_vertical"] and f["simultaneous"] for f in tr)
    assert (min(f["guides"] for f in te) > max(f["guides"] for f in tr)
            or min(f["rods"] for f in te) > max(f["rods"] for f in tr)), "missing structural split shift"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _mechanics_tests():
    shape = (13, 12)
    beam = ("H", 4, 1, 9, 2, (5, 2))
    post = ("V", 5, 2, 8, 3, ())
    upper = ("V", 2, 4, 2, 4, ())
    # At t=5, passenger c6 reaches (9,6). Passenger c3 passed the infinite
    # continuation at t=2, outside finite endpoints (8,5) and (10,7).
    rods = [beam, post, upper]
    guide = {(9, 6): 1}
    final, events, pins = _run(rods, shape, guide, {3})
    assert [p[1] for p in final] == [9, 7] and pins == {0}
    assert events == [(0, 5, 9, 6, 1, 5)]
    ordinary, _, _ = _run(rods, shape, {}, {3}, latching=False)
    assert ordinary[0][1] == 12 and ordinary[1][1] == 10
    infinite = {p: 1 for p in _full_line((9, 6), 1, shape)}
    too_early, _, _ = _run(rods, shape, infinite, {3})
    assert too_early[0][1] == 6
    # Time zero, simultaneous two-dot pinning, and whole-rod removal.
    initial = {_cell(beam, k): (-1 if k == 2 else 1) for k in beam[5]}
    caught, fired, pinned = _run([beam], shape, initial)
    assert caught == [beam] and len(fired) == 2 and pinned == {0}
    assert _run([beam], shape, initial, {2}) == ([], [], set())
    # Two separate carriers caught in the same snapshot do not depend on order.
    pair = [("H", 3, 0, 3, 2, (1,)), ("H", 3, 7, 3, 4, (1,))]
    simultaneous, e, pins = _run(pair, shape, {(5, 1): 1, (5, 8): -1})
    assert [p[1] for p in simultaneous] == [5, 5] and len(pins) == 2
    assert all(event[-1] == 2 for event in e)
    # Exercise the public input-only interface on actual encoded scenes, with
    # coordinate expectations fixed above, including the unsampled t=0 case.
    band = np.array([1, 6, 3] + [0] * 9, dtype=np.int8)
    endpoints = {(8, 5), (10, 7)}
    inp = _input(shape, 0, band, 1, 6, rods, endpoints)
    untouched = inp.copy()
    out = solve(inp)
    assert np.array_equal(inp, untouched)
    assert np.all(out[9, 1:10][[0, 1, 3, 4, 6, 7, 8]] == 2)
    assert out[9, 3] == out[9, 6] == 6
    assert out[7, 4] == out[8, 4] == 4
    assert not np.any(out[2:] == 3) and out[5, 10] == 6 and out[12, 3] == 6
    assert np.array_equal(solve(inp.tolist()), out)
    zero_band = band.copy()
    zero_band[2] = 5  # An absent construction identity leaves the supports intact.
    zero = _input(shape, 0, zero_band, 1, 6, rods, {(3, 5), (5, 7)})
    zero_out = solve(zero)
    assert zero_out[4, 6] == 6 and np.all(zero_out[5:13, 2] == 3)
    assert zero_out[2, 8] == 6 and zero_out[3, 7] == 6
    # Color equality and horizontal symmetry must commute with the full solver.
    permutation = np.array([8, 4, 7, 0, 9, 2, 5, 1, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[out])
    reflected = inp.copy()
    reflected[2:] = inp[2:, ::-1]
    reflected_out = out.copy()
    reflected_out[2:] = out[2:, ::-1]
    assert np.array_equal(solve(reflected), reflected_out)
    # Two real, disjoint finite guides produce a simultaneous two-carrier catch.
    twin_rods = pair + [("V", 4, 1, 9, 3, ()), ("V", 4, 8, 9, 3, ())]
    twin_points = {(4, 0), (6, 2), (4, 9), (6, 7)}
    twin = _input(shape, 0, band, 1, 6, twin_rods, twin_points)
    twin_out = solve(twin)
    assert twin_out[5, 0] == twin_out[5, 2] == 2
    assert twin_out[5, 7] == twin_out[5, 9] == 4
    assert twin_out[5, 1] == twin_out[5, 8] == 6


def self_test():
    _mechanics_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len({r["shape"] for r in records}) >= 50
    assert len({r["roles"] for r in records}) >= 20
    assert len({r["rods"] for r in records}) >= 8
    assert {1, 2, 3} <= {r["guides"] for r in records}
    assert {1, 2, 3} <= {r["commands"] for r in records}
    assert len({n for r in records for n in r["h_lengths"]}) >= 12
    assert len({n for r in records for n in r["v_lengths"]}) >= 10
    assert len({n for r in records for n in r["guide_lengths"]}) >= 8
    assert any(r["pins"] >= 2 for r in records)
    assert any(len(set(r["fall_distances"]) - {0}) >= 2 for r in records)
    assert any(r["events"] == 0 for r in records)
    assert any(r["unpaired"] for r in records)
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    fallback_used = set()
    for index in range(14):
        pair = _fallback(np.random.default_rng(800 + index), (0, 1, 6),
                         1 if index % 2 else -1, 3, 12, fallback_used)
        fallback_used.add(_key(pair["input"]))
        assert np.array_equal(solve(pair["input"]), pair["output"])
        f = features(pair)
        assert f["without_B_changes_rods"] and f["infinite_changes_rods"]
    assert len(fallback_used) == 14
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(617, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutation rejection exercises the validator rather than merely the renderer.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][1, 0] = (int(bad["test"][0]["output"][1, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
