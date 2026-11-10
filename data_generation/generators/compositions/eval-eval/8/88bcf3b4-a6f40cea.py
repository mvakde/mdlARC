"""X-ray tension screen: reconstruct plates, stretch one counted thread.

Only NumPy and the standard library are used. No files are read or written.
The upper hull is a family convention, including upward digital rounding.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
    return g


def _bbox(points):
    r, c = zip(*points)
    return min(r), max(r), min(c), max(c)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)} | {(b, c) for c in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)} | {(y, r) for y in range(t, b + 1)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _components(points, directions=DIR4):
    unseen, groups = set(points), []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        group, todo = {p}, [p]
        while todo:
            y, x = todo.pop()
            for dy, dx in directions:
                q = y + dy, x + dx
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _local(p, endpoint, forward, lateral):
    dr, dc = p[0] - endpoint[0], p[1] - endpoint[1]
    return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]


def _global(p, endpoint, forward, lateral):
    u, v = p
    return (endpoint[0] + u * forward[0] + v * lateral[0],
            endpoint[1] + u * forward[1] + v * lateral[1])


def _hull_route(plates, length):
    """Least concave clearance roof; each plate is (near, far, low, high).

    There is one thread cell per forward coordinate. The outgoing ray has
    slope -1. Exact integer arithmetic rounds every interpolated height up.
    """
    assert plates and length >= 1
    assert all(2 <= a <= b and c <= d <= a for a, b, c, d in plates)
    far = max(b for a, b, c, d in plates)
    intercept = max(b + d + 1 for a, b, c, d in plates)
    end = max(far + 4, length + 1)
    levels = {0: 1, end: intercept - end}
    for a, b, c, d in plates:
        for u in (a, b):
            levels[u] = max(levels.get(u, -1000), d + 1)
    hull = []
    for p in sorted(levels.items()):
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = (b[0] - a[0]) * (p[1] - b[1]) - (b[1] - a[1]) * (p[0] - b[0])
            if cross < 0:
                break
            hull.pop()
        hull.append(p)
    route, segment = [], 0
    for u in range(length):
        while u > hull[segment + 1][0]:
            segment += 1
        (x, y), (xx, yy) = hull[segment:segment + 2]
        denominator = xx - x
        numerator = y * denominator + (u - x) * (yy - y)
        v = -((-numerator) // denominator)
        route.append((u, v))
    assert route[0] == (0, 1)
    assert all(abs(a[1] - b[1]) <= 1 for a, b in zip(route, route[1:]))
    assert all(v > d for u, v in route for a, b, c, d in plates if a <= u <= b)
    return route, hull


def _read_loops(g, points, mask):
    parts = _components(points)
    candidates = []
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            visible = first if i == j else first | parts[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            full = _perimeter(box)
            if full - mask != visible:
                continue
            phase, witnesses = {}, Counter()
            for p in sorted(visible):
                k, color = sum(p) % 2, int(g[p])
                if k in phase and phase[k] != color:
                    break
                phase[k] = color
                witnesses[k] += 1
            else:
                if len(phase) == 2 and min(witnesses.values()) >= 2:
                    candidates.append({"parts": {i, j}, "box": box, "full": full,
                                       "visible": visible, "phase": phase})
    covers = []

    def visit(remaining, chosen, occupied, colors):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for q in candidates:
            palette = set(q["phase"].values())
            if (first in q["parts"] and q["parts"] <= remaining
                    and not q["full"] & occupied and not palette & colors):
                visit(remaining - q["parts"], chosen + [q], occupied | q["full"], colors | palette)

    visit(set(range(len(parts))), [], set(), set())
    assert len(covers) == 1 and covers[0], "ambiguous rectangular plate completion"
    return covers[0]


def _read(g):
    """All role assignment uses input pixels; no output shape or seed is used."""
    g = _grid(g)
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    positions = {int(c): {tuple(map(int, p)) for p in np.argwhere(g == c)} for c in np.unique(g)}
    frames = []
    for color, cells in positions.items():
        if color == bg:
            continue
        box = _bbox(cells)
        t, b, l, r = box
        if b - t < 3 or r - l < 3 or cells != _perimeter(box):
            continue
        inner = g[t + 1:b, l + 1:r]
        if np.all(inner == inner[0, 0]) and int(inner[0, 0]) not in (bg, color):
            frames.append((box, color, int(inner[0, 0])))
    assert len(frames) == 1, "require one complete opaque screen"
    screen, frame, fill = frames[0]
    mask = _area(screen)
    assert positions[fill] == mask - _perimeter(screen)
    available = {c: cells for c, cells in positions.items() if c not in (bg, frame, fill)}
    scenes = []
    for support_color, support in available.items():
        if not 3 <= len(support) <= 6:
            continue
        t, b, l, r = _bbox(support)
        if not ((t == b or l == r) and len(support) == max(b - t, r - l) + 1):
            continue
        positive = (1, 0) if l == r else (0, 1)
        for endpoint, forward in ((max(support), positive), (min(support), tuple(-v for v in positive))):
            for lateral in ((forward[1], -forward[0]), (-forward[1], forward[0])):
                attachment = _global((0, 1), endpoint, forward, lateral)
                for thread_color, slack in available.items():
                    if thread_color == support_color or attachment not in slack or len(slack) < 4:
                        continue
                    local_slack = {_local(p, endpoint, forward, lateral) for p in slack}
                    if any(u > 0 or v < 1 for u, v in local_slack) or len(_components(slack, DIR8)) != 1:
                        continue
                    rest = set().union(*(cells for c, cells in available.items()
                                         if c not in (support_color, thread_color)))
                    try:
                        loops = _read_loops(g, rest, mask)
                        plates = [_bbox({_local(p, endpoint, forward, lateral) for p in q["full"]}) for q in loops]
                        route, hull = _hull_route(plates, len(slack))
                    except (AssertionError, ValueError):
                        continue
                    scenes.append({"background": bg, "screen": screen, "fill": fill,
                                   "frame": frame, "support_color": support_color,
                                   "thread_color": thread_color, "support": support,
                                   "slack": slack, "endpoint": endpoint, "forward": forward,
                                   "lateral": lateral, "loops": loops, "plates": plates,
                                   "route": route, "hull": hull})
    assert len(scenes) == 1, "support, thread, and plates must have unique roles"
    return scenes[0]


def _render_answer(g, s, *, length=None, visible_only=False, no_plates=False):
    t, b, l, r = s["screen"]
    out = np.full((b - t - 1, r - l - 1), s["fill"], dtype=np.int8)
    for loop in s["loops"]:
        for y, x in loop["full"]:
            if t < y < b and l < x < r:
                out[y - t - 1, x - l - 1] = loop["phase"][(y + x) % 2]
    n = len(s["slack"]) if length is None else length
    plates = s["plates"]
    if visible_only:
        # Deliberate wrong rule: use the visible plate cells as point obstacles.
        pts = [_local(p, s["endpoint"], s["forward"], s["lateral"])
               for loop in s["loops"] for p in loop["visible"]]
        plates = [(u, u, v, v) for u, v in pts]
    if no_plates:
        route = [(u, 1) for u in range(n)]
    else:
        route, _ = _hull_route(plates, n)
    for p in route:
        y, x = _global(p, s["endpoint"], s["forward"], s["lateral"])
        if t < y < b and l < x < r:
            out[y - t - 1, x - l - 1] = s["thread_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict only from the query input, under the documented fixed grammar."""
    g = _grid(input_grid)
    return _render_answer(g, _read(g))


def _slack(rng, length, depth, width, style):
    cells = {(0, 1), (-1, 2)}
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in DIR8
                           if -depth <= u + du <= -1 and 2 <= v + dv <= width} - cells)
        assert frontier
        weights = []
        for p in frontier:
            degree = sum((p[0] + a, p[1] + b) in cells for a, b in DIR8)
            weights.append((degree + 1) ** 2 if style == 0 else
                           1 / (degree + 1) ** 2 if style == 1 else
                           1 + 4 * (p[0] == -1) if style == 2 else
                           1 + 4 * (p[1] == 2) if style == 3 else 1)
        weights = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _make_pair(rng, count, partial, require_bridge, pattern_lesson=None):
    """Construct one scene directly, not by composing parent generators."""
    for attempt in range(512):
        near = int(rng.integers(2, 5))
        spans = []
        for k in range(count):
            width = int(rng.integers(3, 5 if count <= 2 else 4))
            spans.append((near, near + width - 1))
            near += width + int(rng.integers(1, 3 if count >= 3 else 4))
        if spans[-1][1] > 21:
            continue
        first_hi = spans[0][0] if count > 1 else int(rng.integers(1, spans[0][0] + 1))
        last_hi = int(rng.choice([c for c in range(1, 6) if c != first_hi])) if count > 1 else first_hi
        plates = []
        for k, (a, b) in enumerate(spans):
            if k == 0:
                hi = first_hi
            elif k == count - 1:
                hi = last_hi
            else:
                hi = int(rng.integers(-2, min(a, 10) + 1))
            lo = hi - int(rng.integers(2, 7))
            plates.append((a, b, lo, hi))
        # Hide an actual hull contact corner, with visible witnesses for all
        # four box extrema. Other plates can be higher, lower, or skipped.
        _, roof = _hull_route(plates, spans[-1][1] + 4)
        contacts = [(plate, end) for plate in plates for end in (plate[0], plate[1])
                    if (end, plate[3] + 1) in roof]
        target_plate, contact = contacts[int(rng.integers(len(contacts)))]
        a, z, lo, hi = target_plate
        left = int(rng.integers(lo + 1, hi))
        right = hi + int(rng.integers(2, 5))
        if contact == a:
            top = max(1, a - int(rng.integers(2, 7)))
            bottom = int(rng.integers(a, z))
        else:
            top = int(rng.integers(a + 1, z + 1))
            bottom = z + int(rng.integers(2, 7))
        screen = top, bottom, left, right
        if bottom - top < 3 or right - left < 3:
            continue
        if partial and bottom - top < 4:
            continue
        full_length = spans[-1][1] + int(rng.integers(1, 4) if count >= 3 else rng.integers(4, 7))
        length = int(rng.integers(top + 3, bottom)) if partial else full_length
        depth = int(rng.integers(3, 6))
        slack_width = max(int(rng.integers(4, 9)), (length - 2) // depth + 2)
        support = {(-u, 0) for u in range(int(rng.integers(3, 7)))}
        slack = _slack(rng, length, depth, slack_width, int(rng.integers(5)))
        route, hull = _hull_route(plates, length)
        all_cells = support | slack | _area(screen) | set(route)
        for plate in plates:
            all_cells |= _perimeter(plate)
        u0, u1, v0, v1 = _bbox(all_cells)
        margins = list(map(int, rng.integers(1, 3, size=4)))
        h = u1 - u0 + 1 + margins[0] + margins[1]
        w = v1 - v0 + 1 + margins[2] + margins[3]
        if max(h, w) > 30:
            continue
        du, dv = margins[0] - u0, margins[2] - v0
        colors = list(map(int, rng.permutation(10)))
        bg, frame, fill, support_color, thread_color = colors[:5]
        budget = colors[5:]
        palettes = []
        for k in range(count):
            extra = (False if pattern_lesson == 0 else
                     plates[k] == target_plate if pattern_lesson == 1 else
                     len(budget) > count - k and rng.random() < .6)
            palettes.append([budget.pop()] + ([budget.pop()] if extra else []))
        under = np.full((h, w), bg, dtype=np.int8)
        for plate, palette in zip(plates, palettes):
            phase = int(rng.integers(2))
            for u, v in _perimeter(plate):
                under[u + du, v + dv] = palette[(u + v + phase) % len(palette)]
        for u, v in support:
            under[u + du, v + dv] = support_color
        for u, v in slack:
            under[u + du, v + dv] = thread_color
        truth = under.copy()
        for u, v in slack:
            truth[u + du, v + dv] = bg
        for u, v in route:
            assert truth[u + du, v + dv] == bg
            truth[u + du, v + dv] = thread_color
        t, b, l, r = top + du, bottom + du, left + dv, right + dv
        output = truth[t + 1:b, l + 1:r].copy()
        output[output == bg] = fill
        inp = under.copy()
        inp[t:b + 1, l:r + 1] = fill
        for p in _perimeter((t, b, l, r)):
            inp[p] = frame
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        inp, output = np.rot90(inp, turns), np.rot90(output, turns)
        if flip:
            inp, output = np.fliplr(inp), np.fliplr(output)
        inp, output = inp.copy(), output.copy()
        try:
            s = _read(inp)
            assert len(s["loops"]) == count
            assert np.array_equal(_render_answer(inp, s), output)
            f = _features(inp, s)
            assert f["thread_pixels"] >= 2 and f["hidden_plate_pixels"] >= 2
            assert f["occlusion_changes_route"] and f["routing_matters"]
            if require_bridge:
                assert f["bridge"] and f["fractional_hull"] and f["active_plates"] >= 2
                assert f["visible_bridge_pixels"] >= 1
            if partial:
                assert f["finite_budget_matters"]
            if pattern_lesson == 1:
                assert f["hidden_striped"] >= 1
        except AssertionError:
            continue
        return {"input": inp, "output": output}
    raise AssertionError("bounded scene construction exhausted")


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            plates = (1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))) if split == "train" else int(rng.integers(3, 5))
            partial = i == 1 if split == "train" else bool(rng.integers(2))
            for retry in range(12):
                lesson = i if split == "train" and i < 2 else None
                pair = _make_pair(rng, plates, partial, plates >= 2, lesson)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded uniqueness attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(g, s=None):
    s = _read(g) if s is None else s
    answer = _render_answer(g, s)
    t, b, l, r = s["screen"]
    hidden = {(y, x) for q in s["loops"] for y, x in q["full"] if t < y < b and l < x < r}
    long_length = max(p[1] for p in s["plates"]) + 8
    active = sum(any(v == d + 1 and u in (a, z) for u, v in s["hull"])
                 for a, z, c, d in s["plates"])
    bridge_cells = [(u, v) for u, v in s["route"]
                 if min(p[0] for p in s["plates"]) < u < max(p[1] for p in s["plates"])
                 and all(not a <= u <= z for a, z, c, d in s["plates"])
                 and v > max(d + 1 - min(abs(u - a), abs(u - z)) for a, z, c, d in s["plates"])
                 ]
    visible_bridge = sum(t < y < b and l < x < r for y, x in
                         (_global(p, s["endpoint"], s["forward"], s["lateral"]) for p in bridge_cells))
    fractional = any(x > 0 and xx <= max(p[1] for p in s["plates"])
                     and (yy - y) % (xx - x) != 0
                     for (x, y), (xx, yy) in zip(s["hull"], s["hull"][1:]))
    return {"count": len(s["loops"]), "shape": tuple(g.shape), "output_shape": tuple(answer.shape),
            "background": s["background"], "length": len(s["slack"]),
            "direction": s["forward"], "side": s["lateral"], "bridge": bool(bridge_cells),
            "visible_bridge_pixels": visible_bridge,
            "active_plates": active, "fractional_hull": fractional,
            "striped": sum(len(set(q["phase"].values())) == 2 for q in s["loops"]),
            "hidden_striped": sum(len(set(q["phase"].values())) == 2 and
                                  {sum(p) % 2 for p in q["full"] if t < p[0] < b and l < p[1] < r} == {0, 1}
                                  for q in s["loops"]),
            "thread_pixels": int(np.sum(answer == s["thread_color"])),
            "hidden_plate_pixels": len(hidden),
            "occlusion_changes_route": not np.array_equal(answer, _render_answer(g, s, visible_only=True)),
            "routing_matters": not np.array_equal(answer, _render_answer(g, s, no_plates=True)),
            "finite_budget_matters": not np.array_equal(answer, _render_answer(g, s, length=long_length))}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            s = _read(inp)
            assert np.array_equal(out, _render_answer(inp, s)), "wrong reconstructed scene"
            # Check geometry and occlusion evidence, not just a whole-grid hash.
            assert 1 <= len(s["loops"]) <= 4
            for q in s["loops"]:
                assert _bbox(q["visible"]) == q["box"]
                assert len(set(q["phase"].values())) in (1, 2)
            assert s["route"][0] == (0, 1)
            assert len(s["route"]) == len(s["slack"])
            assert len(_components(s["slack"], DIR8)) == 1
            for p in s["route"]:
                y, x = _global(p, s["endpoint"], s["forward"], s["lateral"])
                assert 0 <= y < inp.shape[0] and 0 <= x < inp.shape[1]
                assert not any((y, x) in _area(q["box"]) for q in s["loops"])
            f = _features(inp, s)
            assert f["routing_matters"] and f["occlusion_changes_route"]
            assert f["thread_pixels"] >= 2 and f["hidden_plate_pixels"] >= 2
            features[split].append(f)
    train = features["train"]
    assert train[0]["count"] == 1 and train[1]["count"] == 2
    assert train[1]["bridge"] and train[1]["fractional_hull"]
    assert train[1]["visible_bridge_pixels"] >= 1 and train[1]["hidden_striped"] >= 1
    assert train[1]["finite_budget_matters"]
    assert all(f["count"] <= 2 for f in train)
    assert all(f["count"] >= 3 and f["active_plates"] >= 2 for f in features["test"])


def self_test():
    """Bounded, in-memory 42-seed audit. Returns compact serializable metrics."""
    # Hand-computed regression: unequal plates need a gap-spanning 1/3 slope.
    route, hull = _hull_route([(2, 4, -1, 2), (8, 10, -1, 4)], 14)
    assert route == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 4), (6, 5),
                     (7, 5), (8, 5), (9, 5), (10, 5), (11, 4), (12, 3), (13, 2)]
    measured, signatures, normalized = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                digest.update(bytes(inp.shape) + inp.tobytes() + bytes(out.shape) + out.tobytes())
                measured.append(_features(inp))
                colors = {}
                normalized.add((inp.shape, bytes(colors.setdefault(int(c), len(colors)) for c in inp.flat)))
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(normalized) >= 180
    assert {f["count"] for f in measured} == {1, 2, 3, 4}
    assert len({f["shape"] for f in measured}) >= 40
    assert len({f["output_shape"] for f in measured}) >= 30
    assert {f["background"] for f in measured} == set(range(10))
    assert {f["direction"] for f in measured} == set(DIR4)
    assert any(f["striped"] for f in measured) and any(not f["striped"] for f in measured)
    assert any(f["finite_budget_matters"] for f in measured)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(991, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (4, 4), (2, 3), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(57, 2, 1), generate_puzzle(57, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    damaged = {s: [{k: g.copy() for k, g in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")
    return {"seeds": 42, "pairs": len(measured), "distinct_puzzles": len(signatures),
            "normalized_inputs": len(normalized), "input_shapes": len({f["shape"] for f in measured}),
            "output_shapes": len({f["output_shape"] for f in measured}),
            "plate_counts": dict(sorted(Counter(f["count"] for f in measured).items())),
            "bridge_pairs": sum(f["bridge"] for f in measured),
            "fractional_hull_pairs": sum(f["fractional_hull"] for f in measured),
            "occlusion_route_witnesses": sum(f["occlusion_changes_route"] for f in measured),
            "finite_budget_witnesses": sum(f["finite_budget_matters"] for f in measured)}
