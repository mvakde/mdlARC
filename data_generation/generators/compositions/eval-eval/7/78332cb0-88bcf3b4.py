"""Tension-revealed ports: length-limited cords determine a panel chain.

All semantics are visible in each input. No parent code, files, generation
metadata, examples, or random state are used by solve.
"""
from itertools import permutations
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=True):
    remaining, result = set(cells), []
    dirs = _D8 if diagonal else _D4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in dirs:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _cells(panel, color):
    return {tuple(map(int, p)) for p in np.argwhere(panel == color)}


def _local(p, row, tip, side):
    return p[1] - tip, side * (p[0] - row)


def _global(p, row, tip, side):
    u, v = p
    return row + side * v, tip + u


def _route(obstacle, length):
    """Actual-cell clearance envelope with source-evidenced digital tangents."""
    assert obstacle and min(u for u, v in obstacle) >= 1
    peak = max(v for u, v in obstacle)
    assert peak >= 0
    assert peak == 0 or max(v - u for u, v in obstacle) == 0
    first_peak = min(u for u, v in obstacle if v == peak)
    far = max(u for u, v in obstacle)
    route = [(u, 1 if peak == 0 and u <= first_peak else
              max(v + 1 - abs(u - t) for t, v in obstacle))
             for u in range(max(length, far + 4))]
    slopes = [q[1] - p[1] for p, q in zip(route, route[1:])]
    assert route[0] == (0, 1)
    assert all(d in (-1, 0, 1) for d in slopes)
    assert slopes == sorted(slopes, reverse=True), "unsupported exposed contour"
    assert not set(route) & obstacle
    return route[:length]


def _sweep_route(obstacle, length):
    """Construction check: two clearance sweeps, rather than the analytic sum."""
    n = max(length, max(u for u, v in obstacle) + 4)
    heights = [-1000] * n
    for u, v in obstacle:
        heights[u] = max(heights[u], v + 1)
    for u in range(1, n):
        heights[u] = max(heights[u], heights[u - 1] - 1)
    for u in range(n - 2, -1, -1):
        heights[u] = max(heights[u], heights[u + 1] - 1)
    if max(v for u, v in obstacle) == 0:
        first = min(u for u, v in obstacle if v == 0)
        heights[:first + 1] = [1] * (first + 1)
    return list(enumerate(heights[:length]))


def _partition(inp):
    candidates = []
    h, w = inp.shape
    for divider in map(int, np.unique(inp)):
        rows = list(map(int, np.flatnonzero(np.all(inp == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(inp == divider, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        heights = [b - a - 1 for a, b in zip(rr, rr[1:])]
        widths = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(heights + widths) < 5 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        panels = [inp[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 5 or any(np.any(p == divider) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        # Initial border contacts are ONLY inlet rails, on one common side.
        borders = [np.concatenate([p[:, 0] for p in panels]),
                   np.concatenate([p[-1, :] for p in panels]),
                   np.concatenate([p[:, -1] for p in panels]),
                   np.concatenate([p[0, :] for p in panels])]
        sides = [i for i, b in enumerate(borders) if np.any(b != bg)]
        if len(sides) != 1:
            continue
        rotation = sides[0]  # left, bottom, right, top after 0/1/2/3 CCW turns
        colors = set(map(int, borders[rotation])) - {bg}
        if len(colors) != 1:
            continue
        rail = colors.pop()
        foreground = set(map(int, np.unique(inp))) - {bg, divider}
        if len(foreground) != 3 or rail not in foreground:
            continue
        canonical = [np.rot90(p, -rotation).copy() for p in panels]
        candidates.append(dict(panels=canonical, original_panels=panels,
                               rotation=rotation, bg=bg, divider=divider,
                               rail=rail, other=sorted(foreground - {rail}),
                               layout=(len(heights), len(widths))))
    assert len(candidates) == 1, "partition and directed inlet side must be unique"
    return candidates[0]


def _decode_panel(panel, bg, rail, cord, obstacle_color):
    h, w = panel.shape
    rails = _components(_cells(panel, rail), diagonal=False)
    strands = _components(_cells(panel, cord))
    obstacles = _components(_cells(panel, obstacle_color))
    assert 1 <= len(rails) <= 2 and len(strands) == len(obstacles) == len(rails)
    modes, inlets, candidates = [], [], []
    for support in rails:
        assert len({r for r, c in support}) == 1 and len(support) >= 2
        row = next(iter(support))[0]
        lo, tip = min(c for r, c in support), max(c for r, c in support)
        assert support == {(row, c) for c in range(lo, tip + 1)}
        assert 1 <= row < h - 1 and 2 <= tip <= w - 3
        modes.append(lo == 0)
        if lo == 0:
            inlets.append(row)
        choices = []
        for j, strand in enumerate(strands):
            for side in (-1, 1):
                attachment = row + side, tip
                slack = {_local(p, row, tip, side) for p in strand}
                if (attachment not in strand or len(strand) < 2 or
                        any(u > 0 or v < 1 for u, v in slack) or
                        any(c == 0 or c == w - 1 or r in (0, h - 1) for r, c in strand)):
                    continue
                for k, obstacle in enumerate(obstacles):
                    local_obstacle = {_local(p, row, tip, side) for p in obstacle}
                    if any(c in (0, w - 1) or r in (0, h - 1) for r, c in obstacle):
                        continue
                    try:
                        full_path = [_global(p, row, tip, side)
                                     for p in _route(local_obstacle, max(len(strand), w - tip))]
                        path = full_path[:len(strand)]
                    except AssertionError:
                        continue
                    if any(not 1 <= r < h - 1 for r, c in full_path if c < w):
                        continue
                    potential = {p for p in full_path if p[1] < w}
                    visible = {p for p in path if 0 <= p[0] < h and 0 <= p[1] < w}
                    if visible & (support | obstacle):
                        continue
                    choices.append(dict(strand_id=j, obstacle_id=k, support=support,
                                        strand=strand, obstacle=obstacle, row=row, tip=tip,
                                        side=side, slack=slack, local_obstacle=local_obstacle,
                                        path=path, visible=visible, potential=potential, length=len(strand)))
        assert choices, "every rail needs an attached cord and its obstacle"
        candidates.append(choices)
    assert len(set(modes)) == 1, "a starting module closes all its inlets"
    assignments = []

    def search(i, used_s, used_o, chosen):
        if len(assignments) >= 2:
            return
        if i == len(rails):
            assignments.append(chosen)
            return
        for a in candidates[i]:
            if a["strand_id"] not in used_s and a["obstacle_id"] not in used_o:
                search(i + 1, used_s | {a["strand_id"]}, used_o | {a["obstacle_id"]}, chosen + [a])

    search(0, set(), set(), [])
    assert len(assignments) == 1, "local lead assignment must be unique"
    leads = assignments[0]
    footprints = [a["support"] | a["strand"] | a["obstacle"] | a["potential"] for a in leads]
    if len(footprints) == 2:
        expanded = {(r + dr, c + dc) for r, c in footprints[0] for dr, dc in _D8 + ((0, 0),)}
        assert not expanded & footprints[1], "leads must remain separated"
    out = panel.copy()
    for a in leads:
        for p in a["strand"]:
            out[p] = bg
    for a in leads:
        for p in a["visible"]:
            assert out[p] == bg
            out[p] = cord
    exits = tuple(map(int, np.flatnonzero(out[:, -1] != bg)))
    assert len(exits) == sum(a["length"] >= w - a["tip"] for a in leads)
    assert np.all(out[0] == bg) and np.all(out[-1] == bg)
    return dict(inlet=tuple(sorted(inlets)), outlet=exits, leads=leads, tightened=out)


def _orders(records):
    starts = [i for i, p in enumerate(records) if not p["inlet"]]
    ends = [i for i, p in enumerate(records) if not p["outlet"]]
    if len(starts) != 1 or len(ends) != 1 or starts == ends:
        return []
    answers = []

    def visit(order, unused):
        if len(answers) >= 2:
            return
        if not unused:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        port = records[order[-1]]["outlet"]
        if not port:
            return
        for i in unused:
            if records[i]["inlet"] == port:
                visit(order + [i], tuple(j for j in unused if j != i))

    visit([starts[0]], tuple(i for i in range(len(records)) if i != starts[0]))
    return answers


def _decode(inp):
    parsed = _partition(inp)
    possibilities = []
    for cord, obstacle in (parsed["other"], parsed["other"][::-1]):
        try:
            records = [_decode_panel(p, parsed["bg"], parsed["rail"], cord, obstacle)
                       for p in parsed["panels"]]
            orders = _orders(records)
            if len(orders) == 1:
                possibilities.append((cord, obstacle, records, orders[0]))
        except AssertionError:
            continue
    assert len(possibilities) == 1, "roles and complete chain must have one solution"
    cord, obstacle, records, order = possibilities[0]
    parsed.update(cord=cord, obstacle=obstacle, records=records, order=order)
    return parsed


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[i]
    return grid


def _answer(parsed):
    panels = [p["tightened"] for p in parsed["records"]]
    out = _layout(panels, parsed["order"], (1, len(panels)), parsed["divider"])
    return np.rot90(out, parsed["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive palette roles, taut exits, and panel order from this input alone."""
    return _answer(_decode(_grid(input_grid)))


def _new_obstacle(rng, available, approach, irregular=False):
    # Sample actual silhouettes, not bounding boxes. Convex exposed envelopes
    # are checked after placement; rear/internal geometry is preserved.
    if irregular:
        assert available >= 2
        span = int(rng.integers(2, min(3, available) + 1))
        raw = {(u, u) for u in range(span)}
    else:
        span = int(rng.integers(1, min(3, available) + 1))
        thickness = int(rng.integers(1, 4))
        kind = int(rng.integers(6))
        if kind == 0:
            raw = {(0, 0)}
        elif kind == 1:
            raw = {(u, 0) for u in range(span)}
        elif kind == 2:
            raw = {(0, v) for v in range(thickness)}
        elif kind == 3:
            raw = {(u, v) for u in range(span) for v in range(thickness)}
        elif kind == 4:
            raw = {(u, min(u, span - 1 - u)) for u in range(span)}
        else:
            raw = {(u, v) for u in range(span) for v in range(u + 1)}
    span = max(u for u, v in raw) + 1
    near = int(rng.integers(1, available - span + 2))
    shift = -max(v for u, v in raw) if approach == "parallel" else near - max(v - u for u, v in raw)
    obstacle = {(u + near, v + shift) for u, v in raw}
    _route(obstacle, 2)
    return obstacle


def _slack(rng, length, tip, span, diagonal_witness=False):
    cells = {(0, 1)}
    allowed = {(u, v) for u in range(1 - tip, 1) for v in range(1, span + 1)}
    if diagonal_witness:
        # Only a diagonal link joins the attachment to the rest of the mass.
        allowed = {(u, v) for u in range(1 - tip, 0) for v in range(2, span + 1)} | cells
        assert length <= len(allowed)
    style = int(rng.integers(4))
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in _D8}
                          & allowed - cells)
        assert frontier
        scores = []
        for u, v in frontier:
            neighbors = sum((u + du, v + dv) in cells for du, dv in _D8)
            scores.append((neighbors + 1) ** 2 if style == 0 else
                          1 / (neighbors + 1) ** 2 if style == 1 else
                          (v + 1) if style == 2 else (1 - u))
        probs = np.array(scores, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=probs / probs.sum()))])
    return cells


def _new_lead(rng, h, w, row, start, continues, approach=None, irregular=False, witness=False, clip_witness=False):
    tips = [x for x in range(2, min(4, w - 3) + 1)
            if w - x + int(clip_witness) <= 4 * x and (not irregular or w - 2 - x >= 2)
            and (not witness or w - x <= 1 + (x - 1) * 3)]
    assert tips
    tip = _pick(rng, tips)
    side = _pick(rng, (-1, 1))
    approach = approach or _pick(rng, ("parallel", "diagonal"))
    obstacle = _new_obstacle(rng, w - 2 - tip, approach, irregular)
    span = 4 if witness or clip_witness else int(rng.integers(2, 5))
    need = w - tip
    capacity = 1 + (tip - 1) * (span - 1) if witness else tip * span
    assert capacity >= (need if continues else 2)
    if continues:
        if clip_witness:
            length = int(rng.integers(need + 1, min(capacity, need + 4) + 1))
        else:
            length = need if witness or rng.random() < .5 else int(rng.integers(need, min(capacity, need + 4) + 1))
    else:
        length = int(rng.integers(2, min(need - 1, capacity) + 1))
    strand = _slack(rng, length, tip, span, witness)
    lo = int(rng.integers(1, tip)) if start else 0
    support = {(u, 0) for u in range(lo - tip, 1)}
    path = _sweep_route(obstacle, length)
    assert path == _route(obstacle, length)
    a = {name: {_global(p, row, tip, side) for p in value}
         for name, value in (("support", support), ("strand", strand), ("obstacle", obstacle))}
    a["path"] = [_global(p, row, tip, side) for p in path]
    potential = [_global(p, row, tip, side) for p in _sweep_route(obstacle, max(length, need))]
    occupied = a["support"] | a["strand"] | a["obstacle"]
    assert all(1 <= r < h - 1 and 0 <= c < w - 1 for r, c in occupied)
    assert all(1 <= r < h - 1 for r, c in potential if c < w)
    a["visible"] = {p for p in a["path"] if p[1] < w}
    a["potential"] = {p for p in potential if p[1] < w}
    assert not a["visible"] & (a["support"] | a["obstacle"])
    a["exit"] = next((r for r, c in a["path"] if c == w - 1), None)
    return a


def _new_module(rng, h, w, rows, start, continuing, palette, lesson=False, parallel=False):
    bg, rail, cord, obstacle = palette
    for _ in range(100):
        leads, occupied = [], set()
        try:
            for j, row in enumerate(rows):
                # The first lesson exposes both tangents, an irregular actual
                # silhouette, and a full visible diagonally connected budget.
                mode = ("diagonal" if j == 0 else "parallel") if lesson else ("parallel" if parallel else None)
                a = _new_lead(rng, h, w, row, start, j in continuing,
                              mode, irregular=lesson and j == 0, witness=lesson and j == 0,
                              clip_witness=lesson and j == 1)
                footprint = a["support"] | a["strand"] | a["obstacle"] | a["potential"]
                assert not footprint & occupied
                occupied.update((r + dr, c + dc) for r, c in footprint for dr, dc in _D8 + ((0, 0),))
                leads.append(a)
            inp = np.full((h, w), bg, dtype=np.int8)
            for a in leads:
                for role, color in (("support", rail), ("strand", cord), ("obstacle", obstacle)):
                    for p in a[role]:
                        assert inp[p] == bg
                        inp[p] = color
            out = inp.copy()
            for a in leads:
                for p in a["strand"]:
                    out[p] = bg
            for a in leads:
                for p in a["visible"]:
                    assert out[p] == bg
                    out[p] = cord
            decoded = _decode_panel(inp, bg, rail, cord, obstacle)
            assert np.array_equal(out, decoded["tightened"])
            return inp, out, tuple(sorted(a["exit"] for a in leads if a["exit"] is not None))
        except AssertionError:
            continue
    raise AssertionError("module sampling exhausted")


def _make_pair(rng, n, bundle, lesson, rotation, bg, preserve_bundle=False):
    maximum_w = min(12, 31 // n - 1)
    for _ in range(180):
        w = int(rng.integers(6 if lesson else 5, maximum_w + 1))
        h = int(rng.integers(20, 29)) if bundle else int(rng.integers(9, 18))
        palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
        rail, cord, obstacle, divider = palette[:4]
        if bundle:
            rows = (int(rng.integers(4, h // 2 - 2)), int(rng.integers(h // 2 + 3, h - 4)))
        else:
            rows = (int(rng.integers(4, h - 4)),)
        drop = (int(rng.integers(1, n - 1)) if bundle and not preserve_bundle
                and (lesson or rng.random() < .7) and n > 2 else None)
        inputs, outputs = [], []
        try:
            for i in range(n):
                continuing = set(range(len(rows))) if i < n - 1 else set()
                if i == drop:
                    continuing.remove(int(rng.integers(len(rows))))
                inp, out, rows = _new_module(rng, h, w, rows, i == 0, continuing,
                                             (bg, rail, cord, obstacle), lesson=lesson and i == 0,
                                             parallel=not bundle and i == 0)
                inputs.append(inp)
                outputs.append(out)
            records = [_decode_panel(p, bg, rail, cord, obstacle) for p in inputs]
            assert _orders(records) == [tuple(range(n))]
            layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
                       and nr * (h + 1) - 1 <= 30 and n // nr * (w + 1) - 1 <= 30]
            layout = _pick(rng, layouts)
            order = list(map(int, rng.permutation(n)))
            if order == list(range(n)):
                order = order[1:] + order[:1]
            inp = np.rot90(_layout(inputs, order, layout, divider), rotation).copy()
            out = np.rot90(_layout(outputs, list(range(n)), (1, n), divider), rotation).copy()
            assert np.array_equal(solve(inp), out)
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    return _fallback_pair(rng, n, bundle, lesson, rotation, bg, preserve_bundle)


def _fallback_pair(rng, n, bundle, lesson, rotation, bg, preserve_bundle=False):
    """Constructive safety net with the same budget, contour and matching rule."""
    w, h = (6, 24) if lesson else (5, 24 if bundle else 14)
    rail, cord, obstacle, divider = list(map(int, rng.permutation([c for c in range(10) if c != bg])))[:4]
    rows = [4, 15] if bundle else [4]
    panels, tightened = [], []
    for index in range(n):
        inp = np.full((h, w), bg, dtype=np.int8)
        out = inp.copy()
        next_rows = []
        for j, row in enumerate(rows):
            continuing = index < n - 1 and not (bundle and not preserve_bundle and index == 1 and j == 1)
            o = {(1, 1), (2, 2)} if lesson and index == 0 and j == 0 else (
                {(1, 0)} if index == 0 and (not bundle or lesson and j == 1) else {(1, 1)})
            need = w - 2
            length = (need + 1 if lesson and index == 0 and j == 1 else need) if continuing else 2
            if lesson and index == 0 and j == 0:
                slack = {(0, 1), (-1, 2), (-1, 3), (-1, 4)}
            else:
                slack = {(0, 1)} | {(-1, 1 + k) for k in range(length - 1)}
                if length == 5:
                    slack = {(0, 1), (-1, 1), (-1, 2), (0, 2), (-1, 3)}
            support = {(row, c) for c in range(1 if index == 0 else 0, 3)}
            obs = {_global(p, row, 2, 1) for p in o}
            mass = {_global(p, row, 2, 1) for p in slack}
            path = [_global(p, row, 2, 1) for p in _sweep_route(o, length)]
            for cells, color in ((support, rail), (obs, obstacle), (mass, cord)):
                for p in cells:
                    assert inp[p] == bg
                    inp[p] = color
            for cells, color in ((support, rail), (obs, obstacle)):
                for p in cells:
                    out[p] = color
            for p in path:
                if p[1] < w:
                    assert out[p] == bg
                    out[p] = cord
            if continuing:
                next_rows.append(path[need - 1][0])
        panels.append(inp)
        tightened.append(out)
        rows = next_rows
    order = list(reversed(range(n)))
    inp = np.rot90(_layout(panels, order, (1, n), divider), rotation).copy()
    out = np.rot90(_layout(tightened, list(range(n)), (1, n), divider), rotation).copy()
    assert np.array_equal(solve(inp), out)
    return {"input": inp, "output": out}


def _fresh_palette(pair, seen, bg):
    """At most 14 inputs exist; 9P4 palette assignments guarantee a fresh one."""
    foreground = sorted(set(map(int, np.unique(pair["input"]))) - {bg})
    for assignment in permutations([c for c in range(10) if c != bg], 4):
        lookup = np.arange(10, dtype=np.int8)
        lookup[foreground] = assignment
        fresh = {k: lookup[v] for k, v in pair.items()}
        if _key(fresh["input"]) not in seen:
            return fresh
    raise AssertionError("finite palette space exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    direction = int(rng.integers(4))
    directions = (direction,) if rng.random() < .3 else (direction, (direction + 1) % 4)
    ceiling = int(rng.integers(3, 5))
    persistence_shift = bool(rng.integers(2))
    puzzle, seen, train_ns = {"train": [], "test": []}, set(), []
    # Witness indices are randomized after construction, so index is not a rule.
    for i in range(nt + nq):
        training = i < nt
        preserve_bundle = False
        if i == 0:
            n, bundle, lesson = 3, True, True
        elif i == 1:
            # Two intermediate panels have identical port cardinalities.
            # Their positions, rather than terminal/count labels, fix the order.
            n, bundle, lesson = 4, False, False
        elif training:
            n = int(rng.integers(2, ceiling + 1))
            bundle, lesson = bool(rng.integers(2)) and not persistence_shift, False
        else:
            if persistence_shift:
                n, bundle, lesson, preserve_bundle = int(rng.integers(4, 6)), True, False, True
            else:
                n, bundle, lesson = 5, bool(rng.integers(2)), False
        rotation = directions[i % len(directions)] if i < 2 else _pick(rng, directions)
        for _ in range(12):
            pair = _make_pair(rng, n, bundle, lesson, rotation, backgrounds[i % 10], preserve_bundle)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if training else "test"].append(pair)
                if training:
                    train_ns.append(n)
                break
        else:
            pair = _fresh_palette(pair, seen, backgrounds[i % 10])
            seen.add(_key(pair["input"]))
            puzzle["train" if training else "test"].append(pair)
            if training:
                train_ns.append(n)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _in_place(parsed):
    """Tighten each panel at its old position: the counterfactual omitting A."""
    return _layout([np.rot90(p["tightened"], parsed["rotation"]) for p in parsed["records"]],
                   list(range(len(parsed["records"]))), parsed["layout"], parsed["divider"])


def _measure(parsed):
    records = parsed["records"]
    leads = [a for p in records for a in p["leads"]]
    cardinality_only = [dict(inlet=(0,) * len(p["inlet"]), outlet=(0,) * len(p["outlet"])) for p in records]
    return dict(n=len(records), rotation=parsed["rotation"],
                shape=parsed["panels"][0].shape, layout=parsed["layout"],
                bg=parsed["bg"], rail=parsed["rail"], cord=parsed["cord"], obstacle=parsed["obstacle"],
                order=parsed["order"], reorders=not np.array_equal(_in_place(parsed), _answer(parsed)),
                position_needed=len(_orders(cardinality_only)) > 1,
                bundle=any(len(p["inlet"]) == 2 for p in records),
                drop=any(p["inlet"] and 0 < len(p["outlet"]) < len(p["inlet"]) for p in records),
                parallel=any(max(v for u, v in a["local_obstacle"]) == 0 for a in leads),
                diagonal=any(max(v for u, v in a["local_obstacle"]) > 0 for a in leads),
                diagonal_slack=any(len(_components(a["strand"], False)) > 1 and
                                   len(a["visible"]) == a["length"] for a in leads),
                irregular=any(len(a["local_obstacle"]) !=
                              (max(u for u, v in a["local_obstacle"]) - min(u for u, v in a["local_obstacle"]) + 1) *
                              (max(v for u, v in a["local_obstacle"]) - min(v for u, v in a["local_obstacle"]) + 1)
                              for a in leads),
                clipped=any(len(a["visible"]) < a["length"] for a in leads),
                lengths=tuple(a["length"] for a in leads),
                signatures=tuple((p["inlet"], p["outlet"]) for p in records))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            parsed = _decode(inp)
            expected = _answer(parsed)
            assert np.array_equal(out, expected), "incorrect tightened chain"
            assert not np.array_equal(inp, out)
            h, w = parsed["panels"][0].shape
            assert np.rot90(out, -parsed["rotation"]).shape == (h, len(parsed["panels"]) * (w + 1) - 1)
            # Check preserved material panel by panel, not just solve agreement.
            for p, record in zip(parsed["panels"], parsed["records"]):
                q = record["tightened"]
                for color in (parsed["rail"], parsed["obstacle"]):
                    assert np.array_equal(p == color, q == color)
                for a in record["leads"]:
                    assert len(a["visible"]) == min(a["length"], w - a["tip"])
                    assert (a["row"] + a["side"], a["tip"]) in a["visible"]
                    assert _sweep_route(a["local_obstacle"], a["length"]) == _route(a["local_obstacle"], a["length"])
                assert np.count_nonzero(q == parsed["cord"]) == sum(len(a["visible"]) for a in record["leads"])
            # Geometry and budgets, rather than a color ordering, determine links.
            for i, j in zip(parsed["order"], parsed["order"][1:]):
                assert parsed["records"][i]["outlet"] == parsed["records"][j]["inlet"]
            metrics[split].append(_measure(parsed))
    training = metrics["train"]
    assert any(p["drop"] for p in training), "teach cord exhaustion changing a signature"
    assert any(not p["bundle"] for p in training), "teach single-lead modules too"
    assert any(p["parallel"] and p["diagonal"] for p in training)
    assert any(p["diagonal_slack"] for p in training), "teach eight-connected unary counting"
    assert any(p["clipped"] for p in training), "teach clipping before test use"
    assert any(p["irregular"] for p in training), "teach actual obstacle silhouettes"
    assert any(p["position_needed"] for p in training), "port counts alone must not fix all demonstrations"
    assert all(p["reorders"] for p in training)
    assert len({p["bg"] for p in training}) >= 2
    count_extension = min(p["n"] for p in metrics["test"]) > max(p["n"] for p in training)
    persistence = (max(p["n"] for p in training if p["bundle"]) == 3 and
                   all(p["bundle"] and not p["drop"] and p["n"] >= 4 for p in metrics["test"]))
    assert count_extension or persistence, "missing structural train/test shift"


def self_test():
    """Bounded route regressions, counterfactuals, and structural 42-seed audit."""
    cases = [({(2, 0)}, 6, [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]),
             ({(1, 1), (2, 2)}, 6, [(0, 1), (1, 2), (2, 3), (3, 2), (4, 1), (5, 0)]),
             ({(u, v) for u in (2, 3) for v in (0, 1, 2)}, 6,
              [(0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1)])]
    for obstacle, length, expected in cases:
        assert _route(obstacle, length) == expected == _sweep_route(obstacle, length)
    metrics, puzzles, shapes, silhouettes = [], set(), set(), set()
    roles = {name: set() for name in ("bg", "rail", "cord", "obstacle")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(pair["input"]) for split in puzzle for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"].copy()
                parsed = _decode(inp)
                prediction = solve(inp)
                assert np.array_equal(prediction, pair["output"]) and np.array_equal(inp, pair["input"])
                m = _measure(parsed)
                metrics.append(m)
                shapes.add(pair["input"].shape)
                for name in roles:
                    roles[name].add(m[name])
                # Small all-permutation oracle for chain uniqueness only.
                records = parsed["records"]
                orders = [p for p in permutations(range(len(records))) if not records[p[0]]["inlet"]
                          and not records[p[-1]]["outlet"]
                          and all(records[a]["outlet"] and records[a]["outlet"] == records[b]["inlet"]
                                  for a, b in zip(p, p[1:]))]
                assert orders == [parsed["order"]]
                for p in records:
                    for a in p["leads"]:
                        o = a["local_obstacle"]
                        lo_u, lo_v = min(u for u, v in o), min(v for u, v in o)
                        silhouettes.add(tuple(sorted((u - lo_u, v - lo_v) for u, v in o)))
                        assert len(_components(a["slack"])) == 1
    assert len(puzzles) == 42 and len(shapes) >= 25 and len(silhouettes) >= 8
    assert {m["n"] for m in metrics} == {2, 3, 4, 5}
    assert {m["rotation"] for m in metrics} == {0, 1, 2, 3}
    assert len({m["shape"] for m in metrics}) >= 20
    assert all(len(colors) == 10 for colors in roles.values())
    assert any(m["clipped"] for m in metrics)
    assert any(m["layout"] == (2, 2) for m in metrics)
    assert any(m["drop"] and m["clipped"] for m in metrics)
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    # Palette and orientation are completely input-local conventions.
    for pair in a["train"]:
        assert np.array_equal(solve(9 - pair["input"]), 9 - pair["output"])
        for turn in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
    # Removing A: tighten panels but retain the input order/layout.
    witness = next(p for p in a["train"] if _measure(_decode(p["input"]))["drop"])
    parsed = _decode(witness["input"])
    naive = _in_place(parsed)
    assert not np.array_equal(naive, witness["output"])
    # Removing B's length: a dropped internal cord would create an extra exit.
    record = next(p for p in parsed["records"] if p["inlet"] and 0 < len(p["outlet"]) < len(p["inlet"]))
    short = next(a for a in record["leads"] if a["length"] < parsed["panels"][0].shape[1] - a["tip"])
    need = parsed["panels"][0].shape[1] - short["tip"]
    extended = _route(short["local_obstacle"], need)
    assert len(extended) > short["length"] and extended[-1][0] + short["tip"] == parsed["panels"][0].shape[1] - 1
    # Removing B's obstacle: at least one learned cord bends off a parallel ray.
    assert any(any(v != 1 for u, v in _route(a["local_obstacle"], min(a["length"], parsed["panels"][0].shape[1] - a["tip"])))
               for p in parsed["records"] for a in p["leads"])
    # Equal-length slack rearrangements must leave the entire answer unchanged.
    rng = np.random.default_rng(123)
    changed = False
    for index, record in enumerate(parsed["records"]):
        for lead in record["leads"]:
            for _ in range(30):
                shape = _slack(rng, lead["length"], lead["tip"], 4)
                cells = {_global(p, lead["row"], lead["tip"], lead["side"]) for p in shape}
                p = parsed["panels"][index].copy()
                if cells == lead["strand"] or any(not 1 <= r < p.shape[0] - 1 or not 1 <= c < p.shape[1] - 1 for r, c in cells):
                    continue
                for point in lead["strand"]:
                    p[point] = parsed["bg"]
                if any(p[point] != parsed["bg"] for point in cells):
                    continue
                for point in cells:
                    p[point] = parsed["cord"]
                panels = [np.rot90(q, parsed["rotation"]) for q in parsed["panels"]]
                panels[index] = np.rot90(p, parsed["rotation"])
                changed_input = _layout(panels, list(range(len(panels))), parsed["layout"], parsed["divider"])
                try:
                    changed_answer = solve(changed_input)
                except AssertionError:
                    continue
                assert np.array_equal(changed_answer, witness["output"])
                changed = True
                break
            if changed:
                break
        if changed:
            break
    assert changed, "must exercise an equal-budget slack rearrangement"
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for counts in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(901826715, *counts))
    rng = np.random.default_rng(456)
    for n in (2, 5):
        for bundle in (False, True):
            for rotation in range(4):
                pair = _fallback_pair(rng, n, bundle, False, rotation, rotation, preserve_bundle=True)
                assert np.array_equal(solve(pair["input"]), pair["output"])
    fallback = {"train": [_fallback_pair(rng, 3, True, True, 0, 0),
                          _fallback_pair(rng, 4, False, False, 1, 1)],
                "test": [_fallback_pair(rng, 4, True, False, 0, 2, True)]}
    validate_generated_puzzle(fallback)
    # Negative semantic check: preserved obstacles cannot be recolored.
    damaged = generate_puzzle(826653882, 2, 1)
    p = damaged["train"][0]
    decoded = _decode(p["input"])
    cells = np.argwhere(p["output"] == decoded["obstacle"])
    p["output"][tuple(cells[0])] = decoded["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("validator accepted obstacle deletion")
