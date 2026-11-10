"""Pilot-selected docking. Only the standard library and NumPy are used.

Fixed roles: 0 blank, 8 temporary registration, 9 pilot. The common
foreground and rail identities are inferred from marker backing geometry.
All output calculations use foreground sets, never opaque rectangles.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _components(cells, diagonal=True):
    remaining = set(cells)
    parts = []
    directions = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
                  if (y or x) and (diagonal or not (y and x))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    pending.append(q)
        parts.append(part)
    return parts


def _cells(g, color):
    return {tuple(map(int, p)) for p in np.argwhere(g == color)}


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _crop(cells, fg):
    top, bottom, left, right = _bbox(cells)
    _require(bottom - top < 30 and right - left < 30, "assembled crop too large")
    g = np.zeros((bottom - top + 1, right - left + 1), dtype=np.int8)
    for r, c in cells:
        g[r - top, c - left] = fg
    return g


def _undo(g, rotation, reflected):
    if reflected:
        g = np.fliplr(g)
    return np.rot90(g, -rotation).copy()


def _canonical(g):
    """Find the keyed bar's tangent and backing normal from the input alone."""
    _require(np.count_nonzero(g == 9) == 1, "one pilot required")
    bars = _components(_cells(g, 8) | _cells(g, 9), diagonal=False)
    keyed = next((b for b in bars if any(g[p] == 9 for p in b)), None)
    _require(keyed is not None and 2 <= len(keyed) <= 5, "keyed bar length 2..5")
    length = len(keyed)
    _require(3 <= len(bars) <= 5, "one cargo bar and two to four ports")
    _require(all(len(b) == length and len({c for r, c in b}) == 1
                 and max(r for r, c in b) - min(r for r, c in b) + 1 == length
                 for b in bars), "parallel congruent straight bars required")
    start = min(keyed)
    _require(g[start] == 9, "pilot must be first bar endpoint")
    r, c = start
    _require(c > 0, "cargo must back its keyed bar")
    fg = int(g[r, c - 1])
    _require(1 <= fg <= 7, "foreground backing must be ordinary color")
    _require(all(g[y, x - 1] == fg for y, x in keyed), "full cargo backing required")
    ports = sorted(min(b) for b in bars if b is not keyed)
    for y, x in ports:
        _require(x + 1 < g.shape[1] and
                 all(g[y + i, x + 1] == fg for i in range(length)),
                 "ports must have opposite full backing")
    parts = _components(_cells(g, fg) | _cells(g, 8) | _cells(g, 9))
    _require(len(parts) == 2, "exactly two marker-bearing eight-connected fragments")
    cargo_part = next(b for b in parts if start in b)
    anchor_part = next(b for b in parts if start not in b)
    _require(all(p in anchor_part for p in ports), "all ports belong to the anchor")
    _require(keyed <= cargo_part, "cargo ownership")
    cargo = {p for p in cargo_part if g[p] == fg}
    anchor = {p for p in anchor_part if g[p] == fg}
    guides = set(map(int, np.unique(g))) - {0, fg, 8, 9}
    _require(len(guides) >= 2, "at least two guide identities required")
    return dict(grid=g, foreground=fg, start=start, bar=keyed, length=length,
                ports=ports, cargo=cargo, anchor=anchor, guides=guides)


def _parse(input_grid):
    inp = _grid(input_grid)
    choices = []
    for reflected in (False, True):
        for rotation in range(4):
            g = np.rot90(inp, rotation)
            if reflected:
                g = np.fliplr(g)
            try:
                p = _canonical(g)
            except ValueError:
                continue
            p.update(rotation=rotation, reflected=reflected)
            choices.append(p)
    _require(len(choices) == 1, "bar geometry must identify one coordinate frame")
    return choices[0]


def _plus(p, d):
    return (p[0] + d[0], p[1] + d[1])


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _flanks(parsed, q, d):
    g = parsed["grid"]
    normal = (-d[1], d[0])
    a, b = _plus(q, normal), _plus(q, (-normal[0], -normal[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return None
    ca, cb = int(g[a]), int(g[b])
    if ca not in parsed["guides"] or cb not in parsed["guides"]:
        return None
    return (a, b, ca, cb)


def _trace(parsed):
    """Directional motion, with no flood fill and no arbitrary turn priority."""
    g, start = parsed["grid"], parsed["start"]
    # The first visible pair ahead of the keyed pilot must activate it.
    activated = False
    for col in range(start[1] + 1, g.shape[1]):
        q = (start[0], col)
        if g[q] != 0:
            break
        flanks = _flanks(parsed, q, (0, 1))
        if flanks is not None:
            activated = flanks[2] == flanks[3]
            break
    _require(activated, "pilot needs a matching entrance")
    p, direction = start, (0, 1)
    path, seen, turns, forks = [p], set(), [], []
    while True:
        state = (p, direction)
        _require(state not in seen, "route cycle")
        seen.add(state)
        q = _plus(p, direction)
        _require(_inside(g, q), "route must terminate at an internal port")
        if g[q] == 0:
            p = q
            path.append(p)
            continue
        geometry, matching = [], []
        for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
            candidate = _plus(p, d)
            if _inside(g, candidate) and g[candidate] == 0:
                flanks = _flanks(parsed, candidate, d)
                if flanks is not None:
                    record = dict(direction=d, next=candidate, flanks=flanks)
                    geometry.append(record)
                    if flanks[2] == flanks[3]:
                        matching.append(record)
        _require(len(matching) <= 1, "ambiguous matching turns")
        if len(geometry) == 2:
            forks.append(dict(position=p, incoming=direction, options=geometry))
        if not matching:
            _require(direction == (0, 1) and q in parsed["ports"],
                     "terminal must be the corresponding endpoint of a port")
            port = q
            break
        d = matching[0]["direction"]
        turns.append((p, direction, d))
        direction = d
    _require(len(path) == len(set(path)), "pilot cannot revisit cells")
    # Generation has one fork; the reader can navigate multiple unique forks.
    return dict(port=port, terminal=p, path=path, turns=turns, forks=forks)


def _docked_sets(parsed, port, seam=1):
    delta = (port[0] - parsed["start"][0],
             port[1] + seam - parsed["start"][1])
    return parsed["anchor"], _shift(parsed["cargo"], delta), delta


def _assembled(parsed, port, mode="union"):
    seam = {"terminal": -1, "coincident": 0}.get(mode, 1)
    a, m, delta = _docked_sets(parsed, port, seam)
    if mode == "xor":
        cells = a ^ m
    elif mode == "cargo_opaque":
        t, b, l, r = _bbox(m | _shift(parsed["bar"], delta))
        cells = m | {p for p in a if not (t <= p[0] <= b and l <= p[1] <= r)}
    elif mode == "anchor_opaque":
        t, b, l, r = _bbox(a | _cells(parsed["grid"], 8) - parsed["bar"])
        cells = a | {p for p in m if not (t <= p[0] <= b and l <= p[1] <= r)}
    elif mode == "erase_marks":
        cells = (a | m) - (_cells(parsed["grid"], 8) - parsed["bar"]
                            | _shift(parsed["bar"], delta))
    elif mode == "paint_marks":
        cells = a | m | (_cells(parsed["grid"], 8) - parsed["bar"]) | _shift(parsed["bar"], delta)
    elif mode == "no_registration":
        cells = a | parsed["cargo"]
    else:
        cells = a | m
    return _crop(cells, parsed["foreground"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The complete rule is recovered from this input; no examples or RNG."""
    parsed = _parse(input_grid)
    route = _trace(parsed)
    out = _assembled(parsed, route["port"])
    return _undo(out, parsed["rotation"], parsed["reflected"])


def _switch_fork(parsed, route):
    """Counterfactual: swap equality at the two exits, preserving all geometry."""
    _require(len(route["forks"]) == 1, "generated network has one selection fork")
    fork = route["forks"][0]
    _require(fork["incoming"] == (0, 1), "generated fork faces east")
    g = parsed["grid"].copy()
    for option in fork["options"]:
        y, x = option["next"]
        west, east = (y, x - 1), (y, x + 1)
        if g[west] == g[east]:
            g[west] = next(c for c in sorted(parsed["guides"]) if c != int(g[east]))
        else:
            g[west] = g[east]
    return g


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        _require((d[0] == 0) != (d[1] == 0), "nonzero orthogonal segments")
        while p != target:
            p = _plus(p, d)
            path.append(p)
    _require(len(path) == len(set(path)), "sampled path revisits itself")
    cells = set(path)
    for i, p in enumerate(path):
        degree = sum(_plus(p, d) in cells for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        _require(degree == (1 if i in (0, len(path) - 1) else 2), "sampled path self-contact")
    return path


def _cargo(rng, length, style, width, witness):
    above, below = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    top, bottom = -above, length - 1 + below
    cells = set()
    for y in range(top, bottom + 1):
        for x in range(-width, 0):
            if style == "ring":
                yes = y in (top, bottom) or x in (-width, -1)
            elif style == "open":
                yes = y in (top, bottom) or x == -1
            elif style == "step":
                left = -width + min(2, max(0, y - top))
                yes = x >= left and (x in (left, -1) or y in (top, bottom))
            elif style == "branch":
                yes = x == -1 or (y in (top, bottom, 1) and x >= -width + int(y == 1))
            else:
                yes = not (x < -2 and top < y < 0)
            if yes:
                cells.add((y, x))
    cells.update((i, -1) for i in range(length))
    if witness or rng.random() < .65:
        reach = 6 if witness else int(rng.integers(1, 7))
        depth = 3 if witness else int(rng.integers(1, 4))
        cells.update((length, x) for x in range(-1, reach + 1))
        cells.update((length + i, reach) for i in range(depth + 1))
        if witness or rng.random() < .5:
            cells.update((length + depth, x) for x in range(-1, reach + 1))
            cells.update((length + i, -1) for i in range(depth + 1))
    _require(len(_components(cells)) == 1, "cargo stroke connectivity")
    return cells


def _anchor(rng, ports, length, style, witness):
    spine = max(c for r, c in ports) + (4 if witness else int(rng.integers(3, 5)))
    top = min(r for r, c in ports) - int(rng.integers(0, 2))
    bottom = max(r for r, c in ports) + length + 3
    thickness = 1 if witness else int(rng.integers(1, 3))
    cells = {(r, spine + dx) for r in range(top, bottom + 1) for dx in range(thickness)}
    for y, x in ports:
        cells.update((y + i, x + 1) for i in range(length))
        mids = [y + int(rng.integers(length))]
        if rng.random() < .45:
            mids.append(y + length - 1)
        for r in mids:
            cells.update((r, c) for c in range(x + 1, spine + 1))
    # A frame must stay behind every port. Using the leftmost port's backing
    # column would obstruct a farther-right port when port depths differ.
    left = max(c for r, c in ports) + 1
    if style == "frame":
        cells.update((r, left) for r in range(top, bottom + 1))
        cells.update((r, c) for r in (top, bottom) for c in range(left, spine + 1))
    elif style == "mass":
        cells.update((r, c) for r in range(top, bottom + 1) for c in range(spine - 2, spine + 1))
        cells.difference_update((r, spine - 2) for r in range(top + 2, min(top + 5, bottom)))
    elif style == "step":
        cells.update((r, spine - min(2, (r - top) % 4)) for r in range(top, bottom + 1))
    else:
        for r in range(top, bottom + 1, int(rng.integers(3, 6))):
            cells.update((r, c) for c in range(spine - int(rng.integers(1, 4)), spine + 1))
    _require(len(_components(cells)) == 1, "anchor connectivity")
    return cells


def _branch(rng, start, fork_col, target, steps, upper, height, length):
    r, c = start
    low, high = (1, r - 3) if upper else (r + 3, height - length - 5)
    goal, port_col = target
    rows = [goal]
    if steps > 1:
        possibilities = [y for y in range(low, high + 1) if abs(y - goal) >= 3]
        _require(bool(possibilities), "dogleg rows do not fit")
        rows.insert(0, int(rng.choice(possibilities)))
    if steps > 2:
        possibilities = [y for y in range(low, high + 1) if abs(y - rows[0]) >= 3]
        _require(bool(possibilities), "second dogleg rows do not fit")
        rows.insert(0, int(rng.choice(possibilities)))
    n_intermediate = steps - 1
    available = list(range(fork_col + 3, port_col - 3))
    if n_intermediate:
        columns = []
        for _ in range(32):
            _require(len(available) >= n_intermediate, "dogleg columns do not fit")
            columns = sorted(map(int, rng.choice(available, n_intermediate, replace=False)))
            if all(b - a >= 3 for a, b in zip([fork_col] + columns, columns + [port_col - 1])):
                break
        else:
            raise ValueError("dogleg columns exhausted")
    else:
        columns = []
    points = [start, (r, fork_col)]
    x = fork_col
    for i, y in enumerate(rows):
        points.append((y, x))
        end_col = columns[i] if i < n_intermediate else port_col - 1
        points.append((y, end_col))
        x = end_col
    return _polyline(points)


_ASSEMBLY_ALTERNATIVES = ("terminal", "coincident", "xor", "cargo_opaque",
                          "anchor_opaque", "erase_marks", "paint_marks", "no_registration")


def _make_pair(rng, profile):
    length, fg, palette = profile["length"], profile["fg"], profile["guides"]
    h = int(rng.integers(max(22, 2 * length + 16), 31))
    w = int(rng.integers(*profile["width_range"]))
    core_width = int(rng.integers(*profile["cargo_width"]))
    pilot_col = core_width + int(rng.integers(0, 3))
    fork_col = pilot_col + int(rng.integers(6, 9))
    port_col = w - 6
    _require(port_col - fork_col >= 5, "rail workspace too short")
    row = int(rng.integers(8, h - length - 7))
    upper_rows = list(range(1, row - 2))
    lower_rows = list(range(row + 3, h - length - 4))
    _require(upper_rows and lower_rows, "two branch spaces needed")
    upper = (int(rng.choice(upper_rows)), port_col)
    lower = (int(rng.choice(lower_rows)), port_col)
    if profile["offset_ports"]:
        upper_offset = int(rng.integers(0, 3))
        lower_offset = int(rng.choice([i for i in range(3) if i != upper_offset])) if profile["distinct_offsets"] else int(rng.integers(0, 3))
        upper = (upper[0], upper[1] - upper_offset)
        lower = (lower[0], lower[1] - lower_offset)
    targets = [upper, lower]
    selected = profile["selected"]
    distances = [abs(r - row) + abs(c - pilot_col) for r, c in targets]
    if profile["distance"] == "far":
        _require(distances[selected] > distances[1 - selected], "far-port teaching case")
    elif profile["distance"] == "near":
        _require(distances[selected] < distances[1 - selected], "near-port teaching case")
    ports = list(targets)
    options = list(range(1, h - length - 4))
    for _ in range(profile["ports"] - 2):
        feasible = [y for y in options if all(abs(y - p[0]) >= length + 2 for p in ports)]
        _require(bool(feasible), "additional ports do not fit")
        y = int(rng.choice(feasible))
        x = port_col - (int(rng.integers(0, 3)) if profile["offset_ports"] else 0)
        ports.append((y, x))
    _require(all(abs(a[0] - b[0]) >= length + 2 for i, a in enumerate(ports) for b in ports[i + 1:]),
             "port bars need visible separation")
    start = (row, pilot_col)
    routes = [_branch(rng, start, fork_col, target, steps, i == 0, h, length)
              for i, (target, steps) in enumerate(zip(targets, profile["steps"]))]
    all_paths = set(routes[0]) | set(routes[1])
    cargo_local = _cargo(rng, length, profile["cargo_style"], core_width, profile["witness"])
    cargo = _shift(cargo_local, start)
    anchor = _anchor(rng, ports, length, profile["anchor_style"], profile["witness"])
    keyed = {(row + i, pilot_col) for i in range(length)}
    port_cells = {(r + i, c) for r, c in ports for i in range(length)}
    _require(not (cargo & anchor or (cargo | anchor) & (keyed | port_cells)), "staging collision")
    g = np.zeros((h, w), dtype=np.int8)
    for p in cargo | anchor:
        _require(_inside(g, p), "foreground staging outside canvas")
        g[p] = fg
    for p in keyed | port_cells:
        g[p] = 8
    g[start] = 9
    _require(all(g[p] == 0 for p in all_paths - {start}), "foreground blocks a planned ray")
    base = palette[profile["base_index"] % len(palette)]
    other = next(c for c in palette if c != base)
    approach = int(rng.integers(1, 3))
    walls = set()
    for p in all_paths:
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = _plus(p, d)
            if (_inside(g, q) and q not in all_paths and q[1] >= pilot_col + approach
                    and q[1] <= port_col - 2 and g[q] == 0):
                walls.add(q)
    for p in walls:
        g[p] = base
    sign = 1 if selected == 0 else -1
    # Mismatch is local to the rejected exit; neither literal color means inactive.
    for distance in range(1, int(rng.integers(2, 4))):
        p = (row + sign * distance, fork_col - 1)
        _require(p in walls, "rejected fork rail must be visible")
        g[p] = other
    gap = profile["gap"]
    if gap:
        begin = pilot_col + approach + 2
        gap = min(gap, fork_col - 1 - begin)
        _require(gap >= 1, "gap needs support before and after")
        for x in range(begin, begin + gap):
            g[row - 1, x] = g[row + 1, x] = 0
    # Extra colors may appear on paired straight evidence, not as commands.
    if len(palette) > 2:
        route = routes[selected]
        remaining_colors = list(palette[2:])
        for i in range(3, len(route) - 3):
            before, p, after = route[i - 1:i + 2]
            d = (after[0] - p[0], after[1] - p[1])
            if d != (p[0] - before[0], p[1] - before[1]) or p[1] <= fork_col + 2:
                continue
            normal = (-d[1], d[0])
            a, b = _plus(p, normal), _plus(p, (-normal[0], -normal[1]))
            if a in walls and b in walls and g[a] == g[b] == base:
                g[a] = g[b] = remaining_colors.pop(0)
                if not remaining_colors:
                    break
    p = _canonical(g)
    trace = _trace(p)
    _require(trace["port"] == targets[selected] and trace["path"] == routes[selected],
             "visible ray must agree with the planned branch")
    _require(len(trace["forks"]) == 1, "one explicit color-selection fork")
    swapped = _canonical(_switch_fork(p, trace))
    alternate_trace = _trace(swapped)
    _require(alternate_trace["port"] == targets[1 - selected], "mismatch exit must be a real alternative")
    out = _crop(anchor | _shift(cargo, (targets[selected][0] - row,
                                      targets[selected][1] + 1 - pilot_col)), fg)
    _require(np.array_equal(out, _assembled(p, trace["port"])), "construction/reader union disagreement")
    _require(not np.array_equal(out, _assembled(swapped, alternate_trace["port"])), "B must change the answer")
    if profile["witness"]:
        _require(all(not np.array_equal(out, _assembled(p, trace["port"], mode))
                     for mode in _ASSEMBLY_ALTERNATIVES), "A teaching witness incomplete")
    inp = _undo(g, profile["rotation"], profile["reflected"])
    answer = _undo(out, profile["rotation"], profile["reflected"])
    return {"input": inp, "output": answer}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return value


def _signature(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    fg = int(rng.integers(1, 8))
    palette = list(map(int, rng.choice([c for c in range(1, 8) if c != fg],
                                     int(rng.integers(2, 5)), replace=False)))
    changing_palette = bool(rng.integers(2))
    mixed_orientation = bool(rng.integers(2))
    rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
    first_choice = int(rng.integers(2))
    regime = str(rng.choice(["bar_interpolation", "more_ports", "longer_runs", "smaller_cargo", "offset_ports"]))
    used, puzzle = set(), {"train": [], "test": []}
    for split, count in (("train", n), ("test", m)):
        for index in range(count):
            test = split == "test"
            length = int(rng.integers(2, 5))
            if regime == "bar_interpolation":
                length = 3 if test else (2 if index % 2 == 0 else 4)
            ports = int(rng.integers(2, 4))
            if index < 2 and not test:
                ports = 2
            if regime == "more_ports":
                length = 2
                ports = int(rng.integers(3, 5)) if test else 2
            width_range = (25, 31)
            if regime == "longer_runs":
                width_range = (28, 31) if test else (24, 28)
            cargo_width = (3, 6)
            if regime == "smaller_cargo":
                cargo_width = (2, 4) if test else (4, 6)
            selected = first_choice if index == 0 and not test else 1 - first_choice if index == 1 and not test else int(rng.integers(2))
            steps = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
            if not test and index == 0:
                steps = [1, 1]
            elif not test and index == 1:
                steps = [2, 2]
            if changing_palette:
                pair_fg = int(rng.integers(1, 8))
                pair_palette = list(map(int, rng.choice([c for c in range(1, 8) if c != pair_fg],
                                                        len(palette), replace=False)))
            else:
                pair_fg, pair_palette = fg, palette
            profile = dict(length=length, fg=pair_fg, guides=pair_palette,
                           base_index=index % 2, ports=ports, selected=selected,
                           distance="far" if index == 0 and not test else "near" if index == 1 and not test else "any",
                           steps=steps, width_range=width_range, cargo_width=cargo_width,
                           cargo_style="ring" if index == 0 and not test else str(rng.choice(["ring", "open", "branch", "step", "notched"])),
                           anchor_style="branch" if index == 0 and not test else str(rng.choice(["branch", "frame", "mass", "step"])),
                           witness=index == 0 and not test,
                           gap=2 if index == 1 and not test else int(rng.integers(0, 3)),
                           offset_ports=test if regime == "offset_ports" else bool(rng.integers(2)),
                           distinct_offsets=test and regime == "offset_ports",
                           rotation=(rotation + index % 4) % 4 if mixed_orientation else rotation,
                           reflected=bool(rng.integers(2)) if mixed_orientation else reflected)
            # Retry structural sampling only; no stored templates or parent calls.
            for attempt in range(600):
                try:
                    pair = _make_pair(rng, profile)
                except ValueError:
                    continue
                signature = _signature(pair["input"])
                if signature in used:
                    continue
                if split == "train" and index == 1 and pair["input"].shape == puzzle["train"][0]["input"].shape:
                    continue
                used.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f"bounded docking construction exhausted: {split}/{index}, {profile}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    p = _parse(inp)
    route = _trace(p)
    expected = _undo(_assembled(p, route["port"]), p["rotation"], p["reflected"])
    _require(np.array_equal(expected, out), "incorrect docking output")
    a, m, delta = _docked_sets(p, route["port"])
    merged = a | m
    _require(int(np.count_nonzero(out == p["foreground"])) == len(a) + len(m) - len(a & m),
             "union foreground conservation")
    _require(set(map(int, np.unique(out))) <= {0, p["foreground"]}, "navigation marks must disappear")
    _require(all(np.any(edge == p["foreground"]) for edge in (out[0], out[-1], out[:, 0], out[:, -1])),
             "output must be tightly cropped")
    port_bar = {(route["port"][0] + i, route["port"][1]) for i in range(p["length"])}
    _require(port_bar <= m and _shift(p["bar"], delta) <= a,
             "each fragment's backing must survive over the other bar")
    _require(len(route["forks"]) == 1, "one choice fork in generated examples")
    switched = _canonical(_switch_fork(p, route))
    other = _trace(switched)
    _require(other["port"] != route["port"] and
             not np.array_equal(_assembled(p, route["port"]), _assembled(switched, other["port"])),
             "equality ablation must change the assembled answer")
    alternatives = {mode: not np.array_equal(_assembled(p, route["port"]), _assembled(p, route["port"], mode))
                    for mode in _ASSEMBLY_ALTERNATIVES}
    canonical_out = _assembled(p, route["port"])
    distances = [abs(r - p["start"][0]) + abs(c - p["start"][1]) for r, c in p["ports"]]
    chosen_distance = abs(route["port"][0] - p["start"][0]) + abs(route["port"][1] - p["start"][1])
    # Straight missing-flank evidence inside the route, excluding natural corners.
    gaps = 0
    for i in range(2, len(route["path"]) - 2):
        before, q, after = route["path"][i - 1:i + 2]
        d = (after[0] - q[0], after[1] - q[1])
        if d == (q[0] - before[0], q[1] - before[1]):
            if _flanks(p, q, d) is None and q[1] > p["start"][1] + 2 and q[1] < route["forks"][0]["position"][1] - 1:
                gaps += 1
    return dict(shape=inp.shape, output_shape=out.shape, length=p["length"], ports=len(p["ports"]),
                turns=len(route["turns"]), route_length=len(route["path"]), overlap=len(a & m),
                palette=(p["foreground"], tuple(sorted(p["guides"]))),
                orientation=(p["rotation"], p["reflected"]), selected_upper=route["port"][0] < p["start"][0],
                nearest=chosen_distance == min(distances), farthest=chosen_distance == max(distances),
                gaps=gaps, alternatives=alternatives, output_mask=_signature(canonical_out != 0))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            key = _signature(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(_pair_info(pair))
    train = records["train"]
    _require({r["selected_upper"] for r in train} == {False, True}, "both fork choices must be taught")
    _require(any(not r["nearest"] for r in train) and any(not r["farthest"] for r in train),
             "neither nearest nor farthest port is the rule")
    _require(any(r["gaps"] for r in train), "straight continuation across gaps must be taught")
    _require(any(r["turns"] >= 4 for r in train), "multi-elbow navigation must be taught")
    _require(len({r["shape"] for r in train}) >= 2, "demonstration dimensions must vary")
    _require(len({r["output_mask"] for r in train}) >= 2, "demonstration structures must vary")
    for mode in _ASSEMBLY_ALTERNATIVES:
        if any(r["alternatives"][mode] for r in records["test"]):
            _require(any(r["alternatives"][mode] for r in train), f"unexposed assembly alternative {mode}")


def self_test():
    """Bounded 42-seed survey, semantic mutations, D4 equivariance and counts."""
    records, puzzles, signatures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeated = generate_puzzle(seed)
        _require(all(np.array_equal(x[k], y[k]) for split in puzzle
                     for x, y in zip(puzzle[split], repeated[split]) for k in ("input", "output")),
                 "seed reproducibility")
        signatures.add(tuple(_signature(pair["input"]) for split in puzzle for pair in puzzle[split]))
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve")
                records.append(_pair_info(pair))
    _require(len(signatures) == 42, "distinct puzzles across 42 seeds")
    _require(len({r["shape"] for r in records}) >= 20, "canvas geometry diversity")
    _require(len({r["output_shape"] for r in records}) >= 25, "assembly aspect diversity")
    _require(len({r["output_mask"] for r in records}) >= 150, "actual shape diversity")
    _require({r["length"] for r in records} >= {2, 3, 4}, "bar length variation")
    _require({r["ports"] for r in records} == {2, 3, 4}, "port count variation")
    _require({r["turns"] for r in records} >= {2, 4, 6}, "route geometry variation")
    _require({r["palette"][0] for r in records} == set(range(1, 8)), "foreground roles")
    _require(len({r["orientation"] for r in records}) == 8, "global D4 variation")
    _require(any(r["overlap"] == 0 for r in records) and any(r["overlap"] >= 4 for r in records),
             "collision-free and overlapping assemblies")
    # Geometric navigation guides do not become new pixel commands under D4.
    pair = puzzles[0]["train"][0]
    for reflected in (False, True):
        for k in range(4):
            inp, expected = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
            if reflected:
                inp, expected = np.fliplr(inp), np.fliplr(expected)
            _require(np.array_equal(solve(inp), expected), "rigid scene equivariance")
    # The specified reader supports length five beyond the sampled 2..4 regime.
    rng = np.random.default_rng(530)
    profile = dict(length=5, fg=4, guides=[1, 2, 6], base_index=0, ports=2,
                   selected=0, distance="any", steps=[1, 2], width_range=(29, 31),
                   cargo_width=(3, 6), cargo_style="step", anchor_style="frame",
                   witness=False, gap=1, offset_ports=True, distinct_offsets=True,
                   rotation=0, reflected=False)
    for attempt in range(100):
        try:
            extended = _make_pair(rng, profile)
            break
        except ValueError:
            continue
    else:
        raise AssertionError("length-five support regression exhausted")
    original = extended["input"].copy()
    _require(np.array_equal(solve(original), extended["output"]), "length-five solve")
    _require(np.array_equal(original, extended["input"]), "solve must not mutate its input")
    points = np.argwhere(original != 0)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0)
    tight_input = original[top:bottom + 1, left:right + 1]
    _require(np.array_equal(solve(tight_input), extended["output"]), "irrelevant staging margins")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(173 + n * 5 + m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Every witness independently rejects all listed A omissions/alternatives.
    for mode in _ASSEMBLY_ALTERNATIVES:
        _require(all(_pair_info(p["train"][0])["alternatives"][mode] for p in puzzles),
                 f"missing first-demonstration discriminator {mode}")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"] = np.zeros_like(bad["test"][0]["output"])
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid output accepted")
    return dict(seeds=42, pairs=len(records), input_shapes=len({r["shape"] for r in records}),
                output_shapes=len({r["output_shape"] for r in records}),
                binary_output_structures=len({r["output_mask"] for r in records}),
                bars=sorted({r["length"] for r in records}), ports=sorted({r["ports"] for r in records}),
                turns=sorted({r["turns"] for r in records}),
                overlap_range=[min(r["overlap"] for r in records), max(r["overlap"] for r in records)])


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
