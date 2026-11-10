"""Census piers: object counts rebuild foundations under a demolished scaffold.

Only the standard library and NumPy are used. No parent generator is imported.
The solver reads all roles, objects, rods, and commands from the queried grid.
"""
from __future__ import annotations

import json
import numpy as np


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _check(np.all((0 <= a) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=True)


def _components(grid, background):
    """Four-connected, monochromatic components below the control separator."""
    h, w = grid.shape
    seen = set()
    parts = []
    for r in range(2, h):
        for c in range(w):
            if (r, c) in seen or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            cells = set()
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p = rr + dr, cc + dc
                    if (2 <= p[0] < h and 0 <= p[1] < w and p not in seen
                            and int(grid[p]) == color):
                        seen.add(p)
                        stack.append(p)
            parts.append((color, frozenset(cells)))
    return parts


def _socket_cells(floor, left, width, wall):
    return frozenset({(floor, c) for c in range(left, left + width)} |
                     {(r, c) for r in range(floor - wall, floor)
                      for c in (left, left + width - 1)})


def _socket(color, cells, floor):
    rows, cols = zip(*cells)
    left, right = min(cols), max(cols)
    wall = floor - min(rows)
    width = right - left + 1
    if (max(rows) == floor and width in (3, 5, 7) and wall in (1, 2)
            and cells == _socket_cells(floor, left, width, wall)):
        return dict(color=color, left=left, width=width, wall=wall,
                    center=(left + right) // 2, cells=cells)
    return None


# Rod: (orientation, top row, left column, length, color).
def _cells(rod):
    o, r, c, n, _ = rod
    return frozenset((r + (i if o == "V" else 0),
                      c + (i if o == "H" else 0)) for i in range(n))


def _shift(rod, distance):
    o, r, c, n, color = rod
    return o, r + distance, c, n, color


def _settle(rods, fixed, height):
    """Greatest simultaneously movable set; any contact can support a rod.

    Moving an entire supported pile in one tick avoids treating a falling rod
    as a permanent obstruction. All pieces translate without rotation.
    """
    result = list(rods)
    for _ in range(height * (len(result) + 1)):
        bodies = [_cells(p) for p in result]
        targets = [frozenset((r + 1, c) for r, c in s) for s in bodies]
        moving = {i for i, s in enumerate(targets)
                  if all(r < height for r, c in s) and not s & fixed}
        while moving:
            blocked = {i for i in moving if any(targets[i] & bodies[j]
                       for j in range(len(result)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1) if i in moving else p for i, p in enumerate(result)]
    raise AssertionError("gravity did not converge")


def _piers(sockets, counts, floor):
    return [(s["color"], frozenset((floor - i, s["center"])
                                  for i in range(1, n + 1)))
            for s, n in zip(sockets, counts)]


def _render(shape, bg, band, sockets, piers, rods):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[0] = band
    occupied = set()
    for color, cells in ([(s["color"], s["cells"]) for s in sockets] +
                         list(piers) + [(p[4], _cells(p)) for p in rods]):
        _check(not occupied & cells, "overlapping physical pieces")
        for cell in cells:
            grid[cell] = color
        occupied.update(cells)
    return grid


def _inspect(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    _check(h >= 8 and w >= 5, "scene too small")
    bg = int(grid[1, 0])
    _check(np.all(grid[1] == bg), "control separator is not blank")
    commands = tuple(int(x) for x in grid[0] if int(x) != bg)
    _check(1 <= len(commands) <= 3 and len(set(commands)) == len(commands), "commands")
    parts = _components(grid, bg)
    sockets = []
    tickets_and_rods = []
    for color, cells in parts:
        s = _socket(color, cells, h - 1)
        if s is not None:
            sockets.append(s)
        else:
            tickets_and_rods.append((color, cells))
    sockets.sort(key=lambda s: s["center"])
    _check(1 <= len(sockets) <= 4, "expected one to four open sockets")
    category_colors = {s["color"] for s in sockets}
    _check(len(category_colors) == len(sockets), "socket category colors repeat")
    _check(not category_colors & set(commands), "commands must name scaffold colors")
    fixed = frozenset().union(*(s["cells"] for s in sockets))
    _check(sum(len(s["cells"]) for s in sockets) == len(fixed), "sockets overlap")
    for a, b in zip(sockets, sockets[1:]):
        _check(a["left"] + a["width"] < b["left"], "sockets need a ground gap")
    tickets, rods = [], []
    for color, cells in tickets_and_rods:
        if color in category_colors:
            _check(1 <= len(cells) <= 12, "ticket area outside family")
            _check(all(r < h - 1 for r, c in cells), "ticket on ground")
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        p = r + dr, c + dc
                        if 2 <= p[0] < h and 0 <= p[1] < w and p not in cells:
                            _check(int(grid[p]) == bg, "ticket must have a background moat")
            tickets.append((color, cells))
        else:
            rows, cols = zip(*cells)
            _check(len(cells) >= 2 and (len(set(rows)) == 1 or len(set(cols)) == 1),
                   "scaffold component is not one fully visible straight rod")
            o = "H" if len(set(rows)) == 1 else "V"
            p = o, min(rows), min(cols), len(cells), color
            _check(_cells(p) == cells, "rod has a gap")
            rods.append(p)
    _check(bool(rods), "scaffold missing")
    counts = tuple(sum(c == s["color"] for c, cells in tickets) for s in sockets)
    areas = tuple(sum(len(cells) for c, cells in tickets if c == s["color"])
                  for s in sockets)
    _check(all(0 <= n <= 6 for n in counts), "pier count exceeds six")
    _check(all(h - 1 - n >= 2 for n in counts), "pier reaches control band")
    piers = _piers(sockets, counts, h - 1)
    foundation = fixed | frozenset().union(*(cells for color, cells in piers))
    survivors = [p for p in rods if p[4] not in commands]
    _check(all(not _cells(p) & foundation for p in survivors), "pier grows into a survivor")
    _check(_settle(rods, fixed, h) == rods, "input scaffold is not initially stable")
    settled = _settle(survivors, foundation, h)
    return dict(grid=grid, background=bg, commands=commands, sockets=sockets,
                tickets=tickets, counts=counts, areas=areas, rods=rods,
                survivors=survivors, settled=settled, piers=piers,
                fixed=fixed, foundation=foundation)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible input; no generation state."""
    d = _inspect(input_grid)
    return _render(d["grid"].shape, d["background"], d["grid"][0],
                   d["sockets"], d["piers"], d["settled"])


def _shape(rng, area=None):
    if area is None:
        area = int(rng.choice((1, 2, 3, 4, 5, 6, 7, 8),
                              p=(.12, .22, .19, .16, .12, .10, .05, .04)))
    kind = int(rng.integers(4))
    if kind == 0 and area <= 6:
        cells = {(0, i) for i in range(area)}
    elif kind == 1:
        width = int(rng.integers(2, 4))
        cells = {(i // width, i % width) for i in range(area)}
    else:
        cells = {(1, 1)}
        while len(cells) < area:
            choices = sorted({(r + dr, c + dc) for r, c in cells
                              for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                              if 0 <= r + dr < 3 and 0 <= c + dc < 4} - cells)
            cells.add(choices[int(rng.integers(len(choices)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    low_r, low_c = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - low_r, c - low_c) for r, c in cells))


def _band(rng, width, bg, commands):
    result = np.full(width, bg, dtype=np.int8)
    positions = rng.choice(width, size=len(commands), replace=False)
    for c, color in zip(positions, commands):
        result[int(c)] = color
    return result


def _ground_layout(rng, k, width, floor, colors, core):
    widths = [3] * k
    budget = width - (3 * k + k - 1)
    for i in rng.permutation(k):
        extra = int(rng.choice([v for v in (0, 2, 4) if v <= budget - 2])) if budget >= 2 else 0
        widths[int(i)] += extra
        budget -= extra
    gaps = [1] * (k - 1)
    for i in range(k - 1):
        extra = int(rng.integers(min(3, budget) + 1))
        gaps[i] += extra
        budget -= extra
    left = int(rng.integers(budget + 1))
    sockets = []
    for i in range(k):
        wall = 1 if core else int(rng.integers(1, 3))
        sockets.append(dict(color=colors[i], left=left, width=widths[i], wall=wall,
                            center=left + widths[i] // 2,
                            cells=_socket_cells(floor, left, widths[i], wall)))
        left += widths[i] + (gaps[i] if i < k - 1 else 0)
    return sockets


def _rod_color(rng, body, rods, choices):
    neighbors = body | frozenset((r + dr, c + dc) for r, c in body
                                 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    forbidden = {p[4] for p in rods if neighbors & _cells(p)}
    available = [int(c) for c in choices if int(c) not in forbidden]
    return None if not available else int(rng.choice(available))


def _drop_add(rng, rods, fixed, h, w, o, col, length, choices, future):
    p = o, 2, col, length, -1
    body = _cells(p)
    occupied = fixed | frozenset().union(*(_cells(q) for q in rods))
    if any(r >= h or c >= w or c < 0 for r, c in body) or body & occupied:
        return False
    while True:
        target = _cells(_shift(p, 1))
        if any(r >= h for r, c in target) or target & occupied:
            break
        p = _shift(p, 1)
    color = _rod_color(rng, _cells(p), rods, choices)
    if color is None or (color not in future[0] and _cells(p) & future[1]):
        return False
    rods.append((*p[:4], color))
    return True


def _model(rng, k, count_ceiling, core=False, roomy=False):
    h = 30 if roomy else int(rng.integers(19 if core else 22, 28 if core else 31))
    w = 30 if roomy else int(rng.integers(max(14 if core else 18, 4 * k), 31))
    palette = [int(c) for c in rng.permutation(10)]
    bg, categories = palette[0], palette[1:k + 1]
    rod_palette = palette[k + 1:k + 5]
    ncmd = 1 if core else int(rng.integers(1, 3))
    commands = tuple(rod_palette[:ncmd])
    sockets = _ground_layout(rng, k, w, h - 1, categories, core)
    fixed = frozenset().union(*(s["cells"] for s in sockets))
    # Future columns reserve enough room for every later ticket count. These
    # reservations constrain generation only; the solver counts actual pixels.
    future_cells = frozenset((h - 1 - n, s["center"])
                             for s in sockets for n in range(1, count_ceiling + 1))
    future = set(commands), future_cells
    rods = []
    leg_length = int(rng.integers(count_ceiling + 2, count_ceiling + 5))
    for s in sockets:
        length = leg_length if core else leg_length + int(rng.integers(-1, 2))
        color = int(rng.choice(commands))
        rods.append(("V", h - 1 - length, s["center"], length, color))
    # The teaching bridge covers the first two of three sockets. In its second
    # counterfactual the tallest pier is outside this span. Ordinary scenes
    # combine whole-span bridges with local or overlapping shelves.
    if core:
        first, last = 0, 1
    elif k == 1 or rng.random() < .35:
        first, last = 0, k - 1
    else:
        first = int(rng.integers(k))
        last = int(rng.integers(first, k))
        if first == 0 and last == k - 1:
            last -= 1
    left = sockets[first]["left"]
    right = sockets[last]["left"] + sockets[last]["width"] - 1
    survivor_colors = [c for c in rod_palette if c not in commands]
    _check(_drop_add(rng, rods, fixed, h, w, "H", left, right - left + 1,
                     survivor_colors, future), "could not add main bridge")
    if core:
        # A floor-standing rail touches the bridge's side without supporting
        # its downward footprint. The released bridge slides beside this rail;
        # adjacency does not glue independent differently colored straight rods.
        rail = ("V", rods[-1][1], right + 1, h - rods[-1][1], -1)
        rail_color = _rod_color(rng, _cells(rail), rods, survivor_colors)
        _check(rail_color is not None, "could not color side rail")
        rods.append((*rail[:4], rail_color))
        x = int(rng.integers(left, right + 1))
        _drop_add(rng, rods, fixed, h, w, "V", x, int(rng.integers(2, 5)),
                  survivor_colors, future)
        span = min(right - left + 1, int(rng.integers(3, 9)))
        cap_left = max(left, min(x - span // 2, right - span + 1))
        _drop_add(rng, rods, fixed, h, w, "H", cap_left, span,
                  survivor_colors, future)
    else:
        if first > 0 or last < k - 1:
            other = int(rng.choice([i for i in range(k) if i < first or i > last]))
            s = sockets[other]
            _drop_add(rng, rods, fixed, h, w, "H", s["left"], s["width"],
                      survivor_colors, future)
            if rng.random() < .30:
                broad_left = sockets[0]["left"]
                broad_right = sockets[-1]["left"] + sockets[-1]["width"] - 1
                _drop_add(rng, rods, fixed, h, w, "H", broad_left,
                          broad_right - broad_left + 1, survivor_colors, future)
        # A shorter surviving post can compete with a rebuilt pier. Its column
        # is outside all tally columns and socket walls.
        forbidden_x = {s["center"] for s in sockets} | {
            c for s in sockets for c in (s["left"], s["left"] + s["width"] - 1)}
        xs = [x for x in range(w) if x not in forbidden_x]
        if xs and rng.random() < .65:
            x = int(rng.choice(xs))
            length = int(rng.integers(2, count_ceiling + 3))
            # Insert from above only when it can reach an unobstructed low spot.
            candidate = ("V", h - length, x, length, survivor_colors[0])
            occupied = fixed | frozenset().union(*(_cells(q) for q in rods))
            if not _cells(candidate) & occupied:
                color = _rod_color(rng, _cells(candidate), rods, survivor_colors)
                if color is not None:
                    rods.append((*candidate[:4], color))
        for _ in range(int(rng.integers(4, 11))):
            o = "H" if rng.random() < .62 else "V"
            length = int(rng.integers(2, min(w, 14) + 1)) if o == "H" else int(rng.integers(2, 6))
            x = int(rng.integers(0, w - length + 1)) if o == "H" else int(rng.integers(w))
            _drop_add(rng, rods, fixed, h, w, o, x, length, rod_palette, future)
    # Occasionally name an absent identity as well; equality remains local.
    if not core and len(commands) < 3 and rng.random() < .25:
        available = [c for c in palette[k + 5:] if c not in commands]
        if available:
            commands += (int(rng.choice(available)),)
    band = _band(rng, w, bg, commands)
    _check(_settle(rods, fixed, h) == rods, "builder made unstable rods")
    return dict(shape=(h, w), background=bg, band=band, sockets=sockets,
                commands=commands, rods=rods, fixed=fixed)


def _pack_tickets(rng, base, specs):
    h, w = base.shape
    for _ in range(16):
        grid = base.copy()
        occupied = grid != int(base[1, 0])
        # Ticket pixels have an eight-neighbor background moat. Row 1 must
        # stay blank, and row 0 is always a control band.
        forbidden = occupied.copy()
        for r, c in zip(*np.nonzero(occupied)):
            forbidden[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
        forbidden[:2] = True
        forbidden[-1] = True
        order = sorted(range(len(specs)), key=lambda i: (-len(specs[i][1]), float(rng.random())))
        success = True
        for i in order:
            color, shape = specs[i]
            sh = max(r for r, c in shape) + 1
            sw = max(c for r, c in shape) + 1
            candidates = np.ones((h - sh + 1, w - sw + 1), dtype=bool)
            for dr, dc in shape:
                candidates &= ~forbidden[dr:dr + h - sh + 1, dc:dc + w - sw + 1]
            rr, cc = np.nonzero(candidates)
            if len(rr) == 0:
                success = False
                break
            j = int(rng.integers(len(rr)))
            r0, c0 = int(rr[j]), int(cc[j])
            for dr, dc in shape:
                r, c = r0 + dr, c0 + dc
                grid[r, c] = color
                forbidden[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
        if success:
            return grid
    return None


def _pair(rng, model, counts, forced=None):
    forced = {} if forced is None else forced
    tickets = [(s["color"], _shape(rng, forced.get(i, [None] * n)[j]))
               for i, (s, n) in enumerate(zip(model["sockets"], counts)) for j in range(n)]
    base = _render(model["shape"], model["background"], model["band"],
                   model["sockets"], [], model["rods"])
    inp = _pack_tickets(rng, base, tickets)
    if inp is None:
        return None
    piers = _piers(model["sockets"], counts, model["shape"][0] - 1)
    fixed = model["fixed"] | frozenset().union(*(cells for color, cells in piers))
    survivors = [p for p in model["rods"] if p[4] not in model["commands"]]
    out = _render(model["shape"], model["background"], model["band"], model["sockets"],
                  piers, _settle(survivors, fixed, model["shape"][0]))
    return {"input": inp, "output": out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Two controlled demonstrations always survive the minimum legal count.
    k = 3
    c0 = [1, 3, 0]
    c1 = [2, 1, 3]
    for attempt in range(32):
        m = _model(rng, k, 3, core=True, roomy=attempt >= 16)
        a = _pair(rng, m, c0, {0: [4], 1: [1, 1, 2]})
        b = _pair(rng, m, c1, {0: [2, 2], 1: [6]})
        if a is not None and b is not None:
            break
    else:
        raise AssertionError("could not pack teaching counterfactuals")
    puzzle = {"train": [a, b], "test": []}
    seen = {_key(a["input"]), _key(b["input"])}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for index in range(count):
            for attempt in range(48):
                k = int(rng.integers(1, 5))
                maximum = 3 if split == "train" else 6
                counts = [int(x) for x in rng.integers(0, maximum + 1, size=k)]
                if split == "test" and index == 0:
                    counts[int(rng.integers(k))] = int(rng.integers(4, 7))
                while sum(counts) > 16:
                    counts[int(np.argmax(counts))] -= 1
                model = _model(rng, k, maximum, roomy=attempt >= 24)
                pair = _pair(rng, model, counts)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("could not generate distinct packed scene")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(details):
    rows = [(s["color"], n, a) for d in details
            for s, n, a in zip(d["sockets"], d["counts"], d["areas"])]
    return (any(c == cc and a == aa and n != nn for c, n, a in rows for cc, nn, aa in rows),
            any(n == nn and a != aa for c, n, a in rows for cc, nn, aa in rows))


def _metrics(d):
    moved = [q[1] - p[1] for p, q in zip(d["survivors"], d["settled"])]
    without_piers = _settle(d["survivors"], d["fixed"], d["grid"].shape[0])
    independent_contact = False
    for i, p in enumerate(d["survivors"]):
        neighbors = frozenset((r + dr, c + dc) for r, c in _cells(p)
                              for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if any(neighbors & _cells(q) and moved[i] != moved[j]
               for j, q in enumerate(d["survivors"]) if i != j):
            independent_contact = True
    envelopes = []
    local_span_witness = False
    for p in d["settled"]:
        if p[0] == "H":
            covered = [p[2] <= s["center"] < p[2] + p[3] for s in d["sockets"]]
            envelopes.append(sum(covered))
            inside = [n for n, cover in zip(d["counts"], covered) if cover]
            outside = [n for n, cover in zip(d["counts"], covered) if not cover]
            if (inside and outside and max(outside) > max(inside)
                    and p[1] >= d["grid"].shape[0] - 1 - max(outside)):
                local_span_witness = True
    return dict(shape=d["grid"].shape, background=d["background"],
                sockets=len(d["sockets"]), widths=tuple(s["width"] for s in d["sockets"]),
                walls=tuple(s["wall"] for s in d["sockets"]), counts=d["counts"],
                areas=d["areas"], commands=len(d["commands"]), rods=len(d["rods"]),
                selected=sum(p[4] in d["commands"] for p in d["rods"]),
                lengths=tuple(p[3] for p in d["rods"]),
                moving=sum(x > 0 for x in moved), distances=tuple(sorted(set(moved) - {0})),
                mechanical_count_effect=without_piers != d["settled"],
                pooled_support=any(x >= 2 for x in envelopes),
                local_span_witness=local_span_witness,
                independent_contact=independent_contact,
                ticket_masks=tuple(tuple(sorted((r - min(rr for rr, cc in cells),
                                                 c - min(cc for rr, cc in cells))
                                                for r, c in cells)) for color, cells in d["tickets"]))


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "output shape changed")
            key = _key(inp)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect(inp)
            expected = _render(inp.shape, d["background"], inp[0], d["sockets"],
                               d["piers"], d["settled"])
            _check(np.array_equal(out, expected), "wrong pier/deletion/gravity output")
            _check(np.array_equal(inp[:2], out[:2]), "control band changed")
            _check(all(q[1] >= p[1] and q[0] == p[0] and q[2:] == p[2:]
                       for p, q in zip(d["survivors"], d["settled"])), "rigidity violated")
            _check(_settle(d["settled"], d["foundation"], inp.shape[0]) == d["settled"],
                   "output scaffold not stable")
            _check(not any(np.any(out[2:] == c) for c in d["commands"]), "selected rod survived")
            for color, cells in d["piers"]:
                _check(all(int(out[p]) == color for p in cells), "unary pier altered")
            details[split].append(d)
    collision, independence = _evidence(details["train"])
    _check(collision and independence, "area independence not demonstrated")
    train_metrics = [_metrics(d) for d in details["train"]]
    _check(any(m["moving"] for m in train_metrics), "gravity not demonstrated")
    _check(any(m["mechanical_count_effect"] for m in train_metrics),
           "counting never changes a survivor's landing")
    _check(any(m["pooled_support"] for m in train_metrics), "no bridge over several categories")
    _check(any(m["local_span_witness"] for m in train_metrics),
           "training permits a scene-wide maximum height shortcut")
    _check(any(m["independent_contact"] for m in train_metrics),
           "independent motion at a side contact not demonstrated")
    _check({1, 2, 3} <= {n for d in details["train"] for n in d["counts"]},
           "unary height progression missing")


def _mechanics_tests():
    # Two redundant legs keep a beam up; deleting both lets it land on the
    # tallest new pier within its span, while a shorter pier remains visible.
    sockets = [dict(color=2, left=1, width=3, wall=1, center=2,
                    cells=_socket_cells(11, 1, 3, 1)),
               dict(color=3, left=9, width=3, wall=1, center=10,
                    cells=_socket_cells(11, 9, 3, 1))]
    fixed = frozenset().union(*(s["cells"] for s in sockets))
    legs = [("V", 6, 2, 5, 1), ("V", 6, 10, 5, 6)]
    beam = ("H", 5, 1, 11, 4)
    _check(_settle([legs[0], beam], fixed, 12)[1] == beam, "redundant leg test")
    piers = _piers(sockets, [1, 3], 11)
    foundation = fixed | frozenset().union(*(cells for color, cells in piers))
    _check(_settle([beam], foundation, 12) == [("H", 7, 1, 11, 4)], "support maximum")
    # A beam outside the high pier's column settles lower, so a global maximum
    # height for every survivor is not the rule.
    short = ("H", 4, 1, 4, 5)
    _check(_settle([short], foundation, 12) == [("H", 9, 1, 4, 5)], "span-specific support")
    # Upper pieces fall with their support rather than staying suspended.
    top = ("V", 3, 6, 2, 5)
    _check(_settle([beam, top], foundation, 12) ==
           [("H", 7, 1, 11, 4), ("V", 5, 6, 2, 5)], "support cascade")
    # Input-only fixture outside the builder's templates: demolition removes a
    # horizontal floor plank, the surviving post drops one row, and a new pier
    # keeps the formerly supported beam at its old height.
    s = [dict(color=2, left=1, width=3, wall=1, center=2,
              cells=_socket_cells(11, 1, 3, 1))]
    rods = [("H", 11, 6, 5, 1), ("V", 9, 8, 2, 4),
            ("H", 8, 1, 9, 5), ("V", 6, 4, 2, 6)]
    band = np.zeros(13, dtype=np.int8)
    band[3] = 1
    inp = _render((12, 13), 0, band, s, [], rods)
    inp[2, 1] = inp[2, 6] = 2
    expected = _render((12, 13), 0, band, s, _piers(s, [2], 11),
                       [("V", 10, 8, 2, 4), rods[2], rods[3]])
    _check(np.array_equal(solve(inp), expected), "input-only non-template fixture")


def self_test():
    """Bounded checks and actual feature measurements on 42 default puzzles."""
    _mechanics_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = set()
    records = []
    for puzzle in puzzles:
        signatures.add(tuple(_key(p[k]) for s in ("train", "test")
                             for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _check(np.array_equal(solve(pair["input"]), pair["output"]), "solver agreement")
                records.append(_metrics(_inspect(pair["input"])))
    _check(len(signatures) == 42, "seed diversity")
    _check({r["background"] for r in records} == set(range(10)), "background coverage")
    _check({r["sockets"] for r in records} == {1, 2, 3, 4}, "socket-count variation")
    _check({n for r in records for n in r["counts"]} == set(range(7)), "count coverage")
    _check({n for r in records for n in r["widths"]} == {3, 5, 7}, "socket width variation")
    _check({n for r in records for n in r["walls"]} == {1, 2}, "wall variation")
    _check(len({r["shape"] for r in records}) >= 40, "canvas diversity")
    _check(len({m for r in records for m in r["ticket_masks"]}) >= 40, "shape vocabulary")
    _check(len({r["rods"] for r in records}) >= 6, "scaffold-count variation")
    _check(len({n for r in records for n in r["lengths"]}) >= 12, "rod lengths")
    _check(sum(len(r["distances"]) >= 2 for r in records) >= 10, "unequal falls missing")
    _check(sum(r["mechanical_count_effect"] for r in records) >= 80, "piers seldom matter")
    a, b = generate_puzzle(137, 7, 3), generate_puzzle(137, 7, 3)
    _check(all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output")), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _check((len(p["train"]), len(p["test"])) == (nt, nq), "legal counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt a label to make sure validation checks the actual answer.
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted bad output")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                canvas_shapes=len({r["shape"] for r in records}),
                ticket_masks=len({m for r in records for m in r["ticket_masks"]}),
                rod_counts=sorted({r["rods"] for r in records}),
                unequal_fall_pairs=sum(len(r["distances"]) >= 2 for r in records),
                local_span_pairs=sum(r["local_span_witness"] for r in records),
                independent_contact_pairs=sum(r["independent_contact"] for r in records),
                mechanical_count_pairs=sum(r["mechanical_count_effect"] for r in records))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
