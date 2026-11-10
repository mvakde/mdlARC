"""Registered contact circuits. Only Python's standard library and NumPy.

The construction keeps explicit cable paths. The solver independently recovers
rectangles, palettes, directional paths, and registered contacts from pixels.
"""
from itertools import permutations

import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
PATTERNS = (DIAMOND, CHECKER)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=False)


def _components(cells):
    remaining, result = {tuple(map(int, p)) for p in cells}, []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        found, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _crop(g):
    points = np.argwhere(g != 0)
    _require(len(points) > 0, "empty circuit")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return g[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _trace(field, carrier, centers):
    """Trace orthogonal strands; a degree-four crossing continues straight."""
    cells = {tuple(map(int, p)) for p in np.argwhere(field == carrier)}
    top, bottom = 0, field.shape[0] - 1
    ends = {(r, c) for r in (top, bottom) for c in centers}
    edges = {}
    for p in cells:
        edges[p] = {d: (p[0] + dr, p[1] + dc)
                    for d, (dr, dc) in enumerate(DIRECTIONS)
                    if (p[0] + dr, p[1] + dc) in cells}
        _require(len(edges[p]) == (1 if p in ends else 2) or
                 (p not in ends and len(edges[p]) == 4), "broken or branching cable")
    _require(ends <= cells, "missing cable endpoint")
    pairing, routes, seen = [], [], set()
    for c in centers:
        p, incoming, route, states = (top, c), None, [], set()
        while True:
            _require((p, incoming) not in states, "cyclic cable")
            states.add((p, incoming))
            route.append(p)
            seen.add(p)
            if p[0] == bottom:
                _require(p in ends, "unexpected endpoint")
                pairing.append(centers.index(p[1]))
                break
            if incoming is None:
                d = 2
            elif len(edges[p]) == 4:
                d = (incoming + 2) % 4
            else:
                choices = [d for d in edges[p] if d != incoming]
                _require(len(choices) == 1, "ambiguous cable")
                d = choices[0]
            _require(d in edges[p], "missing directional edge")
            p, incoming = edges[p][d], (d + 2) % 4
            _require(p[0] != top, "route returns to first bank")
        routes.append(route)
    _require(sorted(pairing) == list(range(len(centers))), "routing is not a bijection")
    _require(seen == cells, "unattached wire")
    return pairing, routes, sum(len(e) == 4 for e in edges.values())


def _read_terminal(block, tint, source):
    if np.all(block == tint):
        return None, "blank"
    colors = set(map(int, np.unique(block))) - {tint}
    _require(len(colors) == 1, "multicolor terminal")
    color = colors.pop()
    if source and np.all(block == color):
        return color, "solid"
    pattern = DIAMOND if source else CHECKER
    _require(np.array_equal(block, np.where(pattern, color, tint)), "invalid terminal stencil")
    return color, "diamond" if source else "checker"


def _read_card(card, source):
    h, w = card.shape
    guide = 8 if source else 9
    tint = int(card[0, -1])
    _require(tint in range(1, 8) and h >= 9, "invalid transparent sheet")
    marks = np.argwhere(card == guide)
    _require(2 <= len(marks) <= 5, "registration length must be 2..5")
    row = h - 2 if source else 1
    xs = sorted(map(int, marks[:, 1]))
    _require(np.all(marks[:, 0] == row) and xs == list(range(xs[0], xs[0] + len(xs))),
             "invalid registration bar")
    field = card[3:h - 3]
    inks = set(map(int, np.unique(field))) - {tint}
    _require(len(inks) == 1, "routing field must use one ink")
    carrier = inks.pop()
    _require(carrier in range(1, 8) and carrier != tint, "invalid wire ink")
    centers = list(map(int, np.flatnonzero(field[0] == carrier)))
    _require(2 <= len(centers) <= 4, "there must be 2..4 cables per sheet")
    _require(centers == list(map(int, np.flatnonzero(field[-1] == carrier))), "unequal port banks")
    _require(centers[0] >= 1 and centers[-1] + 1 < xs[0], "bank outside sheet")
    gaps = set(map(int, np.diff(centers)))
    _require(len(gaps) == 1 and next(iter(gaps)) in (4, 5), "invalid bank pitch")
    pairing, routes, crossings = _trace(field, carrier, centers)
    outer = 0 if source else h - 3
    inner = h - 3 if source else 0
    labels, kinds, pads = [], [], []
    used = np.zeros(card.shape, dtype=bool)
    used[3:h - 3] = True
    used[row, xs] = True
    for c in centers:
        label, kind = _read_terminal(card[outer:outer + 3, c - 1:c + 2], tint, source)
        labels.append(label)
        kinds.append(kind)
        block = card[inner:inner + 3, c - 1:c + 2]
        matches = [i for i, pattern in enumerate(PATTERNS)
                   if np.array_equal(block, np.where(pattern, carrier, tint))]
        _require(len(matches) == 1, "invalid contact pad")
        pads.append(matches[0])
        used[outer:outer + 3, c - 1:c + 2] = True
        used[inner:inner + 3, c - 1:c + 2] = True
    _require(np.all(card[~used] == tint), "unexpected sheet content")
    return dict(card=card, source=source, tint=tint, carrier=carrier, guide=guide,
                marker=(row, xs[0]), length=len(xs), centers=centers, pitch=next(iter(gaps)),
                permutation=pairing, routes=routes, crossings=crossings, pads=pads,
                labels=labels, kinds=kinds, outer=outer, inner=inner)


def _parse(input_grid):
    original = _grid(input_grid)
    parts = _components(map(tuple, np.argwhere(original != 0)))
    _require(len(parts) == 2, "exactly two sheets required")
    rectangles = []
    for part in parts:
        t, b, l, r = _bounds(part)
        _require(len(part) == (b - t + 1) * (r - l + 1), "sheet is not a filled rectangle")
        block = original[t:b + 1, l:r + 1]
        guides = set(map(int, np.unique(block))) & {8, 9}
        _require(len(guides) == 1, "one guide per sheet required")
        rectangles.append((guides.pop(), block, (t, b, l, r)))
    _require({guide for guide, _, _ in rectangles} == {8, 9}, "missing sheet")
    candidates = []
    for rotation in range(4):
        try:
            cards = {}
            for guide, block, bounds in rectangles:
                info = _read_card(np.rot90(block, rotation), guide == 8)
                t, b, l, r = bounds
                h, w = original.shape
                info["origin"] = ((t, l), (w - 1 - r, t),
                                  (h - 1 - b, w - 1 - r), (l, h - 1 - b))[rotation]
                cards[guide] = info
            a, b = cards[8], cards[9]
            _require(a["length"] == b["length"] and a["pitch"] == b["pitch"], "incongruent registration")
            _require(a["carrier"] == b["carrier"] and a["tint"] != b["tint"], "inconsistent sheet palette")
            relative_a = [c - a["marker"][1] for c in a["centers"]]
            relative_b = [c - b["marker"][1] for c in b["centers"]]
            _require(all(x == y or abs(x - y) >= 3 for x in relative_a for y in relative_b),
                     "contact pads overlap only partially")
            structural = {a["carrier"], a["tint"], b["tint"], 0, 8, 9}
            _require(all(c is None or c not in structural for x in (a, b) for c in x["labels"]),
                     "label color is structural")
            # Normalize each marker to (0,0); only the visible bar determines
            # translation. These are the coordinates of the assembled sheets.
            for x in (a, b):
                x["placed"] = (-x["marker"][0], -x["marker"][1])
            candidates.append(dict(a=a, b=b, rotation=rotation, grid=np.rot90(original, rotation)))
        except ValueError:
            continue
    _require(len(candidates) == 1, "input must have one geometric interpretation")
    return candidates[0]


def _registered_contacts(info, mode="actual"):
    a, b = info["a"], info["b"]
    if mode == "rank":
        candidates = list(zip(range(len(a["centers"])), range(len(b["centers"]))))
    else:
        candidates = [(j, k) for j, x in enumerate(a["centers"])
                      for k, y in enumerate(b["centers"])
                      if x - a["marker"][1] == y - b["marker"][1]]
    return [(j, k) for j, k in candidates
            if mode == "all" or (mode != "none" and a["pads"][j] != b["pads"][k])]


def _assemble(info, contact_mode="actual", routing=True, reverse=True, solid=False):
    """Pixel-only completion, with explicit counterfactuals for validation."""
    a, b = info["a"], info["b"]
    labels_a, labels_b = list(a["labels"]), list(b["labels"])
    for j, k in _registered_contacts(info, contact_mode):
        i = a["permutation"].index(j) if routing else j
        q = b["permutation"][k] if routing else k
        values = {v for v in (labels_a[i], labels_b[q]) if v is not None}
        _require(len(values) == 1, "conducting cable has absent or conflicting clues")
        label = values.pop()
        if reverse or labels_a[i] is not None:
            labels_a[i] = labels_b[q] = label
    coords = []
    for card in (a, b):
        dy, dx = card["placed"]
        coords.extend([(dy, dx), (dy + card["card"].shape[0] - 1,
                                  dx + card["card"].shape[1] - 1)])
    top, bottom, left, right = _bounds(coords)
    _require(bottom - top < 30 and right - left < 30, "assembled sheets exceed 30")
    out = np.zeros((bottom - top + 1, right - left + 1), dtype=np.int8)
    # Union of ink, not opaque rectangle pasting. Terminal clues are redrawn
    # below; they do not take part in the conductivity test.
    for card in (a, b):
        dy, dx = card["placed"][0] - top, card["placed"][1] - left
        for r, c in np.argwhere(card["card"] == card["carrier"]):
            out[r + dy, c + dx] = card["carrier"]
    for card, labels in ((a, labels_a), (b, labels_b)):
        dy, dx = card["placed"][0] - top, card["placed"][1] - left
        pattern = DIAMOND if card["source"] else CHECKER
        if solid:
            pattern = np.ones((3, 3), dtype=bool)
        for c, label in zip(card["centers"], labels):
            if label is not None:
                r = card["outer"] + dy
                out[r:r + 3, c + dx - 1:c + dx + 2] = np.where(pattern, label, 0)
    return np.rot90(_crop(out), -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the query pixels alone. No RNG, seed, examples, or cache."""
    return _assemble(_parse(input_grid))


def _moves(order, rng):
    """Realize a permutation by moving cycles through a vacant exterior track."""
    n = len(order)
    state, result = list(range(n)) + [None], []
    for slot in map(int, rng.permutation(n)):
        if state[slot] == order[slot]:
            continue
        saved = state[slot]
        result.append((slot, n))
        state[n], state[slot] = saved, None
        hole = slot
        while order[hole] != saved:
            old = state.index(order[hole])
            result.append((old, hole))
            state[hole], state[old] = state[old], None
            hole = old
        result.append((n, hole))
        state[hole], state[n] = saved, None
    assert state[:n] == list(order)
    return result


def _order(rng, n, complex_route=False, identity=False):
    result = list(range(n))
    if identity:
        return result
    if complex_route:
        # A 3/4-cycle or two disjoint swaps: 4..6 lateral moves.
        ids = list(map(int, rng.permutation(n)))
        if n == 4 and rng.random() < .35:
            for i, j in ((ids[0], ids[1]), (ids[2], ids[3])):
                result[i], result[j] = result[j], result[i]
        else:
            for i, j in zip(ids, ids[1:] + ids[:1]):
                result[i] = j
    elif rng.random() < .75:
        i, j = map(int, rng.choice(n, 2, replace=False))
        result[i], result[j] = result[j], result[i]
    return result


def _plan(rng, n, order):
    moves = _moves(order, rng)
    return dict(n=n, order=order, moves=moves, lead=1, tail=1,
                gaps=[2] * max(0, len(moves) - 1), straight=2)


def _height(plan):
    m = len(plan["moves"])
    return 7 + (plan["lead"] + plan["tail"] + sum(plan["gaps"]) if m else plan["straight"])


def _draw_card(rng, plan, lattice, pitch, spare, guide_x, length, palette, source, pad_types):
    tint, carrier = palette[0 if source else 1], palette[2]
    left_pad, right_pad = int(rng.integers(2)), int(rng.integers(2))
    origin_x = min(lattice) * pitch - 1 - left_pad
    centers = [k * pitch - origin_x for k in lattice]
    gx = guide_x - origin_x
    h, w = _height(plan), gx + length + right_pad
    _require(w <= 30, "sheet too wide")
    card = np.full((h, w), tint, dtype=np.int8)
    columns = centers + [centers[-1] + spare]
    start, previous = 3, 3
    paths = [[(start, c)] for c in centers]
    state = list(range(len(centers))) + [None]
    row = start + plan["lead"]
    for index, (old, new) in enumerate(plan["moves"]):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        owner = state[old]
        step = 1 if new > old else -1
        paths[owner].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
        if index < len(plan["gaps"]):
            row += plan["gaps"][index]
    end = h - 4
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    for path in paths:
        for p in path:
            card[p] = carrier
    inner, outer = (h - 3, 0) if source else (0, h - 3)
    for c, kind in zip(centers, pad_types):
        card[inner:inner + 3, c - 1:c + 2] = np.where(PATTERNS[kind], carrier, tint)
    marker_row = inner + 1
    card[marker_row, gx:gx + length] = 8 if source else 9
    pairing = [plan["order"].index(i) for i in range(len(centers))]
    return dict(card=card, centers=centers, paths=paths, pairing=pairing,
                lattice=lattice, pads=pad_types, inner=inner, outer=outer,
                placed=(-marker_row, -gx), tint=tint, carrier=carrier)


def _terminal(card, index, color, kind):
    if color is None:
        return
    c, r = card["centers"][index], card["outer"]
    pattern = np.ones((3, 3), dtype=bool) if kind == "solid" else (
        DIAMOND if kind == "diamond" else CHECKER)
    card["card"][r:r + 3, c - 1:c + 2] = np.where(pattern, color, card["tint"])


def _construction_output(a, b, labels_a, labels_b):
    """Ground truth from explicit construction, not the input parser/solver."""
    t = min(a["placed"][0], b["placed"][0])
    l = min(a["placed"][1], b["placed"][1])
    bot = max(x["placed"][0] + x["card"].shape[0] for x in (a, b))
    right = max(x["placed"][1] + x["card"].shape[1] for x in (a, b))
    _require(bot - t <= 30 and right - l <= 30, "assembly too large")
    out = np.zeros((bot - t, right - l), dtype=np.int8)
    for card, labels, pattern in ((a, labels_a, DIAMOND), (b, labels_b, CHECKER)):
        dy, dx = card["placed"][0] - t, card["placed"][1] - l
        # Preserve all procedural paths and the logical union of pad stencils.
        for path in card["paths"]:
            for r, c in path:
                out[r + dy, c + dx] = card["carrier"]
        for c, kind in zip(card["centers"], card["pads"]):
            for dr, dc in np.argwhere(PATTERNS[kind]):
                out[card["inner"] + int(dr) + dy, c - 1 + int(dc) + dx] = card["carrier"]
        for c, color in zip(card["centers"], labels):
            if color is not None:
                r = card["outer"] + dy
                out[r:r + 3, c - 1 + dx:c + 2 + dx] = np.where(pattern, color, 0)
    return _crop(out)


def _stage(rng, a, b):
    cards = [a["card"], b["card"]]
    if rng.integers(2):
        cards.reverse()
    horizontal = sum(x.shape[1] for x in cards) < 29 and rng.random() < .55
    axis = 1 if horizontal else 0
    base = sum(x.shape[axis] for x in cards)
    _require(base < 30, "staging budget exhausted")
    gap = int(rng.integers(1, min(4, 30 - base) + 1))
    size_axis = base + gap
    pre = int(rng.integers(30 - size_axis + 1))
    post = int(rng.integers(min(2, 30 - size_axis - pre) + 1))
    other = 1 - axis
    origins = [int(rng.integers(30 - x.shape[other] + 1)) for x in cards]
    max_other = max(o + x.shape[other] for o, x in zip(origins, cards))
    # Trim some unused exterior margin, retaining independently displaced cards.
    extra = int(rng.integers(min(2, 30 - max_other) + 1))
    shape = [0, 0]
    shape[axis], shape[other] = size_axis + pre + post, max_other + extra
    g = np.zeros(tuple(shape), dtype=np.int8)
    along = pre
    for card, across in zip(cards, origins):
        r, c = (along, across) if axis == 0 else (across, along)
        g[r:r + card.shape[0], c:c + card.shape[1]] = card
        along += card.shape[axis] + gap
    return g


def _model_diff(info, expected, **kwargs):
    try:
        return not np.array_equal(_assemble(info, **kwargs), expected)
    except ValueError:
        return True


def _make_pair(rng, params, lesson, is_test, index):
    for attempt in range(128):
        try:
            if lesson == 0:
                na = nb = 3
                offset = 0
            elif lesson == 1:
                na = nb = 4
                offset = 1
            else:
                na, nb = map(int, rng.integers(2, 5, size=2))
                offset = int(rng.choice([0, 1]))
            complex_route = is_test and params["shift"] == "routing"
            if complex_route:
                na = nb = int(rng.integers(3, 5))
            if is_test and params["shift"] == "offset":
                offset = -1
            la, lb = list(range(na)), list(range(offset, offset + nb))
            common = sorted(set(la) & set(lb))
            _require(bool(common), "no possible contact")
            pitch = int(rng.choice([4, 4, 5]))
            length = (3 if is_test else (2 if index % 2 == 0 else 4)) if params["shift"] == "guide" else int(rng.integers(2, 6))
            palette = (list(map(int, rng.permutation(np.arange(1, 8))))
                       if params["vary_palette"] else params["palette"])
            spare = int(rng.integers(3, 5))
            guide_x = max(la + lb) * pitch + spare + 2
            oa = _order(rng, na, complex_route=complex_route, identity=False)
            ob = _order(rng, nb, identity=complex_route or lesson == 0)
            if lesson == 0:
                oa = list(range(na))
                i, j = map(int, rng.choice(na, 2, replace=False))
                oa[i], oa[j] = oa[j], oa[i]
            if complex_route and rng.integers(2):
                oa, ob = list(range(na)), _order(rng, nb, complex_route=True)
            pa, pb = _plan(rng, na, oa), _plan(rng, nb, ob)
            base = _height(pa) + _height(pb)
            _require(base <= 29, "wire routing exceeds staging budget")
            # Allocate unused longitudinal space independently to leads, tails,
            # and lateral-run separation; cable length is not a swap counter.
            extra = int(rng.integers(min(5, 29 - base) + 1))
            for _ in range(extra):
                p = (pa, pb)[int(rng.integers(2))]
                if not p["moves"]:
                    p["straight"] += 1
                else:
                    options = ["lead", "tail"] + list(range(len(p["gaps"])))
                    option = options[int(rng.integers(len(options)))]
                    if isinstance(option, str):
                        p[option] += 1
                    else:
                        p["gaps"][option] += 1
            types_a = list(map(int, rng.integers(2, size=na)))
            types_b = list(map(int, rng.integers(2, size=nb)))
            if lesson in (0, 1):
                # Two working contacts in opposite stencil orders, plus an
                # identical-stencil stop. Both stop types occur across lessons.
                active = list(map(int, rng.choice(common, 2, replace=False)))
                off = next(k for k in common if k not in active)
                for kind, k in enumerate(active):
                    types_a[la.index(k)], types_b[lb.index(k)] = kind, 1 - kind
                types_a[la.index(off)] = types_b[lb.index(off)] = lesson
            else:
                active = [k for k in common if types_a[la.index(k)] != types_b[lb.index(k)]]
                if not active:
                    k = common[int(rng.integers(len(common)))]
                    types_b[lb.index(k)] = 1 - types_a[la.index(k)]
                    active = [k]
            a = _draw_card(rng, pa, la, pitch, spare, guide_x, length, palette, True, types_a)
            b = _draw_card(rng, pb, lb, pitch, spare, guide_x, length, palette, False, types_b)
            labels_a, labels_b = [None] * na, [None] * nb
            out_a, out_b = [None] * na, [None] * nb
            kinds_a, kinds_b = ["blank"] * na, ["blank"] * nb
            free_a, free_b = set(range(na)), set(range(nb))
            colors = palette[3:]
            for order_index, k in enumerate(active):
                ai = a["pairing"].index(la.index(k))
                bi = b["pairing"][lb.index(k)]
                color = colors[order_index % len(colors)]
                if lesson == 0:
                    mode = "forward" if order_index == 0 else "reverse"
                elif lesson == 1:
                    mode = "both" if order_index == 0 else str(rng.choice(["forward", "reverse"]))
                else:
                    mode = str(rng.choice(["forward", "reverse", "both"]))
                    color = int(rng.choice(colors))
                if mode in ("forward", "both"):
                    labels_a[ai] = color
                    kinds_a[ai] = "diamond" if mode == "both" else str(rng.choice(["solid", "diamond"]))
                if mode in ("reverse", "both"):
                    labels_b[bi], kinds_b[bi] = color, "checker"
                if lesson == 0 and order_index == 0:
                    kinds_a[ai] = "solid"
                out_a[ai] = out_b[bi] = color
                free_a.remove(ai)
                free_b.remove(bi)
            # Blocked and unpaired wires keep their own labels. They cannot
            # borrow a clue from a geometrically neighboring cable.
            for ai in sorted(free_a):
                if lesson in (0, 1) or rng.random() < .65:
                    labels_a[ai] = out_a[ai] = colors[2] if lesson == 0 else int(rng.choice(colors))
                    kinds_a[ai] = "diamond" if lesson == 0 else str(rng.choice(["solid", "diamond"]))
            for bi in sorted(free_b):
                if lesson not in (0, 1) and rng.random() < .45:
                    labels_b[bi] = out_b[bi] = int(rng.choice(colors))
                    kinds_b[bi] = "checker"
            for card, labels, kinds in ((a, labels_a, kinds_a), (b, labels_b, kinds_b)):
                for q, (color, kind) in enumerate(zip(labels, kinds)):
                    _terminal(card, q, color, kind)
            expected = _construction_output(a, b, out_a, out_b)
            staged = _stage(rng, a, b)
            rotation = (params["rotation"] + (index % 2 if params["mixed_axis"] else 0)) % 4
            pair = {"input": np.rot90(staged, rotation).copy(),
                    "output": np.rot90(expected, rotation).copy()}
            parsed = _parse(pair["input"])
            _require(np.array_equal(_assemble(parsed), pair["output"]), "construction/parser disagreement")
            if lesson == 1:
                _require(_model_diff(parsed, pair["output"], contact_mode="rank"), "rank shortcut survived lesson")
            if lesson == 0:
                _require(_model_diff(parsed, pair["output"], routing=False), "routing shortcut survived lesson")
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded circuit construction exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = dict(palette=list(map(int, rng.permutation(np.arange(1, 8)))),
                  vary_palette=bool(rng.integers(2)), mixed_axis=bool(rng.integers(2)),
                  rotation=int(rng.integers(4)), shift=str(rng.choice(["offset", "routing", "guide"])))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else None
            for _ in range(32):
                pair = _make_pair(rng, params, lesson, split == "test", index)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(info):
    a, b = info["a"], info["b"]
    contacts = _registered_contacts(info)
    potential = [(j, k) for j, x in enumerate(a["centers"]) for k, y in enumerate(b["centers"])
                 if x - a["marker"][1] == y - b["marker"][1]]
    connected = [(a["permutation"].index(j), b["permutation"][k]) for j, k in contacts]
    forward = sum(a["labels"][i] is not None and b["labels"][q] is None for i, q in connected)
    reverse = sum(a["labels"][i] is None and b["labels"][q] is not None for i, q in connected)
    both = sum(a["labels"][i] is not None and b["labels"][q] is not None for i, q in connected)
    return dict(counts=(len(a["centers"]), len(b["centers"])), length=a["length"], pitch=a["pitch"],
                offset=(b["centers"][0] - b["marker"][1] - a["centers"][0] + a["marker"][1]) // a["pitch"],
                crossings=a["crossings"] + b["crossings"], active=len(contacts),
                blocked=len(potential) - len(contacts), unmatched=sum(len(x["centers"]) for x in (a, b)) - 2 * len(potential),
                forward=forward, reverse=reverse, both=both,
                axes=info["rotation"] % 2, palette=(a["tint"], b["tint"], a["carrier"]),
                permutations=(tuple(a["permutation"]), tuple(b["permutation"])),
                pads=(tuple(a["pads"]), tuple(b["pads"])))


def validate_generated_puzzle(puzzle):
    """Validate schema and circuit semantics, including teaching counterexamples."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, lessons, evidence = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            expected = _assemble(info)
            _require(np.array_equal(out, expected), "wrong assembly, contact, trace, stencil, or crop")
            m = _inspect(info)
            _require(m["active"] > 0, "no conducting contacts")
            a, b = info["a"], info["b"]
            final_a, final_b = list(a["labels"]), list(b["labels"])
            for j, k in _registered_contacts(info):
                i, q = a["permutation"].index(j), b["permutation"][k]
                label = final_a[i] if final_a[i] is not None else final_b[q]
                final_a[i] = final_b[q] = label
            terminal_ink = 4 * sum(v is not None for v in final_a) + 5 * sum(v is not None for v in final_b)
            _require(int(np.sum((out != 0) & (out != a["carrier"]))) == terminal_ink,
                     "terminal stencil cell counts disagree")
            # The carrier conservation check is independent of label completion:
            # transparent union preserves every wire/pad pixel exactly once.
            occupied, total = set(), 0
            for card in (info["a"], info["b"]):
                ink = {(int(r) + card["placed"][0], int(c) + card["placed"][1])
                       for r, c in np.argwhere(card["card"] == card["carrier"])}
                total += len(ink)
                occupied |= ink
            carrier = info["a"]["carrier"]
            _require(int(np.sum(out == carrier)) == len(occupied) <= total, "wire union lost pixels")
            _require(not ({8, 9, info["a"]["tint"], info["b"]["tint"]} & set(map(int, np.unique(out)))),
                     "temporary sheet/guide pixels survived")
            _require(all(np.any(edge != 0) for edge in (out[0], out[-1], out[:, 0], out[:, -1])), "loose crop")
            if split == "train":
                lessons.append((info, out))
                for name in ("forward", "reverse", "both"):
                    if m[name]:
                        evidence.add(name)
                if m["blocked"]:
                    evidence.add("blocked")
                if m["unmatched"]:
                    evidence.add("unmatched")
                if "solid" in info["a"]["kinds"]:
                    evidence.add("solid")
                if "diamond" in info["a"]["kinds"]:
                    evidence.add("canonical")
                if m["crossings"]:
                    evidence.add("crossing")
                for j, x in enumerate(a["centers"]):
                    for k, y in enumerate(b["centers"]):
                        if x - a["marker"][1] == y - b["marker"][1]:
                            evidence.add(f"gate{a['pads'][j]}{b['pads'][k]}")
                for card, labels in ((a, final_a), (b, final_b)):
                    route_labels = labels if card["source"] else [labels[j] for j in card["permutation"]]
                    for i, path in enumerate(card["routes"]):
                        for j in range(i):
                            if (route_labels[i] is not None and route_labels[j] is not None and
                                    route_labels[i] != route_labels[j] and set(path) & set(card["routes"][j])):
                                evidence.add("distinct_crossing")
    _require({"forward", "reverse", "both", "blocked", "unmatched", "solid", "canonical", "crossing",
              "distinct_crossing", "gate00", "gate01", "gate10", "gate11"} <= evidence,
             "demonstrations lack a required lesson")
    for alternative in ({"contact_mode": "rank"}, {"routing": False}, {"contact_mode": "all"},
                        {"contact_mode": "none"}, {"reverse": False}, {"solid": True}):
        _require(any(_model_diff(info, out, **alternative) for info, out in lessons),
                 f"training admits shortcut {alternative}")


def self_test():
    """Bounded 42-seed survey, exhaustive small routing, and negative checks."""
    structures, shapes, palettes, signatures, metrics = set(), set(), set(), set(), []
    regimes = {"offset": 0, "routing": 0, "guide": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(_key(x[k]) for split in ("train", "test") for x in p[split] for k in ("input", "output"))
        signatures.add(signature)
        train_info, test_info = [], []
        for split in ("train", "test"):
            for pair in p[split]:
                before = pair["input"].copy()
                prediction = solve(pair["input"])
                assert np.array_equal(prediction, pair["output"]) and np.array_equal(before, pair["input"])
                m = _inspect(_parse(pair["input"]))
                (train_info if split == "train" else test_info).append(m)
                metrics.append(m)
                palettes.add(m["palette"])
                shapes.add(pair["input"].shape)
                roles = {}
                normalized = tuple(roles.setdefault(int(v), len(roles)) for v in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
        if any(m["offset"] == -1 for m in test_info) and all(m["offset"] >= 0 for m in train_info):
            regimes["offset"] += 1
        if {m["length"] for m in train_info} == {2, 4} and {m["length"] for m in test_info} == {3}:
            regimes["guide"] += 1
        def complex_permutation(perm):
            return sum(i != j for i, j in enumerate(perm)) >= 3
        if any(any(complex_permutation(q) for q in m["permutations"]) for m in test_info):
            regimes["routing"] += 1
    assert len(signatures) == 42 and len(structures) >= 180 and len(shapes) >= 35 and len(palettes) >= 35
    assert {n for m in metrics for n in m["counts"]} == {2, 3, 4}
    assert {m["pitch"] for m in metrics} == {4, 5}
    assert {m["length"] for m in metrics} == {2, 3, 4, 5}
    assert {m["axes"] for m in metrics} == {0, 1}
    assert all(regimes.values())
    assert any(a != b for a, b in (m["counts"] for m in metrics))
    assert any(m["active"] >= 3 for m in metrics) and any(m["blocked"] >= 2 for m in metrics)
    assert any(m["crossings"] == 0 for m in metrics) and max(m["crossings"] for m in metrics) >= 6
    # Trace every <=4 cable permutation, independently of the panel budget.
    rng = np.random.default_rng(4142)
    for n in (2, 3, 4):
        for order in permutations(range(n)):
            plan = _plan(rng, n, list(order))
            centers = list(range(1, 1 + 4 * n, 4))
            card = _draw_card(rng, plan, list(range(n)), 4, 3,
                              (n - 1) * 4 + 5, 2, [1, 2, 3, 4, 5, 6, 7], True, [0] * n)
            parsed = _read_card(card["card"], True)
            assert parsed["permutation"] == [order.index(i) for i in range(n)]
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4)):
        generate_puzzle(412, *counts)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    g = corrupted["test"][0]["output"]
    g[0, 0] = (int(g[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted circuit accepted")
    return {"seeds": 42, "pairs": len(metrics), "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "structural_palettes": len(palettes), "observed_shifts": regimes,
            "routing_permutations_checked": 32}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
