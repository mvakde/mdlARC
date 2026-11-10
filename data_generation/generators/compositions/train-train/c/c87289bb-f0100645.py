"""Deflector gravity: a composition of ARC c87289bb (A) and f0100645 (B).

Two differently coloured rails run along opposite borders.  Every interior
piece (maximal eight-connected component) coloured like a rail slides rigidly
toward that rail (B).  Straight one-cell-wide segments of a third colour,
parallel to the rails, are deflectors (A): a piece whose next step would enter
one slides sideways along it toward the nearer end -- measured with the
piece's whole transverse extent -- just far enough to clear it, then keeps
travelling in the displaced lane without returning.  A piece stops when its
next step would touch its rail or a piece that has already settled (B's
packing); pieces settle nearest-first.  Barriers, rails and background stay.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------

_NEIGHBOURS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components8(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.discard(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBOURS8:
                nb = (r + dr, c + dc)
                if nb in remaining:
                    remaining.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        result.append(frozenset(comp))
    result.sort(key=min)
    return result


def _rows(cells):
    return [r for r, _ in cells]


def _cols(cells):
    return [c for _, c in cells]


# ---------------------------------------------------------------------------
# reading an input grid (solver side, input only)
# ---------------------------------------------------------------------------

def _canonicalize(grid):
    """Return (canonical grid with rails at left/right, transposed flag)."""
    g = np.asarray(grid).astype(int)
    if g.ndim != 2 or min(g.shape) < 1:
        raise ValueError("grid must be a non-empty 2D array")
    vertical = (g.shape[1] >= 3 and bool(np.all(g[:, 0] == g[0, 0]))
                and bool(np.all(g[:, -1] == g[0, -1])) and g[0, 0] != g[0, -1])
    horizontal = (g.shape[0] >= 3 and bool(np.all(g[0, :] == g[0, 0]))
                  and bool(np.all(g[-1, :] == g[-1, 0])) and g[0, 0] != g[-1, 0])
    if vertical and not horizontal:
        return g.copy(), False
    if horizontal and not vertical:
        return g.T.copy(), True
    raise ValueError("input does not have exactly one pair of opposite rails")


def _read_scene(canon):
    """Decompose a canonical grid into colours, pieces and barriers."""
    height, width = canon.shape
    left, right = int(canon[0, 0]), int(canon[0, -1])
    interior = canon[:, 1:width - 1]
    values, counts = np.unique(interior, return_counts=True)
    others = [(int(n), int(v)) for v, n in zip(values, counts) if int(v) not in (left, right)]
    if others:
        others.sort(key=lambda item: (-item[0], item[1]))
        background = others[0][1]
        barrier_colors = [v for _, v in others[1:]]
    else:
        background, barrier_colors = None, []
    pieces = []
    for color, direction in ((left, -1), (right, 1)):
        cells = {(r, c) for r in range(height) for c in range(1, width - 1)
                 if int(canon[r, c]) == color}
        for comp in _components8(cells):
            pieces.append((comp, direction))
    barrier_cells = {(r, c) for r in range(height) for c in range(1, width - 1)
                     if int(canon[r, c]) in barrier_colors}
    barriers = _components8(barrier_cells)
    barrier_color = barrier_colors[0] if len(barrier_colors) == 1 else None
    return {"height": height, "width": width, "left": left, "right": right,
            "background": background, "barrier_color": barrier_color,
            "barrier_colors": barrier_colors, "pieces": pieces, "barriers": barriers}


# ---------------------------------------------------------------------------
# the rule: sequential nearest-first settling with nearest-end deflection
# ---------------------------------------------------------------------------

def _lead(cells, direction, width):
    """Free cells between a piece's leading edge and its own rail."""
    if direction < 0:
        return min(_cols(cells)) - 1
    return width - 2 - max(_cols(cells))


def _deflection(cells, top, bottom):
    """Minimal lateral move clearing barrier rows [top, bottom] with the whole piece."""
    rows = _rows(cells)
    up = max(rows) - top + 1          # move toward smaller row index
    down = bottom - min(rows) + 1     # move toward larger row index
    if up < down:
        return -1, up, False
    if down < up:
        return 1, down, False
    return -1, up, True               # exact tie (never generated)


def _contact_choice(contact_rows, top, bottom):
    """A-style single-lane choice using only the contacting rows (a shortcut)."""
    twice_center = top + bottom
    if all(2 * r < twice_center for r in contact_rows):
        return -1
    if all(2 * r > twice_center for r in contact_rows):
        return 1
    return 0


def _settle(height, width, pieces, barriers, strict=False, variant="rule"):
    """Simulate the rule.  Returns None for an invalid scene when strict.

    variant: "rule" (the task rule), "contact" (deflect by contact rows only),
    "stop" (barriers stop pieces), "alone" (no packing between pieces).
    """
    bcell = {}
    spans = []
    for k, b in enumerate(barriers):
        rows = _rows(b)
        spans.append((min(rows), max(rows)))
        for cell in b:
            bcell[cell] = k
    n = len(pieces)
    order = sorted(range(n), key=lambda i: (_lead(pieces[i][0], pieces[i][1], width),
                                            min(pieces[i][0])))
    occupied = {}
    records = [None] * n
    for i in order:
        cells, direction = pieces[i]
        cells = frozenset(cells)
        traj = set(cells)
        events = []
        forward = 0
        stop, blockers = "rail", set()
        guard = 0
        while True:
            guard += 1
            if guard > 4 * (height + width) + 20:
                if strict:
                    return None
                break
            nxt = _shift(cells, 0, direction)
            rail = any(c <= 0 or c >= width - 1 for _, c in nxt)
            hit_p = {occupied[x] for x in nxt if x in occupied} if variant != "alone" else set()
            hit_b = {bcell[x] for x in nxt if x in bcell}
            if variant == "stop" and hit_b:
                stop = "barrier"
                break
            if rail or hit_p:
                if strict and hit_b:
                    return None
                stop, blockers = ("rail" if rail else "piece"), hit_p
                break
            if hit_b:
                if strict and len(hit_b) != 1:
                    return None
                top = min(spans[k][0] for k in hit_b)
                bottom = max(spans[k][1] for k in hit_b)
                lateral, amount, tie = _deflection(cells, top, bottom)
                if strict and tie:
                    return None
                contact = sorted({r for r, c in cells if (r, c + direction) in bcell})
                alt = _contact_choice(contact, top, bottom)
                alt_ok = False
                if alt and alt != lateral:
                    rows = _rows(cells)
                    alt_amount = max(rows) - top + 1 if alt < 0 else bottom - min(rows) + 1
                    probe, alt_ok = cells, True
                    for _ in range(alt_amount):
                        probe = _shift(probe, alt, 0)
                        if (any(r < 0 or r >= height for r, _ in probe)
                                or any(x in bcell or x in occupied for x in probe)):
                            alt_ok = False
                            break
                if variant == "contact":
                    choice = _contact_choice(contact, top, bottom)
                    if choice and choice != lateral:
                        lateral = choice
                        rows = _rows(cells)
                        amount = max(rows) - top + 1 if choice < 0 else bottom - min(rows) + 1
                blocked = False
                for _ in range(amount):
                    side = _shift(cells, lateral, 0)
                    if (any(r < 0 or r >= height for r, _ in side)
                            or any(x in bcell or x in occupied for x in side)):
                        blocked = True
                        break
                    cells = side
                    traj.update(cells)
                if blocked:
                    if strict:
                        return None
                    stop = "jammed"
                    break
                room = (top, height - 1 - bottom)
                tight = (room[0] < room[1]) if lateral < 0 else (room[1] < room[0])
                events.append({"barrier": min(hit_b), "lateral": lateral, "amount": amount,
                               "contact": tuple(contact), "span": (top, bottom),
                               "direction": direction, "forward_before": forward,
                               "alt_ok": alt_ok, "tight": tight})
                continue
            cells = nxt
            traj.update(cells)
            forward += 1
        if variant != "alone":
            for x in cells:
                occupied[x] = i
        records[i] = {"final": cells, "traj": frozenset(traj), "events": events,
                      "stop": stop, "blockers": frozenset(blockers),
                      "direction": direction, "forward": forward}
    return {"order": order, "records": records}


def _tick_settle(height, width, pieces, barriers):
    """Simultaneous-motion reading of the same rule; None when it is ambiguous."""
    bcell = {}
    spans = []
    for k, b in enumerate(barriers):
        rows = _rows(b)
        spans.append((min(rows), max(rows)))
        for cell in b:
            bcell[cell] = k
    n = len(pieces)
    cells = [frozenset(p[0]) for p in pieces]
    direction = [p[1] for p in pieces]
    mode = ["f"] * n
    lateral = [0] * n
    remaining = [0] * n
    done = [False] * n
    for _ in range(6 * (height + width) + 20):
        live = [i for i in range(n) if not done[i]]
        if not live:
            return cells
        owner = {}
        for i in range(n):
            for x in cells[i]:
                owner[x] = i
        moves, settle = {}, []
        for i in live:
            if mode[i] == "f":
                nxt = _shift(cells[i], 0, direction[i])
                rail = any(c <= 0 or c >= width - 1 for _, c in nxt)
                hit_b = {bcell[x] for x in nxt if x in bcell}
                others = {owner[x] for x in nxt if x in owner and owner[x] != i}
                hit_settled = any(done[j] for j in others)
                moving = {j for j in others if not done[j]}
                if rail or hit_settled:
                    if hit_b:
                        return None
                    settle.append(i)
                    continue
                if hit_b:
                    if moving or len(hit_b) != 1:
                        return None
                    k = next(iter(hit_b))
                    lat, amount, tie = _deflection(cells[i], *spans[k])
                    if tie:
                        return None
                    mode[i], lateral[i], remaining[i] = "l", lat, amount
                elif moving:
                    continue
                else:
                    moves[i] = nxt
                    continue
            nxt = _shift(cells[i], lateral[i], 0)
            if (any(r < 0 or r >= height for r, _ in nxt) or any(x in bcell for x in nxt)
                    or any(x in owner and owner[x] != i for x in nxt)):
                return None
            moves[i] = nxt
        claimed = set()
        for i, nxt in moves.items():
            if claimed & nxt:
                return None
            claimed |= nxt
        if not moves and not settle:
            return None
        for i in settle:
            done[i] = True
        for i, nxt in moves.items():
            cells[i] = nxt
            if mode[i] == "l":
                remaining[i] -= 1
                if remaining[i] == 0:
                    mode[i] = "f"
    return None


def _analyze(height, width, pieces, barriers):
    """Strict analysis used by the generator/validator; None if ambiguous."""
    result = _settle(height, width, pieces, barriers, strict=True)
    if result is None:
        return None
    order, records = result["order"], result["records"]
    position = {i: k for k, i in enumerate(order)}
    n = len(pieces)
    for i in range(n):
        rec = records[i]
        if rec["final"] == frozenset(pieces[i][0]):
            return None
        for j in range(n):
            if j == i:
                continue
            if position[j] > position[i] and rec["traj"] & pieces[j][0]:
                return None
            if pieces[j][1] != pieces[i][1] and rec["traj"] & records[j]["traj"]:
                return None
            if (pieces[j][1] == pieces[i][1] and j > i
                    and _lead(pieces[i][0], pieces[i][1], width)
                    == _lead(pieces[j][0], pieces[j][1], width)
                    and (set(_rows(pieces[i][0])) & set(_rows(pieces[j][0]))
                         or rec["traj"] & records[j]["traj"])):
                return None    # equally near pieces must be fully independent
    ticked = _tick_settle(height, width, pieces, barriers)
    if ticked is None or any(ticked[i] != records[i]["final"] for i in range(n)):
        return None
    return result


def _features(height, width, pieces, barriers, result):
    records = result["records"]
    events = [e for rec in records for e in rec["events"]]
    turns = {e["lateral"] * e["direction"] for e in events}
    laterals = {e["lateral"] for e in events}
    extent = any(e["alt_ok"] for e in events)
    tight = any(e["tight"] for e in events)
    stack = any(rec["blockers"] for rec in records)
    interaction = any(rec["blockers"] and (rec["events"] or any(records[j]["events"]
                                                                 for j in rec["blockers"]))
                      for rec in records)
    cascade = any(len(rec["events"]) >= 2 for rec in records)
    far = False
    for cells, direction in pieces:
        own = _lead(cells, direction, width)
        other = _lead(cells, -direction, width)
        if own > other:
            far = True
    hit = {e["barrier"] for e in events}
    both_sides_barrier = any(len({e["direction"] for e in events if e["barrier"] == k}) == 2
                             for k in hit)
    return {"deflections": len(events), "turns": turns, "laterals": laterals,
            "extent": extent, "stack": stack, "interaction": interaction,
            "cascade": cascade, "far_rail": far, "tight": tight, "unhit": len(hit) < len(barriers),
            "both_sides_barrier": both_sides_barrier,
            "pieces": len(pieces), "barriers": len(barriers),
            "left_pieces": sum(d < 0 for _, d in pieces),
            "right_pieces": sum(d > 0 for _, d in pieces)}


def _render(height, width, colors, barriers, pieces_cells):
    background, left, right, barrier = colors
    grid = np.full((height, width), background, dtype=int)
    grid[:, 0] = left
    grid[:, -1] = right
    for b in barriers:
        for r, c in b:
            grid[r, c] = barrier
    for cells, direction in pieces_cells:
        for r, c in cells:
            grid[r, c] = left if direction < 0 else right
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    canon, transposed = _canonicalize(input_grid)
    scene = _read_scene(canon)
    height, width = scene["height"], scene["width"]
    result = _settle(height, width, scene["pieces"], scene["barriers"], strict=False)
    output = canon.copy()
    background = scene["background"] if scene["background"] is not None else 0
    for cells, _ in scene["pieces"]:
        for r, c in cells:
            output[r, c] = background
    for (cells, direction), rec in zip(scene["pieces"], result["records"]):
        color = scene["left"] if direction < 0 else scene["right"]
        for r, c in rec["final"]:
            output[r, c] = color
    return output.T.copy() if transposed else output


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _normalize(cells):
    top = min(_rows(cells))
    left = min(_cols(cells))
    return frozenset((r - top, c - left) for r, c in cells)


_ARCHETYPES = tuple(_normalize(s) for s in (
    {(0, 0), (0, 1), (0, 2)},                       # guide-like bar along motion
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (1, 0), (2, 0)},                       # bar across motion
    {(0, 0), (1, 0), (2, 0), (3, 0)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},               # L
    {(0, 1), (1, 1), (2, 1), (2, 0), (3, 0)},       # stair
    {(0, 0), (1, 1), (2, 2)},                       # diagonal
    {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},       # tall foot
    {(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)},       # tall flag
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)},       # C
    {(0, 0), (1, 0), (1, 1), (2, 0)},               # T
    {(0, 0), (0, 1), (1, 0), (1, 1)},               # square
))


_SLIM = tuple(_normalize(s) for s in (
    {(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (0, 1), (0, 2)}, {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}, {(0, 0), (1, 0)}, {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 1)}, {(0, 0), (1, 1), (2, 2)}, {(0, 0), (0, 1), (1, 2)},
))


def _random_shape(rng, tall=False, style="mixed", max_h=5, max_w=4):
    """Random eight-connected silhouette; `style` is a puzzle-level vocabulary bias."""
    for _ in range(30):
        roll = rng.random()
        if style == "slim" and not tall and roll < 0.65:
            shape = _SLIM[int(rng.integers(len(_SLIM)))]
        elif roll < 0.3:
            shape = _ARCHETYPES[int(rng.integers(len(_ARCHETYPES)))]
        else:
            low = 3 if tall else (4 if style == "chunky" else 1)
            high = 9 if style == "chunky" else 8
            size = int(rng.integers(low, high))
            cells = {(0, 0)}
            diagonal = rng.random() < 0.35
            while len(cells) < size:
                base = sorted(cells)[int(rng.integers(len(cells)))]
                if diagonal:
                    dr, dc = _NEIGHBOURS8[int(rng.integers(8))]
                else:
                    dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                if tall and rng.random() < 0.4:
                    dr, dc = (1, 0)
                cells.add((base[0] + dr, base[1] + dc))
            shape = _normalize(cells)
        if rng.random() < 0.5:
            shape = _normalize({(r, -c) for r, c in shape})
        if rng.random() < 0.5:
            shape = _normalize({(-r, c) for r, c in shape})
        if not tall and rng.random() < 0.3:
            shape = _normalize({(c, r) for r, c in shape})
        h = 1 + max(_rows(shape))
        w = 1 + max(_cols(shape))
        if h <= max_h and w <= max_w and (not tall or h >= 3):
            return shape
    return frozenset({(0, 0), (1, 0), (2, 0)}) if tall else frozenset({(0, 0)})


class _Scene:
    def __init__(self, height, width, style="mixed"):
        self.height, self.width, self.style = height, width, style
        self.barriers = []
        self.pieces = []
        self.result = None

    def blocked(self):
        cells = set()
        for b in self.barriers:
            cells |= _halo(b)
        for p, _ in self.pieces:
            cells |= _halo(p)
        return cells

    def piece_fits(self, cells):
        return (all(0 <= r < self.height and 2 <= c <= self.width - 3 for r, c in cells)
                and not (set(cells) & self.blocked()))

    def barrier_fits(self, cells):
        return (all(0 <= r < self.height and 3 <= c <= self.width - 4 for r, c in cells)
                and not (set(cells) & self.blocked()))

    def analyze(self):
        return _analyze(self.height, self.width, self.pieces, self.barriers)

    def try_add_piece(self, cells, direction, accept=None):
        if not self.piece_fits(cells):
            return False
        self.pieces.append((frozenset(cells), direction))
        result = self.analyze()
        if result is None or (accept is not None and not accept(result, len(self.pieces) - 1)):
            self.pieces.pop()
            return False
        self.result = result
        return True

    def try_add_barrier(self, cells, accept=None):
        if not self.barrier_fits(cells):
            return False
        self.barriers.append(frozenset(cells))
        result = self.analyze() if self.pieces else {"order": [], "records": []}
        if result is None or (accept is not None and not accept(result, len(self.barriers) - 1)):
            self.barriers.pop()
            return False
        self.result = result
        return True

    def features(self):
        return _features(self.height, self.width, self.pieces, self.barriers, self.result)


def _random_barrier(rng, scene, length_cap=6):
    h, w = scene.height, scene.width
    length = int(rng.integers(2, max(2, min(length_cap, h - 3)) + 1))
    col = int(rng.integers(3, w - 3))
    top = int(rng.integers(0, h - length + 1))
    return frozenset((top + k, col) for k in range(length))


def _extent_offsets(shape, direction, top, bottom, height, rmin, rmax):
    """Row offsets where contact rows and full extent pick opposite, feasible sides."""
    found = []
    for r0 in range(rmin, rmax + 1):
        cells = [(r0 + r, c) for r, c in shape]
        inside = [(r, c) for r, c in cells if top <= r <= bottom]
        if not inside:
            continue
        lead = max(c * direction for _, c in inside)
        contact = [r for r, c in inside if c * direction == lead]
        lateral, amount, tie = _deflection(cells, top, bottom)
        alt = _contact_choice(contact, top, bottom)
        if tie or alt in (0, lateral):
            continue
        rows = _rows(cells)
        alt_amount = max(rows) - top + 1 if alt < 0 else bottom - min(rows) + 1
        if not (0 <= min(rows) + lateral * amount and max(rows) + lateral * amount < height):
            continue
        if not (0 <= min(rows) + alt * alt_amount and max(rows) + alt * alt_amount < height):
            continue
        found.append(r0)
    return found


def _aimed_piece(rng, scene, barrier, direction, tall=False):
    """A piece upstream of `barrier` whose rows overlap it (an extent witness if tall)."""
    rows = _rows(barrier)
    bc = barrier and next(iter(barrier))[1]
    top, bottom = min(rows), max(rows)
    shape = _random_shape(rng, tall=tall, style=scene.style)
    sh = 1 + max(_rows(shape))
    sw = 1 + max(_cols(shape))
    if direction > 0:
        cmin, cmax = 2, bc - 2 - (sw - 1)
    else:
        cmin, cmax = bc + 2, scene.width - 3 - (sw - 1)
    if cmin > cmax:
        return None
    rmin = max(0, top - (sh - 1))
    rmax = min(scene.height - sh, bottom)
    if rmin > rmax:
        return None
    if tall:
        offsets = _extent_offsets(shape, direction, top, bottom, scene.height, rmin, rmax)
        if not offsets:
            return None
        r0 = offsets[int(rng.integers(len(offsets)))]
    else:
        r0 = int(rng.integers(rmin, rmax + 1))
    c0 = int(rng.integers(cmin, cmax + 1))
    cells = _shift(shape, r0, c0)
    if not any(top <= r <= bottom for r, _ in cells):
        return None
    return cells


def _lane_piece(rng, scene, lane_rows, direction):
    """A piece anywhere across the grid whose rows overlap `lane_rows`."""
    shape = _random_shape(rng, style=scene.style)
    sh = 1 + max(_rows(shape))
    sw = 1 + max(_cols(shape))
    lo, hi = min(lane_rows), max(lane_rows)
    rmin = max(0, lo - (sh - 1))
    rmax = min(scene.height - sh, hi)
    if rmin > rmax or scene.width - 3 - (sw - 1) < 2:
        return None
    r0 = int(rng.integers(rmin, rmax + 1))
    c0 = int(rng.integers(2, scene.width - 3 - (sw - 1) + 1))
    return _shift(shape, r0, c0)


def _free_piece(rng, scene):
    shape = _random_shape(rng, style=scene.style)
    sh = 1 + max(_rows(shape))
    sw = 1 + max(_cols(shape))
    if scene.height - sh < 0 or scene.width - 3 - (sw - 1) < 2:
        return None
    r0 = int(rng.integers(0, scene.height - sh + 1))
    c0 = int(rng.integers(2, scene.width - 3 - (sw - 1) + 1))
    return _shift(shape, r0, c0)


def _deflects(index, barrier=None):
    def accept(result, _):
        events = result["records"][index]["events"]
        return bool(events) and (barrier is None or events[0]["barrier"] == barrier)
    return accept


def _place_deflector(rng, scene, attempts=40, tall=False, accept_extra=None, pre=None,
                     direction=None):
    """Add a piece aimed at a barrier.  `pre` is a cheap filter on the piece's
    behaviour when simulated alone; the strict whole-scene analysis decides."""
    fixed = direction
    for _ in range(attempts):
        if not scene.barriers:
            return None
        k = int(rng.integers(len(scene.barriers)))
        direction = fixed if fixed is not None else int(rng.choice((-1, 1)))
        cells = _aimed_piece(rng, scene, scene.barriers[k], direction, tall=tall)
        if cells is None or not scene.piece_fits(cells):
            continue
        alone = _settle(scene.height, scene.width, [(cells, direction)], scene.barriers)
        events = alone["records"][0]["events"]
        if not events or events[0]["barrier"] != k or (pre is not None and not pre(events)):
            continue
        index = len(scene.pieces)

        def accept(result, _, index=index, k=k):
            if not _deflects(index, k)(result, None):
                return False
            return accept_extra is None or accept_extra(result, index)
        if scene.try_add_piece(cells, direction, accept):
            return index
    return None


def _build_scene(rng, height, width, mission, regime, style="mixed"):
    scene = _Scene(height, width, style)
    n_barriers = int(rng.integers(1, 4)) if regime == "train" else int(rng.integers(2, 5))
    for _ in range(n_barriers):
        for _ in range(12):
            if scene.try_add_barrier(_random_barrier(rng, scene)):
                break
    if not scene.barriers:
        return None

    # core deflection (optionally an extent witness)
    def extent_event(result, index):
        return any(e["alt_ok"] for e in result["records"][index]["events"])
    core = _place_deflector(rng, scene, attempts=80, tall="extent" in mission,
                            accept_extra=extent_event if "extent" in mission else None,
                            pre=(lambda ev: any(e["alt_ok"] for e in ev))
                            if "extent" in mission else None)
    if core is None:
        return None

    if "cascade" in mission:
        rec = scene.result["records"][core]
        direction = scene.pieces[core][1]
        lane = _rows(rec["final"])
        bc = next(iter(scene.barriers[rec["events"][0]["barrier"]]))[1]
        ok = False
        for _ in range(40):
            length = int(rng.integers(2, max(2, min(6, height - 3)) + 1))
            lo, hi = min(lane), max(lane)
            top = int(rng.integers(max(0, lo - length + 1), min(height - length, hi) + 1))
            if direction > 0:
                if bc + 2 > width - 4:
                    break
                col = int(rng.integers(bc + 2, width - 3))
            else:
                if 3 > bc - 2:
                    break
                col = int(rng.integers(3, bc - 1))
            cells = frozenset((top + j, col) for j in range(length))

            def accept(result, _, core=core):
                return len(result["records"][core]["events"]) >= 2
            if scene.try_add_barrier(cells, accept):
                ok = True
                break
        if not ok:
            return None

    if "interaction" in mission:
        ok = False
        for _ in range(60):
            rec = scene.result["records"][core]
            direction = scene.pieces[core][1]
            cells = _lane_piece(rng, scene, _rows(rec["final"]), direction)
            if cells is None:
                continue

            def accept(result, _):
                return _features(height, width, scene.pieces, scene.barriers,
                                 result)["interaction"]
            if scene.try_add_piece(cells, direction, accept):
                ok = True
                break
        if not ok:
            return None

    if "turns" in mission:
        def two_turns(result, _):
            f = _features(height, width, scene.pieces, scene.barriers, result)
            return len(f["turns"]) == 2 and len(f["laterals"]) == 2
        if not two_turns(scene.result, None):
            first = scene.result["records"][core]["events"][0]
            need = -first["lateral"]
            if _place_deflector(rng, scene, attempts=80, accept_extra=two_turns,
                                pre=lambda ev: any(e["lateral"] == need for e in ev),
                                direction=first["direction"]) is None:
                return None

    # extra pieces and distractors
    if regime == "train":
        target = int(rng.integers(len(scene.pieces) + 1, max(len(scene.pieces) + 2, 7)))
    else:
        target = int(rng.integers(max(len(scene.pieces) + 2, 5), 10))
    tries = 0
    while len(scene.pieces) < target and tries < 40:
        tries += 1
        roll = rng.random()
        if roll < 0.35:
            _place_deflector(rng, scene, attempts=3)
            continue
        if roll < 0.6 and scene.pieces:
            j = int(rng.integers(len(scene.pieces)))
            cells = _lane_piece(rng, scene, _rows(scene.pieces[j][0]), scene.pieces[j][1])
            if cells is not None:
                scene.try_add_piece(cells, scene.pieces[j][1])
            continue
        cells = _free_piece(rng, scene)
        if cells is not None:
            scene.try_add_piece(cells, int(rng.choice((-1, 1))))
    # both rail identities must have interior pieces
    for direction in (-1, 1):
        if not any(d == direction for _, d in scene.pieces):
            for _ in range(30):
                cells = _free_piece(rng, scene)
                if cells is not None and scene.try_add_piece(cells, direction):
                    break
            else:
                return None
    if regime == "train" and rng.random() < 0.3:
        for _ in range(6):
            if scene.try_add_barrier(_random_barrier(rng, scene, 4)):
                break
    feats = scene.features()
    for need in mission:
        if need == "turns":
            if len(feats["turns"]) < 2 or len(feats["laterals"]) < 2:
                return None
        elif need.startswith("!"):
            if feats[need[1:]]:
                return None
        elif not feats[need]:
            return None
    if feats["deflections"] < 1:
        return None
    return scene


def _scene_grids(scene, colors, transposed):
    pieces_in = list(scene.pieces)
    pieces_out = [(rec["final"], d) for rec, (_, d) in zip(scene.result["records"], scene.pieces)]
    a = _render(scene.height, scene.width, colors, scene.barriers, pieces_in)
    b = _render(scene.height, scene.width, colors, scene.barriers, pieces_out)
    if transposed:
        return a.T.copy(), b.T.copy()
    return a, b


def _dimensions(rng, regime):
    if regime == "train":
        return int(rng.integers(8, 17)), int(rng.integers(10, 19))
    return int(rng.integers(12, 23)), int(rng.integers(15, 26))


def _generate_pair(rng, mission, regime, colors, transposed, style):
    for attempt in range(400):
        height, width = _dimensions(rng, regime)
        if attempt > 150:
            height, width = height + 3, width + 3
        scene = _build_scene(rng, min(height, 28), min(width, 28), mission, regime, style)
        if scene is not None:
            return _scene_grids(scene, colors, transposed)
    raise RuntimeError(f"could not build a {regime} pair for mission {sorted(mission)}")


def _check_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("pair counts must be integers")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("need 2 <= num_train <= 10, 1 <= num_test <= 4, num_train > num_test")
    return num_train, num_test


def _train_missions(rng, num_train, cascade):
    required = [{"interaction"}, {"extent"}, {"turns"}]
    if cascade:
        required.append({"cascade"})
    missions = [set() for _ in range(num_train)]
    for k, need in enumerate(required):
        missions[k % num_train] |= need
    for k in range(len(required), num_train):
        if rng.random() < 0.3:
            missions[k] |= {("interaction", "extent", "turns")[int(rng.integers(3))]}
    order = rng.permutation(num_train)
    return [missions[int(k)] for k in order]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(12):
        puzzle = _attempt_puzzle(rng, num_train, num_test)
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not build a valid puzzle")


def _attempt_puzzle(rng, num_train, num_test):
    style = ("mixed", "mixed", "slim", "chunky")[int(rng.integers(4))]
    fixed_palette = rng.random() < 0.35
    fixed_orientation = rng.random() < 0.4
    base_transposed = bool(rng.integers(2))
    base_colors = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    cascade = rng.random() < 0.5
    train_missions = _train_missions(rng, num_train, cascade)
    test_missions = []
    for _ in range(num_test):
        mission = {"interaction"}
        if cascade and rng.random() < 0.6:
            mission.add("cascade")
        elif not cascade:
            mission.add("!cascade")
        if rng.random() < 0.4:
            mission.add("extent")
        test_missions.append(mission)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_orientations = []
    for split, missions in (("train", train_missions), ("test", test_missions)):
        for mission in missions:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2 ** 62)))
            for _ in range(20):
                if fixed_palette:
                    colors = base_colors
                else:
                    colors = tuple(int(x) for x in pair_rng.choice(10, size=4, replace=False))
                if fixed_orientation:
                    transposed = base_transposed
                elif split == "train":
                    transposed = bool(pair_rng.integers(2))
                else:   # a test orientation is always demonstrated in training
                    transposed = train_orientations[int(pair_rng.integers(len(train_orientations)))]
                a, b = _generate_pair(pair_rng, mission, split, colors, transposed, style)
                key = (a.shape, a.tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not build distinct inputs")
            seen.add(key)
            if split == "train":
                train_orientations.append(transposed)
            puzzle[split].append({"input": a, "output": b})
    return puzzle


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pair_analysis(grid):
    """Strictly analyse one generated input; returns (scene, result, features)."""
    canon, transposed = _canonicalize(grid)
    scene = _read_scene(canon)
    height, width = scene["height"], scene["width"]
    _require(scene["background"] is not None and scene["barrier_color"] is not None,
             "input needs a background and exactly one barrier colour")
    colors = {scene["background"], scene["left"], scene["right"], scene["barrier_color"]}
    _require(len(colors) == 4, "four distinct colour roles")
    _require(set(int(v) for v in np.unique(canon)) == colors, "no stray colours")
    pieces, barriers = scene["pieces"], scene["barriers"]
    _require(any(d < 0 for _, d in pieces) and any(d > 0 for _, d in pieces),
             "both rail identities need interior pieces")
    for b in barriers:
        _require(len({c for _, c in b}) == 1 and len(b) >= 2, "barriers are straight rail-parallel segments")
        rows = sorted(_rows(b))
        _require(rows == list(range(rows[0], rows[-1] + 1)), "barrier segments are contiguous")
        _require(3 <= next(iter(b))[1] <= width - 4, "barrier away from rails")
    objects = [p for p, _ in pieces] + list(barriers)
    for k, first in enumerate(objects):
        for second in objects[k + 1:]:
            _require(not (_halo(first) & second), "input objects must not touch")
    for p, _ in pieces:
        _require(all(2 <= c <= width - 3 for _, c in p), "pieces start away from rails")
    result = _analyze(height, width, pieces, barriers)
    _require(result is not None, "scene is ambiguous under the rule")
    return scene, result, _features(height, width, pieces, barriers, result), transposed


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "pair counts")
    seen = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a = np.asarray(pair["input"])
            b = np.asarray(pair["output"])
            for g in (a, b):
                _require(g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "grid size")
                _require(np.issubdtype(g.dtype, np.integer), "integer grid")
                _require(bool(np.all((g >= 0) & (g <= 9))), "colours 0..9")
            _require(a.shape == b.shape and not np.array_equal(a, b), "shape kept and change present")
            key = (a.shape, a.astype(int).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene, result, f, transposed = _pair_analysis(a)
            _require(f["deflections"] >= 1, "every pair shows a deflection")
            _require(np.array_equal(solve(a), b), "output differs from the rule")
            # invariants: rails, barriers kept; each piece's shape and colour kept
            canon_b = b.T if transposed else b
            _require(np.array_equal(canon_b[:, 0], np.asarray(a.T if transposed else a)[:, 0]),
                     "left rail kept")
            for bar in scene["barriers"]:
                _require(all(canon_b[r, c] == scene["barrier_color"] for r, c in bar),
                         "barrier kept")
            for (cells, d), rec in zip(scene["pieces"], result["records"]):
                _require(_normalize(cells) == _normalize(rec["final"]), "rigid translation")
            f["transposed"] = transposed
            feats[split].append(f)
    train = feats["train"]
    _require(any(f["interaction"] for f in train), "training shows deflection-dependent packing")
    _require(any(f["extent"] for f in train), "training shows full-extent nearest-end choice")
    _require(set().union(*(f["turns"] for f in train)) == {-1, 1}, "training shows both turn senses")
    _require(set().union(*(f["laterals"] for f in train)) == {-1, 1}, "training shows both lateral sides")
    _require(any(f["far_rail"] for f in train), "training shows colour, not proximity, choosing the rail")
    _require(any(f["tight"] for f in train), "training shows a deflection toward the tighter side")
    for f in feats["test"]:
        _require(f["interaction"], "each test needs deflection-dependent packing")
        _require(any(t["transposed"] == f["transposed"] for t in train),
                 "test rail orientation is demonstrated in training")
    if any(f["cascade"] for f in feats["test"]):
        _require(any(f["cascade"] for f in train), "test cascade needs a training cascade")


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def _alt_output(grid, variant):
    canon, transposed = _canonicalize(grid)
    scene = _read_scene(canon)
    result = _settle(scene["height"], scene["width"], scene["pieces"], scene["barriers"],
                     strict=False, variant=variant)
    out = canon.copy()
    for cells, _ in scene["pieces"]:
        for r, c in cells:
            out[r, c] = scene["background"]
    for (cells, d), rec in zip(scene["pieces"], result["records"]):
        for r, c in rec["final"]:
            out[r, c] = scene["left"] if d < 0 else scene["right"]
    return out.T if transposed else out


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, palettes, orientations = set(), set(), set()
    counts = {"cascade": 0, "unhit": 0, "both_sides_barrier": 0, "stack": 0}
    piece_counts, barrier_counts = set(), set()
    mixed_orientation = mixed_palette = 0
    distinguishes = {"contact": 0, "stop": 0, "alone": 0}
    for p in puzzles:
        orients, pals = set(), set()
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                shapes.add(a.shape)
                scene, result, f, transposed = _pair_analysis(a)
                orients.add(transposed)
                pals.add((scene["background"], scene["left"], scene["right"], scene["barrier_color"]))
                for key in counts:
                    counts[key] += bool(f[key])
                piece_counts.add(f["pieces"])
                barrier_counts.add(f["barriers"])
        orientations |= orients
        palettes |= pals
        mixed_orientation += len(orients) > 1
        mixed_palette += len(pals) > 1
        for variant in distinguishes:
            if any(not np.array_equal(_alt_output(pair["input"], variant), pair["output"])
                   for pair in p["train"]):
                distinguishes[variant] += 1
    _require(len({(tuple(map(tuple, p["test"][0]["input"])),) for p in puzzles}) == 42, "seed variety")
    _require(len(shapes) >= 40, "grid size diversity")
    _require(orientations == {False, True}, "both rail orientations")
    _require(mixed_orientation >= 10 and mixed_palette >= 10, "within-puzzle variation")
    _require(len(palettes) >= 40, "palette diversity")
    _require(counts["cascade"] >= 10 and counts["unhit"] >= 10 and counts["stack"] >= 40,
             "structural feature diversity")
    _require(counts["both_sides_barrier"] >= 3, "barriers deflecting both colours")
    _require(len(piece_counts) >= 6 and len(barrier_counts) >= 4, "count diversity")
    _require(all(v == 42 for v in distinguishes.values()), "training refutes shortcuts")
    # determinism and explicit counts
    _require(all(np.array_equal(x["input"], y["input"]) for x, y in
                 zip(generate_puzzle(5, 3, 2)["train"], generate_puzzle(5, 3, 2)["train"])),
             "determinism")
    for a, b in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(11, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
