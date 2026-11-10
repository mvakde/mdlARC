"""Repair-controlled stencil networks. Only NumPy and the standard library.

Source widgets visibly encode every stencil. Diagonal wire deviations repair
three-cell breaks; four-connected reachability then determines which endpoint
printers receive a single, unambiguous stencil identity. Nothing is learned or
remembered by solve(). See task_spec.md for the complete input grammar.
"""
from __future__ import annotations

import copy
import json

import numpy as np


_ORTH = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not ((0 <= a) & (a <= 9)).all()):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _at(a, r, c):
    return 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and bool(a[r, c])


def _components(mask):
    """Four-connected components, independent of color or generation records."""
    labels = np.full(mask.shape, -1, dtype=int)
    objects = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(objects)
        stack, cells = [(r, c)], []
        labels[r, c] = index
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in _ORTH:
                q = p[0] + dr, p[1] + dc
                if _inside(mask.shape, q) and mask[q] and labels[q] < 0:
                    labels[q] = index
                    stack.append(q)
        objects.append(cells)
    return labels, objects


def _straight_support(mask):
    strong = np.zeros_like(mask)
    for m, s in ((mask, strong), (mask.T, strong.T)):
        if m.shape[1] < 3:
            continue
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            s[:, k:k + triples.shape[1]] |= triples
    return strong


def _repair_wires(wire):
    """Use original wire evidence, then erase weak diagonal-only material.

    A match contains a bracketed three-cell gap, straight continuation outside
    a bracket, and exactly two weak diagonal cells bending into that gap.
    No receiver, source frame, or colored stencil is treated as wire evidence.
    """
    strong = _straight_support(wire)
    added, protected = np.zeros_like(wire), np.zeros_like(wire)
    evidence = []
    for dr, dc in ((0, 1), (1, 0)):
        m = wire if dc else wire.T
        if m.shape[1] < 5:
            continue
        candidates = (m[:, :-4] & m[:, 4:]
                      & ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1]))
        for y, x in np.argwhere(candidates):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            if not (_at(wire, r - dr, c - dc) or _at(wire, r + 5*dr, c + 5*dc)):
                continue
            matches = []
            for endpoint in (0, 4):
                ar, ac = r + endpoint*dr, c + endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward*dr + side*dc, toward*dc + side*dr
                    spur = [(ar + k*sr, ac + k*sc) for k in (1, 2)]
                    if (all(_at(wire, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                            and not _at(wire, ar + 3*sr, ac + 3*sc)):
                        matches.append(tuple(spur))
            if len(matches) > 1:
                raise ValueError("Ambiguous wire deviation")
            if matches:
                gap = tuple((r + k*dr, c + k*dc) for k in (1, 2, 3))
                for p in gap:
                    added[p] = True
                protected[r, c] = protected[r + 4*dr, c + 4*dc] = True
                evidence.append((gap, matches[0]))
    weak = wire & ~_straight_support(wire | added) & ~protected
    diagonal = np.zeros_like(wire)
    for dc in (-1, 1):
        left = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        right = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        linked = left & right
        if dc == 1:
            diagonal[:-1, :-1] |= linked
            diagonal[1:, 1:] |= linked
        else:
            diagonal[:-1, 1:] |= linked
            diagonal[1:, :-1] |= linked
    return (wire & ~diagonal) | added, evidence


def _decode(value):
    """Recover role colors and source rectangles from this input alone."""
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError("Background must occupy more than half the input")
    boxes = []
    H, W = a.shape
    # A source is a solid one-cell rectangular border around a tight, single-ink
    # 3..5 by 3..5 motif. Neither layout nor literal frame color is prescribed.
    for r in range(H - 4):
        for c in range(W - 4):
            frame = int(a[r, c])
            if frame == bg or a[r, c+1] != frame or a[r+1, c] != frame:
                continue
            for h in range(5, min(7, H-r) + 1):
                for w in range(5, min(7, W-c) + 1):
                    b = a[r:r+h, c:c+w]
                    if not (np.all(b[0] == frame) and np.all(b[-1] == frame)
                            and np.all(b[:, 0] == frame) and np.all(b[:, -1] == frame)):
                        continue
                    inner = b[1:-1, 1:-1]
                    inks = set(map(int, np.unique(inner))) - {bg}
                    if len(inks) != 1 or frame in inks:
                        continue
                    ink = inks.pop()
                    mask = inner == ink
                    if not (mask[0].any() and mask[-1].any()
                            and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    boxes.append((r, c, h, w, frame, ink, mask.copy()))
    if not boxes or len({b[4] for b in boxes}) != 1:
        raise ValueError("Sources must have one identifiable frame role")
    cover = np.zeros_like(a, dtype=bool)
    book = {}
    for r, c, h, w, frame, ink, mask in boxes:
        if cover[r:r+h, c:c+w].any():
            raise ValueError("Source rectangles overlap or are ambiguous")
        cover[r:r+h, c:c+w] = True
        if ink in book and not np.array_equal(book[ink], mask):
            raise ValueError("A stencil identity has inconsistent source copies")
        book[ink] = mask
    outside = set(map(int, np.unique(a[~cover]))) - {bg}
    wire_candidates = []
    for color in outside:
        m = (a == color) & ~cover
        if (m[:, :-1] & m[:, 1:]).any() or (m[:-1] & m[1:]).any():
            wire_candidates.append(color)
    if len(wire_candidates) != 1:
        raise ValueError("Exactly one outside color must contain straight wire runs")
    wire = wire_candidates[0]
    targets = outside - {wire}
    if len(targets) != 1:
        raise ValueError("Expected one isolated printer-marker color")
    target = targets.pop()
    if len({bg, wire, target, boxes[0][4]} | set(book)) != 4 + len(book):
        raise ValueError("Semantic palette roles must be disjoint")
    return a, bg, wire, target, boxes, cover, book


def _turn(direction):
    """Clockwise quarter-turns, relative to a wire entering from the west."""
    return {(0, -1): 0, (-1, 0): 1, (0, 1): 2, (1, 0): 3}[direction]


def _print(out, anchor, ink, mask, direction):
    rotated = np.rot90(mask, -_turn(direction))
    r, c = anchor
    h, w = min(rotated.shape[0], out.shape[0]-r), min(rotated.shape[1], out.shape[1]-c)
    out[r:r+h, c:c+w][rotated[:h, :w]] = ink


def _evaluate(value, repair=True, stamp=True, fill_all=False, rotate=True):
    a, bg, wire_color, target, boxes, cover, book = _decode(value)
    raw = (a == wire_color) & ~cover
    clean, evidence = _repair_wires(raw)
    wires = clean.copy() if repair else raw.copy()
    if fill_all:
        # Keep receiver orientation from the real input. Filling unsupported
        # breaks can create incidental extra contacts, which are precisely a
        # reason this alternative is not the intended input grammar.
        for m in (wires, wires.T):
            if m.shape[1] < 5:
                continue
            candidates = m[:, :-4] & m[:, 4:]
            candidates &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
            for r, c in np.argwhere(candidates):
                m[r, c+1:c+4] = True
    printers = list(map(tuple, np.argwhere((a == target) & ~cover).tolist()))
    # A complete source widget is a conducting hub. This joins all of its ports
    # without mistaking its motif pixels for wire or diagonal repair evidence.
    conductor = wires | cover | ((a == target) & ~cover)
    labels, components = _components(conductor)
    signals = [set() for _ in components]
    for r, c, h, w, _, ink, _ in boxes:
        signals[int(labels[r, c])].add(ink)
    out = np.full(a.shape, bg, dtype=np.int8)
    records = []
    for p in printers:
        neighbors = [(dr, dc) for dr, dc in _ORTH
                     if _at(clean, p[0]+dr, p[1]+dc)]
        if len(neighbors) != 1:
            raise ValueError("Every printer has exactly one incident straight wire")
        identity = signals[int(labels[p])]
        ink = next(iter(identity)) if len(identity) == 1 else None
        records.append((p, neighbors[0], frozenset(identity), ink))
        if ink is not None:
            if stamp:
                _print(out, p, ink, book[ink], neighbors[0] if rotate else (0, -1))
            else:
                out[p] = ink
    info = {"bg": bg, "wire": wire_color, "target": target, "boxes": boxes,
            "book": book, "cover": cover, "raw": raw, "clean": clean,
            "repairs": evidence, "records": records, "labels": labels}
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output pixel exclusively from the supplied input."""
    return _evaluate(input_grid)[0]


def _sample_mask(rng, h, w):
    m = np.zeros((h, w), dtype=bool)
    family = int(rng.integers(6))
    if family == 0:
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    elif family == 1:
        m[0, 1:-1] = m[-1, 1:-1] = True
        m[1:-1, 0] = m[1:-1, -1] = True
    elif family == 2:
        m[:] = True
        # Optional notches provide asymmetric dense masks with tight bounds.
        for _ in range(int(rng.integers(1, h*w//3 + 1))):
            m[int(rng.integers(1, h-1)), int(rng.integers(1, w-1))] = False
    elif family == 3:
        for r in range(h):
            c = round(r * (w-1) / (h-1))
            m[r, c] = m[r, w-1-c] = True
    elif family == 4:
        m[:, 0] = m[-1, :] = True
        m[int(rng.integers(h-1)), :int(rng.integers(2, w+1))] = True
    else:
        r = c = 0
        m[r, c] = True
        while r < h-1 or c < w-1:
            if r == h-1 or (c < w-1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            m[r, c] = True
        for _ in range(int(rng.integers(1, h+w))):
            r, c = map(int, np.argwhere(m)[int(rng.integers(np.count_nonzero(m)))])
            dr, dc = _ORTH[int(rng.integers(4))]
            if _inside(m.shape, (r+dr, c+dc)):
                m[r+dr, c+dc] = True
    if rng.integers(2):
        m = m[::-1]
    if rng.integers(2):
        m = m[:, ::-1]
    return m.copy()


def _graph_labels(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    labels = [-1]*n
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i], stack = i, [i]
        while stack:
            for j in adj[stack.pop()]:
                if labels[j] < 0:
                    labels[j] = i
                    stack.append(j)
    return labels


def _edge_cells(p, q):
    dr, dc = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
    return [(p[0]+k*dr, p[1]+k*dc) for k in range(abs(q[0]-p[0])+abs(q[1]-p[1])+1)]


def _neighbor_counts(a, bg):
    """Count non-background cells in each clipped 3-by-3 neighborhood."""
    H, W = a.shape
    occupied = np.pad(a != bg, 1)
    counts = np.zeros(a.shape, dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            counts += occupied[r:r+H, c:c+W]
    return counts


def _gap_options(a, cells, bg, wire, neighbors=None):
    if neighbors is None:
        neighbors = _neighbor_counts(a, bg)
    options = []
    dr, dc = cells[1][0]-cells[0][0], cells[1][1]-cells[0][1]
    for i in range(len(cells)-4):
        five = cells[i:i+5]
        if not all(a[p] == wire for p in five):
            continue
        r, c = five[0]
        if not ((_inside(a.shape, (r-dr, c-dc)) and a[r-dr, c-dc] == wire)
                or (_inside(a.shape, (r+5*dr, c+5*dc)) and a[r+5*dr, c+5*dc] == wire)):
            continue
        gap = tuple(five[1:4])
        for endpoint in (0, 4):
            ar, ac = five[endpoint]
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                if not all(_inside(a.shape, p) for p in spur):
                    continue
                # The first diagonal cell neighbors exactly three allowed
                # occupied cells: the anchor and the first two gap cells.
                # The second neighbors none. These counts also guarantee that
                # both spur cells are blank. Any extra occupied neighbor would
                # be an incidental contact with a junction, box, or printer.
                if neighbors[spur[0]] != 3 or neighbors[spur[1]] != 0:
                    continue
                third = ar+3*sr, ac+3*sc
                if _inside(a.shape, third) and a[third] != bg and third not in gap:
                    continue
                options.append((gap, spur))
    return options


def _axis(rng, count, extent):
    start = int(rng.integers(0, 4))
    steps = [int(rng.integers(7, 10)) for _ in range(count-1)]
    coords = [start]
    for step in steps:
        coords.append(coords[-1] + step)
    size = coords[-1] + int(rng.integers(1, max(2, extent+1)))
    return coords, size


def _make_pair(rng, palette, book, mode, taught=None):
    bg, wire, frame, target = palette
    inks = list(book)
    mh, mw = next(iter(book.values())).shape
    # The only retry loop is bounded; failed candidates have no external effect.
    for attempt in range(12000):
        R, C = map(int, rng.integers(3, 5, size=2))
        rows, H = _axis(rng, R, max(mh, mw))
        cols, W = _axis(rng, C, max(mh, mw))
        if max(H, W) > 30 or min(H, W) < 18:
            continue
        coords = [(r, c) for r in rows for c in cols]
        n = len(coords)
        possible = [(r*C+c, (r+dr)*C+c+dc)
                    for r in range(R) for c in range(C) for dr, dc in ((1, 0), (0, 1))
                    if r+dr < R and c+dc < C]
        probability = float(rng.uniform(.30, .53))
        edges = [e for e in possible if rng.random() < probability]
        degree = [0]*n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        bh, bw = mh+2, mw+2
        eligible = [i for i, (r, c) in enumerate(coords) if degree[i] > 0
                    and r-bh//2 >= 0 and c-bw//2 >= 0
                    and r-bh//2+bh <= H and c-bw//2+bw <= W]
        ns = int(rng.integers(3 if mode == "conflict" else 2, 5))
        if len(eligible) < ns:
            continue
        sources = list(map(int, rng.choice(eligible, ns, replace=False)))
        assignments = {sources[0]: inks[0], sources[1]: inks[1]}
        for i in sources[2:]:
            assignments[i] = int(rng.choice(inks))
        leaves = [i for i in range(n) if degree[i] == 1 and i not in sources]
        if len(leaves) < 3:
            continue
        a = np.full((H, W), bg, dtype=np.int8)
        for u, v in edges:
            for p in _edge_cells(coords[u], coords[v]):
                a[p] = wire
        source_rects = []
        for i in sources:
            r, c = coords[i][0]-bh//2, coords[i][1]-bw//2
            a[r:r+bh, c:c+bw] = frame
            inner = a[r+1:r+bh-1, c+1:c+bw-1]
            inner[:] = bg
            inner[book[assignments[i]]] = assignments[i]
            source_rects.append((r, c, bh, bw))
        # Vary occupancy: not every straight endpoint is a printing instruction.
        printers = [i for i in leaves if rng.random() < .82]
        if len(printers) < 3:
            printers = leaves
        for i in printers:
            a[coords[i]] = target
        neighbors = _neighbor_counts(a, bg)
        choices = {e: _gap_options(a, _edge_cells(coords[e[0]], coords[e[1]]), bg, wire, neighbors)
                   for e in edges}
        repairable = [e for e in edges if choices[e]]
        nr = int(rng.integers(2, 4)) if mode == "serial" else int(rng.choice([1, 1, 1, 2]))
        if len(repairable) < nr+1:
            continue
        order = rng.permutation(len(repairable))
        repairs = [repairable[int(i)] for i in order[:nr]]
        remaining = [repairable[int(i)] for i in order[nr:]]
        nb = int(rng.integers(1, min(3, len(remaining))+1))
        broken = remaining[:nb]
        final_edges = [e for e in edges if e not in broken]
        before_edges = [e for e in final_edges if e not in repairs]
        after = _graph_labels(n, final_edges)
        before = _graph_labels(n, before_edges)
        all_joined = _graph_labels(n, edges)

        def signals(i, labels):
            return {assignments[s] for s in sources if labels[s] == labels[i]}

        active = [i for i in printers if len(signals(i, after)) == 1]
        silent = [i for i in printers if len(signals(i, after)) != 1]
        activated = [i for i in active if not signals(i, before)]
        conflicted = [i for i in silent if len(signals(i, after)) > 1
                      and len(signals(i, before)) == 1]
        if not active or not silent:
            continue
        live_directions, live_clipping = set(), set()
        for i in active:
            j = next(v if u == i else u for u, v in edges if i in (u, v))
            p, q = coords[i], coords[j]
            direction = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
            live_directions.add(direction)
            ink = next(iter(signals(i, after)))
            mask = np.rot90(book[ink], -_turn(direction))
            if p[0]+mask.shape[0] > H:
                live_clipping.add("bottom")
            if p[1]+mask.shape[1] > W:
                live_clipping.add("right")
            if p[0]+mask.shape[0] > H and p[1]+mask.shape[1] > W:
                live_clipping.add("corner")
        if taught is not None and (not live_directions <= taught[0] or not live_clipping <= taught[1]):
            continue
        if taught is not None and not taught[2] and any(
                sum(after[s] == after[i] for s in sources) > 1 for i in active):
            continue
        if mode == "activate":
            if (not activated or {next(iter(signals(i, after))) for i in active} != set(inks)
                    or not any(not signals(i, after) for i in silent)
                    or len(live_directions) < 2):
                continue
        elif mode == "conflict":
            if not conflicted:
                continue
        elif mode == "serial":
            # At least one receiver needs two or more repairs in SERIES, rather
            # than merely coexisting independent repairs on unrelated branches.
            witnesses = []
            for i in activated:
                necessary = 0
                for edge in repairs:
                    ids = _graph_labels(n, [e for e in final_edges if e != edge])
                    necessary += not signals(i, ids)
                if necessary >= 2:
                    witnesses.append(i)
            if not witnesses:
                continue
        # This observable negative refutes closing every three-cell break.
        if not any((signals(i, after) if len(signals(i, after)) == 1 else set())
                   != (signals(i, all_joined) if len(signals(i, all_joined)) == 1 else set())
                   for i in printers):
            continue
        clean = a.copy()
        used = set()
        selected = {}
        collision = False
        for edge in repairs + broken:
            opts = choices[edge]
            order = rng.permutation(len(opts))
            for k in order:
                gap, spur = opts[int(k)]
                footprint = set(gap) | (set(spur) if edge in repairs else set())
                if any(max(abs(r-x), abs(c-y)) <= 1
                       for r, c in footprint for x, y in used):
                    continue
                selected[edge] = gap, spur
                used |= footprint
                break
            else:
                collision = True
                break
            for p in gap:
                a[p] = bg
                if edge in broken:
                    clean[p] = bg
            if edge in repairs:
                for p in spur:
                    a[p] = wire
        if collision:
            continue
        # Add 0..3 detached diagonal fragments. Their erasure is shared with A,
        # but they carry neither reachability nor stencil identity.
        for _ in range(int(rng.integers(4))):
            for _ in range(25):
                r, c = int(rng.integers(H)), int(rng.integers(W))
                dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
                length = int(rng.choice([2, 3]))
                pts = [(r+k*dr, c+k*dc) for k in range(length)]
                if all(_inside(a.shape, p) and a[p] == bg
                       and all(not _inside(a.shape, (p[0]+d1, p[1]+d2))
                               or a[p[0]+d1, p[1]+d2] == bg
                               for d1 in (-1, 0, 1) for d2 in (-1, 0, 1)) for p in pts):
                    for p in pts:
                        a[p] = wire
                    break
        if np.count_nonzero(a == bg) <= a.size//2:
            continue
        # Ground truth comes from abstract graph connectivity, separately from
        # solve's pixel flood fill and source parsing.
        out = np.full(a.shape, bg, dtype=np.int8)
        for i in active:
            j = next(v if u == i else u for u, v in edges if i in (u, v))
            p, q = coords[i], coords[j]
            direction = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
            ink = next(iter(signals(i, after)))
            _print(out, p, ink, book[ink], direction)
        try:
            predicted, info = _evaluate(a)
        except ValueError:
            continue
        if (not np.array_equal(info["clean"], clean == wire)
                or len(info["repairs"]) != nr or len(info["boxes"]) != ns
                or not np.array_equal(predicted, out)):
            continue
        observed = {p: set(ids) for p, _, ids, _ in info["records"]}
        if any(observed[coords[i]] != signals(i, after) for i in printers):
            continue
        actual_counts = _counterfactual_signals(info, info["clean"])
        if any(actual_counts[coords[i]][1] != sum(after[s] == after[i] for s in sources)
               for i in printers):
            continue
        if not (out != bg).any() or np.array_equal(_evaluate(a, stamp=False)[0], out):
            continue
        before_out, before_info = _evaluate(a, repair=False)
        observed_before = {p: set(ids) for p, _, ids, _ in before_info["records"]}
        if any(observed_before[coords[i]] != signals(i, before) for i in printers):
            continue
        original_counts = _counterfactual_signals(info, info["raw"])
        if any(original_counts[coords[i]][1] != sum(before[s] == before[i] for s in sources)
               for i in printers):
            continue
        if (np.array_equal(before_out, out) or np.array_equal(_fill_every_gap(a), out)):
            continue
        # All active identities get a full, unobscured output demonstration in
        # the activating training pair; source widgets are always fully visible.
        if mode == "activate":
            fully_taught = set()
            for p, direction, _, ink in info["records"]:
                if ink is None:
                    continue
                mask = np.rot90(book[ink], -_turn(direction))
                if p[0]+mask.shape[0] <= H and p[1]+mask.shape[1] <= W:
                    fully_taught.add(ink)
            if fully_taught != set(inks):
                continue
            if np.array_equal(_evaluate(a, rotate=False)[0], out):
                continue
        return {"input": a, "output": out}
    raise RuntimeError(f"Could not construct {mode} network within the bounded search")


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    palette = tuple(colors[:4])
    h, w = map(int, rng.integers(3, 6, size=2))
    book = {ink: _sample_mask(rng, h, w) for ink in colors[4:6]}
    # Keep orientation informative. Symmetric motifs may coexist with a motif
    # whose quarter-turn changes its geometry; no invisible orientation state.
    if all(np.array_equal(mask, np.rot90(mask)) for mask in book.values()):
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = mask[-1, :] = True
        mask[0, 1] = True
        book[colors[5]] = mask
    modes = ["activate", "conflict"] + [str(rng.choice(["activate", "conflict"])) for _ in range(nt-2)]
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, schedule in (("train", modes), ("test", ["serial"]*nq)):
        taught = None
        if split == "test":
            training_infos = [_analyze_pair(p) for p in puzzle["train"]]
            taught = (set.union(*(i["live_directions"] for i in training_infos)),
                      set.union(*(i["visible_clipping"] for i in training_infos)),
                      any(i["duplicate_active"] for i in training_infos))
        for mode in schedule:
            for _ in range(20):
                pair = _make_pair(rng, palette, book, mode, taught)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Failed to make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fill_every_gap(value):
    """A deliberately simpler alternative, used only in negative checks."""
    return _evaluate(value, fill_all=True)[0]


def _counterfactual_signals(info, wires):
    """Source sets at the SAME visible printers with specified wires."""
    conductor = wires | info["cover"]
    for p, _, _, _ in info["records"]:
        conductor[p] = True
    labels, components = _components(conductor)
    source_sets = [set() for _ in components]
    source_counts = [0 for _ in components]
    for r, c, _, _, _, ink, _ in info["boxes"]:
        index = int(labels[r, c])
        source_sets[index].add(ink)
        source_counts[index] += 1
    return {p: (source_sets[int(labels[p])], source_counts[int(labels[p])])
            for p, _, _, _ in info["records"]}


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    expected, info = _evaluate(inp)
    assert np.array_equal(expected, out), "Output disagrees with complete input-only prediction"
    assert info["repairs"], "Every pair must expose the repair concept"
    assert any(x[3] is not None for x in info["records"]), "Need a live printer"
    assert any(x[3] is None for x in info["records"]), "Need a visibly rejected printer"
    assert (out != info["bg"]).any(), "Need a visible live stamp"
    before_out, before_info = _evaluate(inp, repair=False)
    assert not np.array_equal(before_out, out), "Ignoring A must change an answer"
    assert not np.array_equal(_evaluate(inp, stamp=False)[0], out), "Ignoring B must change an answer"
    assert not np.array_equal(_fill_every_gap(inp), out), "Closing unsupported gaps must change an answer"
    assert set(map(int, np.unique(out))) <= {info["bg"]} | set(info["book"])
    occupancy = np.zeros(inp.shape, dtype=bool)
    taught, directions, live_directions, clipped, visible_clipping = set(), set(), set(), set(), set()
    for p, direction, _, ink in info["records"]:
        directions.add(direction)
        if ink is None:
            continue
        live_directions.add(direction)
        mask = np.rot90(info["book"][ink], -_turn(direction))
        h, w = mask.shape
        r, c = p
        categories = set()
        rect = np.zeros(inp.shape, dtype=bool)
        rect[r:r+h, c:c+w] = True
        assert not (rect & occupancy).any(), "Printer windows overlap"
        occupancy |= rect
        if r+h > inp.shape[0]:
            categories.add("bottom")
        if c+w > inp.shape[1]:
            categories.add("right")
        if r+h > inp.shape[0] and c+w > inp.shape[1]:
            categories.add("corner")
        clipped |= categories
        if mask[:inp.shape[0]-r, :inp.shape[1]-c].any():
            visible_clipping |= categories
        if r+h <= inp.shape[0] and c+w <= inp.shape[1]:
            taught.add(ink)
    assert not (info["clean"][:-1, :-1] & info["clean"][1:, :-1]
                & info["clean"][:-1, 1:] & info["clean"][1:, 1:]).any(), "Wire thickness changed"
    necessary = {p: 0 for p, _, _, ink in info["records"] if ink is not None}
    for gap, _ in info["repairs"]:
        cut = info["clean"].copy()
        for p in gap:
            assert not info["raw"][p] and cut[p]
            cut[p] = False
        alternatives = _counterfactual_signals(info, cut)
        for p in necessary:
            necessary[p] += not alternatives[p][0]
    component_counts = _counterfactual_signals(info, info["clean"])
    duplicate_active = any(ink is not None and component_counts[p][1] > 1
                           for p, _, _, ink in info["records"])
    before_sets = {p: ids for p, _, ids, _ in before_info["records"]}
    activated = sum(len(ids) == 1 and not before_sets[p] for p, _, ids, _ in info["records"])
    suppressed = sum(len(ids) > 1 and len(before_sets[p]) == 1 for p, _, ids, _ in info["records"])
    info.update({"shape": inp.shape, "taught": taught, "directions": directions,
                 "live_directions": live_directions,
                 "clipped": clipped, "visible_clipping": visible_clipping,
                 "active": sum(x[3] is not None for x in info["records"]),
                 "conflicts": sum(len(x[2]) > 1 for x in info["records"]),
                 "empty": sum(not x[2] for x in info["records"]),
                 "serial": max(necessary.values(), default=0),
                 "duplicate_active": duplicate_active,
                 "activated": activated, "suppressed": suppressed,
                 "rotation_contrast": not np.array_equal(_evaluate(inp, rotate=False)[0], out)})
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos, book_key, roles = set(), {"train": [], "test": []}, None, None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _analyze_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "Duplicate input"
            seen.add(key)
            dictionary = tuple(sorted((ink, _key(mask)) for ink, mask in info["book"].items()))
            palette = (info["bg"], info["wire"], info["boxes"][0][4], info["target"])
            if book_key is None:
                book_key, roles = dictionary, palette
            assert dictionary == book_key and palette == roles, "Puzzle conventions changed"
            infos[split].append(info)
    train = infos["train"]
    assert set.union(*(i["taught"] for i in train)) == set(train[0]["book"])
    assert any(i["conflicts"] for i in train), "Training must teach conflicting identities"
    assert any(i["empty"] for i in train), "Training must teach source-free printers"
    assert any(i["activated"] for i in train), "Training must show repair activating a printer"
    assert any(i["suppressed"] for i in train), "Training must show repair creating a conflict"
    assert any(i["rotation_contrast"] for i in train), "Training must distinguish copying canonical orientation"
    directions = set.union(*(i["live_directions"] for i in train))
    clipping = set.union(*(i["visible_clipping"] for i in train))
    for info in infos["test"]:
        assert len(info["repairs"]) >= 2
        assert info["serial"] >= 2, "Test must need repairs in series"
        assert info["live_directions"] <= directions, "Untaught test rotation"
        assert info["clipped"] <= clipping, "Untaught test clipping"
        assert not info["duplicate_active"] or any(i["duplicate_active"] for i in train), "Untaught duplicate-source agreement"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed sweep, count checks, and semantic mutation regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, masks, normalized, source_counts, printer_counts, repair_counts = set(), set(), set(), set(), set(), set()
    directions, palettes, clipping = set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                shapes.add(info["shape"])
                source_counts.add(len(info["boxes"]))
                printer_counts.add(len(info["records"]))
                repair_counts.add(len(info["repairs"]))
                directions |= info["directions"]
                clipping |= info["clipped"]
                palettes.add((info["bg"], info["wire"], info["target"]))
                masks.update(_key(mask) for mask in info["book"].values())
                role_map = {}
                norm = np.array([role_map.setdefault(int(c), len(role_map))
                                 for c in pair["input"].ravel()], dtype=np.int8).reshape(info["shape"])
                normalized.add(_key(norm))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(masks) >= 25 and len(normalized) >= 180
    assert source_counts >= {2, 3, 4} and len(printer_counts) >= 3
    assert repair_counts >= {1, 2, 3} and directions == set(_ORTH)
    assert len(palettes) >= 35 and clipping >= {"bottom", "right"}
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for nt, nq in ((2, 1), (10, 4), (6, 3)):
        p = generate_puzzle(123, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    corrupted = copy.deepcopy(puzzles[0])
    query = corrupted["test"][0]
    info = _analyze_pair(query)
    p, direction, _, ink = next(x for x in info["records"] if x[3] is not None)
    mask = np.rot90(info["book"][ink], -_turn(direction))
    r, c = p
    query["output"][r:r+mask.shape[0], c:c+mask.shape[1]] = info["bg"]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("An omitted live stamp passed validation")
    return {"seeds": 42, "distinct_shapes": len(shapes), "distinct_masks": len(masks),
            "normalized_inputs": len(normalized), "source_counts": sorted(source_counts),
            "printer_counts": sorted(printer_counts), "repair_counts": sorted(repair_counts),
            "semantic_verification": "generator-owned checks only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
