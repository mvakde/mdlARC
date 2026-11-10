"""Badge-anchored connector assembly followed by one global bottom-left shear.

Only standard library and NumPy are used. No parent module, file, example, seed,
or generation metadata is consulted by solve(). Diagnostics stay in memory.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 611999946, 955146464, 963482647, 40988114, 29744195,
              279600614, 802348988, 548356502, 830125930, 963463429,
              336563648, 52354526, 789637450, 337774786, 137326653,
              325904277, 811265362, 612122827, 735747091, 435048893,
              647535368, 845912898, 638598389, 482073932, 796736397,
              585744982, 692187277, 709353302, 878256544, 302192188,
              854945982, 131085548, 601210354, 699626185, 810957614,
              181258081, 752833530, 459088921, 417669293, 521709864,
              481026707)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _analyze(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size:
        raise ValueError("background must occupy more than half the input")
    groups = _components(map(tuple, np.argwhere(a != bg).tolist()))
    if not 3 <= len(groups) <= 6:
        raise ValueError("expected three through six separated pieces")
    membership = {q: i for i, group in enumerate(groups) for q in group}
    if any(membership.get((r + dr, c + dc), i) != i
           for (r, c), i in membership.items() for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))):
        raise ValueError("scattered pieces must be separated even diagonally")
    palettes = [{int(a[q]) for q in group} for group in groups]
    common = set.intersection(*palettes)
    if len(common) != 1:
        raise ValueError("pieces must have exactly one common connector hue")
    marker = common.pop()
    anchors = [i for i, colors in enumerate(palettes) if len(colors) == 3]
    if len(anchors) != 1 or any(len(colors) != 2 for i, colors in enumerate(palettes)
                                 if i != anchors[0]):
        raise ValueError("exactly one piece must carry a third-color badge")
    pieces = []
    for group, colors in zip(groups, palettes):
        payload = {q: int(a[q]) for q in sorted(group) if a[q] != marker}
        pins = group - set(payload)
        if len(_components(payload)) != 1 or not 2 <= len(pins) <= 6:
            raise ValueError("invalid rigid payload or connector count")
        if any(not any((r + dr, c + dc) in payload for dr, dc in N4)
               for r, c in pins):
            raise ValueError("each connector cell must touch its own payload")
        pieces.append(dict(payload=payload, pins=pins, cells=group, box=_box(group)))
    ai = anchors[0]
    frequencies = Counter(pieces[ai]["payload"].values())
    badges = [c for c, n in frequencies.items() if n == 1]
    if len(badges) != 1:
        raise ValueError("anchor badge must be one cell of its own hue")
    badge = badges[0]
    if np.count_nonzero(a == badge) != 1:
        raise ValueError("badge hue must occur only once in the input")
    anchor_body = next(c for c in frequencies if c != badge)
    if any(anchor_body in p["payload"].values() for i, p in enumerate(pieces) if i != ai):
        raise ValueError("anchor body hue is distinct from movable payload hues")
    return a, bg, marker, badge, pieces, ai


class AssemblySearchLimit(ValueError):
    """An unfinished uniqueness check never counts as a unique assembly."""


def _assemble(value, limit=12000):
    """Find the unique in-bounds chain by complete two-owner pin coverage.

    Sparse signatures are not partitioned by connected components or area.
    Search considers every pin-induced translation with a 2/3-cell joint.
    The graph is grown from the visible anchor; degrees <=2 forbid branches.
    """
    a, bg, marker, badge, pieces, ai = _analyze(value)
    n = len(pieces)
    links = {}
    for i, p in enumerate(pieces):
        for j, q in enumerate(pieces):
            if i == j:
                continue
            by_pin = {}
            deltas = sorted({(r - s, c - t) for r, c in p["pins"] for s, t in q["pins"]})
            for d in deltas:
                body, pins = _shift(q["payload"], d), _shift(q["pins"], d)
                overlap = p["pins"] & pins
                if len(overlap) not in (2, 3) or body & p["cells"] or pins & set(p["payload"]):
                    continue
                for cell in sorted(overlap):
                    by_pin.setdefault(cell, []).append(d)
            links[i, j] = by_pin
    solutions, visited = [], set()
    nodes = 0

    def visit(positions, bodies, singles, doubles, owners, degrees):
        nonlocal nodes
        state = tuple(sorted(positions.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise AssemblySearchLimit("connector uniqueness search exceeded its bounded node budget")
        if not singles:
            if len(positions) == n:
                solutions.append(dict(positions))
            return
        if len(positions) == n:
            return
        best = None
        for cell in sorted(singles):
            i = owners[cell]
            if degrees[i] >= 2:
                return
            di = positions[i]
            local = cell[0] - di[0], cell[1] - di[1]
            choices = []
            for j, p in enumerate(pieces):
                if j in positions:
                    continue
                for relative in links[i, j].get(local, ()):
                    d = di[0] + relative[0], di[1] + relative[1]
                    t, l, b, r = p["box"]
                    if t + d[0] < 0 or l + d[1] < 0 or b + d[0] >= a.shape[0] or r + d[1] >= a.shape[1]:
                        continue
                    body, pins = _shift(p["payload"], d), _shift(p["pins"], d)
                    if body & (bodies | singles | doubles) or pins & (bodies | doubles):
                        continue
                    overlap = pins & singles
                    if len(overlap) not in (2, 3) or {owners[x] for x in overlap} != {i}:
                        continue
                    choices.append((j, d, body, pins, overlap, i))
            if not choices:
                return
            if best is None or len(choices) < len(best):
                best = choices
        for j, d, body, pins, overlap, parent in best:
            new_owners = dict(owners)
            new_owners.update({q: j for q in pins - overlap})
            dd = dict(degrees)
            dd[parent] += 1
            dd[j] = 1
            positions[j] = d
            visit(positions, bodies | body, (singles | pins) - overlap,
                  doubles | overlap, new_owners, dd)
            del positions[j]
            if len(solutions) > 1:
                return

    p = pieces[ai]
    visit({ai: (0, 0)}, set(p["payload"]), p["pins"], set(),
          {q: ai for q in p["pins"]}, {ai: 0})
    if len(solutions) != 1:
        raise ValueError(f"expected one anchored chain, found {len(solutions)}")
    positions = solutions[0]
    assembled = np.full(a.shape, bg, dtype=np.int8)
    for i, p in enumerate(pieces):
        dr, dc = positions[i]
        for (r, c), color in p["payload"].items():
            assembled[r + dr, c + dc] = color
        for r, c in p["pins"]:
            assembled[r + dr, c + dc] = marker
    return assembled, (bg, marker, badge, pieces, ai, positions, nodes)


def _shear(assembled, bg, bottom=None, direction=-1, rate=1):
    rows, cols = np.nonzero(assembled != bg)
    if bottom is None:
        bottom = int(rows.max())
    targets = cols + direction * rate * (bottom - rows)
    keep = (targets >= 0) & (targets < assembled.shape[1])
    out = np.full(assembled.shape, bg, dtype=np.int8)
    out[rows[keep], targets[keep]] = assembled[rows[keep], cols[keep]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    assembled, data = _assemble(input_grid)
    return _shear(assembled, data[0])


def _rectangle(t, l, h, w):
    return {(r, c) for r in range(t, t + h) for c in range(l, l + w)}


def _layout(rng, count, scale, large):
    """Invent a path of independent body envelopes and sparse keyed joints."""
    for restart in range(16):
        h = int(rng.integers(2, scale + 1))
        w = int(rng.integers(2, scale + 1))
        if large:
            h, w = int(rng.integers(5, 7)), int(rng.integers(5, 7))
        nodes = [dict(rect=_rectangle(0, 0, h, w), arms=set(), pins=set(), required=set())]
        edges = []
        incoming = None
        for index in range(1, count):
            parent = nodes[-1]
            for attempt in range(72):
                dr, dc = _pick(rng, N4)
                if incoming == (dr, dc):
                    continue
                tr, tc = -dc, dr
                projections = [(r * dr + c * dc, r * tr + c * tc) for r, c in parent["rect"]]
                normal = max(x for x, y in projections)
                lo, hi = min(y for x, y in projections), max(y for x, y in projections)
                depth, span = int(rng.integers(2, scale + 1)), int(rng.integers(2, scale + 1))
                start = int(rng.integers(lo - span + 2, hi))
                shared = list(range(max(lo, start), min(hi, start + span - 1) + 1))
                if len(shared) < 2:
                    continue
                size = 3 if len(shared) >= 3 and rng.random() < 0.25 else 2
                tangents = sorted(int(x) for x in rng.choice(shared, size, replace=False))
                gap = int(rng.integers(1, 3))
                offsets = [int(rng.integers(gap)) for _ in tangents]

                def coord(normal_coord, tangent_coord):
                    return normal_coord * dr + tangent_coord * tr, normal_coord * dc + tangent_coord * tc

                pins = {coord(normal + 1 + offset, s) for s, offset in zip(tangents, offsets)}
                pa = {coord(k, s) for s, offset in zip(tangents, offsets)
                      for k in range(normal + 1, normal + 1 + offset)}
                ca = {coord(k, s) for s, offset in zip(tangents, offsets)
                      for k in range(normal + 2 + offset, normal + gap + 1)}
                rect = {coord(k, s) for k in range(normal + gap + 1, normal + gap + 1 + depth)
                        for s in range(start, start + span)}
                child = rect | ca | pins
                old = set().union(*(p["rect"] | p["arms"] | p["pins"] for p in nodes))
                if child & old or pa & parent["pins"]:
                    continue
                older = set().union(*(p["rect"] | p["arms"] | p["pins"] for p in nodes[:-1]))
                halo = {(r + a, c + b) for r, c in older for a, b in N4 + ((0, 0),)}
                if (child | pa) & halo:
                    continue
                all_cells = old | child | pa
                t, l, b, r = _box(all_cells)
                if b - t > 24 or r - l > 24:
                    continue
                parent["arms"] |= pa
                parent["pins"] |= pins
                parent["required"] |= {coord(normal, s) for s in tangents}
                nodes.append(dict(rect=rect, arms=ca, pins=set(pins),
                                  required={coord(normal + gap + 1, s) for s in tangents}))
                edges.append((index - 1, index, pins))
                incoming = -dr, -dc
                break
            else:
                break
        if len(nodes) == count:
            keys = [_normal(pins) for _, _, pins in edges]
            if any(len(x) == len(y) and x != y for i, x in enumerate(keys) for y in keys[:i]):
                return nodes, edges
    return None


def _body(rng, node, solid=False):
    t, l, b, r = _box(node["rect"])
    kind = 0 if solid else int(rng.integers(5))
    if kind == 0:
        body = set(node["rect"])
    elif kind in (1, 2):
        side = int(rng.integers(4)) if kind == 2 else -1
        body = {q for q in node["rect"] if any(
            (q[0] == t, q[1] == l, q[0] == b, q[1] == r)[i]
            for i in range(4) if i != side)}
    else:
        body = set(node["rect"])
        for q in [tuple(x) for x in rng.permutation(sorted(body))]:
            if q in node["required"] or rng.random() < 0.58 or len(body) <= 4:
                continue
            body.remove(q)
            if len(_components(body)) != 1:
                body.add(q)
    for target in sorted(node["required"]):
        r0, c0 = min(body, key=lambda q: (abs(q[0] - target[0]) + abs(q[1] - target[1]), q))
        while r0 != target[0]:
            r0 += 1 if target[0] > r0 else -1
            body.add((r0, c0))
        while c0 != target[1]:
            c0 += 1 if target[1] > c0 else -1
            body.add((r0, c0))
    return body | node["arms"]


def _scatter(rng, shape, pieces, ai, bg, marker):
    for restart in range(10):
        inp = np.full(shape, bg, dtype=np.int8)
        occupied = set(pieces[ai]["payload"]) | pieces[ai]["pins"]
        placements = {ai: (0, 0)}
        forbidden = {(r + a, c + b) for r, c in occupied for a in (-1, 0, 1) for b in (-1, 0, 1)}
        order = [i for i in range(len(pieces)) if i != ai]
        rng.shuffle(order)
        for i in order:
            p = pieces[i]
            cells = set(p["payload"]) | p["pins"]
            t, l, b, r = _box(cells)
            for attempt in range(220):
                d = int(rng.integers(shape[0] - b + t)) - t, int(rng.integers(shape[1] - r + l)) - l
                if d == (0, 0):
                    continue
                moved = _shift(cells, d)
                if moved & forbidden:
                    continue
                placements[i] = d
                forbidden |= {(r0 + a, c0 + b0) for r0, c0 in moved
                              for a in (-1, 0, 1) for b0 in (-1, 0, 1)}
                break
            else:
                break
        if len(placements) != len(pieces):
            continue
        for i, d in placements.items():
            for (r, c), color in pieces[i]["payload"].items():
                inp[r + d[0], c + d[1]] = color
            for r, c in pieces[i]["pins"]:
                inp[r + d[0], c + d[1]] = marker
        return inp
    return None


def _separate_shear(shape, bg, marker, pieces, positions):
    """Counterfactual: each already-docked piece uses its own bottom row."""
    out = np.full(shape, bg, dtype=np.int8)
    for i, p in enumerate(pieces):
        dr, dc = positions[i]
        cells = {(r + dr, c + dc): color for (r, c), color in p["payload"].items()}
        cells.update({(r + dr, c + dc): marker for r, c in p["pins"]})
        bottom = max(r for r, c in cells)
        for (r, c), color in cells.items():
            cc = c - (bottom - r)
            if cc >= 0:
                if out[r, cc] not in (bg, color):
                    return None  # A collision is already a distinguishing failure.
                out[r, cc] = color
    return out


def _pair(rng, count, clipped, roles, scale, large=False):
    bg, marker, badge = roles
    palette = [c for c in range(10) if c not in roles]
    for attempt in range(180):
        layout = _layout(rng, count, scale, large)
        if layout is None:
            continue
        nodes, edges = layout
        ai = int(rng.integers(count))
        anchor_hue = int(_pick(rng, palette))
        movable = [c for c in palette if c != anchor_hue]
        pieces = []
        for i, node in enumerate(nodes):
            body = _body(rng, node, solid=large and i == 0)
            color = anchor_hue if i == ai else int(_pick(rng, movable))
            payload = {q: color for q in sorted(body)}
            if i == ai:
                interior = [q for q in sorted(body) if sum((q[0] + dr, q[1] + dc) in body for dr, dc in N4) >= 2]
                payload[_pick(rng, interior)] = badge
            pieces.append(dict(payload=payload, pins=node["pins"]))
        cells = set().union(*(set(p["payload"]) | p["pins"] for p in pieces))
        t, l, b, r = _box(cells)
        bh, bw = b - t + 1, r - l + 1
        required_left = max(b - rr - (cc - l) for rr, cc in cells)
        if clipped:
            options = list(range(max(0, required_left - 4), min(required_left - 1, 30 - bw) + 1))
        else:
            options = list(range(max(0, required_left), min(required_left + 5, 30 - bw) + 1))
        if not options or bh > 25 or bw > 25:
            continue
        left = int(_pick(rng, options))
        h = min(30, max(bh + 2, int(rng.integers(18, 27))) + int(rng.integers(0, 4)))
        w = min(30, max(left + bw, int(rng.integers(18, 27))) + int(rng.integers(0, 4)))
        top = int(rng.integers(h - bh - 1))
        delta = top - t, left - l
        placed = [dict(payload={(rr + delta[0], cc + delta[1]): color for (rr, cc), color in p["payload"].items()},
                       pins=_shift(p["pins"], delta)) for p in pieces]
        assembled = np.full((h, w), bg, dtype=np.int8)
        for p in placed:
            for q, color in p["payload"].items():
                assembled[q] = color
            for q in p["pins"]:
                assembled[q] = marker
        output = _shear(assembled, bg)
        bottom = top + bh - 1
        if np.count_nonzero(output != bg) < 0.60 * np.count_nonzero(assembled != bg):
            continue
        if any(not any(cc - (bottom - rr) >= 0 for rr, cc in p["payload"]) for p in placed):
            continue
        per_piece = _separate_shear((h, w), bg, marker, placed, {i: (0, 0) for i in range(count)})
        if per_piece is not None and np.array_equal(per_piece, output):
            continue
        # The formerly fixed anchor must visibly participate in the final shear.
        if not any(output[q] != color for q, color in placed[ai]["payload"].items()):
            continue
        if np.array_equal(output, _shear(assembled, bg, bottom=h - 1)):
            continue
        inp = _scatter(rng, (h, w), placed, ai, bg, marker)
        if inp is None or np.count_nonzero(inp == bg) * 2 <= inp.size:
            continue
        if int(np.nonzero(inp != bg)[0].max()) == bottom or np.array_equal(_shear(inp, bg), output):
            continue
        try:
            recovered, data = _assemble(inp)
        except ValueError:
            continue
        if not np.array_equal(recovered, assembled):
            continue
        if any(data[5][i] == (0, 0) for i in range(count) if i != data[4]):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded construction could not find a unique, informative chain")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    size_shift = bool(rng.integers(2))
    train_specs = [(3, False), (4, True)]
    if rng.random() < 0.5:
        train_specs = [(3, True), (4, False)]
    train_specs += [(int(rng.integers(3, 6 if size_shift else 5)), bool(rng.integers(2)))
                    for _ in range(num_train - 2)]
    rng.shuffle(train_specs)
    test_specs = [(int(rng.integers(3, 6) if size_shift else rng.integers(5, 7)), bool(rng.integers(2)))
                  for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for count, clipped in specs:
            for retry in range(6):
                large = size_shift and split == "test"
                pair = _pair(rng, count, clipped, roles, scale=4 if size_shift and split == "train" else 5, large=large)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
    return puzzle


def _signature(grid):
    a = np.ascontiguousarray(grid, dtype=np.uint8)
    return a.shape, hashlib.sha256(a.tobytes()).hexdigest()


def _facts(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    assembled, data = _assemble(inp)
    bg, marker, badge, pieces, ai, positions, search_nodes = data
    assert np.array_equal(out, _shear(assembled, bg)), "wrong dock-then-shear output"
    assert positions[ai] == (0, 0)
    assert all(positions[i] != (0, 0) for i in positions if i != ai)
    bodies = [_shift(p["payload"], positions[i]) for i, p in enumerate(pieces)]
    pins = [_shift(p["pins"], positions[i]) for i, p in enumerate(pieces)]
    multiplicity = Counter(q for cells in pins for q in cells)
    assert set(multiplicity.values()) == {2}
    assert not set(multiplicity) & set().union(*bodies)
    assert sum(map(len, bodies)) == len(set().union(*bodies))
    joints = [(i, j, pins[i] & pins[j]) for i in range(len(pieces)) for j in range(i) if pins[i] & pins[j]]
    assert len(joints) == len(pieces) - 1
    degrees = Counter(i for i, j, cells in joints) + Counter(j for i, j, cells in joints)
    assert sorted(degrees.values()) == [1, 1] + [2] * (len(pieces) - 2)
    assert all(len(cells) in (2, 3) for i, j, cells in joints)
    keys = [_normal(cells) for i, j, cells in joints]
    assert any(len(x) == len(y) and x != y for i, x in enumerate(keys) for y in keys[:i])
    assert len(_components(map(tuple, np.argwhere(assembled != bg).tolist()))) == 1
    bottom = int(np.nonzero(assembled != bg)[0].max())
    assert bottom <= inp.shape[0] - 3
    assert int(np.nonzero(inp != bg)[0].max()) != bottom
    assert np.array_equal(out[bottom], assembled[bottom])
    assert not np.array_equal(out, assembled), "omitting A must change the answer"
    assert not np.array_equal(out, _shear(inp, bg)), "omitting B must change the answer"
    assert not np.array_equal(out, _shear(assembled, bg, bottom=inp.shape[0] - 1))
    per_piece = _separate_shear(inp.shape, bg, marker, pieces, positions)
    assert per_piece is None or not np.array_equal(out, per_piece)
    assert any(out[r + positions[ai][0], c + positions[ai][1]] != color
               for (r, c), color in pieces[ai]["payload"].items())
    survivors = int(np.count_nonzero(out != bg))
    mass = int(np.count_nonzero(assembled != bg))
    assert survivors >= 0.60 * mass
    assert all(any(c - (bottom - r) >= 0 for r, c in body) for body in bodies)
    # After docking, plausible shear models must actually differ on this figure.
    for direction, rate in ((1, 1), (-1, 2)):
        assert not np.array_equal(out, _shear(assembled, bg, direction=direction, rate=rate))
    body_colors = [tuple(sorted(set(p["payload"].values()) - {badge})) for p in pieces]
    return dict(background=bg, connector=marker, badge=badge, count=len(pieces),
                canvas=list(inp.shape), assembled_box=list(np.ptp(np.argwhere(assembled != bg), axis=0) + 1),
                bottom=bottom, clipped=mass - survivors, mass=mass, anchor_degree=degrees[ai],
                anchor_row=int(min(r for r, c in bodies[ai])), search_nodes=search_nodes,
                body_boxes=[list(np.ptp(np.array(sorted(p["payload"])), axis=0) + 1) for p in pieces],
                body_areas=[len(p["payload"]) for p in pieces],
                body_shapes=[_normal(p["payload"]) for p in pieces], joint_keys=keys,
                repeated_body_colors=len(set(body_colors)) < len(body_colors),
                sparse_keys=sum(len(_components(cells)) > 1 for i, j, cells in joints),
                geometry=_signature(assembled != bg)[1])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            records[split].append(_facts(pair))
    all_records = records["train"] + records["test"]
    assert len({(f["background"], f["connector"], f["badge"]) for f in all_records}) == 1
    assert {f["clipped"] > 0 for f in records["train"]} == {False, True}
    assert len({f["count"] for f in records["train"]}) >= 2
    count_shift = min(f["count"] for f in records["test"]) > max(f["count"] for f in records["train"])
    size_shift = min(max(f["body_areas"]) for f in records["test"]) > max(max(f["body_areas"]) for f in records["train"])
    assert count_shift or size_shift, "missing visible structural test extension"


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """42 seeds, streaming summaries only; no disk writes or retained runs."""
    counts, shapes, geometry, bodies, keys = set(), set(), set(), set(), set()
    roles = [set(), set(), set()]
    clipping, anchors, shifts = set(), set(), set()
    repeats = sparse = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal(puzzle, generate_puzzle(seed))
        facts = {s: [_facts(p) for p in puzzle[s]] for s in ("train", "test")}
        shifts.add("count" if min(f["count"] for f in facts["test"]) > max(f["count"] for f in facts["train"]) else "size")
        for split in ("train", "test"):
            for pair, f in zip(puzzle[split], facts[split]):
                before = _signature(pair["input"])
                assert np.array_equal(solve(np.asarray(pair["input"])), pair["output"])
                assert _signature(pair["input"]) == before
                counts.add(f["count"])
                shapes.add(tuple(f["canvas"]))
                geometry.add(f["geometry"])
                bodies.update(f["body_shapes"])
                keys.update(f["joint_keys"])
                clipping.add((split, f["clipped"] > 0))
                anchors.add(f["anchor_degree"])
                for target, name in zip(roles, ("background", "connector", "badge")):
                    target.add(f[name])
                repeats += f["repeated_body_colors"]
                sparse += f["sparse_keys"] > 0
                pairs += 1
    assert counts == {3, 4, 5, 6} and shifts == {"count", "size"}
    assert len(shapes) >= 30 and len(geometry) >= 150 and len(bodies) >= 200
    assert len(keys) >= 15 and min(map(len, roles)) >= 8
    assert anchors == {1, 2} and repeats >= 20 and sparse >= 30
    assert clipping == {(s, c) for s in ("train", "test") for c in (False, True)}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(931, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (3.0, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Bad labels and missing connectors must not be silently accepted.
    p = generate_puzzle(28, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bg = _analyze(bad["test"][0]["input"])[1]
    cell = tuple(np.argwhere(bad["test"][0]["output"] != bg)[0])
    bad["test"][0]["output"][cell] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased output cell accepted")
    inp = p["test"][0]["input"].copy()
    _, bg, marker, _, _, _ = _analyze(inp)
    inp[tuple(np.argwhere(inp == marker)[0])] = bg
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired missing connector accepted")
    # Two identical right-facing ports can exchange different terminal colors.
    # Both are collision-free chains: a tie must be rejected, never ordered by hue.
    ambiguous = np.zeros((20, 22), dtype=np.int8)
    ambiguous[5:13, 8:11] = 1
    ambiguous[8, 9] = 9
    ambiguous[[5, 7, 10, 12], 11] = 2
    ambiguous[0:3, 1:3] = 3
    ambiguous[[0, 2], 0] = 2
    ambiguous[16:19, 18:20] = 4
    ambiguous[[16, 18], 17] = 2
    try:
        solve(ambiguous)
    except ValueError as error:
        assert "found 2" in str(error)
    else:
        raise AssertionError("interchangeable terminal pieces accepted")
    return {"seeds": 42, "pairs": pairs, "canvas_shapes": len(shapes),
            "assembled_masks": len(geometry), "body_masks": len(bodies), "joint_keys": len(keys)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
