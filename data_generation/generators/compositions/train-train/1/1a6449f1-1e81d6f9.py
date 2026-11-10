"""Soluble room dividers: keyed erasure changes the largest sealed interior.

Only standard-library modules and NumPy are used. `solve` reads pixels only;
construction's graph is never passed to it or retained as global state.
"""
from __future__ import annotations

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    peaks = np.flatnonzero(counts == counts.max())
    if len(peaks) != 1:
        raise ValueError("background must be uniquely most frequent")
    return int(peaks[0])


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _view(a, corner):
    """View the selected corner at the upper left; reflections are self-inverse."""
    if corner & 1:
        a = a[::-1, :]
    if corner & 2:
        a = a[:, ::-1]
    return a


def _selector(a):
    bg = _background(a)
    found = []
    h, w = a.shape
    for corner in range(4):
        v = _view(a, corner)
        prefix = np.cumsum(np.cumsum(v != bg, axis=0), axis=1)
        # No hidden bounds on chamber dimensions: two sides plus exterior fit.
        for ch in range(2, h - 1):
            wall = int(v[ch, 0])
            if wall == bg:
                continue
            breaks = np.flatnonzero(v[ch] != wall)
            run = int(breaks[0]) if len(breaks) else w
            for cw in range(2, min(run, w - 1)):
                if prefix[ch - 1, cw - 1] != 1 or not np.all(v[:ch, cw] == wall):
                    continue
                marks = np.argwhere(v[:ch, :cw] != bg)
                if len(marks) != 1:
                    continue
                r, c = map(int, marks[0])
                key = int(v[r, c])
                if key == wall:
                    continue
                pos = (h - 1 - r if corner & 1 else r,
                       w - 1 - c if corner & 2 else c)
                found.append(dict(background=bg, wall=wall, key=key,
                                  position=pos, corner=corner, chamber=(ch, cw)))
    if len(found) != 1:
        raise ValueError("input must have exactly one corner-backed selector")
    return found[0]


def _components(blocked):
    """Return all free components as (boolean mask, touches_canvas_edge)."""
    h, w = blocked.shape
    seen = blocked.copy()
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            cells = []
            stack = [(r, c)]
            seen[r, c] = True
            edge = False
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                edge |= y == 0 or x == 0 or y == h - 1 or x == w - 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            mask = np.zeros((h, w), dtype=bool)
            yy, xx = zip(*cells)
            mask[yy, xx] = True
            result.append((mask, bool(edge)))
    return result


def _barriers(a, wall, bg):
    """Permanent walls plus colored one-cell doors bridging opposing walls.

    Door colors are interpreted geometrically, so the same color may also be
    ordinary payload away from a wall. Background in a bridge is an open door.
    All tests use the permanent-wall mask, never a progressively grown mask.
    """
    permanent = a == wall
    bridges = np.zeros(a.shape, dtype=bool)
    bridges[1:-1, :] |= permanent[:-2, :] & permanent[2:, :]
    bridges[:, 1:-1] |= permanent[:, :-2] & permanent[:, 2:]
    return permanent | (bridges & (a != bg))


def _bounds(mask):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("empty region")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _box_area(mask):
    r0, c0, r1, c1 = _bounds(mask)
    return (r1 - r0) * (c1 - c0)


def _extract(a, mask, bg):
    r0, c0, r1, c1 = _bounds(mask)
    return np.where(mask[r0:r1, c0:c1], a[r0:r1, c0:c1], bg).astype(np.int8)


def _analysis(a):
    d = _selector(a)
    clean = a.copy()
    clean[clean == d["key"]] = d["background"]
    clean[d["position"]] = d["key"]
    sealed = [m for m, edge in _components(_barriers(clean, d["wall"], d["background"])) if not edge]
    if len(sealed) < 2:
        raise ValueError("at least two sealed components must survive")
    areas = [int(m.sum()) for m in sealed]
    maximum = max(areas)
    if areas.count(maximum) != 1:
        raise ValueError("largest sealed component must be unique")
    winner = sealed[areas.index(maximum)]
    return d, clean, sealed, winner


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase the visible key color, then crop the largest sealed free region."""
    a = _grid(input_grid)
    d, clean, _, winner = _analysis(a)
    return _extract(clean, winner, d["background"])


# The functions below build room graphs and materialize them as pixels. They
# are not used by solve or its pixel-based component finder.

def _graph_groups(n, edges):
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    left = set(range(n))
    groups = []
    while left:
        u = min(left)
        left.remove(u)
        group, stack = {u}, [u]
        while stack:
            for v in adjacency[stack.pop()]:
                if v in left:
                    left.remove(v)
                    group.add(v)
                    stack.append(v)
        groups.append(group)
    return groups


def _mesh_edges(nr, nc):
    edges = []
    for r in range(nr):
        for c in range(nc):
            u = r * nc + c
            if c + 1 < nc:
                edges.append((u, u + 1))
            if r + 1 < nr:
                edges.append((u, u + nc))
    return edges


def _opening(rng, length):
    """One or more spaced single-cell doors, each between permanent walls."""
    if length < 3 or rng.random() < 0.5:
        return [int(rng.integers(length))]
    candidates = list(range(length))
    rng.shuffle(candidates)
    selected = []
    for z in candidates:
        if all(abs(z - old) > 1 for old in selected):
            selected.append(z)
    count = int(rng.integers(1, len(selected) + 1))
    return sorted(selected[:count])


def _edge_length(edge, hs, ws):
    u, v = edge
    nc = len(ws)
    return hs[u // nc] if v == u + 1 else ws[u % nc]


def _is_nonrectangular(group, nc):
    rows = [u // nc for u in group]
    cols = [u % nc for u in group]
    return len(group) < (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _group_area(group, base_areas, gates):
    return sum(base_areas[u] for u in group) + sum(
        len(cells) for (u, v), cells in gates.items() if u in group and v in group)


def _outside_sides(group, nr, nc):
    choices = []
    for u in sorted(group):
        r, c = divmod(u, nc)
        if r == 0:
            choices.append((u, "top"))
        if r == nr - 1:
            choices.append((u, "bottom"))
        if c == 0:
            choices.append((u, "left"))
        if c == nc - 1:
            choices.append((u, "right"))
    return choices


def _plan(rng, profile, large_network=False):
    for _ in range(1200):
        if profile == "area":
            # Two complementary L groups. The narrower bounding box can hold
            # more free area because the wide row belongs mostly to that L.
            hs = [int(rng.integers(5, 8)), int(rng.integers(2, 4))]
            c0 = int(rng.integers(3, 6))
            ws = [c0, int(rng.integers(3, 6)), c0 + int(rng.integers(1, 3))]
            if ws[2] <= max(ws[:2]):
                continue
            edges = [(0, 1), (0, 3), (2, 5), (4, 5)]
        elif profile == "leak":
            hs = [int(x) for x in rng.integers(3, 8, size=2)]
            ws = [int(x) for x in rng.integers(3, 8, size=2)]
            edges = []
        else:
            shapes = ((2, 2), (2, 3), (3, 2)) if profile == "merge" else ((2, 3), (3, 2))
            nr, nc = (3, 3) if large_network else _pick(rng, shapes)
            hs = [int(x) for x in rng.integers(2, 7, size=nr)]
            ws = [int(x) for x in rng.integers(2, 7, size=nc)]
            probability = float(rng.uniform(0.25, 0.55))
            edges = [e for e in _mesh_edges(nr, nc) if rng.random() < probability]
        nr, nc = len(hs), len(ws)
        n = nr * nc
        gates = {e: _opening(rng, _edge_length(e, hs, ws)) for e in edges}
        groups = _graph_groups(n, edges)
        base = [hs[r] * ws[c] for r in range(nr) for c in range(nc)]
        areas = [_group_area(g, base, gates) for g in groups]
        leaks = set()
        if profile in ("leak", "combined"):
            largest = max(areas)
            if areas.count(largest) != 1:
                continue
            leak_index = areas.index(largest)
            if not _outside_sides(groups[leak_index], nr, nc):
                continue
            leaks.add(leak_index)
        survivors = [i for i in range(len(groups)) if i not in leaks]
        if len(survivors) < 2:
            continue
        best = max(areas[i] for i in survivors)
        winners = [i for i in survivors if areas[i] == best]
        if len(winners) != 1:
            continue
        wi = winners[0]
        winner = groups[wi]
        if profile in ("area", "merge", "combined") and len(winner) < 2:
            continue
        if profile == "area":
            if winner != {0, 1, 3} or not _is_nonrectangular(winner, nc):
                continue
            winner_box = (sum(hs) + 1) * (ws[0] + ws[1] + 1)
            rival_box = (sum(hs) + 1) * (ws[1] + ws[2] + 1)
            if winner_box >= rival_box or base.index(max(base)) in winner:
                continue
        # Optional extra leaks add variety but never remove the last rival.
        if profile == "combined" and len(survivors) >= 3 and rng.random() < 0.4:
            candidates = [i for i in survivors if i != wi and _outside_sides(groups[i], nr, nc)]
            if candidates:
                leaks.add(_pick(rng, candidates))
        outer = []
        for index in sorted(leaks):
            choices = _outside_sides(groups[index], nr, nc)
            rng.shuffle(choices)
            for u, side in choices[:int(rng.integers(1, min(3, len(choices)) + 1))]:
                length = ws[u % nc] if side in ("top", "bottom") else hs[u // nc]
                outer.append((u, side, _opening(rng, length)))
        return dict(hs=hs, ws=ws, gates=gates, outer=outer, groups=groups,
                    leaks=leaks, winner=winner, areas=areas)
    raise RuntimeError(f"could not sample {profile} room topology")


def _payload(rng, shape, bg, palette):
    h, w = shape
    out = np.full(shape, bg, dtype=np.int8)
    yy, xx = np.indices(shape)
    mode = int(rng.integers(5))
    if mode == 0:
        mask = rng.random(shape) < rng.uniform(0.1, 0.5)
    elif mode == 1:
        mask = np.zeros(shape, dtype=bool)
        ph, pw = int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1))
        r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        mask[r:r + ph, c:c + pw] = True
    elif mode == 2:
        step = int(rng.integers(2, 4))
        mask = (yy if rng.random() < 0.5 else xx) % step == int(rng.integers(step))
    elif mode == 3:
        step = int(rng.integers(2, 5))
        mask = (yy + xx) % step == int(rng.integers(step))
    else:
        mask = np.zeros(shape, dtype=bool)
        if rng.random() < 0.5:
            mask[int(rng.integers(h)), :] = True
        else:
            mask[:, int(rng.integers(w))] = True
    if not mask.any():
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    subset = rng.choice(palette, size=int(rng.integers(1, len(palette) + 1)), replace=False)
    out[mask] = rng.choice(subset, int(mask.sum()))
    return out


def _transform(a, turns, reflect):
    a = np.rot90(a, turns)
    if reflect:
        a = a[:, ::-1]
    return a.copy()


def _make_pair(rng, profile, key, other_key=None, large_network=False):
    for _ in range(100):
        p = _plan(rng, profile, large_network)
        hs, ws = p["hs"], p["ws"]
        nr, nc = len(hs), len(ws)
        excluded = {key} | ({other_key} if other_key is not None else set())
        role_choices = [c for c in range(10) if c not in excluded]
        bg = int(_pick(rng, role_choices))
        wall = int(_pick(rng, [c for c in role_choices if c != bg]))
        pool = [c for c in range(10) if c not in (bg, wall, key)]
        rng.shuffle(pool)
        palette = pool[:int(rng.integers(1, len(pool) + 1))]
        if other_key is not None and other_key not in palette:
            palette[0] = other_key
        ch, cw = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        bh, bw = sum(hs) + nr + 1, sum(ws) + nc + 1
        min_h, min_w = bh + ch + 2, max(bw + 2, cw + 3, 12)
        if min_h > 30 or min_w > 30:
            continue
        h = int(rng.integers(min_h, min(30, min_h + 5) + 1))
        w = int(rng.integers(min_w, min(30, min_w + 6) + 1))
        br, bc = int(rng.integers(ch + 2, h - bh + 1)), int(rng.integers(w - bw + 1))
        a = np.full((h, w), bg, dtype=np.int8)
        # Non-wall clutter is traversable and cannot introduce extra barriers.
        outside = rng.random(a.shape) < rng.uniform(0.0, 0.045)
        a[outside] = rng.choice(palette + [key], int(outside.sum()))
        a[:ch + 1, :cw + 1] = bg
        a[ch, :cw + 1] = wall
        a[:ch, cw] = wall
        marker = (int(rng.integers(ch)), int(rng.integers(cw)))
        a[marker] = key
        a[br:br + bh, bc:bc + bw] = bg
        # ys/xs contain the permanent grid-line coordinates.
        ys, xs = [br], [bc]
        for size in hs:
            ys.append(ys[-1] + size + 1)
        for size in ws:
            xs.append(xs[-1] + size + 1)
        a[ys, bc:bc + bw] = wall
        a[br:br + bh, xs] = wall
        room_cells = []
        for r in range(nr):
            for c in range(nc):
                sl = (slice(ys[r] + 1, ys[r + 1]), slice(xs[c] + 1, xs[c + 1]))
                a[sl] = _payload(rng, (hs[r], ws[c]), bg, palette)
                mask = np.zeros(a.shape, dtype=bool)
                mask[sl] = True
                room_cells.append(mask)
        gate_cells = {}
        for (u, v), positions in p["gates"].items():
            r, c = divmod(u, nc)
            if v == u + 1:
                cells = [(ys[r] + 1 + z, xs[c + 1]) for z in positions]
            else:
                cells = [(ys[r + 1], xs[c] + 1 + z) for z in positions]
            gate_cells[(u, v)] = cells
            for pos in cells:
                a[pos] = key
        for u, side, positions in p["outer"]:
            r, c = divmod(u, nc)
            if side in ("top", "bottom"):
                y = ys[0] if side == "top" else ys[-1]
                cells = [(y, xs[c] + 1 + z) for z in positions]
            else:
                x = xs[0] if side == "left" else xs[-1]
                cells = [(ys[r] + 1 + z, x) for z in positions]
            for pos in cells:
                a[pos] = key
        # Resistant doors are geometrically the same as selected doors, but
        # their color differs from the key. The key must therefore be read to
        # determine topology, not merely to recolor the final payload.
        resistant = other_key if other_key is not None else int(_pick(rng, palette))
        closed_edges = [e for e in _mesh_edges(nr, nc) if e not in p["gates"]]
        border_edges = [e for e in closed_edges if (e[0] in p["winner"]) != (e[1] in p["winner"])]
        chosen_closed = set(e for e in closed_edges if rng.random() < 0.3)
        if border_edges:
            chosen_closed.add(_pick(rng, border_edges))
        for u, v in sorted(chosen_closed):
            r, c = divmod(u, nc)
            positions = _opening(rng, _edge_length((u, v), hs, ws))
            for z in positions:
                pos = (ys[r] + 1 + z, xs[c + 1]) if v == u + 1 else (ys[r + 1], xs[c] + 1 + z)
                a[pos] = resistant
        # Resistant exterior doors on rivals as well as the winner teach that
        # an aperture is not automatically open, without labeling the winner.
        for group_index, group in enumerate(p["groups"]):
            choices = _outside_sides(group, nr, nc)
            if group_index in p["leaks"] or not choices:
                continue
            u, side = _pick(rng, choices)
            rr, cc = divmod(u, nc)
            if side in ("top", "bottom"):
                pos = (ys[0] if side == "top" else ys[-1], xs[cc] + 1 + int(rng.integers(ws[cc])))
            else:
                pos = (ys[rr] + 1 + int(rng.integers(hs[rr])), xs[0] if side == "left" else xs[-1])
            a[pos] = resistant
        # Visible cellwise deletion inside the extracted interior, with a
        # surviving neighbor. This is independent of opening the wall gates.
        chosen_room = int(_pick(rng, sorted(p["winner"])))
        rr, cc = divmod(chosen_room, nc)
        r = int(rng.integers(ys[rr] + 1, ys[rr + 1]))
        c = int(rng.integers(xs[cc] + 1, xs[cc + 1] - 1))
        a[r, c] = key
        a[r, c + 1] = other_key if other_key is not None else _pick(rng, palette)
        # Optional selected-color marks in other rooms are also erased.
        for u in range(nr * nc):
            if u == chosen_room or rng.random() >= 0.5:
                continue
            rr, cc = divmod(u, nc)
            pos = (int(rng.integers(ys[rr] + 1, ys[rr + 1])),
                   int(rng.integers(xs[cc] + 1, xs[cc + 1])))
            a[pos] = key
            # Every room retains a non-key payload, independently of ornaments.
            neighbor = (pos[0], xs[cc] + 1 if pos[1] != xs[cc] + 1 else xs[cc] + 2)
            a[neighbor] = _pick(rng, palette)
        # Construct expected regions from the plan, without a pixel flood fill.
        expected_masks = []
        for i, group in enumerate(p["groups"]):
            if i in p["leaks"]:
                continue
            mask = np.zeros(a.shape, dtype=bool)
            for u in group:
                mask |= room_cells[u]
            for (u, v), cells in gate_cells.items():
                if u in group and v in group:
                    for pos in cells:
                        mask[pos] = True
            expected_masks.append(mask)
        planned_winner = np.zeros(a.shape, dtype=bool)
        for u in p["winner"]:
            planned_winner |= room_cells[u]
        for (u, v), cells in gate_cells.items():
            if u in p["winner"] and v in p["winner"]:
                for pos in cells:
                    planned_winner[pos] = True
        try:
            correct_background = _background(a) == bg
        except ValueError:
            correct_background = False
        if not correct_background:
            continue
        clean = a.copy()
        clean[clean == key] = bg
        clean[marker] = key
        expected = _extract(clean, planned_winner, bg)
        turns = int(rng.integers(4))
        reflect = bool(rng.integers(2))
        a = _transform(a, turns, reflect)
        expected = _transform(expected, turns, reflect)
        try:
            d, _, actual_masks, actual_winner = _analysis(a)
        except ValueError:
            continue
        transformed_masks = [_transform(m, turns, reflect) for m in expected_masks]
        if (d["key"] != key or d["wall"] != wall
                or len(actual_masks) != len(transformed_masks)
                or not all(any(np.array_equal(m, q) for q in actual_masks) for m in transformed_masks)
                or not np.array_equal(actual_winner, _transform(planned_winner, turns, reflect))):
            continue
        return {"input": a, "output": expected}
    raise RuntimeError(f"could not materialize {profile} scene")


def _identity(a):
    return a.shape, a.astype(np.int8).tobytes()


def _witnesses(a, d, sealed, winner):
    # Before solvent action, all geometrically recognized doors are closed.
    # Interior ornaments remain traversable regardless of their color.
    barriers = _barriers(a, d["wall"], d["background"])
    old = [m for m, edge in _components(barriers) if not edge]
    merged = sum(bool(np.any(m & winner)) for m in old) >= 2
    all_sealed = np.logical_or.reduce(sealed)
    leaked = any(np.any(m & ~all_sealed) for m in old)
    mixed = False
    for r, c in np.argwhere(winner & (a == d["key"])):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = int(r + dy), int(c + dx)
            if (0 <= y < a.shape[0] and 0 <= x < a.shape[1] and winner[y, x]
                    and int(a[y, x]) not in (d["background"], d["wall"], d["key"])):
                mixed = True
    box_rival = any(_box_area(m) > _box_area(winner) for m in sealed)
    merged_groups = sum(sum(bool(np.any(old_m & m)) for old_m in old) >= 2 for m in sealed)
    resistant = barriers & (a != d["wall"]) & (a != d["key"])
    return dict(merged=merged, leaked=leaked, mixed=mixed, resistant=bool(resistant.any()),
                box_rival=box_rival, merged_groups=merged_groups, old=old)


def validate_generated_puzzle(puzzle):
    """Validate the family rule and the demonstrations' distinguishing evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, train = set(), []
    without_a_differs = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            ident = _identity(a)
            assert ident not in seen, "duplicate input"
            seen.add(ident)
            d, clean, sealed, winner = _analysis(a)
            assert np.array_equal(b, _extract(clean, winner, d["background"]))
            assert not np.any((b == d["key"]) | (b == d["wall"]))
            assert np.count_nonzero(clean == d["key"]) == 1
            assert clean[d["position"]] == d["key"]
            for m in sealed:
                assert np.any(m & ~np.isin(a, [d["background"], d["wall"], d["key"]])), "room lacks retained content"
            evidence = _witnesses(a, d, sealed, winner)
            assert evidence["mixed"], "must visibly erase inside mixed payload"
            assert evidence["resistant"], "must include a nonselected closed door"
            # Every selected non-key pixel is preserved at its relative crop position.
            r0, c0, r1, c1 = _bounds(winner)
            local = winner[r0:r1, c0:c1]
            original_crop = a[r0:r1, c0:c1]
            assert np.array_equal(b[local & (original_crop != d["key"])],
                                  original_crop[local & (original_crop != d["key"])])
            assert np.all(b[~local] == d["background"])
            assert np.all(b[local & (original_crop == d["key"])] == d["background"])
            if section == "train":
                all_rooms = np.logical_or.reduce(sealed)
                without_a_differs |= not np.array_equal(b, _extract(clean, all_rooms, d["background"]))
                train.append((a, b, d, evidence, winner))
    assert without_a_differs, "retaining all sealed rooms must differ from selecting the largest"
    assert len({d["key"] for _, _, d, _, _ in train}) >= 2
    assert any(e["merged"] and e["box_rival"] and e["merged_groups"] >= 2 for _, _, _, e, _ in train)
    assert any(e["leaked"] for _, _, _, e, _ in train)
    # Omitting key dissolution changes the geometric answer in training.
    without_b_differs = False
    for a, b, d, e, winner in train:
        old = e["old"]
        if old:
            largest = max(old, key=lambda m: int(m.sum()))
            without_b_differs |= not np.array_equal(b, _extract(a, largest, d["background"]))
    assert without_b_differs
    # Each of the first two literal keys is retained as payload under the other.
    for index in range(2):
        a, b, d, _, winner = train[index]
        other = train[1 - index][2]["key"]
        assert np.any(winner & (a == other)) and np.any(b == other)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, with train > test")
    rng = np.random.default_rng(int(seed))
    keys = [int(x) for x in rng.permutation(10)]
    train_keys = keys[:2] + [int(_pick(rng, keys[:6])) for _ in range(num_train - 2)]
    held = [c for c in keys if c not in train_keys]
    test_keys = [int(_pick(rng, held if rng.random() < 0.6 else train_keys)) for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_corner = None
    for index, key in enumerate(train_keys + test_keys):
        section = "train" if index < num_train else "test"
        profile = ("area" if index == 0 else "leak" if index == 1 else
                   "combined" if section == "test" else _pick(rng, ("merge", "leak", "combined")))
        other = train_keys[1 - index] if index < 2 else None
        large_network = section == "test" and (index == num_train or rng.random() < 0.4)
        for _ in range(50):
            pair = _make_pair(rng, profile, key, other,
                              large_network=large_network)
            ident = _identity(pair["input"])
            corner = _selector(pair["input"])["corner"]
            if ident in seen or (index == 1 and corner == first_corner):
                continue
            seen.add(ident)
            if index == 0:
                first_corner = corner
            puzzle[section].append(pair)
            break
        else:
            raise RuntimeError("could not make a distinct informative pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """Bounded 42-seed study plus metamorphic and boundary checks."""
    assert _same_puzzle(generate_puzzle(0), generate_puzzle(0))
    assert not _same_puzzle(generate_puzzle(0), generate_puzzle(1))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ns}")
    shapes, outputs, structures, backgrounds, walls, keys, corners = (set() for _ in range(7))
    area_counts, component_counts, counts = set(), set(), set()
    original_cells = {"train": set(), "test": set()}
    merged = leaked = masked = box_rivals = pairs = 0
    for seed in range(42):
        nt = 2 + seed % 9
        ns = 1 + (seed // 3) % min(4, nt - 1)
        p = generate_puzzle(seed, nt, ns)
        counts.add((nt, ns))
        for section in ("train", "test"):
            for index, pair in enumerate(p[section]):
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(a), b)
                d, _, sealed, winner = _analysis(a)
                e = _witnesses(a, d, sealed, winner)
                original_cells[section].add(len(e["old"]))
                if section == "train":
                    assert len(e["old"]) <= 6
                elif index == 0:
                    assert len(e["old"]) == 9
                shapes.add(a.shape)
                outputs.add(b.shape)
                backgrounds.add(d["background"])
                walls.add(d["wall"])
                keys.add(d["key"])
                corners.add(d["corner"])
                area_counts.add(int(winner.sum()))
                component_counts.add(len(sealed))
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(x), len(roles)) for x in a.ravel())))
                merged += e["merged"]
                leaked += e["leaked"]
                masked += _box_area(winner) > int(winner.sum())
                box_rivals += e["box_rival"]
                pairs += 1
    assert len(shapes) >= 40 and len(outputs) >= 30
    assert len(structures) == pairs
    assert len(backgrounds) == len(walls) == len(keys) == 10
    assert len(corners) == 4 and len(area_counts) >= 40 and len(component_counts) >= 3
    assert min(merged, leaked, masked, box_rivals) >= 42
    # Any palette permutation and dihedral transform preserve the rule.
    a = generate_puzzle(23)["test"][0]["input"]
    expected = solve(a)
    perm = np.array([8, 0, 7, 2, 9, 3, 1, 6, 4, 5], dtype=np.int8)
    for turns in range(4):
        for reflect in (False, True):
            transformed = _transform(perm[a], turns, reflect)
            assert np.array_equal(solve(transformed), _transform(perm[expected], turns, reflect))
    # A non-mesh scene with a wall island. Changing only the marker changes
    # which exterior door dissolves and therefore which room is extracted.
    hand = np.zeros((15, 25), dtype=np.int8)
    hand[2, :3] = 5
    hand[:2, 2] = 5
    hand[0, 0] = 2
    for r, c, h, w in ((4, 1, 8, 8), (4, 11, 6, 6), (10, 18, 5, 6)):
        hand[r, c:c + w] = hand[r + h - 1, c:c + w] = 5
        hand[r:r + h, c] = hand[r:r + h, c + w - 1] = 5
    hand[4, 4] = 3             # closed when key=2, opens when key=3
    hand[4, 13] = 2            # opens when key=2, closed when key=3
    hand[6, 3], hand[6, 4] = 2, 8
    hand[6, 13], hand[12, 20] = 4, 8
    hand[8, 5:7] = 5           # internal wall island, masked out
    expected_hand = np.zeros((6, 6), dtype=np.int8)
    expected_hand[1, 2] = 8
    assert np.array_equal(solve(hand), expected_hand)
    changed_key = hand.copy()
    changed_key[0, 0] = 3
    expected_changed = np.zeros((4, 4), dtype=np.int8)
    expected_changed[1, 1] = 4
    assert np.array_equal(solve(changed_key), expected_changed)
    # Explicitly test four-neighbor leakage, diagonal gaps, and edge walls.
    blocked = np.ones((7, 7), dtype=bool)
    blocked[0, 0] = False       # exterior cell touches center only diagonally
    blocked[1, 1] = False
    blocked[2:5, 2:5] = False
    components = _components(blocked)
    assert sorted(int(m.sum()) for m, edge in components if not edge) == [1, 9]
    blocked[0:3, 4] = False    # a one-cell-wide route to the canvas edge
    components = _components(blocked)
    assert sorted(int(m.sum()) for m, edge in components if not edge) == [1]
    return dict(seeds=42, pairs=pairs, count_combinations=len(counts),
                input_shapes=len(shapes), output_shapes=len(outputs),
                normalized_structures=len(structures), winner_areas=len(area_counts),
                original_cells={s: sorted(v) for s, v in original_cells.items()},
                sealed_counts=sorted(component_counts), merged=merged, leaked=leaked,
                irregular_or_perforated=masked, area_over_box=box_rivals)
