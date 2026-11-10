"""Keyed hoses and a turning switchboard.

Fixed visible roles: 0 background, 8 board rail, 7 divider, 9 hose connector,
3 bead. Four different labels come from 1/2/4/5/6. See task_spec.md.
There is no puzzle state, file I/O, parent import, or learned color dictionary.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


LABELS = (1, 2, 4, 5, 6)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _adj(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]


def _paths(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        stack, component = [start], {start}
        remaining.remove(start)
        while stack:
            for q in _adj(stack.pop(), cells):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        degrees = {p: len(_adj(p, cells)) for p in component}
        ends = sorted(p for p, d in degrees.items() if d == 1)
        if len(ends) != 2 or any(d not in (1, 2) for d in degrees.values()):
            raise ValueError("tracks must be disjoint simple paths")
        order, previous = [ends[0]], None
        while order[-1] != ends[1]:
            choices = [p for p in _adj(order[-1], cells) if p != previous]
            if len(choices) != 1:
                raise ValueError("ambiguous path")
            previous = order[-1]
            order.append(choices[0])
        if len(order) != len(component):
            raise ValueError("cycle")
        result.append(order)
    return result


def _turn_core(core):
    """Simultaneous, agreeing 3x3 writes; labels and breaks cannot be erased."""
    h, w = core.shape
    writes, centers = {}, []
    for r in range(1, h, 2):
        xs = set(map(int, np.flatnonzero(core[r] == 7)))
        for a in sorted(x for x in xs if x - 1 not in xs):
            if not {a, a + 1, a + 2} <= xs or a + 3 in xs:
                raise ValueError("dividers have length three")
            c = a + 1
            if not 4 <= c <= w - 5:
                raise ValueError("divider touches a socket")
            if not np.all(core[[r - 1, r + 1], c - 1:c + 2] == 8):
                raise ValueError("divider must sit between intact rails")
            centers.append((r, c))
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    val = 7 if cc == c else 8
                    if (rr, cc) in writes and writes[rr, cc] != val:
                        raise ValueError("conflicting switch neighborhoods")
                    writes[rr, cc] = val
    if not centers:
        raise ValueError("at least one switch is required")
    out = core.copy()
    for p, val in writes.items():
        out[p] = val
    _paths((out != 0) & (out != 7))
    return out, centers


def _board(grid):
    """8 occurs only on the board, so its bounding box locates the board."""
    loc = np.argwhere(grid == 8)
    if not len(loc):
        raise ValueError("missing board")
    r0, c0 = map(int, loc.min(axis=0))
    r1, c1 = map(int, loc.max(axis=0) + 1)
    crop = grid[r0:r1, c0:c1]
    found = []
    for axis in (0, 1):
        core = crop.copy() if axis == 0 else crop.T.copy()
        h, w = core.shape
        if h < 3 or h % 2 != 1 or w < 9:
            continue
        slots = ((0, 1), (0, w - 3), (h - 1, 1), (h - 1, w - 3))
        labels = []
        check = core.copy()
        for r, c in slots:
            label = int(core[r, c])
            if label not in LABELS or core[r, c + 1] != label:
                break
            labels.append(label)
            check[r, c:c + 2] = 8
        else:
            if len(set(labels)) != 4:
                continue
            if not np.all(np.isin(check[::2], (0, 8))):
                continue
            if not np.all(check[::2, [0, w - 1]] == 8):
                continue
            if not np.all(np.isin(check[1::2], (0, 7))):
                continue
            # Breaks stay inside the two sockets on each rail.
            if np.any(check[::2, :3] != 8) or np.any(check[::2, -3:] != 8):
                continue
            try:
                turned, centers = _turn_core(core)
            except ValueError:
                continue
            def world(r, c):
                return (r0 + r, c0 + c) if axis == 0 else (r0 + c, c0 + r)
            ports = {}
            for label, (r, c) in zip(labels, slots):
                left = c == 1
                tip = world(r, -1 if left else w)
                direction = (0, -1 if left else 1) if axis == 0 else (-1 if left else 1, 0)
                ports[label] = (tip, direction)
            found.append(dict(box=(r0, r1, c0, c1), axis=axis, core=core,
                              turned=turned, centers=centers, ports=ports))
    if len(found) != 1:
        raise ValueError("board is missing or geometrically ambiguous")
    b = found[0]
    if np.any(grid[:r0] == 7) or np.any(grid[r1:] == 7):
        raise ValueError("divider outside board")
    outside = grid.copy()
    outside[r0:r1, c0:c1] = 0
    if np.any(np.isin(outside, (7, 8))):
        raise ValueError("reserved board color outside board")
    b['outside'] = outside
    return b


def _pieces(outside):
    paths = _paths(outside != 0)
    pieces = []
    for path in paths:
        colors = set(int(outside[p]) for p in path) - {3, 9}
        markers = [p for p in path if outside[p] == 9]
        if len(colors) != 1 or not colors <= set(LABELS):
            raise ValueError("one hose label per loose piece")
        if len(markers) != 1 or markers[0] not in (path[0], path[-1]):
            raise ValueError("one terminal connector per hose")
        if len(path) < 4:
            raise ValueError("hose is too short")
        far = path[-1] if markers[0] == path[0] else path[0]
        beads = [p for p in path if outside[p] == 3]
        if beads not in ([], [far]):
            raise ValueError("a loose hose has at most one bead, at its outer tip")
        coords = np.asarray(path, dtype=int)
        low, high = coords.min(axis=0), coords.max(axis=0) + 1
        mask = outside[low[0]:high[0], low[1]:high[1]].copy()
        occupied = set(path)
        # Crops can include another object in their holes; copy only this path.
        mask[:] = 0
        for p in path:
            mask[tuple(np.asarray(p) - low)] = outside[p]
        pieces.append(dict(color=colors.pop(), mask=mask, source=set(path),
                           bead=bool(beads), size=len(path)))
    # Objects have a complete blank moat, also preventing ambiguous segmentation.
    for i, p in enumerate(pieces):
        for q in pieces[i + 1:]:
            if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
                   for a in p['source'] for b in q['source']):
                raise ValueError("loose hoses must be separated")
    return pieces


def _orient(mask, direction):
    candidates = []
    for turn in range(4):
        a = np.rot90(mask, turn).copy()
        tip = tuple(map(int, np.argwhere(a == 9)[0]))
        neighbors = _adj(tip, set(map(tuple, np.argwhere(a != 0))))
        if len(neighbors) == 1 and tuple(np.asarray(neighbors[0]) - tip) == direction:
            candidates.append((a, tip, turn))
    if len(candidates) != 1:
        raise ValueError("connector has no unique orientation")
    return candidates[0]


def _assemble(grid, turn_switches=True):
    b = _board(grid)
    pieces = _pieces(b['outside'])
    r0, r1, c0, c1 = b['box']
    # Input hoses may occupy their eventual output cells, but never the board moat.
    for p in pieces:
        if any(r0 - 1 <= r <= r1 and c0 - 1 <= c <= c1 for r, c in p['source']):
            raise ValueError("loose hose touches the board")
    core = b['turned'] if turn_switches else b['core']
    result = np.zeros_like(grid)
    result[r0:r1, c0:c1] = core if b['axis'] == 0 else core.T
    beads, used = set(), set()
    for p in pieces:
        if p['color'] not in b['ports']:
            continue
        if p['color'] in used:
            raise ValueError("duplicate matching hose")
        used.add(p['color'])
        target, direction = b['ports'][p['color']]
        mask, tip, amount = _orient(p['mask'], direction)
        shift = np.asarray(target) - tip
        for local in np.argwhere(mask != 0):
            dest = tuple(map(int, local + shift))
            if not (0 <= dest[0] < grid.shape[0] and 0 <= dest[1] < grid.shape[1]):
                raise ValueError("docked hose clips boundary")
            if result[dest] != 0:
                raise ValueError("docked hoses overlap")
            val = int(mask[tuple(local)])
            if val == 3:
                beads.add(dest)
                val = p['color']
            result[dest] = val
        p['turn'] = amount
    paths = _paths((result != 0) & (result != 7))
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    if not beads <= endpoints:
        raise ValueError("docked bead is not at an exposed tip")
    b.update(pieces=pieces, used=used, beads=beads, paths=paths)
    return result, b


def _settle(base, b):
    out = base.copy()
    targets, counts = set(), []
    collisions = singles = levels = transfers = 0
    for path in b['paths']:
        beads = b['beads'] & {path[0], path[-1]}
        n = len(beads)
        counts.append(n)
        if path[0][0] == path[-1][0]:
            dest = beads
            levels += bool(n)
        else:
            order = path if path[0][0] > path[-1][0] else path[::-1]
            dest = set(order[:n])
            collisions += n == 2
            singles += n == 1 and dest != beads
            transfers += bool(n and sorted(int(base[p]) for p in dest) != sorted(int(base[p]) for p in beads))
        targets.update(dest)
    for p in targets:
        out[p] = 3
    b.update(targets=targets, counts=counts, collisions=collisions, singles=singles,
             levels=levels, transfers=transfers)
    return out, b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Dock by the current input's labels, turn the switches, drain each path."""
    grid = _grid(input_grid)
    return _settle(*_assemble(grid))[0]


def _core(lanes, width, centers, cuts, labels):
    a = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    a[::2] = 8
    for (r, c), label in zip(((0, 1), (0, width - 3), (len(a) - 1, 1),
                             (len(a) - 1, width - 3)), labels):
        a[r, c:c + 2] = label
    for r, c in centers:
        a[r, c - 1:c + 2] = 7
    for p in cuts:
        a[p] = 0
    return a


def _layout(rng, mode, labels):
    if mode == 'packing':
        width = int(rng.integers(11, 16))
        center = int(rng.integers(5, width - 4))
        return _core(2, width, [(1, center)], [(0, 3)], labels)
    for _ in range(40):
        lanes = 3 if mode == 'single' else int(rng.integers(4, 6) if mode == 'test' else rng.integers(2, 5))
        width = int(rng.integers(11, 16))
        options = list(range(4, width - 4))
        if mode == 'test':
            lo, hi = options[0], options[-1]
            centers = [(2 * i + 1, lo if i % 2 == 0 else hi) for i in range(lanes - 1)]
        elif mode == 'single':
            c = sorted(map(int, rng.choice(options, 2, replace=False)))
            if c[1] - c[0] < 2:
                continue
            if rng.random() < .5:
                c.reverse()
            centers = [(1, c[0]), (3, c[1])]
        else:
            centers = []
            for i in range(lanes - 1):
                pool = [c for c in options if all(abs(c - d) >= 2 for r, d in centers if r == 2 * i - 1)]
                if pool and (not centers or rng.random() < .7) and len(centers) < 2:
                    centers.append((2 * i + 1, int(rng.choice(pool))))
        # Optional extra switch, checked against both neighboring separators.
        if mode == 'test' and rng.random() < .35:
            pool = [(r, c) for r in range(1, 2 * lanes - 2, 2) for c in options
                    if all(abs(c - d) >= (4 if r == rr else 2)
                           for rr, d in centers if abs(r - rr) <= 2)]
            if pool:
                centers.append(pool[int(rng.integers(len(pool)))])
        cuts = []
        for r in range(0, 2 * lanes - 1, 2):
            joins = sorted((c, rr) for rr, c in centers if abs(rr - r) == 1)
            for (a, ra), (b, rb) in zip(joins, joins[1:]):
                if ra == rb:
                    cuts.append((r, int(rng.integers(a + 2, b - 1))))
        if mode not in ('single', 'test'):
            pool = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(3, width - 3)
                    if all(abs(c - d) > 1 for rr, d in centers if abs(rr - r) == 1)
                    and all(abs(c - d) >= 3 for rr, d in cuts if rr == r)]
            if pool and rng.random() < .55:
                cuts.append(pool[int(rng.integers(len(pool)))])
        a = _core(lanes, width, centers, cuts, labels)
        try:
            turned, _ = _turn_core(a)
        except ValueError:
            continue
        paths = _paths((turned != 0) & (turned != 7))
        if mode == 'test':
            corners = {(0, 0), (0, width - 1), (2 * lanes - 2, 0), (2 * lanes - 2, width - 1)}
            if not any(sum(r % 2 for r, c in p) >= 3 and p[0] in corners and p[-1] in corners for p in paths):
                continue
        return a
    raise RuntimeError("bounded board construction failed")


def _hose(rng, label, upper, left, flat=False, radial=None):
    """A procedurally sampled monotone staircase with a two-cell straight neck."""
    horizontal = int(rng.integers(2, 6)) if radial is None else radial
    vertical = 0 if flat else int(rng.integers(0, 4))
    xsign, ysign = (-1 if left else 1), (-1 if upper else 1)
    path = [(0, 0), (0, xsign), (0, 2 * xsign)]
    remaining = ['x'] * (horizontal - 2) + ['y'] * vertical
    rng.shuffle(remaining)
    for step in remaining:
        r, c = path[-1]
        path.append((r + (ysign if step == 'y' else 0), c + (xsign if step == 'x' else 0)))
    if len(path) < 4:
        path.append((path[-1][0], path[-1][1] + xsign))
    coords = np.asarray(path)
    low, high = coords.min(axis=0), coords.max(axis=0) + 1
    mask = np.zeros(tuple(high - low), dtype=np.int8)
    for p in coords:
        mask[tuple(p - low)] = label
    mask[tuple(-low)] = 9
    return mask


def _draw_input(core, pieces, rng, shape, centered=False):
    """Bounded random packing, then a guaranteed eight-slot 30x30 fallback."""
    h, w = core.shape
    for attempt in range(7):
        if attempt == 6:
            shape, centered = (30, 30), True
        if centered:
            r0, c0 = (shape[0] - h) // 2, (shape[1] - w) // 2
        else:
            # Six cells of outward room suffice for every sampled hose.
            low_r, high_r = min(6, (shape[0] - h) // 2), shape[0] - h - min(6, (shape[0] - h) // 2)
            low_c, high_c = min(6, (shape[1] - w) // 2), shape[1] - w - min(6, (shape[1] - w) // 2)
            r0, c0 = int(rng.integers(low_r, high_r + 1)), int(rng.integers(low_c, high_c + 1))
        grid = np.zeros(shape, dtype=np.int8)
        grid[r0:r0 + h, c0:c0 + w] = core
        blocked = np.zeros(shape, dtype=bool)
        blocked[max(0, r0 - 1):r0 + h + 1, max(0, c0 - 1):c0 + w + 1] = True
        order = list(map(int, rng.permutation(len(pieces))))
        order.sort(key=lambda i: -pieces[i].size)
        good = True
        slots = [(r, c) for r in (0, 24) for c in (0, 8, 16, 24)]
        rng.shuffle(slots)
        for rank, i in enumerate(order):
            mask = pieces[i]
            mh, mw = mask.shape
            coords = np.argwhere(mask != 0)
            if attempt == 6:
                top, left = slots[rank]
            else:
                tops, lefts = np.indices((shape[0] - mh + 1, shape[1] - mw + 1))
                rr = tops.ravel()[:, None] + coords[:, 0]
                cc = lefts.ravel()[:, None] + coords[:, 1]
                choices = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
                if not len(choices):
                    good = False
                    break
                choice = int(rng.choice(choices))
                top, left = int(tops.ravel()[choice]), int(lefts.ravel()[choice])
            for local in coords:
                r, c = map(int, local + (top, left))
                grid[r, c] = mask[tuple(local)]
                blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        if good:
            try:
                _assemble(grid)
            except ValueError:
                good = False
            else:
                return grid
        shape = tuple(min(30, n + 2) for n in shape)
    raise RuntimeError("guaranteed packing fallback failed")


def _pair(rng, mode, orientation, absent=None, required_color=None):
    labels = list(map(int, rng.choice(LABELS, 4, replace=False))) if absent is None else list(map(int, rng.permutation([c for c in LABELS if c != absent])))
    core = _layout(rng, mode, labels)
    h, w = core.shape
    route_indices = set()
    if mode == 'packing':
        selected = set(range(4))
    elif mode in ('single', 'test'):
        corners = {(0, 0): 0, (0, w - 1): 1, (h - 1, 0): 2, (h - 1, w - 1): 3}
        turned_core, _ = _turn_core(core)
        route = next(path for path in _paths((turned_core != 0) & (turned_core != 7))
                     if path[0] in corners and path[-1] in corners
                     and (mode != 'test' or sum(r % 2 for r, c in path) >= 3))
        selected = {corners[route[0]], corners[route[-1]]}
        route_indices = selected.copy()
        if mode == 'test':
            selected.update(i for i in range(4) if rng.random() < .5)
        elif rng.random() < .5:
            selected.add(int(rng.choice(sorted(set(range(4)) - selected))))
        if required_color is not None and labels.index(required_color) not in selected:
            old = core.copy()
            i, j = labels.index(required_color), int(rng.choice(sorted(selected)))
            before = labels.copy()
            labels[i], labels[j] = labels[j], labels[i]
            for a, z in zip(before, labels):
                core[old == a] = z
    else:
        selected = set(map(int, rng.choice(4, int(rng.integers(1, 5)), replace=False)))
    route_radials = dict(zip(sorted(route_indices), map(int, rng.choice((3, 4, 5), 2, replace=False)))) if mode == 'test' else {}
    masks = []
    for i in sorted(selected):
        radial = (3 if i == 1 else 5 if i == 3 else None) if mode == 'packing' else None
        radial = route_radials.get(i, radial)
        mask = _hose(rng, labels[i], i < 2, i % 2 == 0,
                     flat=(mode == 'packing' and i == 0), radial=radial)
        masks.append((i, mask))
    # Measure the assembled geometry before choosing meaningful bead witnesses.
    # This temporary input is in memory only; all final labels come from solve.
    board = np.rot90(core, orientation).copy()
    masks = [(i, np.rot90(a, orientation).copy()) for i, a in masks]
    shape = tuple(max(22, n + 12) for n in board.shape)
    shape = tuple(min(30, n) for n in shape)
    preview = np.zeros(shape, dtype=np.int8)
    r0, c0 = (shape[0] - board.shape[0]) // 2, (shape[1] - board.shape[1]) // 2
    preview[r0:r0 + board.shape[0], c0:c0 + board.shape[1]] = board
    b = _board(preview)
    terminal_by_index = {}
    for i, mask in masks:
        target, direction = b['ports'][labels[i]]
        a, tip, _ = _orient(mask, direction)
        shift = np.asarray(target) - tip
        path = _paths(a != 0)[0]
        terminal = path[-1] if a[path[0]] == 9 else path[0]
        terminal_by_index[i] = tuple(map(int, np.asarray(terminal) + shift))
        for local in np.argwhere(a != 0):
            p = tuple(map(int, local + shift))
            if preview[p] != 0:
                raise ValueError("unexpected sampled overlap")
            preview[p] = a[tuple(local)]
    turned = preview.copy()
    turned[r0:r0 + board.shape[0], c0:c0 + board.shape[1]] = b['turned'] if b['axis'] == 0 else b['turned'].T
    paths = _paths((turned != 0) & (turned != 7))
    if mode == 'packing':
        # Orientation is horizontal (0): left upper flat path is level;
        # right upper/lower hoses demonstrate two-bead consolidation.
        loaded = {0, 1, 3}
    elif mode == 'single':
        candidates = [i for i, p in terminal_by_index.items()
                      if any(p in (path[0], path[-1]) and p[0] < max(path[0][0], path[-1][0]) for path in paths)]
        if not candidates:
            raise RuntimeError("no moving single-bead witness")
        loaded = {int(rng.choice(candidates))}
    else:
        probability = float(rng.uniform(.2, .95))
        loaded = {i for i in selected if rng.random() < probability}
        if not loaded:
            loaded.add(int(rng.choice(sorted(selected))))
        if mode == 'test':
            # A live payload, not an empty long route, must exercise the shift.
            higher = min(route_indices, key=lambda i: terminal_by_index[i][0])
            loaded.add(higher)
    loose = []
    rotations = list(map(int, rng.permutation(4)))
    for rank, (i, mask) in enumerate(masks):
        a = mask.copy()
        path = _paths(a != 0)[0]
        far = path[-1] if a[path[0]] == 9 else path[0]
        if i in loaded:
            a[far] = 3
        # In the packing witness, these are all four required docking turns.
        loose.append(np.rot90(a, rotations[rank] if mode == 'packing' else int(rng.integers(4))).copy())
    distractor = next(c for c in LABELS if c not in labels)
    nd = 1 if mode == 'packing' else int(rng.integers(0, 3))
    for j in range(nd):
        if mode == 'packing':
            # Identical geometry to a selected hose rules out shape-only filtering.
            a = masks[int(rng.integers(len(masks)))][1].copy()
            a[np.isin(a, LABELS)] = distractor
        else:
            a = _hose(rng, distractor, bool(rng.integers(2)), bool(rng.integers(2)))
        if rng.random() < .5:
            path = _paths(a != 0)[0]
            a[path[-1] if a[path[0]] == 9 else path[0]] = 3
        loose.append(np.rot90(a, int(rng.integers(4))).copy())
    shape = tuple(min(30, max(n + 12, int(rng.integers(22, 31)))) for n in board.shape)
    inp = _draw_input(board, loose, rng, shape)
    return {'input': inp, 'output': solve(inp)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= int(value) <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    witness = list(map(int, rng.choice(nt, 2, replace=False)))
    absents = list(map(int, rng.choice(LABELS, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            mode = 'test' if split == 'test' else ('packing' if i == witness[0] else 'single' if i == witness[1] else 'general')
            orientation = 0 if mode == 'packing' else int(rng.choice((1, 3))) if mode == 'single' else int(rng.integers(4))
            for _ in range(12):
                absent = absents[0] if mode == 'packing' else absents[1] if mode == 'single' else None
                pair = _pair(rng, mode, orientation, absent,
                             required_color=absents[0] if mode == 'single' else None)
                signature = (pair['input'].shape, pair['input'].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    base, b = _assemble(inp)
    expected, b = _settle(base, b)
    assert np.array_equal(out, expected), "wrong output"
    assert inp.shape == out.shape
    assert int(np.count_nonzero(out == 3)) == len(b['beads']), "matched bead conservation"
    # The full assembled support and its underlying colors survive settling.
    assert np.array_equal(out != 0, base != 0), "shape support changed"
    assert np.all(out[(out != 3)] == base[(out != 3)]), "rail, connector or divider changed"
    assert np.count_nonzero(out == 9) == len(b['used']), "one connector per selected hose"
    assert not np.any(out == next(c for c in LABELS if c not in b['ports'])), "distractor survived"
    # Count routed separator crossings on the cropped board alone.
    cp = _paths((b['turned'] != 0) & (b['turned'] != 7))
    b['span'] = int(max(sum(r % 2 for r, c in path) for path in cp))
    r0, r1, c0, c1 = b['box']
    b['active_span'] = 0
    for path in b['paths']:
        before = b['beads'] & {path[0], path[-1]}
        after = b['targets'] & set(path)
        if before and after != before:
            span = sum(((r - r0) if b['axis'] == 0 else (c - c0)) % 2
                       for r, c in path if r0 <= r < r1 and c0 <= c < c1)
            b['active_span'] = max(b['active_span'], int(span))
    b['shape'] = inp.shape
    b['hose_shapes'] = {_silhouette(p['mask']) for p in b['pieces']}
    return b


def _silhouette(mask):
    return min((np.rot90(mask != 0, t).shape, np.rot90(mask != 0, t).tobytes())
               for t in range(4))


def _compatible_directions(base, b, out):
    compatible = set()
    for dr, dc in DIRS:
        targets = set()
        for path in b['paths']:
            beads = b['beads'] & {path[0], path[-1]}
            a, z = (dr * p[0] + dc * p[1] for p in (path[0], path[-1]))
            if a == z:
                targets.update(beads)
            else:
                order = path if a > z else path[::-1]
                targets.update(order[:len(beads)])
        if targets == set(map(tuple, np.argwhere(out == 3))):
            compatible.add((dr, dc))
    return compatible


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, directions = set(), [], set(DIRS)
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            b = _features(pair)
            a = _grid(pair['input'])
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            if split == 'train':
                training.append(b)
                base, _ = _assemble(a)
                directions &= _compatible_directions(base, b, _grid(pair['output']))
            else:
                assert b['active_span'] >= 3, "test must move beads through at least three switches"
    assert any(b['collisions'] for b in training), "training lacks packing"
    assert any(b['singles'] for b in training), "training lacks single-bead travel"
    assert any(b['levels'] for b in training), "training lacks level retention"
    assert any(b['transfers'] for b in training), "show payload changing carrier label"
    assert {b['axis'] for b in training} == {0, 1}, "show both board axes"
    assert any(len(b['pieces']) > len(b['used']) for b in training), "show filtering"
    assert directions == {(1, 0)}, "training must identify world-down gravity"
    selected_colors = set().union(*(b['used'] for b in training))
    discarded_colors = {p['color'] for b in training for p in b['pieces'] if p['color'] not in b['used']}
    assert selected_colors & discarded_colors, "demonstrate a literal color changing role"
    assert any({_silhouette(p['mask']) for p in b['pieces'] if p['color'] in b['used']}
               & {_silhouette(p['mask']) for p in b['pieces'] if p['color'] not in b['used']}
               for b in training), "show an identically shaped distractor"
    assert any(len(b['used']) < 4 for b in training), "show an empty socket"
    assert set().union(*(set(p['turn'] for p in b['pieces'] if 'turn' in p) for b in training)) == set(range(4))
    # A must change actual bead destinations, beyond merely drawing a turned bar.
    assert any(not np.array_equal(_settle(*_assemble(_grid(pair['input']), False))[0] == 3,
                                        _grid(pair['output']) == 3) for pair in puzzle['train'])
    # B must move the track support; retaining loose hoses cannot give the answer.
    assert any(not np.array_equal(_grid(p['input']) != 0, _grid(p['output']) != 0) for p in puzzle['train'])


def self_test():
    """42 sequential seeds, small retained summaries, no files or execution traces."""
    shapes, hoses, layouts, counts, axes, palettes, spans = set(), set(), set(), set(), set(), set(), set()
    lengths, lanes, dividers, cuts, loads, distractors = set(), set(), set(), set(), set(), set()
    collisions = singles = levels = transfers = overlaps = busy = 0
    signatures = set()
    active_tests = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                b = _features(pair)
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                shapes.add(b['shape'])
                hoses.update(b['hose_shapes'])
                counts.add(len(b['used']))
                axes.add(b['axis'])
                palettes.add(tuple(b['ports']))
                spans.add(b['span'])
                if split == 'test':
                    active_tests.add(b['active_span'])
                lengths.update(p['size'] for p in b['pieces'])
                lanes.add((b['core'].shape[0] + 1) // 2)
                dividers.add(len(b['centers']))
                cuts.add(int(np.count_nonzero(b['core'][::2] == 0)))
                loads.update(b['counts'])
                distractors.add(len(b['pieces']) - len(b['used']))
                collisions += b['collisions']
                singles += b['singles']
                levels += b['levels']
                transfers += b['transfers']
                footprints = [set((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)) for r, c in b['centers']]
                overlaps += any(a & z for i, a in enumerate(footprints) for z in footprints[i + 1:])
                busy += any(sum(rr == r for rr, c in b['centers']) > 1 for r in range(1, len(b['core']), 2))
                core = b['core'].copy()
                core[np.isin(core, LABELS)] = 1
                layouts.add((core.shape, core.tobytes()))
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(hoses) >= 60 and len(layouts) >= 40
    assert counts == {1, 2, 3, 4} and axes == {0, 1}
    assert len(palettes) >= 20 and {1, 2, 3, 4} <= spans
    assert active_tests and min(active_tests) >= 3
    assert loads == {0, 1, 2} and distractors == {0, 1, 2}
    assert len(lengths) >= 5 and lanes == {2, 3, 4, 5} and overlaps and busy
    first, again = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(7000 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = 7
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("accepted damaged output")
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'hose_masks': len(hoses), 'board_layouts': len(layouts), 'match_counts': sorted(counts),
            'board_axes': sorted(axes), 'socket_assignments': len(palettes), 'switch_spans': sorted(spans),
            'hose_lengths': sorted(lengths), 'lanes': sorted(lanes), 'divider_counts': sorted(dividers),
            'break_counts': sorted(cuts), 'path_bead_counts': sorted(loads), 'distractor_counts': sorted(distractors),
            'collision_paths': collisions, 'moving_single_paths': singles, 'loaded_level_paths': levels,
            'carrier_transfer_paths': transfers, 'pairs_with_overlaps': overlaps, 'pairs_with_busy_separators': busy,
            'active_test_spans': sorted(active_tests)}
