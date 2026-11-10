"""Shoreline docking: topological ports and local color compasses.

Only the standard library and NumPy are used. Construction never supplies state
to solve(); all roles, shores, ports and rigid placements come from the input.
No files are written by this module, including by self_test().
"""
from __future__ import annotations

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g


def _flood(allowed, starts):
    h, w = allowed.shape
    result = np.zeros_like(allowed, dtype=bool)
    stack = []
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and allowed[r, c] and not result[r, c]:
            result[r, c] = True
            stack.append((int(r), int(c)))
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not result[nr, nc]:
                result[nr, nc] = True
                stack.append((nr, nc))
    return result


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in N8 if diagonal else N4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _mask(shape, cells):
    result = np.zeros(shape, bool)
    for p in cells:
        result[p] = True
    return result


def _pose(cells, t):
    result = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result.append((r, c, v))
    r0, c0 = min(r for r, c, v in result), min(c for r, c, v in result)
    return {(r - r0, c - c0): int(v) for r, c, v in result}


def _shift(cells, r, c):
    return {(x + r, y + c): v for (x, y), v in cells.items()}


def _signature(cells):
    return tuple(sorted(_pose(cells, 0).items()))


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _label_normals(cells, shore_colors):
    """A pin lies on exactly one extreme side of its whole piece."""
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    labels = {}
    for (r, c), value in cells.items():
        if value not in shore_colors:
            continue
        directions = [d for d, yes in (((-1, 0), r == r0), ((1, 0), r == r1),
                                       ((0, -1), c == c0), ((0, 1), c == c1)) if yes]
        if len(directions) != 1:
            return None
        labels[r, c] = directions[0]
    return labels


def _topology(walls, marker):
    selected = _flood(~walls, [marker])
    allowed = np.pad(~selected, 1, constant_values=True)
    outside = _flood(allowed, [(0, 0)])
    h, w = walls.shape
    rim = np.zeros_like(walls)
    for dr, dc in N8:
        rim |= outside[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return selected, outside[1:-1, 1:-1], rim & selected


def _parse(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > max(n for i, n in enumerate(counts) if i != bg)
    base = int(g[0, 0])
    assert base != bg and all(int(g[p]) == base for p in ((0, -1), (-1, 0), (-1, -1)))
    shore = set(map(int, np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))))
    assert bg not in shore and len(shore) >= 3
    # Shore colors on loose pieces are pins. Structural components, including
    # enclosed islands, always contain the primary corner color; pins do not.
    walls = np.zeros(g.shape, bool)
    for cells in _components(np.isin(g, list(shore))):
        if any(g[p] == base for p in cells):
            for p in cells:
                walls[p] = True
    assert walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all()
    objects = _components((g != bg) & ~walls, diagonal=True)
    sources, monochrome = [], []
    for cells in objects:
        colored = {p: int(g[p]) for p in cells}
        if len(set(colored.values())) == 1:
            monochrome.append(cells)
        else:
            sources.append(colored)
    assert sources
    source_colors = set().union(*(set(p.values()) for p in sources))
    registration_candidates = {int(g[min(p)]) for p in monochrome} & (source_colors - shore)
    assert len(registration_candidates) == 1
    reg = registration_candidates.pop()
    targets = [p for p in monochrome if int(g[min(p)]) == reg]
    seeds = [p for p in monochrome if int(g[min(p)]) != reg]
    assert len(seeds) == 1 and len(seeds[0]) == 1
    marker = next(iter(seeds[0]))
    marker_color = int(g[marker])
    assert counts[marker_color] == 1 and marker_color not in source_colors | shore
    bodies = source_colors - shore - {reg}
    assert len(bodies) == 1
    body = bodies.pop()
    for piece in sources:
        assert {body, reg} <= set(piece.values())
        labels = _label_normals(piece, shore)
        assert labels and len(labels) >= 2 and base not in piece.values()
        assert len(_components(_mask(g.shape, piece))) == 1
    selected, outside, rim = _topology(walls, marker)
    active = [i for i, target in enumerate(targets)
              if all(selected[p] for p in target) and any(rim[p] for p in target)]
    assert len(active) == len(sources), "one loose piece per active external-rim dock"
    return dict(grid=g, bg=bg, base=base, shore=shore, walls=walls,
                reg=reg, body=body, marker=marker, marker_color=marker_color,
                sources=sources, targets=targets, selected=selected,
                outside=outside, rim=rim, active=active)


def _candidates(source, info, use_labels=True, use_topology=True):
    h, w = info['grid'].shape
    choices, seen = [], set()
    target_indices = info['active'] if use_topology else range(len(info['targets']))
    all_marks = set().union(*info['targets'])
    for t in range(8):
        cells = _pose(source, t)
        marks = {p for p, v in cells.items() if v == info['reg']}
        labels = _label_normals(cells, info['shore'])
        if not labels:
            continue
        for j in target_indices:
            target = info['targets'][j]
            if len(marks) != len(target):
                continue
            a, b = min(marks), min(target)
            dr, dc = b[0] - a[0], b[1] - a[1]
            if {(r + dr, c + dc) for r, c in marks} != target:
                continue
            placed = _shift(cells, dr, dc)
            if any(not (0 <= r < h and 0 <= c < w) or info['walls'][r, c]
                   or (use_topology and not info['selected'][r, c]) for r, c in placed):
                continue
            if info['marker'] in placed or (set(placed) & (all_marks - target)):
                continue
            if use_labels:
                valid = True
                for (r, c), (nr, nc) in labels.items():
                    q = r + dr + nr, c + dc + nc
                    if (not (0 <= q[0] < h and 0 <= q[1] < w)
                            or not info['walls'][q] or int(info['grid'][q]) != cells[r, c]
                            or (use_topology and not info['outside'][q])):
                        valid = False
                        break
                if not valid:
                    continue
            sig = j, tuple(sorted(placed.items()))
            if sig not in seen:
                seen.add(sig)
                choices.append((j, placed))
    return choices


def _answers(info, use_labels=True, limit=2):
    choices = [_candidates(p, info, use_labels) for p in info['sources']]
    order = sorted(range(len(choices)), key=lambda j: len(choices[j]))
    results = {}

    def search(i, used, occupied, parts):
        if len(results) >= limit:
            return
        if i == len(order):
            results[tuple(sorted(occupied.items()))] = list(parts)
            return
        for target, cells in choices[order[i]]:
            if target not in used and not (occupied.keys() & cells.keys()):
                search(i + 1, used | {target}, {**occupied, **cells}, parts + [cells])

    search(0, set(), {}, [])
    return list(results.values())


def _render(info, parts):
    out = info['grid'].copy()
    for piece in info['sources']:
        for p in piece:
            out[p] = info['bg']
    # Inactive registration marks stay as they are. Active ones are overlaid.
    for piece in parts:
        for p, v in piece.items():
            out[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover a unique pixel answer using only the supplied grid."""
    info = _parse(input_grid)
    answers = _answers(info)
    if len(answers) != 1:
        raise ValueError("docking constraints do not specify one pixel output")
    return _render(info, answers[0])


def _piece(rng, style, colors, pins, symmetric_marks=False):
    _, _, body, reg, _ = colors[:5]
    h = int(rng.integers(3, 6))
    w = int(rng.integers(4, 7) if style == 'edge' else rng.integers(3, 6))
    labels = {(0, 1): pins[0], (0, w - 2) if style == 'edge' else (1, 0): pins[1]}
    marks = {(0, 0)}
    size = int(rng.integers(1, 3) if symmetric_marks else rng.integers(1, 5))
    for _ in range(size - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in marks for dr, dc in N8
                           if 0 <= r + dr < min(3, h) and 0 <= c + dc < min(3, w)}
                          - marks - labels.keys())
        if frontier:
            marks.add(frontier[int(rng.integers(len(frontier)))])
    # Start with a rectangle, then remove independently chosen cells while
    # retaining connectivity, registration, pins, and both bounding extents.
    cells = {(r, c): body for r in range(h) for c in range(w)}
    cells.update({p: reg for p in marks})
    cells.update(labels)
    protected = marks | labels.keys() | {(h - 1, w - 1)}
    candidates = sorted(set(cells) - protected)
    rng.shuffle(candidates)
    for p in candidates:
        if rng.random() < .65 and len(cells) > len(protected) + 3:
            trial = set(cells) - {p}
            if len(_components(_mask((h, w), trial))) == 1:
                del cells[p]
    assert _label_normals(cells, set(colors[5:]))
    return cells


def _base_scene(rng, colors, test, witness):
    bg, base = colors[:2]
    open_scene = not witness and rng.random() < .25
    low_h, low_w = (18, 18) if open_scene and not test else (25, 23)
    h, w = int(rng.integers(low_h, 31)), int(rng.integers(low_w, 31))
    g = np.full((h, w), bg, dtype=np.int8)
    g[[0, -1], :] = base
    g[:, [0, -1]] = base
    cut = h // 2 if open_scene else int(rng.integers(11, h - 10))
    if not open_scene:
        g[cut, :] = base
    # A branching divider may make a third compartment, independently of ports.
    if not open_scene and rng.random() < .35:
        col = int(rng.integers(7, w - 6))
        g[cut:, col] = base
    # Attached masses create local corners and uneven shore profiles.
    for _ in range(int(rng.integers(2 if test else 0, 6 if test else 4))):
        edge = int(rng.integers(4))
        length, depth = int(rng.integers(1, 6)), int(rng.integers(1, 4))
        if edge < 2:
            c = int(rng.integers(2, max(3, w - length - 1)))
            r = 0 if edge == 0 else (h - depth if open_scene else cut - depth + 1)
            g[r:r + depth, c:c + length] = base
        else:
            r = int(rng.integers(2, h - depth - 1))
            c = 0 if edge == 2 else w - length
            g[r:r + depth, c:c + length] = base
    walls = g == base
    regions = _components(~walls)
    eligible = [p for p in regions if len(p) >= 210
                and max(r for r, c in p) - min(r for r, c in p) >= 8
                and max(c for r, c in p) - min(c for r, c in p) >= 10]
    if not eligible:
        return None
    if witness:
        areas = [len(p) for p in regions]
        eligible = [p for p in eligible if (len(p) < max(areas) if witness == 1
                                          else len(p) > min(areas))]
        if not eligible:
            return None
        region = min(eligible, key=len) if witness == 1 else max(eligible, key=len)
    else:
        region = eligible[int(rng.integers(len(eligible)))]
    selected = _mask(g.shape, region)
    # All shore identities are visibly present on the outer frame. Their edge
    # positions are independent of their uses at local docks.
    edges = [(0, c) for c in range(1, w - 1)] + [(h - 1, c) for c in range(1, w - 1)]
    edges += [(r, 0) for r in range(1, h - 1)] + [(r, w - 1) for r in range(1, h - 1)]
    rng.shuffle(edges)
    for color, p in zip(colors[5:], edges):
        g[p] = color
    return g, walls, selected


def _fit_locations(cells, selected, walls, blocked, labels, outside):
    # Reject missing shore contact before allocating a translated footprint.
    h, w = selected.shape
    ph, pw = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
    for r in range(1, h - ph):
        for c in range(1, w - pw):
            if any(not walls[x + r + dr, y + c + dc]
                   or not outside[x + r + dr, y + c + dc]
                   for (x, y), (dr, dc) in labels.items()):
                continue
            placed = _shift(cells, r, c)
            if set(placed) & blocked:
                continue
            if all(selected[p] and not walls[p] for p in placed):
                yield r, c, placed


def _construct(rng, colors, count, test, witness):
    scene = _base_scene(rng, colors, test, witness)
    if scene is None:
        return None
    g, walls, selected = scene
    bg, base, body, reg, marker_color = colors[:5]
    pin_colors = colors[5:]
    # A floating island supports a locally perfect edge dock. It must be
    # ignored because its complement component cannot reach the outside.
    edge_piece = _piece(rng, 'edge', colors, pin_colors[:2])
    eh, ew = max(r for r, c in edge_piece) + 1, max(c for r, c in edge_piece) + 1
    decoy_marks, reserved = {}, set()
    has_island_dock = bool(witness or rng.random() < .7)
    if has_island_dock:
        island_places = []
        for r in range(3, g.shape[0] - eh - 3):
            for c in range(3, g.shape[1] - ew - 3):
                if selected[r - 2:r + eh + 3, c - 2:c + ew + 2].all():
                    island_places.append((r, c))
        if not island_places:
            return None
        r, c = island_places[int(rng.integers(len(island_places)))]
        island_height = int(rng.integers(1, 3))
        g[r:r + island_height, c:c + ew] = base
        walls[r:r + island_height, c:c + ew] = True
        selected[walls] = False
        decoy = _shift(edge_piece, r + island_height, c)
        reserved = _halo(decoy)
        edge_labels = _label_normals(edge_piece, set(pin_colors))
        for p, d in edge_labels.items():
            q = r + island_height + p[0] + d[0], c + p[1] + d[1]
            g[q] = edge_piece[p]
        decoy_marks = {p: reg for p, v in decoy.items() if v == reg}
    # Other islands vary independently and need not carry a registration mark.
    for _ in range(int(rng.integers(3))):
        sh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        widths = [sw] + [int(rng.integers(1, sw + 1)) for _ in range(sh - 1)]
        shape = {(r, c) for r in range(sh) for c in range(widths[r])}
        # Row prefixes share a left spine and cannot make detached pixels.
        shape = {p: base for p in shape}
        locations = []
        for rr in range(3, g.shape[0] - sh - 2):
            for cc in range(3, g.shape[1] - sw - 2):
                points = set(_shift(shape, rr, cc))
                wide_halo = _halo(_halo(points))
                if not (wide_halo & reserved) and all(selected[p] for p in wide_halo):
                    locations.append(points)
        if locations:
            points = locations[int(rng.integers(len(locations)))]
            for p in points:
                g[p], walls[p], selected[p] = base, True, False
    seed_probe = min(map(tuple, np.argwhere(selected).tolist()))
    selected, outside, rim = _topology(walls, seed_probe)
    if any(rim[p] or not selected[p] for p in decoy_marks):
        return None
    region_sizes = [len(p) for p in _components(~walls)]
    if witness == 1 and not selected.sum() < max(region_sizes):
        return None
    if witness == 2 and not selected.sum() > min(region_sizes):
        return None
    pieces = [edge_piece]
    # Each piece has a different unordered pair of shore identities. This can
    # resolve repeated registration geometries without a count dictionary.
    pin_pairs = [(pin_colors[0], pin_colors[1]), (pin_colors[0], pin_colors[2]),
                 (pin_colors[1], pin_colors[2])]
    for i in range(1, count):
        style = 'corner' if i == 1 else str(rng.choice(['edge', 'corner']))
        pieces.append(_piece(rng, style, colors, pin_pairs[i], symmetric_marks=witness))
    finals, coast_assignments = [], {}
    blocked = _halo(decoy_marks)
    for piece in pieces:
        options = []
        for t in range(8):
            posed = _pose(piece, t)
            labels = _label_normals(posed, set(pin_colors))
            for rr, cc, placed in _fit_locations(posed, selected, walls, blocked, labels, outside):
                marks = {p for p, v in placed.items() if v == reg}
                if not any(rim[p] for p in marks):
                    continue
                coast = {}
                for p, d in labels.items():
                    q = p[0] + rr + d[0], p[1] + cc + d[1]
                    if (not walls[q] or not outside[q] or q in ((0, 0), (0, g.shape[1] - 1),
                                                               (g.shape[0] - 1, 0), (g.shape[0] - 1, g.shape[1] - 1))
                            or q in coast_assignments and coast_assignments[q] != posed[p]):
                        break
                    coast[q] = posed[p]
                else:
                    options.append((placed, coast))
        if not options:
            return None
        final, coast = options[int(rng.integers(len(options)))]
        finals.append(final)
        coast_assignments.update(coast)
        blocked.update(_halo(final))
    for p, v in coast_assignments.items():
        g[p] = v
    # Paint decoy and active registration marks. Another congruent mark is
    # placed across the divider, with a pose independent of its source.
    targets = dict(decoy_marks)
    for final in finals:
        targets.update({p: reg for p, v in final.items() if v == reg})
    foreign_pattern = _pose({p: reg for p, v in pieces[int(rng.integers(len(pieces)))].items()
                             if v == reg}, int(rng.integers(8)))
    foreign = []
    for rr in range(2, g.shape[0] - 3):
        for cc in range(2, g.shape[1] - 3):
            marks = _shift(foreign_pattern, rr, cc)
            if all(0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]
                   and not walls[p] and not selected[p] for p in _halo(marks)):
                foreign.append(marks)
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        choices = [marks for marks in foreign if not (set(marks) & _halo(targets))]
        if choices:
            targets.update(choices[int(rng.integers(len(choices)))])
    # Source objects are scattered throughout the free canvas. Their old cells
    # may intersect final body cells; all originals are erased before docking.
    occupied = _halo(targets)
    source_blocked = occupied | _halo(set(map(tuple, np.argwhere(walls).tolist())))
    sources = []
    for i, piece in enumerate(pieces):
        transforms = list(map(int, rng.permutation(8)))
        if witness and i == 1:
            transforms = [1, 2, 3] if witness == 1 else [4, 5, 6, 7]
            rng.shuffle(transforms)
        options = []
        for t in transforms:
            posed = _pose(piece, t)
            ph, pw = max(r for r, c in posed) + 1, max(c for r, c in posed) + 1
            for rr in range(2, g.shape[0] - ph - 1):
                for cc in range(2, g.shape[1] - pw - 1):
                    placed = _shift(posed, rr, cc)
                    if placed.keys() & source_blocked:
                        continue
                    options.append(placed)
            if options:
                break
        if not options:
            return None
        source = options[int(rng.integers(len(options)))]
        sources.append(source)
        occupied.update(_halo(source))
        source_blocked.update(_halo(source))
    final_cells = set().union(*(set(p) for p in finals))
    marker_options = [tuple(map(int, p)) for p in np.argwhere(selected)
                      if tuple(p) not in occupied | _halo(final_cells)]
    if not marker_options:
        return None
    marker = marker_options[int(rng.integers(len(marker_options)))]
    inp, out = g.copy(), g.copy()
    for p, v in targets.items():
        inp[p] = out[p] = v
    for source in sources:
        for p, v in source.items():
            inp[p] = v
    for final in finals:
        for p, v in final.items():
            out[p] = v
    inp[marker] = out[marker] = marker_color
    # Equivariant scene transforms change the divider axis and every local
    # compass together. The solver has no preferred global direction.
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    inp, out = inp.copy(), out.copy()
    try:
        info = _parse(inp)
        answers = _answers(info)
        if len(answers) != 1 or not np.array_equal(_render(info, answers[0]), out):
            return None
        if witness and len(_answers(info, use_labels=False)) < 2:
            return None
        # Prove that the island decoy would be a usable dock if topology were
        # ignored, rather than merely putting an arbitrary dot near a hole.
        inactive = set(range(len(info['targets']))) - set(info['active'])
        if has_island_dock and not any(
                j in inactive and all(info['selected'][p] for p in info['targets'][j])
                for source in info['sources'] for j, placed in
                _candidates(source, info, use_topology=False)):
            return None
    except (AssertionError, ValueError):
        return None
    return {'input': inp, 'output': out}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 2, 10, 'num_train')
    ntest = _count(num_test, 1, 4, 'num_test')
    if ntrain <= ntest:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shared_palette = bool(rng.integers(2))
    palette = list(map(int, rng.choice(10, 8, replace=False)))
    puzzle, seen, first_side = {'train': [], 'test': []}, set(), None
    for split, number in (('train', ntrain), ('test', ntest)):
        for i in range(number):
            witness = i + 1 if split == 'train' and i < 2 else 0
            count = (2 if witness else 3 if split == 'test' and i == 0 else
                     int(rng.integers(2, 4) if split == 'test' else rng.integers(1, 3)))
            colors = palette if shared_palette else list(map(int, rng.choice(10, 8, replace=False)))
            for _ in range(100):
                pair = _construct(rng, colors, count, split == 'test', witness)
                if pair is not None and _key(pair['input']) not in seen:
                    if witness:
                        info = _parse(pair['input'])
                        center = np.argwhere(info['selected']).mean(axis=0)
                        delta = (center + .5) / np.array(info['grid'].shape) - .5
                        axis = int(np.argmax(np.abs(delta)))
                        side = axis, bool(delta[axis] > 0)
                        if witness == 2:
                            desired = first_side[0], not first_side[1]
                            # Choose the second witness's global pose so its
                            # selected side is opposite the first's. This
                            # changes no local rule and avoids extra retries.
                            while side != desired:
                                pair = {k: np.rot90(v).copy() for k, v in pair.items()}
                                side = (1, side[1]) if side[0] == 0 else (0, not side[1])
                        if witness == 1:
                            first_side = side
                    break
            else:
                raise RuntimeError('bounded shoreline construction exhausted 100 attempts')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _features(pair):
    info = _parse(pair['input'])
    answers = _answers(info)
    assert len(answers) == 1
    parts = answers[0]
    center = np.argwhere(info['selected']).mean(axis=0)
    delta = (center + .5) / np.array(info['grid'].shape) - .5
    axis = int(np.argmax(np.abs(delta)))
    inactive = set(range(len(info['targets']))) - set(info['active'])
    hole_viable = set()
    islands = sum(not any(info['outside'][p] for p in cells) for cells in _components(info['walls']))
    for source in info['sources']:
        for j, placed in _candidates(source, info, use_topology=False):
            if j in inactive and all(info['selected'][p] for p in info['targets'][j]):
                hole_viable.add(j)
    return dict(shape=tuple(info['grid'].shape), count=len(parts),
                compartments=len(_components(~info['walls'])),
                selected_area=int(info['selected'].sum()),
                smaller=any(len(p) > info['selected'].sum() for p in _components(~info['walls'])),
                larger=any(len(p) < info['selected'].sum() for p in _components(~info['walls'])),
                selected_side=(axis, bool(delta[axis] > 0)),
                hole_decoys=len(hole_viable),
                islands=islands,
                foreign_decoys=sum(not any(info['selected'][p] for p in info['targets'][j]) for j in inactive),
                compass_needed=len(_answers(info, use_labels=False)) > 1,
                palette=(info['bg'], info['base'], info['body'], info['reg'], info['marker_color']),
                piece_shapes=tuple((max(r for r, c in _pose(p, 0)) + 1,
                                    max(c for r, c in _pose(p, 0)) + 1) for p in parts),
                patterns=tuple(tuple(sorted(_pose({p: 1 for p, v in part.items() if v == info['reg']}, 0)))
                               for part in parts),
                pin_directions=tuple(len(set(_label_normals(p, info['shore']).values())) for p in parts))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            before = inp.copy()
            info = _parse(inp)
            answers = _answers(info)
            assert len(answers) == 1 and np.array_equal(out, _render(info, answers[0]))
            assert np.array_equal(inp, before) and not np.array_equal(inp, out)
            assert np.array_equal(out[info['walls']], inp[info['walls']])
            assert out[info['marker']] == info['marker_color']
            assert all(out[p] == info['reg'] for target in info['targets'] for p in target)
            # Complete rigid pieces, including the pins and gaps, are preserved.
            assert sorted(min(_signature(_pose(p, t)) for t in range(8)) for p in info['sources']) == \
                   sorted(min(_signature(_pose(p, t)) for t in range(8)) for p in answers[0])
            features = _features({'input': inp, 'output': out})
            if split == 'train':
                training.append(features)
    assert any(f['compass_needed'] for f in training)
    assert any(f['hole_decoys'] for f in training)
    assert any(f['foreign_decoys'] for f in training)
    assert {1, 2} <= {d for f in training for d in f['pin_directions']}
    assert any(f['smaller'] for f in training) and any(f['larger'] for f in training)
    assert any((a['selected_side'][0] == b['selected_side'][0]
                and a['selected_side'][1] != b['selected_side'][1])
               for a in training for b in training)


def self_test():
    """Bounded 42-seed audit plus input-only metamorphic and contract checks."""
    signatures, shapes, palettes, patterns, piece_shapes = set(), set(), set(), set(), set()
    counts, compartments, selections = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        signatures.add(signature)
        local = []
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                f = _features(pair)
                local.append(f)
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                patterns.update(f['patterns'])
                piece_shapes.update(f['piece_shapes'])
                counts.add(f['count'])
                compartments.add(f['compartments'])
                selections.add(f['smaller'])
        assert len({f['shape'] for f in local}) >= 2
        if seed < 3:
            sample = puzzle['train'][0]
            # All eight global symmetries must transform the answer identically.
            for rotation in range(4):
                for mirror in (False, True):
                    inp, out = np.rot90(sample['input'], rotation), np.rot90(sample['output'], rotation)
                    if mirror:
                        inp, out = np.fliplr(inp), np.fliplr(out)
                    assert np.array_equal(solve(inp.copy()), out)
            # Moving the marker inside its compartment leaves all docking intact.
            info = _parse(sample['input'])
            for p in map(tuple, np.argwhere(info['selected'] & (sample['input'] == info['bg'])).tolist()):
                if p in _halo(set().union(*(set(x) for x in info['sources'] + info['targets']))):
                    continue
                if sample['output'][p] != info['bg']:
                    continue
                moved, expected = sample['input'].copy(), sample['output'].copy()
                moved[info['marker']] = expected[info['marker']] = info['bg']
                moved[p] = expected[p] = info['marker_color']
                assert np.array_equal(solve(moved), expected)
                break
            # Role colors are semantic: a global permutation commutes with solve.
            perm = np.random.default_rng(seed + 500).permutation(10)
            assert np.array_equal(solve(perm[sample['input']]), perm[sample['output']])
    assert len(signatures) == 42 and len(shapes) >= 25 and len(palettes) >= 20
    assert len(patterns) >= 12 and len(piece_shapes) >= 12
    assert counts == {1, 2, 3} and {1, 2, 3} <= compartments and selections == {False, True}
    first, second = generate_puzzle(77, 2, 1), generate_puzzle(77, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'grid_shapes': len(shapes),
            'palettes': len(palettes), 'registration_poses': len(patterns),
            'piece_shapes': len(piece_shapes), 'piece_counts': sorted(counts),
            'compartments': sorted(compartments)}
