"""Reassemble a circuit board by silhouette and mating wire ports, then paint wells.

Only observable input geometry enters the solver. Generation constructs a board
and independently labels its wells, then checks all full-width fragment orders.
"""
from collections import deque
from itertools import combinations
import json

import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _components(cells, steps=D4):
    # Native integers make the small-grid flood fill much cheaper than doing
    # every neighbor addition and hash through NumPy scalar objects.
    left = {(int(r), int(c)) for r, c in cells}
    result = []
    while left:
        root = min(left)
        left.remove(root)
        part, queue = {root}, [root]
        for r, c in queue:
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    part.add(p)
                    queue.append(p)
        result.append(part)
    return result


def _neighbors(cells, steps=D4):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int64, copy=True)


def _parse(value):
    a = _grid(value)
    histogram = np.bincount(a.ravel(), minlength=10)
    assert np.count_nonzero(histogram == histogram.max()) == 1, 'Exterior color must be uniquely modal'
    bg = int(histogram.argmax())
    # The rim is the only color confined to an almost-complete large rectangle.
    rims = []
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(a == color)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 6 or c1 - c0 < 6:
            continue
        on_perimeter = ((coords[:, 0] == r0) | (coords[:, 0] == r1) |
                        (coords[:, 1] == c0) | (coords[:, 1] == c1))
        perimeter_size = 2 * (r1 - r0 + c1 - c0)
        corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
        if np.all(on_perimeter) and len(coords) >= .85 * perimeter_size and all(a[p] == color for p in corners):
            perimeter = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1)}
            rims.append((color, perimeter))
    assert len(rims) == 1, 'Expected one recognizable circuit-board rim'
    rim, perimeter = rims[0]
    gate_colors = {int(a[p]) for p in perimeter if a[p] != rim}
    assert len(gate_colors) == 1 and bg not in gate_colors
    wire = gate_colors.pop()
    groups = _components(map(tuple, np.argwhere(a != bg)))
    fixed = next(g for g in groups if any(a[p] == rim for p in g))
    pieces = [g for g in groups if g is not fixed]
    assert 2 <= len(pieces) <= 5
    # Rotate the whole scene for analysis until the fixed jagged anchor is atop
    # the board. This is a coordinate change, never a rotation of a moved tile.
    canonical = []
    fixed_mask = np.zeros_like(a, dtype=bool)
    for p in fixed:
        fixed_mask[p] = True
    for rotation in range(4):
        b, mask = np.rot90(a, rotation), np.rot90(fixed_mask, rotation)
        rc = np.argwhere(b == rim)
        r0, c0 = rc.min(axis=0)
        r1, c1 = rc.max(axis=0)
        anchor = mask[r0 + 1:r1, c0 + 1:c1]
        profile = anchor.sum(axis=0)
        expected = np.arange(anchor.shape[0])[:, None] < profile
        if np.all(profile > 0) and np.max(profile) < anchor.shape[0] and np.array_equal(anchor, expected):
            canonical.append((rotation, b.copy(), mask, tuple(map(int, (r0, c0, r1, c1))), profile))
    assert len(canonical) == 1, 'Anchor must specify a unique assembly direction'
    rotation, b, fixed_mask, box, profile = canonical[0]
    r0, c0, r1, c1 = box
    h, w = r1 - r0 - 1, c1 - c0 - 1
    inside = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    fixed_cells = set(map(tuple, np.argwhere(fixed_mask).tolist()))
    substrate_colors = {int(b[p]) for p in fixed_cells & inside if b[p] != wire}
    assert len(substrate_colors) == 1
    substrate = substrate_colors.pop()
    assert len({bg, rim, wire, substrate}) == 4
    sources = [(p, int(b[p])) for p in sorted(fixed_cells - inside)
               if b[p] not in (bg, rim, wire)]
    assert 1 <= len(sources) <= 2
    for (r, c), paint in sources:
        cup = {(r + dr, c + dc) for dr, dc in D8
               if 0 <= r + dr < b.shape[0] and 0 <= c + dc < b.shape[1] and b[r + dr, c + dc] == wire}
        assert len(cup) == 3 and paint not in (bg, rim, wire, substrate)
    movable = _components(map(tuple, np.argwhere((b != bg) & ~fixed_mask)))
    descriptions = []
    other_colors = set()
    for cells in movable:
        rc = np.array(sorted(cells), dtype=int)
        origin = rc.min(axis=0)
        local = rc - origin
        assert local[:, 1].max() + 1 == w, 'Fragments must span the board cross section'
        lo, hi = [], []
        for c in range(w):
            rows = np.sort(local[local[:, 1] == c, 0])
            assert len(rows) and np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
            lo.append(int(rows[0]))
            hi.append(int(rows[-1]) + 1)
        values = b[rc[:, 0], rc[:, 1]]
        assert substrate in values and rim not in values
        other_colors.update(int(v) for v in values if v not in (substrate, wire))
        descriptions.append(dict(cells=rc, local=local, values=values,
                                 lower=np.array(lo), upper=np.array(hi)))
    assert len(other_colors) == 1
    empty = other_colors.pop()
    assert empty not in {bg, rim, wire, substrate} | {col for _, col in sources}
    base = np.full_like(b, bg)
    base[fixed_mask] = b[fixed_mask]
    return dict(grid=b, background=bg, rim=rim, wire=wire, substrate=substrate,
                empty=empty, sources=sources, pieces=descriptions, base=base,
                rotation=rotation, box=box, height=h, width=w, profile=profile)


def _mate_ports(board, labels, info):
    wire = info['wire']
    # Equality is required only across physical seams, not along every edge of
    # a conductor within a tile. Thus ordinary free internal wire ends are legal.
    for axis in (0, 1):
        if axis == 0:
            a, b, la, lb = board[:-1], board[1:], labels[:-1], labels[1:]
        else:
            a, b, la, lb = board[:, :-1], board[:, 1:], labels[:, :-1], labels[:, 1:]
        if np.any((la != lb) & ((a == wire) != (b == wire))):
            return False
    r0, c0, r1, c1 = info['box']
    for inner, outer in ((board[0], info['base'][r0, c0 + 1:c1]),
                         (board[-1], info['base'][r1, c0 + 1:c1]),
                         (board[:, 0], info['base'][r0 + 1:r1, c0]),
                         (board[:, -1], info['base'][r0 + 1:r1, c1])):
        if np.any((inner == wire) != (outer == wire)):
            return False
    return True


def _paint(assembled, info):
    wire, empty = info['wire'], info['empty']
    nets = _components(map(tuple, np.argwhere(assembled == wire)))
    membership = {p: i for i, net in enumerate(nets) for p in net}
    paints = {}
    for p, color in info['sources']:
        adjacent = {membership[q] for q in _neighbors({p}) if q in membership}
        if len(adjacent) != 1:
            return None
        net = adjacent.pop()
        if net in paints and paints[net] != color:
            return None
        paints[net] = color
    wells = _components(map(tuple, np.argwhere(assembled == empty)))
    result = assembled.copy()
    supplied, unsupplied = 0, 0
    well_records = []
    for cells in wells:
        boundary = _neighbors(cells)
        if not boundary <= membership.keys():
            return None  # Every blank chamber must close into a wire contour.
        owners = {membership[p] for p in boundary}
        if len(owners) != 1:
            return None
        net = owners.pop()
        color = paints.get(net)
        if color is not None:
            supplied += 1
            for p in cells:
                result[p] = color
        else:
            unsupplied += 1
        well_records.append(dict(cells=cells, boundary=boundary, paint=color, net=net))
    if not supplied:
        return None
    return result, dict(wells=well_records, supplied=supplied, unsupplied=unsupplied,
                        nets=nets, paints=paints)


def _assemblies(info, enforce_ports=True):
    h, w = info['height'], info['width']
    r0, c0, r1, c1 = info['box']
    pieces = info['pieces']
    solutions = {}
    # Five fragments have at most 120 orders. Matching profiles prunes before
    # port and cavity checks; no sampled order or expected label is supplied.
    def visit(profile, unused, chosen):
        if not unused:
            if not np.all(profile == h):
                return
            assembled = info['base'].copy()
            board = assembled[r0 + 1:r1, c0 + 1:c1]
            labels = np.full((h, w), -1, dtype=int)
            labels[np.arange(h)[:, None] < info['profile']] = -2
            for i, offset in chosen:
                p = pieces[i]
                local = p['local'] + (offset, 0)
                board[local[:, 0], local[:, 1]] = p['values']
                labels[local[:, 0], local[:, 1]] = i
            assert np.all(labels != -1)
            if enforce_ports and not _mate_ports(board, labels, info):
                return
            key = assembled.tobytes()
            if enforce_ports:
                painted = _paint(assembled, info)
                if painted is None:
                    return
                answer, paint_info = painted
                for well in paint_info['wells']:
                    well['fragment_count'] = len({int(labels[r - r0 - 1, c - c0 - 1]) for r, c in well['cells']})
                solutions.setdefault(key, (answer, assembled.copy(), chosen, paint_info))
            else:
                solutions.setdefault(key, (assembled.copy(), chosen))
            return
        offset = int(profile.min())
        normalized = profile - offset
        for i in unused:
            p = pieces[i]
            if not np.array_equal(p['lower'], normalized):
                continue
            following = p['upper'] + offset
            if np.any(following > h):
                continue
            # Every fragment changes its location by a nonzero translation.
            dest = p['local'] + (r0 + 1 + offset, c0 + 1)
            if np.array_equal(dest, p['cells']):
                continue
            visit(following, tuple(j for j in unused if j != i), chosen + [(i, offset)])
    visit(info['profile'], tuple(range(len(pieces))), [])
    return list(solutions.values())


def _analyze(input_grid):
    info = _parse(input_grid)
    candidates = _assemblies(info)
    assert len(candidates) == 1, 'Board must have one distinct assembly with mating wires and closed wells'
    answer, unpainted, placements, paint_info = candidates[0]
    info.update(paint_info, assembled=unpainted, placements=placements)
    return np.rot90(answer, -info['rotation']).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from silhouettes, visible inlays, frame, and source cups."""
    return _analyze(input_grid)[0]


def _ri(rng, a, b):
    return int(rng.integers(a, b + 1))


def _profiles(rng, width, count, equal_layers=False):
    root = 1 + rng.integers(0, 3, size=width)
    if np.ptp(root) == 0:
        root[width // 2] = 1 if root[0] != 1 else 2
    cuts = [root]
    if equal_layers:
        first = int(root.max()) + _ri(rng, 3, 4)
        depth = _ri(rng, 6, 8) if count == 2 else _ri(rng, 4, 5) if count == 3 else _ri(rng, 3, 4)
        cuts += [np.full(width, first + i * depth) for i in range(count)]
    else:
        for i in range(count - 1):
            shape = rng.integers(0, 2, size=width) if rng.random() < .65 else np.zeros(width, dtype=int)
            cuts.append(int(cuts[-1].max()) + _ri(rng, 1, 3) + shape)
        cuts.append(np.full(width, int(cuts[-1].max()) + _ri(rng, 3, 5)))
    return cuts


def _well(rng, rectangular=False, max_size=3):
    h, w = _ri(rng, 1, max_size), _ri(rng, 1, max_size)
    inner = {(r, c) for r in range(h) for c in range(w)}
    if not rectangular and h > 1 and w > 1 and rng.random() < .45:
        inner.remove((0, 0))
        if h == 3 and rng.random() < .5:
            inner.remove((h - 1, w - 1))
    shell = _neighbors(inner, D8)
    return inner, shell


def _path(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    directions = [D4[i] for i in rng.permutation(4)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        raise ValueError('No insulated routing path')
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _compatible_cut(target, profile, wire):
    """A proposed physical cut must expose matched wire faces on both sides."""
    side = np.arange(target.shape[0])[:, None] < profile
    conductor = target == wire
    return not (np.any((side[:-1] != side[1:]) & (conductor[:-1] != conductor[1:])) or
                np.any((side[:, :-1] != side[:, 1:]) & (conductor[:, :-1] != conductor[:, 1:])))


def _recut(rng, target, root, count, wire):
    """Sample cuts of an actual circuit rather than hoping every tooth is safe.

    This only constructs physical fragments. The solver still reconstructs all
    their possible orders from the resulting input without knowing these cuts.
    """
    h, w = target.shape
    levels = [k for k in range(int(root.max()) + 2, h - 1)
              if _compatible_cut(target, np.full(w, k), wire)]
    choices = [ks for ks in combinations(levels, count - 1)
               if all(b - a >= 3 for a, b in zip(ks, ks[1:]))]
    if not choices:
        raise ValueError('No compatible set of jagged cuts')
    chosen = choices[_ri(rng, 0, len(choices) - 1)]
    cuts = [root.copy()]
    for level in chosen:
        profile = np.full(w, level)
        # Greedy one-cell teeth keep every proposed face compatible. Seams
        # remain separated, making all fragment column intervals connected.
        for c in rng.permutation(w):
            if rng.random() < .55:
                proposed = profile.copy()
                proposed[c] += 1
                if _compatible_cut(target, proposed, wire):
                    profile = proposed
        cuts.append(profile)
    cuts.append(np.full(w, h))
    return cuts


def _dilate(mask):
    h, w = mask.shape
    result = mask.copy()
    for dr, dc in D8:
        ra, rb, ca, cb = max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)
        result[ra + dr:rb + dr, ca + dc:cb + dc] |= mask[ra:rb, ca:cb]
    return result


def _scatter(rng, base, fragments, bg, box):
    reserved = base != bg
    r0, c0, r1, c1 = box
    reserved[r0:r1 + 1, c0:c1 + 1] = True
    for _ in range(12):
        a = base.copy()
        forbidden = _dilate(reserved)
        success = True
        for fragment in sorted(fragments, key=lambda p: -len(p['cells'])):
            coords = fragment['cells']
            local = coords - coords.min(axis=0)
            ph, pw = local.max(axis=0) + 1
            choices = np.ones((a.shape[0] - ph + 1, a.shape[1] - pw + 1), dtype=bool)
            for r, c in local:
                choices &= ~forbidden[r:r + choices.shape[0], c:c + choices.shape[1]]
            positions = np.argwhere(choices)
            if not len(positions):
                success = False
                break
            shifted = local + positions[_ri(rng, 0, len(positions) - 1)]
            a[shifted[:, 0], shifted[:, 1]] = fragment['values']
            occupied = np.zeros_like(forbidden)
            occupied[shifted[:, 0], shifted[:, 1]] = True
            forbidden |= _dilate(occupied)
        if success:
            return a
    raise ValueError('Insufficient scattering space')


def _make_pair(rng, palette, count, active, inactive, extra_wells, equal_layers,
               rotation, require_split=False, require_geometric_tie=False):
    bg, rim, substrate, wire, empty, p1, p2 = palette
    well_count = active + inactive + extra_wells
    width = _ri(rng, 12, 13) if well_count >= 3 else _ri(rng, 9 if active == 2 or inactive else 7, 11)
    cuts = _profiles(rng, width, count, equal_layers)
    # More contours need more interior area even when there are only two
    # fragments. Counts determine a spatial minimum, never a label exception.
    cuts[-1] = np.full(width, max(int(cuts[-1][0]), 10 + 2 * (well_count - 1),
                                int(cuts[0].max()) + 3 * count))
    height = int(cuts[-1][0])
    if height > 19:
        raise ValueError('Board too tall')
    target = np.full((height, width), substrate, dtype=int)
    root_max = int(cuts[0].max())
    owned = []
    occupied = set()
    owners = list(range(active)) + [i % active for i in range(extra_wells)] + [None] * inactive
    for index, owner in enumerate(owners):
        inner, shell = _well(rng, max_size=2 if well_count >= 3 else 3)
        if require_split and index == 0:
            inner = {(r, c) for r in range(3) for c in range(_ri(rng, 1, 2))}
            shell = _neighbors(inner, D8)
        placed = False
        for _ in range(70):
            rr = _ri(rng, root_max + 1, height - 3)
            cc = _ri(rng, 2, width - 3)
            if require_split and index == 0:
                rr = int(cuts[1].min()) - 1
            ins = {(r + rr, c + cc) for r, c in inner}
            edge = {(r + rr, c + cc) for r, c in shell}
            all_cells = ins | edge
            if any(r <= root_max or r >= height - 1 or c < 1 or c >= width - 1 for r, c in all_cells):
                continue
            if (all_cells | _neighbors(all_cells)) & occupied:
                continue
            owned.append(dict(inner=ins, shell=edge, owner=owner))
            occupied |= all_cells
            placed = True
            break
        if not placed:
            raise ValueError('Well packing')
    routes = [set() for _ in range(active)]
    ports = []
    for owner in range(active):
        choices = [c for c in range(2, width - 2) if all(abs(c - p) >= 3 for p in ports)]
        if not choices:
            raise ValueError('Source clearance')
        port = choices[_ri(rng, 0, len(choices) - 1)]
        ports.append(port)
        routes[owner].add((0, port))
    # Straight gateways cross a flat local part of the jagged starter. A wire
    # must not acquire an exposed side contact with an adjacent starter tooth.
    for port in ports:
        cuts[0][port - 1:port + 2] = root_max
    cuts[0][0] = 1
    cuts[0][-1] = 2
    domain = {(r, c) for r in range(height - 1) for c in range(1, width - 1)}
    for owner in rng.permutation(active):
        owner = int(owner)
        other = set().union(*(q['inner'] | q['shell'] for q in owned if q['owner'] != owner),
                            *(routes[j] for j in range(active) if j != owner))
        forbidden = other | _neighbors(other)
        forbidden |= set().union(*(q['inner'] for q in owned))
        # Routes may enter the fixed anchor only through their own straight
        # gateway; this keeps its visible role independent of floating inlays.
        allowed = domain - forbidden
        for c in range(width):
            for r in range(root_max + 1):
                if c != ports[owner]:
                    allowed.discard((r, c))
        for q in [q for q in owned if q['owner'] == owner]:
            route = _path(rng, routes[owner], q['shell'], allowed | routes[owner])
            routes[owner] |= route | q['shell']
    for q in owned:
        for p in q['shell']:
            target[p] = wire
        for p in q['inner']:
            target[p] = empty
    for route in routes:
        for p in route:
            target[p] = wire
    if not equal_layers:
        cuts = _recut(rng, target, cuts[0], count, wire)
    # Reject impossible intended seams before spending time scattering and
    # solving the scene. The input-only solver still checks every assembly.
    if not all(_compatible_cut(target, cut, wire) for cut in cuts[:-1]):
        raise ValueError('Intended cut has unmatched wire faces')
    # Fixed rim and jagged starter material; no final-generator composition or
    # parent import is used here.
    canvas_h = _ri(rng, max(24, height + 5), 30)
    canvas_w = _ri(rng, max(25, 2 * width + 4), 30)
    frame_h, frame_w = height + 2, width + 2
    r0 = _ri(rng, 2, canvas_h - frame_h - 1)
    if width >= 12:
        c0 = 1 if rng.random() < .5 else canvas_w - frame_w - 1
    elif rng.random() < .75:
        c0 = _ri(rng, 1, 3) if rng.random() < .5 else _ri(rng, canvas_w - frame_w - 3, canvas_w - frame_w - 1)
    else:
        c0 = _ri(rng, 1, canvas_w - frame_w - 1)
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    base = np.full((canvas_h, canvas_w), bg, dtype=int)
    base[r0, c0:c1 + 1] = rim
    base[r1, c0:c1 + 1] = rim
    base[r0:r1 + 1, c0] = rim
    base[r0:r1 + 1, c1] = rim
    for c in range(width):
        base[r0 + 1:r0 + 1 + cuts[0][c], c0 + 1 + c] = target[:cuts[0][c], c]
    for i, port in enumerate(ports):
        c = c0 + 1 + port
        base[r0, c] = wire
        base[r0 - 2, c] = (p1, p2)[i]
        for p in ((r0 - 2, c + 1), (r0 - 1, c), (r0 - 1, c + 1)):
            base[p] = wire
    fragments = []
    for low, high in zip(cuts, cuts[1:]):
        cells = np.array([(r0 + 1 + r, c0 + 1 + c) for c in range(width)
                          for r in range(int(low[c]), int(high[c]))], dtype=int)
        vals = target[cells[:, 0] - r0 - 1, cells[:, 1] - c0 - 1]
        if substrate not in vals:
            raise ValueError('Every tile needs a visible substrate')
        fragments.append(dict(cells=cells, values=vals))
    inp = _scatter(rng, base, fragments, bg, (r0, c0, r1, c1))
    expected = base.copy()
    expected[r0 + 1:r1, c0 + 1:c1] = target
    for q in owned:
        if q['owner'] is not None:
            for r, c in q['inner']:
                expected[r0 + 1 + r, c0 + 1 + c] = (p1, p2)[q['owner']]
    if int(np.bincount(inp.ravel(), minlength=10).argmax()) != bg:
        raise ValueError('Exterior background must be the modal color')
    inp, expected = np.rot90(inp, rotation).copy(), np.rot90(expected, rotation).copy()
    try:
        prediction, info = _analyze(inp)
    except AssertionError as error:
        raise ValueError(str(error)) from error
    if not np.array_equal(prediction, expected):
        raise ValueError('Intended ownership disagrees with assembled circuit')
    if require_geometric_tie and len(_assemblies(info, False)) < 2:
        raise ValueError('Lesson must distinguish geometric-only assembly')
    if require_split and not any(q['fragment_count'] > 1 and q['paint'] is not None for q in info['wells']):
        raise ValueError('Lesson must show a chamber closing across fragments')
    return dict(input=inp, output=expected)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError('Counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _fallback_pair(rng, palette, spec):
    """A sparse constructive member of the same rule after bounded rejection.

    All count, paint, inactive, split-well, and geometric-tie lessons survive.
    This is not an answer convention: ordinary input-derived search must accept
    the constructed member and agree with independently assigned well ownership.
    """
    bg, rim, substrate, wire, empty, p1, p2 = palette
    h, w = 18, 13
    target = np.full((h, w), substrate, dtype=int)
    active, inactive, extra = spec['active'], spec['inactive'], spec['extra_wells']
    regions = []
    def well(r, c, owner):
        inner = {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 2)}
        shell = _neighbors(inner, D8)
        for p in inner:
            target[p] = empty
        for p in shell:
            target[p] = wire
        regions.append((inner, owner))
    well(5, 3, 0)
    if active == 2:
        well(5, 8, 1)
    if extra:
        well(11, 3, 0)
    if inactive:
        well(11, 8, None)
    target[:14 if extra else 8, 2] = wire
    if active == 2:
        target[:8, 10] = wire
    root = 1 + rng.integers(0, 3, size=w)
    ports = [2] + ([10] if active == 2 else [])
    for port in ports:
        root[port - 1:port + 2] = 3
    root[0], root[-1] = 1, 2
    n = spec['count']
    levels = {2: [6, 18], 3: [6, 12, 18], 4: [6, 12, 16, 18], 5: [6, 12, 16, 17, 18]}[n]
    cuts = [root] + [np.full(w, k) for k in levels]
    assert all(_compatible_cut(target, p, wire) for p in cuts[:-1])
    r0 = _ri(rng, 2, 6)
    c0 = 1 if rng.random() < .5 else 14
    r1, c1 = r0 + h + 1, c0 + w + 1
    base = np.full((30, 30), bg, dtype=int)
    base[r0, c0:c1 + 1] = rim
    base[r1, c0:c1 + 1] = rim
    base[r0:r1 + 1, c0] = rim
    base[r0:r1 + 1, c1] = rim
    for c in range(w):
        base[r0 + 1:r0 + 1 + root[c], c0 + 1 + c] = target[:root[c], c]
    for i, port in enumerate(ports):
        c = c0 + 1 + port
        base[r0, c] = wire
        base[r0 - 2, c] = (p1, p2)[i]
        for p in ((r0 - 2, c + 1), (r0 - 1, c), (r0 - 1, c + 1)):
            base[p] = wire
    fragments = []
    for low, high in zip(cuts, cuts[1:]):
        cells = np.array([(r0 + 1 + r, c0 + 1 + c) for c in range(w)
                          for r in range(int(low[c]), int(high[c]))], dtype=int)
        fragments.append(dict(cells=cells, values=target[cells[:, 0] - r0 - 1, cells[:, 1] - c0 - 1]))
    try:
        inp = _scatter(rng, base, fragments, bg, (r0, c0, r1, c1))
    except ValueError:
        inp = base.copy()
        row, col = _ri(rng, 0, 5), 17 if c0 == 1 else 0
        for p in fragments:
            local = p['cells'] - p['cells'].min(axis=0)
            destination = local + (row, col)
            inp[destination[:, 0], destination[:, 1]] = p['values']
            row += int(local[:, 0].max()) + 3
    out = base.copy()
    out[r0 + 1:r1, c0 + 1:c1] = target
    for inner, owner in regions:
        if owner is not None:
            for r, c in inner:
                out[r0 + 1 + r, c0 + 1 + c] = (p1, p2)[owner]
    inp, out = np.rot90(inp, spec['rotation']).copy(), np.rot90(out, spec['rotation']).copy()
    prediction, info = _analyze(inp)
    assert np.array_equal(prediction, out)
    if spec.get('require_geometric_tie'):
        assert len(_assemblies(info, False)) > 1
    if spec.get('require_split'):
        assert any(q['fragment_count'] > 1 and q['paint'] is not None for q in info['wells'])
    return dict(input=inp, output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)[:7]))
    stable = bool(rng.integers(2))
    orientation = _ri(rng, 0, 3)
    shift = _ri(rng, 0, 2)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            lesson0 = split == 'train' and index == 0
            lesson1 = split == 'train' and index == 1
            if lesson0:
                spec = dict(count=2, active=1, inactive=1, extra_wells=0, equal_layers=True,
                            rotation=orientation, require_split=True)
            elif lesson1:
                spec = dict(count=3, active=2, inactive=0, extra_wells=1, equal_layers=True,
                            rotation=orientation, require_geometric_tie=True)
            else:
                spec = dict(count=_ri(rng, 2, 4), active=_ri(rng, 1, 2), inactive=_ri(rng, 0, 1),
                            extra_wells=_ri(rng, 0, 1), equal_layers=rng.random() < .5,
                            rotation=(orientation + _ri(rng, 0, 3)) % 4)
                if split == 'train':
                    if shift == 0:
                        spec['count'] = min(spec['count'], 3)
                    elif shift == 1:
                        spec['rotation'] = (orientation + 2 * _ri(rng, 0, 1)) % 4
                    elif spec['active'] + spec['inactive'] + spec['extra_wells'] > 3:
                        spec['extra_wells'] = 0
                if split == 'test':
                    if shift == 0:
                        spec['count'] = 4
                    elif shift == 1:
                        spec['rotation'] = (orientation + 1 + 2 * _ri(rng, 0, 1)) % 4
                    else:
                        spec.update(active=2, inactive=1, extra_wells=1)
            local_palette = palette if stable else tuple(map(int, rng.permutation(10)[:7]))
            # The already-checked constructive fallback preserves all lesson
            # witnesses and shifts; prolonged rejection adds no rule coverage.
            for attempt in range(64):
                if not (lesson0 or lesson1):
                    spec['equal_layers'] = rng.random() < .5
                try:
                    pair = _make_pair(rng, local_palette, **spec)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                for _ in range(20):
                    pair = _fallback_pair(rng, local_palette, spec)
                    key = _key(pair['input'])
                    if key not in seen:
                        break
                else:
                    raise RuntimeError('Could not scatter a distinct safety-net example')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_info, test_info = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            prediction, info = _analyze(inp)
            assert np.array_equal(prediction, out), 'Label violates assembly or route-derived paint'
            canonical_out = np.rot90(out, info['rotation'])
            # Fixed sources, rim, and anchor stay exactly as observed.
            fixed = info['base'] != info['background']
            assert np.array_equal(canonical_out[fixed], info['base'][fixed])
            # Compare inlays before filling. Each component is copied exactly,
            # by translation, including its unpainted chamber cells.
            r0, c0, _, _ = info['box']
            for i, offset in info['placements']:
                p = info['pieces'][i]
                destination = p['local'] + (r0 + 1 + offset, c0 + 1)
                assert np.array_equal(info['assembled'][destination[:, 0], destination[:, 1]], p['values'])
                assert not np.array_equal(destination, p['cells'])
                assert np.all(canonical_out[p['cells'][:, 0], p['cells'][:, 1]] == info['background'])
            changed_by_paint = info['assembled'] != canonical_out
            assert np.all(info['assembled'][changed_by_paint] == info['empty'])
            assert sum(len(q['cells']) for q in info['wells'] if q['paint'] is not None) == np.count_nonzero(changed_by_paint)
            if split == 'train':
                train_info.append(info)
            else:
                test_info.append(info)
    assert any(i['unsupplied'] for i in train_info), 'Show preservation of an unpowered well'
    assert any(len(set(i['paints'].values())) == 2 for i in train_info), 'Show independent local paints'
    assert any(len([q for q in i['wells'] if q['net'] == net]) > 1
               for i in train_info for net in i['paints']), 'Show propagation to several wells'
    assert any(len(_assemblies(i, False)) > 1 for i in train_info), 'Show a tie resolved by wire ports'
    assert any(q['fragment_count'] > 1 and q['paint'] is not None for i in train_info for q in i['wells']), 'Show a split well'
    axis_transfer = {i['rotation'] % 2 for i in train_info}.isdisjoint({i['rotation'] % 2 for i in test_info})
    fragment_transfer = min(len(i['pieces']) for i in test_info) > max(len(i['pieces']) for i in train_info)
    well_transfer = min(len(i['wells']) for i in test_info) > max(len(i['wells']) for i in train_info)
    assert axis_transfer or fragment_transfer or well_transfer, 'Require an observable train/test shift'


def self_test():
    # Constructive safety-net and unsampled five-fragment members obey exactly
    # the same input-only rule, including same-colored indistinguishable caps.
    rng = np.random.default_rng(901)
    for n in range(2, 6):
        for active in (1, 2):
            for inactive in (0, 1):
                for extra in (0, 1):
                    spec = dict(count=n, active=active, inactive=inactive, extra_wells=extra,
                                rotation=(n + active + inactive) % 4)
                    pair = _fallback_pair(rng, (0, 8, 2, 5, 1, 3, 4), spec)
                    assert np.array_equal(solve(pair['input']), pair['output'])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, structures = [], set()
    for puzzle in puzzles:
        # generate_puzzle has already run the complete validator. Analyze once
        # here to collect diversity measurements rather than validating twice.
        for pair in puzzle['train'] + puzzle['test']:
            _, info = _analyze(pair['input'])
            colors = {}
            normalized = tuple(colors.setdefault(int(v), len(colors)) for v in pair['input'].flat)
            structures.add((pair['input'].shape, normalized))
            features.append(dict(shape=pair['input'].shape, pieces=len(info['pieces']),
                                 rotation=info['rotation'], board=(info['height'], info['width']),
                                 source_count=len(info['sources']), wells=len(info['wells']),
                                 split_wells=sum(q['fragment_count'] > 1 for q in info['wells']),
                                 inactive=info['unsupplied'], tie=len(_assemblies(info, False)),
                                 role_colors=(info['background'], info['rim'], info['substrate'], info['wire'], info['empty']),
                                 masks=tuple((tuple(p['lower']), tuple(p['upper'])) for p in info['pieces'])))
    assert len(structures) >= 200
    assert len({f['shape'] for f in features}) >= 25
    assert len({f['board'] for f in features}) >= 20
    assert {f['pieces'] for f in features} >= {2, 3, 4}
    assert {f['rotation'] for f in features} == {0, 1, 2, 3}
    assert {f['source_count'] for f in features} == {1, 2}
    assert max(f['wells'] for f in features) >= 4
    assert len({f['masks'] for f in features}) >= 100
    assert len({f['role_colors'] for f in features}) >= 40
    assert sum(f['tie'] > 1 for f in features) >= 42
    assert sum(f['split_wells'] > 0 for f in features) >= 42
    assert sum(f['inactive'] > 0 for f in features) >= 84
    repeat = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in puzzles[0] for a, b in zip(puzzles[0][s], repeat[s]) for k in a)
    # Boundary and near-boundary counts suffice here; verify.py independently
    # executes every legal count combination, including all intermediate ones.
    for nt, ne in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        generate_puzzle(1000 + nt * 10 + ne, nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    # Removing a supplied cell, painting an inactive well, or using the
    # geometric-only assembly cannot masquerade as a valid answer.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    pair = broken['train'][0]
    _, info = _analyze(pair['input'])
    q = next(q for q in info['wells'] if q['paint'] is not None)
    r, c = next(iter(q['cells']))
    out = np.rot90(pair['output'], info['rotation']).copy()
    out[r, c] = info['empty']
    pair['output'] = np.rot90(out, -info['rotation']).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Partial painting accepted')
    return dict(seeds=42, pairs=len(features), distinct_structures=len(structures),
                grid_shapes=len({f['shape'] for f in features}), board_shapes=len({f['board'] for f in features}),
                geometric_ties=sum(f['tie'] > 1 for f in features), split_wells=sum(f['split_wells'] > 0 for f in features))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
