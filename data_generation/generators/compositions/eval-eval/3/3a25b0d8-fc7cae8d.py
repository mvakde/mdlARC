"""Compass rings: transfer chamber colors by an oriented cyclic correspondence.

Only standard Python and NumPy are used. Generation retains no hidden state;
solve decodes the two local compasses and the visible chamber cycle.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from numbers import Integral

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
# Corner numbers follow clockwise order: top-left, top-right, bottom-right, bottom-left.
CORNERS = (("T", "L"), ("T", "R"), ("B", "R"), ("B", "L"))
HOST_SEEDS = (0,602343957,599790244,498869883,647168757,699676096,404425455,923197112,241557658,760918251,37165000,834615757,856749643,951475944,700751310,306855071,546923798,398399509,173052902,354456210,578772392,654434799,215547395,376276641,523081523,348771899,307402948,621060657,696807475,293995882,974143224,849886720,622170821,499154263,701694608,976706880,607592436,492948494,812894382,127504,593841393,624638374)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = ((-1, 0), (0, 1), (1, 0), (0, -1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in steps:
                q = p[0] + dr, p[1] + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.asarray(cells, dtype=int))
    return result


def _perimeter(h, w):
    """Clockwise walk just inside the outer rectangular wall, with no repeats."""
    return ([(1, c) for c in range(1, w - 1)]
            + [(r, w - 2) for r in range(2, h - 1)]
            + [(h - 2, c) for c in range(w - 3, 0, -1)]
            + [(r, 1) for r in range(h - 3, 1, -1)])


def _corner_position(shape, corner):
    h, w = shape
    return ((1, 1), (1, w - 2), (h - 2, w - 2), (h - 2, 1))[corner]


def _order(labels, corner, direction=1):
    path = _perimeter(*labels.shape)
    start = path.index(_corner_position(labels.shape, corner))
    assert labels[path[start]] >= 0, "compass corner must lie in a chamber"
    order = []
    for i in range(len(path)):
        label = int(labels[path[(start + direction * i) % len(path)]])
        if label >= 0 and (not order or label != order[-1]):
            order.append(label)
    if len(order) > 1 and order[-1] == order[0]:
        order.pop()
    assert len(order) == len(set(order)), "a chamber occupies disconnected perimeter arcs"
    assert set(order) == set(map(int, np.unique(labels))) - {-1}, "nonperimeter chamber"
    return order


def _corner(guides):
    matches = [i for i, sides in enumerate(CORNERS) if set(sides) == set(guides)]
    assert len(matches) == 1, "guides must occupy adjacent sides"
    return matches[0]


def _compass_matrix(donor_guides, target_guides):
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in donor_guides.items():
        target = next(s for s, c in target_guides.items() if c == color)
        matrix += np.outer(NORMALS[target], NORMALS[side])
    assert np.array_equal(matrix.T @ matrix, np.eye(2, dtype=int))
    return matrix


def _bars(grid, points, markers):
    descriptions = []
    for color in markers:
        p = points[grid[tuple(points.T)] == color]
        assert len(p) >= 7
        lo, hi = p.min(axis=0), p.max(axis=0)
        assert (lo[0] == hi[0]) != (lo[1] == hi[1]), "guide is not straight"
        assert len(p) == int(np.prod(hi - lo + 1)), "broken guide"
        descriptions.append((int(color), lo, hi))
    horizontal = [d for d in descriptions if d[1][0] == d[2][0]]
    vertical = [d for d in descriptions if d[1][1] == d[2][1]]
    assert len(horizontal) == len(vertical) == 1
    hc, hl, hh = horizontal[0]
    vc, vl, vh = vertical[0]
    r, c = int(vl[0]), int(hl[1])
    h, w = int(vh[0] - vl[0] + 1), int(hh[1] - hl[1] + 1)
    assert int(hl[0]) in (r - 1, r + h)
    assert int(vl[1]) in (c - 1, c + w)
    guides = {"T" if hl[0] == r - 1 else "B": hc,
              "L" if vl[1] == c - 1 else "R": vc}
    assert r >= 1 and c >= 1 and r + h < grid.shape[0] and c + w < grid.shape[1]
    return (r, c, h, w), guides


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(np.all(line == bg) for line in (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    parts = _components(grid != bg, diagonal=True)
    assert len(parts) == 2, "expected two separate compass figures"
    palettes = [set(map(int, grid[tuple(p.T)])) for p in parts]
    blank = [i for i, palette in enumerate(palettes) if len(palette) == 3]
    assert len(blank) == 1, "exactly one unfilled figure"
    ti, di = blank[0], 1 - blank[0]
    assert len(palettes[di]) >= 5 and palettes[ti] < palettes[di]
    shared = palettes[ti]
    markers = []
    for color in sorted(shared):
        p = parts[ti][grid[tuple(parts[ti].T)] == color]
        if np.all(p[:, 0] == p[0, 0]) or np.all(p[:, 1] == p[0, 1]):
            markers.append(color)
    assert len(markers) == 2
    wall = next(iter(shared - set(markers)))
    figures = []
    for index in (di, ti):
        points = parts[index]
        bounds, guides = _bars(grid, points, markers)
        r, c, h, w = bounds
        crop = grid[r:r + h, c:c + w].copy()
        assert all(np.all(line == wall) for line in (crop[0], crop[-1], crop[:, 0], crop[:, -1]))
        assert not set(markers) & set(map(int, np.unique(crop)))
        # Every pixel of this object belongs to its rectangle or its two guides.
        inside = ((points[:, 0] >= r) & (points[:, 0] < r + h)
                  & (points[:, 1] >= c) & (points[:, 1] < c + w))
        assert set(map(int, grid[tuple(points[~inside].T)])) == set(markers)
        assert np.count_nonzero(~inside) == h + w
        rooms = _components(crop != wall)
        assert 4 <= len(rooms) <= 10
        labels = np.full(crop.shape, -1, dtype=int)
        colors = []
        for label, room in enumerate(rooms):
            values = np.unique(crop[tuple(room.T)])
            assert len(values) == 1, "nonuniform chamber"
            color = int(values[0])
            assert color == bg if index == ti else color not in (bg, wall, *markers)
            labels[tuple(room.T)] = label
            colors.append(color)
        corner = _corner(guides)
        order = _order(labels, corner)
        figures.append(dict(crop=crop, labels=labels, rooms=rooms, colors=colors,
                            guides=guides, bounds=bounds, corner=corner, order=order))
    donor, target = figures
    assert len(donor['rooms']) == len(target['rooms'])
    matrix = _compass_matrix(donor['guides'], target['guides'])
    direction = int(round(np.linalg.det(matrix)))
    target_order = _order(target['labels'], target['corner'], direction)
    return dict(background=bg, wall=wall, markers=tuple(markers), donor=donor,
                target=target, matrix=matrix, direction=direction, target_order=target_order)


def _answer(info, direction=None):
    donor, target = info['donor'], info['target']
    order = info['target_order'] if direction is None else _order(target['labels'], target['corner'], direction)
    out = target['crop'].copy()
    for a, b in zip(donor['order'], order):
        out[target['labels'] == b] = donor['colors'][a]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only this input's enclosures, guide identities and cyclic order."""
    return _answer(_decode(input_grid))


def _coarse_cycle(rows, cols):
    return ([(0, c) for c in range(cols)] + [(r, cols - 1) for r in range(1, rows)]
            + [(rows - 1, c) for c in range(cols - 2, -1, -1)]
            + [(r, 0) for r in range(rows - 2, 0, -1)])


def _plan(rows, cols, cuts):
    path = _coarse_cycle(rows, cols)
    plan = np.full((rows, cols), -1, dtype=int)
    cuts = set(map(int, cuts))
    start = min(cuts)
    label = -1
    for i in range(len(path)):
        j = (start + i) % len(path)
        if j in cuts:
            label += 1
        plan[path[j]] = label
    return plan


def _sizes(rng, count):
    sizes = np.ones(count, dtype=int)
    # Final figure dimension is 1 + sum(sizes + 1), never more than 11.
    for _ in range(int(rng.integers(0, 11 - (2 * count + 1) + 1))):
        sizes[int(rng.integers(count))] += 1
    return sizes


def _raster(plan, heights, widths, rng=None):
    ys = np.concatenate(([0], np.cumsum(heights + 1)))
    xs = np.concatenate(([0], np.cumsum(widths + 1)))
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in _coarse_cycle(*plan.shape):
        v = int(plan[r, c])
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = v
        if r and plan[r - 1, c] == v:
            labels[ys[r], xs[c] + 1:xs[c + 1]] = v
        if c and plan[r, c - 1] == v:
            labels[ys[r] + 1:ys[r + 1], xs[c]] = v
    # Small inward wall notches vary contours without changing the visible cycle.
    if rng is not None:
        trace = set(_perimeter(*labels.shape))
        for v in sorted(set(map(int, plan.flat)) - {-1}):
            points = np.argwhere(labels == v)
            candidates = [tuple(p) for p in points if tuple(p) not in trace
                          and any(labels[p[0] + dr, p[1] + dc] == -1
                                  for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))]
            if candidates and rng.random() < .5:
                p = candidates[int(rng.integers(len(candidates)))]
                trial = labels == v
                trial[p] = False
                if np.any(trial) and len(_components(trial)) == 1:
                    labels[p] = -1
    return labels


def _random_ring(rng, count):
    choices = [(r, c) for r in range(3, 6) for c in range(3, 6)
               if 2 * r + 2 * c - 4 >= count]
    rows, cols = choices[int(rng.integers(len(choices)))]
    n = 2 * rows + 2 * cols - 4
    cuts = rng.choice(n, count, replace=False)
    return _raster(_plan(rows, cols, cuts), _sizes(rng, rows), _sizes(rng, cols), rng)


def _guides(corner, colors, reverse=False):
    return dict(zip(CORNERS[corner], colors[::-1] if reverse else colors))


def _frame(labels, bg, wall, guides, colors=None):
    h, w = labels.shape
    grid = np.full((h + 2, w + 2), bg, dtype=np.int8)
    crop = grid[1:-1, 1:-1]
    crop[labels < 0] = wall
    if colors is not None:
        for label, color in enumerate(colors):
            crop[labels == label] = color
    for side, color in guides.items():
        if side == 'T':
            grid[0, 1:-1] = color
        elif side == 'B':
            grid[-1, 1:-1] = color
        elif side == 'L':
            grid[1:-1, 0] = color
        else:
            grid[1:-1, -1] = color
    return grid


def _embed(rng, a, b, bg):
    horizontal = bool(rng.integers(2))
    if not horizontal:
        a, b = a.T, b.T
    if rng.integers(2):
        a, b = b, a
    ah, aw = a.shape
    bh, bw = b.shape
    gap = int(rng.integers(1, min(3, 28 - aw - bw) + 1))
    minimum = aw + bw + gap + 2
    W = int(rng.integers(minimum, 31))
    H = int(rng.integers(max(ah, bh) + 2, min(30, max(ah, bh) + 10) + 1))
    left = int(rng.integers(1, W - (aw + bw + gap)))
    ra, rb = int(rng.integers(1, H - ah)), int(rng.integers(1, H - bh))
    grid = np.full((H, W), bg, dtype=np.int8)
    grid[ra:ra + ah, left:left + aw] = a
    grid[rb:rb + bh, left + aw + gap:left + aw + gap + bw] = b
    return grid if horizontal else grid.T.copy()


def _geometric_conflicts(d, t, dc, tc, matrix):
    """Pairs of donor rooms where rigid corner matching and cyclic order disagree."""
    do = _order(d, dc)
    to = _order(t, tc, int(round(np.linalg.det(matrix))))
    mapping = dict(zip(to, do))
    conflicts = []
    for corner in range(4):
        normal = sum((np.array(NORMALS[s]) for s in CORNERS[corner]))
        moved = matrix @ normal
        target_corner = next(k for k in range(4) if np.array_equal(
            moved, sum((np.array(NORMALS[s]) for s in CORNERS[k]))))
        by_cycle = mapping[int(t[_corner_position(t.shape, target_corner)])]
        by_corner = int(d[_corner_position(d.shape, corner)])
        if by_cycle != by_corner:
            conflicts.append((by_cycle, by_corner))
    return conflicts


def _geometric_conflict(d, t, dc, tc, matrix):
    return bool(_geometric_conflicts(d, t, dc, tc, matrix))


def _distinguish_colors(colors, edges, palette):
    """Two inequality edges can always be colored with two visible fill colors."""
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    painted = {}
    for root in sorted(graph):
        if root in painted:
            continue
        painted[root] = 0
        stack = [root]
        while stack:
            a = stack.pop()
            for b in sorted(graph[a]):
                if b not in painted:
                    painted[b] = 1 - painted[a]
                    stack.append(b)
                else:
                    assert painted[b] != painted[a]
    for label, color_index in painted.items():
        colors[label] = palette[color_index]


def _make_pair(rng, dc, tc, reverse, count, teaching=False):
    palette = list(map(int, rng.permutation(10)))
    bg, wall, ma, mb = palette[:4]
    dg, tg = _guides(dc, (ma, mb)), _guides(tc, (ma, mb), reverse)
    matrix = _compass_matrix(dg, tg)
    direction = int(round(np.linalg.det(matrix)))
    # Bounded rejection only strengthens teaching witnesses. The fallback below
    # enumerates a finite set of ring cuts rather than relaxing the rule.
    candidates = ((_random_ring(rng, count), _random_ring(rng, count)) for _ in range(64))
    selected = None
    for d, t in candidates:
        if d.shape == t.shape and np.array_equal(d < 0, t < 0):
            continue
        if teaching and not _geometric_conflict(d, t, dc, tc, matrix):
            continue
        selected = d, t
        break
    if selected is None:
        # At most C(12, count) <= 924 small candidates; no disk payloads.
        d = _raster(_plan(4, 4, range(count)), np.ones(4, int), np.ones(4, int))
        for cuts in itertools.combinations(range(12), count):
            t = _raster(_plan(4, 4, cuts), np.array([1, 2, 1, 1]), np.ones(4, int))
            if not teaching or _geometric_conflict(d, t, dc, tc, matrix):
                selected = d, t
                break
    assert selected is not None, "finite ring construction exhausted"
    d, t = selected
    richness = int(rng.integers(2, min(6, count) + 1))
    colors = palette[4:4 + richness] + [palette[4 + int(rng.integers(richness))] for _ in range(count - richness)]
    rng.shuffle(colors)
    do, to = _order(d, dc), _order(t, tc, direction)
    # Both the compass ablation and the topological/geometric distinction must
    # change actual colors, rather than merely change hidden construction labels.
    edges = [(do[1], do[-1])]
    if teaching:
        edges.append(_geometric_conflicts(d, t, dc, tc, matrix)[0])
    _distinguish_colors(colors, edges, palette[4:6])
    target_colors = [0] * count
    for a, b in zip(do, to):
        target_colors[b] = colors[a]
    a = _frame(d, bg, wall, dg, colors)
    b = _frame(t, bg, wall, tg)
    inp = _embed(rng, a, b, bg)
    out = np.full(t.shape, wall, dtype=np.int8)
    for label, color in enumerate(target_colors):
        out[t == label] = color
    # Packing changes placement, never either figure's local orientation.
    info = _decode(inp)
    assert np.array_equal(out, _answer(info)), 'visible decoding disagrees with construction'
    return {'input': inp, 'output': out}


def _swap_target_markers(pair):
    inp = np.array(pair['input'], copy=True)
    info = _decode(inp)
    r, c, h, w = info['target']['bounds']
    marker_a, marker_b = info['markers']
    # Restrict replacement to this target's two bars, keeping the donor intact.
    area = inp[r - 1:r + h + 1, c - 1:c + w + 1]
    a, b = area == marker_a, area == marker_b
    area[a], area[b] = marker_b, marker_a
    return {'input': inp, 'output': solve(inp)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    dc = int(rng.integers(4))
    tc = (dc + int(rng.choice((1, 3)))) % 4
    # Choose correspondence yielding a proper quarter-turn for the first pair.
    reverse = next(rev for rev in (False, True) if round(np.linalg.det(
        _compass_matrix(_guides(dc, (0, 1)), _guides(tc, (0, 1), rev)))) == 1)
    first = _make_pair(rng, dc, tc, reverse, int(rng.integers(5, 9)), teaching=True)
    second = _swap_target_markers(first)
    # Hold out an observable pair of compass corners, not a hidden parameter.
    initial = _decode(first['input'])
    first_anchors = initial['donor']['corner'], initial['target']['corner']
    choices = [(a, b) for a in range(4) for b in range(4) if (a, b) != first_anchors]
    held = choices[int(rng.integers(len(choices)))]
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(first['input']), _key(second['input'])}
    for split, n in (('train', num_train - 2), ('test', num_test)):
        for _ in range(n):
            for attempt in range(64):
                a, b = held if split == 'test' else choices[int(rng.integers(len(choices)))]
                pair = _make_pair(rng, a, b, bool(rng.integers(2)), int(rng.integers(4, 11)))
                info = _decode(pair['input'])
                actual = info['donor']['corner'], info['target']['corner']
                # Enforce the visible held combination through input decoding.
                if (split == 'test' and actual != held) or (split == 'train' and actual == held):
                    continue
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('unique held-combination construction exhausted')
    return puzzle


def _measure(pair):
    info = _decode(pair['input'])
    d, t, wall = info['donor'], info['target'], info['wall']
    output = _grid(pair['output'])
    assert np.array_equal(output, _answer(info)), "wrong chamber correspondence"
    assert output.shape == t['crop'].shape
    assert np.array_equal(output == wall, t['crop'] == wall), "target walls changed"
    assert np.array_equal(output != t['crop'], t['labels'] >= 0), "must fill exactly the target rooms"
    for a, b in zip(d['order'], info['target_order']):
        assert np.all(output[t['labels'] == b] == d['colors'][a])
    signs = [int(np.sign(len(t['rooms'][b]) - len(d['rooms'][a])))
             for a, b in zip(d['order'], info['target_order'])]
    conflicts = _geometric_conflicts(d['labels'], t['labels'], d['corner'], t['corner'], info['matrix'])
    return dict(info=info, count=len(d['rooms']), anchors=(d['corner'], t['corner']),
                direction=info['direction'], matrix=tuple(info['matrix'].flat),
                shapes=(d['crop'].shape, t['crop'].shape), signs=set(signs),
                colors=len(set(d['colors'])), nonmetric=bool(conflicts),
                color_conflict=any(d['colors'][a] != d['colors'][b] for a, b in conflicts))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, "duplicate inputs"
            seen.add(key)
            assert _key(inp) != _key(out)
            measurements[split].append(_measure(pair))
    a, b = measurements['train'][:2]
    assert a['direction'] == 1 and b['direction'] == -1
    assert a['anchors'] == b['anchors'] and a['color_conflict']
    assert np.array_equal(_swap_target_markers(puzzle['train'][0])['input'], puzzle['train'][1]['input'])
    assert not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output'])
    assert not np.array_equal(_answer(b['info'], direction=1), puzzle['train'][1]['output']), "compass ablation invisible"
    assert all(m['anchors'] not in {t['anchors'] for t in measurements['train']}
               for m in measurements['test']), "missing train/test shift"
    return None


def self_test():
    """Bounded, streaming checks: 42 host seeds, counts, mutations and equivariance."""
    matrices, counts, shapes, anchors, roles, structures, signs, richness = (set() for _ in range(8))
    nonmetric = nonrectangular = mixed_area = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _measure(pair)
                info = m['info']
                matrices.add(m['matrix']); counts.add(m['count']); anchors.add(m['anchors'])
                shapes.add(m['shapes']); signs.update(m['signs']); richness.add(m['colors'])
                roles.add((info['background'], info['wall'], info['markers']))
                nonmetric += m['nonmetric']
                mixed_area += {-1, 1} <= m['signs']
                for figure in (info['donor'], info['target']):
                    mask = figure['labels'] < 0
                    structures.add(hashlib.sha256(bytes(mask.shape) + mask.tobytes()).digest())
                    nonrectangular += sum(len(p) < np.prod(p.max(0) - p.min(0) + 1) for p in figure['rooms'])
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
            for pair in puzzle['train'] + puzzle['test']:
                # Common rotation/reflection and arbitrary palette permutation
                # preserve the rule; these tests do not reuse generation labels.
                for transformed, expected in ((pair['input'].T, pair['output'].T),
                                               (np.rot90(pair['input']), np.rot90(pair['output']))):
                    assert np.array_equal(solve(transformed), expected)
                permutation = np.array([3, 8, 0, 6, 1, 9, 4, 2, 7, 5], dtype=np.int8)
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(matrices) == 8 and counts == set(range(4, 11))
    assert len(shapes) >= 70 and len(anchors) >= 12 and len(roles) >= 100
    assert len(structures) >= 200 and signs == {-1, 0, 1}
    assert richness >= {2, 3, 4, 5, 6} and nonmetric >= 100
    assert nonrectangular >= 100 and mixed_area >= 50
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(71 + 17 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(0, 2, 1)
    for kind in ('fill', 'wall', 'guide', 'crop'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad['test'][0]
        info = _decode(pair['input'])
        if kind == 'fill':
            p = tuple(info['target']['rooms'][0][0])
            pair['output'][p] = info['background']
        elif kind == 'wall':
            pair['output'][0, 0] = info['background']
        elif kind == 'crop':
            pair['output'] = pair['output'][1:]
        else:
            r, c, h, w = info['target']['bounds']
            side = next(iter(info['target']['guides']))
            p = {'T': (r - 1, c + w // 2), 'B': (r + h, c + w // 2),
                 'L': (r + h // 2, c - 1), 'R': (r + h // 2, c + w)}[side]
            pair['input'][p] = info['background']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{kind} mutation accepted')
    return {'seeds': 42, 'pairs': 210, 'symmetries': len(matrices), 'room_counts': sorted(counts),
            'geometry_pairs': len(shapes), 'frameworks': len(structures),
            'nonmetric_pairs': nonmetric, 'mixed_growth_shrinkage_pairs': mixed_area,
            'nonrectangular_chambers': int(nonrectangular), 'disk_writes': 0}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
