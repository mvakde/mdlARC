"""Jigsaw circuits: geometry and wire ports jointly determine assembly.

Reserved roles: 0 empty, 1 wire, 2 repair waypoint, 4 direction.
All other colors are board material. No imports from either parent.
"""
from collections import Counter
import hashlib
import json
import numpy as np


MATERIALS = (3, 5, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 188486507, 519203882, 277872021, 452379993, 140527700,
              96122833, 138685293, 934762303, 972033954, 530903367, 734149855,
              479494749, 206879521, 738779016, 163684784, 338767941, 67879512,
              120013721, 961023194, 256899445, 900950831, 411306476, 143876658,
              516200068, 835075290, 612997164, 833607316, 538202107, 502395580,
              474290909, 22698476, 680331045, 418373165, 472461571, 5034818,
              70428412, 715299706, 845918086, 235446577, 133681288, 248972508)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _components(cells):
    remaining, answer = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            for q in _neighbors(stack.pop(), remaining):
                remaining.remove(q)
                component.add(q)
                stack.append(q)
        answer.append(component)
    return answer


def _path(cells):
    """Order one simple path, or reject ambiguity/branching/loops."""
    cells = set(cells)
    ends = [p for p in cells if len(_neighbors(p, cells)) == 1]
    if len(ends) != 2 or any(len(_neighbors(p, cells)) > 2 for p in cells):
        raise ValueError('wire must be a simple open path')
    walk, previous, current = [], None, min(ends)
    while current is not None:
        walk.append(current)
        following = [p for p in _neighbors(current, cells) if p != previous]
        previous, current = current, following[0] if following else None
        if len(walk) > len(cells):
            raise ValueError('wire loop')
    if len(walk) != len(cells):
        raise ValueError('disconnected wire')
    return walk


def _line(a, b):
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError('nonorthogonal segment')
    d = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    return [(a[0]+i*d[0], a[1]+i*d[1])
            for i in range(max(abs(a[0]-b[0]), abs(a[1]-b[1]))+1)]


def _polyline(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError('expected a 1..30 integer ARC grid')
    return g.astype(np.int8, copy=True)


def _wire_cells(g):
    return {tuple(map(int, p)) for p in np.argwhere((g == 1) | (g == 2))}


def _tile(g, cells):
    origin = np.min(list(cells), axis=0)
    local = np.array(sorted(cells), dtype=int)-origin
    h, w = local.max(axis=0)+1
    raw = np.zeros((h, w), dtype=np.int8)
    absolute = local+origin
    raw[local[:, 0], local[:, 1]] = g[absolute[:, 0], absolute[:, 1]]
    colors = set(map(int, np.unique(raw))) & set(MATERIALS)
    if len(colors) != 1:
        raise ValueError('each tile has one material color')
    low, high = [], []
    for c in range(w):
        rows = np.flatnonzero(raw[:, c])
        if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1]+1)):
            raise ValueError('tiles are column-convex full-width layers')
        low.append(int(rows[0]))
        high.append(int(rows[-1])+1)
    walk = _path(_wire_cells(raw))
    ends = [walk[0], walk[-1]]
    entry = [p for p in ends if p[0] == low[p[1]]]
    exit_ = [p for p in ends if p[0] == high[p[1]]-1]
    if len(entry) != 1 or len(exit_) != 1 or entry[0] == exit_[0]:
        raise ValueError('tile must have one upper and one lower wire port')
    return {'raw': raw, 'origin': origin, 'low': tuple(low), 'high': tuple(high),
            'entry': entry[0][1], 'exit': exit_[0][1], 'walk': walk,
            'material': next(iter(colors))}


def _parse_canonical(g):
    components = _components(map(tuple, np.argwhere(g != 0)))
    spanning = [s for s in components if len({p[1] for p in s}) == g.shape[1]]
    if len(spanning) != 1:
        raise ValueError('one horizontally spanning band required')
    band = spanning[0]
    others = [s for s in components if s is not band]
    if not 2 <= len(others) <= 4:
        raise ValueError('expected two to four movable tiles')
    top, edge = [], []
    for c in range(g.shape[1]):
        rows = sorted(r for r, col in band if col == c)
        if rows != list(range(rows[0], rows[-1]+1)):
            raise ValueError('band columns must be solid')
        top.append(rows[0])
        edge.append(rows[-1]+1)
    if len(set(top)) != 1 or len(set(edge)) == 1 or min(edge)-top[0] < 2:
        raise ValueError('band must have flat back and jagged front')
    colors = {int(g[p]) for p in band} & set(MATERIALS)
    if len(colors) != 1 or any(g[p] == 4 for p in band):
        raise ValueError('invalid band material/marker')
    band_walk = _path({p for p in band if g[p] in (1, 2)})
    endpoints = (band_walk[0], band_walk[-1])
    roots = [p for p in endpoints if p[0] == edge[p[1]]-1 and 0 < p[1] < g.shape[1]-1]
    feeds = [p for p in endpoints if p[1] in (0, g.shape[1]-1)]
    if len(roots) != 1 or len(feeds) != 1:
        raise ValueError('band wire must lead from grid edge to a front port')
    base = np.zeros_like(g)
    for p in band:
        base[p] = g[p]
    tiles = [_tile(g, cells) for cells in others]
    if sum(np.count_nonzero(t['raw'] == 4) for t in tiles) != 1:
        raise ValueError('exactly one direction marker required')
    return base, np.array(edge), tiles, roots[0][1], next(iter(colors)), feeds[0]


def _assemblies(g, use_ports=True):
    """Enumerate all boundary windows and orders, deduplicate colored boards."""
    base, edge, tiles, root, band_color, feed = _parse_canonical(g)
    answers = {}
    for width in sorted({t['raw'].shape[1] for t in tiles}):
        if any(t['raw'].shape[1] != width for t in tiles):
            continue
        for start in range(g.shape[1]-width+1):
            initial = tuple(map(int, edge[start:start+width]))

            def extend(profile, port, used, placements):
                offset = min(profile)
                normalized = tuple(x-offset for x in profile)
                if len(used) == len(tiles):
                    if len(set(profile)) != 1:
                        return
                    board = base.copy()
                    material = np.zeros_like(g)
                    material[base != 0] = band_color
                    owner = np.full(g.shape, -1, dtype=int)
                    for i, r in placements:
                        t = tiles[i]
                        rr, cc = np.nonzero(t['raw'])
                        if np.any(board[rr+r, cc+start]):
                            return
                        board[rr+r, cc+start] = t['raw'][rr, cc]
                        material[rr+r, cc+start] = t['material']
                        owner[rr+r, cc+start] = i
                    info = {'tiles': tiles, 'placements': placements, 'start': start,
                            'width': width, 'initial': initial, 'end': profile[0],
                            'material': material, 'owner': owner, 'feed': feed,
                            'base': base}
                    answers.setdefault(board.tobytes(), (board, info))
                    return
                for i, t in enumerate(tiles):
                    if i in used or t['low'] != normalized:
                        continue
                    if use_ports and start+t['entry'] != port:
                        continue
                    new = tuple(offset+x for x in t['high'])
                    if max(new) > g.shape[0]:
                        continue
                    extend(new, start+t['exit'], used | {i}, placements+[(i, offset)])

            extend(initial, root, set(), [])
    return list(answers.values())


def _route(board, info):
    """Interpret the transported marker in the assembled material domain."""
    wire = _wire_cells(board)
    walk = _path(wire)
    feed = info['feed']
    if walk[-1] == feed:
        walk.reverse()
    if walk[0] != feed or walk[-1][0] != info['end']-1:
        raise ValueError('wire must connect band feed to outer board edge')
    q = tuple(map(int, np.argwhere(board == 4)[0]))
    candidates = []
    for d in DIRS:
        p = q[0]-d[0], q[1]-d[1]
        old_start = p[0]-d[0], p[1]-d[1]
        if p not in wire or old_start not in wire:
            continue
        j = walk.index(p)
        if j == 0 or j == len(walk)-1 or walk[j+1] != old_start:
            continue
        incoming = walk[j-1][0]-p[0], walk[j-1][1]-p[1]
        if incoming[0]*d[0]+incoming[1]*d[1] != 0:
            continue
        ray, at = [], q
        while _inside(at, board.shape) and info['material'][at] != 0:
            ray.append(at)
            at = _add(at, d)
        if ray:
            candidates.append((p, d, j, ray))
    if len(candidates) != 1:
        raise ValueError('direction marker must determine one terminal branch')
    p, d, j, ray = candidates[0]
    out = info['material'].copy()
    for point in walk[:j+1]+ray:
        out[point] = 1
    info = dict(info, pivot=p, direction=d, old=walk[j+1:], ray=ray,
                walk=walk, board=board,
                deleted_tiles=len({int(info['owner'][x]) for x in walk[j+1:] if info['owner'][x] >= 0}))
    return out, info


def _solve_details(inp):
    g = _grid(inp)
    results = {}
    for k in range(4):
        canonical = np.rot90(g, k)
        try:
            assemblies = _assemblies(canonical)
            # Uniqueness is required BEFORE routing can erase distinguishing data.
            if len(assemblies) != 1:
                continue
            out, info = _route(*assemblies[0])
            out = np.rot90(out, -k).copy()
            results.setdefault(out.tobytes(), (out, dict(info, rotation=k)))
        except ValueError:
            continue
    if len(results) != 1:
        raise ValueError('input has no unique circuit assembly and rerouting')
    return next(iter(results.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver: no seed, training examples, RNG, or hidden state."""
    return _solve_details(input_grid)[0]


def _dilate(mask):
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for r in range(3):
        for c in range(3):
            out |= padded[r:r+mask.shape[0], c:c+mask.shape[1]]
    return out


def _scatter(rng, base, raw_tiles):
    # At most 36 attempts, four tiles, 900 origins: bounded memory, no files.
    for _ in range(36):
        inp = base.copy()
        for i in rng.permutation(len(raw_tiles)):
            tile = raw_tiles[int(i)]
            h, w = tile.shape
            H, W = inp.shape
            forbidden = _dilate(inp != 0)
            valid = np.ones((H-h+1, W-w+1), dtype=bool)
            rr, cc = np.nonzero(tile)
            for r, c in zip(rr, cc):
                valid &= ~forbidden[r:r+H-h+1, c:c+W-w+1]
            choices = np.argwhere(valid)
            if not len(choices):
                break
            r, c = choices[int(rng.integers(len(choices)))]
            inp[rr+r, cc+c] = tile[rr, cc]
        else:
            return inp
    raise ValueError('scatter clearance')


def _make_pair(rng, n, pivot_tile, rotation):
    width = int(rng.integers(5, 10))
    amplitude = int(rng.integers(1, 3))
    profile = rng.integers(0, amplitude+1, size=width)
    profile[int(rng.integers(1, width-1))] = 0
    profile[0] = int(rng.integers(1, amplitude+1))
    profile[-1] = int(rng.integers(1, amplitude+1))
    amplitude = int(profile.max())
    profiles = [profile.copy() for _ in range(n)]
    # Four-tile boards can change their interlock contour. Two loops sharing
    # one profile remain geometrically exchangeable; ports resolve their order.
    if n == 4 and rng.random() < .65:
        other = rng.integers(0, amplitude+1, size=width)
        other[int(rng.integers(width))] = 0
        if len(set(map(int, other))) == 1:
            other[0] = 1
        change = 1 if rng.random() < .5 else 3
        profiles[change:] = [other.copy() for _ in range(n-change)]
    depths = [int(profiles[i].max())+int(rng.integers(4, 6)) for i in range(n-1)]
    cap = int(profiles[-1].max())+int(rng.integers(4, 7))
    core = int(rng.integers(2, 4))
    top = int(rng.integers(0, 3))
    required = top+core+sum(depths)+cap
    if required > 30:
        raise ValueError('board too long')
    H = int(rng.integers(max(23, required), 31))
    W = int(rng.integers(max(22, 2*width+3), 31))
    start = int(rng.integers(2, W-width-1))
    b = top+core
    ports = list(map(int, rng.choice(width, n+1, replace=False)))
    if ports[pivot_tile] in (0, width-1):
        raise ValueError('direction marker would leave its tile')
    band_color = int(rng.choice(MATERIALS))
    colors = list(map(int, rng.choice(MATERIALS, n, replace=True)))
    base = np.zeros((H, W), dtype=np.int8)
    base[top:b, :] = band_color
    for c, p in enumerate(profile):
        base[b:b+p, start+c] = band_color
    feed_side = int(rng.integers(2))
    feed = (b-1, 0 if feed_side == 0 else W-1)
    root = (b+int(profile[ports[0]])-1, start+ports[0])
    band_wire = _polyline([feed, (b-1, root[1]), root])
    for point in band_wire:
        base[point] = 1
    # A retained repair in the fixed band and a deleted repair downstream.
    band_repair = band_wire[int(rng.integers(len(band_wire)))]
    base[band_repair] = 2
    raw_tiles, tile_walks = [], []
    offset = b
    intended = base.copy()
    material = np.zeros_like(base)
    material[base != 0] = band_color
    for i in range(n):
        lower = profiles[i]
        upper = profiles[i+1]+depths[i] if i < n-1 else np.full(width, cap)
        raw = np.zeros((int(upper.max()), width), dtype=np.int8)
        for c in range(width):
            raw[int(lower[c]):int(upper[c]), c] = colors[i]
        middle = int(rng.integers(int(lower.max())+1, int(upper.min())-1))
        walk = _polyline([(int(lower[ports[i]]), ports[i]),
                          (middle, ports[i]), (middle, ports[i+1]),
                          (int(upper[ports[i+1]])-1, ports[i+1])])
        _path(walk)
        for point in walk:
            raw[point] = 1
        if i == pivot_tile:
            pivot = (middle, ports[i])
            d = (0, -int(np.sign(ports[i+1]-ports[i])))
            raw[_add(pivot, d)] = 4
            global_pivot = (offset+middle, start+ports[i])
            global_direction = d
        # Include marked bends and endpoints, and markers on erased branches.
        for at in rng.choice(len(walk), int(rng.integers(0, min(3, len(walk))+1)), replace=False):
            raw[walk[int(at)]] = 2
        if i == n-1:
            raw[walk[-1]] = 2
        rr, cc = np.nonzero(raw)
        intended[rr+offset, cc+start] = raw[rr, cc]
        material[rr+offset, cc+start] = colors[i]
        tile_walks.append([(r+offset, c+start) for r, c in walk])
        raw_tiles.append(raw)
        if i < n-1:
            offset += depths[i]
    complete_walk = band_wire+sum(tile_walks, [])
    at = complete_walk.index(global_pivot)
    expected = material.copy()
    for point in complete_walk[:at+1]:
        expected[point] = 1
    point = _add(global_pivot, global_direction)
    while _inside(point, expected.shape) and material[point] != 0:
        expected[point] = 1
        point = _add(point, global_direction)
    inp = _scatter(rng, base, raw_tiles)
    if np.count_nonzero(inp == 0) <= inp.size//2:
        raise ValueError('background must remain dominant')
    if len(_assemblies(inp)) != 1 or len(_assemblies(inp, use_ports=False)) < 2:
        raise ValueError('ports must resolve a real geometric ambiguity')
    predicted = solve(inp)
    if not np.array_equal(predicted, expected):
        raise ValueError('construction differs from input-only solve')
    if rng.random() < .5:
        inp, expected = inp[:, ::-1], expected[:, ::-1]
    return {'input': np.rot90(inp, rotation).copy(),
            'output': np.rot90(expected, rotation).copy()}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= value <= high:
        raise ValueError(name+' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(2))
    orientation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            n = (3 if split == 'train' else 4) if shift == 0 else int(rng.integers(3, 5))
            pivot = (0 if index == 0 else n-1) if split == 'train' and index < 2 else int(rng.integers(n))
            k = (orientation if split == 'train' else (orientation+1)%4) if shift == 1 else int(rng.integers(4))
            for _ in range(160):
                try:
                    pair = _make_pair(rng, n, pivot, k)
                except ValueError:
                    continue
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded circuit construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError('puzzle keys')
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError('split types')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 2, 10, 'num_train')
    _count(nq, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('count ordering')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError('pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            if inp.shape != out.shape or np.array_equal(inp, out):
                raise ValueError('shape/identity')
            key = inp.shape, inp.tobytes()
            if key in seen:
                raise ValueError('duplicate input')
            seen.add(key)
            expected, info = _solve_details(inp)
            if not np.array_equal(expected, out):
                raise ValueError('full output violates assembly/routing')
            canonical = np.rot90(inp, info['rotation'])
            geometric = _assemblies(canonical, use_ports=False)
            if len(geometric) < 2:
                raise ValueError('wire continuity is not necessary for assembly')
            material = info['material']
            assembled = info['board']
            if np.any(np.isin(out, (2, 4))):
                raise ValueError('unremoved annotations')
            # Count silhouettes, exact translated material, and band preservation.
            assert np.count_nonzero(inp) == np.count_nonzero(out)
            assert np.array_equal(assembled != 0, material != 0)
            assert all(np.any(t['raw'] == t['material']) for t in info['tiles'])
            # B ablation: assembled+repaired board with no branch change differs.
            no_redirect = np.where(assembled == 2, 1, assembled)
            no_redirect[no_redirect == 4] = material[no_redirect == 4]
            assert not np.array_equal(no_redirect, np.rot90(out, info['rotation']))
            assert any(assembled[p] == 2 for p in info['old'])
            assert any(assembled[p] == 2 for p in info['walk'] if p not in info['old'])
            if split == 'train':
                training.append(info)
    # Two demonstrations suffice: one deletion traverses joins, one is local.
    assert any(i['deleted_tiles'] >= 2 for i in training)
    assert any(i['deleted_tiles'] == 1 for i in training)


def _signature(puzzle):
    return tuple((s, p['input'].shape, p['input'].tobytes(), p['output'].tobytes())
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """42 seeds, streamed; compact scalar evidence, no generated files."""
    shapes, structures, profiles, palettes, directions, tile_counts = set(), set(), set(), set(), set(), set()
    stats = Counter()
    depths, widths, marker_tiles, interface_counts, old_bends, ray_lengths = set(), set(), set(), set(), set(), set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                _, info = _solve_details(inp)
                shapes.add(inp.shape)
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape)+norm).hexdigest())
                profiles.add(tuple(x-min(info['initial']) for x in info['initial']))
                palettes.add(tuple(sorted(set(map(int, np.unique(inp))) & set(MATERIALS))))
                d = info['direction']
                for _ in range((-info['rotation']) % 4):
                    d = (-d[1], d[0])
                directions.add(d)
                tile_counts.add(len(info['tiles']))
                widths.add(info['width'])
                depths.update(t['raw'].shape[0] for t in info['tiles'])
                interface_counts.add(len({t['low'] for t in info['tiles']}))
                marker_tiles.add(next(j for j, (i, _) in enumerate(info['placements'])
                                      if np.any(info['tiles'][i]['raw'] == 4)))
                steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(info['old'], info['old'][1:])]
                old_bends.add(sum(a != b for a, b in zip(steps, steps[1:])))
                ray_lengths.add(len(info['ray']))
                stats['pairs'] += 1
                stats['cross_tile_deletions'] += info['deleted_tiles'] >= 2
                stats['local_deletions'] += info['deleted_tiles'] == 1
                stats['repeated_material'] += len({t['material'] for t in info['tiles']}) < len(info['tiles'])
                stats['marked_pivots'] += info['board'][info['pivot']] == 2
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
        assert _signature(p) == _signature(generate_puzzle(seed))
    assert len(shapes) >= 30 and len(structures) >= 200 and len(profiles) >= 40
    assert len(palettes) >= 15 and len(directions) == 4 and tile_counts == {3, 4}
    assert stats['cross_tile_deletions'] >= 42 and stats['local_deletions'] >= 42
    assert stats['repeated_material'] >= 25 and stats['marked_pivots'] > 0
    assert widths == set(range(5, 10)) and len(depths) >= 4
    assert interface_counts >= {1, 2} and marker_tiles >= {0, 1, 2, 3}
    assert max(old_bends) >= 5 and min(ray_lengths) == 1
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            q = generate_puzzle(71, nt, nq)
            assert len(q['train']) == nt and len(q['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0, 2, 1)
    for site in ((0, 0), tuple(np.argwhere(p['test'][0]['output'] == 1)[0])):
        broken = {'train': p['train'], 'test': [dict(p['test'][0])]}
        broken['test'][0]['output'] = p['test'][0]['output'].copy()
        broken['test'][0]['output'][site] = (int(broken['test'][0]['output'][site])+1)%10
        try:
            validate_generated_puzzle(broken)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError('mutated answer accepted')
    # Specific wrong answers: retain an erased bend, miss a new ray cell,
    # erase substrate, or fail to repair a waypoint in the fixed band.
    inp = p['test'][0]['input']
    out, info = _solve_details(inp)
    canonical = np.rot90(out, info['rotation'])
    sites = [info['old'][-1], info['ray'][0], info['old'][0],
             next(tuple(map(int, x)) for x in np.argwhere(info['base'] == 2))]
    for point in sites:
        corrupt = canonical.copy()
        corrupt[point] = (int(corrupt[point])+1) % 10
        broken = {'train': p['train'], 'test': [{'input': inp, 'output': np.rot90(corrupt, -info['rotation'])}]}
        try:
            validate_generated_puzzle(broken)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError('family-specific wrong answer accepted')
    result = {k: int(v) for k, v in stats.items()}
    result.update(seeds=42, shapes=len(shapes), normalized_structures=len(structures),
                  profiles=len(profiles), material_palettes=len(palettes), orientations=len(directions),
                  widths=sorted(widths), tile_depths=sorted(depths),
                  interface_counts=sorted(interface_counts), pivot_tiles=sorted(marker_tiles),
                  removed_branch_bends=sorted(old_bends), ray_lengths=sorted(ray_lengths))
    return result


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
