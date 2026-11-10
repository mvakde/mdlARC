"""Ranked bridges: a unary-controlled, consuming walk by a rigid arrow.

Only the standard library and NumPy are used. The decoder reads the key tablet,
actual arrow bitmap, station colors, and capped bridges from the input pixels.
No sampled route, seed, or example is consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PORT_DISTANCE = 3


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=False)


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "motion must be orthogonal and nonzero")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _pose(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c) for r, c in
            (_rotate(p, direction - 1) for p in mask)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _arrow(span=1, extension=1, solid=False, tail=0):
    cells = {(0, -j) for j in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - extension - depth) for r in rows)
    for j in range(1, tail + 1):
        cells.update((r, 1 - extension - span - j) for r in (-span, span))
    return tuple(sorted(cells))


SMALL_MASKS = tuple(_arrow(1, 1, solid, tail)
                    for solid in (False, True) for tail in (0, 1))
NEW_MASKS = tuple(_arrow(span, extension, solid)
                  for span, extension in ((2, 1), (1, 2)) for solid in (False, True))
MASKS = SMALL_MASKS + NEW_MASKS


def _identify_arrow(cells):
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            offsets = {_rotate((r - tip[0], c - tip[1]), 1 - d) for r, c in cells}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in offsets) == 0:
                continue
            if {(-r, c) for r, c in offsets} == offsets:
                matches.append((tuple(sorted(offsets)), tip, d))
    _require(len(matches) == 1, "arrow needs one symmetry axis and projecting tip")
    mask, tip, d = matches[0]
    _require(max(max(abs(r), abs(c)) for r, c in mask) <= 2, "arrow exceeds port clearance")
    return mask, tip, d


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, component = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in unseen:
                        unseen.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _read_tablet(a, bg):
    candidates = []
    for frame in np.unique(a):
        frame = int(frame)
        if frame == bg:
            continue
        positions = np.argwhere(a == frame)
        r0, c0 = positions.min(axis=0)
        r1, c1 = positions.max(axis=0)
        if not (3 <= r1 - r0 + 1 <= 5 and 3 <= c1 - c0 + 1 <= 5):
            continue
        block = a[r0:r1 + 1, c0:c1 + 1]
        ring = np.ones(block.shape, dtype=bool)
        ring[1:-1, 1:-1] = False
        if not (np.all(block[ring] == frame) and np.all(block[~ring] != frame)
                and int(np.count_nonzero(a == frame)) == int(np.count_nonzero(ring))):
            continue
        core = block[1:-1, 1:-1]
        valid = False
        # Orientation does not change the counts. A square tablet may have
        # more than one equivalent reading direction, with the same commands.
        for rotation in range(4):
            b = np.rot90(core, rotation)
            colors = []
            for row in b:
                occupied = np.flatnonzero(row != bg)
                if not len(occupied) or not np.array_equal(occupied, np.arange(len(occupied))):
                    break
                color = int(row[0])
                if not np.all(row[occupied] == color):
                    break
                colors.append(color)
            else:
                valid |= len(set(colors)) == len(colors)
        if valid:
            ranks = {int(c): int(np.count_nonzero(core == c)) for c in np.unique(core) if c != bg}
            if all(1 <= n <= 3 for n in ranks.values()) and 1 <= len(ranks) <= 3:
                candidates.append((frame, (int(r0), int(c0), int(r1), int(c1)), ranks))
    _require(len(candidates) == 1, "missing or ambiguous framed unary tablet")
    return candidates[0]


def _decode(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(frequencies[bg] > a.size // 2, "background must occupy a majority")
    frame, box, ranks = _read_tablet(a, bg)
    occupied = set(zip(*np.where(a != bg)))
    r0, c0, r1, c1 = box
    occupied -= {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    nodes, bridges, movers = {}, [], []
    for comp in _components(occupied):
        if len(comp) == 1:
            p = next(iter(comp))
            color = int(a[p])
            _require(color in ranks, "station has no unary command")
            nodes[p] = color
            continue
        rows, cols = {r for r, c in comp}, {c for r, c in comp}
        if len(rows) == 1 or len(cols) == 1:
            cells = sorted(comp)
            values = [int(a[p]) for p in cells]
            _require(len(cells) >= 3 and values[0] == values[-1] and values[0] in ranks,
                     "bridge must have matching control caps")
            _require(values[1] not in set(ranks) | {bg, frame}
                     and all(v == values[1] for v in values[1:-1]), "bad bridge payload")
            _require(all(abs(x[0] - y[0]) + abs(x[1] - y[1]) == 1
                         for x, y in zip(cells, cells[1:])), "broken bridge")
            direction = (0, 1) if len(rows) == 1 else (1, 0)
            first, last = cells[0], cells[-1]
            u = tuple(first[i] - PORT_DISTANCE * direction[i] for i in (0, 1))
            v = tuple(last[i] + PORT_DISTANCE * direction[i] for i in (0, 1))
            bridges.append(dict(a=u, b=v, cap=values[0], payload=values[1],
                                cells=tuple(cells), length=len(cells)))
        else:
            colors = {int(a[p]) for p in comp}
            _require(len(colors) == 1 and not colors.intersection(set(ranks) | {bg, frame}),
                     "mover must be uniform in its own color")
            mask, tip, facing = _identify_arrow(comp)
            movers.append((next(iter(colors)), mask, tip, facing, comp))
    _require(len(movers) == 1 and nodes and bridges, "need one mover and a station network")
    ink, mask, tip, facing, initial_cells = movers[0]
    _require(all(e['payload'] != ink for e in bridges), "mover and payload roles overlap")
    adjacency = {p: [] for p in nodes}
    seen_edges = set()
    for i, e in enumerate(bridges):
        _require(e['a'] in nodes and e['b'] in nodes, "bridge ports lack stations three cells away")
        key = frozenset((e['a'], e['b']))
        _require(key not in seen_edges, "duplicate bridge")
        seen_edges.add(key)
        adjacency[e['a']].append(i)
        adjacency[e['b']].append(i)
    _require(all(adjacency.values()), "isolated station")
    connected, stack = {next(iter(nodes))}, [next(iter(nodes))]
    while stack:
        p = stack.pop()
        for i in adjacency[p]:
            e = bridges[i]
            q = e['b'] if p == e['a'] else e['a']
            if q not in connected:
                connected.add(q)
                stack.append(q)
    _require(len(connected) == len(nodes), "network must be connected")
    for p in nodes:
        for cap in ranks:
            lengths = [bridges[i]['length'] for i in adjacency[p] if bridges[i]['cap'] == cap]
            _require(len(lengths) == len(set(lengths)), "tied local bridge lengths")
        for d in range(4):
            cells = _pose(mask, p, d)
            _require(_inside(cells, a.shape), "station cannot accommodate the mover")
            _require(all(q == p or int(a[q]) == bg for q in cells), "station clearance is obstructed")
    dr, dc = DIRS[facing]
    forward = [p for p in nodes if (p[0] - tip[0]) * dc == (p[1] - tip[1]) * dr
               and (p[0] - tip[0]) * dr + (p[1] - tip[1]) * dc > 0]
    _require(forward, "arrow has no station ahead")
    first = min(forward, key=lambda p: abs(p[0] - tip[0]) + abs(p[1] - tip[1]))
    # The initial unbridged approach must cross background only.
    distance = abs(first[0] - tip[0]) + abs(first[1] - tip[1])
    _require(distance >= 3 and all(int(a[tip[0] + j * dr, tip[1] + j * dc]) == bg
                                 for j in range(1, distance)), "approach is obstructed")
    return dict(grid=a, bg=bg, frame=frame, box=box, ranks=ranks, nodes=nodes,
                bridges=bridges, adjacency=adjacency, ink=ink, mask=mask, tip=tip,
                facing=facing, initial_cells=set(initial_cells), first=first)


def _walk(info, ranks=None, ignore_color=False):
    """Used edges leave the local competition; station colors survive logically."""
    ranks = info['ranks'] if ranks is None else ranks
    p = info['first']
    stations, used, directions, decisions = [p], [], [info['facing']], []
    while True:
        color = info['nodes'][p]
        eligible = [i for i in info['adjacency'][p] if i not in used
                    and (ignore_color or info['bridges'][i]['cap'] == color)]
        ordered = sorted(eligible, key=lambda i: info['bridges'][i]['length'], reverse=True)
        n = ranks[color]
        decisions.append((p, color, n, tuple(ordered)))
        if len(ordered) < n:
            break
        i = ordered[n - 1]
        used.append(i)
        e = info['bridges'][i]
        q = e['b'] if p == e['a'] else e['a']
        directions.append(_direction(p, q))
        stations.append(q)
        p = q
        _require(len(used) <= len(info['bridges']), "walk did not terminate")
    return dict(stations=stations, used=used, directions=directions, decisions=decisions,
                final=p, final_direction=directions[-1])


def _output(info, walk, rotate=True):
    out = info['grid'].copy()
    for p in info['initial_cells'] | set(walk['stations']):
        out[p] = info['bg']
    for i in walk['used']:
        e = info['bridges'][i]
        for p in e['cells'][1:-1]:
            out[p] = e['cap']
    direction = walk['final_direction'] if rotate else info['facing']
    for p in _pose(info['mask'], walk['final'], direction):
        out[p] = info['ink']
    return out


def _consumption_witness(info, walk):
    """At the stop, counting used edges would restore the requested rank."""
    p, color, n, choices = walk['decisions'][-1]
    original_count = sum(info['bridges'][i]['cap'] == color for i in info['adjacency'][p])
    return len(choices) < n <= original_count


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _output(info, _walk(info))


def _signature(a):
    return a.shape, a.tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def _tree_edges(rng, ids, options, mandatory, cyclic):
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    chosen = list(mandatory)
    for u, v in mandatory:
        _require(find(u) != find(v), 'mandatory cycle in tree constructor')
        parent[find(u)] = find(v)
    remaining = [e for e in options if e not in chosen]
    rng.shuffle(remaining)
    for u, v in remaining:
        if find(u) != find(v):
            chosen.append((u, v))
            parent[find(u)] = find(v)
    if cyclic:
        remaining = [e for e in options if e not in chosen]
        rng.shuffle(remaining)
        count = int(rng.integers(1, len(remaining) + 1))
        chosen.extend(remaining[:count])
    return chosen


def _render_scene(scene, ranks):
    a = np.full(scene['shape'], scene['bg'], dtype=np.int8)
    top, left = scene['key_origin']
    h = len(scene['controls']) + 2
    a[top:top + h, left:left + 5] = scene['frame']
    a[top + 1:top + h - 1, left + 1:left + 4] = scene['bg']
    for j, c in enumerate(scene['key_order']):
        a[top + 1 + j, left + 1:left + 1 + ranks[c]] = c
    for i, p in scene['positions'].items():
        _require(int(a[p]) == scene['bg'], 'station overlaps tablet')
        a[p] = scene['node_colors'][i]
    for u, v, cap, payload in scene['edges']:
        p, q = scene['positions'][u], scene['positions'][v]
        d = _direction(p, q)
        dr, dc = DIRS[d]
        length = abs(p[0] - q[0]) + abs(p[1] - q[1])
        _require(length >= 8, 'bridge too short')
        for j in range(3, length - 2):
            point = (p[0] + j * dr, p[1] + j * dc)
            _require(int(a[point]) == scene['bg'], 'bridge overlaps another object')
            a[point] = cap if j in (3, length - 3) else payload
    for p in _pose(scene['mask'], scene['tip'], 1):
        _require(_inside({p}, a.shape) and int(a[p]) == scene['bg'], 'initial arrow overlaps')
        a[p] = scene['ink']
    return a


def _scene(rng, mask, palette, *, contrast=False, cyclic=False, rank=1,
           all_one=False, jitter=True):
    """Build an embedded graph; branches are not whole parent puzzles."""
    nr, nc = (3, 3) if contrast else (int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    f = 1 if all_one else (int(rng.integers(2, 4)) if contrast
                           else int(rng.integers(1, min(3, (nr * nc - 1) // 2) + 1)))
    span = max(abs(r) for r, c in mask)
    depth = -min(c for r, c in mask)
    clearance = max(span, depth)
    bg, frame, ink = (int(c) for c in palette[:3])
    payloads = [int(c) for c in palette[3:5]]
    controls = [int(c) for c in palette[5:5 + f]]
    start_row = 1 if contrast else int(rng.integers(nr))
    start = start_row * nc
    approach = int(rng.integers(3, 6))
    key_top = int(rng.integers(2))
    row0 = key_top + f + 2 + clearance + 1
    for _ in range(200):
        if contrast:
            # Center arrival is longest or middle. Its wrong-color outgoing
            # bridge is longest, so local color grouping is visibly necessary.
            if bool(rng.integers(2)):
                rg, cg = [8, 9], [10, 11]
            else:
                rg, cg = [8, 10], [9, 11]
            rng.shuffle(rg)
        else:
            pool = np.arange(8, 12 if nr == nc == 3 else 18)
            gaps = [int(x) for x in rng.choice(pool, nr + nc - 2, replace=False)]
            rg, cg = gaps[:nr - 1], gaps[nr - 1:]
        col_min = approach + depth
        col_max = 29 - sum(cg) - clearance
        if row0 + sum(rg) + clearance < 30 and col_min <= col_max:
            break
    else:
        raise AssertionError('axis budget exhausted')
    col0 = int(rng.integers(col_min, min(col_max, col_min + 2) + 1))
    rows = [row0] + [row0 + sum(rg[:j]) for j in range(1, nr)]
    cols = [col0] + [col0 + sum(cg[:j]) for j in range(1, nc)]
    h, w = rows[-1] + clearance + 1, cols[-1] + clearance + 1
    h += int(rng.integers(0, min(2, 30 - h) + 1))
    w += int(rng.integers(0, min(2, 30 - w) + 1))
    positions = {r * nc + c: (rows[r], cols[c]) for r in range(nr) for c in range(nc)}
    options = []
    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            if c + 1 < nc:
                options.append((i, i + 1))
            if r + 1 < nr:
                options.append((i, i + nc))
    incident = [e for e in options if start in e]
    if contrast:
        center = start + 1
        mandatory = list(dict.fromkeys(incident + [e for e in options if center in e]))
    else:
        # Ensure an initial selection exists. A rank-two corner has degree two.
        rank = min(rank, len(incident))
        mandatory = incident
    edges = _tree_edges(rng, positions, options, mandatory, cyclic)
    if not contrast and not cyclic and rng.random() < 0.65:
        for _ in range(int(rng.integers(1, 3))):
            degrees = Counter(i for e in edges for i in e)
            leaves = [i for i in positions if i != start and degrees[i] == 1]
            if len(positions) <= 4 or not leaves:
                break
            i = int(rng.choice(leaves))
            edges = [e for e in edges if i not in e]
            del positions[i]
    node_colors = {i: int(rng.choice(controls)) for i in positions}
    node_colors[start] = controls[0]
    if all_one:
        node_colors = {i: controls[0] for i in positions}
    elif contrast:
        center = start + 1
        node_colors[center] = controls[1]
        if f == 3:
            for i in (2, 8):
                node_colors[i] = controls[2]
    else:
        free = [i for i in positions if i != start]
        rng.shuffle(free)
        for i, color in zip(free, controls[1:]):
            node_colors[i] = color
    common_payload = bool(rng.integers(2))
    records = []
    for u, v in edges:
        cap = int(rng.choice((node_colors[u], node_colors[v])))
        if start in (u, v):
            cap = controls[0]
        elif contrast and start + 1 in (u, v):
            cap = controls[0] if start + 2 in (u, v) else controls[1]
        elif contrast and f == 3 and (2 in (u, v) or 8 in (u, v)):
            cap = controls[2]
        records.append((u, v, cap, payloads[0] if common_payload else int(rng.choice(payloads))))
    key_order = [int(c) for c in rng.permutation(controls)]
    ranks = {c: int(rng.integers(1, 3)) for c in controls}
    ranks[controls[0]] = rank
    if contrast:
        ranks[controls[1]] = 1
    if all_one:
        ranks[controls[0]] = 1
    scene = dict(shape=(h, w), bg=bg, frame=frame, ink=ink, controls=controls,
                 key_order=key_order, key_origin=(key_top, int(rng.integers(w - 4))),
                 positions=positions, node_colors=node_colors, edges=records,
                 mask=mask, tip=(positions[start][0], positions[start][1] - approach),
                 ranks=ranks, start=start)
    # Sliding degree-one stations along their bridge breaks the Cartesian
    # scaffold without changing connectivity. Full pixel decoding checks each
    # tentative deformation, including intersections, clearance, and ties.
    if jitter:
        degrees = Counter(i for e in edges for i in e)
        locked = {start}
        if contrast:
            locked |= {i for e in mandatory for i in e}
        leaves = [i for i in positions if degrees[i] == 1 and i not in locked]
        rng.shuffle(leaves)
        for i in leaves[:3]:
            edge = next(e for e in edges if i in e)
            other = edge[1] if edge[0] == i else edge[0]
            p, old = positions[other], positions[i]
            dr, dc = DIRS[_direction(p, old)]
            for length in rng.permutation(np.arange(8, 19))[:5]:
                positions[i] = (p[0] + dr * int(length), p[1] + dc * int(length))
                try:
                    _require(_inside({positions[i]}, scene['shape']), 'leaf out of bounds')
                    _decode(_render_scene(scene, ranks))
                except (AssertionError, IndexError):
                    positions[i] = old
                    continue
                break
    return scene


def _transform(a, rotation, reflected):
    return np.ascontiguousarray(np.rot90(np.fliplr(a) if reflected else a, rotation))


def _pair(scene, ranks, rotation, reflected):
    a = _render_scene(scene, ranks)
    # Labels are built from a fresh input decode, never the scene's edge IDs.
    b = solve(a)
    return dict(input=_transform(a, rotation, reflected), output=_transform(b, rotation, reflected))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('rank', 'topology', 'mask')))
    palette = [int(c) for c in rng.permutation(10)]
    shared_palette = bool(rng.random() < 0.2)
    rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
    mask = SMALL_MASKS[int(rng.integers(len(SMALL_MASKS)))]
    for _ in range(60):
        scene = _scene(rng, mask, palette, contrast=True,
                       cyclic=regime != 'topology' and bool(rng.integers(2)))
        ranks1, ranks2 = dict(scene['ranks']), dict(scene['ranks'])
        anchor = scene['controls'][0]
        ranks1[anchor], ranks2[anchor] = 1, 2
        p1, p2 = _pair(scene, ranks1, rotation, reflected), _pair(scene, ranks2, rotation, reflected)
        infos = [_decode(p['input']) for p in (p1, p2)]
        walks = [_walk(a) for a in infos]
        if (any(len(w['used']) >= 2 for w in walks)
                and any(w['final_direction'] != a['facing'] for a, w in zip(infos, walks))
                and any(not np.array_equal(_output(a, _walk(a, ignore_color=True)), _output(a, w))
                        for a, w in zip(infos, walks))
                and any(_consumption_witness(a, w) for a, w in zip(infos, walks))):
            break
    else:
        raise AssertionError('failed to construct teaching contrast')
    puzzle = {'train': [p1, p2], 'test': []}
    if bool(rng.integers(2)):
        puzzle['train'].reverse()
    seen = {_signature(p['input']) for p in puzzle['train']}
    mixed_orientation = bool(rng.integers(2))
    train_rotations = {rotation}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for _ in range(count):
            for attempt in range(80):
                pal = palette if shared_palette else [int(c) for c in rng.permutation(10)]
                if split == 'train':
                    chosen_mask = SMALL_MASKS[int(rng.integers(len(SMALL_MASKS)))]
                elif regime == 'mask':
                    chosen_mask = NEW_MASKS[int(rng.integers(len(NEW_MASKS)))]
                else:
                    chosen_mask = mask
                scene = _scene(rng, chosen_mask, pal,
                               contrast=split == 'test' and regime == 'rank',
                               cyclic=(split == 'test' if regime == 'topology' else bool(rng.integers(2))),
                               rank=3 if split == 'test' and regime == 'rank' else int(rng.integers(1, 3)),
                               all_one=split == 'test' and regime == 'topology')
                rot = rotation
                if mixed_orientation and split == 'train':
                    rot = int(rng.integers(4))
                elif mixed_orientation:
                    rot = int(rng.choice(sorted(train_rotations)))
                pair = _pair(scene, scene['ranks'], rot, bool(rng.integers(2)))
                info = _decode(pair['input'])
                walk = _walk(info)
                if not walk['used'] or _signature(pair['input']) in seen:
                    continue
                if split == 'test' and regime == 'topology' and len(set(walk['stations'])) == len(walk['stations']):
                    continue
                break
            else:
                raise AssertionError('pair sampling budget exhausted')
            if split == 'train':
                train_rotations.add(rot)
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, 'canvas must be preserved')
    info = _decode(inp)
    walk = _walk(info)
    _require(np.array_equal(out, _output(info, walk)), 'wrong consuming walk output')
    _require(walk['used'], 'generated example must traverse a bridge')
    _require(len(walk['used']) == len(set(walk['used'])), 'bridge reused')
    _require(int(np.count_nonzero(out == info['ink'])) == len(info['mask']), 'mover mask not preserved')
    r0, c0, r1, c1 = info['box']
    _require(np.array_equal(inp[r0:r1 + 1, c0:c1 + 1], out[r0:r1 + 1, c0:c1 + 1]), 'key changed')
    for i, e in enumerate(info['bridges']):
        _require(int(out[e['cells'][0]]) == e['cap'] == int(out[e['cells'][-1]]), 'caps changed')
        target = e['cap'] if i in walk['used'] else e['payload']
        _require(all(int(out[p]) == target for p in e['cells'][1:-1]), 'wrong bridge fill')
    for p, color in info['nodes'].items():
        target = info['ink'] if p == walk['final'] else info['bg'] if p in walk['stations'] else color
        _require(int(out[p]) == target, 'wrong station consumption')
    for p, d in zip(walk['stations'], walk['directions']):
        _require(_inside(_pose(info['mask'], p, d), inp.shape), 'intermediate pose out of bounds')
    terminal = walk['decisions'][-1]
    _require(len(terminal[3]) < terminal[2], 'stopped despite an available rank')
    return info, walk


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'bad example counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info, walk = _analyze(pair)
            signature = _signature(info['grid'])
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            if split == 'train':
                training.append((info, walk))
    demonstrated = {n for info, w in training for p, c, n, choices in w['decisions'][:-1]}
    _require({1, 2} <= demonstrated, 'training must show ranks one and two')
    _require(any(1 < n < len(choices) for info, w in training
                 for p, c, n, choices in w['decisions'][:-1]), 'need a non-extreme choice')
    _require(any(len(w['used']) >= 2 for info, w in training), 'need a multi-bridge journey')
    _require(any(not np.array_equal(_output(info, w, rotate=False), _output(info, w))
                 for info, w in training), 'training must expose rigid rotation')
    _require(any(not np.array_equal(_output(info, _walk(info, {c: 1 for c in info['ranks']})),
                                   _output(info, w)) for info, w in training), 'ignoring unary key survives')
    _require(any(not np.array_equal(_output(info, _walk(info, ignore_color=True)), _output(info, w))
                 for info, w in training),
             'ignoring cap/station correspondence survives')
    _require(any(_consumption_witness(info, w) for info, w in training),
             'training must distinguish used edges from original candidates')


def measure_pair(pair):
    info, w = _analyze(pair)
    degrees = [len(v) for v in info['adjacency'].values()]
    local_lengths = [tuple(info['bridges'][i]['length'] for i in choices)
                     for p, c, n, choices in w['decisions'][:-1]]
    points = list(info['nodes'])
    cartesian = len({r for r, c in points}) * len({c for r, c in points}) == len(points)
    return dict(shape=list(info['grid'].shape), stations=len(points), bridges=len(info['bridges']),
                cycles=len(info['bridges']) - len(points) + 1, degrees=degrees,
                families=len(info['ranks']), ranks=sorted(info['ranks'].values()),
                area=len(info['mask']), mask=info['mask'], background=info['bg'],
                payloads=len({e['payload'] for e in info['bridges']}), cartesian=cartesian,
                used=len(w['used']), revisits=len(w['stations']) - len(set(w['stations'])),
                directions=w['directions'], local_lengths=local_lengths,
                shortest=sum(n == len(v) for p, c, n, v in w['decisions'][:-1]),
                non_extreme=sum(1 < n < len(v) for p, c, n, v in w['decisions'][:-1]))


def self_test():
    """Bounded semantic, contrast, mutation, and 42-seed variation checks."""
    puzzles, rows, signatures = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        signatures.add(tuple(_signature(pair['input']) for pair in p['train'] + p['test']))
        rows.extend(measure_pair(pair) for pair in p['train'] + p['test'])
    _require(len(signatures) == 42, 'seed diversity')
    _require(len({tuple(m['shape']) for m in rows}) >= 30, 'size diversity')
    _require(len({m['stations'] for m in rows}) >= 4, 'station count diversity')
    _require({1, 2, 3} <= {n for m in rows for n in m['ranks']}, 'ordinal coverage')
    _require({1, 2, 3} <= {m['families'] for m in rows}, 'family coverage')
    _require(len({m['mask'] for m in rows}) >= 6, 'mask geometry diversity')
    _require({0, 1, 2, 3} == {d for m in rows for d in m['directions']}, 'direction coverage')
    _require(any(m['cycles'] == 0 for m in rows) and any(m['cycles'] >= 2 for m in rows), 'topology coverage')
    _require(sum(m['revisits'] > 0 for m in rows) >= 10, 'consumption with station revisits')
    _require(sum(not m['cartesian'] for m in rows) >= 15, 'irregular embeddings')
    _require(sum(m['non_extreme'] for m in rows) >= 42 and sum(m['shortest'] for m in rows) >= 20,
             'middle and last ordinal witnesses')
    _require({1, 2} <= {m['payloads'] for m in rows}, 'payload sharing variation')
    _require(len({m['background'] for m in rows}) >= 8, 'palette role variation')
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        original = pair['input'].copy()
        solve(pair['input'])
        _require(np.array_equal(original, pair['input']), 'solve mutated input')
        for rotation in range(4):
            for reflected in (False, True):
                mapping = np.array([7, 4, 0, 9, 1, 6, 3, 8, 2, 5], dtype=np.int8)
                a = mapping[_transform(pair['input'], rotation, reflected)]
                b = mapping[_transform(pair['output'], rotation, reflected)]
                _require(np.array_equal(solve(a), b), 'geometry or color equivariance')
    p, q = generate_puzzle(449008680, 10, 4), generate_puzzle(449008680, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), 'determinism')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9137 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    info = _decode(bad['test'][0]['input'])
    walk = _walk(info)
    e = info['bridges'][walk['used'][0]]
    bad['test'][0]['output'][e['cells'][1]] = e['payload']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong trail accepted')
    corrupt = puzzles[0]['test'][0]['input'].copy()
    corrupt[e['cells'][0]] = info['bg']
    try:
        solve(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged cap accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, count boundaries, semantic invariants)')
