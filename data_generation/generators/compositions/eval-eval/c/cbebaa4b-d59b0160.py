"""Geometrically assemble a cargo tree, cut color-complete joints, keep the root.

Only standard Python and NumPy are used. The solver reads one input and has no
access to generation parameters, demonstrations, seeds, or stored puzzle data.
"""
from collections import Counter, deque
from itertools import combinations

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _rect(box):
    t, l, b, r = box
    return {(a, c) for a in range(t, b + 1) for c in range(l, r + 1)}


def _border(box):
    t, l, b, r = box
    return {q for q in _rect(box) if q[0] in (t, b) or q[1] in (l, r)}


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _outside(box, side):
    t, l, b, r = box
    if side == 0:
        return {(t - 1, c) for c in range(l, r + 1)}
    if side == 1:
        return {(a, r + 1) for a in range(t, b + 1)}
    if side == 2:
        return {(b + 1, c) for c in range(l, r + 1)}
    return {(a, l - 1) for a in range(t, b + 1)}


def _ports(box, pins):
    groups = []
    covered = set()
    for side in range(4):
        group = pins & _outside(box, side)
        if group:
            assert len(group) in (2, 3), "A side must contain one complete two/three-pin port"
            groups.append((side, group))
            covered |= group
    assert covered == pins and groups, "Pins must sit immediately outside a body side"
    return groups


def _analyze(value):
    """Recover roles, separate tiles, and complete ports entirely from pixels."""
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > a.size, "Background must be a strict majority"
    roots = []
    for filler in range(10):
        if filler == bg:
            continue
        for comp in _components(map(tuple, np.argwhere(a == filler).tolist())):
            box = _box(comp)
            t, l, b, r = box
            if not (4 <= b - t + 1 <= 7 and 4 <= r - l + 1 <= 7):
                continue
            border = _border(box)
            if comp != border:
                continue
            interior = _rect(box) - border
            dots = {q for q in interior if a[q] != bg}
            query = {int(a[q]) for q in dots}
            if len(dots) != 3 or len(query) != 3 or filler in query or len(interior) - 3 < 3:
                continue
            outside = set().union(*(_outside(box, s) for s in range(4)))
            pins = {q for q in outside if 0 <= q[0] < a.shape[0]
                    and 0 <= q[1] < a.shape[1] and a[q] != bg}
            colors = {int(a[q]) for q in pins}
            if len(colors) != 1 or colors & (query | {bg, filler}):
                continue
            roots.append((box, filler, colors.pop(), query, border | dots, pins))
    assert len(roots) == 1, "Exactly one hollow selector must be recognizable"
    box, filler, marker, query, body, pins = roots[0]
    pieces = [dict(box=box, body={q: int(a[q]) for q in body}, pins=pins,
                   labels=set(), marks=0, ports=_ports(box, pins))]
    left = set(map(tuple, np.argwhere(a != bg).tolist())) - _rect(box) - pins
    for comp in _components(left):
        cargo = {q for q in comp if a[q] != marker}
        assert cargo, "No detached connector objects"
        cbox = _box(cargo)
        t, l, b, r = cbox
        assert 3 <= b - t + 1 <= 7 and 3 <= r - l + 1 <= 7
        assert cargo == _rect(cbox), "Cargo must be a complete solid rectangle"
        assert all(a[q] == filler for q in _border(cbox)), "Cargo perimeter is intact filler"
        cpins = comp - cargo
        labels = {int(a[q]) for q in cargo if a[q] != filler}
        assert not labels & {bg, filler, marker}
        pieces.append(dict(box=cbox, body={q: int(a[q]) for q in cargo}, pins=cpins,
                           labels=labels, marks=sum(a[q] != filler for q in cargo),
                           ports=_ports(cbox, cpins)))
    assert 3 <= len(pieces) <= 8
    # Input components are separate even diagonally. The selector's dots are
    # explicitly enclosed and belong to its body despite possible disconnection.
    owned = []
    for p in pieces:
        cells = set(p['body']) | p['pins']
        for old in owned:
            assert not cells & {(r + dr, c + dc) for r, c in old
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        owned.append(cells)
    return dict(grid=a, bg=bg, filler=filler, marker=marker, query=query, pieces=pieces)


def _reconstruct(info):
    """Unique keys determine the full tree before any color cuts are applied."""
    pieces = info['pieces']
    occurrences = {}
    for i, p in enumerate(pieces):
        for side, pins in p['ports']:
            occurrences.setdefault(_normal(pins), []).append((i, side, pins))
    adjacency = [[] for _ in pieces]
    edges = []
    assert len(occurrences) == len(pieces) - 1
    for key, endpoints in sorted(occurrences.items()):
        assert len(endpoints) == 2, "Every geometric key has exactly two owners"
        (i, si, pi), (j, sj, pj) = endpoints
        assert i != j and (si + 2) % 4 == sj, "Docking bodies face each other"
        bi, bj = _box(pi), _box(pj)
        delta = (bi[0] - bj[0], bi[1] - bj[1])
        assert _shift(pj, delta) == pi
        adjacency[i].append((j, delta))
        adjacency[j].append((i, (-delta[0], -delta[1])))
        edges.append(dict(i=i, j=j, pi=pi, pj=pj, key=key))
    positions = {0: (0, 0)}
    parent = {0: None}
    depth = {0: 0}
    queue = deque([0])
    while queue:
        i = queue.popleft()
        for j, d in adjacency[i]:
            proposed = (positions[i][0] + d[0], positions[i][1] + d[1])
            if j in positions:
                assert positions[j] == proposed
            else:
                positions[j], parent[j], depth[j] = proposed, i, depth[i] + 1
                queue.append(j)
    assert len(positions) == len(pieces), "Keys must form one connected tree"
    # Check every full-tree body, even cargo that will later disappear.
    bodies = set()
    all_pins = Counter()
    h, w = info['grid'].shape
    for i, p in enumerate(pieces):
        body = _shift(p['body'], positions[i])
        pins = _shift(p['pins'], positions[i])
        assert all(0 <= r < h and 0 <= c < w for r, c in body | pins)
        assert not bodies & body
        bodies |= body
        all_pins.update(pins)
    assert not bodies & set(all_pins) and set(all_pins.values()) == {2}
    for edge in edges:
        i, j = edge['i'], edge['j']
        edge['cells'] = _shift(edge['pi'], positions[i])
        assert edge['cells'] == _shift(edge['pj'], positions[j])
        edge['pooled'] = pieces[i]['labels'] | pieces[j]['labels']
        edge['cut'] = info['query'] <= edge['pooled']
    safe = [[] for _ in pieces]
    for edge in edges:
        if not edge['cut']:
            i, j = edge['i'], edge['j']
            safe[i].append(j)
            safe[j].append(i)
    kept, queue = {0}, [0]
    for i in queue:
        for j in safe[i]:
            if j not in kept:
                kept.add(j)
                queue.append(j)
    info.update(edges=edges, adjacency=adjacency, positions=positions, parent=parent,
                depth=depth, kept=kept)
    return info


def _render(info, full=False, local_only=False):
    pieces = info['pieces']
    if full:
        kept = set(range(len(pieces)))
    elif local_only:
        # Deliberately wrong ablation: cut only when an endpoint is locally full.
        kept, queue = {0}, [0]
        neighbors = [[] for _ in pieces]
        for e in info['edges']:
            if not any(info['query'] <= pieces[i]['labels'] for i in (e['i'], e['j'])):
                neighbors[e['i']].append(e['j'])
                neighbors[e['j']].append(e['i'])
        for i in queue:
            for j in neighbors[i]:
                if j not in kept:
                    kept.add(j)
                    queue.append(j)
    else:
        kept = info['kept']
    out = np.full(info['grid'].shape, info['bg'], dtype=np.int8)
    for i in sorted(kept):
        dr, dc = info['positions'][i]
        for (r, c), color in pieces[i]['body'].items():
            out[r + dr, c + dc] = color
    for e in info['edges']:
        if e['i'] in kept and e['j'] in kept:
            for q in e['cells']:
                out[q] = info['marker']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_reconstruct(_analyze(input_grid)))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _dimensions(rng, marks, root=False):
    choices = [(h, w) for h in range(4 if root else 3, 8)
               for w in range(4 if root else 3, 8)
               if (h - 2) * (w - 2) >= (6 if root else marks)]
    return _choose(rng, choices)


def _tree(rng, n):
    edges, degrees = [], [0] * n
    chain = rng.random() < .3
    for j in range(1, n):
        available = [i for i in range(j) if degrees[i] < 4]
        i = j - 1 if chain else _choose(rng, available)
        edges.append((i, j))
        degrees[i] += 1
        degrees[j] += 1
    return edges


def _inventories(rng, query, extras, n, package):
    a, b, c = query
    x, y = extras[:2]
    if package == 'pooling':
        return ([[], [a, b, x, y, a, x], [c, x], [],
                 [a, c, x, y, c, y], [b, c, x, y, b, y]],
                [(0, 1), (1, 2), (2, 3), (0, 4), (0, 5)])
    if package == 'local':
        # A retained a--b--c path pools all three along its ancestry, but no
        # adjacent pair is complete. This distinguishes edge-local union.
        return ([[], [], [a, b, c], [a, x, y, a, x],
                 [b, x, y, b, y], [c, x, y, c, x]],
                [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)])
    edges = _tree(rng, n)
    mode = str(rng.choice(['mixed', 'safe', 'isolated'], p=[.72, .14, .14]))
    missing = _choose(rng, query)
    marks = [[]]
    root_children = {j for i, j in edges if i == 0}
    for j in range(1, n):
        if mode == 'isolated' and j in root_children:
            identities = list(query)
        else:
            eligible = [q for q in query if mode != 'safe' or q != missing]
            k = int(rng.integers(len(eligible) + 1))
            identities = [int(v) for v in rng.choice(eligible, k, replace=False)]
        k = int(rng.integers(min(3, len(extras)) + 1))
        identities += [int(v) for v in rng.choice(extras, k, replace=False)]
        cargo = list(identities)
        if cargo:
            for _ in range(int(rng.integers(7))):
                cargo.append(_choose(rng, identities))
        marks.append(cargo)
    return marks, edges


def _layout(rng, marks, edges, mixed, teaching):
    """Fresh rectangle-growth construction, with no calls to either parent."""
    for restart in range(80):
        sizes = [_dimensions(rng, len(m), i == 0) for i, m in enumerate(marks)]
        h, w = sizes[0]
        boxes = [(0, 0, h - 1, w - 1)]
        ports = [set() for _ in marks]
        used_sides = [set() for _ in marks]
        occupied = _rect(boxes[0])
        keys, joints = set(), []
        for index, (i, j) in enumerate(edges):
            assert j == len(boxes)
            h, w = sizes[j]
            t, l, b, r = boxes[i]
            added = False
            for attempt in range(180):
                sides = [s for s in range(4) if s not in used_sides[i]]
                if not sides:
                    break
                side = _choose(rng, sides)
                if teaching and index < 3:
                    pin_count = 2
                elif teaching and mixed and index == 3:
                    pin_count = 3  # Demonstrate the mixed convention before tests.
                else:
                    pin_count = 3 if mixed and rng.random() < .3 else 2
                if side in (0, 2):
                    left = int(rng.integers(l - w + 1, r + 1))
                    top = t - h - 1 if side == 0 else b + 2
                    tangent = list(range(max(l, left), min(r, left + w - 1) + 1))
                    normal = t - 1 if side == 0 else b + 1
                    horizontal = True
                else:
                    top = int(rng.integers(t - h + 1, b + 1))
                    left = r + 2 if side == 1 else l - w - 1
                    tangent = list(range(max(t, top), min(b, top + h - 1) + 1))
                    normal = r + 1 if side == 1 else l - 1
                    horizontal = False
                if len(tangent) < pin_count:
                    continue
                selected = [int(v) for v in rng.choice(tangent, pin_count, replace=False)]
                pins = {(normal, v) if horizontal else (v, normal) for v in selected}
                key = _normal(pins)
                child_box = (top, left, top + h - 1, left + w - 1)
                child = _rect(child_box)
                if key in keys or (child | pins) & occupied:
                    continue
                # Unrelated bodies do not touch. The one-cell gap between the
                # intended endpoints contains their shared pins.
                near = {(a + dr, c + dc) for a, c in child for dr, dc in N4}
                existing_bodies = set().union(*(_rect(z) for z in boxes))
                if near & existing_bodies:
                    continue
                other_bodies = existing_bodies - _rect(boxes[i])
                if any((a + dr, c + dc) in other_bodies for a, c in pins for dr, dc in N4):
                    continue
                extent = _box(occupied | child | pins)
                if extent[2] - extent[0] + 1 > 28 or extent[3] - extent[1] + 1 > 28:
                    continue
                boxes.append(child_box)
                ports[i] |= pins
                ports[j] |= pins
                used_sides[i].add(side)
                used_sides[j].add((side + 2) % 4)
                occupied |= child | pins
                joints.append((i, j, pins))
                keys.add(key)
                added = True
                break
            if not added:
                break
        if len(boxes) == len(marks):
            return boxes, ports, joints
    return None


def _paint_bodies(rng, boxes, marks, query, filler):
    bodies = []
    for i, box in enumerate(boxes):
        interior = sorted(_rect(box) - _border(box))
        if i == 0:
            body = {q: filler for q in _border(box)}
            labels = query
        else:
            body = {q: filler for q in _rect(box)}
            labels = marks[i]
        locations = rng.choice(len(interior), len(labels), replace=False)
        for k, color in zip(locations, rng.permutation(labels)):
            body[interior[int(k)]] = int(color)
        bodies.append(body)
    return bodies


def _scatter(rng, shape, bodies, ports, bg, marker):
    h, w = shape
    cells = [set(body) | pins for body, pins in zip(bodies, ports)]
    boxes = [_box(c) for c in cells]

    def reserve(occupied, box):
        t, l, b, r = box
        occupied[max(0, t - 1):min(h, b + 2), max(0, l - 1):min(w, r + 2)] = True

    for restart in range(30):
        occupied = np.zeros(shape, dtype=bool)
        reserve(occupied, boxes[0])
        deltas = {0: (0, 0)}
        order = [int(i) for i in rng.permutation(np.arange(1, len(bodies)))]
        order.sort(key=lambda i: -(boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1))
        for i in order:
            t, l, b, r = boxes[i]
            bh, bw = b - t + 1, r - l + 1
            candidates = rng.permutation((h - bh + 1) * (w - bw + 1))
            for candidate in candidates:
                row, col = divmod(int(candidate), w - bw + 1)
                d = row - t, col - l
                if d == (0, 0) or np.any(occupied[row:row + bh, col:col + bw]):
                    continue
                deltas[i] = d
                reserve(occupied, (row, col, row + bh - 1, col + bw - 1))
                break
            else:
                break
        if len(deltas) == len(bodies):
            inp = np.full(shape, bg, dtype=np.int8)
            for i, (dr, dc) in deltas.items():
                for (a, c), color in bodies[i].items():
                    inp[a + dr, c + dc] = color
                for a, c in ports[i]:
                    inp[a + dr, c + dc] = marker
            return inp
    return None


def _make_pair(rng, roles, query, palette, package, n, mixed):
    bg, filler, marker = roles
    extras = [x for x in palette if x not in query]
    marks, tree = _inventories(rng, query, extras, n, package)
    for retry in range(48):
        layout = _layout(rng, marks, tree, mixed, package != 'free')
        if layout is None:
            continue
        boxes, ports, joints = layout
        bodies = _paint_bodies(rng, boxes, marks, query, filler)
        extent = _box(set().union(*(set(b) | p for b, p in zip(bodies, ports))))
        t, l, b, r = extent
        bh, bw = b - t + 1, r - l + 1
        foreground = sum(len(body) + len(pins) for body, pins in zip(bodies, ports))
        h = int(rng.integers(max(16, bh), 31))
        w = int(rng.integers(max(16, bw), 31))
        while h * w <= foreground * 2:
            if h <= w and h < 30:
                h += 1
            elif w < 30:
                w += 1
            elif h < 30:
                h += 1
            else:
                raise RuntimeError("Too much foreground for a legal majority background")
        offset = (int(rng.integers(h - bh + 1)) - t, int(rng.integers(w - bw + 1)) - l)
        bodies = [{(a + offset[0], c + offset[1]): color for (a, c), color in body.items()}
                  for body in bodies]
        ports = [_shift(p, offset) for p in ports]
        joints = [(i, j, _shift(p, offset)) for i, j, p in joints]
        inp = _scatter(rng, (h, w), bodies, ports, bg, marker)
        if inp is None:
            continue
        # Truth uses the constructed graph and actual mark identities. It is
        # not labeled by calling solve; validation will recover the graph anew.
        safe = [[] for _ in marks]
        for i, j, p in joints:
            if not set(query) <= set(marks[i]) | set(marks[j]):
                safe[i].append(j)
                safe[j].append(i)
        kept, queue = {0}, [0]
        for i in queue:
            for j in safe[i]:
                if j not in kept:
                    kept.add(j)
                    queue.append(j)
        out = np.full(inp.shape, bg, dtype=np.int8)
        for i in kept:
            for q, color in bodies[i].items():
                out[q] = color
        for i, j, pins in joints:
            if i in kept and j in kept:
                for q in pins:
                    out[q] = marker
        k = int(rng.integers(4))
        inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
        if rng.random() < .5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        return {'input': inp, 'output': out}
    raise RuntimeError("Bounded layout/scatter search exhausted")


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    colors = [int(c) for c in rng.permutation(10)]
    roles = colors[:3]
    palette = colors[3:3 + int(rng.integers(5, 8))]
    mixed = bool(rng.integers(2))
    first_query = [int(c) for c in rng.choice(palette, 3, replace=False)]
    second_query = list(first_query)
    second_query[int(rng.integers(3))] = _choose(rng, [c for c in palette if c not in first_query])
    puzzle, seen, selected = {'train': [], 'test': []}, set(), []
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            if split == 'train':
                package = 'pooling' if index == 0 else 'local' if index == 1 else 'free'
                n = 6 if index < 2 else int(rng.integers(3, 7))
                query = first_query if index == 0 else second_query if index == 1 else [int(c) for c in rng.choice(palette, 3, replace=False)]
            else:
                package, n = 'free', int(rng.integers(7, 9))
                familiar = sorted(set().union(*selected))
                fresh = [list(q) for q in combinations(familiar, 3) if set(q) not in selected]
                query = _choose(rng, fresh) if fresh else [int(c) for c in rng.choice(familiar, 3, replace=False)]
            for attempt in range(8):
                pair = _make_pair(rng, roles, query, palette, package, n, mixed)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct input sampling exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                selected.append(set(query))
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching_checks(infos):
    assert any(all(any(i in info['kept'] and j in info['kept']
                      and (info['pieces'][i]['labels'] | info['pieces'][j]['labels']) & info['query'] == info['query'] - {q}
                      for i, j in ((e['i'], e['j']) for e in info['edges']))
                   for q in info['query']) for info in infos), "Teach each missing identity"
    assert any(any(i != 0 and not p['labels'] and i in info['kept']
                   for i, p in enumerate(info['pieces'])) for info in infos), "Teach empty survival"
    pooled_witness = local_witness = innocent_witness = False
    for info in infos:
        pieces, query = info['pieces'], info['query']
        for e in info['edges']:
            i, j = e['i'], e['j']
            child = j if info['parent'].get(j) == i else i
            par = info['parent'][child]
            if e['cut'] and par in info['kept']:
                if not any(query <= pieces[v]['labels'] for v in (i, j)):
                    pooled_witness = True
                if not pieces[par]['labels'] and pieces[child]['labels'] == query and pieces[child]['marks'] == 3:
                    local_witness = True
            if not e['cut'] and i not in info['kept'] and j not in info['kept']:
                if not pieces[child]['labels']:
                    innocent_witness = True
    assert pooled_witness and local_witness and innocent_witness
    # Unlike path accumulation, only the two immediate endpoints form a record.
    assert any(any(i in info['kept'] and not info['query'] <=
                   (info['pieces'][i]['labels'] | info['pieces'][info['parent'][i]]['labels'])
                   and info['query'] <= _ancestor_inventory(info, i)
                   for i in range(1, len(info['pieces']))) for info in infos)
    assert any(len({e['key'] for e in info['edges'] if len(e['key']) == 2}) >= 3
               and not np.array_equal(_render(info), _render(info, local_only=True))
               for info in infos), "Teach pooling with equal-cardinality distinct keys"
    assert any(any(i in info['kept'] and len(p['labels']) >= 4 and p['marks'] > 3
                   and not info['query'] <= p['labels'] for i, p in enumerate(info['pieces']))
               for info in infos), "Teach more-colorful incomplete survival"


def _ancestor_inventory(info, i):
    labels = set()
    while i is not None:
        labels |= info['pieces'][i]['labels']
        i = info['parent'][i]
    return labels


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, infos = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _reconstruct(_analyze(inp))
            assert np.array_equal(_render(info), out), "Wrong complete-body/joint deletion"
            roles.add((info['bg'], info['filler'], info['marker']))
            infos[split].append(info)
            # Root body is preserved at fixed coordinates, including its query.
            assert info['positions'][0] == (0, 0)
            assert all(out[q] == color for q, color in info['pieces'][0]['body'].items())
            assert all(info['positions'][i] != (0, 0) for i in range(1, len(info['pieces'])))
            expected_pins = set().union(*(e['cells'] for e in info['edges']
                                         if e['i'] in info['kept'] and e['j'] in info['kept']))
            assert set(map(tuple, np.argwhere(out == info['marker']).tolist())) == expected_pins
    assert len(roles) == 1
    _teaching_checks(infos['train'])
    assert any(len(i['kept']) > 1 and len(i['kept']) < len(i['pieces']) for i in infos['train'])
    familiar = set().union(*(i['query'] for i in infos['train']))
    assert all(i['query'] <= familiar for i in infos['test'])
    assert all(len(i['pieces']) > max(len(t['pieces']) for t in infos['train']) for i in infos['test'])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test() -> dict:
    """Bounded semantic checks and measured diversity over 42 reproducible seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {_puzzle_key(p) for p in puzzles}
    assert len(signatures) == 42
    measurements = {k: set() for k in ('canvas', 'roles', 'bodies', 'keys', 'nodes',
                                      'degrees', 'depths', 'survivors', 'queries', 'layouts')}
    pooled = cascades = branches = all_safe = isolated = three_pin = 0
    for seed, puzzle in enumerate(puzzles):
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _reconstruct(_analyze(pair['input']))
                measurements['canvas'].add(info['grid'].shape)
                measurements['roles'].add((info['bg'], info['filler'], info['marker']))
                measurements['nodes'].add(len(info['pieces']))
                measurements['degrees'].update(len(x) for x in info['adjacency'])
                measurements['depths'].add(max(info['depth'].values()))
                measurements['survivors'].add(len(info['kept']) - 1)
                measurements['queries'].add(tuple(sorted(info['query'])))
                measurements['layouts'].add(tuple(p['box'] for p in info['pieces']))
                for p in info['pieces'][1:]:
                    t, l, b, r = p['box']
                    measurements['bodies'].add((b - t + 1, r - l + 1))
                measurements['keys'].update(e['key'] for e in info['edges'])
                branches += max(len(x) for x in info['adjacency']) >= 3
                all_safe += len(info['kept']) == len(info['pieces'])
                isolated += len(info['kept']) == 1
                three_pin += any(len(e['key']) == 3 for e in info['edges'])
                for e in info['edges']:
                    pooled += e['cut'] and not any(info['query'] <= info['pieces'][i]['labels'] for i in (e['i'], e['j']))
                    cascades += not e['cut'] and e['i'] not in info['kept'] and e['j'] not in info['kept']
        # Query-dot geometry is irrelevant; its content is preserved verbatim.
        pair = puzzle['train'][0]
        info = _reconstruct(_analyze(pair['input']))
        changed = pair['input'].copy()
        box = info['pieces'][0]['box']
        inside = sorted(_rect(box) - _border(box))
        values = [int(changed[q]) for q in inside]
        for q, color in zip(inside, values[1:] + values[:1]):
            changed[q] = color
        expected = pair['output'].copy()
        for q in inside:
            expected[q] = changed[q]
        assert np.array_equal(solve(changed), expected)
        # Changing only one selector identity can change cuts and connectivity.
        training_infos = [_reconstruct(_analyze(p['input'])) for p in puzzle['train']]
        witness = next(info for info in training_infos
                       if any(e['cut'] and (e['i'] in info['kept'] or e['j'] in info['kept'])
                              and not any(info['query'] <= info['pieces'][v]['labels'] for v in (e['i'], e['j']))
                              for e in info['edges']))
        dots = sorted(q for q, color in witness['pieces'][0]['body'].items() if color != witness['filler'])
        available = [c for c in range(10) if c not in witness['query'] | {witness['bg'], witness['filler'], witness['marker']}]
        outside_selector = np.ones(witness['grid'].shape, dtype=bool)
        for q in _rect(witness['pieces'][0]['box']):
            outside_selector[q] = False
        original = _render(witness)
        found = False
        for q in dots:
            for color in available:
                changed = witness['grid'].copy()
                changed[q] = color
                if not np.array_equal(solve(changed)[outside_selector], original[outside_selector]):
                    found = True
                    break
            if found:
                break
        assert found
        assert not np.array_equal(_render(witness), _render(witness, full=True))
        assert not np.array_equal(_render(witness), _render(witness, local_only=True))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(901 + a, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Cargo mark locations have no influence on cuts, but move rigidly with
    # their body. Rotate a retained body's interior and check both facts.
    fixture = next(p for p in puzzles[0]['train']
                   if any(i in info['kept'] and bool(tile['labels'])
                          for info in [_reconstruct(_analyze(p['input']))]
                          for i, tile in enumerate(info['pieces']) if i != 0))
    info = _reconstruct(_analyze(fixture['input']))
    index = next(i for i in info['kept'] if i != 0 and info['pieces'][i]['labels'])
    box = info['pieces'][index]['box']
    inside = sorted(_rect(box) - _border(box))
    changed, expected = fixture['input'].copy(), fixture['output'].copy()
    values = [int(changed[q]) for q in inside]
    dr, dc = info['positions'][index]
    for (r, c), color in zip(inside, values[1:] + values[:1]):
        changed[r, c] = color
        expected[r + dr, c + dc] = color
    assert _reconstruct(_analyze(changed))['kept'] == info['kept']
    assert np.array_equal(solve(changed), expected)
    # Reject damage to the anchor, an active seam, and a partial resurrection
    # of a discarded body. Each targets a different part of the intended rule.
    fixture_index = next(i for i, p in enumerate(puzzles[0]['train'])
                         if 1 < len(_reconstruct(_analyze(p['input']))['kept'])
                         < len(_reconstruct(_analyze(p['input']))['pieces']))
    for corruption in ('anchor', 'seam', 'discarded_body'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
        pair = bad['train'][fixture_index]
        info = _reconstruct(_analyze(pair['input']))
        if corruption == 'anchor':
            q = min(info['pieces'][0]['body'])
            pair['output'][q] = info['bg']
        elif corruption == 'seam':
            edge = next(e for e in info['edges'] if e['i'] in info['kept'] and e['j'] in info['kept'])
            pair['output'][min(edge['cells'])] = info['bg']
        else:
            i = next(i for i in range(len(info['pieces'])) if i not in info['kept'])
            q, color = min(info['pieces'][i]['body'].items())
            dr, dc = info['positions'][i]
            pair['output'][q[0] + dr, q[1] + dc] = color
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Corrupted {corruption} accepted")
    assert measurements['nodes'] == set(range(3, 9))
    assert measurements['degrees'] >= {1, 2, 3, 4}
    assert len(measurements['canvas']) >= 40 and len(measurements['layouts']) >= 150
    assert len(measurements['bodies']) >= 15 and len(measurements['keys']) >= 15
    assert len(measurements['roles']) >= 30 and len(measurements['queries']) >= 40
    assert max(measurements['depths']) >= 4 and len(measurements['survivors']) >= 5
    assert pooled >= 42 and cascades >= 42 and branches >= 42
    assert all_safe and isolated and three_pin
    return {'seeds': 42, 'pairs': 210,
            'distinct_values': {k: len(v) for k, v in measurements.items()},
            'pooled_cuts': int(pooled), 'discarded_safe_edges': int(cascades),
            'branched_scenes': int(branches), 'all_safe_scenes': int(all_safe),
            'root_isolated_scenes': int(isolated), 'three_pin_scenes': int(three_pin)}
