"""Joint-counted rotor trees: geometric docking with accumulated unary turns.

Only standard-library modules and NumPy are used. The solver receives one input;
it has no access to the seed, the sampler, demonstrations, or generation metadata.
"""

from collections import Counter, deque

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= int(a) <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= int(b) <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _grid(x):
    a = np.asarray(x)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


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


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _rot(q, k):
    # Clockwise in row/column coordinates, around the origin. Translation later
    # absorbs the choice of pivot; this is ordinary rigid quarter-turn motion.
    r, c = q
    for _ in range(k % 4):
        r, c = c, -r
    return r, c


def _norm(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r-t, c-l) for r, c in cells))


def _colored_norm(values):
    t, l, _, _ = _box(values)
    return tuple(sorted((r-t, c-l, int(v)) for (r, c), v in values.items()))


def _pose(values, marker, k):
    mapping = {_rot(q, k): v for q, v in values.items()}
    pins = {q for q, v in mapping.items() if v == marker}
    return dict(values=mapping, pins=pins, body=set(mapping)-pins,
                cells=set(mapping), box=_box(mapping))


def _parse(input_grid):
    a = _grid(input_grid)
    hues, areas = np.unique(a, return_counts=True)
    if np.count_nonzero(areas == areas.max()) != 1:
        raise ValueError("the background must be the unique modal color")
    bg = int(hues[areas.argmax()])
    objects = _components(map(tuple, np.argwhere(a != bg).tolist()))
    if not 3 <= len(objects) <= 6:
        raise ValueError("expected 3..6 separated pieces")
    palettes = [{int(a[q]) for q in obj} for obj in objects]
    common = set.intersection(*palettes)
    if len(common) != 1:
        raise ValueError("a unique common connector color is required")
    marker = common.pop()
    roots = [i for i, colors in enumerate(palettes) if len(colors) == 3]
    if len(roots) != 1 or any(len(p) != (3 if i == roots[0] else 2)
                              for i, p in enumerate(palettes)):
        raise ValueError("only the riveted root has an extra payload color")
    root = roots[0]
    outside_colors = set.union(*(p for i, p in enumerate(palettes) if i != root))
    rare = [c for c in palettes[root] - {marker}
            if np.count_nonzero(a == c) == 1 and c not in outside_colors]
    if len(rare) != 1:
        raise ValueError("the root must contain one unique singleton rivet")
    rivet = rare[0]
    pieces = []
    for obj in objects:
        t, l, _, _ = _box(obj)
        values = {(r-t, c-l): int(a[r, c]) for r, c in obj}
        poses = [_pose(values, marker, k) for k in range(4)]
        if len(_components(poses[0]['body'])) != 1:
            raise ValueError("each complete payload must be connected")
        pieces.append(dict(origin=(t, l), values=values, poses=poses))
    return dict(grid=a, background=bg, marker=marker, rivet=rivet,
                root=root, pieces=pieces)


class _SearchLimit(ValueError):
    pass


def _reconstruct(info, convention="sum", direction=1, limit=18000):
    """Enumerate complete rooted assemblies; reject distinct answer grids.

    A newly placed piece shares pins with exactly one already placed parent.
    All pins must ultimately have two owners. No connected-component grouping
    of the pins or predefined partition into ports is used by this search.
    """
    pieces, root = info['pieces'], info['root']
    h, w = info['grid'].shape
    links = {}

    def options(i, phase, j):
        cache_key = i, phase, j
        if cache_key in links:
            return links[cache_key]
        a = pieces[i]['poses'][phase]
        by_pin = {}
        for k in range(4):
            if convention == 'zero' and k != 0:
                continue
            b = pieces[j]['poses'][k]
            offsets = sorted({(r-s, c-t) for r, c in a['pins']
                              for s, t in b['pins']})
            for delta in offsets:
                bm = _shift(b['pins'], delta)
                overlap = a['pins'] & bm
                m = len(overlap)
                target = ((phase if convention == 'sum' else 0) + direction*m) % 4
                if convention != 'zero' and k != target:
                    continue
                bp = _shift(b['body'], delta)
                if bp & a['cells'] or bm & a['body']:
                    continue
                for q in sorted(overlap):
                    by_pin.setdefault(q, []).append((k, delta))
        links[cache_key] = by_pin
        return by_pin

    start = pieces[root]['origin']
    rpose = pieces[root]['poses'][0]
    bodies = _shift(rpose['body'], start)
    singles = _shift(rpose['pins'], start)
    poses = {root: (0, start)}
    answers, visited = {}, set()
    nodes = 0

    def search(body, single, double, owners):
        nonlocal nodes
        state = tuple(sorted(poses.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise _SearchLimit("complete uniqueness search exceeded its budget")
        if not single:
            if len(poses) != len(pieces):
                return
            out = np.full((h, w), info['background'], dtype=np.int8)
            for i, (phase, delta) in poses.items():
                p = pieces[i]['poses'][phase]
                for (r, c), hue in p['values'].items():
                    out[r+delta[0], c+delta[1]] = hue
            answers.setdefault(out.tobytes(), (out, dict(poses)))
            return
        if len(poses) == len(pieces):
            return
        best = None
        # Every outstanding pin must be covered by a still-unplaced piece.
        for q in sorted(single):
            parent = owners[q]
            phase, origin = poses[parent]
            local_q = q[0]-origin[0], q[1]-origin[1]
            choices = []
            for j in range(len(pieces)):
                if j in poses:
                    continue
                for k, relative in options(parent, phase, j).get(local_q, ()):
                    delta = origin[0]+relative[0], origin[1]+relative[1]
                    p = pieces[j]['poses'][k]
                    t, l, b, r = p['box']
                    if t+delta[0] < 0 or l+delta[1] < 0 or b+delta[0] >= h or r+delta[1] >= w:
                        continue
                    bp, bm = _shift(p['body'], delta), _shift(p['pins'], delta)
                    if bp & (body | single | double) or bm & (body | double):
                        continue
                    overlap = bm & single
                    if any(owners[x] != parent for x in overlap):
                        continue
                    choices.append((j, k, delta, bp, bm, overlap))
            if not choices:
                return
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
        for j, k, delta, bp, bm, overlap in best:
            poses[j] = k, delta
            new_owners = dict(owners)
            new_owners.update({q: j for q in bm-overlap})
            search(body | bp, (single | bm)-overlap, double | overlap, new_owners)
            del poses[j]
            if len(answers) > 1:
                return

    search(bodies, singles, set(), {q: root for q in singles})
    return list(answers.values()), nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique joint-counted rooted assembly, from this input alone."""
    answers, _ = _reconstruct(_parse(input_grid))
    if len(answers) != 1:
        raise ValueError("input has no unique complete rotor-tree assembly")
    return answers[0][0].copy()


def _rectangle(t, l, h, w):
    return {(r, c) for r in range(t, t+h) for c in range(l, l+w)}


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _layout(rng, parents, counts):
    """Grow bodies with uneven offsets and sparse keys, in final coordinates."""
    n = len(parents)+1
    incident = [[] for _ in range(n)]
    for j, (i, m) in enumerate(zip(parents, counts), 1):
        incident[i].append(m)
        incident[j].append(m)
    for _ in range(55):
        sizes = []
        for ms in incident:
            low = max(2, max(ms))
            sizes.append((int(rng.integers(low, min(9, low+3)+1)),
                          int(rng.integers(low, min(9, low+3)+1))))
        h, w = sizes[0]
        nodes = [dict(rect=_rectangle(0, 0, h, w), arms=set(), pins=set(),
                      required=set(), sides=set())]
        edges = []
        for j, (i, m) in enumerate(zip(parents, counts), 1):
            parent = nodes[i]
            found = False
            for _ in range(65):
                k = int(rng.integers(4))
                if k in parent['sides']:
                    continue
                rotated_rect = {_rot(q, k) for q in parent['rect']}
                t, l, b, r = _box(rotated_rect)
                h, w = sizes[j] if k % 2 == 0 else sizes[j][::-1]
                # At least m different tangent rows can carry marker cells.
                lo, hi = t-h+m, b-m+1
                if lo > hi:
                    continue
                top = int(rng.integers(lo, hi+1))
                gap = int(rng.integers(1, 4))
                left = r+gap+1
                overlap_rows = list(range(max(t, top), min(b, top+h-1)+1))
                rows = sorted(int(x) for x in rng.choice(overlap_rows, m, replace=False))
                offsets = [int(rng.integers(gap)) for _ in rows]
                pins = {(y, r+1+o) for y, o in zip(rows, offsets)}
                pa = {(y, c) for y, o in zip(rows, offsets) for c in range(r+1, r+1+o)}
                ca = {(y, c) for y, o in zip(rows, offsets) for c in range(r+2+o, left)}
                required_a = {(y, r) for y in rows}
                required_b = {(y, left) for y in rows}
                child_rect = _rectangle(top, left, h, w)
                pins, pa, ca, required_a, required_b, child_rect = [
                    {_rot(q, -k) for q in cells}
                    for cells in (pins, pa, ca, required_a, required_b, child_rect)]
                pcells = parent['rect'] | parent['arms'] | parent['pins']
                child_cells = child_rect | ca | pins
                if pins & pcells or pa & parent['pins'] or child_cells & (pcells | pa):
                    continue
                new_cells = child_cells | pa
                if any(new_cells & (z['rect'] | z['arms'] | z['pins'])
                       for v, z in enumerate(nodes) if v != i):
                    continue
                all_cells = set().union(*(z['rect'] | z['arms'] | z['pins'] for z in nodes)) | new_cells
                bt, bl, bb, br = _box(all_cells)
                if bb-bt+1 > 29 or br-bl+1 > 29:
                    continue
                parent['arms'] |= pa
                parent['pins'] |= pins
                parent['required'] |= required_a
                parent['sides'].add(k)
                nodes.append(dict(rect=child_rect, arms=ca, pins=set(pins),
                                  required=required_b, sides={(k+2) % 4}))
                edges.append((i, j, pins))
                found = True
                break
            if not found:
                break
        if len(nodes) == n:
            # Repeated cardinalities must genuinely require geometry.
            keys = [(len(p), _norm(p)) for i, j, p in edges]
            if len(counts) != len(set(counts)) and not any(
                    a == b and x != y for v, (a, x) in enumerate(keys) for b, y in keys[:v]):
                continue
            return nodes, edges
    return None


def _body(rng, node):
    t, l, b, r = _box(node['rect'])
    h, w = b-t+1, r-l+1
    style = int(rng.integers(5))
    if style == 0:
        body = set(node['rect'])
    elif style in (1, 2):
        missing_side = int(rng.integers(4)) if style == 2 else -1
        body = {q for q in node['rect']
                if any(v == 0 and side != missing_side
                       for side, v in enumerate((q[0]-t, r-q[1], b-q[0], q[1]-l)))}
    else:
        body = {_pick(rng, sorted(node['rect']))}
        target = int(rng.integers(max(2, h*w//4), max(3, 3*h*w//4+1)))
        while len(body) < target:
            frontier = sorted({(y+dy, x+dx) for y, x in body for dy, dx in N4}
                              & node['rect'] - body)
            body.add(_pick(rng, frontier))
    for target in sorted(node['required']):
        y, x = min(body, key=lambda p: (abs(p[0]-target[0])+abs(p[1]-target[1]), p))
        for axis in rng.permutation(2):
            if axis == 0:
                while y != target[0]:
                    y += 1 if target[0] > y else -1
                    body.add((y, x))
            else:
                while x != target[1]:
                    x += 1 if target[1] > x else -1
                    body.add((y, x))
    body |= node['arms']
    assert len(_components(body)) == 1
    return body


def _scatter(rng, shapes, root_origin, canvas, bg, final_values):
    h, w = canvas
    for _ in range(28):
        origins = {0: root_origin}
        used = _shift(set(shapes[0]), root_origin)
        forbidden = {(r+dy, c+dx) for r, c in used for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        order = [int(i) for i in rng.permutation(np.arange(1, len(shapes)))]
        order.sort(key=lambda i: -len(shapes[i]))
        for i in order:
            _, _, b, r = _box(shapes[i])
            found = False
            for _ in range(450):
                delta = int(rng.integers(h-b)), int(rng.integers(w-r))
                candidate = _shift(set(shapes[i]), delta)
                if candidate & forbidden:
                    continue
                mapping = {(y+delta[0], x+delta[1]): v for (y, x), v in shapes[i].items()}
                if mapping == final_values[i]:
                    continue
                origins[i] = delta
                forbidden |= {(y+dy, x+dx) for y, x in candidate for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                found = True
                break
            if not found:
                break
        if len(origins) == len(shapes):
            a = np.full(canvas, bg, dtype=np.int8)
            for i, values in enumerate(shapes):
                dy, dx = origins[i]
                for (y, x), v in values.items():
                    a[y+dy, x+dx] = v
            return a
    return None


def _make_pair(rng, parents, counts, previous_bg=None, informative=False):
    n = len(parents)+1
    for attempt in range(40):
        layout = _layout(rng, parents, counts)
        if layout is None:
            continue
        nodes, edges = layout
        bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
        roles = [int(c) for c in rng.choice([c for c in range(10) if c != bg], 3, replace=False)]
        marker, rivet, root_hue = roles
        palette = [c for c in range(10) if c not in (bg, marker, rivet, root_hue)]
        # Reuse is optional and never changes spatial piece identity.
        color_count = int(rng.integers(1, min(n-1, len(palette))+1))
        colors = [int(c) for c in rng.choice(palette, color_count, replace=False)]
        movable = colors + [int(rng.choice(colors)) for _ in range(n-1-color_count)]
        rng.shuffle(movable)
        body_colors = [root_hue] + movable
        values = []
        for i, node in enumerate(nodes):
            body = _body(rng, node)
            mapping = {q: body_colors[i] for q in body}
            if i == 0:
                interior = [q for q in sorted(body) if all((q[0]+dy, q[1]+dx) in body for dy, dx in N4)]
                cue = _pick(rng, interior or sorted(body))
                mapping[cue] = rivet
            mapping.update({q: marker for q in node['pins']})
            values.append(mapping)
        all_cells = set().union(*(set(v) for v in values))
        t, l, b, r = _box(all_cells)
        bh, bw = b-t+1, r-l+1
        # Enough independent scatter space, while allowing tight output margins.
        packing = sum((_box(v)[2]-_box(v)[0]+2)*(_box(v)[3]-_box(v)[1]+2) for v in values)
        minimum = min(30, max(20, int(np.ceil(np.sqrt(1.5*packing)))))
        h = int(rng.integers(max(bh, minimum), 31))
        w = int(rng.integers(max(bw, minimum), 31))
        offset = int(rng.integers(h-bh+1))-t, int(rng.integers(w-bw+1))-l
        placed = [{(y+offset[0], x+offset[1]): v for (y, x), v in p.items()} for p in values]
        out = np.full((h, w), bg, dtype=np.int8)
        for p in placed:
            for q, hue in p.items():
                assert out[q] in (bg, marker)
                out[q] = hue
        phases = [0]
        for i, m in zip(parents, counts):
            phases.append((phases[i]+m) % 4)
        shapes = []
        for i, p in enumerate(placed):
            raw = {_rot(q, -phases[i]): v for q, v in p.items()}
            top, left, _, _ = _box(raw)
            shapes.append({(y-top, x-left): v for (y, x), v in raw.items()})
        root_top, root_left, _, _ = _box(placed[0])
        inp = _scatter(rng, shapes, (root_top, root_left), (h, w), bg, placed)
        if inp is None:
            continue
        # Rotating the complete scene commutes with clockwise relative turns.
        k = int(rng.integers(4))
        inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
        try:
            info = _parse(inp)
            answers, _ = _reconstruct(info, limit=5500)
        except (ValueError, AssertionError):
            continue
        if len(answers) != 1 or not np.array_equal(answers[0][0], out):
            continue
        feature = _features(info, out, answers[0][1])
        if informative and (not feature['rotation_changes'] or not feature['accumulation_changes']):
            continue
        return dict(input=inp, output=out)
    raise RuntimeError(f"bounded sampler could not construct a unique rotor tree: {parents}, {counts}")


def _features(info, out, poses):
    """Reconstruct and check semantics from visible input plus recovered poses."""
    pieces, root = info['pieces'], info['root']
    bg, marker = info['background'], info['marker']
    out = _grid(out)
    assert out.shape == info['grid'].shape
    assert set(poses) == set(range(len(pieces)))
    assert poses[root] == (0, pieces[root]['origin'])
    pins, bodies = [], []
    multiplicity = Counter()
    expected = np.full(out.shape, bg, dtype=np.int8)
    body_areas, body_boxes, body_shapes = [], [], []
    rotation_changes = 0
    for i in range(len(pieces)):
        phase, delta = poses[i]
        p = pieces[i]['poses'][phase]
        # Independent array-based rotation check, separate from _rot and the
        # construction/search coordinate implementation.
        source = pieces[i]['values']
        _, _, sb, sr = _box(source)
        stamp = np.full((sb+1, sr+1), bg, dtype=np.int8)
        for q, hue in source.items():
            stamp[q] = hue
        stamp = np.rot90(stamp, -phase)
        st, sl, _, _ = p['box']
        normalized = {(r-st, c-sl): hue for (r, c), hue in p['values'].items()}
        checked = {tuple(q): int(stamp[tuple(q)]) for q in np.argwhere(stamp != bg)}
        assert normalized == checked
        bp, bm = _shift(p['body'], delta), _shift(p['pins'], delta)
        for y, x in bp | bm:
            assert 0 <= y < out.shape[0] and 0 <= x < out.shape[1]
        assert not bp & set().union(*bodies)
        for q, hue in p['values'].items():
            if hue != marker:
                dest = q[0]+delta[0], q[1]+delta[1]
                assert expected[dest] == bg
                expected[dest] = hue
        multiplicity.update(bm)
        pins.append(bm)
        bodies.append(bp)
        box = _box(bp)
        body_boxes.append((box[2]-box[0]+1, box[3]-box[1]+1))
        body_areas.append(len(bp))
        body_shapes.append(_norm(bp))
        if i != root:
            rotation_changes += _colored_norm(p['values']) != _colored_norm(pieces[i]['values'])
            actual = {(r+delta[0], c+delta[1]): hue for (r, c), hue in p['values'].items()}
            original = {(r+pieces[i]['origin'][0], c+pieces[i]['origin'][1]): hue
                        for (r, c), hue in pieces[i]['values'].items()}
            assert actual != original
    assert set(multiplicity.values()) == {2}
    assert not set(multiplicity) & set().union(*bodies)
    for q in multiplicity:
        expected[q] = marker
    assert np.array_equal(out, expected)
    assert np.count_nonzero(info['grid'] == marker) == 2*np.count_nonzero(out == marker)
    for hue in range(10):
        if hue not in (bg, marker):
            assert np.count_nonzero(info['grid'] == hue) == np.count_nonzero(out == hue)
    adjacency = [[] for _ in pieces]
    keys = []
    for i in range(len(pieces)):
        for j in range(i):
            shared = pins[i] & pins[j]
            if shared:
                adjacency[i].append((j, len(shared)))
                adjacency[j].append((i, len(shared)))
                keys.append((len(shared), _norm(shared)))
    assert len(keys) == len(pieces)-1
    depth, raw_phase, incoming = {root: 0}, {root: 0}, {}
    todo = deque([root])
    accumulation_changes = 0
    while todo:
        i = todo.popleft()
        for j, m in adjacency[i]:
            if j in depth:
                continue
            depth[j] = depth[i]+1
            raw_phase[j] = raw_phase[i]+m
            incoming[j] = m
            assert poses[j][0] == raw_phase[j] % 4
            true_pose = pieces[j]['poses'][poses[j][0]]
            absolute_pose = pieces[j]['poses'][m % 4]
            accumulation_changes += _colored_norm(true_pose['values']) != _colored_norm(absolute_pose['values'])
            todo.append(j)
    assert len(depth) == len(pieces)
    assert all(1 <= m <= 7 for m, key in keys)
    t, l, b, r = _box(set().union(*pins, *bodies))
    movable_hues = [next(v for v in pieces[i]['values'].values() if v != marker)
                     for i in range(len(pieces)) if i != root]
    return dict(piece_count=len(pieces), background=bg, connector=marker, rivet=info['rivet'],
                canvas=out.shape, extent=(b-t+1, r-l+1),
                margins=(t, l, out.shape[0]-b-1, out.shape[1]-r-1),
                joint_counts=sorted(m for m, key in keys), joint_keys=keys,
                equal_count_different_keys=any(a == b and x != y
                    for v, (a, x) in enumerate(keys) for b, y in keys[:v]),
                max_depth=max(depth.values()), degrees=[len(x) for x in adjacency],
                phases=[poses[i][0] for i in range(len(pieces)) if i != root],
                raw_phases=list(raw_phase.values()), incoming=list(incoming.values()),
                rotation_changes=int(rotation_changes), accumulation_changes=int(accumulation_changes),
                body_boxes=body_boxes, body_areas=body_areas, body_shapes=body_shapes,
                repeated_payload=len(movable_hues) > len(set(movable_hues)))


def _analyze_pair(pair):
    info = _parse(pair['input'])
    answers, nodes = _reconstruct(info)
    assert len(answers) == 1, "assembly must have one distinct output"
    out, poses = answers[0]
    assert np.array_equal(out, np.asarray(pair['output'])), "incorrect assembly"
    feature = _features(info, pair['output'], poses)
    feature['search_nodes'] = nodes
    return feature


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(2))
    first_odd = int(rng.choice([1, 3]))
    puzzle, used, previous_bg = {'train': [], 'test': []}, set(), None
    for split, count in (('train', a), ('test', b)):
        for index in range(count):
            if split == 'train' and index == 0:
                parents, counts = [0, 1, 2], [first_odd, 2, 2]
            elif split == 'train' and index == 1:
                parents, counts = [0, 1, 1, 3], [4-first_odd, 4, 5, 2]
            else:
                if split == 'test' and regime == 1:
                    n = 6
                    # Deeper than the teaching examples, with familiar counts.
                    parents = list(range(n-1))
                else:
                    n = int(rng.integers(4, 6))
                    parents = [int(rng.integers(j)) for j in range(1, n)]
                    if max(Counter(parents).values()) > 3:
                        parents = [0, 1, 1] + ([3] if n == 5 else [])
                counts = [int(rng.integers(1, 6)) for _ in parents]
                # Several indistinguishable singleton ports usually leave an
                # arbitrary choice of branch. Keep one; equal-area larger keys
                # still require comparing their actual, variable geometry.
                ones = [v for v, m in enumerate(counts) if m == 1]
                for v in ones[1:]:
                    counts[v] = int(rng.integers(2, 6))
                if split == 'test' and regime == 0:
                    counts[int(rng.integers(len(counts)))] = int(rng.integers(6, 8))
            for _ in range(12):
                pair = _make_pair(rng, parents, counts, previous_bg,
                                  informative=split == 'train' and index < 2)
                if _key(pair['input']) not in used:
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
            used.add(_key(pair['input']))
            previous_bg = int(np.bincount(pair['input'].ravel(), minlength=10).argmax())
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    used, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in used
            used.add(_key(inp))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            records[split].append(_analyze_pair(pair))
    train = records['train']
    assert {1, 2, 3, 4, 5} <= {m for f in train for m in f['joint_counts']}
    assert any(f['equal_count_different_keys'] for f in train)
    assert all(f['rotation_changes'] > 0 and f['accumulation_changes'] > 0 for f in train[:2])
    assert any(max(f['degrees']) >= 3 for f in train)
    assert any(max(f['degrees']) == 2 for f in train)
    assert any(0 in f['phases'] for f in train)
    assert any(m > 4 for f in train for m in f['raw_phases'])
    assert len({f['background'] for fs in records.values() for f in fs}) >= 2
    assert len({f['piece_count'] for f in train}) >= 2
    return None


def self_test():
    """Bounded diversity, conservation, ablation, and malformed-output checks."""
    records, signatures, puzzles = [], set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        for s in ('train', 'test'):
            for pair in p[s]:
                f = _analyze_pair(pair)
                f['split'] = s
                records.append(f)
    assert len(signatures) == 42
    assert {f['piece_count'] for f in records} >= {4, 5, 6}
    assert {f['background'] for f in records} == set(range(10))
    assert len({f['canvas'] for f in records}) >= 20
    assert len({f['extent'] for f in records}) >= 45
    assert len({b for f in records for b in f['body_boxes']}) >= 30
    assert len({b for f in records for b in f['body_shapes']}) >= 180
    assert {m for f in records for m in f['joint_counts']} == set(range(1, 8))
    assert {k for f in records for k in f['phases']} == set(range(4))
    assert any(f['repeated_payload'] for f in records)
    assert any(0 in f['margins'] for f in records)
    assert any(f['max_depth'] == 5 for f in records)
    assert any(6 in f['joint_counts'] or 7 in f['joint_counts'] for f in records if f['split'] == 'test')
    assert sum(bool(f['rotation_changes'] and f['accumulation_changes']) for f in records) >= 84
    assert _key(generate_puzzle(17)['train'][0]['input']) == _key(puzzles[17]['train'][0]['input'])
    permutation = np.random.default_rng(802).permutation(10)
    for pair in puzzles[0]['train'][:2]:
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Ignoring A (translation only), ignoring accumulation, or reversing the
    # turn direction must fail to reproduce actual teaching answers.
    for convention, direction in [('zero', 1), ('absolute', 1), ('sum', -1)]:
        differs = False
        for pair in puzzles[0]['train'][:2]:
            info = _parse(pair['input'])
            try:
                answers, _ = _reconstruct(info, convention, direction, limit=18000)
            except _SearchLimit:
                # A budget exhaustion is not evidence of a differing answer.
                continue
            differs |= not any(np.array_equal(a, pair['output']) for a, poses in answers)
        assert differs, "teaching set failed an orientation ablation"
    # Ignoring B and leaving the scattered pieces in place cannot merge the
    # two copies of every joint. This is an exact invariant, not an intuition.
    for pair in puzzles[0]['train']:
        info = _parse(pair['input'])
        assert np.count_nonzero(info['grid'] == info['marker']) == 2*np.count_nonzero(pair['output'] == info['marker'])
    for a, b in [(2, 1), (10, 4)]:
        p = generate_puzzle(913+a, a, b)
        assert len(p['train']) == a and len(p['test']) == b
        validate_generated_puzzle(p)
    for a, b in [(1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = puzzles[0]['test'][0]
    bad = {k: v.copy() for k, v in pair.items()}
    info = _parse(bad['input'])
    q = tuple(np.argwhere(bad['output'] == info['marker'])[0])
    bad['output'][q] = info['background']
    try:
        _analyze_pair(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased joint accepted")


if __name__ == '__main__':
    self_test()
