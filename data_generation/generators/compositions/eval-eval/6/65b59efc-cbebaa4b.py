"""Contact printing: rigid docking creates two-owner stamp placement layers.

Only standard-library modules and NumPy are used. No parent code, file, example,
seed, training label, or cached generation data is consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b):
        raise ValueError("counts must be integers")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(a), int(b)


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


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


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _d4(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4)


def _parse(input_grid):
    """Read framed library, then segment physical objects, independently of hue."""
    g = _grid(input_grid)
    frame = int(g[0, 0])
    run = 0
    while run < g.shape[1] and g[0, run] == frame:
        run += 1
    n = run - 2
    if n not in (2, 3) or g.shape[0] < n + 10 or run >= g.shape[1]:
        raise ValueError("invalid library header")
    bg = int(g[0, run])
    if bg == frame:
        raise ValueError("invisible frame")
    starts = []
    c = 0
    while c + n + 2 <= g.shape[1] and g[0, c] == frame:
        starts.append(c)
        c += n + 3
    if not 2 <= len(starts) <= 4:
        raise ValueError("library must contain 2..4 cards")
    expected = np.full((n + 4, g.shape[1]), bg, dtype=np.int8)
    templates, targets = {}, {}
    order = []
    for c in starts:
        card = g[:n + 4, c:c + n + 2]
        if card.shape != (n + 4, n + 2):
            raise ValueError("truncated library")
        colors = set(map(int, np.unique(card[1:n + 1, 1:n + 1]))) - {bg}
        if len(colors) != 1:
            raise ValueError("a template needs one visible source identity")
        source = colors.pop()
        if source in (bg, frame) or source in templates:
            raise ValueError("invalid source identity")
        target = int(card[n + 2, 1 + n // 2])
        if target == bg:
            raise ValueError("invisible target swatch")
        e = np.full(card.shape, bg, dtype=np.int8)
        e[0] = e[-1] = frame
        e[:, 0] = e[:, -1] = frame
        mask = card[1:n + 1, 1:n + 1] == source
        e[1:n + 1, 1:n + 1][mask] = source
        e[n + 2, 1 + n // 2] = target
        expected[:, c:c + n + 2] = e
        templates[source], targets[source] = mask.copy(), target
        order.append(source)
    if not np.array_equal(g[:n + 4], expected):
        raise ValueError("malformed frame, template, key, or header padding")
    if (not np.all(g[n + 4] == bg) or not np.all(g[n + 5] == frame)
            or not np.all(g[n + 6] == bg)):
        raise ValueError("invalid scene divider")
    scene = g[n + 7:]
    cells = {tuple(map(int, q)) for q in np.argwhere(scene != bg)}
    cc = _components(cells)
    if not 3 <= len(cc) <= 5:
        raise ValueError("expected 3..5 spatially separate pieces")
    palettes = [{int(scene[q]) for q in component} for component in cc]
    common = set.intersection(*palettes)
    if len(common) != 1 or any(len(p) != 2 for p in palettes):
        raise ValueError("no unique common connector hue")
    marker = common.pop()
    if marker in {bg, frame, *templates} or marker in targets.values():
        raise ValueError("connector hue must be distinct from all ink roles")
    pieces = []
    used = set()
    for component, palette in zip(cc, palettes):
        source = (palette - {marker}).pop()
        if source not in templates:
            raise ValueError("piece has no library entry")
        body = {q for q in component if scene[q] == source}
        pins = component - body
        if len(_components(body)) != 1 or len(pins) < 2:
            raise ValueError("invalid piece body or pins")
        # Explicitly check the separation convention used for object identity.
        for r, c in component:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in cells and q not in component:
                        raise ValueError("scattered pieces touch")
        pieces.append(dict(body=body, pins=pins, source=source))
        used.add(source)
    if used != set(templates):
        raise ValueError("all library identities must be exercised")
    return dict(n=n, bg=bg, marker=marker, frame=frame, templates=templates,
                targets=targets, order=order, pieces=pieces, scene=scene)


class _SearchLimit(ValueError):
    pass


def _assemble(pieces, limit=7000):
    """Enumerate complete tree assemblies, identifying global translations.

    No pin components, colors, areas, source order, or generated joint labels
    are used as matching keys. Every candidate translation compares whole cells.
    A second positional solution or an incomplete search rejects the input.
    """
    count = len(pieces)
    links = {}
    for i, a in enumerate(pieces):
        for j, b in enumerate(pieces):
            if i == j:
                continue
            choices = []
            offsets = sorted({(r - u, c - v) for r, c in a['pins']
                              for u, v in b['pins']})
            for d in offsets:
                body, pins = _shift(b['body'], d), _shift(b['pins'], d)
                joint = a['pins'] & pins
                if len(joint) not in (2, 3):
                    continue
                if body & (a['body'] | a['pins']) or pins & a['body']:
                    continue
                choices.append((d, joint))
            links[i, j] = choices
    found, visited, nodes = [], set(), 0

    def search(positions, bodies, singles, doubles, owner, joints):
        nonlocal nodes
        state = tuple(sorted(positions.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise _SearchLimit("assembly uniqueness budget exhausted")
        if not singles:
            if len(positions) == count:
                found.append((dict(positions), list(joints)))
            return
        if len(positions) == count:
            return
        best = None
        for q in sorted(singles):
            i = owner[q]
            di = positions[i]
            local = q[0] - di[0], q[1] - di[1]
            options = []
            for j, b in enumerate(pieces):
                if j in positions:
                    continue
                for relative, local_joint in links[i, j]:
                    if local not in local_joint:
                        continue
                    d = di[0] + relative[0], di[1] + relative[1]
                    body, pins = _shift(b['body'], d), _shift(b['pins'], d)
                    if body & (bodies | singles | doubles) or pins & (bodies | doubles):
                        continue
                    overlap = pins & singles
                    parents = {owner[x] for x in overlap}
                    # A new tree vertex has exactly one neighbor in the
                    # already connected subassembly; cycles are outside family.
                    if parents != {i} or len(overlap) not in (2, 3):
                        continue
                    options.append((j, d, body, pins, overlap, i))
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for j, d, body, pins, overlap, i in best:
            new_owner = dict(owner)
            new_owner.update({x: j for x in pins - overlap})
            positions[j] = d
            search(positions, bodies | body, (singles | pins) - overlap,
                   doubles | overlap, new_owner, joints + [(i, j, overlap)])
            del positions[j]
            if len(found) >= 2:
                return

    a = pieces[0]
    search({0: (0, 0)}, set(a['body']), set(a['pins']), set(),
           {q: 0 for q in a['pins']}, [])
    if len(found) != 1:
        raise ValueError("assembly is impossible or ambiguous")
    positions, joints = found[0]
    return positions, joints, nodes


def _print_contacts(p, joints, *, mode='correct', turn=0):
    """Render only contact layers; preserve empty positions in their tight box."""
    occupied = set().union(*(cells for _, _, cells in joints))
    t, l, b, r = _box(occupied)
    n, bg, marker = p['n'], p['bg'], p['marker']
    out = np.full(((b - t + 1) * n, (r - l + 1) * n), bg, dtype=np.int8)
    labels = p['order']
    for i, j, cells in joints:
        a, z = p['pieces'][i]['source'], p['pieces'][j]['source']
        if mode == 'wrong_identity':
            a = labels[(labels.index(a) + 1) % len(labels)]
            z = labels[(labels.index(z) + 1) % len(labels)]
        ma, mz = _d4(p['templates'][a], turn), _d4(p['templates'][z], turn)
        ta, tz = p['targets'][a], p['targets'][z]
        if mode == 'no_key':
            ta, tz = a, z
        tile = np.full((n, n), bg, dtype=np.int8)
        tile[ma] = ta
        tile[mz] = tz
        if mode != 'last_wins':
            tile[ma & mz] = marker
        if mode == 'intersection_only':
            tile[ma ^ mz] = bg
        if mode == 'full_blocks':
            tile[:] = marker
        for x, y in cells:
            out[(x - t) * n:(x - t + 1) * n,
                (y - l) * n:(y - l + 1) * n] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    _, joints, _ = _assemble(p['pieces'])
    return _grid(_print_contacts(p, joints))


def _rect(t, l, h, w):
    return {(r, c) for r in range(t, t + h) for c in range(l, l + w)}


def _rotate_set(cells, turns):
    result = set(cells)
    for _ in range(turns % 4):
        result = {(c, -r) for r, c in result}
    return result


def _layout(rng, count, span):
    """Grow small rigid carriers with unequal rectangles and offset port arms."""
    h, w = map(int, rng.integers(2, 5, size=2))
    nodes = [dict(rect=_rect(0, 0, h, w), pins=set(), arms=set(), required=set(), sides=set())]
    edges = []
    for j in range(1, count):
        for attempt in range(100):
            i = int(rng.integers(j))
            k = int(rng.integers(4))
            if k in nodes[i]['sides']:
                continue
            rect = _rotate_set(nodes[i]['rect'], k)
            t, l, b, r = _box(rect)
            h, w = map(int, rng.integers(2, 5, size=2))
            top = int(rng.integers(t - h + 1, b + 1))
            gap = int(rng.integers(1, 3))
            left = r + gap + 1
            rows = list(range(max(t, top), min(b, top + h - 1) + 1))
            size = 3 if len(rows) >= 3 and rng.random() < .15 else 2
            if len(rows) < size:
                continue
            chosen = sorted(map(int, rng.choice(rows, size, replace=False)))
            offsets = list(map(int, rng.integers(gap, size=size)))
            pins = {(a, r + 1 + d) for a, d in zip(chosen, offsets)}
            pa = {(a, c) for a, d in zip(chosen, offsets) for c in range(r + 1, r + 1 + d)}
            ca = {(a, c) for a, d in zip(chosen, offsets) for c in range(r + 2 + d, left)}
            required_p = {(a, r) for a in chosen}
            required_c = {(a, left) for a in chosen}
            child = _rect(top, left, h, w)
            pins, pa, ca, required_p, required_c, child = [
                _rotate_set(x, -k) for x in (pins, pa, ca, required_p, required_c, child)]
            existing = [x['rect'] | x['pins'] | x['arms'] for x in nodes]
            if (child | ca | pins) & existing[i] or pa & nodes[i]['pins']:
                continue
            if any((child | ca | pins | pa) & x for m, x in enumerate(existing) if m != i):
                continue
            contacts = pins | set().union(*(e[2] for e in edges))
            t0, l0, b0, r0 = _box(contacts)
            if b0 - t0 + 1 > span or r0 - l0 + 1 > span:
                continue
            nodes[i]['pins'] |= pins
            nodes[i]['arms'] |= pa
            nodes[i]['required'] |= required_p
            nodes[i]['sides'].add(k)
            nodes.append(dict(rect=child, pins=set(pins), arms=ca,
                              required=required_c, sides={(k + 2) % 4}))
            edges.append((i, j, set(pins)))
            break
        else:
            return None
    return nodes, edges


def _body(rng, node):
    t, l, b, r = _box(node['rect'])
    style = int(rng.integers(4))
    if style == 0:
        body = set(node['rect'])
    elif style == 1:
        body = {q for q in node['rect'] if q[0] in (t, b) or q[1] in (l, r)}
    else:
        missing = int(rng.integers(4))
        body = {q for q in node['rect'] if any(
            (q[0] == t, q[0] == b, q[1] == l, q[1] == r)[s]
            for s in range(4) if s != missing)}
    for target in sorted(node['required']):
        a, c = min(body, key=lambda q: (abs(q[0] - target[0]) + abs(q[1] - target[1]), q))
        for axis in rng.permutation(2):
            if axis == 0:
                while a != target[0]:
                    a += 1 if target[0] > a else -1
                    body.add((a, c))
            else:
                while c != target[1]:
                    c += 1 if target[1] > c else -1
                    body.add((a, c))
    return body | node['arms']


def _scatter(rng, pieces, shape, bg, marker):
    h, w = shape
    for _ in range(12):
        occupied = set()
        result = np.full(shape, bg, dtype=np.int8)
        order = list(map(int, rng.permutation(len(pieces))))
        order.sort(key=lambda i: -len(pieces[i]['body'] | pieces[i]['pins']))
        for i in order:
            p = pieces[i]
            cells = p['body'] | p['pins']
            t, l, b, r = _box(cells)
            if b - t >= h or r - l >= w:
                break
            for attempt in range(250):
                d = int(rng.integers(h - b + t)) - t, int(rng.integers(w - r + l)) - l
                moved = _shift(cells, d)
                if moved & occupied:
                    continue
                for q in _shift(p['body'], d):
                    result[q] = p['source']
                for q in _shift(p['pins'], d):
                    result[q] = marker
                occupied |= {(a + dr, c + dc) for a, c in moved
                             for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                break
            else:
                break
        else:
            return result
    return None


def _header(n, bg, frame, templates, targets, order, width):
    result = np.full((n + 7, width), bg, dtype=np.int8)
    for j, source in enumerate(order):
        c = j * (n + 3)
        card = result[:n + 4, c:c + n + 2]
        card[0] = card[-1] = frame
        card[:, 0] = card[:, -1] = frame
        card[1:n + 1, 1:n + 1][templates[source]] = source
        card[n + 2, 1 + n // 2] = targets[source]
    result[n + 5] = frame
    return result


def _witness(p, joints):
    correct = _print_contacts(p, joints)
    if any(np.array_equal(correct, _print_contacts(p, joints, mode=m)) for m in
           ('wrong_identity', 'no_key', 'last_wins', 'intersection_only', 'full_blocks')):
        return False
    if any(np.array_equal(correct, _print_contacts(p, joints, turn=t)) for t in range(1, 8)):
        return False
    keys = [_normal(cells) for _, _, cells in joints]
    return any(len(a) == len(b) and a != b for i, a in enumerate(keys) for b in keys[:i])


def _pair(rng, n, count, witness):
    for attempt in range(180):
        colors = list(map(int, rng.permutation(10)))
        bg, marker, frame = colors[:3]
        k = 3 if witness else int(rng.integers(2, min(4, count) + 1))
        labels = colors[3:3 + k]
        templates = {}
        for source in labels:
            style = int(rng.integers(5))
            rr, cc = np.indices((n, n))
            if style == 0:
                mask = (rr + cc) % 2 == int(rng.integers(2))
            elif style == 1:
                mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
            elif style == 2 and n == 3:
                mask = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
            else:
                mask = rng.random((n, n)) < rng.uniform(.30, .85)
            if not mask.any():
                mask.flat[int(rng.integers(n * n))] = True
            templates[source] = mask
        target_pool = [c for c in range(10) if c not in (bg, marker)]
        values = list(map(int, rng.choice(target_pool, k, replace=not witness)))
        targets = dict(zip(labels, values))
        layout = _layout(rng, count, 30 // n)
        if layout is None:
            continue
        nodes, joints = layout
        sources = labels + list(map(int, rng.choice(labels, count - k)))
        rng.shuffle(sources)
        pieces = [dict(body=_body(rng, node), pins=node['pins'], source=source)
                  for node, source in zip(nodes, sources)]
        order = list(map(int, rng.permutation(labels)))
        p = dict(n=n, bg=bg, marker=marker, frame=frame, templates=templates,
                 targets=targets, order=order, pieces=pieces)
        if witness and not _witness(p, joints):
            continue
        # Check intended layout with a different input-relative placement search.
        try:
            positions, inferred, _ = _assemble(pieces)
        except ValueError:
            continue
        if any(d != (0, 0) for d in positions.values()):
            continue
        output = _print_contacts(p, joints)
        width = int(rng.integers(max(22, k * (n + 3) - 1), 31))
        height = int(rng.integers(15, 31 - (n + 7)))
        scene = _scatter(rng, pieces, (height, width), bg, marker)
        if scene is None:
            continue
        inp = np.concatenate([_header(n, bg, frame, templates, targets, order, width), scene])
        return dict(input=inp, output=output)
    raise RuntimeError("bounded contact-print construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    if regime == 0:  # Scale extension, with familiar piece counts.
        train = [(2, 3 if i == 0 else int(rng.integers(3, 5))) for i in range(a)]
        test = [(3, int(rng.integers(3, 5))) for _ in range(b)]
    elif regime == 1:  # Count extension with a fixed visibly encoded scale.
        n = int(rng.integers(2, 4))
        train = [(n, 3) for _ in range(a)]
        test = [(n, int(rng.integers(4, 6))) for _ in range(b)]
    else:  # Both scales demonstrated, then more interacting pieces.
        train = [(2 if i % 2 == 0 else 3, 3 if i == 0 else 4) for i in range(a)]
        test = [(int(rng.integers(2, 4)), 5) for _ in range(b)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, schedule in (('train', train), ('test', test)):
        for i, (n, count) in enumerate(schedule):
            for retry in range(8):
                pair = _pair(rng, n, count, split == 'train' and i == 0)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retry budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
        raise AssertionError("invalid pair schema")
    inp, output = _grid(pair['input']), _grid(pair['output'])
    p = _parse(inp)
    positions, joints, nodes = _assemble(p['pieces'])
    assert len(joints) == len(p['pieces']) - 1
    bodies, pins, owners = set(), set(), Counter()
    adjacency = [set() for _ in p['pieces']]
    for i, piece in enumerate(p['pieces']):
        body, marker_cells = _shift(piece['body'], positions[i]), _shift(piece['pins'], positions[i])
        assert not bodies & body and not body & pins and not marker_cells & bodies
        bodies |= body
        pins |= marker_cells
        owners.update(marker_cells)
    assert set(owners.values()) == {2}
    assert all(len(cells) in (2, 3) for _, _, cells in joints)
    for i, j, cells in joints:
        adjacency[i].add(j)
        adjacency[j].add(i)
    assert sum(map(len, adjacency)) == 2 * (len(p['pieces']) - 1)
    predicted = _print_contacts(p, joints)
    assert np.array_equal(output, predicted), "contact print mismatch"
    # Independently build incidence counts per output pixel, rather than use
    # tile overwrite precedence, to verify the actual overprinting invariant.
    t, l, _, _ = _box(pins)
    coverage = np.zeros(output.shape, dtype=np.int8)
    ink = np.full(output.shape, p['bg'], dtype=np.int8)
    for i, piece in enumerate(p['pieces']):
        mask = p['templates'][piece['source']]
        for r, c in _shift(piece['pins'], positions[i]):
            y, x = (r - t) * p['n'], (c - l) * p['n']
            coverage[y:y + p['n'], x:x + p['n']] += mask
            tile = ink[y:y + p['n'], x:x + p['n']]
            tile[mask] = p['targets'][piece['source']]
    ink[coverage == 2] = p['marker']
    assert np.max(coverage) <= 2 and np.array_equal(ink, output)
    return p, joints, dict(n=p['n'], pieces=len(p['pieces']), labels=len(p['templates']),
        degrees=sorted(map(len, adjacency)), pin_keys=[_normal(x) for _, _, x in joints],
        body_boxes=[(b - t + 1, r - l + 1) for t, l, b, r in
                    (_box(x['body']) for x in p['pieces'])],
        body_masks=[_normal(x['body']) for x in p['pieces']],
        motifs=[tuple(map(int, x.flat)) for x in p['templates'].values()],
        colors=[p['bg'], p['marker'], p['frame']], shape=inp.shape,
        output_shape=output.shape, search_nodes=nodes,
        reuse=len(p['pieces']) - len(p['templates']),
        double_ink=int(np.count_nonzero(coverage == 2)),
        single_ink=int(np.count_nonzero(coverage == 1)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records, training = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            p, joints, record = _check_pair(pair)
            inp = np.asarray(pair['input'])
            key = inp.shape, inp.astype(np.int8).tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            records.append((split, record))
            if split == 'train':
                training.append((p, joints))
    assert any(_witness(p, joints) for p, joints in training), "missing demonstration counterexamples"
    assert any(r['double_ink'] and r['single_ink'] for s, r in records if s == 'train')
    tr = [r for s, r in records if s == 'train']
    te = [r for s, r in records if s == 'test']
    assert (min(r['n'] for r in te) > max(r['n'] for r in tr)
            or min(r['pieces'] for r in te) > max(r['pieces'] for r in tr)), "missing scale/count shift"


def self_test():
    """Finite 42-seed suite; no files, logs, caches, or retained puzzle list."""
    shapes, bodies, motifs, keys, roles, topologies, signatures = set(), set(), set(), set(), set(), set(), set()
    counts, scales, reuses, nodes = set(), set(), set(), []
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p, joints, r = _check_pair(pair)
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                shapes.add(tuple(r['shape']))
                bodies.update(r['body_masks'])
                motifs.update((r['n'], m) for m in r['motifs'])
                keys.update(r['pin_keys'])
                roles.add(tuple(r['colors']))
                topologies.add(tuple(r['degrees']))
                counts.add(r['pieces'])
                scales.add(r['n'])
                reuses.add(r['reuse'])
                nodes.append(r['search_nodes'])
                totals['pairs'] += 1
                totals['branches'] += max(r['degrees']) >= 3
                totals['reused_source_pairs'] += r['reuse'] > 0
                totals['offset_keys'] += any(len({c for _, c in k}) > 1 and len({a for a, _ in k}) > 1 for k in r['pin_keys'])
                for role in ('input', 'output'):
                    a = pair[role]
                    signature.update(bytes(a.shape) + a.tobytes())
        signatures.add(signature.digest())
    assert len(signatures) == 42
    assert counts == {3, 4, 5} and scales == {2, 3}
    assert len(shapes) >= 12 and len(bodies) >= 60 and len(motifs) >= 45
    assert len(keys) >= 8 and len(roles) >= 35 and len(topologies) >= 4
    assert max(reuses) >= 2
    assert totals['branches'] >= 5 and totals['offset_keys'] >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(7, 2, 1), generate_puzzle(7, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    wrong = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    wrong['test'][0]['output'][0, 0] = (int(wrong['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect contact pixel accepted")
    # Input-only invariance: relocate complete pieces, keeping the library
    # fixed. Neither reading order nor initial distances may determine print.
    pair = p['train'][0]
    parsed = _parse(pair['input'])
    scattered = _scatter(np.random.default_rng(1024), parsed['pieces'],
                         parsed['scene'].shape, parsed['bg'], parsed['marker'])
    assert scattered is not None
    relocated = np.concatenate([pair['input'][:parsed['n'] + 7], scattered])
    assert not np.array_equal(relocated, pair['input'])
    assert np.array_equal(solve(relocated), pair['output'])
    # Input-only invariance: reorder whole library cards, including their
    # keys. Source identity, rather than library position, must select stamps.
    permuted = pair['input'].copy()
    permuted[:parsed['n'] + 7] = _header(parsed['n'], parsed['bg'], parsed['frame'],
        parsed['templates'], parsed['targets'], parsed['order'][::-1], permuted.shape[1])
    assert np.array_equal(solve(permuted), pair['output'])
    # A missing physical pin must not be silently repaired or ignored.
    erased = pair['input'].copy()
    q0 = min(parsed['pieces'][0]['pins'])
    erased[q0[0] + parsed['n'] + 7, q0[1]] = parsed['bg']
    try:
        solve(erased)
    except ValueError:
        pass
    else:
        raise AssertionError("missing connector accepted")
    # Same-area, same-shape local keys can allow interchangeable leaves.
    # Exhaustive search must reject the two different placements.
    ambiguous = [
        dict(body={(r, 0) for r in range(5)},
             pins={(0, 1), (1, 1), (3, 1), (4, 1)}, source=1),
        dict(body={(10, 10), (11, 10)}, pins={(10, 9), (11, 9)}, source=2),
        dict(body={(20, 20), (21, 20)}, pins={(20, 19), (21, 19)}, source=3)]
    try:
        _assemble(ambiguous)
    except ValueError as exc:
        assert 'ambiguous' in str(exc)
    else:
        raise AssertionError("interchangeable leaves accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), pairs=totals['pairs'],
        input_shapes=len(shapes), body_masks=len(bodies), motifs=len(motifs),
        pin_keys=len(keys), role_assignments=len(roles), topologies=len(topologies),
        max_search_nodes=max(nodes), branches=totals['branches'],
        reused_source_pairs=totals['reused_source_pairs'], offset_key_pairs=totals['offset_keys'])


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
