"""Rooted tally networks: subtree addition constrains geometric docking.

Only Python's standard library and NumPy are used. The constructor draws labels
from explicit pieces; solve reconstructs pieces, orientation, tree and tallies
from the input alone. No parent generator is imported.
"""
from itertools import product
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        todo, found = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    todo.append(q)
                    found.add(q)
        result.append(found)
    return result


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _rect(y, x, h, w):
    return {(r, c) for r in range(y, y + h) for c in range(x, x + w)}


def _norm(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _in_bounds(cells, shape):
    t, l, b, r = _box(cells)
    return t >= 0 and l >= 0 and b < shape[0] and r < shape[1]


def _token_cells(bottom, mask, count):
    h = max(r for r, c in mask) + 1
    return {(bottom[0] - k * (h + 1) + r, bottom[1] + c)
            for k in range(count) for r, c in mask}


def _canonical_parses(grid):
    """Try visible stack directions, chassis-ring hues, and common pin hues."""
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    for turn in range(4):
        a = np.rot90(grid, turn)
        rings = []
        for y in range(a.shape[0] - 2):
            for x in range(a.shape[1] - 2):
                hue = int(a[y, x])
                if hue == bg or int(a[y + 1, x + 1]) != bg:
                    continue
                perimeter = _rect(y, x, 3, 3) - {(y + 1, x + 1)}
                if all(int(a[q]) == hue for q in perimeter):
                    rings.append((hue, perimeter))
        # The single hollow chassis square is the visible fixed-root cue.
        if len(rings) != 1:
            continue
        frame, ring = rings[0]
        colors = sorted(set(int(v) for v in a.flat) - {bg, frame})
        for marker in colors:
            skeleton = {tuple(q) for q in np.argwhere((a == frame) | (a == marker))}
            cc = _components(skeleton)
            if not 3 <= len(cc) <= 8:
                continue
            if not all(any(a[q] == frame for q in comp) and
                       sum(a[q] == marker for q in comp) >= 2 for comp in cc):
                continue
            roots = [i for i, comp in enumerate(cc) if ring <= comp]
            if len(roots) != 1:
                continue
            owner = {q: i for i, comp in enumerate(cc) for q in comp}
            tokens = _components({tuple(q) for q in np.argwhere(
                (a != bg) & (a != frame) & (a != marker))})
            groups = [[] for _ in cc]
            good = True
            for comp in tokens:
                y, x, b, r = _box(comp)
                mask = _norm(comp)
                hues = {int(a[q]) for q in comp}
                if len(hues) != 1 or b - y > 1 or r - x > 1:
                    good = False
                    break
                hit = next(((v, x) for v in range(b + 1, a.shape[0])
                            if (v, x) in owner), None)
                if hit is None or int(a[hit]) != frame:
                    good = False
                    break
                groups[owner[hit]].append((y, x, mask, hues.pop()))
            if not good or any(not g for g in groups):
                continue
            pieces = []
            masks = set()
            for comp, group in zip(cc, groups):
                group.sort()
                mask, hue = group[0][2:]
                h = max(r for r, c in mask) + 1
                y, x = group[-1][:2]
                if any(g[1:] != (x, mask, hue) for g in group):
                    good = False
                    break
                if [g[0] for g in group] != list(range(
                        y - (len(group) - 1) * (h + 1), y + 1, h + 1)):
                    good = False
                    break
                if (y + h + 1, x) not in comp or a[y + h + 1, x] != frame:
                    good = False
                    break
                pins = {q for q in comp if a[q] == marker}
                chassis = comp - pins
                tally = _token_cells((y, x), mask, len(group))
                pieces.append(dict(frame=chassis, pins=pins, tokens=tally,
                                   body=chassis | tally, cells=comp | tally,
                                   mask=mask, bottom=(y, x), count=len(group), hue=hue))
                masks.add(mask)
            if good and len(masks) == 1:
                yield dict(grid=a, turn=turn, bg=bg, frame=frame,
                           marker=marker, root=roots[0], pieces=pieces)


def _tree_counts(pieces, pins, root):
    n = len(pieces)
    adjacency = [[] for _ in pieces]
    edges = 0
    for i in range(n):
        for j in range(i):
            if pins[i] & pins[j]:
                adjacency[i].append(j)
                adjacency[j].append(i)
                edges += 1
    if edges != n - 1:
        raise ValueError('docking graph must be a tree')
    parent, order = {root: -1}, [root]
    for i in order:
        for j in adjacency[i]:
            if j == parent[i]:
                continue
            if j in parent:
                raise ValueError('cycle')
            parent[j] = i
            order.append(j)
    if len(order) != n:
        raise ValueError('disconnected graph')
    counts = [p['count'] for p in pieces]
    for i in reversed(order[1:]):
        counts[parent[i]] += counts[i]
    return counts, adjacency


def _render(parsed, positions, *, aggregate=True):
    ps, root = parsed['pieces'], parsed['root']
    shape = parsed['grid'].shape
    pins = [_shift(p['pins'], positions[i]) for i, p in enumerate(ps)]
    counts, adjacency = _tree_counts(ps, pins, root)
    result = np.full(shape, parsed['bg'], dtype=np.int8)
    occupied, pin_owners = set(), {}
    for i, p in enumerate(ps):
        chassis = _shift(p['frame'], positions[i])
        if not _in_bounds(chassis, shape) or occupied & chassis:
            raise ValueError('chassis collision or boundary')
        occupied |= chassis
        for q in chassis:
            result[q] = parsed['frame']
        for q in pins[i]:
            pin_owners[q] = pin_owners.get(q, 0) + 1
    if set(pin_owners.values()) != {2} or occupied & set(pin_owners):
        raise ValueError('pins must have exactly two owners')
    occupied |= set(pin_owners)
    for q in pin_owners:
        result[q] = parsed['marker']
    for i, p in enumerate(ps):
        d = positions[i]
        bottom = (p['bottom'][0] + d[0], p['bottom'][1] + d[1])
        tally = _token_cells(bottom, p['mask'], counts[i] if aggregate else p['count'])
        if not _in_bounds(tally, shape):
            raise ValueError('tally boundary')
        if occupied & tally:
            raise ValueError('tally collision')
        occupied |= tally
        hue = p['hue'] if i == root or not aggregate else parsed['frame']
        for q in tally:
            result[q] = hue
    return result, counts, adjacency


class _SearchLimit(ValueError):
    pass


def _reconstruct(parsed, limit=12000):
    """Search whole rigid carriers, then enforce the expanded tally constraints.

    Individual pins, not their connected components or area, induce alignments.
    Every accepted joint pixel has exactly two owners. The search considers all
    connected placements before enforcing the tree rule and subtree arithmetic.
    """
    ps, root = parsed['pieces'], parsed['root']
    n, shape = len(ps), parsed['grid'].shape
    links = {}
    for i, a in enumerate(ps):
        for j, b in enumerate(ps):
            if i == j:
                continue
            candidates = {}
            offsets = sorted({(r - s, c - t) for r, c in a['pins'] for s, t in b['pins']})
            for d in offsets:
                body, pins = _shift(b['body'], d), _shift(b['pins'], d)
                if a['cells'] & body or a['body'] & pins:
                    continue
                for q in sorted(a['pins'] & pins):
                    candidates.setdefault(q, []).append((j, d))
            links[i, j] = candidates
    answers, visited, geometric = {}, set(), set()
    rejected = set()
    nodes = 0

    def visit(positions, bodies, singles, doubles, owners):
        nonlocal nodes
        state = tuple(sorted(positions.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise _SearchLimit('complete uniqueness search exceeded budget')
        if not singles:
            if len(positions) != n:
                return
            try:
                original, _, _ = _render(parsed, positions, aggregate=False)
            except ValueError:
                return
            geometric.add(original.tobytes())
            try:
                out, counts, adj = _render(parsed, positions)
            except ValueError as exc:
                rejected.add(str(exc))
                return
            answers.setdefault(out.tobytes(), (out, dict(positions), counts, adj))
            return
        if len(positions) == n:
            return
        best = None
        for q in sorted(singles):
            i, di = owners[q], positions[owners[q]]
            local = q[0] - di[0], q[1] - di[1]
            choices = []
            for j, b in enumerate(ps):
                if j in positions:
                    continue
                for _, relative in links[i, j].get(local, []):
                    d = di[0] + relative[0], di[1] + relative[1]
                    body, pins = _shift(b['body'], d), _shift(b['pins'], d)
                    if not _in_bounds(body | pins, shape):
                        continue
                    if body & (bodies | singles | doubles) or pins & (bodies | doubles):
                        continue
                    overlap = pins & singles
                    choices.append((j, d, body, pins, overlap))
            if not choices:
                return
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
        for j, d, body, pins, overlap in best:
            new_owners = dict(owners)
            new_owners.update({q: j for q in pins - overlap})
            positions[j] = d
            visit(positions, bodies | body, (singles | pins) - overlap,
                  doubles | overlap, new_owners)
            del positions[j]
            if len(answers) > 1:
                return

    a = ps[root]
    visit({root: (0, 0)}, a['body'], a['pins'], set(), {q: root for q in a['pins']})
    return list(answers.values()), dict(nodes=nodes, geometric=len(geometric),
                                      rejected=tuple(sorted(rejected)))


def _analyze(input_grid):
    a = np.asarray(input_grid)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('invalid input grid')
    parses = list(_canonical_parses(a))
    if len(parses) != 1:
        raise ValueError('input must visibly identify one stack direction and palette')
    parsed = parses[0]
    answers, info = _reconstruct(parsed)
    if len(answers) != 1:
        raise ValueError('input must have exactly one expanded tally assembly')
    return parsed, answers[0], info


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed, (out, _, _, _), _ = _analyze(input_grid)
    return np.rot90(out, -parsed['turn']).copy()


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError('pair counts must be integers within their legal ranges')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _count_options(length, extra, capacity):
    options = []
    for branch in product(range(1, 4), repeat=length):
        for leaf in range(1, 4):
            if max(branch) > leaf or sum(branch) <= leaf:
                continue
            for side in (range(1, 3) if extra else (0,)):
                s = sum(branch) + leaf + side
                for d in range(1, 4):
                    if s + d <= capacity:
                        options.append((d, leaf, branch, side))
    return options


def _operands(config):
    d, leaf, branch, side = config
    return sum(branch) + leaf + side, d


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _new_piece(frame, pins, bottom, mask, count, hue):
    tokens = _token_cells(bottom, mask, count)
    return dict(frame=frame, pins=pins, bottom=bottom, mask=mask,
                count=count, hue=hue, tokens=tokens, body=frame | tokens,
                cells=frame | pins | tokens)


def _layout(rng, config, mask, palette):
    d, leaf_count, branch_counts, side_count = config
    h = max(r for r, c in mask) + 1
    tw = max(c for r, c in mask) + 1
    pitch = h + 1
    spacing = int(rng.integers(2, 5))
    bh = int(rng.integers(spacing + 1, 6))
    leaf_width, first_width = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    # Upper slot accommodates both original stacks, but not the branch subtotal.
    high = h + 1 + (leaf_count - 1) * pitch + int(rng.integers(
        0, min(3, (sum(branch_counts) - leaf_count) * pitch)))
    gap = bh + h + (sum(branch_counts) - 1) * pitch + 1 + int(rng.integers(3))
    low = high + gap
    total = d + leaf_count + sum(branch_counts) + side_count
    root_y = max(low + bh + int(rng.integers(3)), h + 1 + (total - 1) * pitch)
    ring = _rect(root_y, 0, 3, 3) - {(root_y + 1, 1)}
    root_frame = ring | {(r, 2) for r in range(high, root_y + 3)}
    root_pins = {(r, 3) for y in (high, low) for r in (y, y + spacing)}
    if side_count:
        root_pins |= {(root_y, -1), (root_y + 1, -1)}
    ps = [_new_piece(root_frame, root_pins, (root_y - h - 1, 0 if tw == 2 else 1),
                     mask, d, palette[3])]
    leaf_frame = _rect(high, 4, bh, leaf_width)
    leaf_pins = {(high, 3), (high + spacing, 3)}
    ps.append(_new_piece(leaf_frame, leaf_pins,
                        (high - h - 1, 4 + (leaf_width - tw) // 2),
                        mask, leaf_count, _pick(rng, palette[3:])))
    starts = [(low, 4, bh, first_width)]
    frames = [_rect(*starts[0])]
    pins = [{(low, 3), (low + spacing, 3)}]
    used_keys = {(spacing, 0)}
    for _ in branch_counts[1:]:
        py, px, ph, pw = starts[-1]
        keys = [(dy, dx) for dy in range(1, ph) for dx in (-1, 0, 1)
                if (dy, dx) not in used_keys]
        dy, dx = _pick(rng, keys)
        used_keys.add((dy, dx))
        tangent = int(rng.integers(ph - dy))
        row = py + tangent
        normal_gap = int(rng.integers(1, 3)) if dx == 0 else 2
        left = px + pw + normal_gap
        x1 = px + pw + (1 if dx < 0 else 0)
        x2 = x1 + dx
        joint = {(row, x1), (row + dy, x2)}
        ch = int(rng.integers(dy + 1, 6))
        cy = row
        cw = int(rng.integers(3, 6))
        child = _rect(cy, left, ch, cw)
        for r, c in joint:
            frames[-1] |= {(r, x) for x in range(px + pw, c)}
            child |= {(r, x) for x in range(c + 1, left)}
        pins[-1] |= joint
        frames.append(child)
        pins.append(set(joint))
        starts.append((cy, left, ch, cw))
    for (y, x, bh_i, bw_i), frame, pin, count in zip(starts, frames, pins, branch_counts):
        ps.append(_new_piece(frame, pin, (y - h - 1, x + (bw_i - tw) // 2),
                             mask, count, _pick(rng, palette[3:])))
    if side_count:
        cw, ch = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        cy, cx = root_y - int(rng.integers(ch - 1)), -1 - cw
        ps.append(_new_piece(_rect(cy, cx, ch, cw),
                            {(root_y, -1), (root_y + 1, -1)},
                            (cy - h - 1, cx + (cw - tw) // 2),
                            mask, side_count, _pick(rng, palette[3:])))
    return ps


def _scatter(rng, shape, ps, frame, marker, bg):
    boxes = [_box(p['cells']) for p in ps]
    for _ in range(20):
        positions = {0: (0, 0)}
        blocked = [boxes[0]]
        order = list(range(1, len(ps)))
        rng.shuffle(order)
        order.sort(key=lambda i: -(boxes[i][2]-boxes[i][0]+2) * (boxes[i][3]-boxes[i][1]+2))
        for i in order:
            t, l, b, r = boxes[i]
            bh, bw = b - t + 1, r - l + 1
            if bh + 2 > shape[0] or bw + 2 > shape[1]:
                break
            for _ in range(400):
                y = int(rng.integers(1, shape[0] - bh))
                x = int(rng.integers(1, shape[1] - bw))
                delta = y - t, x - l
                if delta == (0, 0):
                    continue
                candidate = (y, x, y + bh - 1, x + bw - 1)
                if any(not (candidate[2] + 1 < a or candidate[0] - 1 > c or
                            candidate[3] + 1 < b0 or candidate[1] - 1 > d0)
                       for a, b0, c, d0 in blocked):
                    continue
                positions[i] = delta
                blocked.append(candidate)
                break
            else:
                break
        if len(positions) == len(ps):
            a = np.full(shape, bg, dtype=np.int8)
            for i, p in enumerate(ps):
                for role, hue in (('frame', frame), ('pins', marker), ('tokens', p['hue'])):
                    for q in _shift(p[role], positions[i]):
                        a[q] = hue
            return a
    return None


def _build_pair(rng, config, mask, palette, rotation, reflect):
    for attempt in range(160):
        ps = _layout(rng, config, mask, palette)
        all_cells = set().union(*(p['cells'] for p in ps))
        t, l, b, r = _box(all_cells)
        if t < 0 or b + 2 > 30 or r - l + 3 > 30:
            continue
        height = int(rng.integers(max(20, b + 2), 31))
        width = int(rng.integers(max(16, r - l + 3), 31))
        offset = (0, int(rng.integers(1, width - (r - l) - 1)) - l)
        placed = []
        for p in ps:
            q = dict(p)
            for key in ('frame', 'pins', 'tokens', 'body', 'cells'):
                q[key] = _shift(p[key], offset)
            q['bottom'] = p['bottom'][0], p['bottom'][1] + offset[1]
            placed.append(q)
        parsed = dict(grid=np.empty((height, width), dtype=np.int8), bg=palette[0],
                      frame=palette[1], marker=palette[2], root=0, pieces=placed)
        try:
            output, _, _ = _render(parsed, {i: (0, 0) for i in range(len(ps))})
        except ValueError:
            continue
        inp = _scatter(rng, (height, width), placed, palette[1], palette[2], palette[0])
        if inp is None:
            continue
        try:
            inferred, (pred, _, _, _), info = _analyze(inp)
        except ValueError:
            continue
        if (not np.array_equal(pred, output) or inferred['turn'] != 0
                or info['geometric'] < 2 or 'tally boundary' not in info['rejected']):
            continue
        inp, output = np.rot90(inp, rotation).copy(), np.rot90(output, rotation).copy()
        if reflect:
            inp, output = inp[:, ::-1].copy(), output[:, ::-1].copy()
        return {'input': inp, 'output': output}
    raise RuntimeError('bounded construction failed to make a unique tally-constrained assembly')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    h = int(rng.choice([1, 2], p=[0.7, 0.3]))
    capacity = 8 if h == 1 else 6
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    # Both branch aggregation and ordinary leaves are demonstrated even at 2/1.
    extras = [0, 1]
    rng.shuffle(extras)
    choices = [_count_options(2, e, capacity) for e in extras]
    compatible = [(a, b) for a in choices[0] for b in choices[1]
                  if _operands(a)[0] != _operands(b)[0]
                  and a[0] != b[0]
                  and _operands(a)[0] * b[0] != _operands(b)[0] * a[0]
                  and (_operands(a)[0] != len(a[2]) + 1 + bool(a[3])
                       or _operands(b)[0] != len(b[2]) + 1 + bool(b[3]))]
    first, second = _pick(rng, compatible)
    configs = [first, second]
    for index in range(2, num_train + num_test):
        length = 2 if index < num_train else int(rng.integers(3, 5))
        extra = int(rng.integers(2))
        opts = _count_options(length, extra, capacity)
        if not opts:
            opts = _count_options(length, 0, capacity)
        configs.append(_pick(rng, opts))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, config in enumerate(configs):
        # Local visible silhouette determines pitch, including singleton queries.
        w = int(rng.integers(1, 3))
        mask = _rect(0, 0, h, w)
        if h == w == 2 and rng.integers(2):
            mask.remove((0, int(rng.integers(2))))
        mask = tuple(sorted(mask))
        for _ in range(12):
            pair = _build_pair(rng, config, mask, palette, rotation, reflect)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError('bounded distinct-input construction failed')
        puzzle['train' if index < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def puzzle_features(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            parsed, (out, positions, counts, adj), info = _analyze(pair['input'])
            expected = np.rot90(out, -parsed['turn'])
            assert np.array_equal(expected, pair['output']), 'wrong expanded assembly'
            ps, root = parsed['pieces'], parsed['root']
            depths, todo = {root: 0}, [root]
            for i in todo:
                for j in adj[i]:
                    if j not in depths:
                        depths[j] = depths[i] + 1
                        todo.append(j)
            assert positions[root] == (0, 0)
            assert all(positions[i] != (0, 0) for i in range(len(ps)) if i != root)
            # Rigid chassis and connector multiplicity, plus exact subtree tallies.
            assert np.count_nonzero(parsed['grid'] == parsed['marker']) == 2 * np.count_nonzero(out == parsed['marker'])
            assert counts[root] == sum(p['count'] for p in ps)
            assert any(counts[i] > ps[i]['count'] for i in range(len(ps)) if i != root)
            assert info['geometric'] >= 2 and 'tally boundary' in info['rejected']
            bodies = []
            keys = []
            moved_pins = [_shift(p['pins'], positions[i]) for i, p in enumerate(ps)]
            for i, p in enumerate(ps):
                bodies.append(_norm(p['frame']))
                for j in range(i):
                    joint = moved_pins[i] & moved_pins[j]
                    if joint:
                        keys.append(_norm(joint))
            assert len({len(k) for k in keys}) == 1 and len(set(keys)) > 1
            sources = sum(p['count'] for i, p in enumerate(ps) if i != root)
            records.append(dict(split=split, shape=tuple(np.asarray(pair['input']).shape),
                                roles=(parsed['bg'], parsed['frame'], parsed['marker'], ps[root]['hue']),
                                count=len(ps), depth=max(depths.values()), root_degree=len(adj[root]),
                                source_total=sources, destination=ps[root]['count'], final=counts[root],
                                local_counts=tuple(sorted(p['count'] for p in ps)),
                                subtree_counts=tuple(sorted(counts)), keys=tuple(sorted(keys)),
                                bodies=tuple(bodies), mask=ps[root]['mask'], direction=parsed['turn'],
                                repeated=len({p['hue'] for p in ps}) < len(ps),
                                geometric=info['geometric'], search_nodes=info['nodes']))
    return records


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        for value in pair.values():
            a = np.asarray(value)
            assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
            assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
        inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        key = inp.shape, inp.astype(np.int8).tobytes()
        assert key not in seen
        seen.add(key)
    rows = puzzle_features(puzzle)
    assert len({r['roles'] for r in rows}) == 1
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    assert {2, 3} <= {r['root_degree'] for r in train}
    assert min(r['depth'] for r in test) > max(r['depth'] for r in train)
    ss, dd = [r['source_total'] for r in train], [r['destination'] for r in train]
    assert len(set(ss)) > 1 and len(set(dd)) > 1
    assert any(ss[0] * d != s * dd[0] for s, d in zip(ss, dd))
    rivals = (lambda s, d: 2*d, lambda s, d: s*d, lambda s, d: max(s, d),
              lambda s, d: s, lambda s, d: 2*s)
    for fn in rivals:
        assert any(fn(s, d) != s + d for s, d in zip(ss, dd))
    assert any(r['source_total'] != r['count'] - 1 for r in train), 'counting carriers is insufficient'


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
               for s in ('train', 'test'))


def self_test() -> dict:
    """Bounded, pixel-derived diversity and family-invariant checks on 42 seeds."""
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        rows = puzzle_features(p)
        records.extend(rows)
        signatures.add(tuple((q['input'].shape, q['input'].tobytes())
                             for q in p['train'] + p['test']))
        assert _equal(p, generate_puzzle(seed))
        # Dropping either mechanism changes at least one demonstrated answer.
        parsed, (out, positions, counts, _), _ = _analyze(p['train'][0]['input'])
        without_addition, _, _ = _render(parsed, positions, aggregate=False)
        assert not np.array_equal(out, without_addition)
        stationary = {i: (0, 0) for i in range(len(parsed['pieces']))}
        assert any(positions[i] != stationary[i] for i in positions)
    assert len(signatures) == 42
    assert len({r['roles'] for r in records}) >= 30
    assert {4, 5, 6, 7} <= {r['count'] for r in records}
    assert {2, 3, 4} <= {r['depth'] for r in records}
    assert {2, 3} <= {r['root_degree'] for r in records}
    assert len({r['shape'] for r in records}) >= 20
    assert len({b for r in records for b in r['bodies']}) >= 30
    assert len({k for r in records for k in r['keys']}) >= 10
    assert len({r['mask'] for r in records}) >= 6
    assert len({r['direction'] for r in records}) == 4
    assert any(r['repeated'] for r in records) and any(not r['repeated'] for r in records)
    assert max(r['final'] for r in records) >= 7
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7919 + a * 10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0, 2, 1)
    parsed, (out, positions, _, _), _ = _analyze(p['test'][0]['input'])
    root = parsed['root']
    for defect in ('missing_token', 'wrong_highlight', 'erased_joint', 'local_only'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in p[s]]
               for s in ('train', 'test')}
        modified = out.copy()
        if defect == 'missing_token':
            rp = parsed['pieces'][root]
            modified[tuple(np.argwhere(modified == rp['hue'])[0])] = parsed['bg']
        elif defect == 'wrong_highlight':
            i = next(i for i in positions if i != root)
            q = next(iter(_shift(parsed['pieces'][i]['tokens'], positions[i])))
            modified[q] = parsed['pieces'][i]['hue']
        elif defect == 'erased_joint':
            modified[tuple(np.argwhere(modified == parsed['marker'])[0])] = parsed['bg']
        else:
            modified = _render(parsed, positions, aggregate=False)[0]
        bad['test'][0]['output'] = np.rot90(modified, -parsed['turn']).copy()
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('accepted defect: ' + defect)
    return dict(seeds=42, pairs=len(records), palettes=len({r['roles'] for r in records}),
                shapes=len({r['shape'] for r in records}),
                chassis_masks=len({b for r in records for b in r['bodies']}),
                key_masks=len({k for r in records for k in r['keys']}),
                token_masks=len({r['mask'] for r in records}),
                piece_counts=sorted({r['count'] for r in records}),
                depths=sorted({r['depth'] for r in records}),
                max_search_nodes=max(r['search_nodes'] for r in records))
