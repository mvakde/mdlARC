"""Delta-cartography: repeated maps reveal T-shaped transfer commands.

Only the visible input is needed by solve. Construction keeps region IDs;
prediction instead segments the repeated original map from scratch.
"""
from collections import Counter
import hashlib
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _step(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _stencil(p, d):
    side = (-d[1], d[0])
    return {p, _step(p, d), _step(p, side), _step(p, side, -1)}


def _read_t(points):
    points = set(points)
    found = [(p, d) for p in sorted(points) for d in DIRECTIONS
             if _stencil(p, d) == points]
    _need(len(found) == 1, "difference mask must be exactly one T tetromino")
    return found[0]


def _components(a):
    ids = np.full(a.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for p in np.ndindex(a.shape):
        if ids[p] >= 0:
            continue
        label = len(colors)
        colors.append(int(a[p]))
        stack, group = [p], []
        ids[p] = label
        while stack:
            q = stack.pop()
            group.append(q)
            for d in DIRECTIONS:
                v = _step(q, d)
                if _inside(v, a.shape) and ids[v] < 0 and a[v] == a[p]:
                    ids[v] = label
                    stack.append(v)
        cells.append(group)
    return ids, colors, cells


def _assemble(rows, lattice):
    ph, pw = rows[0][0].shape
    out = np.full((len(rows)*(ph+1)+1, len(rows[0])*(pw+1)+1), lattice, dtype=np.int8)
    for b, row in enumerate(rows):
        for c, panel in enumerate(row):
            r0, c0 = 1+b*(ph+1), 1+c*(pw+1)
            out[r0:r0+ph, c0:c0+pw] = panel
    return out


def _parse(a):
    lattice = int(a[0, 0])
    rs = np.flatnonzero(np.all(a == lattice, axis=1))
    cs = np.flatnonzero(np.all(a == lattice, axis=0))
    _need(len(rs) >= 3 and len(cs) >= 4, "need at least two bands and three copies")
    _need(rs[0] == cs[0] == 0 and rs[-1] == a.shape[0]-1 and cs[-1] == a.shape[1]-1,
          "incomplete outer lattice")
    _need(bool(np.all(np.diff(rs) == rs[1])) and bool(np.all(np.diff(cs) == cs[1])),
          "irregular lattice")
    rows = [[a[r+1:s, c+1:t].copy() for c, t in zip(cs, cs[1:])]
            for r, s in zip(rs, rs[1:])]
    _need(all(p.size and np.all(p != lattice) for row in rows for p in row),
          "lattice color inside a map")
    counts = Counter(_key(p) for row in rows for p in row)
    maximum = max(counts.values())
    winners = [k for k, n in counts.items() if n == maximum]
    _need(len(winners) == 1, "tied master")
    master = next(p for row in rows for p in row if _key(p) == winners[0])
    _need(maximum > sum(counts.values())//2, "master lacks strict majority")
    return lattice, rows, master


def _analyze(input_grid):
    a = _grid(input_grid)
    lattice, rows, master = _parse(a)
    ids, colors, cells = _components(master)
    commands, columns, balanced, crossing = [], [], [], []
    assignments = {}
    for row in rows:
        variants = [(c, p) for c, p in enumerate(row) if not np.array_equal(p, master)]
        _need(len(variants) == 1, "each band must have one visible exceptional map")
        col, panel = variants[0]
        points = set(map(tuple, np.argwhere(panel != master)))
        center, direction = _read_t(points)
        tip, targetcell = _step(center, direction), _step(center, direction, 2)
        _need(_inside(targetcell, master.shape), "T points outside the map")
        source, target = int(ids[center]), int(ids[targetcell])
        _need(ids[tip] == source and source != target,
              "center and tip must share a source and point into a different territory")
        payload = colors[source]
        _need(target not in assignments or assignments[target] == payload,
              "conflicting incoming commands")
        assignments[target] = payload
        commands.append(dict(center=center, direction=direction, points=points,
                             source=source, target=target, payload=payload))
        columns.append(col)
        balanced.append(Counter(map(int, master.ravel())) == Counter(map(int, panel.ravel())))
        crossing.append(len({int(ids[p]) for p in points}) > 1)
    painted = master.copy()
    for target, payload in assignments.items():
        painted[ids == target] = payload
    final = [assignments.get(i, color) for i, color in enumerate(colors)]
    info = dict(lattice=lattice, rows=rows, master=master, ids=ids, colors=colors, cells=cells,
                commands=commands, assignments=assignments, columns=columns,
                balanced=balanced, crossing=crossing, painted=painted,
                locality=any(colors[i] == colors[j] and final[i] != final[j]
                             for i in range(len(colors)) for j in range(i)),
                chain=any(x['target'] == y['source'] and x['payload'] != y['payload']
                          and x['source'] != y['target'] for x in commands for y in commands),
                cycle=any(x['target'] == y['source'] and x['source'] == y['target']
                          for x in commands for y in commands))
    return _assemble([[painted]], lattice), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the master, decode difference shapes, and recolor original components."""
    return _analyze(input_grid)[0]


def _palette(rng, shared):
    return shared if shared is not None else tuple(map(int, rng.permutation(10)))


def _candidate_commands(labels):
    result = []
    for center in np.ndindex(labels.shape):
        for d in DIRECTIONS:
            points, target = _stencil(center, d), _step(center, d, 2)
            if not all(_inside(p, labels.shape) for p in points | {target}):
                continue
            source = int(labels[center])
            if labels[_step(center, d)] == source and labels[target] != source:
                result.append(dict(center=center, direction=d, points=points,
                                   source=source, target=int(labels[target])))
    return result


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        for d in DIRECTIONS:
            v = _step(stack[-1], d)
            if v in points:
                points.remove(v)
                stack.append(v)
                break
        else:
            stack.pop()
    return not points


def _partition(rng, shape, count, style):
    h, w = shape
    labels = np.zeros(shape, dtype=np.int16)
    rr, cc = np.indices(shape)
    for n in range(1, count):
        for _ in range(45):
            sizes = np.bincount(labels.ravel(), minlength=n)
            host = int(rng.choice(n, p=sizes/sizes.sum()))
            mask = labels == host
            r, c = np.nonzero(mask)
            r0, r1, c0, c1 = int(r.min()), int(r.max()+1), int(c.min()), int(c.max()+1)
            if style == 2 and r1-r0 >= 5 and c1-c0 >= 5:
                top = int(rng.integers(r0+1, r1-2))
                left = int(rng.integers(c0+1, c1-2))
                bottom = int(rng.integers(top+1, r1))
                right = int(rng.integers(left+1, c1))
                section = (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
            else:
                axis = int(rng.integers(2))
                lo, hi = (r0, r1) if axis == 0 else (c0, c1)
                if hi-lo < 2:
                    continue
                cut = int(rng.integers(lo+1, hi))
                along = w if axis == 0 else h
                path = np.full(along, cut)
                if style == 1:
                    k = int(rng.integers(1, along))
                    path[k:] = int(np.clip(cut+int(rng.choice([-1, 1])), lo+1, hi-1))
                section = rr < path[cc] if axis == 0 else cc < path[rr]
            left_mask, right_mask = mask & section, mask & ~section
            if (left_mask.sum() >= 2 and right_mask.sum() >= 2
                    and _connected(left_mask) and _connected(right_mask)):
                labels[left_mask] = n
                break
        else:
            return None
    return labels


def _color_regions(rng, labels, palette):
    count = int(labels.max())+1
    colors = list(palette[1:1+count])
    # Reuse fills only between nonadjacent original components.
    if count >= 3 and rng.random() < .65:
        neighbors = set()
        for x, y in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
            neighbors.update((int(a), int(b)) for a, b in zip(x.ravel(), y.ravel()) if a != b)
        options = [(i, j) for i in range(count) for j in range(i)
                   if (i, j) not in neighbors and (j, i) not in neighbors]
        if options:
            i, j = options[int(rng.integers(len(options)))]
            colors[i] = colors[j]
    return np.array(colors, dtype=np.int8)


def _render_pair(rng, labels, colors, masks, palette, copies, force_balanced=False, teach_columns=False):
    master = colors[labels]
    lattice = palette[0]
    rows, assignments = [], {}
    columns = rng.integers(copies, size=len(masks))
    if teach_columns and len(set(map(int, columns))) == 1:
        columns[-1] = (columns[-1]+1) % copies
    for b, points in enumerate(masks):
        center, d = _read_t(points)
        source, target = int(labels[center]), int(labels[_step(center, d, 2)])
        payload = int(colors[source])
        _need(target not in assignments or assignments[target] == payload, "constructor conflict")
        assignments[target] = payload
        variant = master.copy()
        original = [int(master[p]) for p in sorted(points)]
        histogram = Counter(original)
        balanced = (force_balanced and b == 0) or (len(histogram) > 1 and max(histogram.values()) <= 2 and rng.random() < .55)
        if balanced:
            _need(max(histogram.values()) <= 2, "balanced stencil cannot be permuted")
            ordered = sorted(points, key=lambda p: (int(master[p]), p))
            values = [int(master[p]) for p in ordered]
            shift = max(histogram.values())
            for i, p in enumerate(ordered):
                variant[p] = values[(i+shift) % 4]
        else:
            # No dedicated arrow ink: changes can camouflage as neighboring land.
            target_color = int(colors[target])
            uniform = next(c for c in palette[1:] if c not in histogram)
            for p in sorted(points):
                choices = [c for c in palette[1:] if c != int(master[p])]
                if rng.random() < .65 and target_color != master[p]:
                    variant[p] = target_color
                else:
                    variant[p] = uniform if b % 2 else int(rng.choice(choices))
        _need(np.count_nonzero(variant != master) == 4, "not all four T cells changed")
        row = [master.copy() for _ in range(copies)]
        row[int(columns[b])] = variant
        rows.append(row)
    painted = master.copy()
    for target, payload in assignments.items():
        painted[labels == target] = payload
    return dict(input=_assemble(rows, lattice), output=_assemble([[painted]], lattice))


def _witness(rng, kind, palette, turn):
    h, w = int(rng.integers(6, 9)), int(rng.integers(6, 9))
    if kind == 0:
        top = int(rng.integers(2, h-2))
        cut = int(rng.integers(2, w-1))
        labels = np.zeros((h, w), dtype=np.int16)
        labels[top:, :cut], labels[top:, cut:] = 1, 2
        commands = [((top-2, 1), (1, 0)), ((top+1, cut-2), (0, 1))]
        colors = np.array(palette[1:4], dtype=np.int8)
    else:
        labels = np.zeros((h, w), dtype=np.int16)
        r, c = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
        labels[r:r+2, c] = 1
        # A separate same-colored component remains untouched.
        labels[:int(rng.integers(1, 3)), :int(rng.integers(1, 3))] = 2
        commands = [((r+1, c), (-1, 0)), ((r, c+2), (0, -1))]
        colors = np.array([palette[1], palette[2], palette[2]], dtype=np.int8)
    masks = []
    for center, d in commands:
        mask = np.zeros(labels.shape, dtype=bool)
        for p in _stencil(center, d):
            mask[p] = True
        masks.append(set(map(tuple, np.argwhere(np.rot90(mask, turn)))))
    labels = np.rot90(labels, turn).copy()
    return _render_pair(rng, labels, colors, masks, palette, 3, kind == 1, True)


def _general_pair(rng, palette, test):
    # Test extends either map height or command count; no new semantics.
    expanded = bool(rng.integers(2)) if test else False
    for _ in range(100):
        if test and expanded:
            h, bands = int(rng.integers(9, 14)), 2
        elif test:
            h, bands = int(rng.integers(4, 7)), 4
        else:
            h, bands = int(rng.integers(5, 9)), int(rng.integers(2, 4))
        w = int(rng.integers(4, 9))
        copies = int(rng.integers(3, min(5, 29//(w+1))+1))
        count = int(rng.integers(2, 6))
        labels = _partition(rng, (h, w), count, int(rng.integers(3)))
        if labels is None:
            continue
        colors = _color_regions(rng, labels, palette)
        candidates = _candidate_commands(labels)
        chosen, assignments = [], {}
        for i in rng.permutation(len(candidates)):
            cmd = candidates[int(i)]
            target, payload = cmd['target'], int(colors[cmd['source']])
            if target in assignments and (assignments[target] != payload or rng.random() < .75):
                continue
            if any(cmd['points'] == old['points'] for old in chosen):
                continue
            assignments[target] = payload
            chosen.append(cmd)
            if len(chosen) == bands:
                break
        if len(chosen) != bands or len(assignments) < 2:
            continue
        rng.shuffle(chosen)
        return _render_pair(rng, labels, colors, [x['points'] for x in chosen], palette, copies)
    # Guaranteed finite constructive fallback: four commands on a three-stripe map.
    h, w = (11, 8) if test and expanded else ((5, 8) if test else (7, 8))
    bands = 2 if test and expanded else (4 if test else 3)
    labels = np.zeros((h, w), dtype=np.int16)
    labels[:, 2:5], labels[:, 5:] = 1, 2
    commands = [((1, 0), (0, 1)), ((h-2, 3), (0, 1)),
                ((1, 6), (0, -1)), ((h-2, 6), (0, -1))][:bands]
    # Both sources assigning the middle stripe must agree.
    colors = np.array([palette[1], palette[2], palette[1]], dtype=np.int8)
    return _render_pair(rng, labels, colors, [_stencil(p, d) for p, d in commands], palette, 3)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(name+' outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shared = tuple(map(int, rng.permutation(10))) if rng.random() < .25 else None
    turn = int(rng.integers(4))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            for _ in range(20):
                palette = _palette(rng, shared)
                pair = (_witness(rng, i, palette, turn) if split == 'train' and i < 2
                        else _general_pair(rng, palette, split == 'test'))
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded duplicate avoidance exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal pair counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            predicted, info = _analyze(a)
            _need(np.array_equal(predicted, out), 'incorrect recoloring or output framing')
            _need(not np.array_equal(info['master'], info['painted']), 'no territory changes')
            # Output has one map, not a stack of exception panels.
            _need(out.shape == tuple(n+2 for n in info['master'].shape), 'wrong output dimensions')
            _need(all(out[p] == info['lattice'] for p in
                      [(0, c) for c in range(out.shape[1])] + [(out.shape[0]-1, c) for c in range(out.shape[1])]
                      + [(r, 0) for r in range(out.shape[0])] + [(r, out.shape[1]-1) for r in range(out.shape[0])]),
                  'broken output border')
            features[split].append(info)
    train = features['train']
    _need(any(f['chain'] for f in train), 'missing original-state chain demonstration')
    _need(any(f['locality'] for f in train), 'missing component-local demonstration')
    _need(any(any(f['balanced']) for f in train), 'missing count-preserving exception')
    _need(any(any(f['crossing']) for f in train), 'missing boundary-crossing difference stencil')
    directions = {x['direction'] for f in train for x in f['commands']}
    _need(directions == set(DIRECTIONS), 'training must expose all arrow directions')
    # In each test either the master is taller or there are more commands.
    max_h = max(f['master'].shape[0] for f in train)
    max_bands = max(len(f['rows']) for f in train)
    _need(all(f['master'].shape[0] > max_h or len(f['rows']) > max_bands
              for f in features['test']), 'missing train/test structural shift')


def self_test():
    """42 seeds, streaming feature sets, mutation tests, and count boundaries."""
    puzzles, shapes, maps, palettes, layouts, sizes, counts, balanced = set(), set(), set(), set(), set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        cascade_witness = False
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                digest.update(np.asarray(pair['input'], dtype=np.int8).tobytes())
                out, f = _analyze(pair['input'])
                assert np.array_equal(out, pair['output'])
                shapes.add(tuple(pair['input'].shape))
                normalized = {}
                token = tuple(normalized.setdefault(int(c), len(normalized)) for c in f['master'].ravel())
                maps.add((f['master'].shape, token))
                layouts.add(tuple(f['columns']))
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                sizes.update(map(len, f['cells']))
                counts.add((len(f['cells']), len(f['rows']), len(f['rows'][0])))
                balanced += sum(f['balanced'])
                assert not np.array_equal(f['master'], f['painted']), 'omitting territory transfer has no effect'
                wrong_source, cascaded = f['master'].copy(), f['master'].copy()
                for b, cmd in enumerate(f['commands']):
                    # These counterfactuals retain correct territory masks to
                    # isolate original-map lookup and simultaneous timing.
                    modified = f['rows'][b][f['columns'][b]]
                    wrong_source[f['ids'] == cmd['target']] = modified[cmd['center']]
                    cascaded[f['ids'] == cmd['target']] = cascaded[cmd['center']]
                assert not np.array_equal(wrong_source, f['painted']), 'modified-color shortcut survived'
                if split == 'train' and not np.array_equal(cascaded, f['painted']):
                    cascade_witness = True
        assert cascade_witness, 'missing training contrast with cascading'
        puzzles.add(digest.digest())
    assert len(puzzles) == 42 and len(shapes) >= 25 and len(maps) >= 100
    assert len(layouts) >= 25 and len(palettes) >= 30 and len(sizes) >= 20 and len(counts) >= 15
    assert balanced >= 42
    p, q = generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    base = generate_puzzle(0)
    for defect in ('output', 'missing', 'delta', 'border'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad['train'][0]
        _, f = _analyze(pair['input'])
        if defect == 'output':
            pair['output'][1, 1] = (int(pair['output'][1, 1])+1) % 10
        elif defect == 'border':
            pair['input'][0, 0] = (int(pair['input'][0, 0])+1) % 10
        else:
            ph, pw = f['master'].shape
            c = f['columns'][0]
            panel = pair['input'][1:1+ph, 1+c*(pw+1):1+c*(pw+1)+pw]
            if defect == 'missing':
                panel[:] = f['master']
            else:
                point = next(iter(f['commands'][0]['points']))
                panel[point] = f['master'][point]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('validator accepted '+defect)
    # Formal-family boundary members outside the constructor's usual ranges:
    # minimal maps, seven bands/copies, 13-high maps, repeated exact exceptions.
    for h, w, bands, copies in ((3, 3, 2, 3), (3, 3, 7, 7), (13, 3, 2, 3)):
        master = np.full((h, w), 2, dtype=np.int8)
        master[:, -1] = 4
        points = _stencil((1, w-3), (0, 1))
        variant = master.copy()
        for point in points:
            variant[point] = 4
        rows = [[master.copy() for _ in range(copies)] for _ in range(bands)]
        for row in rows:
            row[-1] = variant.copy()
        inp = _assemble(rows, 0)
        expected = _assemble([[np.full((h, w), 2, dtype=np.int8)]], 0)
        assert np.array_equal(solve(inp), expected)
        # A palette permutation changes roles without changing interpretation.
        lookup = np.array([7, 6, 0, 9, 1, 2, 8, 4, 3, 5], dtype=np.int8)
        assert np.array_equal(solve(lookup[inp]), lookup[expected])
    master = np.array([[2, 2, 4, 4, 3, 3]]*3, dtype=np.int8)
    rows = []
    for center, d in (((1, 0), (0, 1)), ((1, 5), (0, -1))):
        variant = master.copy()
        for point in _stencil(center, d):
            variant[point] = 9
        rows.append([master.copy(), variant, master.copy()])
    try:
        solve(_assemble(rows, 0))
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting destination colors were accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, and malformed-puzzle checks')
