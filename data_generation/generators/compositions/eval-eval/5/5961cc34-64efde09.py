"""Broadcast closure from an input-specified, template-relative ray code.

No files, parent generators, stored examples, or global random state are used.
Construction walks a cell-owner array; solve projects onto decoded rectangles.
"""
from collections import deque
import hashlib

import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(a, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bar(template, transform, r, c):
    a = _orient(template, transform)
    labels = _orient(np.arange(template.size).reshape(template.shape), transform)
    cells = {(r + y, c + x) for y in range(a.shape[0]) for x in range(a.shape[1])}
    return {'arr': a, 'labels': labels, 'r': r, 'c': c, 'cells': cells,
            'transform': transform}


def _origin(bar, side, slot, length):
    p = np.argwhere(bar['labels'] == side * length + slot)[0]
    q = np.argwhere(bar['labels'] == (1 - side) * length + slot)[0]
    direction = tuple(map(int, p - q))
    return (bar['r'] + int(p[0]), bar['c'] + int(p[1])), direction


def _walk(origin, direction, shape, owner):
    """Construction collision implementation: advance one pixel at a time."""
    r, c = origin
    dr, dc = direction
    path = []
    r, c = r + dr, c + dc
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        if owner[r, c] >= 0:
            return path, int(owner[r, c])
        path.append((r, c))
        r, c = r + dr, c + dc
    return path, -1


def _project(origin, direction, shape, bars):
    """Solver collision implementation: choose the nearest rectangle face."""
    r, c = origin
    dr, dc = direction
    limit = r + 1 if dr < 0 else shape[0] - r if dr > 0 else c + 1 if dc < 0 else shape[1] - c
    distance, hit = limit, -1
    for index, b in enumerate(bars):
        y, x = b['r'], b['c']
        h, w = b['arr'].shape
        k = None
        if dr and x <= c < x + w:
            k = y - r if dr > 0 else r - (y + h - 1)
        elif dc and y <= r < y + h:
            k = x - c if dc > 0 else c - (x + w - 1)
        if k is not None and 0 < k < distance:
            distance, hit = k, index
    return [(r + k * dr, c + k * dc) for k in range(1, distance)], hit


def _walk_geometry(shape, bars, side, code):
    """One candidate geometry is reused across different launcher choices."""
    owner = np.full(shape, -1, dtype=np.int8)
    for i, b in enumerate(bars):
        for p in b['cells']:
            owner[p] = i
    length, graph = max(bars[0]['arr'].shape), []
    for b in bars:
        emissions = []
        for slot, color in code.items():
            p, direction = _origin(b, side, slot, length)
            cells, hit = _walk(p, direction, shape, owner)
            emissions.append((slot, color, cells, hit))
        graph.append(emissions)
    return owner, graph


def _render(shape, bg, signal, launcher, bars, side, code, *, walk=False,
            activate_all=False, local_codes=None, precomputed=None):
    length = max(bars[0]['arr'].shape)
    if walk:
        assert local_codes is None
        owner, graph = _walk_geometry(shape, bars, side, code) if precomputed is None else precomputed

    def trace(p, direction):
        return _walk(p, direction, shape, owner) if walk else _project(p, direction, shape, bars)

    path, root = trace(launcher['line'][-1], launcher['direction'])
    energy = set(launcher['line']) | set(path)
    masks = {}
    if not walk:
        graph = []
        for i, b in enumerate(bars):
            emissions = []
            for slot, color in (code if local_codes is None else local_codes[i]).items():
                p, direction = _origin(b, side, slot, length)
                cells, hit = trace(p, direction)
                emissions.append((slot, color, cells, hit))
            graph.append(emissions)
    queue = deque((i, 0) for i in range(len(bars))) if activate_all else deque([] if root < 0 else [(root, 0)])
    depths = {}
    while queue:
        i, depth = queue.popleft()
        if i in depths:
            continue
        depths[i] = depth
        energy.update(bars[i]['cells'])
        for slot, color, cells, hit in graph[i]:
            for p in cells:
                masks[p] = masks.get(p, 0) | (1 << color)
            if hit >= 0 and hit not in depths:
                queue.append((hit, depth + 1))
    out = np.full(shape, bg, dtype=np.int8)
    mixed = mixed_total = 0
    for p, mask in masks.items():
        if mask.bit_count() > 1:
            out[p] = signal
            mixed_total += 1
            mixed += int(p not in energy)
        else:
            out[p] = mask.bit_length() - 1
    energy_overlaps = len(energy.intersection(masks))
    for p in energy:
        out[p] = signal
    targets = [{e[3] for e in graph[i] if e[3] >= 0} for i in depths]
    facts = {'active': tuple(sorted(depths)), 'depth': max(depths.values(), default=-1),
             'branch': max(map(len, targets), default=0), 'mixed': mixed,
             'mixed_total': mixed_total,
             'energy_overlaps': energy_overlaps, 'root': root,
             'edges': tuple((i, slot, hit) for i in sorted(depths)
                            for slot, _, _, hit in graph[i] if hit >= 0)}
    return out, facts


def _decode(inp):
    values, counts = np.unique(inp, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    assert counts.max() > inp.size // 2
    h, w = inp.shape
    assert all(inp[p] == bg for p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    bars, hints, launchers = [], [], []
    components = _components(inp, bg)
    for cells in components:
        boundary = [p for p in cells if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
        if len(cells) == 1:
            assert len(boundary) == 1
            p = next(iter(cells))
            hints.append((p, int(inp[p])))
        elif boundary:
            assert len(boundary) == 1 and 3 <= len(cells) <= 5
            r, c = boundary[0]
            direction = (1, 0) if r == 0 else (-1, 0) if r == h - 1 else (0, 1) if c == 0 else (0, -1)
            line = [(r + k * direction[0], c + k * direction[1]) for k in range(len(cells))]
            assert set(line) == cells
            signal, cap = int(inp[line[0]]), int(inp[line[-1]])
            assert signal != cap and all(int(inp[p]) == signal for p in line[:-1])
            launchers.append({'line': line, 'direction': direction, 'signal': signal, 'cap': cap})
        else:
            r, c = min(y for y, x in cells), min(x for y, x in cells)
            a = inp[r:max(y for y, x in cells) + 1, c:max(x for y, x in cells) + 1].copy()
            assert a.size == len(cells) and min(a.shape) == 2 and 4 <= max(a.shape) <= 10
            bars.append({'r': r, 'c': c, 'arr': a, 'cells': cells})
    assert len(launchers) == 1 and 2 <= len(bars) <= 8 and hints
    launcher = launchers[0]
    template = bars[0]['arr']
    if template.shape[0] != 2:
        template = np.rot90(template)
    length = template.shape[1]
    forms = [_orient(template, t) for t in range(8)]
    assert len({_key(a) for a in forms}) == 8
    structural = set(map(int, np.unique(template)))
    assert 2 <= len(structural) <= 3
    assert not structural.intersection({bg, launcher['signal'], launcher['cap']})
    assert len({bg, launcher['signal'], launcher['cap']}) == 3
    labels = np.arange(2 * length).reshape(2, length)
    for b in bars:
        ts = [t for t, a in enumerate(forms) if np.array_equal(a, b['arr'])]
        assert len(ts) == 1
        b['transform'] = ts[0]
        b['labels'] = _orient(labels, ts[0])
    # Every pair of foreground components has an eight-neighbor gap.
    owners = {p: i for i, cells in enumerate(components) for p in cells}
    for p, i in owners.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                assert owners.get((p[0] + dr, p[1] + dc), i) == i
    corridors = {}
    for i, b in enumerate(bars):
        for side in (0, 1):
            for slot in range(length):
                p, direction = _origin(b, side, slot, length)
                ray, hit = _project(p, direction, inp.shape, bars)
                if hit < 0 and ray and all(inp[p] == bg for p in ray[:-1]):
                    corridors.setdefault(ray[-1], []).append((i, side, slot))
    code, sides, observations = {}, set(), []
    for p, color in hints:
        assert color not in structural | {bg, launcher['signal'], launcher['cap']}
        matches = corridors.get(p, [])
        assert len(matches) == 1, 'ambiguous or blocked boundary observation'
        i, side, slot = matches[0]
        assert slot not in code or code[slot] == color, 'inconsistent shared code'
        code[slot] = color
        sides.add(side)
        observations.append((i, slot, color))
    assert len(sides) == 1 and 1 <= len(code) <= 5
    side = next(iter(sides))
    return bg, launcher, bars, side, code, observations


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    bg, launcher, bars, side, code, _ = _decode(inp)
    return _render(inp.shape, bg, launcher['signal'], launcher, bars, side, code)[0]


def _analyze(inp, out=None):
    bg, launcher, bars, side, code, observations = _decode(inp)
    predicted, f = _render(inp.shape, bg, launcher['signal'], launcher, bars, side, code)
    if out is not None:
        assert np.array_equal(predicted, out), 'incorrect broadcast closure'
    local = [dict() for _ in bars]
    for i, slot, color in observations:
        local[i][slot] = color
    all_out = _render(inp.shape, bg, launcher['signal'], launcher, bars, side, code, activate_all=True)[0]
    local_out = _render(inp.shape, bg, launcher['signal'], launcher, bars, side, code, local_codes=local)[0]
    active = set(f['active'])
    sources = {slot: {i for i, s, _ in observations if s == slot} for slot in code}
    directions = tuple(_origin(b, side, next(iter(code)), max(b['arr'].shape))[1] for b in bars)
    active_transforms = {bars[i]['transform'] >= 4 for i in active}
    active_axes = {directions[i][0] == 0 for i in active}
    f.update(shape=inp.shape, length=max(bars[0]['arr'].shape), slots=tuple(sorted(code)),
             code=tuple(sorted(code.items())), count=len(bars), decoys=len(bars) - len(active),
             allocations=tuple(len(x) for x in local), observations=len(observations),
             pooled=all(len(x) < len(code) for x in local),
             remote=any(not ids.intersection(active) for ids in sources.values()),
             without_A_diff=not np.array_equal(all_out, predicted),
             without_B_diff=not np.array_equal(local_out, predicted),
             directions=directions, transforms=tuple(b['transform'] for b in bars),
             active_reflection_classes=len(active_transforms), active_axes=len(active_axes),
             asymmetric_code=code != {max(bars[0]['arr'].shape) - 1 - s: c for s, c in code.items()},
             palette=(bg, launcher['signal'], launcher['cap'], tuple(sorted(set(map(int, np.unique(bars[0]['arr'])))))),
             adjacent=any(b - a == 1 for a, b in zip(sorted(code), sorted(code)[1:])),
             irregular=len({b - a for a, b in zip(sorted(code), sorted(code)[1:])}) > 1,
             repeated_color=len(set(code.values())) < len(code))
    return f


def _template(rng, length, colors):
    for _ in range(32):
        a = rng.choice(colors, (2, length)).astype(np.int8)
        if rng.random() < .6:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, int(rng.integers(length)):] = colors[-1]
            a[0, -1] = colors[1]
        for j, color in enumerate(colors):
            a[1, j] = color
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    for j, color in enumerate(colors[2:], 2):
        a[1, j] = color
    return a


def _separated(cells, occupied):
    return all((r + dr, c + dc) not in occupied for r, c in cells
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _place(rng, template, count, shape):
    bars, occupied = [], set()
    for _ in range(count):
        for _ in range(70):
            t = int(rng.integers(8))
            h, w = _orient(template, t).shape
            r = int(rng.integers(1, shape[0] - h))
            c = int(rng.integers(1, shape[1] - w))
            b = _bar(template, t, r, c)
            if _separated(b['cells'], occupied):
                bars.append(b)
                occupied.update(b['cells'])
                break
        else:
            return None
    return bars


def _launcher(rng, shape, bars):
    direction = _DIRS[int(rng.integers(4))]
    h, w = shape
    k = int(rng.integers(2, (w if direction[0] else h) - 2))
    start = (h - 1, k) if direction[0] < 0 else (0, k) if direction[0] > 0 else (k, 0) if direction[1] > 0 else (k, w - 1)
    length = int(rng.integers(3, 6))
    line = [(start[0] + j * direction[0], start[1] + j * direction[1]) for j in range(length)]
    occupied = set().union(*(b['cells'] for b in bars))
    if not _separated(set(line), occupied):
        return None
    return {'line': line, 'direction': direction}


def _clues(rng, shape, bars, side, code, launcher, active, require_remote):
    """Select uniquely attributable endpoints, without hiding any code slot."""
    occupied = set(launcher['line']) | set().union(*(b['cells'] for b in bars))
    length = max(bars[0]['arr'].shape)
    endpoints = {}
    for i, b in enumerate(bars):
        for s in (0, 1):
            for slot in range(length):
                p, direction = _origin(b, s, slot, length)
                ray, hit = _project(p, direction, shape, bars)
                if hit < 0 and len(ray) >= 2 and not occupied.intersection(ray):
                    endpoints.setdefault(ray[-1], []).append((i, s, slot))
    choices = {slot: [] for slot in code}
    for p, matches in endpoints.items():
        if len(matches) == 1:
            i, s, slot = matches[0]
            if s == side and slot in choices and _separated({p}, occupied):
                choices[slot].append((i, slot, p))
    if any(not options for options in choices.values()):
        return None
    slots = list(code)
    rng.shuffle(slots)
    for _ in range(24):
        result = []
        for slot in slots:
            options = [x for x in choices[slot] if _separated({x[2]}, {v[2] for v in result})]
            if not options:
                break
            result.append(options[int(rng.integers(len(options)))])
        if len(result) != len(slots):
            continue
        allocation = [sum(x[0] == i for x in result) for i in range(len(bars))]
        if len(slots) > 1 and max(allocation) == len(slots):
            continue
        if require_remote and not any(i not in active for i, _, _ in result):
            continue
        # Optional redundant evidence, agreeing in local coordinates.
        if not require_remote and rng.random() < .25:
            extras = [x for options in choices.values() for x in options]
            rng.shuffle(extras)
            for x in extras:
                if _separated({x[2]}, {v[2] for v in result}):
                    result.append(x)
                    break
        return result
    return None


def _fits(f, mode):
    if f['root'] < 0 or len(f['active']) < 3:
        return False
    if mode == 'serial':
        return f['depth'] >= 2 and f['branch'] <= 1
    if mode == 'branch':
        return f['branch'] >= 2 and f['depth'] == 1 and f['mixed'] > 0
    return len(f['active']) >= 4 and f['branch'] >= 2 and f['depth'] >= 2


def _finish(shape, palette, template, bars, launcher, side, code, clues):
    bg, signal, cap = palette[:3]
    inp = np.full(shape, bg, dtype=np.int8)
    for b in bars:
        r, c = b['r'], b['c']
        h, w = b['arr'].shape
        inp[r:r + h, c:c + w] = b['arr']
    for p in launcher['line'][:-1]:
        inp[p] = signal
    inp[launcher['line'][-1]] = cap
    for i, slot, p in clues:
        inp[p] = code[slot]
    out = _render(shape, bg, signal, launcher, bars, side, code, walk=True)[0]
    return {'input': inp, 'output': out}


def _fallback(rng, mode):
    """Guaranteed packing retains serial/branch/combined obligations.

    Used only after bounded free placement. Random palette, template, whole-scene
    dihedral transform still vary; it does not silently simplify.
    """
    palette = list(map(int, rng.permutation(10)))
    template = _template(rng, 5, palette[3:6])
    bars = [_bar(template, 0, 21, 4), _bar(template, 7, 11, 4)]
    if mode != 'serial':
        bars.append(_bar(template, 3, 5, 8))
    if mode != 'branch':
        bars.append(_bar(template, 0, 14, 17))
    bars.append(_bar(template, 0, 23, 18))
    launcher = {'line': [(27, 5), (26, 5), (25, 5)], 'direction': (-1, 0)}
    slots = (0, 2, 4) if mode == 'branch' else (0, 4)
    code = dict(zip(slots, palette[6:6 + len(slots)]))
    f = _render((28, 28), palette[0], palette[1], launcher, bars, 0, code, walk=True)[1]
    assert _fits(f, mode)
    clues = _clues(rng, (28, 28), bars, 0, code, launcher, set(f['active']), True)
    assert clues is not None
    pair = _finish((28, 28), palette, template, bars, launcher, 0, code, clues)
    t = int(rng.integers(8))
    return {k: _orient(v, t) for k, v in pair.items()}


def _generate_pair(rng, mode, slots_count):
    # A fixed node/topology request is never replaced by an easier topology.
    palette = list(map(int, rng.permutation(10)))
    structural_count = int(rng.integers(2, 4))
    for _ in range(450):
        length = int(rng.integers(max(4, slots_count), 9))
        template = _template(rng, length, palette[3:3 + structural_count])
        slots = tuple(sorted(map(int, rng.choice(length, slots_count, replace=False))))
        signal_colors = palette[3 + structural_count:3 + structural_count + min(3, slots_count)]
        colors = list(map(int, rng.choice(signal_colors, slots_count)))
        if len(set(colors)) == 1 and slots_count > 1:
            colors[-1] = signal_colors[1]
        code = dict(zip(slots, colors))
        shape = tuple(map(int, rng.integers(20, 31, size=2)))
        count = int(rng.integers(5 if mode == 'combined' else 4, 8))
        bars = _place(rng, template, count, shape)
        if bars is None:
            continue
        side = int(rng.integers(2))
        geometry = _walk_geometry(shape, bars, side, code)
        # Reuse one geometry for a bounded set of different boundary entries.
        for _ in range(12):
            launcher = _launcher(rng, shape, bars)
            if launcher is None:
                continue
            out, f = _render(shape, palette[0], palette[1], launcher, bars, side, code, walk=True, precomputed=geometry)
            if not _fits(f, mode) or len(f['active']) == count:
                continue
            if len({_origin(bars[i], side, slots[0], length)[1] for i in f['active']}) < 2:
                continue
            if len({bars[i]['transform'] >= 4 for i in f['active']}) < 2:
                continue
            if len({_origin(bars[i], side, slots[0], length)[1][0] == 0 for i in f['active']}) < 2:
                continue
            if code == {length - 1 - s: c for s, c in code.items()}:
                continue
            clues = _clues(rng, shape, bars, side, code, launcher, set(f['active']), True)
            if clues is None:
                continue
            pair = _finish(shape, palette, template, bars, launcher, side, code, clues)
            assert np.array_equal(out, pair['output'])
            return pair
    return _fallback(rng, mode)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            mode = 'combined' if split == 'test' else 'serial' if index == 0 else 'branch' if index == 1 else str(rng.choice(['serial', 'branch']))
            slots = 2 if split == 'train' and index == 0 else 3 if split == 'train' and index == 1 else int(rng.integers(2, 4))
            for _ in range(8):
                pair = _generate_pair(rng, mode, slots)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input construction failed')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    facts, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _analyze(inp, out)
            assert f['without_A_diff'] and f['without_B_diff']
            assert f['remote'] and f['decoys'] >= 1
            assert len(f['active']) >= 3 and not np.array_equal(inp, out)
            assert f['active_reflection_classes'] == 2 and f['active_axes'] == 2
            assert f['asymmetric_code']
            facts[split].append(f)
    train = facts['train']
    assert any(f['depth'] >= 2 and f['branch'] <= 1 for f in train), 'missing serial evidence'
    assert any(f['branch'] >= 2 and f['depth'] == 1 and f['mixed'] > 0 for f in train), 'missing branch/crossing evidence'
    assert all(f['pooled'] for f in train)
    assert any(len(f['slots']) == 2 for f in train) and any(len(f['slots']) == 3 for f in train)
    assert all(f['depth'] >= 2 and f['branch'] >= 2 and len(f['active']) >= 4 for f in facts['test'])


def self_test():
    """42 seeds, streamed measurements, metamorphic and ablation checks; no I/O."""
    fingerprints, shapes, lengths, slots, counts, directions, structures = set(), set(), set(), set(), set(), set(), set()
    remote = mixed = reflected = adjacent = repeated = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                f = _analyze(inp, out)
                assert f['without_A_diff'] and f['without_B_diff']
                digest.update(repr(_key(inp)).encode())
                shapes.add(inp.shape)
                lengths.add(f['length'])
                slots.add((f['length'], f['slots']))
                counts.add(f['count'])
                directions.update(f['directions'])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(normalized).digest())
                remote += f['remote']
                mixed += f['mixed'] > 0
                reflected += any(t >= 4 for t in f['transforms'])
                adjacent += f['adjacent']
                repeated += f['repeated_color']
        fingerprints.add(digest.digest())
        # Every whole-scene dihedral mapping and color permutation preserves semantics.
        pair = puzzle['test'][0]
        t = seed % 8
        permutation = np.random.default_rng(10000 + seed).permutation(10)
        transformed_in = _orient(permutation[pair['input']], t)
        transformed_out = _orient(permutation[pair['output']], t)
        assert np.array_equal(solve(transformed_in), transformed_out)
        assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert len(fingerprints) == 42 and len(structures) >= 150
    assert len(shapes) >= 30 and len(slots) >= 30 and len(lengths) >= 4
    assert len(counts) >= 3 and directions == set(_DIRS)
    assert remote == 210 and mixed >= 42 and reflected >= 100 and adjacent >= 20 and repeated >= 10
    # Exercise the guaranteed packing explicitly, rather than hoping random search uses it.
    rng = np.random.default_rng(90001)
    for mode in ('serial', 'branch', 'combined'):
        pair = _fallback(rng, mode)
        f = _analyze(pair['input'], pair['output'])
        assert _fits(f, mode) and f['remote']
    # Solver coverage beyond the teaching sampler: L=10, five and one slots,
    # a complete code on one copy, redundant observations, two bars, misses,
    # and an all-reached network. These are rule checks, not sample diversity.
    palette = list(range(10))
    template = _template(rng, 10, [3, 4, 5])
    for slots in ((3,), (0, 2, 4, 6, 8)):
        code = {s: (6, 7, 6, 8, 9)[j] for j, s in enumerate(slots)}
        for mode in ('remote', 'redundant', 'miss', 'all_reached'):
            c = 2 if mode == 'all_reached' else 15
            bars = [_bar(template, 0, 5, 2), _bar(template, 4, 16, c)]
            col = 13 if mode == 'miss' else c + 3
            launcher = {'line': [(29, col), (28, col), (27, col)], 'direction': (-1, 0)}
            clues = [(0, s, (0, 2 + s)) for s in slots]
            if mode == 'redundant':
                s = slots[0]
                clues.append((1, s, (0, c + 9 - s)))
            pair = _finish((30, 30), palette, template, bars, launcher, 0, code, clues)
            assert np.array_equal(solve(pair['input']), pair['output'])
            f = _analyze(pair['input'], pair['output'])
            if mode == 'miss':
                assert not f['active']
            elif mode == 'all_reached':
                assert len(f['active']) == 2
            else:
                assert len(f['active']) == 1
    original = generate_puzzle(12345, 2, 1)
    again = generate_puzzle(12345, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in original for a, b in zip(original[s], again[s]) for k in a)
    original['test'][0]['output'][0, 0] = (int(original['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(original)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
