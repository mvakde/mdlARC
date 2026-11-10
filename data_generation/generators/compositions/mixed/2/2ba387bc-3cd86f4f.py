"""Rank-pair square tiles, then shear the assembled queue by pixel row.

Only NumPy and the standard library are used. Generation has no file I/O;
solve reads only the supplied grid. All randomness comes from one seed RNG.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


HOLLOW, SOLID = 0, 1
HOST_SEEDS = (
    0, 622000750, 781875135, 788158280, 594235582, 426700480,
    146380468, 565122547, 771989192, 957840930, 165334138,
    93773226, 254514664, 201310589, 425461644, 489251101,
    754781818, 671176144, 206158007, 530429413, 899583603,
    806799953, 152998460, 264329478, 648814820, 419763502,
    77710421, 89047125, 414687577, 500379988, 936055536,
    400256392, 344513298, 377759830, 912842309, 367752223,
    8396158, 115999328, 650538439, 308583487, 960125624,
    594997903,
)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _tile(s, color, bg, kind):
    a = np.full((s, s), color, dtype=np.int8)
    if kind == HOLLOW:
        a[1:-1, 1:-1] = bg
    return a


def _parse_background(a, bg):
    """Recognize separated instances, never merge instances by color."""
    h, w = a.shape
    remaining = set(zip(*np.nonzero(a != bg)))
    objects = []
    common_size = None
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    component.append(p)
        top, left = (int(min(p[i] for p in component)) for i in (0, 1))
        bottom, right = (int(max(p[i] for p in component)) + 1 for i in (0, 1))
        s = bottom - top
        if s != right-left or not 3 <= s <= 10 or common_size not in (None, s):
            raise ValueError("not common-sized square tiles")
        common_size = s
        tile = a[top:bottom, left:right].copy()
        color = int(tile[0, 0])
        if color == bg:
            raise ValueError("missing tile rim")
        kind = SOLID if np.all(tile == color) else HOLLOW
        if not np.array_equal(tile, _tile(s, color, bg, kind)):
            raise ValueError("tile must be solid or a one-cell hollow frame")
        # A full background gap is required, including diagonal neighbors.
        r0, c0 = max(0, top-1), max(0, left-1)
        neighborhood = a[r0:min(h, bottom+1), c0:min(w, right+1)].copy()
        neighborhood[top-r0:bottom-r0, left-c0:right-c0] = bg
        if np.any(neighborhood != bg):
            raise ValueError("tiles must have a background gap")
        objects.append(dict(top=top, left=left, kind=kind, color=color, tile=tile))
    if not objects:
        raise ValueError("no tiles")
    objects.sort(key=lambda o: (o['top'], o['left']))
    queues = [[o for o in objects if o['kind'] == k] for k in (HOLLOW, SOLID)]
    if any(not q or len({o['top'] for o in q}) != len(q) for q in queues):
        raise ValueError("both classes required; same-class top ties forbidden")
    if common_size * (max(map(len, queues)) + 2) - 1 > 30:
        raise ValueError("sheared queue exceeds width 30")
    return dict(background=int(bg), size=common_size, objects=objects, queues=queues)


def _analyze(a):
    candidates = []
    for bg in np.unique(a):
        try:
            candidates.append(_parse_background(a, int(bg)))
        except ValueError:
            pass
    if len(candidates) != 1:
        raise ValueError("input must have one unique square-tile interpretation")
    return candidates[0]


def _queue(analysis, queues=None):
    s, bg = analysis['size'], analysis['background']
    queues = analysis['queues'] if queues is None else queues
    result = np.full((s * max(map(len, queues)), 2*s), bg, dtype=np.int8)
    for col, q in enumerate(queues):
        for rank, obj in enumerate(q):
            result[rank*s:(rank+1)*s, col*s:(col+1)*s] = obj['tile']
    return result


def _shear(a, bg):
    h, w = a.shape
    result = np.full((h, h+w-1), bg, dtype=np.int8)
    for r in range(h):
        offset = h-1-r
        result[r, offset:offset+w] = a[r]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Hollows left, solids right, independent vertical ranks, global shear."""
    analysis = _analyze(_grid(input_grid))
    return _shear(_queue(analysis), analysis['background'])


def _direct_render(objects, s, bg):
    """Labels from placed tiles, using the explicit source-cell coordinate map.

    This intentionally does not call solve, _queue, or _shear. Validation also
    uses this coordinate construction to check the queue/shear implementation.
    It is an authored consistency check, not an independent semantic oracle.
    """
    queues = [sorted((o for o in objects if o['kind'] == k),
                     key=lambda o: o['top']) for k in (HOLLOW, SOLID)]
    height = s * max(map(len, queues))
    result = np.full((height, height+2*s-1), bg, dtype=np.int8)
    for col, q in enumerate(queues):
        for rank, obj in enumerate(q):
            for u in range(s):
                r = rank*s+u
                for v in range(s):
                    c = height-1-r+col*s+v
                    result[r, c] = obj['tile'][u, v]
    return result


def _greedy_queue(a):
    slots = [[None, None]]
    for obj in a['objects']:
        k = obj['kind']
        if slots[-1][k] is not None:
            slots.append([None, None])
        slots[-1][k] = obj
    s = a['size']
    out = np.full((len(slots)*s, 2*s), a['background'], dtype=np.int8)
    for rank, row in enumerate(slots):
        for k, obj in enumerate(row):
            if obj is not None:
                out[rank*s:(rank+1)*s, k*s:(k+1)*s] = obj['tile']
    return out


def _alternatives(a):
    """Finite, named shortcut checks used to select informative demos."""
    bg, s = a['background'], a['size']
    correct_queue = _queue(a)
    alternatives = {
        'omit_A': _shear_input_unbounded(a['input'], bg),
        'omit_B': correct_queue,
        'greedy': _shear(_greedy_queue(a), bg),
        'swap_columns': _shear(_queue(a, a['queues'][::-1]), bg),
        'reverse_vertical': _shear(_queue(a, [q[::-1] for q in a['queues']]), bg),
    }
    for name, key in (('horizontal', lambda o: (o['left'], o['top'])),
                      ('color', lambda o: o['color'])):
        for reverse in (False, True):
            queues = [sorted(q, key=key, reverse=reverse) for q in a['queues']]
            alternatives[f'{name}_{reverse}'] = _shear(_queue(a, queues), bg)
    h, w = correct_queue.shape
    # Same canvas as the true answer: differentiate placement, not just shape.
    for name, offsets in (
        ('uniform_translation', [h-1]*h),
        ('opposite_shear', list(range(h))),
        ('tile_band_shear', [s*((h//s)-1-r//s) for r in range(h)]),
        ('reset_each_tile', [s-1-r % s for r in range(h)]),
    ):
        out = np.full((h, h+w-1), bg, dtype=np.int8)
        for r, offset in enumerate(offsets):
            out[r, offset:offset+w] = correct_queue[r]
        alternatives[name] = out
    return alternatives


def _shear_input_unbounded(a, bg):
    # A deliberately wrong rule can exceed ARC bounds; keep it only in RAM.
    return _shear(a, bg)


def _analyze_pair(pair):
    a = _analyze(_grid(pair['input']))
    a['input'] = np.asarray(pair['input'], dtype=np.int8)
    return a


def _has_anchor_evidence(pair):
    a = _analyze_pair(pair)
    out = pair['output']
    alternatives = _alternatives(a)
    return (all(not np.array_equal(out, value) for value in alternatives.values())
            and all(len({o['color'] for o in q}) >= 2 for q in a['queues']))


def _states(sizes):
    return [(s, nh, ns) for s in sizes
            for nh in range(1, 31//s-1)
            for ns in range(1, 31//s-1)]


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _positions(rng, s, n):
    """Feasible staggered shelves followed by unconstrained legal relocation."""
    margin = int(rng.random() < 0.32)
    extent = 30-2*margin
    capacity = (extent+1)//(s+1)
    shelf_options = [b for b in range(1, n+1)
                     if b*capacity >= n and n+b*s-1 <= extent]
    # Very large sizes can make a one-cell margin infeasible.
    if not shelf_options:
        margin, extent = 0, 30
        capacity = 31//(s+1)
        shelf_options = [b for b in range(1, n+1)
                         if b*capacity >= n and n+b*s-1 <= extent]
    shelves = _choose(rng, shelf_options)
    amounts = [1]*shelves
    for _ in range(n-shelves):
        i = _choose(rng, [i for i, count in enumerate(amounts) if count < capacity])
        amounts[i] += 1
    min_h, min_w = n+shelves*s-1, max(amounts)*(s+1)-1
    mode = int(rng.integers(6))
    if mode == 0:
        h = int(rng.integers(min_h, min(extent, min_h+3)+1))
        w = int(rng.integers(min_w, min(extent, min_w+3)+1))
    elif mode == 1:
        h = w = int(rng.integers(max(min_h, min_w), extent+1))
    elif mode == 2:
        h, w = extent, int(rng.integers(min_w, extent+1))
    elif mode == 3:
        h, w = int(rng.integers(min_h, extent+1)), extent
    elif mode == 4:
        h = w = extent
    else:
        h = int(rng.integers(min_h, extent+1))
        w = int(rng.integers(min_w, extent+1))
    vertical = rng.multinomial(h-min_h, np.full(shelves+1, 1/(shelves+1)))
    top, positions = int(vertical[0]), []
    for i, count in enumerate(amounts):
        slack = w-(count*(s+1)-1)
        gaps = rng.multinomial(slack, np.full(count+1, 1/(count+1)))
        left = int(gaps[0])
        for j, stagger in enumerate(rng.permutation(count)):
            positions.append((top+int(stagger), left))
            left += s+1+int(gaps[j+1])
        top += count+s+int(vertical[i+1])
    rr, cc = np.indices((h-s+1, w-s+1))
    for _ in range(2):
        for idx in rng.permutation(n):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != idx:
                    legal &= ((np.abs(rr-r) > s) | (np.abs(cc-c) > s))
                    legal &= rr != r
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(idx)] = (int(r), int(c))
    if margin == 0 and rng.random() < 0.55:
        side = int(rng.integers(4))
        dr = (-min(r for r, _ in positions) if side == 0 else
              h-s-max(r for r, _ in positions) if side == 1 else 0)
        dc = (-min(c for _, c in positions) if side == 2 else
              w-s-max(c for _, c in positions) if side == 3 else 0)
        positions = [(r+dr, c+dc) for r, c in positions]
    return (h+2*margin, w+2*margin), sorted((r+margin, c+margin) for r, c in positions)


def _kinds(rng, nh, ns, anchor):
    mode = int(rng.integers(4))
    if anchor or mode == 0:
        result = [HOLLOW]*nh+[SOLID]*ns
        if rng.integers(2):
            result.reverse()
    elif mode == 1:
        counts, result = [nh, ns], []
        first = int(rng.integers(2))
        while sum(counts):
            for k in (first, 1-first):
                if counts[k]:
                    result.append(k)
                    counts[k] -= 1
    else:
        result = [int(k) for k in rng.permutation([HOLLOW]*nh+[SOLID]*ns)]
    return result


def _make_pair(rng, state, bg, demonstration=None, anchor=False):
    s, nh, ns = state
    shape, positions = _positions(rng, s, nh+ns)
    kinds = _kinds(rng, nh, ns, anchor)
    # Opposite-class top ties are harmless. Same-class ties are never emitted.
    if not anchor and rng.random() < 0.28:
        idx = int(rng.integers(len(kinds)))
        for j in rng.permutation(len(kinds)):
            if kinds[j] == kinds[idx]:
                continue
            r, c = positions[idx]
            target = positions[int(j)][0]
            if all(i == idx or (
                    (kinds[i] != kinds[idx] or target != r2)
                    and (abs(target-r2) > s or abs(c-c2) > s))
                   for i, (r2, c2) in enumerate(positions)):
                positions[idx] = (target, c)
                break
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
    if nh+ns <= 9 and (anchor or rng.random() < 0.38):
        colors = palette[:nh+ns]
    else:
        pool = palette[:int(rng.integers(1, min(7, nh+ns)+1))]
        colors = [int(c) for c in rng.choice(pool, size=nh+ns)]
    if demonstration is not None:
        kind, color = demonstration
        idx = kinds.index(kind)
        if color in colors:
            other = colors.index(color)
            colors[idx], colors[other] = colors[other], colors[idx]
        else:
            colors[idx] = color
    if anchor:
        # Each queue's order must be visible; a >=3 queue rejects color sorting.
        for k in (HOLLOW, SOLID):
            indices = [i for i, kind in enumerate(kinds) if kind == k]
            if colors[indices[0]] == colors[indices[-1]]:
                colors[indices[-1]] = _choose(rng, [c for c in palette if c != colors[indices[0]]])
    inp = np.full(shape, bg, dtype=np.int8)
    objects = []
    for (r, c), kind, color in zip(positions, kinds, colors):
        tile = _tile(s, color, bg, kind)
        inp[r:r+s, c:c+s] = tile
        objects.append(dict(top=r, left=c, kind=kind, color=color, tile=tile))
    return dict(input=inp, output=_direct_render(objects, s, bg))


def _fallback_anchor(s, bg, demonstration, sign):
    """A bounded five-tile witness with a zigzag in the three-member class."""
    positions = [(0, s+1), (1, 0), (s+2, 0), (s+3, s+1), (2*s+4, 0)]
    kinds = [HOLLOW, HOLLOW, SOLID, SOLID, SOLID]
    if sign < 0:
        kinds = [1-k for k in kinds]
    palette = [demonstration]+[c for c in range(10) if c not in (bg, demonstration)]
    colors = [palette[0], palette[1], palette[2], palette[4], palette[3]]
    target = kinds.index(HOLLOW)
    colors[0], colors[target] = colors[target], colors[0]
    # Regardless of the demonstration's numeric value, make the three-color
    # class nonmonotone while retaining the same color/geometry memberships.
    group = [i for i, k in enumerate(kinds) if kinds.count(k) == 3]
    low, middle, high = sorted(colors[i] for i in group)
    for i, color in zip(group, (low, high, middle)):
        colors[i] = color
    inp = np.full((3*s+4, 2*s+1), bg, dtype=np.int8)
    objects = []
    for (r, c), k, color in zip(positions, kinds, colors):
        tile = _tile(s, color, bg, k)
        inp[r:r+s, c:c+s] = tile
        objects.append(dict(top=r, left=c, kind=k, color=color, tile=tile))
    return dict(input=inp, output=_direct_render(objects, s, bg))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_size = int(rng.integers(3, 7))
    sizes = [first_size] if rng.random() < 0.25 else list(range(3, 11))
    states = _states(sizes)
    demonstration = int(rng.integers(10))
    varying_bg = bool(rng.random() < 0.65)
    base_bg = _choose(rng, [c for c in range(10) if c != demonstration])
    first_sign = _choose(rng, [-1, 1])
    puzzle, seen, train_heights = dict(train=[], test=[]), set(), set()
    reserved_height = None
    for index in range(num_train+num_test):
        training = index < num_train
        if index == 0:
            candidates = [x for x in states if x[0] == first_size
                          and min(x[1:]) >= 2 and np.sign(x[2]-x[1]) == first_sign]
        elif index == 1:
            candidates = [x for x in states if np.sign(x[2]-x[1]) == -first_sign
                          and x[0]*max(x[1:]) not in train_heights]
        elif training:
            candidates = [x for x in states if x[0]*max(x[1:]) != reserved_height]
        elif index == num_train:
            candidates = [x for x in states if x[0]*max(x[1:]) == reserved_height]
        else:
            candidates = states
        # Uniform sampling of heights avoids letting the many small-scale
        # count combinations swamp larger squares and singleton queues.
        possible_heights = sorted({x[0]*max(x[1:]) for x in candidates})
        height = _choose(rng, possible_heights)
        state = _choose(rng, [x for x in candidates if x[0]*max(x[1:]) == height])
        bg = _choose(rng, [c for c in range(10) if c != demonstration]) if varying_bg else base_bg
        forced = (HOLLOW, demonstration) if index == 0 else ((SOLID, demonstration) if index == 1 else None)
        for attempt in range(48):
            pair = _make_pair(rng, state, bg, forced, anchor=index == 0)
            if _key(pair['input']) in seen:
                continue
            if index == 0 and not _has_anchor_evidence(pair):
                continue
            break
        else:
            if index != 0:
                raise RuntimeError("48 bounded attempts failed to make a distinct pair")
            pair = _fallback_anchor(first_size, bg, demonstration, first_sign)
            if not _has_anchor_evidence(pair):
                raise RuntimeError("invalid anchor fallback")
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
        if training:
            a = _analyze(pair['input'])
            train_heights.add(a['size']*max(map(len, a['queues'])))
        if index == 1:
            remaining = sorted({x[0]*max(x[1:]) for x in states}-train_heights)
            reserved_height = _choose(rng, remaining)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, test_heights = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            try:
                inp, out = _grid(pair['input']), _grid(pair['output'])
                a = _analyze(inp)
            except (ValueError, TypeError) as exc:
                raise AssertionError(f"malformed pair: {exc}") from exc
            a['input'] = inp
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            bg, s = a['background'], a['size']
            q = _queue(a)
            h = q.shape[0]
            assert out.shape == (h, h+2*s-1), 'wrong expanded dimensions'
            assert np.array_equal(out, _direct_render(a['objects'], s, bg)), 'wrong cell map'
            # Inverse shear must recover the entire rectangular queue, even
            # its hollow interiors and trailing empty partner slots.
            for r in range(h):
                offset = h-1-r
                assert np.array_equal(out[r, offset:offset+2*s], q[r]), 'row vector changed'
                assert np.all(out[r, :offset] == bg) and np.all(out[r, offset+2*s:] == bg)
            for color in range(10):
                if color != bg:
                    assert np.count_nonzero(inp == color) == np.count_nonzero(out == color), 'payload lost'
            assert not np.array_equal(out, q), 'B must affect the answer'
            assert not np.array_equal(out, _shear_input_unbounded(inp, bg)), 'A must affect the answer'
            if split == 'train':
                training.append((a, out))
            else:
                test_heights.append(h)
    train_heights = {out.shape[0] for _, out in training}
    assert len(train_heights) >= 2, 'show more than one queue height'
    assert test_heights[0] not in train_heights, 'missing held-out-height test shift'
    differences = [len(a['queues'][1])-len(a['queues'][0]) for a, _ in training]
    assert min(differences) < 0 < max(differences), 'show padding on both sides'
    roles = {}
    for a, _ in training:
        for o in a['objects']:
            roles.setdefault(o['color'], set()).add(o['kind'])
    assert any(len(v) == 2 for v in roles.values()), 'missing color-role switch'
    names = set(_alternatives(training[0][0]))
    rejected = set()
    for a, out in training:
        rejected.update(name for name, prediction in _alternatives(a).items()
                        if not np.array_equal(prediction, out))
    assert rejected == names, 'demonstrations admit a checked shortcut'
    for k in (HOLLOW, SOLID):
        assert any(len({o['color'] for o in a['queues'][k]}) >= 2 for a, _ in training), 'invisible rank order'


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test() -> dict:
    """Stream 42 seeds; retain small metrics and hashes, not grid corpora."""
    signatures, shapes, layouts, sizes, backgrounds, counts = set(), set(), set(), set(), set(), set()
    structures, height_shifts = set(), Counter()
    properties = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        heights = [np.asarray(p['output']).shape[0] for p in puzzle['train']]
        t = puzzle['test'][0]['output'].shape[0]
        height_shifts['larger' if t > max(heights) else 'smaller' if t < min(heights) else 'between'] += 1
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                assert np.array_equal(solve(inp.copy()), out)
                a = _analyze(inp)
                s, bg = a['size'], a['background']
                sizes.add(s)
                backgrounds.add(bg)
                shapes.add(inp.shape)
                nh, ns = map(len, a['queues'])
                counts.add((nh, ns))
                layout = (s, inp.shape, tuple((o['top'], o['left'], o['kind']) for o in a['objects']))
                layouts.add(layout)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).hexdigest()))
                properties['pairs'] += 1
                properties['repeated_colors'] += len({o['color'] for o in a['objects']}) < nh+ns
                properties['cross_class_top_ties'] += len({o['top'] for o in a['objects']}) < nh+ns
                properties['edge_tiles'] += any(o['top'] == 0 or o['left'] == 0
                                                or o['top']+s == inp.shape[0]
                                                or o['left']+s == inp.shape[1] for o in a['objects'])
                properties['large_imbalance'] += abs(nh-ns) >= 2
                properties['balanced'] += nh == ns
                properties['black_tiles'] += bg != 0 and any(o['color'] == 0 for o in a['objects'])
    assert len(signatures) == 42 and len(layouts) >= 180 and len(structures) >= 180
    assert sizes == set(range(3, 11)) and backgrounds == set(range(10))
    assert len(shapes) >= 40 and len(counts) >= 20
    assert all(properties[k] >= 5 for k in ('repeated_colors', 'edge_tiles', 'large_imbalance', 'balanced', 'black_tiles'))
    assert properties['cross_class_top_ties'] >= 1
    assert height_shifts['larger'] and height_shifts['smaller']
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    # All legal counts, with the same strong two-demo evidence requirement.
    legal_count_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
            legal_count_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Hand-computed small fixture: H at (4,0), S at (0,5); A swaps their
    # spatial order, B shifts the three compact rows by 2,1,0.
    inp = np.zeros((7, 8), dtype=np.int8)
    inp[4:7, :3] = _tile(3, 2, 0, HOLLOW)
    inp[:3, 5:8] = 3
    expected = np.array([[0, 0, 2, 2, 2, 3, 3, 3],
                         [0, 2, 0, 2, 3, 3, 3, 0],
                         [2, 2, 2, 3, 3, 3, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(inp), expected)
    # Both signs and all supported anchor scales have guaranteed fallbacks.
    for s in range(3, 7):
        for sign in (-1, 1):
            assert _has_anchor_evidence(_fallback_anchor(s, 0, 5, sign))
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), distinct_layouts=len(layouts),
                color_normalized_inputs=len(structures), input_shapes=len(shapes),
                tile_sizes=sorted(sizes), backgrounds=sorted(backgrounds),
                count_combinations=len(counts), height_shifts=dict(height_shifts),
                properties=dict(properties), legal_count_calls=legal_count_calls,
                semantic_verification='authored checks; no independent solver')


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
