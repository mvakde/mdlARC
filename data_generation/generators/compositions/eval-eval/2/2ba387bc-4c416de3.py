"""Ordinal negative-space stamps: input-only geometry and local reflection.

No parent generator is imported. All sampling uses the caller's one NumPy RNG.
The generator paints labels from its construction; solve parses visible pixels.
No files, traces, caches, or JSON are written by this module.
"""

from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(points):
    if not points:
        return False
    todo = [min(points)]
    seen = set(todo)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = u + du, v + dv
                if p in points and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return len(seen) == len(points)


def _local(s, q, p):
    u, v = p
    dy, dx = _DIRS[q]
    return (u if dy == 1 else s - 1 - u,
            v if dx == 1 else s - 1 - v)


def _patches(s):
    k = (s - 3) // 2
    return [{_local(s, q, (u, v)): (u, v)
             for u in range(1, k + 1) for v in range(1, k + 1)}
            for q in range(4)]


def _parse(value):
    """Read separated square rims, then their cores and corner occupancy."""
    inp = _grid(value)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in
               (inp[0], inp[-1], inp[:, 0], inp[:, -1])), "background border"
    todo = inp != bg
    objects = []
    sizes = set()
    while np.any(todo):
        r, c = map(int, np.argwhere(todo)[0])
        color = int(inp[r, c])
        s = 0
        while c + s < inp.shape[1] and inp[r, c + s] == color:
            s += 1
        assert 7 <= s <= 11 and r + s < inp.shape[0]
        tile = inp[r:r+s, c:c+s]
        assert tile.shape == (s, s)
        assert all(np.all(edge == color) for edge in
                   (tile[0], tile[-1], tile[:, 0], tile[:, -1])), "square rim"
        assert np.all(np.isin(tile, [bg, color])), "monochrome carrier"
        halo = inp[r-1:r+s+1, c-1:c+s+1].copy()
        halo[1:-1, 1:-1] = bg
        assert np.all(halo == bg), "tiles must have a full background gap"
        patches = _patches(s)
        patch_union = set().union(*(set(p) for p in patches))
        interior = {(u, v) for u in range(1, s-1) for v in range(1, s-1)}
        guard = interior - patch_union
        core = {int(tile[p]) for p in guard}
        obj = dict(top=r, left=c, size=s, color=color)
        if core == {color}:
            obj['kind'] = 'D'
            holes = {p for p in interior if tile[p] == bg}
            qs = [q for q, patch in enumerate(patches) if holes <= set(patch)]
            assert len(qs) == 1 and 2 <= len(holes) <= 10, "one intact hole stencil"
            q = qs[0]
            pattern = frozenset(patches[q][p] for p in holes)
            assert _connected(pattern), "hole stencil must be 8-connected"
            obj.update(q=q, pattern=pattern)
        else:
            assert core == {bg}, "core must be solid or hollow"
            obj['kind'] = 'R'
            ink = {p for p in interior if tile[p] == color}
            assert ink and ink <= patch_union, "hollow corner clues"
            obj['marks'] = {q: frozenset(patch[p] for p in ink & set(patch))
                            for q, patch in enumerate(patches) if ink & set(patch)}
        todo[r:r+s, c:c+s] = False
        sizes.add(s)
        objects.append(obj)
    assert len(sizes) == 1, "one tile scale per input"
    queues = {}
    for kind in ('D', 'R'):
        queue = sorted((o for o in objects if o['kind'] == kind), key=lambda o: o['top'])
        assert queue and len({o['top'] for o in queue}) == len(queue), "no within-class top ties"
        queues[kind] = queue
    return dict(input=inp, background=bg, size=sizes.pop(), objects=objects,
                donors=queues['D'], receivers=queues['R'])


def _rank_assignment(scene):
    return [i if i < len(scene['donors']) else None
            for i in range(len(scene['receivers']))]


def _render(scene, assignment=None, *, rotation=False, all_corners=False):
    """None means an unpaired receiver. Incompatible countermodels fail."""
    donors, receivers = scene['donors'], scene['receivers']
    assignment = _rank_assignment(scene) if assignment is None else assignment
    out = scene['input'].copy()
    for receiver, donor_id in zip(receivers, assignment):
        if donor_id is None:
            continue
        donor = donors[donor_id]
        for q in (range(4) if all_corners else receiver['marks']):
            pattern = donor['pattern']
            if rotation and ((donor['q'] in (1, 2)) != (q in (1, 2))):
                pattern = frozenset((v, u) for u, v in pattern)
            assert receiver['marks'].get(q, frozenset()) <= pattern, "incompatible clue"
            for p in pattern:
                u, v = _local(scene['size'], q, p)
                out[receiver['top'] + u, receiver['left'] + v] = receiver['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, reference stencil, rank, orientation and clue from input."""
    return _render(_parse(input_grid))


def _assignments(scene):
    """Finite plausible alternatives used as evidence checks, not a solver."""
    ds, rs = scene['donors'], scene['receivers']
    n, m = len(ds), len(rs)
    result = {}
    for reverse in (False, True):
        di = sorted(range(n), key=lambda i: ds[i]['left'], reverse=reverse)
        ri = sorted(range(m), key=lambda i: rs[i]['left'], reverse=reverse)
        a = [None] * m
        for d, r in zip(di, ri):
            a[r] = d
        result['horizontal_desc' if reverse else 'horizontal_asc'] = a
    result['reverse_donors'] = [n-1-i if i < n else None for i in range(m)]
    result['reverse_receivers'] = list(reversed(_rank_assignment(scene)))
    a = [None] * m
    used = set()
    distances = sorted((abs(d['top']-r['top']) + abs(d['left']-r['left']), j, i)
                       for j, d in enumerate(ds) for i, r in enumerate(rs))
    for _, j, i in distances:
        if j not in used and a[i] is None:
            a[i] = j
            used.add(j)
    result['nearest'] = a
    a = [None] * m
    row = {}
    for obj in sorted(scene['objects'], key=lambda o: (o['top'], o['left'])):
        kind = obj['kind']
        if kind in row:
            row = {}
        row[kind] = obj
        if len(row) == 2:
            ri = next(i for i, r in enumerate(rs) if r is row['R'])
            a[ri] = next(i for i, d in enumerate(ds) if d is row['D'])
    result['greedy_rows'] = a
    result['one_global_stencil'] = [0] * m
    result['cycle_donors'] = [i % n for i in range(m)]
    if n >= 2 and m >= 2:
        a = _rank_assignment(scene)
        a[0], a[1] = a[1], a[0]
        result['swap_first_two'] = a
    return result


def _witnesses(scene, expected):
    result = {}
    for name, assignment in _assignments(scene).items():
        try:
            result[name] = not np.array_equal(_render(scene, assignment), expected)
        except AssertionError:
            result[name] = True
    for name, options in (('rotation', dict(rotation=True)),
                          ('all_corners', dict(all_corners=True))):
        try:
            result[name] = not np.array_equal(_render(scene, **options), expected)
        except AssertionError:
            result[name] = True
    result['no_completion'] = not np.array_equal(scene['input'], expected)
    return result


def _pattern(rng, k, *, anchor=None, asymmetric=False, reach=False, exclude=None):
    domain = [(u, v) for u in range(1, k+1) for v in range(1, k+1)]
    for _ in range(100):
        start = anchor if anchor is not None else domain[int(rng.integers(len(domain)))]
        points = {start}
        target = int(rng.integers(2, min(10, k*k) + 1))
        while len(points) < target:
            frontier = [p for p in domain if p not in points and any(
                max(abs(p[0]-u), abs(p[1]-v)) == 1 for u, v in points)]
            points.add(frontier[int(rng.integers(len(frontier)))])
        p = frozenset(points)
        if asymmetric and p == frozenset((v, u) for u, v in p):
            continue
        if reach and max(max(q) for q in p) <= 2:
            continue
        if p == exclude:
            continue
        return p
    # Constructive fallback also preserves all requested properties.
    start = anchor or ((k, k) if reach else (1, 1))
    u, v = start
    neighbor = (u, v-1 if v > 1 else v+1)
    p = frozenset((start, neighbor))
    if p == exclude:
        p = frozenset((start, (u-1 if u > 1 else u+1, v)))
    return p


def _pack(rng, s, count):
    """Feasible variable shelves, followed by free legal moves on the canvas."""
    cap = 29 // (s+1)
    bands = [b for b in range(1, count+1)
             if b*cap >= count and b*s+count+1 <= 30]
    b = int(rng.choice(bands))
    amounts = [1]*b
    for _ in range(count-b):
        choices = [i for i, n in enumerate(amounts) if n < cap]
        amounts[int(rng.choice(choices))] += 1
    min_h = b*s + count + 1
    min_w = max(amounts)*(s+1) + 1
    if rng.random() < .25:
        h = w = int(rng.integers(max(min_h, min_w), 31))
    else:
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
    vg = rng.multinomial(h-min_h, np.full(b+1, 1/(b+1)))
    positions = []
    top = 1 + int(vg[0])
    for band, amount in enumerate(amounts):
        hg = rng.multinomial(w-(amount*(s+1)+1), np.full(amount+1, 1/(amount+1)))
        left = 1 + int(hg[0])
        for j, offset in enumerate(rng.permutation(amount)):
            positions.append((top+int(offset), left))
            left += s+1+int(hg[j+1])
        top += s+amount+int(vg[band+1])
    rr, cc = np.indices((h-s-1, w-s-1))
    rr, cc = rr+1, cc+1
    for _ in range(2):
        for i in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if i != j:
                    legal &= ((np.abs(rr-r) > s) | (np.abs(cc-c) > s)) & (rr != r)
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(i)] = (int(r)+1, int(c)+1)
    return (h, w), positions


def _construct(rng, size, nd, nr, *, witness=False, reach=False):
    shape, positions = _pack(rng, size, nd+nr)
    indices = rng.permutation(nd+nr)
    dpos = sorted(positions[int(i)] for i in indices[:nd])
    rpos = sorted(positions[int(i)] for i in indices[nd:])
    # Cross-class top ties do not affect either queue. Same-class ties are absent.
    if not witness and rng.random() < .3:
        i = int(rng.integers(nr))
        top = dpos[int(rng.integers(nd))][0]
        other = dpos + [p for j, p in enumerate(rpos) if j != i]
        if all(top != r for j, (r, _) in enumerate(rpos) if j != i):
            for left in rng.permutation(np.arange(1, shape[1]-size)):
                if all(abs(top-r) > size or abs(int(left)-c) > size for r, c in other):
                    rpos[i] = (top, int(left))
                    rpos.sort()
                    break
    bg = int(rng.integers(10))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    pool = palette[:int(rng.integers(1, min(6, nd+nr)+1))]
    dc = list(map(int, rng.choice(pool, nd)))
    rc = list(map(int, rng.choice(pool, nr)))
    if witness:
        dc[1] = dc[0]
        if rc[0] == dc[0]:
            rc[0] = next(c for c in palette if c != dc[0])
        rc[1] = rc[0]
    k = (size-3)//2
    a = int(rng.integers(1, k+1))
    common = (a, a)
    patterns = []
    for i in range(nd):
        patterns.append(_pattern(rng, k, anchor=common if witness and i < 2 else None,
                                 asymmetric=witness and i == 0, reach=reach,
                                 exclude=patterns[0] if witness and i == 1 else None))
    inp = np.full(shape, bg, dtype=np.int8)
    for (r, c), color, pattern in zip(dpos, dc, patterns):
        inp[r:r+size, c:c+size] = color
        q = int(rng.integers(4))
        for p in pattern:
            u, v = _local(size, q, p)
            inp[r+u, c+v] = bg
    planned = []
    for i, ((r, c), color) in enumerate(zip(rpos, rc)):
        tile = inp[r:r+size, c:c+size]
        tile[0] = tile[-1] = color
        tile[:, 0] = tile[:, -1] = color
        pattern = patterns[i % nd]
        if witness and i == 0:
            q = int(rng.integers(4))
            qs = [q, q ^ (1 if rng.integers(2) else 2)]
        else:
            qs = list(map(int, rng.choice(4, int(rng.integers(1, 5)), replace=False)))
        for q in qs:
            if witness and i < 2:
                retained = [common]
            else:
                cells = sorted(pattern)
                keep = int(rng.integers(1, len(cells)+(1 if i < nd else 0)))
                retained = [cells[int(j)] for j in rng.permutation(len(cells))[:keep]]
            for p in retained:
                u, v = _local(size, q, p)
                tile[u, v] = color
            if i < nd:
                planned.append((r, c, color, q, pattern))
    # The construction label uses the planned local coordinates, not solve.
    out = inp.copy()
    for r, c, color, q, pattern in planned:
        for u, v in pattern:
            dy, dx = _DIRS[q]
            y = r + (u if dy > 0 else size-1-u)
            x = c + (v if dx > 0 else size-1-v)
            out[y, x] = color
    return dict(input=inp, output=out)


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    nq = _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    extent_shift = bool(rng.integers(2))
    first_extra_donor = bool(rng.integers(2))
    puzzle = dict(train=[], test=[])
    seen = set()
    required = ('horizontal_asc', 'horizontal_desc', 'reverse_donors',
                'reverse_receivers', 'nearest', 'greedy_rows',
                'one_global_stencil', 'rotation', 'all_corners')
    for index in range(nt+nq):
        training = index < nt
        witness = index == 0
        reach = not training and extent_shift
        if index < 2:
            nd, nr = (3, 2) if first_extra_donor == (index == 0) else (2, 3)
            size = int(rng.choice([7, 8]))
        else:
            size = int(rng.choice([7, 8] if training and extent_shift else
                                  [9, 10, 11] if reach else [7, 8, 9, 10, 11]))
            limit = 6 if size <= 8 else 4
            counts = [(d, r) for d in range(1, 5) for r in range(1, 5)
                      if d+r <= limit and (abs(d-r) <= 1 if training else
                                           True if extent_shift else abs(d-r) >= 2)]
            nd, nr = counts[int(rng.integers(len(counts)))]
        for attempt in range(256):
            pair = _construct(rng, size, nd, nr, witness=witness, reach=reach)
            if _key(pair['input']) in seen or np.array_equal(pair['input'], pair['output']):
                continue
            scene = _parse(pair['input'])
            assert np.array_equal(_render(scene), pair['output'])
            if witness:
                ws = _witnesses(scene, pair['output'])
                if not all(ws[name] for name in required):
                    continue
                swapped = _render(scene, _assignments(scene)['swap_first_two'])
                assert not np.array_equal(swapped, pair['output'])
            break
        else:
            raise RuntimeError("bounded construction exhausted")
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    training_witnesses = []
    training_scenes = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and np.any(inp != out)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            scene = _parse(inp)
            predicted = _render(scene)
            assert np.array_equal(predicted, out), "input-only reconstruction differs"
            # Completion is additive ink only; no outline, donor or unused tile changes.
            changed = inp != out
            assert np.all(inp[changed] == scene['background'])
            allowed = np.zeros(inp.shape, dtype=bool)
            for i, r in enumerate(scene['receivers']):
                tile_in = inp[r['top']:r['top']+scene['size'], r['left']:r['left']+scene['size']]
                tile_out = out[r['top']:r['top']+scene['size'], r['left']:r['left']+scene['size']]
                if i >= len(scene['donors']):
                    assert np.array_equal(tile_in, tile_out), "unmatched receiver changed"
                    continue
                p = scene['donors'][i]['pattern']
                for q, clue in r['marks'].items():
                    assert clue <= p
                    for point in p:
                        u, v = _local(scene['size'], q, point)
                        allowed[r['top']+u, r['left']+v] = True
                        assert tile_out[u, v] == r['color']
                assert np.all(tile_out[tile_in == r['color']] == r['color'])
            assert np.all(~changed | allowed), "change outside a requested stamp"
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            if split == 'train':
                training_scenes.append(scene)
                training_witnesses.append(_witnesses(scene, out))
    for name in ('horizontal_asc', 'horizontal_desc', 'reverse_donors',
                 'reverse_receivers', 'nearest', 'greedy_rows', 'one_global_stencil',
                 'rotation', 'all_corners', 'cycle_donors', 'no_completion'):
        assert any(w[name] for w in training_witnesses), f"missing witness against {name}"
    gaps = [len(s['donors'])-len(s['receivers']) for s in training_scenes]
    assert min(gaps) < 0 < max(gaps), "teach both unmatched directions"
    assert any(d['color'] != r['color'] and any(clue != d['pattern'] for clue in r['marks'].values())
               for s in training_scenes for d, r in zip(s['donors'], s['receivers'])), "teach local ink color"
    # A direct, clue-compatible ablation of A must change at least one answer.
    assert any(_swap_is_compatible(s) for s in training_scenes), "rank is redundant"


def _swap_is_compatible(scene):
    if min(len(scene['donors']), len(scene['receivers'])) < 2:
        return False
    try:
        alternative = _render(scene, _assignments(scene)['swap_first_two'])
    except AssertionError:
        return False
    return not np.array_equal(alternative, _render(scene))


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds, metamorphic properties, ablations and negative controls."""
    signatures, shapes, layouts, patterns, counts, scales, colors = (set() for _ in range(7))
    mark_counts, reference_qs, patch_qs = set(), set(), set()
    asymmetric = symmetric = repeated = offset = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                scene = _parse(inp)
                assert np.array_equal(solve(inp.copy()), out)
                shapes.add(inp.shape)
                scales.add(scene['size'])
                colors.add(scene['background'])
                ds, rs = scene['donors'], scene['receivers']
                counts.add((len(ds), len(rs)))
                layouts.add(tuple((o['top'], o['left'], o['kind']) for o in scene['objects']))
                repeated += len({o['color'] for o in scene['objects']}) < len(scene['objects'])
                for d in ds:
                    p = d['pattern']
                    patterns.add(tuple(sorted(p)))
                    is_symmetric = p == frozenset((v, u) for u, v in p)
                    symmetric += is_symmetric
                    asymmetric += not is_symmetric
                    reference_qs.add(d['q'])
                for r in rs:
                    mark_counts.add(len(r['marks']))
                    patch_qs.update(r['marks'])
                    offset += any((1, 1) not in p for p in r['marks'].values())
                # Palette equivariance: no numeric color encodes an identity.
                permutation = np.array([3, 8, 0, 6, 1, 9, 4, 2, 7, 5], dtype=np.int8)
                assert np.array_equal(solve(permutation[inp]), permutation[out])
                # Horizontal reflection preserves vertical ranks and reflects stencils.
                assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(layouts) >= 180
    assert len(patterns) >= 50 and len(counts) >= 10
    assert scales == {7, 8, 9, 10, 11} and colors == set(range(10))
    assert mark_counts == {1, 2, 3, 4} and reference_qs == patch_qs == set(range(4))
    assert min(asymmetric, symmetric, repeated, offset) >= 15
    # The parser's family is wider than generate_puzzle's six-tile sampling cap.
    wide = _construct(np.random.default_rng(991), 7, 4, 4)
    assert len(_parse(wide['input'])['objects']) == 8
    assert np.array_equal(solve(wide['input']), wide['output'])
    assert _digest(generate_puzzle(619960834)) == _digest(generate_puzzle(619960834))
    # Exercise every legal count without retaining the resulting puzzles.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(184657683, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated output mutation accepted")
    # Erasing completion from both demos must fail the family validator.
    for p in damaged['train']:
        p['output'] = p['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity demonstrations accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 streamed seeds; no artifacts written)')
