"""Rotational echo scanner: periodic cuts become a shared intrinsic shape mask.

Only the visible input is used by solve.  Generation and validation do not read
the parent files, and all sampling uses the caller's default_rng stream.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _view(a, side):
    """A rotation, not a reflection: the head becomes the top edge."""
    return np.rot90(a, side)


def _unview(a, side):
    return np.rot90(a, -side).copy()


def _mask_key(mask):
    return mask.shape[0], mask.shape[1], tuple(int(x) for x in mask.ravel())


def _crop(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _forms(mask, reflections=False):
    # Duplicates deliberately remain: all valid registrations count.
    for reflected in range(2 if reflections else 1):
        base = np.fliplr(mask) if reflected else mask
        for turn in range(4):
            yield reflected, turn, np.rot90(base, turn)


def _canonical(mask, reflections=False):
    return min(_mask_key(m) for _, _, m in _forms(mask, reflections))


def _apply(mask, reflected, turn):
    return np.rot90(np.fliplr(mask) if reflected else mask, turn)


def _undo(mask, reflected, turn):
    a = np.rot90(mask, -turn)
    return np.fliplr(a) if reflected else a


def _segment(a, background, excluded=(), directions=DIR8):
    visited = np.zeros(a.shape, dtype=bool)
    objects = []
    for r, c in np.ndindex(a.shape):
        color = int(a[r, c])
        if visited[r, c] or color == background or color in excluded:
            continue
        stack, cells = [(r, c)], []
        visited[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in directions:
                yy, xx = y + dy, x + dx
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and not visited[yy, xx] and a[yy, xx] == color):
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        top, left = min(y for y, x in cells), min(x for y, x in cells)
        mask = np.zeros((max(y for y, x in cells) - top + 1,
                         max(x for y, x in cells) - left + 1), dtype=bool)
        for y, x in cells:
            mask[y - top, x - left] = True
        objects.append(dict(top=top, left=left, color=color, mask=mask))
    return objects


def _analyze(a):
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() <= a.size / 2:
        raise ValueError("background must occupy more than half the input")
    candidates = []
    for side in range(4):
        g = _view(a, side)
        if len(g) < 5:
            continue
        for marker in values:
            marker = int(marker)
            if marker == bg:
                continue
            cols = np.flatnonzero(g[0] == marker)
            if not len(cols) or not np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
                continue
            start, end = int(cols[0]), int(cols[-1]) + 1
            # Head ends stay away from corners; reserved colors occur only here.
            if start == 0 or end == g.shape[1]:
                continue
            marker_mask = np.zeros(g.shape, dtype=bool)
            marker_mask[0, start:end] = marker_mask[2, start:end] = True
            if not np.array_equal(g == marker, marker_mask):
                continue
            highlight = int(g[4, start])
            if highlight in (bg, marker) or not np.all(g[4, start:end] == highlight):
                continue
            high_mask = np.zeros(g.shape, dtype=bool)
            high_mask[4, start:end] = True
            if not np.array_equal(g == highlight, high_mask):
                continue
            head = g[:5, start:end]
            if not np.all(head[[1, 3]] == bg):
                continue
            candidates.append((side, g, bg, marker, highlight, start, end))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous two-color calibration head")
    side, g, bg, marker, highlight, start, end = candidates[0]
    objects = _segment(g, bg, (marker, highlight))
    if not objects:
        raise ValueError("expected at least one object")
    # An empty scanning corridor leaves the objects visibly intact.
    corridor = g[:, start:end].copy()
    corridor[:5] = bg
    if np.any(corridor != bg):
        raise ValueError("objects must lie outside the ruler corridor")
    return dict(side=side, grid=g, background=bg, marker=marker,
                highlight=highlight, start=start, end=end, objects=objects)


def _prediction(info, alternative=None):
    """Shared-mask construction; alternatives are used only as evidence checks."""
    g, objects = info['grid'], info['objects']
    out = g.copy()
    bg, marker, high = info['background'], info['marker'], info['highlight']
    if alternative == 'four_connectivity':
        objects = _segment(g, bg, (marker, high), DIR4)
    start, end = info['start'], info['end']
    for r in range(len(g)):
        out[r, start:end] = bg if r % 2 else high if r % 6 == 4 else marker
    reflections = alternative == 'reflection'
    groups = defaultdict(list)
    for obj in objects:
        mask = obj['mask']
        key = _mask_key(mask) if alternative == 'identity' else _canonical(mask, reflections)
        if alternative == 'color':
            key = (key, obj['color'])
        groups[key].append(obj)
    for group in groups.values():
        # A singleton has nobody to echo to: rotational registration is used
        # only by repeated classes.  Even a symmetric singleton is cut locally.
        local_only = ((len(group) == 1 and alternative != 'singleton_symmetry') or alternative == 'local_only'
                      or (alternative == 'exactly_two' and len(group) != 2))
        reference = (_mask_key(group[0]['mask']) if alternative == 'identity'
                     else _canonical(group[0]['mask'], reflections))
        h, w = reference[:2]
        shared = np.zeros((h, w), dtype=bool)
        registrations, hits = [], []
        for obj in group:
            mask = obj['mask']
            rows = np.arange(mask.shape[0]) + obj['top']
            scanned = (rows % 6 == (0 if alternative == 'phase_zero' else 4))[:, None]
            if alternative == 'every_even':
                scanned = (rows % 2 == 0)[:, None]
            hit = mask & scanned
            if alternative == 'all_cells':
                hit = mask.copy()
            if alternative == 'strip_only':
                hit = np.zeros_like(mask)
            hits.append(hit)
            regs = [(f, k) for f, k, m in _forms(mask, reflections)
                    if _mask_key(m) == reference]
            if alternative == 'identity':
                regs = [(0, 0)]
            if alternative == 'one_registration':
                regs = regs[:1]
            registrations.append(regs)
            if alternative != 'first_source' or len(hits) == 1:
                for f, k in regs:
                    shared |= _apply(hit, f, k)
        if alternative == 'whole_object' and shared.any():
            # The chosen reference includes the complete occupied-cell mask.
            shared = np.array(reference[2], dtype=bool).reshape(h, w)
        for obj, regs, hit in zip(group, registrations, hits):
            marked = hit.copy() if local_only else np.zeros_like(obj['mask'])
            if not local_only:
                for f, k in regs:
                    marked |= _undo(shared, f, k)
            ys, xs = np.nonzero(marked)
            out[obj['top'] + ys, obj['left'] + xs] = high
    return _unview(out, info['side'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the scanner and broadcast the union of registered scan cuts."""
    return _prediction(_analyze(_grid(input_grid)))


def _connected(mask, directions=DIR4):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    root = min(remaining)
    seen, stack = {root}, [root]
    while stack:
        r, c = stack.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in remaining and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == remaining


def _random_mask(rng, limit=5, area=None):
    kind = int(rng.integers(6)) if area is None else 0
    h, w = (int(rng.integers(2, limit + 1)) for _ in range(2))
    if kind in (0, 1):
        count = int(rng.integers(3, min(18, limit * limit) + 1)) if area is None else area
        cells = {(0, 0)}
        directions = DIR8 if kind == 1 else DIR4
        while len(cells) < count:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in directions
                               if 0 <= r + dr < limit and 0 <= c + dc < limit} - cells)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        m = np.zeros((limit, limit), dtype=bool)
        for p in cells:
            m[p] = True
    elif kind == 2:
        m = np.zeros((h, w), dtype=bool)
        m[:, 0] = m[-1, :] = True
    elif kind == 3:
        m = np.ones((h, w), dtype=bool)
        if min(h, w) > 2 and rng.random() < .7:
            m[1:-1, 1:-1] = False
    elif kind == 4:
        m = np.ones((1, int(rng.integers(1, limit + 1))), dtype=bool)
    else:
        m = np.zeros((h, w), dtype=bool)
        m[1:, 1] = m[-1, 1:] = True
        m[0, 0] = True
    return _crop(np.rot90(m, int(rng.integers(4))))


def _sample_mask(rng, forbidden, chiral=False, diagonal=False, area=None, symmetric=False):
    for _ in range(200):
        if symmetric:
            h = int(rng.integers(2, 5))
            w = h if rng.random() < .65 else int(rng.integers(2, 5))
            m = np.ones((h, w), dtype=bool)
            if min(h, w) > 2 and rng.random() < .5:
                m[1:-1, 1:-1] = False
        else:
            m = _random_mask(rng, int(rng.integers(3, 7)), area)
        key = _canonical(m)
        mirror = _canonical(np.fliplr(m))
        if ((area is not None and int(m.sum()) != area) or
                key in forbidden or mirror in forbidden or
                (chiral and (key == mirror or len({_mask_key(x) for _, _, x in _forms(m)}) != 4)) or
                (diagonal and (_connected(m) or len({_mask_key(x) for _, _, x in _forms(m)}) != 4))):
            continue
        return m
    return None


def _objects(rng, recipe, max_test_size=3):
    objects, forbidden = [], set()

    def add(base, count):
        key = _canonical(base)
        forbidden.update((key, _canonical(np.fliplr(base))))
        turns = [int(rng.integers(4)) for _ in range(count)]
        if recipe == 'negative' and not objects:
            turns[1] = (turns[0] + 1) % 4
        for k in turns:
            objects.append(dict(mask=np.rot90(base, k).copy(), class_key=key))

    if recipe == 'negative':
        base = _sample_mask(rng, forbidden, chiral=True)
        if base is None:
            return None
        count = 4 if max_test_size == 4 else int(rng.choice([2, 3]))
        add(base, count)
        add(np.fliplr(base), 1)
        ordinary = _sample_mask(rng, forbidden, area=int(base.sum()))
        if ordinary is None:
            return None
        add(ordinary, 1)
        symmetric = _sample_mask(rng, forbidden, symmetric=True)
        if symmetric is None:
            return None
        add(symmetric, 2)
    elif recipe == 'consensus':
        base = _sample_mask(rng, forbidden, diagonal=True)
        if base is None:
            return None
        add(base, 3)
        ordinary = _sample_mask(rng, forbidden, symmetric=True)
        if ordinary is None:
            return None
        add(ordinary, 1)
    else:
        mode = int(rng.integers(8))
        sizes = ([1] * int(rng.integers(3, 7)) if mode == 0 else
                 [int(rng.integers(2, 5)) for _ in range(int(rng.integers(1, 3)))])
        if mode not in (0, 1):
            sizes += [1] * int(rng.integers(1, 4))
        while sum(sizes) > 8:
            sizes.pop()
        for count in sizes:
            base = _sample_mask(rng, forbidden)
            if base is None:
                return None
            add(base, count)
    if len(objects) < 8 and recipe in ('negative', 'consensus') and rng.random() < .4:
        base = _sample_mask(rng, forbidden)
        if base is not None:
            add(base, 1 if recipe == 'negative' else int(rng.choice([1, 2])))
    return objects


def _object_region(info):
    region = np.zeros(info['grid'].shape, dtype=bool)
    for o in info['objects']:
        h, w = o['mask'].shape
        region[o['top']:o['top'] + h, o['left']:o['left'] + w] |= o['mask']
    return _unview(region, info['side'])


def _different(info, output, alternative):
    return bool(np.any((_prediction(info, alternative) != output) & _object_region(info)))


def _multi_source(info):
    groups = defaultdict(list)
    for o in info['objects']:
        ref = _canonical(o['mask'])
        local = o['mask'] & ((np.arange(o['mask'].shape[0]) + o['top']) % 6 == 4)[:, None]
        hit = np.zeros(ref[:2], dtype=bool)
        for f, k, m in _forms(o['mask']):
            if _mask_key(m) == ref:
                hit |= _apply(local, f, k)
        groups[ref].append(hit)
    for hits in groups.values():
        union = np.logical_or.reduce(hits)
        if len(hits) >= 3 and all(np.any(union & ~hit) for hit in hits):
            return True
    return False


def _make_pair(rng, side, recipe='general', max_test_size=3):
    for attempt in range(320):
        objects = _objects(rng, recipe, max_test_size)
        if objects is None:
            continue
        depth = int(rng.integers(17, 31))
        span = depth if rng.random() < .22 else int(rng.integers(15, 31))
        length = int(rng.integers(1, 4))
        start = int(rng.integers(1, span - length))
        end = start + length
        palette = [int(c) for c in rng.permutation(10)]
        bg, marker, high = palette[:3]
        colors = palette[3:3 + int(rng.integers(1, 8))]
        for o in objects:
            o['color'] = int(rng.choice(colors))
        if recipe == 'negative':
            objects[0]['color'], objects[1]['color'], objects[-3]['color'] = palette[3], palette[4], palette[3]
        g = np.full((depth, span), bg, dtype=np.int8)
        g[0, start:end] = g[2, start:end] = marker
        g[4, start:end] = high
        occupied = set()
        success = True
        # Shuffle placement order; classes are not assigned spatial lanes.
        for idx in rng.permutation(len(objects)):
            obj = objects[int(idx)]
            h, w = obj['mask'].shape
            candidates = [(r, c) for r in range(1, depth - h + 1)
                          for c in range(span - w + 1) if c + w <= start or c >= end]
            blocked = {(r + dr, c + dc) for r, c in occupied for dr, dc in DIR8 + ((0, 0),)}
            local = [tuple(p) for p in np.argwhere(obj['mask']).tolist()]
            placed = False
            for ci in rng.permutation(len(candidates))[:200]:
                r, c = candidates[int(ci)]
                cells = {(r + y, c + x) for y, x in local}
                if cells & blocked:
                    continue
                obj['top'], obj['left'] = r, c
                occupied.update(cells)
                for p in cells:
                    g[p] = obj['color']
                placed = True
                break
            if not placed:
                success = False
                break
        if not success or np.sum(g == bg) <= g.size / 2:
            continue
        inp = _unview(g, side)
        try:
            info = _analyze(inp)
        except ValueError:
            # A chance arrangement of ordinary colored lines can resemble
            # a second head.  Reject that scene instead of emitting ambiguity.
            continue
        if len(info['objects']) != len(objects):
            continue
        output = _prediction(info)
        if recipe == 'negative':
            required = ('local_only', 'identity', 'reflection', 'color', 'one_registration',
                        'whole_object', 'all_cells', 'phase_zero', 'every_even', 'strip_only')
            if not all(_different(info, output, alt) for alt in required):
                continue
        if recipe == 'consensus':
            if (not _multi_source(info) or not _different(info, output, 'exactly_two')
                    or not _different(info, output, 'singleton_symmetry')
                    or not _different(info, output, 'four_connectivity')):
                continue
        return dict(input=inp, output=output)
    raise RuntimeError("bounded scene sampler exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _input_key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    ns = _count(num_test, 1, 4, 'num_test')
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()

    def append(split, side, recipe='general', maximum=3):
        for _ in range(20):
            pair = _make_pair(rng, side, recipe, maximum)
            key = _input_key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError('distinct-input budget exhausted')

    # Query multiplicity is visible in its input; training witnesses cover it.
    for _ in range(ns):
        append('test', (base + int(rng.choice([2, 3]))) % 4)
    max_test = max(max(_multiplicities(_analyze(p['input']))) for p in puzzle['test'])
    recipes = ['negative', 'consensus'] + ['general'] * (nt - 2)
    rng.shuffle(recipes)
    for i, recipe in enumerate(recipes):
        append('train', (base + i % 2) % 4, recipe, max_test)
    validate_generated_puzzle(puzzle)
    return puzzle


def _multiplicities(info):
    counts = defaultdict(int)
    for o in info['objects']:
        counts[_canonical(o['mask'])] += 1
    return list(counts.values())


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, train, test = set(), [], []
    alternatives = ('local_only', 'identity', 'reflection', 'color', 'exactly_two',
                    'whole_object', 'first_source', 'one_registration', 'all_cells',
                    'phase_zero', 'every_even', 'strip_only', 'singleton_symmetry', 'four_connectivity')
    excluded = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(out, _prediction(info)), 'incorrect echo or ruler pixels'
            assert not np.array_equal(inp, out)
            region = _object_region(info)
            # Support and unmarked colors are preserved exactly; only occupied
            # object pixels and the visible ruler corridor may change.
            canonical_region = _view(region, info['side'])
            go = _view(out, info['side'])
            corridor = np.zeros(inp.shape if info['side'] % 2 == 0 else inp.T.shape, dtype=bool)
            corridor[:, info['start']:info['end']] = True
            assert np.array_equal(go[~(canonical_region | corridor)], info['grid'][~(canonical_region | corridor)])
            assert np.all((out[region] == inp[region]) | (out[region] == info['highlight']))
            occupied = set()
            shared_observations = defaultdict(list)
            frequencies = defaultdict(int)
            for o in info['objects']:
                frequencies[_canonical(o['mask'])] += 1
            for o in info['objects']:
                cells = {(o['top'] + int(r), o['left'] + int(c)) for r, c in np.argwhere(o['mask'])}
                assert not cells & {(r + dr, c + dc) for r, c in occupied for dr, dc in DIR8}
                occupied |= cells
                assert _connected(o['mask'], DIR8)
                box = (slice(o['top'], o['top'] + o['mask'].shape[0]),
                       slice(o['left'], o['left'] + o['mask'].shape[1]))
                assert not np.any((go[box] == info['highlight']) & (info['grid'][box] == info['background']))
                marked = (go[box] == info['highlight']) & o['mask']
                local = o['mask'] & ((np.arange(o['mask'].shape[0]) + o['top']) % 6 == 4)[:, None]
                assert not np.any(local & ~marked), 'a local scan cut was lost'
                reference = _canonical(o['mask'])
                if frequencies[reference] == 1:
                    assert np.array_equal(marked, local), 'singleton has a nonlocal echo'
                else:
                    for f, k, m in _forms(o['mask']):
                        if _mask_key(m) == reference:
                            shared_observations[reference].append(_apply(marked, f, k))
            for markings in shared_observations.values():
                assert all(np.array_equal(markings[0], m) for m in markings), 'companions disagree after alignment'
            (train if split == 'train' else test).append(info)
            if split == 'train':
                excluded.update(alt for alt in alternatives if _different(info, out, alt))
    assert excluded == set(alternatives), 'training leaves a documented shortcut unresolved'
    assert any(_multi_source(i) for i in train), 'no complementary cuts in training'
    assert any(3 in _multiplicities(i) for i in train), 'no triplet witness'
    assert max(max(_multiplicities(i)) for i in train) >= max(max(_multiplicities(i)) for i in test)
    train_sides, test_sides = {i['side'] for i in train}, {i['side'] for i in test}
    assert len(train_sides) == 2 and train_sides.isdisjoint(test_sides), 'missing orientation shift'
    a, b = sorted(train_sides)
    assert (b - a) % 4 in (1, 3), 'training sides must be adjacent'


def self_test():
    """Bounded 42-seed variation check plus counts, equivariance, and corruption."""
    shapes, structures, counts, multiplicities, backgrounds, heads, areas = (set() for _ in range(7))
    palettes, diagonals, symmetric = set(), False, False
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_input_key(p['input']) for s in ('train', 'test') for p in puzzle[s])
        fingerprints.add(fingerprint)
        if seed in (0, 41):
            again = generate_puzzle(seed)
            assert fingerprint == tuple(_input_key(p['input']) for s in ('train', 'test') for p in again[s])
        for split in ('train', 'test'):
            for p in puzzle[split]:
                a, b = p['input'], p['output']
                info = _analyze(a)
                shapes.add(a.shape)
                backgrounds.add(info['background'])
                heads.add((info['side'], info['end'] - info['start']))
                counts.add(len(info['objects']))
                multiplicities.update(_multiplicities(info))
                palettes.add(tuple(sorted({o['color'] for o in info['objects']})))
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.ravel())))
                for o in info['objects']:
                    areas.add(int(o['mask'].sum()))
                    diagonals |= not _connected(o['mask'])
                    symmetric |= len({_mask_key(m) for _, _, m in _forms(o['mask'])}) < 4
                if seed < 3:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(structures) >= 190
    assert {1, 2, 3, 4} <= multiplicities
    assert len(counts) >= 5 and len(palettes) >= 20
    assert backgrounds == set(range(10)) and len(heads) == 12
    assert min(areas) <= 3 and max(areas) >= 20 and diagonals and symmetric
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345 + nt * 10 + ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    broken = generate_puzzle(13)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corruption accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
