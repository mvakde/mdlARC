"""Mirror-orbit ballots: repair replicas, count independent ballots, fill a stencil.

Only NumPy and the standard library are used. No parent generator is imported.
Background is 0, missing pixels are 9; the intact stencil defines both folds.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import numpy as np


def _norm(cells):
    cells = tuple(cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-y), int(c-x)) for r, c in cells))


@lru_cache(maxsize=2048)
def _variants(shape):
    result = set()
    for flipped in (False, True):
        current = _norm((r, -c if flipped else c) for r, c in shape)
        for _ in range(4):
            result.add(current)
            current = _norm((c, -r) for r, c in current)
    return tuple(sorted(result))


def _canon(cells):
    return min(_variants(_norm(cells)))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        stack, cells = [point], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _stencil(g):
    candidates = []
    for color in range(1, 9):
        cells = np.argwhere(g == color)
        if not len(cells):
            continue
        y, x = cells.min(axis=0)
        b, d = cells.max(axis=0)
        box = g[y:b+1, x:d+1]
        if min(box.shape) < 5:
            continue
        if (np.all(box[0] == color) and np.all(box[-1] == color)
                and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)
                and set(map(int, np.unique(box))) == {0, color}
                and len(_components(box == color)) == 1):
            candidates.append((color, int(y), int(x), int(b), int(d)))
    assert len(candidates) == 1, 'need one intact rectangular stencil'
    return candidates[0]


def _point_orbit(p, shape, axes):
    r, c = p
    a, b = axes
    rows = {r, a-r} & set(range(shape[0]))
    cols = {c, b-c} & set(range(shape[1]))
    return tuple(sorted((y, x) for y in rows for x in cols))


def _images(cells, axes):
    a, b = axes
    return tuple(sorted({tuple(sorted((a-r if hr else r, b-c if vr else c)
                                     for r, c in cells))
                         for hr in (False, True) for vr in (False, True)}))


def _repair(g, bbox):
    _, y, x, bottom, right = bbox
    axes = y+bottom, x+right
    out = g.copy()
    exterior = np.ones(g.shape, dtype=bool)
    exterior[y:bottom+1, x:right+1] = False
    seen = set()
    diagonal = erased = 0
    for r, c in zip(*np.nonzero(exterior)):
        p = int(r), int(c)
        if p in seen:
            continue
        orbit = _point_orbit(p, g.shape, axes)
        assert all(exterior[q] for q in orbit)
        seen.update(orbit)
        colors = {int(g[q]) for q in orbit if g[q] != 9}
        assert len(colors) == 1, 'empty or conflicting exterior reflection orbit'
        color = next(iter(colors))
        for q in orbit:
            if g[q] == 9:
                erased += 1
                rr, cc = axes[0]-q[0], axes[1]-q[1]
                direct = [(rr, q[1]), (q[0], cc)]
                diagonal += int(not any(0 <= s < g.shape[0] and 0 <= t < g.shape[1]
                                        and (s, t) != q and g[s, t] != 9
                                        for s, t in direct))
                out[q] = color
    return out, exterior, axes, erased, diagonal


def _analyze(input_grid, *, quotient=True, equivalence='shape', repair=True):
    g = _grid(input_grid)
    bbox = _stencil(g)
    carrier, y, x, bottom, right = bbox
    restored, exterior, axes, erased, diagonal = _repair(g, bbox)
    if not repair:
        restored = g.copy()
        restored[restored == 9] = 0
    objects = []
    for color in range(1, 9):
        if color == carrier:
            continue
        objects.extend((color, cells) for cells in _components((restored == color) & exterior))
    assert objects, 'need exterior ballots'
    by_cells = {cells: color for color, cells in objects}
    def signature(cells):
        if equivalence == 'area':
            return len(cells)
        if equivalence == 'box':
            return tuple(sorted(_size(_norm(cells))))
        if equivalence == 'all':
            return 'all'
        if equivalence == 'translation':
            return _norm(cells)
        return _canon(cells)
    ballots = {}
    multiplicities = []
    for color, cells in objects:
        images = _images(cells, axes)
        in_bounds = [v for v in images if all(0 <= r < g.shape[0] and 0 <= c < g.shape[1]
                                            for r, c in v)]
        if repair:
            assert all(by_cells.get(v) == color for v in in_bounds), 'incomplete component orbit'
        identity = min(images) if quotient else cells
        if identity not in ballots:
            ballots[identity] = color, cells
            multiplicities.append(len(in_bounds))
    counts = defaultdict(Counter)
    for color, cells in ballots.values():
        counts[signature(cells)][color] += 1
    box = g[y:bottom+1, x:right+1]
    cavities = _components(box == 0)
    cavity_keys = {signature(cells) for cells in cavities}
    if equivalence == 'shape' and repair:
        assert set(counts) == cavity_keys, 'every vote and cavity must have a shape match'
    winners = {}
    for key in cavity_keys:
        assert counts[key], 'unsupported cavity'
        high = max(counts[key].values())
        leaders = [c for c, n in counts[key].items() if n == high]
        assert len(leaders) == 1, 'no unique plurality'
        winners[key] = leaders[0]
    out = box.copy()
    for cells in cavities:
        for p in cells:
            out[p] = winners[signature(cells)]
    return out, dict(bbox=bbox, axes=axes, restored=restored, exterior=exterior,
                     objects=objects, ballots=ballots, counts=dict(counts), winners=winners,
                     cavities=cavities, multiplicities=multiplicities,
                     erased=erased, diagonal=diagonal)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive both folds, recovered ballots, plurality colors, and crop from input."""
    return _analyze(input_grid)[0]


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _pool():
    """Exhaustively grow small polyominoes, rather than a fixed object catalog."""
    level = {((0, 0),)}
    pool = set()
    for area in range(1, 8):
        pool.update(s for s in level if _hole_free(s))
        next_level = set()
        if area == 7:
            break
        for s in sorted(level):
            frontier = {(r+dr, c+dc) for r, c in s
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - set(s)
            for p in sorted(frontier):
                t = _canon(s+(p,))
                if max(_size(t)) <= 3:
                    next_level.add(t)
        level = next_level
    return tuple(sorted(pool))


_ANCHORS = (
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
)


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _positions(available, s):
    h, w = _size(s)
    nr, nc = available.shape[0]-h+1, available.shape[1]-w+1
    if min(nr, nc) <= 0:
        return []
    valid = np.ones((nr, nc), dtype=bool)
    for r, c in s:
        valid &= available[r:r+nr, c:c+nc]
    return np.argwhere(valid)


def _pack(available, shape, rng):
    options = _variants(shape)
    for i in rng.permutation(len(options)):
        s = options[int(i)]
        positions = _positions(available, s)
        if len(positions):
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((y+r, x+c) for r, c in s)
            _block(available, cells)
            return cells
    return None


def _make_pair(rng, shapes, anchor, rich, hard, forbidden_winner=None, axis_wins=True,
               fallback_only=False):
    anchor_key = _canon(anchor)
    palette = list(map(int, rng.permutation(np.arange(1, 9))))
    carrier, colors = palette[0], palette[1:]
    winners = {s: int(_choice(rng, colors)) for s in shapes}
    if winners[anchor_key] == forbidden_winner:
        winners[anchor_key] = _choice(rng, [c for c in colors if c != forbidden_winner])
    # Same-area/box classes must visibly get different answers.
    winners[shapes[1]] = _choice(rng, [c for c in colors if c != winners[anchor_key]])
    axis_color = winners[anchor_key] if axis_wins else _choice(
        rng, [c for c in colors if c != winners[anchor_key]])
    specifications = []
    for i, s in enumerate(shapes):
        if i == 0:
            if axis_wins:
                losers = [c for c in colors if c != axis_color]
                rng.shuffle(losers)
                specifications.extend((s, color) for color in losers[:2 if rich else 1]
                                      for _ in range(2))
            else:
                specifications.extend((s, winners[s]) for _ in range(4))
                if rich:
                    third = _choice(rng, [c for c in colors if c not in (axis_color, winners[s])])
                    specifications.extend((s, third) for _ in range(2))
            continue
        counts = _choice(rng, ((1,), (2, 1), (2, 1, 1))) if i == 1 and axis_wins else (1,)
        if i == 1 and rich and len(shapes) == 4 and len(counts) == 3:
            counts = (2, 1)
        losers = [c for c in colors if c != winners[s]]
        rng.shuffle(losers)
        for j, (color, n) in enumerate(zip([winners[s]]+losers, counts)):
            specifications.extend((s, color) for _ in range(n))
    assert len(specifications) <= 9
    ah, aw = _size(anchor)
    layout = None
    for attempt in ((48,) if fallback_only else range(49)):
        fallback = attempt == 48
        sh = 5 if ah == 3 or fallback else int(_choice(rng, (5, 7, 9, 11)))
        sw = max(9, 4*len(shapes)+1) if fallback else int(_choice(rng, (9, 11, 13, 15, 17, 19)))
        if sw < 4*len(shapes)+1 and sh == 5:
            sw = 4*len(shapes)+1
        # Cavities occupy arbitrary legal coordinates, including repetitions.
        free = np.ones((sh, sw), dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        holes = []
        if fallback:
            # Guaranteed fit: each shape fits its own 3x3 interior region.
            for i, s in enumerate(shapes):
                raw = _choice(rng, _variants(s))
                holes.append((s, tuple((1+r, 1+4*i+c) for r, c in raw)))
        else:
            for s in sorted(shapes, key=lambda s: (-max(_size(s)), -len(s), s)):
                cells = _pack(free, s, rng)
                if cells is None:
                    break
                holes.append((s, cells))
            if len(holes) != len(shapes):
                continue
            for _ in range(int(rng.integers(0, 5))):
                s = _choice(rng, shapes)
                cells = _pack(free, s, rng)
                if cells is not None:
                    holes.append((s, cells))
        minimum = sh + 2*(3*ah+3)
        if minimum > 30:
            continue
        heights = list(range(minimum, 31, 2))
        h = 29 if fallback else int(_choice(rng, heights))
        w = 29 if fallback else int(_choice(rng, (25, 27, 29)))
        if w < sw+4:
            continue
        y, x = (h-sh)//2, (w-sw)//2
        axes = h-1, w-1
        free = np.ones((h, w), dtype=bool)
        free[max(0, y-1):y+sh+1, max(0, x-1):x+sw+1] = False
        group_specs = []
        gap_slack = y-(3*ah+3)
        start = 0 if fallback else int(rng.integers(gap_slack+1))
        for i in range(3):
            r0 = start+i*(ah+1)
            c0 = (w-aw)//2
            cells = tuple((r0+r, c0+c) for r, c in anchor)
            images = _images(cells, axes)
            assert len(images) == 2
            for image in images:
                _block(free, image)
            group_specs.append((axis_color, images))
        # Off-axis ballots have four components. Work in one quadrant and reflect.
        free[h//2:, :] = False
        free[:, w//2:] = False
        ordered = [specifications[int(i)] for i in rng.permutation(len(specifications))]
        ordered.sort(key=lambda sc: (-max(_size(sc[0])), -len(sc[0])))
        success = True
        for i, (s, color) in enumerate(ordered):
            if fallback:
                # Nine disjoint 3x3 slots lie above a 5-row central stencil.
                # The ordinary maximum is nine off-axis ballots.
                r0, c0 = 4*(i//3), 4*(i % 3)
                raw = _choice(rng, _variants(s))
                cells = tuple((r0+r, c0+c) for r, c in raw)
                if not all(free[p] for p in cells):
                    success = False
                    break
                _block(free, cells)
            else:
                cells = _pack(free, s, rng)
                if cells is None:
                    success = False
                    break
            images = _images(cells, axes)
            assert len(images) == 4
            group_specs.append((color, images))
        if success:
            layout = h, w, y, x, sh, sw, holes, group_specs, axes
            break
    assert layout is not None, 'bounded layout construction failed'
    h, w, y, x, sh, sw, holes, groups, axes = layout
    clean = np.zeros((h, w), dtype=np.int8)
    box = np.full((sh, sw), carrier, dtype=np.int8)
    answer = box.copy()
    for s, cells in holes:
        for p in cells:
            box[p], answer[p] = 0, winners[s]
    assert len(_components(box == carrier)) == 1
    clean[y:y+sh, x:x+sw] = box
    for color, images in groups:
        for cells in images:
            for p in cells:
                assert clean[p] == 0
                clean[p] = color
    inp = clean.copy()
    protected = np.zeros(clean.shape, dtype=bool)
    visible = np.zeros(clean.shape, dtype=bool)
    # Complementary fragments: neither replica of two winning ballots is whole.
    # Every erased pixel still has a witness, but skipping repair loses votes.
    for color, images in groups[:2]:
        first = images[0][0]
        first_orbit = _point_orbit(first, clean.shape, axes)
        second = next(p for p in images[1] if p not in first_orbit)
        protected[first] = protected[second] = True
    for color, images in groups:
        if len(images) == 4 and color != winners[anchor_key] and _canon(images[0]) == anchor_key:
            for p in images[0]:
                visible[p] = True
    # Whole replicas disappear; one survivor may supply only a diagonal mirror.
    four_groups = [images for color, images in groups
                   if len(images) == 4 and _canon(images[0]) != anchor_key]
    images = _choice(rng, four_groups)
    survivor = int(rng.integers(4))
    erased_images = [j for j in range(4) if j != survivor]
    if not hard:
        erased_images = list(map(int, rng.choice(erased_images, size=2, replace=False)))
    for j in erased_images:
        for p in images[j]:
            inp[p] = 9
            protected[p] = True
    # Distributed damage and, for held-out cases, an unfamiliar strip/step patch.
    mask = rng.random(clean.shape) < float(rng.uniform(.025, .10))
    if hard:
        rh, cw = int(rng.integers(3, 8)), int(rng.integers(1, 5))
        r0, c0 = int(rng.integers(h-rh+1)), int(rng.integers(w-cw+1))
        for dr in range(rh):
            mask[r0+dr, c0:c0+max(1, cw-dr % 2)] = True
    mask[y:y+sh, x:x+sw] = False
    mask[visible] = False
    mask |= protected
    seen = set()
    for r in range(h):
        for c in range(w):
            if y <= r < y+sh and x <= c < x+sw or (r, c) in seen:
                continue
            orbit = _point_orbit((r, c), clean.shape, axes)
            seen.update(orbit)
            if all(mask[p] for p in orbit):
                choices = [p for p in orbit if not protected[p]]
                assert choices
                mask[_choice(rng, choices)] = False
    inp[mask] = 9
    # Asymmetric blank fringes produce visibly encoded off-center folds.
    top = int(rng.integers(min(2, 30-h)+1))
    bottom = int(rng.integers(min(2, 30-h-top)+1))
    left = int(rng.integers(min(2, 30-w)+1))
    right = int(rng.integers(min(2, 30-w-left)+1))
    inp = np.pad(inp, ((top, bottom), (left, right)), constant_values=0)
    turns = int(rng.integers(4))
    inp, answer = np.rot90(inp, turns).copy(), np.rot90(answer, turns).copy()
    if rng.random() < .5:
        inp, answer = inp[:, ::-1].copy(), answer[:, ::-1].copy()
    return {'input': inp, 'output': answer}


def _count_argument(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count_argument(num_train, 2, 10, 'num_train')
    ns = _count_argument(num_test, 1, 4, 'num_test')
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    anchor = _choice(rng, _ANCHORS)
    a = _canon(anchor)
    pool = _pool()
    same_box = [s for s in pool if s != a and len(s) == len(a)
                and sorted(_size(s)) == sorted(_size(a))]
    companion = _choice(rng, same_box)
    other = [s for s in pool if s not in (a, companion)]
    novel = _choice(rng, [s for s in other if len(s) >= 3])
    training_pool = [s for s in other if s != novel]
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_winner = None
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            shapes = [a, companion]
            if split == 'test':
                shapes.append(novel)
            elif i >= 2 and rng.random() < .55:
                shapes.append(_choice(rng, training_pool))
            if len(shapes) == 3 and rng.random() < .25:
                candidates = [s for s in training_pool if s not in shapes]
                shapes.append(_choice(rng, candidates))
            rich = split == 'train' and i == 1
            rich |= split == 'test' and rng.random() < .25
            axis_wins = i == 0 if split == 'train' and i < 2 else bool(rng.integers(2))
            for _ in range(8):
                pair = _make_pair(rng, shapes, anchor, rich, split == 'test' or i == 1,
                                  previous_winner if split == 'train' and i == 1 else None,
                                  axis_wins=axis_wins)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('distinct input retries exhausted')
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and i == 0:
                previous_winner = _analyze(pair['input'])[1]['winners'][a]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_fails(pair, **kwargs):
    try:
        predicted, _ = _analyze(pair['input'], **kwargs)
        return not np.array_equal(predicted, pair['output'])
    except (AssertionError, KeyError):
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, data = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(g)
            assert key not in seen
            seen.add(key)
            expected, d = _analyze(g)
            assert np.array_equal(out, expected), 'wrong cavity fill or crop'
            carrier, y, x, b, r = d['bbox']
            box = g[y:b+1, x:r+1]
            assert np.all(out[box == carrier] == carrier)
            assert not np.any((out == 0) | (out == 9))
            assert d['erased'] > 0
            assert set(d['multiplicities']) <= {1, 2, 4}
            # Adjacent colors cannot masquerade as separate objects.
            assert len(_components((d['restored'] != 0) & d['exterior'])) == len(d['objects'])
            data[split].append(d)
    train = puzzle['train']
    assert any(_alternative_fails(p, quotient=False) for p in train), 'replica-counting shortcut survives'
    assert any(_alternative_fails(p, equivalence='area') for p in train), 'area shortcut survives'
    assert any(_alternative_fails(p, equivalence='box') for p in train), 'box shortcut survives'
    assert any(_alternative_fails(p, equivalence='all') for p in train), 'shape-blind shortcut survives'
    assert any(_alternative_fails(p, repair=False) for p in train), 'erasure-ignoring shortcut survives'
    assert any(d['diagonal'] > 0 for d in data['train']), 'both folds must be taught'
    axis_outcomes = set()
    for d in data['train']:
        for (color, cells), m in zip(d['ballots'].values(), d['multiplicities']):
            if m == 2:
                axis_outcomes.add(color == d['winners'][_canon(cells)])
    assert axis_outcomes == {False, True}, 'always choosing a fold color survives'
    assert any(len(v) >= 3 and 2*max(v.values()) <= sum(v.values())
               for d in data['train'] for v in d['counts'].values()), 'plurality evidence absent'
    history = defaultdict(set)
    for d in data['train']:
        for s, color in d['winners'].items():
            history[s].add(color)
    assert any(len(v) > 1 for v in history.values()), 'fixed shape-color map not contradicted'


def self_test():
    """42 sequential seeds; retain metrics, not a corpus of reconstructed grids."""
    shapes, crops, vocab, structures, palettes, masks, offsets = set(), set(), set(), set(), set(), set(), set()
    classes, cavities, multiplicities, ballots, carriers = set(), set(), set(), set(), set()
    signatures, diagonal_tests, novel_tests, pair_count = set(), 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = sha256()
        training_shapes = set()
        for split in ('train', 'test'):
            for pair in p[split]:
                g = pair['input']
                _, d = _analyze(g)
                pair_count += 1
                shapes.add(g.shape)
                crops.add(pair['output'].shape)
                vocab.update(d['counts'])
                if split == 'train':
                    training_shapes.update(d['counts'])
                else:
                    diagonal_tests += d['diagonal'] > 0
                    novel_tests += bool(set(d['counts']) - training_shapes)
                classes.add(len(d['counts']))
                cavities.add(len(d['cavities']))
                multiplicities.update(d['multiplicities'])
                ballots.update(tuple(sorted(c.values(), reverse=True)) for c in d['counts'].values())
                carriers.add(d['bbox'][0])
                palettes.add(tuple(map(int, np.unique(g))))
                masks.add((int(np.sum(g == 9)), len(_components(g == 9))))
                offsets.add((d['axes'][0]-g.shape[0]+1, d['axes'][1]-g.shape[1]+1))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in g.flat)
                structures.add((g.shape, sha256(normalized).digest()))
                signature.update(g.tobytes())
        signatures.add(signature.digest())
    assert len(signatures) == 42
    assert len(shapes) >= 20 and len(crops) >= 20 and len(vocab) >= 15
    assert len(structures) >= 200 and len(palettes) >= 35 and len(masks) >= 100
    assert len(offsets) >= 8 and classes == {2, 3, 4}
    assert len(cavities) >= 5 and multiplicities == {2, 4}
    assert {(3, 2), (4, 3, 2), (1,)} <= ballots
    assert carriers == set(range(1, 9)) and diagonal_tests == novel_tests == 42
    # Exercise the finite-search termination construction, including its maximum
    # nine off-axis ballots, for all three anchors and both election outcomes.
    for index, anchor in enumerate(_ANCHORS):
        a = _canon(anchor)
        companion = next(s for s in _pool() if s != a and len(s) == len(a)
                         and sorted(_size(s)) == sorted(_size(a)))
        extras = [s for s in _pool() if s not in (a, companion) and len(s) >= 6][:2]
        for axis_wins in (False, True):
            pair = _make_pair(np.random.default_rng(4000+index), [a, companion]+extras,
                              anchor, True, True, axis_wins=axis_wins, fallback_only=True)
            assert np.array_equal(solve(pair['input']), pair['output'])
    # A rule-level fixture outside the sampler's geometry: between-cell folds,
    # an eight-cell 3x4 bracket, and a repairable nonzero unmatched fringe.
    extended = np.zeros((20, 24), dtype=np.int8)
    extended[7:13, 8:16] = 5
    bracket = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 3))
    domino = ((0, 0), (1, 0))
    expected = extended[7:13, 8:16].copy()
    for shape, offset, color in ((bracket, (1, 1), 1), (domino, (1, 6), 3)):
        for r, c in shape:
            p0 = offset[0]+r, offset[1]+c
            extended[7+p0[0], 8+p0[1]] = 0
            expected[p0] = color
    for shape, offset, color in ((bracket, (0, 0), 1), (bracket, (0, 6), 1),
                                 (bracket, (4, 0), 2), (domino, (4, 6), 3)):
        cells = tuple((offset[0]+r, offset[1]+c) for r, c in shape)
        for image in _images(cells, (19, 23)):
            for p0 in image:
                extended[p0] = color
    for p0 in _point_orbit((0, 0), extended.shape, (19, 23))[:-1]:
        extended[p0] = 9
    extended = np.pad(extended, ((3, 0), (0, 0)), constant_values=0)
    extended[1, 11], extended[1, 12] = 9, 3
    assert np.array_equal(solve(extended), expected)
    p = generate_puzzle(73, 2, 1)
    q = generate_puzzle(73, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for nt, ns in ((10, 4), (10, 1), (5, 4)):
        validate_generated_puzzle(generate_puzzle(9173, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The solver is input-only, rotation-covariant, and does not mutate its input.
    pair = p['test'][0]
    original = pair['input'].copy()
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(original, turns)), np.rot90(pair['output'], turns))
    assert np.array_equal(pair['input'], original)
    broken = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong carrier accepted')
    # Contradictory visible mirrors are rejected rather than silently repaired.
    _, d = _analyze(original)
    contradictory = original.copy()
    p0 = next(p for color, cells in d['objects'] for p in cells
              if original[p] not in (0, 9)
              and sum(original[q] != 9 for q in _point_orbit(p, original.shape, d['axes'])) >= 2)
    contradictory[p0] = next(c for c in range(1, 9)
                            if c not in (int(original[p0]), d['bbox'][0]))
    try:
        solve(contradictory)
    except AssertionError:
        pass
    else:
        raise AssertionError('contradictory mirror witnesses accepted')
    return dict(seeds=42, pairs=pair_count, distinct_input_shapes=len(shapes),
                distinct_output_shapes=len(crops), shape_classes=len(vocab),
                normalized_input_structures=len(structures), mask_descriptors=len(masks),
                axis_offsets=len(offsets), test_diagonal_repairs=diagonal_tests,
                test_novel_shape_examples=novel_tests)
