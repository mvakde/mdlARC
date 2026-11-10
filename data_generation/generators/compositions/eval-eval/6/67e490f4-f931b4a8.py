"""Self-stenciled ballots: shape elections choose clipped, native-size textures.

No parent code or examples are imported. All decoding uses the current input.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _norm(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _rot(cells):
    return _norm([(c, -r) for r, c in cells])


def _mirror(cells):
    return _norm([(r, -c) for r, c in cells])


@lru_cache(maxsize=4096)
def _rotations(cells):
    result = set()
    for _ in range(4):
        result.add(cells)
        cells = _rot(cells)
    return tuple(sorted(result))


@lru_cache(maxsize=4096)
def _variants(cells):
    return tuple(sorted(set(_rotations(cells)) | set(_rotations(_mirror(cells)))))


def _canonical(cells):
    return min(_variants(_norm(cells)))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _mask(cells):
    a = np.zeros(_size(cells), dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _components(mask):
    left = set(zip(*np.nonzero(mask)))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in left:
                    left.remove(q)
                    todo.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _hole_free(cells):
    a = np.pad(_mask(cells), 1)
    return len(_components(~a)) == 1


def _separated(groups):
    owners = {p: i for i, group in enumerate(groups) for p in group}
    for (r, c), owner in owners.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if owners.get((r+dr, c+dc), owner) != owner:
                    return False
    return True


@lru_cache(maxsize=1)
def _small_pool():
    """Enumerate free connected polyominoes, instead of fixing an anchor list."""
    current = {((0, 0),)}
    pool = set(current)
    for area in range(2, 7):
        grown = set()
        for shape in sorted(current):
            cells = set(shape)
            edge = {(r+dr, c+dc) for r, c in cells
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells
            for p in sorted(edge):
                s = _canonical(cells | {p})
                if max(_size(s)) <= 4 and _hole_free(s):
                    grown.add(s)
        current = grown
        pool.update(current)
    return tuple(sorted(pool))


def _pool(rng):
    shapes = set(_small_pool())
    for _ in range(24):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(6, 9))-1):
            edge = sorted({(r+dr, c+dc) for r, c in cells
                           for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
            cells.add(edge[int(rng.integers(len(edge)))])
        s = _canonical(cells)
        if max(_size(s)) <= 4 and _hole_free(s):
            shapes.add(s)
    return tuple(sorted(shapes))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the outer canvas border must show background")
    candidates = []
    for ink in sorted(set(map(int, np.unique(a))) - {bg}):
        for cells in _components(a == ink):
            y, x = min(r for r, c in cells), min(c for r, c in cells)
            h = max(r for r, c in cells)-y+1
            w = max(c for r, c in cells)-x+1
            if min(h, w) < 4 or y < 3 or x < 3:
                continue
            box = a[y:y+h, x:x+w]
            if (np.all(box[0] == ink) and np.all(box[-1] == ink)
                    and np.all(box[:, 0] == ink) and np.all(box[:, -1] == ink)
                    and set(map(int, np.unique(box))) == {bg, ink}
                    and np.count_nonzero(a == ink) == len(cells)):
                candidates.append((ink, y, x, box.copy()))
    if len(candidates) != 1:
        raise ValueError("expected exactly one connected rectangular plate")
    carrier, y, x, box = candidates[0]
    h, w = box.shape
    # The narrow strips two cells outside the plate are positional counters.
    top, side = a[y-2, x:x+w], a[y:y+h, x-2]
    sy, sx = int(np.sum(side != bg)), int(np.sum(top != bg))
    if not (1 <= sy <= 4 and 1 <= sx <= 4 and h*sy <= 30 and w*sx <= 30):
        raise ValueError("illegal rail counts / enlarged dimensions")
    if np.any(a[y-1, x:x+w] != bg) or np.any(a[y:y+h, x-1] != bg):
        raise ValueError("rails must be separated from the plate")
    if carrier in top or carrier in side:
        raise ValueError("rail inks cannot be the carrier")
    cavities = _components(box == bg)
    if not _separated(cavities):
        raise ValueError("distinct cavities need a one-cell separation")
    for cells in cavities:
        if not _hole_free(_norm(cells)):
            raise ValueError("cavities cannot enclose carrier islands")
    exterior = a.copy()
    exterior[y:y+h, x:x+w] = bg
    exterior[y-2, x:x+w] = bg
    exterior[y:y+h, x-2] = bg
    if np.any(exterior[y-3:y+h+1, x-3:x+w+1] != bg):
        raise ValueError("ballots cannot enter the plate/rail reservation")
    objects, votes, raws = [], defaultdict(Counter), defaultdict(list)
    # Count connected objects regardless of their area, then check monochromy.
    for cells in _components(exterior != bg):
        colors = {int(exterior[p]) for p in cells}
        if len(colors) != 1:
            raise ValueError("ballots must be isolated monochromatic objects")
        color = colors.pop()
        raw = _norm(cells)
        if not _hole_free(raw):
            raise ValueError("ballot shapes must have no internal hole")
        signature = _canonical(raw)
        votes[signature][color] += 1
        raws[signature].append((color, raw))
        objects.append((color, cells))
    if not _separated([cells for color, cells in objects]):
        raise ValueError("distinct ballots need a one-cell separation")
    shapes = {_canonical(cells) for cells in cavities}
    if set(votes) != shapes:
        raise ValueError("each cavity class must have votes, with no extra classes")
    winners = {}
    for shape, counts in votes.items():
        best = max(counts.values())
        colors = [c for c, n in counts.items() if n == best]
        if len(colors) != 1:
            raise ValueError("ties are outside the family")
        winners[shape] = colors[0]
    return dict(background=bg, carrier=carrier, origin=(y, x), box=box,
                scale=(sy, sx), cavities=cavities, votes=dict(votes),
                winners=winners, objects=objects, raw_votes=dict(raws),
                rails=(side.copy(), top.copy()))


def _render(data, mode="correct"):
    """Input-reading renderer, also used to measure explicit alternative rules."""
    box = data['box']
    sy, sx = data['scale']
    out = np.repeat(np.repeat(box, sy, axis=0), sx, axis=1)
    all_votes = Counter()
    for counts in data['votes'].values():
        all_votes.update(counts)
    global_winner = min(all_votes, key=lambda c: (-all_votes[c], c))
    filled = box.copy()
    for cells in data['cavities']:
        shape = _canonical(cells)
        winner = data['winners'][shape]
        if mode == 'global_vote':
            winner = global_winner
        if mode in ('area_vote', 'bbox_vote'):
            key = len(shape) if mode == 'area_vote' else tuple(sorted(_size(shape)))
            counts = Counter()
            for s, v in data['votes'].items():
                other = len(s) if mode == 'area_vote' else tuple(sorted(_size(s)))
                if other == key:
                    counts.update(v)
            winner = min(counts, key=lambda c: (-counts[c], c))
        if mode in ('translation_vote', 'rotation_vote'):
            raw = _norm(cells)
            matches = (raw,) if mode == 'translation_vote' else _rotations(raw)
            counts = Counter(c for c, s in data['raw_votes'][shape] if s in matches)
            if not counts or len([c for c in counts if counts[c] == max(counts.values())]) != 1:
                raise ValueError("alternative cannot elect a unique color")
            winner = max(counts, key=counts.get)
        for p in cells:
            filled[p] = winner
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        mask = _mask(_norm(cells))
        mh, mw = mask.shape
        large = np.repeat(np.repeat(mask, sy, axis=0), sx, axis=1)
        motif = _mask(shape) if mode == 'canonical_motif' else mask
        rows, cols = np.arange(mh*sy)[:, None], np.arange(mw*sx)[None, :]
        if mode == 'global_phase':
            rows, cols = rows+r0*sy, cols+c0*sx
        if mode == 'phase_rows':
            rows = rows+1
        if mode == 'phase_columns':
            cols = cols+1
        tiled = motif[rows % motif.shape[0], cols % motif.shape[1]]
        if mode == 'solid':
            tiled = np.ones_like(tiled)
        active = tiled if mode == 'no_clip' else tiled & large
        region = out[r0*sy:(r0+mh)*sy, c0*sx:(c0+mw)*sx]
        region[active] = winner
    if mode == 'plate_tile':
        return np.tile(filled, (sy, sx))
    if mode == 'ignore_votes':
        return np.repeat(np.repeat(box, sy, axis=0), sx, axis=1)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Elect each shape's ink and clip its native-period texture to its enlargement."""
    return _render(_decode(input_grid))


def _positions(available, shape):
    sh, sw = _size(shape)
    h, w = available.shape
    if sh > h or sw > w:
        return []
    valid = np.ones((h-sh+1, w-sw+1), dtype=bool)
    for r, c in shape:
        valid &= available[r:r+h-sh+1, c:c+w-sw+1]
    return np.argwhere(valid)


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _place(available, variants, rng):
    for i in rng.permutation(len(variants)):
        shape = variants[int(i)]
        origins = _positions(available, shape)
        if len(origins):
            y, x = map(int, origins[int(rng.integers(len(origins)))])
            cells = tuple((r+y, c+x) for r, c in shape)
            _block(available, cells)
            return cells
    return None


def _ballot(rng):
    return _choose(rng, ((1,), (2,), (3,), (2, 1), (3, 1), (3, 2),
                         (3, 2, 2), (2, 1, 1), (4, 2, 1), (4, 3)))


def _make_pair(rng, shapes, scale, witness=False, changed=None):
    """Sample a whole plate and dispersed ballots; no fixed cavity slot lattice."""
    bg, carrier = map(int, rng.permutation(10)[:2])
    palette = [c for c in range(10) if c not in (bg, carrier)]
    rng.shuffle(palette)
    winners = {s: int(_choose(rng, palette)) for s in shapes}
    if witness:
        winners[shapes[1]] = int(_choose(rng, [c for c in palette if c != winners[shapes[0]]]))
    if changed is not None:
        winners[shapes[0]] = int(_choose(rng, [c for c in palette if c != changed]))
        if witness:
            winners[shapes[1]] = int(_choose(rng, [c for c in palette if c != winners[shapes[0]]]))
    counts = {s: _ballot(rng) for s in shapes}
    if witness:
        counts[shapes[0]], counts[shapes[1]] = (3, 2, 2), (2, 1)
    # Retain teaching ballots while keeping exterior packing comfortably bounded.
    def cost(s):
        h, w = _size(s)
        return (h+1)*(w+1)*sum(counts[s])
    for s in sorted(shapes, key=cost, reverse=True):
        if sum(cost(t) for t in shapes) <= 270:
            break
        if not witness or s not in shapes[:2]:
            counts[s] = (1,)
    sy, sx = scale
    for attempt in range(90):
        hmax, wmax = min(13, 30//sy), min(13, 30//sx)
        h = int(rng.integers(6 if witness else 5, hmax+1))
        w = int(rng.integers(6 if witness else 5, wmax+1))
        avail = np.ones((h, w), dtype=bool)
        avail[[0, -1]] = False
        avail[:, [0, -1]] = False
        placements, first_raw = [], None
        failed = False
        order = sorted(shapes, key=lambda s: (max(_size(s)), len(s)), reverse=True)
        for s in order:
            cells = _place(avail, _variants(s), rng)
            if cells is None:
                failed = True
                break
            placements.append((s, cells))
            if s == shapes[0]:
                first_raw = _norm(cells)
        if failed:
            continue
        for _ in range(int(rng.integers(4))):
            s = _choose(rng, shapes)
            cells = _place(avail, _variants(s), rng)
            if cells is not None:
                placements.append((s, cells))
        box = np.full((h, w), carrier, dtype=np.int64)
        for s, cells in placements:
            for p in cells:
                box[p] = bg
        if len(_components(box == carrier)) != 1:
            continue
        specs = []
        for s in shapes:
            inks = [winners[s]] + list(rng.permutation([c for c in palette if c != winners[s]]))
            for color, n in zip(inks, counts[s]):
                variants = _variants(s)
                if witness and s == shapes[0]:
                    # The winning handedness is absent from rotation-only matches.
                    variants = (_mirror(first_raw),) if color == winners[s] else (first_raw,)
                specs.extend((s, int(color), variants) for _ in range(n))
        load = sum(cost(s) for s in shapes)
        low_h, low_w = max(h+5, 18), max(w+5, 18)
        ch = 30 if attempt >= 35 or load > 180 else int(rng.integers(low_h, 31))
        cw = 30 if attempt >= 35 or load > 180 else int(rng.integers(low_w, 31))
        y, x = int(rng.integers(3, ch-h)), int(rng.integers(3, cw-w))
        outside = np.ones((ch, cw), dtype=bool)
        outside[[0, -1]] = False
        outside[:, [0, -1]] = False
        outside[y-3:y+h+1, x-3:x+w+1] = False
        objects = []
        specs = [specs[int(i)] for i in rng.permutation(len(specs))]
        specs.sort(key=lambda spec: (max(_size(spec[0])), len(spec[0])), reverse=True)
        for s, color, variants in specs:
            cells = _place(outside, variants, rng)
            if cells is None:
                failed = True
                break
            objects.append((color, cells))
        if failed:
            continue
        inp = np.full((ch, cw), bg, dtype=np.int64)
        inp[y:y+h, x:x+w] = box
        # Occupancy counts, not lengths or ink identities. Gaps and colors vary.
        for index in rng.choice(w, sx, replace=False):
            inp[y-2, x+int(index)] = int(_choose(rng, palette))
        for index in rng.choice(h, sy, replace=False):
            inp[y+int(index), x-2] = int(_choose(rng, palette))
        for color, cells in objects:
            for p in cells:
                inp[p] = color
        # Construct labels from sampled placements, without calling solve/decode.
        out = np.repeat(np.repeat(box, sy, axis=0), sx, axis=1)
        for s, cells in placements:
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            raw = set(_norm(cells))
            mh, mw = _size(raw)
            for r in range(mh*sy):
                for c in range(mw*sx):
                    if (r//sy, c//sx) in raw and (r % mh, c % mw) in raw:
                        out[r0*sy+r, c0*sx+c] = winners[s]
        return {'input': inp, 'output': out}
    return None


_ALTERNATIVES = ('global_vote', 'area_vote', 'bbox_vote', 'translation_vote',
                 'rotation_vote', 'solid', 'plate_tile', 'ignore_votes',
                 'global_phase', 'phase_rows', 'phase_columns', 'no_clip',
                 'canonical_motif')


def _evidence(pair):
    data = _decode(pair['input'])
    result = set()
    for mode in _ALTERNATIVES:
        try:
            alternative = _render(data, mode)
            differs = not np.array_equal(alternative, pair['output'])
        except ValueError:
            differs = True
        if differs:
            result.add(mode)
    sy, sx = data['scale']
    if sy != sx:
        result.add('swapped_rails')
    for rail in data['rails']:
        occupied = np.flatnonzero(rail != data['background'])
        if occupied[-1]-occupied[0]+1 != len(occupied):
            result.add('rail_length')
        _, frequencies = np.unique(rail[occupied], return_counts=True)
        if len(frequencies) < len(occupied):
            result.add('rail_color_count')
        if int(frequencies.max()) < len(occupied):
            result.add('rail_one_ink')
        if len(occupied) > 1 and np.any(np.diff(occupied) == 1):
            result.add('rail_components')
    return result


_REQUIRED = frozenset(_ALTERNATIVES) | {'swapped_rails', 'rail_length', 'rail_color_count',
                                      'rail_one_ink', 'rail_components'}


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10),
                                (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    pool = _pool(rng)
    anchors = [s for s in pool if 4 <= len(s) <= 6 and max(_size(s)) <= 3
               and _mirror(s) not in _rotations(s)
               and any(t != s and len(t) == len(s) and _size(t) == _size(s) for t in pool)]
    anchor = _choose(rng, anchors)
    companion = _choose(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                             and _size(s) == _size(anchor)])
    # Two random geometric witnesses collectively teach the same rule at the
    # minimum count. Retrying tests observable evidence, not a latent convention.
    for _ in range(80):
        scales = [_choose(rng, ((2, 2), (2, 3), (3, 2))),
                  _choose(rng, ((1, 2), (2, 1), (2, 3), (3, 2)))]
        first = _make_pair(rng, (anchor, companion), scales[0], witness=True)
        if first is None:
            continue
        old = _decode(first['input'])['winners'][anchor]
        second = _make_pair(rng, (anchor, companion), scales[1], witness=True, changed=old)
        if second is None:
            continue
        if _REQUIRED <= _evidence(first) | _evidence(second):
            break
    else:
        raise RuntimeError("bounded training evidence search failed")
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    known = {anchor, companion}
    for split, count in (('train', nt), ('test', ne)):
        while len(puzzle[split]) < count:
            for _ in range(40):
                k = int(rng.integers(1, 5))
                available = [s for s in pool if len(s) <= (6 if k >= 3 else 8)]
                shapes = [available[int(i)] for i in rng.choice(len(available), k, replace=False)]
                if split == 'test' and not puzzle['test']:
                    # A novel shape has complete ballots in this very input.
                    shapes[0] = _choose(rng, [s for s in available if s not in known])
                    shapes = list(dict.fromkeys(shapes))
                scale = _choose(rng, ((1, 1), (1, 2), (2, 1), (2, 2),
                                      (2, 3), (3, 2), (1, 3), (3, 1), (3, 3), (4, 2), (2, 4)))
                pair = _make_pair(rng, tuple(shapes), scale)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input search failed")
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            if split == 'train':
                known.update(shapes)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses, evidence = set(), {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            data = _decode(inp)
            sy, sx = data['scale']
            assert out.shape == (data['box'].shape[0]*sy, data['box'].shape[1]*sx)
            assert np.array_equal(out, _render(data)), 'incorrect self-stenciled output'
            macro = np.repeat(np.repeat(data['box'], sy, axis=0), sx, axis=1)
            assert np.all(out[macro == data['carrier']] == data['carrier']), 'ink leaked into carrier'
            assert set(map(int, np.unique(out))) <= {data['background'], data['carrier']} | set(data['winners'].values())
            for cells in data['cavities']:
                # Each cavity's enlarged footprint receives only its elected ink
                # or transparent background, and actually exhibits that ink.
                positions = [(r*sy+dr, c*sx+dc) for r, c in cells
                             for dr in range(sy) for dc in range(sx)]
                values = {int(out[p]) for p in positions}
                winner = data['winners'][_canonical(cells)]
                assert winner in values and values <= {winner, data['background']}
            analyses[split].append(data)
            if split == 'train':
                evidence |= _evidence(pair)
    assert _REQUIRED <= evidence, 'missing teaching evidence: ' + str(_REQUIRED-evidence)
    history = defaultdict(set)
    for data in analyses['train']:
        for shape, winner in data['winners'].items():
            history[shape].add(winner)
    assert any(len(colors) > 1 for colors in history.values()), 'fixed shape-color map remains plausible'
    known = set().union(*(set(d['votes']) for d in analyses['train']))
    assert any(set(d['votes'])-known for d in analyses['test']), 'no novel test shape'


def _normalized(a):
    mapping = {}
    return a.shape, bytes(mapping.setdefault(int(v), len(mapping)) for v in a.flat)


def self_test():
    """Bounded 42-seed audit; returns compact descriptive coverage measurements."""
    puzzles, features, unique = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        unique.add(tuple(_key(pair['input']) for split in p for pair in p[split]))
        for split in p:
            for pair in p[split]:
                d = _decode(pair['input'])
                features.append((split, pair, d))
    assert len(unique) == 42
    vocabulary = set().union(*(set(d['votes']) for _, _, d in features))
    scales = {d['scale'] for _, _, d in features}
    plates = {_normalized(d['box']) for _, _, d in features}
    outputs = {_normalized(pair['output']) for _, pair, _ in features}
    assert len(vocabulary) >= 30 and len(scales) >= 9
    assert len(plates) >= 120 and len(outputs) >= 170
    assert {1, 2, 3, 4} <= {len(d['votes']) for _, _, d in features}
    assert len({pair['output'].shape for _, pair, _ in features}) >= 65
    assert len({d['background'] for _, _, d in features}) == 10
    assert len({len(d['cavities']) for _, _, d in features}) >= 6
    assert any(len(d['cavities']) > len(d['votes']) for _, _, d in features)
    assert any(len(set(d['winners'].values())) < len(d['winners']) for _, _, d in features)
    ballots = {tuple(sorted(v.values(), reverse=True)) for _, _, d in features for v in d['votes'].values()}
    assert {(1,), (3, 2, 2), (4, 3)} <= ballots
    for split in ('train', 'test'):
        chosen = [(p, d) for s, p, d in features if s == split]
        assert any(p['output'].shape[0] > p['output'].shape[1] for p, _ in chosen)
        assert any(p['output'].shape[0] < p['output'].shape[1] for p, _ in chosen)
        assert any(np.any(p['output'] == d['background']) for p, d in chosen)
    assert len({_normalized(p['test'][0]['output']) for p in puzzles}) >= 38
    replay = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in replay
               for a, b in zip(replay[s], puzzles[0][s]) for k in a)
    # Geometry and role encoding survive arbitrary palette relabeling and
    # canvas translation, without outputs or seed information in solve.
    permutation = np.random.default_rng(8301).permutation(10)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    pair = next(pair for pair in puzzles[0]['train'] if max(pair['input'].shape) < 30)
    bg = int(pair['input'][0, 0])
    shifted = np.pad(pair['input'], ((1, 0), (1, 0)), constant_values=bg)
    assert np.array_equal(solve(shifted), pair['output'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(71000+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for request in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                    (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, *request)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return dict(seeds=42, pairs=len(features), shape_classes=len(vocabulary),
                scales=sorted(scales), plate_structures=len(plates),
                output_structures=len(outputs), output_dimensions=len({p['output'].shape for _, p, _ in features}),
                ballots=sorted(ballots), independent_solver=False)
