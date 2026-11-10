"""A capped scanning ruler records line-conditioned connected-object censuses.

Only standard library and NumPy. No files are read or written by this module.
"""
import hashlib
import numpy as np


HOST_SEEDS = (0, 162824317, 750298181, 740948499, 429249161, 4776142,
              481992908, 748833767, 335524940, 73166541, 50068883, 360981523,
              950850054, 109480940, 186459152, 659434719, 200527157, 111256961,
              770496015, 371338446, 975632948, 981215176, 735616053, 829992283,
              928401286, 545353353, 817695199, 571194820, 451192591, 924740009,
              272151029, 116274285, 91570982, 302631104, 790757691, 126743733,
              821593879, 57893324, 678807104, 405573620, 290121500, 352804522)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in directions:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(cells))
    return result


def _counts(objects, depth):
    result = np.zeros(depth, dtype=int)
    for cells in objects:
        for r in {r for r, _ in cells}:
            result[r] += 1
    return result


def _parse(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if frequencies[bg] <= a.size / 2:
        raise ValueError("The background must occupy a strict majority")
    candidates = []
    for rotation in range(4):
        g = np.rot90(a, rotation)
        occupied = np.flatnonzero(g[0] != bg)
        if len(occupied) < 3 or occupied[0] == 0 or occupied[-1] == g.shape[1] - 1:
            continue
        start, end = int(occupied[0]), int(occupied[-1]) + 1
        if len(occupied) != end - start:
            continue
        bar = g[0, start:end]
        if len(set(map(int, bar))) != 2:
            continue
        for left in (True, False):
            cap = int(bar[0 if left else -1])
            key_run = bar[1:] if left else bar[:-1]
            key = int(key_run[0])
            if key == cap or not np.all(key_run == key) or frequencies[cap] != 1:
                continue
            if len(g) < 3 or not np.all(g[1] == bg):
                continue
            if not (np.all(g[-1] == bg) and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
                continue
            mask = g == key
            mask[0] = False
            objects = _components(mask)
            if not objects:
                continue
            colors = sorted(set(map(int, np.unique(g[2:]))) - {bg})
            all_objects = []
            owner = np.full(g.shape, -1, int)
            for color in colors:
                color_mask = g == color
                color_mask[0] = False
                for cells in _components(color_mask):
                    serial = len(all_objects)
                    for r, c in cells:
                        owner[r, c] = serial
                    all_objects.append((color, cells))
            separated = True
            for r, c in np.argwhere(owner >= 0):
                neighbors = owner[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                if np.any((neighbors >= 0) & (neighbors != owner[r, c])):
                    separated = False
                    break
            if not separated:
                continue
            counts = _counts(objects, len(g))
            if np.any(counts[4::6] > (end - start + 1) // 2):
                continue
            candidates.append(dict(g=g.copy(), rotation=rotation, bg=bg, key=key,
                                   cap=cap, start=start, end=end, left=left,
                                   objects=objects, all_objects=all_objects, counts=counts))
    if len(candidates) != 1:
        raise ValueError("Expected one unambiguous capped boundary ruler and valid objects")
    return candidates[0]


def _paint(g, bg, key, cap, start, end, left, counts):
    """Paint using supplied counts; generation supplies construction-list counts."""
    out = g.copy()
    foreground = g != bg
    foreground[0] = False
    out[:, start:end] = bg
    out[0] = g[0]
    for d in range(2, len(g), 2):
        if d % 6 == 4:
            out[d, foreground[d]] = cap
            out[d, start:end] = bg  # Tally has precedence over foreground.
            for i in range(int(counts[d])):
                c = start + 2 * i if left else end - 1 - 2 * i
                out[d, c] = cap
        else:
            out[d, start:end] = key
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result from visible pixels only; never uses generation state."""
    m = _parse(input_grid)
    out = _paint(m['g'], m['bg'], m['key'], m['cap'], m['start'], m['end'], m['left'], m['counts'])
    return np.rot90(out, -m['rotation']).copy()


def _pixel_prediction(m):
    """Validator evaluates the rule cell by cell, separately from the painter."""
    g = m['g']
    out = g.copy()
    for r in range(1, len(g)):
        n = sum(any(rr == r for rr, _ in obj) for obj in m['objects'])
        for c in range(g.shape[1]):
            inside = m['start'] <= c < m['end']
            if inside:
                offset = c - m['start'] if m['left'] else m['end'] - 1 - c
                out[r, c] = (m['bg'] if r % 2 else
                             m['cap'] if r % 6 == 4 and offset % 2 == 0 and offset // 2 < n else
                             m['bg'] if r % 6 == 4 else m['key'])
            elif r % 6 == 4 and g[r, c] != m['bg']:
                out[r, c] = m['cap']
    return np.rot90(out, -m['rotation']).copy()


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape(rng):
    mode = int(rng.integers(8))
    if mode == 0:
        cells = _DIAMOND if rng.random() < .5 else _X
    elif mode == 1:
        h, w = int(rng.integers(1, 5)), int(rng.integers(1, 6))
        cells = [(r, c) for r in range(h) for c in range(w)]
    elif mode == 2:
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        cells = [(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)]
    elif mode == 3:
        cells = [(0, c) for c in range(int(rng.integers(1, 9)))]
    elif mode == 4:
        h, w = int(rng.integers(3, 7)), int(rng.integers(2, 6))
        cells = [(r, 0) for r in range(h)] + [(h - 1, c) for c in range(1, w)]
    else:
        side = int(rng.integers(3, 7))
        target = int(rng.integers(2, min(18, side * side) + 1))
        cells = {(side // 2, side // 2)}
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r + dr < side and 0 <= c + dc < side} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = tuple(cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    if rng.random() < .5:
        cells = tuple((r, -c) for r, c in cells)
    return _normalize(cells)


class _Scene:
    """Bounded in-memory packer; original placed objects supply generation labels."""
    def __init__(self, depth, width, start, length, palette):
        self.bg, self.key, self.cap = palette[:3]
        self.start, self.end = start, start + length
        self.g = np.full((depth, width), self.bg, dtype=np.int8)
        self.g[0, start:self.end] = self.key
        self.g[0, start] = self.cap
        self.blocked = np.zeros(self.g.shape, bool)
        self.blocked[:2] = self.blocked[-1] = True
        self.blocked[:, 0] = self.blocked[:, -1] = True
        self.objects = []
        self.counts = np.zeros(depth, dtype=int)

    def add_at(self, color, cells):
        cells = tuple(cells)
        assert all(not self.blocked[r, c] for r, c in cells)
        if color == self.key:
            addition = _counts([cells], len(self.g))
            assert np.all((self.counts + addition)[4::6] <= (self.end - self.start + 1) // 2)
            self.counts += addition
        for r, c in cells:
            self.g[r, c] = color
            self.blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        self.objects.append((color, cells))

    def place(self, rng, color, shape, hit=None, avoid_samples=False, inside=False):
        h, w = 1 + max(r for r, c in shape), 1 + max(c for r, c in shape)
        nr, nc = self.g.shape[0] - h + 1, self.g.shape[1] - w + 1
        if min(nr, nc) <= 0:
            return False
        allowed = np.ones((nr, nc), bool)
        for r, c in shape:
            allowed &= ~self.blocked[r:r + nr, c:c + nc]
        candidates = np.argwhere(allowed)
        for index in rng.permutation(len(candidates)):
            r0, c0 = map(int, candidates[int(index)])
            cells = tuple((r + r0, c + c0) for r, c in shape)
            rows = {r for r, c in cells}
            if hit is not None and hit not in rows:
                continue
            if avoid_samples and any(r % 6 == 4 for r in rows):
                continue
            if inside and not all(self.start <= c < self.end for r, c in cells):
                continue
            if color == self.key:
                counts = self.counts + _counts([cells], len(self.g))
                if np.any(counts[4::6] > (self.end - self.start + 1) // 2):
                    continue
            self.add_at(color, cells)
            return True
        return False


def _make_scene(rng, mode, palette):
    if mode == 'loop':
        depth = int(rng.integers(20, 31))
        length = int(rng.integers(5, 12))
        width = int(rng.integers(max(24, length + 16), 31))
        start = int(rng.integers(7, width - length - 6))
        s = _Scene(depth, width, start, length, palette)
        d = int(rng.choice([r for r in range(4, depth - 6, 6)]))
        a, b = ((2, 3), (3, 2), (2, 5), (4, 3))[int(rng.integers(4))]
        if d - a < 2:
            a = 2
        top, bottom, left, right = d - a, d + b, start - 2, start + length + 1
        ring = [(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)
                if r in (top, bottom) or c in (left, right)]
        s.add_at(s.key, ring)
        s.add_at(s.key, [(d - 1 + r, start - 6 + c) for r, c in _DIAMOND])
        # A same-line ignored glyph also gets highlighted, without adding a tally.
        s.add_at(palette[3], [(d + r, start + length + 3 + c)
                              for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))])
        assert s.place(rng, s.key, ((0, 0),), avoid_samples=True)
    elif mode == 'repeat':
        depth = int(rng.integers(19, 31))
        width = int(rng.integers(21, 31))
        length = int(rng.integers(7, min(14, width - 7)))
        start = int(rng.integers(6, width - length))
        s = _Scene(depth, width, start, length, palette)
        top, bottom = int(rng.integers(2, 4)), int(rng.integers(11, 15))
        w = int(rng.integers(2, 5))
        if rng.random() < .5:
            shape = [(r, c) for r in range(top, bottom + 1) for c in range(1, 1 + w)]
        else:
            shape = [(r, c) for r in range(top, bottom + 1) for c in range(1, 1 + w)
                     if c in (1, w) or r in (top, bottom)]
        s.add_at(s.key, shape)
        inside = _DIAMOND if rng.random() < .5 else ((0, 0), (0, 1), (1, 0), (1, 1))
        offset = int(rng.integers(start + 1, start + length - 3))
        s.add_at(s.key, [(9 + r, offset + c) for r, c in inside])
        assert s.place(rng, palette[3], ((0, 0), (1, 0)), hit=4)
    else:
        depth = int(rng.integers(13, 31))
        width = int(rng.integers(16, 31))
        length = int(rng.integers(5, min(14, width - 2)))
        start = int(rng.integers(1, width - length))
        s = _Scene(depth, width, start, length, palette)
        # At least one matching object, without forcing a fixed contact location.
        assert s.place(rng, s.key, _DIAMOND if rng.random() < .4 else ((0, 0),))
    if mode == 'repeat':
        # Keep a plainly empty late census to teach zero; extras need not be keys.
        count = int(rng.integers(1, 5))
    else:
        count = int(rng.integers(3, 12))
    prototype = _shape(rng)
    for i in range(count):
        color = s.key if rng.random() < .6 else int(rng.choice(palette[3:6]))
        if mode == 'repeat' and color == s.key:
            # Only unsampled extras; the long object still supplies two different tallies.
            s.place(rng, color, ((0, 0),), avoid_samples=True)
        else:
            s.place(rng, color, prototype if rng.random() < .3 else _shape(rng))
    out = _paint(s.g, s.bg, s.key, s.cap, s.start, s.end, True, s.counts)
    return s, out


def _checked_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(name + ' is outside its legal integer range')
    return int(value)


def _input_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _checked_count(num_train, 'num_train', 2, 10)
    nq = _checked_count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    regime, base = int(rng.integers(3)), int(rng.integers(4))
    first_palette = list(map(int, rng.permutation(10)))
    initial_flip = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', nq)):
        for i in range(n):
            mode = ('loop' if i == 0 else 'repeat' if i == 1 else 'generic') if split == 'train' else 'generic'
            if split == 'train':
                rotation = base if regime < 2 else (base + i % 2) % 4
            else:
                rotation = ((base + int(rng.choice((1, 3)))) % 4 if regime == 0 else
                            (base + 2) % 4 if regime == 1 else (base + int(rng.choice((2, 3)))) % 4)
            palette = list(map(int, rng.permutation(10)))
            if split == 'train' and i == 0:
                palette = first_palette.copy()
            elif split == 'train' and i == 1:
                key, distractor = first_palette[3], first_palette[1]
                rest = [c for c in palette if c not in (key, distractor)]
                palette = [rest[0], key, rest[1], distractor] + rest[2:]
            flip = (initial_flip if i == 0 else not initial_flip) if split == 'train' and i < 2 else bool(rng.integers(2))
            for _ in range(32):
                s, out = _make_scene(rng, mode, palette)
                inp = s.g
                if flip:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
                key = _input_key(inp)
                if key not in seen:
                    # Construction counts must agree with finished original pixels.
                    m = _parse(inp)
                    assert np.array_equal(m['counts'], s.counts)
                    break
            else:
                raise RuntimeError('Could not construct a distinct input within 32 attempts')
            seen.add(key)
            puzzle[split].append({'input': inp, 'output': out})
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(m, counts):
    # Overflow remains visibly different; it is never clamped into a valid label.
    if np.any(counts[4::6] > (m['end'] - m['start'] + 1) // 2):
        return None
    return _paint(m['g'], m['bg'], m['key'], m['cap'], m['start'], m['end'], m['left'], counts)


def _features(m):
    g, objects, counts = m['g'], m['objects'], m['counts']
    samples = list(range(4, len(g), 6))
    key_mask = g == m['key']
    key_mask[0] = False
    erased = key_mask.copy()
    erased[:, m['start']:m['end']] = False
    once = np.zeros(len(g), int)
    for obj in objects:
        touched = sorted({r for r, c in obj if r % 6 == 4})
        if touched:
            once[touched[0]] += 1
    alternatives = {
        'global': np.full(len(g), len(objects), int),
        'all_colors': _counts([obj for color, obj in m['all_objects']], len(g)),
        'pixels': key_mask.sum(axis=1),
        'line_fragments': np.array([len(_components(key_mask[r:r + 1])) for r in range(len(g))]),
        'four_connected': _counts(_components(key_mask, False), len(g)),
        'after_erasure': _counts(_components(erased), len(g)),
        'first_hit_only': once,
    }
    expected = _paint(g, m['bg'], m['key'], m['cap'], m['start'], m['end'], m['left'], counts)
    differences = {name: not np.array_equal(_alternative(m, n), expected) for name, n in alternatives.items()}
    multi_contact = False
    for obj in objects:
        for d in samples:
            cols = sorted(c for r, c in obj if r == d)
            multi_contact |= bool(cols and any(b - a > 1 for a, b in zip(cols, cols[1:])))
    remote, clear, overwrite = False, False, False
    for r, c in np.argwhere(g != m['bg']):
        if r == 0:
            continue
        inside = m['start'] <= c < m['end']
        remote |= not inside and r % 6 == 4
        clear |= inside and r % 2 == 1
        overwrite |= inside and r > 0 and r % 6 in (0, 2)
    return dict(differences=differences, multi_contact=multi_contact,
                repeat=any(len({r for r, c in obj if r % 6 == 4}) >= 2 for obj in objects),
                inside=any(all(m['start'] <= c < m['end'] for r, c in obj) and
                           any(r % 6 == 4 for r, c in obj) for obj in objects),
                zero=any(counts[d] == 0 for d in samples),
                remote=remote, clear=clear, overwrite=overwrite,
                positive_counts={int(counts[d]) for d in samples if counts[d]},
                key=m['key'], ignored={color for color, obj in m['all_objects'] if color != m['key']})


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in seen, 'Duplicate input'
            seen.add(key)
            m = _parse(inp)
            assert np.array_equal(out, _pixel_prediction(m)), 'Incorrect output'
            assert not np.array_equal(inp, out)
            assert np.array_equal(out, solve(inp))
            if split == 'train':
                training.append(_features(m))
    for name in ('global', 'all_colors', 'pixels', 'line_fragments', 'four_connected', 'after_erasure', 'first_hit_only'):
        assert any(f['differences'][name] for f in training), 'Unexposed alternative: ' + name
    for name in ('multi_contact', 'repeat', 'inside', 'zero', 'remote', 'clear', 'overwrite'):
        assert any(f[name] for f in training), 'Missing teaching witness: ' + name
    assert len(set().union(*(f['positive_counts'] for f in training))) >= 2
    assert any(a['key'] in b['ignored'] for a in training for b in training)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 specified seeds and a count matrix; retain only small summary sets."""
    digests, shapes, vocab, sides, cap_ends, lengths, profiles, palette_roles = set(), set(), set(), set(), set(), set(), set(), [set(), set(), set()]
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        digest = _digest(p)
        assert digest == _digest(generate_puzzle(seed)), 'Nondeterministic seed'
        digests.add(digest)
        assert {_parse(pair['input'])['rotation'] for pair in p['train']}.isdisjoint(
            {_parse(pair['input'])['rotation'] for pair in p['test']})
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                m = _parse(a)
                shapes.add(a.shape)
                sides.add(m['rotation'])
                cap_ends.add(m['left'])
                lengths.add(m['end'] - m['start'])
                profiles.add(tuple(map(int, m['counts'][4::6])))
                vocab.update(_normalize(obj) for color, obj in m['all_objects'])
                for pool, role in zip(palette_roles, ('bg', 'key', 'cap')):
                    pool.add(m[role])
                # All rigid motions must commute with solving; no learned orientation.
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
    assert len(digests) == 42 and len(shapes) >= 50 and len(vocab) >= 100
    assert sides == set(range(4)) and cap_ends == {False, True}
    assert len(lengths) >= 8 and len(profiles) >= 20
    assert all(pool == set(range(10)) for pool in palette_roles)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, False), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Short valid members need no census line, and family membership is input-only.
    short = np.zeros((4, 9), dtype=np.int8)
    short[0, 2:5], short[0, 2], short[2, 6] = 1, 2, 1
    prediction = solve(short)
    assert np.all(prediction[2, 2:5] == 1) and prediction[2, 6] == 1
    assert np.all(prediction[1, 2:5] == 0)
    # A sampled object wholly inside the corridor is counted before erasure.
    tiny = np.zeros((8, 11), dtype=np.int8)
    tiny[0, 3:8], tiny[0, 3], tiny[4, 5] = 1, 2, 1
    answer = solve(tiny)
    assert answer[4, 3] == 2 and answer[4, 5] == 0
    bad = generate_puzzle(55, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt label accepted')
