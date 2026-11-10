"""Corroborated mirror fractures: local proposals need remote witnesses.

No parent module, stored example, random state, or training label is used by solve.
Black is background. Four border tabs visibly encode two fold axes; their color
is excluded from the interior drawing. All decisions use the original input.
"""
from functools import lru_cache
from hashlib import sha256
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _guides(a):
    border = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
    colors = set(map(int, border)) - {0}
    if len(colors) != 1:
        raise ValueError("exactly one nonblack border-tab color is required")
    marker = colors.pop()
    if np.any(a[1:-1, 1:-1] == marker):
        raise ValueError("guide color cannot be a drawing color")
    r0, r1 = np.flatnonzero(a[:, 0] == marker), np.flatnonzero(a[:, -1] == marker)
    c0, c1 = np.flatnonzero(a[0] == marker), np.flatnonzero(a[-1] == marker)
    for first, second, size in ((r0, r1, a.shape[0]), (c0, c1, a.shape[1])):
        if (not np.array_equal(first, second) or len(first) not in (1, 2)
                or first[0] <= 0 or first[-1] >= size - 1
                or (len(first) == 2 and first[1] != first[0] + 1)):
            raise ValueError("opposite tabs must match and have length one or two")
    return marker, (int(r0[0] + r0[-1]), int(c0[0] + c0[-1]))


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _orthogonal_support(mask):
    support = np.zeros_like(mask)
    adjacent = mask[:-1] & mask[1:]
    support[:-1] |= adjacent
    support[1:] |= adjacent
    adjacent = mask[:, :-1] & mask[:, 1:]
    support[:, :-1] |= adjacent
    support[:, 1:] |= adjacent
    return support


def _gaps(mask, a):
    """Retained endpoints, three black cells, and continuation past an endpoint."""
    found = []
    for vertical in (False, True):
        m, raw = (mask.T, a.T) if vertical else (mask, a)
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= (raw[:, 1:-3] == 0) & (raw[:, 2:-2] == 0) & (raw[:, 3:-1] == 0)
        for y, x in np.argwhere(bracket):
            y, x = int(y), int(x)
            if not (_at(m, y, x - 1) or _at(m, y, x + 5)):
                continue
            r, c, dr, dc = (x, y, 1, 0) if vertical else (y, x, 0, 1)
            found.append((r, c, dr, dc))
    return found


def _slot_orbit(slot, axes, shape):
    result = set()
    for flip_r in (False, True):
        for flip_c in (False, True):
            cells = tuple(sorted((axes[0] - r if flip_r else r,
                                  axes[1] - c if flip_c else c) for r, c in slot))
            if all(0 < r < shape[0] - 1 and 0 < c < shape[1] - 1 for r, c in cells):
                result.add(cells)
    return tuple(sorted(result))


def _decode(a):
    marker, axes = _guides(a)
    out = a.copy()
    records, noise = {}, np.zeros_like(a, dtype=bool)
    colors = sorted(set(map(int, a.flat)) - {0, marker})
    for color in colors:
        mask = a == color
        supported = _orthogonal_support(mask)
        gaps = _gaps(mask, a)
        anchors = np.zeros_like(mask)
        for r, c, dr, dc in gaps:
            anchors[r, c] = anchors[r + 4 * dr, c + 4 * dc] = True
        weak = mask & ~supported & ~anchors
        diagonal = np.zeros_like(mask)
        for dc in (-1, 1):
            left, right = (weak[:-1, :-1], weak[1:, 1:]) if dc == 1 else (weak[:-1, 1:], weak[1:, :-1])
            link = left & right
            if dc == 1:
                diagonal[:-1, :-1] |= link
                diagonal[1:, 1:] |= link
            else:
                diagonal[:-1, 1:] |= link
                diagonal[1:, :-1] |= link
        noise |= diagonal
        for r, c, dr, dc in gaps:
            proposals = []
            for endpoint in (0, 4):
                ar, ac = r + endpoint * dr, c + endpoint * dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward * dr + side * dc, toward * dc + side * dr
                    spur = tuple((ar + k * sr, ac + k * sc) for k in (1, 2))
                    if (all(_at(weak, rr, cc) for rr, cc in spur)
                            and not _at(mask, ar + 3 * sr, ac + 3 * sc)):
                        proposals.append(spur)
            if not proposals:
                continue
            slot = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
            orbit = _slot_orbit(slot, axes, a.shape)
            key = color, orbit
            if key not in records:
                states = []
                for cells in orbit:
                    values = [int(a[p]) for p in cells]
                    if all(v == color for v in values):
                        states.append('I')
                    elif all(v == 0 for v in values):
                        states.append('B')
                    else:
                        raise ValueError("a proposed slot orbit contains a partial or conflicting segment")
                records[key] = {"color": color, "orbit": orbit, "states": states,
                                "proposals": [], "active": 'I' in states}
            records[key]["proposals"].append((slot, tuple(proposals)))
    # Removal and copying are simultaneous: cleaned or repaired pixels cannot
    # manufacture a new witness in this pass.
    out[noise] = 0
    filled = np.zeros_like(a, dtype=bool)
    for rec in records.values():
        if rec['active']:
            for cells in rec['orbit']:
                for p in cells:
                    if filled[p] and out[p] != rec['color']:
                        raise ValueError("conflicting repairs")
                    if a[p] == 0:
                        out[p] = rec['color']
                        filled[p] = True
    return out, list(records.values()), noise, axes, marker


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the tabs, corroborate raw-input proposals, delete oblique debris."""
    return _decode(_grid(input_grid))[0]


def _fits(h, w):
    return (h >= 3 and w >= 6) or (h >= 6 and w >= 3)


@lru_cache(maxsize=2048)
def _can_pack(h, w, n):
    if n == 1:
        return _fits(h, w)
    if h * w < 18 * n:
        return False
    for axis, size in ((0, h), (1, w)):
        for cut in range(3, size - 3):
            for k in range(1, n):
                a, b = ((cut, w), (h - cut - 1, w)) if axis == 0 else ((h, cut), (h, w - cut - 1))
                if _can_pack(*a, k) and _can_pack(*b, n - k):
                    return True
    return False


def _regions(rng, r, c, h, w, n):
    if n == 1:
        return [(r, c, h, w)]
    options = []
    for axis, size in ((0, h), (1, w)):
        for cut in range(3, size - 3):
            for k in range(1, n):
                a, b = ((cut, w), (h - cut - 1, w)) if axis == 0 else ((h, cut), (h, w - cut - 1))
                if _can_pack(*a, k) and _can_pack(*b, n - k):
                    options.append((axis, cut, k))
    axis, cut, k = options[int(rng.integers(len(options)))]
    if axis == 0:
        return (_regions(rng, r, c, cut, w, k)
                + _regions(rng, r + cut + 1, c, h - cut - 1, w, n - k))
    return (_regions(rng, r, c, h, cut, k)
            + _regions(rng, r, c + cut + 1, h, w - cut - 1, n - k))


def _motif(rng, h, w):
    orientations = [v for v in (False, True)
                    if (w if v else h) >= 3 and (h if v else w) >= 6]
    vertical = bool(rng.choice(orientations))
    hh, ww = (w, h) if vertical else (h, w)
    for _ in range(32):
        mask = np.zeros((hh, ww), bool)
        row = int(rng.integers(hh))
        left = int(rng.integers(ww - 5))
        right = int(rng.integers(left + 5, ww))
        start = int(rng.integers(left + 1, right - 2))
        slot = [(row, c) for c in range(start, start + 3)]
        mask[row, left:right + 1] = True
        # Variable L/T/comb geometry, with all branch attachments outside the slot.
        columns = [left if rng.integers(2) else right]
        if rng.random() < .55:
            columns += [x for x in (left, right) if x not in columns]
        candidates = [x for x in range(left + 1, right) if x not in range(start - 1, start + 4)
                      and all(abs(x - y) >= 3 for y in columns)]
        if candidates and rng.random() < .35:
            columns.append(int(rng.choice(candidates)))
        for col in columns:
            lo = int(rng.integers(row + 1))
            hi = int(rng.integers(row, hh))
            if lo == hi:
                if row:
                    lo -= 1
                else:
                    hi += 1
            mask[lo:hi + 1, col] = True
        broken = mask.copy()
        for p in slot:
            broken[p] = False
        spurs = []
        for anchor, toward in ((start - 1, 1), (start + 3, -1)):
            for side in (-1, 1):
                spur = [(row + k * side, anchor + k * toward) for k in (1, 2)]
                if not all(0 <= r < hh and 0 <= c < ww and not mask[r, c] for r, c in spur):
                    continue
                trial = broken.copy()
                for p in spur:
                    trial[p] = True
                if all(not _orthogonal_support(trial)[p] for p in spur):
                    spurs.append(spur)
        if spurs:
            if vertical:
                return mask.T.copy(), [(c, r) for r, c in slot], [[(c, r) for r, c in s] for s in spurs]
            return mask, slot, spurs
    # Guaranteed fitting leaf: a six-cell L with a spur bent away from its stub.
    mask = np.zeros((hh, ww), bool)
    mask[0, :6] = True
    mask[:3, 5] = True
    slot, spurs = [(0, c) for c in (1, 2, 3)], [[(1, 1), (2, 2)]]
    if vertical:
        return mask.T.copy(), [(c, r) for r, c in slot], [[(c, r) for r, c in s] for s in spurs]
    return mask, slot, spurs


def _statuses(rng, kind, hard):
    if kind == 'active':
        intact = 1 if hard else int(rng.integers(1, 3))
        spurred = int(rng.integers(1, 5 - intact))
        values = ['I'] * intact + ['S'] * spurred + ['B'] * (4 - intact - spurred)
    elif kind == 'unsupported':
        spurred = int(rng.integers(1, 5))
        values = ['S'] * spurred + ['B'] * (4 - spurred)
    else:
        intact = int(rng.integers(1, 4))
        values = ['I'] * intact + ['B'] * (4 - intact)
    rng.shuffle(values)
    return values


def _pair(rng, kinds, hard, large=False):
    n = len(kinds)
    shapes = [(h, w) for h in range(3, 14) for w in range(3, 14)
              if _can_pack(h, w, n) and (not large or h * w >= 110)]
    qh, qw = shapes[int(rng.integers(len(shapes)))]
    through = [bool(rng.integers(2)), bool(rng.integers(2))]
    # One black center row/column for a through-cell fold; two for a fold
    # between cells. Reflected one-pixel strokes cannot merge into thick bands.
    separators = [1 if odd else 2 for odd in through]
    before, after = [], []
    for q, sep in zip((qh, qw), separators):
        room = min(2, 30 - (2 * q + sep + 2))
        total = int(rng.integers(room + 1))
        first = int(rng.integers(total + 1))
        before.append(first)
        after.append(total - first)
    h, w = [2 * q + sep + 2 + b + z for q, sep, b, z in zip((qh, qw), separators, before, after)]
    origin = (1 + before[0], 1 + before[1])
    axes = tuple(2 * o + 2 * q + int(not odd) for o, q, odd in zip(origin, (qh, qw), through))
    marker = int(rng.integers(1, 10))
    palette = np.array([c for c in range(1, 10) if c != marker], dtype=np.int8)
    rng.shuffle(palette)
    pool = palette[:int(rng.integers(1, min(4, n) + 1))]
    inp, out = np.zeros((h, w), np.int8), np.zeros((h, w), np.int8)
    regions = _regions(rng, *origin, qh, qw, n)
    rng.shuffle(regions)
    flips = [(False, False), (False, True), (True, False), (True, True)]
    for j, ((r0, c0, rh, cw), kind) in enumerate(zip(regions, kinds)):
        mask, slot, spurs = _motif(rng, rh, cw)
        color = int(pool[j % len(pool)])
        states = _statuses(rng, kind, hard)
        points = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
        slot = [(r0 + r, c0 + c) for r, c in slot]
        spurs = [[(r0 + r, c0 + c) for r, c in s] for s in spurs]
        for state, (fr, fc) in zip(states, flips):
            def transform(p):
                r, c = p
                return (axes[0] - r if fr else r, axes[1] - c if fc else c)
            for p in points:
                inp[transform(p)] = out[transform(p)] = color
            if state != 'I':
                for p in slot:
                    inp[transform(p)] = 0
                    if kind != 'active':
                        out[transform(p)] = 0
            if state == 'S':
                spur = spurs[int(rng.integers(len(spurs)))]
                for p in spur:
                    inp[transform(p)] = color
    # Independent oblique debris has no reserved corner or repeated location.
    for _ in range(int(rng.integers(4))):
        for _ in range(20):
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            cells = [(r + k * dr, c + k * dc) for k in range(int(rng.integers(2, 4)))]
            if not all(0 < y < h - 1 and 0 < x < w - 1 and
                       not inp[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any() for y, x in cells):
                continue
            color = int(rng.choice(pool))
            for p in cells:
                inp[p] = color
            break
    for board in (inp, out):
        rows = [axes[0] // 2] if axes[0] % 2 == 0 else [axes[0] // 2, axes[0] // 2 + 1]
        cols = [axes[1] // 2] if axes[1] % 2 == 0 else [axes[1] // 2, axes[1] // 2 + 1]
        board[rows, 0] = board[rows, -1] = marker
        board[0, cols] = board[-1, cols] = marker
    return {'input': inp, 'output': out}


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    # Randomized demonstration ordering; the first two collectively refute both
    # one-parent shortcuts even at the minimum legal training count.
    contrasts = ['unsupported', 'uncued']
    rng.shuffle(contrasts)
    crowded_shift = bool(rng.integers(2))
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            kinds = ['active', contrasts[i % 2]]
            if split == 'test':
                kinds.append(contrasts[(i + 1) % 2])
                if crowded_shift:
                    kinds.append(str(rng.choice(['active', 'unsupported', 'uncued'])))
            elif not crowded_shift and rng.random() < .45:
                kinds.append(contrasts[(i + 1) % 2])
            rng.shuffle(kinds)
            for _ in range(64):
                pair = _pair(rng, kinds, hard=(split == 'test' or i == 0),
                             large=split == 'test' and not crowded_shift)
                try:
                    info = _analyze(pair)
                except (AssertionError, ValueError):
                    continue
                # Ensure actual scene-wide parsing agrees with the intended
                # independently built slots; no accidental global attachments.
                if (info['active'] != kinds.count('active')
                        or info['unsupported'] != kinds.count('unsupported')
                        or ('uncued' in kinds and info['uncued'] == 0)):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct scene construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    predicted, records, noise, axes, marker = _decode(inp)
    assert np.array_equal(predicted, out), 'incomplete or incorrect corroborated repair'
    added, removed = (inp == 0) & (out != 0), (inp != 0) & (out == 0)
    assert added.any() and removed.any()
    assert np.array_equal(removed, noise), 'only oblique debris may be removed'
    assert np.all(out[inp == marker] == marker)
    assert np.all((inp == out) | added | removed), 'visible drawing recolored'
    for color in set(map(int, out.flat)) - {0, marker}:
        m = out == color
        assert not np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]), 'merged thick strokes'
    active = [r for r in records if r['active']]
    inactive = [r for r in records if not r['active']]
    assert active, 'each example must demonstrate a corroborated repair'
    assert all(len(r['orbit']) == 4 for r in records), 'sampler uses four distinct slots'
    assert all(len(p) == 3 for r in records for p in r['orbit'])
    for rec in records:
        for slot in rec['orbit']:
            if rec['active']:
                assert all(out[p] == rec['color'] for p in slot)
            else:
                assert all(out[p] == 0 for p in slot), 'unsupported proposal filled'
    # Detect uncued asymmetric slot orbits from the raw input independently of
    # the proposal list. Every absent mate remains absent in the returned output.
    uncued = set()
    proposal_keys = {(r['color'], r['orbit']) for r in records}
    for color in set(map(int, inp.flat)) - {0, marker}:
        for r, c, dr, dc in _gaps(inp == color, inp):
            slot = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
            orbit = _slot_orbit(slot, axes, inp.shape)
            if (color, orbit) in proposal_keys or len(orbit) != 4:
                continue
            if any(all(inp[p] == color for p in cells) for cells in orbit):
                uncued.add((color, orbit))
                assert all(out[p] == 0 for p in slot), 'uncued asymmetry filled'
    double_only = 0
    orientations = set()
    intact_counts = []
    for rec in active:
        intact_counts.append(rec['states'].count('I'))
        for slot in rec['orbit']:
            orientations.add('h' if len({r for r, c in slot}) == 1 else 'v')
            if all(inp[p] == 0 for p in slot):
                direct = [((axes[0] - y, x) for y, x in slot),
                          ((y, axes[1] - x) for y, x in slot)]
                if not any(all(inp[p] == rec['color'] for p in cells) for cells in direct):
                    double_only += 1
    return {'shape': inp.shape, 'marker': marker, 'axes': axes,
            'payload': tuple(sorted(set(map(int, inp.flat)) - {0, marker})),
            'active': len(active), 'unsupported': len(inactive), 'uncued': len(uncued),
            'added': int(added.sum()), 'removed': int(removed.sum()),
            'orientations': tuple(sorted(orientations)), 'double_only': double_only,
            'intact_counts': tuple(intact_counts), 'records': records}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(np.asarray(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            if split == 'train':
                train.append(info)
    assert any(x['unsupported'] for x in train), 'no rejected local proposal demonstration'
    assert any(x['uncued'] for x in train), 'no preserved mirror-asymmetry demonstration'
    assert any(x['double_only'] for x in train), 'both reflection operations must be demonstrated'


def self_test():
    """Stream 42 puzzles; store metrics, never a corpus of reproducible grids."""
    shapes, markers, payloads, parities, offsets, orientations = set(), set(), set(), set(), set(), set()
    signatures, structures, added_counts, unsupported, uncued = set(), set(), set(), 0, 0
    double_only, multi_active, mixed, pairs = 0, 0, 0, 0
    local_shortcut_failures = mirror_shortcut_failures = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(sha256(repr(_signature(p)).encode()).hexdigest())
        local_shapes = set()
        rejects_local = rejects_mirror = False
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _analyze(pair)
                pairs += 1
                h, w = info['shape']
                a, b = info['axes']
                shapes.add((h, w))
                local_shapes.add((h, w))
                markers.add(info['marker'])
                payloads.add(len(info['payload']))
                parities.add((a % 2, b % 2))
                offsets.add((int(np.sign(a - h + 1)), int(np.sign(b - w + 1))))
                orientations.update(info['orientations'])
                added_counts.add(info['added'])
                unsupported += info['unsupported'] > 0
                uncued += info['uncued'] > 0
                if split == 'train':
                    # A local-only straightener necessarily fills an unsupported
                    # proposed slot; a cue-free mirror copier necessarily fills
                    # an uncued asymmetric slot. Both differ from these labels.
                    rejects_local |= info['unsupported'] > 0
                    rejects_mirror |= info['uncued'] > 0
                double_only += info['double_only'] > 0
                multi_active += info['active'] > 1
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                structures.add((info['shape'], sha256(normalized).hexdigest()))
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
        mixed += len(local_shapes) > 1
        assert rejects_local and rejects_mirror
        local_shortcut_failures += rejects_local
        mirror_shortcut_failures += rejects_mirror
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 50 and markers == set(range(1, 10))
    assert payloads == {1, 2, 3, 4} and len(parities) == 4 and len(offsets) == 9
    assert orientations == {'h', 'v'} and len(added_counts) >= 3
    assert unsupported >= 42 and uncued >= 42 and double_only >= 42
    assert multi_active > 0 and mixed >= 35
    original = generate_puzzle(991, 2, 1)
    assert _signature(original) == _signature(generate_puzzle(991, 2, 1))
    assert _signature(original) != _signature(generate_puzzle(992, 2, 1))
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(71, n, t)
            assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Rotations, reflections, and arbitrary permutations of the nine nonblack
    # roles preserve meaning because the border tabs identify the marker.
    for pair in original['train'] + original['test']:
        for k in range(4):
            for flip in (False, True):
                a, b = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                if flip:
                    a, b = a[:, ::-1], b[:, ::-1]
                assert np.array_equal(solve(a), b)
        permutation = np.array([0, 8, 7, 6, 5, 4, 3, 2, 1, 9], np.int8)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Full-label regressions: omitted repair, kept spur, damaged stable cell.
    for mode in ('missing_repair', 'kept_spur', 'stable_cell', 'unsupported_fill', 'uncued_fill'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in x.items()} for x in original[s]] for s in ('train', 'test')}
        pair = bad['test'][0]
        inp, out = pair['input'], pair['output']
        if mode == 'missing_repair':
            p = tuple(np.argwhere((inp == 0) & (out != 0))[0])
            out[p] = 0
        elif mode == 'kept_spur':
            p = tuple(np.argwhere((inp != 0) & (out == 0))[0])
            out[p] = inp[p]
        elif mode == 'stable_cell':
            p = tuple(np.argwhere((inp == out) & (out != 0))[0])
            out[p] = 0
        elif mode == 'unsupported_fill':
            rec = next(r for r in _decode(inp)[1] if not r['active'])
            out[rec['proposals'][0][0][0]] = rec['color']
        else:
            _, records, _, axes, marker = _decode(inp)
            proposal_keys = {(r['color'], r['orbit']) for r in records}
            changed = False
            for color in set(map(int, inp.flat)) - {0, marker}:
                for r, c, dr, dc in _gaps(inp == color, inp):
                    slot = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
                    orbit = _slot_orbit(slot, axes, inp.shape)
                    if ((color, orbit) not in proposal_keys and len(orbit) == 4
                            and any(all(inp[p] == color for p in cells) for cells in orbit)):
                        out[slot[0]] = color
                        changed = True
                        break
                if changed:
                    break
            assert changed, 'missing uncued mutation target'
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('corrupted complete label accepted')
    return {'seeds': 42, 'pairs': pairs, 'distinct_structures': len(structures),
            'shapes': len(shapes), 'markers': sorted(markers), 'payload_counts': sorted(payloads),
            'fold_parities': sorted(parities), 'offset_sides': sorted(offsets),
            'added_counts': sorted(added_counts), 'unsupported_pairs': unsupported,
            'uncued_pairs': uncued, 'double_only_pairs': double_only,
            'multiple_active_pairs': multi_active, 'mixed_shape_puzzles': mixed,
            'puzzles_refuting_local_only': local_shortcut_failures,
            'puzzles_refuting_cue_free_mirrors': mirror_shortcut_failures}
