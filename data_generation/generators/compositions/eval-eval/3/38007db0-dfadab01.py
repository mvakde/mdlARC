"""Contextual disagreement stamps, with an input-visible codebook.

No files, global random state, training labels, or cached examples are used.
All diagnostics in self_test are bounded and stay in memory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    """Deterministically ordered eight-connected components."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _decode(input_grid):
    """Read frames, modal interiors, exact differences, and remaining material."""
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max()) > inp.size // 2, 'unique majority background required'
    rings = []
    for color in map(int, colors):
        if color == bg:
            continue
        for cells in _components(inp == color):
            r, c, b, d = _box(cells)
            h, w = b - r, d - c
            if not (5 <= h <= 7 and 5 <= w <= 7):
                continue
            block = inp[r:b, c:d]
            if (len(cells) == 2 * h + 2 * w - 4
                    and np.all(block[[0, -1], :] == color)
                    and np.all(block[:, [0, -1]] == color)
                    and np.all(block[1:-1, 1:-1] != color)):
                rings.append((color, r, c, b, d))
    # A hollow motif inside a genuine frame is not another specimen frame.
    rings = [x for x in rings if not any(
        y[1] < x[1] and y[2] < x[2] and x[3] < y[3] and x[4] < y[4]
        for y in rings)]
    groups = defaultdict(list)
    for color, r, c, b, d in rings:
        groups[color].append((r, c, b, d))
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    assert 1 <= len(groups) <= 2, 'need one or two visibly labeled specimen groups'
    book, details, frames = {}, {}, []
    view = inp.copy()
    for code, boxes in sorted(groups.items()):
        assert 3 <= len(boxes) <= 5
        interiors = [inp[r+1:b-1, c+1:d-1] for r, c, b, d in boxes]
        assert len({p.shape for p in interiors}) == 1
        frequencies = Counter(_key(p) for p in interiors)
        assert sorted(frequencies.values()) == [1, len(boxes)-1], 'one visible exception'
        master_key = frequencies.most_common(1)[0][0]
        master = next(p for p in interiors if _key(p) == master_key)
        variant = next(p for p in interiors if _key(p) != master_key)
        pigment = set(map(int, np.unique(np.stack(interiors)))) - {bg}
        assert len(pigment) == 1
        ink = pigment.pop()
        assert ink not in groups and ink != code
        assert set(map(int, np.unique(master))) == {bg, ink}
        assert set(map(int, np.unique(variant))) == {bg, ink}
        delta = master != variant
        assert 2 <= int(delta.sum()) <= delta.size - 2
        book[code] = (ink, delta.copy())
        details[code] = {'master': master.copy(), 'variant': variant.copy(),
                         'boxes': boxes, 'added': (master == bg) & delta,
                         'removed': (master == ink) & delta}
        for r, c, b, d in boxes:
            view[r:b, c:d] = bg
            frames.append((r, c, b, d))
    assert len({ink for ink, _ in book.values()}) == len(book)
    assert set(map(int, colors)) <= {bg} | set(book) | {v[0] for v in book.values()}
    return bg, book, details, frames, view


def _members(view, bg, codes):
    live, attached, bare, painted = [], [], [], []
    for cells in _components(view != bg):
        points = [(r, c, int(view[r, c])) for r, c in cells
                  if int(view[r, c]) in codes]
        r, c, _, _ = _box(cells)
        if len(points) == 1 and points[0][:2] == (r, c):
            live.append(points[0])
            if len(cells) > 1:
                painted.append(points[0])
        elif points:
            attached.extend(points)
        else:
            bare.append(cells)
    return sorted(live), sorted(attached), bare, painted


def _check_material(view, bg, book):
    """Check artifacts against the current input-derived masks, including padding."""
    by_ink = {ink: code for code, (ink, _) in book.items()}
    for cells in _components(view != bg):
        points = [p for p in cells if int(view[p]) in book]
        if len(cells) == 1 and len(points) == 1:
            continue
        assert len(points) <= 1, 'unexplained multi-code material'
        ink_values = {int(view[p]) for p in cells if int(view[p]) in by_ink}
        assert len(ink_values) == 1
        code = by_ink[ink_values.pop()]
        ink, mask = book[code]
        if points:
            assert int(view[points[0]]) == code
        h, w = mask.shape
        a, b, _, _ = _box(cells)
        actual = {p: int(view[p]) for p in cells}
        matched = False
        last = max(map(tuple, np.argwhere(mask)))
        for r in range(max(0, a-h), a+1):
            for c in range(max(0, b-w), b+1):
                if r+h > view.shape[0] or c+w > view.shape[1]:
                    continue
                expected = {(r+int(x), c+int(y)): ink for x, y in np.argwhere(mask)}
                if points:
                    p = points[0]
                    painted = p == (r, c) and not mask[0, 0]
                    attached = p == (r+int(last[0])+1, c+int(last[1])+1)
                    if not (painted or attached):
                        continue
                    expected[p] = code
                if expected == actual:
                    matched = True
        assert matched, 'unexplained unframed material'


def _stamp(out, r, c, ink, mask):
    h, w = min(mask.shape[0], out.shape[0]-r), min(mask.shape[1], out.shape[1]-c)
    out[r:r+h, c:c+w][mask[:h, :w]] = ink


def _render(shape, bg, records, book):
    out = np.full(shape, bg, dtype=np.int8)
    for r, c, code in records:
        _stamp(out, r, c, *book[code])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The entire rule is recovered from this input alone."""
    inp = _grid(input_grid)
    bg, book, _, _, view = _decode(inp)
    live = _members(view, bg, set(book))[0]
    assert live, 'at least one live instruction required'
    return _render(inp.shape, bg, live, book)


def _grow(rng, h, w, n, diagonal=False):
    mask = np.zeros((h, w), dtype=bool)
    mask[int(rng.integers(h)), int(rng.integers(w))] = True
    steps = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
             if (a or b) and (diagonal or abs(a)+abs(b) == 1)]
    for _ in range(n-1):
        border = sorted({(int(r)+dr, int(c)+dc) for r, c in np.argwhere(mask)
                         for dr, dc in steps if 0 <= r+dr < h and 0 <= c+dc < w
                         and not mask[r+dr, c+dc]})
        mask[border[int(rng.integers(len(border)))]] = True
    return mask


def _delta(rng, h, w, style=None):
    if style == 'round':
        mask = np.zeros((h, w), dtype=bool)
        mask[0, 1:-1] = mask[-1, 1:-1] = True
        mask[1:-1, 0] = mask[1:-1, -1] = True
        return mask
    if style == 'padded':
        mask = np.zeros((h, w), dtype=bool)
        mask[-2:, -2:] = True
        return mask
    mode = int(rng.integers(5))
    if mode <= 1:
        return _grow(rng, h, w, int(rng.integers(2, min(h*w-2, 12)+1)), mode == 1)
    mask = np.zeros((h, w), dtype=bool)
    if mode == 2:  # Translated small hollow loop, possibly with blank padding.
        r, c = int(rng.integers(h-2)), int(rng.integers(w-2))
        mask[r:r+3, c:c+3] = True
        mask[r+1, c+1] = False
        if h*w == 9:
            mask[r, c] = False  # Retain at least two unchanged context cells.
    elif mode == 3:  # Variable asymmetric elbow.
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, :] = True
        mask[:, c] = True
    else:  # A short rectangle, not necessarily aligned to the panel origin.
        hh, ww = int(rng.integers(1, h)), int(rng.integers(2, w+1))
        r, c = int(rng.integers(h-hh+1)), int(rng.integers(w-ww+1))
        mask[r:r+hh, c:c+ww] = True
    if rng.random() < .5:
        mask = mask[::-1, :]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    assert 2 <= int(mask.sum()) <= h*w-2
    return mask.copy()


def _specimens(rng, mask, bg, ink, mode):
    """Flip a spatial mask, retaining both colors outside it in both specimens."""
    cells, other = np.flatnonzero(mask), np.flatnonzero(~mask)
    assert len(other) >= 2
    master = np.zeros(mask.shape, dtype=bool)
    n = int(rng.integers(1, len(other)))
    master.flat[rng.choice(other, n, replace=False)] = True
    if mode == 'balanced':
        assert len(cells) % 2 == 0
        n = len(cells)//2
    elif mode == 'add':
        n = 0
    elif mode == 'remove':
        n = len(cells)
    else:
        n = int(rng.integers(1, len(cells)))
    if n:
        master.flat[rng.choice(cells, n, replace=False)] = True
    variant = master ^ mask
    return np.where(master, ink, bg).astype(np.int8), np.where(variant, ink, bg).astype(np.int8)


def _rect(shape, r, c, h, w):
    result = np.zeros(shape, dtype=bool)
    result[r:min(r+h, shape[0]), c:min(c+w, shape[1])] = True
    return result


def _halo(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[a:a+h, b:b+w] for a in range(3) for b in range(3)])


def _make_pair(rng, palette, kind):
    bg, codes, inks = palette
    rich = kind == 'witness'
    if rich:
        artifacts = ['painted', 'attached', 'bare']
    elif kind == 'test':
        artifacts = list(rng.choice(['attached', 'bare'], int(rng.integers(3)), replace=False))
    else:
        artifacts = []
    book, specimens, copies = {}, {}, {}
    for j, (code, ink) in enumerate(zip(codes, inks)):
        if kind == 'test':
            # At least one axis is larger than every training stamp.
            h, w = (5, int(rng.integers(3, 6))) if rng.random() < .5 else (int(rng.integers(3, 6)), 5)
        else:
            h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        style = 'round' if rich and j == 0 else 'padded' if kind == 'offset' and j == 0 else None
        mask = _delta(rng, h, w, style)
        mode = 'balanced' if rich and j == 0 else 'remove' if kind == 'offset' and j == 0 else str(rng.choice(['add', 'remove', 'mixed']))
        master, variant = _specimens(rng, mask, bg, ink, mode)
        book[code] = (ink, mask)
        specimens[code] = (master, variant)
        copies[code] = 3 if artifacts else int(rng.integers(3, 6 if len(codes) == 1 else 5))
    nframes = sum(copies.values())
    ph = max(v[1].shape[0]+4 for v in book.values())
    pw = max(v[1].shape[1]+4 for v in book.values())
    layouts = [(nr, nc) for nr in range(1, 4) for nc in range(1, 4)
               if nr*nc >= nframes+max(1, len(artifacts))
               and nr*ph+3 <= 30 and nc*pw+3 <= 30]
    smallest = min(nr*nc for nr, nc in layouts)
    layouts = [(nr, nc) for nr, nc in layouts if nr*nc == smallest]
    nr, nc = layouts[int(rng.integers(len(layouts)))]
    H, W = (30, 30) if rich else (int(rng.integers(nr*ph+3, 31)), int(rng.integers(nc*pw+3, 31)))
    inp = np.full((H, W), bg, dtype=np.int8)
    slots = [(2+r*ph, 2+c*pw) for r in range(nr) for c in range(nc)]
    rng.shuffle(slots)
    tiles = [(code, i) for code in codes for i in range(copies[code])]
    rng.shuffle(tiles)
    frames = []
    odd_indices = {code: int(rng.integers(copies[code])) for code in codes}
    for (code, i), (r0, c0) in zip(tiles, slots):
        r, c = r0+int(rng.integers(2)), c0+int(rng.integers(2))
        master, variant = specimens[code]
        h, w = master.shape
        inp[r:r+h+2, c:c+w+2] = code
        inp[r+1:r+h+1, c+1:c+w+1] = variant if i == odd_indices[code] else master
        frames.append((r, c, r+h+2, c+w+2))
    records = []
    occupied = np.zeros(inp.shape, dtype=bool)
    specimen_windows = np.zeros(inp.shape, dtype=bool)
    for r, c, b, d in frames:
        specimen_windows[r:b, c:d] = True
    for material, (r0, c0) in zip(artifacts, slots[nframes:]):
        code = codes[0] if material == 'painted' else int(rng.choice(codes))
        ink, mask = book[code]
        r, c = r0+int(rng.integers(2)), c0+int(rng.integers(2))
        _stamp(inp, r, c, ink, mask)
        if material == 'attached':
            a, b = max(map(tuple, np.argwhere(mask)))
            inp[r+int(a)+1, c+int(b)+1] = code
        elif material == 'painted':
            assert not mask[0, 0] and mask[:2, :2].any()
            inp[r, c] = code
            records.append((r, c, code))
            occupied |= _rect(inp.shape, r, c, *mask.shape)

    def place(code, boundary='full'):
        nonlocal occupied
        ink, mask = book[code]
        h, w = mask.shape
        blocked = _halo(inp != bg) | specimen_windows
        rr, cc = np.indices(inp.shape)
        end_r, end_c = np.minimum(rr+h, H), np.minimum(cc+w, W)
        # Summed areas test all visible rectangles without allocating a full
        # canvas per candidate. Row-major candidates and RNG draws are unchanged.
        sums = np.pad(occupied.astype(np.int16).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        overlaps = sums[end_r, end_c]-sums[rr, end_c]-sums[end_r, cc]+sums[rr, cc]
        prefixes = mask.astype(np.int16).cumsum(0).cumsum(1)
        visible = prefixes[np.minimum(H-rr, h)-1, np.minimum(W-cc, w)-1] > 0
        allowed = ~blocked & (overlaps == 0) & visible
        bottom, right = rr+h > H, cc+w > W
        if boundary in ('full', 'through'):
            allowed &= ~bottom & ~right
        elif boundary == 'bottom':
            allowed &= bottom & ~right
        elif boundary == 'right':
            allowed &= right & ~bottom
        elif boundary == 'corner':
            allowed &= bottom & right
        if boundary == 'through':
            through = np.zeros(inp.shape, dtype=bool)
            for a, b in np.argwhere(mask):
                a, b = int(a), int(b)
                through[:H-a, :W-b] |= specimen_windows[a:, b:]
            allowed &= through
        locations = np.argwhere(allowed)
        if not len(locations):
            raise RuntimeError('no nonconflicting visible marker position')
        r, c = map(int, locations[int(rng.integers(len(locations)))])
        inp[r, c] = code
        records.append((r, c, code))
        occupied |= _rect(inp.shape, r, c, h, w)

    if rich:
        place(codes[0], 'through')
    # Full occurrences make every input-visible dictionary entry observable in output.
    for code in codes:
        if not any(k == code for _, _, k in records):
            place(code)
    if rich:
        # The rounded stencil has ink in its visible 2x2 corner. Restricting
        # placement to visible intersections also covers masks with padding.
        for boundary in ('corner', 'bottom', 'right'):
            place(codes[0], boundary)
    target = int(rng.integers(7, 10)) if kind == 'test' else int(rng.integers(max(4, len(records)), 7))
    while len(records) < target:
        place(int(rng.choice(codes)), 'any' if rich or kind == 'test' else 'full')
    # Construction records create labels; the decoder subsequently checks them
    # from returned pixels, without being passed these records or specimens.
    view = inp.copy()
    for r, c, b, d in frames:
        view[r:b, c:d] = bg
    assert _members(view, bg, set(codes))[0] == sorted(records)
    return {'input': inp, 'output': _render(inp.shape, bg, records, book)}


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    n = int(rng.integers(1, 3))
    colors = list(map(int, rng.permutation(10)))
    palette = colors[0], colors[1:1+n], colors[1+n:1+2*n]
    kinds = ['witness', 'offset'] + ['ordinary']*(nt-2)
    rng.shuffle(kinds)
    puzzle, seen = {'train': [], 'test': []}, set()
    for kind in kinds + ['test']*nq:
        # Bounded packing retries; failed candidates are not retained or logged.
        for _ in range(32):
            try:
                pair = _make_pair(rng, palette, kind)
            except RuntimeError:
                continue
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('bounded packing exhausted')
        seen.add(key)
        puzzle['test' if kind == 'test' else 'train'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    bg, book, details, frames, view = _decode(inp)
    _check_material(view, bg, book)
    live, attached, bare, painted = _members(view, bg, set(book))
    assert live and np.array_equal(out, _render(inp.shape, bg, live, book)), 'incorrect output'
    assert not np.array_equal(inp, out)
    occupied, frame_mask = np.zeros(inp.shape, bool), np.zeros(inp.shape, bool)
    for r, c, b, d in frames:
        assert not frame_mask[r:b, c:d].any()
        frame_mask[r:b, c:d] = True
    full_codes, boundaries, through = set(), set(), False
    for r, c, code in live:
        ink, mask = book[code]
        h, w = mask.shape
        rect = _rect(inp.shape, r, c, h, w)
        assert not (occupied & rect).any(), 'stamp windows conflict'
        occupied |= rect
        bottom, right = r+h > inp.shape[0], c+w > inp.shape[1]
        if bottom:
            boundaries.add('bottom')
        if right:
            boundaries.add('right')
        if bottom and right:
            boundaries.add('corner')
        if not bottom and not right:
            full_codes.add(code)
        ink_field = np.zeros(inp.shape, dtype=np.int8)
        _stamp(ink_field, r, c, 1, mask)
        through |= bool((ink_field.astype(bool) & frame_mask).any())
    assert full_codes == set(book), 'each code needs a full output occurrence'
    masks = {code: (ink, mask) for code, (ink, mask) in book.items()}
    alternatives = {}
    for name in ('master', 'variant', 'added', 'removed', 'trimmed'):
        alternative = {}
        for code, (ink, mask) in masks.items():
            if name in ('master', 'variant'):
                alt = details[code][name] == ink
            elif name == 'trimmed':
                r, c, b, d = _box(list(map(tuple, np.argwhere(mask))))
                alt = mask[r:b, c:d]
            else:
                alt = details[code][name]
            alternative[code] = (ink, alt)
        alternatives[name] = not np.array_equal(out, _render(inp.shape, bg, live, alternative))
    all_points = [(int(r), int(c), int(view[r, c])) for r, c in np.argwhere(np.isin(view, list(book)))]
    alternatives['all_markers'] = not np.array_equal(out, _render(inp.shape, bg, all_points, book))
    markers_only = np.full(inp.shape, bg, np.int8)
    for r, c, code in live:
        markers_only[r, c] = code
    alternatives['no_substitution'] = not np.array_equal(markers_only, out)
    return {'shape': inp.shape, 'palette': (bg, tuple((c, v[0]) for c, v in sorted(book.items()))),
            'sizes': tuple(v[1].shape for v in book.values()), 'codes': len(book),
            'frames': len(frames), 'live': len(live), 'attached': len(attached),
            'bare': len(bare), 'painted': len(painted), 'boundaries': boundaries,
            'balanced': any(d['added'].sum() == d['removed'].sum() > 0 for d in details.values()),
            'through_frames': through, 'alternatives': alternatives,
            'mask_keys': tuple(_key(v[1]) for v in book.values()),
            'delta_counts': tuple(int(v[1].sum()) for v in book.values()),
            'copy_counts': tuple(len(d['boxes']) for d in details.values())}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            records[split].append(_analyze(pair))
    train, tests = records['train'], records['test']
    assert len({r['palette'] for r in train+tests}) == 1, 'palette roles fixed within puzzle'
    assert any(r['balanced'] for r in train), 'show equal-count spatial disagreement'
    assert any(r['attached'] and r['bare'] and r['painted'] for r in train)
    assert set.union(*(r['boundaries'] for r in train)) == {'bottom', 'right', 'corner'}
    for name in ('master', 'variant', 'added', 'removed', 'trimmed', 'all_markers', 'no_substitution'):
        assert any(r['alternatives'][name] for r in train), f'undistinguished shortcut: {name}'
    assert any(r['through_frames'] for r in train), 'teach drawing over erased specimen positions'
    assert len({r['mask_keys'] for r in train}) >= 2, 'dictionary must visibly change between examples'
    assert all(max(max(h, w) for h, w in r['sizes']) <= 4 for r in train)
    assert all(all(max(h, w) == 5 for h, w in r['sizes']) for r in tests)
    assert min(r['live'] for r in tests) > max(r['live'] for r in train)


def _digest(p):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        h.update(bytes([len(p[split])]))
        for pair in p[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test() -> dict:
    """42 sequential seeds; retain hashes and small feature sets, not puzzles."""
    signatures, shapes, masks, layouts, palettes, copies, counts = (set() for _ in range(7))
    totals = Counter()
    sample = None
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_digest(p))
        if seed == 0:
            sample = p
        for split in ('train', 'test'):
            for pair in p[split]:
                m = _analyze(pair)
                shapes.add(m['shape'])
                masks.update(m['mask_keys'])
                palettes.add(m['palette'])
                copies.update(m['copy_counts'])
                counts.update(m['delta_counts'])
                layouts.add((m['shape'], m['sizes'], m['frames'], m['live']))
                for feature in ('attached', 'bare', 'painted', 'balanced', 'through_frames'):
                    totals[feature] += bool(m[feature])
                totals['pairs'] += 1
        assert _digest(p) == _digest(generate_puzzle(seed)), 'determinism'
    assert len(signatures) == 42 and len(palettes) >= 35
    assert len(shapes) >= 20 and len(masks) >= 70 and len(layouts) >= 100
    assert copies >= {3, 4, 5} and len(counts) >= 7
    assert all(totals[k] >= 42 for k in ('attached', 'bare', 'painted', 'balanced', 'through_frames'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(10000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Mutation checks preserve inputs and demonstrations when damaging a query.
    for mutation in ('missing_stamp', 'copied_specimen', 'missing_exception'):
        bad = {s: [{k: v.copy() for k, v in q.items()} for q in sample[s]] for s in sample}
        q = bad['test'][0]
        bg, book, details, frames, view = _decode(q['input'])
        r, c, code = _members(view, bg, set(book))[0][0]
        if mutation == 'missing_stamp':
            h, w = book[code][1].shape
            q['output'][r:r+h, c:c+w] = bg
        elif mutation == 'copied_specimen':
            q['output'] = _render(q['input'].shape, bg, _members(view, bg, set(book))[0],
                                  {k: (v[0], details[k]['variant'] == v[0]) for k, v in book.items()})
        else:
            for r, c, b, d in details[code]['boxes']:
                q['input'][r+1:b-1, c+1:d-1] = details[code]['master']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted '+mutation)
    return {'seeds': 42, 'pairs': totals['pairs'], 'distinct_puzzles': len(signatures),
            'input_shapes': len(shapes), 'difference_masks': len(masks), 'layouts': len(layouts),
            'palettes': len(palettes), 'copy_counts': sorted(copies), 'difference_counts': sorted(counts),
            'features': {k: totals[k] for k in ('attached', 'bare', 'painted', 'balanced', 'through_frames')},
            'semantic_scope': 'generator-owned invariants and counterfactuals; no independent solver'}
