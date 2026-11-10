"""Directional stencil receivers sample a hub's periodic ribbon.

No parent generator, examples, seed, or training data enter solve().
Coordinates (u,v) mean outward along a source and clockwise across its ribbon.
"""
from __future__ import annotations

import copy
import json
import numpy as np


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
HOST_SEEDS = (0, 27048335, 241082646, 161294956, 787357162, 172976544,
              155062126, 542250706, 206041327, 921171, 670680855, 579428376,
              96302860, 894296868, 72068476, 895344308, 23309515, 390799399,
              294804432, 897348972, 125070671, 96363940, 236316184, 384950104,
              936813313, 122136986, 610822499, 757628917, 820727754, 207883026,
              654788825, 760132031, 655818346, 46141076, 946924802, 569642205,
              489079199, 761296544, 896413997, 482175652, 947641224, 595239728)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(cells):
    left = set(cells)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in left:
                        left.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _point(source, u, v=0):
    r, c = source['origin']
    dr, dc = DIRS[source['direction']]
    return r + u * dr + v * dc, c + u * dc - v * dr


def _local(source, p):
    r, c = p[0] - source['origin'][0], p[1] - source['origin'][1]
    dr, dc = DIRS[source['direction']]
    return r * dr + c * dc, r * dc - c * dr


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _origin(box, direction):
    top, left, bottom, right = box
    return ((top, left), (top, right), (bottom, right), (bottom, left))[direction]


def _box(cells):
    return (min(p[0] for p in cells), min(p[1] for p in cells),
            max(p[0] for p in cells), max(p[1] for p in cells))


def _source_pixels(s):
    pixels = {_point(s, int(u), int(v)): s['code'] for u, v in np.argwhere(s['mask'])}
    h, w = s['mask'].shape
    pixels.update({_point(s, h + u, v): int(s['tile'][u, v])
                   for u in range(len(s['tile'])) for v in range(w)})
    return pixels


def _source_window(s):
    h, w = s['mask'].shape
    return {_point(s, u, v) for u in range(h + len(s['tile'])) for v in range(w)}


def _source_interpretations(grid, cells):
    """Recognize a notched monochrome body plus exactly one complete ribbon."""
    candidates = []
    for code in sorted({int(grid[p]) for p in cells}):
        body = {p for p in cells if int(grid[p]) == code}
        box = _box(body)
        bh, bw = box[2] - box[0] + 1, box[3] - box[1] + 1
        if not (3 <= bh <= 5 and 3 <= bw <= 5 and len(body) >= 5):
            continue
        if len(_components(body)) != 1 or len(body) == bh * bw:
            continue
        for direction in range(4):
            h, w = (bw, bh) if direction % 2 == 0 else (bh, bw)
            s = dict(code=code, origin=_origin(box, direction), direction=direction)
            mask = np.array([[_point(s, u, v) in body for v in range(w)]
                             for u in range(h)], dtype=bool)
            if not mask[-1].all():
                continue
            for length in range(1, 5):
                ribbon = [_point(s, h + u, v) for u in range(length) for v in range(w)]
                if body | set(ribbon) != cells or any(int(grid[p]) == code for p in ribbon):
                    continue
                s = dict(s, mask=mask, tile=np.array(
                    [[int(grid[_point(s, h + u, v)]) for v in range(w)]
                     for u in range(length)], dtype=np.int8))
                candidates.append(s)
    return candidates


def _analyze(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > grid.size / 2, 'background must occupy a majority'
    objects = _components(map(tuple, np.argwhere(grid != bg)))
    sources, rest, source_cells = {}, [], set()
    for cells in objects:
        choices = _source_interpretations(grid, cells)
        assert len(choices) <= 1, 'ambiguous transmitter'
        if choices:
            s = choices[0]
            assert s['code'] not in sources, 'one transmitter per code'
            sources[s['code']] = s
            source_cells.update(cells)
        else:
            rest.append(cells)
    assert 1 <= len(sources) <= 3, 'missing transmitter'
    codes = set(sources)
    signals = set().union(*(set(map(int, s['tile'].ravel())) for s in sources.values()))
    assert not (codes & signals) and bg not in signals
    live, ignored = [], []
    for cells in rest:
        if len(cells) == 1:
            p = next(iter(cells))
            code = int(grid[p])
            if code in sources:
                s = sources[code]
                u, v = _local(s, p)
                if v == 0 and u >= s['mask'].shape[0] + len(s['tile']) + 1:
                    live.append((code, p, u))
                    continue
        ignored.append(cells)
    # Explicit exclusion of stamp/source and stamp/stamp conflicts. Artifacts
    # are erased before rendering, so their footprints need not be protected.
    occupied = set()
    for s in sources.values():
        window = _source_window(s)
        assert not (occupied & window), 'overlapping transmitter rectangles'
        occupied.update(window)
    windows = []
    for code, p, distance in live:
        s = sources[code]
        h, w = s['mask'].shape
        window = {_point(s, distance + u, v) for u in range(h) for v in range(w)
                  if _inside(grid.shape, _point(s, distance + u, v))}
        assert not (occupied & window), 'overlapping live windows'
        occupied.update(window)
        windows.append(window)
    return dict(grid=grid, background=bg, sources=sources, source_cells=source_cells,
                live=sorted(live), ignored=ignored, windows=windows)


def _render(shape, bg, sources, live, *, phase=True, stencil=True, prefix=False, ordinal=False):
    out = np.full(shape, bg, dtype=np.int8)
    for s in sources.values():
        for p, color in _source_pixels(s).items():
            out[p] = color
    ordinal_distances = {}
    if ordinal:
        for code in sources:
            distances = sorted(u for c, p, u in live if c == code)
            ordinal_distances.update({(code, u): distances[0] + i for i, u in enumerate(distances)})
    for code, p, distance in live:
        s = sources[code]
        h, w = s['mask'].shape
        for u in range(h):
            for v in range(w):
                q = _point(s, distance + u, v)
                if (stencil and not s['mask'][u, v]) or not _inside(shape, q):
                    continue
                period = len(s['tile'])
                if prefix and period == 3 and np.array_equal(s['tile'][0], s['tile'][2]):
                    period = 2
                field_distance = ordinal_distances.get((code, distance), distance)
                index = (field_distance + u - h) % period if phase else u % period
                out[q] = s['tile'][index, v]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode all templates, rays, phases and singleton membership from input."""
    a = _analyze(input_grid)
    return _render(a['grid'].shape, a['background'], a['sources'], a['live'])


def _mask(rng, h, w):
    style = int(rng.integers(4))
    mask = np.zeros((h, w), dtype=bool)
    mask[-1] = True  # solid outbound face supplies the ribbon's full width
    if style == 0:  # loop
        mask[0] = True
        mask[:, 0] = mask[:, -1] = True
    elif style == 1:  # comb with an offset stem
        stem = int(rng.integers(w))
        mask[:, stem] = True
        endpoint = int(rng.integers(w))
        mask[0, min(stem, endpoint):max(stem, endpoint) + 1] = True
    elif style == 2:  # notched body
        mask[:] = True
        mask[int(rng.integers(h - 1)), int(rng.integers(w))] = False
        if h * w > 12:
            mask[int(rng.integers(h - 1)), int(rng.integers(w))] = False
    else:  # branched path with a full distal bar
        v = int(rng.integers(w))
        for u in range(h - 1):
            mask[u, v] = True
            if rng.random() < .7:
                mask[u, max(0, v - 1):min(w, v + 2)] = True
            v = int(np.clip(v + rng.integers(-1, 2), 0, w - 1))
    if rng.random() < .5:
        mask = mask[:, ::-1]
    assert mask[0].any() and mask[-1].all() and not mask.all()
    assert len(_components(map(tuple, np.argwhere(mask)))) == 1
    return mask.copy()


def _book(rng):
    n = int(rng.integers(1, 4))
    palette = list(map(int, rng.permutation(10)))
    codes, backgrounds = palette[:n], palette[n:n + 2]
    signals = palette[n + 2:n + 2 + int(rng.integers(2, min(4, 8 - n) + 1))]
    book = {}
    for i, code in enumerate(codes):
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        length = 3 if i == 0 else int(rng.integers(1, 5))
        tile = rng.choice(signals, size=(length, w)).astype(np.int8)
        if i == 0:
            tile[0, 0], tile[0, -1] = signals[0], signals[1]
            tile[1] = [signals[(signals.index(int(c)) + 1) % len(signals)] for c in tile[0]]
            tile[2] = tile[0]  # whole-word ABA witness, with vector asymmetry
        book[code] = dict(code=code, mask=_mask(rng, h, w), tile=tile)
    return book, backgrounds


def _maximum(shape, s):
    r, c = s['origin']
    return (shape[1] - c - 1, shape[0] - r - 1, c, r)[s['direction']]


def _window(shape, s, u):
    h, w = s['mask'].shape
    return {_point(s, u + i, j) for i in range(h) for j in range(w)
            if _inside(shape, _point(s, u + i, j))}


def _candidate(rng, shape, entry, kind, direction=None, origin=None):
    direction = int(rng.integers(4)) if direction is None else direction
    h, w = entry['mask'].shape
    bh, bw = (w, h) if direction % 2 == 0 else (h, w)
    if origin is None:
        top = int(rng.integers(shape[0] - bh + 1))
        left = int(rng.integers(shape[1] - bw + 1))
        origin = _origin((top, left, top + bh - 1, left + bw - 1), direction)
    s = dict(entry, origin=origin, direction=direction)
    pixels = _source_pixels(s)
    if any(not _inside(shape, p) for p in pixels):
        return None
    maximum = _maximum(shape, s)
    first_min = h + len(s['tile']) + 1
    if kind == 'phase':
        gap = h + 4
        gap += (2 - gap) % 3  # physical phase advances two; marker ordinal advances one
        last_first = maximum - (h - 1) - gap
        if last_first < first_min:
            return None
        first = int(rng.integers(first_min, min(first_min + 3, last_first) + 1))
        distances = [first, first + gap]
    elif kind == 'clip':
        last_first = maximum - 2 * h - 1
        if last_first < first_min:
            return None
        first = int(rng.integers(first_min, min(first_min + 3, last_first) + 1))
        distances = [first, int(rng.integers(maximum - h + 2, maximum + 1))]
    elif kind == 'dense':
        gap = h + int(rng.integers(1, 3))
        n = 4 if h == 3 and rng.random() < .5 else 3
        last_first = maximum - (n - 1) * gap
        if last_first < first_min:
            return None
        first = int(rng.integers(first_min, min(first_min + 2, last_first) + 1))
        distances = [first + i * gap for i in range(n)]
    else:
        last_first = maximum - h + 1
        if last_first < first_min:
            return None
        first = int(rng.integers(first_min, last_first + 1))
        distances = [first]
        if kind == 'random' and maximum >= first + 2 * h + 1 and rng.random() < .5:
            distances.append(int(rng.integers(first + h + 2, maximum + 1)))
    live = [(s['code'], _point(s, u), u) for u in distances]
    foreground = set(pixels) | {p for _, p, _ in live}
    windows = _source_window(s).union(*(_window(shape, s, u) for u in distances))
    return s, live, foreground, windows


def _artifact(s, origin, ink, attached):
    h, w = s['mask'].shape
    shadow = dict(s, origin=origin)
    pixels = {_point(shadow, int(u), int(v)): ink for u, v in np.argwhere(s['mask'])}
    if attached:
        pixels[_point(shadow, h, w)] = s['code']
    return pixels


def _add_artifacts(rng, grid, sources, live, require_attached, fallback=False):
    bg = int(np.argmax(np.bincount(grid.ravel(), minlength=10)))
    foreground = set(map(tuple, np.argwhere(grid != bg)))
    primary = next(iter(sources.values()))
    h, w = primary['mask'].shape
    ink = int(primary['tile'][0, 0])
    if fallback and require_attached:
        # The parallel escape scene reserves a band above the primary source.
        # These three placements are guaranteed by its two-receiver gap.
        first = min(u for code, p, u in live if code == primary['code'])
        pieces = [_artifact(primary, _point(primary, first + 2, -w), ink, True),
                  {_point(primary, first + h + 1, w - 1): primary['code']},
                  _artifact(primary, (0, 0), ink, False)]
        for pixels in pieces:
            cells = set(pixels)
            assert all(_inside(grid.shape, p) for p in cells)
            assert not (_halo(cells) & foreground)
            for p, color in pixels.items():
                grid[p] = color
            foreground.update(cells)
        return True
    # Distal diagonal attachment: its code point lies ON the proper reference
    # ray. Only component membership prevents this false request.
    attached_ok = False
    candidates = list(range(h + len(primary['tile']) + 1, _maximum(grid.shape, primary) + 1))
    rng.shuffle(candidates)
    for distance in candidates:
        origin = _point(primary, distance - h, -w)
        pixels = _artifact(primary, origin, ink, True)
        cells = set(pixels)
        if all(_inside(grid.shape, p) for p in cells) and not (_halo(cells) & foreground):
            for p, color in pixels.items():
                grid[p] = color
            foreground.update(cells)
            attached_ok = True
            break
    if require_attached and not attached_ok:
        return False
    # Bare stamps and off-ray singleton codes are independent erasure witnesses.
    live_windows = set().union(*(_window(grid.shape, sources[code], u) for code, p, u in live))
    for kind in ('offray', 'bare'):
        placed = False
        positions = list(map(tuple, np.ndindex(grid.shape)))
        rng.shuffle(positions)
        for p in positions:
            if kind == 'bare':
                pixels = _artifact(primary, p, ink, False)
            else:
                u, v = _local(primary, p)
                if not (0 < v < w and u >= h + len(primary['tile']) + 1) or p in live_windows:
                    continue
                pixels = {p: primary['code']}
            cells = set(pixels)
            if any(not _inside(grid.shape, q) for q in cells) or _halo(cells) & foreground:
                continue
            for q, color in pixels.items():
                grid[q] = color
            foreground.update(cells)
            placed = True
            break
        if require_attached and not placed:
            return False
    return True


def _make_pair(rng, book, bg, mode, all_codes, common_direction, artifacts, primary_required=False):
    entries = list(book.values())
    if not all_codes:
        indices = sorted(map(int, rng.choice(len(entries), int(rng.integers(1, len(entries) + 1)), replace=False)))
        if primary_required and 0 not in indices:
            indices.insert(0, 0)
        entries = [entries[i] for i in indices]
    for scene in range(16):
        shape = ((int(rng.integers(25, 31)), int(rng.integers(25, 31)))
                 if mode in ('phase', 'clip', 'dense') else
                 (int(rng.integers(18, 31)), int(rng.integers(18, 31))))
        fallback = scene == 15
        if fallback:
            shape = (30, 30)
        sources, live, foreground, occupied = {}, [], set(), set()
        row = entries[0]['mask'].shape[1] + 1
        for index, entry in enumerate(entries):
            kind = mode if index == 0 else 'single'
            for attempt in range(180 if not fallback else 1):
                candidate = _candidate(rng, shape, entry, kind,
                                       direction=0 if fallback else common_direction,
                                       origin=(row, 0) if fallback else None)
                if candidate is None:
                    continue
                s, markers, fg, windows = candidate
                if _halo(fg) & foreground or windows & occupied:
                    continue
                sources[s['code']] = s
                live.extend(markers)
                foreground.update(fg)
                occupied.update(windows)
                row += s['mask'].shape[1] + 3
                break
            else:
                break
        if len(sources) != len(entries):
            continue
        grid = np.full(shape, bg, dtype=np.int8)
        for s in sources.values():
            for p, color in _source_pixels(s).items():
                grid[p] = color
        for code, p, distance in live:
            grid[p] = code
        if artifacts and not _add_artifacts(rng, grid, sources, live, mode == 'phase', fallback):
            continue
        out = _render(shape, bg, sources, live)
        a = _analyze(grid)
        assert a['live'] == sorted(live), 'construction changed receiver membership'
        assert np.array_equal(solve(grid), out)
        return {'input': grid, 'output': out}
    raise AssertionError('bounded scene construction exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    book, backgrounds = _book(rng)
    regime = int(rng.integers(3))  # denser, changed spacing, mixed directions
    direction = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + nq):
        train = i < nt
        mode = ('phase' if i == 0 else 'clip' if i == 1 else 'random') if train else ('dense' if regime == 0 else 'single')
        all_codes = i < 2 or (not train and regime == 2)
        common = (direction + i) % 4 if i < 2 else None
        if not train and regime != 2:
            common = (direction + 2 + i - nt) % 4
        artifacts = i == 0 or (mode != 'dense' and rng.random() < .55)
        for retry in range(20):
            pair = _make_pair(rng, book, backgrounds[i % 2], mode, all_codes, common, artifacts,
                              primary_required=not train)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise AssertionError('distinct input construction exhausted')
        seen.add(key)
        puzzle['train' if train else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(s):
    return s['mask'].shape, s['mask'].tobytes(), s['tile'].shape, s['tile'].tobytes()


def _artifact_kind(a, cells):
    grid, sources = a['grid'], a['sources']
    signals = set().union(*(set(map(int, s['tile'].ravel())) for s in sources.values()))
    if len(cells) == 1:
        p = next(iter(cells))
        assert int(grid[p]) in sources, 'unknown singleton'
        return 'offray'
    actual = {p: int(grid[p]) for p in cells}
    for s in sources.values():
        h, w = s['mask'].shape
        code_points = [p for p in cells if int(grid[p]) == s['code']]
        if len(code_points) == 1:
            p = code_points[0]
            dr, dc = DIRS[s['direction']]
            origin = (p[0] - h * dr - w * dc, p[1] - h * dc + w * dr)
            inks = {int(grid[q]) for q in cells if q != p}
            if len(inks) == 1 and inks <= signals and actual == _artifact(s, origin, inks.pop(), True):
                return 'attached'
        if len({int(grid[p]) for p in cells}) == 1:
            ink = int(grid[next(iter(cells))])
            origin = _origin(_box(cells), s['direction'])
            if ink in signals and actual == _artifact(s, origin, ink, False):
                return 'bare'
    raise AssertionError('unexplained discarded material')


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, book, training, testing = set(), {}, [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert a['live'], 'each generated scene has a real receiver'
            expected = _render(inp.shape, a['background'], a['sources'], a['live'])
            assert np.array_equal(out, expected), 'wrong mask, phase, membership or preservation'
            for code, s in a['sources'].items():
                if code in book:
                    assert _signature(s) == book[code], 'codebook changes between examples'
                elif split == 'train':
                    book[code] = _signature(s)
                else:
                    raise AssertionError('untaught test source')
            a['features'] = {_artifact_kind(a, cells) for cells in a['ignored']}
            a['clipped'] = any(u + a['sources'][code]['mask'].shape[0] - 1 > _maximum(inp.shape, a['sources'][code])
                               for code, p, u in a['live'])
            (training if split == 'train' else testing).append(a)
    taught = set().union(*(a['features'] for a in training))
    backgrounds = {a['background'] for a in training + testing}
    signals = set().union(*(set(map(int, s['tile'].ravel()))
                            for a in training + testing for s in a['sources'].values()))
    assert 1 <= len(book) <= 3 and 2 <= len(signals) <= 4 and len(backgrounds) == 2
    assert not (set(book) & signals or backgrounds & (set(book) | signals))
    assert {'attached', 'bare', 'offray'} <= taught
    assert all(a['features'] <= taught for a in testing)
    assert any(a['clipped'] for a in training), 'teach boundary clipping'
    assert len({a['background'] for a in training}) == 2
    for code in book:
        assert sum(any(c == code and u + a['sources'][c]['mask'].shape[0] - 1 <= _maximum(a['grid'].shape, a['sources'][c])
                       for c, p, u in a['live']) for a in training) >= 2, 'two full demonstrations per code'
    # Quantitative ablations: preserve transmission sources in both alternatives.
    assert any(not np.array_equal(_render(a['grid'].shape, a['background'], a['sources'], a['live'], phase=False),
                                  solve(a['grid'])) for a in training), 'show physical phase, not reset-at-anchor'
    assert any(not np.array_equal(_render(a['grid'].shape, a['background'], a['sources'], a['live'], stencil=False),
                                  solve(a['grid'])) for a in training), 'show shape rather than solid ribbons'
    assert any(not np.array_equal(_render(a['grid'].shape, a['background'], a['sources'], a['live'], prefix=True),
                                  solve(a['grid'])) for a in training), 'distinguish ABA from guessed AB'
    assert any(not np.array_equal(_render(a['grid'].shape, a['background'], a['sources'], a['live'], ordinal=True),
                                  solve(a['grid'])) for a in training), 'physical gaps rather than receiver ordinal'
    # A whole word of ABA must be shown continuing as ABA, not the guessed AB.
    whole_word = False
    membership_witness = False
    for a in training:
        for code, s in a['sources'].items():
            tile = s['tile']
            if len(tile) == 3 and np.array_equal(tile[0], tile[2]) and not np.array_equal(tile[0], tile[1]):
                full = [u for c, p, u in a['live'] if c == code and u + s['mask'].shape[0] - 1 <= _maximum(a['grid'].shape, s)]
                if len({u % 3 for u in full}) >= 2:
                    whole_word = True
        for cells in a['ignored']:
            for p in cells:
                code = int(a['grid'][p])
                if code in a['sources'] and len(cells) > 1:
                    s = a['sources'][code]
                    u, v = _local(s, p)
                    if v == 0 and u >= s['mask'].shape[0] + len(s['tile']) + 1:
                        extra = a['live'] + [(code, p, u)]
                        membership_witness |= not np.array_equal(
                            _render(a['grid'].shape, a['background'], a['sources'], extra), solve(a['grid']))
    assert whole_word, 'teach whole-word phase at unequal marker gaps'
    assert membership_witness, 'attached code on a live ray must be rejected'


def _semantic_fixture():
    """Literal expected receiver pixels, independent of the output renderer."""
    s = dict(code=7, origin=(4, 1), direction=0,
             mask=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
             tile=np.array([[2, 3, 4], [3, 4, 2], [2, 3, 4]], dtype=np.int8))
    inp, expected = np.zeros((12, 24), dtype=np.int8), np.zeros((12, 24), dtype=np.int8)
    for p, color in _source_pixels(s).items():
        inp[p] = expected[p] = color
    for u in (8, 16, 22):
        inp[_point(s, u)] = 7
    for pixels in (_artifact(s, _point(s, 10, -3), 2, True),
                   _artifact(s, (0, 19), 2, False), {_point(s, 11, 2): 7}):
        for p, color in pixels.items():
            inp[p] = color
    for distance, rows in ((8, ((2, 3, 4), (2, 0, 4), (3, 4, 2))),
                           (16, ((3, 4, 2), (2, 0, 4), (2, 3, 4))),
                           (22, ((3, 4, 2),))):
        for u, row in enumerate(rows):
            for v, color in enumerate(row):
                expected[_point(s, distance + u, v)] = color
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(expected, turns))
    return {'input': inp, 'output': expected}


def self_test():
    """Bounded structural/semantic checks on the 42 required host seeds."""
    _semantic_fixture()
    puzzles, shapes, masks, tiles, geometry, directions, backgrounds = [], set(), set(), set(), set(), set(), set()
    clipped = attached = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyze(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add(pair['input'].shape)
                backgrounds.add(a['background'])
                attached += sum(_artifact_kind(a, cells) == 'attached' for cells in a['ignored'])
                for s in a['sources'].values():
                    masks.add((s['mask'].shape, s['mask'].tobytes()))
                    tiles.add((s['tile'].shape, s['tile'].tobytes()))
                    directions.add(s['direction'])
                for c, point, u in a['live']:
                    s = a['sources'][c]
                    geometry.add((len(a['sources']), len(a['live']), s['direction'], s['mask'].shape,
                                  len(s['tile']), u, u % len(s['tile'])))
                    clipped += u + s['mask'].shape[0] - 1 > _maximum(a['grid'].shape, s)
    signatures = [tuple(_key(pair['input']) for split in ('train', 'test') for pair in p[split]) for p in puzzles]
    assert len(set(signatures)) == 42
    assert len(shapes) >= 30 and len(masks) >= 35 and len(tiles) >= 50 and len(geometry) >= 150
    assert directions == {0, 1, 2, 3} and len(backgrounds) >= 8 and clipped >= 42 and attached >= 42
    assert signatures[0] == tuple(_key(pair['input']) for split in ('train', 'test')
                                 for pair in generate_puzzle(0)[split])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(19, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    bad = copy.deepcopy(puzzles[0])
    a = _analyze(bad['test'][0]['input'])
    code, point, u = a['live'][0]
    s = a['sources'][code]
    for q in _window(a['grid'].shape, s, u):
        bad['test'][0]['output'][q] = a['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('deleted receiver stamp accepted')
    return {'seeds': 42, 'input_shapes': len(shapes), 'stencil_masks': len(masks),
            'ribbon_words': len(tiles), 'receiver_geometries': len(geometry),
            'directions': sorted(directions), 'backgrounds': len(backgrounds),
            'clipped_receivers': int(clipped), 'attached_distractors': attached}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
