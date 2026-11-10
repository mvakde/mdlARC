"""Registered shadow sorting: exact swatches identify full stencil strips.

Only standard library and NumPy. All reference information is in the input.
The reference's double frame stays; the swatches' single frames disappear.
"""
from __future__ import annotations

import numpy as np


def _transform(a, i):
    return np.rot90(np.fliplr(a) if i >= 4 else a, i % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return tuple(a.shape), a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid, colors 0..9")
    return a.astype(np.int8, copy=True)


def _parse(value):
    """Read geometry and roles; there are no hardcoded literal colors."""
    original = _grid(value)
    bg = int(original[0, 0])
    rows = [r for r in range(1, original.shape[0] - 1)
            if original[r, 0] == original[r, -1] != bg]
    cols = [c for c in range(1, original.shape[1] - 1)
            if original[0, c] == original[-1, c] != bg]
    if len(rows) + len(cols) != 1:
        raise ValueError("rail must have exactly one pair of boundary endpoints")
    vertical = bool(cols)
    g = original.T.copy() if vertical else original
    rail = int(cols[0] if vertical else rows[0])
    carrier = int(g[rail, 0])
    references = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, carrier):
            continue
        rr, cc = np.nonzero(g == color)
        t, b, l, r = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        if b - t < 8 or r - l < 8:
            continue
        rect = g[t:b + 1, l:r + 1]
        border = np.ones(rect.shape, dtype=bool)
        border[2:-2, 2:-2] = False
        interior = rect[2:-2, 2:-2]
        colors = set(int(x) for x in np.unique(interior))
        if (np.all(rect[border] == color) and color not in colors
                and bg in colors and len(colors) == 2):
            ink = next(x for x in colors if x != bg)
            if ink != carrier and t + 2 <= rail <= b - 2:
                references.append((t, b, l, r, color, ink, interior == ink))
    if len(references) != 1:
        raise ValueError("expected one double-framed binary reference")
    t, b, l, r, frame, ink, mask = references[0]
    if not (l == 1 or r == g.shape[1] - 2):
        raise ValueError("reference must dock at one end of the rail")
    sides = []
    for c in range(1, g.shape[1] - 1):
        if l <= c <= r or g[rail, c] != carrier:
            continue
        top = bottom = rail
        while top > 0 and g[top - 1, c] == carrier:
            top -= 1
        while bottom + 1 < g.shape[0] and g[bottom + 1, c] == carrier:
            bottom += 1
        if bottom - top >= 3:
            sides.append((c, top, bottom))
    if len(sides) % 2:
        raise ValueError("incomplete swatch frame")
    slots = []
    for left, right in zip(sides[::2], sides[1::2]):
        x, top, bottom = left
        end, top2, bottom2 = right
        if (top, bottom) != (top2, bottom2) or end <= x + 1:
            raise ValueError("malformed swatch rectangle")
        rect = g[top:bottom + 1, x:end + 1]
        if not (np.all(rect[0] == carrier) and np.all(rect[-1] == carrier)):
            raise ValueError("open swatch frame")
        patch = rect[1:-1, 1:-1]
        palette = set(int(z) for z in np.unique(patch))
        if bg not in palette or len(palette) != 2:
            raise ValueError("swatch must show both binary roles")
        accent = next(z for z in palette if z != bg)
        if accent in (carrier, frame, ink):
            raise ValueError("swatch accent collides with a reference role")
        anchor = rail - top - 1
        binary = (patch == accent).astype(np.int8)
        if not 0 <= anchor < patch.shape[0] or not binary[anchor].any():
            raise ValueError("rail must register a visible ink row of the swatch")
        slots.append(dict(x=x + 1, width=end - x - 1, color=accent,
                          top=top, bottom=bottom, anchor=anchor, mask=binary))
    if not 2 <= len(slots) <= 4:
        raise ValueError("expected two to four swatches")
    rebuilt = np.full(g.shape, bg, dtype=np.int8)
    rebuilt[rail] = carrier
    rebuilt[t:b + 1, l:r + 1] = g[t:b + 1, l:r + 1]
    for s in slots:
        x, w, top, bottom = s['x'], s['width'], s['top'], s['bottom']
        rebuilt[top:bottom + 1, x - 1:x + w + 1] = carrier
        rebuilt[top + 1:bottom, x:x + w] = np.where(s['mask'], s['color'], bg)
    if not np.array_equal(rebuilt, g):
        raise ValueError("unexplained cells or overlapping frames")
    return dict(grid=g, background=bg, carrier=carrier, rail=rail,
                vertical=vertical, reference=mask.astype(np.int8),
                reference_frame=frame, reference_ink=ink,
                reference_box=(t, b, l, r), slots=slots)


def _match(reference, patch, anchor):
    """Match ALL D4 orientations and ALL offsets, then measure full-strip bounds."""
    h, w = patch.shape
    matches = []
    for i in range(8):
        full = _transform(reference, i)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        rr, cc = np.nonzero(np.all(windows == patch, axis=(-2, -1)))
        for row, col in zip(rr, cc):
            row, col = int(row), int(col)
            occupied = np.flatnonzero(np.any(full[:, col:col + w], axis=1))
            origin = row + anchor
            span = (int(occupied[0]) - origin, int(occupied[-1]) - origin)
            matches.append((i, row, col, span))
    if not matches or len({m[3] for m in matches}) != 1:
        raise ValueError("all exact swatch matches must imply one registered interval")
    return matches[0]


def _analyze(value):
    a = _parse(value)
    tied = {}
    for s in a['slots']:
        i, row, col, span = _match(a['reference'], s['mask'], s['anchor'])
        s.update(orientation=i, source_row=row, source_col=col, span=span,
                 length=span[1] - span[0] + 1)
        full = _transform(a['reference'], i)
        center = np.flatnonzero(full[:, col + s['width'] // 2])
        origin = row + s['anchor']
        s['center_span'] = ((int(center[0]) - origin, int(center[-1]) - origin)
                            if len(center) else None)
        if tied.setdefault(s['length'], span) != span:
            raise ValueError("equal lengths must have identical registered intervals")
        if not (span[0] <= 0 <= span[1]
                and 1 <= a['rail'] + span[0]
                and a['rail'] + span[1] < a['grid'].shape[0] - 1):
            raise ValueError("shadow must cross the rail and fit with a margin")
    return a


def _base_output(a):
    g = a['grid']
    out = np.full(g.shape, a['background'], dtype=np.int8)
    out[a['rail']] = a['carrier']
    t, b, l, r = a['reference_box']
    out[t:b + 1, l:r + 1] = g[t:b + 1, l:r + 1]
    return out


def _paint(a, spans):
    out = _base_output(a)
    for s, (lo, hi) in zip(a['slots'], spans):
        out[a['rail'] + lo:a['rail'] + hi + 1, s['x']:s['x'] + s['width']] = s['color']
    return out.T.copy() if a['vertical'] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive everything from this input, including its complete reference."""
    a = _analyze(input_grid)
    remaining = [s['span'] for s in a['slots']]
    ordered = []
    while remaining:
        span = min(remaining, key=lambda p: p[1] - p[0])
        remaining.remove(span)
        ordered.append(span)
    return _paint(a, ordered)


def _bank(reference):
    """Finite exact-crop index; construction never relies on hidden disambiguation."""
    tokens = {}
    for i in range(8):
        full = _transform(reference, i)
        H, W = full.shape
        for w in range(1, min(4, W) + 1):
            for h in range(2, H + 1):
                if (h, w) == (H, W):
                    continue
                for c in range(W - w + 1):
                    occupied = np.flatnonzero(np.any(full[:, c:c + w], axis=1))
                    if not len(occupied):
                        continue
                    lo, hi = int(occupied[0]), int(occupied[-1])
                    for r in range(H - h + 1):
                        patch = full[r:r + h, c:c + w]
                        if not (patch.any() and np.any(patch == 0)):
                            continue
                        key = _key(patch)
                        interval = (lo - r, hi - r)
                        tokens.setdefault(key, {}).setdefault(interval,
                            (i, r, c, lo, hi, patch.copy()))
    result = []
    for alternatives in tokens.values():
        if len(alternatives) != 1:
            continue
        value = next(iter(alternatives.values()))
        i, r, c, lo, hi, patch = value
        h, w = patch.shape
        visible = max(0, min(hi, r + h - 1) - max(lo, r) + 1)
        anchors = [k for k in range(h) if patch[k].any()]
        # Require completion: at least one endpoint lies beyond the viewport.
        hidden = lo < r or hi >= r + h
        center = np.flatnonzero(_transform(reference, i)[:, c + w // 2])
        center_interval = (int(center[0]) - r, int(center[-1]) - r) if len(center) else None
        result.append(dict(mask=patch, h=h, width=w, orientation=i,
                           source_row=r, source_col=c, lo=lo, hi=hi,
                           length=hi - lo + 1, anchors=anchors, hidden=hidden,
                           union_diff=center_interval != (lo - r, hi - r),
                           fraction=visible / (hi - lo + 1)))
    return result


def _reference(rng):
    H, W = int(rng.integers(8, 13)), int(rng.integers(5, 9))
    spine = int(rng.integers(2, H - 2))
    a = np.zeros((H, W), dtype=np.int8)
    # Connected, irregular comb/loop silhouette; texture holes preserve a spine.
    for c in range(W):
        lo, hi = int(rng.integers(spine + 1)), int(rng.integers(spine, H))
        a[lo:hi + 1, c] = 1
    holes = rng.random((H, W)) < float(rng.uniform(.12, .28))
    holes[spine] = False
    a[holes] = 0
    # Occasional local rung reinforces readable repetition without fixed bitmap.
    if rng.random() < .4:
        r = int(rng.integers(H))
        c = int(rng.integers(W - 2))
        a[r, c:c + 3] = 1
    return a


def _select(rng, bank, n, budget, witness=0, test=False):
    pools = []
    for j in range(n):
        pool = [s for s in bank if s['hidden']
                and (s['fraction'] >= .65 if witness else s['fraction'] <= .60 if test else True)
                and (s['orientation'] == (0, 1, 4)[j] if witness == 1 else True)]
        if not pool:
            return None
        pools.append(pool)
    for _ in range(700):
        selected = [pool[int(rng.integers(len(pool)))].copy() for pool in pools]
        if sum(s['width'] for s in selected) > budget:
            continue
        if len({_key(s['mask']) for s in selected}) < n:
            continue
        templates = {}
        for s in selected:
            k = int(rng.choice(s['anchors']))
            s['anchor'] = k
            s['span'] = (s['lo'] - s['source_row'] - k, s['hi'] - s['source_row'] - k)
            length = s['length']
            if length in templates:
                # Choose an anchor producing the same interval, or reject.
                lo, hi = templates[length]
                k = s['lo'] - s['source_row'] - lo
                if k not in s['anchors']:
                    break
                s['anchor'], s['span'] = k, (lo, hi)
            templates[length] = s['span']
        else:
            if witness:
                if len(templates) != n:
                    continue
                selected.sort(key=lambda s: s['length'])
                # Cyclic order distinguishes sort from reversal for three slots.
                selected = selected[1:] + selected[:1]
                spans = [s['span'] for s in selected]
                wanted = [s['span'] for s in sorted(selected, key=lambda s: s['length'])]
                visible = sorted([(-s['anchor'], s['h'] - s['anchor'] - 1) for s in selected],
                                 key=lambda z: z[1] - z[0])
                if visible == wanted:
                    continue
                if not any(abs(lo + hi) > 1 for lo, hi in spans):
                    continue
                if witness == 1:
                    if (len({s['width'] for s in selected}) < 2
                            or not any(s['union_diff'] for s in selected)
                            or not any(a['length'] < b['length'] and a['length'] * a['width'] > b['length'] * b['width']
                                       for a in selected for b in selected)
                            or [s['span'] for s in sorted(selected, key=lambda s: s['length'] * s['width'])] == wanted
                            or [s['span'] for s in sorted(selected, key=lambda s: s['h'])] == wanted):
                        continue
                return selected
            rng.shuffle(selected)
            return selected
    return None


def _counts(rng, nt, nq):
    for name, value, lo, hi in [('num_train', nt, 2, 10), ('num_test', nq, 1, 4)]:
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} outside its legal integer range')
    if nq is None:
        nq = int(rng.integers(1, min(4, (nt or 10) - 1) + 1))
    if nt is None:
        nt = int(rng.integers(max(2, nq + 1), 11))
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    return int(nt), int(nq)


def _construct(rng, reference, selected, roles, vertical, dock_right):
    bg, carrier, frame, ink = roles[:4]
    n = len(selected)
    RH, RW = reference.shape
    minimum_width = RW + 4 + sum(s['width'] + 2 for s in selected) + n + 2
    W = int(rng.integers(minimum_width, 31))
    # Reference and every possible reassigned interval must fit with margins.
    dock_anchor = int(rng.integers(RH))
    upper = max(dock_anchor + 2, *(max(s['anchor'] + 1, -s['span'][0]) for s in selected))
    lower = max(RH - dock_anchor + 1, *(max(s['h'] - s['anchor'], s['span'][1]) for s in selected))
    minimum_height = upper + lower + 3
    H = int(rng.integers(minimum_height, min(30, minimum_height + 6) + 1))
    rail = int(rng.integers(upper + 1, H - lower - 1))
    g = np.full((H, W), bg, dtype=np.int8)
    g[rail] = carrier
    ref_x = W - RW - 5 if dock_right else 1
    ref_y = rail - dock_anchor - 2
    g[ref_y:ref_y + RH + 4, ref_x:ref_x + RW + 4] = frame
    g[ref_y + 2:ref_y + RH + 2, ref_x + 2:ref_x + RW + 2] = np.where(reference, ink, bg)
    # Allocate extra parallel space as irregular gaps, including both margins.
    gaps = [1] * (n + 1)
    for j, extra in enumerate(rng.multinomial(W - minimum_width, [1 / (n + 1)] * (n + 1))):
        gaps[j] += int(extra)
    x = gaps[0] if dock_right else ref_x + RW + 4 + gaps[0]
    pool = roles[4:]
    colors = [int(rng.choice(pool)) for _ in range(n)]
    # At least two slot colors make fixed-color assignment visible.
    if len(set(colors)) == 1:
        colors[-1] = next(int(c) for c in pool if c != colors[0])
    slots = []
    for j, (s, color) in enumerate(zip(selected, colors)):
        top = rail - s['anchor'] - 1
        h, w = s['mask'].shape
        g[top:top + h + 2, x:x + w + 2] = carrier
        g[top + 1:top + h + 1, x + 1:x + w + 1] = np.where(s['mask'], color, bg)
        slots.append(dict(x=x + 1, width=w, color=color, span=s['span']))
        x += w + 2 + gaps[j + 1]
    # Generation labels use recorded crop coordinates, not solve or _analyze.
    out = np.full((H, W), bg, dtype=np.int8)
    out[rail] = carrier
    out[ref_y:ref_y + RH + 4, ref_x:ref_x + RW + 4] = g[ref_y:ref_y + RH + 4, ref_x:ref_x + RW + 4]
    for slot, template in zip(slots, sorted(selected, key=lambda s: s['length'])):
        lo, hi = template['span']
        out[rail + lo:rail + hi + 1, slot['x']:slot['x'] + slot['width']] = slot['color']
    return {'input': g.T.copy() if vertical else g,
            'output': out.T.copy() if vertical else out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    roles = [int(x) for x in rng.permutation(10)]
    # Finite template rejection checks feasibility of both instructional witnesses.
    for _ in range(120):
        reference = _reference(rng)
        bank = _bank(reference)
        if len({_key(_transform(reference, i)) for i in range(8)}) != 8:
            continue
        witness1 = _select(rng, bank, 3, 15 - reference.shape[1], witness=1)
        witness2 = _select(rng, bank, 2, 18 - reference.shape[1], witness=2)
        small = _select(rng, bank, 2, 18 - reference.shape[1], test=True)
        if witness1 and witness2 and small:
            break
    else:
        raise RuntimeError('could not construct a sufficiently informative reference')
    axis_mode = int(rng.integers(3))
    axes = ([False] * (nt + nq) if axis_mode == 0 else [True] * (nt + nq) if axis_mode == 1
            else [False, True] + [bool(rng.integers(2)) for _ in range(nt + nq - 2)])
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt + nq):
        split = 'train' if index < nt else 'test'
        if index < 2:
            selected = (witness1, witness2)[index]
            shown = reference
        else:
            # Reference display can reflect/rotate, with axis exchange where it fits.
            shown = _transform(reference, int(rng.integers(8)))
            shown_bank = _bank(shown)
            allowed = [n for n in (2, 3, 4) if 24 - shown.shape[1] - 3 * n >= n]
            # Width budget: 30 - reference frame - 2n sides - n gaps - 2 margins.
            for _ in range(60):
                n = int(rng.choice(allowed))
                budget = 24 - shown.shape[1] - 3 * n
                selected = _select(rng, shown_bank, n, budget, test=split == 'test')
                if selected:
                    break
            else:
                # A proven two-slot crop choice remains usable in the original display.
                shown, selected = reference, small if split == 'test' else witness2
        for attempt in range(100):
            pair = _construct(rng, shown, selected, roles, axes[index], bool(rng.integers(2)))
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('distinct-input geometry sampler exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    slots = a['slots']
    spans = [s['span'] for s in slots]
    ordered = [s['span'] for s in sorted(slots, key=lambda s: s['length'])]
    return dict(count=len(slots), shape=tuple(a['grid'].shape), vertical=a['vertical'],
                lengths=tuple(s['length'] for s in slots), widths=tuple(s['width'] for s in slots),
                heights=tuple(s['mask'].shape[0] for s in slots),
                orientations=tuple(s['orientation'] for s in slots),
                offsets=tuple(spans), hidden=any(s['span'][0] < -s['anchor']
                    or s['span'][1] >= s['mask'].shape[0] - s['anchor'] for s in slots),
                changed=spans != ordered, nonreversal=spans[::-1] != ordered,
                area_diff=[s['span'] for s in sorted(slots, key=lambda s: s['length'] * s['width'])] != ordered,
                area_inversion=any(x['length'] < y['length'] and x['length'] * x['width'] > y['length'] * y['width']
                                   for x in slots for y in slots),
                height_diff=[s['span'] for s in sorted(slots, key=lambda s: s['mask'].shape[0])] != ordered,
                offset_moved=any(x != y and abs(sum(y)) > 1 for x, y in zip(spans, ordered)),
                tied=len(set(s['length'] for s in slots)) < len(slots))


def _check_pair(pair):
    """Relational invariant: conservation plus order determines the answer."""
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    a = _analyze(inp)
    canonical = out.T if a['vertical'] else out
    remaining = [s['span'] for s in a['slots']]
    rebuilt = _base_output(a)
    previous = 0
    for s in a['slots']:
        x, w, color = s['x'], s['width'], s['color']
        rr = np.flatnonzero(np.any(canonical[:, x:x + w] == color, axis=1))
        assert len(rr), 'missing destination bar'
        lo, hi = int(rr[0]) - a['rail'], int(rr[-1]) - a['rail']
        length = hi - lo + 1
        assert np.all(canonical[rr[0]:rr[-1] + 1, x:x + w] == color), 'bar must be solid'
        assert length >= previous, 'length order must increase'
        assert (lo, hi) in remaining, 'registered intervals must transfer exactly'
        remaining.remove((lo, hi))
        rebuilt[rr[0]:rr[-1] + 1, x:x + w] = color
        previous = length
    assert not remaining
    assert np.array_equal(rebuilt, canonical), 'reference/rail changed or swatch frames retained'
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen
            seen.add(key)
            analyses[split].append(_check_pair(pair))
    first = analyses['train'][0]
    orbit = {_key(_transform(first['reference'], i)) for i in range(8)}
    roles = tuple(first[k] for k in ('background', 'carrier', 'reference_frame', 'reference_ink'))
    for a in analyses['train'] + analyses['test']:
        assert _key(a['reference']) in orbit, 'one shared stencil per puzzle'
        assert tuple(a[k] for k in ('background', 'carrier', 'reference_frame', 'reference_ink')) == roles
    train = [_features(a) for a in analyses['train']]
    assert len({f['count'] for f in train}) >= 2
    assert sum(f['changed'] for f in train) >= 2
    assert any(f['changed'] and f['nonreversal'] and f['area_inversion'] and f['height_diff'] for f in train)
    assert any(f['offset_moved'] for f in train)
    assert {a['vertical'] for a in analyses['test']} <= {a['vertical'] for a in analyses['train']}
    assert any(any(s['center_span'] != s['span'] for s in a['slots']) for a in analyses['train'])
    assert any(not np.array_equal(_paint(a, [s['span'] for s in a['slots']]), pair['output'])
               for a, pair in zip(analyses['train'], puzzle['train'])), 'B must change an answer'
    assert any(not np.array_equal(_paint(a, sorted(
                   [(-s['anchor'], s['mask'].shape[0] - s['anchor'] - 1) for s in a['slots']],
                   key=lambda z: z[1] - z[0])), pair['output'])
               for a, pair in zip(analyses['train'], puzzle['train'])), 'A must change an answer'
    def fractions(a):
        return [(min(s['span'][1], s['mask'].shape[0] - s['anchor'] - 1)
                 - max(s['span'][0], -s['anchor']) + 1) / s['length'] for s in a['slots']]
    assert sum(all(x >= .65 for x in fractions(a)) for a in analyses['train']) >= 2
    assert all(all(x <= .60 for x in fractions(a)) for a in analyses['test'])


def self_test():
    corpus = [generate_puzzle(seed) for seed in range(42)]
    signature = lambda p: tuple((_key(q['input']), _key(q['output']))
                               for s in ('train', 'test') for q in p[s])
    assert signature(corpus[0]) == signature(generate_puzzle(0))
    assert len({signature(p) for p in corpus}) == 42
    features, references, role_colors = [], set(), {k: set() for k in ('background', 'carrier', 'reference_frame', 'reference_ink')}
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _check_pair(pair)
                features.append(_features(a))
                for k in role_colors:
                    role_colors[k].add(a[k])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                if split == 'train':
                    references.add(min(_key(_transform(a['reference'], i)) for i in range(8)))
    assert len(references) == 42
    assert len({f['shape'] for f in features}) >= 25
    assert {f['vertical'] for f in features} == {False, True}
    assert {w for f in features for w in f['widths']} >= {1, 2, 3, 4}
    assert {i for f in features for i in f['orientations']} == set(range(8))
    assert {f['count'] for f in features} == {2, 3, 4}
    assert any(f['tied'] for f in features)
    assert all(f['hidden'] for f in features)
    assert all(len(colors) >= 8 for colors in role_colors.values())
    # Ablation/counter-rule mutations: omit matching, sorting, interval offset,
    # use area, reverse, or sort by the visible crop height. Every one is rejected.
    rejected = set()
    for puzzle in corpus:
        for pair in puzzle['train']:
            a = _analyze(pair['input'])
            slots = a['slots']
            correct = [s['span'] for s in sorted(slots, key=lambda s: s['length'])]
            variants = {
                'no_A': sorted([(-s['anchor'], s['mask'].shape[0] - s['anchor'] - 1) for s in slots], key=lambda z: z[1] - z[0]),
                'no_B': [s['span'] for s in slots],
                'area': [s['span'] for s in sorted(slots, key=lambda s: s['length'] * s['width'])],
                'visible_height': [s['span'] for s in sorted(slots, key=lambda s: s['mask'].shape[0])],
                'ink_count': [s['span'] for s in sorted(slots, key=lambda s: int(s['mask'].sum()))],
                'width': [s['span'] for s in sorted(slots, key=lambda s: s['width'])],
                'color': [s['span'] for s in sorted(slots, key=lambda s: s['color'])],
                'middle_column': sorted([s['center_span'] or (0, 0) for s in slots], key=lambda z: z[1] - z[0]),
                'reverse': [s['span'] for s in slots[::-1]],
                'recenter': [(-((hi - lo + 1) // 2), (hi - lo) // 2) for lo, hi in correct],
            }
            for name, spans in variants.items():
                if name in rejected or spans == correct:
                    continue
                if any(a['rail'] + lo < 0 or a['rail'] + hi >= a['grid'].shape[0] for lo, hi in spans):
                    continue
                bad = {'input': pair['input'], 'output': _paint(a, spans)}
                try:
                    _check_pair(bad)
                except (AssertionError, ValueError):
                    rejected.add(name)
                else:
                    raise AssertionError(f'accepted {name} mutation')
    assert rejected == {'no_A', 'no_B', 'area', 'visible_height', 'ink_count', 'width', 'color', 'middle_column', 'reverse', 'recenter'}
    ambiguous = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
    try:
        _match(ambiguous, ambiguous[:2, :2], 0)
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting registered intervals were accepted')
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in corpus[0][s]] for s in ('train', 'test')}
    out = bad['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('an unexplained output pixel was accepted')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1200 + 10 * nt + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1)]:
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for nt, nq in [(None, None), (2, None), (None, 4), (10, None)]:
        validate_generated_puzzle(generate_puzzle(500, nt, nq))


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, relational checks and counter-rule mutations)')
