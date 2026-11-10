"""Sealed-pocket weaving: noisy seams determine two-scale, clipped weaves.

Only NumPy and the standard library are used. No parent code, files, generation
metadata, or training labels are consulted by solve().
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for name, v, lo, hi in (("num_train", num_train, 2, 10),
                            ("num_test", num_test, 1, 4)):
        if (isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer)) or not lo <= v <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)


def _strong(mask):
    result = np.zeros_like(mask, dtype=bool)
    for axis in (0, 1):
        a = mask if axis == 0 else mask.T
        b = result if axis == 0 else result.T
        triples = a[:, :-2] & a[:, 1:-1] & a[:, 2:]
        for k in range(3):
            b[:, k:k + triples.shape[1]] |= triples
    return result


def _thick(mask):
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1]
                       & mask[:-1, 1:] & mask[1:, 1:]))


def _inside(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _at(mask, r, c):
    return _inside(mask.shape, r, c) and bool(mask[r, c])


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask if dc else mask.T
        matches = (a[:, :-4] & a[:, 4:]
                   & ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1]))
        for y, x in np.argwhere(matches):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _repair(mask):
    """Recognize cues simultaneously; then remove diagonal-only fragments."""
    strong = _strong(mask)
    evidence, additions, anchors = [], np.zeros_like(mask), np.zeros_like(mask)
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5*dr, c + 5*dc)):
            continue
        matches = []
        for end in (0, 4):
            ar, ac = r + end*dr, c + end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                vr, vc = toward*dr + side*dc, toward*dc + side*dr
                cells = [(ar + k*vr, ac + k*vc) for k in (1, 2)]
                if (all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in cells)
                        and not _at(mask, ar + 3*vr, ac + 3*vc)):
                    matches.append((end, side, tuple(cells)))
        if matches:
            if len(matches) != 1:
                raise ValueError("ambiguous seam-repair cue")
            evidence.append(((r, c, dr, dc), matches[0]))
            for k in (1, 2, 3):
                additions[r + k*dr, c + k*dc] = True
            anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
    supported = _strong(mask | additions) | anchors
    weak = mask & ~supported
    removed = np.zeros_like(mask)
    for dc in (-1, 1):
        left = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        right = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        linked = left & right
        if dc == 1:
            removed[:-1, :-1] |= linked
            removed[1:, 1:] |= linked
        else:
            removed[:-1, 1:] |= linked
            removed[1:, :-1] |= linked
    return (mask & ~removed) | additions, evidence, removed


def _regions(walls):
    """All four-connected free components, with explicit boundary reachability."""
    h, w = walls.shape
    labels = np.full((h, w), -1, dtype=int)
    records = []
    for r in range(h):
        for c in range(w):
            if walls[r, c] or labels[r, c] >= 0:
                continue
            number = len(records)
            labels[r, c] = number
            todo, cells, exterior = deque([(r, c)]), [], False
            while todo:
                rr, cc = todo.popleft()
                cells.append((rr, cc))
                exterior |= rr in (0, h - 1) or cc in (0, w - 1)
                for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if (_inside(walls.shape, nr, nc) and not walls[nr, nc]
                            and labels[nr, nc] < 0):
                        labels[nr, nc] = number
                        todo.append((nr, nc))
            a = np.asarray(cells, dtype=int)
            lo, hi = a.min(axis=0), a.max(axis=0)
            records.append(dict(cells=a, exterior=bool(exterior),
                                origin=tuple(map(int, lo)),
                                shape=tuple(map(int, hi-lo+1))))
    return labels, records


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not np.all(a[0] == bg):
        raise ValueError("missing uniform background row")
    dividers = [r for r in range(1, a.shape[0])
                if int(a[r, 0]) != bg and np.all(a[r] == a[r, 0])]
    if len(dividers) != 1:
        raise ValueError("expected one full-width seam divider")
    d = dividers[0]
    seam, p = int(a[d, 0]), d - 4
    if not 2 <= p <= 4 or d + 1 >= a.shape[0]:
        raise ValueError("invalid swatch height or missing scene")
    positions = np.flatnonzero(a[1] == seam)
    runs = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    if len(runs) != 2 or len(runs[0]) != len(runs[1]):
        raise ValueError("expected equal swatch frames")
    q = len(runs[0]) - 2
    if not 2 <= q <= 4:
        raise ValueError("invalid swatch width")
    if (int(runs[0][0]), int(runs[1][0])) != (1, q + 4):
        raise ValueError("invalid swatch positions")
    carrier = a[2:d-2, 2:q+2].copy()
    stencil = a[2:d-2, q+5:2*q+5].copy()
    expected = np.full((d+1, a.shape[1]), bg, dtype=np.int8)
    for start, interior in ((1, carrier), (q+4, stencil)):
        expected[1:p+3, start:start+q+2] = seam
        expected[2:p+2, start+1:start+q+1] = interior
    expected[d] = seam
    if not np.array_equal(expected, a[:d+1]):
        raise ValueError("malformed swatch tray")
    if np.any((carrier == bg) | (carrier == seam)) or np.any(stencil == seam):
        raise ValueError("reserved color used as payload")
    scene = a[d+1:].copy()
    if not np.all((scene == bg) | (scene == seam)):
        raise ValueError("scene must use only background and seam colors")
    return bg, seam, carrier, stencil, scene


def _render(walls, carrier, stencil, bg, seam, variant="correct", *, regions=None):
    out = np.where(walls, seam, bg).astype(np.int8)
    records = _regions(walls)[1] if regions is None else regions
    p, q = carrier.shape
    for record in records:
        if record['exterior']:
            continue
        cells = record['cells']
        r0, c0 = record['origin']
        h, w = record['shape']
        if variant == 'rectangle':
            rr, cc = np.indices((h, w))
            rr, cc = (rr+r0).ravel(), (cc+c0).ravel()
            eligible = ~walls[rr, cc]
            cells = np.column_stack((rr[eligible], cc[eligible]))
        rr, cc = cells[:, 0], cells[:, 1]
        y, x = rr-r0, cc-c0
        sr, sc = y*p//h, x*q//w
        if variant == 'center_resize':
            sr, sc = (2*y+1)*p//(2*h), (2*x+1)*q//(2*w)
        base = carrier[sr, sc]
        if variant == 'tile_carrier':
            base = carrier[y % p, x % q]
        ty, tx = y, x
        if variant == 'global_phase':
            ty, tx = rr, cc
        elif variant == 'first_cell_phase':
            first = cells[np.lexsort((cc, rr))[0]]
            ty, tx = rr-int(first[0]), cc-int(first[1])
        elif variant == 'phase_rows':
            ty = y+1
        elif variant == 'phase_columns':
            tx = x+1
        overlay = stencil[ty % p, tx % q]
        if variant in ('scale_stencil', 'fuse_then_scale'):
            overlay = stencil[sr, sc]
        if variant == 'ignore_stencil':
            values = base
        elif variant == 'constant_fill':
            values = np.full(len(cells), carrier[0, 0], dtype=np.int8)
        elif variant == 'opaque_background':
            values = overlay
        else:
            values = np.where(overlay == bg, base, overlay)
        out[rr, cc] = values
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell solely from visible input structure."""
    bg, seam, carrier, stencil, scene = _decode(input_grid)
    walls, _, _ = _repair(scene == seam)
    return _render(walls, carrier, stencil, bg, seam)


def _tray(scene, carrier, stencil, bg, seam):
    p, q = carrier.shape
    h, w = scene.shape
    if w < 2*q+7 or h+p+5 > 30:
        raise ValueError("swatches and scene do not fit")
    a = np.full((p+5+h, w), bg, dtype=np.int8)
    for c, interior in ((1, carrier), (q+4, stencil)):
        a[1:p+3, c:c+q+2] = seam
        a[2:p+2, c+1:c+q+1] = interior
    a[p+4] = seam
    a[p+5:] = scene
    return a


def _draw_polygon(shape, vertices):
    mask = np.zeros(shape, dtype=bool)
    for (r, c), (rr, cc) in zip(vertices, vertices[1:]+vertices[:1]):
        if r == rr:
            mask[r, min(c, cc):max(c, cc)+1] = True
        elif c == cc:
            mask[min(r, rr):max(r, rr)+1, c] = True
        else:
            raise ValueError("nonorthogonal polygon")
    return mask


def _outline(rng, h, w, kind, origin_witness=False):
    transpose = bool(rng.integers(2))
    if transpose:
        h, w = w, h
    r0, c0 = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    r1, c1 = h-1-int(rng.integers(0, 3)), w-1-int(rng.integers(0, 3))
    # Leave some exterior margin on each axis: the tray divider is the only
    # uniform non-background row in the complete input.
    if r0 == 0 and r1 == h-1:
        r1 -= 1
    if c0 == 0 and c1 == w-1:
        c1 -= 1
    if kind == 'corner':
        rh = int(rng.integers(3, max(4, (r1-r0)//2)))
        cw = int(rng.integers(3, max(4, (c1-c0)//2)))
        vertices = [(r0, c0+cw), (r0, c1), (r1, c1), (r1, c0),
                    (r0+rh, c0), (r0+rh, c0+cw)]
    elif kind == 'notch':
        left = c0+int(rng.integers(3, max(4, (c1-c0)//2)))
        right = int(rng.integers(left+3, c1-2))
        depth = int(rng.integers(3, max(4, (r1-r0)//2)))
        vertices = [(r0, c0), (r0, left), (r0+depth, left),
                    (r0+depth, right), (r0, right), (r0, c1), (r1, c1), (r1, c0)]
    else:
        vertices = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    walls = _draw_polygon((h, w), vertices)
    # Reflections and half turns keep arbitrary rectangular scene dimensions.
    if not origin_witness and rng.integers(2):
        walls = walls[::-1]
    if not origin_witness and rng.integers(2):
        walls = walls[:, ::-1]
    if transpose:
        walls = walls.T
    return walls.copy()


def _sealed(walls):
    return [x for x in _regions(walls)[1] if not x['exterior']]


def _absent_origin(record):
    return not np.any(np.all(record['cells'] == np.asarray(record['origin']), axis=1))


def _subdivide(rng, walls, target, origin_witness=False):
    while len(_sealed(walls)) < target:
        records = _sealed(walls)
        choices = []
        for index, a in enumerate(records):
            r, c = a['origin']
            h, w = a['shape']
            choices += [(index, 0, cut) for cut in range(r+3, r+h-3)]
            choices += [(index, 1, cut) for cut in range(c+3, c+w-3)]
        rng.shuffle(choices)
        accepted = False
        for index, axis, cut in choices[:40]:
            cells = records[index]['cells']
            chosen = cells[cells[:, axis] == cut]
            if len(chosen) < 3:
                continue
            trial = walls.copy()
            trial[chosen[:, 0], chosen[:, 1]] = True
            parts = _sealed(trial)
            if (len(parts) == len(records)+1
                    and not _thick(trial)
                    and all(min(a['shape']) >= 3 and len(a['cells']) >= 12 for a in parts)
                    and (not origin_witness or any(_absent_origin(a) for a in parts))
                    and np.array_equal(_repair(trial)[0], trial)):
                walls = trial
                accepted = True
                break
        if not accepted:
            raise ValueError("could not subdivide outline")
    return walls


def _segments(walls):
    """Legal five-cell brackets in a straight seam, away from junctions."""
    candidates = []
    labels, records = _regions(walls)
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(walls):
            r, c = int(r), int(c)
            if not all(_at(walls, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(walls, r-dr, c-dc) or _at(walls, r+5*dr, c+5*dc)):
                continue
            gap = [(r+k*dr, c+k*dc) for k in (1, 2, 3)]
            if any(_at(walls, rr+dc, cc+dr) or _at(walls, rr-dc, cc-dr)
                   for rr, cc in gap):
                continue
            mr, mc = r+2*dr, c+2*dc
            neighbors = [(mr+dc, mc+dr), (mr-dc, mc-dr)]
            sides = [labels[rr, cc] if _inside(walls.shape, rr, cc) else -1
                     for rr, cc in neighbors]
            closed = [i >= 0 and not records[i]['exterior'] for i in sides]
            kind = 'internal' if all(closed) and sides[0] != sides[1] else 'outer'
            if not any(closed):
                continue
            candidates.append(((r, c, dr, dc), gap, kind))
    return candidates


def _add_vent(rng, walls):
    candidates = [x for x in _segments(walls) if x[2] == 'outer']
    rng.shuffle(candidates)
    for _, gap, _ in candidates:
        trial = walls.copy()
        for cell in gap:
            trial[cell] = False
        if _sealed(trial) and np.array_equal(_repair(trial)[0], trial):
            return trial
    raise ValueError("no legal vent")


def _halo(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[r:r+h, c:c+w] for r in range(3) for c in range(3)])


def _corrupt(rng, walls, maximum):
    raw, occupied = walls.copy(), np.zeros_like(walls)
    options = []
    for (r, c, dr, dc), gap, kind in _segments(walls):
        for end in (0, 4):
            ar, ac = r+end*dr, c+end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                vr, vc = toward*dr+side*dc, toward*dc+side*dr
                spur = [(ar+k*vr, ac+k*vc) for k in (1, 2)]
                if all(_inside(walls.shape, rr, cc) and not walls[rr, cc] for rr, cc in spur):
                    options.append((gap, spur, kind))
    rng.shuffle(options)
    count = 0
    for gap, spur, _ in options:
        footprint = np.zeros_like(walls)
        for cell in gap+spur:
            footprint[cell] = True
        if np.any(_halo(occupied) & footprint):
            continue
        trial = raw.copy()
        for cell in gap:
            trial[cell] = False
        for cell in spur:
            trial[cell] = True
        try:
            predicted, cues, _ = _repair(trial)
        except ValueError:
            continue
        if np.array_equal(predicted, walls) and len(cues) == count+1:
            raw = trial
            occupied |= footprint
            count += 1
            if count >= maximum:
                break
    if not count:
        raise ValueError("no reconstructible damage")
    # Unattached clutter varies independently of gap count and direction.
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(20):
            r, c = int(rng.integers(walls.shape[0])), int(rng.integers(walls.shape[1]))
            dc, length = int(rng.choice([-1, 1])), int(rng.choice([2, 3]))
            cells = [(r+k, c+dc*k) for k in range(length)]
            if not all(_inside(walls.shape, rr, cc) and not _halo(raw)[rr, cc]
                       for rr, cc in cells):
                continue
            trial = raw.copy()
            for cell in cells:
                trial[cell] = True
            try:
                predicted, cues, _ = _repair(trial)
            except ValueError:
                continue
            if np.array_equal(predicted, walls) and len(cues) == count:
                raw = trial
                break
    return raw


def _swatches(rng, p, q, payload, bg, rich):
    rr, cc = np.indices((p, q))
    k = int(rng.integers(2, min(len(payload), 4)+1))
    inks = rng.choice(payload, k, replace=False)
    style = int(rng.integers(6 if rich else 4))
    if style == 0:
        labels = (rr+cc) % k
    elif style == 1:
        labels = (rr//int(rng.integers(1, p)) + cc//int(rng.integers(1, q))) % k
    elif style == 2:
        labels = np.tile(rng.integers(k, size=q), (p, 1))
    elif style == 3:
        labels = np.tile(rng.integers(k, size=p)[:, None], (1, q))
    elif style == 4:
        labels = rng.integers(k, size=(p, q))
    else:
        labels = np.zeros((p, q), dtype=int)
        labels[int(rng.integers(p)): , int(rng.integers(q)):] = 1
    carrier = inks[labels].astype(np.int8)
    stencil = np.full((p, q), bg, dtype=np.int8)
    style = int(rng.integers(7))
    if style == 0:
        mask = rng.random((p, q)) < rng.uniform(.15, .75)
    elif style == 1:
        mask = (rr+cc+int(rng.integers(2))) % 2 == 0
    elif style == 2:
        mask = rr == int(rng.integers(p))
    elif style == 3:
        mask = cc == int(rng.integers(q))
    elif style == 4:
        mask = (rr == 0) | (rr == p-1) | (cc == 0) | (cc == q-1)
    elif style == 5:
        mask = abs(rr-int(rng.integers(p)))+abs(cc-int(rng.integers(q))) <= 1
    else:
        mask = rng.random((p, q)) < .8
    if rng.random() < .08:
        mask[:] = False
    elif mask.all():
        mask.ravel()[int(rng.integers(p*q))] = False
    elif not mask.any():
        mask.ravel()[int(rng.integers(p*q))] = True
    stencil[mask] = rng.choice(payload, int(mask.sum())) if rng.random() < .5 else int(rng.choice(payload))
    return carrier, stencil


def _sample_pair(rng, params, target, phase_witness=False):
    bg, seam = params['roles']
    for _ in range(36):
        p, q = map(int, rng.choice(params['sizes'], 2))
        h = int(rng.integers(16 if target <= 3 else 19, 26-p))
        w = int(rng.integers(19 if target <= 3 else 23, 31))
        kind = ('corner' if phase_witness else
                str(rng.choice(['rectangle', 'corner', 'notch'], p=[.22, .48, .30])))
        try:
            walls = _subdivide(rng, _outline(rng, h, w, kind, phase_witness),
                               target, phase_witness)
            if not phase_witness and rng.random() < .45:
                walls = _add_vent(rng, walls)
            raw = _corrupt(rng, walls, int(rng.integers(1, 4)))
        except ValueError:
            continue
        carrier, stencil = _swatches(rng, p, q, params['payload'], bg, params['rich'])
        scene = np.where(raw, seam, bg).astype(np.int8)
        pair = {'input': _tray(scene, carrier, stencil, bg, seam),
                'output': _render(walls, carrier, stencil, bg, seam)}
        if phase_witness and np.array_equal(
                pair['output'], _render(walls, carrier, stencil, bg, seam, 'first_cell_phase')):
            continue
        if np.array_equal(solve(pair['input']), pair['output']):
            return pair
    raise RuntimeError("bounded scene construction exhausted")


_REQUIRED = frozenset({
    'ignore_repair', 'delete_only', 'keep_diagonals', 'fill_all_gaps',
    'tile_carrier', 'scale_stencil', 'fuse_then_scale', 'opaque_background',
    'ignore_stencil', 'constant_fill', 'phase_rows', 'phase_columns',
    'global_phase', 'first_cell_phase', 'rectangle', 'center_resize',
    'visible_carrier', 'visible_overwrite', 'outer_repair', 'internal_repair',
    'outer_activation', 'internal_rescale', 'vent', 'irregular', 'missing_origin',
})


def _evidence(pair):
    bg, seam, carrier, stencil, scene = _decode(pair['input'])
    raw = scene == seam
    walls, cues, removed = _repair(raw)
    labels, records = _regions(walls)
    out = _grid(pair['output'])
    cover = set()
    def compare(name, prediction):
        if not np.array_equal(prediction, out):
            cover.add(name)
    for name, alternative in (('ignore_repair', raw), ('delete_only', raw & ~removed),
                              ('keep_diagonals', walls | raw)):
        compare(name, _render(alternative, carrier, stencil, bg, seam))
    filled = walls.copy()
    for r, c, dr, dc in _gaps(walls):
        for k in (1, 2, 3):
            filled[r+k*dr, c+k*dc] = True
    compare('fill_all_gaps', _render(filled, carrier, stencil, bg, seam))
    for name in _REQUIRED - {'ignore_repair', 'delete_only', 'keep_diagonals',
                             'fill_all_gaps', 'visible_carrier', 'visible_overwrite',
                             'outer_repair', 'internal_repair', 'outer_activation',
                             'internal_rescale', 'vent', 'irregular', 'missing_origin'}:
        compare(name, _render(walls, carrier, stencil, bg, seam, name, regions=records))
    for (r, c, dr, dc), _ in cues:
        mr, mc = r+2*dr, c+2*dc
        sides = [(mr+dc, mc+dr), (mr-dc, mc-dr)]
        ids = [labels[rr, cc] if _inside(walls.shape, rr, cc) else -1 for rr, cc in sides]
        sealed = [i >= 0 and not records[i]['exterior'] for i in ids]
        if all(sealed) and ids[0] != ids[1]:
            cover.add('internal_repair')
            kind = 'internal'
        elif any(sealed):
            cover.add('outer_repair')
            kind = 'outer'
        else:
            continue
        # Reopen just this repair in the otherwise cleaned drawing. A visible
        # seam edit alone does not establish the interaction with the swatches.
        opened = walls.copy()
        for k in (1, 2, 3):
            opened[r+k*dr, c+k*dc] = False
        opened_labels, opened_regions = _regions(opened)
        if kind == 'outer':
            for index in ids:
                if index < 0 or records[index]['exterior']:
                    continue
                cells = records[index]['cells']
                rr, cc = cells[:, 0], cells[:, 1]
                if any(opened_regions[i]['exterior'] for i in np.unique(opened_labels[rr, cc])):
                    cover.add('outer_activation')
        elif 'internal_rescale' not in cover:
            p, q = carrier.shape
            for index in ids:
                record = records[index]
                cells = record['cells']
                rr, cc = cells[:, 0], cells[:, 1]
                merged = opened_regions[int(opened_labels[rr[0], cc[0]])]
                if merged['exterior']:
                    continue
                y, x = rr-record['origin'][0], cc-record['origin'][1]
                my, mx = rr-merged['origin'][0], cc-merged['origin'][1]
                h, w = record['shape']
                mh, mw = merged['shape']
                before = carrier[y*p//h, x*q//w]
                after = carrier[my*p//mh, mx*q//mw]
                transparent = ((stencil[y % p, x % q] == bg)
                               & (stencil[my % p, mx % q] == bg))
                if np.any(transparent & (before != after)):
                    cover.add('internal_rescale')
                    break
    # A genuine vent exposes free space to the exterior. Incidental aligned
    # brackets between two already sealed chambers do not establish leakage.
    exterior = np.zeros_like(walls)
    for record in records:
        if record['exterior']:
            cells = record['cells']
            exterior[cells[:, 0], cells[:, 1]] = True
    for r, c, dr, dc in _gaps(walls):
        trial = walls.copy()
        for k in (1, 2, 3):
            trial[r+k*dr, c+k*dc] = True
        if any(np.any(exterior[a['cells'][:, 0], a['cells'][:, 1]]) for a in _sealed(trial)):
            cover.add('vent')
            break
    p, q = carrier.shape
    for a in records:
        if a['exterior']:
            continue
        r0, c0 = a['origin']
        h, w = a['shape']
        cells = a['cells']
        if len(cells) < h*w:
            cover.add('irregular')
        if labels[r0, c0] != labels[cells[0, 0], cells[0, 1]]:
            cover.add('missing_origin')
        y, x = cells[:, 0]-r0, cells[:, 1]-c0
        base = carrier[y*p//h, x*q//w]
        tiled = stencil[y % p, x % q]
        if np.any(tiled == bg):
            cover.add('visible_carrier')
        if np.any((tiled != bg) & (tiled != base)):
            cover.add('visible_overwrite')
    return cover


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    available = [c for c in range(10) if c not in roles]
    params = dict(roles=roles, payload=rng.choice(available, int(rng.integers(3, 7)), replace=False),
                  sizes=[2, 3] if rng.random() < .3 else [2, 3, 4],
                  rich=bool(rng.integers(2)), ceiling=int(rng.choice([2, 3])))
    candidates, covers, seen = [], [], set()
    teaching = None
    for candidate_index in range(72):
        target = int(rng.integers(2, params['ceiling']+1))
        # One candidate stratum explicitly teaches an origin outside its pocket;
        # this avoids relying on a rare conjunction of orientation and period.
        pair = _sample_pair(rng, params, target, phase_witness=candidate_index % 4 == 3)
        if _key(pair['input']) in seen:
            continue
        cover = _evidence(pair)
        for index in rng.permutation(len(candidates)):
            if _REQUIRED <= cover | covers[int(index)]:
                teaching = [candidates[int(index)], pair]
                break
        candidates.append(pair)
        covers.append(cover)
        seen.add(_key(pair['input']))
        if teaching is not None:
            break
    if teaching is None:
        best = min((_REQUIRED-(a | b) for a in covers for b in covers), key=len)
        raise RuntimeError("bounded teaching pool missed evidence: " + str(sorted(best)))
    puzzle = {'train': teaching, 'test': []}
    seen = {_key(x['input']) for x in teaching}
    for split, count in (('train', n), ('test', t)):
        while len(puzzle[split]) < count:
            target = (params['ceiling']+1 if split == 'test'
                      else int(rng.integers(2, params['ceiling']+1)))
            for _ in range(20):
                pair = _sample_pair(rng, params, target)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _diagonal_groups(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while cells:
        start = min(cells)
        cells.remove(start)
        todo, group = [start], []
        while todo:
            r, c = todo.pop()
            group.append((r, c))
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nxt = r+dr, c+dc
                if nxt in cells:
                    cells.remove(nxt)
                    todo.append(nxt)
        groups.append(sorted(group))
    return groups


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, roles, coverage = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            bg, seam, carrier, stencil, scene = _decode(inp)
            roles.add((bg, seam))
            walls, cues, removed = _repair(scene == seam)
            assert cues and removed.any(), "repair and deletion must be visible"
            assert out.shape == scene.shape
            assert np.array_equal(solve(inp), out), "incorrect or incomplete output"
            assert np.array_equal(out == seam, walls), "payload overwrote a seam"
            assert not _thick(walls), "seams merged into a solid 2x2 block"
            assert _sealed(walls), "no sealed pocket for the weave"
            for group in _diagonal_groups(removed):
                assert len(group) in (2, 3)
                assert all(group[i+1][0]-group[i][0] == 1 for i in range(len(group)-1))
                steps = {group[i+1][1]-group[i][1] for i in range(len(group)-1)}
                assert len(steps) == 1 and abs(next(iter(steps))) == 1
            labels, regions = _regions(walls)
            for record in regions:
                cells = record['cells']
                values = out[cells[:, 0], cells[:, 1]]
                if record['exterior']:
                    assert np.all(values == bg), "leaking component was woven"
                else:
                    assert np.all(values != bg), "sealed pocket was left unfilled"
            if split == 'train':
                coverage |= _evidence(pair)
    assert len(roles) == 1, "palette conventions changed within the puzzle"
    assert _REQUIRED <= coverage, "training misses evidence: " + str(_REQUIRED-coverage)


def self_test():
    """Bounded 42-seed sweep, concrete rule probes, counts and negative labels."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_key(x['input']) for s in p for x in p[s]) for p in puzzles}) == 42
    shapes, palettes, carriers, stencils, geometries = set(), set(), set(), set(), set()
    axes, endpoints, bends, pockets, repair_counts = set(), set(), set(), set(), set()
    per_split_counts = {'train': set(), 'test': set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                bg, seam, carrier, stencil, scene = _decode(pair['input'])
                walls, cues, _ = _repair(scene == seam)
                shapes.add(scene.shape)
                palettes.add((bg, seam))
                carriers.add(_normalized(carrier))
                stencils.add(_normalized(stencil))
                geometries.add(_key(walls))
                sealed = _sealed(walls)
                pockets.add(len(sealed))
                per_split_counts[split].add(len(sealed))
                repair_counts.add(len(cues))
                for (r, c, dr, dc), (end, side, _) in cues:
                    axes.add('h' if dc else 'v')
                    endpoints.add(end)
                    bends.add(side)
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
    assert len(shapes) >= 35 and len(palettes) >= 25
    assert len(carriers) >= 60 and len(stencils) >= 60 and len(geometries) >= 180
    assert axes == {'h', 'v'} and endpoints == {0, 4} and bends == {-1, 1}
    assert {1, 2, 3} <= repair_counts and {1, 2, 3, 4} <= pockets
    assert max(per_split_counts['test']) > max(per_split_counts['train'])
    replay = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in replay
               for a, b in zip(replay[s], puzzles[0][s]) for k in a)
    # Exact small semantic probes independent of the random scene sampler.
    wall = np.zeros((10, 12), dtype=bool)
    wall[1, 1:11] = wall[8, 1:11] = True
    wall[1:9, 1] = wall[1:9, 10] = True
    raw = wall.copy()
    raw[1, 4:7] = False
    raw[2, 4] = raw[3, 5] = True
    repaired, _, _ = _repair(raw)
    assert np.array_equal(repaired, wall)
    vent = wall.copy()
    vent[1, 4:7] = False
    assert np.array_equal(_repair(vent)[0], vent) and not _sealed(vent)
    carrier = np.array([[2, 3], [4, 2]], dtype=np.int8)
    stencil = np.array([[0, 6], [0, 0]], dtype=np.int8)
    out = _render(wall, carrier, stencil, 0, 8)
    assert (int(out[2, 2]), int(out[2, 3]), int(out[5, 2]), int(out[7, 9])) == (2, 6, 4, 2)
    # Hand-specified internal repair: its non-seam consequence is a change of
    # carrier color at (2,6), transparent in both the split and merged pockets.
    split = np.zeros((12, 16), dtype=bool)
    split[1, 1:15] = split[10, 1:15] = True
    split[1:11, 1] = split[1:11, 7] = split[1:11, 14] = True
    damaged = split.copy()
    damaged[4:7, 7] = False
    damaged[4, 8] = damaged[5, 9] = True
    interaction_input = _tray(np.where(damaged, 8, 0), carrier, stencil, 0, 8)
    interaction_output = solve(interaction_input)
    assert int(interaction_output[2, 6]) == 3
    opened = split.copy()
    opened[4:7, 7] = False
    assert int(_render(opened, carrier, stencil, 0, 8)[2, 6]) == 2
    assert 'internal_rescale' in _evidence(dict(input=interaction_input, output=interaction_output))
    # All legal explicit counts, including the two-demonstration boundary.
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            task = generate_puzzle(100+n*10+t, n, t)
            assert len(task['train']) == n and len(task['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # Reject an omitted repair together with its undeleted diagonal, even when
    # all the other repairs in this multi-repair label are correct.
    partial = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    pair = next(pair for pair in partial['train']
                if len(_repair(_decode(pair['input'])[4] == _decode(pair['input'])[1])[1]) >= 2)
    bg, seam, _, _, scene = _decode(pair['input'])
    (r, c, dr, dc), (_, _, spur) = _repair(scene == seam)[1][0]
    for k in (1, 2, 3):
        pair['output'][r+k*dr, c+k*dc] = bg
    for rr, cc in spur:
        pair['output'][rr, cc] = seam
    try:
        validate_generated_puzzle(partial)
    except AssertionError:
        pass
    else:
        raise AssertionError("omitted repair accepted")
    return dict(seeds=42, scene_shapes=len(shapes), role_palettes=len(palettes),
                carrier_structures=len(carriers), stencil_structures=len(stencils),
                seam_geometries=len(geometries), repairs=sorted(repair_counts),
                sealed_pocket_counts=sorted(pockets))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
