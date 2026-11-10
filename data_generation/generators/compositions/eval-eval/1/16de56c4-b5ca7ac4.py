"""Periodic docking of intact framed cargo under sparse, input-visible rulers.

No parent code is imported. Generation paints sampled geometry; solve decodes
only its argument. Validation also runs a separate one-berth-at-a-time model.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _band(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b):
    return (_band(a, b)
            and max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']))


def _rail(values, bg):
    positions = np.flatnonzero(values != bg).tolist()
    if len(positions) < 2 or values[0] != bg or values[-1] != bg:
        raise ValueError("rulers have blank corners and a repeated pair")
    counts = {}
    for x in positions:
        color = int(values[x])
        counts[color] = counts.get(color, 0) + 1
    repeated = [color for color, count in counts.items() if count > 1]
    if len(repeated) != 1 or counts[repeated[0]] != 2:
        raise ValueError("a ruler needs exactly one equal-colored pair")
    ref = repeated[0]
    a, b = [x for x in positions if values[x] == ref]
    p = b - a
    aligned = [x for x in positions if x not in (a, b) and (x - a) % p == 0]
    if len(aligned) > 1:
        raise ValueError("more than one on-phase terminal")
    t = aligned[0] if aligned else None
    if t is not None and a < t < b:
        raise ValueError("terminal must lie beyond the pair")
    lo, hi = (0, len(values) - 1) if t is None else (min(a, t), max(b, t))
    return dict(a=a, b=b, p=p, t=t, lo=lo, hi=hi, ref=ref,
                paint=ref if t is None else int(values[t]),
                noise=tuple(x for x in positions if x not in (a, b, t)),
                docks=tuple(x for x in range(lo, hi + 1) if (x - a) % p == 0))


def _decode_canonical(g, bg):
    """Horizontal representation: rulers in first/last row, blank spacers."""
    h, w = g.shape
    if h < 7 or w < 5 or np.any(g[1] != bg) or np.any(g[-2] != bg):
        raise ValueError("rulers require a blank spacer and a cargo region")
    guides = [_rail(g[0], bg), _rail(g[-1], bg)]
    if guides[0]['ref'] == guides[1]['ref']:
        raise ValueError("the two cargo classes need distinct ruler colors")
    ref_to_class = {s['ref']: i for i, s in enumerate(guides)}
    covered = g == bg
    covered[:2] = True
    covered[-2:] = True
    objects = []
    for r in range(2, h - 2):
        for c in range(w):
            if covered[r, c]:
                continue
            frame = int(g[r, c])
            if frame not in ref_to_class or r + 2 >= h - 2 or c + 2 >= w:
                raise ValueError("unrecognized or incomplete cargo frame")
            if g[r + 1, c + 1] in (frame, bg):
                raise ValueError("interior must be distinguishable from its frame")
            bottom, right = r + 2, c + 2
            while bottom < h - 2 and g[bottom, c + 1] != frame:
                bottom += 1
            while right < w and g[r + 1, right] != frame:
                right += 1
            if bottom >= h - 2 or right >= w:
                raise ValueError("open cargo frame")
            patch = g[r:bottom + 1, c:right + 1].copy()
            if not (np.all(patch[0] == frame) and np.all(patch[-1] == frame)
                    and np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
                    and np.all(patch[1:-1, 1:-1] != frame)
                    and np.all(patch[1:-1, 1:-1] != bg)):
                raise ValueError("cargo is not a complete rectangular frame")
            if np.any(covered[r:bottom + 1, c:right + 1]):
                raise ValueError("overlapping cargo")
            covered[r:bottom + 1, c:right + 1] = True
            d = ref_to_class[frame]
            obj = dict(r=r, c=c, h=bottom-r+1, w=right-c+1, d=d, patch=patch)
            lead = c if d == 0 else right
            if lead not in guides[d]['docks']:
                raise ValueError("initial leading edge is not an allowed berth")
            objects.append(obj)
    if not objects or not np.all(covered):
        raise ValueError("cargo region contains unexplained cells")
    return guides, objects


def _decode(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    candidates = []
    for axis in (1, 0):
        g = grid if axis == 1 else grid.T.copy()
        try:
            guides, objects = _decode_canonical(g, bg)
        except ValueError:
            continue
        candidates.append((axis, bg, g, guides, objects))
    if len(candidates) != 1:
        raise ValueError("input does not specify one docking axis")
    return candidates[0]


def _order(objects, d):
    ids = [i for i, o in enumerate(objects) if o['d'] == d]
    return sorted(ids, key=lambda i: (objects[i]['c'] if d == 0
                                     else -objects[i]['c'] - objects[i]['w'] + 1,
                                     objects[i]['r'], i))


def _pack(objects, guides, width, *, stable=True):
    """Leading edges occupy berths; overlapping predecessors cannot be passed."""
    result = [dict(o) for o in objects]
    for d in (0, 1):
        settled = []
        for i in _order(objects, d):
            obj = result[i]
            docks = guides[d]['docks']
            proposals = docks if d == 0 else reversed(docks)
            found = False
            for q in proposals:
                c = q if d == 0 else q - obj['w'] + 1
                if not 0 <= c <= width - obj['w']:
                    continue
                trial = dict(obj, c=c)
                if stable:
                    valid = all(not _band(trial, prev)
                                or (c >= prev['c'] + prev['w'] if d == 0
                                    else c + obj['w'] <= prev['c']) for prev in settled)
                else:
                    valid = not any(_overlap(trial, prev) for prev in settled)
                if valid:
                    obj['c'] = c
                    found = True
                    break
            if not found:
                raise ValueError("cargo cannot fit into the allowed docking span")
            if stable and ((d == 0 and obj['c'] > objects[i]['c'])
                           or (d == 1 and obj['c'] < objects[i]['c'])):
                raise ValueError("packing would reverse the commanded motion")
            settled.append(obj)
    if stable and any(_overlap(a, b) for i, a in enumerate(result) for b in result[i+1:]):
        raise ValueError("opposite final classes overlap")
    return result


def _render(g, bg, objects, guides, *, repaint=True):
    out = np.full_like(g, bg)
    out[0], out[-1] = g[0], g[-1]
    occupied = np.zeros(g.shape, dtype=bool)
    for o in objects:
        r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
        if np.any(occupied[r:r+h, c:c+w]):
            raise ValueError("rendering would hide another cargo object")
        occupied[r:r+h, c:c+w] = True
        patch = o['patch'].copy()
        if repaint:
            paint = guides[o['d']]['paint']
            patch[0] = patch[-1] = paint
            patch[:, 0] = patch[:, -1] = paint
        out[r:r+h, c:c+w] = patch
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    axis, bg, g, guides, objects = _decode(input_grid)
    out = _render(g, bg, _pack(objects, guides, g.shape[1]), guides)
    return out if axis == 1 else out.T.copy()


def _tick_model(g, bg, guides, objects):
    """Validator: repeated single-berth steps with original pairwise order.

    Unlike the solver's ordered allocation, this starts at all source positions
    and repeatedly moves each crate by one period. Opposite classes are ignored
    until the final collision check. A step may never cross a same-class peer.
    """
    current = [dict(o) for o in objects]
    for _ in range(g.shape[1] * len(objects) + 1):
        moved = False
        for i, o in enumerate(current):
            d = o['d']
            c = o['c'] + (-1 if d == 0 else 1) * guides[d]['p']
            lead = c if d == 0 else c + o['w'] - 1
            if not 0 <= c <= g.shape[1] - o['w'] or lead not in guides[d]['docks']:
                continue
            permitted = True
            for j, peer in enumerate(current):
                if i == j or d != peer['d'] or not _band(o, peer):
                    continue
                # Their original left/right relationship remains strict even
                # when a large lattice step could leap over an entire peer.
                if objects[i]['c'] < objects[j]['c']:
                    permitted &= c + o['w'] <= peer['c']
                else:
                    permitted &= c >= peer['c'] + peer['w']
            if permitted:
                o['c'] = c
                moved = True
        if not moved:
            return _render(g, bg, current, guides), current
    raise AssertionError("single-berth model failed to settle")


def _make_guide(rng, width, p, ref, paint, kind='random', noise=False, noise_color=9):
    """Return a row's construction geometry, not a solver-derived label."""
    if kind == 'right_bound':
        a = p + int(rng.integers(1, 3))
        b = a + p
        t = a + ((width - 2 - a) // p) * p
    elif kind == 'left_bound':
        b = width - 2 - int(rng.integers(min(p, 2)))
        a = b - p
        t = b - ((b - 1) // p) * p
    else:
        possibilities = []
        for a in range(1, width - 1 - p):
            b = a + p
            if kind == 'open':
                possibilities.append((a, b, None))
            else:
                for t in range(1, width - 1):
                    if (t < a or t > b) and (t - a) % p == 0:
                        possibilities.append((a, b, t))
        a, b, t = _pick(rng, possibilities)
    if t is not None and t in (a, b):
        raise ValueError("insufficient room for a terminal")
    lo, hi = (0, width - 1) if t is None else (min(a, t), max(b, t))
    noise_positions = [x for x in range(1, width-1) if (x - a) % p]
    marks = {a: ref, b: ref}
    if t is not None:
        marks[t] = paint
    if noise and noise_positions:
        marks[int(_pick(rng, noise_positions))] = noise_color
    return dict(a=a, b=b, p=p, t=t, lo=lo, hi=hi, ref=ref,
                paint=ref if t is None else paint,
                noise=tuple(x for x in marks if x not in (a, b, t)),
                docks=tuple(x for x in range(lo, hi+1) if (x-a) % p == 0), marks=marks)


def _blank(shape, bg, guides):
    g = np.full(shape, bg, dtype=np.int8)
    for row, guide in zip((0, shape[0]-1), guides):
        for x, color in guide['marks'].items():
            g[row, x] = color
    return g


def _appearance(rng, objects, guides, bg, *, shared=False):
    shared_patch = None
    common = [c for c in range(10)
              if c not in (bg, guides[0]['ref'], guides[1]['ref'],
                           guides[0]['paint'], guides[1]['paint'])]
    for i, o in enumerate(objects):
        guide = guides[o['d']]
        allowed = [c for c in range(10) if c not in (bg, guide['ref'], guide['paint'])]
        colors = rng.choice(allowed, int(rng.integers(1, min(4, len(allowed)) + 1)), replace=False)
        rr, cc = np.indices((o['h']-2, o['w']-2))
        mode = int(rng.integers(5))
        ix = np.zeros(rr.shape, dtype=int) if mode < 2 else rr if mode == 2 else cc if mode == 3 else rr+cc
        interior = np.asarray(colors[ix % len(colors)], dtype=np.int8)
        if shared and i in (0, 2):
            if shared_patch is None:
                interior[:] = _pick(rng, common)
                shared_patch = interior.copy()
            interior = shared_patch.copy()
        patch = np.full((o['h'], o['w']), guide['ref'], dtype=np.int8)
        patch[1:-1, 1:-1] = interior
        o['patch'] = patch


def _start_choices(o, guide, width):
    return [q if o['d'] == 0 else q - o['w'] + 1 for q in guide['docks']
            if (0 <= q <= width-o['w'] if o['d'] == 0 else o['w']-1 <= q < width)]


def _palette(rng, fixed):
    return fixed if fixed is not None else [int(c) for c in rng.permutation(10)]


def _make_pair(rng, purpose, axis, fixed, shifted_class=0):
    colors = _palette(rng, fixed)
    bg, ref0, ref1, paint0, paint1, noise0, noise1 = colors[:7]
    refs, paints, noises = [ref0, ref1], [paint0, paint1], [noise0, noise1]
    if purpose == 'calibration':
        width, height = int(rng.integers(27, 31)), int(rng.integers(10, 23))
        ps = [int(rng.integers(2, 4)), int(rng.integers(2, 4))]
        guides = [_make_guide(rng, width, ps[d], refs[d], paints[d],
                              'right_bound' if d == 0 else 'open',
                              noise=d == 1, noise_color=noises[d]) for d in (0, 1)]
        # A bounded ruler constrains the leading edge, not the entire crate.
        # The second settled width-5 crate visibly extends past the terminal.
        s = guides[0]
        del s['marks'][s['t']]
        s['t'] = s['a'] + ((5 + s['p'] - 1)//s['p'] + 1)*s['p']
        s['hi'] = s['t']
        s['marks'][s['t']] = s['paint']
        s['docks'] = tuple(range(s['a'], s['t']+1, s['p']))
        h = int(rng.integers(3, 6))
        w0 = w1 = 5
        row = int(rng.integers(2, height - h - 1))
        base = [dict(r=row, c=0, h=h, w=w0, d=0),
                dict(r=row, c=0, h=h, w=w1, d=0),
                dict(r=row, c=0, h=h, w=w0, d=1)]
        for _ in range(400):
            objects = [dict(o, c=int(_pick(rng, _start_choices(o, guides[o['d']], width)))) for o in base]
            if any(_overlap(a, b) for i, a in enumerate(objects) for b in objects[i+1:]):
                continue
            try:
                settled = _pack(objects, guides, width)
            except ValueError:
                continue
            moved_classes = {o['d'] for o, f in zip(objects, settled) if o['c'] != f['c']}
            if moved_classes == {0, 1}:
                break
        else:
            # Three moving crates, constructed within the same finite lattice.
            objects = [dict(o) for o in base]
            objects[0]['c'] = s['a'] + s['p']
            objects[1]['c'] = s['t']
            choices = [c for c in _start_choices(objects[2], guides[1], width)
                       if c >= s['t'] + 5]
            objects[2]['c'] = min(choices)
        _appearance(rng, objects, guides, bg, shared=True)
    elif purpose == 'chain':
        width = int(rng.integers(27, 31))
        p = int(rng.integers(2, 4))
        guides = [_make_guide(rng, width, p, refs[0], paints[0], 'open'),
                  _make_guide(rng, width, 1, refs[1], paints[1], 'left_bound')]
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        step = int(rng.integers((h+1)//2, h))
        extent = h + 2*step
        other_h, other_w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        height = int(rng.integers(extent + other_h + 5, 31))
        start_row = int(rng.integers(2, height - extent - other_h - 2))
        phase = guides[0]['a'] % p
        stride = ((w+1+p-1)//p)*p
        max_offset = (width-w-phase-2*stride)//p
        offset = int(rng.integers(1, max_offset+1))
        reverse = bool(rng.integers(2))
        objects = [dict(r=start_row+(2-i if reverse else i)*step,
                        c=phase+offset*p+i*stride, h=h, w=w, d=0) for i in range(3)]
        other = dict(r=start_row+extent+1, c=0, h=other_h, w=other_w, d=1)
        independent = dict(r=other['r'], c=phase+2*p, h=other_h, w=w, d=0)
        choices = [c for c in _start_choices(other, guides[1], width)
                   if c >= independent['c'] + w]
        other['c'] = int(_pick(rng, choices[:-1]))
        # The isolated low mover shares the chain's first berth. This teaches
        # collision by perpendicular overlap, rather than a global cargo queue.
        objects.extend((other, independent))
        _appearance(rng, objects, guides, bg)
    else:
        test = purpose == 'test'
        width = int(rng.integers(24 if test else 16, 31))
        height = int(rng.integers(9, 31))
        if rng.random() < .2:
            height = width
        ps = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
        if test:
            ps[shifted_class] = int(rng.integers(4, min(7, (width-3)//3)+1))
        guides = []
        for d in (0, 1):
            kind = 'random' if rng.random() < .5 else 'open'
            guides.append(_make_guide(rng, width, ps[d], refs[d], paints[d], kind,
                                      noise=rng.random() < .4, noise_color=noises[d]))
        objects, area = [], 0
        requested = int(rng.integers(1, 10))
        bias = float(_pick(rng, (0., .2, .5, .5, .8, 1.)))
        for index in range(requested):
            for _ in range(90):
                d = shifted_class if test and index == 0 else int(rng.random() > bias)
                h, w = int(rng.integers(3, min(7, height-4)+1)), int(rng.integers(3, 8))
                o = dict(r=int(rng.integers(2, height-h-1)), c=0, h=h, w=w, d=d)
                choices = _start_choices(o, guides[d], width)
                if not choices or 2*(area+h*w+8) >= height*width:
                    continue
                o['c'] = int(_pick(rng, choices))
                if any(_overlap(o, prev) for prev in objects):
                    continue
                try:
                    _pack(objects+[o], guides, width)
                except ValueError:
                    continue
                objects.append(o)
                area += h*w
                break
            else:
                if objects:
                    break
        if not objects:
            # Both the ruler lattice and the minimum footprint are visibly
            # feasible. A tiny sole crate supplies a constructive rescue.
            d = shifted_class if test else 0
            o = dict(r=2, c=0, h=3, w=3, d=d)
            o['c'] = max(_start_choices(o, guides[d], width)) if d == 0 else min(_start_choices(o, guides[d], width))
            objects = [o]
        _appearance(rng, objects, guides, bg)
    bare = _blank((height, width), bg, guides)
    inp = _render(bare, bg, objects, guides, repaint=False)
    out = _render(bare, bg, _pack(objects, guides, width), guides)
    if axis == 0:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(2))
    fixed = [int(c) for c in rng.permutation(10)] if rng.random() < .3 else None
    shifted_class = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            purpose = ('calibration' if i == 0 else 'chain' if i == 1 else 'free') if split == 'train' else 'test'
            for _ in range(100):
                pair = _make_pair(rng, purpose, axis, fixed, shifted_class)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternate(g, bg, guides, objects, mode):
    altered = [dict(s) for s in guides]
    if mode == 'dense':
        for s in altered:
            s['docks'] = tuple(range(s['lo'], s['hi']+1))
    elif mode == 'unbounded':
        for s in altered:
            s['docks'] = tuple(x for x in range(g.shape[1]) if (x-s['a']) % s['p'] == 0)
    elif mode == 'plain':
        for s in altered:
            s['docks'] = tuple(range(g.shape[1]))
            s['paint'] = s['ref']
    elif mode == 'stationary':
        return _render(g, bg, objects, altered)
    settled = _pack(objects, altered, g.shape[1])
    return _render(g, bg, settled, altered)


def _features(pair):
    axis, bg, g, guides, objects = _decode(pair['input'])
    final = _pack(objects, guides, g.shape[1])
    expected, simulated = _tick_model(g, bg, guides, objects)
    actual = _grid(pair['output'])
    actual = actual if axis == 1 else actual.T.copy()
    assert np.array_equal(expected, actual), 'single-berth model disagrees with label'
    assert [o['c'] for o in simulated] == [o['c'] for o in final]
    assert np.array_equal(actual[0], g[0]) and np.array_equal(actual[-1], g[-1])
    assert np.all(actual[1] == bg) and np.all(actual[-2] == bg)
    for o, f in zip(objects, final):
        r, c, h, w = (f[k] for k in ('r', 'c', 'h', 'w'))
        assert np.array_equal(actual[r+1:r+h-1, c+1:c+w-1], o['patch'][1:-1, 1:-1]), 'payload changed'
        assert (c if o['d'] == 0 else c+w-1) in guides[o['d']]['docks']
    alternates = {}
    for mode in ('dense', 'unbounded', 'plain', 'stationary'):
        try:
            alternates[mode] = not np.array_equal(actual, _alternate(g, bg, guides, objects, mode))
        except ValueError:
            # An alternate which creates a class collision also fails to
            # explain this example; record the distinction rather than hide it.
            alternates[mode] = True
    old = _pack(objects, guides, g.shape[1], stable=False)
    old_witness = [o['c'] for o in old] != [o['c'] for o in final]
    moved = {o['d'] for o, f in zip(objects, final) if o['c'] != f['c']}
    terminal_classes = {o['d'] for o in objects if guides[o['d']]['t'] is not None}
    open_classes = {o['d'] for o in objects if guides[o['d']]['t'] is None}
    shared = any(a['d'] != b['d'] and a['patch'].shape == b['patch'].shape
                 and np.array_equal(a['patch'][1:-1, 1:-1], b['patch'][1:-1, 1:-1])
                 for a in objects for b in objects)
    return dict(axis=axis, background=bg, shape=pair['input'].shape,
                count=len(objects), periods=tuple(s['p'] for s in guides),
                used_periods=tuple(guides[o['d']]['p'] for o in objects),
                terminal_count=sum(s['t'] is not None for s in guides),
                terminal_directions=tuple(-1 if s['t'] < s['a'] else 1
                                          for s in guides if s['t'] is not None),
                noise=sum(len(s['noise']) for s in guides), stable_witness=old_witness,
                moved=moved, terminal_classes=terminal_classes, open_classes=open_classes,
                shared_payload=shared, alternatives=alternates,
                footprints=tuple(sorted({(o['h'], o['w']) for o in objects})),
                stationary=sum(o['c'] == f['c'] for o, f in zip(objects, final)),
                patterned=sum(len(np.unique(o['patch'][1:-1, 1:-1])) > 1 for o in objects),
                classes=tuple(sum(o['d'] == d for o in objects) for d in (0, 1)),
                overlap_edges=sum(a['d'] == b['d'] and _band(a, b)
                                  for i, a in enumerate(objects) for b in objects[i+1:]),
                span_overhang=any(guides[o['d']]['t'] is not None
                                  and (o['c'] < guides[o['d']]['lo']
                                       or o['c']+o['w']-1 > guides[o['d']]['hi']) for o in final),
                shared_berths=any(a['d'] == b['d'] and not _band(a, b)
                                  and (a['c'] if a['d'] == 0 else a['c']+a['w']-1)
                                  == (b['c'] if b['d'] == 0 else b['c']+b['w']-1)
                                  for i, a in enumerate(final) for b in final[i+1:]),
                signature=(g.shape, tuple((s['p'], s['a'] % s['p'], s['lo'], s['hi']) for s in guides),
                           tuple((o['r'], o['c'], o['h'], o['w'], o['d']) for o in objects)))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    analyses, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            # Accept both NumPy arrays and nested-list interface values.
            normalized = {'input': inp, 'output': out}
            a = _features(normalized)
            assert np.array_equal(solve(inp), out), 'input-only solve disagrees'
            analyses[split].append(a)
    train = analyses['train']
    assert len({a['axis'] for s in analyses.values() for a in s}) == 1
    assert set().union(*(a['moved'] for a in train)) == {0, 1}, 'both directions must move'
    assert set().union(*(a['terminal_classes'] for a in train)) == {0, 1}
    assert set().union(*(a['open_classes'] for a in train)) == {0, 1}
    assert any(a['shared_payload'] for a in train), 'payload cannot determine direction'
    assert any(a['stable_witness'] for a in train), 'stable order needs a demonstrated chain'
    assert any(a['noise'] for a in train), 'off-phase ruler marks need a demonstration'
    assert any(a['span_overhang'] for a in train), 'leading-edge rather than whole-footprint bounds need evidence'
    assert any(a['shared_berths'] for a in train), 'disjoint bands must demonstrate berth sharing'
    assert {-1, 1} <= {d for a in train for d in a['terminal_directions']}
    for mode in ('dense', 'unbounded', 'plain', 'stationary'):
        assert any(a['alternatives'][mode] for a in train), f'training fails to refute {mode}'
    max_train = max(p for a in train for p in a['used_periods'])
    assert all(max(a['used_periods']) > max_train for a in analyses['test']), 'missing actual period shift'


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test() -> dict:
    """Bounded 42-seed corpus plus concrete semantics and corruption checks."""
    corpus = [generate_puzzle(i) for i in range(42)]
    rows = [_features(p) for task in corpus for s in ('train', 'test') for p in task[s]]
    assert len({_fingerprint(task) for task in corpus}) == 42
    assert {a['axis'] for a in rows} == {0, 1}
    assert len({a['shape'] for a in rows}) >= 40
    assert len({a['signature'] for a in rows}) >= 150
    assert {1, 2, 3, 4, 5, 6, 7} <= {p for a in rows for p in a['periods']}
    assert len({a['background'] for a in rows}) == 10
    assert {0, 1, 2} <= {a['terminal_count'] for a in rows}
    assert any(a['count'] == 1 for a in rows) and max(a['count'] for a in rows) >= 6
    assert any(0 in a['classes'] for a in rows)
    assert any(len(a['footprints']) > 1 for a in rows)
    assert any(a['patterned'] for a in rows) and any(not a['patterned'] for a in rows)
    assert any(a['noise'] == 0 for a in rows)
    assert any(0 < a['stationary'] < a['count'] for a in rows)
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    # A hand-designed case: phase 1 mod 3, two width-4 overlapping crates.
    # Berths 1,4,7,... give positions 1 and 7, not ordinary positions 0,4.
    guides = [dict(p=3, a=4, docks=(1, 4, 7, 10, 13, 16, 19), ref=2, paint=2),
              dict(p=1, a=1, docks=tuple(range(20)), ref=3, paint=3)]
    boxes = [dict(r=2, c=7, h=3, w=4, d=0), dict(r=2, c=13, h=3, w=4, d=0)]
    assert [o['c'] for o in _pack(boxes, guides, 20)] == [1, 7]
    # Equal lattice, stable chain: ends do not overlap in rows, but their
    # common middle predecessor forbids the last crate returning to the edge.
    boxes = [dict(r=r, c=c, h=3, w=3, d=0) for r, c in ((2, 4), (4, 8), (6, 12))]
    guides[0]['docks'], guides[0]['p'] = tuple(range(20)), 1
    assert [o['c'] for o in _pack(boxes, guides, 20)] == [0, 3, 6]
    assert [o['c'] for o in _pack(boxes, guides, 20, stable=False)] == [0, 3, 0]
    # Bounds restrict docks even when the paint terminal is on the wrong side
    # to act as an attraction point; oversized footprints are not clipped.
    guides[0]['docks'] = (4, 7, 10)
    assert _pack([dict(r=2, c=10, h=3, w=4, d=0)], guides, 20)[0]['c'] == 4
    # Literal input/output fixture checks the entire parser, frame repaint,
    # payload copy, wrong-side terminal, and multiple inert singleton colors.
    inp = np.zeros((8, 20), dtype=np.int8)
    inp[0, [4, 7]], inp[0, 13] = 2, 5
    inp[0, 11], inp[0, 2], inp[-1, 1:3] = 9, 7, 3
    inp[2:6, 10:14] = np.array([[2, 2, 2, 2], [2, 6, 8, 2],
                                [2, 8, 6, 2], [2, 2, 2, 2]])
    expected = np.zeros_like(inp)
    expected[0], expected[-1] = inp[0], inp[-1]
    expected[2:6, 4:8] = np.array([[5, 5, 5, 5], [5, 6, 8, 5],
                                  [5, 8, 6, 5], [5, 5, 5, 5]])
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(inp.T), expected.T)
    assert np.array_equal(solve(9-inp), 9-expected)
    no_noise = inp.copy()
    no_noise[0, 11] = 0
    no_noise_expected = expected.copy()
    no_noise_expected[0, 11] = 0
    assert np.array_equal(solve(no_noise), no_noise_expected)
    no_terminal = inp.copy()
    no_terminal[0, 13] = 0
    open_expected = np.zeros_like(inp)
    open_expected[0], open_expected[-1] = no_terminal[0], no_terminal[-1]
    open_expected[2:6, 1:5] = inp[2:6, 10:14]
    assert np.array_equal(solve(no_terminal), open_expected)
    # The intended family has no period-seven cap. This period-fourteen
    # input has only one cargo class, and needs no training-side parameters.
    large_period = np.zeros((7, 30), dtype=np.int8)
    large_period[0, [1, 15]] = 2
    large_period[-1, [2, 4]] = 3
    large_period[2:5, 15:18] = np.array([[2, 2, 2], [2, 4, 2], [2, 2, 2]])
    large_expected = np.zeros_like(large_period)
    large_expected[0], large_expected[-1] = large_period[0], large_period[-1]
    large_expected[2:5, 1:4] = large_period[2:5, 15:18]
    assert np.array_equal(solve(large_period), large_expected)
    broken = copy.deepcopy(corpus[0])
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('validator accepted an unchanged held-out grid')
    for counts in ((2, 1), (10, 4), (6, 3)):
        validate_generated_puzzle(generate_puzzle(123456, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len({a['shape'] for a in rows}),
                geometric_signatures=len({a['signature'] for a in rows}),
                counts=sorted({a['count'] for a in rows}),
                periods=sorted({p for a in rows for p in a['periods']}),
                footprints=sorted({f for a in rows for f in a['footprints']}),
                stable_witnesses=sum(a['stable_witness'] for a in rows),
                note='Generator-owned semantic checks; no independent approval.')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
