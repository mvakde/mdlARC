"""Stencil-powered engraving: reflected rays move color-selected cutting tools.

All interpretation comes from the input. Generation preferences only select
demonstrations; there is no seed-dependent rule, color dictionary, or route plan.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _components(a, excluded):
    """Four-neighbor components respect color, including touching colors."""
    left = {tuple(map(int, p)) for p in np.argwhere(a != excluded)}
    result = []
    while left:
        start = min(left)
        left.remove(start)
        todo, cells = [start], []
        shade = int(a[start])
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in left and a[p] == shade:
                    left.remove(p)
                    todo.append(p)
        result.append((shade, tuple(sorted(cells))))
    return result


def _objects(plate, material):
    sources, mirrors = [], []
    for shade, cells in _components(plate, material):
        rs, cs = {r for r, _ in cells}, {c for _, c in cells}
        if len(cells) == 3 and len(rs) == len(cs) == 2:
            assert max(rs)-min(rs) == max(cs)-min(cs) == 1
            missing = next(iter({(r, c) for r in rs for c in cs} - set(cells)))
            elbow = (min(rs)+max(rs)-missing[0], min(cs)+max(cs)-missing[1])
            direction = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append(dict(color=shade, cells=cells, elbow=elbow,
                                direction=direction))
        else:
            assert len(cells) % 2 == 1 and (len(rs) == 1 or len(cs) == 1), "invalid glyph"
            assert len(cells) == max(max(rs)-min(rs), max(cs)-min(cs))+1
            orientation = 'point' if len(cells) == 1 else 'horizontal' if len(rs) == 1 else 'vertical'
            mirrors.append(dict(color=shade, cells=cells, orientation=orientation,
                                midpoint=((min(rs)+max(rs))//2, (min(cs)+max(cs))//2)))
    assert sources, "no L source"
    h, w = plate.shape
    assert all(0 < r < h-1 and 0 < c < w-1
               for obj in sources+mirrors for r, c in obj['cells'])
    return sources, mirrors


def _decode(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])), "canvas border"
    candidates = []
    for shade in np.unique(a):
        shade = int(shade)
        if shade == bg:
            continue
        cells = np.argwhere(a == shade)
        y, x = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0)+1)
        block = a[y:bottom, x:right]
        if min(block.shape) >= 8 and all(np.all(edge == shade) for edge in
                (block[0], block[-1], block[:, 0], block[:, -1])):
            candidates.append((shade, y, x, bottom, right))
    assert len(candidates) == 1, "plate rim must identify one workpiece"
    material, y, x, bottom, right = candidates[0]
    assert y >= 1 and x >= 1 and bottom < a.shape[0] and right < a.shape[1]
    crop = (y-1, x-1, bottom+1, right+1)
    inside_crop = np.zeros(a.shape, dtype=bool)
    inside_crop[crop[0]:crop[2], crop[1]:crop[3]] = True
    plate = a[y:bottom, x:right].copy()
    assert not np.any(plate == bg), "input plate is solid beneath its glyphs"
    border = a[crop[0]:crop[2], crop[1]:crop[3]]
    assert all(np.all(edge == bg) for edge in
               (border[0], border[-1], border[:, 0], border[:, -1]))
    sources, mirrors = _objects(plate, material)
    external = (~inside_crop) & (a != bg)
    masks, boxes = {}, {}
    for shade in sorted(map(int, np.unique(a[external]))):
        assert shade != material
        cells = np.argwhere(external & (a == shade))
        sy, sx = map(int, cells.min(axis=0))
        sb, sr = map(int, cells.max(axis=0)+1)
        k = sb-sy
        assert k == sr-sx and k >= 3 and k % 2 == 1, "odd full-span register"
        assert not inside_crop[sy:sb, sx:sr].any()
        register = a[sy:sb, sx:sr]
        assert np.all((register == bg) | (register == shade))
        mask = register == shade
        assert 0 < mask.sum() < k*k
        assert all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))
        masks[shade], boxes[shade] = mask, (sy, sx, sb, sr)
    assert masks and len({m.shape for m in masks.values()}) == 1
    assert {o['color'] for o in sources+mirrors} <= set(masks), "missing color-matched tool"
    bxs = list(boxes.values())
    for i, b1 in enumerate(bxs):
        for b2 in bxs[i+1:]:
            assert b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1], "register gap"
    return dict(input=a, background=bg, material=material, plate=plate,
                origin=(y, x), crop=crop, sources=sources, mirrors=mirrors,
                masks=masks, boxes=boxes)


def _contact(mirror, position, direction):
    r, c = position
    mr, mc = mirror['midpoint']
    dr, dc = direction
    if mirror['orientation'] in ('point', 'horizontal') and c == mc and mr-r == dr:
        return 0
    if mirror['orientation'] in ('point', 'vertical') and r == mr and mc-c == dc:
        return 1
    return None


def _trace(plate, material, sources, mirrors, *, strict=True,
           reflect=True, recolor=True, late_contact=False):
    """Trace the original plate, never the holes or other beams' results."""
    h, w = plate.shape
    routes, used = [], set()
    for source in sources:
        position, direction, shade = source['elbow'], source['direction'], source['color']
        seen, path, contacts = set(), [], []
        while True:
            position = (position[0]+direction[0], position[1]+direction[1])
            if not (0 < position[0] < h-1 and 0 < position[1] < w-1):
                break
            state = position, direction
            assert state not in seen, "beam loop"
            seen.add(state)
            if strict:
                assert plate[position] == material, "beam center hits a glyph"
            hits = [(i, _contact(m, position, direction)) for i, m in enumerate(mirrors)]
            hits = [(i, axis) for i, axis in hits if axis is not None]
            if strict:
                assert len(hits) <= 1, "competing midpoint faces"
            old = shade
            if reflect and hits:
                i, axis = hits[0]
                used.add(i)
                if recolor:
                    shade = mirrors[i]['color']
                direction = (-direction[0], direction[1]) if axis == 0 else (direction[0], -direction[1])
                contacts.append((i, axis, position, old, shade))
            path.append((position, old if late_contact else shade))
        assert path, "source needs a visible flight"
        routes.append(dict(path=path, contacts=contacts))
    if strict:
        assert used == set(range(len(mirrors))), "unused mirror"
    return routes


def _transforms(mask):
    return [np.rot90(mask, i) for i in range(4)] + [np.rot90(np.fliplr(mask), i) for i in range(4)]


def _holes(info, routes, masks=None, centers_only=False, parity=False):
    plate, material = info['plate'], info['material']
    masks = info['masks'] if masks is None else masks
    result, per_source = np.zeros(plate.shape, dtype=bool), []
    for route in routes:
        local = np.zeros(plate.shape, dtype=bool)
        for (r, c), shade in route['path']:
            mask = masks[shade]
            radius = len(mask)//2
            offsets = [(0, 0)] if centers_only else [(int(y)-radius, int(x)-radius)
                                                    for y, x in np.argwhere(mask)]
            for dy, dx in offsets:
                yy, xx = r+dy, c+dx
                if 0 < yy < plate.shape[0]-1 and 0 < xx < plate.shape[1]-1 and plate[yy, xx] == material:
                    if parity:
                        local[yy, xx] ^= True
                    else:
                        local[yy, xx] = True
        if parity:
            result ^= local
        else:
            result |= local
        per_source.append(local)
    return result, per_source


def _output(info, routes=None, masks=None, centers_only=False, parity=False):
    if routes is None:
        routes = _trace(info['plate'], info['material'], info['sources'], info['mirrors'])
    holes, _ = _holes(info, routes, masks, centers_only, parity)
    out = np.full(tuple(d+2 for d in info['plate'].shape), info['background'], dtype=int)
    out[1:-1, 1:-1] = info['plate']
    out[1:-1, 1:-1][holes] = info['background']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible rim, glyphs, and stencils; return the engraved crop."""
    return _output(_decode(input_grid))


def _alternatives(info):
    """Finite teaching probes, not an independent ARC solver."""
    plate, mat, sources, mirrors = (info[k] for k in ('plate', 'material', 'sources', 'mirrors'))
    normal = _trace(plate, mat, sources, mirrors)
    straight = _trace(plate, mat, sources, mirrors, strict=False, reflect=False)
    fixed = _trace(plate, mat, sources, mirrors, recolor=False)
    late = _trace(plate, mat, sources, mirrors, late_contact=True)
    return dict(no_reflection=_output(info, straight), no_selection=_output(info, fixed),
                center_line=_output(info, normal, centers_only=True),
                late_contact=_output(info, late),
                complement=_output(info, normal, {c: ~m for c, m in info['masks'].items()}),
                xor=_output(info, normal, parity=True))


def _teaching_evidence(infos):
    axes, directions, distinguished, orientations = set(), set(), set(), set()
    for info in infos:
        routes = _trace(info['plate'], info['material'], info['sources'], info['mirrors'])
        axes.update(axis for route in routes for _, axis, _, _, _ in route['contacts'])
        directions.update(s['direction'] for s in info['sources'])
        out = _output(info, routes)
        distinguished.update(name for name, alt in _alternatives(info).items() if not np.array_equal(out, alt))
        for i in range(1, 8):
            transformed = {c: _transforms(m)[i] for c, m in info['masks'].items()}
            if not np.array_equal(out, _output(info, routes, transformed)):
                orientations.add(i)
    return (axes == {0, 1} and len(directions) >= 2 and len(orientations) == 7
            and distinguished == {'no_reflection', 'no_selection', 'center_line', 'late_contact', 'complement', 'xor'})


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _mirror_cells(mid, axis, length):
    r, c = mid
    return {(r, c+t) if axis == 0 else (r+t, c) for t in range(-(length//2), length//2+1)}


def _place(plate, material, cells, shade):
    h, w = plate.shape
    for r, c in cells:
        if not (0 < r < h-1 and 0 < c < w-1) or plate[r, c] != material:
            return None
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p not in cells and plate[p] == shade:
                return None
    trial = plate.copy()
    for cell in cells:
        trial[cell] = shade
    return trial


def _extend(plate, material, elbow, rng, colors, axis=None, changed=False):
    sources, mirrors = _objects(plate, material)
    routes = _trace(plate, material, sources, mirrors)
    source, route = next((s, r) for s, r in zip(sources, routes) if s['elbow'] == elbow)
    direction = source['direction']
    for _, a, _, _, _ in route['contacts']:
        direction = (-direction[0], direction[1]) if a == 0 else (direction[0], -direction[1])
    anchor = route['contacts'][-1][2] if route['contacts'] else elbow
    incoming = route['contacts'][-1][4] if route['contacts'] else source['color']
    options = []
    for distance in range(2, 30):
        contact = (anchor[0]+distance*direction[0], anchor[1]+distance*direction[1])
        if not (0 < contact[0] < plate.shape[0]-1 and 0 < contact[1] < plate.shape[1]-1):
            break
        for a in ([axis] if axis is not None else [0, 1]):
            for length in (1, 3, 5):
                options.append((contact, a, length))
    rng.shuffle(options)
    for contact, a, length in options:
        mid = (contact[0]+(direction[0] if a == 0 else 0),
               contact[1]+(direction[1] if a == 1 else 0))
        choices = [c for c in colors if not changed or c != incoming]
        shade = int(rng.choice(choices))
        trial = _place(plate, material, _mirror_cells(mid, a, length), shade)
        if trial is None:
            continue
        try:
            ss, mm = _objects(trial, material)
            rr = _trace(trial, material, ss, mm)
        except AssertionError:
            continue
        new_route = next(r for s, r in zip(ss, rr) if s['elbow'] == elbow)
        old = [(mirrors[i]['midpoint'], a0, p) for i, a0, p, _, _ in route['contacts']]
        new = [(mm[i]['midpoint'], a0, p) for i, a0, p, _, _ in new_route['contacts']]
        if new == old+[(mid, a, contact)] and len(new_route['path']) >= len(route['contacts'])+4:
            return trial
    return None


def _add_source(plate, material, rng, colors, excluded_directions=()):
    options = [(r, c, d) for r in range(1, plate.shape[0]-1)
               for c in range(1, plate.shape[1]-1) for d in DIRECTIONS if d not in excluded_directions]
    rng.shuffle(options)
    for r, c, d in options:
        trial = _place(plate, material, _source_cells((r, c), d), int(rng.choice(colors)))
        if trial is None:
            continue
        try:
            ss, mm = _objects(trial, material)
            rr = _trace(trial, material, ss, mm)
        except AssertionError:
            continue
        route = next(route for s, route in zip(ss, rr) if s['elbow'] == (r, c))
        if len(route['path']) >= 4:
            return trial, (r, c)
    return None


def _sample_mask(rng, k, asymmetric=False):
    for _ in range(200):
        m = np.zeros((k, k), dtype=bool)
        family = int(rng.integers(7))
        if family == 0:
            np.fill_diagonal(m, True)
        elif family == 1:
            m[0] = True
            m[:, int(rng.integers(k))] = True
        elif family == 2:
            m[0, 0] = m[-1, -1] = True
        elif family == 3:
            m[:] = True
            m.flat[int(rng.integers(k*k))] = False
        else:
            m = rng.random((k, k)) < rng.uniform(.15, .75)
        m = _transforms(m)[int(rng.integers(8))].copy()
        if not (0 < m.sum() < m.size and all((m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any()))):
            continue
        if asymmetric and len({t.tobytes() for t in _transforms(m)}) != 8:
            continue
        return m
    m = np.zeros((k, k), dtype=bool)
    m[0, 0] = m[0, -1] = m[-1, 0] = m[1, -1] = True
    return m


def _layouts(ph, pw, k, n):
    choices = []
    for columns in range(1, n+1):
        rows = (n+columns-1)//columns
        for gap in (1, 2, 3):
            bh, bw = rows*k+(rows-1)*gap, columns*k+(columns-1)*gap
            h, w = max(ph+2, bh+2), pw+bw+5
            if h <= 30 and w <= 30:
                choices.append(('side', rows, columns, gap, h, w, bh, bw))
            h, w = ph+bh+5, max(pw+2, bw+2)
            if h <= 30 and w <= 30:
                choices.append(('above', rows, columns, gap, h, w, bh, bw))
    return choices


def _pack(rng, plate, bg, masks):
    k, n = len(next(iter(masks.values()))), len(masks)
    choices = _layouts(*plate.shape, k, n)
    assert choices
    mode, nr, nc, gap, h, w, bh, bw = choices[int(rng.integers(len(choices)))]
    ph, pw = plate.shape
    if mode == 'side':
        py, px = int(rng.integers(1, h-ph)), 1
        sy, sx = int(rng.integers(1, h-bh)), pw+4
    else:
        py, px = 1, int(rng.integers(1, w-pw))
        sy, sx = ph+4, int(rng.integers(1, w-bw))
    slots = [(sy+r*(k+gap), sx+c*(k+gap)) for r in range(nr) for c in range(nc)]
    rng.shuffle(slots)
    slots = slots[:n]
    # In some scenes, move the registers anywhere outside the protected crop.
    if rng.random() < .5:
        hh, ww = int(rng.integers(h, 31)), int(rng.integers(w, 31))
        yy, xx = int(rng.integers(1, hh-ph)), int(rng.integers(1, ww-pw))
        blocked = np.zeros((hh, ww), dtype=bool)
        blocked[yy-1:yy+ph+1, xx-1:xx+pw+1] = True
        scattered = []
        positions = [(r, c) for r in range(1, hh-k) for c in range(1, ww-k)]
        rng.shuffle(positions)
        for r, c in positions:
            if blocked[r:r+k, c:c+k].any():
                continue
            scattered.append((r, c))
            blocked[r-1:r+k+1, c-1:c+k+1] = True
            if len(scattered) == n:
                h, w, py, px, slots = hh, ww, yy, xx, scattered
                break
    inp = np.full((h, w), bg, dtype=int)
    inp[py:py+ph, px:px+pw] = plate
    colors = list(masks)
    rng.shuffle(colors)
    for (r, c), shade in zip(slots, colors):
        inp[r:r+k, c:c+k][masks[shade]] = shade
    # Whole-scene transforms vary geometry, not the local tool orientation rule.
    if rng.integers(2):
        inp = inp.T.copy()
    if rng.integers(2):
        inp = inp[::-1].copy()
    if rng.integers(2):
        inp = inp[:, ::-1].copy()
    padding = []
    for dim in inp.shape:
        amount = int(rng.integers(min(3, 30-dim)+1))
        before = int(rng.integers(amount+1))
        padding.append((before, amount-before))
    return np.pad(inp, padding, constant_values=bg)


def _make_pair(rng, teaching=None, excluded_directions=(), test=False):
    for _ in range(120):
        k = int(rng.choice([3, 5]))
        n = int(rng.integers(3, 7))
        ph, pw = int(rng.integers(10, 23)), int(rng.integers(10, 23))
        if not _layouts(ph, pw, k, n):
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, material, colors = palette[0], palette[1], palette[2:2+n]
        # Keep a genuine unmatched tool in the two minimum demonstrations.
        active = colors[:-1] if teaching is not None else colors[:int(rng.integers(2, n+1))]
        plate = np.full((ph, pw), material, dtype=int)
        added = _add_source(plate, material, rng, active)
        if added is None:
            continue
        plate, elbow = added
        desired_bounces = (1 if teaching == 0 else 2) if teaching is not None else int(rng.integers(0, 4))
        for bounce in range(desired_bounces):
            axis = teaching if teaching is not None and bounce == 0 else None
            trial = _extend(plate, material, elbow, rng, active, axis, changed=(bounce == 0))
            if trial is None:
                break
            plate = trial
        ss, mm = _objects(plate, material)
        rr = _trace(plate, material, ss, mm)
        if teaching is not None and len(rr[0]['contacts']) < desired_bounces:
            continue
        wanted_sources = int(rng.integers(2, 5)) if test else 1 if teaching is not None else int(rng.integers(1, 4))
        for _ in range(wanted_sources-1):
            added = _add_source(plate, material, rng, active)
            if added is None:
                break
            plate, new_elbow = added
            if rng.random() < .55:
                trial = _extend(plate, material, new_elbow, rng, active)
                if trial is not None:
                    plate = trial
        ss, mm = _objects(plate, material)
        if test and len(ss) < 2:
            continue
        masks = {c: _sample_mask(rng, k, asymmetric=(teaching is not None and c == ss[0]['color'])) for c in colors}
        inp = _pack(rng, plate, bg, masks)
        info = _decode(inp)
        if teaching is not None and info['sources'][0]['direction'] in excluded_directions:
            continue
        routes = _trace(info['plate'], info['material'], info['sources'], info['mirrors'])
        holes, per_source = _holes(info, routes)
        if not holes.any():
            continue
        if test and not any(np.any(a & b) for i, a in enumerate(per_source) for b in per_source[i+1:]):
            continue
        if teaching is not None:
            out = _output(info, routes)
            alts = _alternatives(info)
            if any(np.array_equal(out, alts[name]) for name in ('no_reflection', 'no_selection', 'center_line', 'complement', 'xor')):
                continue
        return {'input': inp, 'output': _output(info, routes)}
    raise RuntimeError('bounded scene search exhausted')


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _checked_count(num_train, 'num_train', 2, 10)
    ne = _checked_count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    for _ in range(40):
        first = _make_pair(rng, teaching=0)
        directions = [s['direction'] for s in _decode(first['input'])['sources']]
        second = _make_pair(rng, teaching=1, excluded_directions=directions)
        infos = [_decode(p['input']) for p in (first, second)]
        if _teaching_evidence(infos):
            break
    else:
        raise RuntimeError('bounded demonstration search exhausted')
    puzzle, seen = {'train': [first, second], 'test': []}, {_key(first['input']), _key(second['input'])}
    for split, count in (('train', nt), ('test', ne)):
        while len(puzzle[split]) < count:
            for _ in range(40):
                pair = _make_pair(rng, test=(split == 'test'))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness search exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _decode(inp)
            routes = _trace(info['plate'], info['material'], info['sources'], info['mirrors'])
            holes, per_source = _holes(info, routes)
            assert holes.any()
            assert np.array_equal(holes, np.logical_or.reduce(per_source))
            assert np.array_equal(holes, _holes(info, list(reversed(routes)))[0]), 'source order affects cuts'
            assert np.array_equal(out, _output(info, routes)), 'incorrect holes or crop'
            assert out.shape == tuple(d+2 for d in info['plate'].shape)
            core = out[1:-1, 1:-1]
            changed = core != info['plate']
            assert np.array_equal(changed, holes)
            assert np.all(core[changed] == info['background'])
            assert np.all(core[info['plate'] != info['material']] == info['plate'][info['plate'] != info['material']])
            assert all(np.all(edge == info['material']) for edge in (core[0], core[-1], core[:, 0], core[:, -1]))
            assert all(np.all(edge == info['background']) for edge in (out[0], out[-1], out[:, 0], out[:, -1]))
            assert set(np.unique(out)) <= set(np.unique(inp))
            assert np.array_equal(solve(inp.copy()), out)
            if split == 'train':
                train_infos.append(info)
    assert _teaching_evidence(train_infos), 'insufficient teaching evidence'


def _fingerprint(puzzle):
    return tuple(_key(_grid(pair[k])) for s in ('train', 'test')
                 for pair in puzzle[s] for k in ('input', 'output'))


def _normalized_key(a):
    colors = {}
    return a.shape, bytes(colors.setdefault(int(c), len(colors)) for c in a.flat)


def self_test() -> dict:
    """Bounded 42-seed audit plus explicit rule and damaged-output checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, plate_shapes, structures, tools, backgrounds = set(), set(), set(), set(), set()
    source_counts, lengths, directions, bounces, axis_sequences = set(), set(), set(), set(), set()
    overlaps = unchanged = returning = distractors = symmetric = asymmetric = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                routes = _trace(info['plate'], info['material'], info['sources'], info['mirrors'])
                _, per_source = _holes(info, routes)
                shapes.add(pair['input'].shape)
                plate_shapes.add(info['plate'].shape)
                structures.add(_normalized_key(info['plate']))
                tools.add(len(next(iter(info['masks'].values()))))
                backgrounds.add(info['background'])
                source_counts.add(len(info['sources']))
                lengths.update(len(m['cells']) for m in info['mirrors'])
                directions.update(s['direction'] for s in info['sources'])
                used_colors = {o['color'] for o in info['sources']+info['mirrors']}
                distractors += bool(set(info['masks'])-used_colors)
                for mask in info['masks'].values():
                    asym = len({t.tobytes() for t in _transforms(mask)}) == 8
                    asymmetric += asym
                    symmetric += not asym
                overlaps += any(np.any(a & b) for i, a in enumerate(per_source) for b in per_source[i+1:])
                for s, route in zip(info['sources'], routes):
                    bounces.add(len(route['contacts']))
                    seq = tuple(a for _, a, _, _, _ in route['contacts'])
                    axis_sequences.add(seq)
                    unchanged += any(old == new for _, _, _, old, new in route['contacts'])
                    shades = [s['color']]+[new for _, _, _, _, new in route['contacts']]
                    returning += any(c in shades[:i-1] and c != shades[i-1] for i, c in enumerate(shades) if i >= 2)
                if split == 'test':
                    assert len(info['sources']) >= 2
                    assert any(np.any(a & b) for i, a in enumerate(per_source) for b in per_source[i+1:])
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len(shapes) >= 40 and len(plate_shapes) >= 40 and len(structures) >= 180
    assert tools == {3, 5} and backgrounds == set(range(10))
    assert source_counts >= {1, 2, 3} and lengths == {1, 3, 5} and directions == set(DIRECTIONS)
    assert bounces >= {0, 1, 2, 3}
    assert any(len(s) >= 2 and s[0] == s[1] for s in axis_sequences)
    assert min(overlaps, unchanged, returning, distractors, symmetric, asymmetric) > 0
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(100+nt*10+ne, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = copy.deepcopy(puzzles[0])
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    _rule_regressions()
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), plate_shapes=len(plate_shapes),
                plate_structures=len(structures), tool_sizes=sorted(tools),
                source_counts=sorted(source_counts), reflection_counts=sorted(bounces),
                overlapping_pairs=overlaps, unchanged_color_routes=unchanged,
                returning_color_routes=returning, pairs_with_unused_tools=distractors,
                symmetric_tools=symmetric, asymmetric_tools=asymmetric,
                independent_verification=False)


def _rule_regressions():
    # Hand-specified geometry and coordinates, independent of the scene builder.
    a = np.zeros((14, 22), dtype=int)
    a[1:13, 1:13] = 8
    for p in ((3, 3), (2, 3), (3, 2)):
        a[p] = 1
    a[7, 7:10] = 2  # horizontal midpoint (7,8), contact (6,8)
    # Correct the source-to-contact diagonal: elbow (3,5) -> (6,8).
    a[a == 1] = 8
    for p in ((3, 5), (2, 5), (3, 4)):
        a[p] = 1
    a[1:4, 16:19] = [[1, 0, 1], [0, 0, 1], [1, 0, 0]]
    a[6:9, 16:19] = [[2, 0, 0], [0, 0, 0], [0, 0, 2]]
    info = _decode(a)
    routes = _trace(info['plate'], info['material'], info['sources'], info['mirrors'])
    # Local plate coordinates are global coordinates minus (1,1).
    expected_path = [((3, 5), 1), ((4, 6), 1), ((5, 7), 2),
                     ((4, 8), 2), ((3, 9), 2), ((2, 10), 2)]
    assert routes[0]['path'] == expected_path
    expected = info['plate'].copy()
    offsets = {1: [(-1, -1), (-1, 1), (0, 1), (1, -1)],
               2: [(-1, -1), (1, 1)]}
    holes = set()
    for (r, c), shade in expected_path:
        for dr, dc in offsets[shade]:
            p = r+dr, c+dc
            if 0 < p[0] < 11 and 0 < p[1] < 11 and expected[p] == 8:
                holes.add(p)
    for p in holes:
        expected[p] = 0
    assert np.array_equal(solve(a)[1:-1, 1:-1], expected)
    assert solve(a)[6, 8] == 8, 'empty tool center does not cut the beam center'
    assert solve(a)[7, 8] == 2, 'mirror ink survives footprints'
    # A fitting extension with 7-cell masks and a 7-cell bar. This checks the
    # decoder's general odd-size rule, beyond the sampler's 3/5 preferences.
    extended = np.zeros((22, 25), dtype=int)
    extended[1:21, 1:14] = 8
    for p in ((3, 3), (2, 3), (3, 2)):
        extended[p] = 1
    extended[10, 6:13] = 2
    extended[1:8, 16:23][np.eye(7, dtype=bool)] = 1
    extended[11:18, 16:23][np.fliplr(np.eye(7, dtype=bool))] = 2
    ex_info = _decode(extended)
    ex_routes = _trace(ex_info['plate'], ex_info['material'], ex_info['sources'], ex_info['mirrors'])
    assert ex_routes[0]['path'] == [((r-1, r-1), 1) for r in range(4, 9)] + [
        ((8, 8), 2), ((7, 9), 2), ((6, 10), 2), ((5, 11), 2)]
    expected = ex_info['plate'].copy()
    for (r, c), shade in ex_routes[0]['path']:
        for t in range(-3, 4):
            yy, xx = r+t, c+(t if shade == 1 else -t)
            if 0 < yy < 19 and 0 < xx < 12 and expected[yy, xx] == 8:
                expected[yy, xx] = 0
    assert np.array_equal(solve(extended)[1:-1, 1:-1], expected)
    baseline = solve(a)
    assert np.array_equal(baseline, solve(a.tolist()))
    for i in range(8):
        assert np.array_equal(solve(_transforms(a)[i]), _transforms(baseline)[i]), 'scene equivariance'


if __name__ == '__main__':
    report = self_test()
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/self_test.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(report, separators=(',', ':')))
