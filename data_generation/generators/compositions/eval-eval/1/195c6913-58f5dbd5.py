"""Pigment spills in a carved laminate. All decoding uses the input alone."""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts: 2..10 train, 1..4 test, train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _orient(a, code):
    return np.rot90(a[:, ::-1] if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    a = np.rot90(a, -(code % 4))
    return (a[:, ::-1] if code >= 4 else a).copy()


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
            if diagonal:
                neighbors += [(y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)]
            for yy, xx in neighbors:
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        groups.append(tuple(cells))
    return groups


def _box(cells):
    return min(p[0] for p in cells), min(p[1] for p in cells), max(p[0] for p in cells)+1, max(p[1] for p in cells)+1


def _span(m):
    return bool(m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any())


def _trace(material, start):
    """Monotone maximal east/north legs. Collision events remember tip phase."""
    r, c = start
    direction = 0
    path, events, lengths = [], [], []
    for _ in range(sum(material.shape)+1):
        dr, dc = ((0, 1), (-1, 0))[direction]
        length = 0
        while 0 <= r < material.shape[0] and 0 <= c < material.shape[1] and material[r, c]:
            path.append((r, c))
            length += 1
            r, c = r+dr, c+dc
        assert length
        lengths.append(length)
        if not (0 <= r < material.shape[0] and 0 <= c < material.shape[1]):
            break  # Canvas exits do not cause a turn.
        events.append(((r, c), len(path)-1))
        tip = path[-1]
        direction = 1-direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        r, c = tip[0]+dr, tip[1]+dc
        if not (0 <= r < material.shape[0] and 0 <= c < material.shape[1] and material[r, c]):
            break
    else:
        raise AssertionError("nonterminating trace")
    assert len(set(path)) == len(path)
    return dict(path=tuple(path), events=tuple(events), lengths=tuple(lengths))


def _decode_canonical(inp):
    # An untouched outer border identifies background even for dense laminates.
    border = np.r_[inp[0], inp[-1], inp[:, 0], inp[:, -1]]
    assert len(set(map(int, border))) == 1
    bg = int(border[0])
    colors = [int(x) for x in np.unique(inp) if x != bg]
    comps = {color: _components(inp == color) for color in colors}
    tiles = []
    for color in colors:
        candidates = []
        for cells in comps[color]:
            y, x, b, e = _box(cells)
            region = inp[y:b, x:e]
            if min(region.shape) < 5:
                continue
            other = np.argwhere(region != color)
            if len(other) == 0 or (len(other) == 1 and int(region[tuple(other[0])]) != bg):
                candidates.append((region.size, y, x, b, e, color))
        if candidates:
            _, y, x, b, e, color = max(candidates)
            tiles.append((y, x, b, e, color))
    assert 1 <= len(tiles) <= 3
    tile_colors = {t[4] for t in tiles}
    occupied = np.zeros(inp.shape, bool)
    starts = []
    for y, x, b, e, color in tiles:
        assert not occupied[y:b, x:e].any()
        occupied[y:b, x:e] = True
        for dy, dx in np.argwhere(inp[y:b, x:e] != color):
            starts.append((y+int(dy), x+int(dx)))
    assert len(_components(occupied)) == 1, "laminate must be connected"
    assert len(starts) == 1
    start = starts[0]
    first = int(inp[start])
    assert first not in tile_colors and first != bg
    y0 = min(t[0] for t in tiles)-1
    x0 = min(t[1] for t in tiles)-1
    y1 = max(t[2] for t in tiles)+1
    x1 = max(t[3] for t in tiles)+1
    assert y0 >= 0 and x0 >= 0 and y1 <= inp.shape[0] and x1 <= inp.shape[1]
    assert np.all(inp[y0:y1, x0:x1][~occupied[y0:y1, x0:x1]] == bg)
    # The start is on an exposed west edge, establishing the reading direction.
    assert occupied[start] and not occupied[start[0], start[1]-1]
    assert occupied[start[0], start[1]+1]
    external = ~occupied & (inp != bg)
    patterns, registers, swatches = {}, {}, []
    for color in colors:
        cells = [tuple(map(int, p)) for p in np.argwhere(external & (inp == color))]
        if not cells:
            continue
        parts = _components(external & (inp == color))
        is_swatches = color not in tile_colors and all(len(p) == 4 and _box(p)[2]-_box(p)[0] == 2 and _box(p)[3]-_box(p)[1] == 2 for p in parts)
        if is_swatches:
            swatches.extend((_box(p)[0], _box(p)[1], color) for p in parts)
            continue
        y, x, b, e = _box(cells)
        assert b-y >= 3 and e-x >= 3
        assert not occupied[y:b, x:e].any()
        assert b <= y0 or y >= y1 or e <= x0 or x >= x1
        region = inp[y:b, x:e]
        assert np.all((region == color) | (region == bg))
        mask = region == color
        assert _span(mask) and 0 < mask.sum() < mask.size
        patterns[color] = mask
        registers[color] = (y, x, b, e)
    assert tile_colors <= set(patterns)
    for y, x, b, e, color in tiles:
        assert patterns[color].shape == (b-y-2, e-x-2)
    groups = defaultdict(list)
    for y, x, color in swatches:
        groups[y].append((y, x, color))
    legends = [sorted(g, key=lambda p: p[1]) for g in groups.values() if 2 <= len(g) <= 6]
    assert len(legends) == 1
    legend = legends[0]
    assert len(swatches) == len(legend)+1
    marker = next(p for p in swatches if p not in legend)
    word = tuple(p[2] for p in legend)
    assert word[0] == first and len(set(word)) >= 2
    assert marker[2] not in word and marker[2] not in patterns
    assert set(word).isdisjoint(patterns)
    assert all(q[1]-p[1] >= 3 for p, q in zip(legend, legend[1:]))
    assert max(y+2 for y, x, color in swatches) < min(t[0] for t in tiles)
    for y, x, color in swatches:
        assert y >= 1 and x >= 1 and y+2 < inp.shape[0] and x+2 < inp.shape[1]
        ring = inp[y-1:y+3, x-1:x+3].copy()
        ring[1:3, 1:3] = bg
        assert np.all(ring == bg)
    # Require separated stencil boxes (their own disconnected pixels still form one register).
    boxes = list(registers.values())
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]
    shape = (y1-y0, x1-x0)
    base = np.full(shape, bg, np.int8)
    local_tiles = []
    for y, x, b, e, color in tiles:
        y, x, b, e = y-y0, x-x0, b-y0, e-x0
        base[y:b, x:e] = color
        local_tiles.append((y, x, b, e, color))
    return dict(bg=bg, base=base, tiles=tuple(local_tiles), patterns=patterns,
                registers=registers, word=word, marker=marker[2],
                start=(start[0]-y0, start[1]-x0), crop=(y0, x0, y1, x1))


def _decode(inp):
    candidates = []
    for code in range(8):
        try:
            info = _decode_canonical(_unorient(inp, code))
        except (AssertionError, IndexError, StopIteration):
            continue
        info['orientation'] = code
        candidates.append(info)
    assert len(candidates) == 1, "input has no unique geometric interpretation"
    return candidates[0]


def _holes(info, variant='correct'):
    holes = np.zeros(info['base'].shape, bool)
    for y, x, b, e, color in info['tiles']:
        m = info['patterns'][color]
        if variant == 'ignore_stencils':
            m = np.zeros(m.shape, bool)
        elif variant == 'copy_stencil':
            m = ~m
        elif variant.startswith('mask_'):
            code = int(variant.split('_')[1])
            transformed = _orient(m, code)
            if transformed.shape == m.shape:
                m = transformed
        elif variant == 'wrong_match':
            alternatives = [mask for c, mask in info['patterns'].items() if c != color and mask.shape == m.shape]
            if alternatives:
                m = alternatives[0]
        holes[y+1:b-1, x+1:e-1] = m
    return holes


def _render(info, variant='correct', details=False):
    base, bg, word = info['base'], info['bg'], info['word']
    if variant == 'dedup':
        word = tuple(dict.fromkeys(word))
    holes = _holes(info, variant)
    out = base.copy()
    out[holes] = bg
    record = _trace((base != bg) & ~holes, info['start'])
    if variant == 'ignore_route':
        return (out, record) if details else out
    ids = np.full(base.shape, -1, int)
    components = _components(holes, diagonal=variant == 'diagonal_spill')
    for i, cells in enumerate(components):
        for p in cells:
            ids[p] = i
    phases = list(range(len(record['path'])))
    if variant in ('reset_bends', 'consume_markers'):
        phases = []
        offset = 0
        for j, length in enumerate(record['lengths']):
            phases += [i if variant == 'reset_bends' else offset+i+j for i in range(length)]
            offset += length
    elif variant == 'reset_seams':
        phases, current, phase = [], None, 0
        for p in record['path']:
            color = int(base[p])
            if color != current:
                phase, current = 0, color
            phases.append(phase)
            phase += 1
    tip_colors = [word[p % len(word)] for p in phases]
    contacted, contacts = set(), []
    for point, endpoint_index in record['events']:
        i = int(ids[point])
        if i >= 0:
            color = (word[(phases[endpoint_index]+1) % len(word)]
                     if variant == 'next_spill' else tip_colors[endpoint_index])
            contacts.append((i, color, point))
            if variant != 'no_spill' and (i not in contacted or variant == 'last_spill'):
                for cell in components[i]:
                    out[cell] = color
            contacted.add(i)
    for point, color in zip(record['path'], tip_colors):
        out[point] = color
    for point, _ in record['events']:
        out[point] = info['marker']
    record.update(holes=holes, components=components, contacts=tuple(contacts), phases=tuple(phases))
    return (out, record) if details else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(_grid(input_grid))
    return _orient(_render(info), info['orientation'])


_ALTERNATIVES = ('ignore_stencils', 'ignore_route', 'copy_stencil', 'no_spill',
                 'dedup', 'reset_bends', 'consume_markers', 'reset_seams',
                 'last_spill', 'wrong_match', 'diagonal_spill', 'next_spill') + tuple('mask_'+str(i) for i in range(1, 8))

_REQUIRED = set(_ALTERNATIVES) | {'cycling', 'relay_spill'}


def _evidence(info):
    correct, record = _render(info, details=True)
    evidence = {v for v in _ALTERNATIVES if not np.array_equal(correct, _render(info, v))}
    if len(record['path']) >= 2*len(info['word']):
        evidence.add('cycling')
    start_color = int(info['base'][info['start']])
    if any(int(info['base'][point]) != start_color for i, color, point in record['contacts']):
        evidence.add('relay_spill')
    return evidence


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _pattern(rng, h, w):
    for _ in range(100):
        m = np.zeros((h, w), bool)
        kind = int(rng.integers(8))
        if kind < 4:
            m = rng.random((h, w)) < float(rng.uniform(.16, .56))
        elif kind == 4:
            m[int(rng.integers(h)), :] = True
            m[:, int(rng.integers(w))] = True
        elif kind == 5:
            m[0, 0] = m[-1, -1] = True
            for j in range(min(h, w)):
                m[j, j] = True
        elif kind == 6:
            m[0] = m[-1] = True
            m[:, int(rng.integers(w))] = True
        else:
            m[:] = True
            m[int(rng.integers(h)), int(rng.integers(w))] = False
        if rng.random() < .5:
            m = m[:, ::-1].copy()
        if rng.random() < .5:
            m = m[::-1].copy()
        # Expose the full register even for disconnected ink.
        for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
            if not edge.any():
                edge[int(rng.integers(len(edge)))] = True
        parts = _components(m)
        solid_large = any(min(_box(p)[2]-_box(p)[0], _box(p)[3]-_box(p)[1]) >= 5 and len(p) == (_box(p)[2]-_box(p)[0])*(_box(p)[3]-_box(p)[1]) for p in parts)
        swatch_only = all(len(p) == 4 and (_box(p)[2]-_box(p)[0], _box(p)[3]-_box(p)[1]) == (2, 2) for p in parts)
        if not m.all() and not solid_large and not swatch_only:
            return m
    raise AssertionError("failed to sample registered mask")


def _layout(rng, count, kind, equal=False, large=False):
    sizes = [(int(rng.integers(5, 10 if not large else 11)), int(rng.integers(5, 10 if not large else 11))) for _ in range(count)]
    if equal:
        side = int(rng.integers(7, 10))
        sizes = [(side, side)]*count
    positions = []
    if kind == 'row':
        x = 0
        offsets = [0]*count if equal else list(map(int, rng.integers(0, 4, count)))
        for (h, w), y in zip(sizes, offsets):
            positions.append((y, x, h, w))
            x += w
    elif kind == 'column':
        # Stored first-to-last from bottom to top, so the start has room to climb.
        y = sum(h for h, w in sizes)
        offsets = [0]*count if equal else list(map(int, rng.integers(0, 4, count)))
        for (h, w), x in zip(sizes, offsets):
            y -= h
            positions.append((y, x, h, w))
    else:
        assert count == 3
        a, b, c = sizes
        if rng.random() < .5:
            positions = [(c[0]+max(0, b[0]-a[0]), 0, *a), (c[0], a[1], *b), (0, a[1], *c)]
        else:
            positions = [(b[0], 0, *a), (0, 0, *b), (b[0]-c[0], b[1], *c)]
    y0, x0 = min(p[0] for p in positions), min(p[1] for p in positions)
    positions = [(y-y0, x-x0, h, w) for y, x, h, w in positions]
    return positions


def _word(rng, repeated=False):
    n = int(rng.integers(3 if repeated else 2, 6))
    d = int(rng.integers(2, min(4, n if not repeated else n-1)+1))
    word = list(range(d)) + list(map(int, rng.integers(0, d, n-d)))
    rng.shuffle(word)
    return tuple(word)


def _bank(rng, shapes):
    """Shelf layouts have independent order, variable gaps and varied rows."""
    order = list(map(int, rng.permutation(len(shapes))))
    options = []
    for ncols in range(1, len(shapes)+1):
        gap = int(rng.integers(1, 4))
        slots, y, width = {}, 0, 0
        for i in range(0, len(order), ncols):
            row = order[i:i+ncols]
            height = max(shapes[j][0] for j in row)
            x = 0
            for j in row:
                h, w = shapes[j]
                slots[j] = (y+int(rng.integers(height-h+1)), x)
                x += w+gap
            width = max(width, x-gap)
            y += height+gap
        options.append((y-gap, width, slots))
    return options


def _candidate(rng, count, kind, orientation, witness=False, test=False, opening=False):
    positions = _layout(rng, count, kind, equal=witness, large=test)
    bh = max(y+h for y, x, h, w in positions)
    bw = max(x+w for y, x, h, w in positions)
    if bh > 20 or bw > 27:
        return None
    abstract = _word(rng, repeated=witness)
    distinct = max(abstract)+1
    nd = 1 if witness else int(rng.integers(0, min(2, 8-count-distinct)+1))
    palette = list(map(int, rng.permutation(10)))
    bg, marker = palette[:2]
    tile_colors = palette[2:2+count]
    program = palette[2+count:2+count+distinct]
    extras = palette[2+count+distinct:2+count+distinct+nd]
    word = tuple(program[i] for i in abstract)
    masks = [_pattern(rng, h-2, w-2) for y, x, h, w in positions]
    shapes = [m.shape for m in masks]
    for _ in extras:
        shape = shapes[0] if witness else _pick(rng, shapes)
        masks.append(_pattern(rng, *shape))
        shapes.append(shape)
    all_colors = tile_colors+extras
    # A 7-row header reserves a grouped program and one separate marker swatch.
    gaps = list(map(int, rng.integers(1, 3, len(word)-1)))
    legend_x = [1]
    for gap in gaps:
        legend_x.append(legend_x[-1]+2+gap)
    legend_width = legend_x[-1]+3
    layouts = []
    for sh, sw, slots in _bank(rng, shapes):
        for mode in ('side', 'below'):
            h = 8+max(bh, sh)+1 if mode == 'side' else 8+bh+2+sh+1
            w = max(legend_width, 1+bw+2+sw+1) if mode == 'side' else max(legend_width, 1+max(bw, sw)+1)
            if h <= 30 and w <= 30:
                layouts.append((h, w, slots, mode))
    if not layouts:
        return None
    h, w, slots, mode = _pick(rng, layouts)
    inp = np.full((h, w), bg, np.int8)
    ay, ax = 8, 1
    for (y, x, dh, dw), color in zip(positions, tile_colors):
        inp[ay+y:ay+y+dh, ax+x:ax+x+dw] = color
    first_y, first_x, first_h, first_w = positions[0]
    sr = first_y+int(rng.integers(1, first_h-1))
    inp[ay+sr, ax+first_x] = word[0]
    for x, color in zip(legend_x, word):
        inp[1:3, x:x+2] = color
    mx = int(rng.integers(1, w-2))
    inp[5:7, mx:mx+2] = marker
    by, bx = (8, 1+bw+2) if mode == 'side' else (8+bh+2, 1)
    for j, color in enumerate(all_colors):
        y, x = slots[j]
        dh, dw = masks[j].shape
        inp[by+y:by+y+dh, bx+x:bx+x+dw][masks[j]] = color
    # Extra whitespace varies independently; all padding stays within 30.
    padding = []
    for dim in inp.shape:
        total = int(rng.integers(min(3, 30-dim)+1))
        a = int(rng.integers(total+1))
        padding.append((a, total-a))
    inp = np.pad(inp, padding, constant_values=bg)
    try:
        info = _decode_canonical(inp)
    except AssertionError:
        return None
    info['orientation'] = orientation
    out, record = _render(info, details=True)
    if len(record['path']) < 4 or not record['contacts']:
        return None
    if test and (len({int(info['base'][p]) for p in record['path']}) < min(2, count)
                 or np.array_equal(out, _render(info, 'no_spill'))):
        return None
    if opening:
        start_color = int(info['base'][info['start']])
        if (len(record['path']) < 2*len(word)
                or not any(int(info['base'][p]) != start_color for i, color, p in record['contacts'])
                or np.array_equal(out, _render(info, 'reset_seams'))):
            return None
    return dict(input=_orient(inp, orientation), output=_orient(out, orientation)), info


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # scale/count growth, quarter-turn shift, changed assembly
    base_orientation = int(rng.integers(8))
    train_kind = _pick(rng, ('row', 'column'))
    puzzle, seen, evidence = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            required = set()
            witness = split == 'train' and index < 2
            if witness:
                # Two actual demonstrations together must reject every listed shortcut.
                required = _REQUIRED-evidence
                need_relay = bool({'reset_seams', 'relay_spill'} & required)
                ntiles, kind = (2 if index == 0 or need_relay else int(rng.integers(1, 3))), train_kind
            elif split == 'test':
                ntiles = 3 if mode == 0 else int(rng.integers(2, 4))
                kind = ('column' if train_kind == 'row' else 'row') if mode == 2 else _pick(rng, ('row', 'column', 'elbow') if ntiles == 3 else ('row', 'column'))
            else:
                ntiles = int(rng.integers(1, 3 if mode in (0, 2) else 4))
                kind = train_kind if mode == 2 else _pick(rng, ('row', 'column', 'elbow') if ntiles == 3 else ('row', 'column'))
            orientation = (base_orientation+1) % 4 + (4 if base_orientation >= 4 else 0) if split == 'test' and mode == 1 else base_orientation
            if mode != 1 and rng.random() < .45:
                orientation = int(rng.integers(8))
            best = None
            for attempt in range(1200 if witness else 300):
                candidate = _candidate(rng, ntiles, kind, orientation, witness=witness,
                                       test=split == 'test', opening=witness and index == 0)
                if candidate is None:
                    continue
                pair, info = candidate
                key = _key(pair['input'])
                if key in seen:
                    continue
                ev = _evidence(info) if witness else set()
                score = len(required & ev)
                if best is None or score > best[0]:
                    best = score, pair, info, ev
                opening_ok = index == 0 and score >= len(required)-2 and {'cycling', 'relay_spill', 'reset_seams'} <= ev
                if not witness or opening_ok or (index == 1 and required <= ev):
                    best = score, pair, info, ev
                    break
            assert best is not None, "no fitting scene"
            _, pair, info, ev = best
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            if split == 'train':
                evidence |= ev
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected, record = _render(info, details=True)
            assert np.array_equal(out, _orient(expected, info['orientation']))
            assert record['contacts'] and len(record['path']) >= 4
            assert not np.array_equal(inp, out)
            # Four-connectivity, preserved material, and subtractive polarity.
            assert all(info['base'][p] != info['bg'] and not record['holes'][p] for p in record['path'])
            changed = set(record['path']) | {p for p, _ in record['events']}
            touched = {i for i, color, p in record['contacts']}
            for i in touched:
                changed.update(record['components'][i])
            for p in np.ndindex(expected.shape):
                if p not in changed:
                    assert expected[p] == (info['bg'] if record['holes'][p] else info['base'][p])
            for p, endpoint in record['events']:
                assert expected[p] == info['marker'] and not ((info['base'] != info['bg']) & ~record['holes'])[p]
            first_dyes = {}
            collision_points = {p for p, _ in record['events']}
            for component_id, color, point in record['contacts']:
                first_dyes.setdefault(component_id, color)
            for component_id, cells in enumerate(record['components']):
                for point in cells:
                    if point not in collision_points:
                        assert expected[point] == first_dyes.get(component_id, info['bg'])
            if split == 'train':
                evidence |= _evidence(info)
    assert _REQUIRED <= evidence, "unexposed training alternatives: "+str(_REQUIRED-evidence)


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}) == 42
    infos = [_decode(pair['input']) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert {len(i['tiles']) for i in infos} == {1, 2, 3}
    assert len({i['base'].shape for i in infos}) >= 20
    assert len({i['orientation'] for i in infos}) == 8
    assert {i['bg'] for i in infos} == set(range(10))
    assert len({tuple((t[2]-t[0], t[3]-t[1]) for t in i['tiles']) for i in infos}) >= 25
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(314159, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123, 2, 1)
    q = generate_puzzle(123, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    # Literal monotone walker fixtures pin continuous phase and edge termination.
    material = np.ones((4, 7), bool)
    r = _trace(material, (2, 0))
    assert r['path'] == tuple((2, c) for c in range(7)) and not r['events']
    material[2, 4:] = False
    r = _trace(material, (2, 0))
    assert r['path'] == tuple([(2, c) for c in range(4)]+[(1, 3), (0, 3)])
    assert r['events'] == (((2, 4), 3),)
    # A literal spill fixture: two hits, different tip phases, diagonal isolation.
    base = np.zeros((8, 9), np.int8)
    base[1:7, 1:8] = 4
    mask = np.zeros((4, 5), bool)
    for y, x in ((2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 5), (5, 4), (5, 6)):
        mask[y-2, x-2] = True
    fixture = dict(bg=0, base=base, tiles=((1, 1, 7, 8, 4),), patterns={4: mask},
                   word=(1, 1, 2), marker=7, start=(5, 1))
    expected = np.array([[int(c) for c in row] for row in (
        '000000000', '044444440', '042724440', '044124440',
        '044122440', '011274040', '044444440', '000000000')], np.int8)
    assert np.array_equal(_render(fixture), expected)
    assert not np.array_equal(_render(fixture, 'last_spill'), expected)
    assert not np.array_equal(_render(fixture, 'diagonal_spill'), expected)
    assert not np.array_equal(_render(fixture, 'next_spill'), expected)
    return dict(valid=True, seeds=42, distinct_puzzles=len(puzzles),
                input_shapes=len({_key(pair['input'])[0] for p in puzzles for s in p for pair in p[s]}),
                output_shapes=len({i['base'].shape for i in infos}),
                orientations=sorted({i['orientation'] for i in infos}),
                plate_counts=sorted({len(i['tiles']) for i in infos}),
                plate_size_sequences=len({tuple((t[2]-t[0], t[3]-t[1]) for t in i['tiles']) for i in infos}),
                backgrounds=sorted({i['bg'] for i in infos}),
                word_patterns=len({tuple(i['word'].index(c) for c in i['word']) for i in infos}),
                note='Generator-owned invariants and mutation checks; no independent solver.')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
