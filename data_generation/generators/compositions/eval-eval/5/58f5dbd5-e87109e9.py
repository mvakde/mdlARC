"""Porous turning mirrors: color-matched negative stencils route beam lanes.

The legend is an optical material description, not a persistent color table.
All interpretation is reconstructed from the input. Only stdlib and NumPy.
"""
from __future__ import annotations

import itertools
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _d4(mask):
    return [np.rot90(mask, i) for i in range(4)] + [
        np.rot90(np.fliplr(mask), i) for i in range(4)]


def _span(mask):
    return all((mask[0].any(), mask[-1].any(),
                mask[:, 0].any(), mask[:, -1].any()))


def _components(field, colors):
    seen, result = set(), []
    h, w = field.shape
    for rr, cc in np.argwhere(np.isin(field, list(colors))):
        r, c = int(rr), int(cc)
        if (r, c) in seen:
            continue
        color = int(field[r, c])
        cells, stack = [], [(r, c)]
        seen.add((r, c))
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in DIRS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and (yy, xx) not in seen
                        and field[yy, xx] == color):
                    seen.add((yy, xx))
                    stack.append((yy, xx))
        lo = np.min(cells, axis=0)
        hi = np.max(cells, axis=0) + 1
        y, x, b, e = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
        _require(len(cells) == (b - y) * (e - x), "input object is not solid")
        result.append((y, x, b - y, e - x, color))
    return result


def _decode(input_grid):
    """Read framed cards, registered masks, square mirrors and unkeyed source."""
    grid = _grid(input_grid)
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "legend needs a uniform top frame")
    endings = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(endings) > 0, "legend needs a bottom frame")
    bottom = int(endings[0])
    s = bottom - 1
    _require(2 <= s <= 5 and bottom + 1 < len(grid), "invalid stencil scale or crop")
    band = grid[1:bottom]
    walls = np.all(band == frame, axis=0)
    _require(walls[0] and walls[-1], "legend needs side walls")
    runs, x = [], 0
    while x < grid.shape[1]:
        if walls[x]:
            x += 1
            continue
        start = x
        while x < grid.shape[1] and not walls[x]:
            x += 1
        runs.append((start, x))
    _require(1 <= len(runs) <= 3 and all(b - a >= s + 4 for a, b in runs),
             "invalid card count or width")
    bg = int(band[0, runs[0][0] + 1])
    _require(bg != frame, "frame and background must differ")
    masks, turns, card_order = {}, {}, []
    for a, b in runs:
        _require(np.all(band[:, a + 1] == bg) and np.all(band[:, b - 2] == bg),
                 "missing empty gap beside side bar")
        left, right = band[:, a], band[:, b - 1]
        is_left = np.all(left == left[0]) and int(left[0]) != bg
        is_right = np.all(right == right[0]) and int(right[0]) != bg
        _require(bool(is_left) != bool(is_right), "ambiguous turn bar")
        color = int(left[0] if is_left else right[0])
        _require(np.all((right if is_left else left) == bg), "two turn bars")
        _require(color not in (bg, frame) and color not in masks, "conflicting card")
        center = band[:, a + 1:b - 1]
        _require(np.all((center == bg) | (center == color)), "foreign stencil color")
        cells = np.argwhere(center == color)
        _require(len(cells) > 0, "empty invisible stencil")
        lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
        _require(tuple(hi - lo) == (s, s), "stencil must span its square register")
        mask = center[lo[0]:hi[0], lo[1]:hi[1]] == color
        _require(_span(mask) and 0 < mask.sum() < s * s, "stencil must mix holes and material")
        masks[color] = mask.copy()
        turns[color] = -1 if is_left else 1
        card_order.append(color)
    field = grid[bottom + 1:].copy()
    _require(not np.any(field == frame), "frame color is reserved for legend")
    other = set(map(int, np.unique(field))) - set(masks) - {bg}
    _require(len(other) == 1, "field needs one unkeyed source color")
    source = other.pop()
    sources = _components(field, {source})
    _require(len(sources) == 1, "source must be unique")
    r, c, kh, kw, _ = sources[0]
    _require(kh == kw and 1 <= kh <= 4, "source must be a 1..4 square")
    gates = _components(field, set(masks))
    _require(1 <= len(gates) <= 8, "field needs 1..8 mirrors")
    _require(all(h == w == s for _, _, h, w, _ in gates), "mirror and stencil scales differ")
    boxes = gates + sources
    for i, (y, x, hh, ww, _) in enumerate(boxes):
        for yy, xx, hhh, www, _ in boxes[i + 1:]:
            _require(y + hh < yy or yy + hhh < y or x + ww < xx or xx + www < x,
                     "input objects need at least one empty separating row or column")
    return dict(field=field, bg=bg, frame=frame, source=source,
                s=s, k=kh, position=(r, c), masks=masks, turns=turns,
                gates=gates, cards=tuple(card_order), crop=bottom + 1,
                panel_widths=tuple(b - a + 2 for a, b in runs))


def _trace(info, orientation=0, polarity=False, solid=False, turn_map=None,
           stop_at_material=False, mask_map=None):
    """Trace static material, separately from accumulated paint.

    Each state has one successor. Revisiting a state ends that lane's trace:
    the union has already reached its fixed point, even on a closed orbit.
    Nondefault arguments are explicit counterfactual checks, never solver state.
    """
    field, bg, source = info['field'], info['bg'], info['source']
    material = field.copy()
    gate_id = np.full(field.shape, -1, dtype=int)
    holes = np.zeros(field.shape, dtype=bool)
    for index, (y, x, s, _, color) in enumerate(info['gates']):
        gate_id[y:y + s, x:x + s] = index
        mask = info['masks'][color if mask_map is None else mask_map[color]]
        mask = _d4(mask)[orientation]
        if polarity:
            mask = ~mask
        if not solid:
            material[y:y + s, x:x + s][mask] = bg
            holes[y:y + s, x:x + s] = mask
    out = material.copy()
    paths = np.zeros(field.shape, dtype=bool)
    events, chains, cycles = [], [], 0
    r, c = info['position']
    k, (h, w) = info['k'], field.shape
    starts = ([(r, c + j, 0) for j in range(k)]
              + [(r + j, c + k - 1, 1) for j in range(k)]
              + [(r + k - 1, c + j, 2) for j in range(k)]
              + [(r + j, c, 3) for j in range(k)])
    for lane, (y, x, d) in enumerate(starts):
        seen, chain = set(), []
        while (y, x, d) not in seen:
            seen.add((y, x, d))
            dy, dx = DIRS[d]
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w):
                break
            color = int(material[yy, xx])
            if color in info['turns']:
                if stop_at_material:
                    break
                side = info['turns'][color]
                delta = side if turn_map is None else turn_map[side]
                events.append((color, d, y, x, int(gate_id[yy, xx]), lane))
                chain.append(int(gate_id[yy, xx]))
                d = (d + delta) % 4
            else:
                _require(color in (bg, source), "unkeyed optical material")
                y, x = yy, xx
                paths[y, x] = True
                if color == bg:
                    out[y, x] = source
        else:
            cycles += 1
        chains.append(tuple(chain))
    return out, dict(paths=paths, holes=holes, material=material, events=events,
                     chains=chains, cycles=cycles, gate_id=gate_id,
                     max_turns=max(map(len, chains), default=0))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only optical negative + simultaneous union of all beam lanes."""
    return _trace(_decode(input_grid))[0]


def _mask(rng, s):
    for _ in range(120):
        kind = int(rng.integers(8))
        m = np.zeros((s, s), dtype=bool)
        if kind == 0:
            np.fill_diagonal(m, True)
        elif kind == 1:
            m[0] = m[-1] = True
        elif kind == 2:
            m[int(rng.integers(s))] = True
            m[:, int(rng.integers(s))] = True
        elif kind == 3:
            m[0] = True
            m[:, 0] = True
        elif kind == 4:
            m[:] = True
            m.flat[int(rng.integers(s * s))] = False
        elif kind == 5:
            m[0, 0] = m[-1, -1] = True
            if rng.random() < .6:
                m.flat[int(rng.integers(s * s))] = True
        else:
            m = rng.random((s, s)) < float(rng.uniform(.2, .8))
        m = _d4(m)[int(rng.integers(8))].copy()
        if _span(m) and 0 < m.sum() < s * s:
            return m
    raise RuntimeError("bounded mask sampler exhausted")


def _header(rng, width, s, masks, turns):
    colors = list(masks)
    widths = np.full(len(colors), s + 6, dtype=int)
    widths += rng.multinomial(width - int(widths.sum()), np.ones(len(colors)) / len(colors))
    header = np.zeros((s + 2, width), dtype=np.int8)
    header[[0, -1]] = 9
    start = 0
    for color, width_i in zip(rng.permutation(colors), widths):
        color, size = int(color), int(width_i)
        header[:, start] = header[:, start + size - 1] = 9
        bar = start + 1 if turns[color] == -1 else start + size - 2
        header[1:-1, bar] = color
        stencil_x = start + 3 + int(rng.integers(size - s - 5))
        header[1:-1, stencil_x:stencil_x + s][masks[color]] = color
        start += size
    return header


def _clear(field, y, x, s):
    h, w = field.shape
    return (0 <= y <= h - s and 0 <= x <= w - s
            and not np.any(field[max(0, y - 1):min(h, y + s + 1),
                                 max(0, x - 1):min(w, x + s + 1)]))


def _make_pair(rng, s, k, target, teaching=False, wide=False):
    max_keys = min(3, 30 // (s + 6))
    nkeys = int(rng.integers(2 if teaching else 1, max_keys + 1))
    colors = list(range(2, 2 + nkeys))
    masks = {color: _mask(rng, s) for color in colors}
    turns = {color: int(rng.choice((-1, 1))) for color in colors}
    if teaching:
        turns[colors[0]], turns[colors[1]] = -1, 1
        if rng.random() < .5:
            turns = {color: -side for color, side in turns.items()}
    low_h = max(11, 2 * s + k + 2)
    low_w = max(nkeys * (s + 6), 2 * s + k + 2, 15)
    h = int(rng.integers(max(low_h, 20) if wide else low_h, 29 - s))
    w = int(rng.integers(max(low_w, 25) if wide else low_w, 31))
    field = np.zeros((h, w), dtype=np.int8)
    edge = not teaching and rng.random() < .2
    r = int(rng.integers(0 if edge else 1, h - k + (1 if edge else 0)))
    c = int(rng.integers(0 if edge else 1, w - k + (1 if edge else 0)))
    field[r:r + k, c:c + k] = 1
    gates = []
    info = dict(field=field, bg=0, source=1, masks=masks, turns=turns,
                gates=gates, k=k, position=(r, c))
    color_order = [int(q) for q in rng.permutation(colors)]
    for i in range(target):
        _, trace = _trace(info)
        candidates = np.argwhere(trace['paths'] & (field == 0))
        color = color_order[i] if i < nkeys else int(rng.choice(colors))
        for _ in range(100):
            if len(candidates) and rng.random() < .83:
                yy, xx = candidates[int(rng.integers(len(candidates)))]
                y, x = int(yy) - int(rng.integers(s)), int(xx) - int(rng.integers(s))
            else:
                y, x = int(rng.integers(h - s + 1)), int(rng.integers(w - s + 1))
            if not _clear(field, y, x, s):
                continue
            field[y:y + s, x:x + s] = color
            gates.append((y, x, s, s, color))
            break
    if not gates:
        return None
    output = _trace(info)[0]
    header = _header(rng, w, s, masks, turns)
    palette = rng.permutation(10)
    return dict(input=palette[np.vstack((header, field))].astype(np.int8),
                output=palette[output].astype(np.int8))


def _evidence(infos, detailed=True):
    """Observable teaching checks, including route effects away from mirrors."""
    headings = {-1: set(), 1: set()}
    hole_entry = interaction = turning_effect = recursive = False
    for info in infos:
        _, t = _trace(info)
        _, no_a = _trace(info, solid=True)
        _, no_b = _trace(info, stop_at_material=True)
        external = info['field'] == info['bg']
        interaction |= np.count_nonzero((t['paths'] ^ no_a['paths']) & external) >= 3
        turning_effect |= np.count_nonzero((t['paths'] ^ no_b['paths']) & external) >= 3
        hole_entry |= bool(np.any(t['paths'] & t['holes']))
        recursive |= any(len(set(chain)) >= 2 for chain in t['chains'])
        for color, d, *_ in t['events']:
            headings[info['turns'][color]].add(d)
    basic = (all(len(v) >= 2 for v in headings.values()) and hole_entry
             and interaction and turning_effect and recursive)
    if not detailed or not basic:
        return basic
    outputs = [_trace(q)[0] for q in infos]
    turn_candidates = []
    for left, right in itertools.product((-1, 1), repeat=2):
        table = {-1: left, 1: right}
        if all(np.array_equal(_trace(q, turn_map=table)[0], out)
               for q, out in zip(infos, outputs)):
            turn_candidates.append((left, right))
    mask_candidates = []
    for orientation, polarity in itertools.product(range(8), (False, True)):
        if all(np.array_equal(_trace(q, orientation, polarity)[0], out)
               for q, out in zip(infos, outputs)):
            mask_candidates.append((orientation, polarity))
    selection = False
    for q, out in zip(infos, outputs):
        colors = list(q['masks'])
        if len(colors) >= 2:
            swapped = {color: color for color in colors}
            swapped[colors[0]], swapped[colors[1]] = colors[1], colors[0]
            selection |= not np.array_equal(_trace(q, mask_map=swapped)[0], out)
    return basic and turn_candidates == [(-1, 1)] and mask_candidates == [(0, False)] and selection


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproduce the complete puzzle with one default_rng(seed)."""
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(('source_scale', 'stencil_scale', 'route_extension')))
    ks = [1, 2] if rng.random() < .6 else [2, 3]
    ss = [2, 3] if shift == 'stencil_scale' else sorted(
        map(int, rng.choice((2, 3, 4, 5), 2, replace=False)))
    rng.shuffle(ks)
    rng.shuffle(ss)
    puzzle = {'train': [], 'test': []}
    infos, seen = [], set()
    # Build two complementary, random lessons. No fixed scene is a fallback.
    for index in range(nt):
        s, k = ss[index % 2], ks[index % 2]
        accepted = None
        for attempt in range(240):
            target = int(rng.integers(3, 5)) if index < 2 else int(rng.integers(1, 5))
            pair = _make_pair(rng, s, k, target, teaching=index < 2)
            if pair is None or _key(pair['input']) in seen:
                continue
            q = _decode(pair['input'])
            if index == 0:
                # Start with a route-sensitive example; other evidence may be
                # supplied by the complementary lesson.
                _, t = _trace(q)
                _, no_a = _trace(q, solid=True)
                if (not t['events'] or not np.any(t['paths'] & t['holes'])
                        or np.count_nonzero((t['paths'] ^ no_a['paths'])
                                           & (q['field'] == q['bg'])) < 3):
                    continue
            elif index == 1 and not _evidence(infos + [q]):
                continue
            accepted = (pair, q)
            break
        if accepted is None:
            raise RuntimeError("bounded teaching search exhausted")
        pair, q = accepted
        puzzle['train'].append(pair)
        infos.append(q)
        seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    for _ in range(ne):
        s = int(rng.choice(ss)) if shift != 'stencil_scale' else int(rng.choice((4, 5)))
        k = max(ks) + 1 if shift == 'source_scale' else int(rng.choice(ks))
        target = int(rng.integers(5, 8)) if shift == 'route_extension' else int(rng.integers(2, 7))
        for attempt in range(120):
            pair = _make_pair(rng, s, k, target, wide=shift == 'route_extension')
            if pair is None or _key(pair['input']) in seen:
                continue
            q = _decode(pair['input'])
            if shift == 'route_extension' and len(q['gates']) <= max(len(t['gates']) for t in infos):
                continue
            # Held-out cases actually use both mechanisms, including their
            # interaction, rather than merely having an unused legend.
            _, t = _trace(q)
            _, no_a = _trace(q, solid=True)
            _, no_b = _trace(q, stop_at_material=True)
            if t['cycles'] and not any(_trace(z)[1]['cycles'] for z in infos):
                continue
            if (not t['events'] or not np.any(t['paths'] & t['holes'])
                    or np.count_nonzero((t['paths'] ^ no_a['paths'])
                                       & (q['field'] == q['bg'])) < 3
                    or np.count_nonzero((t['paths'] ^ no_b['paths'])
                                       & (q['field'] == q['bg'])) < 3):
                continue
            puzzle['test'].append(pair)
            seen.add(_key(pair['input']))
            break
        else:
            raise RuntimeError("bounded held-out search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 'train count', 2, 10)
    _count(ne, 'test count', 1, 4)
    _require(nt > ne, "train count must exceed test count")
    seen, infos = set(), {'train': [], 'test': []}
    cycles = {'train': False, 'test': False}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            q = _decode(inp)
            expected, t = _trace(q)
            _require(np.array_equal(expected, out), "incorrect etching, ray union or crop")
            _require(out.shape == q['field'].shape, "wrong field crop")
            _require(np.all(out[t['material'] != q['bg']] == t['material'][t['material'] != q['bg']]),
                     "surviving mirror material or source changed")
            _require(np.all(np.isin(out, [q['bg'], q['source']] + list(q['masks']))),
                     "legend leaked into field")
            _require(np.all(out[t['paths'] & (t['material'] == q['bg'])] == q['source']),
                     "traversed hole or background was not painted")
            if split == 'test':
                _, no_a = _trace(q, solid=True)
                _, no_b = _trace(q, stop_at_material=True)
                _require(bool(t['events']) and np.any(t['paths'] & t['holes'])
                         and np.count_nonzero((t['paths'] ^ no_a['paths'])
                                              & (q['field'] == q['bg'])) >= 3
                         and np.count_nonzero((t['paths'] ^ no_b['paths'])
                                              & (q['field'] == q['bg'])) >= 3,
                         "test does not exercise porous keyed propagation")
            cycles[split] |= bool(t['cycles'])
            infos[split].append(q)
    _require(_evidence(infos['train']), "demonstrations leave important alternatives unresolved")
    _require(len({q['k'] for q in infos['train']}) >= 2, "source width is undemonstrated")
    _require(len({q['s'] for q in infos['train']}) >= 2, "stencil scale is undemonstrated")
    _require(not cycles['test'] or cycles['train'], "closed test orbit is undemonstrated")
    # Shift requirements concern the curriculum, never solve's interpretation.
    tr_k, tr_s = {q['k'] for q in infos['train']}, {q['s'] for q in infos['train']}
    tr_n = max(len(q['gates']) for q in infos['train'])
    _require(all(q['k'] > max(tr_k) or q['s'] > max(tr_s) or len(q['gates']) > tr_n
                 for q in infos['test']), "test lacks the promised structural shift")


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded 42-seed diversity, evidence, count and semantic mutation checks."""
    shapes, layouts, palettes, scales, counts, fingerprints = set(), set(), set(), set(), set(), set()
    cycles = repeated = unused = unhit = split_lanes = 0
    shifts = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(_fingerprint(p))
        train = [_decode(pair['input']) for pair in p['train']]
        for split in ('train', 'test'):
            for pair in p[split]:
                q = _decode(pair['input'])
                out, t = _trace(q)
                assert np.array_equal(solve(np.array(pair['input'])), out)
                shapes.add(q['field'].shape)
                layouts.add((q['position'], tuple(q['gates']), q['field'].shape))
                palettes.add((q['bg'], q['frame'], q['source']))
                scales.add((q['s'], q['k']))
                counts.add(len(q['gates']))
                cycles += bool(t['cycles'])
                gate_colors = [gate[-1] for gate in q['gates']]
                repeated += len(gate_colors) > len(set(gate_colors))
                unused += bool(set(q['masks']) - set(gate_colors))
                reached = {e[4] for e in t['events']} | set(map(int, t['gate_id'][t['paths'] & t['holes']]))
                unhit += len(reached) < len(q['gates'])
                # At least one launch face has unequal turn histories among its lanes.
                split_lanes += any(len(set(t['chains'][a:a + q['k']])) > 1
                                   for a in range(0, 4 * q['k'], q['k']))
                if split == 'test':
                    if q['k'] > max(z['k'] for z in train): shifts.add('source')
                    if q['s'] > max(z['s'] for z in train): shifts.add('stencil')
                    if len(q['gates']) > max(len(z['gates']) for z in train): shifts.add('routes')
    assert len(fingerprints) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert len(palettes) >= 100 and {2, 3, 4, 5} == {s for s, _ in scales}
    assert {1, 2, 3, 4} == {k for _, k in scales} and {1, 2, 3, 4, 5, 6} <= counts
    assert repeated >= 40 and split_lanes >= 35 and min(unused, unhit, cycles) >= 2
    assert shifts == {'source', 'stencil', 'routes'}
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 10 + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(123, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(123, 2, 1))
    # Serialization has no role in decoding; no ndarray-only validator assumption.
    lists = {split: [{role: a.tolist() for role, a in pair.items()} for pair in p[split]]
             for split in ('train', 'test')}
    validate_generated_puzzle(lists)
    for pair in p['train']:
        recolored = (pair['input'] + 7) % 10
        assert np.array_equal(solve(recolored), (pair['output'] + 7) % 10)
        q = _decode(pair['input'])
        cuts = np.cumsum((0,) + q['panel_widths'])
        header = pair['input'][:q['crop']]
        cards = [header[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        reordered = np.vstack((np.hstack(cards[::-1]), q['field']))
        assert np.array_equal(solve(reordered), pair['output'])
    changed = {split: [{role: a.copy() for role, a in pair.items()} for pair in p[split]]
               for split in ('train', 'test')}
    changed['test'][0]['output'][0, 0] = (int(changed['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    _fixture_test()


def _fixture_test():
    """Hand-calculated porous-mirror rays, independent of scene sampling."""
    # A 2x2 source sends two east lanes toward a 3x3 mirror. The upper
    # lane passes through a full hole row; the lower hits material and
    # turns up immediately outside. The diagonal holes expose alignment.
    field = np.zeros((9, 12), dtype=np.int8)
    field[3:5, 1:3] = 1
    field[2:5, 6:9] = 2
    mask = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool)
    info = dict(field=field, bg=0, source=1, k=2, position=(3, 1),
                gates=[(2, 6, 3, 3, 2)], masks={2: mask}, turns={2: -1})
    out, t = _trace(info)
    expected = np.array([[int(c) for c in row] for row in (
        '011001000000', '011001000000', '011001022000',
        '111111111111', '111111220000', '011000000000',
        '011000000000', '011000000000', '011000000000')], dtype=np.int8)
    assert np.array_equal(out, expected)
    assert np.all(out[3, 3:] == 1)  # transparent row, beyond the plate
    assert np.all(out[:5, 5] == 1)  # lower lane turns north at (4,5)
    assert out[4, 6] == 2 and out[2, 7] == 2  # material remains
    assert out[4, 8] == 0  # a carved, untraversed hole remains background
    _, solid = _trace(info, solid=True)
    assert not solid['paths'][3, 10] and t['paths'][3, 10]
    stopped, _ = _trace(info, stop_at_material=True)
    assert stopped[0, 5] == 0 and out[0, 5] == 1
    # Rotate the entire physical fixture and its registered stencil; left
    # remains relative left. Reflecting the same scene swaps left and right.
    for rotation in range(4):
        rotated = np.rot90(field, rotation)
        legend = _header(np.random.default_rng(19), rotated.shape[1], 3,
                         {2: np.rot90(mask, rotation)}, {2: -1})
        assert np.array_equal(solve(np.vstack((legend, rotated))), np.rot90(expected, rotation))
    reflected = np.fliplr(field)
    legend = _header(np.random.default_rng(19), reflected.shape[1], 3,
                     {2: np.fliplr(mask)}, {2: 1})
    assert np.array_equal(solve(np.vstack((legend, reflected))), np.fliplr(expected))
    # A legal four-mirror orbit is entered from the west. All mirrors have
    # cross-shaped holes (four material corners) and the same right key.
    orbit = np.zeros((14, 15), dtype=np.int8)
    orbit[5, 1] = 1
    for y, x in ((3, 10), (10, 7), (9, 2), (2, 3)):
        orbit[y:y + 3, x:x + 3] = 2
    cross = np.zeros((3, 3), dtype=bool)
    cross[1, :] = cross[:, 1] = True
    legend = _header(np.random.default_rng(23), 15, 3, {2: cross}, {2: 1})
    q = _decode(np.vstack((legend, orbit)))
    orbit_out, cycle = _trace(q)
    assert cycle['cycles'] == 1 and cycle['max_turns'] == 4
    assert np.all(orbit_out[5, 1:10] == 1)
    assert np.all(orbit_out[5:10, 9] == 1)
    assert np.all(orbit_out[9, 5:10] == 1)
    assert np.all(orbit_out[5:10, 5] == 1)
    assert orbit_out[13, 9] == 0  # no false exit from the closed lane
    assert np.array_equal(solve(np.vstack((legend, orbit))), orbit_out)
