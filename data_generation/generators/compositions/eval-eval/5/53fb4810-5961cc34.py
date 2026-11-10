"""Periodic signals choose a route by matching relay keys at arrival.

Only Python's standard library and NumPy are used. The construction painter
uses sampled geometry; the solver recovers geometry and clocks from input cells.
"""
from __future__ import annotations

import itertools
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _step(p, direction, distance=1):
    dr, dc = DIRS[direction]
    return p[0] + distance * dr, p[1] + distance * dc


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, group = [first], {first}
        while stack:
            p = stack.pop()
            for d in range(4):
                q = _step(p, d)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.add(q)
        result.append(group)
    return result


def _dilate(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows), max(cols)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _decode(value):
    g = _grid(value)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > g.size // 2
    h, w = g.shape
    assert all(int(g[p]) == bg for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))
    comps = _components(map(tuple, np.argwhere(g != bg)))
    boundary = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    roots = [c for c in comps if any(boundary(p) for p in c)]
    assert len(roots) == 1, "one boundary emitter is required"
    root = roots[0]
    starts = [p for p in root if boundary(p)]
    assert len(starts) == 1
    start = starts[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h-1 else 1 if start[1] == 0 else 3
    source = [_step(start, direction, k) for k in range(len(root))]
    assert set(source) == root and 2 <= len(source) <= 6
    marker = int(g[start])
    word = [int(g[p]) for p in source[1:]]
    payload = set(word)
    assert marker not in payload and bg not in payload and marker != bg
    objects = []
    for comp in comps:
        if comp == root:
            continue
        palette, amounts = np.unique([g[p] for p in comp], return_counts=True)
        carrier = int(palette[int(np.argmax(amounts))])
        assert carrier not in payload | {bg, marker}
        ports = [p for p in comp if int(g[p]) == marker]
        keys = [p for p in comp if int(g[p]) in payload]
        assert len(ports) == len(keys) == 1
        assert all(int(g[p]) in {carrier, marker} | payload for p in comp)
        assert sum(int(g[p]) == carrier for p in comp) > 2
        t, l, b, r = _bbox(comp)
        assert 0 < t < b < h-1 and 0 < l < r < w-1
        assert b-t+1 >= 3 and r-l+1 >= 3
        port, key = ports[0], keys[0]
        assert t < key[0] < b and l < key[1] < r
        sides = [d for d, ok in enumerate((port[0] == t, port[1] == r,
                                           port[0] == b, port[1] == l)) if ok]
        assert len(sides) == 1, "exit must have one outward normal"
        outgoing = sides[0]
        assert _step(port, (outgoing+2) % 4) in comp
        objects.append(dict(cells=comp, port=port, key=int(g[key]),
                            key_cell=key, out_dir=outgoing, carrier=carrier,
                            box=(t, l, b, r)))
    assert all(not (_dilate(a) & b) for a, b in itertools.combinations(comps, 2))
    return dict(grid=g, bg=bg, marker=marker, word=word, source=source,
                initial=direction, objects=objects)


def _trace(scene, *, constant=False, ignore_relays=False,
           unconditional=False, restart=False, strict=True):
    """Input-decoded advancing clock. Ablations are validator diagnostics."""
    g, word, objects = scene['grid'], scene['word'], scene['objects']
    out = np.full_like(g, scene['bg'])
    source = scene['source']
    out[source[0]] = word[-1]
    for p, color in zip(source[1:], word):
        out[p] = color
    owner = {p: i for i, o in enumerate(objects) for p in o['cells']}
    used_cells = set(source)
    ray_cells, activated, rejected, encounters, emissions = set(), [], [], [], []
    seen, encounter_by_index = set(), {}
    direction, phase = scene['initial'], 0
    front = _step(source[-1], direction)
    gap = 0
    for _ in range(g.size * (len(objects)+2)):
        if not _inside(front, g.shape):
            facts = dict(scene=scene, activated=activated, rejected=rejected,
                         encounters=encounters, emissions=emissions,
                         ray_cells=ray_cells, trace_cells=used_cells,
                         decoys=len(objects)-len(seen))
            return out, facts
        index = owner.get(front)
        if strict:
            assert front not in used_cells, "route crosses itself or revisits a footprint"
            if index in seen:
                # A cutout does not end passage through a rejected footprint.
                previous = encounter_by_index[index]
                dr, dc = DIRS[previous['incoming']]
                delta = (front[0]-previous['point'][0], front[1]-previous['point'][1])
                assert not previous['active'] and direction == previous['incoming']
                assert delta[0]*dc-delta[1]*dr == 0 and delta[0]*dr+delta[1]*dc > 0
        color = word[0] if constant else word[phase]
        if index is not None and index not in seen and not ignore_relays:
            obj = objects[index]
            seen.add(index)
            fire = unconditional or color == obj['key']
            record = dict(index=index, phase=phase, color=color, key=obj['key'],
                          point=front, incoming=direction, gap=gap,
                          outgoing=obj['out_dir'], active=fire)
            encounters.append(record)
            encounter_by_index[index] = record
            gap = 0
            if fire:
                if strict:
                    assert obj['out_dir'] != (direction+2) % 4
                    assert not (obj['cells'] & used_cells)
                activated.append(index)
                for p in obj['cells']:
                    out[p] = color
                used_cells.update(obj['cells'])
                direction = obj['out_dir']
                phase = 0 if restart else (phase+1) % len(word)
                front = _step(obj['port'], direction)
                emissions.append((record, front, phase))
                continue
            rejected.append(index)
        out[front] = color
        ray_cells.add(front)
        used_cells.add(front)
        phase = (phase+1) % len(word)
        front = _step(front, direction)
        gap += 1
    raise AssertionError("route did not terminate")


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every color, gate decision, phase and exit from this input only."""
    return _trace(_decode(input_grid))[0]


def _mask(rng, maximum=5, kind=None, large=False):
    h, w = map(int, rng.integers(3, maximum+1, size=2))
    if large:
        if rng.integers(2):
            h = int(rng.integers(7, 9))
        else:
            w = int(rng.integers(7, 9))
    kind = int(rng.integers(4)) if kind is None else kind
    cells = {(r, c) for r in range(h) for c in range(w)}
    if kind == 1:
        cells -= {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
    elif kind == 2:
        rh, cw = (h-1)//2, (w-1)//2
        rows = {rh} if h % 2 else {rh, rh+1}
        cols = {cw} if w % 2 else {cw, cw+1}
        cells = {(r, c) for r in range(h) for c in range(w) if r in rows or c in cols}
    elif kind == 3:
        # A single edge notch. It preserves full axes and a connected footprint.
        cells.remove((0, int(rng.integers(1, w-1))))
    return cells, h, w


def _port_options(cells, h, w, direction):
    return [p for p in sorted(cells)
            if ((direction == 0 and p[0] == 0 and 0 < p[1] < w-1)
                or (direction == 2 and p[0] == h-1 and 0 < p[1] < w-1)
                or (direction == 1 and p[1] == w-1 and 0 < p[0] < h-1)
                or (direction == 3 and p[1] == 0 and 0 < p[0] < h-1))
            and _step(p, (direction+2) % 4) in cells]


def _translate(cells, t, l):
    return {(r+t, c+l) for r, c in cells}


def _source(shape, direction, length, rng):
    h, w = shape
    line = int(rng.integers(2, (w if direction % 2 == 0 else h)-2))
    start = ((h-1, line) if direction == 0 else (line, 0) if direction == 1
             else (0, line) if direction == 2 else (line, w-1))
    return [_step(start, direction, i) for i in range(length+1)]


def _ray(front, direction, shape):
    cells = []
    while _inside(front, shape):
        cells.append(front)
        front = _step(front, direction)
    return cells


def _choose_band(rng, choices):
    bands = [[x for x in choices if lo <= x[0] <= hi]
             for lo, hi in ((1, 3), (4, 8), (9, 25))]
    bands = [b for b in bands if b]
    band = bands[int(rng.integers(len(bands)))]
    return band[int(rng.integers(len(band)))]


def _try_layout(rng, spec, word):
    total = len(spec['events'])
    low = 12 if total <= 2 else 18 if total <= 3 else 23
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    source = _source(shape, spec['initial'], len(word), rng)
    occupied, trace = set(source), set(source)
    front, direction, phase = _step(source[-1], spec['initial']), spec['initial'], 0
    objects, active_index = [], 0
    for is_active in spec['events']:
        success = False
        for _ in range(55):
            turn = int(rng.choice([-1, 0, 1]))
            if spec['demo'] == 0 and is_active:
                turn = int(rng.choice([-1, 1]))
            elif spec['demo'] == 1 and is_active:
                turn = 0 if active_index == 0 else int(rng.choice([-1, 1]))
            outgoing = (direction+turn) % 4 if is_active else int(rng.integers(4))
            large = is_active and spec['large_index'] == active_index
            kind = int(rng.integers(1, 4)) if spec['demo'] == 1 and is_active and active_index == 0 else None
            mask, h, w = _mask(rng, spec['maximum'], kind, large)
            ports = _port_options(mask, h, w, outgoing)
            entries = _port_options(mask, h, w, (direction+2) % 4)
            keys = [p for p in sorted(mask) if 0 < p[0] < h-1 and 0 < p[1] < w-1]
            if not ports or not entries or not keys:
                continue
            port = ports[int(rng.integers(len(ports)))]
            entry = entries[int(rng.integers(len(entries)))]
            key_cell = keys[int(rng.integers(len(keys)))]
            choices = []
            for gap in range(1, 26):
                hit_phase = (phase+gap) % len(word)
                if spec['demo'] is not None and is_active and active_index == 0 and hit_phase != 0:
                    continue
                if spec['demo'] == 0 and not is_active and hit_phase != 1:
                    continue
                target = _step(front, direction, gap)
                t, l = target[0]-entry[0], target[1]-entry[1]
                if t < 1 or l < 1 or t+h >= shape[0] or l+w >= shape[1]:
                    continue
                cells = _translate(mask, t, l)
                approach = [_step(front, direction, k) for k in range(gap)]
                if _dilate(cells) & occupied or cells & trace:
                    continue
                if set(approach) & (occupied | trace):
                    continue
                depth = h if direction % 2 == 0 else w
                through = [_step(target, direction, k) for k in range(depth)]
                # Every constructed line-footprint intersection is contiguous.
                hits = [k for k, p in enumerate(through) if p in cells]
                if hits != list(range(min(hits), max(hits)+1)):
                    continue
                if not is_active and set(through) & (occupied | trace):
                    continue
                choices.append((gap, cells, (port[0]+t, port[1]+l),
                                (key_cell[0]+t, key_cell[1]+l), target,
                                approach, through, hit_phase))
            if not choices:
                continue
            gap, cells, port, key_cell, target, approach, through, hit_phase = _choose_band(rng, choices)
            color = word[hit_phase]
            if is_active:
                key = color
            elif spec['demo'] == 0:
                key = word[0]
            else:
                key = int(rng.choice(sorted(set(word)-{color})))
            obj = dict(cells=cells, port=port, key_cell=key_cell, key=key,
                       out_dir=outgoing, active=is_active, arrival=hit_phase,
                       approach=approach, before_phase=phase, through=through)
            objects.append(obj)
            occupied |= cells
            trace.update(approach)
            if is_active:
                trace.update(cells)
                front, direction, phase = _step(port, outgoing), outgoing, (hit_phase+1) % len(word)
                active_index += 1
            else:
                trace.update(through)
                front, phase = _step(through[-1], direction), (hit_phase+len(through)) % len(word)
            success = True
            break
        if not success:
            return None
    final = _ray(front, direction, shape)
    if set(final) & (occupied | trace):
        return None
    trace.update(final)
    for _ in range(spec['decoys']):
        for _ in range(120):
            outgoing = int(rng.integers(4))
            mask, h, w = _mask(rng, spec['maximum'])
            ports = _port_options(mask, h, w, outgoing)
            keys = [p for p in sorted(mask) if 0 < p[0] < h-1 and 0 < p[1] < w-1]
            if not ports or not keys:
                continue
            t, l = int(rng.integers(1, shape[0]-h)), int(rng.integers(1, shape[1]-w))
            cells = _translate(mask, t, l)
            if cells & trace or _dilate(cells) & occupied:
                continue
            port, key_cell = ports[int(rng.integers(len(ports)))], keys[int(rng.integers(len(keys)))]
            objects.append(dict(cells=cells, port=(t+port[0], l+port[1]),
                                key_cell=(t+key_cell[0], l+key_cell[1]),
                                key=int(rng.choice(word)), out_dir=outgoing, active=None))
            occupied |= cells
            break
        else:
            return None
    if len(occupied) >= np.prod(shape)//2:
        return None
    return dict(shape=shape, source=source, objects=objects, final=final, final_phase=phase)


def _fallback(rng, spec, word):
    """Keep counts in a horizontal encounter band, then turn the last match."""
    source = [(5, k) for k in range(len(word)+1)]
    front, phase, objects, active_index = (5, len(word)+1), 0, [], 0
    for event_index, is_active in enumerate(spec['events']):
        gap = 1
        desired = 1 if spec['demo'] == 0 and not is_active else 0
        if spec['demo'] is not None and (not is_active or active_index == 0):
            while (phase+gap) % len(word) != desired:
                gap += 1
        left, top = front[1]+gap, 4
        h = int(rng.integers(7, 9)) if is_active and spec['large_index'] == active_index else int(rng.choice([3, 4, 5]))
        cells = {(r, c) for r in range(top, top+h) for c in range(left, left+3)}
        # A clipped corner witnesses exact-footprint activation in fallback too.
        cells.remove((top+h-1, left))
        hit_phase = (phase+gap) % len(word)
        color = word[hit_phase]
        key = color if is_active else word[0] if spec['demo'] == 0 else int(rng.choice(sorted(set(word)-{color})))
        last_turn = is_active and event_index == len(spec['events'])-1
        outgoing = 0 if last_turn else 1
        port = (top, left+1) if last_turn else (5, left+2)
        through = [(5, c) for c in range(left, left+3)]
        obj = dict(cells=cells, port=port, key_cell=(5, left+1), key=key,
                   out_dir=outgoing, active=is_active, arrival=hit_phase,
                   approach=[(5, c) for c in range(front[1], left)],
                   before_phase=phase, through=through)
        objects.append(obj)
        if is_active:
            front, phase = _step(port, outgoing), (hit_phase+1) % len(word)
            active_index += 1
        else:
            front, phase = (5, left+3), (hit_phase+3) % len(word)
    direction = objects[-1]['out_dir'] if objects and objects[-1]['active'] else 1
    final = _ray(front, direction, (30, 30))
    for i in range(spec['decoys']):
        top, left = 17 + 7*(i//3), 2 + 9*(i % 3)
        cells = {(r, c) for r in range(top, top+4) for c in range(left, left+4)}
        cells.remove((top+3, left+3))
        objects.append(dict(cells=cells, port=(top, left+1), key_cell=(top+1, left+1),
                            key=int(rng.choice(word)), out_dir=0, active=None))
    layout = dict(shape=(30, 30), source=source, objects=objects, final=final, final_phase=phase)
    # Rotation changes the initial side without changing the word or clock.
    for _ in range((1-spec['initial']) % 4):
        rot = lambda p: (29-p[1], p[0])
        layout['source'] = list(map(rot, layout['source']))
        layout['final'] = list(map(rot, layout['final']))
        for obj in objects:
            obj['cells'] = set(map(rot, obj['cells']))
            obj['port'], obj['key_cell'] = rot(obj['port']), rot(obj['key_cell'])
            obj['out_dir'] = (obj['out_dir']-1) % 4
            if obj['active'] is not None:
                obj['approach'] = list(map(rot, obj['approach']))
                obj['through'] = list(map(rot, obj['through']))
    return layout


def _paint(layout, word, palette, rng):
    bg, marker = palette[:2]
    inp = np.full(layout['shape'], bg, dtype=np.int8)
    out = np.full_like(inp, bg)
    source = layout['source']
    inp[source[0]], out[source[0]] = marker, word[-1]
    for p, color in zip(source[1:], word):
        inp[p] = out[p] = color
    carriers = palette[2:4]
    for obj in layout['objects']:
        carrier = int(rng.choice(carriers))
        for p in obj['cells']:
            inp[p] = carrier
        inp[obj['port']], inp[obj['key_cell']] = marker, obj['key']
        if obj['active'] is None:
            continue
        for i, p in enumerate(obj['approach']):
            out[p] = word[(obj['before_phase']+i) % len(word)]
        if obj['active']:
            for p in obj['cells']:
                out[p] = word[obj['arrival']]
        else:
            for i, p in enumerate(obj['through']):
                out[p] = word[(obj['arrival']+i) % len(word)]
    for i, p in enumerate(layout['final']):
        out[p] = word[(layout['final_phase']+i) % len(word)]
    return dict(input=inp, output=out)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} through {high}')
    return int(value)


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _make_spec(rng, regime, test, demo, initial):
    n = 1 if demo == 0 else 2 if demo == 1 else int(rng.integers(0, 3))
    skips = 1 if demo == 0 else 0 if demo == 1 else int(rng.integers(0, 3))
    if test:
        n = int(rng.integers(3, 5)) if regime == 'route' else int(rng.integers(1, 4))
        skips = int(rng.integers(1, min(2, 5-n)+1))
    skips = min(skips, 5-n)
    if demo == 0:
        events = [False, True]
    elif demo == 1:
        events = [True, True]
    else:
        events = [True]*n + [False]*skips
        rng.shuffle(events)
    return dict(events=events, decoys=int(rng.integers(1 if demo == 0 else 0, 4)),
                initial=initial, maximum=5 if regime == 'scale' else 6,
                large_index=int(rng.integers(n)) if test and regime == 'scale' else None,
                demo=demo)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = str(rng.choice(['route', 'word', 'scale']))
    fixed_side, side = bool(rng.random() < .30), int(rng.integers(4))
    fixed_palette, base = bool(rng.random() < .35), list(map(int, rng.permutation(10)))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            demo = i if split == 'train' and i < 2 else None
            initial = side if fixed_side else int(rng.integers(4))
            spec = _make_spec(rng, regime, split == 'test', demo, initial)
            palette = base if fixed_palette else list(map(int, rng.permutation(10)))
            payload = palette[4:]
            if demo == 0:
                word = [payload[0], payload[1], payload[0]]
            elif demo == 1:
                word = [payload[1], payload[0]]
            else:
                length = int(rng.integers(4, 6)) if split == 'test' and regime == 'word' else int(rng.integers(2, 4))
                colors = payload[:int(rng.integers(2, min(4, length)+1))]
                word = list(map(int, rng.choice(colors, size=length)))
                if len(set(word)) == 1:
                    word[-1] = next(c for c in colors if c != word[0])
            pair = None
            for attempt in range(25):
                layout = _try_layout(rng, spec, word) if attempt < 24 else _fallback(rng, spec, word)
                if layout is None:
                    continue
                candidate = _paint(layout, word, palette, rng)
                try:
                    predicted, facts = _trace(_decode(candidate['input']))
                    assert np.array_equal(predicted, candidate['output'])
                    assert len(facts['activated']) == sum(spec['events'])
                    assert len(facts['rejected']) == len(spec['events'])-sum(spec['events'])
                    assert facts['decoys'] == spec['decoys']
                except AssertionError:
                    if attempt == 24:
                        raise
                    continue
                pair = candidate
                break
            assert pair is not None
            if _key(pair['input']) in used:
                # Finite injective recoloring guarantees a distinct input without
                # reducing the structural specification or introducing state.
                visible = list(map(int, np.unique(pair['input'])))
                for values in itertools.permutations(range(10), len(visible)):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[visible] = values
                    repaired = {k: mapping[v] for k, v in pair.items()}
                    if _key(repaired['input']) not in used:
                        pair = repaired
                        break
                else:
                    raise AssertionError('recoloring space exhausted')
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _period(word):
    return next(p for p in range(1, len(word)+1) if len(word) % p == 0
                and all(c == word[i % p] for i, c in enumerate(word)))


def _facts(pair):
    scene = _decode(pair['input'])
    predicted, f = _trace(scene)
    assert np.array_equal(predicted, _grid(pair['output'])), 'wrong clock, gate, footprint, or erasure'
    f['word_length'] = len(scene['word'])
    f['period'] = _period(scene['word'])
    f['dimensions'] = [(o['box'][2]-o['box'][0]+1, o['box'][3]-o['box'][1]+1) for o in scene['objects']]
    f['turns'] = [(e['outgoing']-e['incoming']) % 4 for e in f['encounters'] if e['active']]
    for e in f['encounters']:
        o = scene['objects'][e['index']]
        assert e['active'] == (e['color'] == o['key'])
        if e['active']:
            assert all(int(predicted[p]) == e['color'] for p in o['cells'])
        else:
            assert any(p in f['ray_cells'] for p in o['cells'])
            assert all(int(predicted[p]) == scene['bg'] for p in o['cells']-f['ray_cells'])
    reached = set(f['activated']) | set(f['rejected'])
    assert all(all(int(predicted[p]) == scene['bg'] for p in o['cells'])
               for i, o in enumerate(scene['objects']) if i not in reached)
    return f


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    facts, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _facts(pair)
            assert not np.array_equal(inp, out)
            facts[split].append(f)
    train, test = facts['train'], facts['test']
    assert any(f['activated'] for f in train)
    assert any(f['rejected'] for f in train) and any(f['decoys'] for f in train)
    assert any(0 in f['turns'] for f in train)
    assert any(t in (1, 3) for f in train for t in f['turns'])
    assert any(any(x['key'] == y['key'] and x['active'] != y['active']
                   for x, y in itertools.combinations(f['encounters'], 2)) for f in train)
    assert any(f['scene']['word'][0] == f['scene']['word'][-1]
               and len(f['scene']['word']) == 3 and f['period'] == 3 for f in train)
    assert any(any(len(f['scene']['objects'][i]['cells']) < np.prod(f['dimensions'][i])
                   for i in f['activated']) for f in train)
    # Verify differences in actual output, not merely different latent phases.
    for mode in ('constant', 'ignore_relays', 'unconditional', 'restart'):
        assert any(not np.array_equal(_trace(f['scene'], strict=False, **{mode: True})[0], pair['output'])
                   for f, pair in zip(train, puzzle['train'])), f'no witness against {mode}'
    route_shift = min(len(f['activated']) for f in test) > max(len(f['activated']) for f in train)
    word_shift = min(f['word_length'] for f in test) > max(f['word_length'] for f in train)
    largest_training = max((max(d) for f in train for d in f['dimensions']), default=0)
    scale_shift = all(any(max(f['dimensions'][i]) > largest_training for i in f['activated']) for f in test)
    assert route_shift or word_shift or scale_shift, 'missing measurable train/test shift'


def _semantic_regressions() -> int:
    input_grid = np.zeros((11, 17), dtype=np.int8)
    input_grid[5, :4] = [1, 2, 3, 2]
    input_grid[4:7, 6:9] = 4
    input_grid[4:7, 11:14] = 5
    input_grid[4, 7] = input_grid[4, 12] = 1
    input_grid[5, 7] = input_grid[5, 12] = 3
    expected_output = np.zeros_like(input_grid)
    expected_output[5] = [int(color) for color in '22322322322333000']
    expected_output[4, 11:14] = expected_output[6, 11:14] = 3
    expected_output[:4, 12] = [2, 3, 2, 2]

    changed_key = input_grid.copy()
    changed_key[5, 7] = 2
    changed_output = np.zeros_like(input_grid)
    changed_output[5, :6] = [2, 2, 3, 2, 2, 3]
    changed_output[4:7, 6:9] = 2
    changed_output[:4, 7] = [2, 2, 3, 2]
    assert not np.array_equal(expected_output, changed_output)

    recoloring = np.array([8, 0, 7, 4, 1, 9, 5, 2, 6, 3], dtype=np.int8)
    cases_checked = 0
    for original, expected in ((input_grid, expected_output), (changed_key, changed_output)):
        for rotations in range(4):
            for mirrored in (False, True):
                oriented_input = np.rot90(original, rotations)
                oriented_output = np.rot90(expected, rotations)
                if mirrored:
                    oriented_input = np.fliplr(oriented_input)
                    oriented_output = np.fliplr(oriented_output)
                for mapping in (np.arange(10, dtype=np.int8), recoloring):
                    test_input = mapping[oriented_input]
                    unchanged_input = test_input.copy()
                    assert np.array_equal(solve(test_input), mapping[oriented_output])
                    assert np.array_equal(test_input, unchanged_input)
                    cases_checked += 1
    return cases_checked


def self_test() -> dict:
    """Bounded 42-seed semantic/diversity probe; no external evidence access."""
    semantic_regressions = _semantic_regressions()
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                records.append((split, _facts(pair)))
    signature = lambda p: tuple((_key(q['input']), _key(q['output'])) for s in ('train', 'test') for q in p[s])
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    fs = [f for _, f in records]
    shapes = {tuple(f['scene']['grid'].shape) for f in fs}
    assert len(shapes) >= 40
    assert {f['scene']['initial'] for f in fs} == set(range(4))
    assert {f['period'] for f in fs} >= {2, 3, 4, 5}
    assert {len(f['activated']) for f in fs} >= {0, 1, 2, 3, 4}
    assert {len(f['rejected']) for f in fs} >= {0, 1, 2}
    assert {f['decoys'] for f in fs} >= {0, 1, 2, 3}
    assert {t for f in fs for t in f['turns']} == {0, 1, 3}
    assert any(e['gap'] >= 9 for f in fs for e in f['encounters'])
    assert len({d for f in fs for d in f['dimensions']}) >= 20
    assert len({tuple(f['scene']['word']) for f in fs}) >= 50
    assert {f['scene']['bg'] for f in fs} == set(range(10))
    # Directly exercise fallback counts, all sides, and scale extrapolation.
    rng = np.random.default_rng(90013)
    for demo in (0, 1, None):
        for initial in range(4):
            spec = dict(events=[False, True] if demo == 0 else [True, True] if demo == 1
                        else [False, True, False, True, True], decoys=3,
                        initial=initial, maximum=5, large_index=1, demo=demo)
            word = [2, 3, 2] if demo == 0 else [2, 3]
            pair = _paint(_fallback(rng, spec, word), word, [0, 1, 4, 5], rng)
            f = _facts(pair)
            assert len(f['activated']) == sum(spec['events'])
            assert f['decoys'] == 3
    # A one-color word and no relays are valid family members for solve.
    simple = np.zeros((12, 17), dtype=np.int8)
    simple[4, 0], simple[4, 1] = 1, 2
    predicted = solve(simple)
    assert np.all(predicted[4] == 2) and np.count_nonzero(predicted) == 17
    # Reject once, then pass through both pieces of a notched footprint.
    notched = np.zeros((12, 17), dtype=np.int8)
    notched[3, :3] = [1, 2, 3]
    notched[2:7, 6:12] = 4
    notched[3, 8:10] = 0
    notched[2, 7], notched[5, 9] = 1, 2
    predicted = solve(notched)
    assert np.count_nonzero(predicted) == 17
    assert np.array_equal(predicted[3], np.array([3]+[2, 3]*8))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(175 + a*7 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), shapes=len(shapes),
                semantic_regressions=semantic_regressions,
                words=len({tuple(f['scene']['word']) for f in fs}),
                footprint_sizes=len({d for f in fs for d in f['dimensions']}),
                active_counts=sorted({len(f['activated']) for f in fs}),
                rejected_counts=sorted({len(f['rejected']) for f in fs}),
                semantic_verification='author-owned input-only solver and invariants')


if __name__ == '__main__':
    print(self_test())
