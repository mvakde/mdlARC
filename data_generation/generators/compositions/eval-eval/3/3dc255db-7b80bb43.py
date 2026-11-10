"""Finite paint pulses on orthogonal rails with visibly displaced gaps.

Construction paints labels from sampled paths. The solver independently reads
frames, local gap evidence, and the input's orthogonal adjacency graph.
"""
from __future__ import annotations

import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _add(a, b, k=1):
    return a[0] + k*b[0], a[1] + k*b[1]


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = DIRS if not diagonal else tuple((r, c) for r in (-1, 0, 1)
                                            for c in (-1, 0, 1) if r or c)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            p = stack.pop()
            for d in steps:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _background(a):
    values, counts = np.unique(a, return_counts=True)
    i = int(np.argmax(counts))
    if counts[i] <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    return int(values[i])


def _frame_read(cells, d):
    # y increases behind the nozzle; x is transverse to it.
    dr, dc = d
    local = {(-r*dr-c*dc, r*dc-c*dr) for r, c in cells}
    y0, y1 = min(y for y, x in local), max(y for y, x in local)
    rows = [sorted(x for y, x in local if y == v) for v in range(y0, y1+1)]
    if len(rows) < 4 or len(rows[0]) != 1:
        return None
    tipx = rows[0][0]
    lo = hi = tipx
    split, cavity = False, set()
    for y, row in enumerate(rows[1:], y0+1):
        if len(row) == 1 and not split and row[0] == tipx:
            continue
        if len(row) != 2:
            return None
        left, right = row
        if (left > lo or right < hi or lo-left > 1 or right-hi > 1
                or left >= right):
            return None
        split, lo, hi = True, left, right
        cavity.update((-yy*dr+xx*dc, -yy*dc-xx*dr)
                      for yy, xx in ((y, x) for x in range(lo+1, hi)))
    if not split or not lo < tipx < hi or hi-lo < 4:
        return None
    tip = (-y0*dr+tipx*dc, -y0*dc-tipx*dr)
    return tip, cavity


def _sources(a, bg):
    candidates = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for cells in _components(map(tuple, np.argwhere(a == color)), True):
            if len(cells) < 8:
                continue
            for d in DIRS:
                frame = _frame_read(cells, d)
                if frame is None:
                    continue
                tip, cavity = frame
                if not all(_inside(a.shape, p) for p in cavity):
                    continue
                values = {int(a[p]) for p in cavity} - {bg}
                if len(values) != 1 or color in values:
                    continue
                payload_color = values.pop()
                payload = {p for p in cavity if int(a[p]) == payload_color}
                if len(payload) == len(cavity):
                    continue
                start = _add(tip, d)
                if (not _inside(a.shape, start)
                        or int(a[start]) in (bg, color, payload_color)):
                    continue
                candidates.append(dict(frame=cells, cavity=cavity, tip=tip,
                    direction=d, start=start, frame_color=color,
                    payload_color=payload_color, payload=payload,
                    quantity=len(payload), guide_color=int(a[start])))
    candidates = [s for s in candidates if not any(
        s is not t and s['frame'] <= t['cavity'] for t in candidates)]
    if not candidates:
        raise ValueError("no pointed reservoir attached to a rail")
    if len({frozenset(s['frame']) for s in candidates}) != len(candidates):
        raise ValueError("ambiguous reservoir direction")
    return sorted(candidates, key=lambda s: s['tip'])


def _gap_evidence(a, guide, bg):
    """Bracketed three-cell break + exactly two diagonal cells at an anchor.

    Cue cells have no orthogonal rail neighbor. At least one bracket continues
    straight away from the break. Multiple matching cues are invalid.
    """
    cells = set(map(tuple, np.argwhere(a == guide)))
    gaps = []
    for p in sorted(cells):
        for d in ((0, 1), (1, 0)):
            end = _add(p, d, 4)
            missing = tuple(_add(p, d, k) for k in (1, 2, 3))
            if (end not in cells or any(not _inside(a.shape, q) or a[q] != bg for q in missing)
                    or not (_add(p, d, -1) in cells or _add(end, d) in cells)):
                continue
            cues = []
            for anchor, toward in ((p, 1), (end, -1)):
                for side in (-1, 1):
                    step = (toward*d[0]-side*d[1], toward*d[1]+side*d[0])
                    spur = tuple(_add(anchor, step, k) for k in (1, 2))
                    if (all(q in cells and not any(_add(q, v) in cells for v in DIRS) for q in spur)
                            and _add(anchor, step, 3) not in cells):
                        cues.append(spur)
            if len(cues) > 1:
                raise ValueError("a break has several diagonal repair cues")
            if cues:
                gaps.append(dict(cells=missing, cue=cues[0], endpoints=(p, end)))
    if len({p for g in gaps for p in g['cue']}) != 2*len(gaps):
        raise ValueError("a repair cue is shared by multiple breaks")
    return cells, gaps


def _walk(nodes, start):
    route, visited = [], set()
    previous, p = None, start
    while p is not None:
        if p in visited:
            raise ValueError("rail has a cycle")
        visited.add(p)
        route.append(p)
        onward = [q for d in DIRS if (q := _add(p, d)) in nodes and q != previous]
        if len(onward) > 1:
            raise ValueError("rail branches or touches itself")
        previous, p = p, onward[0] if onward else None
    return tuple(route)


def _read(a, bridge=True):
    bg = _background(a)
    sources = _sources(a, bg)
    frame_colors = {s['frame_color'] for s in sources}
    guide_colors = {s['guide_color'] for s in sources}
    if (len(frame_colors) != 1 or len(guide_colors) != 1
            or frame_colors & guide_colors
            or any(s['payload_color'] in frame_colors | guide_colors for s in sources)):
        raise ValueError("scene color roles must be unambiguous and distinct")
    reservoirs = set()
    for s in sources:
        region = s['frame'] | s['cavity']
        if reservoirs & region:
            raise ValueError("reservoir hollows overlap")
        reservoirs.update(region)
    by_color = {}
    for s in sources:
        guide = s['guide_color']
        if guide not in by_color:
            rail, gaps = _gap_evidence(a, guide, bg)
            cues = {p for g in gaps for p in g['cue']}
            nodes = rail - cues
            if bridge:
                nodes |= {p for g in gaps for p in g['cells']}
            by_color[guide] = (nodes, gaps)
        nodes, gaps = by_color[guide]
        s['route'] = _walk(nodes, s['start'])
        if set(s['route']) & reservoirs:
            raise ValueError("rail enters a reservoir")
        s['gaps'] = [g for g in gaps if set(g['cells']) & set(s['route'])]
        s['prefix'] = s['route'][:s['quantity']]
        s['reached'] = [g for g in s['gaps'] if set(g['cells']) & set(s['prefix'])]
    occupied = set()
    for s in sources:
        if occupied & set(s['route']):
            raise ValueError("two reservoirs feed the same rail")
        occupied.update(s['route'])
    return bg, sources


def _paint(a, bg, sources):
    out = a.copy()
    for s in sources:
        for p in s['payload']:
            out[p] = bg
        for p in s['prefix']:
            out[p] = s['payload_color']
        for g in s['reached']:
            for p in g['cue']:
                out[p] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive each pulse and its reachable repair edits from this input only."""
    a = _grid(input_grid)
    bg, sources = _read(a)
    return _paint(a, bg, sources)


def _reservoir(rng, small):
    a, b = (2, 2) if small else (int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    stem = 0 if small else int(rng.integers(2))
    tail = int(rng.integers(2, 4)) if small else int(rng.integers(3, 6))
    frame, cavity = set(), set()
    for depth in range(stem+max(a, b)+tail+1):
        slope = max(0, depth-stem)
        lo, hi = -min(a, slope), min(b, slope)
        frame.update(((lo, -depth), (hi, -depth)))
        cavity.update((x, -depth) for x in range(lo+1, hi))
    return frame, cavity


def _geometry(rng, small=False, segments=3):
    frame, cavity = _reservoir(rng, small)
    lengths = [int(rng.integers(3, 5 if small else 7))]
    for j in range(1, segments):
        lengths.append(int(rng.integers(7, 9)) if small else
                       int(rng.integers(7, 11)) if j % 2 else int(rng.integers(6, 10)))
    path, runs, p = [], [], (0, 0)
    for j, length in enumerate(lengths):
        d = (0, 1) if j % 2 == 0 else (int(rng.choice([-1, 1])), 0)
        start = p
        index = len(path)-1
        for _ in range(length):
            p = _add(p, d)
            path.append(p)
        runs.append((start, d, length, index))
    nodes = set(path)
    assert len(nodes) == len(path) and nodes.isdisjoint(frame | cavity)
    assert _walk(nodes, path[0]) == tuple(path)
    return dict(frame=frame, cavity=cavity, path=path, runs=runs)


def _bounds(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs), max(cs)


def _damage(rng, geometry, small, mode):
    path, frame = geometry['path'], geometry['frame']
    rail = set(path)
    excluded, cues, gaps = frame | geometry['cavity'] | rail, set(), []
    minr, minc, maxr, maxc = _bounds(excluded)
    eligible = list(range(1, len(geometry['runs'])))
    if mode == 'witness':
        chosen = eligible[:2]
    elif mode == 'blocker':
        chosen = eligible[:2]
    else:
        rng.shuffle(eligible)
        chosen = sorted(eligible[:int(rng.integers(1, min(3, len(eligible))+1))])
    for j in chosen:
        start, d, length, index = geometry['runs'][j]
        options = []
        for distance in range(1, length-4):
            anchor = _add(start, d, distance)
            missing = tuple(_add(anchor, d, k) for k in (1, 2, 3))
            for endpoint in (0, 4):
                for side in (-1, 1):
                    toward = 1 if endpoint == 0 else -1
                    q = _add(anchor, d, endpoint)
                    step = (toward*d[0]-side*d[1], toward*d[1]+side*d[0])
                    spur = tuple(_add(q, step, k) for k in (1, 2))
                    visible_rail = rail-set(missing)
                    if (set(spur) & excluded or any(_add(p, v) in visible_rail for p in spur for v in DIRS)
                            or any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for p in spur for q in cues)
                            or small and any(not (minr <= p[0] <= maxr and minc <= p[1] <= maxc) for p in spur)):
                        continue
                    options.append((missing, spur, index+distance+1))
        if not options:
            raise ValueError("no clear diagonal cue placement")
        # Paired witnesses share a visibly identical virtual rail and first gap.
        if mode in ('witness', 'blocker'):
            first = min(x[2] for x in options)
            options = [x for x in options if x[2] == first]
        missing, spur, first_index = options[int(rng.integers(len(options)))]
        supported = not (mode == 'blocker' and j == chosen[0])
        if mode == 'ordinary' and j == chosen[-1] and len(chosen) > 1:
            supported = bool(rng.random() >= .25)
        if supported:
            cues.update(spur)
            excluded.update(spur)
        gaps.append(dict(cells=missing, cue=spur if supported else (),
                         index=first_index, supported=supported))
    geometry = dict(geometry, gaps=gaps)
    return geometry


def _payload(rng, cavity, q, arrangement):
    cells = sorted(cavity)
    selected = {cells[int(rng.integers(len(cells)))]}
    while len(selected) < q:
        other = sorted(cavity-selected)
        if arrangement == 'compact':
            near = [p for p in other if any(_add(p, d) in selected for d in DIRS)]
            other = near or other
        elif arrangement == 'scattered':
            distances = [min(max(abs(p[0]-s[0]), abs(p[1]-s[1])) for s in selected) for p in other]
            other = [p for p, dist in zip(other, distances) if dist == max(distances)]
        selected.add(other[int(rng.integers(len(other)))])
    return selected


def _circuit(rng, small, segments, mode='ordinary', quota=None, geometry=None, arrangement=None):
    g = geometry if geometry is not None else _damage(rng, _geometry(rng, small, segments), small, mode)
    capacity = len(g['cavity'])
    if quota is None:
        quota = int(rng.integers(2, min(capacity-1, len(g['path'])+5)+1))
    if not 1 <= quota < capacity:
        raise ValueError("pulse does not fit the reservoir")
    payload = _payload(rng, g['cavity'], quota, arrangement or str(rng.choice(['compact', 'scattered', 'random'])))
    end = min([x['index'] for x in g['gaps'] if not x['supported']] or [len(g['path'])])
    prefix = g['path'][:min(quota, end)]
    missing = {p for x in g['gaps'] for p in x['cells']}
    im = {p: 'frame' for p in g['frame']}
    im.update({p: 'payload' for p in payload})
    im.update({p: 'guide' for p in g['path'] if p not in missing})
    im.update({p: 'guide' for x in g['gaps'] for p in x['cue']})
    om = {p: role for p, role in im.items() if role != 'payload'}
    om.update({p: 'payload' for p in prefix})
    for x in g['gaps']:
        if set(x['cells']) & set(prefix):
            for p in x['cue']:
                om.pop(p, None)
    return dict(input=im, output=om, geometry=g, quota=quota)


def _transform_circuit(circuit, turns, flip):
    def transform(p):
        r, c = p
        if flip:
            r = -r
        for _ in range(turns):
            r, c = -c, r
        return r, c
    im = {transform(p): role for p, role in circuit['input'].items()}
    om = {transform(p): role for p, role in circuit['output'].items()}
    # Reserve virtual gaps too, even when they are absent from both grids.
    footprint = {transform(p) for p in circuit['geometry']['frame'] | set(circuit['geometry']['path'])}
    y0, x0, y1, x1 = _bounds(footprint | set(im) | set(om))
    return dict(input={(r-y0, c-x0): v for (r, c), v in im.items()},
                output={(r-y0, c-x0): v for (r, c), v in om.items()},
                shape=(y1-y0+1, x1-x0+1))


def _pack(rng, circuits, palette, turns, flips=None):
    if flips is None:
        flips = [bool(rng.integers(2)) for _ in circuits]
    specs = [_transform_circuit(x, t, f) for x, t, f in zip(circuits, turns, flips)]
    if any(max(s['shape']) > 30 for s in specs):
        raise ValueError("circuit too large")
    for attempt in range(8):
        if attempt < 5:
            h = int(rng.integers(max(s['shape'][0] for s in specs), 31))
            w = int(rng.integers(max(s['shape'][1] for s in specs), 31))
        else:
            h = w = 30
        rectangles, placements = [], []
        for s in specs:
            sh, sw = s['shape']
            positions = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)]
            rng.shuffle(positions)
            for r, c in positions:
                if all(r+sh+1 <= y or y+hh+1 <= r or c+sw+1 <= x or x+ww+1 <= c
                       for y, x, hh, ww in rectangles):
                    rectangles.append((r, c, sh, sw))
                    placements.append((r, c))
                    break
            else:
                break
        if len(placements) == len(specs):
            break
    else:
        raise ValueError("circuits do not fit")
    bg, frame, guide, payload_base = palette
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    for i, (s, (r, c)) in enumerate(zip(specs, placements)):
        payload = payload_base if i == 0 or rng.random() < .5 else int(rng.choice([x for x in range(10) if x not in (bg, frame, guide)]))
        colors = dict(frame=frame, guide=guide, payload=payload)
        for role, grid in (('input', inp), ('output', out)):
            for (y, x), name in s[role].items():
                grid[r+y, c+x] = colors[name]
    # Direct path-based construction must agree with input graph reconstruction.
    if not np.array_equal(solve(inp), out):
        raise ValueError("constructed path and input reconstruction disagree")
    return {'input': inp, 'output': out}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(n, t):
    for v, name, low, high in ((n, 'num_train', 2, 10), (t, 'num_test', 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= int(v) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    return int(n), int(t)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    direction = int(rng.integers(4))
    shift = str(rng.choice(['turns', 'rotation', 'reach']))
    # The smallest training set still contains a full repair, a partial repair,
    # an unreached repair cue, and an unsupported stop with surplus ink.
    for _ in range(30):
        try:
            witness = _damage(rng, _geometry(rng, True, 3), True, 'witness')
            k = witness['gaps'][0]['index']
            low, high = k+int(rng.integers(1, 3)), k+4
            if high >= len(witness['cavity']):
                continue
            full = _circuit(rng, True, 3, quota=high, geometry=witness, arrangement='compact')
            partial = _circuit(rng, True, 3, quota=low, geometry=witness, arrangement='scattered')
            blocker = _circuit(rng, True, 3, 'blocker', quota=high, arrangement='compact')
            witness_flip = bool(rng.integers(2))
            anchors = [_pack(rng, [full, blocker], palette, [direction, direction], [witness_flip, bool(rng.integers(2))]),
                       _pack(rng, [partial], palette, [direction], [witness_flip])]
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("could not construct teaching witnesses")
    puzzle, seen = {'train': anchors, 'test': []}, {_key(p['input']) for p in anchors}
    for split, count in (('train', n-2), ('test', t)):
        for i in range(count):
            for attempt in range(100):
                try:
                    number = (1 if split == 'test' and i == 0 and shift in ('turns', 'reach')
                              else int(rng.choice([1, 1, 2])))
                    circuits, orientations = [], []
                    for j in range(number):
                        small = number == 2
                        segments = 3 if small else int(rng.integers(2, 4 if split == 'train' and shift == 'turns' else 6))
                        if split == 'test' and i == 0 and shift == 'turns':
                            segments = int(rng.integers(4, 6))
                        g = _damage(rng, _geometry(rng, small, segments), small, 'ordinary')
                        quota = None
                        if split == 'test' and i == 0 and shift == 'turns':
                            minimum = sum(run[2] for run in g['runs'][:3])+1
                            maximum = min(len(g['cavity'])-1, len(g['path'])-1)
                            if maximum < minimum:
                                raise ValueError("reservoir too small for the turn-shift quota")
                            quota = int(rng.integers(minimum, maximum+1))
                        if split == 'test' and i == 0 and shift == 'reach':
                            # Make an actual later gap reachable with a larger
                            # pulse, rather than merely enlarging a blank canvas.
                            g = _damage(rng, _geometry(rng, False, 3), False, 'witness')
                            quota = g['gaps'][1]['index']+2
                        circuits.append(_circuit(rng, small, segments, quota=quota, geometry=g))
                        if shift == 'rotation':
                            orientations.append(direction if split == 'train' else (direction+1)%4)
                        else:
                            orientations.append(int(rng.integers(4)))
                    local_palette = palette if rng.random() < .45 else tuple(map(int, rng.choice(10, 4, replace=False)))
                    pair = _pack(rng, circuits, local_palette, orientations)
                    if _key(pair['input']) in seen:
                        continue
                    if split == 'test' and i == 0:
                        _, ss = _read(pair['input'])
                        if not any(s['reached'] and s['quantity'] < len(s['route']) for s in ss):
                            continue
                        if shift == 'turns' and not any(sum(
                                (b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1])
                                for a, b, c in zip(s['prefix'], s['prefix'][1:], s['prefix'][2:])) >= 3 for s in ss):
                            continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("could not construct a distinct valid scene")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, out = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == out.shape
    bg, sources = _read(a)
    assert np.array_equal(_paint(a, bg, sources), out), "incorrect pulse or missing/extra repair edit"
    assert not np.array_equal(a, out)
    records = []
    for s in sources:
        q, prefix = s['quantity'], s['prefix']
        assert all(out[p] == bg for p in s['payload'])
        assert all(out[p] == s['payload_color'] for p in prefix)
        assert all(out[p] == s['frame_color'] for p in s['frame'])
        assert len(prefix) == min(q, len(s['route']))
        turns = sum((b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1])
                    for a, b, c in zip(s['route'], s['route'][1:], s['route'][2:]))
        partial = sum(0 < len(set(g['cells']) & set(prefix)) < 3 for g in s['gaps'])
        full = sum(set(g['cells']) <= set(prefix) for g in s['gaps'])
        for g in s['gaps']:
            reached = bool(set(g['cells']) & set(prefix))
            assert all(out[p] == (bg if reached else s['guide_color']) for p in g['cue'])
            assert all(out[p] == (s['payload_color'] if p in prefix else bg) for p in g['cells'])
        tr, tc = s['tip']
        frame_profile = tuple(sorted((r-tr, c-tc) for r, c in s['frame']))
        route_profile = tuple((r-tr, c-tc) for r, c in s['route'])
        records.append(dict(quantity=q, visible=len(prefix), full=full, partial=partial,
            untouched=len(s['gaps'])-len(s['reached']), clipped=q>len(prefix),
            turns=turns, direction=s['direction'], frame=frame_profile,
            route=route_profile, components=len(_components(s['payload'], True)),
            signature=(frame_profile, route_profile, s['frame_color'], s['guide_color'], s['payload_color'])))
    return records


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, "duplicate input"
            seen.add(key)
            records = _features(pair)
            (train if split == 'train' else test).extend(records)
    assert any(x['full'] and x['visible'] == x['quantity'] and x['turns'] for x in train)
    assert any(x['partial'] and not x['clipped'] for x in train)
    assert any(x['untouched'] for x in train), "no evidence against globally repairing all damage"
    assert any(x['clipped'] for x in train), "no unsupported/terminal stop demonstration"
    assert any(x['components'] < x['quantity'] and not x['clipped'] for x in train)
    by_signature = {}
    for x in train:
        if not x['clipped']:
            by_signature.setdefault(x['signature'], set()).add(x['quantity'])
    assert any(len(q) >= 2 for q in by_signature.values()), "geometry alone can explain the quotas"
    assert any(x['full'] or x['partial'] for x in test), "tests omit gap reasoning"
    # Both parent ablations must disagree on actual training and test grids.
    for split in ('train', 'test'):
        assert any(not np.array_equal(_paint(a := _grid(p['input']), *_read(a, bridge=False)), _grid(p['output']))
                   for p in puzzle[split]), "ignoring B changes no answer"
        assert any(any(s['quantity'] < len(s['route']) for s in _read(_grid(p['input']))[1])
                   for p in puzzle[split]), "ignoring A's finite count changes no answer"


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded 42-seed semantic/diversity sweep plus metamorphic regressions."""
    shapes, directions, profiles, quantities, turns, source_counts, palettes = set(), set(), set(), set(), set(), set(), set()
    fingerprints, totals = set(), dict(full=0, partial=0, untouched=0, clipped=0)
    split_counts = {'train': set(), 'test': set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, out = pair['input'], pair['output']
                records = _features(pair)
                shapes.add(a.shape)
                source_counts.add(len(records))
                split_counts[split].add(len(records))
                palettes.add(tuple(map(int, np.unique(a))))
                for x in records:
                    directions.add(x['direction'])
                    profiles.add(x['frame'])
                    quantities.add(x['quantity'])
                    turns.add(x['turns'])
                    for name in totals:
                        totals[name] += x[name]
        if seed < 3:
            for pair in puzzle['train'] + puzzle['test']:
                # A geometric rule must commute with rotation/reflection and
                # every literal-color permutation, including background swaps.
                permutation = np.random.default_rng(seed+900).permutation(10)
                for k in range(4):
                    a, out = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                    assert np.array_equal(solve(a), out)
                    assert np.array_equal(solve(permutation[a[:, ::-1]]), permutation[out[:, ::-1]])
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(profiles) >= 25 and len(palettes) >= 20
    assert directions == set(DIRS) and source_counts == {1, 2}
    assert all(counts == {1, 2} for counts in split_counts.values())
    assert len(quantities) >= 12 and max(turns) >= 4 and all(totals.values())
    # A compatible coil with a backward horizontal leg is outside the sampler's
    # monotone route construction. Exercise it, unit quota, and endpoint clipping.
    rng = np.random.default_rng(991)
    frame, cavity = _reservoir(rng, False)
    path, runs, position = [], [], (0, 0)
    for d, length in (((0, 1), 4), ((1, 0), 8), ((0, -1), 8), ((1, 0), 8), ((0, 1), 9)):
        start, index = position, len(path)-1
        for _ in range(length):
            position = _add(position, d)
            path.append(position)
        runs.append((start, d, length, index))
    coil = _damage(rng, dict(frame=frame, cavity=cavity, path=path, runs=runs), False, 'witness')
    for quota in (1, coil['gaps'][1]['index']+2, len(cavity)-1):
        if quota >= len(cavity):
            continue
        circuit = _circuit(rng, False, 5, geometry=coil, quota=quota)
        for rotation in range(4):
            pair = _pack(rng, [circuit], (8, 0, 3, 6), [rotation])
            assert np.array_equal(solve(pair['input']), pair['output'])
    # The canonical route deliberately ends shortly after its last cue. A
    # deeper reservoir supplies more ink than this intact route can accept.
    short = _geometry(rng, False, 2)
    frame, cavity = set(), set()
    for depth in range(10):
        reach = min(depth, 3)
        frame.update(((-reach, -depth), (reach, -depth)))
        cavity.update((r, -depth) for r in range(-reach+1, reach))
    short.update(frame=frame, cavity=cavity)
    short = _damage(rng, short, False, 'ordinary')
    circuit = _circuit(rng, False, 2, geometry=short, quota=len(short['path'])+1)
    pair = _pack(rng, [circuit], (0, 1, 2, 3), [0])
    assert _features(pair)[0]['clipped']
    p = generate_puzzle(824664356, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(824664356, 2, 1))
    # Wrong labels: extra virtual gap cells, omitted cue deletion, one too many
    # painted positions, or restoring the reservoir's payload must all fail.
    for mutation in ('gap', 'cue', 'count', 'payload'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['train'][1]
        bg, sources = _read(pair['input'])
        source = sources[0]
        if mutation == 'gap':
            g = next(g for g in source['reached'] if not set(g['cells']) <= set(source['prefix']))
            q = next(q for q in g['cells'] if q not in source['prefix'])
            pair['output'][q] = source['guide_color']
        elif mutation == 'cue':
            pair['output'][source['reached'][0]['cue'][0]] = source['guide_color']
        elif mutation == 'count':
            pair['output'][source['route'][source['quantity']]] = source['payload_color']
        else:
            q = min(source['payload'])
            pair['output'][q] = source['payload_color']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"bad {mutation} label accepted")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(100+n*5+t, n, t))
    for counts in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (3,1.5), (None,1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Compact measurements are optional scratch artifacts, never solver state.
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                frame_profiles=len(profiles), palettes=len(palettes), quantities=sorted(quantities),
                turns=sorted(turns), source_counts=sorted(source_counts), **totals)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
