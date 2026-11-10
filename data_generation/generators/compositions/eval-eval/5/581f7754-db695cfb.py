"""Finite diagonal launch gates with boundary-registered rigid cargo.

Only Python's standard library and NumPy are used. Generation builds colored
material and traces from placement records; solve reconstructs them from input.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, bg):
    left = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while left:
        start = min(left)
        left.remove(start)
        todo, points = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    points.add(p)
                    todo.append(p)
        result.append(points)
    return result


def _segment(a, b):
    _require(a[0] < b[0] and abs(a[1] - b[1]) == b[0] - a[0], "not a diagonal")
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k, a[1] + slope * k) for k in range(b[0] - a[0] + 1))


def _line(p, slope, shape):
    h, w = shape
    return {(r, p[1] + slope * (r - p[0])) for r in range(h)
            if 0 <= p[1] + slope * (r - p[0]) < w}


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _destination(p, slope, system):
    axis, guide = system['axis'], system['guide']
    d = guide - p[axis]
    return (p[0] + d, p[1] - slope * d) if axis == 0 else (p[0] - slope * d, p[1] + d)


def _touch(points, occupied):
    return bool(points & occupied) or any((r + dr, c + dc) in occupied
                                          for r, c in points for dr, dc in DIRS)


def _decode(a):
    """Input-only component, anchor, rail, launch and destination reconstruction."""
    a = _grid(a)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > a.size / 2, "background must occupy over half the input")
    h, w = a.shape
    anchors, rest = {}, []
    for component in _components(a, bg):
        p = min(component)
        edge = p[0] in (0, h - 1) or p[1] in (0, w - 1)
        if len(component) == 1 and edge:
            _require(not (p[0] in (0, h - 1) and p[1] in (0, w - 1)), "corner anchor")
            anchors.setdefault(int(a[p]), set()).add(p)
        else:
            rest.append(component)
    _require(1 <= len(anchors) <= 3, "need one to three registration identities")
    systems = {}
    fixed = {}
    for color, points in anchors.items():
        rows, cols = {p[0] for p in points}, {p[1] for p in points}
        horizontal = len(rows) == 1 and all(c in (0, w - 1) for _, c in points)
        vertical = len(cols) == 1 and all(r in (0, h - 1) for r, _ in points)
        _require(horizontal != vertical, "inconsistent boundary guide")
        axis = 0 if horizontal else 1
        systems[color] = {'axis': axis, 'guide': next(iter(rows if horizontal else cols)),
                          'anchors': points}
        fixed.update({p: color for p in points})
    dots, movers = {}, []
    for component in rest:
        marks = [p for p in component if int(a[p]) in systems]
        if not marks:
            _require(len(component) == 1, "unregistered material")
            p = min(component)
            dots[p] = int(a[p])
            continue
        _require(len(marks) == 1, "cargo must have exactly one registration cell")
        p = marks[0]
        movers.append({'points': {q: int(a[q]) for q in component}, 'mark': p,
                       'color': int(a[p])})
    _require(dots and len(set(dots.values())) == 1, "rail dots must share one color")
    primary = next(iter(dots.values()))
    _require(primary not in systems, "rail and registration colors must differ")
    _require(all(primary not in m['points'].values() for m in movers), "rail color in cargo")
    _require(all(any(m['color'] == c for m in movers) for c in systems), "anchor without cargo")
    points, degrees, rails = sorted(dots), Counter(), []
    for i, p in enumerate(points):
        for q in points[i + 1:]:
            if q[0] != p[0] and abs(q[0] - p[0]) == abs(q[1] - p[1]):
                cells = _segment(p, q)
                _require(len(cells) >= 3, "rail too short")
                rails.append(cells)
                degrees[p] += 1
                degrees[q] += 1
    _require(1 <= len(rails) <= 4 and all(n == 1 for n in degrees.values()), "ambiguous endpoint partners")
    fixed.update(dots)
    union = set().union(*(set(s) for s in rails))
    traces, footprints = {}, set(fixed)
    for m in movers:
        p = m['mark']
        hits = [s for s in rails if p in s[1:-1]]
        _require(len(hits) <= 1, "two-slope launch ambiguity")
        m['active'] = bool(hits)
        m['slope'] = (1 if hits[0][-1][1] > hits[0][0][1] else -1) if hits else None
        q = _destination(p, m['slope'], systems[m['color']]) if hits else p
        _require(_inside(q, a.shape), "registration destination leaves canvas")
        m['delta'] = q[0] - p[0], q[1] - p[1]
        m['target'] = {(r + m['delta'][0], c + m['delta'][1]): v
                       for (r, c), v in m['points'].items()}
        visible = {p for p in m['target'] if _inside(p, a.shape)}
        _require(not (visible & footprints), "material destinations overlap")
        footprints.update(visible)
        m['clipped'] = len(visible) < len(m['target'])
        m['trace'] = _line(p, -m['slope'], a.shape) if hits else set()
        if hits:
            _require(m['trace'] & union == {p}, "trace crosses another rail")
            for t in m['trace']:
                _require(t not in traces or traces[t] == m['color'], "different-colored traces cross")
                _require(t not in fixed or fixed[t] == m['color'], "trace crosses a fixed dot of another color")
                traces[t] = m['color']
    return {'grid': a, 'bg': bg, 'primary': primary, 'systems': systems,
            'fixed': fixed, 'rails': rails, 'union': union, 'movers': movers, 'traces': traces}


def _paint(data, actions=None, marker_only=False):
    a = data['grid']
    out = np.full(a.shape, data['bg'], dtype=np.int8)
    for s in data['rails']:
        for p in s:
            out[p] = data['primary']
    movers = data['movers'] if actions is None else actions
    for m in movers:
        for p in m['trace']:
            out[p] = m['color']
    for p, v in data['fixed'].items():
        out[p] = v
    for m in movers:
        for p, v in m['points'].items():
            dr, dc = m['delta'] if not marker_only or p == m['mark'] else (0, 0)
            q = p[0] + dr, p[1] + dc
            if _inside(q, a.shape):
                out[q] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; there is no seed or memorized convention."""
    return _paint(_decode(input_grid))


def _alternative(data, mode):
    """Specific simpler rules used only as teaching-evidence diagnostics."""
    actions = []
    for original in data['movers']:
        m = dict(original)
        if mode == 'no_boundary':
            m['delta'] = (0, 0)
        elif mode in ('orthogonal', 'only_A'):
            axis = data['systems'][m['color']]['axis']
            d = data['systems'][m['color']]['guide'] - m['mark'][axis]
            m['delta'] = (d, 0) if axis == 0 else (0, d)
        elif mode in ('infinite', 'any_body') and not m['active']:
            p = m['mark']
            hits = []
            for s in data['rails']:
                slope = 1 if s[-1][1] > s[0][1] else -1
                if mode == 'infinite':
                    hit = p[1] - s[0][1] == slope * (p[0] - s[0][0])
                else:
                    hit = bool(set(m['points']) & set(s))
                if hit:
                    hits.append(slope)
            if len(hits) == 1:
                q = _destination(p, hits[0], data['systems'][m['color']])
                m['delta'] = q[0] - p[0], q[1] - p[1]
                m['trace'] = _line(p, -hits[0], data['grid'].shape)
        actions.append(m)
    if mode == 'only_A':
        # Remove B's gating, rails and traces altogether, retaining A's ordinary
        # boundary projection on every registered component.
        a = np.full(data['grid'].shape, data['bg'], np.int8)
        for p, v in data['fixed'].items():
            a[p] = v
        for m in actions:
            for (r, c), v in m['points'].items():
                p = r + m['delta'][0], c + m['delta'][1]
                if _inside(p, a.shape):
                    a[p] = v
        return a
    return _paint(data, actions, marker_only=mode == 'marker_only')


def _shape(rng, marker, payload, *, substantial=False):
    """Procedural polyominoes, bars, rectangles, frames and branches."""
    kind = int(rng.integers(6))
    if kind == 0:
        n = int(rng.integers(3 if substantial else 1, 12))
        cells = {(0, 0)}
        for _ in range(n - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRS} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    elif kind == 1:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 2:
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 3:
        n = int(rng.integers(3, 8))
        cells = {(r, 0) for r in range(n)}
        for r in rng.choice(n, size=int(rng.integers(1, min(n, 4) + 1)), replace=False):
            sign = int(rng.choice((-1, 1)))
            cells.update((int(r), sign * j) for j in range(1, int(rng.integers(2, 4))))
    elif kind == 4:
        n, k = int(rng.integers(2, 6)), int(rng.integers(2, 5))
        cells = {(r, 0) for r in range(n)} | {(n - 1, c) for c in range(k)}
    else:
        n = int(rng.integers(3, 8))
        cells = {(0, c) for c in range(n)}
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    ordered = sorted(cells)
    if substantial:
        ordered = [p for p in ordered if any(2 * p[k] != min(q[k] for q in cells) + max(q[k] for q in cells)
                                            for k in (0, 1))]
    mark = ordered[int(rng.integers(len(ordered)))]
    body = int(rng.choice(payload))
    colors = {p: body for p in cells}
    other = sorted(cells - {mark})
    if len(other) >= 4 and rng.random() < .45:
        for i in rng.choice(len(other), size=int(rng.integers(1, min(3, len(other)) + 1)), replace=False):
            colors[other[int(i)]] = int(rng.choice(payload))
    colors[mark] = marker
    return {(r - mark[0], c - mark[1]): colors[r, c] for r, c in sorted(cells)}


def _new_state(shape, bg, primary, systems):
    inp = np.full(shape, bg, np.int8)
    fixed = {}
    for color, system in systems.items():
        for p in system['anchors']:
            inp[p] = color
            fixed[p] = color
    return {'input': inp, 'bg': bg, 'primary': primary, 'systems': systems,
            'fixed': fixed, 'rails': [], 'union': set(), 'records': [], 'traces': {},
            'source': set(fixed), 'targets': set(fixed)}


def _add_rail(rng, state, slope=None, minimum=3):
    h, w = state['input'].shape
    endpoints = [p for s in state['rails'] for p in (s[0], s[-1])]
    marks = {m['mark'] for m in state['records']}
    for _ in range(180):
        sign = int(rng.choice((-1, 1))) if slope is None else slope
        maximum = min(h, w) - 2
        length = int(rng.integers(minimum, maximum + 1))
        d = length - 1
        r = int(rng.integers(1, h - d - 1))
        c = int(rng.integers(1, w - d - 1)) + (d if sign == -1 else 0)
        s = _segment((r, c), (r + d, c + sign * d))
        ends, cells = {s[0], s[-1]}, set(s)
        if _touch(ends, state['source']):
            continue
        if any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for p in ends for q in endpoints):
            continue
        if cells & (set(state['traces']) | marks) or ends & state['targets']:
            continue
        # Do not cover an existing primary endpoint with a new rail.
        if cells & set(endpoints):
            continue
        state['rails'].append(s)
        state['union'].update(cells)
        for p in ends:
            state['input'][p] = state['primary']
            state['fixed'][p] = state['primary']
        state['source'].update(ends)
        state['targets'].update(ends)
        return True
    return False


def _put_carrier(state, offsets, p, marker, active, slope, *, moving=False, clip=None,
                 feasible_alternative=False):
    shape = state['input'].shape
    points = {(p[0] + dr, p[1] + dc): v for (dr, dc), v in offsets.items()}
    # An edge singleton is visibly an anchor, so singleton cargo must be interior.
    if len(points) == 1 and (p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)):
        return False
    if not all(_inside(q, shape) for q in points) or _touch(set(points), state['source']):
        return False
    q = _destination(p, slope, state['systems'][marker]) if active else p
    if not _inside(q, shape):
        return False
    delta = q[0] - p[0], q[1] - p[1]
    if moving and delta == (0, 0):
        return False
    target = {(r + delta[0], c + delta[1]): v for (r, c), v in points.items()}
    visible = {q for q in target if _inside(q, shape)}
    clipped = len(visible) < len(points)
    if (clip is not None and clipped != clip) or visible & state['targets']:
        return False
    trace = _line(p, -slope, shape) if active else set()
    if active:
        if trace & state['union'] != {p}:
            return False
        if any(t in state['traces'] and state['traces'][t] != marker for t in trace):
            return False
        if any(t in state['fixed'] and state['fixed'][t] != marker for t in trace):
            return False
    elif p in state['union']:
        return False
    if feasible_alternative:
        alt = _destination(p, slope, state['systems'][marker])
        if not _inside(alt, shape) or alt == p:
            return False
    for q, v in points.items():
        state['input'][q] = v
    state['records'].append({'points': points, 'mark': p, 'color': marker,
                             'active': active, 'slope': slope if active else None,
                             'delta': delta, 'target': target, 'trace': trace, 'clipped': clipped})
    state['source'].update(points)
    state['targets'].update(visible)
    state['traces'].update({t: marker for t in trace})
    return True


def _active(rng, state, index, marker, payload, *, substantial=False, moving=False,
            clip=None, near=False):
    s = state['rails'][index]
    slope = 1 if s[-1][1] > s[0][1] else -1
    candidates = list(s[1:-1])
    if near:
        candidates = [s[1], s[-2]]
    rng.shuffle(candidates)
    for p in candidates:
        if sum(p in rail for rail in state['rails']) != 1:
            continue
        for _ in range(12):
            offsets = _shape(rng, marker, payload, substantial=substantial)
            if _put_carrier(state, offsets, p, marker, True, slope,
                            moving=moving, clip=clip):
                return True
    return False


def _inactive(rng, state, marker, payload, mode='free'):
    h, w = state['input'].shape
    # Choose shapes before positions so body-contact evidence is genuine geometry.
    for _ in range(35):
        offsets = _shape(rng, marker, payload, substantial=mode != 'free')
        candidates = []
        if mode == 'beyond':
            for s in state['rails']:
                slope = 1 if s[-1][1] > s[0][1] else -1
                for k in range(2, max(h, w)):
                    candidates.extend(((s[0][0] - k, s[0][1] - slope * k, slope),
                                       (s[-1][0] + k, s[-1][1] + slope * k, slope)))
        elif mode == 'body':
            for s in state['rails']:
                slope = 1 if s[-1][1] > s[0][1] else -1
                for r, c in s[1:-1]:
                    candidates.extend((r - dr, c - dc, slope) for dr, dc in offsets if (dr, dc) != (0, 0))
        else:
            candidates = [(int(i) // w, int(i) % w, 1) for i in rng.permutation(h * w)[:100]]
        rng.shuffle(candidates)
        for r, c, slope in candidates:
            p = r, c
            if not _inside(p, (h, w)) or p in state['union']:
                continue
            if mode == 'beyond':
                hits = sum(p[1] - s[0][1] == (1 if s[-1][1] > s[0][1] else -1) * (p[0] - s[0][0])
                           for s in state['rails'])
                if hits != 1:
                    continue
            if mode == 'body':
                hit_rails = [s for s in state['rails'] if any((r + dr, c + dc) in s for dr, dc in offsets)]
                if len(hit_rails) != 1:
                    continue
            if _put_carrier(state, offsets, p, marker, False, slope,
                            feasible_alternative=mode != 'free'):
                return True
    return False


def _render(state):
    """Render from construction records, without calling the input decoder."""
    data = {'grid': state['input'], 'bg': state['bg'], 'primary': state['primary'],
            'rails': state['rails'], 'fixed': state['fixed'], 'movers': state['records']}
    return {'input': state['input'], 'output': _paint(data)}


def _primary_noise(rng, state):
    """Optional isolated primary dots with no possible endpoint partner."""
    h, w = state['input'].shape
    dots = [p for p, v in state['fixed'].items() if v == state['primary']]
    desired = int(rng.choice((0, 0, 0, 1, 1, 2)))
    added = 0
    for index in rng.permutation((h - 2) * (w - 2)):
        if added == desired:
            break
        p = 1 + int(index) // (w - 2), 1 + int(index) % (w - 2)
        if p in state['union'] or p in state['traces'] or p in state['targets'] or _touch({p}, state['source']):
            continue
        if any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in dots):
            continue
        state['input'][p] = state['primary']
        state['fixed'][p] = state['primary']
        state['source'].add(p)
        state['targets'].add(p)
        dots.append(p)
        added += 1


def _systems(rng, shape, colors, first_axis, clip):
    result, occupied = {}, set()
    for i, color in enumerate(colors):
        axis = first_axis if i == 0 else int(rng.integers(2))
        size = shape[axis]
        for _ in range(60):
            guide = (1 if rng.random() < .5 else size - 2) if i == 0 and clip else int(rng.integers(1, size - 1))
            ends = (0, 1) if rng.random() < .3 else (int(rng.integers(2)),)
            anchors = ({(guide, 0 if e == 0 else shape[1] - 1) for e in ends} if axis == 0 else
                       {(0 if e == 0 else shape[0] - 1, guide) for e in ends})
            if not _touch(anchors, occupied):
                result[color] = {'axis': axis, 'guide': guide, 'anchors': anchors}
                occupied.update(anchors)
                break
        else:
            return None
    return result


def _scene(rng, palette, plan, spacious=False):
    lo, hi = plan['band']
    h, w = (hi, hi) if spacious else (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
    if not spacious and rng.random() < .18 and plan['rails'] <= 2:
        h = max(11, h - int(rng.integers(3, 7)))
    n = int(rng.choice((1, 1, 1, 2, 2, 3)))
    if min(h, w) <= 14:
        n = min(n, 2)
    bg, primary = palette[:2]
    markers, payload = palette[2:2 + n], palette[2 + n:]
    systems = _systems(rng, (h, w), markers, plan['axis'], plan['clip'])
    if systems is None:
        return None
    # Trial the first rail jointly with its launch; its boundary intersection
    # must be feasible before additional rails or inert objects consume space.
    for _ in range(100):
        state = _new_state((h, w), bg, primary, systems)
        if not _add_rail(rng, state, plan['slope'], minimum=6):
            continue
        if _active(rng, state, 0, markers[0], payload, substantial=True, moving=True,
                   clip=plan['clip'], near=plan['near']):
            break
    else:
        return None
    for _ in range(plan['rails'] - 1):
        if not _add_rail(rng, state):
            return None
    # All rails are completed, even if no cargo is placed on them. A teaching
    # plan reserves one rail untriggered; other activation counts vary freely.
    for index in rng.permutation(len(state['rails'])):
        index = int(index)
        if plan['zero'] and index == len(state['rails']) - 1:
            continue
        extra = int(rng.choice((0, 0, 1, 1, 2)))
        for _ in range(extra):
            marker = int(rng.choice(markers))
            _active(rng, state, index, marker, payload, clip=None if plan['allow_clip'] else False)
    if plan['inert'] and not _inactive(rng, state, markers[0], payload, plan['inert']):
        return None
    for marker in markers:
        if not any(m['color'] == marker for m in state['records']):
            if not _inactive(rng, state, marker, payload):
                return None
    for _ in range(int(rng.integers(4))):
        _inactive(rng, state, int(rng.choice(markers)), payload)
    _primary_noise(rng, state)
    if np.mean(state['input'] == bg) <= .5:
        return None
    pair = _render(state)
    # A required witness must actually refute its alternative, including when
    # foreground occlusion happens to conceal some of the construction.
    data = _decode(pair['input'])
    if plan['inert'] and np.array_equal(pair['output'], _alternative(data, 'infinite' if plan['inert'] == 'beyond' else 'any_body')):
        return None
    if np.array_equal(pair['output'], _alternative(data, 'no_boundary')):
        return None
    return pair


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _emergency_base(palette, plan):
    """A proved-capacity construction after free placement exhausts its budget.

    All current bands have maximum side at least 20. This square construction
    preserves the requested extent, rail count, active slope, guide axis, clipping
    and both negative witnesses. Only this emergency path uses fixed coordinates.
    """
    n = plan['band'][1]
    bg, primary, marker, body, decoration = palette[:5]
    guide = 1 if plan['clip'] else 10
    systems = {marker: {'axis': 0, 'guide': guide, 'anchors': {(guide, 0)}}}
    state = _new_state((n, n), bg, primary, systems)
    endpoints = (((5, 4), (12, 11)), ((2, 12), (5, 15)),
                 ((15, 3), (18, 6)), ((13, 17), (16, 14)))
    for a, b in endpoints[:plan['rails']]:
        s = _segment(a, b)
        state['rails'].append(s)
        state['union'].update(s)
        for p in (a, b):
            state['input'][p] = primary
            state['fixed'][p] = primary
            state['source'].add(p)
            state['targets'].add(p)
    offsets = ({(0, 0): marker, (-2, 0): body, (-1, 0): body, (-1, 1): decoration}
               if plan['clip'] else
               {(0, 0): marker, (0, 1): body, (-1, 1): decoration, (-1, 2): body})
    _require(_put_carrier(state, offsets, (8, 7), marker, True, 1,
                          moving=True, clip=plan['clip']), 'emergency launch geometry')
    p = (2, 1) if plan['clip'] else (14, 13)
    offsets = ({(0, 0): marker, (0, 1): body, (1, 1): decoration} if plan['clip'] else
               {(0, 0): marker, (0, -1): body, (1, -1): decoration})
    _require(_put_carrier(state, offsets, p, marker, False, 1, feasible_alternative=True),
             'emergency beyond-endpoint geometry')
    _require(_put_carrier(state, {(0, 0): marker, (0, -1): body, (1, -1): decoration},
                          (9, 10), marker, False, 1, feasible_alternative=True),
             'emergency body-contact geometry')
    pair = _render(state)
    dots = [p for p, v in state['fixed'].items() if v == primary]
    choices = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
               if (r, c) not in state['union'] and (r, c) not in state['traces']
               and (r, c) not in state['targets'] and not _touch({(r, c)}, state['source'])
               and all(abs(r - p[0]) != abs(c - p[1]) for p in dots)]
    _require(len(choices) >= 14, 'emergency input capacity')
    return pair, choices


def _fallback(palette, plan, used):
    pair, choices = _emergency_base(palette, plan)
    primary = palette[1]
    # At most thirteen inputs precede a pair. Fourteen eligible primary-dot
    # positions suffice even if every previous input came from this same base.
    for p in choices[:14]:
        candidate = {k: a.copy() for k, a in pair.items()}
        for a in candidate.values():
            a[p] = primary
        if plan['axis'] == 1:
            candidate = {k: a.T.copy() for k, a in candidate.items()}
        if plan['slope'] == -1:
            candidate = {k: np.fliplr(a).copy() for k, a in candidate.items()}
        if _key(candidate['input']) not in used:
            return candidate
    raise AssertionError('emergency capacity contradicted the maximum puzzle count')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .55)
    regime = int(rng.integers(4))
    axis, slope = int(rng.integers(2)), int(rng.choice((-1, 1)))
    clipping = bool(regime == 3 or rng.random() < .45)
    bands = (((14, 20), (22, 30)), ((22, 30), (13, 20)),
             ((19, 29), (24, 30)), ((16, 29), (16, 29)))[regime]
    plans = []
    for i in range(a + b):
        test = i >= a
        rails = int(rng.choice((1, 1, 2, 2, 3)))
        if regime == 2:
            rails = int(rng.integers(3, 5)) if test else int(rng.integers(1, 3))
        plan = {'band': bands[int(test)], 'axis': int(rng.integers(2)),
                'slope': int(rng.choice((-1, 1))), 'rails': rails, 'zero': False,
                'clip': bool(clipping and test and rng.random() < .5),
                'near': bool(test and regime != 3 and rng.random() < .3),
                'allow_clip': clipping, 'inert': None}
        plans.append(plan)
    # Two randomly ordered teaching witnesses work even at the legal minimum.
    slots = list(map(int, rng.choice(a, size=2, replace=False)))
    plans[slots[0]].update(axis=axis, slope=slope, clip=False, near=False, inert='beyond')
    plans[slots[1]].update(axis=1 - axis, slope=-slope, clip=clipping,
                           near=False, inert='body', zero=True, rails=2)
    if regime == 3:
        for plan in plans[a:]:
            plan.update(axis=axis, clip=True, near=False)
    puzzle, used = {'train': [], 'test': []}, set()
    for i, plan in enumerate(plans):
        colors = palette if stable else tuple(map(int, rng.permutation(10)))
        for attempt in range(45):
            pair = _scene(rng, colors, plan, spacious=attempt >= 35)
            if pair is not None and _key(pair['input']) not in used:
                break
        else:
            pair = _fallback(colors, plan, used)
        used.add(_key(pair['input']))
        puzzle['train' if i < a else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _holes(points):
    """Enclosed empty lattice cells, independent of rendered background colors."""
    points = set(points)
    r0, r1 = min(r for r, _ in points) - 1, max(r for r, _ in points) + 1
    c0, c1 = min(c for _, c in points) - 1, max(c for _, c in points) + 1
    empty = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - points
    todo, reached = [(r0, c0)], {(r0, c0)}
    while todo:
        r, c = todo.pop()
        for dr, dc in DIRS:
            p = r + dr, c + dc
            if p in empty and p not in reached:
                reached.add(p)
                todo.append(p)
    return empty - reached


def _measure(pair):
    d = _decode(pair['input'])
    movers = d['movers']
    h, w = d['grid'].shape
    endpoints = {p for s in d['rails'] for p in (s[0], s[-1])}
    clips = {side for m in movers for r, c in m['target']
             for side, hit in (('top', r < 0), ('bottom', r >= h), ('left', c < 0), ('right', c >= w)) if hit}
    return {'shape': tuple(d['grid'].shape), 'bg': d['bg'], 'primary': d['primary'],
            'systems': len(d['systems']), 'axes': tuple(sorted({s['axis'] for s in d['systems'].values()})),
            'rails': len(d['rails']), 'lengths': tuple(len(s) for s in d['rails']),
            'slopes': tuple(1 if s[-1][1] > s[0][1] else -1 for s in d['rails']),
            'cargo': len(movers), 'sizes': tuple(len(m['points']) for m in movers),
            'active': sum(m['active'] for m in movers), 'inactive': sum(not m['active'] for m in movers),
            'clips': sum(m['clipped'] for m in movers),
            'clip_edges': tuple(sorted(clips)),
            'bilateral': sum(len(s['anchors']) == 2 for s in d['systems'].values()),
            'primary_noise': sum(v == d['primary'] and p not in endpoints for p, v in d['fixed'].items()),
            'boundary_cargo': sum(any(r in (0, h - 1) or c in (0, w - 1) for r, c in m['points']) for m in movers),
            'holes': sum(bool(_holes(m['points'])) for m in movers),
            'hole_traces': sum(bool(_holes(m['target']) & m['trace']) for m in movers),
            'body_contact_inert': sum(not m['active'] and bool(set(m['points']) & d['union']) for m in movers),
            'triggers': tuple(sum(m['active'] and m['mark'] in s for m in movers) for s in d['rails']),
            'zero': sum(m['active'] and m['delta'] == (0, 0) for m in movers),
            'untriggered': sum(not any(m['active'] and m['mark'] in s for m in movers) for s in d['rails'])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    used, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas changed')
            _require(_key(inp) not in used, 'duplicate input')
            used.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(_paint(d), out), 'wrong rail-launch output')
            _require(not np.array_equal(inp, out), 'unchanged pair')
            for p, v in d['fixed'].items():
                _require(out[p] == v, 'fixed anchor/endpoint changed')
            for m in d['movers']:
                for p, v in m['target'].items():
                    if _inside(p, inp.shape):
                        _require(out[p] == v, 'cargo color/shape not preserved')
                if m['active']:
                    q = (m['mark'][0] + m['delta'][0], m['mark'][1] + m['delta'][1])
                    system = d['systems'][m['color']]
                    _require(q[system['axis']] == system['guide'], 'registration misses guide')
                    _require(m['delta'][1] == -m['slope'] * m['delta'][0], 'motion leaves launch diagonal')
                else:
                    _require(m['delta'] == (0, 0), 'inactive cargo moved')
            if split == 'train':
                training.append((d, out))
    _require({s['axis'] for d, _ in training for s in d['systems'].values()} == {0, 1}, 'teach both guide axes')
    _require({m['slope'] for d, _ in training for m in d['movers'] if m['active']} == {-1, 1}, 'teach both launch slopes')
    _require(any(m['active'] and m['delta'] != (0, 0) and not m['clipped'] and len(m['points']) >= 3
                 for d, _ in training for m in d['movers']), 'no complete cargo movement witness')
    _require(any(not any(m['active'] and m['mark'] in s for m in d['movers'])
                 for d, _ in training for s in d['rails']), 'no untriggered rail demonstration')
    for mode in ('no_boundary', 'only_A', 'orthogonal', 'infinite', 'any_body', 'marker_only'):
        _require(any(not np.array_equal(out, _alternative(d, mode)) for d, out in training),
                 'training fails to distinguish ' + mode)
    if any(m['clipped'] for p in puzzle['test'] for m in _decode(p['input'])['movers']):
        _require(any(m['clipped'] for d, _ in training for m in d['movers']), 'test clipping lacks a demonstration')


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _hand_case_checks():
    """Small coordinate-specified regressions, not generated or learned examples."""
    a = np.zeros((16, 18), np.int8)
    a[3, 3] = a[10, 10] = 1
    a[9, 17] = 2
    cargo = {(6, 6): 2, (6, 7): 3, (7, 7): 3, (7, 8): 4, (8, 7): 3}
    inert = {(12, 12): 2, (12, 13): 5, (13, 13): 5,
             (4, 5): 2, (4, 4): 5, (3, 5): 5}
    for p, v in {**cargo, **inert}.items():
        a[p] = v
    expected = np.zeros_like(a)
    for k in range(3, 11):
        expected[k, k] = 1
    for r in range(13):
        expected[r, 12 - r] = 2
    expected[9, 17] = 2
    for p, v in inert.items():
        expected[p] = v
    for (r, c), v in cargo.items():
        expected[r + 3, c - 3] = v
    _require(np.array_equal(solve(a), expected), 'hand transport/gating/layer regression')
    _require(expected[6, 7] == 0 and expected[7, 7] == 1 and expected[6, 6] == 2,
             'source erasure must reveal the appropriate substrate')
    for mode in ('no_boundary', 'only_A', 'orthogonal', 'infinite', 'any_body', 'marker_only'):
        _require(not np.array_equal(expected, _alternative(_decode(a), mode)), 'hand ablation ' + mode)
    # A hollow frame travels down/right on a +1 beam emitted from a -1 rail.
    frame = np.zeros((16, 18), np.int8)
    frame[3, 9] = frame[10, 2] = 1
    frame[9, 0] = 2
    for r in range(6, 9):
        for c in range(6, 9):
            if r in (6, 8) or c in (6, 8):
                frame[r, c] = 2 if (r, c) == (6, 6) else 3
    answer = solve(frame)
    _require(answer[9, 9] == 2 and answer[10, 10] == 2 and answer[10, 11] == 3,
             'transparent hole must reveal the trace after rigid translation')
    _require(answer[6, 7] == 0, 'frame source not erased')
    # Zero displacement is still a launch, so it still completes the trace.
    frame[9, 0], frame[6, 0] = 0, 2
    answer = solve(frame)
    _require(answer[7, 7] == 2 and answer[6, 7] == 3, 'zero displacement launch')
    # Clipping discards only the out-of-bounds material, retaining the handle.
    clipped = np.zeros((16, 18), np.int8)
    clipped[3, 3] = clipped[10, 10] = 1
    clipped[1, 17] = 2
    for p, v in {(6, 6): 2, (5, 6): 3, (4, 6): 3, (5, 7): 4}.items():
        clipped[p] = v
    answer = solve(clipped)
    _require(answer[1, 11] == 2 and answer[0, 11] == 3 and answer[0, 12] == 4,
             'boundary-clipping coordinates')
    _require(int(np.sum(answer == 3)) == 1, 'incorrect clipping amount')
    # Reject geometries whose visible information would require a new convention.
    for p, color, reason in (((14, 14), 1, 'ambiguous endpoint pairing'),
                             ((6, 7), 2, 'two registration cells'),
                             ((2, 10), 1, 'trace crossing a primary dot')):
        bad = a.copy()
        bad[p] = color
        try:
            solve(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted ' + reason)


def self_test():
    """Bounded 42-seed coverage plus semantic mutations and metamorphisms."""
    _hand_case_checks()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [_measure(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, 'puzzle diversity')
    _require(len({r['shape'] for r in rows}) >= 50, 'canvas diversity')
    _require({1, 2, 3, 4} <= {r['rails'] for r in rows}, 'rail count diversity')
    _require({1, 2, 3} <= {r['systems'] for r in rows}, 'registration count diversity')
    _require(any(r['clips'] for r in rows) and any(r['inactive'] for r in rows), 'boundary/gating coverage')
    for feature in ('zero', 'holes', 'hole_traces', 'boundary_cargo', 'bilateral', 'primary_noise', 'body_contact_inert'):
        _require(any(r[feature] for r in rows), 'missing structural variation: ' + feature)
    _require({edge for r in rows for edge in r['clip_edges']} == {'top', 'bottom', 'left', 'right'}, 'clip edge coverage')
    _require(any(max(r['triggers']) >= 2 for r in rows), 'multiple launches from one rail')
    _require(len({r['lengths'] for r in rows}) >= 40 and len({r['sizes'] for r in rows}) >= 100,
             'geometry diversity')
    _require({r['bg'] for r in rows} == set(range(10)), 'background palette diversity')
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), 'determinism')
    # Exhaust the discrete geometry of the emergency path: every allowed maximum
    # side, rail count, axis, slope and clipping state. All fourteen variants use
    # the same checked geometric isometries and independently safe unpaired dots.
    palette = tuple(range(10))
    for n in range(20, 31):
        for rails in range(1, 5):
            for clipped in (False, True):
                base_plan = {'band': (n, n), 'rails': rails, 'clip': clipped}
                base, choices = _emergency_base(palette, base_plan)
                _require(len(choices) >= 14, 'emergency distinct-input guarantee')
                for axis in (0, 1):
                    for slope in (-1, 1):
                        plan = dict(base_plan, axis=axis, slope=slope)
                        pair = _fallback(palette, plan, set())
                        _require(np.array_equal(solve(pair['input']), pair['output']), 'emergency solver agreement')
    used = set()
    plan = {'band': (20, 20), 'rails': 4, 'clip': True, 'axis': 1, 'slope': -1}
    for _ in range(14):
        pair = _fallback(palette, plan, used)
        used.add(_key(pair['input']))
    _require(len(used) == 14, 'emergency uniqueness')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, 'explicit counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(99, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Per-example palettes and D4 transformations preserve the visible rule.
    rng = np.random.default_rng(8726)
    for turns in (1, 2, 3):
        transformed = {'train': [], 'test': []}
        for split in transformed:
            for pair in puzzles[0][split]:
                colors = rng.permutation(10)
                transformed[split].append({k: colors[np.rot90(v, turns)].astype(np.int8)
                                           for k, v in pair.items()})
        validate_generated_puzzle(transformed)
    reflected = {s: [{k: np.fliplr(v).copy() for k, v in p.items()} for p in puzzles[1][s]]
                 for s in ('train', 'test')}
    validate_generated_puzzle(reflected)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(rows), 'distinct_shapes': len({r['shape'] for r in rows}),
            'distinct_cargo_size_vectors': len({r['sizes'] for r in rows})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
