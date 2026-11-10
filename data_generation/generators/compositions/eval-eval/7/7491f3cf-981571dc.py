"""Repair reflection orbits separately on the two sides of a control membrane.

Only standard-library modules and NumPy are used. Nothing is read from disk.
The solver has no access to seeds, examples, or the sampling routines.
"""
from __future__ import annotations

from collections import deque
import copy
import json
import numpy as np


FAMILIES = ("horizontal", "vertical", "diagonal", "elbow", "stepped")
REQUIRED_EVIDENCE = frozenset(("barrier", "horizontal", "vertical", "double",
                               "priority", "fallback", "nonlocal"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC colors")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, component = deque([start]), []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        result.append(component)
    return result


def _cut_info(cut):
    """Validate a digital path, then return regions and separated marker sites."""
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for r, c in sorted(cells):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if not (dr or dc) or q not in cells:
                    continue
                # No spurious diagonal chord across a right-angle elbow.
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p, ns in graph.items() if len(ns) == 1]
    _require(len(cells) >= 3 and len(ends) == 2 and
             all(1 <= len(ns) <= 2 for ns in graph.values()), "cut is not a simple path")
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends) and
             all(not border(p) for p in cells if p not in ends), "cut has extra boundary contacts")
    path, previous, current = [], None, min(ends)
    while current is not None and current not in path:
        path.append(current)
        onward = [q for q in graph[current] if q != previous]
        previous, current = current, onward[0] if onward else None
    _require(len(path) == len(cells), "cut is disconnected or has a loop")
    parts = _components(~cut)
    _require(len(parts) == 2, "cut must leave two regions")
    labels = np.full((h, w), -1, dtype=np.int8)
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    sites = []
    for index, part in enumerate(parts):
        for p in part:
            labels[p] = index
        sites.append([p for p in part if not near[p]])
    _require(all(sites), "both regions must admit an isolated marker")
    return labels, sites, path


def _orbits(h, w):
    return [tuple(sorted({(r, c), (h - 1 - r, c), (r, w - 1 - c),
                          (h - 1 - r, w - 1 - c)}))
            for r in range((h + 1) // 2) for c in range((w + 1) // 2)]


def _orbit_parts(labels):
    result = []
    for orbit in _orbits(*labels.shape):
        groups = [[p for p in orbit if labels[p] == side] for side in (0, 1)]
        seam = [p for p in orbit if labels[p] == -1]
        result.append((orbit, groups, seam))
    return result


def _decode(value):
    inp = _grid(value)
    height, width = inp.shape
    _require((width - 3) % 2 == 0, "canvas must contain two equal panels")
    h, w = height - 2, (width - 3) // 2
    _require(5 <= h <= 28 and 5 <= w <= 13, "panel dimensions outside this family")
    frame = int(inp[0, 0])
    _require(bool(np.all(inp[[0, -1], :] == frame)) and
             bool(np.all(inp[:, [0, w + 1, -1]] == frame)), "broken frame or divider")
    control, mosaic = inp[1:-1, 1:w + 1], inp[1:-1, w + 2:-1]
    cv, mv = set(map(int, np.unique(control))), set(map(int, np.unique(mosaic)))
    _require(len(cv) == 2 and frame not in cv and frame not in mv, "invalid panel colors")
    common = cv & mv
    _require(len(common) == 1, "control background must visibly encode the missing color")
    erase = next(iter(common))
    ink = next(iter(cv - {erase}))
    _require(ink not in mv and 2 <= len(mv - {erase}) <= 7, "invalid payload roles")
    components = sorted(_components(control == ink, True), key=len)
    _require(len(components) == 2 and len(components[0]) == 1 and len(components[1]) > 1,
             "control must have a cut and one isolated dot")
    marker = components[0][0]
    cut = np.zeros((h, w), dtype=bool)
    for p in components[1]:
        cut[p] = True
    labels, sites, path = _cut_info(cut)
    side = int(labels[marker])
    _require(side in (0, 1) and marker in sites[side], "dot touches the cut")
    return dict(input=inp, control=control, mosaic=mosaic, frame=frame, erase=erase,
                ink=ink, cut=cut, labels=labels, sites=sites, marker=marker,
                side=side, path=path, parts=_orbit_parts(labels))


def _witness_values(data, groups):
    result = []
    for group in groups:
        values = {int(data['mosaic'][p]) for p in group if data['mosaic'][p] != data['erase']}
        _require(not group or len(values) == 1, "empty or contradictory region/orbit witnesses")
        result.append(next(iter(values)) if values else None)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, region, and donor solely from the supplied input."""
    data = _decode(input_grid)
    field, erase, marked = data['mosaic'], data['erase'], data['side']
    repaired = field.copy()
    for orbit, groups, seam in data['parts']:
        values = _witness_values(data, groups)
        for side, group in enumerate(groups):
            for p in group:
                if field[p] == erase:
                    repaired[p] = values[side]
        seam_value = values[marked] if groups[marked] else values[1 - marked]
        if seam_value is None:
            _require(all(field[p] != erase for p in seam) and
                     len({int(field[p]) for p in seam}) == 1, "cut-only orbit must be intact")
        else:
            for p in seam:
                _require(field[p] == erase or field[p] == seam_value, "visible seam has the wrong owner")
                if field[p] == erase:
                    repaired[p] = seam_value
    output = data['input'].copy()
    w = field.shape[1]
    output[1:-1, w + 2:-1] = repaired
    return output


def _line(start, end):
    """Integer Bresenham segment, for a thin readable diagonal partition."""
    r, c = start
    y, x = end
    dr, dc = abs(y - r), abs(x - c)
    sr, sc = (1 if y > r else -1), (1 if x > c else -1)
    error, points = dc - dr, []
    while True:
        points.append((r, c))
        if (r, c) == end:
            return points
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _cut_family(path):
    rows, cols = zip(*path)
    if len(set(rows)) == 1:
        return 'horizontal'
    if len(set(cols)) == 1:
        return 'vertical'
    if path == _line(path[0], path[-1]) or path[::-1] == _line(path[-1], path[0]):
        return 'diagonal'
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])]
    return 'elbow' if sum(a != b for a, b in zip(steps, steps[1:])) == 1 else 'stepped'


def _sample_cut(rng, h, w, family):
    for _ in range(100):
        cut = np.zeros((h, w), dtype=bool)
        if family == 'horizontal':
            cut[int(rng.integers(2, h - 2)), :] = True
        elif family == 'vertical':
            cut[:, int(rng.integers(2, w - 2))] = True
        elif family == 'elbow':
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        elif family == 'diagonal':
            across = bool(rng.integers(2))
            ah, aw = (w, h) if across else (h, w)
            a, b = map(int, rng.choice(aw, 2, replace=False))
            points = _line((0, a), (ah - 1, b))
            for r, c in points:
                cut[(c, r) if across else (r, c)] = True
        elif family == 'stepped':
            across = bool(rng.integers(2))
            ah, aw = (w, h) if across else (h, w)
            trial = np.zeros((ah, aw), dtype=bool)
            available = list(range(2, ah - 2, 2))
            count = int(rng.integers(1, min(3, len(available)) + 1))
            rows = sorted(map(int, rng.choice(available, count, replace=False)))
            col, last = int(rng.integers(1, aw - 1)), 0
            for row in rows:
                trial[last:row + 1, col] = True
                choices = [c for c in range(1, aw - 1) if abs(c - col) >= 2]
                following = choices[int(rng.integers(len(choices)))]
                trial[row, min(col, following):max(col, following) + 1] = True
                col, last = following, row
            trial[last:, col] = True
            cut = trial.T.copy() if across else trial
        else:
            raise ValueError('unknown cut family')
        if rng.integers(2):
            cut = cut[::-1].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        try:
            labels, sites, path = _cut_info(cut)
            if _cut_family(path) == family:
                return cut, labels, sites
        except ValueError:
            pass
    raise RuntimeError('cut sampling exhausted')


def _evidence_plan(rng, parts):
    """Find a compact input geometry capable of teaching all donor operations."""
    seam_both = [i for i, (_, gs, ss) in enumerate(parts) if ss and all(gs)]
    seam_one = [i for i, (_, gs, ss) in enumerate(parts) if ss and sum(bool(g) for g in gs) == 1]
    double = [(i, s, p, q) for i, (_, gs, _) in enumerate(parts) for s, g in enumerate(gs)
              if gs[1 - s] for p in g for q in g if p[0] != q[0] and p[1] != q[1]]
    if not seam_both or not seam_one or not double:
        return None
    d = double[int(rng.integers(len(double)))]
    used, selected = {d[0]}, [d]
    for dimension in (0, 1):
        candidates = [(i, s, p, q) for i, (_, gs, _) in enumerate(parts) if i not in used
                      for s, g in enumerate(gs) for p in g for q in g
                      if p[dimension] != q[dimension] and p[1 - dimension] == q[1 - dimension]]
        if not candidates:
            return None
        choice = candidates[int(rng.integers(len(candidates)))]
        selected.append(choice)
        used.add(choice[0])
    return selected, seam_both[int(rng.integers(len(seam_both)))], seam_one[int(rng.integers(len(seam_one)))]


def _palette(rng, base=None):
    if base is not None:
        return base
    colors = list(map(int, rng.permutation(10)))
    k = int(rng.integers(2, 8))
    return colors[0], colors[1], colors[2], tuple(colors[3:3 + k])


def _material(rng, parts, palette, h, w):
    """Orbit-indexed textures: clean motifs, structured bands, and busy fields."""
    rc = np.array([orbit[0] for orbit, _, _ in parts], dtype=int)
    r, c = rc[:, 0], rc[:, 1]
    k, style = len(palette), int(rng.integers(8))
    bh, bw = map(int, rng.integers(1, 4, size=2))
    phase = int(rng.integers(k))
    if style == 0:
        indices = rng.integers(k, size=len(parts))
    elif style == 1:
        coarse = rng.integers(k, size=((h + 1) // 2, (w + 1) // 2))
        indices = coarse[r // bh, c // bw]
    elif style == 2:
        indices = (r // bh + (c // bw) // 2 + phase) % k
    elif style == 3:
        indices = ((r + int(rng.choice([-2, -1, 1, 2])) * c) // bw + phase) % k
    elif style == 4:
        cr, cc = int(rng.integers((h + 1) // 2)), int(rng.integers((w + 1) // 2))
        indices = (np.maximum(abs(r - cr), abs(c - cc)) // bh + phase) % k
    elif style == 5:
        indices = ((r if rng.integers(2) else c) // bh + phase) % k
    elif style == 6:
        # Reflect a solid, hollow, cross, or diamond fragment, without added speckles.
        cr, cc = int(rng.integers((h + 1) // 2)), int(rng.integers((w + 1) // 2))
        radius = int(rng.integers(1, max(2, min(h, w) // 3)))
        shape = int(rng.integers(4))
        distance = abs(r - cr) + abs(c - cc)
        mask = ((abs(r - cr) <= radius) & (abs(c - cc) <= radius) if shape == 0 else
                (np.maximum(abs(r - cr), abs(c - cc)) == radius if shape == 1 else
                 ((r == cr) | (c == cc) if shape == 2 else distance <= radius)))
        indices = (phase + mask.astype(int)) % k
    else:
        indices = np.full(len(parts), phase, dtype=int)
        for _ in range(int(rng.integers(2, 6))):
            r0, c0 = int(rng.integers((h + 1) // 2)), int(rng.integers((w + 1) // 2))
            rh, cw = map(int, rng.integers(1, 5, size=2))
            inside = (r >= r0) & (r < r0 + rh) & (c >= c0) & (c < c0 + cw)
            indices[inside] = int(rng.integers(k))
    return np.asarray(palette, dtype=np.int8)[indices]


def _raw_damage(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(6))
    if kind == 0:
        return rng.random((h, w)) < float(rng.uniform(.08, .45))
    for _ in range(int(rng.integers(1, 5))):
        rh = int(rng.integers(1, max(3, h // 2 + 1)))
        cw = int(rng.integers(1, max(3, w // 2 + 1)))
        if kind == 1:
            rh = int(rng.integers(1, 3))
        elif kind == 2:
            cw = int(rng.integers(1, 3))
        r0, c0 = int(rng.integers(h - rh + 1)), int(rng.integers(w - cw + 1))
        if kind in (1, 2, 3):
            mask[r0:r0 + rh, c0:c0 + cw] = True
        elif kind == 4:
            for dr in range(rh):
                reach = 1 + dr * (cw - 1) // max(1, rh - 1)
                mask[r0 + dr, c0:c0 + reach] = True
        else:
            r, c = r0, c0
            for _ in range(rh * cw):
                mask[r, c] = True
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                r, c = max(0, min(h - 1, r + dr)), max(0, min(w - 1, c + dc))
    return mask


def _full_mosaic(parts, material, marked, seam_colors, shape):
    full = np.empty(shape, dtype=np.int8)
    for i, (_, groups, seam) in enumerate(parts):
        for side, group in enumerate(groups):
            for p in group:
                full[p] = material[side][i]
        value = (material[marked][i] if groups[marked] else
                 (material[1 - marked][i] if groups[1 - marked] else seam_colors[i]))
        for p in seam:
            full[p] = value
    return full


def _pack(cut, marker, field, palette):
    frame, erase, ink, _ = palette
    h, w = cut.shape
    canvas = np.full((h + 2, 2 * w + 3), frame, dtype=np.int8)
    control = np.full((h, w), erase, dtype=np.int8)
    control[cut], control[marker] = ink, ink
    canvas[1:-1, 1:w + 1], canvas[1:-1, w + 2:-1] = control, field
    return canvas


def _new_pairs(rng, family, palette, initial=False, must_barrier=False):
    for _ in range(160):
        h = int(rng.integers(7, 14 if initial else 29))
        w = int(rng.integers(7, 14))
        try:
            cut, labels, sites = _sample_cut(rng, h, w, family)
        except RuntimeError:
            continue
        parts = _orbit_parts(labels)
        plan = _evidence_plan(rng, parts) if initial else None
        if initial and plan is None:
            continue
        material = [_material(rng, parts, palette[3], h, w) for _ in range(2)]
        seam_colors = rng.choice(palette[3], size=len(parts)).astype(np.int8)
        mask = _raw_damage(rng, h, w)
        protected = {}
        if initial:
            selections, seam_both, seam_one = plan
            # Different materials must be visibly distinguishable at key orbits.
            for i in (selections[0][0], seam_both):
                other_colors = [c for c in palette[3] if c != material[0][i]]
                material[1][i] = int(rng.choice(other_colors))
            for i, side, target, survivor in selections:
                protected[i, side] = survivor
                for p in parts[i][1][side]:
                    mask[p] = p != survivor
            for i, (_, groups, seam) in enumerate(parts):
                if all(groups) and material[0][i] != material[1][i]:
                    for p in seam:
                        mask[p] = True
            for p in parts[seam_one][2]:
                mask[p] = True
        # A region-orbit subgroup always has a witness; seam-only orbits are intact.
        for i, (_, groups, seam) in enumerate(parts):
            for side, group in enumerate(groups):
                if not group:
                    continue
                if (i, side) in protected:
                    mask[protected[i, side]] = False
                elif all(mask[p] for p in group):
                    mask[group[int(rng.integers(len(group)))]] = False
            if not any(groups):
                for p in seam:
                    mask[p] = False
        if not mask.any():
            possible = [p for _, groups, seam in parts for group in groups if len(group) > 1 for p in group]
            possible += [p for _, groups, seam in parts if any(groups) for p in seam]
            if not possible:
                continue
            mask[possible[int(rng.integers(len(possible)))]] = True
        first_side = int(rng.integers(2))
        pairs = []
        for side in (first_side, 1 - first_side) if initial else (first_side,):
            marker = sites[side][int(rng.integers(len(sites[side])))]
            full = _full_mosaic(parts, material, side, seam_colors, (h, w))
            damaged = full.copy()
            damaged[mask] = palette[1]
            inp = _pack(cut, marker, damaged, palette)
            out = _pack(cut, marker, full, palette)
            if len(set(map(int, np.unique(full)))) < 2:
                break
            pairs.append({'input': inp, 'output': out})
        if len(pairs) != (2 if initial else 1):
            continue
        if initial:
            if not np.array_equal(pairs[0]['input'][1:-1, w + 2:-1], pairs[1]['input'][1:-1, w + 2:-1]):
                raise AssertionError('counterfactual mosaics differ')
            evidence = _evidence(pairs[0]['input']) | _evidence(pairs[1]['input'])
            if not REQUIRED_EVIDENCE <= evidence:
                continue
        elif must_barrier and 'barrier' not in _evidence(pairs[0]['input']):
            continue
        return pairs
    raise RuntimeError('readable, informative pair construction exhausted')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _evidence(value):
    data = _decode(value)
    result = set()
    field, labels, erase = data['mosaic'], data['labels'], data['erase']
    h, w = field.shape
    visible_by_side = [np.argwhere((labels == side) & (field != erase)) for side in (0, 1)]
    for _, groups, seam in data['parts']:
        values = _witness_values(data, groups)
        if any(field[p] == erase for p in seam):
            if all(groups) and values[0] != values[1]:
                result.add('priority')
            if not groups[data['side']] and groups[1 - data['side']]:
                result.add('fallback')
        for side, group in enumerate(groups):
            for r, c in group:
                if field[r, c] != erase:
                    continue
                if groups[1 - side] and values[side] != values[1 - side]:
                    result.add('barrier')
                horizontal = h - 1 - r, c
                vertical = r, w - 1 - c
                diagonal = h - 1 - r, w - 1 - c
                usable = lambda p: p != (r, c) and labels[p] == side and field[p] != erase
                hh, vv, dd = usable(horizontal), usable(vertical), usable(diagonal)
                if hh and not vv:
                    result.add('horizontal')
                if vv and not hh:
                    result.add('vertical')
                if dd and not hh and not vv:
                    result.add('double')
                visible = visible_by_side[side]
                distances = np.abs(visible - np.array([r, c])).sum(axis=1)
                nearest = visible[distances == distances.min()]
                if all(field[tuple(p)] != values[side] for p in nearest):
                    result.add('nonlocal')
    return result


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    held_out = FAMILIES[int(rng.integers(len(FAMILIES)))]
    initial_choices = [f for f in ('diagonal', 'elbow', 'stepped') if f != held_out]
    initial_family = initial_choices[int(rng.integers(len(initial_choices)))]
    base = _palette(rng)
    fixed_palette = bool(rng.random() < .35)
    puzzle = {'train': _new_pairs(rng, initial_family, base, initial=True), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    training_families = [f for f in FAMILIES if f != held_out]
    for split, count in (('train', nt - 2), ('test', nq)):
        for _ in range(count):
            for attempt in range(80):
                family = held_out if split == 'test' else training_families[int(rng.integers(len(training_families)))]
                palette = _palette(rng, base if fixed_palette else None)
                pair = _new_pairs(rng, family, palette, must_barrier=split == 'test')[0]
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique-pair construction exhausted')
    order = rng.permutation(nt)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, evidence, meaningful_sides = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "input/output shape mismatch")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            data = _decode(inp)
            h, w = data['mosaic'].shape
            changes = inp != out
            intended = np.zeros_like(changes)
            intended[1:-1, w + 2:-1] = data['mosaic'] == data['erase']
            _require(intended.any() and np.array_equal(changes, intended), "changed visible/map/frame cells")
            _require(not np.any(out[1:-1, w + 2:-1] == data['erase']), "unrepaired cells")
            _require(np.array_equal(solve(inp), out), "incorrect repair")
            # Explicit output invariants: each region's orbit fragment is uniform;
            # the seam copies the selected region, not a vote over the whole orbit.
            repaired = out[1:-1, w + 2:-1]
            for _, groups, seam in data['parts']:
                values = []
                for group in groups:
                    colors = {int(repaired[p]) for p in group}
                    _require(not group or len(colors) == 1, "nonuniform repaired material")
                    values.append(next(iter(colors)) if colors else None)
                seam_value = values[data['side']] if groups[data['side']] else values[1 - data['side']]
                if seam_value is not None:
                    _require(all(repaired[p] == seam_value for p in seam), "wrong seam priority")
            if split == 'train':
                flags = _evidence(inp)
                evidence |= flags
                if 'priority' in flags:
                    meaningful_sides.add(data['side'])
            else:
                _require('barrier' in _evidence(inp), "test must require region-filtered reflection")
    _require(REQUIRED_EVIDENCE <= evidence, "training does not teach the full rule")
    _require(meaningful_sides == {0, 1}, "both effective seam selections must be demonstrated")


def _transform_pair(pair, operation):
    data = _decode(pair['input'])
    w = data['mosaic'].shape[1]
    transformed = {}
    for key in ('input', 'output'):
        control = operation(np.asarray(pair[key])[1:-1, 1:w + 1])
        field = operation(np.asarray(pair[key])[1:-1, w + 2:-1])
        h2, w2 = field.shape
        canvas = np.full((h2 + 2, 2 * w2 + 3), data['frame'], dtype=np.int8)
        canvas[1:-1, 1:w2 + 1], canvas[1:-1, w2 + 2:-1] = control, field
        transformed[key] = canvas
    return transformed


def _must_reject(function):
    try:
        function()
    except (ValueError, AssertionError):
        return
    raise AssertionError('an invalid case was accepted')


def self_test():
    """Bounded structural, semantic-invariant, and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, cuts, roles, mask_counts, parities = set(), set(), set(), set(), set()
    all_families, selected_sides, payload_sizes = set(), set(), set()
    material_pairs, signatures = set(), set()
    fixed_palettes = mixed_palettes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        families = {'train': set(), 'test': set()}
        local_roles = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                saved = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, saved)
                assert np.array_equal(solve(inp.tolist()), out)
                data = _decode(inp)
                h, w = data['mosaic'].shape
                family = _cut_family(data['path'])
                families[split].add(family)
                all_families.add(family)
                selected_sides.add((family, data['side']))
                shapes.add((h, w))
                cuts.add((h, w, data['cut'].tobytes()))
                parities.add((h % 2, w % 2))
                role = data['frame'], data['erase'], data['ink']
                roles.add(role)
                local_roles.add(role)
                mask_counts.add(int((data['mosaic'] == data['erase']).sum()))
                payload_sizes.add(len(set(map(int, np.unique(data['mosaic']))) - {data['erase']}))
                material_pairs.add((h, w, out[1:-1, w + 2:-1].tobytes()))
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                signatures.add((inp.shape, normalized))
        assert not families['train'] & families['test'], 'held-out cut family leaked'
        fixed_palettes += len(local_roles) == 1
        mixed_palettes += len(local_roles) > 1
        # Locate the controlled twin from actual grids, irrespective of train order.
        twins = []
        for i, a in enumerate(puzzle['train']):
            da = _decode(a['input'])
            for b in puzzle['train'][i + 1:]:
                db = _decode(b['input'])
                if (da['mosaic'].shape == db['mosaic'].shape and da['side'] != db['side'] and
                        np.array_equal(da['cut'], db['cut']) and np.array_equal(da['mosaic'], db['mosaic'])):
                    twins.append((a, b))
        assert twins, 'controlled dot demonstration missing'
        a, b = twins[0]
        d = _decode(a['input'])
        w = d['mosaic'].shape[1]
        different = a['output'][1:-1, w + 2:-1] != b['output'][1:-1, w + 2:-1]
        assert different.any() and np.all(~different | d['cut'])
        for operation in (lambda x: x[::-1].copy(), lambda x: x[:, ::-1].copy(), lambda x: x.T.copy()):
            transformed = _transform_pair(a, operation)
            assert np.array_equal(solve(transformed['input']), transformed['output'])
        # Arbitrary bijective recoloring preserves roles and predictions.
        perm = np.random.default_rng(100).permutation(10)
        assert np.array_equal(solve(perm[a['input']]), perm[a['output']])
    assert all_families == set(FAMILIES)
    assert len(selected_sides) == 10 and len(parities) == 4
    assert len(shapes) >= 65 and len(cuts) >= 100 and len(roles) >= 70
    assert len(mask_counts) >= 35 and len(material_pairs) >= 150 and len(signatures) >= 180
    assert fixed_palettes >= 5 and mixed_palettes >= 15 and payload_sizes == set(range(2, 8))
    assert any(h == 28 for h, w in shapes) and any(w == 13 for h, w in shapes)
    # Exercise every legal explicit count combination, with deterministic replay.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == nq
            validate_generated_puzzle(puzzle)
    assert _puzzle_key(generate_puzzle(1234, 10, 4)) == _puzzle_key(generate_puzzle(1234, 10, 4))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        _must_reject(lambda nt=nt, nq=nq: generate_puzzle(0, nt, nq))
    # A rule-level minimum-size case, outside the informative sampler's 7-cell minimum.
    # The centered cut has intact cut-only orbits and an intact axis-intersection singleton.
    small_cut = np.zeros((5, 5), dtype=bool)
    small_cut[:, 2] = True
    small_full = np.ones((5, 5), dtype=np.int8)
    small_full[:, 3:] = 2
    small_full[[0, 4], 2] = 2
    small_damage = small_full.copy()
    small_damage[0, 0] = small_damage[4, 1] = small_damage[1, 4] = 0
    small_palette = 9, 0, 8, (1, 2)
    small_input = _pack(small_cut, (0, 0), small_damage, small_palette)
    assert np.array_equal(solve(small_input), _pack(small_cut, (0, 0), small_full, small_palette))
    for kind in ('gap', 'visible', 'frame'):
        bad = copy.deepcopy(puzzles[0])
        pair = bad['test'][0]
        data = _decode(pair['input'])
        w = data['mosaic'].shape[1]
        if kind == 'frame':
            p = 0, 0
        else:
            mask = data['mosaic'] == data['erase']
            local = tuple(map(int, np.argwhere(mask if kind == 'gap' else ~mask)[0]))
            p = local[0] + 1, local[1] + w + 2
        pair['output'][p] = (int(pair['output'][p]) + 1) % 10
        _must_reject(lambda: validate_generated_puzzle(bad))
    # Topology and separation negatives use readable, otherwise ordinary maps.
    a = puzzles[0]['train'][0]
    d = _decode(a['input'])
    # Missing whole groups, inconsistent material witnesses, and contradictory
    # visible seams are invalid inputs, rather than ambiguous tie-breaking cases.
    w = d['mosaic'].shape[1]
    group = next(g for _, groups, _ in d['parts'] for g in groups if len(g) >= 2)
    all_missing = a['input'].copy()
    for r, c in group:
        all_missing[r + 1, c + w + 2] = d['erase']
    _must_reject(lambda: solve(all_missing))
    payload = sorted(set(map(int, np.unique(d['mosaic']))) - {d['erase']})
    contradictory = a['input'].copy()
    for p, color in zip(group[:2], payload[:2]):
        contradictory[p[0] + 1, p[1] + w + 2] = color
    _must_reject(lambda: solve(contradictory))
    wrong_seam = a['input'].copy()
    seam_point = next(p for _, groups, seam in d['parts'] if any(groups) for p in seam)
    actual = int(a['output'][seam_point[0] + 1, seam_point[1] + w + 2])
    wrong_seam[seam_point[0] + 1, seam_point[1] + w + 2] = next(c for c in payload if c != actual)
    _must_reject(lambda: solve(wrong_seam))
    control = d['control'].copy()
    control[d['marker']] = d['erase']
    touching = next(tuple(map(int, p)) for p in np.argwhere(~d['cut'])
                    if np.any(d['cut'][max(0, p[0] - 1):p[0] + 2, max(0, p[1] - 1):p[1] + 2]))
    control[touching] = d['ink']
    malformed = a['input'].copy()
    malformed[1:-1, 1:d['mosaic'].shape[1] + 1] = control
    _must_reject(lambda: solve(malformed))
    for style in ('branch', 'loop', 'three_regions'):
        cut = np.zeros((9, 9), dtype=bool)
        if style == 'branch':
            cut[:, 4] = True
            cut[4, 4:7] = True
        elif style == 'loop':
            cut[2:7, 2] = cut[2:7, 6] = True
            cut[2, 2:7] = cut[6, 2:7] = True
        else:
            cut[:, 2] = cut[:, 6] = True
        _must_reject(lambda cut=cut: _cut_info(cut))
    return {'seeds_checked': 42, 'interior_shapes': len(shapes), 'cut_geometries': len(cuts),
            'structural_signatures': len(signatures), 'palette_role_tuples': len(roles),
            'material_mosaics': len(material_pairs), 'hole_counts': len(mask_counts),
            'fold_parities': len(parities), 'cut_families': sorted(all_families),
            'fixed_palette_puzzles': fixed_palettes, 'mixed_palette_puzzles': mixed_palettes,
            'semantic_verification': 'generator-owned checks only'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
