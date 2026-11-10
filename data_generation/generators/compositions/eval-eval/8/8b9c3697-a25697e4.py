"""Fork molds: whole-key fitting, axial insertion, and silhouette trails.

No parent module, filesystem evidence, seed, or training label is used by solve.
Ground truth is constructed from chosen geometry, then reconstructed from input.
"""
from __future__ import annotations

import json
import numpy as np

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        found, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(frozenset(found))
    return result


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _center2(cells):
    a, b, c, d = _bbox(cells)
    return a + b, c + d


def _rect(box):
    a, b, c, d = box
    return frozenset((r, q) for r in range(a, b + 1) for q in range(c, d + 1))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _raw(p, op):
    r, c = p
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _transform(regions, op):
    return tuple(frozenset(_raw(p, op) for p in region) for region in regions)


def _signature(cells):
    a, _, c, _ = _bbox(cells)
    return tuple(sorted((r - a, q - c) for r, q in cells))


def _canonical(cells):
    return min(_signature(_transform((cells,), op)[0]) for op in range(8))


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _valid_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("invalid ARC grid")
    return grid


def _parse_receiver(cells):
    parses = []
    for out in DIRS:
        perp = (-out[1], out[0])
        ka = min(r * out[0] + c * out[1] for r, c in cells)
        qa = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - ka,
                  r * perp[0] + c * perp[1] - qa) for r, c in cells}
        width = max(q for _, q in local) + 1
        height = max(k for k, _ in local)
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [max(k for k, q in local if q == j) for j in range(width)]
        expected = {(k, q) for q, n in enumerate(lengths) for k in range(n + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((start, q, n))
        if (len(runs) != 3 or runs[0][0] != 0 or runs[-1][1] != width
                or any(runs[j][1] >= runs[j + 1][0] for j in (0, 1))
                or runs[1][2] >= max(runs[0][2], runs[2][2])):
            continue
        box = _bbox(cells)
        rectangle = _rect(box)
        slot = rectangle - cells
        if not slot or len(_components(slot)) != 1:
            continue
        parses.append(dict(cells=cells, out=out, perp=perp, ka=ka, qa=qa,
                           width=width, height=height, runs=tuple(runs),
                           rectangle=rectangle, slot=slot))
    if len(parses) != 1:
        raise ValueError("receiver must have one positive three-prong parse")
    return parses[0]


def _inspect(input_grid):
    grid = _valid_grid(input_grid)
    if np.any(grid == 0):
        raise ValueError("black is reserved for trails and absent from inputs")
    colors, counts = np.unique(grid, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely modal")
    bg = int(colors[np.argmax(counts)])
    receivers, keys = [], []
    for cells in _components(map(tuple, np.argwhere(grid != bg))):
        palette = tuple(sorted({int(grid[p]) for p in cells}))
        if len(palette) == 1:
            receiver = _parse_receiver(cells)
            receiver['color'] = palette[0]
            receivers.append(receiver)
        elif len(palette) == 2:
            regions = tuple(frozenset(p for p in cells if int(grid[p]) == color)
                            for color in palette)
            if any(len(_components(region)) != 1 for region in regions):
                raise ValueError("each key region must be connected")
            keys.append(dict(cells=cells, colors=palette, regions=regions))
        else:
            raise ValueError("input objects must be monochrome molds or two-colored keys")
    if not receivers or not keys:
        raise ValueError("a scene needs receivers and keys")
    return dict(grid=grid, bg=bg, receivers=receivers, keys=keys)


def _candidate_moves(analysis, ri, *, axis=True):
    """Axis=False is a documented B-only counterfactual, not used by solve."""
    grid = analysis['grid']
    rec = analysis['receivers'][ri]
    fixed = frozenset(p for r in analysis['receivers'] for p in r['cells'])
    out = rec['out']
    center = _center2(rec['slot'])
    slot_signature = _signature(rec['slot'])
    sa, _, sc, _ = _bbox(rec['slot'])
    moves = {}
    for ki, key in enumerate(analysis['keys']):
        for pi in (0, 1):
            source_center = _center2(key['regions'][pi])
            delta2 = (source_center[0] - center[0], source_center[1] - center[1])
            d2 = delta2[0] * out[0] + delta2[1] * out[1]
            if axis and (delta2 != (d2 * out[0], d2 * out[1]) or d2 < 2 or d2 % 2):
                continue
            distance = d2 // 2 if axis else 0
            for op in range(8):
                transformed = _transform(key['regions'], op)
                if _signature(transformed[pi]) != slot_signature:
                    continue
                pa, _, pc, _ = _bbox(transformed[pi])
                absolute = tuple(_shift(region, sa - pa, sc - pc) for region in transformed)
                other = absolute[1 - pi]
                # The attachment exits the open ends, not a side or the base.
                if any(r * out[0] + c * out[1] <= rec['ka'] + rec['height']
                       for r, c in other):
                    continue
                full = absolute[0] | absolute[1]
                path = frozenset((r + t * out[0], c + t * out[1])
                                 for r, c in full for t in range(distance + 1))
                if path & fixed or any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
                                       for r, c in path):
                    continue
                # Exact shape fitting plus center alignment proves that the
                # prepared source pose is a lattice-preserving pivot transform.
                palette = key['colors'][::-1] if op >= 4 else key['colors']
                placed = tuple(sorted((p, palette[j]) for j in (0, 1) for p in absolute[j]))
                trail = path - full
                signature = placed, tuple(sorted(trail))
                if signature not in moves:
                    moves[signature] = dict(ri=ri, ki=ki, pi=pi, op=op, distance=distance,
                                            regions=absolute, full=full, path=path,
                                            trail=trail, placed=placed)
    return list(moves.values())


def _resolve(input_grid):
    analysis = _inspect(input_grid)
    moves, used = [], set()
    for ri in range(len(analysis['receivers'])):
        options = _candidate_moves(analysis, ri)
        if len(options) > 1:
            raise ValueError("receiver has competing colored insertions")
        if options:
            move = options[0]
            if move['ki'] in used:
                raise ValueError("one key would serve multiple receivers")
            used.add(move['ki'])
            moves.append(move)
    if not moves:
        raise ValueError("generated scenes must have at least one active mold")
    for i, move in enumerate(moves):
        for other in moves[i + 1:]:
            if move['full'] & other['path'] or other['full'] & move['path']:
                raise ValueError("a final key overlaps another insertion")
    output = np.full(analysis['grid'].shape, analysis['bg'], dtype=np.int64)
    for rec in analysis['receivers']:
        _paint(output, rec['cells'], rec['color'])
    for move in moves:
        _paint(output, move['trail'], 0)
    for move in moves:
        for p, color in move['placed']:
            output[p] = color
    return output, analysis, moves


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every pose, assignment, color exchange, and trail from this input."""
    return _resolve(input_grid)[0]


def _profile(rng, *, small=False, symmetric=False, quarter=False):
    for _ in range(100):
        thickness = 1 if small or symmetric else int(rng.integers(1, 3))
        right_thickness = thickness if small or symmetric else int(rng.integers(1, 3))
        span = int(rng.integers(1, 3 if small else 4))
        left_gap = int(rng.integers(1, 3))
        right_gap = left_gap if symmetric else int(rng.integers(1, 3))
        width = thickness + right_thickness + left_gap + span + right_gap
        height = int(rng.integers(3 if small else 4, 5 if small else 7))
        left = right = height
        if not symmetric:
            if rng.integers(2):
                left = int(rng.integers(1, height + 1))
            else:
                right = int(rng.integers(1, height + 1))
        central = int(rng.integers(1, height))
        q0 = thickness + left_gap
        rec = {(0, q) for q in range(width)}
        for a, b, n in ((0, thickness, left), (q0, q0 + span, central),
                         (width - right_thickness, width, right)):
            rec.update((k, q) for k in range(1, n + 1) for q in range(a, b))
        rec = frozenset(rec)
        slot = _rect((0, height, 0, width - 1)) - rec
        a, b, c, d = _bbox(slot)
        if quarter and (b - a - d + c) % 2:
            continue
        orbit = len({_signature(_transform((slot,), op)[0]) for op in range(8)})
        if symmetric and orbit != 4:
            continue
        if not symmetric and orbit != 8:
            continue
        return rec, slot, height, width
    raise RuntimeError("profile sampling exhausted")


def _handle(rng, slot, height, width, *, short=False, wing=False):
    if wing:
        q = min(c for r, c in slot if r == height)
        end = -int(rng.integers(1, 4))
        return frozenset({(height + 1, c) for c in range(end, q + 1)}
                         | {(height + k, end) for k in range(2, int(rng.integers(3, 5)))})
    start_q = int(rng.choice(sorted(c for r, c in slot if r == height)))
    cells = {(height + 1, start_q)}
    style = int(rng.integers(3))
    target_size = int(rng.integers(2, 7 if short else min(25, len(slot) + 8)))
    max_depth = 3 if short else int(rng.integers(3, 6))
    tip = next(iter(cells))
    for _ in range(target_size - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRS
                           if height < r + dr <= height + max_depth
                           and -2 <= c + dc <= width + 1} - cells)
        weights = []
        for r, c in frontier:
            neighbors = sum((r + dr, c + dc) in cells for dr, dc in DIRS)
            if style == 0:
                weights.append(neighbors ** 2 / (1 + abs(c - start_q) + r - height))
            elif style == 1:
                weights.append((8 if abs(r - tip[0]) + abs(c - tip[1]) == 1 else .1)
                               / neighbors ** 2)
            else:
                weights.append(1 / neighbors)
        if not frontier:
            break
        weights = np.asarray(weights)
        tip = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
        cells.add(tip)
    return frozenset(cells)


def _source_pose(regions, slot, op):
    source = _transform(regions, INVERSE[op])
    a, b = _center2(source[0])
    c, d = _center2(slot)
    if (c - a) % 2 or (d - b) % 2:
        raise ValueError("quarter-turn needs a lattice-preserving center")
    return tuple(_shift(s, (c - a) // 2, (d - b) // 2) for s in source)


def _system(rng, *, small=False, symmetric=False, op=None, wing=False, active=True):
    if op is None:
        op = int(rng.integers(8))
    rec, slot, height, width = _profile(rng, small=small, symmetric=symmetric,
                                       quarter=op in (1, 3, 5, 7))
    handle = _handle(rng, slot, height, width, short=small, wing=wing)
    final = (slot, handle)
    source0 = _source_pose(final, slot, op)
    source_full = source0[0] | source0[1]
    distance = max(height + 2 - min(r for r, _ in source_full), height + 1)
    distance += int(rng.integers(1, 4))
    source = tuple(_shift(s, distance, 0) for s in source0)
    path = frozenset((r + t, c) for r, c in slot | handle for t in range(distance + 1))
    return dict(rec=rec, slot=slot, final=final, source=source, source0=source0,
                distance=distance, op=op, height=height, width=width, active=active,
                path=path if active else frozenset(), decoys=[])


def _wrong_region(rng, slot):
    box = _bbox(slot)
    additions = sorted(_rect(box) - slot)
    removals = sorted(slot)
    rng.shuffle(additions)
    rng.shuffle(removals)
    target = _canonical(slot)
    for removed in removals:
        smaller = slot - {removed}
        if not smaller or _bbox(smaller) != box or len(_components(smaller)) != 1:
            continue
        for added in additions:
            changed = smaller | {added}
            if len(_components(changed)) == 1 and _canonical(changed) != target:
                return frozenset(changed)
    raise ValueError("no same-area same-box contour mutation")


def _add_wrong_axis_decoy(rng, system):
    wrong = _wrong_region(rng, system['slot'])
    h = system['height']
    q = min(c for r, c in system['slot'] if r == h)
    handle = frozenset({(h + 1, q), (h + 2, q)})
    decoy_distance = h + 1
    decoy = tuple(_shift(s, decoy_distance, 0) for s in (wrong, handle))
    decoy_max = max(r for s in decoy for r, _ in s)
    source_min = min(r for s in system['source0'] for r, _ in s)
    distance = max(system['distance'], decoy_max + 2 - source_min)
    system['source'] = tuple(_shift(s, distance, 0) for s in system['source0'])
    system['distance'] = distance
    system['path'] = frozenset((r + t, c) for s in system['final'] for r, c in s
                               for t in range(distance + 1))
    system['decoys'].append(decoy)


def _normalize_system(system, op):
    pieces = [system['rec'], system['path']]
    if system['active']:
        pieces.extend(system['source'])
    pieces.extend(s for d in system['decoys'] for s in d)
    union = frozenset(p for s in pieces for p in s)
    transformed = _transform((union,), op)[0]
    a, b, c, d = _bbox(transformed)
    def convert(cells):
        return _shift(_transform((cells,), op)[0], -a, -c) if cells else frozenset()
    return dict(rec=convert(system['rec']), slot=convert(system['slot']),
                final=tuple(convert(s) for s in system['final']),
                source=tuple(convert(s) for s in system['source']) if system['active'] else (),
                path=convert(system['path']), op=system['op'], active=system['active'],
                decoys=[tuple(convert(s) for s in ds) for ds in system['decoys']],
                shape=(b - a + 1, d - c + 1))


def _colors(rng, bg, stable_palette=None):
    available = [c for c in range(1, 10) if c != bg]
    if stable_palette is not None:
        return stable_palette
    return tuple(map(int, rng.choice(available, 3, replace=False)))


def _render(systems, origins, shape, bg, rng, palette=None):
    inp = np.full(shape, bg, dtype=np.int64)
    out = inp.copy()
    for system, (dr, dc) in zip(systems, origins):
        fc, pc, hc = _colors(rng, bg, palette)
        rec = _shift(system['rec'], dr, dc)
        _paint(inp, rec, fc)
        _paint(out, rec, fc)
        if system['active']:
            final = tuple(_shift(s, dr, dc) for s in system['final'])
            path = _shift(system['path'], dr, dc)
            _paint(out, path - final[0] - final[1], 0)
            colors = (hc, pc) if system['op'] >= 4 else (pc, hc)
            for region, color in zip(final, colors):
                _paint(out, region, color)
            for region, color in zip(system['source'], (pc, hc)):
                _paint(inp, _shift(region, dr, dc), color)
        for decoy in system['decoys']:
            # Distinct local palettes make selection visibly consequential.
            options = [c for c in range(1, 10) if c not in (bg, fc, pc, hc)]
            dc1, dc2 = map(int, rng.choice(options, 2, replace=False))
            for region, color in zip(decoy, (dc1, dc2)):
                _paint(inp, _shift(region, dr, dc), color)
    return {'input': inp, 'output': out}


def _witness_pair(rng, index, bg, palette=None):
    for _ in range(80):
        try:
            if index == 0:
                system = _system(rng, symmetric=True, op=6, wing=True)
                source_min = min(c for s in system['source'] for _, c in s)
                source_max = max(c for s in system['source'] for _, c in s)
                lateral = source_max - source_min + int(rng.integers(2, 5))
                system['decoys'].append(tuple(_shift(s, 0, -lateral) for s in system['source']))
            else:
                system = _system(rng, small=True, op=int(rng.choice([1, 3])))
                _add_wrong_axis_decoy(rng, system)
            normalized = _normalize_system(system, 0)
            h, w = normalized['shape']
            pad_top, pad_bottom, pad_left = map(int, rng.integers(0, 3, size=3))
            # The symmetric witness keeps the receiver's long right guard on
            # the boundary; the wrong reflected wing must be clipped there.
            pad_right = 0 if index == 0 else int(rng.integers(0, 3))
            shape = h + pad_top + pad_bottom, w + pad_left + pad_right
            if max(shape) > 30:
                continue
            pair = _render([normalized], [(pad_top, pad_left)], shape, bg, rng, palette)
            turn = int(rng.integers(4))
            pair = {k: np.rot90(v, turn).copy() for k, v in pair.items()}
            predicted, analysis, moves = _resolve(pair['input'])
            if (not np.array_equal(predicted, pair['output']) or len(moves) != 1
                    or not _witness_features(analysis, moves)[index]):
                continue
            return pair
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError("required witness construction exhausted")


def _pack(rng, systems):
    shapes = [s['shape'] for s in systems]
    for attempt in range(30):
        h = int(rng.integers(max(18, max(s[0] for s in shapes)), 31)) if attempt < 20 else 30
        w = int(rng.integers(max(18, max(s[1] for s in shapes)), 31)) if attempt < 20 else 30
        occupied = set()
        origins = []
        for bh, bw in shapes:
            if bh > h or bw > w:
                break
            choices = [(r, c) for r in range(h - bh + 1) for c in range(w - bw + 1)]
            rng.shuffle(choices)
            for r, c in choices:
                box = _rect((r, r + bh - 1, c, c + bw - 1))
                if box & occupied:
                    continue
                origins.append((r, c))
                occupied.update((rr + dr, cc + dc) for rr, cc in box
                                for dr, dc in ((0, 0),) + DIRS)
                break
            else:
                break
        if len(origins) == len(systems):
            return origins, (h, w)
    return None


def _ordinary_pair(rng, count, bg, palette=None, *, inactive=False):
    for attempt in range(100):
        try:
            small = count >= 3 or attempt >= 30
            raw = [_system(rng, small=small, active=not (inactive and i == count - 1))
                   for i in range(count)]
            # Independently oriented systems when feasible; a common direction
            # is a packing aid for three long insertion routes, not a rule.
            operations = [int(rng.integers(4)) for _ in raw] if count <= 2 else [0] * count
            systems = [_normalize_system(s, op) for s, op in zip(raw, operations)]
            if any(max(s['shape']) > 30 for s in systems):
                continue
            packed = _pack(rng, systems)
            if packed is None:
                continue
            origins, shape = packed
            pair = _render(systems, origins, shape, bg, rng, palette)
            # Remove variable unused margins based on the combined actual ink.
            used = set(map(tuple, np.argwhere((pair['input'] != bg) | (pair['output'] != bg))))
            a, b, c, d = _bbox(used)
            pads = list(map(int, rng.integers(0, 4, size=4)))
            a, b = max(0, a - pads[0]), min(shape[0] - 1, b + pads[1])
            c, d = max(0, c - pads[2]), min(shape[1] - 1, d + pads[3])
            if rng.random() < .8:
                pair = {k: v[a:b + 1, c:d + 1].copy() for k, v in pair.items()}
            turn = int(rng.integers(4))
            pair = {k: np.rot90(v, turn).copy() for k, v in pair.items()}
            predicted, analysis, moves = _resolve(pair['input'])
            expected_active = count - int(inactive)
            if (np.array_equal(predicted, pair['output']) and len(moves) == expected_active
                    and len(analysis['receivers']) == count):
                return pair
        except (ValueError, RuntimeError):
            continue
    # A constructive fallback uses two/three compact equal-orientation systems.
    # It obeys the same rule and remains structurally randomized.
    for _ in range(80):
        try:
            raw = [_system(rng, small=True, op=0, active=not (inactive and i == count - 1))
                   for i in range(count)]
            systems = [_normalize_system(s, 0) for s in raw]
            h = max(s['shape'][0] for s in systems)
            w = sum(s['shape'][1] for s in systems) + count - 1
            if max(h, w) > 30:
                continue
            origins, c = [], 0
            for system in systems:
                origins.append((int(rng.integers(h - system['shape'][0] + 1)), c))
                c += system['shape'][1] + 1
            pair = _render(systems, origins, (h, w), bg, rng, palette)
            turn = int(rng.integers(4))
            pair = {k: np.rot90(v, turn).copy() for k, v in pair.items()}
            if np.array_equal(solve(pair['input']), pair['output']):
                return pair
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _is_integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (not _is_integer(num_train) or not _is_integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("counts require 2..10 training, 1..4 test, and train > test")
    if not _is_integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .25)
    background = int(rng.integers(1, 10))
    palette = _colors(rng, background) if stable else None
    result = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(12):
                bg = background if stable else int(rng.integers(1, 10))
                if split == 'train' and index < 2:
                    pair = _witness_pair(rng, index, bg, palette)
                else:
                    n = int(rng.integers(1, 3)) if split == 'train' else int(rng.integers(2, 4))
                    inactive = bool(n > 1 and rng.random() < .20 and (split == 'train' or n == 3))
                    pair = _ordinary_pair(rng, n, bg, palette, inactive=inactive)
                key = _array_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retries exhausted")
    rng.shuffle(result['train'])
    return result


def _array_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _witness_features(analysis, moves):
    axis_witness = contour_witness = False
    used = {m['ki'] for m in moves}
    for move in moves:
        rec = analysis['receivers'][move['ri']]
        center = _center2(rec['slot'])
        out = rec['out']
        box = _bbox(rec['slot'])
        dims = box[1] - box[0] + 1, box[3] - box[2] + 1
        for ki, key in enumerate(analysis['keys']):
            if ki in used:
                continue
            for region in key['regions']:
                c2 = _center2(region)
                diff = c2[0] - center[0], c2[1] - center[1]
                d2 = diff[0] * out[0] + diff[1] * out[1]
                aligned = diff == (d2 * out[0], d2 * out[1]) and d2 >= 2 and d2 % 2 == 0
                same_shape = _canonical(region) == _canonical(rec['slot'])
                if same_shape and not aligned:
                    # A B-only fitting check really permits this distinct key.
                    if any(m['ki'] == ki for m in _candidate_moves(analysis, move['ri'], axis=False)):
                        axis_witness = True
                b = _bbox(region)
                region_dims = b[1] - b[0] + 1, b[3] - b[2] + 1
                if aligned and region_dims == dims and len(region) == len(rec['slot']) and not same_shape:
                    contour_witness = True
    return axis_witness, contour_witness


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_parities = set(), set()
    witnesses = [False, False]
    symmetric_resolved = False
    train_counts, test_counts = [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (_valid_grid(pair[k]) for k in ('input', 'output'))
            assert inp.shape == out.shape
            key = _array_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, analysis, moves = _resolve(inp)
            assert np.array_equal(expected, out), "incorrect assembled colors or swept silhouette"
            assert not np.array_equal(inp, out) and np.any(out == 0)
            (train_counts if split == 'train' else test_counts).append(len(analysis['receivers']))
            for rec in analysis['receivers']:
                assert all(int(out[p]) == rec['color'] for p in rec['cells'])
                assert min(run[2] for run in rec['runs']) >= 1
            for move in moves:
                rec = analysis['receivers'][move['ri']]
                assert move['regions'][move['pi']] == rec['slot']
                assert move['distance'] >= 1 and move['trail']
                assert all(out[p] == 0 for p in move['trail'])
                assert all(out[p] == c for p, c in move['placed'])
                if split == 'train':
                    train_parities.add(int(move['op'] >= 4))
                    orbit = len({_signature(_transform((rec['slot'],), op)[0]) for op in range(8)})
                    symmetric_resolved |= orbit < 8
            if split == 'train':
                features = _witness_features(analysis, moves)
                witnesses = [old or new for old, new in zip(witnesses, features)]
    assert train_parities == {0, 1}, "both color policies need demonstrations"
    assert all(witnesses), "training needs off-axis congruence and aligned contour-mismatch witnesses"
    assert symmetric_resolved, "training needs whole-key resolution of a symmetric cavity"
    assert min(train_counts) == 1 and max(train_counts) <= 2
    assert min(test_counts) >= 2 and max(test_counts) <= 3


def self_test():
    """A bounded 42-seed invariant/diversity audit, with semantic negatives."""
    shapes, profiles, slots, handles, puzzles = set(), set(), set(), set(), set()
    directions, operations, backgrounds, counts, distances, relations = set(), set(), set(), set(), set(), set()
    silhouette_differences = inactive = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_array_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, analysis, moves = _resolve(pair['input'])
                shapes.add(pair['input'].shape)
                backgrounds.add(analysis['bg'])
                counts.add(len(analysis['receivers']))
                inactive += len(analysis['receivers']) - len(moves)
                for rec in analysis['receivers']:
                    directions.add(rec['out'])
                    profiles.add((rec['width'], rec['runs']))
                    slots.add(_canonical(rec['slot']))
                for move in moves:
                    operations.add(move['op'])
                    distances.add(move['distance'])
                    pi = move['pi']
                    plug, handle = move['regions'][pi], move['regions'][1 - pi]
                    handles.add(_canonical(handle))
                    relations.add(int(np.sign(len(plug) - len(handle))))
                    # Explicit alternatives: only plug motion or the whole
                    # bounding box would yield different black footprints.
                    outdir = analysis['receivers'][move['ri']]['out']
                    plug_path = frozenset((r + t * outdir[0], c + t * outdir[1])
                                          for r, c in plug for t in range(move['distance'] + 1))
                    plug_trail = plug_path - move['full']
                    box_trail = _rect(_bbox(move['path'])) - move['full']
                    silhouette_differences += (plug_trail != move['trail'] and box_trail != move['trail'])
    assert len(puzzles) == 42 and len(shapes) >= 30
    assert len(profiles) >= 40 and len(slots) >= 35 and len(handles) >= 50
    assert directions == set(DIRS) and operations == set(range(8))
    assert backgrounds == set(range(1, 10)) and counts == {1, 2, 3}
    assert len(distances) >= 7 and {-1, 0, 1} <= relations
    assert silhouette_differences >= 42 and inactive > 0
    a, b = generate_puzzle(701, 3, 2), generate_puzzle(701, 3, 2)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    reordered = {'train': list(reversed(a['train'])), 'test': a['test']}
    validate_generated_puzzle(reordered)
    # The host verifier exercises every legal count. Keep this audit bounded
    # by checking the minimal and maximal requests plus an interior request.
    for nt, ne in ((2, 1), (10, 4), (3, 2)):
        p = generate_puzzle(11000 + 10 * nt + ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
        validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in a[s]] for s in a}
    _, _, moves = _resolve(damaged['test'][0]['input'])
    damaged['test'][0]['output'][min(moves[0]['trail'])] = 1
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted silhouette accepted")
    # A cavity's mirror symmetry is genuinely ambiguous when its wing has
    # room on both sides. An edge-resolved fit must not become a tie-break.
    rng = np.random.default_rng(218)
    symmetric = _normalize_system(_system(rng, symmetric=True, op=6, wing=True), 0)
    h, w = symmetric['shape']
    ambiguous = _render([symmetric], [(1, 5)], (h + 2, w + 10), 9, rng)
    try:
        solve(ambiguous['input'])
    except ValueError as exc:
        assert 'competing colored insertions' in str(exc)
    else:
        raise AssertionError("ambiguous symmetric cavity accepted")
    no_center = frozenset({(0, q) for q in range(7)}
                          | {(k, q) for k in range(1, 5) for q in (0, 6)})
    try:
        _parse_receiver(no_center)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length central prong accepted")
    return dict(seeds=42, distinct_puzzles=len(puzzles), grid_shapes=len(shapes),
                receiver_profiles=len(profiles), cavity_shapes=len(slots), attachment_shapes=len(handles),
                operations=sorted(operations), distances=list(map(int, sorted(distances))), area_relationships=sorted(relations),
                inactive_receivers=inactive, silhouette_ablation_witnesses=silhouette_differences,
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
