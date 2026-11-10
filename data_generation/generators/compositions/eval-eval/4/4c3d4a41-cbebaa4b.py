"""Pin-constrained docking of rigid, multi-tooth cams into an anchored press.

The input-only solver enumerates geometric docking assignments and admits only
assemblies in which every pin can contact a tooth by moving toward the roof.
Generation uses separately constructed placements and a direct pin formula.
"""
from itertools import permutations, product
import json

import numpy as np


HOST_SEEDS = (0, 19986467, 791550260, 934282926, 632111050, 14477442,
              77689767, 76606485, 301558882, 242996505, 896085432, 423796699,
              450343048, 925178464, 168608374, 789556850, 587293164, 354353747,
              955616003, 827399252, 554187319, 189333174, 179355600, 312357760,
              762581336, 804019956, 49946074, 530714863, 425894644, 370252381,
              267212661, 429281475, 429373158, 51443029, 659861483, 384809238,
              571159928, 52152138, 32579462, 236929474, 690730999, 300107116)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    left, result = set(cells), []
    while left:
        start = min(left)
        left.remove(start)
        component, queue = {start}, [start]
        while queue:
            r, c = queue.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    component.add(q)
                    queue.append(q)
        result.append(component)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _frame(t, l, b, r):
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(a, l) for a in range(t, b + 1)}
            | {(a, r) for a in range(t, b + 1)})


def _read_cam(a, cells, marker):
    stamps = {q for q in cells if a[q] == marker}
    body = cells - stamps
    colors = {int(a[q]) for q in body}
    assert len(colors) == 1 and len(stamps) == 2
    t, l, b, r = _box(body)
    w = r - l + 1
    assert 3 <= w <= 7 and len(_components(body)) == 1
    bases = [row for row in range(t, b + 1)
             if {(row, c) for c in range(l, r + 1)} <= body]
    assert len(bases) == 1
    base = bases[0]
    stamp_list = sorted(stamps, key=lambda q: q[1])
    assert stamp_list[0][1] < stamp_list[1][1]
    assert all(base < row <= base + 3 and l <= col <= r
               for row, col in stamps)
    teeth, rebuilt = {}, {(base, c) for c in range(l, r + 1)}
    for col in range(l, r + 1):
        rows = sorted(row for row, c in body if c == col and row < base)
        if rows:
            assert rows == list(range(rows[0], base))
            teeth[col] = rows[0]
            rebuilt.update((row, col) for row in rows)
    assert 2 <= len(teeth) <= 3
    assert all(y - x >= 2 for x, y in zip(sorted(teeth), sorted(teeth)[1:]))
    for row, col in stamps:
        rebuilt.update((q, col) for q in range(base + 1, row))
    assert rebuilt == body
    return dict(body=body, stamps=stamps, teeth=teeth, color=colors.pop(),
                base=base, width=w, heights=tuple(base - teeth[c] for c in sorted(teeth)))


def _read_canonical(a):
    values, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    cc = _components(map(tuple, np.argwhere(a != bg).tolist()))
    assert 3 <= len(cc) <= 5
    housing = []
    for i, cells in enumerate(cc):
        t, l, b, r = _box(cells)
        if b - t < 6 or r - l < 8:
            continue
        edge = _frame(t, l, b, r)
        color = int(a[t, l])
        if edge <= cells and all(a[q] == color for q in edge):
            housing.append((i, t, l, b, r, color))
    assert len(housing) == 1
    ai, t, l, b, r, frame_color = housing[0]
    common = set.intersection(*[{int(a[q]) for q in cells} for cells in cc])
    assert len(common) == 1
    marker = common.pop()
    assert marker not in (bg, frame_color)
    anchor = cc[ai]
    stamps = sorted((q for q in anchor if a[q] == marker), key=lambda q: q[1])
    n = len(cc) - 1
    assert len(stamps) == 2 * n and len({c for row, c in stamps}) == 2 * n
    ports = [set(stamps[2*i:2*i+2]) for i in range(n)]
    for i, port in enumerate(ports):
        ordered = sorted(port, key=lambda q: q[1])
        assert 1 <= ordered[1][1] - ordered[0][1] <= 3
        assert abs(ordered[1][0] - ordered[0][0]) <= 2
        if i:
            assert min(c for row, c in port) - max(c for row, c in ports[i-1]) >= 4
    fixed_body = {q for q in anchor if a[q] == frame_color}
    reconstructed = _frame(t, l, b, r)
    for row, col in stamps:
        assert t + 3 < row < b and l < col < r
        reconstructed.update((q, col) for q in range(row + 1, b))
    assert fixed_body == reconstructed
    pin_cells = anchor - fixed_body - set(stamps)
    assert pin_cells
    pins = {}
    for col in sorted({c for row, c in pin_cells}):
        rows = sorted(row for row, c in pin_cells if c == col)
        assert rows == list(range(t + 1, rows[-1] + 1))
        assert rows[-1] < b - 2 and l < col < r
        pins[col] = tuple(int(a[row, col]) for row in rows)
    cams = [_read_cam(a, cells, marker) for i, cells in enumerate(cc) if i != ai]
    # Scattered pieces are wholly outside the closed housing and do not encode
    # an order. Payload colors cannot masquerade as pin or housing colors.
    pin_colors = {v for seq in pins.values() for v in seq}
    assert all(p['color'] not in pin_colors | {bg, marker, frame_color} for p in cams)
    assert all(not any(t <= row <= b and l <= col <= r for row, col in p['body'] | p['stamps'])
               for p in cams)
    return dict(bg=bg, marker=marker, frame_color=frame_color, box=(t, l, b, r),
                fixed_body=fixed_body, stamps=set(stamps), ports=ports, pins=pins, cams=cams)


def _dock_options(f, match_geometry=True):
    """Marker geometry fixes translation; pin feasibility filters whole cams."""
    t, l, b, r = f['box']
    choices = {}
    for slot, port in enumerate(f['ports']):
        for i, cam in enumerate(f['cams']):
            options = []
            tried = set()
            port_points = sorted(port)
            cam_points = sorted(cam['stamps'])
            for p in port_points:
                for q in cam_points:
                    delta = p[0] - q[0], p[1] - q[1]
                    if delta in tried:
                        continue
                    tried.add(delta)
                    if match_geometry and _shift(cam['stamps'], delta) != port:
                        continue
                    body = _shift(cam['body'], delta)
                    if any(not (t < row < b and l < col < r) for row, col in body):
                        continue
                    if body & (f['fixed_body'] | f['stamps']):
                        continue
                    teeth = {col + delta[1]: row + delta[0]
                             for col, row in cam['teeth'].items()}
                    if not set(teeth) <= set(f['pins']):
                        continue
                    capacities = {col: row - t - 1 for col, row in teeth.items()}
                    if min(capacities.values()) < 1:
                        continue
                    contact = all(len(f['pins'][col]) >= cap for col, cap in capacities.items())
                    options.append(dict(index=i, delta=delta, body=body, teeth=teeth,
                                        capacities=capacities, contact=contact))
            # A finite two-point set has no nonzero translational symmetry.
            if match_geometry:
                assert len(options) <= 1
            choices[slot, i] = options
    return choices


def _assemblies(f, require_contact=True, match_geometry=True):
    choices, answers = _dock_options(f, match_geometry), []
    for order in permutations(range(len(f['cams']))):
        for candidates in product(*(choices[slot, i] for slot, i in enumerate(order))):
            chosen, bodies, lanes = [], set(), set()
            for candidate in candidates:
                if ((require_contact and not candidate['contact'])
                        or bodies & candidate['body'] or lanes & set(candidate['teeth'])):
                    break
                chosen.append(candidate)
                bodies |= candidate['body']
                lanes |= set(candidate['teeth'])
            else:
                if lanes == set(f['pins']):
                    answers.append(chosen)
    return answers


def _render_solution(a, f, placement):
    # Simulate actual ordered pin cells moving one cell at a time. The direct
    # construction formula in _make_pair is deliberately separate.
    out = np.full(a.shape, f['bg'], dtype=np.int8)
    for q in f['fixed_body']:
        out[q] = f['frame_color']
    for q in f['stamps']:
        out[q] = f['marker']
    roof = f['box'][0]
    for chosen in placement:
        cam = f['cams'][chosen['index']]
        for q in chosen['body']:
            out[q] = cam['color']
        for col, tooth_top in chosen['teeth'].items():
            cells = [(roof + 1 + j, value) for j, value in enumerate(f['pins'][col])]
            while cells[-1][0] >= tooth_top:
                cells = [(row - 1, value) for row, value in cells if row - 1 > roof]
                assert cells
            assert cells[-1][0] == tooth_top - 1
            for row, value in cells:
                out[row, col] = value
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    readings = []
    for k in range(4):
        canonical = np.rot90(a, k).copy()
        try:
            f = _read_canonical(canonical)
            answers = _assemblies(f)
            grids = {_key(_render_solution(canonical, f, x)): x for x in answers}
            assert len(grids) == 1
        except AssertionError:
            continue
        placement = next(iter(grids.values()))
        readings.append((k, canonical, f, placement))
    if len(readings) != 1:
        raise ValueError("input must specify one pin-feasible geometric assembly")
    return readings[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    k, a, f, placement = _analyze(input_grid)
    return np.rot90(_render_solution(a, f, placement), -k).copy()


def _cam(width, cols, heights, stamp_cols, dy, color):
    body = {(0, c) for c in range(width)}
    for col, h in zip(cols, heights):
        body.update((row, col) for row in range(-h, 0))
    offsets = (max(0, -dy), max(0, dy))
    stamps = {(1 + o, c) for o, c in zip(offsets, stamp_cols)}
    for row, col in stamps:
        body.update((a, col) for a in range(1, row))
    return dict(body=body, stamps=stamps, color=color, heights=tuple(heights),
                cols=tuple(cols), width=width, dy=dy)


def _make_pair(rng, n, roles, witness=0, high=4):
    bg, frame_color, marker, body_a, body_b, pin_a, pin_b = roles
    # Always repeat a geometric signature in the first two modules. Remaining
    # modules may reuse it or introduce another equal-cardinality signature.
    teeth_count = 2 if witness == 1 or n == 4 else int(rng.integers(2, 4))
    min_width = 2 * teeth_count - 1
    max_width = min(7, (26 - 3 * (n - 1)) // n)
    width = int(rng.integers(min_width, max_width + 1))
    tooth_offset = int(rng.integers(width - min_width + 1))
    cols = tuple(tooth_offset + 2 * i for i in range(teeth_count))
    dx = int(rng.integers(1, min(3, width - 1) + 1))
    sx = int(rng.integers(width - dx))
    if witness == 2:
        sx = 0
    stamp_cols = (sx, sx + dx)
    dy = int(rng.integers(-2, 3))
    cams = []
    for i in range(n):
        own_dy, own_dx = dy, dx
        if i >= 2 and (witness == 2 or rng.random() < 0.7):
            signatures = [(y, x) for y in ([dy] if witness == 2 else range(-2, 3))
                          for x in range(1, min(3, width-1)+1) if (y, x) != (dy, dx)]
            own_dy, own_dx = signatures[int(rng.integers(len(signatures)))]
        own_sx = sx if own_dx == dx else int(rng.integers(width - own_dx))
        if witness == 2:
            own_sx = 0
        if witness in (1, 2):
            low = rng.integers(1, high, teeth_count).tolist() if i == 0 else None
            if i == 0:
                low[:2] = [1, 2]
                rng.shuffle(low)
                first_profile = low
                h = first_profile
            elif i == 1:
                h = [v + 1 for v in first_profile]
            else:
                h = list(first_profile)
        else:
            h = rng.integers(1, high + 1, teeth_count).tolist()
            if high > 3 and i == 0:
                h[int(rng.integers(teeth_count))] = int(rng.integers(4, high + 1))
            if i == 1 and h == list(cams[0]['heights']):
                h[0] = h[0] % high + 1
        color = body_a if rng.random() < 0.7 else body_b
        if witness:
            color = body_a  # repeated payload color cannot identify assignments
        if witness == 2 and i == 2:
            color = body_b
        cams.append(_cam(width, cols, h, (own_sx, own_sx + own_dx), own_dy, color))
    rng.shuffle(cams)
    t, l = int(rng.integers(2)), int(rng.integers(2))
    pedestals = rng.integers(0, 3, n).tolist()
    max_reach = max(max(p['heights']) + abs(p['dy']) + z for p, z in zip(cams, pedestals))
    depth = max_reach + int(rng.integers(5, 7))
    b = t + depth
    r = l + n * width + 3 * (n - 1) + 3
    fixed = _frame(t, l, b, r)
    ports, placements, pin_specs = [], [], {}
    for i, (cam, pedestal) in enumerate(zip(cams, pedestals)):
        start = l + 2 + i * (width + 3)
        base = b - 2 - abs(cam['dy']) - pedestal
        delta = base, start
        port = _shift(cam['stamps'], delta)
        ports.append(port)
        for row, col in port:
            fixed.update((a, col) for a in range(row + 1, b))
        placements.append(delta)
        for j, (col, height) in enumerate(zip(cam['cols'], cam['heights'])):
            capacity = base - height - t - 1
            assert capacity >= 2
            shift = int(rng.integers(0, height + 1))
            if witness == 1:
                # Only the taller cam can occupy its short-pin berth. The
                # lower cam's berth also accepts the taller one until the
                # global one-use-per-piece constraint resolves the assignment.
                taller = max(cam['heights']) > max(first_profile)
                shift = 0 if taller and j == 0 else int(rng.integers(1, height + 1))
            elif witness == 2:
                shift = 0 if j == 0 else 1
            length = capacity + shift
            seq = rng.choice([pin_a, pin_b], length).astype(int).tolist()
            # Surviving two-color order is observable, even when cells are lost.
            seq[shift], seq[shift + 1] = pin_a, pin_b
            pin_specs[start + col] = (tuple(seq), capacity, shift)
    # Pack the scattered cams below the housing, changing their order and
    # independent vertical offsets. This is a source inventory, not slot order.
    order = rng.permutation(n).tolist()
    cam_boxes = [_box(p['body'] | p['stamps']) for p in cams]
    max_h = max(bb - tt + 1 for tt, ll, bb, rr in cam_boxes)
    H = b + 2 + max_h
    if H > 30:
        return None
    H += int(rng.integers(0, min(2, 30 - H) + 1))
    W = r + 1 + int(rng.integers(0, min(2, 29-r) + 1))
    inp = np.full((H, W), bg, dtype=np.int8)
    out = inp.copy()
    for q in fixed:
        inp[q] = out[q] = frame_color
    for port in ports:
        for q in port:
            inp[q] = out[q] = marker
    for col, (seq, capacity, shift) in pin_specs.items():
        inp[t + 1:t + 1 + len(seq), col] = seq
        out[t + 1:t + 1 + capacity, col] = seq[shift:]
    # Output construction copies the intended rigid placements directly.
    for cam, delta in zip(cams, placements):
        for q in _shift(cam['body'], delta):
            if out[q] != bg:
                return None
            out[q] = cam['color']
    used_width = n * width + n - 1
    x = int(rng.integers(W - used_width + 1))
    for i in order:
        cam = cams[i]
        tt, ll, bb, rr = cam_boxes[i]
        top = b + 2 + int(rng.integers(H - (b + 2) - (bb - tt + 1) + 1))
        delta = top - tt, x - ll
        for q in _shift(cam['body'], delta):
            inp[q] = cam['color']
        for q in _shift(cam['stamps'], delta):
            inp[q] = marker
        x += width + 1
    try:
        f = _read_canonical(inp)
        answers = _assemblies(f)
        grids = {_key(_render_solution(inp, f, a)) for a in answers}
        if grids != {_key(out)}:
            return None
        if witness == 1 and len(_assemblies(f, False)) < 2:
            return None
        if witness == 2 and len({_stamp_shape(p) for p in f['ports']}) < 2:
            return None
        if witness == 2:
            alternatives = {_key(_render_solution(inp, f, a))
                            for a in _assemblies(f, match_geometry=False)}
            if len(alternatives) < 2:
                return None
    except AssertionError:
        return None
    k = int(rng.integers(4))
    if rng.random() < 0.5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return dict(input=np.rot90(inp, k).copy(), output=np.rot90(out, k).copy())


def _stamp_shape(cells):
    p, q = sorted(cells, key=lambda x: x[1])
    return q[0] - p[0], q[1] - p[1]


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.choice(10, 7, replace=False))
    regime = int(rng.integers(2))  # count extension or deeper profiles
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = index + 1 if split == 'train' and index < 2 else 0
            n = witness + 1 if witness else (4 if split == 'test' and regime == 0 else int(rng.integers(2, 4)))
            high = 5 if split == 'test' and regime == 1 else 3
            for attempt in range(200):
                pair = _make_pair(rng, n, roles, witness, high)
                if pair is not None and _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError("bounded sampling failed to obtain a unique press")
        rng.shuffle(puzzle[split])
    return puzzle


def _features(f, placement):
    shifts, capacities = [], []
    for p in placement:
        for col, cap in p['capacities'].items():
            capacities.append(cap)
            shifts.append(len(f['pins'][col]) - cap)
    return dict(cams=len(f['cams']), lanes=len(f['pins']),
                signatures=tuple(_stamp_shape(p) for p in f['ports']),
                profiles=tuple(p['heights'] for p in f['cams']),
                widths=tuple(p['width'] for p in f['cams']),
                shifts=tuple(shifts), capacities=tuple(capacities),
                docking_only=len(_assemblies(f, False)),
                shape_blind=len(_assemblies(f, match_geometry=False)),
                colors=(f['bg'], f['frame_color'], f['marker']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            k, a, f, placement = _analyze(inp)
            expected = np.rot90(_render_solution(a, f, placement), -k)
            assert np.array_equal(out, expected) and not np.array_equal(inp, out)
            canonical_out = np.rot90(out, k)
            # Each root stamp and cam stamp coincide once. Fixed frame/stems
            # persist; every body is translated intact; only pin prefixes vanish.
            assert np.count_nonzero(inp == f['marker']) == 2 * np.count_nonzero(out == f['marker'])
            assert all(canonical_out[q] == f['frame_color'] for q in f['fixed_body'])
            assert np.count_nonzero(out == f['frame_color']) == len(f['fixed_body'])
            for chosen in placement:
                cam = f['cams'][chosen['index']]
                assert all(canonical_out[q] == cam['color'] for q in chosen['body'])
                for col, cap in chosen['capacities'].items():
                    seq = f['pins'][col]
                    assert tuple(canonical_out[f['box'][0]+1:f['box'][0]+1+cap, col]) == seq[-cap:]
            record = _features(f, placement)
            if split == 'train':
                record['shape_blind_outputs'] = len({_key(_render_solution(a, f, p))
                    for p in _assemblies(f, match_geometry=False)})
                training.append(record)
    assert any(d['docking_only'] > 1 for d in training), "pin feasibility must resolve a docking ambiguity"
    assert any(len(set(d['signatures'])) > 1 for d in training), "equal-area stamp geometry must matter"
    assert any(d['shape_blind_outputs'] > 1 for d in training), "pin constraints alone must not determine the docking"
    assert any(0 in d['shifts'] and max(d['shifts']) > 0 for d in training)
    assert any(len(set(d['shifts'])) > 1 for d in training)
    assert any(len(set(h)) > 1 for d in training for h in d['profiles'])


def self_test():
    """Bounded semantic regressions and measurements on the 42 host seeds."""
    records, puzzle_keys, shapes, structures, orientations = [], set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                assert np.array_equal(solve(inp.copy()), out)
                k, a, f, placement = _analyze(inp)
                orientations.add(k)
                shapes.add(inp.shape)
                structures.add((inp.shape, (inp != f['bg']).tobytes()))
                records.append(_features(f, placement))
    assert len(puzzle_keys) == 42 and len(shapes) >= 25 and len(structures) >= 190
    assert {d['cams'] for d in records} == {2, 3, 4}
    assert len({p for d in records for p in d['profiles']}) >= 40
    assert len({s for d in records for s in d['signatures']}) >= 8
    assert len({d['lanes'] for d in records}) >= 4 and len(orientations) == 4
    assert max(max(d['shifts']) for d in records) >= 4
    assert any(len(set(p)) == 1 for d in records for p in d['profiles'])
    assert any(len(set(p)) > 1 for d in records for p in d['profiles'])
    # Repeated-stamp press: low and high nonuniform profiles share a stamp.
    # A short pin in the high berth forbids the low cam there, even though
    # the stamps admit swapping. This assertion is measured from input pixels.
    witness = generate_puzzle(0, 2, 1)
    readings = [_analyze(p['input']) for p in witness['train']]
    assert any(len(_assemblies(f, False)) >= 2 and len(_assemblies(f)) == 1
               for k, a, f, placement in readings)
    assert any(len({_key(_render_solution(a, f, p))
                    for p in _assemblies(f, match_geometry=False)}) >= 2
               for k, a, f, placement in readings)
    # Lengthen all pins in the repeated-stamp witness. Both cam permutations
    # can now push upward; the input becomes ambiguous and must be rejected.
    k, a, f, placement = next(x for x in readings if len(x[2]['cams']) == 2)
    ambiguous = a.copy()
    for chosen in placement:
        cam = f['cams'][chosen['index']]
        base = cam['base'] + chosen['delta'][0]
        for col in chosen['teeth']:
            ambiguous[f['box'][0]+1:base, col] = f['pins'][col][0]
    assert len(_assemblies(_read_canonical(ambiguous))) >= 2
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous press accepted")
    # A hand-constructed boundary scene retains exactly one cell in one lane.
    # Expected positions are drawn literally, without using either renderer.
    tiny = np.zeros((18, 13), dtype=np.int8)
    expected_tiny = tiny.copy()
    for q in _frame(0, 0, 8, 12):
        tiny[q] = expected_tiny[q] = 5
    for col in (2, 4, 8, 10):
        tiny[7, col] = expected_tiny[7, col] = 2
    seqs = {2: (4,), 4: (4, 6, 4, 6), 8: (6, 4, 6, 4), 10: (6, 4, 4, 6, 4)}
    capacities = {2: 1, 4: 2, 8: 3, 10: 4}
    for col, seq in seqs.items():
        tiny[1:1+len(seq), col] = seq
        expected_tiny[1:1+capacities[col], col] = seq[-capacities[col]:]
    for heights, start, source_base in (((4, 3), 2, 14), ((2, 1), 8, 13)):
        for grid, base in ((tiny, source_base), (expected_tiny, 6)):
            grid[base, start:start+3] = 3
            for col, h in zip((start, start+2), heights):
                grid[base-h:base, col] = 3
                grid[base+1, col] = 2
    assert np.array_equal(solve(tiny), expected_tiny)
    first = witness['train'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(first['input'], k)), np.rot90(first['output'], k))
    assert np.array_equal(solve(np.fliplr(first['input'])), np.fliplr(first['output']))
    mapping = np.array([(3*i + 7) % 10 for i in range(10)], dtype=np.int8)
    assert np.array_equal(solve(mapping[first['input']]), mapping[first['output']])
    # A changed pin cell cannot be accepted merely because docking is right.
    bad = {s: [{'input': p['input'].copy(), 'output': p['output'].copy()} for p in witness[s]]
           for s in witness}
    k, a, f, placement = _analyze(bad['test'][0]['input'])
    corrupted = np.rot90(bad['test'][0]['output'], k).copy()
    col = next(iter(f['pins']))
    corrupted[f['box'][0]+1, col] = f['bg']
    bad['test'][0]['output'] = np.rot90(corrupted, -k).copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("pin erasure accepted")
    repeat = generate_puzzle(HOST_SEEDS[3])
    original = generate_puzzle(HOST_SEEDS[3])
    assert all(_key(x[q]) == _key(y[q]) for s in repeat for x, y in zip(repeat[s], original[s]) for q in x)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(81, a, b))
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.0), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=len(records), shapes=len(shapes), structures=len(structures),
                profiles=len({p for d in records for p in d['profiles']}),
                signatures=len({s for d in records for s in d['signatures']}),
                pin_resolved=sum(d['docking_only'] > 1 for d in records),
                geometry_resolved=sum(d['shape_blind'] > 1 for d in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
