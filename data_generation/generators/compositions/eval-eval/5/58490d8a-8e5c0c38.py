"""Mirror-orbit census: count reflection orbits of supported components.

No parent module, file, seed, training pair, or generator state enters solve().
The panel's key alignment visibly supplies the reflection orientation.
"""
from collections import deque
import hashlib
import numpy as np


_STEPS8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))


def _components(cells, diagonal=True):
    remaining, result = set(cells), []
    steps = _STEPS8 if diagonal else _STEPS4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _checked_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return grid


def _grid_key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _best_axis(cells, width):
    """Double the axis coordinate, allowing integral and half-integral axes."""
    if not cells:
        raise ValueError("A keyed scene is empty")
    scores = [sum((r, a - c) in cells for r, c in cells)
              for a in range(2 * width - 1)]
    best = max(scores)
    winners = [a for a, score in enumerate(scores) if score == best]
    if len(winners) != 1:
        raise ValueError("Reflection support has tied maximum axes")
    axis = winners[0]
    return axis, {p for p in cells if (p[0], axis - p[1]) in cells}


def _fold(cells, axis):
    # Both parities give successive distance layers 0,1,2,... .
    return {(r, abs(2 * c - axis) // 2) for r, c in cells}


def _find_panel(grid):
    candidates = []
    for field in map(int, np.unique(grid)):
        rr, cc = np.where(grid == field)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        panel = grid[r0:r1, c0:c1]
        if min(panel.shape) < 5:
            continue
        if not all(np.all(edge == field) for edge in
                   (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        positions = np.argwhere(panel != field)
        if not 2 <= len(positions) <= 7:
            continue
        colors = [int(panel[r, c]) for r, c in positions]
        if len(set(colors)) != len(colors):
            continue
        if len(set(positions[:, 1])) == 1 and np.all(np.diff(positions[:, 0]) >= 2):
            inset = int(positions[0, 1])
            if 2 * inset == panel.shape[1] - 1:
                continue
            direction = 0 if 2 * inset < panel.shape[1] - 1 else 2
        elif len(set(positions[:, 0])) == 1 and np.all(np.diff(positions[:, 1]) >= 2):
            inset = int(positions[0, 0])
            if 2 * inset == panel.shape[0] - 1:
                continue
            direction = 3 if 2 * inset < panel.shape[0] - 1 else 1
        else:
            continue
        outside = np.ones(grid.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        values = grid[outside]
        bg = int(np.argmax(np.bincount(values, minlength=10)))
        if bg == field or bg in colors or np.mean(values == bg) <= .55:
            continue
        ring = np.zeros(grid.shape, bool)
        ring[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
        ring[r0:r1, c0:c1] = False
        if np.all(grid[ring] == bg):
            candidates.append((field, bg, direction))
    if len(candidates) != 1:
        raise ValueError("Expected one bordered panel with aligned singleton keys")
    return candidates[0]


def _analyze(input_grid):
    grid = _checked_grid(input_grid)
    field, bg, direction = _find_panel(grid)
    work = np.rot90(grid, -direction)
    r0, r1, c0, c1 = _bounds(set(map(tuple, np.argwhere(work == field))))
    panel = work[r0:r1, c0:c1]
    outside = np.ones(work.shape, bool)
    outside[r0:r1, c0:c1] = False
    positions = np.argwhere(panel != field)
    output = np.full(panel.shape, field, dtype=np.int8)
    records = []
    for row, col in positions:
        color = int(panel[row, col])
        cells = set(map(tuple, np.argwhere((work == color) & outside)))
        axis, core = _best_axis(cells, work.shape[1])
        folded = _fold(core, axis)
        parts = _components(folded)
        n = len(parts)
        if col + 2 * (n - 1) >= panel.shape[1] - 1:
            raise ValueError("Census exceeds the visible panel capacity")
        output[row, col:col + 2 * n:2] = color
        records.append(dict(color=color, axis=int(axis), count=n,
                            ordinary=len(_components(core)),
                            raw_folded=len(_components(_fold(cells, axis))),
                            four_folded=len(_components(folded, False)),
                            row_bands=len(_components({(r, 0) for r, c in folded})),
                            areas=tuple(sorted(map(len, parts))),
                            core=core, cells=cells, folded=folded,
                            noise=len(cells - core)))
    keys = {m['color'] for m in records}
    ignored = set(map(int, np.unique(work[outside]))) - {bg} - keys
    return dict(expected=np.rot90(output, direction).copy(), records=records,
                field=field, background=bg, direction=direction,
                bounds=(r0, r1, c0, c1), panel_shape=panel.shape,
                keys=keys, ignored=ignored)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Extract the chart and tally supported component orbits, using input only."""
    return _analyze(input_grid)['expected']


def _glyph(rng):
    mode = int(rng.integers(6))
    if mode == 0:
        return _DIAMOND
    if mode == 1:
        return _SQUARE
    if mode == 2:
        return ((0, 0), (1, 1), (2, 2))
    if mode == 3:
        return tuple((0, c) for c in range(int(rng.integers(2, 5))))
    if mode == 4:
        return ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
    cells = {(1, 1)}
    target = int(rng.integers(3, 8))
    while len(cells) < target:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _STEPS8
                           if 0 <= r + dr < 3 and 0 <= c + dc < 3} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _bridge(parts, height, max_d):
    """A shortest off-seam path joining two folded components."""
    starts = sorted(p for p in parts[0] if p[1] > 0)
    goals = set.union(*parts[1:])
    goals = {p for p in goals if p[1] > 0}
    occupied = set.union(*parts)
    queue, previous = deque(starts), {p: None for p in starts}
    endpoint = None
    while queue:
        p = queue.popleft()
        if p in goals:
            endpoint = p
            break
        for dr, dc in _STEPS8:
            q = p[0] + dr, p[1] + dc
            if (0 <= q[0] < height and 1 <= q[1] <= max_d + 2
                    and q not in previous and (q not in occupied or q in goals)):
                previous[q] = p
                queue.append(q)
    path = set()
    while endpoint is not None:
        path.add(endpoint)
        endpoint = previous[endpoint]
    return path - occupied


def _make_scene(rng, count, parity, witness=None, compact=False):
    # Two-dimensional packing and separated bands are both sampled. Components
    # may share rows, so counting row bands cannot replace connectivity.
    parts, row = [], 0
    if witness:
        parts = [set(_DIAMOND), {(r, c + 4) for r, c in _SQUARE}]
        if count == 3:
            parts.append({(5, 0)})
    else:
        shapes = [tuple((0, c) for c in range(2)) if compact else _glyph(rng)
                  for _ in range(count)]
        if not compact and rng.random() < .60:
            canvas = np.zeros((int(rng.integers(6, 15)), int(rng.integers(5, 9))), np.int8)
            blocked = np.zeros(canvas.shape, bool)
            for i, shape in enumerate(shapes, 1):
                if not _place_scene(canvas, blocked, set(shape), i, rng):
                    break
            else:
                parts = [set(map(tuple, np.argwhere(canvas == i))) for i in range(1, count + 1)]
        if not parts:
            for shape in shapes:
                distance = int(rng.integers(0, 4 if not compact else 3))
                part = {(row + r, distance + c) for r, c in shape}
                parts.append(part)
                row = max(r for r, c in part) + 2 + (int(rng.integers(2)) if not compact else 0)
    folded = set.union(*parts)
    max_d = max(c for r, c in folded)
    height = max(r for r, c in folded) + 1
    # Odd-width core: axis=max_d. Even-width core: axis=max_d+0.5.
    axis = 2 * max_d + parity
    core = {(r, max_d - d) for r, d in folded}
    core |= {(r, max_d + parity + d) for r, d in folded}
    if witness == 'bridge':
        noise_folded = _bridge(parts, height, max_d)
    elif witness == 'island':
        noise_folded = {(height + 1, max_d + 1)}
    elif compact or rng.random() < .20:
        noise_folded = set()
    else:
        mode = int(rng.integers(3))
        if mode == 0 and count > 1:
            noise_folded = _bridge(parts, height, max_d)
        elif mode == 1:
            noise_folded = {(height + 1, int(rng.integers(1, max_d + 3)))}
            if rng.random() < .4:
                noise_folded.add((height + 1, max_d + 2))
        else:
            choices = sorted({(r + dr, d + dc) for r, d in folded for dr, dc in _STEPS8
                              if r + dr >= 0 and d + dc >= 1} - folded)
            rng.shuffle(choices)
            noise_folded = set(choices[:int(rng.integers(1, 4))])
    # All noise is one-sided and off the seam, hence has no intended partner.
    noise = {(r, max_d + parity + d) for r, d in noise_folded}
    cells = core | noise
    width = max(c for r, c in cells) + 1
    try:
        found, supported = _best_axis(cells, width)
        good = found == axis and supported == core
    except ValueError:
        good = False
    if not good:
        if witness:
            raise AssertionError("Informative witness lost its unique axis")
        cells = core
    if witness == 'bridge':
        assert len(_components(_fold(cells, axis))) < count
    elif witness == 'island':
        assert len(_components(_fold(cells, axis))) > count
    assert len(_components(_fold(core, axis))) == count
    return cells


def _place_scene(grid, blocked, cells, color, rng):
    _, height, _, width = _bounds(cells)
    nh, nw = grid.shape[0] - height + 1, grid.shape[1] - width + 1
    if min(nh, nw) < 1:
        return False
    allowed = np.ones((nh, nw), bool)
    for r, c in cells:
        allowed &= ~blocked[r:r + nh, c:c + nw]
    candidates = np.argwhere(allowed)
    if len(candidates) == 0:
        return False
    if rng.random() < .20:
        edge = candidates[(candidates[:, 0] == 0) | (candidates[:, 0] == nh - 1)
                          | (candidates[:, 1] == 0) | (candidates[:, 1] == nw - 1)]
        if len(edge):
            candidates = edge
    top, left = candidates[int(rng.integers(len(candidates)))]
    for r, c in cells:
        rr, cc = r + top, c + left
        grid[rr, cc] = color
        blocked[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2] = True
    return True


def _make_pair(rng, spec):
    counts, keys = spec['counts'], spec['keys']
    top, bottom = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    rows = [top]
    for _ in keys[1:]:
        rows.append(rows[-1] + int(rng.integers(2, 4)))
    ph = min(10, rows[-1] + bottom + 1)
    inset, trailing = int(rng.integers(1, 3)), int(rng.integers(1, 4))
    pw = max(2 * inset + 3, inset + 2 * (max(counts) - 1) + trailing + 1)
    scene_specs = [(color, n, spec['parities'][i], spec['witness'] if i == 0 else None)
                   for i, (color, n) in enumerate(zip(keys, counts))]
    scene_specs += [(color, int(rng.integers(1, 4)), int(rng.integers(2)), None)
                    for color in spec['ignored']]
    for attempt in range(13):
        compact = attempt == 12
        scenes = [(color, _make_scene(rng, n, parity, witness, compact))
                  for color, n, parity, witness in scene_specs]
        if compact or attempt >= 8:
            h = w = 30
        else:
            min_h = max(14, ph + 2, max(max(r for r, c in s) + 1 for _, s in scenes))
            min_w = max(14, pw + 2, max(max(c for r, c in s) + 1 for _, s in scenes))
            h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        if compact:
            r0, c0 = 30 - ph, int(rng.integers(31 - pw))
        else:
            r0, c0 = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
            if rng.random() < .35:
                if rng.random() < .5:
                    r0 = int(rng.choice([0, h - ph]))
                else:
                    c0 = int(rng.choice([0, w - pw]))
        grid = np.full((h, w), spec['background'], np.int8)
        grid[r0:r0 + ph, c0:c0 + pw] = spec['field']
        for row, color in zip(rows, keys):
            grid[r0 + row, c0 + inset] = color
        blocked = np.zeros(grid.shape, bool)
        blocked[max(0, r0 - 1):r0 + ph + 1, max(0, c0 - 1):c0 + pw + 1] = True
        if compact:
            # Guaranteed finite fallback: four <=9-by-12 scenes above the panel.
            # Required diagonal, bridge, island and orbit witnesses survive.
            for (color, cells), (r, c) in zip(scenes, ((0, 0), (0, 15), (10, 0), (10, 15))):
                assert max(rr for rr, cc in cells) < 9 and max(cc for rr, cc in cells) < 13
                for rr, cc in cells:
                    assert not blocked[r + rr, c + cc]
                    grid[r + rr, c + cc] = color
        else:
            scenes.sort(key=lambda item: -(max(r for r, c in item[1]) + 1)
                        * (max(c for r, c in item[1]) + 1))
            if not all(_place_scene(grid, blocked, cells, color, rng) for color, cells in scenes):
                continue
        output = np.full((ph, pw), spec['field'], np.int8)
        for row, color, n in zip(rows, keys, counts):
            output[row, inset:inset + 2 * n:2] = color
        direction = spec['direction']
        grid, output = np.rot90(grid, direction).copy(), np.rot90(output, direction).copy()
        # Construction labels come from the component list, not from solve().
        try:
            predicted = solve(grid)
        except ValueError:
            if compact:
                raise
            continue
        assert np.array_equal(predicted, output), "Placement changed the intended census"
        return {'input': grid, 'output': output}
    raise AssertionError("Unreachable: bounded compact fallback failed")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(name + " is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    fixed_direction = int(rng.integers(4))
    mixed_directions = bool(rng.random() < .35)
    puzzle, seen, role_color = {'train': [], 'test': []}, set(), None
    for split, length in (('train', nt), ('test', nq)):
        for i in range(length):
            witness = ('bridge' if i == 0 else 'island' if i == 1 else None) if split == 'train' else None
            palette = list(map(int, rng.permutation(10)))
            if split == 'train' and i == 1:
                palette.remove(role_color)
            k = int(rng.integers(2, 4))
            bg, field, keys = palette[0], palette[1], palette[2:2 + k]
            if split == 'train' and i == 0:
                role_color = keys[0]
            ignored = ([role_color] if split == 'train' and i == 1 else
                       palette[2 + k:3 + k] if rng.random() < .35 else [])
            counts = list(map(int, rng.integers(1, 4, size=k)))
            if witness:
                counts[0] = 3 if witness == 'bridge' else 2
            if split == 'test':
                counts[0] = int(rng.integers(4, 6))
            parities = list(map(int, rng.integers(2, size=k)))
            if witness:
                parities[0] = 0 if witness == 'bridge' else 1
            direction = ((fixed_direction + i) % 4 if mixed_directions and split == 'train'
                         else int(rng.integers(4)) if mixed_directions else fixed_direction)
            spec = dict(background=bg, field=field, keys=keys, ignored=ignored,
                        counts=counts, parities=parities, direction=direction, witness=witness)
            for _ in range(8):
                pair = _make_pair(rng, spec)
                key = _grid_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not obtain a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible geometry, orbit topology, witnesses and count extrapolation."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _checked_grid(pair['input']), _checked_grid(pair['output'])
            key = _grid_key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            m = _analyze(inp)
            assert np.array_equal(out, m['expected']), "Incorrect orbit tally"
            assert 2 <= len(m['keys']) <= 3
            for rec in m['records']:
                axis, core = rec['axis'], rec['core']
                assert core <= rec['cells']
                assert {(r, axis - c) for r, c in core} == core
                # Check orbit count directly on original connected components.
                # This cross-check does not use folding or folded connectivity.
                comps = _components(core)
                orbit_keys = set()
                for comp in comps:
                    reflected = {(r, axis - c) for r, c in comp}
                    assert reflected in comps
                    orbit_keys.add(min(tuple(sorted(comp)), tuple(sorted(reflected))))
                assert len(orbit_keys) == rec['count']
                assert 1 <= rec['count'] <= 5
            (training if split == 'train' else testing).append(m)
    trs = [r for m in training for r in m['records']]
    assert any(r['raw_folded'] < r['count'] for r in trs), "No asymmetric bridge witness"
    assert any(r['raw_folded'] > r['count'] for r in trs), "No false-island witness"
    assert any(r['ordinary'] > r['count'] for r in trs), "No mirror-pair quotient witness"
    assert any(r['four_folded'] > r['count'] for r in trs), "No diagonal connectivity witness"
    assert any(r['row_bands'] < r['count'] for r in trs), "Row bands could replace components"
    assert any(len(set(r['areas'])) > 1 for r in trs), "No unequal-area groups"
    assert {r['axis'] % 2 for r in trs if r['noise']} == {0, 1}
    assert any(a['keys'] & b['ignored'] for a in training for b in training)
    train_max = max(r['count'] for r in trs)
    assert train_max == 3
    assert all(max(r['count'] for r in m['records']) > train_max for m in testing)


def self_test():
    """Stream 42 seeds; save no grids or logs. Return small measured coverage."""
    shapes, panels, normalized, fingerprints = set(), set(), set(), set()
    backgrounds, fields, directions, key_counts, counts, noise_amounts = [set() for _ in range(6)]
    bridge = island = quotient = diagonal = shared_rows = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                digest.update(bytes(inp.shape) + inp.tobytes() + bytes(out.shape) + out.tobytes())
                m = _analyze(inp)
                shapes.add(inp.shape)
                panels.add(out.shape)
                backgrounds.add(m['background'])
                fields.add(m['field'])
                directions.add(m['direction'])
                key_counts.add(len(m['keys']))
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.ravel())
                normalized.add(hashlib.sha256(bytes(inp.shape) + norm).digest())
                for rec in m['records']:
                    counts.add(rec['count'])
                    noise_amounts.add(rec['noise'])
                    bridge += rec['raw_folded'] < rec['count']
                    island += rec['raw_folded'] > rec['count']
                    quotient += rec['ordinary'] > rec['count']
                    diagonal += rec['four_folded'] > rec['count']
                    shared_rows += rec['row_bands'] < rec['count']
                # Rotation equivariance exercises the input-visible orientation cue.
                if seed < 2:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                pairs += 1
        fingerprints.add(digest.digest())
        if seed < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                       for p, q in zip(puzzle[s], again[s]) for k in ('input', 'output'))
    assert len(fingerprints) == 42 and len(normalized) >= 180
    assert len(shapes) >= 30 and len(panels) >= 20
    assert backgrounds == fields == set(range(10)) and directions == set(range(4))
    assert key_counts == {2, 3} and counts == set(range(1, 6))
    assert 0 in noise_amounts and max(noise_amounts) >= 3
    assert min(bridge, island, quotient, diagonal, shared_rows) >= 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(42 + 10 * nt + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    try:
        _best_axis({(0, 0), (1, 2)}, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("A tied axis was accepted")
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes), output_shapes=len(panels),
                normalized_structures=len(normalized), bridge_witnesses=bridge,
                island_witnesses=island, quotient_witnesses=quotient,
                diagonal_witnesses=diagonal, shared_row_witnesses=shared_rows,
                directions=sorted(directions),
                counts=sorted(counts), noise_amounts=sorted(noise_amounts))
