"""Two-color orthogonal drawings with displaced three-cell continuations.

A repair has two off-axis cells, a retained anchor, a bracket at the other end,
and visible straight continuation. Unattached diagonals disappear; unsupported
breaks and straight strokes survive. Geometry, rather than figure identity or a
train/test flag, determines every edit.
"""

import numpy as np


def _count_value(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _resolve_counts(seed, num_train, num_test):
    n = _count_value('num_train', num_train, 2, 10)
    t = _count_value('num_test', num_test, 1, 4)
    if n is None and t is None:
        t = 1 + seed % 4
        n = t + 1 + (seed // 4) % (10 - t)
    elif n is None:
        n = t + 1 + seed % (10 - t)
    elif t is None:
        t = 1 + seed % min(4, n - 1)
    if n <= t:
        raise ValueError('num_train must be greater than num_test')
    return n, t


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong_straight(mask):
    """Cells belonging to an intact run of at least three cells."""
    strong = np.zeros_like(mask)
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        s = strong if axis == 0 else strong.T
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            s[:, offset:offset + triples.shape[1]] |= triples
    return strong


def _gaps(mask):
    """All visibly bracketed three-cell gaps, including intentional breaks."""
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _evidence(mask):
    """Find repairs from the input alone, before considering any proposed label."""
    strong = _strong_straight(mask)
    repairs = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5*dr, c + 5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = [(ar + k*sr, ac + k*sc) for k in (1, 2)]
                if all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in spur):
                    # A detached three-cell stroke is not a two-cell deviation.
                    if not _at(mask, ar + 3*sr, ac + 3*sc):
                        matches.append((endpoint, side, tuple(spur)))
        if matches:
            repairs.append(((r, c, dr, dc), matches))
    return repairs


def _reconstruct_mask(mask):
    repairs = _evidence(mask)
    added = np.zeros_like(mask)
    anchors = np.zeros_like(mask)
    for (r, c, dr, dc), _ in repairs:
        for k in (1, 2, 3):
            added[r + k*dr, c + k*dc] = True
        anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
    # Short diagonal chains can begin one cell beside an intact line. Merely
    # keeping every cell with an orthogonal neighbor would leave such clutter.
    weak = mask & ~_strong_straight(mask | added) & ~anchors
    diagonal = np.zeros_like(mask)
    for dr, dc in ((1, 1), (1, -1)):
        a = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        linked = a & b
        if dc == 1:
            diagonal[:-1, :-1] |= linked
            diagonal[1:, 1:] |= linked
        else:
            diagonal[:-1, 1:] |= linked
            diagonal[1:, :-1] |= linked
    return (mask & ~diagonal) | added


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    bg = int(colors[np.argmax(counts)])
    line = int(colors[np.argmin(counts)])
    return bg, line


def solve(train_pairs, test_input):
    """Prediction interface uses only training demonstrations and a test input.

    This is a local diagnostic, not a host-provided independent reference solver.
    There are no hidden color, direction, or scaffold-selection conventions.
    """
    for pair in train_pairs:
        bg, line = _roles(pair['input'])
        expected = np.where(_reconstruct_mask(pair['input'] == line), line, bg)
        assert np.array_equal(expected, pair['output'])
    bg, line = _roles(test_input)
    return np.where(_reconstruct_mask(test_input == line), line, bg).astype(np.int8)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in steps:
                nxt = r + dr, c + dc
                if nxt in cells:
                    cells.remove(nxt)
                    stack.append(nxt)
        result.append(component)
    return result


def _thick(mask):
    return bool((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).any())


def _halo(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _skeleton(rng, h, w, richness):
    """Grow perpendicular branches on a variable trunk; occasionally close a loop."""
    out = np.zeros((h, w), bool)
    left = int(rng.integers(0, max(1, w - 5)))
    right = int(rng.integers(left + 5, w))
    row = int(rng.integers(h))
    out[row, left:right+1] = True
    # An endpoint branch leaves room for a source-like single-junction repair.
    columns = [int(rng.choice([left, right]))]
    candidates = [c for c in range(left, right+1) if abs(c-columns[0]) >= 3]
    wanted = int(rng.integers(1, min(5, 2 + richness)))
    while candidates and len(columns) < wanted:
        col = int(rng.choice(candidates))
        columns.append(col)
        candidates = [c for c in candidates if abs(c-col) >= 3]
    branches = []
    for col in columns:
        lo = int(rng.integers(row + 1))
        hi = int(rng.integers(row, h))
        if lo == hi:
            if row:
                lo -= 1
            else:
                hi += 1
        if rng.random() < .5:
            if row and rng.random() < .5:
                hi = row
            elif row < h-1:
                lo = row
        if lo == hi:
            if row:
                lo -= 1
            else:
                hi += 1
        out[lo:hi+1, col] = True
        branches.append((col, lo, hi))
    if len(columns) >= 2 and rng.random() < .18:
        levels = [r for r in range(h) if abs(r-row) >= 3
                  and sum(lo <= r <= hi for _, lo, hi in branches) >= 2]
        if levels:
            level = int(rng.choice(levels))
            endpoints = [c for c, lo, hi in branches if lo <= level <= hi]
            out[level, min(endpoints):max(endpoints)+1] = True
    # Secondary arms produce trees beyond a single comb and vary attachment sites.
    if richness and rng.random() < .45:
        for col, lo, hi in branches:
            level = int(rng.choice([lo, hi]))
            if abs(level-row) >= 3:
                a, b = sorted([col, int(rng.integers(w))])
                candidate = out.copy()
                candidate[level, a:b+1] = True
                if not _thick(candidate):
                    out = candidate
                break
    return out


def _damage_candidates(out, axis):
    candidates = []
    for dr, dc in ((0, 1), (1, 0)):
        if axis != 'mixed' and axis != ('h' if dc else 'v'):
            continue
        for r, c in np.argwhere(out):
            r, c = int(r), int(c)
            if not all(_at(out, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(out, r-dr, c-dc) or _at(out, r+5*dr, c+5*dc)):
                continue
            gap = [(r+k*dr, c+k*dc) for k in (1, 2, 3)]
            if any(_at(out, rr+dc, cc+dr) or _at(out, rr-dc, cc-dr) for rr, cc in gap):
                continue
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    spur = [(ar+k*(toward*dr+side*dc), ac+k*(toward*dc+side*dr)) for k in (1, 2)]
                    if all(0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
                           and not out[rr, cc] for rr, cc in spur):
                        candidates.append((gap, spur))
    return candidates


def _corrupt(rng, out, axis, target):
    inp = out.copy()
    candidates = _damage_candidates(out, axis)
    order = rng.permutation(len(candidates))
    count = 0
    changed = np.zeros_like(out)
    for index in order:
        gap, spur = candidates[int(index)]
        footprint = np.zeros_like(out)
        for r, c in gap + spur:
            footprint[r, c] = True
        if (_halo(changed) & footprint).any():
            continue
        trial = inp.copy()
        for r, c in gap:
            trial[r, c] = False
        for r, c in spur:
            trial[r, c] = True
        evidence = _evidence(trial)
        if (np.array_equal(_reconstruct_mask(trial), out)
                and len(evidence) == count + 1
                and all(len(matches) == 1 for _, matches in evidence)):
            inp = trial
            changed |= footprint
            count += 1
            if count == target:
                break
    return inp, count


def _figure(rng, h, w, axis, damaged, richness):
    vertical = axis == 'v' or (axis == 'mixed' and bool(rng.integers(2)))
    lh, lw = (w, h) if vertical else (h, w)
    target = int(rng.integers(1, richness + 3)) if damaged else 0
    for _ in range(24):
        out = _skeleton(rng, lh, lw, richness)
        if _thick(out) or not np.array_equal(_reconstruct_mask(out), out):
            continue
        inp, count = _corrupt(rng, out, 'mixed' if axis == 'mixed' else 'h', target) if damaged else (out.copy(), 0)
        if not damaged or count:
            break
    else:
        # A deterministic open L guarantees a legal repair on every region size.
        out = np.zeros((lh, lw), bool)
        out[2, :] = True
        out[2:, -1] = True
        inp = out.copy()
        if damaged:
            inp[2, 1:4] = False
            inp[3, 1] = inp[4, 2] = True
    if vertical:
        inp, out = inp.T, out.T
    if rng.integers(2):
        inp, out = inp[::-1], out[::-1]
    if rng.integers(2):
        inp, out = inp[:, ::-1], out[:, ::-1]
    return inp.copy(), out.copy()


def _partition(rng, region, target):
    regions = [region]
    while len(regions) < target:
        choices = []
        for i, (r, c, h, w) in enumerate(regions):
            if h >= 18:
                choices.append((i, 0))
            if w >= 18:
                choices.append((i, 1))
        if not choices:
            break
        index, axis = choices[int(rng.integers(len(choices)))]
        r, c, h, w = regions.pop(index)
        size = h if axis == 0 else w
        cut = int(rng.integers(8, size-9))
        if axis == 0:
            regions.extend([(r, c, cut, w), (r+cut+2, c, h-cut-2, w)])
        else:
            regions.extend([(r, c, h, cut), (r, c+cut+2, h, w-cut-2)])
    rng.shuffle(regions)
    return regions


def _contrast_band(rng, width):
    """An intentional gap and a nearby diagonal with a missing attachment."""
    out = np.zeros((4, width), bool)
    start = int(rng.integers(max(1, width-5)))
    end = int(rng.integers(start+5, width))
    gap = int(rng.integers(start+1, end-2))
    out[0, start:end+1] = True
    out[0, gap:gap+3] = False
    # Longer pieces can contain several deliberate breaks with variable spacing.
    available = [c for c in range(start+1, end-2) if abs(c-gap) >= 5]
    for _ in range(int(rng.integers(3))):
        if not available:
            break
        other = int(rng.choice(available))
        out[0, other:other+3] = False
        available = [c for c in available if abs(c-other) >= 5]
    inp = out.copy()
    # The extrapolated diagonal anchor lies in the empty row 1.
    inp[2, gap] = inp[3, gap+1] = True
    if rng.integers(2):
        inp, out = inp[::-1], out[::-1]
    if rng.integers(2):
        inp, out = inp[:, ::-1], out[:, ::-1]
    return inp.copy(), out.copy()


def _scatter(rng, inp, out, richness):
    # Intact pieces and detached strokes have no fixed corner or reserved column.
    for _ in range(int(rng.integers(richness + 4))):
        length = int(rng.integers(2, 7))
        dr, dc = (1, 0) if rng.integers(2) else (0, 1)
        forbidden = _halo(inp | out)
        options = []
        for r, c in np.argwhere(~forbidden):
            cells = [(int(r)+k*dr, int(c)+k*dc) for k in range(length)]
            if all(0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and not forbidden[rr, cc] for rr, cc in cells):
                options.append(cells)
        if options:
            trial_in, trial_out = inp.copy(), out.copy()
            for r, c in options[int(rng.integers(len(options)))]:
                trial_in[r, c] = trial_out[r, c] = True
            if np.array_equal(_reconstruct_mask(trial_in), trial_out):
                inp, out = trial_in, trial_out
    for _ in range(int(rng.integers(richness + 4))):
        length = int(rng.choice([2, 3]))
        for _ in range(35):
            r, c = int(rng.integers(out.shape[0])), int(rng.integers(out.shape[1]))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            cells = [(r+k*dr, c+k*dc) for k in range(length)]
            if not all(0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and not inp[rr, cc] and not out[rr, cc] for rr, cc in cells):
                continue
            trial = inp.copy()
            for rr, cc in cells:
                trial[rr, cc] = True
            if np.array_equal(_reconstruct_mask(trial), out):
                # Keep erased components straight and separate, including from
                # existing deviations; permit attachment beside intact linework.
                erased = trial & ~out
                if all(_diagonal_component(x) for x in _components(erased, True)):
                    inp = trial
                    break
    return inp, out


def _sample_parameters(rng):
    params = {
        'degree': int(rng.choice([0, 1, 2], p=[.20, .55, .25])),
        'base_shape': (int(rng.integers(12, 31)), int(rng.integers(12, 31))),
        'axis': str(rng.choice(['h', 'v'])),
        'orientation': str(rng.choice(['steady', 'varied', 'varied', 'varied'])),
        'palette_mode': str(rng.choice(['fixed', 'swap', 'varied', 'varied'])),
        'palette': tuple(map(int, rng.choice(10, 2, replace=False))),
        'shift': str(rng.choice(['matched', 'matched', 'matched', 'boundary', 'count', 'orientation', 'aspect'])),
    }
    if params['degree'] == 0:
        # Simple puzzles deliberately keep several presentation axes steady.
        # The other modes independently combine orientation, color, and shifts.
        params['orientation'] = 'steady'
        params['palette_mode'] = str(rng.choice(['fixed','fixed','swap']))
        params['shift'] = str(rng.choice(['matched','matched','matched','boundary','aspect']))
    return params


def _shape(rng, params, regime, contrast):
    degree = params['degree']
    if rng.random() < (.12 if degree == 0 else .45):
        h, w = int(rng.integers(10, 31)), int(rng.integers(10, 31))
    else:
        delta = (2, 5, 10)[degree]
        h, w = [int(np.clip(x + rng.integers(-delta, delta+1), 10, 30)) for x in params['base_shape']]
    if rng.random() < .22:
        w = h
    if regime == 'target_test' and params['shift'] == 'aspect':
        h, w = int(rng.integers(23, 31)), int(rng.integers(10, 20))
        if rng.integers(2):
            h, w = w, h
    if contrast and max(h, w) < 13:
        h = 13
    return h, w


def _compose_pair(rng, params, regime, index):
    assert regime in ('source_train', 'target_test')
    degree = params['degree']
    contrast = (regime == 'source_train' and index == 0) or rng.random() < .65
    h, w = _shape(rng, params, regime, contrast)
    inp, out = np.zeros((h, w), bool), np.zeros((h, w), bool)
    region = (0, 0, h, w)
    if contrast:
        horizontal = h >= 13 and (w < 13 or bool(rng.integers(2)))
        if horizontal:
            a, b = _contrast_band(rng, w)
            offset = int(rng.integers(0, min(8, h-12)))
            if rng.integers(2):
                inp[offset:offset+4], out[offset:offset+4] = a, b
                region = (offset+5, 0, h-offset-5, w)
            else:
                inp[h-offset-4:h-offset], out[h-offset-4:h-offset] = a, b
                region = (0, 0, h-offset-5, w)
        else:
            a, b = _contrast_band(rng, h)
            offset = int(rng.integers(0, min(8, w-12)))
            if rng.integers(2):
                inp[:, offset:offset+4], out[:, offset:offset+4] = a.T, b.T
                region = (0, offset+5, h, w-offset-5)
            else:
                inp[:, w-offset-4:w-offset], out[:, w-offset-4:w-offset] = a.T, b.T
                region = (0, 0, h, w-offset-5)
    target_figures = int(rng.integers(1, degree+3))
    if regime == 'target_test' and params['shift'] == 'count':
        target_figures = 4
    regions = _partition(rng, region, min(4, target_figures))
    damaged_count = int(rng.integers(1, len(regions)+1))
    if regime == 'target_test' and params['shift'] == 'count':
        damaged_count = len(regions)
    if params['shift'] == 'orientation':
        axis = params['axis'] if regime == 'source_train' else ('v' if params['axis'] == 'h' else 'h')
    elif params['orientation'] == 'steady':
        axis = params['axis']
    else:
        axis = str(rng.choice(['h', 'v', 'mixed']))
    for j, (r, c, rh, rw) in enumerate(regions):
        # Insets vary independently; target boundary cases retain the full region.
        boundary = regime == 'target_test' and params['shift'] == 'boundary'
        top = 0 if boundary else int(rng.integers(0, min(3, rh-7)))
        bottom = 0 if boundary else int(rng.integers(0, min(3, rh-top-7)))
        left = 0 if boundary else int(rng.integers(0, min(3, rw-7)))
        right = 0 if boundary else int(rng.integers(0, min(3, rw-left-7)))
        a, b = _figure(rng, rh-top-bottom, rw-left-right, axis, j < damaged_count, degree)
        dest = np.s_[r+top:r+rh-bottom, c+left:c+rw-right]
        inp[dest], out[dest] = a, b
    inp, out = _scatter(rng, inp, out, degree)
    assert np.array_equal(_reconstruct_mask(inp), out)
    palette = params['palette']
    if params['palette_mode'] == 'swap':
        if rng.integers(2):
            palette = palette[::-1]
    elif params['palette_mode'] == 'varied':
        choice = int(rng.integers(4))
        if choice == 1:
            palette = palette[::-1]
        elif choice >= 2:
            palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    bg, line = palette
    return {'input': np.where(inp, line, bg).astype(np.int8),
            'output': np.where(out, line, bg).astype(np.int8)}


def _make_pair(rng, params, regime, index):
    # Independently valid figures can occasionally align across their regions.
    # Check the composed scene as well; never assign conflicting local labels.
    for _ in range(12):
        try:
            pair = _compose_pair(rng, params, regime, index)
            _analyze_pair(pair['input'], pair['output'])
            return pair
        except AssertionError:
            continue
    # Bounded deterministic fallback, with an intentional gap and its near miss
    # far from the repaired L. It has no scene-level accidental continuation.
    out = np.zeros((14, 14), bool)
    out[0, :2] = out[0, 5:9] = True
    out[8, 1:13] = out[8:14, 12] = True
    inp = out.copy()
    inp[2, 2] = inp[3, 3] = True
    inp[8, 4:7] = False
    inp[9, 4] = inp[10, 5] = True
    turns = int(rng.integers(4))
    inp, out = np.rot90(inp, turns), np.rot90(out, turns)
    bg, line = params['palette']
    return {'input': np.where(inp, line, bg).astype(np.int8),
            'output': np.where(out, line, bg).astype(np.int8)}


def _input_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_input_key(p['input']), _input_key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    n, t = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(seed)
    params = _sample_parameters(rng)
    puzzle, seen = {'train': [], 'test': []}, set()
    source_keys = {_input_key(p['input']) for p in _source_pairs()}
    for split, regime, count in [('train', 'source_train', n), ('test', 'target_test', t)]:
        for index in range(count):
            pair = _make_pair(rng, params, regime, index)
            if _input_key(pair['input']) in seen | source_keys:
                # A finite deterministic palette enumeration has 90 possibilities
                # and fewer than 17 forbidden inputs, so it always resolves a tie.
                bg, line = _roles(pair['input'])
                im, om = pair['input'] == line, pair['output'] == line
                for a in range(10):
                    for b in range(10):
                        if a == b:
                            continue
                        candidate = {'input': np.where(im, b, a).astype(np.int8),
                                     'output': np.where(om, b, a).astype(np.int8)}
                        if _input_key(candidate['input']) not in seen | source_keys:
                            pair = candidate
                            break
                    else:
                        continue
                    break
            seen.add(_input_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _diagonal_component(cells):
    cells = sorted(cells)
    return (len(cells) in (2, 3)
            and all(cells[i+1][0]-cells[i][0] == 1 for i in range(len(cells)-1))
            and len({cells[i+1][1]-cells[i][1] for i in range(len(cells)-1)}) == 1
            and abs(cells[1][1]-cells[0][1]) == 1)


def _analyze_pair(inp, out):
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= d <= 30 for d in grid.shape)
        assert ((0 <= grid) & (grid <= 9)).all()
    assert inp.shape == out.shape
    bg, line = _roles(inp)
    assert _roles(out) == (bg, line)
    im, om = inp == line, out == line
    assert not _thick(om), 'parallel strokes merged into a solid 2x2 block'
    assert np.array_equal(_reconstruct_mask(im), om), 'incomplete or incorrect reconstruction'
    added, removed = om & ~im, im & ~om
    assert added.any() and removed.any()
    components = _components(added)
    for cells in components:
        assert len(cells) == 3
        assert len({r for r, _ in cells}) == 1 or len({c for _, c in cells}) == 1
    erased = _components(removed, True)
    assert all(_diagonal_component(cells) for cells in erased)
    evidence = _evidence(im)
    assert len(evidence) == len(components)
    for _, matches in evidence:
        assert len(matches) == 1, 'ambiguous overlapping deviations'
        assert set(matches[0][2]) in [set(cells) for cells in erased]
    multi, kinds = [], []
    cycles = 0
    for cells in _components(om):
        rs, cs = {r for r, _ in cells}, {c for _, c in cells}
        if len(rs) > 1 and len(cs) > 1:
            multi.append(cells)
        cellset = set(cells)
        edges = sum((r+1, c) in cellset for r, c in cells) + sum((r, c+1) in cellset for r, c in cells)
        loops = max(0, edges-len(cells)+1)
        cycles += loops
        if len(rs) > 1 and len(cs) > 1:
            degrees = [sum((r+dr, c+dc) in cellset for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))) for r,c in cells]
            branches = sum(d >= 3 for d in degrees)
            kinds.append('loop' if loops else 'open_path' if not branches
                         else 'branching' if branches > 1 else 'cross' if max(degrees) == 4 else 'T')
    junctions, run_lengths, boundary_runs = [], [], set()
    for (r,c,dr,dc), _ in evidence:
        lo, hi = 0, 4
        while _at(om, r+(lo-1)*dr, c+(lo-1)*dc):
            lo -= 1
        while _at(om, r+(hi+1)*dr, c+(hi+1)*dc):
            hi += 1
        cells = [(r+k*dr,c+k*dc) for k in range(lo,hi+1)]
        junctions.append(sum(_at(om, rr+dc, cc+dr) or _at(om, rr-dc, cc-dr) for rr,cc in cells))
        run_lengths.append(hi-lo+1)
        for rr,cc in cells:
            if rr == 0: boundary_runs.add('top')
            if rr == om.shape[0]-1: boundary_runs.add('bottom')
            if cc == 0: boundary_runs.add('left')
            if cc == om.shape[1]-1: boundary_runs.add('right')
    def near_edges(mask):
        return tuple(name for name, present in [('top',mask[:3].any()), ('bottom',mask[-3:].any()),
                                                ('left',mask[:,:3].any()), ('right',mask[:,-3:].any())] if present)
    scaffold_cells = {cell for component in multi for cell in component}
    assert all(tuple(cell) in scaffold_cells for cell in np.argwhere(added).tolist())
    return {'shape': inp.shape, 'palette': (bg, line), 'repairs': len(components),
            'axes': ''.join(sorted({'h' if dr == 0 else 'v' for (r, c, dr, dc), _ in evidence})),
            'figures': len(multi), 'damaged_figures': sum(any(added[r, c] for r, c in cells) for cells in multi),
            'cycles': cycles, 'intentional_gaps': len(_gaps(om)), 'clutter': len(erased)-len(evidence),
            'evidence': evidence, 'removed': erased, 'figure_cells': multi, 'kinds': tuple(kinds),
            'junctions': tuple(junctions), 'run_lengths': tuple(run_lengths),
            'boundary_runs': tuple(sorted(boundary_runs)), 'repair_edges': near_edges(added),
            'clutter_edges': near_edges(removed)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen_inputs, seen_pairs = set(), set()
    sources = {(_input_key(p['input']), _input_key(p['output'])) for p in _source_pairs()}
    train_infos = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            info = _analyze_pair(pair['input'], pair['output'])
            ik, ok = _input_key(pair['input']), _input_key(pair['output'])
            assert ik not in seen_inputs and (ik, ok) not in seen_pairs
            assert (ik, ok) not in sources
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
            if split == 'train':
                train_infos.append(info)
    # All geometry and palette roles are visible. Both regimes use the same
    # repair rule; training must also refute filling every gap and keeping decoys.
    assert any(x['intentional_gaps'] > 0 and x['clutter'] > 0 for x in train_infos)


def _source_pairs():
    pairs = []
    for bg, line, rows, added, removed in _SOURCE_FIXTURES:
        inp = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        out = inp.copy()
        for r, c in added:
            out[r, c] = line
        for r, c in removed:
            out[r, c] = bg
        pairs.append({'input': inp, 'output': out})
    return pairs


def self_test() -> None:
    # Exact source reconstruction includes the right vertical boundary repair.
    source = _source_pairs()
    for pair in source:
        assert np.array_equal(solve([], pair['input']), pair['output'])
        # Rotation/reflection changes no repair or deletion convention.
        for turns in range(4):
            for flip in (False, True):
                inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
                if flip:
                    inp, out = inp[:, ::-1], out[:, ::-1]
                assert np.array_equal(solve([], inp), out)
    source_infos = [_analyze_pair(p['input'],p['output']) for p in source]
    assert [x['repairs'] for x in source_infos] == [2,3,2]
    assert sum(j == 1 for x in source_infos for j in x['junctions']) == 6
    assert np.array_equal(solve(source[:2], source[2]['input']), source[2]['output'])
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in automatic}) >= 4
    assert len({len(p['test']) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) == 42
    infos = {'train': [], 'test': []}
    varying = {'shape': 0, 'palette': 0, 'repairs': 0, 'figures': 0, 'axes': 0, 'clutter': 0}
    simple_count = rich_count = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze_pair(pair['input'], pair['output'])
                infos[split].append(info)
                local.append(info)
                assert np.array_equal(solve([], pair['input']), pair['output'])
        for key in varying:
            varying[key] += len({x[key] for x in local}) > 1
        simple_count += (len({x['axes'] for x in local}) == 1
                         and len({x['palette'] for x in local}) <= 2
                         and max(x['figures'] for x in local) <= 2)
        rich_count += (len({x['axes'] for x in local}) >= 3
                       and len({x['palette'] for x in local}) >= 3
                       and len({x['figures'] for x in local}) >= 3)
        for pair in puzzle['test']:
            prediction = solve([{'input': p['input'].copy(), 'output': p['output'].copy()} for p in puzzle['train']], pair['input'].copy())
            assert np.array_equal(prediction, pair['output'])
    all_infos = infos['train'] + infos['test']
    assert len({x['shape'] for x in all_infos}) >= 70
    assert len({x['palette'] for x in all_infos}) >= 40
    assert min(min(x['shape']) for x in all_infos) <= 11
    assert max(x['figures'] for x in all_infos) >= 3
    assert max(x['damaged_figures'] for x in all_infos) >= 3
    assert any(x['cycles'] for x in all_infos) and any(x['cycles'] == 0 for x in all_infos)
    assert any(x['figures'] > x['damaged_figures'] >= 1 for x in all_infos)
    for split, rows in infos.items():
        assert {x['axes'] for x in rows} == {'h', 'v', 'hv'}
        assert max(x['repairs'] for x in rows) >= 3
        assert any(h == w for h, w in (x['shape'] for x in rows))
        assert max(x['shape'][0] for x in rows) == 30
        assert max(x['shape'][1] for x in rows) == 30
    assert all(v >= 12 for k, v in varying.items() if k != 'palette')
    assert varying['palette'] >= 15
    assert simple_count >= 3 and rich_count >= 2
    assert {len(c) for x in all_infos for c in x['removed']} == {2, 3}
    assert {k for x in all_infos for k in x['kinds']} == {'open_path','T','cross','branching','loop'}
    assert {e for x in all_infos for e in x['repair_edges']} == {'top','bottom','left','right'}
    assert {e for x in all_infos for e in x['clutter_edges']} == {'top','bottom','left','right'}
    assert len({length for x in all_infos for length in x['run_lengths']}) >= 12
    assert any(j == 1 for x in all_infos for j in x['junctions'])
    palette_sets = [{_roles(pair['input']) for s in ('train','test') for pair in p[s]} for p in automatic]
    assert sum(any(palette[::-1] in ps for palette in ps) for ps in palette_sets) >= 10
    assert sum(len(ps) == 1 for ps in palette_sets) >= 3
    assert sum(len(ps) >= 3 for ps in palette_sets) >= 10
    assert {(endpoint, side) for x in all_infos for _, matches in x['evidence'] for endpoint, side, _ in matches} == {(0,-1), (0,1), (4,-1), (4,1)}
    for n, t in [(2,1), (10,1), (10,4), (6,3), (None,1), (None,4), (2,None), (10,None)]:
        p = generate_puzzle(784, n, t)
        if n is not None:
            assert len(p['train']) == n
        if t is not None:
            assert len(p['test']) == t
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(784, n, t))
    assert _puzzle_key(generate_puzzle(784)) != _puzzle_key(generate_puzzle(785))
    for n, t in [(1,None), (11,None), (None,0), (None,5), (2,2), (3,4), (True,1), (2.5,1)]:
        try:
            generate_puzzle(3, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Omit one complete repair, including its deviation, from an otherwise valid
    # output. This specifically catches the review's partial-repair defect.
    original = next(p for p in automatic if any(len(_evidence(x['input'] == _roles(x['input'])[1])) >= 2 for x in p['train']))
    p = {s:[{k:v.copy() for k,v in x.items()} for x in original[s]] for s in ('train','test')}
    pair = next(x for x in p['train'] if len(_evidence(x['input'] == _roles(x['input'])[1])) >= 2)
    bg, line = _roles(pair['input'])
    (r,c,dr,dc), matches = _evidence(pair['input'] == line)[0]
    for k in (1,2,3):
        pair['output'][r+k*dr, c+k*dc] = bg
    for rr, cc in matches[0][2]:
        pair['output'][rr, cc] = line
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('omitted repair accepted')
# Immutable source evidence, used only for reconstruction regressions and exact
# content-based copy exclusion. These fixtures are never sampled as layouts.
_SOURCE_FIXTURES = (
    (1, 6, (
        '111161111111111111111111111111',
        '111111111111111111111111111111',
        '111111111111111111111111111111',
        '111161111111111111161111166666',
        '111161111111111111116111161111',
        '111161111111111111111611161111',
        '111161111111111111111111161111',
        '666111666666666111666111661111',
        '111111111111116116111111161111',
        '111111111111116161111111161111',
        '111111111111161111111111166666',
        '111111111111611111111111161111',
        '111111111111111111111111161111',
        '111111111111116111111111161111',
        '111111111111116111111111161111',
        '111111111111116111111111161111',
    ),
     ((7, 15), (7, 16), (7, 17), (10, 14), (11, 14), (12, 14)),
     ((3, 19), (4, 20), (5, 21), (8, 17), (9, 16), (10, 13), (11, 12))),
    (0, 3, (
        '000030000000000000300000',
        '000030000000000000300000',
        '000000000000000000300000',
        '000000000000000000300000',
        '000000000000000000300000',
        '000000000000000000000000',
        '000030000000000000003000',
        '000030000000000000030000',
        '000030000000000000300000',
        '000033300033333333330003',
        '000030000003000000003000',
        '000030000003000000000300',
        '000000000003000000000000',
        '000000000003000000000000',
        '000000000003000000000000',
        '000030000003000000000000',
        '000030000003000000000000',
        '000030000003000000030000',
        '000030000003000000300000',
        '000000000003000003000000',
        '000000000003000000000000',
        '000000000003000000000000',
        '000030000003000000000000',
        '000030000000300000000000',
        '000030000000030000000000',
        '000030000000000000000000',
        '000030000003000000000000',
        '000030000003000000000000',
    ),
     ((5, 18), (6, 18), (7, 18), (9, 20), (9, 21), (9, 22), (23, 11), (24, 11), (25, 11)),
     ((6, 20), (7, 19), (10, 20), (11, 21), (17, 19), (18, 18), (19, 17), (23, 12), (24, 13))),
    (8, 9, (
        '88888888888888988',
        '88888888888888988',
        '88888888888888988',
        '99888998888888988',
        '88889898888888988',
        '88898898888888988',
        '88888898888888988',
        '88888898888888988',
        '88888898888888988',
        '88888898888888988',
        '88888898888888988',
        '88888898888898988',
        '88888898888988988',
        '88888898888888988',
        '99999999998889988',
        '88988888888888988',
        '88898888888888988',
        '88888888888888988',
        '88888888888888988',
        '88888888888888988',
        '88888888888888898',
        '88888888888888889',
        '88888888888888888',
        '88888888888888988',
        '88888888888888988',
        '88888888888888988',
        '88888888888888888',
        '88888888888888988',
        '88888888888888988',
    ),
     ((3, 2), (3, 3), (3, 4), (20, 14), (21, 14), (22, 14)),
     ((4, 4), (5, 3), (11, 12), (12, 11), (15, 2), (16, 3), (20, 15), (21, 16))),
)
