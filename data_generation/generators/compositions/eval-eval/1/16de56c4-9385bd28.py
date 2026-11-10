"""Nested ripple stencils: periodic rectangular shells and ordered occlusion.

The table is vertical, cue on the left and reference ink on the right, with
earlier rows on top. Two ink dots on each box's NW inward diagonal encode
period and phase in *depth from the boundary*. An aligned accent dot bounds
and recolors the shells. Unselected depths are transparent.

Only NumPy and the standard library are used. Generation paints sampled
geometry; solve reconstructs everything from the input and selects the first
covering table row per pixel. No parent module or external file is loaded.
"""

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        if not _integer(value) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.flat))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _corners(box, diagonal):
    r0, c0, r1, c1 = box
    if diagonal == 0:
        return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
                (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}
    return {(r0, c1), (r0, c1 - 1), (r0 + 1, c1),
            (r1, c0), (r1, c0 + 1), (r1 - 1, c0)}


def _inside(cell, box):
    r, c = cell
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _decode_candidate(grid, bg, table):
    tr, tc, n = table
    cues = list(map(int, grid[tr:tr+n, tc]))
    inks = list(map(int, grid[tr:tr+n, tc+1]))
    if (len(set(cues)) != n or len(set(inks)) != n or set(cues) & set(inks)
            or bg in cues + inks):
        return None
    scene = grid.copy()
    scene[tr:tr+n, tc:tc+2] = bg
    objects = []
    for cue, ink in zip(cues, inks):
        coords = {tuple(map(int, rc)) for rc in np.argwhere(scene == cue)}
        if len(coords) != 6:
            return None
        r0, c0 = map(int, np.min(list(coords), axis=0))
        r1, c1 = map(int, np.max(list(coords), axis=0))
        box = r0, c0, r1, c1
        if min(r1-r0+1, c1-c0+1) < 5:
            return None
        # Local corner recognition is separate from the construction helper.
        h, w = r1-r0+1, c1-c0+1
        local = {(r-r0, c-c0) for r, c in coords}
        patterns = ({(0, 0), (0, 1), (1, 0), (h-1, w-1), (h-1, w-2), (h-2, w-1)},
                    {(0, w-1), (0, w-2), (1, w-1), (h-1, 0), (h-1, 1), (h-2, 0)})
        if local not in patterns:
            return None
        # The table has a one-cell moat from every entire rectangle.
        expanded = r0-1, c0-1, r1+1, c1+1
        if any(_inside((r, c), expanded) for r in range(tr, tr+n)
               for c in (tc, tc+1)):
            return None
        D = (min(h, w)-1)//2
        refs = [tuple(map(int, rc)) for rc in np.argwhere(scene == ink)]
        if len(refs) != 2:
            return None
        if any(r-r0 != c-c0 or not 1 <= r-r0 <= D for r, c in refs):
            return None
        a, b = sorted(r-r0 for r, c in refs)
        p = b-a
        if p <= 0:
            return None
        objects.append(dict(cue=cue, ink=ink, box=box, markers=coords,
                            diagonal=patterns.index(local), D=D, a=a, b=b, p=p))
    extras = [(int(r), int(c), int(scene[r, c])) for r, c in
              np.argwhere(~np.isin(scene, [bg] + cues + inks))]
    if len({color for r, c, color in extras}) > 1:
        return None
    for obj in objects:
        obj['extras'] = []
    for r, c, color in extras:
        owners = [o for o in objects if r-o['box'][0] == c-o['box'][1]
                  and 1 <= r-o['box'][0] <= o['D']]
        if len(owners) != 1:
            return None
        o = owners[0]
        o['extras'].append((r-o['box'][0], color))
    for o in objects:
        aligned = [(d, color) for d, color in o['extras'] if (d-o['a']) % o['p'] == 0]
        if len(aligned) > 1 or any(o['a'] < d < o['b'] for d, color in aligned):
            return None
        o['t'], o['accent'] = aligned[0] if aligned else (None, None)
        o['noise'] = tuple(d for d, color in o['extras'] if (d-o['a']) % o['p'])
    return dict(background=bg, table=table, objects=objects)


def _parse(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size // 2:
        raise ValueError("input must have a strict majority background")
    # A cue occurs six times in its corners and once in the table; an ink
    # occurs twice in its probes and once in the table. These are role counts,
    # not special literal colors. Validate each possible complete table.
    left = counts[grid[:, :-1]] == 7
    right = counts[grid[:, 1:]] == 3
    starts = np.argwhere(left & right & (grid[:, :-1] != grid[:, 1:]))
    candidates = []
    for r, c in starts:
        r, c = int(r), int(c)
        for n in range(1, min(4, grid.shape[0]-r)+1):
            if not np.all(left[r:r+n, c] & right[r:r+n, c]):
                break
            parsed = _decode_candidate(grid, bg, (r, c, n))
            if parsed is not None:
                candidates.append(parsed)
    if len(candidates) != 1:
        raise ValueError("input does not have one unambiguous table and stencil interpretation")
    return candidates[0]


def _active_depth(o, d):
    if (d-o['a']) % o['p']:
        return False
    return o['t'] is None or min(o['a'], o['t']) <= d <= max(o['b'], o['t'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover public geometry and pick the highest opaque stencil per pixel."""
    grid = _grid(input_grid)
    parsed = _parse(grid)
    result = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for o in parsed['objects']:
                if not _inside((r, c), o['box']):
                    continue
                if (r, c) in o['markers']:
                    result[r, c] = o['cue']
                    break
                r0, c0, r1, c1 = o['box']
                if r-r0 == c-c0 and r-r0 in o['noise']:
                    result[r, c] = next(color for d, color in o['extras'] if d == r-r0)
                    break
                d = min(r-r0, c-c0, r1-r, c1-c)
                if _active_depth(o, d):
                    result[r, c] = o['ink'] if o['t'] is None else o['accent']
                    break
    return result


def _compose(grid, objects, bg, alternative=None):
    """Generation compositor; alternatives are deliberately wrong audit rules."""
    result = grid.copy()
    ordering = objects if alternative == 'reverse_priority' else reversed(objects)
    for o in ordering:
        r0, c0, r1, c1 = o['box']
        rr, cc = np.indices((r1-r0+1, c1-c0+1))
        depth = np.minimum.reduce((rr, cc, r1-r0-rr, c1-c0-cc))
        a, b, p, t = o['a'], o['b'], o['p'], o['t']
        if alternative == 'short_period' and p > 2:
            p = 2
        paint = o['ink'] if t is None else o['accent']
        if alternative == 'ignore_terminal':
            t, paint = None, o['ink']
        if alternative == 'noise_is_terminal' and o['noise']:
            t = o['noise'][0]
            paint = next(color for d, color in o['extras'] if d == t)
        phase = 0 if alternative == 'zero_phase' else a
        mask = (depth-phase) % p == 0
        if t is not None and alternative != 'unbounded_terminal':
            mask &= (depth >= min(a, t)) & (depth <= max(b, t))
        if alternative == 'reference_span' and t is None:
            mask &= (depth >= a) & (depth <= b)
        if alternative == 'solid_box':
            mask[:] = True
        if alternative == 'diagonal_only':
            mask &= (rr == cc) & (rr <= o['D'])
        target = result[r0:r1+1, c0:c1+1]
        if alternative == 'opaque_gaps':
            target[:] = bg
        target[mask] = paint
        if alternative != 'erase_markers':
            for r, c in o['markers']:
                result[r, c] = o['cue']
        for d, color in o['extras']:
            if d in o['noise']:
                result[r0+d, c0+d] = color
    return result


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _signal(rng, D, p, terminal, direction=0, both=False, zero=False, nonzero=False, stop=False):
    choices = []
    for a in range(1, D-p+1):
        b = a+p
        phase = list(range(a % p, D+1, p))
        if zero and a % p != 0 or nonzero and a % p == 0:
            continue
        if not terminal:
            if len(phase) >= 3 and (not both or (phase[0] < a and phase[-1] > b)):
                choices.append((a, b, None))
        else:
            for t in range(1, D+1):
                if (t-a) % p or a <= t <= b:
                    continue
                if direction and (1 if t > b else -1) != direction:
                    continue
                if stop and not any(d < min(a, t) or d > max(b, t) for d in phase):
                    continue
                choices.append((a, b, t))
    return None if not choices else _pick(rng, choices)


def _recipes(rng, kind):
    n = int(_pick(rng, (2, 2, 3))) if kind in ('boundary', 'terminal') else int(rng.integers(1, 4))
    if kind == 'test':
        n = int(rng.integers(2, 5))
    recipes = []
    for i in range(n):
        p = int(rng.integers(1, 3))
        terminal = bool(rng.integers(2))
        recipe = dict(p=p, terminal=terminal, direction=0, both=False,
                      zero=False, nonzero=False, stop=terminal,
                      noise=(p > 1 and rng.random() < .3), minD=2*p+1)
        if kind == 'boundary' and i == 0:
            recipe.update(p=2, terminal=False, both=True, zero=True, noise=True, minD=6)
        elif kind == 'boundary' and i == 1:
            recipe.update(terminal=True, minD=2*p+2)
        elif kind == 'terminal' and i == 0:
            recipe.update(p=1, terminal=True, direction=1, minD=4, noise=False)
        elif kind == 'terminal' and i == 1:
            recipe.update(p=2, terminal=True, direction=-1, nonzero=True,
                          minD=7, noise=True)
        elif kind == 'test' and i == 0:
            p = int(rng.integers(3, 6))
            recipe.update(p=p, terminal=terminal, minD=2*p+1, stop=False,
                          noise=bool(rng.integers(2)))
        elif kind == 'test' and i == 1:
            recipe['terminal'] = not recipes[0]['terminal']
            recipe['stop'] = recipe['terminal']
        recipes.append(recipe)
    return recipes


def _candidate(rng, kind, profile):
    recipes = _recipes(rng, kind)
    n = len(recipes)
    minimum = 2*max(o['minD'] for o in recipes)+1
    low = max(16, minimum+3)
    if low > 30:
        return None
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    # Occasional squares and independent margins; either canvas axis may be longer.
    if rng.random() < .2:
        w = h
    objects, used = [], set()
    palette = list(map(int, rng.permutation(10)))
    bg, accent = palette[0], palette[-1]
    for index, recipe in enumerate(recipes):
        minside = 2*recipe['minD']+1
        # Compact, elongated and nearly canvas-wide boxes are all possible.
        upper_h, upper_w = max(minside, h-3), max(minside, w-3)
        hh = int(rng.integers(minside, upper_h+1))
        ww = int(rng.integers(minside, upper_w+1))
        if profile == 'compact' and rng.random() < .6:
            hh, ww = min(hh, minside+2), min(ww, minside+2)
        r0 = int(_pick(rng, (0, h-hh, int(rng.integers(h-hh+1)))))
        c0 = int(_pick(rng, (0, w-ww, int(rng.integers(w-ww+1)))))
        box = r0, c0, r0+hh-1, c0+ww-1
        diagonal = int(rng.integers(2))
        markers = _corners(box, diagonal)
        D = (min(hh, ww)-1)//2
        signal = _signal(rng, D, recipe['p'], recipe['terminal'], recipe['direction'],
                         recipe['both'], recipe['zero'], recipe['nonzero'], recipe['stop'])
        if signal is None:
            return None
        a, b, t = signal
        noise = []
        if recipe['noise']:
            choices = [d for d in range(1, D+1) if (d-a) % recipe['p']]
            if not choices:
                return None
            noise = [int(_pick(rng, choices))]
        extras = [(d, accent) for d in noise + ([] if t is None else [t])]
        dots = {(r0+d, c0+d) for d in [a, b] + [d for d, color in extras]}
        if (markers & dots) or (markers | dots) & used:
            return None
        used |= markers | dots
        objects.append(dict(cue=palette[1+index], ink=palette[1+n+index], box=box,
                            markers=markers, diagonal=diagonal, D=D, a=a, b=b,
                            p=recipe['p'], t=t, accent=accent if t is not None else None,
                            extras=extras, noise=tuple(noise)))
    # Accent dots must have a unique depth-diagonal owner. Other layers can
    # cover them in the output, but cannot supply a second input interpretation.
    for o in objects:
        for d, color in o['extras']:
            r, c = o['box'][0]+d, o['box'][1]+d
            owners = [q for q in objects if r-q['box'][0] == c-q['box'][1]
                      and 1 <= r-q['box'][0] <= q['D']]
            if len(owners) != 1:
                return None
    blocked = np.zeros((h, w), dtype=bool)
    for o in objects:
        r0, c0, r1, c1 = o['box']
        blocked[max(0, r0-1):min(h, r1+2), max(0, c0-1):min(w, c1+2)] = True
    locations = [(r, c) for r in range(h-n+1) for c in range(w-1)
                 if not blocked[r:r+n, c:c+2].any()]
    if not locations:
        return None
    tr, tc = _pick(rng, locations)
    grid = np.full((h, w), bg, dtype=np.int8)
    objects = [objects[int(i)] for i in rng.permutation(n)]
    for i, o in enumerate(objects):
        grid[tr+i, tc:tc+2] = o['cue'], o['ink']
        for r, c in o['markers']:
            grid[r, c] = o['cue']
        for d in (o['a'], o['b']):
            grid[o['box'][0]+d, o['box'][1]+d] = o['ink']
        for d, color in o['extras']:
            grid[o['box'][0]+d, o['box'][1]+d] = color
    output = _compose(grid, objects, bg)
    try:
        parsed = _parse(grid)
    except ValueError:
        return None
    if not np.array_equal(output, solve(grid)) or np.array_equal(output, grid):
        return None
    evidence = _evidence(grid, output, parsed)
    if kind == 'boundary' and not {'reverse_priority', 'opaque_gaps', 'reference_span'} <= evidence:
        return None
    if kind == 'terminal' and not {'ignore_terminal', 'unbounded_terminal', 'zero_phase', 'noise_is_terminal'} <= evidence:
        return None
    if kind == 'test' and not {'solid_box', 'reverse_priority', 'ignore_terminal'} <= evidence:
        return None
    if kind == 'test' and not _long_period_visible(grid, output, parsed):
        return None
    return dict(input=grid, output=output)


_ALTERNATIVES = ('solid_box', 'diagonal_only', 'reverse_priority', 'opaque_gaps',
                 'ignore_terminal', 'unbounded_terminal', 'reference_span',
                 'zero_phase', 'noise_is_terminal', 'erase_markers')


def _evidence(grid, output, parsed):
    return {name for name in _ALTERNATIVES if not np.array_equal(
        output, _compose(grid, parsed['objects'], parsed['background'], name))}


def _long_period_visible(grid, output, parsed):
    """A tested long cadence must visibly draw at least two distinct shells.

    Merely hiding a larger-period box behind solid upper layers is not a shift.
    Count newly painted, nonannotation pixels of its highest visible mask.
    """
    if np.array_equal(output, _compose(grid, parsed['objects'], parsed['background'], 'short_period')):
        return False
    depths = [set() for o in parsed['objects']]
    for r, c in np.argwhere(grid != output):
        for i, o in enumerate(parsed['objects']):
            if not _inside((r, c), o['box']):
                continue
            if (r, c) in o['markers']:
                break
            r0, c0, r1, c1 = o['box']
            if r-r0 == c-c0 and r-r0 in o['noise']:
                break
            d = min(r-r0, c-c0, r1-r, c1-c)
            if _active_depth(o, d):
                if o['p'] > 2 and (r-r0 != c-c0 or d not in (o['a'], o['b'], o['t'])):
                    depths[i].add(int(d))
                break
    return any(len(ds) >= 2 for ds in depths)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    profile = str(_pick(rng, ('compact', 'free', 'free')))
    kinds = ['free'] * num_train
    first, second = map(int, rng.choice(num_train, 2, replace=False))
    kinds[first], kinds[second] = 'boundary', 'terminal'
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', kinds), ('test', ['test'] * num_test)):
        for kind in requests:
            for _ in range(1200):
                pair = _candidate(rng, kind, profile)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded stencil construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, decoded geometry, semantic witnesses and the sampling shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, witnesses = set(), set()
    train_directions, test_maxima = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            key = _key(grid)
            assert key not in seen, 'inputs must be distinct within a puzzle'
            seen.add(key)
            parsed = _parse(grid)
            assert np.array_equal(solve(grid), output), 'wrong ring mask, paint, or priority'
            objects = parsed['objects']
            assert len({o['box'] for o in objects}) == len(objects)
            for o in objects:
                phases = list(range(o['a'] % o['p'], o['D']+1, o['p']))
                if o['t'] is None:
                    assert len(phases) >= 3, 'boundary stencil must add a ring'
                else:
                    assert o['accent'] not in (o['cue'], o['ink'], parsed['background'])
                assert all(not _active_depth(o, d) for d in o['noise'])
            # Each changed pixel is on some opaque shell or retained marker.
            for r, c in np.argwhere(grid != output):
                assert any(_inside((r, c), o['box']) and
                           ((r, c) in o['markers'] or _active_depth(o, min(
                               r-o['box'][0], c-o['box'][1], o['box'][2]-r, o['box'][3]-c)))
                           for o in objects)
            tr, tc, n = parsed['table']
            assert np.array_equal(grid[tr:tr+n, tc:tc+2], output[tr:tr+n, tc:tc+2])
            if split == 'train':
                assert max(o['p'] for o in objects) <= 2
                witnesses |= _evidence(grid, output, parsed)
                train_directions |= {1 if o['t'] > o['b'] else -1 for o in objects if o['t'] is not None}
            else:
                test_maxima.append(max(o['p'] for o in objects))
                assert any(o['t'] is None for o in objects) and any(o['t'] is not None for o in objects)
                assert {'solid_box', 'reverse_priority', 'ignore_terminal'} <= _evidence(grid, output, parsed)
                assert _long_period_visible(grid, output, parsed), 'larger-period shells are hidden or inert'
    assert set(_ALTERNATIVES) <= witnesses, 'training admits a listed simpler alternative'
    assert train_directions == {-1, 1}
    assert all(p >= 3 for p in test_maxima), 'test must transfer to a larger spacing'


def _normal_structure(grid):
    roles = {}
    return grid.shape, tuple(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def _fixed_regression():
    """Small hand-specified depth arithmetic, including the collapsed inner shell."""
    grid = np.zeros((11, 15), dtype=np.int8)
    box = (0, 4, 10, 14)
    markers = _corners(box, 0)
    for rc in markers:
        grid[rc] = 2
    grid[1, 0:2] = [2, 3]
    grid[3, 7] = grid[5, 9] = 3  # depths 3 and 5, odd shells through depth 5
    expected = grid.copy()
    for r in range(11):
        for c in range(4, 15):
            d = min(r, c-4, 10-r, 14-c)
            if d in (1, 3, 5):
                expected[r, c] = 3
    for rc in markers:
        expected[rc] = 2
    assert np.array_equal(solve(grid), expected)
    # Outward terminal at depth 1 recolors all three odd rings, not even depths.
    grid[1, 5] = 8
    expected[(expected == 3) & (np.indices(expected.shape)[1] >= 4)] = 8
    assert np.array_equal(solve(grid), expected)
    # An off-phase dot at depth 2 is preserved.
    grid[2, 6] = expected[2, 6] = 8
    assert np.array_equal(solve(grid), expected)
    # The role colors, including a black ink on a nonblack background, permute.
    permutation = np.array([6, 1, 4, 0, 3, 5, 2, 7, 9, 8], dtype=np.int8)
    assert np.array_equal(solve(permutation[grid]), permutation[expected])
    # Two manually registered boxes: a lower ring is visible through an upper
    # gap, but disappears where both rings fire. Lower Ls may also disappear.
    two = np.zeros((18, 23), dtype=np.int8)
    two[2:4, 0:2] = [[2, 3], [5, 6]]
    upper, lower = (0, 4, 12, 16), (4, 9, 16, 21)
    for rc in _corners(upper, 0):
        two[rc] = 2
    for rc in _corners(lower, 0):
        two[rc] = 5
    two[2, 6] = two[4, 8] = 3
    two[5, 10] = two[7, 12] = 6
    out = solve(two)
    assert out[5, 10] == 6                 # lower depth 1 through upper depth 5
    assert out[6, 10] == 3                 # upper depth 6 covers lower depth 1
    assert out[7, 11] == 0                 # both depths are gaps
    assert out[4, 9] == 3                  # upper shell hides lower L
    assert out[12, 15] == 2                # upper L survives its own depth-0 ink
    two[2:4, 0:2] = two[2:4, 0:2][::-1]
    assert solve(two)[6, 10] == 6          # table order alone changes the answer
    # Family members beyond the sampler's period cap: period six, several
    # inert dots, a terminal at the deepest point, and a movable interior table.
    wide_period = np.zeros((30, 30), dtype=np.int8)
    for rc in _corners((0, 0, 26, 26), 1):
        wide_period[rc] = 2
    wide_period[4, 28:30] = [2, 3]
    wide_period[1, 1] = wide_period[7, 7] = 3
    wp = solve(wide_period)
    assert wp[13, 13] == 3 and wp[0, 0] == 0 and wp[1, 20] == 3
    wide_period[13, 13] = 8
    for d in (2, 3, 4):
        wide_period[d, d] = 8
    wp = solve(wide_period)
    assert wp[1, 20] == 8 and wp[7, 18] == 8 and wp[13, 13] == 8
    assert all(wp[d, d] == 8 for d in (2, 3, 4))
    # The deepest shell of an elongated five-wide box is a thirteen-cell line.
    narrow = np.zeros((17, 9), dtype=np.int8)
    narrow[1, 7:9] = [2, 3]
    for rc in _corners((0, 0, 16, 4), 0):
        narrow[rc] = 2
    narrow[1, 1] = narrow[2, 2] = 3
    result = solve(narrow)
    assert np.all(result[2:15, 2] == 3) and result[0, 0] == 2


def self_test():
    """Bounded 42-seed checks plus API boundaries and explicit semantic fixtures."""
    _fixed_regression()
    signatures, shapes, structures, backgrounds, counts, periods = set(), set(), set(), set(), set(), set()
    diagonals, tables, directions, topologies = set(), set(), set(), set()
    square_boxes, elongated, outer_clipped, inner_collapsed = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s])
        assert signature not in signatures
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g = pair['input']
                parsed = _parse(g)
                objects = parsed['objects']
                shapes.add(g.shape)
                structures.add(_normal_structure(g))
                backgrounds.add(parsed['background'])
                counts.add(len(objects))
                periods.update(o['p'] for o in objects)
                diagonals.update(o['diagonal'] for o in objects)
                tables.add(parsed['table'][:2])
                edges = sum(_overlap(a['box'], b['box']) for i, a in enumerate(objects) for b in objects[i+1:])
                topologies.add((len(objects), edges))
                for o in objects:
                    r0, c0, r1, c1 = o['box']
                    hh, ww = r1-r0+1, c1-c0+1
                    square_boxes += hh == ww
                    elongated += abs(hh-ww) >= 5
                    outer_clipped += o['a'] % o['p'] != 0
                    inner_collapsed += min(hh, ww) % 2 == 1 and _active_depth(o, o['D'])
                    if o['t'] is not None:
                        directions.add(1 if o['t'] > o['b'] else -1)
    assert len(signatures) == 42 and len(shapes) >= 45 and len(structures) >= 190
    assert backgrounds == set(range(10)) and counts == {1, 2, 3, 4}
    assert periods == {1, 2, 3, 4, 5} and diagonals == {0, 1} and directions == {-1, 1}
    assert len(tables) >= 60 and len(topologies) >= 7
    assert square_boxes >= 10 and elongated >= 10 and outer_clipped >= 40 and inner_collapsed >= 20
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7000+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None), (None, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, semantic fixtures)')
