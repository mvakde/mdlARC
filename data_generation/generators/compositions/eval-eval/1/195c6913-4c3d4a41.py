"""Folded cyclic tapes pushed by a comb, with material lost at the inlets.

Standard library and NumPy only. Solving reads geometry and roles from the
current input; generation schedules are not part of the solution rule.
"""

from collections import defaultdict
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)):
        raise ValueError("example counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape"
    assert np.issubdtype(a.dtype, np.integer), "grid must contain integers"
    assert np.all((a >= 0) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=True)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(component)))
    return result


def _swatches(a, bg):
    result = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for cells in _components(a == color):
            if len(cells) != 4:
                continue
            r = min(p[0] for p in cells)
            c = min(p[1] for p in cells)
            if set(cells) == {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}:
                result.append((r, c, color))
    return result


def _runs(values, color):
    positions = np.flatnonzero(values == color)
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def _ordered_path(cells, roof, base):
    cells = set(cells)
    neighbors = {}
    for r, c in cells:
        neighbors[(r, c)] = [p for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                             if p in cells]
    ends = [p for p in cells if len(neighbors[p]) == 1]
    assert len(ends) == 2 and all(1 <= len(v) <= 2 for v in neighbors.values())
    starts = [p for p in ends if p[0] == roof + 1]
    assert len(starts) == 1
    path, previous, current = [], None, starts[0]
    while True:
        assert current not in path
        path.append(current)
        nxt = [p for p in neighbors[current] if p != previous]
        if not nxt:
            break
        assert len(nxt) == 1
        q = nxt[0]
        assert (q[0]-current[0], q[1]-current[1]) in ((1, 0), (0, 1))
        previous, current = current, q
    assert len(path) == len(cells) and path[-1][0] == base-1
    assert len(path) >= 3 and path[1][0] == roof+2
    assert path[-2][1] == path[-1][1], "terminal leg must be vertical"
    return tuple(path)


def _terminal_length(path):
    col = path[-1][1]
    return next((j for j, p in enumerate(reversed(path)) if p[1] != col), len(path))


def _draw_frame(a, f):
    s, t, b, l, r = (f[k] for k in ('structure', 'roof', 'base', 'left', 'right'))
    a[t, l:r+1] = s
    a[t:b+1, l] = s
    a[t:b+1, r] = s


def _draw_comb(a, f, base, caps=False):
    columns = [p[-1][1] for p in f['paths']]
    a[base, columns[0]:columns[-1]+1] = f['structure']
    for c, h in zip(columns, f['heights']):
        a[base-h:base, c] = f['structure']
        if caps:
            a[base-h, c] = f['cap']


def _render_input(f):
    a = np.full(f['shape'], f['background'], dtype=np.int8)
    _draw_frame(a, f)
    _draw_comb(a, f, f['external_base'])
    for path, length in zip(f['paths'], f['lengths']):
        for j, p in enumerate(path):
            a[p] = f['neutral'] if j < length else f['channel']
        a[path[0]] = f['word'][0]
    for c, color in zip(f['legend_columns'], f['word']):
        a[f['legend_row']:f['legend_row']+2, c:c+2] = color
    r, c = f['cap_position']
    a[r:r+2, c:c+2] = f['cap']
    return a


def _render_output(f):
    """Direct material-coordinate formula used by construction."""
    a = np.full(f['shape'], f['background'], dtype=np.int8)
    _draw_frame(a, f)
    _draw_comb(a, f, f['base'], caps=True)
    word = f['word']
    for path, length, height in zip(f['paths'], f['lengths'], f['heights']):
        capacity = len(path)-height
        shift = length-capacity
        for i, p in enumerate(path[:capacity]):
            a[p] = word[(shift+i) % len(word)]
    return a


def _read_frame(a, bg, swatches, legend, marker, s, roof, left, right):
    assert right-left >= 6 and roof > legend[0][0]+2
    assert marker[1]+1 < left-1
    base = roof
    while base+1 < a.shape[0] and a[base+1, left] == s and a[base+1, right] == s:
        base += 1
    assert base-roof >= 5 and roof+1 <= marker[0] <= base-2
    inside = np.zeros_like(a, dtype=bool)
    inside[roof+1:base, left+1:right] = a[roof+1:base, left+1:right] != bg
    pieces = _components(inside)
    assert 2 <= len(pieces) <= 5
    paths = tuple(sorted((_ordered_path(p, roof, base) for p in pieces),
                         key=lambda p: p[0][1]))
    columns = tuple(p[-1][1] for p in paths)
    assert all(x < y for x, y in zip(columns, columns[1:]))
    word = tuple(v[2] for v in legend)
    roles_excluded = {bg, s, marker[2], *word}
    neutral = int(a[paths[0][1]])
    assert neutral not in roles_excluded
    lengths, channels = [], set()
    for path in paths:
        assert a[path[0]] == word[0]
        length = 1
        while length < len(path) and a[path[length]] == neutral:
            length += 1
        assert 2 <= length < len(path)
        channel = int(a[path[length]])
        assert channel not in roles_excluded | {neutral}
        assert all(a[p] == channel for p in path[length:])
        channels.add(channel)
        lengths.append(length)
    assert len(channels) == 1
    external_base = int(np.argwhere(a == s)[:, 0].max())
    assert external_base > base+1
    assert np.all(a[external_base, columns[0]:columns[-1]+1] == s)
    heights = []
    for col, path, length in zip(columns, paths, lengths):
        row = external_base-1
        while row >= 0 and a[row, col] == s:
            row -= 1
        h = external_base-1-row
        assert 1 <= h < _terminal_length(path)
        assert len(path)-h >= 2 and 1 <= len(path)-length <= h
        heights.append(h)
    assert external_base-max(heights) > base, "external teeth cannot overlap frame"
    f = dict(shape=a.shape, background=bg, structure=s, channel=channels.pop(),
             neutral=neutral, cap=marker[2], word=word, paths=paths,
             lengths=tuple(lengths), heights=tuple(heights), roof=roof, base=base,
             left=left, right=right, external_base=external_base,
             legend_row=legend[0][0], legend_columns=tuple(v[1] for v in legend),
             cap_position=marker[:2])
    assert len(set(word)) >= 2
    assert len({bg, s, f['channel'], neutral, f['cap']}) == 5
    assert np.array_equal(a, _render_input(f)), "extra cells or malformed hardware"
    return f


def _read_canonical(a, bg):
    swatches = _swatches(a, bg)
    groups = defaultdict(list)
    for swatch in swatches:
        groups[swatch[0]].append(swatch)
    legends = [sorted(v, key=lambda x: x[1]) for v in groups.values() if len(v) >= 2]
    assert len(legends) == 1
    legend = legends[0]
    assert 2 <= len(legend) <= 6 and all(b[1]-a[1] >= 3 for a, b in zip(legend, legend[1:]))
    markers = [p for p in swatches if p not in legend]
    assert len(markers) == 1 and markers[0][2] not in {p[2] for p in legend}
    marker = markers[0]
    excluded = {bg, *(p[2] for p in swatches)}
    candidates = []
    for s in set(map(int, np.unique(a))) - excluded:
        for roof in range(legend[0][0]+3, a.shape[0]-5):
            for left, right in _runs(a[roof], s):
                if right-left < 6:
                    continue
                try:
                    f = _read_frame(a, bg, swatches, legend, marker, s, roof, left, right)
                except (AssertionError, IndexError):
                    continue
                candidates.append(f)
    assert len(candidates) == 1, "no unique canonical hardware interpretation"
    return candidates[0]


def _decode(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for code in range(8):
        try:
            f = _read_canonical(_unorient(a, code), bg)
        except (AssertionError, IndexError):
            continue
        f['orientation'] = code
        candidates.append(f)
    assert len(candidates) == 1, "input must have exactly one interpretation"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover material distance and contact from this input alone."""
    try:
        f = _decode(input_grid)
    except (AssertionError, IndexError, TypeError) as error:
        raise ValueError("input is outside the folded-tape family") from error
    return _orient(_render_output(f), f['orientation'])


def _simulate(input_grid, f):
    """Stepwise clipping check using input pixels and an input-derived reading.

This does not call the direct output formula or receive sampled metadata.
It is an authored check, not a protected independent solver.
"""
    a = _unorient(np.asarray(input_grid), f['orientation'])
    out = a.copy()
    for c in f['legend_columns']:
        out[f['legend_row']:f['legend_row']+2, c:c+2] = f['background']
    r, c = f['cap_position']
    out[r:r+2, c:c+2] = f['background']
    moving = [(int(r), int(c)) for r, c in np.argwhere(a == f['structure'])
              if r > f['base']]
    for p in moving:
        out[p] = f['background']
    for path in f['paths']:
        for p in path:
            out[p] = f['background']
    for r, c in moving:
        out[r-(f['external_base']-f['base']), c] = a[r, c]
    for path, h in zip(f['paths'], f['heights']):
        length = 1
        while length < len(path) and a[path[length]] == f['neutral']:
            length += 1
        material = [f['word'][j % len(f['word'])] for j in range(length)]
        positions = list(range(length))
        contact = len(path)-h
        while positions[-1] >= contact:
            positions = [j-1 for j in positions]
            if positions[0] < 0:
                positions.pop(0)
                material.pop(0)
        assert positions[-1] == contact-1 and positions[0] == 0
        for j, color in zip(positions, material):
            out[path[j]] = color
        out[path[contact]] = f['cap']
    return _orient(out, f['orientation'])


def _leg_starts(path):
    """At a bend the corner belongs to the preceding leg, and counts once."""
    starts = {0}
    old = None
    for i in range(1, len(path)):
        d = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        if old is not None and d != old:
            starts.add(i)
        old = d
    return starts


def _evidence(f):
    word, unique = f['word'], tuple(dict.fromkeys(f['word']))
    shifts = [L-(len(p)-h) for p, L, h in zip(f['paths'], f['lengths'], f['heights'])]
    evidence = set()
    offset = 0
    for path, h, shift in zip(f['paths'], f['heights'], shifts):
        capacity = len(path)-h
        correct = [word[(shift+i) % len(word)] for i in range(capacity)]
        if capacity >= 2*len(word):
            evidence.add('cycling')
        if correct != [unique[(shift+i) % len(unique)] for i in range(capacity)]:
            evidence.add('multiplicity')
        if correct != [word[i % len(word)] for i in range(capacity)]:
            evidence.add('clip_phase')
        if correct != [word[(shift+path[i][0]-path[0][0]) % len(word)] for i in range(capacity)]:
            evidence.add('arclength')
        starts, last, reset = _leg_starts(path), 0, []
        for i in range(capacity):
            if i in starts:
                last = i
            reset.append(word[((shift+i) if last == 0 else i-last) % len(word)])
        if correct != reset:
            evidence.add('bend_phase')
        if correct != [word[(max(shifts)+i) % len(word)] for i in range(capacity)]:
            evidence.add('lane_specific')
        if correct != [word[(shift+offset+i) % len(word)] for i in range(capacity)]:
            evidence.add('restart')
        if shift == 0:
            evidence.add('stationary')
            if correct != [word[(1+i) % len(word)] for i in range(capacity)]:
                evidence.add('zero_contact')
        if shift > 0:
            evidence.add('moving')
        offset += capacity
    return evidence


_REQUIRED = {'cycling', 'multiplicity', 'clip_phase', 'arclength', 'bend_phase',
             'lane_specific', 'restart', 'stationary', 'moving', 'zero_contact'}


def _sample_word(rng, palette, length, repeated=False):
    upper = min(5, length-1 if repeated else length)
    distinct = int(rng.integers(2, upper+1))
    symbols = list(range(distinct)) + rng.integers(0, distinct, length-distinct).tolist()
    rng.shuffle(symbols)
    return tuple(palette[j] for j in symbols)


def _make_path(roof, base, col, offset, bend_rows, steps):
    path = [(roof+1, col)]
    r, c = path[0]
    for row, distance in zip(bend_rows, steps):
        while r < row:
            r += 1
            path.append((r, c))
        for _ in range(distance):
            c += 1
            path.append((r, c))
    while r < base-1:
        r += 1
        path.append((r, c))
    assert c-col == offset
    return tuple(path)


def _sample_features(rng, schedule, test=False, witness=False):
    choices = (4, 5) if test and schedule == 'lanes' else ((2, 3) if schedule == 'lanes' else (2, 3, 4))
    n = int(rng.choice(choices))
    length = int(rng.integers(5, 7) if test and schedule == 'word' else
                 rng.integers(2, 5) if schedule == 'word' else rng.integers(2, 7))
    if witness:
        n, length = int(rng.choice((2, 3))), int(rng.integers(3, 5))
    palette = list(map(int, rng.permutation(10)))
    bg, structure, channel, neutral, cap = palette[:5]
    word = _sample_word(rng, palette[5:], length, repeated=witness)
    legend_row = int(rng.integers(0, 3))
    roof = legend_row+4
    depth = int(rng.integers(10, 14) if test and schedule == 'folds' else rng.integers(6, 14))
    if schedule == 'folds' and not test:
        depth = int(rng.integers(6, 10))
    base = roof+depth+1
    left = int(rng.integers(3, 6))
    inset = int(rng.integers(1, 3))
    gaps = rng.integers(2, 4, n-1).tolist()
    # Allocate real horizontal path distance subject to the 30-cell canvas.
    budget = 29-left-2*inset-n-int(sum(gaps))+1
    assert budget >= 2
    offsets = [0]*n
    bent_lane = int(rng.integers(n))
    offsets[bent_lane] = 2 if test and schedule == 'folds' else 1
    budget -= offsets[bent_lane]
    for i in rng.permutation(n):
        extra = int(rng.integers(0, min(4-offsets[i], budget)+1))
        offsets[i] += extra
        budget -= extra
    starts, c = [], left+inset
    for i, off in enumerate(offsets):
        starts.append(c)
        c += off+1+(gaps[i] if i < n-1 else 0)
    right = c-1+inset
    paths = []
    for i, (col, off) in enumerate(zip(starts, offsets)):
        bends = 0 if off == 0 else int(rng.integers(1, min(3, off, depth-4)+1))
        if schedule == 'folds' and not test:
            bends = min(bends, 1)
        if test and schedule == 'folds' and i == bent_lane:
            bends = int(rng.integers(2, min(3, off)+1))
        rows = sorted(map(int, rng.choice(np.arange(roof+2, base-3), bends, replace=False)))
        if bends:
            cuts = sorted(map(int, rng.choice(np.arange(1, off), bends-1, replace=False)))
            steps = np.diff([0, *cuts, off]).tolist()
        else:
            steps = []
        paths.append(_make_path(roof, base, col, off, rows, steps))
    caps = [min(6, _terminal_length(p)-1, len(p)-2) for p in paths]
    mode = str(rng.choice(('flat', 'levels', 'free'), p=(.20, .15, .65)))
    if mode == 'flat':
        heights = [int(rng.integers(1, min(caps)+1))]*n
    elif mode == 'levels':
        low = int(rng.integers(1, min(caps)+1))
        heights = [low if rng.random() < .5 else cap_i for cap_i in caps]
    else:
        heights = [int(rng.integers(1, cap_i+1)) for cap_i in caps]
    gaps_tooth = [int(rng.integers(1, h+1)) for h in heights]
    if witness:
        moving = int(rng.integers(n))
        stationary = int(rng.choice([j for j in range(n) if j != moving]))
        heights[moving] = int(rng.integers(2, caps[moving]+1))
        gaps_tooth[moving] = int(rng.integers(1, heights[moving]))
        gaps_tooth[stationary] = heights[stationary]
    elif rng.random() < .18:
        gaps_tooth = heights.copy()
    lengths = tuple(len(p)-g for p, g in zip(paths, gaps_tooth))
    clearance = int(rng.integers(1, min(3, 29-base-max(heights))+1))
    external_base = base+max(heights)+clearance
    legend_left = int(rng.integers(0, 4))
    columns = [legend_left]
    for _ in range(len(word)-1):
        columns.append(columns[-1]+int(rng.integers(3, 5)))
    width = max(right+1, columns[-1]+2)+int(rng.integers(0, 3))
    width = min(30, width)
    assert columns[-1]+2 <= width and right < width
    height = external_base+1+int(rng.integers(0, min(2, 29-external_base)+1))
    marker = (int(rng.integers(roof+1, base-1)), int(rng.integers(0, left-2)))
    return dict(shape=(height, width), background=bg, structure=structure,
                channel=channel, neutral=neutral, cap=cap, word=word,
                paths=tuple(paths), lengths=lengths, heights=tuple(heights),
                roof=roof, base=base, left=left, right=right,
                external_base=external_base, legend_row=legend_row,
                legend_columns=tuple(columns), cap_position=marker)


def _fixture():
    """Small explicit regression: s=(2,0), two unequal bent channels, AAB."""
    return dict(shape=(19, 17), background=0, structure=5, channel=1, neutral=8,
                cap=9, word=(2, 2, 4), roof=4, base=13, left=3, right=16,
                external_base=17, legend_row=0, legend_columns=(0, 3, 6),
                cap_position=(9, 0),
                paths=(_make_path(4, 13, 5, 2, [6], [2]),
                       _make_path(4, 13, 10, 3, [8], [3])),
                lengths=(9, 9), heights=(3, 2))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    schedule = str(rng.choice(('lanes', 'word', 'folds')))
    orientation_policy = str(rng.choice(('fixed', 'mixed', 'transfer'), p=(.45, .35, .20)))
    orientation = int(rng.integers(8))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            witness = split == 'train' and i == 0
            for attempt in range(256):
                f = _sample_features(rng, schedule, split == 'test', witness)
                if witness and not _REQUIRED <= _evidence(f):
                    continue
                code = orientation
                if orientation_policy == 'mixed':
                    code = int(rng.integers(8))
                elif orientation_policy == 'transfer' and split == 'test':
                    code = (orientation+int(rng.integers(1, 8))) % 8
                inp = _orient(_render_input(f), code)
                if _key(inp) in seen:
                    continue
                # A rendered input must really support the sampled interpretation.
                parsed = _decode(inp)
                out = _orient(_render_output(f), code)
                assert np.array_equal(out, _orient(_render_output(parsed), parsed['orientation']))
                result[split].append({'input': inp, 'output': out})
                seen.add(_key(inp))
                break
            else:
                # Finite witness backstop, never used as a hidden semantic mode.
                if not witness:
                    raise RuntimeError("distinct scene sampling exhausted")
                f = _fixture()
                palette = np.asarray(rng.permutation(10), dtype=np.int8)
                inp, out = palette[_render_input(f)], palette[_render_output(f)]
                inp, out = _orient(inp, orientation), _orient(out, orientation)
                result[split].append({'input': inp, 'output': out})
                seen.add(_key(inp))
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training_evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _decode(inp)
            assert np.array_equal(solve(inp), out), "incorrect material-coordinate output"
            assert np.array_equal(_simulate(inp, f), out), "stepwise transport disagrees"
            canonical = _unorient(out, f['orientation'])
            assert np.count_nonzero(canonical == f['cap']) == len(f['paths'])
            assert not np.any(canonical == f['neutral']) and not np.any(canonical == f['channel'])
            for path, L, h in zip(f['paths'], f['lengths'], f['heights']):
                C, s = len(path)-h, L-(len(path)-h)
                assert s >= 0 and C+s == L
                assert canonical[path[C]] == f['cap']
                assert all(canonical[p] == f['structure'] for p in path[C+1:])
                assert sum(canonical[p] in f['word'] for p in path) == C
            if split == 'train':
                training_evidence |= _evidence(f)
    assert _REQUIRED <= training_evidence, _REQUIRED-training_evidence


def _pattern(word):
    mapping = {}
    return tuple(mapping.setdefault(c, len(mapping)) for c in word)


def self_test():
    """Bounded semantic fixtures, count checks, and observed 42-seed variation."""
    f = _fixture()
    assert _REQUIRED <= _evidence(f)
    raw, expected = _render_input(f), _render_output(f)
    assert tuple(expected[p] for p in f['paths'][0][:7]) == (4, 2, 2, 4, 2, 2, 4)
    assert tuple(expected[p] for p in f['paths'][1][:9]) == (2, 2, 4, 2, 2, 4, 2, 2, 4)
    assert expected[f['paths'][0][7]] == 9
    for code in range(8):
        palette = np.array([(3*j+code) % 10 for j in range(10)], dtype=np.int8)
        inp = _orient(palette[raw], code)
        assert np.array_equal(solve(inp), _orient(palette[expected], code))
        assert np.array_equal(_simulate(inp, _decode(inp)), solve(inp))
    malformed = raw.copy()
    malformed[f['paths'][0][3]] = f['background']
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("broken channel accepted")
    # Semantic solver accepts straight, stationary, flat hardware as well.
    straight = dict(f, paths=(_make_path(4, 13, 5, 0, [], []),
                             _make_path(4, 13, 12, 0, [], [])),
                    heights=(2, 2), lengths=(6, 6))
    assert np.array_equal(solve(_render_input(straight)), _render_output(straight))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(20261007, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (3, True), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    signatures, shapes, patterns, profiles, counts, bends, codes = (set() for _ in range(7))
    palettes, geometry = set(), set()
    stationary = moving = flat = tied = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _decode(pair['input'])
                shapes.add(f['shape'])
                patterns.add(_pattern(f['word']))
                profiles.add(f['heights'])
                counts.add(len(f['paths']))
                codes.add(f['orientation'])
                palettes.add((f['background'], f['structure'], f['channel'], f['neutral'], f['cap']))
                geometry.add((f['paths'], f['heights'], f['lengths']))
                flat += len(set(f['heights'])) == 1
                tied += f['heights'].count(max(f['heights'])) >= 2
                for path, L, h in zip(f['paths'], f['lengths'], f['heights']):
                    shift = L-(len(path)-h)
                    stationary += shift == 0
                    moving += shift > 0
                    bends.add(sum(path[i][0] == path[i-1][0] and
                                  (i == 1 or path[i-1][0] != path[i-2][0])
                                  for i in range(1, len(path))))
    assert len(signatures) == 42 and len(shapes) >= 30
    assert counts == {2, 3, 4, 5} and codes == set(range(8))
    assert bends == {0, 1, 2, 3}
    assert len(patterns) >= 10 and len(profiles) >= 60 and len(geometry) >= 150
    assert stationary >= 80 and moving >= 80 and flat >= 10 and tied >= 30
    assert len(palettes) >= 100
    # Reject correct-shape outputs that restart phase after real clipping.
    puzzle = generate_puzzle(0, 2, 1)
    witness = next(p for p in puzzle['train'] if 'clip_phase' in _evidence(_decode(p['input'])))
    f = _decode(witness['input'])
    wrong = _unorient(witness['output'], f['orientation'])
    for path, h in zip(f['paths'], f['heights']):
        for i, p in enumerate(path[:len(path)-h]):
            wrong[p] = f['word'][i % len(f['word'])]
    witness['output'] = _orient(wrong, f['orientation'])
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("restart-after-clipping mutation accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                word_patterns=len(patterns), profiles=len(profiles),
                material_geometries=len(geometry), role_palettes=len(palettes),
                lane_counts=sorted(counts), horizontal_leg_counts=sorted(bends),
                orientations=sorted(codes), stationary_lanes=stationary,
                moving_lanes=moving, flat_profiles=flat, tied_maxima=tied)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
