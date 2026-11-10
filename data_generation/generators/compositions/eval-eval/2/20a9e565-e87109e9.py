"""A growing comb transported along a keyed, turning carrier.

Only standard library and NumPy. Generation retains at most one puzzle and a
bounded number of small candidate arrays; it performs no file or network I/O.
"""
import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _need(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _need(np.all((0 <= a) & (a <= 9)), "color bounds")
    return a


def _corners(box):
    a, b, c, d = box
    return {(a, c), (a, c + 1), (a + 1, c),
            (b, d), (b, d - 1), (b - 1, d)}


def _rectangles(field, colors):
    seen, rectangles = set(), []
    h, w = field.shape
    for rr, cc in np.argwhere(np.isin(field, list(colors))):
        origin = (int(rr), int(cc))
        if origin in seen:
            continue
        color = int(field[origin])
        todo, cells = [origin], []
        seen.add(origin)
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and field[q] == color:
                    seen.add(q)
                    todo.append(q)
        lo, hi = np.min(cells, axis=0), np.max(cells, axis=0) + 1
        _need(len(cells) == int(np.prod(hi - lo)), "nonrectangular block")
        rectangles.append((int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), color))
    return rectangles


def _decode(input_grid):
    grid = _grid(input_grid)
    frame = int(grid[0, 0])
    _need(np.all(grid[0] == frame), "top key frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _need(len(bottoms) > 0, "bottom key frame")
    bottom = int(bottoms[0])
    _need(2 <= bottom < len(grid) - 1, "key height")
    band = grid[1:bottom]
    _need(np.all(band == band[0]), "key bars must span their interiors")
    row = band[0]
    _need(row[0] == row[-1] == frame, "key side walls")
    runs, c = [], 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((start, c))
    _need(runs and all(b - a >= 3 for a, b in runs), "legible key panels")
    background = int(row[runs[0][0] + 1])
    turns = {}
    for a, b in runs:
        _need(np.all(row[a + 1:b - 1] == background), "key interior")
        left, right = int(row[a]), int(row[b - 1])
        _need((left == background) != (right == background), "one attached bar per panel")
        color, side = (right, 1) if left == background else (left, -1)
        _need(color not in turns and color not in (frame, background), "unique key color")
        turns[color] = side
    field = grid[bottom + 1:].copy()
    markers = []
    for color in set(map(int, np.unique(field))) - set(turns) - {frame, background}:
        p = np.argwhere(field == color)
        if len(p) != 6:
            continue
        a, c = map(int, p.min(axis=0))
        b, d = map(int, p.max(axis=0))
        box = a, b, c, d
        if b > a and d > c and b - a + d - c >= 4 and set(map(tuple, p)) == _corners(box):
            markers.append((color, box))
    _need(len(markers) == 1, "unique pair of disconnected L markers")
    marker, box = markers[0]
    field[field == marker] = background
    a, b, c, d = box
    _need(np.all(field[a:b + 1, c:d + 1] == background), "request window must be erased")
    others = set(map(int, np.unique(field))) - set(turns) - {background}
    _need(len(others) == 2 and frame not in others, "source and comb ink roles")
    singles = [v for v in others if np.count_nonzero(field == v) == 1]
    _need(len(singles) == 1, "unique singleton source")
    source = singles[0]
    ink = next(iter(others - {source}))
    r, c = map(int, np.argwhere(field == source)[0])
    h, w = field.shape
    headings = [i for i, (dr, dc) in enumerate(DIRS)
                if 0 <= r + dr < h and 0 <= c + dc < w and field[r + dr, c + dc] == ink]
    _need(len(headings) == 1, "prefix must give a unique launch heading")
    heading = headings[0]
    dr, dc = DIRS[heading]
    length = 0
    while 0 <= r + dr * (length + 1) < h and 0 <= c + dc * (length + 1) < w:
        if field[r + dr * (length + 1), c + dc * (length + 1)] != ink:
            break
        length += 1
    stations = {-1: [], 1: []}
    visible_mask = np.zeros(field.shape, dtype=bool)
    for t in range(1, length + 1):
        rr, cc = r + t * dr, c + t * dc
        visible_mask[rr, cc] = True
        for side in (-1, 1):
            tr, tc = DIRS[(heading + side) % 4]
            n = 0
            while 0 <= rr + tr * (n + 1) < h and 0 <= cc + tc * (n + 1) < w:
                if field[rr + tr * (n + 1), cc + tc * (n + 1)] != ink:
                    break
                n += 1
                visible_mask[rr + tr * n, cc + tc * n] = True
            if n:
                stations[side].append((t, n))
    sides = [s for s in stations if stations[s]]
    _need(len(sides) == 1, "teeth must occupy one local side")
    side = sides[0]
    observed = stations[side]
    _need(len(observed) >= 3, "at least three complete teeth")
    spacing = observed[0][0]
    first = observed[0][1]
    delta = observed[1][1] - first
    _need(delta >= 0, "teeth must have nonnegative arithmetic growth")
    _need(spacing >= 2 and length == len(observed) * spacing, "complete regular prefix")
    _need(all(t == (i + 1) * spacing and n == first + i * delta
              for i, (t, n) in enumerate(observed)), "arithmetic teeth")
    _need(np.array_equal(visible_mask, field == ink), "unexplained prefix ink")
    return dict(field=field, background=background, frame=frame, source=source,
                ink=ink, marker=marker, box=box, turns=turns, start=(r, c),
                heading=heading, side=side, spacing=spacing, first=first, delta=delta,
                visible_length=length, visible_count=len(observed), header=bottom + 1,
                rectangles=_rectangles(field, turns))


def _trace(q, mapping=None, stop_at_block=False):
    """Input-only carrier tracer. Old ink and the request markers are transparent."""
    field, turns = q['field'], q['turns']
    r, c = q['start']
    d = q['heading']
    h, w = field.shape
    path, events, seen = [], [], set()
    for _ in range(4 * h * w + 1):
        state = r, c, d
        _need(state not in seen, "cyclic carrier")
        seen.add(state)
        dr, dc = DIRS[d]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return path, events
        color = int(field[rr, cc])
        if color in turns:
            if stop_at_block:
                return path, events
            _need(not events or len(path) > events[-1][4], "no travel between corners")
            side = turns[color] if mapping is None else mapping[turns[color]]
            events.append((r, c, d, color, len(path)))
            d = (d + side) % 4
        else:
            r, c = rr, cc
            path.append((r, c, d))
    raise AssertionError("unterminated carrier")


def _predict(q, mapping=None, mode='normal'):
    path, events = _trace(q, mapping, mode == 'stop')
    field = q['field']
    h, w = field.shape
    mask = np.zeros((h, w), dtype=bool)
    last_corner, ei = 0, 0
    for t, (r, c, d) in enumerate(path, 1):
        while ei < len(events) and events[ei][4] < t:
            last_corner = events[ei][4]
            ei += 1
        mask[r, c] = True
        clock = t - last_corner if mode == 'reset' else t
        if clock % q['spacing'] or mode == 'spine':
            continue
        n = q['first'] + (clock // q['spacing'] - 1) * q['delta']
        if mode == 'freeze':
            n = q['first'] + (q['visible_count'] - 1) * q['delta']
        _need(n > 0, "growth must stay positive until exit")
        axis = q['heading'] if mode == 'screen' else d
        tr, tc = DIRS[(axis + q['side']) % 4]
        # Bound work even for long but legal virtual teeth: only field cells matter.
        js = np.arange(1, min(n, max(h, w)) + 1)
        rr, cc = r + js * tr, c + js * tc
        keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        mask[rr[keep], cc[keep]] = True
    result = field.copy()
    result[mask & ((field == q['background']) | (field == q['ink']))] = q['ink']
    a, b, c, d = q['box']
    return result[a:b + 1, c:d + 1].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct solely from visible framing, prefix geometry, blocks and Ls."""
    return _predict(_decode(input_grid))


def _tip_witnesses(q, output):
    """Measure fully exposed final-leg tooth tips from returned grids only."""
    path, events = _trace(q)
    a, b, e, f = q['box']
    witnesses = []
    for t, (r, c, d) in enumerate(path, 1):
        if t <= events[-1][4] or t % q['spacing']:
            continue
        n = q['first'] + (t // q['spacing'] - 1) * q['delta']
        dr, dc = DIRS[(d + q['side']) % 4]
        x, y = r + dr * n, c + dc * n
        xx, yy = x + dr, y + dc
        if a <= x <= b and e <= y <= f and a <= xx <= b and e <= yy <= f:
            if output[x - a, y - e] == q['ink'] and output[xx - a, yy - e] == q['background']:
                witnesses.append((t // q['spacing'], n, x, y))
    return witnesses


def _edge_distance(r, c, d, shape):
    h, w = shape
    return (r, w - c - 1, h - r - 1, c)[d]


def _plan_mask(shape, segments, spacing, first, delta, side):
    """Generation labels: explicit segment strokes and global station numbers."""
    mask = np.zeros(shape, dtype=bool)
    total = 0
    h, w = shape
    for r, c, d, distance in segments:
        dr, dc = DIRS[d]
        nr, nc = r + dr * distance, c + dc * distance
        mask[min(r, nr):max(r, nr) + 1, min(c, nc):max(c, nc) + 1] = True
        for t in range((total // spacing + 1) * spacing, total + distance + 1, spacing):
            rr, cc = r + dr * (t - total), c + dc * (t - total)
            n = first + (t // spacing - 1) * delta
            _need(n > 0, "nonpositive planned tooth")
            tr, tc = DIRS[(d + side) % 4]
            er, ec = rr + tr * n, cc + tc * n
            mask[max(0, min(rr, er)):min(h, max(rr, er) + 1),
                 max(0, min(cc, ec)):min(w, max(cc, ec) + 1)] = True
        total += distance
    return mask


def _header(rng, width, height, turns):
    widths = np.full(len(turns), 5, dtype=int)
    widths += rng.multinomial(width - int(widths.sum()), np.full(len(turns), 1 / len(turns)))
    a = np.zeros((height, width), dtype=np.int8)
    a[[0, -1]] = 4
    c = 0
    for color, n in zip(rng.permutation(list(turns)), widths):
        n = int(n)
        a[:, c] = a[:, c + n - 1] = 4
        a[1:-1, c + 1 if turns[int(color)] == -1 else c + n - 2] = color
        c += n
    return a


def _scene(rng, profile, split, index, lesson):
    hh = int(rng.integers(3, 6))
    h, w = int(rng.integers(19, 31 - hh)), int(rng.integers(20, 31))
    spacing = int(rng.integers(2, 5))
    visible = int(rng.integers(3, 5))
    first = int(rng.integers(1, 4))
    delta = int(rng.choice([0, 1, 1, 2])) if lesson is None else 1
    side = int(rng.choice([-1, 1]))
    prefix_length = visible * spacing
    max_tooth = first + (visible - 1) * delta
    if profile == 1:
        heading = int(rng.choice([1, 3] if split == 'train' else [0, 2]))
    else:
        heading = int(rng.integers(4))
    dr, dc = DIRS[heading]
    tr, tc = DIRS[(heading + side) % 4]
    starts = [(r, c) for r in range(h) for c in range(w)
              if _edge_distance(r, c, heading, (h, w)) >= prefix_length + 4
              and 0 <= r + tr * max_tooth < h and 0 <= c + tc * max_tooth < w]
    if not starts:
        return None
    sr, sc = starts[int(rng.integers(len(starts)))]
    field = np.zeros((h, w), dtype=np.int8)
    prefix = _plan_mask((h, w), [(sr, sc, heading, prefix_length)], spacing, first, delta, side)
    field[prefix] = 2
    field[sr, sc] = 1
    count = int(rng.integers(2, 5))
    colors = list(range(5, 5 + count))
    turns = {colors[0]: -1, colors[1]: 1}
    turns.update({c: int(rng.choice([-1, 1])) for c in colors[2:]})
    nturns = (int(rng.integers(3, 5)) if split == 'test' else 2) if profile == 0 else int(rng.integers(2, 4))
    if lesson is not None:
        nturns = 2
    q = dict(field=field, start=(sr, sc), heading=heading, turns=turns)
    segments, expected = [], []
    r, c, d = sr, sc, heading
    for k in range(nturns):
        limit = _edge_distance(r, c, d, (h, w))
        low = prefix_length + 1 if k == 0 else 2
        if limit <= low:
            return None
        accepted = False
        for _ in range(28):
            distance = int(rng.integers(low, limit))
            rr, cc = r + DIRS[d][0] * distance, c + DIRS[d][1] * distance
            turn = lesson if lesson is not None else int(rng.choice([-1, 1]))
            color = int(rng.choice([v for v in colors if turns[v] == turn]))
            depth = int(rng.integers(1, 4))
            left, right = int(rng.integers(4)), int(rng.integers(4))
            if d == 0:
                box = max(0, rr - depth), rr, max(0, cc - left), min(w, cc + right + 1)
            elif d == 2:
                box = rr + 1, min(h, rr + depth + 1), max(0, cc - left), min(w, cc + right + 1)
            elif d == 1:
                box = max(0, rr - left), min(h, rr + right + 1), cc + 1, min(w, cc + depth + 1)
            else:
                box = max(0, rr - left), min(h, rr + right + 1), max(0, cc - depth), cc
            a, b, e, f = box
            if a >= b or e >= f or np.any(field[a:b, e:f] != 0):
                continue
            if np.any(field[max(0, a - 1):min(h, b + 1), max(0, e - 1):min(w, f + 1)] == color):
                continue
            trial = field.copy()
            trial[a:b, e:f] = color
            q['field'] = trial
            try:
                path, events = _trace(q)
            except AssertionError:
                continue
            event = (rr, cc, d, color, sum(s[3] for s in segments) + distance)
            if events != expected + [event]:
                continue
            if k == nturns - 1 and _edge_distance(rr, cc, (d + turn) % 4, (h, w)) < 2 * spacing + 1:
                continue
            segments.append((r, c, d, distance))
            expected.append(event)
            field = trial
            r, c, d = rr, cc, (d + turn) % 4
            accepted = True
            break
        if not accepted:
            return None
    segments.append((r, c, d, _edge_distance(r, c, d, (h, w))))
    # A few unused physical blocks vary occupancy independently of route length.
    for _ in range(int(rng.integers(3))):
        a, e = int(rng.integers(h)), int(rng.integers(w))
        b, f = min(h, a + int(rng.integers(1, 4))), min(w, e + int(rng.integers(1, 4)))
        color = int(rng.choice(colors))
        if np.any(field[a:b, e:f] != 0) or np.any(field[max(0, a - 1):min(h, b + 1), max(0, e - 1):min(w, f + 1)] == color):
            continue
        trial = field.copy()
        trial[a:b, e:f] = color
        q['field'] = trial
        try:
            _, events = _trace(q)
        except AssertionError:
            continue
        if events == expected:
            field = trial
    q['field'] = field
    planned = _plan_mask((h, w), segments, spacing, first, delta, side)
    # Show actual tips of at least two late teeth, rather than a featureless
    # slice through a large projected bar. Each tip has a visible blank cell
    # immediately beyond it, so cropping cannot manufacture apparent lengths.
    tips = []
    total = sum(s[3] for s in segments[:-1])
    rr, cc, dd, length = segments[-1]
    for t in range(1, length + 1):
        x, y = rr + DIRS[dd][0] * t, cc + DIRS[dd][1] * t
        if (total + t) % spacing == 0:
            n = first + ((total + t) // spacing - 1) * delta
            tr, tc = DIRS[(dd + side) % 4]
            x2, y2 = x + tr * n, y + tc * n
            x3, y3 = x2 + tr, y2 + tc
            if (0 <= x2 < h and 0 <= y2 < w and 0 <= x3 < h and 0 <= y3 < w
                    and planned[x2, y2] and not planned[x3, y3]
                    and field[x2, y2] == field[x3, y3] == 0):
                tips.append((x2, y2, x3, y3))
    if len(tips) < 2:
        return None
    header = _header(rng, w, hh, turns)
    palette = rng.permutation(10)
    for _ in range(100):
        x, y, _, _ = tips[int(rng.integers(len(tips)))]
        bh, bw = int(rng.integers(3, min(17, h) + 1)), int(rng.integers(3, min(17, w) + 1))
        a = int(np.clip(x - int(rng.integers(bh)), 0, h - bh))
        e = int(np.clip(y - int(rng.integers(bw)), 0, w - bw))
        b, f = a + bh, e + bw
        if np.any(field[a:b, e:f] != 0):
            continue
        if sum(a <= x < b and e <= y < f and a <= xx < b and e <= yy < f
               for x, y, xx, yy in tips) < 2:
            continue
        mask = planned[a:b, e:f]
        if not 0.08 <= float(mask.mean()) <= 0.87 or np.count_nonzero(mask) < 3:
            continue
        inp = field.copy()
        box = a, b - 1, e, f - 1
        for cell in _corners(box):
            inp[cell] = 3
        inp = palette[np.vstack((header, inp))].astype(np.int8)
        output = np.where(mask, palette[2], palette[0]).astype(np.int8)
        try:
            decoded = _decode(inp)
            _need(np.array_equal(_predict(decoded), output), "planned and input predictions disagree")
            # Select legible evidence for both inherited mechanisms in every pair.
            _need(not np.array_equal(_predict(decoded, mode='stop'), output), "turning invisible in crop")
            _need(not np.array_equal(_predict(decoded, mode='reset'), output), "clock continuity invisible")
            if delta:
                _need(not np.array_equal(_predict(decoded, mode='freeze'), output), "growth invisible")
            if lesson is not None:
                _need(not np.array_equal(_predict(decoded, mode='screen'), output), "local side invisible")
        except AssertionError:
            continue
        return dict(input=inp, output=output)
    return None


def _signature(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k], dtype=np.int8).tobytes())
                 for split in ('train', 'test') for p in puzzle[split] for k in ('input', 'output'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name + ' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    profile = int(rng.integers(2))
    first_lesson = int(rng.choice([-1, 1]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            lesson = (first_lesson if i == 0 else -first_lesson) if split == 'train' and i < 2 else None
            for _ in range(160):
                pair = _scene(rng, profile, split, i, lesson)
                if pair is None:
                    continue
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen, train_infos, test_infos = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = inp.shape, inp.astype(np.int8).tobytes()
            _need(key not in seen, "duplicate input")
            seen.add(key)
            q = _decode(inp)
            path, events = _trace(q)
            _need(events and events[0][4] > q['visible_length'], "prefix before all turns")
            _need(len(events) >= 2, "continued routing must be present")
            _need(np.array_equal(_predict(q), out), "incorrect output")
            _need(q['marker'] not in out, "markers must be replaced")
            _need(set(map(int, np.unique(out))) == {q['background'], q['ink']}, "nontrivial construction crop")
            _need(len(_tip_witnesses(q, out)) >= 2, "two fully visible late tooth tips")
            _need(not np.array_equal(_predict(q, mode='stop'), out), "B ablation must change answer")
            _need(not np.array_equal(_predict(q, mode='reset'), out), "corner reset shortcut")
            if q['delta']:
                _need(not np.array_equal(_predict(q, mode='freeze'), out), "A ablation must change answer")
            if split == 'train':
                train_infos.append(q)
            else:
                test_infos.append(q)
    extra_corner = (all(len(_trace(q)[1]) == 2 for q in train_infos)
                    and all(len(_trace(q)[1]) >= 3 for q in test_infos))
    axis_change = (all(q['heading'] % 2 == 1 for q in train_infos)
                   and all(q['heading'] % 2 == 0 for q in test_infos))
    _need(extra_corner or axis_change, "a visible structural train/test shift is required")
    # All four interpretations of the two panel positions are compared using
    # demonstrations only. This checks evidence, not an independent solver.
    fits = []
    for left in (-1, 1):
        for right in (-1, 1):
            mapping = {-1: left, 1: right}
            try:
                good = all(np.array_equal(_predict(q, mapping), p['output'])
                           for q, p in zip(train_infos, puzzle['train']))
            except AssertionError:
                good = False
            if good:
                fits.append((left, right))
    _need(fits == [(-1, 1)], "training must distinguish both relative turn meanings")
    for side in (-1, 1):
        headings = {e[2] for q in train_infos for e in _trace(q)[1] if q['turns'][e[3]] == side}
        _need(len(headings) >= 2, "each turn side needs multiple incoming headings")
    for mode in ('freeze', 'spine', 'reset', 'screen'):
        _need(any(not np.array_equal(_predict(q, mode=mode), p['output'])
                  for q, p in zip(train_infos, puzzle['train'])), "undistinguished shortcut: " + mode)


def self_test():
    """42 streamed seeds; retain summaries/hashes, never all generated puzzles."""
    shapes, structures, backgrounds, headings, spacings, growth, ribs, turns = (set() for _ in range(8))
    puzzle_hashes = set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_hashes.add(hashlib.sha256(repr(_signature(puzzle)).encode()).digest())
        for split in ('train', 'test'):
            for p in puzzle[split]:
                q = _decode(p['input'])
                _, events = _trace(q)
                shapes.add((p['input'].shape, p['output'].shape))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in p['input'].flat)
                structures.add(hashlib.sha256(normalized).digest())
                backgrounds.add(q['background'])
                headings.add(q['heading'])
                spacings.add(q['spacing'])
                growth.add(q['delta'])
                ribs.add(q['visible_count'])
                turns.add(len(events))
                total += 1
    _need(len(puzzle_hashes) == 42 and len(structures) >= 180, "structural variation")
    _need(len(shapes) >= 100 and len(backgrounds) == 10 and len(headings) == 4, "geometry/palette variation")
    _need(spacings == {2, 3, 4} and growth == {0, 1, 2} and ribs == {3, 4}, "comb variation")
    _need({2, 3, 4} <= turns, "route variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1700 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
    _need(_signature(generate_puzzle(77, 2, 1)) == _signature(generate_puzzle(77, 2, 1)), "determinism")
    # Prediction is input-only and must not alter even a read-only input array.
    immutable = generate_puzzle(55, 2, 1)['test'][0]
    immutable['input'].flags.writeable = False
    _need(np.array_equal(solve(immutable['input']), immutable['output']), "pure input-only solve")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(8, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    # Inconsistent arithmetic evidence and a genuine carrier cycle are rejected.
    p = generate_puzzle(19, 2, 1)
    original = next(pair for pair in p['train'] if _decode(pair['input'])['delta'] > 0)
    q = _decode(original['input'])
    damaged = original['input'].copy()
    dr, dc = DIRS[q['heading']]
    tr, tc = DIRS[(q['heading'] + q['side']) % 4]
    n = q['first'] + (q['visible_count'] - 1) * q['delta']
    r = q['start'][0] + dr * q['visible_length'] + tr * n + q['header']
    c = q['start'][1] + dc * q['visible_length'] + tc * n
    damaged[r, c] = q['background']
    try:
        _decode(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('inconsistent arithmetic prefix accepted')
    field = np.zeros((10, 10), dtype=np.int8)
    for r, c in ((4, 7), (7, 6), (6, 3), (3, 4)):
        field[r, c] = 5
    try:
        _trace(dict(field=field, start=(4, 4), heading=1, turns={5: 1}))
    except AssertionError:
        pass
    else:
        raise AssertionError('cyclic carrier accepted')
    return dict(seeds=42, pairs=total, structures=len(structures), shapes=len(shapes),
                headings=sorted(headings), spacings=sorted(spacings), growth=sorted(growth),
                visible_teeth=sorted(ribs), turn_counts=sorted(turns))
