"""Collect, turn, and clear: an input-decoded, stateful ARC beam family.

Only NumPy and the standard library are used. No parent code/data is imported.
"""
from __future__ import annotations

import itertools
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = [start]
        for r, c in cells:
            for dr, dc in DIRS:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    cells.append(point)
        r, c = np.min(cells, axis=0)
        b, f = np.max(cells, axis=0) + 1
        _require(len(cells) == (b-r)*(f-c), "nonrectangular component")
        result.append((int(r), int(b), int(c), int(f)))
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    frame = int(a[0, 0])
    _require(np.all(a[0] == frame), "missing header top")
    bottoms = np.flatnonzero(np.all(a[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing header bottom")
    bottom = int(bottoms[0])
    _require(bottom >= 4 and bottom < len(a)-1, "invalid header height")
    band = a[1:bottom]
    walls = np.all(band == frame, axis=0)
    _require(walls[0] and walls[-1], "missing header side walls")
    runs, c = [], 0
    while c < a.shape[1]:
        if walls[c]:
            c += 1
            continue
        first = c
        while c < a.shape[1] and not walls[c]:
            c += 1
        runs.append((first, c))
    _require(len(runs) == 3 and all(f-e >= 3 for e, f in runs), "three legible panels required")
    queries, raw_turns = [], []
    for e, f in runs:
        panel = band[:, e:f]
        _require(not np.any(panel == frame), "frame inside panel")
        colors, counts = np.unique(panel, return_counts=True)
        bg = int(colors[np.argmax(counts)])
        other = set(map(int, colors)) - {bg}
        if len(other) == 3 and all(np.count_nonzero(panel == x) == 1 for x in other):
            queries.append((bg, frozenset(other)))
        elif len(other) == 1:
            color = next(iter(other))
            left = np.all(panel[:, 0] == color) and np.all(panel[:, 1:] == bg)
            right = np.all(panel[:, -1] == color) and np.all(panel[:, :-1] == bg)
            _require(bool(left) != bool(right), "ambiguous turn bar")
            raw_turns.append((bg, color, -1 if left else 1))
        else:
            raise AssertionError("unrecognized instruction panel")
    _require(len(queries) == 1 and len(raw_turns) == 2, "ambiguous query/key")
    background, query = queries[0]
    _require(all(bg == background for bg, _, _ in raw_turns), "panel padding differs")
    turns = {color: side for _, color, side in raw_turns}
    _require(len(turns) == 2 and set(turns.values()) == {-1, 1}, "one key for each turn required")
    _require(not (query & set(turns)) and frame not in query | set(turns) | {background}, "role collision")
    field = a[bottom+1:].copy()
    rectangles, emitter = [], []
    for box in _components(field != background):
        r, b, e, f = box
        block = field[r:b, e:f]
        colors = set(map(int, np.unique(block)))
        if frame in colors:
            rest = colors - {frame}
            _require(len(rest) == 1, "ambiguous emitter")
            source = next(iter(rest))
            _require(source not in query | set(turns) | {background}, "invalid emitter color")
            points = np.argwhere(field == source)
            sr, sc = map(int, points.min(axis=0))
            er, ec = map(int, points.max(axis=0) + 1)
            k = er-sr
            _require(k == ec-sc and len(points) == k*k and k > 0, "source must be one solid square")
            marker = set(map(tuple, np.argwhere(field == frame)))
            faces = [set((sr-1, sc+i) for i in range(k)),
                     set((sr+i, sc+k) for i in range(k)),
                     set((sr+k, sc+i) for i in range(k)),
                     set((sr+i, sc-1) for i in range(k))]
            matches = [d for d, face in enumerate(faces) if marker == face]
            _require(len(matches) == 1, "nozzle must mark exactly one source face")
            _require(block.size == k*k+k, "extra emitter cells")
            emitter.append((source, sr, sc, k, matches[0]))
        else:
            _require(min(block.shape) >= 3, "records need an interior")
            body = int(block[0, 0])
            _require(body in turns and colors & set(turns) == {body}, "record must have one keyed body")
            _require(np.all(block[[0, -1]] == body) and np.all(block[:, [0, -1]] == body), "record border broken")
            _require(background not in colors, "record has a hole")
            labels = frozenset(colors - {body})
            counts = {int(x): int(np.count_nonzero(block == x)) for x in labels}
            rectangles.append(dict(box=box, body=body, labels=labels, counts=counts))
    _require(len(emitter) == 1, "one directed emitter required")
    source, sr, sc, k, direction = emitter[0]
    _require(all(source not in rec['labels'] for rec in rectangles), "source color used as a label")
    # A complete empty row/column gap separates all component bounding boxes.
    boxes = _components(field != background)
    for x, y in itertools.combinations(boxes, 2):
        _require(x[1] < y[0] or y[1] < x[0] or x[3] < y[2] or y[3] < x[2], "objects need a background gap")
    return dict(field=field, background=background, frame=frame, source=source,
                query=query, turns=turns, rectangles=rectangles, origin=(sr, sc),
                k=k, direction=direction, crop=bottom+1, panels=tuple(runs))


def _trace(info, *, predicate="union", reset=True, turn_override=None, query_override=None):
    """Slide the beam square; decisions use whole input records, never paint.

    Optional hypotheses are used only by counterexample checks, not by solve.
    """
    field = info['field']
    h, w = field.shape
    k, (r, c), d = info['k'], info['origin'], info['direction']
    query = info['query'] if query_override is None else frozenset(query_override)
    ids = np.full(field.shape, -1, dtype=int)
    for i, rec in enumerate(info['rectangles']):
        a, b, e, f = rec['box']
        ids[a:b, e:f] = i
    inventory, erased, seen = frozenset(), set(), set()
    paint = np.zeros(field.shape, dtype=bool)
    events, segments = [], []
    start_r, start_c, distance = r, c, 0
    after_turn = False
    while True:
        state = (r, c, d, inventory, frozenset(erased))
        _require(state not in seen, "nonterminating beam cycle")
        seen.add(state)
        if (d == 0 and r == 0) or (d == 1 and c+k == w) or (d == 2 and r+k == h) or (d == 3 and c == 0):
            segments.append((start_r, start_c, d, distance, -1))
            break
        if d == 0:
            rows, cols = np.full(k, r-1), np.arange(c, c+k)
        elif d == 1:
            rows, cols = np.arange(r, r+k), np.full(k, c+k)
        elif d == 2:
            rows, cols = np.full(k, r+k), np.arange(c, c+k)
        else:
            rows, cols = np.arange(r, r+k), np.full(k, c-1)
        face = ids[rows, cols].copy()
        for gone in erased:
            face[face == gone] = -1
        if np.any(face >= 0):
            hit = int(face[0])
            _require(hit >= 0 and np.all(face == hit), "partial or simultaneous record impact")
            _require(not after_turn or distance > 0, "no clear travel after turning")
            rec = info['rectangles'][hit]
            before = inventory
            inventory = inventory | (rec['labels'] & query)
            complete = query <= (rec['labels'] if predicate == 'local' else inventory)
            if predicate == 'always':
                complete = True
            if predicate == 'never':
                complete = False
            old_d = d
            if complete:
                erased.add(hit)
                if reset:
                    inventory = frozenset()
                after_turn = False
            else:
                side = info['turns'][rec['body']]
                if turn_override == 'reverse':
                    side = -side
                elif turn_override in (-1, 1):
                    side = turn_override
                d = (d + side) % 4
                after_turn = True
            events.append(dict(hit=hit, complete=bool(complete), before=before,
                               collected=rec['labels'] & query, after=inventory,
                               incoming=old_d, outgoing=d, at=(r, c)))
            segments.append((start_r, start_c, old_d, distance, hit))
            start_r, start_c, distance = r, c, 0
            continue
        paint[rows, cols] = True
        dr, dc = DIRS[d]
        r, c = r+dr, c+dc
        distance += 1
    output = field.copy()
    for i in erased:
        a, b, e, f = info['rectangles'][i]['box']
        output[a:b, e:f] = info['background']
    paint &= (output == info['background']) | (output == info['source'])
    output[paint] = info['source']
    return output, events, segments, paint, erased


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible query, turn bars, nozzle and records; trace one beam."""
    return _trace(_decode(input_grid))[0]


def _counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _header(rng, width, query, height=5):
    widths = np.full(3, 5, dtype=int)
    widths += rng.multinomial(width-15, [1/3]*3)
    panels = list(rng.permutation(3))
    a = np.zeros((height, width), dtype=np.int8)
    a[[0, -1]] = 1
    c = 0
    for role, size in zip(panels, widths):
        a[:, c] = a[:, c+size-1] = 1
        if role == 0:
            interior = (height-2)*(size-2)
            positions = rng.choice(interior, 3, replace=False)
            for pos, color in zip(positions, rng.permutation(list(query))):
                a[1+int(pos)//(size-2), c+1+int(pos)%(size-2)] = color
        else:
            a[1:-1, c+1 if role == 1 else c+size-2] = 2+role
        c += int(size)
    return a


def _clear(field, box):
    a, b, e, f = box
    h, w = field.shape
    return (0 <= a < b <= h and 0 <= e < f <= w and
            not np.any(field[max(0, a-1):min(h, b+1), max(0, e-1):min(w, f+1)] != 0))


def _paint_record(field, box, body, marks, rng):
    a, b, e, f = box
    trial = field.copy()
    trial[a:b, e:f] = body
    hh, ww = b-a-2, f-e-2
    for pos, color in zip(rng.choice(hh*ww, len(marks), replace=False), rng.permutation(marks)):
        trial[a+1+int(pos)//ww, e+1+int(pos)%ww] = color
    return trial


def _canonical_info(field, query, origin, k, direction, records):
    return dict(field=field, background=0, frame=1, source=2, query=frozenset(query),
                turns={3: -1, 4: 1}, rectangles=records, origin=origin, k=k, direction=direction)


def _append(rng, field, records, query, origin, k, direction, marks, side, current, compact=False):
    """Insert on the actual final ray segment, retaining its complete event prefix."""
    r, c, d, length, _ = current[2][-1]
    if length < 4:
        return None
    body = 3 if side == -1 else 4
    compact_choices = [(gap, before) for gap in range(1, length-1)
                       for before in sorted(range(6-k), key=lambda x: abs(x-(5-k)/2))]
    for attempt in range(len(compact_choices) if compact else 95):
        # Small early gaps leave room for later records; some attempts use the
        # entire available segment and independent long transverse extensions.
        if compact:
            distance = compact_choices[attempt][0]
        elif attempt < 70:
            distance = int(rng.integers(1, min(length-2, 7)+1))
        else:
            distance = int(rng.integers(1, length-1))
        dr, dc = DIRS[d]
        x, y = r+dr*distance, c+dc*distance
        depth = 4 if compact else int(rng.integers(3, 7))
        before = compact_choices[attempt][1] if compact else int(rng.integers(0, 4))
        after = 5-k-before if compact else int(rng.integers(0, 4))
        transverse = max(3, k+before+after)
        after += transverse-(k+before+after)
        if not compact and attempt > 75 and rng.random() < .25:
            before += int(rng.integers(4))
            after += int(rng.integers(4))
        if (depth-2)*(transverse-2) < len(marks):
            continue
        if d == 0:
            box = (x-depth, x, y-before, y+k+after)
        elif d == 1:
            box = (x-before, x+k+after, y+k, y+k+depth)
        elif d == 2:
            box = (x+k, x+k+depth, y-before, y+k+after)
        else:
            box = (x-before, x+k+after, y-depth, y)
        if not _clear(field, box):
            continue
        trial = _paint_record(field, box, body, marks, rng)
        rec = dict(box=box, body=body, labels=frozenset(marks),
                   counts={int(x): marks.count(x) for x in set(marks)})
        try:
            traced = _trace(_canonical_info(trial, query, origin, k, direction, records+[rec]))
        except AssertionError:
            continue
        # Retain encounter order and every inventory/turn decision already made.
        if len(traced[1]) == len(current[1])+1 and traced[1][:-1] == current[1] and traced[1][-1]['hit'] == len(records):
            return trial, records+[rec], traced
    return None


def _plan(rng, query, package):
    q = list(rng.permutation(list(query)))
    extras = list(rng.permutation([x for x in range(5, 10) if x not in query]))
    x, y = extras
    if package == 'identity':
        return [q[:2], [q[2]], [q[1], q[2], x, q[1], x],
                [q[0], y, q[0], y], [q[0], q[2], x], [q[1], x, y]]
    if package == 'memory':
        return [q, [q[0]], [q[1], x, y, q[1], x],
                [q[2], x, q[2], x], [q[2]], [q[0], q[1], y]]
    if package == 'test':
        result = []
        if rng.random() < .4:
            result.append(q + [x])
        for _ in range(2 if result else int(rng.integers(2, 4))):
            subset = list(rng.permutation(q)[:int(rng.integers(1, 3))])
            result.append(subset + ([x, subset[0], x] if rng.random() < .5 else []))
            result.append([z for z in q if z not in subset] + ([y] if rng.random() < .5 else []))
        return result
    # Freely sampled event content, including empty and nonprogress records.
    result = []
    for _ in range(int(rng.integers(1, 6))):
        subset = list(rng.permutation(q)[:int(rng.integers(4))])
        extra = extras[:int(rng.integers(3))]
        marks = subset + extra
        if marks and rng.random() < .5:
            marks += [int(rng.choice(marks))]
        result.append(marks)
    return result


def _make_pair(rng, package, k, direction, first_side, palette, selected_query=None):
    query = frozenset(map(int, rng.choice(range(5, 10), 3, replace=False))) if selected_query is None else frozenset(selected_query)
    plan = _plan(rng, query, package)
    teaching = package in ('identity', 'memory')
    for restart in range(100):
        compact = restart >= 80
        hh = 5 if compact else int(rng.integers(5, 7))
        low = 21 if len(plan) >= 5 else 15 if package == 'test' else max(12, 3*k+6)
        h, w = (25, 30) if compact else (int(rng.integers(low, 31-hh)), int(rng.integers(max(15, low), 31)))
        # Alternating relative turns give a spacious two-heading lesson. Other
        # examples first attempt unrestricted heading sequences and origins.
        staircase = teaching or restart >= 25 or rng.random() < .4
        d = direction if teaching else int(rng.integers(4))
        side0 = first_side if teaching else int(rng.choice([-1, 1]))
        if staircase:
            dr1, dc1 = DIRS[d]
            dr2, dc2 = DIRS[(d+side0) % 4]
            dr, dc = dr1+dr2, dc1+dc2
            rgap, cgap = (1, 1) if compact else (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
            sr = rgap if dr > 0 else h-k-rgap
            sc = cgap if dc > 0 else w-k-cgap
        else:
            sr, sc = int(rng.integers(1, h-k)), int(rng.integers(1, w-k))
        field = np.zeros((h, w), dtype=np.int8)
        field[sr:sr+k, sc:sc+k] = 2
        if d == 0: field[sr-1, sc:sc+k] = 1
        elif d == 1: field[sr:sr+k, sc+k] = 1
        elif d == 2: field[sr+k, sc:sc+k] = 1
        else: field[sr:sr+k, sc-1] = 1
        records, turn_count = [], 0
        current = _trace(_canonical_info(field, query, (sr, sc), k, d, records))
        for marks in plan:
            side = side0*(-1)**turn_count if staircase else int(rng.choice([-1, 1]))
            added = _append(rng, field, records, query, (sr, sc), k, d, marks, side, current, compact)
            if added is None:
                break
            field, records, current = added
            turn_count += not current[1][-1]['complete']
        else:
            # Off-route records are never globally filtered. Teaching includes
            # an exact-query decoy; free examples may include empty records.
            target = 1 if teaching else int(rng.integers(3))
            for index in range(target):
                marks = list(query) if teaching else list(rng.permutation(range(5, 10))[:int(rng.integers(6))])
                for _ in range(130):
                    rh, rw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
                    if (rh-2)*(rw-2) < len(marks): continue
                    a, e = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
                    box = (a, a+rh, e, e+rw)
                    if not _clear(field, box): continue
                    body = int(rng.choice([3, 4]))
                    trial = _paint_record(field, box, body, marks, rng)
                    rec = dict(box=box, body=body, labels=frozenset(marks), counts={x: marks.count(x) for x in set(marks)})
                    try:
                        traced = _trace(_canonical_info(trial, query, (sr, sc), k, d, records+[rec]))
                    except AssertionError:
                        continue
                    if traced[1] == current[1]:
                        field, records, current = trial, records+[rec], traced
                        break
                else:
                    if teaching: break
            if teaching and len(records) != len(plan)+1:
                continue
            header = _header(rng, w, query, hh)
            inp = palette[np.vstack((header, field))].astype(np.int8)
            out = palette[current[0]].astype(np.int8)
            return dict(input=inp, output=out)
    raise AssertionError(f'bounded scene construction exhausted: {package}, k={k}, events={len(plan)}')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['larger', 'smaller', 'recombine']))
    sizes = [2, 3] if shift == 'smaller' else [1, 2]
    direction, side = int(rng.integers(4)), int(rng.choice([-1, 1]))
    stable_palette = rng.permutation(10)
    vary_palette = bool(rng.integers(2)) and shift != 'recombine'
    triples = list(itertools.combinations(range(5, 10), 3))
    rng.shuffle(triples)
    reserved = triples[-nq:]
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            package = ('identity' if index == 0 else 'memory' if index == 1 else 'free') if split == 'train' else 'test'
            k = sizes[index % 2] if split == 'train' else 3 if shift == 'larger' else 1 if shift == 'smaller' else int(rng.choice(sizes))
            heading = direction if index == 0 else (direction+2) % 4
            first_side = side if index == 0 else -side
            for _ in range(8):
                palette = rng.permutation(10) if vary_palette else stable_palette
                selected = (reserved[index] if split == 'test' else triples[index % (10-nq)]) if shift == 'recombine' else None
                pair = _make_pair(rng, package, k, heading, first_side, palette, selected)
                signature = _key(pair['input'])
                if signature not in seen:
                    break
            else:
                raise AssertionError('unique input sampling exhausted')
            seen.add(signature)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(info, expected, **kwargs):
    try:
        return not np.array_equal(_trace(info, **kwargs)[0], expected)
    except AssertionError:
        return True


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, decoded = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            expected, events, segments, paint, erased = _trace(info)
            _require(np.array_equal(out, expected), 'incorrect stateful beam output')
            _require(out.shape == info['field'].shape, 'incorrect field crop')
            for i, rec in enumerate(info['rectangles']):
                a, b, e, f = rec['box']
                if i not in erased:
                    _require(np.array_equal(out[a:b, e:f], info['field'][a:b, e:f]), 'preserved record changed')
                else:
                    _require(np.all(np.isin(out[a:b, e:f], [info['background'], info['source']])), 'erased record residue')
            _require(np.all(out[info['field'] == info['frame']] == info['frame']), 'nozzle changed')
            info.update(events=events, segments=segments, erased=erased, expected=expected)
            decoded[split].append(info)
    train = decoded['train']
    _require(len({q['k'] for q in train}) >= 2, 'training must show thickness from source')
    arrivals = {s: set() for s in (-1, 1)}
    partial_finish = rich_negative = local_exact = offroute_complete = False
    repeated = {True: False, False: False}
    for q in train:
        hit = {e['hit'] for e in q['events']}
        for i, rec in enumerate(q['rectangles']):
            offroute_complete |= i not in hit and q['query'] <= rec['labels']
        for event in q['events']:
            rec = q['rectangles'][event['hit']]
            repeated[event['complete']] |= (
                any(rec['counts'].get(x, 0) > 1 for x in q['query']) and
                any(rec['counts'][x] > 1 for x in rec['labels'] - q['query']))
            if event['complete']:
                partial_finish |= not q['query'] <= rec['labels']
                local_exact |= q['query'] == rec['labels'] and sum(rec['counts'].values()) == 3
            else:
                arrivals[q['turns'][rec['body']]].add(event['incoming'])
                rich_negative |= sum(rec['counts'].values()) > 3 and bool(rec['labels'] - q['query'])
    _require(partial_finish and rich_negative and local_exact and offroute_complete, 'missing membership/route witnesses')
    _require(all(repeated.values()), 'repetition of query and extra labels must occur in both outcomes')
    _require(all(len(headings) >= 2 for headings in arrivals.values()), 'both relative turns need multiple headings')
    for alternative in ({'predicate': 'local'}, {'reset': False}, {'turn_override': 'reverse'},
                        {'turn_override': -1}, {'turn_override': 1}, {'predicate': 'always'}, {'predicate': 'never'}):
        _require(any(_different(q, q['expected'], **alternative) for q in train), 'undistinguished simpler rule')
    _require(any(all(_different(q, q['expected'], query_override=q['query']-{color}) for color in q['query'])
                 for q in train), 'each selected identity must matter')
    return None


def self_test():
    """Bounded 42-seed structural, semantic, counterfactual and diversity audit."""
    fingerprints, shapes, layouts, ks, directions, lengths, queries = set(), set(), set(), set(), set(), set(), set()
    roles, erased_counts, record_counts, turn_sides = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(pair['input']) for split in puzzle.values() for pair in split))
        for split in puzzle.values():
            for pair in split:
                q = _decode(pair['input'])
                out, events, _, _, erased = _trace(q)
                assert np.array_equal(out, pair['output'])
                shapes.add(q['field'].shape); ks.add(q['k']); directions.add(q['direction'])
                lengths.add(len(events)); erased_counts.add(len(erased)); record_counts.add(len(q['rectangles']))
                queries.add(tuple(sorted(q['query'])))
                roles.add((q['background'], q['frame'], q['source']))
                layouts.add((q['field'].shape, q['origin'], q['direction'], tuple(r['box'] for r in q['rectangles'])))
                turn_sides.update(q['turns'][q['rectangles'][e['hit']]['body']] for e in events if not e['complete'])
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(p[n]) == _key(q[n]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for n in p)
    assert len(fingerprints) == 42 and len(layouts) > 180 and len(shapes) >= 35
    assert ks == {1, 2, 3} and directions == {0, 1, 2, 3} and turn_sides == {-1, 1}
    assert len(lengths) >= 5 and len(record_counts) >= 6 and len(erased_counts) >= 4
    assert len(roles) >= 40 and len(queries) >= 25
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(84, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Spatial query/label arrangement is immaterial. Literal identity is not.
    p = generate_puzzle(0, 2, 1)
    for pair in p['train']:
        q = _decode(pair['input'])
        changed = np.array(pair['input'], copy=True)
        band = changed[1:q['crop']-1]
        positions = [(int(r), int(c)) for r, c in np.argwhere(np.isin(band, list(q['query'])))]
        values = [int(band[point]) for point in positions]
        for point, value in zip(positions, values[1:]+values[:1]):
            band[point] = value
        assert np.array_equal(solve(changed), pair['output'])
        # Move both marks and padding within the query interior, retaining the
        # enclosing frame and the unordered identities.
        query_panel = next((e, f) for e, f in q['panels']
                           if np.any(np.isin(band[:, e:f], list(q['query']))))
        e, f = query_panel
        band[:, e:f] = np.roll(np.roll(band[:, e:f], 1, axis=0), 1, axis=1)
        assert np.array_equal(solve(changed), pair['output'])
        # Rearranging the finishing record, including filler positions, changes
        # neither qualification nor its complete erasure.
        finishing = next(iter(_trace(q)[4]))
        a, b, e, f = q['rectangles'][finishing]['box']
        moved = np.array(pair['input'], copy=True)
        interior = moved[q['crop']+a+1:q['crop']+b-1, e+1:f-1]
        interior[:] = np.roll(interior.ravel(), 1).reshape(interior.shape)
        assert np.array_equal(solve(moved), pair['output'])
    # A manually specified edge completion exercises whole-object erasure,
    # full-width traversal, immutability of the nozzle, and boundary stopping.
    field = np.zeros((9, 15), dtype=np.int8)
    field[4:6, 1:3] = 2
    field[4:6, 3] = 1
    field[2:7, 8:15] = 3
    field[3, 9:12] = [5, 6, 7]
    header = _header(np.random.default_rng(991), 15, {5, 6, 7})
    edge_input = np.vstack((header, field))
    edge_expected = field.copy()
    edge_expected[2:7, 8:15] = 0
    edge_expected[4:6, 4:] = 2
    assert np.array_equal(solve(edge_input), edge_expected)
    # Moving a record to cover only one row of that two-wide face is invalid.
    partial = field.copy()
    partial[2:7, 8:15] = 0
    partial[5:8, 8:12] = 3
    partial[6, 9] = 5
    try:
        solve(np.vstack((header, partial)))
    except AssertionError:
        pass
    else:
        raise AssertionError('partial-width impact accepted')
    # Reject output corruption, including the immutable nozzle.
    bad = {s: [{n: np.array(g, copy=True) for n, g in pair.items()} for pair in pairs] for s, pairs in p.items()}
    q = _decode(bad['train'][0]['input'])
    rr, cc = np.argwhere(q['field'] == q['frame'])[0]
    bad['train'][0]['output'][rr, cc] = q['source']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted nozzle accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), field_shapes=len(shapes),
                layouts=len(layouts), event_counts=sorted(lengths), record_counts=sorted(record_counts),
                deletion_counts=sorted(erased_counts), source_sizes=sorted(ks), palette_roles=len(roles))


if __name__ == '__main__':
    print(self_test())
