"""Hidden patterned rectangular loops act as permeable relative-turn tiles.

Only standard-library modules and NumPy are used. The renderer knows complete
loops; solve independently recovers them from visible pixels. No parent code,
saved examples, seed-dependent answer convention, or external files are used.
"""

import json
from collections import Counter

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= d <= 30 for d in array.shape), "invalid grid shape")
    _require(np.issubdtype(array.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= array) & (array <= 9)), "invalid color")
    return array.astype(np.int8, copy=False)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, c) for y in range(t, b + 1) for c in range(l, r + 1)}


def _bbox(points):
    rows, columns = zip(*points)
    return min(rows), max(rows), min(columns), max(columns)


def _halo(points):
    return points | {(y + dy, x + dx) for y, x in points for dy, dx in _DIRECTIONS}


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            y, x = stack.pop()
            for dy, dx in _DIRECTIONS:
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _pattern(grid, visible, hidden):
    """Infer any word of period 1..4 indexed by absolute row+column.

    Require two visible witnesses for every residue. Multiple fitting words
    are harmless only if they agree on every concealed perimeter cell.
    """
    explanations = []
    ordered = sorted(hidden)
    for period in range(1, 5):
        word, counts = {}, [0] * period
        for y, x in sorted(visible):
            residue, color = (y + x) % period, int(grid[y, x])
            if residue in word and word[residue] != color:
                break
            word[residue] = color
            counts[residue] += 1
        else:
            if min(counts) >= 2:
                values = tuple(word[(y + x) % period] for y, x in ordered)
                explanations.append((period, values))
    if not explanations or len({values for _, values in explanations}) != 1:
        return None
    return min(p for p, _ in explanations), dict(zip(ordered, explanations[0][1]))


def _parse_header(grid):
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing top key border")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom key border")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "invalid key height")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "key bars must be vertical")
    row, runs, c = band[0], [], 0
    _require(row[0] == row[-1] == frame, "missing key side borders")
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((start, c))
    _require(2 <= len(runs) <= 4 and all(b - a >= 3 for a, b in runs), "invalid panels")
    empty = int(row[runs[0][0] + 1])
    turns = {}
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == empty), "inconsistent key interior")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == empty) != (right == empty), "ambiguous key bar")
        color, turn = (right, 1) if left == empty else (left, -1)
        _require(color not in turns and color not in (frame, empty), "duplicate key color")
        turns[color] = turn
    return bottom + 1, frame, empty, turns, tuple(b - a + 2 for a, b in runs)


def _decode(input_grid):
    grid = _grid(input_grid)
    crop, frame, background, turns, widths = _parse_header(grid)
    field = grid[crop:]
    frame_points = {tuple(map(int, p)) for p in np.argwhere(field == frame)}
    _require(bool(frame_points), "no viewport frame")
    main = _bbox(frame_points)
    t, b, l, r = main
    _require(b - t >= 4 and r - l >= 4 and frame_points == _perimeter(main), "invalid viewport")
    _require(t > 0 and l > 0 and b < len(field) - 1 and r < field.shape[1] - 1, "viewport needs exterior")
    mask = _area(main)
    interior = field[t + 1:b, l + 1:r]
    counts = Counter(map(int, interior.flat))
    _require(len(counts) == 2, "viewport must contain fill and one source")
    source_candidates = [c for c, count in counts.items() if count == 1]
    _require(len(source_candidates) == 1, "source is not a unique dot")
    source = source_candidates[0]
    fill = next(c for c in counts if c != source)
    _require(source not in turns and source not in (frame, background)
             and fill not in turns and fill != frame, "role colors overlap")
    source_position = tuple(map(int, np.argwhere(interior == source)[0]))
    _require(all(field[p] == background or int(field[p]) in turns
                 for p in _area((0, len(field) - 1, 0, field.shape[1] - 1)) - mask),
             "unrecognized exterior material")
    _require(all(field[p] == background for p in ((0, 0), (0, field.shape[1] - 1),
                                                 (len(field) - 1, 0), (len(field) - 1, field.shape[1] - 1))),
             "canvas corners must be background")
    visible_points = {tuple(map(int, p)) for p in np.argwhere(np.isin(field, list(turns)))} - mask
    parts = _components(visible_points)
    _require(bool(parts), "no rectangular loop evidence")
    candidates = []
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            visible = first if i == j else first | parts[j]
            box = _bbox(visible)
            a, z, e, f = box
            if z - a < 2 or f - e < 2:
                continue
            full = _perimeter(box)
            if full - mask != visible:
                continue
            inferred = _pattern(field, visible, full & mask)
            if inferred is None:
                continue
            period, assignments = inferred
            word = tuple(next(int(field[p]) for p in sorted(visible)
                              if sum(p) % period == residue) for residue in range(period))
            candidates.append(dict(parts=frozenset((i, j)), box=box, full=full,
                                   assignments=assignments, period=period, visible=visible,
                                   word=word, reference_color=int(field[min(visible)])))
    covers = []

    def cover(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for candidate in candidates:
            if (first in candidate['parts'] and candidate['parts'] <= remaining
                    and not candidate['full'] & _halo(occupied)):
                cover(remaining - candidate['parts'], chosen + [candidate], occupied | candidate['full'])

    cover(set(range(len(parts))), [], set())
    _require(len(covers) == 1, "ambiguous or incomplete loop geometry")
    loops = covers[0]
    controls = np.full(interior.shape, -1, dtype=np.int8)
    owners = np.full(interior.shape, -1, dtype=np.int8)
    for i, loop in enumerate(loops):
        for (y, x), color in loop['assignments'].items():
            if t < y < b and l < x < r:
                controls[y - t - 1, x - l - 1] = color
                owners[y - t - 1, x - l - 1] = i
    _require(controls[source_position] == -1, "source covers a hidden control")
    return dict(controls=controls, owners=owners, source_position=source_position,
                source=source, fill=fill, background=background, frame=frame,
                turns=turns, loops=loops, main=main, header_height=crop, panel_widths=widths)


def _trace(controls, source_position, turns, owners=None, stop_at_control=False,
           first_turn_only=False):
    """Trace the static switch graph; painted cells never change transitions."""
    h, w = controls.shape
    illuminated = np.zeros((h, w), dtype=bool)
    branches, events, cycles, masks = [], [], [], []
    for initial in range(4):
        y, x = source_position
        direction = initial
        seen, hits = set(), []
        mask = np.zeros((h, w), dtype=bool)
        while (y, x, direction) not in seen:
            seen.add((y, x, direction))
            illuminated[y, x] = mask[y, x] = True
            dy, dx = _DIRECTIONS[direction]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                cycles.append(False)
                break
            color = int(controls[ny, nx])
            if color >= 0 and stop_at_control:
                cycles.append(False)
                break
            y, x = ny, nx
            if color >= 0 and (not first_turn_only or not hits):
                owner = -1 if owners is None else int(owners[y, x])
                event = (initial, y, x, color, direction, turns[color], owner)
                hits.append(event)
                events.append(event)
                direction = (direction + turns[color]) % 4
        else:
            cycles.append(True)
        branches.append(hits)
        masks.append(mask)
    return illuminated, dict(branches=branches, events=events, cycles=cycles, masks=masks)


def _answer(info, turns=None, controls=None, **options):
    switches = info['controls'] if controls is None else controls
    table = info['turns'] if turns is None else turns
    lit, tracing = _trace(switches, info['source_position'], table, info['owners'], **options)
    result = np.full(switches.shape, info['fill'], dtype=np.int8)
    result[lit] = info['source']
    return result, tracing


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the viewport light trails from this input's geometry and key."""
    return _answer(_decode(input_grid))[0]


def _header(rng, height, width, frame, background, turns):
    widths = np.full(len(turns), 5, dtype=int)
    widths += rng.multinomial(width - 5 * len(turns), np.full(len(turns), 1 / len(turns)))
    band = np.full((height, width), background, dtype=np.int8)
    band[[0, -1], :] = frame
    c = 0
    for color, size in zip(rng.permutation(list(turns)), widths):
        band[:, c] = band[:, c + size - 1] = frame
        band[1:-1, c + 1 if turns[int(color)] == -1 else c + size - 2] = color
        c += size
    return band


def _geometry(rng, count, exterior, header_height, width):
    height = int(rng.integers(21, 31 - header_height))
    oh = int(rng.integers(7, min(16, height - 8) + 1))
    ow = int(rng.integers(7, min(18, width - 8) + 1))
    t = int(rng.integers(3, height - oh - 3))
    l = int(rng.integers(3, width - ow - 3))
    main = (t, t + oh + 1, l, l + ow + 1)
    mask, inside = _area(main), _area((t + 1, t + oh, l + 1, l + ow))
    boxes, occupied = [], set()
    for crossing in [True] * count + [False] * exterior:
        for _ in range(650):
            a, z = sorted(map(int, rng.choice(np.arange(1, height - 1), 2, replace=False)))
            e, f = sorted(map(int, rng.choice(np.arange(1, width - 1), 2, replace=False)))
            if z - a < 2 or f - e < 2:
                continue
            box = (a, z, e, f)
            full, visible = _perimeter(box), _perimeter(box) - mask
            if not visible or _bbox(visible) != box or full & _halo(occupied):
                continue
            if crossing and len(full & inside) < 3:
                continue
            if not crossing and full & mask:
                continue
            boxes.append(box)
            occupied.update(full)
            break
        else:
            return None
    return height, main, boxes


def _primitive_word(rng, period, colors):
    if period == 1:
        return (int(rng.choice(colors)),)
    for _ in range(30):
        word = [int(colors[0]), int(colors[1])]
        word += [int(rng.choice(colors)) for _ in range(period - 2)]
        rng.shuffle(word)
        if not any(period % p == 0 and all(word[i] == word[i % p] for i in range(period))
                   for p in range(1, period)):
            return tuple(word)
    # For period 4, AAAB is always primitive and exercises opposite turn sides.
    return tuple([int(colors[0])] * (period - 1) + [int(colors[1])])


def _render_candidate(rng, plan, parameters):
    key_count = int(rng.integers(2, 5))
    if plan['lesson'] in (0, 1):
        key_colors = list(parameters['teaching_keys'])
        extras = [c for c in range(10) if c not in key_colors]
        key_colors += list(map(int, rng.choice(extras, key_count - 2, replace=False)))
    else:
        key_colors = list(map(int, rng.choice(10, key_count, replace=False)))
    flip = plan['lesson'] == 1 or (plan['lesson'] > 1 and bool(rng.integers(2)))
    turns = {key_colors[0]: 1 if flip else -1, key_colors[1]: -1 if flip else 1}
    turns.update({c: int(rng.choice((-1, 1))) for c in key_colors[2:]})
    remaining = list(map(int, rng.permutation([c for c in range(10) if c not in turns])))
    background, frame, source, distinct_fill = remaining[:4]
    fill = background if plan['fill_matches'] else distinct_fill
    hh = int(rng.integers(3, 6))
    width = int(rng.integers(max(20, key_count * 5), 31))
    geometry = _geometry(rng, plan['count'], plan['exterior'], hh, width)
    if geometry is None:
        return None
    height, main, boxes = geometry
    t, b, l, r = main
    field = np.full((height, width), background, dtype=np.int8)
    controls = np.full((b - t - 1, r - l - 1), -1, dtype=np.int8)
    owners = np.full(controls.shape, -1, dtype=np.int8)
    for i, box in enumerate(boxes):
        if plan['lesson'] == 1:
            word = (key_colors[i % 2],)
        else:
            period = parameters['period_cap'] if plan['lesson'] == 0 and i == 0 else int(rng.integers(1, parameters['period_cap'] + 1))
            word = _primitive_word(rng, period, key_colors)
        phase = int(rng.integers(len(word)))
        for y, x in _perimeter(box):
            color = word[(y + x + phase) % len(word)]
            field[y, x] = color
            if t < y < b and l < x < r:
                controls[y - t - 1, x - l - 1] = color
                owners[y - t - 1, x - l - 1] = i
    free = np.argwhere(controls < 0)
    sr, sc = map(int, free[int(rng.integers(len(free)))])
    field[t + 1:b, l + 1:r] = fill
    for y, x in _perimeter(main):
        field[y, x] = frame
    field[t + 1 + sr, l + 1 + sc] = source
    input_grid = np.vstack((_header(rng, hh, width, frame, background, turns), field))
    lit, _ = _trace(controls, (sr, sc), turns, owners)
    output_grid = np.full(controls.shape, fill, dtype=np.int8)
    output_grid[lit] = source
    pair = dict(input=input_grid, output=output_grid)
    try:
        info = _decode(input_grid)
    except ValueError:
        return None
    if not np.array_equal(controls, info['controls']):
        return None
    expected, tracing = _answer(info)
    _require(np.array_equal(expected, output_grid), "renderer/reconstruction disagreement")
    return pair, info, tracing


def _different(first, second):
    return not np.array_equal(first, second)


def _features(info, tracing, output):
    events = tracing['events']
    collapsed = info['controls'].copy()
    shifted = info['controls'].copy()
    t, b, l, r = info['main']
    for owner, loop in enumerate(info['loops']):
        collapsed[info['owners'] == owner] = loop['reference_color']
        for (y, x), _ in loop['assignments'].items():
            if t < y < b and l < x < r:
                shifted[y - t - 1, x - l - 1] = loop['word'][(y + x + 1) % loop['period']]
    straight = _answer(info, controls=np.full(info['controls'].shape, -1, dtype=np.int8))[0]
    reversed_key = _answer(info, turns={c: -v for c, v in info['turns'].items()})[0]
    all_left = _answer(info, turns={c: -1 for c in info['turns']})[0]
    all_right = _answer(info, turns={c: 1 for c in info['turns']})[0]
    sides = {side: {event[4] for event in events if event[5] == side} for side in (-1, 1)}
    return dict(crossing_count=sum(np.any(info['owners'] == i) for i in range(len(info['loops']))),
                loop_count=len(info['loops']), periods=tuple(q['period'] for q in info['loops']),
                headings=sides, cycles=sum(tracing['cycles']), max_turns=max(map(len, tracing['branches'])),
                bridge=any(len({e[6] for e in branch}) >= 2 for branch in tracing['branches']),
                hit_loop_count=len({e[6] for e in events}),
                max_loop_chain=max(len({e[6] for e in branch}) for branch in tracing['branches']),
                no_A_diff=_different(straight, output), key_diff=_different(reversed_key, output),
                all_left_diff=_different(all_left, output), all_right_diff=_different(all_right, output),
                pattern_diff=_different(_answer(info, controls=collapsed)[0], output),
                phase_diff=_different(_answer(info, controls=shifted)[0], output),
                recursion_diff=_different(_answer(info, first_turn_only=True)[0], output),
                stopping_diff=_different(_answer(info, stop_at_control=True)[0], output),
                shape=output.shape, main=info['main'], background=info['background'],
                source_position=info['source_position'], fill_matches=info['fill'] == info['background'],
                nested=any(a < e < f < z and c < g < h < d
                           for a, z, c, d in [q['box'] for q in info['loops']]
                           for e, f, g, h in [q['box'] for q in info['loops']]),
                spanning=any((a < t and z > b) or (c < l and d > r)
                             for a, z, c, d in [q['box'] for q in info['loops']]),
                exterior_count=sum(not np.any(info['owners'] == i) for i in range(len(info['loops']))),
                source_edge_gap=min(info['source_position'][0], info['source_position'][1],
                                    output.shape[0] - 1 - info['source_position'][0],
                                    output.shape[1] - 1 - info['source_position'][1]),
                layout=tuple(loop['box'] for loop in info['loops']))


def _signature(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    _require(seed >= 0, "seed must be nonnegative")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test, "illegal pair counts")
    rng = np.random.default_rng(seed)
    parameters = dict(teaching_keys=tuple(map(int, rng.choice(10, 2, replace=False))),
                      period_cap=int(rng.integers(2, 5)), teach_cycle=bool(rng.random() < .3))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else 2
            plan = dict(lesson=lesson, count=2 if lesson < 2 else 3 if split == 'test' else int(rng.integers(1, 3)),
                        exterior=0 if lesson < 2 else int(rng.random() < .3),
                        fill_matches=lesson == 0 or (lesson > 1 and bool(rng.integers(2))))
            for attempt in range(600):
                candidate = _render_candidate(rng, plan, parameters)
                if candidate is None:
                    continue
                pair, info, tracing = candidate
                if _signature(pair['input']) in seen:
                    continue
                f = _features(info, tracing, pair['output'])
                if not f['no_A_diff'] or not f['stopping_diff'] or (lesson != 1 and not f['key_diff']):
                    continue
                if lesson == 0 and not all(len(q) >= 2 for q in f['headings'].values()):
                    continue
                if lesson == 0 and not (f['pattern_diff'] and f['phase_diff'] and f['bridge'] and f['recursion_diff']
                                        and parameters['period_cap'] in f['periods']):
                    continue
                if lesson == 1 and parameters['teach_cycle'] and not f['cycles']:
                    continue
                if lesson > 1 and not parameters['teach_cycle'] and f['cycles']:
                    continue
                if split == 'test' and not (f['bridge'] and f['hit_loop_count'] == 3):
                    continue
                puzzle[split].append(pair)
                seen.add(_signature(pair['input']))
                break
            else:
                raise RuntimeError("bounded construction exhausted for " + split + "/" + str(index))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned pixels, geometry, answers, teaching evidence, and shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, features, infos = set(), {'train': [], 'test': []}, {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _decode(inp)
            expected, tracing = _answer(info)
            _require(np.array_equal(expected, out), "incorrect light-trail output")
            _require(set(map(int, np.unique(out))) == {info['fill'], info['source']}, "output must show trails and unlit fill")
            # Every reached switch is illuminated; unvisited controls stay fill.
            for _, y, x, _, _, _, _ in tracing['events']:
                _require(out[y, x] == info['source'], "control entry cell not illuminated")
            f = _features(info, tracing, out)
            _require(f['no_A_diff'] and f['stopping_diff'], "hidden controls or entry turns are inactive")
            features[split].append(f)
            infos[split].append(info)
    training = features['train']
    _require(any(f['key_diff'] for f in training), "turn table is inactive in training")
    for side in (-1, 1):
        _require(len(set().union(*(f['headings'][side] for f in training))) >= 2, "relative turns underdemonstrated")
    _require(any(f['pattern_diff'] and f['phase_diff'] and f['bridge'] and f['recursion_diff']
                 for f in training), "no patterned multi-loop lesson")
    _require({f['fill_matches'] for f in training} == {False, True}, "fill roles underdemonstrated")
    tables = [info['turns'] for info in infos['train']]
    _require(any(len({table[c] for table in tables if c in table}) == 2 for c in range(10)), "local key reassignment not demonstrated")
    # Four possible meanings of the two bar placements; exactly one fits.
    hypotheses = []
    for left in (-1, 1):
        for right in (-1, 1):
            if all(np.array_equal(_answer(info, turns={c: left if s == -1 else right for c, s in info['turns'].items()})[0],
                                  pair['output']) for info, pair in zip(infos['train'], puzzle['train'])):
                hypotheses.append((left, right))
    _require(hypotheses == [(-1, 1)], "key meanings not uniquely demonstrated")
    for f in features['test']:
        _require(f['crossing_count'] == 3 and f['bridge'] and f['hit_loop_count'] == 3,
                 "test lacks active multi-loop shift")
        _require(f['key_diff'], "test ignores the turn table")
        _require(max(f['periods']) <= max(max(q['periods']) for q in training), "undemonstrated pattern scale")
        if f['cycles']:
            _require(any(q['cycles'] for q in training), "closed test route was not demonstrated")


def self_test():
    """Bounded checks over 42 seeds plus count, corruption, and trace fixtures."""
    metrics, fingerprints, layouts, shapes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                out, tracing = _answer(info)
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list-input solve failed")
                f = _features(info, tracing, out)
                metrics.append(f)
                layouts.add((f['main'], f['layout'], f['source_position']))
                shapes.add(f['shape'])
    _require(len(fingerprints) == 42 and len(layouts) >= 180 and len(shapes) >= 35, "insufficient structural variation")
    _require({f['background'] for f in metrics} == set(range(10)), "background roles do not span palette")
    _require({p for f in metrics for p in f['periods']} == {1, 2, 3, 4}, "missing pattern families")
    _require({f['crossing_count'] for f in metrics} >= {1, 2, 3}, "missing loop counts")
    _require(sum(f['cycles'] > 0 for f in metrics) >= 5 and sum(f['bridge'] for f in metrics) >= 60, "missing route families")
    _require(any(f['max_turns'] >= 8 for f in metrics), "missing longer routes")
    _require(any(f['nested'] for f in metrics) and any(f['spanning'] for f in metrics)
             and any(f['exterior_count'] > 0 for f in metrics), "missing loop geometries")
    _require(any(f['source_edge_gap'] == 0 for f in metrics), "missing boundary-adjacent sources")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        first, second = generate_puzzle(101 + nt + nq, nt, nq), generate_puzzle(101 + nt + nq, nt, nq)
        _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a), "nondeterminism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(42, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    # A four-switch square: right turns close one branch; finite-state tracing
    # must keep both headings at shared positions and terminate without a cap.
    controls = np.full((7, 7), -1, dtype=np.int8)
    for p in ((1, 5), (5, 5), (5, 1), (1, 1)):
        controls[p] = 2
    lit, tracing = _trace(controls, (1, 3), {2: 1})
    _require(sum(tracing['cycles']) >= 1 and lit[1, 5] and lit[5, 1], "closed-route fixture failed")
    return dict(seeds=42, pairs=len(metrics), distinct_layouts=len(layouts), output_shapes=len(shapes),
                cyclic_pairs=sum(f['cycles'] > 0 for f in metrics),
                multi_loop_pairs=sum(f['bridge'] for f in metrics),
                nested_pairs=sum(f['nested'] for f in metrics),
                spanning_pairs=sum(f['spanning'] for f in metrics),
                exterior_loop_pairs=sum(f['exterior_count'] > 0 for f in metrics),
                periods=sorted({p for f in metrics for p in f['periods']}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
