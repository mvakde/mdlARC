"""Cyclic locksmith: moving paint-window queries open rectangular obstacles.

Only the visible grid is used by solve. No parent module is imported.
"""
from __future__ import annotations

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("Example counts must be integers")
    a, b = int(num_train), int(num_test)
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _orient(grid, code):
    return np.rot90(grid[:, ::-1] if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    a = np.rot90(grid, -(code % 4))
    return (a[:, ::-1] if code >= 4 else a).copy()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells = [first]
        for r, c in cells:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
        result.append(cells)
    return result


def _separate(a, b):
    """Half-open boxes have at least one complete cell of separation."""
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _box_cells(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _canonical_info(a):
    h, w = a.shape
    assert h >= 10 and w >= 11
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[3:5] == bg)
    assert np.array_equal(a[1], a[2])
    columns, word, c = [], [], 0
    while c < w:
        if a[1, c] == bg:
            c += 1
            continue
        assert c+1 < w and a[1, c+1] == a[1, c]
        assert c+2 == w or a[1, c+2] == bg
        columns.append(c)
        word.append(int(a[1, c]))
        c += 2
    assert 3 <= len(word) <= 7 and len(set(word)) >= 2
    # The unique launch singleton sits on the entry edge. Record borders never
    # touch that edge, so label cells cannot be mistaken for a launch marker.
    starts = [(r, 0) for r in range(5, h) if int(a[r, 0]) == word[0]]
    assert len(starts) == 1
    start = starts[0]
    route = int(a[start[0], 1])
    assert route != bg and route not in word
    for p in ((start[0]-1, 0), (start[0]+1, 0)):
        assert not _inside(a.shape, p) or int(a[p]) in (bg, route)
    mask = ~np.isin(a, (bg, route))
    mask[:5] = False
    mask[start] = False
    records, filler = [], None
    for cells in _components(mask):
        rows, cols = zip(*cells)
        box = (int(min(rows)), int(max(rows)+1), int(min(cols)), int(max(cols)+1))
        r0, r1, c0, c1 = box
        assert r0 >= 5 and c0 >= 1 and r1-r0 >= 3 and c1-c0 >= 3
        assert len(cells) == (r1-r0)*(c1-c0)
        block = a[r0:r1, c0:c1]
        border = np.r_[block[0], block[-1], block[1:-1, 0], block[1:-1, -1]]
        f = int(border[0])
        assert np.all(border == f) and f not in set(word) | {bg, route}
        assert filler is None or filler == f
        filler = f
        labels = frozenset(map(int, np.unique(block))) - {f}
        assert not labels & {bg, route, f}
        assert all(_separate(box, old['box']) for old in records)
        counts = {int(x): int(n) for x, n in zip(*np.unique(block, return_counts=True))}
        records.append(dict(box=box, labels=labels, counts=counts,
                            marks=int(block.size-counts[f])))
    assert records
    restored = a.copy()
    restored[:5] = bg
    restored[start] = route
    return dict(grid=a, restored=restored, background=bg, route=route,
                filler=filler, word=tuple(word), columns=tuple(columns),
                start=start, records=records)


def _decode(grid):
    candidates = []
    for code in range(8):
        try:
            info = _canonical_info(_unorient(grid, code))
        except AssertionError:
            continue
        info['orientation'] = code
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("Input must have one readable program/entry orientation")
    return candidates[0]


def _simulate(info, alternative=None):
    """Monotone cell walker with dynamic opening; alternatives are audit probes."""
    out = info['restored'].copy()
    terrain = out == info['route']
    owner = np.full(out.shape, -1, dtype=int)
    for index, obj in enumerate(info['records']):
        r0, r1, c0, c1 = obj['box']
        owner[r0:r1, c0:c1] = index
    word = info['word']
    if alternative == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    directions = ((0, 1), (-1, 0))
    current, direction, phase = info['start'], 0, 0
    history, leg, path, contacts, turns, opened = [], [], [], [], [], set()
    lengths = [0]
    for _ in range(sum(out.shape)+1):
        color = word[phase % len(word)]
        out[current] = color
        history.append(color)
        leg.append(color)
        path.append((current, color))
        phase += 1
        lengths[-1] += 1
        dr, dc = directions[direction]
        ahead = (current[0]+dr, current[1]+dc)
        if not _inside(out.shape, ahead):
            end = 'edge'
            break
        index = int(owner[ahead])
        if index >= 0:
            query = frozenset(history[-3:])
            if alternative == 'whole_word':
                query = frozenset(word)
            elif alternative == 'whole_history':
                query = frozenset(history)
            elif alternative == 'whole_leg':
                query = frozenset(leg)
            elif alternative == 'next_three':
                query = frozenset(word[(phase+j) % len(word)] for j in range(3))
            elif alternative == 'last_color':
                query = frozenset(history[-1:])
            elif alternative == 'first_window':
                query = frozenset(word[:3])
            elif alternative == 'drop_oldest':
                query = frozenset(history[-2:])
            elif alternative == 'drop_middle':
                query = frozenset((history[-3], history[-1]))
            elif alternative == 'drop_newest':
                query = frozenset(history[-3:-1])
            obj = info['records'][index]
            labels = obj['labels']
            success = query <= labels
            if alternative == 'open_all':
                success = True
            elif alternative == 'open_none':
                success = False
            elif alternative == 'overlap':
                success = bool(query & labels)
            elif alternative == 'exact_set':
                success = query == labels
            elif alternative == 'mark_count':
                success = obj['marks'] >= len(query)
            elif alternative == 'color_count':
                success = len(labels) >= len(query)
            contacts.append(dict(index=index, query=query, success=success,
                                 tail=tuple(history[-3:]), leg_length=len(leg),
                                 painted=len(history), direction=direction))
            if success:
                r0, r1, c0, c1 = obj['box']
                out[r0:r1, c0:c1] = info['route']
                terrain[r0:r1, c0:c1] = True
                owner[r0:r1, c0:c1] = -1
                opened.add(index)
                if alternative == 'consume_open':
                    phase += 1
        if terrain[ahead]:
            current = ahead
            continue
        # Walls and failed records end a leg. A turn cannot itself open a gate.
        direction = 1-direction
        dr, dc = directions[direction]
        after_turn = (current[0]+dr, current[1]+dc)
        if not _inside(out.shape, after_turn) or not terrain[after_turn]:
            end = 'blocked'
            break
        turns.append(current)
        lengths.append(0)
        leg = []
        if alternative == 'reset_bends':
            phase = 0
        current = after_turn
    else:
        raise AssertionError("Monotone walk exceeded the canvas extent")
    return out, dict(path=path, contacts=contacts, turns=turns, opened=opened,
                     lengths=tuple(lengths), end=end)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(_grid(input_grid))
    out, _ = _simulate(info)
    return _orient(out, info['orientation'])


_PROBES = ('deduplicate', 'reset_bends', 'consume_open', 'whole_word',
           'whole_history', 'whole_leg', 'next_three', 'last_color',
           'first_window', 'open_all', 'open_none', 'overlap', 'exact_set',
           'mark_count', 'color_count', 'drop_oldest', 'drop_middle', 'drop_newest')
_REQUIRED = {'deduplicate', 'reset_bends', 'consume_open', 'whole_word',
             'whole_history', 'whole_leg', 'next_three', 'last_color',
             'first_window', 'open_all', 'open_none', 'overlap', 'exact_set',
             'mark_count', 'color_count', 'matched_counts', 'unvisited',
             'cross_bend_query', 'small_query', 'drop_oldest', 'drop_middle', 'drop_newest'}


def _evidence(info):
    correct, trace = _simulate(info)
    evidence = {name for name in _PROBES
                if not np.array_equal(correct, _simulate(info, name)[0])}
    contacts = trace['contacts']
    positive = [c for c in contacts if c['success']]
    negative = [c for c in contacts if not c['success']]
    if any(c['leg_length'] < 3 and c['painted'] >= 3 and not c['success']
           and frozenset(c['tail'][-c['leg_length']:]) <=
           info['records'][c['index']]['labels'] for c in contacts):
        evidence.add('cross_bend_query')
    if any(len(c['query']) < 3 for c in contacts):
        evidence.add('small_query')
    for p in positive:
        a = info['records'][p['index']]
        for n in negative:
            b = info['records'][n['index']]
            if a['marks'] == b['marks'] and len(a['labels']) == len(b['labels']):
                evidence.add('matched_counts')
    hit = {c['index'] for c in contacts}
    if any(j not in hit and set(info['word']) <= obj['labels']
           for j, obj in enumerate(info['records'])):
        evidence.add('unvisited')
    return evidence


def _sample_word(rng, length=None, forbidden_length=None):
    n = (int(length) if length is not None else
         int(rng.choice([x for x in range(4, 8) if x != forbidden_length])))
    d = int(rng.integers(3, min(5, n-1)+1))
    word = list(range(d)) + list(map(int, rng.integers(d, size=n-d)))
    rng.shuffle(word)
    return tuple(word)


def _marks(rng, query, labels, capacity, success, package, missing):
    q = list(dict.fromkeys(query))
    extra = [x for x in labels if x not in q]
    rng.shuffle(extra)
    if success:
        marks = q.copy()
        if package == 'extras' and extra and len(marks) < capacity:
            marks.append(extra[0])
    else:
        absent = q[missing % len(q)]
        marks = [x for x in q if x != absent]
        if extra:
            marks.append(extra[0])
        if package == 'extras' and len(extra) > 1 and len(marks) < capacity:
            marks.append(extra[1])
    if package == 'extras' and marks:
        target = int(rng.integers(len(marks), capacity+1))
        while len(marks) < target:
            marks.append(int(rng.choice(marks)))
    return marks


def _paint_record(grid, box, marks, filler, rng):
    r0, r1, c0, c1 = box
    grid[r0:r1, c0:c1] = filler
    interior = [(r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)]
    assert len(marks) <= len(interior)
    for j, color in zip(rng.permutation(len(interior)), marks):
        grid[interior[int(j)]] = color


def _planned_scene(rng, h, w, word, roles, labels, package, outcomes,
                   force_cross_bend=False):
    """Plant encounters on a monotone walk; truth comes from actual paint tails."""
    bg, route, filler = roles
    grid = np.full((h, w), route, dtype=np.int8)
    grid[:5] = bg
    left = int(rng.integers(0, max(1, min(4, w-(3*len(word)-1)))))
    columns = [left+3*j for j in range(len(word))]
    for c, color in zip(columns, word):
        grid[1:3, c:c+2] = color
    start = (h-int(rng.integers(4, 8)), 0)
    cursor, direction = start, 0
    walked, history, boxes, planned = set(), [], [], []
    negative_count = 0
    shape_options = [(3, 5), (4, 4), (5, 3), (4, 5), (5, 4), (5, 5)]
    for event, success in enumerate(outcomes):
        choices = []
        dr, dc = ((0, 1), (-1, 0))[direction]
        distances = range(1, 3) if force_cross_bend and event == 1 else range(3 if event == 0 else 1, 8)
        for distance in distances:
            hit = (cursor[0]+distance*dr, cursor[1]+distance*dc)
            approach = [(cursor[0]+j*dr, cursor[1]+j*dc) for j in range(distance)]
            if package == 'identity' and not success:
                proposed = history + [word[(len(history)+j) % len(word)] for j in range(distance)]
                if len(set(proposed[-3:])) != 3:
                    continue
            if not all(_inside(grid.shape, p) and p[0] >= 5 for p in approach):
                continue
            if any(_box_cells(b) & set(approach) for b in boxes):
                continue
            for gh, gw in shape_options:
                for offset in range(gh if direction == 0 else gw):
                    if force_cross_bend and event == 0 and offset != 0:
                        continue
                    box = ((hit[0]-offset, hit[0]-offset+gh, hit[1], hit[1]+gw)
                           if direction == 0 else
                           (hit[0]-gh+1, hit[0]+1, hit[1]-offset, hit[1]-offset+gw))
                    r0, r1, c0, c1 = box
                    if not (6 <= r0 < r1 <= h and 1 <= c0 < c1 <= w):
                        continue
                    if not all(_separate(box, old) for old in boxes):
                        continue
                    if _box_cells(box) & (walked | set(approach)):
                        continue
                    endpoint = approach[-1]
                    if success:
                        crossing = [(hit[0]+j*dr, hit[1]+j*dc)
                                    for j in range(gw if direction == 0 else gh)]
                        following = (crossing[-1][0]+dr, crossing[-1][1]+dc)
                    else:
                        crossing = []
                        tr, tc = ((-1, 0), (0, 1))[direction]
                        following = (endpoint[0]+tr, endpoint[1]+tc)
                    if not _inside(grid.shape, following) or following[0] < 5:
                        continue
                    if any(following in _box_cells(b) for b in boxes+[box]):
                        continue
                    choices.append((approach, crossing, following, box))
        if not choices:
            return None
        # Short approaches sometimes make the window straddle a bend; longer
        # ones exercise continuous phase, wraparound, and varied leg lengths.
        approach, crossing, following, box = choices[int(rng.integers(len(choices)))]
        for p in approach:
            history.append(word[len(history) % len(word)])
            walked.add(p)
        tail = history[-3:]
        capacity = (box[1]-box[0]-2)*(box[3]-box[2]-2)
        missing = 2-negative_count if package == 'identity' else event-1
        marks = _marks(rng, tail, labels, capacity, success, package, missing)
        negative_count += not success
        _paint_record(grid, box, marks, filler, rng)
        planned.append((box, frozenset(tail), success))
        boxes.append(box)
        for p in crossing:
            history.append(word[len(history) % len(word)])
            walked.add(p)
        cursor = following
        if not success:
            direction = 1-direction
    grid[start] = word[0]
    info = _canonical_info(grid)
    _, trace = _simulate(info)
    observed = [(info['records'][c['index']]['box'], c['query'], c['success'])
                for c in trace['contacts']]
    if observed != planned:
        return None
    protected = {p for p, _ in trace['path']}
    # Add records off the route. In teaching scenes, one contains every word
    # identity but is untouched: membership alone is not a global deletion cue.
    extras_count = int(rng.integers(1, 5))
    complete_added = False
    for _ in range(extras_count):
        possible = []
        for _ in range(160):
            gh, gw = int(rng.integers(3, 8)), int(rng.integers(3, 8))
            r0 = int(rng.integers(5, h-gh+1))
            c0 = int(rng.integers(1, w-gw+1))
            box = (r0, r0+gh, c0, c0+gw)
            if (all(_separate(box, b) for b in boxes)
                    and not _box_cells(box) & protected):
                possible.append(box)
        if not possible:
            continue
        box = possible[int(rng.integers(len(possible)))]
        capacity = (box[1]-box[0]-2)*(box[3]-box[2]-2)
        if not complete_added and capacity >= len(set(word)):
            marks = list(dict.fromkeys(word))
            complete_added = True
        else:
            marks = list(map(int, rng.choice(labels, int(rng.integers(capacity+1)))))
        _paint_record(grid, box, marks, filler, rng)
        boxes.append(box)
    if package != 'free' and not complete_added:
        return None
    # Carve independent bays/holes in unused terrain, without altering the
    # planned route or hiding records. No constant-width ribbon is used.
    occupied = protected | set().union(*(_box_cells(b) for b in boxes))
    for _ in range(int(rng.integers(4, 20))):
        gh, gw = int(rng.integers(1, 7)), int(rng.integers(1, 8))
        r0, c0 = int(rng.integers(5, h-gh+1)), int(rng.integers(w-gw+1))
        box = (r0, r0+gh, c0, c0+gw)
        if not _box_cells(box) & occupied:
            grid[r0:r0+gh, c0:c0+gw] = bg
    # A terminal terrain barrier sometimes stops the route before the edge.
    # It is added beyond every planned encounter, so it cannot remove teaching
    # evidence. Both the forward and the alternate neighbor are visibly blocked.
    if rng.random() < .35:
        path = [p for p, _ in trace['path']]
        last_contact = max(c['painted'] for c in trace['contacts'])
        record_cells = set().union(*(_box_cells(b) for b in boxes))
        endings = []
        for j in range(last_contact+5, len(path)-1):
            current, ahead = path[j], path[j+1]
            dr, dc = ahead[0]-current[0], ahead[1]-current[1]
            turn = (current[0]-1, current[1]) if dc else (current[0], current[1]+1)
            if (_inside(grid.shape, turn) and turn[0] >= 5
                    and not {ahead, turn} & (set(path[:j+1]) | record_cells)):
                endings.append((ahead, turn))
        if endings:
            ahead, turn = endings[int(rng.integers(len(endings)))]
            grid[ahead] = bg
            grid[turn] = bg
    return grid


def _make_pair(rng, package, code, length=None, forbidden_length=None,
               needed=frozenset()):
    for _ in range(260):
        palette = tuple(map(int, rng.permutation(10)))
        roles, labels = palette[:3], palette[3:]
        abstract = _sample_word(rng, length, forbidden_length)
        word = tuple(labels[j] for j in abstract)
        teaching = package != 'free'
        h = int(rng.integers(26 if teaching else 18, 31))
        w = int(rng.integers(max(26 if teaching else 18, 3*len(word)+1), 31))
        if teaching:
            plans = ([False, True, False, True, False],
                     [False, False, True, False, True],
                     [True, False, False, True, False],
                     [True, False, True, False])
            outcomes = list(plans[int(rng.integers(len(plans)))])
            if package == 'identity':
                # Negatives omit newest, middle, and oldest entries regardless
                # of encounter order. Vary successes and adjacent failures.
                identity_plans = ([False, True, False, True, False],
                                  [False, False, True, False],
                                  [True, False, False, False],
                                  [True, False, True, False, False],
                                  [False, True, False, False],
                                  [True, False, True, False, True, False])
                outcomes = list(identity_plans[int(rng.integers(len(identity_plans)))])
            if 'cross_bend_query' in needed:
                outcomes = [False, False, True, False, True]
        else:
            outcomes = list(map(bool, rng.integers(2, size=int(rng.integers(1, 6)))))
        grid = _planned_scene(rng, h, w, word, roles, labels, package, outcomes,
                              force_cross_bend='cross_bend_query' in needed)
        if grid is None:
            continue
        info = _canonical_info(grid)
        info['orientation'] = 0
        if not set(needed) <= _evidence(info):
            continue
        out, _ = _simulate(info)
        return dict(input=_orient(grid, code), output=_orient(out, code))
    raise RuntimeError("Bounded scene sampling could not expose the requested witnesses")


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = int(rng.integers(8))
    mixed = bool(rng.integers(2))
    codes = [base, (base+1) % 8] if mixed else [base]
    held_out_length = int(rng.choice((5, 6, 7)))
    # The second teaching scene supplies any distinction absent from the first.
    puzzle, seen, evidence = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            package = ('identity' if index == 0 else 'extras' if index == 1
                       else 'free') if split == 'train' else 'free'
            code = codes[index % len(codes)] if split == 'train' else int(rng.choice(codes))
            needed = (_REQUIRED - evidence) if split == 'train' and index == 1 else (
                {'matched_counts', 'unvisited', 'drop_oldest', 'drop_middle', 'drop_newest'}
                if split == 'train' and index == 0 else set())
            for _ in range(8):
                pair = _make_pair(rng, package, code,
                                  length=held_out_length if split == 'test' else None,
                                  forbidden_length=held_out_length if split == 'train' else None,
                                  needed=needed)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct input sampling exhausted")
            puzzle[split].append(pair)
            seen.add(key)
            if split == 'train':
                evidence |= _evidence(_decode(pair['input']))
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, train_codes, test_codes = set(), set(), set(), set()
    train_lengths, test_lengths = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected, trace = _simulate(info)
            observed = _unorient(out, info['orientation'])
            assert np.array_equal(observed, expected), 'Incorrect dynamic opening or paint'
            path = [p for p, _ in trace['path']]
            assert len(path) == len(set(path)) and len(path) <= sum(inp.shape)-1
            assert all(c['painted'] >= 3 for c in trace['contacts'])
            # Reconstruct the output footprint from the original records and
            # traced paints: whole opening, untouched records, annotation erasure.
            reconstructed = info['restored'].copy()
            for index, obj in enumerate(info['records']):
                r0, r1, c0, c1 = obj['box']
                if index in trace['opened']:
                    reconstructed[r0:r1, c0:c1] = info['route']
                else:
                    assert np.array_equal(observed[r0:r1, c0:c1],
                                          info['grid'][r0:r1, c0:c1])
            for p, color in trace['path']:
                reconstructed[p] = color
            assert np.array_equal(observed, reconstructed)
            assert all(c['success'] == (c['query'] <= info['records'][c['index']]['labels'])
                       for c in trace['contacts'])
            if split == 'train':
                evidence |= _evidence(info)
                train_codes.add(info['orientation'])
                train_lengths.add(len(info['word']))
            else:
                test_codes.add(info['orientation'])
                test_lengths.add(len(info['word']))
    assert _REQUIRED <= evidence, ('Missing training evidence', _REQUIRED-evidence)
    assert test_codes <= train_codes, 'Tests must use a demonstrated orientation'
    assert train_lengths.isdisjoint(test_lengths), 'Cycle length shift must be visible'


def self_test():
    _semantic_fixture()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, patterns, roles, codes, layouts, sizes, queries, opened_counts, endings = (set() for _ in range(9))
    for seed, puzzle in enumerate(puzzles):
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                _, trace = _simulate(info)
                shapes.add(pair['input'].shape)
                mapping = {}
                patterns.add(tuple(mapping.setdefault(c, len(mapping)) for c in info['word']))
                roles.add((info['background'], info['route'], info['filler']))
                codes.add(info['orientation'])
                layouts.add((info['grid'].shape, tuple(o['box'] for o in info['records'])))
                sizes.update((o['box'][1]-o['box'][0], o['box'][3]-o['box'][2])
                             for o in info['records'])
                queries.update(len(c['query']) for c in trace['contacts'])
                opened_counts.add(len(trace['opened']))
                endings.add(trace['end'])
                # Relocating/repeating interior labels while retaining their set
                # must leave all route decisions and paints unchanged.
                mutated = info['grid'].copy()
                for obj in info['records']:
                    r0, r1, c0, c1 = obj['box']
                    mutated[r0+1:r1-1, c0+1:c1-1] = np.flip(
                        mutated[r0+1:r1-1, c0+1:c1-1].copy())
                changed_info = _canonical_info(mutated)
                changed, changed_trace = _simulate(changed_info)
                assert trace['path'] == changed_trace['path']
                assert trace['opened'] == changed_trace['opened']
                for j, obj in enumerate(changed_info['records']):
                    if j not in trace['opened']:
                        r0, r1, c0, c1 = obj['box']
                        assert np.array_equal(changed[r0:r1, c0:c1], mutated[r0:r1, c0:c1])
                positive = next((c for c in trace['contacts'] if c['success']), None)
                if positive is not None:
                    # Remove one required identity, changing neither geometry
                    # nor program. The same encounter must now fail and survive.
                    obj = info['records'][positive['index']]
                    altered = info['grid'].copy()
                    r0, r1, c0, c1 = obj['box']
                    absent = min(positive['query'])
                    replacement = next(c for c in range(10)
                                       if c not in positive['query'] |
                                       {info['background'], info['route'], info['filler']})
                    block = altered[r0:r1, c0:c1]
                    block[block == absent] = replacement
                    changed, counter = _simulate(_canonical_info(altered))
                    encounter = next(c for c in counter['contacts']
                                     if c['index'] == positive['index'])
                    assert encounter['query'] == positive['query'] and not encounter['success']
                    assert np.array_equal(changed[r0:r1, c0:c1], altered[r0:r1, c0:c1])
    assert len(shapes) >= 35 and len(patterns) >= 20 and len(layouts) >= 180
    assert len(roles) >= 100 and codes == set(range(8))
    assert len(sizes) >= 12 and queries == {1, 2, 3}
    assert len(opened_counts) >= 4
    assert endings == {'edge', 'blocked'}
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(317, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal example counts accepted')
    # Direct termination fixture: exiting right never turns up, even if that
    # neighbor is available. Removing gates from a teaching scene leaves this.
    fixture = _decode(puzzles[0]['train'][0]['input'])
    plain = fixture['grid'].copy()
    plain[5:] = fixture['route']
    plain[fixture['start']] = fixture['word'][0]
    # Keep one isolated record, away from the launch row and its route.
    r = 6 if fixture['start'][0] >= 10 else 12
    plain[r:r+3, 1:6] = fixture['filler']
    direct = _canonical_info(plain)
    _, trace = _simulate(direct)
    assert trace['end'] == 'edge' and not trace['turns']
    assert all(p[0] == fixture['start'][0] for p, _ in trace['path'])


def _semantic_fixture():
    """An analytically specified case, with no seed or sampler-derived labels."""
    a = np.ones((18, 22), dtype=np.int8)
    a[:5] = 0
    word = (3, 4, 5, 3, 6)
    for j, color in enumerate(word):
        a[1:3, 1+3*j:3+3*j] = color
    a[14, 0] = 3
    a[13:17, 5:9] = 2
    a[14:16, 6:8] = [[5, 3], [6, 7]]
    a[14:18, 12:16] = 2
    a[15:17, 13:15] = [[3, 4], [7, 2]]
    a[8:12, 9:13] = 2
    a[9:11, 10:12] = [[5, 3], [7, 2]]
    path = ([(14, c) for c in range(12)] + [(13, 11), (12, 11)]
            + [(12, c) for c in range(12, 22)])
    expected = a.copy()
    expected[:5] = 0
    expected[13:17, 5:9] = 1
    for i, p in enumerate(path):
        expected[p] = word[i % len(word)]
    for code in range(8):
        assert np.array_equal(solve(_orient(a, code)), _orient(expected, code))
    info = _decode(a)
    _, trace = _simulate(info)
    assert trace['lengths'] == (12, 2, 10)
    assert [(c['query'], c['success']) for c in trace['contacts']] == [
        (frozenset((3, 5, 6)), True), (frozenset((3, 4, 6)), False),
        (frozenset((3, 4, 5)), False)]
    assert 'cross_bend_query' in _evidence(info)
    # The endpoint has available terrain above, yet exiting right stops.
    assert trace['end'] == 'edge' and expected[11, 21] == 1


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, explicit count matrix, semantic probes)')
