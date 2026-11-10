"""Chromatic lock mirrors: cyclic paint controls diagonal reflection.

Only NumPy and the standard library are used. All routing/program conventions
are fixed here and in task_spec.md; solve has no access to generation state.
"""
from __future__ import annotations

import json
import numpy as np


DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
REQUIRED = {"cycle", "multiplicity", "pause", "continuity", "restart",
            "conditional", "reflection", "program", "prospective",
            "order", "launch_phase", "continuity_one"}


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(grid, bg, excluded=()):
    todo = {tuple(map(int, p)) for p in np.argwhere(grid != bg)} - set(excluded)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, cells = [first], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in NEIGHBORS:
                q = (p[0] + dr, p[1] + dc)
                if q in todo and grid[q] == grid[first]:
                    todo.remove(q)
                    stack.append(q)
        result.append((int(grid[first]), tuple(sorted(cells))))
    return result


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size / 2, "background must be a strict majority"
    return bg


def _decode(grid):
    bg = _background(grid)
    swatches = []
    for shade, cells in _components(grid, bg):
        rows, cols = {p[0] for p in cells}, {p[1] for p in cells}
        if (len(cells) == 4 and len(rows) == len(cols) == 2
                and max(rows) - min(rows) == max(cols) - min(cols) == 1):
            swatches.append(dict(color=shade, cells=cells,
                                 origin=(min(rows), min(cols))))
    assert 4 <= len(swatches) <= 7, "three to six program squares plus marker"
    candidates = []
    for marker in swatches:
        legend = [s for s in swatches if s is not marker]
        for axis in (0, 1):  # horizontal: constant row; vertical: constant col
            if len({s['origin'][axis] for s in legend}) != 1:
                continue
            if marker['origin'][axis] == legend[0]['origin'][axis]:
                continue
            ordered = sorted(legend, key=lambda s: s['origin'][1-axis])
            if any(b['origin'][1-axis] - a['origin'][1-axis] < 3
                   for a, b in zip(ordered, ordered[1:])):
                continue
            word = tuple(s['color'] for s in ordered)
            if len(set(word)) < 2 or marker['color'] in word:
                continue
            candidates.append(dict(bg=bg, word=word, marker=marker['color'],
                                   legend=ordered, marker_square=marker,
                                   legend_axis=axis))
    assert len(candidates) == 1, "unique ordered legend and separate marker required"
    info = candidates[0]
    info['annotations'] = {p for s in swatches for p in s['cells']}
    for p in info['annotations']:
        for dr, dc in NEIGHBORS:
            q = p[0]+dr, p[1]+dc
            if _inside(grid.shape, q) and q not in info['annotations']:
                assert grid[q] == bg, "square annotations restore to visible background"
    return info


def _objects(grid, info):
    sources, mirrors = [], []
    for shade, cells in _components(grid, info['bg'], info['annotations']):
        rows, cols = {p[0] for p in cells}, {p[1] for p in cells}
        if (len(cells) == 3 and len(rows) == len(cols) == 2
                and max(rows)-min(rows) == max(cols)-min(cols) == 1):
            missing = next(iter({(r, c) for r in rows for c in cols} - set(cells)))
            elbow = (min(rows)+max(rows)-missing[0],
                     min(cols)+max(cols)-missing[1])
            assert shade == info['word'][0], "sources carry the first program color"
            sources.append(dict(color=shade, cells=cells, elbow=elbow,
                                direction=(elbow[0]-missing[0], elbow[1]-missing[1])))
        else:
            assert len(cells) % 2 == 1 and (len(rows) == 1 or len(cols) == 1)
            assert len(cells) == max(max(rows)-min(rows), max(cols)-min(cols))+1
            assert shade in info['word'], "mirrors are keyed by program colors"
            mirrors.append(dict(color=shade, cells=cells,
                                midpoint=((min(rows)+max(rows))//2,
                                          (min(cols)+max(cols))//2),
                                axis=(None if len(cells) == 1 else
                                      0 if len(rows) == 1 else 1)))
    assert sources, "at least one L source required"
    return sources, mirrors


def _contact(mirror, p, direction):
    mid = mirror['midpoint']
    for axis in (0, 1):
        if mirror['axis'] not in (None, axis):
            continue
        if (p[1-axis] == mid[1-axis]
                and mid[axis]-p[axis] == direction[axis]):
            return axis
    return None


def _run(grid, info, mode="correct"):
    """Input-derived cell tracer; incorrect modes are bounded coverage probes."""
    sources, mirrors = _objects(grid, info)
    word = info['word']
    if mode == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    elif mode == 'constant':
        word = (word[0],)
    elif mode == 'reverse':
        word = (word[0],) + word[:0:-1]
    occupied = {p for o in sources+mirrors for p in o['cells']}
    out = grid.copy()
    for p in info['annotations']:
        out[p] = info['bg']
    painted, routes = set(), []
    global_phase = 1
    for source in sources:
        direction = source['direction']
        phase = global_phase if mode == 'global' else 0 if mode == 'start_zero' else 1
        p = tuple(source['elbow'][a]+direction[a] for a in (0, 1))
        path, contacts, seen = [], [], set()
        while _inside(grid.shape, p):
            assert p not in seen, "self-intersecting or looping ray"
            seen.add(p)
            assert p not in occupied, "off-center object strike"
            assert p not in info['annotations'], "ray crosses a program annotation"
            assert p not in painted, "rays must have disjoint cell footprints"
            hits = [(i, _contact(m, p, direction)) for i, m in enumerate(mirrors)]
            hits = [(i, a) for i, a in hits if a is not None]
            assert len(hits) <= 1, "ambiguous midpoint contact"
            matched = None
            if hits:
                index, axis = hits[0]
                due = word[(phase - (mode == 'previous')) % len(word)]
                matched = due == mirrors[index]['color']
                if mode == 'always':
                    matched = True
                elif mode == 'never':
                    matched = False
                contacts.append(dict(mirror=index, axis=axis, cell=p,
                                     phase=phase % len(word), matched=bool(matched)))
                color = info['marker']
                if matched:
                    direction = tuple(-d if a == axis else d
                                      for a, d in enumerate(direction))
                if mode == 'consume':
                    phase += 1
                elif mode == 'reset' and matched:
                    phase = 0
                elif mode == 'reset_one' and matched:
                    phase = 1
            else:
                color = word[phase % len(word)]
                phase += 1
            path.append(dict(cell=p, color=int(color),
                             contact=bool(hits)))
            painted.add(p)
            out[p] = color
            if matched is False:
                end = 'locked'
                break
            p = tuple(p[a]+direction[a] for a in (0, 1))
        else:
            end = 'edge'
        global_phase = phase
        routes.append(dict(source=source, path=path, contacts=contacts,
                           end=end, phase=phase % len(word)))
    return out, dict(info, sources=sources, mirrors=mirrors, routes=routes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _run(grid, _decode(grid))[0]


def _l_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _bar(mid, axis, length):
    tangent = 1-axis
    return {tuple(mid[a] + (k if a == tangent else 0) for a in (0, 1))
            for k in range(-(length//2), length//2+1)}


def _place(grid, cells, color, bg):
    if not all(_inside(grid.shape, p) for p in cells):
        return None
    for p in cells:
        if grid[p] != bg:
            return None
        for dr, dc in NEIGHBORS:
            q = p[0]+dr, p[1]+dc
            if _inside(grid.shape, q) and q not in cells and grid[q] != bg:
                return None
    result = grid.copy()
    for p in cells:
        result[p] = color
    return result


def _contact_signature(route, mirrors):
    return [(mirrors[c['mirror']]['midpoint'], c['axis'], c['cell'], c['matched'])
            for c in route['contacts']]


def _source_route(info, elbow):
    return next(r for r in info['routes'] if r['source']['elbow'] == elbow)


def _add_source(grid, info, rng, minimum=5):
    options = [(r, c, d) for r in range(grid.shape[0])
               for c in range(grid.shape[1]) for d in DIRS]
    rng.shuffle(options)
    for r, c, direction in options[:450]:
        trial = _place(grid, _l_cells((r, c), direction), info['word'][0], info['bg'])
        if trial is None:
            continue
        try:
            _, traced = _run(trial, info)
        except AssertionError:
            continue
        route = _source_route(traced, (r, c))
        if len(route['path']) >= minimum and not route['contacts']:
            return trial, (r, c)
    return None


def _append_mirror(grid, info, rng, elbow, matched, color=None, force_axis=None):
    _, before = _run(grid, info)
    route = _source_route(before, elbow)
    if route['end'] != 'edge':
        return None
    direction = route['source']['direction']
    anchor = elbow
    for event in route['contacts']:
        axis = event['axis']
        direction = tuple(-d if a == axis else d for a, d in enumerate(direction))
        anchor = event['cell']
    phase = 1
    tail = []
    last_contact = max((i for i, s in enumerate(route['path']) if s['contact']), default=-1)
    for i, s in enumerate(route['path']):
        if i > last_contact:
            distance = abs(s['cell'][0]-anchor[0])
            if distance >= 2:
                due = info['word'][phase % len(info['word'])]
                shades = [due] if matched else [x for x in set(info['word']) if x != due]
                for shade in shades:
                    if color is None or color == shade:
                        tail.append((s['cell'], shade))
        if not s['contact']:
            phase += 1
    options = [(p, shade, axis, length) for p, shade in tail
               for axis in ((force_axis,) if force_axis is not None else (0, 1))
               for length in (1, 3, 5, 7)]
    rng.shuffle(options)
    old = _contact_signature(route, before['mirrors'])
    for contact, shade, axis, length in options[:200]:
        mid = tuple(contact[a]+(direction[a] if a == axis else 0) for a in (0, 1))
        trial = _place(grid, _bar(mid, axis, length), shade, info['bg'])
        if trial is None:
            continue
        try:
            _, after = _run(trial, info)
        except AssertionError:
            continue
        new_route = _source_route(after, elbow)
        if _contact_signature(new_route, after['mirrors']) != old + [(mid, axis, contact, matched)]:
            continue
        # A bounce must show at least three outgoing colored cells.
        if matched:
            j = next(i for i, s in enumerate(new_route['path']) if s['cell'] == contact)
            if len(new_route['path'])-j-1 < 3:
                continue
        # Do not silently reroute a previously completed source.
        if any(_source_route(after, r['source']['elbow'])['path'] != r['path']
               for r in before['routes'] if r['source']['elbow'] != elbow):
            continue
        return trial, shade
    return None


def _sample_word(rng, length, repeats=False):
    distinct = int(rng.integers(2, min(5, length)+1))
    if repeats:
        distinct = min(distinct, length-1)
    labels = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length-distinct)))
    rng.shuffle(labels)
    # Exclude programs equivalent to repeating a shorter deduplicated cycle.
    if repeats and all(x == tuple(dict.fromkeys(labels))[i % distinct]
                       for i, x in enumerate(labels)):
        labels[0], labels[1] = labels[1], labels[0]
    return labels


def _new_canvas(rng, length, repeats, large=False):
    palette = list(map(int, rng.permutation(10)))
    bg, marker = palette[:2]
    word = tuple(palette[2+i] for i in _sample_word(rng, length, repeats))
    axis = int(rng.integers(2))
    gaps = list(map(int, rng.integers(1, 3, size=length-1)))
    span = 2*length + sum(gaps)
    shape = [int(rng.integers(23 if large else 16, 31)) for _ in (0, 1)]
    shape[1-axis] = max(shape[1-axis], span+2)
    grid = np.full(tuple(shape), bg, np.int8)
    # The aligned group may be on either side, inset, horizontal or vertical.
    fixed = int(rng.integers(0, 4))
    if rng.integers(2):
        fixed = shape[axis]-2-fixed
    advance = int(rng.integers(0, shape[1-axis]-span+1))
    annotations = set()
    for i, shade in enumerate(word):
        origin = [0, 0]
        origin[axis], origin[1-axis] = fixed, advance
        cells = {(origin[0]+r, origin[1]+c) for r in (0, 1) for c in (0, 1)}
        for p in cells:
            grid[p] = shade
        annotations |= cells
        if i < length-1:
            advance += 2+gaps[i]
    info = dict(bg=bg, word=word, marker=marker, annotations=annotations,
                legend_axis=axis, legend_fixed=fixed)
    return grid, info


def _add_marker(grid, info, rng):
    _, traced = _run(grid, info)
    forbidden = {s['cell'] for r in traced['routes'] for s in r['path']}
    options = [(r, c) for r in range(grid.shape[0]-1) for c in range(grid.shape[1]-1)
               if (r if info['legend_axis'] == 0 else c) != info['legend_fixed']]
    rng.shuffle(options)
    for r, c in options:
        cells = {(r+i, c+j) for i in (0, 1) for j in (0, 1)}
        if cells & forbidden:
            continue
        trial = _place(grid, cells, info['marker'], info['bg'])
        if trial is not None:
            decoded = _decode(trial)
            out, final = _run(trial, decoded)
            return dict(input=trial, output=out), final
    return None


def _coverage(grid, info, expected):
    evidence = set()
    for name, mode in (("multiplicity", 'deduplicate'), ("pause", 'consume'),
                       ("continuity", 'reset'), ("restart", 'global'),
                       ("conditional", 'always'), ("reflection", 'never'),
                       ("program", 'constant'), ("prospective", 'previous'),
                       ("order", 'reverse'), ("launch_phase", 'start_zero'),
                       ("continuity_one", 'reset_one')):
        try:
            alternative = _run(grid, info, mode)[0]
            differs = not np.array_equal(alternative, expected)
        except AssertionError:
            differs = True  # This alternative demands an inadmissible route.
        if differs:
            evidence.add(name)
    if any(sum(not s['contact'] for s in r['path']) >= 2*len(info['word'])
           for r in info['routes']):
        evidence.add('cycle')
    return evidence


def _make_pair(rng, kind, length=None, count=None, bounces=None):
    for attempt in range(160):
        n = length if length is not None else int(rng.integers(3, 6))
        large = kind in ('teach_chain', 'chain_test', 'count_test')
        grid, info = _new_canvas(rng, n, kind.startswith('teach'), large)
        if kind == 'teach_chain':
            plans = [(2, True)]
        elif kind == 'teach_multi':
            plans = [(1, False), (1, True)]
        elif kind == 'chain_test':
            plans = [(3, bool(rng.integers(2)))]
        elif kind == 'count_test':
            plans = [(int(rng.integers(1, 3)), i % 2 == 1) for i in range(count)]
        else:
            k = count if count is not None else int(rng.integers(1, 3))
            plans = [(int(rng.integers(0, 3)) if bounces is None else bounces,
                      bool(rng.integers(2))) for _ in range(k)]
        ok = True
        for source_index, (bounce_count, locked_end) in enumerate(plans):
            addition = _add_source(grid, info, rng, minimum=8 if kind.startswith('teach') else 5)
            if addition is None:
                ok = False
                break
            grid, elbow = addition
            lock_color = None
            for j in range(bounce_count + int(locked_end)):
                matched = j < bounce_count
                key_color = lock_color if kind == 'teach_chain' and not matched else None
                addition = _append_mirror(grid, info, rng, elbow, matched, key_color)
                if addition is None:
                    ok = False
                    break
                grid, shade = addition
                if j == 0:
                    lock_color = shade
            if not ok:
                break
        if not ok:
            continue
        finished = _add_marker(grid, info, rng)
        if finished is None:
            continue
        pair, final = finished
        if kind == 'teach_chain':
            need = REQUIRED - {'restart'}
            if not need <= _coverage(pair['input'], final, pair['output']):
                continue
        if kind == 'teach_multi' and 'restart' not in _coverage(pair['input'], final, pair['output']):
            continue
        return pair, final
    raise RuntimeError(f"bounded construction could not build {kind}")


def _key(grid):
    return grid.shape, grid.tobytes()


def _legal_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, train > test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    result = {'train': [], 'test': []}
    seen = set()
    # Select a complete pair of teaching scenes by visible distinctions.
    for _ in range(30):
        demos = [_make_pair(rng, 'teach_chain'), _make_pair(rng, 'teach_multi')]
        axes = {c['axis'] for _, i in demos for r in i['routes'] for c in r['contacts']
                if len(i['mirrors'][c['mirror']]['cells']) > 1 and c['matched']}
        directions = {s['direction'] for _, i in demos for s in i['sources']}
        if (axes == {0, 1} and len(directions) >= 2
                and {i['legend_axis'] for _, i in demos} == {0, 1}):
            result['train'] = [p for p, _ in demos]
            break
    else:
        raise RuntimeError("failed to expose both mirror axes")
    seen.update(_key(p['input']) for p in result['train'])
    for split, total in (('train', num_train), ('test', num_test)):
        while len(result[split]) < total:
            kind, length, count = 'ordinary', None, None
            if split == 'test':
                if regime == 0:
                    length = 6
                elif regime == 1:
                    kind = 'chain_test'
                else:
                    kind, count = 'count_test', int(rng.integers(3, 5))
            for _ in range(20):
                pair, _ = _make_pair(rng, kind, length, count)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("failed to construct distinct inputs")
            seen.add(_key(pair['input']))
            result[split].append(pair)
    rng.shuffle(result['train'])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _legal_counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, axes, directions, legends = set(), set(), set(), set(), set()
    matched_keys, failed_keys = set(), set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            decoded = _decode(inp)
            expected, info = _run(inp, decoded)
            infos[split].append(info)
            assert np.array_equal(out, expected), "wrong complete output"
            assert not np.array_equal(inp, out)
            preserved = {p for obj in info['sources']+info['mirrors'] for p in obj['cells']}
            assert all(out[p] == inp[p] for p in preserved)
            assert all(out[p] == info['bg'] for p in info['annotations'])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            # Reconstruct phase in a ledger rather than trusting event labels.
            for route in info['routes']:
                phase, events = 1, iter(route['contacts'])
                for step in route['path']:
                    if step['contact']:
                        event = next(events)
                        mirror = info['mirrors'][event['mirror']]
                        assert event['cell'] == step['cell'] and step['color'] == info['marker']
                        assert event['phase'] == phase % len(info['word'])
                        assert event['matched'] == (info['word'][phase % len(info['word'])] == mirror['color'])
                        assert event['matched'] or step is route['path'][-1]
                    else:
                        assert step['color'] == info['word'][phase % len(info['word'])]
                        phase += 1
                assert route['phase'] == phase % len(info['word'])
                if route['end'] == 'locked':
                    assert route['contacts'] and not route['contacts'][-1]['matched']
            if split == 'train':
                evidence |= _coverage(inp, info, out)
                directions |= {s['direction'] for s in info['sources']}
                legends.add(info['legend_axis'])
                for r in info['routes']:
                    for c in r['contacts']:
                        m = info['mirrors'][c['mirror']]
                        if c['matched'] and len(m['cells']) > 1:
                            axes.add(c['axis'])
                        (matched_keys if c['matched'] else failed_keys).add(m['color'])
    assert REQUIRED <= evidence, REQUIRED-evidence
    assert axes == {0, 1} and len(directions) >= 2
    assert legends == {0, 1}, "training exposes both legend reading orientations"
    assert matched_keys & failed_keys, "same key must both unlock and stop in training"
    train_word = max(len(i['word']) for i in infos['train'])
    train_count = max(len(i['sources']) for i in infos['train'])
    train_bounces = max(sum(c['matched'] for c in r['contacts'])
                        for i in infos['train'] for r in i['routes'])
    word_shift = all(len(i['word']) > train_word for i in infos['test'])
    count_shift = all(len(i['sources']) > train_count for i in infos['test'])
    chain_shift = all(max(sum(c['matched'] for c in r['contacts']) for r in i['routes'])
                      > train_bounces for i in infos['test'])
    assert word_shift or count_shift or chain_shift, "missing structural train/test shift"


def _pattern(word):
    mapping = {}
    return tuple(mapping.setdefault(c, len(mapping)) for c in word)


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    features, structures = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                out, info = _run(inp, _decode(inp))
                assert np.array_equal(solve(inp.tolist()), out)
                features.append(info)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, normalized))
    assert len(structures) >= 180
    assert len({p['input'].shape for q in puzzles for s in q for p in q[s]}) >= 50
    assert {i['bg'] for i in features} == set(range(10))
    assert {len(i['word']) for i in features} == {3, 4, 5, 6}
    assert len({_pattern(i['word']) for i in features}) >= 20
    assert {i['legend_axis'] for i in features} == {0, 1}
    assert {len(i['sources']) for i in features} >= {1, 2, 3, 4}
    assert {s['direction'] for i in features for s in i['sources']} == set(DIRS)
    assert {len(m['cells']) for i in features for m in i['mirrors']} == {1, 3, 5, 7}
    assert max(sum(c['matched'] for c in r['contacts']) for i in features for r in i['routes']) >= 3
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {'train': puzzles[0]['train'], 'test': [dict(puzzles[0]['test'][0])]}
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    p = tuple(np.argwhere(bad['test'][0]['input'] != bad['test'][0]['output'])[0])
    bad['test'][0]['output'][p] = bad['test'][0]['input'][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing output cell accepted")
    _literal_regression()


def _literal_regression():
    # ABCB. Source elbow at (6,2) is A; first new cell must be B.
    grid = np.zeros((15, 17), dtype=np.int8)
    for j, color in enumerate((1, 2, 3, 2)):
        grid[0:2, 1+3*j:3+3*j] = color
    grid[12:14, 14:16] = 4
    for p in _l_cells((6, 2), (1, 1)):
        grid[p] = 1
    grid[10, 4:7] = 2  # contact (9,5), due B -> reflect upward
    grid[3:6, 11] = 1  # contact (4,10), due B -> stop, not reflect
    expected = grid.copy()
    expected[0:2, 1:3] = expected[0:2, 4:6] = 0
    expected[0:2, 7:9] = expected[0:2, 10:12] = 0
    expected[12:14, 14:16] = 0
    for p, color in (((7, 3), 2), ((8, 4), 3), ((9, 5), 4),
                     ((8, 6), 2), ((7, 7), 1), ((6, 8), 2),
                     ((5, 9), 3), ((4, 10), 4)):
        expected[p] = color
    assert np.array_equal(solve(grid), expected)
    # Point mirror uses the same approached face, without an orientation tag.
    point, point_expected = grid.copy(), expected.copy()
    point[10, 4:7] = point_expected[10, 4:7] = 0
    point[10, 5] = point_expected[10, 5] = 2
    assert np.array_equal(solve(point), point_expected)
    # A longer odd bar outside the reached trajectory remains intact.
    unused, unused_expected = grid.copy(), expected.copy()
    unused[14, 1:10] = unused_expected[14, 1:10] = 2
    assert np.array_equal(solve(unused), unused_expected)
    # Straight exit: paint to the last in-bounds cell, with no terminal marker.
    edge = np.zeros_like(grid)
    for j, color in enumerate((1, 2, 3, 2)):
        edge[0:2, 1+3*j:3+3*j] = color
    edge[12:14, 14:16] = 4
    for p in _l_cells((10, 3), (1, 1)):
        edge[p] = 1
    edge_expected = np.zeros_like(edge)
    for p in _l_cells((10, 3), (1, 1)):
        edge_expected[p] = 1
    for p, color in (((11, 4), 2), ((12, 5), 3), ((13, 6), 2), ((14, 7), 1)):
        edge_expected[p] = color
    assert np.array_equal(solve(edge), edge_expected)


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
