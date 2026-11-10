"""Routed stencil fitting: paint all legal anchor positions inside chambers.

Only standard Python and NumPy are used. Generation constructs labels from
placed sets; solve reconstructs the dictionary, cavities and route graph from
the input. No generation metadata, examples or seeds enter solve.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D4):
    left = set(cells)
    result = []
    while left:
        root = min(left)
        left.remove(root)
        group, queue = {root}, [root]
        for r, c in queue:
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _neighbors(cells, steps=D8):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _move(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _normalize(cells):
    return _move(cells, -min(r for r, c in cells), -min(c for r, c in cells))


def _shape_key(cells):
    return tuple(sorted(_normalize(cells)))


def _extent(cells):
    return max(r for r, c in cells) - min(r for r, c in cells) + 1, max(c for r, c in cells) - min(c for r, c in cells) + 1


def _no_holes(cells):
    cells = _normalize(cells)
    h, w = _extent(cells)
    empty = {(r, c) for r in range(-1, h + 1) for c in range(-1, w + 1)} - cells
    return len(_components(empty)) == 1


def _fits(cavity, offsets):
    """All translations, retaining the reference's distinguished anchor."""
    return {p for p in cavity if all((p[0] + dr, p[1] + dc) in cavity for dr, dc in offsets)}


def _box_offsets(offsets):
    return {(r, c) for r in range(min(r for r, c in offsets), max(r for r, c in offsets) + 1)
            for c in range(min(c for r, c in offsets), max(c for r, c in offsets) + 1)}


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grid must be 2D with dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _interpret(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    assert np.all(border == border[0]), "one-cell clear background border required"
    bg = int(border[0])
    regions = _components(zip(*np.where(a == bg)))
    cavities = [g for g in regions if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in g)]
    assert cavities, "no chambers"
    shells = [_neighbors(g, D4) for g in cavities]
    inks = {int(a[p]) for shell in shells for p in shell}
    assert len(inks) == 1 and bg not in inks, "uniform chamber ink required"
    ink = inks.pop()
    all_shell = set().union(*shells)
    foreground = set(zip(*np.where(a != bg)))
    # Complete reference objects are the detached foreground components that
    # contain no chamber wall. Even touching chamber walls are NOT route edges.
    refs = [g for g in _components(foreground, D8) if not g & all_shell]
    assert refs, "missing stencil exemplars"
    templates, reference_cells, repetitions = {}, set(), {}
    for g in refs:
        anchors = [p for p in g if int(a[p]) != ink]
        assert len(anchors) == 1 and len(g) >= 3, "one key plus at least two ink cells per stencil"
        ar, ac = anchors[0]
        key = int(a[ar, ac])
        offsets = frozenset((r - ar, c - ac) for r, c in g)
        assert len(_components(g)) == 1 and _no_holes(g), "stencils are connected and hole-free"
        assert key not in templates or templates[key] == offsets, "conflicting stencil definitions"
        templates[key] = offsets
        repetitions[key] = repetitions.get(key, 0) + 1
        reference_cells.update(g)
    keys = set(templates)
    assert not keys & {ink, bg}
    wire_cells = {p for p in foreground - reference_cells if int(a[p]) not in keys | {ink}}
    assert wire_cells and len({int(a[p]) for p in wire_cells}) == 1, "one visible wire color per input"
    wires = _components(wire_cells, D8)
    wire_color = int(a[next(iter(wire_cells))])
    assert {p for p in foreground if int(a[p]) == ink} <= all_shell | reference_cells, "unexplained ink"
    sources = [(p, int(a[p])) for p in sorted(foreground - reference_cells) if int(a[p]) in keys]
    assert sources, "no keyed wire endpoints"
    n = len(cavities)
    graph = [set() for _ in range(n + len(wires))]
    for i, shell in enumerate(shells):
        adjacent = _neighbors(shell)
        for j, wire in enumerate(wires, n):
            if adjacent & wire:
                graph[i].add(j)
                graph[j].add(i)
    node_sources = {}
    for p, key in sources:
        adjacent = _neighbors({p}) & foreground
        assert len(adjacent) == 1 and adjacent <= wire_cells, "source must be a free wire tip"
        starts = [i for i, wire in enumerate(wires, n) if wire & adjacent]
        assert len(starts) == 1
        node_sources.setdefault(starts[0], []).append((p, key))
    ownership = {}
    for group in _components_graph(graph):
        labels = [s for i in group for s in node_sources.get(i, [])]
        assert len(labels) <= 1, "conflicting or duplicate sources on one route system"
        if labels:
            assert group & set(range(n)), "source-only routes are outside this family"
            for i in group & set(range(n)):
                ownership[i] = labels[0][1]
    fits = {i: _fits(cavities[i], templates[key]) for i, key in ownership.items()}
    out = a.copy()
    for i, positions in fits.items():
        for p in positions:
            out[p] = ownership[i]
    return out, dict(background=bg, ink=ink, wire_color=wire_color, templates=templates,
                     repetitions=repetitions, reference_cells=reference_cells, cavities=cavities,
                     shells=shells, wires=wires, sources=sources, ownership=ownership, fits=fits)


def _components_graph(graph):
    left = set(range(len(graph)))
    groups = []
    while left:
        root = min(left)
        left.remove(root)
        group, queue = {root}, [root]
        for i in queue:
            for j in sorted(graph[i] & left):
                left.remove(j)
                group.add(j)
                queue.append(j)
        groups.append(group)
    return groups


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every convention from this input; leave the caller's array alone."""
    return _interpret(input_grid)[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _stencil(rng, small=False):
    for _ in range(80):
        h, w = _ri(rng, 2, 3 if small else 4), _ri(rng, 2, 3 if small else 4)
        count = _ri(rng, 3, min(6 if small else 9, h * w - 1))
        cells = {(_ri(rng, 0, h - 1), _ri(rng, 0, w - 1))}
        while len(cells) < count:
            frontier = sorted(p for p in _neighbors(cells, D4) if 0 <= p[0] < h and 0 <= p[1] < w)
            cells.add(_pick(rng, frontier))
        if not _no_holes(cells) or len(cells) == np.prod(_extent(cells)):
            continue
        # Extremal anchors are common, but ordinary interior anchors also occur.
        anchor = min(cells) if small or rng.random() < .65 else _pick(rng, sorted(cells))
        return frozenset(_move(cells, -anchor[0], -anchor[1]))
    return frozenset(((0, 0), (1, 0), (1, 1)))


def _room(rng, offsets, exact=False, multiple=False):
    shape = _normalize(offsets)
    if exact:
        inner = shape
    else:
        for _ in range(30):
            # A connected path of translations creates more than one legal
            # anchor. Irregular frontier growth adds unusable spurs and notches.
            anchors, p = {(0, 0)}, (0, 0)
            for _ in range(_ri(rng, 1, 4)):
                dr, dc = _pick(rng, D4)
                p = p[0] + dr, p[1] + dc
                anchors.add(p)
            inner = set().union(*(_move(shape, *p) for p in anchors))
            for _ in range(_ri(rng, 0, 5)):
                inner.add(_pick(rng, sorted(_neighbors(inner, D4))))
            if max(_extent(inner)) <= 7 and _no_holes(inner) and (not multiple or len(_fits(inner, offsets)) > 1):
                break
        else:
            inner = shape | _move(shape, 0, 1)
    shell = _neighbors(inner, D4)
    r0, c0 = min(r for r, c in inner | shell), min(c for r, c in inner | shell)
    return _move(inner, -r0, -c0), _move(shell, -r0, -c0)


def _route(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    steps = [D8[i] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in steps:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _scene(rng, lesson, is_test, test_types):
    ntypes = 1 if lesson == 0 else 2 if lesson == 1 else test_types if is_test else _ri(rng, 1, 2)
    palette = [int(c) for c in rng.permutation(10)]
    bg, ink, wire_color = palette[:3]
    keys = palette[3:3 + ntypes]
    templates = []
    for i in range(ntypes):
        for _ in range(40):
            s = _stencil(rng, small=lesson == 0)
            if s not in templates:
                break
        else:
            return None
        templates.append(s)
    if lesson == 0:
        owners = [0, 0, None]
    elif lesson == 1:
        owners = [0, 1, None]
    else:
        owners = list(range(ntypes)) + [_ri(rng, 0, ntypes - 1) for _ in range(_ri(rng, 1, 2))]
        owners += [None] * _ri(rng, 0 if not is_test else 1, 1)
    count = len(owners)
    low = 22 if count <= 3 else 26
    h, w = _ri(rng, low, 30), _ri(rng, low, 30)
    canvas = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    rooms, occupied = [], set()
    touching = lesson == 0 or (None in owners and rng.random() < .35)
    for i, owner in enumerate(owners):
        stencil = templates[0 if owner is None else owner]
        inner, shell = _room(rng, stencil, exact=lesson == 0 and i in (0, 2), multiple=lesson == 0 and i == 1)
        rh, rw = _extent(inner | shell)
        placed = False
        for _ in range(180):
            rr, cc = _ri(rng, 1, h - rh - 1), _ri(rng, 1, w - rw - 1)
            is_touch = touching and owner is None
            if is_touch:
                p = _pick(rng, sorted(rooms[0]['shell']))
                q = _pick(rng, sorted(shell))
                dr, dc = _pick(rng, D8)
                rr, cc = p[0] + dr - q[0], p[1] + dc - q[1]
            ins, edge = _move(inner, rr, cc), _move(shell, rr, cc)
            cells = ins | edge
            if not cells <= canvas or cells & occupied:
                continue
            near = _neighbors(cells) & occupied
            if is_touch:
                if not near or not near <= rooms[0]['shell'] or not (_neighbors(edge) & rooms[0]['shell']):
                    continue
                if _neighbors(ins, D4) & rooms[0]['inner']:
                    continue
            elif near:
                continue
            rooms.append(dict(inner=ins, shell=edge, owner=owner))
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    # Detached complete motifs are a distributed legend, not a second puzzle
    # panel. Their single key cell distinguishes the anchor from stencil ink.
    references = []
    for k in range(ntypes):
        for _ in range(2 if lesson == 0 else 1):
            shape = _normalize(templates[k])
            sh, sw = _extent(shape)
            anchor_shift = -min(r for r, c in templates[k]), -min(c for r, c in templates[k])
            for _ in range(160):
                rr, cc = _ri(rng, 1, h - sh - 1), _ri(rng, 1, w - sw - 1)
                cells = _move(shape, rr, cc)
                if not (cells | _neighbors(cells)) & occupied:
                    anchor = rr + anchor_shift[0], cc + anchor_shift[1]
                    references.append((cells, anchor, k))
                    occupied.update(cells)
                    break
            else:
                return None
    sources, wire_sets = [], []
    for k in range(ntypes):
        for _ in range(180):
            p = _ri(rng, 2, h - 3), _ri(rng, 2, w - 3)
            dr, dc = _pick(rng, D8)
            q = p[0] + dr, p[1] + dc
            cells = {p, q}
            if not (cells | _neighbors(cells)) & occupied:
                sources.append(p)
                wire_sets.append({q})
                occupied.update(cells)
                break
        else:
            return None
    all_rooms = set().union(*(q['inner'] | q['shell'] for q in rooms))
    ref_cells = set().union(*(q[0] for q in references))
    source_block = set(sources) | _neighbors(set(sources))
    for k in rng.permutation(ntypes):
        k = int(k)
        own = wire_sets[k]
        other_wires = set().union(*(s for i, s in enumerate(wire_sets) if i != k))
        targets = [i for i, q in enumerate(rooms) if q['owner'] == k]
        rng.shuffle(targets)
        for i in targets:
            q = rooms[i]
            other_rooms = all_rooms - (q['inner'] | q['shell'])
            blocked = all_rooms | _neighbors(other_rooms) | ref_cells | _neighbors(ref_cells)
            blocked |= source_block | other_wires | _neighbors(other_wires)
            allowed = canvas - blocked
            goals = (_neighbors(q['shell']) & allowed)
            path = _route(rng, own, goals, allowed | own)
            if path is None:
                return None
            own.update(path)
    a = np.full((h, w), bg, dtype=np.int8)
    for q in rooms:
        for p in q['shell']:
            a[p] = ink
    for cells, anchor, k in references:
        for p in cells:
            a[p] = ink
        a[anchor] = keys[k]
    for k, cells in enumerate(wire_sets):
        for p in cells:
            a[p] = wire_color
        a[sources[k]] = keys[k]
    # Construction-owned label: use the intended cavity sets, not parsing.
    b = a.copy()
    for q in rooms:
        if q['owner'] is not None:
            k = q['owner']
            for p in _fits(q['inner'], templates[k]):
                b[p] = keys[k]
    try:
        pred, info = _interpret(a)
        assert {frozenset(g) for g in info['cavities']} == {frozenset(q['inner']) for q in rooms}
        assert len(info['sources']) == ntypes
        assert len(info['ownership']) == sum(q['owner'] is not None for q in rooms)
        assert np.array_equal(pred, b) and not np.array_equal(a, b)
        assert all(info['fits'].values())
    except (AssertionError, ValueError):
        return None
    # Whole-scene transforms change both reference orientation and chamber
    # geometry together; individual stamps are never rotated during fitting.
    turns = _ri(rng, 0, 3)
    a, b = np.rot90(a, turns), np.rot90(b, turns)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    if rng.random() < .6:
        rr, cc = np.where(a != bg)
        r0, r1 = max(0, int(rr.min()) - _ri(rng, 1, 3)), min(a.shape[0], int(rr.max()) + _ri(rng, 2, 4))
        c0, c1 = max(0, int(cc.min()) - _ri(rng, 1, 3)), min(a.shape[1], int(cc.max()) + _ri(rng, 2, 4))
        a, b = a[r0:r1, c0:c1], b[r0:r1, c0:c1]
    if lesson == 1 and not _features(a)['selection_difference']:
        return None
    return {'input': a.copy(), 'output': b.copy()}


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError("counts must be integers: train 2..10 and test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    test_types = _ri(rng, 2, 3)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else -1
            for _ in range(100):
                pair = _scene(rng, lesson, split == 'test', test_types)
                if pair is None:
                    continue
                a = pair['input']
                token = (a.shape, a.tobytes())
                if token not in seen:
                    seen.add(token)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    _, q = _interpret(a)
    cavities, templates, owners = q['cavities'], q['templates'], q['ownership']
    fitting = q['fits']
    touching = sum(bool(_neighbors(q['shells'][i]) & q['shells'][j])
                   for i in range(len(cavities)) for j in range(i)
                   if (i in owners) != (j in owners))
    bbox_difference = any(fitting[i] != _fits(cavities[i], _box_offsets(templates[k])) for i, k in owners.items())
    rotation_difference = any(fitting[i] != _fits(cavities[i], {(-c, r) for r, c in templates[k]}) for i, k in owners.items())
    selection_difference = any(fitting[i] != _fits(cavities[i], s) for i, k in owners.items()
                               for other, s in templates.items() if other != k)
    return dict(shape=tuple(np.asarray(a).shape), background=q['background'], ink=q['ink'], wire=q['wire_color'],
                types=len(templates), chambers=len(cavities), supplied=len(owners), unsupplied=len(cavities)-len(owners),
                fits=[len(fitting[i]) for i in sorted(fitting)],
                multiroom=max(sum(k == x for k in owners.values()) for x in templates),
                touching_unsupplied=touching, bbox_difference=bbox_difference, rotation_difference=rotation_difference,
                selection_difference=selection_difference,
                geometries=tuple(sorted(tuple(sorted(s)) for s in templates.values())),
                chamber_shapes=tuple(sorted(_shape_key(g) for g in cavities)),
                stencil_sizes=tuple(sorted(len(s) for s in templates.values())),
                reference_counts=tuple(sorted(q['repetitions'].values())))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            token = (a.shape, a.tobytes())
            assert token not in seen, "duplicate input"
            seen.add(token)
            predicted, q = _interpret(a)
            assert np.array_equal(predicted, b), "incorrect legal-anchor map"
            assert np.any(a != b) and np.all(a[a != b] == q['background'])
            assert all(q['fits'].values()), "generated supplied chambers must admit a fit"
            for i, key in q['ownership'].items():
                assert len(q['fits'][i]) < len(q['cavities'][i]), "erosion must differ from full filling"
                assert all(all((p[0]+dr, p[1]+dc) in q['cavities'][i] for dr, dc in q['templates'][key]) for p in q['fits'][i])
            for i, cavity in enumerate(q['cavities']):
                if i not in q['ownership']:
                    assert all(b[p] == q['background'] for p in cavity)
            features[split].append(_features(a))
    first, second = features['train'][:2]
    assert first['types'] == 1 and first['multiroom'] >= 2
    assert first['touching_unsupplied'] and first['unsupplied'] and first['bbox_difference']
    assert 1 in first['fits'] and max(first['fits']) > 1 and first['reference_counts'] == (2,)
    assert second['types'] == 2 and second['selection_difference'] and second['unsupplied']
    assert any(f['rotation_difference'] for f in features['train'])
    assert all(f['types'] >= 2 and f['multiroom'] >= 2 and f['unsupplied'] for f in features['test'])


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming seeds; retain only hashes/structural signatures, no files."""
    hashes, shapes, geometries, chambers, backgrounds, sizes, counts = set(), set(), set(), set(), set(), set(), set()
    bbox = directional = selection = contact = 0
    first_puzzle = None
    for seed in range(42):
        p = generate_puzzle(seed)
        if seed == 0:
            first_puzzle = p
        hashes.add(_puzzle_hash(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), "solver mutated input"
                f = _features(before)
                shapes.add(f['shape'])
                geometries.update(f['geometries'])
                chambers.update(f['chamber_shapes'])
                backgrounds.add(f['background'])
                sizes.update(f['stencil_sizes'])
                counts.add(f['chambers'])
                bbox += f['bbox_difference']
                directional += f['rotation_difference']
                selection += f['selection_difference']
                contact += bool(f['touching_unsupplied'])
    assert len(hashes) == 42 and len(shapes) >= 25
    assert len(geometries) >= 50 and len(chambers) >= 100
    assert len(backgrounds) == 10 and {3, 4, 5, 6, 7, 8, 9} <= sizes
    assert {3, 4, 5} <= counts and min(bbox, directional, selection, contact) >= 42
    assert _puzzle_hash(first_puzzle) == _puzzle_hash(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 100 * nt + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.0, 1), (4, None), ('4', 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # All D4 rigid transformations and arbitrary palette permutations preserve
    # the input-visible rule. These checks also exercise zero in key/ink roles.
    demo = first_puzzle['train'][0]
    for turns in range(4):
        for flip in (False, True):
            a, b = np.rot90(demo['input'], turns), np.rot90(demo['output'], turns)
            if flip:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a.tolist()), b)
    permutation = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
    assert np.array_equal(solve(permutation[demo['input']]), permutation[demo['output']])
    # The general rule also has a well-defined empty fit map. This small valid
    # scene routes a four-cell stencil to a one-cell chamber, so nothing fits.
    tiny = np.zeros((11, 13), dtype=np.int8)
    tiny[2, 2] = 2
    for p in ((3, 2), (4, 2), (4, 3), (7, 9), (8, 8), (8, 10), (9, 9)):
        tiny[p] = 8
    tiny[8, 3] = 2
    tiny[8, 4:8] = 5
    assert np.array_equal(solve(tiny), tiny)
    assert list(_interpret(tiny)[1]['fits'].values()) == [set()]
    broken = {s: [{k: a.copy() for k, a in q.items()} for q in first_puzzle[s]] for s in first_puzzle}
    pair = broken['test'][0]
    r, c = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][r, c] = pair['input'][r, c]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing valid anchor accepted")
    return dict(seeds=42, distinct_puzzles=len(hashes), grid_shapes=len(shapes),
                anchor_relative_geometries=len(geometries), chamber_shapes=len(chambers),
                backgrounds=len(backgrounds), stencil_sizes=sorted(sizes), chamber_counts=sorted(counts),
                bbox_counterexamples=bbox, orientation_counterexamples=directional,
                stencil_selection_counterexamples=selection, touching_unsupplied_pairs=contact,
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
