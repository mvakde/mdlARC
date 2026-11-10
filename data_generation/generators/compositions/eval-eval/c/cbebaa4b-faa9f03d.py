"""Portable rerouting stencils: exact geometric registration transports an edit.

Fixed roles: 0 background, 8 repair/key dots, 4 direction, 9 stencil carrier.
All other colors are trace payloads. No parent generator is imported.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
PAYLOAD = (1, 2, 3, 5, 6, 7)
HOST_SEEDS = (0, 52093166, 551737303, 592766995, 833647209, 465914804,
              629743170, 358232250, 226482871, 140050641, 303461229,
              430139686, 666358625, 549541688, 968942274, 67671751,
              769547072, 321829741, 195457951, 353469004, 572067057,
              979320877, 515960298, 462579923, 855720836, 108604998,
              564985552, 890091901, 650571331, 241270832, 552966304,
              229732112, 452571153, 374356476, 584549666, 367924162,
              612925460, 470274434, 425345724, 824532355, 128588364,
              345416170)


def _add(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _edge(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [_add(a, d, i) for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _polyline(points):
    walk = [points[0]]
    for a, b in zip(points, points[1:]):
        walk.extend(_line(a, b)[1:])
    return walk


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            for d in N4:
                q = _add(stack[-1], d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
                    break
            else:
                stack.pop()
        result.append(component)
    return result


def _normalize(cells):
    r, c = min(x for x, y in cells), min(y for x, y in cells)
    return tuple(sorted((x-r, y-c) for x, y in cells))


def _shift(cells, delta):
    return {_add(p, delta) for p in cells}


def _buffer(cells):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _simple(walk):
    cells = set(walk)
    return (len(cells) == len(walk) and len(walk) >= 3
            and all(sum(_add(p, d) in cells for d in N4) ==
                    (1 if p in (walk[0], walk[-1]) else 2) for p in cells))


def _follow(cells, pivot, d):
    """Follow one arm of a simple path, excluding the pivot."""
    previous, current, walk = pivot, _add(pivot, d), []
    assert current in cells, "instruction has no opposite old arm"
    while True:
        assert current not in walk, "cyclic terminal arm"
        walk.append(current)
        following = [q for step in N4 if (q := _add(current, step)) in cells and q != previous]
        assert len(following) <= 1, "old arm is branched"
        if not following:
            return walk
        previous, current = current, following[0]


def _analyze(input_grid):
    inp = np.asarray(input_grid)
    assert inp.ndim == 2 and all(1 <= x <= 30 for x in inp.shape)
    assert np.issubdtype(inp.dtype, np.integer) and np.all((0 <= inp) & (inp <= 9))
    networks, tools = [], []
    for cells in _components(map(tuple, np.argwhere(inp != 0).tolist())):
        colors = {int(inp[p]) for p in cells}
        pins = {p for p in cells if inp[p] == 8}
        if 9 in colors:
            assert colors == {4, 8, 9} and len(pins) == 3
            arrows = [p for p in cells if inp[p] == 4]
            assert len(arrows) == 1
            arrow = arrows[0]
            pivots = [p for p in pins if sum(abs(x-y) for x, y in zip(p, arrow)) == 1]
            assert len(pivots) == 1
            pivot = pivots[0]
            direction = (arrow[0]-pivot[0], arrow[1]-pivot[1])
            vectors = [(p[0]-pivot[0], p[1]-pivot[1]) for p in sorted(pins - {pivot})]
            assert all((r == 0) != (c == 0) and 2 <= abs(r)+abs(c) <= 5 for r, c in vectors)
            assert sum(a*b for a, b in zip(*vectors)) == 0
            assert any(r*direction[0]+c*direction[1] < 0 for r, c in vectors)
            skeleton = set().union(*(set(_line(pivot, p)) for p in pins))
            assert cells == skeleton | {arrow}, 'stencil carrier must connect its complete key'
            tools.append(dict(cells=cells, pins=pins, pivot=pivot, direction=direction))
        else:
            payloads = colors - {8}
            assert len(payloads) == 1 and payloads <= set(PAYLOAD)
            assert len(pins) in (0, 3)
            ends = sorted(p for p in cells if sum(_add(p, d) in cells for d in N4) == 1)
            assert len(ends) == 2
            assert all(sum(_add(p, d) in cells for d in N4) == (1 if p in ends else 2) for p in cells)
            networks.append(dict(cells=cells, pins=pins, color=payloads.pop()))
    assert tools and networks
    sockets = {_normalize(n['pins']): i for i, n in enumerate(networks) if n['pins']}
    assert len(sockets) == sum(bool(n['pins']) for n in networks) == len(tools)
    edits, matched = [], set()
    for tool in tools:
        key = _normalize(tool['pins'])
        assert key in sockets, "stencil has no complete translation match"
        index = sockets[key]
        assert index not in matched
        matched.add(index)
        net = networks[index]
        origin = (min(r for r, c in net['pins']), min(c for r, c in net['pins']))
        source = (min(r for r, c in tool['pins']), min(c for r, c in tool['pins']))
        delta = (origin[0]-source[0], origin[1]-source[1])
        assert _shift(tool['pins'], delta) == net['pins'] and delta != (0, 0)
        pivot = _add(tool['pivot'], delta)
        d = tool['direction']
        assert pivot in net['pins'] and _add(pivot, d) not in net['cells']
        assert _shift(tool['cells'] - {_add(tool['pivot'], d)}, delta) <= net['cells']
        assert any(_add(pivot, t) in net['cells'] for t in ((-d[1], d[0]), (d[1], -d[0])))
        old = _follow(net['cells'], pivot, (-d[0], -d[1]))
        assert _edge(old[-1], inp.shape), "old arm must terminate at the boundary"
        edits.append(dict(index=index, pivot=pivot, direction=d, old=set(old),
                          walk=[pivot]+old, new=set(_ray(pivot, d, inp.shape)), tool=tool,
                          delta=delta))
    for a, b in combinations(edits, 2):
        assert (networks[a['index']]['color'] == networks[b['index']]['color']
                or not a['new'] & b['new']), "different-colored new rays have no tie convention"
    return inp, networks, edits


def _render(shape, networks, edits, *, repair_only=False, reverse=False):
    out = np.zeros(shape, dtype=np.int8)
    removals = {e['index']: e['old'] for e in edits} if not repair_only else {}
    for i, net in enumerate(networks):
        for p in net['cells'] - removals.get(i, set()):
            out[p] = net['color']
    if not repair_only:
        for edit in (list(reversed(edits)) if reverse else edits):
            for p in edit['new']:
                out[p] = networks[edit['index']]['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer registration and all edits exclusively from this input."""
    inp, networks, edits = _analyze(input_grid)
    return _render(inp.shape, networks, edits)


def _turns(walk):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _arm(rng, p, direction, first_min, bends, shape, boundary):
    """Random unequal orthogonal segments; no background gap interpolation."""
    points, d = [p], direction
    for index in range(bends+1):
        ray = _ray(points[-1], d, shape)
        minimum = first_min if index == 0 else 3
        if len(ray) < minimum:
            return None
        if index == bends and boundary:
            endpoint = ray[-1]
        else:
            maximum = min(len(ray)-1, minimum + 5)
            if maximum < minimum:
                return None
            endpoint = ray[int(rng.integers(minimum, maximum+1))-1]
        points.append(endpoint)
        d = (-d[1], d[0]) if rng.random() < .5 else (d[1], -d[0])
    walk = _polyline(points)
    return walk if _simple(walk) else None


def _network(pivot, old, kept, a, b, color):
    cells = set(old) | set(kept)
    assert old[0] == kept[0] == pivot
    assert _simple(list(reversed(old)) + kept[1:])
    d = (pivot[0]-old[1][0], pivot[1]-old[1][1])
    return dict(cells=cells, pins={pivot, old[a], kept[b]}, color=color,
                pivot=pivot, direction=d, old=set(old[1:]), walk=old,
                kept=set(kept), new=None, a=a, b=b)


def _random_network(rng, shape, occupied, keys, color, bends_range):
    for _ in range(220):
        pivot = (int(rng.integers(4, shape[0]-4)), int(rng.integers(4, shape[1]-4)))
        direction = N4[int(rng.integers(4))]
        retained = (-direction[1], direction[0])
        if rng.random() < .5:
            retained = (-retained[0], -retained[1])
        a, b = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        old = _arm(rng, pivot, (-direction[0], -direction[1]), a,
                   int(rng.integers(*bends_range)), shape, True)
        kept = _arm(rng, pivot, retained, b, int(rng.integers(3)), shape, False)
        if old is None or kept is None:
            continue
        if not _simple(list(reversed(old)) + kept[1:]):
            continue
        net = _network(pivot, old, kept, a, b, color)
        if net['cells'] & _buffer(occupied) or _normalize(net['pins']) in keys:
            continue
        net['new'] = set(_ray(pivot, direction, shape))
        return net
    return None


def _witness(rng, shape, colors, bent):
    """Two nested, uneven elbows: equal-area keys, distinct instructions."""
    h, w = shape
    r1, c1 = int(rng.integers(5, 9)), int(rng.integers(5, 9))
    r2 = int(rng.integers(r1+6, h-4))
    c2 = int(rng.integers(c1+8, w-3))
    pivots = ((r1, c1), (r2, c2))
    equal_area_spans = [(2, 4), (3, 3), (4, 2)]
    span_order = list(map(int, rng.permutation(3)))
    keys = set()
    nets = []
    for i, p in enumerate(pivots):
        # Distinct spans, not distinct cardinalities. Both choices of arm to
        # delete occur in the first demonstration; the second teaches a bend.
        for _ in range(64):
            a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            if not bent:
                horizontal, vertical = equal_area_spans[span_order[i]]
                a, b = (vertical, horizontal) if i == 1 else (horizontal, vertical)
            up, left = _line(p, (0, p[1])), _line(p, (p[0], 0))
            if i == 1 and not bent:
                old, kept = up, left
            else:
                old, kept = left, up
            if i == 1 and bent:
                turn = (p[0], p[1]-int(rng.integers(4, p[1]-c1-1)))
                old = _polyline([p, turn, (h-1, turn[1])])
            net = _network(p, old, kept, a, b, colors[i])
            key = _normalize(net['pins'])
            if key not in keys:
                break
        else:
            raise RuntimeError('bounded witness key selection exhausted')
        keys.add(key)
        net['new'] = set(_ray(p, net['direction'], shape))
        nets.append(net)
    return nets


def _tool(net):
    pivot = net['pivot']
    mapping = {}
    for endpoint in sorted(net['pins'] - {pivot}):
        for p in _line(pivot, endpoint):
            mapping[p] = 9
    for p in net['pins']:
        mapping[p] = 8
    mapping[_add(pivot, net['direction'])] = 4
    return mapping


def _scatter(rng, shape, networks, require_wrong_nearest=False):
    base = np.zeros(shape, dtype=np.int8)
    occupied = set()
    for net in networks:
        for p in net['cells']:
            base[p] = 8 if p in net['pins'] else net['color']
        occupied.update(net['cells'])
    tools = [_tool(n) for n in networks if n['pins']]
    for _ in range(14):
        inp, forbidden, wrong_nearest = base.copy(), _buffer(occupied), False
        order = list(map(int, rng.permutation(len(tools))))
        for index in order:
            mapping = tools[index]
            top = min(r for r, c in mapping)
            left = min(c for r, c in mapping)
            height = max(r for r, c in mapping)-top+1
            width = max(c for r, c in mapping)-left+1
            candidates = []
            for r in range(shape[0]-height+1):
                for c in range(shape[1]-width+1):
                    d = (r-top, c-left)
                    if d != (0, 0) and not _shift(mapping, d) & forbidden:
                        candidates.append(d)
            if not candidates:
                break
            d = candidates[int(rng.integers(len(candidates)))]
            moved_pivot = _add(networks[index]['pivot'], d)
            distances = [abs(moved_pivot[0]-n['pivot'][0])+abs(moved_pivot[1]-n['pivot'][1])
                         for n in networks if n['pins']]
            wrong_nearest |= distances[index] > min(distances)
            for p, color in mapping.items():
                inp[_add(p, d)] = color
            forbidden |= _buffer(_shift(mapping, d))
        else:
            if not require_wrong_nearest or wrong_nearest:
                return inp
    return None


def _ground_truth(shape, networks):
    """Construction rendering uses sampled retained walks, not solver traversal."""
    out = np.zeros(shape, dtype=np.int8)
    for net in networks:
        for p in net['kept'] if net['pins'] else net['cells']:
            out[p] = net['color']
    for net in networks:
        if net['pins']:
            for p in net['new']:
                out[p] = net['color']
    return out


def _transform(pair, k, reflect):
    return {role: np.rot90(grid[:, ::-1] if reflect else grid, k).copy()
            for role, grid in pair.items()}


def _make_pair(rng, palette, mode, index):
    for _ in range(64):
        shape = (int(rng.integers(23, 31)), int(rng.integers(24, 31)))
        colors = list(map(int, rng.choice(palette, 3, replace=True)))
        if mode in ('binding', 'bent'):
            # An explicit different-color crossing teaches new-ray precedence.
            colors[1] = next(c for c in palette if c != colors[0])
            nets = _witness(rng, shape, colors, mode == 'bent')
        else:
            count = int(rng.integers(1, 4))
            nets, occupied, keys = [], set(), set()
            for j in range(count):
                bend_range = ((2, 4) if j == 0 else (1, 4)) if mode == 'query' else (0, 3)
                net = _random_network(rng, shape, occupied, keys, colors[j], bend_range)
                if net is None:
                    break
                nets.append(net)
                occupied |= net['cells']
                keys.add(_normalize(net['pins']))
            if len(nets) != count:
                continue
            # Optional unchanged component: it has no keys and no instruction.
            if rng.random() < .45:
                for _extra in range(40):
                    p = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
                    d = N4[int(rng.integers(4))]
                    cells = set(_ray(p, d, shape)[:int(rng.integers(3, 9))]) | {p}
                    if len(cells) < 4 or cells & _buffer(occupied):
                        continue
                    nets.append(dict(cells=cells, kept=cells, pins=set(), color=int(rng.choice(palette))))
                    break
        if any(a['color'] != b['color'] and a['new'] & b['new']
               for a, b in combinations([n for n in nets if n['pins']], 2)):
            continue
        inp = _scatter(rng, shape, nets, require_wrong_nearest=mode == 'binding')
        if inp is None:
            continue
        pair = dict(input=inp, output=_ground_truth(shape, nets))
        pair = _transform(pair, int(rng.integers(4)), bool(rng.integers(2)))
        try:
            predicted = solve(pair['input'])
            assert np.array_equal(predicted, pair['output'])
        except AssertionError:
            continue
        return pair
    raise RuntimeError(f'bounded scene construction exhausted for {mode}/{index}')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(PAYLOAD, int(rng.integers(2, 7)), replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            mode = ('binding' if index == 0 else 'bent' if index == 1 else 'practice') if split == 'train' else 'query'
            for _ in range(12):
                pair = _make_pair(rng, palette, mode, index)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    # Randomize witness order without weakening minimum two-example teaching.
    rng.shuffle(puzzle['train'])
    return puzzle


def _wrong_binding(inp, networks, edits):
    """Counterfactual: align two instruction pivots, ignoring their dot masks."""
    if len(edits) < 2:
        return None
    wrong = [dict(e) for e in edits]
    for a, b in ((0, 1), (1, 0)):
        e = wrong[a]
        d = edits[b]['direction']
        if _add(e['pivot'], d) in networks[e['index']]['cells']:
            return None
        try:
            old = _follow(networks[e['index']]['cells'], e['pivot'], (-d[0], -d[1]))
        except AssertionError:
            return None
        if not _edge(old[-1], inp.shape):
            return None
        e.update(direction=d, old=set(old), new=set(_ray(e['pivot'], d, inp.shape)))
    return _render(inp.shape, networks, wrong)


def _features(pair):
    inp, nets, edits = _analyze(pair['input'])
    kept = {i: n['cells'] - next((e['old'] for e in edits if e['index'] == i), set())
            for i, n in enumerate(nets)}
    return dict(shape=inp.shape, switches=len(edits), paths=len(nets),
                keys=[_normalize(n['pins']) for n in nets if n['pins']],
                directions=[e['direction'] for e in edits],
                old_bends=[_turns(e['walk']) for e in edits],
                old_lengths=[len(e['old']) for e in edits],
                ray_lengths=[len(e['new']) for e in edits],
                payloads=tuple(sorted({n['color'] for n in nets})),
                repeated_colors=len({n['color'] for n in nets}) < len(nets),
                unchanged=sum(not n['pins'] for n in nets),
                retained_hits=sum(bool(e['new'] & kept[i]) for e in edits for i, n in enumerate(nets)
                                  if n['color'] != nets[e['index']]['color']),
                old_hits=sum(bool(a['new'] & b['old']) for a, b in combinations(edits, 2))
                         + sum(bool(b['new'] & a['old']) for a, b in combinations(edits, 2)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, binding_witness, bent_witness, crossing_witness = set(), False, False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            _, nets, edits = _analyze(inp)
            expected = _render(inp.shape, nets, edits)
            assert np.array_equal(expected, out), 'full output violates registered rerouting'
            assert np.array_equal(expected, _render(inp.shape, nets, edits, reverse=True))
            assert not np.any(np.isin(out, (4, 8, 9)))
            assert not np.array_equal(out, _render(inp.shape, nets, edits, repair_only=True)), 'B concept has no effect'
            assert all(out[e['pivot']] in PAYLOAD for e in edits), 'pivot was not repaired'
            if split == 'train':
                wrong = _wrong_binding(inp, nets, edits)
                binding_witness |= wrong is not None and not np.array_equal(wrong, expected)
                bent_witness |= any(_turns(e['walk']) for e in edits)
                crossing_witness |= _features(pair)['retained_hits'] > 0
    assert binding_witness, 'training must distinguish geometric binding from exchanging instructions'
    assert bent_witness, 'training must show deletion through a bend'
    assert crossing_witness, 'training must teach new-ray-over-retained precedence'


def _signature(puzzle):
    return tuple((pair['input'].shape, pair['input'].tobytes(), pair['output'].tobytes())
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """42 seeds, streaming with one reference plus the current puzzle; no files."""
    signatures, structures, shapes, keys, directions, bends, counts, palettes = (set() for _ in range(8))
    repeated = unchanged = crossings = pairs = 0
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest()
        signatures.add(digest)
        if seed == 0:
            first = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                f = _features(pair)
                shapes.add(f['shape'])
                keys.update(f['keys'])
                directions.update(f['directions'])
                bends.update(f['old_bends'])
                counts.add(f['switches'])
                palettes.add(f['payloads'])
                repeated += f['repeated_colors']
                unchanged += f['unchanged']
                crossings += f['retained_hits']
                # Normalize colors in appearance order, retaining every cell.
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                structures.add((pair['input'].shape, hashlib.sha256(normalized).hexdigest()))
                moved = _transform(pair, 1, True)
                assert np.array_equal(solve(moved['input']), moved['output'])
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(shapes) >= 30 and len(keys) >= 40 and directions == set(N4)
    assert {0, 1, 2} <= bends and counts == {1, 2, 3}
    assert len(palettes) >= 12 and repeated > 0 and unchanged > 0 and crossings >= 42
    assert _signature(first) == _signature(generate_puzzle(0))
    # Counts, including boundary combinations; host additionally checks all legal counts.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(177+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Full-grid negatives cover vacated tools, preserved cells, repair, deletion and rays.
    q = first['test'][0]
    inp, nets, edits = _analyze(q['input'])
    sites = [min(edits[0]['tool']['cells']), edits[0]['pivot'], min(edits[0]['old']),
             min(edits[0]['new'])]
    sites += [tuple(map(int, p)) for p in np.argwhere(inp == q['output'])[:1]]
    for p in sites:
        corrupted = {s: [dict(pair) for pair in first[s]] for s in ('train', 'test')}
        corrupted['test'][0]['output'] = q['output'].copy()
        corrupted['test'][0]['output'][p] = (int(q['output'][p])+1) % 10
        try:
            validate_generated_puzzle(corrupted)
        except AssertionError:
            pass
        else:
            raise AssertionError('mutated output accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                color_normalized_inputs=len(structures), shapes=len(shapes),
                key_geometries=len(keys), switch_counts=sorted(counts),
                old_bends=sorted(bends), repeated_color_pairs=repeated,
                unchanged_paths=unchanged, retained_crossings=crossings)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
