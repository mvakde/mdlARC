"""Ordered overflow cups: a color itinerary governs conserved-volume allocation.

Gravity acts within each open cup. The itinerary is an abstract directed relay:
conduits transfer overflow and drain completely; they are never painted. No
attempt is made to simulate unrestricted fluid motion through outside space.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import permutations

import numpy as np


HOST_SEEDS = (0, 974737333, 78822632, 679523769, 382530767, 863739714,
              383275649, 756598434, 152070723, 232498276, 342390513,
              165537549, 328429237, 580956465, 25728783, 239853150,
              146611737, 475329294, 712382022, 109609572, 166746531,
              780979790, 914980131, 595775257, 672818987, 857278838,
              878820426, 606146977, 750191191, 979220489, 78995552,
              441158077, 835188439, 58810848, 509621663, 402713290,
              262327011, 19055320, 417010866, 190832318, 366619737,
              568170159)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(x):
    a = np.asarray(x)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, part = [p], []
        while todo:
            r, c = todo.pop()
            part.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(part)
    return result


def _graph(cups):
    """Nearest visible nameplates on each axis, with no upward transfers."""
    centers = [(c['bottom'], c['left'] + c['width']//2) for c in cups]
    graph = [[] for _ in cups]
    for i, (r, x) in enumerate(centers):
        for j, (s, y) in enumerate(centers):
            if i == j or s < r:
                continue
            if s == r:
                lo, hi = sorted((x, y))
                clear = not any(t == r and lo < z < hi for t, z in centers)
            elif x == y:
                clear = not any(z == x and r < t < s for t, z in centers)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _routes(graph, labels, key):
    answers = []
    def walk(path):
        if len(answers) == 2:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        options = range(len(labels)) if not path else graph[path[-1]]
        for j in options:
            if j not in path and labels[j] == key[len(path)]:
                walk(path + [j])
    walk([])
    return answers


def _parse(a):
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("the outside frame must be background")
    interpretations = []
    for wall in map(int, np.unique(a)):
        if wall == bg:
            continue
        cups, accounted = [], np.zeros(a.shape, dtype=bool)
        valid = True
        for part in _components(a == wall):
            ys, xs = zip(*part)
            top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
            h, w = bottom-top+1, right-left+1
            if not (4 <= h <= 6 and w in (5, 7)):
                valid = False
                break
            block = a[top:bottom+1, left:right+1]
            label = int(block[-1, w//2])
            if (label in (bg, wall) or not np.all(block[:, (0, -1)] == wall)
                    or not np.all(block[-2] == wall)
                    or not np.all(np.delete(block[-1], w//2) == wall)):
                valid = False
                break
            cavity, floors = [], []
            for col in range(1, w-1):
                solid = np.flatnonzero(block[:-1, col] == wall)
                if not len(solid):
                    valid = False
                    break
                floor = int(solid[0])
                if not (1 <= floor <= h-2 and np.all(block[floor:-1, col] == wall)):
                    valid = False
                    break
                floors.append(top+floor)
                cavity.extend((r, left+col) for r in range(top, top+floor))
            if not valid:
                break
            cups.append(dict(top=top, bottom=bottom, left=left, width=w,
                             label=label, floors=tuple(floors), cells=tuple(cavity)))
            for p in part:
                accounted[p] = True
            accounted[bottom, left+w//2] = True
        if not valid or not 3 <= len(cups) <= 12:
            continue
        cups.sort(key=lambda c: (c['bottom'], c['left']))
        if len({c['width'] for c in cups}) != 1:
            continue
        bottom = max(c['bottom'] for c in cups)
        key_rows = []
        for r in range(bottom+2, a.shape[0]-1):
            xs = np.flatnonzero(a[r] != bg)
            if (3 <= len(xs) <= 8 and np.all(np.diff(xs) == 2)
                    and all(int(a[r, x]) not in (bg, wall) for x in xs)):
                key_rows.append((r, xs))
        if len(key_rows) != 1:
            continue
        kr, kxs = key_rows[0]
        key = tuple(map(int, a[kr, kxs]))
        if len(set(key)) < 2:
            continue
        accounted[kr, kxs] = True
        other = (a != bg) & ~accounted
        liquid_colors = np.unique(a[other])
        labels = {c['label'] for c in cups}
        if len(liquid_colors) != 1:
            continue
        liquid = int(liquid_colors[0])
        if liquid in labels or liquid in (bg, wall) or liquid in key:
            continue
        graph = _graph(cups)
        routes = _routes(graph, [c['label'] for c in cups], key)
        if len(routes) != 1:
            continue
        route = routes[0]
        first = cups[route[0]]
        if first['bottom'] != min(c['bottom'] for c in cups):
            continue
        eligible = set(first['cells'])
        eligible.update((r, x) for r in range(1, first['top'])
                        for x in range(first['left']+1, first['left']+first['width']-1))
        source = set(map(tuple, np.argwhere(a == liquid).tolist()))
        if not source or not source <= eligible:
            continue
        interpretations.append(dict(background=bg, wall=wall, liquid=liquid,
                                    cups=cups, graph=graph, key=key, route=route,
                                    key_row=kr, source=source))
    if len(interpretations) != 1:
        raise ValueError("expected one visibly decoded cup/key/liquid interpretation")
    return interpretations[0]


def _levels(cup):
    return [tuple(p for p in cup['cells'] if p[0] == r)
            for r in sorted({p[0] for p in cup['cells']}, reverse=True)]


def _allocate(cups, route, volume):
    wet = set()
    for index in route:
        for level in _levels(cups[index]):
            if not volume:
                return wet
            if volume < len(level):
                raise ValueError("volume would leave an ambiguous partial horizontal level")
            wet.update(level)
            volume -= len(level)
    if volume:
        raise ValueError("volume exceeds the relay's total capacity")
    return wet


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only this input; allocate count to complete cup levels in key order."""
    a = _grid(input_grid)
    info = _parse(a)
    wet = _allocate(info['cups'], info['route'], len(info['source']))
    a[a == info['liquid']] = info['background']
    for p in wet:
        a[p] = info['liquid']
    return a


def _sample_path(rng, cells, graph, length, elbow):
    budget = 1500
    top = min(c[0] for c in cells)
    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j in path:
                continue
            if elbow and len(path) == 2:
                a, b, c = cells[path[0]], cells[path[1]], cells[j]
                if (a[0] == b[0]) == (b[0] == c[0]):
                    continue
            answer = visit(path+[j])
            if answer is not None:
                return answer
        return None
    for i in rng.permutation(len(cells)):
        i = int(i)
        if cells[i][0] == top:
            answer = visit([i])
            if answer is not None:
                return answer
    return None


def _labels(rng, graph, path, colors, repeat_start=False):
    for attempt in range(72):
        k = int(rng.integers(2, min(len(colors), len(path))+1))
        pool = list(map(int, rng.permutation(colors)[:k]))
        key = pool + list(map(int, rng.choice(pool, len(path)-k)))
        key = list(map(int, rng.permutation(key)))
        if attempt >= 48 and len(path) <= len(colors):
            key = list(map(int, rng.permutation(colors)[:len(path)]))
        if repeat_start:
            key[2] = key[0]
            key[-1] = key[0]
        if len(set(key)) < 2:
            continue
        labels = list(map(int, rng.choice(colors, len(graph))))
        for i, color in zip(path, key):
            labels[i] = color
        if _routes(graph, labels, key) == [tuple(path)]:
            return labels, key
    return None


def _axis(rng, count, size, allowance):
    gaps = [1] * (count-1)
    spare = allowance - count*size - sum(gaps)
    if spare < 0:
        return None
    for i in rng.permutation(len(gaps)):
        add = int(rng.integers(min(4, spare)+1))
        gaps[int(i)] += add
        spare -= add
    return [0] + list(map(int, np.cumsum([size+g for g in gaps])))


def _floor_profile(rng, top, bottom, width, style):
    n, limit = width-2, bottom-top-1
    if style == 0:
        depths = np.full(n, int(rng.integers(1, limit+1)))
    elif style == 1:
        depths = np.clip(np.arange(n)+int(rng.integers(1, limit+1)), 1, limit)
        if rng.random() < .5:
            depths = depths[::-1]
    elif style == 2:
        depths = np.clip(1+np.abs(np.arange(n)-n//2), 1, limit)
    elif style == 3:
        depths = np.clip(limit-np.abs(np.arange(n)-n//2), 1, limit)
    else:
        depths = rng.integers(1, limit+1, size=n)
    return tuple(map(int, top+depths))


def _spatial_wet(cups, volume):
    # A diagnostic shortcut, not a second solver: ignore the key and pour in
    # reading order, arbitrarily breaking any incomplete level by column.
    cells = [p for c in cups for lev in _levels(c) for p in sorted(lev)]
    return set(cells[:volume])


def _scene(rng, bg, liquid, width, kind, force_fallback=False):
    """Bounded geometry/label search plus a constructive same-rule fallback."""
    wet_count = {'retain': 1, 'relay': 3, 'extend': 4}[kind]
    wall = int(rng.choice([c for c in range(10) if c not in (bg, liquid)]))
    colors = [c for c in range(10) if c not in (bg, liquid, wall)]
    for attempt in (range(64, 65) if force_fallback else range(65)):
        fallback = attempt == 64
        if fallback:
            nr, nc, slot, head = (2 if kind == 'retain' else 3), 2, 4, 6
            cells = [(r, c) for r in range(nr) for c in range(nc)]
            path_cells = ([(0, 1), (0, 0), (1, 0), (1, 1)] if kind == 'retain'
                          else [(0, 1), (1, 1), (1, 0), (2, 0), (2, 1)])
            path = [cells.index(c) for c in path_cells]
            ys, xs = [r*5 for r in range(nr)], [0, width+2]
            labels = list(map(int, rng.permutation(colors)[:len(cells)]))
            if kind == 'relay':
                labels[path[2]] = labels[path[0]]
                labels[path[-1]] = labels[path[0]]
            key = [labels[i] for i in path]
        else:
            nr = int(rng.choice((1, 2, 3))) if kind == 'retain' else int(rng.choice((2, 3)))
            nc = int(rng.integers(2, (4 if width == 5 else 3)+1))
            slot = int(rng.integers(4, 7))
            head = int(rng.integers(2, 7)) if kind == 'retain' else int(rng.integers(5, 7))
            ys = _axis(rng, nr, slot, 30-head-4)
            xs = _axis(rng, nc, width, 28)
            if ys is None or xs is None:
                continue
            maximum = min(8, nr*nc)
            minimum = 3 if kind == 'retain' else wet_count+1
            if maximum < minimum:
                continue
            if kind == 'relay':
                maximum = min(maximum, nr*nc-1)
            if maximum < minimum:
                continue
            length = int(rng.integers(minimum, min(maximum, minimum+3)+1))
            n = int(rng.integers(length+(kind == 'relay'), min(nr*nc, length+3)+1))
            universe = [(r, c) for r in range(nr) for c in range(nc)]
            cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
            prototypes = [dict(bottom=r, left=c*width, width=width) for r, c in cells]
            graph = _graph(prototypes)
            path = _sample_path(rng, cells, graph, length, kind != 'retain')
            if path is None:
                continue
            assigned = _labels(rng, graph, path, colors, kind == 'relay')
            if assigned is None:
                continue
            labels, key = assigned
        block_width = xs[-1]+width
        left = int(rng.integers(1, 30-block_width))
        right_margin = int(rng.integers(1, min(3, 30-left-block_width)+1))
        baselines = [head+y+slot-1 for y in ys]
        key_row = baselines[-1]+2
        height = key_row+2
        canvas_width = max(left+block_width+right_margin, 2*len(key)+1)
        if height > 30 or canvas_width > 30:
            continue
        cups = []
        for (r, c), label in zip(cells, labels):
            h = 4 if fallback else int(rng.integers(4, slot+1))
            bottom = baselines[r]
            top, x = bottom-h+1, left+xs[c]
            floors = (top+2,)*(width-2) if fallback else _floor_profile(
                rng, top, bottom, width, int(rng.integers(5)))
            cavity = tuple((y, x+j+1) for j, floor in enumerate(floors)
                           for y in range(top, floor))
            cups.append(dict(top=top, bottom=bottom, left=x, width=width,
                             label=label, floors=floors, cells=cavity))
        first = cups[path[0]]
        if fallback and kind in ('retain', 'relay'):
            first['floors'] = tuple(first['top']+1+(j % 2) for j in range(width-2))
            first['cells'] = tuple((y, first['left']+j+1)
                                   for j, floor in enumerate(first['floors'])
                                   for y in range(first['top'], floor))
        if kind == 'retain' and len(set(first['floors'])) == 1:
            continue
        if kind == 'relay' and not any(len(set(cups[i]['floors'])) > 1 for i in path[:2]):
            continue
        eligible = list(first['cells']) + [(r, x) for r in range(1, first['top'])
                    for x in range(first['left']+1, first['left']+width-1)]
        prefix = sum(len(cups[i]['cells']) for i in path[:wet_count-1])
        final_levels = _levels(cups[path[wet_count-1]])
        partial = np.cumsum([len(level) for level in final_levels])[:-1]
        volumes = [prefix+int(v) for v in partial if prefix+int(v) <= len(eligible)]
        if not volumes:
            continue
        volume = int(rng.choice(volumes))
        target = _allocate(cups, path, volume)
        if kind != 'retain' and target == _spatial_wet(cups, volume):
            continue
        grid = np.full((height, canvas_width), bg, dtype=np.int8)
        for cup in cups:
            t, b, x, w = cup['top'], cup['bottom'], cup['left'], cup['width']
            grid[t:b+1, x] = wall
            grid[t:b+1, x+w-1] = wall
            grid[b-1:b+1, x:x+w] = wall
            for j, floor in enumerate(cup['floors']):
                grid[floor:b+1, x+j+1] = wall
            grid[b, x+w//2] = cup['label']
        key_x = int(rng.integers(1, canvas_width-2*len(key)+1))
        grid[key_row, key_x:key_x+2*len(key):2] = key
        points = np.asarray(eligible, dtype=int)
        style = 1 if kind == 'extend' else int(rng.integers(5))
        anchor = points[int(rng.integers(len(points)))]
        dr, dc = abs(points[:, 0]-anchor[0]), abs(points[:, 1]-anchor[1])
        jitter = rng.random(len(points))
        scores = (dr+dc+jitter if style == 0 else jitter if style == 1 else
                  4*dr+dc+jitter if style == 2 else 4*dc+dr+jitter if style == 3 else
                  np.array([p not in target for p in eligible])*10+jitter)
        source = set(map(tuple, points[np.argsort(scores)[:volume]].tolist()))
        if source == target:
            outside = next(p for p in eligible if p not in target)
            source.remove(min(source))
            source.add(outside)
        output = grid.copy()
        for p in target:
            output[p] = liquid
        for p in source:
            grid[p] = liquid
        return dict(input=grid, output=output)
    raise RuntimeError("constructive overflow-cup fallback failed")


def _signature(puzzle):
    digest = sha256()
    for split in ('train', 'test'):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    bg, liquid = map(int, rng.choice(10, 2, replace=False))
    widths = list(map(int, rng.permutation((5, 7))))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            kind = ('retain' if i == 0 else 'relay' if i == 1 else
                    str(rng.choice(('retain', 'relay', 'extend')))) if split == 'train' else 'extend'
            width = widths[i] if split == 'train' and i < 2 else int(rng.choice((5, 7)))
            pair = _scene(rng, bg, liquid, width, kind)
            sig = pair['input'].shape, pair['input'].tobytes()
            if sig in seen:
                # Roles bg/liquid stay fixed. Permuting the remaining eight
                # colors offers >14 distinct variants even in minimal scenes.
                movable = [c for c in range(10) if c not in (bg, liquid)]
                for perm in permutations(movable):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[movable] = perm
                    candidate = {k: mapping[v] for k, v in pair.items()}
                    sig = candidate['input'].shape, candidate['input'].tobytes()
                    if sig not in seen:
                        pair = candidate
                        break
                else:
                    raise RuntimeError("distinct-input palette fallback exhausted")
            seen.add(sig)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    a, b = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == b.shape
    info = _parse(a)
    cups, route, liquid = info['cups'], info['route'], info['liquid']
    source = info['source']
    wet = set(map(tuple, np.argwhere(b == liquid).tolist()))
    assert len(wet) == len(source) > 0, 'liquid volume changed'
    assert sum(r for r, x in wet) > sum(r for r, x in source), 'gravity did not lower liquid'
    assert np.array_equal(solve(a), b), 'wrong relay allocation'
    assert not np.array_equal(a, b), 'no settling occurred'
    terrain = (a != info['background']) & (a != liquid)
    assert np.array_equal(a[terrain], b[terrain]), 'terrain/key changed'
    assert not np.any(((a != b) & (a != liquid) & (a != info['background'])))
    counts = [len(wet & set(c['cells'])) for c in cups]
    assert sum(counts) == len(wet), 'liquid outside cups'
    route_counts = [counts[i] for i in route]
    active = sum(n > 0 for n in route_counts)
    assert active and route_counts[:active] and all(route_counts[:active])
    assert not any(route_counts[active:]), 'skipped a cup'
    assert all(counts[i] == len(cups[i]['cells']) for i in route[:active-1])
    assert counts[route[active-1]] < len(cups[route[active-1]]['cells'])
    assert active < len(route), 'no dry route suffix'
    for i, c in enumerate(cups):
        pool = wet & set(c['cells'])
        if pool:
            surface = min(r for r, x in pool)
            assert pool == {p for p in c['cells'] if p[0] >= surface}, 'unfilled level'
    assert all(counts[i] == 0 for i in range(len(cups)) if i not in route)
    # Literal B-style full route painting / all selected cups full ignores A.
    all_full = {p for i in route for p in cups[i]['cells']}
    assert all_full != wet
    centers = [(cups[i]['bottom'], cups[i]['left']) for i in route[:active]]
    axes = {'h' if a[0] == b[0] else 'v' for a, b in zip(centers, centers[1:])}
    info.update(active=active, counts=counts, volume=len(source), wet=wet,
                axes=axes, ignores_key_diff=wet != _spatial_wet(cups, len(source)),
                stepped=any(len(set(c['floors'])) > 1 for c in cups),
                unused=len(cups)-len(route),
                repeated=len(set(info['key'])) < len(info['key']))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    roles, seen, infos = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _inspect(pair)
            a = _grid(pair['input'])
            sig = a.shape, a.tobytes()
            assert sig not in seen, 'duplicate input'
            seen.add(sig)
            roles.add((info['background'], info['liquid']))
            infos[split].append(info)
    assert len(roles) == 1
    assert infos['train'][0]['active'] == 1
    first = infos['train'][0]['cups'][infos['train'][0]['route'][0]]
    assert len(set(first['floors'])) > 1
    assert infos['train'][1]['active'] == 3
    assert infos['train'][1]['axes'] == {'h', 'v'}
    assert infos['train'][1]['ignores_key_diff']
    assert infos['train'][1]['unused'] >= 1 and infos['train'][1]['repeated']
    assert infos['train'][1]['key'][0] == infos['train'][1]['key'][2]
    assert {i['cups'][0]['width'] for i in infos['train'][:2]} == {5, 7}
    assert all(i['active'] == 4 and i['axes'] == {'h', 'v'} for i in infos['test'])


def _semantic_fixtures():
    """Hand-counted 6-cell cups: change the key, then change only the supply."""
    base = np.zeros((17, 15), dtype=np.int8)
    for top, left, label in ((5, 1, 1), (5, 9, 3), (10, 1, 4), (10, 9, 5)):
        base[top:top+4, left] = 8
        base[top:top+4, left+4] = 8
        base[top+2:top+4, left:left+5] = 8
        base[top+3, left+2] = label
    outputs = []
    for key, receiving_row, receiving_col in (((1, 3, 5), 6, 10), ((1, 4, 5), 11, 2)):
        inp = base.copy()
        inp[15, (1, 3, 5)] = key
        inp[1:4, 2:5] = 2  # 9 source cells, first capacity 6, excess 3.
        expected = base.copy()
        expected[15] = inp[15]
        expected[5:7, 2:5] = 2
        expected[receiving_row, receiving_col:receiving_col+3] = 2
        before = inp.copy()
        out = solve(inp)
        assert np.array_equal(out, expected)
        assert np.array_equal(inp, before), 'solve mutated the input'
        outputs.append(out == 2)
        # Three cells stop in the first cup. The later key entries stay dry.
        small = inp.copy()
        small[small == 2] = 0
        small[1, 2:5] = 2
        expected_small = base.copy()
        expected_small[15] = inp[15]
        expected_small[6, 2:5] = 2
        assert np.array_equal(solve(small), expected_small)
        # Six cells exactly fill one cup; no partially filled final cup needed.
        exact = small.copy()
        exact[2, 2:5] = 2
        expected_exact = expected_small.copy()
        expected_exact[5, 2:5] = 2
        assert np.array_equal(solve(exact), expected_exact)
        # Eight cells would split a three-cell level and must be rejected.
        ambiguous = inp.copy()
        ambiguous[1, 2] = 0
        try:
            solve(ambiguous)
        except ValueError:
            pass
        else:
            raise AssertionError('partial equal-height level accepted')
    assert not np.array_equal(*outputs), 'changing only the key did not matter'


def self_test():
    """42 streaming seeds, metamorphic checks, count API, corruption rejection."""
    _semantic_fixtures()
    signatures, shapes, geometries, roles, widths, volumes = (set() for _ in range(6))
    flags = dict(repeated=0, unused=0, stepped=0, key_counterexamples=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        assert _signature(p) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _inspect(pair)
                a, b = pair['input'], pair['output']
                roles.add((info['background'], info['liquid']))
                shapes.add(a.shape)
                widths.add(info['cups'][0]['width'])
                volumes.add(info['volume'])
                geom = tuple((c['top'], c['bottom'], c['left'], c['floors']) for c in info['cups'])
                geometries.add(geom)
                for flag in ('repeated', 'unused', 'stepped'):
                    flags[flag] += bool(info[flag])
                flags['key_counterexamples'] += info['ignores_key_diff']
                # Horizontal reflection preserves key-reading direction by
                # restoring the key row after reflecting the physical scene.
                reflected = a[:, ::-1].copy()
                reflected[info['key_row']] = a[info['key_row']]
                expected = b[:, ::-1].copy()
                expected[info['key_row']] = b[info['key_row']]
                assert np.array_equal(solve(reflected), expected)
                remap = np.random.default_rng(seed+12345).permutation(10)
                assert np.array_equal(solve(remap[a]), remap[b])
                assert np.array_equal(solve(a.tolist()), b)
    assert len(signatures) == 42 and len(geometries) >= 150
    assert len(shapes) >= 35 and len(roles) >= 20
    assert widths == {5, 7} and len(volumes) >= 20
    assert flags['repeated'] >= 15 and flags['unused'] >= 15
    assert flags['stepped'] >= 100 and flags['key_counterexamples'] >= 80
    for width in (5, 7):
        for kind in ('retain', 'relay', 'extend'):
            for bg, liquid in ((0, 1), (8, 0)):
                pair = _scene(np.random.default_rng(77), bg, liquid, width,
                              kind, force_fallback=True)
                info = _inspect(pair)
                assert info['active'] == {'retain': 1, 'relay': 3, 'extend': 4}[kind]
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(98765, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                geometries=len(geometries), roles=len(roles), volumes=len(volumes), **flags)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
