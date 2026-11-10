"""Mirror agreement between opposite, color-addressed hub arms.

The tab supplies a rigid coordinate frame. Symmetry support is measured across
the hub after those frames are aligned, not inside each loose component.
Only standard-library modules and NumPy are used; solve has no generation state.
"""
from collections import Counter
import numpy as np


# Clockwise: top, right, bottom, left. Tangents rotate with the normal.
NORMALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
TANGENTS = ((0, 1), (1, 0), (0, -1), (-1, 0))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1)
                  for c in (-1, 0, 1) if r or c)
TAB = frozenset(((0, 0), (0, 1)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(cells):
    remaining, components = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        components.append(component)
    return components


def _hub_bars(r, c):
    return (frozenset(((r, c + 1), (r, c + 2))),
            frozenset(((r + 1, c + 3), (r + 2, c + 3))),
            frozenset(((r + 3, c + 1), (r + 3, c + 2))),
            frozenset(((r + 1, c), (r + 2, c))))


def _hub(grid, bg):
    candidates = []
    for r in range(grid.shape[0] - 3):
        for c in range(grid.shape[1] - 3):
            bars = _hub_bars(r, c)
            cells = set().union(*bars)
            if any(int(grid[p]) != bg for p in
                   ((rr, cc) for rr in range(r, r + 4)
                    for cc in range(c, c + 4) if (rr, cc) not in cells)):
                continue
            labels = [tuple(int(grid[p]) for p in sorted(bar)) for bar in bars]
            if all(a == b and a != bg for a, b in labels):
                colors = tuple(a for a, _ in labels)
                if len(set(colors)) == 4:
                    candidates.append((r, c, colors))
    _require(len(candidates) == 1, "one isolated four-port hollow hub is required")
    return candidates[0]


def _frame(component, grid):
    counts = Counter(int(grid[p]) for p in component)
    _require(len(counts) == 2, "each loose arm has a tab color and a material color")
    tab_colors = [color for color, n in counts.items() if n == 2]
    _require(len(tab_colors) == 1, "one two-cell colored tab is required")
    label = tab_colors[0]
    material = next(color for color in counts if color != label)
    _require(counts[material] >= 4, "an arm body has at least four cells")
    tab = sorted(p for p in component if int(grid[p]) == label)
    _require(sum(abs(tab[0][k] - tab[1][k]) for k in (0, 1)) == 1,
             "the tab must be a domino")
    body = component - set(tab)
    choices = []
    for side, (n, tangent) in enumerate(zip(NORMALS, TANGENTS)):
        if sum((tab[1][k] - tab[0][k]) * n[k] for k in (0, 1)) != 0:
            continue
        origin = min(tab, key=lambda p: sum(p[k] * tangent[k] for k in (0, 1)))
        if not all(sum((p[k] - origin[k]) * n[k] for k in (0, 1)) > 0 for p in body):
            continue
        local = {(sum((p[k] - origin[k]) * n[k] for k in (0, 1)),
                  sum((p[k] - origin[k]) * tangent[k] for k in (0, 1))) for p in body}
        choices.append((side, origin, local))
    _require(len(choices) == 1, "all body cells must lie strictly on one side of the tab")
    side, origin, local = choices[0]
    return dict(label=label, material=material, source_side=side, origin=origin,
                body=local, source=component)


def _point(local, side, origin):
    d, t = local
    n, tangent = NORMALS[side], TANGENTS[side]
    return (origin[0] + d * n[0] + t * tangent[0],
            origin[1] + d * n[1] + t * tangent[1])


def _dock_origin(side, hub):
    r, c = hub
    return ((r - 1, c + 1), (r + 1, c + 4),
            (r + 4, c + 2), (r + 2, c - 1))[side]


def _reflect(body):
    # Reflection across the hub reverses the opposite arm's rotated tangent.
    return {(d, 1 - t) for d, t in body}


def _inspect(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > grid.size / 2, "background must occupy a strict majority")
    r, c, labels = _hub(grid, bg)
    hub_cells = set().union(*_hub_bars(r, c))
    components = _components(set(map(tuple, np.argwhere(grid != bg).tolist())))
    _require(hub_cells in components, "the hub must be separated from loose arms")
    arms = [_frame(comp, grid) for comp in components if comp != hub_cells]
    _require(1 <= len(arms) <= 6, "one through six separated loose arms are required")
    materials = {a['material'] for a in arms}
    _require(len(materials) == 1 and not materials & (set(labels) | {bg}),
             "all bodies share one material color, distinct from the hub labels")
    material = materials.pop()
    by_side = {}
    raw_occupied = set(hub_cells)
    for a in arms:
        side = labels.index(a['label']) if a['label'] in labels else None
        a['side'] = side
        if side is None:
            a['kept'] = set()
            continue
        _require(side not in by_side, "at most one arm per hub label")
        by_side[side] = a
        target = {_point(p, side, _dock_origin(side, (r, c))) for p in a['body'] | TAB}
        _require(all(0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                     for rr, cc in target), "a docked raw arm must fit without clipping")
        _require(not raw_occupied & target, "docked raw arms must not overlap")
        raw_occupied |= target
    for side, a in by_side.items():
        other = by_side.get((side + 2) % 4)
        a['kept'] = a['body'] & _reflect(other['body']) if other else set()
        a['paired'] = other is not None
    return dict(grid=grid, background=bg, hub=(r, c), labels=labels,
                material=material, arms=arms, by_side=by_side)


def _render(info, agreement=True, move=True):
    grid, bg = info['grid'], info['background']
    out = np.full_like(grid, bg)
    for label, bar in zip(info['labels'], _hub_bars(*info['hub'])):
        for p in bar:
            out[p] = label
    for a in info['by_side'].values():
        if agreement and not a['paired']:
            continue
        body = a['kept'] if agreement else a['body']
        side = a['side'] if move else a['source_side']
        origin = _dock_origin(side, info['hub']) if move else a['origin']
        for p in body:
            q = _point(p, side, origin)
            _require(out[q] == bg, "retained arms overlap")
            out[q] = info['material']
        for p in TAB:
            q = _point(p, side, origin)
            _require(out[q] == bg, "retained tabs overlap")
            out[q] = a['label']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble opposite arms and retain their shared reflected support."""
    return _render(_inspect(input_grid))


def _make_body(rng, asymmetric=False, small=False):
    for _ in range(60):
        depth = int(rng.integers(2, 5 if small else 7))
        radius = int(rng.integers(1, 4))
        kind = int(rng.integers(4))
        body = {(d, t) for d in range(1, depth + 1) for t in (0, 1)}
        for d in range(2, depth + 1):
            reach = min(d, radius)
            if kind == 0:  # independently offset shoulders and lobes
                left, right = int(rng.integers(-reach, 1)), int(rng.integers(1, reach + 2))
                body |= {(d, t) for t in range(left, right + 1)}
            elif kind == 1:  # thin diagonal wings and a narrow central stem
                body |= {(d, -reach), (d, 1 + reach)}
                if rng.random() < .45:
                    body |= {(d, t) for t in range(-reach, 2 + reach)}
            elif kind == 2:  # a contour, with variable internal holes
                body |= {(d, -reach), (d, 1 + reach)}
                if d == depth or rng.random() < .25:
                    body |= {(d, t) for t in range(-reach, 2 + reach)}
                elif rng.random() < .5:
                    body -= {(d, 0), (d, 1)}
            else:  # growing branches; connectivity is checked below
                for t in range(-reach, 2 + reach):
                    if rng.random() < .55:
                        body.add((d, t))
        if asymmetric:
            # The neck fixes the possible local axis at t=0.5. A preserved
            # one-sided shoulder witnesses support from a DIFFERENT arm.
            d = int(rng.integers(2, depth + 1))
            body.add((d, 2))
            body.discard((d, -1))
        if len(_components(body | TAB)) != 1:
            continue
        if asymmetric and body == _reflect(body):
            continue
        return body
    return {(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}


def _noise(rng, body, forbidden, amount):
    noisy = set(body)
    # Neither the tab nor its immediately adjacent two-cell neck is corrupted.
    domain = {(d, t) for d in range(2, 7)
              for t in range(-min(3, d), min(4, d + 1) + 1)}
    additions = set()
    for _ in range(amount):
        candidates = sorted((domain & _halo(noisy)) - noisy - forbidden)
        if not candidates:
            break
        p = candidates[int(rng.integers(len(candidates)))]
        noisy.add(p)
        additions.add(p)
    return additions


def _scene_arms(rng, config, small=False):
    arms = {}
    sides = set(config['sides'])
    for side in (0, 1):
        opposite = side + 2
        if side not in sides or opposite not in sides:
            continue
        core = _make_body(rng, asymmetric=True, small=small or config['witness'])
        budget = int(rng.integers(1, 5)) if config['witness'] else int(rng.integers(0, 7))
        first = _noise(rng, core, set(), budget)
        second = _noise(rng, core, first, budget)
        # Disjoint additions: neither larger nor smaller whole-arm copying works.
        arms[side] = core | first
        arms[opposite] = _reflect(core | second)
    for side in sorted(sides - set(arms)):
        arms[side] = _make_body(rng, small=small)
    return arms


def _local_cells(body, side):
    return {_point(p, side, (0, 0)) for p in body | TAB}


def _relative_cells(arms):
    return set().union(*_hub_bars(0, 0),
                       *({_point(p, side, _dock_origin(side, (0, 0)))
                          for p in body | TAB} for side, body in arms.items()))


def _place(rng, shape, bg, labels, material, hub, pieces, fallback=False):
    grid = np.full(shape, bg, dtype=np.int8)
    occupied = set().union(*_hub_bars(*hub))
    for label, bar in zip(labels, _hub_bars(*hub)):
        for p in bar:
            grid[p] = label
    blocked = _halo(occupied)
    order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]['body']))
    for rank, i in enumerate(order):
        a = pieces[i]
        local = _local_cells(a['body'], a['source_side'])
        lo = tuple(min(p[k] for p in local) for k in (0, 1))
        local = {(p[0] - lo[0], p[1] - lo[1]) for p in local}
        height, width = (max(p[k] for p in local) + 1 for k in (0, 1))
        if fallback:
            # Six <=8x8 masks fit separated perimeter boxes, preserving shapes.
            top, left = (0 if rank < 3 else 22), 10 * (rank % 3)
        else:
            options = [(r, c) for r in range(shape[0] - height + 1)
                       for c in range(shape[1] - width + 1)]
            rng.shuffle(options)
            for top, left in options:
                if not {(r + top, c + left) for r, c in local} & blocked:
                    break
            else:
                return None
        origin = (top - lo[0], left - lo[1])
        coords = {_point(p, a['source_side'], origin) for p in a['body'] | TAB}
        if coords & blocked:
            return None
        for p in a['body']:
            grid[_point(p, a['source_side'], origin)] = material
        for p in TAB:
            grid[_point(p, a['source_side'], origin)] = a['label']
        blocked |= _halo(coords)
    return grid


def _build_pair(rng, config, previous_bg):
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_bg:
        palette[0], palette[9] = palette[9], palette[0]
    bg, material, labels = palette[0], palette[1], tuple(palette[2:6])
    arms = _scene_arms(rng, config)
    pieces = []
    for side, body in sorted(arms.items()):
        delta = config['rotations'].get(side, int(rng.integers(4)))
        pieces.append(dict(body=body, label=labels[side], source_side=(side - delta) % 4))
    for i in range(config['distractors']):
        pieces.append(dict(body=_make_body(rng), label=palette[6 + i],
                           source_side=int(rng.integers(4))))
    relative = _relative_cells(arms)
    low = np.array([min(p[k] for p in relative) for k in (0, 1)])
    high = np.array([max(p[k] for p in relative) for k in (0, 1)])
    minimum = high - low + 1
    shape = np.maximum(minimum, rng.integers(15, 29, size=2))
    for attempt in range(7):
        if attempt:
            shape = np.minimum(30, shape + 2)
        limits = shape - 1 - high
        hub = tuple(int(rng.choice((int(-low[k]), int(limits[k]))))
                    if rng.random() < .28 else int(rng.integers(-low[k], limits[k] + 1))
                    for k in (0, 1))
        grid = _place(rng, tuple(map(int, shape)), bg, labels, material, hub, pieces)
        if grid is not None and np.count_nonzero(grid == bg) > grid.size / 2:
            return dict(input=grid, output=solve(grid))
    grid = _place(rng, (30, 30), bg, labels, material, (13, 13), pieces, fallback=True)
    _require(grid is not None, "guaranteed perimeter packing failed")
    return dict(input=grid, output=solve(grid))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    combine = bool(rng.integers(2))
    configs = []
    for i in range(num_train + num_test):
        train = i < num_train
        if combine:
            if train:
                axis = int(rng.integers(2))
                sides = {axis, axis + 2}
                if rng.random() < .5:
                    sides.add(int(rng.choice((1 - axis, 3 - axis))))
            else:
                sides = set(range(4))
        else:
            count = int(rng.integers(0 if train else 1, 5 if train else 4))
            sides = set(map(int, rng.choice(4, count, replace=False)))
        distractors = int(rng.choice(3, p=(.4, .4, .2)))
        if not sides:
            distractors = max(1, distractors)
        configs.append(dict(sides=sides, distractors=distractors, rotations={}, witness=False))
    # The two smallest-count demonstrations still expose both reflection planes,
    # all rotations, deletion on BOTH partners, color rejection and empty ports.
    a, b = map(int, rng.choice(num_train, 2, replace=False))
    axis = int(rng.integers(2))
    configs[a]['sides'] = {axis, axis + 2} if combine else set(range(4))
    configs[b]['sides'] = {1 - axis, 3 - axis, axis}
    configs[a]['distractors'] = max(1, configs[a]['distractors'])
    configs[a]['witness'] = configs[b]['witness'] = True
    slots = [(a, axis), (a, axis + 2), (b, 1 - axis), (b, 3 - axis)]
    for delta, (i, side) in zip(map(int, rng.permutation(4)), slots):
        configs[i]['rotations'][side] = delta
    puzzle, seen, previous_bg = {'train': [], 'test': []}, set(), None
    for i, config in enumerate(configs):
        for _ in range(12):
            pair = _build_pair(rng, config, previous_bg)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("could not generate distinct inputs")
        seen.add(_key(pair['input']))
        previous_bg = int(np.bincount(pair['input'].ravel(), minlength=10).argmax())
        puzzle['train' if i < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    paired = [a for a in info['by_side'].values() if a['paired']]
    noisy = [a for a in paired if a['body'] != a['kept']]
    return dict(sides=tuple(sorted(info['by_side'])),
                paired=paired, noisy=noisy,
                orphan=any(not a['paired'] for a in info['by_side'].values()),
                distractor=any(a['side'] is None for a in info['arms']),
                rotations={(a['side'] - a['source_side']) % 4 for a in paired},
                asymmetric=any(a['kept'] != _reflect(a['kept']) for a in paired),
                both_noisy=any(a['body'] != a['kept'] and
                               info['by_side'][(a['side'] + 2) % 4]['body'] !=
                               info['by_side'][(a['side'] + 2) % 4]['kept'] for a in paired))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "output must preserve the canvas")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _inspect(inp)
            prediction = _render(info)
            _require(np.array_equal(out, prediction), "output violates cross-hub mirror agreement")
            bg = info['background']
            _require(np.count_nonzero(out == bg) > out.size / 2, "output background lost")
            # Check physical reflection in GLOBAL coordinates, independently of
            # the canonical (d,1-t) intersection used by the renderer.
            r, c = info['hub']
            target_bodies = {}
            for side, a in info['by_side'].items():
                target_bodies[side] = {_point(p, side, _dock_origin(side, info['hub']))
                                       for p in a['body']}
            expected_material = set()
            for side, cells in target_bodies.items():
                other = target_bodies.get((side + 2) % 4, set())
                for rr, cc in cells:
                    partner = (2 * r + 3 - rr, cc) if side % 2 == 0 else (rr, 2 * c + 3 - cc)
                    if partner in other:
                        expected_material.add((rr, cc))
            actual = set(map(tuple, np.argwhere(out == info['material']).tolist()))
            _require(actual == expected_material, "global reflection support is incorrect")
            if split == 'train':
                train_info.append(info)
    fs = [_features(i) for i in train_info]
    _require(set().union(*(set(a['side'] for a in f['paired']) for f in fs)) == set(range(4)),
             "training must demonstrate both opposed port pairs")
    _require(set().union(*(f['rotations'] for f in fs)) == set(range(4)),
             "training must demonstrate all four rotation amounts")
    for feature in ('orphan', 'distractor', 'asymmetric', 'both_noisy'):
        _require(any(f[feature] for f in fs), f"training is missing the {feature} witness")
    _require(any(not np.array_equal(_render(i, agreement=False), _render(i)) for i in train_info),
             "omitting symmetry agreement must change a training answer")
    _require(any(not np.array_equal(_render(i, move=False), _render(i)) for i in train_info),
             "omitting docking must change a training answer")


def self_test():
    """Bounded, grid-derived coverage over 42 seeds plus semantic regressions."""
    shapes, backgrounds, occupied, body_shapes, turns = set(), set(), set(), set(), set()
    widths, depths, counts, noise_counts, edge_count = set(), set(), set(), set(), 0
    fingerprints = set()
    for seed in range(42):
        # Include every legal test count in the finite diversity sample.
        nq = 1 + seed % 4
        nt = max(nq + 1, 2 + seed % 9)
        puzzle = generate_puzzle(seed, nt, nq)
        fingerprints.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                info = _inspect(p['input'])
                shapes.add(p['input'].shape)
                backgrounds.add(info['background'])
                occupied.add(tuple(sorted(info['by_side'])))
                counts.add(len(info['arms']))
                turns |= _features(info)['rotations']
                for a in info['by_side'].values():
                    body_shapes.add(tuple(sorted(a['body'])))
                    depths.add(max(d for d, _ in a['body']))
                    widths.add(max(t for _, t in a['body']) - min(t for _, t in a['body']) + 1)
                    if a['paired']:
                        noise_counts.add(len(a['body'] - a['kept']))
                coords = np.argwhere(p['output'] != info['background'])
                edge_count += bool(np.any((coords == 0) | (coords == np.array(p['output'].shape) - 1)))
        if seed < 2:
            again = generate_puzzle(seed, nt, nq)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in ('input', 'output'))
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and any(h != w for h, w in shapes)
    assert backgrounds == set(range(10)) and turns == set(range(4))
    assert len(occupied) >= 12 and counts >= {2, 3, 4, 5, 6}
    assert len(body_shapes) >= 150 and depths >= {2, 3, 4, 5, 6}
    assert len(widths) >= 5 and max(noise_counts) >= 4 and edge_count >= 10
    # Concrete regression: nonlocal asymmetric shoulders survive, disjoint
    # additions on BOTH partners disappear, and a valid unpaired arm disappears.
    body = {(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}
    arms = {0: body | {(3, 2)}, 2: _reflect(body | {(3, 0)}), 1: set(body)}
    pieces = [dict(body=b, label=(1, 2, 3, 4)[s], source_side=(s + 1) % 4)
              for s, b in sorted(arms.items())]
    g = _place(np.random.default_rng(1), (30, 30), 0, (1, 2, 3, 4), 5,
               (13, 13), pieces, fallback=True)
    info = _inspect(g)
    assert info['by_side'][0]['kept'] == body
    assert info['by_side'][2]['kept'] == _reflect(body)
    assert not info['by_side'][1]['paired']
    out = solve(g)
    assert np.count_nonzero(out == 5) == 2 * len(body)
    assert not np.array_equal(out, _render(info, agreement=False))
    assert not np.array_equal(out, _render(info, move=False))
    # Compatible members beyond the sampled connected common cores: shared
    # additions survive; unsupported bridges can leave supported islands; an
    # empty body intersection retains the mutually supported connector tabs.
    islands = {(1, 0), (1, 1), (3, 0), (3, 1)}
    stem0 = {(d, 0) for d in range(1, 5)}
    stem1 = {(d, 1) for d in range(1, 5)}
    examples = ((body | {(3, 1), (3, 2)}, body | {(3, 1), (3, 0)}, body | {(3, 1)}),
                (islands | {(2, 0)}, islands | {(2, 1)}, islands),
                (stem0, stem1, set()))
    for first, second, expected in examples:
        pieces = [dict(body=first, label=1, source_side=1),
                  dict(body=_reflect(second), label=3, source_side=3),
                  # Repeated distractor labels are valid even though the sampler
                  # normally uses distinct ones. They are grouped spatially.
                  dict(body=body, label=6, source_side=0),
                  dict(body=body, label=6, source_side=2)]
        sample = _place(np.random.default_rng(5), (30, 30), 0, (1, 2, 3, 4), 5,
                        (13, 13), pieces, fallback=True)
        parsed = _inspect(sample)
        assert parsed['by_side'][0]['kept'] == expected
        assert parsed['by_side'][2]['kept'] == _reflect(expected)
        result = solve(sample)
        assert np.count_nonzero(result == 5) == 2 * len(expected)
        assert np.count_nonzero(result == 6) == 0
        for side in (0, 2):
            for p in TAB:
                assert result[_point(p, side, _dock_origin(side, (13, 13)))] == (1, 2, 3, 4)[side]
    # Legal counts and invalid requests, including non-integer and bool counts.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
    bad = generate_puzzle(991, 2, 1)
    info = _inspect(bad['train'][0]['input'])
    bad['train'][0]['output'][info['hub'][0], info['hub'][1] + 1] = info['background']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('validator accepted a damaged answer')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
