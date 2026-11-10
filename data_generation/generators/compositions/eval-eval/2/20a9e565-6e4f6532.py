"""Compass-aligned, rail-indexed affine growth. Standard library + NumPy only."""
from __future__ import annotations

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_HOST_SEEDS = (0, 234373492, 140749563, 910120904, 356163452, 515578050,
               61937205, 866567573, 393651452, 238173368, 44998753,
               949563290, 204544869, 266272179, 685586917, 123955700,
               760951994, 928125247, 990576786, 725337190, 163129913,
               732545205, 815275943, 753130382, 481572000, 10028329,
               798328057, 339542562, 128780930, 702271560, 100081223,
               108627752, 283405524, 816758465, 289268095, 834230585,
               11852964, 701817761, 705827941, 291329837, 964256881,
               180472231)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or a.dtype.kind not in 'iu' or not all(1 <= x <= 30 for x in a.shape):
        raise ValueError('expected an integer grid of dimensions 1..30')
    if np.any((a < 0) | (a > 9)):
        raise ValueError('colors must be 0..9')
    return a.astype(int, copy=True)


def _point(point, pose):
    r, c = point
    if pose >= 4:
        c = -c
    for _ in range(pose % 4):
        r, c = c, -r
    return int(r), int(c)


def _raw_transform(cells, pose):
    return {_point(p, pose): int(v) for p, v in cells.items()}


def _shift(cells, dr, dc):
    return {(r + dr, c + dc): v for (r, c), v in cells.items()}


def _normalize(cells):
    return _shift(cells, -min(r for r, _ in cells), -min(c for _, c in cells))


def _transform(cells, pose):
    return _normalize(_raw_transform(cells, pose))


def _signature(cells):
    return tuple(sorted(_normalize(cells).items()))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _marker_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_markers(a):
    found = []
    for color in np.unique(a):
        positions = set(map(tuple, np.argwhere(a == color)))
        if len(positions) != 6:
            continue
        r0 = min(r for r, _ in positions)
        r1 = max(r for r, _ in positions)
        c0 = min(c for _, c in positions)
        c1 = max(c for _, c in positions)
        bounds = (r0, r1, c0, c1)
        if r1 > r0 and c1 > c0 and positions == _marker_cells(bounds):
            if sorted(map(len, _components(positions))) == [3, 3]:
                found.append((int(color), bounds))
    assert len(found) == 1, 'ambiguous or malformed L markers'
    return found[0]


def _frame(a, bg):
    h, w = a.shape
    candidates = []
    for thickness in (1, 2):
        z = thickness
        if min(h, w) <= 2 * z + 4:
            continue
        bands = (a[:z, z:-z], a[z:-z, -z:], a[-z:, z:-z], a[z:-z, :z])
        colors = tuple(int(b.flat[0]) for b in bands)
        corners = (a[:z, :z], a[:z, -z:], a[-z:, :z], a[-z:, -z:])
        if (len(set(colors)) == 4 and bg not in colors
                and all(np.all(b == c) for b, c in zip(bands, colors))
                and all(np.all(b == bg) for b in corners)):
            mask = np.ones(a.shape, dtype=bool)
            mask[z:-z, z:-z] = False
            candidates.append((z, colors, mask))
    assert len(candidates) == 1, 'input must have one four-sided compass frame'
    return candidates[0]


def _parse(a):
    vals, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == max(counts)) == 1, 'background tie'
    bg = int(vals[np.argmax(counts)])
    marker, bounds = _find_markers(a)
    thickness, bands, structural = _frame(a, bg)
    assert marker != bg and marker not in bands
    active = (a != bg) & (a != marker) & ~structural
    sources, targets = [], []
    for component in _components(set(map(tuple, np.argwhere(active)))):
        cells = {p: int(a[p]) for p in component}
        if len(set(cells.values())) == 1:
            targets.append(cells)
        else:
            sources.append(cells)
    assert len(targets) == 1 and 3 <= len(sources) <= 4, 'expected specimens and one bare rail'
    target = targets[0]
    reg = next(iter(target.values()))
    assert reg not in {bg, marker, *bands}
    r0, r1, c0, c1 = bounds
    assert thickness <= r0 < r1 < a.shape[0] - thickness
    assert thickness <= c0 < c1 < a.shape[1] - thickness
    assert set(map(int, np.unique(a[r0:r1 + 1, c0:c1 + 1]))) <= {bg, reg, marker}
    assert all(not (r0 <= r <= r1 and c0 <= c <= c1) for s in sources for r, c in s)
    components = [set(s) for s in sources] + [set(target)]
    assert all(not (_halo(x) & y) for i, x in enumerate(components) for y in components[i + 1:])
    return dict(background=bg, marker=marker, bounds=bounds, thickness=thickness,
                bands=bands, registration=reg, sources=sources, target=target)


def _canonical_source(source, info):
    reg, bands = info['registration'], info['bands']
    candidates = []
    for pose in range(8):
        transformed = _raw_transform(source, pose)
        marks = {p for p, v in transformed.items() if v == reg}
        assert marks
        r0, c0 = min(r for r, _ in marks), min(c for _, c in marks)
        cells = _shift(transformed, -r0, -c0)
        marks = {p for p, v in cells.items() if v == reg}
        width = len(marks)
        if marks != {(0, c) for c in range(width)}:
            continue
        labels = {p: v for p, v in cells.items() if v in bands}
        if set(labels) != {(-1, 0), (0, -1)} or len(set(labels.values())) != 2:
            continue
        north, west = labels[(-1, 0)], labels[(0, -1)]
        world = [t for t in range(8)
                 if _point((-1, 0), t) == _DIRS[bands.index(north)]
                 and _point((0, -1), t) == _DIRS[bands.index(west)]]
        if len(world) != 1:
            continue
        body = {p: v for p, v in cells.items() if v not in {reg, *bands}}
        if not body or any(r < 1 or not 0 <= c < width for r, c in body):
            continue
        if set(body.values()) & {info['background'], info['marker']}:
            continue
        first = sorted(c for r, c in body if r == 1)
        groups = []
        for c in first:
            if not groups or c != groups[-1][-1] + 1:
                groups.append([c])
            else:
                groups[-1].append(c)
        n = len(groups)
        if n < 2:
            continue
        tooth_width = len(groups[0])
        pitch = groups[1][0] - groups[0][0]
        if not (1 <= tooth_width <= 3 and 1 <= pitch - tooth_width <= 2):
            continue
        if groups != [list(range(j * pitch, j * pitch + tooth_width)) for j in range(n)]:
            continue
        if width != (n - 1) * pitch + tooth_width:
            continue
        teeth, lengths, colors = [], [], []
        for j in range(n):
            tooth = {(r, c - j * pitch): v for (r, c), v in body.items()
                     if j * pitch <= c < j * pitch + tooth_width}
            if not tooth or len(set(tooth.values())) != 1:
                break
            teeth.append(set(tooth))
            lengths.append(max(r for r, _ in tooth))
            colors.append(next(iter(tooth.values())))
        if len(teeth) != n or sum(len(x) for x in teeth) != len(body):
            continue
        aligned = _transform(cells, world[0])
        same = [t for t in range(8) if _signature(_transform(aligned, t)) == _signature(source)]
        motion = 'translation' if 0 in same else 'rotation' if any(t < 4 for t in same) else 'reflection'
        candidates.append(dict(cells=cells, n=n, width=tooth_width, pitch=pitch,
                               teeth=teeth, lengths=lengths, colors=colors,
                               north=north, west=west, world=world[0], motion=motion))
    assert len(candidates) == 1, 'specimen lacks a unique intrinsic coordinate system'
    return candidates[0]


def _tooth_mask(style, length, width, period):
    """Solver-side cell predicates, independent of the stroke painter."""
    r, c = np.indices((length, width))
    edge_r = (r == 0) | (r == length - 1)
    if style == 'solid':
        mask = np.ones((length, width), dtype=bool)
    elif style == 'hook':
        mask = edge_r | (c == 0)
    elif style == 'hollow':
        mask = edge_r | (c == 0) | (c == width - 1)
    else:
        mask = edge_r | (c == 0) | (c == width - 1) | (r % period == 0)
    return {(int(rr) + 1, int(cc)) for rr, cc in np.argwhere(mask)}


def _fit_models(specimens):
    specimens = sorted(specimens, key=lambda s: s['n'])
    first, second = specimens[:2]
    width, pitch = first['width'], first['pitch']
    assert len({s['n'] for s in specimens}) == len(specimens)
    assert all((s['width'], s['pitch'], s['world'], s['north'], s['west']) ==
               (width, pitch, first['world'], first['north'], first['west']) for s in specimens)
    delta = first['lengths'][1] - first['lengths'][0]
    dn = second['n'] - first['n']
    change = second['lengths'][0] - first['lengths'][0]
    assert change % dn == 0, 'nonintegral age growth'
    beta = change // dn
    alpha = first['lengths'][0] - beta * first['n']
    assert all(s['lengths'] == [alpha + beta * s['n'] + delta * j for j in range(s['n'])]
               for s in specimens), 'non-affine tooth lengths'
    styles = []
    for style, period in (('solid', 0), ('hook', 0), ('hollow', 0), ('ladder', 2), ('ladder', 3)):
        if style == 'hook' and width < 2 or style in ('hollow', 'ladder') and width < 3:
            continue
        if all(points == _tooth_mask(style, length, width, period)
               for s in specimens for points, length in zip(s['teeth'], s['lengths'])):
            styles.append((style, period))
    assert styles, 'unknown tooth construction'
    palette_set = {color for s in specimens for color in s['colors']}
    p = len(palette_set)
    assert 1 <= p <= 3
    cycles = []
    for advance in range(p):
        mapping, valid = {}, True
        for s in specimens:
            for j, color in enumerate(s['colors']):
                residue = (j + advance * s['n']) % p
                if residue in mapping and mapping[residue] != color:
                    valid = False
                mapping[residue] = color
        if valid and len(mapping) == p and set(mapping.values()) == palette_set:
            cycles.append((advance, tuple(mapping[j] for j in range(p))))
    assert cycles, 'color phase is not established'
    return [dict(alpha=alpha, beta=beta, delta=delta, width=width, pitch=pitch,
                 style=style, period=period, advance=advance, palette=palette,
                 north=first['north'], west=first['west'], world=first['world'])
            for style, period in styles for advance, palette in cycles]


def _model_cells(model, age, freeze=None, reflect=False):
    w, s = model['width'], model['pitch']
    cells = {(0, c): model['reg'] for c in range((age - 1) * s + w)}
    cells[(-1, 0)], cells[(0, -1)] = model['north'], model['west']
    length_age = age if freeze is None else freeze
    for j in range(age):
        length = model['alpha'] + model['beta'] * length_age + model['delta'] * j
        assert length > 0
        color = model['palette'][(j + model['advance'] * length_age) % len(model['palette'])]
        for r, c in _tooth_mask(model['style'], length, w, model['period']):
            cells[(r, j * s + c)] = color
    if reflect:
        cells = {(-r, c): v for (r, c), v in cells.items()}
    return cells


def _dock_and_crop(cells, model, info, check_fit=True):
    world = _raw_transform(cells, model['world'])
    marks = {p for p, v in world.items() if v == info['registration']}
    target = set(info['target'])
    dr, dc = min(target)[0] - min(marks)[0], min(target)[1] - min(marks)[1]
    placed = _shift(world, dr, dc)
    assert {p for p, v in placed.items() if v == info['registration']} == target
    if check_fit:
        z = info['thickness']
        h, w = info['shape']
        assert all(z <= r < h - z and z <= c < w - z for r, c in placed), 'missing specimen exceeds frame'
    r0, r1, c0, c1 = info['bounds']
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), info['background'], dtype=int)
    for (r, c), color in placed.items():
        if r0 <= r <= r1 and c0 <= c <= c1:
            out[r - r0, c - c0] = color
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    info = _parse(a)
    info['shape'] = a.shape
    specimens = [_canonical_source(s, info) for s in info['sources']]
    models = _fit_models(specimens)
    answers = []
    for model in models:
        model['reg'] = info['registration']
        rail_width = len(info['target'])
        assert (rail_width - model['width']) % model['pitch'] == 0
        age = (rail_width - model['width']) // model['pitch'] + 1
        assert age >= 2
        lo = min([age] + [s['n'] for s in specimens])
        hi = max([age] + [s['n'] for s in specimens])
        assert all(model['alpha'] + model['beta'] * n + model['delta'] * j > 0
                   for n in range(lo, hi + 1) for j in range(n)), 'invalid intervening age'
        answers.append(_dock_and_crop(_model_cells(model, age), model, info))
    assert all(np.array_equal(answers[0], other) for other in answers[1:]), 'ambiguous continuation crop'
    answer, model = answers[0], models[0]
    assert np.any(answer != info['background']), 'empty reconstruction'
    nearest = min((s['n'] for s in specimens), key=lambda n: abs(n - age))
    try:
        no_a = _dock_and_crop(_model_cells(model, age, freeze=nearest), model, info, False)
        needs_a = not np.array_equal(no_a, answer)
    except AssertionError:
        needs_a = True  # Frozen geometry cannot even form a valid missing object.
    no_b = _dock_and_crop(_model_cells(model, age, reflect=True), model, info, False)
    try:
        _dock_and_crop(_model_cells(model, age, reflect=True), model, info, True)
        compass_needed = (not np.array_equal(no_b, answer)
                          and bool(set(map(int, np.unique(no_b))) & set(model['palette'])))
    except AssertionError:
        compass_needed = False
    info.update(model=model, specimens=specimens, target_age=age,
                observed_ages=sorted(s['n'] for s in specimens),
                age_gap=min(abs(age - s['n']) for s in specimens),
                needs_a=needs_a, needs_b=not np.array_equal(no_b, answer),
                compass_needed=compass_needed,
                styles=tuple(sorted(set((m['style'], m['period']) for m in models))),
                body_colors=len(model['palette']),
                motions=tuple(s['motion'] for s in specimens))
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct solely from this input, without demonstrations or RNG state."""
    return _analyze(input_grid)[0]


def _paint_specimen(n, params, palette, north, west):
    """Ground truth uses explicit strokes, not solver masks or fitted models."""
    bg, marker, reg = palette[:3]
    w, s = params['width'], params['pitch']
    cells = {(0, c): reg for c in range((n - 1) * s + w)}
    cells[(-1, 0)], cells[(0, -1)] = north, west
    for j in range(n):
        length = params['base'] + params['beta'] * (n - params['n0']) + params['delta'] * j
        color = palette[7:7 + params['p']][(j + params['advance'] * n) % params['p']]
        origin = j * s
        if params['style'] == 'solid':
            for r in range(1, length + 1):
                for c in range(w):
                    cells[(r, origin + c)] = color
        else:
            for r in range(1, length + 1):
                cells[(r, origin)] = color
                if params['style'] in ('hollow', 'ladder'):
                    cells[(r, origin + w - 1)] = color
            bars = {1, length}
            if params['style'] == 'ladder':
                bars.update(range(1, length + 1, params['period']))
            for r in bars:
                for c in range(w):
                    cells[(r, origin + c)] = color
    return cells


def _sample_parameters(rng, split, index):
    if split == 'train' and index == 0:
        style, width, count, n0, beta, delta, p, advance = 'solid', 1, 3, 2, 1, int(rng.integers(2)), 2, 1
    elif split == 'train' and index == 1:
        style = str(rng.choice(['hook', 'hollow', 'ladder']))
        width = 2 if style == 'hook' else 3
        count, n0, beta, delta, p = 3, 2, 1, 0, int(rng.integers(1, 4))
        advance = int(rng.integers(p))
    else:
        style = str(rng.choice(['solid', 'hook', 'hollow', 'ladder']))
        width = int(rng.integers(1, 4)) if style == 'solid' else 2 if style == 'hook' else 3
        count = 4 if width == 1 and rng.random() < .45 else 3
        n0 = 3 if width <= 2 and count == 3 and rng.random() < .25 else 2
        beta = int(rng.choice([-1, 0, 1, 2], p=[.22, .23, .45, .10]))
        delta = int(rng.choice([-1, 0, 1], p=[.22, .50, .28]))
        p = int(rng.integers(1, 4))
        advance = int(rng.integers(p))
    gap = 1 if width == 3 else int(rng.integers(1, 3))
    age_gap = 1 if split == 'train' else int(rng.integers(2, 4))
    target = n0 + count - 1 + age_gap
    minimum = min(beta * (n - n0) + delta * j for n in range(n0, target + 1) for j in range(n))
    base = (2 if style == 'solid' else 3) - minimum + int(rng.integers(3))
    return dict(style=style, width=width, pitch=width + gap, count=count,
                n0=n0, beta=beta, delta=delta, base=base, p=p,
                advance=advance, target=target, period=int(rng.integers(2, 4)))


def _crop_bounds(final, params, rng, full):
    length = max(r for r, _ in final)
    rail_width = (params['target'] - 1) * params['pitch'] + params['width']
    if full:
        return -2, length + 1, -2, rail_width
    mode = int(rng.integers(3))
    if mode == 0:
        return 2, length + 1, -1, rail_width
    if mode == 1:
        j = int(rng.integers(params['target'] - 1))
        c0 = j * params['pitch']
        c1 = min(rail_width, c0 + int(rng.integers(2, 4)) * params['pitch'])
        return -2, length + 1, c0, c1
    r0 = int(rng.integers(1, max(2, length - 1)))
    return r0, max(r0 + 2, length), 0, rail_width - 1


def _construct(rng, palette, params, split, index, world_pose, guaranteed=False):
    bg, marker, reg = palette[:3]
    bands = tuple(palette[3:7])
    north = bands[_DIRS.index(_point((-1, 0), world_pose))]
    west = bands[_DIRS.index(_point((0, -1), world_pose))]
    final = _paint_specimen(params['target'], params, palette, north, west)
    if split == 'train' and index < 2:
        full = index == 0
    else:
        full = rng.random() < .25
    crop = _crop_bounds(final, params, rng, full)
    corners = [_point((r, c), world_pose) for r in crop[:2] for c in crop[2:]]
    raw_final = _raw_transform(final, world_pose)
    all_r = [r for r, _ in raw_final] + [r for r, _ in corners]
    all_c = [c for _, c in raw_final] + [c for _, c in corners]
    dr, dc = -min(all_r), -min(all_c)
    final_world = _shift(raw_final, dr, dc)
    crop_world = (min(r for r, _ in corners) + dr, max(r for r, _ in corners) + dr,
                  min(c for _, c in corners) + dc, max(c for _, c in corners) + dc)
    extent = (max(all_r) - min(all_r) + 1, max(all_c) - min(all_c) + 1)
    z = 1 if guaranteed else int(rng.integers(1, 3))
    h, w = (30, 30) if guaranteed else (int(rng.integers(23, 31)), int(rng.integers(23, 31)))
    if extent[0] > h - 2 * z or extent[1] > w - 2 * z:
        return None
    if guaranteed:
        origin = (h - z - extent[0], w - z - extent[1])
    else:
        origin = (int(rng.integers(z, h - z - extent[0] + 1)),
                  int(rng.integers(z, w - z - extent[1] + 1)))
    final_world = _shift(final_world, *origin)
    bb = (crop_world[0] + origin[0], crop_world[1] + origin[0],
          crop_world[2] + origin[1], crop_world[3] + origin[1])
    marks = {p for p, v in final_world.items() if v == reg}
    if _marker_cells(bb) & marks:
        return None
    inp = np.full((h, w), bg, dtype=int)
    inp[:z, z:-z], inp[z:-z, -z:] = bands[0], bands[1]
    inp[-z:, z:-z], inp[z:-z, :z] = bands[2], bands[3]
    occupied = {(r, c) for r in range(bb[0], bb[1] + 1) for c in range(bb[2], bb[3] + 1)} | marks
    blocked = _halo(occupied)
    sources = []
    ages = list(range(params['n0'], params['n0'] + params['count']))
    rng.shuffle(ages)
    for k, age in enumerate(ages):
        canonical = _paint_specimen(age, params, palette, north, west)
        aligned = _transform(canonical, world_pose)
        pose = (1 if index == 0 else 4) if split == 'train' and k == 0 and index < 2 else int(rng.integers(8))
        source = _transform(aligned, pose)
        sh, sw = max(r for r, _ in source) + 1, max(c for _, c in source) + 1
        origins = [(r, c) for r in range(z + 1, h - z - sh)
                   for c in range(z + 1, w - z - sw)]
        rng.shuffle(origins)
        placed = None
        for r, c in origins:
            trial = _shift(source, r, c)
            if not (set(trial) & blocked):
                placed = trial
                break
        if placed is None:
            return None
        sources.append(placed)
        blocked.update(_halo(placed))
    for p in marks:
        inp[p] = reg
    for cells in sources:
        for p, color in cells.items():
            inp[p] = color
    for p in _marker_cells(bb):
        inp[p] = marker
    out = np.full((bb[1] - bb[0] + 1, bb[3] - bb[2] + 1), bg, dtype=int)
    for (r, c), color in final_world.items():
        if bb[0] <= r <= bb[1] and bb[2] <= c <= bb[3]:
            out[r - bb[0], c - bb[2]] = color
    try:
        predicted, info = _analyze(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(predicted, out) or not info['needs_b']:
        return None
    if split == 'train' and index == 0 and not (info['needs_a'] and info['compass_needed']):
        return None
    if split == 'train' and index == 1 and info['styles'] == (('solid', 0),):
        return None
    return {'input': inp, 'output': out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= value <= hi:
        raise ValueError(name + ' is out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    common_palette = tuple(int(c) for c in rng.permutation(10))
    stable_palette = bool(rng.integers(2))
    preferred_axis = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            palette = common_palette if stable_palette else tuple(int(c) for c in rng.permutation(10))
            for attempt in range(160):
                guaranteed = attempt >= 80
                if not guaranteed:
                    params = _sample_parameters(rng, split, index)
                else:
                    # Same-rule smaller strokes: still two teaching styles and the distance shift.
                    second = split == 'train' and index == 1
                    params = dict(style='hook' if second else 'solid', width=2 if second else 1,
                                  pitch=3 if second else 2, count=3, n0=2, beta=1, delta=0,
                                  base=4 if second else 3, p=2, advance=1,
                                  target=5 if split == 'train' else 6, period=2)
                axis = preferred_axis if rng.random() < .60 else int(rng.integers(2))
                world_pose = int(rng.choice([t for t in range(8) if t % 2 == axis]))
                pair = _construct(rng, palette, params, split, index, world_pose, guaranteed)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded same-rule packing exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train_info, test_info = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            predicted, info = _analyze(inp)
            assert np.array_equal(out, predicted), 'incorrect reconstruction'
            assert info['marker'] not in out
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert np.any(out == info['background']) and np.any(out != info['background'])
            assert info['needs_b'], 'compass-free reflection shortcut survives'
            (train_info if split == 'train' else test_info).append(info)
    assert any(i['needs_a'] for i in train_info), 'growth-free shortcut survives every demonstration'
    assert any(i['compass_needed'] for i in train_info), 'frame fit alone determines every demonstration'
    assert 'rotation' in train_info[0]['motions'] and 'reflection' in train_info[1]['motions']
    assert train_info[0]['styles'] != train_info[1]['styles']
    assert all(i['target_age'] > max(i['observed_ages']) and i['age_gap'] == 1 for i in train_info)
    assert all(i['target_age'] > max(i['observed_ages']) and i['age_gap'] >= 2 for i in test_info)


def self_test():
    """Bounded 42-seed semantic/variation audit, plus focused corruptions."""
    # Literal stroke fixtures exercise reverse requests and nonconsecutive ages.
    # Their expected digit rows are specified without calling the growth painter.
    for interpolation in (False, True):
        fixture = np.zeros((30, 28), dtype=int)
        fixture[0, 1:-1], fixture[1:-1, -1] = 3, 4
        fixture[-1, 1:-1], fixture[1:-1, 0] = 5, 6
        blocks = [(3, 3, ['030000', '622222'] + ['080108'] * 4),
                  (14, 3, ['0300000000', '6222222222'] + ['0801080108'] * 6)]
        if interpolation:
            blocks.append((3, 14, ['030000000000', '622222222222'] + ['010801080108'] * 7))
            bb, rail = (13, 21, 17, 26), (15, 19, 26)
            expected_rows = ['0000000000', '0030000000', '0622222220'] + ['0010801080'] * 5 + ['0000000000']
        else:
            blocks.append((3, 14, ['03000000', '62222222'] + ['01080108'] * 5))
            bb, rail = (12, 18, 18, 23), (14, 20, 23)
            expected_rows = ['000000', '003000', '062220'] + ['001080'] * 3 + ['000000']
        for r, c, rows in blocks:
            tile = np.asarray([[int(v) for v in line] for line in rows], dtype=int)
            fixture[r:r + len(tile), c:c + tile.shape[1]] = tile
        fixture[rail[0], rail[1]:rail[2]] = 2
        for p in _marker_cells(bb):
            fixture[p] = 9
        expected = np.asarray([[int(v) for v in line] for line in expected_rows], dtype=int)
        assert np.array_equal(solve(fixture), expected), 'reverse/interpolation regression'
    records, keys = [], set()
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                prediction, info = _analyze(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                keys.add(_key(pair['input']))
                records.append(info)
    assert len(keys) == len(records)
    assert len({r['shape'] for r in records}) >= 25
    assert len({r['bounds'][1] - r['bounds'][0] for r in records}) >= 10
    assert {r['model']['style'] for r in records} == {'solid', 'hook', 'hollow', 'ladder'}, 'missing tooth style'
    assert {r['model']['beta'] for r in records} >= {-1, 0, 1}, 'missing age growth regime'
    assert {r['model']['delta'] for r in records} == {-1, 0, 1}, 'missing tooth slope'
    assert {r['body_colors'] for r in records} == {1, 2, 3}, 'missing color cycle length'
    assert {len(r['specimens']) for r in records} == {3, 4}, 'missing specimen count'
    assert len({r['model']['pitch'] for r in records}) >= 3
    assert len({r['model']['world'] for r in records}) == 8
    assert {r['thickness'] for r in records} == {1, 2}
    assert len({r['background'] for r in records}) >= 8
    assert {r['age_gap'] for r in records} >= {1, 2, 3}
    assert {m for r in records for m in r['motions']} == {'translation', 'rotation', 'reflection'}
    assert sum(r['needs_a'] and r['needs_b'] for r in records) >= 100
    for a, b in ((2, 1), (10, 1), (10, 4)):
        one, two = generate_puzzle(81, a, b), generate_puzzle(81, a, b)
        validate_generated_puzzle(one)
        assert all(_key(x[k]) == _key(y[k]) for s in one for x, y in zip(one[s], two[s]) for k in x)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    bad = generate_puzzle(77, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # A single tooth corruption must not be accepted as affine evidence.
    example = generate_puzzle(7, 2, 1)['train'][0]['input'].copy()
    _, info = _analyze(example)
    source = info['sources'][0]
    body = [p for p, v in source.items() if v not in {*info['bands'], info['registration']}]
    example[body[-1]] = info['background']
    try:
        solve(example)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged specimen accepted')
