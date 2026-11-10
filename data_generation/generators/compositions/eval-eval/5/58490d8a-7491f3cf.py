"""Seam census: count original objects selected by a two-source regional mask.

Only standard-library modules and NumPy are used. No parent files, generation
state, seed, or demonstrations are consulted by solve().
"""
from collections import Counter
import numpy as np


_D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SHAPES = (
    ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)), _DIAMOND,
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grids must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a


def _grid_key(a):
    a = np.asarray(a)
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_grid_key(q['input']), _grid_key(q['output']))
                       for q in p[s]) for s in ('train', 'test'))


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = _D8 if diagonal else _D4
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(cells))
    return result


def _mask(shape, cells):
    result = np.zeros(shape, bool)
    for p in cells:
        result[p] = True
    return result


def _path_info(cut):
    """A digital simple path; suppress diagonal shortcuts around orthogonal bends."""
    cells = set(map(tuple, np.argwhere(cut)))
    graph = {}
    for r, c in sorted(cells):
        adjacent = []
        for dr, dc in _D8:
            q = (r + dr, c + dc)
            if q not in cells:
                continue
            if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                continue
            adjacent.append(q)
        graph[r, c] = adjacent
    ends = sorted(p for p, ns in graph.items() if len(ns) == 1)
    _require(len(cells) >= 6 and len(ends) == 2 and
             all(1 <= len(ns) <= 2 for ns in graph.values()), "cut is not one simple path")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends), "cut endpoints must reach scene boundaries")
    _require(all(not border(p) for p in cells - set(ends)), "intermediate boundary contact")
    order, previous, current = [], None, ends[0]
    while current is not None and current not in order:
        order.append(current)
        ns = [q for q in graph[current] if q != previous]
        previous, current = current, ns[0] if ns else None
    _require(len(order) == len(cells), "disconnected cut")
    regions = [_mask(cut.shape, c) for c in _components(~cut, False)]
    _require(len(regions) == 2, "cut must separate exactly two four-connected regions")
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    sites = [list(map(tuple, np.argwhere(m & ~near))) for m in regions]
    _require(all(sites), "both sides must allow an isolated marker")
    return order, regions, sites


def _pack(panels, frame, vertical=False):
    h, w = panels[0].shape
    if vertical:
        result = np.full((3 * h + 4, w + 2), frame, np.int8)
        for i, p in enumerate(panels):
            result[1 + i * (h + 1):1 + i * (h + 1) + h, 1:-1] = p
    else:
        result = np.full((h + 2, 3 * w + 4), frame, np.int8)
        for i, p in enumerate(panels):
            result[1:-1, 1 + i * (w + 1):1 + i * (w + 1) + w] = p
    return result


def _unpack(a):
    h, w = a.shape
    frame = int(a[0, 0])
    _require(all(np.all(edge == frame) for edge in
                 (a[0], a[-1], a[:, 0], a[:, -1])), "broken outer frame")
    candidates = []
    if (w - 4) % 3 == 0 and w >= 7:
        pw = (w - 4) // 3
        if np.all(a[:, ::pw + 1] == frame):
            panels = [a[1:-1, 1 + i * (pw + 1):1 + i * (pw + 1) + pw] for i in range(3)]
            if all(not np.any(p == frame) for p in panels):
                candidates.append((panels, False))
    if (h - 4) % 3 == 0 and h >= 7:
        ph = (h - 4) // 3
        if np.all(a[::ph + 1, :] == frame):
            panels = [a[1 + i * (ph + 1):1 + i * (ph + 1) + ph, 1:-1] for i in range(3)]
            if all(not np.any(p == frame) for p in panels):
                candidates.append((panels, True))
    _require(len(candidates) == 1, "three congruent ordered panels must be unique")
    panels, vertical = candidates[0]
    return panels, frame, vertical


def _objects(panel, bg, diagonal=True):
    return [(int(color), cells) for color in np.unique(panel) if color != bg
            for cells in _components(panel == color, diagonal)]


def _decode(input_grid):
    a = _grid(input_grid)
    panels, frame, vertical = _unpack(a)
    control, x, y = panels
    bg = int(np.argmax(np.bincount(control.ravel(), minlength=10)))
    _require(np.mean(control == bg) > .60, "control background must be clear")
    colors = set(map(int, np.unique(control))) - {bg}
    _require(len(colors) >= 2, "missing path color or keys")
    line_color = max(sorted(colors), key=lambda c: int(np.sum(control == c)))
    pieces = sorted(_components(control != bg), key=len)
    _require(len(pieces) == 2 and len(pieces[0]) == 1 and len(pieces[1]) >= 6,
             "control needs one path and one isolated marker")
    marker = pieces[0][0]
    _require(control[marker] == line_color, "marker must use path color")
    cut = _mask(control.shape, pieces[1])
    path, regions, sites = _path_info(cut)
    endpoints_with_key = [p for p in (path[0], path[-1]) if control[p] != line_color]
    _require(len(endpoints_with_key) == 1, "one keyed endpoint must orient the path")
    if path[0] != endpoints_with_key[0]:
        path.reverse()
    keys = [(i, int(control[p])) for i, p in enumerate(path) if control[p] != line_color]
    _require(1 <= len(keys) <= 3 and len({c for _, c in keys}) == len(keys),
             "keys must be distinct singleton colors")
    _require(keys[0][0] == 0 and all(b[0] - a[0] >= 4 for a, b in zip(keys, keys[1:])),
             "key blocks must have room for separated tallies")
    _require(len(path) - keys[-1][0] >= 4, "last key block is too short")
    side = next((i for i, m in enumerate(regions) if m[marker]), None)
    _require(side is not None and marker in sites[side], "marker touches the path")
    marked, other = regions[side], regions[1 - side]
    _require(not any(np.any(p == line_color) for p in (x, y)), "path color in source scenes")
    _require(all(np.mean(p == bg) > .45 for p in (x, y)), "shared source background missing")
    objects = [_objects(p, bg) for p in (x, y)]
    source_colors = {c for scene in objects for c, _ in scene}
    _require({c for _, c in keys} <= source_colors, "every key needs a source object")
    # Within each source, distinct glyphs have a halo, even across different colors.
    for scene in objects:
        owner = np.full(control.shape, -1, int)
        for i, (_, cells) in enumerate(scene):
            for p in cells:
                owner[p] = i
        for r, c in np.argwhere(owner >= 0):
            patch = owner[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            _require(np.all((patch == -1) | (patch == owner[r, c])), "source glyphs touch")
    limits = [b[0] for b in keys[1:]] + [len(path)]
    capacities = [(stop - start) // 2 for (start, _), stop in zip(keys, limits)]
    return dict(panels=panels, frame=frame, vertical=vertical, background=bg,
                line_color=line_color, cut=cut, path=path, marker=marker, side=side,
                regions=regions, sites=sites, marked=marked, other=other,
                keys=keys, capacities=capacities, objects=objects)


def _census(objects, marked, other):
    counts = Counter()
    for scene, region in zip(objects, (marked, other)):
        for color, cells in scene:
            if any(region[p] for p in cells):
                counts[color] += 1
    return counts


def _render(shape, bg, path, keys, counts):
    result = np.full(shape, bg, np.int8)
    for start, color in keys:
        for i in range(counts[color]):
            result[path[start + 2 * i]] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only object census, rendered on the input's ordered contour."""
    d = _decode(input_grid)
    counts = _census(d['objects'], d['marked'], d['other'])
    _require(all(counts[c] <= cap for (_, c), cap in zip(d['keys'], d['capacities'])),
             "a census does not fit its visible key block")
    return _render(d['cut'].shape, d['background'], d['path'], d['keys'], counts)


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape(rng, small=False):
    if small or rng.random() < .8:
        choices = _SHAPES[:4] if small else _SHAPES
        cells = choices[int(rng.integers(len(choices)))]
    else:
        cells = {(1, 1)}
        target = int(rng.integers(3, 11))
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _D8
                               if 0 <= r + dr < 4 and 0 <= c + dc < 4} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    return _normalize(cells)


def _source(rng, shape, bg, palette, number, small=False):
    grid = np.full(shape, bg, np.int8)
    blocked = np.zeros(shape, bool)
    objects = []
    for i in range(number):
        glyph = _shape(rng, small)
        gh, gw = 1 + max(r for r, c in glyph), 1 + max(c for r, c in glyph)
        if gh > shape[0] or gw > shape[1]:
            continue
        ah, aw = shape[0] - gh + 1, shape[1] - gw + 1
        available = np.ones((ah, aw), bool)
        for r, c in glyph:
            available &= ~blocked[r:r + ah, c:c + aw]
        starts = np.argwhere(available)
        if not len(starts):
            continue
        r0, c0 = map(int, starts[int(rng.integers(len(starts)))])
        color = int(palette[i % len(palette)]) if i < len(palette) else int(rng.choice(palette))
        cells = tuple((r0 + r, c0 + c) for r, c in glyph)
        for r, c in cells:
            grid[r, c] = color
            blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        objects.append((color, cells))
    return grid, objects


def _line(p, q):
    r, c = p
    r1, c1 = q
    dr, dc = abs(r1 - r), abs(c1 - c)
    sr, sc = (1 if r < r1 else -1), (1 if c < c1 else -1)
    error, cells = dc - dr, []
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _sample_cut(rng, h, w, family):
    for _ in range(100):
        cut = np.zeros((h, w), bool)
        c = int(rng.integers(2, w - 2))
        if family == 'straight':
            cut[:, c] = True
        elif family == 'diagonal':
            end = int(rng.integers(1, w - 1))
            if end == c:
                continue
            for p in _line((0, c), (h - 1, end)):
                cut[p] = True
        elif family == 'elbow':
            r = int(rng.integers(max(4, h // 3), h - 3))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        elif family == 'stepped':
            rows = sorted(map(int, rng.choice(np.arange(3, h - 3, 3),
                          size=int(rng.integers(1, min(4, (h - 6) // 3) + 1)), replace=False)))
            last = 0
            for r in rows:
                cut[last:r + 1, c] = True
                choices = [v for v in range(1, w - 1) if v != c]
                nxt = int(rng.choice(choices))
                cut[r, min(c, nxt):max(c, nxt) + 1] = True
                last, c = r, nxt
            cut[last:, c] = True
        else:
            raise ValueError('unknown cut family')
        try:
            _path_info(cut)
            return cut
        except AssertionError:
            continue
    raise RuntimeError('could not sample a readable cut')


def _key_starts(rng, length, k):
    # Geometry is chosen before populations; block length is a capacity, not a label.
    starts = [0]
    for i in range(1, k):
        lo = starts[-1] + 4
        hi = length - 4 * (k - i)
        target = int(round(length * i / k)) + int(rng.integers(-1, 2))
        starts.append(max(lo, min(hi, target)))
    return starts


def _make_pair(rng, cut, marker, sources, records, keys, roles, transforms=None):
    frame, bg, line = roles
    path, regions, _ = _path_info(cut)
    # Construction order starts at a chosen boundary endpoint.
    if rng.random() < .5:
        path.reverse()
    control = np.full(cut.shape, bg, np.int8)
    control[cut] = line
    control[marker] = line
    for start, color in keys:
        control[path[start]] = color
    side = 0 if regions[0][marker] else 1
    counts = _census(records, regions[side], regions[1 - side])
    output = _render(cut.shape, bg, path, keys, counts)
    panels = [control, *sources]
    flip_r, flip_c, transpose = (tuple(bool(rng.integers(2)) for _ in range(3))
                                if transforms is None else transforms)
    for axis, yes in ((0, flip_r), (1, flip_c)):
        if yes:
            panels = [np.flip(p, axis).copy() for p in panels]
            output = np.flip(output, axis).copy()
    if transpose:
        panels = [p.T.copy() for p in panels]
        output = output.T.copy()
    pair = dict(input=_pack(panels, frame, transpose), output=output)
    _require(np.array_equal(solve(pair['input']), output), 'construction and visible census disagree')
    return pair


def _witness_pair(rng, base_palette):
    """One whole U, a diagonal diamond, heterogeneous glyphs, seam-only and ignored objects."""
    frame, bg, line, primary, second, distractor = base_palette[:6]
    h, w = int(rng.integers(24, 29)), 8
    c = int(rng.integers(3, 5))
    cut = np.zeros((h, w), bool)
    cut[:, c] = True
    x, y = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    records = [[], []]

    def put(source, color, cells):
        cells = tuple(cells)
        for p in cells:
            (x, y)[source][p] = color
        records[source].append((color, cells))

    put(0, primary, [(1, c - 1), (1, c), (1, c + 1), (2, c + 1),
                     (3, c + 1), (3, c), (3, c - 1)])
    put(0, primary, [(6 + r, col) for r, col in _DIAMOND])
    put(0, primary, [(11, 1)])
    put(0, primary, [(r, col) for r in (17, 18) for col in (6, 7)])
    put(0, second, [(21, c)])  # Seam-only; keep this separate from the primary merging witness.
    put(1, primary, [(r, c + 1) for r in (1, 2, 3)])
    put(1, primary, [(r, col) for r in (7, 8) for col in (6, 7)])
    put(1, primary, [(17, 1)])
    put(0, second, [(r, col) for r in (12, 13) for col in (6, 7)])
    put(1, second, [(12, 1)])
    put(0, distractor, [(r, col) for r in (15, 16) for col in (0, 1)])
    put(1, distractor, [(22, 0), (23, 1)])
    _, _, sites = _path_info(cut)
    marker = sites[0][int(rng.integers(len(sites[0])))]
    # First block comfortably exceeds both observed primary counts (5 and 3).
    starts = [0, int(rng.integers(12, h - 4))]
    return _make_pair(rng, cut, marker, [x, y], records,
                      list(zip(starts, (primary, second))), (frame, bg, line))


def _opposite(pair, rng):
    d = _decode(pair['input'])
    control = d['panels'][0].copy()
    control[d['marker']] = d['background']
    marker = d['sites'][1 - d['side']][int(rng.integers(len(d['sites'][1 - d['side']])))]
    control[marker] = d['line_color']
    inp = _pack([control, *d['panels'][1:]], d['frame'], d['vertical'])
    return {'input': inp, 'output': solve(inp)}


def _random_pair(rng, base_palette, fixed_palette, family, k, extrapolate=False):
    for attempt in range(180):
        h = int(rng.integers(24 if extrapolate or k == 3 else 12, 29))
        w = int(rng.integers(5, 9))
        cut = _sample_cut(rng, h, w, family)
        path, regions, sites = _path_info(cut)
        if len(path) < 4 * k:
            continue
        starts = _key_starts(rng, len(path), k)
        capacities = [(b - a) // 2 for a, b in zip(starts, starts[1:] + [len(path)])]
        palette = base_palette if fixed_palette else list(map(int, rng.permutation(10)))
        frame, bg, line = palette[:3]
        colors = palette[3:3 + k]
        nd = int(rng.integers(0, 3)) if not extrapolate else int(rng.integers(0, 2))
        available = colors + palette[3 + k:3 + k + nd]
        number = int(rng.integers(7, 11) if extrapolate else rng.integers(max(k, 3), 9))
        source_data = [_source(rng, (h, w), bg, available, number, extrapolate) for _ in range(2)]
        sources, records = [v[0] for v in source_data], [v[1] for v in source_data]
        if not set(colors) <= {c for scene in records for c, _ in scene}:
            continue
        side = int(rng.integers(2))
        counts = _census(records, regions[side], regions[1 - side])
        if not all(counts[c] <= cap for c, cap in zip(colors, capacities)):
            continue
        if not any(counts[c] for c in colors):
            continue
        if extrapolate and max(counts[c] for c in colors) < 6:
            continue
        if not extrapolate and max(counts[c] for c in colors) > 5:
            continue
        marker = sites[side][int(rng.integers(len(sites[side])))]
        return _make_pair(rng, cut, marker, sources, records,
                          list(zip(starts, colors)), (frame, bg, line))
    raise RuntimeError('could not construct a census within visible block capacities')


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    fixed_palette = bool(rng.random() < .25)
    regime = int(rng.integers(3))  # held-out bends, larger populations, more categories
    first = _witness_pair(rng, palette)
    twin = _opposite(first, rng)
    puzzle = {'train': [first, twin], 'test': []}
    seen = {_grid_key(p['input']) for p in puzzle['train']}
    for split, n in (('train', nt - 2), ('test', nq)):
        for _ in range(n):
            train = split == 'train'
            for retry in range(40):
                if regime == 0:
                    family = str(rng.choice(('straight', 'diagonal') if train else ('elbow', 'stepped')))
                    k = int(rng.integers(1, 4))
                    extrapolate = False
                elif regime == 1:
                    family = str(rng.choice(('straight', 'diagonal', 'stepped')))
                    k = int(rng.integers(1, 3)) if train else 1
                    extrapolate = not train
                else:
                    family = str(rng.choice(('straight', 'diagonal', 'elbow', 'stepped')))
                    k = int(rng.integers(1, 3)) if train else 3
                    extrapolate = False
                pair = _random_pair(rng, palette, fixed_palette, family, k, extrapolate)
                if _grid_key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('could not generate distinct examples')
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    """Counterfactual counts reconstructed from grids; no sampled metadata."""
    bg = d['background']
    x, y = d['panels'][1:]
    keyed = [c for _, c in d['keys']]
    counts = _census(d['objects'], d['marked'], d['other'])
    opposite = _census(d['objects'], d['other'], d['marked'])
    all_counts = Counter(c for scene in d['objects'] for c, _ in scene)
    wholly = Counter()
    clipped = Counter()
    area = Counter()
    for panel, scene, region in zip((x, y), d['objects'], (d['marked'], d['other'])):
        for color, cells in scene:
            if all(region[p] for p in cells):
                wholly[color] += 1
            area[color] += sum(bool(region[p]) for p in cells)
        for color in np.unique(panel):
            if color != bg:
                clipped[int(color)] += len(_components((panel == color) & region))
    four = _census([_objects(p, bg, False) for p in (x, y)], d['marked'], d['other'])
    composite = np.full(x.shape, bg, np.int8)
    composite[d['marked']] = x[d['marked']]
    composite[d['other']] = y[d['other']]
    # Parent B's seam foreground precedence included here only as an ablation.
    composite[d['cut']] = np.where(x[d['cut']] != bg, x[d['cut']], y[d['cut']])
    merged = Counter(c for c, _ in _objects(composite, bg))
    values = lambda c: tuple(c[color] for color in keyed)
    crossing = any(any(d['marked'][p] for p in cells) and any(d['other'][p] for p in cells)
                   for scene in d['objects'] for _, cells in scene)
    seam_only = any(all(d['cut'][p] for p in cells)
                    for scene in d['objects'] for color, cells in scene if color in keyed)
    return dict(counts=values(counts), opposite=values(opposite), all_counts=values(all_counts),
                clipped=values(clipped), four=values(four), wholly=values(wholly),
                merged=values(merged), area=values(area), crossing=crossing, seam_only=seam_only,
                distractor=any(c not in keyed for scene in d['objects'] for c, _ in scene),
                count_x=values(_census([d['objects'][0], []], d['marked'], d['other'])),
                count_y=values(_census([[], d['objects'][1]], d['marked'], d['other'])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, teaching, training = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _grid_key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            d = _decode(inp)
            _require(np.array_equal(solve(inp), out), 'incorrect extracted seam census')
            _require(out.shape == d['cut'].shape, 'wrong extraction dimensions')
            _require(np.all(out[~d['cut']] == d['background']), 'off-track output cells')
            _require(not np.any(out == d['line_color']), 'control color leaked into census')
            _require(set(map(int, np.unique(out))) <= {d['background']} | {c for _, c in d['keys']},
                     'unkeyed color leaked into census')
            if split == 'train':
                f = _features(d)
                teaching.append(f)
                training.append(d)
    _require(any(f['counts'] != f['four'] for f in teaching), 'diagonal connectivity not taught')
    for alternative in ('all_counts', 'clipped', 'wholly', 'merged', 'area', 'count_x', 'count_y', 'opposite'):
        _require(any(f['counts'] != f[alternative] for f in teaching), 'shortcut not distinguished: ' + alternative)
    _require(any(f['crossing'] for f in teaching), 'no straddling object demonstration')
    _require(any(f['seam_only'] for f in teaching), 'seam-only exclusion not taught')
    _require(any(f['distractor'] for f in teaching), 'distractor exclusion not taught')
    _require(any(0 in f['counts'] for f in teaching), 'zero-count erasure not taught')
    _require(any(max(f['counts']) >= 3 for f in teaching), 'tally extension not taught')
    # Find an actual control intervention: only the marker differs.
    twin_found = False
    for a in training:
        for b in training:
            if (a['side'] == b['side'] or a['cut'].shape != b['cut'].shape or
                    a['keys'] != b['keys'] or a['frame'] != b['frame']):
                continue
            if not all(np.array_equal(p, q) for p, q in zip(a['panels'][1:], b['panels'][1:])):
                continue
            ca, cb = a['panels'][0].copy(), b['panels'][0].copy()
            ca[a['marker']], cb[b['marker']] = a['background'], b['background']
            if np.array_equal(ca, cb) and _features(a)['counts'] != _features(b)['counts']:
                twin_found = True
    _require(twin_found, 'missing effective opposite-marker control pair')


def self_test():
    witness = _witness_pair(np.random.default_rng(77), list(range(10)))
    f = _features(_decode(witness['input']))
    _require(f['counts'] == (5, 0) and f['clipped'] == (6, 0) and f['merged'] == (4, 1),
             'known object-permanence/merging witness changed')
    opposite = _opposite(witness, np.random.default_rng(78))
    _require(_features(_decode(opposite['input']))['counts'] == (3, 2), 'known opposite census changed')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, 'seed diversity collapsed')
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), 'nondeterminism')
    shapes, outputs, cuts, glyphs, palettes, categories, populations, directions = set(), set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_cuts = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                shapes.add(np.asarray(pair['input']).shape)
                outputs.add(d['cut'].shape)
                local_cuts.add((d['cut'].shape, d['cut'].tobytes()))
                cuts.add((d['cut'].shape, d['cut'].tobytes()))
                glyphs.update(_normalize(cells) for scene in d['objects'] for _, cells in scene)
                palettes.add((d['frame'], d['background'], d['line_color']))
                categories.add(len(d['keys']))
                populations.update(_features(d)['counts'])
                p, q = d['path'][0], d['path'][-1]
                delta = (q[0] - p[0], q[1] - p[1])
                directions.add((int(np.sign(delta[0])), int(np.sign(delta[1]))))
                # Recoloring changes roles only; all ten colors are equally usable.
                lut = (np.arange(10) + 3) % 10
                _require(np.array_equal(solve(lut[pair['input']]), lut[pair['output']]), 'palette equivariance')
        _require(len(local_cuts) >= 2, 'within-puzzle geometry collapsed')
        # Reflect aligned interiors without reversing source order.
        pair = puzzle['train'][0]
        d = _decode(pair['input'])
        for axis in (0, 1):
            inp = _pack([np.flip(p, axis) for p in d['panels']], d['frame'], d['vertical'])
            _require(np.array_equal(solve(inp), np.flip(pair['output'], axis)), 'reflection equivariance')
        inp = _pack([p.T for p in d['panels']], d['frame'], not d['vertical'])
        _require(np.array_equal(solve(inp), pair['output'].T), 'transposition equivariance')
    _require(len(shapes) >= 30 and len(outputs) >= 25 and len(cuts) >= 80, 'geometry variation too narrow')
    _require(len(glyphs) >= 35 and len(palettes) >= 30, 'glyph/palette variation too narrow')
    _require(categories == {1, 2, 3} and {0, 1, 2, 3, 4, 5, 6} <= populations, 'count/category variation missing')
    _require({(1, 0), (-1, 0), (0, 1), (0, -1)} <= directions, 'directed contour orientations missing')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit count ignored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Actual marker movement changes census but does not change the ordered track.
    pair = puzzles[0]['train'][0]
    swapped = _opposite(pair, np.random.default_rng(0))
    d, e = _decode(pair['input']), _decode(swapped['input'])
    _require(d['path'] == e['path'], 'marker movement changed contour direction')
    _require(_features(e)['counts'] == _features(d)['opposite'], 'marker-swap census incorrect')
    bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted census')
    for style in ('branch', 'loop', 'three_regions'):
        cut = np.zeros((12, 8), bool)
        if style == 'branch':
            cut[:, 3] = True
            cut[5, 3:7] = True
        elif style == 'loop':
            cut[3:9, 2] = cut[3:9, 5] = True
            cut[3, 2:6] = cut[8, 2:6] = True
        else:
            cut[:, 2] = cut[:, 5] = True
        try:
            _path_info(cut)
        except AssertionError:
            pass
        else:
            raise AssertionError('malformed cut accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
