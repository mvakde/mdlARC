"""Antipodal stencil exchange: complete a polygon about a line's midpoint.

Only NumPy and the Python standard library are used.  No parent code is imported.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


# Clockwise compass directions spanning half a circle in (row, column) space.
_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1))
_MOTIFS = (
    frozenset({(0, 0)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}),
    frozenset({(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}),
    frozenset({(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)}),
    frozenset({(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (1, -1)}),
    frozenset({(-2, 0), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 1), (2, 1)}),
    frozenset({(-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
               (0, 0), (0, 1), (1, 0)}),
    frozenset({(-1, -1), (-1, 1), (0, -1), (0, 0), (0, 1),
               (1, -1), (1, 0), (1, 1)}),
    frozenset({(-1, -2), (-1, -1), (0, -1), (0, 0), (0, 1), (1, 1), (1, 2)}),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or any(not 1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(frozenset(component))
    return result


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("edge must be horizontal, vertical, or 45-degree diagonal")
    n = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + i * sr, a[1] + i * sc) for i in range(n + 1))


def _hull(points):
    """Monotone-chain hull; collinear intermediate points are not vertices."""
    points = sorted(set(points))
    if len(points) <= 1:
        return tuple(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    halves = []
    for order in (points, points[::-1]):
        half = []
        for p in order:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half[:-1])
    return tuple(halves[0] + halves[1])


def _outline(vertices):
    cells = set()
    for i, p in enumerate(vertices):
        cells.update(_line(p, vertices[(i + 1) % len(vertices)]))
    return frozenset(cells)


def _stamps(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _separate(a, b):
    # At least one background cell between any two input components, even diagonally.
    return not any((r + dr, c + dc) in b for r, c in a
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _inside(cells, shape):
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _paint(shape, background, layers):
    result = np.full(shape, background, dtype=np.int8)
    for cells, color in layers:
        for r, c in cells:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                result[r, c] = color
    return result


def _analyze(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or int(counts.max()) * 2 <= grid.size:
        raise ValueError("input needs three colors and a majority background")
    background = int(colors[counts.argmax()])
    foreground = [int(c) for c in colors if c != background]
    candidates = []
    for edge_color in foreground:
        edge = frozenset(map(tuple, np.argwhere(grid == edge_color).tolist()))
        # Lexicographic extremes are endpoints of all four compass line directions.
        endpoints = min(edge), max(edge)
        try:
            if len(edge) < 5 or len(edge) % 2 != 1 or edge != _line(*endpoints):
                continue
        except ValueError:
            continue
        pivot = tuple((endpoints[0][i] + endpoints[1][i]) // 2 for i in (0, 1))
        glyph_color = next(c for c in foreground if c != edge_color)
        glyph_cells = frozenset(map(tuple, np.argwhere(grid == glyph_color).tolist()))
        components = _components(glyph_cells)
        if len(components) not in (2, 3) or not _separate(edge, glyph_cells):
            continue
        centers, motifs = [], []
        valid = True
        for component in components:
            lo = tuple(min(p[i] for p in component) for i in (0, 1))
            hi = tuple(max(p[i] for p in component) for i in (0, 1))
            dims = tuple(hi[i] - lo[i] + 1 for i in (0, 1))
            if any(d not in (1, 3, 5) for d in dims):
                valid = False
                break
            center = tuple((lo[i] + hi[i]) // 2 for i in (0, 1))
            motif = frozenset((r - center[0], c - center[1]) for r, c in component)
            if (0, 0) not in motif:
                valid = False
                break
            centers.append(center)
            motifs.append(motif)
        if not valid or any(m != motifs[0] for m in motifs):
            continue
        centers = tuple(sorted(centers))
        opposite = tuple((2 * pivot[0] - r, 2 * pivot[1] - c) for r, c in centers)
        completed = set(centers) | set(opposite)
        hull = _hull(completed)
        if len(hull) != 2 * len(centers):
            continue  # Each mark must contribute one distinct opposite vertex pair.
        try:
            boundary = _outline(hull)
        except ValueError:
            continue
        stamps = _stamps(endpoints, motifs[0])
        if (not _inside(boundary | stamps, grid.shape) or boundary & stamps
                or len(stamps) != 2 * len(motifs[0]) or len(_components(stamps)) != 2):
            continue
        candidates.append(dict(background=background, edge_color=edge_color,
                               glyph_color=glyph_color, edge=edge, endpoints=endpoints,
                               pivot=pivot, centers=centers, motif=motifs[0],
                               glyph_cells=glyph_cells, vertices=hull, boundary=boundary,
                               stamps=stamps, shape=grid.shape))
    if len(candidates) != 1:
        raise ValueError("input does not have one unambiguous antipodal exchange interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles, anchors, missing vertices, and copies from this input alone."""
    a = _analyze(input_grid)
    return _paint(a['shape'], a['background'],
                  ((a['boundary'], a['edge_color']), (a['stamps'], a['glyph_color'])))


def _pose(motif, rng):
    turn, reflect = int(rng.integers(4)), bool(rng.integers(2))
    result = set()
    for r, c in motif:
        if reflect:
            c = -c
        for _ in range(turn):
            r, c = c, -r
        result.add((r, c))
    return frozenset(result)


def _sample_motif(rng, asymmetric=False):
    for _ in range(100):
        if rng.random() < .26:
            h, w = ((3, 3), (3, 5), (5, 3))[int(rng.integers(3))]
            mask = rng.random((h, w)) < float(rng.uniform(.35, .7))
            mask[h // 2, w // 2] = True
            motif = frozenset((int(r) - h // 2, int(c) - w // 2)
                              for r, c in np.argwhere(mask))
            if (len(_components(motif)) != 1 or
                    max(r for r, c in motif) - min(r for r, c in motif) != h - 1 or
                    max(c for r, c in motif) - min(c for r, c in motif) != w - 1):
                continue
        else:
            motif = _MOTIFS[int(rng.integers(len(_MOTIFS)))]
        motif = _pose(motif, rng)
        if not asymmetric or motif != frozenset((-r, -c) for r, c in motif):
            return motif
    return _MOTIFS[4]


def _sample_vertices(rng, k, scale):
    directions = sorted(map(int, rng.choice(4, size=k, replace=False)))
    low, high = scale
    lengths = rng.integers(low, high + 1, size=k)
    vectors = [(int(length) * _DIRECTIONS[d][0], int(length) * _DIRECTIONS[d][1])
               for d, length in zip(directions, lengths)]
    total = tuple(sum(v[i] for v in vectors) for i in (0, 1))
    if total[0] % 2 or total[1] % 2:
        return None
    p = (-total[0] // 2, -total[1] // 2)
    vertices = []
    for dr, dc in vectors + [(-r, -c) for r, c in vectors]:
        vertices.append(p)
        p = p[0] + dr, p[1] + dc
    return tuple(vertices)


def _representatives(rng, vertices, style):
    k = len(vertices) // 2
    if style == 'spread' and k == 3:
        start = int(rng.integers(2))
        return tuple(vertices[(start + 2 * i) % (2 * k)] for i in range(k))
    if style == 'cluster':
        start = int(rng.integers(2 * k))
        return tuple(vertices[(start + i) % (2 * k)] for i in range(k))
    return tuple(vertices[i + k * int(rng.integers(2))] for i in range(k))


def _palette(rng, base, previous):
    mode = int(rng.integers(4))
    if mode == 0:
        return base
    if mode == 1 and previous is not None:
        return previous[0], previous[2], previous[1]
    if mode == 2:
        other = [c for c in range(10) if c != base[0]]
        a, b = map(int, rng.choice(other, 2, replace=False))
        return base[0], a, b
    return tuple(map(int, rng.choice(10, 3, replace=False)))


def _make_pair(rng, k, direction, motif, palette, scale, style, seen):
    background, edge_color, glyph_color = palette
    for attempt in range(6000):
        vertices = _sample_vertices(rng, k, scale)
        if vertices is None:
            continue
        centers = _representatives(rng, vertices, style)
        dr, dc = _DIRECTIONS[direction]
        radius = int(rng.integers(2, 7))
        endpoints = ((-radius * dr, -radius * dc), (radius * dr, radius * dc))
        edge = _line(*endpoints)
        glyphs = _stamps(centers, motif)
        boundary = _outline(vertices)
        stamps = _stamps(endpoints, motif)
        if (not _separate(edge, glyphs) or len(_components(glyphs)) != k
                or len(glyphs) != k * len(motif) or boundary & stamps
                or len(_components(stamps)) != 2 or len(stamps) != 2 * len(motif)):
            continue
        all_cells = edge | glyphs | boundary | stamps
        lo = tuple(min(p[i] for p in all_cells) for i in (0, 1))
        hi = tuple(max(p[i] for p in all_cells) for i in (0, 1))
        raw_shape = tuple(hi[i] - lo[i] + 1 for i in (0, 1))
        if max(raw_shape) > 30:
            continue
        margins = []
        for size in raw_shape:
            extra = int(rng.integers(max(0, 12 - size), min(10, 30 - size) + 1))
            before = int(rng.integers(extra + 1))
            margins.append((before, extra - before))
        shape = tuple(raw_shape[i] + sum(margins[i]) for i in (0, 1))
        shift = tuple(-lo[i] + margins[i][0] for i in (0, 1))
        if all(2 * shift[i] == shape[i] - 1 for i in (0, 1)):
            continue  # Make the local center distinct from the canvas center.

        def translated(cells):
            return frozenset((r + shift[0], c + shift[1]) for r, c in cells)

        input_grid = _paint(shape, background,
                            ((translated(edge), edge_color), (translated(glyphs), glyph_color)))
        output_grid = _paint(shape, background,
                             ((translated(boundary), edge_color), (translated(stamps), glyph_color)))
        if np.count_nonzero(input_grid == background) * 2 <= input_grid.size:
            continue
        key = (shape, input_grid.tobytes())
        if key in seen:
            continue
        seen.add(key)
        return {'input': input_grid, 'output': output_grid}
    raise RuntimeError("could not place a separated polygon and endpoint glyphs")


def _build_puzzle(rng, nt, nq):
    regime = str(rng.choice(('spread', 'direction', 'scale')))
    base = tuple(map(int, rng.choice(10, 3, replace=False)))
    featured = _sample_motif(rng, asymmetric=True)
    order = list(map(int, rng.permutation(4)))
    teaching_counts = [2, 3]
    rng.shuffle(teaching_counts)
    puzzle, seen, previous = {'train': [], 'test': []}, set(), None
    for index in range(nt + nq):
        testing = index >= nt
        k = (teaching_counts[index] if index < 2 else int(rng.integers(2, 4)))
        if testing and regime == 'spread':
            k = 3
        if regime == 'direction':
            choices = order[2:] if testing else order[:2]
            direction = choices[(index - nt) % 2] if testing else choices[index % 2]
        else:
            direction = order[index] if index < 2 else int(rng.integers(4))
        style = ('spread' if testing else 'cluster') if regime == 'spread' else 'any'
        scale = (7, 10) if testing and regime == 'scale' else ((3, 6) if regime == 'scale' else (4, 9))
        motif = featured if index == 0 or rng.random() < .45 else _sample_motif(rng)
        palette = _palette(rng, base, previous)
        if index == 1 and palette == previous:
            palette = palette[0], palette[2], palette[1]
        previous = palette
        pair = _make_pair(rng, k, direction, motif, palette, scale, style, seen)
        puzzle['test' if testing else 'train'].append(pair)
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    for _ in range(64):
        puzzle = _build_puzzle(rng, nt, nq)
        teaching = [(_analyze(p['input']), p['output']) for p in puzzle['train']]
        if any(all(np.array_equal(_alternative(a, mode), output) for a, output in teaching)
               for mode in _ALTERNATIVES):
            continue  # Resample the blueprint, rather than excluding symmetric pairs.
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not construct a jointly diagnostic training set")


def _raster_general(a, b):
    """Only for testing tempting alternative rules; the actual family uses _line."""
    r, c = a
    er, ec = b
    dr, dc = abs(er - r), abs(ec - c)
    sr, sc = (1 if r < er else -1), (1 if c < ec else -1)
    error, cells = dr - dc, set()
    while True:
        cells.add((r, c))
        if (r, c) == (er, ec):
            return cells
        twice = 2 * error
        if twice > -dc:
            error -= dc
            r += sr
        if twice < dr:
            error += dr
            c += sc


def _alternative(a, mode):
    """Input-only counterfactuals used to audit demonstration information."""
    centers, pivot, motif = a['centers'], a['pivot'], a['motif']
    if mode == 'no_exchange':
        rotated = {(2 * pivot[0] - r, 2 * pivot[1] - c) for r, c in a['glyph_cells']}
        return _paint(a['shape'], a['background'],
                      ((a['edge'], a['edge_color']), (a['glyph_cells'] | rotated, a['glyph_color'])))
    if mode in ('visible_only', 'sequential'):
        points = centers
    elif mode == 'canvas_center':
        points = centers + tuple((a['shape'][0] - 1 - r, a['shape'][1] - 1 - c) for r, c in centers)
    elif mode.startswith('mirror'):
        p, q = a['endpoints']
        dr, dc = q[0] - p[0], q[1] - p[1]
        reflected = []
        for r, c in centers:
            r, c = r - pivot[0], c - pivot[1]
            if dr == 0:
                r, c = -r, c
            elif dc == 0:
                r, c = r, -c
            elif dr * dc > 0:
                r, c = c, r
            else:
                r, c = -c, -r
            if mode == 'mirror_perpendicular':
                r, c = -r, -c
            reflected.append((r + pivot[0], c + pivot[1]))
        points = centers + tuple(reflected)
    else:
        points = a['vertices']
    vertices = _hull(points)
    edge_pairs = (combinations(vertices, 2) if mode == 'all_chords' else
                  [(p, vertices[(i + 1) % len(vertices)]) for i, p in enumerate(vertices)])
    boundary = set()
    for p, q in edge_pairs:
        boundary.update(_raster_general(p, q))
    if mode == 'turned_glyphs':
        motif = frozenset((-r, -c) for r, c in motif)
    result = _paint(a['shape'], a['background'],
                    ((boundary, a['edge_color']), (_stamps(a['endpoints'], motif), a['glyph_color'])))
    return result[::-1, ::-1].copy() if mode == 'sequential' else result


_ALTERNATIVES = ('visible_only', 'no_exchange', 'canvas_center', 'mirror_line',
                 'mirror_perpendicular', 'all_chords', 'turned_glyphs', 'sequential')


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            input_grid, output_grid = _grid(pair['input']), _grid(pair['output'])
            a = _analyze(input_grid)
            _require(input_grid.shape == output_grid.shape, 'canvas dimensions changed')
            _require(np.array_equal(solve(input_grid), output_grid), 'incorrect output')
            key = (input_grid.shape, input_grid.astype(np.int8).tobytes())
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            _require(len(a['vertices']) == 2 * len(a['centers']), 'lost an antipodal pair')
            _require(len(_components(a['boundary'])) == 1, 'disconnected perimeter')
            _require(np.count_nonzero(output_grid == a['glyph_color']) == 2 * len(a['motif']),
                     'endpoint glyphs were clipped, merged, or changed')
            _require(np.count_nonzero(output_grid == a['edge_color']) == len(a['boundary']),
                     'perimeter was occluded or contains extra edges')
            _require(all((2 * a['pivot'][0] - r, 2 * a['pivot'][1] - c) in a['boundary']
                         for r, c in a['boundary']), 'perimeter lost central symmetry')
            _require(not np.array_equal(input_grid, output_grid), 'unchanged pair')
            _require(not np.array_equal(input_grid[::-1, ::-1], output_grid), 'A alone suffices')
            _require(not np.array_equal(_alternative(a, 'visible_only'), output_grid), 'B alone suffices')
            if split == 'train':
                training.append((a, output_grid))
    _require({len(a['centers']) for a, b in training} == {2, 3}, 'teach both polygon counts')
    _require(len({_direction(a) for a, b in training}) >= 2, 'teach multiple line directions')
    _require(len({(a['background'], a['edge_color'], a['glyph_color']) for a, b in training}) >= 2,
             'teach pair-local color roles')
    for mode in _ALTERNATIVES:
        _require(any(not np.array_equal(_alternative(a, mode), b) for a, b in training),
                 'training permits the alternative: ' + mode)


def _direction(a):
    p, q = a['endpoints']
    dr, dc = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
    return (dr, dc) if dr >= 0 else (-dr, -dc)


def _signature(puzzle):
    return tuple((split, p['input'].shape, p['input'].tobytes(), p['output'].tobytes())
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded checks, 42 seeded puzzles, structural metrics, and metamorphic cases."""
    shapes, palettes, motifs, structures, directions, signatures = set(), set(), set(), set(), set(), set()
    counts, side_lengths, parities, source_radii = set(), set(), set(), set()
    test_counts, scattered, pairs = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), 'not deterministic')
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                x, y = pair['input'], pair['output']
                a = _analyze(x)
                shapes.add(x.shape)
                parities.add((x.shape[0] % 2, x.shape[1] % 2))
                palettes.add((a['background'], a['edge_color'], a['glyph_color']))
                motifs.add(a['motif'])
                directions.add(_direction(a))
                counts.add(len(a['centers']))
                source_radii.add((len(a['edge']) - 1) // 2)
                side_lengths.update(max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                                    for p, q in zip(a['vertices'], a['vertices'][1:] + a['vertices'][:1]))
                relative = tuple((r - a['pivot'][0], c - a['pivot'][1]) for r, c in a['centers'])
                structures.add((relative, _direction(a), a['motif'], len(a['edge'])))
                if split == 'test':
                    test_counts.add(len(a['centers']))
                    scattered += len(a['centers']) == 3 and len(_hull(a['centers'])) == 3 and _strictly_inside(a['pivot'], _hull(a['centers']))
                pairs += 1
        # The rule must also handle family members presented in other orientations
        # and under arbitrary recolorings, with no generated-example lookup.
        if seed < 6:
            pair = puzzle['test'][0]
            permutation = np.random.default_rng(9000 + seed).permutation(10)
            for turns in (1, 2, 3):
                x = permutation[np.rot90(pair['input'], turns)]
                expected = permutation[np.rot90(pair['output'], turns)]
                _require(np.array_equal(solve(x), expected), 'rotation/recolor covariance failed')
            _require(np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output'])),
                     'reflection covariance failed')
    _require(len(signatures) == 42, 'seed diversity failed')
    _require(len(shapes) >= 45 and any(h == 30 or w == 30 for h, w in shapes), 'size diversity failed')
    _require(any(h < w for h, w in shapes) and any(h > w for h, w in shapes), 'aspect diversity failed')
    _require(len(palettes) >= 45 and len(motifs) >= 25 and len(structures) >= 150,
             'palette, motif, or structural diversity failed')
    _require(counts == test_counts == {2, 3} and len(directions) == 4, 'count/direction diversity failed')
    _require(len(side_lengths) >= 7 and len(source_radii) >= 4 and len(parities) == 4 and scattered >= 5,
             'geometry diversity failed')
    # These two minimal-count seeds formerly left a reflection alternative.
    for seed, nt, nq in ((3, 2, 1), (8, 2, 1), (123, 2, 1), (456, 10, 4), (789, 10, 1)):
        p = generate_puzzle(seed, nt, nq)
        _require(len(p['train']) == nt and len(p['test']) == nq, 'counts not honored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), shapes=len(shapes),
                semantic_palettes=len(palettes), motifs=len(motifs), structures=len(structures),
                source_directions=len(directions), polygon_side_lengths=sorted(side_lengths),
                source_half_lengths=sorted(source_radii), scattered_tests=int(scattered))


def _strictly_inside(point, polygon):
    signs = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        signs.append(int(np.sign(cross)))
    return bool(signs) and (all(v > 0 for v in signs) or all(v < 0 for v in signs))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
