"""Smallest seed, connected scaffold: a self-contained ARC task family.

Only standard-library modules and NumPy are used. The solver reads its input;
generation, validation, and demonstration scheduling are separate concerns.
Boxes use (row, column, height, width, color); envelope boxes are half-open.
"""

from collections import deque
import json

import numpy as np


STEM_COLOR = 1
WIDTHS = (2, 4, 6, 8)
SELF_TEST_SEEDS = (
    0, 485560734, 662160683, 920388867, 332432074, 28732517,
    559113138, 287775200, 335840162, 68398490, 718450049, 220226191,
    816502064, 341206742, 555211640, 710555780, 404866916, 596488413,
    74951338, 531312634, 61572527, 341669385, 684229132, 313242723,
    944229146, 405822137, 410501994, 340935550, 586941171, 89227243,
    698583408, 44976660, 597329355, 353536018, 914435509, 655919430,
    564309702, 742529474, 274673152, 406222562, 168058791, 208893073,
)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"illegal {name}: {value!r}")
    return int(value)


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= x <= 30 for x in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid


def _separated(a, b):
    r, c, h, w = a[:4]
    y, x, hh, ww = b[:4]
    # Strict inequality leaves at least one whole background row or column.
    return r + h < y or y + hh < r or c + w < x or x + ww < c


def _parse(input_grid):
    grid = _array(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() * 2 <= grid.size or STEM_COLOR in values:
        raise ValueError("input needs a majority background and no blue cells")
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    cores = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            todo = [(r, c)]
            cells = []
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < height and 0 <= xx < width
                            and not seen[yy, xx] and int(grid[yy, xx]) == color):
                        seen[yy, xx] = True
                        todo.append((yy, xx))
            r0 = min(y for y, _ in cells)
            c0 = min(x for _, x in cells)
            h = max(y for y, _ in cells) - r0 + 1
            w = max(x for _, x in cells) - c0 + 1
            if len(cells) != h * w or w not in WIDTHS or not 1 <= h <= 10:
                raise ValueError("cores must be complete rectangles with allowed widths")
            cores.append((r0, c0, h, w, color))
    if not 2 <= len(cores) <= 6:
        raise ValueError("expected 2..6 cores")
    if not all(_separated(a, b) for i, a in enumerate(cores) for b in cores[:i]):
        raise ValueError("input cores require a background gap")
    areas = [h * w for _, _, h, w, _ in cores]
    if areas.count(min(areas)) != 1:
        raise ValueError("minimum core area must be unique")
    return grid, bg, cores


def _parts(shape, cores, *, unit=False, stems=True):
    """Each scaffold consists of a frame envelope and possibly a stem box."""
    height, width = shape
    result = []
    for r, c, h, w, *_ in cores:
        t = 1 if unit else w // 2
        envelope = (max(0, r-t), max(0, c-t),
                    min(height, r+h+t), min(width, c+w+t))
        boxes = [envelope]
        if stems and r+h+t < height:
            boxes.append((r+h+t, c, height, c+w))
        result.append(boxes)
    return result


def _contact(a, b):
    ar, ac, ay, ax = a
    br, bc, by, bx = b
    # Positive overlap on one axis and overlap-or-adjacency on the other.
    # Requiring one positive overlap excludes diagonal-only contact.
    return ((max(ar, br) < min(ay, by) and max(ac, bc) <= min(ax, bx))
            or (max(ac, bc) < min(ax, bx) and max(ar, br) <= min(ay, by)))


def _graph(shape, cores, *, unit=False, stems=True):
    parts = _parts(shape, cores, unit=unit, stems=stems)
    graph = [set() for _ in cores]
    for i in range(len(cores)):
        for j in range(i):
            if any(_contact(a, b) for a in parts[i] for b in parts[j]):
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _distances(graph, root):
    distance = {root: 0}
    queue = deque([root])
    while queue:
        i = queue.popleft()
        for j in sorted(graph[i]):
            if j not in distance:
                distance[j] = distance[i] + 1
                queue.append(j)
    return distance


def _root(cores):
    return min(range(len(cores)), key=lambda i: cores[i][2] * cores[i][3])


def _masks(shape, cores, *, unit=False, stems=True):
    envelopes, stem_masks = [], []
    for boxes in _parts(shape, cores, unit=unit, stems=stems):
        frame = np.zeros(shape, dtype=bool)
        stem = np.zeros(shape, dtype=bool)
        r, c, y, x = boxes[0]
        frame[r:y, c:x] = True
        if len(boxes) == 2:
            r, c, y, x = boxes[1]
            stem[r:y, c:x] = True
        envelopes.append(frame)
        stem_masks.append(stem)
    return envelopes, stem_masks


def _render(shape, bg, cores, root=None, *, unit=False, stems=True):
    if root is None:
        root = _root(cores)
    members = set(_distances(_graph(shape, cores, unit=unit, stems=stems), root))
    envelopes, stem_masks = _masks(shape, cores, unit=unit, stems=stems)
    frame = np.logical_or.reduce([envelopes[i] for i in sorted(members)])
    stem = np.logical_or.reduce([stem_masks[i] for i in sorted(members)])
    occupied = frame | stem
    output = np.full(shape, bg, dtype=np.int8)
    output[stem] = STEM_COLOR
    output[frame] = cores[root][4]
    for i in sorted(members):
        r, c, h, w, color = cores[i]
        output[r:r+h, c:c+w] = color
    ys, xs = np.nonzero(occupied)
    bounds = (int(ys.min()), int(xs.min()), int(ys.max())+1, int(xs.max())+1)
    r, c, y, x = bounds
    return output[r:y, c:x].copy(), members, bounds


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the selected network and its exact colored crop from input alone."""
    grid, bg, cores = _parse(input_grid)
    return _render(grid.shape, bg, cores)[0]


def _features(shape, cores):
    root = _root(cores)
    graph = _graph(shape, cores)
    distance = _distances(graph, root)
    members = set(distance)
    r, c, h, w, color = cores[root]
    color_totals = {k: sum(a[2]*a[3] for a in cores if a[4] == k)
                    for k in {a[4] for a in cores}}
    scaffold_areas = [sum((y-r0)*(x-c0) for r0, c0, y, x in boxes)
                      for boxes in _parts(shape, cores)]
    clipping = []
    for i in members:
        y, x, hh, ww = cores[i][:4]
        t = ww // 2
        clipping.append((y-t < 0, x-t < 0, y+hh+t > shape[0], x+ww+t > shape[1]))
    return {
        'root': root, 'members': members, 'depth': max(distance.values()),
        'discarded': len(members) < len(cores),
        'stem_sensitive': set(_distances(_graph(shape, cores, stems=False), root)) != members,
        'scale_sensitive': set(_distances(_graph(shape, cores, unit=True), root)) != members,
        'perimeter_counterexample': any(a[2]+a[3] < h+w for a in cores),
        'width_counterexample': any(a[3] < w for a in cores),
        'winner_color_unique': sum(a[4] == color for a in cores) == 1,
        'grown_area_counterexample': any(area < scaffold_areas[root]
                                        for area in scaffold_areas),
        'frequency_counterexample': color_totals[color] > min(color_totals.values()),
        'retained_tall': any(cores[i][2] >= 8 for i in members),
        'clipped': any(any(flags) for flags in clipping),
        'top_clip': any(flags[0] for flags in clipping),
        'side_clip': any(flags[1] or flags[3] for flags in clipping),
        'bottom_clip': any(flags[2] for flags in clipping),
        'edges': sum(map(len, graph)) // 2,
    }


def _visible_frame_widths(shape, bg, cores, output, bounds):
    """Widths with a full-thickness side and an exposed outer edge in the label.

    A conservative demonstration-readability check, not another solving rule.
    The exterior can be background, a blue stem, or a different-color core.
    """
    _, members, _ = _render(shape, bg, cores)
    canvas = np.full(shape, bg, dtype=np.int8)
    r0, c0, y, x = bounds
    canvas[r0:y, c0:x] = output
    color = cores[_root(cores)][4]
    result = set()
    for i in sorted(members):
        r, c, h, w, _ = cores[i]
        t = w // 2
        sides = []
        if c-t > 0:
            sides.append((canvas[r:r+h, c-t:c], canvas[r:r+h, c-t-1]))
        if c+w+t < shape[1]:
            sides.append((canvas[r:r+h, c+w:c+w+t], canvas[r:r+h, c+w+t]))
        if r-t > 0:
            sides.append((canvas[r-t:r, c:c+w], canvas[r-t-1, c:c+w]))
        if r+h+t < shape[0]:
            sides.append((canvas[r+h:r+h+t, c:c+w], canvas[r+h+t, c:c+w]))
        if any(np.all(strip == color) and np.any(outside != color)
               for strip, outside in sides):
            result.add(w)
    return result


def _sample_dimensions(rng, mode):
    if mode == 'area':
        if rng.random() < .5:
            dims = [(7, 2), (4, 4), (int(rng.integers(3, 8)), 6)]
        else:
            dims = [(1, 6), (2, 4), (int(rng.integers(5, 8)), 2)]
        count = int(rng.integers(3, 6))
        minimum = dims[0][0] * dims[0][1]
    elif mode == 'bridge':
        w = int(rng.choice((4, 6)))
        dims = [(1, w), (int(rng.integers(w//2+1, 8)), 2),
                (int(rng.integers(2, 8)), 10-w)]
        count = int(rng.integers(4, 6))
        minimum = w
    elif mode == 'test':
        count = int(rng.integers(4, 7))
        dims = [(int(rng.integers(8, 11)), int(rng.choice(WIDTHS)))]
        minimum = 0
    else:
        count = int(rng.integers(2, 6))
        dims, minimum = [], 0
    while len(dims) < count:
        h = int(rng.integers(1, 11 if mode == 'test' else 8))
        w = int(rng.choice(WIDTHS if mode == 'test' else WIDTHS[:3]))
        if h * w > minimum:
            dims.append((h, w))
    areas = [h*w for h, w in dims]
    if areas.count(min(areas)) != 1:
        return None
    return dims


def _coordinate(rng, last):
    if rng.random() < .18:
        return 0 if rng.random() < .5 else last
    return int(rng.integers(last + 1))


def _place(rng, shape, dimensions):
    height, width = shape
    cores = []
    for h, w in dimensions:
        for _ in range(70):
            candidate = (_coordinate(rng, height-h), _coordinate(rng, width-w), h, w)
            if all(_separated(candidate, old) for old in cores):
                cores.append(candidate)
                break
        else:
            return None
    return cores


def _sample_pair(rng, mode, avoid_roles=None):
    for _ in range(4000):
        dimensions = _sample_dimensions(rng, mode)
        if dimensions is None:
            continue
        if mode in ('area', 'bridge', 'test'):
            shape = (int(rng.integers(18, 31)), int(rng.integers(21, 31)))
        else:
            low = 12 if len(dimensions) <= 3 else 18
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        if sum(h*w for h, w in dimensions) * 2 >= shape[0]*shape[1]:
            continue
        geometry = _place(rng, shape, dimensions)
        if geometry is None:
            continue
        palette = [int(x) for x in rng.permutation([c for c in range(10) if c != STEM_COLOR])]
        bg, colors = palette[0], palette[1:len(geometry)+1]
        root = min(range(len(geometry)), key=lambda i: geometry[i][2]*geometry[i][3])
        if avoid_roles is not None and (bg == avoid_roles[0] or colors[root] == avoid_roles[1]):
            continue
        if mode == 'area':
            colors[2] = colors[0]
        elif rng.random() < .25:
            a, b = (int(x) for x in rng.choice(len(colors), 2, replace=False))
            colors[b] = colors[a]
            if avoid_roles is not None and colors[root] == avoid_roles[1]:
                continue
        cores = [(*rect, color) for rect, color in zip(geometry, colors)]
        features = _features(shape, cores)
        if mode == 'area':
            good = (features['perimeter_counterexample'] and features['frequency_counterexample']
                    and features['stem_sensitive'] and features['discarded'])
        elif mode == 'bridge':
            good = (features['width_counterexample'] and features['depth'] >= 2
                    and features['scale_sensitive'] and features['discarded']
                    and features['winner_color_unique']
                    and features['grown_area_counterexample'])
        elif mode == 'test':
            good = features['depth'] >= 2 and features['retained_tall']
        else:
            good = True
        if not good:
            continue
        output, _, bounds = _render(shape, bg, cores)
        if mode == 'area' and STEM_COLOR not in output:
            continue
        if mode == 'bridge' and len(_visible_frame_widths(shape, bg, cores, output, bounds)) < 2:
            continue
        grid = np.full(shape, bg, dtype=np.int8)
        for r, c, h, w, color in cores:
            grid[r:r+h, c:c+w] = color
        return {'input': grid, 'output': output}
    raise RuntimeError(f"bounded placement exhausted for {mode}")


def _key(grid):
    array = np.asarray(grid, dtype=np.uint8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    modes = ['area', 'bridge'] + ['random'] * (num_train-2)
    rng.shuffle(modes)
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_roles = None
    for split, specifications in (('train', modes), ('test', ['test'] * num_test)):
        for index, mode in enumerate(specifications):
            for _ in range(20):
                avoid = first_roles if split == 'train' and index == 1 else None
                pair = _sample_pair(rng, mode, avoid_roles=avoid)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and index == 0:
                        _, bg, cores = _parse(pair['input'])
                        first_roles = (bg, cores[_root(cores)][4])
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_component(mask, start):
    """Cell flood fill, used to check the analytical contact graph."""
    seen = np.zeros(mask.shape, dtype=bool)
    seen[start] = True
    todo = [start]
    while todo:
        r, c = todo.pop()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                    and mask[y, x] and not seen[y, x]):
                seen[y, x] = True
                todo.append((y, x))
    return seen


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, bg, cores = _parse(pair['input'])
    out = _array(pair['output'])
    expected, members, bounds = _render(inp.shape, bg, cores)
    assert np.array_equal(out, expected), "incorrect scaffold crop or layers"
    assert not np.array_equal(inp, out), "identity example"
    root = _root(cores)
    envelopes, stems = _masks(inp.shape, cores)
    full_union = np.logical_or.reduce(envelopes + stems)
    actual_component = _cell_component(full_union, cores[root][:2])
    selected_union = np.logical_or.reduce([envelopes[i] | stems[i] for i in sorted(members)])
    assert np.array_equal(actual_component, selected_union), "graph/cell connectivity mismatch"
    for i, (r, c, h, w, color) in enumerate(cores):
        assert bool(actual_component[r, c]) == (i in members)
        if i in members:
            y0, x0 = r-bounds[0], c-bounds[1]
            assert np.all(out[y0:y0+h, x0:x0+w] == color), "core not preserved"
    occupied = out != bg
    assert np.any(occupied[0]) and np.any(occupied[-1])
    assert np.any(occupied[:, 0]) and np.any(occupied[:, -1]), "crop not tight"
    assert set(int(c) for c in np.unique(out)) <= set(int(c) for c in np.unique(inp)) | {STEM_COLOR}
    features = _features(inp.shape, cores)
    features.update({'background': bg, 'winner_color': cores[root][4],
                     'cores': cores, 'shape': inp.shape, 'out_shape': out.shape,
                     'visible_frame_widths': _visible_frame_widths(inp.shape, bg, cores, out, bounds)})
    return features


def validate_generated_puzzle(puzzle):
    """Check returned arrays and family-specific demonstration/test evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    profiles = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            features = _check_pair(pair)
            key = _key(pair['input'])
            assert key not in seen, "duplicate input"
            seen.add(key)
            profiles[split].append(features)
    train = profiles['train']
    assert {2, 4, 6} <= {a[3] for f in train for a in f['cores']}
    assert all(a[2] <= 7 and a[3] <= 6 for f in train for a in f['cores'])
    assert any(f['perimeter_counterexample'] and f['frequency_counterexample']
               and f['stem_sensitive'] and f['discarded']
               and STEM_COLOR in np.asarray(pair['output'])
               for f, pair in zip(train, puzzle['train'])), "missing area/stem witness"
    assert any(f['width_counterexample'] and f['depth'] >= 2
               and f['scale_sensitive'] and f['discarded'] and f['winner_color_unique']
               and f['grown_area_counterexample'] and len(f['visible_frame_widths']) >= 2
               for f in train), "missing bridge witness"
    assert len({f['background'] for f in train}) >= 2
    assert len({f['winner_color'] for f in train}) >= 2
    assert all(f['retained_tall'] and f['depth'] >= 2 for f in profiles['test'])
    # Removing A's minimum selection must alter a demonstrated answer.
    assert any(not np.array_equal(
        np.asarray(pair['output']),
        _render(f['shape'], f['background'], f['cores'],
                max(range(len(f['cores'])), key=lambda i: f['cores'][i][2]*f['cores'][i][3]))[0])
        for f, pair in zip(train, puzzle['train']))


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def _structure(grid):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in np.asarray(grid).flat)


def self_test():
    """A bounded 42-seed structural suite plus API and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in SELF_TEST_SEEDS]
    assert _same(puzzles[0], generate_puzzle(SELF_TEST_SEEDS[0]))
    assert not _same(puzzles[0], puzzles[1])
    shapes, out_shapes, structures, winner_shapes = set(), set(), set(), set()
    backgrounds, widths, counts, depths, group_sizes, edge_counts = (set() for _ in range(6))
    coverage = {k: 0 for k in ('stem_sensitive', 'scale_sensitive', 'discarded',
                              'clipped', 'top_clip', 'side_clip', 'bottom_clip',
                              'retained_tall', 'isolated_winner', 'all_connected')}
    vertical_failures = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                f = _check_pair(pair)
                shapes.add(inp.shape)
                out_shapes.add(out.shape)
                structures.add((inp.shape, _structure(inp)))
                backgrounds.add(f['background'])
                widths.update(a[3] for a in f['cores'])
                counts.add(len(f['cores']))
                depths.add(f['depth'])
                group_sizes.add(len(f['members']))
                edge_counts.add(f['edges'])
                root_core = f['cores'][f['root']]
                winner_shapes.add(root_core[2:4])
                for k in coverage:
                    if k == 'isolated_winner':
                        coverage[k] += len(f['members']) == 1
                    elif k == 'all_connected':
                        coverage[k] += not f['discarded']
                    else:
                        coverage[k] += int(f[k])
                assert np.array_equal(solve(inp), out)
        # Horizontal reflection preserves widths and downward orientation.
        # Arbitrary color permutation fixing the stem role preserves the rule.
        pair = puzzle['train'][0]
        inp, out = pair['input'], pair['output']
        assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
        palette = [c for c in range(10) if c != STEM_COLOR]
        permutation = np.arange(10)
        permutation[palette] = palette[1:] + palette[:1]
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        vertical_failures += not np.array_equal(solve(np.flipud(inp)), np.flipud(out))
    assert len(shapes) >= 60 and len(out_shapes) >= 50 and len(structures) == 210
    assert len(winner_shapes) >= 15 and len(backgrounds) == 9 and widths == set(WIDTHS)
    assert counts == {2, 3, 4, 5, 6} and {0, 1, 2} <= depths
    assert {1, 2, 3, 4, 5} <= group_sizes and len(edge_counts) >= 6
    assert all(coverage[k] >= 42 for k in ('stem_sensitive', 'scale_sensitive', 'discarded', 'retained_tall'))
    assert coverage['isolated_winner'] >= 5 and coverage['all_connected'] >= 10
    assert all(coverage[k] >= 10 for k in ('top_clip', 'side_clip', 'bottom_clip'))
    assert vertical_failures >= 30
    # Every explicit legal count; minimum counts keep both witness demonstrations.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            puzzle = generate_puzzle(10000 + 10*ntrain + ntest, ntrain, ntest)
            assert len(puzzle['train']) == ntrain and len(puzzle['test']) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a!r}/{b!r}")
    # Exact diagonal contact is not an edge; face contact and overlap are.
    assert not _contact((0, 0, 2, 2), (2, 2, 4, 4))
    assert _contact((0, 0, 2, 2), (2, 1, 4, 3))
    assert _contact((0, 0, 3, 3), (1, 1, 2, 2))
    return {'seeds': 42, 'pairs': 210, 'input_shapes': len(shapes),
            'output_shapes': len(out_shapes), 'normalized_inputs': len(structures),
            'winner_shapes': len(winner_shapes), 'backgrounds': len(backgrounds),
            'widths': sorted(widths), 'core_counts': sorted(counts),
            'graph_depths': sorted(depths), 'selected_counts': sorted(group_sizes),
            'coverage': coverage, 'vertical_reflection_changes': vertical_failures}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
