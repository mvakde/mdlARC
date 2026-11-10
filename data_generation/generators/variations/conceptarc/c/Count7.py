import numpy as np


_SOURCE_INPUTS = {
    ('000000000000', '033373333300', '030000000300', '030000000300',
     '030000000300', '030000000300', '030000000300', '030000000700',
     '030000000300', '030000000300', '033373333300', '000000000000'),
    ('00000000000', '05455555540', '05000000050', '05000000050',
     '05000000050', '04555455450', '00000000000'),
    ('000000000000000', '066663636366660', '060000000000060',
     '060000000000060', '060000000000060', '030000000000060',
     '060000000000060', '060000000000060', '060000000000060',
     '060000000000060', '060000000000060', '060000000000060',
     '060000000000060', '066666366636660', '000000000000000'),
    ('0000000002000000000', '0000000020200000000', '0000000200020000000',
     '0000002000001000000', '0000020000000200000', '0000100000000020000',
     '0002000000000002000', '0020000000000000200', '0022222222222222200',
     '0000000000000000000', '0000000000000000000'),
    ('00000000', '09999990', '09000090', '09000090', '09989990',
     '00000000'),
}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError('pair count outside legal range')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _outline(kind, height, width=None, variant=0, rotation=0):
    if kind == 'rectangle':
        path = [(0, c) for c in range(width)]
        path += [(r, width - 1) for r in range(1, height)]
        path += [(height - 1, c) for c in range(width - 2, -1, -1)]
        path += [(r, 0) for r in range(height - 2, 0, -1)]
        vertices = [(0, 0), (0, width - 1),
                    (height - 1, width - 1), (height - 1, 0)]
        sides = [
            [(0, c) for c in range(width)],
            [(r, width - 1) for r in range(height)],
            [(height - 1, c) for c in range(width)],
            [(r, 0) for r in range(height)],
        ]
    else:
        width = 2 * height - (1 if variant == 0 else 3)
        center = width // 2
        right = [(r, center + r) for r in range(height - 1)]
        left = [(r, center - r) for r in range(height - 1)]
        base = [(height - 1, c) for c in range(width)]
        path = right + list(reversed(base)) + list(reversed(left[1:]))
        vertices = [(0, center), (height - 1, width - 1),
                    (height - 1, 0)]
        if variant == 1:
            vertices += [(height - 2, width - 1), (height - 2, 0)]
        sides = [right, base, left]
    path = list(dict.fromkeys(path))
    for _ in range(rotation):
        path = [(width - 1 - c, r) for r, c in path]
        vertices = [(width - 1 - c, r) for r, c in vertices]
        sides = [[(width - 1 - c, r) for r, c in side] for side in sides]
        height, width = width, height
    mask = np.zeros((height, width), dtype=bool)
    for r, c in path:
        mask[r, c] = True
    return mask, path, vertices, sides


def _pick_markers(rng, path, vertices, sides, count):
    perimeter = len(path)
    mode = int(rng.integers(0, 5))
    if mode == 0:
        indices = rng.choice(perimeter, size=count, replace=False)
        return [path[int(i)] for i in indices]
    if mode == 1:
        start = int(rng.integers(perimeter))
        return [path[(start + j) % perimeter] for j in range(count)]
    if mode == 2:
        vertex = vertices[int(rng.integers(len(vertices)))]
        start = (path.index(vertex) - int(rng.integers(count))) % perimeter
        return [path[(start + j) % perimeter] for j in range(count)]
    if mode == 3:
        possible = [side for side in sides if len(side) >= count]
        if possible:
            side = possible[int(rng.integers(len(possible)))]
            indices = rng.choice(len(side), size=count, replace=False)
            return [side[int(i)] for i in indices]
    offset = int(rng.integers(perimeter))
    indices = [(offset + int((j + 0.25) * perimeter / count)) % perimeter
               for j in range(count)]
    return [path[i] for i in indices]


def _sample_shape(rng, kind, count):
    band = int(rng.integers(3))
    if kind == 'rectangle':
        ranges = (((3, 8), (4, 11)), ((6, 15), (7, 19)),
                  ((11, 23), (12, 27)))
        (hlo, hhi), (wlo, whi) = ranges[band]
        options = [(h, w) for h in range(hlo, hhi)
                   for w in range(wlo, whi)
                   if count <= int((2 * h + 2 * w - 4) * 0.38)]
        h, w = options[int(rng.integers(len(options)))]
        variant = 0
    else:
        hlo, hhi = ((5, 8), (7, 11), (10, 14))[band]
        options = [(h, v) for h in range(hlo, hhi) for v in (0, 1)
                   if count <= int((4 * h - (4 if v == 0 else 6)) * 0.38)]
        h, variant = options[int(rng.integers(len(options)))]
        w = None
    rotation = int(rng.integers(4)) if kind == 'triangle' else 0
    return _outline(kind, h, w, variant, rotation), variant, rotation


def _padding(rng, height, width):
    source_canvases = ((12, 12), (7, 11), (15, 15), (11, 19), (6, 8))
    fitting = [(h, w) for h, w in source_canvases
               if h >= height and w >= width and h * w > 2 * (2 * height + 2 * width - 4)]
    if fitting and rng.random() < 0.12:
        canvas_h, canvas_w = fitting[int(rng.integers(len(fitting)))]
        top = int(rng.integers(canvas_h - height + 1))
        left = int(rng.integers(canvas_w - width + 1))
        return top, canvas_h - height - top, left, canvas_w - width - left
    edge = rng.random() < 0.38
    if edge:
        side = int(rng.integers(4))
        top, bottom, left, right = [int(rng.integers(0, 4)) for _ in range(4)]
        if side == 0:
            top, bottom = 0, max(1, bottom)
        elif side == 1:
            bottom, top = 0, max(1, top)
        elif side == 2:
            left, right = 0, max(1, right)
        else:
            right, left = 0, max(1, left)
    else:
        top, bottom, left, right = [int(rng.integers(1, 5)) for _ in range(4)]
    top = min(top, 30 - height)
    bottom = min(bottom, 30 - height - top)
    left = min(left, 30 - width)
    right = min(right, 30 - width - left)
    if top == bottom == left == right == 0:
        if height < 30:
            bottom = 1
        else:
            right = 1
    return top, bottom, left, right


def _palette(rng, background, output_color, force_absent=False):
    other = [c for c in range(10) if c not in (background, output_color)]
    if background == output_color:
        role = 'background'
    elif force_absent:
        role = 'absent'
    else:
        role = ('absent', 'outline', 'marker')[int(rng.choice(3, p=[0.5, 0.27, 0.23]))]
    if role == 'outline':
        outline = output_color
        marker = int(rng.choice(other))
    elif role == 'marker':
        marker = output_color
        outline = int(rng.choice(other))
    else:
        outline, marker = [int(c) for c in rng.choice(other, size=2, replace=False)]
    return outline, marker


def _make_pair(rng, kind, count, background, output_color, force_absent=False):
    (mask, path, vertices, sides), _, _ = _sample_shape(rng, kind, count)
    h, w = mask.shape
    top, bottom, left, right = _padding(rng, h, w)
    while (top + h + bottom) * (left + w + right) <= 2 * len(path):
        if bottom + h < 30:
            bottom += 1
        else:
            right += 1
    grid = np.full((top + h + bottom, left + w + right), background, dtype=np.int8)
    outline, marker = _palette(rng, background, output_color, force_absent)
    area = grid[top:top + h, left:left + w]
    area[mask] = outline
    for r, c in _pick_markers(rng, path, vertices, sides, count):
        area[r, c] = marker
    return {'input': grid, 'output': np.full((count, 1), output_color, dtype=np.int8)}


def _grid_key(grid):
    return tuple(''.join(str(int(c)) for c in row) for row in grid)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    output_color = 2 if rng.random() < 0.38 else int(rng.integers(10))
    source_like = rng.random() < 0.42
    if source_like and rng.random() < 0.45:
        train_counts = [3, 5]
        rng.shuffle(train_counts)
    else:
        first = int(rng.integers(1, 9))
        second = int(rng.choice([c for c in range(1, 9) if c != first]))
        train_counts = [first, second]
    for index in range(2, num_train):
        train_counts.append(int(rng.choice(train_counts[:2])) if index == 2
                            and rng.random() < 0.65 else int(rng.integers(1, 9)))
    rng.shuffle(train_counts)
    if source_like:
        train_shapes = ['rectangle'] * num_train
        test_shapes = ['rectangle'] * num_test
        test_shapes[int(rng.integers(num_test))] = 'triangle'
    else:
        train_shapes = ['rectangle', 'triangle']
        rng.shuffle(train_shapes)
        train_shapes += [str(rng.choice(['rectangle', 'triangle'], p=[0.62, 0.38]))
                         for _ in range(num_train - 2)]
        test_shapes = [str(rng.choice(['rectangle', 'triangle'], p=[0.62, 0.38]))
                       for _ in range(num_test)]
        if num_test >= 2 and all(x == 'triangle' for x in test_shapes):
            test_shapes[-1] = 'rectangle'
    test_counts = []
    for index in range(num_test):
        familiar = (index == 0 or rng.random() < 0.48) if num_test >= 2 else rng.random() < 0.5
        pool = (sorted(set(train_counts)) if familiar else
                [c for c in range(1, 9) if c not in train_counts])
        if not pool:
            pool = list(range(1, 9))
        test_counts.append(int(rng.choice(pool)))
    if num_test >= 2:
        novel = [c for c in range(1, 9) if c not in train_counts]
        if novel:
            test_counts[1] = int(rng.choice(novel))
    backgrounds = [int(rng.choice([c for c in range(10) if c != output_color]))]
    backgrounds.append(int(rng.choice([c for c in range(10)
                                       if c != backgrounds[0]])))
    backgrounds += [int(rng.integers(10)) for _ in range(num_train + num_test - 2)]
    pair_seeds = rng.integers(0, 2**63 - 1, size=num_train + num_test, dtype=np.int64)
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_marker = None
    for index in range(num_train + num_test):
        group = 'train' if index < num_train else 'test'
        local = index if group == 'train' else index - num_train
        kind = train_shapes[local] if group == 'train' else test_shapes[local]
        count = train_counts[local] if group == 'train' else test_counts[local]
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        for _ in range(40):
            pair = _make_pair(pair_rng, kind, count, backgrounds[index],
                              output_color, force_absent=index == 0)
            key = _grid_key(pair['input'])
            if key not in seen and key not in _SOURCE_INPUTS and (
                    index != 1 or _inspect_input(pair['input'])['marker'] != first_marker):
                break
        else:
            raise RuntimeError('could not construct a distinct input')
        seen.add(key)
        if index == 0:
            first_marker = _inspect_input(pair['input'])['marker']
        puzzle[group].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    ys, xs = np.nonzero(grid != background)
    assert len(ys) > 0
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    area = grid[top:bottom + 1, left:right + 1]
    mask = area != background
    h, w = mask.shape
    kind = None
    variant = None
    rotation_found = 0
    vertices = None
    sides = None
    if h >= 3 and w >= 4:
        candidate, path, verts, segments = _outline('rectangle', h, w)
        if np.array_equal(mask, candidate):
            kind, variant, vertices, sides = 'rectangle', 0, verts, segments
    if kind is None:
        for source_height in range(5, 14):
            for triangle_variant in (0, 1):
                for rotation in range(4):
                    candidate, path, verts, segments = _outline(
                        'triangle', source_height, variant=triangle_variant,
                        rotation=rotation)
                    if mask.shape == candidate.shape and np.array_equal(mask, candidate):
                        kind, variant, vertices, sides = (
                            'triangle', triangle_variant, verts, segments)
                        rotation_found = rotation
                        break
                if kind is not None:
                    break
            if kind is not None:
                break
    assert kind is not None, 'input needs one supported hollow outline'
    remaining = [int(c) for c in colors if c != background]
    a, b = remaining
    acount = int(np.count_nonzero(grid == a))
    bcount = int(np.count_nonzero(grid == b))
    assert acount != bcount
    marker = a if acount < bcount else b
    outline = b if marker == a else a
    count = min(acount, bcount)
    assert 1 <= count <= 8
    assert count <= int(np.count_nonzero(mask) * 0.38)
    marker_points = {(int(r), int(c)) for r, c in np.argwhere(area == marker)}
    assert marker_points <= {(int(r), int(c)) for r, c in np.argwhere(mask)}
    marker_indices = {j for j, point in enumerate(path) if point in marker_points}
    vertex_cluster = any(path[j] in vertices
                         and (j - 1) % len(path) in marker_indices
                         and (j + 1) % len(path) in marker_indices
                         for j in marker_indices)
    widely_separated = count >= 2 and all(
        (j + 1) % len(path) not in marker_indices for j in marker_indices)
    return {'background': background, 'outline': outline, 'marker': marker,
            'count': count, 'shape': kind, 'variant': variant,
            'rotation': rotation_found,
            'size': (h, w), 'position': (top, left),
            'edge': top == 0 or left == 0 or bottom == grid.shape[0] - 1
                    or right == grid.shape[1] - 1,
            'corner': bool(marker_points & set(vertices)),
            'side_only': any(marker_points <= set(side) for side in sides),
            'vertex_cluster': vertex_cluster, 'widely_separated': widely_separated,
            'marker_points': marker_points}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    ntrain, ntest = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen = set()
    train_info = []
    test_info = []
    output_colors = set()
    for group in ('train', 'test'):
        for pair in puzzle[group]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            key = _grid_key(inp)
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            info = _inspect_input(inp)
            assert out.shape == (info['count'], 1)
            assert np.all(out == out[0, 0])
            assert not np.array_equal(inp, out)
            output_colors.add(int(out[0, 0]))
            (train_info if group == 'train' else test_info).append(info)
    assert len(output_colors) == 1
    assert len({info['count'] for info in train_info}) >= 2
    assert len({info['background'] for info in train_info}) >= 2
    assert len({info['marker'] for info in train_info}) >= 2
    assert int(next(iter(output_colors))) not in set(np.unique(puzzle['train'][0]['input']))
    assert all(1 <= info['count'] <= 8 for info in test_info)


def _puzzles_equal(a, b):
    return all(len(a[group]) == len(b[group]) and all(
        np.array_equal(x['input'], y['input']) and
        np.array_equal(x['output'], y['output'])
        for x, y in zip(a[group], b[group])) for group in ('train', 'test'))


def self_test() -> None:
    train_sizes, test_sizes, backgrounds, sizes, layouts = set(), set(), set(), set(), set()
    output_colors, role_set, train_shapes, test_shapes = set(), set(), set(), set()
    triangle_variants, triangle_rotations = set(), set()
    shape_counts, triangle_test_positions = set(), set()
    edges, corners, same_side, vertex_clusters, wide_sets = 0, 0, 0, 0, 0
    familiar, novel, large_one, small_many = 0, 0, 0, 0
    source_dims = {(12, 12), (7, 11), (15, 15), (11, 19), (6, 8)}
    observed_source_dims = set()
    comparable_count_changes = 0
    repeated_counts_new_geometry = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_sizes.add(len(puzzle['train']))
        test_sizes.add(len(puzzle['test']))
        train_known = {_inspect_input(p['input'])['count'] for p in puzzle['train']}
        assert len(train_known) >= 2
        assert len({_inspect_input(p['input'])['background'] for p in puzzle['train']}) >= 2
        comparable_count_changes += any(
            a['count'] != b['count'] and a['shape'] == b['shape']
            and abs(a['size'][0] - b['size'][0]) <= 2
            and abs(a['size'][1] - b['size'][1]) <= 2
            for n, a in enumerate([_inspect_input(q['input']) for q in puzzle['train']])
            for b in [_inspect_input(q['input']) for q in puzzle['train'][n + 1:]])
        repeated_counts_new_geometry += any(
            a['count'] == b['count'] and (a['shape'], a['size']) != (b['shape'], b['size'])
            for n, a in enumerate([_inspect_input(q['input']) for q in puzzle['train']])
            for b in [_inspect_input(q['input']) for q in puzzle['train'][n + 1:]])
        color = int(puzzle['train'][0]['output'][0, 0])
        output_colors.add(color)
        layouts.add(tuple(_grid_key(p['input']) for group in ('train', 'test')
                          for p in puzzle[group]))
        for group in ('train', 'test'):
            for pair_index, pair in enumerate(puzzle[group]):
                inp, out = pair['input'], pair['output']
                info = _inspect_input(inp)
                assert np.array_equal(out, np.full((info['count'], 1), color))
                assert _grid_key(inp) not in _SOURCE_INPUTS
                backgrounds.add(info['background'])
                sizes.add((info['shape'], info['size']))
                observed_source_dims.add(inp.shape)
                role_set.add('background' if color == info['background'] else
                             'outline' if color == info['outline'] else
                             'marker' if color == info['marker'] else 'absent')
                edges += info['edge']
                corners += info['corner']
                same_side += info['side_only'] and info['count'] >= 2
                vertex_clusters += info['vertex_cluster']
                wide_sets += info['widely_separated']
                shape_counts.add((info['shape'], info['count']))
                if info['shape'] == 'triangle':
                    triangle_variants.add(info['variant'])
                    triangle_rotations.add(info['rotation'])
                    if group == 'test':
                        triangle_test_positions.add(pair_index)
                (train_shapes if group == 'train' else test_shapes).add(info['shape'])
                if group == 'test':
                    familiar += info['count'] in train_known
                    novel += info['count'] not in train_known
                    large_one += info['count'] == 1 and max(info['size']) >= 14
                    small_many += info['count'] >= 3 and max(info['size']) <= 12
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3
    assert len(layouts) >= 40 and len(sizes) >= 50
    assert len(backgrounds) >= 8 and len(output_colors) >= 4
    assert role_set == {'background', 'outline', 'marker', 'absent'}
    assert train_shapes == test_shapes == {'rectangle', 'triangle'}
    assert triangle_variants == {0, 1}
    assert triangle_rotations == {0, 1, 2, 3}
    assert shape_counts == {(shape, count)
                            for shape in ('rectangle', 'triangle')
                            for count in range(1, 9)}
    assert triangle_test_positions == {0, 1, 2, 3}
    assert edges >= 10 and corners >= 10 and same_side >= 10
    assert vertex_clusters >= 10 and wide_sets >= 10
    assert comparable_count_changes >= 5 and repeated_counts_new_geometry >= 5
    assert familiar >= 10 and novel >= 10
    assert large_one >= 1 and small_many >= 1
    assert source_dims <= observed_source_dims
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4),
                          (None, 1), (None, 4), (2, None), (10, None)):
        seed = 900 + (ntrain or 0) * 10 + (ntest or 0)
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle['train']) == ntrain
        assert ntest is None or len(puzzle['test']) == ntest
        assert _puzzles_equal(puzzle, generate_puzzle(seed, ntrain, ntest))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid pair counts accepted')
