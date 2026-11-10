import hashlib
import numpy as np


_SOURCE_DIGESTS = frozenset({
    '9e79355de52b68b7bfe4a5507542ddfd4d4e5972308de6529e8dcb331072e62d',
    '193f48c02ae5878e7b7cab4677e2a6d5f0c118f9d9a316f9a8c7852883bc2998',
    '24c14aa62c36d9c3b588bded6989f739042755679e64784b8e73b514cfef355d',
    '125f58fdfaa6f9ddaa309ff110f95ecb2bc7deff7448e5a9a86fdb597dfc0456',
})
_TRAIN_SPANS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1),
                (2, 2), (2, 3), (3, 2), (3, 3))
_TEST_SPANS = _TRAIN_SPANS + ((1, 4), (4, 1), (2, 4), (4, 2), (3, 4), (4, 3))


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _base(rows, cols, background, carrier):
    grid = np.full((4 * rows + 1, 4 * cols + 1), background, np.int8)
    for r in range(rows):
        for c in range(cols):
            grid[4*r+1:4*r+4, 4*c+1:4*c+4] = carrier
            grid[4*r+2, 4*c+2] = background
    return grid


def _signature(rect):
    r0, r1, c0, c1 = rect
    top, bottom, left, right = 4*r0, 4*r1, 4*c0, 4*c1
    cells = {(top, c) for c in range(left, right+1)}
    cells.update((bottom, c) for c in range(left, right+1))
    cells.update((r, left) for r in range(top, bottom+1))
    cells.update((r, right) for r in range(top, bottom+1))
    cells.update((4*r+2, 4*c+2) for r in range(r0, r1) for c in range(c0, c1))
    return cells


def _perimeter(rect):
    return {cell for cell in _signature(rect)
            if cell[0] in (4*rect[0], 4*rect[1]) or
            cell[1] in (4*rect[2], 4*rect[3])}


def _centers(rect):
    r0, r1, c0, c1 = rect
    return {(4*r+2, 4*c+2) for r in range(r0, r1) for c in range(c0, c1)}


def _infer(rows, cols, clues):
    clues = set(clues)
    if not clues:
        raise AssertionError('empty clues')
    best, minima = None, []
    for r0 in range(rows):
        for r1 in range(r0+1, rows+1):
            for c0 in range(cols):
                for c1 in range(c0+1, cols+1):
                    rect = (r0, r1, c0, c1)
                    area = (r1-r0)*(c1-c0)
                    if best is not None and area > best:
                        continue
                    if clues <= _signature(rect):
                        if best is None or area < best:
                            best, minima = area, [rect]
                        else:
                            minima.append(rect)
    if len(minima) != 1:
        raise AssertionError('clues do not specify one smallest rectangle')
    return minima[0]


def _clues(rng, rows, cols, rect, profile):
    perim = sorted(_perimeter(rect))
    centers = sorted(_centers(rect))
    if profile == 'centers':
        r0, r1, c0, c1 = rect
        clues = {(4*r0+2, 4*c0+2), (4*(r1-1)+2, 4*(c1-1)+2)}
        if rng.random() < .3:
            clues.update(centers)
        assert _infer(rows, cols, clues) == rect
        return sorted(clues)
    if profile == 'original_bar' and rect[1]-rect[0] == 1 and rect[3]-rect[2] == 2:
        top, bottom, left, right = 4*rect[0], 4*rect[1], 4*rect[2], 4*rect[3]
        clues = set(centers) | {(top+1, right), (bottom-1, left)}
        assert _infer(rows, cols, clues) == rect
        return sorted(clues)
    for _ in range(80):
        if profile == 'mixed':
            nc = int(rng.integers(1, min(3, len(centers))+1))
            np_ = int(rng.integers(1, 5))
        elif profile == 'run':
            nc = int(rng.integers(0, 2))
            np_ = int(rng.integers(3, 9))
        else:
            nc = 0
            np_ = int(rng.integers(2, 7))
        nc, np_ = min(nc, len(centers)), min(np_, len(perim))
        selected = set()
        if nc:
            selected.update(centers[int(i)] for i in rng.choice(len(centers), nc, replace=False))
        if profile == 'run':
            side = int(rng.integers(0, 4))
            top, bottom, left, right = 4*rect[0], 4*rect[1], 4*rect[2], 4*rect[3]
            edge = ([(top, c) for c in range(left, right+1)] if side == 0 else
                    [(bottom, c) for c in range(left, right+1)] if side == 1 else
                    [(r, left) for r in range(top, bottom+1)] if side == 2 else
                    [(r, right) for r in range(top, bottom+1)])
            length = min(len(edge), max(2, np_//2))
            start = int(rng.integers(0, len(edge)-length+1))
            selected.update(edge[start:start+length])
        remaining = sorted(set(perim)-selected)
        selected.update(remaining[int(i)] for i in rng.choice(len(remaining), min(np_, len(remaining)), replace=False))
        try:
            if _infer(rows, cols, selected) == rect:
                return sorted(selected)
        except AssertionError:
            pass
    top, bottom, left, right = 4*rect[0], 4*rect[1], 4*rect[2], 4*rect[3]
    selected = {(top, left), (bottom, right)}
    if profile == 'mixed':
        selected.add(centers[int(rng.integers(len(centers)))])
    assert _infer(rows, cols, selected) == rect
    return sorted(selected)


def _place(rng, rows, cols, spans):
    choices = []
    for height, width in spans:
        if height > rows or width > cols:
            return None
        options = [(r, r+height, c, c+width)
                   for r in range(rows-height+1) for c in range(cols-width+1)]
        choices.append([options[int(i)] for i in rng.permutation(len(options))])
    order = sorted(range(len(spans)), key=lambda i: len(choices[i]))
    placed = [None] * len(spans)
    def visit(index, used):
        if index == len(order):
            return True
        j = order[index]
        for rect in choices[j]:
            sig = _signature(rect)
            if sig.isdisjoint(used):
                placed[j] = rect
                if visit(index+1, used | sig):
                    return True
        return False
    return placed if visit(0, set()) else None


def _make_pair(rows, cols, roles, rectangles, colors, clues):
    background, carrier = roles
    inp = _base(rows, cols, background, carrier)
    out = inp.copy()
    for rect, color, group in zip(rectangles, colors, clues):
        assert set(group) <= _signature(rect)
        assert _infer(rows, cols, group) == rect
        for r, c in group:
            assert inp[r, c] == background
            inp[r, c] = color
        for r, c in _signature(rect):
            assert out[r, c] == background
            out[r, c] = color
    assert not np.array_equal(inp, out)
    return {'input': inp, 'output': out}


def _choose_layout(rng, regime, count, desired_spans):
    for attempt in range(100):
        if count == 1:
            rows, cols = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        else:
            rows, cols = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        allowed = _TRAIN_SPANS if regime == 'source_train' else _TEST_SPANS
        spans = []
        for j in range(count):
            target = desired_spans[j] if desired_spans and j < len(desired_spans) else None
            candidates = [s for s in allowed if s[0] <= rows and s[1] <= cols]
            if count > 1:
                candidates = [s for s in candidates if s[0]*s[1] <= (rows*cols)//2]
            if target in candidates:
                spans.append(target)
            else:
                spans.append(candidates[int(rng.integers(len(candidates)))])
        rects = _place(rng, rows, cols, spans)
        if rects is not None:
            return rows, cols, spans, rects
    rows = cols = 3
    spans = [(1, 1)] * count
    rects = _place(rng, rows, cols, spans)
    assert rects is not None
    return rows, cols, spans, rects


def _roles(rng, role_colors, variable, index):
    if not variable:
        return role_colors[:2]
    if index == 0:
        return role_colors[0], role_colors[1]
    if index == 1:
        return role_colors[1], role_colors[2]
    candidates = [(a, b) for a in role_colors for b in role_colors if a != b]
    return candidates[int(rng.integers(len(candidates)))]


def _profiles(rng, regime, spans):
    profiles = []
    for span in spans:
        if regime == 'source_train':
            options = ('sparse', 'sparse', 'mixed', 'run', 'centers')
        else:
            options = ('sparse', 'mixed', 'run', 'centers', 'original_bar') if span == (1, 2) else ('sparse', 'mixed', 'run', 'centers')
        profiles.append(options[int(rng.integers(len(options)))])
    return profiles


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    palette = tuple(int(x) for x in rng.permutation(10))
    role_colors, accent_colors = palette[:3], palette[3:]
    familiar, novel = accent_colors[:4], accent_colors[4:]
    variable_roles = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    train_color_spans = {color: set() for color in familiar}
    train_spans = set()
    for index in range(train_count + test_count):
        regime = 'source_train' if index < train_count else 'target_test'
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        if regime == 'source_train':
            count = 2 if index == 0 else int(pair_rng.integers(1, 3))
            colors = ([familiar[0], familiar[1]] if index == 0 else
                      [familiar[2 + j] for j in range(count)] if index == 1 else
                      [int(x) for x in pair_rng.choice(familiar, count, replace=False)])
            wanted = None
        else:
            count = int(pair_rng.choice((1, 2, 3), p=(.22, .55, .23)))
            modes = [int(pair_rng.integers(4)) for _ in range(count)]
            colors = []
            wanted = []
            for mode in modes:
                observed = [c for c in familiar if train_color_spans[c]]
                pool = observed if mode in (0, 2) else novel
                choices = [c for c in pool if c not in colors]
                if not choices:
                    choices = [c for c in accent_colors if c not in colors]
                color = choices[int(pair_rng.integers(len(choices)))]
                colors.append(color)
                if mode == 0:
                    shape_choices = [s for s in _TEST_SPANS if s not in train_color_spans[color]]
                elif mode == 1:
                    shape_choices = sorted(train_spans)
                elif mode == 2:
                    shape_choices = sorted(train_color_spans[color]) or sorted(train_spans)
                else:
                    shape_choices = [s for s in _TEST_SPANS if s not in train_spans]
                wanted.append(shape_choices[int(pair_rng.integers(len(shape_choices)))])
        for _ in range(30):
            rows, cols, spans, rects = _choose_layout(pair_rng, regime, count, wanted)
            if regime == 'source_train' and index == 0 and len(set(spans)) < 2:
                continue
            roles = _roles(pair_rng, role_colors, variable_roles, index)
            profiles = _profiles(pair_rng, regime, spans)
            clues = [_clues(pair_rng, rows, cols, rect, profile)
                     for rect, profile in zip(rects, profiles)]
            pair = _make_pair(rows, cols, roles, rects, colors, clues)
            try:
                _read_pair(pair['input'], pair['output'])
            except AssertionError:
                continue
            key = _grid_key(pair['input'])
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError('could not produce unique pair')
        if regime == 'source_train':
            puzzle['train'].append(pair)
            for color, span in zip(colors, spans):
                train_color_spans[color].add(span)
                train_spans.add(span)
        else:
            puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _read_pair(inp, out):
    assert inp.shape == out.shape
    height, width = inp.shape
    assert (height-1) % 4 == 0 and (width-1) % 4 == 0
    rows, cols = (height-1)//4, (width-1)//4
    carrier = int(inp[1, 1])
    interpretations = []
    for background in sorted(int(x) for x in np.unique(inp) if x != carrier):
        base = _base(rows, cols, background, carrier)
        if np.any(inp[base == carrier] != carrier):
            continue
        if np.any(inp[base == background] == carrier):
            continue
        accents = sorted(set(int(x) for x in np.unique(inp)) - {background, carrier})
        if not accents:
            continue
        expected = base.copy()
        objects = []
        occupied = set()
        try:
            for color in accents:
                clues = [tuple(map(int, cell)) for cell in np.argwhere(inp == color)]
                rect = _infer(rows, cols, clues)
                signature = _signature(rect)
                assert signature.isdisjoint(occupied)
                occupied.update(signature)
                assert set(clues) <= signature
                for r, c in signature:
                    expected[r, c] = color
                objects.append({'color': color, 'rect': rect,
                                'span': (rect[1]-rect[0], rect[3]-rect[2]),
                                'clues': clues,
                                'centers': len(set(clues) & _centers(rect)),
                                'perimeter': len(set(clues) & _perimeter(rect))})
            interpretations.append({'background': background, 'carrier': carrier,
                                    'rows': rows, 'cols': cols, 'objects': objects,
                                    'expected': expected})
        except AssertionError:
            pass
    assert len(interpretations) == 1, 'input lacks a unique structural interpretation'
    interpretation = interpretations[0]
    assert np.array_equal(interpretation.pop('expected'), out)
    assert not np.array_equal(inp, out)
    return interpretation


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            if inp.shape == (17, 17):
                digest = hashlib.sha256(inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()).hexdigest()
                assert digest not in _SOURCE_DIGESTS
            info = _read_pair(inp, out)
            if split == 'train':
                assert 1 <= len(info['objects']) <= 2
                assert all(obj['span'] in _TRAIN_SPANS for obj in info['objects'])
            else:
                assert 1 <= len(info['objects']) <= 3
                assert all(obj['span'] in _TEST_SPANS for obj in info['objects'])
            infos[split].append(info)
    train_spans = {obj['span'] for info in infos['train'] for obj in info['objects']}
    assert len(train_spans) >= 2
    assert any(len(info['objects']) == 2 for info in infos['train'])
    for info in infos['test']:
        assert info['objects']
    # Colors encode spatial roles within a pair, and are never required as a
    # hidden puzzle-wide lookup table. Each train output displays both parts
    # of the common perimeter-and-centers rule.
    for info in infos['train']:
        assert all(obj['perimeter'] + obj['centers'] == len(obj['clues'])
                   for obj in info['objects'])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x['input'], y['input']) and np.array_equal(x['output'], y['output'])
        for x, y in zip(a[s], b[s])) for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, sizes, spans, layouts, roles = set(), set(), set(), set(), set(), set()
    clue_types, test_counts_objects, color_modes = set(), set(), set()
    combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_info = [_read_pair(p['input'], p['output']) for p in puzzle['train']]
        test_info = [_read_pair(p['input'], p['output']) for p in puzzle['test']]
        known = {o['color'] for p in train_info for o in p['objects']}
        old_spans = {o['span'] for p in train_info for o in p['objects']}
        for split, entries in (('train', train_info), ('test', test_info)):
            for p in entries:
                sizes.add((4*p['rows']+1, 4*p['cols']+1))
                roles.add((p['background'], p['carrier']))
                layouts.add((split, p['rows'], p['cols'], tuple(o['rect'] for o in p['objects'])))
                if split == 'test':
                    test_counts_objects.add(len(p['objects']))
                for o in p['objects']:
                    spans.add(o['span'])
                    clue_types.add((split, bool(o['centers']), bool(o['perimeter'])))
                    if split == 'test':
                        color_modes.add((o['color'] in known, o['span'] in old_spans))
                        if o['centers'] and o['perimeter'] and len(p['objects']) > 1:
                            combined += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 15 and len(spans) >= 10 and len(layouts) >= 150
    assert len(roles) >= 15 and test_counts_objects == {1, 2, 3}
    assert len(color_modes) == 4 and combined > 5
    assert ('train', True, False) in clue_types and ('train', True, True) in clue_types
    assert ('test', True, False) in clue_types and ('test', True, True) in clue_types
    assert _equal(generate_puzzle(9137, 7, 3), generate_puzzle(9137, 7, 3))
    assert _equal(generate_puzzle(2718), generate_puzzle(2718))
    assert not _equal(generate_puzzle(2718), generate_puzzle(2719))
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(99, tr, te)
        assert tr is None or len(p['train']) == tr
        assert te is None or len(p['test']) == te
        validate_generated_puzzle(p)
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts accepted')
