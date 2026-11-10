"""Contextual stencil completion puzzles."""

import numpy as np

_SOURCE_INPUTS = {
    ''.join(('0555500505','5050000555','5555505005','5055500055','5555005055','5222250000','0555555055','0055500550','5055050505','5505055555')),
    ''.join(('0555002555','0500025205','0550002050','5055550505','5000050050','5505550055','0000000500','0505500005','5005005055','0005500550')),
    ''.join(('5555055555','5555055505','5050550555','5050550055','5000055505','5550505005','0500505555','5550005250','0555505250','5000005225')),
    ''.join(('0555055550','5550555505','0055550500','0055550555','0055522005','5500022555','0055022555','0555555000','5500555000','5050500500')),
}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError('invalid pair count')
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('train count must exceed test count')
    return int(num_train), int(num_test)


def _motif(rng, kind):
    if kind == 'bar':
        n = int(rng.integers(4, 7))
        vertical = bool(rng.integers(2))
        stencil = {(i, 0) if vertical else (0, i) for i in range(n)}
        context = {(n - 2, 1), (n - 1, 1)} if vertical else {(1, n - 2), (1, n - 1)}
    elif kind == 'bend':
        leg, foot = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        stencil = {(r, 0) for r in range(leg)} | {(leg - 1, c) for c in range(foot)}
        context = {(r, 1) for r in range(leg - 1)}
    elif kind == 'zigzag':
        h = int(rng.integers(3, 5))
        stencil = {(r, min(2, r // 2)) for r in range(h)} | {(h - 1, 2)}
        context = {(0, 1), (h - 2, 2)} - stencil
    elif kind == 'arms':
        gap = int(rng.integers(1, 3))
        stencil = {(0, gap), (gap, 0), (gap, 2 * gap), (2 * gap, gap)}
        context = {(gap, gap)}
    elif kind == 'hollow':
        h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        stencil = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
        context = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    else:
        h, w = ((2, 3), (3, 2), (2, 4), (3, 3))[int(rng.integers(4))]
        stencil = {(r, c) for r in range(h) for c in range(w)}
        context = set()
    return frozenset(stencil), frozenset(context)


def _extent(stencil, context):
    cells = stencil | context
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _forms(stencil, context):
    _, w = _extent(stencil, context)
    left = (stencil, context)
    right = (frozenset((r, w - 1 - c) for r, c in stencil), frozenset((r, w - 1 - c) for r, c in context))
    return (left,) if left[0] == right[0] else (left, right)


def _cells(relative, row, col):
    return {(row + r, col + c) for r, c in relative}


def _field_cues(stencil):
    if _kind_for_shape(stencil) == 'arms':
        h = max(r for r, _ in stencil) + 1
        return {(h // 2, -1)}
    return set()


def _scan(grid, stencil, context, field, structure, marker):
    h, w = _extent(stencil, context)
    field_cues = _field_cues(stencil)
    forms = _forms(stencil, context)
    hits = []
    for flip, (s, c) in enumerate(forms):
        for row in range(grid.shape[0] - h + 1):
            for col in range(grid.shape[1] - w + 1):
                if all(grid[row + r, col + q] == structure for r, q in c) and all(grid[row + r, col + q] in (field, marker) for r, q in s) and all(col + q < 0 or grid[row + r, col + q] in (field, marker) for r, q in field_cues):
                    hits.append((row, col, flip))
    if len(forms) > 1:
        direct = [_cells(forms[0][0], r, c) for r, c, flip in hits if flip == 0]
        hits = [(r, c, flip) for r, c, flip in hits if flip == 0 or not any(_cells(forms[flip][0], r, c) & cells for cells in direct)]
    return hits


def _kind_for_shape(stencil):
    h = max(r for r, _ in stencil) + 1
    w = max(c for _, c in stencil) + 1
    if h == 1 or w == 1:
        return 'bar'
    if len(stencil) == h * w:
        return 'filled'
    if len(stencil) == 4 and h == w and h in (3, 5) and {(0, w // 2), (h // 2, 0), (h // 2, w - 1), (h - 1, w // 2)} == set(stencil):
        return 'arms'
    if len(stencil) == 2 * h + 2 * w - 4 and all((r, c) in stencil for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)):
        return 'hollow'
    if all((r, 0) in stencil for r in range(h)) and all((h - 1, c) in stencil for c in range(w)):
        return 'bend'
    return 'zigzag'


def _context_for_shape(stencil):
    kind = _kind_for_shape(stencil)
    h = max(r for r, _ in stencil) + 1
    w = max(c for _, c in stencil) + 1
    if kind == 'bar':
        return frozenset({(h - 2, 1), (h - 1, 1)} if w == 1 else {(1, w - 2), (1, w - 1)})
    if kind == 'bend':
        return frozenset((r, 1) for r in range(h - 1))
    if kind == 'zigzag':
        return frozenset({(0, 1), (h - 2, 2)} - set(stencil))
    if kind == 'arms':
        return frozenset({(h // 2, w // 2)})
    if kind == 'hollow':
        return frozenset((r, c) for r in range(1, h - 1) for c in range(1, w - 1))
    return frozenset()


def _random_place(rng, height, width, h, w, edge):
    if edge and rng.random() < 0.50:
        side = int(rng.integers(4))
        if side == 0:
            return 0, int(rng.integers(width - w + 1))
        if side == 1:
            return height - h, int(rng.integers(width - w + 1))
        if side == 2:
            return int(rng.integers(height - h + 1)), 0
        return int(rng.integers(height - h + 1)), width - w
    if not edge and height - h > 1 and width - w > 1 and rng.random() < 0.75:
        return int(rng.integers(1, height - h)), int(rng.integers(1, width - w))
    return int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))


def _make_pair(rng, roles, kind, regime, force_mirror=False, copy_count=None):
    field, structure, marker = roles
    stencil, context = _motif(rng, kind)
    h, w = _extent(stencil, context)
    forms = _forms(stencil, context)
    for _ in range(150):
        height = int(rng.integers(max(10, h + 5), 23))
        width = int(rng.integers(max(10, w + 5), 23))
        count = int(rng.integers(1, 5)) if copy_count is None else copy_count
        if copy_count is None and height * width < 150:
            count = min(count, 3)
        constraints = {}
        plants = []
        exemplar = _random_place(rng, height, width, h, w, True)
        if kind == 'arms' and exemplar[1] == 0:
            continue
        edge_mode = regime == 'target_test' and rng.random() < 0.65
        for index in range(count + 1):
            choice = None
            for trial in range(180):
                if index == 0:
                    row, col, flip = *exemplar, 0
                else:
                    row, col = _random_place(rng, height, width, h, w, edge_mode if regime == 'target_test' else rng.random() < 0.35)
                    flip = int(rng.integers(len(forms)))
                    if force_mirror and index == 1 and len(forms) > 1:
                        flip = 1
                s, c = forms[flip]
                scells, ccells = _cells(s, row, col), _cells(c, row, col)
                cue_cells = {(row + dr, col + dc) for dr, dc in _field_cues(stencil) if col + dc >= 0}
                if (row, col, flip) in plants or any(constraints.get(p) == structure for p in scells | cue_cells) or any(constraints.get(p) == field for p in ccells):
                    continue
                if index and scells & _cells(forms[0][0], *exemplar):
                    continue
                choice = (row, col, flip, scells, ccells)
                break
            if choice is None:
                break
            row, col, flip, scells, ccells = choice
            plants.append((row, col, flip))
            constraints.update({p: field for p in scells})
            constraints.update({p: structure for p in ccells})
            constraints.update({p: field for p in cue_cells})
        if len(plants) != count + 1:
            continue
        # Add incomplete copies whose missing stencil or structural cue is visible.
        for _ in range(int(rng.integers(1, 4))):
            for _ in range(80):
                row, col = _random_place(rng, height, width, h, w, True)
                flip = int(rng.integers(len(forms)))
                s, c = forms[flip]
                scells, ccells = _cells(s, row, col), _cells(c, row, col)
                if any(p in constraints for p in scells | ccells):
                    continue
                if c and rng.random() < 0.5:
                    missing = tuple(ccells)[int(rng.integers(len(ccells)))]
                    constraints.update({p: field for p in scells})
                    constraints.update({p: structure for p in ccells})
                    constraints[missing] = field
                else:
                    missing = tuple(scells)[int(rng.integers(len(scells)))]
                    constraints.update({p: field for p in scells})
                    constraints.update({p: structure for p in ccells})
                    constraints[missing] = structure
                break
        density = float(rng.uniform(0.53, 0.79))
        for _ in range(18):
            grid = np.where(rng.random((height, width)) < density, structure, field).astype(np.int8)
            for (r, c), color in constraints.items():
                grid[r, c] = color
            for r, c in _cells(forms[0][0], *exemplar):
                grid[r, c] = marker
            hits = _scan(grid, stencil, context, field, structure, marker)
            if set(hits) != set(plants) or len(hits) != len(plants):
                continue
            output = grid.copy()
            for row, col, flip in hits:
                for r, c in _cells(forms[flip][0], row, col):
                    output[r, c] = marker
            if not np.array_equal(grid, output):
                return {'input': grid, 'output': output}
    raise RuntimeError('could not construct motif pair')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    roles = tuple(int(x) for x in rng.permutation(10)[:3])
    train_count, test_count = _counts(rng, num_train, num_test)
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_kind = ('bend', 'zigzag')[int(rng.integers(2))]
    remaining = [kind for kind in ('bar', 'zigzag', 'hollow', 'filled', 'bend', 'arms') if kind != first_kind]
    rng.shuffle(remaining)
    train_kinds = [first_kind] + remaining[:train_count - 1]
    while len(train_kinds) < train_count:
        train_kinds.append(('bar', 'zigzag', 'hollow', 'filled', 'bend', 'arms')[int(rng.integers(6))])
    test_kinds = sorted(set(train_kinds) | {'filled', 'hollow'})
    for regime, kinds, name in (('source_train', train_kinds, 'train'), ('target_test', [test_kinds[int(rng.integers(len(test_kinds)))] for _ in range(test_count)], 'test')):
        for index, kind in enumerate(kinds):
            forced_count = None
            if regime == 'source_train' and index == 1:
                first = puzzle['train'][0]
                first_roles, first_stencil, first_context, _ = _features(first)
                first_count = len(_scan(first['input'], first_stencil, first_context, *first_roles)) - 1
                forced_count = int(rng.choice([n for n in (1, 2, 3, 4) if n != first_count]))
            for _ in range(12):
                pair = _make_pair(rng, roles, kind, regime, force_mirror=(regime == 'source_train' and index == 0), copy_count=forced_count)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[name].append(pair)
                    break
            else:
                raise RuntimeError('could not obtain unique input')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    grid, output = pair['input'], pair['output']
    changed = grid != output
    assert np.any(changed)
    old, new = np.unique(grid[changed]), np.unique(output[changed])
    assert len(old) == len(new) == 1
    field, marker = int(old[0]), int(new[0])
    assert field != marker
    colors = set(np.unique(grid).tolist())
    assert len(colors) == 3 and field in colors and marker in colors
    structure = (colors - {field, marker}).pop()
    assert np.array_equal(grid == structure, output == structure)
    assert np.all(output[grid == marker] == marker)
    marked = np.argwhere(grid == marker)
    top, left = marked.min(axis=0)
    stencil = frozenset((int(r - top), int(c - left)) for r, c in marked)
    context = _context_for_shape(stencil)
    return (field, structure, marker), stencil, context, (int(top), int(left))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    common_roles = None
    mirror_seen = False
    for regime, pairs in (('source_train', train), ('target_test', test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = pair['input'], pair['output']
            for a in (grid, output):
                assert isinstance(a, np.ndarray) and a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            assert grid.shape == output.shape and all(10 <= n <= 22 for n in grid.shape)
            key = (grid.shape, grid.tobytes())
            assert key not in seen
            seen.add(key)
            if grid.shape == (10, 10):
                assert ''.join(str(int(x)) for x in grid.flat) not in _SOURCE_INPUTS
            roles, stencil, context, origin = _features(pair)
            if common_roles is None:
                common_roles = roles
            assert roles == common_roles
            field, structure, marker = roles
            forms = _forms(stencil, context)
            hits = _scan(grid, stencil, context, field, structure, marker)
            assert (origin[0], origin[1], 0) in hits and len(hits) >= 2
            expected = grid.copy()
            for row, col, flip in hits:
                cells = _cells(forms[flip][0], row, col)
                marked_count = sum(grid[r, c] == marker for r, c in cells)
                assert marked_count in (0, len(stencil))
                if marked_count:
                    assert (row, col, flip) == (origin[0], origin[1], 0)
                elif flip:
                    mirror_seen = mirror_seen or regime == 'source_train'
                for r, c in cells:
                    expected[r, c] = marker
            assert np.array_equal(output, expected)
    assert mirror_seen


def _same_puzzle(first, second):
    return all(len(first[k]) == len(second[k]) and all(np.array_equal(a[x], b[x]) for a, b in zip(first[k], second[k]) for x in ('input', 'output')) for k in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert all(_same_puzzle(generate_puzzle(s), generate_puzzle(s)) for s in (0, 7, 19))
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 3), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(91, a, b)
        validate_generated_puzzle(p)
        assert (a is None or len(p['train']) == a) and (b is None or len(p['test']) == b)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(91, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    kinds = {_kind_for_shape(_features(pair)[1]) for p in puzzles for pair in p['train'] + p['test']}
    sizes = {pair['input'].shape for p in puzzles for pair in p['train'] + p['test']}
    palettes = {_features(p['train'][0])[0] for p in puzzles}
    copy_counts = {len(_scan(pair['input'], _features(pair)[1], _features(pair)[2], *_features(pair)[0])) - 1 for p in puzzles for pair in p['train'] + p['test']}
    assert len(kinds) == 6 and len(sizes) >= 30 and len(palettes) >= 25
    assert copy_counts >= {1, 2, 3, 4}
    mirror_pairs = overlap_pairs = new_test_kinds = 0
    exemplar_edges = hidden_edges = interior_exemplars = interior_hidden = 0
    for puzzle in puzzles:
        puzzle_counts = set()
        train_kinds = {_kind_for_shape(_features(pair)[1]) for pair in puzzle['train']}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                roles, stencil, context, origin = _features(pair)
                forms = _forms(stencil, context)
                hidden = [place for place in _scan(grid, stencil, context, *roles) if place != (origin[0], origin[1], 0)]
                puzzle_counts.add(len(hidden))
                h, w = _extent(stencil, context)
                on_edge = lambda r, c: r == 0 or c == 0 or r + h == grid.shape[0] or c + w == grid.shape[1]
                exemplar_edges += on_edge(*origin)
                interior_exemplars += not on_edge(*origin)
                hidden_edges += any(on_edge(r, c) for r, c, _ in hidden)
                interior_hidden += all(not on_edge(r, c) for r, c, _ in hidden)
                mirror_pairs += any(f == 1 for _, _, f in hidden)
                groups = [_cells(forms[f][0], r, c) for r, c, f in hidden]
                overlap_pairs += any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups)))
                new_test_kinds += split == 'test' and _kind_for_shape(stencil) not in train_kinds
        assert len(puzzle_counts) >= 2
    assert mirror_pairs >= 30 and overlap_pairs >= 5 and new_test_kinds >= 3
    assert min(exemplar_edges, interior_exemplars, hidden_edges, interior_hidden) >= 20
