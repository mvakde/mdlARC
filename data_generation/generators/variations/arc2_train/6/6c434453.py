import numpy as np


_RING_SHAPE = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)

_PLUS_SHAPE = (
            (0, 1),
    (1, 0), (1, 1), (1, 2),
            (2, 1),
)

_BLOCK_SHAPE = (
    (0, 0), (0, 1),
    (1, 0), (1, 1),
)

_NONPLUS_DISTRACTOR_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    _BLOCK_SHAPE,
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)

_ALL_DISTRACTOR_SHAPES = (_PLUS_SHAPE,) + _NONPLUS_DISTRACTOR_SHAPES

_SOURCE_PAIRS = (
    (
        (
            "0000111000",
            "1100101000",
            "1100111000",
            "0000000010",
            "0000000111",
            "0111000010",
            "0101000000",
            "0111001000",
            "0000001110",
            "0000000000",
        ),
        (
            "0000020000",
            "1100222000",
            "1100020000",
            "0000000010",
            "0000000111",
            "0020000010",
            "0222000000",
            "0020001000",
            "0000001110",
            "0000000000",
        ),
    ),
    (
        (
            "1110000000",
            "1010000100",
            "1110001110",
            "0000000100",
            "0100000000",
            "1110000000",
            "0101110000",
            "0001010011",
            "0001110000",
            "0000000000",
        ),
        (
            "0200000000",
            "2220000100",
            "0200001110",
            "0000000100",
            "0100000000",
            "1110000000",
            "0100200000",
            "0002220011",
            "0000200000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000010",
            "0000000111",
            "0111000010",
            "0101000000",
            "0111000000",
            "0000000000",
            "0000001100",
            "0111000000",
            "0101000011",
            "0111000011",
        ),
        (
            "0000000010",
            "0000000111",
            "0020000010",
            "0222000000",
            "0020000000",
            "0000000000",
            "0000001100",
            "0020000000",
            "0222000011",
            "0020000011",
        ),
    ),
)




def _normalize_cells(cells):
    cells = tuple(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair['input']), _grid_key(pair['output']))


def _matches_source_pair(inp, out):
    for old_in, old_out in _SOURCE_PAIRS:
        if inp.shape == (10, 10) and out.shape == (10, 10):
            if (tuple(''.join(str(int(x)) for x in row) for row in inp) == old_in
                    and tuple(''.join(str(int(x)) for x in row) for row in out) == old_out):
                return True
    return False


def _components(grid, background):
    """Four-connected, monochrome foreground objects; diagonal contact is free."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            visited[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                            and int(grid[yy, xx]) == color):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            result.append((color, tuple(sorted(cells))))
    return result


def _targets(grid, background):
    targets = []
    for color, cells in _components(grid, background):
        if len(cells) != 8 or _normalize_cells(cells) != _RING_SHAPE:
            continue
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        if int(grid[top + 1, left + 1]) == background:
            targets.append((top, left, color))
    return sorted(targets)


def _scan_exact(grid, background):
    h, w = grid.shape
    found = []
    for top in range(h - 2):
        for left in range(w - 2):
            color = int(grid[top, left])
            if color == background or int(grid[top + 1, left + 1]) != background:
                continue
            if all(int(grid[top + dr, left + dc]) == color for dr, dc in _RING_SHAPE):
                found.append((top, left, color))
    return sorted(found)


def _apply_rule(inp, background, highlight):
    out = inp.copy()
    for top, left, _ in _targets(inp, background):
        for dr, dc in ((0, 0), (0, 2), (2, 0), (2, 2)):
            out[top + dr, left + dc] = background
        for dr, dc in _PLUS_SHAPE:
            out[top + dr, left + dc] = highlight
    return out


def _ring_outline(h, w):
    return tuple((r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1))


def _shape(kind, rng):
    if kind == 'ring':
        return _RING_SHAPE
    if kind == 'plus':
        return _PLUS_SHAPE
    if kind == 'missing':
        removed = _RING_SHAPE[int(rng.integers(0, len(_RING_SHAPE)))]
        return tuple(cell for cell in _RING_SHAPE if cell != removed)
    if kind in ('filled', 'centered'):
        return tuple((r, c) for r in range(3) for c in range(3))
    if kind == 'large':
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        return _ring_outline(h, w)
    if kind == 'bar':
        n = int(rng.integers(2, 7))
        return tuple((0, i) for i in range(n)) if rng.integers(2) else tuple((i, 0) for i in range(n))
    if kind == 'block':
        return _BLOCK_SHAPE
    if kind == 'ell':
        n = int(rng.integers(2, 5))
        return tuple(sorted(set([(r, 0) for r in range(n)] + [(n - 1, c) for c in range(n)])))
    if kind == 'tee':
        return ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))
    if kind == 'zigzag':
        return ((0, 0), (0, 1), (1, 1), (1, 2))
    raise ValueError('unknown shape kind')


_NEAR_MISSES = ('missing', 'filled', 'centered', 'large')
_ORDINARY = ('plus', 'bar', 'block', 'ell', 'tee', 'zigzag')


def _valid_position(shape, top, left, painted, boxes):
    h, w = painted.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    bottom, right = top + max_r, left + max_c
    if top < 0 or left < 0 or bottom >= h or right >= w:
        return False
    for t, l, b, rr in boxes:
        if not (bottom < t or b < top or right < l or rr < left):
            return False
    for dr, dc in shape:
        r, c = top + dr, left + dc
        if painted[r, c]:
            return False
        for yy, xx in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= yy < h and 0 <= xx < w and painted[yy, xx]:
                return False
    return True


def _place(rng, grid, painted, boxes, shape, color, mode='random', anchor=None,
           center_color=None):
    h, w = grid.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    if sh > h or sw > w:
        return None
    candidates = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    rng.shuffle(candidates)
    if mode == 'edge':
        preferred = [p for p in candidates if p[0] == 0 or p[1] == 0
                     or p[0] + sh == h or p[1] + sw == w]
    elif mode == 'center':
        preferred = [p for p in candidates if abs(p[0] + sh / 2 - h / 2) <= h / 5
                     and abs(p[1] + sw / 2 - w / 2) <= w / 5]
    elif mode == 'same_row' and anchor is not None:
        preferred = [p for p in candidates if abs(p[0] - anchor[0]) <= 1]
    elif mode == 'same_col' and anchor is not None:
        preferred = [p for p in candidates if abs(p[1] - anchor[1]) <= 1]
    elif mode == 'cluster' and anchor is not None:
        preferred = sorted(candidates, key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]))[:max(20, len(candidates) // 4)]
    else:
        preferred = []
    # A preferred placement is tried first, while the complete shuffled list
    # provides a deterministic fallback for compact canvases.
    for top, left in preferred + candidates:
        if not _valid_position(shape, top, left, painted, boxes):
            continue
        for dr, dc in shape:
            r, c = top + dr, left + dc
            grid[r, c] = center_color if center_color is not None and (dr, dc) == (1, 1) else color
            painted[r, c] = True
        boxes.append((top, left, top + sh - 1, left + sw - 1))
        return top, left
    return None


def _choose_dimensions(rng, ring_count, required):
    bands = ('source', 'compact', 'medium', 'large', 'wide', 'tall')
    band = bands[int(rng.integers(len(bands)))]
    if band == 'source':
        h, w = 10, 10
    elif band == 'compact':
        h, w = int(rng.integers(8, 13)), int(rng.integers(8, 13))
    elif band == 'medium':
        h, w = int(rng.integers(12, 21)), int(rng.integers(12, 21))
    elif band == 'large':
        h, w = int(rng.integers(21, 31)), int(rng.integers(21, 31))
    elif band == 'wide':
        h, w = int(rng.integers(8, 14)), int(rng.integers(18, 31))
    else:
        h, w = int(rng.integers(18, 31)), int(rng.integers(8, 14))
    if ring_count >= 3 or required == 'large':
        h = max(h, 12)
        w = max(w, 12)
    return h, w


def _pair_colors(rng, highlight, previous_background=None):
    available = [x for x in range(10) if x != highlight]
    if previous_background is not None:
        backgrounds = [x for x in available if x != previous_background]
    else:
        backgrounds = available
    background = backgrounds[int(rng.integers(len(backgrounds)))]
    original = [x for x in available if x != background]
    primary = original[int(rng.integers(len(original)))]
    secondary_choices = [x for x in original if x != primary]
    secondary = secondary_choices[int(rng.integers(len(secondary_choices)))]
    return background, primary, secondary


def generate_pair(rng, regime, parameters, ring_count, required=None,
                  mode='random', previous_background=None, near_misses=()):
    if regime not in ('source_train', 'target_test'):
        raise ValueError('unknown regime')
    highlight = parameters['highlight']
    background, primary, secondary = _pair_colors(rng, highlight, previous_background)
    for attempt in range(45):
        h, w = _choose_dimensions(rng, ring_count, required) if attempt < 35 else (18, 18)
        grid = np.full((h, w), background, dtype=np.int8)
        painted = np.zeros((h, w), dtype=bool)
        boxes = []
        ring_positions = []
        multicolor = bool(rng.integers(0, 3) == 0)
        for index in range(ring_count):
            color = (secondary if multicolor and index % 2 else primary)
            ring_mode = mode if index == 0 else (
                mode if mode in ('same_row', 'same_col', 'cluster') else
                ('same_row', 'same_col', 'cluster', 'random')[int(rng.integers(4))])
            pos = _place(rng, grid, painted, boxes, _RING_SHAPE, color,
                         ring_mode, ring_positions[0] if ring_positions else None)
            if pos is None:
                break
            ring_positions.append(pos)
        if len(ring_positions) != ring_count:
            continue
        kinds = []
        if required is not None:
            kinds.append(required)
        if regime == 'target_test' and near_misses and rng.integers(0, 4) != 0:
            kind = near_misses[int(rng.integers(len(near_misses)))]
            if kind not in kinds:
                kinds.append(kind)
        extra_cap = min(5, (h * w) // 70 + 1)
        extra_count = int(rng.integers(0, extra_cap + 1))
        for _ in range(extra_count):
            kinds.append(_ORDINARY[int(rng.integers(len(_ORDINARY)))])
        # A second plus appears sometimes, but other pairs have none.
        if h * w > 170 and rng.integers(0, 7) == 0:
            kinds.extend(('plus', 'plus'))
        success = True
        for kind in kinds:
            shape = _shape(kind, rng)
            color = secondary if multicolor and rng.integers(2) else primary
            center_color = secondary if kind == 'centered' else None
            if kind == 'centered' and center_color == color:
                color = primary if center_color == secondary else secondary
            if _place(rng, grid, painted, boxes, shape, color,
                      center_color=center_color) is None:
                success = False
                break
        if not success:
            continue
        targets = _targets(grid, background)
        if len(targets) != ring_count or _scan_exact(grid, background) != targets:
            continue
        values, counts = np.unique(grid, return_counts=True)
        if int(values[int(np.argmax(counts))]) != background:
            continue
        return grid, _apply_rule(grid, background, highlight)
    raise RuntimeError('unable to construct pair')


def _checked_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _checked_count('num_train', num_train, 2, 10)
    test = _checked_count('num_test', num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    parameters = {'highlight': int(rng.integers(10))}
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    near_misses = [_NEAR_MISSES[int(rng.integers(len(_NEAR_MISSES)))]]
    if train_count >= 4 and rng.integers(2):
        other = [x for x in _NEAR_MISSES if x not in near_misses]
        near_misses.append(other[int(rng.integers(len(other)))])
    shifts = ('simple', 'count', 'edge', 'cluster', 'near_miss')
    shift = shifts[int(rng.integers(len(shifts)))]
    puzzle = {'train': [], 'test': []}
    seen = set()
    previous_background = None
    for section, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            required = None
            if section == 'train':
                if index == 0:
                    required = 'plus'
                elif index - 1 < len(near_misses):
                    required = near_misses[index - 1]
                ring_count = (1, 2, 3)[index] if index < 3 else int(rng.integers(1, 4))
                mode = ('random', 'center', 'edge', 'same_row', 'same_col', 'cluster')[int(rng.integers(6))]
            else:
                if shift == 'near_miss' and index == 0:
                    required = near_misses[0]
                if index == 0 and (train_count == 2 or shift == 'count'):
                    ring_count = 3
                elif index == 0 and shift == 'simple':
                    ring_count = 2
                else:
                    ring_count = int(rng.integers(1, 5))
                mode = {'edge': 'edge', 'cluster': 'cluster'}.get(shift, 'random') if index == 0 else ('random', 'edge', 'center', 'same_row', 'same_col', 'cluster')[int(rng.integers(6))]
            for _ in range(16):
                child_seed = int(rng.integers(0, 2**63, dtype=np.int64))
                child = np.random.default_rng(child_seed)
                inp, out = generate_pair(child, 'source_train' if section == 'train' else 'target_test',
                                         parameters, ring_count, required, mode, previous_background if section == 'train' and index == 1 else None,
                                         near_misses)
                key = _grid_key(inp)
                if key not in seen and not _matches_source_pair(inp, out):
                    break
            else:
                raise RuntimeError('unable to generate unique pair')
            pair = {'input': inp, 'output': out}
            seen.add(key)
            puzzle[section].append(pair)
            if section == 'train' and index == 0:
                previous_background = int(np.unique(inp, return_counts=True)[0][np.argmax(np.unique(inp, return_counts=True)[1])])
    validate_generated_puzzle(puzzle)
    return puzzle


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _near_miss_types(grid, background):
    found = set()
    for color, cells in _components(grid, background):
        shape = _normalize_cells(cells)
        if len(cells) == 7 and any(shape == _normalize_cells(x for x in _RING_SHAPE if x != removed)
                                   for removed in _RING_SHAPE):
            found.add('missing')
        if shape == tuple((r, c) for r in range(3) for c in range(3)):
            found.add('filled')
        if any(shape == _ring_outline(h, w) for h in range(4, 7) for w in range(4, 7)):
            found.add('large')
        if shape == _RING_SHAPE:
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            center = int(grid[top + 1, left + 1])
            if center != background and center != color:
                found.add('centered')
    return found


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs = set()
    seen_pairs = set()
    highlights = set()
    train_counts = set()
    all_counts = set()
    train_near = set()
    test_near = set()
    plus_in_train = False
    train_backgrounds = set()
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert inp.shape == out.shape
            ik, pk = _grid_key(inp), _pair_key(pair)
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            assert not _matches_source_pair(inp, out)
            background = _background(inp)
            input_colors = set(int(x) for x in np.unique(inp))
            output_colors = set(int(x) for x in np.unique(out))
            new = output_colors - input_colors
            assert len(new) == 1
            highlight = next(iter(new))
            highlights.add(highlight)
            assert background in input_colors and highlight not in input_colors
            targets = _targets(inp, background)
            assert targets == _scan_exact(inp, background)
            assert 1 <= len(targets) <= 4
            all_counts.add(len(targets))
            assert np.array_equal(out, _apply_rule(inp, background, highlight))
            assert not np.array_equal(inp, out)
            near = _near_miss_types(inp, background)
            if section == 'train':
                train_counts.add(len(targets))
                train_near.update(near)
                train_backgrounds.add(background)
                for color, cells in _components(inp, background):
                    if _normalize_cells(cells) == _PLUS_SHAPE:
                        if all(int(out[r, c]) == color for r, c in cells):
                            plus_in_train = True
            else:
                test_near.update(near)
    assert len(highlights) == 1
    assert {1, 2}.issubset(train_counts)
    assert {1, 2, 3}.issubset(all_counts)
    assert len(train_backgrounds) >= 2
    assert plus_in_train
    assert train_near and test_near.issubset(train_near)


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x['input'], y['input']) and np.array_equal(x['output'], y['output'])
        for x, y in zip(a[s], b[s])) for s in ('train', 'test'))


def self_test() -> None:
    # The source edge and diagonal cases directly test the component rule.
    for old_in, old_out in _SOURCE_PAIRS:
        inp = np.array([[int(x) for x in row] for row in old_in], dtype=np.int8)
        out = np.array([[int(x) for x in row] for row in old_out], dtype=np.int8)
        assert np.array_equal(_apply_rule(inp, 0, 2), out)
        assert len(_targets(inp, 0)) == 2
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(314160, 7, 3))
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                (None, 4), (2, None), (10, None))):
        p = generate_puzzle(5000 + i, *counts)
        if counts[0] is not None:
            assert len(p['train']) == counts[0]
        if counts[1] is not None:
            assert len(p['test']) == counts[1]
        validate_generated_puzzle(p)
    for i, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                (2, 2), (3, 4))):
        try:
            generate_puzzle(6000 + i, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p['train']) for p in puzzles}
    test_counts = {len(p['test']) for p in puzzles}
    shapes = set()
    layouts = set()
    ring_counts = set()
    near = set()
    highlights = set()
    edge = 0
    multi_colors = 0
    coverage = {name: 0 for name in (
        'source_size', 'compact', 'large', 'axis_30', 'elongated',
        'central', 'same_row', 'same_side', 'clustered', 'diagonal',
        'no_plus', 'many_plus', 'two_bar', 'multi_target_colors',
    )}
    edge_sides = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        assert len({len(_targets(pair['input'], _background(pair['input'])))
                    for section in ('train', 'test') for pair in puzzle[section]}) >= 3
        highlights.add(next(iter(
            set(int(x) for x in np.unique(puzzle['train'][0]['output']))
            - set(int(x) for x in np.unique(puzzle['train'][0]['input']))
        )))
        for section in ('train', 'test'):
            for pair in puzzle[section]:
                inp, out = pair['input'], pair['output']
                bg = _background(inp)
                targets = _targets(inp, bg)
                assert np.array_equal(_apply_rule(inp, bg,
                    next(iter(set(int(x) for x in np.unique(out)) - set(int(x) for x in np.unique(inp))))), out)
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple((r, c) for r, c, _ in targets),
                             tuple(sorted(_normalize_cells(cells) for _, cells in _components(inp, bg)))))
                ring_counts.add(len(targets))
                near.update(_near_miss_types(inp, bg))
                multi_colors += len(np.unique(inp)) >= 3
                edge += any(r == 0 or c == 0 or r + 3 == inp.shape[0] or c + 3 == inp.shape[1]
                            for r, c, _ in targets)
                h, w = inp.shape
                coverage['source_size'] += (h, w) == (10, 10)
                coverage['compact'] += h <= 12 and w <= 12
                coverage['large'] += max(h, w) >= 21
                coverage['axis_30'] += max(h, w) == 30
                coverage['elongated'] += max(h, w) / min(h, w) >= 1.5
                coverage['central'] += any(h // 3 <= r + 1 < 2 * h // 3
                                           and w // 3 <= c + 1 < 2 * w // 3
                                           for r, c, _ in targets)
                coverage['same_row'] += any(abs(a[0] - b[0]) <= 1
                                            for i, a in enumerate(targets)
                                            for b in targets[i + 1:])
                coverage['same_side'] += (len(targets) >= 2 and
                                          (all(c + 1 < w / 2 for _, c, _ in targets)
                                           or all(c + 1 >= w / 2 for _, c, _ in targets)))
                coverage['clustered'] += any(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 6
                                             for i, a in enumerate(targets)
                                             for b in targets[i + 1:])
                for r, c, _ in targets:
                    if r == 0:
                        edge_sides.add('top')
                    if c == 0:
                        edge_sides.add('left')
                    if r + 3 == h:
                        edge_sides.add('bottom')
                    if c + 3 == w:
                        edge_sides.add('right')
                components = _components(inp, bg)
                ring_cells = {(r + dr, c + dc) for r, c, _ in targets
                              for dr, dc in _RING_SHAPE}
                other_cells = {cell for _, cells in components for cell in cells
                               if cell not in ring_cells}
                coverage['diagonal'] += any((r + dr, c + dc) in other_cells
                                            for r, c in ring_cells
                                            for dr in (-1, 1) for dc in (-1, 1))
                plus_count = sum(_normalize_cells(cells) == _PLUS_SHAPE
                                 for _, cells in components)
                coverage['no_plus'] += plus_count == 0
                coverage['many_plus'] += plus_count >= 2
                coverage['two_bar'] += any(len(cells) == 2
                                           and len({r for r, _ in cells}) == 1
                                           for _, cells in components)
                coverage['multi_target_colors'] += len({color for _, _, color in targets}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 100
    assert {1, 2, 3, 4}.issubset(ring_counts)
    assert near == set(_NEAR_MISSES)
    assert len(highlights) >= 8 and edge >= 10 and multi_colors >= 20
    assert edge_sides == {'top', 'bottom', 'left', 'right'}
    for name, minimum in (
        ('source_size', 10), ('compact', 50), ('large', 50),
        ('axis_30', 5), ('elongated', 50), ('central', 20),
        ('same_row', 20), ('same_side', 20), ('clustered', 20),
        ('diagonal', 10), ('no_plus', 50), ('many_plus', 10),
        ('two_bar', 3), ('multi_target_colors', 10),
    ):
        assert coverage[name] >= minimum, (name, coverage[name])
