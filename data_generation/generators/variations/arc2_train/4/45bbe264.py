"""Sparse colored markers extended into full-grid crosses."""
import numpy as np

_SOURCE = ((14,14,((5,9,3),(10,3,4))), (16,16,((2,4,8),(9,10,5))),
           (13,13,((2,11,3),(4,2,5),(11,6,7))),
           (15,15,((2,3,8),(6,9,4),(12,6,1))))


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} out of range')
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    train = None if num_train is None else _count(num_train, 'num_train', 2, 10)
    test = None if num_test is None else _count(num_test, 'num_test', 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _signature(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _is_source_input(grid):
    for h, w, markers in _SOURCE:
        if grid.shape == (h, w):
            source = np.zeros((h, w), dtype=np.int8)
            for r, c, color in markers:
                source[r, c] = color
            if np.array_equal(grid, source):
                return True
    return False


def _output(shape, markers, background, highlight):
    result = np.full(shape, background, dtype=np.int8)
    for r, c, color in markers:
        result[r, :] = color
        result[:, c] = color
    for i, (r, _, _) in enumerate(markers):
        for j, (_, c, _) in enumerate(markers):
            if i != j:
                result[r, c] = highlight
    for r, c, color in markers:
        result[r, c] = color
    return result


def _marker_count(rng):
    return int(rng.choice([1,2,2,2,2,2,2,2,2,
                           3,3,3,3,3,3,3,3,4,4,4,5,5,6,7,8]))


def _shape(rng, count):
    minimum = max(5, count + 2)
    style = int(rng.choice(6, p=[.16,.28,.19,.14,.14,.09]))
    if style == 0:
        return (int(rng.integers(minimum, 11)), int(rng.integers(minimum, 11)))
    if style == 1:
        return (int(rng.integers(10, 21)), int(rng.integers(10, 21)))
    if style == 2:
        return (int(rng.integers(21, 31)), int(rng.integers(21, 31)))
    if style in (3, 4):
        short = int(rng.integers(minimum, 13))
        long = int(rng.integers(20, 31))
        return (short, long) if style == 3 else (long, short)
    side = int(rng.integers(max(13, minimum), 31))
    return (side, min(30, max(minimum, side + int(rng.integers(-2, 3)))))


def _test_shape(rng, count, training):
    minimum = max(5, count + 2)
    lo_h, hi_h = min(s[0] for s in training), max(s[0] for s in training)
    lo_w, hi_w = min(s[1] for s in training), max(s[1] for s in training)
    style = int(rng.choice(6, p=[.22,.18,.13,.15,.15,.17]))
    if style == 1:
        choices = [s for s in training if min(s) >= minimum]
        if choices:
            return choices[int(rng.integers(len(choices)))]
    if style == 2 and hi_h - lo_h >= 2 and hi_w - lo_w >= 2 and hi_h > minimum and hi_w > minimum:
        return (int(rng.integers(max(minimum, lo_h + 1), hi_h)),
                int(rng.integers(max(minimum, lo_w + 1), hi_w)))
    if style == 3 and lo_h > minimum and lo_w > minimum:
        return (int(rng.integers(minimum, lo_h)), int(rng.integers(minimum, lo_w)))
    if style == 4 and hi_h < 30 and hi_w < 30:
        return (int(rng.integers(max(minimum, hi_h + 1), 31)),
                int(rng.integers(max(minimum, hi_w + 1), 31)))
    if style == 5:
        if hi_h < 30 and lo_w > minimum:
            return (int(rng.integers(max(minimum, hi_h + 1), 31)),
                    int(rng.integers(minimum, lo_w)))
        if lo_h > minimum and hi_w < 30:
            return (int(rng.integers(minimum, lo_h)),
                    int(rng.integers(max(minimum, hi_w + 1), 31)))
    return _shape(rng, count)


def _training_shapes(rng, counts):
    if rng.random() >= .45:
        return [_shape(rng, count) for count in counts]
    low_h, low_w = int(rng.integers(9, 14)), int(rng.integers(9, 14))
    high_h = min(24, low_h + int(rng.integers(6, 11)))
    high_w = min(24, low_w + int(rng.integers(6, 11)))
    return [(int(rng.integers(max(low_h, count + 2), high_h + 1)),
             int(rng.integers(max(low_w, count + 2), high_w + 1)))
            for count in counts]


def _indices(rng, length, count, style):
    if style == 1:
        span = min(length, count + int(rng.integers(0, max(2, count // 2 + 3))))
        start = int(rng.integers(length - span + 1))
        available = np.arange(start, start + span)
    elif style == 2:
        chosen = np.clip(np.rint(np.linspace(1, length - 2, count)).astype(int)
                         + rng.integers(-1, 2, size=count), 0, length - 1)
        if len(set(chosen.tolist())) == count:
            return np.sort(chosen)
        available = np.arange(length)
    elif style == 3:
        edge = 0 if int(rng.integers(2)) == 0 else length - 1
        rest = rng.choice([x for x in range(length) if x != edge], count - 1, replace=False)
        return np.sort(np.append(rest, edge))
    elif style == 4 and count >= 2:
        start = int(rng.integers(length - 1))
        rest = rng.choice([x for x in range(length) if x not in (start, start + 1)],
                          count - 2, replace=False)
        return np.sort(np.append(rest, (start, start + 1)))
    else:
        available = np.arange(length)
    return np.sort(rng.choice(available, count, replace=False))


def _positions(rng, shape, count):
    h, w = shape
    for _ in range(64):
        row_style, column_style = (2, 2) if rng.random() < .25 else \
                                  (int(rng.integers(5)), int(rng.integers(5)))
        rows = _indices(rng, h, count, row_style)
        columns = _indices(rng, w, count, column_style)
        for _ in range(128):
            assigned = rng.permutation(columns)
            if all(rows[i+1]-rows[i] > 1 or abs(int(assigned[i+1])-int(assigned[i])) > 1
                   for i in range(count-1)):
                return [(int(r), int(c)) for r, c in zip(rows, assigned)]
    # A fixed sparse fallback makes the construction bounded for every legal size.
    rows = list(range(count))
    columns = list(range(0, 2*count, 2)) if w >= 2*count-1 else list(range(count))
    if columns[-1] >= w:
        columns = list(range(count))
    def search(prefix, remaining):
        if not remaining:
            return prefix
        for c in remaining:
            if not prefix or abs(prefix[-1]-c) > 1:
                found = search(prefix+[c], [x for x in remaining if x != c])
                if found is not None:
                    return found
        return None
    assigned = search([], columns)
    if assigned is None:
        raise ValueError('could not place isolated markers')
    return list(zip(rows, assigned))


def _train_colors(rng, palette, counts):
    slots = [pair for pair, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(slots)
    colors = [[] for _ in counts]
    for color, pair in zip(palette, slots):
        colors[pair].append(color)
    for pair, count in enumerate(counts):
        while len(colors[pair]) < count:
            options = [color for color in palette if color not in colors[pair]]
            colors[pair].append(int(rng.choice(options)))
        rng.shuffle(colors[pair])
    return colors


def _test_colors(rng, count, palette, highlight):
    novel = [c for c in range(10) if c != highlight and c not in palette]
    style = int(rng.choice(4, p=[.32,.24,.24,.20]))
    target = 0 if style == 0 else count if style == 1 else 1 if style == 2 else int(rng.integers(count+1))
    n = max(count-len(palette), min(target, count, len(novel)))
    return [int(c) for c in list(rng.choice(novel, n, replace=False))
            + list(rng.choice(palette, count-n, replace=False))]


def _pair(rng, shape, colors, highlight, preferred_background=None):
    choices = [c for c in range(10) if c != highlight and c not in colors]
    if preferred_background in choices:
        background = preferred_background
    else:
        background = 0 if 0 in choices and rng.random() < .48 else int(rng.choice(choices))
    positions = _positions(rng, shape, len(colors))
    colors = list(colors)
    rng.shuffle(colors)
    markers = [(r,c,color) for (r,c),color in zip(positions, colors)]
    inp = np.full(shape, background, dtype=np.int8)
    for r,c,color in markers:
        inp[r,c] = color
    return {'input': inp, 'output': _output(shape, markers, background, highlight)}


def _validate_grid(grid, label):
    _check(isinstance(grid, np.ndarray), f'{label} must be an ndarray')
    _check(grid.ndim == 2, f'{label} must have two dimensions')
    _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
           f'{label} has invalid dimensions')
    _check(np.issubdtype(grid.dtype, np.integer), f'{label} has invalid dtype')
    _check(bool(np.all((grid >= 0) & (grid <= 9))), f'{label} has invalid colors')


def _read_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    coords = np.argwhere(grid != background)
    _check(1 <= len(coords) <= 8, 'input needs one to eight markers')
    _check(int(np.sum(grid == background)) > len(coords), 'background is not dominant')
    markers = [(int(r), int(c), int(grid[r,c])) for r,c in coords]
    _check(len({r for r,_,_ in markers}) == len(markers), 'markers share a row')
    _check(len({c for _,c,_ in markers}) == len(markers), 'markers share a column')
    _check(len({color for _,_,color in markers}) == len(markers), 'marker colors repeat')
    for i, (r,c,_) in enumerate(markers):
        for rr,cc,_ in markers[i+1:]:
            _check(max(abs(r-rr), abs(c-cc)) > 1, 'markers touch')
    return background, markers


def _crossing_color(output, markers):
    colors = [int(output[r,c]) for i,(r,_,_) in enumerate(markers)
              for j,(_,c,_) in enumerate(markers) if i != j]
    if not colors:
        return None
    _check(len(set(colors)) == 1, 'crossings use different colors')
    return colors[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train','test'},
           'puzzle keys must be train and test')
    _check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list),
           'train/test must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'invalid pair counts')
    seen_inputs, seen_pairs = set(), set()
    records = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for index, pair in enumerate(puzzle[section]):
            _check(isinstance(pair, dict) and set(pair) == {'input','output'},
                   f'{section} {index} has invalid keys')
            inp, out = pair['input'], pair['output']
            _validate_grid(inp, f'{section} input {index}')
            _validate_grid(out, f'{section} output {index}')
            _check(inp.shape == out.shape, 'input/output shape differs')
            inp_sig = _signature(inp)
            pair_sig = (inp_sig, _signature(out))
            _check(inp_sig not in seen_inputs and pair_sig not in seen_pairs,
                   'duplicate input or pair')
            seen_inputs.add(inp_sig)
            seen_pairs.add(pair_sig)
            _check(not _is_source_input(inp), 'source input copied')
            background, markers = _read_input(inp)
            _check(not np.array_equal(inp, out), 'unchanged pair')
            records[section].append((background, markers, out))
    train_counts = [len(markers) for _,markers,_ in records['train']]
    _check(max(train_counts) >= 3, 'training lacks a three-cross demonstration')
    _check(len(set(train_counts)) >= 2, 'training marker counts do not vary')
    colors = [_crossing_color(out, markers) for _,markers,out in records['train']
              if len(markers) >= 2]
    _check(colors and len(set(colors)) == 1, 'training highlight is ambiguous')
    highlight = colors[0]
    for section in ('train','test'):
        for background, markers, out in records[section]:
            _check(highlight != background and
                   highlight not in {color for _,_,color in markers},
                   'highlight appears in input')
            if len(markers) >= 2:
                _check(_crossing_color(out, markers) == highlight,
                       'inconsistent highlight')
            expected = _output(out.shape, markers, background, highlight)
            _check(np.array_equal(out, expected), 'output violates the cross rule')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_total, test_total = _resolve_counts(rng, num_train, num_test)
    highlight = int(rng.integers(10))
    train_counts = [_marker_count(rng) for _ in range(train_total)]
    if max(train_counts) < 3:
        train_counts[int(rng.integers(train_total))] = 3
    if len(set(train_counts)) == 1:
        index = int(rng.integers(train_total))
        train_counts[index] = 2 if train_counts[index] != 2 else 3
    test_counts = [_marker_count(rng) for _ in range(test_total)]
    if max(train_counts) < 8 and rng.random() < .28:
        index = int(rng.integers(test_total))
        test_counts[index] = int(rng.integers(max(train_counts)+1, 9))
    colors = [c for c in range(10) if c != highlight]
    rng.shuffle(colors)
    palette_size = (max(train_counts) if rng.random() < .45 else
                    int(rng.integers(max(train_counts), min(9,sum(train_counts))+1)))
    palette = colors[:palette_size]
    unused = [c for c in colors if c not in palette]
    preferred_background = (int(rng.choice(unused))
                            if unused and rng.random() < .30 else None)
    train_colors = _train_colors(rng, palette, train_counts)
    test_colors = [_test_colors(rng, count, palette, highlight) for count in test_counts]
    train_shapes = _training_shapes(rng, train_counts)
    test_shapes = [_test_shape(rng, count, train_shapes) for count in test_counts]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for section, shapes, selections in (('train', train_shapes, train_colors),
                                        ('test', test_shapes, test_colors)):
        for shape, selection in zip(shapes, selections):
            for _ in range(128):
                pair = _pair(rng, shape, selection, highlight, preferred_background)
                sig = _signature(pair['input'])
                if sig in seen or _is_source_input(pair['input']):
                    continue
                seen.add(sig)
                puzzle[section].append(pair)
                break
            else:
                raise ValueError('could not generate a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple((section, tuple((_signature(pair['input']), _signature(pair['output']))
                                 for pair in puzzle[section])) for section in ('train','test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    counts, shapes, backgrounds, highlights, layouts = set(),set(),set(),set(),set()
    edge = adjacent_row = adjacent_column = single = 0
    corner = black_marker = small = large = wide = tall = generous = 0
    familiar_counts = extrapolating_counts = mixed_sizes = 0
    narrow_palettes = stable_backgrounds = 0
    palette_relations, size_relations = set(),set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = {color for pair in puzzle['train']
                        for _,_,color in _read_input(pair['input'])[1]}
        train_shapes = [pair['input'].shape for pair in puzzle['train']]
        train_marker_counts = [len(_read_input(pair['input'])[1])
                               for pair in puzzle['train']]
        train_backgrounds = {_read_input(pair['input'])[0]
                             for pair in puzzle['train']}
        narrow_palettes += len(train_colors) <= 5
        stable_backgrounds += len(train_backgrounds) == 1
        demonstrated = {_crossing_color(pair['output'], _read_input(pair['input'])[1])
                        for pair in puzzle['train']
                        if len(_read_input(pair['input'])[1]) >= 2}
        assert len(demonstrated) == 1
        learned_highlight = next(iter(demonstrated))
        for section in ('train','test'):
            for pair in puzzle[section]:
                inp, out = pair['input'], pair['output']
                background, markers = _read_input(inp)
                count = len(markers)
                counts.add(count)
                shapes.add(inp.shape)
                backgrounds.add(background)
                layouts.add((inp.shape, tuple((r,c) for r,c,_ in markers)))
                edge += any(r in (0,inp.shape[0]-1) or c in (0,inp.shape[1]-1)
                            for r,c,_ in markers)
                adjacent_row += any(abs(a[0]-b[0]) == 1 for a in markers for b in markers if a != b)
                adjacent_column += any(abs(a[1]-b[1]) == 1 for a in markers for b in markers if a != b)
                single += count == 1
                black_marker += any(color == 0 for _,_,color in markers)
                h, w = inp.shape
                small += h <= 8 and w <= 8
                large += h >= 25 and w >= 25
                wide += w >= 2*h
                tall += h >= 2*w
                corner += any(r in (0,h-1) and c in (0,w-1) for r,c,_ in markers)
                if count >= 2:
                    rows = sorted(r for r,_,_ in markers)
                    columns = sorted(c for _,c,_ in markers)
                    generous += (min(np.diff(rows)) >= 2 and
                                 min(np.diff(columns)) >= 2 and
                                 rows[-1]-rows[0] >= h/2 and
                                 columns[-1]-columns[0] >= w/2)
                # Predict every cell using only the visible markers and the
                # highlight learned from training, without the construction helper.
                row_markers = {r: (r,c,color) for r,c,color in markers}
                column_markers = {c: (r,c,color) for r,c,color in markers}
                for r in range(h):
                    for c in range(w):
                        horizontal = row_markers.get(r)
                        vertical = column_markers.get(c)
                        if horizontal and vertical:
                            expected = (horizontal[2] if horizontal == vertical
                                        else learned_highlight)
                        elif horizontal:
                            expected = horizontal[2]
                        elif vertical:
                            expected = vertical[2]
                        else:
                            expected = background
                        assert int(out[r,c]) == expected
                if count >= 2:
                    highlights.add(_crossing_color(out, markers))
                if section == 'test':
                    novel = len({color for _,_,color in markers} - train_colors)
                    palette_relations.add('familiar' if novel == 0 else
                                          'novel' if novel == count else 'mixed')
                    familiar_counts += count in train_marker_counts
                    extrapolating_counts += count > max(train_marker_counts)
                    mixed_sizes += ((h > max(x[0] for x in train_shapes) and
                                     w < min(x[1] for x in train_shapes)) or
                                    (h < min(x[0] for x in train_shapes) and
                                     w > max(x[1] for x in train_shapes)))
                    if inp.shape in train_shapes:
                        size_relations.add('equal')
                    if h < min(x[0] for x in train_shapes) and w < min(x[1] for x in train_shapes):
                        size_relations.add('smaller')
                    if h > max(x[0] for x in train_shapes) and w > max(x[1] for x in train_shapes):
                        size_relations.add('larger')
                    if min(x[0] for x in train_shapes) < h < max(x[0] for x in train_shapes) and \
                       min(x[1] for x in train_shapes) < w < max(x[1] for x in train_shapes):
                        size_relations.add('intermediate')
    assert counts == set(range(1,9))
    assert len(shapes) >= 100 and len(layouts) >= 200
    assert len(backgrounds) >= 8 and len(highlights) >= 8
    assert edge > 15 and adjacent_row > 10 and adjacent_column > 10 and single > 0
    assert corner > 0 and black_marker > 0
    assert small > 0 and large > 0 and wide > 0 and tall > 0
    assert generous > 20 and narrow_palettes > 0 and stable_backgrounds > 0
    assert familiar_counts > 0 and extrapolating_counts > 0 and mixed_sizes > 0
    assert palette_relations == {'familiar','mixed','novel'}
    assert size_relations == {'equal','smaller','larger','intermediate'}
    requests = ((2,1),(10,1),(10,4),(6,2),(None,1),(None,4),(2,None),(10,None))
    for index, (train,test) in enumerate(requests):
        puzzle = generate_puzzle(10000+index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle['train']) == train
        if test is not None:
            assert len(puzzle['test']) == test
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(2,3),(3,4),(4,4)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal count request accepted: {args}')
    assert _puzzle_signature(generate_puzzle(31415,7,3)) == _puzzle_signature(generate_puzzle(31415,7,3))
    assert _puzzle_signature(generate_puzzle(27182)) == _puzzle_signature(generate_puzzle(27182))
