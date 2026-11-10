import numpy as np


_SOURCE_INPUT_ROWS = {
    tuple(block.split())
    for block in (
        """
        8888888848884888
        8888888488848884
        8888888888488848
        8888888884888488
        8888888888884888
        8888888888888888
        8888888888888888
        8888888888888888
        8888488888888888
        8884888848888848
        8848888488888488
        8888884888884888
        8888848888848888
        8888488888488888
        8884888884888888
        8888888848888888
        """,
        """
        8888888888838888
        8888888888388888
        8888388883888888
        8883888838888888
        8838888388888888
        8388883888888888
        3888888883888888
        8888888838888888
        8888888388888883
        8888883888888838
        8888838888888388
        8888388888883888
        8883888888838888
        8838888888388888
        8388888883888888
        8888888838888888
        """,
        """
        8888888888888888
        8888888888888888
        8888888888888888
        8888888888888888
        8888988889888888
        8889888898888988
        8888888988889888
        8888889888898889
        8888898888988898
        8888988889888988
        8888888898889888
        8888888988898888
        8888889888988888
        8888898889888888
        8888888898888888
        8888888988888888
        """,
    )
}


def _validate_count_value(
    name: str,
    value: int,
    lower: int,
    upper: int,
) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = (
        None
        if num_train is None
        else _validate_count_value("num_train", num_train, 2, 10)
    )
    test_count = (
        None
        if num_test is None
        else _validate_count_value("num_test", num_test, 1, 4)
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        lower = max(2, test_count + 1)
        train_count = lower + (seed // 4) % (11 - lower)

    elif train_count is None:
        lower = max(2, test_count + 1)
        train_count = lower + seed % (11 - lower)

    else:
        largest_test_count = min(4, train_count - 1)
        test_count = 1 + seed % largest_test_count

    return int(train_count), int(test_count)


def _interval_containing(
    rng: np.random.Generator,
    q: int,
    core: int,
    length: int,
) -> tuple[int, int]:
    """
    Construct an inclusive continuous t interval containing core.

    Marker cells on the stroke have t values with the same parity as q,
    separated by two. The continuous interval also covers the intermediate
    opposite-parity longitudinal coordinates.
    """
    anchor = core - ((core - q) % 2)
    parity_mismatch = anchor != core

    maximum_before = (
        length - 2 if parity_mismatch else length - 1
    )

    if length >= 4 and maximum_before >= 1:
        lower_before = 1
        upper_before = min(maximum_before, length - 2)
    else:
        lower_before = 0
        upper_before = maximum_before

    before = int(
        rng.integers(lower_before, upper_before + 1)
    )
    lo = anchor - 2 * before
    hi = lo + 2 * (length - 1)

    assert (lo - q) % 2 == 0
    assert lo <= core <= hi
    return int(lo), int(hi)


def _abstract_stroke_cells(
    q: int,
    lo: int,
    hi: int,
    slope: int,
) -> list[tuple[int, int]]:
    if (
        lo > hi
        or (lo - q) % 2
        or (hi - q) % 2
        or (hi - lo) % 2
    ):
        raise ValueError("invalid diagonal-coordinate interval")

    cells = []
    for t in range(lo, hi + 1, 2):
        row = (t - slope * q) // 2
        column = (q + slope * t) // 2
        cells.append((int(row), int(column)))
    return cells


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(array.shape),
        tuple(int(value) for value in array.ravel()),
    )


def _is_source_input(array: np.ndarray) -> bool:
    if array.shape != (16, 16):
        return False

    rows = tuple(
        "".join(str(int(value)) for value in row)
        for row in array
    )
    return rows in _SOURCE_INPUT_ROWS


def _components(strokes: list[dict]) -> list[list[int]]:
    count = len(strokes)
    adjacency = [[] for _ in range(count)]

    for first in range(count):
        for second in range(first + 1, count):
            overlaps = (
                max(
                    strokes[first]["lo"],
                    strokes[second]["lo"],
                )
                <= min(
                    strokes[first]["hi"],
                    strokes[second]["hi"],
                )
            )
            if overlaps:
                adjacency[first].append(second)
                adjacency[second].append(first)

    result = []
    seen = set()

    for start in range(count):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        result.append(component)

    return result


def _inspect_input(grid: np.ndarray) -> dict:
    """Recognize the field and the unique slope using the input alone."""
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert counts.max() > grid.size // 2, "field must be unambiguous"
    cells = [(int(r), int(c)) for r, c in np.argwhere(grid != background)]
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(c - slope * r, []).append(r)
        strokes = []
        valid = True
        for q, rows in lines.items():
            rows.sort()
            runs = []
            for row in rows:
                if not runs or row != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(row)
            for run in runs:
                colors = {int(grid[r, q + slope * r]) for r in run}
                if len(run) < 2 or len(colors) != 1:
                    valid = False
                    break
                strokes.append(dict(q=q, lo=2*run[0] + slope*q,
                                    hi=2*run[-1] + slope*q,
                                    length=len(run), color=colors.pop()))
        if valid and len(strokes) >= 2:
            strokes.sort(key=lambda x: (x['q'], x['lo']))
            candidates.append((slope, strokes))
    assert len(candidates) == 1, "strokes must have one recognizable slope"
    slope, strokes = candidates[0]
    for stroke in strokes:
        assert (stroke['lo'] - stroke['q']) % 2 == 0
        assert (stroke['hi'] - stroke['q']) % 2 == 0
    return dict(background=background, slope=slope, strokes=strokes)


def _expected_output(grid: np.ndarray, highlight: int, info: dict) -> np.ndarray:
    """Label construction: sweep continuous longitudinal coordinates."""
    rows, cols = np.indices(grid.shape)
    q = cols - info['slope'] * rows
    t = rows + info['slope'] * cols
    output = grid.copy()
    for level in range(int(t.min()), int(t.max()) + 1):
        active = [s['q'] for s in info['strokes'] if s['lo'] <= level <= s['hi']]
        if len(active) >= 2:
            mask = ((grid == info['background']) & (t == level)
                    & (q > min(active)) & (q < max(active)))
            output[mask] = highlight
    return output


def _geometric_prediction(grid: np.ndarray, highlight: int,
                          alternative: str = '') -> np.ndarray:
    """Separate check: union the parallelograms bracketed by segment pairs.

    This does not call the sweep used to construct labels. Alternatives are
    deliberately weaker rules, used to check visible training evidence.
    """
    info = _inspect_input(grid)
    strokes = info['strokes']
    pairs = [(i, j) for i in range(len(strokes)) for j in range(i+1, len(strokes))]
    if alternative == 'consecutive':
        pairs = []
        for component in _components(strokes):
            ordered = sorted(component, key=lambda i: strokes[i]['q'])
            pairs.extend(zip(ordered, ordered[1:]))
    rows, cols = np.indices(grid.shape)
    transverse = cols - info['slope'] * rows
    longitudinal = rows + info['slope'] * cols
    filled = np.zeros(grid.shape, dtype=bool)
    for i, j in pairs:
        a, b = strokes[i], strokes[j]
        if alternative == 'same_color' and a['color'] != b['color']:
            continue
        low, high = max(a['lo'], b['lo']), min(a['hi'], b['hi'])
        mask = ((longitudinal >= low) & (longitudinal <= high)
                & (transverse > min(a['q'], b['q']))
                & (transverse < max(a['q'], b['q'])))
        if alternative == 'pixel_centers':
            mask &= ((longitudinal-a['q']) % 2 == 0) & ((longitudinal-b['q']) % 2 == 0)
        filled |= mask
    output = grid.copy()
    output[filled & (grid == info['background'])] = highlight
    return output


def _training_highlight(train_pairs: list[dict]) -> int:
    changed_colors = set()
    for pair in train_pairs:
        mask = pair['input'] != pair['output']
        changed_colors.update(int(c) for c in np.unique(pair['output'][mask]))
    assert len(changed_colors) == 1, "training must demonstrate one highlight"
    highlight = changed_colors.pop()
    assert all(not np.any(p['input'] == highlight) for p in train_pairs)
    return highlight


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # No test label, seed, sampled settings, or generation metadata is used.
    return _geometric_prediction(test_input, _training_highlight(train_pairs))


def _different_alternative(pair: dict, highlight: int, alternative: str) -> bool:
    return not np.array_equal(pair['output'],
                              _geometric_prediction(pair['input'], highlight, alternative))


def _sample_group(rng: np.random.Generator, count: int, max_length: int,
                  max_gap: int, witness: bool) -> list[tuple[int, int, int]]:
    qs = [0]
    for _ in range(count-1):
        qs.append(qs[-1] + int(rng.integers(2, max_gap+1)))
    length_mode = int(rng.integers(4))
    if length_mode == 0:
        lengths = [int(rng.integers(2, max_length+1))] * count
    elif length_mode == 1:
        center = int(rng.integers(3, max_length+1))
        lengths = [max(2, min(max_length, center+int(rng.integers(-1, 2)))) for _ in qs]
    else:
        lengths = [int(rng.integers(2, max_length+1)) for _ in qs]
        if length_mode == 3:
            for i in rng.choice(count, size=int(rng.integers(1, count+1)), replace=False):
                lengths[int(i)] = int(rng.integers(2, 4))
    mode = int(rng.choice(5, p=[.25, .15, .12, .18, .30]))
    intervals = [None] * count
    order = [int(i) for i in rng.permutation(count)]
    previous = None
    for i in order:
        q, length = qs[i], lengths[i]
        if mode == 0:
            lo, hi = _interval_containing(rng, q, 0, length)
        elif mode == 1:
            lo = q % 2
            hi = lo + 2*(length-1)
        elif mode == 2:
            hi = -(q % 2)
            lo = hi - 2*(length-1)
        elif previous is None:
            lo = q % 2
            hi = lo + 2*(length-1)
        elif mode == 3:
            anchor = int(rng.integers(previous[0], previous[1]+1))
            lo, hi = _interval_containing(rng, q, anchor, length)
        else:
            overlap = int(rng.integers(0, min(4, 2*(length-1))))
            lo = previous[1] - overlap
            lo -= (lo-q) % 2
            hi = lo + 2*(length-1)
        intervals[i] = (lo, hi)
        previous = (lo, hi)
    if witness:
        # Rank, lengths, spacing, and endpoint phases all remain variable.
        interior = int(rng.integers(1, count-1))
        lo = int(rng.integers(-2, 3))
        lo -= (lo-qs[interior]) % 2
        hi = lo + 2*int(rng.integers(1, 3))
        intervals[interior] = (lo, hi)
        for neighbor in (interior-1, interior+1):
            start = lo - int(rng.integers(3, 8))
            start -= (start-qs[neighbor]) % 2
            end = hi + int(rng.integers(3, 8))
            end += (qs[neighbor]-end) % 2
            intervals[neighbor] = (start, end)
    return [(q, int(lo), int(hi)) for q, (lo, hi) in zip(qs, intervals)]


def _sample_specs(rng: np.random.Generator, regime: str,
                  witness: bool, allow_unchanged: bool) -> list[tuple[int, int, int]]:
    if allow_unchanged and rng.random() < .04:
        # Two visible but longitudinally disjoint intervals: no bracket.
        n_groups, small, singleton = 2, True, True
    else:
        n_groups = int(rng.choice([1, 2, 3, 4], p=[.57, .27, .12, .04]))
        small = bool(rng.random() < .25)
        singleton = False
    specs = []
    for group in range(n_groups):
        if singleton or (group > 0 and rng.random() < .10):
            count = 1
        else:
            upper = 3 if small or n_groups == 4 else (8 if n_groups == 1 else 5)
            if regime == 'target_test' and not small and n_groups == 1:
                upper = 9
            lower = 3 if witness and group == 0 else 2
            count = int(rng.integers(lower, upper+1))
        max_length = 3 if n_groups == 4 else (5 if small or n_groups == 3 else (10 if n_groups == 2 else 16))
        max_gap = int(rng.integers(3, 7 if small or n_groups == 4 else 13))
        current = _sample_group(rng, count, max_length, max_gap, witness and group == 0)
        # Even independent shifts retain q/t pixel parity. Overlap is always
        # interpreted globally; no visual-distance grouping is ever applied.
        extent = 3 if n_groups == 4 else 7
        q_shift = 2*int(rng.integers(-extent, extent+1))
        t_shift = 0
        if specs:
            desired = max(hi for _, _, hi in specs) + int(rng.integers(4, 10))
            t_shift = desired - min(lo for _, lo, _ in current)
            t_shift += t_shift % 2
        specs.extend((q+q_shift, lo+t_shift, hi+t_shift) for q, lo, hi in current)
    return specs


def _render_specs(rng: np.random.Generator, specs: list, slope: int,
                  background: int, colors: list[int], highlight: int,
                  canvas: tuple | None) -> dict:
    strokes = [_abstract_stroke_cells(q, lo, hi, slope) for q, lo, hi in specs]
    cells = [cell for stroke in strokes for cell in stroke]
    if len(set(cells)) != len(cells):
        raise ValueError('overlapping stroke pixels')
    rmin, cmin = np.min(cells, axis=0)
    rmax, cmax = np.max(cells, axis=0)
    height, width = int(rmax-rmin+1), int(cmax-cmin+1)
    if height > 30 or width > 30:
        raise ValueError('layout too large')
    placement = str(rng.choice(['interior', 'edge', 'tight', 'free'], p=[.35, .35, .15, .15]))
    if canvas is None:
        # Small bounding boxes can stay small; padding/aspect ratio is free.
        dims = []
        padding_cap = 3 if max(height, width) <= 8 else 8
        for span in (height, width):
            extra = 30-span
            minimum = min(2, extra) if placement == 'interior' else 0
            if rng.random() < .18:
                dims.append(int(rng.integers(span+minimum, 31)))
            else:
                dims.append(span+int(rng.integers(minimum, min(extra, padding_cap)+1)))
        canvas = tuple(dims)
    if height > canvas[0] or width > canvas[1]:
        raise ValueError('layout does not fit')
    offsets = []
    for span, size in zip((height, width), canvas):
        if placement == 'interior' and size-span >= 2:
            offsets.append(int(rng.integers(1, size-span)))
        elif placement == 'tight':
            offsets.append(int(rng.choice([0, size-span])))
        else:
            offsets.append(int(rng.integers(0, size-span+1)))
    if placement == 'edge':
        axis = int(rng.integers(2))
        offsets[axis] = int(rng.choice([0, canvas[axis]-(height, width)[axis]]))
    grid = np.full(canvas, background, dtype=np.int8)
    for index, stroke in enumerate(strokes):
        for row, col in stroke:
            grid[row-int(rmin)+offsets[0], col-int(cmin)+offsets[1]] = colors[index % len(colors)]
    try:
        info = _inspect_input(grid)
    except AssertionError as exc:
        raise ValueError('ambiguous stroke recognition') from exc
    return dict(input=grid, output=_expected_output(grid, highlight, info))


def _make_pair(rng: np.random.Generator, regime: str, slope: int,
               background: int, colors: list[int], highlight: int,
               canvas: tuple | None, witness: bool, color_witness: bool,
               parity_witness: bool, must_change: bool, seen: set,
               excluded_count: int | None = None) -> dict:
    for attempt in range(96):
        if attempt < 80:
            specs = _sample_specs(rng, regime, witness, not must_change)
        else:
            # A 9x9 witness has sixteen distinct placements in every canvas
            # used here (at least 12x12). At most thirteen inputs precede it.
            k = attempt-80
            specs = [(0, -4, 4), (4, 0, 2), (8, -4, 4)]
            if excluded_count == 3:
                specs.append((10, 0, 2))
        try:
            render_colors = colors
            render_canvas = canvas
            if attempt >= 80:
                render_canvas = canvas or (12, 12)
                if color_witness:
                    render_colors = [colors[0]] + [colors[1]]*(len(specs)-1)
            pair = _render_specs(rng, specs, slope, background, render_colors, highlight, render_canvas)
        except ValueError:
            continue
        if attempt >= 80:
            rows, cols = np.where(pair['input'] != background)
            grid = np.full(pair['input'].shape, background, dtype=np.int8)
            grid[rows-rows.min()+k//4, cols-cols.min()+k % 4] = pair['input'][rows, cols]
            pair = dict(input=grid, output=_expected_output(grid, highlight, _inspect_input(grid)))
        if _array_key(pair['input']) in seen or _is_source_input(pair['input']):
            continue
        if excluded_count is not None and len(_inspect_input(pair['input'])['strokes']) == excluded_count:
            continue
        if must_change and np.array_equal(pair['input'], pair['output']):
            continue
        if witness and not _different_alternative(pair, highlight, 'consecutive'):
            continue
        if color_witness and not _different_alternative(pair, highlight, 'same_color'):
            continue
        if parity_witness and not _different_alternative(pair, highlight, 'pixel_centers'):
            continue
        return pair
    raise AssertionError('bounded construction exhausted')


def _sample_parameters(rng: np.random.Generator) -> dict:
    palette = [int(x) for x in rng.permutation(10)]
    highlight = palette.pop()
    bg_count = int(rng.integers(2, 4)) if rng.random() < .5 else 1
    backgrounds, markers = palette[:bg_count], palette[bg_count:]
    multi = bool(rng.random() < .45)
    policy = str(rng.choice(['shared', 'pooled', 'novel'], p=[.25, .50, .25]))
    if policy == 'novel':
        train_colors, test_colors = markers[:3], markers[3:]
    else:
        count = int(rng.integers(2 if multi else 1, min(5, len(markers))+1))
        train_colors = markers[:count]
        test_colors = train_colors
    slope = int(rng.choice([-1, 1]))
    slope_mode = str(rng.choice(['fixed', 'mixed', 'transfer'], p=[.4, .4, .2]))
    size_mode = str(rng.choice(['same', 'larger', 'smaller', 'free'], p=[.18, .17, .17, .48]))
    common_canvas = (int(rng.integers(16, 31)), int(rng.integers(16, 31)))
    return dict(highlight=highlight, backgrounds=backgrounds,
                train_colors=train_colors, test_colors=test_colors, multi=multi,
                slope=slope, slope_mode=slope_mode, size_mode=size_mode,
                common_canvas=common_canvas)


def _choose_canvas(rng: np.random.Generator, parameters: dict, regime: str) -> tuple | None:
    mode = parameters['size_mode']
    if mode == 'free':
        return None
    if mode == 'same':
        return parameters['common_canvas']
    large = (mode == 'larger') == (regime == 'target_test')
    lower, upper = (22, 30) if large else (12, 19)
    return (int(rng.integers(lower, upper+1)), int(rng.integers(lower, upper+1)))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    settings = _sample_parameters(rng)
    highlight = settings['highlight']
    puzzle = dict(train=[], test=[])
    seen = set()
    # Construct tests first so their actual geometric dependencies determine
    # the necessary demonstrations. No test label is used by the predictor.
    test_multi = False
    need_witness = False
    need_subpixel = False
    witness_index = int(rng.integers(train_count))
    for section, count, regime in [('test', test_count, 'target_test'),
                                   ('train', train_count, 'source_train')]:
        pool = settings[section+'_colors']
        for index in range(count):
            slope = settings['slope']
            if settings['slope_mode'] == 'transfer' and section == 'test':
                slope = -slope
            elif settings['slope_mode'] == 'mixed':
                # Both slopes in training, even for a two-pair request.
                slope *= (-1 if index % 2 else 1) if section == 'train' else int(rng.choice([-1, 1]))
            background = settings['backgrounds'][index % len(settings['backgrounds'])]
            if index >= 2:
                background = int(rng.choice(settings['backgrounds']))
            multicolor = settings['multi'] and (rng.random() < .55 or (section == 'train' and index == witness_index and test_multi))
            base = index % len(pool) if section == 'train' else int(rng.integers(len(pool)))
            colors = [pool[base]]
            if multicolor:
                rest = [c for c in pool if c != colors[0]]
                rng.shuffle(rest)
                colors += rest[:int(rng.integers(1, min(3, len(rest))+1))]
            witness = section == 'train' and index == witness_index and need_witness
            color_witness = section == 'train' and index == witness_index and test_multi
            excluded_count = None
            if section == 'train' and index == count-1:
                prior_counts = {len(_inspect_input(p['input'])['strokes'])
                                for p in puzzle['test']+puzzle['train']}
                if len(prior_counts) == 1:
                    excluded_count = prior_counts.pop()
            pair = _make_pair(rng, regime, slope, background, colors, highlight,
                              _choose_canvas(rng, settings, regime), witness, color_witness,
                              section == 'train' and index == 0 and need_subpixel,
                              section == 'train' and (index == 0 or witness or color_witness),
                              seen, excluded_count)
            puzzle[section].append(pair)
            seen.add(_array_key(pair['input']))
            if section == 'test':
                need_witness |= _different_alternative(pair, highlight, 'consecutive')
                need_subpixel |= _different_alternative(pair, highlight, 'pixel_centers')
                test_multi |= len({s['color'] for s in _inspect_input(pair['input'])['strokes']}) > 1
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            key = _array_key(pair['input'])
            pair_key = (key, _array_key(pair['output']))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert not _is_source_input(pair['input'])
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
    highlight = _training_highlight(puzzle['train'])
    infos = {split: [_inspect_input(p['input']) for p in puzzle[split]] for split in puzzle}
    for section in puzzle:
        for pair, info in zip(puzzle[section], infos[section]):
            assert not np.any(pair['input'] == highlight)
            marker_mask = pair['input'] != info['background']
            assert np.array_equal(pair['output'][marker_mask], pair['input'][marker_mask])
            assert np.array_equal(pair['output'], _predict_from_training(puzzle['train'], pair['input']))
    if any(_different_alternative(p, highlight, 'consecutive') for p in puzzle['test']):
        assert any(_different_alternative(p, highlight, 'consecutive') for p in puzzle['train']), 'missing inactive-interior training witness'
    if any(_different_alternative(p, highlight, 'pixel_centers') for p in puzzle['test']):
        assert any(_different_alternative(p, highlight, 'pixel_centers') for p in puzzle['train']), 'missing continuous-interval training witness'
    if any(len({s['color'] for s in i['strokes']}) > 1 for i in infos['test']):
        assert any(_different_alternative(p, highlight, 'same_color') for p in puzzle['train']), 'missing color-independent participation witness'
    train_markers = {s['color'] for i in infos['train'] for s in i['strokes']}
    test_markers = {s['color'] for i in infos['test'] for s in i['strokes']}
    if not test_markers <= train_markers:
        assert len(train_markers) >= 2, 'novel markers require role-based training evidence'
    train_backgrounds = {i['background'] for i in infos['train']}
    if any(i['background'] not in train_backgrounds for i in infos['test']):
        assert len(train_backgrounds) >= 2


def _pair_features(pair: dict, highlight: int) -> dict:
    """Review measurements reconstructed from returned grids, not sampler modes."""
    grid = pair['input']
    info = _inspect_input(grid)
    strokes = info['strokes']
    groups = [[strokes[i] for i in ids] for ids in _components(strokes)]
    gaps, overlaps, interior_short, outer_short, longest_ranks = [], [], [], [], []
    common, staggered, equal, ties, contained, aligned_start, aligned_end = 0, 0, 0, 0, 0, 0, 0
    group_centers = []
    for group in groups:
        group.sort(key=lambda s: s['q'])
        group_centers.append(sum(s['q'] for s in group)/len(group))
        lengths = [s['length'] for s in group]
        gaps += [b['q']-a['q'] for a, b in zip(group, group[1:])]
        for i, s in enumerate(group):
            if s['length'] == min(lengths):
                (outer_short if i in (0, len(group)-1) else interior_short).append(i)
            if s['length'] == max(lengths):
                longest_ranks.append(i)
        if len(group) > 1:
            common += max(s['lo'] for s in group) <= min(s['hi'] for s in group)
            staggered += max(s['lo'] for s in group) > min(s['hi'] for s in group)
            equal += len(set(lengths)) == 1
            ties += len(set(lengths)) < len(lengths)
            aligned_start += len({s['lo'] for s in group}) < len(group)
            aligned_end += len({s['hi'] for s in group}) < len(group)
        for i, a in enumerate(group):
            for b in group[i+1:]:
                overlap = min(a['hi'], b['hi'])-max(a['lo'], b['lo'])+1
                if overlap > 0:
                    overlaps.append(overlap)
                contained += ((a['lo'] <= b['lo'] and a['hi'] >= b['hi']) or
                              (b['lo'] <= a['lo'] and b['hi'] >= a['hi']))
    occupied = grid != info['background']
    rows, cols = np.where(occupied)
    padding = [int(rows.min()), int(cols.min()), int(grid.shape[0]-1-rows.max()), int(grid.shape[1]-1-cols.max())]
    return dict(height=grid.shape[0], width=grid.shape[1], strokes=len(strokes),
                groups=len(groups), group_sizes=sorted(len(g) for g in groups),
                transverse_group_shift=len(set(group_centers)) > 1,
                gaps=gaps, lengths=[s['length'] for s in strokes], overlaps=overlaps,
                common_groups=int(common), staggered_groups=int(staggered),
                equal_groups=int(equal), tied_groups=int(ties), contained_pairs=int(contained),
                aligned_starts=int(aligned_start), aligned_ends=int(aligned_end),
                shortest_interior=bool(interior_short), shortest_outer=bool(outer_short),
                shortest_left=any(g[0]['length'] == min(s['length'] for s in g) for g in groups),
                shortest_right=any(g[-1]['length'] == min(s['length'] for s in g) for g in groups),
                longest_ranks=longest_ranks, padding=padding, edge=0 in padding,
                corner=bool(np.any(occupied[np.ix_([0, grid.shape[0]-1], [0, grid.shape[1]-1])])),
                background=info['background'], marker_colors=sorted({s['color'] for s in strokes}),
                highlight=highlight, slope=info['slope'],
                changed=int(np.sum(grid != pair['output'])),
                skip_witness=_different_alternative(pair, highlight, 'consecutive'),
                color_witness=_different_alternative(pair, highlight, 'same_color'),
                subpixel_witness=_different_alternative(pair, highlight, 'pixel_centers'))


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((_array_key(p['input']), _array_key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test() -> None:
    # Hand-calculated endpoint/parity case: seven field pixels are bracketed,
    # including four on scanlines between both boundary pixels' centers.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[(0, 1, 2, 2, 3, 4), (2, 1, 0, 4, 3, 2)] = 3
    expected = grid.copy()
    expected[(1, 1, 2, 2, 2, 3, 3), (2, 3, 1, 2, 3, 1, 2)] = 2
    assert np.array_equal(_expected_output(grid, 2, _inspect_input(grid)), expected)
    assert np.array_equal(_geometric_prediction(grid, 2), expected)
    assert not np.array_equal(_geometric_prediction(grid, 2, 'pixel_centers'), expected)
    grid[2, 2] = grid[3, 1] = 7
    expected[2, 2] = expected[3, 1] = 7
    assert np.array_equal(_geometric_prediction(grid, 2), expected)
    assert _geometric_prediction(grid, 2, 'consecutive')[1, 3] == 0
    assert expected[1, 3] == 2
    # Correct labels can still form an inadequate teaching set. Two simple
    # brackets do not demonstrate bypassing an inactive intermediate stroke.
    simple = grid.copy()
    simple[simple == 7] = 0
    simple_output = _geometric_prediction(simple, 2)
    other = simple.copy()
    other[other == 3] = 5
    train = [dict(input=simple, output=simple_output),
             dict(input=other, output=_geometric_prediction(other, 2))]
    inactive = grid.copy()
    inactive[inactive == 7] = 3
    inadequate = dict(train=train, test=[dict(input=inactive, output=_geometric_prediction(inactive, 2))])
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as exc:
        assert 'inactive-interior' in str(exc)
    else:
        raise AssertionError('missing training witness accepted')
    mixed = simple.copy()
    mixed[(2, 3, 4), (4, 3, 2)] = 5
    inadequate['test'] = [dict(input=mixed, output=_geometric_prediction(mixed, 2))]
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as exc:
        assert 'color-independent' in str(exc)
    else:
        raise AssertionError('untaught multicolor participation accepted')
    fingerprints, shapes, layouts, train_counts, test_counts = set(), set(), set(), set(), set()
    records, puzzles = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        features = {split: [_pair_features(p, _training_highlight(puzzle['train'])) for p in puzzle[split]] for split in puzzle}
        puzzles.append(features)
        for split in puzzle:
            for pair, record in zip(puzzle[split], features[split]):
                # Independent of label construction, though self-authored.
                assert np.array_equal(_predict_from_training(puzzle['train'], pair['input']), pair['output'])
                records.append(record)
                shapes.add(pair['input'].shape)
                info = _inspect_input(pair['input'])
                specs = info['strokes']
                q0, t0 = min(s['q'] for s in specs), min(s['lo'] for s in specs)
                layouts.add(tuple((s['q']-q0, s['lo']-t0, s['hi']-t0) for s in specs))
    assert len(fingerprints) == 42 and len(layouts) >= 180 and len(shapes) >= 70
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({r['highlight'] for r in records}) >= 8
    assert {r['slope'] for r in records} == {-1, 1}
    assert len({r['strokes'] for r in records}) >= 8
    assert {1, 2, 3} <= {r['groups'] for r in records}
    assert any(r['groups'] == 4 for r in records)
    assert min(g for r in records for g in r['gaps']) == 2
    assert max(g for r in records for g in r['gaps']) >= 10
    assert min(n for r in records for n in r['lengths']) == 2
    assert max(n for r in records for n in r['lengths']) >= 14
    assert any(any(t <= 3 for t in r['overlaps']) for r in records)
    assert any(any(t >= 12 for t in r['overlaps']) for r in records)
    assert any(sum(n <= 3 for n in r['lengths']) >= 2 for r in records)
    assert any(4 <= min(r['lengths']) <= max(r['lengths']) <= 9
               and max(r['lengths'])-min(r['lengths']) <= 2 for r in records)
    assert min(min(r['height'], r['width']) for r in records) <= 5
    assert max(max(r['height'], r['width']) for r in records) == 30
    for name in ('staggered_groups', 'equal_groups', 'tied_groups', 'contained_pairs',
                 'aligned_starts', 'aligned_ends', 'shortest_interior', 'shortest_left',
                 'shortest_right', 'edge', 'corner', 'skip_witness', 'color_witness',
                 'subpixel_witness', 'transverse_group_shift'):
        assert sum(bool(r[name]) for r in records) >= 4, name
    assert sum(not r['changed'] for r in records) >= 2
    assert any(r['background'] == 2 for r in records)
    assert any(2 in r['marker_colors'] for r in records)
    assert any(len(set(r['group_sizes'])) > 1 for r in records)
    assert any(r['staggered_groups'] and len(r['marker_colors']) > 1 and r['edge'] for r in records)
    assert sum(not r['edge'] for r in records) >= 20
    for split in ('train', 'test'):
        selected = [r for p in puzzles for r in p[split]]
        assert {2, 3, 5} <= {r['strokes'] for r in selected}
        assert any(r['strokes'] == 4 and r['groups'] == 1
                   and 2 not in r['longest_ranks'] for r in selected)
    for name in ('strokes', 'background', 'slope'):
        assert sum(len({r[name] for split in p.values() for r in split}) > 1 for p in puzzles) >= 8
    assert all(len({r['strokes'] for split in p.values() for r in split}) > 1 for p in puzzles)
    size_relations, transfers, reused, novel = set(), set(), 0, 0
    for p in puzzles:
        train_shapes = [(r['height'], r['width']) for r in p['train']]
        for r in p['test']:
            shape = (r['height'], r['width'])
            if shape in train_shapes:
                size_relations.add('same')
            if all(shape[0] > h and shape[1] > w for h, w in train_shapes):
                size_relations.add('larger')
            if all(shape[0] < h and shape[1] < w for h, w in train_shapes):
                size_relations.add('smaller')
        train_slopes, test_slopes = {r['slope'] for r in p['train']}, {r['slope'] for r in p['test']}
        if len(train_slopes) == len(test_slopes) == 1 and train_slopes != test_slopes:
            transfers.add(next(iter(train_slopes)))
        a = {c for r in p['train'] for c in r['marker_colors']}
        b = {c for r in p['test'] for c in r['marker_colors']}
        reused += bool(a & b)
        novel += bool(b-a)
    assert size_relations == {'same', 'larger', 'smaller'}
    assert transfers == {-1, 1} and reused >= 5 and novel >= 5
    for seed in (21, 48, 57):
        p = generate_puzzle(seed, 2, 1)
        validate_generated_puzzle(p)
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(301, train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p['train']) == train
        assert test is None or len(p['test']) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1)):
        try:
            generate_puzzle(999, train, test)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Corrupt one predicted background cell; structural validity alone must
    # never let a geometrically incorrect label through the validator.
    p = generate_puzzle(21, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        return
    raise AssertionError('corrupt output accepted')
