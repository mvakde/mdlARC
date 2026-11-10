import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "000000000000",
            "000000000000",
            "000002200000",
            "000552255000",
            "000533335000",
            "002231132200",
            "002231132200",
            "000533335000",
            "000552255000",
            "000002200000",
            "000000000000",
            "000000000000",
        ),
        (
            "0002",
            "0552",
            "0533",
            "2231",
        ),
    ),
    (
        (
            "000220000000",
            "002222000000",
            "023113200000",
            "221001220000",
            "221001220000",
            "023113200000",
            "002222000000",
            "000220000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
        (
            "0002",
            "0022",
            "0231",
            "2210",
        ),
    ),
    (
        (
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000007700770",
            "000072233227",
            "000072888827",
            "000003800830",
            "000003800830",
            "000072888827",
            "000072233227",
            "000007700770",
        ),
        (
            "0770",
            "7223",
            "7288",
            "0380",
        ),
    ),
    (
        (
            "001005500100",
            "000538835000",
            "000328823000",
            "005886688500",
            "005886688500",
            "000328823000",
            "000538835000",
            "001005500100",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
        (
            "1005",
            "0538",
            "0328",
            "5886",
        ),
    ),
)


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(character) for row in rows for character in row),
    )


_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(input_rows), _rows_key(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)

def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(int(n) for n in grid.shape),
            tuple(int(v) for v in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair['input']), _grid_key(pair['output']))


def _checked_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    return int(seed)


def _checked_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _checked_count(num_train, 'num_train', 2, 10)
    test = _checked_count(num_test, 'num_test', 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        assert test is not None
        low = test + 1
        train = low + (seed // 7) % (11 - low)
    elif test is None:
        test = 1 + (seed // 11) % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _encode_seed(tile: np.ndarray) -> np.ndarray:
    top = np.concatenate((tile, np.fliplr(tile)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _background(grid: np.ndarray) -> int:
    counts = np.bincount(grid.reshape(-1).astype(int), minlength=10)
    return int(np.argmax(counts))


def _visible_bounds(grid: np.ndarray, background: int) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(grid != background)
    _require(len(rows) > 0, 'a construction needs visible motif cells')
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _locate(grid: np.ndarray, shape: tuple[int, int], background: int) -> tuple[int, int, np.ndarray] | None:
    """Find the sole centered placement of a given reflected tile size."""
    h, w = shape
    if 2 * h > grid.shape[0] or 2 * w > grid.shape[1]:
        return None
    r0, r1, c0, c1 = _visible_bounds(grid, background)
    row_sum = r0 + r1 - 2 * h
    col_sum = c0 + c1 - 2 * w
    if row_sum % 2 or col_sum % 2:
        return None
    row, col = row_sum // 2, col_sum // 2
    if row < 0 or col < 0 or row + 2*h > grid.shape[0] or col + 2*w > grid.shape[1]:
        return None
    block = grid[row:row+2*h, col:col+2*w]
    tile = block[:h, :w]
    if not np.array_equal(block, _encode_seed(tile)):
        return None
    outside = grid.copy()
    outside[row:row+2*h, col:col+2*w] = background
    if not np.all(outside == background):
        return None
    return row, col, tile.copy()


def _tight_tile(grid: np.ndarray, background: int) -> np.ndarray | None:
    r0, r1, c0, c1 = _visible_bounds(grid, background)
    h, w = r1-r0, c1-c0
    if h % 2 or w % 2 or h < 4 or w < 4:
        return None
    block = grid[r0:r1, c0:c1]
    tile = block[:h//2, :w//2]
    return tile.copy() if np.array_equal(block, _encode_seed(tile)) else None


def _sample_motif(rng: np.random.Generator, shape: tuple[int, int],
                  background: int, palette_size: int, style: int,
                  blank_top: bool, blank_left: bool) -> np.ndarray:
    h, w = shape
    first_r, first_c = int(blank_top), int(blank_left)
    ah, aw = h-first_r, w-first_c
    colors = np.asarray(rng.choice([c for c in range(10) if c != background],
                                   size=palette_size, replace=False), dtype=np.int8)
    for _ in range(96):
        rr, cc = np.indices((ah, aw))
        phase = int(rng.integers(0, palette_size))
        if style == 0:  # small color regions
            values = np.full((ah, aw), background, dtype=np.int8)
            for k in range(palette_size + int(rng.integers(2, 6))):
                a, b = int(rng.integers(0, ah)), int(rng.integers(0, aw))
                dh, dw = int(rng.integers(1, min(4, ah-a)+1)), int(rng.integers(1, min(4, aw-b)+1))
                values[a:a+dh, b:b+dw] = colors[(k+phase) % palette_size]
        elif style == 1:
            values = np.broadcast_to(colors[(rr+phase) % palette_size], (ah, aw)).copy()
        elif style == 2:
            values = np.broadcast_to(colors[(cc+phase) % palette_size], (ah, aw)).copy()
        elif style == 3:
            values = colors[(rr+cc+phase) % palette_size].copy()
        elif style == 4:
            rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            values = colors[((rr//rh) + 2*(cc//cw) + phase) % palette_size].copy()
        else:
            values = colors[rng.integers(0, palette_size, size=(ah, aw))].copy()
        values[rng.random((ah, aw)) < float(rng.uniform(0.25, 0.55))] = background
        # Every palette color is visible; the outer active rows and columns
        # anchor the actual block extent, even when one perimeter is blank.
        indexes = rng.choice(ah*aw, size=palette_size, replace=False)
        values.flat[indexes] = colors
        for r, c in ((0, int(rng.integers(0, aw))),
                     (ah-1, int(rng.integers(0, aw))),
                     (int(rng.integers(0, ah)), 0),
                     (int(rng.integers(0, ah)), aw-1)):
            values[r, c] = colors[int(rng.integers(0, palette_size))]
        tile = np.full((h, w), background, dtype=np.int8)
        tile[first_r:, first_c:] = values
        if set(int(v) for v in np.unique(tile)) != {background, *map(int, colors)}:
            continue
        if (np.all(values[0] == background) or np.all(values[-1] == background)
                or np.all(values[:, 0] == background) or np.all(values[:, -1] == background)):
            continue
        if (np.array_equal(tile, np.fliplr(tile)) or np.array_equal(tile, np.flipud(tile))
                or np.array_equal(tile, np.flipud(np.fliplr(tile)))):
            continue
        return tile
    raise RuntimeError('motif construction failed')


def _canvas_and_position(rng: np.random.Generator, block: tuple[int, int],
                         source_like_test: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    bh, bw = block
    mode = int(rng.integers(0, 5))
    if mode == 0 and bh < 30:  # full height
        eh, ew = 0, int(rng.integers(min(30-bw, max(2, bw//2)), 31-bw)) if bw < 29 else 1
    elif mode == 1 and bw < 30:  # full width
        eh, ew = int(rng.integers(min(30-bh, max(2, bh//2)), 31-bh)) if bh < 29 else 1, 0
    elif mode == 2:  # compact
        eh = int(rng.integers(0, min(6, 31-bh)))
        ew = int(rng.integers(0, min(6, 31-bw)))
    elif mode == 3:  # roomy
        eh = int(rng.integers(min(30-bh, 5), 31-bh))
        ew = int(rng.integers(min(30-bw, 5), 31-bw))
    else:
        eh, ew = int(rng.integers(0, 31-bh)), int(rng.integers(0, 31-bw))
    if eh == ew == 0:
        if bh < 30:
            eh = 1
        else:
            ew = 1
    style = int(rng.integers(0, 6))
    if source_like_test:
        style = 4
    if style == 0:  # corner
        row, col = int(rng.choice((0, eh))), int(rng.choice((0, ew)))
    elif style == 1:  # edge, with an independent offset on the other axis
        if bool(rng.integers(0, 2)):
            row, col = int(rng.choice((0, eh))), int(rng.integers(0, ew+1))
        else:
            row, col = int(rng.integers(0, eh+1)), int(rng.choice((0, ew)))
    elif style == 2:  # deliberately offset by one cell
        row, col = (min(1, eh), int(rng.integers(0, ew+1))) if bool(rng.integers(0, 2)) else (int(rng.integers(0, eh+1)), min(1, ew))
    elif style == 3:  # interior
        row = int(rng.integers(1, eh)) if eh >= 2 else int(rng.integers(0, eh+1))
        col = int(rng.integers(1, ew)) if ew >= 2 else int(rng.integers(0, ew+1))
    elif style == 4:  # top center
        row, col = 0, ew//2
    else:
        row, col = int(rng.integers(0, eh+1)), int(rng.integers(0, ew+1))
    return (bh+eh, bw+ew), (row, col)


def _make_pair(rng: np.random.Generator, shape: tuple[int, int], background: int,
               palette_goal: int, blank: bool, source_like_test: bool) -> dict:
    h, w = shape
    for _ in range(128):
        blank_top = blank and bool(rng.integers(0, 2))
        blank_left = blank and (not blank_top or bool(rng.integers(0, 2)))
        if not blank:
            blank_top = blank_left = False
        active_area = (h-int(blank_top))*(w-int(blank_left))
        colors = min(palette_goal, max(2, active_area//3))
        tile = _sample_motif(rng, shape, background, colors,
                             int(rng.integers(0, 6)), blank_top, blank_left)
        block = _encode_seed(tile)
        canvas, (row, col) = _canvas_and_position(rng, block.shape, source_like_test)
        inp = np.full(canvas, background, dtype=np.int8)
        inp[row:row+block.shape[0], col:col+block.shape[1]] = block
        if _background(inp) != background:
            continue
        located = _locate(inp, shape, background)
        if located is None or located[:2] != (row, col) or not np.array_equal(located[2], tile):
            continue
        if not blank and _tight_tile(inp, background) is None:
            continue
        return {'input': inp, 'output': tile.copy()}
    raise RuntimeError('could not construct a visible reflected pair')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _checked_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    # The reflection rule is the only universal convention. Some puzzles also
    # demonstrate a shared crop size or background; both are learned in train.
    variable_sizes = bool(rng.integers(0, 2))
    variable_backgrounds = bool(rng.integers(0, 2))
    palette_regime = int(rng.integers(0, 4))
    shapes = [(h, w) for h in range(2, 8) for w in range(2, 8) if h*w >= 8]
    rich_shapes = [shape for shape in shapes if shape[0]*shape[1] >= 20]
    reverse_shapes = [shape for shape in shapes if min(shape) >= 5]
    if variable_sizes:
        train_pool = rich_shapes if palette_regime == 2 else shapes
        train_shapes = [train_pool[int(rng.integers(0, len(train_pool)))]]
        choices = [s for s in train_pool if s != train_shapes[0]]
        train_shapes.append(choices[int(rng.integers(0, len(choices)))])
        train_shapes += [train_pool[int(rng.integers(0, len(train_pool)))] for _ in range(train_count-2)]
        test_pool = [s for s in (rich_shapes if palette_regime == 1 else shapes) if s not in train_shapes]
        test_shapes = [test_pool[int(rng.integers(0, len(test_pool)))]]
        test_shapes += [shapes[int(rng.integers(0, len(shapes)))] for _ in range(test_count-1)]
    else:
        pool = reverse_shapes if palette_regime == 2 else (
            rich_shapes if palette_regime == 1 else [s for s in shapes if min(s) >= 3])
        shared = pool[int(rng.integers(0, len(pool)))]
        train_shapes, test_shapes = [shared]*train_count, [shared]*test_count
    total = train_count + test_count
    if variable_backgrounds:
        order = [int(v) for v in rng.permutation(10)]
        backgrounds = [order[i % 10] for i in range(total)]
    else:
        backgrounds = [int(rng.integers(0, 10))]*total
    # At least one fixed-size training example shows a blank outer boundary;
    # another has a tight boundary, so the shared size is explicit.
    blank_train = int(rng.integers(0, train_count))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for index, (split, shape) in enumerate(
            [('train', s) for s in train_shapes] + [('test', s) for s in test_shapes]):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
        is_train = split == 'train'
        if palette_regime == 1:
            goal = int(pair_rng.integers(2, 5)) if is_train else int(pair_rng.integers(5, 7))
        elif palette_regime == 2:
            goal = int(pair_rng.integers(5, 7)) if is_train else int(pair_rng.integers(2, 5))
        else:
            goal = int(pair_rng.integers(2, 7))
        blank = (not variable_sizes and
                 (index == blank_train if is_train else bool(pair_rng.integers(0, 2))))
        source_like = not is_train and bool(pair_rng.random() < 0.23)
        for _ in range(80):
            pair = _make_pair(pair_rng, shape, backgrounds[index], goal, blank, source_like)
            key = _pair_key(pair)
            if key[0] not in seen and key not in _SOURCE_PAIR_KEYS:
                break
        else:
            raise RuntimeError('could not make a unique pair')
        seen.add(key[0])
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'sections must be lists')
    train, test = puzzle['train'], puzzle['test']
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), 'invalid pair counts')
    seen_inputs, seen_pairs = set(), set()
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair')
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _require(all(1 <= d <= 30 for d in grid.shape), 'grid dimensions out of range')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid needs integer cells')
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, 'invalid ARC color')
            inp, out = pair['input'], pair['output']
            key = _pair_key(pair)
            _require(key[0] not in seen_inputs and key not in seen_pairs, 'duplicate input or pair')
            _require(key not in _SOURCE_PAIR_KEYS, 'source pair copied')
            seen_inputs.add(key[0]); seen_pairs.add(key)
            bg = _background(inp)
            _require(bg in set(int(v) for v in np.unique(out)), 'tile must retain background')
            found = _locate(inp, out.shape, bg)
            _require(found is not None and np.array_equal(found[2], out), 'wrong upper-left crop or reflection')
            row, col, _ = found
            r0, r1, c0, c1 = _visible_bounds(inp, bg)
            tight = (r1-r0, c1-c0) == (2*out.shape[0], 2*out.shape[1])
            _require(not np.array_equal(inp, out), 'untransformed pair')
            _require(not np.array_equal(out, np.fliplr(out)) and not np.array_equal(out, np.flipud(out)), 'orientation ambiguous')
            records.append((split, inp.shape, out.shape, bg, row, col, tight))
    train_shapes = {r[2] for r in records if r[0] == 'train'}
    if len(train_shapes) == 1:
        _require(all(r[2] in train_shapes for r in records), 'untrained crop size')
        _require(any(not r[6] for r in records if r[0] == 'train'), 'training omits padded boundary')
        _require(any(r[6] for r in records if r[0] == 'train'), 'training omits tight boundary')
    else:
        _require(all(r[6] for r in records), 'varying sizes require visible extent')
        _require(any(r[2] not in train_shapes for r in records if r[0] == 'test'), 'test omits a new size')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                bg = _background(pair['input'])
                inferred = _tight_tile(pair['input'], bg)
                _require(inferred is not None and np.array_equal(inferred, pair['output']), 'input alone does not reveal crop')
    _require(len({(r[1], r[2], r[4], r[5]) for r in records}) > 1, 'layouts do not vary')


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, 'insufficient puzzle diversity')
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'train counts do not vary')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'test counts do not vary')
    measurements = {'variable_size': 0, 'variable_bg': 0, 'odd_canvas': 0,
                    'full_span': 0, 'one_cell': 0, 'blank_boundary': 0,
                    'rich_train': 0, 'small_test': 0, 'top_center_test': 0,
                    'mixed_blank_odd': 0, 'size_bg_combo': 0,
                    'forward_palette': 0, 'reverse_palette': 0,
                    'overlap_palette': 0, 'varied_canvas': 0,
                    'varied_placement': 0}
    canvas_shapes, tile_shapes, palettes, densities, positions = set(), set(), set(), set(), set()
    all_inputs, total = set(), 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        shapes, bgs, canvases, placements = set(), set(), set(), set()
        train_palettes, test_palettes = [], []
        has_blank, has_odd = False, False
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                bg = _background(inp)
                found = _locate(inp, out.shape, bg)
                _require(found is not None and np.array_equal(found[2], out), 'self-test transformation failed')
                row, col, _ = found
                block = inp[row:row+2*out.shape[0], col:col+2*out.shape[1]]
                _require(np.array_equal(block[:out.shape[0], out.shape[1]:], np.fliplr(out)), 'right reflection failed')
                _require(np.array_equal(block[out.shape[0]:, :out.shape[1]], np.flipud(out)), 'lower reflection failed')
                _require(_pair_key(pair) not in _SOURCE_PAIR_KEYS, 'source copy')
                shapes.add(out.shape); bgs.add(bg)
                canvases.add(inp.shape); placements.add((row, col))
                canvas_shapes.add(inp.shape); tile_shapes.add(out.shape)
                palettes.add(tuple(int(x) for x in np.unique(out)))
                densities.add(round(float(np.mean(out != bg)), 1))
                positions.add((row, col, inp.shape[0]-row-2*out.shape[0], inp.shape[1]-col-2*out.shape[1]))
                all_inputs.add(_grid_key(inp)); total += 1
                r0, r1, c0, c1 = _visible_bounds(inp, bg)
                blank = (r1-r0, c1-c0) != (2*out.shape[0], 2*out.shape[1])
                has_blank |= blank
                has_odd |= any(n % 2 for n in inp.shape)
                measurements['blank_boundary'] += int(blank)
                measurements['odd_canvas'] += int(any(n % 2 for n in inp.shape))
                measurements['full_span'] += int(inp.shape[0] == 2*out.shape[0] or inp.shape[1] == 2*out.shape[1])
                measurements['one_cell'] += int(row == 1 or col == 1)
                k = len(np.unique(out))-1
                (train_palettes if split == 'train' else test_palettes).append(k)
                measurements['rich_train'] += int(split == 'train' and k >= 5)
                measurements['small_test'] += int(split == 'test' and k <= 4)
                measurements['top_center_test'] += int(split == 'test' and row == 0 and col == (inp.shape[1]-2*out.shape[1])//2)
        measurements['variable_size'] += int(len(shapes) > 1)
        measurements['variable_bg'] += int(len(bgs) > 1)
        measurements['mixed_blank_odd'] += int(has_blank and has_odd)
        measurements['size_bg_combo'] += int(len(shapes) > 1 and len(bgs) > 1)
        measurements['varied_canvas'] += int(len(canvases) > 1)
        measurements['varied_placement'] += int(len(placements) > 1)
        measurements['forward_palette'] += int(min(test_palettes) > max(train_palettes))
        measurements['reverse_palette'] += int(max(test_palettes) < min(train_palettes))
        measurements['overlap_palette'] += int(bool(set(train_palettes) & set(test_palettes)))
    _require(len(canvas_shapes) >= 25 and len(tile_shapes) >= 12, 'shape diversity is low')
    _require(len(palettes) >= 30 and len(densities) >= 5 and len(positions) >= 40, 'motif or layout diversity is low')
    _require(len(all_inputs)*100 >= total*95, 'inputs repeat across seeds')
    for name, minimum in {'variable_size': 8, 'variable_bg': 8, 'odd_canvas': 30,
                          'full_span': 8, 'one_cell': 8, 'blank_boundary': 12,
                          'rich_train': 5, 'small_test': 10, 'top_center_test': 5,
                          'mixed_blank_odd': 5, 'size_bg_combo': 4,
                          'forward_palette': 3, 'reverse_palette': 3,
                          'overlap_palette': 8, 'varied_canvas': 30,
                          'varied_placement': 30}.items():
        _require(measurements[name] >= minimum, f'{name} coverage is low: {measurements[name]}')
    for seed, train, test in ((1001,2,1),(1002,10,1),(1003,10,4),(1004,7,3)):
        p = generate_puzzle(seed, train, test)
        _require((len(p['train']),len(p['test'])) == (train,test), 'explicit counts ignored')
    for seed, train, test in ((2001,None,1),(2002,None,4),(2003,2,None),(2004,10,None)):
        p = generate_puzzle(seed, train, test)
        _require(train is None or len(p['train']) == train, 'partial train ignored')
        _require(test is None or len(p['test']) == test, 'partial test ignored')
    _require(_puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17)), 'nondeterministic automatic puzzle')
    _require(_puzzle_key(generate_puzzle(3001,8,3)) == _puzzle_key(generate_puzzle(3001,8,3)), 'nondeterministic explicit puzzle')
    for kwargs in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},
                   {'num_train':2,'num_test':2},{'num_train':3,'num_test':4},{'num_train':4,'num_test':4}):
        try:
            generate_puzzle(4001, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f'illegal counts accepted: {kwargs}')
