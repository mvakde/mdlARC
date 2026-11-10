import numpy as np


_TRAIN_LARGE = (
    ("111", "101", "111"),
    ("010", "111", "010"),
    ("111", "100", "100"),
    ("100", "110", "011"),
    ("101", "111", "100"),
    ("110", "011", "001"),
)

_TRAIN_SMALL = (
    ("11", "10"),
    ("11", "01"),
    ("10", "11"),
    ("01", "11"),
)

_TEST_SHAPES = (
    ("11", "10"),
    ("111", "101", "111"),
    ("111", "010"),
    ("010", "111", "010"),
    ("100", "110", "011"),
    ("111", "100", "100"),
)

_SOURCE_INPUTS = {
    (
        "0000000000", "0101000000", "0111020200", "0000022200",
        "0000000000", "0000000000", "0006060000", "0006660000",
        "0000000000", "0000000000",
    ),
    (
        "0000000000", "0000000000", "0333007770", "0300007000",
        "0000000000", "0008880000", "0008000000", "0000000000",
        "0000000222", "0000000200",
    ),
    (
        "0000000000", "0000000000", "0000000000", "0000000000",
        "4440000000", "4040000000", "4440008880", "3330008080",
        "3030008880", "3330000000",
    ),
    (
        "0400000000", "0400000000", "0400000002", "0400000002",
        "0000600002", "0000600002", "0000600800", "0000600800",
        "0000000800", "0000000800",
    ),
    (
        "0000000000", "0000004400", "0040004400", "0404000000",
        "4000400000", "0000000000", "0040000000", "0404000000",
        "0404000000", "0000000000",
    ),
    (
        "000000000", "022220000", "023320000", "023320444",
        "022220454", "000000444", "077700000", "071700000",
        "077700000",
    ),
}


def _mask(rows):
    return np.array(
        [[int(ch) for ch in row] for row in rows],
        dtype=bool,
    )


def _counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(
                f"{name} must be an integer from {low} through {high}"
            )

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(
            rng.integers(1, min(4, num_train - 1) + 1)
        )

    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    return int(num_train), int(num_test)


def _shape_mask(rng, kind):
    if kind == 'line':
        m = np.ones((int(rng.integers(2, 7)), 1), bool)
    elif kind == 'ring':
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        m = np.ones((h, w), bool)
        m[1:-1, 1:-1] = False
    elif kind == 'block':
        m = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 6))), bool)
    elif kind == 'cross':
        h, w = int(rng.choice((3, 5))), int(rng.choice((3, 5)))
        m = np.zeros((h, w), bool)
        m[h // 2, :] = True
        m[:, w // 2] = True
    elif kind == 'stair':
        h = int(rng.integers(3, 6))
        m = np.zeros((h, h + 1), bool)
        for r in range(h):
            m[r, r:r + 2] = True
    elif kind == 'poly':
        m = np.zeros((4, 4), bool)
        m[1, 1] = True
        for _ in range(int(rng.integers(4, 9))):
            options = set()
            for r, c in np.argwhere(m):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < 4 and 0 <= nc < 4 and not m[nr, nc]:
                        options.add((nr, nc))
            if options:
                ordered = sorted(options)
                m[ordered[int(rng.integers(len(ordered)))]] = True
        rr, cc = np.where(m)
        m = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    else:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        m = np.zeros((h, w), bool)
        m[:, 0] = True
        m[-1, :] = True
    if kind in ('stair', 'poly', 'L'):
        m = np.rot90(m, int(rng.integers(4)))
        if rng.integers(2):
            m = np.fliplr(m)
    return m


def _compound(rng, outer, inner, bg):
    thickness = 2 if rng.integers(4) == 0 else 1
    low = 2 * thickness + 1
    h, w = int(rng.integers(low, 8)), int(rng.integers(low, 8))
    tile = np.full((h, w), bg, dtype=np.int8)
    tile[:thickness, :] = outer
    tile[-thickness:, :] = outer
    tile[:, :thickness] = outer
    tile[:, -thickness:] = outer
    ch = int(rng.integers(1, h - 2 * thickness + 1))
    cw = int(rng.integers(1, w - 2 * thickness + 1))
    r = int(rng.integers(thickness, h - ch - thickness + 1))
    c = int(rng.integers(thickness, w - cw - thickness + 1))
    tile[r:r + ch, c:c + cw] = inner
    return tile


def _apart(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _place(rng, tiles, h, w, touch):
    n = len(tiles)
    sizes = [tile.shape for tile in tiles]
    wh, ww = sizes[-1]
    bottom = h - 1 - int(rng.integers(0, min(5, h - 5)))
    if touch:
        bottom = max(bottom, wh + sizes[0][0] - 1)
    if bottom >= h or bottom < wh - 1:
        return None
    origins = [(bottom - wh + 1, c) for c in range(w - ww + 1)]
    rng.shuffle(origins)
    for wr, wc in origins:
        boxes = [None] * n
        boxes[-1] = (wr, wc, wh, ww)
        if touch:
            th, tw = sizes[0]
            tc = min(wc, w - tw)
            boxes[0] = (wr - th, tc, th, tw)
            if boxes[0][0] < 0:
                continue
        order = [i for i in range(n - 1) if boxes[i] is None]
        order.sort(key=lambda i: -sizes[i][0] * sizes[i][1])
        for i in order:
            mh, mw = sizes[i]
            candidates = [(r, c, mh, mw)
                          for r in range(min(h - mh + 1, bottom - mh + 1))
                          for c in range(w - mw + 1)
                          if all(_apart((r, c, mh, mw), b) for b in boxes if b is not None)]
            if not candidates:
                break
            boxes[i] = candidates[int(rng.integers(len(candidates)))]
        if all(b is not None for b in boxes):
            return boxes
    return None


def _signature(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _blobs(grid, bg):
    seen = np.zeros(grid.shape, bool)
    blobs = []
    for r, c in np.argwhere(grid != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        color = grid[r, c]
        stack = [(r, c)]
        seen[r, c] = True
        cells = set()
        while stack:
            cr, cc = stack.pop()
            cells.add((cr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        blobs.append(cells)
    return blobs


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs), max(cs)


def _objects(grid, bg):
    blobs = _blobs(grid, bg)
    boxes = [_bbox(b) for b in blobs]
    owners = list(range(len(blobs)))
    for i, outer in enumerate(blobs):
        r0, c0, r1, c1 = boxes[i]
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)}
        if not border <= outer:
            continue
        for j in range(len(blobs)):
            if i == j:
                continue
            a, b, d, e = boxes[j]
            if r0 < a <= d < r1 and c0 < b <= e < c1:
                owners[j] = i
    groups = {}
    for i, cells in enumerate(blobs):
        groups.setdefault(owners[i], set()).update(cells)
    return list(groups.values())


def _predict(inp, bg):
    objects = _objects(inp, bg)
    assert 2 <= len(objects) <= 6
    bottoms = [max(r for r, c in obj) for obj in objects]
    assert bottoms.count(max(bottoms)) == 1
    out = np.full_like(inp, bg)
    for r, c in objects[bottoms.index(max(bottoms))]:
        out[r, c] = inp[r, c]
    return out


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _make_pair(rng, bg, regime, index, teach_colors=None):
    first = regime == 'train' and index == 0
    available = [c for c in range(10) if c != bg]
    for _ in range(180):
        if first:
            h = w = 10
            n, profile = 3, 'same_shape'
        else:
            h, w = int(rng.integers(9, 17)), int(rng.integers(9, 17))
            if regime == 'train' and rng.integers(3) == 0:
                h = w = 10
            n = int(rng.integers(3, 6 if min(h, w) >= 12 else 5))
            profiles = ('line', 'equal_line', 'same_color', 'compound', 'mixed', 'same_shape')
            profile = profiles[int(rng.integers(len(profiles)))]
            if regime == 'test' and profile == 'same_shape':
                profile = 'mixed'
        outer = [int(available[int(rng.integers(len(available)))]) for _ in range(n)]
        if first:
            outer = [int(x) for x in rng.choice(available, size=n, replace=False)]
        elif profile == 'same_color' or (profile == 'mixed' and rng.integers(7) == 0):
            outer = [outer[0]] * n
        if teach_colors is not None:
            outer[0], outer[-1] = teach_colors
        if profile in ('line', 'equal_line'):
            kinds = ['line'] * n
        elif profile == 'compound':
            kinds = ['compound' if i == n - 1 or rng.integers(3) == 0 else str(rng.choice(('line', 'ring', 'block', 'poly'))) for i in range(n)]
        elif profile == 'same_shape':
            kinds = [str(rng.choice(('ring', 'cross', 'L', 'block', 'stair')))] * n
        else:
            kinds = [str(rng.choice(('line', 'ring', 'block', 'cross', 'stair', 'poly', 'L', 'compound'))) for _ in range(n)]
        touch = not first and profile not in ('line', 'equal_line', 'same_color') and rng.integers(5) == 0
        if touch:
            kinds[0] = kinds[-1] = 'ring'
            if outer[0] == outer[-1]:
                outer[0] = next(c for c in available if c != outer[-1])
        common = _shape_mask(rng, kinds[0]) if profile == 'same_shape' else None
        contact_mask = _shape_mask(rng, 'ring') if touch else None
        line_length = int(rng.integers(2, 6)) if profile == 'equal_line' else None
        tiles = []
        for i, kind in enumerate(kinds):
            if kind == 'compound':
                inner_choices = [c for c in available if c != outer[i] and not (teach_colors is not None and i == n - 1 and c == teach_colors[0])]
                inner = int(rng.choice(inner_choices))
                tile = _compound(rng, outer[i], inner, bg)
            else:
                mask = contact_mask.copy() if touch and i in (0, n - 1) else common.copy() if common is not None else _shape_mask(rng, kind)
                if line_length is not None:
                    mask = np.ones((line_length, 1), bool)
                tile = np.where(mask, outer[i], bg).astype(np.int8)
            tiles.append(tile)
        if any(t.shape[0] >= h or t.shape[1] > w for t in tiles):
            continue
        boxes = _place(rng, tiles, h, w, touch)
        if boxes is None:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        for tile, (r, c, mh, mw) in zip(tiles, boxes):
            view = inp[r:r + mh, c:c + mw]
            view[tile != bg] = tile[tile != bg]
        out = np.full_like(inp, bg)
        r, c, mh, mw = boxes[-1]
        out[r:r + mh, c:c + mw] = tiles[-1]
        try:
            if np.array_equal(_predict(inp, bg), out) and _signature(inp) not in _SOURCE_INPUTS:
                return {'input': inp, 'output': out}
        except AssertionError:
            pass
    raise RuntimeError('bounded object placement exhausted')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    bg = 0 if rng.integers(6) else int(rng.integers(1, 10))
    puzzle = {'train': [], 'test': []}
    for regime, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teach_colors = None
            if regime == 'train' and index == 1:
                demonstration = puzzle['train'][0]
                first_winner = int(next(v for v in demonstration['output'].flat if v != bg))
                first_loser = int(next(v for v in demonstration['input'].flat if v not in (bg, first_winner)))
                teach_colors = (first_winner, first_loser)
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, bg, regime, index, teach_colors)
                if all(not np.array_equal(pair['input'], old['input']) for old in puzzle['train'] + puzzle['test']):
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError('could not make a unique pair')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    train_bg = [_background(p['output']) for p in puzzle['train']]
    assert len(set(train_bg)) == 1
    bg = train_bg[0]
    seen_inputs, seen_pairs = set(), set()
    for regime in ('train', 'test'):
        for index, pair in enumerate(puzzle[regime]):
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _background(out) == bg
            ik = (inp.shape, inp.tobytes())
            pk = (ik, out.tobytes())
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            assert _signature(inp) not in _SOURCE_INPUTS
            assert np.array_equal(out, _predict(inp, bg))
    demonstrations = []
    for pair in puzzle['train']:
        winning = {int(v) for v in np.unique(pair['output']) if v != bg}
        losing = {int(v) for v in np.unique(pair['input']) if v != bg} - winning
        demonstrations.append((winning, losing))
    assert any(a[0] & b[1] and a[1] & b[0] for i, a in enumerate(demonstrations) for b in demonstrations[i + 1:])
    def simple(pair):
        inp = pair['input']
        objs = _objects(inp, bg)
        forms = []
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            forms.append(tuple(tuple(int((r, c) in obj) for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1)))
        return inp.shape == (10, 10) and len(objs) == 3 and len(set(forms)) == 1 and all(len({int(inp[r, c]) for r, c in obj}) == 1 for obj in objs)
    assert any(simple(pair) for pair in puzzle['train'])


def _same_puzzle(a, b):
    return all(len(a[k]) == len(b[k]) and all(np.array_equal(x[s], y[s]) for x, y in zip(a[k], b[k]) for s in ('input', 'output')) for k in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, dimensions, counts = set(), set(), set(), set()
    silhouettes, palettes, layouts, margin_buckets, signatures = set(), set(), set(), set(), set()
    coverage = {name: 0 for name in ('touch', 'compound', 'core4', 'thick', 'single_color', 'mixed_shape', 'smaller_winner', 'less_colorful_winner', 'one_row_gap', 'high_margin', 'equal_lines', 'unequal_lines', 'four_lines', 'same_color_mixed_test', 'novel_test_color', 'nonzero_background', 'source_like_shift', 'compound_train', 'compound_winner')}
    within = {name: 0 for name in ('dimensions', 'palettes', 'counts', 'shapes', 'margins')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(tuple(_signature(p['input']) for p in puzzle['train'] + puzzle['test']))
        bg = _background(puzzle['train'][0]['output'])
        coverage['nonzero_background'] += int(bg != 0)
        train_colors = {int(v) for p in puzzle['train'] for v in np.unique(p['input']) if v != bg}
        test_colors = {int(v) for p in puzzle['test'] for v in np.unique(p['input']) if v != bg}
        coverage['novel_test_color'] += int(bool(test_colors - train_colors))
        local = {name: set() for name in within}
        train_compound = test_compound = False
        for split, pair in [(split, p) for split in ('train', 'test') for p in puzzle[split]]:
            inp = pair['input']
            objs = _objects(inp, bg)
            assert np.array_equal(_predict(inp, bg), pair['output'])
            dimensions.add(inp.shape)
            counts.add(len(objs))
            palette = tuple(int(c) for c in np.unique(inp) if c != bg)
            palettes.add(palette)
            layouts.add((inp.shape, tuple(sorted(_bbox(o) for o in objs))))
            bottoms = sorted(max(r for r, c in o) for o in objs)
            margin, gap = inp.shape[0] - 1 - bottoms[-1], bottoms[-1] - bottoms[-2]
            margin_buckets.add((margin, gap))
            local['dimensions'].add(inp.shape)
            local['palettes'].add(palette)
            local['counts'].add(len(objs))
            local['margins'].add(margin)
            forms = []
            for obj in objs:
                r0, c0, r1, c1 = _bbox(obj)
                form = tuple(tuple(int((r, c) in obj) for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))
                forms.append(form)
                silhouettes.add(form)
            local['shapes'].add(tuple(sorted(str(f) for f in forms)))
            colors = [{int(inp[r, c]) for r, c in obj} for obj in objs]
            winner = next(i for i, obj in enumerate(objs) if all(pair['output'][r, c] == inp[r, c] for r, c in obj))
            losers = [i for i in range(len(objs)) if i != winner]
            coverage['mixed_shape'] += int(len(set(forms)) > 1)
            coverage['single_color'] += int(len(set().union(*colors)) == 1)
            coverage['smaller_winner'] += int(len(objs[winner]) < max(len(objs[i]) for i in losers))
            coverage['less_colorful_winner'] += int(len(colors[winner]) < max(len(colors[i]) for i in losers))
            coverage['one_row_gap'] += int(gap == 1)
            coverage['high_margin'] += int(margin >= 3)
            coverage['compound_winner'] += int(len(colors[winner]) > 1)
            has_compound = any(len(co) > 1 for co in colors)
            coverage['compound'] += int(has_compound)
            if split == 'train':
                train_compound |= has_compound
            else:
                test_compound |= has_compound
            for obj, co in zip(objs, colors):
                if len(co) < 2:
                    continue
                r0, c0, r1, c1 = _bbox(obj)
                outer = int(inp[r0, c0])
                coverage['core4'] += int(sum(int(inp[r, c] != outer) for r, c in obj) >= 4)
                if r1 - r0 >= 4 and c1 - c0 >= 4:
                    border2 = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if r - r0 < 2 or r1 - r < 2 or c - c0 < 2 or c1 - c < 2}
                    coverage['thick'] += int(all((r, c) in obj and inp[r, c] == outer for r, c in border2))
            coverage['touch'] += int(any(any(max(abs(r-u), abs(c-v)) <= 1 for r, c in obj for u, v in other) for i, obj in enumerate(objs) for other in objs[i + 1:]))
            if split == 'test':
                if all(len({c for r, c in obj}) == 1 for obj in objs):
                    lengths = {len(obj) for obj in objs}
                    coverage['equal_lines'] += int(len(lengths) == 1)
                    coverage['unequal_lines'] += int(len(lengths) > 1)
                    coverage['four_lines'] += int(len(objs) == 4)
                coverage['same_color_mixed_test'] += int(len(set().union(*colors)) == 1 and len(set(forms)) > 1)
        coverage['source_like_shift'] += int(test_compound and not train_compound)
        coverage['compound_train'] += int(train_compound)
        for name, values in local.items():
            within[name] += int(len(values) > 1)
        if seed < 4:
            assert _same_puzzle(puzzle, generate_puzzle(seed))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 15 and len(counts) >= 2
    assert len(palettes) >= 20 and len(layouts) >= 100
    assert len(silhouettes) >= 20 and len(margin_buckets) >= 6
    assert len(signatures) == 42
    assert all(within[name] >= 38 for name in within)
    assert all(coverage[name] >= minimum for name, minimum in {
        'touch': 8, 'compound': 20, 'core4': 5, 'thick': 3,
        'single_color': 8, 'mixed_shape': 100, 'smaller_winner': 40,
        'less_colorful_winner': 5, 'one_row_gap': 40, 'high_margin': 30,
        'equal_lines': 5, 'unequal_lines': 5, 'four_lines': 5,
        'same_color_mixed_test': 3, 'novel_test_color': 5,
        'nonzero_background': 2, 'source_like_shift': 2,
        'compound_train': 10, 'compound_winner': 10,
    }.items())
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(103, train, test)
        assert train is None or len(puzzle['train']) == train
        assert test is None or len(puzzle['test']) == test
        assert _same_puzzle(puzzle, generate_puzzle(103, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
