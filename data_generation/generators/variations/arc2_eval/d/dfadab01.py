import numpy as np


# An exact source pair necessarily has an exact source input. Rejecting these
# five inputs therefore rejects every exact source input/output pair.
_SOURCE_INPUT_TEXTS = {
    (
        "00000000000000000000/06600000200002000000/"
        "06600000000000000000/00066000000000000000/"
        "00066000000000000000/00000500000000000000/"
        "00000000200002000000/00000000000000000000/"
        "00000000000000000000/00000000000000000000/"
        "00000000000000000000/00300000500005000000/"
        "00000000000000000000/00000000000000000000/"
        "00000000000000000000/00000000000000000000/"
        "00300000300003000000/00000000000000000000/"
        "00000000000000000000/00000000000000000000"
    ),
    (
        "0000000000/0300020000/0000000000/0000000000/0000000000/"
        "0200031100/0000010010/0000010010/0000001100/0000000003"
    ),
    (
        "4444002000/4004000000/4004000000/4444000000/0000200000/"
        "0000000000/2000002000/0000000000/0000000000/0000000000"
    ),
    (
        "20002000500080008000/00000000000000000000/"
        "00000000000000000000/00000000000000000000/"
        "20003000200030008000/00000000000000000000/"
        "00000000000000000000/00000000000000000000/"
        "50003000500020005000/00000000000000000000/"
        "00000000000000000000/00000000000000000000/"
        "80003000200030002000/00000000000000000000/"
        "00000000000000000000/00000000000000000000/"
        "80008000500020002000/00000000000000000000/"
        "00000000000000000000/00000000000000000000"
    ),
    (
        "00000000000000000000/80008000800080008000/"
        "00000000000000000000/00000000000000000000/"
        "00000000000000000000/80007007000000008000/"
        "00000770000000000000/00000770000000000000/"
        "00007007000000000000/80000000800000008000/"
        "00000000000000000000/00000000000000000000/"
        "00000000000000000000/80000000000080008000/"
        "00000000000000000000/00000000000000000000/"
        "00000000000000000000/80008000800080008000/"
        "00000000000000000000/00000000000000000000"
    ),
}


def _input_text(grid):
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid):
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _validate_optional_count(name, value, low, high):
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= int(value) <= high
    ):
        raise ValueError(
            f"{name} must be an integer from {low} through {high}"
        )


def _resolve_counts(seed, num_train, num_test):
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + (seed // 4) % (11 - minimum_train)
        )
    elif num_train is None:
        test_count = int(num_test)
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + seed % (11 - minimum_train)
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(grid, background):
    """Eight-connected input objects; diagonal attachments are significant."""
    remaining = set(map(tuple, np.argwhere(grid != background)))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
        objects.append(cells)
    return objects


def _members(grid, codes, background):
    live, rejected, bare, painted = [], [], [], []
    for cells in _components(grid, background):
        points = [(r, c, int(grid[r, c])) for r, c in cells
                  if int(grid[r, c]) in codes]
        origin = (min(r for r, _ in cells), min(c for _, c in cells))
        if len(points) == 1 and points[0][:2] == origin:
            live.append(points[0])
            if len(cells) > 1:
                painted.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    return live, rejected, bare, painted


def _stamp(grid, r, c, ink, mask):
    h = min(mask.shape[0], grid.shape[0] - r)
    w = min(mask.shape[1], grid.shape[1] - c)
    grid[r:r + h, c:c + w][mask[:h, :w]] = ink


def _render_records(shape, background, records, book):
    out = np.full(shape, background, dtype=np.int8)
    for r, c, code in records:
        _stamp(out, r, c, *book[code])
    return out


def _predict(grid, book):
    """Only input and a training-derived dictionary affect membership."""
    bg = _background(grid)
    records = _members(grid, set(book), bg)[0]
    return _render_records(grid.shape, bg, records, book)


def _learn(train_pairs):
    """Recover the dictionary from observable training arrays, without metadata.

    Enumerate common bounding-box sizes. A candidate needs an unobscured full
    occurrence of every code and must explain every training pixel. Tight mask
    bounds rule out invisible padding; no test output enters this procedure.
    """
    in_colors, out_colors = set(), set()
    for pair in train_pairs:
        bg = _background(pair['input'])
        in_colors.update(set(map(int, np.unique(pair['input']))) - {bg})
        out_colors.update(set(map(int, np.unique(pair['output']))) - {bg})
    codes = in_colors - out_colors
    assert 1 <= len(codes) <= 4, 'unrecoverable code colors'
    observations = []
    for pair in train_pairs:
        inp, out = pair['input'], pair['output']
        bg = _background(inp)
        records = _members(inp, codes, bg)[0]
        observations.append((inp, out, bg, records))
    candidates = []
    for h in range(3, 6):
        for w in range(3, 6):
            book, consistent = {}, True
            for inp, out, bg, records in observations:
                for r, c, code in records:
                    if r + h > inp.shape[0] or c + w > inp.shape[1]:
                        continue
                    block = out[r:r + h, c:c + w]
                    colors = set(map(int, np.unique(block))) - {bg}
                    if len(colors) != 1:
                        consistent = False
                        break
                    ink = colors.pop()
                    mask = block == ink
                    if not (mask[0].any() and mask[-1].any()
                            and mask[:, 0].any() and mask[:, -1].any()):
                        consistent = False
                        break
                    if code in book:
                        if ink != book[code][0] or not np.array_equal(mask, book[code][1]):
                            consistent = False
                            break
                    else:
                        book[code] = (ink, mask.copy())
                if not consistent:
                    break
            if not consistent or set(book) != codes:
                continue
            if all(np.array_equal(_render_records(inp.shape, bg, rec, book), out)
                   for inp, out, bg, rec in observations):
                candidates.append(book)
    assert len(candidates) == 1, 'training must identify one complete dictionary'
    return candidates[0]


def _sample_mask(rng, h, w):
    """Connected motifs with tight bounds, including compositional asymmetry."""
    family = int(rng.integers(8))
    mask = np.zeros((h, w), dtype=bool)
    if family == 0:                         # rectangular loop
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif family == 1:                       # rounded loop, blank anchor pixel
        mask[0, 1:-1] = mask[-1, 1:-1] = True
        mask[1:-1, 0] = mask[1:-1, -1] = True
    elif family == 2:                       # solid block
        mask[:] = True
    elif family == 3:                       # two diagonal blocks
        for r in range(h):
            mask[r, : (w + 1) // 2] = r < (h + 1) // 2
            mask[r, w // 2:] |= r >= h // 2
    elif family == 4:                       # X at rectangular scales
        for r in range(h):
            c = round(r * (w - 1) / (h - 1))
            mask[r, c] = mask[r, w - 1 - c] = True
    elif family == 5:                       # asymmetric corner and interior arm
        mask[:, 0] = mask[-1, :] = True
        mask[int(rng.integers(h - 1)), :int(rng.integers(2, w + 1))] = True
    else:                                  # branched monotone path, not a bank
        r = c = 0
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            if r == h - 1 or (c < w - 1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            mask[r, c] = True
        for _ in range(int(rng.integers(1, h * w // 2 + 1))):
            cells = np.argwhere(mask)
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = ((0, 1), (0, -1), (1, 0), (-1, 0))[int(rng.integers(4))]
            if 0 <= r + dr < h and 0 <= c + dc < w:
                mask[r + dr, c + dc] = True
    # Rectangular Xs can have a two-cell jump. Connect all foreground before
    # reflecting it, so a rendered object cannot split into nuisance singletons.
    while len(_components(mask.astype(np.int8), 0)) > 1:
        objects = _components(mask.astype(np.int8), 0)
        a, b = min(((a, b) for a in objects[0] for b in objects[1]),
                   key=lambda p: abs(p[0][0] - p[1][0]) + abs(p[0][1] - p[1][1]))
        r, c = a
        while (r, c) != b:
            r += int(np.sign(b[0] - r))
            c += int(np.sign(b[1] - c))
            mask[r, c] = True
    if rng.random() < .5:
        mask = mask[::-1, :]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    if h == w:
        mask = np.rot90(mask, int(rng.integers(4)))
    return mask.copy()


def _parameters(rng):
    count = int(rng.integers(1, 5))
    palette = rng.permutation(10)
    h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    book = {int(palette[i]): (int(palette[count + i]), _sample_mask(rng, h, w))
            for i in range(count)}
    backgrounds = tuple(map(int, palette[2 * count:2 * count + 2]))
    return book, backgrounds, bool(rng.random() < .75)


def _axis(rng, length, size, clipped, regular=False):
    end = length - int(rng.integers(2, size)) if clipped else length - size
    first = int(rng.integers(0, min(5, end - size + 1 if clipped else end + 1)))
    positions = [first]
    pitch = size + int(rng.integers(4))
    while True:
        step = pitch if regular else size + int(rng.integers(5))
        nxt = positions[-1] + step
        if nxt > (end - size if clipped else end):
            break
        positions.append(nxt)
    if clipped:
        positions.append(end)
    return positions


def _rect_mask(shape, r, c, h, w):
    mask = np.zeros(shape, dtype=bool)
    mask[r:min(shape[0], r + h), c:min(shape[1], c + w)] = True
    return mask


def _halo(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[dr:dr + h, dc:dc + w]
                                 for dr in range(3) for dc in range(3)])


def _try_material(inp, occupied, book, code, r, c, kind, bg):
    ink, mask = book[code]
    h, w = mask.shape
    if r + h > inp.shape[0] or c + w > inp.shape[1]:
        return False
    obj = np.full(inp.shape, bg, dtype=np.int8)
    _stamp(obj, r, c, ink, mask)
    if kind == 'attached':
        last = max(zip(*np.nonzero(mask)))
        ar, ac = r + int(last[0]) + 1, c + int(last[1]) + 1
        if ar >= inp.shape[0] or ac >= inp.shape[1]:
            return False
        obj[ar, ac] = code
    elif kind == 'painted':
        if mask[0, 0]:
            return False
        obj[r, c] = code
        if len(_components(obj, bg)) != 1:
            return False
    fg = obj != bg
    old = inp != bg
    if kind == 'painted':
        old = old.copy()
        old[r, c] = False
    elif np.any(fg & occupied):
        return False
    if np.any(_halo(fg) & old):
        return False
    inp[fg] = obj[fg]
    return True


def _make_pair(rng, book, backgrounds, required, features, density, bg,
               avoid_shape=None):
    codes = list(book)
    h, w = next(iter(book.values()))[1].shape
    subset = list(map(int, rng.choice(codes, int(rng.integers(1, len(codes) + 1)), replace=False)))
    subset = list(dict.fromkeys(list(required) + subset))
    # The final attempt is a spacious deterministic construction, independent
    # of probabilistic packing success. All earlier attempts are bounded.
    for attempt in range(41):
        fallback = attempt == 40
        scale = rng.random()
        if fallback or scale < .06:
            H, W = 30, 30
        elif scale < .18:
            H, W = 10, 10
        else:
            H, W = int(rng.integers(10, 31)), int(rng.integers(10, 31))
        if fallback and (H, W) == avoid_shape:
            H = 29
        if (H, W) == avoid_shape and not fallback:
            continue
        clip = 'clip' in features or rng.random() < .38
        cr = clip and ('clip' in features or rng.random() < .7)
        cc = clip and ('clip' in features or not cr or rng.random() < .65)
        if fallback:
            rows, cols = [0, 7, 14, 21, H - 2], [0, 7, 14, 21, W - 2]
            cr = cc = True
        else:
            regular = rng.random() < .4
            rows = _axis(rng, H, h, cr, regular)
            cols = _axis(rng, W, w, cc, regular)
        anchors = [(r, c) for r in rows for c in cols]
        mode = int(rng.choice(6, p=density)) if not fallback else 4
        if mode == 5 and not features and not fallback:
            # Greedy free placement uses visible rectangles, not latent rows.
            anchors = []
            used = np.zeros((H, W), dtype=bool)
            for _ in range(120):
                r = int(rng.integers(H - (1 if cr else h) + 1))
                c = int(rng.integers(W - (1 if cc else w) + 1))
                rect = _rect_mask((H, W), r, c, h, w)
                if not np.any(rect & used) and all(max(abs(r - a), abs(c - b)) > 1 for a, b in anchors):
                    anchors.append((r, c))
                    used |= rect
                if len(anchors) >= int(rng.integers(4, 17)):
                    break
        full = [q for q in anchors if q[0] + h <= H and q[1] + w <= W]
        extra_paint = int('painted' in features)
        if len(full) < max(1, len(required) + extra_paint):
            continue
        rng.shuffle(full)
        selected = set(full[:max(1, len(required) + extra_paint)])
        if mode == 0:
            selected.update(anchors)                         # complete
        elif mode == 1:
            selected.update(q for q in anchors if rng.random() < .24)
        elif mode == 2:
            selected.update(q for q in anchors if q[0] in (rows[0], rows[-1]) or q[1] in (cols[0], cols[-1]))
        elif mode in (3, 5):
            selected.update(q for q in anchors if rng.random() < rng.uniform(.4, .85))
        else:
            indices = rng.permutation(len(anchors))[:int(rng.integers(1, 5))]
            selected.update(anchors[i] for i in indices)
        if 'clip' in features:
            selected.update(((rows[0], cols[-1]), (rows[-1], cols[0]), (rows[-1], cols[-1])))
        assignments = {q: int(rng.choice(subset)) for q in sorted(selected)}
        for q, code in zip(full, required):
            assignments[q] = code
        if extra_paint:
            eligible = [c for c in subset if not book[c][1][0, 0]
                        and book[c][1][:2, :2].any()]
            assignments[full[len(required)]] = int(rng.choice(eligible))
        records = [(r, c, code) for (r, c), code in sorted(assignments.items())]
        # A clipped blank corner is still a legitimate instruction, as in the
        # source loop at (9,9). Its fully visible training copies teach the mask.
        inp = np.full((H, W), bg, dtype=np.int8)
        occupied = np.zeros((H, W), dtype=bool)
        for r, c, code in records:
            inp[r, c] = code
            occupied |= _rect_mask(inp.shape, r, c, h, w)
        desired = set(features) - {'clip'}
        actual = set()
        if 'painted' in desired:
            for r, c, code in records:
                if (r, c) in full[:len(required)]:
                    continue  # preserve the unobscured dictionary demonstrations
                if _try_material(inp, occupied, book, code, r, c, 'painted', bg):
                    actual.add('painted')
                    if rng.random() < .75:
                        break
        for kind in ('attached', 'bare'):
            if kind not in desired:
                continue
            placed = 0
            locations = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
            rng.shuffle(locations)
            desired_count = int(rng.integers(1, 4))
            for r, c in locations:
                code = int(rng.choice(codes))
                if _try_material(inp, occupied, book, code, r, c, kind, bg):
                    placed += 1
                    actual.add(kind)
                if placed >= desired_count:
                    break
        if not desired <= actual:
            continue
        live = _members(inp, set(book), bg)[0]
        if sorted(live) != sorted(records):
            continue
        out = _render_records(inp.shape, bg, records, book)
        if np.count_nonzero(inp == bg) <= inp.size // 2:
            continue
        return {'input': inp, 'output': out}
    raise AssertionError('deterministic packing fallback failed')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    book, backgrounds, rich = _parameters(rng)
    required = [set() for _ in range(nt)]
    for code in book:
        for i in rng.choice(nt, 2, replace=False):
            required[int(i)].add(code)
    features = [set() for _ in range(nt)]
    features[int(rng.integers(nt))].add('clip')
    paint_codes = [code for code, (_, mask) in book.items()
                   if not mask[0, 0] and mask[:2, :2].any()]
    if rich:
        for feature in ('bare', 'attached'):
            features[int(rng.integers(nt))].add(feature)
        if paint_codes:
            i = int(rng.integers(nt))
            features[i].add('painted')
            required[i].add(int(rng.choice(paint_codes)))
    ordinary = [.18, .18, .16, .25, .13, .10]
    sparse = [.06, .32, .20, .16, .18, .08]
    dense = [.38, .07, .18, .24, .05, .08]
    shift = int(rng.choice(3, p=[.5, .25, .25]))
    train_density, test_density = [(ordinary, ordinary), (sparse, dense), (dense, sparse)][shift]
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + nq):
        split = 'train' if i < nt else 'test'
        pair_features = set(features[i]) if i < nt else set()
        if rich and rng.random() < .55:
            pair_features.update(f for f in ('bare', 'attached') if rng.random() < .55)
            if paint_codes and rng.random() < .35:
                pair_features.add('painted')
        needed = sorted(required[i]) if i < nt else []
        if 'painted' in pair_features and not any(c in paint_codes for c in needed):
            needed = needed + [int(rng.choice(paint_codes))]
        bg = backgrounds[i] if i < 2 else int(rng.choice(backgrounds))
        avoid = puzzle['train'][0]['input'].shape if i == 1 else None
        for retry in range(12):
            pair = _make_pair(rng, book, backgrounds, needed, pair_features,
                              train_density if i < nt else test_density, bg, avoid)
            key = _array_key(pair['input'])
            if key not in seen and _input_text(pair['input']) not in _SOURCE_INPUT_TEXTS:
                break
        else:
            raise AssertionError('could not construct a distinct pair')
        seen.add(key)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_material(inp, book, bg):
    """Only the three source-supported rendered-object cases are generated."""
    inks = {ink: code for code, (ink, _) in book.items()}
    for cells in _components(inp, bg):
        if len(cells) == 1 and int(inp[cells[0]]) in book:
            continue
        ink_values = {int(inp[q]) for q in cells if int(inp[q]) in inks}
        assert len(ink_values) == 1, 'a composite must contain one learned ink'
        code = inks[ink_values.pop()]
        ink, mask = book[code]
        r, c = min(q[0] for q in cells), min(q[1] for q in cells)
        expected = {(r + int(a), c + int(b)): ink for a, b in np.argwhere(mask)}
        points = [q for q in cells if int(inp[q]) in book]
        if points:
            assert len(points) == 1 and int(inp[points[0]]) == code
            p = points[0]
            if p == (r, c):
                assert not mask[0, 0]
            else:
                a, b = max(zip(*np.nonzero(mask)))
                assert p == (r + int(a) + 1, c + int(b) + 1)
            expected[p] = code
        assert {q: int(inp[q]) for q in cells} == expected, 'unexplained drawn material'


def _pair_metrics(pair, book):
    inp = pair['input']
    bg = _background(inp)
    live, rejected, bare, painted = _members(inp, set(book), bg)
    h, w = next(iter(book.values()))[1].shape
    H, W = inp.shape
    rows = sorted({r for r, _, _ in live})
    cols = sorted({c for _, c, _ in live})
    bottom = sum(r + h > H for r, _, _ in live)
    right = sum(c + w > W for _, c, _ in live)
    corner = sum(r + h > H and c + w > W for r, c, _ in live)
    full = sum(r + h <= H and c + w <= W for r, c, _ in live)
    complete = len(live) == len(rows) * len(cols)
    interior = [(r, c) for r in rows[1:-1] for c in cols[1:-1]]
    positions = {(r, c) for r, c, _ in live}
    margins = (min(rows), max(0, H - max(rows) - h), min(cols), max(0, W - max(cols) - w))
    asym = any(not np.array_equal(m, m[::-1, :]) or not np.array_equal(m, m[:, ::-1]) for _, m in book.values())
    return {
        'shape': (H, W), 'background': bg, 'live': len(live),
        'classes': len({code for _, _, code in live}),
        'code_set': tuple(sorted({code for _, _, code in live})),
        'bottom': bottom, 'right': right, 'corner': corner, 'full': full,
        'invisible': sum(not book[code][1][:H-r, :W-c].any() for r, c, code in live),
        'rows': len(rows), 'columns': len(cols), 'complete': complete,
        'occupancy': len(live) / (len(rows) * len(cols)),
        'empty_interior': bool(interior) and not any(q in positions for q in interior),
        'unequal_gaps': len(set(np.diff(rows))) > 1 or len(set(np.diff(cols))) > 1,
        'free_geometry': any(g < h for g in np.diff(rows)) or any(g < w for g in np.diff(cols)),
        'margins': margins, 'edge': min(rows) == 0 or min(cols) == 0 or bool(bottom or right),
        'bare': len(bare), 'attached': len(rejected), 'painted': len(painted),
        'interior_attached': sum(r + h <= H and c + w <= W for r, c, _ in rejected),
        'clean': not (rejected or bare or painted), 'mask_shape': (h, w), 'asymmetric': asym,
        'combined': bool(asym and (bottom or right) and rejected and len(live) < len(rows) * len(cols)),
        'shortcut_fails': not np.array_equal(pair['output'], _render_records(
            inp.shape, bg,
            [(int(r), int(c), int(inp[r, c])) for r, c in np.argwhere(np.isin(inp, list(book)))
             if r + h <= H and c + w <= W], book)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _array_key(inp) not in seen
            seen.add(_array_key(inp))
            assert _input_text(inp) not in _SOURCE_INPUT_TEXTS
    # This is computed once from training. Test labels never determine which
    # input markers are legitimate, even when an entire output stamp is missing.
    book = _learn(puzzle['train'])
    inks = {ink for ink, _ in book.values()}
    assert len(inks) == len(book) and not inks.intersection(book)
    backgrounds = {_background(p['input']) for s in puzzle.values() for p in s}
    assert backgrounds.isdisjoint(set(book) | inks)
    assert len({_background(p['input']) for p in puzzle['train']}) >= 2
    assert len({p['input'].shape for p in puzzle['train']}) >= 2
    taught = {code: 0 for code in book}
    train_features, test_features = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            inp, out = pair['input'], pair['output']
            bg = _background(inp)
            assert np.count_nonzero(inp == bg) > inp.size // 2
            assert set(map(int, np.unique(inp))) <= {bg} | set(book) | inks
            assert np.array_equal(_predict(inp.copy(), book), out), 'incorrect predicted output'
            _check_material(inp, book, bg)
            live, rejected, bare, painted = _members(inp, set(book), bg)
            assert live
            occupied = np.zeros(inp.shape, dtype=bool)
            features = train_features if split == 'train' else test_features
            if bare:
                features.add('bare')
            if rejected:
                features.add('attached')
            if painted:
                features.add('painted')
            for r, c, code in live:
                ink, mask = book[code]
                h, w = mask.shape
                rect = _rect_mask(inp.shape, r, c, h, w)
                assert not np.any(occupied & rect), 'conflicting stamp windows'
                occupied |= rect
                if r + h > inp.shape[0]:
                    features.add('bottom')
                if c + w > inp.shape[1]:
                    features.add('right')
                if r + h > inp.shape[0] and c + w > inp.shape[1]:
                    features.add('corner')
                if split == 'train' and r + h <= inp.shape[0] and c + w <= inp.shape[1] and (r, c, code) not in painted:
                    taught[code] += 1
    assert all(n >= 2 for n in taught.values()), 'two unobscured full demonstrations per code'
    assert test_features <= train_features, 'untaught test membership or boundary case'


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def _source_regressions():
    # Literal source masks are regression fixtures only, never generation choices.
    book = {2: (4, np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]], bool)),
            3: (1, np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]], bool)),
            5: (6, np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]], bool)),
            8: (7, np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]], bool))}
    grids = [np.array([[int(c) for c in row] for row in text.split('/')], dtype=np.int8)
             for text in _SOURCE_INPUT_TEXTS]
    for inp in grids:
        live, rejected, _, painted = _members(inp, set(book), 0)
        if inp.shape == (20, 20) and inp[1, 1] == 6:
            assert (5, 5, 5) in rejected and len(live) == 10
        elif inp.shape == (10, 10) and inp[0, 0] == 4:
            assert (4, 4, 2) in rejected and len(live) == 3
        elif inp.shape == (10, 10):
            assert (5, 5, 3) in painted and (9, 9, 3) in live
            assert _predict(inp, book)[9, 9] == 0
        elif inp[1, 0] == 8:
            assert (9, 8, 8) in rejected and len(live) == 17
            out = _predict(inp, book)
            for c in (0, 4, 8, 12, 16):
                assert np.array_equal(out[17:20, c:c + 4] == 7, book[8][1][:3])
        else:
            assert len(live) == 25 and not rejected


def self_test() -> None:
    _source_regressions()
    puzzles, metrics, books = [], [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        book = _learn(p['train'])
        books.append(tuple((c, ink, _array_key(mask)) for c, (ink, mask) in sorted(book.items())))
        for split in ('train', 'test'):
            for pair in p[split]:
                # All predictions use a freshly reconstructed training dictionary.
                assert np.array_equal(_predict(pair['input'], book), pair['output'])
                metrics.append((seed, split, _pair_metrics(pair, book)))
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len(set(books)) >= 40
    assert {len(book) for book in books} == {1, 2, 3, 4}
    assert len({m['shape'] for _, _, m in metrics}) >= 60
    assert len({m['mask_shape'] for _, _, m in metrics}) >= 7
    assert len({m['background'] for _, _, m in metrics}) == 10
    assert len({m['live'] for _, _, m in metrics}) >= 15
    for key in ('bottom', 'right', 'corner', 'complete', 'empty_interior', 'unequal_gaps',
                'free_geometry', 'bare', 'attached', 'painted', 'interior_attached',
                'clean', 'combined', 'shortcut_fails'):
        assert sum(bool(m[key]) for _, _, m in metrics) >= 4, key
    assert any(m['rows'] >= 5 and m['columns'] >= 5 for _, split, m in metrics if split == 'test')
    for split in ('train', 'test'):
        selected = [m for _, s, m in metrics if s == split]
        assert any(m['classes'] == 1 for m in selected)
        assert any(m['classes'] > 1 for m in selected)
        assert any(min(m['shape']) == 10 for m in selected)
        assert any(max(m['shape']) == 30 for m in selected)
        assert any(m['shape'] == (10, 10) for m in selected)
    for p in puzzles:
        book = _learn(p['train'])
        local = [_pair_metrics(pair, book) for s in p.values() for pair in s]
        assert len({m['shape'] for m in local}) >= 2
        assert len({m['background'] for m in local}) >= 2
    assert _puzzles_equal(puzzles[9], generate_puzzle(9))
    assert not _puzzles_equal(puzzles[9], puzzles[10])
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(314159, ntrain, ntest)
        assert ntrain is None or len(p['train']) == ntrain
        assert ntest is None or len(p['test']) == ntest
        assert _puzzles_equal(p, generate_puzzle(314159, ntrain, ntest))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, ntrain, ntest)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # The old validator accepted deletion of an entire live test stamp. It must
    # now fail, even though all inputs and all training labels remain intact.
    p = generate_puzzle(123, 2, 1)
    book = _learn(p['train'])
    query = p['test'][0]
    bg = _background(query['input'])
    r, c, code = _members(query['input'], set(book), bg)[0][0]
    h, w = book[code][1].shape
    query['output'][r:r+h, c:c+w] = bg
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('a deleted test stamp was accepted')
