"""Generate two-motif ARC puzzles by copying the visible structural template."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    '705f9cbbbb9ee42e51ad2af909501ceadc4385d320508fde3f3d27a5a27ab732',
    'c2f36cbb9ca891f303b554b8021fe35ea3f36855121a355ab5841efddd462a09',
    '669583fbb987ced20e2dea801caf7bfcf5d7176a8214206ea26c5e9cedf64d48',
    '0dfa439d98c170a3320906affa18befa410dd60156879c72fabb41cc88542dba',
    '15e6b6131d7979b8670545f08cbf4341ab7a4ff74336bd998935e085190bcdca',
}
_FORMS = ('full', 'open', 'dotted', 'diamond', 'irregular')


def _fingerprint(inp, out):
    data = bytes(inp.shape) + inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()
    return hashlib.sha256(data).hexdigest()


def _counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError('Illegal pair count')
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('Training count must exceed test count')
    return int(num_train), int(num_test)


def _frame(rng, form, horizontal, compact):
    if form == 'diamond':
        radius = int(rng.integers(3, 7))
        h = w = 2 * radius + 1
        structure = {(r, c) for r in range(h) for c in range(w)
                     if abs(r - radius) + abs(c - radius) == radius}
        interior = {(r, c) for r in range(h) for c in range(w)
                    if abs(r - radius) + abs(c - radius) <= radius - 2}
        return h, w, structure, interior

    if form == 'irregular':
        h = int(rng.integers(9, 14))
        w = int(rng.integers(9, 14))
        top, left, bottom, right = 2, 2, h - 3, w - 3
    else:
        if compact and horizontal:
            h = int(rng.integers(4, 6))
            w = int(rng.integers(8, 12))
        elif compact:
            h = int(rng.integers(7, 10))
            w = int(rng.integers(5, 7))
        else:
            h = int(rng.integers(4, 14))
            w = int(rng.integers(5, 14))
        top, left, bottom, right = 0, 0, h - 1, w - 1
    sides = [
        [(top, c) for c in range(left, right + 1)],
        [(bottom, c) for c in range(left, right + 1)],
        [(r, left) for r in range(top, bottom + 1)],
        [(r, right) for r in range(top, bottom + 1)],
    ]
    if form == 'full':
        structure = set().union(*sides)
    elif form == 'open':
        missing = int(rng.integers(0, 4))
        structure = set().union(*(side for i, side in enumerate(sides) if i != missing))
    elif form == 'dotted':
        step = int(rng.integers(2, 5))
        structure = set()
        for side in sides:
            phase = int(rng.integers(0, step))
            structure.update(side[phase::step])
            structure.add(side[0])
            structure.add(side[-1])
        if rng.random() < .55:
            # A short absent run keeps the actual input template decisive.
            side = sides[int(rng.integers(0, 4))]
            start = int(rng.integers(1, max(2, len(side) - 1)))
            structure.difference_update(side[start:start + int(rng.integers(1, 4))])
    else:
        structure = set().union(*sides)
        side = sides[int(rng.integers(0, 4))]
        start = int(rng.integers(1, max(2, len(side) - 2)))
        structure.difference_update(side[start:start + int(rng.integers(1, 4))])
        direction = int(rng.integers(0, 4))
        if direction == 0:
            c = int(rng.integers(left + 1, right))
            structure.update(((top - 1, c), (0, c), (0, c + 1)))
            extras = ((h - 1, 0), (h - 1, w - 1))
        elif direction == 1:
            c = int(rng.integers(left + 1, right))
            structure.update(((bottom + 1, c), (h - 1, c), (h - 1, c - 1)))
            extras = ((0, 0), (0, w - 1))
        elif direction == 2:
            r = int(rng.integers(top + 1, bottom))
            structure.update(((r, left - 1), (r, 0), (r + 1, 0)))
            extras = ((0, w - 1), (h - 1, w - 1))
        else:
            r = int(rng.integers(top + 1, bottom))
            structure.update(((r, right + 1), (r, w - 1), (r - 1, w - 1)))
            extras = ((0, 0), (h - 1, 0))
        structure.add(extras[int(rng.integers(0, 2))])
    interior = {(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)}
    return h, w, structure, interior


def _motif(rng, interior, structure):
    available = interior - structure
    if len(available) < 2:
        return None
    rows = [r for r, _ in available]
    cols = [c for _, c in available]
    top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
    for _ in range(20):
        mh = int(rng.integers(1, min(6, bottom - top + 1) + 1))
        mw = int(rng.integers(1, min(8, right - left + 1) + 1))
        y = int(rng.integers(top, bottom - mh + 2))
        x = int(rng.integers(left, right - mw + 2))
        region = {(r, c) for r in range(y, y + mh) for c in range(x, x + mw)
                  if (r, c) in available}
        if len(region) < 2:
            continue
        style = int(rng.integers(0, 6))
        if style == 0:
            motif = region
        elif style == 1:
            motif = {(r, c) for r, c in region if r == y + mh // 2 or c == x + mw // 2}
        elif style == 2:
            motif = {(r, c) for r, c in region if r in (y, y + mh - 1) or c == x}
        elif style == 3:
            motif = {(r, c) for r, c in region if (r + c) % 2 == 0}
        elif style == 4:
            motif = {(r, c) for r, c in region if rng.random() < .55}
        else:
            motif = {(r, c) for r, c in region if r == y or c == x + mw - 1}
            if region:
                motif.add(sorted(region)[int(rng.integers(0, len(region)))])
        if len(motif) >= 2:
            return motif
    return None


def _positions(rng, h, w, horizontal, compact):
    axis = w if horizontal else h
    gap = int(rng.choice((0, 1, 1, 2, 3))) if compact else int(rng.choice((0, 1, 1, 2, 3, 4, 5, 7)))
    if 2 * axis + gap > 30:
        gap = max(0, 30 - 2 * axis)
    room = 30 - 2 * axis - gap
    pad_limit = 1 if compact else 4
    before = int(rng.integers(0, min(pad_limit, room) + 1))
    after = int(rng.integers(0, min(pad_limit, room - before) + 1))
    stagger_limit = 1 if compact else min(5, (30 - (h if horizontal else w)) // 2)
    stagger = 0 if compact and rng.random() < .7 else int(rng.integers(-stagger_limit, stagger_limit + 1))
    perpendicular = h if horizontal else w
    extra = abs(stagger)
    free = 30 - perpendicular - extra
    cross_before = int(rng.integers(0, min(pad_limit, free) + 1))
    cross_after = int(rng.integers(0, min(pad_limit, free - cross_before) + 1))
    if horizontal:
        a = (cross_before + max(0, -stagger), before)
        b = (cross_before + max(0, stagger), before + w + gap)
        shape = (perpendicular + extra + cross_before + cross_after,
                 2 * w + gap + before + after)
    else:
        a = (before, cross_before + max(0, -stagger))
        b = (before + h + gap, cross_before + max(0, stagger))
        shape = (2 * h + gap + before + after,
                 perpendicular + extra + cross_before + cross_after)
    return shape, a, b


def _candidate_solutions(inp):
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 3 or len(set(counts)) == 1:
        return []
    background = int(colors[np.argmax(counts)])
    if list(counts).count(max(counts)) != 1:
        return []
    foreground = [int(v) for v in colors if v != background]
    candidates = {}
    height, width = inp.shape
    for motif_color in foreground:
        structure_color = next(v for v in foreground if v != motif_color)
        motif = {(int(r), int(c)) for r, c in np.argwhere(inp == motif_color)}
        structure = {(int(r), int(c)) for r, c in np.argwhere(inp == structure_color)}
        if not motif or len(structure) < 4:
            continue
        for axis in (0, 1):
            for cut in range(1, inp.shape[axis]):
                first = {p for p in motif if p[axis] < cut}
                second = motif - first
                if len(first) < 2 or len(first) != len(second):
                    continue
                if any(p[axis] == cut for p in motif):
                    continue
                side = {p[axis] < cut for p in structure}
                if len(side) != 1:
                    continue
                source = first if True in side else second
                target_motif = second if True in side else first
                dr = min(r for r, _ in target_motif) - min(r for r, _ in source)
                dc = min(c for _, c in target_motif) - min(c for _, c in source)
                if {(r + dr, c + dc) for r, c in source} != target_motif:
                    continue
                mr = [r for r, _ in source]
                mc = [c for _, c in source]
                sr = [r for r, _ in structure]
                sc = [c for _, c in structure]
                if not (min(sr) <= min(mr) <= max(mr) <= max(sr)
                        and min(sc) <= min(mc) <= max(mc) <= max(sc)):
                    continue
                copied = {(r + dr, c + dc) for r, c in structure}
                if any(r < 0 or r >= height or c < 0 or c >= width
                       or inp[r, c] != background for r, c in copied):
                    continue
                span_h = max(sr) - min(sr) + 1
                span_w = max(sc) - min(sc) + 1
                out = inp.copy()
                for r, c in copied:
                    out[r, c] = structure_color
                key = out.tobytes()
                candidates[key] = (out, {'background': background,
                                         'motif_color': motif_color,
                                         'structure_color': structure_color,
                                         'motif': source,
                                         'structure': structure,
                                         'shift': (dr, dc),
                                         'orientation': 'horizontal' if abs(dc) * span_h >= abs(dr) * span_w else 'vertical'})
    return list(candidates.values())


def _make_pair(rng, form, horizontal, palette):
    background, motif_color, structure_color = palette
    for _ in range(100):
        compact = form in ('full', 'open', 'dotted') and rng.random() < .38
        h, w, structure, interior = _frame(rng, form, horizontal, compact)
        motif = _motif(rng, interior, structure)
        if motif is None:
            continue
        shape, a, b = _positions(rng, h, w, horizontal, compact)
        if rng.random() < .5:
            a, b = b, a
        inp = np.full(shape, background, dtype=np.int8)
        for base in (a, b):
            for r, c in motif:
                inp[base[0] + r, base[1] + c] = motif_color
        for r, c in structure:
            inp[a[0] + r, a[1] + c] = structure_color
        out = inp.copy()
        for r, c in structure:
            out[b[0] + r, b[1] + c] = structure_color
        candidates = _candidate_solutions(inp)
        if len(candidates) == 1 and np.array_equal(out, candidates[0][0]) \
                and _fingerprint(inp, out) not in _SOURCE_DIGESTS:
            pair = {'input': inp, 'output': out}
            measured = _shape_features(pair)
            if measured[0] == form and measured[1] == ('horizontal' if horizontal else 'vertical'):
                return pair
    raise RuntimeError('Could not construct an unambiguous pair')


def _random_palette(rng):
    background = 0 if rng.random() < .55 else int(rng.integers(1, 10))
    others = [v for v in range(1, 10) if v != background]
    chosen = rng.choice(others, size=2, replace=False)
    return background, int(chosen[0]), int(chosen[1])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    # The copying rule is the puzzle-wide convention; colors and geometry have
    # roles visible in each pair and therefore need no hidden shared mapping.
    first_palette = _random_palette(rng)
    second_background = int(rng.choice([v for v in range(10)
                                        if v not in first_palette[1:]]))
    second_palette = (second_background, first_palette[2], first_palette[1])
    train_forms = [str(rng.choice(_FORMS[:4]))]
    train_forms.append(str(rng.choice([f for f in _FORMS if f != train_forms[0]])))
    train_forms += [str(rng.choice(_FORMS)) for _ in range(ntrain - 2)]
    first_horizontal = bool(rng.integers(0, 2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            if split == 'train':
                form = train_forms[i]
                horizontal = first_horizontal if i == 0 else (not first_horizontal if i == 1 else bool(rng.integers(0, 2)))
                palette = first_palette if i == 0 else second_palette if i == 1 else _random_palette(rng)
            else:
                form = str(rng.choice(train_forms)) if rng.random() < .45 else str(rng.choice(_FORMS))
                horizontal = bool(rng.integers(0, 2))
                palette = _random_palette(rng)
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, form, horizontal, palette)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct distinct pairs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert _fingerprint(inp, out) not in _SOURCE_DIGESTS
    candidates = _candidate_solutions(inp)
    assert len(candidates) == 1
    expected, feature = candidates[0]
    assert np.array_equal(expected, out)
    changed = inp != out
    assert np.all(inp[changed] == feature['background'])
    assert np.all(out[changed] == feature['structure_color'])
    assert int(np.count_nonzero(changed)) == len(feature['structure'])
    return feature


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            features[split].append(_inspect_pair(pair))
    # With two training examples, both orientations and two surround forms
    # are exposed. Test geometry and colors are always readable in its input.
    assert {f['orientation'] for f in features['train']} == {'horizontal', 'vertical'}
    assert len({_shape_features(p)[0] for p in puzzle['train']}) >= 2
    first, second = features['train'][:2]
    assert first['motif_color'] == second['structure_color']
    assert first['structure_color'] == second['motif_color']


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def _shape_features(pair):
    feature = _inspect_pair(pair)
    st = feature['structure']
    mr = feature['motif']
    r0, r1 = min(r for r, _ in st), max(r for r, _ in st)
    c0, c1 = min(c for _, c in st), max(c for _, c in st)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    normalized = {(r - r0, c - c0) for r, c in st}
    perimeter = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    if normalized == perimeter:
        form = 'full'
    elif any(normalized == perimeter - side for side in (
            {(0, c) for c in range(1, w - 1)},
            {(h - 1, c) for c in range(1, w - 1)},
            {(r, 0) for r in range(1, h - 1)},
            {(r, w - 1) for r in range(1, h - 1)})):
        form = 'open'
    elif h == w and h % 2 == 1 and normalized == {
            (r, c) for r in range(h) for c in range(w)
            if abs(r - h // 2) + abs(c - w // 2) == h // 2}:
        form = 'diamond'
    elif normalized <= perimeter:
        form = 'dotted'
    else:
        form = 'irregular'
    shift = feature['shift']
    return (form, feature['orientation'], feature['background'],
            feature['motif_color'], feature['structure_color'],
            len(mr), h, w, shift, pair['input'].shape,
            min(r0, c0, pair['input'].shape[0] - 1 - r1,
                pair['input'].shape[1] - 1 - c1) == 0)


def _component_count(points):
    remaining = set(points)
    count = 0
    while remaining:
        frontier = [remaining.pop()]
        count += 1
        while frontier:
            r, c = frontier.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (r + dr, c + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        frontier.append(neighbor)
    return count


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, forms = set(), set(), set()
    orientations = {'train': set(), 'test': set()}
    split_forms, form_orientations, open_sides = set(), set(), set()
    signs, gaps, motif_shapes, motif_offsets = set(), set(), set(), set()
    sign_counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
    dotted_gap_lengths = set()
    dotted_shapes, irregular_shapes = set(), set()
    backgrounds = set()
    edge_count = source_edge = small_motifs = staggered = disconnected = 0
    detached_irregular = zero_gap = one_gap = 0
    shallow = narrow = combined = 0
    mixed_forms = mixed_orientation = reused_test_form = new_test_form = 0
    mixed_backgrounds = mixed_sizes = 0
    first_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        first = puzzle['train'][0]['input']
        first_inputs.add((first.shape, first.tobytes()))
        by_split = {s: [_shape_features(p) for p in puzzle[s]] for s in ('train', 'test')}
        train_forms = {f[0] for f in by_split['train']}
        mixed_forms += len(train_forms) > 1
        mixed_orientation += len({f[1] for f in by_split['train']}) > 1
        reused_test_form += any(f[0] in train_forms for f in by_split['test'])
        new_test_form += any(f[0] not in train_forms for f in by_split['test'])
        mixed_backgrounds += len({f[2] for split in by_split for f in by_split[split]}) > 1
        mixed_sizes += len({f[9] for split in by_split for f in by_split[split]}) > 1
        for split in ('train', 'test'):
            for pair, f in zip(puzzle[split], by_split[split]):
                form, orient, bg, motif, structure, size, h, w, shift, shape, edge = f
                detail = _inspect_pair(pair)
                st = detail['structure']
                mt = detail['motif']
                r0, r1 = min(r for r, _ in st), max(r for r, _ in st)
                c0, c1 = min(c for _, c in st), max(c for _, c in st)
                mr, mc = min(r for r, _ in mt), min(c for _, c in mt)
                motif_offsets.add((mr - r0, mc - c0))
                motif_shapes.add(tuple(sorted((r - mr, c - mc) for r, c in mt)))
                translated = {(r + shift[0], c + shift[1]) for r, c in st}
                if orient == 'horizontal':
                    other_lo, other_hi = min(c for _, c in translated), max(c for _, c in translated)
                    gap = max(c0, other_lo) - min(c1, other_hi) - 1
                    sign = 'right' if shift[1] > 0 else 'left'
                else:
                    other_lo, other_hi = min(r for r, _ in translated), max(r for r, _ in translated)
                    gap = max(r0, other_lo) - min(r1, other_hi) - 1
                    sign = 'down' if shift[0] > 0 else 'up'
                gaps.add(gap)
                signs.add(sign)
                sign_counts[sign] += 1
                zero_gap += gap == 0
                one_gap += gap == 1
                disconnected += _component_count(mt) > 1
                normalized = frozenset((r - r0, c - c0) for r, c in st)
                if form == 'dotted':
                    dotted_shapes.add(normalized)
                    sides = (
                        sorted(c for r, c in st if r == r0),
                        sorted(c for r, c in st if r == r1),
                        sorted(r for r, c in st if c == c0),
                        sorted(r for r, c in st if c == c1),
                    )
                    for side in sides:
                        dotted_gap_lengths.update(b - a - 1 for a, b in zip(side, side[1:]))
                if form == 'irregular':
                    irregular_shapes.add(normalized)
                    detached_irregular += _component_count(st) > 1
                if form == 'open':
                    perimeter = {(r, c) for r in range(r0, r1 + 1)
                                 for c in range(c0, c1 + 1)
                                 if r in (r0, r1) or c in (c0, c1)}
                    missing = perimeter - st
                    if all(r == r0 for r, _ in missing):
                        open_sides.add('top')
                    elif all(r == r1 for r, _ in missing):
                        open_sides.add('bottom')
                    elif all(c == c0 for _, c in missing):
                        open_sides.add('left')
                    elif all(c == c1 for _, c in missing):
                        open_sides.add('right')
                split_forms.add((split, form))
                form_orientations.add((form, orient))
                forms.add(form)
                orientations[split].add(orient)
                backgrounds.add(bg)
                shapes.add(shape)
                layouts.add((form, orient, size, h, w, shift))
                edge_count += edge
                source_edge += any(r in (0, shape[0] - 1) or c in (0, shape[1] - 1)
                                   for r, c in st)
                small_motifs += size <= 3
                staggered += shift[0] != 0 and shift[1] != 0
                shallow += shape[0] <= 6 and shape[1] >= 18
                narrow += shape[0] >= 16 and shape[1] <= 7
                combined += bg != 0 and form == 'open' and shift[0] != 0 and shift[1] != 0
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(first_inputs) >= 40 and len(shapes) >= 35 and len(layouts) >= 80
    assert forms == set(_FORMS)
    assert all(v == {'horizontal', 'vertical'} for v in orientations.values())
    assert split_forms == {(split, form) for split in ('train', 'test') for form in _FORMS}
    assert form_orientations == {(form, orient) for form in _FORMS
                                 for orient in ('horizontal', 'vertical')}
    assert open_sides == {'top', 'bottom', 'left', 'right'}
    assert signs == {'left', 'right', 'up', 'down'}
    assert min(sign_counts.values()) >= 20
    assert 0 in gaps and 1 in gaps and len(gaps) >= 6
    assert zero_gap >= 10 and one_gap >= 10
    assert len(backgrounds) >= 4
    assert edge_count >= 5 and small_motifs >= 5 and staggered >= 10
    assert source_edge >= 20 and detached_irregular >= 15
    assert {1, 2, 3, 4} <= dotted_gap_lengths
    assert shallow >= 10 and narrow >= 10 and combined >= 5
    assert (17, 7) in shapes and (6, 21) in shapes
    assert disconnected >= 5 and len(motif_shapes) >= 80 and len(motif_offsets) >= 8
    assert len(dotted_shapes) >= 15 and len(irregular_shapes) >= 15
    assert mixed_forms >= 35 and mixed_orientation >= 35
    assert reused_test_form >= 10 and new_test_form >= 10
    assert mixed_backgrounds >= 30 and mixed_sizes >= 40
    for seed, ntrain, ntest in ((50, 2, 1), (51, 10, 1), (52, 10, 4),
                                (53, None, 1), (54, None, 4),
                                (55, 2, None), (56, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle['train']) == ntrain
        assert ntest is None or len(puzzle['test']) == ntest
        assert _same_puzzle(puzzle, generate_puzzle(seed, ntrain, ntest))
    assert not _same_puzzle(generate_puzzle(9), generate_puzzle(10))
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(3, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
