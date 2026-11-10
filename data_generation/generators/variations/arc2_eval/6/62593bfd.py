"""Rotational companionship, with curated evidence and independent scene variation."""
from hashlib import sha256

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR8 = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dy, dx) != (0, 0)
)


def _normalise_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    assert len(rows) > 0
    return mask[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1,
    ].copy()


def _mask_key(mask):
    mask = _normalise_mask(mask)
    return (
        int(mask.shape[0]),
        int(mask.shape[1]),
        tuple(int(value) for value in mask.ravel()),
    )


def _rotations(mask):
    result = []
    seen = set()
    current = _normalise_mask(mask)

    for _ in range(4):
        current = _normalise_mask(current)
        key = _mask_key(current)
        if key not in seen:
            seen.add(key)
            result.append(current.copy())
        current = np.rot90(current)

    return result


def _canonical(mask):
    return min(_mask_key(rotated) for rotated in _rotations(mask))


def _mirror_canonical(mask):
    return _canonical(np.fliplr(_normalise_mask(mask)))


def _is_four_connected(mask):
    mask = _normalise_mask(mask)
    start = tuple(np.argwhere(mask)[0])
    stack = [start]
    seen = {start}

    while stack:
        y, x = stack.pop()
        for dy, dx in _DIR4:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < mask.shape[0]
                and 0 <= nx < mask.shape[1]
                and mask[ny, nx]
                and (ny, nx) not in seen
            ):
                seen.add((ny, nx))
                stack.append((ny, nx))

    return len(seen) == int(mask.sum())


def _find_background(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)

    assert len(values) >= 2
    assert counts[order[-1]] > counts[order[-2]]

    return int(values[order[-1]])


def _segment(grid, background=None):
    if background is None:
        background = _find_background(grid)

    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    objects = []

    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])

            if color == background or visited[y, x]:
                continue

            visited[y, x] = True
            stack = [(y, x)]
            cells = []

            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))

                for dy, dx in _DIR8:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            top = min(cell[0] for cell in cells)
            bottom = max(cell[0] for cell in cells)
            left = min(cell[1] for cell in cells)
            right = max(cell[1] for cell in cells)

            mask = np.zeros(
                (bottom - top + 1, right - left + 1),
                dtype=bool,
            )
            for cy, cx in cells:
                mask[cy - top, cx - left] = True

            objects.append(
                {
                    "color": color,
                    "top": top,
                    "left": left,
                    "mask": mask,
                    "canonical": _canonical(mask),
                }
            )

    return objects


def _paint(grid, obj, top, background):
    ys, xs = np.nonzero(obj["mask"])

    for dy, dx in zip(ys, xs):
        y = top + int(dy)
        x = obj["left"] + int(dx)

        assert int(grid[y, x]) == background
        grid[y, x] = obj["color"]


def _solve_grid(input_grid):
    background = _find_background(input_grid)
    objects = _segment(input_grid, background)

    frequencies = {}
    for obj in objects:
        key = obj["canonical"]
        frequencies[key] = frequencies.get(key, 0) + 1

    output = np.full(input_grid.shape, background, dtype=np.int8)

    for obj in objects:
        obj_height = obj["mask"].shape[0]
        if frequencies[obj["canonical"]] > 1:
            destination = 0
        else:
            destination = output.shape[0] - obj_height

        _paint(output, obj, destination, background)

    return output


def _cells_mask(cells):
    ymin = min(y for y, x in cells)
    xmin = min(x for y, x in cells)
    mask = np.zeros((max(y for y, x in cells) - ymin + 1,
                     max(x for y, x in cells) - xmin + 1), dtype=bool)
    for y, x in cells:
        mask[y-ymin, x-xmin] = True
    return mask


def _random_mask(rng, limit):
    """A compositional vocabulary, with no scale-invariant equivalence."""
    kind = int(rng.choice(7, p=[.10, .12, .13, .12, .18, .20, .15]))
    h, w = (int(rng.integers(2, limit+1)) for _ in range(2))
    if kind == 0:
        return np.ones((1, int(rng.integers(1, limit+1))), dtype=bool)
    if kind == 1:
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    elif kind == 2:
        mask = np.ones((h, w), dtype=bool)
        if h > 2 and w > 2 and rng.random() < .80:
            mask[1:-1, 1:-1] = False
    elif kind == 3:
        # Unequal branches can be asymmetric even with equal total area.
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
    elif kind in (4, 5):
        target = int(rng.integers(3, min(25, limit*limit)+1))
        cells = {(0, 0)}
        directions = _DIR4 if kind == 4 else _DIR8
        for _ in range(target-1):
            candidates = set()
            for y, x in sorted(cells):
                for dy, dx in directions:
                    p = (y+dy, x+dx)
                    if p in cells:
                        continue
                    ys = [a for a, b in cells] + [p[0]]
                    xs = [b for a, b in cells] + [p[1]]
                    if max(ys)-min(ys) < limit and max(xs)-min(xs) < limit:
                        candidates.add(p)
            if not candidates:
                break
            candidates = sorted(candidates)
            cells.add(candidates[int(rng.integers(len(candidates)))])
        mask = _cells_mask(cells)
    else:
        # A four-connected core with an exclusively diagonal attachment.
        h = int(rng.integers(2, limit+1))
        w = int(rng.integers(2, limit+1))
        mask = np.zeros((h, w), dtype=bool)
        mask[1:, 1] = True
        mask[-1, 1:] = True
        if h > 3 and w > 3 and rng.random() < .5:
            mask[1:-1, 1:-1] = True
        mask[0, 0] = True
    return _normalise_mask(np.rot90(mask, int(rng.integers(4))))


def _sample_shape(rng, forbidden, limit, chiral=False, diagonal=False,
                  asymmetric=False, ordinary=False):
    def permitted(mask):
        key = _canonical(mask)
        return (key not in forbidden
                and (not ordinary or _mirror_canonical(mask) not in forbidden)
                and (not chiral or key != _mirror_canonical(mask))
                and (not diagonal or not _is_four_connected(mask))
                and (not asymmetric or len(_rotations(mask)) == 4))

    for _ in range(160):
        mask = _random_mask(rng, limit)
        if permitted(mask):
            return mask
    # Finite exhaustive connected-mask fallback, independent of the random seed.
    for bits in range(1, 512):
        mask = _normalise_mask(np.array([(bits >> i) & 1 for i in range(9)],
                                       dtype=bool).reshape(3, 3))
        cells = set(map(tuple, np.argwhere(mask)))
        seen = {min(cells)}
        stack = list(seen)
        while stack:
            y, x = stack.pop()
            for dy, dx in _DIR8:
                point = y+dy, x+dx
                if point in cells and point not in seen:
                    seen.add(point)
                    stack.append(point)
        if seen == cells and permitted(mask):
            return mask
    raise AssertionError('finite shape vocabulary exhausted')


def _profile(rng, needs):
    repeat_count = int(rng.choice(4, p=[.07, .52, .33, .08]))
    sizes = [int(rng.choice([2, 3, 4], p=[.68, .25, .07]))
             for _ in range(repeat_count)]
    singles = int(rng.choice(5, p=[.09, .37, .29, .18, .07]))
    if not sizes:
        singles = max(3, singles)
    mode = str(rng.choice(['ordinary', 'negative', 'both'], p=[.72, .18, .10]))
    if 'mirror' in needs:
        mode = 'negative'
    if 'mixed' in needs:
        sizes = sizes or [2]
        singles = max(1, singles)
    if 'large' in needs:
        sizes = sizes or [needs['large']]
        if max(sizes) < needs['large']:
            sizes[-1] = needs['large']
    if 'pair' in needs and 2 not in sizes:
        sizes.insert(0, 2)
    if 'triplet' in needs and 3 not in sizes:
        sizes.append(3)
    if mode == 'negative':
        singles = max(2, singles)
    elif mode == 'both':
        while len(sizes) < 2:
            sizes.append(2)
    if 'ordinary' in needs:
        singles = max(singles, 3 if mode == 'negative' else 1)
    if sum(sizes) + singles < 3:
        singles = 1
    # Limit scene complexity without collapsing class profiles to templates.
    while sum(sizes) + singles > 10:
        removable = [i for i, size in enumerate(sizes)
                     if not (size == 2 and 'pair' in needs and sizes.count(2) == 1)
                     and not (size == 3 and 'triplet' in needs and sizes.count(3) == 1)
                     and not (size >= needs.get('large', 5)
                              and sum(s >= needs.get('large', 5) for s in sizes) == 1)]
        if len(sizes) > (2 if mode == 'both' else 1) and removable:
            sizes.pop(removable[-1])
        else:
            singles -= 1
    return sizes, singles, mode


def _build_objects(rng, sizes, singles, mode, needs, limit, background):
    forbidden, objects = set(), []
    bases = []
    diagonal_pending = 'diagonal' in needs
    for i, count in enumerate(sizes):
        if mode == 'both' and i == 1:
            base = np.fliplr(bases[0]).copy()
        else:
            base = _sample_shape(rng, forbidden, limit,
                                 chiral=mode == 'both' and i == 0,
                                 diagonal=diagonal_pending,
                                 asymmetric='rotation' in needs and i == 0)
        diagonal_pending = False
        bases.append(base)
        forbidden.add(_canonical(base))
        if mode == 'negative' or 'ordinary' in needs:
            forbidden.add(_mirror_canonical(base))
        turns = [int(rng.integers(4)) for _ in range(count)]
        if 'rotation' in needs and i == 0:
            turns[1] = (turns[0]+int(rng.choice([1, 3]))) % 4
        for turn in turns:
            objects.append({'mask': _normalise_mask(np.rot90(base, turn)),
                            'class_id': i, 'up': True})
    mirror = None
    for i in range(singles):
        if mode == 'negative' and i == 1:
            base = np.fliplr(mirror).copy()
        else:
            base = _sample_shape(rng, forbidden, limit,
                                 chiral=mode == 'negative' and i == 0,
                                 diagonal=diagonal_pending,
                                 ordinary=True)
        diagonal_pending = False
        if mode == 'negative' and i == 0:
            mirror = base
        forbidden.add(_canonical(base))
        forbidden.add(_mirror_canonical(base))
        objects.append({'mask': _normalise_mask(np.rot90(base, int(rng.integers(4)))),
                        'class_id': len(sizes)+i, 'up': False})
    palette = [int(c) for c in rng.permutation(10) if c != background]
    n = len(objects)
    if rng.random() < .35 and n <= 9:
        colors = palette[:n]
    else:
        palette_size = int(rng.integers(1, min(n, 9)+1))
        colors = [int(rng.choice(palette[:palette_size])) for _ in objects]
    if 'color' in needs:
        # Explicit evidence; every other color assignment remains unrestricted.
        group = [i for i, o in enumerate(objects) if o['class_id'] == 0]
        singleton = next(i for i, o in enumerate(objects) if not o['up'])
        colors[group[0]], colors[group[1]], colors[singleton] = palette[0], palette[1], palette[0]
    for obj, color in zip(objects, colors):
        obj['color'] = color
    return objects


def _halo(cells):
    return {(y+dy, x+dx) for y, x in cells
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def _edge_positions(objects, up, rng, compact):
    indices = [i for i, obj in enumerate(objects) if obj['up'] == up]
    occupied, positions = set(), {}
    left = -1
    for i in rng.permutation(indices):
        i = int(i)
        obj = objects[i]
        local = [(int(y)-(0 if up else obj['mask'].shape[0]), int(x))
                 for y, x in np.argwhere(obj['mask'])]
        blocked = _halo(occupied)
        x = max(0, left+1)
        while any((y, dx+x) in blocked for y, dx in local):
            x += 1
        # Different gaps for different neighbors, including close mask packing.
        x += int(rng.choice([0, 1, 2, 3], p=[.75, .18, .06, .01] if compact
                            else [.40, .35, .20, .05]))
        while any((y, dx+x) in blocked for y, dx in local):
            x += 1
        positions[i] = x
        occupied.update((y, dx+x) for y, dx in local)
        left = x
    return positions, max((x for y, x in occupied), default=-1)+1


def _choose_dimensions(rng, needed_width, min_height, aspect):
    if rng.random() < .09 and aspect in (None, 'square'):
        return 30, 30
    aspect = aspect or str(rng.choice(['portrait', 'landscape', 'square']))
    options = [(h, w) for h in range(max(7, min_height), 31)
               for w in range(max(6, needed_width), 31)
               if (h == w if aspect == 'square' else h > w if aspect == 'portrait' else w > h)]
    if not options:
        options = [(h, w) for h in range(max(7, min_height), 31)
                   for w in range(max(6, needed_width), 31)]
    areas = np.array([h*w for h, w in options], dtype=float)
    temperature = float(rng.choice([55, 180, 700], p=[.35, .40, .25]))
    weights = np.exp(-(areas-areas.min())/temperature)
    return options[int(rng.choice(len(options), p=weights/weights.sum()))]


def _assign_rows(objects, height, width, rng):
    # Bit sets make cell-accurate clearance cheap even on compact canvases.
    candidates = []
    for obj in objects:
        h = obj['mask'].shape[0]
        destination = 0 if obj['up'] else height-h
        opposite = height-h if obj['up'] else 0
        rows = [int(r) for r in rng.permutation(height-h+1)]
        mode = int(rng.choice(4, p=[.13, .13, .22, .52]))
        preferred = (destination if mode == 0 else opposite if mode == 1 else
                     min(height-h, max(0, destination+int(rng.choice([-2, -1, 1, 2]))))
                     if mode == 2 else rows[0])
        rows.remove(preferred)
        rows.insert(0, preferred)
        local = [(int(y), obj['left']+int(x)) for y, x in np.argwhere(obj['mask'])]
        variants = []
        for top in rows:
            cells = {(y+top, x) for y, x in local}
            occupied = sum(1 << (y*width+x) for y, x in cells)
            halo = sum(1 << (y*width+x) for y, x in _halo(cells)
                       if 0 <= y < height and 0 <= x < width)
            variants.append((top, occupied, halo))
        candidates.append(variants)
    order = sorted(range(len(objects)), key=lambda i: -int(objects[i]['mask'].sum()))
    chosen, budget = [0]*len(objects), [2500]

    def place(step, occupied):
        if step == len(order):
            return True
        i = order[step]
        for top, cells, halo in candidates[i]:
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if occupied & halo:
                continue
            chosen[i] = top
            if place(step+1, occupied | cells):
                return True
        return False

    if not place(0, 0) or all(chosen[i] == (0 if o['up'] else height-o['mask'].shape[0])
                              for i, o in enumerate(objects)):
        # Guaranteed fit: shift the two already separated edge groups inward.
        chosen = [1 if o['up'] else height-o['mask'].shape[0]-1 for o in objects]
    return chosen


def _generate_pair(rng, regime, parameters, needs=None):
    assert regime in ('source_train', 'target_test')
    needs = {} if needs is None else needs
    background = parameters['background']
    if background is None:
        background = int(rng.integers(10))
    sizes, singles, mode = _profile(rng, needs)
    for attempt in range(100):
        limit = int(rng.choice([3, 4, 5, 6, 7], p=[.15, .24, .29, .24, .08])) if attempt < 35 else 3
        objects = _build_objects(rng, sizes, singles, mode, needs, limit, background)
        compact = rng.random() < .5 or attempt >= 35
        upper, upper_width = _edge_positions(objects, True, rng, compact)
        lower, lower_width = _edge_positions(objects, False, rng, compact)
        needed = max(upper_width, lower_width)
        if needed > 30:
            continue
        upper_h = max((o['mask'].shape[0] for o in objects if o['up']), default=0)
        lower_h = max((o['mask'].shape[0] for o in objects if not o['up']), default=0)
        height, width = _choose_dimensions(rng, needed, upper_h+lower_h+3, parameters['aspect'])
        if sum(int(o['mask'].sum()) for o in objects)*2 >= height*width:
            continue
        for positions, need in ((upper, upper_width), (lower, lower_width)):
            offset = int(rng.integers(width-need+1))
            for i, x in positions.items():
                objects[i]['left'] = x+offset
        rows = _assign_rows(objects, height, width, rng)
        moves = [row != (0 if o['up'] else height-o['mask'].shape[0])
                 for o, row in zip(objects, rows)]
        if (('mixed' in needs and sum(moves) < 2) or
                ('diagonal' in needs and not any(moving and not _is_four_connected(o['mask'])
                                                 for o, moving in zip(objects, moves)))):
            rows = [1 if o['up'] else height-o['mask'].shape[0]-1 for o in objects]
        inp = np.full((height, width), background, dtype=np.int8)
        out = inp.copy()
        for obj, top in zip(objects, rows):
            _paint(inp, obj, top, background)
            _paint(out, obj, 0 if obj['up'] else height-obj['mask'].shape[0], background)
        assert len(_segment(inp)) == len(objects)
        assert np.array_equal(_solve_grid(inp), out)
        return {'input': inp, 'output': out}
    return _fallback_pair(background, needs)


def _fallback_pair(background, needs):
    """Guaranteed separated construction if a sampled packing budget runs out."""
    elbow = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    diagonal = np.eye(2, dtype=bool)
    mirror = np.array([[1, 1], [0, 1], [0, 1], [0, 1]], dtype=bool)
    count = max(3, needs.get('large', 3))
    masks = [elbow, np.rot90(elbow)] + [np.rot90(diagonal, i % 2) for i in range(count)]
    masks += [mirror, np.fliplr(mirror), np.ones((1, 1), dtype=bool)]
    palette = [c for c in range(10) if c != background]
    inp = np.full((30, 30), background, dtype=np.int8)
    out = inp.copy()
    x = 0
    for i, mask in enumerate(masks):
        up = i < count+2
        if i == count+2:
            x = 0
        obj = {'mask': mask, 'left': x, 'color': palette[i % 2]}
        dest = 0 if up else 30-mask.shape[0]
        _paint(inp, obj, 1 if up else dest-2, background)
        _paint(out, obj, dest, background)
        x += mask.shape[1]+1
    return {'input': inp, 'output': out}


def _validate_count(value, name, low, high):
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} is outside its legal range")


def _resolve_counts(seed, num_train, num_test):
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)

    token = int(seed)

    if num_train is None and num_test is None:
        num_test = 1 + token % 4
        low_train = max(2, num_test + 1)
        num_train = (
            low_train
            + (token // 4) % (11 - low_train)
        )
    elif num_train is None:
        low_train = max(2, int(num_test) + 1)
        num_train = low_train + token % (11 - low_train)
    elif num_test is None:
        high_test = min(4, int(num_train) - 1)

        if high_test < 1:
            raise ValueError("no legal test count exists")

        num_test = 1 + token % high_test

    num_train = int(num_train)
    num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _grid_fingerprint(grid):
    return (tuple(grid.shape), grid.astype(np.uint8).tobytes())


def _source_digest(grid):
    return sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest()


# Exact source inputs only: their dimensions and colors are still legal.
_SOURCE_INPUTS = {
    '45670833f7527062ca3b03b254ac254c48222c3840016ecdeb88e57f341f8891',
    'e5f93f925307fbee22c33a8968b59be16b0a5a8b141ca191065697d65dea2236',
    '455c4f573473f71bdf11ad70ead37ea7a38740ce2b75f3e2e09cb7eacf7014b1',
    '11ab409fd817e8b96db1a906f137f828b5b8924c4f98fc3070f88983be9c30bb',
}


def _groups(objects):
    groups = {}
    for obj in objects:
        groups.setdefault(obj['canonical'], []).append(obj)
    return groups


def _has_hole(mask):
    padded = np.pad(mask, 1)
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        y, x = stack.pop()
        for dy, dx in _DIR4:
            p = y+dy, x+dx
            if (0 <= p[0] < padded.shape[0] and 0 <= p[1] < padded.shape[1]
                    and not padded[p] and p not in seen):
                seen.add(p)
                stack.append(p)
    return len(seen) < int((~padded).sum())


def _features(pair):
    """Measurements reconstructed from visible grids, not construction metadata."""
    grid = pair['input']
    h, w = grid.shape
    bg = _find_background(grid)
    objects = _segment(grid, bg)
    groups = _groups(objects)
    sizes = [len(group) for group in groups.values()]
    mirrors = {key: _mirror_canonical(group[0]['mask']) for key, group in groups.items()}
    repeated = [o for o in objects if len(groups[o['canonical']]) > 1]
    singletons = [o for o in objects if len(groups[o['canonical']]) == 1]
    destinations = [0 if len(groups[o['canonical']]) > 1 else h-o['mask'].shape[0]
                    for o in objects]
    moves = [abs(o['top']-dest) for o, dest in zip(objects, destinations)]
    diagonal = sum(not _is_four_connected(o['mask']) for o in objects)
    feature = {
        'height': h, 'width': w, 'aspect': 'square' if h == w else 'portrait' if h > w else 'landscape',
        'background': bg, 'object_count': len(objects),
        'repeated_sizes': sorted(s for s in sizes if s > 1),
        'singletons': len(singletons), 'repeated_classes': sum(s > 1 for s in sizes),
        'max_multiplicity': max(sizes),
        'ordinary_singleton': any(mirrors[o['canonical']] == o['canonical'] or
                                   mirrors[o['canonical']] not in groups for o in singletons),
        'mirror_negative': any(mirrors[o['canonical']] != o['canonical'] and
                              len(groups.get(mirrors[o['canonical']], [])) == 1 for o in singletons),
        'both_handed_repeated': any(mirrors[o['canonical']] != o['canonical'] and
                                   len(groups.get(mirrors[o['canonical']], [])) > 1 for o in repeated),
        'mixed_destinations': bool(repeated and singletons),
        'all_singleton': not repeated, 'all_repeated': not singletons,
        'diagonal_objects': diagonal, 'four_connected_objects': len(objects)-diagonal,
        'moving_diagonal': any(d > 0 and not _is_four_connected(o['mask']) for o, d in zip(objects, moves)),
        'mixed_connectivity': 0 < diagonal < len(objects),
        'areas': [int(o['mask'].sum()) for o in objects],
        'extents': [list(o['mask'].shape) for o in objects],
        'hole': any(_has_hole(o['mask']) for o in objects),
        'line': any(min(o['mask'].shape) == 1 for o in objects),
        'symmetric_repeated': any(len(_rotations(o['mask'])) < 4 for o in repeated),
        'square_repeated': any(o['mask'].shape[0] == o['mask'].shape[1] and
                               bool(o['mask'].all()) for o in repeated),
        'palette_size': len({o['color'] for o in objects}),
        'all_distinct_colors': len({o['color'] for o in objects}) == len(objects),
        'stationary': sum(d == 0 for d in moves),
        'opposite_edge': sum(o['top'] == h-o['mask'].shape[0]-dest for o, dest in zip(objects, destinations)),
        'moves': moves, 'short_and_long': any(0 < d <= 2 for d in moves) and any(d >= 6 for d in moves),
        'density': sum(int(o['mask'].sum()) for o in objects)/(h*w),
        'identity_copies': False, 'half_turn_only': False, 'quarter_turn': False,
        'same_color_copies': False, 'different_color_copies': False,
        'cross_destination_color': False, 'unrelated_color_reuse': False,
        'same_area_noncongruent': False, 'same_area_cross_destination': False,
        'shared_columns_opposite': False, 'output_bbox_overlap': False,
    }
    for i, a in enumerate(objects):
        for j in range(i+1, len(objects)):
            b = objects[j]
            same = a['canonical'] == b['canonical']
            different_dest = (len(groups[a['canonical']]) > 1) != (len(groups[b['canonical']]) > 1)
            same_color = a['color'] == b['color']
            if same:
                ka, kb = _mask_key(a['mask']), _mask_key(b['mask'])
                quarter = kb in (_mask_key(np.rot90(a['mask'])), _mask_key(np.rot90(a['mask'], 3)))
                half = kb == _mask_key(np.rot90(a['mask'], 2))
                feature['identity_copies'] |= ka == kb
                feature['half_turn_only'] |= half and not quarter and ka != kb
                feature['quarter_turn'] |= quarter and ka != kb
                feature['same_color_copies'] |= same_color
                feature['different_color_copies'] |= not same_color
            feature['cross_destination_color'] |= different_dest and same_color
            feature['unrelated_color_reuse'] |= not same and same_color
            same_area = int(a['mask'].sum()) == int(b['mask'].sum())
            feature['same_area_noncongruent'] |= same_area and not same
            feature['same_area_cross_destination'] |= same_area and different_dest
            x_overlap = max(a['left'], b['left']) < min(a['left']+a['mask'].shape[1], b['left']+b['mask'].shape[1])
            feature['shared_columns_opposite'] |= x_overlap and different_dest
            feature['output_bbox_overlap'] |= x_overlap and max(destinations[i], destinations[j]) < min(
                destinations[i]+a['mask'].shape[0], destinations[j]+b['mask'].shape[0])
    return feature


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    # These are visible stylistic preferences, not hidden semantic conventions.
    parameters = {
        'background': int(rng.integers(10)) if rng.random() < .22 else None,
        'aspect': str(rng.choice(['portrait', 'landscape', 'square'])) if rng.random() < .22 else None,
    }
    puzzle, seen = {'train': [], 'test': []}, set()

    def append(split, needs):
        for _ in range(30):
            pair = _generate_pair(rng, 'source_train' if split == 'train' else 'target_test', parameters, needs)
            key = _grid_fingerprint(pair['input'])
            if key not in seen and _source_digest(pair['input']) not in _SOURCE_INPUTS:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise AssertionError('distinct-pair construction exhausted')

    for _ in range(test_count):
        append('test', {})
    test_features = [_features(p) for p in puzzle['test']]
    needs = [{} for _ in range(train_count)]
    a, b = [int(i) for i in rng.choice(train_count, 2, replace=False)]
    needs[a].update(mixed=True, mirror=True, pair=True, rotation=True)
    needs[b].update(mixed=True, ordinary=True)
    needs[int(rng.choice([a, b]))]['color'] = True
    largest = max(f['max_multiplicity'] for f in test_features)
    if largest >= 3:
        needs[b]['large'] = largest
        if largest == 3:
            needs[b]['triplet'] = True
    if largest == 4:
        needs[a]['triplet'] = True
    if any(f['diagonal_objects'] for f in test_features):
        needs[int(rng.choice([a, b]))]['diagonal'] = True
    for requirement in needs:
        append('train', requirement)
    validate_generated_puzzle(puzzle)
    return puzzle


def _object_signature(obj, top):
    return (obj['color'], top, obj['left'], _mask_key(obj['mask']))


def _validate_pair(pair):
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape
    bg = _find_background(inp)
    assert _find_background(out) == bg
    assert int((inp == bg).sum()) > inp.size//2
    objects = _segment(inp, bg)
    assert 3 <= len(objects) <= 10
    groups = _groups(objects)
    expected = []
    for obj in objects:
        top = 0 if len(groups[obj['canonical']]) > 1 else inp.shape[0]-obj['mask'].shape[0]
        expected.append(_object_signature(obj, top))
    actual = [_object_signature(o, o['top']) for o in _segment(out, bg)]
    assert sorted(expected) == sorted(actual), 'object changed, merged, or moved incorrectly'
    assert np.array_equal(_solve_grid(inp), out)
    assert not np.array_equal(inp, out)
    for grid in (inp, out):
        occupied = set()
        for obj in _segment(grid, bg):
            cells = {(obj['top']+int(y), obj['left']+int(x)) for y, x in np.argwhere(obj['mask'])}
            assert not cells & _halo(occupied), 'separate objects touch'
            occupied.update(cells)
    assert _source_digest(inp) not in _SOURCE_INPUTS
    return _features(pair)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    features = {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = _grid_fingerprint(pair['input'])
            pkey = ikey, _grid_fingerprint(pair['output'])
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            features[split].append(_validate_pair(pair))
    train, test = features['train'], features['test']
    assert sum(f['mixed_destinations'] for f in train) >= 2
    # These observations contradict the review's alternative classification rules.
    assert any(f['ordinary_singleton'] for f in train), 'mirror-only alternative not excluded'
    assert any(f['mirror_negative'] for f in train), 'reflection-inclusive alternative not excluded'
    assert any(2 in f['repeated_sizes'] for f in train)
    assert any(f['quarter_turn'] for f in train), 'rotation convention not demonstrated'
    assert any(f['different_color_copies'] for f in train)
    assert any(f['cross_destination_color'] for f in train)
    if max(f['max_multiplicity'] for f in test) >= 3:
        assert any(3 in f['repeated_sizes'] for f in train), 'triplet threshold not demonstrated'
        assert max(f['max_multiplicity'] for f in train) >= max(f['max_multiplicity'] for f in test)
    if any(f['diagonal_objects'] for f in test):
        assert any(f['moving_diagonal'] for f in train)


def _puzzle_fingerprint(puzzle):
    return tuple((_grid_fingerprint(p['input']), _grid_fingerprint(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test() -> None:
    for count in (3, 4):
        fallback = _fallback_pair(0, {'large': count})
        features = _validate_pair(fallback)
        assert features['mirror_negative'] and features['ordinary_singleton']
        assert features['quarter_turn'] and features['moving_diagonal']
        assert features['cross_destination_color'] and count in features['repeated_sizes']
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_fingerprint(puzzles[0]) == _puzzle_fingerprint(generate_puzzle(0))
    assert len({_puzzle_fingerprint(p) for p in puzzles}) == 42
    for n, (tr, te) in enumerate([(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]):
        p = generate_puzzle(2000+n, tr, te)
        assert tr is None or len(p['train']) == tr
        assert te is None or len(p['test']) == te
        assert _puzzle_fingerprint(p) == _puzzle_fingerprint(generate_puzzle(2000+n, tr, te))
    for tr, te in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)]:
        try:
            generate_puzzle(1, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    by_puzzle = [{s: [_features(pair) for pair in p[s]] for s in p} for p in puzzles]
    all_features = [f for p in by_puzzle for s in p for f in p[s]]
    for split in ('train', 'test'):
        fs = [f for p in by_puzzle for f in p[split]]
        assert {f['aspect'] for f in fs} == {'portrait', 'landscape', 'square'}
        for name in ['ordinary_singleton', 'mirror_negative', 'both_handed_repeated',
                     'all_singleton', 'all_repeated', 'identity_copies', 'half_turn_only',
                     'quarter_turn', 'symmetric_repeated', 'square_repeated', 'hole', 'line',
                     'same_color_copies', 'different_color_copies', 'cross_destination_color',
                     'all_distinct_colors', 'unrelated_color_reuse', 'same_area_noncongruent',
                     'same_area_cross_destination', 'mixed_connectivity', 'stationary',
                     'opposite_edge', 'shared_columns_opposite', 'output_bbox_overlap', 'short_and_long']:
            assert any(f[name] for f in fs), (split, name)
    assert len({(f['height'], f['width']) for f in all_features}) >= 70
    assert any(f['height'] == f['width'] == 30 for f in all_features)
    assert any(max(f['height'], f['width']) <= 15 for f in all_features)
    assert any(f['density'] >= .2 for f in all_features)
    assert {f['background'] for f in all_features} == set(range(10))
    assert {1, 2, 3, 4, 5, 9, 11, 12} <= {a for f in all_features for a in f['areas']}
    assert max(a for f in all_features for a in f['areas']) >= 20
    assert max(max(e) for f in all_features for e in f['extents']) >= 6
    assert {3, 4, 5, 6, 7, 8} <= {f['object_count'] for f in all_features}
    assert {2, 3, 4} <= {n for f in all_features for n in f['repeated_sizes']}
    assert any(2 in f['repeated_sizes'] and 3 in f['repeated_sizes'] for f in all_features)
    for name in ['object_count', 'singletons', 'repeated_classes', 'max_multiplicity',
                 'aspect', 'background', 'palette_size', 'diagonal_objects']:
        assert sum(len({f[name] for s in p for f in p[s]}) > 1 for p in by_puzzle) >= 12, name
    assert any(len({f['background'] for s in p for f in p[s]}) == 1 for p in by_puzzle)
    assert any(len({f['aspect'] for s in p for f in p[s]}) == 1 for p in by_puzzle)
    assert any(f['both_handed_repeated'] and f['cross_destination_color'] and f['diagonal_objects']
               for f in all_features)
    # Deliberate output corruption must fail the reconstruction validator.
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
