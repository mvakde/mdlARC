import numpy as np
from functools import lru_cache


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate(cells):
    return _normalize((c, -r) for r, c in cells)


def _reflect(cells):
    return _normalize((r, -c) for r, c in cells)


@lru_cache(maxsize=8192)
def _rotation_variants(cells):
    variants = set()
    current = _normalize(cells)
    for _ in range(4):
        variants.add(current)
        current = _rotate(current)
    return tuple(sorted(variants))


@lru_cache(maxsize=8192)
def _variants(cells):
    variants = set(_rotation_variants(cells))
    variants.update(_rotation_variants(_reflect(cells)))
    return tuple(sorted(variants))


def _canonical(cells):
    return min(_variants(_normalize(cells)))


def _shape_size(cells):
    return (
        max(r for r, _ in cells) + 1,
        max(c for _, c in cells) + 1,
    )


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if (
            not isinstance(num_train, (int, np.integer))
            or isinstance(num_train, (bool, np.bool_))
        ):
            raise ValueError("num_train must be an integer or None")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if (
            not isinstance(num_test, (int, np.integer))
            or isinstance(num_test, (bool, np.bool_))
        ):
            raise ValueError("num_test must be an integer or None")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        legal = [
            (train_count, test_count)
            for test_count in range(1, 5)
            for train_count in range(2, 11)
            if train_count > test_count
        ]
        return legal[int(rng.integers(len(legal)))]

    if num_train is None:
        choices = np.arange(max(2, num_test + 1), 11)
        chosen = int(choices[int(rng.integers(len(choices)))])
        return chosen, num_test

    choices = np.arange(1, min(4, num_train - 1) + 1)
    chosen = int(choices[int(rng.integers(len(choices)))])
    return num_train, chosen


def _binary_components(mask):
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            component = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))

                for dy, dx in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    ny = y + dy
                    nx = x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            components.append(component)

    return components


def _colored_components(grid, background):
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue

            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            component = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))

                for dy, dx in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    ny = y + dy
                    nx = x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            components.append((color, component))

    return components



# Exact solved source grids: regression fixtures, never sampled templates.
_SOURCE_ROWS = ((('111111111111111111111111111111',
   '111881111111111444444444441111',
   '111881111111111411441441141111',
   '111111112211111411441441141111',
   '111111112211111444444444441111',
   '111111111111111444441444441111',
   '112211111111111411411141141111',
   '112211111111111444441444441111',
   '111111116111111444444444441111',
   '111111166611111411441441141111',
   '111111116111111411441441141111',
   '111111111111111444444444441111',
   '111111111111111111111111111111',
   '111111111111111111111111111211',
   '111151111188111111111111111211',
   '111555111188111112211188111111',
   '111151111111111112211188111111',
   '111111111111111111111111111111',
   '111111111111111111511111111111',
   '111111111111111115551111111111',
   '111111111111111111511111111111',
   '111111111111111111111111111111',
   '111111111111331111111118811111',
   '111111111111331111111118811111',
   '118811111111111111111111111111',
   '118811111111111111191111111111',
   '111111111111111111191111111111',
   '111111111111111111111111111111',
   '111111111991111111111111111111',
   '111111111111111111111111111111'),
  ('44444444444',
   '48844944884',
   '48844944884',
   '44444444444',
   '44444544444',
   '49945554994',
   '44444544444',
   '44444444444',
   '48844944884',
   '48844944884',
   '44444444444')),
 (('888888888888888888888888888888',
   '881111111111111888888588288888',
   '881881118111881888888888228888',
   '881811111111181888888888888888',
   '881118881888111888448888888228',
   '881118111118111888488888888828',
   '881118118118111888888858888888',
   '881811188811181888888888888888',
   '881118118118111888888888888888',
   '881118111118111888588848888888',
   '881118881888111888888844888788',
   '881811111111181888888888887778',
   '881881118111881888388888888788',
   '881111111111111888388888888888',
   '888888888888888888333885888888',
   '888888888888888888888888888888',
   '888828888888588898888888885888',
   '888228888888888999888883338888',
   '888888858844888898888883888888',
   '888888888848888888888883888888',
   '885888888888888888888588888788',
   '888888888888888866688888887778',
   '888888488858888868888888888788',
   '858888448888888868888887888888',
   '888888888888888888888877788888',
   '888898888887888888888887888888',
   '888999888877788888988888888858',
   '888898888887888889998888888888',
   '888888888888885888988888588888',
   '888888888888888888888888888888'),
  ('1111111111111',
   '1441115111441',
   '1411111111141',
   '1113331333111',
   '1113111113111',
   '1113117113111',
   '1511177711151',
   '1113117113111',
   '1113111113111',
   '1113331333111',
   '1411111111141',
   '1441115111441',
   '1111111111111')),
 (('444444444444444444444444444444',
   '411444111414111111414111444114',
   '414414414414144441414414414414',
   '444111444414441144414444111444',
   '414414414414144441414414414414',
   '411444111414111111414111444114',
   '444444444444444444444444444444',
   '111111111111118111111111111111',
   '111711119111188811111111122111',
   '117711199911118111111116111111',
   '111111111111111111881166611111',
   '111111111111111111181116111111',
   '111111111811111111111111113111',
   '111111118881111111111111113311',
   '112111111811111161111911111111',
   '112111111111111666111991111111',
   '111111999111111161111911111111',
   '111111191111111111111111111111',
   '116111111111771199111111181111',
   '166611131111171191111111888111',
   '116111333111111191111111181111',
   '111111131111111191111111111113',
   '111111111111211191118111111133',
   '111111111112221199188811111113',
   '111111121111111111118111111111',
   '117111121111111121111111119111',
   '117711111111111122111111199911',
   '111111116666611121111311119111',
   '111111111111111111113331111111',
   '111111111111111111111111111111'),
  ('444444444444444444444444444444',
   '477444999464999999464999444774',
   '474484494464944449464494484474',
   '444888444464442244464444888444',
   '474484494464944449464494484474',
   '477444999464999999464999444774',
   '444444444444444444444444444444')))


@lru_cache(maxsize=1)
def _source_pairs():
    return tuple({role: np.array([[int(c) for c in row] for row in rows],
                                dtype=np.int8)
                  for role, rows in zip(('input', 'output'), pair)}
                 for pair in _SOURCE_ROWS)


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _decode(grid, equivalence='free'):
    """Recover all visible data without output grids or sampling metadata."""
    rectangles = []
    for color in np.unique(grid):
        for cells in _binary_components(grid == color):
            if len(cells) < 8:
                continue
            rs, cs = zip(*cells)
            y, x, bottom, right = min(rs), min(cs), max(rs), max(cs)
            box = grid[y:bottom+1, x:right+1]
            if min(box.shape) < 3:
                continue
            if not (np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                continue
            other = set(map(int, np.unique(box))) - {int(color)}
            if len(other) == 1 and np.count_nonzero(grid == color) == len(cells):
                rectangles.append((int(color), other.pop(), y, x, bottom, right))
    assert len(rectangles) == 1, 'the rectangular carrier must be unique'
    carrier, bg, y, x, bottom, right = rectangles[0]
    box = grid[y:bottom+1, x:right+1]
    cavities = _binary_components(box == bg)
    exterior = grid.copy()
    exterior[y:bottom+1, x:right+1] = bg
    objects = _colored_components(exterior, bg)
    # Count whole monochromatic components, never pixels.
    def signature(cells):
        raw = _normalize(cells)
        if equivalence == 'area':
            return len(raw)
        if equivalence == 'bbox':
            return tuple(sorted(_shape_size(raw)))
        if equivalence == 'translation':
            return raw
        if equivalence == 'rotation':
            return min(_rotation_variants(raw))
        return _canonical(raw)
    votes, raws, object_cells = {}, {}, []
    for color, cells in objects:
        assert color not in (bg, carrier)
        raw, key = _normalize(cells), signature(cells)
        votes.setdefault(key, {})[color] = votes.setdefault(key, {}).get(color, 0) + 1
        raws.setdefault(key, []).append((color, raw))
        object_cells.append(cells)
    keys = {signature(c) for c in cavities}
    if equivalence == 'free':
        assert set(votes) == keys, 'unsupported cavities or irrelevant objects'
    else:
        # Give shortcut solvers the benefit of ignoring unmatched objects;
        # evidence must change a cavity answer, not just leave unused votes.
        assert keys <= set(votes), 'unsupported cavity under alternative rule'
    winners = {}
    for key in sorted(keys):
        counts = votes[key]
        high = max(counts.values())
        candidates = [color for color, count in counts.items() if count == high]
        assert len(candidates) == 1, 'tied vote'
        winners[key] = candidates[0]
    output = box.copy()
    cavity_raws, cavity_cells = {}, {}
    for cells in cavities:
        key = signature(cells)
        cavity_raws.setdefault(key, []).append(_normalize(cells))
        cavity_cells.setdefault(key, []).append(tuple(cells))
        for r, c in cells:
            output[r, c] = winners[key]
    return output, dict(background=bg, carrier=carrier, bbox=(y, x, bottom, right),
                        votes=votes, winners=winners, cavity_raws=cavity_raws,
                        object_raws=raws, cavity_cells=cavity_cells,
                        object_cells=object_cells)


def _fails_alternative(pair, equivalence):
    try:
        prediction, _ = _decode(pair['input'], equivalence)
    except AssertionError:
        return True
    return not np.array_equal(prediction, pair['output'])


def _hole_free(shape):
    h, w = _shape_size(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_binary_components(mask)) == 1


def _shape_pool(rng):
    """Parameterized families plus freshly grown, hole-free polyominoes."""
    shapes = {((0, 0),)}
    for length in range(2, 8):
        shapes.add(_canonical(tuple((0, c) for c in range(length))))
    for height in range(2, 6):
        for width in range(2, 7):
            # Independently vary both arms, including the source five-cell L.
            shapes.add(_canonical(tuple({(r, 0) for r in range(height)} |
                                        {(height-1, c) for c in range(width)})))
            if height <= 4 and width >= 3:
                # Returns at either end of a bar: the source 8-cell bracket
                # is height=2, width=6. Deeper returns produce U forms.
                shapes.add(_canonical(tuple({(r, 0) for r in range(height)} |
                                            {(r, width-1) for r in range(height)} |
                                            {(height-1, c) for c in range(width)})))
    for h in range(2, 5):
        for w in range(2, 5):
            if h*w <= 9:
                shapes.add(_canonical(tuple((r, c) for r in range(h) for c in range(w))))
            if h >= 3 and w >= 3:
                shapes.add(_canonical(tuple((r, c) for r in range(h) for c in range(w)
                                            if r >= h-2 or c < 2)))
    shapes.add(_canonical(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))))
    for _ in range(55):
        cells = {(0, 0)}
        target = int(rng.integers(4, 11))
        for _ in range(target-1):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
            point = frontier[int(rng.integers(len(frontier)))]
            cells.add(point)
        shape = _canonical(cells)
        if max(_shape_size(shape)) <= 6 and _hole_free(shape):
            shapes.add(shape)
    return tuple(sorted(shapes))


def _choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _ballot(rng, size, force_plurality=False, easy=False):
    if force_plurality:
        return (3, 2, 2) if rng.random() < .7 else (4, 3, 1, 1)
    mode = int(rng.integers(7))
    if easy:
        mode = int(rng.integers(3))
    if mode == 0:
        return (1,)
    if mode == 1:
        return (int(rng.integers(2, 15 if size == 1 else 6)),)
    if mode == 2:
        return (int(rng.integers(2, 4)), 1)
    if mode == 3:
        n = int(rng.integers(3, 7))
        return (n, n-1)
    if mode == 4:
        return _choice(rng, ((3, 2, 2), (4, 3, 1, 1), (3, 2, 1), (2, 1, 1, 1)))
    colors = int(rng.integers(3, 6))
    high = int(rng.integers(2, 6))
    return tuple([high] + sorted([int(rng.integers(1, high))
                                  for _ in range(colors-1)], reverse=True))


def _block(available, cells, distance=1):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-distance):min(h, r+distance+1),
                  max(0, c-distance):min(w, c+distance+1)] = False


def _positions(available, shape):
    h, w = _shape_size(shape)
    rows, cols = available.shape[0]-h+1, available.shape[1]-w+1
    if rows <= 0 or cols <= 0:
        return np.empty((0, 2), dtype=int)
    valid = np.ones((rows, cols), dtype=bool)
    for r, c in shape:
        valid &= available[r:r+rows, c:c+cols]
    return np.argwhere(valid)


def _pack_one(available, variants, rng, symmetric=False):
    """Uniform legal coordinates; no common slot pitch or fixed lattice."""
    height, width = available.shape
    for index in rng.permutation(len(variants)):
        shape = variants[int(index)]
        positions = _positions(available, shape)
        if not len(positions):
            continue
        for pos in rng.permutation(len(positions))[:80 if symmetric else 1]:
            y, x = map(int, positions[int(pos)])
            cells = tuple((y+r, x+c) for r, c in shape)
            groups = [cells]
            if symmetric:
                reflected = tuple((height-1-r, width-1-c) for r, c in cells)
                if any(not available[r, c] for r, c in reflected):
                    continue
                near = {(r+dr, c+dc) for r, c in cells
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if any(cell in near for cell in reflected):
                    continue
                groups.append(reflected)
            for group in groups:
                _block(available, group)
            return groups
    return None


def _geometry(rng, category, count, footprint, fallback=False):
    # Size follows total content, with independent slack and aspect ratio.
    if fallback:
        return 14, 26
    area = max(35, int(footprint * rng.uniform(1.45, 2.1)))
    area = min(350, area)
    if category == 'square':
        side = int(np.clip(np.sqrt(area) + rng.integers(0, 3), 7, 19))
        return side, side
    if category in ('wide', 'tall', 'strip'):
        width = int(rng.integers(max(14, min(27, int(np.sqrt(area)*1.5))), 31))
        height = int(np.clip(np.ceil(area / (width-2)) + 2, 5, 13))
        return (width, height) if category == 'tall' else (height, width)
    h = int(rng.integers(7, 20))
    w = int(np.clip(np.ceil(area / (h-2)) + 2, 7, 24))
    return h, w


def _make_pair(rng, shapes, reflections, category, anchor=None, evidence=None,
               forbidden_color=None, previous_roles=None, simple=False):
    """Bounded layout search. Mandatory shape classes and ballots are retained."""
    shapes = tuple(shapes)
    palette = list(map(int, rng.permutation(10)))
    if previous_roles is not None:
        # Explicitly demonstrate that both roles are local to the input.
        bg_choices = [c for c in palette if c != previous_roles[0]]
        bg = _choice(rng, bg_choices)
        carrier = _choice(rng, [c for c in palette if c not in (bg, previous_roles[1])])
        palette = [bg, carrier] + [c for c in palette if c not in (bg, carrier)]
    bg, carrier = palette[:2]
    colors = palette[2:]
    group_count = _choice(rng, list(range(1, min(len(shapes), 8)+1)))
    if rng.random() < .4:
        group_count = min(len(shapes), 8)
    winning_palette = list(map(int, rng.choice(colors, group_count, replace=False)))
    winners = {s: winning_palette[i % group_count] for i, s in enumerate(shapes)}
    if anchor is not None and forbidden_color is not None and winners[anchor] == forbidden_color:
        winners[anchor] = _choice(rng, [c for c in colors if c != forbidden_color])
    if evidence == 'reflection' or evidence == 'rotation':
        other = shapes[1]
        winners[other] = _choice(rng, [c for c in colors if c != winners[anchor]])
    ballots = {s: _ballot(rng, len(s), force_plurality=(s == anchor and evidence is not None),
                           easy=simple or len(shapes) >= 7) for s in shapes}
    # Limit combined physical load without eliminating rare rich ballots.
    # At least one independent rich ballot remains; larger shapes often need
    # fewer votes, as in the source's single bar and bracket.
    def cost(s, counts):
        h, w = _shape_size(s)
        return (h+1)*(w+1)*sum(counts)
    budget = 380
    for s in sorted(shapes, key=lambda s: cost(s, ballots[s]), reverse=True):
        if sum(cost(t, ballots[t]) for t in shapes) <= budget:
            break
        if s != anchor or evidence is None:
            ballots[s] = (1,) if rng.random() < .6 else (2, 1)
    specs = []
    for s in shapes:
        losers = [c for c in colors if c != winners[s]]
        rng.shuffle(losers)
        for color, n in zip([winners[s]] + losers, ballots[s]):
            allowed = _variants(s) if reflections else _rotation_variants(s)
            if s == anchor and evidence is not None:
                if evidence == 'reflection':
                    allowed = (_reflect(s),) if color == winners[s] else (s,)
                else:
                    allowed = (_rotate(s),) if color == winners[s] else (s,)
            specs.extend((s, color, allowed) for _ in range(n))
    symmetric = rng.random() < .28 and len(shapes) <= 6
    multiplicities = {s: (2 if symmetric else 1) for s in shapes}
    extra_limit = min(6, max(2, 19 - sum(multiplicities.values()))) if simple else max(2, 19 - sum(multiplicities.values()))
    extra = int(rng.integers(0, extra_limit))
    for _ in range(extra // (2 if symmetric else 1)):
        s = _choice(rng, shapes)
        multiplicities[s] += 2 if symmetric else 1
    footprint = sum(((_shape_size(s)[0]+1)*(_shape_size(s)[1]+1)) * n
                    for s, n in multiplicities.items())
    for attempt in range(72):
        sh, sw = _geometry(rng, category, len(shapes), footprint,
                           fallback=attempt >= 60)
        available = np.ones((sh, sw), dtype=bool)
        available[[0, -1], :] = False
        available[:, [0, -1]] = False
        placements = []
        # Place all distinct classes before optional repeated cavities.
        order = sorted(shapes, key=lambda s: (max(_shape_size(s)), len(s)), reverse=True)
        failed = False
        for s in order:
            variants = _variants(s) if reflections else _rotation_variants(s)
            if s == anchor and evidence is not None:
                variants = (s,)
            groups = _pack_one(available, variants, rng, symmetric)
            if groups is None:
                failed = True
                break
            placements.extend((s, group) for group in groups)
        if failed:
            continue
        extras = [s for s in shapes for _ in range(multiplicities[s] // (2 if symmetric else 1) - 1)]
        rng.shuffle(extras)
        for s in extras:
            groups = _pack_one(available, _variants(s) if reflections else _rotation_variants(s),
                               rng, symmetric)
            if groups is not None:
                placements.extend((s, group) for group in groups)
        # More generous canvases for a large exterior ballot; small content
        # also occurs on small canvases. Thirty-by-thirty is fully legal.
        exterior_cost = sum(cost(s, ballots[s]) for s in shapes)
        side = int(np.clip(np.sqrt(sh*sw + exterior_cost*1.65) + rng.integers(1, 5), 12, 30))
        height = min(30, max(sh + int(rng.integers(3, 8)), side + int(rng.integers(-2, 3))))
        width = min(30, max(sw + int(rng.integers(0, 6)), side + int(rng.integers(-2, 3))))
        if attempt >= 20 or rng.random() < .14:
            height = width = 30
        if category == 'strip':
            width = sw
            y, x = 0, 0
        else:
            y = int(rng.integers(height-sh+1))
            x = int(rng.integers(width-sw+1))
            edge = int(rng.integers(7))
            if edge == 0:
                y = 0
            elif edge == 1:
                y = height-sh
            elif edge == 2:
                x = 0
            elif edge == 3:
                x = width-sw
        available = np.ones((height, width), dtype=bool)
        available[max(0, y-1):min(height, y+sh+1), max(0, x-1):min(width, x+sw+1)] = False
        external = []
        # Randomize equal-sized objects before packing large components first.
        shuffled = [specs[int(i)] for i in rng.permutation(len(specs))]
        shuffled.sort(key=lambda v: (max(_shape_size(v[0])), len(v[0])), reverse=True)
        for s, color, variants in shuffled:
            group = _pack_one(available, variants, rng)
            if group is None:
                failed = True
                break
            external.append((color, group[0]))
        if failed:
            continue
        inp = np.full((height, width), bg, dtype=np.int8)
        box = np.full((sh, sw), carrier, dtype=np.int8)
        out = box.copy()
        for s, cells in placements:
            for r, c in cells:
                box[r, c], out[r, c] = bg, winners[s]
        if len(_binary_components(box == carrier)) != 1:
            continue
        inp[y:y+sh, x:x+sw] = box
        for color, cells in external:
            for r, c in cells:
                inp[r, c] = color
        return {'input': inp, 'output': out}
    return None


def _fallback_pair(rng, shapes, reflections, anchor, evidence, forbidden, roles):
    # Smaller repertoire and no optional geometry constraint. There are only
    # two <=6-cell mandatory teaching shapes, so this has ample packing room.
    result = _make_pair(rng, shapes[:2], reflections, 'rectangle', anchor,
                        evidence, forbidden, roles, simple=True)
    assert result is not None, 'failed bounded conservative construction'
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Shape-equivalence is the only convention needing training evidence.
    # Rotation-only puzzles use a consistent chirality; reflection puzzles
    # explicitly refute rotation-only matching in their demonstrations.
    reflections = rng.random() < .7
    pool = _shape_pool(rng)
    eligible = [s for s in pool if 4 <= len(s) <= 6 and len(_rotation_variants(s)) > 1
                and max(_shape_size(s)) <= 4
                and (not reflections or _reflect(s) not in _rotation_variants(s))]
    anchor = _choice(rng, eligible)
    companions = [s for s in pool if len(s) == len(anchor) and s != anchor
                  and sorted(_shape_size(s)) == sorted(_shape_size(anchor))]
    # The base families include S/L tetrominoes and several 5/6-cell pairs.
    # A procedural anchor with no same-box companion is replaced by one that
    # can visibly distinguish both area-only and bounding-box-only grouping.
    if not companions:
        eligible = [s for s in eligible if any(t != s and len(t) == len(s)
                    and sorted(_shape_size(t)) == sorted(_shape_size(s)) for t in pool)]
        anchor = _choice(rng, eligible)
        companions = [s for s in pool if len(s) == len(anchor) and s != anchor
                      and sorted(_shape_size(s)) == sorted(_shape_size(anchor))]
    companion = _choice(rng, companions)
    geometry_regime = _choice(rng, ('forward', 'reverse', 'comparable', 'mixed', 'mixed', 'mixed'))
    common_geometry = _choice(rng, ('square', 'wide', 'tall', 'rectangle'))
    vocabulary_regime = _choice(rng, ('familiar', 'overlap', 'overlap', 'novel'))
    puzzle, seen, train_shapes = {'train': [], 'test': []}, set(), set()
    first_roles, first_winner = None, None
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            teaching = split == 'train' and i < 2
            simple = not teaching and rng.random() < .24
            if teaching:
                k = int(rng.integers(2, 5))
                others = [s for s in pool if s not in (anchor, companion) and len(s) <= 7]
                shapes = [anchor, companion] + [others[int(j)] for j in rng.choice(len(others), k-2, replace=False)]
            else:
                k = int(rng.integers(1, 4)) if simple else int(rng.choice(
                    np.arange(1, 9), p=(.04, .09, .19, .21, .19, .14, .09, .05)))
                available = list(pool)
                if simple or k >= 7:
                    available = [s for s in available if len(s) <= 5]
                elif k >= 5:
                    available = [s for s in available if len(s) <= 8]
                if split == 'test' and vocabulary_regime == 'familiar':
                    available = [s for s in available if s in train_shapes]
                elif split == 'test' and vocabulary_regime == 'novel':
                    available = [s for s in available if s not in train_shapes]
                if not available:
                    available = [s for s in pool if (s in train_shapes) ==
                                 (vocabulary_regime == 'familiar')]
                k = min(k, len(available))
                shapes = [available[int(j)] for j in rng.choice(len(available), k, replace=False)]
                if split == 'test' and vocabulary_regime == 'overlap' and k >= 2:
                    known = [s for s in available if s in train_shapes]
                    unseen = [s for s in available if s not in train_shapes]
                    if known and unseen:
                        shapes[:2] = [_choice(rng, known), _choice(rng, unseen)]
                        shapes = list(dict.fromkeys(shapes))
            category = _choice(rng, ('square', 'wide', 'tall', 'rectangle', 'strip'))
            if geometry_regime == 'forward':
                category = 'square' if split == 'train' else 'strip'
            elif geometry_regime == 'reverse':
                category = 'strip' if split == 'train' else 'square'
            elif geometry_regime == 'comparable' and rng.random() < .75:
                category = common_geometry
            evidence = ('reflection' if reflections and i == 0 else 'rotation') if teaching else None
            forbidden = first_winner if teaching and i == 1 else None
            roles = first_roles if teaching and i == 1 else None
            for duplicate_attempt in range(8):
                pair = _make_pair(rng, shapes, reflections, category,
                                  anchor if teaching else None, evidence, forbidden, roles,
                                  simple=simple)
                if pair is None:
                    fallback_shapes = [anchor, companion] if teaching else shapes[:2]
                    pair = _fallback_pair(rng, fallback_shapes, reflections,
                                          anchor if teaching else None, evidence, forbidden, roles)
                if _grid_key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('duplicate bounded constructions')
            puzzle[split].append(pair)
            seen.add(_grid_key(pair['input']))
            _, data = _decode(pair['input'])
            if split == 'train':
                train_shapes.update(data['votes'])
                if i == 0:
                    first_roles = data['background'], data['carrier']
                    first_winner = data['winners'][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    sources = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}
    seen, seen_pairs, analyses = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair['input'])
            pair_key = key, _grid_key(pair['output'])
            assert key not in seen and pair_key not in seen_pairs
            assert pair_key not in sources
            seen.add(key)
            seen_pairs.add(pair_key)
            prediction, data = _decode(pair['input'])
            assert np.array_equal(pair['output'], prediction), 'wrong fill or crop'
            assert not np.array_equal(pair['input'], pair['output'])
            assert 1 <= len(data['votes']) <= 8
            # All components have exact geometric matches; carrier and border
            # checks, support and unique plurality are inside the decoder.
            # Check exterior separation independently of monochrome grouping.
            exterior = pair['input'].copy()
            y, x, bottom, right = data['bbox']
            exterior[y:bottom+1, x:right+1] = data['background']
            assert len(_binary_components(exterior != data['background'])) == len(data['object_cells'])
            assert len(_binary_components(pair['input'][y:bottom+1, x:right+1] == data['carrier'])) == 1
            analyses[split].append(data)
    train, test = puzzle['train'], puzzle['test']
    # Explicit witnesses reject area grouping and a memorized shape-color map.
    assert any(_fails_alternative(p, 'area') for p in train)
    assert any(_fails_alternative(p, 'translation') for p in train)
    if any(_fails_alternative(p, 'rotation') for p in test):
        assert any(_fails_alternative(p, 'rotation') for p in train)
    if any(_fails_alternative(p, 'bbox') for p in test):
        # Exact-shape evidence (same-area distinct shapes) is always required;
        # bounding-box discrimination is measured, and taught when necessary.
        assert any(_fails_alternative(p, 'bbox') for p in train)
    history = {}
    for data in analyses['train']:
        for shape, winner in data['winners'].items():
            history.setdefault(shape, set()).add(winner)
    assert any(len(colors) > 1 for colors in history.values())
    assert len({a['background'] for a in analyses['train']}) >= 2
    assert len({a['carrier'] for a in analyses['train']}) >= 2
    assert any(len(v) >= 3 and max(v.values()) * 2 <= sum(v.values())
               for a in analyses['train'] for v in a['votes'].values())
    # There is no shared literal palette, winner map, repertoire, or crop
    # geometry to recover. New test shapes carry complete local ballots.


def _features(pair):
    _, a = _decode(pair['input'])
    counts = [tuple(sorted(v.values(), reverse=True)) for v in a['votes'].values()]
    shapes = list(a['votes'])
    y, x, bottom, right = a['bbox']
    h, w = pair['input'].shape
    oh, ow = pair['output'].shape
    mask = pair['input'][y:bottom+1, x:right+1] == a['background']
    groups = [list(a['winners'].values()).count(c) for c in set(a['winners'].values())]
    reflection_witnesses = []
    for shape, cavity_raws in a['cavity_raws'].items():
        for raw in cavity_raws:
            rotations, rotation_counts = _rotation_variants(raw), {}
            for color, object_raw in a['object_raws'][shape]:
                if object_raw in rotations:
                    rotation_counts[color] = rotation_counts.get(color, 0) + 1
            high = max(rotation_counts.values(), default=0)
            best = [c for c, n in rotation_counts.items() if n == high]
            if best != [a['winners'][shape]]:
                reflection_witnesses.append(shape)
                break
    return dict(input_shape=(h, w), output_shape=(oh, ow),
                aspect=('square' if oh == ow else 'tall' if oh > ow else 'wide'),
                edges=(y == 0, x == 0, bottom == h-1, right == w-1),
                roles=(a['background'], a['carrier']), class_count=len(shapes),
                cavity_count=sum(map(len, a['cavity_raws'].values())),
                repetitions=tuple(sorted(map(len, a['cavity_raws'].values()))),
                density=float(mask.mean()), central_symmetry=bool(np.array_equal(mask, mask[::-1, ::-1])),
                max_area=max(map(len, shapes)), max_extent=max(max(_shape_size(s)) for s in shapes),
                votes=sum(map(sum, counts)), ballots=tuple(sorted(counts)),
                single_support=any(c == (1,) for c in counts),
                one_color=any(len(c) == 1 for c in counts),
                close=any(len(c) > 1 and c[0]-c[1] == 1 for c in counts),
                nonmajority=any(len(c) > 1 and c[0]*2 <= sum(c) for c in counts),
                repeated_losers=any(len(c) > 1 and c[1] > 1 for c in counts),
                shared_groups=sum(g > 1 for g in groups),
                distinct_winners=len(groups) == len(shapes),
                area_discrimination=_fails_alternative(pair, 'area'),
                bbox_discrimination=_fails_alternative(pair, 'bbox'),
                reflection_required=_fails_alternative(pair, 'rotation'),
                rotation_required=_fails_alternative(pair, 'translation'),
                reflection_witnesses=tuple(sorted(reflection_witnesses)),
                object_row_residues=len({min(r for r, c in cells) % 4 for cells in a['object_cells']}),
                object_col_residues=len({min(c for r, c in cells) % 4 for cells in a['object_cells']}),
                shapes=frozenset(shapes))


def self_test() -> None:
    # Exact source regression, including singleton ballots and the large
    # bracket, rather than validation against an impoverished catalog.
    source_data = []
    for pair in _source_pairs():
        answer, data = _decode(pair['input'])
        assert np.array_equal(answer, pair['output'])
        source_data.append(data)
    assert [len(a['votes']) for a in source_data] == [3, 4, 6]
    assert [sum(map(len, a['cavity_raws'].values())) for a in source_data] == [9, 13, 15]
    source_ballots = {tuple(sorted(v.values(), reverse=True))
                      for a in source_data for v in a['votes'].values()}
    assert {(14,), (1,), (3, 2, 2), (4, 3, 1, 1)} <= source_ballots
    puzzles, records, keys = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        keys.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                records.append((split, _features(pair)))
    assert len(keys) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    features = [f for _, f in records]
    assert len({f['input_shape'] for f in features}) >= 40
    assert len({f['output_shape'] for f in features}) >= 50
    assert any(f['input_shape'] == (30, 30) for f in features)
    assert any(max(f['input_shape']) <= 20 for f in features)
    assert len({f['roles'] for f in features}) >= 40
    assert set(range(1, 9)) <= {f['class_count'] for f in features}
    assert max(f['cavity_count'] for f in features) >= 15
    assert len({f['cavity_count'] for f in features}) >= 12
    assert len({f['repetitions'] for f in features}) >= 25
    assert max(f['max_extent'] for f in features) >= 6
    assert max(f['max_area'] for f in features) >= 10
    vocabulary = set().union(*(f['shapes'] for f in features))
    assert len(vocabulary) >= 65
    assert len({s for f in features for s in f['reflection_witnesses']}) >= 12
    assert sum(f['object_row_residues'] == 4 and f['object_col_residues'] == 4
               for f in features) >= 40
    assert sum(f['central_symmetry'] for f in features) >= 10
    assert sum(not f['central_symmetry'] for f in features) >= 30
    assert max(f['density'] for f in features) - min(f['density'] for f in features) > .15
    for feature in ('single_support', 'one_color', 'close', 'nonmajority', 'repeated_losers'):
        assert sum(f[feature] for f in features) >= 25, feature
    all_ballots = {b for f in features for b in f['ballots']}
    assert len(all_ballots) >= 25 and max(map(sum, all_ballots)) >= 12
    assert sum(len(set(f['ballots'])) > 1 for f in features) >= 100
    for split in ('train', 'test'):
        selected = [f for s, f in records if s == split]
        assert {f['aspect'] for f in selected} == {'square', 'tall', 'wide'}
        assert all(any(f['edges'][i] for f in selected) for i in range(4))
        assert any(not any(f['edges']) for f in selected)
        assert any(f['shared_groups'] >= 2 for f in selected)
        assert any(f['distinct_winners'] and f['class_count'] >= 3 for f in selected)
    unseen, familiar, combined, rotation_only = 0, 0, 0, 0
    for puzzle in puzzles:
        train = [_features(p) for p in puzzle['train']]
        test = [_features(p) for p in puzzle['test']]
        known = set().union(*(f['shapes'] for f in train))
        unseen += any(f['shapes'] - known for f in test)
        familiar += any(f['shapes'] <= known for f in test)
        rotation_only += not any(f['reflection_required'] for f in train+test)
        for f in test:
            combined += bool(f['shapes'] - known and f['max_extent'] >= 4
                             and f['cavity_count'] > f['class_count'] and f['nonmajority'])
        assert len({f['roles'] for f in train}) >= 2
    assert unseen >= 20 and familiar >= 5 and combined >= 5 and rotation_only >= 5
    for nt, ne in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9173, nt, ne)
        assert nt is None or len(p['train']) == nt
        assert ne is None or len(p['test']) == ne
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(9173, nt, ne))
    assert _puzzle_key(generate_puzzle(9173)) != _puzzle_key(generate_puzzle(9174))
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corruption checks ensure the validator is a check on visible grids.
    p = puzzles[0]
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect carrier accepted')
