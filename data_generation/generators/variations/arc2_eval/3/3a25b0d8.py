"""Transfer uniform chamber colors between visibly corresponding frameworks.

The invariant is enclosure and spatial correspondence, not pixel repetition.
Literal colors and both geometries are local to an example. No learned hidden
palette, orientation, or index is needed to solve a query.
"""
import hashlib
import numpy as np

_SOURCE_DIGESTS = ('54e2700ee37c4b4e545dbddd63fa8a76aea3833f6c7e499dab0e77c99d7bc787', 'ad80b565cc3832c5f92d01385f4c5f95286d7ce5a6f74a67e91bf20898574f3d', 'fce79453e2c7f1af168af2d824827132141d799d474fc1a3d1617ec9b38bca47', '8b083092f71c59d5b8709b5ad0bd0f1494ffaff32e78a03abe85d24100e1335d')

def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if name == "num_train" and not 2 <= value <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if name == "num_test" and not 1 <= value <= 4:
        raise ValueError("num_test must be between 1 and 4")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train")
    num_test = _normalize_count(num_test, "num_test")

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees broad count diversity over
        # consecutive seeds while retaining legal relationships.
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        span = 11 - low
        num_train = low + (seed // 4) % span
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = int(rng.integers(low, 11))
    elif num_test is None:
        high = min(4, num_train - 1)
        if high < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, high + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("pair counts must satisfy num_train > num_test")

    return num_train, num_test


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    part.append(point)
        result.append(np.asarray(part, dtype=int))
    return result


def _bbox(points):
    return tuple(int(v) for v in points.min(0)) + tuple(
        int(v) for v in points.max(0) + 1
    )


def _rooms(grid, wall):
    # Padding distinguishes exterior from a chamber, even at a canvas edge.
    mask = np.pad(grid != wall, 1, constant_values=True)
    return [p - 1 for p in _components(mask)
            if np.all(p > 0) and np.all(p < np.array(mask.shape) - 1)]


def _extract(inp):
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    background = int(np.bincount(border, minlength=10).argmax())
    parts = _components(inp != background, diagonal=True)
    assert len(parts) == 2, "expected two separate figures"
    objects = []
    for part in parts:
        r, c, R, C = _bbox(part)
        crop = np.full((R - r, C - c), background, dtype=np.int8)
        crop[part[:, 0] - r, part[:, 1] - c] = inp[tuple(part.T)]
        objects.append((crop, (r, c, R, C)))
    monochrome = [o for o in objects if len(set(o[0].flat) - {background}) == 1]
    colored = [o for o in objects if len(set(o[0].flat) - {background}) > 1]
    assert len(monochrome) == len(colored) == 1
    target, tb = monochrome[0]
    donor, db = colored[0]
    wall = int(next(iter(set(target.flat) - {background})))
    assert wall in donor
    dr, tr = _rooms(donor, wall), _rooms(target, wall)
    assert len(dr) == len(tr) and len(dr) >= 2
    colors = []
    for room in dr:
        values = np.unique(donor[tuple(room.T)])
        assert len(values) == 1 and values[0] not in (background, wall)
        colors.append(int(values[0]))
    painted = sum(len(room) for room in dr)
    assert painted == np.count_nonzero((donor != background) & (donor != wall))
    assert all(np.all(target[tuple(room.T)] == background) for room in tr)
    return background, wall, donor, target, dr, tr, colors, db, tb


def _relations(rooms):
    """Complete spatial graph: separated/overlapping projections on each axis.

    Room discovery order, color, area, and generation labels are absent.
    """
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _correspondences(donor_rooms, target_rooms):
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    # A local boundary can change from overlap to separation, but cannot
    # reverse a visible left/right or upper/lower relationship.
    compatible = np.zeros((n, n, n, n), dtype=bool)
    score = np.zeros((n, n, n, n), dtype=np.int8)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                for v in range(n):
                    compatible[i, k, j, v] = all(
                        not (u and w and u != w) for u, w in zip(a[i][k], b[j][v]))
                    score[i, k, j, v] = sum(u == w for u, w in zip(a[i][k], b[j][v]))
    best_score, best, visits = -1, [], 0

    def search(mapping, used, value):
        nonlocal best_score, best, visits
        visits += 1
        assert visits <= 20000, "insufficiently distinguished spatial graph"
        if len(mapping) == n:
            if value > best_score:
                best_score, best = value, [dict(mapping)]
            elif value == best_score:
                best.append(dict(mapping))
            return
        options = {}
        for i in range(n):
            if i not in mapping:
                options[i] = [j for j in range(n) if j not in used and all(
                    compatible[i, k, j, v] for k, v in mapping.items())]
        i = min(options, key=lambda k: len(options[k]))
        for j in options[i]:
            extra = sum(int(score[i, k, j, v]) for k, v in mapping.items())
            mapping[i] = j
            search(mapping, used | {j}, value + extra)
            del mapping[i]

    search({}, set(), 0)
    assert best, "no orientation-preserving correspondence"
    return best


def _predict(inp):
    info = _extract(inp)
    bg, wall, donor, target, dr, tr, colors, db, tb = info
    mappings = _correspondences(dr, tr)
    answers = []
    for mapping in mappings:
        answer = target.copy()
        for i, j in mapping.items():
            answer[tuple(tr[j].T)] = colors[i]
        if answers:
            assert np.array_equal(answer, answers[0]), "ambiguous compartment colors"
        else:
            answers.append(answer)
    return answers[0], info, mappings[0]


def solve(train_pairs, test_input):
    """Input-only prediction; training establishes literal fill and tight crop."""
    assert len(train_pairs) >= 1
    return _predict(np.array(test_input, copy=True))[0]


def _abstract(rng, richness):
    # A connected, irregular polyomino is partitioned by connected growth.
    # This produces many floorplans, rather than selecting a template bank.
    shape_options = ((2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (3, 4),
                     (4, 2), (2, 4), (5, 2), (2, 5), (5, 3), (3, 5))
    options = [s for s in shape_options if s[0] * s[1] >= richness]
    rows, cols = options[int(rng.integers(len(options)))]
    mask = np.ones((rows, cols), dtype=bool)
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    for i in rng.permutation(4):
        r, c = corners[int(i)]
        if mask.sum() > richness and rng.random() < 0.65:
            candidate = mask.copy()
            candidate[r, c] = False
            if len(_components(candidate)) == 1:
                mask = candidate
    labels = np.full(mask.shape, -1, dtype=int)
    points = np.argwhere(mask)
    for i, p in enumerate(points):
        labels[tuple(p)] = i
    groups = len(points)
    while groups > richness:
        edges = [(int(labels[r, c]), int(labels[r + dr, c + dc]))
                 for r, c in points for dr, dc in ((1, 0), (0, 1))
                 if r + dr < rows and c + dc < cols
                 and labels[r + dr, c + dc] >= 0
                 and labels[r, c] != labels[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        labels[labels == b] = a
        groups -= 1
    for new, old in enumerate(sorted(set(labels.flat) - {-1})):
        labels[labels == old] = new
    # Common orientation within a pair; the family has no rotation operation.
    return labels


def _geometry(rng, plan, widths, heights, detail):
    ys = np.concatenate(([0], np.cumsum(heights + 1)))
    xs = np.concatenate(([0], np.cumsum(widths + 1)))
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1]):
            label = plan[r, c]
            if label < 0:
                continue
            labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = label
            if r and plan[r - 1, c] == label:
                labels[ys[r], xs[c] + 1:xs[c + 1]] = label
            if c and plan[r, c - 1] == label:
                labels[ys[r] + 1:ys[r + 1], xs[c]] = label
            if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == label):
                labels[ys[r], xs[c]] = label
    # Independent local notches and shifted chamber starts change boundaries
    # without treating colored rows as textures to stretch.
    for label in sorted(set(plan.flat) - {-1}):
        for _ in range(detail):
            points = np.argwhere(labels == label)
            if len(points) <= 1 or rng.random() < 0.45:
                continue
            r, c, R, C = _bbox(points)
            corner = [(r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1)][int(rng.integers(4))]
            if labels[corner] == label:
                trial = labels == label
                trial[corner] = False
                if len(_components(trial)) == 1:
                    labels[corner] = -2
    # Framework comes from room boundaries, including shared junctions.
    occupied = labels >= 0
    near = np.zeros(labels.shape, dtype=bool)
    padded = np.pad(occupied, 1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    # Beveled exterior corners allow genuine diagonal wall connections.
    if rng.random() < 0.7:
        orthogonal = (padded[:-2, 1:-1] | padded[2:, 1:-1]
                      | padded[1:-1, :-2] | padded[1:-1, 2:])
        diagonal_count = sum(padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]].astype(int)
                             for dr, dc in ((0, 0), (0, 2), (2, 0), (2, 2)))
        labels[(labels == -2) & ~orthogonal & (diagonal_count == 1)] = -1
    r, c, R, C = _bbox(np.argwhere(labels != -1))
    return labels[r:R, c:C]


def _paint(labels, background, wall, colors=None):
    grid = np.full(labels.shape, background, dtype=np.int8)
    grid[labels == -2] = wall
    if colors is not None:
        for label, color in enumerate(colors):
            grid[labels == label] = color
    return grid


def _placement_options(donor, target, bg):
    a, b = np.argwhere(donor != bg), np.argwhere(target != bg)
    # Mark every relative translation that would touch in eight-connectivity.
    delta = a[:, None, :] - b[None, :, :]
    blocked = np.zeros((63, 63), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            q = delta.reshape(-1, 2) + (31 + dr, 31 + dc)
            blocked[q[:, 0], q[:, 1]] = True
    options = {k: [] for k in ('side', 'stack', 'diagonal', 'interleaved')}
    dh, dw = donor.shape
    th, tw = target.shape
    for r in range(-th - 4, dh + 5):
        for c in range(-tw - 4, dw + 5):
            height, width = max(dh, r + th) - min(0, r), max(dw, c + tw) - min(0, c)
            if height > 30 or width > 30 or blocked[r + 31, c + 31]:
                continue
            row_overlap = min(dh, r + th) > max(0, r)
            col_overlap = min(dw, c + tw) > max(0, c)
            mode = ('interleaved' if row_overlap and col_overlap else
                    'side' if row_overlap else 'stack' if col_overlap else 'diagonal')
            options[mode].append((r, c, height, width))
    return options


def _embed(rng, donor, target, bg, requested_layout, padding):
    options = _placement_options(donor, target, bg)
    available = [k for k, v in options.items() if v]
    assert available, "figures do not fit"
    layout = requested_layout if options[requested_layout] else available[int(rng.integers(len(available)))]
    r, c, height, width = options[layout][int(rng.integers(len(options[layout])))]
    dh, dw = donor.shape
    th, tw = target.shape
    extra_r = int(rng.integers(min(30 - height, padding) + 1))
    extra_c = int(rng.integers(min(30 - width, padding) + 1))
    top = int(rng.integers(extra_r + 1))
    left = int(rng.integers(extra_c + 1))
    canvas = np.full((height + extra_r, width + extra_c), bg, dtype=np.int8)
    for grid, rr, cc in ((donor, top - min(0, r), left - min(0, c)),
                         (target, top + max(0, r), left + max(0, c))):
        region = canvas[rr:rr + grid.shape[0], cc:cc + grid.shape[1]]
        mask = grid != bg
        region[mask] = grid[mask]
    return canvas


def _pair(rng, regime, settings, index, forced_count=None, forced_colors=None,
          excluded_roles=None):
    assert regime in ('source_train', 'target_test')
    mode, held_factor = settings
    is_test = regime == 'target_test'
    max_rooms = 12 if mode != 'held_out' or held_factor != 'rooms' or is_test else 7
    min_rooms = 8 if mode == 'held_out' and held_factor == 'rooms' and is_test else 3
    count = forced_count or int(rng.integers(min_rooms, max_rooms + 1))
    palette = rng.permutation(10)
    if excluded_roles is not None:
        background = next(int(c) for c in palette if c != excluded_roles[0])
        framework = next(int(c) for c in palette if c not in (background, excluded_roles[1]))
        palette = np.array([background, framework] +
                           [int(c) for c in palette if c not in (background, framework)])
    bg, wall = int(palette[0]), int(palette[1])
    n_colors = forced_colors or int(rng.integers(2, min(count, 8) + 1))
    colors = list(map(int, palette[2:2 + n_colors]))
    colors += [colors[int(rng.integers(n_colors))] for _ in range(count - n_colors)]
    rng.shuffle(colors)
    layouts = ('side', 'stack', 'diagonal', 'interleaved')
    if mode == 'held_out' and held_factor == 'layout':
        layout = 'interleaved' if is_test else layouts[int(rng.integers(3))]
    else:
        layout = layouts[int(rng.integers(4))]
    detail = int(rng.integers(0, 5))
    if mode == 'held_out' and held_factor == 'detail':
        detail = int(rng.integers(2, 5)) if is_test else 0
    padding = int(rng.integers(0, 13))
    for attempt in range(48):
        plan = _abstract(rng, count)
        # The last construction is a fully ordered grid with guaranteed
        # distinguishable rooms, so rejection cannot exhaust a legal request.
        if attempt == 47:
            cols = 3 if count > 6 else 2
            rows = (count + cols - 1) // cols
            plan = np.arange(rows * cols).reshape(rows, cols)
            plan[plan >= count] = count - 1
        widths = rng.integers(1, 4, size=plan.shape[1])
        heights = rng.integers(1, 4, size=plan.shape[0])
        other_widths = rng.integers(1, 4, size=plan.shape[1])
        other_heights = rng.integers(1, 4, size=plan.shape[0])
        # Equal total heights are a regular mode, with local redistribution.
        if rng.random() < 0.25:
            other_heights = heights[rng.permutation(len(heights))].copy()
        if attempt == 47:
            widths[:] = 2
            heights[:] = 2
            other_widths[:] = 1
            other_heights[:] = 3
        d = _geometry(rng, plan, widths, heights, detail if attempt < 47 else 0)
        t = _geometry(rng, plan, other_widths, other_heights, detail if attempt < 47 else 0)
        donor, target = _paint(d, bg, wall, colors), _paint(t, bg, wall)
        expected = _paint(t, bg, wall, colors)
        try:
            if attempt == 47:
                # Guaranteed fit and an unambiguous exterior for the bounded
                # fallback, regardless of the requested placement or margins.
                inp = np.full((30, 30), bg, dtype=np.int8)
                inp[1:1 + donor.shape[0], 1:1 + donor.shape[1]] = donor
                col = donor.shape[1] + 3
                inp[1:1 + target.shape[0], col:col + target.shape[1]] = target
            else:
                inp = _embed(rng, donor, target, bg, layout, padding)
            predicted, info, mapping = _predict(inp)
            assert np.array_equal(predicted, expected)
            assert len(info[4]) == count
            assert not np.array_equal(donor, expected)
            return {'input': inp, 'output': expected}
        except AssertionError:
            continue
    raise AssertionError('deterministic compartment construction failed')


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_digest(pair):
    return hashlib.sha256(b''.join(bytes(pair[k].shape) + pair[k].astype(np.int8).tobytes()
                                   for k in ('input', 'output'))).hexdigest()


def measure_pair(pair):
    predicted, info, mapping = _predict(pair['input'])
    bg, wall, donor, target, dr, tr, colors, db, tb = info
    assert np.array_equal(predicted, pair['output'])
    assert np.array_equal(predicted == wall, target == wall)
    changes = predicted != target
    interior = np.zeros(target.shape, dtype=bool)
    for room in tr:
        interior[tuple(room.T)] = True
    assert np.array_equal(changes, interior)
    assert np.all(target[changes] == bg)
    assert np.array_equal(predicted[~interior], target[~interior])
    sizes = [len(p) for p in dr]
    target_sizes = [len(tr[mapping[i]]) for i in range(len(dr))]
    signs = [int(np.sign(b - a)) for a, b in zip(sizes, target_sizes)]
    row_overlap = min(db[2], tb[2]) > max(db[0], tb[0])
    col_overlap = min(db[3], tb[3]) > max(db[1], tb[1])
    layout = ('interleaved' if row_overlap and col_overlap else 'side' if row_overlap
              else 'stack' if col_overlap else 'diagonal')
    raw_target = pair['input'][tb[0]:tb[2], tb[1]:tb[3]]
    narrow_channels = 0
    for room in dr + tr:
        points = set(map(tuple, room))
        r, c, R, C = _bbox(room)
        if len(room) >= 5 and R - r > 1 and C - c > 1:
            for x, y in points:
                neighbors = ((x - 1, y) in points, (x + 1, y) in points,
                             (x, y - 1) in points, (x, y + 1) in points)
                if neighbors in ((True, True, False, False), (False, False, True, True)):
                    narrow_channels += 1
                    break
    relation = _relations(tr)
    spatial_profile = tuple(sorted(tuple(sorted(row)) for row in relation))
    paired_sides = any(
        relation[left][middle][1] == -1 and relation[right][middle][1] == 1
        and relation[left][middle][0] == relation[right][middle][0] == 0
        for middle in range(len(tr)) for left in range(len(tr)) for right in range(len(tr)))
    return {
        'background': bg, 'wall': wall, 'compartments': len(dr),
        'accent_count': len(set(colors)), 'reused_colors': len(set(colors)) < len(dr),
        'equal_area_different_colors': any(sizes[i] == sizes[j] and colors[i] != colors[j]
                                          for i in range(len(dr)) for j in range(i)),
        'layout': layout, 'edge_contact': any(box[0] == 0 or box[1] == 0 or
                 box[2] == pair['input'].shape[0] or box[3] == pair['input'].shape[1]
                 for box in (db, tb)),
        'crop_contamination': bool(np.any(raw_target != target)),
        'donor_shape': donor.shape, 'target_shape': target.shape,
        'equal_height': donor.shape[0] == target.shape[0],
        'horizontal_change': donor.shape[1] != target.shape[1],
        'height_direction': int(np.sign(target.shape[0] - donor.shape[0])),
        'area_signs': tuple(sorted(set(signs))), 'filled_cells': int(changes.sum()),
        'min_room': min(sizes + target_sizes), 'max_room': max(sizes + target_sizes),
        'nonrectangular_rooms': sum(len(p) != (_bbox(p)[2] - _bbox(p)[0]) *
                                     (_bbox(p)[3] - _bbox(p)[1]) for p in dr + tr),
        'scan_order_changed': any(i != j for i, j in mapping.items()),
        'repeated_wall_rows': any(np.array_equal(target[r] == wall, target[r - 1] == wall)
                                  for r in range(1, len(target))),
        'diagonal_framework': len(_components(target == wall)) > 1,
        'narrow_channels': narrow_channels, 'paired_sides': paired_sides,
        'asymmetric_framework': not np.array_equal(target == wall, (target == wall)[:, ::-1]),
        'spatial_profile': spatial_profile,
        'structure': (target.shape, (target == wall).tobytes()),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair['input'])
            assert key not in seen
            seen.add(key)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            assert not np.array_equal(pair['input'], pair['output'])
            infos[split].append(measure_pair(pair))
    # Training exhibits literal multicolor transfer on different frameworks.
    # Color roles are deliberately per pair; no literal palette is held out.
    assert all(i['accent_count'] >= 2 for i in infos['train'])
    assert len({i['compartments'] for i in infos['train']}) >= 2
    assert len({i['structure'] for i in infos['train']}) >= 2
    for role in ('background', 'wall', 'accent_count'):
        assert len({i[role] for i in infos['train']}) >= 2
    for pair in puzzle['test']:
        assert np.array_equal(solve(puzzle['train'], pair['input']), pair['output'])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    # Only the curriculum is sampled once. The source supports one invariant
    # convention: same orientation, shared local framework, literal colors.
    mode = ('comparable', 'held_out', 'combined')[int(rng.integers(3))]
    factor = ('rooms', 'layout', 'detail')[int(rng.integers(3))]
    settings = mode, factor
    puzzle, seen = {'train': [], 'test': []}, set()
    training_max = 7 if mode == 'held_out' and factor == 'rooms' else 12
    first_count = int(rng.integers(3, training_max + 1))
    second_count = int(rng.choice([n for n in range(3, training_max + 1) if n != first_count]))
    first_colors = int(rng.integers(2, min(first_count, 8) + 1))
    second_colors = int(rng.choice([n for n in range(2, min(second_count, 8) + 1)
                                   if n != first_colors]))
    first_roles = None
    for split, regime, count in (('train', 'source_train', num_train),
                                 ('test', 'target_test', num_test)):
        for index in range(count):
            forced = first_count if split == 'train' and index == 0 else (
                second_count if split == 'train' and index == 1 else None)
            if mode == 'combined' and split == 'test':
                # Richer combinations remain in the training support, without
                # coupling canvas, placement, colors, or deformation direction.
                forced = int(rng.integers(6, 13))
            color_count = (first_colors, second_colors)[index] if split == 'train' and index < 2 else None
            excluded = first_roles if split == 'train' and index == 1 else None
            for retry in range(32):
                pair = _pair(rng, regime, settings, index, forced, color_count, excluded)
                if _grid_key(pair['input']) not in seen:
                    break
            else:
                # Guaranteed unique recoloring of the background/framework
                # roles (at most 14 examples, 90 ordered role pairs).
                background, framework = _extract(pair['input'])[:2]
                old = [background, framework] + [c for c in range(10)
                                                if c not in (background, framework)]
                for bg, wall in ((a, b) for a in range(10) for b in range(10) if a != b):
                    new = [bg, wall] + [c for c in range(10) if c not in (bg, wall)]
                    recolor = np.empty(10, dtype=np.int8)
                    recolor[old] = new
                    replacement = {k: recolor[v] for k, v in pair.items()}
                    if _grid_key(replacement['input']) not in seen:
                        pair = replacement
                        break
                assert _grid_key(pair['input']) not in seen
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train' and index == 0:
                first_roles = _extract(pair['input'])[:2]
    return puzzle


_SOURCE_REGRESSIONS = (
    ('8888888888888888888888/8888888888888888888888/8888111118888888888888/8881188811888888888888/8811188811188888888888/8181111111818888888888/8188188818818888888888/8181111111818888888888/8811881881188111118888/8881111111881177711888/8888188818811117111188/8888188818131141411318/8888111118133144413318/8888888888111111111118/8888888888881661661888/8888888888881111111888/8888888888888199918888/8888888888888811188888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888',
     '88811111888/88117771188/81117771118/13111111131/13314441331/13111111131/81166166118/88111111188/88819991888/88819991888/88811111888'),
    ('1122122111111111111111/1244244211111111111111/1122322111111111111111/1123332111111111111111/1123232111111111111111/1222822211112212211111/1112821111121121121111/1112821111121121121111/1122822111112212211111/1128882111112111211111/1222222211112111211111/1111111111122222221111/1111111111111212111111/1111111111111212111111/1111111111112212211111/1111111111112111211111/1111111111122111221111/1111111111122222221111/1111111111111111111111/1111111111111111111111/1111111111111111111111/1111111111111111111111',
     '1221221/2442442/2442442/1223221/1233321/1233321/2222222/1128211/1128211/1228221/1288821/2288822/2222222'),
    ('000000000000000000000000000000/000444444444000000000000000000/004333348888400000000000000000/004333348888400000000000000000/004443348844400000000000000000/000044444440000000000000000000/000455424554000000000000000000/000455424554000000000000000000/000444222444000000000000000000/004147424741400000000000000000/004447444744400000000000000000/000477747774000000044000440000/044777777777440000400444004000/047777777777740004000040000400/047777444777740004000040000400/044444666444440004440040044400/000000444000000000444444444000/000000040000000004000444000400/000000000000000004000404000400/000000000000000004444000444400/000000000000000004044404440400/000000000000000004440444044400/000000000000000000400040004000/000000000000000044000000000440/000000000000000004000000000400/000000000000000004000040000400/000000000000000004444444444400/000000000000000000004000400000/000000000000000000004444400000/000000000000000000000000000000',
     '0004400044000/0043344488400/0433334888840/0433334888840/0444334884440/0044444444400/0455544455540/0455542455540/0444422244440/0414442444140/0444744474440/0047774777400/4477777777744/0477777777740/0477774777740/0444444444440/0000466640000/0000444440000'),
    ('4444444444444444444444444444/4444444444444444422244444444/4444444444444444428244444444/4444444444444444288824444444/4444444444444222222222224444/4444444444444211277729924444/4444444444444222277722224444/4444444444444442772772444444/4444444444444442222222444444/4444222444444428882888244444/4442242244444428882888244444/4222444222444252223222524444/2442242244244224233324224444/2442424244244444422244444444/2222444222244444424244444444/4224444422444444444444444444/4222222222444444444444444444/4244424442444444444444444444/4244424442444444444444444444/2422242224244444444444444444/2222444222244444444444444444/4442242244444444444444444444/4444222444444444444444444444/4444444444444444444444444444/4444444444444444444444444444/4444444444444444444444444444/4444444444444444444444444444/4444444444444444444444444444',
     '44442224444/44422822444/42228882224/21122822992/21127272992/22227772222/42277777224/42222222224/42888288824/42888288824/25222322252/22223332222/44422322444/44442224444'),
)


def self_test() -> None:
    for inp, output in _SOURCE_REGRESSIONS:
        grid = np.array([[int(c) for c in row] for row in inp.split("/")])
        expected = np.array([[int(c) for c in row] for row in output.split("/")])
        assert np.array_equal(_predict(grid)[0], expected)
    train_counts, test_counts, structures, shapes, roles = set(), set(), set(), set(), set()
    metrics = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                metrics.append(m)
                structures.add(m['structure'])
                shapes.add(pair['input'].shape)
                roles.add((m['background'], m['wall']))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(structures) >= 200 and len(shapes) >= 60 and len(roles) >= 50
    assert len({m['compartments'] for m in metrics}) >= 9
    assert {m['layout'] for m in metrics} == {'side', 'stack', 'diagonal', 'interleaved'}
    assert {m['height_direction'] for m in metrics} == {-1, 0, 1}
    assert max(m['accent_count'] for m in metrics) == 8
    for flag in ('equal_height', 'horizontal_change', 'reused_colors', 'edge_contact',
                 'crop_contamination', 'equal_area_different_colors', 'scan_order_changed',
                 'repeated_wall_rows', 'diagonal_framework'):
        assert sum(bool(m[flag]) for m in metrics) >= 3, flag
    assert sum(m['area_signs'] == (-1, 0, 1) for m in metrics) >= 10
    assert sum(m['nonrectangular_rooms'] >= 3 for m in metrics) >= 30
    assert sum(m['narrow_channels'] > 0 for m in metrics) >= 5
    assert sum(m['paired_sides'] for m in metrics) >= 30
    assert sum(m['asymmetric_framework'] for m in metrics) >= 100
    assert len({m['spatial_profile'] for m in metrics}) >= 40
    assert min(m['min_room'] for m in metrics) == 1
    assert max(m['max_room'] for m in metrics) >= 25
    for requested in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(702, *requested)
        b = generate_puzzle(702, *requested)
        validate_generated_puzzle(a)
        assert all(_grid_key(p[k]) == _grid_key(q[k]) for s in a
                   for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
        for s, value in zip(('train', 'test'), requested):
            assert value is None or len(a[s]) == value
    assert _grid_key(generate_puzzle(701)['train'][0]['input']) != _grid_key(generate_puzzle(702)['train'][0]['input'])
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2, 1.5)):
        try:
            generate_puzzle(703, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Corrupting a wall or an interior must be rejected by visible validation.
    puzzle = generate_puzzle(704, 2, 1)
    pair = puzzle['test'][0]
    pair['output'] = pair['output'].copy()
    pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError('corrupted answer accepted')
