"""Borrowed-color shorelines: cavity addresses route colors between rooms.

No parent generator, external data, training labels, or random state is used by
solve(). Construction labels are used only to make independently checkable pairs.
"""
from __future__ import annotations

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _flood(mask, start):
    reached = np.zeros(mask.shape, dtype=bool)
    if not mask[start]:
        return reached
    reached[start] = True
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            a, b = r + dr, c + dc
            if 0 <= a < h and 0 <= b < w and mask[a, b] and not reached[a, b]:
                reached[a, b] = True
                stack.append((a, b))
    return reached


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    steps = N8 if diagonal else N4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(cells)
    return components


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = mask.copy()
    for dr, dc in N8 if diagonal else N4:
        out |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _check_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _decode(input_grid):
    """Recover walls, singleton counters, donors and selected regions from input."""
    grid = _check_grid(input_grid)
    h, w = grid.shape
    assert h >= 8 and w >= 8
    wall_color = int(grid[0, 0])
    assert np.all(_perimeter(grid) == wall_color), 'incomplete frame'
    bg = int(grid[1, 1])
    assert bg != wall_color
    assert all(grid[r, c] == bg for r, c in ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2)))
    walls = _flood(grid == wall_color, (0, 0))
    dots = (grid == wall_color) & ~walls
    assert all(len(c) == 1 for c in _components(dots)), 'counters must be isolated'
    assert dots.any()
    donors = {}
    foreground = (grid != bg) & (grid != wall_color)
    for cells in _components(foreground):
        r0, r1, c0, c1 = _box(cells)
        motif = grid[r0:r1, c0:c1]
        colors = set(map(int, np.unique(motif))) - {bg}
        assert len(colors) == 1, 'donor must be monochrome'
        color = colors.pop()
        assert color != wall_color and min(motif.shape) >= 4
        assert np.all(_perimeter(motif) == color), 'donor perimeter must be closed'
        assert np.count_nonzero(motif == color) == len(cells)
        holes = motif == bg
        cavities = _components(holes)
        count = len(cavities)
        assert count >= 1 and count not in donors, 'ambiguous cavity address'
        donors[count] = {'color': color, 'box': (r0, r1, c0, c1),
                         'holes': holes.copy(), 'areas': tuple(sorted(map(len, cavities)))}
    assert len(donors) >= 2
    assert len({d['color'] for d in donors.values()}) == len(donors)
    # Donors are interior islands with one ambient cell around their full boxes.
    occupied = np.zeros(grid.shape, bool)
    for donor in donors.values():
        r0, r1, c0, c1 = donor['box']
        assert r0 >= 2 and c0 >= 2 and r1 <= h-2 and c1 <= w-2
        halo = np.zeros(grid.shape, bool)
        halo[r0-1:r1+1, c0-1:c1+1] = True
        box = np.zeros(grid.shape, bool)
        box[r0:r1, c0:c1] = True
        assert not np.any(walls & halo)
        assert not np.any(occupied & halo), 'donors must be separated'
        occupied |= box
    passable = (grid == bg) | dots
    room_list, visited = [], np.zeros(grid.shape, bool)
    for cell in map(tuple, np.argwhere(dots).tolist()):
        if visited[cell]:
            continue
        room = _flood(passable, cell)
        visited |= room
        count = int(np.count_nonzero(room & dots))
        assert count in donors, 'no donor at room address'
        # The outside of a room includes walls and other rooms, but not islands.
        allowed = np.pad(~room, 1, constant_values=True)
        exterior = _flood(allowed, (0, 0))
        rim = _dilate(exterior)[1:-1, 1:-1] & room
        orth_rim = _dilate(exterior, False)[1:-1, 1:-1] & room
        room_list.append({'mask': room, 'count': count, 'rim': rim,
                          'diagonal': rim & ~orth_rim,
                          'color': donors[count]['color']})
    return {'grid': grid, 'background': bg, 'wall_color': wall_color,
            'walls': walls, 'dots': dots, 'donors': donors, 'rooms': room_list}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    output = info['grid'].copy()
    for room in info['rooms']:
        output[room['rim'] & (info['grid'] == info['background'])] = room['color']
    return output


def _cavities(rng, h, w, count, witness=False):
    """Connected growth of independent cavities with an intact donor shell."""
    mask = np.zeros((h, w), bool)
    if witness and count == 1:
        mask[1:-1, 1:-1] = True
        return mask
    if witness and count == 2:
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
        if rng.integers(2):
            mask = np.flipud(mask)
        if rng.integers(2):
            mask = np.fliplr(mask)
        return mask.copy()
    if count == 1 and rng.random() < .35:
        mask[1:-1, 1:-1] = True
        return mask
    # An independent set initializes the cavities. Keep all foreground joined
    # to the outer shell; four- and eight-connectivity are intentionally distinct.
    for _ in range(40):
        trial = np.zeros((h, w), bool)
        positions = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
        rng.shuffle(positions)
        for r, c in positions:
            if any(trial[r+dr, c+dc] for dr, dc in N4):
                continue
            trial[r, c] = True
            if len(_components(~trial)) != 1:
                trial[r, c] = False
            if trial.sum() == count:
                break
        if trial.sum() == count:
            mask = trial
            break
    else:
        # Five mutually nonadjacent cells fit a 3x3 interior. The shell stays
        # connected. This finite fallback covers every sampled size/key.
        for r, c in [(1, 1), (1, 3), (3, 1), (3, 3), (2, 2)][:count]:
            mask[r, c] = True
        assert len(_components(mask)) == count and len(_components(~mask)) == 1
    target = int(rng.integers(count, (h-2)*(w-2)+1))
    cells = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
    for _ in range(h*w):
        if mask.sum() >= target:
            break
        rng.shuffle(cells)
        grew = False
        for r, c in cells:
            if mask[r, c] or not any(mask[r+dr, c+dc] for dr, dc in N4):
                continue
            mask[r, c] = True
            if len(_components(mask)) == count and len(_components(~mask)) == 1:
                grew = True
                break
            mask[r, c] = False
        if not grew:
            break
    return mask


def _layout(rng, number, witness=False, stair=False):
    """Frame-connected partitions; no square-tile mosaic is used."""
    if witness:
        h, w, minimum = int(rng.integers(15, 29)), int(rng.integers(23, 31)), 10
    else:
        h = int(rng.integers(19 if number >= 3 else 12, 31))
        w = int(rng.integers(21, 31))
        minimum = 8
    labels = np.zeros((h, w), dtype=int)
    labels[1:-1, 1:-1] = 1
    rectangles = [(1, h-1, 1, w-1)]
    while len(rectangles) < number:
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rectangles)
                   for axis, extent in ((0, b-a), (1, d-c)) if extent >= 2*minimum+1]
        if not options:
            return None
        if (witness or stair) and len(rectangles) == 1:
            options = [option for option in options if option[1] == 1]
        i, axis = _pick(rng, options)
        a, b, c, d = rectangles.pop(i)
        if axis == 0:
            cut = int(rng.integers(a+minimum, b-minimum))
            rectangles.extend(((a, cut, c, d), (cut+1, b, c, d)))
        else:
            cut = int(rng.integers(c+minimum, d-minimum))
            rectangles.extend(((a, b, c, cut), (a, b, cut+1, d)))
    labels[:] = 0
    for index, (a, b, c, d) in enumerate(rectangles, 1):
        labels[a:b, c:d] = index
    if stair and number == 2:
        # Offset a vertical divider over several independently sized bands.
        # A horizontal first split is equivalent after a transpose.
        wall_columns = [c for c in range(2, w-2) if np.all(labels[1:-1, c] == 0)]
        assert wall_columns
        center = wall_columns[0]
        cuts = sorted(set([1, h-1] + list(map(int, rng.integers(3, h-3, size=3)))))
        previous = center
        for a, b in zip(cuts, cuts[1:]):
            col = int(np.clip(center + int(rng.integers(-2, 3)), 9, w-10))
            labels[a:b, 1:col] = 1
            labels[a:b, col] = 0
            labels[a:b, col+1:w-1] = 2
            if a > 1:
                labels[a, min(col, previous):max(col, previous)+1] = 0
            previous = col
    # Attached rectangular intrusions produce concave shorelines. They alter
    # the actual room labels, so the label constructor ignores donor holes.
    attempts = int(rng.integers(2, 8))
    for index in range(attempts):
        boundaries = []
        for r, c in np.argwhere(labels == 0):
            r, c = int(r), int(c)
            for dr, dc in N4:
                a, b = r+dr, c+dc
                if 2 <= a < h-2 and 2 <= b < w-2 and labels[a, b] > 0:
                    boundaries.append((a, b, dr, dc))
        if not boundaries:
            break
        a, b, dr, dc = _pick(rng, boundaries)
        depth = 1 if witness and index == 0 else int(rng.integers(1, 4))
        width = int(rng.integers(1, 4))
        trial = labels.copy()
        for distance in range(depth):
            for offset in range(width):
                r = a + dr*distance + dc*offset
                c = b + dc*distance + dr*offset
                if 2 <= r < h-2 and 2 <= c < w-2:
                    trial[r, c] = 0
        if all(len(_components(trial == k)) == 1 for k in range(1, number+1)):
            labels = trial
    return labels


def _origins(labels, room_id, occupied, height, width):
    result = []
    h, w = labels.shape
    for r in range(2, h-height-1):
        for c in range(2, w-width-1):
            area = np.s_[r-1:r+height+1, c-1:c+width+1]
            if np.all(labels[area] == room_id) and not np.any(occupied[area]):
                result.append((r, c))
    return result


def _make_pair(rng, number, active, keys, witness, stair, roles, cross=False):
    labels = _layout(rng, number, witness=witness, stair=stair)
    if labels is None:
        return None
    h, w = labels.shape
    bg, wall = roles
    donor_colors = list(map(int, rng.choice([c for c in range(10) if c not in roles], len(keys), replace=False)))
    active_ids = list(map(int, rng.permutation(np.arange(1, number+1))[:active]))
    addressed = list(map(int, rng.permutation(keys)[:active]))
    # The special witness uses exactly keys 1 and 2, with exchanged hosts.
    if witness:
        addressed = list(map(int, rng.permutation([1, 2])))
    addresses = dict(zip(active_ids, addressed))
    grid = np.full(labels.shape, bg, dtype=np.int8)
    grid[labels == 0] = wall
    occupied = np.zeros(labels.shape, bool)
    ambient = labels > 0
    donors = {}
    for key, color in zip(keys, donor_colors):
        sizes = [(a, b) for a in range(5, 8) for b in range(5, 8)]
        rng.shuffle(sizes)
        # Size is independent of address. Larger keys may use the same box.
        placed = False
        for sh, sw in sizes:
            hosts = list(map(int, rng.permutation(np.arange(1, number+1))))
            if cross or witness:
                hosts = [host for host in hosts if addresses.get(host) != key]
            choices = []
            for host in hosts:
                origins = _origins(labels, host, occupied, sh, sw)
                if origins:
                    choices.append((host, origins))
            if not choices:
                continue
            host, origins = _pick(rng, choices)
            r, c = _pick(rng, origins)
            holes = _cavities(rng, sh, sw, key, witness=witness)
            motif = np.full((sh, sw), color, dtype=np.int8)
            motif[holes] = bg
            grid[r:r+sh, c:c+sw] = motif
            occupied[r:r+sh, c:c+sw] = True
            ambient[r:r+sh, c:c+sw] = False
            donors[key] = color
            placed = True
            break
        if not placed:
            return None
    # Count positions are independent of cavity geometry and host arrangement.
    for host, key in addresses.items():
        candidates = np.argwhere((labels == host) & ambient)
        rng.shuffle(candidates)
        selected = []
        for r, c in candidates:
            r, c = int(r), int(c)
            if any(grid[r+dr, c+dc] == wall for dr, dc in N4):
                continue
            grid[r, c] = wall
            selected.append((r, c))
            if len(selected) == key:
                break
        if len(selected) != key:
            return None
    output = grid.copy()
    for host, key in addresses.items():
        room = labels == host
        # Construction uses the pre-donor room geometry, independently of the
        # input-only solver's complement flood. Cavities never become shores.
        rim = room & _dilate(~room)
        output[rim & (grid == bg)] = donors[key]
    turns = int(rng.integers(4))
    grid, output = np.rot90(grid, turns), np.rot90(output, turns)
    if rng.integers(2):
        grid, output = np.fliplr(grid), np.fliplr(output)
    pair = {'input': grid.copy(), 'output': output.copy()}
    try:
        info = _decode(pair['input'])
        if not np.array_equal(solve(pair['input']), pair['output']):
            return None
        features = _features(info)
        if not features['unoutlined']:
            return None
        if witness and not (features['area_inversion'] and features['diagonal']):
            return None
    except AssertionError:
        return None
    return pair


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _validate_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _validate_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # These are curricula, never a hidden rule or a learned color dictionary.
    regime = int(rng.integers(3))
    stable_palette = bool(rng.integers(2))
    base_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = split == 'train' and index == 0
            if witness:
                number, active, keys = 2, 2, [1, 2]
            elif split == 'train':
                number = 3 if index == 1 else int(rng.integers(2, 4))
                active = 2
                keys = list(map(int, rng.choice(np.arange(1, 5), 2, replace=False)))
                if index == 1 and set(keys) == {1, 2}:
                    keys = [1, int(rng.integers(3, 5))]
                if index > 1 and rng.random() < .4:
                    keys.append(_pick(rng, [k for k in range(1, 6) if k not in keys and (regime != 1 or k != 5)]))
            else:
                number = 4 if regime == 0 else 2 if regime == 2 else int(rng.integers(2, 5))
                active = 3 if regime == 0 else min(number, int(rng.integers(2, 4)))
                keys = list(map(int, rng.choice(np.arange(1, 6), active, replace=False)))
                if regime == 1 and 5 not in keys:
                    keys[-1] = 5
                if len(keys) < 4 and rng.random() < .4:
                    keys.append(_pick(rng, [k for k in range(1, 6) if k not in keys]))
            roles = base_roles if stable_palette else tuple(map(int, rng.choice(10, 2, replace=False)))
            for attempt in range(160):
                # The key curriculum explicitly checks that address 5 is active,
                # rather than merely present on an unused donor.
                pair = _make_pair(rng, number, active, keys, witness,
                                  stair=(regime == 2 and split == 'test' and number == 2),
                                  roles=roles, cross=(witness or index == 1))
                if pair is None or _grid_key(pair['input']) in seen:
                    continue
                info = _decode(pair['input'])
                if split == 'test' and regime == 1 and not any(r['count'] == 5 for r in info['rooms']):
                    continue
                if split == 'train' and index == 1 and not _features(info)['inactive']:
                    continue
                seen.add(_grid_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f'bounded scene construction exhausted: {split}/{index}')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    grid, bg, dots, donors = info['grid'], info['background'], info['dots'], info['donors']
    rooms = info['rooms']
    selected = np.zeros(grid.shape, bool)
    for room in rooms:
        selected |= room['mask']
    donor_boxes = np.zeros(grid.shape, bool)
    nonlocal_routes = 0
    unoutlined = 0
    for key, donor in donors.items():
        r0, r1, c0, c1 = donor['box']
        donor_boxes[r0:r1, c0:c1] = True
        box = np.zeros(grid.shape, bool)
        box[r0:r1, c0:c1] = True
        halo = _dilate(box) & ~box
        host = [room for room in rooms if np.any(room['mask'] & halo)]
        if any(room['count'] == key for room in rooms) and not any(room['count'] == key for room in host):
            nonlocal_routes += 1
        for room in host:
            unoutlined += int(np.count_nonzero(halo & room['mask'] & ~room['rim'] & (grid == bg)))
    ambient = ((grid == bg) | dots) & ~donor_boxes
    components = _components(ambient)
    active_sizes = [int(room['mask'].sum()) for room in rooms]
    total_sizes = [len(cells) for cells in components]
    areas = {key: sum(donor['areas']) for key, donor in donors.items()}
    return {'shape': grid.shape, 'room_count': len(components), 'active_count': len(rooms),
            'donor_count': len(donors), 'keys': tuple(sorted(donors)),
            'active_keys': tuple(sorted(room['count'] for room in rooms)),
            'inactive': len(components) > len(rooms), 'nonlocal': nonlocal_routes,
            'unoutlined': unoutlined,
            'area_inversion': any(a < b and areas[a] > areas[b] for a in areas for b in areas),
            'four_eight': any(len(_components(d['holes'], True)) < key for key, d in donors.items()),
            'diagonal': sum(int(np.count_nonzero(r['diagonal'] & (grid == bg))) for r in rooms),
            'dot_on_rim': any(np.any(r['rim'] & dots) for r in rooms),
            'small_selected': min(active_sizes) < max(total_sizes),
            'palette': (bg, info['wall_color'])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _validate_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _decode(inp)
            assert np.array_equal(solve(inp), out), 'wrong shoreline or routed color'
            change = inp != out
            assert change.any() and np.all(inp[change] == info['background'])
            assert np.array_equal(inp[info['dots']], out[info['dots']])
            assert np.array_equal(inp[info['walls']], out[info['walls']])
            for donor in info['donors'].values():
                a, b, c, d = donor['box']
                assert np.array_equal(inp[a:b, c:d], out[a:b, c:d]), 'donor altered'
            f = _features(info)
            assert f['active_count'] >= 2 and f['unoutlined'] > 0
            if split == 'train':
                training.append(f)
    # Whole-split evidence, including a two-demonstration puzzle.
    assert any(f['area_inversion'] and f['four_eight'] for f in training)
    assert any(f['diagonal'] for f in training)
    assert any(f['nonlocal'] >= 2 for f in training)
    assert any(f['inactive'] for f in training)
    assert len({f['active_keys'] for f in training}) >= 2


def _puzzle_key(puzzle):
    return tuple(_grid_key(pair[role]) for split in ('train', 'test')
                 for pair in puzzle[split] for role in ('input', 'output'))


def self_test():
    """Bounded semantic, metamorphic and 42-seed structural checks."""
    signatures, shapes, room_counts, donor_shapes, palettes, key_sets = set(), set(), set(), set(), set(), set()
    features = []
    counter_fixtures = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                f = _features(info)
                features.append(f)
                shapes.add(f['shape'])
                room_counts.add(f['room_count'])
                palettes.add(f['palette'])
                key_sets.add(f['keys'])
                if f['inactive'] and 1 in f['active_keys']:
                    counter_fixtures.append(pair)
                for donor in info['donors'].values():
                    donor_shapes.add((donor['holes'].shape, donor['holes'].tobytes()))
    assert len(signatures) == 42
    assert len(shapes) >= 35 and {2, 3, 4} <= room_counts
    assert len(donor_shapes) >= 100 and len(palettes) >= 25 and len(key_sets) >= 12
    assert {2, 3} <= {f['active_count'] for f in features}
    assert any(f['donor_count'] > f['active_count'] for f in features)
    assert any(5 in f['active_keys'] for f in features)
    assert any(f['dot_on_rim'] for f in features)
    # D4 equivariance and arbitrary bijective recoloring preserve the rule.
    puzzle = generate_puzzle(0, 2, 1)
    assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(0, 2, 1))
    pair = puzzle['train'][0]
    for turns in range(4):
        for flip in (False, True):
            inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
            if flip:
                inp, out = np.fliplr(inp), np.fliplr(out)
            before = inp.copy()
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(inp, before)
    lookup = np.random.default_rng(19).permutation(10)
    assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])
    # Recoloring one addressed donor changes only its recipient shorelines.
    inp = pair['input'].copy()
    info = _decode(inp)
    key = info['rooms'][0]['count']
    donor = info['donors'][key]
    fresh = next(c for c in range(10) if c not in inp)
    changed = inp.copy()
    changed[inp == donor['color']] = fresh
    changed_out = solve(changed)
    expected = pair['output'].copy()
    expected[expected == donor['color']] = fresh
    assert np.array_equal(changed_out, expected)
    # Move one counter into an unmarked room. The selected region and address
    # both change; a global counter sum or fixed-region heuristic cannot work.
    moved_counter = False
    for counter_fixture in counter_fixtures:
        info = _decode(counter_fixture['input'])
        key1 = next(r for r in info['rooms'] if r['count'] == 1)
        if _counter_move_test(counter_fixture, info, key1):
            moved_counter = True
            break
    assert moved_counter, 'no fixture permitted moving a counter across a wall'
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted boundary accepted')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            candidate = generate_puzzle(42, nt, ne)
            assert len(candidate['train']) == nt and len(candidate['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'input_shapes': len(shapes),
            'room_counts': sorted(room_counts), 'donor_masks': len(donor_shapes),
            'role_palettes': len(palettes), 'key_sets': len(key_sets)}


def _counter_move_test(pair, info, room):
    inp, bg = pair['input'].copy(), info['background']
    donor_boxes = np.zeros(inp.shape, bool)
    for donor in info['donors'].values():
        a, b, c, d = donor['box']
        donor_boxes[a:b, c:d] = True
    selected = np.zeros(inp.shape, bool)
    for other in info['rooms']:
        selected |= other['mask']
    available = (inp == bg) & ~donor_boxes & ~selected
    for r, c in np.argwhere(available):
        r, c = int(r), int(c)
        if all(inp[r+dr, c+dc] != info['wall_color'] for dr, dc in N4):
            inp[room['mask'] & info['dots']] = bg
            inp[r, c] = info['wall_color']
            result = solve(inp)
            assert np.any((pair['output'] != pair['input']) & (result == bg))
            assert np.any((result != inp) & ~selected)
            return True
    return False
