"""Invisible selective shells: a layer code supplies color-specific ray barriers.

All semantics are visible in the input. Black is empty; the shared frame/tail
color is identified geometrically. No generated metadata enters solve().
"""
from collections import Counter

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must have dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    unseen = mask.copy()
    result = []
    neighbors = DIRECTIONS if diagonal else CARDINAL
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if not unseen[start]:
            continue
        unseen[start] = False
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _slice(box):
    r0, r1, c0, c1 = box
    return slice(r0, r1), slice(c0, c1)


def _read(input_grid):
    """Recover the two framed objects, then decode the remaining bare arrows."""
    a = _grid(input_grid)
    codes, boards = [], []
    for structural in range(1, 10):
        for cells in _components(a == structural):
            box = _bbox(cells)
            r0, r1, c0, c1 = box
            h, w = r1 - r0, c1 - c0
            if h < 3 or w < 3 or len(cells) != 2 * h + 2 * w - 4:
                continue
            sub = a[_slice(box)]
            if not (np.all(sub[0] == structural) and np.all(sub[-1] == structural)
                    and np.all(sub[:, 0] == structural) and np.all(sub[:, -1] == structural)):
                continue
            inside = sub[1:-1, 1:-1]
            if (2 <= inside.shape[0] <= 15 and 1 <= inside.shape[1] <= 4
                    and np.all(inside != 0) and np.all(inside != structural)
                    and np.all(inside == inside[:, :1]) and inside[-1, 0] == inside[-2, 0]):
                codes.append((structural, box, inside[:, 0].copy()))
            if min(inside.shape) >= 6 and np.count_nonzero(inside == 0) > inside.size // 2:
                boards.append((structural, box))
    matches = [(s, cb, code, bb) for s, cb, code in codes for t, bb in boards
               if s == t and min(bb[1] - bb[0] - 2, bb[3] - bb[2] - 2) == 2 * (len(code) - 1)]
    _require(len(matches) == 1, "need one unambiguous stripe code and arena")
    structural, code_box, code, board_box = matches[0]
    r0, r1, c0, c1 = board_box
    arena = a[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
    materials = set(map(int, code))
    _require(set(map(int, np.unique(arena))) <= materials | {0}, "unexplained arena colors")
    remaining = a != 0
    remaining[_slice(code_box)] = False
    remaining[_slice(board_box)] = False
    keys = {}
    for cells in _components(remaining, diagonal=True):
        tips = [p for p in cells if a[p] != structural]
        _require(len(tips) == 1 and len(cells) >= 3, "an arrow needs one colored tip and >=2 tail cells")
        tip = tips[0]
        color = int(a[tip])
        tails = set(cells) - {tip}
        possibilities = []
        for dr, dc in DIRECTIONS:
            wanted = {(tip[0] - j * dr, tip[1] - j * dc) for j in range(1, len(tails) + 1)}
            if tails == wanted:
                possibilities.append((dr, dc))
        _require(len(possibilities) == 1 and color in materials and color not in keys,
                 "arrow color/direction must be unique and occur in the code")
        keys[color] = dict(direction=possibilities[0], tip=tip, length=len(tails), box=_bbox(cells))
    marker_colors = set(map(int, np.unique(arena))) - {0}
    _require(marker_colors == set(keys) and keys, "each marker color needs exactly one arrow")
    return dict(grid=a, structural=structural, code=code, code_box=code_box,
                board_box=board_box, arena=arena, keys=keys)


def _depths(shape):
    h, w = shape
    r, c = np.indices(shape)
    return np.minimum.reduce((r, c, h - 1 - r, w - 1 - c))


def _trace(layers, start, color, direction, *, corner=True, frame_only=False):
    """Return a straight path; material equality, rather than occupancy, blocks."""
    h, w = layers.shape
    dr, dc = direction

    def blocked(r, c):
        return not (0 <= r < h and 0 <= c < w) or (not frame_only and layers[r, c] == color)

    r, c = start
    path = [(r, c)]
    while True:
        nr, nc = r + dr, c + dc
        if blocked(nr, nc) or (corner and dr and dc and (blocked(nr, c) or blocked(r, nc))):
            return path
        r, c = nr, nc
        path.append((r, c))


def _predict(info, *, code=None, reverse=False, corner=True, frame_only=False, check=True):
    a = info['grid']
    code = info['code'][:-1] if code is None else np.asarray(code)
    layers = code[_depths(info['arena'].shape)]
    result = a.copy()
    r0, r1, c0, c1 = info['board_box']
    output_arena = result[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
    output_arena[:] = 0
    paths, used = [], set()
    for rr, cc in np.argwhere(info['arena'] != 0):
        start = int(rr), int(cc)
        color = int(info['arena'][start])
        if check:
            _require(layers[start] != color, "a marker cannot start in its own invisible wall")
        direction = info['keys'][color]['direction']
        if reverse:
            direction = (-direction[0], -direction[1])
        path = _trace(layers, start, color, direction, corner=corner, frame_only=frame_only)
        if check:
            _require(used.isdisjoint(path), "marker paths must not intersect")
        used.update(path)
        output_arena[path[-1]] = color
        paths.append((color, direction, path))
    return result, paths


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode this input's code, arena and keys; move all markers simultaneously."""
    info = _read(input_grid)
    return _predict(info)[0]


def _alternative(info, name):
    """Deliberately simpler hypotheses, used only to check training evidence."""
    if name == 'physical_frame_only':
        return _predict(info, frame_only=True, check=False)[0]
    if name == 'reverse_arrow':
        return _predict(info, reverse=True, check=False)[0]
    if name == 'next_diagonal_cell_only':
        return _predict(info, corner=False, check=False)[0]
    layers = info['code'][:-1]
    if name == 'reverse_depth_order':
        changed = layers[::-1]
    elif name == 'compress_runs':
        runs = layers[np.r_[True, layers[1:] != layers[:-1]]]
        changed = np.pad(runs, (0, len(layers) - len(runs)), mode='edge')
    elif name == 'horizontal_stripes':
        # Replace nearest-edge depth by top/bottom row depth only.
        h, w = info['arena'].shape
        row_depth = np.minimum(np.arange(h), np.arange(h)[::-1])
        row_depth = np.minimum(row_depth, len(layers) - 1)
        field = np.repeat(layers[row_depth, None], w, axis=1)
        result = info['grid'].copy()
        r0, r1, c0, c1 = info['board_box']
        out = result[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
        out[:] = 0
        for rr, cc in np.argwhere(info['arena'] != 0):
            start = int(rr), int(cc)
            color = int(info['arena'][start])
            path = _trace(field, start, color, info['keys'][color]['direction'])
            out[path[-1]] = color
        return result
    else:
        raise ValueError(name)
    return _predict(info, code=changed, check=False)[0]


ALTERNATIVES = ('physical_frame_only', 'reverse_arrow', 'next_diagonal_cell_only',
                'reverse_depth_order', 'compress_runs', 'horizontal_stripes')


def _arrow(structural, color, direction, length):
    dr, dc = direction
    points = [(j * dr, j * dc) for j in range(length + 1)]
    box = _bbox(points)
    a = np.zeros((box[1] - box[0], box[3] - box[2]), np.int8)
    for j, (r, c) in enumerate(points):
        a[r - box[0], c - box[2]] = color if j == length else structural
    return a


def _pack(rng, objects):
    """Reserve whole object boxes and a visible one-cell separating gap."""
    occupied = np.zeros((30, 30), bool)
    placements = []
    for obj in objects:
        h, w = obj.shape
        padded = np.pad(occupied, 1)
        obstacles = occupied.copy()
        for dr, dc in DIRECTIONS:
            obstacles |= padded[1 + dr:31 + dr, 1 + dc:31 + dc]
        windows = np.lib.stride_tricks.sliding_window_view(obstacles, (h, w))
        available = np.argwhere(~np.any(windows, axis=(-1, -2)))
        if not len(available):
            return None
        r, c = map(int, available[int(rng.integers(len(available)))])
        occupied[r:r + h, c:c + w] = True
        placements.append((r, c))
    return placements


def _sample_code(rng, length, palette, active, corner_color=None):
    k = int(rng.integers(max(2, len(active)), min(len(palette), length, 5) + 1))
    others = [c for c in palette if c not in active]
    colors = active + list(map(int, rng.choice(others, k - len(active), replace=False)))
    layers = np.empty(length, np.int8)
    for i in range(length):
        layers[i] = layers[i - 1] if i and rng.random() < .30 else int(rng.choice(colors))
    if corner_color is not None:
        # First C shell is q>=3: a diagonal can move toward its corner through
        # repeated transparent outer layers before a side cell, alone, blocks.
        q = int(rng.integers(3, length))
        non_c = [c for c in colors if c != corner_color]
        layers[:q] = rng.choice(non_c, q)
        layers[1] = layers[0]
        layers[q] = corner_color
        slots = [i for i in range(length) if i not in (1, q)]
        rng.shuffle(slots)
        for color, slot in zip(active[1:], slots):
            layers[slot] = color
        layers[1] = layers[0]
    else:
        # Preserve palette coverage without demanding recurrence in every pair.
        slots = list(map(int, rng.permutation(length)))
        for color, slot in zip(colors, slots):
            layers[slot] = color
    return np.append(layers, layers[-1]).astype(np.int8)


def _corner_witness(layers, color, direction, path):
    r, c = path[-1]
    nr, nc = r + direction[0], c + direction[1]
    return bool(0 not in direction and 0 <= nr < layers.shape[0] and 0 <= nc < layers.shape[1]
                and layers[nr, nc] != color
                and (layers[nr, c] == color or layers[r, nc] == color))


def _choose_paths(rng, layers, active, directions, witness_color=None):
    selected, used = [], set()
    for color in active:
        direction = directions[color]
        candidates = []
        for rr, cc in np.argwhere(layers != color):
            start = int(rr), int(cc)
            path = _trace(layers, start, color, direction)
            if len(path) > 1:
                candidates.append(path)
        priority = [p for p in candidates if color != witness_color or _corner_witness(layers, color, direction, p)]
        rng.shuffle(priority)
        possible = [p for p in priority if used.isdisjoint(p)]
        if not possible:
            return None
        first = possible[0]
        selected.append((color, first))
        used.update(first)
        desired = int(rng.choice([1, 1, 2, 2, 3]))
        if desired == 1:
            continue
        rng.shuffle(candidates)
        for path in candidates:
            if used.isdisjoint(path):
                selected.append((color, path))
                used.update(path)
                desired -= 1
                if desired <= 1:
                    break
    return selected


def _make_pair(rng, structural, palette, length, stretch, active_count,
               common, witness=False, cardinal=False, excluded_width=None, informative=False):
    for attempt in range(100):
        active = common[:min(2, active_count)]
        active += list(map(int, rng.choice([c for c in palette if c not in active],
                                          max(0, active_count - 2), replace=False)))
        if witness:
            dirs = {active[0]: DIAGONAL[int(rng.integers(4))]}
        elif cardinal:
            dirs = {active[0]: CARDINAL[int(rng.integers(4))]}
        else:
            dirs = {active[0]: DIRECTIONS[int(rng.integers(8))]}
        dirs.update({c: DIRECTIONS[int(rng.integers(8))] for c in active[1:]})
        code = _sample_code(rng, length, palette, active, active[0] if witness else None)
        inner_shape = (2 * length, 2 * length + stretch)
        if rng.integers(2):
            inner_shape = inner_shape[::-1]
        layers = code[:-1][_depths(inner_shape)]
        paths = _choose_paths(rng, layers, active, dirs, active[0] if witness else None)
        if paths is None:
            continue
        # Ground truth is constructed from selected paths, not from solve().
        board = np.full((inner_shape[0] + 2, inner_shape[1] + 2), structural, np.int8)
        board[1:-1, 1:-1] = 0
        moved = board.copy()
        for color, path in paths:
            board[path[0][0] + 1, path[0][1] + 1] = color
            moved[path[-1][0] + 1, path[-1][1] + 1] = color
        widths = [w for w in range(1, 5) if w != excluded_width]
        width = widths[0] if attempt >= 80 else int(rng.choice(widths))
        stripe = np.full((len(code) + 2, width + 2), structural, np.int8)
        stripe[1:-1, 1:-1] = code[:, None]
        arrows = [_arrow(structural, c, dirs[c], int(rng.integers(2, 5))) for c in active]
        rng.shuffle(arrows)
        objects = [board, stripe] + arrows
        positions = _pack(rng, objects)
        if positions is None:
            continue
        a, b = np.zeros((30, 30), np.int8), np.zeros((30, 30), np.int8)
        for i, (obj, (r, c)) in enumerate(zip(objects, positions)):
            sl = slice(r, r + obj.shape[0]), slice(c, c + obj.shape[1])
            a[sl] = obj
            b[sl] = moved if i == 0 else obj
        if rng.random() < .85:
            r0, r1, c0, c1 = _bbox(list(map(tuple, np.argwhere(a != 0))))
            r0, c0 = max(0, r0 - int(rng.integers(3))), max(0, c0 - int(rng.integers(3)))
            r1, c1 = min(30, r1 + int(rng.integers(3))), min(30, c1 + int(rng.integers(3)))
            a, b = a[r0:r1, c0:c1].copy(), b[r0:r1, c0:c1].copy()
        info = _read(a)
        # Informative demonstrations independently expose order, width of runs,
        # invisible internal stopping, arrow polarity, and nearest-edge geometry.
        needed = ['physical_frame_only', 'reverse_arrow', 'reverse_depth_order', 'horizontal_stripes'] if informative else []
        if witness:
            needed += ['next_diagonal_cell_only', 'compress_runs']
        if all(not np.array_equal(_alternative(info, name), b) for name in needed):
            return dict(input=a, output=b)
    raise AssertionError('bounded construction could not find informative geometry')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _fingerprint(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    structural = int(rng.integers(1, 10))
    palette = list(map(int, rng.choice([c for c in range(1, 10) if c != structural],
                                     int(rng.integers(4, 8)), replace=False)))
    common = list(map(int, rng.choice(palette, 2, replace=False)))
    shift = int(rng.integers(3))
    puzzle, seen = dict(train=[], test=[]), set()
    first_length, first_width = None, None
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            if shift == 0:
                length = int(rng.integers(4, 8) if split == 'train' else rng.integers(8, 11))
                stretch = int(rng.integers(7))
            elif shift == 1:
                length = int(rng.integers(4, 10))
                stretch = int(rng.integers(3) if split == 'train' else rng.integers(4, 7))
            else:
                length = int(rng.integers(4, 10))
                stretch = int(rng.integers(7))
            if split == 'train' and i == 1:
                allowed = range(4, 8) if shift == 0 else range(4, 10)
                length = int(rng.choice([n for n in allowed if n != first_length]))
                if shift == 1:
                    stretch = int(rng.integers(1, 3))
            active_count = 2 if shift == 2 and split == 'train' else int(rng.integers(1, min(4, length) + 1))
            if split == 'train' and i < 2:
                active_count = max(2, active_count)
            if shift == 2 and split == 'test':
                active_count = int(rng.integers(3, 5))
            witness = split == 'train' and i == 1
            if witness:
                active_count = min(active_count, length - 1)
            for _ in range(10):
                pair = _make_pair(rng, structural, palette, length, stretch, active_count, common,
                                  witness=witness, cardinal=split == 'train' and i == 0,
                                  excluded_width=first_width if witness else None,
                                  informative=split == 'train' and i < 2)
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    break
            _require(fp not in seen, 'duplicate input construction')
            seen.add(fp)
            puzzle[split].append(pair)
            if split == 'train' and i == 0:
                info = _read(pair['input'])
                first_length = len(info['code']) - 1
                first_width = info['code_box'][3] - info['code_box'][2] - 2
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'illegal counts')
    seen, structural_roles, evidence, training_directions = set(), set(), set(), set()
    train_lengths, train_widths = set(), set()
    color_directions = {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and not np.array_equal(a, b), 'every pair must show motion at fixed size')
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _read(a)
            structural_roles.add(info['structural'])
            expected, paths = _predict(info)
            _require(np.array_equal(expected, b), 'output violates the decoded rule')
            r0, r1, c0, c1 = info['board_box']
            arena_out = b[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
            _require(Counter(map(int, info['arena'][info['arena'] != 0]))
                     == Counter(map(int, arena_out[arena_out != 0])), 'marker multiset changed')
            fixed = np.ones(a.shape, bool)
            fixed[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = False
            _require(np.array_equal(a[fixed], b[fixed]), 'code, frame, keys or outside background changed')
            layers = info['code'][:-1][_depths(info['arena'].shape)]
            for color, direction, path in paths:
                _require(all(layers[p] != color for p in path), 'a ray enters its selective wall')
                endpoint = path[-1]
                _require(_trace(layers, endpoint, color, direction) == [endpoint], 'nonmaximal ray endpoint')
            if split == 'train':
                train_lengths.add(len(info['code']) - 1)
                train_widths.add(info['code_box'][3] - info['code_box'][2] - 2)
                evidence.update(name for name in ALTERNATIVES if not np.array_equal(_alternative(info, name), b))
                for color, key in info['keys'].items():
                    training_directions.add(key['direction'])
                for color, direction, path in paths:
                    if len(path) > 1:
                        color_directions.setdefault(color, set()).add(direction)
    _require(len(structural_roles) == 1, 'frame/tail convention changed within puzzle')
    _require(len(train_lengths) >= 2 and len(train_widths) >= 2,
             'training must vary code length and irrelevant stripe width')
    _require(evidence == set(ALTERNATIVES), 'training leaves a documented shortcut indistinguishable')
    _require(any(0 in d for d in training_directions) and any(0 not in d for d in training_directions),
             'training must show cardinal and diagonal motion')
    _require(any(len(ds) >= 2 for ds in color_directions.values()), 'keys must override fixed directions by color')


def _signature(puzzle):
    return tuple(tuple((_fingerprint(p['input']), _fingerprint(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _features(pair):
    info = _read(pair['input'])
    _, paths = _predict(info)
    layers = info['code'][:-1][_depths(info['arena'].shape)]
    roles = {}
    normalized = tuple(roles.setdefault(int(c), len(roles)) for c in info['code'][:-1])
    return dict(shape=pair['input'].shape, arena=info['arena'].shape, length=len(info['code']) - 1,
                stretch=abs(info['arena'].shape[0] - info['arena'].shape[1]),
                code_width=info['code_box'][3] - info['code_box'][2] - 2,
                active=len(info['keys']), markers=len(paths), normalized=normalized,
                directions=tuple(k['direction'] for k in info['keys'].values()),
                arrow_lengths=tuple(k['length'] for k in info['keys'].values()),
                repeat=bool(np.any(info['code'][1:-1] == info['code'][:-2])),
                return_color=len(np.unique(info['code'])) < len(info['code'][:-1][np.r_[True, info['code'][1:-1] != info['code'][:-2]]]),
                corner=sum(_corner_witness(layers, color, direction, path) for color, direction, path in paths),
                moves=tuple(len(path) - 1 for _, _, path in paths))


def self_test():
    """Bounded content and variation checks over 42 complete seeded puzzles."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [_features(pair) for puzzle in puzzles for s in ('train', 'test') for pair in puzzle[s]]
    _require(len({_signature(p) for p in puzzles}) == 42, 'seed diversity')
    _require(len({r['shape'] for r in rows}) >= 35, 'canvas shape variation')
    _require(len({r['arena'] for r in rows}) >= 35, 'arena shape variation')
    _require({r['length'] for r in rows} == set(range(4, 11)), 'depth range coverage')
    _require({r['stretch'] for r in rows} == set(range(7)), 'rectangular layer coverage')
    _require({r['code_width'] for r in rows} == set(range(1, 5)), 'code-width variation')
    _require({r['active'] for r in rows} == {1, 2, 3, 4}, 'active color counts')
    _require({d for r in rows for d in r['directions']} == set(DIRECTIONS), 'eight directions')
    _require({n for r in rows for n in r['arrow_lengths']} == {2, 3, 4}, 'tail-length variation')
    _require(len({r['normalized'] for r in rows}) >= 90, 'layer-sequence variation')
    _require(sum(r['repeat'] for r in rows) >= 50 and sum(r['return_color'] for r in rows) >= 25,
             'thick and recurrent materials')
    _require(sum(r['corner'] for r in rows) >= 42, 'corner-block evidence')
    _require(1 in {n for r in rows for n in r['moves']} and max(n for r in rows for n in r['moves']) >= 8,
             'short and long moves')
    _require({int(c) for p in puzzles for s in ('train', 'test') for pair in p[s]
              for c in np.unique(pair['input'])} == set(range(10)), 'literal palette variation')
    _require(_signature(generate_puzzle(987654, 7, 3)) == _signature(generate_puzzle(987654, 7, 3)),
             'nondeterminism')
    # Hand-derived corner fixture: (3,2) is transparent, but side cell (3,3)
    # is material 7. Testing only diagonal destinations wrongly reaches (5,0).
    layers = np.array([2, 2, 3, 7])[_depths((8, 8))]
    _require(_trace(layers, (0, 5), 7, (1, -1)) == [(0, 5), (1, 4), (2, 3)],
             'hand-derived selective corner fixture')
    _require(_trace(layers, (0, 5), 7, (1, -1), corner=False)[-1] == (5, 0),
             'fixture must distinguish next-cell-only motion')
    # Compatible layouts outside the sampler's L=4..10 range, plus a blocked
    # marker. Expected endpoints below are literal coordinate assertions.
    for length in (3, 4, 11, 12):
        a = np.zeros((30, 30), np.int8)
        side = 2 * length
        board = a[:side + 2, :side + 2]
        board[0] = board[-1] = 5
        board[:, 0] = board[:, -1] = 5
        effective = [2, 7, 3] if length == 3 else [2, 2, 3, 7] + [2] * (length - 4)
        code = effective + [effective[-1]]
        plaque = a[:length + 3, side + 3:side + 6]
        plaque[:] = 5
        plaque[1:-1, 1] = code
        direction = (0, 1) if length == 3 else (1, -1)
        start = (0, 1) if length == 3 else (0, 5)
        end = (0, 5) if length == 3 else (2, 3)
        arrow = _arrow(5, 7, direction, 2)
        a[length + 5:length + 5 + arrow.shape[0], side + 3:side + 3 + arrow.shape[1]] = arrow
        a[start[0] + 1, start[1] + 1] = 7
        expected = a.copy()
        expected[start[0] + 1, start[1] + 1] = 0
        expected[end[0] + 1, end[1] + 1] = 7
        _require(np.array_equal(solve(a), expected), 'compatible depth fixture failed')
        if length == 4:
            _require(np.array_equal(solve(expected), expected), 'already-blocked particle must stay put')
    for i, (nt, nq) in enumerate(((2, 1), (3, 2), (5, 4), (10, 1), (10, 4))):
        validate_generated_puzzle(generate_puzzle(2000 + i, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted an unmoved output')
    return dict(seeds=42, pairs=len(rows), canvas_shapes=len({r['shape'] for r in rows}),
                arena_shapes=len({r['arena'] for r in rows}),
                layer_sequences=len({r['normalized'] for r in rows}),
                corner_witnesses=sum(r['corner'] for r in rows))
