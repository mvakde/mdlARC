"""Wind-pressed jigsaw: keyed, simultaneous rigid-polyomino settling.

Only standard library and NumPy. No file access, retained examples, RNG state in
solve, execution traces, or imported parent generators. Generation and self-test
keep bounded arrays in memory and write no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np

_CARDINAL = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _CARDINAL:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _box(cells):
    return cells.min(axis=0), cells.max(axis=0) + 1


def _find_key(a, bg):
    keys = []
    for cells in _components(a != bg):
        lo, hi = _box(cells)
        h, w = hi - lo
        if min(h, w) < 5 or len(cells) != h * w:
            continue
        s = a[lo[0]:hi[0], lo[1]:hi[1]]
        carrier = int(s[0, 0])
        border = np.concatenate((s[0], s[-1], s[:, 0], s[:, -1]))
        other = set(map(int, np.unique(s))) - {carrier}
        if len(other) != 2 or np.any(border != carrier):
            continue
        readings = []
        for tip in sorted(other):
            trail = next(iter(other - {tip}))
            tips = set(map(tuple, np.argwhere(s == tip)))
            trails = set(map(tuple, np.argwhere(s == trail)))
            if len(tips) < 2:
                continue
            for direction in _CARDINAL:
                dr, dc = direction
                used, lengths = set(), []
                for r, c in sorted(tips):
                    stroke = []
                    r, c = r - dr, c - dc
                    while (r, c) in trails:
                        stroke.append((r, c))
                        r, c = r - dr, c - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    readings.append((direction, trail, tip, tuple(lengths)))
        if len(readings) == 1:
            direction, trail, tip, lengths = readings[0]
            keys.append(dict(lo=lo, hi=hi, carrier=carrier, trail=trail,
                             tip=tip, direction=direction, lengths=lengths))
    _require(len(keys) == 1, "expected one uniquely decoded legend")
    return keys[0]


def _read(a):
    histogram = np.bincount(a.ravel(), minlength=10)
    bg = int(histogram.argmax())
    _require(np.count_nonzero(histogram == histogram[bg]) == 1, "background is not uniquely dominant")
    key = _find_key(a, bg)
    key_mask = np.zeros(a.shape, bool)
    lo, hi = key['lo'], key['hi']
    key_mask[lo[0]:hi[0], lo[1]:hi[1]] = True
    walls = (a == key['carrier']) & ~key_mask
    _require(np.any(walls), "missing chamber")
    payload = (a != bg) & ~walls & ~key_mask
    _require(not np.any(payload & np.isin(a, [key['trail'], key['tip']])), "key colors reused by pieces")
    pieces = []
    for color in sorted(map(int, np.unique(a[payload]))):
        for cells in _components(payload & (a == color)):
            pieces.append((cells, color))
    _require(len(pieces) >= 2, "missing jigsaw pieces")
    enclosed = []
    for cells in _components(~walls):
        touches_edge = np.any((cells == 0) | (cells == np.array(a.shape) - 1))
        if not touches_edge and np.any(payload[cells[:, 0], cells[:, 1]]):
            enclosed.append(set(map(tuple, cells)))
    _require(len(enclosed) == 1 and all(set(map(tuple, p)) <= enclosed[0] for p, _ in pieces),
             "pieces must share one closed chamber")
    return dict(background=bg, key=key, key_mask=key_mask, walls=walls,
                fixed=walls | key_mask, payload=payload, pieces=pieces,
                direction=key['direction'])


def _settle(shape, fixed, pieces, direction):
    """Maximal simultaneous movement, with dependencies on vacated cells.

    A dependency is blocked if it leads to a wall or to another blocked piece.
    Computing its closure permits touching pieces to move together; arbitrary
    component enumeration order cannot change the result.
    """
    positions = [p.copy() for p, _ in pieces]
    distances = np.zeros(len(pieces), dtype=int)
    delta = np.array(direction, dtype=int)
    height, width = shape
    # Every successful round increases the bounded sum of cell projections.
    limit = len(pieces) * (height + width) + 1
    for _ in range(limit):
        owner = np.full(shape, -1, dtype=int)
        owner[fixed] = -2
        for i, cells in enumerate(positions):
            _require(np.all(owner[cells[:, 0], cells[:, 1]] == -1), "overlapping bodies")
            owner[cells[:, 0], cells[:, 1]] = i
        movable = set(range(len(pieces)))
        dependencies = []
        for i, cells in enumerate(positions):
            dest = cells + delta
            inside = np.all((dest >= 0) & (dest < np.array(shape)))
            if not inside:
                movable.discard(i)
                dependencies.append(set())
                continue
            ahead = set(map(int, owner[dest[:, 0], dest[:, 1]]))
            if -2 in ahead:
                movable.discard(i)
            dependencies.append(ahead - {-2, -1, i})
        while True:
            blocked = {i for i in movable if not dependencies[i] <= movable}
            if not blocked:
                break
            movable.difference_update(blocked)
        if not movable:
            return positions, distances
        for i in sorted(movable):
            positions[i] += delta
            distances[i] += 1
    raise ValueError("settling did not terminate within the projection bound")


def _answer(a, info, reverse=False):
    direction = tuple(-x for x in info['direction']) if reverse else info['direction']
    positions, distances = _settle(a.shape, info['fixed'], info['pieces'], direction)
    out = a.copy()
    out[info['payload']] = info['background']
    for cells, (_, color) in zip(positions, info['pieces']):
        out[cells[:, 0], cells[:, 1]] = color
    return out, positions, distances


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible roles and simulate rigid bodies; input only, no metadata."""
    a = _grid(input_grid)
    return _answer(a, _read(a))[0]


def _unit_gravity(a, info):
    """Counterfactual: preserve color order but let cells fall independently."""
    turns = {(1, 0): 0, (0, 1): 3, (-1, 0): 2, (0, -1): 1}[info['direction']]
    s = np.rot90(a, turns)
    fixed = np.rot90(info['fixed'], turns)
    payload = np.rot90(info['payload'], turns)
    out = s.copy()
    out[payload] = info['background']
    for c in range(s.shape[1]):
        stops = [-1] + list(map(int, np.flatnonzero(fixed[:, c]))) + [s.shape[0]]
        for start, stop in zip(stops, stops[1:]):
            rows = np.flatnonzero(payload[start + 1:stop, c]) + start + 1
            if len(rows):
                out[stop - len(rows):stop, c] = s[rows, c]
    return np.rot90(out, -turns).copy()


def _contacts(positions, direction):
    lookup = {tuple(p): i for i, cells in enumerate(positions) for p in cells}
    delta = np.array(direction)
    return [{lookup[tuple(q)] for q in cells + delta
             if tuple(q) in lookup and lookup[tuple(q)] != i}
            for i, cells in enumerate(positions)]


def _interpenetrating(positions):
    for i, a in enumerate(positions):
        alo, ahi = _box(a)
        for b in positions[:i]:
            blo, bhi = _box(b)
            if np.all(np.maximum(alo, blo) < np.minimum(ahi, bhi)):
                return True
    return False


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, "changed dimensions")
    info = _read(a)
    expected, positions, distances = _answer(a, info)
    _require(np.array_equal(expected, b), "incorrect rigid-body endpoint")
    _require(np.all(distances > 0) and len(set(map(int, distances))) >= 2,
             "must show distinct positive piece travel distances")
    _require(np.array_equal(a[info['fixed']], b[info['fixed']]), "wall or key changed")
    _require(np.array_equal(np.bincount(a.ravel(), minlength=10), np.bincount(b.ravel(), minlength=10)),
             "color area changed")
    all_cells = np.concatenate(positions)
    lo, hi = _box(all_cells)
    _require(len(all_cells) == int(np.prod(hi - lo)), "settled mosaic is not a solid rectangle")
    _require(_interpenetrating(positions), "missing complementary interlock")
    for cells, (source, color), distance in zip(positions, info['pieces'], distances):
        _require(np.array_equal(cells, source + np.array(info['direction']) * distance), "changed piece mask")
        _require(np.all(b[cells[:, 0], cells[:, 1]] == color), "changed piece color")
    _require(not np.array_equal(_unit_gravity(a, info), b), "rigidity is redundant")
    reverse_out, reverse_positions, reverse_distances = _answer(a, info, reverse=True)
    _require(not np.array_equal(reverse_out, b), "legend polarity is redundant")
    reverse_cells = np.concatenate(reverse_positions)
    reverse_lo, reverse_hi = _box(reverse_cells)
    _require(len(reverse_cells) == int(np.prod(reverse_hi - reverse_lo))
             and np.all(reverse_distances > 0), "the other end must also accept the mosaic")
    reverse_info = dict(info, direction=tuple(-v for v in info['direction']))
    _require(not np.array_equal(_unit_gravity(a, reverse_info), reverse_out), "other-end rigidity is redundant")
    contacts = _contacts(positions, info['direction'])
    info.update(positions=positions, distances=distances, contacts=contacts,
                branching=any(len(x) >= 2 for x in contacts), mosaic_shape=tuple(map(int, hi - lo)))
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            a = _grid(pair['input'])
            fingerprint = (a.shape, a.tobytes())
            _require(fingerprint not in seen, "duplicate input")
            seen.add(fingerprint)
            if split == 'train':
                training.append(info)
    _require(len({i['direction'] for i in training}) >= 2, "missing direction-controlled training contrast")
    _require(any(i['branching'] for i in training), "missing demonstrated branching contact")
    _require(any(any(c == d and i['distances'][x] != i['distances'][y]
                     for x, (_, c) in enumerate(i['pieces'])
                     for y, (_, d) in enumerate(i['pieces']) if x < y)
                 for i in training), "missing shared-color components with different movements")


def _profile(rng, width):
    mode = int(rng.integers(4))
    if mode == 0:
        result = rng.integers(0, 3, size=width)
    elif mode == 1:
        result = np.floor(np.linspace(0, 2, width)).astype(int)
        if rng.random() < .5:
            result = result[::-1].copy()
    elif mode == 2:
        result = (np.arange(width) + int(rng.integers(3))) % 3
    else:
        result = np.zeros(width, int)
        result[int(rng.integers(1, width)):] = 2
    if result.min() == result.max():
        result[0], result[-1] = 0, 2
    return result


def _partition(rng, width, count):
    if count == 1:
        return [(0, width)]
    # Enumerate at most 36 possible cuts; no rejection loop or large payload.
    if count == 2:
        cuts = [int(rng.integers(2, width - 1))]
    else:
        choices = [(x, y) for x in range(2, width - 3) for y in range(x + 2, width - 1)]
        cuts = list(choices[int(rng.integers(len(choices)))])
    edges = [0] + cuts + [width]
    return list(zip(edges, edges[1:]))


def _make_key(rng, carrier, trail, tip):
    # Canonical downward legend; whole-scene rotation supplies the direction.
    length = int(rng.integers(2, 4))
    spacing = int(rng.integers(2, 4))
    top, bottom, left, right = map(int, rng.integers(1, 3, size=4))
    key = np.full((top + length + 1 + bottom, left + spacing + 1 + right), carrier, np.int8)
    for c in (left, left + spacing):
        key[top:top + length, c] = trail
        key[top + length, c] = tip
    return key


def _make_pair(rng, layers, rotation, palette, witness=0, canonical_up=None):
    bg, carrier, trail, tip = map(int, palette[:4])
    colors = list(map(int, palette[4:]))
    wind_up = bool(rng.integers(2)) if canonical_up is None else bool(canonical_up)
    width = int(rng.integers(5, 11))
    boundaries = [np.zeros(width, int)]
    boundaries += [3 * j + _profile(rng, width) for j in range(1, layers)]
    depth = 3 * layers + int(rng.integers(2))
    boundaries.append(np.full(width, depth, int))
    chunks = []
    for j in range(layers):
        max_chunks = min(3, width // 2)
        count = int(rng.choice(np.arange(1, max_chunks + 1), p=([.45, .55] if max_chunks == 2 else [.4, .5, .1])))
        if witness == 1:
            count = 1 if j == (layers - 1 if wind_up else 0) else 2
        elif witness == 2:
            count = 1 if j == 1 else 2
        chunks.append(_partition(rng, width, count))
    layer_offsets = [0] * layers
    layer_offsets[-1] = int(rng.integers(1, 3))
    for j in range(layers - 2, -1, -1):
        # Four-layer scenes need smaller separations to fit both receiving ends.
        layer_offsets[j] = layer_offsets[j + 1] + (1 if layers == 4 else int(rng.integers(1, 3)))
    records = []
    for j in range(layers):
        for c0, c1 in chunks[j]:
            cells = np.array([(r, c) for c in range(c0, c1)
                              for r in range(int(boundaries[j][c]), int(boundaries[j + 1][c]))], int)
            offset = layer_offsets[j] + int(rng.integers(2))
            records.append((cells, offset, j))
    max_offset = max(record[1] for record in records)
    top = max_offset + 6
    # BOTH ends can receive the same mosaic. Only the arrow selects the end.
    left_clear, right_clear = map(int, rng.integers(1, 4, size=2))
    chamber_width = width + left_clear + right_clear + 2
    x = 1 + left_clear
    dips = rng.integers(0, 3, size=chamber_width)
    upper_dips = rng.integers(0, 3, size=chamber_width)
    for c0, c1 in chunks[-1]:
        support = int(rng.integers(c0, c1))
        recess = int(rng.choice([c for c in range(c0, c1) if c != support]))
        dips[x + support] = 0
        dips[x + recess] = int(rng.integers(1, 3))
    for c0, c1 in chunks[0]:
        support = int(rng.integers(c0, c1))
        recess = int(rng.choice([c for c in range(c0, c1) if c != support]))
        upper_dips[x + support] = 0
        upper_dips[x + recess] = int(rng.integers(1, 3))
    minimum_height = top + depth + int(dips.max()) + 3
    _require(minimum_height <= 30, "chamber height overflow")
    extra = int(rng.integers(min(2, 30 - minimum_height) + 1))
    top += extra
    height = minimum_height + extra
    wall = np.zeros((height, chamber_width), bool)
    wall[:, 0] = wall[:, -1] = True
    # Thick, overlapping floor steps close the chamber but leave an irregular
    # exterior silhouette below them. Unused steps outside the mosaic persist.
    for c in range(1, chamber_width - 1):
        floor = top + depth + int(dips[c])
        wall[floor:floor + 3, c] = True
        ceiling = 4 - int(upper_dips[c])
        wall[ceiling - 2:ceiling + 1, c] = True
    inp = np.full(wall.shape, bg, np.int8)
    inp[wall] = carrier
    out = inp.copy()
    reuse_color = None
    for cells, offset, layer in records:
        source = cells + np.array((top - offset, x))
        target = cells + np.array((5 if wind_up else top, x))
        _require(np.all(inp[source[:, 0], source[:, 1]] == bg), "input construction overlaps")
        adjacent = set()
        for dr, dc in _CARDINAL:
            q = source + np.array((dr, dc))
            adjacent.update(map(int, inp[q[:, 0], q[:, 1]]))
        options = [c for c in colors if c not in adjacent]
        if witness == 2 and layer == 1:
            options = [c for c in options if c != reuse_color]
        _require(options, "payload adjacency needs another coloring")
        color = reuse_color if witness == 2 and layer == layers - 1 and reuse_color in options else int(rng.choice(options))
        if reuse_color is None:
            reuse_color = color
        inp[source[:, 0], source[:, 1]] = color
        out[target[:, 0], target[:, 1]] = color
    key = _make_key(rng, carrier, trail, tip)
    if wind_up:
        key = key[::-1].copy()
        # Keep the planned *visible* vector independent of the sampled end.
        rotation = (rotation + 2) % 4
    gap = int(rng.integers(1, 3))
    total_width = chamber_width + gap + key.shape[1]
    _require(total_width <= 30, "scene width overflow")
    scene = np.full((height, total_width), bg, np.int8)
    result = scene.copy()
    key_on_left = bool(rng.integers(2))
    chamber_x = key.shape[1] + gap if key_on_left else 0
    key_x = 0 if key_on_left else chamber_width + gap
    key_y = int(rng.integers(height - key.shape[0] + 1))
    for grid, chamber in ((scene, inp), (result, out)):
        grid[:, chamber_x:chamber_x + chamber_width] = chamber
        grid[key_y:key_y + key.shape[0], key_x:key_x + key.shape[1]] = key
    if rng.random() < .5:
        scene, result = scene[:, ::-1], result[:, ::-1]
    pair = {'input': np.rot90(scene, rotation).copy(), 'output': np.rot90(result, rotation).copy()}
    info = _analyze_pair(pair)
    _require(not witness or info['branching'], "witness lacks branching contacts")
    return pair


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " is outside its legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.integers(2))
    palette = rng.permutation(10)
    base_rotation = int(rng.integers(4))
    orientation_transfer = bool(rng.integers(2))
    train_rots = [base_rotation, (base_rotation + (2 if orientation_transfer else 1)) % 4]
    first_end = bool(rng.integers(2))
    train_plans = [(2, train_rots[0], 1, first_end), (3, train_rots[1], 2, not first_end)]
    train_plans += [(int(rng.integers(2, 4)), int(rng.choice(train_rots)), 0, None) for _ in range(nt - 2)]
    train_plans = [train_plans[int(i)] for i in rng.permutation(nt)]
    test_rots = [(base_rotation + 1) % 4, (base_rotation + 3) % 4] if orientation_transfer else list(range(4))
    test_plans = [(int(rng.integers(3, 5)), int(rng.choice(test_rots)), 0, None) for _ in range(nq)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, plans in (('train', train_plans), ('test', test_plans)):
        for layers, rotation, witness, canonical_up in plans:
            for _ in range(64):
                local_palette = palette if stable_palette else rng.permutation(10)
                try:
                    pair = _make_pair(rng, layers, rotation, local_palette, witness, canonical_up)
                except ValueError:
                    continue
                fingerprint = (pair['input'].shape, pair['input'].tobytes())
                if fingerprint not in seen:
                    break
            else:
                raise RuntimeError("bounded construction could not find a valid distinct scene")
            seen.add(fingerprint)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming 42-seed audit plus count and adversarial checks."""
    shapes, key_shapes, directions, counts, mosaic_shapes = set(), set(), set(), set(), set()
    normalized, puzzle_hashes, distances, masks, roles = set(), set(), set(), set(), set()
    reused = touching = branching = total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_hashes.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                info = _analyze_pair(pair)
                shapes.add(a.shape)
                key_shapes.add(tuple(map(int, info['key']['hi'] - info['key']['lo'])))
                directions.add(info['direction'])
                counts.add(len(info['pieces']))
                mosaic_shapes.add(info['mosaic_shape'])
                distances.update(map(int, info['distances']))
                roles.add((info['background'], info['key']['carrier']))
                colors = [c for _, c in info['pieces']]
                reused += len(set(colors)) < len(colors)
                branching += info['branching']
                owner = {tuple(p): i for i, (cells, _) in enumerate(info['pieces']) for p in cells}
                touching += any(tuple(p + np.array(d)) in owner and owner[tuple(p + np.array(d))] != i
                                for i, (cells, _) in enumerate(info['pieces']) for p in cells for d in _CARDINAL)
                for cells, _ in info['pieces']:
                    local = cells - cells.min(axis=0)
                    masks.add(hashlib.sha256(local.astype(np.int8).tobytes()).digest())
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                normalized.add((a.shape, hashlib.sha256(norm).digest()))
                # Rotation/reflection equivariance checks use actual query inputs.
                if split == 'test':
                    for k in (1, 2, 3):
                        _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(pair['output'], k)), "rotation failed")
                    _require(np.array_equal(solve(a[:, ::-1]), pair['output'][:, ::-1]), "reflection failed")
                    # Change ONLY the key. Both physical destinations are valid
                    # rectangular interlocks, so geometry cannot replace reading.
                    counter = a.copy()
                    lo, hi = info['key']['lo'], info['key']['hi']
                    glyph = counter[lo[0]:hi[0], lo[1]:hi[1]].copy()
                    glyph = glyph[::-1] if info['direction'][0] else glyph[:, ::-1]
                    counter[lo[0]:hi[0], lo[1]:hi[1]] = glyph
                    reversed_out = _answer(a, info, reverse=True)[0]
                    reversed_out[info['key_mask']] = counter[info['key_mask']]
                    counter_info = _analyze_pair({'input': counter, 'output': reversed_out})
                    _require(counter_info['direction'] == tuple(-v for v in info['direction']), "key reversal failed")
                    _require(not np.array_equal(reversed_out[~info['key_mask']], pair['output'][~info['key_mask']]),
                             "key-only intervention did not change the mosaic")
                total += 1
    _require(len(puzzle_hashes) == 42 and len(normalized) >= .98 * total, "insufficient structural diversity")
    _require(directions == set(_CARDINAL), "missing cardinal orientation")
    _require(len(shapes) >= 35 and len(key_shapes) >= 12 and len(mosaic_shapes) >= 30, "geometry variation too narrow")
    _require(len(counts) >= 5 and max(counts) >= 7 and len(masks) >= 120, "piece variation too narrow")
    _require(len(distances) >= 8 and len(roles) >= 30, "travel/palette variation too narrow")
    _require(reused >= 35 and touching >= 35 and branching >= 80, "missing interacting component variation")
    # Every legal explicit count request is exercised; no puzzles are retained.
    legal_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "count mismatch")
            legal_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A one-pass static-obstacle ray incorrectly freezes these touching bodies.
    fixed = np.zeros((8, 6), bool)
    fixed[-1] = True
    pieces = [(np.array([[1, 2]]), 2), (np.array([[2, 2]]), 3)]
    pos, travel = _settle(fixed.shape, fixed, pieces, (1, 0))
    _require(tuple(travel) == (4, 4) and tuple(pos[0][0]) == (5, 2), "vacated-cell dependency failed")
    return dict(seeds=42, pairs=total, distinct_puzzles=len(puzzle_hashes),
                normalized_structures=len(normalized), canvas_shapes=len(shapes),
                key_shapes=len(key_shapes), mosaic_shapes=len(mosaic_shapes),
                piece_counts=sorted(counts), unique_piece_masks=len(masks),
                directions=sorted(directions), travel_distances=sorted(distances),
                role_assignments=len(roles), reused_color_pairs=reused,
                initially_touching_pairs=touching, branching_pairs=branching,
                legal_count_calls=legal_calls)
