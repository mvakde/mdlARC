"""Census-stretched ribbons: disconnected objects set symbolic run lengths.

No parent generator is imported. Parsing, counting and solving use input pixels
only. Experiments are in-memory; this module never writes files.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors outside 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(grid, bg, excluded=()):
    remaining = set(zip(*np.nonzero(grid != bg))) - set(excluded)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color, cells, stack = int(grid[start]), {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining and int(grid[p]) == color:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append((color, cells))
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return int(min(rr)), int(min(cc)), int(max(rr) - min(rr) + 1), int(max(cc) - min(cc) + 1)


def _rectangle(box):
    r, c, h, w = box
    return {(i, j) for i in range(r, r + h) for j in range(c, c + w)}


def _border(box):
    r, c, h, w = box
    return {(i, j) for i, j in _rectangle(box)
            if i in (r, r + h - 1) or j in (c, c + w - 1)}


def _dilate(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _maximum(shape, box, direction):
    r, c, h, w = box
    return (r, shape[0] - r - h, c, shape[1] - c - w)[direction]


def _slice(box, direction, width, distance):
    r, c, h, w = box
    if direction < 2:
        rr = r - distance if direction == 0 else r + h - 1 + distance
        return [(rr, c + (w - width) // 2 + j) for j in range(width)]
    cc = c - distance if direction == 2 else c + w - 1 + distance
    return [(r + (h - width) // 2 + j, cc) for j in range(width)]


def _expand(word, counts):
    return [int(c) for c in word for _ in range(counts.get(int(c), 0))]


def _inspect(input_grid):
    """Recover hollow hubs, attached programs and the external color census."""
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(int(np.sum(frequencies == frequencies[bg])) == 1, "ambiguous background")
    parts = _components(grid, bg)
    hubs = []
    for color, cells in parts:
        box = _box(cells)
        r, c, h, w = box
        if (5 <= h <= 8 and 5 <= w <= 8 and cells == _border(box)
                and np.all(grid[r + 1:r + h - 1, c + 1:c + w - 1] == bg)):
            hubs.append(dict(box=box, color=color, cells=cells))
    _require(1 <= len(hubs) <= 3, "need one through three hollow hubs")
    _require(len({s['color'] for s in hubs}) == 1, "hub color must be common")
    hc = hubs[0]['color']
    _require(set().union(*(cells for color, cells in parts if color == hc))
             == set().union(*(s['cells'] for s in hubs)), "hub color used externally")
    claimed = set().union(*(s['cells'] for s in hubs))
    hub_boxes = set().union(*(_rectangle(s['box']) for s in hubs))
    paths = set()
    for s in hubs:
        box = s['box']
        choices = []
        for d in range(4):
            maximum = _maximum(grid.shape, box, d)
            if maximum < 1:
                continue
            transverse = box[3] if d < 2 else box[2]
            face = _slice(box, d, transverse, 1)
            active = [i for i, p in enumerate(face) if int(grid[p]) not in (bg, hc)]
            width = len(active)
            off = (transverse - width) // 2
            if (1 <= width <= 3 and off >= 1 and (transverse - width) % 2 == 0
                    and active == list(range(off, off + width))):
                choices.append((d, width, maximum))
        _require(len(choices) == 1, "one uniquely centered program required")
        d, width, maximum = choices[0]
        rows, original = [], set()
        for distance in range(1, maximum + 1):
            cells = _slice(box, d, width, distance)
            values = [int(grid[p]) for p in cells]
            if all(c not in (bg, hc) for c in values):
                rows.append(values)
                original.update(cells)
            else:
                _require(all(c == bg for c in values), "program ends in a partial slice")
                break
        _require(1 <= len(rows) <= 5 and len(rows) < maximum, "program length or missing end gap")
        _require(not (claimed & original), "programs overlap")
        path = {p for i in range(1, maximum + 1) for p in _slice(box, d, width, i)}
        _require(not (path & hub_boxes) and not (path & paths), "ribbons collide")
        claimed.update(original)
        paths.update(path)
        s.update(direction=d, width=width, maximum=maximum,
                 program=np.array(rows, dtype=np.int8), original=original, path=path)
    objects = _components(grid, bg, claimed)
    _require(all(1 <= len(cells) <= 10 and color != hc for color, cells in objects), "object size/color")
    occupied = set(claimed)
    for color, cells in objects:
        _require(not (_dilate(cells) & occupied), "external objects need a background moat")
        occupied.update(cells)
    _require(occupied == set(zip(*np.nonzero(grid != bg))), "unparsed cells")
    counts = Counter(color for color, cells in objects)
    lexicon = {int(c) for s in hubs for c in s['program'].flat}
    _require(all(counts[c] <= 4 for c in lexicon), "category count above four")
    _require(any(counts[c] > 0 for c in lexicon), "at least one active category required")
    return dict(grid=grid, background=bg, hub_color=hc, hubs=hubs,
                objects=objects, counts=counts, lexicon=lexicon, claimed=claimed, paths=paths)


def _render(shape, bg, hc, hubs, counts, mode='correct'):
    output = np.full(shape, bg, dtype=np.int8)
    for s in hubs:
        for p in _border(s['box']):
            output[p] = hc
        for j in range(s['width']):
            word = [int(c) for c in s['program'][:, j]]
            if mode == 'unit':
                expanded = word
            else:
                expanded = _expand(word, counts)
            if not expanded:
                continue
            stop = min(s['maximum'], len(expanded)) if mode == 'once' else s['maximum']
            for distance in range(1, stop + 1):
                p = _slice(s['box'], s['direction'], s['width'], distance)[j]
                output[p] = expanded[(distance - 1) % len(expanded)]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The attached rectangle is a complete symbolic word, read from its hub."""
    a = _inspect(input_grid)
    return _render(a['grid'].shape, a['background'], a['hub_color'], a['hubs'], a['counts'])


def _shape(rng, area):
    """Sample geometry independently of a color's component count."""
    mode = int(rng.integers(4))
    if mode == 0:
        cells = {(0, i) for i in range(area)}
    elif mode == 1:
        w = int(rng.integers(1, min(area, 4) + 1))
        cells = {(i // w, i % w) for i in range(area)}
    elif mode == 2:
        arm = int(rng.integers(1, area + 1))
        cells = {(0, i) for i in range(arm)} | {(i, 0) for i in range(area - arm + 1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRS} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0, _, _ = _box(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _layout(rng, shape, programs, counts, directions=None):
    hubs = []
    for index, program in enumerate(programs):
        width = program.shape[1]
        # Every nonempty lane is seen for more than one complete expanded word.
        need = max([len(_expand(program[:, j], counts)) for j in range(width)] + [len(program)]) + 2
        for _ in range(180):
            d = directions[index] if directions is not None else int(rng.integers(4))
            transverse = int(rng.choice([v for v in range(5, 9) if v >= width + 2 and (v - width) % 2 == 0]))
            longitudinal = int(rng.integers(5, 9))
            h, w = (longitudinal, transverse) if d < 2 else (transverse, longitudinal)
            if h > shape[0] or w > shape[1]:
                continue
            box = (int(rng.integers(shape[0] - h + 1)), int(rng.integers(shape[1] - w + 1)), h, w)
            maximum = _maximum(shape, box, d)
            if maximum < need:
                continue
            original = {p for i in range(1, len(program) + 1) for p in _slice(box, d, width, i)}
            path = {p for i in range(1, maximum + 1) for p in _slice(box, d, width, i)}
            region = _rectangle(box)
            if any((_dilate(region) & (s['region'] | s['path'])
                    or _dilate(path) & (s['region'] | s['path'])) for s in hubs):
                continue
            hubs.append(dict(box=box, direction=d, width=width, program=program,
                             maximum=maximum, original=original, path=path, region=region))
            break
        else:
            return None
    return hubs


def _place_objects(rng, shape, bg, hc, hubs, specifications):
    grid = np.full(shape, bg, dtype=np.int8)
    blocked = np.zeros(shape, dtype=bool)
    for s in hubs:
        for p in _border(s['box']):
            grid[p] = hc
        for i, values in enumerate(s['program'], 1):
            for p, c in zip(_slice(s['box'], s['direction'], s['width'], i), values):
                grid[p] = c
        for r, c in _dilate(_rectangle(s['box']) | s['original']):
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                blocked[r, c] = True
    # Actual cells, free pixel origins, and largest-first packing; no lattice.
    order = sorted(range(len(specifications)), key=lambda i: (-len(specifications[i][1]), float(rng.random())))
    for index in order:
        color, cells = specifications[index]
        _, _, h, w = _box(cells)
        if h > shape[0] or w > shape[1]:
            return None
        legal = np.ones((shape[0] - h + 1, shape[1] - w + 1), dtype=bool)
        for r, c in cells:
            legal &= ~blocked[r:r + legal.shape[0], c:c + legal.shape[1]]
        rr, cc = np.nonzero(legal)
        if not len(rr):
            return None
        j = int(rng.integers(len(rr)))
        placed = {(int(rr[j]) + r, int(cc[j]) + c) for r, c in cells}
        for p in placed:
            grid[p] = color
        for r, c in _dilate(placed):
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                blocked[r, c] = True
    return grid


def _fallback(programs, specifications, bg, hc):
    """Finite escape: preserve all counts/areas/programs; compact only geometry."""
    shape, hubs = (30, 30), []
    for i, program in enumerate(programs):
        width = program.shape[1]
        transverse = 6 if width == 2 else 5
        box = (0, 10 * i, 5, transverse)
        hubs.append(dict(box=box, direction=1, width=width, program=program, maximum=25))
    grid = np.full(shape, bg, dtype=np.int8)
    for s in hubs:
        for p in _border(s['box']):
            grid[p] = hc
        for i, values in enumerate(s['program'], 1):
            for p, c in zip(_slice(s['box'], 1, s['width'], i), values):
                grid[p] = c
    _require(len(specifications) <= 18, "fallback object budget")
    for i, (color, cells) in enumerate(specifications):
        # At most 10 cells in a 3-by-4 box, separated by empty rows/columns.
        for j in range(len(cells)):
            grid[11 + 4 * (i // 6) + j // 4, 5 * (i % 6) + j % 4] = color
    return grid, hubs


def _make_pair(rng, programs, areas, bg, hc, directions=None, force_fallback=False):
    specifications = [(color, _shape(rng, area)) for color, values in areas.items() for area in values]
    counts = {color: len(values) for color, values in areas.items()}
    for attempt in range(8 if not force_fallback else 0):
        low = 13 if len(programs) == 1 else 20 if len(programs) == 2 else 25
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31))) if attempt < 5 else (30, 30)
        hubs = _layout(rng, shape, programs, counts, directions)
        if hubs is None:
            continue
        grid = _place_objects(rng, shape, bg, hc, hubs, specifications)
        if grid is None:
            continue
        freq = np.bincount(grid.ravel().astype(int), minlength=10)
        if int(np.argmax(freq)) == bg and int(np.sum(freq == freq[bg])) == 1:
            break
    else:
        grid, hubs = _fallback(programs, specifications, bg, hc)
    output = _render(grid.shape, bg, hc, hubs, counts)
    turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
    pair = {}
    for role, value in (('input', grid), ('output', output)):
        value = np.rot90(value, turns)
        pair[role] = np.ascontiguousarray(np.fliplr(value) if mirror else value)
    return pair


def _random_pair(rng, regime, test):
    palette = [int(c) for c in rng.permutation(10)]
    bg, hc = palette[:2]
    colors = palette[2:2 + int(rng.integers(2, 5))]
    maximum = 4 if test and regime == 'counts' else 2
    counts = [int(rng.integers(maximum + 1)) for c in colors]
    if not any(counts):
        counts[0] = 1
    if test and regime == 'counts':
        counts[0] = int(rng.integers(3, 5))
    if test:
        # A held-out factor must affect the answer, not just a uniform ribbon's
        # unobservable word length. Every target has two active symbols.
        counts[0] = max(1, counts[0])
        counts[1] = max(1, counts[1])
        if regime == 'lanes':
            counts[:2] = [2, 1]
    while sum(counts) > 12:
        j = max(range(1, len(counts)), key=lambda k: counts[k])
        counts[j] -= 1
    number = 3 if test and regime == 'layout' else int(rng.integers(1, 3))
    programs = []
    for i in range(number):
        width = 3 if test and regime == 'lanes' and i == 0 else int(rng.integers(1, 3))
        length = 5 if test and regime == 'word' and i == 0 else int(rng.integers(1, 5))
        if test and regime == 'lanes' and i == 0:
            length = 3
        if test and (i == 0 or regime == 'layout'):
            length = max(3 if regime == 'lanes' else 2, length)
        subset = rng.choice(colors, int(rng.integers(1, len(colors) + 1)), replace=False)
        program = rng.choice(subset, (length, width)).astype(np.int8)
        # Guarantee an active, and in the count regime extrapolated, symbol.
        program[0, 0] = colors[0] if test and regime == 'counts' else colors[next(j for j, n in enumerate(counts) if n)]
        if test and (i == 0 or regime == 'layout'):
            program[0, 0], program[1, 0] = colors[:2]
        if test and regime == 'lanes' and i == 0:
            program[:3, :] = np.array([[colors[0], colors[1], colors[0]],
                                       [colors[0], colors[0], colors[1]],
                                       [colors[1], colors[1], colors[0]]], dtype=np.int8)
        if test and regime == 'word' and i == 0:
            program[:, 0] = np.roll(np.array([colors[0], colors[1], colors[0], colors[0], colors[1]]),
                                     int(rng.integers(5)))
        programs.append(program)
    areas = {c: [int(rng.integers(1, 11)) for _ in range(n)] for c, n in zip(colors, counts)}
    used = {int(c) for p in programs for c in p.flat}
    unmatched = [c for c in palette[2:] if c not in used and c not in colors]
    for _ in range(int(rng.integers(4))):
        c = int(rng.choice(unmatched))
        areas.setdefault(c, []).append(int(rng.integers(1, 11)))
    return _make_pair(rng, programs, areas, bg, hc)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    seed = _integer(seed, 'seed') % (1 << 64)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('lanes', 'counts', 'word', 'layout')))
    palette = [int(c) for c in rng.permutation(10)]
    a, b, c, z = palette[:4]
    collision_area = int(rng.integers(5, 10))
    split_area = int(rng.integers(1, collision_area))
    d = int(rng.integers(4))
    programs1 = [np.array([[a], [b], [a]], dtype=np.int8),
                 np.array([[b], [a], [a]] if rng.random() < .5 else [[a], [a], [b]], dtype=np.int8)]
    areas1 = {a: [collision_area], b: [1, int(rng.integers(1, 3))], c: [int(rng.integers(2, 8))]}
    first = _make_pair(rng, programs1, areas1, palette[3], palette[4], [d, d ^ 1])
    # Same literal color, same total area, but a different component count.
    # Unequal lane multiplicities now produce different stretched periods.
    lane1 = np.array([a, a, c, z], dtype=np.int8)
    lane2 = np.array([b, a, c, z], dtype=np.int8)
    lane1 = np.roll(lane1, int(rng.integers(4)))
    lane2 = np.roll(lane2, int(rng.integers(4)))
    program2 = np.column_stack((lane1, lane2))
    if rng.random() < .5:
        program2 = program2[:, ::-1].copy()
    areas2 = {a: [split_area, collision_area - split_area], b: [int(rng.integers(3, 9))],
              c: [int(rng.integers(1, 11))]}
    second = _make_pair(rng, [program2], areas2, palette[4], palette[5], [(d + 2) % 4])
    puzzle, seen = {'train': [first, second], 'test': []}, {_key(first['input']), _key(second['input'])}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for _ in range(count):
            for attempt in range(20):
                pair = _random_pair(rng, regime, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded distinct-input retries exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def _features(a):
    rows = []
    for s in a['hubs']:
        lengths = tuple(len(_expand(s['program'][:, j], a['counts'])) for j in range(s['width']))
        rows.append((s['width'], len(s['program']), s['direction'], s['maximum'], lengths))
    return rows


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _inspect(inp)
            expected = _render(inp.shape, a['background'], a['hub_color'], a['hubs'], a['counts'])
            _require(np.array_equal(out, expected), "wrong census, lane phase, erasure or preservation")
            for s in a['hubs']:
                _require(all(out[p] == inp[p] for p in s['cells']), "hub damaged")
            if split == 'train':
                training.append(a)
    # Teaching checks are reconstructed from grids, with no sampler metadata.
    observations = [(c, a['counts'][c], sum(len(cells) for cc, cells in a['objects'] if cc == c))
                    for a in training for c in a['lexicon']]
    _require(any(c == cc and n != nn and area == aa for c, n, area in observations
                 for cc, nn, aa in observations), "missing same-color area collision")
    _require(any(n == nn and area != aa for c, n, area in observations
                 for cc, nn, aa in observations), "missing same-count area variation")
    _require({0, 1, 2} <= {n for c, n, area in observations}, "show zero/one/two census")
    _require(any(s['width'] >= 2 and len(set(lengths)) > 1
                 for a in training for s, (_, _, _, _, lengths) in zip(a['hubs'], _features(a))),
             "show independent lane clocks")
    _require(any(len(a['hubs']) >= 2 for a in training), "show shared census across hubs")
    for mode in ('unit', 'once'):
        _require(any(not np.array_equal(_render(a['grid'].shape, a['background'], a['hub_color'],
                                              a['hubs'], a['counts'], mode), solve(a['grid']))
                     for a in training), "training fails to reject shortcut " + mode)
    _require(any(not np.array_equal(
        _render(a['grid'].shape, a['background'], a['hub_color'], a['hubs'],
                {c: sum(len(cells) for cc, cells in a['objects'] if cc == c) for c in a['lexicon']}),
        solve(a['grid'])) for a in training), "training fails to reject area counting")
    _require(any(any(color not in a['lexicon'] for color, cells in a['objects']) for a in training),
             "show unmatched-color erasure")
    active = {c for a in training for c in a['lexicon'] if a['counts'][c]}
    discarded = {c for a in training for c, cells in a['objects'] if c not in a['lexicon']}
    _require(bool(active & discarded), "show a discarded color becoming active")


def self_test():
    """Bounded 42-seed checks, all counts, and semantic metamorphic checks."""
    signatures, shapes, backgrounds, widths, lengths, counts, directions, masks = (set() for _ in range(8))
    hub_counts, joint, zero_lanes = set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _inspect(pair['input'])
                predicted = solve(pair['input'])
                _require(np.array_equal(predicted, pair['output']), "solve mismatch")
                shapes.add(a['grid'].shape)
                backgrounds.add(a['background'])
                hub_counts.add(len(a['hubs']))
                digest.update(repr(_key(a['grid'])).encode())
                for w, length, d, maximum, periods in _features(a):
                    widths.add(w)
                    lengths.add(length)
                    directions.add(d)
                    joint.add((w, length, maximum, periods))
                    zero_lanes += 0 in periods
                counts.update(a['counts'][c] for c in a['lexicon'])
                for color, cells in a['objects']:
                    r, c, _, _ = _box(cells)
                    masks.add(tuple(sorted((rr - r, cc - c) for rr, cc in cells)))
                # Geometry and color roles, including direction, are local.
                turned = np.rot90(pair['input'])
                _require(np.array_equal(solve(turned), np.rot90(predicted)), "rotation equivariance")
                mapping = np.array([(int(c) + 3) % 10 for c in range(10)], dtype=np.int8)
                _require(np.array_equal(solve(mapping[pair['input']]), mapping[predicted]), "palette equivariance")
        signatures.add(digest.hexdigest())
        repeat = generate_puzzle(seed)
        _require(all(_key(x[k]) == _key(y[k]) for s in puzzle for x, y in zip(puzzle[s], repeat[s])
                     for k in ('input', 'output')), "nondeterministic")
    _require(len(signatures) == 42 and len(shapes) >= 30, "insufficient structural variation")
    _require(backgrounds == set(range(10)) and directions == set(range(4)), "role/direction coverage")
    _require(widths == {1, 2, 3} and lengths == {1, 2, 3, 4, 5}, "program geometry coverage")
    _require(counts == {0, 1, 2, 3, 4} and hub_counts == {1, 2, 3}, "census/layout coverage")
    _require(len(masks) >= 75 and len(joint) >= 100, "geometry/clock diversity")
    _require(zero_lanes > 0, "no empty lane cases")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(103, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (4.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Exercise the finite packer escape, including the widest/longest programs.
    programs = [np.array([[2, 3, 4], [3, 2, 4], [2, 4, 3], [3, 2, 4], [2, 3, 4]], dtype=np.int8)] * 3
    pair = _make_pair(np.random.default_rng(71), programs, {2: [1, 3], 3: [7], 4: [2, 2, 6], 5: [10]}, 0, 1,
                      force_fallback=True)
    _require(np.array_equal(solve(pair['input']), pair['output']), "fallback mismatch")
    # Enlarge a freestanding component without changing the output census.
    pair = generate_puzzle(0)['train'][0]
    a, enlarged = _inspect(pair['input']), None
    for color, cells in a['objects']:
        if len(cells) >= 10:
            continue
        others = a['claimed'] | set().union(*(cc for _, cc in a['objects'] if cc is not cells))
        for r, c in sorted(cells):
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if (0 <= p[0] < a['grid'].shape[0] and 0 <= p[1] < a['grid'].shape[1]
                        and a['grid'][p] == a['background'] and not (_dilate({p}) & others)):
                    enlarged = a['grid'].copy()
                    enlarged[p] = color
                    break
            if enlarged is not None:
                break
        if enlarged is not None:
            break
    _require(enlarged is not None and np.array_equal(solve(enlarged), pair['output']), "object-area invariance")
    damaged = generate_puzzle(0)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('validator accepted output corruption')
