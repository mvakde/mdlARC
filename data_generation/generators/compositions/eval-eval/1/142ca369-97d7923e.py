"""Contact census: reflected rays form unary keys for ranked capped bars.

No evidence files or parent modules are imported. All semantics are reconstructed
from the input. Construction records never enter solve or validation. The module
writes no files; diagnostics can be serialized with compact JSON by a caller.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(grid, bg):
    """Objects have a cardinal background moat, including differently colored ones."""
    remaining = set(map(tuple, np.argwhere(grid != bg)))
    objects = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in _NEIGHBORS:
                q = (p[0] + dr, p[1] + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        objects.append(tuple(sorted(cells)))
    return objects


def _decode(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > grid.size / 2, "background must occupy a strict majority")
    sources, mirrors, bars = [], [], []
    for cells in _components(grid, bg):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        values = [int(grid[p]) for p in cells]
        if len(set(values)) == 1:
            color = values[0]
            if len(cells) == 3 and r1 - r0 == c1 - c0 == 1:
                missing = next(p for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)) if p not in cells)
                elbow = (r0 + r1 - missing[0], c0 + c1 - missing[1])
                direction = (elbow[0] - missing[0], elbow[1] - missing[1])
                sources.append({"cells": cells, "color": color, "elbow": elbow, "direction": direction})
            else:
                _require(len(cells) % 2 == 1 and (r0 == r1 or c0 == c1), "unrecognized monochrome object")
                _require(len(cells) == max(r1 - r0, c1 - c0) + 1, "broken mirror")
                mirrors.append({"cells": cells, "color": color, "mid": ((r0 + r1) // 2, (c0 + c1) // 2),
                                "axis": None if len(cells) == 1 else (0 if r0 == r1 else 1)})
        else:
            _require((r0 == r1 or c0 == c1) and len(cells) == max(r1-r0, c1-c0)+1,
                     "a multicolor object must be a straight capped bar")
            _require(len(cells) >= 3 and values[0] == values[-1], "unequal or missing caps")
            _require(len(set(values[1:-1])) == 1 and values[1] != values[0], "nonuniform payload")
            bars.append({"cells": cells, "cap": values[0], "payload": values[1], "length": len(cells)})
    _require(sources, "no L source")
    return {"background": bg, "sources": sources, "mirrors": mirrors, "bars": bars}


def _trace(grid, scene):
    """Cell tracer; count contact events, including repeated-color contacts."""
    h, w = grid.shape
    bg = scene['background']
    painted, routes, census, used = {}, [], Counter(), Counter()
    for source in scene['sources']:
        position, direction, shade = source['elbow'], source['direction'], source['color']
        path, contacts, visited = [], [], set()
        while True:
            position = (position[0] + direction[0], position[1] + direction[1])
            if not (0 <= position[0] < h and 0 <= position[1] < w):
                break
            state = position, direction
            _require(state not in visited, "optical loop")
            visited.add(state)
            _require(int(grid[position]) == bg, "a ray crosses an input object")
            hits = []
            for i, mirror in enumerate(scene['mirrors']):
                axes = (0, 1) if mirror['axis'] is None else (mirror['axis'],)
                for axis in axes:
                    other = 1 - axis
                    if (position[other] == mirror['mid'][other]
                            and mirror['mid'][axis] - position[axis] == direction[axis]):
                        hits.append((i, axis))
            _require(len(hits) <= 1, "ambiguous midpoint contact")
            if hits:
                i, axis = hits[0]
                mirror = scene['mirrors'][i]
                used[i] += 1
                # The family deliberately uses distinct encounters, not repeated
                # traversals of one physical mirror. Decoys may have zero visits.
                _require(used[i] == 1, "mirror visited more than once")
                shade = mirror['color']
                census[shade] += 1
                contacts.append((mirror['mid'], axis, position, shade, len(mirror['cells'])))
                direction = tuple(-d if k == axis else d for k, d in enumerate(direction))
            _require(position not in painted, "ray paths must be cell-disjoint")
            painted[position] = shade
            path.append((position, shade))
        routes.append({"source": source, "path": path, "contacts": contacts})
    return {"painted": painted, "routes": routes, "census": census, "used": used}


def _analyze(input_grid):
    grid = _grid(input_grid)
    scene = _decode(grid)
    _require(scene['bars'], "missing receiver bars")
    controls = {b['cap'] for b in scene['bars']}
    _require(all(m['color'] in controls for m in scene['mirrors']), "mirror lacks a receiver family")
    _require(all(b['payload'] not in controls for b in scene['bars']), "payload and cap roles overlap")
    families = {cap: [b for b in scene['bars'] if b['cap'] == cap] for cap in controls}
    optical = _trace(grid, scene)
    choices = {}
    for cap, family in families.items():
        _require(len(family) >= 2, "receiver family needs at least two competitors")
        _require(len({b['length'] for b in family}) == len(family), "tied receiver lengths")
        n = optical['census'][cap]
        _require(n <= len(family), "census rank exceeds receiver population")
        choices[cap] = sorted(family, key=lambda b: -b['length'])[n-1] if n else None
    return grid, scene, optical, families, choices


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Draw every reflected ray; its per-color contact count selects a bar."""
    grid, scene, optical, families, choices = _analyze(input_grid)
    result = grid.copy()
    for p, shade in optical['painted'].items():
        result[p] = shade
    for chosen in choices.values():
        if chosen is not None:
            for p in chosen['cells'][1:-1]:
                result[p] = chosen['cap']
    return result


def _selection_signature(choices):
    return tuple(sorted((cap, None if b is None else b['cells']) for cap, b in choices.items()))


def _alternatives(scene, optical, families, choices):
    """Named simple selection shortcuts, tested without using output pixels."""
    static = Counter(m['color'] for m in scene['mirrors'])
    with_launch = optical['census'] + Counter(s['color'] for s in scene['sources'])
    launch_colors = {s['color'] for s in scene['sources']}
    intended = _selection_signature(choices)
    answer = {}
    for name in ('all_mirrors', 'include_source', 'source_second', 'longest', 'shortest', 'ascending', 'spatial', 'reverse_spatial'):
        alternate = {}
        invalid = False
        for cap, family in families.items():
            n = optical['census'][cap]
            if name == 'all_mirrors':
                n = static[cap]
            elif name == 'include_source':
                n = with_launch[cap]
            if n == 0:
                alternate[cap] = None
                continue
            if name == 'source_second':
                n = 2 if cap in launch_colors else 1
            if n > len(family):
                invalid = True
                break
            if name == 'longest':
                chosen = max(family, key=lambda b: b['length'])
            elif name == 'shortest':
                chosen = min(family, key=lambda b: b['length'])
            elif name == 'ascending':
                chosen = sorted(family, key=lambda b: b['length'])[n-1]
            elif name in ('spatial', 'reverse_spatial'):
                chosen = sorted(family, key=lambda b: b['cells'][0], reverse=name == 'reverse_spatial')[n-1]
            else:
                chosen = sorted(family, key=lambda b: -b['length'])[n-1]
            alternate[cap] = chosen
        answer[name] = invalid or _selection_signature(alternate) != intended
    return answer


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split])


def _features(grid):
    grid, scene, optical, families, choices = _analyze(grid)
    axes = {c[1] for route in optical['routes'] for c in route['contacts'] if c[4] > 1}
    return {"shape": tuple(grid.shape), "background": scene['background'], "sources": len(scene['sources']),
            "families": len(families), "ranks": tuple(sorted(optical['census'].values())),
            "zero": sum(optical['census'][cap] == 0 for cap in families), "axes": axes,
            "points": any(c[4] == 1 for r in optical['routes'] for c in r['contacts']),
            "decoys": len(scene['mirrors']) - len(optical['used']),
            "directions": {s['direction'] for s in scene['sources']},
            "mirror_lengths": {len(m['cells']) for m in scene['mirrors']},
            "bar_axes": {0 if b['cells'][0][0] == b['cells'][-1][0] else 1 for b in scene['bars']},
            "populations": tuple(sorted(len(f) for f in families.values())),
            "pooled": any(sum(any(c[3] == cap for c in r['contacts']) for r in optical['routes']) > 1
                          for cap in families),
            "alternatives": _alternatives(scene, optical, families, choices),
            "nonextreme": any(1 < optical['census'][cap] < len(f) for cap, f in families.items()),
            "cap_ranks": {cap: optical['census'][cap] for cap in families},
            "transitions": [(r['source']['color'] if i == 0 else r['contacts'][i-1][3], c[3])
                            for r in optical['routes'] for i, c in enumerate(r['contacts'])]}


def validate_generated_puzzle(puzzle):
    """Semantic checks from input pixels, plus the documented teaching contract."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "invalid pair counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            grid, scene, optical, families, choices = _analyze(inp)
            # Explicit difference-mask checks complement full solve equality.
            changed = set(optical['painted'])
            for bar in choices.values():
                if bar is not None:
                    changed.update(bar['cells'][1:-1])
            _require(changed == set(map(tuple, np.argwhere(inp != out))), "wrong changed-cell set")
            for p, shade in optical['painted'].items():
                _require(out[p] == shade, "wrong ray color")
            for cap, family in families.items():
                for bar in family:
                    for p in bar['cells']:
                        desired = cap if bar is choices[cap] and p in bar['cells'][1:-1] else inp[p]
                        _require(out[p] == desired, "wrong ordinal receiver fill")
            _require(np.array_equal(solve(inp), out), "solver mismatch")
            _require(bool(changed) and any(b is not None for b in choices.values()), "no active receiver")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "new color introduced")
            features[split].append(_features(inp))
    train = features['train']
    _require(len(set().union(*(f['directions'] for f in train))) >= 2, "two launch directions need demonstrations")
    _require({0, 1} <= set().union(*(f['axes'] for f in train)), "both mirror normals need demonstrations")
    _require(any(f['points'] for f in train), "point mirror not demonstrated")
    _require({1, 2} <= {r for f in train for r in f['ranks']}, "ranks one and two not demonstrated")
    histories = {}
    for f in train:
        for cap, rank in f['cap_ranks'].items():
            histories.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= ranks for ranks in histories.values()), "same cap must change ordinal")
    _require(any(f['nonextreme'] for f in train), "no non-extreme training selection")
    _require(any(f['zero'] for f in train), "unvisited family must be demonstrated")
    for name in train[0]['alternatives']:
        _require(any(f['alternatives'][name] for f in train), "surviving shortcut: " + name)
    transitions = [t for f in train for t in f['transitions']]
    _require(any(a == b for a, b in transitions) and any(a != b for a, b in transitions),
             "changed and unchanged contact colors need demonstrations")
    trained = {r for f in train for r in f['ranks']}
    for f in features['test']:
        _require(all(r in trained or set(range(1, r)) <= trained for r in f['ranks']), "unsupported rank shift")
    return None


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _place(grid, cells, colors, forbidden=()):
    cells = tuple(cells)
    if not _inside(cells, grid.shape) or set(cells) & set(forbidden):
        return None
    bg = int(np.argmax(np.bincount(grid.ravel(), minlength=10)))
    for r, c in cells:
        if grid[r, c] != bg:
            return None
        for dr, dc in _NEIGHBORS:
            q = (r + dr, c + dc)
            if q not in cells and _inside((q,), grid.shape) and grid[q] != bg:
                return None
    trial = grid.copy()
    for p, shade in zip(cells, colors):
        trial[p] = shade
    return trial


def _l_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return ((r, c), (r-dr, c), (r, c-dc))


def _mirror_cells(mid, axis, length):
    radius = length // 2
    return tuple((mid[0], mid[1] + i) if axis == 0 else (mid[0] + i, mid[1])
                 for i in range(-radius, radius + 1))


def _route_geometry(scene, optical):
    return {r['source']['elbow']: tuple((c[0], c[1], c[2]) for c in r['contacts']) for r in optical['routes']}


def _add_source(grid, rng, color):
    candidates = []
    for _ in range(100):
        direction = _DIRECTIONS[int(rng.integers(4))]
        elbow = (int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1])))
        candidates.append((elbow, direction))
    rng.shuffle(candidates)
    try:
        before = _trace(grid, _decode(grid))
    except AssertionError:
        # An empty optical canvas has no source yet.
        before = None
    for elbow, direction in candidates:
        trial = _place(grid, _l_cells(elbow, direction), [color] * 3)
        if trial is None:
            continue
        try:
            scene = _decode(trial)
            optical = _trace(trial, scene)
        except AssertionError:
            continue
        route = next(r for r in optical['routes'] if r['source']['elbow'] == elbow)
        if len(route['path']) < 5 or route['contacts']:
            continue
        if before is not None:
            old = _route_geometry(None, before)
            new = _route_geometry(None, optical)
            if any(new[p] != contacts for p, contacts in old.items()):
                continue
        return trial, elbow
    return None


def _extend(grid, elbow, rng, color, axis, length):
    scene = _decode(grid)
    before = _trace(grid, scene)
    route = next(r for r in before['routes'] if r['source']['elbow'] == elbow)
    direction = route['source']['direction']
    for contact in route['contacts']:
        direction = tuple(-d if k == contact[1] else d for k, d in enumerate(direction))
    anchor = route['contacts'][-1][2] if route['contacts'] else elbow
    options = []
    for distance in range(2, 29):
        contact = (anchor[0] + distance * direction[0], anchor[1] + distance * direction[1])
        reflected = tuple(-d if k == axis else d for k, d in enumerate(direction))
        if not _inside((contact, (contact[0]+2*reflected[0], contact[1]+2*reflected[1])), grid.shape):
            continue
        midpoint = tuple(contact[k] + (direction[k] if k == axis else 0) for k in range(2))
        options.append((contact, midpoint))
    rng.shuffle(options)
    old = _route_geometry(scene, before)
    for contact, midpoint in options:
        cells = _mirror_cells(midpoint, axis, length)
        trial = _place(grid, cells, [color] * len(cells))
        if trial is None:
            continue
        try:
            optical = _trace(trial, _decode(trial))
        except AssertionError:
            continue
        new = _route_geometry(None, optical)
        expected = dict(old)
        expected[elbow] = old[elbow] + ((midpoint, axis, contact),)
        if new == expected:
            return trial
    return None


def _optics(rng, bg, source_color, sequences, teaching):
    """Bounded rejection of geometric candidates, with no saved trial payloads."""
    for attempt in range(100):
        h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
        grid = np.full((h, w), bg, dtype=np.int8)
        success = True
        for index, sequence in enumerate(sequences):
            added = _add_source(grid, rng, source_color)
            if added is None:
                success = False
                break
            grid, elbow = added
            n = len(sequence)
            axes = [int(rng.integers(2)) for _ in sequence]
            lengths = [int(rng.choice([1, 3, 5, 7])) for _ in sequence]
            if teaching:
                # Ensure both normals and a point example in each anchor;
                # contact order, dimensions, flights, rotations all vary.
                order = list(map(int, rng.permutation(n)))
                axes[order[0]], axes[order[1]] = 0, 1
                lengths[order[0]] = int(rng.choice([3, 5]))
                lengths[order[1]] = int(rng.choice([3, 5]))
                lengths[order[2]] = 1
            for color, axis, length in zip(sequence, axes, lengths):
                trial = _extend(grid, elbow, rng, color, axis, length)
                if trial is None:
                    success = False
                    break
                grid = trial
            if not success:
                break
        if success:
            return grid
    raise RuntimeError('bounded optical construction exhausted')


def _add_decoys(grid, rng, colors):
    before = _trace(grid, _decode(grid))
    old_path = before['painted']
    for color in colors:
        placed = False
        for _ in range(160):
            axis, length = int(rng.integers(2)), int(rng.choice([1, 3, 5, 7]))
            mid = (int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1])))
            cells = _mirror_cells(mid, axis, length)
            trial = _place(grid, cells, [color] * len(cells), old_path)
            if trial is None:
                continue
            try:
                after = _trace(trial, _decode(trial))
            except AssertionError:
                continue
            if after['painted'] == old_path:
                grid = trial
                placed = True
                break
        if not placed:
            return None
    return grid


def _receivers(grid, rng, caps, bg):
    """Scatter receivers throughout the arena, with mixed axes and no rack panel."""
    scene = _decode(grid)
    optical = _trace(grid, scene)
    static = Counter(m['color'] for m in scene['mirrors'])
    records = []
    payloads = [c for c in range(10) if c != bg and c not in caps]
    rng.shuffle(payloads)
    payloads = payloads[:int(rng.integers(1, min(3, len(payloads)) + 1))]
    for cap in caps:
        n = optical['census'][cap]
        minimum = max(2, n + 1, static[cap])
        population = int(rng.integers(minimum, min(6, minimum + 2) + 1))
        upper = int(rng.integers(max(8, population + 2), 15))
        lengths = [int(v) for v in rng.choice(np.arange(3, upper+1), population, replace=False)]
        for length in lengths:
            records.append((length, cap, int(rng.choice(payloads))))
    # Longer objects placed first reduce packing retries; their positions still
    # do not encode ranks. A failed packing discards only this in-memory trial.
    records.sort(reverse=True)
    original = grid
    for _ in range(10):
        grid = original.copy()
        complete = True
        for length, cap, payload in records:
            placed = False
            for _ in range(240):
                axis = int(rng.integers(2))
                r, c = int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1]))
                cells = tuple((r, c+i) if axis == 0 else (r+i, c) for i in range(length))
                shades = [cap] + [payload] * (length-2) + [cap]
                trial = _place(grid, cells, shades, optical['painted'])
                if trial is not None:
                    grid = trial
                    placed = True
                    break
            if not placed:
                complete = False
                break
        if complete:
            try:
                analysis = _analyze(grid)
                alt = _alternatives(analysis[1], analysis[2], analysis[3], analysis[4])
            except AssertionError:
                continue
            if alt['spatial'] and alt['reverse_spatial']:
                return grid
    return None


def _pair(rng, controls, bg, sequences, dormant, teaching=False):
    caps = sorted({c for sequence in sequences for c in sequence} | set(dormant))
    # Keep the SAME source color across anchors while its census rank changes.
    # This rules out selecting rank two just because the source has that color.
    if teaching:
        source_color = controls[0]
    else:
        source_color = sequences[0][0] if rng.random() < .6 else int(rng.choice([c for c in range(10) if c != bg]))
    for _ in range(12):
        try:
            grid = _optics(rng, bg, source_color, sequences, teaching)
        except RuntimeError:
            continue
        # A same-color decoy changes a static inventory's rank, but never the
        # contact census. A dormant family can itself own decoy mirrors.
        decoys = [sequences[0][0]]
        if dormant and rng.random() < .8:
            decoys.append(int(rng.choice(dormant)))
        if rng.random() < .3:
            decoys.append(int(rng.choice(caps)))
        grid = _add_decoys(grid, rng, decoys)
        if grid is None:
            continue
        grid = _receivers(grid, rng, caps, bg)
        if grid is not None:
            return {'input': grid, 'output': solve(grid)}
    raise RuntimeError('bounded receiver construction exhausted')


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(name + ' has an illegal count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_count(num_train, 'num_train', 2, 10)
    num_test = _checked_count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    controls = palette[:3]
    backgrounds = palette[3:]
    a, b, c = controls
    shift = 'rank' if rng.integers(2) else 'pool'
    puzzle, seen = {'train': [], 'test': []}, set()
    specs = [('train', [[a, a, b]], [], True), ('train', [[b, a, b]], [c], True)]
    for _ in range(num_train - 2):
        active = list(map(int, rng.choice(controls, int(rng.integers(1, 4)), replace=False)))
        sequence = [cap for cap in active for _ in range(int(rng.integers(1, 3)))]
        rng.shuffle(sequence)
        specs.append(('train', [sequence], [cap for cap in controls if cap not in active] if rng.random() < .4 else [], False))
    for i in range(num_test):
        if shift == 'rank':
            active = [a, b]
            rng.shuffle(active)
            sequence = [active[0]] * 3 + [active[1]] * int(rng.integers(1, 3))
            rng.shuffle(sequence)
            specs.append(('test', [sequence], [c] if rng.random() < .5 else [], False))
        else:
            # Pool the same control across disjoint rays. Each source alone
            # contributes only rank one for a, but their union requests 2 or 3.
            sequences = [[a, b], [a]]
            if rng.random() < .35:
                sequences.append([b])
            specs.append(('test', sequences, [c] if rng.random() < .5 else [], False))
    for split, sequences, dormant, teaching in specs:
        for attempt in range(8):
            bg = int(rng.choice(backgrounds))
            pair = _pair(rng, controls, bg, sequences, dormant, teaching)
            if teaching and puzzle['train']:
                first_direction = _decode(puzzle['train'][0]['input'])['sources'][0]['direction']
                this_direction = _decode(pair['input'])['sources'][0]['direction']
                if first_direction == this_direction:
                    pair = {k: np.ascontiguousarray(np.rot90(v)) for k, v in pair.items()}
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('bounded uniqueness construction exhausted')
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """42 seeds, streamed feature aggregates, invariances and negative checks.

    No files or traces are saved. At most one 14-pair puzzle and transformed
    copies of one pair are retained; diagnostics contain counts only.
    """
    _hand_regression()
    shapes, backgrounds, directions, mirror_lengths, populations, families = set(), set(), set(), set(), set(), set()
    rank_values, source_counts, structures, fingerprints = set(), set(), set(), set()
    pooled = dormant = nonextreme = pair_count = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        if seed == 0:
            first = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'])
                shapes.add(f['shape'])
                backgrounds.add(f['background'])
                directions.update(f['directions'])
                mirror_lengths.update(f['mirror_lengths'])
                populations.update(f['populations'])
                families.add(f['families'])
                rank_values.update(f['ranks'])
                source_counts.add(f['sources'])
                pooled += f['pooled']
                dormant += bool(f['zero'])
                nonextreme += f['nonextreme']
                pair_count += 1
                roles, normalized = {}, []
                for v in pair['input'].flat:
                    normalized.append(roles.setdefault(int(v), len(roles)))
                structures.add(hashlib.sha256(bytes(pair['input'].shape) + bytes(normalized)).hexdigest())
        # Dihedral and palette invariance on one test per seed, including contact
        # recoloring and scattered horizontal/vertical receivers.
        pair = puzzle['test'][0]
        for turn in range(4):
            inp, out = np.rot90(pair['input'], turn), np.rot90(pair['output'], turn)
            if seed % 2:
                inp, out = np.fliplr(inp), np.fliplr(out)
            mapping = np.array([(i + 3) % 10 for i in range(10)], dtype=np.int8)
            _require(np.array_equal(solve(mapping[inp]), mapping[out]), "rotation/reflection/palette failure")
    _require(len(fingerprints) == 42 and len(structures) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 35 and len(backgrounds) >= 8, "insufficient size/palette variation")
    _require(directions == set(_DIRECTIONS), "missing launch rotation")
    _require({1, 3, 5, 7} <= mirror_lengths, "missing mirror lengths")
    _require({1, 2, 3} <= rank_values and {1, 2, 3} <= source_counts, "rank/pooling variation")
    _require({1, 2, 3} <= families and len(populations) >= 4, "receiver family variation")
    _require(pooled >= 10 and dormant >= 42 and nonextreme >= 100, "semantic diversity failure")
    _require(_signature(first) == _signature(generate_puzzle(0)), "nondeterministic seed")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(100 + 10*train + test, train, test)
            _require(len(p['train']) == train and len(p['test']) == test, "explicit count ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Wrong receiver fills and missing ray cells must both be rejected.
    for location in ('ray', 'bar'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in first[s]] for s in first}
        pair = bad['test'][0]
        _, _, optical, _, choices = _analyze(pair['input'])
        p = next(iter(optical['painted'])) if location == 'ray' else next(b for b in choices.values() if b is not None)['cells'][1]
        pair['output'][p] = pair['input'][p]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('mutated output accepted')
    return {"seeds": 42, "pairs": pair_count, "shapes": len(shapes), "normalized_structures": len(structures),
            "backgrounds": len(backgrounds), "source_counts": sorted(source_counts), "ranks": sorted(rank_values),
            "mirror_lengths": sorted(mirror_lengths), "populations": sorted(populations),
            "pooled_pairs": pooled, "dormant_pairs": dormant, "nonextreme_pairs": nonextreme}


def _hand_regression():
    """A hand-enumerated small arena, outside the constructor's size range.

    Expected rays and target interior are explicit coordinates. A second case
    moves only one optical object: the selected receiver must then change.
    These are checks only and are never consulted by solve or generation.
    """
    inp = np.zeros((15, 20), dtype=np.int8)
    for p in ((2, 2), (1, 2), (2, 1)):
        inp[p] = 1
    inp[6, 4:7] = 2
    inp[2, 9] = 2
    inp[10, 7] = 2  # Off-route decoy, not a third unary unit.
    for row, start, length in ((6, 11, 7), (10, 12, 5), (13, 14, 3)):
        inp[row, start:start+length] = 3
        inp[row, start] = inp[row, start+length-1] = 2
    for row, col, length in ((8, 1, 4), (12, 4, 3)):
        inp[row:row+length, col] = 3
        inp[row, col] = inp[row+length-1, col] = 4
    expected = inp.copy()
    for p in ((3, 3), (4, 4)):
        expected[p] = 1
    for p in ((5, 5), (4, 6), (3, 7), (2, 8), (1, 7), (0, 6)):
        expected[p] = 2
    expected[10, 13:16] = 2
    _require(np.array_equal(solve(inp), expected), "hand-enumerated reflection/census regression")
    moved = inp.copy()
    moved[2, 9], moved[8, 9] = 0, 2
    expected = moved.copy()
    for p in ((3, 3), (4, 4)):
        expected[p] = 1
    for p in ((5, 5), (4, 6), (3, 7), (2, 8), (1, 9), (0, 10)):
        expected[p] = 2
    expected[6, 12:17] = 2
    _require(np.array_equal(solve(moved), expected), "moving a mirror must change the ranked receiver")
