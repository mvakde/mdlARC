"""Shape-addressed broadcast ballots. Standard library and NumPy only.

The public solver reads only pixels. Generation constructs spatial elections,
not parent puzzle outputs. See task_spec.md for the intended rule and its limits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np


VECTORS = ((-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1))
DISTANCE = 5


def _normalize(cells):
    cells = tuple(cells)
    y, x = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r - y, c - x) for r, c in cells))


@lru_cache(maxsize=4096)
def _rotations(shape):
    shape = _normalize(shape)
    result = set()
    for _ in range(4):
        result.add(shape)
        shape = _normalize((c, -r) for r, c in shape)
    return tuple(sorted(result))


@lru_cache(maxsize=4096)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) |
                        set(_rotations(_normalize((r, -c) for r, c in shape)))))


def _canonical(cells):
    return min(_variants(_normalize(cells)))


def _extent(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    neighbors = VECTORS if diagonal else VECTORS[:4]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _arrow(tip, vector):
    r, c = tip
    dr, dc = vector
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple(sorted((r + a, c + b) for a, b in offsets))


def _read(input_grid):
    grid = np.asarray(input_grid)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30")
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("The outer border must visibly identify the background")
    targets, arrows = [], []
    for cells in _components(grid != bg):
        colors = {int(grid[p]) for p in cells}
        if len(colors) != 1:
            raise ValueError("Separated objects must be monochromatic")
        color = colors.pop()
        if len(cells) == 3:
            matches = [(tip, vector) for tip in cells for vector in VECTORS
                       if _arrow(tip, vector) == cells]
            if len(matches) != 1:
                raise ValueError("Three-cell components must be unambiguous arrows")
            tip, vector = matches[0]
            arrows.append(dict(cells=cells, color=color, tip=tip, vector=vector))
        elif len(cells) >= 4:
            rs, cs = zip(*cells)
            mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), bool)
            for r, c in cells:
                mask[r - min(rs), c - min(cs)] = True
            if len(_components(mask, diagonal=False)) != 1:
                raise ValueError("Silhouettes must be four-connected")
            targets.append(dict(cells=cells, color=color, shape=_canonical(cells),
                                raw=_normalize(cells)))
        else:
            raise ValueError("No singleton or two-cell objects in this family")
    if not arrows or not targets or len({t['color'] for t in targets}) != 1:
        raise ValueError("Need arrows and a single neutral silhouette color")
    neutral = targets[0]['color']
    if any(a['color'] == neutral for a in arrows):
        raise ValueError("Arrow colors and the neutral color have distinct roles")
    owner = {cell: i for i, t in enumerate(targets) for cell in t['cells']}
    for a in arrows:
        r, c = a['tip']
        dr, dc = a['vector']
        p = r + DISTANCE * dr, c + DISTANCE * dc
        if not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]):
            raise ValueError("Every five-step endpoint must be in bounds")
    return dict(grid=grid, background=bg, neutral=neutral, targets=targets,
                arrows=arrows, owner=owner)


def _predict(data, *, distance=DISTANCE, collisions=True, equivalence='free',
             broadcast=True, object_collisions=False, first_obstacle=False):
    """Also supports explicit counterfactuals used by validation, not solving."""
    def signature(t):
        if not broadcast:
            return t['cells']
        if equivalence == 'area':
            return len(t['cells'])
        if equivalence == 'box':
            return tuple(sorted(_extent(t['raw'])))
        if equivalence == 'rotation':
            return min(_rotations(t['raw']))
        return t['shape']

    deliveries = defaultdict(list)
    for a in data['arrows']:
        r, c = a['tip']
        dr, dc = a['vector']
        p = r + distance * dr, c + distance * dc
        if first_obstacle:
            for k in range(1, distance + 1):
                q = r + k * dr, c + k * dc
                if q in data['owner']:
                    p = q
                    break
        deliveries[p].append(a['color'])
    hit_totals = Counter()
    for p, colors in deliveries.items():
        if p in data['owner']:
            hit_totals[data['owner'][p]] += len(colors)
    votes = defaultdict(Counter)
    for p, colors in deliveries.items():
        if p not in data['owner']:
            continue
        index = data['owner'][p]
        if collisions and len(colors) > 1:
            continue
        if object_collisions and hit_totals[index] > 1:
            continue
        votes[signature(data['targets'][index])].update(colors)
    winners = {}
    for key in {signature(t) for t in data['targets']}:
        counts = votes[key]
        high = max(counts.values(), default=0)
        best = [c for c, n in counts.items() if n == high]
        # Counterfactuals may make ties; unchanged neutral is a generous default.
        winners[key] = best[0] if len(best) == 1 else data['neutral']
    out = data['grid'].copy()
    for t in data['targets']:
        for p in t['cells']:
            out[p] = winners[signature(t)]
    return out, dict(votes=dict(votes), winners=winners,
                     deliveries=dict(deliveries), hit_totals=hit_totals)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention, address, ballot and winner from this input."""
    data = _read(input_grid)
    out, election = _predict(data)
    if any(c == data['neutral'] for c in election['winners'].values()):
        raise ValueError("Every shape class requires a supported unique plurality")
    return out


@lru_cache(maxsize=1)
def _shape_pool():
    """Enumerate varied free polyominoes, rather than choosing a tiny catalog."""
    layer = {((0, 0),)}
    pool = set()
    for size in range(2, 9):
        grown = set()
        for s in sorted(layer):
            cells = set(s)
            frontier = {(r + dr, c + dc) for r, c in cells
                        for dr, dc in VECTORS[:4]} - cells
            for p in sorted(frontier):
                shape = _canonical(cells | {p})
                if max(_extent(shape)) <= 5:
                    grown.add(shape)
        layer = grown
        if size >= 4:
            for s in layer:
                h, w = _extent(s)
                blank = np.ones((h + 2, w + 2), bool)
                for r, c in s:
                    blank[r + 1, c + 1] = False
                if len(_components(blank, diagonal=False)) == 1:
                    pool.add(s)
    for length in (6, 7):
        pool.add(_canonical((0, c) for c in range(length)))
        pool.add(_canonical([(0, c) for c in range(length - 2)] +
                            [(1, 0), (1, length - 3)]))
    return tuple(sorted(pool))


def _block(available, cells, gap=1):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r - gap):min(h, r + gap + 1),
                  max(0, c - gap):min(w, c + gap + 1)] = False


def _place_shape(rng, available, variants, central=False):
    height, width = available.shape
    for j in rng.permutation(len(variants)):
        s = variants[int(j)]
        sh, sw = _extent(s)
        nh, nw = height - sh + 1, width - sw + 1
        valid = np.ones((nh, nw), bool)
        for r, c in s:
            valid &= available[r:r + nh, c:c + nw]
        if central:
            # This is a sampler preference, not an additional solving convention.
            valid[:5] = False
            valid[max(0, height - sh - 4):] = False
            valid[:, :5] = False
            valid[:, max(0, width - sw - 4):] = False
        positions = np.argwhere(valid)
        if len(positions):
            y, x = map(int, _choice(rng, positions))
            cells = tuple((y + r, x + c) for r, c in s)
            _block(available, cells)
            return cells
    return None


def _fits(cells, available):
    h, w = available.shape
    return all(0 <= r < h and 0 <= c < w and available[r, c]
               for r, c in cells)


def _candidate(p, vector, available):
    tip = p[0] - DISTANCE * vector[0], p[1] - DISTANCE * vector[1]
    cells = _arrow(tip, vector)
    return (tip, vector, cells, p) if _fits(cells, available) else None


def _install(rng, requests, collisions, targets, available):
    """Place arrows by solving tip + 5*vector = addressed body cell."""
    arrows, used = [], set()
    for indices, colors in collisions:
        endpoints = [p for i in indices for p in targets[i][1]]
        rng.shuffle(endpoints)
        placed = False
        for p in endpoints:
            if p in used:
                continue
            candidates = [a for v in VECTORS if (a := _candidate(p, v, available))]
            combos = list(combinations(candidates, len(colors)))
            for j in rng.permutation(len(combos)):
                group = combos[int(j)]
                local = available.copy()
                for a in group:
                    if not _fits(a[2], local):
                        break
                    _block(local, a[2])
                else:
                    available[:] = local
                    arrows.extend((a, color) for a, color in zip(group, colors))
                    used.add(p)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    # More constrained recipient subsets first; randomize within equal budgets.
    requests = [requests[int(i)] for i in rng.permutation(len(requests))]
    requests.sort(key=lambda req: sum(len(targets[i][1]) for i in req[0]))
    for indices, color in requests:
        options = [a for i in indices for p in targets[i][1] if p not in used
                   for v in VECTORS if (a := _candidate(p, v, available))]
        if not options:
            return None
        a = _choice(rng, options)
        arrows.append((a, color))
        used.add(a[3])
        _block(available, a[2])
    return arrows


def _miss(rng, available, occupied, *, through=False):
    options = []
    h, w = available.shape
    for p in sorted(occupied) if through else [(int(r), int(c))
                                              for r, c in np.argwhere(available)]:
        for v in VECTORS:
            steps = (2, 3, 4) if through else (0,)
            for step in steps:
                end = p[0] + (DISTANCE - step) * v[0], p[1] + (DISTANCE - step) * v[1]
                if not (0 <= end[0] < h and 0 <= end[1] < w) or end in occupied:
                    continue
                a = _candidate(end, v, available)
                if a is not None:
                    options.append(a)
    return _choice(rng, options) if options else None


def _build_pair(rng, shapes, colors, teaching=None, forbidden_winner=None,
                previous_roles=None):
    bg, neutral = map(int, rng.choice(10, 2, replace=False))
    if previous_roles:
        bg = _choice(rng, [c for c in range(10) if c != previous_roles[0]])
        neutral = _choice(rng, [c for c in range(10) if c not in (bg, previous_roles[1])])
    palette = [c for c in colors if c not in (bg, neutral)]
    # 'colors' is a shuffled complete palette, hence at least eight available.
    rng.shuffle(palette)
    winners = [_choice(rng, palette) for _ in shapes]
    if teaching is not None:
        winners[0] = _choice(rng, [c for c in palette if c != forbidden_winner])
        winners[1] = _choice(rng, [c for c in palette if c != winners[0]])
    if teaching == 0:
        repeats, ballots = [2, 1], [(2, 1, 1), (2, 1)]
    elif teaching == 1:
        repeats, ballots = [2, 2], [(3, 2), (1,)]
    else:
        repeats = [int(rng.integers(1, 3)) for _ in shapes]
        if len(shapes) == 1:
            repeats[0] = int(rng.integers(1, 4))
        ballots = [_choice(rng, ((1,), (2,), (2, 1), (2, 1, 1), (3, 2), (3, 2, 2)))
                   for _ in shapes]
        while sum(map(sum, ballots)) > 11:
            index = max(range(len(shapes)), key=lambda i: sum(ballots[i]))
            ballots[index] = (2, 1) if sum(ballots[index]) > 3 else (1,)
        # More addressed cells than single endpoints, reserving collision cells.
        for i, s in enumerate(shapes):
            if sum(ballots[i]) >= len(s) * repeats[i]:
                repeats[i] = min(3, repeats[i] + 1)
    vote_colors = []
    for winner, ballot in zip(winners, ballots):
        losers = [c for c in palette if c != winner]
        rng.shuffle(losers)
        vote_colors.append([winner] + losers[:len(ballot) - 1])

    for attempt in range(100):
        load = sum(repeats) + sum(map(sum, ballots))
        floor = 24 if teaching is not None or load >= 11 else 18
        height, width = map(int, rng.integers(floor, 31, 2))
        if attempt >= 65:
            height = width = 30
        available = np.ones((height, width), bool)
        available[[0, -1], :] = False
        available[:, [0, -1]] = False
        targets, class_indices = [], [[] for _ in shapes]
        failed = False
        # A receiver's order has no visible meaning. The first class gets an
        # opposite-handed replica in the first teaching demonstration.
        for i, s in enumerate(shapes):
            for copy in range(repeats[i]):
                allowed = _variants(s)
                if teaching == 0 and i == 0:
                    allowed = (s,) if copy == 0 else (_normalize((r, -c) for r, c in s),)
                cells = _place_shape(rng, available, allowed, central=(copy == 0))
                if cells is None:
                    failed = True
                    break
                class_indices[i].append(len(targets))
                targets.append((s, cells))
            if failed:
                break
        if failed:
            continue
        requests, collision_specs = [], []
        for i, (ballot, vc) in enumerate(zip(ballots, vote_colors)):
            for color_index, (count, color) in enumerate(zip(ballot, vc)):
                if teaching == 0 and i == 0:
                    requests.extend(([class_indices[i][0]], color) for _ in range(count))
                elif teaching == 1 and i == 0:
                    # Local 1–2 loses at one copy; local 2–0 wins at the other.
                    counts = (1, 2) if color_index == 0 else (2, 0)
                    requests.extend(([index], color) for index, n in zip(class_indices[i], counts)
                                    for _ in range(n))
                else:
                    requests.extend((class_indices[i], color) for _ in range(count))
        if teaching is not None:
            loser = vote_colors[1][-1] if len(vote_colors[1]) > 1 else _choice(
                rng, [c for c in palette if c != winners[1]])
            other = loser if teaching == 0 else _choice(
                rng, [c for c in palette if c not in (winners[1], loser)])
            # Both equal-color and mixed-color collisions are taught even 2/1.
            collision_specs = [(class_indices[1], [loser, other])]
        elif rng.random() < .7:
            i = int(rng.integers(len(shapes)))
            if len(shapes[i]) * repeats[i] > sum(ballots[i]):
                n = int(rng.choice((2, 2, 3)))
                collision_specs = [(class_indices[i], [_choice(rng, palette) for _ in range(n)])]
        arrows = _install(rng, requests, collision_specs, targets, available)
        if arrows is None:
            continue
        occupied = {p for _, cells in targets for p in cells}
        for _ in range(1 if teaching is not None else int(rng.integers(0, 3))):
            a = _miss(rng, available, occupied, through=teaching is not None or rng.random() < .6)
            if a is not None:
                arrows.append((a, _choice(rng, palette)))
                _block(available, a[2])
        inp = np.full((height, width), bg, dtype=np.int8)
        for _, cells in targets:
            for p in cells:
                inp[p] = neutral
        for a, color in arrows:
            for p in a[2]:
                inp[p] = color
        # Labels use the intended constructed elections, not public solve().
        out = inp.copy()
        for s, cells in targets:
            color = winners[shapes.index(s)]
            for p in cells:
                out[p] = color
        data = _read(inp)
        predicted, _ = _predict(data)
        if not np.array_equal(out, predicted):
            continue
        if teaching is not None:
            alternatives = ({'collisions': False}, {'equivalence': 'area'},
                            {'equivalence': 'box'}, {'broadcast': False},
                            {'object_collisions': True})
            if not all(not np.array_equal(out, _predict(data, **a)[0]) for a in alternatives):
                continue
            if teaching == 0 and np.array_equal(out, _predict(data, equivalence='rotation')[0]):
                continue
            if teaching == 0:
                if np.array_equal(out, _predict(data, first_obstacle=True)[0]):
                    continue
                if any(np.array_equal(out, _predict(data, distance=d)[0])
                       for d in range(1, 30) if d != DISTANCE):
                    continue
        return {'input': inp, 'output': out}
    raise RuntimeError("Bounded spatial election construction exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    pool = _shape_pool()
    eligible = [s for s in pool if 4 <= len(s) <= 6 and max(_extent(s)) <= 4
                and _normalize((r, -c) for r, c in s) not in _rotations(s)
                and any(t != s and len(t) == len(s)
                        and sorted(_extent(t)) == sorted(_extent(s)) for t in pool)]
    anchor = _choice(rng, eligible)
    companion = _choice(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                            and sorted(_extent(s)) == sorted(_extent(anchor))])
    colors = list(map(int, rng.permutation(10)))
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    first_roles, first_winner = None, None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = i if split == 'train' and i < 2 else None
            if teaching is not None:
                shapes = [anchor, companion]
            else:
                k = int(rng.integers(1, 4 if split == 'train' else 5))
                shapes = [pool[int(j)] for j in rng.choice(len(pool), k, replace=False)]
                if split == 'test':
                    # Every query carries at least one unfamiliar but supported
                    # shape. Its election is fully visible, so no hidden map.
                    new = [s for s in pool if s not in known and s not in shapes[1:]]
                    shapes[0] = _choice(rng, new)
            for _ in range(12):
                pair = _build_pair(rng, shapes, colors, teaching,
                                   first_winner if teaching == 1 else None,
                                   first_roles if teaching == 1 else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Failed to construct distinct inputs")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                known.update(shapes)
                if i == 0:
                    d = _read(pair['input'])
                    first_roles = d['background'], d['neutral']
                    first_winner = _predict(d)[1]['winners'][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (inp, out):
                assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
                assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            data = _read(inp)
            prediction, election = _predict(data)
            assert all(c != data['neutral'] for c in election['winners'].values()), 'tied/unsupported class'
            assert np.array_equal(out, prediction), 'wrong election result'
            assert np.array_equal(out, solve(inp.copy()))
            mask = np.zeros(inp.shape, bool)
            for t in data['targets']:
                for p in t['cells']:
                    mask[p] = True
                assert len({int(out[p]) for p in t['cells']}) == 1
            assert np.array_equal(inp[~mask], out[~mask]), 'arrows/background must be preserved'
            assert np.all(inp[mask] != out[mask]), 'each supported class gets a non-neutral winner'
            records[split].append((data, election))
    # These are actual answer differences, not merely sampled parameter labels.
    for alternative in ({'collisions': False}, {'equivalence': 'area'}, {'equivalence': 'box'},
                        {'equivalence': 'rotation'}, {'broadcast': False}, {'object_collisions': True},
                        {'first_obstacle': True}):
        assert any(not np.array_equal(pair['output'], _predict(data, **alternative)[0])
                   for pair, (data, _) in zip(puzzle['train'], records['train'])), alternative
    # Recoverability check within the finite endpoint/distance hypothesis family.
    fitting_distances = [d for d in range(1, 30)
                         if all(np.array_equal(pair['output'], _predict(data, distance=d)[0])
                                for pair, (data, _) in zip(puzzle['train'], records['train']))]
    assert fitting_distances == [DISTANCE], 'training must identify five-step projection'
    collision_colors = [colors for data, e in records['train']
                        for p, colors in e['deliveries'].items()
                        if len(colors) > 1 and p in data['owner']]
    assert any(len(set(cs)) == 1 for cs in collision_colors), 'teach equal-color cancellation'
    assert any(len(set(cs)) > 1 for cs in collision_colors), 'teach mixed-color cancellation'
    history = defaultdict(set)
    for _, e in records['train']:
        for s, c in e['winners'].items():
            history[s].add(c)
    assert any(len(cs) > 1 for cs in history.values()), 'teach locally supplied colors'
    assert len({d['background'] for d, _ in records['train']}) >= 2
    assert len({d['neutral'] for d, _ in records['train']}) >= 2
    assert any(len(v) >= 3 and 2 * max(v.values()) <= sum(v.values())
               for _, e in records['train'] for v in e['votes'].values()), 'teach plurality without majority'
    known = set().union(*(set(e['winners']) for _, e in records['train']))
    assert all(set(e['winners']) - known for _, e in records['test']), 'novel supported test shape'


def self_test() -> dict:
    """Bounded 42-seed pixel/invariant/diversity suite; no external oracle."""
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _read(pair['input'])
                _, e = _predict(d)
                records.append((split, pair, d, e))
    signatures = {tuple(_key(p[r]) for s in ('train', 'test') for p in task[s]
                        for r in ('input', 'output')) for task in puzzles}
    shapes = {p['input'].shape for _, p, _, _ in records}
    vocabulary = {t['shape'] for _, _, d, _ in records for t in d['targets']}
    directions = {a['vector'] for _, _, d, _ in records for a in d['arrows']}
    target_counts = {len(d['targets']) for _, _, d, _ in records}
    arrow_counts = {len(d['arrows']) for _, _, d, _ in records}
    ballots = {tuple(sorted(v.values(), reverse=True)) for _, _, _, e in records for v in e['votes'].values()}
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(vocabulary) >= 50 and directions == set(VECTORS)
    assert len(target_counts) >= 5 and len(arrow_counts) >= 8
    assert {(1,), (2, 1), (2, 1, 1), (3, 2)} <= ballots
    assert any(max(map(len, e['deliveries'].values())) >= 3 for _, _, _, e in records)
    assert any(d['grid'].shape[0] != d['grid'].shape[1] for _, _, d, _ in records)
    assert len({(d['background'], d['neutral']) for _, _, d, _ in records}) >= 30
    # Transformation covariance and immutability test the whole routing rule.
    p = puzzles[0]
    rotated = {s: [{r: np.rot90(pair[r]).copy() for r in pair} for pair in p[s]] for s in p}
    validate_generated_puzzle(rotated)
    inp = p['test'][0]['input'].copy()
    saved = inp.copy()
    solve(inp)
    assert np.array_equal(inp, saved)
    assert all(_key(a[r]) == _key(b[r]) for s in p for a, b in zip(p[s], generate_puzzle(0)[s])
               for r in ('input', 'output'))
    broken = {s: [{r: pair[r].copy() for r in pair} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted background accepted')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                canvas_shapes=len(shapes), shape_classes=len(vocabulary),
                arrow_counts=sorted(arrow_counts), target_counts=sorted(target_counts),
                ballots=sorted(ballots), independent_semantic_check=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
