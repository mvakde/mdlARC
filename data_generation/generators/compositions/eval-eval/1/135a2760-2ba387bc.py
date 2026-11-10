"""Woven color clocks: class-specific periodic repair and joint-cycle inlays.

Only NumPy and the standard library are used. The solver receives pixels only.
The generation checks are authored checks, not an independent reference solver.
"""
from __future__ import annotations

from math import lcm

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _tile(size, color, background, hollow):
    tile = np.full((size, size), color, dtype=np.int8)
    if hollow:
        tile[1:-1, 1:-1] = background
    return tile


def _recover(colors):
    """Enumerate all strict-majority, <=1-error periodic explanations."""
    observed = np.asarray(colors, dtype=np.int8)
    palette = np.unique(observed)
    _require(len(palette) == 2, "each clock must visibly use two colors")
    candidates = []
    for period in range(1, min(4, len(observed) // 3) + 1):
        word = []
        for phase in range(period):
            values = observed[phase::period]
            counts = [int(np.count_nonzero(values == c)) for c in palette]
            if max(counts) * 2 <= len(values):
                break
            word.append(int(palette[int(np.argmax(counts))]))
        else:
            word = np.array(word, dtype=np.int8)
            clean = word[np.arange(len(observed)) % period]
            if len(np.unique(word)) == 2 and np.count_nonzero(clean != observed) <= 1:
                candidates.append((period, word, clean))
    _require(bool(candidates), "no supported periodic clock")
    period, word, clean = candidates[0]
    _require(all(np.array_equal(c[2], clean) for c in candidates),
             "competing periodic repairs")
    _require(2 <= period <= 4, "primitive period must be 2..4")
    return dict(period=period, word=word, clean=clean,
                errors=tuple(int(i) for i in np.flatnonzero(clean != observed)),
                candidates=tuple(c[0] for c in candidates))


def _parse(grid):
    background = int(grid[0, 0])
    _require(all(np.all(edge == background) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "uniform background border required")
    remaining = set(zip(*np.nonzero(grid != background)))
    objects = []
    size = None
    height, width = grid.shape
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    component.append(point)
                    stack.append(point)
        top = int(min(r for r, _ in component))
        left = int(min(c for _, c in component))
        bottom = int(max(r for r, _ in component)) + 1
        right = int(max(c for _, c in component)) + 1
        s = bottom - top
        _require(s == right-left and 3 <= s <= 6, "complete square required")
        _require(size is None or s == size, "one tile size per input required")
        size = s
        tile = grid[top:bottom, left:right]
        color = int(tile[0, 0])
        _require(color != background, "tile must differ from background")
        if np.all(tile == color):
            kind = 1  # solid: interior donor
        elif np.array_equal(tile, _tile(s, color, background, True)):
            kind = 0  # hollow: rim receiver
        else:
            raise ValueError("tile is neither monochrome hollow nor solid")
        neighborhood = grid[max(0, top-1):min(height, bottom+1),
                            max(0, left-1):min(width, right+1)].copy()
        neighborhood[top-max(0, top-1):bottom-max(0, top-1),
                     left-max(0, left-1):right-max(0, left-1)] = background
        _require(bool(np.all(neighborhood == background)), "tiles must be separated")
        objects.append(dict(top=top, left=left, color=color, kind=kind))
    _require(bool(objects), "tiles required")
    objects.sort(key=lambda o: (o['top'], o['left']))
    queues = [[o for o in objects if o['kind'] == k] for k in (0, 1)]
    _require(all(queues), "both shape classes required")
    return dict(background=background, size=size, objects=objects, queues=queues)


def _analyze(grid):
    analysis = _parse(grid)
    clocks = [_recover([o['color'] for o in q]) for q in analysis['queues']]
    joint = lcm(*(clock['period'] for clock in clocks))
    _require(joint * analysis['size'] <= 30, "joint cycle exceeds ARC limit")
    _require(min(map(len, analysis['queues'])) >= joint,
             "joint cycle must be represented in both queues")
    _require(any(clock['errors'] for clock in clocks), "at least one error required")
    for queue, clock in zip(analysis['queues'], clocks):
        for obj, color in zip(queue, clock['clean']):
            obj['clean_color'] = int(color)
    analysis.update(clocks=clocks, joint=joint)
    return analysis


def _inlays(rims, interiors, size):
    _require(len(rims) == len(interiors) and len(rims) * size <= 30,
             "invalid strip length")
    output = np.empty((len(rims)*size, size), dtype=np.int8)
    for rank, (rim, interior) in enumerate(zip(rims, interiors)):
        output[rank*size:(rank+1)*size] = _tile(size, rim, interior, True)
    return output


def _answer(analysis):
    ranks = np.arange(analysis['joint'])
    colors = [c['word'][ranks % c['period']] for c in analysis['clocks']]
    return _inlays(*colors, analysis['size'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover clocks from geometry and repetition, then weave one joint cycle."""
    return _answer(_analyze(_grid(input_grid)))


def _alternative(analysis, mode):
    """Concrete ablations; these isolate repair or ordering, not an oracle."""
    joint, bg = analysis['joint'], analysis['background']
    if mode == 'greedy':
        rows = [[bg, bg]]
        occupied = set()
        for obj in analysis['objects']:
            kind = obj['kind']
            if kind in occupied:
                rows.append([bg, bg])
                occupied = set()
            rows[-1][kind] = obj['clean_color']
            occupied.add(kind)
        # Give this competing rule the correct cycle length to isolate pairing.
        rows += [[bg, bg]] * max(0, joint-len(rows))
        return _inlays([r[0] for r in rows[:joint]],
                       [r[1] for r in rows[:joint]], analysis['size'])
    streams = []
    for queue in analysis['queues']:
        if mode == 'horizontal':
            queue = sorted(queue, key=lambda o: (o['left'], o['top']))
        elif mode == 'reverse':
            queue = list(reversed(queue))
        elif mode == 'reverse_ties':
            queue = sorted(queue, key=lambda o: (o['top'], -o['left']))
        field = 'color' if mode == 'no_repair' else 'clean_color'
        streams.append([o[field] for o in queue[:joint]])
    return _inlays(*streams, analysis['size'])


def _witnesses(analysis):
    answer = _answer(analysis)
    return {mode: not np.array_equal(answer, _alternative(analysis, mode))
            for mode in ('no_repair', 'greedy', 'horizontal', 'reverse', 'reverse_ties')}


def _primitive(bits):
    p = len(bits)
    return len(set(bits)) == 2 and not any(
        p % d == 0 and all(bits[i] == bits[i % d] for i in range(p))
        for d in range(1, p))


def _make_clock(rng, period, count, palette, first_error=False):
    # Finite enumeration supplies a fallback after randomized proposals.
    words = [tuple((number >> i) & 1 for i in range(period))
             for number in range(1, 2**period-1)]
    words = [word for word in words if _primitive(word)]
    trials = [(words[int(rng.integers(len(words)))],
               int(rng.integers(period if first_error else count))) for _ in range(32)]
    trials += [(word, rank) for word in words
               for rank in range(period if first_error else count)]
    for bits, error in trials:
        clean_bits = np.array(bits, dtype=np.int8)[np.arange(count) % period]
        noisy_bits = clean_bits.copy()
        noisy_bits[error] ^= 1
        clean = np.array(palette, dtype=np.int8)[clean_bits]
        noisy = np.array(palette, dtype=np.int8)[noisy_bits]
        try:
            recovered = _recover(noisy)
        except ValueError:
            continue
        if recovered['period'] == period and np.array_equal(recovered['clean'], clean):
            return noisy, clean
    raise RuntimeError("finite clock construction exhausted")


def _sizes(periods):
    return [s for s in range(3, 7)
            if s*lcm(*periods) <= 30
            and 3*sum(periods) <= ((29)//(s+1))**2]


def _counts(rng, periods, size, testing):
    counts = [3*p for p in periods]
    capacity = (29//(size+1))**2
    spare = min(capacity-sum(counts), 32-sum(counts), (10 if testing else 5))
    # Test tails always contain a partial copy; first allocate one extra rank.
    extra = int(rng.integers(1 if testing else 0, spare+1))
    for _ in range(extra):
        choices = [i for i, n in enumerate(counts) if n < 16]
        counts[int(rng.choice(choices))] += 1
    if testing and not any(n % p for n, p in zip(counts, periods)):
        # Swap one surplus observation between queues or remove it. This
        # preserves at least three full cycles and guarantees a partial tail.
        donor = next(i for i in (0, 1) if counts[i] > 3*periods[i])
        counts[donor] -= 1
    return counts


def _positions(rng, size, count, tied=False):
    capacity = 29//(size+1)
    columns = int(rng.choice([c for c in range(1, capacity+1)
                              if (count+c-1)//c <= capacity]))
    rows = (count+columns-1)//columns
    minimum_h, minimum_w = rows*(size+1)+1, columns*(size+1)+1
    mode = int(rng.integers(4))
    if mode == 0:
        h = int(rng.integers(minimum_h, min(30, minimum_h+3)+1))
        w = int(rng.integers(minimum_w, min(30, minimum_w+3)+1))
    elif mode == 1:
        h = w = int(rng.integers(max(minimum_h, minimum_w), 31))
    else:
        h, w = int(rng.integers(minimum_h, 31)), int(rng.integers(minimum_w, 31))
    vertical = rng.multinomial(h-minimum_h, np.full(rows+1, 1/(rows+1)))
    horizontal = rng.multinomial(w-minimum_w, np.full(columns+1, 1/(columns+1)))
    tops, lefts = [], []
    pos = 1+int(vertical[0])
    for i in range(rows):
        tops.append(pos)
        pos += size+1+int(vertical[i+1])
    pos = 1+int(horizontal[0])
    for i in range(columns):
        lefts.append(pos)
        pos += size+1+int(horizontal[i+1])
    sites = [(r, c) for r in tops for c in lefts]
    chosen = rng.choice(len(sites), size=count, replace=False)
    positions = [sites[int(i)] for i in chosen]
    # Retain some tied rows, otherwise abandon the shelf initialization.
    if not tied:
        rr, cc = np.indices((h-size-1, w-size-1))
        rr, cc = rr+1, cc+1
        for _ in range(2):
            for index in rng.permutation(count):
                legal = np.ones(rr.shape, dtype=bool)
                for j, (r, c) in enumerate(positions):
                    if j != index:
                        legal &= (np.abs(rr-r) > size) | (np.abs(cc-c) > size)
                options = np.argwhere(legal)
                selected = options[int(rng.integers(len(options)))]
                positions[int(index)] = (int(selected[0])+1, int(selected[1])+1)
    return (h, w), sorted(positions)


def _kinds(rng, counts, grouped=False):
    kinds = [0]*counts[0] + [1]*counts[1]
    mode = 0 if grouped else int(rng.integers(3))
    if mode == 0:
        if rng.integers(2):
            kinds.reverse()
    elif mode == 1:
        rng.shuffle(kinds)
    else:
        remaining = list(counts)
        kinds = []
        first = int(rng.integers(2))
        while any(remaining):
            for kind in (first, 1-first):
                if remaining[kind]:
                    kinds.append(kind)
                    remaining[kind] -= 1
    return kinds


def _build(rng, periods, size, testing=False, witness=False, fallback=False):
    counts = [6, 6] if fallback else _counts(rng, periods, size, testing)
    background = int(rng.integers(10))
    available = [int(c) for c in rng.permutation([c for c in range(10)
                                                if c != background])]
    overlap = int(rng.integers(3))
    palettes = [available[:2], available[2-overlap:4-overlap]]
    streams = [_make_clock(rng, p, n, palette, first_error=witness)
               for i, (p, n, palette) in enumerate(zip(periods, counts, palettes))]
    if fallback:
        # Guaranteed runs and observable row ties for the period-two anchor.
        columns = 4
        positions = [(1+(i//columns)*(size+1), 1+(i % columns)*(size+1))
                     for i in range(12)]
        shape = (3*(size+1)+1, 4*(size+1)+1)
        kinds = [0]*6 + [1]*6
    else:
        shape, positions = _positions(rng, size, sum(counts), tied=witness)
        kinds = _kinds(rng, counts, grouped=witness)
    inp = np.full(shape, background, dtype=np.int8)
    ranks = [0, 0]
    # Keep an explicit construction-side clean clock for output checking.
    for (top, left), kind in zip(positions, kinds):
        color = int(streams[kind][0][ranks[kind]])
        inp[top:top+size, left:left+size] = _tile(size, color, background, kind == 0)
        ranks[kind] += 1
    joint = lcm(*periods)
    output = _inlays(streams[0][1][:joint], streams[1][1][:joint], size)
    analysis = _analyze(inp)
    _require(tuple(c['period'] for c in analysis['clocks']) == tuple(periods),
             "constructed periods do not match pixels")
    _require(np.array_equal(_answer(analysis), output), "construction/solve disagreement")
    return dict(input=inp, output=output), analysis


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer('seed', seed, 0)
    num_train = _integer('num_train', num_train, 2, 10)
    num_test = _integer('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    unequal = (2, 3) if rng.integers(2) else (3, 2)
    fourth = (2, 4) if rng.integers(2) else (4, 2)
    known = [(2, 2), unequal, tuple(reversed(unequal)), (3, 3)]
    if num_train >= 3:
        known += [fourth, tuple(reversed(fourth)), (4, 4)]
    plan = [(2, 2), unequal]
    if num_train >= 3:
        plan.append(fourth)
    plan += [known[int(rng.integers(len(known)))] for _ in range(num_train-len(plan))]
    plan += [tuple(reversed(unequal))]
    plan += [known[int(rng.integers(len(known)))] for _ in range(num_test-1)]
    puzzle, seen = dict(train=[], test=[]), set()
    first_size = None
    for index, periods in enumerate(plan):
        testing, witness = index >= num_train, index == 0
        sizes = _sizes(periods)
        if index == 1:
            sizes = [s for s in sizes if s != first_size]
        # Test choices need room for at least one surplus observation.
        if testing:
            sizes = [s for s in sizes if (29//(s+1))**2 > 3*sum(periods)]
        size = int(rng.choice(sizes))
        for attempt in range(48):
            pair, analysis = _build(rng, periods, size, testing, witness)
            if witness and not all(_witnesses(analysis).values()):
                continue
            if _key(pair['input']) not in seen:
                break
        else:
            if witness:
                # Period-two fallback is feasible at every first-anchor size.
                # Size six needs a 22x29 array, which remains legal.
                pair, analysis = _build(rng, periods, min(size, 6), fallback=True,
                                        witness=True)
                _require(all(_witnesses(analysis).values()), "fallback evidence failure")
            else:
                raise RuntimeError("distinct input construction exhausted")
        _require(_key(pair['input']) not in seen, "duplicate input")
        seen.add(_key(pair['input']))
        puzzle['test' if testing else 'train'].append(pair)
        if index == 0:
            first_size = analysis['size']
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "pair lists required")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     "exact input/output keys required")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            analysis = _analyze(inp)
            _require(np.array_equal(out, _answer(analysis)), "incorrect woven strip")
            # Pixel-level inlay and cycle-boundary invariants.
            s, joint = analysis['size'], analysis['joint']
            _require(out.shape == (s*joint, s), "wrong cycle dimensions")
            for rank in range(joint):
                tile = out[rank*s:(rank+1)*s]
                h, d = (int(c['word'][rank % c['period']]) for c in analysis['clocks'])
                _require(np.all(tile[1:-1, 1:-1] == d), "wrong donor interior")
                _require(all(np.all(edge == h) for edge in
                             (tile[0], tile[-1], tile[:, 0], tile[:, -1])), "wrong receiver rim")
            # The composite cycle has no shorter period, even if some tiles
            # look solid because the two clocks share a phase color.
            for p in range(1, joint):
                if joint % p == 0:
                    _require(any(not np.array_equal(out[i*s:(i+1)*s],
                                                     out[(i % p)*s:(i % p+1)*s])
                                 for i in range(joint)), "nonprimitive composite strip")
            (training if split == 'train' else testing).append(analysis)
    for mode in ('no_repair', 'greedy', 'horizontal', 'reverse', 'reverse_ties'):
        _require(any(_witnesses(t)[mode] for t in training), f"missing {mode} witness")
    for kind in (0, 1):
        _require(any(any(i < t['joint'] for i in t['clocks'][kind]['errors']) for t in training),
                 "both clock repairs must affect output")
    _require(any(any(i < c['period'] for i in c['errors'])
                 for t in training for c in t['clocks']), "missing first-cycle error")
    _require(any(t['joint'] > max(c['period'] for c in t['clocks']) for t in training),
             "missing joint-cycle interaction")
    _require(len({t['size'] for t in training}) >= 2, "missing scale evidence")
    # Generator-specific shift: every test retains a partial tail.
    _require(all(any(len(q) % c['period'] for q, c in zip(t['queues'], t['clocks']))
                 for t in testing), "missing partial test repetition")


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded 42-seed diversity, ablations, count coverage and negative controls."""
    puzzles, shapes, layouts, palettes, periods, sizes, counts = [], set(), set(), set(), set(), set(), set()
    ties = overlaps = partials = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input'].copy()
                prediction = solve(inp)
                _require(np.array_equal(prediction, pair['output']), "solver mismatch")
                _require(np.array_equal(inp, pair['input']), "solver mutates input")
                t = _analyze(inp)
                shapes.add(inp.shape)
                layouts.add((inp.shape, t['size'], tuple((o['top'], o['left'], o['kind'])
                                                       for o in t['objects'])))
                palettes.add((t['background'], tuple(tuple(c['word']) for c in t['clocks'])))
                periods.add(tuple(c['period'] for c in t['clocks']))
                sizes.add(t['size'])
                counts.add(tuple(map(len, t['queues'])))
                ties += any(len({o['top'] for o in q}) < len(q) for q in t['queues'])
                overlaps += bool(set(t['clocks'][0]['word']) & set(t['clocks'][1]['word']))
                partials += any(len(q) % c['period'] for q, c in zip(t['queues'], t['clocks']))
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity failure")
    _require(len(shapes) >= 25 and len(layouts) >= 180 and len(palettes) >= 180,
             "insufficient geometry/palette variation")
    _require(sizes == {3, 4, 5, 6} and len(periods) == 7 and len(counts) >= 35,
             "insufficient size/clock/count variation")
    _require(ties >= 42 and 0 < overlaps < 210 and 42 <= partials < 210,
             "insufficient ties/overlap/partial variation")
    _require(_signature(puzzles[0]) == _signature(generate_puzzle(0)), "nondeterminism")
    # Exercise every legal explicit count, including minimal demonstrations.
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1000+10*train+test, train, test)
            _require(len(p['train']) == train and len(p['test']) == test, "wrong requested counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt an actual output; the validator must reject it.
    damaged = {split: [{role: grid.copy() for role, grid in pair.items()}
                       for pair in puzzles[0][split]] for split in ('train', 'test')}
    damaged['test'][0]['output'][1, 1] = (int(damaged['test'][0]['output'][1, 1])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    # Keep perfectly solved pairs but omit the coprime training interaction.
    insufficient = dict(train=[puzzles[0]['train'][0], puzzles[1]['train'][0]],
                        test=puzzles[0]['test'])
    try:
        validate_generated_puzzle(insufficient)
    except ValueError:
        pass
    else:
        raise AssertionError("missing interaction evidence accepted")
    # Exact ambiguity/tie controls for the recovery algorithm.
    try:
        _recover([1, 2, 2, 1, 1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported clock accepted")
    clean = _recover([1, 2, 1, 2, 1, 2])
    _require(clean['period'] == 2 and not clean['errors'], "clean clock support failure")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
