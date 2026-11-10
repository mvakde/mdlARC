"""Complete the ballots, pool shape-equivalent votes, and color the cavities.

The solver reads only pixels. The cached library is geometry for sampling only;
it is never used by solve or the validator's reconstruction.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    if not cells:
        raise ValueError("empty shape")
    top = min(r for r, c in cells)
    left = min(c for r, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _extent(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


@lru_cache(maxsize=8192)
def _rotations(shape):
    result = set()
    current = _normalize(shape)
    for _ in range(4):
        result.add(current)
        current = _normalize((c, -r) for r, c in current)
    return tuple(sorted(result))


@lru_cache(maxsize=8192)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) |
                        set(_rotations(_normalize((r, -c) for r, c in shape)))))


def _canonical(shape):
    return min(_variants(_normalize(shape)))


def _complete(cells, mode):
    """0 = row gaps, 1 = column gaps, 2 = simultaneous union, no iteration."""
    cells = tuple(cells)
    result = set(cells)
    if mode in (0, 2):
        rows = defaultdict(list)
        for r, c in cells:
            rows[r].append(c)
        for r, xs in rows.items():
            result.update((r, c) for c in range(min(xs), max(xs) + 1))
    if mode in (1, 2):
        cols = defaultdict(list)
        for r, c in cells:
            cols[c].append(r)
        for c, ys in cols.items():
            result.update((r, c) for r in range(min(ys), max(ys) + 1))
    return tuple(sorted(result))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in _NEIGHBORS:
                point = (r + dr, c + dc)
                if point in todo:
                    todo.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(component)))
    return result


def _connected(cells):
    h, w = _extent(cells)
    mask = np.zeros((h, w), bool)
    for p in cells:
        mask[p] = True
    return len(_components(mask)) == 1


def _hole_free(cells):
    h, w = _extent(cells)
    mask = np.ones((h + 2, w + 2), bool)
    for r, c in cells:
        mask[r + 1, c + 1] = False
    return len(_components(mask)) == 1


def _border(array, color):
    return (np.all(array[0] == color) and np.all(array[-1] == color)
            and np.all(array[:, 0] == color) and np.all(array[:, -1] == color))


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(array):
    return array.shape, array.astype(np.int8).tobytes()


def _read_reference(grid):
    """Find two pure outline components; negative/positive polarity gives direction."""
    candidates = []
    for color in np.unique(grid):
        frames = []
        for cells in _components(grid == color):
            y, x, end_y, end_x = _bbox(cells)
            h, w = end_y - y, end_x - x
            if min(h, w) < 5 or len(cells) != 2 * h + 2 * w - 4:
                continue
            box = grid[y:end_y, x:end_x]
            if _border(box, color) and not np.any(box[1:-1, 1:-1] == color):
                frames.append((y, x, end_y, end_x))
        if len(frames) != 2:
            continue
        interiors = [grid[y + 1:ey - 1, x + 1:ex - 1] for y, x, ey, ex in frames]
        if interiors[0].shape != interiors[1].shape:
            continue
        for source_index in (0, 1):
            source, partner = interiors[source_index], interiors[1 - source_index]
            palette = set(map(int, np.unique(source)))
            if len(palette) != 2 or set(map(int, np.unique(partner))) != palette:
                continue
            carrier = int(source[0, 0])
            bg = next(iter(palette - {carrier}))
            if not _border(source, carrier) or not _border(partner, bg):
                continue
            seeds = _components(source == bg)
            if len(seeds) != 1:
                continue
            seed = seeds[0]
            positive = tuple(map(tuple, np.argwhere(partner == carrier)))
            modes = [m for m in range(3) if _complete(seed, m) == positive]
            if len(modes) != 1 or positive == seed:
                continue
            candidates.append(dict(background=bg, carrier=carrier,
                                   frame=int(color), frames=tuple(frames),
                                   source_index=source_index, mode=modes[0],
                                   seed=seed, positive=positive))
    if len(candidates) != 1:
        raise ValueError("reference must uniquely demonstrate row, column, or both-axis completion")
    return candidates[0]


def _read_scene(input_grid):
    grid = _grid(input_grid)
    data = _read_reference(grid)
    bg, carrier = data['background'], data['carrier']
    exterior = grid.copy()
    for y, x, ey, ex in data['frames']:
        exterior[y:ey, x:ex] = bg
    components = _components(exterior == carrier)
    if len(components) != 1:
        raise ValueError("one connected rectangular target carrier is required")
    box = _bbox(components[0])
    y, x, ey, ex = box
    stencil = exterior[y:ey, x:ex].copy()
    if (min(stencil.shape) < 3 or not _border(stencil, carrier)
            or set(map(int, np.unique(stencil))) != {bg, carrier}):
        raise ValueError("invalid target stencil")
    cavities = _components(stencil == bg)
    for cells in cavities:
        if any(r in (0, stencil.shape[0] - 1) or c in (0, stencil.shape[1] - 1)
               for r, c in cells):
            raise ValueError("cavities must be enclosed")
    exterior[y:ey, x:ex] = bg
    if np.any(exterior == data['frame']):
        raise ValueError("frame color occurs outside the reference")
    objects = []
    for cells in _components(exterior != bg):
        colors = {int(exterior[p]) for p in cells}
        if len(colors) != 1 or colors & {bg, carrier, data['frame']}:
            raise ValueError("ballots must be separate monochromatic objects")
        objects.append((colors.pop(), _normalize(cells), cells))
    if not objects:
        raise ValueError("no ballots")
    data.update(grid=grid, bbox=box, stencil=stencil, cavities=cavities,
                objects=objects, classes={_canonical(c) for c in cavities})
    return data


def _predict(data, *, mode=None, grouping='shape', weights='objects', rotations_only=False):
    mode = data['mode'] if mode is None else mode
    def signature(cells):
        shape = _normalize(cells)
        if grouping == 'area':
            return len(shape)
        if grouping == 'box':
            return tuple(sorted(_extent(shape)))
        return min(_rotations(shape)) if rotations_only else _canonical(shape)
    votes = defaultdict(Counter)
    for color, raw, cells in data['objects']:
        completed = raw if mode == -1 else _complete(raw, mode)
        votes[signature(completed)][color] += len(raw) if weights == 'pixels' else 1
    keys = {signature(c) for c in data['cavities']}
    # Alternatives may ignore unmatched ballots; the intended rule may not.
    if grouping == 'shape' and weights == 'objects' and not rotations_only and mode == data['mode']:
        if set(votes) != keys:
            raise ValueError("every completed ballot must match a cavity, and every cavity needs votes")
    winners = {}
    for key in keys:
        counts = votes.get(key, {})
        maximum = max(counts.values(), default=0)
        best = [c for c, n in counts.items() if n == maximum]
        if len(best) != 1:
            raise ValueError("missing support or tied plurality")
        winners[key] = best[0]
    output = data['stencil'].copy()
    for cells in data['cavities']:
        color = winners[signature(cells)]
        for p in cells:
            output[p] = color
    return output, dict(votes=dict(votes), winners=winners)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the completion from the input, then pool votes and crop the stencil."""
    return _predict(_read_scene(input_grid))[0]


def _fails(data, expected, **kwargs):
    try:
        answer = _predict(data, **kwargs)[0]
    except ValueError:
        return True
    return not np.array_equal(answer, expected)


@lru_cache(maxsize=1)
def _sampling_library():
    """Connected 3x4 polyominoes plus parameterized rectangular completions.

    This finite sampling vocabulary is not a solver restriction. Enumerating
    geometries rather than hand-picking a few masks gives many aliases per key.
    """
    shapes = set()
    for bits in range(1, 1 << 12):
        n = bits.bit_count()
        if not 4 <= n <= 11:
            continue
        s = tuple(divmod(i, 4) for i in range(12) if bits >> i & 1)
        s = _normalize(s)
        if _connected(s):
            shapes.add(s)
            shapes.add(_normalize((c, r) for r, c in s))
    for h in range(2, 5):
        for w in range(3, 7):
            u = {(r, c) for r in range(h) for c in (0, w - 1)}
            u.update((h - 1, c) for c in range(w))
            for inside in (None, (0, 1), (h - 2, w - 2)):
                raw = tuple(sorted(u | ({inside} if inside is not None else set())))
                shapes.update(_variants(raw))
    tables = [defaultdict(set) for _ in range(3)]
    references = []
    for raw in sorted(shapes):
        outputs = [_complete(raw, mode) for mode in range(3)]
        if (max(_extent(raw)) <= 4 and len(set(outputs)) == 3
                and all(s != raw for s in outputs)):
            references.append(raw)
        for mode, full in enumerate(outputs):
            key = _canonical(full)
            if (_canonical(raw) != key and _hole_free(full)
                    and _complete(full, mode) == full):
                tables[mode][key].add(raw)
    tables = tuple({k: tuple(sorted(v)) for k, v in table.items()
                    if len({_canonical(s) for s in v}) >= 2} for table in tables)
    return tables, tuple(references)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _full_forms(key, mode):
    return [s for s in _variants(key) if _complete(s, mode) == s]


def _alias_pair(aliases):
    ordered = sorted(aliases, key=lambda s: (len(s), s))
    first = ordered[0]
    second = next(s for s in ordered if _canonical(s) != _canonical(first))
    return first, second


def _ballots(rng, keys, mode, colors, teaching=0):
    tables, refs = _sampling_library()
    winners, specs, cavity_anchor = {}, [], None
    for index, key in enumerate(keys):
        winner = colors[index] if teaching else _choice(rng, colors[:min(3, len(colors))])
        winners[key] = winner
        losers = [c for c in colors if c != winner]
        rng.shuffle(losers)
        aliases = tables[mode][key]
        full = _choice(rng, _full_forms(key, mode))
        if teaching == 2 and index == 0:
            # Opposite handedness wins; a direct same-handed minority loses.
            cavity_anchor = full
            opposite = [s for s in aliases if min(_rotations(_normalize(_complete(s, mode))))
                        != min(_rotations(full))]
            a, b = _alias_pair(opposite)
            specs.extend([(a, winner), (b, winner), (a, winner), (full, losers[0])])
        elif teaching == 1 and index == 0:
            # Rectangle 3x4: four small seeds beat three full forms. The full
            # forms have more colored pixels; component counting is essential.
            a, b = _alias_pair(aliases)
            specs.extend([(a, winner)] * 3 + [(b, winner)] +
                         [(full, losers[0])] * 3 + [(_choice(rng, aliases), losers[1])])
        else:
            counts = (2, 1) if teaching else _choice(rng, ((1,), (2,), (2, 1), (3, 1), (2, 1, 1), (3, 2)))
            a, b = _alias_pair(aliases)
            for j, count in enumerate(counts):
                color = winner if j == 0 else losers[j - 1]
                for vote in range(count):
                    raw = (a if vote % 2 == 0 else b) if j == 0 else full
                    specs.append((raw, color))
    return specs, winners, cavity_anchor


def _positions(available, shape):
    h, w = _extent(shape)
    rows, cols = available.shape[0] - h + 1, available.shape[1] - w + 1
    if min(rows, cols) <= 0:
        return []
    valid = np.ones((rows, cols), bool)
    for r, c in shape:
        valid &= available[r:r + rows, c:c + cols]
    return np.argwhere(valid)


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = False


def _pack(available, forms, rng):
    for i in rng.permutation(len(forms)):
        raw = forms[int(i)]
        positions = _positions(available, raw)
        if len(positions):
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((r + y, c + x) for r, c in raw)
            _block(available, cells)
            return cells
    return None


def _outline(grid, box, color):
    y, x, ey, ex = box
    grid[y, x:ex] = grid[ey - 1, x:ex] = color
    grid[y:ey, x] = grid[y:ey, ex - 1] = color


def _render(rng, mode, reference, keys, specs, winners, palette,
            repetitions=0, cavity_anchor=None, compact=False):
    bg, carrier, frame = palette[:3]
    rh, rw = _extent(reference)
    for attempt in range(120):
        # The source and prototype have the same coordinate system, padded so
        # source-hole versus positive-prototype polarity is visible.
        ih, iw = rh + 2 + int(rng.integers(2)), rw + 2 + int(rng.integers(2))
        ph, pw = ih + 2, iw + 2
        gap = int(rng.integers(1, 4))
        height = int(rng.integers(24, 31)) if attempt < 20 else 30
        width = max(2 * pw + gap, int(rng.integers(25, 31))) if attempt < 20 else 30
        if width > 30:
            continue
        # Freely packed cavities, without a fixed slot lattice.
        sh = int(rng.integers(7, 13))
        sw = int(rng.integers(10, 18))
        if compact or attempt >= 60:
            sh, sw = 8 + max(0, len(keys) - 2), 16
        available = np.ones((sh, sw), bool)
        available[[0, -1]] = False
        available[:, [0, -1]] = False
        cavities, failed = [], False
        order = sorted(enumerate(keys), key=lambda p: (-len(p[1]), p[0]))
        for index, key in order:
            forms = [cavity_anchor] if cavity_anchor is not None and index == 0 else _variants(key)
            cells = _pack(available, forms, rng)
            if cells is None:
                failed = True
                break
            cavities.append((key, cells))
        if failed:
            continue
        for _ in range(repetitions):
            key = _choice(rng, keys)
            cells = _pack(available, _variants(key), rng)
            if cells is not None:
                cavities.append((key, cells))
        stencil = np.full((sh, sw), carrier, np.int8)
        answer = stencil.copy()
        for key, cells in cavities:
            for point in cells:
                stencil[point], answer[point] = bg, winners[key]
        if len(_components(stencil == carrier)) != 1:
            continue
        grid = np.full((height, width), bg, np.int8)
        top = int(rng.integers(0, 3))
        left = int(rng.integers(width - 2 * pw - gap + 1))
        frames = [(top, left, top + ph, left + pw),
                  (top, left + pw + gap, top + ph, left + 2 * pw + gap)]
        before = np.full((ih, iw), carrier, np.int8)
        after = np.full((ih, iw), bg, np.int8)
        # Offset is shared by both panels, not inferred from object origins.
        oy, ox = int(rng.integers(1, ih - rh)), int(rng.integers(1, iw - rw))
        for r, c in reference:
            before[r + oy, c + ox] = bg
        for r, c in _complete(reference, mode):
            after[r + oy, c + ox] = carrier
        contents = [before, after]
        if rng.integers(2):
            contents.reverse()
        for box, content in zip(frames, contents):
            y, x, ey, ex = box
            _outline(grid, box, frame)
            grid[y + 1:ey - 1, x + 1:ex - 1] = content
        sy = int(rng.integers(top + ph + 1, height - sh + 1)) if top + ph + 1 < height - sh + 1 else height - sh
        if sy <= top + ph:
            continue
        sx = int(rng.integers(width - sw + 1))
        grid[sy:sy + sh, sx:sx + sw] = stencil
        available = np.ones(grid.shape, bool)
        for y, x, ey, ex in frames + [(sy, sx, sy + sh, sx + sw)]:
            available[max(0, y - 1):min(height, ey + 1), max(0, x - 1):min(width, ex + 1)] = False
        # Sample actual vote locations, retaining orientation: axes apply BEFORE
        # shape canonicalization, so an arbitrary local rotation is not legal.
        ordered_specs = [specs[int(i)] for i in rng.permutation(len(specs))]
        ordered_specs.sort(key=lambda p: (max(_extent(p[0])), len(p[0])), reverse=True)
        for raw, color in ordered_specs:
            cells = _pack(available, (raw,), rng)
            if cells is None:
                failed = True
                break
            for point in cells:
                grid[point] = color
        if failed:
            continue
        return {'input': grid, 'output': answer}
    raise RuntimeError("bounded packing exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    tables, references = _sampling_library()
    modes = list(map(int, rng.permutation(3)))
    # A shared quarter-turn keeps the held-out axis genuinely absent from
    # training. Per-pair reflections still vary handedness and placement.
    turns = int(rng.integers(4))
    common = set(tables[0]) & set(tables[1]) & set(tables[2])
    # The second demonstration separates exact shape, box, and handedness.
    teaching_pairs = []
    for a, b in combinations(sorted(common), 2):
        if (len(a) == len(b) and sorted(_extent(a)) == sorted(_extent(b))
                and len(_rotations(a)) < len(_variants(a))):
            teaching_pairs.append((a, b))
    if not teaching_pairs:
        raise AssertionError('no exact-shape teaching pair')
    second_keys = list(_choice(rng, teaching_pairs))
    rectangle34 = _canonical(tuple((r, c) for r in range(3) for c in range(4)))
    rectangle26 = _canonical(tuple((r, c) for r in range(2) for c in range(6)))
    puzzle, seen, known, first_winner = {'train': [], 'test': []}, set(), set(), None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = i + 1 if split == 'train' and i < 2 else 0
            mode = modes[i % 2] if split == 'train' else modes[2]
            if teaching == 1:
                keys = [rectangle34, rectangle26]
            elif teaching == 2:
                keys = second_keys
            else:
                pool = sorted(tables[mode])
                n = int(rng.integers(1, 4)) if split == 'train' else int(rng.integers(3, 5))
                selected = [pool[int(j)] for j in rng.choice(len(pool), n, replace=False)]
                if split == 'test':
                    novel = [k for k in pool if k not in known]
                    if novel:
                        new_key = _choice(rng, novel)
                        rest = [k for k in pool if k != new_key]
                        selected = [new_key] + [rest[int(j)] for j in
                                                rng.choice(len(rest), n - 1, replace=False)]
                keys = list(dict.fromkeys(selected))
            for _ in range(12):
                palette = list(map(int, rng.permutation(10)))
                colors = palette[3:]
                if teaching == 2 and colors[0] == first_winner:
                    colors[0], colors[1] = colors[1], colors[0]
                specs, winners, anchor = _ballots(rng, keys, mode, colors, teaching)
                # Four-class scenes use modest ballots to fit without changing
                # the voting rule. All keys and teaching witnesses are retained.
                if len(keys) == 4:
                    specs = [(raw, color) for key in keys
                             for raw, color in [(tables[mode][key][0], winners[key]),
                                               (tables[mode][key][-1], winners[key])]]
                reference = _choice(rng, references)
                pair = _render(rng, mode, reference, keys, specs, winners, palette,
                               repetitions=int(rng.integers(0, 4)), cavity_anchor=anchor)
                flip = bool(rng.integers(2))
                for role in ('input', 'output'):
                    array = np.rot90(pair[role], turns)
                    if flip:
                        array = array[:, ::-1]
                    pair[role] = array.copy()
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('could not construct a distinct input')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                known.update(keys)
                if teaching == 1:
                    first_winner = winners[keys[0]]
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            data = _read_scene(inp)
            expected, ballot = _predict(data)
            assert np.array_equal(out, expected), 'wrong vote, cavity fill, or crop'
            mask = data['stencil'] == data['carrier']
            assert np.array_equal(out[mask], data['stencil'][mask]), 'carrier changed'
            assert not np.any(out == data['background']), 'unfilled cavity'
            for cells in data['cavities']:
                assert len({int(out[p]) for p in cells}) == 1, 'nonuniform fill'
            # Every ballot is a single connected monochromatic component, every
            # completed class is supported, and all pluralities are unique.
            data.update(ballot)
            analyses[split].append((data, out))
    training = analyses['train']
    assert any(_fails(d, out, mode=-1) for d, out in training), 'completion can be bypassed'
    assert any(_fails(d, out, grouping='area') for d, out in training), 'area shortcut unrefuted'
    assert any(_fails(d, out, grouping='box') for d, out in training), 'box shortcut unrefuted'
    assert any(_fails(d, out, weights='pixels') for d, out in training), 'pixel weighting unrefuted'
    assert any(_fails(d, out, rotations_only=True) for d, out in training), 'reflection untaught'
    assert any(len(set(d['winners'].values())) > 1 for d, out in training), 'global color shortcut unrefuted'
    assert any(max(v.values()) * 2 <= sum(v.values()) and len(v) >= 3
               for d, out in training for v in d['votes'].values()), 'nonmajority plurality untaught'
    assert any(len({_canonical(raw) for color, raw, cells in d['objects']
                    if _canonical(_complete(raw, d['mode'])) == key}) > 1
               for d, out in training for key in d['classes']), 'alias pooling absent'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _counterfactual_test():
    # Identical stencil and ballots, different reference partner. Rows route
    # red to A and blue to B; columns route those same votes in reverse.
    seed = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 2), (2, 3))
    transposed = _normalize((c, r) for r, c in seed)
    a, b = _canonical(_complete(seed, 0)), _canonical(_complete(seed, 1))
    assert a != b
    specs = [(seed, 3)] * 3 + [(transposed, 4)] * 3
    pairs, records = [], []
    for mode, winners in ((0, {a: 3, b: 4}), (1, {a: 4, b: 3})):
        pair = _render(np.random.default_rng(90210), mode, seed, [a, b], specs,
                       winners, list(range(10)), compact=True)
        data = _read_scene(pair['input'])
        assert np.array_equal(solve(pair['input']), pair['output'])
        pairs.append(pair)
        records.append(data)
    mask = np.ones(pairs[0]['input'].shape, bool)
    positive_frame = records[0]['frames'][1 - records[0]['source_index']]
    y, x, ey, ex = positive_frame
    mask[y:ey, x:ex] = False
    assert np.array_equal(pairs[0]['input'][mask], pairs[1]['input'][mask])
    assert not np.array_equal(pairs[0]['output'], pairs[1]['output'])
    # Keep the reference fixed and remove one red vote: the visible vote count,
    # rather than a cached shape-color dictionary, must change the result.
    specs2 = [(seed, 3)] + [(seed, 4)] * 2 + [(transposed, 5)]
    pair = _render(np.random.default_rng(13), 0, seed, [a, b], specs2,
                   {a: 4, b: 5}, list(range(10)), compact=True)
    assert np.array_equal(solve(pair['input']), pair['output'])
    tied = _render(np.random.default_rng(14), 0, seed, [a, b],
                   [(seed, 3), (seed, 4), (transposed, 5)],
                   {a: 3, b: 5}, list(range(10)), compact=True)
    try:
        solve(tied['input'])
    except ValueError:
        pass
    else:
        raise AssertionError('tied votes accepted')
    # Union completion is simultaneous. A new row-fill cell must not launch
    # an additional column fill in the same pass.
    s = ((0, 0), (0, 2), (2, 1))
    assert (1, 1) not in _complete(s, 2)


def self_test():
    puzzles, canvases, crops, vocab, modes, profiles, palettes, repetitions = set(), set(), set(), set(), set(), set(), set(), set()
    alias_counts, total_ballots, edges = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        training_data = [_read_scene(p['input']) for p in puzzle['train']]
        test_data = [_read_scene(p['input']) for p in puzzle['test']]
        train_modes = {d['mode'] for d in training_data}
        assert len(train_modes) == 2
        assert all(d['mode'] not in train_modes for d in test_data)
        known = set().union(*(d['classes'] for d in training_data))
        assert all(d['classes'] - known for d in test_data)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read_scene(pair['input'])
                out, ballot = _predict(data)
                canvases.add(pair['input'].shape)
                crops.add(out.shape)
                vocab.update(data['classes'])
                modes.add(data['mode'])
                profiles.add((len(data['classes']), len(data['cavities'])))
                palettes.add((data['background'], data['carrier'], data['frame']))
                repetitions.add(len(data['cavities']) - len(data['classes']))
                total_ballots.add(len(data['objects']))
                for key in data['classes']:
                    alias_counts.add(len({_canonical(raw) for color, raw, cells in data['objects']
                                          if _canonical(_complete(raw, data['mode'])) == key}))
                y, x, ey, ex = data['bbox']
                edges.add((y == 0, x == 0, ey == data['grid'].shape[0], ex == data['grid'].shape[1]))
    assert len(puzzles) == 42
    assert len(canvases) >= 15 and len(crops) >= 25
    assert len(vocab) >= 30 and modes == {0, 1, 2}
    assert len(profiles) >= 10 and len(palettes) >= 100
    assert max(repetitions) >= 2 and max(alias_counts) >= 2
    assert len(total_ballots) >= 8 and any(any(e) for e in edges)
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    for nt, ne in ((2, 1), (10, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(123 + nt, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('incorrect output accepted')
    _counterfactual_test()
