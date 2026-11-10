"""Translation-only complementary tab assembly with input-derived global search."""

import hashlib

import numpy as np


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    return int(seed) % (1 << 64)


def _checked_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _resolve_counts(
    seed_value: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count(num_train, "num_train", 2, 10)
    test_count = _checked_count(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed_value % 4
        train_count = max(
            test_count + 1,
            2 + (seed_value // 4) % 9,
        )
        return int(train_count), int(test_count)

    if train_count is None:
        minimum = max(2, test_count + 1)
        train_count = int(rng.integers(minimum, 11))
        return train_count, test_count

    maximum = min(4, train_count - 1)
    if maximum < 1:
        raise ValueError("no legal num_test exists")
    test_count = int(rng.integers(1, maximum + 1))
    return train_count, test_count



_SOURCE_PAIR_DIGESTS = ('98163dee2b40de7a5a05cbd73d276a3ccd065a94a9269a2c38a37950b764d55a', '63b513c2dd22002f58ad1abee0d6e8ec0288badfcd9fca096f150b6586d28862', 'c0b0741707d352da6920931fb2aa967e4b291a533ac73c493415e8683f5a134d')

def _dilate(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    result = np.zeros_like(mask)

    for row_delta in (-1, 0, 1):
        source_row_start = max(0, -row_delta)
        source_row_end = min(height, height - row_delta)
        target_row_start = source_row_start + row_delta
        target_row_end = source_row_end + row_delta

        for column_delta in (-1, 0, 1):
            source_column_start = max(0, -column_delta)
            source_column_end = min(width, width - column_delta)
            target_column_start = (
                source_column_start + column_delta
            )
            target_column_end = source_column_end + column_delta

            result[
                target_row_start:target_row_end,
                target_column_start:target_column_end,
            ] |= mask[
                source_row_start:source_row_end,
                source_column_start:source_column_end,
            ]

    return result


def _array_key(array):
    return array.shape, array.astype(np.int8).tobytes()


def _components(grid, background):
    """Four-connected monochrome components, including repeated colors."""
    remaining = {(int(r), int(c)) for r, c in np.argwhere(grid != background)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(grid[first])
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and grid[p] == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _scene(grid):
    background = int(np.argmax(np.bincount(grid.ravel(), minlength=10)))
    components = _components(grid, background)
    bands = []
    for i, (cells, color) in enumerate(components):
        for axis in (0, 1):
            cross = 1-axis
            if len(np.unique(cells[:, cross])) == grid.shape[cross]:
                bands.append((i, axis, color))
    assert len(bands) == 1, 'expected one connected spanning band'
    index, axis, band = bands[0]
    assert all(color != band for i, (_, color) in enumerate(components) if i != index)
    canonical = grid if axis == 0 else grid.T
    mask = canonical == band
    edges = []
    for c in range(canonical.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        assert np.array_equal(rows, np.arange(rows[0], rows[-1]+1))
        edges.append((int(rows[0]), int(rows[-1])+1))
    edges = np.array(edges)
    assert np.all(edges[1:, 0] < edges[:-1, 1])
    assert np.all(edges[:-1, 0] < edges[1:, 1])
    pieces = []
    forbidden = _dilate(mask)
    occupied = np.zeros_like(mask)
    for i, (cells, color) in enumerate(components):
        if i == index:
            continue
        cells = cells if axis == 0 else cells[:, ::-1]
        assert not np.any(forbidden[cells[:, 0], cells[:, 1]]), 'piece touches band'
        assert not np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]), 'pieces touch'
        occupied[cells[:, 0], cells[:, 1]] = True
        pieces.append((cells, color))
    assert pieces
    return canonical, background, band, axis, edges, pieces


def _description(cells, sign):
    cross = cells[:, 1] - cells[:, 1].min()
    outward = sign * cells[:, 0]
    lower, upper = [], []
    for c in range(int(cross.max())+1):
        values = np.sort(outward[cross == c])
        assert len(values) and np.array_equal(values, np.arange(values[0], values[-1]+1))
        lower.append(int(values[0]))
        upper.append(int(values[-1])+1)
    origin = min(lower)
    return (tuple(x-origin for x in lower), tuple(x-origin for x in upper),
            np.column_stack((outward-origin, cross)))


def _assemblies(grid):
    """All full-width chains at ALL boundary alignments, then global exact cover.

    No sockets, colors-as-identities, intended sites, or generation metadata enter
    this search. Every terminal subset is considered. Distinct colored outputs,
    rather than component permutations, determine uniqueness. This covers the
    source family's column-convex, full-width interlocking chains, not arbitrary
    side-by-side tilings or internally holed polyominoes.
    """
    canonical, bg, band, axis, edges, pieces = _scene(grid)
    height, width = canonical.shape
    base_grid = np.full_like(canonical, bg)
    base_grid[canonical == band] = band
    candidates, starts = [], []
    for sign in (-1, 1):
        descriptions = [_description(cells, sign) for cells, _ in pieces]
        lookup = {}
        for i, (lo, hi, local) in enumerate(descriptions):
            lookup.setdefault(lo, []).append(i)
        boundary = 1-edges[:, 0] if sign == -1 else edges[:, 1]
        limit = 1 if sign == -1 else height
        for tab_width in sorted({len(d[0]) for d in descriptions}):
            for start in range(width-tab_width+1):
                initial = tuple(map(int, boundary[start:start+tab_width]))

                def extend(profile, used, placements):
                    offset = min(profile)
                    normalized = tuple(v-offset for v in profile)
                    if len(set(profile)) == 1 and used:
                        all_cells = frozenset((int(r), int(c))
                                              for _, cells in placements for r, c in cells)
                        candidates.append({'bits': used, 'placements': placements,
                                           'cells': all_cells, 'side': sign,
                                           'start': start, 'width': tab_width,
                                           'depth': profile[0]-min(initial),
                                           'initial': initial})
                    for i in lookup.get(normalized, ()):
                        if used & (1 << i):
                            continue
                        lo, hi, local = descriptions[i]
                        next_profile = tuple(offset+v for v in hi)
                        if max(next_profile) > limit:
                            continue
                        cells = np.column_stack((sign*(local[:, 0]+offset), local[:, 1]+start))
                        if np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= height):
                            continue
                        if np.any(base_grid[cells[:, 0], cells[:, 1]] != bg):
                            continue
                        # All input components really move.
                        if set(map(tuple, cells)) == set(map(tuple, pieces[i][0])):
                            continue
                        if not used:
                            starts.append((i, sign, start))
                        extend(next_profile, used | (1 << i), placements+[(i, cells)])

                extend(initial, 0, [])
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate['bits'] & (1 << i):
                by_piece[i].append(candidate)
    full = (1 << len(pieces))-1
    solutions = {}

    def cover(used, occupied, chosen):
        if len(solutions) > 1:
            return
        if used == full:
            answer = base_grid.copy()
            for candidate in chosen:
                for i, cells in candidate['placements']:
                    answer[cells[:, 0], cells[:, 1]] = pieces[i][1]
            key = answer.tobytes()
            solutions.setdefault(key, (answer, chosen))
            return
        available = [(i, [c for c in by_piece[i]
                          if not c['bits'] & used and not c['cells'] & occupied])
                     for i in range(len(pieces)) if not used & (1 << i)]
        _, options = min(available, key=lambda item: len(item[1]))
        for candidate in options:
            cover(used | candidate['bits'], occupied | candidate['cells'], chosen+[candidate])

    cover(0, frozenset(), [])
    info = {'background': bg, 'band': band, 'orientation': 'horizontal' if axis == 0 else 'vertical',
            'pieces': pieces, 'edges': edges, 'axis': axis,
            'candidate_count': len(candidates), 'local_starts': len(set(starts))}
    return list(solutions.values()), info


def _solve_from_input(grid):
    solutions, info = _assemblies(grid)
    assert len(solutions) == 1, 'assembly must have exactly one distinct colored output'
    answer, tabs = solutions[0]
    info['tabs'] = tabs
    return (answer if info['axis'] == 0 else answer.T).copy(), info


def _profile(rng, width, amplitude):
    """Asymmetric teeth, staircases, ramps and skewed walks, including shallow cuts."""
    mode = int(rng.integers(4))
    if mode == 0:
        values = rng.integers(0, amplitude+1, size=width)
    elif mode == 1:
        values = np.clip(np.cumsum(rng.integers(-1, 2, size=width)), -amplitude, amplitude)
    elif mode == 2:
        values = np.floor(np.linspace(0, amplitude, width)).astype(int)
        if rng.random() < .5:
            values = values[::-1]
    else:
        values = np.zeros(width, dtype=int)
        for _ in range(int(rng.integers(1, width))):
            a = int(rng.integers(width))
            b = int(rng.integers(a+1, width+1))
            values[a:b] = int(rng.integers(amplitude+1))
    values -= values.min()
    if values.max() > amplitude:
        values = np.rint(values * amplitude / values.max()).astype(int)
    if values.max() == 0:
        values[int(rng.integers(width))] = 1
    return values


def _interfaces(rng, width, count, richness):
    amplitude = int(rng.integers(1, 2+min(richness, 1)))
    profiles = [_profile(rng, width, amplitude)]
    for _ in range(count-1):
        current = profiles[-1]
        shape = _profile(rng, width, int(rng.integers(1, 3+richness)))
        # Positive column thickness and overlapping neighboring intervals ensure
        # connectivity while permitting one-cell necks and unequal layer heights.
        offset = max(int(np.max(current+1-shape)),
                     int(np.max(np.maximum(current[:-1], current[1:])+1-
                                np.minimum(shape[:-1], shape[1:]))))
        profiles.append(shape+offset+int(rng.integers(0, 3)))
    profiles.append(np.full(width, int(profiles[-1].max())+int(rng.integers(1, 4))))
    return profiles


def _scatter_pieces(rng, base, pieces, background):
    normalized = [(cells-cells.min(axis=0), color, cells.min(axis=0)) for cells, color in pieces]
    height, width = base.shape
    for _ in range(24):
        result = base.copy()
        forbidden = _dilate(base != background)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*8)
        success = True
        for i in order:
            cells, color, origin = normalized[i]
            h, w = cells.max(axis=0)+1
            if h > height or w > width:
                success = False
                break
            valid = np.ones((height-h+1, width-w+1), dtype=bool)
            for r, c in cells:
                valid &= ~forbidden[r:r+height-h+1, c:c+width-w+1]
            if 0 <= origin[0] < valid.shape[0] and 0 <= origin[1] < valid.shape[1]:
                valid[tuple(origin)] = False
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            target = cells+choices[int(rng.integers(len(choices)))]
            result[target[:, 0], target[:, 1]] = color
            forbidden = _dilate(result != background)
        if success:
            return result
    raise ValueError('scatter clearance')


def _palette(rng):
    bg = 0 if rng.random() < .5 else int(rng.integers(10))
    band = int(rng.choice([c for c in range(10) if c != bg]))
    payload = tuple(int(c) for c in rng.permutation([c for c in range(10) if c not in (bg, band)]))
    return bg, band, payload


def _make_pair(rng, orientation, tab_count, richness, palette, reuse):
    bg, band, palette = palette
    widths = [int(rng.integers(3, 9)) for _ in range(tab_count)]
    if tab_count > 1 and rng.random() < .65:
        widths = [widths[0]]*tab_count
    sides = [int(rng.choice([-1, 1])) for _ in range(tab_count)]
    if tab_count == 3 and len(set(sides)) == 1 and sum(widths)+4 > 30:
        sides[-1] *= -1
    minimum_cross = max(12, max(sum(w for w, s in zip(widths, sides) if s == side)+
                                   2*sum(s == side for s in sides) for side in (-1, 1)))
    if minimum_cross > 30:
        raise ValueError('cross clearance')
    size_mode = int(rng.integers(4))
    upper = (20, 25, 30, 30)[size_mode]
    cross_length = 30 if size_mode == 3 else int(rng.integers(minimum_cross, max(minimum_cross, upper)+1))
    starts = []
    reserved = {-1: set(), 1: set()}
    for width, side in zip(widths, sides):
        choices = [x for x in range(cross_length-width+1)
                   if not set(range(x-1, x+width+1)) & reserved[side]]
        if not choices:
            raise ValueError('tab clearance')
        if starts and side != sides[0] and widths[0] == width and rng.random() < .3 and starts[0] in choices:
            start = starts[0]
        else:
            start = choices[int(rng.integers(len(choices)))]
        starts.append(start)
        reserved[side].update(range(start, start+width))
    counts = [int(rng.choice([1, 2, 3, 4, 5], p=[.09, .33, .36, .18, .04])) for _ in sides]
    if sum(counts) > 10:
        raise ValueError('too many pieces to scatter')
    boundaries = [_interfaces(rng, w, n, richness) for w, n in zip(widths, counts)]
    # The entire boundary varies; unused detail is never interpreted as a command.
    edge = {}
    context_depth = int(rng.integers(1, 4))
    for side in (-1, 1):
        values = np.zeros(cross_length, dtype=int)
        for _ in range(int(rng.integers(2, 9))):
            a = int(rng.integers(cross_length))
            b = min(cross_length, a+int(rng.integers(1, 6)))
            values[a:b] = int(rng.integers(1, context_depth+1))
        edge[side] = values
    for side, start, width, cuts in zip(sides, starts, widths, boundaries):
        edge[side][start:start+width] = cuts[0]+int(rng.integers(0, 2))
        delta = int(edge[side][start]-cuts[0][0])
        for cut in cuts:
            cut += delta
    core = int(rng.integers(2, 5))
    extents = {s: int(edge[s].max()) for s in (-1, 1)}
    for side, cuts in zip(sides, boundaries):
        extents[side] = max(extents[side], int(cuts[-1][0]))
    minimum_height = max(14, core+extents[-1]+extents[1])
    if minimum_height > 30:
        raise ValueError('outward clearance')
    height = 30 if size_mode == 3 else int(rng.integers(minimum_height, max(minimum_height, upper)+1))
    spare = height-core-extents[-1]-extents[1]
    center = extents[-1]+int(rng.integers(spare+1))
    base = np.full((height, cross_length), bg, dtype=np.int8)
    for c in range(cross_length):
        base[center-edge[-1][c]:center+core+edge[1][c], c] = band
    output = base.copy()
    pieces = []
    total = sum(counts)
    palette_size = int(rng.integers(1 if reuse else min(total, 8), min(total, 8)+1))
    colors = [palette[i % palette_size] for i in rng.permutation(total)]
    for side, start, width, cuts in zip(sides, starts, widths, boundaries):
        for lower, upper_cut in zip(cuts, cuts[1:]):
            cells = np.array([(center-1-u if side == -1 else center+core+u, start+c)
                              for c in range(width) for u in range(int(lower[c]), int(upper_cut[c]))])
            color = colors[len(pieces)]
            assert np.all(output[cells[:, 0], cells[:, 1]] == bg)
            output[cells[:, 0], cells[:, 1]] = color
            pieces.append((cells, color))
    if np.count_nonzero(base == band) >= np.count_nonzero(base == bg)-sum(len(c) for c, _ in pieces):
        raise ValueError('background must remain recognizable')
    inp = _scatter_pieces(rng, base, pieces, bg)
    if rng.random() < .5:
        inp, output = inp[:, ::-1], output[:, ::-1]
    if orientation == 'vertical':
        inp, output = inp.T, output.T
    pair = {'input': inp.copy(), 'output': output.copy()}
    expected, _ = _solve_from_input(pair['input'])
    if not np.array_equal(expected, pair['output']):
        raise ValueError('construction and global search disagree')
    return pair


def _fallback_pair(rng, orientation, palette, serial):
    """Constructive safety net: one asymmetric, uniquely rooted two-piece chain."""
    bg, band, colors = palette
    width = 5+serial % 3
    start = 2+serial % (26-width)
    base = np.full((30, 30), bg, dtype=np.int8)
    base[6:10, :] = band
    lower = np.array([0, 1, 2]+[0]*(width-3))
    middle = np.array([1, 0, 1]+[0]*(width-3))+4
    for c in range(width):
        base[10:10+lower[c], start+c] = band
    output = base.copy()
    pieces = []
    for i, (lo, hi) in enumerate(((lower, middle), (middle, np.full(width, 8)))):
        cells = np.array([(10+u, start+c) for c in range(width) for u in range(lo[c], hi[c])])
        output[cells[:, 0], cells[:, 1]] = colors[i]
        pieces.append((cells, colors[i]))
    inp = base.copy()
    for i, (cells, color) in enumerate(pieces):
        origin = np.array([0, 15*i])
        scattered = cells-cells.min(axis=0)+origin
        inp[scattered[:, 0], scattered[:, 1]] = color
    if orientation == 'vertical':
        inp, output = inp.T, output.T
    pair = {'input': inp.copy(), 'output': output.copy()}
    _validate_pair(pair)
    return pair


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= x <= 30 for x in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    expected, info = _solve_from_input(inp)
    assert np.array_equal(expected, out)
    assert np.array_equal(inp == info['band'], out == info['band'])
    assert np.array_equal(np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10))
    canonical = out if info['axis'] == 0 else out.T
    used = set()
    for tab in info['tabs']:
        for i, cells in tab['placements']:
            assert i not in used
            used.add(i)
            source, color = info['pieces'][i]
            shifted = source + cells.min(axis=0)-source.min(axis=0)
            assert set(map(tuple, shifted)) == set(map(tuple, cells))
            assert np.all(canonical[cells[:, 0], cells[:, 1]] == color)
    assert used == set(range(len(info['pieces'])))
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _validate_pair(pair)
            key = _array_key(pair['input'])
            assert key not in seen
            seen.add(key)
            if pair['input'].shape == (30, 30):
                raw = pair['input'].astype(np.int8).tobytes()+pair['output'].astype(np.int8).tobytes()
                assert hashlib.sha256(raw).hexdigest() not in _SOURCE_PAIR_DIGESTS
    # The sole convention is geometric translation and preservation. Both splits
    # are validated by that identical rule. Literal colors and orientation carry
    # no hidden dictionary; new test colors require no training palette coverage.


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, rng, num_train, num_test)
    # Regime controls sample orientation coverage, never output interpretation.
    mode = int(rng.integers(5))
    stable_palette = rng.random() < .25
    shared_palette = _palette(rng)
    richness = int(rng.integers(3))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', train_count), ('test', test_count)):
        if mode == 0:
            orientations = ['horizontal']*count
        elif mode == 1:
            orientations = ['vertical']*count
        elif mode in (2, 3):
            direction = (mode == 2) == (split == 'train')
            orientations = [('horizontal' if direction else 'vertical')]*count
        else:
            orientations = ['horizontal' if i % 2 else 'vertical' for i in range(count)]
            rng.shuffle(orientations)
        tab_counts = [1, 2]+[int(rng.choice([1, 2, 3], p=[.25, .55, .2])) for _ in range(max(0, count-2))]
        rng.shuffle(tab_counts)
        for index in range(count):
            orientation = orientations[index]
            palette = shared_palette if stable_palette else _palette(rng)
            tab_count = tab_counts[index] if split == 'train' else int(rng.choice([1, 2, 3], p=[.2, .55, .25]))
            reuse = bool(rng.integers(2))
            for attempt in range(100):
                try:
                    pair = _make_pair(rng, orientation, tab_count, richness, palette, reuse)
                except (ValueError, AssertionError):
                    continue
                if _array_key(pair['input']) not in seen:
                    break
            else:
                pair = _fallback_pair(rng, orientation, palette, index+10*(split == 'test'))
            seen.add(_array_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and
               all(all(np.array_equal(a[k], b[k]) for k in ('input', 'output'))
                   for a, b in zip(left[s], right[s])) for s in ('train', 'test'))


def _features(pair):
    _, info = _solve_from_input(pair['input'])
    pieces, tabs, edges = info['pieces'], info['tabs'], info['edges']
    widths = [t['width'] for t in tabs]
    counts = [len(t['placements']) for t in tabs]
    colors = [c for _, c in pieces]
    tab_colors = [{pieces[i][1] for i, _ in tab['placements']} for tab in tabs]
    unused = 0
    for sign in (-1, 1):
        boundary = edges[:, 0] if sign == -1 else edges[:, 1]
        covered = {x for tab in tabs if tab['side'] == sign for x in range(tab['start'], tab['start']+tab['width'])}
        unused += sum(int(boundary[x] != boundary[x-1]) for x in range(1, len(boundary))
                      if x not in covered and x-1 not in covered)
    placements = {i: (tab['side'], cells) for tab in tabs for i, cells in tab['placements']}
    opposite = 0
    for i, (cells, _) in enumerate(pieces):
        sign, target = placements[i]
        cross = cells[:, 1]
        source_side = -1 if np.all(cells[:, 0] < edges[cross, 0]) else 1
        opposite += source_side != sign
    layout = 'single'
    if len(tabs) > 1:
        if len({t['side'] for t in tabs}) == 1:
            layout = 'same_side'
        else:
            negative = [t for t in tabs if t['side'] == -1][0]
            positive = [t for t in tabs if t['side'] == 1][0]
            delta = negative['start']-positive['start']
            layout = 'aligned' if delta == 0 else ('forward' if delta < 0 else 'reverse')
    depths, amplitudes, necks = [], [], []
    for cells, _ in pieces:
        lo, hi, _ = _description(cells, 1)
        depths.append(max(hi))
        amplitudes.extend((max(lo)-min(lo), max(hi)-min(hi)))
        necks.append(min(b-a for a, b in zip(lo, hi)))
    return {'shape': pair['input'].shape, 'orientation': info['orientation'],
            'roles': (info['background'], info['band']), 'palette': tuple(sorted(set(colors))),
            'band_envelope': int(edges[:, 1].max()-edges[:, 0].min()),
            'band_margin': int(min(edges[:, 0].min(), (pair['input'] if info['axis'] == 0 else pair['input'].T).shape[0]-edges[:, 1].max())),
            'tabs': len(tabs), 'widths': widths, 'counts': counts, 'depths': depths,
            'amplitudes': amplitudes, 'necks': necks, 'layout': layout,
            'tab_lengths': [t['depth'] for t in tabs], 'repeat_color': len(set(colors)) < len(colors),
            'equal_width': len(set(widths)) < len(widths), 'unused_detail': unused,
            'equal_width_different_counts': any(widths[i] == widths[j] and counts[i] != counts[j]
                                                for i in range(len(tabs)) for j in range(i)),
            'cross_tab_color_reuse': any(tab_colors[i] & tab_colors[j]
                                        for i in range(len(tabs)) for j in range(i)),
            'opposite_origins': int(opposite), 'piece_count': len(pieces),
            'edge_origin': any(np.any((cells == 0) | (cells == np.array((pair['input'] if info['axis'] == 0 else pair['input'].T).shape)-1)) for cells, _ in pieces),
            'local_starts': info['local_starts'], 'candidate_count': info['candidate_count']}


def self_test() -> None:
    # Two identical roots: one object has two destinations. Two differently
    # colored objects have two answers; two identical same-colored objects have
    # one colored answer despite two permutations of their component identities.
    fixture = np.zeros((16, 18), dtype=np.int8)
    fixture[4:7, :] = 8
    fixture[7, [3, 10]] = 8
    cap = np.array([(r, c) for c, low in enumerate((0, 1, 0)) for r in range(low, 3)])
    fixture[cap[:, 0], cap[:, 1]] = 2
    assert len(_assemblies(fixture)[0]) == 2
    fixture[cap[:, 0], cap[:, 1]+7] = 3
    assert len(_assemblies(fixture)[0]) == 2
    fixture[fixture == 3] = 2
    assert len(_assemblies(fixture)[0]) == 1
    first = generate_puzzle(7)
    assert _puzzles_equal(first, generate_puzzle(7))
    assert not _puzzles_equal(first, generate_puzzle(8))
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (6, 3))):
        puzzle = generate_puzzle(1000+i, *counts)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(1000+i, *counts))
        for split, count in zip(('train', 'test'), counts):
            assert count is None or len(puzzle[split]) == count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4), (3, 3), (4, 4)):
        try:
            generate_puzzle(7, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    features, structures, orientation_modes = [], set(), set()
    training_equal_width, training_reuse = 0, 0
    within = {'roles': 0, 'counts': 0, 'tabs': 0, 'band_envelope': 0, 'shape': 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local = []
        orientation_modes.add(tuple(tuple(sorted({_features(p)['orientation'] for p in puzzle[s]})) for s in ('train', 'test')))
        for pair in puzzle['train']+puzzle['test']:
            feature = _features(pair)
            features.append(feature)
            local.append(feature)
            # Color-normalized actual input geometry, not sampled parameters.
            palette, normalized = {}, []
            for value in pair['input'].flat:
                palette.setdefault(int(value), len(palette))
                normalized.append(palette[int(value)])
            structures.add((pair['input'].shape, tuple(normalized)))
            predicted, _ = _solve_from_input(pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
        for name in within:
            within[name] += len({str(f[name]) for f in local}) > 1
        training_equal_width += any(f['equal_width'] for f in local[:len(puzzle['train'])])
        training_reuse += any(f['repeat_color'] for f in local[:len(puzzle['train'])])
    assert len(structures) > .98*len(features)
    assert len({f['shape'] for f in features}) >= 50
    assert any(f['shape'] == (30, 30) for f in features)
    assert any(max(f['shape']) <= 22 for f in features)
    assert any(max(f['shape'])/min(f['shape']) >= 1.5 for f in features)
    h, v = ('horizontal',), ('vertical',)
    assert orientation_modes >= {(h, h), (v, v), (h, v), (v, h), (('horizontal', 'vertical'), ('horizontal', 'vertical'))}
    assert {f['tabs'] for f in features} >= {1, 2, 3}
    assert {n for f in features for n in f['counts']} >= {1, 2, 3, 4, 5}
    assert {w for f in features for w in f['widths']} >= set(range(3, 9))
    assert {f['layout'] for f in features} >= {'single', 'same_side', 'aligned', 'forward', 'reverse'}
    assert len({f['band_envelope'] for f in features}) >= 5
    assert len({f['band_margin'] for f in features}) >= 5
    assert len({d for f in features for d in f['depths']}) >= 6
    assert max(a for f in features for a in f['amplitudes']) >= 4
    assert any(1 in f['necks'] for f in features)
    assert sum(f['equal_width'] for f in features) >= 30
    assert sum(f['equal_width_different_counts'] for f in features) >= 20
    assert sum(f['cross_tab_color_reuse'] for f in features) >= 20
    assert training_equal_width >= 25 and training_reuse >= 25
    assert sum(f['repeat_color'] for f in features) >= 50
    assert sum(f['unused_detail'] > 0 for f in features) >= .8*len(features)
    assert sum(f['equal_width'] and f['repeat_color'] and max(f['amplitudes']) >= 3 and f['unused_detail'] > 0 for f in features) >= 10
    assert sum(f['opposite_origins'] for f in features) >= 100
    assert sum(f['piece_count']-f['opposite_origins'] for f in features) >= 100
    assert sum(f['edge_origin'] for f in features) >= 20
    assert sum(f['local_starts'] > f['tabs'] for f in features) >= 5
    assert all(n >= 30 for k, n in within.items() if k != 'roles')
    assert 20 <= within['roles'] <= 37  # Both stable and changing role palettes.
    # Mutation checks distinguish validity from mere successful execution.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('unchanged/mislabeled output accepted')
