"""Enclosure stencil puzzle generator."""
import numpy as np

_SOURCE_INPUTS = {
    '000000000000/000000000000/002222222200/002000000200/002003333333/002000000200/002000000200/333333000200/002000000200/002222222200/000000000000/000000000000',
    '000010000000/000014000000/000010400000/000410040000/004010004000/040010111111/004000004000/000411140000/000040400010/000004000010/000000000010/000000000010',
    '000006000006/077777770060/077777770600/077000770000/077666666660/077000770000/077000770000/077060770000/077600770000/076777770000/067777770000/600000000000',
    '000000000090/000000000090/088888880090/080099998090/088000008800/008809990800/008800000080/008099999080/008800000880/000888888890/000000000090/000000000090',
    '0000020000000/0000020000000/0011121111100/0010000000100/0010000000100/0010000000100/2220000000222/0010000000100/0010000000100/0011112111100/0000002000000/0000002000000',
}


def _edge_mask(filled):
    up = np.zeros_like(filled)
    down = np.zeros_like(filled)
    left = np.zeros_like(filled)
    right = np.zeros_like(filled)
    up[1:] = filled[:-1]
    down[:-1] = filled[1:]
    left[:, 1:] = filled[:, :-1]
    right[:, :-1] = filled[:, 1:]
    return filled & ~(up & down & left & right)


def _outside(structure):
    h, w = structure.shape
    seen = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in (0, w - 1):
            if not structure[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not structure[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < h and 0 <= nc < w and not structure[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _geometry(rng, kind):
    if kind == 'diamond':
        radius = int(rng.integers(4, 9))
        top, bottom, left, right = (int(rng.integers(1, 5)) for _ in range(4))
        h, w = 2*radius+1+top+bottom, 2*radius+1+left+right
        rr, cc = np.indices((h, w))
        dist = abs(rr-(top+radius)) + abs(cc-(left+radius))
        return dist == radius, dist < radius
    minimum = 12 if kind == 'irregular' else (10 if kind == 'thick' else 8)
    h = max(minimum+2, int(rng.integers(11, 31)))
    w = max(minimum+2, int(rng.integers(11, 31)))
    top = int(rng.integers(1, min(5, h-minimum)))
    bottom = int(rng.integers(1, min(5, h-top-minimum+1)))
    left = int(rng.integers(1, min(5, w-minimum)))
    right = int(rng.integers(1, min(5, w-left-minimum+1)))
    r0, r1, c0, c1 = top, h-bottom-1, left, w-right-1
    rr, cc = np.indices((h, w))
    if kind in ('thin', 'thick'):
        t = int(rng.integers(2, 4)) if kind == 'thick' else 1
        box = (r0 <= rr) & (rr <= r1) & (c0 <= cc) & (cc <= c1)
        outline = box & ((rr-r0 < t) | (r1-rr < t) | (cc-c0 < t) | (c1-cc < t))
        return outline, box & ~outline
    # Row-convex stepped contour with multiple independently moving sides.
    row_count = r1-r0+1
    change_count = min((row_count-3)//3, int(rng.integers(3, 7)))
    slack = row_count-2-(2+3*(change_count-1))
    shifts = np.sort(rng.integers(0, slack+1, size=change_count))
    changes = set(2+3*k+int(shifts[k]) for k in range(change_count))
    offset = min(3, (c1-c0-5)//2)
    lo = c0 + int(rng.integers(0, offset+1))
    hi = c1 - int(rng.integers(0, offset+1))
    filled = np.zeros((h, w), dtype=bool)
    for k, r in enumerate(range(r0, r1+1)):
        if k in changes:
            lo = c0 + int(rng.choice([x for x in range(offset+1) if c0+x != lo]))
            hi = c1 - int(rng.choice([x for x in range(offset+1) if c1-x != hi]))
        filled[r, lo:hi+1] = True
    filled = np.rot90(filled, int(rng.integers(4)))
    if rng.integers(2):
        filled = np.fliplr(filled)
    outline = _edge_mask(filled)
    return outline, filled & ~outline


def _interior_bar(rng, interior):
    runs = []
    for r in range(interior.shape[0]):
        cols = np.flatnonzero(interior[r])
        if not len(cols):
            continue
        start = last = int(cols[0])
        for col in cols[1:]:
            col = int(col)
            if col != last+1:
                if last-start >= 1:
                    runs.append((r, start, last))
                start = col
            last = col
        if last-start >= 1:
            runs.append((r, start, last))
    r, first, final = runs[int(rng.integers(len(runs)))]
    left = int(rng.integers(first, final))
    right = min(final, left+int(rng.integers(2, 7))-1)
    if right == left:
        left -= 1
    return [(r, c) for c in range(left, right+1)]


def _exterior_bar(rng, outline):
    h, w = outline.shape
    side = int(rng.integers(4))
    if side < 2:
        r = 0 if side == 0 else h-1
        c = int(rng.integers(1, w-2))
        return [(r, x) for x in range(c, min(w-1, c+int(rng.integers(2, 6))))]
    c = 0 if side == 2 else w-1
    r = int(rng.integers(1, h-2))
    return [(x, c) for x in range(r, min(h-1, r+int(rng.integers(2, 6))))]


def _crossing(rng, outline, interior, protected):
    h, w = outline.shape
    rows = [r for r in range(h) if np.any(interior[r])]
    for _ in range(80):
        r = int(rng.choice(rows))
        inner = np.flatnonzero(interior[r])
        if rng.integers(2):
            start = int(rng.integers(0, max(1, np.flatnonzero(outline[r])[0])))
            end = min(int(inner[-1]), int(inner[0])+int(rng.integers(0, 4)))
        else:
            start = max(int(inner[0]), int(inner[-1])-int(rng.integers(0, 4)))
            end = int(rng.integers(np.flatnonzero(outline[r])[-1]+1, w))
        cells = [(r, c) for c in range(min(start, end), max(start, end)+1)]
        if not any(protected[x] for x in cells):
            return cells
    return []


def _diagonal(rng, outline, interior, protected):
    h, w = outline.shape
    for _ in range(160):
        length = int(rng.integers(4, min(h, w, 17)+1))
        r = int(rng.integers(0, h-length+1))
        dc = 1 if rng.integers(2) else -1
        c = int(rng.integers(0, w-length+1)) if dc == 1 else int(rng.integers(length-1, w))
        cells = [(r+k, c+k*dc) for k in range(length)]
        b = sum(bool(outline[x]) for x in cells)
        i = sum(bool(interior[x]) for x in cells)
        if 1 <= b <= 5 and i >= 1 and length-b-i >= 1 and not any(protected[x] for x in cells):
            return cells
    return []


def _tangent(rng, outline, interior, protected):
    boundary = list(zip(*np.where(outline)))
    rng.shuffle(boundary)
    h, w = outline.shape
    for r, c in boundary:
        if protected[r, c]:
            continue
        neighbors = [(r+dr, c+dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        outside = [(x, y) for x, y in neighbors if 0 <= x < h and 0 <= y < w and not outline[x, y] and not interior[x, y]]
        if outside:
            return [(r, c), outside[int(rng.integers(len(outside)))]]
    return []


def _irregular_intervals(filled):
    rows = np.flatnonzero(np.any(filled, axis=1))
    if len(rows) < 8 or len(rows) != rows[-1]-rows[0]+1:
        return None
    intervals = []
    for r in rows:
        occupied = np.flatnonzero(filled[r])
        if len(occupied) < 5 or len(occupied) != occupied[-1]-occupied[0]+1:
            return None
        intervals.append((int(occupied[0]), int(occupied[-1])))
    turns = sum(a != b for a, b in zip(intervals, intervals[1:]))
    if turns < 3:
        return None
    return rows, intervals, turns


def _irregular_orientation(filled):
    options = []
    for axis, view in ((0, filled), (1, filled.T)):
        data = _irregular_intervals(view)
        if data is not None:
            options.append((data[2], axis))
    assert options
    return max(options)[1]


def _protected_turns(outline, interior):
    filled = outline | interior
    axis = _irregular_orientation(filled)
    view = outline if axis == 0 else outline.T
    rows, intervals, _ = _irregular_intervals(filled if axis == 0 else filled.T)
    protected = np.zeros_like(view)
    protected[rows[0]] = view[rows[0]]
    protected[rows[-1]] = view[rows[-1]]
    for k in range(1, len(rows)):
        if intervals[k] != intervals[k-1]:
            protected[rows[k-1]] = view[rows[k-1]]
            protected[rows[k]] = view[rows[k]]
    return protected if axis == 0 else protected.T


def _make_pair(rng, kind, palette, regime, emphasis):
    background, structural, secondary = palette
    outline, interior = _geometry(rng, kind)
    h, w = outline.shape
    marker = np.zeros((h, w), dtype=bool)
    protected = _protected_turns(outline, interior) if kind == 'irregular' else np.zeros_like(outline)
    if emphasis == 'cross':
        recipes = ['cross', 'inside', 'outside']
    elif emphasis == 'no_boundary':
        recipes = ['inside', 'outside']
    elif emphasis == 'no_inside':
        recipes = ['tangent', 'outside']
    else:
        recipes = [str(rng.choice(('cross', 'inside', 'outside', 'diag', 'tangent'))) for _ in range(int(rng.integers(2, 5)))]
        if regime == 'train' and 'inside' not in recipes:
            recipes.append('inside')
    if emphasis == 'cross' and rng.integers(2):
        recipes.append('diag')
    cap = max(3, min(9, int(outline.sum()*.22)))
    for recipe in recipes:
        if recipe == 'cross':
            cells = _crossing(rng, outline, interior, protected)
        elif recipe == 'inside':
            cells = _interior_bar(rng, interior)
        elif recipe == 'outside':
            cells = _exterior_bar(rng, outline)
        elif recipe == 'diag':
            cells = _diagonal(rng, outline, interior, protected)
        else:
            cells = _tangent(rng, outline, interior, protected)
        candidate = marker.copy()
        for r, c in cells:
            if (r, c) not in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
                candidate[r, c] = True
        if int((candidate & outline).sum()) <= cap and not np.any(candidate & protected):
            marker = candidate
    if not np.any(marker & ~interior):
        for r, c in _exterior_bar(rng, outline):
            marker[r, c] = True
    inp = np.full((h, w), background, dtype=np.int8)
    out = inp.copy()
    inp[outline] = structural
    inp[marker] = secondary
    out[outline] = structural
    out[marker & interior] = secondary
    return {'input': inp, 'output': out}


def _valid_count(value, low, high):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and low <= value <= high


def _signature(pair):
    return pair['input'].shape, pair['input'].tobytes(), pair['output'].tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None and not _valid_count(num_train, 2, 10):
        raise ValueError('num_train must be 2..10')
    if num_test is not None and not _valid_count(num_test, 1, 4):
        raise ValueError('num_test must be 1..4')
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    elif num_train is None:
        num_train = int(rng.integers(num_test+1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    variable_palette = bool(rng.integers(2))
    base = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    source_shift = bool(rng.integers(4) == 0)
    result = {'train': [], 'test': []}
    seen = set()
    kinds = ('thin', 'thick', 'diamond', 'irregular')
    for regime, count in (('train', num_train), ('test', num_test)):
        for j in range(count):
            for attempt in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                if source_shift and regime == 'train' and j < 2:
                    kind = ('thin', 'diamond')[j]
                elif source_shift and regime == 'test' and j < 2:
                    kind = ('thick', 'irregular')[j]
                elif regime == 'train' and j == 0:
                    kind = str(pair_rng.choice(('thin', 'thick', 'diamond')))
                else:
                    kind = str(pair_rng.choice(kinds))
                if variable_palette and (regime != 'train' or j > 0):
                    if pair_rng.integers(4) == 0:
                        palette = (base[0], base[2], base[1])
                    else:
                        palette = tuple(int(x) for x in pair_rng.choice(10, size=3, replace=False))
                else:
                    palette = base
                if regime == 'train':
                    emphasis = 'cross' if j == 0 else str(pair_rng.choice(('cross', 'no_boundary', 'mixed')))
                else:
                    emphasis = str(pair_rng.choice(('cross', 'no_boundary', 'no_inside', 'mixed')))
                pair = _make_pair(pair_rng, kind, palette, regime, emphasis)
                sig = _signature(pair)
                if sig not in seen:
                    seen.add(sig)
                    result[regime].append(pair)
                    break
            else:
                raise RuntimeError('could not make a unique pair')
    validate_generated_puzzle(result)
    return result


def _shape_kind(structure, outside):
    rows, cols = np.where(structure)
    assert len(rows) > 0
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    h, w = structure.shape
    assert min(r0, c0, h-r1-1, w-c1-1) >= 1
    rr, cc = np.indices((h, w))
    box = (r0 <= rr) & (rr <= r1) & (c0 <= cc) & (cc <= c1)
    for t in (1, 2, 3):
        ring = box & ((rr-r0 < t) | (r1-rr < t) | (cc-c0 < t) | (c1-cc < t))
        if np.array_equal(structure, ring):
            return 'thick' if t > 1 else 'thin'
    if r1-r0 == c1-c0 and (r1-r0) % 2 == 0:
        radius = (r1-r0)//2
        diamond = abs(rr-(r0+radius)) + abs(cc-(c0+radius)) == radius
        if np.array_equal(structure, diamond):
            return 'diamond'
    filled = ~outside
    assert np.array_equal(structure, _edge_mask(filled))
    _irregular_orientation(filled)
    assert np.any(filled & ~structure)
    return 'irregular'


def _inspect_pair(pair):
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape
    colors = set(int(x) for x in np.unique(inp))
    background = int(out[0, 0])
    assert all(int(out[r, c]) == background for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    assert len(colors) == 3 and background in colors
    candidates = []
    for structural in colors-{background}:
        secondary = next(iter(colors-{background, structural}))
        structure = out == structural
        try:
            outside = _outside(structure)
            kind = _shape_kind(structure, outside)
            interior = ~outside & ~structure
            assert int(interior.sum()) >= 3
            assert np.all((inp[structure] == structural) | (inp[structure] == secondary))
            assert not np.any(inp[~structure] == structural)
            expected = np.full(inp.shape, background, dtype=inp.dtype)
            expected[structure] = structural
            expected[(inp == secondary) & interior] = secondary
            assert np.array_equal(out, expected)
            if kind == 'irregular':
                assert not np.any((inp == secondary) & _protected_turns(structure, interior))
            counts = tuple(int(np.sum((inp == secondary) & region)) for region in (interior, structure, outside))
            assert counts[1]+counts[2] > 0
            assert counts[1] <= max(3, min(9, int(structure.sum()*.22)))
            candidates.append((kind, (background, structural, secondary), counts))
        except AssertionError:
            pass
    assert len(candidates) == 1
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    train_counts = [0, 0, 0]
    for regime in ('train', 'test'):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            sig = _signature(pair)
            assert key not in seen_inputs and sig not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(sig)
            assert '/'.join(''.join(str(int(x)) for x in row) for row in inp) not in _SOURCE_INPUTS
            _, _, counts = _inspect_pair(pair)
            if regime == 'train':
                train_counts = [x+y for x, y in zip(train_counts, counts)]
    assert all(x > 0 for x in train_counts)


def _puzzle_signature(puzzle):
    return tuple(_signature(pair) for regime in ('train', 'test') for pair in puzzle[regime])


def _has_diagonal_cross(pair, palette):
    inp, out = pair['input'], pair['output']
    outline = out == palette[1]
    interior = ~_outside(outline) & ~outline
    marker = inp == palette[2]
    h, w = inp.shape
    for dc in (-1, 1):
        for r in range(h):
            for c in range(w):
                if not marker[r, c] or (r > 0 and 0 <= c-dc < w and marker[r-1, c-dc]):
                    continue
                cells = []
                x, y = r, c
                while x < h and 0 <= y < w and marker[x, y]:
                    cells.append((x, y))
                    x += 1
                    y += dc
                if (len(cells) >= 3 and any(outline[x] for x in cells)
                        and any(interior[x] for x in cells)
                        and any(not outline[x] and not interior[x] for x in cells)):
                    return True
    return False


def _covered_components(pair, palette):
    covered = (pair['input'] == palette[2]) & (pair['output'] == palette[1])
    seen = np.zeros_like(covered)
    count = 0
    h, w = covered.shape
    for r, c in zip(*np.where(covered)):
        if seen[r, c]:
            continue
        count += 1
        queue = [(r, c)]
        seen[r, c] = True
        while queue:
            x, y = queue.pop()
            for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if 0 <= nx < h and 0 <= ny < w and covered[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    queue.append((nx, ny))
    return count


def _near_irregular_turn(pair, palette):
    outline = pair['output'] == palette[1]
    filled = ~_outside(outline)
    axis = _irregular_orientation(filled)
    rows, intervals, _ = _irregular_intervals(filled if axis == 0 else filled.T)
    marker = pair['input'] == palette[2]
    marks = list(zip(*np.where(marker if axis == 0 else marker.T)))
    for k in range(1, len(rows)):
        if intervals[k] != intervals[k-1]:
            for j in (k-1, k):
                for c in intervals[j]:
                    if any(abs(int(mr)-int(rows[j]))+abs(int(mc)-c) <= 2 for mr, mc in marks):
                        return True
    return False


def self_test() -> None:
    signatures, sizes, palettes, shapes = set(), set(), set(), set()
    train_counts, test_counts, modes = set(), set(), set()
    shape_modes, diagonal_shapes, train_shapes, test_shapes = set(), set(), set(), set()
    thick_depths = set()
    irregular_axes = set()
    margins = set()
    compact = large = nonzero_background = zero_background = 0
    stable_palettes = varied_palettes = role_swaps = 0
    separated_contacts = near_turns = source_like = familiar_tests = 0
    train_zero_boundary = test_zero_inside = 0
    rect_aspects = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        sig = _puzzle_signature(puzzle)
        assert sig == _puzzle_signature(generate_puzzle(seed))
        signatures.add(sig)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        assert all(_inspect_pair(puzzle['train'][0])[2])
        puzzle_palettes = []
        train_kinds, test_kinds = [], []
        for regime in ('train', 'test'):
            for pair in puzzle[regime]:
                kind, palette, counts = _inspect_pair(pair)
                (train_kinds if regime == 'train' else test_kinds).append(kind)
                shapes.add(kind)
                (train_shapes if regime == 'train' else test_shapes).add(kind)
                palettes.add(palette)
                puzzle_palettes.append(palette)
                h, w = pair['input'].shape
                sizes.add((h, w))
                mode = tuple(x > 0 for x in counts)
                modes.add(mode)
                shape_modes.add((kind, mode))
                nonzero_background += palette[0] != 0
                zero_background += palette[0] == 0
                compact += h <= 13 and w <= 13
                large += h >= 24 or w >= 24
                outline = pair['output'] == palette[1]
                rr, cc = np.where(outline)
                margins.add(int(min(rr.min(), cc.min(), h-rr.max()-1, w-cc.max()-1)))
                if _has_diagonal_cross(pair, palette):
                    diagonal_shapes.add(kind)
                separated_contacts += _covered_components(pair, palette) >= 2
                if kind == 'irregular':
                    irregular_axes.add(_irregular_orientation(~_outside(outline)))
                    near_turns += _near_irregular_turn(pair, palette)
                if regime == 'train':
                    train_zero_boundary += counts[1] == 0
                else:
                    test_zero_inside += counts[0] == 0
                if kind in ('thin', 'thick'):
                    bh, bw = rr.max()-rr.min()+1, cc.max()-cc.min()+1
                    rect_aspects.append(float(bw/bh))
                    if kind == 'thick':
                        depth = 0
                        while outline[rr.min()+depth, cc.min()+depth]:
                            depth += 1
                        thick_depths.add(depth)
        if len(set(puzzle_palettes)) == 1:
            stable_palettes += 1
        else:
            varied_palettes += 1
        role_swaps += any(a[1] == b[2] or a[2] == b[1]
                          for i, a in enumerate(puzzle_palettes)
                          for b in puzzle_palettes[i+1:])
        source_like += train_kinds[:2] == ['thin', 'diamond'] and 'thick' in test_kinds and 'irregular' in test_kinds
        familiar_tests += any(kind in ('thin', 'diamond') for kind in test_kinds)
    assert len(signatures) == 42 and len(sizes) >= 30 and len(palettes) >= 20
    assert shapes == {'thin', 'thick', 'diamond', 'irregular'}
    assert {'thick', 'irregular'} <= train_shapes
    assert {'thin', 'diamond'} <= test_shapes
    assert len(modes) >= 3 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert all((kind, mode) in shape_modes
               for kind in shapes for mode in ((True, True, True), (True, False, True), (False, True, True)))
    assert diagonal_shapes == shapes and thick_depths == {2, 3}
    assert irregular_axes == {0, 1}
    assert 1 in margins and max(margins) >= 3
    assert compact >= 4 and large >= 40
    assert zero_background >= 5 and nonzero_background >= 100
    assert stable_palettes >= 10 and varied_palettes >= 10 and role_swaps >= 5
    assert min(rect_aspects) < .6 and max(rect_aspects) > 2
    assert separated_contacts >= 50 and near_turns >= 20
    assert source_like >= 3 and familiar_tests >= 10
    assert train_zero_boundary >= 30 and test_zero_inside >= 10
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(917, nt, ns)
        validate_generated_puzzle(puzzle)
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(917, nt, ns))
        assert nt is None or len(puzzle['train']) == nt
        assert ns is None or len(puzzle['test']) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(12, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
