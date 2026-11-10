"""Bridge mosaic: rigid tiles complete one square across an immovable band.

Only Python's standard library and NumPy are used. Generation retains no state,
and solve searches the visible input, including every possible square location.
No examples, seeds, answer cache, or parent modules are consulted by solve.
"""

from collections import Counter
import hashlib
import math

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_), "Integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, neighbors=EIGHT):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _tight(cells, colors):
    local = cells - cells.min(axis=0)
    a = np.full(tuple(local.max(axis=0) + 1), -1, dtype=np.int8)
    a[local[:, 0], local[:, 1]] = colors
    return a


def _rotations(a):
    seen, result = set(), []
    for k in range(4):
        b = np.rot90(a, k).copy()
        if _key(b) not in seen:
            seen.add(_key(b))
            result.append(b)
    return result


def _dilate(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[dr:dr+h, dc:dc+w] for dr in range(3) for dc in range(3)])


def _read(a):
    hist = np.bincount(a.ravel(), minlength=10)
    bg = int(hist.argmax())
    _require(np.sum(hist == hist[bg]) == 1, "Background must be uniquely most frequent")
    bands = []
    for color in range(10):
        if color == bg or not hist[color]:
            continue
        mask = a == color
        if len(_components(mask, EDGE)) != 1:
            continue
        for transpose in (False, True):
            m = mask.T if transpose else mask
            if not np.all(m.any(axis=0)):
                continue
            lo = np.argmax(m, axis=0)
            hi = m.shape[0] - np.argmax(m[::-1], axis=0)
            if all(np.all(m[lo[c]:hi[c], c]) for c in range(m.shape[1])):
                bands.append((color, transpose, lo, hi))
    _require(len(bands) == 1, "Expected one connected column-convex spanning band")
    band, transpose, lo, hi = bands[0]
    g = a.T.copy() if transpose else a.copy()
    singletons = [c for c, n in enumerate(hist) if n == 1 and c not in (bg, band)]
    _require(len(singletons) == 1, "Expected a unique singleton marker color")
    marker = singletons[0]
    masks = _components((g != bg) & (g != band))
    _require(3 <= len(masks) <= 6, "Expected three to six loose pieces")
    forbidden = _dilate(g == band)
    pieces, carriers, origins = [], [], []
    for cells in masks:
        values = g[cells[:, 0], cells[:, 1]]
        carrier = set(map(int, values)) - {marker}
        _require(len(cells) >= 3 and len(carrier) == 1, "Each piece needs one carrier and at least three cells")
        _require(not np.any(forbidden[cells[:, 0], cells[:, 1]]), "Loose pieces must be separated from band")
        carriers.append(next(iter(carrier)))
        pieces.append(_tight(cells, values))
        origins.append(cells)
    _require(len(set(carriers)) == len(carriers), "Carrier colors must be distinct")
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    _require(len(marked) == 1, "One piece must carry the marker")
    base = np.full_like(g, bg)
    base[g == band] = band
    return dict(grid=g, base=base, bg=bg, band=band, marker=marker,
                transpose=transpose, lo=lo, hi=hi, pieces=pieces,
                carriers=carriers, origins=origins, anchor=marked[0],
                area=sum(int(np.sum(p >= 0)) for p in pieces))


def _windows(f):
    """All in-bounds squares containing the entire band depth at every column.

    Each column must leave at least one free cell on BOTH sides of the band.
    Area is only a candidate filter; shape matching determines the answer.
    """
    h, w = f['grid'].shape
    thickness = f['hi'] - f['lo']
    for side in range(3, min(h, w) + 1):
        for col in range(w - side + 1):
            if side * side - int(thickness[col:col+side].sum()) != f['area']:
                continue
            first = max(0, int(f['hi'][col:col+side].max()) - side + 1)
            last = min(h - side, int(f['lo'][col:col+side].min()) - 1)
            for row in range(first, last + 1):
                yield row, col, side


def _placements(pieces, side, marker, anchor, rotation=True):
    result = []
    for i, piece in enumerate(pieces):
        candidates = {}
        for p in (_rotations(piece) if rotation else [piece]):
            h, w = p.shape
            if h > side or w > side:
                continue
            if i == anchor and p[0, 0] != marker:
                continue
            rr, cc = np.nonzero(p >= 0)
            cells = tuple((int(r)*side + int(c), int(p[r, c])) for r, c in zip(rr, cc))
            bits = sum(1 << pos for pos, _ in cells)
            offsets = [(0, 0)] if i == anchor else [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
            for r, c in offsets:
                offset = r*side+c
                mask = bits << offset
                # Include colored values in deduplication, including the marker.
                values = tuple((pos+offset, color) for pos, color in cells)
                candidates.setdefault((mask, values), (mask, values))
        result.append(list(candidates.values()))
    return result


def _pack(options, fixed, side, limit=2):
    full = (1 << (side*side)) - 1
    options = [[p for p in choices if not p[0] & fixed] for choices in options]
    if any(not choices for choices in options):
        return []
    answers, seen = [], set()
    placed = [None] * len(options)

    def visit(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied == full:
                a = np.full((side, side), -1, dtype=np.int8)
                for p in placed:
                    for pos, color in p[1]:
                        a.flat[pos] = color
                key = a.tobytes()
                if key not in seen:
                    seen.add(key)
                    answers.append(a)
            return
        empty = full ^ occupied
        first = empty & -empty
        # Exactly one remaining rigid piece must cover this cell.
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for p in options[i]:
                if p[0] & first and not p[0] & occupied:
                    placed[i] = p
                    visit(occupied | p[0], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    visit(fixed, tuple(range(len(options))))
    return answers


def _assemblies(a, rotation=True):
    f = _read(a)
    solutions, options = {}, {}
    windows = list(_windows(f))
    for row, col, side in windows:
        if side not in options:
            options[side] = _placements(f['pieces'], side, f['marker'], f['anchor'], rotation)
        bandmask = f['base'][row:row+side, col:col+side] == f['band']
        fixed = sum(1 << (int(r)*side+int(c)) for r, c in np.argwhere(bandmask))
        for local in _pack(options[side], fixed, side):
            answer = f['base'].copy()
            view = answer[row:row+side, col:col+side]
            view[local >= 0] = local[local >= 0]
            solutions.setdefault(answer.tobytes(), (answer, (row, col, side)))
            if len(solutions) >= 2:
                f['area_windows'] = len(windows)
                return list(solutions.values()), f
    f['area_windows'] = len(windows)
    return list(solutions.values()), f


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique square completion using only this input's cells."""
    solutions, f = _assemblies(_grid(input_grid))
    _require(len(solutions) == 1, "Expected one distinct colored anchored completion")
    answer = solutions[0][0]
    return (answer.T if f['transpose'] else answer).copy()


def _profile(rng, side):
    maximum = (side - 2) // 2
    mode = int(rng.integers(3))
    if mode == 0:
        p = rng.integers(1, maximum+1, side)
    elif mode == 1:
        p = np.clip(np.cumsum(rng.integers(-1, 2, side)) + int(rng.integers(1, maximum+1)), 1, maximum)
    else:
        p = np.ones(side, dtype=int)
        for _ in range(3):
            a = int(rng.integers(side))
            b = int(rng.integers(a+1, side+1))
            p[a:b] = int(rng.integers(1, maximum+1))
    if len(set(map(int, p))) < 2:
        p[int(rng.integers(side))] = 1 if int(p[0]) != 1 else maximum
    return p


def _partition(rng, mask, count, style):
    """Connected cap partitions: weighted growth or stepped vertical cuts."""
    cells = np.argwhere(mask)
    if len(cells) < count*3:
        return None
    if count == 1:
        return np.where(mask, 0, -1).astype(np.int8)
    labels = np.full(mask.shape, -1, dtype=np.int8)
    if style == 0:
        points = cells[rng.choice(len(cells), count, replace=False)]
        frontiers = [set() for _ in range(count)]
        for i, point in enumerate(points):
            labels[tuple(point)] = i
        def add(i, r, c):
            for dr, dc in EDGE:
                nr, nc = r+dr, c+dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and labels[nr, nc] < 0:
                    frontiers[i].add((nr, nc))
        for i, (r, c) in enumerate(points):
            add(i, int(r), int(c))
        sizes = [1] * count
        weights = np.exp(rng.uniform(-.8, .8, count))
        for _ in range(len(cells)-count):
            eligible = [i for i in range(count) if frontiers[i]]
            i = min(eligible, key=lambda j: sizes[j]/weights[j] + float(rng.random())*.7)
            choices = sorted(frontiers[i])
            r, c = choices[int(rng.integers(len(choices)))]
            labels[r, c] = i
            sizes[i] += 1
            for frontier in frontiers:
                frontier.discard((r, c))
            add(i, r, c)
    else:
        cuts = sorted(map(int, rng.choice(np.arange(1, mask.shape[1]), count-1, replace=False)))
        for r, c in cells:
            labels[r, c] = sum(c >= cut for cut in cuts)
        for _ in range(mask.size * 2):
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = EDGE[int(rng.integers(4))]
            nr, nc = int(r)+dr, int(c)+dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]):
                continue
            old, new = int(labels[r, c]), int(labels[nr, nc])
            if old == new or np.sum(labels == old) <= 3:
                continue
            labels[r, c] = new
            if len(_components(labels == old, EDGE)) != 1:
                labels[r, c] = old
    if any(np.sum(labels == i) < 3 or len(_components(labels == i, EDGE)) != 1 for i in range(count)):
        return None
    return labels


def _scatter(rng, base, pieces, bg):
    oriented = [np.rot90(p, int(rng.integers(4))).copy() for p in pieces]
    # Always give a visible asymmetric rigid-rotation witness.
    for i, piece in enumerate(pieces):
        if len(_rotations((piece >= 0).astype(np.int8))) == 4:
            oriented[i] = np.rot90(piece, int(rng.integers(1, 4))).copy()
            break
    else:
        return None
    h, w = base.shape
    for _ in range(16):
        result = base.copy()
        forbidden = _dilate(base != bg)
        order = sorted(range(len(pieces)), key=lambda i: -np.sum(oriented[i] >= 0) + float(rng.random())*4)
        for i in order:
            p = oriented[i]
            ph, pw = p.shape
            rr, cc = np.nonzero(p >= 0)
            if ph > h or pw > w:
                break
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in zip(rr, cc):
                valid &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            places = np.argwhere(valid)
            if not len(places):
                break
            row, col = places[int(rng.integers(len(places)))]
            result[row+rr, col+cc] = p[rr, cc]
            forbidden = _dilate(result != bg)
        else:
            return result
    return None


def _make_pair(rng, side, count, transpose, palette):
    bg, band, marker, colors = palette
    for _ in range(240):
        p = _profile(rng, side)
        # Nonflat edges and nonsquare loose area make both inheritances visible.
        area = 2*int(p.sum())
        if len(set(map(int, p))) < 2 or math.isqrt(area)**2 == area:
            continue
        mask = np.arange(side)[:, None] < p[None, :]
        upper_count = int(rng.integers(1, count))
        lower_count = count-upper_count
        upper = _partition(rng, mask, upper_count, int(rng.integers(2)))
        lower = _partition(rng, mask, lower_count, int(rng.integers(2)))
        if upper is None or lower is None:
            continue
        owners = np.full((side, side), -1, dtype=np.int8)
        owners[mask] = upper[mask]
        lower = lower[::-1, ::-1]
        lower_mask = mask[::-1, ::-1]
        owners[lower_mask] = lower[lower_mask]+upper_count
        local = np.full((side, side), band, dtype=np.int8)
        for i in range(count):
            local[owners == i] = colors[i]
        local[0, 0] = marker
        pieces = [_tight(np.argwhere(owners == i), local[owners == i]) for i in range(count)]
        # Canvas sizing reserves actual clearance, rather than output-size slots.
        h = int(rng.integers(max(18, side+12), 31))
        w = int(rng.integers(max(16, side+7), 31))
        row = int(rng.integers(5, h-side-3))
        col = int(rng.integers(1, w-side))
        center = row+side//2
        # Unused contour context is independently varied and must be preserved.
        lo = center-rng.integers(0, 3, w)
        hi = center+int(rng.integers(2, 4))+rng.integers(0, 3, w)
        lo[col:col+side] = row+p
        hi[col:col+side] = row+side-p[::-1]
        # All columns intersect a common core, so the band remains connected.
        if int(lo.max()) >= int(hi.min()):
            continue
        base = np.full((h, w), bg, dtype=np.int8)
        for c in range(w):
            base[lo[c]:hi[c], c] = band
        output = base.copy()
        output[row:row+side, col:col+side] = local
        inp = _scatter(rng, base, pieces, bg)
        if inp is None:
            continue
        try:
            solutions, f = _assemblies(inp)
        except AssertionError:
            continue
        if len(solutions) != 1 or not np.array_equal(solutions[0][0], output):
            continue
        if transpose:
            inp, output = inp.T, output.T
        return {'input': inp.copy(), 'output': output.copy()}
    raise RuntimeError("Could not construct a unique bridge mosaic in bounded attempts")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 'num_train', 2, 10)
    ntest = _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    stable = bool(rng.random() < .25)
    shared = list(map(int, rng.permutation([c for c in range(10) if c != marker])))
    mode = int(rng.integers(3))
    direction = bool(rng.integers(2))
    train_counts = [3, 4] + [int(rng.integers(3, 7)) for _ in range(ntrain-2)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', ntrain), ('test', ntest)):
        for i in range(number):
            is_test = split == 'test'
            if mode == 0:  # Scale transfer, familiar band orientations.
                side = int(rng.integers(9, 11) if is_test else rng.integers(6, 9))
                transpose = bool(i % 2) if not is_test else bool(rng.integers(2))
            else:  # Transfer band direction, with overlapping size support.
                side = int(rng.integers(6, 10))
                transpose = direction if not is_test else not direction
            count = int(rng.choice(train_counts)) if is_test else train_counts[i]
            if count >= 5:
                side = max(side, 7)
            if count == 6:
                side = max(side, 8)
            available = shared if stable else list(map(int, rng.permutation([c for c in range(10) if c != marker])))
            bg, band = available[:2]
            palette = bg, band, marker, available[2:]
            for _ in range(8):
                try:
                    pair = _make_pair(rng, side, count, transpose, palette)
                except RuntimeError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
    return puzzle


def _pair_facts(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Pair schema")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "Preserve canvas size")
    solutions, f = _assemblies(inp)
    _require(len(solutions) == 1, "Unique anchored completion required")
    expected, window = solutions[0]
    target = out.T if f['transpose'] else out
    _require(np.array_equal(expected, target), "Incorrect completed mosaic")
    _require(np.array_equal(inp == f['band'], out == f['band']), "Fixed band changed")
    _require(np.array_equal(np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)), "Preserve all colors and area")
    row, col, side = window
    view = target[row:row+side, col:col+side]
    _require(np.all(view != f['bg']) and view[0, 0] == f['marker'], "Square and marker conditions")
    _require(np.array_equal(view == f['band'], np.rot90(view == f['band'], 2)), "Generated witness band should be centrally symmetric")
    _require(math.isqrt(f['area'])**2 != f['area'], "Loose area should distinguish a band-free square rule")
    rotation_required, asymmetric_rotation = False, False
    areas, shapes = [], []
    for i, (piece, color) in enumerate(zip(f['pieces'], f['carriers'])):
        mask = target == color
        if i == f['anchor']:
            mask |= target == f['marker']
        cells = np.argwhere(mask)
        _require(np.all((cells[:, 0] >= row) & (cells[:, 0] < row+side) & (cells[:, 1] >= col) & (cells[:, 1] < col+side)), "All payload belongs to the square")
        final = _tight(cells, target[cells[:, 0], cells[:, 1]])
        _require(any(np.array_equal(final, p) for p in _rotations(piece)), "Piece deformed or reflected")
        rotated = not np.array_equal(final, piece)
        rotation_required |= rotated
        asymmetric_rotation |= rotated and len(_rotations((piece >= 0).astype(np.int8))) == 4
        areas.append(len(cells))
        shapes.append(_key((piece >= 0).astype(np.int8)))
    _require(asymmetric_rotation, "Need a clear asymmetric-rotation demonstration")
    # The 180-degree alternative proves marker anchoring changes the answer.
    alternate = target.copy()
    alternate[row:row+side, col:col+side] = np.rot90(view, 2)
    _require(not np.array_equal(alternate, target), "Markerless alternative must differ")
    _require(np.array_equal(alternate == f['band'], target == f['band']), "Markerless witness must preserve band")
    f.update(window=window, side=side, count=len(areas), areas=areas,
             rotation_required=rotation_required, asymmetric_rotation=asymmetric_rotation,
             band_area=int(np.sum(view == f['band'])),
             amplitude=int(np.ptp(f['lo'][col:col+side])),
             shapes=shapes)
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal pair counts")
    seen, facts = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _pair_facts(pair)
            key = _key(_grid(pair['input']))
            _require(key not in seen, "Duplicate inputs")
            seen.add(key)
            facts.append(f)
    _require(len({f['marker'] for f in facts}) == 1, "Marker convention must be consistent")
    _require(len({f['count'] for f in facts[:a]}) >= 2, "Demonstrate that piece count is variable")
    _require(all(f['count'] in {g['count'] for g in facts[:a]} for f in facts[a:]), "Test count must have a training witness")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded, in-memory checks: 42 puzzles, counts, mutations and ablations."""
    structures, shapes, sides, counts, palettes, amplitudes = set(), set(), set(), set(), set(), set()
    orientations, windows, ratios = Counter(), [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _pair_facts(pair)
                _require(np.array_equal(solve(pair['input']), pair['output']), "Solve disagreement")
                # Ignoring B's rotations cannot recover this colored output.
                translation_answers, _ = _assemblies(_grid(pair['input']), rotation=False)
                _require(not translation_answers, "Translation-only shortcut unexpectedly succeeds")
                inp = pair['input']
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                shapes.add(inp.shape)
                sides.add(f['side'])
                counts.add(f['count'])
                palettes.add((f['bg'], f['band'], f['marker']))
                amplitudes.add(f['amplitude'])
                orientations[(split, f['transpose'])] += 1
                windows.append(f['area_windows'])
                ratios.append(max(f['areas'])/min(f['areas']))
    _require(len(structures) == 210, "Expected 210 distinct normalized inputs")
    _require(len(shapes) >= 35 and len(palettes) >= 30, "Canvas/palette diversity collapsed")
    _require(sides >= {6, 7, 8, 9, 10} and counts == {3, 4, 5, 6}, "Scale/count diversity collapsed")
    _require(set(orientations) == {(s, t) for s in ('train', 'test') for t in (False, True)}, "Use both band directions in both splits")
    _require(max(amplitudes) >= 3 and max(windows) >= 5 and max(ratios) >= 4, "Geometric diversity collapsed")
    a = generate_puzzle(913276381)
    _require(_equal(a, generate_puzzle(913276381)), "Nondeterministic generation")
    _require(not _equal(a, generate_puzzle(205359645)), "Different seeds match")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1234+10*nt+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "Explicit counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect identity output accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
