"""Perforated blueprints: code stencils open passages in room partitions.

Only Python's standard library and NumPy are used. No file I/O or latent
generation state is involved in solve(), validation, or self_test().
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_HEADER = 5
_HOST_SEEDS = (0, 645040732, 473051365, 732989317, 609008375,
               162293767, 871445544, 693381438, 271657389, 917970398,
               278255445, 898938939, 89657864, 850027400, 584247584,
               203958475, 410253109, 929917439, 620224470, 159539957,
               526606440, 229194851, 806035307, 39330357, 249849188,
               910597394, 167912706, 914827960, 462132572, 660227586,
               919014600, 839251876, 346071401, 48811766, 254784306,
               641025732, 388110750, 790164398, 63291656, 552063379,
               979357105, 428661523)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be an integer 2D array, 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _rooms(mask):
    """Actual bounded four-connected regions, with a padded exterior."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    result = []
    h, w = free.shape
    for r, c in np.argwhere(free):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells, outside = [(r, c)], [], False
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y - 1, x - 1))
            outside |= y == 0 or x == 0 or y == h - 1 or x == w - 1
            for dy, dx in _DIRS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and free[yy, xx]
                        and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if not outside:
            result.append(cells)
    return result


def _walls(shape, boxes):
    result = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        result[t, l:r + 1] = result[b, l:r + 1] = True
        result[t:b + 1, l] = result[t:b + 1, r] = True
    return result


def _clean_properties(mask):
    """Check the pre-punch rectangular-room partition, not its construction."""
    rooms = _rooms(mask)
    boundaries, sizes = [], []
    for cells in rooms:
        a = np.asarray(cells)
        t, l = map(int, a.min(axis=0))
        b, r = map(int, a.max(axis=0))
        if len(cells) != (b - t + 1) * (r - l + 1):
            raise ValueError("initial rooms must be rectangles")
        boundaries.append((t - 1, l - 1, b + 1, r + 1))
        sizes.append((b - t + 1, r - l + 1))
    if not rooms or not np.array_equal(mask, _walls(mask.shape, boundaries)):
        raise ValueError("initial walls must be exactly a room partition")
    # An interior T-junction distinguishes partial partitions from full grids.
    degree = np.zeros(mask.shape, dtype=np.int8)
    degree[1:] += mask[:-1]
    degree[:-1] += mask[1:]
    degree[:, 1:] += mask[:, :-1]
    degree[:, :-1] += mask[:, 1:]
    partial = bool(np.any((degree[1:-1, 1:-1] == 3) & mask[1:-1, 1:-1]))
    # A second, planar count checks initial room counting independently.
    edges = int(np.sum(mask[1:] & mask[:-1]) + np.sum(mask[:, 1:] & mask[:, :-1]))
    units = int(np.sum(mask[:-1, :-1] & mask[1:, :-1]
                       & mask[:-1, 1:] & mask[1:, 1:]))
    if edges - int(mask.sum()) + 1 - units != len(rooms):
        raise ValueError("initial flood and planar room counts disagree")
    return len(rooms), partial, tuple(sizes)


def _read_input(value):
    inp = _grid(value)
    if inp.shape[0] < 10 or inp.shape[1] < 10:
        raise ValueError("missing legend or blueprint")
    separators = np.flatnonzero(np.all(inp == 9, axis=1)).tolist()
    if separators != [4] or np.any(inp[3] != 0):
        raise ValueError("the legend ends in a blank row and a full nine row")
    centers = [(int(c), int(inp[1, c])) for c in range(inp.shape[1])
               if int(inp[1, c]) not in (0, 9)]
    if not 2 <= len(centers) <= 3 or len({code for _, code in centers}) != len(centers):
        raise ValueError("the legend needs two or three distinct codes")
    book, expected = {}, np.zeros((4, inp.shape[1]), dtype=np.int8)
    for c, code in centers:
        if not 1 <= c < inp.shape[1] - 1:
            raise ValueError("incomplete legend stencil")
        block = inp[:3, c - 1:c + 2]
        mask = block == 9
        allowed = np.zeros((3, 3), dtype=bool)
        for dy, dx in _DIRS:
            allowed[1 + dy, 1 + dx] = True
        if not mask.any() or np.any(mask & ~allowed):
            raise ValueError("punches use a nonempty subset of the four cardinal neighbors")
        book[code] = tuple((int(y) - 1, int(x) - 1) for y, x in np.argwhere(mask))
        expected[:3, c - 1:c + 2][mask] = 9
        expected[1, c] = code
    if any(b[0] - a[0] < 5 for a, b in zip(centers, centers[1:])):
        raise ValueError("legend windows need two columns of separation")
    if not np.array_equal(expected, inp[:4]):
        raise ValueError("unexplained legend pixels")
    scene = inp[_HEADER:]
    if np.any(scene == 9):
        raise ValueError("nine is not blueprint material")
    colors = set(map(int, np.unique(scene))) - {0} - set(book)
    if not 1 <= len(colors) <= 3:
        raise ValueError("one to three blueprint colors are required")
    restored = scene.copy()
    markers = {color: [] for color in colors}
    points = list(map(tuple, np.argwhere(np.isin(scene, list(book)))))
    for r, c in points:
        owners = {int(scene[r + dy, c + dx]) for dy, dx in _DIRS
                  if 0 <= r + dy < scene.shape[0] and 0 <= c + dx < scene.shape[1]
                  and int(scene[r + dy, c + dx]) in colors}
        if len(owners) != 1:
            raise ValueError("a code must have a unique adjacent wall owner")
        owner = owners.pop()
        restored[r, c] = owner
        markers[owner].append((int(r), int(c), int(scene[r, c])))
    if any(max(abs(int(r) - int(y)), abs(int(c) - int(x))) <= 1
           for (r, c), (y, x) in itertools.combinations(points, 2)):
        raise ValueError("code centers may not touch, including diagonally")
    networks = []
    for color in sorted(colors):
        positions = np.argwhere(restored == color)
        t, l = map(int, positions.min(axis=0))
        b, r = map(int, positions.max(axis=0))
        mask = restored[t:b + 1, l:r + 1] == color
        if min(mask.shape) < 3 or not (mask[0].all() and mask[-1].all()
                                      and mask[:, 0].all() and mask[:, -1].all()):
            raise ValueError("every blueprint has a complete rectangular casing")
        before, partial, sizes = _clean_properties(mask)
        local = [(y - t, x - l, code) for y, x, code in markers[color]]
        if any(not (0 < y < mask.shape[0] - 1 and 0 < x < mask.shape[1] - 1)
               for y, x, _ in local):
            raise ValueError("a code center must be inside its casing")
        if not all(_safe_tool(mask, y, x, book[code]) for y, x, code in local):
            raise ValueError("tools must cut thin walls, not drill isolated junction pinholes")
        networks.append(dict(color=color, box=(t, l, b, r), mask=mask,
                             markers=local, before=before, partial=partial, sizes=sizes))
    for a, b in itertools.combinations(networks, 2):
        t, l, y, x = a['box']
        u, v, j, k = b['box']
        if not (y + 1 < u or j + 1 < t or x + 1 < v or k + 1 < l):
            raise ValueError("blueprint boxes require a background gap")
    return inp, book, networks


def _punch(network, book, records=None, cross=False, protect=True):
    mask = network['mask']
    removed = np.zeros(mask.shape, dtype=bool)
    clipped = missed = False
    records = network['markers'] if records is None else records
    h, w = mask.shape
    for y, x, code in records:
        for dy, dx in (_DIRS if cross else book[code]):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w):
                clipped = True
                continue
            if protect and (yy in (0, h - 1) or xx in (0, w - 1)):
                clipped |= bool(mask[yy, xx])
                continue
            missed |= not bool(mask[yy, xx])
            removed[yy, xx] |= mask[yy, xx]
    opened = mask & ~removed
    rooms = _rooms(opened)
    return len(rooms), opened, clipped, missed


def _report(networks, counts):
    order = sorted(range(len(networks)), key=lambda i: (counts[i], networks[i]['box'][1],
                                                        networks[i]['box'][0]))
    out = np.full((len(order), max(counts)), 9, dtype=np.int8)
    for row, i in enumerate(order):
        out[row, :counts[i]] = networks[i]['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only this input; restore, punch simultaneously, count, report."""
    _, book, networks = _read_input(input_grid)
    counts = [_punch(n, book)[0] for n in networks]
    if min(counts) < 1:
        raise ValueError("protected casings must retain at least one room")
    return _report(networks, counts)


def _partition(rng, h, w, count, style):
    boxes = [(0, 0, h - 1, w - 1)]
    if (h - 1) // 2 * ((w - 1) // 2) < count:
        return None
    if style == 'grid':
        factors = [(a, count // a) for a in range(1, count + 1) if count % a == 0
                   and a * 2 < h and (count // a) * 2 < w]
        mixed = [p for p in factors if min(p) > 1]
        if mixed:
            nr, nc = mixed[int(rng.integers(len(mixed)))]
            def axis(parts, span):
                gaps = np.full(parts, 2, dtype=int)
                for _ in range(span - 2 * parts):
                    gaps[int(rng.integers(parts))] += 1
                return [0] + np.cumsum(gaps).tolist()
            ys, xs = axis(nr, h - 1), axis(nc, w - 1)
            boxes = [(ys[a], xs[b], ys[a + 1], xs[b + 1])
                     for a in range(nr) for b in range(nc)]
            return _walls((h, w), boxes)
    first_axis = int(rng.integers(2))
    for step in range(count - 1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if style == 'strip' and axis != first_axis:
                    continue
                if style == 'partial' and step == 1 and axis == first_axis:
                    continue
                for cut in range(start + 2, end - 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    capacity = sum((z[2] - z[0]) // 2 * ((z[3] - z[1]) // 2)
                                   for j, z in enumerate(boxes) if j != i)
                    capacity += sum((z[2] - z[0]) // 2 * ((z[3] - z[1]) // 2) for z in halves)
                    if capacity >= count:
                        choices.append((i, axis, halves))
        if not choices:
            return None
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        if step == 0:
            first_axis = axis
        boxes[i:i + 1] = halves
    return _walls((h, w), boxes)


def _layout(rng, h, w, number, horizontal=False):
    """Sample guillotine slots, then independently inset each blueprint."""
    slots = [(0, 0, h - 1, w - 1)]
    if horizontal:
        gaps = np.full(number, 7, dtype=int)
        extra = w - 2 * (number - 1) - int(gaps.sum())
        if extra < 0:
            return None
        for _ in range(extra):
            gaps[int(rng.integers(number))] += 1
        slots, x = [], 0
        for span in gaps:
            slots.append((0, x, h - 1, x + int(span) - 1))
            x += int(span) + 2
    else:
        for _ in range(number - 1):
            choices = []
            for i, (t, l, b, r) in enumerate(slots):
                if b - t + 1 >= 14:
                    choices.extend((i, 0, c) for c in range(t + 6, b - 6))
                if r - l + 1 >= 16:
                    choices.extend((i, 1, c) for c in range(l + 7, r - 7))
            if not choices:
                return None
            i, axis, cut = choices[int(rng.integers(len(choices)))]
            t, l, b, r = slots[i]
            halves = [(t, l, cut - 1, r), (cut + 1, l, b, r)] if axis == 0 else [(t, l, b, cut - 1), (t, cut + 1, b, r)]
            slots[i:i + 1] = halves
    result = []
    for t, l, b, r in slots:
        # Margins, aspect ratio and edge contacts vary independently.
        dt = int(rng.integers(min(3, max(1, (b - t - 4) // 2))))
        db = int(rng.integers(min(3, max(1, (b - t - 4) // 2))))
        dl = int(rng.integers(min(3, max(1, (r - l - 4) // 2))))
        dr = int(rng.integers(min(3, max(1, (r - l - 4) // 2))))
        result.append((t + dt, l + dl, b - db, r - dr))
    rng.shuffle(result)
    return result


def _add_markers(rng, n, book, amount, boundary=False):
    mask = n['mask']
    h, w = mask.shape
    candidates = [(int(y), int(x)) for y, x in np.argwhere(mask)
                  if 0 < y < h - 1 and 0 < x < w - 1]
    rng.shuffle(candidates)
    records = []
    if boundary:
        choices = [(y, x, code) for y, x in candidates for code, offsets in book.items()
                   if _safe_tool(mask, y, x, offsets)
                   and any(y + dy in (0, h - 1) or x + dx in (0, w - 1) for dy, dx in offsets)]
        if not choices:
            return False
        records.append(choices[int(rng.integers(len(choices)))])
    # Junction anchors mix with ordinary wall anchors. No center can lose all
    # visible ownership evidence to an incidentally neighboring marker.
    junctions = [(y, x) for y, x in candidates
                 if sum(bool(mask[y + dy, x + dx]) for dy, dx in _DIRS) >= 3]
    rng.shuffle(junctions)
    ordered = junctions[:1] + candidates
    codes = list(book)
    rng.shuffle(codes)
    for y, x in ordered:
        if any(max(abs(y - a), abs(x - b)) <= 1 for a, b, _ in records):
            continue
        preferred = codes[len(records) % len(codes)] if len(records) < len(codes) else int(rng.choice(codes))
        eligible = [code for code in [preferred] + codes
                    if _safe_tool(mask, y, x, book[code])]
        if not eligible:
            continue
        code = eligible[0]
        records.append((y, x, code))
        if len(records) >= amount:
            break
    n['markers'] = records
    return len(records) >= min(amount, 2)


def _safe_tool(mask, y, x, offsets):
    """Generated cuts meet old room space; no isolated hole inside a joint."""
    h, w = mask.shape
    for dy, dx in offsets:
        yy, xx = y + dy, x + dx
        if 0 < yy < h - 1 and 0 < xx < w - 1 and mask[yy, xx]:
            if all(mask[yy + a, xx + b] for a, b in _DIRS):
                return False
    return True


def _legend(book, width):
    legend = np.zeros((_HEADER, width), dtype=np.int8)
    legend[4] = 9
    for i, (code, offsets) in enumerate(book.items()):
        c = 1 + 5 * i
        legend[1, c] = code
        for dy, dx in offsets:
            legend[1 + dy, c + dx] = 9
    return legend


def _tie_clear(networks, counts):
    return all(a != b or x['box'][3] < y['box'][1] or y['box'][3] < x['box'][1]
               for i, (x, a) in enumerate(zip(networks, counts))
               for y, b in list(zip(networks, counts))[i + 1:])


def _features(networks, book):
    counts, redundant, boundary, missed = [], False, False, False
    linear_counts, lattice_counts = [], []
    for n in networks:
        after, _, clipped, off_wall = _punch(n, book)
        counts.append(after)
        boundary |= clipped
        missed |= off_wall
        individual = sum(n['before'] - _punch(n, book, [p])[0] for p in n['markers'])
        redundant |= individual > n['before'] - after
        linear_counts.append(max(1, n['before'] - individual))
        mask = n['mask']
        hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
        vs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
        lattice = np.zeros(mask.shape, dtype=bool)
        lattice[hs] = True
        lattice[:, vs] = True
        lattice_counts.append(_punch(dict(n, mask=lattice), book)[0])
    before = [n['before'] for n in networks]
    cross = [_punch(n, book, cross=True)[0] for n in networks]
    marker_counts = [max(1, len(n['markers'])) for n in networks]
    spatial = sorted(range(len(networks)), key=lambda i: (networks[i]['box'][1], networks[i]['box'][0]))
    output_order = sorted(range(len(networks)), key=lambda i: (counts[i], networks[i]['box'][1], networks[i]['box'][0]))
    output_colors = [networks[i]['color'] for i in output_order]
    return dict(counts=counts, before=before, redundant=redundant, boundary=boundary,
                missed=missed, partial=any(n['partial'] for n in networks),
                effect=counts != before, selective=counts != cross,
                linear_fails=counts != linear_counts, lattice_fails=counts != lattice_counts,
                count_not_markers=counts != marker_counts,
                color_ascending=output_colors == sorted(output_colors),
                order=[counts[i] for i in spatial] != sorted(counts),
                padding=min(counts) < max(counts), tie=len(set(counts)) < len(counts))


def _make_pair(rng, parameters, kind):
    book, colors = parameters['book'], parameters['colors']
    witness, tied, test = kind == 'witness', kind == 'tie', kind == 'test'
    for attempt in range(500):
        number = 2 if witness or tied else int(rng.choice([1, 2, 2, 3]))
        if test:
            number = int(rng.choice([2, 2, 3]))
        min_width = max(5 * len(book), 20 if number >= 2 else 10)
        w = int(rng.integers(min_width, 31))
        h = int(rng.integers(16 if witness or tied or test else 9, 26))
        boxes = _layout(rng, h, w, number, horizontal=tied)
        if boxes is None:
            continue
        networks = []
        chosen_colors = list(map(int, rng.choice(colors, size=number, replace=False)))
        failed = False
        for i, (box, color) in enumerate(zip(boxes, chosen_colors)):
            t, l, b, r = box
            bh, bw = b - t + 1, r - l + 1
            capacity = (bh - 1) // 2 * ((bw - 1) // 2)
            low = 4 if witness else 3 if tied else 1
            high = min(capacity, parameters['train_max'])
            if test and parameters['shift'] == 'rooms':
                low, high = parameters['train_max'] + 1, min(capacity, parameters['train_max'] + 4)
            elif test:
                low = 4
            if high < low:
                failed = True
                break
            count = int(rng.integers(low, high + 1))
            style = 'partial' if witness and i == 0 else str(rng.choice(['partial', 'partial', 'grid', 'strip']))
            mask = _partition(rng, bh, bw, count, style)
            if mask is None:
                failed = True
                break
            before, partial, sizes = _clean_properties(mask)
            n = dict(color=color, box=box, mask=mask, markers=[], before=before,
                     partial=partial, sizes=sizes)
            amount = int(rng.integers(3, 7)) if witness or tied else int(rng.integers(1, 6))
            if test and parameters['shift'] == 'density':
                amount = int(rng.integers(9, 14))
            if before > 1 and not _add_markers(rng, n, book, amount, boundary=tied and i == 0):
                failed = True
                break
            networks.append(n)
        if failed:
            continue
        features = _features(networks, book)
        if min(features['counts']) < 1 or not _tie_clear(networks, features['counts']):
            continue
        if witness and not all(features[k] for k in ('partial', 'redundant', 'effect', 'selective',
                                                     'linear_fails', 'lattice_fails',
                                                     'order', 'padding', 'count_not_markers')):
            continue
        if tied and not (features['tie'] and features['boundary']):
            continue
        if tied and features['color_ascending'] == parameters['witness_color_ascending']:
            continue
        if test and parameters['shift'] == 'density' and max(len(n['markers']) for n in networks) < 8:
            continue
        scene = np.zeros((h, w), dtype=np.int8)
        for n in networks:
            t, l, b, r = n['box']
            scene[t:b + 1, l:r + 1][n['mask']] = n['color']
            for y, x, code in n['markers']:
                scene[t + y, l + x] = code
        inp = np.concatenate((_legend(book, w), scene), axis=0)
        # Ground truth uses the constructed walls. solve() reconstructs them
        # afresh from the returned pixels, which validation compares exactly.
        out = _report(networks, features['counts'])
        return {'input': inp, 'output': out}
    raise RuntimeError(f"bounded construction exhausted for {kind}")


def _parameters(rng):
    palette = list(map(int, rng.permutation(np.arange(1, 9))))
    count = int(rng.integers(2, 4))
    single = (_DIRS[int(rng.integers(4))],)
    pairs = list(itertools.combinations(_DIRS, 2))
    double = tuple(pairs[int(rng.integers(len(pairs)))])
    masks = [single, double]
    if count == 3:
        if rng.random() < .4:
            masks.append(tuple(_DIRS))
        else:
            omit = int(rng.integers(4))
            masks.append(tuple(d for i, d in enumerate(_DIRS) if i != omit))
    book = {palette[i]: tuple(masks[i]) for i in range(count)}
    return dict(book=book, colors=palette[count:count + 3],
                train_max=int(rng.integers(6, 9)),
                shift=str(rng.choice(['rooms', 'density'])))


def _check_counts(num_train, num_test):
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (num_train, num_test)):
        raise ValueError("example counts must be integers")
    nt, nq = int(num_train), int(num_test)
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return nt, nq


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    parameters = _parameters(rng)
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + nq):
        split = 'train' if i < nt else 'test'
        kind = 'witness' if i == 0 else 'tie' if i == 1 else 'ordinary' if i < nt else 'test'
        for _ in range(8):
            pair = _make_pair(rng, parameters, kind)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError("bounded duplicate avoidance exhausted")
        seen.add(key)
        puzzle[split].append(pair)
        if i == 0:
            out = pair['output']
            parameters['witness_color_ascending'] = bool(out[0, 0] < out[1, 0])
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, books, train_features, train_codes = set(), [], [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            _, book, networks = _read_input(inp)
            books.append(tuple(sorted(book.items())))
            features = _features(networks, book)
            counts = features['counts']
            assert _tie_clear(networks, counts), 'tied blueprints need clear horizontal order'
            assert all(1 <= after <= n['before'] for n, after in zip(networks, counts))
            assert np.array_equal(out, solve(inp.copy())), 'incorrect answer'
            assert not np.array_equal(inp, out)
            if split == 'train':
                train_features.append(features)
                train_codes.update(code for n in networks for _, _, code in n['markers'])
    assert len(set(books)) == 1, 'one visibly supplied codebook per puzzle'
    assert set(dict(books[0])) <= train_codes, 'every tool must be used in training'
    for feature in ('partial', 'redundant', 'effect', 'selective', 'linear_fails',
                    'lattice_fails', 'order', 'padding',
                    'tie', 'boundary', 'count_not_markers'):
        assert any(f[feature] for f in train_features), f'missing training witness: {feature}'
    assert {f['color_ascending'] for f in train_features} == {False, True}, 'color order shortcut remains'


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed semantic/diversity audit; no samples or logs are stored."""
    signatures, shapes, books, topologies, sizes = set(), set(), set(), set(), set()
    counts, marker_counts, palettes = set(), set(), set()
    partial = redundant = boundary = selective = 0
    first = None
    for seed in _HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = _puzzle_digest(p)
        signatures.add(digest)
        if seed == 0:
            first = digest
        for split in ('train', 'test'):
            for pair in p[split]:
                _, book, ns = _read_input(pair['input'])
                shapes.add(pair['input'].shape)
                books.add(tuple(sorted(book.items())))
                palettes.add(tuple(sorted(int(v) for v in np.unique(pair['input']))))
                f = _features(ns, book)
                partial += f['partial']
                redundant += f['redundant']
                boundary += f['boundary']
                selective += f['selective']
                counts.add(tuple(sorted(zip(f['before'], f['counts']))))
                for n in ns:
                    marker_counts.add(len(n['markers']))
                    sizes.update(n['sizes'])
                    # Normalize away numeric room dimensions, retaining the
                    # rank order of actual horizontal/vertical wall coordinates.
                    mask = n['mask']
                    ys = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
                    xs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
                    horizontal = tuple(bool(mask[y, a:b + 1].all())
                                       for y in ys for a, b in zip(xs, xs[1:]))
                    vertical = tuple(bool(mask[a:b + 1, x].all())
                                     for x in xs for a, b in zip(ys, ys[1:]))
                    topologies.add((len(ys), len(xs), horizontal, vertical))
    assert len(signatures) == 42 and len(books) >= 35
    assert len(shapes) >= 35 and len(counts) >= 45
    assert len(topologies) >= 50 and len(marker_counts) >= 8
    assert len(sizes) >= 35 and len(palettes) >= 12
    assert min(min(s) for s in sizes) == 1 and max(max(s) for s in sizes) >= 12
    assert min(partial, redundant, boundary, selective) >= 42
    assert first == _puzzle_digest(generate_puzzle(0))
    # Simultaneity is invariant to dictionary iteration order and rotations.
    p = generate_puzzle(23, 2, 1)
    for pair in p['train'] + p['test']:
        _, book, ns = _read_input(pair['input'])
        reverse = dict(reversed(list(book.items())))
        assert all(_punch(n, book)[0] == _punch(n, reverse)[0] for n in ns)
        for n in ns:
            h, w = n['mask'].shape
            rotated = dict(n, mask=np.rot90(n['mask']), markers=[
                (w - 1 - x, y, code) for y, x, code in n['markers']])
            rb = {code: tuple((-dx, dy) for dy, dx in offsets) for code, offsets in book.items()}
            assert _punch(n, book)[0] == _punch(rotated, rb)[0]
    # A label mutation cannot redefine live markers or the supplied dictionary.
    q = p['test'][0]
    q['output'] = q['output'].copy()
    q['output'][0, 0] = 0
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt test answer accepted')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(991, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
