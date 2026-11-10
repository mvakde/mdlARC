"""Periodic subtractive stencils distinguish designed voids from damage.

No file I/O, parent imports, hidden per-seed conventions, or saved examples.
The only dependencies are the standard library and NumPy.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _d4(array, index):
    return np.rot90(array if index < 4 else np.fliplr(array), index % 4)


def _components(mask):
    """Four-connected component bounding boxes and sizes, in scan order."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        ys, xs = zip(*cells)
        result.append((min(ys), min(xs), max(ys)+1, max(xs)+1, len(cells)))
    return result


def _perimeter(shape):
    m = np.ones(shape, dtype=bool)
    m[1:-1, 1:-1] = False
    return m


def _separated(a, b):
    # Half-open boxes, with at least one intervening row or column.
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int64, copy=False)


def _parse(input_grid):
    """Recover registered cards and maximal plates using only this input."""
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    candidates = []
    for frame in sorted(set(map(int, np.unique(inp))) - {bg}):
        components = _components(inp == frame)
        if not 2 <= len(components) <= 4:
            continue
        patterns, boxes, sizes = {}, [], set()
        for y, x, b, r, n in components:
            side = b-y
            if side != r-x or side < 4 or n != 4*side-4:
                break
            window = inp[y:b, x:r]
            edge = _perimeter(window.shape)
            if not np.all(window[edge] == frame) or np.any(window[~edge] == frame):
                break
            interior = window[1:-1, 1:-1]
            inks = set(map(int, np.unique(interior))) - {bg}
            if len(inks) != 1:
                break
            color = next(iter(inks))
            mask = interior == color
            if color in patterns or not 0 < mask.sum() < mask.size:
                break
            patterns[color] = mask.copy()
            boxes.append((y, x, b, r))
            sizes.add(side-2)
        else:
            if len(sizes) == 1 and all(_separated(a, b) for a, b in itertools.combinations(boxes, 2)):
                candidates.append((frame, patterns, boxes, sizes.pop()))
    assert len(candidates) == 1, 'ambiguous or absent stencil frames'
    frame, patterns, cards, k = candidates[0]
    card_area = np.zeros(inp.shape, dtype=bool)
    for y, x, b, r in cards:
        card_area[y:b, x:r] = True
    plates = []
    for color in sorted(patterns):
        for y, x, b, r, _ in _components((inp == color) & ~card_area):
            h, w = b-y, r-x
            if min(h, w) < 2*k+2 or (h-2) % k or (w-2) % k:
                continue
            window = inp[y:b, x:r]
            if np.all(window[_perimeter(window.shape)] == color):
                plates.append((y, x, b, r, color))
    # A loop inside a plate is material geometry, not another plate.
    plates = [a for a in plates if not any(
        a != b and b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2] and b[3] >= a[3]
        for b in plates)]
    plates.sort()
    assert 2 <= len(plates) <= 4, 'expected two to four plates'
    fills = tuple(sorted({p[4] for p in plates}))
    assert len(fills) == 2 and set(fills) <= set(patterns)
    boxes = [p[:4] for p in plates] + cards
    assert all(_separated(a, b) for a, b in itertools.combinations(boxes, 2))
    occupied = card_area.copy()
    for y, x, b, r, _ in plates:
        assert y >= 1 and x >= 1 and b < inp.shape[0] and r < inp.shape[1]
        occupied[y:b, x:r] = True
    assert np.all(inp[~occupied] == bg), 'unexplained exterior content'
    assert all(np.all(np.isin(inp[y:b, x:r], (bg, *fills))) for y, x, b, r, _ in plates)
    return {'input': inp, 'background': bg, 'highlight': frame, 'patterns': patterns,
            'cards': cards, 'plates': plates, 'fills': fills, 'k': k}


def _material(shape, mask):
    h, w = shape
    k = mask.shape[0]
    m = np.ones(shape, dtype=bool)
    m[1:-1, 1:-1] = ~np.tile(mask, ((h-2)//k, (w-2)//k))
    return m


def _dilate8(mask):
    result = np.zeros_like(mask)
    h, w = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            result[max(0, dy):min(h, h+dy), max(0, dx):min(w, w+dx)] |= mask[
                max(0, -dy):min(h, h-dy), max(0, -dx):min(w, w-dx)]
    return result


def _crop(plates):
    return (min(p[0] for p in plates)-1, min(p[1] for p in plates)-1,
            max(p[2] for p in plates)+1, max(p[3] for p in plates)+1)


def _render(info, *, orientation=0, mode='normal'):
    """Render from decoded or construction geometry; modes are audit ablations."""
    inp, bg, highlight, fills = (info[t] for t in ('input', 'background', 'highlight', 'fills'))
    out = np.full(inp.shape, bg, dtype=np.int64)
    for y, x, b, r, color in info['plates']:
        stencil_color = next(c for c in fills if c != color) if mode == 'swap' else color
        mask = _d4(info['patterns'][stencil_color], orientation)
        material = _material((b-y, r-x), mask)
        if mode == 'solid':
            material[:] = True
        elif mode == 'copy':
            material[1:-1, 1:-1] ^= True
        elif mode == 'once':
            material[1:-1, 1:-1] = True
            material[1:1+mask.shape[0], 1:1+mask.shape[1]] = ~mask
        observed = inp[y:b, x:r]
        clean = np.where(material, color, bg)
        erased = material & (observed == bg)
        if mode == 'all_errors':
            erased = observed != clean
        local = clean.copy()
        if mode != 'no_mark':
            halo = _dilate8(erased)
            if mode != 'ungated':
                halo &= material
            other = color if mode == 'same_fill' else next(c for c in fills if c != color)
            local[halo] = other
            if mode != 'no_centers':
                local[erased] = highlight
        out[y:b, x:r] = local
    y, x, b, r = _crop(info['plates'])
    return out[y:b, x:r].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """A pure input-only solution, independent of seed and generation state."""
    return _render(_parse(input_grid))


def _mask(rng, k, witness=False):
    for _ in range(128):
        mode = int(rng.integers(8))
        m = np.zeros((k, k), dtype=bool)
        if mode == 0:
            np.fill_diagonal(m, True)
        elif mode == 1:
            m[int(rng.integers(k)), :] = True
        elif mode == 2:
            m[int(rng.integers(k)), :] = True
            m[:, int(rng.integers(k))] = True
        elif mode == 3:
            m[0, 0] = m[-1, -1] = True
        elif mode == 4:
            m[:] = True
            m.flat[int(rng.integers(k*k))] = False
        else:
            m = rng.random((k, k)) < float(rng.uniform(.15, .65))
        m = _d4(m, int(rng.integers(8))).copy()
        if not 0 < m.sum() < m.size:
            continue
        if witness and (len({_d4(m, i).tobytes() for i in range(8)}) != 8
                        or not 2 <= m.sum() <= 4):
            continue
        return m
    # An asymmetric k=3 motif with abundant adjacent retained material.
    m = np.zeros((k, k), dtype=bool)
    m[0, 0] = m[1, -1] = m[-1, 1] = True
    return m


def _shelf_placements(rng, sizes):
    """Variable row breaks and offsets; all permutations are unnecessary."""
    result = []
    n = len(sizes)
    for _ in range(3):
        order = list(map(int, rng.permutation(n)))
        for breaks in range(1 << (n-1)):
            rows, current = [], []
            for i, index in enumerate(order):
                current.append(index)
                if i == n-1 or breaks & (1 << i):
                    rows.append(current)
                    current = []
            widths = [sum(sizes[i][1] for i in row)+len(row)-1 for row in rows]
            heights = [max(sizes[i][0] for i in row) for row in rows]
            h, w = sum(heights)+len(rows)-1, max(widths)
            if h > 28 or w > 28:
                continue
            positions = [None]*n
            yy = 0
            for row, rh, rw in zip(rows, heights, widths):
                xx = int(rng.integers(w-rw+1))
                for i in row:
                    dy = int(rng.integers(rh-sizes[i][0]+1))
                    positions[i] = (yy+dy, xx)
                    xx += sizes[i][1]+1
                yy += rh+1
            result.append((h, w, positions))
    return result


def _layout(rng, sizes, scatter):
    """Random free placement, then a finite shelf fallback with free offsets."""
    if scatter:
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0]*sizes[i][1])
        for _ in range(10):
            occupied = np.zeros((28, 28), dtype=bool)
            pos = [None]*len(sizes)
            for i in order:
                h, w = sizes[i]
                candidates = [(y, x) for y in range(29-h) for x in range(29-w)]
                rng.shuffle(candidates)
                for y, x in candidates:
                    if not occupied[y:y+h, x:x+w].any():
                        pos[i] = (y, x)
                        occupied[max(0, y-1):min(28, y+h+1), max(0, x-1):min(28, x+w+1)] = True
                        break
                else:
                    break
            else:
                return 28, 28, pos
    options = _shelf_placements(rng, sizes)
    return _pick(rng, options) if options else None


def _make_pair(rng, *, count, k, maximum_repeat, grow, palette, teach, axis=None):
    bg, highlight, c0, c1, *distractor_pool = palette
    fills = (c0, c1)
    for attempt in range(64):
        # One card per fill; distractors consume neither fill role.
        nd = int(rng.integers(3)) if not teach else int(rng.integers(1, 3))
        colors = [c0, c1] + distractor_pool[:nd]
        patterns = {c: _mask(rng, k, witness=teach and i == 0) for i, c in enumerate(colors)}
        if teach and np.array_equal(patterns[c0], patterns[c1]):
            continue
        repeats = [(int(rng.integers(2, maximum_repeat+1)), int(rng.integers(2, maximum_repeat+1)))
                   for _ in range(count)]
        if grow:
            a, b = repeats[0]
            repeats[0] = (4, b) if rng.integers(2) else (a, 4)
        if attempt >= 40:
            repeats = [(2, 2)]*count
            if grow:
                repeats[0] = (4, 2)
        sizes = [(a*k+2, b*k+2) for a, b in repeats] + [(k+2, k+2)]*len(colors)
        layout = _layout(rng, sizes, scatter=axis is None and rng.random() < .55)
        if layout is None:
            continue
        h, w, positions = layout
        # The layout-shift regime uses separated horizontal/vertical plate rows.
        if axis is not None:
            plate_h = max(s[0] for s in sizes[:count])
            plate_w = sum(s[1] for s in sizes[:count])+count-1
            card_h = k+2
            card_w = len(colors)*(k+3)-1
            if plate_h+card_h+1 > 28 or max(plate_w, card_w) > 28:
                continue
            h, w = plate_h+card_h+1, max(plate_w, card_w)
            positions = []
            xx = int(rng.integers(w-plate_w+1))
            for ph, pw in sizes[:count]:
                positions.append((int(rng.integers(plate_h-ph+1)), xx))
                xx += pw+1
            xx = int(rng.integers(w-card_w+1))
            positions += [(plate_h+1, xx+i*(k+3)) for i in range(len(colors))]
        # Padding and object-to-slot permutations are unrelated to matching.
        extra_h = int(rng.integers(min(3, 28-h)+1))
        extra_w = int(rng.integers(min(3, 28-w)+1))
        top = 1+int(rng.integers(extra_h+1))
        left = 1+int(rng.integers(extra_w+1))
        inp = np.full((h+2+extra_h, w+2+extra_w), bg, dtype=np.int64)
        plate_colors = [c0, c1] + [int(_pick(rng, fills)) for _ in range(count-2)]
        rng.shuffle(plate_colors)
        plates, materials = [], []
        for i, color in enumerate(plate_colors):
            yy, xx = positions[i]
            y, x = top+yy, left+xx
            ph, pw = sizes[i]
            material = _material((ph, pw), patterns[color])
            inp[y:y+ph, x:x+pw] = np.where(material, color, bg)
            plates.append((y, x, y+ph, x+pw, color))
            materials.append(material)
        rng.shuffle(colors)
        cards = []
        for (yy, xx), color in zip(positions[count:], colors):
            y, x = top+yy, left+xx
            inp[y:y+k+2, x:x+k+2] = highlight
            inp[y+1:y+k+1, x+1:x+k+1] = np.where(patterns[color], color, bg)
            cards.append((y, x, y+k+2, x+k+2))
        possible = True
        for j, ((y, x, b, r, color), material) in enumerate(zip(plates, materials)):
            eligible = material.copy()
            eligible[_perimeter(material.shape)] = False
            points = [tuple(map(int, p)) for p in np.argwhere(eligible)]
            rng.shuffle(points)
            if teach and j == 0:
                # A same-plate adjacent pair beside an intentional cutout.
                options = []
                for p in points:
                    if not _dilate8(~material)[p]:
                        continue
                    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                        q = (p[0]+dy, p[1]+dx)
                        if 0 < q[0] < material.shape[0]-1 and 0 < q[1] < material.shape[1]-1 and eligible[q]:
                            options.append((p, q))
                if not options:
                    possible = False
                    break
                erased = list(_pick(rng, options))
            else:
                erased = [points[0]]
            target = int(rng.integers(max(2, len(erased)), max(3, min(9, len(points)//8)+1)))
            erased += [p for p in points if p not in erased][:max(0, target-len(erased))]
            local = inp[y:b, x:r]
            for p in erased:
                local[p] = bg
            wrong = [p for p in points if p not in erased]
            number_wrong = int(rng.integers(1 if teach else 0, min(3, len(wrong))+1))
            for p in wrong[:number_wrong]:
                local[p] = next(c for c in fills if c != color)
            voids = [tuple(map(int, p)) for p in np.argwhere(~material)]
            rng.shuffle(voids)
            number_speckles = int(rng.integers(1 if teach else 0, min(3, len(voids))+1))
            for p in voids[:number_speckles]:
                local[p] = int(_pick(rng, fills))
        if not possible:
            continue
        construction = {'input': inp, 'background': bg, 'highlight': highlight,
                        'patterns': patterns, 'plates': plates, 'fills': fills}
        out = _render(construction)
        # Rotating the entire registered scene preserves its displayed phase.
        transform = int(rng.integers(8)) if axis is None else int(_pick(rng, (0, 2, 4, 6)))
        if axis == 'vertical':
            transform += 1
        inp, out = _d4(inp, transform).copy(), _d4(out, transform).copy()
        try:
            recovered = _parse(inp)
            if not np.array_equal(_render(recovered), out):
                continue
        except AssertionError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded packing/evidence retries exhausted')


def _key(array):
    return array.shape, array.astype(np.uint8).tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.uint8).tobytes())
    return digest.hexdigest()


def _evidence(infos, outputs):
    alternatives = {f'orientation_{i}': {'orientation': i} for i in range(1, 8)}
    alternatives.update({name: {'mode': name} for name in
                         ('solid', 'copy', 'once', 'no_mark', 'ungated', 'same_fill',
                          'no_centers', 'all_errors', 'swap')})
    return {name: any(not np.array_equal(_render(a, **options), out)
                      for a, out in zip(infos, outputs))
            for name, options in alternatives.items()}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, outputs = set(), [], []
    adjacent = both_fills = gated = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse(inp)
            expected = _render(info)
            assert np.array_equal(expected, out), 'input-derived inspection disagrees'
            assert not np.array_equal(inp, out)
            erased_fills = set()
            for y, x, b, r, color in info['plates']:
                m = _material((b-y, r-x), info['patterns'][color])
                holes = m & (inp[y:b, x:r] == info['background'])
                assert not holes[_perimeter(m.shape)].any(), 'input border must be intact'
                if holes.any():
                    erased_fills.add(color)
                if split == 'train':
                    adjacent |= bool((_dilate8(holes) & holes).any())
                    gated |= bool((_dilate8(holes) & ~m).any())
            assert erased_fills, 'no damage to inspect'
            if split == 'train':
                both_fills |= erased_fills == set(info['fills'])
                training.append(info)
                outputs.append(out)
    assert adjacent and both_fills and gated, 'training omits an interaction'
    evidence = _evidence(training, outputs)
    assert all(evidence.values()), f'training permits alternatives: {evidence}'


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # period growth, plate growth, layout axis
    recurring_palette = bool(rng.integers(2))
    base_palette = list(map(int, rng.permutation(10)))
    train_axis = _pick(rng, ('horizontal', 'vertical'))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', num_train), ('test', num_test)):
        for index in range(total):
            teach = split == 'train' and index < 2
            k = 3 if teach else int(rng.integers(2, 5))
            if regime == 2:
                k = 3 if teach else int(rng.integers(2, 4))
            count = 2 if teach or regime in (1, 2) and split == 'train' else int(rng.integers(2, 5))
            if regime == 1 and split == 'test':
                count = int(rng.integers(3, 5))
            if regime == 0 and split == 'test':
                count = 2
            axis = None
            if regime == 2:
                # Exactly two plates make a visible, well-defined axis shift.
                count = 2
                axis = train_axis if split == 'train' else ('vertical' if train_axis == 'horizontal' else 'horizontal')
            palette = base_palette if recurring_palette else list(map(int, rng.permutation(10)))
            for _ in range(8):
                pair = _make_pair(rng, count=count, k=k,
                                  maximum_repeat=3 if regime == 0 and split == 'train' else 4,
                                  grow=regime == 0 and split == 'test',
                                  palette=palette, teach=teach, axis=axis)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded duplicate retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """42 streamed seeds, semantic probes, and deterministic resource bounds."""
    shapes, signatures, normalized = set(), set(), set()
    periods, plate_counts, backgrounds, frame_colors, mask_shapes = set(), set(), set(), set(), set()
    metrics = {'seeds': 42, 'pairs': 0, 'adjacent_pairs': 0, 'gated_pairs': 0,
               'symmetric_masks': 0, 'asymmetric_masks': 0, 'blank_mask_edges': 0,
               'repeated_fill_pairs': 0, 'distractor_counts': set()}
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                a = _parse(inp)
                shapes.add(inp.shape)
                plate_counts.add(len(a['plates']))
                backgrounds.add(a['background'])
                frame_colors.add(a['highlight'])
                mask_shapes.add(a['k'])
                metrics['distractor_counts'].add(len(a['cards'])-2)
                metrics['repeated_fill_pairs'] += len(a['plates']) > 2
                clean = np.full(inp.shape, a['background'], dtype=int)
                centers = np.zeros(inp.shape, dtype=bool)
                rings = []
                adjacent = gated = False
                for y, x, b, r, color in a['plates']:
                    m = _material((b-y, r-x), a['patterns'][color])
                    periods.add(((b-y-2)//a['k'], (r-x-2)//a['k']))
                    holes = m & (inp[y:b, x:r] == a['background'])
                    adjacent |= bool((_dilate8(holes) & holes).any())
                    gated |= bool((_dilate8(holes) & ~m).any())
                    clean[y:b, x:r] = np.where(m, color, a['background'])
                    centers[y:b, x:r] = holes
                    # A separate point-based implementation checks the vectorized renderer.
                    other = next(c for c in a['fills'] if c != color)
                    demands = set()
                    for yy, xx in np.argwhere(holes):
                        for dy, dx in itertools.product((-1, 0, 1), repeat=2):
                            q = int(yy)+dy, int(xx)+dx
                            if (dy or dx) and 0 <= q[0] < m.shape[0] and 0 <= q[1] < m.shape[1] and m[q]:
                                demands.add((y+q[0], x+q[1]))
                    rings.append((demands, other))
                expected = clean.copy()
                for demands, color in reversed(rings):
                    for q in demands:
                        expected[q] = color
                expected[centers] = a['highlight']
                y, x, b, r = _crop(a['plates'])
                assert np.array_equal(expected[y:b, x:r], out)
                metrics['pairs'] += 1
                metrics['adjacent_pairs'] += adjacent
                metrics['gated_pairs'] += gated
                for m in a['patterns'].values():
                    asym = len({_d4(m, i).tobytes() for i in range(8)}) == 8
                    metrics['asymmetric_masks'] += asym
                    metrics['symmetric_masks'] += not asym
                    metrics['blank_mask_edges'] += not all((m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any()))
                roles = {}
                normalized.add((inp.shape, bytes(roles.setdefault(int(v), len(roles)) for v in inp.flat)))
                if seed < 3:
                    permutation = np.array([(i+3) % 10 for i in range(10)])
                    assert np.array_equal(solve(permutation[inp]), permutation[out])
                    assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
    assert len(signatures) == 42 and len(normalized) == metrics['pairs']
    assert len(shapes) >= 25 and mask_shapes == {2, 3, 4} and plate_counts == {2, 3, 4}
    assert len(periods) >= 8 and backgrounds == frame_colors == set(range(10))
    assert metrics['distractor_counts'] == {0, 1, 2}
    assert min(metrics[t] for t in ('adjacent_pairs', 'gated_pairs', 'symmetric_masks',
                                  'asymmetric_masks', 'blank_mask_edges', 'repeated_fill_pairs')) > 10
    assert _fingerprint(generate_puzzle(7)) == _fingerprint(generate_puzzle(7))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1200+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(9, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    p = generate_puzzle(9, 2, 1)
    a = _parse(p['test'][0]['input'])
    color = a['fills'][0]
    card = next(box for box in a['cards'] if np.any(
        a['input'][box[0]+1:box[2]-1, box[1]+1:box[3]-1] == color))
    yy, xx = card[0]+1, card[1]+1
    p['test'][0]['input'][yy, xx] = color if p['test'][0]['input'][yy, xx] == a['background'] else a['background']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed stencil with stale label accepted')
    metrics['distractor_counts'] = sorted(metrics['distractor_counts'])
    metrics.update({'input_shapes': len(shapes), 'period_count_shapes': len(periods),
                    'distinct_puzzles': len(signatures), 'normalized_inputs': len(normalized)})
    return metrics
