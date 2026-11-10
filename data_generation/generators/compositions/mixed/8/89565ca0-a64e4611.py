"""Repair colored partitions; serialize rooms and their eroded negative space.

Only Python and NumPy are required. No files, parent code, or saved examples are
read by generation or solving. All diagnostics in self_test stay in memory.
"""

from collections import deque
import hashlib
import numpy as np


HOST_SEEDS = (0, 835988364, 566267052, 924636311, 382713064, 240610660,
              202748613, 197514603, 165446721, 919521790, 588724042,
              633638433, 373007565, 637050250, 101497032, 785855212,
              129087613, 581480299, 443348487, 293678515, 836202432,
              377162330, 860996032, 701166709, 490676634, 987048372,
              500667939, 652729488, 870021481, 512147861, 303233456,
              351429529, 25319220, 284555854, 593104351, 533406359,
              217401926, 227401948, 615800330, 588484499, 107681777,
              247721668)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _regions(mask):
    """Four-connected free components inside a closed rectangular frame."""
    h, w = mask.shape
    seen = mask.copy()
    rooms = []
    for r, c in np.argwhere(~mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        coords = np.array(cells)
        t, l = map(int, coords.min(axis=0))
        b, r = map(int, coords.max(axis=0))
        if t == 0 or l == 0 or b == h-1 or r == w-1:
            raise ValueError("unclosed frame")
        if len(cells) != (b-t+1) * (r-l+1):
            raise ValueError("elementary rooms must be rectangular")
        rooms.append((t, l, b, r))
    return rooms


def _bounded_count(mask):
    """Count arbitrary bounded free components for the no-repair ablation."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = ~free
    count = 0
    h, w = free.shape
    for r, c in np.argwhere(free):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, outside = [(r, c)], False
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            outside |= y == 0 or x == 0 or y == h-1 or x == w-1
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        count += not outside
    return count


def _wall_union(shape, boxes):
    walls = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        walls[t, l:r+1] = walls[b, l:r+1] = True
        walls[t:b+1, l] = walls[t:b+1, r] = True
    return walls


def _empty_centers(ink):
    """Clipped Chebyshev-radius-one exclusion; count each center once."""
    h, w = ink.shape
    padded = np.pad(ink, 1, constant_values=False)
    near = np.zeros_like(ink)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy+h, dx:dx+w]
    return ~near


def _recover(grid, color, background):
    """Supported collinear completion through occupied pixels, never background.

    Each true line has an adjacent same-color pair. Each elementary segment has
    some same-color support; missing pixels are occupied by the noise or another
    network. Open background witnesses distinguish absent partial continuations.
    """
    visible = grid == color
    coords = np.argwhere(visible)
    if len(coords) < 8:
        return None
    t, l = map(int, coords.min(axis=0))
    b, r = map(int, coords.max(axis=0))
    local = grid[t:b+1, l:r+1]
    vis = local == color
    ys = np.flatnonzero(np.any(vis[:, :-1] & vis[:, 1:], axis=1))
    xs = np.flatnonzero(np.any(vis[:-1, :] & vis[1:, :], axis=0))
    if len(ys) < 2 or len(xs) < 2:
        return None
    if (ys[0], xs[0], ys[-1], xs[-1]) != (0, 0, b-t, r-l):
        return None
    mask = vis.copy()
    for y in ys:
        for x1, x2 in zip(xs, xs[1:]):
            line = local[y, x1:x2+1]
            if np.all(line != background) and np.any(line == color):
                mask[y, x1:x2+1] = True
    for x in xs:
        for y1, y2 in zip(ys, ys[1:]):
            line = local[y1:y2+1, x]
            if np.all(line != background) and np.any(line == color):
                mask[y1:y2+1, x] = True
    if not (mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all()):
        return None
    try:
        rooms = _regions(mask)
    except ValueError:
        return None
    if not rooms:
        return None
    boundaries = [(a-1, d-1, c+1, e+1) for a, d, c, e in rooms]
    if not np.array_equal(mask, _wall_union(mask.shape, boundaries)):
        return None
    for y in ys:
        for x1, x2 in zip(xs, xs[1:]):
            if mask[y, x1] and mask[y, x2] and not mask[y, x1:x2+1].all():
                if not np.any(local[y, x1+1:x2] == background):
                    return None
    for x in xs:
        for y1, y2 in zip(ys, ys[1:]):
            if mask[y1, x] and mask[y2, x] and not mask[y1:y2+1, x].all():
                if not np.any(local[y1+1:y2, x] == background):
                    return None
    return dict(color=int(color), box=(t, l, b, r), mask=mask, rooms=rooms,
                c=len(rooms), e=int(_empty_centers(mask).sum()))


def _decode(grid):
    colors, freq = np.unique(grid, return_counts=True)
    if len(colors) < 3 or np.count_nonzero(freq == freq.max()) != 1:
        raise ValueError("input needs unique background, networks, and noise")
    background = int(colors[np.argmax(freq)])
    networks, noise = [], []
    for color in colors:
        if color == background:
            continue
        network = _recover(grid, int(color), background)
        if network is None:
            noise.append(int(color))
        else:
            networks.append(network)
    if len(noise) != 1 or not networks:
        raise ValueError("exactly one nonstructural noise color is required")
    networks.sort(key=lambda n: (n['c'], n['box'][1], n['box'][0]))
    return background, noise[0], networks


def _encode(background, noise, networks):
    ordered = sorted(networks, key=lambda n: (n['c'], n['box'][1], n['box'][0]))
    width = max(n['c'] + n['e'] for n in ordered)
    if width > 30:
        raise ValueError("unary row exceeds ARC width; padding is never truncated")
    out = np.full((len(ordered), width), background, dtype=np.int8)
    for row, n in enumerate(ordered):
        out[row, :n['c']] = n['color']
        out[row, n['c']:n['c']+n['e']] = noise
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from visible roles and repaired input geometry."""
    background, noise, networks = _decode(_grid(input_grid))
    return _encode(background, noise, networks)


def _partition(rng, h, w, count):
    boxes = [(0, 0, h-1, w-1)]
    for _ in range(count-1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for y in range(t+2, b-1):
                halves = [(t, l, y, r), (y, l, b, r)]
                capacity = sum(((d-a)//2)*((e-f)//2)
                               for a, f, d, e in boxes[:i]+boxes[i+1:]+halves)
                if capacity >= count:
                    choices.append((i, halves))
            for x in range(l+2, r-1):
                halves = [(t, l, b, x), (t, x, b, r)]
                capacity = sum(((d-a)//2)*((e-f)//2)
                               for a, f, d, e in boxes[:i]+boxes[i+1:]+halves)
                if capacity >= count:
                    choices.append((i, halves))
        if not choices:
            return None
        i, halves = choices[int(rng.integers(len(choices)))]
        boxes[i:i+1] = halves
    return _wall_union((h, w), boxes)


def _orient(mask, rng):
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _narrow_partial(rng):
    """Many rooms, zero eroded core, and a genuine partial divider."""
    count = int(rng.integers(3, 6))
    widths = list(map(int, rng.integers(1, 3, size=2)))
    gaps = list(map(int, rng.integers(2, 5, size=count-1)))
    ys = [0] + list(map(int, np.cumsum(gaps)))
    split, right = widths[0]+1, sum(widths)+2
    boxes = [(0, 0, ys[-1], split)]
    boxes += [(ys[k], split, ys[k+1], right) for k in range(count-1)]
    return _orient(_wall_union((ys[-1]+1, right+1), boxes), rng)


def _random_mask(rng, count, maxh, maxw, e_range=(0, 24)):
    for _ in range(180):
        h, w = int(rng.integers(3, maxh+1)), int(rng.integers(3, maxw+1))
        if ((h-1)//2)*((w-1)//2) < count:
            continue
        mask = _partition(rng, h, w, count)
        if mask is None:
            continue
        e = int(_empty_centers(mask).sum())
        if e_range[0] <= e <= min(e_range[1], 30-count):
            return mask
    # Finite safety construction: count horizontal strips, with optional core.
    # Only used when dimensions support it; caller can retry the geometry.
    core = e_range[0]
    gaps = [2] * count
    if core:
        gaps[0] = core+3
    ys = [0] + list(map(int, np.cumsum(gaps)))
    width = 5 if core else 3
    mask = _wall_union((ys[-1]+1, width),
                       [(ys[i], 0, ys[i+1], width-1) for i in range(count)])
    if mask.shape[0] <= maxh and mask.shape[1] <= maxw:
        return mask
    return None


def _place(rng, masks, shape, counts, witness, allow_overlap):
    h, w = shape
    networks = []
    # Witnesses have clear horizontal order; ordinary scenes may overlap.
    overlap = allow_overlap and witness is None and rng.random() < .35
    for index, mask in enumerate(masks):
        mh, mw = mask.shape
        if mh > h or mw > w:
            return None
        accepted = None
        for _ in range(90):
            t, l = int(rng.integers(h-mh+1)), int(rng.integers(w-mw+1))
            b, r = t+mh-1, l+mw-1
            if witness and index and networks[0]['box'][3] + 1 >= l:
                continue
            valid = True
            for previous in networks:
                pt, pl, pb, pr = previous['box']
                intersects = not (b < pt or pb < t or r < pl or pr < l)
                if intersects and (not overlap or counts[index] == previous['c']):
                    valid = False
                    break
                if counts[index] == previous['c'] and not (r < pl or pr < l):
                    valid = False
                    break
            if valid:
                accepted = dict(mask=mask, box=(t, l, b, r), c=counts[index])
                break
        if accepted is None:
            return None
        networks.append(accepted)
    return networks


def _make_pair(rng, background, noise, witness=None, test=False, allow_overlap=True):
    for attempt in range(180):
        if witness == 'order':
            high = _narrow_partial(rng)
            lowc = int(rng.integers(1, 3))
            low = _random_mask(rng, lowc, 14, 12, (7, 18))
            if low is None:
                continue
            masks = [high, low]
            counts = [len(_regions(high)), lowc]
        elif witness == 'tie':
            c = int(rng.integers(2, 4))
            masks = [_random_mask(rng, c, 14, 11, (1, 4)),
                     _random_mask(rng, c, 14, 11, (5, 15))]
            counts = [c, c]
        else:
            number = int(rng.choice([3, 3, 4])) if test else int(rng.choice([1, 2, 2, 3, 4]))
            counts = list(map(int, rng.choice(np.arange(1, 7), size=number, replace=False)))
            maxdim = 12 if number >= 3 else 22
            masks = [_random_mask(rng, c, maxdim, maxdim) for c in counts]
        if any(m is None for m in masks):
            continue
        # Different shapes, offsets, counts, and local topology, not just colors.
        h, w = map(int, rng.integers(22 if len(masks) >= 3 else 18, 31, size=2))
        if witness:
            minimum_width = max(23, sum(m.shape[1] for m in masks)+2)
            if minimum_width > 30:
                continue
            w = int(rng.integers(minimum_width, 31))
        ns = _place(rng, masks, (h, w), counts, witness, allow_overlap)
        if ns is None:
            continue
        palette = [c for c in range(10) if c not in (background, noise)]
        colors = list(map(int, rng.choice(palette, size=len(ns), replace=False)))
        grid = np.full((h, w), background, dtype=np.int8)
        for n, color in zip(ns, colors):
            n['color'] = color
            n['e'] = int(_empty_centers(n['mask']).sum())
        for idx in rng.permutation(len(ns)):
            n = ns[int(idx)]
            t, l, b, r = n['box']
            grid[t:b+1, l:r+1][n['mask']] = n['color']
        # One long interruption guarantees that raw geometry gives wrong windows.
        if witness == 'order':
            n = ns[1]
            t, l, b, r = n['box']
            candidates = []
            for y in (t, b):
                for x in range(l+1, r-2):
                    if np.all(grid[y, x:x+3] == n['color']):
                        candidates.append([(y, xx) for xx in range(x, x+3)])
            for x in (l, r):
                for y in range(t+1, b-2):
                    if np.all(grid[y:y+3, x] == n['color']):
                        candidates.append([(yy, x) for yy in range(y, y+3)])
            if not candidates:
                continue
            for y, x in candidates[int(rng.integers(len(candidates)))]:
                grid[y, x] = noise
        for n in ns:
            if rng.random() < .75:
                sites = np.argwhere(grid == n['color'])
                k = min(len(sites), int(rng.integers(1, 4)))
                for y, x in sites[rng.choice(len(sites), size=k, replace=False)]:
                    grid[y, x] = noise
        # Scatter is ignored after isolating a repaired network. It is not ink.
        free = np.argwhere(grid == background)
        for y, x in free[rng.choice(len(free), size=int(rng.integers(2, 9)), replace=False)]:
            grid[y, x] = noise
        try:
            bg, nscolor, decoded = _decode(grid)
        except ValueError:
            continue
        intended = {n['color']: n for n in ns}
        if bg != background or nscolor != noise or len(decoded) != len(ns):
            continue
        if not all(n['color'] in intended and n['box'] == intended[n['color']]['box']
                   and np.array_equal(n['mask'], intended[n['color']]['mask'])
                   and n['c'] == intended[n['color']]['c'] for n in decoded):
            continue
        out = _encode(background, noise, ns)
        pair = dict(input=grid, output=out)
        features = _features(pair, (bg, nscolor, decoded))
        if witness == 'order' and not all(features[k] for k in
                                         ('order', 'length_inversion', 'repair_windows', 'repair_counts', 'partial', 'zero')):
            continue
        if witness == 'tie' and not (features['small'] and features['tie'] and features['multiple_cores']):
            continue
        return pair
    raise RuntimeError("bounded generation failed to retain required visible evidence")


def _features(pair, decoded=None):
    g = np.asarray(pair['input'])
    bg, noise, ns = decoded if decoded is not None else _decode(g)
    lengths = [n['c'] + n['e'] for n in ns]
    spatial = sorted(ns, key=lambda n: (n['box'][1], n['box'][0]))
    partial = repair_windows = repair_counts = multiple_cores = False
    ink_layers = np.zeros(g.shape, dtype=np.int8)
    for n in ns:
        mask = n['mask']
        multiple_cores |= sum(b-a >= 2 and d-c >= 2 for a, c, b, d in n['rooms']) >= 2
        ys = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
        xs = np.flatnonzero(np.any(mask[:-1, :] & mask[1:, :], axis=0))
        partial |= any(not mask[y, :].all() for y in ys) or any(not mask[:, x].all() for x in xs)
        t, l, b, r = n['box']
        ink_layers[t:b+1, l:r+1] += mask
        raw = g[t:b+1, l:r+1] == n['color']
        repair_windows |= int(_empty_centers(raw).sum()) != n['e']
        repair_counts |= _bounded_count(raw) != n['c']
    return dict(order=[n['c'] for n in spatial] != sorted(n['c'] for n in ns),
                length_inversion=any(a['c'] < b['c'] and a['c']+a['e'] > b['c']+b['e']
                                     for a in ns for b in ns),
                partial=partial, repair_windows=repair_windows, repair_counts=repair_counts,
                overlap=bool(np.any(ink_layers > 1)), multiple_cores=multiple_cores,
                tie=len({n['c'] for n in ns}) < len(ns),
                zero=any(n['e'] == 0 for n in ns), small=any(1 <= n['e'] <= 4 for n in ns),
                padding=any(n['e'] > 0 for n in ns), ragged=len(set(lengths)) > 1)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    background, noise = map(int, rng.choice(10, size=2, replace=False))
    puzzle, seen = dict(train=[], test=[]), set()
    overlap_seen = False
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            witness = ('order' if i == 0 else 'tie') if split == 'train' and i < 2 else None
            for _ in range(8):
                pair = _make_pair(rng, background, noise, witness, split == 'test',
                                  allow_overlap=split == 'train' or overlap_seen)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        overlap_seen |= _features(pair)['overlap']
                    break
            else:
                raise RuntimeError('distinct inputs not found within retry budget')
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, roles, evidence, test_features = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(g) not in seen
            seen.add(_key(g))
            bg, noise, ns = _decode(g)
            roles.add((bg, noise))
            assert np.array_equal(out, _encode(bg, noise, ns))
            for n in ns:
                mask, rooms = n['mask'], n['rooms']
                assert 1 <= n['c'] <= 6 and 0 <= n['e'] <= 24 and n['c']+n['e'] <= 30
                # Independent rectangular-room formula checks all-window counting.
                expected_e = sum(max(0, b-t-1)*max(0, r-l-1) for t, l, b, r in rooms)
                assert n['e'] == expected_e
                # Planar Euler check on the pixel wall graph, without flood fill.
                edges = int((mask[:, :-1] & mask[:, 1:]).sum()) + int((mask[:-1] & mask[1:]).sum())
                filled = int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum())
                assert edges - int(mask.sum()) + 1 - filled == n['c']
            for a in ns:
                for b in ns:
                    if a is not b and a['c'] == b['c']:
                        assert a['box'][3] < b['box'][1] or b['box'][3] < a['box'][1]
            if split == 'train':
                evidence.append(_features(pair, (bg, noise, ns)))
            else:
                test_features.append(_features(pair, (bg, noise, ns)))
    assert len(roles) == 1
    for key in ('order', 'length_inversion', 'repair_windows', 'repair_counts', 'partial', 'zero', 'small', 'tie', 'padding', 'ragged', 'multiple_cores'):
        assert any(f[key] for f in evidence), f'missing training witness: {key}'
    assert not any(f['overlap'] for f in test_features) or any(f['overlap'] for f in evidence)


def self_test():
    """Bounded 42-seed audit, streamed; no artifacts and no retained grids."""
    shapes, outputs, structures, partitions, roles, empty_values = set(), set(), set(), set(), set(), set()
    overlap = intact = damaged = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                g, out = pair['input'], pair['output']
                bg, noise, ns = _decode(g)
                roles.add((bg, noise))
                shapes.add(g.shape)
                outputs.add(out.shape)
                digest.update(bytes(g.shape) + g.tobytes())
                canvas = np.zeros(g.shape, dtype=int)
                for n in ns:
                    t, l, b, r = n['box']
                    canvas[t:b+1, l:r+1] += n['mask']
                    partitions.add((n['mask'].shape, n['mask'].tobytes()))
                    empty_values.add(n['e'])
                    damaged += bool(np.any(g[t:b+1, l:r+1][n['mask']] != n['color']))
                    intact += bool(np.all(g[t:b+1, l:r+1][n['mask']] == n['color']))
                    # Omitting B removes the noise run, whenever e > 0.
                    if n['e']:
                        assert np.count_nonzero(out == noise) > 0
                overlap += bool(np.any(canvas > 1))
                # Geometric invariance for distinct compartment counts.
                if seed in HOST_SEEDS[:3] and len({n['c'] for n in ns}) == len(ns):
                    for transformed in (g.T, np.rot90(g), np.fliplr(g)):
                        assert np.array_equal(solve(transformed), out)
                    permutation = np.array([7, 5, 9, 1, 0, 8, 2, 6, 4, 3], dtype=np.int8)
                    assert np.array_equal(solve(permutation[g]), permutation[out])
        structures.add(digest.hexdigest())
    assert len(structures) == 42 and len(shapes) >= 45 and len(outputs) >= 25
    assert len(partitions) >= 140 and len(roles) >= 20 and len(empty_values) >= 15
    assert overlap > 0 and intact > 0 and damaged > 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(17+nt*5+ne, nt, ne)
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    return dict(seeds=42, input_shapes=len(shapes), output_shapes=len(outputs),
                network_geometries=len(partitions), role_assignments=len(roles),
                empty_counts=len(empty_values), overlapping_pairs=overlap,
                intact_networks=intact, damaged_networks=damaged)
