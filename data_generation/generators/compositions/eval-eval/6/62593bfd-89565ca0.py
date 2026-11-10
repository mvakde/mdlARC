"""Matching locks open doors: rotational glyph equality changes room topology.

No parent code, files, RNG, or construction records are used by solve().
Generation uses room boxes and union-find; solving reconstructs pixels and
flood-fills the opened walls. Only standard-library modules and NumPy are used.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import itertools
import numpy as np


HOST_SEEDS = (0, 622376416, 936969842, 510932933, 533291821, 200491724,
              563242594, 467668084, 478979598, 346823055, 675252198, 718165967,
              378053616, 675282472, 512422168, 33352329, 858659788, 264807176,
              303497544, 216039753, 50967422, 817299048, 722078473, 496979985,
              272466380, 361669062, 48255474, 307985756, 131580364, 778635438,
              954839148, 277149298, 582082704, 373008161, 927224595, 664535856,
              879498501, 673760015, 995921918, 781270624, 798906026, 534893648)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('grids must be integer rectangles of size 1..30 and color 0..9')
    return a.astype(np.int8, copy=False)


def _crop(mask):
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()


def _key(mask):
    m = _crop(mask)
    return (m.shape[0], m.shape[1], m.tobytes())


def _canonical(mask):
    return min(_key(np.rot90(mask, k)) for k in range(4))


def _connected(mask, directions=DIR8):
    cells = set(map(tuple, np.argwhere(mask)))
    todo = [min(cells)]
    seen = set(todo)
    while todo:
        y, x = todo.pop()
        for dy, dx in directions:
            p = y+dy, x+dx
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _bounded_regions(walls):
    """Actual bounded 4-connected negative spaces, even at canvas edges."""
    free = np.pad(~walls, 1, constant_values=True)
    seen = np.zeros_like(free)
    regions = []
    h, w = free.shape
    for y in range(h):
        for x in range(w):
            if not free[y, x] or seen[y, x]:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            outside = False
            while stack:
                r, c = stack.pop()
                outside |= r in (0, h-1) or c in (0, w-1)
                cells.append((r-1, c-1))
                for dy, dx in DIR4:
                    nr, nc = r+dy, c+dx
                    if (0 <= nr < h and 0 <= nc < w and free[nr, nc]
                            and not seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not outside:
                regions.append(cells)
    return regions


def _box(cells):
    a = np.asarray(cells)
    return tuple(map(int, (*a.min(axis=0), *a.max(axis=0))))


def _wall_mask(shape, interiors):
    result = np.zeros(shape, dtype=bool)
    for t, l, b, r in interiors:
        result[t-1, l-1:r+2] = result[b+1, l-1:r+2] = True
        result[t-1:b+2, l-1] = result[t-1:b+2, r+1] = True
    return result


def _edges(boxes):
    """Shared wall interiors; corner contact never creates an edge."""
    result = []
    for i, a in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            b = boxes[j]
            if a[2]+2 == b[0] or b[2]+2 == a[0]:
                lo, hi = max(a[1], b[1]), min(a[3], b[3])
                if lo <= hi:
                    y = min(a[2], b[2])+1
                    result.append((i, j, tuple((y, x) for x in range(lo, hi+1))))
            if a[3]+2 == b[1] or b[3]+2 == a[1]:
                lo, hi = max(a[0], b[0]), min(a[2], b[2])
                if lo <= hi:
                    x = min(a[3], b[3])+1
                    result.append((i, j, tuple((y, x) for y in range(lo, hi+1))))
    return result


def _union_count(keys, edges):
    parent = list(range(len(keys)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in edges:
        if keys[i] == keys[j]:
            parent[root(i)] = root(j)
    return len({root(i) for i in range(len(keys))})


def _open(walls, keys, edges):
    opened = walls.copy()
    for i, j, cells in edges:
        if keys[i] == keys[j]:
            for p in cells:
                opened[p] = False
    return opened


def _encode(noise, networks):
    ordered = sorted(networks, key=lambda n: (n['count'], n['left']))
    out = np.full((len(ordered), max(n['count'] for n in ordered)), noise, dtype=np.int8)
    for y, n in enumerate(ordered):
        out[y, :n['count']] = n['color']
    return out


def _decode(input_grid):
    """Input-only role discovery, supported repair, glyph reading and topology."""
    grid = _array(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max())*2 > grid.size, 'background must occupy a strict majority'
    frames = []
    for color in colors:
        if color == bg:
            continue
        t, l, b, r = _box(np.argwhere(grid == color))
        if b-t < 6 or r-l < 6:
            continue
        if (np.all(grid[t, l:r+1] == color) and np.all(grid[b, l:r+1] == color)
                and np.all(grid[t:b+1, l] == color) and np.all(grid[t:b+1, r] == color)):
            frames.append(dict(color=int(color), top=t, left=l, bottom=b, right=r))
    assert 1 <= len(frames) <= 2, 'one or two intact outer frames required'
    boxes_mask = np.zeros_like(grid, dtype=bool)
    for n in frames:
        t, l, b, r = n['top'], n['left'], n['bottom'], n['right']
        assert not boxes_mask[t:b+1, l:r+1].any(), 'outer boxes must be disjoint'
        boxes_mask[t:b+1, l:r+1] = True
    outside_colors = set(map(int, grid[~boxes_mask])) - {bg}
    assert len(outside_colors) == 1, 'exactly one repair color is visible outside frames'
    noise = outside_colors.pop()
    wall_colors = {n['color'] for n in frames}
    for n in frames:
        t, l, b, r = n['top'], n['left'], n['bottom'], n['right']
        local = grid[t:b+1, l:r+1]
        visible = local == n['color']
        walls = visible.copy()
        for y, x in np.argwhere(local == noise):
            y, x = int(y), int(x)
            horizontal = 0 < x < local.shape[1]-1 and visible[y, x-1] and visible[y, x+1]
            vertical = 0 < y < local.shape[0]-1 and visible[y-1, x] and visible[y+1, x]
            assert horizontal or vertical, 'unsupported repair mark inside a frame'
            walls[y, x] = True
        regions = _bounded_regions(walls)
        assert regions
        regions.sort(key=lambda cells: _box(cells)[:2])
        boxes, glyphs, inks = [], [], []
        for cells in regions:
            a, c, d, e = _box(cells)
            assert len(cells) == (d-a+1)*(e-c+1), 'elementary rooms must be rectangles'
            tile = local[a:d+1, c:e+1]
            ink = set(map(int, np.unique(tile))) - {bg}
            assert len(ink) == 1 and not ink & (wall_colors | {noise})
            mask = tile != bg
            assert (not mask[0].any() and not mask[-1].any()
                    and not mask[:, 0].any() and not mask[:, -1].any()), 'glyph clearance required'
            glyph = _crop(mask)
            assert max(glyph.shape) <= 3 and _connected(glyph)
            boxes.append((a, c, d, e))
            glyphs.append(glyph)
            inks.append(ink.pop())
        assert np.array_equal(walls, _wall_mask(local.shape, boxes)), 'unexplained wall pixels'
        edges = _edges(boxes)
        keys = [_canonical(g) for g in glyphs]
        opened = _open(walls, keys, edges)
        n.update(walls=walls, visible=visible, boxes=boxes, glyphs=glyphs,
                 inks=inks, edges=edges, keys=keys, opened=opened,
                 count=len(_bounded_regions(opened)))
    for a, b in itertools.combinations(frames, 2):
        if a['count'] == b['count']:
            assert a['right'] < b['left'] or b['right'] < a['left'], 'ties need clear horizontal order'
    return bg, noise, frames


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return sorted unary counts of chambers after matching doors open."""
    _, noise, frames = _decode(input_grid)
    return _encode(noise, frames)


def _sample_shape(rng, forbidden, *, chiral=False, diagonal=False):
    def permitted(mask):
        k, mirror = _canonical(mask), _canonical(np.fliplr(mask))
        return (k not in forbidden and mirror not in forbidden and _connected(mask)
                and (not chiral or k != mirror)
                and (not diagonal or not _connected(mask, DIR4)))

    for _ in range(300):
        bits = np.zeros(9, dtype=bool)
        bits[rng.choice(9, size=int(rng.integers(3, 8)), replace=False)] = True
        mask = _crop(bits.reshape(3, 3))
        if permitted(mask):
            return mask
    # At most 511 tiny masks; no unbounded retries or growing vocabulary cache.
    for bits in range(1, 512):
        mask = _crop(np.array([(bits >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3))
        if 3 <= int(mask.sum()) <= 7 and permitted(mask):
            return mask
    raise AssertionError('finite shape vocabulary exhausted')


def _capacity(box):
    t, l, b, r = box
    return ((b-t)//6)*((r-l)//6)


def _partition(rng, count, small=False):
    if small:
        h = 6*count+1
        w = int(rng.integers(7, 10))
    else:
        options = [(h, w) for h in range(7, 26) for w in range(7, 20)
                   if count <= ((h-1)//6)*((w-1)//6) <= count+4]
        h, w = options[int(rng.integers(len(options)))]
    rooms = [(0, 0, h-1, w-1)]
    while len(rooms) < count:
        choices = []
        for i, (t, l, b, r) in enumerate(rooms):
            for axis, low, high in ((0, t, b), (1, l, r)):
                for cut in range(low+6, high-5):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    cap = sum(_capacity(z) for j, z in enumerate(rooms) if j != i)
                    if cap + sum(_capacity(z) for z in halves) >= count:
                        choices.append((i, halves))
        assert choices
        i, halves = choices[int(rng.integers(len(choices)))]
        rooms[i:i+1] = halves
    interiors = sorted((t+1, l+1, b-1, r-1) for t, l, b, r in rooms)
    return (h, w), interiors


def _witness_assignment(boxes):
    edges = _edges(boxes)
    adjacent = {frozenset((i, j)) for i, j, _ in edges}
    for remote in itertools.combinations(range(len(boxes)), 2):
        if frozenset(remote) in adjacent:
            continue
        for i, j, _ in edges:
            if set(remote) & {i, j}:
                continue
            for a, b, _ in edges:
                if not (set(remote) | {i, j}) & {a, b}:
                    return remote, (i, j), (a, b)
    return None


def _has_cycle(n, edges):
    # The complete dual graph is connected; E >= V therefore means a cycle.
    return len(edges) >= n


def _has_partial(shape, rooms):
    walls = _wall_mask(shape, rooms)
    ys = {v for t, _, b, _ in rooms for v in (t-1, b+1)}
    xs = {v for _, l, _, r in rooms for v in (l-1, r+1)}
    return any(not walls[y].all() for y in ys) or any(not walls[:, x].all() for x in xs)


def _fallback_partition(count, mode):
    if mode == 'witness':
        shape = (13, 25)
        rooms = [(1+6*y, 1+6*x, 5+6*y, 5+6*x) for y in range(2) for x in range(4)]
        if count == 7:
            rooms[2:4] = [(1, 13, 5, 23)]
    else:
        shape = (13, 13)
        rooms = [(1, 1, 5, 5), (7, 1, 11, 5), (1, 7, 11, 11)]
    return shape, rooms


def _make_network(rng, count, color, palette, bg, mode='random', small=False):
    assignment = None
    for _ in range(60):
        shape, rooms = _partition(rng, count, small)
        edges = _edges(rooms)
        if mode == 'witness':
            assignment = _witness_assignment(rooms)
            if assignment is None:
                continue
        if mode == 'cycle' and (not _has_cycle(count, edges) or not _has_partial(shape, rooms)):
            continue
        break
    else:
        # Explicit finite geometry fallback, still following the same rule.
        shape, rooms = _fallback_partition(count, mode)
        if mode == 'witness':
            assignment = _witness_assignment(rooms)
            assert assignment is not None
        else:
            # Three rectangular rooms, with a true T junction and dual cycle.
            count = 3
        edges = _edges(rooms)
    forbidden, bases = set(), []

    def new_shape(**kwargs):
        m = _sample_shape(rng, forbidden, **kwargs)
        forbidden.update((_canonical(m), _canonical(np.fliplr(m))))
        bases.append(m)
        return len(bases)-1

    labels = [None]*count
    if mode == 'witness':
        remote, matched, mirrors = assignment
        for group, kwargs in ((remote, {}), (matched, {'chiral': True, 'diagonal': True})):
            label = new_shape(**kwargs)
            for i in group:
                labels[i] = label
        label = new_shape(chiral=True)
        labels[mirrors[0]] = label
        bases.append(np.fliplr(bases[label]).copy())
        labels[mirrors[1]] = len(bases)-1
        for i in range(count):
            if labels[i] is None:
                labels[i] = new_shape()
    elif mode in ('cycle', 'all'):
        label = new_shape(chiral=True)
        labels = [label]*count
    else:
        amount = (int(rng.integers(2, min(5, count)+1)) if mode == 'mixed' else
                  int(rng.choice([1, count, int(rng.integers(1, min(5, count)+1))])))
        for _ in range(amount):
            new_shape()
        labels = list(map(int, rng.integers(amount, size=count)))
        for i in range(amount):
            labels[i] = i
        rng.shuffle(labels)
        if mode == 'mixed':
            if not any(labels[i] == labels[j] for i, j, _ in edges):
                a, b, _ = edges[int(rng.integers(len(edges)))]
                labels[b] = labels[a]
            # Keep a genuinely closed boundary as well as an opened one.
            if len(set(labels)) == 1:
                labels[next(i for i in range(count) if i not in (a, b))] = (labels[a]+1) % amount
    turns = list(map(int, rng.integers(4, size=count)))
    inks = list(map(int, rng.choice(palette, size=count)))
    if mode == 'witness':
        a, b = assignment[1]
        turns[b] = (turns[a]+1) % 4
        # Make the color-only count differ by construction, without accidental
        # cancellation between a missed true match and a false mirror match.
        # Removing the lower-degree endpoint leaves at least two edges among
        # the other >=6 vertices, all of which the color shortcut opens.
        degree = Counter(i for i, j, _ in edges)+Counter(j for i, j, _ in edges)
        exceptional = min((a, b), key=lambda i: degree[i])
        inks = [palette[0]]*count
        inks[exceptional] = palette[1]
    glyphs = [_crop(np.rot90(bases[label], turn)) for label, turn in zip(labels, turns)]
    walls = _wall_mask(shape, rooms)
    tile = np.full(shape, bg, dtype=np.int8)
    tile[walls] = color
    for (t, l, b, r), mask, ink in zip(rooms, glyphs, inks):
        y = int(rng.integers(t+1, b-mask.shape[0]+1))
        x = int(rng.integers(l+1, r-mask.shape[1]+1))
        view = tile[y:y+mask.shape[0], x:x+mask.shape[1]]
        view[mask] = ink
    keys = [_canonical(g) for g in glyphs]
    return dict(color=color, tile=tile, walls=walls, rooms=rooms, edges=edges,
                keys=keys, count=_union_count(keys, edges))


def _damage(rng, network, noise, critical=False):
    tile, walls = network['tile'], network['walls']
    # Only straight internal wall cells; immediate support is never damaged.
    candidates = []
    h, w = walls.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if not walls[y, x]:
                continue
            neighbors = [bool(walls[y+dy, x+dx]) for dy, dx in DIR4]
            if neighbors in ([True, True, False, False], [False, False, True, True]):
                candidates.append((y, x))
    if not candidates:
        return
    protected = set()
    selected = []
    if critical:
        closed = {p for i, j, cells in network['edges'] if network['keys'][i] != network['keys'][j]
                  for p in cells}
        eligible = [p for p in candidates if p in closed]
        assert eligible
        selected.append(eligible[int(rng.integers(len(eligible)))])
    amount = int(rng.integers(1, min(6, len(candidates))+1))
    for p in selected:
        protected.update((p[0]+dy, p[1]+dx) for dy, dx in DIR8)
        protected.add(p)
    for k in rng.permutation(len(candidates)):
        p = candidates[int(k)]
        if len(selected) >= amount:
            break
        if p not in protected:
            selected.append(p)
            protected.update((p[0]+dy, p[1]+dx) for dy, dx in DIR8)
            protected.add(p)
    for p in selected:
        tile[p] = noise


def _place(rng, networks, bg, noise, order=False):
    if len(networks) == 2:
        a, b = networks
        # Put a long narrow frame upright if that permits horizontal packing.
        for n in networks:
            if n['tile'].shape[1] > n['tile'].shape[0]:
                n['tile'] = np.rot90(n['tile'])
        h1, w1 = a['tile'].shape
        h2, w2 = b['tile'].shape
        tie = a['count'] == b['count']
        gap = int(rng.integers(1, min(3, 30-w1-w2)+1))
        vertical = not tie and h1+h2+1 <= 28 and rng.random() < .4
        if vertical:
            needed_h, needed_w = h1+h2+gap, max(w1, w2)+2
            positions = [(0, 0), (h1+gap, 1)]
            if not order and rng.random() < .5:
                positions = [(h2+gap, 1), (0, 0)]
        else:
            needed_h, needed_w = max(h1, h2), w1+w2+gap
            positions = [(int(rng.integers(needed_h-h1+1)), 0),
                         (int(rng.integers(needed_h-h2+1)), w1+gap)]
            if not order and rng.random() < .5:
                positions = [(positions[0][0], w2+gap), (positions[1][0], 0)]
    else:
        tile = networks[0]['tile']
        if rng.random() < .5:
            networks[0]['tile'] = tile.T
        needed_h, needed_w = networks[0]['tile'].shape
        positions = [(0, 0)]
    assert needed_h <= 30 and needed_w <= 30
    height = needed_h + int(rng.integers(min(4, 30-needed_h)+1))
    width = needed_w + int(rng.integers(min(4, 30-needed_w)+1))
    # Reserve exterior background both for role discovery and a clear majority.
    foreground = sum(int(np.count_nonzero(n['tile'] != bg)) for n in networks)+4
    while height*width <= 2*foreground or height*width-needed_h*needed_w < 5:
        if height < 30 and (height <= width or width == 30):
            height += 1
        elif width < 30:
            width += 1
        else:
            raise AssertionError('canvas density could not be bounded')
    yoff, xoff = int(rng.integers(height-needed_h+1)), int(rng.integers(width-needed_w+1))
    grid = np.full((height, width), bg, dtype=np.int8)
    occupied = np.zeros_like(grid, dtype=bool)
    for n, (y, x) in zip(networks, positions):
        y, x = y+yoff, x+xoff
        h, w = n['tile'].shape
        assert not occupied[y:y+h, x:x+w].any()
        grid[y:y+h, x:x+w] = n['tile']
        occupied[y:y+h, x:x+w] = True
        n['left'] = x
    sites = np.argwhere(~occupied)
    amount = int(rng.integers(1, min(4, len(sites))+1))
    for y, x in sites[rng.choice(len(sites), size=amount, replace=False)]:
        grid[y, x] = noise
    return dict(input=grid, output=_encode(noise, networks))


def _make_pair(rng, mode, fixed_roles):
    colors = list(map(int, rng.permutation(10)))
    if fixed_roles is not None:
        bg, noise = fixed_roles
        colors = [bg, noise]+[c for c in colors if c not in fixed_roles]
    bg, noise = colors[:2]
    ink_palette = colors[4:4+int(rng.integers(2, 7))]
    if mode == 'witness':
        counts, modes = [int(rng.integers(7, 9)), int(rng.integers(1, 4))], ['witness', 'random']
    elif mode == 'cycle':
        counts, modes = [int(rng.integers(3, 7)), int(rng.integers(1, 4))], ['cycle', 'all']
    elif mode in ('test', 'test_active'):
        counts = [int(rng.integers(9, 13))]
        if rng.random() < .65:
            counts.append(int(rng.integers(1, 4)))
        modes = ['mixed' if mode == 'test_active' else 'random']+['random']*(len(counts)-1)
    else:
        counts = [int(rng.integers(1, 9))]
        if rng.random() < .65:
            counts.append(int(rng.integers(1, 4)))
        modes = ['random']*len(counts)
    networks = [_make_network(rng, count, colors[2+i], ink_palette, bg, submode, small=i > 0)
                for i, (count, submode) in enumerate(zip(counts, modes))]
    for i, network in enumerate(networks):
        if mode in ('witness', 'cycle', 'test_active') or rng.random() < .8:
            _damage(rng, network, noise, critical=mode in ('witness', 'test_active') and i == 0)
    return _place(rng, networks, bg, noise, order=mode == 'witness')


def _legal_counts(num_train, num_test):
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError('num_train must be 2..10 and num_test must be 1..4 integers')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    fixed_roles = tuple(map(int, rng.choice(10, size=2, replace=False))) if rng.random() < .25 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = (('witness' if i == 0 else 'cycle' if i == 1 else 'random') if split == 'train'
                    else 'test_active' if i == 0 else 'test')
            for _ in range(12):
                pair = _make_pair(rng, mode, fixed_roles)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('distinct input budget exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    bg, noise, frames = _decode(pair['input'])
    assert np.array_equal(_encode(noise, frames), np.asarray(pair['output'])), 'incorrect output'
    f = dict(background=bg, noise=noise, shape=tuple(np.asarray(pair['input']).shape),
             networks=len(frames), rooms=[], chambers=[], partial=False, damaged=False,
             A_witness=False, B_witness=False, remote=False, reflection=False,
             rotation=False, colors=False, cycle=False, triplet=False, diagonal=False,
             topology=[], glyphs=[])
    for n in frames:
        keys, edges, glyphs = n['keys'], n['edges'], n['glyphs']
        assert n['count'] == _union_count(keys, edges), 'flood/union disagreement'
        assert _union_count([0]*len(keys), edges) == 1, 'room adjacency must be connected'
        f['rooms'].append(len(keys))
        f['chambers'].append(n['count'])
        f['A_witness'] |= n['count'] != len(keys)
        f['damaged'] |= not np.array_equal(n['walls'], n['visible'])
        # Ignoring repair opens at least one wrong boundary on the first witness.
        unrepaired = n['opened'] & n['visible']
        f['B_witness'] |= len(_bounded_regions(unrepaired)) != n['count']
        f['remote'] |= len(set(keys)) != n['count']
        matched_edges = sum(keys[i] == keys[j] for i, j, _ in edges)
        f['cycle'] |= len(keys)-matched_edges != n['count']
        freq = Counter(keys)
        exact_two_keys = [(k if freq[k] == 2 else ('singleton', i)) for i, k in enumerate(keys)]
        f['triplet'] |= _union_count(exact_two_keys, edges) != n['count']
        reflected_keys = [min(_canonical(g), _canonical(np.fliplr(g))) for g in glyphs]
        f['reflection'] |= _union_count(reflected_keys, edges) != n['count']
        f['rotation'] |= _union_count([_key(g) for g in glyphs], edges) != n['count']
        f['colors'] |= _union_count(n['inks'], edges) != n['count']
        f['diagonal'] |= any(not _connected(g, DIR4) for g in glyphs)
        for (t, l, b, r) in n['boxes']:
            # A full lattice has no divider ending inside the outer frame.
            for y, x in ((t-1, l-1), (t-1, r+1), (b+1, l-1), (b+1, r+1)):
                if 0 < y < n['walls'].shape[0]-1 and 0 < x < n['walls'].shape[1]-1:
                    degree = sum(bool(n['walls'][y+dy, x+dx]) for dy, dx in DIR4)
                    f['partial'] |= degree == 3
        ys = sorted({v for a, _, b, _ in n['boxes'] for v in (a-1, b+1)})
        xs = sorted({v for _, a, _, b in n['boxes'] for v in (a-1, b+1)})
        f['topology'].append(tuple((ys.index(t-1), xs.index(l-1), ys.index(b+1), xs.index(r+1))
                                   for t, l, b, r in n['boxes']))
        f['glyphs'].extend(keys)
    f['tie'] = len(frames) == 2 and frames[0]['count'] == frames[1]['count']
    f['padding'] = min(f['chambers']) < max(f['chambers'])
    spatial = sorted(frames, key=lambda n: n['left'])
    f['order'] = [n['count'] for n in spatial] != sorted(f['chambers'])
    return f


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _legal_counts(len(puzzle['train']), len(puzzle['test']))
    features, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            key = _fingerprint(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            assert not np.array_equal(inp, out)
            features[split].append(_features(pair))
    train = features['train']
    for name in ('A_witness', 'B_witness', 'remote', 'reflection', 'rotation', 'colors',
                 'cycle', 'triplet', 'diagonal', 'partial', 'damaged', 'tie', 'padding', 'order'):
        assert any(f[name] for f in train), 'missing demonstration: '+name
    assert all(max(f['rooms']) <= 8 for f in train)
    assert all(9 <= max(f['rooms']) <= 12 for f in features['test'])
    assert any(f['A_witness'] and f['B_witness'] for f in features['test'])


def self_test():
    """Streaming 42-seed audit; no saved puzzles, traces, logs, or RNG state."""
    for count in (7, 8):
        shape, rooms = _fallback_partition(count, 'witness')
        assert len(rooms) == count and _witness_assignment(rooms) is not None
        assert len(_bounded_regions(_wall_mask(shape, rooms))) == count
    shape, rooms = _fallback_partition(6, 'cycle')
    assert len(rooms) == 3 and _has_partial(shape, rooms) and _has_cycle(3, _edges(rooms))
    assert len(_bounded_regions(_wall_mask(shape, rooms))) == 3
    signatures, shapes, topologies, glyphs, role_colors, profiles = set(), set(), set(), set(), set(), set()
    flags = Counter()
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        digest = sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                pairs += 1
                for role in ('input', 'output'):
                    a = np.asarray(pair[role], dtype=np.uint8)
                    digest.update(bytes(a.shape)+a.tobytes())
                f = _features(pair)
                shapes.add(f['shape'])
                topologies.update(f['topology'])
                glyphs.update(f['glyphs'])
                role_colors.add(f['background'])
                profiles.add((tuple(sorted(f['rooms'])), tuple(sorted(f['chambers']))))
                for name in ('networks', 'partial', 'tie', 'A_witness', 'B_witness', 'cycle', 'remote'):
                    flags[(name, f[name])] += 1
                # All color roles, including padding, are equivariant.
                if seed == 0:
                    permutation = np.array([8, 6, 4, 2, 0, 9, 7, 5, 3, 1], dtype=np.int8)
                    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                    if not f['tie']:
                        assert np.array_equal(solve(np.rot90(pair['input'])), pair['output'])
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(topologies) >= 35 and len(glyphs) >= 20 and len(profiles) >= 35
    assert role_colors == set(range(10))
    assert flags[('networks', 1)] and flags[('networks', 2)]
    assert all(flags[(name, True)] >= 42 for name in ('A_witness', 'B_witness', 'cycle', 'remote', 'partial'))
    for counts in ((2, 1), (10, 4)):
        p = generate_puzzle(12345, *counts)
        q = generate_puzzle(12345, *counts)
        assert all(_fingerprint(a[r]) == _fingerprint(b[r]) for s in p
                   for a, b in zip(p[s], q[s]) for r in a)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                scale_free_partitions=len(topologies), rotational_glyph_classes=len(glyphs),
                room_chamber_profiles=len(profiles), backgrounds=len(role_colors))
