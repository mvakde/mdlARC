"""Room-bundle keys: enclosure-derived virtual pins and exact-cover docking.

Only NumPy and the standard library are used. The solver reads the input alone.
0 is background, 5 is damage/speckle, 9 is the card-dot hue. All other color
roles are inferred locally. Ground truth is painted from sampled clean room
boxes, independently of the reconstruction used by solve().
"""
from itertools import combinations
import numpy as np

BACKGROUND = 0
DAMAGE = 5
PIN = 9
STRUCTURAL = (1, 2, 3, 4, 6, 7, 8)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _box(cells):
    a = np.asarray(sorted(cells), dtype=int)
    return tuple(map(int, (*a.min(axis=0), *a.max(axis=0))))


def _normalize(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _ring(shape):
    m = np.zeros(shape, dtype=bool)
    m[[0, -1], :] = True
    m[:, [0, -1]] = True
    return m


def _walls(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _regions(mask):
    """Bounded four-connected empty regions; padded exterior handles edges."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(free):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo = [(r, c)]
        cells = []
        outside = False
        while todo:
            y, x = todo.pop()
            cells.append((y - 1, x - 1))
            outside |= y in (0, free.shape[0] - 1) or x in (0, free.shape[1] - 1)
            for dy, dx in N4:
                yy, xx = y + dy, x + dx
                if (0 <= yy < free.shape[0] and 0 <= xx < free.shape[1]
                        and free[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        if not outside:
            result.append(set(cells))
    return result


def _cycle_count(mask):
    """Independent planar graph count for a connected thin wall network."""
    v = int(mask.sum())
    e = int((mask[1:, :] & mask[:-1, :]).sum())
    e += int((mask[:, 1:] & mask[:, :-1]).sum())
    tiny = int((mask[:-1, :-1] & mask[1:, :-1]
                & mask[:-1, 1:] & mask[1:, 1:]).sum())
    return e - v + 1 - tiny


def _opposite(mask):
    vertical = np.zeros_like(mask)
    horizontal = np.zeros_like(mask)
    vertical[1:-1, :] = mask[:-2, :] & mask[2:, :]
    horizontal[:, 1:-1] = mask[:, :-2] & mask[:, 2:]
    return vertical | horizontal


def _repair(local, color):
    observed = local == color
    return observed | ((local == DAMAGE) & _opposite(observed))


def _full_lattice(mask):
    ys = np.flatnonzero(np.any(mask[:, 1:] & mask[:, :-1], axis=1))
    xs = np.flatnonzero(np.any(mask[1:, :] & mask[:-1, :], axis=0))
    full = np.zeros_like(mask)
    full[ys, :] = True
    full[:, xs] = True
    return full


def _topology_key(mask):
    """Remove repeated rows/columns, keeping divider endpoints and their order."""
    rows = np.r_[True, np.any(mask[1:] != mask[:-1], axis=1)]
    compressed = mask[rows]
    columns = np.r_[True, np.any(compressed[:, 1:] != compressed[:, :-1], axis=0)]
    compressed = compressed[:, columns]
    return tuple(tuple(map(int, row)) for row in compressed)


def _noncollinear(cells):
    points = sorted(cells)
    if len(points) < 3:
        return False
    a, b = points[:2]
    return any((b[0] - a[0]) * (c[1] - a[1])
               != (b[1] - a[1]) * (c[0] - a[0]) for c in points[2:])


def _decode(input_grid):
    g = _check_grid(input_grid)
    colors = [int(c) for c in np.unique(g) if c in STRUCTURAL]
    cards, boards, occupied = [], [], np.zeros(g.shape, dtype=bool)
    accounted_pins = set()
    for color in colors:
        cells = set(map(tuple, np.argwhere(g == color).tolist()))
        t, l, b, r = _box(cells)
        if b - t < 2 or r - l < 2:
            raise ValueError("structural hue lacks a rectangular extent")
        if occupied[t:b + 1, l:r + 1].any():
            raise ValueError("object rectangles overlap")
        occupied[t:b + 1, l:r + 1] = True
        local = g[t:b + 1, l:r + 1]
        dots = set(map(tuple, np.argwhere(local == PIN).tolist()))
        if dots:
            border = _ring(local.shape)
            if not np.array_equal(local == color, border):
                raise ValueError("a key must have exactly its intact colored perimeter")
            if not np.all(np.isin(local[~border], [BACKGROUND, PIN])):
                raise ValueError("key interiors contain only background and dots")
            if any(y < 2 or x < 2 or y > local.shape[0] - 3
                   or x > local.shape[1] - 3 for y, x in dots):
                raise ValueError("dots require an empty cell between them and the rim")
            absolute = _shift(dots, (t, l))
            accounted_pins |= absolute
            cards.append(dict(color=color, dots=absolute, pattern=_normalize(dots),
                              box=(t, l, b, r)))
        else:
            if not np.all(np.isin(local, [BACKGROUND, DAMAGE, color])):
                raise ValueError("board has foreign structural cells")
            mask = _repair(local, color)
            rooms = _regions(mask)
            if not rooms:
                raise ValueError("board has no bounded rooms")
            room_boxes, centers = [], []
            for room in rooms:
                rt, rl, rb, rr = _box(room)
                if len(room) != (rb - rt + 1) * (rr - rl + 1):
                    raise ValueError("rooms must be rectangles")
                if (rb - rt) % 2 or (rr - rl) % 2:
                    raise ValueError("room interiors must have odd dimensions")
                room_boxes.append((rt + t, rl + l, rb + t, rr + l))
                centers.append(((rt + rb) // 2 + t, (rl + rr) // 2 + l))
            boundaries = [(rt - t - 1, rl - l - 1, rb - t + 1, rr - l + 1)
                          for rt, rl, rb, rr in room_boxes]
            if not np.array_equal(mask, _walls(local.shape, boundaries)):
                raise ValueError("walls must be exactly the room boundary union")
            if not (mask[0].all() and mask[-1].all()
                    and mask[:, 0].all() and mask[:, -1].all()):
                raise ValueError("board outer boundary is incomplete")
            boards.append(dict(color=color, mask=mask, box=(t, l, b, r),
                               rooms=room_boxes, centers=centers,
                               observed=local == color, local=local))
    if len(boards) != 1 or not 2 <= len(cards) <= 4:
        raise ValueError("expected one board and two through four keys")
    if accounted_pins != set(map(tuple, np.argwhere(g == PIN).tolist())):
        raise ValueError("every dot must be inside a key")
    board = boards[0]
    if not 4 <= len(board['rooms']) <= 10:
        raise ValueError("board room count must be 4..10")
    if any(not 1 <= len(c['dots']) <= 4 for c in cards):
        raise ValueError("key size must be 1..4")
    if sum(len(c['dots']) >= 2 for c in cards) < 2:
        raise ValueError("need at least two multi-dot keys")
    if sum(len(c['dots']) for c in cards) != len(board['centers']):
        raise ValueError("pins and rooms have different total counts")
    return g, board, cards


def _candidates(centers, patterns):
    targets = set(centers)
    result = []
    for pattern in patterns:
        reference = pattern[0]
        choices = []
        for target in sorted(targets):
            delta = target[0] - reference[0], target[1] - reference[1]
            moved = frozenset(_shift(pattern, delta))
            if moved <= targets:
                choices.append((moved, delta))
        result.append(choices)
    return result


def _covers(centers, patterns):
    """Enumerate at most two answers, considering every oriented translation."""
    candidates = _candidates(centers, patterns)
    answers = []
    targets = set(centers)

    def visit(assigned, used):
        if len(assigned) == len(patterns):
            if used == targets:
                answers.append(dict(assigned))
            return
        rows = []
        for i in range(len(patterns)):
            if i not in assigned:
                options = [(subset, d) for subset, d in candidates[i] if not subset & used]
                if not options:
                    return
                rows.append((len(options), i, options))
        _, i, options = min(rows, key=lambda row: (row[0], row[1]))
        for subset, delta in options:
            assigned[i] = (subset, delta)
            visit(assigned, used | set(subset))
            del assigned[i]
            if len(answers) == 2:
                return

    visit({}, set())
    return answers, candidates


def _infer(input_grid):
    g, board, cards = _decode(input_grid)
    answers, candidates = _covers(board['centers'], [c['pattern'] for c in cards])
    if len(answers) != 1:
        raise ValueError("docking cover is missing or ambiguous")
    ownership = {}
    for index, (subset, _) in answers[0].items():
        ownership.update({center: cards[index]['color'] for center in subset})
    out = np.zeros(g.shape, dtype=np.int8)
    t, l, b, r = board['box']
    view = out[t:b + 1, l:r + 1]
    view[board['mask']] = board['color']
    for box, center in zip(board['rooms'], board['centers']):
        rt, rl, rb, rr = box
        out[rt:rb + 1, rl:rr + 1] = ownership[center]
    return out, board, cards, answers[0], candidates


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible walls, dots, card hues, and fixed conventions."""
    return _infer(input_grid)[0]


def _partition(rng, count, style):
    """Rectilinear partitions on even wall coordinates; no lattice assumption."""
    for _ in range(40):
        uh, uw = map(int, rng.integers(3, 8, size=2))
        if style == 'strip':
            uh, uw = int(rng.integers(2, 5)), count + int(rng.integers(0, 2))
            if rng.random() < 0.5:
                uh, uw = uw, uh
        if uh * uw < count:
            continue
        outer = (0, 0, 2 * uh, 2 * uw)
        if style == 'grid':
            factors = [(nr, count // nr) for nr in range(2, count)
                       if count % nr == 0 and nr <= uh and count // nr <= uw]
            if not factors:
                continue
            nr, nc = factors[int(rng.integers(len(factors)))]
            ys = [0] + sorted(map(int, rng.choice(np.arange(1, uh), nr - 1, replace=False))) + [uh]
            xs = [0] + sorted(map(int, rng.choice(np.arange(1, uw), nc - 1, replace=False))) + [uw]
            boxes = [(2 * ys[i], 2 * xs[j], 2 * ys[i + 1], 2 * xs[j + 1])
                     for i in range(nr) for j in range(nc)]
        else:
            boxes = [outer]
            for step in range(count - 1):
                options = []
                for i, (t, l, b, r) in enumerate(boxes):
                    for axis in (0, 1):
                        if style == 'strip' and axis != int(uw >= uh):
                            continue
                        start, end = (t, b) if axis == 0 else (l, r)
                        for cut in range(start + 2, end - 1, 2):
                            halves = [(t, l, cut, r), (cut, l, b, r)] if axis == 0 else [(t, l, b, cut), (t, cut, b, r)]
                            capacity = sum(((bb - tt) // 2) * ((rr - ll) // 2)
                                           for tt, ll, bb, rr in halves)
                            capacity += sum(((bb - tt) // 2) * ((rr - ll) // 2)
                                            for j, (tt, ll, bb, rr) in enumerate(boxes) if j != i)
                            if capacity >= count:
                                options.append((i, halves))
                if not options:
                    break
                i, halves = options[int(rng.integers(len(options)))]
                boxes[i:i + 1] = halves
            if len(boxes) != count:
                continue
        shape = (2 * uh + 1, 2 * uw + 1)
        mask = _walls(shape, boxes)
        if style == 'partial' and np.array_equal(mask, _full_lattice(mask)):
            continue
        return mask, boxes
    return None


def _groups(rng, centers, sizes):
    remaining = set(centers)
    groups = []
    shuffled_sizes = list(sizes)
    rng.shuffle(shuffled_sizes)
    for size in shuffled_sizes:
        ordered = sorted(remaining)
        start = ordered[int(rng.integers(len(ordered)))]
        group = {start}
        remaining.remove(start)
        for _ in range(size - 1):
            ordered = sorted(remaining, key=lambda q: (min(abs(q[0] - p[0]) + abs(q[1] - p[1])
                                                         for p in group), q))
            chosen = ordered[int(rng.integers(min(3, len(ordered))))]
            group.add(chosen)
            remaining.remove(chosen)
        groups.append(group)
    return groups


def _sizes(rng, count, keys, maximum=3):
    possibilities = []
    # Small bounded enumeration (4**4 at most), independent of color and layout.
    def visit(values, left):
        if len(values) == keys:
            # Two singleton cards have identical signatures and can always
            # exchange their hues. Such a size request cannot be unique.
            if left == 0 and sum(x >= 2 for x in values) >= 2 and values.count(1) <= 1:
                possibilities.append(values)
            return
        for value in range(1, maximum + 1):
            if value <= left:
                visit(values + [value], left - value)
    visit([], count)
    if not possibilities:
        return None
    return possibilities[int(rng.integers(len(possibilities)))]


def _key_grid(rng, group, color):
    t, l, b, r = _box(group)
    margins = [2 + int(rng.random() < 0.3) for _ in range(4)]
    mt, ml, mb, mr = margins
    shape = (b - t + mt + mb + 1, r - l + ml + mr + 1)
    key = np.zeros(shape, dtype=np.int8)
    key[_ring(shape)] = color
    for y, x in group:
        key[y - t + mt, x - l + ml] = PIN
    return key


def _pack(rng, shapes):
    area = sum(h * w for h, w in shapes)
    choices = [(h, w) for h in range(max(14, max(s[0] for s in shapes)), 31)
               for w in range(max(14, max(s[1] for s in shapes)), 31)
               if h * w >= min(900, 2.1 * area)]
    if not choices or area > 470:
        return None
    for restart in range(18):
        if restart >= 12:
            h, w = 30, 30
        else:
            h, w = choices[int(rng.integers(len(choices)))]
        blocked = np.zeros((h, w), dtype=bool)
        order = list(rng.permutation(len(shapes)))
        order.sort(key=lambda i: -max(shapes[i]) * (shapes[i][0] * shapes[i][1]) ** 0.5)
        placements = {}
        for i in order:
            hh, ww = shapes[i]
            for _ in range(150):
                t = int(rng.integers(h - hh + 1))
                l = int(rng.integers(w - ww + 1))
                if blocked[t:t + hh, l:l + ww].any():
                    continue
                placements[i] = (t, l)
                blocked[max(0, t - 1):min(h, t + hh + 1),
                        max(0, l - 1):min(w, l + ww + 1)] = True
                break
            else:
                break
        if len(placements) == len(shapes):
            return (h, w), placements
    return None


def _equal_size_different(patterns):
    return any(len(a) == len(b) and a != b for a, b in combinations(patterns, 2))


def _damage_board(rng, clean, color):
    mask = clean == color
    neighbor_count = np.zeros_like(mask, dtype=int)
    neighbor_count[1:] += mask[:-1]
    neighbor_count[:-1] += mask[1:]
    neighbor_count[:, 1:] += mask[:, :-1]
    neighbor_count[:, :-1] += mask[:, 1:]
    safe = mask & _opposite(mask) & (neighbor_count == 2)
    candidates = np.argwhere(safe)
    rng.shuffle(candidates)
    damaged = clean.copy()
    used = set()
    target = int(rng.integers(1, 4))
    for y, x in candidates:
        q = int(y), int(x)
        if any(abs(q[0] - a) + abs(q[1] - b) <= 1 for a, b in used):
            continue
        damaged[q] = DAMAGE
        used.add(q)
        if len(used) == target:
            break
    if not used or not np.array_equal(_repair(damaged, color), mask):
        return None
    # Speckle cannot masquerade as a supported wall interruption.
    sites = np.argwhere((damaged == 0) & ~_opposite(mask))
    if len(sites):
        amount = min(len(sites), int(rng.integers(0, 4)))
        for y, x in sites[rng.choice(len(sites), amount, replace=False)]:
            damaged[y, x] = DAMAGE
    return damaged


def _make_pair(rng, count, sizes, witness=None, style=None):
    for _ in range(180):
        chosen_style = style or str(rng.choice(['strip', 'grid', 'partial', 'partial']))
        if chosen_style == 'grid' and not any(count % n == 0 for n in range(2, count)):
            chosen_style = 'partial'
        partition = _partition(rng, count, chosen_style)
        if partition is None:
            continue
        mask, boxes = partition
        centers = [((t + b) // 2, (l + r) // 2) for t, l, b, r in boxes]
        room_sizes = [(b - t - 1, r - l - 1) for t, l, b, r in boxes]
        for _ in range(12):
            groups = _groups(rng, centers, sizes)
            patterns = [_normalize(group) for group in groups]
            answers, candidates = _covers(centers, patterns)
            if len(answers) != 1:
                continue
            if witness == 'global' and (not _equal_size_different(patterns)
                    or max(map(len, candidates)) < 2 or len(set(room_sizes)) == 1
                    or np.array_equal(mask, _full_lattice(mask))):
                continue
            if witness == 'triple' and not any(len(g) == 3 and _noncollinear(g) for g in groups):
                continue
            hues = list(map(int, rng.choice(STRUCTURAL, len(groups) + 1, replace=False)))
            board_color, key_colors = hues[0], hues[1:]
            key_grids = [_key_grid(rng, group, color) for group, color in zip(groups, key_colors)]
            packed = _pack(rng, [mask.shape] + [g.shape for g in key_grids])
            if packed is None:
                continue
            shape, positions = packed
            board = np.zeros(mask.shape, dtype=np.int8)
            board[mask] = board_color
            damaged = _damage_board(rng, board, board_color)
            if damaged is None:
                continue
            if witness == 'global' and len(_regions(damaged == board_color)) == count:
                continue
            inp = np.zeros(shape, dtype=np.int8)
            out = np.zeros(shape, dtype=np.int8)
            bt, bl = positions[0]
            bh, bw = mask.shape
            inp[bt:bt + bh, bl:bl + bw] = damaged
            painted = board.copy()
            # Labels from construction: clean rectangular boxes, not solve().
            owners = {p: color for group, color in zip(groups, key_colors) for p in group}
            for (t, l, b, r), center in zip(boxes, centers):
                painted[t + 1:b, l + 1:r] = owners[center]
            out[bt:bt + bh, bl:bl + bw] = painted
            for i, key in enumerate(key_grids, 1):
                t, l = positions[i]
                h, w = key.shape
                inp[t:t + h, l:l + w] = key
            # Unsupported damage specks also vary outside all containers.
            reserved = np.zeros(shape, dtype=bool)
            for i, (h, w) in enumerate([mask.shape] + [k.shape for k in key_grids]):
                t, l = positions[i]
                reserved[t:t + h, l:l + w] = True
            sites = np.argwhere(~reserved)
            amount = min(len(sites), int(rng.integers(0, 5)))
            for y, x in sites[rng.choice(len(sites), amount, replace=False)]:
                inp[y, x] = DAMAGE
            k = int(rng.integers(4))
            inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
            if rng.random() < 0.5:
                inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
            predicted, recovered, _, _, _ = _infer(inp)
            if not np.array_equal(predicted, out):
                raise AssertionError("constructed room boxes disagree with observed recovery")
            if _cycle_count(recovered['mask']) != count:
                raise AssertionError("flood regions disagree with planar cycle count")
            return dict(input=inp, output=out)
    raise RuntimeError("bounded scene sampler exhausted; no ambiguous fallback accepted")


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _validate_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'),
                                   (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _validate_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, number in (('train', nt), ('test', ne)):
        for index in range(number):
            witness, style = None, None
            if split == 'train' and index == 0:
                count = int(rng.integers(5, 7))
                sizes, witness, style = [2, 2, count - 4], 'global', 'partial'
            elif split == 'train' and index == 1:
                count, sizes, witness, style = 7, [3, 2, 2], 'triple', 'partial'
            elif split == 'train':
                count = int(rng.integers(4, 8))
                keys = int(rng.choice([k for k in (2, 3) if k + 2 <= count <= 3 * k]))
                sizes = _sizes(rng, count, keys)
            elif regime == 0:
                count = int(rng.integers(8, 11))
                keys = 3 if count <= 9 and rng.random() < 0.5 else 4
                sizes = _sizes(rng, count, keys)
            elif regime == 1:
                count = int(rng.integers(6, 10))
                sizes = {6: [4, 2], 7: [4, 2, 1], 8: [4, 2, 2], 9: [4, 3, 2]}[count]
                style = 'partial'
            else:
                count, keys = int(rng.integers(7, 10)), 4
                sizes = _sizes(rng, count, keys)
            for _ in range(8):
                pair = _make_pair(rng, count, sizes, witness, style)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate rejection exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(input_grid):
    out, board, cards, cover, candidates = _infer(input_grid)
    mask = board['mask']
    sizes = [(b - t + 1, r - l + 1) for t, l, b, r in board['rooms']]
    patterns = [c['pattern'] for c in cards]
    full = _full_lattice(mask)
    room_centers = board['centers']
    normalized_centers = _normalize(room_centers)
    return dict(rooms=len(room_centers), keys=len(cards), board_shape=mask.shape,
                canvas=out.shape, room_sizes=tuple(sorted(sizes)),
                key_sizes=tuple(sorted(map(len, patterns))),
                patterns=tuple(sorted(patterns)), centers=normalized_centers,
                board_color=board['color'], key_colors=tuple(sorted(c['color'] for c in cards)),
                topology=_topology_key(mask),
                wall_layout=tuple(tuple(map(int, row)) for row in mask),
                partial=not np.array_equal(mask, full),
                lattice_rooms=len(_regions(full)),
                raw_rooms=len(_regions(board['observed'])),
                damaged=int((mask & (board['local'] == DAMAGE)).sum()),
                unsupported_noise=int(((input_grid == DAMAGE).sum())
                                      - (mask & (board['local'] == DAMAGE)).sum()),
                unequal=len(set(sizes)) > 1,
                equal_size_different=_equal_size_different(patterns),
                local_ambiguity=max(map(len, candidates)) > 1,
                triple=any(len(p) == 3 and _noncollinear(p) for p in patterns),
                edge=any(x == 0 for x in board['box'][:2])
                     or board['box'][2] == out.shape[0] - 1
                     or board['box'][3] == out.shape[1] - 1)


def _is_global_witness(f):
    return (f['partial'] and f['unequal'] and f['raw_rooms'] != f['rooms']
            and f['lattice_rooms'] != f['rooms'] and f['equal_size_different']
            and f['local_ambiguity'] and f['damaged'] > 0)


def validate_generated_puzzle(puzzle):
    """Validate returned pixels, the global cover, and demonstration evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _validate_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (_check_grid(pair[k]) for k in ('input', 'output'))
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, board, cards, cover, _ = _infer(inp)
            assert np.array_equal(out, predicted), "room coloring differs from input-derived cover"
            assert not np.array_equal(inp, out)
            assert not np.any(np.isin(out, [DAMAGE, PIN]))
            assert _cycle_count(board['mask']) == len(board['rooms'])
            assert np.count_nonzero(inp == PIN) == len(board['rooms'])
            boxes = [board['box']] + [c['box'] for c in cards]
            for (at, al, ab, ar), (bt, bl, bb, br) in combinations(boxes, 2):
                assert ab + 1 < bt or bb + 1 < at or ar + 1 < bl or br + 1 < al
            damaged = board['mask'] & (board['local'] == DAMAGE)
            assert damaged.any(), "generated boards must demonstrate marked repair"
            degree = np.zeros_like(board['mask'], dtype=int)
            degree[1:] += board['mask'][:-1]
            degree[:-1] += board['mask'][1:]
            degree[:, 1:] += board['mask'][:, :-1]
            degree[:, :-1] += board['mask'][:, 1:]
            assert np.all(degree[damaged] == 2), "damage cannot remove junctions"
            assert np.all(_opposite(board['observed'])[damaged])
            assert not np.any(damaged[1:] & damaged[:-1])
            assert not np.any(damaged[:, 1:] & damaged[:, :-1])
            t, l, b, r = board['box']
            assert np.all(out[t:b + 1, l:r + 1][board['mask']] == board['color'])
            inside = np.zeros(inp.shape, dtype=bool)
            inside[t:b + 1, l:r + 1] = True
            assert np.all(out[~inside] == BACKGROUND)
            owners = {}
            for i, (subset, delta) in cover.items():
                # Exact preservation of the entire oriented marker constellation.
                assert _shift(cards[i]['pattern'], delta) == set(subset)
                assert not set(subset) & set(owners)
                owners.update({p: cards[i]['color'] for p in subset})
            assert set(owners) == set(board['centers'])
            for (rt, rl, rb, rr), center in zip(board['rooms'], board['centers']):
                assert np.all(out[rt:rb + 1, rl:rr + 1] == owners[center])
            if split == 'train':
                training.append(_features(inp))
            else:
                testing.append(_features(inp))
    assert any(_is_global_witness(f) for f in training), "missing integrated repair/global-cover witness"
    assert any(f['rooms'] == 7 and f['key_sizes'] == (2, 2, 3) and f['triple']
               for f in training), "missing noncollinear triple witness"
    assert all(f['rooms'] <= 7 and f['keys'] <= 3 and max(f['key_sizes']) <= 3
               for f in training)
    assert all(f['rooms'] > 7 or f['keys'] == 4 or max(f['key_sizes']) == 4
               for f in testing), "test does not extend a demonstrated structural axis"


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
               for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structural audit plus semantic counterfactual checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(p['input']) for puzzle in puzzles
               for split in ('train', 'test') for p in puzzle[split]]
    assert len({r['canvas'] for r in records}) >= 25
    assert len({r['topology'] for r in records}) >= 60
    assert len({r['centers'] for r in records}) >= 60
    assert len({p for r in records for p in r['patterns']}) >= 60
    assert {r['rooms'] for r in records} == set(range(4, 11))
    assert {r['keys'] for r in records} == {2, 3, 4}
    assert {s for r in records for s in r['key_sizes']} == {1, 2, 3, 4}
    assert {r['board_color'] for r in records} == set(STRUCTURAL)
    assert any(not r['partial'] for r in records)
    assert sum(r['local_ambiguity'] for r in records) >= 42
    assert any(r['edge'] for r in records)
    assert _equal_puzzles(puzzles[0], generate_puzzle(0))
    assert not _equal_puzzles(puzzles[0], puzzles[1])
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(120 + nt * 10 + ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2, False), (2.0, 1), (3, None), ('4', 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    # One returned scene is an oracle only for transformations of that scene.
    for puzzle in puzzles[:3]:
        for pair in puzzle['train'][:2] + puzzle['test']:
            for k in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
            assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
            permutation = np.arange(10, dtype=np.int8)
            permutation[list(STRUCTURAL)] = list(reversed(STRUCTURAL))
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Translation-only fixture: its oriented triangle covers one subset, and
    # reflecting only the key destroys the possibility of any cover.
    centers = [(0, 0), (0, 3), (2, 1), (4, 6), (7, 6)]
    keys = [((0, 0), (0, 3), (2, 1)), ((0, 0), (3, 0))]
    assert len(_covers(centers, keys)[0]) == 1
    reflected = [_normalize({(r, -c) for r, c in keys[0]}), keys[1]]
    assert len(_covers(centers, reflected)[0]) == 0
    # Identical geometry in different hues cannot use an arbitrary tie-break.
    assert len(_covers([(0, 0), (0, 2), (3, 0), (3, 2)],
                       [((0, 0), (0, 2)), ((0, 0), (0, 2))])[0]) == 2
    # Shuffle all card locations while keeping the board fixed and dots rigid.
    original = puzzles[0]['train'][0]
    _, board, cards = _decode(original['input'])
    rng = np.random.default_rng(992)
    shapes = [(c['box'][2] - c['box'][0] + 1, c['box'][3] - c['box'][1] + 1) for c in cards]
    for _ in range(20):
        relocated = np.zeros((30, 30), dtype=np.int8)
        t, l, b, r = board['box']
        relocated[t:b + 1, l:r + 1] = original['input'][t:b + 1, l:r + 1]
        blocked = np.zeros((30, 30), dtype=bool)
        blocked[max(0, t - 1):b + 2, max(0, l - 1):r + 2] = True
        success = True
        for c, (h, w) in zip(cards, shapes):
            ct, cl, cb, cr = c['box']
            for _ in range(200):
                y, x = int(rng.integers(31 - h)), int(rng.integers(31 - w))
                if not blocked[y:y + h, x:x + w].any():
                    relocated[y:y + h, x:x + w] = original['input'][ct:cb + 1, cl:cr + 1]
                    blocked[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = True
                    break
            else:
                success = False
                break
        if success:
            expected = np.zeros((30, 30), dtype=np.int8)
            expected[:original['output'].shape[0], :original['output'].shape[1]] = original['output']
            assert np.array_equal(solve(relocated), expected)
            break
    else:
        raise AssertionError("card relocation fixture could not be packed")
    # Removing either inherited concept changes a witnessed answer.
    witness = next(p for p in puzzles[0]['train'] if _is_global_witness(_features(p['input'])))
    _, board, cards, cover, _ = _infer(witness['input'])
    assert len(_regions(board['observed'])) != len(board['rooms'])
    assert len(_regions(_full_lattice(board['mask']))) != len(board['rooms'])
    a, b = next((a, b) for a, b in combinations(range(len(cards)), 2)
                if len(cards[a]['pattern']) == len(cards[b]['pattern'])
                and cards[a]['pattern'] != cards[b]['pattern'])
    assert cards[a]['color'] != cards[b]['color']
    wrong = witness['output'].copy()
    colors = (cards[a]['color'], cards[b]['color'])
    wrong[witness['output'] == colors[0]] = colors[1]
    wrong[witness['output'] == colors[1]] = colors[0]
    assert not np.array_equal(wrong, witness['output'])
    assert cover[a][0] not in [subset for subset, _ in _candidates(board['centers'], [cards[b]['pattern']])[0]]
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]]
           for s in ('train', 'test')}
    q = bad['test'][0]
    center = _decode(q['input'])[1]['centers'][0]
    q['output'][center] = BACKGROUND
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt room coloring accepted")
    return dict(seeds=42, pairs=len(records), canvas_shapes=len({r['canvas'] for r in records}),
                wall_topologies=len({r['topology'] for r in records}),
                wall_layouts=len({r['wall_layout'] for r in records}),
                center_constellations=len({r['centers'] for r in records}),
                key_patterns=len({p for r in records for p in r['patterns']}),
                room_counts=sorted({r['rooms'] for r in records}),
                key_counts=sorted({r['keys'] for r in records}),
                partial_pairs=sum(r['partial'] for r in records),
                local_ambiguity_pairs=sum(r['local_ambiguity'] for r in records))
