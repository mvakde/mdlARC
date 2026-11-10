"""Seam-distance atlas: an anchored rigid jigsaw with graph-distance coloring.

Only standard library and NumPy are used. Solver state comes from the input.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations
import numpy as np

EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(x):
    a = np.asarray(x)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, neighbors=EIGHT):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    answer = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
        answer.append(np.asarray(component, dtype=int))
    return answer


def _connected(mask):
    return len(_components(mask, EDGE)) == 1


def _tight(mask, colors):
    coords = np.argwhere(mask)
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    result = np.full(tuple(hi - lo), -1, dtype=np.int8)
    local = coords - lo
    result[local[:, 0], local[:, 1]] = colors[coords[:, 0], coords[:, 1]]
    return result


def _legend(piece):
    """Find the unique 3..5-cell straight singleton swatch and leftover marker."""
    values, counts = np.unique(piece[piece >= 0], return_counts=True)
    _require(len(values[counts > 1]) == 1, "keyed piece needs one repeated carrier")
    singleton = values[counts == 1]
    _require(4 <= len(singleton) <= 6, "keyed piece needs a swatch and one marker")
    points = np.argwhere(np.isin(piece, singleton))
    candidates = []
    length = len(singleton) - 1
    for coords in combinations(map(tuple, points.tolist()), length):
        rows, cols = zip(*coords)
        horizontal = len(set(rows)) == 1 and max(cols) - min(cols) == length - 1
        vertical = len(set(cols)) == 1 and max(rows) - min(rows) == length - 1
        if horizontal or vertical:
            chosen = set(coords)
            marker_pos = next(p for p in map(tuple, points.tolist()) if p not in chosen)
            candidates.append((tuple(sorted(coords)), marker_pos))
    _require(len(candidates) == 1, "swatch must be unambiguous")
    coords, marker_pos = candidates[0]
    return coords, marker_pos, int(piece[marker_pos])


def _parse(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _require(len(np.unique(border)) == 1, "input has a complete background border")
    bg = int(border[0])
    components = _components(grid != bg)
    _require(4 <= len(components) <= 7, "four to seven fragments required")
    pieces = []
    anchor = None
    for i, coords in enumerate(components):
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        p = _tight(mask, grid)
        _require(len(coords) >= 3, "fragment area must be at least three")
        if len(np.unique(p[p >= 0])) > 1:
            _require(anchor is None, "one keyed fragment required")
            _, _, marker = _legend(p)
            anchor = i
        pieces.append(p)
    _require(anchor is not None, "keyed fragment missing")
    # Reordering is reconstructed, not retained from generation.
    pieces = [pieces[anchor]] + [p for i, p in enumerate(pieces) if i != anchor]
    area = sum(int(np.sum(p >= 0)) for p in pieces)
    side = int(round(area ** 0.5))
    _require(side * side == area and 6 <= side <= 13, "fragment area must form a supported square")
    key_coords, _, marker = _legend(pieces[0])
    key_colors = {int(pieces[0][p]) for p in key_coords}
    carriers = {int(c) for p in pieces for c in np.unique(p[p >= 0]) if c not in key_colors and c != marker}
    _require(not (carriers & key_colors) and bg not in key_colors and marker != bg, "distinct semantic palette roles required")
    _require(all(len(set(map(int, p[p >= 0]))) == 1 and int(p[p >= 0][0]) not in key_colors | {marker}
                 for p in pieces[1:]), "plain fragments need temporary carriers, not key colors")
    # Eight-neighbor segmentation also enforces a full blank-cell separation.
    return dict(grid=grid, background=bg, pieces=pieces, marker=marker, side=side)


def _orientations(piece):
    seen = set()
    for k in range(4):
        p = np.rot90(piece, k).copy()
        sig = _signature(p)
        if sig not in seen:
            seen.add(sig)
            yield p, k


def _pack(pieces, side, marker, limit=2, allow_rotations=True):
    """Exhaustive exact cover; reject distinct partitions, not duplicate symmetries.

    Plain congruent fragments of the same carrier may exchange identities.
    Their masks are canonicalized without hiding different internal seams.
    """
    tables, anchors = [], []
    for i, piece in enumerate(pieces):
        candidates, seen = [], set()
        for p, k in _orientations(piece):
            if not allow_rotations and k != 0:
                continue
            h, w = p.shape
            if h > side or w > side:
                continue
            coords = np.argwhere(p >= 0)
            cells = tuple((int(r) * side + int(c), int(p[r, c])) for r, c in coords)
            base = sum(1 << pos for pos, _ in cells)
            if i == 0:
                if p[0, 0] != marker:
                    continue
                key_coords, _, _ = _legend(p)
                if len({r for r, c in key_coords}) != 1:
                    continue
                key = tuple(int(p[q]) for q in sorted(key_coords))
                positions = [(0, 0)]
            else:
                key = None
                positions = [(r, c) for r in range(side - h + 1) for c in range(side - w + 1)]
            for r, c in positions:
                offset = r * side + c
                mask = base << offset
                sig = (mask, tuple((pos + offset, val) for pos, val in cells))
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append((mask, offset, cells, k, key))
        if not candidates:
            return []
        if i == 0:
            anchors = candidates
        tables.append(candidates)
    # Index by first uncovered cell to avoid scanning all translations per node.
    indexed = []
    for table in tables:
        by_cell = [[] for _ in range(side * side)]
        for candidate in table:
            bits = candidate[0]
            while bits:
                bit = bits & -bits
                by_cell[bit.bit_length() - 1].append(candidate)
                bits ^= bit
        indexed.append(by_cell)
    full = (1 << (side * side)) - 1
    placed = [None] * len(pieces)
    answers, seen_answers = [], set()

    def visit(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            # Include anchor colors, and all separate fragment footprints.
            sig = (tuple((pos, val) for pos, val in placed[0][2]),
                   tuple(sorted((q[0], tuple(v for _, v in q[2])) for q in placed[1:])))
            if sig in seen_answers:
                return
            seen_answers.add(sig)
            owner = np.empty((side, side), dtype=np.int8)
            assembled = np.empty_like(owner)
            for i, (_, offset, cells, _, _) in enumerate(placed):
                for pos, value in cells:
                    owner.flat[pos + offset] = i
                    assembled.flat[pos + offset] = value
            answers.append(dict(owner=owner, assembled=assembled,
                                key=placed[0][4], rotations=tuple(q[3] for q in placed)))
            return
        empty = full ^ occupied
        first = (empty & -empty).bit_length() - 1
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for q in indexed[i][first]:
                if not q[0] & occupied:
                    placed[i] = q
                    visit(occupied | q[0], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    for q in anchors:
        placed[0] = q
        visit(q[0], tuple(range(1, len(pieces))))
        if len(answers) >= limit:
            break
    return answers


def _distances(owner, diagonal=False):
    n = int(owner.max()) + 1
    graph = [set() for _ in range(n)]
    neighbors = EIGHT if diagonal else EDGE
    h, w = owner.shape
    for r in range(h):
        for c in range(w):
            a = int(owner[r, c])
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    b = int(owner[rr, cc])
                    if a != b:
                        graph[a].add(b)
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    while queue:
        a = queue.popleft()
        for b in sorted(graph[a]):
            if dist[b] < 0:
                dist[b] = dist[a] + 1
                queue.append(b)
    _require(min(dist) >= 0, "contact graph must be connected")
    return np.asarray(dist, dtype=np.int8), graph


def _analyze(input_grid):
    facts = _parse(input_grid)
    answers = _pack(facts['pieces'], facts['side'], facts['marker'])
    _require(len(answers) == 1, "require a unique anchored fragment partition")
    facts.update(answers[0])
    dist, graph = _distances(facts['owner'])
    _require(int(dist.max()) < len(facts['key']), "swatch must cover every seam distance")
    facts.update(distances=dist, graph=graph)
    return facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _analyze(input_grid)
    key = np.asarray(facts['key'], dtype=np.int8)
    return key[facts['distances'][facts['owner']]].copy()


def _mutate(rng, owner, key_length, amount):
    side = len(owner)
    protected = np.zeros_like(owner, dtype=bool)
    protected[:2, :key_length + 1] = True
    for _ in range(amount):
        r, c = map(int, rng.integers(side, size=2))
        if protected[r, c]:
            continue
        dr, dc = EDGE[int(rng.integers(4))]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < side and 0 <= cc < side):
            continue
        a, b = int(owner[r, c]), int(owner[rr, cc])
        if a == b or np.sum(owner == a) <= 3:
            continue
        owner[r, c] = b
        if not _connected(owner == a):
            owner[r, c] = a
    return owner


def _partition(rng, side, length, count, kind):
    """Procedural partitions, with geometry checked rather than presumed unique."""
    if kind == 'growth':
        owner = np.full((side, side), -1, dtype=np.int8)
        owner[:2, :length+1] = 0
        available = np.flatnonzero(owner < 0)
        for label, pos in enumerate(rng.choice(available, count - 1, replace=False), 1):
            owner.flat[int(pos)] = label
        weights = np.exp(rng.uniform(-1.3, 1.3, count))
        while np.any(owner < 0):
            frontiers = [set() for _ in range(count)]
            for r, c in np.argwhere(owner >= 0):
                label = int(owner[r, c])
                for dr, dc in EDGE:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < side and 0 <= cc < side and owner[rr, cc] < 0:
                        frontiers[label].add((rr, cc))
            active = [i for i in range(count) if frontiers[i]]
            chosen = min(active, key=lambda i: np.sum(owner == i) / weights[i] + float(rng.random()))
            cells = sorted(frontiers[chosen])
            owner[cells[int(rng.integers(len(cells)))]] = chosen
    elif kind == 'mesh':
        y = int(rng.integers(2, side - 2))
        x = int(rng.integers(length + 1, side - 1))
        z = int(rng.integers(x + 1, side))
        owner = np.empty((side, side), dtype=np.int8)
        owner[:y, :x], owner[:y, x:] = 0, 1
        owner[y:, :x], owner[y:, x:z], owner[y:, z:] = 2, 3, 4
    else:
        if kind == 'elbow':
            # Successive L-shaped shells: seam depth winds around a corner.
            # Unlike a corner key, the root may occupy several square corners.
            owner = np.full((side, side), length - 1, dtype=np.int8)
            r0 = c0 = 0
            for label in range(length - 1):
                minimum = 2 if label == 0 else 1
                room = side - max(r0, c0) - (length - label - 1)
                if room < minimum:
                    return None
                h = int(rng.integers(minimum, min(room, minimum + 1) + 1))
                w = int(rng.integers(minimum, min(room, minimum + 1) + 1))
                owner[r0:r0+h, c0:] = label
                owner[r0:, c0:c0+w] = label
                r0, c0 = r0 + h, c0 + w
        else:
            vertical = bool(rng.integers(2))
            minimum = length + 1 if vertical else 2
            if side < minimum + length - 1:
                vertical, minimum = False, 2
            choices = np.arange(minimum, side)
            cuts = sorted(map(int, rng.choice(choices, length - 1, replace=False)))
            bounds = [0] + cuts + [side]
            owner = np.empty((side, side), dtype=np.int8)
            for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
                if vertical:
                    owner[:, a:b] = i
                else:
                    owner[a:b] = i
        for label in range(length, count):
            eligible = []
            for old in range(1, label):
                coords = np.argwhere(owner == old)
                lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
                h, w = hi - lo
                if kind == 'inserts' and min(h, w) >= 4:
                    eligible.append((old, lo, hi, True))
                if (h >= 2 and w >= 3) or (h >= 3 and w >= 2) or max(h, w) >= 6:
                    eligible.append((old, lo, hi, False))
            if not eligible:
                return None
            old, lo, hi, inset = eligible[int(rng.integers(len(eligible)))]
            r0, c0 = map(int, lo)
            r1, c1 = map(int, hi)
            if inset:
                r = int(rng.integers(r0 + 1, r1 - 2))
                c = int(rng.integers(c0 + 1, c1 - 2))
                if not np.all(owner[r:r+2, c:c+2] == old):
                    return None
                owner[r:r+2, c:c+2] = label
                if rng.random() < .5:
                    owner[r+1, c+1] = old
            elif (c1 - c0 >= 6) or (c1 - c0 >= 3 and r1 - r0 >= 2 and rng.random() < .7):
                cut = int(rng.integers(c0 + 1, c1))
                view = owner[r0:r1, cut:c1]
                view[view == old] = label
            else:
                cut = int(rng.integers(r0 + 1, r1))
                view = owner[cut:r1, c0:c1]
                view[view == old] = label
    return _mutate(rng, owner, length, int(rng.integers(side * 3, side * 10)))


def _free_scatter(rng, oriented):
    """Place real footprints with a blank halo; bounding boxes may interleave."""
    width = 28
    full = (1 << (width * width)) - 1
    first_col = sum(1 << (r * width) for r in range(width))
    last_col = first_col << (width - 1)
    options = []
    for p in oriented:
        h, w = p.shape
        base = sum(1 << (int(r) * width + int(c)) for r, c in np.argwhere(p >= 0))
        placements = []
        for r in range(width - h + 1):
            for c in range(width - w + 1):
                mask = base << (r * width + c)
                horiz = mask | ((mask & ~first_col) >> 1) | ((mask & ~last_col) << 1)
                halo = (horiz | (horiz << width) | (horiz >> width)) & full
                placements.append((mask, halo, r, c))
        options.append(placements)
    for attempt in range(12):
        order = (sorted(range(len(oriented)), key=lambda i: -oriented[i].size)
                 if attempt == 0 else list(map(int, rng.permutation(len(oriented)))))
        offsets, blocked, maxr, maxc = [None] * len(oriented), 0, 0, 0
        aspect = float(rng.uniform(.65, 1.6))
        for i in order:
            h, w = oriented[i].shape
            choices = [p for p in options[i] if not p[0] & blocked]
            if not choices:
                break
            score = lambda p: max(maxr, p[2]+h) * max(maxc, p[3]+w) + abs(max(maxr, p[2]+h) - aspect * max(maxc, p[3]+w)) * 2 + float(rng.random()) * 20
            q = min(choices, key=score)
            offsets[i] = (q[2], q[3])
            blocked |= q[1]
            maxr, maxc = max(maxr, q[2]+h), max(maxc, q[3]+w)
        else:
            return offsets, (maxr, maxc)
    return None


def _scatter(rng, pieces, background, turns):
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, turns)]
    n = len(pieces)
    # Random shelf grouping uses actual tight dimensions, not output-sized slots.
    best = []
    for _ in range(80):
        order = list(map(int, rng.permutation(n)))
        transposed = bool(rng.integers(2))
        dims = [p.shape[::-1] if transposed else p.shape for p in oriented]
        groups, group = [], []
        for j, i in enumerate(order):
            group.append(i)
            if j == n - 1 or rng.random() < .35:
                groups.append(group)
                group = []
        gap = int(rng.choice([1, 1, 2, 3]))
        heights = [max(dims[i][0] for i in g) for g in groups]
        widths = [sum(dims[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
        H, W = sum(heights) + gap * (len(groups) - 1), max(widths)
        if H > 28 or W > 28:
            continue
        offsets, row = [None] * n, 0
        for g, h, w in zip(groups, heights, widths):
            col = int(rng.integers(W - w + 1))
            for i in g:
                r = row + int(rng.integers(h - dims[i][0] + 1))
                offsets[i] = (col, r) if transposed else (r, col)
                col += dims[i][1] + gap
            row += h + gap
        shape = (W, H) if transposed else (H, W)
        best.append((shape[0] * shape[1] + float(rng.random()) * 100, offsets, shape))
    free = _free_scatter(rng, oriented) if not best or rng.random() < .35 else None
    if free is not None:
        offsets, shape = free
    elif best:
        _, offsets, shape = min(best, key=lambda x: x[0])
    else:
        return None
    pads = []
    for d in shape:
        room = 30 - d
        before = int(rng.integers(1, min(4, room)))
        after = int(rng.integers(1, min(4, room - before + 1)))
        pads.append((before, after))
    canvas = np.full(tuple(shape[a] + sum(pads[a]) for a in (0, 1)), background, dtype=np.int8)
    for p, (r, c) in zip(oriented, offsets):
        r, c = r + pads[0][0], c + pads[1][0]
        view = canvas[r:r+p.shape[0], c:c+p.shape[1]]
        view[p >= 0] = p[p >= 0]
    return canvas


def _make_pair(rng, side, length, count, kind, anchor_turn):
    for attempt in range(350):
        # Growth with prescribed long graph distances is rarer. After a finite
        # search use winding shells, still checked by the same exact packer.
        actual_kind = 'elbow' if kind == 'growth' and attempt >= 160 else kind
        owner = _partition(rng, side, length, count, actual_kind)
        if owner is None:
            continue
        if any(np.sum(owner == i) < 3 or not _connected(owner == i) for i in range(count)):
            continue
        distances, graph = _distances(owner)
        if int(distances.max()) != length - 1:
            continue
        if kind == 'mesh':
            diagonal, _ = _distances(owner, True)
            multiple_routes = any(sum(distances[j] == distances[i] - 1 for j in graph[i]) >= 2
                                  for i in range(1, count))
            if (np.array_equal(diagonal, distances) or len(set(distances)) == count or
                    sum(map(len, graph)) // 2 < count or not multiple_routes):
                continue
        palette = list(map(int, rng.permutation(10)))
        background, marker = palette[:2]
        key = tuple(palette[2:2+length])
        carriers_available = palette[2+length:]
        pool = list(map(int, rng.choice(carriers_available, int(rng.integers(1, min(3, len(carriers_available)) + 1)), replace=False)))
        carriers = [int(rng.choice(pool)) for _ in range(count)]
        colored = np.asarray(carriers, dtype=np.int8)[owner]
        colored[0, 0] = marker
        key_locations = [(r, c) for r in range(1, side) for c in range(1, side-length+1)
                         if np.all(owner[r, c:c+length] == 0)]
        if not key_locations:
            continue
        r, c = key_locations[int(rng.integers(len(key_locations)))]
        colored[r, c:c+length] = key
        pieces = [_tight(owner == i, colored) for i in range(count)]
        answers = _pack(pieces, side, marker)
        if len(answers) != 1 or not np.array_equal(answers[0]['owner'], owner):
            continue
        turns = [anchor_turn] + list(map(int, rng.integers(4, size=count - 1)))
        # Guarantee an asymmetric rigid rotation demonstration in the first pair.
        if kind == 'chain':
            eligible = [i for i in range(1, count) if len(list(_orientations(pieces[i]))) == 4]
            if not eligible:
                continue
            turns[eligible[0]] = int(rng.integers(1, 4))
        grid = _scatter(rng, pieces, background, turns)
        if grid is None:
            continue
        output = np.asarray(key, dtype=np.int8)[distances[owner]]
        return {'input': grid, 'output': output}
    raise RuntimeError(f"bounded construction exhausted for {side}/{length}/{count}/{kind}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    rng = np.random.default_rng(seed)
    scale_shift = bool(rng.integers(2))
    # Raw input scan order is reversed once and correct once. Even two
    # demonstrations therefore establish that the legend moves with the frame.
    first_turn = int(rng.choice([1, 2]))
    second_turn = int(rng.choice([0, 3]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(num_train + num_test):
        split = 'train' if i < num_train else 'test'
        if i == 0:
            length, count, kind, turn = 4, 4, 'chain', first_turn
        elif i == 1:
            length, count, kind, turn = 3, 5, 'mesh', second_turn
        elif split == 'train':
            length = int(rng.choice([3, 4]))
            count = int(rng.integers(max(4, length), 6))
            kind = str(rng.choice(['branch', 'inserts', 'elbow', 'growth']))
            turn = int(rng.integers(4))
        else:
            length = int(rng.choice([4, 5]))
            count = length + 2
            kind = str(rng.choice(['branch', 'inserts', 'elbow', 'growth'] if length == 4
                                  else ['branch', 'inserts', 'elbow']))
            turn = int(rng.integers(4))
        low, high = ((8, 11) if scale_shift and split == 'train' else
                     (11, 14) if scale_shift else (max(8, length + 5), 14))
        for _ in range(12):
            side = int(rng.integers(low, high))
            pair = _make_pair(rng, side, length, count, kind, turn)
            sig = _signature(pair['input'])
            if sig not in seen:
                seen.add(sig)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not construct a distinct input")
    return puzzle


def _features(pair):
    facts = _analyze(pair['input'])
    owner, d, graph = facts['owner'], facts['distances'], facts['graph']
    expected = np.asarray(facts['key'], dtype=np.int8)[d[owner]]
    _require(np.array_equal(expected, np.asarray(pair['output'])), "output differs from input-derived seam distances")
    diagonal, _ = _distances(owner, True)
    areas = tuple(int(np.sum(owner == i)) for i in range(len(d)))
    key_coords, _, _ = _legend(facts['pieces'][0])
    raw_key = tuple(int(facts['pieces'][0][p]) for p in key_coords)
    return dict(facts, count=len(d), areas=areas, edges=sum(map(len, graph)) // 2,
                raw_swatch_reversed=raw_key != facts['key'],
                branching=any(len(g) >= 3 for g in graph),
                multiple_routes=any(sum(d[j] == d[i] - 1 for j in graph[i]) >= 2 for i in range(1, len(d))),
                repeated_depth=len(set(map(int, d))) < len(d),
                diagonal_shortcut=not np.array_equal(d, diagonal),
                min_fill=min(float(np.sum(p >= 0)) / p.size for p in facts['pieces']),
                interior=sum(not (np.any(owner[0] == i) or np.any(owner[-1] == i) or
                                  np.any(owner[:, 0] == i) or np.any(owner[:, -1] == i)) for i in range(len(d))))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "exact input/output keys required")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            f = _features(pair)
            _require(out.shape == (f['side'], f['side']), "area determines output dimensions")
            _require(set(map(int, np.unique(out))) == set(f['key']), "every swatch distance must occur")
            _require(int(out[0, 0]) == f['key'][0], "first swatch color belongs to marker's fragment")
            _require(f['marker'] not in out and f['background'] not in out, "remove marker and background")
            # Omitting the legend (color preservation) changes every generated answer.
            _require(not np.array_equal(f['assembled'], out), "legend must change the answer")
            # Ignoring seams and using cell distance/constant color cannot match.
            _require(len(np.unique(out)) >= 3 and len(f['pieces']) >= 4, "nontrivial graph layers required")
            records[split].append(f)
    train = records['train']
    _require({3, 4} <= {len(f['key']) for f in train}, "show variable swatch length")
    _require(any(f['repeated_depth'] and f['branching'] for f in train), "demonstrate graph branches and equal-depth fragments")
    _require(any(f['diagonal_shortcut'] for f in train), "demonstrate edge rather than diagonal contact")
    _require(any(f['multiple_routes'] and f['edges'] >= f['count'] for f in train), "demonstrate shortest-route ties in a cycle")
    _require(len({f['key'] for f in train}) >= 2, "demonstrate input-controlled palette")
    _require({f['raw_swatch_reversed'] for f in train} == {False, True}, "demonstrate both raw swatch reading orders")
    _require(any(any(k != 0 for k in f['rotations']) for f in train), "demonstrate rigid rotations")
    _require(max(f['count'] for f in train) <= 5 and min(f['count'] for f in records['test']) >= 6,
             "tests extend fragment count after demonstrations")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
               for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed geometry/topology audit, API checks, and semantic mutations."""
    records, seen, puzzles = [], set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                sig = _signature(np.asarray(pair['input']))
                _require(sig not in seen, "cross-seed duplicate input")
                seen.add(sig)
                records.append(f)
    _require(len({f['side'] for f in records}) >= 5, "vary assembly scale")
    _require({4, 5, 6, 7} <= {f['count'] for f in records}, "vary fragment counts")
    _require({3, 4, 5} == {len(f['key']) for f in records}, "vary swatch length and graph depth")
    _require(len({f['grid'].shape for f in records}) >= 35, "vary scattering geometry")
    _require(len({f['areas'] for f in records}) >= 100, "vary shapes and area budgets")
    _require(sum(f['edges'] >= f['count'] for f in records) >= 42, "cycles must occur")
    _require(sum(f['branching'] and f['repeated_depth'] for f in records) >= 42, "branched distance layers must occur")
    _require(sum(f['multiple_routes'] for f in records) >= 42, "shortest-route ties must occur")
    _require(sum(f['interior'] > 0 for f in records) >= 10, "internal fragments must occur")
    _require(sum(sum(f['owner'][p] == 0 for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))) >= 3
                 for f in records) >= 5, "wrapping roots must defeat four-corner coloring")
    _require(min(f['min_fill'] for f in records) < .65, "include wrapping/irregular shapes")
    _require({f['background'] for f in records} == set(range(10)), "vary background roles")
    _require({k for f in records for k in f['rotations']} == set(range(4)), "exercise all quarter-turns")
    _require(_equal(puzzles[0], generate_puzzle(0)), "determinism failed")
    _require(not _equal(puzzles[0], puzzles[1]), "seeds must change puzzles")
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(100 + a + b, a, b)
        validate_generated_puzzle(p)
    # Small legal family members, outside the default sampling size emphasis.
    for side in (6, 7):
        for kind, length, count in (('chain', 4, 4), ('mesh', 3, 5), ('elbow', 3, 5)):
            pair = _make_pair(np.random.default_rng(side * 100 + count), side, length, count, kind, 1)
            _features(pair)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    # Legend-only intervention: preserve pieces and contacts, swap swatch colors.
    original = puzzles[0]['train'][0]
    f = _analyze(original['input'])
    _require(not _pack(f['pieces'], f['side'], f['marker'], allow_rotations=False),
             "freezing rigid orientations must change the first demonstration")
    changed = original['input'].copy()
    u, v = f['key'][:2]
    changed[original['input'] == u] = v
    changed[original['input'] == v] = u
    prediction = solve(changed)
    expected = original['output'].copy()
    expected[original['output'] == u] = v
    expected[original['output'] == v] = u
    _require(np.array_equal(prediction, expected), "legend intervention must change whole distance layers")
    return None
