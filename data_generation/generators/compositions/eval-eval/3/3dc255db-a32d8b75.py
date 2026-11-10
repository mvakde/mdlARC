"""Finite graph-distance reservoirs compete to stamp exchanged binary tiles.

Only standard-library modules and NumPy are used. Labels are painted from sampled
objects; solve reconstructs all objects and their graph solely from input pixels.
"""
from collections import deque
from itertools import combinations
import json
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # U, R, D, L


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=True)


def _components(cells, diagonal=False):
    todo = set(cells)
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _world(p, k):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[k]


def _canonical(p, k):
    return _world(p, (-k) % 4)


def _envelope(frame, k):
    """Read thin expanding arms and a terminal stem, without a shape lookup."""
    pts = {_canonical(p, k) for p in frame}
    y0, y1 = min(y for y, _ in pts), max(y for y, _ in pts)
    rows = [sorted(x for y, x in pts if y == level) for level in range(y0, y1 + 1)]
    if y1 - y0 < 4 or len(rows[0]) != 1:
        return None
    x_tip = rows[0][0]
    old_l = old_r = x_tip
    split, cavity = False, set()
    for level, xs in enumerate(rows[1:], y0 + 1):
        if not xs or len(xs) > 2:
            return None
        l, r = xs[0], xs[-1]
        if not (old_l - 1 <= l <= old_l and old_r <= r <= old_r + 1):
            return None
        if len(xs) == 1 and (split or xs[0] != x_tip):
            return None
        if len(xs) == 2:
            split = True
            cavity.update(_world((level, x), k) for x in range(l + 1, r))
        old_l, old_r = l, r
    if not cavity or not old_l < x_tip < old_r or len(rows[-1]) != 2:
        return None
    return _world((y0, x_tip), k), cavity


def _read_contents(grid, cavity, bg, fc, k):
    colors = sorted({int(grid[p]) for p in cavity} - {bg})
    if len(colors) != 3 or fc in colors:
        return None
    choices = []
    for a, b in combinations(colors, 2):
        cells = {p for p in cavity if int(grid[p]) in (a, b)}
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        r1, c1 = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
        n = r1 - r0
        square = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
        if not (2 <= n <= 4 and c1 - c0 == n and cells == square):
            continue
        pc = next(v for v in colors if v not in (a, b))
        payload = {p for p in cavity if grid[p] == pc}
        # A blank cross-section separates the tipward tokens from the tile.
        tile_y = min(_canonical(p, k)[0] for p in square)
        if not payload or max(_canonical(p, k)[0] for p in payload) > tile_y - 2:
            continue
        choices.append(dict(tile=grid[r0:r1, c0:c1].copy(), tile_cells=square,
                            payload=payload, q=len(payload), pc=pc, n=n))
    return choices[0] if len(choices) == 1 else None


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if not np.all(border == bg):
        raise ValueError("the input requires a uniform background border")
    sources = []
    for fc in sorted(int(c) for c in np.unique(grid) if c != bg):
        for frame in _components(map(tuple, np.argwhere(grid == fc)), diagonal=True):
            for k in range(4):
                geometry = _envelope(frame, k)
                if geometry is None:
                    continue
                tip, cavity = geometry
                if not all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in cavity):
                    continue
                content = _read_contents(grid, cavity, bg, fc, k)
                if content:
                    dr, dc = _DIRS[k]
                    sources.append(dict(content, frame=frame, cavity=cavity, tip=tip,
                                        root=(tip[0] + dr, tip[1] + dc), k=k, fc=fc))
    if not 1 <= len(sources) <= 2:
        raise ValueError("expected one or two uniquely pointed reservoirs")
    sources.sort(key=lambda s: s["tip"])
    if len({s["pc"] for s in sources}) != 1 or len({s["fc"] for s in sources}) != 1:
        raise ValueError("reservoirs must share the structural frame and token roles")
    if len({s["n"] for s in sources}) != 1:
        raise ValueError("all tiles must have the same side length")
    used, boxes = set(), []
    for s in sources:
        rs, cs = zip(*s["frame"])
        box = {(r, c) for r in range(min(rs), max(rs) + 1)
               for c in range(min(cs), max(cs) + 1)}
        if any(box & other for other in boxes):
            raise ValueError("reservoir bounding boxes overlap")
        boxes.append(box)
        used.update(s["frame"] | s["payload"] | s["tile_cells"])
    pc = sources[0]["pc"]
    interiors = set().union(*(s["cavity"] for s in sources))
    network = set(map(tuple, np.argwhere(grid == pc))) - interiors
    if not network or len(_components(network)) != 1:
        raise ValueError("the stencil must be one four-connected network")
    r0, c0 = min(r for r, _ in network), min(c for _, c in network)
    r1, c1 = max(r for r, _ in network) + 1, max(c for _, c in network) + 1
    for s in sources:
        if s["root"] not in network:
            raise ValueError("every tip must touch a network entry cell")
        r, c = s["root"]
        if not ((s["k"] == 0 and r == r1 - 1) or (s["k"] == 1 and c == c0)
                or (s["k"] == 2 and r == r0) or (s["k"] == 3 and c == c1 - 1)):
            raise ValueError("entries must be on the corresponding outer stencil side")
    if len({s["root"] for s in sources}) != len(sources):
        raise ValueError("entry cells must be distinct")
    if set(map(tuple, np.argwhere(grid != bg))) != used | network:
        raise ValueError("unexpected pixels outside reservoirs and the network")
    n = sources[0]["n"]
    if max(n * (r1 - r0), n * (c1 - c0)) > 30:
        raise ValueError("expanded network exceeds ARC bounds")
    mask = np.zeros((r1 - r0, c1 - c0), bool)
    for r, c in network:
        mask[r - r0, c - c0] = True
    for s in sources:
        s["root_local"] = (s["root"][0] - r0, s["root"][1] - c0)
    return grid, bg, pc, mask, sources


def _distances(mask, root):
    dist = np.full(mask.shape, 10000, int)
    dist[root] = 0
    queue = deque([root])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]
                    and dist[rr, cc] == 10000):
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def _owners(mask, sources, variant="normal"):
    scores = []
    rr, cc = np.indices(mask.shape)
    for s in sources:
        root = s["root_local"]
        if variant == "ignore_ports":
            root = tuple(map(int, np.argwhere(mask)[0]))
        if variant == "manhattan":
            dist = abs(rr - root[0]) + abs(cc - root[1])
        elif variant == "straight":
            dr, dc = _DIRS[s["k"]]
            axial = (rr - root[0]) * dr + (cc - root[1]) * dc
            lateral = (rr - root[0]) * dc - (cc - root[1]) * dr
            dist = np.where((lateral == 0) & (axial >= 0), axial, 10000)
        else:
            dist = _distances(np.ones_like(mask) if variant == "fill_holes" else mask, root)
        q = (10000 if variant == "no_count" else len(_components(s["payload"], True))
             if variant == "components" else s["q"])
        scores.append(np.where(dist < q, dist, 10000))
    scores = np.stack(scores)
    best = scores.min(axis=0)
    unique = (scores == best).sum(axis=0) == 1
    if variant == "first_tie":
        unique[:] = True
    domain = np.ones_like(mask) if variant == "fill_holes" else mask
    owner = np.where(domain & (best < 10000) & unique, scores.argmin(axis=0), -1)
    tie = mask & (best < 10000) & ((scores == best).sum(axis=0) > 1)
    return owner, tie


def _prediction(bg, pc, mask, sources, variant="normal"):
    owner, tie = _owners(mask, sources, variant)
    n = sources[0]["n"]
    out = np.full((mask.shape[0] * n, mask.shape[1] * n), bg, int)
    tiles = []
    for s in sources:
        tile = np.rot90(s["tile"], -s["k"]) if variant == "rotate_tile" else s["tile"]
        a, b = map(int, np.unique(tile))
        tiles.append(np.full_like(tile, pc) if variant == "solid" else tile
                     if variant == "no_swap" else a + b - tile)
    # Pixel lookup makes the solver's block support explicit.
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            i = int(owner[r // n, c // n])
            if i < 0:
                continue
            out[r, c] = tiles[i][r % n, c % n]
    return out, owner, tie


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the visible border, pointed frames, source squares, and graph."""
    _, bg, pc, mask, sources = _decode(input_grid)
    return _prediction(bg, pc, mask, sources)[0]


def _profile(rng, n):
    radius = (n + 2) // 2
    a, b = radius + int(rng.integers(2)), radius + int(rng.integers(2))
    stem = int(rng.integers(2))
    tail = n + int(rng.integers(4, 7))
    frame = {(y, 0) for y in range(stem + 1)}
    cavity = set()
    y, l, r = stem, 0, 0
    while l < a or r < b:
        dl = l < a and (r == b or rng.random() < .85)
        dr = r < b and (l == a or rng.random() < .85)
        if not dl and not dr:
            dl = l < a
            dr = not dl
        l, r, y = l + int(dl), r + int(dr), y + 1
        frame.update(((y, -l), (y, r)))
        cavity.update((y, x) for x in range(-l + 1, r))
    for _ in range(tail):
        y += 1
        frame.update(((y, -a), (y, b)))
        cavity.update((y, x) for x in range(-a + 1, b))
    ty = y - n + 1
    tx = int(rng.integers(-a + 1, b - n + 1))
    square = {(yy, xx) for yy in range(ty, ty + n) for xx in range(tx, tx + n)}
    tokens = {p for p in cavity if p[0] < ty - 1}
    assert square <= cavity and tokens
    return dict(frame=frame, cavity=cavity, square=square, tokens=tokens)


def _tokens(rng, cells, q, scattered=False):
    options = sorted(cells)
    start = options[int(rng.integers(len(options)))]
    selected = {start}
    while len(selected) < q:
        rest = sorted(cells - selected)
        if scattered:
            distances = [min(max(abs(r - y), abs(c - x)) for y, x in selected) for r, c in rest]
            choices = [p for p, d in zip(rest, distances) if d == max(distances)]
        else:
            choices = [p for p in rest if any(abs(p[0] - y) + abs(p[1] - x) == 1 for y, x in selected)]
            if not choices:
                choices = rest
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _tile(rng, n, colors):
    for _ in range(80):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if bits.any() and not bits.all() and all(not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _source(rng, profile, root, k, q, tile, scattered=False):
    dr, dc = _DIRS[k]
    tip = (root[0] - dr, root[1] - dc)
    def move(p):
        r, c = _world(p, k)
        return r + tip[0], c + tip[1]
    square = {move(p) for p in profile["square"]}
    return dict(frame={move(p) for p in profile["frame"]},
                cavity={move(p) for p in profile["cavity"]}, tile_cells=square,
                payload={move(p) for p in _tokens(rng, profile["tokens"], q, scattered)},
                tip=tip, root=root, root_local=root, k=k, q=q, n=len(tile), tile=tile.copy())


def _transform(mask, roots, directions, turn, reflection):
    h, w = mask.shape
    if reflection:
        mask = mask[:, ::-1]
        roots = [(r, w - 1 - c) for r, c in roots]
        directions = [(-k) % 4 for k in directions]
    for _ in range(turn):
        mask = np.rot90(mask, -1)
        roots = [(c, h - 1 - r) for r, c in roots]
        directions = [(k + 1) % 4 for k in directions]
        h, w = w, h
    return mask.copy(), roots, directions


def _render(rng, bg, fc, pc, mask, sources):
    network = set(map(tuple, np.argwhere(mask)))
    boxes = []
    for s in sources:
        rs, cs = zip(*s["frame"])
        box = {(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)}
        if box & network or any(box & other for other in boxes):
            return None
        boxes.append(box)
    if len(sources) == 2 and any(max(abs(r-y), abs(c-x)) <= 1
        for r, c in sources[0]["frame"] for y, x in sources[1]["frame"]):
        return None
    all_cells = network | set().union(*boxes)
    lo = tuple(min(p[i] for p in all_cells) for i in (0, 1))
    hi = tuple(max(p[i] for p in all_cells) for i in (0, 1))
    extents = tuple(hi[i] - lo[i] + 1 for i in (0, 1))
    if max(extents) > 28:
        return None
    pads = [int(rng.integers(0, min(3, 29 - d))) for d in extents]
    shape = tuple(d + 2 + p for d, p in zip(extents, pads))
    shift = tuple(1 - lo[i] + int(rng.integers(pads[i] + 1)) for i in (0, 1))
    inp = np.full(shape, bg, int)
    def at(p):
        return p[0] + shift[0], p[1] + shift[1]
    for p in network:
        inp[at(p)] = pc
    for s in sources:
        for p in s["frame"]:
            inp[at(p)] = fc
        for p in s["payload"]:
            inp[at(p)] = pc
        r0 = min(r for r, _ in s["tile_cells"]) + shift[0]
        c0 = min(c for _, c in s["tile_cells"]) + shift[1]
        n = s["n"]
        inp[r0:r0+n, c0:c0+n] = s["tile"]
    # Direct block painting from sampled sources (not by calling solve).
    owner, _ = _owners(mask, sources)
    n = sources[0]["n"]
    out = np.full(tuple(d * n for d in mask.shape), bg, int)
    for r, c in np.argwhere(owner >= 0):
        tile = sources[int(owner[r, c])]["tile"]
        a, b = map(int, np.unique(tile))
        out[r*n:(r+1)*n, c*n:(c+1)*n] = a + b - tile
    pair = {"input": inp, "output": out}
    try:
        decoded = _decode(inp)
    except ValueError:
        return None
    if not np.array_equal(_prediction(*decoded[1:])[0], out):
        return None
    return pair


def _witnesses(rng):
    for _ in range(200):
        n, w, depth = int(rng.choice([2, 3])), int(rng.choice([5, 7])), int(rng.integers(2, 5))
        mid = w // 2
        mask = np.zeros((depth + 1, w), bool)
        mask[0] = True
        mask[:, mid] = True
        mask[-1, :mid + 1] = True
        if rng.random() < .7:
            mask[1:int(rng.integers(2, depth + 2)), -1] = True
        mask, roots, ks = _transform(mask, [(0, 0), (0, w-1)], [1, 3],
                                    int(rng.integers(4)), bool(rng.integers(2)))
        low, high = mid + 1, mid + int(rng.integers(4, 6))
        p1, p2 = _profile(rng, n), _profile(rng, n)
        if len(p1["tokens"]) <= high or len(p2["tokens"]) <= low:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, fc, pc = palette[:3]
        t1, t2 = _tile(rng, n, palette[3:5]), _tile(rng, n, palette[4:6])
        a = _source(rng, p1, roots[0], ks[0], low, t1)
        b = _source(rng, p1, roots[0], ks[0], high, t1)
        c = _source(rng, p2, roots[1], ks[1], low, t2, True)
        first, second = _render(rng, bg, fc, pc, mask, [a]), _render(rng, bg, fc, pc, mask, [b, c])
        if first is None or second is None:
            continue
        if len(_components(a["payload"], True)) == len(_components(c["payload"], True)):
            continue
        if _training_evidence([first, second], raise_on_failure=False):
            return [first, second]
    raise RuntimeError("bounded witness construction failed")


def _network(rng, h, w):
    style = int(rng.integers(5))
    mask = np.zeros((h, w), bool)
    if style == 0:  # Backtracking maze, then attach even-dimension fringes.
        stack, seen = [(0, 0)], {(0, 0)}
        mask[0, 0] = True
        while stack:
            r, c = stack[-1]
            moves = [(r+2*dr, c+2*dc, dr, dc) for dr, dc in _DIRS
                     if 0 <= r+2*dr < h and 0 <= c+2*dc < w and (r+2*dr, c+2*dc) not in seen]
            if not moves:
                stack.pop()
            else:
                rr, cc, dr, dc = moves[int(rng.integers(len(moves)))]
                mask[r+dr, c+dc] = mask[rr, cc] = True
                seen.add((rr, cc))
                stack.append((rr, cc))
        if h % 2 == 0:
            c = 2 * int(rng.integers((w + 1)//2))
            mask[-1, c] = True
        if w % 2 == 0:
            r = 2 * int(rng.integers((h + 1)//2))
            mask[r, -1] = True
    elif style == 1:  # Loop with interior spurs; holes remain structural.
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        mask[r, :c+1] = True
    elif style == 2:  # Comb with alternating-length cross branches.
        c = int(rng.integers(w))
        mask[:, c] = True
        for r in range(0, h, 2):
            left, right = int(rng.integers(c+1)), int(rng.integers(c, w))
            mask[r, left:right+1] = True
        mask[0, :c+1] = True
        mask[-1, c:] = True
    else:  # Connected clusters of independently varied density.
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, :] = mask[:, c] = True
        target = int(rng.uniform(.35, .9) * h * w)
        while mask.sum() < target:
            empty = [(rr, cc) for rr in range(h) for cc in range(w) if not mask[rr, cc]
                     and any(0 <= rr+dr < h and 0 <= cc+dc < w and mask[rr+dr, cc+dc] for dr, dc in _DIRS)]
            if not empty:
                break
            p = empty[int(rng.integers(len(empty)))]
            mask[p] = True
    return mask


def _random_pair(rng, test=False):
    for attempt in range(300):
        n = int(rng.choice([2, 2, 3, 3, 4]))
        if test:
            long = int(rng.integers(8, min(10, 30//n) + 1)) if n <= 3 else 0
            if not long:
                continue
            short = int(rng.integers(3, min(7, 30//n) + 1))
            h, w = (long, short) if rng.random() < .5 else (short, long)
        else:
            h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        mask = _network(rng, h, w)
        count = int(rng.choice([1, 1, 2, 2]))
        ks = list(map(int, rng.choice(4, count, replace=bool(rng.random() < .2))))
        palette = list(map(int, rng.permutation(10)))
        bg, fc, pc = palette[:3]
        tile_pool = palette[3:int(rng.integers(5, 8))]
        sources = []
        for k in ks:
            points = [(int(r), int(c)) for r, c in np.argwhere(mask)
                      if (k == 0 and r == h-1) or (k == 1 and c == 0)
                      or (k == 2 and r == 0) or (k == 3 and c == w-1)]
            root = points[int(rng.integers(len(points)))]
            profile = _profile(rng, n)
            maxq = min(14, len(profile["tokens"]) - 1)
            q = int(rng.integers(3 if test else 1, maxq + 1))
            tile = _tile(rng, n, rng.choice(tile_pool, 2, replace=False))
            sources.append(_source(rng, profile, root, k, q, tile, bool(rng.integers(2))))
        if len({s["root"] for s in sources}) != count:
            continue
        pair = _render(rng, bg, fc, pc, mask, sources)
        if pair is not None and test:
            d = _decode(pair["input"])
            if any(np.array_equal(_prediction(*d[1:], variant=v)[0], pair["output"])
                   for v in ("no_count", "straight")):
                continue
        if pair is not None:
            return pair
    raise RuntimeError("bounded random scene construction failed")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train = _witnesses(rng)
    puzzle = {"train": train, "test": []}
    seen = {_key(p["input"]) for p in train}
    for split, remaining in (("train", nt-2), ("test", nq)):
        for _ in range(remaining):
            for attempt in range(30):
                pair = _random_pair(rng, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


_ALTERNATIVES = ("no_count", "components", "straight", "manhattan", "ignore_ports",
                 "no_swap", "rotate_tile", "solid", "first_tie", "fill_holes")


def _source_signature(s):
    pts = [_canonical(p, s["k"]) for p in s["frame"]]
    y0, x0 = min(y for y, _ in pts), min(x for _, x in pts)
    return (tuple(sorted((y-y0, x-x0) for y, x in pts)), s["k"], _key(s["tile"]))


def _training_evidence(pairs, raise_on_failure=True):
    decoded = [_decode(p["input"]) for p in pairs]
    evidence = all(any(not np.array_equal(_prediction(*d[1:], variant=v)[0], p["output"])
                       for d, p in zip(decoded, pairs)) for v in _ALTERNATIVES)
    counts, arrangements = {}, {}
    tie_seen = False
    for d in decoded:
        _, bg, pc, mask, sources = d
        _, owner, tie = _prediction(bg, pc, mask, sources)
        tie_seen |= bool(tie.any())
        for s in sources:
            counts.setdefault(_source_signature(s), set()).add(s["q"])
            arrangements.setdefault(s["q"], set()).add(len(_components(s["payload"], True)))
    evidence &= (tie_seen and any(len(qs) >= 2 for qs in counts.values())
                 and any(len(cs) >= 2 for cs in arrangements.values()))
    if raise_on_failure:
        assert evidence, "training fails to expose counting, topology, competition, or substitution"
    return bool(evidence)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_span, test_span = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            _, bg, pc, mask, sources = _decode(inp)
            expected, owner, ties = _prediction(bg, pc, mask, sources)
            assert np.array_equal(expected, out), "incorrect ownership, exchange, or tile orientation"
            assert np.any(owner >= 0), "no claimed network cells"
            n = sources[0]["n"]
            assert out.shape == tuple(n*d for d in mask.shape)
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    block = out[r*n:(r+1)*n, c*n:(c+1)*n]
                    i = int(owner[r, c])
                    if i < 0:
                        assert np.all(block == bg), "holes, ties, and unreachable nodes must be blank"
                    else:
                        tile = sources[i]["tile"]
                        assert np.all(block != tile), "both binary roles must be exchanged"
                        assert set(np.unique(block)) == set(np.unique(tile)), "whole binary tiles must be copied"
            (train_span if split == "train" else test_span).append(max(mask.shape))
            if split == "test":
                assert not np.array_equal(_prediction(bg, pc, mask, sources, "no_count")[0], out), "test hides all count-dependent stopping"
                assert not np.array_equal(_prediction(bg, pc, mask, sources, "straight")[0], out), "test never exercises turns or branches"
    _training_evidence(puzzle["train"])
    assert min(test_span) > max(train_span), "missing structural span shift"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed coverage, counterfactual evidence, and API checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, graphs, profiles, directions, quantities, palettes, sizes, counts = (set() for _ in range(8))
    loops = ties = detours = partial = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
                _, bg, pc, mask, sources = _decode(before)
                shapes.add(before.shape)
                graphs.add(_key(mask))
                palettes.add(tuple(map(int, np.unique(before))))
                sizes.add(sources[0]["n"])
                counts.add(len(sources))
                owner, tie = _owners(mask, sources)
                ties += bool(tie.any())
                partial += bool(np.any(mask & (owner < 0) & ~tie))
                nodes = set(map(tuple, np.argwhere(mask)))
                edges = sum(sum((r+dr, c+dc) in nodes for dr, dc in ((1, 0), (0, 1)))
                            for r, c in np.argwhere(mask))
                loops += edges >= int(mask.sum())
                rr, cc = np.indices(mask.shape)
                for s in sources:
                    profiles.add(_source_signature(s)[0])
                    quantities.add(s["q"])
                    directions.add(s["k"])
                    dist = _distances(mask, s["root_local"])
                    mr, mc = s["root_local"]
                    detours += bool(np.any(mask & (dist > abs(rr-mr) + abs(cc-mc))))
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(graphs) >= 80 and len(profiles) >= 30
    assert directions == set(range(4)) and sizes == {2, 3, 4} and counts == {1, 2}
    assert len(quantities) >= 10 and len(palettes) >= 20
    assert loops and ties >= 42 and detours >= 42 and partial >= 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    # Explicit narrow-network cases exercise a budget-exhausted nearer source,
    # a central tie, symmetric templates, and the root's zero-based distance.
    profile = dict(frame={(0, 0), (1, -1), (1, 1)} |
                   {(y, x) for y in range(2, 9) for x in (-2, 2)},
                   cavity={(1, 0)} | {(y, x) for y in range(2, 9) for x in (-1, 0, 1)},
                   square={(y, x) for y in (7, 8) for x in (0, 1)})
    profile["tokens"] = {p for p in profile["cavity"] if p[0] <= 5}
    tile_a = np.array([[4, 5], [5, 4]], int)  # Symmetric, supported by solve.
    tile_b = np.array([[5, 7], [7, 7]], int)
    fixture_rng = np.random.default_rng(903)
    for width, qa, qb, owners in ((8, 7, 2, [0, 0, 0, 0, 0, 0, 1, 1]),
                                  (5, 3, 3, [0, 0, -1, 1, 1])):
        mask = np.ones((1, width), bool)
        sources = [_source(fixture_rng, profile, (0, 0), 1, qa, tile_a),
                   _source(fixture_rng, profile, (0, width-1), 3, qb, tile_b)]
        pair = _render(fixture_rng, 0, 2, 3, mask, sources)
        assert pair is not None
        expected = np.zeros((2, 2*width), int)
        for c, i in enumerate(owners):
            if i >= 0:
                tile = (tile_a, tile_b)[i]
                expected[:, 2*c:2*c+2] = int(np.unique(tile).sum()) - tile
        assert np.array_equal(solve(pair["input"]), expected), "explicit distance/tie regression"
        permutation = np.array([8, 0, 6, 2, 9, 1, 3, 5, 4, 7])
        for k in range(4):
            inp = permutation[np.rot90(pair["input"], k)]
            target = permutation[np.rot90(expected, k)]
            assert np.array_equal(solve(inp), target), "rotation or palette-role regression"
        _, _, _, _, decoded_sources = _decode(pair["input"])
        erased = pair["input"].copy()
        bottom = max(r for r, _ in decoded_sources[0]["tile_cells"])
        for r, c in decoded_sources[0]["tile_cells"]:
            if r == bottom:
                erased[r, c] = 0
        try:
            solve(erased)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete template was silently completed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1) + 1):
            puzzle = generate_puzzle(2026 + 10*nt + nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for request in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    damaged = {s: [{k: np.asarray(a).copy() for k, a in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (damaged["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "networks": len(graphs),
            "frame_profiles": len(profiles), "tile_sizes": sorted(sizes),
            "quantities": sorted(quantities), "loop_examples": loops,
            "tie_examples": ties, "detour_sources": detours, "partial_examples": partial,
            "explicit_semantic_fixtures": 2, "rotated_recolored_fixtures": 8}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
