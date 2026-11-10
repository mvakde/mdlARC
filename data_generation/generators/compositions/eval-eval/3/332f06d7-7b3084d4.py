"""Tunnel-qualified rigid jigsaw assembly. No files or parent modules are used."""
from collections import deque
from hashlib import sha256
from math import isqrt

import numpy as np


BACKGROUND, ENTRY, EXIT, WALL, FLOOR, MARKER = 0, 1, 2, 5, 8, 9
CARRIERS = (3, 4, 6, 7)
EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEAR = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask, neighbors=NEAR):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(cells)
    return result


def _tight(a, mask):
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
    return np.where(mask[r0:r1, c0:c1], a[r0:r1, c0:c1], -1).astype(np.int8)


def _orientations(piece):
    seen = set()
    for k in range(4):
        q = np.rot90(piece, k).copy()
        if _key(q) not in seen:
            seen.add(_key(q))
            yield q


def _placements(mask, footprint):
    """All translations of the *occupied cells*, not its bounding rectangle."""
    h, w = footprint.shape
    H, W = mask.shape
    if h > H or w > W:
        return np.zeros((0, 0), dtype=bool)
    valid = np.ones((H-h+1, W-w+1), dtype=bool)
    for r, c in np.argwhere(footprint):
        valid &= mask[r:r+H-h+1, c:c+W-w+1]
    return valid


def _transit(panel, piece):
    """One fixed orientation can translate from wholly blue to wholly red."""
    footprint = piece >= 0
    allowed = _placements(np.isin(panel, (ENTRY, EXIT, FLOOR)), footprint)
    starts = _placements(panel == ENTRY, footprint)
    targets = _placements(panel == EXIT, footprint)
    if not starts.size or not starts.any() or not targets.any():
        return False
    seen = starts.copy()
    queue = deque(map(tuple, np.argwhere(starts)))
    H, W = allowed.shape
    while queue:
        r, c = queue.popleft()
        if targets[r, c]:
            return True
        for dr, dc in EDGE:
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W and allowed[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return False


def _read(inp):
    wall_cells = np.argwhere(inp == WALL)
    _require(len(wall_cells) > 0, "missing tunnel walls")
    r0, c0 = wall_cells.min(axis=0)
    r1, c1 = wall_cells.max(axis=0)+1
    panel = inp[r0:r1, c0:c1].copy()
    _require(set(map(int, np.unique(panel))) == {ENTRY, EXIT, WALL, FLOOR}, "tunnel palette")
    _require(np.all(panel[[0, -1], :] == WALL) and np.all(panel[:, [0, -1]] == WALL), "closed tunnel frame")
    _require(len(_components(panel != WALL, EDGE)) == 1, "disconnected passage")
    for role in (ENTRY, EXIT):
        cells = panel == role
        tight = _tight(panel, cells)
        _require(tight.shape[0] == tight.shape[1] and np.all(tight == role), "bays must be solid squares")
    outside = inp.copy()
    outside[r0:r1, c0:c1] = BACKGROUND
    _require(set(map(int, np.unique(outside))) <= {BACKGROUND, MARKER, *CARRIERS}, "unexpected object color")
    _require(np.count_nonzero(outside == MARKER) == 1, "unique attached marker required")
    objects = _components(outside != BACKGROUND)
    _require(3 <= len(objects) <= 4, "three or four separated pieces required")
    pieces, carriers = [], []
    for cells in objects:
        mask = np.zeros(inp.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        q = _tight(inp, mask)
        values = set(map(int, q[q >= 0])) - {MARKER}
        _require(len(values) == 1 and next(iter(values)) in CARRIERS, "one carrier per object")
        _require(len(cells) >= 2, "pieces need at least two cells")
        pieces.append(q)
        carriers.append(next(iter(values)))
    _require(len(set(carriers)) == len(carriers), "distinct carriers required")
    area = sum(int(np.count_nonzero(p >= 0)) for p in pieces)
    n = isqrt(area)
    _require(n*n == area, "piece area must be square")
    _require(all(max(p.shape) <= n for p in pieces), "piece exceeds output square")
    return panel, pieces, n


def _admitted(panel, pieces):
    return [[q for q in _orientations(p) if _transit(panel, q)] for p in pieces]


def _pack(pieces, n, orientations=None, limit=2):
    """Exhaustive rotation/translation exact cover; stop at distinct colored answers."""
    if orientations is None:
        orientations = [list(_orientations(p)) for p in pieces]
    candidates = []
    for choices in orientations:
        options = {}
        for q in choices:
            h, w = q.shape
            if h > n or w > n:
                continue
            marked = np.any(q == MARKER)
            if marked and q[0, 0] != MARKER:
                continue
            offsets = [(0, 0)] if marked else [(r, c) for r in range(n-h+1) for c in range(n-w+1)]
            local = [(int(r)*n+int(c), int(q[r, c])) for r, c in np.argwhere(q >= 0)]
            base = sum(1 << pos for pos, _ in local)
            for r, c in offsets:
                offset = r*n+c
                colored = tuple((pos+offset, color) for pos, color in local)
                options[colored] = (base << offset, colored)
        if not options:
            return []
        candidates.append(list(options.values()))
    full = (1 << (n*n))-1
    solutions, seen, chosen = [], set(), []

    def search(occupied, remaining):
        if len(solutions) >= limit:
            return
        if not remaining:
            if occupied == full:
                out = np.empty((n, n), dtype=np.int8)
                for cells in chosen:
                    for pos, color in cells:
                        out.flat[pos] = color
                key = out.tobytes()
                if key not in seen:
                    seen.add(key)
                    solutions.append(out)
            return
        empty = full ^ occupied
        bit = empty & -empty
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for mask, cells in candidates[i]:
                if mask & bit and not mask & occupied:
                    chosen.append(cells)
                    search(occupied | mask, rest)
                    chosen.pop()
                    if len(solutions) >= limit:
                        return

    search(0, tuple(range(len(pieces))))
    return solutions


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    panel, pieces, n = _read(inp)
    answers = _pack(pieces, n, _admitted(panel, pieces))
    _require(len(answers) == 1, "tunnel-qualified assembly must have one colored answer")
    return answers[0].copy()


def _extract(labels, colors):
    out = np.empty(labels.shape, dtype=np.int8)
    for i, color in enumerate(colors):
        out[labels == i] = color
    out[0, 0] = MARKER
    return out, [_tight(out, labels == i) for i in range(len(colors))]


def _corner_partition(rng, n, insert):
    """A procedural witness: the long slab can occupy the bottom or the right."""
    a = int(rng.integers(4 if insert else 2, n-1))
    labels = np.empty((n, n), dtype=np.int8)
    labels[:a, :a] = 0
    labels[:a, a:] = 1
    labels[a:, :] = 2
    if insert:
        # A carrier frame and an asymmetric/rectangular inset: not a template bank.
        h = int(rng.integers(1, a-1))
        w = int(rng.integers(1, a-1))
        if h*w < 2:
            w = 2
        r = int(rng.integers(1, a-h))
        c = int(rng.integers(1, a-w))
        labels[r:r+h, c:c+w] = 3
    vertical = bool(rng.integers(2))
    if vertical:
        labels = labels.T.copy()
    return labels, vertical


def _grown_partition(rng, n, count):
    labels = np.full((n, n), -1, dtype=np.int8)
    seeds = list(map(int, rng.choice(n*n, count, replace=False)))
    frontier, sizes = [set() for _ in range(count)], [1]*count
    weights = np.exp(rng.uniform(-1, 1, count))
    for i, pos in enumerate(seeds):
        labels.flat[pos] = i
    def add(i, pos):
        r, c = divmod(pos, n)
        for dr, dc in EDGE:
            rr, cc = r+dr, c+dc
            if 0 <= rr < n and 0 <= cc < n and labels[rr, cc] < 0:
                frontier[i].add(rr*n+cc)
    for i, pos in enumerate(seeds):
        add(i, pos)
    for _ in range(n*n-count):
        eligible = [i for i in range(count) if frontier[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*.5)
        cells = sorted(frontier[i])
        pos = cells[int(rng.integers(len(cells)))]
        labels.flat[pos] = i
        for f in frontier:
            f.discard(pos)
        sizes[i] += 1
        add(i, pos)
    return labels


def _panel(rng, n, capacity, vertical, stages, short_centered=False, first_min=None):
    # Square maneuvering rooms between separately positioned narrow passages.
    room_count = stages+1
    budget = 28-room_count*n
    _require(budget >= stages, "tunnel construction cannot fit")
    lengths = [1]*stages
    if not short_centered:
        for j in rng.permutation(stages):
            extra = int(rng.integers(0, min(2, budget-sum(lengths))+1))
            lengths[int(j)] += extra
    W = room_count*n+sum(lengths)+2
    panel = np.full((n+2, W), WALL, dtype=np.int8)
    x = 1
    widths = list(map(int, rng.integers(capacity, n+1, stages)))
    widths[int(rng.integers(stages))] = capacity
    if short_centered:
        widths = [capacity]*stages
    if first_min is not None:
        widths[0] = int(rng.integers(first_min, n+1))
        widths[-1] = capacity
    for j in range(room_count):
        panel[1:n+1, x:x+n] = ENTRY if j == 0 else EXIT if j == stages else FLOOR
        x += n
        if j < stages:
            width = widths[j]
            offset = 1 if short_centered else int(rng.integers(n-width+1))
            panel[1+offset:1+offset+width, x:x+lengths[j]] = FLOOR
            x += lengths[j]
    if vertical:
        panel = panel.T.copy()
    if rng.integers(2):
        panel = np.flip(panel, axis=0 if vertical else 1).copy()
    return panel


def _make_thread_pair(rng, n, stages, used):
    # A stair-shaped carrier spans n rows but can thread a short (n-1)-row
    # aperture by shifting sideways inside (n+1)-cell rooms. Complementary
    # triangles make this an ordinary rigid jigsaw under the same rule.
    labels = np.empty((n, n), dtype=np.int8)
    for r in range(n):
        for c in range(n):
            labels[r, c] = 2 if c < r else 0 if c <= r+1 else 1
    vertical = bool(rng.integers(2))
    if vertical:
        labels = labels.T.copy()
    colors = list(map(int, rng.choice(CARRIERS, 3, replace=False)))
    out, pieces = _extract(labels, colors)
    panel = _panel(rng, n+1, n-1, vertical, stages, short_centered=True)
    answers = _pack(pieces, n, _admitted(panel, pieces))
    _require(len(answers) == 1 and np.array_equal(answers[0], out), "stair threading construction")
    for _ in range(20):
        inp = _scatter(rng, panel, pieces)
        if _key(inp) not in used:
            used.add(_key(inp))
            return {"input": inp, "output": out}
    raise AssertionError("distinct stair scattering exhausted")


def _scatter(rng, panel, pieces):
    """Scatter by occupied masks, around any side of the physical tunnel."""
    for _ in range(20):
        canvas = np.zeros((30, 30), dtype=np.int8)
        occupied = np.zeros((30, 30), dtype=bool)
        ph, pw = panel.shape
        r0, c0 = int(rng.integers(31-ph)), int(rng.integers(31-pw))
        canvas[r0:r0+ph, c0:c0+pw] = panel
        occupied[r0:r0+ph, c0:c0+pw] = True
        oriented = [np.rot90(p, int(rng.integers(4))).copy() for p in pieces]
        if all(np.array_equal(p, q) for p, q in zip(pieces, oriented)):
            # Rotation is visibly demonstrated, including with only 2 trains.
            i = next(i for i, p in enumerate(pieces) if len(list(_orientations(p))) > 1)
            oriented[i] = next(q for q in _orientations(pieces[i]) if not np.array_equal(q, pieces[i]))
        for i in sorted(range(len(pieces)), key=lambda j: -oriented[j].size):
            q = oriented[i]
            blocked = np.zeros_like(occupied)
            padded = np.pad(occupied, 1)
            for dr in range(3):
                for dc in range(3):
                    blocked |= padded[dr:dr+30, dc:dc+30]
            possible = np.argwhere(_placements(~blocked, q >= 0))
            if not len(possible):
                break
            r, c = possible[int(rng.integers(len(possible)))]
            h, w = q.shape
            view = canvas[r:r+h, c:c+w]
            view[q >= 0] = q[q >= 0]
            occupied[r:r+h, c:c+w] |= q >= 0
        else:
            rr, cc = np.nonzero(occupied)
            top, bottom = max(0, int(rr.min())-int(rng.integers(3))), min(30, int(rr.max())+1+int(rng.integers(3)))
            left, right = max(0, int(cc.min())-int(rng.integers(3))), min(30, int(cc.max())+1+int(rng.integers(3)))
            return canvas[top:bottom, left:right].copy()
    raise AssertionError("bounded mask scattering exhausted")


def _make_pair(rng, n, stages, witness, insert, used):
    for attempt in range(100):
        use_witness = witness or attempt >= 72
        if use_witness:
            labels, vertical = _corner_partition(rng, n, insert)
        else:
            count = int(rng.integers(3, 5))
            labels = _grown_partition(rng, n, count)
            if min(np.bincount(labels.ravel())) < 2:
                continue
            vertical = bool(rng.integers(2))
        count = int(labels.max())+1
        colors = list(map(int, rng.choice(CARRIERS, count, replace=False)))
        out, pieces = _extract(labels, colors)
        capacity = max(p.shape[1 if vertical else 0] for p in pieces)
        if capacity >= n:
            continue
        # Extra width is sometimes harmless; a witness always has a true choke.
        capacity = int(rng.integers(capacity, n))
        room_side = min(8, n+int(rng.integers(3)))
        panel = _panel(rng, room_side, capacity, vertical, stages,
                       first_min=n if witness and stages == 2 else None)
        admitted = _admitted(panel, pieces)
        answers = _pack(pieces, n, admitted)
        if len(answers) != 1 or not np.array_equal(answers[0], out):
            continue
        if witness and len(_pack(pieces, n)) < 2:
            continue
        try:
            inp = _scatter(rng, panel, pieces)
        except AssertionError:
            continue
        key = _key(inp)
        if key in used:
            continue
        used.add(key)
        return {"input": inp, "output": out}
    raise AssertionError("bounded pair sampling exhausted")


def _counts(num_train, num_test):
    for value, lower, upper, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= int(value) <= upper:
            raise ValueError(name+" is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    insert = bool(rng.random() < .4)
    larger = bool(rng.integers(2))
    # At least one explicit coupling witness is present even with 2/1 counts.
    first_n = 6 if insert else int(rng.integers(4, 7))
    training = [(first_n, True, insert, False), (int(rng.integers(4, 7)), False, False, True)]
    training += [(int(rng.integers(4, 7)), False, False, False) for _ in range(ntrain-2)]
    training = [training[int(i)] for i in rng.permutation(ntrain)]
    puzzle, used = {"train": [], "test": []}, set()
    for n, witness, inset, thread in training:
        puzzle["train"].append(_make_thread_pair(rng, n, 1, used) if thread else _make_pair(rng, n, 1, witness, inset, used))
    for j in range(ntest):
        n = int(rng.integers(7, 9)) if larger else (6 if insert else int(rng.integers(4, 8)))
        # First test transfers the genuine interaction to two offset throats.
        puzzle["test"].append(_make_pair(rng, n, 2, j == 0, insert if j == 0 else False, used))
    return puzzle


def _facts(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    panel, pieces, n = _read(inp)
    admitted = _admitted(panel, pieces)
    answers = _pack(pieces, n, admitted)
    _require(len(answers) == 1 and np.array_equal(answers[0], out), "incorrect qualified packing")
    _require(out.shape == (n, n) and out[0, 0] == MARKER, "area or anchor violation")
    for color in (*CARRIERS, MARKER):
        _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color), "piece color/area changed")
    _require(set(map(int, np.unique(out))) <= {*CARRIERS, MARKER}, "scenery in answer")
    rotation_required = False
    for p in pieces:
        color = next(int(v) for v in np.unique(p) if v >= 0 and v != MARKER)
        mask = (out == color) | ((out == MARKER) if np.any(p == MARKER) else False)
        q = _tight(out, mask)
        _require(any(np.array_equal(q, v) for v in _orientations(p)), "nonrigid or reflected piece")
        rotation_required |= not np.array_equal(p, q)
        _require(_transit(panel, q), "output piece cannot transit")
    # Moving constraints alone leave several positions/orientations for at
    # least one unmarked object; a square exact-cover constraint is necessary.
    nonanchor_freedom = max(sum((n-q.shape[0]+1)*(n-q.shape[1]+1) for q in options)
                           for p, options in zip(pieces, admitted) if not np.any(p == MARKER))
    loose_count = len(_pack(pieces, n))
    entry_center = np.argwhere(panel == ENTRY).mean(axis=0)
    exit_center = np.argwhere(panel == EXIT).mean(axis=0)
    vertical = bool(entry_center[0] != exit_center[0])
    profiles = (panel != WALL).sum(axis=1 if vertical else 0)
    capacity = int(profiles[profiles > 0].min())
    shape_threading = any(q.shape[1 if vertical else 0] > capacity for qs in admitted for q in qs)
    normalized = np.full(inp.shape, -1, dtype=np.int8)
    mapping = {}
    for p in np.ndindex(inp.shape):
        value = int(inp[p])
        if value in CARRIERS:
            mapping.setdefault(value, len(mapping)+3)
            normalized[p] = mapping[value]
        else:
            normalized[p] = value
    return {"n": n, "count": len(pieces), "shape": inp.shape, "panel_shape": panel.shape,
            "coupled": loose_count > 1, "rotation": rotation_required,
            "capacity": capacity, "direction": tuple(map(int, np.sign(exit_center-entry_center))),
            "shape_threading": shape_threading,
            "nonanchor_freedom": nonanchor_freedom,
            "areas": tuple(sorted(int(np.count_nonzero(p >= 0)) for p in pieces)),
            "irregular": sum(not np.all(p >= 0) for p in pieces),
            "structure": sha256(bytes(normalized.shape)+normalized.tobytes()).hexdigest()}


def _widen_exit_half(panel):
    """Remove only interior walls on the exit side of the middle room."""
    a = np.argwhere(panel == ENTRY).mean(axis=0)
    b = np.argwhere(panel == EXIT).mean(axis=0)
    axis = 0 if a[0] != b[0] else 1
    slices = [slice(1, -1), slice(1, -1)]
    middle = panel.shape[axis]//2
    slices[axis] = slice(middle, -1) if b[axis] > a[axis] else slice(1, middle)
    wide = panel.copy()
    view = wide[tuple(slices)]
    view[view == WALL] = FLOOR
    return wide


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _counts(n, m)
    seen, records = set(), {}
    for split in ("train", "test"):
        records[split] = []
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(_facts(pair))
    _require(any(f["coupled"] for f in records["train"]), "training lacks a movement/packing interaction witness")
    _require(any(f["coupled"] for f in records["test"]), "test lacks an interaction witness")
    _require(any(f["rotation"] for f in records["train"]), "training must demonstrate rigid rotation")
    _require(any(f["shape_threading"] for f in records["train"]), "training must distinguish silhouette from bounding box")
    _require(all(f["nonanchor_freedom"] > 1 for fs in records.values() for f in fs), "packing contribution is vacuous")
    panel, pieces, n = _read(_grid(puzzle["test"][0]["input"]))
    _require(len(_pack(pieces, n, _admitted(_widen_exit_half(panel), pieces))) > 1,
             "first test must need its later constriction")
    return True


def _fingerprint(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape)+a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds; retain aggregate sets, never a corpus or execution traces."""
    shapes, areas, structures, fingerprints, sizes, counts = set(), set(), set(), set(), set(), set()
    coupled = irregular = rotated = pairs = threaded = 0
    directions, capacities = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 2:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _facts(pair)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                shapes.add(f["shape"])
                areas.add(f["areas"])
                structures.add(f["structure"])
                sizes.add(f["n"])
                counts.add(f["count"])
                coupled += f["coupled"]
                irregular += f["irregular"] > 0
                rotated += f["rotation"]
                threaded += f["shape_threading"]
                directions.add(f["direction"])
                capacities.add(f["capacity"])
                pairs += 1
    _require(len(fingerprints) == 42 and len(structures) >= 190, "structural diversity collapsed")
    _require(len(shapes) >= 25 and len(areas) >= 25, "geometry/area diversity collapsed")
    _require(sizes == {4, 5, 6, 7, 8} and counts == {3, 4}, "scale/count coverage")
    _require(coupled >= 84 and irregular >= 60 and rotated >= 160, "concept coverage")
    _require(len(directions) == 4 and len(capacities) >= 4, "tunnel direction/width coverage")
    _require(threaded >= 42, "full-silhouette threading coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(19, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(12, 2, 1)
    bad["test"][0]["output"][0, 0] = BACKGROUND
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    # Controlled counterfactual: widen the entire tunnel interior. The
    # original ambiguous jigsaw must then lose its unique answer.
    witness = next(p for p in generate_puzzle(0)["train"] if _facts(p)["coupled"])
    panel, pieces, n = _read(witness["input"])
    wide = panel.copy()
    wide[1:-1, 1:-1] = FLOOR
    wide[panel == ENTRY], wide[panel == EXIT] = ENTRY, EXIT
    _require(len(_pack(pieces, n, _admitted(wide, pieces))) == 2, "tunnel ablation ineffective")
    # A literal silhouette check: this staircase threads a short 3-cell
    # opening between 5-cell rooms; its solid 4x4 bounding box cannot.
    gate = np.full((7, 13), WALL, dtype=np.int8)
    gate[1:6, 1:6], gate[1:6, 7:12] = ENTRY, EXIT
    gate[2:5, 6] = FLOOR
    stair = np.array([[3, 3, -1, -1], [-1, 3, 3, -1],
                      [-1, -1, 3, 3], [-1, -1, -1, 3]], dtype=np.int8)
    _require(_transit(gate, stair), "occupied-cell staircase should pass")
    _require(not _transit(gate, np.full((4, 4), 3, dtype=np.int8)), "bounding box must be blocked")
    return {"seeds": 42, "pairs": pairs, "structures": len(structures), "input_shapes": len(shapes),
            "area_profiles": len(areas), "coupled_pairs": coupled, "irregular_pairs": irregular,
            "rotation_pairs": rotated, "threading_pairs": threaded, "tunnel_widths": sorted(capacities),
            "directions": [list(v) for v in sorted(directions)],
            "sides": sorted(sizes), "piece_counts": sorted(counts)}
