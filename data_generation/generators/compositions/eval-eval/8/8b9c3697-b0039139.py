"""Self-counting stencil capsules dock and print within their swept capacity.

Only NumPy and the Python standard library are used. Construction paints from
local object geometry; solve reconstructs global objects from its input alone.
"""
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _components(mask, diagonal=False):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    group.add(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _rect(r, c, h, w):
    return {(i, j) for i in range(r, r + h) for j in range(c, c + w)}


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _perimeter(r, c, h, w):
    return _rect(r, c, h, w) - _rect(r + 1, c + 1, h - 2, w - 2)


def _paint(grid, cells, color):
    for point in cells:
        grid[point] = color


def _dot(point, axis):
    return point[0] * axis[0] + point[1] * axis[1]


def _cell(fork, k, q):
    k += fork["kmin"]
    q += fork["qmin"]
    out, cross = fork["out"], fork["cross"]
    return k * out[0] + q * cross[0], k * out[1] + q * cross[1]


def _parse_fork(cells):
    candidates = []
    for out in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cross = (0, 1) if out[0] else (1, 0)
        kmin = min(_dot(p, out) for p in cells)
        qmin = min(_dot(p, cross) for p in cells)
        local = {(_dot(p, out) - kmin, _dot(p, cross) - qmin) for p in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 30)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, n in enumerate(lengths) for k in range(1, n + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        q0, q1, center = runs[1]
        if q0 + q1 != width:
            continue
        candidates.append(dict(out=out, cross=cross, kmin=kmin, qmin=qmin,
                               q0=q0, span=q1-q0, stop=center+1,
                               lengths=tuple(x[2] for x in runs), width=width,
                               cells=cells))
    _require(len(candidates) == 1, "fork lacks a unique positive three-spoke parse")
    return candidates[0]


def _capsules(grid, background, shell, ink):
    capsules, all_ink, covered = [], set(), set()
    for frame in _components(grid == shell):
        r0, r1, c0, c1 = _bounds(frame)
        h, w = r1-r0+1, c1-c0+1
        _require(h >= 3 and w >= 3 and frame == _perimeter(r0, c0, h, w),
                 "shell is not a rectangular perimeter")
        box = _rect(r0, c0, h, w)
        _require(not covered.intersection(box), "overlapping capsules")
        covered |= box
        inner = grid[r0+1:r1, c0+1:c1]
        _require(set(map(int, np.unique(inner))) <= {background, ink}, "foreign ink in capsule")
        cells = {p for p in box - frame if int(grid[p]) == ink}
        _require(bool(cells), "empty capsule")
        all_ink |= cells
        x0, x1, y0, y1 = _bounds(cells)
        mask = grid[x0:x1+1, y0:y1+1] == ink
        count = len(_components(mask))
        _require(count == len(_components(mask, True)), "diagonal counter ambiguity")
        capsules.append(dict(frame=frame, box=box, ink=cells, mask=mask,
                             count=count, bounds=(x0, x1, y0, y1),
                             padding=(x0-r0-1, r1-x1-1, y0-c0-1, c1-y1-1)))
    _require(capsules and all_ink == {tuple(map(int, p)) for p in np.argwhere(grid == ink)},
             "ink must be enclosed in capsules")
    return capsules


def _candidate(fork, capsule, grid, background):
    out, cross = fork["out"], fork["cross"]
    q0 = min(_dot(p, cross) for p in capsule["ink"]) - fork["qmin"]
    q1 = max(_dot(p, cross) for p in capsule["ink"]) - fork["qmin"]
    if q0 != fork["q0"] or q1 + 1 != q0 + fork["span"]:
        return None
    near = min(_dot(p, out) for p in capsule["ink"]) - fork["kmin"]
    far = max(_dot(p, out) for p in capsule["ink"]) - fork["kmin"]
    if near <= fork["stop"]:
        return None
    route = {_cell(fork, k, q) for k in range(fork["stop"], far + 1)
             for q in range(q0, q1 + 1)}
    for r, c in route:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return None
        if (r, c) not in capsule["box"] and int(grid[r, c]) != background:
            return None
    a, length = far-near+1, far-fork["stop"]+1
    capacity = (length + 1) // (a + 1)
    return dict(route=route, near=near, far=far, axial=a, length=length,
                capacity=capacity, repeats=min(capsule["count"], capacity),
                distance=near-fork["stop"])


def _analyze(input_grid):
    grid = np.asarray(input_grid)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9)),
             "invalid cells")
    background = int(grid[0, 0])
    boundary = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _require(np.all(boundary == background), "background boundary must be uniform")
    colors, frequencies = np.unique(grid, return_counts=True)
    _require(len(colors) == 4 and int(colors[np.argmax(frequencies)]) == background
             and sum(frequencies == frequencies.max()) == 1, "ambiguous background or palette")
    others = [int(c) for c in colors if c != background]
    parses = []
    for structure, shell, ink in permutations(others):
        try:
            forks = [_parse_fork(c) for c in _components(grid == structure)]
            capsules = _capsules(grid, background, shell, ink)
            occupied_capsules = set().union(*(c["box"] for c in capsules))
            _require(not any(f["cells"] & occupied_capsules for f in forks), "fork inside capsule")
            # Capsules and forks are separate objects even under eight-neighbor adjacency.
            objects = [f["cells"] for f in forks] + [c["box"] for c in capsules]
            seen = set()
            for cells in objects:
                expanded = {(r+dr, c+dc) for r, c in cells
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                _require(not seen.intersection(expanded), "touching input objects")
                seen |= cells
            parses.append(dict(background=background, structure=structure, shell=shell,
                               ink=ink, forks=forks, capsules=capsules))
        except AssertionError:
            continue
    _require(len(parses) == 1, "input lacks a unique geometric role parse")
    result = parses[0]
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(result["forks"]):
        choices = []
        for ci, capsule in enumerate(result["capsules"]):
            candidate = _candidate(fork, capsule, grid, background)
            if candidate is not None:
                choices.append((ci, candidate))
        _require(len(choices) <= 1, "multiple capsules feed one fork")
        if choices:
            ci, candidate = choices[0]
            _require(ci not in used and not swept.intersection(candidate["route"]),
                     "ambiguous assignments or intersecting routes")
            # Foreign capsule padding must not be silently overwritten, either.
            _require(all(not candidate["route"].intersection(c["box"])
                         for i, c in enumerate(result["capsules"]) if i != ci),
                     "route enters another capsule")
            used.add(ci)
            swept |= candidate["route"]
            matches.append(dict(candidate, fi=fi, ci=ci))
    result.update(matches=matches, used=used, swept=swept)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only visible input geometry and paint via axial modular indexing."""
    grid = np.asarray(input_grid)
    data = _analyze(grid)
    output = grid.astype(int, copy=True)
    for capsule in data["capsules"]:
        _paint(output, capsule["box"], data["background"])
    for match in data["matches"]:
        fork = data["forks"][match["fi"]]
        capsule = data["capsules"][match["ci"]]
        out = fork["out"]
        _paint(output, match["route"], data["shell"])
        for point in match["route"]:
            u = _dot(point, out) - fork["kmin"] - fork["stop"]
            i, t = divmod(u, match["axial"] + 1)
            if i < match["repeats"] and t < match["axial"]:
                shift = fork["stop"] - match["near"] + i * (match["axial"] + 1)
                original = point[0] - shift*out[0], point[1] - shift*out[1]
                if original in capsule["ink"]:
                    output[point] = data["ink"]
    return output


def _partition(length, n, rng):
    sizes = [1] * n
    gaps = [1] * (n-1)
    for _ in range(length - (2*n-1)):
        index = int(rng.integers(2*n-1))
        if index < n:
            sizes[index] += 1
        else:
            gaps[index-n] += 1
    result, start = [], 0
    for i, size in enumerate(sizes):
        result.append((start, start+size))
        start += size + (gaps[i] if i < n-1 else 0)
    return result


def _grow(h, w, area, rng):
    start = int(rng.integers(h)), int(rng.integers(w))
    cells, frontier = set(), {start}
    for _ in range(area):
        ordered = sorted(frontier)
        r, c = ordered[int(rng.integers(len(ordered)))]
        frontier.remove((r, c))
        cells.add((r, c))
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= p[0] < h and 0 <= p[1] < w and p not in cells:
                frontier.add(p)
    return cells


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4))
               for k in range(1, 8))


def _motif(rng, n, witness=False, test=False):
    """Random separated regions and connected growth, not a motif template bank."""
    for _ in range(240):
        if witness:
            h, w, area = int(rng.integers(3, 5)), int(rng.integers(5, 7)), 6
        elif test:
            h, w, area = (5 if n >= 5 else int(rng.integers(3, 6))), 3, None
        else:
            h, w, area = int(rng.integers(2, 6)), int(rng.integers(2, 7)), None
        choices = [(nr, nc) for nr in range(1, (h+1)//2+1)
                   for nc in range(1, (w+1)//2+1) if n <= nr*nc <= n+2]
        if not choices:
            continue
        nr, nc = choices[int(rng.integers(len(choices)))]
        rows, cols = _partition(h, nr, rng), _partition(w, nc, rng)
        boxes = [(r0, r1, c0, c1) for r0, r1 in rows for c0, c1 in cols]
        rng.shuffle(boxes)
        boxes = boxes[:n]
        capacities = [(r1-r0)*(c1-c0) for r0, r1, c0, c1 in boxes]
        if area is not None and sum(capacities) < area:
            continue
        sizes = [1] * n
        if area is None:
            sizes = [int(rng.integers(1, cap+1)) for cap in capacities]
        else:
            for _ in range(area-n):
                options = [i for i, cap in enumerate(capacities) if sizes[i] < cap]
                sizes[int(rng.choice(options))] += 1
        mask = np.zeros((h, w), dtype=bool)
        for (r0, r1, c0, c1), size in zip(boxes, sizes):
            bh, bw = r1-r0, c1-c0
            if area is None and n == 1 and bh >= 3 and bw >= 3 and rng.random() < .22:
                cells = _perimeter(0, 0, bh, bw)
            else:
                cells = _grow(bh, bw, size, rng)
            for r, c in cells:
                mask[r0+r, c0+c] = True
        points = np.argwhere(mask)
        lo, hi = points.min(axis=0), points.max(axis=0)+1
        mask = mask[lo[0]:hi[0], lo[1]:hi[1]].copy()
        if mask.shape[1] < 2 or (test and mask.shape[1] != 3):
            continue
        if witness and (min(mask.shape) < 2 or not _asymmetric(mask)):
            continue
        _require(len(_components(mask)) == n == len(_components(mask, True)), "islands merged")
        return mask
    raise AssertionError("bounded motif sampling exhausted")


def _module(rng, n, capacity, axis, sign, witness=False, test=False,
            exact=False, long_center=False):
    mask = _motif(rng, n, witness, test)
    a, s = mask.shape
    remainder = 0 if exact else int(rng.integers(0, min(a, 1 if witness else 2)+1))
    length = capacity*(a+1)-1+remainder
    distance = length-a
    center = 3 if long_center else int(rng.integers(1, 3 if witness else 4))
    stop, k0 = center+1, center+1+distance
    near_pad = int(rng.integers(0, min(1 if (witness or test) else 2, distance-2)+1))
    if witness and distance >= 3:
        near_pad = 1
    far_pad = int(rng.integers(0, 2 if (witness or test) else 3))
    p0, p1 = map(int, rng.integers(0, 2 if (witness or test) else 3, size=2))
    if witness:
        p0 = 1
    t, gap = (1, 1) if witness or test else (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
    width, q0 = s+2*(t+gap), t+gap
    frame_start = k0-near_pad-1
    guard_cap = min(5, frame_start-2, center-1 if long_center else 5)
    _require(guard_cap >= 1 and frame_start >= center+2, "capsule too close to fork")
    guards = list(map(int, rng.integers(1, guard_cap+1, size=2)))
    if witness and not long_center and guard_cap >= 2:
        guards = [1, guard_cap]
    fork = _rect(0, 0, 1, width)
    for q, thick, depth in ((0, t, guards[0]), (q0, s, center), (width-t, t, guards[1])):
        fork |= _rect(1, q, depth, thick)
    box = _rect(frame_start, q0-p0-1, a+near_pad+far_pad+2, s+p0+p1+2)
    shell = _perimeter(frame_start, q0-p0-1, a+near_pad+far_pad+2, s+p0+p1+2)
    ink = {(k0+int(k), q0+int(q)) for k, q in np.argwhere(mask)}
    route = _rect(stop, q0, length, s)
    qmin = min(q for _, q in fork | box)
    all_cells = fork | box | route
    depth = max(k for k, _ in all_cells)+1
    cross_size = max(q for _, q in all_cells)-qmin+1

    def physical(cells):
        result = set()
        for k, q in cells:
            q -= qmin
            z = k if sign == 1 else depth-1-k
            result.add((z, q) if axis == 0 else (q, z))
        return result

    printed = set()
    for i in range(min(n, capacity)):
        printed |= {(k0+int(k)+stop-k0+i*(a+1), q0+int(q))
                    for k, q in np.argwhere(mask)}
    return dict(fork=physical(fork), shell=physical(shell), ink=physical(ink),
                box=physical(box), route=physical(route), printed=physical(printed),
                shape=(depth, cross_size) if axis == 0 else (cross_size, depth))


def _distractor(rng, axis, witness=False, test=False):
    # Witness distractors have the same six-cell ink area as the active
    # payloads, and quota two. Their geometry, rather than their quantity,
    # rejects them. Other examples include blocks, bends, rings and fragments.
    a = int(rng.integers(3, 5)) if witness else int(rng.integers(1, 5))
    s = 3 if witness else int(rng.integers(1, 3 if test else 4))
    mask = np.ones((a, s), dtype=bool)
    mode = 2 if witness else int(rng.integers(4))
    if mode == 1:
        mask[:] = False
        mask[0, :] = mask[:, 0] = True
    elif mode == 2 and a >= 3:
        mask[:] = False
        mask[[0, -1], :] = True
    elif mode == 3 and min(a, s) >= 3:
        mask[1:-1, 1:-1] = False
    near, far = map(int, rng.integers(0, 2, size=2))
    side = 0 if test else int(rng.integers(2))
    before = int(rng.integers(side+1))
    shape = a+near+far+2, s+side+2
    box = _rect(0, 0, *shape)
    shell = _perimeter(0, 0, *shape)
    ink = {(int(k)+near+1, int(q)+before+1) for k, q in np.argwhere(mask)}
    if axis == 1:
        shape = shape[::-1]
        box, shell, ink = ({(c, r) for r, c in cells} for cells in (box, shell, ink))
    return dict(fork=set(), shell=shell, ink=ink, box=box,
                route=set(), printed=set(), shape=shape)


def _pack(rng, modules, stack_axis=None):
    gap = 1
    options = []
    for axis in (0, 1) if stack_axis is None else (stack_axis,):
        extent = sum(m["shape"][axis] for m in modules)+gap*(len(modules)-1)
        other = max(m["shape"][1-axis] for m in modules)
        if extent <= 28 and other <= 28:
            options.append((axis, extent, other))
    _require(bool(options), "reservations do not fit")
    axis, extent, other = options[int(rng.integers(len(options)))]
    core = (extent, other) if axis == 0 else (other, extent)
    extra = [int(rng.integers(min(4, 28-n)+1)) for n in core]
    shape = tuple(core[i]+extra[i]+2 for i in (0, 1))
    border = [int(rng.integers(extra[i]+1))+1 for i in (0, 1)]
    rng.shuffle(modules)
    result, offset = [], 0
    for module in modules:
        cross_offset = int(rng.integers(other-module["shape"][1-axis]+1))
        r, c = (offset, cross_offset) if axis == 0 else (cross_offset, offset)
        r += border[0]
        c += border[1]
        result.append({name: {(x+r, y+c) for x, y in module[name]}
                       for name in ("fork", "shell", "ink", "box", "route", "printed")})
        offset += module["shape"][axis]+gap
    return shape, result


def _pair(rng, palette, kind, first_axis):
    for attempt in range(60):
        if kind == 0:
            axis = first_axis
            specs = [(2, 3, 1, False, False), (3, 3, -1, False, False)]
        elif kind == 1:
            axis = 1-first_axis
            specs = [(3, 2, 1, True, True), (2, 3, -1, False, False)]
        elif kind == 3:
            axis = int(rng.integers(2))
            specs = [(int(rng.integers(4, 7)), int(rng.integers(2, 5)), int(rng.choice([-1, 1])),
                      bool(rng.integers(2)), bool(rng.integers(2)))]
            specs += [(int(rng.integers(1, 5)), int(rng.integers(2, 5)), int(rng.choice([-1, 1])),
                       bool(rng.integers(2)), bool(rng.integers(2))) for _ in range(2)]
        else:
            axis = int(rng.integers(2))
            specs = [(int(rng.integers(1, 4)), int(rng.integers(2, 5)), int(rng.choice([-1, 1])),
                      bool(rng.integers(2)), bool(rng.integers(2)))
                     for _ in range(int(rng.integers(1, 3)))]
        try:
            modules = [_module(rng, n, cap, axis, sign, witness=kind in (0, 1),
                               test=kind == 3, exact=exact, long_center=long)
                       for n, cap, sign, exact, long in specs]
            modules.append(_distractor(rng, axis, witness=kind in (0, 1), test=kind == 3))
            shape, placed = _pack(rng, modules, 1-axis if kind in (0, 1, 3) else None)
            background, structure, shell, ink = palette
            grid = np.full(shape, background, dtype=int)
            output = grid.copy()
            for module in placed:
                _paint(grid, module["fork"], structure)
                _paint(grid, module["shell"], shell)
                _paint(grid, module["ink"], ink)
                _paint(output, module["fork"], structure)
                _paint(output, module["route"], shell)
                _paint(output, module["printed"], ink)
            data = _analyze(grid)
            _require(len(data["matches"]) == len(specs), "unexpected global assignment")
            _require(len(data["capsules"])-len(data["used"]) >= 1, "missing distractor")
            return {"input": grid, "output": output}
        except AssertionError:
            # Retry geometry/packing, never infer a parse from a proposed label.
            continue
    raise AssertionError("bounded pair construction exhausted")


def _valid_count(value, lo, hi):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and lo <= int(value) <= hi


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_count(num_train, 2, 10) or not _valid_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("require integer counts 2<=num_train<=10, 1<=num_test<=4, num_train>num_test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palettes = [tuple(map(int, rng.choice(10, 4, replace=False))) for _ in range(3)]
    first_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(amount):
            kind = (index if index < 2 else 2) if split == "train" else 3
            for _ in range(20):
                palette = palettes[index % 3] if index < 2 else palettes[int(rng.integers(3))]
                if rng.random() < .35:
                    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _pair(rng, palette, kind, first_axis)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct-input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, output = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, output):
                _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid shape")
                _require(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)), "invalid color")
            _require(inp.shape == output.shape and _key(inp) not in seen, "shape change or duplicate input")
            seen.add(_key(inp))
            data = _analyze(inp)
            _require(data["matches"] and len(data["used"]) < len(data["capsules"]), "missing active dock or distractor")
            _require(np.array_equal(solve(inp), output), "output violates docking/printing rule")
            _require(not np.array_equal(inp, output), "identity example")
            capsule_cells = set().union(*(c["box"] for c in data["capsules"]))
            for fork in data["forks"]:
                _require(all(int(output[p]) == data["structure"] for p in fork["cells"]), "fork changed")
            for point in capsule_cells - data["swept"]:
                _require(int(output[point]) == data["background"], "capsule shell or distractor survived")
            changed_region = capsule_cells | data["swept"]
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if (r, c) not in changed_region:
                        _require(inp[r, c] == output[r, c], "unrelated cell changed")
            for m in data["matches"]:
                capsule = data["capsules"][m["ci"]]
                _require(m["repeats"] >= 1 and m["repeats"]*(m["axial"]+1)-1 <= m["length"], "partial or oversized print")
                primary_area = sum(int(output[p]) == data["ink"] for p in m["route"])
                _require(primary_area == m["repeats"]*len(capsule["ink"]), "ink geometry lost or duplicated")
                _require(all(int(output[p]) in (data["ink"], data["shell"]) for p in m["route"]), "route has wrong palette")
            (training if split == "train" else tests).append(data)
    active = [(d, m, d["capsules"][m["ci"]], d["forks"][m["fi"]])
              for d in training for m in d["matches"]]
    _require({f["out"] for _, _, _, f in active} == {(1, 0), (-1, 0), (0, 1), (0, -1)}, "training misses docking direction")
    _require(any(c["count"] < m["capacity"] for _, m, c, _ in active), "B quota ablation does not change answer")
    _require(any(c["count"] > m["capacity"] for _, m, c, _ in active), "A capacity ablation does not change answer")
    _require(any(c["count"] > m["capacity"] and m["length"] == m["repeats"]*(m["axial"]+1)-1
                 for _, m, c, _ in active), "missing exact-fit clipping witness")
    _require(any(len(a[2]["ink"]) == len(b[2]["ink"]) == 6 and a[2]["count"] == 2 and b[2]["count"] == 3
                 and a[1]["repeats"] == 2 and b[1]["repeats"] == 3 for a in active for b in active),
             "pixel count not distinguished from component count")
    _require(any(f["lengths"][1] > max(f["lengths"][0], f["lengths"][2]) for _, _, _, f in active), "missing central-tip witness")
    _require(any(sum(c["padding"]) > 0 and c["count"] > 1 and _asymmetric(c["mask"]) for _, _, c, _ in active), "missing crop/orientation witness")
    _require(any(c["count"] == 2 and len(c["ink"]) == 6 for d in training
                 for i, c in enumerate(d["capsules"]) if i not in d["used"]),
             "rejection is confounded with small/connected payloads")
    _require(all(1 <= len(d["forks"]) <= 2 for d in training), "training receiver envelope")
    maximum_train_quota = max(c["count"] for _, _, c, _ in active)
    _require(all(len(d["forks"]) == len(d["matches"]) == 3
                 and max(c["count"] for c in d["capsules"] if c["count"] > 1) > maximum_train_quota
                 for d in tests), "missing simultaneous-receiver/quantity test shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded semantic, corruption, count-handling and 42-seed diversity checks."""
    shapes, geometries, motifs, directions, quotas, capacities, signatures = set(), set(), set(), set(), set(), set(), set()
    areas, layouts, palettes, clippings, repeats = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        if seed in (0, 17, 41):
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        for split in puzzle:
            for pair in puzzle[split]:
                data = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                palettes.add(tuple(data[k] for k in ("background", "structure", "shell", "ink")))
                layouts.add(tuple(sorted(tuple(sorted(f["cells"])) for f in data["forks"])))
                for f in data["forks"]:
                    directions.add(f["out"])
                    geometries.add((f["span"], f["width"], f["lengths"]))
                for c in data["capsules"]:
                    motifs.add(_key(c["mask"]))
                    quotas.add(c["count"])
                    areas.add(tuple(sorted(len(x) for x in _components(c["mask"]))))
                for m in data["matches"]:
                    capacities.add(m["capacity"])
                    repeats.add(m["repeats"])
                    clippings.add(data["capsules"][m["ci"]]["count"] > m["capacity"])
    _require(len(signatures) == 42 and len(shapes) >= 20 and len(layouts) >= 150, "layout diversity collapsed")
    _require(len(motifs) >= 80 and len(geometries) >= 40 and len(areas) >= 20, "object geometry collapsed")
    _require(len(palettes) >= 40 and directions == {(1, 0), (-1, 0), (0, 1), (0, -1)}, "palette/direction diversity collapsed")
    _require(set(range(1, 7)) <= quotas and {2, 3, 4} <= capacities and clippings == {False, True}, "quota/capacity diversity collapsed")
    _require({1, 2, 3, 4} <= repeats, "printed copy counts fail to extrapolate")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(6000+10*nt+nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, None), (4.0, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupted = generate_puzzle(9, 2, 1)
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A hand-constructed input outside the ordinary sampling envelope: capacity
    # one, one inactive fork, and no unmatched capsule. It checks solve itself,
    # not the deliberately stricter demonstration-set validator.
    grid = np.zeros((14, 14), dtype=int)
    fork = _rect(1, 1, 1, 8) | _rect(2, 1, 1, 1) | _rect(2, 3, 1, 4) | _rect(2, 8, 1, 1)
    _paint(grid, fork, 1)
    shell = _perimeter(6, 2, 6, 6)
    _paint(grid, shell, 2)
    ink = {(7, 3), (8, 3), (9, 3), (10, 3), (7, 6), (8, 6)}
    _paint(grid, ink, 3)
    inactive = _rect(2, 11, 8, 1) | _rect(2, 10, 1, 1) | _rect(4, 10, 4, 1) | _rect(9, 10, 1, 1)
    _paint(grid, inactive, 1)
    expected = grid.copy()
    _paint(expected, _rect(6, 2, 6, 6), 0)
    _paint(expected, _rect(3, 3, 8, 4), 2)
    _paint(expected, {(r-4, c) for r, c in ink}, 3)
    data = _analyze(grid)
    _require(len(data["matches"]) == 1 and data["matches"][0]["capacity"] == 1, "hand fixture capacity")
    _require(np.array_equal(solve(grid), expected), "input-only capacity-one/inactive-fork solve failed")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "fork_geometries": len(geometries), "stencil_structures": len(motifs),
            "island_size_lists": len(areas), "palettes": len(palettes),
            "quotas": sorted(quotas), "capacities": sorted(capacities),
            "printed_copy_counts": sorted(repeats)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
