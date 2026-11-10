"""Capacity-ordered reservoirs: pool ink and repack it inside intact frames.

Only standard-library modules and NumPy are used.  No parent generator is imported.
All semantic color roles are recoverable from each input independently.
"""

from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes() if a.dtype == np.int8 else a.astype(np.int8).tobytes()


def _capacity(rect):
    return (rect[2] - 2) * (rect[3] - 2)


def _components(mask):
    remaining = set(zip(*np.where(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((int(r), int(c)))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(cells)
    return result


def _separate(a, b):
    r, c, h, w = a
    s, d, j, k = b
    return r + h < s or s + j < r or c + w < d or d + k < c


def _decode(input_grid):
    """Find the unique closed-frame interpretation, without frequency heuristics."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (a[0, -1] == a[-1, 0] == a[-1, -1] == bg):
        raise ValueError("The four canvas corners must be background")
    colors = [int(c) for c in np.unique(a)]
    if len(colors) != 3:
        raise ValueError("A reservoir input has exactly three colors")
    candidates = []
    for frame in (c for c in colors if c != bg):
        ink = next(c for c in colors if c not in (bg, frame))
        rectangles, covered = [], np.zeros(a.shape, dtype=bool)
        valid = True
        for cells in _components(a == frame):
            rr, cc = zip(*cells)
            r, c = min(rr), min(cc)
            h, w = max(rr) - r + 1, max(cc) - c + 1
            if h < 3 or w < 3 or len(cells) != 2 * h + 2 * w - 4:
                valid = False
                break
            sub = a[r:r + h, c:c + w]
            if not (np.all(sub[0] == frame) and np.all(sub[-1] == frame)
                    and np.all(sub[:, 0] == frame) and np.all(sub[:, -1] == frame)
                    and not np.any(sub[1:-1, 1:-1] == frame)):
                valid = False
                break
            rectangles.append((r, c, h, w))
            covered[r + 1:r + h - 1, c + 1:c + w - 1] = True
        if not valid or not 2 <= len(rectangles) <= 5:
            continue
        if any(not _separate(x, y) for i, x in enumerate(rectangles)
               for y in rectangles[i + 1:]):
            continue
        capacities = [_capacity(x) for x in rectangles]
        if len(set(capacities)) != len(capacities):
            continue
        if np.any((a == ink) & ~covered):
            continue
        n = int(np.count_nonzero(a == ink))
        if not 1 <= n < sum(capacities):
            continue
        candidates.append((bg, frame, ink, sorted(rectangles, key=_capacity), n))
    if len(candidates) != 1:
        raise ValueError("Expected one unambiguous set of separated rectangular frames")
    return candidates[0]


def _pack(a, bg, ink, rectangles, n):
    out = a.copy()
    for r, c, h, w in rectangles:
        out[r + 1:r + h - 1, c + 1:c + w - 1] = bg
    for r, c, h, w in rectangles:
        take = min(n, (h - 2) * (w - 2))
        # Explicit coordinates avoid reshaping a noncontiguous NumPy slice.
        for i in range(take):
            dr, dc = divmod(i, w - 2)
            out[r + 1 + dr, c + 1 + dc] = ink
        n -= take
    if n:
        raise ValueError("Ink exceeds total capacity")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pool all ink; fill smallest usable interiors first, each in reading order."""
    a = _grid(input_grid)
    bg, _, ink, rectangles, n = _decode(a)
    return _pack(a, bg, ink, rectangles, n)


_DIMS = tuple((h, w) for h in range(3, 13) for w in range(3, 13)
              if (h - 2) * (w - 2) <= 32)
_LARGE = tuple((h, w) for h in range(3, 13) for w in range(3, 13)
               if 32 < (h - 2) * (w - 2) <= 64)


def _dim_capacity(d):
    return (d[0] - 2) * (d[1] - 2)


_CONFLICTS = tuple((a, b) for a in _DIMS for b in _DIMS
                   if _dim_capacity(a) < _dim_capacity(b) <= 24
                   and a[0] * a[1] > b[0] * b[1] and min(b) >= 4)
_MODES = ("scatter", "connected", "runs", "spread", "corner", "block")


def _sample_dims(rng, k, large=False):
    pool = _DIMS
    capacities = sorted({_dim_capacity(d) for d in pool})
    chosen = [int(x) for x in rng.choice(capacities, k - int(large), replace=False)]
    dims = []
    for cap in chosen:
        options = [d for d in pool if _dim_capacity(d) == cap]
        dims.append(options[int(rng.integers(len(options)))])
    if large:
        dims.append(_LARGE[int(rng.integers(len(_LARGE)))])
    return sorted(dims, key=_dim_capacity)


def _layout(rng, dims, small_not_first=False):
    """Random packing, with a one-cell gap; crop and add independent margins."""
    total = sum(h * w for h, w in dims)
    order = sorted(range(len(dims)), key=lambda i: -dims[i][0] * dims[i][1])
    for attempt in range(160):
        if attempt < 110:
            scale = float(rng.uniform(1.35, 1.95))
            aspect = float(rng.uniform(0.68, 1.48))
            gh = min(26, max(max(h for h, _ in dims), int(np.sqrt(total) * scale * aspect)))
            gw = min(26, max(max(w for _, w in dims), int(np.sqrt(total) * scale / aspect)))
        else:
            gh = gw = 26
        placed = {}
        for index in order:
            h, w = dims[index]
            positions = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
            rng.shuffle(positions)
            for r, c in positions:
                candidate = (r, c, h, w)
                if all(_separate(candidate, rect) for rect in placed.values()):
                    placed[index] = candidate
                    break
            else:
                break
        if len(placed) != len(dims):
            continue
        rects = [placed[i] for i in range(len(dims))]
        if small_not_first and min(range(len(rects)), key=lambda i: rects[i][:2]) == 0:
            continue
        r0, c0 = min(x[0] for x in rects), min(x[1] for x in rects)
        rh = max(x[0] + x[2] for x in rects) - r0
        cw = max(x[1] + x[3] for x in rects) - c0
        margins = [int(x) for x in rng.integers(0, 3, size=4)]
        # Keeping either pair of opposite margins nonzero guarantees four blank
        # corners while still allowing frames on the other two canvas edges.
        if bool(rng.integers(2)):
            margins[0] = max(1, margins[0])
            margins[1] = max(1, margins[1])
        else:
            margins[2] = max(1, margins[2])
            margins[3] = max(1, margins[3])
        top, bottom, left, right = margins
        shape = (rh + top + bottom, cw + left + right)
        rects = [(r - r0 + top, c - c0 + left, h, w) for r, c, h, w in rects]
        return shape, rects
    raise RuntimeError("Could not pack the reservoirs")


def _positions(rng, h, w, n, mode):
    area = h * w
    if n == 0:
        return []
    if mode == "connected":
        chosen = {int(rng.integers(area))}
        while len(chosen) < n:
            frontier = sorted({rr * w + cc for x in chosen for rr, cc in
                               ((x // w - 1, x % w), (x // w + 1, x % w),
                                (x // w, x % w - 1), (x // w, x % w + 1))
                               if 0 <= rr < h and 0 <= cc < w} - chosen)
            chosen.add(frontier[int(rng.integers(len(frontier)))])
        return sorted(chosen)
    if mode == "runs":
        rows, cols = list(range(h)), list(range(w))
        if rng.integers(2):
            rows.reverse()
        if rng.integers(2):
            cols.reverse()
        order = ([r * w + c for r in rows for c in cols] if rng.integers(2)
                 else [r * w + c for c in cols for r in rows])
        offset = int(rng.integers(area - n + 1))
        return order[offset:offset + n]
    if mode == "spread":
        chosen = [int(rng.integers(area))]
        while len(chosen) < n:
            remaining = [i for i in range(area) if i not in chosen]
            distances = [min(abs(i // w - j // w) + abs(i % w - j % w)
                             for j in chosen) for i in remaining]
            best = max(distances)
            options = [i for i, d in zip(remaining, distances) if d == best]
            chosen.append(options[int(rng.integers(len(options)))])
        return chosen
    if mode in ("corner", "block"):
        cr = int(rng.choice([0, h - 1])) if mode == "corner" else float(rng.uniform(0, h))
        cc = int(rng.choice([0, w - 1])) if mode == "corner" else float(rng.uniform(0, w))
        jitter = rng.uniform(0, 0.8, area)
        return sorted(range(area), key=lambda i:
                      abs(i // w - cr) + abs(i % w - cc) + jitter[i])[:n]
    return [int(x) for x in rng.choice(area, n, replace=False)]


def _render(rng, dims, n, colors, mode=None, witness=False):
    bg, frame, ink = colors
    shape, rects = _layout(rng, dims, small_not_first=witness)
    a = np.full(shape, bg, dtype=np.int8)
    for r, c, h, w in rects:
        a[r:r + h, c:c + w] = frame
        a[r + 1:r + h - 1, c + 1:c + w - 1] = bg
    caps = [_capacity(x) for x in rects]
    allocations = [0] * len(rects)
    # Witnesses start with ink in every bin, including the bin emptied by pooling.
    if witness:
        allocations = [1] * len(rects)
        allocations[-1] += 1
    remaining = n - sum(allocations)
    slots = [i for i, cap in enumerate(caps) for _ in range(cap - allocations[i])]
    for index in rng.choice(len(slots), remaining, replace=False):
        allocations[slots[int(index)]] += 1
    for (r, c, h, w), take in zip(rects, allocations):
        local_mode = mode or _MODES[int(rng.integers(len(_MODES)))]
        for index in _positions(rng, h - 2, w - 2, take, local_mode):
            dr, dc = divmod(index, w - 2)
            a[r + 1 + dr, c + 1 + dc] = ink
    # Ground truth is rendered from the sampled geometry/count, not by solve().
    b = np.full(shape, bg, dtype=np.int8)
    for r, c, h, w in rects:
        b[r:r + h, c:c + w] = frame
        b[r + 1:r + h - 1, c + 1:c + w - 1] = bg
        take = min(n, (h - 2) * (w - 2))
        full_rows, tail = divmod(take, w - 2)
        b[r + 1:r + 1 + full_rows, c + 1:c + w - 1] = ink
        if tail:
            b[r + 1 + full_rows, c + 1:c + 1 + tail] = ink
        n -= take
    return {"input": a, "output": b}


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                 (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    palette_mode, shift = int(rng.integers(3)), int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_count = None
    train_max_count = 0

    def colors():
        if palette_mode == 0:
            return palette
        if palette_mode == 1:
            return palette[:2] + (int(rng.choice([c for c in range(10) if c not in palette[:2]])),)
        return tuple(int(x) for x in rng.choice(10, 3, replace=False))

    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for _ in range(150):
                witness = split == "train" and index < 2
                if split == "train" and index == 0:
                    x, y = _CONFLICTS[int(rng.integers(len(_CONFLICTS)))]
                    options = [d for d in _DIMS if _dim_capacity(d) > _dim_capacity(y)]
                    dims = [x, y, options[int(rng.integers(len(options)))]]
                    # A partial odd row distinguishes row-major from serpentine,
                    # column-major, bottom-up, and one-horizontal-bar alternatives.
                    width = y[1] - 2
                    # Exclude capacity-1: with only the bottom-right cell empty,
                    # row-major and column-major would give the same pattern.
                    choices = list(range(width + 1, min(2 * width, _dim_capacity(y) - 1)))
                    if not choices:
                        continue
                    partial = choices[int(rng.integers(len(choices)))]
                    n = _dim_capacity(x) + partial
                    mode = "connected"
                elif split == "train" and index == 1:
                    dims = _sample_dims(rng, 3)
                    n = sum(_dim_capacity(d) for d in dims[:2])
                    if n < 7 or n == first_count:
                        continue
                    mode = "spread"
                elif split == "train":
                    dims = _sample_dims(rng, int(rng.integers(2, 4)))
                    cap = sum(_dim_capacity(d) for d in dims)
                    upper = min(cap - 1, sum(_dim_capacity(d) for d in dims[:2]))
                    n = (first_count if index == 2 and first_count <= upper and rng.integers(2)
                         else int(rng.integers(1, upper + 1)))
                    mode = None
                elif shift == 0:  # New number of reservoirs.
                    dims = _sample_dims(rng, int(rng.integers(4, 6)))
                    cap = sum(_dim_capacity(d) for d in dims)
                    n = int(rng.integers(_dim_capacity(dims[0]) + 1, cap))
                    mode = None
                elif shift == 1:  # Same quantity, a new larger usable capacity.
                    dims = _sample_dims(rng, int(rng.integers(2, 5)), large=True)
                    n = first_count
                    if n >= sum(_dim_capacity(d) for d in dims):
                        continue
                    mode = None
                else:  # Carry continues into the third bin, beyond training count.
                    dims = _sample_dims(rng, 3)
                    prefix = sum(_dim_capacity(d) for d in dims[:2])
                    cap = prefix + _dim_capacity(dims[2])
                    lower = max(prefix + 1, train_max_count + 1)
                    if lower >= cap:
                        continue
                    n = int(rng.integers(lower, cap))
                    mode = None
                pair = _render(rng, dims, n, colors(), mode, witness)
                key = _key(pair["input"])
                if key in seen or np.array_equal(pair["input"], pair["output"]):
                    continue
                # Parsing here also rejects any accidental role ambiguity.
                _decode(pair["input"])
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_max_count = max(train_max_count, n)
                    if index == 0:
                        first_count = n
                break
            else:
                raise RuntimeError("Could not generate a distinct reservoir example")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check conservation, capacity priority, scan order, and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "Canvas shape changed")
            bg, frame, ink, rects, n = _decode(a)
            _require(_key(a) not in seen, "Repeated input")
            seen.add(_key(a))
            _require(not np.array_equal(a, b), "Generated identity example")
            _require(np.count_nonzero(b == ink) == n, "Ink was not conserved")
            _require(np.array_equal(a == frame, b == frame), "Frame cells changed")
            inside = np.zeros(a.shape, dtype=bool)
            left = n
            occupancies = []
            for r, c, h, w in rects:
                inside[r + 1:r + h - 1, c + 1:c + w - 1] = True
                flat = b[r + 1:r + h - 1, c + 1:c + w - 1].flatten()
                take = min(left, len(flat))
                _require(np.all(flat[:take] == ink) and np.all(flat[take:] == bg),
                         "Interior does not follow capacity priority and row-major packing")
                occupancies.append(take)
                left -= take
            _require(np.array_equal(a[~inside], b[~inside]), "A cell outside an interior changed")
            _require(np.array_equal(solve(a), b), "Input-only solver disagrees")
            analyses[split].append((bg, frame, ink, rects, n, occupancies))
    # The first demonstration distinguishes geometry ranking, outer area,
    # connected-component counting, and wrapping conventions simultaneously.
    a, b = (np.asarray(puzzle["train"][0][k]) for k in ("input", "output"))
    bg, _, ink, rects, n, occ = analyses["train"][0]
    _require(len(rects) == 3 and occ[0] == _capacity(rects[0]) and 0 < occ[1] < _capacity(rects[1])
             and occ[2] == 0, "Missing full/partial/empty teaching contrast")
    _require(not np.array_equal(_pack(a, bg, ink, sorted(rects, key=lambda x: x[2] * x[3]), n), b),
             "Outer-area shortcut not distinguished")
    _require(not np.array_equal(_pack(a, bg, ink, sorted(rects), n), b),
             "Spatial-order shortcut not distinguished")
    _require(len(_components(a == ink)) < n, "Cell-count versus component-count contrast missing")
    partial_width = rects[1][3] - 2
    _require(partial_width < occ[1] < 2 * partial_width, "Multirow non-serpentine evidence missing")
    r, c, h, w = rects[1]
    column_major = np.full((h - 2, w - 2), bg, dtype=np.int8)
    for i in range(occ[1]):
        dr, dc = i % (h - 2), i // (h - 2)
        column_major[dr, dc] = ink
    _require(not np.array_equal(column_major, b[r + 1:r + h - 1, c + 1:c + w - 1]),
             "Column-major shortcut not distinguished")
    _require(np.any(a[rects[2][0] + 1:rects[2][0] + rects[2][2] - 1,
                         rects[2][1] + 1:rects[2][1] + rects[2][3] - 1] == ink),
             "Pooling versus local compaction contrast missing")
    second = analyses["train"][1]
    _require(second[4] != n and second[5] == [_capacity(second[3][0]), _capacity(second[3][1]), 0],
             "Different count and exact-capacity stopping evidence missing")
    train = analyses["train"]
    test = analyses["test"]
    max_bins = max(len(x[3]) for x in train)
    max_cap = max(_capacity(r) for x in train for r in x[3])
    max_n = max(x[4] for x in train)
    shifts = (
        all(len(x[3]) > max_bins for x in test),
        all(max(map(_capacity, x[3])) > max_cap and x[4] == n for x in test),
        all(x[4] > max_n and len(x[3]) == 3 and 0 < x[5][-1] < _capacity(x[3][-1]) for x in test),
    )
    _require(any(shifts), "No intended train/test shift is present")


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[k])
                 for k in ("train", "test"))


def self_test():
    """Bounded 42-seed survey, counts, determinism, and semantic metamorphisms."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "Repeated puzzles across seeds")
    _require(_puzzle_key(generate_puzzle(371, 8, 3)) == _puzzle_key(generate_puzzle(371, 8, 3)),
             "Seed reproducibility failed")
    shapes, geometry, capacities, quantities, palettes, bins = set(), set(), set(), set(), set(), set()
    edges = set()
    shift_modes = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        tr = [_decode(x["input"]) for x in puzzle["train"]]
        te = [_decode(x["input"]) for x in puzzle["test"]]
        if all(len(x[3]) > max(len(y[3]) for y in tr) for x in te):
            shift_modes.add("bins")
        if all(max(map(_capacity, x[3])) > max(_capacity(r) for y in tr for r in y[3]) for x in te):
            shift_modes.add("capacity")
        if all(x[4] > max(y[4] for y in tr) for x in te):
            shift_modes.add("quantity")
        for pair in puzzle["train"] + puzzle["test"]:
            a = pair["input"]
            bg, frame, ink, rects, n = _decode(a)
            shapes.add(a.shape)
            geometry.add(tuple((r[2], r[3]) for r in rects))
            capacities.update(map(_capacity, rects))
            quantities.add(n)
            palettes.add((bg, frame, ink))
            bins.add(len(rects))
            for r, c, h, w in rects:
                edges.add((r == 0 or c == 0 or r + h == a.shape[0] or c + w == a.shape[1]))
        # Same quantity under a maximally different source arrangement must have
        # the same answer; changing the count by one must change the answer.
        pair = puzzle["train"][0]
        a = pair["input"]
        bg, _, ink, rects, n = _decode(a)
        alt = a.copy()
        slots = []
        for r, c, h, w in rects:
            alt[r + 1:r + h - 1, c + 1:c + w - 1] = bg
            slots.extend((rr, cc) for rr in range(r + 1, r + h - 1)
                         for cc in range(c + 1, c + w - 1))
        for r, c in slots[-n:]:
            alt[r, c] = ink
        _require(np.array_equal(solve(alt), pair["output"]), "Arrangement invariance failed")
        r, c = slots[-n]
        alt[r, c] = bg
        _require(not np.array_equal(solve(alt), pair["output"]), "Exact count is not causal")
        # A bijective color relabeling must preserve all geometry and behavior.
        permutation = np.array([7, 3, 9, 1, 6, 0, 8, 5, 2, 4], dtype=np.int8)
        _require(np.array_equal(solve(permutation[a]), permutation[pair["output"]]),
                 "Palette-role invariance failed")
    _require(len(shapes) >= 35 and len(geometry) >= 100, "Insufficient geometric variation")
    _require(len(capacities) >= 20 and len(quantities) >= 30, "Insufficient capacity/count variation")
    _require(len(palettes) >= 30 and bins == {2, 3, 4, 5}, "Insufficient palette/bin variation")
    _require(edges == {False, True} and len(shift_modes) == 3, "Missing edge or shift variation")
    # Hand-built family members beyond the sampler's maximum side length of 12.
    # The solver must use the visible rectangles, not the sampling vocabulary.
    unusual = [(8, 10, 6, 11), (7, 0, 15, 5), (0, 2, 4, 22)]  # capacities 36,39,40
    canvas = np.full((30, 30), 2, dtype=np.int8)
    slots = []
    for r, c, h, w in unusual:
        canvas[r:r + h, c:c + w] = 0
        canvas[r + 1:r + h - 1, c + 1:c + w - 1] = 2
        slots.extend((rr, cc) for rr in range(r + 1, r + h - 1)
                     for cc in range(c + 1, c + w - 1))
    for n in (1, 36, 37, 75, 114):
        a = canvas.copy()
        for r, c in slots[-n:]:
            a[r, c] = 9
        b = solve(a)
        _require(np.count_nonzero(b == 9) == n, "Unseen geometry lost ink")
        _require(np.array_equal(solve(b), b), "Canonical packing must be idempotent")
        left = n
        for r, c, h, w in unusual:
            take = min(left, (h - 2) * (w - 2))
            _require(np.count_nonzero(b[r + 1:r + h - 1, c + 1:c + w - 1] == 9) == take,
                     "Unseen geometry used the wrong capacity")
            left -= take
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "Count request not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "geometry_sets": len(geometry),
            "capacities": len(capacities), "quantities": len(quantities),
            "palettes": len(palettes), "bin_counts": sorted(bins), "shifts": sorted(shift_modes)}


if __name__ == "__main__":
    print(self_test())
