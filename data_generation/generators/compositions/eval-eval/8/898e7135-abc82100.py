"""Cavity-guided printing: scale, fit, register, and aperture-clip.

Only standard-library modules and NumPy are used. No parent code is imported.
The decoder searches roles, scale, and registered fits using the input alone.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(int, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(frozenset((int(r), int(c)) for r, c in group))
    return groups


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _norm(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _transform(cells, turn, mirror=False):
    result = [(r, -c if mirror else c) for r, c in cells]
    for _ in range(turn):
        result = [(c, -r) for r, c in result]
    return tuple(sorted(result))


def _variants(cells):
    return tuple(sorted({_norm(_transform(cells, t, m))
                         for m in (False, True) for t in range(4)}))


def _canonical(cells):
    return _variants(cells)[0]


def _size(cells):
    r0, r1, c0, c1 = _bounds(cells)
    return r1-r0, c1-c0


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _macro(cells, k):
    """Decode a whole monochromatic component, relative to its own box."""
    r0, r1, c0, c1 = _bounds(cells)
    if (r1-r0) % k or (c1-c0) % k or len(cells) % (k*k):
        return None
    blocks = {}
    for r, c in cells:
        p = (r-r0)//k, (c-c0)//k
        blocks[p] = blocks.get(p, 0) + 1
    if any(n != k*k for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _near(q, cells):
    return min(max(abs(q[0]-r), abs(q[1]-c)) for r, c in cells) <= 1


def _external_records(grid, board_bounds, bg, frame, neutral, k):
    """Decode complete block labels and exact masks, consuming every large object."""
    r0, r1, c0, c1 = board_bounds
    outside = np.ones(grid.shape, bool)
    outside[r0:r1, c0:c1] = False
    objects = [(color, cells) for color in range(10) if color != bg
               for cells in _components((grid == color) & outside)]
    if any(color == frame for color, cells in objects):
        return None
    large = [(color, cells) for color, cells in objects if len(cells) >= 4]
    masks = [i for i, (color, cells) in enumerate(large) if color == neutral]
    if not masks or len(large) != 3*len(masks) or len(masks) > 3:
        return None
    squares = {}
    for i, (color, cells) in enumerate(large):
        a, b, c, d = _bounds(cells)
        if color != neutral and (b-a, d-c) == (k, k) and len(cells) == k*k:
            squares[(a, c)] = (i, color)
    records, used = [], set(masks)
    for i in masks:
        shape = _macro(large[i][1], k)
        if shape is None or not 1 <= len(shape) <= 12:
            return None
        a, b, c, d = _bounds(large[i][1])
        options = []
        for (sr, sc), (si, source) in squares.items():
            for dr, dc in _DIRS:
                target_block = squares.get((sr+k*dr, sc+k*dc))
                if target_block is None:
                    continue
                ti, target = target_block
                if target == source:
                    continue
                qr, qc = sr+2*k*dr, sc+2*k*dc
                if (a-qr) % k or (c-qc) % k:
                    continue
                q = (qr-a)//k, (qc-c)//k
                if not _near(q, shape):
                    continue
                if not (0 <= qr <= grid.shape[0]-k and 0 <= qc <= grid.shape[1]-k):
                    continue
                reg_block = grid[qr:qr+k, qc:qc+k]
                if not (np.all(reg_block == bg) or np.all(reg_block == neutral)):
                    continue
                offsets = tuple(sorted((r-q[0], c-q[1]) for r, c in shape))
                options.append({"source": source, "target": target, "offsets": offsets,
                                "shape": shape, "q": q, "direction": (dr, dc),
                                "labels": (si, ti)})
        # Local label order itself is uniquely readable, including empty q.
        if len(options) != 1 or used.intersection(options[0]["labels"]):
            return None
        records.append(options[0])
        used.update(options[0]["labels"])
    if len(used) != len(large):
        return None
    return records


def _registered_fits(offsets, cavity_shape):
    return tuple(sorted({_transform(offsets, t, m)
                         for m in (False, True) for t in range(4)
                         if _norm(_transform(offsets, t, m)) == cavity_shape}))


def _impression(offsets, anchors, aperture):
    return frozenset((r+dr, c+dc) for r, c in anchors for dr, dc in offsets
                     if (r+dr, c+dc) in aperture)


def _interpret_board(board, bg, frame, records):
    cavities = _components(board != frame)
    if not cavities:
        return None
    logical = board.copy()
    logical[logical != frame] = bg
    infos, used = [], set()
    for cells in cavities:
        source_colors = {int(board[p]) for p in cells if int(board[p]) != bg}
        if len(source_colors) != 1:
            return None
        source = next(iter(source_colors))
        anchors = frozenset(p for p in cells if int(board[p]) == source)
        shape = _norm(cells)
        candidates = []
        for i, rec in enumerate(records):
            if rec["source"] != source:
                continue
            fits = _registered_fits(rec["offsets"], shape)
            for offsets in fits:
                paint = _impression(offsets, anchors, cells)
                candidates.append((i, rec, offsets, paint))
        # Reject empty and ambiguous results. Symmetric fits may be interchangeable.
        if not candidates or not candidates[0][3] or len({(x[1]["target"], x[3]) for x in candidates}) != 1:
            return None
        i, rec, offsets, paint = candidates[0]
        used.update(x[0] for x in candidates)
        for p in paint:
            logical[p] = rec["target"]
        visits = [set(_impression(offsets, [a], cells)) for a in anchors]
        multi_effect = len(anchors) > 1 and all(set(paint) != v for v in visits)
        rotations = {_norm(_transform(rec["offsets"], t)) for t in range(4)}
        infos.append({"source": source, "target": rec["target"], "shape": shape,
                      "cells": cells, "anchors": anchors, "offsets": offsets,
                      "paint": paint, "partial": len(paint) < len(cells),
                      "raw_difference": _impression(rec["offsets"], anchors, cells) != paint,
                      "empty_registration": (0, 0) not in rec["offsets"],
                      "reflection": shape not in rotations,
                      "multi_effect": multi_effect,
                      "overlap": sum(map(len, visits)) > len(paint)})
    if used != set(range(len(records))):
        return None
    return logical, infos


def _recover(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > grid.size//2, "background must occupy a strict majority")
    solutions = []
    for frame in range(10):
        if frame == bg:
            continue
        for cells in _components(grid == frame):
            r0, r1, c0, c1 = _bounds(cells)
            board = grid[r0:r1, c0:c1]
            if min(board.shape) < 4:
                continue
            if not (np.all(board[0] == frame) and np.all(board[-1] == frame)
                    and np.all(board[:, 0] == frame) and np.all(board[:, -1] == frame)):
                continue
            bounds = r0, r1, c0, c1
            for k in range(2, min(4, 30//max(board.shape))+1):
                for neutral in range(10):
                    if neutral in (bg, frame):
                        continue
                    records = _external_records(grid, bounds, bg, frame, neutral, k)
                    if records is None:
                        continue
                    interpretation = _interpret_board(board, bg, frame, records)
                    if interpretation is None:
                        continue
                    logical, cavities = interpretation
                    output = np.repeat(np.repeat(logical, k, 0), k, 1)
                    info = {"background": bg, "frame": frame, "neutral": neutral,
                            "scale": k, "board": board, "bounds": bounds,
                            "logical": logical, "records": records, "cavities": cavities}
                    solutions.append((output, info))
    _require(len(solutions) == 1, "input must identify one board, scale and instruction reading")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this grid only; no demonstrations or construction state."""
    return _recover(input_grid)[0]


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _grow(rng, area, span):
    cells = {(0, 0)}
    style = int(rng.integers(3))
    while len(cells) < area:
        candidates = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS} - cells)
        candidates = [p for p in candidates if max(_size(cells | {p})) <= span]
        if not candidates:
            return None
        weights = []
        for r, c in candidates:
            degree = sum((r+dr, c+dc) in cells for dr, dc in _DIRS)
            h, w = _size(cells | {(r, c)})
            weights.append(3**degree if style == 0 else max(h, w)/min(h, w)
                           if style == 1 else 1/degree)
        weights = np.asarray(weights, float)
        p = int(rng.choice(len(candidates), p=weights/weights.sum()))
        cells.add(candidates[p])
    return _norm(cells)


@lru_cache(maxsize=1)
def _compact_shapes():
    """All 31 connected D4 classes of area <=7 fitting a 3-by-3 box.

    A bounded fallback for novel test geometry, rather than a stencil catalog
    used by the solver. Normal sampling grows shapes procedurally.
    """
    cells = [(r, c) for r in range(3) for c in range(3)]
    result = set()
    for area in range(1, 8):
        for subset in combinations(cells, area):
            a = np.zeros((3, 3), int)
            for p in subset:
                a[p] = 1
            if len(_components(a == 1)) == 1:
                result.add(_canonical(subset))
    return tuple(sorted(result))


def _registration_options(shape, wanted=None):
    near = {(r+dr, c+dc) for r, c in shape for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    cells, options = set(shape), []
    for q in sorted(near):
        occupied = q in cells
        if wanted == "occupied" and not occupied or wanted in ("hole", "interior_hole") and occupied:
            continue
        if wanted == "interior_hole":
            h, w = _size(shape)
            if not (0 < q[0] < h-1 and 0 < q[1] < w-1):
                continue
        for dr, dc in _DIRS:
            labels = [(q[0]-j*dr, q[1]-j*dc) for j in (1, 2)]
            # Reversing the two labels must not point back toward this mask.
            reverse = q[0]-3*dr, q[1]-3*dc
            if any(p in cells for p in labels) or _near(reverse, shape):
                continue
            options.append((q, (dr, dc)))
    return options


def _die(rng, shape, source, target, k, wanted=None):
    options = _registration_options(shape, wanted)
    if not options:
        return None
    q, direction = _choose(rng, options)
    offsets = tuple(sorted((r-q[0], c-q[1]) for r, c in shape))
    turn, mirror = int(rng.integers(4)), bool(rng.integers(2))
    offsets = _transform(offsets, turn, mirror)
    dr, dc = _transform([direction], turn, mirror)[0]
    locations = set(offsets) | {(0, 0), (-dr, -dc), (-2*dr, -2*dc)}
    r0, r1, c0, c1 = _bounds(locations)
    logical = np.full((r1-r0, c1-c0), -1, int)
    for r, c in offsets:
        logical[r-r0, c-c0] = -2
    logical[-dr-r0, -dc-c0] = target
    logical[-2*dr-r0, -2*dc-c0] = source
    return {"source": source, "target": target, "offsets": offsets,
            "shape": _norm(offsets), "patch": logical, "scale": k}


def _anchor_choice(rng, cells, offsets, force_partial=False, force_multi=False):
    all_fits = _registered_fits(offsets, _norm(cells))
    cells = frozenset(cells)
    ordered = sorted(cells)
    for attempt in range(60):
        n = 1 if force_partial else int(rng.integers(2 if force_multi else 1, min(5, len(cells))+1))
        anchors = frozenset(ordered[int(i)] for i in rng.choice(len(ordered), n, replace=False))
        paints = {_impression(f, anchors, cells) for f in all_fits}
        if len(paints) != 1:
            continue
        paint = next(iter(paints))
        if not paint or force_partial and len(paint) == len(cells):
            continue
        if force_multi and any(_impression(all_fits[0], [a], cells) == paint for a in anchors):
            continue
        return anchors, paint
    # Symmetric occupied registrations can use all pixels as anchors, yielding
    # the same complete impression under every symmetry. Teaching cases never
    # use this fallback, because they must show displacement or multiplicity.
    if not force_partial and not force_multi and (0, 0) in offsets:
        return cells, cells
    return None


def _board(rng, dies, k, bg, frame, teaching):
    limit = 30//k
    n = len(dies)
    extra = 0 if teaching == 0 else 1 if teaching == 1 else int(rng.integers(0, 3))
    indexes = list(range(n)) + [int(rng.integers(n)) for _ in range(extra)]
    shapes, anchor_patterns = [], []
    for j, index in enumerate(indexes):
        variants = _variants(dies[index]["shape"])
        if teaching in (0, 1) and j == 0:
            rotations = {_norm(_transform(dies[index]["shape"], t)) for t in range(4)}
            variants = tuple(v for v in variants if v not in rotations)
        shape = _choose(rng, variants)
        selected = _anchor_choice(rng, shape, dies[index]["offsets"],
                                  force_partial=teaching == 0 and j == 0,
                                  force_multi=teaching == 0 and j == 1)
        if selected is None:
            return None
        shapes.append(shape)
        anchor_patterns.append(selected)
    for attempt in range(30):
        low = 6 if n == 1 else 7
        h, w = int(rng.integers(low, limit+1)), int(rng.integers(low, limit+1))
        if rng.random() < .2:
            w = h
        grid = np.full((h, w), frame, int)
        answer = grid.copy()
        occupied = set()
        success = True
        order = sorted(range(len(indexes)), key=lambda j: -len(shapes[j]))
        for j in order:
            sh, sw = _size(shapes[j])
            origins = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            rng.shuffle(origins)
            for r0, c0 in origins:
                cells = frozenset((r+r0, c+c0) for r, c in shapes[j])
                if any(p in occupied or any((p[0]+dr, p[1]+dc) in occupied for dr, dc in _DIRS)
                       for p in cells):
                    continue
                die = dies[indexes[j]]
                relative_anchors, relative_paint = anchor_patterns[j]
                anchors = {(r+r0, c+c0) for r, c in relative_anchors}
                paint = {(r+r0, c+c0) for r, c in relative_paint}
                for p in cells:
                    grid[p] = answer[p] = bg
                for p in anchors:
                    grid[p] = die["source"]
                for p in paint:
                    answer[p] = die["target"]
                occupied.update(cells)
                break
            else:
                success = False
                break
        if success:
            return grid, answer
    return None


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, dies, bg, neutral, noise_color):
    patches = [board]
    for die in dies:
        logical = die["patch"].copy()
        logical[logical == -1] = bg
        logical[logical == -2] = neutral
        k = die["scale"]
        patches.append(np.repeat(np.repeat(logical, k, 0), k, 1))
    for attempt in range(36):
        min_h = max(12, max(p.shape[0] for p in patches))
        min_w = max(12, max(p.shape[1] for p in patches))
        h, w = (int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))) if attempt < 12 else (30, 30)
        grid = np.full((h, w), bg, int)
        blocked = np.zeros((h, w), bool)
        order = sorted(range(len(patches)), key=lambda j: (-patches[j].size, float(rng.random())))
        for j in order:
            ph, pw = patches[j].shape
            origins = _free_origins(blocked, ph, pw)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            grid[r:r+ph, c:c+pw] = patches[j]
            blocked[max(0, r-1):r+ph+1, max(0, c-1):c+pw+1] = True
        else:
            # Optional singleton/domino noise, always below the smallest block.
            for _ in range(int(rng.integers(0, 9))):
                ph, pw = (1, 2) if rng.random() < .3 else (1, 1)
                origins = _free_origins(blocked, ph, pw)
                if not len(origins):
                    break
                r, c = map(int, origins[int(rng.integers(len(origins)))])
                grid[r:r+ph, c:c+pw] = noise_color
                blocked[max(0, r-1):r+ph+1, max(0, c-1):c+pw+1] = True
            if np.count_nonzero(grid == bg) <= grid.size//2:
                continue
            if rng.random() < .8:
                r0, r1, c0, c1 = _bounds(set(map(tuple, np.argwhere(grid != bg))))
                r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
                r1, c1 = min(h, r1+int(rng.integers(4))), min(w, c1+int(rng.integers(4)))
                crop = grid[r0:r1, c0:c1]
                if np.count_nonzero(crop == bg) > crop.size//2:
                    grid = crop.copy()
            return grid
    return None


def _make_pair(rng, k, palette, teaching=None, forbidden_classes=frozenset()):
    bg, frame, neutral, noise = map(int, palette[:4])
    colors = list(map(int, palette[4:]))
    for attempt in range(120):
        n = 2 if teaching is not None else int(rng.choice([1, 2] if k >= 4 else [1, 2, 2, 3]))
        span = 4 if k == 2 else 3
        shapes, wanted = [], []
        if teaching == 0:
            area = int(rng.integers(4, 7))
            first = _grow(rng, area, span)
            second = _grow(rng, area, span)
            if first is None or second is None or len(_variants(first)) != 8 or _canonical(first) == _canonical(second):
                continue
            shapes, wanted = [first, second], ["occupied", "occupied"]
            sources = [colors[0], colors[0]]
            targets = [colors[1], colors[2]]
        elif teaching == 1:
            # A chiral, open, notched outline whose interior registration is empty.
            w = 3 if k >= 3 else int(rng.integers(3, 5))
            h = 3 if k >= 3 else int(rng.integers(3, 5))
            shape = tuple(sorted({(0, c) for c in range(w)} | {(h-1, c) for c in range(1, w)}
                                 | {(r, w-1) for r in range(h)}))
            if len(_variants(shape)) != 8:
                continue
            shapes, wanted = [shape, shape], ["interior_hole", "interior_hole"]
            sources, targets = [colors[0], colors[3]], [colors[3], colors[0]]
        else:
            sources = [int(rng.choice(colors[:3])) for _ in range(n)]
            targets = [int(rng.choice([c for c in colors if c != s])) for s in sources]
            for i in range(n):
                area = int(rng.integers(1, (10 if k == 2 else 7)+1))
                if forbidden_classes and i == 0 and attempt >= 60:
                    shape = _choose(rng, [s for s in _compact_shapes() if s not in forbidden_classes])
                else:
                    shape = _grow(rng, area, span)
                if shape is None or any(sources[i] == sources[j] and _canonical(shape) == _canonical(shapes[j])
                                        for j in range(len(shapes))) or i == 0 and _canonical(shape) in forbidden_classes:
                    break
                shapes.append(shape)
                wanted.append(None)
            if len(shapes) != n:
                continue
        dies = [_die(rng, shape, s, t, k, kind) for shape, s, t, kind in zip(shapes, sources, targets, wanted)]
        if any(d is None for d in dies):
            continue
        board_result = _board(rng, dies, k, bg, frame, teaching)
        if board_result is None:
            continue
        board, logical = board_result
        inp = _pack(rng, board, dies, bg, neutral, noise)
        if inp is None:
            continue
        output = np.repeat(np.repeat(logical, k, 0), k, 1)
        try:
            recovered, info = _recover(inp)
        except ValueError:
            continue
        if not np.array_equal(recovered, output):
            continue
        if teaching == 0 and not (any(c["partial"] for c in info["cavities"])
                                  and any(c["multi_effect"] for c in info["cavities"])
                                  and any(c["raw_difference"] for c in info["cavities"])):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded pair construction exhausted")


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _schedule(rng, ntrain, ntest):
    mode = int(rng.integers(6))
    if mode < 2:
        k = mode+2
        return [k]*ntrain, [k]*ntest
    if mode < 4:
        a, b = (2, 3) if mode == 2 else (3, 2)
        return [a]*ntrain, [b]*ntest
    train = [2, 3] + [int(rng.choice([2, 3])) for _ in range(ntrain-2)]
    rng.shuffle(train)
    return train, [4 if mode == 5 else int(rng.choice([2, 3])) for _ in range(ntest)]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    train_scales, test_scales = _schedule(rng, num_train, num_test)
    puzzle, seen = {"train": [], "test": []}, set()
    training_classes = set()
    for split, scales in (("train", train_scales), ("test", test_scales)):
        for index, k in enumerate(scales):
            teaching = index if split == "train" and index < 2 else None
            for _ in range(12):
                pair = _make_pair(rng, k, palette, teaching,
                                  training_classes if split == "test" else frozenset())
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                training_classes.update(_canonical(r["shape"]) for r in _recover(pair["input"])[1]["records"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid counts")
    seen, training, testing, roles = set(), [], [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            predicted, info = _recover(inp)
            _require(np.array_equal(out, predicted), "incorrect input-derived impression")
            k, board, logical, frame = info["scale"], info["board"], info["logical"], info["frame"]
            _require(out.shape == tuple(k*n for n in board.shape), "wrong enlargement")
            _require(np.all(logical[board == frame] == frame), "structural cells changed")
            _require(all(len(c["anchors"]) >= 1 and c["paint"] <= c["cells"] for c in info["cavities"]),
                     "anchor or aperture invariant broken")
            _require(all(logical[p] == c["target"] if p in c["paint"] else logical[p] == info["background"]
                         for c in info["cavities"] for p in c["cells"]), "erasure/painting violated")
            roles.add((info["background"], frame, info["neutral"]))
            if split == "train":
                training.append(info)
            else:
                testing.append(info)
    _require(len(roles) == 1, "generated puzzle uses inconsistent role colors")
    all_cavities = [c for info in training for c in info["cavities"]]
    _require(any(c["partial"] for c in all_cavities), "no displacement/aperture teaching witness")
    _require(any(c["multi_effect"] for c in all_cavities), "no multiple-anchor witness")
    _require(any(c["empty_registration"] for c in all_cavities), "no empty-registration witness")
    _require(any(c["reflection"] for c in all_cavities), "no reflection witness")
    _require(any(c["raw_difference"] for c in training[0]["cavities"]),
             "no witness against un-oriented stamping")
    _require(any((0, 0) in r["offsets"] and
                 (sum(a for a, b in r["offsets"]) != 0 or sum(b for a, b in r["offsets"]) != 0)
                 for info in training for r in info["records"]), "no off-center registration witness")
    training_classes = {_canonical(r["shape"]) for info in training for r in info["records"]}
    _require(all(any(_canonical(r["shape"]) not in training_classes for r in info["records"])
                 for info in testing), "test lacks new silhouette geometry")
    records = training[0]["records"]
    _require(any(x["source"] == y["source"] and x["target"] != y["target"]
                 and len(x["shape"]) == len(y["shape"]) and _canonical(x["shape"]) != _canonical(y["shape"])
                 for x in records for y in records), "no shape-conditioned target witness")
    records = training[1]["records"]
    _require(len(records) == 2 and records[0]["source"] == records[1]["target"]
             and records[1]["source"] == records[0]["target"], "no simultaneous reciprocal witness")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _hand_regressions():
    """Small literal diagrams with independently specified logical answers."""
    def run(board_rows, card_rows, answer_rows, *, shape=(14, 14), card_at=(1, 9)):
        board = np.array([[int(c) for c in row] for row in board_rows])
        card = np.array([[int(c) for c in row] for row in card_rows])
        answer = np.array([[int(c) for c in row] for row in answer_rows])
        inp = np.zeros(shape, int)
        inp[1:1+board.shape[0], 1:1+board.shape[1]] = board
        card = np.repeat(np.repeat(card, 2, 0), 2, 1)
        r, c = card_at
        inp[r:r+card.shape[0], c:c+card.shape[1]] = card
        expected = np.repeat(np.repeat(answer, 2, 0), 2, 1)
        assert np.array_equal(solve(inp), expected)
        return inp

    # Off-center occupied registration: centering would paint all four cells.
    occupied = run(("44444", "40444", "41444", "40044", "44444"),
                   ("10", "20", "80", "80", "88"),
                   ("44444", "40444", "42444", "42044", "44444"))
    # Reflection of a chiral L is necessary, including its registered offsets.
    mirrored = run(("44444", "44144", "44044", "40044", "44444"),
                   ("10", "20", "80", "80", "88"),
                   ("44444", "44244", "44244", "42244", "44444"))
    assert _recover(mirrored)[1]["cavities"][0]["reflection"]
    # Empty registration: the original anchor itself becomes background.
    run(("44444", "40004", "44414", "44004", "44444"),
        ("0888", "1208", "0088"),
        ("44444", "40224", "44404", "44024", "44444"),
        shape=(16, 16), card_at=(8, 3))
    # All-cell anchors resolve symmetric fits, preserving an interior island.
    island = run(("4444444", "4444444", "4411144", "4414144", "4411144", "4444444", "4444444"),
                 ("100", "200", "888", "808", "888"),
                 ("4444444", "4444444", "4422244", "4424244", "4422244", "4444444", "4444444"),
                 shape=(18, 18), card_at=(1, 11))
    assert len(_components(_recover(island)[1]["board"] == 4)) == 2
    ambiguous = island.copy()
    dots = list(map(tuple, np.argwhere(ambiguous[:9, :9] == 1)))
    for p in dots[1:]:
        ambiguous[p] = 0
    broken = occupied.copy()
    broken[tuple(np.argwhere(broken == 8)[0])] = 0
    for invalid in (ambiguous, broken):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous registration or broken macrocell accepted")


def _recursive_counterfactual(info):
    """An explicitly unwanted second rewrite, used only for a negative check."""
    answer = info["logical"].copy()
    bg = info["background"]
    for cavity in info["cavities"]:
        source = cavity["target"]
        choices = [r for r in info["records"] if r["source"] == source
                   and _canonical(r["shape"]) == _canonical(cavity["shape"])]
        if not choices:
            continue
        rec = choices[0]
        anchors = [p for p in cavity["cells"] if int(info["logical"][p]) == source]
        offsets = _registered_fits(rec["offsets"], cavity["shape"])[0]
        paint = _impression(offsets, anchors, cavity["cells"])
        for p in anchors:
            answer[p] = bg
        for p in paint:
            answer[p] = rec["target"]
    return answer


def self_test():
    """Bounded 42-seed family audit plus count and negative regressions."""
    _hand_regressions()
    keys, shapes, boards, geometries, schedules, scales, counts, directions = (set() for _ in range(8))
    partials = overlaps = empty = reflections = islands = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        info = _recover(puzzle["train"][1]["input"])[1]
        assert not np.array_equal(_recursive_counterfactual(info), info["logical"])
        keys.add(_puzzle_key(puzzle))
        schedule = []
        for split in ("train", "test"):
            ks = set()
            for pair in puzzle[split]:
                predicted, info = _recover(pair["input"].copy())
                assert np.array_equal(predicted, pair["output"])
                shapes.add(pair["input"].shape)
                boards.add(info["board"].shape)
                ks.add(info["scale"])
                scales.add(info["scale"])
                counts.add(len(info["records"]))
                directions.update(r["direction"] for r in info["records"])
                geometries.update(_canonical(r["shape"]) for r in info["records"])
                islands += len(_components(info["board"] == info["frame"])) > 1
                for c in info["cavities"]:
                    partials += c["partial"]
                    overlaps += c["overlap"]
                    empty += c["empty_registration"]
                    reflections += c["reflection"]
            schedule.append(tuple(sorted(ks)))
        schedules.add(tuple(schedule))
    assert len(keys) == 42 and len(shapes) >= 30 and len(boards) >= 20
    assert len(geometries) >= 25 and len(schedules) >= 5
    assert scales >= {2, 3, 4} and counts == {1, 2, 3} and directions == set(_DIRS)
    assert min(partials, overlaps, empty, reflections) >= 42
    first = generate_puzzle(12345, 2, 1)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(12345, 2, 1))
    for a, b in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(678+a*10+b, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
