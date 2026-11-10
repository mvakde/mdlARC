"""Corner-keyed square packing followed by adjacent full-vector contraction.

Only Python/NumPy are needed. Generation constructs cuts, not parent puzzles;
solve reconstructs everything from the visible input. No persistent caches,
trace files, or generated data files are used by this module.
"""
from __future__ import annotations

import hashlib
import math
import numbers

import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                 (num_test, 1, 4, "num_test")):
        if not _integer(value) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _run_starts(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]


def _compress(a):
    # Both tests use the full assembled vectors. Retaining representatives
    # of equal rows cannot change whether two columns are equal (and vice versa).
    return a[np.ix_(_run_starts(a, 0), _run_starts(a, 1))].copy()


def _runs(a, axis):
    starts = _run_starts(a, axis)
    return tuple(int(x) for x in np.diff(np.r_[starts, a.shape[axis]]))


def _separated_repeat(a, axis):
    vectors = _compress(a) if axis == 0 else _compress(a).T
    seen = set()
    for vector in vectors:
        key = vector.tobytes()
        if key in seen:
            return True
        seen.add(key)
    return False


def _components(mask):
    height, width = mask.shape
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if 0 <= p[0] < height and 0 <= p[1] < width and p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(np.asarray(sorted(cells), dtype=np.int16))
    return result


def _normalize(coords):
    a = np.asarray(coords, dtype=np.int16)
    a = a - a.min(axis=0)
    return tuple(sorted(map(tuple, a.tolist())))


def _rotate(shape, turns):
    a = np.asarray(shape, dtype=np.int16)
    for _ in range(turns % 4):
        a = np.column_stack((a[:, 1], -a[:, 0]))
    return _normalize(a)


def _dimensions(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _mask(shape, side, offset=(0, 0)):
    bits = 0
    for r, c in shape:
        bits |= 1 << ((r + offset[0]) * side + c + offset[1])
    return bits


def _pack(shapes, side, limit=2):
    """Exhaust all corner placements, deduplicating identical colored squares.

    A corner fixes the bounding-box translation. Thus at most 6*4**3
    arrangements remain after fixing the unrotated keyed anchor at TL.
    Symmetric turns and permutations of identical pieces do not create ties.
    """
    if len(shapes) != 4 or sum(map(len, shapes)) != side * side:
        return []
    corners = (0, side - 1, side * (side - 1), side * side - 1)
    anchor = shapes[0]
    if (0, 0) not in anchor or max(_dimensions(anchor)) > side:
        return []
    amask = _mask(anchor, side)
    if tuple(bool(amask & (1 << x)) for x in corners) != (True, False, False, False):
        return []
    table = {}
    for piece in range(1, 4):
        for corner in range(1, 4):
            candidates = {}
            for turns in range(4):
                shape = _rotate(shapes[piece], turns)
                h, w = _dimensions(shape)
                if h > side or w > side:
                    continue
                offset = (0 if corner < 2 else side - h,
                          0 if corner % 2 == 0 else side - w)
                bits = _mask(shape, side, offset)
                if sum(bool(bits & (1 << x)) for x in corners) == 1 and bits & (1 << corners[corner]):
                    candidates.setdefault(bits, turns)
            table[piece, corner] = tuple(candidates.items())
    full = (1 << (side * side)) - 1
    answers = {}

    def visit(corner, remaining, occupied, masks, assignment, rotations):
        if len(answers) >= limit:
            return
        if corner == 4:
            if occupied == full:
                score = (sum(k != 0 for k in rotations), rotations)
                old = answers.get(masks)
                if old is None or score < old[0]:
                    answers[masks] = (score, {"masks": masks, "pieces": assignment,
                                             "turns": rotations})
            return
        for piece in remaining:
            for bits, turns in table[piece, corner]:
                if not bits & occupied:
                    visit(corner + 1, tuple(p for p in remaining if p != piece),
                          occupied | bits, masks + (bits,), assignment + (piece,),
                          rotations + (turns,))

    visit(1, (1, 2, 3), amask, (amask,), (0,), (0,))
    return [entry[1] for entry in answers.values()]


def _owner_from_solution(solution, side):
    result = np.full((side, side), -1, dtype=np.int8)
    for corner, bits in enumerate(solution["masks"]):
        for index in range(side * side):
            if bits & (1 << index):
                result[index // side, index % side] = corner
    _require(bool(np.all(result >= 0)), "packing has a gap")
    return result


def _parse(a):
    border = np.r_[a[0], a[-1], a[1:-1, 0], a[1:-1, -1]]
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(int(np.sum(border == bg)) * 2 > len(border), "background lacks a perimeter majority")
    comps = _components(a != bg)
    _require(len(comps) == 4, "expected four connected foreground pieces")
    anchors, plain = [], []
    for coords in comps:
        values = a[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) == 1:
            plain.append((coords, int(colors[0])))
            continue
        _require(len(colors) == 5 and int(np.sum(counts == 1)) == 4 and int(np.sum(counts > 1)) == 1,
                 "keyed anchor must contain four singleton key colors and a carrier")
        points = coords[np.isin(values, colors[counts == 1])]
        r, c = map(int, points.min(axis=0))
        _require(set(map(tuple, points)) == {(r, c), (r, c+1), (r+1, c), (r+1, c+1)},
                 "key must be an ordered 2x2 patch")
        location = (r - int(coords[:, 0].min()), c - int(coords[:, 1].min()))
        anchors.append((coords, int(colors[counts > 1][0]), a[r:r+2, c:c+2].copy(), location))
    _require(len(anchors) == 1 and len(plain) == 3, "expected one keyed anchor")
    coords, carrier, key, location = anchors[0]
    ordered = [coords] + [x[0] for x in plain]
    for i, component in enumerate(ordered):
        halo = {(int(r) + dr, int(c) + dc) for r, c in component
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        _require(all(not any(tuple(p) in halo for p in other)
                     for other in ordered[i+1:]), "pieces need a full background-cell gap")
    shapes = [_normalize(x) for x in ordered]
    total = sum(map(len, shapes))
    side = math.isqrt(total)
    _require(side * side == total, "piece area must be a square")
    _require(bg not in key and carrier not in key, "unreadable anchor palette")
    return {"side": side, "shapes": shapes, "key": key, "background": bg,
            "carriers": [carrier] + [x[1] for x in plain], "components": ordered,
            "key_location": location}


def _analyze(input_grid):
    a = _grid(input_grid)
    data = _parse(a)
    solutions = _pack(data["shapes"], data["side"])
    _require(len(solutions) == 1, "packing must have exactly one uncompressed colored square")
    data["solution"] = solutions[0]
    data["owner"] = _owner_from_solution(solutions[0], data["side"])
    data["square"] = data["key"].ravel()[data["owner"]]
    data["output"] = _compress(data["square"])
    return data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pack, recolor including legend cells, then contract equal vector runs."""
    return _analyze(input_grid)["output"].copy()


def _key_locations(owner):
    return [(r, c) for r in range(owner.shape[0] - 1)
            for c in range(owner.shape[1] - 1)
            if bool(np.all(owner[r:r+2, c:c+2] == 0))]


def _walk(rng, length):
    x = int(rng.integers(2, length - 1))
    values = []
    for _ in range(length):
        x = int(np.clip(x + rng.choice([-1, 0, 0, 1]), 1, length - 1))
        values.append(x)
    return values


def _base_partition(rng, size, kind):
    if kind == 0:
        # Two independent meandering seams. Their returning sections supply
        # equal but nonadjacent cross-sections, rather than a four-rectangle bank.
        hcut, vcut = _walk(rng, size), _walk(rng, size)
        r, c = np.indices((size, size))
        owner = (2 * (r >= np.asarray(hcut)[None, :])
                 + (c >= np.asarray(vcut)[:, None])).astype(np.int8)
    elif kind == 1:
        # One large remainder can wrap around three smaller corner pieces.
        large = int(rng.integers(4))
        owner = np.full((size, size), large, dtype=np.int8)
        for corner in rng.permutation([x for x in range(4) if x != large]):
            corner = int(corner)
            height = int(rng.integers(2, size - 1))
            width = int(rng.integers(1, size - 1))
            style = int(rng.integers(3))
            for depth in range(height):
                if style == 1:
                    width = max(1, width - int(rng.integers(2)))
                elif style == 2:
                    width = int(np.clip(width + rng.integers(-1, 2), 1, size - 2))
                w = max(2, width) if corner == 0 and depth < 2 else width
                for x in range(w):
                    rr = depth if corner < 2 else size - 1 - depth
                    cc = x if corner % 2 == 0 else size - 1 - x
                    if owner[rr, cc] != large:
                        return None
                    owner[rr, cc] = corner
    else:
        # Horizontal halves with independent, returning vertical cut profiles.
        split = int(rng.integers(2, size - 1))
        cuts = _walk(rng, size)
        owner = np.empty((size, size), dtype=np.int8)
        for r in range(size):
            left = 0 if r < split else 2
            owner[r, :cuts[r]], owner[r, cuts[r]:] = left, left + 1
        for _ in range(int(rng.integers(1, size + 1))):
            r, c = int(rng.integers(1, size - 1)), int(rng.integers(1, size - 1))
            dr, dc = _NEIGHBORS[int(rng.integers(4))]
            owner[r, c] = owner[r+dr, c+dc]
    return owner


def _expand_lengths(rng, count, side, concentrated):
    extra = side - count
    if concentrated and extra > 1:
        lengths = np.ones(count, dtype=int)
        target = int(rng.integers(count))
        bulk = int(rng.integers((extra + 1) // 2, extra + 1))
        lengths[target] += bulk
        lengths += rng.multinomial(extra - bulk, np.ones(count) / count)
    else:
        lengths = 1 + rng.multinomial(extra, np.ones(count) / count)
    return lengths


def _fallback_geometry(side):
    """A returning-seam construction, verified for every supported side.

    Bulk stretching the first row and column yields a key-bearing anchor,
    asymmetric pieces, separated repeats on both axes, and a unique packing.
    This is a final geometry fallback, never a stored input/output pair.
    """
    _require(9 <= side <= 13, "unsupported fallback side")
    r, c = np.indices((7, 7))
    hcut = np.asarray([3, 3, 4, 4, 4, 3, 4])
    vcut = np.asarray([4, 4, 4, 4, 3, 2, 3])
    core = _compress((2 * (r >= hcut[None, :]) + (c >= vcut[:, None])).astype(np.int8))
    rows = [side - core.shape[0] + 1] + [1] * (core.shape[0] - 1)
    cols = [side - core.shape[1] + 1] + [1] * (core.shape[1] - 1)
    owner = np.repeat(np.repeat(core, rows, axis=0), cols, axis=1)
    return owner, [_normalize(np.argwhere(owner == i)) for i in range(4)]


def _geometry(rng, side, demonstration, concentrated, attempt_limit=360):
    for _ in range(attempt_limit):
        size = int(rng.integers(5, min(9, side - 1) + 1))
        kind = 0 if demonstration else int(rng.choice(3, p=[0.45, 0.25, 0.30]))
        owner = _base_partition(rng, size, kind)
        if owner is None:
            continue
        owner = _compress(owner)
        if min(owner.shape) < 3 or max(owner.shape) >= side:
            continue
        # Connectivity is checked after expansion as well as through packing.
        if any(len(_components(owner == i)) != 1 for i in range(4)):
            continue
        if demonstration and not all(_separated_repeat(owner, axis) for axis in (0, 1)):
            continue
        if demonstration and owner.shape[0] == owner.shape[1]:
            continue
        rows = _expand_lengths(rng, owner.shape[0], side, concentrated)
        cols = _expand_lengths(rng, owner.shape[1], side, concentrated)
        owner = np.repeat(np.repeat(owner, rows, axis=0), cols, axis=1)
        shapes = [_normalize(np.argwhere(owner == i)) for i in range(4)]
        if min(map(len, shapes)) < 3 or len(shapes[0]) < 6 or not _key_locations(owner):
            continue
        if tuple(int(owner[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))) != (0, 1, 2, 3):
            continue
        if demonstration:
            orbit_sizes = [len({_rotate(s, k) for k in range(4)}) for s in shapes]
            if orbit_sizes != [4, 4, 4, 4]:
                continue
            if len({min(_rotate(s, k) for k in range(4)) for s in shapes}) != 4:
                continue
        solutions = _pack(shapes, side)
        if len(solutions) == 1 and np.array_equal(_owner_from_solution(solutions[0], side), owner):
            return owner, shapes
    return _fallback_geometry(side)


def _scatter(rng, shapes):
    dims = [_dimensions(s) for s in shapes]
    # Sample shelf groupings and both axes; their footprint ordering is random.
    # 2+2 with gap=1 always fits (side<=13), including a blank border.
    for attempt in range(40):
        transpose = bool(rng.integers(2))
        sizes = [(w, h) if transpose else (h, w) for h, w in dims]
        order = [int(i) for i in rng.permutation(4)]
        cuts = int(rng.integers(8)) if attempt < 39 else 2
        gap = int(rng.integers(1, 4)) if attempt < 39 else 1
        groups, group = [], []
        for j, i in enumerate(order):
            group.append(i)
            if j == 3 or cuts & (1 << j):
                groups.append(group)
                group = []
        heights = [max(sizes[i][0] for i in group) for group in groups]
        widths = [sum(sizes[i][1] for i in group) + gap * (len(group) - 1) for group in groups]
        extent = (sum(heights) + gap * (len(groups) - 1), max(widths))
        if max(extent) > 28:
            continue
        offsets, r = [None] * 4, 0
        for group, height, width in zip(groups, heights, widths):
            c = int(rng.integers(extent[1] - width + 1))
            for i in group:
                rr = r + int(rng.integers(height - sizes[i][0] + 1))
                offsets[i] = (rr, c)
                c += sizes[i][1] + gap
            r += height + gap
        pads = []
        for length in extent:
            room = 30 - length
            # Full blank perimeter makes arbitrary palettes locally readable.
            before = int(rng.integers(1, min(4, room)))
            after = int(rng.integers(1, min(5, room - before + 1)))
            pads.append((before, after))
        canvas = tuple(extent[a] + sum(pads[a]) for a in (0, 1))
        offsets = [(r + pads[0][0], c + pads[1][0]) for r, c in offsets]
        if transpose:
            canvas = canvas[::-1]
            offsets = [(c, r) for r, c in offsets]
        return offsets, canvas
    raise AssertionError("guaranteed two-shelf layout did not fit")


def _make_pair(rng, side, first, second, rotation_free, concentrated, key):
    owner, shapes = _geometry(rng, side, first, concentrated)
    if rotation_free or second:
        turns = [0, 0, 0, 0]
    elif first:
        turns = [0] + [int(x) for x in rng.permutation([1, 2, 3])]
    else:
        turns = [0, 0, 0, 0]
        chosen = rng.choice([1, 2, 3], size=int(rng.integers(4)), replace=False)
        repeat = int(rng.integers(1, 4))
        for i in chosen:
            turns[int(i)] = repeat if rng.random() < 0.5 else int(rng.integers(1, 4))
    oriented = [_rotate(s, k) for s, k in zip(shapes, turns)]
    offsets, canvas = _scatter(rng, oriented)
    available = [c for c in range(10) if c not in key]
    bg, anchor_carrier = [int(c) for c in rng.choice(available, 2, replace=False)]
    carriers = [anchor_carrier]
    for _ in range(3):
        carriers.append(anchor_carrier if rng.random() < 0.45 else int(rng.choice([c for c in range(10) if c != bg])))
    grid = np.full(canvas, bg, dtype=np.int8)
    for shape, offset, color in zip(oriented, offsets, carriers):
        coords = np.asarray(shape) + offset
        grid[coords[:, 0], coords[:, 1]] = color
    kr, kc = _key_locations(owner)[int(rng.integers(len(_key_locations(owner))))]
    r, c = offsets[0]
    grid[r+kr:r+kr+2, c+kc:c+kc+2] = key.reshape(2, 2)
    return {"input": grid, "output": _compress(key[owner])}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    rotation_free = bool(rng.random() < 0.20)
    fixed_side = int(rng.integers(10, 14))
    first_key = rng.choice(10, 4, replace=False).astype(np.int8)
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train + num_test):
        train = i < num_train
        if regime == 0:
            side = int(rng.integers(9, 12) if train else rng.integers(12, 14))
        elif regime == 1:
            side = int(rng.integers(11, 14) if train else rng.integers(9, 11))
        else:
            side = fixed_side
        if i == 0:
            key = first_key.copy()
        elif i == 1:
            key = np.roll(first_key, 1).copy()
        elif rng.random() < 0.25:
            key = first_key.copy()
        else:
            key = rng.choice(10, 4, replace=False).astype(np.int8)
        for _ in range(12):
            pair = _make_pair(rng, side, i == 0, i == 1, rotation_free,
                              (not train) or (i > 1 and rng.random() < 0.35), key)
            signature = _grid_key(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle["train" if train else "test"].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned pixels, unique packing, contraction and teaching cases.

    These are family-owned checks, not an independent semantic certificate.
    Generation deliberately uses two-axis shrinkage; solve also handles a
    no-op axis, symmetric pieces, and other uniquely packable square sizes.
    """
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "illegal pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            source, output = _grid(pair["input"]), _grid(pair["output"])
            signature = _grid_key(source)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            a = _analyze(source)
            _require(np.array_equal(a["output"], output), "incorrect output")
            _require(output.shape[0] < a["side"] and output.shape[1] < a["side"], "both axes must compress")
            _require(len(np.unique(output)) == 4, "all corner colors must survive")
            _require(np.array_equal(_compress(output), output), "contraction must be idempotent")
            # Exact representative selection preserves values at all junctions.
            rs, cs = _run_starts(a["square"], 0), _run_starts(a["square"], 1)
            _require(np.array_equal(a["square"][np.ix_(rs, cs)], output), "retained values changed")
            _require(np.array_equal(np.repeat(np.repeat(output, _runs(a["square"], 0), axis=0),
                                              _runs(a["square"], 1), axis=1), a["square"]), "runs do not reconstruct square")
            # Parent ablations: just B, just A, and packing without the key.
            _require(not np.array_equal(a["square"], output), "A is dispensable")
            _require(not np.array_equal(_compress(source), output), "B is dispensable")
            colors = np.asarray([a["carriers"][j] for j in a["solution"]["pieces"]], dtype=np.int8)
            _require(not np.array_equal(_compress(colors[a["owner"]]), output), "key coloring is dispensable")
            analyses[split].append(a)
    first = analyses["train"][0]
    _require(all(_separated_repeat(first["square"], axis) for axis in (0, 1)), "training must retain separated repeated vectors on both axes")
    _require(first["output"].shape[0] != first["output"].shape[1], "training must distinguish independent axes from square scaling")
    _require(not np.array_equal(first["key"], analyses["train"][1]["key"]), "key assignment must vary in training")
    _require(all(len({_rotate(s, k) for k in range(4)}) == 4 for s in first["shapes"]),
             "first demonstration must use asymmetric footprints")
    _require(len({min(_rotate(s, k) for k in range(4)) for s in first["shapes"]}) == 4,
             "first demonstration must use noncongruent footprints")
    demonstrated = {k for a in analyses["train"] for k in a["solution"]["turns"]}
    _require(all(set(a["solution"]["turns"]) <= demonstrated for a in analyses["test"]),
             "test rotations lack training evidence")
    # Explicitly reject the main simpler rules on the teaching example.
    unique_rows = np.unique(first["square"], axis=0, return_index=True)[1]
    unique_cols = np.unique(first["square"], axis=1, return_index=True)[1]
    global_dedup = first["square"][np.ix_(np.sort(unique_rows), np.sort(unique_cols))]
    _require(not np.array_equal(global_dedup, first["output"]),
             "global duplicate removal must disagree with the demonstration")
    _require(not np.array_equal(first["square"][_run_starts(first["square"], 0)], first["output"]),
             "row-only compression must disagree")
    _require(not np.array_equal(first["square"][:, _run_starts(first["square"], 1)], first["output"]),
             "column-only compression must disagree")
    uncleared = first["square"].copy()
    r, c = first["key_location"]
    uncleared[r:r+2, c:c+2] = first["key"]
    _require(not np.array_equal(_compress(uncleared), first["output"]), "training must show overwriting the key before compression")


def self_test():
    """Bounded 42-seed sweep; metrics are small sets, never saved grid traces."""
    for side in range(9, 14):
        owner, shapes = _fallback_geometry(side)
        packed = _pack(shapes, side)
        _require(len(packed) == 1 and np.array_equal(_owner_from_solution(packed[0], side), owner),
                 "fallback packing not unique")
        _require(all(_separated_repeat(owner, axis) for axis in (0, 1)), "fallback loses separated repeats")
        _require(all(len({_rotate(s, k) for k in range(4)}) == 4 for s in shapes), "fallback piece symmetric")
        _require(len({min(_rotate(s, k) for k in range(4)) for s in shapes}) == 4, "fallback pieces congruent")
        _require(bool(_key_locations(owner)), "fallback has no key site")
    signatures, canvases, outputs, cuts, backgrounds, turns = set(), set(), set(), set(), set(), set()
    sides, area_profiles, run_profiles, carrier_counts, key_sites = set(), set(), set(), set(), set()
    rotation_demands = {"train": set(), "test": set()}
    size_shifts, varied_cuts = set(), 0
    largest_ratio, third_edge_pairs = 0.0, 0
    separated = [0, 0]
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        local_cuts, local_sides = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                for role in ("input", "output"):
                    grid = pair[role]
                    digest.update(bytes(grid.shape))
                    digest.update(grid.tobytes())
                canvases.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                cuts.add(_grid_key(_compress(a["owner"])))
                local_cuts.add(_grid_key(_compress(a["owner"])))
                backgrounds.add(a["background"])
                turns.update(a["solution"]["turns"])
                sides.add(a["side"])
                local_sides[split].append(a["side"])
                areas = tuple(int(np.sum(a["owner"] == j)) for j in range(4))
                area_profiles.add(areas)
                largest_ratio = max(largest_ratio, max(areas) / min(areas))
                run_profiles.add((_runs(a["square"], 0), _runs(a["square"], 1)))
                carrier_counts.add(len(set(a["carriers"])))
                key_sites.add(a["key_location"])
                rotation_demands[split].add(sum(k != 0 for k in a["solution"]["turns"]))
                third_edge_pairs += int(any(sum(bool(np.any(edge == j)) for edge in
                                               (a["owner"][0], a["owner"][-1],
                                                a["owner"][:, 0], a["owner"][:, -1])) >= 3
                                            for j in range(4)))
                for axis in (0, 1):
                    separated[axis] += int(_separated_repeat(a["square"], axis))
                pairs += 1
        signatures.add(digest.digest())
        varied_cuts += int(len(local_cuts) > 1)
        train_sides = local_sides["train"]
        for side in local_sides["test"]:
            size_shifts.add("larger" if side > max(train_sides) else
                            "smaller" if side < min(train_sides) else "equal")
    _require(len(signatures) == 42, "seeds repeat puzzles")
    _require(len(canvases) >= 35 and len(outputs) >= 15, "dimension variation insufficient")
    _require(len(cuts) >= 100 and len(backgrounds) == 10, "geometry or palette variation insufficient")
    _require(turns == {0, 1, 2, 3} and min(separated) >= 42, "motion or repeat evidence absent")
    _require(sides == set(range(9, 14)) and size_shifts == {"larger", "smaller", "equal"},
             "assembly-size shifts missing")
    _require(len(area_profiles) >= 100 and len(run_profiles) >= 100 and varied_cuts == 42,
             "piece-area, run-length or within-puzzle geometry variation missing")
    _require(carrier_counts == {1, 2, 3, 4} and len(key_sites) >= 15,
             "carrier sharing or embedded-key variation missing")
    _require(largest_ratio >= 5 and third_edge_pairs >= 5, "unequal and wrapping shapes missing")
    _require(all(demands == {0, 1, 2, 3} for demands in rotation_demands.values()),
             "rotation demands must vary in each split")
    a, b = generate_puzzle(917, 2, 1), generate_puzzle(917, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for split in a for x, y in zip(a[split], b[split])
                 for k in ("input", "output")), "determinism failed")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1),
                   (None, 1), (4, None), (4, False), (4, 1.0)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(canvases),
            "output_shapes": len(outputs), "normalized_cuts": len(cuts),
            "backgrounds": len(backgrounds), "separated_repeat_pairs_by_axis": separated,
            "assembly_sides": sorted(sides), "area_profiles": len(area_profiles),
            "run_profiles": len(run_profiles), "carrier_counts": sorted(carrier_counts),
            "key_locations": len(key_sites), "size_shifts": sorted(size_shifts),
            "rotation_demands": {s: sorted(v) for s, v in rotation_demands.items()},
            "largest_area_ratio": round(largest_ratio, 3), "third_edge_pairs": third_edge_pairs,
            "puzzles_with_varied_cuts": varied_cuts}
