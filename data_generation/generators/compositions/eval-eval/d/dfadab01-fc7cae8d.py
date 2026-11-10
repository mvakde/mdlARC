"""Directed printing: the same visible frame controls membership and ink offsets.

Only standard library and NumPy are used. No parent module or stored example is
loaded. The decoder and solver operate entirely on a single input grid.
"""
from __future__ import annotations

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
# Source positive row/column axes expressed in the scene. Rotations are CCW.
MATRICES = (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
    ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0)),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be two-dimensional with sides 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _side(a, side):
    return {"T": a[0, :], "R": a[:, -1],
            "B": a[-1, :], "L": a[:, 0]}[side]


def _adjacent(sides):
    return len(sides) == 2 and any(
        set(sides) == {SIDES[i], SIDES[(i + 1) % 4]} for i in range(4))


def _components(a, bg, diagonal=True):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    objects = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or not dr or not dc)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        objects.append(sorted(cells))
    return objects


def _decode(input_grid):
    a = _grid(input_grid)
    H, W = a.shape
    _require(H >= 5 and W >= 5, "a readable key requires an interior")
    bg = int(a[0, 0])
    _require(all(int(a[q]) == bg for q in ((0, -1), (-1, 0), (-1, -1))),
             "all four corners must encode background")
    outer = {}
    for s in SIDES:
        values = np.unique(_side(a, s)[1:-1])
        if len(values) == 1 and int(values[0]) != bg:
            outer[s] = int(values[0])
        else:
            _require(bool(np.all(_side(a, s)[1:-1] == bg)),
                     "outer controls must be clean full spans")
    _require(_adjacent(outer) and len(set(outer.values())) == 2,
             "need two distinct adjacent outer identities")
    interior = a.copy()
    interior[0, :] = interior[-1, :] = bg
    interior[:, 0] = interior[:, -1] = bg
    bars = []
    for color in outer.values():
        binary = (interior == color).astype(np.int8)
        objects = _components(binary, 0, diagonal=False)
        _require(len(objects) == 1, "each identity needs one internal guide")
        cells = objects[0]
        rr, cc = sorted({r for r, _ in cells}), sorted({c for _, c in cells})
        if len(rr) == 1 and cc == list(range(cc[0], cc[-1] + 1)):
            bars.append(("H", rr[0], cc[0], cc[-1], color, cells))
        elif len(cc) == 1 and rr == list(range(rr[0], rr[-1] + 1)):
            bars.append(("V", cc[0], rr[0], rr[-1], color, cells))
        else:
            raise ValueError("guides must be straight contiguous bars")
    _require({b[0] for b in bars} == {"H", "V"}, "guides must be perpendicular")
    horizontal = next(b for b in bars if b[0] == "H")
    vertical = next(b for b in bars if b[0] == "V")
    _, hr, c0, c1, hc, _ = horizontal
    _, vc, r0, r1, vcolor, _ = vertical
    h, w = r1 - r0 + 1, c1 - c0 + 1
    _require(h >= 2 and w >= 2, "prototype needs nontrivial guide lengths")
    guides = {}
    if hr == r0 - 1:
        guides["T"] = hc
    elif hr == r1 + 1:
        guides["B"] = hc
    if vc == c0 - 1:
        guides["L"] = vcolor
    elif vc == c1 + 1:
        guides["R"] = vcolor
    _require(_adjacent(guides), "guide endpoints must define adjacent rectangle sides")
    _require(r0 >= 2 and c0 >= 2 and r1 <= H - 3 and c1 <= W - 3,
             "prototype must be separated from the frame")
    prototype = a[r0:r1 + 1, c0:c1 + 1].copy()
    code = int(prototype[0, 0])
    _require(code not in {bg, *outer.values()}, "prototype corner must be a code")
    values = set(map(int, np.unique(prototype))) - {bg, code}
    _require(len(values) == 1, "prototype needs one distinct ink")
    ink = values.pop()
    _require(ink not in outer.values() and np.count_nonzero(prototype == code) == 1,
             "prototype code and ink must have distinct roles")
    matrix = np.zeros((2, 2), dtype=int)
    for s, color in guides.items():
        target = next(t for t, value in outer.items() if value == color)
        matrix += np.outer(NORMALS[target], NORMALS[s])
    _require(tuple(map(tuple, matrix)) in MATRICES, "identity matches must define one symmetry")
    scene = interior.copy()
    scene[r0:r1 + 1, c0:c1 + 1] = bg
    for bar in bars:
        for q in bar[-1]:
            scene[q] = bg
    # Check a clear one-cell neighborhood around the definition and its bars.
    rr0, rr1 = r0 - ("T" in guides), r1 + ("B" in guides)
    cc0, cc1 = c0 - ("L" in guides), c1 + ("R" in guides)
    moat = scene[rr0 - 1:rr1 + 2, cc0 - 1:cc1 + 2]
    _require(bool(np.all(moat == bg)), "scene objects must not touch the definition")
    _require(set(map(int, np.unique(scene))) <= {bg, code, ink},
             "scene material must use code or ink")
    return dict(grid=a, bg=bg, code=code, ink=ink, mask=prototype == ink,
                matrix=matrix, outer=outer, guides=guides,
                bounds=(r0, c0, h, w), scene=scene)


def _classify(info, matrix=None):
    matrix = info["matrix"] if matrix is None else matrix
    signs = np.sum(matrix, axis=1)
    active, rejected, bare, multi, painted = [], [], [], [], []
    for cells in _components(info["scene"], info["bg"]):
        codes = [q for q in cells if info["scene"][q] == info["code"]]
        if not codes:
            bare.append(cells)
            continue
        if len(codes) != 1:
            multi.append(cells)
            rejected.extend(codes)
            continue
        origin = tuple((min if signs[d] == 1 else max)(q[d] for q in cells)
                       for d in range(2))
        if codes[0] == origin:
            active.append(codes[0])
            if len(cells) > 1:
                painted.append(codes[0])
        else:
            rejected.extend(codes)
    return dict(active=active, rejected=rejected, bare=bare,
                multi=multi, painted=painted)


def _offsets(info, matrix=None):
    matrix = info["matrix"] if matrix is None else matrix
    return np.argwhere(info["mask"]) @ matrix.T


def _print(info, points=None, matrix=None, mode="clip"):
    if points is None:
        points = _classify(info)["active"]
    offsets = _offsets(info, matrix)
    if mode == "normalized":
        # Incorrect ordinary top-left placement, used only as an ablation.
        corners = np.array(((0, 0), (0, info["mask"].shape[1] - 1),
                            (info["mask"].shape[0] - 1, 0),
                            (info["mask"].shape[0] - 1, info["mask"].shape[1] - 1)))
        m = info["matrix"] if matrix is None else matrix
        offsets = offsets - (corners @ m.T).min(axis=0)
    out = np.full(info["grid"].shape, info["bg"], dtype=np.int8)
    H, W = out.shape
    for p in points:
        target = offsets + p
        valid = ((target[:, 0] >= 0) & (target[:, 0] < H)
                 & (target[:, 1] >= 0) & (target[:, 1] < W))
        if mode == "contained" and not np.all(valid):
            continue
        target = target[valid]
        out[target[:, 0], target[:, 1]] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a clean print using only the visible input definition and frame."""
    return _print(_decode(input_grid))


def _palette(rng):
    return tuple(map(int, rng.permutation(10)[:5]))  # bg, keys, code, ink


def _mask_ok(mask):
    # Connected ink plus pivot, and eight distinct oriented arrays.
    occupied = mask.copy()
    occupied[0, 0] = True
    views = []
    for m in MATRICES:
        m = np.array(m)
        positions = np.indices(mask.shape).reshape(2, -1).T @ m.T
        positions -= positions.min(axis=0)
        view = np.zeros(tuple(positions.max(axis=0) + 1), dtype=np.int8)
        view[positions[:, 0], positions[:, 1]] = mask.ravel()
        views.append(_key(view))
    return len(_components(occupied.astype(np.int8), 0)) == 1 and len(set(views)) == 8


def _sample_mask(rng, shape, leading_blank=False):
    h, w = shape
    for _ in range(80):
        mask = np.zeros(shape, dtype=bool)
        r = c = 1
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            if r == h - 1 or (c < w - 1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            mask[r, c] = True
        # Sometimes retain one or both entire leading blank edges.
        blank_top, blank_left = rng.random() < .45, rng.random() < .45
        if leading_blank and not (blank_top or blank_left):
            if rng.random() < .5:
                blank_top = True
            else:
                blank_left = True
        for _ in range(int(rng.integers(2, h * w))):
            cells = np.argwhere(mask)
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            r, c = int(r + dr), int(c + dc)
            if 0 <= r < h and 0 <= c < w and (r, c) != (0, 0):
                if not (blank_top and r == 0 or blank_left and c == 0):
                    mask[r, c] = True
        if _mask_ok(mask):
            return mask
    # A bounded asymmetric fallback, not a memorized dictionary.
    mask = np.zeros(shape, dtype=bool)
    mask[1:, 1] = True
    mask[-1, 1:] = True
    if h == w:
        mask[0, 1] = True
    assert _mask_ok(mask)
    return mask


def _different_mask(rng, mask):
    for index in rng.permutation(mask.size):
        q = divmod(int(index), mask.shape[1])
        if q in ((0, 0), (1, 1), (mask.shape[0] - 1, mask.shape[1] - 1)):
            continue
        candidate = mask.copy()
        candidate[q] = ~candidate[q]
        if _mask_ok(candidate):
            return candidate
    for _ in range(20):
        candidate = _sample_mask(rng, mask.shape)
        if not np.array_equal(candidate, mask):
            return candidate
    raise AssertionError("could not vary the visible definition")


def _target_side(matrix, side):
    normal = tuple(matrix @ NORMALS[side])
    return next(s for s, n in NORMALS.items() if n == normal)


def _paint_keys(a, bg, guides, matrix, bounds):
    r, c, h, w = bounds
    a[0, :] = a[-1, :] = bg
    a[:, 0] = a[:, -1] = bg
    for side, color in guides.items():
        target = _target_side(matrix, side)
        _side(a, target)[1:-1] = color
        if side == "T":
            a[r - 1, c:c + w] = color
        elif side == "R":
            a[r:r + h, c + w] = color
        elif side == "B":
            a[r + h, c:c + w] = color
        else:
            a[r:r + h, c - 1] = color


def _halo(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[dr:dr + h, dc:dc + w]
                                 for dr in range(3) for dc in range(3)])


def _scaffold(rng, matrix, code, ink, reverse=False):
    # A variable L/patch. A code at its corner is visibly connected to ink.
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    cells = {(0, 0): code}
    for r in range(h):
        for c in range(w):
            if (r, c) != (0, 0) and (r == 0 or c == 0 or rng.random() < .65):
                q = tuple(((-matrix if reverse else matrix) @ (r, c)).tolist())
                cells[q] = ink
    return cells


def _place(rng, a, blocked, cells, points=None, stencil_offsets=None, full=False):
    """Exhaustively try a shuffled finite set of legal translations."""
    H, W = a.shape
    candidates = ([(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
                  if points is None else list(points))
    for index in rng.permutation(len(candidates)):
        origin = candidates[int(index)]
        translated = [(r + origin[0], c + origin[1], value)
                      for (r, c), value in cells.items()]
        if any(not (1 <= r < H - 1 and 1 <= c < W - 1) or blocked[r, c]
               for r, c, _ in translated):
            continue
        if full and stencil_offsets is not None:
            pivots = [(r, c) for r, c, value in translated if value == cells[(0, 0)]]
            if any(np.any((stencil_offsets + p < 0)
                          | (stencil_offsets + p >= np.array((H, W)))) for p in pivots):
                continue
        fg = np.zeros(a.shape, dtype=bool)
        for r, c, value in translated:
            a[r, c] = value
            fg[r, c] = True
        blocked |= _halo(fg)
        return origin
    return None


def _make_scene(rng, palette, mask, inner, matrix, teaching=False):
    bg, k1, k2, code, ink = palette
    h, w = mask.shape
    # The teaching scene has room for independent semantic witnesses.
    for attempt in range(12):
        H = int(rng.integers(24 if teaching else 16, 31))
        W = int(rng.integers(24 if teaching else 16, 31))
        if attempt == 11:
            H = W = 30
        r = int(rng.integers(4, H - h - 3))
        c = int(rng.integers(4, W - w - 3))
        bounds = (r, c, h, w)
        guides = dict(zip(inner, (k1, k2)))
        a = np.full((H, W), bg, dtype=np.int8)
        a[r:r + h, c:c + w][mask] = ink
        a[r, c] = code
        _paint_keys(a, bg, guides, matrix, bounds)
        blocked = np.zeros(a.shape, dtype=bool)
        blocked[r - 2:r + h + 2, c - 2:c + w + 2] = True
        offsets = np.argwhere(mask) @ matrix.T
        failed = False
        if teaching:
            for q in ((1, 1), (1, W - 2), (H - 2, 1), (H - 2, W - 2)):
                if _place(rng, a, blocked, {(0, 0): code}, points=[q]) is None:
                    failed = True
        else:
            # At least one edge opportunity; later random points can be clean.
            if rng.random() < .65:
                _place(rng, a, blocked, {(0, 0): code},
                       points=[(1, 1), (1, W - 2), (H - 2, 1), (H - 2, W - 2)])
        # Always include a fully visible, unobscured live request.
        if _place(rng, a, blocked, {(0, 0): code}, stencil_offsets=offsets, full=True) is None:
            failed = True
        if teaching or rng.random() < .45:
            # Mask endpoints create a real shared ink pixel, while pivots are
            # not eight-neighbors. This is a union witness, not mere adjacency.
            delta = matrix @ (np.array(mask.shape) - 2)
            if max(abs(delta)) >= 2:
                pair = {(0, 0): code, tuple(map(int, delta)): code}
                if _place(rng, a, blocked, pair, stencil_offsets=offsets, full=True) is None and teaching:
                    failed = True
        kinds = ["front", "back", "bare", "multi"] if teaching else [
            kind for kind in ("front", "back", "bare", "multi") if rng.random() < .55]
        for kind in kinds:
            cells = _scaffold(rng, matrix, code, ink, reverse=kind == "back")
            if kind == "bare":
                cells = {q: ink for q in cells}
            elif kind == "multi":
                q = next(q for q in cells if q != (0, 0))
                cells[q] = code
            if _place(rng, a, blocked, cells) is None and teaching:
                failed = True
        for _ in range(int(rng.integers(0, 5 if teaching else 10))):
            _place(rng, a, blocked, {(0, 0): code})
        if failed:
            continue
        info = _decode(a)
        member = _classify(info)
        if not member["active"]:
            continue
        if teaching:
            m = _metrics(info)
            if not (m["even_overlap"] and m["painted"] and m["rejected"]
                    and m["bare"] and m["multi"] and m["clipped"]):
                continue
        return a
    raise AssertionError("bounded scene packing failed")


def _validate_count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = _palette(rng)
    stable = bool(rng.random() < .3)
    shape = tuple(map(int, rng.integers(4, 6, size=2)))
    mask1 = _sample_mask(rng, shape, leading_blank=True)
    inner_index = int(rng.choice((0, 2)))  # TR or BL: swap changes origin corner.
    inner = (SIDES[inner_index], SIDES[(inner_index + 1) % 4])
    matrix1 = np.array(MATRICES[int(rng.choice((1, 3)))], dtype=int)
    first = _make_scene(rng, palette, mask1, inner, matrix1, teaching=True)
    info1 = _decode(first)
    # Swap outer identities alone. Then change the visible stencil definition
    # without moving guides or scene objects. Both changes are observable.
    for _ in range(40):
        mask2 = _different_mask(rng, mask1)
        second = first.copy()
        controls = list(info1["outer"])
        for side, replacement in zip(controls, [info1["outer"][s] for s in controls[::-1]]):
            _side(second, side)[1:-1] = replacement
        r, c, h, w = info1["bounds"]
        second[r:r + h, c:c + w] = palette[0]
        second[r:r + h, c:c + w][mask2] = palette[4]
        second[r, c] = palette[3]
        info2 = _decode(second)
        if not np.array_equal(_print(info2), _print(dict(info2, mask=mask1))):
            break
    else:
        raise AssertionError("changed prototype must change visible teaching ink")
    inputs = [first, second]
    seen = {_key(a) for a in inputs}
    for _ in range(int(num_train) - 2):
        for attempt in range(12):
            local = palette if stable else _palette(rng)
            mask = _sample_mask(rng, tuple(map(int, rng.integers(3, 6, size=2))))
            i = int(rng.integers(4))
            m = np.array(MATRICES[int(rng.integers(8))])
            a = _make_scene(rng, local, mask, (SIDES[i], SIDES[(i + 1) % 4]), m)
            if _key(a) not in seen:
                break
        else:
            raise AssertionError("distinct scene generation failed")
        inputs.append(a)
        seen.add(_key(a))
    taught = {_arrangement(_decode(a)) for a in inputs}
    novel = []
    for i in range(4):
        local_inner = (SIDES[i], SIDES[(i + 1) % 4])
        for values in MATRICES:
            m = np.array(values)
            arrangement = (tuple(sorted(local_inner)),
                           tuple(sorted(_target_side(m, s) for s in local_inner)))
            if arrangement not in taught:
                novel.append((local_inner, m))
    assert novel
    for test_index in range(int(num_test)):
        # Balance visible test frames across seeds. This selects inputs only;
        # the rule remains entirely determined by their keys. If ten training
        # examples have exhausted that frame's side combinations, choose any
        # remaining novel combination instead.
        preferred = MATRICES[(int(seed) + test_index) % len(MATRICES)]
        pool = [(s, m) for s, m in novel if tuple(map(tuple, m)) == preferred]
        pool = pool or novel
        for attempt in range(12):
            local = palette if stable else _palette(rng)
            shape = tuple(map(int, rng.integers(3, 6, size=2)))
            mask = _sample_mask(rng, shape)
            local_inner, matrix = pool[int(rng.integers(len(pool)))]
            a = _make_scene(rng, local, mask, local_inner, matrix)
            if _key(a) not in seen:
                break
        else:
            raise AssertionError("distinct test generation failed")
        inputs.append(a)
        seen.add(_key(a))
    puzzle = {"train": [], "test": []}
    for i, a in enumerate(inputs):
        puzzle["train" if i < num_train else "test"].append({"input": a, "output": solve(a)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _arrangement(info):
    return tuple(sorted(info["guides"])), tuple(sorted(info["outer"]))


def _metrics(info):
    member = _classify(info)
    offsets = _offsets(info)
    H, W = info["grid"].shape
    occupancy = np.zeros((H, W), dtype=int)
    clipped, edges = 0, set()
    full = 0
    for p in member["active"]:
        target = offsets + p
        for axis, limit, low_name, high_name in ((0, H, "T", "B"), (1, W, "L", "R")):
            if np.any(target[:, axis] < 0):
                edges.add(low_name)
            if np.any(target[:, axis] >= limit):
                edges.add(high_name)
        valid = ((target[:, 0] >= 0) & (target[:, 0] < H)
                 & (target[:, 1] >= 0) & (target[:, 1] < W))
        clipped += not bool(np.all(valid))
        full += bool(np.all(valid))
        target = target[valid]
        occupancy[target[:, 0], target[:, 1]] += 1
    return dict(active=len(member["active"]), rejected=len(member["rejected"]),
                painted=len(member["painted"]), bare=len(member["bare"]),
                multi=len(member["multi"]), clipped=clipped, full=full,
                edges=tuple(sorted(edges)), overlap=int(np.count_nonzero(occupancy > 1)),
                even_overlap=int(np.count_nonzero((occupancy > 0) & (occupancy % 2 == 0))),
                shape=info["grid"].shape, mask_shape=info["mask"].shape,
                matrix=tuple(map(tuple, info["matrix"])),
                palette=(info["bg"], *sorted(info["outer"].values()), info["code"], info["ink"]),
                guides=tuple(sorted(info["guides"])), outer=tuple(sorted(info["outer"])),
                blank_top=not bool(info["mask"][0].any()),
                blank_left=not bool(info["mask"][:, 0].any()),
                front=tuple(map(int, info["matrix"].sum(axis=1))))


def _ablations(info):
    identity = np.eye(2, dtype=int)
    all_codes = [tuple(map(int, q)) for q in np.argwhere(info["scene"] == info["code"])]
    singletons = [cells[0] for cells in _components(info["scene"], info["bg"])
                  if len(cells) == 1 and info["scene"][cells[0]] == info["code"]]
    screen = _classify(info, identity)["active"]
    ink_positions = np.argwhere(info["mask"])
    start, end = ink_positions.min(axis=0), ink_positions.max(axis=0) + 1
    cropped = info["mask"][start[0]:end[0], start[1]:end[1]]
    occupancy = np.zeros(info["grid"].shape, dtype=int)
    H, W = occupancy.shape
    for point in _classify(info)["active"]:
        for r, c in _offsets(info) + point:
            if 0 <= r < H and 0 <= c < W:
                occupancy[r, c] += 1
    parity = np.full((H, W), info["bg"], dtype=np.int8)
    parity[occupancy % 2 == 1] = info["ink"]
    return dict(all_codes=_print(info, all_codes), singletons=_print(info, singletons),
                screen_membership=_print(info, screen), identity_offsets=_print(info, matrix=identity),
                omit_b=_print(info, screen, identity), normalized=_print(info, mode="normalized"),
                contained=_print(info, mode="contained"), copy_scene=info["scene"].copy(),
                tight_prototype=_print(dict(info, mask=cropped)), parity=parity)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    information = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _print(info)), "incorrect input-derived print"
            assert set(map(int, np.unique(out))) <= {info["bg"], info["ink"]}
            metric = _metrics(info)
            assert metric["active"] >= 1 and metric["full"] >= 1
            information[split].append(info)
    train, test = information["train"], information["test"]
    first, second = train[:2]
    assert _arrangement(first) == _arrangement(second), "need an identity correspondence contrast"
    assert np.array_equal(first["scene"], second["scene"]), "teaching scene must be shared"
    assert first["guides"] == second["guides"] and first["bounds"] == second["bounds"]
    assert not np.array_equal(first["mask"], second["mask"]), "prototype must visibly change"
    assert not np.array_equal(_print(second), _print(dict(second, mask=first["mask"]))), \
        "a memorized first prototype must fail on demonstration two"
    assert np.array_equal(first["matrix"].sum(axis=1), -second["matrix"].sum(axis=1))
    assert not np.array_equal(first["matrix"], second["matrix"])
    assert round(np.linalg.det(first["matrix"])) == 1 and first["matrix"][0, 0] == 0
    assert round(np.linalg.det(second["matrix"])) == -1
    af, az = _classify(first), _classify(second)
    assert set(af["painted"]) & set(az["rejected"]), "forward scaffold must switch off"
    assert set(az["painted"]) & set(af["rejected"]), "opposite scaffold must switch on"
    metrics = [_metrics(i) for i in train]
    for name in ("painted", "rejected", "bare", "multi", "clipped", "overlap"):
        assert metrics[0][name] > 0, f"training lacks {name} witness"
    alternatives = [_ablations(i) for i in train]
    for name in alternatives[0]:
        assert any(not np.array_equal(p["output"], alt[name])
                   for p, alt in zip(puzzle["train"], alternatives)), f"shortcut survives: {name}"
    taught = {_arrangement(i) for i in train}
    assert all(_arrangement(i) not in taught for i in test), "test side combination must be held out"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> dict:
    """Bounded checks over 42 seeds; reports observed, decoded variation."""
    puzzles, records, masks = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                masks.add(_key(info["mask"]))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                records.append((split, _metrics(info)))
        assert _equal(p, generate_puzzle(seed))
    assert len({tuple(_key(p["input"]) for s in q.values() for p in s) for q in puzzles}) == 42
    metrics = [m for _, m in records]
    transforms = {m["matrix"] for m in metrics}
    assert transforms == set(MATRICES)
    assert {m["front"] for m in metrics} == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert {e for m in metrics for e in m["edges"]} == set(SIDES)
    assert len({m["shape"] for m in metrics}) >= 45
    assert len({m["mask_shape"] for m in metrics}) >= 8
    assert len({m["palette"] for m in metrics}) >= 40
    assert len(masks) >= 100
    assert len({m["active"] for m in metrics}) >= 6
    assert any(m["blank_top"] for m in metrics) and any(m["blank_left"] for m in metrics)
    assert any(not m["painted"] and not m["multi"] for m in metrics)
    for split in ("train", "test"):
        local = [m for s, m in records if s == split]
        assert {m["matrix"] for m in local} == set(MATRICES)
        assert {m["guides"] for m in local} == {tuple(sorted((SIDES[i], SIDES[(i + 1) % 4]))) for i in range(4)}
    # Check the bounded motif fallback at every generated size, even when the
    # random sampler does not need it. Its square cases must not be symmetric.
    for h in range(3, 6):
        for w in range(3, 6):
            fallback = np.zeros((h, w), dtype=bool)
            fallback[1:, 1] = fallback[-1, 1:] = True
            if h == w:
                fallback[0, 1] = True
            assert _mask_ok(fallback)
    # Every legal explicit count combination, including minimal training.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(400 + 17 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Verify geometry directly with every signed matrix, preserving the pivot.
    base = _decode(puzzles[0]["train"][0]["input"])
    for values in MATRICES:
        matrix = np.array(values)
        offsets = _offsets(base, matrix)
        for point in ((1, 1), (1, 28), (28, 1), (28, 28), (15, 15)):
            fixture = dict(base, grid=np.full((30, 30), base["bg"], dtype=np.int8), matrix=matrix)
            expected = np.full((30, 30), base["bg"], dtype=np.int8)
            for r, c in np.argwhere(base["mask"]):
                rr = point[0] + int(matrix[0, 0] * r + matrix[0, 1] * c)
                cc = point[1] + int(matrix[1, 0] * r + matrix[1, 1] * c)
                if 0 <= rr < 30 and 0 <= cc < 30:
                    expected[rr, cc] = base["ink"]
            assert np.array_equal(_print(fixture, [point]), expected)
            assert np.array_equal(offsets, np.argwhere(base["mask"]) @ matrix.T)
    # Wrong cells, a deleted entire print, truncation, and a broken guide fail.
    for kind in ("cell", "erase", "crop", "guide"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if kind == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "erase":
            pair["output"][:] = info["bg"]
        elif kind == "crop":
            pair["output"] = pair["output"][1:, :].copy()
        else:
            r, c, h, w = info["bounds"]
            s = next(iter(info["guides"]))
            q = {"T": (r - 1, c + w // 2), "R": (r + h // 2, c + w),
                 "B": (r + h, c + w // 2), "L": (r + h // 2, c - 1)}[s]
            pair["input"][q] = info["bg"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted {kind} corruption")
    return dict(seeds=42, pairs=len(records), transforms=len(transforms),
                canvas_shapes=len({m["shape"] for m in metrics}),
                prototype_shapes=len({m["mask_shape"] for m in metrics}),
                binary_prototypes=len(masks),
                palettes=len({m["palette"] for m in metrics}),
                active_counts=sorted({m["active"] for m in metrics}),
                overlap_pairs=sum(m["overlap"] > 0 for m in metrics),
                clipped_pairs=sum(m["clipped"] > 0 for m in metrics),
                note="Generator-owned checks; no independent solver or blind prediction.")


if __name__ == "__main__":
    print(self_test())
