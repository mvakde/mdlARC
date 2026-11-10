"""Restore a fractured cut, then expand its two corner-color keys regionally.

No parent module, stored example, or generation parameter is used by solve.
The exhaustive search has at most 3! * 4**3 full corner arrangements.
"""
from math import isqrt

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    directions = DIR8 if diagonal else DIR4
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for p in cells:
        result[p] = True
    return result


def _control_info(control):
    """One simple digital boundary-to-boundary path and one isolated dot."""
    components = sorted(_components(control, True), key=len)
    if len(components) != 2 or len(components[0]) != 1 or len(components[1]) < 3:
        return None
    marker = components[0][0]
    path_cells = set(components[1])
    graph = {}
    for r, c in path_cells:
        neighbors = []
        for dr, dc in DIR8:
            p = r + dr, c + dc
            if p not in path_cells:
                continue
            # A right angle has orthogonal edges, not a diagonal shortcut.
            if dr and dc and ((r + dr, c) in path_cells or (r, c + dc) in path_cells):
                continue
            neighbors.append(p)
        graph[r, c] = neighbors
    ends = sorted(p for p, ns in graph.items() if len(ns) == 1)
    if len(ends) != 2 or any(len(ns) not in (1, 2) for ns in graph.values()):
        return None
    h, w = control.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    if not all(border(p) for p in ends) or any(border(p) for p in path_cells - set(ends)):
        return None
    order, previous, current = [], None, ends[0]
    while current is not None and current not in order:
        order.append(current)
        following = [p for p in graph[current] if p != previous]
        previous, current = current, following[0] if following else None
    if len(order) != len(path_cells):
        return None
    cut = _mask(control.shape, path_cells)
    regions = _components(~cut)
    if len(regions) != 2:
        return None
    region_masks = [_mask(control.shape, cells) for cells in regions]
    if not any(m[marker] for m in region_masks):
        return None
    side = 0 if region_masks[0][marker] else 1
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(order, order[1:])]
    bends = sum(a != b for a, b in zip(steps, steps[1:]))
    return {"cut": cut, "marker": marker, "marked": region_masks[side],
            "regions": region_masks, "side": side, "path": order, "bends": bends}


def _parse(grid):
    grid = _grid(grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "uniform background border required")
    components = _components(grid != bg)
    _require(len(components) == 4, "four connected footprints required")
    for i, x in enumerate(components):
        for y in components[i + 1:]:
            distance = np.max(np.abs(np.array(x)[:, None] - np.array(y)[None, :]), axis=2)
            _require(int(distance.min()) >= 2, "pieces must have a background-cell gap")
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    _require(len(colors) >= 6, "missing semantic colors")
    maximum = counts.max()
    _require(np.count_nonzero(counts == maximum) == 1, "carrier must be dominant")
    carrier = int(colors[counts.argmax()])
    pieces = []
    for cells in components:
        coords = np.array(cells)
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        p = np.full(tuple(hi - lo), bg, dtype=np.int8)
        local = coords - lo
        p[local[:, 0], local[:, 1]] = grid[coords[:, 0], coords[:, 1]]
        pieces.append({"pixels": p, "offset": tuple(map(int, lo))})
    # The sole 2x4 patch holds two adjacent 2x2 keys. Everything special
    # outside it is the single control color. This also permits key/cut contact.
    keys = []
    for i, piece in enumerate(pieces):
        p = piece["pixels"]
        for r in range(p.shape[0] - 1):
            for c in range(p.shape[1] - 3):
                block = p[r:r + 2, c:c + 4]
                if np.any((block == bg) | (block == carrier)):
                    continue
                if len(np.unique(block[:, :2])) != 4 or len(np.unique(block[:, 2:])) != 4:
                    continue
                outside = set()
                for j, q in enumerate(pieces):
                    special = (q["pixels"] != bg) & (q["pixels"] != carrier)
                    if j == i:
                        special = special.copy()
                        special[r:r + 2, c:c + 4] = False
                    outside.update(map(int, q["pixels"][special]))
                if len(outside) == 1 and not outside.intersection(map(int, block.flat)):
                    keys.append((i, r, c, block.copy(), next(iter(outside))))
    _require(len(keys) == 1, "unique two-key anchor required")
    anchor, r, c, block, control = keys[0]
    pieces = [pieces[anchor]] + [q for i, q in enumerate(pieces) if i != anchor]
    area = sum(int(np.count_nonzero(q["pixels"] != bg)) for q in pieces)
    side = isqrt(area)
    _require(side * side == area, "footprint area must be a square")
    _require(side >= 5, "square too small for the keys")
    _require(np.count_nonzero(block[:, :2] != block[:, 2:]) >= 2, "keys must differ at two corners")
    return {"grid": grid, "background": bg, "carrier": carrier, "control": control,
            "pieces": pieces, "side": side, "key_a": block[:, :2].copy(),
            "key_b": block[:, 2:].copy(), "key_location": (r, c)}


def _candidate(pixels, bg, side, corner, turns):
    p = np.rot90(pixels, turns).copy()
    h, w = p.shape
    if h > side or w > side:
        return None
    r = 0 if corner < 2 else side - h
    c = 0 if corner % 2 == 0 else side - w
    cells = np.argwhere(p != bg) + (r, c)
    corners = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))
    cell_set = set(map(tuple, cells.tolist()))
    if [x in cell_set for x in corners] != [j == corner for j in range(4)]:
        return None
    bitmask = sum(1 << (int(rr) * side + int(cc)) for rr, cc in cells)
    return {"mask": bitmask, "pixels": p, "offset": (r, c), "turns": turns}


def _paint_answer(owner, info, key_a, key_b):
    a, b = key_a.reshape(-1)[owner], key_b.reshape(-1)[owner]
    return np.where(info["marked"] | info["cut"], a, b).astype(np.int8)


def _reconstruct(parsed, require_unique=True):
    side, bg = parsed["side"], parsed["background"]
    pieces = parsed["pieces"]
    anchor = _candidate(pieces[0]["pixels"], bg, side, 0, 0)
    _require(anchor is not None, "anchor must remain unrotated at top-left")
    table = {}
    for i in range(1, 4):
        for corner in range(1, 4):
            candidates, seen = [], set()
            for turns in range(4):
                q = _candidate(pieces[i]["pixels"], bg, side, corner, turns)
                if q is not None:
                    signature = (q["mask"], q["pixels"].tobytes())
                    if signature not in seen:
                        seen.add(signature)
                        candidates.append(q)
            table[i, corner] = candidates
    full = (1 << (side * side)) - 1
    answers, geometry_controls = {}, set()
    geometry_count = 0
    selected = [anchor, None, None, None]
    identities = [0, None, None, None]

    def search(corner, remaining, occupied):
        nonlocal geometry_count
        if require_unique and len(answers) > 1:
            return
        if corner == 4:
            if occupied != full:
                return
            geometry_count += 1
            owner = np.empty((side, side), dtype=np.int8)
            control_board = np.zeros((side, side), dtype=bool)
            for j, q in enumerate(selected):
                rr, cc = q["offset"]
                p = q["pixels"]
                occupied_patch = p != bg
                view = owner[rr:rr + p.shape[0], cc:cc + p.shape[1]]
                view[occupied_patch] = j
                control_board[rr:rr + p.shape[0], cc:cc + p.shape[1]] |= p == parsed["control"]
            geometry_controls.add(control_board.tobytes())
            info = _control_info(control_board)
            if info is None:
                return
            answer = _paint_answer(owner, info, parsed["key_a"], parsed["key_b"])
            signature = answer.tobytes()
            if signature not in answers:
                answers[signature] = {"output": answer, "owner": owner, "info": info,
                                      "control_board": control_board,
                                      "placement": tuple(dict(q) for q in selected),
                                      "identities": tuple(identities)}
            return
        for i in remaining:
            for q in table[i, corner]:
                if not (occupied & q["mask"]):
                    selected[corner], identities[corner] = q, i
                    search(corner + 1, tuple(j for j in remaining if j != i), occupied | q["mask"])

    search(1, (1, 2, 3), anchor["mask"])
    _require(len(answers) == 1, "exactly one answer under packing and cut constraints required")
    result = next(iter(answers.values()))
    result["geometry_count"] = geometry_count
    result["geometry_controls"] = len(geometry_controls)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, placements, regions, and colors solely from input pixels."""
    return _reconstruct(_parse(input_grid))["output"].copy()


def _key_sites(owner, cut):
    n = len(owner)
    return [(r, c) for r in range(n - 1) for c in range(n - 3)
            if np.all(owner[r:r + 2, c:c + 4] == 0) and not cut[r:r + 2, c:c + 4].any()]


def _partition(rng, side, style):
    """Procedural corner pieces: rectangles, wandering seams, or boundary growth."""
    n = side
    owner = np.empty((n, n), dtype=np.int8)
    if style == 0:
        y, top, bottom = n // 2, n // 2, n // 2
        owner[:y, :top], owner[:y, top:] = 0, 1
        owner[y:, :bottom], owner[y:, bottom:] = 2, 3
        return owner
    if style == 1:
        y = int(rng.integers(3, n - 2))
        top, bottom = int(rng.integers(4, n - 1)), int(rng.integers(1, n - 1))
        for r in range(n):
            if r < y:
                if r >= 2:
                    top = int(np.clip(top + rng.integers(-1, 2), 1, n - 1))
                owner[r, :top], owner[r, top:] = 0, 1
            else:
                bottom = int(np.clip(bottom + rng.integers(-1, 2), 1, n - 1))
                owner[r, :bottom], owner[r, bottom:] = 2, 3
    else:
        owner.fill(-1)
        owner[:2, :4] = 0
        for label, p in enumerate(((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))):
            owner[p] = label
        weights = rng.uniform(.25, 2.0, size=4)
        while np.any(owner == -1):
            frontiers = [[] for _ in range(4)]
            for r, c in np.argwhere(owner == -1):
                adjacent = {int(owner[r + dr, c + dc]) for dr, dc in DIR4
                            if 0 <= r + dr < n and 0 <= c + dc < n and owner[r + dr, c + dc] >= 0}
                for label in adjacent:
                    frontiers[label].append((int(r), int(c)))
            available = [i for i, cells in enumerate(frontiers) if cells]
            probs = weights[available] / weights[available].sum()
            label = int(rng.choice(available, p=probs))
            cells = frontiers[label]
            owner[cells[int(rng.integers(len(cells)))]] = label
    # Local transfers introduce tabs and recesses without separating footprints.
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    for _ in range(2 * n):
        r, c = int(rng.integers(n)), int(rng.integers(n))
        if (r, c) in corners:
            continue
        old = int(owner[r, c])
        adjacent = [int(owner[r + dr, c + dc]) for dr, dc in DIR4
                    if 0 <= r + dr < n and 0 <= c + dc < n and owner[r + dr, c + dc] != old]
        if not adjacent:
            continue
        new = int(rng.choice(adjacent))
        owner[r, c] = new
        if len(_components(owner == old)) != 1 or not _key_sites(owner, np.zeros_like(owner, dtype=bool)):
            owner[r, c] = old
    return owner


def _cut(rng, n, kind):
    cut = np.zeros((n, n), dtype=bool)
    if kind == "straight":
        cut[int(rng.integers(2, n - 2)), :] = True
    elif kind == "diagonal":
        cut[np.arange(n), np.arange(n)] = True
    elif kind == "elbow":
        r, c = int(rng.integers(2, n - 2)), int(rng.integers(2, n - 2))
        cut[:r + 1, c] = True
        cut[r, c:] = True
    else:
        r = int(rng.integers(2, n - 2))
        a, b = sorted(map(int, rng.choice(np.arange(2, n - 2), 2, replace=False)))
        cut[:r + 1, a] = True
        cut[r, a:b + 1] = True
        cut[r:, b] = True
    return np.rot90(cut, int(rng.integers(4))).copy()


def _marker_sites(cut, excluded=None):
    forbidden = cut.copy()
    n = len(cut)
    for r, c in np.argwhere(cut):
        forbidden[max(0, r - 1):min(n, r + 2), max(0, c - 1):min(n, c + 2)] = True
    if excluded is not None:
        forbidden |= excluded
    regions = _components(~cut)
    if len(regions) != 2:
        return []
    return [[p for p in cells if not forbidden[p]] for cells in regions]


def _scatter(rng, painted, bg):
    """Shuffle horizontal/vertical shelves, jitter, and pad with background."""
    for attempt in range(20):
        order = list(map(int, rng.permutation(4)))
        groups = [[order[0], order[1]], [order[2], order[3]]]
        if attempt < 10:
            cuts = int(rng.integers(8))
            groups, group = [], []
            for j, i in enumerate(order):
                group.append(i)
                if j == 3 or (cuts >> j) & 1:
                    groups.append(group)
                    group = []
        gap = int(rng.integers(1, 4)) if attempt < 10 else 1
        heights = [max(painted[i].shape[0] for i in g) for g in groups]
        widths = [sum(painted[i].shape[1] for i in g) + gap * (len(g) - 1) for g in groups]
        h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
        if h > 28 or w > 28:
            continue
        pad_r = int(rng.integers(1, min(3, 29 - h) + 1))
        pad_c = int(rng.integers(1, min(3, 29 - w) + 1))
        grid = np.full((h + pad_r + 1, w + pad_c + 1), bg, dtype=np.int8)
        row = pad_r
        for g, height, width in zip(groups, heights, widths):
            col = pad_c + int(rng.integers(w - width + 1))
            for i in g:
                p = painted[i]
                rr = row + int(rng.integers(height - p.shape[0] + 1))
                view = grid[rr:rr + p.shape[0], col:col + p.shape[1]]
                view[p != bg] = p[p != bg]
                col += p.shape[1] + gap
            row += height + gap
        return grid
    return None


def _build(rng, n, style, kind, side_selection=None, teaching=False):
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, ctrl = palette[:3]
    key_a = np.array(palette[3:7], dtype=np.int8).reshape(2, 2)
    if teaching or rng.random() < .45:
        # Reuse colors in a changed spatial assignment; their literal identities
        # cannot reveal which of the two keys was selected.
        key_b = np.roll(key_a.reshape(-1), int(rng.integers(1, 4))).reshape(2, 2)
    else:
        key_b = np.array(rng.choice(palette[3:9], 4, replace=False), dtype=np.int8).reshape(2, 2)
        if np.count_nonzero(key_a != key_b) < 2:
            key_b = np.roll(key_a.reshape(-1), 1).reshape(2, 2)
    owner = _partition(rng, n, style)
    if any(np.count_nonzero(owner == i) < 3 for i in range(4)):
        return None
    if teaching:
        cut = np.zeros((n, n), dtype=bool)
        cut[int(rng.integers(2, n // 2)), :] = True
    else:
        cut = _cut(rng, n, kind)
    locations = _key_sites(owner, cut)
    if not locations:
        return None
    kr, kc = locations[int(rng.integers(len(locations)))]
    metadata = np.zeros((n, n), dtype=bool)
    metadata[kr:kr + 2, kc:kc + 4] = True
    sites = _marker_sites(cut, metadata)
    if len(sites) != 2 or not all(sites):
        return None
    side_selection = int(rng.integers(2)) if side_selection is None else side_selection
    marker = sites[side_selection][int(rng.integers(len(sites[side_selection])))]
    control_board = cut.copy()
    control_board[marker] = True
    info = _control_info(control_board)
    if info is None:
        return None
    board = np.full((n, n), carrier, dtype=np.int8)
    board[control_board] = ctrl
    board[kr:kr + 2, kc:kc + 4] = np.concatenate((key_a, key_b), axis=1)
    painted = []
    rotations = [0, 1, 2, 3] if teaching else [0] + list(map(int, rng.integers(4, size=3)))
    for i in range(4):
        coords = np.argwhere(owner == i)
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        p = np.full(tuple(hi - lo), bg, dtype=np.int8)
        cells = coords - lo
        p[cells[:, 0], cells[:, 1]] = board[coords[:, 0], coords[:, 1]]
        painted.append(np.rot90(p, rotations[i]).copy())
    grid = _scatter(rng, painted, bg)
    if grid is None:
        return None
    expected = _paint_answer(owner, info, key_a, key_b)
    try:
        parsed = _parse(grid)
        restored = _reconstruct(parsed)
    except AssertionError:
        return None
    if not np.array_equal(restored["output"], expected):
        return None
    if teaching and parsed["pieces"][0]["offset"] == min(q["offset"] for q in parsed["pieces"]):
        return None
    a, b = key_a.reshape(-1)[owner], key_b.reshape(-1)[owner]
    if not (np.any((a != b) & info["marked"]) and
            np.any((a != b) & ~info["marked"] & ~info["cut"])):
        return None
    return {"input": grid, "output": expected}


def _opposite_marker(pair, rng):
    parsed = _parse(pair["input"])
    restored = _reconstruct(parsed)
    info, owner = restored["info"], restored["owner"]
    metadata = np.zeros(owner.shape, dtype=bool)
    r, c = parsed["key_location"]
    metadata[r:r + 2, c:c + 4] = True
    sites = _marker_sites(info["cut"], metadata)[1 - info["side"]]
    for index in rng.permutation(len(sites)):
        new_marker = sites[int(index)]
        grid = pair["input"].copy()
        for point, value in ((info["marker"], parsed["carrier"]), (new_marker, parsed["control"])):
            corner = int(owner[point])
            q = restored["placement"][corner]
            piece_i = restored["identities"][corner]
            original = parsed["pieces"][piece_i]
            labels = np.arange(original["pixels"].size).reshape(original["pixels"].shape)
            labels = np.rot90(labels, q["turns"])
            local = point[0] - q["offset"][0], point[1] - q["offset"][1]
            rr, cc = divmod(int(labels[local]), original["pixels"].shape[1])
            grid[rr + original["offset"][0], cc + original["offset"][1]] = value
        try:
            new_restored = _reconstruct(_parse(grid))
        except AssertionError:
            continue
        if np.any(pair["output"] != new_restored["output"]):
            return {"input": grid, "output": new_restored["output"].copy()}
    return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # larger scale, smaller scale, more bends
    train_sizes = (8, 9, 10) if regime == 0 else ((10, 11, 12) if regime == 1 else (8, 9, 10, 11, 12))
    test_sizes = (11, 12) if regime == 0 else ((8, 9) if regime == 1 else train_sizes)
    puzzle, seen = {"train": [], "test": []}, set()
    first_side = int(rng.integers(2))
    for attempt in range(160):
        n = int(rng.choice([x for x in train_sizes if x % 2 == 0]))
        first = _build(rng, n, 0, "straight", first_side, teaching=True)
        if first is None:
            continue
        second = _opposite_marker(first, rng)
        if second is not None:
            break
    else:
        raise RuntimeError("unable to construct the controlled demonstrations")
    puzzle["train"] = [first, second]
    seen.update((p["input"].shape, p["input"].tobytes()) for p in puzzle["train"])
    for split, count in (("train", nt - 2), ("test", nq)):
        for _ in range(count):
            for attempt in range(220):
                n = int(rng.choice(train_sizes if split == "train" else test_sizes))
                style = int(rng.choice([0, 1, 2], p=[.12, .48, .4]))
                if regime == 2:
                    pool = ("straight", "diagonal") if split == "train" else ("elbow", "step")
                elif nt == 2:
                    pool = ("straight",)
                else:
                    pool = ("straight", "diagonal", "elbow", "step")
                kind = str(rng.choice(pool))
                pair = _build(rng, n, style, kind)
                if pair is None:
                    continue
                signature = pair["input"].shape, pair["input"].tobytes()
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                # A randomized simple geometry still honors the intended size
                # shift. The restored path can be bent when that is the shift.
                for attempt in range(220):
                    n = int(rng.choice([x for x in (train_sizes if split == "train" else test_sizes) if x % 2 == 0]))
                    pair = _build(rng, n, 0, "step" if regime == 2 and split == "test" else "straight")
                    if pair is not None and (pair["input"].shape, pair["input"].tobytes()) not in seen:
                        seen.add((pair["input"].shape, pair["input"].tobytes()))
                        puzzle[split].append(pair)
                        break
                else:
                    raise RuntimeError("bounded input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, analyses = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = inp.shape, inp.tobytes()
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            parsed = _parse(inp)
            restored = _reconstruct(parsed)
            _require(out.shape == (parsed["side"],) * 2, "area conservation")
            _require(np.array_equal(out, restored["output"]), "incorrect regional keyed assembly")
            _require(not np.array_equal(inp, out), "identity pair")
            info, owner = restored["info"], restored["owner"]
            a, b = parsed["key_a"].reshape(-1)[owner], parsed["key_b"].reshape(-1)[owner]
            _require(np.any((a != b) & info["marked"]), "first source must visibly matter")
            _require(np.any((a != b) & ~info["marked"] & ~info["cut"]), "second source must visibly matter")
            if split == "train":
                analyses.append((parsed, restored))
    p, first = analyses[0]
    q, second = analyses[1]
    _require(np.array_equal(first["owner"], second["owner"]), "controlled demonstrations need identical assembly")
    _require(np.array_equal(first["info"]["cut"], second["info"]["cut"]), "controlled cut reuse")
    _require(np.array_equal(p["key_a"], q["key_a"]) and np.array_equal(p["key_b"], q["key_b"]), "controlled key reuse")
    _require(np.count_nonzero(p["grid"] != q["grid"]) == 2, "only the marker should move")
    _require(np.array_equal(first["info"]["marked"], ~second["info"]["marked"] & ~first["info"]["cut"]), "marker must select opposite regions")
    _require(np.any((first["output"] != second["output"]) & ~first["info"]["cut"]), "marker must change the answer")
    _require(np.array_equal(first["output"][first["info"]["cut"]], second["output"][first["info"]["cut"]]), "cut priority must not follow the marker")
    _require(np.any(p["key_a"].reshape(-1)[first["owner"]][first["info"]["cut"]] !=
                    p["key_b"].reshape(-1)[first["owner"]][first["info"]["cut"]]), "cut priority evidence")
    _require(first["geometry_controls"] >= 2, "cut must disambiguate geometry-only control placements")
    _require(p["pieces"][0]["offset"] != min(x["offset"] for x in p["pieces"]),
             "the anchor must demonstrate that scatter position is not output position")
    # These counterfactuals remove the inherited region mechanism and the
    # inherited corner assignment respectively; both must change the label.
    first_field = p["key_a"].reshape(-1)[first["owner"]]
    _require(np.any(first_field != first["output"]), "omitting A did not change the answer")
    order = sorted(range(4), key=lambda i: p["pieces"][i]["offset"])
    assigned = np.array([order.index(i) for i in first["identities"]])
    wrong_a = p["key_a"].reshape(-1)[assigned].reshape(2, 2)
    wrong_b = p["key_b"].reshape(-1)[assigned].reshape(2, 2)
    wrong = _paint_answer(first["owner"], first["info"], wrong_a, wrong_b)
    _require(np.any(wrong != first["output"]), "omitting B's assembled-corner assignment did not change the answer")


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, sides, controls, profiles, palettes, keys = set(), set(), set(), set(), set(), set()
    bends, reading_orders, key_sites, within_geometry = set(), set(), set(), 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p, x = _parse(pair["input"]), _reconstruct(_parse(pair["input"]))
                shapes.add(pair["input"].shape)
                sides.add(p["side"])
                controls.add((p["side"], x["info"]["cut"].tobytes()))
                profile = tuple(int(np.count_nonzero(x["owner"] == i)) for i in range(4))
                profiles.add(profile)
                local.add((p["side"], x["owner"].tobytes(), x["info"]["cut"].tobytes()))
                palettes.add((p["background"], p["carrier"], p["control"]))
                keys.add(tuple(map(int, np.concatenate((p["key_a"], p["key_b"]), axis=1).flat)))
                bends.add(x["info"]["bends"])
                key_sites.add(p["key_location"])
                positions = sorted(range(4), key=lambda i: p["pieces"][i]["offset"])
                corner_of = {i: c for c, i in enumerate(x["identities"])}
                reading_orders.add(tuple(corner_of[i] for i in positions))
                # Color-role relabeling is valid, including zero as a foreground.
                permutation = np.random.default_rng(123).permutation(10)
                _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "palette equivariance")
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list support")
        within_geometry += len(local) >= 2
    _require(len(shapes) >= 40 and sides == set(range(8, 13)), "dimension variation")
    _require(len(controls) >= 60 and len(profiles) >= 45, "structural variation")
    _require(len(palettes) >= 90 and len(keys) >= 90, "palette and key variation")
    _require({0, 1, 2} <= bends and len(key_sites) >= 8, "cut and key placement variation")
    _require(len(reading_orders) >= 15 and within_geometry == 42, "scatter and within-puzzle variation")
    _require(all(np.array_equal(a[k], b[k]) for s in puzzles[0] for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in a), "seed determinism")
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(73, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Control topology rejects branches, loops, and two parallel cuts.
    for kind in range(3):
        board = np.zeros((9, 9), dtype=bool)
        board[0, 0] = True
        if kind == 0:
            board[:, 4] = True
            board[4, 4:7] = True
        elif kind == 1:
            board[2:7, 2] = board[2:7, 6] = True
            board[2, 2:7] = board[6, 2:7] = True
        else:
            board[:, 3] = board[:, 6] = True
        _require(_control_info(board) is None, "invalid topology accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "square_sides": sorted(sides),
            "cuts": len(controls), "area_profiles": len(profiles), "reading_orders": len(reading_orders),
            "key_locations": len(key_sites), "within_puzzle_geometry_variation": within_geometry}


if __name__ == "__main__":
    print(self_test())
