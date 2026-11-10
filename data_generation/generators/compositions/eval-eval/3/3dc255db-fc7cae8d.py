"""Registered cancellation: keyed occupancy comparison followed by tip export.

Only the standard library and NumPy are used. No parent imports or fixtures,
cached examples, seeds, or construction records are used by solve().
"""
from __future__ import annotations

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
SIDE_MAPS = (
    dict(zip(SIDES, ("T", "R", "B", "L"))),
    dict(zip(SIDES, ("L", "T", "R", "B"))),
    dict(zip(SIDES, ("B", "L", "T", "R"))),
    dict(zip(SIDES, ("R", "B", "L", "T"))),
    dict(zip(SIDES, ("T", "L", "B", "R"))),
    dict(zip(SIDES, ("B", "R", "T", "L"))),
    dict(zip(SIDES, ("L", "B", "R", "T"))),
    dict(zip(SIDES, ("R", "T", "L", "B"))),
)
HOST_SEEDS = (
    0, 26164844, 150807608, 797035560, 47835174, 794647004,
    564702506, 275489843, 208691599, 124572226, 272921041,
    752287002, 397288561, 309713454, 910974511, 33795492,
    330125687, 548716354, 944320189, 899133019, 819161362,
    218929936, 759890687, 686519686, 535725861, 741509079,
    289472094, 714276040, 672857035, 877857498, 245619407,
    139535647, 323259306, 283698960, 627476743, 474368692,
    130534886, 949827944, 450345887, 779295117, 933449813,
    824460709,
)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _check_grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _side_cells(r, c, h, w, side):
    if side == "T":
        return [(r - 1, x) for x in range(c, c + w)]
    if side == "B":
        return [(r + h, x) for x in range(c, c + w)]
    if side == "L":
        return [(y, c - 1) for y in range(r, r + h)]
    return [(y, c + w) for y in range(r, r + h)]


def _point(r, c, h, w, side, offset):
    """Apex first, then two shoulders, relative to a rectangular window."""
    if side == "T":
        return ((r - 3, c + offset), (r - 2, c + offset - 1),
                (r - 2, c + offset + 1))
    if side == "B":
        return ((r + h + 2, c + offset), (r + h + 1, c + offset - 1),
                (r + h + 1, c + offset + 1))
    if side == "L":
        return ((r + offset, c - 3), (r + offset - 1, c - 2),
                (r + offset + 1, c - 2))
    return ((r + offset, c + w + 2), (r + offset - 1, c + w + 1),
            (r + offset + 1, c + w + 1))


def _ray(shape, tip, side, count):
    dr, dc = NORMALS[side]
    r, c = tip
    result = []
    for _ in range(int(count)):
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        result.append((r, c))
    return result


def _clearance(shape, tip, side):
    r, c = tip
    return {"T": r, "B": shape[0] - 1 - r,
            "L": c, "R": shape[1] - 1 - c}[side]


def _decode(input_grid):
    """Find complete rectangles from corners/sides, not constructor templates."""
    g = _check_grid(input_grid)
    H, W = g.shape
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    candidates = []
    for frame in (int(v) for v in np.unique(g) if v != bg):
        points = {tuple(int(x) for x in p) for p in np.argwhere(g == frame)}
        rows, cols = {}, {}
        for r, c in sorted(points):
            rows.setdefault(r, []).append(c)
            cols.setdefault(c, []).append(r)
        for r0, c0 in sorted(points):
            for c1 in rows[r0]:
                if c1 - c0 < 4:
                    continue
                top = g[r0, c0 + 1:c1]
                if not np.all(top == top[0]):
                    continue
                for r1 in cols[c0]:
                    if r1 - r0 < 4 or (r1, c1) not in points:
                        continue
                    r, c, h, w = r0 + 1, c0 + 1, r1 - r0 - 1, c1 - c0 - 1
                    segments = {
                        "T": top, "B": g[r1, c:c1],
                        "L": g[r:r1, c0], "R": g[r:r1, c1],
                    }
                    if any(not np.all(a == a[0]) for a in segments.values()):
                        continue
                    colors = {s: int(a[0]) for s, a in segments.items()}
                    guides = {s: v for s, v in colors.items() if v != frame}
                    if len(guides) != 2 or len(set(guides.values())) != 2:
                        continue
                    if bg in guides.values():
                        continue
                    ss = list(guides)
                    if np.dot(NORMALS[ss[0]], NORMALS[ss[1]]) != 0:
                        continue
                    mask = g[r:r + h, c:c + w] != bg
                    vals = set(int(v) for v in g[r:r + h, c:c + w][mask])
                    if len(vals) != 1 or vals & ({frame} | set(guides.values())):
                        continue
                    arrows = []
                    for side in SIDES:
                        length = w if side in ("T", "B") else h
                        for offset in range(1, length - 1):
                            head = _point(r, c, h, w, side, offset)
                            if all(1 <= y < H - 1 and 1 <= x < W - 1 and g[y, x] == frame
                                   for y, x in head):
                                arrows.append((side, head))
                    if len(arrows) != 1:
                        continue
                    side, head = arrows[0]
                    perimeter = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                    perimeter.update(p for s in SIDES for p in _side_cells(r, c, h, w, s))
                    structural = perimeter | set(head)
                    occupied = structural | {(r + y, c + x) for y, x in np.argwhere(mask)}
                    ys, xs = zip(*structural)
                    box = {(y, x) for y in range(min(ys), max(ys) + 1)
                           for x in range(min(xs), max(xs) + 1)}
                    candidates.append(dict(bounds=(r, c, h, w), guides=guides,
                        frame=frame, token=next(iter(vals)), mask=mask.copy(),
                        side=side, tip=head[0], structural=structural,
                        occupied=occupied, box=box))
    assert 2 <= len(candidates) <= 4, "input needs two to four unique complete wells"
    wells = sorted(candidates, key=lambda o: o["bounds"])
    assert len({o["bounds"] for o in wells}) == len(wells)
    frames = {o["frame"] for o in wells}
    keys = {frozenset(o["guides"].values()) for o in wells}
    assert len(frames) == len(keys) == 1, "wells must share frame and guide identities"
    reserved = {bg} | frames | set(wells[0]["guides"].values())
    assert all(o["token"] not in reserved for o in wells)
    explained, previous_boxes = set(), set()
    for o in wells:
        expanded = {(y + dy, x + dx) for y, x in o["box"]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        assert expanded.isdisjoint(previous_boxes), "well bounding boxes need a background moat"
        previous_boxes.update(o["box"])
        assert explained.isdisjoint(o["occupied"])
        explained.update(o["occupied"])
    assert explained == {tuple(int(x) for x in p) for p in np.argwhere(g != bg)}, "unassigned ink"
    return dict(grid=g, background=bg, wells=wells)


def _matrix(source, target):
    """Signed image axes are derived from the two visible identities."""
    m = np.zeros((2, 2), dtype=int)
    for side, color in source.items():
        dest = next(s for s, v in target.items() if v == color)
        m += np.outer(NORMALS[dest], NORMALS[side])
    assert np.array_equal(m.T @ m, np.eye(2, dtype=int))
    return m


def _move_mask(mask, matrix):
    # No named D4 transform helper is needed by the prediction reader.
    locations = np.indices(mask.shape).reshape(2, -1)
    mapped = matrix @ locations
    mapped -= mapped.min(axis=1, keepdims=True)
    shape = tuple(int(n) for n in mapped.max(axis=1) + 1)
    out = np.zeros(shape, dtype=bool)
    out[mapped[0], mapped[1]] = mask.ravel()
    return out


def _registered(info, reference=0, *, tight=False, ignore_keys=False):
    wells = info["wells"]
    target = wells[reference]["guides"]
    aligned = []
    for well in wells:
        mask = well["mask"]
        if tight:
            ys, xs = np.where(mask)
            mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if ignore_keys:
            registered = mask.copy()
        else:
            registered = _move_mask(mask, _matrix(well["guides"], target))
        aligned.append(registered)
    if not (tight or ignore_keys):
        assert len({a.shape for a in aligned}) == 1, "incompatible registration domains"
    # Padding is used only to formalize the two deliberately wrong alternatives.
    height = max(a.shape[0] for a in aligned)
    width = max(a.shape[1] for a in aligned)
    return [np.pad(a, ((0, height - a.shape[0]), (0, width - a.shape[1]))) for a in aligned]


def _quantities(masks):
    total = np.sum(masks, axis=0)
    return [int(np.count_nonzero(m & (total == 1))) for m in masks]


def _paint_answer(info, quantities, *, check_paths=True):
    out = info["grid"].copy()
    rays = set()
    for well in info["wells"]:
        r, c, h, w = well["bounds"]
        out[r:r + h, c:c + w] = info["background"]
    for well, quantity in zip(info["wells"], quantities):
        cells = _ray(out.shape, well["tip"], well["side"], quantity)
        if check_paths:
            assert all(info["grid"][p] == info["background"] for p in cells), "ray obstruction"
            assert rays.isdisjoint(cells), "ray intersection"
        rays.update(cells)
        for p in cells:
            out[p] = well["token"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all ownership, correspondence, quantity and geometry from input."""
    info = _decode(input_grid)
    return _paint_answer(info, _quantities(_registered(info)))


def _components(mask):
    cells = {tuple(int(x) for x in p) for p in np.argwhere(mask)}
    count = 0
    while cells:
        count += 1
        stack = [min(cells)]
        cells.remove(stack[0])
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (r + dr, c + dc)
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
    return count


def _training_evidence(pairs, information):
    full_counts, triple, components, cross_color, tight_counter = set(), False, False, False, False
    raw_counter, ignored_counter = False, False
    for pair, info in zip(pairs, information):
        masks = _registered(info)
        total = np.sum(masks, axis=0)
        quantities = _quantities(masks)
        triple |= bool(np.any(total >= 3))
        exclusive = [m & (total == 1) for m in masks]
        for i, (well, quantity) in enumerate(zip(info["wells"], quantities)):
            room = _clearance(info["grid"].shape, well["tip"], well["side"])
            if room > quantity and quantity > 0:
                full_counts.add(quantity)
                components |= _components(exclusive[i]) < quantity
                raw_counter |= int(well["mask"].sum()) > quantity
        same_color_counts = []
        for i, well in enumerate(info["wells"]):
            peer = np.zeros(masks[i].shape, dtype=bool)
            for j, other in enumerate(info["wells"]):
                if j != i and other["token"] == well["token"]:
                    peer |= masks[j]
            same_color_counts.append(int(np.count_nonzero(masks[i] & ~peer)))
        cross_color |= not np.array_equal(_paint_answer(info, same_color_counts, check_paths=False), pair["output"])
        tight_counter |= _quantities(_registered(info, tight=True)) != quantities
        ignored_counter |= not np.array_equal(_paint_answer(info,
            _quantities(_registered(info, ignore_keys=True)), check_paths=False), pair["output"])
    assert len(full_counts) >= 2 and components and raw_counter
    assert triple and cross_color and tight_counter and ignored_counter
    # The first two inputs must be a clean correspondence-only intervention.
    a, b = information[:2]
    assert a["grid"].shape == b["grid"].shape
    assert len(a["wells"]) == len(b["wells"]) == 3
    changed = []
    for i, (wa, wb) in enumerate(zip(a["wells"], b["wells"])):
        assert wa["bounds"] == wb["bounds"] and wa["tip"] == wb["tip"]
        assert wa["token"] == wb["token"] and np.array_equal(wa["mask"], wb["mask"])
        assert set(wa["guides"]) == set(wb["guides"])
        if wa["guides"] != wb["guides"]:
            changed.append(i)
            assert list(wa["guides"].values()) == list(wb["guides"].values())[::-1]
    assert len(changed) == 1
    well = a["wells"][changed[0]]
    r, c, h, w = well["bounds"]
    changed_positions = {p for s in well["guides"] for p in _side_cells(r, c, h, w, s)}
    assert changed_positions == {tuple(int(x) for x in p) for p in np.argwhere(a["grid"] != b["grid"])}
    assert _quantities(_registered(a)) != _quantities(_registered(b)), "guide swap must change counts"
    assert not np.array_equal(pairs[0]["output"], pairs[1]["output"])
    relative = [_matrix(x["guides"], y["guides"])
                for x in a["wells"] for y in a["wells"]]
    assert any(round(np.linalg.det(m)) == 1 and np.trace(m) == 0 for m in relative)
    assert any(round(np.linalg.det(m)) == -1 for m in relative)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            masks = _registered(info)
            counts = _quantities(masks)
            assert np.array_equal(_paint_answer(info, counts), out), "wrong erasure, preservation or ray"
            # Comparison must not depend on an arbitrary first well.
            assert all(_quantities(_registered(info, i)) == counts for i in range(len(masks)))
            assert sum(counts) == int(np.count_nonzero(np.sum(masks, axis=0) == 1))
            for well in info["wells"]:
                r, c, h, w = well["bounds"]
                assert np.all(out[r:r + h, c:c + w] == info["background"])
                assert all(inp[p] == out[p] for p in well["structural"])
            information[split].append(info)
    _training_evidence(puzzle["train"], information["train"])
    assert all(len(i["wells"]) <= 3 for i in information["train"])
    assert len(information["test"][0]["wells"]) == 4
    first = information["test"][0]
    assert any(q >= 2 and _clearance(first["grid"].shape, o["tip"], o["side"]) > q
               for o, q in zip(first["wells"], _quantities(_registered(first))))


# Everything below this line is construction and testing; solve never calls it.
def _transform(array, index):
    if index < 4:
        return np.rot90(array, index).copy()
    return (np.fliplr(array), np.flipud(array), array.T, np.rot90(array.T, 2))[index - 4].copy()


def _palette(rng):
    return tuple(int(v) for v in rng.permutation(10))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _grow(rng, cells, count):
    available = set(cells)
    selected = set()
    for _ in range(count):
        frontier = sorted(p for p in available if not selected or any(
            abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in selected))
        if not frontier:
            return None
        p = _choose(rng, frontier)
        selected.add(p)
        available.remove(p)
    return selected


def _contrast_patterns(rng, size):
    """Three-way suppression, connected uniques, and a key intervention."""
    cells = [(r, c) for r in range(1, size) for c in range(size)]
    for _ in range(240):
        order = [cells[int(i)] for i in rng.permutation(len(cells))]
        shared = order[:int(rng.integers(1, 4))]
        left = order[len(shared):]
        masks = [np.zeros((size, size), dtype=bool) for _ in range(3)]
        for p in shared:
            for mask in masks:
                mask[p] = True
        for i, j in ((0, 1), (1, 2), (0, 2)):
            p = left.pop()
            masks[i][p] = masks[j][p] = True
        low = int(rng.integers(2, 4))
        desired = [low, low + 1, low + 2]
        rng.shuffle(desired)
        failed = False
        for i, count in enumerate(desired):
            chosen = _grow(rng, left, count) if i == 0 else set(left[:count])
            if chosen is None or len(chosen) != count:
                failed = True
                break
            for p in sorted(chosen):
                masks[i][p] = True
                left.remove(p)
        if failed:
            continue
        swap = int(rng.integers(3))
        second = [m.copy() for m in masks]
        second[swap] = second[swap].T.copy()
        q0, q1 = _quantities(masks), _quantities(second)
        if q0 == q1 or max(q1) > 8 or len({q for q in q1 if q > 0}) < 2:
            continue
        # Trimming each registered mask shifts independent origins. Force an
        # observable answer difference under that concrete wrong convention.
        def cropped_counts(patterns):
            crops = []
            for m in patterns:
                rr, cc = np.where(m)
                crops.append(m[rr.min():rr.max() + 1, cc.min():cc.max() + 1])
            H, W = max(m.shape[0] for m in crops), max(m.shape[1] for m in crops)
            return _quantities([np.pad(m, ((0, H - m.shape[0]), (0, W - m.shape[1]))) for m in crops])
        if cropped_counts(masks) == q0 and cropped_counts(second) == q1:
            continue
        return masks, swap, q0, q1
    raise AssertionError("contrast pattern search exhausted")


def _ordinary_patterns(rng, n, shape, primary_free):
    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    if min(shape) >= 4 and rng.random() < .35:
        side = _choose(rng, SIDES)
        cells = [p for p in cells if not (
            side == "T" and p[0] == 0 or side == "B" and p[0] == shape[0] - 1 or
            side == "L" and p[1] == 0 or side == "R" and p[1] == shape[1] - 1)]
    masks = [np.zeros(shape, dtype=bool) for _ in range(n)]
    rng.shuffle(cells)
    # One common position keeps every well nonempty even with zero exclusives.
    common = cells.pop()
    for mask in masks:
        mask[common] = True
    for _ in range(int(rng.integers(0, min(6, max(1, len(cells) // 3))))):
        p = cells.pop()
        group = rng.choice(n, size=int(rng.integers(2, n + 1)), replace=False)
        for i in group:
            masks[int(i)][p] = True
    if primary_free:
        for _ in range(2):
            masks[0][cells.pop()] = True
    budget = int(rng.integers(1, min(len(cells), 5 * n) + 1)) if cells else 0
    for _ in range(budget):
        candidates = [i for i, m in enumerate(masks) if _quantities(masks)[i] < 6]
        if not candidates or not cells:
            break
        masks[_choose(rng, candidates)][cells.pop()] = True
    return masks


def _local_well(mask, transform, side, offset, palette, token, reserve, mode):
    local = _transform(mask, transform)
    h, w = local.shape
    bg, frame, m0, m1 = palette[:4]
    guides = {SIDE_MAPS[transform]["T"]: m0, SIDE_MAPS[transform]["L"]: m1}
    head = _point(0, 0, h, w, side, offset)
    corners = ((-1, -1), (-1, w), (h, -1), (h, w))
    structural = {p: frame for p in corners}
    for s in SIDES:
        structural.update({p: guides.get(s, frame) for p in _side_cells(0, 0, h, w, s)})
    structural.update({p: frame for p in head})
    min_r = min(p[0] for p in structural)
    min_c = min(p[1] for p in structural)
    max_r = max(p[0] for p in structural)
    max_c = max(p[1] for p in structural)
    return dict(mask=local, structural=structural, tip=head[0], side=side,
                bounds=(0, 0, h, w), low=(min_r, min_c), high=(max_r, max_c),
                token=token, reserve=reserve, mode=mode, guides=guides)


def _pack(rng, masks, transforms, palette, tokens, reserves, modes, small=False):
    n = len(masks)
    # Independent canvas axes and arbitrary placements, with bounded roomy retry.
    for attempt in range(45):
        low = 18 if n == 2 and small else 25 if n == 3 else 26
        H = int(rng.integers(low, 31)) if attempt < 12 else 30
        W = int(rng.integers(low, 31)) if attempt < 12 else 30
        placed, boxes, guards, ray_guards = [], set(), set(), set()
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            variants = []
            h, w = _transform(masks[i], transforms[i]).shape
            for side in SIDES:
                length = w if side in ("T", "B") else h
                for offset in range(1, length - 1):
                    spec = _local_well(masks[i], transforms[i], side, offset, palette,
                                       tokens[i], reserves[i], modes[i])
                    lr, lc = spec["low"]
                    hr, hc = spec["high"]
                    for r in range(1 - lr, H - 1 - hr):
                        for c in range(1 - lc, W - 1 - hc):
                            tip = (r + spec["tip"][0], c + spec["tip"][1])
                            room = _clearance((H, W), tip, side)
                            q, mode = reserves[i], modes[i]
                            if mode == "free" and room <= q:
                                continue
                            if mode == "exact" and room != q:
                                continue
                            if mode == "clip" and not 0 < room < q:
                                continue
                            variants.append((spec, r, c))
            accepted = None
            for index in rng.permutation(len(variants)):
                spec, r, c = variants[int(index)]
                lr, lc = spec["low"]
                hr, hc = spec["high"]
                box = {(y, x) for y in range(r + lr, r + hr + 1)
                       for x in range(c + lc, c + hc + 1)}
                if not box.isdisjoint(guards | ray_guards):
                    continue
                tip = (r + spec["tip"][0], c + spec["tip"][1])
                path = set(_ray((H, W), tip, spec["side"], reserves[i] + 1))
                if not path.isdisjoint(boxes | ray_guards):
                    continue
                accepted = dict(spec, origin=(r, c), world_tip=tip, index=i)
                boxes.update(box)
                guards.update((y + dy, x + dx) for y, x in box
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                ray_guards.update(path)
                break
            if accepted is None:
                break
            placed.append(accepted)
        if len(placed) == n:
            return (H, W), sorted(placed, key=lambda s: s["index"])
    raise AssertionError("packing search exhausted")


def _render(shape, placed, palette, quantities):
    inp = np.full(shape, palette[0], dtype=np.int8)
    out = inp.copy()
    for spec in placed:
        r, c = spec["origin"]
        for (y, x), color in spec["structural"].items():
            inp[r + y, c + x] = out[r + y, c + x] = color
        for y, x in np.argwhere(spec["mask"]):
            inp[r + y, c + x] = spec["token"]
        for p in _ray(shape, spec["world_tip"], spec["side"], quantities[spec["index"]]):
            out[p] = spec["token"]
    return {"input": inp, "output": out}


def _make_contrast(rng, palette):
    size = int(rng.integers(5, 7))
    masks, swap, q0, q1 = _contrast_patterns(rng, size)
    # Include a quarter turn and a reflection relative to the first view.
    root = int(rng.integers(8))
    def composed(local):
        mapping = {s: SIDE_MAPS[root][SIDE_MAPS[local][s]] for s in SIDES}
        return next(i for i, m in enumerate(SIDE_MAPS) if m == mapping)
    transforms = [root, composed(1), composed(4)]
    reserves = [max(a, b) for a, b in zip(q0, q1)]
    shape, placed = _pack(rng, masks, transforms, palette, list(palette[4:7]), reserves, ["free"] * 3)
    first = _render(shape, placed, palette, q0)
    second = _render(shape, placed, palette, q1)
    spec = placed[swap]
    r, c = spec["origin"]
    _, _, h, w = spec["bounds"]
    for side, color in spec["guides"].items():
        replacement = palette[3] if color == palette[2] else palette[2]
        for p in _side_cells(r, c, h, w, side):
            second["input"][p] = second["output"][p] = replacement
    return first, second


def _contrast(rng, palette):
    # A tightly cropped comparison has an origin that depends on the actual
    # reference orientation. Check the witness on the rendered input, rather
    # than assuming the canonical construction test establishes every view.
    for _ in range(24):
        first, second = _make_contrast(rng, palette)
        for pair in (first, second):
            info = _decode(pair["input"])
            if _quantities(_registered(info, tight=True)) != _quantities(_registered(info)):
                return first, second
    raise AssertionError("rendered crop witness search exhausted")


def _ordinary(rng, n, palette, primary_free=False):
    high = 9 if n == 2 else 7 if n == 3 else 6
    h, w = int(rng.integers(3, high)), int(rng.integers(3, high))
    if rng.random() < .2:
        w = h
    masks = _ordinary_patterns(rng, n, (h, w), primary_free)
    quantities = _quantities(masks)
    transforms = [int(rng.integers(8)) for _ in masks]
    token_pool = list(palette[4:4 + int(rng.integers(1, 5))])
    tokens = [_choose(rng, token_pool) for _ in masks]
    modes = ["free" if q < 2 else _choose(rng, ("free", "free", "any", "exact", "clip"))
             for q in quantities]
    if primary_free:
        modes[0] = "free"
    shape, placed = _pack(rng, masks, transforms, palette, tokens, quantities, modes, small=n == 2)
    return _render(shape, placed, palette, quantities)


def _validate_count(value, lower, upper, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _validate_count(num_train, 2, 10, "num_train")
    ne = _validate_count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = _palette(rng)
    stable = bool(rng.random() < .25)
    first, second = _contrast(rng, palette)
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, number in (("train", nt - 2), ("test", ne)):
        for index in range(number):
            for _ in range(20):
                colors = palette if stable else _palette(rng)
                n = 4 if split == "test" and index == 0 else int(rng.integers(2, 4 if split == "train" else 5))
                pair = _ordinary(rng, n, colors, primary_free=split == "test" and index == 0)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("distinct input search exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed family audit, count boundaries and equivariance checks."""
    metrics = {k: set() for k in ("shapes", "domains", "directions", "matrices",
                                "backgrounds", "quantities", "occupancies", "well_counts",
                                "clearance", "token_colors", "offsets")}
    signatures, examples = set(), []
    zero, blank, shared_colors = 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        examples.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                aligned = _registered(info)
                total = np.sum(aligned, axis=0)
                quantities = _quantities(aligned)
                metrics["shapes"].add(pair["input"].shape)
                metrics["backgrounds"].add(info["background"])
                metrics["well_counts"].add(len(aligned))
                metrics["occupancies"].update(int(v) for v in np.unique(total))
                shared_colors += len({o["token"] for o in info["wells"]}) < len(aligned)
                for o, q in zip(info["wells"], quantities):
                    r, c, h, w = o["bounds"]
                    metrics["domains"].add((h, w))
                    metrics["directions"].add(o["side"])
                    metrics["matrices"].add(tuple(int(v) for v in
                        _matrix(o["guides"], info["wells"][0]["guides"]).flat))
                    metrics["quantities"].add(q)
                    metrics["token_colors"].add(o["token"])
                    room = _clearance(pair["input"].shape, o["tip"], o["side"])
                    metrics["clearance"].add("free" if room > q else "exact" if room == q else "clip")
                    metrics["offsets"].add(o["tip"][1] - c if o["side"] in ("T", "B") else o["tip"][0] - r)
                    zero += q == 0
                    blank += any(not a.any() for a in (o["mask"][0], o["mask"][-1], o["mask"][:, 0], o["mask"][:, -1]))
        if seed in HOST_SEEDS[:2]:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(metrics["shapes"]) >= 12 and len(metrics["domains"]) >= 20
    assert metrics["directions"] == set(SIDES) and len(metrics["matrices"]) == 8
    assert metrics["backgrounds"] == set(range(10)) and metrics["token_colors"] == set(range(10))
    assert metrics["well_counts"] == {2, 3, 4} and metrics["occupancies"] >= {0, 1, 2, 3, 4}
    assert metrics["clearance"] == {"free", "exact", "clip"}
    assert max(metrics["quantities"]) >= 6 and len(metrics["offsets"]) >= 4
    assert zero and blank and shared_colors
    # Rotate/reflect complete scenes and rename all colors bijectively. Even
    # transformed blank windows and keys must be parsed from the input alone.
    rng = np.random.default_rng(123456)
    color_map = rng.permutation(10)
    for pair in examples[0]["train"] + examples[0]["test"]:
        for t in range(8):
            inp = color_map[_transform(pair["input"], t)]
            expected = color_map[_transform(pair["output"], t)]
            saved = inp.copy()
            assert np.array_equal(solve(inp), expected)
            assert np.array_equal(inp, saved), "solve mutated its input"
    # The specification is broader than the sampler's 3..8-cell domains and
    # small exclusive counts. Exercise a larger rectangular domain, a transpose,
    # and quantities 30/20 that both exceed outward clearance.
    large = [np.zeros((9, 10), dtype=bool) for _ in range(2)]
    large[0][:5, :8] = True
    large[1][3:8, 3:9] = True
    large_counts = _quantities(large)
    assert large_counts == [30, 20]
    large_palette = tuple(range(10))
    placed = []
    for i, (t, origin, offset) in enumerate(((0, (14, 2), 3), (6, (15, 16), 5))):
        spec = _local_well(large[i], t, "T", offset, large_palette, 4 + i,
                           large_counts[i], "clip")
        tip = (origin[0] + spec["tip"][0], origin[1] + spec["tip"][1])
        placed.append(dict(spec, origin=origin, world_tip=tip, index=i))
    large_pair = _render((30, 30), placed, large_palette, large_counts)
    assert np.array_equal(solve(large_pair["input"]), large_pair["output"])
    assert _quantities(_registered(_decode(large_pair["input"]))) == large_counts
    # Every legal explicit count, including the minimum, is covered in a bounded
    # sweep. Each call verifies the family witnesses as well as output labels.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(20261008 + a * 11 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Wrong labels and a broken complete guide cannot pass family validation.
    for mutation in ("output", "guide"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in examples[0][s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        if mutation == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        else:
            info = _decode(pair["input"])
            well = info["wells"][0]
            side = next(iter(well["guides"]))
            line = _side_cells(*well["bounds"], side)
            pair["input"][line[len(line) // 2]] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted puzzle accepted")
    return {"seeds": len(HOST_SEEDS), "distinct_puzzles": len(signatures),
            "metrics": {k: sorted(v) for k, v in metrics.items()},
            "zero_survivor_wells": zero, "blank_edge_wells": blank,
            "scenes_with_shared_token_colors": shared_colors,
            "semantic_verification": "generator-owned invariants and equivariance only"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
