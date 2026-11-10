"""Contact printing: the front sheet's tiled stencil, the next sheet's ink.

All solving conventions are visible or fixed by this module's family rule.
Generation constructs labels from geometry; solve parses only its input.
Dependencies: Python standard library and NumPy.
"""

import json
from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 to 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 to 4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, todo = [], [start]
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        components.append(cells)
    return components


def _box(cells):
    a = np.asarray(cells)
    return int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max())+1, int(a[:, 1].max())+1


def _inside(r, c, b):
    return b[0] <= r < b[2] and b[1] <= c < b[3]


def _intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _separated(a, b):
    # Half-open boxes with at least one blank-cell row or column between them.
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _markers(b, diagonal):
    r, c, e, d = b
    if diagonal == 0:
        return {(r, c), (r, c+1), (r+1, c), (e-1, d-1), (e-1, d-2), (e-2, d-1)}
    return {(r, d-1), (r, d-2), (r+1, d-1), (e-1, c), (e-1, c+1), (e-2, c)}


def _draw_frame(a, b, color):
    r, c, e, d = b
    a[r, c:d] = a[e-1, c:d] = color
    a[r:e, c] = a[r:e, d-1] = color


def _outline(cells):
    b = _box(cells)
    r, c, e, d = b
    if e-r < 3 or d-c < 3 or len(cells) != 2*(e-r)+2*(d-c)-4:
        return None
    if all(rr in (r, e-1) or cc in (c, d-1) for rr, cc in cells):
        return b
    return None


def _parse(value):
    a = _grid(value)
    histogram = np.bincount(a.ravel(), minlength=10)
    bg = int(histogram.argmax())
    assert np.count_nonzero(histogram == histogram[bg]) == 1, "ambiguous background"
    frames = []
    outlines = {}
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        shapes = [b for cells in _components(a == color) if (b := _outline(cells)) is not None]
        outlines[color] = shapes
        for b in shapes:
            if b[2]-b[0] >= 8 and b[3]-b[1] >= 8:
                frames.append((color, b))
    assert len(frames) == 1, "need one large complete frame"
    fc, board_box = frames[0]
    panels = [b for b in outlines[fc] if b[3]-b[1] == 4 and 4 <= b[2]-b[0] <= 7]
    assert len(panels) == 1, "need one framed two-column table"
    panel_box = panels[0]
    assert _separated(board_box, panel_box)
    border_count = sum(2*(b[2]-b[0])+2*(b[3]-b[1])-4 for b in (board_box, panel_box))
    assert int(np.count_nonzero(a == fc)) == border_count, "frame color is exclusive"
    pr, pc, pe, pd = panel_box
    entries = a[pr+1:pe-1, pc+1:pd-1]
    cues = list(map(int, entries[:, 0]))
    inks = list(map(int, entries[:, 1]))
    assert 2 <= len(cues) <= 5 and len(set(cues)) == len(cues)
    assert not set(cues) & {bg, fc} and fc not in inks
    br, bc, be, bd = board_box
    board = a[br+1:be-1, bc+1:bd-1]
    assert np.all(np.isin(board, [bg]+cues)), "unknown board object"
    outside = np.ones(a.shape, dtype=bool)
    outside[br:be, bc:bd] = False
    outside[pr:pe, pc:pd] = False
    assert set(map(int, np.unique(a[outside & (a != bg)]))) == set(cues)
    sheets, stencil_boxes, sizes = [], [], set()
    for source, ink in zip(cues, inks):
        cells = set(map(tuple, np.argwhere(board == source)))
        assert len(cells) == 6, "one opposite-L pair per cue"
        b = _box(list(cells))
        hh, ww = b[2]-b[0], b[3]-b[1]
        assert min(hh, ww) >= 2 and (min(hh, ww) != 2 or max(hh, ww) >= 4)
        diagonals = [q for q in (0, 1) if cells == _markers(b, q)]
        assert len(diagonals) == 1, "markers must be opposing L fragments"
        stencil_cells = np.argwhere(outside & (a == source))
        assert len(stencil_cells)
        sb = _box(stencil_cells)
        r, c, e, d = sb
        n = e-r
        assert n == d-c and 2 <= n <= 5
        assert _separated(sb, board_box) and _separated(sb, panel_box)
        assert all(_separated(sb, old) for old in stencil_boxes)
        tile = a[r:e, c:d]
        assert np.all(np.isin(tile, [bg, source]))
        mask = tile == source
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        sizes.add(n)
        stencil_boxes.append(sb)
        sheets.append(dict(source=source, ink=ink, bounds=b, markers=cells,
                           diagonal=diagonals[0], mask=mask.copy(), stencil_box=sb))
    assert len(sizes) == 1, "common stencil size required"
    return dict(background=bg, frame=fc, board_box=board_box, panel_box=panel_box,
                sheets=sheets, n=next(iter(sizes)), shape=board.shape)


def _blank(shape, bg, fc):
    h, w = shape
    output = np.full((h+2, w+2), bg, dtype=np.int8)
    _draw_frame(output, (0, 0, h+2, w+2), fc)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the contact print solely from the visible grid; never mutate it."""
    p = _parse(input_grid)
    output = _blank(p["shape"], p["background"], p["frame"])
    n = p["n"]
    # Per-cell top-two selection, separate from the label constructor below.
    for r in range(p["shape"][0]):
        for c in range(p["shape"][1]):
            covering = [s for s in p["sheets"] if _inside(r, c, s["bounds"])]
            if len(covering) >= 2:
                upper, receiver = covering[:2]
                r0, c0 = upper["bounds"][:2]
                if upper["mask"][(r-r0) % n, (c-c0) % n]:
                    output[r+1, c+1] = receiver["ink"]
    return output


def _construct(shape, bg, fc, sheets):
    """Label construction from sampled sheets, via two accumulated rank fields."""
    top = np.full(shape, -1, dtype=int)
    second = top.copy()
    for i in reversed(range(len(sheets))):
        r, c, e, d = sheets[i]["bounds"]
        second[r:e, c:d] = top[r:e, c:d]
        top[r:e, c:d] = i
    output = _blank(shape, bg, fc)
    interior = output[1:-1, 1:-1]
    ys, xs = np.indices(shape)
    for i, upper in enumerate(sheets):
        mask = upper["mask"]
        n = len(mask)
        r, c = upper["bounds"][:2]
        active = (top == i) & (second >= 0) & mask[(ys-r) % n, (xs-c) % n]
        for j, receiver in enumerate(sheets):
            interior[active & (second == j)] = receiver["ink"]
    return output


def _alternative(p, kind):
    out = _blank(p["shape"], p["background"], p["frame"])
    n, bg = p["n"], p["background"]
    for r in range(p["shape"][0]):
        for c in range(p["shape"][1]):
            stack = [s for s in p["sheets"] if _inside(r, c, s["bounds"])]
            if kind == "reverse":
                stack.reverse()
            if not stack or (len(stack) < 2 and kind != "independent"):
                continue
            upper = stack[0]
            receiver = stack[1] if len(stack) >= 2 else upper
            r0, c0, e, d = upper["bounds"]
            mr, mc = (r-r0) % n, (c-c0) % n
            mask = upper["mask"]
            if kind == "solid":
                active = True
            elif kind == "receiver_mask":
                rr, cc = receiver["bounds"][:2]
                active = receiver["mask"][(r-rr) % n, (c-cc) % n]
            elif kind == "global":
                active = mask[r % n, c % n]
            elif kind == "stretch":
                active = mask[(r-r0)*n//(e-r0), (c-c0)*n//(d-c0)]
            elif kind == "rotate":
                active = np.rot90(mask)[mr, mc]
            elif kind == "reflect":
                active = np.fliplr(mask)[mr, mc]
            else:
                active = mask[mr, mc]
            if kind == "transparent" and not active and len(stack) >= 3:
                upper, receiver = stack[1:3]
                rr, cc = upper["bounds"][:2]
                active = upper["mask"][(r-rr) % n, (c-cc) % n]
            ink = upper["ink"] if kind == "own_ink" else receiver["ink"]
            out[r+1, c+1] = ink if active else bg
    return out


def _witnesses(p, output):
    found = {kind for kind in ("solid", "independent", "own_ink", "receiver_mask", "reverse",
                              "transparent", "global", "stretch", "rotate", "reflect")
             if not np.array_equal(output, _alternative(p, kind))}
    # A single front sheet visibly takes dye from different local receivers.
    receivers = {}
    n = p["n"]
    for r in range(p["shape"][0]):
        for c in range(p["shape"][1]):
            stack = [i for i, s in enumerate(p["sheets"]) if _inside(r, c, s["bounds"])]
            if len(stack) >= 2:
                i, j = stack[:2]
                upper = p["sheets"][i]
                rr, cc = upper["bounds"][:2]
                ink = p["sheets"][j]["ink"]
                if upper["mask"][(r-rr) % n, (c-cc) % n] and ink != p["background"]:
                    receivers.setdefault(i, set()).add(ink)
    if any(len(inks) >= 2 for inks in receivers.values()):
        found.add("receiver_switch")
    return found


def _asymmetric(mask):
    return all(not np.array_equal(mask, x) for x in
               [np.rot90(mask, q) for q in (1, 2, 3)]
               + [np.rot90(np.fliplr(mask), q) for q in range(4)])


def _stencil(rng, n, informative=False):
    for _ in range(150):
        y, x = np.indices((n, n))
        kind = int(rng.integers(8)) if not informative else 0
        if kind == 0:
            mask = rng.random((n, n)) < rng.uniform(.2, .88)
        elif kind == 1:
            mask = (x == 0) | (x == n-1) | (y == 0) | (y == n-1)
        elif kind == 2:
            mask = (x == n//2) | (y == n//2)
        elif kind == 3:
            mask = (x+y) % 2 == 0
        elif kind == 4:
            mask = (x == y) | ((x+y == n-1) if rng.random() < .5 else False)
        elif kind == 5:
            mask = (y % 2 == 0) | (y == n-1)
        elif kind == 6:
            mask = rng.random((n, n)) < .9
        else:
            mask = np.ones((n, n), dtype=bool)
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        if informative and not _asymmetric(mask):
            continue
        return np.rot90(mask, int(rng.integers(4))).copy()
    raise RuntimeError("could not construct an informative stencil")


def _random_box(rng, h, w, thin=False, region=None):
    rr, cc, ee, dd = region or (0, 0, h, w)
    ah, aw = ee-rr, dd-cc
    if thin and rng.random() < .5:
        hh, ww = 2, int(rng.integers(4, aw+1))
    elif thin:
        hh, ww = int(rng.integers(4, ah+1)), 2
    else:
        hh, ww = int(rng.integers(3, ah+1)), int(rng.integers(3, aw+1))
    r = rr+int(rng.integers(ah-hh+1))
    c = cc+int(rng.integers(aw-ww+1))
    return r, c, r+hh, c+ww


def _geometry(rng, h, w, k, family, thin):
    sheets, used = [], set()
    for i in range(k):
        placed = False
        for _ in range(60):
            is_thin = thin and i == 0
            region = None
            if family == "groups":
                cut = w//2
                region = (0, 0, h, cut) if i < 2 else (0, cut+1, h, w)
            elif family == "nested" and i:
                region = sheets[0]["bounds"]
            if region is not None and not is_thin and min(region[2]-region[0], region[3]-region[1]) < 3:
                return None
            if family == "cross":
                nv = k//2
                if i < nv:
                    c = 2+i*4
                    b = (0, c, h, c+2)
                else:
                    r = 2+(i-nv)*3
                    b = (r, 0, r+2, w)
                if b[2] > h or b[3] > w:
                    return None
            else:
                b = _random_box(rng, h, w, is_thin, region)
            if any(b == s["bounds"] for s in sheets):
                continue
            if family == "chain" and i:
                if not _intersects(b, sheets[-1]["bounds"]) or any(
                        _intersects(b, s["bounds"]) for s in sheets[:-1]):
                    continue
            if family == "groups" and i in (1, 3) and not _intersects(b, sheets[-1]["bounds"]):
                continue
            diagonal = int(rng.integers(2))
            marks = _markers(b, diagonal)
            if marks & used:
                continue
            sheets.append(dict(bounds=b, markers=marks, diagonal=diagonal))
            used.update(marks)
            placed = True
            break
        if not placed:
            return None
    return [sheets[int(i)] for i in rng.permutation(k)]


def _fallback_geometry(rng, role):
    boxes = [(1, 1, 12, 11), (4, 5, 15, 15)]
    if role == "three":
        boxes.append((2, 8, 14, 17))
    used, sheets = set(), []
    for b in boxes:
        for diagonal in map(int, rng.permutation(2)):
            marks = _markers(b, diagonal)
            if not marks & used:
                break
        else:
            raise AssertionError("fallback marker collision")
        sheets.append(dict(bounds=b, markers=marks, diagonal=diagonal))
        used.update(marks)
    return (16, 18), sheets


def _decorate(rng, geometry, n, teaching):
    colors = list(map(int, rng.permutation(10)))
    bg, fc = colors[:2]
    palette = colors[2:]
    k = len(geometry)
    cues = palette[:k]
    if teaching:
        inks = list(map(int, rng.choice(palette, k, replace=False)))
    else:
        kind = int(rng.integers(5))
        if kind == 0:
            order = list(map(int, rng.permutation(cues)))
            mapping = {c: order[(i+1) % k] for i, c in enumerate(order)}
            inks = [mapping[c] for c in cues]
        elif kind == 1:
            inks = cues[1:]+[palette[k]]
        elif kind == 2:
            inks = [int(rng.choice([bg]+palette))]*k
        else:
            inks = list(map(int, rng.choice([bg]+palette, k)))
    sheets = []
    for i, (g, source, ink) in enumerate(zip(geometry, cues, inks)):
        if i and not teaching and rng.random() < .16:
            mask = sheets[int(rng.integers(i))]["mask"].copy()
        else:
            mask = _stencil(rng, n, informative=teaching and i == 0)
        sheets.append(dict(g, source=source, ink=ink, mask=mask))
    return bg, fc, sheets


def _available(occupied, hh, ww):
    if hh > occupied.shape[0] or ww > occupied.shape[1]:
        return np.empty((0, 2), int)
    summed = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[hh:, ww:] - summed[:-hh, ww:] - summed[hh:, :-ww] + summed[:-hh, :-ww]
    return np.argwhere(totals == 0)


def _reserve(a, b):
    r, c, e, d = b
    a[max(0, r-1):min(len(a), e+1), max(0, c-1):min(a.shape[1], d+1)] = True


def _pack(rng, shape, k, n):
    bh, bw = shape[0]+2, shape[1]+2
    ph = k+2
    for attempt in range(12):
        occupied = np.zeros((30, 30), bool)
        if attempt % 3:
            r, c = int(rng.choice([0, 30-bh])), int(rng.choice([0, 30-bw]))
        else:
            r, c = int(rng.integers(31-bh)), int(rng.integers(31-bw))
        board_box = (r, c, r+bh, c+bw)
        _reserve(occupied, board_box)
        boxes = {}
        objects = [("panel", ph, 4)] + [(i, n, n) for i in rng.permutation(k)]
        if rng.random() < .5:
            objects = sorted(objects, key=lambda item: -item[1]*item[2])
        for label, hh, ww in objects:
            candidates = _available(occupied, hh, ww)
            if not len(candidates):
                break
            rr, cc = map(int, candidates[int(rng.integers(len(candidates)))])
            box = (rr, cc, rr+hh, cc+ww)
            boxes[label] = box
            _reserve(occupied, box)
        else:
            return (30, 30), board_box, boxes["panel"], [boxes[i] for i in range(k)]
    # Guaranteed shelf packing, reflected by positions without rotating clues.
    height = n+1+bh
    width = max(k*(n+1)-1, bw+5)
    height += int(rng.integers(31-height))
    width += int(rng.integers(31-width))
    board_box = (n+1, 0, n+1+bh, bw)
    rr = n+1+int(rng.integers(bh-ph+1))
    panel_box = (rr, bw+1, rr+ph, bw+5)
    slack = width-(k*(n+1)-1)
    gaps = rng.multinomial(slack, np.full(k+1, 1/(k+1)))
    boxes, c = {}, int(gaps[0])
    for i, index in enumerate(rng.permutation(k)):
        boxes[int(index)] = (0, c, n, c+n)
        c += n+1+int(gaps[i+1])
    flip_y, flip_x = bool(rng.integers(2)), bool(rng.integers(2))
    def move(b):
        r, c, e, d = b
        if flip_y:
            r, e = height-e, height-r
        if flip_x:
            c, d = width-d, width-c
        return r, c, e, d
    return (height, width), move(board_box), move(panel_box), [move(boxes[i]) for i in range(k)]


def _assemble(rng, shape, bg, fc, sheets):
    k, n = len(sheets), len(sheets[0]["mask"])
    canvas_shape, board, panel, tiles = _pack(rng, shape, k, n)
    a = np.full(canvas_shape, bg, dtype=np.int8)
    _draw_frame(a, board, fc)
    _draw_frame(a, panel, fc)
    br, bc = board[:2]
    pr, pc = panel[:2]
    for i, (s, b) in enumerate(zip(sheets, tiles)):
        for r, c in s["markers"]:
            a[br+1+r, bc+1+c] = s["source"]
        a[pr+1+i, pc+1:pc+3] = s["source"], s["ink"]
        r, c, e, d = b
        a[r:e, c:d][s["mask"]] = s["source"]
    # Independently retain zero to two cells of spare canvas on every side.
    r, c, e, d = _box(np.argwhere(a != bg))
    r, c = max(0, r-int(rng.integers(3))), max(0, c-int(rng.integers(3)))
    e, d = min(len(a), e+int(rng.integers(3))), min(a.shape[1], d+int(rng.integers(3)))
    return a[r:e, c:d].copy()


_TWO_WITNESSES = {"solid", "independent", "own_ink", "receiver_mask", "reverse",
                  "global", "stretch", "rotate", "reflect"}
_THREE_WITNESSES = {"solid", "transparent", "receiver_switch"}


def _make_pair(rng, mode, split, role):
    teaching = role in ("two", "three")
    if teaching:
        k = 2 if role == "two" else 3
        n = 3 if mode == "period" else int(rng.integers(3, 5))
    else:
        k = int(rng.integers(4, 6)) if mode == "count" and split == "test" else int(
            rng.integers(2, 4 if mode in ("count", "period") else 6))
        n = int(rng.integers(4, 6)) if mode == "period" and split == "test" else int(
            rng.integers(2, 4 if mode == "period" else 5))
    thin = mode == "thin" and split == "test"
    required = _TWO_WITNESSES if role == "two" else _THREE_WITNESSES if role == "three" else set()
    for attempt in range(320):
        if teaching and attempt >= 260:
            shape, geometry = _fallback_geometry(rng, role)
        else:
            shape = int(rng.integers(10, 21)), int(rng.integers(10, 22))
            families = ["free", "free", "nested", "chain"]
            if not teaching and k >= 4:
                families += ["groups"]
            if not teaching and (mode != "thin" or split == "test"):
                families += ["cross"]
            if role == "three":
                families = ["free", "free", "nested"]
            family = str(rng.choice(families))
            geometry = _geometry(rng, *shape, k, family, thin)
            if geometry is None:
                continue
        bg, fc, sheets = _decorate(rng, geometry, n, teaching)
        output = _construct(shape, bg, fc, sheets)
        if not np.any(output[1:-1, 1:-1] != bg):
            continue
        p = dict(shape=shape, background=bg, frame=fc, sheets=sheets, n=n)
        if required and not required <= _witnesses(p, output):
            continue
        # All generated test cases expose both contact and the stencil holes.
        if split == "test" and "solid" not in _witnesses(p, output):
            continue
        a = _assemble(rng, shape, bg, fc, sheets)
        parsed = _parse(a)
        assert parsed["background"] == bg
        assert np.array_equal(solve(a), output)
        return dict(input=a, output=output)
    raise RuntimeError("bounded contact-print construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle with a visible teaching pair of each kind."""
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["count", "period", "thin"]))
    roles = ["two", "three"]+["free"]*(num_train-2)
    rng.shuffle(roles)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(12):
                pair = _make_pair(rng, mode, split, roles[i] if split == "train" else "free")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    sheets = p["sheets"]
    dims = [(s["bounds"][2]-s["bounds"][0], s["bounds"][3]-s["bounds"][1]) for s in sheets]
    overlap = [(i, j) for i, a in enumerate(sheets) for j, b in enumerate(sheets)
               if i < j and _intersects(a["bounds"], b["bounds"])]
    common = max(s["bounds"][0] for s in sheets) < min(s["bounds"][2] for s in sheets) and \
             max(s["bounds"][1] for s in sheets) < min(s["bounds"][3] for s in sheets)
    degrees = [sum(i in edge for edge in overlap) for i in range(len(sheets))]
    return dict(k=len(sheets), n=p["n"], shape=p["shape"], thin=any(min(d) == 2 for d in dims),
                thin_horizontal=any(h == 2 for h, w in dims), thin_vertical=any(w == 2 for h, w in dims),
                common=bool(common), overlap_edges=len(overlap), isolated=0 in degrees,
                disconnected=sum(len(_components(s["mask"])) > 1 for s in sheets),
                asymmetric=sum(_asymmetric(s["mask"]) for s in sheets),
                solid=sum(s["mask"].all() for s in sheets),
                shared_geometry=len(sheets)-len({s["mask"].tobytes() for s in sheets}),
                ink_overlap=sum(s["ink"] in {q["source"] for q in sheets} for s in sheets),
                background_ink=any(s["ink"] == p["background"] for s in sheets),
                diagonal_set=tuple(sorted({s["diagonal"] for s in sheets})),
                backgrounds=p["background"], frames=p["frame"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    witnesses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            p = _parse(a)
            expected = solve(a)
            assert np.array_equal(expected, output), "incorrect contact print"
            assert np.any(output[1:-1, 1:-1] != p["background"]), "empty print"
            assert output.shape == (p["shape"][0]+2, p["shape"][1]+2)
            assert np.all(output[0] == p["frame"]) and np.all(output[-1] == p["frame"])
            assert np.all(output[:, 0] == p["frame"]) and np.all(output[:, -1] == p["frame"])
            assert set(map(int, np.unique(output))) <= set(map(int, np.unique(a)))
            analyses[split].append(p)
            w = _witnesses(p, output)
            if split == "train":
                witnesses.append((p, w))
            else:
                assert "solid" in w, "test must show stencil holes"
    assert any(len(p["sheets"]) == 2 and _TWO_WITNESSES <= w and
               any(_asymmetric(s["mask"]) for s in p["sheets"]) for p, w in witnesses)
    assert any(len(p["sheets"]) >= 3 and _THREE_WITNESSES <= w for p, w in witnesses)
    tr = [_features(p) for p in analyses["train"]]
    te = [_features(p) for p in analyses["test"]]
    count_shift = max(f["k"] for f in tr) < min(f["k"] for f in te)
    period_shift = max(f["n"] for f in tr) < min(f["n"] for f in te)
    thin_shift = not any(f["thin"] for f in tr) and all(f["thin"] for f in te)
    assert count_shift or period_shift or thin_shift, "missing structural train/test shift"


def _mutation_checks(puzzle):
    pair = puzzle["train"][0]
    a, expected = pair["input"], pair["output"]
    untouched = a.copy()
    assert np.array_equal(solve(a), expected) and np.array_equal(a, untouched)
    # Every literal color may play every role; the rule commutes with recoloring.
    permutation = np.random.default_rng(781).permutation(10)
    assert np.array_equal(solve(permutation[a]), permutation[expected])
    p = _parse(a)
    pr, pc, pe, pd = p["panel_box"]
    changed = a.copy()
    changed[pr+1:pe-1, pc+1:pd-1] = changed[pr+1:pe-1, pc+1:pd-1][::-1].copy()
    assert np.array_equal(solve(changed), _alternative(p, "reverse"))
    # Pure cyclic inks must be read once, without recursively following the map.
    cycled = a.copy()
    sources = [s["source"] for s in p["sheets"]]
    cloned = [dict(s, ink=sources[(i+1) % len(sources)]) for i, s in enumerate(p["sheets"])]
    for i, s in enumerate(cloned):
        cycled[pr+1+i, pc+2] = s["ink"]
    assert np.array_equal(solve(cycled), _construct(p["shape"], p["background"], p["frame"], cloned))
    # Changing one stencil interior cell modifies exactly that source's template.
    for index, s in enumerate(p["sheets"]):
        if p["n"] >= 3:
            modified = a.copy()
            r, c = s["stencil_box"][:2]
            new_mask = s["mask"].copy()
            new_mask[1, 1] = ~new_mask[1, 1]
            modified[r+1, c+1] = s["source"] if new_mask[1, 1] else p["background"]
            sheets = [dict(q) for q in p["sheets"]]
            sheets[index]["mask"] = new_mask
            assert np.array_equal(solve(modified), _construct(p["shape"], p["background"], p["frame"], sheets))
            break
    # Translate all visible content within a padded canvas; local phase is stable.
    if max(a.shape) <= 28:
        padded = np.full((a.shape[0]+2, a.shape[1]+2), p["background"], dtype=np.int8)
        padded[1:-1, 1:-1] = a
        assert np.array_equal(solve(padded), expected)


def self_test():
    """Bounded 42-seed diversity, semantic witnesses, mutations and API checks."""
    puzzles, records, signatures, normalized = [], [], set(), set()
    shifts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signature = tuple(_key(pair[k]) for split in ("train", "test") for pair in puzzle[split]
                          for k in ("input", "output"))
        assert signature not in signatures
        signatures.add(signature)
        features = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                f = _features(p)
                f["input_shape"] = pair["input"].shape
                f["board_box"] = p["board_box"]
                f["panel_box"] = p["panel_box"]
                f["split"] = split
                features[split].append(f)
                records.append(f)
                roles = {}
                normalized.add((pair["input"].shape, tuple(roles.setdefault(int(c), len(roles))
                                                           for c in pair["input"].flat)))
        tr, te = features["train"], features["test"]
        if max(f["k"] for f in tr) < min(f["k"] for f in te):
            shifts["count"] += 1
        if max(f["n"] for f in tr) < min(f["n"] for f in te):
            shifts["period"] += 1
        if not any(f["thin"] for f in tr) and all(f["thin"] for f in te):
            shifts["thin"] += 1
        if seed < 8:
            _mutation_checks(puzzle)
    assert len(signatures) == 42 and len(normalized) >= 180
    assert set(shifts) == {"count", "period", "thin"}
    assert {f["n"] for f in records} == {2, 3, 4, 5}
    assert {f["k"] for f in records} == {2, 3, 4, 5}
    assert {f["backgrounds"] for f in records} == set(range(10))
    assert {f["frames"] for f in records} == set(range(10))
    assert len({f["input_shape"] for f in records}) >= 25
    assert len({f["shape"] for f in records}) >= 60
    assert len({f["board_box"] for f in records}) >= 80
    assert len({f["panel_box"] for f in records}) >= 80
    for feature in ("thin_horizontal", "thin_vertical", "disconnected", "asymmetric", "solid",
                    "shared_geometry", "ink_overlap", "background_ink", "isolated"):
        assert sum(bool(f[feature]) for f in records) >= 3, feature
    assert any(not f["common"] and f["overlap_edges"] >= 2 for f in records)
    assert {f["diagonal_set"] for f in records} >= {(0,), (1,), (0, 1)}
    # Repeated generation must reproduce the full puzzle, not only one pair.
    repeat = generate_puzzle(17)
    for split in ("train", "test"):
        for a, b in zip(repeat[split], puzzles[17][split]):
            assert all(np.array_equal(a[k], b[k]) for k in ("input", "output"))
    legal_calls = 0
    for num_train in range(2, 11):
        for num_test in range(1, min(4, num_train-1)+1):
            p = generate_puzzle(85000+num_train*10+num_test, num_train, num_test)
            assert len(p["train"]) == num_train and len(p["test"]) == num_test
            legal_calls += 1
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, pairs=len(records), legal_count_calls=legal_calls,
                distinct_puzzles=len(signatures), color_normalized_inputs=len(normalized),
                input_shapes=len({f["input_shape"] for f in records}),
                board_shapes=len({f["shape"] for f in records}), holdout_modes=dict(shifts),
                features={name: sum(bool(f[name]) for f in records) for name in
                          ("thin_horizontal", "thin_vertical", "disconnected", "solid",
                           "shared_geometry", "ink_overlap", "background_ink", "isolated")},
                semantic_verification="author-owned checks; no independent solver")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
