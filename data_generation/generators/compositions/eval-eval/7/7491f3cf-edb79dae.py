"""Color-linked stencils exchange shape and pigment across a global seam.

Only Python's standard library and NumPy are used. No parent code or fixtures are
imported. The public solver decodes every convention from a single input.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], bool)
_FLAGS = frozenset(("native_visible", "linked_visible", "native_suppressed",
                    "linked_suppressed", "seam_collision", "seam_native",
                    "seam_linked", "seam_blank"))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(p):
    return tuple((s, _key(x["input"]), _key(x["output"]))
                 for s in ("train", "test") for x in p[s])


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    answer = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        answer.append(cells)
    return answer


def _mask(shape, cells):
    a = np.zeros(shape, bool)
    for p in cells:
        a[p] = True
    return a


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _separated(a, b):
    r, e, c, d = a
    rr, ee, cc, dd = b
    return e < rr or ee < r or d < cc or dd < c


def _cut_geometry(cut):
    cells = set(map(tuple, np.argwhere(cut)))
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if not (dr or dc) or q not in cells:
                    continue
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = sorted(p for p in graph if len(graph[p]) == 1)
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()),
             "cut must be one path without a branch or loop")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends) and
             all(not border(p) for p in cells - set(ends)), "invalid cut endpoints")
    visited, previous, current = [], None, ends[0]
    while current is not None and current not in visited:
        visited.append(current)
        following = [q for q in graph[current] if q != previous]
        previous, current = current, following[0] if following else None
    _require(len(visited) == len(cells), "disconnected cut")
    regions = _components(~cut)
    _require(len(regions) == 2, "cut must partition exactly two regions")
    regions = [_mask(cut.shape, x) for x in regions]
    near = np.zeros(cut.shape, bool)
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    sites = [list(map(tuple, np.argwhere(x & ~near))) for x in regions]
    _require(all(sites), "both sides must admit an isolated marker")
    return regions, sites, visited


def _find_frames(grid):
    candidates = []
    for color in np.unique(grid):
        boxes = []
        for cells in _components(grid == color):
            r, e, c, d = _box(cells)
            h, w = e - r, d - c
            if h < 9 or w < 9 or len(cells) != 2 * (h + w) - 4:
                continue
            tile = grid[r:e, c:d]
            if (np.all(tile[[0, -1], :] == color) and
                    np.all(tile[:, [0, -1]] == color)):
                boxes.append((r, e, c, d))
        if len(boxes) != 2:
            continue
        dims = {(e - r, d - c) for r, e, c, d in boxes}
        if len(dims) == 1 and np.count_nonzero(grid == color) == sum(
                2 * (e - r + d - c) - 4 for r, e, c, d in boxes):
            candidates.append((int(color), boxes))
    _require(len(candidates) == 1, "need two congruent exclusive-color frames")
    return candidates[0]


def _decode(input_grid):
    """Visible-input parser, including disconnected exterior stencil islands."""
    grid = _grid(input_grid)
    frame, boxes = _find_frames(grid)
    interiors = [grid[r + 1:e - 1, c + 1:d - 1] for r, e, c, d in boxes]
    values = [set(map(int, np.unique(x))) for x in interiors]
    _require(sorted(map(len, values))[0] == 2 and max(map(len, values)) >= 3,
             "control and board roles are not identifiable")
    ci = 0 if len(values[0]) == 2 else 1
    bi = 1 - ci
    common = values[0] & values[1]
    _require(len(common) == 1, "fields must share exactly their background")
    bg = next(iter(common))
    control = next(iter(values[ci] - {bg}))
    sources = values[bi] - {bg}
    _require(2 <= len(sources) <= 4 and frame not in common and
             control not in sources, "invalid palette roles")
    ctrl, board = interiors[ci], interiors[bi]
    parts = sorted(_components(ctrl == control, True), key=len)
    _require(len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) > 2,
             "need one isolated marker and one cut")
    marker = parts[0][0]
    cut = _mask(ctrl.shape, parts[1])
    regions, sites, path = _cut_geometry(cut)
    side = 0 if regions[0][marker] else 1
    _require(marker in sites[side], "marker touches the cut")
    places, sizes, block_boxes = [], set(), []
    for cells in _components(board != bg):
        r, e, c, d = _box(cells)
        tile = board[r:e, c:d]
        _require(e - r == d - c and 2 <= e - r <= 5 and
                 len(cells) == (e - r) ** 2 and len(np.unique(tile)) == 1,
                 "board objects must be separated solid squares")
        places.append((r, c, int(tile[0, 0])))
        sizes.add(e - r)
        block_boxes.append((r, e, c, d))
    _require(len(sizes) == 1 and {s for _, _, s in places} == sources,
             "missing dictionary category or unequal square size")
    n = next(iter(sizes))
    _require(all(_separated(a, b) for i, a in enumerate(block_boxes)
                 for b in block_boxes[i + 1:]), "touching square footprints")
    exterior = grid != bg
    for r, e, c, d in boxes:
        exterior[r:e, c:d] = False
    mapping, key_boxes = {}, {}
    remaining = exterior.copy()
    for cells in _components(exterior):
        colors = {int(grid[p]) for p in cells}
        if len(colors) == 1:
            continue
        r, e, c, d = _box(cells)
        tile = grid[r:e, c:d]
        _require(len(colors) == 2 and len(cells) == 12 and tile.size == 12,
                 "unknown two-color clue")
        interpretations = set()
        for q in range(4):
            glyph = np.rot90(_ARROW, q)
            if glyph.shape == tile.shape:
                s, t = int(tile[glyph][0]), int(tile[~glyph][0])
                if s != t and np.all(tile[glyph] == s) and np.all(tile[~glyph] == t):
                    interpretations.add((s, t))
        _require(len(interpretations) == 1, "invalid directed arrow")
        s, t = next(iter(interpretations))
        _require(s in sources and t in sources and s not in mapping,
                 "arrow must link dictionary sources exactly once")
        mapping[s] = t
        key_boxes[s] = (r, e, c, d)
        remaining[r:e, c:d] = False
    _require(set(mapping) == sources, "incomplete link dictionary")
    _require(set(map(int, np.unique(grid[remaining]))) == sources,
             "unknown exterior color")
    masks, stencil_boxes = {}, {}
    for s in sorted(sources):
        r, e, c, d = _box(np.argwhere(remaining & (grid == s)))
        _require((e - r, d - c) == (n, n), "stencil bounds/size are not visible")
        tile = grid[r:e, c:d]
        _require(np.all(np.isin(tile, [bg, s])), "overlapping stencil clues")
        mask = tile == s
        _require(mask[0].any() and mask[-1].any() and mask[:, 0].any()
                 and mask[:, -1].any(), "stencil must touch all four bounds")
        masks[s] = mask.copy()
        stencil_boxes[s] = (r, e, c, d)
    all_boxes = boxes + list(key_boxes.values()) + list(stencil_boxes.values())
    _require(all(_separated(a, b) for i, a in enumerate(all_boxes)
                 for b in all_boxes[i + 1:]), "frame/clue bounding boxes need blank gaps")
    return dict(grid=grid, frame=frame, background=bg, control=control, board=board,
                control_box=boxes[ci], board_box=boxes[bi], cut=cut, marker=marker,
                regions=regions, sites=sites, side=side, path=path,
                marked=regions[side], other=regions[1 - side], n=n,
                places=places, masks=masks, mapping=mapping,
                stencil_boxes=stencil_boxes, key_boxes=key_boxes)


def _render(d, mode="rule"):
    """Vectorized constructor and explicit simpler-rule ablations."""
    out, bg, n = d["board"].copy(), d["background"], d["n"]
    mapping = d["mapping"]
    if mode == "reverse":
        _require(len(set(mapping.values())) == len(mapping), "inverse needs a cycle")
        mapping = {t: s for s, t in mapping.items()}
    for r, c, s in d["places"]:
        t = mapping[s]
        a, b = d["masks"][s], d["masks"][t]
        m = d["marked"][r:r + n, c:c + n]
        seam = d["cut"][r:r + n, c:c + n]
        other = d["other"][r:r + n, c:c + n]
        if mode == "native":
            tile = np.where(a, t, bg)
        elif mode == "linked":
            tile = np.where(b, s, bg)
        elif mode == "union":
            tile = np.where(a, t, np.where(b, s, bg))
        elif mode == "center":
            tile = np.where(a, t, bg) if m[n // 2, n // 2] or seam[n // 2, n // 2] else np.where(b, s, bg)
        else:
            if mode == "same_mask":
                b = a
            native_color, linked_color = (s, t) if mode == "no_exchange" else (t, s)
            tile = np.full((n, n), bg, np.int8)
            tile[(m | seam) & a] = native_color
            tile[(other | (seam & ~a)) & b] = linked_color
        out[r:r + n, c:c + n] = tile
    result = np.full((out.shape[0] + 2, out.shape[1] + 2), d["frame"], np.int8)
    result[1:-1, 1:-1] = out
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive one cropped board from its visible control and dictionary only."""
    return _render(_decode(input_grid))


def _cellwise(d):
    """Validation path: no calls to the vectorized output constructor."""
    r0, e0, c0, d0 = d["board_box"]
    result = d["grid"][r0:e0, c0:d0].copy()
    n, bg = d["n"], d["background"]
    for r, c, s in d["places"]:
        t = d["mapping"][s]
        for u in range(n):
            for v in range(n):
                native, linked = d["masks"][s][u, v], d["masks"][t][u, v]
                pos = (r + u, c + v)
                if d["cut"][pos]:
                    value = t if native else s if linked else bg
                elif d["marked"][pos]:
                    value = t if native else bg
                else:
                    value = s if linked else bg
                result[r + u + 1, c + v + 1] = value
    return result


def _evidence(d):
    flags, mixed = set(), False
    n = d["n"]
    local_results = defaultdict(set)
    answer = _render(d)[1:-1, 1:-1]
    for r, c, s in d["places"]:
        a, b = d["masks"][s], d["masks"][d["mapping"][s]]
        m = d["marked"][r:r + n, c:c + n]
        o = d["other"][r:r + n, c:c + n]
        seam = d["cut"][r:r + n, c:c + n]
        conditions = (m & a, o & b, o & a & ~b, m & ~a & b,
                      seam & a & b, seam & a & ~b, seam & ~a & b, seam & ~a & ~b)
        names = ("native_visible", "linked_visible", "native_suppressed",
                 "linked_suppressed", "seam_collision", "seam_native",
                 "seam_linked", "seam_blank")
        flags.update(name for name, condition in zip(names, conditions) if condition.any())
        mixed |= bool(np.any(m & a) and np.any(o & b))
        local_results[s].add(answer[r:r + n, c:c + n].tobytes())
    repeated = any(len(x) > 1 for x in local_results.values())
    return flags, mixed, repeated


def _orientation_evidence(d):
    answer = _render(d)
    for reflected in (False, True):
        for q in range(4):
            if not reflected and q == 0:
                continue
            masks = {s: np.rot90(np.fliplr(m) if reflected else m, q)
                     for s, m in d["masks"].items()}
            if np.array_equal(_render(dict(d, masks=masks)), answer):
                return False
    return True


def _line(p, q):
    r, c = p
    rr, cc = q
    dr, dc = abs(rr - r), abs(cc - c)
    sr, sc = (1 if rr > r else -1), (1 if cc > c else -1)
    error, points = dc - dr, []
    while True:
        points.append((r, c))
        if (r, c) == (rr, cc):
            return points
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _sample_cut(rng, h, w, family):
    for _ in range(80):
        cut = np.zeros((h, w), bool)
        if family == "vertical":
            cut[:, int(rng.integers(2, w - 2))] = True
        elif family == "horizontal":
            cut[int(rng.integers(2, h - 2)), :] = True
        elif family == "diagonal":
            a, b = map(int, rng.choice(w, 2, replace=False))
            for p in _line((0, a), (h - 1, b)):
                cut[p] = True
        elif family == "elbow":
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        elif family == "stepped":
            rows = list(range(2, h - 2, 2))
            amount = int(rng.integers(1, min(3, len(rows)) + 1))
            rows = sorted(map(int, rng.choice(rows, amount, replace=False)))
            last, c = 0, int(rng.integers(2, w - 2))
            for r in rows:
                cut[last:r + 1, c] = True
                nxt = int(rng.choice([x for x in range(2, w - 2) if x != c]))
                cut[r, min(c, nxt):max(c, nxt) + 1] = True
                c, last = nxt, r
            cut[last:, c] = True
        else:
            raise ValueError("unknown cut family")
        if rng.integers(2):
            cut = cut[::-1, :].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        try:
            regions, sites, _ = _cut_geometry(cut)
            return cut, regions, sites
        except AssertionError:
            continue
    # Deterministic valid geometry if the randomized search uses its budget.
    cut = np.zeros((h, w), bool)
    if family == "horizontal":
        cut[h // 2, :] = True
    elif family == "vertical":
        cut[:, w // 2] = True
    elif family == "diagonal":
        for p in _line((0, 2), (h - 1, w - 3)):
            cut[p] = True
    elif family == "elbow":
        cut[:h // 2 + 1, w // 2] = True
        cut[h // 2, w // 2:] = True
    else:
        cut[:h // 2 + 1, 2] = True
        cut[h // 2, 2:w - 2] = True
        cut[h // 2:, w - 3] = True
    regions, sites, _ = _cut_geometry(cut)
    return cut, regions, sites


def _stencil(rng, n):
    y, x = np.indices((n, n))
    kind = int(rng.integers(9))
    if kind == 0:
        mask = (x == 0) | (y == 0) | (x == n - 1) | (y == n - 1)
    elif kind == 1:
        mask = (x == n // 2) | (y == n // 2)
    elif kind == 2:
        mask = x == y
        if rng.random() < .5:
            mask |= x + y == n - 1
    elif kind == 3:
        mask = (x + y + int(rng.integers(2))) % 2 == 0
    elif kind == 4:
        mask = y % 2 == 0
    elif kind == 5:
        mask = (y == 0) | (x == 0)
    elif kind == 6:
        mask = rng.random((n, n)) < float(rng.uniform(.15, .8))
    elif kind == 7:
        mask = rng.random((n, n)) < float(rng.uniform(.7, .95))
    else:
        mask = np.ones((n, n), bool)
    # Only invisible outer offsets are repaired; no answer-state pixels are forced.
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _mapping(rng, sources, kind):
    order = list(map(int, rng.permutation(sources)))
    if kind == "cycle":
        return {s: order[(i + 1) % len(order)] for i, s in enumerate(order)}
    if kind == "noninjective":
        # A two-cycle with every other category entering it; all arrows nonidentity.
        result = {order[0]: order[1], order[1]: order[0]}
        result.update({s: int(rng.choice(order[:2])) for s in order[2:]})
        return result
    return {s: int(rng.choice([t for t in sources if t != s])) for s in sources}


def _starts(rng, length, n, count):
    slack = length - count * n - count + 1
    gaps = rng.multinomial(slack, np.full(count + 1, 1 / (count + 1)))
    starts, pos = [], int(gaps[0])
    for i in range(count):
        starts.append(pos)
        pos += n + 1 + int(gaps[i + 1])
    return starts


def _placements(rng, h, w, n, k, initial):
    nr, nc = (h + 1) // (n + 1), (w + 1) // (n + 1)
    nr, nc = min(nr, 4), min(nc, 4)
    capacity = nr * nc
    count = int(rng.integers(max(k, 4) if initial else k, min(8, capacity) + 1))
    rows, cols = _starts(rng, h, n, nr), _starts(rng, w, n, nc)
    slots = [(r, c) for r in rows for c in cols]
    spots = [slots[int(i)] for i in rng.permutation(capacity)[:count]]
    # Independently perturb objects, retaining full-footprint blank separation.
    for i in rng.permutation(count):
        r, c = spots[int(i)]
        for _ in range(12):
            rr = int(np.clip(r + rng.integers(-2, 3), 0, h - n))
            cc = int(np.clip(c + rng.integers(-2, 3), 0, w - n))
            candidate = (rr, rr + n, cc, cc + n)
            if all(j == i or _separated(candidate, (a, a + n, b, b + n))
                   for j, (a, b) in enumerate(spots)):
                spots[int(i)] = (rr, cc)
                break
    return spots


def _draw_frame(grid, box, color):
    r, e, c, d = box
    grid[r, c:d] = grid[e - 1, c:d] = color
    grid[r:e, c] = grid[r:e, d - 1] = color


def _sample_example(rng, params, split, initial=False):
    training = split == "train"
    shift = params["shift"]
    if shift == "scale":
        n = 3 if initial else int(rng.choice([2, 3] if training else [4, 5]))
    else:
        n = int(rng.choice([3, 4] if initial else [2, 3, 4, 5]))
    if initial or (shift == "graph" and training):
        k, graph_kind = 3, "cycle"
    elif shift == "graph":
        k, graph_kind = (4, "cycle") if rng.random() < .5 else (int(rng.choice([3, 4])), "noninjective")
    else:
        k, graph_kind = int(rng.integers(2, 5)), str(rng.choice(["cycle", "noninjective", "mixed"]))
    stagger = int(rng.integers(3))
    minimum = max(7, 2 * n + (3 if initial else 1))
    h = int(rng.integers(minimum, 24 - n - stagger))
    w = int(rng.integers(minimum, 13))
    spots = _placements(rng, h, w, n, k, initial)
    colors = params["palette"] if params["shared"] or rng.random() < .25 else rng.permutation(10)
    bg, frame, control = map(int, colors[:3])
    sources = list(map(int, colors[3:3 + k]))
    links = _mapping(rng, sources, graph_kind)
    masks = {}
    for s in sources:
        masks[s] = masks[int(rng.choice(list(masks)))].copy() if masks and rng.random() < .12 else _stencil(rng, n)
    assignments = sources + list(map(int, rng.choice(sources, len(spots) - k)))
    rng.shuffle(assignments)
    places = [(r, c, s) for (r, c), s in zip(spots, assignments)]
    board = np.full((h, w), bg, np.int8)
    for r, c, s in places:
        board[r:r + n, c:c + n] = s
    if shift == "geometry":
        family = str(rng.choice(["vertical", "horizontal", "diagonal"] if training else ["elbow", "stepped"]))
    else:
        family = str(rng.choice(["vertical", "horizontal", "diagonal", "elbow", "stepped"]))
    if initial:
        family = params["initial_family"]
    cut, regions, sites = _sample_cut(rng, h, w, family)
    side = int(rng.integers(2))
    marker = sites[side][int(rng.integers(len(sites[side])))]
    ctrl = np.full((h, w), bg, np.int8)
    ctrl[cut] = control
    ctrl[marker] = control
    raw = dict(board=board, frame=frame, background=bg, n=n, places=places,
               masks=masks, mapping=links, cut=cut, marked=regions[side], other=regions[1 - side])
    output = _render(raw)
    top_stencils = bool(rng.integers(2))
    top_height, bottom_height = (n, 3) if top_stencils else (3, n)
    height, width = h + n + 7 + stagger, max(2 * w + 5, k * (n + 1) - 1, 5 * k - 1)
    grid = np.full((height, width), bg, np.int8)
    base_r = top_height + 1
    left = int(rng.integers(width - (2 * w + 5) + 1))
    flip_order = bool(rng.integers(2))
    ctrl_col, board_col = (left + w + 3, left) if flip_order else (left, left + w + 3)
    ctrl_row, board_row = (base_r + stagger, base_r) if rng.integers(2) else (base_r, base_r + stagger)
    for panel, r, c in ((ctrl, ctrl_row, ctrl_col), (board, board_row, board_col)):
        _draw_frame(grid, (r, r + h + 2, c, c + w + 2), frame)
        grid[r + 1:r + h + 1, c + 1:c + w + 1] = panel
    stencil_row = 0 if top_stencils else height - n
    key_row = height - 3 if top_stencils else 0
    mask_left = int(rng.integers(width - (k * (n + 1) - 1) + 1))
    key_left = int(rng.integers(width - (5 * k - 1) + 1))
    for i, s in enumerate(rng.permutation(sources)):
        c = mask_left + i * (n + 1)
        grid[stencil_row:stencil_row + n, c:c + n] = np.where(masks[int(s)], s, bg)
    for i, s in enumerate(rng.permutation(sources)):
        c = key_left + 5 * i
        grid[key_row:key_row + 3, c:c + 4] = np.where(_ARROW, s, links[int(s)])
    # Small optional canvas margins do not define any decoding convention.
    extra_h, extra_w = min(2, 30 - height), min(2, 30 - width)
    pad_h, pad_w = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    before_h, before_w = int(rng.integers(pad_h + 1)), int(rng.integers(pad_w + 1))
    grid = np.pad(grid, ((before_h, pad_h - before_h), (before_w, pad_w - before_w)), constant_values=bg)
    rotation = int(rng.integers(4))
    return {"input": np.rot90(grid, rotation).copy(), "output": np.rot90(output, rotation).copy()}


def _opposite(pair, rng):
    d = _decode(pair["input"])
    grid = d["grid"].copy()
    r, _, c, _ = d["control_box"]
    grid[r + 1 + d["marker"][0], c + 1 + d["marker"][1]] = d["background"]
    sites = d["sites"][1 - d["side"]]
    marker = sites[int(rng.integers(len(sites)))]
    grid[r + 1 + marker[0], c + 1 + marker[1]] = d["control"]
    return {"input": grid, "output": solve(grid)}


def _constructive_example(rng, params, split):
    """A same-rule backup with explicit geometry, never an answer exception.

    The initial three-column motifs expose all four seam states without later
    pixel editing. Larger masks retain that evidence and add one visible corner.
    Test geometry/size/link choices still implement the puzzle's chosen shift.
    """
    test = split == "test"
    n = 4 if test and params["shift"] == "scale" else 3
    h = w = 2 * n + 3
    colors = params["palette"]
    bg, frame, control = map(int, colors[:3])
    sources = list(map(int, colors[3:6]))
    s0, s1, s2 = sources
    mapping = {s0: s1, s1: s2, s2: s0}
    if test and params["shift"] == "graph":
        mapping = {s0: s1, s1: s0, s2: s0}
    prototypes = ([[1, 1, 0], [0, 1, 1], [1, 0, 0]],
                  [[0, 1, 1], [1, 0, 0], [0, 0, 1]],
                  [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    masks = {}
    for s, prototype in zip(sources, prototypes):
        mask = np.zeros((n, n), bool)
        mask[:3, :3] = prototype
        if n > 3:
            mask[-1, -1] = True
        masks[s] = mask
    places = [(0, 2, s0), (n + 3, 2, s1), (0, n + 3, s0), (n + 3, n + 3, s2)]
    board = np.full((h, w), bg, np.int8)
    for r, c, s in places:
        board[r:r + n, c:c + n] = s
    cut = np.zeros((h, w), bool)
    if test and params["shift"] == "geometry":
        cut[:h // 2 + 1, 3] = True
        cut[h // 2, 3:5] = True
        cut[h // 2:, 4] = True
    else:
        cut[:, 3] = True
    regions, _, _ = _cut_geometry(cut)
    raw = dict(board=board, frame=frame, background=bg, n=n, places=places,
               masks=masks, mapping=mapping, cut=cut, marked=regions[0], other=regions[1])
    ctrl = np.full((h, w), bg, np.int8)
    ctrl[cut] = control
    ctrl[0, 0] = control
    grid = np.full((h + n + 7, 2 * w + 5), bg, np.int8)
    for panel, c in ((ctrl, 0), (board, w + 3)):
        _draw_frame(grid, (n + 1, n + h + 3, c, c + w + 2), frame)
        grid[n + 2:n + h + 2, c + 1:c + w + 1] = panel
    for i, s in enumerate(rng.permutation(sources)):
        c = i * (n + 1)
        grid[:n, c:c + n] = np.where(masks[int(s)], s, bg)
    for i, s in enumerate(rng.permutation(sources)):
        grid[-3:, 5 * i:5 * i + 4] = np.where(_ARROW, s, mapping[int(s)])
    q = int(rng.integers(4))
    return {"input": np.rot90(grid, q).copy(), "output": np.rot90(_render(raw), q).copy()}


def _separate_backup(pair, seen):
    # Forty valid recoloring/rotation choices exceed the maximum 13 prior inputs.
    # This guarantees uniqueness even if several example searches exhaust.
    for shift in range(10):
        for q in range(4):
            candidate = {k: np.rot90((v + shift) % 10, q).copy() for k, v in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("backup orbit unexpectedly exhausted")


def _informative_first(pair, rng):
    d = _decode(pair["input"])
    twin = _opposite(pair, rng)
    flags, mixed, repeated = _evidence(d)
    flags2, mixed2, repeated2 = _evidence(_decode(twin["input"]))
    if flags | flags2 != _FLAGS or not (mixed or mixed2) or not (repeated or repeated2):
        return False
    if np.array_equal(pair["output"], twin["output"]) or not _orientation_evidence(d):
        return False
    return all(not np.array_equal(_render(d, mode), pair["output"])
               for mode in ("native", "linked", "union", "same_mask", "center", "reverse", "no_exchange"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = dict(shift=str(rng.choice(["geometry", "scale", "graph"])),
                  palette=rng.permutation(10), shared=bool(rng.random() < .3),
                  initial_family=str(rng.choice(["vertical", "horizontal", "diagonal"])))
    for _ in range(600):
        first = _sample_example(rng, params, "train", initial=True)
        if _informative_first(first, rng):
            break
    else:
        first = _constructive_example(rng, params, "train")
        _require(_informative_first(first, rng), "constructive training evidence failed")
    twin = _opposite(first, rng)
    puzzle = {"train": [first, twin], "test": []}
    seen = {_key(first["input"]), _key(twin["input"])}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for _ in range(count):
            for attempt in range(200):
                pair = _sample_example(rng, params, split)
                if _key(pair["input"]) in seen:
                    continue
                d = _decode(pair["input"])
                flags, mixed, _ = _evidence(d)
                if not {"native_visible", "linked_visible"} <= flags:
                    continue
                if split == "test" and not mixed:
                    continue
                break
            else:
                pair = _separate_backup(_constructive_example(rng, params, split), seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen, coverage, sides, infos = set(), set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(_cellwise(d), out), "wrong exchange, crop or preserved board cells")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented output color")
            flags, mixed, repeated = _evidence(d)
            if split == "train":
                coverage |= flags
                sides.add(d["side"])
                infos.append((d, mixed, repeated))
            else:
                _require(mixed, "test needs a genuinely spliced square")
    _require(coverage == _FLAGS and sides == {0, 1}, "training does not expose both regions and seam cases")
    _require(any(m for _, m, _ in infos) and any(r for _, _, r in infos),
             "missing straddling/repeated-source registration evidence")
    first = infos[0][0]
    _require(_orientation_evidence(first), "training does not distinguish stencil orientation")
    _require(len(first["mapping"]) == 3 and len(set(first["mapping"].values())) == 3,
             "initial example needs a directed three-cycle")
    answer = _cellwise(first)
    for mode in ("native", "linked", "union", "same_mask", "center", "reverse", "no_exchange"):
        _require(not np.array_equal(_render(first, mode), answer), f"training permits shortcut {mode}")


def _family(d):
    path = d["path"]
    if len({r for r, c in path}) == 1:
        return "horizontal"
    if len({c for r, c in path}) == 1:
        return "vertical"
    r0, c0 = path[0]
    r1, c1 = path[-1]
    if abs(r1 - r0) >= abs(c1 - c0):
        deviations = [abs(c - (c0 + (r - r0) * (c1 - c0) / (r1 - r0)))
                      for r, c in path]
    else:
        deviations = [abs(r - (r0 + (c - c0) * (r1 - r0) / (c1 - c0)))
                      for r, c in path]
    if max(deviations) <= .500001 and len(path) == max(abs(r1 - r0), abs(c1 - c0)) + 1:
        return "diagonal"
    changes = sum(a != b for a, b in zip(
        [(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])],
        [(q[0] - p[0], q[1] - p[1]) for p, q in zip(path[1:], path[2:])]))
    return "elbow" if changes == 1 else "stepped"


def self_test():
    """Bounded functional, metamorphic, count and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity collapsed")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic generation")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7000 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.2), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    shapes, sizes, counts, categories, families = set(), set(), set(), set(), set()
    masks, controls, source_layouts, backgrounds, graphs = set(), set(), set(), set(), set()
    disconnected = symmetric = unequal_positions = 0
    rng = np.random.default_rng(98017)
    # Explicitly exercise all same-rule backup constructions, not only rare
    # randomized searches that happen to exhaust their budgets.
    for shift in ("geometry", "scale", "graph"):
        parameters = dict(shift=shift, palette=rng.permutation(10))
        first = _constructive_example(rng, parameters, "train")
        backup = {"train": [first, _opposite(first, rng)],
                  "test": [_constructive_example(rng, parameters, "test")]}
        validate_generated_puzzle(backup)
        prior = {_key(first["input"])}
        for _ in range(13):
            distinct = _separate_backup(first, prior)
            _require(np.array_equal(solve(distinct["input"]), distinct["output"]),
                     "backup uniqueness broke the rule")
            prior.add(_key(distinct["input"]))
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                shapes.add((d["grid"].shape, pair["output"].shape))
                sizes.add(d["n"])
                counts.add(len(d["places"]))
                categories.add(len(d["mapping"]))
                families.add(_family(d))
                backgrounds.add(d["background"])
                controls.add((d["cut"].shape, d["cut"].tobytes()))
                source_layouts.add((d["board"].shape, tuple((r, c) for r, c, s in d["places"])))
                sources = sorted(d["mapping"])
                graphs.add(tuple(sources.index(d["mapping"][s]) for s in sources))
                unequal_positions += _evidence(d)[2]
                for m in d["masks"].values():
                    masks.add(_key(m))
                    disconnected += len(_components(m)) > 1
                    symmetric += any(np.array_equal(m, np.rot90(m, q)) for q in (1, 2, 3))
        first, twin = puzzle["train"][:2]
        a, b = _decode(first["input"]), _decode(twin["input"])
        _require(np.count_nonzero(first["input"] != twin["input"]) == 2, "marker isolation lost")
        oa, ob = first["output"][1:-1, 1:-1], twin["output"][1:-1, 1:-1]
        _require(np.array_equal(oa[a["cut"]], ob[a["cut"]]), "marker changed seam priority")
        _require(not np.array_equal(oa, ob), "marker is ineffective")
        _require(np.array_equal(a["marked"], b["other"]), "marker failed to swap regions")
        permutation = rng.permutation(10)
        _require(np.array_equal(solve(permutation[first["input"]]), permutation[first["output"]]),
                 "literal color shortcut")
        for q in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(first["input"], q)), np.rot90(first["output"], q)),
                     "scene rotation broke registration")
        # Exchange complete legend tiles; identity association must ignore list order.
        changed = first["input"].copy()
        boxes = list(a["stencil_boxes"].values())[:2]
        (r, e, c, d), (rr, ee, cc, dd) = boxes
        one, two = changed[r:e, c:d].copy(), changed[rr:ee, cc:dd].copy()
        changed[r:e, c:d], changed[rr:ee, cc:dd] = two, one
        _require(np.array_equal(solve(changed), first["output"]), "legend-order shortcut")
    _require(sizes == {2, 3, 4, 5} and categories == {2, 3, 4}, "size/category diversity collapsed")
    _require(len(shapes) >= 60 and len(controls) >= 100 and len(source_layouts) >= 100,
             "geometry diversity collapsed")
    _require(families == {"horizontal", "vertical", "diagonal", "elbow", "stepped"}, "cut vocabulary collapsed")
    _require(len(counts) >= 5 and len(graphs) >= 12 and len(masks) >= 80,
             "count/link/stencil diversity collapsed")
    _require(backgrounds == set(range(10)) and disconnected >= 30 and symmetric >= 30
             and unequal_positions >= 42, "palette/shape/context variation collapsed")
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ("train", "test")}
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted frame")
    for kind in ("branch", "loop", "third_region"):
        cut = np.zeros((9, 9), bool)
        if kind == "branch":
            cut[:, 4] = True
            cut[4, 4:7] = True
        elif kind == "loop":
            cut[2:7, 2] = cut[2:7, 6] = True
            cut[2, 2:7] = cut[6, 2:7] = True
        else:
            cut[:, 2] = cut[:, 6] = True
        try:
            _cut_geometry(cut)
        except AssertionError:
            continue
        raise AssertionError(f"invalid {kind} control accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus count and metamorphic checks)")
