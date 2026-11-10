"""Analogy-powered mosaics: a locally learned symmetry travels through a graph.

Only NumPy and the standard library are used. Nothing from the parents is loaded.
"""
from collections import deque
from itertools import combinations

import numpy as np


FRAME = 5
START = 9
INK_COLORS = (1, 2, 3, 4, 6, 7, 8)
OPERATIONS = ("identity", "ccw90", "half_turn", "cw90", "mirror_lr",
              "mirror_diagonal", "mirror_ud", "mirror_anti_diagonal")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _transform(a, operation):
    """The eight square symmetries, including identity as a parsing hypothesis."""
    return np.rot90(a if operation < 4 else a[:, ::-1], operation % 4).copy()


def _advance(a, operation, steps):
    out = a.copy()
    for _ in range(steps):
        out = _transform(out, operation)
    return out


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _crop(mask):
    pts = np.argwhere(mask)
    _require(len(pts) > 0, "empty stencil")
    lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _distances(sites, origin):
    """Unweighted shortest paths between adjacent occupied logical sites."""
    dist = np.full(sites.shape, -1, dtype=int)
    dist[origin] = 0
    queue = deque([origin])
    while queue:
        r, c = queue.popleft()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < sites.shape[0] and 0 <= x < sites.shape[1]:
                if sites[y, x] and dist[y, x] < 0:
                    dist[y, x] = dist[r, c] + 1
                    queue.append((y, x))
    _require(bool(np.all(dist[sites] >= 0)), "disconnected site graph")
    return dist


def _contains(outer, inner):
    r, c, h, w = outer
    y, x, a, b = inner
    return r < y and c < x and y+a < r+h and x+b < c+w


def _frame_boxes(g):
    boxes = []
    for component in _components(g == FRAME):
        pts = np.array(component)
        r, c = map(int, pts.min(axis=0))
        bottom, right = map(int, pts.max(axis=0) + 1)
        h, w = bottom-r, right-c
        _require(min(h, w) >= 4, "small or broken frame")
        block = g[r:bottom, c:right]
        border = np.zeros(block.shape, dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        expected = border.copy()
        marked = block[0, 0] == START
        if marked:
            expected[0, 0] = False
        component_mask = np.zeros(block.shape, dtype=bool)
        for y, x in component:
            component_mask[y-r, x-c] = True
        _require(np.array_equal(component_mask, expected), "frame is not an outline")
        _require(bool(np.all(block[border] == FRAME)) if not marked else
                 bool(np.all(block[border & expected] == FRAME)), "invalid frame border")
        boxes.append((r, c, h, w))
    _require(len(boxes) == 4, "expected enclosure, reference frames and stencil frame")
    return boxes


def _interior(g, box):
    r, c, h, w = box
    return g[r+1:r+h-1, c+1:c+w-1]


def _binary_panel(a, label):
    colors = set(map(int, np.unique(a))) - {0}
    _require(len(colors) == 1 and not colors & {FRAME, START}, label + " has invalid ink")
    return a != 0, next(iter(colors))


def _decode(input_grid):
    """Recover every semantic parameter from pixels; never inspect an output."""
    g = _grid(input_grid)
    boxes = _frame_boxes(g)
    parents = [b for b in boxes if sum(_contains(b, a) for a in boxes) == 2]
    _require(len(parents) == 1, "reference enclosure is ambiguous")
    outer = parents[0]
    reference = [b for b in boxes if _contains(outer, b)]
    targets = [b for b in boxes if b != outer and b not in reference]
    _require(len(targets) == 1, "missing stencil frame")
    target_box = targets[0]
    marked = [b for b in reference if g[b[0], b[1]] == START]
    _require(len(marked) == 1, "reference needs one marked initial state")
    before_box = marked[0]
    after_box = next(b for b in reference if b != before_box)
    before, ref_ink = _binary_panel(_interior(g, before_box), "reference before")
    after, after_ink = _binary_panel(_interior(g, after_box), "reference after")
    _require(before.shape == after.shape and before.shape[0] == before.shape[1],
             "reference interiors must be equal squares")
    _require(3 <= before.shape[0] <= 4 and ref_ink == after_ink, "invalid reference scale/palette")
    hypotheses = [i for i in range(8) if np.array_equal(_transform(before, i), after)]
    _require(len(hypotheses) == 1 and hypotheses[0] != 0, "reference symmetry is not unique")
    operation = hypotheses[0]
    target_mask, target_ink = _binary_panel(_interior(g, target_box), "target stencil")
    tile = _crop(target_mask)
    _require(tile.shape[0] == tile.shape[1] and 3 <= tile.shape[0] <= 5,
             "target crop must be a square of side 3..5")

    # Remove the two semantic modules, including padding inside the enclosure.
    available = np.ones(g.shape, dtype=bool)
    for r, c, h, w in (outer, target_box):
        available[r:r+h, c:c+w] = False
    # The only nonzero cells outside these modules are the counter objects.
    graph_colors = set(map(int, np.unique(g[available]))) - {0, START}
    _require(len(graph_colors) == 1 and not graph_colors & {FRAME}, "invalid counter ink")
    graph_ink = next(iter(graph_colors))
    _require(len({ref_ink, target_ink, graph_ink}) == 3, "semantic inks must be distinct")
    counter_mask = available & (g != 0)
    components = _components(counter_mask)
    _require(len(components) == len(_components(counter_mask, True)), "diagonal counter contact")
    _require(np.count_nonzero((g == START) & available) == 1 and np.count_nonzero(g == START) == 2,
             "one reference start and one counter origin are required")
    _require(g[target_box[0], target_box[1]] == FRAME, "target frame cannot be marked")
    anchors, origin_anchor, sizes, node_shapes = [], None, [], []
    for component in components:
        pts = np.array(component)
        lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
        anchor = tuple(map(int, lo))
        h, w = map(int, hi-lo)
        _require(max(h, w) <= 3 and bool(counter_mask[anchor]), "invalid anchored counter object")
        anchors.append(anchor)
        sizes.append(len(component))
        node_shapes.append((h, w))
        if any(g[r, c] == START for r, c in component):
            _require(g[anchor] == START and len(component) >= 2, "origin marker must replace its anchor")
            origin_anchor = anchor
    rows, cols = sorted({r for r, _ in anchors}), sorted({c for _, c in anchors})
    _require(all(b-a >= 4 for a, b in zip(rows, rows[1:])) and
             all(b-a >= 4 for a, b in zip(cols, cols[1:])), "counter bands too close")
    _require(1 <= len(rows) <= 5 and 1 <= len(cols) <= 5, "too many site bands")
    sites = np.zeros((len(rows), len(cols)), dtype=bool)
    for r, c in anchors:
        index = (rows.index(r), cols.index(c))
        _require(not sites[index], "two components occupy one site")
        sites[index] = True
    _require(origin_anchor is not None and len(components) >= 2, "no graph origin")
    origin = (rows.index(origin_anchor[0]), cols.index(origin_anchor[1]))
    distance = _distances(sites, origin)
    side = tile.shape[0]
    _require(max(n*(side+1)-1 for n in sites.shape) <= 30, "mosaic exceeds output budget")
    # Enclosure padding is blank. This catches unrecognized hidden cues.
    allowed = np.zeros(g.shape, dtype=bool)
    for r, c, h, w in boxes:
        if (r, c, h, w) != outer:
            allowed[r:r+h, c:c+w] = True
    r, c, h, w = outer
    allowed[r, c:c+w] = allowed[r+h-1, c:c+w] = True
    allowed[r:r+h, c] = allowed[r:r+h, c+w-1] = True
    allowed |= counter_mask
    _require(not bool(np.any((g != 0) & ~allowed)), "unrecognized decoration")
    return dict(tile=tile, ink=target_ink, operation=operation, sites=sites,
                origin=origin, distance=distance, sizes=sizes, node_shapes=node_shapes,
                counter_area=sum(sizes), target_panel=target_mask, boxes=boxes,
                before_box=before_box, after_box=after_box, reference=before,
                ref_ink=ref_ink, graph_ink=graph_ink, anchors=anchors, rows=rows, cols=cols)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the mosaic using only this input's analogy, crop and site graph."""
    d = _decode(input_grid)
    k = d["tile"].shape[0]
    h, w = (n*(k+1)-1 for n in d["sites"].shape)
    out = np.zeros((h, w), dtype=int)
    # Cell indexing provides a separate rendering path from generator assembly.
    states = {int(step): _advance(d["tile"], d["operation"], int(step))
              for step in np.unique(d["distance"][d["sites"]])}
    for r in range(h):
        for c in range(w):
            y, a = divmod(r, k+1)
            x, b = divmod(c, k+1)
            if a < k and b < k and d["sites"][y, x]:
                if states[int(d["distance"][y, x])][a, b]:
                    out[r, c] = d["ink"]
    return out


def _render(tile, ink, operation, sites, origin):
    distance = _distances(sites, origin)
    k = tile.shape[0]
    bands = []
    for r in range(sites.shape[0]):
        parts = []
        for c in range(sites.shape[1]):
            state = _advance(tile, operation, int(distance[r, c])) if sites[r, c] else np.zeros_like(tile)
            parts.append(np.where(state, ink, 0).astype(int))
            if c+1 < sites.shape[1]:
                parts.append(np.zeros((k, 1), dtype=int))
        bands.append(np.concatenate(parts, axis=1))
        if r+1 < sites.shape[0]:
            bands.append(np.zeros((1, bands[0].shape[1]), dtype=int))
    return np.concatenate(bands, axis=0)


def _outline(g, box):
    r, c, h, w = box
    g[r, c:c+w] = g[r+h-1, c:c+w] = FRAME
    g[r:r+h, c] = g[r:r+h, c+w-1] = FRAME


def _motif(rng, side, asymmetric=True):
    """Mixture of connected silhouettes and all-fragment binary stencils."""
    connected = rng.random() < .7
    for _ in range(300):
        mask = rng.random((side, side)) < rng.uniform(.25, .8)
        if not mask.any() or _crop(mask).shape != mask.shape:
            continue
        if connected and len(_components(mask)) != 1:
            continue
        if asymmetric and len({_key(_transform(mask, i)) for i in range(8)}) != 8:
            continue
        return mask
    # A connected branch with a full bounding box and no nontrivial symmetry.
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = mask[:, 0] = True
    mask[-1, 1] = True
    _require(len({_key(_transform(mask, i)) for i in range(8)}) == 8, "motif fallback symmetry")
    return mask


def _reference_motif(rng, side):
    # Preserved area visibly rules out binary complementation as an explanation.
    for _ in range(30):
        mask = _motif(rng, side)
        if 2*int(mask.sum()) != mask.size:
            return mask
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = mask[:, 0] = True
    mask[-1, 1] = True
    if 2*int(mask.sum()) == mask.size:
        mask[-2, 1] = True
    _require(2*int(mask.sum()) != mask.size and
             len({_key(_transform(mask, i)) for i in range(8)}) == 8, "reference fallback")
    return mask


def _object_pool(h, w, area):
    result = []
    for bits in range(1, 1 << (h*w)):
        if bits.bit_count() != area or not bits & 1:
            continue
        mask = np.array([(bits >> i) & 1 for i in range(h*w)], dtype=bool).reshape(h, w)
        if _crop(mask).shape == mask.shape and len(_components(mask)) == 1:
            result.append(mask)
    return result


def _object(rng, area=None, compact=False):
    if area is not None:
        if compact:
            h, w = (2, 3) if rng.integers(2) else (3, 2)
        else:
            h = w = 3
    else:
        h, w = map(int, rng.integers(1, 4, size=2))
        # A connected path from the anchor reaching both far bounding sides fits.
        area = int(rng.integers(h+w-1, h*w+1))
    pool = _object_pool(h, w, area)
    _require(bool(pool), "empty object shape pool")
    return pool[int(rng.integers(len(pool)))].copy()


def _site_graph(rng, h, w):
    sites = np.ones((h, w), dtype=bool)
    desired = int(rng.integers(max(2, h+w-1), h*w+1))
    for flat in rng.permutation(h*w):
        if sites.sum() <= desired:
            break
        r, c = divmod(int(flat), w)
        sites[r, c] = False
        if not sites.any(axis=0).all() or not sites.any(axis=1).all() or len(_components(sites)) != 1:
            sites[r, c] = True
    return sites


def _build_pair(rng, tile, reference, operation, sites, origin, palette, areas=None, transpose=None):
    ref_ink, tile_ink, counter_ink = palette
    s = reference.shape[0]
    cal_h, cal_w = s+6, 2*s+9
    calibration = np.zeros((cal_h, cal_w), dtype=int)
    _outline(calibration, (0, 0, cal_h, cal_w))
    small = [(2, 2, s+2, s+2), (2, s+5, s+2, s+2)]
    if rng.integers(2):
        small.reverse()
    for box, mask in zip(small, (reference, _transform(reference, operation))):
        _outline(calibration, box)
        r, c, _, _ = box
        calibration[r+1:r+s+1, c+1:c+s+1] = np.where(mask, ref_ink, 0)
    calibration[small[0][0], small[0][1]] = START

    k = tile.shape[0]
    margins = list(map(int, rng.integers(0, 3, size=4)))
    # Keep module width/height below ten; cropping margins remain independent.
    while k + margins[0] + margins[1] + 2 > 10:
        i = 0 if margins[0] else 1
        margins[i] -= 1
    while k + margins[2] + margins[3] + 2 > 10:
        i = 2 if margins[2] else 3
        margins[i] -= 1
    if sum(margins) == 0:
        margins[int(rng.integers(4))] = 1
    top, bottom, left, right = margins
    th, tw = k+top+bottom+2, k+left+right+2
    target = np.zeros((th, tw), dtype=int)
    _outline(target, (0, 0, th, tw))
    target[1+top:1+top+k, 1+left:1+left+k] = np.where(tile, tile_ink, 0)

    # Physical band spacing varies independently of logical edge lengths.
    row_gaps = rng.integers(4, 6, size=max(0, sites.shape[0]-1))
    col_gaps = rng.integers(4, 6, size=max(0, sites.shape[1]-1))
    row_budget = 30-max(cal_h, th)-1-3
    while int(row_gaps.sum()) > row_budget:
        candidates = np.flatnonzero(row_gaps > 4)
        _require(len(candidates) > 0, "site rows cannot fit input")
        row_gaps[int(rng.choice(candidates))] -= 1
    rows = np.r_[0, np.cumsum(row_gaps)].astype(int)
    cols = np.r_[0, np.cumsum(col_gaps)].astype(int)
    gh, gw = int(rows[-1])+3, int(cols[-1])+3
    counters = np.zeros((gh, gw), dtype=int)
    compact_index = min(range(len(areas)), key=lambda i: areas[i]) if areas is not None else None
    for index, (r, c) in enumerate(map(tuple, np.argwhere(sites))):
        obj = _object(rng, None if areas is None else areas[index], index == compact_index)
        if (r, c) == origin and obj.size == 1:
            obj = np.ones((1, 2), dtype=bool)
        y, x = int(rows[r]), int(cols[c])
        counters[y:y+obj.shape[0], x:x+obj.shape[1]][obj] = counter_ink
        if (r, c) == origin:
            counters[y, x] = START

    module_h = max(cal_h, th)
    module_w = cal_w + 1 + tw
    gap = int(rng.integers(1, min(3, 30-module_h-gh)+1))
    core_h, core_w = module_h+gap+gh, max(module_w, gw)
    _require(max(core_h, core_w) <= 30, "input packing failed")
    height = int(rng.integers(core_h, min(30, core_h+3)+1))
    width = int(rng.integers(core_w, min(30, core_w+3)+1))
    inp = np.zeros((height, width), dtype=int)
    base = int(rng.integers(height-core_h+1))
    graph_above = bool(rng.integers(2))
    module_y, graph_y = (base+gh+gap, base) if graph_above else (base, base+module_h+gap)
    module_x = int(rng.integers(width-module_w+1))
    target_left = bool(rng.integers(2))
    cx, tx = (module_x+tw+1, module_x) if target_left else (module_x, module_x+cal_w+1)
    cy = module_y + int(rng.integers(module_h-cal_h+1))
    ty = module_y + int(rng.integers(module_h-th+1))
    gx = int(rng.integers(width-gw+1))
    inp[cy:cy+cal_h, cx:cx+cal_w] = calibration
    inp[ty:ty+th, tx:tx+tw] = target
    inp[graph_y:graph_y+gh, gx:gx+gw] = counters
    output = _render(tile, tile_ink, operation, sites, origin)
    transpose = bool(rng.integers(2)) if transpose is None else bool(transpose)
    if transpose:
        inp, output = inp.T.copy(), output.T.copy()
    return {"input": inp, "output": output}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(INK_COLORS, 3, replace=False)))
    stable_palette = rng.random() < .35
    tile = _motif(rng, int(rng.integers(3, 5)))
    reference = _reference_motif(rng, int(rng.integers(3, 5)))
    quarter = int(rng.choice([1, 3]))
    inverse_quarter = 4-quarter
    witness_transpose = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for index in range(amount):
            witness = split == "train" and index < 2
            if witness:
                sites = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool) if index == 0 else np.ones((2, 3), bool)
                origin = (0, 0)
                operation = quarter if index == 0 else inverse_quarter
                current_tile, current_ref, colors = tile, reference, palette
                areas = [5] * int(sites.sum())
                # Equal 30-cell areas, but five versus six components.
                for _ in range(30-sum(areas)):
                    choices = [i for i, area in enumerate(areas) if area < 9]
                    areas[int(rng.choice(choices))] += 1
            else:
                if split == "train":
                    nr, nc = int(rng.integers(1, 4)), int(rng.integers(2, 5))
                    side = int(rng.integers(3, 5))
                else:
                    nr, nc = int(rng.integers(3, 5)), int(rng.integers(3, 6))
                    side = 5  # New tile scale, with the reference still visible.
                sites = _site_graph(rng, nr, nc)
                positions = np.argwhere(sites)
                origin = tuple(map(int, positions[int(rng.integers(len(positions)))]))
                operation = int(rng.integers(1, 8))
                current_tile = _motif(rng, side, asymmetric=rng.random() < .8)
                current_ref = _reference_motif(rng, int(rng.integers(3, 5)))
                colors = palette if stable_palette else tuple(map(int, rng.choice(INK_COLORS, 3, replace=False)))
                areas = None
            for _ in range(20):
                pair = _build_pair(rng, current_tile, current_ref, operation, sites, origin, colors, areas,
                                   witness_transpose if witness else None)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    return puzzle


def _manhattan_differs(d):
    r0, c0 = d["origin"]
    for r, c in map(tuple, np.argwhere(d["sites"])):
        if not np.array_equal(_advance(d["tile"], d["operation"], int(d["distance"][r, c])),
                              _advance(d["tile"], d["operation"], abs(r-r0)+abs(c-c0))):
            return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, train, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            expected_shape = tuple(n*(d["tile"].shape[0]+1)-1 for n in d["sites"].shape)
            _require(out.shape == expected_shape, "wrong mosaic shape")
            _require(set(map(int, np.unique(out))) <= {0, d["ink"]}, "unexpected output ink")
            _require(int(np.count_nonzero(out)) == int(d["sites"].sum()) * int(d["tile"].sum()),
                     "not one unchanged-area tile per component")
            _require(2*int(d["reference"].sum()) != d["reference"].size,
                     "reference does not distinguish complement")
            _require(np.array_equal(solve(inp), out), "incorrect mosaic")
            (train if split == "train" else tests).append(d)
    _require(len({d["operation"] for d in train}) >= 2, "missing local-operation variation")
    _require(any(_manhattan_differs(d) for d in train), "missing graph-distance witness")
    _require(any(d["operation"] in (1, 3) and
                 np.count_nonzero(d["sites"][1:] & d["sites"][:-1]) +
                 np.count_nonzero(d["sites"][:, 1:] & d["sites"][:, :-1]) >= d["sites"].sum()
                 for d in train), "shortest routes on cycles are not demonstrated")
    _require(any(a["counter_area"] == b["counter_area"] and a["sites"].sum() != b["sites"].sum()
                 for a, b in combinations(train, 2)), "counter count confounded with area")
    _require(any(d["target_panel"].shape != d["tile"].shape for d in train), "missing crop witness")
    _require(any(len(set(d["node_shapes"])) > 1 for d in train), "anchor alignment is not demonstrated")
    _require(any(len({_key(_transform(d["tile"], i)) for i in range(8)}) == 8 for d in train),
             "target symmetry hides the analogy")
    _require(all(d["tile"].shape[0] <= 4 for d in train) and
             all(d["tile"].shape[0] == 5 for d in tests), "missing scale transfer")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _diagnostics():
    rng = np.random.default_rng(807)
    tile, reference = _motif(rng, 3), _motif(rng, 3)
    sites = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
    pair = _build_pair(rng, tile, reference, 1, sites, (0, 0), (2, 3, 4), [6]*5)
    d = _decode(pair["input"])
    _require(_manhattan_differs(d), "diagnostic path lacks a detour")
    # Reference-only mutation: the same crop and scaffold must give a new answer.
    twin = pair["input"].copy()
    r, c, h, w = d["after_box"]
    alternative = next(i for i in range(1, 8) if i != d["operation"] and
                       not np.array_equal(_render(d["tile"], d["ink"], i, d["sites"], d["origin"]), pair["output"]))
    twin[r+1:r+h-1, c+1:c+w-1] = np.where(_transform(d["reference"], alternative), d["ref_ink"], 0)
    _require(not np.array_equal(solve(twin), pair["output"]), "reference can be ignored")
    # Reference order is irrelevant: the marked corner identifies the source.
    swapped = pair["input"].copy()
    rb, cb, hb, wb = d["before_box"]
    ra, ca, ha, wa = d["after_box"]
    before_pixels = swapped[rb+1:rb+hb-1, cb+1:cb+wb-1].copy()
    after_pixels = swapped[ra+1:ra+ha-1, ca+1:ca+wa-1].copy()
    swapped[rb+1:rb+hb-1, cb+1:cb+wb-1] = after_pixels
    swapped[ra+1:ra+ha-1, ca+1:ca+wa-1] = before_pixels
    swapped[rb, cb], swapped[ra, ca] = FRAME, START
    _require(np.array_equal(solve(swapped), pair["output"]), "reference position determines direction")
    # Change one object's area without changing its anchor or site membership.
    reshaped = pair["input"].copy()
    anchor = next(p for p in d["anchors"] if reshaped[p] != START)
    y, x = anchor
    reshaped[y:y+3, x:x+3] = 0
    reshaped[anchor] = d["graph_ink"]
    _require(np.array_equal(solve(reshaped), pair["output"]), "object area changes the answer")
    # Add a bridge directly to the input; the solver must update an existing tile.
    bridge = pair["input"].copy()
    missing = tuple(map(int, np.argwhere(~d["sites"])[0]))
    y, x = d["rows"][missing[0]], d["cols"][missing[1]]
    bridge[y, x:x+2] = d["graph_ink"]
    bridged_answer = solve(bridge)
    endpoint = tuple(map(int, np.argwhere(d["distance"] == d["distance"].max())[0]))
    k = d["tile"].shape[0]
    r, c = endpoint[0]*(k+1), endpoint[1]*(k+1)
    _require(not np.array_equal(bridged_answer[r:r+k, c:c+k], pair["output"][r:r+k, c:c+k]),
             "input bridge does not affect existing tile state")
    # Input transposition conjugates the operation and transposes the whole mosaic.
    _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "transpose transfer failed")
    _require(not np.array_equal(_render(d["tile"], d["ink"], 0, d["sites"], d["origin"]), pair["output"]),
             "identity ablation does not change answer")
    _require(not np.array_equal(np.where(d["tile"], d["ink"], 0), pair["output"]),
             "counter omission does not change answer")
    # A fitting family member beyond the routine sampler, at the output bound.
    largest = _build_pair(rng, _motif(rng, 5), reference, 7, np.ones((5, 5), bool),
                          (2, 2), (2, 3, 4))
    _require(largest["output"].shape == (29, 29) and
             np.array_equal(solve(largest["input"]), largest["output"]), "boundary extension failed")


def self_test():
    """Bounded semantic checks and pixel-derived diversity across 42 seeds."""
    shapes, outputs, tile_masks, graphs, operations, palettes, origins, object_shapes = (set() for _ in range(8))
    signatures = set()
    pair_count = branches = cycles = detours = disconnected_tiles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                pair_count += 1
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                tile_masks.add(_key(d["tile"]))
                graphs.add(_key(d["sites"]))
                operations.add(d["operation"])
                palettes.add((d["ref_ink"], d["ink"], d["graph_ink"]))
                origins.add((d["sites"].shape, d["origin"]))
                object_shapes.update(d["node_shapes"])
                disconnected_tiles += len(_components(d["tile"])) > 1
                degree = []
                for r, c in map(tuple, np.argwhere(d["sites"])):
                    degree.append(sum(0 <= y < d["sites"].shape[0] and 0 <= x < d["sites"].shape[1]
                                      and d["sites"][y, x] for y, x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))))
                branches += max(degree) >= 3
                cycles += sum(degree)//2 >= int(d["sites"].sum())
                detours += _manhattan_differs(d)
    _require(len(signatures) == 42, "duplicate seeds")
    _require(operations == set(range(1, 8)), "symmetry coverage collapsed")
    _require(len(shapes) >= 25 and len(outputs) >= 15, "geometry diversity collapsed")
    _require(len(tile_masks) >= 100 and len(graphs) >= 40, "structural diversity collapsed")
    _require(len(palettes) >= 30 and len(origins) >= 30 and len(object_shapes) == 9, "role/site variation collapsed")
    _require(min(branches, cycles, detours, disconnected_tiles) > 0, "missing graph/motif concepts")
    for a, b in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(128, a, b)
        validate_generated_puzzle(p)
        _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    _require(_puzzle_key(generate_puzzle(31)) == _puzzle_key(generate_puzzle(31)), "not deterministic")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = 9
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    _diagnostics()
    return dict(seeds=42, pairs=pair_count, input_shapes=len(shapes), output_shapes=len(outputs),
                binary_stencils=len(tile_masks), site_graphs=len(graphs), operations=sorted(operations),
                role_palettes=len(palettes), origin_configurations=len(origins),
                component_bbox_shapes=len(object_shapes), branching_examples=int(branches),
                cyclic_examples=int(cycles), detour_examples=int(detours),
                disconnected_stencils=int(disconnected_tiles))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
