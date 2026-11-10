"""Corner-frame atlas: register repeated local frames and propagate piece poses.

Only Python's standard library and NumPy are needed. No files or latent state
are consulted by solve(). Generation and self_test() keep all puzzles in RAM.
"""
from collections import Counter, deque
import hashlib
import json

import numpy as np


_SIDES = ("T", "R", "B", "L")
_NORMAL = {"T": np.array((-1, 0)), "R": np.array((0, 1)),
           "B": np.array((1, 0)), "L": np.array((0, -1))}
_MATRICES = tuple(np.array(m, dtype=int) for m in (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
    ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0))))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    return tuple(g.shape), np.asarray(g, dtype=np.int8).tobytes()


def _normalise(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): v for (r, c), v in cells.items()}


def _render(cells, bg):
    cells = _normalise(cells)
    out = np.full((max(r for r, c in cells)+1,
                   max(c for r, c in cells)+1), bg, dtype=np.int8)
    for p, v in cells.items():
        out[p] = v
    return out


def _move(cells, matrix, offset=(0, 0)):
    a, b = map(int, matrix[0]); c, d = map(int, matrix[1])
    x, y = map(int, offset)
    return {(a*r+b*s+x, c*r+d*s+y): v for (r, s), v in cells.items()}


def _union(parts):
    cells = {}
    for part in parts:
        for p, v in part.items():
            if p in cells and cells[p] != v:
                raise ValueError("unequal colors at an overlap")
            cells[p] = v
    return cells


def _components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q); group.add(q); stack.append(q)
        groups.append({p: cells[p] for p in group})
    return groups


def _side(g, s):
    return {"T": g[0, :], "R": g[:, -1],
            "B": g[-1, :], "L": g[:, 0]}[s]


def _side_from_vector(vector):
    return next(s for s in _SIDES if np.array_equal(_NORMAL[s], vector))


def _frame(g):
    cores = [_side(g, s)[1:-1] for s in _SIDES]
    if any(len(v) < 2 or np.any(v != v[0]) for v in cores):
        raise ValueError("outer sides must be uninterrupted")
    colors = [int(v[0]) for v in cores]
    counts = Counter(colors)
    if sorted(counts.values()) != [1, 1, 2]:
        raise ValueError("two control sides and two background sides required")
    bg = next(v for v, n in counts.items() if n == 2)
    outer = {s: v for s, v in zip(_SIDES, colors) if v != bg}
    indices = [_SIDES.index(s) for s in outer]
    if (indices[0]-indices[1]) % 4 not in (1, 3):
        raise ValueError("controls must be adjacent")
    row = 0 if "T" in outer else g.shape[0]-1
    col = 0 if "L" in outer else g.shape[1]-1
    anchor = int(g[row, col])
    if anchor in (bg, *outer.values()):
        raise ValueError("the shared corner must name the anchor")
    return bg, outer, anchor


def _sockets(part, markers, bg):
    """Decode complete perpendicular bars, then canonicalise their payloads.

    markers[0] is sent to top and markers[1] to left solely to make a key.
    This canonical key frame is NOT an instruction to orient every piece.
    """
    bars = []
    for color in markers:
        for group in _components({p: v for p, v in part.items() if v == color}):
            rows = sorted({p[0] for p in group})
            cols = sorted({p[1] for p in group})
            if len(rows) == 1 and len(cols) == len(group) >= 2 and cols[-1]-cols[0]+1 == len(cols):
                bars.append(("H", color, rows[0], cols[0], cols[-1], frozenset(group)))
            elif len(cols) == 1 and len(rows) == len(group) >= 2 and rows[-1]-rows[0]+1 == len(rows):
                bars.append(("V", color, cols[0], rows[0], rows[-1], frozenset(group)))
            else:
                raise ValueError("each interior marker component must be a straight bar")
    found, usage = [], Counter()
    for i, hb in enumerate(bars):
        if hb[0] != "H":
            continue
        _, hc, hr, h0, h1, hp = hb
        for j, vb in enumerate(bars):
            if vb[0] != "V" or vb[1] == hc:
                continue
            _, vc, vcol, v0, v1, vp = vb
            hs = "T" if v0 == hr+1 else "B" if v1 == hr-1 else None
            vs = "L" if vcol == h0-1 else "R" if vcol == h1+1 else None
            if hs is None or vs is None:
                continue
            r0, c0, h, w = v0, h0, v1-v0+1, h1-h0+1
            guides = {hs: hc, vs: vc}
            payload = np.array([[part.get((r0+r, c0+c), bg) for c in range(w)]
                                for r in range(h)], dtype=np.int8)
            if set(np.unique(payload)) & set(markers):
                continue
            corner = (hr, vcol)
            if part.get(corner, bg) != bg:
                continue
            matrix = np.zeros((2, 2), dtype=int)
            for side, color in guides.items():
                target = "T" if color == markers[0] else "L"
                matrix += np.outer(_NORMAL[target], _NORMAL[side])
            coords = np.indices((h, w)).reshape(2, -1)
            moved = matrix @ coords
            lo = moved.min(axis=1)
            moved -= lo[:, None]
            canonical = np.empty(tuple(moved.max(axis=1)+1), dtype=np.int8)
            canonical[moved[0], moved[1]] = payload.ravel()
            origin = np.array((r0, c0)) + matrix.T @ lo
            # Includes the missing corner and background-valued payload cells.
            box = (min(hr, v0), min(vcol, h0), max(hr, v1), max(vcol, h1))
            observed = {(r, c): part.get((r, c), bg)
                        for r in range(box[0], box[2]+1)
                        for c in range(box[1], box[3]+1)}
            found.append(dict(key=_key(canonical), canonical=canonical,
                              matrix=matrix, origin=origin, guides=guides,
                              bars=hp | vp, observed=observed, bounds=(r0, c0, h, w)))
            usage[i] += 1; usage[j] += 1
    if not found or any(usage[i] != 1 for i in range(len(bars))):
        raise ValueError("bars must define unambiguous complete sockets")
    if len({s["key"] for s in found}) != len(found):
        raise ValueError("a key may occur at most once in a fragment")
    return found


def _decode(input_grid):
    g = _grid(input_grid)
    bg, outer, anchor = _frame(g)
    markers = tuple(sorted(outer.values()))
    if (np.any(g[1, 1:-1] != bg) or np.any(g[-2, 1:-1] != bg)
            or np.any(g[1:-1, 1] != bg) or np.any(g[1:-1, -2] != bg)):
        raise ValueError("a clear moat must separate the frame")
    cells = {(int(r)+1, int(c)+1): int(g[r+1, c+1]) for r, c in
             zip(*np.nonzero(g[1:-1, 1:-1] != bg))}
    parts = _components(cells)
    if not 2 <= len(parts) <= 6:
        raise ValueError("need two to six separated fragments")
    sockets = [_sockets(p, markers, bg) for p in parts]
    occurrences = {}
    roots = []
    for i, ss in enumerate(sockets):
        for s in ss:
            occurrences.setdefault(s["key"], []).append((i, s))
            if np.any(s["canonical"] == anchor):
                if np.count_nonzero(s["canonical"] == anchor) != 1:
                    raise ValueError("one anchor dot per anchor copy")
                roots.append((i, s))
    if not roots or len({s["key"] for i, s in roots}) != 1:
        raise ValueError("exactly one anchor motif identity is required")
    root_cells = {p for i, s in roots for p, v in s["observed"].items() if v == anchor}
    if {p for p, v in cells.items() if v == anchor} != root_cells:
        raise ValueError("anchor color may occur only in the anchor motif")
    adjacency = [[] for p in parts]
    for group in occurrences.values():
        i, si = group[0]
        for j, sj in group[1:]:
            adjacency[i].append((j, si, sj))
            adjacency[j].append((i, sj, si))
    ri, rs = roots[0]
    root_matrix = np.zeros((2, 2), dtype=int)
    for side, color in rs["guides"].items():
        target = next(s for s, v in outer.items() if v == color)
        root_matrix += np.outer(_NORMAL[target], _NORMAL[side])
    poses = {ri: (root_matrix, np.zeros(2, dtype=int))}
    queue = deque([ri])
    while queue:
        i = queue.popleft()
        mi, oi = poses[i]
        for j, si, sj in adjacency[i]:
            mj = mi @ si["matrix"].T @ sj["matrix"]
            oj = mi @ si["origin"] + oi - mj @ sj["origin"]
            if j in poses:
                if not (np.array_equal(poses[j][0], mj) and np.array_equal(poses[j][1], oj)):
                    raise ValueError("inconsistent registration cycle")
            else:
                poses[j] = mj, oj
                queue.append(j)
    if len(poses) != len(parts):
        raise ValueError("the shared-key graph must connect all fragments")
    placed = [_move(part, *poses[i]) for i, part in enumerate(parts)]
    merged = _union(placed)
    for i, ss in enumerate(sockets):
        for s in ss:
            for p, v in _move(s["observed"], *poses[i]).items():
                if merged.get(p, bg) != v:
                    raise ValueError("an observed key cell or hole was overwritten")
    for i, s in roots:
        for side, color in s["guides"].items():
            target = _side_from_vector(poses[i][0] @ _NORMAL[side])
            if outer.get(target) != color:
                raise ValueError("anchor copies disagree with outer controls")
    output = _render(merged, bg)
    if max(output.shape) > 30:
        raise ValueError("assembled diagram exceeds 30 cells")
    return dict(output=output, background=bg, markers=markers, anchor=anchor,
                outer=outer, parts=parts, sockets=sockets, occurrences=occurrences,
                poses=poses, placed=placed, root=ri)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all local frames, relative translations and the crop from pixels."""
    return _decode(input_grid)["output"]


def _payload(rng, palette, shape, bg, symmetric=False, anchor=None):
    h, w = shape
    fill = int(_pick(rng, palette))
    p = np.full(shape, fill, dtype=np.int8)
    style = int(rng.integers(5))
    accent = int(_pick(rng, palette))
    if style == 0:
        p[h//2, :] = accent
    elif style == 1:
        p[:, w//2] = accent; p[-1, w//2:] = accent
    elif style == 2:
        p[::2, ::2] = accent
    elif style == 3:
        mask = rng.random(shape) < .35
        p[mask] = rng.choice(palette, int(mask.sum()))
    else:
        p[0, :] = accent; p[:, -1] = accent
    if h >= 3 and w >= 3 and rng.random() < .5:
        p[1:-1, 1:-1] = bg
    if symmetric:
        p = np.triu(p) + np.triu(p, 1).T
    if anchor is not None:
        p[int(rng.integers(h)), int(rng.integers(w))] = anchor
    return p


def _stamp(payload, markers, pose):
    h, w = payload.shape
    # The canonical guide corner is empty; no universal literal colors exist.
    bg = int(markers[2])
    a = np.full((h+1, w+1), bg, dtype=np.int8)
    a[1:, 1:] = payload
    a[0, 1:] = markers[0]; a[1:, 0] = markers[1]
    cells = {(r, c): int(a[r, c]) for r in range(h+1) for c in range(w+1)}
    moved = _normalise(_move(cells, _MATRICES[pose]))
    result = _render(moved, bg)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _near_box(p, box, halo=0):
    r, c = p; r0, c0, r1, c1 = box
    return r0-halo <= r <= r1+halo and c0-halo <= c <= c1+halo


def _edge_path(rng, left, right, direction):
    """Connect facing nonempty boundaries through a fresh variable bend."""
    b0, b1 = _box(left), _box(right)
    axis = 1 if direction in (1, 3) else 0
    sign = 1 if direction in (1, 2) else -1
    edge0 = b0[axis+2] if sign == 1 else b0[axis]
    edge1 = b1[axis] if sign == 1 else b1[axis+2]
    p = _pick(rng, [p for p in left if p[axis] == edge0])
    q = _pick(rng, [p for p in right if p[axis] == edge1])
    bend = int(rng.integers(min(edge0, edge1)+1, max(edge0, edge1)))
    u, v = list(p), list(q); u[axis] = v[axis] = bend
    path = set()
    for a, b in ((p, u), (u, v), (v, q)):
        for r in range(min(a[0], b[0]), max(a[0], b[0])+1):
            for c in range(min(a[1], b[1]), max(a[1], b[1])+1):
                path.add((r, c))
    return path, p, q


def _layout(rng, stamps, stroke, bg, straight=False):
    nodes, boxes, edges = [], [], []
    for index, stamp in enumerate(stamps):
        h, w = stamp.shape
        local = {(r, c): int(stamp[r, c]) for r, c in
                 zip(*np.nonzero(stamp != bg))}
        if index == 0:
            nodes.append(local); boxes.append((0, 0, h-1, w-1)); continue
        for attempt in range(100):
            parent = index-1 if straight else int(rng.integers(index))
            direction = 1 if straight else int(rng.integers(4))
            pr0, pc0, pr1, pc1 = boxes[parent]
            gap = int(rng.integers(2, 5))
            if direction in (1, 3):
                r0 = 0 if straight else pr0+int(rng.integers(-2, 3))
                c0 = pc1+gap+1 if direction == 1 else pc0-gap-w
            else:
                c0 = pc0+int(rng.integers(-2, 3))
                r0 = pr1+gap+1 if direction == 2 else pr0-gap-h
            box = r0, c0, r0+h-1, c0+w-1
            if any(not (box[2]+1 < b[0] or b[2]+1 < box[0]
                        or box[3]+1 < b[1] or b[3]+1 < box[1]) for b in boxes):
                continue
            new = _move(local, np.eye(2, dtype=int), (r0, c0))
            path, p, q = _edge_path(rng, nodes[parent], new, direction)
            if any(_near_box(x, b, 1) for x in path
                   for j, b in enumerate(boxes) if j != parent):
                continue
            if any((_near_box(x, boxes[parent]) and x != p)
                   or (_near_box(x, box) and x != q) for x in path):
                continue
            fullbox = (min(b[0] for b in boxes+[box]), min(b[1] for b in boxes+[box]),
                       max(b[2] for b in boxes+[box]), max(b[3] for b in boxes+[box]))
            if max(fullbox[2]-fullbox[0]+1, fullbox[3]-fullbox[1]+1) > (29 if straight else 24):
                continue
            link = {x: stroke for x in path if x not in (p, q)}
            nodes.append(new); boxes.append(box); edges.append((parent, index, link)); break
        else:
            raise ValueError("bounded layout attempt did not fit")
    return nodes, edges, boxes


def _pieces(rng, nodes, edges, count, redundant=False):
    groups = [{i} for i in range(len(edges))]
    while len(groups) > count:
        options = []
        for i in range(len(groups)):
            ni = {v for e in groups[i] for v in edges[e][:2]}
            for j in range(i+1, len(groups)):
                nj = {v for e in groups[j] for v in edges[e][:2]}
                if ni & nj:
                    options.append((i, j))
        i, j = _pick(rng, options)
        groups[i] |= groups[j]; del groups[j]
    parts = []
    for group in groups:
        ids = {v for e in group for v in edges[e][:2]}
        parts.append(_union([nodes[v] for v in sorted(ids)] + [edges[e][2] for e in sorted(group)]))
    if redundant:
        parts.append(dict(_pick(rng, nodes)))
    return parts


def _pack(rng, parts, bg):
    posed = [_normalise(_move(p, _MATRICES[int(rng.integers(8))])) for p in parts]
    shapes = [(max(r for r, c in p)+1, max(c for r, c in p)+1) for p in posed]
    mh, mw = max(h for h, w in shapes), max(w for h, w in shapes)
    if max(mh, mw) > 26:
        raise ValueError("a piece exceeds the inner canvas")
    for attempt in range(45):
        h = int(rng.integers(mh, 27)); w = int(rng.integers(mw, 27))
        if attempt >= 30:
            h = w = 26
        if h*w < sum((x+1)*(y+1) for x, y in shapes):
            continue
        placed, result = [], np.full((h+4, w+4), bg, dtype=np.int8)
        order = sorted(range(len(parts)), key=lambda i: -(shapes[i][0]*shapes[i][1]+float(rng.random())))
        for i in order:
            ph, pw = shapes[i]
            choices = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)
                       if all(r+ph < x or x+hh < r or c+pw < y or y+ww < c
                              for x, y, hh, ww in placed)]
            if not choices:
                break
            r0, c0 = _pick(rng, choices)
            placed.append((r0, c0, ph, pw))
            for (r, c), v in posed[i].items():
                result[r+r0+2, c+c0+2] = v
        else:
            return result
    raise ValueError("bounded packing attempts exhausted")


def _paint_frame(g, outer, anchor):
    for side, color in outer.items():
        _side(g, side)[:] = color
    g[0 if "T" in outer else -1, 0 if "L" in outer else -1] = anchor


def _make_pair(rng, palette, n, piece_count, outer_corner, contrast=False, redundant=False):
    bg, ma, mb, anchor, stroke = palette[:5]
    colors = palette[5:]
    # Payloads use background only in holes; the other colors are ordinary ink.
    keys, stamps, node_poses = set(), [], []
    for i in range(n):
        for _ in range(40):
            h = int(rng.integers(2, 5)); w = int(rng.integers(2, 6))
            symmetric = contrast and i == 1
            if symmetric:
                w = h
            payload = _payload(rng, colors, (h, w), bg, symmetric, anchor if i == 0 else None)
            if _key(payload) not in keys:
                keys.add(_key(payload)); break
        else:
            raise ValueError("could not obtain distinct key identities")
        pose = int(rng.integers(8))
        if contrast and i == 1:
            # Different local corners within the anchor-containing fragment.
            pose = (node_poses[0]+int(rng.integers(1, 4))) % 4
        node_poses.append(pose)
        stamps.append(_stamp(payload, (ma, mb, bg), pose))
    nodes, edges, boxes = _layout(rng, stamps, stroke, bg, straight=contrast)
    parts = _pieces(rng, nodes, edges, piece_count, redundant)
    grid = _pack(rng, parts, bg)
    # The target corner is held out when that is this puzzle's shift regime.
    root_sides = [_side_from_vector(_MATRICES[node_poses[0]] @ _NORMAL[s]) for s in ("T", "L")]
    targets = (_SIDES[outer_corner], _SIDES[(outer_corner+1) % 4])
    if rng.integers(2):
        targets = targets[::-1]
    matrix = sum((np.outer(_NORMAL[t], _NORMAL[s]) for s, t in zip(root_sides, targets)),
                 np.zeros((2, 2), dtype=int))
    outer = dict(zip(targets, (ma, mb)))
    _paint_frame(grid, outer, anchor)
    expected = _render(_move(_union(parts), matrix), bg)
    info = _decode(grid)
    if not np.array_equal(info["output"], expected):
        raise ValueError("constructed union disagrees with pixel reconstruction")
    pair = {"input": grid, "output": expected}
    if not contrast:
        return pair
    # Independently construct the reflected continuation in assembly coordinates.
    swapped = dict(parts[1])
    for p, v in nodes[1].items():
        if v in (ma, mb):
            swapped[p] = mb if v == ma else ma
    transpose = _MATRICES[6]
    reflection = _MATRICES[node_poses[1]] @ transpose @ _MATRICES[node_poses[1]].T
    r0, c0, r1, c1 = boxes[1]
    center_twice = np.array((r0+r1, c0+c1))
    off_twice = center_twice-reflection @ center_twice
    if np.any(off_twice % 2):
        raise ValueError("nonintegral contrast reflection")
    other = _move(swapped, reflection, off_twice//2)
    second_expected = _render(_move(_union([parts[0], other]), matrix), bg)
    changed = grid.copy()
    shared = [(key, occ) for key, occ in info["occurrences"].items() if len(occ) == 2]
    if len(shared) != 1:
        raise ValueError("contrast needs a single shared socket")
    nonroot = next((i, s) for i, s in shared[0][1] if i != info["root"])
    for p in nonroot[1]["bars"]:
        changed[p] = mb if int(changed[p]) == ma else ma
    if not np.array_equal(solve(changed), second_expected):
        raise ValueError("contrast geometry is inconsistent")
    if _content_key(expected, bg, (ma, mb)) == _content_key(second_expected, bg, (ma, mb)):
        raise ValueError("guide swap must move ordinary content")
    return pair, {"input": changed, "output": second_expected}


def _content_key(g, bg, markers=()):
    cells = {(int(r), int(c)): int(g[r, c]) for r, c in
             zip(*np.nonzero(~np.isin(g, (bg, *markers))))}
    return _key(_render(cells, bg))


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held = int(rng.integers(4))
    shift = int(rng.integers(2))  # held-out anchor corner or longer propagation
    stable = bool(rng.random() < .25)
    base_palette = [int(c) for c in rng.permutation(10)]
    base_palette = base_palette[:int(rng.integers(6, 11))]
    train_corners = [c for c in range(4) if not shift or c != held]
    for _ in range(100):
        try:
            first, second = _make_pair(rng, base_palette, 3, 2, int(_pick(rng, train_corners)), contrast=True)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("bounded contrast construction exhausted")
    result = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", nt-2), ("test", ne)):
        for index in range(count):
            for attempt in range(100):
                try:
                    palette = base_palette if stable else [int(c) for c in rng.permutation(10)][:int(rng.integers(6, 11))]
                    if not shift:
                        n = int(rng.integers(5, 7)) if split == "test" else int(rng.integers(3, 5))
                        pieces = int(rng.integers(3, min(5, n))) if split == "test" else 2
                    else:
                        n = int(rng.integers(3, 6))
                        pieces = int(rng.integers(2, min(4, n)))
                    corner = held if split == "test" and shift else int(_pick(rng, train_corners))
                    redundant = bool(rng.random() < .25)
                    pair = _make_pair(rng, palette, n, pieces, corner, redundant=redundant)
                    key = _key(pair["input"])
                    if key in seen:
                        continue
                    seen.add(key); result[split].append(pair); break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded pair construction exhausted")
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(inp)
            assert np.array_equal(info["output"], out), "wrong registration, union, or crop"
            bg = info["background"]
            all_placed = set().union(*(set(p) for p in info["placed"]))
            contributions = [len(set(p)-set().union(*(set(q) for j, q in enumerate(info["placed"]) if j != i)))
                             for i, p in enumerate(info["placed"])]
            assert len(all_placed) < sum(len(p) for p in info["parts"]), "no actual overlap"
            assert sum(c > 0 for c in contributions) >= 2, "no meaningful multi-piece union"
            anchored = _render(info["placed"][info["root"]], bg)
            assert _key(anchored) != _key(out), "ignoring other pieces would give the same answer"
            assert set(np.unique(out))-{bg} == set(np.unique(inp[1:-1, 1:-1]))-{bg}
            info["contributions"] = contributions
            infos[split].append(info)
    # Two exact teaching inputs differ only at one shared socket's guide bars.
    a, b = [np.asarray(p["input"]) for p in puzzle["train"][:2]]
    ia, ib = infos["train"][:2]
    assert a.shape == b.shape and ia["outer"] == ib["outer"]
    diff = a != b
    assert diff.any() and np.all(np.isin(a[diff], ia["markers"])) and np.all(np.isin(b[diff], ia["markers"]))
    assert any(set(zip(*np.nonzero(diff))) == set(s["bars"])
               for ss in ia["sockets"] for s in ss), "contrast must exchange one whole guide pair"
    assert _content_key(ia["output"], ia["background"], ia["markers"]) != _content_key(ib["output"], ib["background"], ib["markers"])
    return None


def _metrics(info):
    ss = [s for group in info["sockets"] for s in group]
    bg = info["background"]
    return dict(pieces=len(info["parts"]), keys=len(info["occurrences"]),
                multiplicity=max(map(len, info["occurrences"].values())),
                key_shapes={tuple(s["canonical"].shape) for s in ss},
                key_holes=sum(np.count_nonzero(s["canonical"] == bg) for s in ss),
                corner=tuple(s for s in _SIDES if s in info["outer"]),
                poses={tuple(m.flat) for m, o in info["poses"].values()},
                multi_key=any(len(group) >= 3 for group in info["sockets"]))


def self_test():
    """42 seeds, count boundaries, causal witnesses and invalid-grid mutations.

    Puzzles are streamed; determinism briefly retains a second copy. No grids
    are written, and no execution traces or retry logs are retained.
    Returned metrics are compact-JSON serialisable aggregate measurements.
    """
    signatures, in_shapes, out_shapes, key_shapes = set(), set(), set(), set()
    pieces, keys, multiplicities, poses, corners = set(), set(), set(), set(), set()
    holes = multi = pairs = redundant = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _decode(inp); m = _metrics(info)
                digest.update(bytes(inp.shape)); digest.update(inp.tobytes()); digest.update(out.tobytes())
                in_shapes.add(inp.shape); out_shapes.add(out.shape); key_shapes |= m["key_shapes"]
                pieces.add(m["pieces"]); keys.add(m["keys"]); multiplicities.add(m["multiplicity"])
                poses |= m["poses"]; corners.add(m["corner"])
                holes += m["key_holes"] > 0; multi += m["multi_key"]; pairs += 1
                placed = info["placed"]
                redundant += any(set(p) <= set().union(*(set(q) for j, q in enumerate(placed) if j != i))
                                 for i, p in enumerate(placed))
        signatures.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(_key(puzzle[s][i][k]) == _key(again[s][i][k])
                       for s in puzzle for i in range(len(puzzle[s])) for k in ("input", "output"))
            del again
            # Observable conventions survive whole-grid symmetries, arbitrary
            # color renamings, and fresh independent poses/packing of pieces.
            example = puzzle["test"][0]
            original_info = _decode(example["input"])
            for matrix in _MATRICES:
                moved = {}
                for role, g in example.items():
                    all_cells = {(r, c): int(g[r, c]) for r in range(g.shape[0])
                                 for c in range(g.shape[1])}
                    moved[role] = _render(_move(all_cells, matrix), original_info["background"])
                assert np.array_equal(solve(moved["input"]), moved["output"])
            for trial in range(3):
                permutation = np.random.default_rng(900+trial).permutation(10)
                assert np.array_equal(solve(permutation[example["input"]]), permutation[example["output"]])
            repacked = _pack(np.random.default_rng(421), original_info["parts"], original_info["background"])
            _paint_frame(repacked, original_info["outer"], original_info["anchor"])
            assert np.array_equal(solve(repacked), example["output"])
            for kind in ("cell", "crop", "bar", "anchor"):
                bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
                pair = bad["test"][0]; info = _decode(pair["input"])
                if kind == "cell":
                    pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
                elif kind == "crop":
                    pair["output"] = pair["output"][1:, :]
                elif kind == "bar":
                    p = min(info["sockets"][0][0]["bars"])
                    pair["input"][p] = info["background"]
                else:
                    r = 0 if "T" in info["outer"] else -1
                    c = 0 if "L" in info["outer"] else -1
                    pair["input"][r, c] = info["background"]
                try:
                    validate_generated_puzzle(bad)
                except (AssertionError, ValueError):
                    pass
                else:
                    raise AssertionError("invalid mutation accepted")
    assert len(signatures) == 42 and len(in_shapes) >= 25 and len(out_shapes) >= 30
    assert pieces >= {2, 3, 4} and keys >= {3, 4, 5} and max(multiplicities) >= 3
    assert len(poses) == 8 and len(corners) == 4 and len(key_shapes) >= 6
    assert holes >= 10 and multi >= 5 and redundant >= 5
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(700+nt*10+ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
            validate_generated_puzzle(p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pairs, input_shapes=len(in_shapes), output_shapes=len(out_shapes),
                key_shapes=len(key_shapes), piece_counts=sorted(pieces), key_counts=sorted(keys),
                multiplicities=sorted(multiplicities), poses=len(poses), corners=len(corners),
                pairs_with_key_holes=int(holes), pairs_with_multi_key_fragments=int(multi),
                pairs_with_redundancy=int(redundant))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
