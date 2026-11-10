"""Canonical clear-space atlas: topology normalization followed by local erosion.

No imports of parent generators, stored examples, seed-dependent solver state,
network access, files, or non-NumPy dependencies. Diagnostics stay in memory.
"""
from __future__ import annotations

import hashlib
import numpy as np


HIGHLIGHT = 3


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups = []
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        todo = [(int(row), int(col))]
        seen[row, col] = True
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    todo.append((y, x))
        groups.append(np.asarray(cells, dtype=int))
    return groups


def _holes(mask):
    """Enclosure by this component alone, including any enclosed other objects."""
    padded = np.pad(mask, 1)
    exterior = np.zeros(padded.shape, dtype=bool)
    exterior[0, 0] = True
    todo = [(0, 0)]
    h, w = padded.shape
    while todo:
        r, c = todo.pop()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                todo.append((y, x))
    return (~padded & ~exterior)[1:-1, 1:-1]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _parse(grid, background=None):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2, "binary drawing required"
    if background is None:
        assert counts.max()*2 > grid.size, "background must be a strict majority"
        background = int(values[np.argmax(counts)])
    foreground = int(values[values != background][0])
    objects = []
    for cells in _components(grid == foreground):
        low, high = cells.min(0), cells.max(0)
        local = np.zeros(tuple(high-low+1), dtype=bool)
        local[tuple((cells-low).T)] = True
        enclosed = _holes(local)
        hole = np.zeros(grid.shape, dtype=bool)
        hole[low[0]:high[0]+1, low[1]:high[1]+1] = enclosed
        contour = bool(enclosed.any())
        assert len(_components(enclosed)) <= 1, "multiple holes"
        if not contour:
            assert local.sum() >= .35*local.size, "noncompact blob"
        objects.append(dict(cells=cells, lo=low, hi=high, hole=hole,
                            kind="C" if contour else "P", area=len(cells)))
    assert any(o["kind"] == "C" for o in objects), "no contour"
    labels = np.zeros(grid.shape, dtype=int)
    for i, obj in enumerate(objects, 1):
        labels[tuple(obj["cells"].T)] = i
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]),
                 (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b)), "touching components"
    parents = []
    for i, child in enumerate(objects):
        choices = []
        for j, outer in enumerate(objects):
            if i == j or outer["kind"] != "C":
                continue
            inside = outer["hole"][tuple(child["cells"].T)]
            assert inside.all() or not inside.any(), "crossing components"
            if inside.all():
                choices.append(j)
        parents.append(min(choices, key=lambda k: int(objects[k]["hole"].sum()))
                       if choices else None)
    children = [[] for _ in objects]
    roots = []
    for i, parent in enumerate(parents):
        (roots if parent is None else children[parent]).append(i)
    for i, obj in enumerate(objects):
        assert bool(children[i]) == (obj["kind"] == "C"), "empty contour"
    return dict(bg=background, fg=foreground, objects=objects, parents=parents,
                children=children, roots=roots)


# A tile is (foreground mask, [(point identity, row, column), ...]).
def _join(tiles, axis, offsets, gaps):
    starts, along = [], 0
    cross = 1-axis
    low = min(offsets)
    size = [0, 0]
    for i, (mask, _) in enumerate(tiles):
        start = [0, 0]
        start[axis], start[cross] = along, offsets[i]-low
        starts.append(start)
        along += mask.shape[axis] + (gaps[i] if i < len(gaps) else 0)
        size[cross] = max(size[cross], start[cross]+mask.shape[cross])
    size[axis] = along
    result = np.zeros(size, dtype=bool)
    leaves = []
    for (mask, points), (r, c) in zip(tiles, starts):
        h, w = mask.shape
        result[r:r+h, c:c+w] |= mask
        leaves.extend((k, y+r, x+c) for k, y, x in points)
    return result, leaves


def _frame(tile):
    mask, leaves = tile
    result = np.pad(mask, 2)
    result[0] = result[-1] = True
    result[:, 0] = result[:, -1] = True
    return result, [(k, y+2, x+2) for k, y, x in leaves]


def _halo(tile, exterior):
    mask, leaves = tile
    h, w = mask.shape
    t = b = l = r = 0
    for k, y, x in leaves:
        if k in exterior:
            t, b = max(t, 1-y), max(b, y+2-h)
            l, r = max(l, 1-x), max(r, x+2-w)
    return np.pad(mask, ((t, b), (l, r)))


def _normalize(info):
    """Infer all consistent slicing decompositions, never metric-center ties."""
    objects = info["objects"]
    memo = {}

    def align(tiles, axis):
        cross = 1-axis
        edges = []
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                shifts = set()
                for a, ay, ax in tiles[i][1]:
                    for b, by, bx in tiles[j][1]:
                        if max(objects[a]["lo"][cross], objects[b]["lo"][cross]) <= min(objects[a]["hi"][cross], objects[b]["hi"][cross]):
                            shifts.add((ay, ax)[cross]-(by, bx)[cross])
                assert len(shifts) <= 1, "competing point alignments"
                if shifts:
                    edges.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in tiles:
            for i, j, delta in edges:
                if i in offsets and j in offsets:
                    assert offsets[j] == offsets[i]+delta, "inconsistent alignment"
                elif i in offsets:
                    offsets[j] = offsets[i]+delta
                elif j in offsets:
                    offsets[i] = offsets[j]-delta
        assert len(offsets) == len(tiles), "unanchored diagonal layout"
        return [offsets[i] for i in range(len(tiles))]

    def compose(ids):
        key = tuple(sorted(ids))
        if key in memo:
            return memo[key]
        if len(key) == 1:
            i = key[0]
            result = ((np.ones((1, 1), bool), [(i, 0, 0)])
                      if objects[i]["kind"] == "P"
                      else _frame(compose(info["children"][i])))
        else:
            candidates = []
            for axis in (0, 1):
                partitions, end = [], -2
                for i in sorted(key, key=lambda j: int(objects[j]["lo"][axis])):
                    start = int(objects[i]["lo"][axis])
                    if start > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, int(objects[i]["hi"][axis]))
                if len(partitions) < 2:
                    continue
                try:
                    tiles = [compose(part) for part in partitions]
                    candidates.append(_join(tiles, axis, align(tiles, axis),
                                            [1]*(len(tiles)-1)))
                except AssertionError:
                    continue
            assert candidates, "no consistent cardinal slicing"
            result = candidates[0]
            for candidate in candidates[1:]:
                assert np.array_equal(candidate[0], result[0]), "ambiguous layout"
                assert sorted(candidate[1]) == sorted(result[1]), "ambiguous anchors"
        memo[key] = result
        return result

    exterior = {i for i in info["roots"] if objects[i]["kind"] == "P"}
    return _halo(compose(info["roots"]), exterior)


def _clear_centers(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    blocked = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            blocked |= padded[dy:dy+h, dx:dx+w]
    return ~blocked


def _paint(mask, bg, fg):
    out = np.where(mask, fg, bg).astype(np.int8)
    out[_clear_centers(mask)] = HIGHLIGHT
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the canonical canvas and every marked center from the input alone."""
    grid = _grid(input_grid)
    if HIGHLIGHT in grid:
        raise ValueError("3 is reserved for output highlights")
    try:
        info = _parse(grid)
        return _paint(_normalize(info), info["bg"], info["fg"])
    except AssertionError as error:
        raise ValueError(str(error)) from error


# Scene grammar: P; F(content); J(axis, children, selected descendant per child).
P = ("P",)


def _leaves(node):
    if node[0] == "P":
        return 1
    if node[0] == "F":
        return _leaves(node[1])
    return sum(_leaves(child) for child in node[2])


def _group(rng, axis, children, anchors=None):
    if len(children) == 1:
        return children[0]
    if anchors is None:
        anchors = tuple(int(rng.integers(_leaves(child))) for child in children)
    return ("J", axis, tuple(children), tuple(anchors))


def _bay(rng, kind):
    """Broad negative space is induced by normalized extents, never fake padding."""
    if kind == "interior":
        inner = ("F", _group(rng, 0, [P, P]))
        joined = _group(rng, 1, [inner, P], [int(rng.integers(2)), 0])
        return ("F", joined)
    count = int(rng.integers(2, 4))
    inner = ("F", _group(rng, 0, [P]*count))
    points = _group(rng, 1, [P]*int(rng.integers(1, 4)))
    return _group(rng, 1, [inner, points],
                  [int(rng.choice([0, count-1])), int(rng.integers(_leaves(points)))])


def _random_frame(rng, depth=1):
    n = int(rng.integers(1, 5))
    children = [(_random_frame(rng, depth+1)
                 if depth < 3 and rng.random() < (.32 if depth == 1 else .10)
                 else P) for _ in range(n)]
    axis = int(rng.integers(2))
    if n >= 3 and rng.random() < .7:
        cut = int(rng.integers(1, n))
        content = _group(rng, axis, [_group(rng, 1-axis, children[:cut]),
                                      _group(rng, 1-axis, children[cut:])])
    else:
        content = _group(rng, axis, children)
    return ("F", content)


def _scene(rng, rich):
    # These routes share the same grammar and can recombine at any depth.
    mode = int(rng.integers(5 if rich else 3))
    if mode == 0:
        count = int(rng.integers(2, 5))
        inner = ("F", _group(rng, 0, [P]*count))
        row = _group(rng, 1, [P]*int(rng.integers(1, 5)))
        content = _group(rng, 1, [inner, row])
        if rich and rng.random() < .55:
            content = _group(rng, 0, [P, content])
        root = ("F", content)
    elif mode == 1:
        root = _random_frame(rng)
    elif mode == 2:
        root = ("F", _bay(rng, "exterior"))
    else:
        # Unequal sibling frames naturally leave a clear strip above/below
        # the shallower frame, inside their common enclosure.
        deep = ("F", ("F", _group(rng, int(rng.integers(2)), [P]*int(rng.integers(1, 3)))))
        shallow = ("F", P)
        root = ("F", _group(rng, 1, [deep, shallow]))
    if rng.random() < (.55 if rich else .25):
        beside = _group(rng, int(rng.integers(2)), [P]*int(rng.integers(1, 4)))
        root = _group(rng, int(rng.integers(2)), [root, beside])
    return root


def _blob(rng, compact):
    h, w = [int(rng.integers(1, 3 if compact else 5)) for _ in range(2)]
    if max(h, w) > min(h, w)*2+1:
        h = w = 2
    mask = np.ones((h, w), dtype=bool)
    # Corner peeling preserves filled compactness but is checked explicitly.
    for _ in range(int(rng.integers(0, max(2, h*w//2)))):
        y, x = int(rng.integers(h)), int(rng.integers(w))
        candidate = mask.copy()
        candidate[y, x] = False
        if candidate.sum() >= .6*h*w and len(_components(candidate)) == 1 and not _holes(candidate).any():
            mask = candidate
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max()+1, cols.min():cols.max()+1]


def _outline(rng, shape, padding):
    """Rectilinear loops with safe inward bays on independently chosen sides."""
    h, w = shape
    t, b, l, r = padding
    vertices = [(0, 0)]

    def side(start, end, cap, lower, upper, inward, axis):
        # Work along an oriented side, inserting a rectangular detour.
        if cap and upper-lower >= 2 and rng.random() < .75:
            a = int(rng.integers(lower, upper-1))
            z = int(rng.integers(a+1, upper+1))
            d = int(rng.integers(1, cap+1))
            for s, offset in ((a, 0), (a, d), (z, d), (z, 0)):
                p = list(start)
                p[axis] += s if end[axis] > start[axis] else -s
                p[1-axis] += inward*offset
                vertices.append(tuple(p))
        vertices.append(end)

    side((0, 0), (0, w-1), t-2, l-1, w-r, 1, 1)
    side((0, w-1), (h-1, w-1), r-2, t-1, h-b, -1, 0)
    side((h-1, w-1), (h-1, 0), b-2, r-1, w-l, -1, 1)
    side((h-1, 0), (0, 0), l-2, b-1, h-t, 1, 0)
    mask = np.zeros((h, w), dtype=bool)
    for (y, x), (yy, xx) in zip(vertices, vertices[1:]):
        mask[min(y, yy):max(y, yy)+1, min(x, xx):max(x, xx)+1] = True
    return mask


def _draw(node, rng, compact):
    """Return independent metric and canonical tiles from one abstract scene."""
    serial = [0]
    exterior = set()

    def visit(obj, depth):
        if obj[0] == "P":
            k = serial[0]
            serial[0] += 1
            if depth == 0:
                exterior.add(k)
            mask = _blob(rng, compact)
            h, w = mask.shape
            return ((mask, [(k, (h-1)//2, (w-1)//2)]),
                    (np.ones((1, 1), bool), [(k, 0, 0)]))
        if obj[0] == "F":
            metric, canonical = visit(obj[1], depth+1)
            pads = [int(rng.integers(2, 4 if compact else 6)) for _ in range(4)]
            t, b, l, r = pads
            h, w = metric[0].shape
            mask = _outline(rng, (h+t+b, w+l+r), pads)
            mask[t:t+h, l:l+w] |= metric[0]
            leaves = [(k, y+t, x+l) for k, y, x in metric[1]]
            return (mask, leaves), _frame(canonical)
        pairs = [visit(child, depth) for child in obj[2]]
        axis, cross = obj[1], 1-obj[1]
        result = []
        for version in (0, 1):
            tiles = [p[version] for p in pairs]
            offsets = [-tile[1][anchor][1+cross] for tile, anchor in zip(tiles, obj[3])]
            gaps = ([int(rng.integers(1, 4 if compact else 7)) for _ in tiles[:-1]]
                    if version == 0 else [1]*(len(tiles)-1))
            result.append(_join(tiles, axis, offsets, gaps))
        return tuple(result)

    metric, canonical = visit(node, 0)
    return metric[0], _halo(canonical, exterior)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _make_pair(rng, palette, kind, rich, used):
    bg, fg = palette
    for attempt in range(180):
        # Late attempts compact the metric drawing, not the canonical rule.
        scene = _bay(rng, kind) if kind else _scene(rng, rich)
        if attempt >= 140 and not kind:
            scene = ("F", _bay(rng, "exterior"))
        metric, canonical = _draw(scene, rng, compact=attempt >= 4)
        if max(*metric.shape, *canonical.shape) > 30:
            continue
        centers = _clear_centers(canonical)
        if not centers.any():
            continue
        # Independent whole-scene orientation does not encode any new rule.
        rotations = int(rng.integers(4))
        metric, canonical = np.rot90(metric, rotations), np.rot90(canonical, rotations)
        if rng.random() < .5:
            metric, canonical = metric[:, ::-1], canonical[:, ::-1]
        h, w = metric.shape
        height = int(rng.integers(h, min(30, h+7)+1))
        width = int(rng.integers(w, min(30, w+7)+1))
        if rng.random() < .08:
            height = width = 30
        y, x = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
        inp = np.full((height, width), bg, dtype=np.int8)
        inp[y:y+h, x:x+w][metric] = fg
        if _key(inp) in used or np.count_nonzero(inp == bg)*2 <= inp.size:
            continue
        if inp.shape == canonical.shape:
            continue
        out = _paint(canonical, bg, fg)
        try:
            info = _parse(inp)
            assert np.array_equal(_normalize(info), canonical)
            assert np.array_equal(_normalize(_parse(np.where(canonical, fg, bg), bg)), canonical)
            if kind == "interior":
                sizes = sorted(len(g) for g in _components(out == HIGHLIGHT))
                assert sizes == [1, 3]
            if kind == "exterior":
                assert _edge(centers)
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice([0, 1, 2, 4, 5, 6, 7, 8, 9], 2, replace=False))
    stable = rng.random() < .30
    used = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            current = palette if stable else tuple(int(c) for c in rng.choice(
                [0, 1, 2, 4, 5, 6, 7, 8, 9], 2, replace=False))
            kind = ("interior" if i == 0 else "exterior" if i == 1 else None) if split == "train" else None
            pair = _make_pair(rng, current, kind, split == "test", used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _edge(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _topology(info, ids=None):
    ids = info["roots"] if ids is None else ids
    return tuple(sorted(("P",) if info["objects"][i]["kind"] == "P" else
                        ("C", _topology(info, info["children"][i])) for i in ids))


def _literal_centers(mask):
    """Explicit clipped slices used for validation, distinct from shifted ORs."""
    h, w = mask.shape
    return np.array([[not mask[max(0, r-1):min(h, r+2),
                               max(0, c-1):min(w, c+2)].any()
                      for c in range(w)] for r in range(h)], dtype=bool)


def _features(inp, out):
    info = _parse(inp)
    mask = out == info["fg"]
    canonical_info = _parse(np.where(mask, info["fg"], info["bg"]), info["bg"])
    depths = []
    for i, obj in enumerate(info["objects"]):
        if obj["kind"] == "C":
            depth, parent = 1, info["parents"][i]
            while parent is not None:
                depth += 1
                parent = info["parents"][parent]
            depths.append(depth)
    highlight = out == HIGHLIGHT
    interior = highlight & _holes(mask)
    cardinal = mask.copy()
    cardinal[1:] |= mask[:-1]
    cardinal[:-1] |= mask[1:]
    cardinal[:, 1:] |= mask[:, :-1]
    cardinal[:, :-1] |= mask[:, 1:]
    diagonal_only = ~cardinal & ~_clear_centers(mask)
    nonrect = 0
    blob_shapes = set()
    for o in info["objects"]:
        if o["kind"] == "P":
            blob_shapes.add(tuple((o["hi"]-o["lo"]+1).tolist()))
        else:
            local = np.zeros(tuple(o["hi"]-o["lo"]+1), bool)
            local[tuple((o["cells"]-o["lo"]).T)] = True
            nonrect += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
    return dict(topology=_topology(info), depth=max(depths), contours=len(depths),
                points=len(info["objects"])-len(depths), nonrect=nonrect,
                blob_shapes=blob_shapes, interior=int(interior.sum()),
                edge=_edge(highlight), diagonal=bool(diagonal_only.any()),
                core_sizes=tuple(sorted(len(g) for g in _components(highlight))),
                exterior_points=sum(info["objects"][i]["kind"] == "P" for i in info["roots"]),
                canonical=canonical_info)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            assert HIGHLIGHT not in inp
            info = _parse(inp)
            canonical = _normalize(info)
            assert out.shape == canonical.shape
            assert inp.shape != out.shape, "A must affect the canvas"
            centers = _literal_centers(canonical)
            assert centers.any(), "B must affect the answer"
            assert np.array_equal(out == HIGHLIGHT, centers)
            assert np.all(out[canonical] == info["fg"])
            assert np.all(out[~canonical & ~centers] == info["bg"])
            assert set(np.unique(out)) == {info["bg"], info["fg"], HIGHLIGHT}
            # Removing B leaves a distinct binary answer. Removing A gives
            # a different canvas and, on diagnostic examples, different cores.
            binary = np.where(canonical, info["fg"], info["bg"]).astype(np.int8)
            assert not np.array_equal(binary, out)
            normalized_info = _parse(binary, info["bg"])
            assert _topology(info) == _topology(normalized_info)
            assert np.array_equal(_normalize(normalized_info), canonical), "crop/layout not canonical"
            for i, obj in enumerate(normalized_info["objects"]):
                if obj["kind"] == "P":
                    assert obj["area"] == 1
                else:
                    assert obj["area"] == 2*int(np.sum(obj["hi"]-obj["lo"]))
                    children = [normalized_info["objects"][j] for j in normalized_info["children"][i]]
                    assert np.array_equal(np.min([c["lo"] for c in children], 0), obj["lo"]+2)
                    assert np.array_equal(np.max([c["hi"] for c in children], 0), obj["hi"]-2)
            assert np.array_equal(solve(inp), out)
            if split == "train":
                feature = _features(inp, out)
                feature["raw_core_changes"] = int(_clear_centers(inp == info["fg"]).sum()) != int(centers.sum())
                training.append(feature)
    assert any(f["interior"] and f["core_sizes"] == (1, 3) for f in training)
    assert any(f["edge"] and f["exterior_points"] for f in training)
    assert any(f["diagonal"] for f in training)
    assert any(f["raw_core_changes"] for f in training)


def _puzzle_signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream exactly 42 diversity seeds; bounded counters, no saved payloads."""
    signatures, shapes, outputs, topologies, depths = set(), set(), set(), set(), set()
    counts, blobs, core_sizes, palettes = set(), set(), set(), set()
    nonrect = edge_pairs = interior_pairs = multi_cores = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp, out)
                shapes.add(inp.shape)
                info = _parse(inp)
                outputs.add(_key(out == info["fg"]))
                topologies.add(f["topology"])
                depths.add(f["depth"])
                counts.add((f["contours"], f["points"]))
                blobs.update(f["blob_shapes"])
                core_sizes.update(f["core_sizes"])
                palettes.add((info["bg"], info["fg"]))
                nonrect += f["nonrect"]
                edge_pairs += f["edge"]
                interior_pairs += f["interior"] > 0
                multi_cores += len(f["core_sizes"]) > 1
        # Same seed regenerates the whole puzzle, with no stored solver state.
        if seed in (0, 19):
            assert _puzzle_signature(generate_puzzle(seed)) == _puzzle_signature(puzzle)
        pair = puzzle["test"][0]
        inp, answer = pair["input"], pair["output"]
        info = _parse(inp)
        # Permute binary roles but retain the publicly fixed highlight literal.
        permutation = np.arange(10)
        available = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9])
        rng = np.random.default_rng(seed+991)
        permutation[available] = rng.permutation(available)
        assert np.array_equal(solve(permutation[inp]), permutation[answer])
        # Translation and safe nonuniform metric stretch preserve the answer.
        h, w = inp.shape
        if h <= 27 and w <= 27:
            shifted = np.pad(inp, ((1, 2), (2, 1)), constant_values=info["bg"])
            assert np.array_equal(solve(shifted), answer)
        if h < 30:
            repeats = np.ones(h, dtype=int)
            repeats[int(rng.integers(h))] = 2
            stretched = np.repeat(inp, repeats, axis=0)
            assert np.array_equal(solve(stretched), answer)
    assert len(signatures) == 42 and len(shapes) >= 60
    assert len(outputs) >= 35 and len(topologies) >= 12
    assert len(counts) >= 12 and len(blobs) >= 6
    assert len(palettes) >= 35 and len(core_sizes) >= 8
    assert nonrect >= 80 and edge_pairs >= 42 and interior_pairs >= 42 and multi_cores >= 42
    assert {1, 2}.issubset(depths)

    # Every supported count combination and malformed count classes.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(903, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    # Explicit independent local-rule regressions, including corner clipping.
    mask = np.zeros((7, 8), bool)
    mask[2, 2] = mask[4, 5] = True
    centers = _clear_centers(mask)
    assert np.array_equal(centers, _literal_centers(mask))
    assert centers[0, 0] and not centers[1, 1]  # diagonal blocks
    full_windows = centers.copy()
    full_windows[[0, -1]] = False
    full_windows[:, [0, -1]] = False
    assert not np.array_equal(centers, full_windows)
    union = np.zeros(mask.shape, bool)
    for r, c in zip(*np.nonzero(centers)):
        union[max(0, r-1):r+2, max(0, c-1):c+2] = True
    assert np.any(union & ~centers & ~mask), "whole-window union must differ"

    # Same enclosing metric frame boxes, different point counts/answers.
    answers = []
    for n in (1, 2):
        raw = np.zeros((24, 24), dtype=np.int8)
        raw[1, 1:23] = raw[22, 1:23] = 1
        raw[1:23, 1] = raw[1:23, 22] = 1
        raw[5, 5:15] = raw[18, 5:15] = 1
        raw[5:19, 5] = raw[5:19, 14] = 1
        raw[8:10, 8:10] = 1
        if n == 2:
            raw[14:16, 8:10] = 1
        raw[8:10, 18:20] = 1
        answers.append(_key(solve(raw)))
    assert answers[0] != answers[1]

    # Corrupt a real core and a preserved fringe: validator must reject each.
    puzzle = generate_puzzle(0, 2, 1)
    for highlight_cell in (True, False):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
        pair = bad["train"][0]
        info = _parse(pair["input"])
        candidates = pair["output"] == (HIGHLIGHT if highlight_cell else info["bg"])
        r, c = np.argwhere(candidates)[0]
        pair["output"][r, c] = info["bg"] if highlight_cell else HIGHLIGHT
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted local decision accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), canonical_layouts=len(outputs),
                containment_topologies=len(topologies), contour_point_counts=len(counts),
                depths=sorted(depths), blob_shapes=sorted(blobs), core_sizes=sorted(core_sizes),
                ordered_palettes=len(palettes), nonrectangular_contours=nonrect,
                edge_pairs=edge_pairs, interior_pairs=interior_pairs,
                multiple_core_pairs=multi_cores,
                independent_solver=False, blind_model_evaluation=False)
