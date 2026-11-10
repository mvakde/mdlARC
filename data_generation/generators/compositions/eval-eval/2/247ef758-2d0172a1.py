"""Scoped Cartesian completion of a distorted hierarchy of local lattices.

Standard library + NumPy only. Construction records are never consulted by
solve: it segments visible components, flood-fills enclosures, and reconstructs
local address grids. Colors have geometric/equality roles, not numeric roles.
"""
from __future__ import annotations

import itertools
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("need 2..10 train, 1..4 test, and train > test")
    return a, b


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized_key(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    result = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo, cells = [(int(y), int(x))], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    todo = [(0, 0)]
    h, w = padded.shape
    while todo:
        r, c = todo.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                todo.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _room(item):
    return isinstance(item, dict)


def _axes(node):
    axes = {}
    for r, row in enumerate(node["cells"]):
        for c, item in enumerate(row):
            if item is not None and not _room(item):
                rs, cs = axes.setdefault(item[1], (set(), set()))
                rs.add(r)
                cs.add(c)
    return axes


def _complete(node, enabled=True):
    cells = [row.copy() for row in node["cells"]]
    claims = {}
    for color, (rs, cs) in _axes(node).items():
        for r, c in itertools.product(rs, cs):
            if _room(cells[r][c]):
                continue
            old = claims.setdefault((r, c), color)
            assert old == color, "conflicting color products"
    # Even the seed-only ablation checks that the original input is legal.
    for (r, c), color in claims.items():
        assert cells[r][c] is None or cells[r][c] == ("B", color)
        if enabled:
            cells[r][c] = ("B", color)
    return cells


def _parse_scene(grid, background=None):
    """Recover actual containment and local addresses from pixels only."""
    grid = _array(grid)
    if background is None:
        background = int(grid[0, 0])
        assert all(np.all(edge == background) for edge in
                   (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "input needs a background perimeter"
    records = []
    labels = np.zeros(grid.shape, int)
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        for coords in _components(grid == color):
            mask = np.zeros(grid.shape, bool)
            mask[coords[:, 0], coords[:, 1]] = True
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            dims = hi-lo+1
            assert np.all(dims % 2 == 1), "odd tight boxes required"
            hole = _holes(mask)
            is_contour = bool(hole.any())
            if is_contour:
                assert len(_components(hole)) == 1, "a contour must enclose exactly one region"
            else:
                assert len(coords) >= .35*int(np.prod(dims)), "blob is not compact"
            record = dict(color=color, coords=coords, mask=mask, hole=hole,
                          lo=lo, hi=hi, center=(lo+hi)//2, contour=is_contour)
            records.append(record)
            labels[mask] = len(records)
    contours = [i for i, o in enumerate(records) if o["contour"]]
    assert contours, "no rooms"
    wall_colors = {records[i]["color"] for i in contours}
    assert len(wall_colors) == 1, "contours must share a wall color"
    wall = next(iter(wall_colors))
    assert all(o["contour"] or o["color"] != wall for o in records)
    # Enforce separation independently of color, including diagonal contacts.
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b)), "touching objects"
    parents = []
    for i, o in enumerate(records):
        candidates = []
        for j in contours:
            if i == j:
                continue
            inside = records[j]["hole"][o["coords"][:, 0], o["coords"][:, 1]]
            assert inside.all() or not inside.any(), "straddling contour"
            if inside.all():
                candidates.append(j)
        parents.append(min(candidates, key=lambda j: int(records[j]["hole"].sum()))
                       if candidates else None)
    children = {i: [] for i in range(len(records))}
    roots = []
    for i, parent in enumerate(parents):
        (roots if parent is None else children[parent]).append(i)
    assert len(roots) == 1 and records[roots[0]]["contour"], "one root room required"
    assert all(bool(children[i]) == o["contour"] for i, o in enumerate(records)), "empty room"

    def build(i):
        o = records[i]
        if not o["contour"]:
            return ("B", o["color"])
        ids = children[i]
        centers = [sorted({int(records[j]["center"][axis]) for j in ids}) for axis in (0, 1)]
        for axis in (0, 1):
            bands = [[records[j] for j in ids if int(records[j]["center"][axis]) == v]
                     for v in centers[axis]]
            for before, after in zip(bands, bands[1:]):
                assert max(int(x["hi"][axis]) for x in before)+1 < min(int(x["lo"][axis]) for x in after), "overlapping local bands"
        cells = [[None for _ in centers[1]] for _ in centers[0]]
        for j in ids:
            r, c = (centers[a].index(int(records[j]["center"][a])) for a in (0, 1))
            assert cells[r][c] is None, "two objects at one address"
            cells[r][c] = build(j)
        node = dict(cells=cells, record=i, centers=centers)
        _complete(node)  # Color products must be unambiguous even before drawing.
        return node

    root = build(roots[0])
    return dict(root=root, records=records, children=children, parents=parents,
                background=background, wall=wall)


def _pack(tiles, background, row_slack=None, col_slack=None, row_gaps=None, col_gaps=None):
    nr, nc = len(tiles), len(tiles[0])
    heights = [max([1]+[t.shape[0] for t in row if t is not None]) for row in tiles]
    widths = [max([1]+[tiles[r][c].shape[1] for r in range(nr) if tiles[r][c] is not None])
              for c in range(nc)]
    heights = [v+(row_slack or [0]*nr)[i] for i, v in enumerate(heights)]
    widths = [v+(col_slack or [0]*nc)[i] for i, v in enumerate(widths)]
    rg, cg = row_gaps or [1]*(nr-1), col_gaps or [1]*(nc-1)
    ys, xs = [0], [0]
    for h, gap in zip(heights[:-1], rg):
        ys.append(ys[-1]+h+gap)
    for w, gap in zip(widths[:-1], cg):
        xs.append(xs[-1]+w+gap)
    out = np.full((ys[-1]+heights[-1], xs[-1]+widths[-1]), background, np.int8)
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            if tile is None:
                continue
            h, w = tile.shape
            assert h % 2 and w % 2 and heights[r] % 2 and widths[c] % 2
            y, x = ys[r]+(heights[r]-h)//2, xs[c]+(widths[c]-w)//2
            out[y:y+h, x:x+w] = tile
    return out


def _canonical(node, background, wall, complete=True):
    cells = _complete(node, enabled=complete)
    tiles = [[None if t is None else _canonical(t, background, wall, complete)
              if _room(t) else np.array([[t[1]]], np.int8) for t in row] for row in cells]
    content = _pack(tiles, background)
    out = np.pad(content, 2, constant_values=background)
    out[0, :] = out[-1, :] = wall
    out[:, 0] = out[:, -1] = wall
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete compact diagram entirely from one visible input."""
    info = _parse_scene(input_grid)
    return _canonical(info["root"], info["background"], info["wall"])


def _nodes(node):
    yield node
    for row in node["cells"]:
        for item in row:
            if _room(item):
                yield from _nodes(item)


def _depth(node):
    return 1+max([0]+[_depth(t) for row in node["cells"] for t in row if _room(t)])


def _unscoped(info):
    """Diagnostic shortcut: Cartesian completion globally, ignoring rooms."""
    base = _canonical(info["root"], info["background"], info["wall"], complete=False)
    out = base.copy()
    colors = [int(c) for c in np.unique(base) if c not in (info["background"], info["wall"])]
    colors.sort(key=lambda c: tuple(np.argwhere(base == c)[0]))
    for color in colors:
        points = np.argwhere(base == color)
        for r, c in itertools.product(set(points[:, 0]), set(points[:, 1])):
            if out[r, c] == info["background"]:
                out[r, c] = color
    return out


def _features(info):
    added, blocked, products = 0, 0, set()
    patterns = set()
    for node in _nodes(info["root"]):
        cells = node["cells"]
        for color, (rs, cs) in _axes(node).items():
            products.add((len(rs), len(cs)))
            for r, c in itertools.product(rs, cs):
                blocked += _room(cells[r][c])
                added += cells[r][c] is None
        mapping = {}
        pattern = tuple(tuple("C" if _room(t) else "." if t is None
                              else str(mapping.setdefault(t[1], len(mapping))) for t in row)
                        for row in cells)
        patterns.add(pattern)
    out = _canonical(info["root"], info["background"], info["wall"])
    nonrect = 0
    for o in info["records"]:
        if o["contour"]:
            y, x = o["lo"]; yy, xx = o["hi"]
            local = o["mask"][y:yy+1, x:xx+1]
            nonrect += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
    return dict(added=int(added), blocked=int(blocked), products=products,
                rooms=sum(o["contour"] for o in info["records"]), depth=_depth(info["root"]),
                nonrect=nonrect, patterns=patterns,
                scope_difference=not np.array_equal(out, _unscoped(info)))


def _transform(node, rng):
    a = np.empty((len(node["cells"]), len(node["cells"][0])), object)
    for r, row in enumerate(node["cells"]):
        for c, t in enumerate(row):
            a[r, c] = _transform(t, rng) if _room(t) else t
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = a[:, ::-1]
    return dict(cells=a.tolist())


def _witness(rng, nested=False):
    p, q = ("B", 2), ("B", 3)
    inner = dict(cells=[[p, p, None], [p, None, q]])
    if not nested:
        return _transform(inner, rng)
    outer = dict(cells=[[p, p, p], [p, inner, None], [p, None, None]])
    if rng.random() < .35:
        # An independent-color singleton gives a fourth visible band.
        for r, row in enumerate(outer["cells"]):
            row.append(q if r == int(rng.integers(3)) else None)
        # Ensure the added column is actually represented.
        if all(row[-1] is None for row in outer["cells"]):
            outer["cells"][int(rng.integers(3))][-1] = q
    return _transform(outer, rng)


def _prune(cells):
    rows = [row for row in cells if any(t is not None for t in row)]
    cols = [c for c in range(len(rows[0])) if any(row[c] is not None for row in rows)]
    return [[row[c] for c in cols] for row in rows]


def _sample_node(rng, ncolors, budget, depth=1):
    nr, nc = (int(rng.choice([1, 2, 3, 4], p=[.12, .39, .35, .14])) for _ in range(2))
    cells = [[None for _ in range(nc)] for _ in range(nr)]
    if budget[0] > 0 and depth < 3 and rng.random() < .75:
        number = min(budget[0], int(rng.choice([1, 2], p=[.8, .2])), nr*nc)
        for pos in rng.choice(nr*nc, number, replace=False):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            r, c = divmod(int(pos), nc)
            cells[r][c] = _sample_node(rng, ncolors, budget, depth+1)
    reserved = {}
    for color in rng.permutation(np.arange(2, 2+ncolors)):
        color = int(color)
        for _ in range(24):
            rcount = int(rng.integers(1, min(3, nr)+1))
            ccount = int(rng.integers(1, min(3, nc)+1))
            rs = list(map(int, rng.choice(nr, rcount, replace=False)))
            cs = list(map(int, rng.choice(nc, ccount, replace=False)))
            available = {(r, c) for r, c in itertools.product(rs, cs) if not _room(cells[r][c])}
            if not available or any(pos in reserved for pos in available):
                continue
            # Start with the available product, then erase seeds while retaining
            # every represented row/column. The omitted cells are real queries.
            seeds = set(available)
            positions = sorted(seeds)
            rng.shuffle(positions)
            for pos in positions:
                r, c = pos
                if (rng.random() < .72 and sum(y == r for y, x in seeds) > 1
                        and sum(x == c for y, x in seeds) > 1):
                    seeds.remove(pos)
            actual_rs, actual_cs = {r for r, c in seeds}, {c for r, c in seeds}
            for pos in itertools.product(actual_rs, actual_cs):
                if not _room(cells[pos[0]][pos[1]]):
                    reserved[pos] = color
            for r, c in seeds:
                cells[r][c] = ("B", color)
            break
    if not any(t is not None for row in cells for t in row):
        cells[int(rng.integers(nr))][int(rng.integers(nc))] = ("B", 2)
    return dict(cells=_prune(cells))


def _test_tree(rng, ncolors, mode):
    p = ("B", 2)
    if mode == 1:  # New third-level combination, with visible outer siblings.
        inner = _witness(rng)
        middle = dict(cells=[[inner, p]])
        outer = dict(cells=[[p], [middle]])
        return _transform(outer, rng)
    if mode == 2:  # Separate child scopes in a genuinely two-dimensional room.
        a, b = _witness(rng), _witness(rng)
        return _transform(dict(cells=[[a, p], [p, b]]), rng)
    node = _sample_node(rng, ncolors, [int(rng.integers(1, 4))])
    if _depth(node) == 1:
        # Insert a child into a vacant address, or add a separate occupied band.
        empty = [(r, c) for r, row in enumerate(node["cells"]) for c, t in enumerate(row) if t is None]
        if empty:
            r, c = empty[int(rng.integers(len(empty)))]
            node["cells"][r][c] = _witness(rng)
        else:
            width = len(node["cells"][0])
            row = [None]*width
            row[int(rng.integers(width))] = _witness(rng)
            node["cells"].append(row)
    return node


def _blob(rng, compact):
    probs = [.87, .13] if compact else [.58, .42]
    h, w = (int(rng.choice([1, 3], p=probs)) for _ in range(2))
    mask = np.ones((h, w), bool)
    if h == w == 3:
        positions = list(map(tuple, np.argwhere(mask)))
        rng.shuffle(positions)
        for r, c in positions:
            if rng.random() >= .48 or mask.sum() <= 5:
                continue
            candidate = mask.copy()
            candidate[r, c] = False
            if (candidate.any(axis=0).all() and candidate.any(axis=1).all()
                    and len(_components(candidate)) == 1 and not _holes(candidate).any()):
                mask = candidate
    return mask


def _outline(rng, h, w, py, px):
    """Orthogonal loop in four safe corridors; actual holes are audited."""
    k = 1 if min(py, px) >= 3 and rng.random() < .6 else 0

    def profile(length, padding):
        a = np.zeros(length, int)
        if padding >= 3 and length > 2*k+5 and rng.random() < .8:
            low = int(rng.integers(k+2, length-k-3))
            high = int(rng.integers(low+1, length-k-1))
            a[low:high] = int(rng.integers(1, padding-1))
        return a

    top, bottom = profile(w, py), profile(w, py)
    left, right = profile(h, px), profile(h, px)
    path = [(int(top[x]), x) for x in range(k, w-k)]
    path += [(k, w-1-k), (k, w-1)]
    path += [(y, w-1-int(right[y])) for y in range(k, h-k)]
    path += [(h-1-k, w-1-k), (h-1, w-1-k)]
    path += [(h-1-int(bottom[x]), x) for x in range(w-1-k, k-1, -1)]
    path += [(h-1-k, k), (h-1-k, 0)]
    path += [(y, int(left[y])) for y in range(h-1-k, k-1, -1)]
    path += [(k, k), (0, k), path[0]]
    mask = np.zeros((h, w), bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _draw(node, rng, compact=False):
    tiles = []
    for row in node["cells"]:
        tile_row = []
        for t in row:
            if t is None:
                tile_row.append(None)
            elif _room(t):
                tile_row.append(_draw(t, rng, compact))
            else:
                tile_row.append(np.where(_blob(rng, compact), t[1], 0).astype(np.int8))
        tiles.append(tile_row)
    nr, nc = len(tiles), len(tiles[0])
    slack = lambda n: [0]*n if compact else [2*int(rng.integers(2)) for _ in range(n)]
    gaps = lambda n: [1]*n if compact else list(map(int, rng.choice([1, 3, 5], n, p=[.65, .3, .05])))
    content = _pack(tiles, 0, slack(nr), slack(nc), gaps(nr-1), gaps(nc-1))
    py, px = (2, 2) if compact else tuple(map(int, rng.choice([2, 3, 4], 2, p=[.38, .47, .15])))
    h, w = content.shape
    out = np.pad(content, ((py, py), (px, px)))
    walls = _outline(rng, h+2*py, w+2*px, py, px)
    assert not np.any((out != 0) & walls)
    out[walls] = 1
    return out


def _recolor_node(node, palette):
    return dict(cells=[[None if t is None else _recolor_node(t, palette) if _room(t)
                       else ("B", int(palette[t[1]])) for t in row] for row in node["cells"]])


def _make_pair(rng, kind, ncolors, palette, seen, forbidden_outputs=(), test_mode=0):
    for attempt in range(100):
        if kind == "basic":
            node = _witness(rng)
        elif kind == "scope":
            node = _witness(rng, nested=True)
        elif kind == "test":
            # Broaden after early attempts rather than returning a fixed fixture.
            mode = test_mode if attempt < 16 else 0
            node = _test_tree(rng, ncolors, mode)
        else:
            node = _sample_node(rng, ncolors, [int(rng.integers(0, 3))])
        raw = _draw(node, rng, compact=attempt >= 6)
        if max(raw.shape) > 28:
            continue
        available = [30-raw.shape[0], 30-raw.shape[1]]
        margins = [(int(rng.integers(1, min(3, space-1)+1)), 1) for space in available]
        # Add optional trailing slack without ever exceeding the 30-cell bound.
        margins = [(a, int(rng.integers(1, min(3, space-a)+1)))
                   for (a, b), space in zip(margins, available)]
        raw = np.pad(raw, margins)
        inp = palette[raw].astype(np.int8)
        expected = _canonical(_recolor_node(node, palette), int(palette[0]), int(palette[1]))
        if max(expected.shape) > 30 or _key(inp) in seen:
            continue
        if _normalized_key(expected) in forbidden_outputs:
            continue
        try:
            info = _parse_scene(inp)
            assert np.array_equal(expected, _canonical(info["root"], info["background"], info["wall"]))
            f = _features(info)
            if kind == "basic":
                assert f["added"] and any(a > 1 and b > 1 for a, b in f["products"])
            elif kind == "scope":
                assert f["added"] and f["blocked"] and f["scope_difference"] and f["depth"] >= 2
            elif kind == "test":
                assert f["added"] and f["rooms"] >= 2
        except AssertionError:
            continue
        return dict(input=inp, output=expected)
    raise RuntimeError("bounded construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    seed = _integer(seed, "seed") % (1 << 64)
    rng = np.random.default_rng(seed)
    ncolors = int(rng.integers(2, 5))
    palette = rng.permutation(10)
    vary_palette = bool(rng.integers(2))
    test_mode = int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            roles = rng.permutation(10) if vary_palette else palette
            kind = ("basic" if i == 0 else "scope" if i == 1 else "random") if split == "train" else "test"
            forbidden = {_normalized_key(p["output"]) for p in puzzle["train"]} if split == "test" else set()
            pair = _make_pair(rng, kind, ncolors, roles, seen, forbidden, test_mode)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_local_output(inp_node, out_node, out_info):
    expected = _complete(inp_node)
    actual = out_node["cells"]
    assert (len(actual), len(actual[0])) == (len(expected), len(expected[0])), "lost local bands"
    for r, row in enumerate(expected):
        for c, item in enumerate(row):
            other = actual[r][c]
            if _room(item):
                assert _room(other), "child room erased or replaced"
                _check_local_output(item, other, out_info)
            else:
                assert other == item, "wrong local Cartesian closure"
    i = out_node["record"]
    parent = out_info["records"][i]
    h, w = parent["hi"]-parent["lo"]+1
    assert len(parent["coords"]) == 2*(int(h)+int(w))-4, "nonrectangular output frame"
    y, x = parent["lo"]; yy, xx = parent["hi"]
    local = parent["mask"][y:yy+1, x:xx+1]
    assert local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all()
    children = [out_info["records"][j] for j in out_info["children"][i]]
    assert np.array_equal(np.min([o["lo"] for o in children], axis=0), parent["lo"]+2), "wrong inner moat"
    assert np.array_equal(np.max([o["hi"] for o in children], axis=0), parent["hi"]-2), "wrong inner moat"
    for axis in (0, 1):
        centers = out_node["centers"][axis]
        bands = [[o for o in children if int(o["center"][axis]) == v] for v in centers]
        for first, second in zip(bands, bands[1:]):
            assert min(int(o["lo"][axis]) for o in second)-max(int(o["hi"][axis]) for o in first) == 2, "wrong band gap"


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    train_outputs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse_scene(inp)
            expected = _canonical(info["root"], info["background"], info["wall"])
            assert np.array_equal(expected, out), "wrong solved output"
            output_info = _parse_scene(out, info["background"])
            assert output_info["wall"] == info["wall"]
            assert all(o["contour"] or len(o["coords"]) == 1 for o in output_info["records"]), "nonpoint output blob"
            assert sum(o["contour"] for o in info["records"]) == sum(o["contour"] for o in output_info["records"])
            _check_local_output(info["root"], output_info["root"], output_info)
            assert np.array_equal(out, _canonical(output_info["root"], info["background"], info["wall"])), "closure not idempotent"
            f = _features(info)
            features[split].append(f)
            if split == "train":
                train_outputs.add(_normalized_key(out))
            else:
                assert f["rooms"] >= 2 and f["added"] > 0, "missing test recombination"
                assert _normalized_key(out) not in train_outputs, "test layout already demonstrated"
    training = features["train"]
    assert any(f["added"] and any(a > 1 and b > 1 for a, b in f["products"]) for f in training), "no Cartesian witness"
    assert any(f["blocked"] and f["scope_difference"] and f["depth"] >= 2 for f in training), "no scope/blocking witness"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded invariants, explicit ablations, and structural variation: 42 seeds."""
    shapes, outputs, patterns, palettes, depths, room_counts, products = (set() for _ in range(7))
    nonrect, signatures, total_pairs = 0, set(), 0
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        signatures.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _parse_scene(pair["input"])
                f = _features(info)
                shapes.add(pair["input"].shape)
                outputs.add(_normalized_key(pair["output"]))
                patterns.update(f["patterns"])
                palettes.add((info["background"], info["wall"]))
                depths.add(f["depth"]); room_counts.add(f["rooms"])
                products.update(f["products"])
                nonrect += f["nonrect"]
                total_pairs += 1
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(signatures) == 42 and total_pairs == 210
    assert len(shapes) >= 35 and len(outputs) >= 60 and len(patterns) >= 65
    assert len(palettes) >= 25 and {1, 2, 3} <= depths and {1, 2, 3} <= room_counts
    assert {(1, 1), (2, 1), (1, 2), (2, 2), (2, 3), (3, 2), (3, 3)} <= products
    assert nonrect >= 30
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    # The host verifier exhausts all legal combinations. Keep this self-test
    # bounded by exercising the count boundaries and a nondefault middle case.
    for a, b in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(8100+a*10+b, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit local product regression: L + a separate color.
    p, q = ("B", 2), ("B", 3)
    node = dict(cells=[[p, p, None], [p, None, q]])
    expected_cells = [[p, p, None], [p, p, q]]
    assert _complete(node) == expected_cells
    # Metric redraws must preserve the answer, independently of parser metadata.
    rng = np.random.default_rng(90125)
    realizations = set()
    for _ in range(12):
        raw = _draw(node, rng)
        if max(raw.shape) <= 28:
            inp = np.pad(raw, 1)
            prediction = solve(inp)
            assert np.array_equal(prediction, _canonical(node, 0, 1))
            realizations.add(_key(inp))
    assert len(realizations) >= 5
    # Full color permutations include zero in both wall and marker roles.
    for puzzle in puzzles[:8]:
        pair = puzzle["test"][0]
        perm = rng.permutation(10)
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    # Reject answers embodying omission of either inherited mechanism.
    base = puzzles[0]
    candidates = []
    for index, pair in enumerate(base["train"]):
        info = _parse_scene(pair["input"])
        seed_only = _canonical(info["root"], info["background"], info["wall"], False)
        if not np.array_equal(seed_only, pair["output"]):
            candidates.append((index, seed_only))
        if _features(info)["scope_difference"]:
            candidates.append((index, _unscoped(info)))
    for index, wrong in candidates:
        assert not np.array_equal(wrong, base["train"][index]["output"])
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        bad["train"][index]["output"] = wrong
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("ablation answer accepted")
    return dict(seeds=42, pairs=total_pairs, input_shapes=len(shapes),
                canonical_structures=len(outputs), local_patterns=len(patterns),
                role_palettes=len(palettes), depths=sorted(depths), rooms=sorted(room_counts),
                products=sorted(products), nonrectangular_contours=nonrect)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
