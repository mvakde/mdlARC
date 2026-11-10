"""Keyed chambers: nearest enclosing color scope, followed by compact docking.

Only standard-library modules and NumPy are used. No parent module or stored
example is read. Generation labels use an abstract tree; solve rebuilds that
tree and the routing decisions from the visible pixels.
"""
from __future__ import annotations

import copy
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    unseen = set(zip(*np.nonzero(mask)))
    result = []
    neighbors = EIGHT if diagonal else FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((int(y), int(x)))
            for dy, dx in neighbors:
                p = (y + dy, x + dx)
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _hole(mask):
    # Flood through everything except this one object's own pixels. Descendant
    # loops and cargo must not obstruct the enclosure test.
    padded = np.pad(mask, 1)
    exterior = np.zeros(padded.shape, dtype=bool)
    exterior[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in FOUR:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not exterior[yy, xx]:
                exterior[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~exterior)[1:-1, 1:-1]


def _objects(grid, background):
    result = []
    for cells in _components(grid != background):
        mask = np.zeros(grid.shape, bool)
        mask[cells[:, 0], cells[:, 1]] = True
        hole = _hole(mask)
        _require(not hole.any() or len(_components(hole)) == 1, "an outline must enclose exactly one region")
        result.append(dict(cells=cells, mask=mask, hole=hole,
                           lo=cells.min(axis=0), hi=cells.max(axis=0)))
    return result


def _containing(cells, loops):
    enclosing = []
    for i, loop in enumerate(loops):
        inside = loop["hole"][cells[:, 0], cells[:, 1]]
        _require(inside.all() or not inside.any(), "object crosses a chamber boundary")
        if inside.all():
            enclosing.append(i)
    return sorted(enclosing, key=lambda i: int(loops[i]["hole"].sum()))


def _parse_canonical(grid, background):
    objects = _objects(grid, background)
    loops = [o for o in objects if o["hole"].any()]
    _require(bool(loops), "no closed chamber")
    for loop in loops:
        cells = loop["cells"]
        colors, counts = np.unique(grid[cells[:, 0], cells[:, 1]], return_counts=True)
        _require(len(colors) == 2 and counts[0] != counts[1], "outline needs a majority wall and one key")
        wall = int(colors[np.argmax(counts)])
        key = int(colors[np.argmin(counts)])
        gate = cells[grid[cells[:, 0], cells[:, 1]] == key]
        y = int(loop["lo"][0])
        _require(np.all(gate[:, 0] == y), "keys must occupy the common near edge")
        cols = sorted(gate[:, 1].tolist())
        _require(cols == list(range(cols[0], cols[-1] + 1)), "key must be one uninterrupted segment")
        _require(loop["lo"][1] < cols[0] <= cols[-1] < loop["hi"][1], "key cannot occupy a corner")
        _require(grid[y, cols[0] - 1] == wall and grid[y, cols[-1] + 1] == wall,
                 "key must be flanked by its wall")
        loop.update(wall=wall, key=key, gate=gate)
        near = np.zeros(grid.shape, dtype=bool)
        padded = np.pad(loop["mask"], 1)
        h, w = grid.shape
        for dy, dx in EIGHT:
            near |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        _require(not np.any(near & (grid != background) & ~loop["mask"]),
                 "chambers need background clearance from other objects")

    parents = []
    children = [[] for _ in loops]
    for i, loop in enumerate(loops):
        enclosing = _containing(loop["cells"], loops)
        parent = enclosing[0] if enclosing else None
        parents.append(parent)
        if parent is not None:
            children[parent].append(i)
    roots = [i for i, p in enumerate(parents) if p is None]
    _require(len(roots) == 1, "family has one outer chamber")
    for ids in children:
        ids.sort(key=lambda i: int(loops[i]["lo"][1]))
        for a, b in zip(ids, ids[1:]):
            _require(loops[a]["hi"][1] + 1 < loops[b]["lo"][1],
                     "sibling chambers need separate parallel-coordinate slabs")

    cargo = []
    for group in objects:
        if group["hole"].any():
            continue
        # A touching two-color block is two cargo units, rather than one blob.
        for color in np.unique(grid[group["cells"][:, 0], group["cells"][:, 1]]):
            mask = group["mask"] & (grid == color)
            parts = _components(mask)
            _require(len(parts) == len(_components(mask, True)), "diagonal-only same-color contact is excluded")
            for cells in parts:
                local = np.zeros(grid.shape, bool)
                local[cells[:, 0], cells[:, 1]] = True
                _require(not _hole(local).any(), "cargo cannot itself be hollow")
                lo, hi = cells.min(axis=0), cells.max(axis=0)
                _require(len(cells) >= .35 * int(np.prod(hi - lo + 1)), "cargo must be compact")
                enclosing = _containing(cells, loops)
                _require(bool(enclosing), "cargo must start inside the outer chamber")
                matches = [i for i in enclosing if loops[i]["key"] == int(color)]
                _require(bool(matches), "cargo has no matching enclosing key")
                cargo.append(dict(cells=cells, color=int(color), source=enclosing[0],
                                  destination=matches[0], enclosing=enclosing))
    # Check contacts across separate non-background groups too.
    for color in {c["color"] for c in cargo}:
        mask = np.zeros(grid.shape, dtype=bool)
        same = [c for c in cargo if c["color"] == color]
        for c in same:
            mask[c["cells"][:, 0], c["cells"][:, 1]] = True
        _require(len(_components(mask, True)) == len(same), "diagonal-only same-color contact is excluded")
    counts = [0] * len(loops)
    for c in cargo:
        counts[c["destination"]] += 1
    depth = []
    for i in range(len(loops)):
        d, j = 1, parents[i]
        while j is not None:
            d, j = d + 1, parents[j]
        depth.append(d)
    return dict(grid=grid, background=background, loops=loops, cargo=cargo,
                parents=parents, children=children, root=roots[0], counts=counts, depth=depth)


def _analyze(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg) and
             np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg), "outer border must reveal background")
    candidates = []
    for turn in range(4):
        try:
            a = _parse_canonical(np.rot90(grid, turn).copy(), bg)
            a["turn"] = turn
            candidates.append(a)
        except ValueError:
            pass
    _require(len(candidates) == 1, "chamber keys must identify one common direction")
    return candidates[0]


def _dock_tile(child_tiles, count, wall, background):
    """Minimal chamber: bottom-aligned child row, then an optional cargo shelf."""
    if not child_tiles and count == 0:
        result = np.full((3, 3), wall, dtype=np.int8)
        result[1, 1] = background
        return result
    ch = max((p.shape[0] for p in child_tiles), default=0)
    cw = sum(p.shape[1] for p in child_tiles) + max(0, len(child_tiles) - 1)
    pw = max(0, 2 * count - 1)
    width = max(cw, pw)
    height = ch + (2 if child_tiles and count else 1 if count else 0)
    result = np.full((height + 4, width + 4), background, dtype=np.int8)
    result[0, :] = result[-1, :] = wall
    result[:, 0] = result[:, -1] = wall
    x = 2 + (width - cw) // 2
    for p in child_tiles:
        y = 2 + ch - p.shape[0]
        result[y:y + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 1
    if count:
        x = 2 + (width - pw) // 2
        result[height + 1, x:x + pw:2] = wall
    return result


def _render_analysis(a, mode="rule"):
    counts = [0] * len(a["loops"])
    for c in a["cargo"]:
        dest = c["destination"]
        if mode == "ignore_routing":
            dest = c["source"]
        elif mode == "ignore_scope":
            # Choose the outermost matching chamber instead of local scope.
            dest = next(i for i in reversed(c["enclosing"]) if a["loops"][i]["key"] == c["color"])
        counts[dest] += len(c["cells"]) if mode == "count_pixels" else 1

    def visit(i):
        return _dock_tile([visit(j) for j in a["children"][i]], counts[i],
                          a["loops"][i]["wall"], a["background"])
    return np.rot90(visit(a["root"]), -a["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive enclosure, key direction, routing, and compact output from input."""
    return _grid(_render_analysis(_analyze(input_grid)))


def _node(key, wall):
    return dict(key=int(key), wall=int(wall), children=[], cargo=[])


def _blob(rng, compact=False):
    h, w = (int(rng.integers(1, 3 if compact else 4)) for _ in range(2))
    mask = np.ones((h, w), bool)
    # Remove corners, creating blocks, Ls and other compact hole-free shapes.
    for y, x in rng.permutation([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]):
        if mask.sum() > 1 and rng.random() < .35:
            trial = mask.copy()
            trial[y, x] = False
            if len(_components(trial)) == 1 and not _hole(trial).any():
                mask = trial
    yy, xx = np.nonzero(mask)
    return mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]


def _add_cargo(node, colors, rng, compact=False, touching=False):
    # A group may contain several touching monochromatic cargo units. Units of
    # the same color use separate groups to avoid merging them in the input.
    if touching:
        h = int(rng.integers(1, 3))
        group = [(int(color), np.ones((h, int(rng.integers(1, 3))), bool)) for color in colors]
        if all(mask.sum() == 1 for _, mask in group):
            # Guarantee a visible area-versus-unit-count lesson at 2/1 counts.
            group[0] = (group[0][0], np.ones((h, 2), bool))
    else:
        group = [(int(color), _blob(rng, compact)) for color in colors]
    node["cargo"].append(group)


def _tree(rng, lesson, compact):
    palette = list(map(int, rng.permutation(10)))
    background = palette[0]
    keys = palette[1:4]
    # Walls may repeat and may reuse another chamber's key, as in parent B.
    def wall(key):
        return int(rng.choice([c for c in palette[1:] if c != key]))

    a, b = keys[:2]
    root = _node(a, wall(a))
    if lesson == "routing":
        child = _node(b, wall(b))
        root["children"] = [child]
        _add_cargo(child, [a, b] if rng.random() < .5 else [b, a], rng, compact, True)
        for _ in range(int(rng.integers(1, 3))):
            _add_cargo(child if rng.random() < .5 else root, [a], rng, compact)
        _add_cargo(root, [a], rng, compact)
    elif lesson == "scope":
        near = _node(a, wall(a))
        empty = _node(b, wall(b))
        if empty["wall"] == near["wall"]:
            empty["wall"] = int(rng.choice([c for c in palette[1:] if c not in (b, near["wall"])]))
        root["children"] = [near, empty] if rng.random() < .5 else [empty, near]
        for _ in range(int(rng.integers(1, 4))):
            _add_cargo(near, [a], rng, compact)
        _add_cargo(root, [a], rng, compact)
    elif lesson == "test":
        mid_key = int(rng.choice([a, b]))
        deep_key = int(rng.choice(keys))
        middle = _node(mid_key, wall(mid_key))
        deep = _node(deep_key, wall(deep_key))
        root["children"], middle["children"] = [middle], [deep]
        deep_routes = list(dict.fromkeys([a, mid_key, deep_key]))
        if len(deep_routes) > 1 and rng.random() < .6:
            colors = list(map(int, rng.choice(deep_routes, 2, replace=False)))
            _add_cargo(deep, colors, rng, compact, True)
        else:
            _add_cargo(deep, [int(rng.choice(deep_routes))], rng, compact)
        for node, routes in ((root, [a]), (middle, list(dict.fromkeys([a, mid_key]))),
                             (deep, deep_routes)):
            for _ in range(int(rng.integers(0, 3))):
                _add_cargo(node, [int(rng.choice(routes))], rng, compact)
        if sum(len(group) for node in (root, middle, deep) for group in node["cargo"]) < 2:
            _add_cargo(root, [a], rng, compact)
        # Branching, keys, source compartments and assignments are independent
        # choices. No fixed test gate pattern or receiver-count recipe remains.
        if rng.random() < .5:
            sibling = _node(int(rng.choice(keys)), wall(a))
            if sibling["key"] == sibling["wall"]:
                sibling["wall"] = wall(sibling["key"])
            if rng.random() < .65:
                for _ in range(int(rng.integers(1, 3))):
                    _add_cargo(sibling, [int(rng.choice([a, sibling["key"]]))], rng, compact)
            where = root if rng.random() < .6 else middle
            where["children"].insert(int(rng.integers(2)), sibling)
    else:
        # Additional demonstrations include simple chambers and varied sibling
        # trees, rather than a fixed sequence of normalized output templates.
        n = int(rng.choice([0, 0, 1, 2, 3]))
        for _ in range(n):
            key = int(rng.choice(keys))
            child = _node(key, wall(key))
            root["children"].append(child)
            for _ in range(int(rng.integers(0, 4))):
                color = int(rng.choice([a, key]))
                _add_cargo(child, [color], rng, compact)
        for _ in range(int(rng.integers(1, 5))):
            _add_cargo(root, [a], rng, compact)
    return root, background


def _abstract_answer(root, background):
    counts = {}

    def walk(node, path):
        counts[id(node)] = 0
        path = path + [node]
        for group in node["cargo"]:
            for color, mask in group:
                receiver = next(n for n in reversed(path) if n["key"] == color)
                counts[id(receiver)] += 1
        for child in node["children"]:
            walk(child, path)
    walk(root, [])

    def draw(node):
        return _dock_tile([draw(n) for n in node["children"]], counts[id(node)], node["wall"], background)
    return draw(root)


def _pack(tiles, background, axis, rng, compact, align="random"):
    if len(tiles) == 1:
        return tiles[0].copy()
    gap = 1 if compact else int(rng.integers(1, 4))
    cross = 1 - axis
    cross_size = max(t.shape[cross] for t in tiles)
    shape = [0, 0]
    shape[cross] = cross_size
    shape[axis] = sum(t.shape[axis] for t in tiles) + gap * (len(tiles) - 1)
    result = np.full(shape, background, dtype=np.int8)
    cursor = 0
    for tile in tiles:
        delta = cross_size - tile.shape[cross]
        offset = delta if align == "bottom" else int(rng.integers(delta + 1))
        y, x = (cursor, offset) if axis == 0 else (offset, cursor)
        result[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        cursor += tile.shape[axis] + gap
    return result


def _metric_chamber(node, bg, rng, compact):
    fields = []
    if node["children"]:
        children = [_metric_chamber(n, bg, rng, compact) for n in node["children"]]
        fields.append(_pack(children, bg, 1, rng, compact))
    groups = []
    for group in node["cargo"]:
        pieces = [np.where(mask, color, bg).astype(np.int8) for color, mask in group]
        # Touching groups are equal-height full rectangles; join with zero gap.
        groups.append(np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0])
    if groups:
        # Resample source shelf grouping; neither cargo order nor row is an
        # output address. Membership and monochromatic connectivity are.
        if compact:
            rows = [groups]
        else:
            cut = int(rng.integers(1, len(groups) + 1))
            rows = [groups[:cut]] + ([groups[cut:]] if cut < len(groups) else [])
        rows = [_pack(row, bg, 1, rng, compact) for row in rows]
        fields.append(_pack(rows, bg, 0, rng, compact))
    if fields:
        if len(fields) == 2 and rng.random() < .5:
            fields.reverse()
        content = _pack(fields, bg, 0, rng, compact)
    else:
        content = np.full((1, 1), bg, dtype=np.int8)
    padding = [2] * 4 if compact else list(map(int, rng.integers(2, 5, size=4)))
    top, bottom, left, right = padding
    h, w = content.shape[0] + top + bottom, content.shape[1] + left + right
    silhouette = np.zeros((h, w), bool)
    # Independent stepped side profiles, including bulges and inward bays.
    for y in range(h):
        l = int(rng.integers(left - 1)) if not compact else 0
        r = w - 1 - (int(rng.integers(right - 1)) if not compact else 0)
        silhouette[y, l:r + 1] = True
    padded = np.pad(silhouette, 1)
    # Eight-neighbor erosion supplies orthogonal connectors at stepped corners.
    # A four-neighbor shell alone can leave diagonal-only boundary fragments.
    interior = silhouette.copy()
    for dy, dx in EIGHT:
        interior &= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    outline = silhouette & ~interior
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[outline] = node["wall"]
    grid[top:top + content.shape[0], left:left + content.shape[1]] = content
    cols = np.flatnonzero(outline[0])
    gw = 1 if compact else int(rng.integers(1, min(3, len(cols) - 2) + 1))
    start = int(rng.integers(int(cols[0]) + 1, int(cols[-1]) - gw + 1))
    grid[0, start:start + gw] = node["key"]
    return grid


def _make_pair(rng, lesson, turn):
    for attempt in range(100):
        compact = attempt >= 18
        root, bg = _tree(rng, lesson, compact)
        inp = _metric_chamber(root, bg, rng, compact)
        expected = _abstract_answer(root, bg)
        if max(*inp.shape, *expected.shape) > 28:
            continue
        spare = [30 - d for d in inp.shape]
        margins = []
        for room in spare:
            before = int(rng.integers(1, min(4, room)))
            after = int(rng.integers(1, min(4, room - before) + 1))
            margins.append((before, after))
        inp = np.pad(inp, margins, constant_values=bg)
        pair = {"input": np.rot90(inp, turn).copy(), "output": np.rot90(expected, turn).copy()}
        try:
            a = _analyze(pair["input"])
            _require(np.array_equal(_render_analysis(a), pair["output"]), "construction/parser mismatch")
            if lesson == "routing":
                _require(_features(a)["split_routes"] and _features(a)["skip"], "routing lesson absent")
            if lesson == "scope":
                _require(_features(a)["scoped"] and _features(a)["empty_leaf"], "scope lesson absent")
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded chamber construction exhausted")


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train, test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    turns = list(map(int, rng.permutation(4)))
    mixed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", train), ("test", test)):
        for i in range(count):
            lesson = ("routing" if i == 0 else "scope" if i == 1 else None) if split == "train" else "test"
            # A mixed puzzle demonstrates two gate directions before rotating
            # tests into the remaining directions. Fixed puzzles stay fixed.
            turn = turns[i % 2 if split == "train" else 2 + i % 2] if mixed else turns[0]
            for _ in range(32):
                pair = _make_pair(rng, lesson, turn)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(a):
    grid, cargo, loops = a["grid"], a["cargo"], a["loops"]
    split_routes = False
    for i, x in enumerate(cargo):
        if split_routes:
            break
        cells = {tuple(p) for p in x["cells"]}
        for y in cargo[i + 1:]:
            if x["color"] != y["color"] and x["destination"] != y["destination"]:
                split_routes |= any((r + dy, c + dx) in cells for r, c in y["cells"] for dy, dx in FOUR)
    nonrect = 0
    for o in loops:
        y, x = o["lo"]
        yy, xx = o["hi"]
        m = o["mask"][y:yy + 1, x:xx + 1]
        nonrect += not (m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all())
    return dict(depth=max(a["depth"]), chambers=len(loops), cargo=len(cargo),
                skip=any(c["source"] != c["destination"] for c in cargo),
                scoped=any(sum(loops[i]["key"] == c["color"] for i in c["enclosing"]) > 1 for c in cargo),
                split_routes=bool(split_routes),
                empty_leaf=any(not a["children"][i] and a["counts"][i] == 0 for i in range(len(loops))),
                nonrect=int(nonrect), background=a["background"], turn=a["turn"],
                distinct_keys=len({o["key"] for o in loops}),
                areas=tuple(len(c["cells"]) for c in cargo),
                receiving=tuple(a["counts"]), shape=grid.shape)


def _validate_output(a, output):
    out = np.rot90(output, a["turn"])
    bg = a["background"]
    objects = _objects(out, bg)
    frames = [o for o in objects if o["hole"].any()]
    points = [o for o in objects if not o["hole"].any()]
    _require(len(frames) == len(a["loops"]) and len(points) == len(a["cargo"]),
             "chamber count and cargo-unit count must be conserved")
    for frame in frames:
        y, x = frame["lo"]
        yy, xx = frame["hi"]
        _require(len(frame["cells"]) == 2 * (yy - y + xx - x), "output chamber must be a rectangular perimeter")
        colors = np.unique(out[frame["cells"][:, 0], frame["cells"][:, 1]])
        _require(len(colors) == 1, "output gates must become wall cells")
    _require(all(len(p["cells"]) == 1 for p in points), "every cargo unit must become one point")

    def check(input_id, frame_id):
        frame = frames[frame_id]
        colors = np.unique(out[frame["cells"][:, 0], frame["cells"][:, 1]])
        _require(int(colors[0]) == a["loops"][input_id]["wall"], "wall identity changed")
        children = [j for j, f in enumerate(frames) if j != frame_id and
                    (_containing(f["cells"], frames) or [None])[0] == frame_id]
        children.sort(key=lambda j: int(frames[j]["lo"][1]))
        local_points = [p for p in points if (_containing(p["cells"], frames) or [None])[0] == frame_id]
        _require(len(children) == len(a["children"][input_id]), "ordered containment changed")
        _require(len(local_points) == a["counts"][input_id], "cargo routed to the wrong chamber")
        for point in local_points:
            p = tuple(point["cells"][0])
            _require(out[p] == a["loops"][input_id]["wall"], "cargo must use destination wall color")
            _require(p[0] == frame["hi"][0] - 2, "cargo must dock behind the far-edge moat")
        if local_points:
            xs = sorted(int(p["lo"][1]) for p in local_points)
            _require(all(v - u == 2 for u, v in zip(xs, xs[1:])), "cargo shelf must have one-cell gaps")
        for i, j in zip(a["children"][input_id], children):
            check(i, j)
    roots = [i for i, f in enumerate(frames) if not _containing(f["cells"], frames)]
    _require(len(roots) == 1, "output must have one outer frame")
    check(a["root"], roots[0])
    root = frames[roots[0]]
    _require(np.array_equal(root["lo"], [0, 0]) and np.array_equal(root["hi"], np.array(out.shape) - 1),
             "output must be cropped to the outer frame")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features, test_features = set(), [], []
    train_ablations = {"ignore_scope": False, "ignore_routing": False, "count_pixels": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(out, _render_analysis(a)), "output disagrees with visible-input rule")
            _validate_output(a, out)
            f = _features(a)
            (train_features if split == "train" else test_features).append(f)
            if split == "train":
                for mode in train_ablations:
                    train_ablations[mode] |= not np.array_equal(out, _render_analysis(a, mode))
    _require(any(f["split_routes"] and f["skip"] for f in train_features), "training must demonstrate split color routing")
    _require(any(f["scoped"] and f["empty_leaf"] for f in train_features), "training must demonstrate nearest scope and an empty chamber")
    _require(all(train_ablations.values()), "training does not distinguish a parent-omitting shortcut")
    _require(max(f["depth"] for f in train_features) == 2 and all(f["depth"] == 3 for f in test_features),
             "generated tests must recombine the lessons at depth three")


def self_test():
    """Bounded diversity, input-only metamorphisms, and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, normalized, signatures = [], set(), set()
    for p in puzzles:
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(q["input"]) for s in ("train", "test") for q in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                records.append(dict(_features(a), split=split))
                # Remove global rotation as well as literal palette identities.
                canonical_out = np.rot90(pair["output"], a["turn"])
                roles = {}
                normalized.add((canonical_out.shape, tuple(roles.setdefault(int(c), len(roles))
                                                          for c in canonical_out.flat)))
    assert len(signatures) == 42
    assert len(normalized) >= 45
    assert len({r["shape"] for r in records}) >= 40
    assert {r["background"] for r in records} == set(range(10))
    assert {r["turn"] for r in records} == set(range(4))
    assert {r["depth"] for r in records} == {1, 2, 3}
    assert len({r["chambers"] for r in records}) >= 4
    assert len({r["cargo"] for r in records}) >= 5
    assert sum(r["nonrect"] for r in records) >= 100
    assert len({area for r in records for area in r["areas"]}) >= 6
    assert len({r["receiving"] for r in records}) >= 35
    tests = [r for r in records if r["split"] == "test"]
    assert {r["distinct_keys"] for r in tests} == {1, 2, 3}
    assert any(r["chambers"] == 4 for r in tests)
    assert len({r["receiving"] for r in tests}) >= 15
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    # The host verifier exhausts the legal count cross-product. Keep the local
    # self-test bounded while checking its edges and a multi-test interior case.
    for nt, ne in ((2, 1), (10, 1), (10, 4), (6, 3), (5, 4)):
        validate_generated_puzzle(generate_puzzle(901 + nt * 10 + ne, nt, ne))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    rng = np.random.default_rng(7381)
    for p in puzzles[:12]:
        pair = p["train"][0]
        inp, out = pair["input"], pair["output"]
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
        if max(inp.shape) <= 28:
            bg = int(inp[0, 0])
            assert np.array_equal(solve(np.pad(inp, 1, constant_values=bg)), out)
    # Handwritten geometry and expected pixels exercise the packing convention
    # without constructing the expected answer through _dock_tile.
    fixture = np.zeros((18, 18), dtype=np.int8)
    fixture[1, 1:17] = fixture[16, 1:17] = 4
    fixture[1:17, 1] = fixture[1:17, 16] = 4
    fixture[1, 4] = 1
    fixture[5, 5:13] = fixture[12, 5:13] = 5
    fixture[5:13, 5] = fixture[5:13, 12] = 5
    fixture[5, 8] = 1
    fixture[8, 7:9] = 1
    fixture[3, 4] = 1
    expected = np.zeros((11, 9), dtype=np.int8)
    expected[0, :] = expected[-1, :] = expected[:, 0] = expected[:, -1] = 4
    expected[2, 2:7] = expected[6, 2:7] = 5
    expected[2:7, 2] = expected[2:7, 6] = 5
    expected[4, 4], expected[8, 4] = 5, 4
    assert np.array_equal(solve(fixture), expected)
    # Change only a gate and the touching cargo's color partition, retaining
    # identical outline boxes. The routing answer must change.
    fixture2, expected2 = fixture.copy(), expected.copy()
    fixture2[5, 8], fixture2[8, 8] = 2, 2
    expected2[8, 4], expected2[8, 3], expected2[8, 5] = 0, 4, 4
    assert np.array_equal(solve(fixture2), expected2)
    assert not np.array_equal(expected, expected2)
    # Independent nonuniform stretching preserves topology and component count.
    factors = np.ones(18, dtype=int)
    factors[8] = 2
    stretched = np.repeat(np.repeat(fixture2, factors, axis=0), factors, axis=1)
    assert np.array_equal(solve(stretched), expected2)
    p = copy.deepcopy(puzzles[0])
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(records), "normalized_outputs": len(normalized),
            "normalization": "literal colors and global orientation removed",
            "input_shapes": len({r["shape"] for r in records}),
            "chamber_counts": sorted({r["chambers"] for r in records}),
            "cargo_counts": sorted({r["cargo"] for r in records}),
            "receiving_patterns": len({r["receiving"] for r in records}),
            "test_receiving_patterns": len({r["receiving"] for r in tests}),
            "test_distinct_key_counts": sorted({r["distinct_keys"] for r in tests}),
            "nonrectangular_chambers": sum(r["nonrect"] for r in records)}


def _puzzle_key(p):
    return tuple((_key(q["input"]), _key(q["output"])) for s in ("train", "test") for q in p[s])


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
