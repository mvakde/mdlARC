"""Calibrated switch trails: orient instructions, trace pointers, exchange states.

Only Python and NumPy are used. No files, labels, random state, or training pairs
are read by solve(). See task_spec.md for the complete, input-visible convention.
"""

import hashlib
import itertools
import math
import numpy as np


SIDES = "TRBL"
NORMALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MATRICES = (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
    ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0)),
)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _map_side(code, side):
    a, b = MATRICES[code]
    y, x = NORMALS[side]
    return NORMALS.index((a[0] * y + a[1] * x, b[0] * y + b[1] * x))


def _calibration(inner, outer):
    matches = [k for k in range(8) if all(outer.get(_map_side(k, s)) == c
                                        for s, c in inner.items())]
    assert len(matches) == 1, "calibration must uniquely determine a D4 map"
    return matches[0]


def _reversed_calibration(code, corner):
    inner = {corner: 0, (corner + 1) % 4: 1}
    return _calibration(inner, {_map_side(code, s): 1 - c for s, c in inner.items()})


def _counts(rng, nt, ns):
    for name, value, lo, hi in (("num_train", nt, 2, 10), ("num_test", ns, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                 or not isinstance(value, (int, np.integer))
                                 or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if nt is None:
        if ns is None:
            ns = int(rng.integers(1, 5))
        nt = int(rng.integers(int(ns) + 1, 11))
    if ns is None:
        ns = int(rng.integers(1, min(4, int(nt) - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(ns)


def _sig(mapping):
    return tuple(sorted((int(y), int(x), int(c)) for (y, x), c in mapping.items()))


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(g.shape)); h.update(g.tobytes())
    return h.hexdigest()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        first = min(remaining)
        stack, group = [first], []
        remaining.remove(first)
        while stack:
            y, x = stack.pop()
            group.append((int(y), int(x)))
            for dy, dx in itertools.product((-1, 0, 1), repeat=2):
                q = y + dy, x + dx
                if q in remaining:
                    remaining.remove(q); stack.append(q)
        groups.append(group)
    return groups


def _render(grid, template, center, scale):
    half = scale // 2
    for (y, x), c in template.items():
        yy, xx = center[0] + y * scale - half, center[1] + x * scale - half
        assert 0 <= yy and yy + scale <= grid.shape[0]
        assert 0 <= xx and xx + scale <= grid.shape[1]
        grid[yy:yy + scale, xx:xx + scale] = c


def _next_node(nodes, point, direction):
    dy, dx = NORMALS[direction]
    hits = []
    for i, node in enumerate(nodes):
        y, x = node["center"]
        ry, rx = y - point[0], x - point[1]
        distance = ry * dy + rx * dx
        if ry * dx == rx * dy and distance > 0:
            hits.append((distance, i))
    return min(hits)[1] if hits else None


def _walk(nodes, directions, code, spark, entry, *, switched_routing=False):
    current = _next_node(nodes, spark, entry)
    path = []
    while current is not None and current not in path:
        path.append(current)
        state = nodes[current]["state"] ^ int(switched_routing)
        direction = _map_side(code, directions[state])
        current = _next_node(nodes, nodes[current]["center"], direction)
    return path, "repeat" if current is not None else "exit"


def _exchange(board, templates, nodes, path, bg):
    result = board.copy()
    union = set(templates[0]) | set(templates[1])
    blank = {p: bg for p in union}
    for i in path:
        node = nodes[i]
        _render(result, blank, node["center"], node["scale"])
        _render(result, templates[1 - node["state"]], node["center"], node["scale"])
    return result


def _design(rng, compact=False):
    """A common odd-size cross/rim, two differently directed colored flags."""
    h, w = (3, 3) if compact else _pick(rng, ((3, 3), (3, 5), (5, 3)))
    cy, cx = h // 2, w // 2
    carrier = {(y, 0) for y in range(-cy, cy + 1) if y}
    carrier |= {(0, x) for x in range(-cx, cx + 1) if x}
    # Off-axis cells change scaffold geometry, not merely its color or position.
    off_axis = [(y, x) for y in range(-cy, cy + 1) for x in range(-cx, cx + 1) if y and x]
    for point in off_axis:
        if rng.random() < .55:
            carrier.add(point)
    directions = [int(v) for v in rng.choice(4, 2, replace=False)]
    templates = []
    for state, side in enumerate(directions):
        c1, c2 = 2 + state * 2, 3 + state * 2
        mapping = {p: 1 for p in carrier}
        mapping[(0, 0)] = c1
        holes = [p for p in off_axis if p not in carrier]
        for p in holes:
            if rng.random() < .2:
                mapping[p] = c1
        dy, dx = NORMALS[side]
        extent = cy if dy else cx
        depth = 1 if compact else int(rng.integers(1, 3))
        width = int(_pick(rng, (1, 1, 3)))
        two_colors = bool(rng.integers(2))
        for distance in range(extent + 1, extent + depth + 1):
            for lateral in range(-(width // 2), width // 2 + 1):
                p = dy * distance + abs(dx) * lateral, dx * distance + abs(dy) * lateral
                mapping[p] = c2 if two_colors and distance == extent + depth else c1
        templates.append(mapping)
    return templates, directions


def _scene(rng, compact=False, enlarged=False):
    # Labels 0/1, 2..5, 6/7, 8 encode roles only during construction. A fresh
    # permutation makes their actual colors arbitrary and locally observable.
    logical, directions = _design(rng, compact)
    palette = [int(c) for c in rng.permutation(10)]
    templates = [{p: palette[c] for p, c in t.items()} for t in logical]
    bg, markers, spark_color = palette[0], palette[6:8], palette[8]
    union = set(logical[0]) | set(logical[1])
    ry, rx = max(abs(y) for y, x in union), max(abs(x) for y, x in union)
    eh, ew = 2 * ry + 1, 2 * rx + 1
    if enlarged:
        rows = cols = 2
        slots = list(itertools.product(range(2), repeat=2))
        scales = [1] * 4
        scales[int(rng.integers(4))] = 3
    else:
        options = [(r, c) for r in range(2, 4) for c in range(2, 4)
                   if r * eh + r - 1 + 4 <= 25 and c * ew + c - 1 + 4 <= 25]
        rows, cols = _pick(rng, options)
        all_slots = list(itertools.product(range(rows), range(cols)))
        count = int(rng.integers(max(3, rows, cols), min(8, len(all_slots)) + 1))
        chosen = rng.choice(len(all_slots), count, replace=False)
        slots = [all_slots[int(i)] for i in chosen]
        scales = [1] * len(slots)
    row_sizes = [max([eh * s for (r, c), s in zip(slots, scales) if r == y] or [eh]) for y in range(rows)]
    col_sizes = [max([ew * s for (r, c), s in zip(slots, scales) if c == x] or [ew]) for x in range(cols)]
    # Nonuniform distances and missing intersections produce sparse geometric
    # pointer graphs rather than a visible table or fixed layout vocabulary.
    def axes(sizes):
        spare = 25 - sum(sizes) - (len(sizes) - 1) - 4
        assert spare >= 0
        margins = [2, 2]
        gaps = [1] * (len(sizes) - 1)
        for _ in range(int(rng.integers(min(spare, 5) + 1))):
            target = int(rng.integers(len(gaps) + 2))
            if target < len(gaps):
                gaps[target] += 1
            else:
                margins[target - len(gaps)] += 1
        positions, start = [], margins[0]
        for i, size in enumerate(sizes):
            positions.append(start + size // 2)
            start += size + (gaps[i] if i < len(gaps) else 0)
        return positions, start + margins[1]
    ys, H = axes(row_sizes)
    xs, W = axes(col_sizes)
    nodes = [{"center": (ys[r], xs[c]), "scale": scale, "state": int(rng.integers(2))}
             for (r, c), scale in zip(slots, scales)]
    # Both templates are supplied at scale one, even in enlarged scenes.
    references = [i for i, n in enumerate(nodes) if n["scale"] == 1]
    nodes[references[0]]["state"], nodes[references[1]]["state"] = 0, 1
    board = np.full((H, W), bg, dtype=np.int8)
    for node in nodes:
        _render(board, templates[node["state"]], node["center"], node["scale"])
    sources = []
    for side in range(4):
        for node in nodes:
            y, x = node["center"]
            p = ((0, x), (y, W - 1), (H - 1, x), (y, 0))[side]
            if p not in [s[0] for s in sources]:
                sources.append((p, (side + 2) % 4))
    return dict(board=board, nodes=nodes, templates=templates, directions=directions,
                background=bg, markers=markers, spark_color=spark_color, sources=sources)


def _line(grid, side):
    return (grid[0, :], grid[:, -1], grid[-1, :], grid[:, 0])[side]


def _guide(grid, side, color, box):
    r, c, h, w = box
    if side == 0:
        grid[r - 1, c:c + w] = color
    elif side == 1:
        grid[r:r + h, c + w] = color
    elif side == 2:
        grid[r + h, c:c + w] = color
    else:
        grid[r:r + h, c - 1] = color


def _wrap(rng, scene, code, inner_corner, spark, entry):
    board = scene["board"].copy()
    board[spark] = scene["spark_color"]
    h, w = board.shape
    inner = {(inner_corner + i) % 4: scene["markers"][i] for i in range(2)}
    outer = {_map_side(code, s): color for s, color in inner.items()}
    # One blank cell separates guide bars from the outer controls. Corners are
    # always background, so no marker has priority at the shared corner.
    minimum = [2 + (s in inner) for s in range(4)]
    top, right, bottom, left = minimum
    extra_y, extra_x = 30 - h - top - bottom, 30 - w - left - right
    assert extra_y >= 0 and extra_x >= 0
    top += int(rng.integers(min(extra_y, 3) + 1))
    left += int(rng.integers(min(extra_x, 3) + 1))
    canvas = np.full((h + top + bottom, w + left + right), scene["background"], dtype=np.int8)
    box = top, left, h, w
    canvas[top:top + h, left:left + w] = board
    for side, color in inner.items():
        _guide(canvas, side, color, box)
    for side, color in outer.items():
        _line(canvas, side)[1:-1] = color
    path, _ = _walk(scene["nodes"], scene["directions"], code, spark, entry)
    output = _exchange(board, scene["templates"], scene["nodes"], path, scene["background"])
    return {"input": canvas, "output": output}


def _teaching_scene(rng, code, other):
    # Bounded search over tiny in-memory graphs. A contrast holds the complete
    # scene and inner guides fixed, changing only outer marker correspondence.
    for _ in range(256):
        scene = _scene(rng, compact=True, enlarged=True)
        sources = scene["sources"]
        for idx in rng.permutation(len(sources)):
            spark, entry = sources[int(idx)]
            paths = [_walk(scene["nodes"], scene["directions"], k, spark, entry)[0] for k in (code, other)]
            if not all(2 <= len(p) < len(scene["nodes"]) and
                       {scene["nodes"][i]["state"] for i in p} == {0, 1} for p in paths):
                continue
            if set(paths[0]) == set(paths[1]):
                continue
            no_key = _walk(scene["nodes"], scene["directions"], 0, spark, entry)[0]
            switched = [_walk(scene["nodes"], scene["directions"], k, spark, entry,
                              switched_routing=True)[0] for k in (code, other)]
            if set(no_key) == set(paths[0]) or all(set(p) == set(q) for p, q in zip(paths, switched)):
                continue
            # Enlarged substitution is visible, not merely present unchanged.
            if not any(scene["nodes"][i]["scale"] == 3 for p in paths for i in p):
                continue
            return scene, spark, entry
    raise AssertionError("teaching scene search exhausted its fixed budget")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    inner_corner = int(rng.integers(4))
    code = int(_pick(rng, (1, 3)))
    inner = {inner_corner: 0, (inner_corner + 1) % 4: 1}
    outer = {_map_side(code, s): 1 - c for s, c in inner.items()}
    other = _calibration(inner, outer)
    held = int(_pick(rng, [k for k in range(8) if k not in (code, other)]))
    scene, spark, entry = _teaching_scene(rng, code, other)
    # Reuse canvas placement too: exactly the outer correspondence changes.
    first = _wrap(rng, scene, code, inner_corner, spark, entry)
    second_input = first["input"].copy()
    m0, m1 = scene["markers"]
    for side in (_map_side(code, inner_corner), _map_side(code, (inner_corner + 1) % 4)):
        values = _line(second_input, side)[1:-1]
        values[:] = m1 if values[0] == m0 else m0
    board = scene["board"].copy(); board[spark] = scene["spark_color"]
    path = _walk(scene["nodes"], scene["directions"], other, spark, entry)[0]
    second = {"input": second_input,
              "output": _exchange(board, scene["templates"], scene["nodes"], path, scene["background"])}
    puzzle, seen = {"train": [first, second], "test": []}, {_key(first["input"]), _key(second_input)}
    for split, count in (("train", nt - 2), ("test", ns)):
        for _ in range(count):
            for attempt in range(192):
                k = held if split == "test" else int(_pick(rng, [j for j in range(8) if j != held]))
                enlarged = bool(rng.random() < .23)
                candidate = _scene(rng, compact=enlarged, enlarged=enlarged)
                corner = int(rng.integers(4))
                reversed_code = _reversed_calibration(k, corner)
                acceptable = []
                for p, heading in candidate["sources"]:
                    route = _walk(candidate["nodes"], candidate["directions"], k, p, heading)[0]
                    if not (2 <= len(route) < len(candidate["nodes"]) and
                            {candidate["nodes"][i]["state"] for i in route} == {0, 1}):
                        continue
                    reversed_route = _walk(candidate["nodes"], candidate["directions"], reversed_code, p, heading)[0]
                    literal_route = _walk(candidate["nodes"], candidate["directions"], 0, p, heading)[0]
                    if set(route) != set(reversed_route) and (k == 0 or set(route) != set(literal_route)):
                        acceptable.append((p, heading))
                if not acceptable:
                    continue
                p, heading = _pick(rng, acceptable)
                pair = _wrap(rng, candidate, k, corner, p, heading)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key); puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("pair construction exhausted its fixed budget")
    validate_generated_puzzle(puzzle)
    return puzzle


def _array(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
    return grid


def _read_frame(grid):
    bg = int(grid[0, 0])
    assert all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    outer = {}
    for side in range(4):
        values = _line(grid, side)[1:-1]
        assert len(values) >= 4 and len(np.unique(values)) == 1
        if values[0] != bg:
            outer[side] = int(values[0])
    assert len(outer) == 2 and len(set(outer.values())) == 2
    assert any(set(outer) == {i, (i + 1) % 4} for i in range(4))
    bars = []
    for color in outer.values():
        positions = np.argwhere(grid[1:-1, 1:-1] == color) + 1
        assert len(positions) >= 4
        ys, xs = positions[:, 0], positions[:, 1]
        if len(set(ys)) == 1:
            assert len(positions) == int(xs.max() - xs.min() + 1)
            bars.append(("h", int(ys[0]), int(xs.min()), int(xs.max()), color))
        else:
            assert len(set(xs)) == 1 and len(positions) == int(ys.max() - ys.min() + 1)
            bars.append(("v", int(xs[0]), int(ys.min()), int(ys.max()), color))
    assert {b[0] for b in bars} == {"h", "v"}
    hor = next(b for b in bars if b[0] == "h")
    ver = next(b for b in bars if b[0] == "v")
    _, hr, x0, x1, hc = hor
    _, vc, y0, y1, vcolor = ver
    hs = 0 if hr == y0 - 1 else 2 if hr == y1 + 1 else None
    vs = 3 if vc == x0 - 1 else 1 if vc == x1 + 1 else None
    assert hs is not None and vs is not None
    inner, box = {hs: hc, vs: vcolor}, (y0, x0, y1 - y0 + 1, x1 - x0 + 1)
    r, c, h, w = box
    assert r >= 2 + (0 in inner) and c >= 2 + (3 in inner)
    assert r + h <= grid.shape[0] - 2 - (2 in inner)
    assert c + w <= grid.shape[1] - 2 - (1 in inner)
    board = grid[r:r + h, c:c + w].copy()
    assert not set(np.unique(board)) & set(outer.values())
    check = np.full(grid.shape, bg, dtype=grid.dtype)
    check[r:r + h, c:c + w] = board
    for s, color in inner.items():
        _guide(check, s, color, box)
    for s, color in outer.items():
        _line(check, s)[1:-1] = color
    assert np.array_equal(check, grid), "unexpected material outside scene"
    return dict(board=board, background=bg, inner=inner, outer=outer,
                bounds=box, code=_calibration(inner, outer))


def _primitive(crop):
    scale = 0
    for rows in (crop, crop.T):
        for row in rows:
            ends = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1), len(row)]
            for a, b in zip(ends, ends[1:]):
                scale = math.gcd(scale, int(b - a))
    assert scale in (1, 3), "only odd unit or triple pixel blocks are used"
    small = crop[::scale, ::scale]
    assert np.array_equal(crop, np.repeat(np.repeat(small, scale, axis=0), scale, axis=1))
    return scale, small


def _read_scene(info):
    board, bg = info["board"], info["background"]
    groups = _components(board != bg)
    sparks = [g for g in groups if len(g) == 1]
    assert len(sparks) == 1, "one isolated boundary pulse is required"
    spark = sparks[0][0]
    y, x = spark
    edges = [s for s, yes in enumerate((y == 0, x == board.shape[1] - 1,
                                      y == board.shape[0] - 1, x == 0)) if yes]
    assert len(edges) == 1, "pulse must lie on one non-corner boundary cell"
    spark_color = int(board[spark])
    assert np.count_nonzero(board == spark_color) == 1
    objects = []
    for group in groups:
        if len(group) == 1:
            continue
        y0, y1 = min(p[0] for p in group), max(p[0] for p in group)
        x0, x1 = min(p[1] for p in group), max(p[1] for p in group)
        crop = board[y0:y1 + 1, x0:x1 + 1]
        assert np.count_nonzero(crop != bg) == len(group)
        scale, small = _primitive(crop)
        mapping = {(int(y), int(x)): int(small[y, x]) for y, x in zip(*np.nonzero(small != bg))}
        objects.append(dict(origin=(y0, x0), scale=scale, raw=mapping))
    assert 3 <= len(objects) <= 8
    shared = set.intersection(*(set(o["raw"].values()) for o in objects))
    assert len(shared) == 1, "exactly one common scaffold color"
    carrier_color = next(iter(shared))
    signatures, carriers, nodes = [], set(), []
    for obj in objects:
        points = [p for p, c in obj["raw"].items() if c == carrier_color]
        ymin, ymax = min(p[0] for p in points), max(p[0] for p in points)
        xmin, xmax = min(p[1] for p in points), max(p[1] for p in points)
        h, w = ymax - ymin + 1, xmax - xmin + 1
        assert (h, w) in ((3, 3), (3, 5), (5, 3))
        cy, cx = ymin + h // 2, xmin + w // 2
        template = {(y - cy, x - cx): c for (y, x), c in obj["raw"].items()}
        carriers.add(tuple(sorted(p for p, c in template.items() if c == carrier_color)))
        signatures.append(_sig(template))
        s = obj["scale"]
        nodes.append(dict(center=(obj["origin"][0] + cy * s + s // 2,
                                  obj["origin"][1] + cx * s + s // 2), scale=s))
    assert len(carriers) == 1
    vocabulary = sorted(set(signatures))
    assert len(vocabulary) == 2
    templates = [{(y, x): c for y, x, c in sig} for sig in vocabulary]
    carrier = next(iter(carriers))
    cy, cx = max(abs(y) for y, x in carrier), max(abs(x) for y, x in carrier)
    directions = []
    for t in templates:
        exterior = set()
        for y, x in t:
            if y < -cy: exterior.add(0)
            if x > cx: exterior.add(1)
            if y > cy: exterior.add(2)
            if x < -cx: exterior.add(3)
        assert len(exterior) == 1, "a state must have one exterior flag side"
        directions.append(next(iter(exterior)))
    assert len(set(directions)) == 2
    assert set(templates[0].values()) & set(templates[1].values()) == {carrier_color}
    boxes = []
    union = set(templates[0]) | set(templates[1])
    for node, signature in zip(nodes, signatures):
        node["state"] = vocabulary.index(signature)
        s = node["scale"]
        y0 = node["center"][0] + min(y for y, x in union) * s - s // 2
        y1 = node["center"][0] + max(y for y, x in union) * s + s // 2
        x0 = node["center"][1] + min(x for y, x in union) * s - s // 2
        x1 = node["center"][1] + max(x for y, x in union) * s + s // 2
        assert 1 <= y0 <= y1 < board.shape[0] - 1 and 1 <= x0 <= x1 < board.shape[1] - 1
        for a, b, c, d in boxes:
            assert y1 + 1 < a or b + 1 < y0 or x1 + 1 < c or d + 1 < x0, "union regions collide"
        boxes.append((y0, y1, x0, x1))
    assert {n["state"] for n in nodes if n["scale"] == 1} == {0, 1}
    reconstructed = np.full(board.shape, bg, dtype=board.dtype)
    reconstructed[spark] = spark_color
    for n in nodes:
        _render(reconstructed, templates[n["state"]], n["center"], n["scale"])
    assert np.array_equal(reconstructed, board)
    info.update(nodes=nodes, templates=templates, directions=directions, spark=spark,
                entry=(edges[0] + 2) % 4, carrier=carrier, carrier_color=carrier_color)
    return info


def _analyze(input_grid):
    info = _read_scene(_read_frame(_array(input_grid)))
    path, stop = _walk(info["nodes"], info["directions"], info["code"], info["spark"], info["entry"])
    assert path, "pulse must reach a switch"
    info.update(path=path, stop=stop)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    return _exchange(info["board"], info["templates"], info["nodes"], info["path"], info["background"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            assert out.shape == info["board"].shape, "crop must retain the exact guide-defined rectangle"
            nodes, path = info["nodes"], info["path"]
            assert 2 <= len(path) < len(nodes), "need both activated and untouched switches"
            assert {nodes[i]["state"] for i in path} == {0, 1}, "both exchange directions must be visible"
            expected = _exchange(info["board"], info["templates"], nodes, path, info["background"])
            assert np.array_equal(out, expected), "wrong selective scaffold-registered exchange"
            assert np.array_equal(out == info["carrier_color"], info["board"] == info["carrier_color"])
            assert int(out[info["spark"]]) == int(info["board"][info["spark"]])
            # Also derive the activated SET by fixed-point graph reachability.
            reached = {_next_node(nodes, info["spark"], info["entry"])}
            while True:
                more = set(reached)
                for i in reached:
                    target = _next_node(nodes, nodes[i]["center"], _map_side(info["code"], info["directions"][nodes[i]["state"]]))
                    if target is not None:
                        more.add(target)
                if more == reached: break
                reached = more
            assert reached == set(path)
            corner = next(i for i in range(4) if set(info["inner"]) == {i, (i + 1) % 4})
            reversed_route = _walk(nodes, info["directions"], _reversed_calibration(info["code"], corner),
                                   info["spark"], info["entry"])[0]
            assert set(path) != set(reversed_route), "compass correspondence must affect activation"
            infos[split].append(info)
    a, b = infos["train"][:2]
    assert a["bounds"] == b["bounds"] and np.array_equal(a["board"], b["board"])
    assert a["inner"] == b["inner"] and set(a["outer"]) == set(b["outer"])
    assert a["code"] != b["code"] and set(a["path"]) != set(b["path"])
    assert a["code"] in (1, 3) and np.linalg.det(np.array(MATRICES[b["code"]])) == -1
    assert any(n["scale"] == 3 for t in (a, b) for i, n in enumerate(t["nodes"]) if i in t["path"])
    assert set(_walk(a["nodes"], a["directions"], 0, a["spark"], a["entry"])[0]) != set(a["path"])
    assert any(set(_walk(t["nodes"], t["directions"], t["code"], t["spark"], t["entry"],
                         switched_routing=True)[0]) != set(t["path"]) for t in (a, b))
    assert not ({t["code"] for t in infos["train"]} & {t["code"] for t in infos["test"]})
    return None


def _features(info):
    nodes = info["nodes"]
    return dict(code=info["code"], count=len(nodes), active=len(info["path"]), stop=info["stop"],
                scales=tuple(sorted(n["scale"] for n in nodes)), carrier=info["carrier"],
                directions=tuple(sorted(info["directions"])), board_shape=info["board"].shape,
                palette=tuple(int(c) for c in np.unique(info["board"])),
                flag_sizes=tuple(sorted(len(set(a) - set(b)) for a, b in
                                       ((info["templates"][0], info["templates"][1]),
                                        (info["templates"][1], info["templates"][0])))))


def self_test():
    """42 streamed seeds, bounded mutations and explicit count checks; no I/O."""
    features, fingerprints = [], set()
    codes = {"train": set(), "test": set()}
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_digest(p))
        if seed < 3:
            assert _digest(p) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                before = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
                info = _analyze(before)
                features.append(_features(info)); codes[split].add(info["code"])
    assert len(fingerprints) == 42
    assert codes["train"] == codes["test"] == set(range(8))
    assert len({f["carrier"] for f in features}) >= 20
    assert len({f["board_shape"] for f in features}) >= 35
    assert {f["count"] for f in features} >= {3, 4, 5, 6}
    assert {f["stop"] for f in features} == {"repeat", "exit"}
    assert {f["directions"] for f in features} == set(itertools.combinations(range(4), 2))
    assert len({f["flag_sizes"] for f in features}) >= 5
    assert len({f["palette"] for f in features}) >= 50
    assert max(f["active"] for f in features) >= 4
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12000 + nt * 10 + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((None, None), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(100, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    for mutation in ("cell", "crop", "guide", "all_switches", "ignore_key", "reverse_routing"):
        bad = {s: [{r: v.copy() for r, v in pair.items()} for pair in sample[s]] for s in sample}
        pair = bad["train"][0]
        info = _analyze(pair["input"])
        if mutation == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif mutation == "crop":
            pair["output"] = pair["output"][1:].copy()
        elif mutation == "guide":
            r, c, h, w = info["bounds"]
            s = next(iter(info["inner"]))
            point = ((r - 1, c + w // 2), (r + h // 2, c + w),
                     (r + h, c + w // 2), (r + h // 2, c - 1))[s]
            pair["input"][point] = info["background"]
        else:
            if mutation == "all_switches":
                path = list(range(len(info["nodes"])))
            else:
                path = _walk(info["nodes"], info["directions"], 0 if mutation == "ignore_key" else info["code"],
                             info["spark"], info["entry"], switched_routing=mutation == "reverse_routing")[0]
                if set(path) == set(info["path"]):
                    # The contrast guarantees a witness in one of its two scenes.
                    pair = bad["train"][1]; info = _analyze(pair["input"])
                    path = _walk(info["nodes"], info["directions"], info["code"], info["spark"],
                                 info["entry"], switched_routing=True)[0]
            pair["output"] = _exchange(info["board"], info["templates"], info["nodes"], path, info["background"])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError(f"mutation accepted: {mutation}")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(fingerprints),
            "calibrations_by_split": {s: sorted(c) for s, c in codes.items()},
            "carrier_shapes": len({f["carrier"] for f in features}),
            "board_shapes": len({f["board_shape"] for f in features}),
            "motif_counts": sorted({f["count"] for f in features}),
            "path_lengths": sorted({f["active"] for f in features}),
            "stop_modes": sorted({f["stop"] for f in features})}
