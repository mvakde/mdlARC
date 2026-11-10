"""A keyed relay: reflect, rescale, and irreversibly crop a carried motif.

Construction uses a logical lattice. The input-only solver instead decodes
frames/anchors and transfers the actual pixels at exact rational scales.
"""
from itertools import product

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return array.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(component)
    return result


def _routes(graph, labels, key, start, limit=2):
    """Enumerate distinct-station routes, with the visible prototype as start."""
    answers = []

    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for node in graph[path[-1]]:
            if node not in used and labels[node] == key[len(path)]:
                visit(path + [node], used | {node})

    if labels[start] == key[0]:
        visit([start], {start})
    return answers


def _graph(stations):
    """Visibility of anchor centers; a third frame blocks a straight hop."""
    graph = [[] for _ in stations]
    for i, first in enumerate(stations):
        for j, second in enumerate(stations):
            if i == j:
                continue
            ar, ac = first["center"]
            br, bc = second["center"]
            if ar == br:
                lo, hi = sorted((ac, bc))
                blocked = any(2 * s["box"][0] - 1 <= ar <= 2 * s["box"][1] - 1
                              and 2 * s["box"][2] - 1 < hi and 2 * s["box"][3] - 1 > lo
                              for k, s in enumerate(stations) if k not in (i, j))
            elif ac == bc:
                lo, hi = sorted((ar, br))
                blocked = any(2 * s["box"][2] - 1 <= ac <= 2 * s["box"][3] - 1
                              and 2 * s["box"][0] - 1 < hi and 2 * s["box"][1] - 1 > lo
                              for k, s in enumerate(stations) if k not in (i, j))
            else:
                continue
            if not blocked:
                graph[i].append(j)
    return graph


def _decode(grid):
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "background border"
    occupied_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert len(occupied_rows)
    key_y = int(occupied_rows[-1])
    xs = np.flatnonzero(grid[key_y] != bg)
    assert 3 <= len(xs) <= 14 and np.all(np.diff(xs) == 2), "detached ordered key"
    assert np.all(grid[key_y - 1] == bg), "key needs a blank separating row"
    key = [int(grid[key_y, x]) for x in xs]
    assert len(set(key)) >= 2
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg} - set(key)):
        boxes = []
        for comp in _components(grid == color):
            rs, cs = zip(*comp)
            r0, r1, c0, c1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
            h, w = r1 - r0, c1 - c0
            if (min(h, w) < 3 or len(comp) != 2 * h + 2 * w - 4
                    or not all(r in (r0, r1 - 1) or c in (c0, c1 - 1) for r, c in comp)):
                boxes = []
                break
            boxes.append((r0, r1, c0, c1))
        if len(boxes) >= 3:
            candidates.append((color, sorted(boxes)))
    assert len(candidates) == 1, "one shared rectangular-frame color"
    carrier, boxes = candidates[0]
    stations, sources = [], []
    rebuilt = np.full(grid.shape, bg, dtype=np.int8)
    rebuilt[key_y, xs] = key
    for box in boxes:
        r0, r1, c0, c1 = box
        assert r1 <= key_y - 1
        interior = grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
        colors = set(map(int, np.unique(interior))) - {bg}
        assert colors and carrier not in colors
        if len(colors) == 1:
            label = next(iter(colors))
        else:
            label = key[0]
            assert label in colors and not (colors - {label}) & set(key)
            sources.append(len(stations))
        points = np.argwhere(interior == label)
        assert len(points)
        top, left = points.min(0)
        bottom, right = points.max(0) + 1
        side = int(bottom - top)
        assert side == right - left and len(points) == side * side, "solid square label anchor"
        ar, ac = r0 + 1 + int(top), c0 + 1 + int(left)
        stations.append(dict(box=box, anchor=(ar, ac, side), label=label,
                             center=(2 * ar + side - 1, 2 * ac + side - 1)))
        assert np.all(rebuilt[r0:r1, c0:c1] == bg), "frames overlap"
        rebuilt[r0:r1, c0:c1] = grid[r0:r1, c0:c1]
    assert len(sources) == 1 and np.array_equal(rebuilt, grid), "exactly one prototype and no noise"
    source = sources[0]
    labels = [s["label"] for s in stations]
    source_box = stations[source]["box"]
    ar, ac, side = stations[source]["anchor"]
    r0, r1, c0, c1 = source_box
    motif = {(r - ar, c - ac): int(grid[r, c]) for r in range(r0 + 1, r1 - 1)
             for c in range(c0 + 1, c1 - 1) if grid[r, c] != bg}
    details = set(motif.values()) - {labels[source]}
    assert not details & set(labels), "detail colors cannot also be station labels"
    mask = grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1] != bg
    assert len(_components(mask)) == 1, "source prototype is connected"
    graph = _graph(stations)
    paths = _routes(graph, labels, key, source)
    assert len(paths) == 1, "key must resolve one complete route"
    return dict(background=bg, carrier=carrier, key=key, key_y=key_y,
                stations=stations, source=source, motif=motif, side=side,
                route=paths[0], graph=graph)


def _transfer_pixels(motif, source_side, target_side, axis, bg):
    """Refine actual pixels, then merge only uniformly colored rational cells.

    Every pixel is a unit rectangle relative to the top-left anchor boundary.
    Padding to anchor-side multiples gives integral destination raster bounds.
    Reflection includes the internal arrangement of all sub-anchor details.
    """
    if not motif:
        return {}
    rs, cs = zip(*motif)
    top = min(rs) // source_side * source_side
    left = min(cs) // source_side * source_side
    bottom = (max(rs) // source_side + 1) * source_side
    right = (max(cs) // source_side + 1) * source_side
    raw = np.full((bottom - top, right - left), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r - top, c - left] = color
    refined = np.repeat(np.repeat(raw, target_side, axis=0), target_side, axis=1)
    h, w = refined.shape[0] // source_side, refined.shape[1] // source_side
    blocks = refined.reshape(h, source_side, w, source_side)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(low, high), "scale would straddle a color boundary"
    top, left = top * target_side // source_side, left * target_side // source_side
    if axis == "v":
        low = low[::-1]
        top = target_side - top - h
    elif axis == "h":
        low = low[:, ::-1]
        left = target_side - left - w
    return {(int(r) + top, int(c) + left): int(low[r, c])
            for r, c in np.argwhere(low != bg)}


def _clip(motif, station):
    ar, ac, _ = station["anchor"]
    r0, r1, c0, c1 = station["box"]
    return {p: color for p, color in motif.items()
            if r0 < ar + p[0] < r1 - 1 and c0 < ac + p[1] < c1 - 1}


def _prediction(grid, info=None, *, reset=False, reflect=True, scale=True, disabled_axis=None):
    info = _decode(grid) if info is None else info
    result = grid.copy()
    current, side = info["motif"], info["side"]
    previous = info["stations"][info["source"]]
    source_label = previous["label"]
    parity = [False, False]
    records = []
    for node in info["route"][1:]:
        target = info["stations"][node]
        axis = "h" if target["center"][0] == previous["center"][0] else "v"
        parity[axis == "h"] = not parity[axis == "h"]
        target_side = target["anchor"][2] if scale else side
        if reset:
            # An ablation: correct cumulative reflections but restore lost detail.
            transformed = _transfer_pixels(info["motif"], info["side"], target_side, None, info["background"])
            if reflect:
                for active, flip_axis in zip(parity, ("v", "h")):
                    if active and flip_axis != disabled_axis:
                        transformed = _transfer_pixels(transformed, target_side, target_side, flip_axis, info["background"])
            old_label = source_label
        else:
            transformed = _transfer_pixels(current, side, target_side,
                                           axis if reflect and axis != disabled_axis else None,
                                           info["background"])
            old_label = previous["label"]
        transformed = {p: target["label"] if color == old_label else color for p, color in transformed.items()}
        visible = _clip(transformed, target)
        ar, ac, anchor_side = target["anchor"]
        for (r, c), color in visible.items():
            # Anchor geometry supplied by the input is always protected. This
            # also makes the no-scaling ablation a well-defined wrong answer.
            if color != target["label"] and not (0 <= r < anchor_side and 0 <= c < anchor_side):
                result[ar + r, ac + c] = color
        records.append(dict(node=node, axis=axis, source_side=side,
                            target_side=target["anchor"][2],
                            lost=len(transformed) - len(visible),
                            details=sum(color != target["label"] for color in visible.values())))
        current, side, previous = visible, target_side, target
    return result, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the visible key, frames, label anchors, and source pixels."""
    grid = _grid(input_grid)
    return _prediction(grid)[0]


def _without_route(grid, info):
    """Ablation: independently complete every chamber from the original.

    Use position relative to the source to choose reflections, as in a static
    panel-copy interpretation. There is no ordered relay or station selection.
    """
    result = grid.copy()
    source = info["stations"][info["source"]]
    for node, target in enumerate(info["stations"]):
        if node == info["source"]:
            continue
        side = target["anchor"][2]
        transformed = _transfer_pixels(info["motif"], info["side"], side, None, info["background"])
        for coord, axis in ((0, "v"), (1, "h")):
            if target["center"][coord] != source["center"][coord]:
                transformed = _transfer_pixels(transformed, side, side, axis, info["background"])
        ar, ac, _ = target["anchor"]
        for (r, c), color in _clip(transformed, target).items():
            if color != source["label"]:
                result[ar + r, ac + c] = color
    return result


def _logical_graph(cells):
    stations = [dict(center=(2 * r, 2 * c), box=(r, r + 1, c, c + 1)) for r, c in cells]
    return _graph(stations)


def _path(rng, graph, length):
    budget = 1500

    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        for node in rng.permutation(graph[path[-1]]):
            node = int(node)
            if node not in path:
                answer = visit(path + [node])
                if answer is not None:
                    return answer
        return None

    for start in rng.permutation(len(graph)):
        answer = visit([int(start)])
        if answer is not None:
            return answer
    return None


def _orient(cells, a, turns, flip):
    result = dict(cells)
    for _ in range(turns):
        result = {(c, a - 1 - r): color for (r, c), color in result.items()}
    if flip:
        result = {(r, a - 1 - c): color for (r, c), color in result.items()}
    return result


def _motif(rng, a, label, details, small):
    motif = {(r, c): label for r in range(a) for c in range(a)}
    motif[-1, 0] = details[0]
    motif[0, a] = details[-1]
    extra = int(rng.integers(0, 3 if small else 9))
    low, high = (-1, a) if small else (-2, a + 1)
    for step in range(extra):
        frontier = sorted({(r + dr, c + dc) for r, c in motif for dr, dc in
                           ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if low <= r + dr <= high and low <= c + dc <= high
                           and (r + dr, c + dc) not in motif})
        if not frontier:
            break
        motif[frontier[int(rng.integers(len(frontier)))]] = int(rng.choice(details))
    return _orient(motif, a, int(rng.integers(4)), bool(rng.integers(2)))


def _extents(motif, a):
    rs, cs = zip(*motif)
    return (max(0, -min(rs)), max(0, max(rs) + 1 - a),
            max(0, -min(cs)), max(0, max(cs) + 1 - a))


def _logical_clip(motif, a, extents):
    t, b, l, r = extents
    return {(y, x): color for (y, x), color in motif.items() if -t <= y < a + b and -l <= x < a + r}


def _cropped_window(rng, motif, a, label):
    extents = _extents(motif, a)
    choices = []
    for window in product(*(range(e + 1) for e in extents)):
        visible = _logical_clip(motif, a, window)
        if len(visible) < len(motif) and any(color != label for color in visible.values()):
            choices.append(window)
    return choices[int(rng.integers(len(choices)))] if choices else None


def _layout(rng, length, unused, fallback=False):
    if fallback:
        snake = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
        cells = sorted(snake[:length])
        return cells, [cells.index(p) for p in snake[:length]]
    count = length + unused
    shapes = [(r, c) for r in range(1, 5) for c in range(1, 5)
              if count <= r * c <= count + 4 and r * c <= 12]
    rows, cols = shapes[int(rng.integers(len(shapes)))]
    universe = list(product(range(rows), range(cols)))
    cells = sorted(universe[int(i)] for i in rng.choice(len(universe), count, replace=False))
    route = _path(rng, _logical_graph(cells), length)
    return cells, route


def _labels(rng, cells, route, available, witness, fallback):
    graph = _logical_graph(cells)
    if witness == 1:
        # One locally plausible duplicate branch dies at the next key entry.
        key = list(available[:4])
        if rng.random() < .5:
            key[-1] = key[0]
        labels = [key[1]] * len(cells)
        for node, color in zip(route, key):
            labels[node] = color
        return labels, key
    if fallback:
        key = available[:len(route)]
        return [key[route.index(i)] for i in range(len(cells))], key
    for _ in range(96):
        k = min(len(available), int(rng.integers(2, min(len(route), len(available)) + 1)))
        pool = available[:k]
        key = list(rng.permutation(pool + [int(rng.choice(pool)) for _ in range(len(route) - k)]))
        key = [int(x) for x in key]
        labels = [int(rng.choice(available)) for _ in cells]
        for node, color in zip(route, key):
            labels[node] = color
        if _routes(graph, labels, key, route[0]) == [tuple(route)]:
            return labels, key
    return None


def _paint_logical(grid, motif, station, unit):
    ar, ac, _ = station["anchor"]
    for (r, c), color in motif.items():
        grid[ar + r * unit:ar + (r + 1) * unit,
             ac + c * unit:ac + (c + 1) * unit] = color


def _build(rng, witness=None, test=False, fallback=False):
    a = (2 if witness == 1 else 1) if fallback else int(rng.choice((1, 2, 2)))
    palette = [int(c) for c in rng.permutation(10)]
    bg, carrier = palette[:2]
    detail_count = 1 if fallback else int(rng.integers(2, 4))
    details, available = palette[2:2 + detail_count], palette[2 + detail_count:]
    if witness == 0:
        cells = [(0, 0), (0, 1), (0, 2)]
        if rng.random() < .5:
            cells = [(c, r) for r, c in cells]
        route = [0, 1, 2] if rng.random() < .5 else [2, 1, 0]
    elif witness == 1:
        walk = [(1, 1), (1, 0), (0, 0), (0, 1)]
        cells = walk + [(1, 2)]
        # Rotate or reflect the entire layout, independently of motif orientation.
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        coords = {p: i for i, p in enumerate(cells)}
        transformed = _orient(coords, 1, turns, flip)
        mapped = {value: point for point, value in transformed.items()}
        mr, mc = min(r for r, c in mapped.values()), min(c for r, c in mapped.values())
        cells = sorted((r - mr, c - mc) for r, c in mapped.values())
        route = [cells.index((mapped[i][0] - mr, mapped[i][1] - mc)) for i in range(4)]
    else:
        length = int(rng.integers(5, 8) if test else rng.integers(3, 6))
        cells, route = _layout(rng, length, int(rng.integers(0, 3)), fallback)
        if route is None:
            return None
    assigned = _labels(rng, cells, route, available, witness, fallback)
    if assigned is None:
        return None
    labels, key = assigned
    if fallback:
        simple = {(r, c): labels[route[0]] for r in range(a) for c in range(a)}
        simple.update({(-1, 0): details[0], (0, a): details[0]})
        motif = _orient(simple, a, int(rng.integers(4)), bool(rng.integers(2)))
    else:
        motif = _motif(rng, a, labels[route[0]], details, witness is not None)
    # All four axis-reflection possibilities must differ in the full witness.
    if witness == 0 and any(_orient(motif, a, turns, flip) == motif
                            for turns, flip in ((0, True), (2, True), (2, False))):
        return None
    units = [1] * len(cells)
    options = (1, 3) if a == 1 else (1, 2, 3)
    if witness == 0:
        picked = (1, options[1], 1) if fallback or rng.random() < .5 else (options[1], 1, options[1])
        for node, unit in zip(route, picked):
            units[node] = unit
    elif witness == 1:
        units[route[int(rng.integers(2))]] = options[1]
    elif not fallback:
        units = [int(rng.choice(options, p=(.83, .17) if a == 1 else (.78, .19, .03))) for _ in cells]
    windows, carried = [None] * len(cells), {route[0]: motif}
    whole, current = motif, motif
    for step, node in enumerate(route):
        if step:
            previous = route[step - 1]
            horizontal = cells[node][0] == cells[previous][0]
            def hop(pattern):
                return {(r if horizontal else a - 1 - r,
                         a - 1 - c if horizontal else c):
                        labels[node] if color == labels[previous] else color
                        for (r, c), color in pattern.items()}
            whole, current = hop(whole), hop(current)
        should_crop = (witness == 1 and step == 1) or (witness is None and step > 0 and rng.random() < .38)
        window = _cropped_window(rng, current, a, labels[node]) if should_crop else None
        if witness == 1 and step == 1 and window is None:
            return None
        if window is None:
            # Later large windows could accommodate even the lost original parts.
            window = tuple(e + (int(rng.random() < .23) if not fallback else 0) for e in _extents(whole, a))
        current = _logical_clip(current, a, window)
        windows[node], carried[node] = window, current
    for node in range(len(cells)):
        if windows[node] is None:
            # Unused chambers can accommodate details, so copying everywhere
            # visibly disagrees with the key-selected relay.
            t, b, l, rr = _extents(motif, a)
            windows[node] = (max(t, b), max(t, b), max(l, rr), max(l, rr))
    rows, cols = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
    above, below, before, after = [0] * rows, [0] * rows, [0] * cols, [0] * cols
    for node, (r, c) in enumerate(cells):
        t, b, l, rr = windows[node]
        unit, side = units[node], a * units[node]
        above[r] = max(above[r], side // 2 + t * unit + 1)
        below[r] = max(below[r], side - side // 2 + b * unit + 1)
        before[c] = max(before[c], side // 2 + l * unit + 1)
        after[c] = max(after[c], side - side // 2 + rr * unit + 1)
    # Empty coordinate levels retain real space, permitting skipped-level links.
    for vector in (above, below, before, after):
        for i in range(len(vector)):
            vector[i] = max(vector[i], 1)
    row_gaps = [int(rng.integers(1, 4)) for _ in range(rows - 1)]
    col_gaps = [int(rng.integers(1, 5)) for _ in range(cols - 1)]
    block_h = sum(above) + sum(below) + sum(row_gaps)
    block_w = sum(before) + sum(after) + sum(col_gaps)
    min_h, min_w = block_h + 4, max(block_w + 2, 2 * len(key) + 1)
    if max(min_h, min_w) > 30:
        return None
    h = int(rng.integers(min_h, min(30, min_h + 4) + 1))
    w = int(rng.integers(min_w, min(30, min_w + 5) + 1))
    if rng.random() < .10:
        h = w = 30
    top = int(rng.integers(1, h - block_h - 2))
    left = int(rng.integers(1, w - block_w))
    cy, cx = [top + above[0]], [left + before[0]]
    for r in range(1, rows):
        cy.append(cy[-1] + below[r - 1] + row_gaps[r - 1] + above[r])
    for c in range(1, cols):
        cx.append(cx[-1] + after[c - 1] + col_gaps[c - 1] + before[c])
    inp = np.full((h, w), bg, dtype=np.int8)
    stations = []
    for node, (r, c) in enumerate(cells):
        unit, side = units[node], a * units[node]
        ar, ac = cy[r] - side // 2, cx[c] - side // 2
        t, b, l, rr = windows[node]
        r0, r1, c0, c1 = ar - t * unit - 1, ar + (a + b) * unit + 1, ac - l * unit - 1, ac + (a + rr) * unit + 1
        inp[r0:r1, c0] = inp[r0:r1, c1 - 1] = carrier
        inp[r0, c0:c1] = inp[r1 - 1, c0:c1] = carrier
        inp[ar:ar + side, ac:ac + side] = labels[node]
        stations.append(dict(box=(r0, r1, c0, c1), anchor=(ar, ac, side)))
    _paint_logical(inp, motif, stations[route[0]], units[route[0]])
    key_y = int(rng.integers(top + block_h + 1, h - 1))
    key_x = int(rng.integers(1, w - (2 * len(key) - 1)))
    inp[key_y, key_x:key_x + 2 * len(key):2] = key
    out = inp.copy()
    for node in route[1:]:
        _paint_logical(out, carried[node], stations[node], units[node])
    return dict(input=inp, output=out)


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed, nt, nq = _integer(seed, "seed"), _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0 or not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("nonnegative seed, 2..10 train, 1..4 test, and train > test required")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else None
            pair = None
            for attempt in range(128):
                candidate = _build(rng, witness, split == "test", fallback=attempt >= 96)
                if candidate is None:
                    continue
                if _signature(candidate["input"]) in seen:
                    # Forty bijections give different inputs on a scene with
                    # at least four colors; only at most thirteen are occupied.
                    options = [(m * np.arange(10) + b) % 10 for m in (1, 3, 7, 9) for b in range(10)]
                    for remap in options:
                        variant = {k: remap[v].astype(np.int8) for k, v in candidate.items()}
                        if _signature(variant["input"]) not in seen:
                            candidate = variant
                            break
                    else:
                        raise AssertionError("finite palette space exhausted")
                info = _decode(candidate["input"])
                expected, records = _prediction(candidate["input"], info)
                assert np.array_equal(expected, candidate["output"]), "lattice construction / pixel decode disagree"
                if witness == 0:
                    if (np.array_equal(expected, _prediction(candidate["input"], info, reflect=False)[0])
                            or np.array_equal(expected, _prediction(candidate["input"], info, scale=False)[0])):
                        continue
                if witness == 1:
                    if (np.array_equal(expected, _prediction(candidate["input"], info, reset=True)[0])
                            or any(np.array_equal(expected, _prediction(candidate["input"], info, disabled_axis=axis)[0])
                                   for axis in ("h", "v"))):
                        continue
                pair = candidate
                break
            if pair is None:
                raise RuntimeError("bounded construction exhausted")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _decode(inp)
    expected, records = _prediction(inp, info)
    assert np.array_equal(expected, out), "incorrect relay output"
    assert not np.array_equal(inp, out), "identity pair"
    protected = inp != info["background"]
    assert np.array_equal(inp[protected], out[protected]), "input marks must be preserved"
    allowed = np.zeros(inp.shape, dtype=bool)
    for node in info["route"][1:]:
        r0, r1, c0, c1 = info["stations"][node]["box"]
        allowed[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = True
    assert np.array_equal(inp[~allowed], out[~allowed]), "only selected chamber interiors change"
    source_label = info["stations"][info["source"]]["label"]
    previous_details = sum(color != source_label for color in info["motif"].values())
    for record in records:
        before_area = previous_details * record["target_side"] ** 2
        after_area = record["details"] * record["source_side"] ** 2
        assert after_area <= before_area, "normalized carried detail area cannot increase"
        assert before_area - after_area == record["lost"] * record["source_side"] ** 2, "loss is actual aperture clipping"
        previous_details = record["details"]
    info["records"] = records
    info["reset_differs"] = not np.array_equal(expected, _prediction(inp, info, reset=True)[0])
    info["reflection_differs"] = not np.array_equal(expected, _prediction(inp, info, reflect=False)[0])
    info["scaling_differs"] = not np.array_equal(expected, _prediction(inp, info, scale=False)[0])
    info["route_differs"] = not np.array_equal(expected, _without_route(inp, info))
    info["axis_evidence"] = {axis for axis in ("h", "v")
                             if not np.array_equal(expected, _prediction(inp, info, disabled_axis=axis)[0])}
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    analyses, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _signature(np.asarray(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(info)
    training = analyses["train"]
    assert any(a["reflection_differs"] and a["scaling_differs"]
               and not any(r["lost"] for r in a["records"])
               and len(a["records"]) >= 2 and len({r["axis"] for r in a["records"]}) == 1
               for a in training), "complete two-hop reflection/scale witness"
    assert any(a["reset_differs"] and a["route_differs"] and len({r["axis"] for r in a["records"]}) == 2
               and len(a["stations"]) > len(a["route"]) for a in training), "clipping history and route-choice witness"
    assert any(any(sum(node not in a["route"][:step] and a["stations"][node]["label"] == a["key"][step]
                       for node in a["graph"][a["route"][step - 1]]) > 1
                   for step in range(1, len(a["route"]))) for a in training), "global key disambiguation witness"
    train_axes = {r["axis"] for a in training for r in a["records"]}
    assert set.union(*(a["axis_evidence"] for a in training)) == {"h", "v"}, "both reflections must be observable"
    assert all({r["axis"] for r in a["records"]} <= train_axes for a in analyses["test"])
    # No split-specific output rule is permitted. The test distribution simply
    # composes more already-demonstrated hops in a fresh arrangement.
    assert all(5 <= len(a["route"]) <= 7 for a in analyses["test"])


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(np.asarray(p["input"])), _signature(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def _motif_signature(info):
    """Normalize palette and common lattice enlargement, keeping the anchor."""
    motif = info["motif"]
    boundaries = [info["side"]]
    for (r, c), color in motif.items():
        for dr, dc, offset in ((-1, 0, r), (1, 0, r + 1), (0, -1, c), (0, 1, c + 1)):
            if motif.get((r + dr, c + dc), info["background"]) != color:
                boundaries.append(offset)
    quantum = int(np.gcd.reduce(boundaries))
    normalized = {(r // quantum, c // quantum): color for (r, c), color in motif.items()}
    palette = {info["stations"][info["source"]]["label"]: 0}
    return tuple((r, c, palette.setdefault(color, len(palette))) for (r, c), color in sorted(normalized.items()))


def self_test():
    """Bounded 42-seed semantic/diversity, metamorphic, and corruption checks."""
    # A half-cell centerline touching a third frame's outer pixel edge blocks
    # a hop; moving that frame one row farther away restores the clear hop.
    edge_case = [dict(center=(5, 3), box=(1, 5, 0, 4)),
                 dict(center=(5, 23), box=(1, 5, 10, 14)),
                 dict(center=(8, 12), box=(3, 6, 5, 8))]
    assert 1 not in _graph(edge_case)[0]
    edge_case[2] = dict(center=(10, 12), box=(4, 7, 5, 8))
    assert 1 in _graph(edge_case)[0]
    puzzles, shapes, layouts, motifs, sides, ratios, roles = set(), set(), set(), set(), set(), set(), set()
    lost_counts, branches, repeated, absent = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_signature(puzzle))
        if seed < 3:
            assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                shapes.add(pair["input"].shape)
                roles.add((a["background"], a["carrier"]))
                cr, cc = a["stations"][0]["center"]
                layouts.add(tuple((s["center"][0] - cr, s["center"][1] - cc,
                                   s["box"][1] - s["box"][0], s["box"][3] - s["box"][2]) for s in a["stations"]))
                motifs.add(_motif_signature(a))
                sides.update(s["anchor"][2] for s in a["stations"])
                for rec in a["records"]:
                    divisor = int(np.gcd(rec["source_side"], rec["target_side"]))
                    ratios.add((rec["target_side"] // divisor, rec["source_side"] // divisor))
                lost_counts.add(sum(r["lost"] > 0 for r in a["records"]))
                repeated += len(set(a["key"])) < len(a["key"])
                absent += any(s["label"] not in a["key"] for s in a["stations"])
                route = a["route"]
                branches += any(sum(s not in route[:step] and a["stations"][s]["label"] == a["key"][step]
                                    for s in a["graph"][route[step - 1]]) > 1 for step in range(1, len(route)))
    assert len(puzzles) == 42 and len(shapes) >= 35 and len(layouts) >= 100 and len(motifs) >= 75
    assert len(roles) >= 35 and sides >= {1, 2, 3, 4} and len(ratios) >= 5
    assert lost_counts >= {0, 1, 2} and branches >= 42 and repeated >= 30 and absent >= 5
    for seed in range(4):
        for witness in (0, 1, None):
            pair = _build(np.random.default_rng(seed), witness, test=witness is None, fallback=True)
            assert pair is not None
            a = _analyze(pair)
            if witness == 0:
                assert a["reflection_differs"] and a["scaling_differs"] and not any(r["lost"] for r in a["records"])
            if witness == 1:
                assert a["reset_differs"] and a["route_differs"] and a["axis_evidence"] == {"h", "v"}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(201)
    pair = p["test"][0]
    # Color bijection and translation through blank padding preserve the rule.
    palette = np.array([7, 1, 8, 0, 9, 3, 4, 6, 2, 5], dtype=np.int8)
    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    small = next(q for q in p["train"] if max(q["input"].shape) <= 28)
    bg = int(small["input"][0, 0])
    assert np.array_equal(solve(np.pad(small["input"], ((1, 1), (1, 1)), constant_values=bg)),
                          np.pad(small["output"], ((1, 1), (1, 1)), constant_values=bg))
    broken = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed relay self-test passed")
