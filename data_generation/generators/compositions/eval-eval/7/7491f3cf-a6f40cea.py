"""Containment-gated X-ray windows: one scene, two inferred outline layers.

No parent module, external data, generation metadata, or example labels are used
by solve.  Coordinates of boxes are inclusive (top, bottom, left, right).
"""

from collections import defaultdict
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "grid size")
    _require(np.issubdtype(array.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= array) & (array <= 9)), "ARC colors required")
    return array


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _inside(point, box):
    y, x = point
    t, b, l, r = box
    return t < y < b and l < x < r


def _closed(point, box):
    y, x = point
    t, b, l, r = box
    return t <= y <= b and l <= x <= r


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _components(points, diagonal=False):
    remaining = set(points)
    directions = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in directions:
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _regions(cut):
    h, w = cut.shape
    # A path graph excludes diagonal shortcuts around orthogonal elbows.
    points = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for y, x in points:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y + dy, x + dx
                if not (dy or dx) or q not in points:
                    continue
                if dy and dx and ((y + dy, x) in points or (y, x + dx) in points):
                    continue
                neighbors.append(q)
        graph[y, x] = neighbors
    ends = [p for p in points if len(graph[p]) == 1]
    _require(len(points) >= 3 and len(ends) == 2
             and all(1 <= len(v) <= 2 for v in graph.values()), "cut must be a simple path")
    edge = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(edge(p) for p in ends) and all(not edge(p) for p in points - set(ends)),
             "only cut endpoints may touch the window edge")
    _require(len(_components(points, True)) == 1, "disconnected cut")
    groups = _components({(y, x) for y in range(h) for x in range(w) if not cut[y, x]})
    _require(len(groups) == 2, "cut must separate exactly two regions")
    masks = []
    for group in groups:
        mask = np.zeros((h, w), dtype=bool)
        for p in group:
            mask[p] = True
        masks.append(mask)
    return masks


def _control(interior):
    values = np.unique(interior)
    _require(len(values) == 2, "window needs fill and control only")
    candidates = []
    for control in values:
        groups = sorted(_components({tuple(map(int, p)) for p in
                                     np.argwhere(interior == control)}, True), key=len)
        if len(groups) != 2 or len(groups[0]) != 1 or len(groups[1]) < 3:
            continue
        marker = next(iter(groups[0]))
        cut = np.zeros(interior.shape, dtype=bool)
        for p in groups[1]:
            cut[p] = True
        try:
            regions = _regions(cut)
        except AssertionError:
            continue
        if min(max(abs(marker[0] - y), abs(marker[1] - x)) for y, x in groups[1]) < 2:
            continue
        side = 0 if regions[0][marker] else 1
        candidates.append((int(values[values != control][0]), int(control), cut, marker, regions, side))
    _require(len(candidates) == 1, "ambiguous cut/marker")
    return candidates[0]


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), "background corners")
    frames = []
    for color in np.unique(grid):
        if color == bg:
            continue
        positions = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        for group in _components(positions):
            box = _bbox(group)
            t, b, l, r = box
            if b - t < 4 or r - l < 4 or group != _perimeter(box):
                continue
            if not (0 < t < b < grid.shape[0] - 1 and 0 < l < r < grid.shape[1] - 1):
                continue
            try:
                control = _control(grid[t + 1:b, l + 1:r])
            except AssertionError:
                continue
            if int(color) in control[:2]:
                continue
            frames.append((box, int(color), control))
    _require(len(frames) == 1, "one identifiable control window required")
    box, frame, (fill, control, cut, local_marker, regions, side) = frames[0]
    t, b, l, r = box
    marker = local_marker[0] + t + 1, local_marker[1] + l + 1
    exterior = {(y, x) for y in range(grid.shape[0]) for x in range(grid.shape[1])
                if not _closed((y, x), box) and grid[y, x] != bg}
    groups = _components(exterior)
    # Each exposed component contains the entire one- or two-color palette of
    # its loop. Disjoint palettes join separated fragments without hidden data.
    by_palette = defaultdict(set)
    for group in groups:
        palette = frozenset(int(grid[p]) for p in group)
        _require(1 <= len(palette) <= 2 and not palette & {bg, fill, frame, control},
                 "unreadable exterior component palette")
        by_palette[palette].update(group)
    loops, used = [], set()
    for palette, visible in sorted(by_palette.items(), key=lambda item: min(item[1])):
        _require(not used & palette, "outline palettes must be disjoint")
        used.update(palette)
        loop_box = _bbox(visible)
        a, z, c, d = loop_box
        _require(z - a >= 2 and d - c >= 2, "degenerate rectangle")
        full = _perimeter(loop_box)
        _require({p for p in full if not _closed(p, box)} == visible,
                 "visible extrema/sides do not uniquely describe a rectangle")
        hidden = {p for p in full if _inside(p, box)}
        _require(marker not in full, "marker may not lie on a completed perimeter")
        if len(palette) == 1:
            word = (next(iter(palette)),)
        else:
            phases = {}
            for p in visible:
                phase = sum(p) % 2
                _require(phase not in phases or phases[phase] == int(grid[p]), "alternating phase conflict")
                phases[phase] = int(grid[p])
            _require(set(phases) == {0, 1} and set(phases.values()) == set(palette), "unwitnessed phase")
            word = phases[0], phases[1]
        owner = _inside(marker, loop_box)
        loops.append({"box": loop_box, "full": full, "visible": visible, "hidden": hidden,
                      "word": word, "owner": owner})
    # Same-layer conflicts have no visual priority; cross-layer hidden conflicts
    # are resolved by the spatial gate and the enclosing layer's seam priority.
    for i, one in enumerate(loops):
        for two in loops[i + 1:]:
            overlap = one["full"] & two["full"]
            _require(not overlap or (one["owner"] != two["owner"]
                                     and all(_inside(p, box) for p in overlap)),
                     "unresolved same-layer or exterior intersection")
    return {"grid": grid, "background": bg, "frame": frame, "fill": fill,
            "control": control, "window": box, "cut": cut, "marker": marker,
            "local_marker": local_marker, "regions": regions, "side": side, "loops": loops}


def _layers(scene):
    shape = scene["cut"].shape
    p = np.full(shape, scene["fill"], dtype=np.int8)
    q = p.copy()
    t, _, l, _ = scene["window"]
    for loop in scene["loops"]:
        layer = p if loop["owner"] else q
        for y, x in loop["hidden"]:
            layer[y - t - 1, x - l - 1] = loop["word"][(y + x) % len(loop["word"])]
    return p, q


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover rectangles, classify by enclosure, then reveal their gated crop."""
    s = _read(input_grid)
    p, q = _layers(s)
    marked, other = s["regions"][s["side"]], s["regions"][1 - s["side"]]
    out = np.full(s["cut"].shape, s["fill"], dtype=np.int8)
    # Explicit cellwise rule; the generation renderer below works on objects.
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            if s["cut"][y, x]:
                out[y, x] = p[y, x] if p[y, x] != s["fill"] else q[y, x]
            elif marked[y, x]:
                out[y, x] = p[y, x]
            else:
                out[y, x] = q[y, x]
    return out


def _evidence(scene):
    p, q = _layers(scene)
    a, b = p != scene["fill"], q != scene["fill"]
    cut, m = scene["cut"], scene["regions"][scene["side"]]
    o = scene["regions"][1 - scene["side"]]
    return {"keep_enclosing": bool(np.any(a & m)), "keep_other": bool(np.any(b & o)),
            "suppress_other": bool(np.any(b & ~a & m)), "suppress_enclosing": bool(np.any(a & ~b & o)),
            "seam_priority": bool(np.any(cut & a & b)), "seam_other": bool(np.any(cut & ~a & b)),
            "seam_blank": bool(np.any(cut & ~a & ~b))}


def _render(shape, window, loops, cut, local_marker, roles):
    bg, frame, fill, control = roles
    grid = np.full(shape, bg, dtype=np.int8)
    for loop in loops:
        for y, x in _perimeter(loop["box"]):
            grid[y, x] = loop["word"][(y + x) % len(loop["word"])]
    t, b, l, r = window
    grid[t:b + 1, l:r + 1] = frame
    grid[t + 1:b, l + 1:r] = fill
    panel = grid[t + 1:b, l + 1:r]
    panel[cut] = control
    panel[local_marker] = control
    marker = local_marker[0] + t + 1, local_marker[1] + l + 1
    regions = _regions(cut)
    marked = regions[0] if regions[0][local_marker] else regions[1]
    output = np.full(cut.shape, fill, dtype=np.int8)
    # Ground truth uses complete object descriptions and eligibility, not solve.
    for owner in (False, True):
        for loop in loops:
            if _inside(marker, loop["box"]) != owner:
                continue
            for y, x in _perimeter(loop["box"]):
                if not _inside((y, x), window):
                    continue
                v, u = y - t - 1, x - l - 1
                if cut[v, u] or bool(marked[v, u]) == owner:
                    output[v, u] = loop["word"][(y + x) % len(loop["word"])]
    return {"input": grid, "output": output}


def _geometry(rng, count, arbitrary=False):
    h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
    t, l = int(rng.integers(5, 7)), int(rng.integers(5, 7))
    oh = int(rng.integers(10, h - t - 6))
    ow = int(rng.integers(10, w - l - 6))
    b, r = t + oh + 1, l + ow + 1
    window = t, b, l, r
    if arbitrary:
        boxes, occupied = [], set()
        for _ in range(count):
            for _ in range(180):
                y0, y1 = sorted(int(v) for v in rng.choice(np.arange(1, h - 1), 2, replace=False))
                x0, x1 = sorted(int(v) for v in rng.choice(np.arange(1, w - 1), 2, replace=False))
                box = y0, y1, x0, x1
                full = _perimeter(box)
                visible = {p for p in full if not _closed(p, window)}
                hidden = {p for p in full if _inside(p, window)}
                if y1 - y0 < 3 or x1 - x0 < 3 or len(hidden) < 3 or not visible or _bbox(visible) != box:
                    continue
                halo = {q for y, x in occupied for q in ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}
                if visible & halo:
                    continue
                boxes.append(box)
                occupied.update(visible)
                break
            else:
                return None
        return (h, w), window, boxes
    # Opposite-corner loops can intersect only behind the window. Extra loops
    # are nonconcentric nested outlines, with independently sampled insets.
    ay, ax = int(rng.integers(7, oh)), int(rng.integers(7, ow))
    sy, sx = int(rng.integers(2, ay - 3)), int(rng.integers(2, ax - 3))
    nw = (int(rng.integers(1, t - 3)), t + ay, int(rng.integers(1, l - 3)), l + ax)
    se = (t + sy, int(rng.integers(b + 3, h - 1)), l + sx, int(rng.integers(r + 3, w - 1)))
    boxes = [nw, se]
    order = [0, 1] if bool(rng.integers(2)) else [1, 0]
    for i in order[:count - 2]:
        a, z, c, d = boxes[i]
        dy0, dy1, dx0, dx1 = (int(rng.integers(2, 4)) for _ in range(4))
        nested = (a + dy0, z - dy1, c + dx0, d - dx1)
        full = _perimeter(nested)
        visible = {p for p in full if not _closed(p, window)}
        if not visible or _bbox(visible) != nested or not any(_inside(p, window) for p in full):
            return None
        boxes.append(nested)
    return (h, w), window, boxes


def _sample_cut(rng, h, w, family):
    for _ in range(50):
        cut = np.zeros((h, w), dtype=bool)
        if family == "straight":
            if rng.random() < .5:
                cut[int(rng.integers(2, h - 2)), :] = True
            else:
                cut[:, int(rng.integers(2, w - 2))] = True
        elif family == "diagonal":
            # Eight-connected digital diagonal, with no horizontal jumps.
            a, b = sorted(int(v) for v in rng.choice(np.arange(1, w - 1), 2, replace=False))
            if b - a > h - 1:
                continue
            cols = np.rint(np.linspace(a, b, h)).astype(int)
            cut[np.arange(h), cols] = True
        elif family == "elbow":
            y, x = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            cut[:y + 1, x] = True
            cut[y, x:] = True
        elif family == "stepped":
            rows = sorted(int(v) for v in rng.choice(np.arange(2, h - 2, 2),
                                                    int(rng.integers(1, min(3, (h - 3) // 2) + 1)), replace=False))
            last, col = 0, int(rng.integers(2, w - 2))
            for row in rows:
                cut[last:row + 1, col] = True
                options = [x for x in range(2, w - 2) if x != col]
                nxt = int(rng.choice(options))
                cut[row, min(col, nxt):max(col, nxt) + 1] = True
                last, col = row, nxt
            cut[last:, col] = True
        else:
            raise ValueError("unknown cut family")
        if rng.random() < .5:
            cut = cut[:, ::-1].copy()
        if rng.random() < .5:
            cut = cut[::-1].copy()
        try:
            _regions(cut)
            return cut
        except AssertionError:
            continue
    raise AssertionError("failed to sample a two-region path")


def _marker_candidates(window, boxes, cut):
    regions = _regions(cut)
    near = cut.copy()
    h, w = cut.shape
    for y, x in np.argwhere(cut):
        near[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = True
    perimeters = [_perimeter(box) for box in boxes]
    overlaps = [(i, j) for i in range(len(boxes)) for j in range(i)
                if perimeters[i] & perimeters[j]]
    t, _, l, _ = window
    result = []
    for y, x in np.argwhere(~near):
        local = int(y), int(x)
        global_point = local[0] + t + 1, local[1] + l + 1
        if any(global_point in full for full in perimeters):
            continue
        owners = tuple(_inside(global_point, box) for box in boxes)
        if not any(owners) or all(owners) or any(owners[i] == owners[j] for i, j in overlaps):
            continue
        side = 0 if regions[0][local] else 1
        result.append((local, owners, side))
    return result


def _orient(pair, turns, flip):
    return {key: (np.fliplr(np.rot90(value, turns)) if flip else np.rot90(value, turns)).copy()
            for key, value in pair.items()}


def _owner_ranks(scene):
    """Evidence against largest/smallest and first/last rectangle shortcuts."""
    loops = scene["loops"]
    _require(len(loops) == 2 and sum(o["owner"] for o in loops) == 1, "two-class rank witness")
    ordered = sorted(loops, key=lambda o: o["box"])
    areas = [(o["box"][1] - o["box"][0] + 1) * (o["box"][3] - o["box"][2] + 1) for o in ordered]
    if areas[0] == areas[1]:
        return None
    owner = 0 if ordered[0]["owner"] else 1
    return areas[owner] > areas[1 - owner], owner


def _new_pair(rng, count=2, family="straight", demo=None, desired_side=None, desired_ranks=None):
    for attempt in range(220):
        geom = _geometry(rng, count, arbitrary=(demo is None and count <= 3 and attempt < 140 and rng.random() < .5))
        if geom is None:
            continue
        shape, window, boxes = geom
        t, b, l, r = window
        h, w = b - t - 1, r - l - 1
        if demo is not None:
            cut = np.zeros((h, w), dtype=bool)
            # Demo 0: the NW loop is enclosing; demo 1: the SE loop is
            # enclosing. This also reverses which loop terminates on the seam.
            cut_row = (boxes[1][0] if demo == 0 else boxes[0][1]) - t - 1
            cut[cut_row, :] = True
        else:
            cut = _sample_cut(rng, h, w, family)
        sites = _marker_candidates(window, boxes, cut)
        if demo == 0:
            sites = [s for s in sites if s[1] == (True, False) and s[0][0] < cut_row]
        elif demo == 1:
            sites = [s for s in sites if s[1] == (False, True) and s[0][0] < cut_row]
        if not sites:
            continue
        local, _, _ = sites[int(rng.integers(len(sites)))]
        colors = [int(c) for c in rng.permutation(10)]
        bg, frame, fill_color, control = colors[:4]
        fill = bg if (demo == 0 or (demo is None and rng.random() < .4)) else fill_color
        available = colors[4:]
        # Alternating loops appear in both ownership classes in the first two
        # demonstrations. Later examples freely mix solid and alternating loops.
        patterned = set(range(2)) if demo is not None else set(int(i) for i in rng.choice(count, int(rng.integers(0, min(2, count) + 1)), replace=False))
        loops = []
        for i, box in enumerate(boxes):
            if i in patterned:
                word = (available.pop(), available.pop())
                if rng.random() < .5:
                    word = word[::-1]
            else:
                word = (available.pop(),)
            loops.append({"box": box, "word": word})
        pair = _render(shape, window, loops, cut, local, (bg, frame, fill, control))
        orientations = list(range(8))
        rng.shuffle(orientations)
        for orientation in orientations:
            oriented = _orient(pair, orientation % 4, orientation >= 4)
            try:
                s = _read(oriented["input"])
            except AssertionError:
                continue
            if desired_side is not None and s["side"] != desired_side:
                continue
            if demo is not None:
                ranks = _owner_ranks(s)
                if ranks is None or (desired_ranks is not None and ranks != desired_ranks):
                    continue
            flags = _evidence(s)
            p, q = _layers(s)
            out = oriented["output"]
            visible_p = np.any((out == p) & (p != s["fill"]))
            visible_q = np.any((out == q) & (q != s["fill"]))
            if not (visible_p and visible_q):
                continue
            if demo is None and not (flags["suppress_enclosing"] or flags["suppress_other"]):
                continue
            _require(np.array_equal(solve(oriented["input"]), out), "renderer/reader disagree")
            return oriented
    raise AssertionError(f"could not construct readable scene: count={count}, family={family}, demo={demo}")


def _move_marker(pair, rng, same_owner, same_side):
    s = _read(pair["input"])
    owners = tuple(loop["owner"] for loop in s["loops"])
    sites = _marker_candidates(s["window"], [o["box"] for o in s["loops"]], s["cut"])
    sites = [p for p in sites if ((p[1] == owners) == same_owner)
             and ((p[2] == s["side"]) == same_side) and p[0] != s["local_marker"]]
    _require(sites, "no controlled marker move")
    local = sites[int(rng.integers(len(sites)))][0]
    grid = pair["input"].copy()
    grid[s["marker"]] = s["fill"]
    t, _, l, _ = s["window"]
    grid[local[0] + t + 1, local[1] + l + 1] = s["control"]
    reconstructed = _render(grid.shape, s["window"], s["loops"], s["cut"], local,
                            (s["background"], s["frame"], s["fill"], s["control"]))
    _require(np.array_equal(reconstructed["input"], grid), "controlled move altered the scene")
    out = reconstructed["output"]
    _require(np.array_equal(solve(grid), out), "controlled move renderer/reader disagree")
    _require(not np.array_equal(out, pair["output"]), "ineffective marker move")
    return {"input": grid, "output": out}


def _key(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    first = _new_pair(rng, demo=0)
    puzzle["train"].append(first)
    first_scene = _read(first["input"])
    area_rank, position_rank = _owner_ranks(first_scene)
    second = _new_pair(rng, demo=1, desired_side=1 - first_scene["side"],
                       desired_ranks=(not area_rank, 1 - position_rank))
    puzzle["train"].append(second)
    if nt >= 3:
        puzzle["train"].append(_move_marker(first, rng, same_owner=True, same_side=False))
    if nt >= 4:
        puzzle["train"].append(_move_marker(puzzle["train"][2], rng, same_owner=False, same_side=True))
    seen = {_key(p["input"]) for p in puzzle["train"]}
    families = ("straight", "diagonal", "elbow", "stepped")
    for split, count in (("train", nt - len(puzzle["train"])), ("test", nq)):
        for i in range(count):
            for _ in range(15):
                if split == "train":
                    loops = int(rng.integers(2, 4))
                elif i == 0 and nt > 4:
                    loops = 4
                else:
                    loops = int(rng.integers(3, 5))
                family = str(rng.choice(families))
                pair = _new_pair(rng, count=loops, family=family)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("duplicate inputs after bounded retries")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check actual visible geometry, evidence, gating and exact returned labels."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, taught, sides, pattern_classes = set(), set(), set(), set()
    train_counts, test_counts, witness_ranks = [], [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            s = _read(inp)
            _require(out.shape == s["cut"].shape, "strict window crop required")
            _require(np.array_equal(solve(inp.copy()), out), "wrong gated hidden perimeter")
            _require(s["control"] not in out and s["frame"] not in out, "controls leaked into answer")
            _require(all(len(o["hidden"]) >= 2 for o in s["loops"]), "generated outlines must be genuinely hidden")
            _require(any(o["owner"] for o in s["loops"]) and any(not o["owner"] for o in s["loops"]), "both ownership classes needed")
            p, q = _layers(s)
            union = q.copy()
            union[p != s["fill"]] = p[p != s["fill"]]
            _require(not np.array_equal(out, union), "omitting A's gate did not change the answer")
            _require(np.any(out != s["fill"]), "omitting B's completion did not change the answer")
            if split == "train":
                taught.update(k for k, v in _evidence(s).items() if v)
                sides.add(s["side"])
                pattern_classes.update(o["owner"] for o in s["loops"] if len(o["word"]) == 2)
                train_counts.append(len(s["loops"]))
                if len(witness_ranks) < 2:
                    witness_ranks.append(_owner_ranks(s))
            else:
                test_counts.append(len(s["loops"]))
    _require(taught == {"keep_enclosing", "keep_other", "suppress_other", "suppress_enclosing",
                        "seam_priority", "seam_other", "seam_blank"}, "insufficient training evidence")
    _require(sides == {0, 1}, "both effective marker sides must be taught")
    _require(pattern_classes == {False, True}, "phase continuation in both classes must be taught")
    _require(None not in witness_ranks and {p[0] for p in witness_ranks} == {False, True}
             and {p[1] for p in witness_ranks} == {0, 1}, "ownership order shortcuts not distinguished")
    _require(max(test_counts) > max(train_counts), "missing outline-count test shift")


def _family(cut):
    points = np.argwhere(cut)
    if len(np.unique(points[:, 0])) == 1 or len(np.unique(points[:, 1])) == 1:
        return "straight"
    if np.all(cut.sum(axis=1) == 1) or np.all(cut.sum(axis=0) == 1):
        return "diagonal"
    # One elbow has exactly one non-singleton row and one non-singleton column.
    if np.count_nonzero(cut.sum(axis=1) > 1) == 1 and np.count_nonzero(cut.sum(axis=0) > 1) == 1:
        return "elbow"
    return "stepped"


def self_test():
    """Bounded 42-seed coverage plus marker, palette, rotation and corruption checks."""
    fingerprints, shapes, layouts, cut_families, palettes, counts, nesting = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _read(pair["input"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                counts.add(len(s["loops"]))
                palettes.add((s["background"], s["frame"], s["fill"], s["control"]))
                layouts.add((pair["input"].shape, s["window"], tuple(o["box"] for o in s["loops"]), s["cut"].tobytes()))
                cut_families.add(_family(s["cut"]))
                for a in s["loops"]:
                    for b in s["loops"]:
                        if _inside((b["box"][0], b["box"][2]), a["box"]) and _inside((b["box"][1], b["box"][3]), a["box"]):
                            nesting.add((sum(a["box"][:2]) != sum(b["box"][:2]),
                                         sum(a["box"][2:]) != sum(b["box"][2:])))
        original, moved, changed = puzzle["train"][0], puzzle["train"][2], puzzle["train"][3]
        a, b, c = (_read(p["input"]) for p in (original, moved, changed))
        _require(sum(int(x != y) for x, y in zip(original["input"].flat, moved["input"].flat)) == 2, "side twin not isolated")
        _require(tuple(o["owner"] for o in a["loops"]) == tuple(o["owner"] for o in b["loops"]), "side twin changed ownership")
        _require(a["side"] != b["side"] and b["side"] == c["side"], "controlled side moves")
        _require(tuple(o["owner"] for o in b["loops"]) != tuple(o["owner"] for o in c["loops"]), "same-side ownership twin ineffective")
        _require(np.array_equal(original["output"][a["cut"]], moved["output"][a["cut"]]), "side move changed seam")
        _require(not np.array_equal(moved["output"], changed["output"]), "containment move changed nothing")
        for turns, flip in ((1, False), (0, True)):
            transformed = _orient(puzzle["test"][0], turns, flip)
            _require(np.array_equal(solve(transformed["input"]), transformed["output"]), "orientation equivariance")
        mapping = np.random.default_rng(seed + 4000).permutation(10)
        transformed = {k: mapping[v] for k, v in puzzle["test"][0].items()}
        _require(np.array_equal(solve(transformed["input"]), transformed["output"]), "palette equivariance")
    _require(len(fingerprints) == 42 and len(layouts) >= 120, "structural diversity")
    _require(len(shapes) >= 35 and len(palettes) >= 70, "geometry/palette diversity")
    _require(counts == {2, 3, 4} and cut_families == {"straight", "diagonal", "elbow", "stepped"}, "missing structural modes")
    _require((True, True) in nesting, "nonconcentric nesting absent")
    for i, (nt, nq) in enumerate(((2, 1), (3, 2), (10, 4))):
        one = generate_puzzle(2000 + i, nt, nq)
        two = generate_puzzle(2000 + i, nt, nq)
        _require(all(np.array_equal(x[k], y[k]) for s in one for x, y in zip(one[s], two[s]) for k in x), "determinism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(17, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
