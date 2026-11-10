"""Routed mosaic repair: each cable's two corner fragments define one union.

Construction paints answers from sampled templates. The input-only solver
instead segments frames and traces the visible directional cable graph.
Only Python's standard library and NumPy are used.
"""
from itertools import permutations
import numpy as np


_AXES = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DOMAIN = tuple((u, v) for u in range(1, 4) for v in range(1, 4))
_ALTERNATIVES = ("bank_order", "color_groups", "no_reflection", "rotation", "global",
                 "largest", "intersection", "xor")


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        found, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t, b + 1)} |
            {(y, r) for y in range(t, b + 1)})


def _corners(rect):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))


def _positions(tile, pattern):
    r, c = tile["anchor"]
    dr, dc = _AXES[tile["q"]]
    return {(r + dr * u, c + dc * v) for u, v in pattern}


def _find_tiles(g, background, frame):
    components = _components(map(tuple, np.argwhere(g == frame)))
    assert 4 <= len(components) <= 8
    tiles, occupied = [], set()
    for cells in components:
        t = min(p[0] for p in cells); b = max(p[0] for p in cells)
        l = min(p[1] for p in cells); r = max(p[1] for p in cells)
        rect = tuple(map(int, (t, l, b, r)))
        assert 5 <= b - t + 1 <= 7 and 5 <= r - l + 1 <= 7
        outline = _outline(rect)
        missing = outline - cells
        assert cells <= outline and len(missing) == 1
        anchor = next(iter(missing))
        assert anchor in _corners(rect)
        color = int(g[anchor])
        assert color not in (background, frame)
        q = _corners(rect).index(anchor)
        dr, dc = _AXES[q]
        footprint = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
        assert not occupied & footprint
        occupied |= footprint
        fragment = set()
        for y in range(t + 1, b):
            for x in range(l + 1, r):
                assert int(g[y, x]) in (background, color)
                if g[y, x] == color:
                    uv = ((y - anchor[0]) * dr, (x - anchor[1]) * dc)
                    assert uv in _DOMAIN
                    fragment.add(uv)
        tiles.append(dict(rect=rect, anchor=anchor, q=q, color=color,
                          fragment=frozenset(fragment), footprint=footprint))
    return tiles, occupied


def _parse(input_grid):
    """Interpret roles, frames, fragments and pairings solely from input cells."""
    g = _grid(input_grid)
    corners = [int(g[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1, "background corners disagree"
    background = corners[0]
    candidates = []
    for frame in sorted(set(map(int, np.unique(g))) - {background}):
        try:
            tiles, footprint = _find_tiles(g, background, frame)
            field = {tuple(map(int, p)) for p in np.argwhere(g != background)
                     if tuple(p) not in footprint}
            assert field
            colors = {int(g[p]) for p in field}
            assert 1 <= len(colors) <= 2
            endpoints = {}
            for index, tile in enumerate(tiles):
                t, l, b, r = tile["rect"]
                middle_rows = range((t + b) // 2, (t + b + 1) // 2 + 1)
                middle_cols = range((l + r) // 2, (l + r + 1) // 2 + 1)
                options = ([((t - 1, cx), 0) for cx in middle_cols] +
                           [((cy, r + 1), 1) for cy in middle_rows] +
                           [((b + 1, cx), 2) for cx in middle_cols] +
                           [((cy, l - 1), 3) for cy in middle_rows])
                found = [(p, d) for p, d in options if p in field]
                assert len(found) == 1, "terminal has missing or multiple attachments"
                p, d = found[0]
                assert p not in endpoints
                endpoints[p] = index
                tile["port"], tile["side"] = p, d
            # Structural roles come from visible terminal stubs, not frequency.
            port_colors = {int(g[p]) for p in endpoints}
            assert len(port_colors) == 1
            carrier = port_colors.pop()
            other = colors - {carrier}
            crossing = next(iter(other)) if other else None
            assert {t["color"] for t in tiles}.isdisjoint({frame, carrier, crossing})
            graph = {}
            for p in field:
                edges = {d: (p[0] + dr, p[1] + dc)
                         for d, (dr, dc) in enumerate(_DIRS)
                         if (p[0] + dr, p[1] + dc) in field}
                degree = len(edges)
                if p in endpoints:
                    assert degree == 1 and g[p] == carrier
                elif degree == 4:
                    assert crossing is not None and g[p] == crossing
                else:
                    assert degree == 2 and g[p] == carrier
                graph[p] = edges
            if crossing is not None:
                assert any(len(edges) == 4 for edges in graph.values())
            pairs, routes, used, visited = [], [], set(), set()
            for i, tile in enumerate(tiles):
                if i in used:
                    continue
                p = tile["port"]
                incoming = None
                states, route = set(), []
                while True:
                    assert (p, incoming) not in states, "strand contains a loop"
                    states.add((p, incoming)); visited.add(p); route.append(p)
                    if incoming is not None and p in endpoints:
                        j = endpoints[p]
                        assert j != i and j not in used
                        break
                    edges = graph[p]
                    if incoming is None:
                        outgoing = next(iter(edges))
                    elif len(edges) == 4:
                        outgoing = (incoming + 2) % 4
                    else:
                        outgoing = next(d for d in edges if d != incoming)
                    p = edges[outgoing]
                    incoming = (outgoing + 2) % 4
                used.update((i, j)); pairs.append((i, j)); routes.append(route)
            assert len(used) == len(tiles) and visited == field
            assert all(tiles[i]["fragment"] | tiles[j]["fragment"] for i, j in pairs)
            candidates.append(dict(grid=g, background=background, frame=frame,
                                   carrier=carrier, crossing=crossing, tiles=tiles,
                                   footprint=footprint, field=field, graph=graph,
                                   pairs=pairs, routes=routes))
        except (AssertionError, StopIteration):
            continue
    assert len(candidates) == 1, "input lacks a unique frame/cable interpretation"
    return candidates[0]


def _predict(info, alternative=None):
    """Normal union completion plus explicit simpler-rule countermodels."""
    tiles = info["tiles"]
    patterns = [set(t["fragment"]) for t in tiles]
    pairs = info["pairs"]
    if alternative == "bank_order":
        sides = {t["side"] for t in tiles}
        assert sides in ({0, 2}, {1, 3})
        a, b = sorted(sides)
        axis = 1 if a == 0 else 0
        near = sorted((i for i, t in enumerate(tiles) if t["side"] == a),
                      key=lambda i: tiles[i]["port"][axis])
        far = sorted((i for i, t in enumerate(tiles) if t["side"] == b),
                     key=lambda i: tiles[i]["port"][axis])
        assert len(near) == len(far)
        pairs = list(zip(near, far))
    elif alternative == "rotation":
        patterns = [{(v, u) for u, v in p} if t["q"] in (1, 2) else p
                    for p, t in zip(patterns, tiles)]
    elif alternative == "no_reflection":
        patterns = [{(u * _AXES[t["q"]][0], v * _AXES[t["q"]][1]) for u, v in p}
                    for p, t in zip(patterns, tiles)]
    result = info["grid"].copy()
    # Clearing the observed pixels matters for intersection/XOR countermodels.
    for t in tiles:
        for p in _positions(t, t["fragment"]):
            result[p] = info["background"]
    global_pattern = set().union(*patterns)
    for i, j in pairs:
        a, b = patterns[i], patterns[j]
        if alternative == "global":
            union = global_pattern
        elif alternative == "largest":
            union = a if len(a) >= len(b) else b
        elif alternative == "intersection":
            union = a & b
        elif alternative == "xor":
            union = a ^ b
        else:
            union = a | b
        for index in (i, j):
            tile = tiles[index]
            pattern = (set().union(*(p for p, t in zip(patterns, tiles)
                                    if t["color"] == tile["color"]))
                       if alternative == "color_groups" else union)
            if alternative == "rotation" and tile["q"] in (1, 2):
                pattern = {(v, u) for u, v in pattern}
            if alternative == "no_reflection":
                ar, ac = tile["anchor"]
                points = {(ar + u, ac + v) for u, v in pattern}
            else:
                points = _positions(tile, pattern)
            for r, c in points:
                if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                    result[r, c] = tile["color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the repair determined by this input, without seed or examples."""
    return _predict(_parse(input_grid))


def _layout(rng, n):
    """Staggered unequal terminal rectangles, with no shared endpoint rows."""
    heights_a = [5 if n == 4 else int(rng.choice((5, 7))) for _ in range(n)]
    heights_b = [5 if n == 4 else int(rng.choice((5, 7))) for _ in range(n)]
    widths_a = list(map(int, rng.integers(5, 8, n)))
    widths_b = list(map(int, rng.integers(5, 8, n)))
    shifts = list(map(int, rng.integers(2, 4 if n == 4 else 5, n)))
    rows_a = [max(heights_a[0] // 2 + 1, heights_b[0] // 2 + 1 - shifts[0])]
    for i in range(1, n):
        gap = max(heights_a[i - 1] // 2 + heights_a[i] // 2 + 3,
                  heights_b[i - 1] // 2 + heights_b[i] // 2 + 3
                  + shifts[i - 1] - shifts[i], shifts[i - 1] + 2)
        rows_a.append(rows_a[-1] + gap + (0 if n == 4 else int(rng.integers(2))))
    rows_b = [r + s for r, s in zip(rows_a, shifts)]
    all_rows = rows_a + rows_b
    assert min(abs(a - b) for i, a in enumerate(all_rows) for b in all_rows[i + 1:]) >= 2
    height = max(r + h // 2 + 1 for r, h in zip(all_rows, heights_a + heights_b))
    assert height <= 30
    budget = 30 - height
    top = int(rng.integers(min(2, budget) + 1)); budget -= top
    bottom = int(rng.integers(min(3, budget) + 1))
    rows_a = [r + top for r in rows_a]; rows_b = [r + top for r in rows_b]
    height += top + bottom
    lefts = list(map(int, rng.integers(1, 4, n)))
    left = [(r - h // 2, l, r + h // 2, l + w - 1)
            for r, h, l, w in zip(rows_a, heights_a, lefts, widths_a)]
    return height, left, rows_a, rows_b, heights_b, widths_b


def _segment(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    length = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return [(a[0] + k * dr, a[1] + k * dc) for k in range(length + 1)]


def _pattern(rng, asymmetric=False, excluded=None):
    for _ in range(32):
        count = int(rng.integers(3 if asymmetric else 2, 9 if asymmetric else 10))
        p = frozenset(_DOMAIN[int(i)] for i in rng.choice(9, count, replace=False))
        if p != excluded and (not asymmetric or p != frozenset((v, u) for u, v in p)):
            return p
    # Finite, visibly exposed fallback; it is never a hidden solver dictionary.
    for p in (frozenset({(1, 1), (1, 2), (2, 3)}),
              frozenset({(1, 3), (2, 1), (3, 3), (3, 2)})):
        if p != excluded:
            return p
    raise AssertionError("template fallback exhausted")


def _split_pattern(rng, pattern, mode):
    p = sorted(pattern)
    rng.shuffle(p)
    if mode == "overlap":
        cut = int(rng.integers(1, len(p) - 1))
        return set(p[:cut + 1]), set(p[cut:])
    if mode == "disjoint":
        cut = int(rng.integers(1, len(p)))
        return set(p[:cut]), set(p[cut:])
    if mode == "empty_a":
        return set(), set(p)
    if mode == "empty_b":
        return set(p), set()
    if mode == "full":
        a, b = set(p), set(p[:int(rng.integers(len(p)))])
        return (b, a) if rng.random() < .5 else (a, b)
    a, b = set(), set()
    for q in p:
        kind = int(rng.integers(3))
        if kind != 1:
            a.add(q)
        if kind != 0:
            b.add(q)
    if a == b == set(p):
        a.remove(p[0])
    return a, b


def _build_pair(rng, n, lesson, regime, palette, transpose, target_override=None):
    background, frame, carrier, crossing = palette[:4]
    height, left, rows_a, rows_b, heights_b, widths_b = _layout(rng, n)
    target = list(map(int, rng.permutation(n)))
    if target_override is not None:
        target = list(target_override)
    elif lesson == "distributed" or (regime == "routing" and lesson is None):
        target = list(reversed(range(n)))
    all_rows = rows_a + rows_b
    free_rows = [r for r in range(1, height - 1)
                 if all(abs(r - y) >= 2 for y in all_rows)]
    detour = None
    if free_rows and (lesson == "empty" or regime == "routing" or rng.random() < .4):
        detour = int(rng.integers(n))
    group_sizes = [2 if i == detour else 1 for i in range(n)]
    # Reserve the minimum route width first, then distribute only spare cells.
    # A four-cable excursion must still fit without depending on lucky retries.
    slots = sum(group_sizes)
    min_width = max(rect[3] + 1 for rect in left) + 2 + 2 * (slots - 1) + 3 + max(widths_b) + 1
    if min_width > 30:
        left = [(t, 1, b, r - l + 1) for t, l, b, r in left]
        min_width = max(rect[3] + 1 for rect in left) + 2 + 2 * (slots - 1) + 3 + max(widths_b) + 1
    assert min_width <= 30
    # Pick an expansion budget; do not bias every query toward the canvas cap.
    spare = int(rng.integers(30 - min_width + 1))
    lead_extra = int(rng.integers(min(2, spare) + 1)); spare -= lead_extra
    col = max(rect[3] + 1 for rect in left) + 2 + lead_extra
    groups = []
    allocated = 0
    for size in group_sizes:
        group = []
        for _ in range(size):
            group.append(col)
            allocated += 1
            extra = int(rng.integers(min(1, spare) + 1)) if allocated < slots else 0
            spare -= extra
            col += 2 + extra
        groups.append(group)
    # Shuffle groups, retaining the unique paired columns of a detour cable.
    order = list(map(int, rng.permutation(n)))
    if detour is not None:
        # Assign one double-column group to the chosen route, singles arbitrarily.
        doubles = next(g for g in groups if len(g) == 2)
        singles = [g for g in groups if len(g) == 1]
        rng.shuffle(singles)
        assigned = [doubles if i == detour else singles.pop() for i in range(n)]
    else:
        assigned = [groups[i] for i in order]
    last_col = max(c for group in groups for c in group)
    tail_extra = int(rng.integers(min(2, spare) + 1)); spare -= tail_extra
    right_base = last_col + 3 + tail_extra
    offset_budget = int(rng.integers(min(1, spare) + 1)); spare -= offset_budget
    right_offsets = list(map(int, rng.integers(offset_budget + 1, size=n)))
    right = [(r - h // 2, right_base + off, r + h // 2, right_base + off + w - 1)
             for r, h, w, off in zip(rows_b, heights_b, widths_b, right_offsets)]
    width = max(rect[3] for rect in right) + 2 + int(rng.integers(min(2, spare) + 1))
    assert width <= 30
    paths = []
    for i, j in enumerate(target):
        start = (rows_a[i], left[i][3] + 1)
        end = (rows_b[j], right[j][1] - 1)
        group = assigned[i]
        if len(group) == 1:
            waypoints = [start, (start[0], group[0]), (end[0], group[0]), end]
        else:
            mid = int(rng.choice(free_rows))
            waypoints = [start, (start[0], group[0]), (mid, group[0]),
                         (mid, group[1]), (end[0], group[1]), end]
        path = []
        for a, b in zip(waypoints, waypoints[1:]):
            segment = _segment(a, b)
            path.extend(segment if not path else segment[1:])
        assert len(set(path)) == len(path)
        paths.append(path)
    base = np.full((height, width), background, dtype=np.int8)
    for rect in left + right:
        for p in _outline(rect):
            base[p] = frame
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            assert base[p] in (background, carrier), "wire overlaps a terminal"
            base[p] = carrier
            visits.setdefault(p, set()).add(i)
    for p, owners in visits.items():
        if len(owners) > 1:
            assert len(owners) == 2
            base[p] = crossing
    orientations = list(map(int, rng.integers(4, size=2 * n)))
    colors = list(map(int, rng.choice(palette[4:], 2 * n)))
    if lesson == "distributed":
        colors = list(map(int, rng.choice(palette[4:], 2 * n, replace=False)))
        orientations[0], orientations[n + target[0]] = 0, 1
    elif lesson == "empty":
        small_palette = list(map(int, rng.choice(palette[4:], 2, replace=False)))
        colors = [small_palette[i % 2] for i in range(2 * n)]
        orientations[:3] = [0, 2, 3]
    tiles = [dict(rect=rect, q=q, anchor=_corners(rect)[q], color=color)
             for rect, q, color in zip(left + right, orientations, colors)]
    inp, out = base.copy(), base.copy()
    previous, used_patterns = None, set()
    for i, j in enumerate(target):
        pattern = _pattern(rng, asymmetric=lesson == "distributed", excluded=previous)
        if lesson is not None:
            assert pattern not in used_patterns
        used_patterns.add(pattern)
        previous = pattern
        if lesson == "distributed":
            mode = "overlap"
        elif lesson == "empty" and i < 2:
            mode = "empty_a" if i == 0 else "empty_b"
        elif regime == "damage":
            mode = str(rng.choice(("disjoint", "overlap"))) if len(pattern) > 2 else "disjoint"
        else:
            modes = ("disjoint", "empty_a", "empty_b", "full", "mixed")
            mode = str(rng.choice(modes))
        a, b = _split_pattern(rng, pattern, mode)
        for index, fragment in ((i, a), (n + j, b)):
            tile = tiles[index]
            inp[tile["anchor"]] = out[tile["anchor"]] = tile["color"]
            for p in _positions(tile, fragment):
                inp[p] = tile["color"]
            for p in _positions(tile, pattern):
                out[p] = tile["color"]
    expected_links = {frozenset((path[0], path[-1])) for path in paths}
    if rng.random() < .5:
        inp, out = inp[::-1].copy(), out[::-1].copy()
        expected_links = {frozenset((height - 1 - r, c) for r, c in link)
                          for link in expected_links}
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        expected_links = {frozenset((r, width - 1 - c) for r, c in link)
                          for link in expected_links}
    if transpose:
        inp, out = inp.T.copy(), out.T.copy()
        expected_links = {frozenset((c, r) for r, c in link) for link in expected_links}
    assert np.any(inp != out)
    info = _parse(inp)
    inferred_links = {frozenset((info["tiles"][i]["port"], info["tiles"][j]["port"]))
                      for i, j in info["pairs"]}
    assert inferred_links == expected_links, "visible pairing disagrees with construction paths"
    assert np.array_equal(_predict(info), out), "visible routing disagrees with construction"
    if lesson == "distributed":
        assert any(len(e) == 4 for e in info["graph"].values())
        assert all(not np.array_equal(_predict(info, alt), out) for alt in _ALTERNATIVES)
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must lie in {low}..{high}")
    return int(value)


def _key(grid):
    a = _grid(grid)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.integers(2))
    palette = list(map(int, rng.permutation(10)))
    orientation = int(rng.integers(2))
    orientation_shift = bool(rng.integers(2))
    regime = str(rng.choice(("count", "damage", "routing")))
    lessons = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = None
            if split == "train":
                lesson = ("distributed" if index == lessons[0] else
                          "empty" if index == lessons[1] else None)
                n = 2 if lesson == "distributed" else (3 if lesson == "empty" else int(rng.integers(2, 4)))
            else:
                n = (4 if regime == "count" else
                     int(rng.integers(2, 4)) if regime == "damage" else
                     int(rng.integers(3, 5)))
            local_palette = palette if stable else list(map(int, rng.permutation(10)))
            transpose = orientation ^ int(split == "test" and orientation_shift)
            for _ in range(96):
                try:
                    pair = _build_pair(rng, n, lesson,
                                       regime if split == "test" else "train",
                                       local_palette, transpose)
                    key = _key(pair["input"])
                    assert key not in seen
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(key); puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, distinguished, evidence = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and np.any(inp != out)
            key = _key(inp)
            assert key not in seen; seen.add(key)
            info = _parse(inp)
            assert np.array_equal(_predict(info), out), "input-only repair disagrees"
            allowed = set()
            for i, j in info["pairs"]:
                union = info["tiles"][i]["fragment"] | info["tiles"][j]["fragment"]
                assert union and union <= set(_DOMAIN)
                for k in (i, j):
                    tile = info["tiles"][k]
                    points = _positions(tile, union)
                    assert points <= tile["footprint"]
                    allowed |= points
                    assert inp[tile["anchor"]] == out[tile["anchor"]] == tile["color"]
                    assert all(out[p] == tile["color"] for p in points)
            changed = set(map(tuple, np.argwhere(inp != out)))
            assert changed <= allowed and all(inp[p] == info["background"] for p in changed)
            assert all(inp[p] == out[p] for p in info["field"])
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            if split == "train":
                for alt in _ALTERNATIVES:
                    if not np.array_equal(_predict(info, alt), out):
                        distinguished.add(alt)
                for i, j in info["pairs"]:
                    a, b = info["tiles"][i], info["tiles"][j]
                    if not a["fragment"] or not b["fragment"]:
                        evidence.add("empty")
                    if a["fragment"] - b["fragment"] and b["fragment"] - a["fragment"]:
                        evidence.add("distributed")
                    if a["fragment"] & b["fragment"]:
                        evidence.add("overlap")
                    if a["color"] != b["color"]:
                        evidence.add("different_end_colors")
                if len({t["color"] for t in info["tiles"]}) < len(info["tiles"]):
                    evidence.add("reused_colors")
                if any(len(e) == 4 for e in info["graph"].values()):
                    evidence.add("crossing")
    assert set(_ALTERNATIVES) <= distinguished, "training admits a simpler counterrule"
    assert {"empty", "distributed", "overlap", "different_end_colors", "reused_colors", "crossing"} <= evidence


def _metrics(pair):
    info = _parse(pair["input"])
    templates, overlaps, fragments = [], [], []
    for i, j in info["pairs"]:
        a, b = info["tiles"][i]["fragment"], info["tiles"][j]["fragment"]
        templates.append(tuple(sorted(a | b)))
        overlaps.append(len(a & b)); fragments.extend((len(a), len(b)))
    bends = []
    for route in info["routes"]:
        ds = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:])]
        bends.append(sum(a != b for a, b in zip(ds, ds[1:])))
    return dict(shape=info["grid"].shape, cables=len(info["pairs"]),
                frame_shapes=[(t["rect"][2] - t["rect"][0] + 1,
                               t["rect"][3] - t["rect"][1] + 1) for t in info["tiles"]],
                templates=templates, union_sizes=[len(p) for p in templates],
                fragment_sizes=fragments, overlaps=overlaps, bends=bends,
                orientations=[t["q"] for t in info["tiles"]],
                bank_axis="horizontal" if info["tiles"][0]["side"] % 2 else "vertical",
                crossings=sum(len(e) == 4 for e in info["graph"].values()),
                palette=(info["background"], info["frame"], info["carrier"], info["crossing"]),
                reused_colors=len({t["color"] for t in info["tiles"]}) < len(info["tiles"]))


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded graph, semantic countermodel, count, corruption and diversity checks."""
    signatures, metrics = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solver mutated the input"
                metrics.append(_metrics(pair))
    assert len(signatures) == 42
    assert {m["cables"] for m in metrics} == {2, 3, 4}
    assert {m["bank_axis"] for m in metrics} == {"horizontal", "vertical"}
    assert {q for m in metrics for q in m["orientations"]} == {0, 1, 2, 3}
    assert len({m["shape"] for m in metrics}) >= 30
    assert len({p for m in metrics for p in m["templates"]}) >= 60
    assert {s for m in metrics for s in m["union_sizes"]} >= set(range(2, 10))
    assert {b for m in metrics for b in m["bends"]} >= {2, 4}
    assert any(m["crossings"] == 0 for m in metrics) and max(m["crossings"] for m in metrics) >= 4
    assert len({s for m in metrics for s in m["frame_shapes"]}) >= 6
    assert len({m["palette"][:3] for m in metrics}) >= 30
    assert any(0 in m["fragment_sizes"] for m in metrics)
    assert any(0 in m["overlaps"] for m in metrics) and any(max(m["overlaps"]) > 0 for m in metrics)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _signature(generate_puzzle(77, 6, 3)) == _signature(generate_puzzle(77, 6, 3))
    assert _signature(generate_puzzle(77, 6, 3)) != _signature(generate_puzzle(78, 6, 3))
    # Exhaust every four-cable pairing against the visible graph, using different
    # geometry each time. No sampled permutation is passed to the solver.
    rng = np.random.default_rng(9182)
    palette = list(range(10))
    for target in permutations(range(4)):
        for attempt in range(96):
            try:
                pair = _build_pair(rng, 4, None, "train", palette, bool(attempt % 2), target)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                break
            except AssertionError:
                continue
        else:
            raise AssertionError("four-cable permutation failed")
    # The rule is not a dictionary of generated motifs: a one-cell union is
    # solvable too, even though ordinary sampling starts at two cells.
    small = generate_puzzle(91)["train"][0]["input"].copy()
    small_info = _parse(small)
    for tile in small_info["tiles"]:
        for point in _positions(tile, tile["fragment"]):
            small[point] = small_info["background"]
    expected = small.copy()
    for i, j in small_info["pairs"]:
        pattern = {(1, 2)}
        for point in _positions(small_info["tiles"][i], pattern):
            small[point] = small_info["tiles"][i]["color"]
        for k in (i, j):
            for point in _positions(small_info["tiles"][k], pattern):
                expected[point] = small_info["tiles"][k]["color"]
    assert np.array_equal(solve(small.tolist()), expected)
    puzzle = generate_puzzle(12)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
    out = damaged["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated output corruption accepted")
    # Also reject a broken cable input.
    info = _parse(puzzle["test"][0]["input"])
    broken = puzzle["test"][0]["input"].copy()
    p = next(p for p in sorted(info["field"]) if p not in {t["port"] for t in info["tiles"]})
    broken[p] = info["background"]
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken cable accepted")
    return {"seeds": 42, "pairs": len(metrics), "distinct_puzzles": len(signatures),
            "shapes": len({m["shape"] for m in metrics}),
            "templates": len({p for m in metrics for p in m["templates"]}),
            "four_cable_permutations": 24, "countermodels": list(_ALTERNATIVES)}
