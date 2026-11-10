"""Taut lids: a finite strand seals a communicating chamber system.

No parent generator, example bank, external file, or puzzle state is used.
Only standard-library modules and NumPy are required.
"""
from collections import deque
from itertools import permutations
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid size")
    _require(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=True)


def _regions(cells, directions=DIR4):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        reached, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    reached.add(q)
                    todo.append(q)
        result.append(reached)
    return result


def _holes(barrier, directions=DIR4):
    """The canvas boundary is open: explicit free padding represents outside."""
    p = np.pad(barrier.astype(bool), 1)
    outside = np.zeros(p.shape, bool)
    outside[0, 0] = True
    todo = deque([(0, 0)])
    while todo:
        r, c = todo.popleft()
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            if (0 <= rr < p.shape[0] and 0 <= cc < p.shape[1]
                    and not p[rr, cc] and not outside[rr, cc]):
                outside[rr, cc] = True
                todo.append((rr, cc))
    return (~p & ~outside)[1:-1, 1:-1]


def _roof(wall, length):
    """Digital convex lid, followed by an orthogonal staircase rasterization.

    u is forward from the support endpoint; v is toward the strand side.
    Three bounding tangents (+1, 0, -1) come from actual wall extrema.
    Concave bays are bridged. Every intermediate corner uses the outside
    cell, making the lid four-connected and keeping the wall untouched.
    """
    peak = max(v for u, v in wall)
    intercept = max(u + v + 1 for u, v in wall)

    def height(u):
        return min(u + 1, peak + 1, intercept - u)

    _require(height(0) == 1, "unsupported attachment tangent")
    path = [(0, 1)]
    u = 0
    while len(path) < length:
        a, b = height(u), height(u + 1)
        _require(b - a in (-1, 0, 1), "invalid roof slope")
        if b > a:
            path.extend(((u, b), (u + 1, b)))
        elif b < a:
            path.extend(((u + 1, a), (u + 1, b)))
        else:
            path.append((u + 1, b))
        u += 1
    return path[:length]


def _local(p, end, forward, side):
    r, c = p[0] - end[0], p[1] - end[1]
    return r * forward[0] + c * forward[1], r * side[0] + c * side[1]


def _global(p, end, forward, side):
    u, v = p
    return end[0] + u * forward[0] + v * side[0], end[1] + u * forward[1] + v * side[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _analyze(g):
    """Recover every role from input geometry, including orientation and palette."""
    colors, counts = np.unique(g, return_counts=True)
    _require(len(colors) == 4, "expected background, support, strand, wall")
    order = np.argsort(counts)
    _require(counts[order[-1]] > counts[order[-2]], "background must be uniquely modal")
    bg = int(colors[order[-1]])
    masks = {int(c): set(map(tuple, np.argwhere(g == c).tolist())) for c in colors if c != bg}
    candidates = []
    for sc, tc, wc in permutations(masks):
        support, slack, wall = masks[sc], masks[tc], masks[wc]
        if not 3 <= len(support) <= 10 or len(slack) < 2:
            continue
        if len({r for r, c in support}) == 1:
            f = (0, 1)
        elif len({c for r, c in support}) == 1:
            f = (1, 0)
        else:
            continue
        if len(_regions(support)) != 1 or len(_regions(slack, DIR8)) != 1:
            continue
        for end, forward in ((min(support), (-f[0], -f[1])), (max(support), f)):
            for side in ((forward[1], -forward[0]), (-forward[1], forward[0])):
                attachment = _global((0, 1), end, forward, side)
                if attachment not in slack:
                    continue
                local_slack = {_local(p, end, forward, side) for p in slack}
                local_wall = {_local(p, end, forward, side) for p in wall}
                if any(u > 0 or v < 1 for u, v in local_slack):
                    continue
                if min(u for u, v in local_wall) < 1:
                    continue
                peak = max(v for u, v in local_wall)
                if peak < 0 or (peak > 0 and max(v - u for u, v in local_wall) != 0):
                    continue
                path = [_global(p, end, forward, side) for p in _roof(local_wall, len(slack))]
                if set(path) & (support | wall):
                    continue
                candidates.append(dict(background=bg, support_color=sc, thread_color=tc,
                                       wall_color=wc, support=support, slack=slack, wall=wall,
                                       local_wall=local_wall, local_slack=local_slack,
                                       end=end, forward=forward, side=side, path=path))
    _require(len(candidates) == 1, "ambiguous support/strand/wall assignment")
    return candidates[0]


def _zones(g, a):
    base = g.copy()
    for p in a["slack"]:
        base[p] = a["background"]
    visible = {p for p in a["path"] if _inside(p, g.shape)}
    for p in visible:
        _require(base[p] == a["background"], "stitch collision")
        base[p] = a["thread_color"]
    holes = _holes(base != a["background"])
    return base, visible, holes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count, reconstruct the taut lid, then fill only its enclosed background."""
    g = _grid(input_grid)
    a = _analyze(g)
    out, _, holes = _zones(g, a)
    out[holes] = a["thread_color"]
    return out


def _geometry(rng, count, pitched, links=None, max_span=18):
    """One comb system, with shared windows and independently shaped pockets."""
    widths = list(map(int, rng.integers(1, 4, count)))
    # Each outer pier and internal divider can be one or two columns thick.
    dividers = list(map(int, rng.integers(1, 3, count + 1)))
    while sum(widths) + sum(dividers) > max_span:
        options = [(0, i) for i, w in enumerate(widths) if w > 1]
        options += [(1, i) for i, w in enumerate(dividers) if w > 1]
        k, i = options[int(rng.integers(len(options)))]
        (widths if k == 0 else dividers)[i] -= 1
    near = int(rng.integers(1, 3))
    far = near + sum(widths) + sum(dividers) - 1
    if pitched:
        peak = min(near + int(rng.integers(0, min(2, dividers[0]))), far // 2)
        peak = max(near, peak)
        right_height = int(rng.integers(max(0, 2 * peak - far), peak + 1))
    else:
        peak = right_height = 0
    lo = -int(rng.integers(3, 7))
    top = lambda u: min(u, peak, far + right_height - u)
    wall = {(u, v) for u in range(near, far + 1) for v in range(lo, top(u) + 1)}
    bays, separators = [], []
    cursor = near + dividers[0]
    for i, width in enumerate(widths):
        columns = list(range(cursor, cursor + width))
        bottom = min(-1, lo + int(rng.integers(1, 4)))
        bay = {(u, v) for u in columns for v in range(bottom, top(u) + 1)}
        wall -= bay
        bays.append(bay)
        cursor += width
        if i < count - 1:
            separators.append(list(range(cursor, cursor + dividers[i + 1])))
        cursor += dividers[i + 1]
    if links is None:
        links = list(map(bool, rng.integers(0, 2, max(0, count - 1))))
    for i, linked in enumerate(links):
        if linked:
            # A horizontal window is a real passage, not a logical annotation.
            lower = max(min(v for u, v in bays[i]), min(v for u, v in bays[i + 1]))
            upper = min(top(u) for u in separators[i]) - 1
            v = int(rng.integers(lower, upper + 1))
            wall -= {(u, v) for u in separators[i]}
    # A shallow floor step varies the cavity silhouette without changing its mouth.
    for bay in bays:
        columns = sorted({u for u, v in bay})
        if len(columns) >= 3 and rng.random() < .55:
            u = int(rng.choice(columns[1:-1]))
            v = min(v for t, v in bay if t == u)
            wall.add((u, v))
            bay.remove((u, v))
    # Optional pre-existing sealed pinhole below a thick pocket floor.
    if rng.random() < .35:
        options = [(u, lo + 1) for u in range(near + 1, far)
                   if all((u + du, lo + 1 + dv) in wall for du, dv in DIR8)]
        if options:
            wall.remove(options[int(rng.integers(len(options)))])
    # Exterior lobes are legitimate wall material and do not set the roof tangents.
    if rng.random() < .45:
        u = int(rng.integers(near, far + 1))
        wall.update((u, lo - d) for d in range(1, int(rng.integers(2, 4))))
    _require(max(v for u, v in wall) == peak, "lost peak witness")
    _require(max(u + v for u, v in wall) == far + right_height, "lost departure witness")
    return wall, bays


def _local_holes(wall, path):
    cells = wall | set(path)
    u0, u1 = min(u for u, v in cells) - 1, max(u for u, v in cells) + 1
    v0, v1 = min(v for u, v in cells) - 1, max(v for u, v in cells) + 1
    barrier = np.zeros((u1 - u0 + 1, v1 - v0 + 1), bool)
    for u, v in cells:
        barrier[u - u0, v - v0] = True
    h = _holes(barrier)
    return {(int(u) + u0, int(v) + v0) for u, v in np.argwhere(h)}


def _threshold(wall, bays):
    target = set().union(*bays)
    path = _roof(wall, 60)
    for length in range(2, len(path) + 1):
        if target <= _local_holes(wall, path[:length]):
            return length
    raise AssertionError("the lid never seals the mouths")


def _slack(rng, length, depth, width, diagonal):
    cells = {(0, 1)}
    low_v = 2 if diagonal else 1
    cells.add((-1, low_v))
    style = int(rng.integers(0, 3))
    while len(cells) < length:
        dirs = DIR8 if style == 2 else DIR4
        front = sorted({(u + du, v + dv) for u, v in cells if u < 0 for du, dv in dirs
                        if -depth <= u + du <= -1 and low_v <= v + dv <= width} - cells)
        _require(front, "slack capacity exhausted")
        neighbor_counts = np.array([sum((u + du, v + dv) in cells for du, dv in DIR8)
                                    for u, v in front], dtype=float)
        weights = (neighbor_counts + 1) ** (2 if style == 0 else -1 if style == 2 else 0)
        cells.add(front[int(rng.choice(len(front), p=weights / weights.sum()))])
    return cells


def _render_group(rng, wall, lengths, palette, orientation, clip=False, diagonal=None):
    """Shared canvas/support for a controlled length contrast, or a single scene."""
    largest = max(lengths)
    width = int(rng.integers(max(5, (largest - 1 + 7) // 8 + 1), 9))
    depth = max(2, (largest - 1 + width - 2) // (width - 1))
    _require(depth <= 8, "slack exceeds bounded support")
    support = {(-u, 0) for u in range(depth + 2)}
    slacks = [_slack(rng, n, depth, width, bool(rng.integers(2)) if diagonal is None else diagonal)
              for n in lengths]
    paths = [_roof(wall, n) for n in lengths]
    all_input = wall | support | set().union(*slacks)
    all_cells = all_input | set().union(*(set(p) for p in paths))
    far = max(u for u, v in wall)
    u0, u1 = min(u for u, v in all_cells), max(u for u, v in all_cells)
    v0, v1 = min(v for u, v in all_cells), max(v for u, v in all_cells)
    if clip:
        u1 = max(far + 1, max(u for u, v in all_input))
        all_cells = {p for p in all_cells if p[0] <= u1}
        v0, v1 = min(v for u, v in all_cells), max(v for u, v in all_cells)
    # When a long tail would exceed 30, clipping is the ordinary finite-canvas rule.
    if u1 - u0 + 3 > 30:
        u1 = max(far + 1, max(u for u, v in all_input))
    u1 = min(u1, u0 + 28)
    _require(u1 >= max(u for u, v in all_input), "cropping would remove input material")
    visible_cells = {p for p in all_cells if p[0] <= u1}
    v0, v1 = min(v for u, v in visible_cells), max(v for u, v in visible_cells)
    left, right = int(rng.integers(1, 4)), 0 if clip else int(rng.integers(1, 4))
    if u1 - u0 + 1 + left + right > 30:
        left, right = 1, max(0, 30 - (u1 - u0 + 2))
    bottom, upper = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if v1 - v0 + 1 + bottom + upper > 30:
        bottom = upper = 1
    height = v1 - v0 + 1 + bottom + upper
    canvas_width = u1 - u0 + 1 + left + right
    _require(max(height, canvas_width) <= 30, "canvas exceeds ARC bound")
    bg, sc, tc, wc = palette

    def put(g, cells, color):
        for u, v in cells:
            r, c = upper + v1 - v, left + u - u0
            if _inside((r, c), g.shape):
                g[r, c] = color

    pairs = []
    for slack, path in zip(slacks, paths):
        g = np.full((height, canvas_width), bg, np.int8)
        put(g, support, sc)
        put(g, wall, wc)
        put(g, slack, tc)
        out = np.full_like(g, bg)
        put(out, support, sc)
        put(out, wall, wc)
        put(out, path, tc)
        out[_holes(out != bg)] = tc
        g, out = np.rot90(g, orientation[0]), np.rot90(out, orientation[0])
        if orientation[1]:
            g, out = np.fliplr(g), np.fliplr(out)
        pairs.append({"input": g.copy(), "output": out.copy()})
    return pairs


def _key(g):
    return tuple(g.shape), np.asarray(g, np.int8).tobytes()


def _mouth_info(a, visible, holes):
    """Reconstruct communication groups and locally covered mouths from pixels."""
    wall = a["local_wall"]
    peak = max(v for u, v in wall)
    intercept = max(u + v + 1 for u, v in wall)
    near, far = min(u for u, v in wall), max(u for u, v in wall)
    top = lambda u: min(u, peak, intercept - 1 - u)
    gaps = {u for u in range(near, far + 1) if (u, top(u)) not in wall}
    mouths = [sorted(u for u, v in component) for component in _regions({(u, 0) for u in gaps})]
    bottom = min(v for u, v in wall)
    free = {(u, v) for u in range(near, far + 1) for v in range(bottom, top(u) + 1)} - wall
    points = [(columns[0], top(columns[0])) for columns in mouths]
    groups = sorted(sum(p in component for p in points) for component in _regions(free)
                    if any(p in component for p in points))
    covered_unfilled = 0
    for columns, point in zip(mouths, points):
        u = columns[-1] + 1
        contact = _global((u, top(u) + 1), a["end"], a["forward"], a["side"])
        inside_point = _global(point, a["end"], a["forward"], a["side"])
        if contact in visible and not holes[inside_point]:
            covered_unfilled += 1
    return len(mouths), groups, covered_unfilled


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(10)[:4]))
    variable_palette = bool(rng.integers(2))
    mixed_roofs = bool(rng.random() < .8)
    regime = int(rng.integers(3))  # count, orientation, communicating-pocket recombination
    orientation = (int(rng.integers(4)), bool(rng.integers(2)))
    wall, bays = _geometry(rng, 3, mixed_roofs, [False, True], max_span=15)
    threshold = _threshold(wall, bays)
    anchors = _render_group(rng, wall, [threshold - 1, threshold + 6], palette,
                            orientation, clip=bool(rng.integers(2)), diagonal=True)
    puzzle = {"train": anchors, "test": []}
    seen = {_key(p["input"]) for p in anchors}
    for split, number in (("train", nt - 2), ("test", nq)):
        for index in range(number):
            for attempt in range(80):
                if split == "test" and regime == 0:
                    count = 4
                elif split == "test" and regime == 2:
                    count = 3
                else:
                    count = int(rng.integers(1, 4))
                links = None
                if split == "test" and regime == 2:
                    links = [True, True] if index % 2 == 0 else [True, False]
                wall, bays = _geometry(rng, count, mixed_roofs and bool(rng.integers(2)), links)
                threshold = _threshold(wall, bays)
                # Cover early prefixes, sealing thresholds, and outgoing tails.
                stage = int(rng.integers(4))
                length = (int(rng.integers(2, max(3, threshold // 2))) if stage == 0 else
                          max(2, threshold - int(rng.integers(1, 5))) if stage == 1 else
                          threshold if stage == 2 else threshold + int(rng.integers(1, 7)))
                length = min(length, 40)
                colors = tuple(map(int, rng.permutation(10)[:4])) if variable_palette else palette
                rotation = int(rng.integers(4))
                if regime == 1:
                    rotation = (orientation[0] % 2) + 2 * int(rng.integers(2))
                    if split == "test":
                        rotation = (rotation + 1) % 4
                pair = _render_group(rng, wall, [length], colors, (rotation, bool(rng.integers(2))),
                                     clip=bool(rng.integers(2)))[0]
                key = _key(pair["input"])
                # Recover roles before accepting; modal-background validity is pixel based.
                try:
                    a = _analyze(pair["input"])
                    base, _, holes = _zones(pair["input"], a)
                    _require(np.array_equal(holes, _holes(base != a["background"], DIR8)), "diagonal leak")
                except AssertionError:
                    continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded unique-scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    g = _grid(pair["input"])
    a = _analyze(g)
    base, visible, holes = _zones(g, a)
    local_wall = a["local_wall"]
    peak = max(v for u, v in local_wall)
    near, far = min(u for u, v in local_wall), max(u for u, v in local_wall)
    mouths, groups, covered_unfilled = _mouth_info(a, visible, holes)
    original_barrier = (g == a["support_color"]) | (g == a["wall_color"])
    original_holes = _holes(original_barrier)
    return dict(shape=tuple(g.shape), mouths=mouths, mouth_groups=tuple(groups),
                covered_unfilled=covered_unfilled, length=len(a["slack"]), peak=peak,
                forward=tuple(a["forward"]), filled=int(holes.sum()),
                newly_filled=int((holes & ~original_holes).sum()),
                cavity_regions=len(_regions(map(tuple, np.argwhere(holes).tolist()))),
                initial_holes=int(original_holes.sum()), clipped=len(a["path"]) - len(visible),
                diagonal_slack=len(_regions(a["slack"])) > 1,
                wall_components=len(_regions(a["wall"])),
                silhouette=tuple(sorted((u - near, v) for u, v in local_wall)),
                palette=tuple(a[k] for k in ("background", "support_color", "thread_color", "wall_color")))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape and not np.array_equal(g, out), "unchanged scene or size")
            _require(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            a = _analyze(g)
            base, visible, holes = _zones(g, a)
            _require(len(set(a["path"])) == len(a["slack"]), "length is not conserved conceptually")
            _require(all(abs(r - rr) + abs(c - cc) == 1
                         for (r, c), (rr, cc) in zip(a["path"], a["path"][1:])), "nonorthogonal lid")
            _require(_global((0, 1), a["end"], a["forward"], a["side"]) in visible, "lost anchor")
            _require(all(out[p] == a["support_color"] for p in a["support"]), "support changed")
            _require(all(out[p] == a["wall_color"] for p in a["wall"]), "wall changed")
            _require(np.array_equal(holes, _holes(base != a["background"], DIR8)), "ambiguous diagonal enclosure")
            _require(bool(np.all(out[holes] == a["thread_color"])), "unfilled cavity")
            _require(bool(np.all(out[~holes] == base[~holes])), "changed exterior or incorrect lid")
            _require(np.array_equal(out, solve(g)), "input-only solve mismatch")
            features[split].append(_features(pair))
    first, second = puzzle["train"][:2]
    a, b = _analyze(_grid(first["input"])), _analyze(_grid(second["input"]))
    _require(np.asarray(first["input"]).shape == np.asarray(second["input"]).shape, "contrast canvas changed")
    _require(a["wall"] == b["wall"] and a["support"] == b["support"]
             and a["thread_color"] == b["thread_color"] and a["background"] == b["background"],
             "missing controlled length contrast")
    fa, fb = features["train"][:2]
    _require(fa["mouths"] == fb["mouths"] == 3, "contrast needs three chamber mouths")
    _require(fa["mouth_groups"] == fb["mouth_groups"] == (1, 2), "missing actual inter-chamber window")
    _require(fa["covered_unfilled"] >= 1, "locally roofed chamber must retain its exterior connection")
    _require(fa["newly_filled"] > 0 and fb["filled"] > fa["filled"], "missing sealing threshold evidence")
    _require(fa["cavity_regions"] >= 1 and fb["cavity_regions"] > fa["cavity_regions"], "missing communicating chambers")
    # A one-cell extension of the short demonstration must trigger a large fill.
    longer = dict(a, path=[_global(p, a["end"], a["forward"], a["side"])
                           for p in _roof(a["local_wall"], len(a["slack"]) + 1)])
    _, _, more_holes = _zones(_grid(first["input"]), longer)
    _, _, short_holes = _zones(_grid(first["input"]), a)
    _require(int(more_holes.sum()) > int(short_holes.sum()) + 1, "count shortcut not contradicted")
    _require(any(f["newly_filled"] for f in features["train"]), "omitting A leaves fillings unchanged")
    _require(any(f["filled"] for f in features["train"]), "omitting B leaves answer unchanged")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
                                            for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded, pixel-derived 42-seed diversity, regressions, and corruption checks."""
    _require(_roof({(1, 0), (3, 0)}, 8) ==
             [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 0), (5, 0), (5, -1)],
             "flat departure/corner budget regression")
    _require(_roof({(1, 1), (2, 2), (6, 2), (7, 1)}, 13) ==
             [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3),
              (5, 3), (6, 3), (7, 3), (7, 2), (8, 2), (8, 1)],
             "pitched bridge/corner budget regression")
    # Three hand-built bays: the latter two communicate through a low window.
    wall = {(u, -2) for u in range(1, 10)}
    wall |= {(u, v) for u in (1, 4, 7, 9) for v in (-2, -1, 0)}
    wall.remove((7, -1))
    first_room = {(u, v) for u in (2, 3) for v in (-1, 0)}
    _require(_local_holes(wall, _roof(wall, 8)) == first_room, "covered room leaked through neighbor")
    _require(len(_local_holes(wall, _roof(wall, 10))) == 11, "one-cell sealing threshold")
    _require(len(_local_holes(wall | {(7, -1)}, _roof(wall, 8))) == 8, "window changes fill dependency")
    edge_bay = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], bool)
    _require(not _holes(edge_bay).any(), "canvas edge must not act as an extra wall")
    records, structures, silhouettes, shapes, palettes = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                records.append(f)
                shapes.add(f["shape"])
                silhouettes.add(f["silhouette"])
                palettes.add(f["palette"])
                g = pair["input"]
                # Normalize semantic roles, not literal digits.
                a = _analyze(g)
                norm = np.zeros_like(g)
                for i, role in enumerate(("support", "slack", "wall"), 1):
                    for p in a[role]:
                        norm[p] = i
                structures.add(_key(norm))
                before = g.copy()
                _require(np.array_equal(solve(g), pair["output"]) and np.array_equal(g, before), "solve mutation/mismatch")
    _require(len(shapes) >= 40 and len(silhouettes) >= 100 and len(structures) >= 160, "insufficient geometric variation")
    _require(len(palettes) >= 40, "insufficient palette variation")
    _require({f["mouths"] for f in records} >= {1, 2, 3, 4}, "chamber count variation")
    _require({f["forward"] for f in records} == {(-1, 0), (1, 0), (0, -1), (0, 1)}, "direction variation")
    for predicate in (lambda f: f["clipped"] > 0, lambda f: f["filled"] == 0,
                      lambda f: f["initial_holes"] > 0, lambda f: f["wall_components"] > 1,
                      lambda f: f["peak"] == 0, lambda f: f["peak"] > 0,
                      lambda f: not f["diagonal_slack"]):
        _require(any(predicate(f) for f in records), "missing structural feature")
    _require(_same(generate_puzzle(0), generate_puzzle(0)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt each meaningful zone of a returned demonstration, keeping the input fixed.
    original = generate_puzzle(0, 2, 1)
    g = original["train"][1]["input"]
    a = _analyze(g)
    base, visible, holes = _zones(g, a)
    far = np.argwhere((base == a["background"]) & ~holes)
    locations = [min(a["support"]), min(a["wall"]), min(visible),
                 tuple(np.argwhere(holes)[0]), tuple(far[0])]
    for location in locations:
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        out = bad["train"][1]["output"]
        out[location] = (int(out[location]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted pixel corruption")
    return {"seeds": 42, "pairs": len(records), "grid_shapes": len(shapes),
            "wall_silhouettes": len(silhouettes), "role_normalized_inputs": len(structures),
            "palettes": len(palettes), "length_range": [min(f["length"] for f in records), max(f["length"] for f in records)]}


if __name__ == "__main__":
    print(self_test())
