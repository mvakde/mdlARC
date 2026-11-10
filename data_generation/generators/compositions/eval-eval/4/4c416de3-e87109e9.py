"""Latent corner mirrors: infer occupancy, route light, repair reached copies.

Only NumPy and the standard library are used. The solver reconstructs all
geometry and instructions from its argument. No parent modules are imported.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_HOST_SEEDS = (0, 290910567, 38045999, 914124613, 452236289, 546847420,
               342427786, 216410138, 753066130, 919361497, 832074619,
               934316731, 497239412, 894564447, 207263035, 916534184,
               439006434, 856604230, 74801975, 856983262, 491817957,
               373004266, 425239657, 250808576, 280778910, 498483298,
               791612847, 643080914, 173147929, 890076021, 772772458,
               981770280, 148209072, 639348453, 249671133, 406264157,
               147240485, 636517023, 604297276, 543358481, 603677584,
               493881281)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((a >= 0) & (a <= 9)), "bad color")
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _connected(points):
    if not points:
        return False
    seen = {min(points)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = (u + du, v + dv)
                if p in points and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return seen == set(points)


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _positions(corner, pattern):
    y, x, q = corner
    dy, dx = _SIGNS[q]
    return frozenset((y + dy * u, x + dx * v) for u, v in pattern)


def _outline(rect):
    t, l, b, r = rect
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(a, l) for a in range(t, b + 1)}
            | {(a, r) for a in range(t, b + 1)})


def _associate(corners, y, x):
    candidates = []
    for corner in corners:
        a, b, q = corner
        dy, dx = _SIGNS[q]
        u, v = (y - a) * dy, (x - b) * dx
        if 1 <= u <= 4 and 1 <= v <= 4:
            candidates.append((u * u + v * v, corner, (u, v)))
    _require(bool(candidates), "clue has no corner")
    distance = min(p[0] for p in candidates)
    nearest = [p for p in candidates if p[0] == distance]
    _require(len(nearest) == 1, "ambiguous corner clue")
    return nearest[0][1:]


def _parse_header(grid):
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing key top")
    ends = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(ends) > 0, "missing key bottom")
    bottom = int(ends[0])
    _require(2 <= bottom <= 4 and bottom + 1 < len(grid), "bad key height")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "inconsistent key bars")
    row = band[0]
    _require(row[0] == row[-1] == frame, "missing key walls")
    runs = []
    x = 0
    while x < len(row):
        if row[x] == frame:
            x += 1
            continue
        start = x
        while x < len(row) and row[x] != frame:
            x += 1
        runs.append((start, x))
    _require(2 <= len(runs) <= 5 and all(b - a >= 3 for a, b in runs), "bad panels")
    background = int(row[runs[0][0] + 1])
    _require(background != frame, "key colors alias")
    turns = {}
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == background), "bad panel interior")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == background) != (right == background), "ambiguous side bar")
        color, turn = (right, 1) if left == background else (left, -1)
        _require(color not in turns and color not in (frame, background), "duplicate key")
        turns[color] = turn
    return bottom + 1, frame, background, turns


def _frames(field, frame):
    h, w = field.shape
    seen, result = set(), []
    for yy, xx in np.argwhere(field == frame):
        start = (int(yy), int(xx))
        if start in seen:
            continue
        component, todo = {start}, [start]
        seen.add(start)
        while todo:
            y, x = todo.pop()
            for dy, dx in _DIRS:
                p = (y + dy, x + dx)
                if (0 <= p[0] < h and 0 <= p[1] < w
                        and p not in seen and field[p] == frame):
                    seen.add(p)
                    component.add(p)
                    todo.append(p)
        ys, xs = zip(*component)
        rect = (min(ys), min(xs), max(ys), max(xs))
        _require(rect[2] - rect[0] >= 6 and rect[3] - rect[1] >= 6, "small frame")
        _require(component == _outline(rect), "frame is not a closed rectangle")
        result.append(rect)
    _require(1 <= len(result) <= 4, "bad frame count")
    return sorted(result)


def _read(input_grid, rotation=False):
    inp = _grid(input_grid)
    crop, frame, background, turns = _parse_header(inp)
    field = inp[crop:]
    others = set(map(int, np.unique(field))) - {frame, background} - set(turns)
    _require(len(others) == 1, "source color is not unique")
    source = others.pop()
    coords = np.argwhere(field == source)
    _require(len(coords) == 1, "source must be one cell")
    source_pos = tuple(map(int, coords[0]))
    counts = np.bincount(field.ravel(), minlength=10)
    _require(counts[background] > max(counts[c] for c in range(10) if c != background),
             "background must uniquely dominate the field")
    rects = _frames(field, frame)
    _require(all(_separated(a, b) for i, a in enumerate(rects) for b in rects[:i]),
             "guide boxes are too close")
    corners = [c for rect in rects for c in _corners(rect)]
    assigned = {c: [] for c in corners}
    for yy, xx in np.argwhere(np.isin(field, list(turns))):
        y, x = int(yy), int(xx)
        corner, point = _associate(corners, y, x)
        assigned[corner].append((point, int(field[y, x])))
    stamps = []
    for corner, cells in assigned.items():
        if not cells:
            continue
        colors = {c for _, c in cells}
        _require(len(colors) == 1, "two colors on one corner")
        clue = frozenset(p for p, _ in cells)
        if rotation and corner[2] in (1, 2):
            clue = frozenset((v, u) for u, v in clue)
        stamps.append(dict(corner=corner, color=colors.pop(), clue=clue))
    _require(len(stamps) >= 3, "too few marked corners")
    candidates = {s["clue"] for s in stamps
                  if all(other["clue"] <= s["clue"] for other in stamps)}
    _require(len(candidates) == 1, "no unique intact containing stencil")
    pattern = candidates.pop()
    _require(3 <= len(pattern) <= 12 and _connected(pattern), "bad stencil")
    owners = set()
    for stamp in stamps:
        corner = stamp["corner"]
        local = (frozenset((v, u) for u, v in pattern)
                 if rotation and corner[2] in (1, 2) else pattern)
        points = _positions(corner, local)
        _require(all(0 <= y < field.shape[0] and 0 <= x < field.shape[1]
                     for y, x in points), "clipped footprint")
        # Even missing pixels must belong unambiguously to this same corner.
        for y, x in points:
            near, _ = _associate(corners, y, x)
            _require(near == corner, "ambiguous completed footprint")
        _require(not points & owners and source_pos not in points, "footprints overlap")
        _require(all(int(field[p]) in (background, stamp["color"]) for p in points),
                 "footprint covers a guide or another object")
        owners.update(points)
        stamp["points"] = points
        stamp["visible"] = frozenset(p for p in points if field[p] == stamp["color"])
    return dict(field=field, crop=crop, frame=frame, background=background,
                source=source, source_pos=source_pos, frames=rects,
                pattern=pattern, stamps=stamps, turns=turns)


def _trace(info, visible_only=False, turn_map=None, boxes=False):
    """Freeze virtual occupancy, then trace four independent cell states."""
    field, stamps = info["field"], info["stamps"]
    h, w = field.shape
    owner = np.full((h, w), -1, dtype=np.int16)
    for i, stamp in enumerate(stamps):
        points = stamp["visible"] if visible_only else stamp["points"]
        if boxes:
            ys, xs = zip(*stamp["points"])
            points = {(y, x) for y in range(min(ys), max(ys) + 1)
                      for x in range(min(xs), max(xs) + 1)}
        for p in points:
            _require(owner[p] == -1, "overlapping optical obstacles")
            owner[p] = i
    paths = np.zeros((4, h, w), dtype=bool)
    visited, events, chains = set(), [], []
    for branch in range(4):
        y, x = info["source_pos"]
        d, seen, chain, just_turned = branch, set(), [], False
        paths[branch, y, x] = True
        while True:
            state = (y, x, d)
            _require(state not in seen, "cyclic ray")
            seen.add(state)
            dy, dx = _DIRS[d]
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w):
                break
            i = int(owner[yy, xx])
            if i >= 0:
                _require(not just_turned, "no clear cell after a turn")
                side = info["turns"][stamps[i]["color"]]
                if turn_map is not None:
                    side = turn_map[side]
                events.append((branch, i, d, side, (yy, xx),
                               (yy, xx) not in stamps[i]["visible"]))
                visited.add(i)
                chain.append(i)
                d = (d + side) % 4
                just_turned = True
            else:
                y, x = yy, xx
                paths[branch, y, x] = True
                just_turned = False
        chains.append(tuple(chain))
    return dict(paths=paths, visited=visited, events=events, chains=tuple(chains))


def _render(info, trace, complete_all=False):
    out = info["field"].copy()
    out[np.any(trace["paths"], axis=0) & (out == info["background"])] = info["source"]
    visited = range(len(info["stamps"])) if complete_all else trace["visited"]
    for i in visited:
        stamp = info["stamps"][i]
        for p in stamp["points"]:
            out[p] = stamp["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the cropped answer from the visible key, guides and clues only."""
    info = _read(input_grid)
    return _render(info, _trace(info))


def _ablation_differs(info, trace, *, visible=False, boxes=False):
    try:
        altered = _trace(info, visible_only=visible, boxes=boxes)
    except AssertionError:
        return True
    return not np.array_equal(_render(info, altered), _render(info, trace))


def _pattern(rng, radius, teaching=False):
    if teaching:
        # A stepped footprint exposes a bounding-box counterexample as well
        # as asymmetry. The added cells and footprint vary, not a bank.
        domain = {(u, v) for u in range(1, radius + 1) for v in range(1, radius + 1)
                  if u != 1 or v >= 2}
        for _ in range(100):
            size = int(rng.integers(max(4, radius + 1), min(12, len(domain)) + 1))
            chosen = {(1, 2), (radius, 1)}
            extras = sorted(domain - chosen)
            rng.shuffle(extras)
            chosen.update(extras[:size - 2])
            if (_connected(chosen)
                    and chosen != {(v, u) for u, v in chosen}
                    and max(v for u, v in chosen if u == radius) < max(v for _, v in chosen)):
                return frozenset(chosen)
        # Valid procedural boundary case, used only if random subsets fail.
        return frozenset({(1, 2), (1, 3)} | {(u, 1) for u in range(2, radius + 1)}
                         | {(radius - 1, 2)})
    domain = {(u, v) for u in range(1, radius + 1) for v in range(1, radius + 1)}
    size = int(rng.integers(3, min(12, radius * radius) + 1))
    chosen = {(int(rng.integers(1, radius + 1)), int(rng.integers(1, radius + 1)))}
    while len(chosen) < size:
        frontier = sorted(p for p in domain - chosen
                          if any(max(abs(p[0] - a), abs(p[1] - b)) <= 1
                                 for a, b in chosen))
        chosen.add(frontier[int(rng.integers(len(frontier)))])
    # Scale-shift scenes must actually expose the requested outer coordinate.
    if radius == 4 and max(max(p) for p in chosen) < 4:
        # Extend outward through adjacent cells instead of an unbounded retry.
        edge = max(chosen)
        while max(max(p) for p in chosen) < 4:
            edge = (edge[0] + 1, edge[1])
            chosen.add(edge)
    return frozenset(chosen)


def _header(rng, width, height, turns):
    colors = list(turns)
    rng.shuffle(colors)
    widths = np.full(len(colors), 5, dtype=int)
    widths += rng.multinomial(width - 5 * len(colors), [1 / len(colors)] * len(colors))
    header = np.zeros((height, width), dtype=np.int8)
    header[[0, -1], :] = 1
    x = 0
    for color, size in zip(colors, widths):
        header[:, x] = header[:, x + size - 1] = 1
        header[1:-1, x + (1 if turns[color] == -1 else size - 2)] = color
        x += int(size)
    return header


def _finish(rng, info, palette, header_height, rotation=0, reflected=False):
    # Construction ground truth uses explicit sampled geometry. The public
    # solver subsequently rediscovers that geometry from the returned input.
    trace = _trace(info)
    field = info["field"]
    output = _render(info, trace)
    turns = info["turns"].copy()
    if reflected:
        field, output = field[:, ::-1], output[:, ::-1]
        turns = {color: -side for color, side in turns.items()}
    field = np.rot90(field, rotation).copy()
    output = np.rot90(output, rotation).copy()
    inp = np.vstack((_header(rng, field.shape[1], header_height, turns), field))
    return {"input": palette[inp].astype(np.int8), "output": palette[output].astype(np.int8)}


def _construction(field, frames, stamps, pattern, turns, source_pos):
    field[source_pos] = 2
    for stamp in stamps:
        stamp["points"] = _positions(stamp["corner"], pattern)
        stamp["visible"] = _positions(stamp["corner"], stamp["clue"])
        for p in stamp["visible"]:
            field[p] = stamp["color"]
    return dict(field=field, frames=frames, stamps=stamps, pattern=pattern,
                turns=turns, source=2, source_pos=source_pos, background=0, frame=1)


def _teach(rng, radius, secondary, rotation, reflected, palette, excluded_pattern=None):
    # Both lessons use separated freely sized frames. The lower BR is struck
    # along its highest occupied row, so turning north leaves its own mask.
    # The upper BR is struck along its lowest occupied row, so turning east
    # leaves that mask. Adjacent intact TR/BR copies fix axis reflection.
    for _ in range(100):
        pattern = _pattern(rng, radius, teaching=True)
        canonical = (frozenset((v, u) for u, v in pattern) if rotation % 2 else pattern)
        if canonical != excluded_pattern:
            break
    else:
        raise AssertionError('distinct teaching stencil sampling exhausted')
    for _ in range(100):
        top = int(rng.integers(0, 2))
        upper_h = int(rng.integers(9, 11))
        lower_h = int(rng.integers(9, 11))
        lower_t = top + upper_h + int(rng.integers(2, 5))
        h = lower_t + lower_h + int(rng.integers(1, 3))
        if h > 27:
            continue
        lower_l = int(rng.integers(0, 3))
        lower_w = int(rng.integers(15, 21))
        lower = (lower_t, lower_l, lower_t + lower_h - 1, lower_l + lower_w - 1)
        bottom_u = max(u for u, _ in pattern)
        contact_v = max(v for u, v in pattern if u == bottom_u)
        ray_x = lower[3] - contact_v - 1
        entry_v = int(rng.choice([v for u, v in sorted(pattern) if u == 1]))
        upper_r = ray_x + entry_v
        upper_w = int(rng.integers(9, min(13, upper_r + 2)))
        upper = (top, upper_r - upper_w + 1, top + upper_h - 1, upper_r)
        w = int(rng.integers(max(22, lower[3] + 2, upper_r + 2), 28))
        frames = [upper, lower]
        if upper[1] < 0:
            continue
        key_count = int(rng.integers(2, min(5, min(h, w) // 5) + 1))
        turns = {c: int(rng.choice((-1, 1))) for c in range(3, 3 + key_count)}
        turns[3], turns[4] = -1, 1
        uc, lc = _corners(upper), _corners(lower)
        first_contact = (bottom_u, contact_v)
        remaining = sorted(pattern - {first_contact})
        rng.shuffle(remaining)
        n = int(rng.integers(2, len(remaining) + 1))
        hidden_clue = frozenset(remaining[:n])
        other = sorted(pattern)
        rng.shuffle(other)
        other_clue = frozenset(other[:int(rng.integers(2, len(pattern)))])
        stamps = [dict(corner=lc[3], color=3, clue=hidden_clue),
                  dict(corner=uc[3], color=4, clue=pattern),
                  dict(corner=uc[1], color=int(rng.choice(list(turns))), clue=pattern),
                  dict(corner=uc[0], color=int(rng.choice(list(turns))), clue=other_clue),
                  dict(corner=lc[0], color=int(rng.choice(list(turns))), clue=pattern)]
        # Change the optional lower reference's damage in the second lesson.
        if secondary:
            stamps[-1]['clue'] = other_clue
        upper_columns = {x for s in stamps[:4] if s['corner'] in uc
                         for _, x in _positions(s['corner'], pattern)}
        options = [x for x in range(lower_l + radius + 1, ray_x - 1)
                   if x not in upper_columns]
        if not options:
            continue
        source_pos = (lower[2] - bottom_u, int(rng.choice(options)))
        field = np.zeros((h, w), dtype=np.int8)
        for rect in frames:
            for p in _outline(rect):
                field[p] = 1
        info = _construction(field, frames, stamps, pattern, turns, source_pos)
        trace = _trace(info)
        _require(any(len(set(c)) >= 2 for c in trace['chains']), 'teaching chain lost')
        _require(_ablation_differs(info, trace, visible=True), 'teaching lacks latent impact')
        _require(_ablation_differs(info, trace, boxes=True), 'teaching lacks shape-sensitive routing')
        return _finish(rng, info, palette, 3, rotation, reflected), canonical
    raise AssertionError('teaching geometry sampling exhausted')


def _separated(a, b):
    return a[2] + 2 < b[0] or b[2] + 2 < a[0] or a[3] + 2 < b[1] or b[3] + 2 < a[1]


def _pack(rng, h, w, radius, count):
    minimum = 2 * radius + 3
    frames = []
    for _ in range(count):
        for attempt in range(160):
            maximum_h = min(h - 2, minimum + (2 if count >= 3 else 9))
            maximum_w = min(w - 2, minimum + (2 if count >= 3 else 9))
            # Later trials favor small frames; positions remain free.
            fh = minimum if attempt > 80 else int(rng.integers(minimum, maximum_h + 1))
            fw = minimum if attempt > 80 else int(rng.integers(minimum, maximum_w + 1))
            t, l = int(rng.integers(0, h - fh + 1)), int(rng.integers(0, w - fw + 1))
            rect = (t, l, t + fh - 1, l + fw - 1)
            if all(_separated(rect, other) for other in frames):
                frames.append(rect)
                break
        else:
            return None
    return frames


def _random_pair(rng, mode, split, palette):
    is_test = split == "test"
    for attempt in range(240):
        if mode == "count" and is_test:
            count = int(rng.integers(3, 5))
            radius = int(rng.choice((2, 2, 3)))
        else:
            count = int(rng.integers(1, 3))
            radius = 4 if mode == "radius" and is_test else int(rng.choice((2, 3)))
        pattern = _pattern(rng, radius)
        if mode == "sparse" and not is_test and len(pattern) < 4:
            continue
        minimum = (2 * (2 * radius + 3) + 3 if count >= 3 else 2 * radius + 7)
        hh = int(rng.integers(3, 6))
        h = int(rng.integers(max(14, minimum), 31 - hh))
        w = int(rng.integers(max(15, minimum), 31))
        frames = _pack(rng, h, w, radius, count)
        if frames is None:
            continue
        corners = [c for rect in frames for c in _corners(rect)]
        marked = [corners[int(i)] for i in rng.permutation(len(corners))
                  [:int(rng.integers(3, len(corners) + 1))]]
        key_count = int(rng.integers(2, min(5, w // 5) + 1))
        turns = {c: int(rng.choice((-1, 1))) for c in range(3, 3 + key_count)}
        stamps = []
        for i, corner in enumerate(marked):
            if i == 0 or (i >= 3 and rng.random() < .18):
                clue = pattern
            else:
                points = sorted(pattern)
                rng.shuffle(points)
                lower = 2 if mode == "sparse" and not is_test else 1
                n = 1 if mode == "sparse" and is_test else int(rng.integers(lower, len(pattern)))
                clue = frozenset(points[:n])
            stamps.append(dict(corner=corner, color=int(rng.choice(list(turns))), clue=clue,
                               points=_positions(corner, pattern), visible=_positions(corner, clue)))
        field = np.zeros((h, w), dtype=np.int8)
        for rect in frames:
            for p in _outline(rect):
                field[p] = 1
        for stamp in stamps:
            for p in stamp["visible"]:
                field[p] = stamp["color"]
        all_points = set().union(*(s["points"] for s in stamps))
        rows = {y for y, _ in all_points}
        cols = {x for _, x in all_points}
        options = [(y, x) for y in range(h) for x in range(w)
                   if (y in rows or x in cols) and field[y, x] == 0 and (y, x) not in all_points]
        rng.shuffle(options)
        for source_pos in options[:80]:
            info = dict(field=field, frames=frames, pattern=pattern, stamps=stamps,
                        turns=turns, source=2, source_pos=source_pos, background=0, frame=1)
            try:
                trace = _trace(info)
            except AssertionError:
                continue
            partial = {i for i, s in enumerate(stamps) if s["clue"] != pattern}
            if not (partial & trace["visited"] and partial - trace["visited"]):
                continue
            if is_test and not _ablation_differs(info, trace, visible=True):
                continue
            candidate = field.copy()
            candidate[source_pos] = 2
            info["field"] = candidate
            return _finish(rng, info, palette, hh)
    raise AssertionError("bounded free-scene sampling exhausted")


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = str(rng.choice(("count", "sparse", "radius")))
    rotation = int(rng.integers(4))
    reflected = bool(rng.integers(2))
    stable_palette = rng.permutation(10) if rng.random() < .2 else None

    def palette():
        return stable_palette if stable_palette is not None else rng.permutation(10)

    first_radius = 3
    first, pattern = _teach(rng, first_radius, False, rotation, reflected, palette())
    second, _ = _teach(rng, 3, True, (rotation + 1) % 4, reflected, palette(), pattern)
    puzzle = {"train": [first, second], "test": []}
    seen = {_signature(first["input"]), _signature(second["input"])}
    for split, count in (("train", nt - 2), ("test", nq)):
        for _ in range(count):
            for attempt in range(40):
                pair = _random_pair(rng, mode, split, palette())
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("unique input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    headings = {-1: set(), 1: set()}
    latent, recursive, asymmetric_adjacent, unmarked, shape_sensitive = False, False, False, False, False
    hypotheses = [{-1: a, 1: b} for a in (-1, 1) for b in (-1, 1)]
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _read(inp)
            trace = _trace(info)
            predicted = _render(info, trace)
            _require(np.array_equal(predicted, out), "input-only solver disagrees")
            _require(out.shape == info["field"].shape, "wrong crop")
            partial = {i for i, s in enumerate(info["stamps"]) if s["clue"] != info["pattern"]}
            _require(bool(partial & trace["visited"]), "no reached partial ornament")
            _require(bool(partial - trace["visited"]), "no unreached partial ornament")
            _require(not np.array_equal(_render(info, trace, complete_all=True), out),
                     "complete-all shortcut survives")
            field = info["field"]
            _require(np.array_equal(out[field == info["frame"]], field[field == info["frame"]]),
                     "guide outline changed")
            _require(np.array_equal(out[np.isin(field, list(info["turns"]))],
                                    field[np.isin(field, list(info["turns"]))]), "existing accents changed")
            for i in partial - trace["visited"]:
                _require(all(out[p] == field[p] for p in info["stamps"][i]["points"]),
                         "unreached footprint changed")
            info["trace"] = trace
            infos[split].append(info)
            if split == "train":
                latent |= _ablation_differs(info, trace, visible=True)
                shape_sensitive |= _ablation_differs(info, trace, boxes=True)
                recursive |= any(len(set(c)) >= 2 for c in trace["chains"])
                unmarked |= len(info["stamps"]) < 4 * len(info["frames"])
                for _, _, d, side, _, _ in trace["events"]:
                    headings[side].add(d)
                intact = [s["corner"] for s in info["stamps"] if s["clue"] == info["pattern"]]
                asym = info["pattern"] != frozenset((v, u) for u, v in info["pattern"])
                asymmetric_adjacent |= asym and any(a[2] ^ b[2] in (1, 2)
                                                     and any(a in _corners(r) and b in _corners(r)
                                                             for r in info["frames"])
                                                     for a in intact for b in intact)
                remaining = []
                for mapping in hypotheses:
                    try:
                        fits = np.array_equal(_render(info, _trace(info, turn_map=mapping)), out)
                    except AssertionError:
                        fits = False
                    if fits:
                        remaining.append(mapping)
                hypotheses = remaining
            else:
                _require(_ablation_differs(info, trace, visible=True), "test has no latent routing effect")
    _require(latent and recursive and unmarked and shape_sensitive, "missing training interaction evidence")
    _require(asymmetric_adjacent, "missing adjacent asymmetric references")
    _require(all(len(dirs) >= 2 for dirs in headings.values()), "relative turns need multiple headings")
    _require(hypotheses == [{-1: -1, 1: 1}], "turn meanings are not uniquely demonstrated")
    _require(len({i["pattern"] for i in infos["train"]}) >= 2, "training uses one stencil")
    max_count = max(len(i["frames"]) for i in infos["train"])
    max_radius = max(max(max(p) for p in i["pattern"]) for i in infos["train"])
    no_singletons = all(len(s["clue"]) >= 2 for i in infos["train"] for s in i["stamps"])
    for info in infos["test"]:
        count_shift = len(info["frames"]) > max_count
        radius_shift = max(max(p) for p in info["pattern"]) > max_radius
        sparse_shift = no_singletons and all(len(s["clue"]) == 1 for s in info["stamps"]
                                            if s["clue"] != info["pattern"])
        _require(count_shift or radius_shift or sparse_shift, "missing meaningful test shift")


def _features(pair):
    info = _read(pair["input"])
    trace = _trace(info)
    masks = trace["paths"].copy()
    masks[:, info["source_pos"][0], info["source_pos"][1]] = False
    return dict(shape=info["field"].shape, frames=len(info["frames"]),
                stencil=tuple(sorted(info["pattern"])), stencil_size=len(info["pattern"]),
                radius=max(max(p) for p in info["pattern"]),
                marks=len(info["stamps"]), key_count=len(info["turns"]),
                header_height=info["crop"], source_position=info["source_pos"],
                max_encounters=max(map(len, trace["chains"])),
                distinct_chain=max(len(set(c)) for c in trace["chains"]),
                reached=len(trace["visited"]), unreached=len(info["stamps"]) - len(trace["visited"]),
                hidden_impacts=sum(e[-1] for e in trace["events"]),
                repeated_colors=len({s["color"] for s in info["stamps"]}) < len(info["stamps"]),
                unused_keys=len(set(info["turns"]) - {s["color"] for s in info["stamps"]}),
                crossing=bool(np.any(np.sum(masks, axis=0) > 1)),
                role_colors=(info["background"], info["frame"], info["source"]))


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test() -> dict:
    """Bounded semantic checks and descriptive diversity over 42 fixed seeds."""
    signatures, features = set(), []
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                frozen = np.asarray(pair["input"]).copy()
                prediction = solve(frozen)
                _require(np.array_equal(prediction, pair["output"]), "solve mismatch")
                _require(np.array_equal(frozen, pair["input"]), "solve mutated input")
                features.append(_features(pair))
    _require(len(signatures) == 42, "seed collapse")
    _require(len({f["stencil"] for f in features}) >= 60, "too few stencils")
    _require(len({f["shape"] for f in features}) >= 45, "too few field geometries")
    _require({f["frames"] for f in features} >= {1, 2, 3, 4}, "missing frame counts")
    _require({f["radius"] for f in features} >= {2, 3, 4}, "missing stencil scales")
    _require(len({f["stencil_size"] for f in features}) >= 7, "missing stencil size variation")
    _require(any(f["distinct_chain"] >= 3 for f in features), "missing longer recursive route")
    _require(any(f["crossing"] for f in features), "missing ray overlap")
    _require(any(f["max_encounters"] == 1 for f in features), "missing simpler routes")
    _require(any(f["unused_keys"] for f in features), "missing unused instructions")
    _require(any(f["repeated_colors"] for f in features), "missing repeated accent colors")
    _require(len({f["role_colors"] for f in features}) >= 30, "palette roles do not vary")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(71, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "count ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_digest(generate_puzzle(19, 6, 3)) ==
             _puzzle_digest(generate_puzzle(19, 6, 3)), "nondeterministic generation")
    # Check that validation catches an unrelated edit and a complete-all label.
    original = generate_puzzle(0, 2, 1)
    bad = {split: [{k: np.asarray(v).copy() for k, v in p.items()} for p in pairs]
           for split, pairs in original.items()}
    info = _read(bad["test"][0]["input"])
    bad["test"][0]["output"] = _render(info, _trace(info), complete_all=True)
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("complete-all corruption accepted")
    bad["test"][0]["output"] = original["test"][0]["output"].copy()
    guide_pixel = tuple(map(int, np.argwhere(info["field"] == info["frame"])[0]))
    bad["test"][0]["output"][guide_pixel] = info["source"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated guide corruption accepted")
    summary = dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                   field_shapes=len({f["shape"] for f in features}),
                   distinct_stencils=len({f["stencil"] for f in features}),
                   frame_counts=sorted({f["frames"] for f in features}),
                   stencil_sizes=sorted({f["stencil_size"] for f in features}),
                   maximum_encounters=max(f["max_encounters"] for f in features),
                   maximum_distinct_chain=max(f["distinct_chain"] for f in features),
                   pairs_with_hidden_impacts=sum(bool(f["hidden_impacts"]) for f in features),
                   pairs_with_overlapping_rays=sum(f["crossing"] for f in features))
    return summary


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
