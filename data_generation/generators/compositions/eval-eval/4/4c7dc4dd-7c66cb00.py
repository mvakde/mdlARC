"""Echo docks: reference-conditioned stencil substitution and component seating.

Only NumPy and the standard library are used. solve reads its input alone.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, points, diagonal=False):
    todo = set(points)
    result = []
    neighbors = _EIGHT if diagonal else _FOUR
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, cells = [first], set()
        color = int(a[first])
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in todo and int(a[p]) == color:
                    todo.remove(p)
                    stack.append(p)
        result.append((color, frozenset(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normalized(cells):
    r0, c0, _, _ = _bounds(cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _height(cells):
    r0, _, r1, _ = _bounds(cells)
    return r1-r0+1


def _expand(cells, stencil):
    return {(r+dr, c+dc) for r, c in cells for dr, dc in stencil}


def _interpret_canonical(a):
    """Read a downward scene, with no sampled parameters or example outputs."""
    h, w = a.shape
    bg = int(a[0, 0])
    _require(all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
             "corner margins disagree")
    dividers = [r for r in range(h) if int(a[r, 0]) != bg and np.all(a[r] == a[r, 0])]
    _require(len(dividers) == 1, "expected one calibration divider")
    divider = dividers[0]
    frame = int(a[divider, 0])
    points = [tuple(map(int, p)) for p in np.argwhere(a[:divider] == frame)]
    frames = _components(a, points)
    _require(len(frames) == 2, "expected two separate calibration outlines")
    panels, boxes = [], []
    header_allowed = np.zeros((divider, w), bool)
    for color, cells in frames:
        top, left, bottom, right = _bounds(cells)
        _require(bottom-top >= 2 and right-left >= 2, "empty calibration interior")
        perimeter = {(r, c) for r in range(top, bottom+1) for c in range(left, right+1)
                     if r in (top, bottom) or c in (left, right)}
        _require(cells == perimeter and top > 0 and left > 0 and right < w-1 and bottom < divider-1,
                 "invalid calibration outline or margin")
        panels.append(a[top+1:bottom, left+1:right].copy())
        boxes.append((top+1, left+1, bottom, right))
        header_allowed[top:bottom+1, left:right+1] = True
    _require(panels[0].shape == panels[1].shape, "calibration dimensions disagree")
    _require(np.all(a[:divider][~header_allowed] == bg), "extraneous calibration cells")
    counts = [int(np.count_nonzero(p != bg)) for p in panels]
    _require(counts.count(1) == 1 and max(counts) > 1, "calibration must show point and stencil")
    before_index = counts.index(1)
    after_index = 1-before_index
    before, after = panels[before_index], panels[after_index]
    origin = tuple(map(int, np.argwhere(before != bg)[0]))
    point = int(before[origin])
    _require(point != frame and point not in set(map(int, np.unique(after))),
             "reference point color belongs only to the source state")
    ink_colors = set(map(int, np.unique(after))) - {bg}
    _require(len(ink_colors) == 1, "expected a single stencil ink role")
    ink = next(iter(ink_colors))
    _require(len({bg, frame, point, ink}) == 4, "control roles must be distinct")
    _require(int(after[origin]) == ink, "expanded reference must include the source point's location")
    stencil = frozenset((int(r)-origin[0], int(c)-origin[1]) for r, c in np.argwhere(after != bg))
    _require(any(c != 0 for r, c in stencil), "stencil must expand laterally")

    row_info = {}
    for r in range(divider+1, h):
        nz = np.flatnonzero(a[r] != bg)
        if len(nz) < 8:
            continue
        left, right = int(nz[0]), int(nz[-1])
        route, rail = int(a[r, left+1]), int(a[r, left])
        if (right-left >= 7 and int(a[r, right]) == rail and route != rail and route != bg
                and np.all(a[r, left+1:right] == route)):
            row_info[r] = (left, right, route, rail)
    _require(bool(row_info), "no docks")
    first = min(row_info)
    _require(first >= divider+3 and np.all(a[first-1] == bg), "missing staging/dock gap")
    bands = []
    for r in sorted(row_info):
        left, right, route, rail = row_info[r]
        _require(not {route, rail} & {bg, frame, point, ink}, "dock uses a control color")
        if bands and r == bands[-1]["bottom"]+1 and row_info[r] == row_info[bands[-1]["top"]]:
            bands[-1]["bottom"] = r
        else:
            _require(not bands or r >= bands[-1]["bottom"]+2, "docks need separators")
            bands.append(dict(top=r, bottom=r, left=left, right=right, route=route, rail=rail))
    _require(len({b["route"] for b in bands}) == len(bands), "routing colors must be unique")
    for r in range(first, h):
        _require(r in row_info or np.all(a[r] == bg), "non-dock cell below staging")
    stage_points = [tuple(map(int, p)) for p in np.argwhere(a[divider+1:first] != bg)]
    stage_points = [(r+divider+1, c) for r, c in stage_points]
    _require(bool(stage_points), "empty staging field")
    components = _components(a, stage_points)
    routes = {b["route"] for b in bands}
    _require(all(color in routes for color, cells in components), "unroutable staging color")
    return dict(canonical=a, background=bg, divider=divider, frame=frame, point=point,
                ink=ink, panels=panels, boxes=boxes, after_index=after_index, origin=origin,
                stencil=stencil, bands=bands, components=components, first=first)


def _render_analysis(scene, *, stencil=None, group_colors=False):
    """The input-derived transformation; diagnostic ablations are explicit."""
    a = scene["canonical"]
    out = a.copy()
    bg = scene["background"]
    out[scene["divider"]+1:scene["first"]] = bg
    lookup = {b["route"]: b for b in scene["bands"]}
    kernel = scene["stencil"] if stencil is None else stencil
    comps = scene["components"]
    if group_colors:
        groups = {}
        for color, cells in comps:
            groups.setdefault(color, set()).update(cells)
        comps = list(groups.items())
    stamps = []
    for color, cells in comps:
        grown = _expand(cells, kernel)
        b = lookup[color]
        shift = b["bottom"]-max(r for r, c in grown)
        dest = {(r+shift, c) for r, c in grown}
        _require(all(b["top"] <= r <= b["bottom"] and b["left"] < c < b["right"] for r, c in dest),
                 "expanded silhouette crosses a dock boundary")
        for p in dest:
            out[p] = b["rail"]
        stamps.append((color, frozenset(dest)))
    return out, stamps


def _analyze(input_grid):
    a = _grid(input_grid)
    scenes = []
    for turn in range(4):
        canonical = np.rot90(a, turn)
        try:
            scene = _interpret_canonical(canonical)
            expected, stamps = _render_analysis(scene)
        except ValueError:
            continue
        scene.update(turn=turn, expected=expected, stamps=stamps,
                     output=np.rot90(expected, -turn).copy())
        scenes.append(scene)
    _require(len(scenes) == 1, "input needs one recognizable calibration/machine orientation")
    return scenes[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a complete output solely from the provided visible input."""
    return _analyze(input_grid)["output"]


def _random_connected(rng, height, width, count):
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    while len(cells) < min(count, height*width):
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _FOUR
                           if 0 <= r+dr < height and 0 <= c+dc < width and (r+dr, c+dc) not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return frozenset(_normalized(cells))


def _sample_stencil(rng):
    h, w = int(rng.integers(1, 4)), int(rng.integers(2, 6))
    mode = int(rng.integers(7))
    if mode == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif mode == 1:
        cells = {(0, c) for c in range(w)} | {(r, 0) for r in range(h)}
    elif mode == 2:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
    elif mode == 3:
        # Separated lobes deliberately need not form one connected stencil.
        cells = {(r, 0) for r in range(h)} | {(r, w-1) for r in range(h)}
    elif mode == 4:
        spine = int(rng.integers(w))
        cells = {(r, spine) for r in range(h)} | {(int(rng.integers(h)), c) for c in range(w)}
    else:
        cells = set(_random_connected(rng, h, w, int(rng.integers(2, h*w+1))))
        # A horizontal witness makes every generated input depend on growth.
        r, c = min(cells)
        cells.add((r, c+1 if c+1 < w else c-1))
    cells = list(_normalized(cells))
    origin = cells[int(rng.integers(len(cells)))]
    return frozenset((r-origin[0], c-origin[1]) for r, c in cells)


def _source_mask(rng, max_height, max_width=4, exact_height=False):
    h = max_height if exact_height else int(rng.integers(1, max_height+1))
    w = int(rng.integers(1, max_width+1))
    mode = int(rng.integers(5))
    if mode == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif mode == 1:
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif mode == 2:
        spine = int(rng.integers(w))
        cells = {(r, spine) for r in range(h)}
        for r in range(h):
            end = int(rng.integers(w))
            cells.update((r, c) for c in range(min(spine, end), max(spine, end)+1))
    else:
        cells = set(_random_connected(rng, h, w, int(rng.integers(1, h*w+1))))
        if exact_height:
            r0, c0 = min(cells)
            cells.update((r, c0) for r in range(h))
    return frozenset(_normalized(cells))


def _put_group(rng, stage, group, routes, bg, left, right, stencil_limits,
               placed, different_bottom=None, echo_contact=None):
    height = 1+max(r for i, mask in group for r, c in mask)
    width = 1+max(c for i, mask in group for r, c in mask)
    dc_min, dc_max = stencil_limits
    lo, hi = left+1-dc_min, right-1-dc_max-width+1
    if height > stage.shape[0] or hi < lo:
        return False
    candidates = [(r, c) for r in range(stage.shape[0]-height+1) for c in range(lo, hi+1)]
    for j in rng.permutation(len(candidates)):
        top, col = candidates[int(j)]
        translated = [(i, {(r+top, c+col) for r, c in mask}) for i, mask in group]
        if different_bottom is not None and any(max(r for r, c in cells) == different_bottom
                                               for i, cells in translated):
            continue
        if echo_contact is not None:
            parent, kernel = echo_contact
            old_echo = _expand(parent, kernel)
            new_echo = _expand(translated[0][1], kernel)
            if not (old_echo & new_echo or any((r+dr, c+dc) in old_echo
                                              for r, c in new_echo for dr, dc in _FOUR)):
                continue
        if any(int(stage[p]) != bg for i, cells in translated for p in cells):
            continue
        if any(0 <= r+dr < stage.shape[0] and 0 <= c+dc < stage.shape[1]
               and int(stage[r+dr, c+dc]) == routes[i]
               for i, cells in translated for r, c in cells for dr, dc in _EIGHT):
            continue
        for i, cells in translated:
            for p in cells:
                stage[p] = routes[i]
            placed.append((i, frozenset(cells)))
        return True
    return False


def _outline(a, top, left, height, width, color):
    a[top, left:left+width] = a[top+height-1, left:left+width] = color
    a[top:top+height, left] = a[top:top+height, left+width-1] = color


def _construct_scene(rng, kernels, max_source_height, rotation, teaching, allow_unused=False):
    """Construct one machine; multiple kernels give the controlled training twins."""
    kernel_union = set().union(*kernels)
    kr0, kc0, kr1, kc1 = _bounds(kernel_union)
    kh, kw = kr1-kr0+1, kc1-kc0+1
    min_width = 2*(kw+2)+3
    if min_width > 30:
        return None
    width = int(rng.integers(max(20, min_width), 31))
    palette = list(map(int, rng.permutation(10)))
    bg, frame, point, ink = palette[:4]
    count = int(rng.choice([2, 2, 3] if teaching else [1, 2, 2, 3]))
    routes = palette[4:4+count]
    choices = palette[4:]
    rails = [int(rng.choice([c for c in choices if c != route])) for route in routes]
    spans = [(int(rng.integers(3)), width-1-int(rng.integers(3))) for _ in routes]
    left, right = max(x[0] for x in spans), min(x[1] for x in spans)
    stage_h = max_source_height+int(rng.integers(2, 4))
    stage = np.full((stage_h, width), bg, dtype=np.int8)
    placed = []
    limits = kc0, kc1
    if count >= 2:
        h, w = max_source_height, int(rng.integers(2, 5))
        if h >= 3 and w >= 3 and rng.random() < .5:
            outer = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h-1) or c in (0, w-1)}
            inner = {(r, c) for r in range(1, h-1) for c in range(1, w-1)}
        else:
            outer = {(r, c) for r in range(h) for c in range(w) if c == 0 or r == h-1}
            inner = {(r, c) for r in range(h-1) for c in range(1, w)}
        if not _put_group(rng, stage, [(0, outer), (1, inner)], routes, bg, left, right,
                          limits, placed):
            return None
    else:
        mask = _source_mask(rng, max_source_height, exact_height=True)
        if not _put_group(rng, stage, [(0, mask)], routes, bg, left, right, limits, placed):
            return None
    if teaching:
        first_bottom = max(r for r, c in placed[0][1])
        mask = _source_mask(rng, max_source_height)
        if not _put_group(rng, stage, [(0, mask)], routes, bg, left, right, limits,
                          placed, different_bottom=first_bottom, echo_contact=(placed[0][1], kernels[0])):
            return None
    # All routes may be used, or a spare dock may remain visibly unchanged.
    active = list(range(count))
    for i in range(2, count):
        if (teaching or allow_unused) and rng.random() < .35:
            active.remove(i)
            continue
        if not _put_group(rng, stage, [(i, _source_mask(rng, max_source_height))], routes,
                          bg, left, right, limits, placed):
            return None
    for _ in range(int(rng.integers(0, 4))):
        i = int(rng.choice(active))
        _put_group(rng, stage, [(i, _source_mask(rng, max_source_height))], routes, bg,
                   left, right, limits, placed)
    heights = [max([_height(cells) for i, cells in placed if i == route_index] or [0])
               for route_index in range(count)]
    kernel_h = max(_height(s) for s in kernels)
    depths = [n+kernel_h-1+int(rng.integers(2)) if n else int(rng.integers(1, 4)) for n in heights]
    gap = int(rng.integers(1, 3))
    separations = [int(rng.integers(1, 3)) for _ in range(count-1)]
    far_margin = int(rng.integers(1, 3))
    rest = stage_h+gap+sum(depths)+sum(separations)+far_margin
    max_header = 30-1-rest
    if max_header < kh+4:
        return None
    # Extra rows/columns around the reference need no geometric interpretation.
    panel_h = kh+int(rng.integers(min(2, max_header-kh-4)+1))
    panel_w = kw+int(rng.integers(min(2, (width-3)//2-2-kw)+1))
    header_h = panel_h+4
    header_slack = min(2, max_header-header_h)
    header_h += int(rng.integers(header_slack+1))
    height = header_h+1+rest
    a = np.full((height, width), bg, dtype=np.int8)
    divider = header_h
    a[divider] = frame
    stage_top = divider+1
    a[stage_top:stage_top+stage_h] = stage
    cursor = stage_top+stage_h+gap
    order = list(map(int, rng.permutation(count)))
    bands = {}
    for ordinal, i in enumerate(order):
        l, r = spans[i]
        b = dict(top=cursor, bottom=cursor+depths[i]-1, left=l, right=r,
                 route=routes[i], rail=rails[i])
        bands[i] = b
        a[cursor:cursor+depths[i], l:r+1] = routes[i]
        a[cursor:cursor+depths[i], [l, r]] = rails[i]
        cursor += depths[i]+(separations[ordinal] if ordinal < count-1 else 0)
    outer_h, outer_w = panel_h+2, panel_w+2
    available = width-2*outer_w-2
    panel_gap = int(rng.integers(1, available+1))
    left0 = int(rng.integers(1, width-2*outer_w-panel_gap))
    positions = [(int(rng.integers(1, divider-outer_h)), left0),
                 (int(rng.integers(1, divider-outer_h)), left0+outer_w+panel_gap)]
    origin = (-kr0+int(rng.integers(panel_h-kh+1)),
              -kc0+int(rng.integers(panel_w-kw+1)))
    after_slot = int(rng.integers(2))
    for top, col in positions:
        _outline(a, top, col, outer_h, outer_w, frame)
        a[top+1+origin[0], col+1+origin[1]] = point
    pairs = []
    for stencil in kernels:
        inp = a.copy()
        top, col = positions[after_slot]
        for dr, dc in stencil:
            inp[top+1+origin[0]+dr, col+1+origin[1]+dc] = ink
        # Ground truth from construction masks and known destinations. No parser.
        out = inp.copy()
        for i, cells in placed:
            for r, c in cells:
                out[stage_top+r, c] = bg
            grown = {(r+dr, c+dc) for r, c in cells for dr, dc in stencil}
            b = bands[i]
            shift = b["bottom"]-max(r for r, c in grown)
            dest = {(r+shift, c) for r, c in grown}
            if not all(b["top"] <= r <= b["bottom"] and b["left"] < c < b["right"] for r, c in dest):
                return None
            for p in dest:
                out[p] = b["rail"]
        pair = {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}
        try:
            analysis = _analyze(pair["input"])
        except ValueError:
            return None
        if not np.array_equal(analysis["output"], pair["output"]):
            return None
        pairs.append(pair)
    if len(pairs) == 2:
        first_out = np.rot90(pairs[0]["output"], -rotation)
        second_out = np.rot90(pairs[1]["output"], -rotation)
        if np.array_equal(first_out[divider+1:], second_out[divider+1:]):
            return None
    if teaching:
        lesson = _analyze(pairs[0]["input"])
        if not _whole_color_differs(lesson) or not _growth_regroup_differs(lesson):
            return None
    return pairs


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_n, test_n = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_height = int(rng.choice([2, 3]))
    turn = int(rng.integers(4))
    mixed = train_n >= 3 and bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    for _ in range(200):
        kernels = [_sample_stencil(rng), _sample_stencil(rng)]
        # The minimum two-example puzzle must teach the anchor coordinate and
        # seating the expanded far extent, not just an unanchored icon copy.
        first = kernels[0]
        if (kernels[0] == kernels[1] or not any(c < 0 for r, c in first)
                or not any(c > 0 for r, c in first) or not any(r > 0 for r, c in first)):
            continue
        twins = _construct_scene(rng, kernels, train_height, turn, True)
        if twins is not None:
            puzzle["train"].extend(twins)
            break
    else:
        raise RuntimeError("could not construct the controlled training pair")
    seen = {_key(p["input"]) for p in puzzle["train"]}
    teaching_scene = _analyze(puzzle["train"][0]["input"])
    allow_unused = len({color for color, cells in teaching_scene["components"]}) < len(teaching_scene["bands"])
    for split, remaining in (("train", train_n-2), ("test", test_n)):
        for index in range(remaining):
            rotation = ((turn+1+index % 3) % 4 if mixed else turn)
            max_height = train_height+int(split == "test")
            for _ in range(200):
                candidate = _construct_scene(rng, [_sample_stencil(rng)], max_height, rotation, False, allow_unused)
                if candidate is not None and _key(candidate[0]["input"]) not in seen:
                    pair = candidate[0]
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct scene within the space budget")
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_contact(scene):
    a = scene["canonical"]
    points = {p for color, cells in scene["components"] for p in cells}
    return any((r+dr, c+dc) in points and int(a[r, c]) != int(a[r+dr, c+dc])
               for r, c in points for dr, dc in _FOUR)


def _has_independence(scene):
    comps = scene["components"]
    return any(color == other and max(r for r, c in cells) != max(r for r, c in second)
               for j, (color, cells) in enumerate(comps) for other, second in comps[:j])


def _whole_color_differs(scene):
    try:
        grouped, _ = _render_analysis(scene, group_colors=True)
    except ValueError:
        # The alternative can also contradict visible dock capacity.
        return True
    return not np.array_equal(grouped, scene["expected"])


def _regroup_after_growth(scene):
    """Alternative rule probe: expand the field, then segment its new occupancy."""
    out = scene["canonical"].copy()
    out[scene["divider"]+1:scene["first"]] = scene["background"]
    groups = {}
    for color, cells in scene["components"]:
        groups.setdefault(color, set()).update(_expand(cells, scene["stencil"]))
    by_route = {b["route"]: b for b in scene["bands"]}
    for color, grown in groups.items():
        todo = set(grown)
        while todo:
            stack, cells = [todo.pop()], set()
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in _FOUR:
                    p = r+dr, c+dc
                    if p in todo:
                        todo.remove(p)
                        stack.append(p)
            b = by_route[color]
            shift = b["bottom"]-max(r for r, c in cells)
            dest = {(r+shift, c) for r, c in cells}
            _require(all(b["top"] <= r <= b["bottom"] and b["left"] < c < b["right"]
                         for r, c in dest), "regrouped growth violates dock capacity")
            for p in dest:
                out[p] = b["rail"]
    return out


def _growth_regroup_differs(scene):
    try:
        alternative = _regroup_after_growth(scene)
    except ValueError:
        return True
    return not np.array_equal(alternative, scene["expected"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output must preserve dimensions")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(a["output"], out), "incorrect echo-dock output")
            points = {p for color, cells in a["components"] for p in cells}
            four = {cells for color, cells in a["components"]}
            eight = {cells for color, cells in _components(a["canonical"], points, True)}
            _require(four == eight, "generated original components must have unambiguous connectivity")
            raw, _ = _render_analysis(a, stencil={(0, 0)})
            _require(not np.array_equal(raw, a["expected"]), "stencil concept does not affect answer")
            # Exact unchanged-cell check in addition to full-label comparison.
            allowed = np.zeros(a["canonical"].shape, bool)
            for p in points:
                allowed[p] = True
            for color, cells in a["stamps"]:
                for p in cells:
                    allowed[p] = True
            _require(np.array_equal(a["canonical"][~allowed], a["expected"][~allowed]),
                     "modified reference, rail, separator, or unoccupied interior")
            _require(all(any(r == next(b["bottom"] for b in a["bands"] if b["route"] == color)
                             for r, c in cells) for color, cells in a["stamps"]),
                     "each expanded component must touch its own far edge")
            analyses[split].append(a)
    first, second = analyses["train"][:2]
    _require(first["turn"] == second["turn"] and first["canonical"].shape == second["canonical"].shape,
             "controlled examples must share orientation and dimensions")
    mask = np.ones(first["canonical"].shape, bool)
    t, l, b, r = first["boxes"][first["after_index"]]
    mask[t:b, l:r] = False
    _require(first["boxes"] == second["boxes"] and first["after_index"] == second["after_index"]
             and np.array_equal(first["canonical"][mask], second["canonical"][mask]),
             "first two examples must change only the calibration result")
    _require(first["stencil"] != second["stencil"] and
             not np.array_equal(first["expected"][first["divider"]+1:],
                                second["expected"][second["divider"]+1:]),
             "reference comparison must change the answer")
    _require(_has_contact(first) and _has_independence(first), "missing component-routing lessons")
    _require(any(c < 0 for r, c in first["stencil"]) and any(c > 0 for r, c in first["stencil"])
             and any(r > 0 for r, c in first["stencil"]), "missing anchor or expanded-extent teaching witness")
    _require(any(len(cells) > 1 for color, cells in first["components"]), "no per-cell substitution witness")
    _require(_whole_color_differs(first), "component independence does not affect answer")
    _require(_growth_regroup_differs(first), "training must distinguish segmentation before growth")
    max_training = max(_height(cells) for a in analyses["train"] for color, cells in a["components"])
    _require(all(max(_height(cells) for color, cells in a["components"]) > max_training
                 for a in analyses["test"]), "missing taller-component train/test shift")
    if any(a["turn"] != first["turn"] for a in analyses["test"]):
        _require(len({a["turn"] for a in analyses["train"]}) >= 2, "rotation variation needs training evidence")
    if any(len({color for color, cells in a["components"]}) < len(a["bands"]) for a in analyses["test"]):
        _require(any(len({color for color, cells in a["components"]}) < len(a["bands"])
                     for a in analyses["train"]), "unused test dock needs a training demonstration")


def _diagnostic_grid(stencil, pieces, *, depth=5):
    """Small handcrafted scene, separate from randomized construction."""
    a = np.zeros((20, 24), dtype=np.int8)
    for left in (1, 12):
        _outline(a, 1, left, 5, 9, 5)
        a[3, left+3] = 8
    for dr, dc in stencil:
        a[3+dr, 15+dc] = 7
    a[7] = 5
    for color, cells in pieces:
        for r, c in cells:
            a[r, c] = color
    a[14:14+depth, 1:-1] = 1
    a[14:14+depth, [0, -1]] = 2
    return a


def _diagnostics():
    # Two source cells instantiate a separated echo at each cell, leaving a gap.
    pieces = [(1, {(8, 4), (8, 5)})]
    inp = _diagnostic_grid({(0, 0), (0, 3)}, pieces)
    expected = inp.copy()
    expected[8, 4:6] = 0
    expected[18, [4, 5, 7, 8]] = 2
    _require(np.array_equal(solve(inp), expected), "per-cell disconnected substitution regression")
    # Same machine and point, alternate blueprint, different exact answer.
    other = _diagnostic_grid({(0, 0), (1, 0), (1, 1)}, pieces)
    expected_other = other.copy()
    expected_other[8, 4:6] = 0
    expected_other[17, 4:6] = 2
    expected_other[18, 4:7] = 2
    _require(np.array_equal(solve(other), expected_other), "reference-conditioned alignment regression")
    _require(not np.array_equal(solve(other)[14:], solve(inp)[14:]), "reference can be ignored")
    # The before panel supplies the anchor: the identical after blueprint with
    # another corresponding source position changes all lateral stamp offsets.
    square = _diagnostic_grid({(0, 0), (0, 1), (1, 0), (1, 1)}, pieces)
    square_expected = square.copy()
    square_expected[8, 4:6] = 0
    square_expected[17:19, 4:7] = 2
    shifted_anchor = square.copy()
    shifted_anchor[3, 4], shifted_anchor[3, 5] = 0, 8
    anchor_expected = shifted_anchor.copy()
    anchor_expected[8, 4:6] = 0
    anchor_expected[17:19, 3:6] = 2
    _require(np.array_equal(solve(square), square_expected) and
             np.array_equal(solve(shifted_anchor), anchor_expected), "source-panel anchor ignored")
    # Original pieces at unequal depths grow into a common connected source echo.
    # They must still both independently reach row 18.
    merge = _diagnostic_grid({(0, 0), (1, 0), (1, 1)}, [(1, {(8, 4)}), (1, {(10, 5)})])
    merge_expected = merge.copy()
    merge_expected[8, 4] = merge_expected[10, 5] = 0
    merge_expected[17, 4:6] = 2
    merge_expected[18, 4:7] = 2
    _require(np.array_equal(solve(merge), merge_expected), "identity-before-growth regression")
    grown = _expand({(8, 4)}, {(0, 0), (1, 0), (1, 1)}) | _expand({(10, 5)}, {(0, 0), (1, 0), (1, 1)})
    dummy = np.zeros((20, 20), dtype=np.int8)
    for p in grown:
        dummy[p] = 1
    _require(len(_components(dummy, grown)) == 1, "merging diagnostic does not actually merge")
    grouped, _ = _render_analysis(_analyze(merge), group_colors=True)
    _require(not np.array_equal(grouped, merge_expected), "whole-color shortcut regression")
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(other, turn)), np.rot90(expected_other, turn)), "rotation regression")
    # Swapping calibration panels must not invert the transformation.
    swapped = other.copy()
    swapped[1:6, 1:10], swapped[1:6, 12:21] = other[1:6, 12:21], other[1:6, 1:10]
    swapped_expected = expected_other.copy()
    swapped_expected[:7] = swapped[:7]
    _require(np.array_equal(solve(swapped), swapped_expected), "panel order selects semantics")


def self_test():
    """Bounded 42-seed tests, explicit-count checks, and interaction regressions."""
    _diagnostics()
    shapes, stencils, masks, signatures = set(), set(), set(), set()
    directions, counts, role_maps, areas = set(), set(), set(), {}
    facts = Counter()
    depth_sequences, source_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        _require(all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], repeat[s])
                     for k in ("input", "output")), "nondeterministic generation")
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                stencils.add(tuple(sorted(a["stencil"])))
                directions.add(a["turn"])
                counts.add(len(a["bands"]))
                source_counts.add(len(a["components"]))
                depth_sequences.add(tuple(b["bottom"]-b["top"]+1 for b in a["bands"]))
                role_maps.add((a["background"], a["frame"], a["point"], a["ink"]))
                facts["pairs"] += 1
                facts["contact"] += _has_contact(a)
                facts["independent"] += _has_independence(a)
                facts["left_echo"] += any(c < 0 for r, c in a["stencil"])
                facts["right_echo"] += any(c > 0 for r, c in a["stencil"])
                facts["inset"] += any(b["left"] or b["right"] < a["canonical"].shape[1]-1 for b in a["bands"])
                facts["shared_rails"] += len({b["rail"] for b in a["bands"]}) < len(a["bands"])
                facts["role_reuse"] += bool({b["rail"] for b in a["bands"]} & {b["route"] for b in a["bands"]})
                by_route = {b["route"]: i for i, b in enumerate(a["bands"])}
                facts["unused"] += len({color for color, cells in a["components"]}) < len(a["bands"])
                kernel = a["stencil"]
                normalized = _normalized(kernel)
                sh, sw = _height(kernel), max(c for r, c in normalized)+1
                mini = np.zeros((sh, sw), dtype=np.int8)
                for p in normalized:
                    mini[p] = 1
                facts["disconnected_stencil"] += len(_components(mini, normalized)) > 1
                facts["stamp_overlap"] += any(color == other and bool(cells & earlier)
                                               for j, (color, cells) in enumerate(a["stamps"])
                                               for other, earlier in a["stamps"][:j])
                for color, cells in a["components"]:
                    masks.add(_normalized(cells))
                    areas.setdefault(len(cells), set()).add(by_route[color])
    _require(len(signatures) == 42, "cross-seed duplicate puzzles")
    _require(len(shapes) >= 25 and len(stencils) >= 70 and len(masks) >= 40, "narrow structural variation")
    _require(directions == {0, 1, 2, 3} and counts == {1, 2, 3}, "missing orientation or dock-count variation")
    _require(len(role_maps) >= 50 and len(depth_sequences) >= 25 and len(source_counts) >= 4,
             "narrow palette, depth, or multiplicity variation")
    _require(all(facts[k] for k in ("left_echo", "right_echo", "inset", "shared_rails", "role_reuse",
                                    "unused", "disconnected_stencil", "stamp_overlap")), "unexercised structural feature")
    _require(sum(len(dest) >= 2 for dest in areas.values()) >= 5, "shape area predicts dock position")
    # Every legal requested combination is tested once, with extremes included.
    legal_counts = 0
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1200+train*10+test, train, test)
            _require(len(p["train"]) == train and len(p["test"]) == test, "requested counts ignored")
            legal_counts += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(907, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=facts["pairs"], distinct_puzzles=len(signatures),
                canvas_shapes=len(shapes), anchored_stencils=len(stencils), source_masks=len(masks),
                dock_counts=sorted(counts), orientations=sorted(directions), role_palettes=len(role_maps),
                depth_sequences=len(depth_sequences), source_component_counts=sorted(source_counts),
                feature_pairs=dict(facts), legal_count_combinations=legal_counts,
                verification="generator-owned plus handcrafted pixel regressions")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
