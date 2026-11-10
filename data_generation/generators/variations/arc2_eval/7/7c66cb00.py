"""Color-routed component translation into visible, two-railed receptacles.

Geometry and colors are pair-local evidence. Four-connectivity and independent
far-edge alignment are shared rules; complete scenes may be globally rotated.
"""
import hashlib
import numpy as np


_SOURCE_PAIRS = {
    "6e86f8721d11febff86ed62e7abc0ab1cda33e7992b5f666d7988d5fff8a48c3",
    "67a673a4b9b074e1f7619f037bcdd050975ef93ba03e15b5e2c1bace503fdb67",
    "24c363819a9603d394e0ea33fc85557c770d044de4a9a1690ad77b1af37a63a1",
    "7273d57c5bf03690b67e6a237c6c9481b5b4522aa3cf50b884baa40f3225fe96",
}
_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _validate_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _validate_optional_count(num_train, "num_train", 2, 10)
    test = _validate_optional_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        return low + (seed // 4) % (11 - low), test
    if train is None:
        return int(rng.integers(max(2, test + 1), 11)), test
    return train, int(rng.integers(1, min(4, train - 1) + 1))


def _components(grid, points, diagonal=False):
    unseen = set(points)
    result = []
    neighbors = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        color = int(grid[start])
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen and int(grid[p]) == color:
                    unseen.remove(p)
                    stack.append(p)
        result.append({"color": color, "cells": tuple(sorted(cells))})
    return result


def _shape_signature(component):
    cells = component["cells"]
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _stamp_from_analysis(grid, background, bands, components):
    out = grid.copy()
    by_color = {b["route"]: b for b in bands}
    occupied = set()
    targets = []
    for comp in components:
        _require(comp["color"] in by_color, "unroutable staging color")
        band = by_color[comp["color"]]
        bottom = max(r for r, c in comp["cells"])
        cells = {(band["bottom"] - bottom + r, c) for r, c in comp["cells"]}
        _require(not cells & occupied, "same-destination cell collision")
        _require(all(band["top"] <= r <= band["bottom"] and
                     band["left"] < c < band["right"] for r, c in cells),
                 "translated mask crosses a rail or destination boundary")
        for r, c in comp["cells"]:
            out[r, c] = background
        targets.append((band["rail"], cells))
        occupied.update(cells)
    for color, cells in targets:
        for p in cells:
            out[p] = color
    return out


def _canonical_analyses(grid, background):
    """Find complete band suffixes using only visible pixels, including insets."""
    h, w = grid.shape
    rows = []
    for row in grid:
        nz = np.flatnonzero(row != background)
        if not len(nz):
            rows.append(None)
            continue
        left, right = int(nz[0]), int(nz[-1])
        if (right-left >= 5 and row[left] == row[right] and
                row[left] != row[left+1] and row[left+1] != background and
                np.all(row[left+1:right] == row[left+1])):
            rows.append((left, right, int(row[left+1]), int(row[left])))
        else:
            rows.append(False)
    # A candidate first band must follow a clear staging/band separator.
    for first in range(2, h):
        if not rows[first] or rows[first-1] is not None:
            continue
        if any(info is False for info in rows[first:]):
            continue
        bands = []
        r = first
        while r < h:
            if rows[r] is None:
                r += 1
                continue
            left, right, route, rail = rows[r]
            top = r
            while r+1 < h and rows[r+1] == rows[top]:
                r += 1
            bands.append(dict(top=top, bottom=r, left=left, right=right,
                              route=route, rail=rail))
            r += 1
        if len({b["route"] for b in bands}) != len(bands):
            continue
        if any(b["top"] < a["bottom"]+2 for a, b in zip(bands, bands[1:])):
            continue
        points = [tuple(map(int, p)) for p in np.argwhere(grid[:first] != background)]
        if not points:
            continue
        components = _components(grid, points)
        try:
            expected = _stamp_from_analysis(grid, background, bands, components)
        except AssertionError:
            continue
        yield dict(background=background, bands=bands, components=components,
                   first_top=first, canonical=grid, expected=expected)


def _analyze_input(grid):
    candidates = []
    # Staging is laterally inset, so its two near-side corners reveal background.
    backgrounds = sorted({int(grid[0, 0]), int(grid[0, -1]),
                          int(grid[-1, 0]), int(grid[-1, -1])})
    for rotation in range(4):
        canonical = np.rot90(grid, rotation)
        for background in backgrounds:
            for a in _canonical_analyses(canonical, background):
                a["rotation"] = rotation
                a["output"] = np.rot90(a["expected"], -rotation).copy()
                candidates.append(a)
    _require(len(candidates) == 1, "input must have one recognizable staging/band interpretation")
    return candidates[0]


def _solve_input(grid):
    return _analyze_input(grid)["output"]


def _random_polyomino(rng, max_height, max_width, cell_count):
    count = min(int(cell_count), max_height*max_width)
    cells = {(int(rng.integers(max_height)), int(rng.integers(max_width)))}
    while len(cells) < count:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _FOUR
                           if 0 <= r+dr < max_height and 0 <= c+dc < max_width
                           and (r+dr, c+dc) not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _shape_signature({"cells": cells})


def _mask(rng, height, width, level):
    """Compositional solid, bar, branch, asymmetric and staircase masks."""
    h = int(rng.integers(1, height+1))
    w = int(rng.integers(1, width+1))
    if level == 0:
        h, w = min(h, 3), min(w, 4)
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        if rng.random() < .5:
            h = 1
        else:
            w = 1
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 2:
        spine = int(rng.integers(w))
        cells = {(r, spine) for r in range(h)}
        for r in rng.choice(h, size=int(rng.integers(1, h+1)), replace=False):
            end = int(rng.integers(w))
            cells.update((int(r), c) for c in range(min(end, spine), max(end, spine)+1))
    elif kind == 3:
        cells = {(0, 0)}
        r = c = 0
        while r < h-1 or c < w-1:
            if r < h-1 and (c == w-1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            cells.add((r, c))
    else:
        return _random_polyomino(rng, h, w, int(rng.integers(1, h*w+1)))
    return _shape_signature({"cells": cells})


def _frame_group(rng, bands, active, stage, level):
    eligible = [i for i in active if bands[i]["bottom"]-bands[i]["top"]+1 >= 3]
    if not eligible or len(active) < 2:
        return None
    outer = int(rng.choice(eligible))
    others = [i for i in active if i != outer]
    inner = int(rng.choice(others))
    left = max(bands[i]["left"] for i in (outer, inner))+1
    right = min(bands[i]["right"] for i in (outer, inner))-1
    maxh = min(stage, bands[outer]["bottom"]-bands[outer]["top"]+1)
    maxw = right-left+1
    if maxw < 3:
        return None
    # The only absolute limits are the available staging/interior dimensions.
    h = int(rng.integers(3, min(maxh, 4 if level == 0 else maxh)+1))
    w = int(rng.integers(3, min(maxw, 6 if level == 0 else maxw)+1))
    thickness = int(rng.integers(1, min((h-1)//2, (w-1)//2)+1))
    hole_h, hole_w = h-2*thickness, w-2*thickness
    windows = int(rng.integers(1, min(4, (hole_w+1)//2)+1)) if level else 1
    wall = int(rng.integers(1, min(thickness+1, (hole_w-windows)//max(1, windows-1))+1)) if windows > 1 else 0
    widths = [1]*windows
    for _ in range(hole_w-windows-wall*(windows-1)):
        widths[int(rng.integers(windows))] += 1
    outer_cells = {(r, c) for r in range(h) for c in range(w)}
    group = []
    cursor = thickness
    for width in widths:
        hole = {(r, c) for r in range(thickness, h-thickness)
                for c in range(cursor, cursor+width)}
        if level and hole_h > 2 and width > 2 and rng.random() < .3:
            # Irregular windows still have a connected surrounding wall.
            hole = {(r+thickness, c+cursor) for r, c in
                    _random_polyomino(rng, hole_h, width,
                                      int(rng.integers(max(hole_h, width), hole_h*width+1)))}
        outer_cells.difference_update(hole)
        dest = inner if rng.random() < .7 else int(rng.choice(others))
        max_inner_h = bands[dest]["bottom"]-bands[dest]["top"]+1
        available = {(r, c) for r, c in hole if r < thickness+max_inner_h}
        if not available:
            return None
        if rng.random() < .5:
            # Grow a connected, variably filled island starting beside its wall.
            start = min(available)
            fill = {start}
            goal = int(rng.integers(1, len(available)+1))
            while len(fill) < goal:
                frontier = sorted({(r+dr, c+dc) for r, c in fill for dr, dc in _FOUR
                                   if (r+dr, c+dc) in available-fill})
                if not frontier:
                    break
                fill.add(frontier[int(rng.integers(len(frontier)))])
        else:
            fill = available
        # Irregular clipping can split an island: retain its true components.
        mini = np.full((h, w), -1, dtype=np.int8)
        for p in fill:
            mini[p] = dest
        for comp in _components(mini, fill):
            group.append((dest, comp["cells"]))
        cursor += width+wall
    if level and maxw > w and rng.random() < .55:
        # Side extensions preserve a closed, connected frame. Even a shallow
        # three-row frame can have the source's stepped, asymmetric outline.
        extra = int(rng.integers(1, maxw-w+1))
        before = int(rng.integers(extra+1))
        after = extra-before
        outer_cells = {(r, c+before) for r, c in outer_cells}
        group = [(dest, tuple((r, c+before) for r, c in cells)) for dest, cells in group]
        for left, width in ((0, before), (before+w, after)):
            start = int(rng.integers(1, h-1))
            stop = int(rng.integers(start+1, h))
            outer_cells.update((r, c) for r in range(start, stop) for c in range(left, left+width))
        w += extra
    mini = np.zeros((h, w), dtype=np.int8)
    if len(_components(mini, outer_cells)) != 1:
        return None
    return [(outer, tuple(sorted(outer_cells)))]+group


def _try_group(grid, group, bands, occupied, background, stage, rng,
               bottom_constraint=None, edge=None):
    if not group:
        return False
    height = 1+max(r for _, cells in group for r, c in cells)
    width = 1+max(c for _, cells in group for r, c in cells)
    lo = max(bands[i]["left"]+1 for i, _ in group)
    hi = min(bands[i]["right"] for i, _ in group)-width
    if height > stage or hi < lo:
        return False
    rows = list(map(int, rng.permutation(stage-height+1)))
    if edge is True:
        rows = [0]
    elif edge is False:
        rows = [r for r in rows if r > 0]
    cols = list(map(int, rng.permutation(np.arange(lo, hi+1))))
    for top in rows:
        for left in cols:
            source, target, colors = set(), set(), {}
            valid = True
            for i, mask in group:
                band = bands[i]
                color = band["route"]
                cells = {(top+r, left+c) for r, c in mask}
                depth = max(r for r, c in cells)
                if bottom_constraint is not None:
                    mode, value = bottom_constraint
                    if (mode == "different" and depth == value) or (mode == "equal" and depth != value):
                        valid = False
                        break
                dest = {(band["bottom"]-depth+r, c) for r, c in cells}
                if source & cells or target & dest or occupied & dest:
                    valid = False
                    break
                if any(r < band["top"] or not band["left"] < c < band["right"] for r, c in dest):
                    valid = False
                    break
                if any(grid[p] != background for p in cells):
                    valid = False
                    break
                # Avoid all diagonal-only same-color grouping ambiguities.
                if any(0 <= r+dr < stage and 0 <= c+dc < grid.shape[1] and
                       grid[r+dr, c+dc] == color for r, c in cells for dr, dc in _EIGHT):
                    valid = False
                    break
                if any(colors.get((r+dr, c+dc)) == color
                       for r, c in cells for dr, dc in _EIGHT):
                    valid = False
                    break
                source.update(cells)
                target.update(dest)
                colors.update({p: color for p in cells})
            if valid:
                for p, color in colors.items():
                    grid[p] = color
                occupied.update(target)
                return True
    return False


def _make_roles(rng, count, background):
    available = [int(c) for c in rng.permutation(10) if c != background]
    routes = available[:count]
    extras = available[count:]
    reuse = int(rng.integers(count+1)) if count > 1 else 0
    reuse_at = set(map(int, rng.choice(count, size=reuse, replace=False)))
    # Sharing a rail is independent of whether it is a routing color elsewhere.
    common = int(rng.choice(extras))
    rails = []
    for i in range(count):
        choices = [r for r in routes if r != routes[i]] if i in reuse_at else extras
        rails.append(common if i not in reuse_at and rng.random() < .45 else int(rng.choice(choices)))
    return routes, rails


def _layout(rng, level, lesson, unused, background, avoid_count=None):
    counts = [1, 2, 2, 3, 3, 4, 5] if level else [1, 1, 2, 2, 3, 4]
    minimum = 2 if lesson == "split" or unused else 1
    choices = [n for n in counts if n >= minimum and n != avoid_count]
    count = int(rng.choice(choices))
    stage = int(rng.integers(5 if lesson else 3, (8, 12, 15)[level]))
    width = int(rng.integers(16 if lesson else 9, 31))
    if rng.random() < .12:
        width = 30
    gaps = [int(rng.integers(1, 4)) for _ in range(count)]
    margin = int(rng.integers(0, 5))
    required = [1]*count
    if lesson == "split":
        required[int(rng.integers(count))] = 3
    # Reserve the stage, all separators, and minimum destination depths first.
    while stage+sum(gaps)+margin+sum(required) > 30:
        if max(gaps) > 1:
            gaps[gaps.index(max(gaps))] -= 1
        elif margin:
            margin -= 1
        else:
            stage -= 1
    budget = 30-stage-sum(gaps)-margin
    if rng.random() < .22 and max(required)*count <= budget:
        heights = [int(rng.integers(max(required), budget//count+1))]*count
    else:
        heights = required[:]
        extra = int(rng.integers(budget-sum(heights)+1))
        weights = rng.random(count)+.2
        for i in rng.choice(count, size=extra, p=weights/weights.sum()):
            heights[int(i)] += 1
    h = stage+sum(gaps)+sum(heights)+margin
    background = int(background)
    routes, rails = _make_roles(rng, count, background)
    grid = np.full((h, width), background, dtype=np.int8)
    bands, cursor = [], stage
    inset = rng.random() < .45
    for depth, gap, route, rail in zip(heights, gaps, routes, rails):
        cursor += gap
        left = int(rng.integers(0, max(1, width//5))) if inset else 0
        right = width-1-(int(rng.integers(0, max(1, width//5))) if inset else 0)
        band = dict(top=cursor, bottom=cursor+depth-1, left=left,
                    right=right, route=route, rail=rail)
        bands.append(band)
        grid[cursor:cursor+depth, left:right+1] = route
        grid[cursor:cursor+depth, left] = rail
        grid[cursor:cursor+depth, right] = rail
        cursor += depth
    active = list(range(count))
    if unused:
        eligible = [i for i in active if lesson != "split" or
                    any(heights[j] >= 3 for j in active if j != i)]
        if count < 3 and lesson == "split":
            return None
        active.remove(int(rng.choice(eligible)))
    return grid, bands, active, stage


def _has_contact(a):
    grid = a["canonical"]
    cells = {p for comp in a["components"] for p in comp["cells"]}
    return any((r+dr, c+dc) in cells and grid[r, c] != grid[r+dr, c+dc]
               for r, c in cells for dr, dc in _FOUR)


def _has_independence(a):
    return any(x["color"] == y["color"] and
               max(r for r, c in x["cells"]) != max(r for r, c in y["cells"])
               for i, x in enumerate(a["components"]) for y in a["components"][i+1:])


def _unused(a):
    return len(a["bands"])-len({c["color"] for c in a["components"]})


def _finish(grid, bands, background, stage, rotation):
    # Construction labels use a scan of the final staged masks and known bands;
    # the validator separately rediscovers bands, background and orientation.
    points = [tuple(map(int, p)) for p in np.argwhere(grid[:stage] != background)]
    comps = _components(grid, points)
    out = grid.copy()
    out[:stage] = background
    lookup = {b["route"]: b for b in bands}
    occupied = set()
    for comp in comps:
        band = lookup[comp["color"]]
        shift = band["bottom"]-max(r for r, c in comp["cells"])
        for r, c in comp["cells"]:
            p = (r+shift, c)
            _require(p not in occupied, "construction collision")
            occupied.add(p)
            out[p] = band["rail"]
    return {"input": np.rot90(grid, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def _attempt_pair(rng, level, lesson, unused, background, rotation, avoid_count):
    layout = _layout(rng, level, lesson, unused, background, avoid_count)
    if layout is None:
        return None
    grid, bands, active, stage = layout
    occupied = set()
    if lesson == "split" or (len(active) >= 2 and rng.random() < (.25, .6, .85)[level]):
        group = _frame_group(rng, bands, active, stage, level)
        if not _try_group(grid, group, bands, occupied, background, stage, rng,
                          edge=bool(rng.integers(2))):
            if lesson == "split":
                return None
    if lesson == "independent":
        i = int(rng.choice(active))
        depth = min(stage-1, bands[i]["bottom"]-bands[i]["top"]+1)
        width = max(1, (bands[i]["right"]-bands[i]["left"]-2)//3)
        for j in range(2):
            comp = _mask(rng, depth, width, level)
            existing = np.argwhere(grid[:stage] == bands[i]["route"])
            constraint = ("different", int(existing[:, 0].max())) if j else None
            if not _try_group(grid, [(i, comp)], bands, occupied, background, stage, rng,
                              bottom_constraint=constraint):
                return None
        # A third piece can share either source depth, or use a new depth.
        for _ in range(int(rng.integers(1, 4))):
            existing = np.argwhere(grid[:stage] == bands[i]["route"])
            constraint = ("equal", int(existing[:, 0].max())) if rng.random() < .5 else None
            _try_group(grid, [(i, _mask(rng, depth, width, 0))], bands, occupied,
                       background, stage, rng, bottom_constraint=constraint)
    for i in rng.permutation(active):
        i = int(i)
        if np.any(grid[:stage] == bands[i]["route"]):
            continue
        depth = min(stage, bands[i]["bottom"]-bands[i]["top"]+1)
        width = bands[i]["right"]-bands[i]["left"]-1
        if not _try_group(grid, [(i, _mask(rng, depth, width, level))], bands,
                          occupied, background, stage, rng):
            if not _try_group(grid, [(i, ((0, 0),))], bands, occupied, background, stage, rng):
                return None
    for _ in range(int(rng.integers(1, (4, 8, 15)[level]))):
        if len(active) >= 2 and rng.random() < (.15, .35, .5)[level]:
            group = _frame_group(rng, bands, active, stage, level)
        else:
            i = int(rng.choice(active))
            depth = min(stage, bands[i]["bottom"]-bands[i]["top"]+1)
            width = bands[i]["right"]-bands[i]["left"]-1
            group = [(i, _mask(rng, depth, width, level))]
        _try_group(grid, group, bands, occupied, background, stage, rng)
    pair = _finish(grid, bands, background, stage, rotation)
    try:
        a = _analyze_input(pair["input"])
        points = {p for c in a["components"] for p in c["cells"]}
        if len(_components(a["canonical"], points, True)) != len(a["components"]):
            return None
        if lesson == "split" and not _has_contact(a):
            return None
        if lesson == "independent" and not _has_independence(a):
            return None
    except AssertionError:
        return None
    return pair


def _fallback_pair(rng, lesson, unused, background, rotation, avoid_count, serial):
    """Finite constructive reserve: only used after bounded random attempts."""
    count = next(n for n in (3, 2, 4) if n != avoid_count and (not unused or n >= 3))
    w, stage = 24+serial % 7, 7
    routes, rails = _make_roles(rng, count, background)
    grid = np.full((stage+4*count+1, w), background, dtype=np.int8)
    bands = []
    for i in range(count):
        top = stage+1+4*i
        b = dict(top=top, bottom=top+2, left=0, right=w-1, route=routes[i], rail=rails[i])
        bands.append(b)
        grid[top:top+3] = routes[i]
        grid[top:top+3, [0, w-1]] = rails[i]
    active = count-int(unused)
    order = list(map(int, rng.permutation(active)))
    a, b = order[:2]
    top = serial % 3
    fw = 3+serial % 4
    for r in range(3):
        for c in range(fw):
            grid[top+r, 2+c] = routes[a if r in (0, 2) or c in (0, fw-1) else b]
    grid[0:2, 11:13] = routes[a]
    grid[5, 16:18] = routes[a]
    for j, i in enumerate(order[2:]):
        grid[j+1, 20] = routes[i]
    return _finish(grid, bands, background, stage, rotation)


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _source_copy(inp, out):
    digest = hashlib.sha256(bytes(inp.shape)+inp.astype(np.int8).tobytes()+
                            out.astype(np.int8).tobytes()).hexdigest()
    return digest in _SOURCE_PAIRS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # These policies are shared, but never supply hidden output information.
    orientation = int(rng.choice(3, p=[.35, .4, .25]))
    turns = list(map(int, rng.permutation(4)))
    regime = int(rng.choice(4, p=[.25, .3, .25, .2]))
    unused_enabled = bool(rng.random() < .4)
    puzzle, seen = {"train": [], "test": []}, set()
    # source_train supplies both lessons. target_test recombines familiar,
    # larger, or simpler scenes according to the shared complexity policy.
    for split, count in (("train", train_count), ("test", test_count)):
        previous_count = None
        previous_background = None
        for index in range(count):
            lesson = ("split" if index == 0 else "independent" if index == 1 else None) if split == "train" else None
            if regime == 0:
                level = 0
            elif regime == 1:
                level = int(rng.integers(3))
            elif regime == 2:
                level = int(rng.integers(2)) if split == "train" else 2
            else:
                level = int(rng.integers(1, 3)) if split == "train" else 0
            if orientation == 0:
                turn = turns[0]
            elif orientation == 1:
                turn = turns[index % 4] if split == "train" else int(rng.integers(4))
            else:
                turn = turns[index % 2] if split == "train" else turns[2+index % 2]
            unused = unused_enabled and (split == "train" and index == 1 or
                                        lesson != "split" and rng.random() < .45)
            background = int(rng.choice([c for c in range(10) if c != previous_background]))
            pair = None
            for attempt in range(40):
                candidate = _attempt_pair(rng, level, lesson, unused, background, turn, previous_count)
                if candidate is None:
                    continue
                if _grid_key(candidate["input"]) in seen or _source_copy(candidate["input"], candidate["output"]):
                    continue
                if index and candidate["input"].shape == puzzle[split][-1]["input"].shape:
                    continue
                pair = candidate
                break
            if pair is None:
                # More reserve variants than the maximum number of puzzle pairs.
                for serial in range(28):
                    candidate = _fallback_pair(rng, lesson, unused, background, turn, previous_count, serial)
                    if _grid_key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            _require(pair is not None, "constructive reserve exhausted")
            a = _analyze_input(pair["input"])
            previous_count = len(a["bands"])
            previous_background = a["background"]
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
             len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer), "grid must be a 2D integer array")
                _require(all(1 <= d <= 30 for d in grid.shape) and
                         0 <= int(grid.min()) <= int(grid.max()) <= 9, "grid bounds or colors")
            _require(inp.shape == out.shape, "dimensions must be preserved")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(not _source_copy(inp, out), "exact source pair copy")
            a = _analyze_input(inp)
            _require(np.array_equal(out, a["output"]), "incorrect component routing output")
            _require(not np.array_equal(inp, out), "no moving component")
            points = {p for c in a["components"] for p in c["cells"]}
            eight = _components(a["canonical"], points, True)
            _require(len(eight) == len(a["components"]), "diagonal-only grouping requires a training witness")
            analyses[split].append(a)
    _require(any(_has_contact(a) for a in analyses["train"]), "training must show multicolor splitting")
    _require(any(_has_independence(a) for a in analyses["train"]), "training must show independent alignment")
    if any(_unused(a) for a in analyses["test"]):
        _require(any(_unused(a) for a in analyses["train"]), "unused test band was not demonstrated")
    train_turns = {a["rotation"] for a in analyses["train"]}
    if any(a["rotation"] not in train_turns for a in analyses["test"]):
        _require(len(train_turns) >= 2, "orientation holdout needs varying training directions")
    for split in ("train", "test"):
        if len(analyses[split]) > 1:
            _require(len({len(a["bands"]) for a in analyses[split]}) > 1,
                     "multiple pairs must vary their receptacle counts")
            _require(len({pair["input"].shape for pair in puzzle[split]}) > 1,
                     "multiple pairs must vary their dimensions")
            _require(len({a["background"] for a in analyses[split]}) > 1,
                     "multiple pairs must vary their backgrounds")
    # Background, dimensions, palette identities and band order are visible
    # pair-local evidence, not conventions that must agree across the puzzle.


def _hole_regions(cells):
    cells = set(_shape_signature({"cells": cells}))
    h, w = 1+max(r for r, c in cells), 1+max(c for r, c in cells)
    blank = {(r, c) for r in range(-1, h+1) for c in range(-1, w+1)}-cells
    stack = [(-1, -1)]
    blank.remove(stack[0])
    while stack:
        r, c = stack.pop()
        for dr, dc in _FOUR:
            p = r+dr, c+dc
            if p in blank:
                blank.remove(p)
                stack.append(p)
    outside = {(r, c) for r in range(-1, h+1) for c in range(-1, w+1)}-cells-blank
    holes = []
    while blank:
        stack = [blank.pop()]
        region = set()
        while stack:
            r, c = stack.pop()
            region.add((r, c))
            for dr, dc in _FOUR:
                p = r+dr, c+dc
                if p in blank:
                    blank.remove(p)
                    stack.append(p)
        holes.append(region)
    return holes, outside


def _holes(cells):
    return [len(hole) for hole in _hole_regions(cells)[0]]


def measure_pair(pair):
    """Review features reconstructed from returned grids, without sampled flags."""
    a = _analyze_input(pair["input"])
    comps, bands = a["components"], a["bands"]
    shapes = [_shape_signature(c) for c in comps]
    heights = [b["bottom"]-b["top"]+1 for b in bands]
    topology = [_hole_regions(c["cells"]) for c in comps]
    holes = [[len(hole) for hole in regions] for regions, _ in topology]
    thicknesses, inner_shapes = [], []
    composites = 0
    for comp, (regions, exterior) in zip(comps, topology):
        r0 = min(r for r, c in comp["cells"])
        c0 = min(c for r, c in comp["cells"])
        filled = False
        for hole in regions:
            thicknesses.append(min(abs(r-y)+abs(c-x)-1 for r, c in hole for y, x in exterior))
            points = {(r+r0, c+c0) for r, c in hole
                      if a["canonical"][r+r0, c+c0] != a["background"]}
            filled |= bool(points)
            inner_shapes.extend(_shape_signature(c) for c in _components(a["canonical"], points))
        composites += int(filled)
    dimensions = [(1+max(r for r, c in s), 1+max(c for r, c in s)) for s in shapes]
    asymmetry = [set(s) != {(h-1-r, c) for r, c in s} and
                 set(s) != {(r, w-1-c) for r, c in s} for s, (h, w) in zip(shapes, dimensions)]
    branches = [not hs and len(s) < h*w and
                any(sum((r+dr, c+dc) in set(s) for dr, dc in _FOUR) >= 3 for r, c in s)
                for s, (h, w), hs in zip(shapes, dimensions, holes)]
    groups = {}
    for c in comps:
        groups.setdefault(c["color"], []).append(c)
    same = list(groups.values())
    targets = []
    by_color = {b["route"]: b for b in bands}
    destinations = []
    for c in comps:
        b = by_color[c["color"]]
        offset = b["bottom"]-max(r for r, col in c["cells"])
        targets.append({(r+offset, col) for r, col in c["cells"]})
        destinations.append(bands.index(b))
    bbox_overlap = False
    adjacency = False
    for i, x in enumerate(targets):
        for j in range(i):
            y = targets[j]
            if destinations[i] != destinations[j]:
                continue
            bbox_overlap |= (max(min(r for r, c in x), min(r for r, c in y)) <=
                             min(max(r for r, c in x), max(r for r, c in y)) and
                             max(min(c for r, c in x), min(c for r, c in y)) <=
                             min(max(c for r, c in x), max(c for r, c in y)))
            adjacency |= any((r+dr, c+dc) in y for r, c in x for dr, dc in _FOUR)
    return dict(shape=tuple(pair["input"].shape), background=a["background"],
                direction=a["rotation"], bands=len(bands), depths=tuple(heights),
                gaps=tuple(b["top"]-p["bottom"]-1 for p, b in zip(bands, bands[1:])),
                stage_gap=bands[0]["top"]-max(r for c in comps for r, col in c["cells"])-1,
                far_margin=a["canonical"].shape[0]-1-bands[-1]["bottom"],
                inset=any(b["left"] or b["right"] != a["canonical"].shape[1]-1 for b in bands),
                spans=tuple((b["left"], b["right"]) for b in bands),
                reuse=sum(b["rail"] in by_color for b in bands),
                distinct_reused_colors=len({b["rail"] for b in bands} & set(by_color)),
                shared_rails=len({b["rail"] for b in bands}) < len(bands),
                components=len(comps), max_area=max(map(len, shapes)),
                max_height=max(1+max(r for r, c in s) for s in shapes),
                max_width=max(1+max(c for r, c in s) for s in shapes),
                max_windows=max(map(len, holes)), largest_hole=max([0]+[n for hs in holes for n in hs]),
                frames=sum(bool(hs) for hs in holes),
                composites=composites, max_wall=max([0]+thicknesses),
                inner_shapes=tuple(inner_shapes),
                solid_blocks=sum(len(s) == h*w and min(h, w) >= 2 for s, (h, w) in zip(shapes, dimensions)),
                bars=sum(min(h, w) == 1 and max(h, w) >= 2 for h, w in dimensions),
                branches=sum(branches), asymmetric=sum(asymmetry),
                irregular_frames=sum(bool(hs) and len(s) < h*w-sum(hs)
                                     for s, (h, w), hs in zip(shapes, dimensions, holes)),
                frame_tops=tuple(min(r for r, col in c["cells"]) for c, hs in zip(comps, holes) if hs),
                max_same_color=max(map(len, same)),
                three_equal_depth=any(any(depths.count(d) >= 3 for d in depths) for depths in
                                      ([max(r for r, c in part["cells"]) for part in group] for group in same)),
                three_unequal_depth=any(len({max(r for r, c in part["cells"]) for part in group}) >= 3
                                        for group in same),
                equal_depth=any(len({max(r for r, col in c["cells"]) for c in group}) < len(group) for group in same),
                unequal_depth=_has_independence(a), contact=_has_contact(a), unused=_unused(a),
                target_bbox_overlap=bbox_overlap, target_adjacency=adjacency,
                signatures=tuple(shapes), assignments=tuple((len(c["cells"]), d) for c, d in zip(comps, destinations)))


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for seed, counts in enumerate(((None, None), (6, 3), (2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(300+seed, *counts)
        _require(_puzzles_equal(p, generate_puzzle(300+seed, *counts)), "nondeterminism")
        for s, n in zip(("train", "test"), counts):
            _require(n is None or len(p[s]) == n, "requested count ignored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.5)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    records = []
    train_counts, test_counts, first_keys = set(), set(), set()
    regimes = {name: 0 for name in ("fixed", "mixed", "holdout", "more", "larger", "simpler", "familiar")}
    for seed in range(42):
        p = generate_puzzle(seed)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        first_keys.add(_grid_key(p["train"][0]["input"]))
        per_puzzle = []
        for split in ("train", "test"):
            local = [measure_pair(pair) for pair in p[split]]
            per_puzzle += local
            records += [(split, m) for m in local]
            if len(local) > 1:
                _require(len({m["bands"] for m in local}) > 1, "fixed within-split band count")
                _require(len({m["shape"] for m in local}) > 1, "fixed within-split dimensions")
        _require(len({m["background"] for m in per_puzzle}) > 1, "fixed puzzle background")
        train, test = per_puzzle[:len(p["train"])], per_puzzle[len(p["train"]):]
        directions = {m["direction"] for m in train}
        held_out = bool({m["direction"] for m in test}-directions)
        regimes["fixed"] += len({m["direction"] for m in per_puzzle}) == 1
        regimes["mixed"] += len(directions) > 1 and not held_out
        regimes["holdout"] += held_out
        regimes["more"] += any(m["components"] > max(t["components"] for t in train) for m in test)
        regimes["larger"] += any(m["max_area"] > max(t["max_area"] for t in train) for m in test)
        regimes["simpler"] += all(m["components"] < max(t["components"] for t in train) and
                                 m["max_area"] <= max(t["max_area"] for t in train) for m in test)
        regimes["familiar"] += any(m["bands"] in {t["bands"] for t in train} and
                                  min(t["components"] for t in train) <= m["components"] <=
                                  max(t["components"] for t in train) for m in test)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _require(len(first_keys) == 42, "cross-seed duplicates")
    _require(all(regimes.values()), "missing observed train/test regime")
    for split in ("train", "test"):
        ms = [m for s, m in records if s == split]
        _require({m["direction"] for m in ms} == set(range(4)), "missing direction")
        _require({m["bands"] for m in ms} >= {1, 2, 3, 4, 5}, "narrow band counts")
        _require(len({m["shape"] for m in ms}) >= 35, "narrow canvas distribution")
        _require(len({m["depths"] for m in ms}) >= 35, "narrow depth distribution")
        _require(len({sig for m in ms for sig in m["signatures"]}) >= 60, "narrow masks")
        for name in ("inset", "shared_rails", "equal_depth", "unequal_depth", "contact", "target_adjacency",
                     "target_bbox_overlap", "solid_blocks", "bars", "branches", "asymmetric", "irregular_frames",
                     "three_equal_depth", "three_unequal_depth"):
            _require(any(m[name] for m in ms), "unexercised review feature: "+name)
        _require(any(m["max_area"] > 23 and m["max_width"] >= 7 for m in ms), "missing large masks")
        _require(any(m["max_height"] >= 5 for m in ms), "missing tall masks")
        _require(any(m["max_windows"] >= 2 for m in ms), "missing multiwindow frames")
        _require(any(m["largest_hole"] >= 4 for m in ms), "missing large holes")
        _require(any(m["frames"] >= 2 for m in ms), "missing multiple composites")
        _require(any(m["composites"] >= 2 for m in ms), "missing several multicolored motifs")
        _require(any(m["max_wall"] >= 2 for m in ms), "missing thick frame walls")
        _require(len({s for m in ms for s in m["inner_shapes"]}) >= 10, "fixed inner shapes")
        _require(any(m["max_same_color"] >= 3 for m in ms), "missing repeated route components")
        _require(any(m["reuse"] == 0 for m in ms) and any(m["reuse"] >= 3 for m in ms), "narrow reuse")
        _require(any(m["distinct_reused_colors"] >= 3 for m in ms), "narrow cross-role palette relationships")
        _require(any(m["unused"] for m in ms), "missing unused bands")
        _require(any(0 in m["frame_tops"] for m in ms) and
                 any(any(r > 0 for r in m["frame_tops"]) for m in ms), "fixed frame depth")
        _require(any(max(m["gaps"], default=0) >= 2 for m in ms) and
                 any(m["far_margin"] >= 3 for m in ms), "fixed spacing")
        _require(any(m["inset"] and m["direction"] and m["contact"] and m["reuse"] for m in ms),
                 "variation axes do not combine")
        assignments = {}
        for m in ms:
            for area, destination in m["assignments"]:
                assignments.setdefault(area, set()).add(destination)
        _require(sum(len(destinations) >= 3 for destinations in assignments.values()) >= 4,
                 "component area predicts its destination")
    # Exercise the finite reserve directly, even if random sampling never needs it.
    rng = np.random.default_rng(440)
    for turn in range(4):
        for unused in (False, True):
            first = _fallback_pair(rng, "split", False, 0, turn, None, turn)
            count = len(_analyze_input(first["input"])["bands"])
            second = _fallback_pair(rng, "independent", unused, 1, turn, count, turn+1)
            test = _fallback_pair(rng, None, unused, 2, turn, None, turn+2)
            validate_generated_puzzle({"train": [first, second], "test": [test]})
    # An L and an isolated cell have overlapping target boxes but disjoint masks.
    inp = np.zeros((10, 12), dtype=np.int8)
    inp[7:9, 1:-1] = 1
    inp[7:9, [0, -1]] = 2
    inp[0, 2:4] = 1
    inp[1, 2] = inp[4, 3] = 1
    expected = inp.copy()
    expected[:7] = 0
    expected[7:9, 2:4] = 2
    _require(np.array_equal(_solve_input(inp), expected), "valid overlapping target boxes rejected")
    inp[4, 3], inp[4, 2] = 0, 1
    try:
        _solve_input(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError("actual target collision accepted")
    # A wrong label must be caught by the grid-based validator.
    p = generate_puzzle(991, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        return
    raise AssertionError("validator accepted a corrupted output")
