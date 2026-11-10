"""Laced jigsaw: a taut, counted seam constrains a four-piece square assembly.

Only standard-library modules and NumPy are used. The solver reads the query
pixels alone; generation parameters never participate in reconstruction.
"""
from __future__ import annotations

import itertools
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid colors")
    return g.astype(np.int16, copy=False)


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(cells, directions=D4):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(frozenset(component))
    return result


def _normalize(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _rotate(shape, angle):
    for _ in range(angle % 4):
        shape = _normalize([(c, -r) for r, c in shape])
    return shape


def _dimensions(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _mask(cells, side):
    return sum(1 << (r * side + c) for r, c in cells)


def _local(p, endpoint, forward, lateral):
    dr, dc = p[0] - endpoint[0], p[1] - endpoint[1]
    return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]


def _global(p, endpoint, forward, lateral):
    u, v = p
    return (endpoint[0] + u * forward[0] + v * lateral[0],
            endpoint[1] + u * forward[1] + v * lateral[1])


def _taut_path(obstacle, count):
    """The discrete convex clearance envelope, including its outgoing tangent."""
    _require(obstacle and min(u for u, v in obstacle) >= 1, "obstacle must be ahead of peg")
    peak = max(v for u, v in obstacle)
    _require(peak >= 0, "obstacle must reach the strand side")
    _require(peak == 0 or max(v - u for u, v in obstacle) == 0, "unsupported incoming tangent")
    first_peak = min(u for u, v in obstacle if v == peak)
    far = max(u for u, v in obstacle)
    n = max(count, far + 3)
    path = [(u, 1 if peak == 0 and u <= first_peak else
             max(v + 1 - abs(u - t) for t, v in obstacle)) for u in range(n)]
    slopes = [b[1] - a[1] for a, b in zip(path, path[1:])]
    _require(path[0] == (0, 1), "route must start at attachment")
    _require(all(x in (-1, 0, 1) for x in slopes) and slopes == sorted(slopes, reverse=True),
             "exposed route must be digitally convex")
    return path[:count]


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(bool(np.all(g[0] == bg) and np.all(g[-1] == bg)
                  and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)), "blank background border required")
    by_color = {int(c): frozenset(map(tuple, np.argwhere(g == c).tolist())) for c in np.unique(g) if c != bg}
    singles = [c for c, cells in by_color.items() if len(cells) == 1]
    _require(len(singles) == 4, "four singleton legend colors required")
    key_points = set.union(*(set(by_color[c]) for c in singles))
    kr, kc = min(r for r, c in key_points), min(c for r, c in key_points)
    _require(key_points == {(kr, kc), (kr+1, kc), (kr, kc+1), (kr+1, kc+1)}, "legend must be 2 by 2")
    key = g[kr:kr+2, kc:kc+2].copy()
    marker_candidates = []
    for color, cells in by_color.items():
        comps = _components(cells, D8)
        if len(comps) != 2 or sorted(map(len, comps))[0] != 1:
            continue
        line = max(comps, key=len)
        if len(line) >= 3 and (len({r for r, c in line}) == 1 or len({c for r, c in line}) == 1):
            marker_candidates.append((color, line, next(iter(min(comps, key=len)))))
    _require(len(marker_candidates) == 1, "one peg line and matching obstacle dot required")
    marker, peg, dot = marker_candidates[0]
    positive = (0, 1) if len({r for r, c in peg}) == 1 else (1, 0)
    endpoint_options = ((max(peg), positive), (min(peg), (-positive[0], -positive[1])))
    attachments = []
    for endpoint, forward in endpoint_options:
        for lateral in ((forward[1], -forward[0]), (-forward[1], forward[0])):
            attachment = _global((0, 1), endpoint, forward, lateral)
            for color, cells in by_color.items():
                if color in singles or color == marker or attachment not in cells:
                    continue
                if len(_components(cells, D8)) != 1:
                    continue
                if all(u <= 0 and v >= 1 for u, v in (_local(p, endpoint, forward, lateral) for p in cells)):
                    attachments.append((endpoint, forward, lateral, color, cells))
    _require(len(attachments) == 1, "strand attachment must determine one peg frame")
    endpoint, forward, lateral, rope_color, rope = attachments[0]
    _require((forward, lateral) in (((0, 1), (1, 0)), ((1, 0), (0, 1))), "peg must point right or down, strand on inner side")
    rigid = set().union(*(set(cells) for color, cells in by_color.items() if color != rope_color))
    components = _components(rigid)
    _require(len(components) == 4, "four rigid footprints required")
    anchors = [c for c in components if key_points <= c and peg <= c]
    obstacles = [c for c in components if dot in c]
    _require(len(anchors) == len(obstacles) == 1 and anchors[0] != obstacles[0], "keyed anchor and marked obstacle required")
    anchor, obstacle = anchors[0], obstacles[0]
    ordered = [anchor, obstacle] + [c for c in components if c not in (anchor, obstacle)]
    halo = set()
    for i, cells in enumerate(ordered):
        group = cells | rope if i == 0 else cells
        _require(not group & halo, "scattered piece groups need a background-cell gap")
        halo.update((r+dr, c+dc) for r,c in group for dr in (-1,0,1) for dc in (-1,0,1))
    role_colors = set(singles) | {bg, marker, rope_color}
    for i, cells in enumerate(ordered):
        carriers = {int(g[p]) for p in cells} - role_colors
        _require(len(carriers) == 1, "each rigid piece needs one carrier color")
        allowed = carriers | (set(singles) | {marker} if i == 0 else {marker} if i == 1 else set())
        _require(all(int(g[p]) in allowed for p in cells), "unexpected piece markings")
        _require(len(cells) >= 6, "rigid pieces must have at least six cells")
    ar, ac = min(r for r, c in anchor), min(c for r, c in anchor)
    end = endpoint[0] - ar, endpoint[1] - ac
    axis_start = end[0] if forward[0] else end[1]
    area = len(rigid)
    sides = [s for s in range(1, 31) if s > axis_start and s*s == area + s - axis_start]
    _require(len(sides) == 1, "one square side must fit rigid area plus through-seam")
    return {"grid": g, "background": bg, "key": key, "marker": marker,
            "rope_color": rope_color, "rope": rope, "length": len(rope), "peg": peg,
            "components": ordered, "shapes": [_normalize(c) for c in ordered],
            "endpoint": end, "forward": forward, "lateral": lateral,
            "side": sides[0], "gap": sides[0] - axis_start}


def _corner_options(shape, side, corner):
    corners = {(0, 0), (0, side-1), (side-1, 0), (side-1, side-1)}
    target = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))[corner]
    options = {}
    for angle in range(4):
        rotated = _rotate(shape, angle)
        h, w = _dimensions(rotated)
        if h > side or w > side:
            continue
        dr, dc = (0 if corner < 2 else side-h), (0 if corner % 2 == 0 else side-w)
        cells = frozenset((r+dr, c+dc) for r, c in rotated)
        if cells & corners == {target}:
            bits = _mask(cells, side)
            options.setdefault(bits, (bits, cells, angle))
    return list(options.values())


def _assemblies(shapes, side, endpoint, forward, lateral, gap):
    """Enumerate all incomplete corner packings, then impose the taut seam.

    Bit masks make rejection inexpensive. Identical visible layouts caused by
    shape symmetry or interchangeable plain pieces count as one answer.
    """
    anchor = shapes[0]
    corners = {(0, 0), (0, side-1), (side-1, 0), (side-1, side-1)}
    if max(_dimensions(anchor)) > side or anchor & corners != {(0, 0)}:
        return [], 0, 0
    anchor_bits = _mask(anchor, side)
    full = (1 << (side*side)) - 1
    tables = {(i, j): _corner_options(shapes[i], side, j) for i in range(1, 4) for j in range(1, 4)}
    incomplete, channels, accepted = set(), set(), {}
    for order in itertools.permutations((1, 2, 3)):
        for choices in itertools.product(*(tables[i, j] for i, j in zip(order, (1, 2, 3)))):
            occupied, good = anchor_bits, True
            for bits, cells, angle in choices:
                if occupied & bits:
                    good = False
                    break
                occupied |= bits
            if not good:
                continue
            masks = (anchor_bits,) + tuple(c[0] for c in choices)
            incomplete.add(masks)
            # A stronger A-ablation witness: the wrong complement can already
            # be a connected through-channel with the correct attachment.
            holes_bits = full ^ occupied
            hole_points = []
            while holes_bits:
                low = holes_bits & -holes_bits
                index = low.bit_length()-1
                hole_points.append((index//side, index%side))
                holes_bits ^= low
            local_holes = sorted(_local(p, endpoint, forward, lateral) for p in hole_points)
            if ([u for u,v in local_holes] == list(range(gap)) and local_holes[0] == (0,1)
                    and all(abs(b[1]-a[1]) <= 1 for a,b in zip(local_holes,local_holes[1:]))):
                channels.add(masks)
            obstacle = choices[order.index(1)][1]
            local_obstacle = {_local(p, endpoint, forward, lateral) for p in obstacle}
            try:
                local_path = _taut_path(local_obstacle, gap)
            except AssertionError:
                continue
            path = [_global(p, endpoint, forward, lateral) for p in local_path]
            if any(not (0 <= r < side and 0 <= c < side) for r, c in path):
                continue
            if _mask(path, side) != full ^ occupied:
                continue
            accepted.setdefault(masks, {"masks": masks, "path": path, "local_obstacle": local_obstacle,
                                        "rotations": (0,) + tuple(c[2] for c in choices),
                                        "piece_for_corner": (0,) + order})
    return list(accepted.values()), len(incomplete), len(channels)


def _recover(input_grid):
    a = _parse(input_grid)
    solutions, incomplete, channels = _assemblies(a["shapes"], a["side"], a["endpoint"], a["forward"], a["lateral"], a["gap"])
    _require(len(solutions) == 1, "one visible assembly must satisfy the taut seam")
    a.update(solutions[0])
    a["incomplete_packings"] = incomplete
    a["channel_packings"] = channels
    return a


def _render_answer(a):
    side = a["side"]
    out = np.full((side, side), a["background"], dtype=np.int16)
    flat = out.ravel()
    for color, bits in zip(a["key"].ravel(), a["masks"]):
        while bits:
            low = bits & -bits
            flat[low.bit_length()-1] = color
            bits ^= low
    for p in a["path"][:a["length"]]:
        out[p] = a["rope_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict using only this input's shapes, markings, key, and strand count."""
    return _render_answer(_recover(input_grid))


def _partition(rng, side, ambiguous=False):
    """A convex channel with three independently varied adjoining interfaces."""
    start = int(rng.integers(3, side-3))
    gap = side-start
    up = int(rng.integers(1, min(4, (gap-1)//2)+1))
    down = int(rng.integers(1, min(up, gap-1-up)+1))
    flat = gap-1-up-down
    profile = [1]
    for step in [1]*up + [0]*flat + [-1]*down:
        profile.append(profile[-1]+step)
    if ambiguous:
        # Swapping and half-turning the two right-slab pieces leaves a second
        # continuous channel. Balance their heights so it still meets the peg.
        rows = [r for r in range(2, side-up-2)
                if abs(side-2*r-profile[-1]-2) <= 1]
        if not rows:
            return None
        row = int(rng.choice(rows))
        peg_length = int(rng.integers(3, start+1))
    else:
        row = int(rng.integers(2, side-up-2))
        peg_length = int(rng.integers(3, start+2))
    owner = np.full((side, side), 3, dtype=np.int16)
    owner[:row+1, :start+1] = 0
    owner[row+1:, :start+1] = 2
    for u, v in enumerate(profile):
        y, x = row+v, start+u
        if u:
            owner[:y, x] = 1
        owner[y, x] = -1
    # Notches in the lower anchor boundary, leaving the peg intact.
    for x in range(max(0, start-peg_length+1)):
        cut = int(rng.integers(max(2, row-2), row+2))
        owner[cut:row+1, x] = 2
    if ambiguous:
        owner[row, 0] = 2  # asymmetric anchor and a complementary left interlock
    # Interlocks above the obstacle's exposed face do not change its envelope.
    for y in range(0 if ambiguous else max(0, row-1)):
        extra = int(rng.integers(0, max(1, (gap-1)//2)))
        owner[y, start+1:start+1+extra] = 0
    # Independent lower boundary between the two bottom corner pieces.
    for y in range(side if ambiguous else row+max(profile)+1, side):
        cut = int(rng.integers(max(1, start-2), min(side-1, start+4)))
        owner[y, :cut], owner[y, cut:] = 2, 3
    shapes = [_normalize(map(tuple, np.argwhere(owner == i).tolist())) for i in range(4)]
    if any(len(s) < 6 or len(_components(s)) != 1 for s in shapes):
        return None
    if any(owner[p] != i for i, p in enumerate(((0, 0), (0, side-1), (side-1, 0), (side-1, side-1)))):
        return None
    peg = {(row, x) for x in range(start-peg_length+1, start+1)}
    return owner, peg, (row, start)


def _slack(rng, length, start, diagonal=False):
    """Connected unary encodings, independent of the assembled contour."""
    if diagonal:
        cells = {(0, 1), (-1, 2)}
        allowed = {(u, v) for u in range(-min(start, 6), 0)
                   for v in range(2, 3+(length-1)//min(start, 6))}
        directions = D8
    else:
        cells = {(0, 1)}
        width = min(start+1, int(rng.integers(2, 7)))
        depth = 1 + (length+width-1)//width
        allowed = {(u, v) for u in range(-width+1, 1) for v in range(1, depth+1)}
        directions = D4 if rng.random() < .5 else D8
    dense = rng.random() < .5
    while len(cells) < length:
        frontier = sorted({(u+du, v+dv) for u, v in cells for du, dv in directions} & allowed - cells)
        _require(bool(frontier), "slack growth exhausted")
        neighbors = np.array([sum((u+du, v+dv) in cells for du, dv in D8) for u, v in frontier], dtype=float)
        weights = (neighbors+1)**2 if dense else 1/(neighbors+1)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
    return cells


def _scatter(rng, patches):
    """Sample arbitrary ordered shelves, on either axis, with variable padding."""
    dims = [_dimensions(p) for p in patches]
    layouts = []
    gap = int(rng.integers(1, 4))
    for spacing in (gap, 1):
        for transpose in (False, True):
            ds = [(w, h) if transpose else (h, w) for h, w in dims]
            for order in itertools.permutations(range(4)):
                for cuts in range(8):
                    groups, group = [], []
                    for j, i in enumerate(order):
                        group.append(i)
                        if j == 3 or (cuts >> j) & 1:
                            groups.append(group)
                            group = []
                    widths = [sum(ds[i][1] for i in g)+spacing*(len(g)-1) for g in groups]
                    heights = [max(ds[i][0] for i in g) for g in groups]
                    h, w = sum(heights)+spacing*(len(groups)-1), max(widths)
                    if h <= 28 and w <= 28:
                        layouts.append((transpose, ds, groups, widths, heights, h, w, spacing))
        if layouts:
            break
    if not layouts:
        return None
    trans, ds, groups, widths, heights, h, w, spacing = layouts[int(rng.integers(len(layouts)))]
    offsets, row = [None]*4, 0
    for group, width, height in zip(groups, widths, heights):
        col = int(rng.integers(w-width+1))
        for i in group:
            rr = row+int(rng.integers(height-ds[i][0]+1))
            offsets[i] = (col, rr) if trans else (rr, col)
            col += ds[i][1]+spacing
        row += height+spacing
    h, w = (w, h) if trans else (h, w)
    pads = []
    for size in (h, w):
        room = 30-size
        before = int(rng.integers(1, min(4, room)))
        after = int(rng.integers(1, min(5, room-before+1)))
        pads.append((before, after))
    offsets = [(r+pads[0][0], c+pads[1][0]) for r, c in offsets]
    return offsets, (h+sum(pads[0]), w+sum(pads[1]))


def _make_pair(rng, side, axis, lesson=None, full=False):
    for attempt in range(1200):
        data = _partition(rng, side, ambiguous=lesson is not None)
        if data is None:
            continue
        owner, peg, endpoint = data
        if axis:
            owner = owner.T.copy()
            top_right, bottom_left = owner == 1, owner == 2
            owner[top_right], owner[bottom_left] = 2, 1
            peg = {(c, r) for r, c in peg}
            endpoint = endpoint[1], endpoint[0]
        forward, lateral = ((1, 0), (0, 1)) if axis else ((0, 1), (1, 0))
        obstacle_role = 2 if axis else 1
        # Solver ordering: anchor, marked obstacle, two remaining pieces.
        roles = [0, obstacle_role] + [i for i in range(1, 4) if i != obstacle_role]
        absolute = [frozenset(map(tuple, np.argwhere(owner == i).tolist())) for i in roles]
        shapes = [_normalize(c) for c in absolute]
        gap = side-(endpoint[0] if axis else endpoint[1])
        assemblies, incomplete, channels = _assemblies(shapes, side, endpoint, forward, lateral, gap)
        if len(assemblies) != 1 or (lesson is not None and channels < 2):
            continue
        expected = tuple(_mask(np.argwhere(owner == i).tolist(), side) for i in range(4))
        if assemblies[0]["masks"] != expected:
            continue
        if lesson == "long":
            orbits = [{_rotate(sh, k) for k in range(4)} for sh in shapes]
            if any(len(orbit) < 4 for orbit in orbits):
                continue
            if any(orbits[i] & orbits[j] for i in range(4) for j in range(i)):
                continue
        if lesson == "long":
            length = gap+int(rng.integers(1, 5))
        elif lesson == "short":
            length = int(rng.integers(2, gap))
        elif full:
            length = gap
        else:
            length = int(rng.integers(2, gap+5))
        slack = _slack(rng, length, endpoint[0] if axis else endpoint[1], diagonal=lesson == "long" or rng.random() < .3)
        rope = {_global(p, endpoint, forward, lateral) for p in slack}
        if rope & absolute[0]:
            continue
        # A key can sit anywhere a 2 by 2 patch avoids the peg.
        key_sites = [(r, c) for r in range(side-1) for c in range(side-1)
                     if {(r,c), (r+1,c), (r,c+1), (r+1,c+1)} <= absolute[0] - peg]
        if not key_sites:
            continue
        kr, kc = key_sites[int(rng.integers(len(key_sites)))]
        key_points = [(kr, kc), (kr, kc+1), (kr+1, kc), (kr+1, kc+1)]
        dot_sites = sorted(absolute[1])
        dot = dot_sites[int(rng.integers(len(dot_sites)))]
        colors = list(map(int, rng.permutation(10)))
        bg, marker, rope_color = colors[:3]
        key = np.array(colors[3:7], dtype=np.int16).reshape(2, 2)
        carrier_count = int(rng.integers(1, 4))
        carriers = [colors[7+i % carrier_count] for i in range(4)]
        rng.shuffle(carriers)
        rotations = [0] + (list(map(int, rng.permutation([1, 2, 3]))) if lesson == "long"
                           else list(map(int, rng.integers(0, 4, 3))))
        patches = []
        for i, cells in enumerate(absolute):
            patch = {p: carriers[i] for p in cells}
            if i == 0:
                patch.update({p: marker for p in peg})
                patch.update({p: int(c) for p, c in zip(key_points, key.ravel())})
                patch.update({p: rope_color for p in rope})
            elif i == 1:
                patch[dot] = marker
            # Rotate all cells of each nonanchor piece, including its dot.
            for _ in range(rotations[i]):
                patch = {(c, -r): value for (r, c), value in patch.items()}
            r0, c0 = min(r for r,c in patch), min(c for r,c in patch)
            patches.append({(r-r0,c-c0):value for (r,c),value in patch.items()})
        layout = _scatter(rng, patches)
        if layout is None:
            continue
        offsets, shape = layout
        inp = np.full(shape, bg, dtype=np.int16)
        for patch, (dr, dc) in zip(patches, offsets):
            for (r, c), value in patch.items():
                inp[r+dr,c+dc] = value
        out = np.full((side, side), bg, dtype=np.int16)
        for i in range(4):
            out[owner == i] = key.ravel()[i]
        path = assemblies[0]["path"]
        for p in path[:length]:
            out[p] = rope_color
        # Source footprints and the independently reconstructed input must agree.
        try:
            recovered = _recover(inp)
            if not np.array_equal(_render_answer(recovered), out):
                continue
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded search could not construct a uniquely laced square")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    fixed = int(rng.integers(8, 14))
    first_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if regime == 0:
                side = int(rng.integers(8, 11) if split == "train" else rng.integers(11, 14))
            elif regime == 1:
                side = int(rng.integers(11, 14) if split == "train" else rng.integers(8, 11))
            elif regime == 2:
                side = fixed
            else:
                side = int(rng.integers(8, 14))
            axis = (first_axis if i == 0 else 1-first_axis) if split == "train" and i < 2 else int(rng.integers(2))
            lesson = ("long" if i == 0 else "short") if split == "train" and i < 2 else None
            for retry in range(50):
                pair = _make_pair(rng, side, axis, lesson, full=split == "test" and regime in (2, 3))
                sig = _signature(pair["input"])
                keys_differ = split != "train" or i != 1 or not np.array_equal(_parse(pair["input"])["key"], _parse(puzzle["train"][0]["input"])["key"])
                if sig not in seen and keys_differ:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate inputs")
            seen.add(sig)
            a = _recover(inp)
            _require(np.array_equal(_render_answer(a), out), "input-only solution disagrees")
            _require(np.count_nonzero(out == a["rope_color"]) == min(a["length"], a["gap"]), "count/clipping invariant")
            _require(sum(map(len, a["shapes"]))+a["gap"] == a["side"]**2, "rigid area plus channel")
            _require(len(a["path"]) == a["gap"] and len(set(a["path"])) == a["gap"], "simple through seam")
            _require(len(set(a["key"].ravel())) == 4, "distinct spatial key entries")
            _require(set(map(int, np.unique(out))) <= set(map(int, a["key"].ravel())) | {a["background"], a["rope_color"]}, "temporary markings must disappear")
            # Four corner footprints plus the entire conceptual seam cover once.
            occupied = 0
            for bits in a["masks"]:
                _require(not occupied & bits, "overlapping rigid pieces")
                occupied |= bits
            _require((occupied | _mask(a["path"], a["side"])) == (1 << (a["side"]**2))-1, "incomplete square envelope")
            if split == "train":
                training.append(a)
    _require({a["forward"] for a in training} == {(0, 1), (1, 0)}, "both peg axes must be demonstrated")
    _require(sum(a["incomplete_packings"] >= 2 for a in training) >= 2,
             "two demonstrations must require the taut route to disambiguate rigid layouts")
    _require(sum(a["channel_packings"] >= 2 for a in training) >= 2,
             "two demonstrations must distinguish taut routing from an arbitrary connected through-channel")
    _require(any(a["length"] > a["gap"] for a in training), "clipping lesson required")
    _require(any(a["length"] < a["gap"] for a in training), "partial strand lesson required")
    _require(any(len(_components(a["rope"], D4)) > 1 for a in training), "eight-connected count lesson required")
    _require(len({tuple(a["key"].ravel()) for a in training}) >= 2, "key colors must change in training")
    _require(any(set(a["rotations"][1:]) == {1, 2, 3} for a in training), "quarter-turn assembly lesson required")


def self_test():
    """Bounded structural, semantic, counterfactual, and 42-seed variation checks."""
    measurements = {"sides": set(), "canvases": set(), "backgrounds": set(), "lengths": set(),
                    "areas": set(), "profiles": set(), "keys": set(), "rotations": set(),
                    "axes": set(), "incomplete": set(), "channels": set(), "inputs": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _recover(pair["input"])
                measurements["sides"].add(a["side"])
                measurements["canvases"].add(pair["input"].shape)
                measurements["backgrounds"].add(a["background"])
                measurements["lengths"].add(a["length"])
                measurements["areas"].add(tuple(sorted(map(len, a["shapes"]))))
                measurements["profiles"].add(tuple(_local(p, a["endpoint"], a["forward"], a["lateral"]) for p in a["path"]))
                measurements["keys"].add(tuple(a["key"].ravel()))
                measurements["rotations"].update(a["rotations"][1:])
                measurements["axes"].add((split, a["forward"]))
                measurements["incomplete"].add(a["incomplete_packings"])
                measurements["channels"].add(a["channel_packings"])
                measurements["inputs"].add(_signature(pair["input"]))
        # B ablation: permuting the visible legend must change the rigid colors.
        q = puzzle["test"][0]
        a = _parse(q["input"])
        c1, c2 = map(int, a["key"].ravel()[:2])
        changed = q["input"].copy()
        changed[q["input"] == c1], changed[q["input"] == c2] = c2, c1
        swapped = solve(changed)
        expected = q["output"].copy()
        expected[q["output"] == c1], expected[q["output"] == c2] = c2, c1
        _require(np.array_equal(swapped, expected) and not np.array_equal(swapped, q["output"]), "visible legend counterfactual")
        # A ablation: one extra slack cell changes the partial prefix by one.
        q = puzzle["train"][1]
        a = _parse(q["input"])
        anchor = a["components"][0]
        ar, ac = min(r for r,c in anchor), min(c for r,c in anchor)
        endpoint = a["endpoint"][0]+ar, a["endpoint"][1]+ac
        options = sorted({(r+dr,c+dc) for r,c in a["rope"] for dr,dc in D8} - set(a["rope"]))
        # Re-scatter unchanged rigid footprints when adding the slack cell.
        # Tight input margins must not limit this semantic counterfactual.
        probe_rng = np.random.default_rng(10000+seed)
        for p in options:
            r, c = p
            u, v = _local(p, endpoint, a["forward"], a["lateral"])
            if p not in anchor and u <= 0 and v >= 2:
                patches = []
                for i,cells in enumerate(a["components"]):
                    patch = {q:int(a["grid"][q]) for q in cells}
                    if i == 0:
                        patch.update({q:a["rope_color"] for q in a["rope"] | {p}})
                    r0, c0 = min(r for r,c in patch), min(c for r,c in patch)
                    patches.append({(r-r0,c-c0):color for (r,c),color in patch.items()})
                layout = _scatter(probe_rng, patches)
                if layout is None:
                    continue
                offsets, shape = layout
                changed = np.full(shape, a["background"], dtype=np.int16)
                for patch,(dr,dc) in zip(patches,offsets):
                    for (r,c),color in patch.items():
                        changed[r+dr,c+dc] = color
                try:
                    extended = solve(changed)
                except AssertionError:
                    continue
                break
        else:
            raise AssertionError("no room for unary counterfactual")
        delta = extended != q["output"]
        _require(np.count_nonzero(delta) == 1, "unary length must change one seam cell")
    _require(measurements["sides"] == set(range(8, 14)), "square scale diversity")
    _require(len(measurements["canvases"]) >= 40 and len(measurements["areas"]) >= 100, "structural diversity")
    _require(len(measurements["profiles"]) >= 18 and len(measurements["lengths"]) >= 8, "contour and budget diversity")
    _require({max(v for u,v in p) for p in measurements["profiles"]} >= {2, 3, 4},
             "seams must include several transverse reaches")
    _require({len(p) for p in measurements["profiles"]} == set(range(4, 11)), "seam length diversity")
    _require(len(measurements["backgrounds"]) == 10 and len(measurements["keys"]) >= 100, "palette role diversity")
    _require(measurements["rotations"] == {0, 1, 2, 3}, "rotation diversity")
    _require(len(measurements["axes"]) == 4, "both seam axes in both splits")
    _require(len(measurements["inputs"]) == 210, "cross-seed duplicate input")
    p1, p2 = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in p1 for p,q in zip(p1[s],p2[s]) for k in p), "determinism")
    # The supplied verifier exhausts every legal count combination. Keep this
    # repeated self-test bounded by exercising the count boundaries here.
    for nt,nq in ((2,1), (10,1), (10,4)):
        p = generate_puzzle(2026+nt*10+nq, nt, nq)
        validate_generated_puzzle(p)
        _require((len(p["train"]),len(p["test"])) == (nt,nq), "explicit counts")
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s:[{k:np.array(v,copy=True) for k,v in q.items()} for q in p1[s]] for s in p1}
    damaged["test"][0]["output"][0,0] = (int(damaged["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return {"seeds": 42, "pairs": 210, "variation": {k:len(v) for k,v in measurements.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
