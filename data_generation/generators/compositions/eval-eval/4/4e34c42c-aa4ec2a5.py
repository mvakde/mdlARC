"""Sealed-socket assembly: topological windows are translation coordinates.

Only NumPy and the standard library are used. Neither the solver nor validator
uses generation coordinates. Constants 0/8/9 are background/neutral fill/halo;
the wall color and all registration inks are visible in each input.
"""
from collections import defaultdict
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid is not integral")
    _require(bool(((0 <= a) & (a <= 9)).all()), "invalid color")
    return a


def _groups(points):
    todo, groups = set(points), []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _bounds(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normal(cells):
    r0, c0, _, _ = _bounds(cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _move(cells, offset):
    dr, dc = offset
    return {(r+dr, c+dc): int(v) for (r, c), v in cells.items()}


def _merge(parts):
    union = {}
    for part in parts:
        for p, v in part.items():
            _require(p not in union or union[p] == v, "conflicting foreground overlap")
            union[p] = v
    return union


def _holes(cells):
    """Four-connected empty components in a padded object-local rectangle."""
    r0, c0, r1, c1 = _bounds(cells)
    empty = {(r, c) for r in range(r0-1, r1+2) for c in range(c0-1, c1+2)
             if (r, c) not in cells}
    return [group for group in _groups(empty)
            if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in group)]


def _clear_topology(cells):
    """Exclude alternating 2x2 contacts, so no diagonal-only leak is queried."""
    r0, c0, r1, c1 = _bounds(cells)
    mask = np.zeros((r1-r0+3, c1-c0+3), bool)
    for r, c in cells:
        mask[r-r0+1, c-c0+1] = True
    a, b = mask[:-1, :-1], mask[:-1, 1:]
    c, d = mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _rim(hole):
    return {(r+dr, c+dc) for r, c in hole for dr, dc in D4} - set(hole)


def _socket_signature(hole, cells, wall):
    r0 = min(r for r, c in hole)
    c0 = min(c for r, c in hole)
    stamps = [(r-r0, c-c0, cells[r, c]) for r, c in _rim(hole)
              if (r, c) in cells and cells[r, c] != wall]
    if len(stamps) < 2:
        return None
    return (tuple(sorted((r-r0, c-c0) for r, c in hole)), tuple(sorted(stamps)))


def _parse(grid):
    grid = _grid(grid)
    _require(not np.isin(grid, [8, 9]).any(), "input uses reserved output colors")
    _require(np.count_nonzero(grid == 0) * 2 > grid.size, "background must be dominant zero")
    values, counts = np.unique(grid[grid != 0], return_counts=True)
    _require(len(values) >= 2 and np.count_nonzero(counts == counts.max()) == 1,
             "wall color must be the unique most frequent nonzero color")
    wall = int(values[counts.argmax()])
    parts = [_normal({p: int(grid[p]) for p in group})
             for group in _groups(map(tuple, np.argwhere(grid != 0).tolist()))]
    _require(2 <= len(parts) <= 8, "need two to eight separated fragments")
    occurrences = defaultdict(list)
    part_holes = []
    for i, part in enumerate(parts):
        holes = _holes(part)
        part_holes.append(holes)
        seen = set()
        for hole in holes:
            signature = _socket_signature(hole, part, wall)
            if signature is None:
                continue
            _require(signature not in seen, "one fragment has two indistinguishable sealed sockets")
            seen.add(signature)
            anchor = min(r for r, c in hole), min(c for r, c in hole)
            occurrences[signature].append((i, anchor))
        _require(seen, "every fragment needs a sealed marked socket")
    return wall, parts, part_holes, occurrences


def _assemble(grid):
    """Align *all* repeated sealed windows, with no overlap/area scoring."""
    wall, parts, part_holes, occurrences = _parse(grid)
    edges = [[] for _ in parts]
    for items in occurrences.values():
        if len(items) < 2:
            continue
        i, a = items[0]
        for j, b in items[1:]:
            delta = a[0]-b[0], a[1]-b[1]
            edges[i].append((j, delta))
            edges[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in edges[i]:
            candidate = offsets[i][0]+dr, offsets[i][1]+dc
            if j in offsets:
                _require(offsets[j] == candidate, "inconsistent socket coordinates")
            else:
                offsets[j] = candidate
                stack.append(j)
    _require(len(offsets) == len(parts), "socket graph is disconnected")
    placed = [_move(part, offsets[i]) for i, part in enumerate(parts)]
    union = _merge(placed)
    old_holes = []
    for i, holes in enumerate(part_holes):
        dr, dc = offsets[i]
        for hole in holes:
            translated = {(r+dr, c+dc) for r, c in hole}
            _require(not translated & union.keys(), "a fragment fills an observed sealed window")
            old_holes.append(translated)
    _require(len(_groups(union)) == 1, "assembled foreground must be connected")
    r0, c0, r1, c1 = _bounds(union)
    _require(r1-r0+3 <= 30 and c1-c0+3 <= 30, "assembled halo exceeds ARC bounds")
    return wall, parts, offsets, placed, union, old_holes, occurrences


def _finish(union, wall):
    """Fill the assembly's chambers, then outline only its pre-fill exterior."""
    r0, c0, r1, c1 = _bounds(union)
    out = np.zeros((r1-r0+3, c1-c0+3), dtype=np.int8)
    for (r, c), v in union.items():
        out[r-r0+1, c-c0+1] = v
    holes = _holes(union)
    all_holes = set().union(*holes) if holes else set()
    for hole in holes:
        inks = {union[p] for p in _rim(hole) if p in union and union[p] != wall}
        fill = next(iter(inks)) if len(inks) == 1 else 8
        for r, c in hole:
            out[r-r0+1, c-c0+1] = fill
    for r, c in union:
        for dr, dc in D8:
            p = r+dr, c+dc
            if p not in union and p not in all_holes:
                out[p[0]-r0+1, p[1]-c0+1] = 9
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Entirely input-only; the color convention is fixed or pixel-visible."""
    wall, _, _, _, union, _, _ = _assemble(input_grid)
    return _finish(union, wall)


def _room(rng, wall, ink, core=False):
    """A thick/stepped chamber with an ink bar on its bottom inner rim."""
    h = int(rng.integers(5, 8 if core else 7))
    w = int(rng.integers(5, 8 if core else 7))
    hole = {(r, c) for r in range(1, h-1) for c in range(1, w-1)}
    kind = int(rng.integers(5))
    if kind == 1:
        a, b = int(rng.integers(2, h-1)), int(rng.integers(2, w-1))
        hole -= {(r, c) for r in range(1, a) for c in range(b, w-1)}
    elif kind == 2:
        # Thick roof and an off-center narrow pocket; bottom rim stays visible.
        start = int(rng.integers(1, h-2))
        hole = {(r, c) for r, c in hole if r >= start}
    elif kind == 3:
        hole = {(r, c) for r, c in hole if c < w-2 or r == h-2}
    elif kind == 4:
        # A stair whose width grows toward its bottom run.
        hole = {(r, c) for r, c in hole if c <= min(w-2, 1+r) or r == h-2}
    room = {(r, c): wall for r in range(h) for c in range(w) if (r, c) not in hole}
    length = int(rng.integers(2, min(4, w-2)+1))
    start = int(rng.integers(1, w-length))
    for c in range(start, start+length):
        room[h-1, c] = ink
    # All extra keys have independently chosen orientation, unlike the core's
    # paired open/closed copies. Translation never rotates any input fragment.
    if not core:
        room = _orient_cells(room, int(rng.integers(4)), bool(rng.integers(2)))
    return room


def _orient_cells(cells, rotation, flip):
    cells = dict(cells)
    if flip:
        cells = {(r, -c): v for (r, c), v in cells.items()}
    for _ in range(rotation):
        cells = {(c, -r): v for (r, c), v in cells.items()}
    return _normal(cells)


def _core(rng, wall, ink, close):
    primary = _room(rng, wall, ink, core=True)
    _, _, h1, w1 = _bounds(primary)
    h, w = h1+1, w1+1
    gap = int(rng.integers(1, 4))
    shift = w+gap
    # The deceptive bay can share the socket's whole silhouette or only its
    # ink stamp. Unequal size and a perpendicular offset make the eventual
    # interstitial chamber stepped rather than always a rectangular slot.
    bay_local = dict(primary)
    if rng.random() < .55:
        for _ in range(20):
            candidate = _room(rng, wall, ink, core=True)
            if _stamp_pattern(candidate, wall) == _stamp_pattern(primary, wall):
                bay_local = candidate
                break
    _, _, bh1, bw1 = _bounds(bay_local)
    bh, bw = bh1+1, bw1+1
    stagger = int(rng.integers(-2, 3))
    bay = _move(bay_local, (stagger, shift))
    original_hole = _holes(bay_local)[0]
    if close:
        col = min(c for r, c in original_hole)
        top = min(r for r, c in original_hole if c == col)
        for r in range(top+1):
            bay.pop((r+stagger, col+shift), None)
    else:
        row = min(r for r, c in original_hole)
        right = max(c for r, c in original_hole if r == row)
        for c in range(right, bw):
            bay.pop((row+stagger, c+shift), None)
    lower_path = (_line((h-1, w-1), (h-1, shift))
                  | _line((h-1, shift), (stagger+bh-1, shift)))
    upper_path = (_line((0, w-1), (0, shift-1))
                  | _line((0, shift-1), (stagger, shift-1))
                  | _line((stagger, shift-1), (stagger, shift+bw-1)))
    lower = {p: primary.get(p, bay.get(p, wall)) for p in lower_path}
    upper = {p: primary.get(p, wall) for p in upper_path}
    # The lower piece contains the deceptive open copy; the upper piece seals
    # it only if its door lies on the top rim. Both are connected to socket 0.
    first = _merge([primary, bay, lower])
    second = _merge([primary, upper])
    _require(len(_holes(first)) == 1 and len(_holes(second)) == 1, "bad open core")
    return [first, second], [primary], [(0, 0, h-1, w-1),
                                      (stagger, shift, stagger+bh-1, shift+bw-1)]


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal path")
    return {(r, c) for r in range(min(a[0], b[0]), max(a[0], b[0])+1)
            for c in range(min(a[1], b[1]), max(a[1], b[1])+1)}


def _stamp_pattern(cells, wall):
    inked = {p: v for p, v in cells.items() if v != wall}
    normalized = _normal(inked)
    return tuple(sorted((r, c, v) for (r, c), v in normalized.items()))


def _add_node(rng, parts, rooms, boxes, wall, ink, signatures, same_stamp=False):
    """Attach a new socket by an independently bent path to a random parent."""
    base = _merge(parts)
    protected = set().union(*_holes(base))
    for _ in range(100):
        local = _room(rng, wall, ink)
        holes = _holes(local)
        signature = _socket_signature(holes[0], local, wall)
        if signature is None or signature in signatures:
            continue
        if same_stamp and (_stamp_pattern(local, wall) != _stamp_pattern(rooms[0], wall)
                           or any(signature[0] == s[0] for s in signatures)):
            continue
        parent = rooms[int(rng.integers(len(rooms)))]
        r0, c0, r1, c1 = _bounds(parent)
        _, _, nh1, nw1 = _bounds(local)
        nh, nw = nh1+1, nw1+1
        direction = int(rng.integers(4))
        gap = int(rng.integers(1, 5))
        if direction < 2:
            top = r0 + int(rng.integers(-3, 4))
            left = c1+gap+1 if direction == 0 else c0-gap-nw
        else:
            left = c0 + int(rng.integers(-3, 4))
            top = r1+gap+1 if direction == 2 else r0-gap-nh
        box = top, left, top+nh-1, left+nw-1
        if any(not (box[2] < a-1 or box[0] > d+1 or box[3] < b-1 or box[1] > e+1)
               for a, b, d, e in boxes):
            continue
        room = _move(local, (top, left))
        if direction < 2:
            a = (int(rng.integers(r0, r1+1)), c1 if direction == 0 else c0)
            b = (int(rng.integers(top, top+nh)), left if direction == 0 else left+nw-1)
            turn = (a[1]+b[1])//2
            path = _line(a, (a[0], turn)) | _line((a[0], turn), (b[0], turn)) | _line((b[0], turn), b)
        else:
            a = (r1 if direction == 2 else r0, int(rng.integers(c0, c1+1)))
            b = (top if direction == 2 else top+nh-1, int(rng.integers(left, left+nw)))
            turn = (a[0]+b[0])//2
            path = _line(a, (turn, a[1])) | _line((turn, a[1]), (turn, b[1])) | _line((turn, b[1]), b)
        # New edges must not alter a core bay/door, enter any original chamber,
        # or accidentally run through a third socket. Touching plain walls is OK.
        core_bay = boxes[1]
        if any(core_bay[0] <= r <= core_bay[2] and core_bay[1] <= c <= core_bay[3]
               and (r, c) not in parent for r, c in path):
            continue
        if path & protected or any(p in base and p not in parent for p in room):
            continue
        new_hole = set().union(*_holes(room))
        if new_hole & base.keys():
            continue
        connector = {p: base.get(p, room.get(p, wall)) for p in path}
        try:
            edge = _merge([parent, room, connector])
            union = _merge(parts+[edge])
        except AssertionError:
            continue
        er0, ec0, er1, ec1 = _bounds(edge)
        ur0, uc0, ur1, uc1 = _bounds(union)
        if max(er1-er0+1, ec1-ec0+1) > 18 or min(er1-er0+1, ec1-ec0+1) > 10:
            continue
        if max(ur1-ur0+3, uc1-uc0+3) > 29:
            continue
        if len(_groups(edge)) != 1 or len(_holes(edge)) != 2:
            continue
        # Only the two intended socket signatures may be present in this view.
        actual = {_socket_signature(h, edge, wall) for h in _holes(edge)}
        expected = {_socket_signature(h, parent, wall) for h in _holes(parent)} | {signature}
        if actual != expected:
            continue
        parts.append(edge)
        rooms.append(room)
        boxes.append(box)
        signatures.add(signature)
        return
    raise AssertionError("could not place an additional socket")


def _pack(rng, parts):
    parts = [_normal(p) for p in parts]
    masks = []
    for part in parts:
        _, _, r1, c1 = _bounds(part)
        mask = np.zeros((r1+1, c1+1), bool)
        for p in set(part).union(*_holes(part)):
            mask[p] = True
        masks.append(mask)
    mh, mw = max(m.shape[0] for m in masks), max(m.shape[1] for m in masks)
    area = sum(len(p) for p in parts)
    canvases = [(int(rng.integers(max(mh, 8), 31)), int(rng.integers(max(mw, 8), 31)))
                for _ in range(12)] + [(30, 30)] * 14
    for h, w in canvases:
        if h*w <= 2*area:
            continue
        occupied = np.zeros((h, w), bool)
        grid = np.zeros((h, w), np.int8)
        order = sorted(range(len(parts)), key=lambda i: -(masks[i].size + float(rng.random())*30))
        for i in order:
            mask, part = masks[i], parts[i]
            ph, pw = mask.shape
            fh, fw = h-ph+1, w-pw+1
            valid = np.ones((fh, fw), bool)
            for r, c in np.argwhere(mask):
                valid &= ~occupied[r:r+fh, c:c+fw]
            options = np.argwhere(valid)
            if not len(options):
                break
            top, left = map(int, options[int(rng.integers(len(options)))])
            for (r, c), v in part.items():
                grid[r+top, c+left] = v
            for r, c in np.argwhere(mask):
                r, c = int(r)+top, int(c)+left
                occupied[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
        else:
            return grid
    raise AssertionError("fragments do not fit in a separated canvas")


def _make_pair(rng, wall, colors, nodes, close, redundant):
    for _ in range(100):
        try:
            ink = int(rng.choice(colors))
            parts, rooms, boxes = _core(rng, wall, ink, close)
            signatures = {_socket_signature(_holes(rooms[0])[0], rooms[0], wall)}
            for index in range(1, nodes):
                # Same-color/different-shape sockets deliberately refute color-only alignment.
                color = ink if index == 1 else int(rng.choice(colors))
                _add_node(rng, parts, rooms, boxes, wall, color, signatures, same_stamp=index == 1)
            if redundant:
                parts.append(dict(rooms[-1]))
            union = _merge(parts)
            if not all(_clear_topology(p) for p in parts+[union]):
                continue
            rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
            # Apply one common rigid coordinate transformation to the whole scene.
            # No input piece is independently rotated in the solution.
            all_points = _orient_cells(union, rotation, flip)
            transformed = []
            def orient_raw(part):
                if flip:
                    part = {(r, -c): v for (r, c), v in part.items()}
                for _ in range(rotation):
                    part = {(c, -r): v for (r, c), v in part.items()}
                return part
            for part in parts:
                transformed.append(orient_raw(part))
            inp = _pack(rng, transformed)
            predicted = solve(inp)
            intended = _finish(all_points, wall)
            if not np.array_equal(predicted, intended):
                continue
            return {"input": inp, "output": intended}
        except AssertionError:
            continue
    raise RuntimeError("bounded socket-scene construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside allowed range")
    return int(value)


def _key(grid):
    return tuple(np.asarray(grid).shape), np.asarray(grid, dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(1, 8))))
    wall, colors = palette[0], palette[1:]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train" and i == 0:
                nodes, close, redundant = 1, True, False
            elif split == "train" and i == 1:
                nodes, close, redundant = 2, False, True
            elif split == "train":
                nodes = int(rng.integers(1, 3))
                close, redundant = bool(rng.integers(2)), bool(rng.integers(2))
            else:
                nodes = 3
                close, redundant = bool(rng.integers(2)), bool(rng.integers(2))
            for _ in range(20):
                pair = _make_pair(rng, wall, colors, nodes, close, redundant)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    wall, parts, offsets, placed, union, old, occurrences = _assemble(inp)
    _require(all(_clear_topology(p) for p in parts+[union]), "ambiguous diagonal topology")
    _require(np.array_equal(_finish(union, wall), out), "incorrect cavity fill, union, crop, or halo")
    holes = _holes(union)
    new = [h for h in holes if not any(h == earlier for earlier in old)]
    new_marked = [h for h in new if len({union[p] for p in _rim(h) if p in union and union[p] != wall}) == 1]
    neutral = [h for h in holes if len({union[p] for p in _rim(h) if p in union and union[p] != wall}) != 1]
    contributions = [len(set(p)-set().union(*(set(q) for j, q in enumerate(placed) if i != j)))
                     for i, p in enumerate(placed)]
    # A repeated ink bar within one piece, with only one sealed occurrence,
    # is a concrete witness against unrestricted stamp matching.
    misleading = False
    for part in parts:
        patterns = defaultdict(int)
        for ink in set(part.values())-{wall}:
            for group in _groups(p for p, v in part.items() if v == ink):
                r0, c0, _, _ = _bounds(group)
                pattern = ink, tuple(sorted((r-r0, c-c0) for r, c in group))
                patterns[pattern] += 1
        misleading |= any(n > 1 for n in patterns.values())
    all_rims = set().union(*(_rim(h) for h in holes))
    final_stamps = defaultdict(list)
    for ink in set(union.values())-{wall}:
        for group in _groups(p for p, v in union.items() if v == ink):
            r0, c0, _, _ = _bounds(group)
            pattern = ink, tuple(sorted((r-r0, c-c0) for r, c in group))
            final_stamps[pattern].append(bool(group & all_rims))
    open_stamp = any(True in statuses and False in statuses for statuses in final_stamps.values())
    sigs = list(occurrences)
    def rim_pattern(signature):
        cells = {(r, c): v for r, c, v in signature[1]}
        return tuple(sorted((r, c, v) for (r, c), v in _normal(cells).items()))
    different_windows = any(a[0] != b[0] and rim_pattern(a) == rim_pattern(b)
                            for i, a in enumerate(sigs) for b in sigs[i+1:])
    _require(new and neutral and misleading, "missing interacting topology/registration evidence")
    # Inspect output zones directly in union coordinates. These disjoint zones
    # make accidental foreground overwrites and interior halo painting visible.
    r0, c0, _, _ = _bounds(union)
    for (r, c), value in union.items():
        _require(out[r-r0+1, c-c0+1] == value, "body or ink changed")
    for hole in holes:
        inks = {union[p] for p in _rim(hole) if p in union and union[p] != wall}
        color = next(iter(inks)) if len(inks) == 1 else 8
        _require(all(out[r-r0+1, c-c0+1] == color for r, c in hole), "incomplete chamber fill")
    body_mask = np.zeros(out.shape, bool)
    hole_mask = np.zeros(out.shape, bool)
    for r, c in union:
        body_mask[r-r0+1, c-c0+1] = True
    for hole in holes:
        for r, c in hole:
            hole_mask[r-r0+1, c-c0+1] = True
    padded = np.pad(body_mask, 1)
    h, w = out.shape
    dilated = np.logical_or.reduce([padded[1+dr:1+dr+h, 1+dc:1+dc+w] for dr, dc in D8])
    halo = dilated & ~body_mask & ~hole_mask
    far = ~body_mask & ~hole_mask & ~halo
    _require(np.all(out[halo] == 9), "halo must be exterior and include diagonal neighbors")
    _require(np.all(out[far] == 0), "distant exterior or open bay background changed")
    return {"wall": wall, "fragments": len(parts), "socket_signatures": set(occurrences),
            "offsets": list(offsets.values()), "contributions": contributions,
            "new_chambers": len(new), "new_marked_chambers": len(new_marked),
            "neutral_chambers": len(neutral), "open_stamp": open_stamp,
            "different_windows": different_windows,
            "holes": len(holes), "union": union, "parts": parts,
            "misleading": misleading}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(_inspect(pair))
    train = infos["train"]
    _require(len({i["wall"] for i in infos["train"]+infos["test"]}) == 1,
             "generated puzzle changes wall convention")
    _require(any(0 in i["contributions"] for i in train), "absorption not demonstrated")
    _require(any(i["different_windows"] for i in train), "identical stamps on different windows not demonstrated")
    _require(any(i["new_marked_chambers"] for i in train), "newly sealed marked bay not demonstrated")
    _require(any(i["open_stamp"] for i in train), "permanent open bay not demonstrated")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded pixel checks, metamorphic regressions, and 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed repetition")
    pairs = [pair for p in puzzles for s in p for pair in p[s]]
    infos = [_inspect(p) for p in pairs]
    _require(len({p["input"].shape for p in pairs}) >= 30, "input dimension diversity")
    _require(len({p["output"].shape for p in pairs}) >= 35, "assembly dimension diversity")
    _require(len({i["fragments"] for i in infos}) >= 4, "fragment-count diversity")
    _require(len({i["wall"] for i in infos}) == 7, "wall palette diversity")
    _require(len(set().union(*(i["socket_signatures"] for i in infos))) >= 100, "socket geometry diversity")
    _require(any(len({r for r, c in i["offsets"]}) > 1 and len({c for r, c in i["offsets"]}) > 1
                 for i in infos), "two-axis registration missing")
    # Explicit count boundaries, deterministic replay, and malformed counts.
    for nt, nq in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(700+nt+nq, nt, nq)
        _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts not honored")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterminism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Orientation equivariance and visible palette permutation. Output colors
    # 8/9/0 remain fixed; every other color can be permuted consistently.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k)),
                     "rotation changes rule")
        _require(np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"])),
                 "reflection changes rule")
        mapping = np.array([0, 7, 6, 5, 4, 3, 2, 1, 8, 9], dtype=np.int8)
        _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "palette changes rule")
    # Removing A leaves separated pieces; removing B leaves unfilled chambers
    # and no halo. Neither ablation matches even the first teaching example.
    pair = puzzles[0]["train"][0]
    info = _inspect(pair)
    separated = {tuple(p): int(pair["input"][tuple(p)]) for p in np.argwhere(pair["input"] != 0)}
    _require(not np.array_equal(_finish(separated, info["wall"]), pair["output"]), "A ablation has no effect")
    raw = np.zeros_like(pair["output"])
    r0, c0, _, _ = _bounds(info["union"])
    for (r, c), v in info["union"].items():
        raw[r-r0+1, c-c0+1] = v
    _require(not np.array_equal(raw, pair["output"]), "B ablation has no effect")
    # Repacking cannot change the answer. Redundant fragments are absorbed.
    repacked = _pack(np.random.default_rng(934), info["parts"])
    _require(np.array_equal(solve(repacked), pair["output"]), "input placement encodes answer")
    # Zero, one, and multiple rim inks have specified, distinct fill semantics.
    square = {(r, c): 5 for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4)}
    for colors, expected in (((), 8), ((2,), 2), ((2, 3), 8)):
        body = dict(square)
        for c, ink in enumerate(colors, 1):
            body[0, c] = ink
        _require(_finish(body, 5)[2, 2] == expected, "rim-color tie behavior")
    # Corrupt each output role and require rejection by the returned-pixel check.
    for color in (info["wall"], 8, 9):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        grid = bad["train"][0]["output"]
        r, c = np.argwhere(grid == color)[0]
        grid[r, c] = (color+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
