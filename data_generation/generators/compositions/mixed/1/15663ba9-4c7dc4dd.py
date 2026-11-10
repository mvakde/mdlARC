"""A complete framed example teaches the bend labels of a missing companion.

Only standard-library modules and NumPy are used. Generation constructs geometry;
solve reads the six frames and the demonstrated relation afresh from input pixels.
No files, cached examples, RNG state, or parent modules are consulted by solve.
"""
from collections import deque
import hashlib
import numbers

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CHECK_SEEDS = (0, 99689973, 32548222, 350715681, 906061169, 790256854,
               558824764, 150022667, 730742991, 557179638, 817366421,
               157253247, 346240298, 616008428, 739999634, 13055259,
               849983470, 64735619, 53327334, 947433000, 906576325,
               246824873, 900238282, 628689567, 635448860, 47060842,
               857735696, 164247056, 247471294, 544393808, 911979595,
               180806237, 674538774, 531072496, 661759820, 400178611,
               136109748, 742244066, 275830893, 890613132, 51034118,
               965492889)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid cells must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _trace(cells):
    neighbors = {p: sorted((p[0]+dr, p[1]+dc) for dr, dc in _DIRS
                            if (p[0]+dr, p[1]+dc) in cells) for p in cells}
    _require(len(cells) >= 8 and all(len(v) == 2 for v in neighbors.values()),
             "each contour must be a simple orthogonal cycle")
    start = min(cells)
    path, seen = [start], {start}
    previous, current = start, neighbors[start][0]
    while current != start:
        _require(current not in seen, "cycle does not close simply")
        path.append(current)
        seen.add(current)
        following = next(p for p in neighbors[current] if p != previous)
        previous, current = current, following
    _require(len(path) == len(cells), "disconnected cycle")
    return path


def _has_interior(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells) - r0 + 3
    w = max(c for r, c in cells) - c0 + 3
    blocked = {(r-r0+1, c-c0+1) for r, c in cells}
    outside, queue = {(0, 0)}, deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            p = (r+dr, c+dc)
            if (0 <= p[0] < h and 0 <= p[1] < w
                    and p not in blocked and p not in outside):
                outside.add(p)
                queue.append(p)
    return len(outside) + len(blocked) < h*w


def _classes(cells):
    """0 straight, 1 convex, 2 concave, relative to this loop's own interior."""
    path = _trace(cells)
    _require(_has_interior(cells), "contour must enclose at least one cell")
    area = sum(c*path[(i+1) % len(path)][0] - r*path[(i+1) % len(path)][1]
               for i, (r, c) in enumerate(path))
    _require(area != 0, "zero-area contour")
    kinds = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        turn = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        kinds[(r, c)] = 0 if turn == 0 else 1 if turn*area > 0 else 2
    _require(sum(k == 1 for k in kinds.values()) -
             sum(k == 2 for k in kinds.values()) == 4, "invalid turn balance")
    return kinds, path


def _panel_geometry(panel, neutral):
    components = _components(panel != neutral)
    _require(bool(components), "occupied panel is empty")
    all_kinds, paths = {}, []
    for cells in components:
        kinds, path = _classes(cells)
        all_kinds.update(kinds)
        paths.append(path)
    # Distinct loops must have a full neutral gap, and may not nest.
    for i, path in enumerate(paths):
        for other in paths[i+1:]:
            _require(all(max(abs(r-s), abs(c-t)) > 1
                         for r, c in path for s, t in other), "touching loops")
            _require(not _point_inside(path, other[0][0]+.1, other[0][1]+.1)
                     and not _point_inside(other, path[0][0]+.1, path[0][1]+.1),
                     "nested contours are excluded")
    return all_kinds, paths


def _point_inside(path, row, col):
    # Used for validation, independently of the signed-turn classifier.
    crossings = 0
    for i, (r, c) in enumerate(path):
        nr, nc = path[(i+1) % len(path)]
        if c == nc and ((r > row) != (nr > row)) and c > col:
            crossings += 1
    return crossings % 2 == 1


def _rectangle_components(grid):
    boxes = []
    for color in np.unique(grid):
        for cells in _components(grid == color):
            r0 = min(r for r, c in cells)
            c0 = min(c for r, c in cells)
            r1 = max(r for r, c in cells)
            c1 = max(c for r, c in cells)
            h, w = r1-r0+1, c1-c0+1
            if h < 5 or w < 5 or len(cells) != 2*h+2*w-4:
                continue
            if all(r in (r0, r1) or c in (c0, c1) for r, c in cells):
                boxes.append((r0, c0, h, w, int(color)))
    return sorted(boxes)


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r+h < y+bh and c+w < x+bw


def _crop(grid, box):
    r, c, h, w = box[:4]
    return grid[r+1:r+h-1, c+1:c+w-1].copy()


def _scene(input_grid):
    grid = _grid(input_grid)
    boxes = _rectangle_components(grid)
    groups = [b for b in boxes if not any(_inside(b, a) for a in boxes)]
    _require(len(groups) == 2, "expected exactly two outer grouping frames")
    panels = []
    for group in groups:
        inner = [b for b in boxes if _inside(b, group)]
        immediate = [b for b in inner if not any(_inside(b, a) for a in inner)]
        _require(len(immediate) == 2, "each group must have two framed panels")
        panels.append(sorted(immediate))
    _require(len({b[2:4] for pair in panels for b in pair}) == 1,
             "all four panel dimensions must match")
    interiors = [[_crop(grid, b) for b in pair] for pair in panels]
    blanks = [(g, s) for g in range(2) for s in range(2)
              if np.all(interiors[g][s] == interiors[g][s][0, 0])]
    _require(len(blanks) == 1, "expected a unique uniform missing panel")
    target_group, blank_slot = blanks[0]
    neutral = int(interiors[target_group][blank_slot][0, 0])
    target = interiors[target_group][1-blank_slot]
    reference = interiors[1-target_group]
    raw_slots = [s for s in range(2)
                 if len(set(map(int, np.unique(reference[s]))) - {neutral}) == 1]
    _require(len(raw_slots) == 1, "reference must have one unlabelled before-state")
    raw_slot = raw_slots[0]
    before, after = reference[raw_slot], reference[1-raw_slot]
    raw_colors = set(map(int, np.unique(before))) - {neutral}
    target_colors = set(map(int, np.unique(target))) - {neutral}
    _require(len(raw_colors) == len(target_colors) == 1,
             "before and occupied target must each have one contour color")
    raw_color, target_color = next(iter(raw_colors)), next(iter(target_colors))
    kinds, ref_paths = _panel_geometry(before, neutral)
    target_kinds, target_paths = _panel_geometry(target, neutral)
    _require(set(kinds.values()) == {0, 1, 2},
             "reference must demonstrate both bend signs and straight preservation")
    labels = {}
    for kind in (1, 2):
        values = {int(after[p]) for p, k in kinds.items() if k == kind}
        _require(len(values) == 1, "inconsistent demonstrated bend label")
        labels[kind] = values.pop()
    _require(labels[1] != labels[2]
             and not ({labels[1], labels[2]} & {neutral, raw_color, target_color}),
             "the two bend labels must be distinct from background and contour roles")
    expected_ref = _paint(before, kinds, labels)
    _require(np.array_equal(expected_ref, after),
             "reference must preserve all straight and background cells")
    frame_colors = {b[4] for b in groups + [b for pair in panels for b in pair]}
    content_colors = {neutral, raw_color, target_color, *labels.values()}
    _require(not (frame_colors & content_colors), "frames use separate color roles")
    # No external clutter or hidden control markers: all non-frame/non-panel
    # pixels must have the neutral role visibly supplied by the blank.
    covered = np.zeros(grid.shape, dtype=bool)
    for r, c, h, w, color in groups:
        covered[r, c:c+w] = covered[r+h-1, c:c+w] = True
        covered[r:r+h, c] = covered[r:r+h, c+w-1] = True
    for pair in panels:
        for r, c, h, w, color in pair:
            covered[r:r+h, c:c+w] = True
    _require(np.all(grid[~covered] == neutral), "unexpected cells outside the panels")
    return dict(grid=grid, neutral=neutral, groups=groups, panels=panels,
                target_group=target_group, blank_slot=blank_slot, raw_slot=raw_slot,
                before=before, after=after, target=target, labels=labels,
                ref_kinds=kinds, target_kinds=target_kinds,
                ref_paths=ref_paths, target_paths=target_paths)


def _paint(panel, kinds, labels):
    output = panel.copy()
    for cell, kind in kinds.items():
        if kind:
            output[cell] = labels[kind]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the local labels, reconstruct the blank, and return its interior."""
    scene = _scene(input_grid)
    return _paint(scene["target"], scene["target_kinds"], scene["labels"])


def _raster(vertices):
    cells = []
    for i, (r, c) in enumerate(vertices):
        nr, nc = vertices[(i+1) % len(vertices)]
        dr, dc = nr-r, nc-c
        if (dr == 0) == (dc == 0):
            return None
        length = abs(dr)+abs(dc)
        dr, dc = dr//length, dc//length
        cells.extend((r+t*dr, c+t*dc) for t in range(length))
    if len(set(cells)) != len(cells):
        return None
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    normalized = [(r-r0, c-c0) for r, c in cells]
    try:
        kinds, path = _classes(set(normalized))
    except ValueError:
        return None
    mask = np.zeros((max(r for r, c in normalized)+1,
                     max(c for r, c in normalized)+1), dtype=bool)
    for p in normalized:
        mask[p] = True
    return mask, kinds


def _partition(rng, total, lower):
    return [a+int(b) for a, b in zip(lower, rng.multinomial(total-sum(lower),
                                                        [1/len(lower)]*len(lower)))]


def _stair_vertices(rng, concaves, height, width):
    # Top/right bars retain thickness >= 2. Intermediate staircase edges may
    # have unit length, so adjacent bends are genuine supported family members.
    dx = _partition(rng, width-1, [2]+[1]*concaves)
    dy = _partition(rng, height-1, [1]*concaves+[2])
    vertices = [(0, 0), (0, width-1), (height-1, width-1)]
    r, c = height-1, width-1
    for i in range(concaves):
        c -= dx[i]
        vertices.append((r, c))
        r -= dy[i]
        vertices.append((r, c))
    vertices.append((r, 0))
    return vertices


def _edit_vertices(rng, vertices):
    i = int(rng.integers(len(vertices)))
    previous, p, q = vertices[i-1], vertices[i], vertices[(i+1) % len(vertices)]
    length = abs(q[0]-p[0])+abs(q[1]-p[1])
    if rng.random() < .55:
        back = abs(previous[0]-p[0])+abs(previous[1]-p[1])
        if min(back, length) < 2:
            return None
        a, b = int(rng.integers(1, back)), int(rng.integers(1, length))
        u = ((previous[0]-p[0])//back, (previous[1]-p[1])//back)
        v = ((q[0]-p[0])//length, (q[1]-p[1])//length)
        first = (p[0]+a*u[0], p[1]+a*u[1])
        last = (p[0]+b*v[0], p[1]+b*v[1])
        middle = (first[0]+b*v[0], first[1]+b*v[1])
        return vertices[:i]+[first, middle, last]+vertices[i+1:]
    if length < 3:
        return None
    a = int(rng.integers(1, length-1))
    b = int(rng.integers(a+1, length))
    dr, dc = (q[0]-p[0])//length, (q[1]-p[1])//length
    depth = int(rng.choice((-2, -1, 1, 2)))
    off = (dc*depth, -dr*depth)
    first, last = (p[0]+a*dr, p[1]+a*dc), (p[0]+b*dr, p[1]+b*dc)
    return vertices[:i+1]+[first, (first[0]+off[0], first[1]+off[1]),
                           (last[0]+off[0], last[1]+off[1]), last]+vertices[i+1:]


def _shape(rng, max_h, max_w, low, high):
    goal = int(rng.integers(low, min(high, min(max_h, max_w)-3)+1))
    h, w = int(rng.integers(goal+3, max_h+1)), int(rng.integers(goal+3, max_w+1))
    vertices = _stair_vertices(rng, goal, h, w)
    built = _raster(vertices)
    _require(built is not None, "invalid staircase construction")
    # Bounded local modifications admit notches, shoulders and arms on several
    # sides; a valid staircase remains available if every edit is rejected.
    for _ in range(18):
        candidate = _edit_vertices(rng, vertices)
        if candidate is None:
            continue
        result = _raster(candidate)
        if result is None:
            continue
        mask, kinds = result
        concaves = sum(k == 2 for k in kinds.values())
        if (mask.shape[0] <= max_h and mask.shape[1] <= max_w
                and low <= concaves <= high):
            vertices, built = candidate, result
    mask = built[0]
    choices = [np.rot90(mask, k) for k in range(4)]
    choices = [m for m in choices if m.shape[0] <= max_h and m.shape[1] <= max_w]
    mask = choices[int(rng.integers(len(choices)))].copy()
    if rng.integers(2):
        mask = np.fliplr(mask).copy()
    return mask


def _single_panel(rng, shape, neutral, foreground, low, high):
    mask = _shape(rng, *shape, low, high)
    h, w = mask.shape
    r = int(rng.integers(shape[0]-h+1))
    c = int(rng.integers(shape[1]-w+1))
    panel = np.full(shape, neutral, dtype=np.int8)
    panel[r:r+h, c:c+w][mask] = foreground
    return panel


def _multi_panel(rng, shape, neutral, foreground, count):
    h, w = shape
    ah, aw = int(rng.integers(5, h-3)), int(rng.integers(5, w-3))
    main = _shape(rng, ah, aw, 1, 2)
    mh, mw = main.shape
    # Use fixed non-overlapping regions, then vary actual loops and offsets.
    panel = np.full(shape, neutral, dtype=np.int8)
    panel[:mh, :mw][main] = foreground
    right_h = int(rng.integers(3, min(h, 6)+1))
    right_w = int(rng.integers(3, w-mw))
    right_r = int(rng.integers(h-right_h+1))
    right_c = int(rng.integers(mw+1, w-right_w+1))
    rect = np.zeros((right_h, right_w), dtype=bool)
    rect[0] = rect[-1] = True
    rect[:, 0] = rect[:, -1] = True
    panel[right_r:right_r+right_h, right_c:right_c+right_w][rect] = foreground
    if count == 3:
        bh = int(rng.integers(3, h-mh))
        bw = int(rng.integers(3, mw+1))
        br = int(rng.integers(mh+1, h-bh+1))
        bc = int(rng.integers(mw-bw+1))
        rect = np.zeros((bh, bw), dtype=bool)
        rect[0] = rect[-1] = True
        rect[:, 0] = rect[:, -1] = True
        panel[br:br+bh, bc:bc+bw][rect] = foreground
    # Whole-panel D4 symmetries preserve dimensions when applicable.
    if rng.integers(2):
        panel = panel[::-1].copy()
    if rng.integers(2):
        panel = panel[:, ::-1].copy()
    if h == w and rng.integers(2):
        panel = panel.T.copy()
    return panel


def _outline(grid, box, color):
    r, c, h, w = box[:4]
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _layout(rng, shape):
    ph, pw = shape[0]+2, shape[1]+2
    feasible = []
    for a in (0, 1):  # within-group axis: 0 horizontal, 1 vertical
        for b in (0, 1):
            minima = [(ph+2, 2*pw+3) if axis == 0 else (2*ph+3, pw+2)
                      for axis in (a, b)]
            for packing in (0, 1):
                cross = 1-packing
                mh = sum(x[packing] for x in minima)+1
                mw = max(x[cross] for x in minima)
                if max(mh, mw) <= 30:
                    feasible.append(((a, b), minima, packing))
    axes, minima, packing = feasible[int(rng.integers(len(feasible)))]
    cross = 1-packing
    extent = [0, 0]
    extent[packing] = int(rng.integers(sum(x[packing] for x in minima)+1, 31))
    extent[cross] = int(rng.integers(max(x[cross] for x in minima), 31))
    slack = extent[packing]-sum(x[packing] for x in minima)-1
    extra = rng.multinomial(slack, [1/5]*5)
    spans = [minima[0][packing]+int(extra[1]), minima[1][packing]+int(extra[3])]
    starts = [int(extra[0]), int(extra[0])+spans[0]+1+int(extra[2])]
    groups, panels = [], []
    for g, axis in enumerate(axes):
        size = list(minima[g])
        size[packing] = spans[g]
        size[cross] = int(rng.integers(size[cross], extent[cross]+1))
        start = [0, 0]
        start[packing] = starts[g]
        start[cross] = int(rng.integers(extent[cross]-size[cross]+1))
        groups.append((*start, *size))
        along = 1 if axis == 0 else 0
        perp = 1-along
        panel_size = (ph, pw)
        free = size[along]-2*panel_size[along]-2
        gap = int(rng.integers(1, free+1))
        first = int(rng.integers(1, free-gap+2))
        positions = []
        for offset in (first, first+panel_size[along]+gap):
            pos = list(start)
            pos[along] += offset
            pos[perp] += int(rng.integers(1, size[perp]-panel_size[perp]))
            positions.append((*pos, ph, pw))
        panels.append(positions)
    return tuple(extent), groups, panels


def _make_pair(rng, palette, swap, test):
    colors = list(map(int, palette))
    convex, concave = (colors[1], colors[0]) if swap else colors[:2]
    neutral, ref_fg, target_fg = colors[2:5]
    if rng.random() < .3:
        target_fg = ref_fg
    frames = [colors[5], colors[5] if rng.random() < .4 else colors[6]]
    group_colors = [colors[7], colors[7] if rng.random() < .35 else colors[8]]
    if test:
        shape = (int(rng.integers(9, 11)), int(rng.integers(9, 11)))
    else:
        shape = (int(rng.integers(6, 10)), int(rng.integers(6, 10)))
    reference = _single_panel(rng, shape, neutral, ref_fg, 1, 3)
    ref_kinds, _ = _panel_geometry(reference, neutral)
    labels = {1: convex, 2: concave}
    after = _paint(reference, ref_kinds, labels)
    if test and rng.random() < .55:
        target = _multi_panel(rng, shape, neutral, target_fg, int(rng.integers(2, 4)))
    else:
        target = _single_panel(rng, shape, neutral, target_fg,
                               3 if test else 1, 5 if test else 2)
    # Copying the worked panel must not solve the target, even by coincidence.
    for _ in range(8):
        if not np.array_equal(target != neutral, reference != neutral):
            break
        target = _single_panel(rng, shape, neutral, target_fg,
                               3 if test else 1, 5 if test else 2)
    _require(not np.array_equal(target != neutral, reference != neutral),
             "target must have its own geometry")
    target_kinds, _ = _panel_geometry(target, neutral)
    output = _paint(target, target_kinds, labels)
    extent, groups, panels = _layout(rng, shape)
    grid = np.full(extent, neutral, dtype=np.int8)
    complete = int(rng.integers(2))
    ref_slots = [reference, after]
    target_slots = [target, np.full(shape, neutral, dtype=np.int8)]
    if rng.integers(2):
        ref_slots.reverse()
    if rng.integers(2):
        target_slots.reverse()
    contents = [None, None]
    contents[complete], contents[1-complete] = ref_slots, target_slots
    for g, group in enumerate(groups):
        _outline(grid, group, group_colors[g])
        for box, interior in zip(panels[g], contents[g]):
            _outline(grid, box, frames[g])
            r, c, h, w = box
            grid[r+1:r+h-1, c+1:c+w-1] = interior
    return {"input": grid, "output": output}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        _require(isinstance(value, numbers.Integral)
                 and not isinstance(value, (bool, np.bool_)) and lo <= value <= hi,
                 f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    _require(isinstance(seed, numbers.Integral)
             and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = rng.permutation(10)
    strength = float(rng.choice((0., .5, 1.)))
    first_swap = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            palette = base.copy()
            # The first two examples retain the same two label colors but
            # reverse their meanings. Later examples may remap all roles.
            if split == "train" and index < 2:
                swap = first_swap if index == 0 else not first_swap
                if rng.random() < strength:
                    palette[2:] = rng.permutation(palette[2:])
            else:
                if rng.random() < strength:
                    palette = rng.permutation(10)
                swap = bool(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, palette, swap, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input attempts exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, mappings = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input/output keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            scene = _scene(grid)
            target, kinds = scene["target"], scene["target_kinds"]
            _require(output.shape == target.shape, "return only the blank interior")
            _require(not np.array_equal(target != scene["neutral"],
                                        scene["before"] != scene["neutral"]),
                     "reference copying shortcut is possible")
            expected = _paint(target, kinds, scene["labels"])
            _require(np.array_equal(expected, output), "incorrect reconstructed companion")
            _require(np.array_equal(output == scene["neutral"], target == scene["neutral"]),
                     "contour occupancy and all neutral cells must be preserved")
            for p, kind in kinds.items():
                _require(output[p] == (scene["labels"][kind] if kind else target[p]),
                         "incorrect bend or straight-cell color")
            _require({0, 1, 2} <= set(kinds.values()), "both signs and straight cells required")
            # A second geometric check evaluates the four quadrants around
            # each bend using ray intersections, without turn signs or area.
            for paths, classified in ((scene["ref_paths"], scene["ref_kinds"]),
                                      (scene["target_paths"], kinds)):
                for path in paths:
                    for r, c in path:
                        kind = classified[(r, c)]
                        if kind:
                            n = sum(_point_inside(path, r+dr, c+dc)
                                    for dr in (-.25, .25) for dc in (-.25, .25))
                            _require(n == (1 if kind == 1 else 3), "bend geometry mismatch")
            if split == "train":
                mappings.append((scene["labels"][1], scene["labels"][2]))
            else:
                _require(len(scene["target_paths"]) > 1 or
                         sum(k == 2 for k in kinds.values()) >= 3,
                         "test must shift contour count or concavity complexity")
    _require(mappings[0] == mappings[1][::-1],
             "first two demonstrations must disprove fixed color meanings")


def _canonical_turns(path, kinds):
    turns = [kinds[p] for p in path if kinds[p]]
    return min(tuple(x[i:]+x[:i]) for x in (turns, turns[::-1])
               for i in range(len(x)))


def self_test():
    """42 streaming seed checks plus causal, equivariance and negative probes.

    No sample corpus or log is written. Only digests and small feature sets are
    retained, and at most two puzzles coexist for the determinism check.
    """
    canvases, panel_shapes, targets, turn_patterns, mappings, orders = (set() for _ in range(6))
    concaves, loop_counts, puzzle_hashes = set(), set(), set()
    adjacent = edge = 0
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _scene(pair["input"])
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                canvases.add(pair["input"].shape)
                panel_shapes.add(scene["target"].shape)
                targets.add(hashlib.sha256((scene["target"] != scene["neutral"]).tobytes()).digest())
                mappings.add(tuple(scene["labels"].values()))
                orders.add((scene["target_group"], scene["blank_slot"], scene["raw_slot"]))
                loop_counts.add(len(scene["target_paths"]))
                for path in scene["target_paths"]:
                    kinds = scene["target_kinds"]
                    concaves.add(sum(kinds[p] == 2 for p in path))
                    turn_patterns.add(_canonical_turns(path, kinds))
                    adjacent += any(kinds[p] and kinds[path[(i+1) % len(path)]]
                                    for i, p in enumerate(path))
                    h, w = scene["target"].shape
                    edge += any(r in (0, h-1) or c in (0, w-1) for r, c in path)
        puzzle_hashes.add(digest.digest())
    _require(len(puzzle_hashes) == 42, "seed diversity failure")
    _require(len(canvases) >= 20 and len(panel_shapes) >= 8 and len(targets) >= 150,
             "insufficient geometry variation")
    _require({1, 2, 3} <= loop_counts and {0, 1, 2, 3, 4, 5} <= concaves,
             "missing contour counts or bend complexities")
    _require(len(turn_patterns) >= 8 and len(mappings) >= 20 and len(orders) == 8
             and adjacent > 0 and edge > 0, "missing structural variation")
    del puzzle
    first = generate_puzzle(0, 2, 1)
    again = generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in first
                 for x, y in zip(first[s], again[s]) for k in ("input", "output")),
             "seed reproducibility failure")
    del again
    # Strong causal twin: only the labelled reference changes. The occupied
    # target and the two frame groups stay byte-for-byte identical.
    pair = first["train"][0]
    scene = _scene(pair["input"])
    twin = pair["input"].copy()
    a, b = scene["labels"][1], scene["labels"][2]
    after = scene["after"].copy()
    after = np.where(after == a, b, np.where(after == b, a, after)).astype(np.int8)
    box = scene["panels"][1-scene["target_group"]][1-scene["raw_slot"]]
    r, c, h, w = box[:4]
    twin[r+1:r+h-1, c+1:c+w-1] = after
    twin_scene = _scene(twin)
    _require(np.array_equal(twin_scene["target"], scene["target"]), "causal twin changed target")
    expected = _paint(scene["target"], scene["target_kinds"], {1: b, 2: a})
    _require(np.array_equal(solve(twin), expected)
             and not np.array_equal(expected, pair["output"]), "reference is dispensable")
    # Omitting A: copying target, using one color for all bends, or recoloring
    # every contour cell must each disagree with the actual reconstructed panel.
    for shortcut in (scene["target"],
                     _paint(scene["target"], scene["target_kinds"], {1: a, 2: a}),
                     np.where(scene["target"] != scene["neutral"], a, scene["neutral"])):
        _require(not np.array_equal(shortcut, pair["output"]), "A ablation did not change answer")
    # Moving a reference member or rotating/transposing the whole carrier does
    # not change the local rule or turn blank position into an inverse command.
    moved = pair["input"].copy()
    for box, content in zip(scene["panels"][1-scene["target_group"]],
                            [scene["after"], scene["before"]] if scene["raw_slot"] == 0
                            else [scene["before"], scene["after"]]):
        r, c, h, w = box[:4]
        moved[r+1:r+h-1, c+1:c+w-1] = content
    _require(np.array_equal(solve(moved), pair["output"]), "reference order affected answer")
    moved = pair["input"].copy()
    for box, content in zip(scene["panels"][scene["target_group"]],
                            [scene["target"], np.full_like(scene["target"], scene["neutral"])]
                            if scene["blank_slot"] == 0 else
                            [np.full_like(scene["target"], scene["neutral"]), scene["target"]]):
        r, c, h, w = box[:4]
        moved[r+1:r+h-1, c+1:c+w-1] = content
    _require(np.array_equal(solve(moved), pair["output"]), "blank position affected answer")
    _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "transpose failure")
    _require(np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"])),
             "rotation failure")
    permutation = np.random.default_rng(12345).permutation(10)
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
             "palette equivariance failure")
    bad = {s: [dict(p) for p in first[s]] for s in first}
    bad["test"][0]["output"] = first["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzle_hashes),
                canvas_shapes=len(canvases), panel_shapes=len(panel_shapes),
                target_masks=len(targets), turn_patterns=len(turn_patterns),
                label_mappings=len(mappings), layout_orders=len(orders),
                target_loop_counts=sorted(loop_counts), concave_counts=sorted(concaves),
                loops_with_adjacent_bends=adjacent, loops_touching_panel_edge=edge,
                reference_dependence=True, independent_semantic_solver=False)
