"""Counted expeditions: a local growth analogy powered by object counts.

The solver reads only pixels. Generation constructs labels from sampled logical
arenas; validation recovers the arenas, calibration and objects from the scene.
"""
from collections import deque
from itertools import combinations

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_KERNELS = tuple(tuple(_DIRS[i] for i in indices)
                 for size in (2, 3, 4) for indices in combinations(range(4), size))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _check(np.all((0 <= a) & (a <= 9)), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, bg):
    visited = np.zeros(a.shape, dtype=bool)
    result = []
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if visited[r, c] or a[r, c] == bg:
                continue
            color = int(a[r, c])
            visited[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in _DIRS:
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                            and a[yy, xx] == color):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            ys, xs = zip(*cells)
            result.append(dict(color=color, cells=tuple(cells),
                               box=(min(ys), min(xs), max(ys)-min(ys)+1,
                                    max(xs)-min(xs)+1)))
    return result


def _outline(a, box, color):
    r, c, h, w = box
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _interior(a, box):
    r, c, h, w = box
    return a[r+1:r+h-1, c+1:c+w-1].copy()


def _is_outline(a, component):
    r, c, h, w = component["box"]
    if min(h, w) < 5:
        return False
    v = component["color"]
    return bool(np.all(a[r, c:c+w] == v) and np.all(a[r+h-1, c:c+w] == v)
                and np.all(a[r:r+h, c] == v) and np.all(a[r:r+h, c+w-1] == v))


def _calibration(bg, guide, kernel):
    source = np.full((3, 3), bg, dtype=np.int8)
    source[1, 1] = guide
    partner = source.copy()
    for dr, dc in kernel:
        partner[1+dr, 1+dc] = guide
    return source, partner


def _read_calibration(first, second, bg, wall):
    for source, partner in ((first, second), (second, first)):
        colored = np.argwhere(source != bg)
        if len(colored) != 1 or tuple(colored[0]) != (1, 1):
            continue
        guide = int(source[1, 1])
        if guide == wall or not set(map(int, np.unique(partner))) <= {bg, guide}:
            continue
        kernel = tuple(d for d in _DIRS if partner[1+d[0], 1+d[1]] == guide)
        if len(kernel) < 2:
            continue
        if np.array_equal(partner, _calibration(bg, guide, kernel)[1]):
            return guide, kernel
    raise AssertionError("calibration is not a one-step axial growth analogy")


def _inspect(value):
    a = _grid(value)
    bg = int(a[0, 0])
    _check(np.all(a[0] == bg) and np.all(a[-1] == bg)
           and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "uniform outer margin")
    components = _components(a, bg)
    outlines = [p for p in components if _is_outline(a, p)]
    small = [p for p in outlines if p["box"][2:] == (5, 5)]
    large = [p for p in outlines if all(9 <= d <= 17 for d in p["box"][2:])]
    _check(len(outlines) == 3 and len(small) == 2 and len(large) == 1,
           "expected two calibration plaques and one arena")
    frames = small + large
    wall = large[0]["color"]
    _check(all(p["color"] == wall for p in frames), "shared frame/wall role")
    guide, kernel = _read_calibration(*[_interior(a, p["box"]) for p in small], bg, wall)
    arena = _interior(a, large[0]["box"])
    cats = sorted(set(map(int, np.unique(arena))) - {bg, wall})
    _check(2 <= len(cats) <= 4 and guide not in cats, "two through four seed categories")
    anchors = {}
    for color in cats:
        cells = np.argwhere(arena == color)
        _check(len(cells) == 1, "one seed cell per category")
        anchors[color] = tuple(map(int, cells[0]))
    _check(all(abs(r-y)+abs(c-x) >= 3 for (r, c), (y, x)
               in combinations(anchors.values(), 2)), "seeds must be separated")
    excluded = np.zeros(a.shape, dtype=bool)
    for p in frames:
        r, c, h, w = p["box"]
        _check(not np.any(excluded[max(0, r-1):r+h+1, max(0, c-1):c+w+1]),
               "frame rectangles need a background moat")
        excluded[r:r+h, c:c+w] = True
    exterior = a.copy()
    exterior[excluded] = bg
    objects = _components(exterior, bg)
    _check(bool(objects), "at least one external object")
    labels = np.full(a.shape, -1, dtype=int)
    for i, obj in enumerate(objects):
        _check(1 <= len(obj["cells"]) <= 10, "external object areas are 1..10")
        for r, c in obj["cells"]:
            labels[r, c] = i
            _check(not np.any(excluded[r-1:r+2, c-1:c+2]), "object touches a frame")
    for obj in objects:
        for r, c in obj["cells"]:
            i = labels[r, c]
            neighbors = labels[r-1:r+2, c-1:c+2]
            _check(not np.any((neighbors >= 0) & (neighbors != i)), "external components touch")
    budgets = {color: sum(o["color"] == color for o in objects) for color in cats}
    _check(max(budgets.values()) <= 6, "at most six objects per seed category")
    areas = {color: sum(len(o["cells"]) for o in objects if o["color"] == color)
             for color in cats}
    return dict(grid=a, bg=bg, wall=wall, guide=guide, kernel=kernel, arena=arena,
                anchors=anchors, budgets=budgets, areas=areas, objects=objects,
                small=small, large=large[0], frames=frames)


def _distances(arena, wall, start, kernel):
    h, w = arena.shape
    distance = np.full((h, w), 1000, dtype=int)
    distance[start] = 0
    todo = deque([start])
    while todo:
        r, c = todo.popleft()
        for dr, dc in kernel:
            y, x = r+dr, c+dc
            if (0 <= y < h and 0 <= x < w and arena[y, x] != wall
                    and distance[y, x] == 1000):
                distance[y, x] = distance[r, c]+1
                todo.append((y, x))
    return distance


def _grow(arena, bg, wall, guide, anchors, budgets, kernel):
    colors = sorted(anchors)
    distances = np.stack([_distances(arena, wall, anchors[c], kernel) for c in colors])
    costs = np.where(distances <= np.array([budgets[c] for c in colors])[:, None, None],
                     distances, 1000)
    best = costs.min(axis=0)
    winners = costs == best
    out = np.full(arena.shape, bg, dtype=np.int8)
    reached = best < 1000
    unique = winners.sum(axis=0) == 1
    for i, color in enumerate(colors):
        out[reached & unique & winners[i]] = color
    out[reached & ~unique] = guide
    out[arena == wall] = wall
    return out


def _answer(d, *, kernel=None, budgets=None, arena=None):
    return _grow(d["arena"] if arena is None else arena, d["bg"], d["wall"],
                 d["guide"], d["anchors"], d["budgets"] if budgets is None else budgets,
                 d["kernel"] if kernel is None else kernel)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the demonstrated moves, count components, and crop the grown arena."""
    return _answer(_inspect(input_grid))


def _shape(rng, area=None):
    # Geometry is sampled without any category count argument.
    area = int(rng.integers(1, 11)) if area is None else int(area)
    kind = int(rng.integers(4))
    if kind == 0 and area <= 6:
        cells = {(0, c) for c in range(area)}
    elif kind == 1:
        width = int(rng.integers(2, min(4, area)+1)) if area > 1 else 1
        cells = {(i//width, i % width) for i in range(area)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS}
                              - cells)
            compact = [p for p in frontier if -2 <= p[0] <= 2 and -2 <= p[1] <= 2]
            choices = compact if compact else frontier
            cells.add(choices[int(rng.integers(len(choices)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.integers(2):
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _walls(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(4))
    if kind == 0:  # a fence with one or two gates
        c = int(rng.integers(2, w-2))
        mask[1:-1, c] = True
        for r in rng.choice(np.arange(1, h-1), int(rng.integers(1, 3)), replace=False):
            mask[r, c] = False
    elif kind == 1:  # staggered short horizontal and vertical walls
        for _ in range(int(rng.integers(2, 5))):
            r, c = int(rng.integers(h-2)), int(rng.integers(w-2))
            if rng.integers(2):
                mask[r, c:min(w, c+int(rng.integers(2, 6)))] = True
            else:
                mask[r:min(h, r+int(rng.integers(2, 6))), c] = True
    elif kind == 2:  # an open elbow
        r, c = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
        mask[r, 1:c+1] = True
        mask[r:min(h-1, r+4), c] = True
    else:  # a sparse broken diagonal staircase
        r, c = int(rng.integers(2)), int(rng.integers(2))
        for i in range(min(h-r, w-c)-1):
            if rng.random() < .7:
                mask[r+i, c+i] = True
                mask[r+i, c+i+1] = True
    if rng.integers(2):
        mask = mask[:, ::-1].copy()
    if rng.integers(2):
        mask = mask[::-1].copy()
    return mask


def _teaching_arena(rng):
    h, w = int(rng.integers(9, 12)), int(rng.integers(9, 12))
    arena = _walls(rng, h, w).astype(np.int8)  # logical wall role 1
    anchors = [(4, 4), (3, 2), (3, 6)]
    # Clear a meeting lane and the middle seed's radius-two patch. The two
    # two-step seeds tie at (3,4); the one-step seed wins at (4,3)/(4,5).
    arena[2:5, 1:8] = 0
    r, c = anchors[0]
    arena[max(0, r-2):min(h, r+3), max(0, c-2):min(w, c+3)] = 0
    # A visible blocked route: (5,2) would need two downward steps from (3,2).
    arena[4, 2], arena[5, 2] = 1, 0
    for i, p in enumerate(anchors):
        arena[p] = 2+i
    return arena


def _random_arena(rng, n, test):
    h = int(rng.integers(11, 16) if test else rng.integers(7, 12))
    w = int(rng.integers(9, 14) if test else rng.integers(7, 12))
    arena = _walls(rng, h, w).astype(np.int8)
    points = []
    for i in range(n):
        candidates = [(r, c) for r in range(h) for c in range(w)
                      if arena[r, c] == 0 and (test or (0 < r < h-1 and 0 < c < w-1))
                      and all(abs(r-y)+abs(c-x) >= 3 for y, x in points)]
        if test and i == 0:
            candidates = [p for p in candidates if p[0] in (0, h-1) or p[1] in (0, w-1)]
        _check(bool(candidates), "seed placement exhausted")
        p = candidates[int(rng.integers(len(candidates)))]
        points.append(p)
        arena[p] = 2+i
    return arena


def _palette(rng, n, novel, test):
    cats = [novel] if test else []
    pool = [c for c in range(10) if c != novel]
    cats.extend(map(int, rng.choice(pool, n-len(cats), replace=False)))
    rest = [c for c in range(10) if c not in cats]
    bg, wall, guide = map(int, rng.choice(rest, 3, replace=False))
    return bg, wall, guide, cats


def _place_frames(rng, arena, bg, wall, guide, kernel, test, roomy=False):
    ah, aw = arena.shape[0]+2, arena.shape[1]+2
    if roomy:
        h = w = 30
    elif test:
        h, w = int(rng.integers(28, 31)), int(rng.integers(28, 31))
    else:
        h, w = int(rng.integers(max(20, ah+8), 31)), int(rng.integers(max(20, aw+8), 31))
    base = np.full((h, w), bg, dtype=np.int8)
    blocked = np.zeros(base.shape, dtype=bool)
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    boxes = []
    patterns = [arena, *_calibration(bg, guide, kernel)]
    if rng.integers(2):
        patterns[1:] = patterns[1:][::-1]
    for pattern in patterns:
        ph, pw = pattern.shape[0]+2, pattern.shape[1]+2
        candidates = [(r, c) for r in range(1, h-ph) for c in range(1, w-pw)
                      if not np.any(blocked[r:r+ph, c:c+pw])]
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        box = (r, c, ph, pw)
        boxes.append(box)
        _outline(base, box, wall)
        base[r+1:r+ph-1, c+1:c+pw-1] = pattern
        blocked[r-1:r+ph+1, c-1:c+pw+1] = True
    return base, boxes, blocked


def _pack(rng, base, blocked, specs):
    a, occupied = base.copy(), blocked.copy()
    h, w = a.shape
    order = sorted(range(len(specs)), key=lambda i: (-len(specs[i][1]), float(rng.random())))
    for i in order:
        color, shape = specs[i]
        sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
        if sh > h or sw > w:
            return None
        available = np.ones((h-sh+1, w-sw+1), dtype=bool)
        for dr, dc in shape:
            available &= ~occupied[dr:dr+h-sh+1, dc:dc+w-sw+1]
        ys, xs = np.nonzero(available)
        if not len(ys):
            return None
        j = int(rng.integers(len(ys)))
        r0, c0 = int(ys[j]), int(xs[j])
        for dr, dc in shape:
            r, c = r0+dr, c0+dc
            a[r, c] = color
            occupied[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
    return a


def _make_pair(rng, logical, palette, counts, kernel, test, forced=None):
    bg, wall, guide, cats = palette
    mapping = [bg, wall] + cats
    arena = np.array(mapping, dtype=np.int8)[logical]
    anchors = {color: tuple(map(int, np.argwhere(arena == color)[0])) for color in cats}
    budgets = dict(zip(cats, counts))
    output = _grow(arena, bg, wall, guide, anchors, budgets, kernel)
    specs = []
    for i, (color, count) in enumerate(zip(cats, counts)):
        for j in range(count):
            area = forced[i][j] if forced and i in forced else None
            specs.append((color, _shape(rng, area)))
    # The teaching pair deliberately has both a guide-colored and an unrelated
    # object: neither belongs to a category without a seed in the arena.
    unrelated = [c for c in range(10) if c not in {bg, wall, guide, *cats}]
    distractor_colors = [guide, int(rng.choice(unrelated))] if forced else [
        int(rng.choice([guide]+unrelated)) for _ in range(int(rng.integers(4)))]
    specs.extend((c, _shape(rng)) for c in distractor_colors)
    for attempt in range(80):
        layout = _place_frames(rng, arena, bg, wall, guide, kernel, test, roomy=attempt >= 10)
        if layout is None:
            continue
        base, boxes, blocked = layout
        inp = _pack(rng, base, blocked, specs)
        if inp is None:
            continue
        # Reject accidental extra outlines, without changing the logical rule.
        try:
            d = _inspect(inp)
            if not np.array_equal(_answer(d), output):
                continue
        except AssertionError:
            continue
        rotation = int(rng.integers(4))
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(output, rotation).copy()
        if rng.integers(2):
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        return {"input": inp, "output": out}
    raise AssertionError("bounded scene packing exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    novel = int(rng.integers(10))
    teaching = _teaching_arena(rng)
    teaching_palette = _palette(rng, 3, novel, False)
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(a+b):
        test = index >= a
        split = "test" if test else "train"
        if index < 2:
            logical, palette = teaching.copy(), teaching_palette
            counts = [1, 2, 2] if index == 0 else [2, 1, 2]
            kernel = ((0, -1), (0, 1), (1, 0)) if index == 0 else _KERNELS[int(rng.integers(6))]
            forced = {0: [6] if index == 0 else [2, 4],
                      1: [3, 3] if index == 0 else [6],
                      2: [1, 2] if index == 0 else [4, 5]}
        else:
            for _ in range(80):
                n = int(rng.integers(2, 5))
                logical = _random_arena(rng, n, test)
                palette = _palette(rng, n, novel, test)
                counts = [int(rng.integers(4, 7))]+[int(rng.integers(4)) for _ in range(n-1)] if test else [
                    int(rng.integers(4)) for _ in range(n)]
                if not any(counts):
                    counts[int(rng.integers(n))] = 1
                while sum(counts) > 12:
                    j = 1+int(rng.integers(n-1))
                    counts[j] = max(0, counts[j]-1)
                kernel, forced = _KERNELS[int(rng.integers(len(_KERNELS)))], None
                if not test:
                    break
                anchors = {2+i: tuple(map(int, np.argwhere(logical == 2+i)[0])) for i in range(n)}
                grown = _grow(logical, 0, 1, 9, anchors, dict(zip(range(2, 2+n), counts)), kernel)
                distance = _distances(logical, 1, anchors[2], kernel)
                if np.any((grown == 2) & (distance == counts[0])):
                    break
            else:
                raise AssertionError("active test growth-layer construction exhausted")
        for _ in range(12):
            pair = _make_pair(rng, logical, palette, counts, kernel, test, forced)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("distinct input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _inspect(inp)
            _check(out.shape == d["arena"].shape, "output must crop the arena interior")
            _check(np.array_equal(out, _answer(d)), "wrong growth, tie, wall, or crop")
            _check(np.array_equal(out == d["wall"], d["arena"] == d["wall"]), "walls changed")
            _check(all(out[p] == color for color, p in d["anchors"].items()), "seed changed")
            details[split].append((d, out))
    train = details["train"]
    records = [(c, d["budgets"][c], d["areas"][c]) for d, _ in train for c in d["anchors"]]
    _check(any(c == cc and area == aa and n != nn for c, n, area in records
               for cc, nn, aa in records), "missing same-color area/count collision")
    _check(any(n == nn and area != aa for c, n, area in records for cc, nn, aa in records),
           "missing same-count different-area evidence")
    _check(len({tuple(d["kernel"]) for d, _ in train}) >= 2, "training must vary the local reference")
    _check(any(np.any(out == d["guide"]) for d, out in train), "training must teach ties")
    _check(any(not np.array_equal(out, _answer(d, kernel=_DIRS)) for d, out in train),
           "a full-cross shortcut must fail in training")
    _check(any(not np.array_equal(out, _answer(d, budgets={c: d["areas"][c] for c in d["anchors"]}))
               for d, out in train), "an area-budget shortcut must fail in training")
    nearest_witness = False
    for d, out in train:
        distances = {c: _distances(d["arena"], d["wall"], p, d["kernel"])
                     for c, p in d["anchors"].items()}
        for c, cc in combinations(d["anchors"], 2):
            for low, high in ((c, cc), (cc, c)):
                n, nn = d["budgets"][low], d["budgets"][high]
                if n >= nn:
                    continue
                dl, dh = distances[low], distances[high]
                nearest_witness |= bool(np.any((out == low) & (dl <= n) & (dh <= nn)
                    & (dl < dh) & (n-dl <= nn-dh)))
    _check(nearest_witness, "training must distinguish distance from remaining/largest budget")
    obstacle_witness = False
    for d, out in train:
        cleared = d["arena"].copy()
        cleared[cleared == d["wall"]] = d["bg"]
        other = _answer(d, arena=cleared)
        open_cells = d["arena"] != d["wall"]
        obstacle_witness |= bool(np.any(out[open_cells] != other[open_cells]))
    _check(obstacle_witness, "walls must affect reachability beyond their own cells")
    train_cats = {c for d, _ in train for c in d["anchors"]}
    train_max = max(n for d, _ in train for n in d["budgets"].values())
    _check(all(max(d["budgets"].values()) > train_max for d, _ in details["test"]),
           "test must extrapolate the demonstrated growth budget")
    _check(all(any(n > train_max and np.any((out == c) &
                   (_distances(d["arena"], d["wall"], d["anchors"][c], d["kernel"]) == n))
                   for c, n in d["budgets"].items()) for d, out in details["test"]),
           "test's larger budget must visibly reach its final layer")
    _check(any(set(d["anchors"])-train_cats for d, _ in details["test"]), "new test category")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _repack_exterior(d, specs, seed):
    rng = np.random.default_rng(seed)
    base, blocked = d["grid"].copy(), np.zeros(d["grid"].shape, dtype=bool)
    for obj in d["objects"]:
        for p in obj["cells"]:
            base[p] = d["bg"]
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    for frame in d["frames"]:
        r, c, h, w = frame["box"]
        blocked[r-1:r+h+1, c-1:c+w+1] = True
    for _ in range(40):
        inp = _pack(rng, base, blocked, specs)
        if inp is not None:
            return inp
    raise AssertionError("metamorphic packing exhausted")


def self_test():
    """Bounded 42-seed diversity and semantic/metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = dict(seeds=42, pairs=0)
    shapes, arenas, kernels, counts, masks, wall_masks, object_masks, backgrounds = (
        set() for _ in range(8))
    zero = ties = edge_seeds = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                d = _inspect(pair["input"])
                out = pair["output"]
                metrics["pairs"] += 1
                shapes.add(pair["input"].shape)
                arenas.add(out.shape)
                kernels.add(tuple(d["kernel"]))
                backgrounds.add(d["bg"])
                counts.update(d["budgets"].values())
                masks.add(_key(np.where(out == d["bg"], 0, np.where(out == d["wall"], 1,
                              np.where(out == d["guide"], 2, 3))).astype(np.int8)))
                wall_masks.add(_key((d["arena"] == d["wall"]).astype(np.int8)))
                zero += 0 in d["budgets"].values()
                ties += bool(np.any(out == d["guide"]))
                h, w = out.shape
                edge_seeds += any(r in (0, h-1) or c in (0, w-1) for r, c in d["anchors"].values())
                for obj in d["objects"]:
                    r, c, _, _ = obj["box"]
                    object_masks.add(tuple(sorted((y-r, x-c) for y, x in obj["cells"])))
    _check(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed repetition")
    _check(len(shapes) >= 30 and len(arenas) >= 25 and len(masks) >= 150, "geometry variation")
    _check(len(kernels) == 11 and counts == set(range(7)), "kernel/count coverage")
    _check(backgrounds == set(range(10)) and len(wall_masks) >= 100, "palette/wall coverage")
    _check(len(object_masks) >= 100 and min(zero, ties, edge_seeds) > 0, "structural cases missing")
    _check(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterminism")
    # A: hold the entire scene fixed, change only the demonstrated partner.
    d = _inspect(puzzles[0]["train"][0]["input"])
    twin = d["grid"].copy()
    expanded = next(p for p in d["small"] if np.count_nonzero(_interior(twin, p["box"]) != d["bg"]) > 1)
    r, c, _, _ = expanded["box"]
    twin[r+1:r+4, c+1:c+4] = _calibration(d["bg"], d["guide"], _DIRS)[1]
    _check(not np.array_equal(solve(twin), _answer(d)), "reference-dependence twin failed")
    # B: changing every object's area while preserving component counts leaves
    # the result fixed. Splitting one six-cell object into two changes it.
    singleton_specs = [(o["color"], ((0, 0),)) for o in d["objects"]]
    same_counts = _repack_exterior(d, singleton_specs, 731)
    _check(np.array_equal(solve(same_counts), _answer(d)), "object-area invariance failed")
    color = next(c for c in d["anchors"] if d["budgets"][c] == 1 and d["areas"][c] == 6)
    specs = []
    for o in d["objects"]:
        r, c, _, _ = o["box"]
        shape = tuple(sorted((y-r, x-c) for y, x in o["cells"]))
        if o["color"] == color:
            specs.extend([(color, ((0, 0), (0, 1))),
                          (color, ((0, 0), (0, 1), (1, 0), (1, 1)))])
        else:
            specs.append((o["color"], shape))
    split_object = _repack_exterior(d, specs, 732)
    split_d = _inspect(split_object)
    _check(split_d["areas"][color] == 6 and split_d["budgets"][color] == 2, "count twin construction")
    _check(not np.array_equal(solve(split_object), _answer(d)), "component-dependence twin failed")
    # Rotating/reflection-transforming input and output must commute with solve.
    for k in range(4):
        inp = np.rot90(d["grid"], k).copy()
        _check(np.array_equal(solve(inp), np.rot90(_answer(d), k)), "rotation equivariance")
        _check(np.array_equal(solve(inp[:, ::-1]), np.rot90(_answer(d), k)[:, ::-1]), "reflection equivariance")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    metrics.update(canvas_shapes=len(shapes), arena_shapes=len(arenas), move_sets=len(kernels),
                   count_values=sorted(counts), normalized_outputs=len(masks), wall_masks=len(wall_masks),
                   object_masks=len(object_masks), backgrounds=sorted(backgrounds),
                   zero_budget_pairs=zero, tie_pairs=ties, edge_seed_pairs=edge_seeds,
                   reference_twin=True, area_invariance=True, equal_area_count_twin=True)
    return metrics


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
