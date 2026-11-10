"""Component-counted, mask-aware greedy packing at color-coordinate docks.

Only the visible rack/frame geometry determines the convention. No parent
generator, source pair, seed, or generation record is used by solve().
"""
from __future__ import annotations

import copy
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be an integer 1..30 rectangle with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, group = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    todo.append(point)
                    group.append(point)
        result.append(np.array(sorted(group), dtype=int))
    return result


def _parse_canonical(g, d, transposed):
    """Reject unsupported geometry before creating any output."""
    h, w = g.shape
    bg, divider, frame = int(g[0, 0]), int(g[0, d]), int(g[0, d+1])
    _require(len({bg, divider, frame}) == 3, "structural roles coincide")
    _require(np.all(g[:, d] == divider), "divider is not full-span")
    rack = g[:, :d]
    cuts = np.flatnonzero(np.all(rack == divider, axis=1))
    _require(len(cuts) == 1, "rack needs one short divider")
    cut = int(cuts[0])
    _require(3 <= cut <= h-4 and d >= 3, "empty rack compartment")
    upper, lower = rack[:cut], rack[cut+1:]
    for panel in (upper, lower):
        _require(np.all(panel[[0, -1]] == bg) and np.all(panel[:, [0, -1]] == bg),
                 "rack margin does not expose background")
        _require(not np.any((panel == divider) | (panel == frame)), "structure in rack")
    colors = [int(c) for c in np.unique(upper) if c != bg]
    _require(1 <= len(colors) <= 3, "need one to three prototypes")
    _require(set(map(int, np.unique(lower))) <= set(colors) | {bg}, "unknown stock color")
    templates = []
    for color in colors:
        cells = np.argwhere(upper == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        bh, bw = map(int, hi-lo+1)
        _require(bh % 2 == bw % 2 == 1, "prototype centers must be odd")
        mask = upper[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        _require(len(_components(mask, True)) == 1, "prototype fragments")
        stock = _components(lower == color)
        _require(len(stock) == len(_components(lower == color, True)), "diagonal stock contact")
        for component in stock:
            component[:, 0] += cut+1
        templates.append(dict(color=color, lo=tuple(map(int, lo)), mask=mask,
                              cells=cells, stock=stock))
    _require(len(_components(lower != bg, True)) == sum(len(t["stock"]) for t in templates),
             "different stock colors touch")
    templates.sort(key=lambda t: t["lo"])
    _require(len({t["lo"] for t in templates}) == len(templates), "prototype reading-order tie")
    _require(all(int(g[r, c]) == frame for r, c in
                 ((0, d+1), (0, w-1), (h-1, d+1), (h-1, w-1))), "frame corners")
    _require(np.array_equal(g[0, d+2:-1], g[-1, d+2:-1])
             and np.array_equal(g[1:-1, d+1], g[1:-1, -1]), "unpaired coordinates")
    edges = np.concatenate((g[0, d+1:], g[-1, d+1:], g[1:-1, d+1], g[1:-1, -1]))
    _require(set(map(int, edges)) <= {frame} | set(colors), "unknown border role")
    for t in templates:
        t["rows"] = np.flatnonzero(g[1:-1, d+1] == t["color"]).astype(int)+1
        t["cols"] = np.flatnonzero(g[0, d+2:-1] == t["color"]).astype(int)+d+2
        bh, bw = t["mask"].shape
        _require(all(1 <= r-bh//2 and r+bh//2 < h-1 for r in t["rows"]), "row clipping")
        _require(all(d+2 <= c-bw//2 and c+bw//2 < w-1 for c in t["cols"]), "column clipping")
    interior = g[1:-1, d+2:-1]
    other = set(map(int, np.unique(interior))) - {bg}
    _require(len(other) <= 1 and not other.intersection({divider, frame} | set(colors)),
             "field initially contains only background and an obstacle role")
    _require(any(t["rows"].size and t["cols"].size for t in templates), "no coordinate request")
    return dict(grid=g, d=d, cut=cut, bg=bg, divider=divider, frame=frame,
                templates=templates, transposed=transposed, obstacle=next(iter(other), None))


def _parse(value):
    a = _grid(value)
    candidates = []
    for transposed in (False, True):
        g = a.T.copy() if transposed else a
        h, w = g.shape
        for d in range(3, w-4):
            if g[0, d] == g[0, 0] or not np.all(g[:, d] == g[0, d]):
                continue
            try:
                candidates.append(_parse_canonical(g, d, transposed))
            except AssertionError:
                continue
    if len(candidates) != 1:
        raise ValueError("input must have one visible rack/frame parse")
    return candidates[0]


def _simulate(p, *, quota="components", footprint="mask", collision="skip",
              budget="success", order="source", coordinates="cartesian", transform=0):
    """Array-based visible-input implementation, with diagnostic ablations."""
    out = p["grid"].copy()
    ts = list(p["templates"])
    if order == "reverse":
        ts.reverse()
    elif order == "size":
        ts.sort(key=lambda t: t["mask"].size)
    elif order == "destination":
        ts.sort(key=lambda t: (min(t["rows"], default=99), min(t["cols"], default=99)))
    union_rows = sorted({int(r) for t in ts for r in t["rows"]})
    union_cols = sorted({int(c) for t in ts for c in t["cols"]})
    records = []
    for t in ts:
        stock, mask = t["stock"], t["mask"]
        if transform:
            mask = np.rot90(mask[:, ::-1] if transform >= 4 else mask, transform % 4)
        n = len(stock) if quota == "components" else sum(len(x) for x in stock) if quota == "area" else 900
        rows, cols = t["rows"], t["cols"]
        if coordinates == "union":
            rows, cols = union_rows, union_cols
        docks = (list(zip(rows, cols)) if coordinates == "zip" else
                 [(r, c) for r in rows for c in cols])
        accepted, attempted, failed, positions = 0, 0, 0, []
        bh, bw = mask.shape
        for r, c in docks:
            spent = attempted if budget == "attempt" else accepted
            if spent >= n:
                break
            r, c = int(r), int(c)
            attempted += 1
            top, left = r-bh//2, c-bw//2
            # Only alternative coordinate/transform hypotheses can violate fit.
            if top < 1 or left < p["d"]+2 or top+bh > out.shape[0]-1 or left+bw > out.shape[1]-1:
                failed += 1
                continue
            view = out[top:top+bh, left:left+bw]
            check = np.ones_like(mask) if footprint == "box" else mask
            blocked = np.any(view[check] != p["bg"])
            if blocked and collision == "skip":
                failed += 1
                continue
            view[mask] = t["color"]
            accepted += 1
            positions.append((r, c))
        consumed = min(len(stock), attempted if budget == "attempt" else accepted)
        for component in stock[:consumed]:
            out[component[:, 0], component[:, 1]] = p["bg"]
        records.append(dict(color=t["color"], accepted=accepted, attempted=attempted,
                            failed=failed, consumed=consumed, positions=positions))
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    out, _ = _simulate(p)
    return out.T.copy() if p["transposed"] else out


def _grow(rng, h, w, area):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            y, x = point
            if 0 <= y < h and 0 <= x < w and not mask[y, x]:
                frontier.add(point)
    return mask


def _mask(rng, h, w):
    m = np.zeros((h, w), bool)
    mode = int(rng.integers(7))
    if min(h, w) == 1 or mode == 0:
        m[:] = True
    elif mode == 1:
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
    elif mode == 2:
        for r in range(h):
            m[r, round(r*(w-1)/(h-1))] = True
        for c in range(w):
            m[round(c*(h-1)/(w-1)), c] = True
    elif mode == 3:
        m[h//2, :] = True
        m[:, w//2] = True
    else:
        # Join independently selected side points with eight-neighbor paths.
        points = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        rng.shuffle(points)
        r, c = points[0]
        m[r, c] = True
        for y, x in points[1:]:
            while (r, c) != (y, x):
                dr, dc = int(np.sign(y-r)), int(np.sign(x-c))
                if dr and dc and rng.random() < .5:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r+dr, c+dc
                m[r, c] = True
        target = int(rng.uniform(.25, .95)*h*w)
        while m.sum() < target:
            choices = [(y, x) for y in range(h) for x in range(w) if not m[y, x]
                       and m[max(0, y-1):y+2, max(0, x-1):x+2].any()]
            y, x = choices[int(rng.integers(len(choices)))]
            m[y, x] = True
    if rng.random() < .5:
        m = m[::-1]
    if rng.random() < .5:
        m = m[:, ::-1]
    return m.copy()


def _lengths(length, count, rng):
    available = length-count+1
    sizes = [available//count + int(i < available % count) for i in range(count)]
    rng.shuffle(sizes)
    result, start = [], 0
    for size in sizes:
        result.append((start, start+size))
        start += size+1
    return result


def _stocks(rng, h, w, colors, counts, areas=None):
    """Separated rooms let geometry and object area vary without count ambiguity."""
    requests = []
    for i, (color, count) in enumerate(zip(colors, counts)):
        fixed = None if areas is None else areas[i]
        for j in range(count):
            requests.append((color, None if fixed is None else fixed[j]))
    total = len(requests)
    options = []
    for nr in range(1, (h+1)//2+1):
        for nc in range(1, (w+1)//2+1):
            if total <= nr*nc <= total + max(1, total//2):
                min_capacity = ((h-nr+1)//nr)*((w-nc+1)//nc)
                if areas is None or min_capacity >= max(x[1] for x in requests):
                    options.append((nr, nc))
    _require(bool(options), "stock rooms do not fit")
    nr, nc = options[int(rng.integers(len(options)))]
    rs, cs = _lengths(h, nr, rng), _lengths(w, nc, rng)
    boxes = [(a, b, c, d) for a, b in rs for c, d in cs]
    rng.shuffle(boxes)
    rng.shuffle(requests)
    result = {color: [] for color in colors}
    singleton_scene = areas is None and rng.random() < .15
    for (color, fixed_area), (r0, r1, c0, c1) in zip(requests, boxes):
        bh, bw = r1-r0, c1-c0
        if fixed_area is not None:
            mask = _grow(rng, bh, bw, fixed_area)
        elif singleton_scene:
            mask = _grow(rng, bh, bw, 1)
        else:
            oh, ow = int(rng.integers(1, bh+1)), int(rng.integers(1, bw+1))
            mode = int(rng.integers(5))
            if mode == 0:
                oh = 1
            elif mode == 1:
                ow = 1
            if mode <= 2:
                mask = np.ones((oh, ow), bool)
            elif mode == 3 and min(oh, ow) >= 3:
                mask = np.ones((oh, ow), bool)
                mask[1:-1, 1:-1] = False
            else:
                mask = _grow(rng, oh, ow, int(rng.integers(1, oh*ow+1)))
            r0 += int(rng.integers(bh-mask.shape[0]+1))
            c0 += int(rng.integers(bw-mask.shape[1]+1))
        cells = np.argwhere(mask)+np.array([r0, c0])
        result[color].append(cells)
    for groups in result.values():
        groups.sort(key=lambda g: tuple(g[0]))
    return result


def _assemble(rng, h, d, fw, masks, rows, cols, counts, obstacles,
              *, cut=None, areas=None, palette=None, transposed=False):
    """Construct expected labels from records using cell-set reservation."""
    palette = list(map(int, rng.permutation(10))) if palette is None else list(palette)
    bg, divider, frame, obstacle = palette[:4]
    colors = palette[4:4+len(masks)]
    minimum = 2+sum(m.shape[0] for m in masks)+len(masks)-1
    if cut is None:
        cut = minimum+int(rng.integers(0, min(4, h-minimum-6)))
    _require(cut >= minimum, "prototype rack height")
    inp = np.full((h, d+1+fw), bg, np.int8)
    inp[:, d] = divider
    inp[cut, :d] = divider
    inp[[0, -1], d+1:] = frame
    inp[:, [d+1, inp.shape[1]-1]] = frame
    slack = cut-minimum
    gaps = rng.multinomial(slack, np.full(len(masks)+1, 1/(len(masks)+1)))
    templates, top = [], 1+int(gaps[0])
    for i, (mask, color) in enumerate(zip(masks, colors)):
        left = int(rng.integers(1, d-mask.shape[1]))
        source = np.argwhere(mask)+np.array([top, left])
        inp[source[:, 0], source[:, 1]] = color
        for r in rows[i]:
            inp[r, d+1] = inp[r, -1] = color
        for c in cols[i]:
            inp[0, d+1+c] = inp[-1, d+1+c] = color
        templates.append(dict(color=color, mask=mask, rows=rows[i], cols=cols[i]))
        top += mask.shape[0]+1+int(gaps[i+1])
    stocks = _stocks(rng, h-cut-3, d-2, colors, counts, areas)
    for color, groups in stocks.items():
        for group in groups:
            group += np.array([cut+2, 1])
            inp[group[:, 0], group[:, 1]] = color
    claimed = set()
    for r, c in obstacles:
        point = (int(r), d+1+int(c))
        inp[point] = obstacle
        claimed.add(point)
    out = inp.copy()
    accepted_counts = []
    for t in templates:
        color, mask = t["color"], t["mask"]
        offsets = [(int(r)-mask.shape[0]//2, int(c)-mask.shape[1]//2)
                   for r, c in np.argwhere(mask)]
        successes = 0
        for r in sorted(t["rows"]):
            for c in sorted(t["cols"]):
                if successes == len(stocks[color]):
                    break
                pixels = {(r+dy, d+1+c+dx) for dy, dx in offsets}
                if pixels & claimed:
                    continue
                for point in pixels:
                    out[point] = color
                claimed.update(pixels)
                successes += 1
            if successes == len(stocks[color]):
                break
        for group in stocks[color][:successes]:
            out[group[:, 0], group[:, 1]] = bg
        accepted_counts.append(successes)
    if transposed:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}, accepted_counts


def _asymmetric(rng):
    # The missing lower-left pixel shelters the first witness's center obstacle.
    candidates = []
    for bits in range(8):
        m = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]], bool)
        m[0, 2], m[1, 1], m[2, 1] = bool(bits & 1), bool(bits & 2), bool(bits & 4)
        if all(not np.array_equal(m, np.rot90(m[:, ::-1] if k >= 4 else m, k % 4))
               for k in range(1, 8)):
            candidates.append(m)
    return candidates[int(rng.integers(len(candidates)))].copy()


def _witness(rng, index, palette, transposed):
    fw = int(rng.integers(16, 23))
    dy, dx = int(rng.integers(0, 3)), int(rng.integers(0, fw-15))
    small = _asymmetric(rng)
    if index == 0:
        ring = np.ones((3, 3), bool)
        ring[1, 1] = False
        masks = [ring, small]
        rows = [[5+dy], [4+dy, 9+dy]]
        cols = [[3+dx, 7+dx, 11+dx], [8+dx, 12+dx]]
        counts, areas = [1, 2], [[6], [3, 3]]
        obstacles = [(4+dy, 2+dx), (5+dy, 7+dx)]
        h, cut = int(rng.integers(25, 31)), int(rng.integers(9, 13))
    else:
        ring = np.ones((5, 3), bool)
        ring[1:-1, 1] = False
        masks = [ring, small]
        rows = [[6+dy, 7+dy], [5+dy, 11+dy]]
        cols = [[4+dx, 5+dx], [6+dx, 12+dx]]
        counts, areas = [3, 2], [[2, 2, 2], [3, 3]]
        obstacles = []
        h, cut = int(rng.integers(27, 31)), int(rng.integers(11, 14))
    return _assemble(rng, h, 7, fw, masks, rows, cols, counts, obstacles,
                     cut=cut, areas=areas, palette=palette, transposed=transposed)[0]


def _coordinates(rng, masks, h, fw, plentiful=False):
    axes = [set(), set()]
    result = [[None]*len(masks), [None]*len(masks)]
    for axis, length in enumerate((h, fw)):
        # A large mask's center domain is contained in every smaller domain.
        # Allocate those narrower domains first without coupling source order
        # to destination order or to replication count.
        ordering = sorted(range(len(masks)), key=lambda i: -masks[i].shape[axis])
        for position, i in enumerate(ordering):
            mask = masks[i]
            radius = mask.shape[axis]//2
            choices = [x for x in range(1+radius, length-1-radius) if x not in axes[axis]]
            # Enough first labels remain for every other color even in a tight field.
            max_count = min(4 if axis else 3, len(choices)-(len(masks)-position-1))
            _require(max_count >= 1, "coordinate labels do not fit")
            count = min(max_count, (2 if axis == 0 else 3)) if plentiful and i == 0 else int(rng.integers(1, max_count+1))
            first = int(rng.choice(choices))
            choices.remove(first)
            if rng.random() < .5:
                choices.sort(key=lambda x: (abs(x-first), x))
            else:
                rng.shuffle(choices)
            chosen = sorted([first]+choices[:count-1])
            axes[axis].update(chosen)
            result[axis][i] = chosen
    return result


def _random_pair(rng, shift, test=False):
    # All test shifts are visible geometric/count properties, never rule switches.
    t = 3 if test and shift == "colors" else int(rng.integers(1, 3))
    d = 7 if test else int(rng.integers(4, 8))
    h = int(rng.integers(25 if test else 22, 31))
    fw = int(rng.integers(12, 30-d))
    masks = []
    for i in range(t):
        if test and shift == "shape" and i == 0:
            dims = (5, 5)
        else:
            choices = [(a, b) for a in (1, 3, 5) for b in (1, 3, 5)
                       if b <= d-2 and (a, b) != (5, 5)
                       and (t < 3 or a <= (5 if i == 0 else 3))]
            dims = choices[int(rng.integers(len(choices)))]
        masks.append(_mask(rng, *dims))
    # Leave enough stock-space for the positive object counts.
    minimum = 2+sum(m.shape[0] for m in masks)+t-1
    cut = minimum+int(rng.integers(0, min(4, h-minimum-5)))
    capacity = ((h-cut-2)//2)*((d-1)//2)
    counts = [int(rng.integers(1, 4)) for _ in range(t)]
    if test and shift == "stock":
        counts[0] = int(rng.integers(4, 6))
    while sum(counts) > capacity:
        choices = [i for i, n in enumerate(counts) if n > (4 if test and shift == "stock" and i == 0 else 1)]
        _require(bool(choices), "stock count capacity")
        counts[int(rng.choice(choices))] -= 1
    rows, cols = _coordinates(rng, masks, h, fw, test and shift == "stock")
    obstacles = set()
    if rng.random() < .78:
        for _ in range(int(rng.integers(1, 5))):
            oh, ow = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            r, c = int(rng.integers(1, h-oh)), int(rng.integers(1, fw-ow))
            blob = _grow(rng, oh, ow, int(rng.integers(1, oh*ow+1)))
            obstacles.update((r+int(y), c+int(x)) for y, x in np.argwhere(blob))
    palette = list(map(int, rng.permutation(10)))
    transposed = bool(rng.integers(2))
    pair, successes = _assemble(rng, h, d, fw, masks, rows, cols, counts, obstacles,
                                cut=cut, palette=palette, transposed=transposed)
    if not any(successes):
        # An empty field guarantees that the first prototype's first dock fits.
        pair, _ = _assemble(rng, h, d, fw, masks, rows, cols, counts, [], cut=cut,
                            palette=palette, transposed=transposed)
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["colors", "shape", "stock"]))
    palette = list(map(int, rng.permutation(10)))
    first_orientation = bool(rng.integers(2))
    train = []
    # The two required evidence scenes use different structural roles and axes.
    train.append(_witness(rng, 0, palette, first_orientation))
    delta = int(rng.integers(1, 10))
    palette2 = [(c+delta) % 10 for c in palette]
    train.append(_witness(rng, 1, palette2, not first_orientation))
    train.extend(_random_pair(rng, shift) for _ in range(nt-2))
    tests = [_random_pair(rng, shift, True) for _ in range(nq)]
    seen = set()
    for pair in train+tests:
        original = {k: v.copy() for k, v in pair.items()}
        p = _parse(pair["input"])
        for variant in range(20):
            mapping = np.arange(10)
            if variant >= 10:
                mapping[p["bg"]], mapping[p["divider"]] = p["divider"], p["bg"]
            mapping = ((mapping+variant % 10) % 10).astype(np.int8)
            candidate = {k: mapping[v] for k, v in original.items()}
            if _key(candidate["input"]) not in seen:
                pair.update(candidate)
                break
        else:
            raise AssertionError("distinct recoloring fallback exhausted")
        seen.add(_key(pair["input"]))
    rng.shuffle(train)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


_ALTERNATIVES = {
    "unlimited_stock": {"quota": "unlimited"},
    "pixel_stock": {"quota": "area"},
    "attempt_budget": {"budget": "attempt"},
    "box_collision": {"footprint": "box"},
    "overwrite": {"collision": "overwrite"},
    "reverse_priority": {"order": "reverse"},
    "smallest_first": {"order": "size"},
    "destination_first": {"order": "destination"},
    "zipped_axes": {"coordinates": "zip"},
    "unkeyed_axes": {"coordinates": "union"},
}


def _features(p):
    correct, records = _simulate(p)
    differences = {name: not np.array_equal(correct, _simulate(p, **options)[0])
                   for name, options in _ALTERNATIVES.items()}
    ts = p["templates"]
    return dict(templates=len(ts), shapes=tuple(t["mask"].shape for t in ts),
                masks=tuple(_key(t["mask"]) for t in ts), stocks=tuple(len(t["stock"]) for t in ts),
                stock_areas=tuple(sum(len(g) for g in t["stock"]) for t in ts),
                products=tuple((len(t["rows"]), len(t["cols"])) for t in ts),
                leftover=sum(len(t["stock"])-r["consumed"] for t, r in zip(ts, records)),
                failures=sum(r["failed"] for r in records),
                accepted=sum(r["accepted"] for r in records),
                orientation=p["transposed"], background=p["bg"], rack=p["d"],
                dimensions=p["grid"].shape, obstacle=p["obstacle"] is not None,
                centerless=any(not t["mask"][t["mask"].shape[0]//2, t["mask"].shape[1]//2] for t in ts),
                solid=any(t["mask"].all() for t in ts), diagonal=any(len(_components(t["mask"])) > 1 for t in ts),
                stock_singletons=all(len(g) == 1 for t in ts for g in t["stock"]),
                differences=differences)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split types")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, features, parsed_train = set(), {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, supplied = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            p = _parse(inp)
            expected, records = _simulate(p)
            out = supplied.T.copy() if p["transposed"] else supplied
            g, d, cut, bg = p["grid"], p["d"], p["cut"], p["bg"]
            _require(out.shape == g.shape and np.array_equal(out, expected), "incorrect packing/consumption")
            _require(not np.array_equal(g, out), "identity example")
            _require(np.array_equal(g[:cut, :d], out[:cut, :d]), "prototype changed")
            protected = np.ones(g.shape, bool)
            protected[1:-1, d+2:-1] = False
            protected[cut+1:, :d] = False
            _require(np.array_equal(g[protected], out[protected]), "divider/frame changed")
            original_obstacles = g[1:-1, d+2:-1] != bg
            _require(np.array_equal(g[1:-1, d+2:-1][original_obstacles],
                                    out[1:-1, d+2:-1][original_obstacles]), "obstacle changed")
            for t, record in zip(p["templates"], records):
                color, count = t["color"], record["accepted"]
                _require(record["consumed"] == count <= len(t["stock"]), "stock conservation")
                target_cells = np.count_nonzero(out[1:-1, d+2:-1] == color)
                _require(target_cells == count*int(t["mask"].sum()), "partial stamp or intersection")
                remaining = _components(out[cut+1:, :d] == color)
                _require(len(remaining) == len(t["stock"])-count, "wrong remaining stock count")
                for i, group in enumerate(t["stock"]):
                    wanted = bg if i < count else color
                    _require(np.all(out[group[:, 0], group[:, 1]] == wanted), "non-prefix stock erasure")
            features[split].append(_features(p))
            if split == "train":
                parsed_train.append(p)
    training = features["train"]
    for name in _ALTERNATIVES:
        _require(any(f["differences"][name] for f in training), f"missing evidence against {name}")
    _require({f["orientation"] for f in training} == {False, True}, "orientation evidence")
    _require(len({f["background"] for f in training}) >= 2, "fixed background shortcut")
    _require(any(f["leftover"] and f["failures"] for f in training), "missing conservation witness")
    _require(any(a != b and x == y for f in training
                 for i, (a, x) in enumerate(zip(f["stocks"], f["stock_areas"]))
                 for b, y in zip(f["stocks"][i+1:], f["stock_areas"][i+1:])), "count/area confounding")
    for transform in range(1, 8):
        _require(any(not np.array_equal(_simulate(p)[0], _simulate(p, transform=transform)[0])
                     for p in parsed_train), "orientation of prototype not demonstrated")
    max_colors = max(f["templates"] for f in training)
    max_stock = max(n for f in training for n in f["stocks"])
    train_shapes = {shape for f in training for shape in f["shapes"]}
    _require(all(f["templates"] > max_colors or max(f["stocks"]) > max_stock
                 or ((5, 5) in f["shapes"] and (5, 5) not in train_shapes)
                 for f in features["test"]), "missing visible train/test shift")


def _signature(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded local tests, with measured variation over fixed seeds 0..41."""
    # A small hand-authored regression, outside the sampler's canvas range:
    # dock 1 is blocked, dock 2 contains a sheltered obstacle, dock 3 is free
    # but receives no copy because one two-pixel stock object has been spent.
    fixture = np.zeros((16, 21), dtype=np.int8)
    fixture[:, 5] = 1
    fixture[5, :5] = 1
    fixture[[0, -1], 6:] = 2
    fixture[:, [6, 20]] = 2
    fixture[1:4, 1:4] = 4
    fixture[2, 2] = 0
    fixture[4, [6, 20]] = 4
    fixture[0, [9, 13, 17]] = fixture[-1, [9, 13, 17]] = 4
    fixture[3, 8] = fixture[4, 13] = 3
    fixture[8, 1:3] = 4
    expected = fixture.copy()
    expected[8, 1:3] = 0
    for point in ((3, 12), (3, 13), (3, 14), (4, 12),
                  (4, 14), (5, 12), (5, 13), (5, 14)):
        expected[point] = 4
    _require(np.array_equal(solve(fixture), expected), "hand-authored packing regression")
    no_stock = fixture.copy()
    no_stock[8, 1:3] = 0
    _require(np.array_equal(solve(no_stock), no_stock), "zero stock should preserve input")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fs = [_features(_parse(pair["input"])) for puzzle in puzzles
          for split in ("train", "test") for pair in puzzle[split]]
    _require(len({_signature(p) for p in puzzles}) == 42, "puzzle diversity")
    _require(len({f["dimensions"] for f in fs}) >= 35, "geometry diversity")
    _require(len({m for f in fs for m in f["masks"]}) >= 45, "stencil diversity")
    _require(len({f["products"] for f in fs}) >= 25, "coordinate diversity")
    _require(len({f["stock_areas"] for f in fs}) >= 35, "stock-area diversity")
    _require(set(range(10)) == {f["background"] for f in fs}, "palette role diversity")
    _require({1, 2, 3} <= {f["templates"] for f in fs}, "color-count diversity")
    _require({1, 2, 3, 4, 5} <= {n for f in fs for n in f["stocks"]}, "quantity diversity")
    _require(any(f["diagonal"] for f in fs) and any(f["solid"] for f in fs), "mask topology diversity")
    _require(any(f["stock_singletons"] for f in fs) and any(not f["obstacle"] for f in fs), "simple scenes absent")
    for seed in (0, 17, 41):
        _require(_signature(puzzles[seed]) == _signature(generate_puzzle(seed)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Equivariance directly probes input-only solving with unseen palettes/layouts.
    for pair in puzzles[0]["train"]+puzzles[0]["test"]:
        a, b = pair["input"], pair["output"]
        _require(np.array_equal(solve(a.T), b.T), "transpose convention")
        mapping = np.array([7, 3, 0, 9, 5, 2, 8, 1, 6, 4], np.int8)
        _require(np.array_equal(solve(mapping[a]), mapping[b]), "literal color dependence")
        untouched = a.copy()
        solve(a)
        _require(np.array_equal(a, untouched), "solve mutated its input")
    bad = copy.deepcopy(puzzles[0])
    bad["test"][0]["output"].flat[0] = (int(bad["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt label accepted")
    for name, options in _ALTERNATIVES.items():
        bad = copy.deepcopy(puzzles[0])
        for pair in bad["train"]:
            p = _parse(pair["input"])
            alternative = _simulate(p, **options)[0]
            if not np.array_equal(alternative, _simulate(p)[0]):
                pair["output"] = alternative.T.copy() if p["transposed"] else alternative
                break
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"incorrect {name} label accepted")
    return {"seeds": 42, "pairs": len(fs), "input_shapes": len({f["dimensions"] for f in fs}),
            "stencil_masks": len({m for f in fs for m in f["masks"]}),
            "coordinate_configs": len({f["products"] for f in fs}),
            "stock_area_configs": len({f["stock_areas"] for f in fs})}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
