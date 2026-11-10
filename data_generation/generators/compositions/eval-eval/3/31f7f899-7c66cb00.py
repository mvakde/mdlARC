"""Ranked color locks: component height is the ordinal part of a color address.

Only NumPy and the standard library are used. No parent or artifact is read.
"""
from collections import Counter, defaultdict
from itertools import permutations
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer colors required")
    _require(np.all((0 <= a) & (a <= 9)), "color bounds")
    return a


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _height(mask):
    return 1 + max(r for r, c in mask)


def _width(mask):
    return 1 + max(c for r, c in mask)


def _components(grid, background, neighbors=FOUR):
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    result = []
    while unseen:
        first = min(unseen)
        color = int(grid[first])
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = r+dr, c+dc
                if p in unseen and int(grid[p]) == color:
                    unseen.remove(p)
                    stack.append(p)
        result.append(dict(color=color, cells=tuple(sorted(cells)), mask=_normal(cells)))
    return result


def _restore(canonical, vertical, upward):
    a = canonical[::-1] if upward else canonical
    return a.T.copy() if vertical else a.copy()


def _analyze(input_grid):
    """Recover spine, direction, chambers and masks from input pixels alone."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(int(grid[p]) == bg for p in
                 ((0, -1), (-1, 0), (-1, -1))), "background corners")
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    _require(len(rows)+len(cols) == 1, "one full carrier required")
    vertical = bool(len(cols))
    a = grid.T.copy() if vertical else grid.copy()
    row = int(cols[0] if vertical else rows[0])
    line = a[row]
    exclusive = [int(color) for color in np.unique(line)
                 if np.count_nonzero(a == color) == np.count_nonzero(line == color)]
    _require(len(exclusive) == 1, "carrier color must be exclusive to its line")
    carrier = exclusive[0]
    _require(carrier != bg and line[0] == line[-1] == carrier, "exposed carrier ends")
    headers, x = [], 0
    while x < len(line):
        color = int(line[x])
        end = x+1
        while end < len(line) and int(line[end]) == color:
            end += 1
        if color != carrier:
            _require(end-x >= 3 and x > 0 and end < len(line), "chamber header width")
            _require(line[x-1] == carrier and line[end] == carrier, "separate chambers")
            headers.append(dict(left=x, right=end-1, rail=color))
        x = end
    _require(2 <= len(headers) <= 7, "chamber count")

    def is_dock_row(y, p):
        if not 0 <= y < a.shape[0]:
            return False
        l, r, rail = p["left"], p["right"], p["rail"]
        route = int(a[y, l+1])
        return (int(a[y, l]) == int(a[y, r]) == rail and
                route not in (bg, carrier, rail) and np.all(a[y, l+1:r] == route))

    sides = [d for d in (-1, 1) if all(is_dock_row(row+d, p) for p in headers)]
    _require(len(sides) == 1, "docks must determine the direction")
    upward = sides[0] == -1
    if upward:
        a = a[::-1].copy()
        row = a.shape[0]-1-row
    _require(row >= 2 and np.all(a[row-1] == bg), "clear staging separator")
    rebuilt = np.full_like(a, bg)
    rebuilt[row] = a[row]
    ports = []
    for p in headers:
        p = p.copy()
        l, r, rail = p["left"], p["right"], p["rail"]
        route = int(a[row+1, l+1])
        bottom = row+1
        while bottom+1 < a.shape[0] and (np.all(a[bottom+1, l+1:r] == route) and
                int(a[bottom+1, l]) == int(a[bottom+1, r]) == rail):
            bottom += 1
        _require(bottom < a.shape[0]-1, "background far margin required")
        p.update(route=route, top=row+1, bottom=bottom)
        rebuilt[row+1:bottom+1, l:r+1] = a[row+1:bottom+1, l:r+1]
        ports.append(p)
    rebuilt[:row] = a[:row]
    _require(np.array_equal(rebuilt, a), "unexplained cells outside staging/docks")
    _require(np.all(a[:row, 0] == bg) and np.all(a[:row, -1] == bg), "inset staging")
    components = _components(a[:row], bg)
    _require(components, "empty staging field")
    by_color = defaultdict(list)
    by_route = defaultdict(list)
    for comp in components:
        by_color[comp["color"]].append(comp)
    for p in ports:
        by_route[p["route"]].append(p)
    _require(set(by_color) == set(by_route), "every address must be used")
    for route, group in by_color.items():
        _require(len(group) == len(by_route[route]), "one component per matching chamber")
        ties = {}
        for comp in group:
            mask, h = comp["mask"], _height(comp["mask"])
            _require(ties.setdefault(h, mask) == mask, "different masks tied in height")
            for p in by_route[route]:
                _require(h <= p["bottom"]-p["top"]+1 and
                         _width(mask) <= p["right"]-p["left"]-1,
                         "all matching chambers must fit every mask")
    return dict(canonical=a, background=bg, carrier=carrier, row=row,
                vertical=vertical, upward=upward, ports=ports, components=components,
                by_color=by_color, by_route=by_route)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    out = a["canonical"].copy()
    out[:a["row"]] = a["background"]
    for route, ports in a["by_route"].items():
        pieces = sorted(a["by_color"][route], key=lambda c: _height(c["mask"]))
        for p, comp in zip(ports, pieces):
            mask = comp["mask"]
            top = p["bottom"]-_height(mask)+1
            for r, c in mask:
                out[top+r, p["left"]+1+c] = p["rail"]
    return _restore(out, a["vertical"], a["upward"])


def _check_pair(pair):
    """Relational check: inspect stamps and conserved masks, without solve()."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas preservation")
    a = _analyze(inp)
    b = out.T.copy() if a["vertical"] else out.copy()
    if a["upward"]:
        b = b[::-1].copy()
    _require(np.all(b[:a["row"]] == a["background"]), "staging must be erased")
    infrastructure = np.ones(b.shape, dtype=bool)
    infrastructure[:a["row"]] = False
    recovered = defaultdict(list)
    for p in a["ports"]:
        interior = b[p["top"]:p["bottom"]+1, p["left"]+1:p["right"]]
        _require(np.all((interior == p["route"]) | (interior == p["rail"])), "interior palette")
        cells = [tuple(map(int, q)) for q in np.argwhere(interior == p["rail"])]
        _require(cells and min(c for r, c in cells) == 0 and
                 max(r for r, c in cells) == interior.shape[0]-1, "edge registration")
        recovered[p["route"]].append(_normal(cells))
        infrastructure[p["top"]:p["bottom"]+1, p["left"]+1:p["right"]] = False
    _require(np.array_equal(b[infrastructure], a["canonical"][infrastructure]), "fixed infrastructure")
    for route, masks in recovered.items():
        heights = [_height(m) for m in masks]
        _require(heights == sorted(heights), "per-address increasing height")
        _require(Counter(masks) == Counter(c["mask"] for c in a["by_color"][route]),
                 "exact monochromatic mask conservation by address")
    return a


def _shape(rng, h, max_width):
    w = int(rng.integers(1, max_width+1))
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1 and min(h, w) >= 3:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
    elif kind == 2:
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif kind == 3:
        cells, r, c = {(0, 0)}, 0, 0
        while r < h-1 or c < w-1:
            if r < h-1 and (c == w-1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            cells.add((r, c))
    else:
        spine = int(rng.integers(w))
        cells = {(r, spine) for r in range(h)}
        cross = int(rng.integers(h))
        cells.update((cross, c) for c in range(w))
        if kind == 4:
            for r in range(h):
                end = int(rng.integers(w))
                cells.update((r, c) for c in range(min(spine, end), max(spine, end)+1))
        else:
            target = int(rng.integers(len(cells), h*w+1))
            while len(cells) < target:
                frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in FOUR
                                   if 0 <= r+dr < h and 0 <= c+dc < w and
                                   (r+dr, c+dc) not in cells})
                cells.add(frontier[int(rng.integers(len(frontier)))])
    return tuple(sorted(cells))


def _is_frame(mask):
    h, w = _height(mask), _width(mask)
    return min(h, w) >= 3 and set(mask) == {(r, c) for r in range(h) for c in range(w)
                                                       if r in (0, h-1) or c in (0, w-1)}


def _pieces(rng, n, routes, cap, witness, tall):
    """Return address-labelled masks and an optional nested source relation."""
    if witness == 1:
        h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        frame = tuple((r, c) for r in range(h) for c in range(w)
                      if r in (0, h-1) or c in (0, w-1))
        inner_h = int(rng.integers(1, h-1))
        b_width = 1 if w == 4 else 2
        inner = _shape(rng, inner_h, min(w-2, b_width))
        other_h = int(rng.choice([k for k in range(1, 7) if k != inner_h]))
        masks = [(0, frame), (0, tuple((r, 0) for r in range(h+1))),
                 (0, _shape(rng, int(rng.integers(1, 3)), w)),
                 (1, inner), (1, _shape(rng, other_h, b_width))]
        return masks, (0, 3)
    counts = [3, 1] if witness == 2 else [1]*routes
    if witness != 2:
        for _ in range(n-routes):
            counts[int(rng.integers(routes))] += 1
    # Shared per-address width budget guarantees every dock can fit its group.
    budget = 30-2*n-(n+1)
    limits = [1]*routes
    for _ in range(20):
        choices = [i for i in range(routes) if limits[i] < 5 and
                   sum(k*w for k, w in zip(counts, limits))+counts[i] <= budget]
        if not choices or rng.random() < .16:
            break
        limits[int(rng.choice(choices))] += 1
    masks = []
    for i, count in enumerate(counts):
        heights = list(map(int, rng.choice(np.arange(1, cap+1), count, replace=False)))
        if tall and i == 0:
            heights[0] = cap
            heights = list(dict.fromkeys(heights))
            heights += [k for k in range(1, cap+1) if k not in heights][:count-len(heights)]
        group = [_shape(rng, h, limits[i]) for h in heights]
        if count >= 2 and ((witness == 2 and i == 0) or (not tall and rng.random() < .25)):
            group[1] = group[0]
        masks.extend((i, m) for m in group)
    nested = []
    for i, (route, outer) in enumerate(masks):
        if _is_frame(outer):
            for j, (other, inner) in enumerate(masks):
                if other != route and _height(inner) <= _height(outer)-2 and _width(inner) <= _width(outer)-2:
                    nested.append((i, j))
    return masks, nested[int(rng.integers(len(nested)))] if nested and rng.random() < .6 else None


def _sample_pair(rng, witness, n, routes, cap, tall, direction):
    masks, nested = _pieces(rng, n, routes, cap, witness, tall)
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, addresses = palette[0], palette[1], palette[2:2+routes]
    rail_pool = palette[2+routes:]
    group_widths = [max(_width(m) for i, m in masks if i == route) for route in range(routes)]
    group_heights = [max(_height(m) for i, m in masks if i == route) for route in range(routes)]
    port_routes = [i for i, m in masks]
    rng.shuffle(port_routes)
    # The rich lesson makes per-color order differ from one global height sort.
    if witness == 1:
        for _ in range(32):
            cursors = [0]*routes
            sorted_h = [sorted(_height(m) for i, m in masks if i == route) for route in range(routes)]
            sequence = []
            for i in port_routes:
                sequence.append(sorted_h[i][cursors[i]])
                cursors[i] += 1
            if sequence != sorted(sequence):
                break
            rng.shuffle(port_routes)
        else:
            port_routes = [0, 0, 1, 0, 1]  # the frame precedes the shorter inner piece
    widths = [group_widths[i]+2 for i in port_routes]
    minimum = sum(widths)+n+1
    _require(minimum <= 30, "width budget")
    width = int(rng.integers(minimum, 31))
    extra = width-minimum
    # Spare width can enlarge individual docks or exposed carrier segments.
    allocation = list(map(int, rng.multinomial(extra, [1/(2*n+1)]*(2*n+1))))
    widths = [w+allocation[i] for i, w in enumerate(widths)]
    gaps = [1+x for x in allocation[n:]]
    depths = [group_heights[i]+int(rng.integers(0, 4)) for i in port_routes]
    stage = max(_height(m) for i, m in masks)+int(rng.integers(2 if witness else 0, 6))
    margin = int(rng.integers(1, 4))
    stage = min(stage, 30-2-max(depths)-margin)
    row, height = stage+1, stage+2+max(depths)+margin
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[row] = carrier
    ports, x, previous_rails = [], gaps[0], {}
    for k, (route, w, depth) in enumerate(zip(port_routes, widths, depths)):
        choices = rail_pool + [c for c in addresses if c != addresses[route]]
        # At least two output colors in every repeated-address witness group.
        if witness and route in previous_rails:
            choices = [c for c in choices if c != previous_rails[route]]
        rail = int(rng.choice(choices))
        previous_rails[route] = rail
        p = dict(left=x, right=x+w-1, route=addresses[route], rail=rail,
                 top=row+1, bottom=row+depth, group=route)
        ports.append(p)
        inp[row, x:x+w] = rail
        inp[row+1:row+depth+1, x:x+w] = addresses[route]
        inp[row+1:row+depth+1, [x, x+w-1]] = rail
        x += w+gaps[k+1]
    groups = [[(j, 0, 0)] for j in range(len(masks))]
    if nested:
        outer, inner = nested
        oh, ow = _height(masks[outer][1]), _width(masks[outer][1])
        ih, iw = _height(masks[inner][1]), _width(masks[inner][1])
        groups[outer].append((inner, 1+int(rng.integers(oh-ih-1)),
                             1+int(rng.integers(ow-iw-1))))
        groups[inner] = []
    groups = [g for g in groups if g]
    if witness:
        allowed = []
        for order in permutations(range(len(groups))):
            hs = [_height(masks[j][1]) for k in order for j, r, c in groups[k] if masks[j][0] == 0]
            if hs != sorted(hs) and hs[::-1] != sorted(hs):
                allowed.append(order)
        order = allowed[int(rng.integers(len(allowed)))]
    else:
        order = list(map(int, rng.permutation(len(groups))))
    groups = [groups[k] for k in order]
    source_widths = [max(dc+_width(masks[j][1]) for j, dr, dc in g) for g in groups]
    free = width-sum(source_widths)-len(groups)-1
    _require(free >= 0, "staging horizontal fit")
    source_gaps = [1+int(v) for v in rng.multinomial(free, [1/(len(groups)+1)]*(len(groups)+1))]
    x, bottoms = source_gaps[0], {}
    for k, group in enumerate(groups):
        gh = max(dr+_height(masks[j][1]) for j, dr, dc in group)
        tops = list(range(stage-gh+1))
        if witness == 1 and group[0][0] in (0, 1):
            tops = [t for t in tops if t+gh-1 not in bottoms.get(0, set())]
        top = int(rng.choice(tops))
        for j, dr, dc in group:
            route, mask = masks[j]
            for r, c in mask:
                inp[top+dr+r, x+dc+c] = addresses[route]
            bottoms.setdefault(route, set()).add(top+dr+_height(mask)-1)
        x += source_widths[k]+source_gaps[k+1]
    # Labels use the known masks and docks, not the input parser or solve().
    out = inp.copy()
    out[:row] = bg
    ranked = {i: sorted([m for route, m in masks if route == i], key=_height) for i in range(routes)}
    cursors = [0]*routes
    for p in ports:
        i = p["group"]
        m = ranked[i][cursors[i]]
        cursors[i] += 1
        for r, c in m:
            out[p["bottom"]-_height(m)+1+r, p["left"]+1+c] = p["rail"]
    vertical, upward = direction in (2, 3), direction in (1, 3)
    return {"input": _restore(inp, vertical, upward), "output": _restore(out, vertical, upward)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # more chambers / taller masks / more addresses
    mixed = bool(rng.integers(2))
    fixed_direction = int(rng.integers(4))
    train_directions = ([int(rng.integers(2)), 2+int(rng.integers(2))] +
                        [int(rng.integers(4)) for _ in range(nt-2)]) if mixed else [fixed_direction]*nt
    test_directions = [int(rng.choice(train_directions)) for _ in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = index+1 if split == "train" and index < 2 else 0
            if witness == 1:
                n, routes, cap, tall = 5, 2, 6, False
            elif witness == 2:
                n, routes, cap, tall = 4, 2, 6, False
            elif split == "train":
                n, routes, cap, tall = int(rng.integers(2, 6)), 2, 6, False
            elif regime == 0:
                n, routes, cap, tall = 6, int(rng.integers(2, 4)), 6, False
            elif regime == 1:
                n, routes, cap, tall = int(rng.integers(3, 6)), 2, int(rng.integers(7, 10)), True
            else:
                n, routes, cap, tall = int(rng.integers(4, 6)), 3, 6, False
            direction = train_directions[index] if split == "train" else test_directions[index]
            for _ in range(64):
                pair = _sample_pair(rng, witness, n, routes, cap, tall, direction)
                if split == "test":
                    f = _features(_analyze(pair["input"]))
                    if not (f["source"] and f["global_sort"]):
                        continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("bounded distinct-input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    groups = a["by_color"]
    area_disagrees = source_disagrees = reversal_disagrees = False
    ties, independence = False, False
    for comps in groups.values():
        ranked = [c["mask"] for c in sorted(comps, key=lambda c: _height(c["mask"]))]
        area_disagrees |= ranked != [c["mask"] for c in sorted(comps, key=lambda c: len(c["mask"]))]
        source = sorted(comps, key=lambda c: min(col for r, col in c["cells"]))
        source_disagrees |= ranked != [c["mask"] for c in source]
        reversal_disagrees |= ranked != [c["mask"] for c in source[::-1]]
        ties |= len({_height(c["mask"]) for c in comps}) < len(comps)
        independence |= len({max(r for r, col in c["cells"]) for c in comps}) > 1
    cursors = defaultdict(int)
    ranked = {r: sorted(comps, key=lambda c: _height(c["mask"])) for r, comps in groups.items()}
    assigned_h = []
    for p in a["ports"]:
        assigned_h.append(_height(ranked[p["route"]][cursors[p["route"]]]["mask"]))
        cursors[p["route"]] += 1
    stage, bg = a["canonical"][:a["row"]], a["background"]
    contact = any(stage[r, c] != bg and stage[r+dr, c+dc] != bg and
                  stage[r, c] != stage[r+dr, c+dc]
                  for r in range(stage.shape[0]) for c in range(stage.shape[1])
                  for dr, dc in ((0, 1), (1, 0))
                  if r+dr < stage.shape[0] and c+dc < stage.shape[1])
    return dict(area=area_disagrees, source=source_disagrees, reversal=reversal_disagrees,
                global_sort=assigned_h != sorted(assigned_h), ties=ties, independent=independence,
                contact=contact, height=max(_height(c["mask"]) for c in a["components"]),
                count=len(a["ports"]), addresses=len(groups), axis=a["vertical"])


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _check_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            four = {frozenset(c["cells"]) for c in a["components"]}
            eight = {frozenset(c["cells"]) for c in _components(a["canonical"][:a["row"]], a["background"], EIGHT)}
            _require(four == eight, "diagonal-only component ambiguity")
            features[split].append(_features(a))
    train = features["train"]
    for feature in ("area", "source", "reversal", "global_sort", "contact", "independent", "ties"):
        _require(any(f[feature] for f in train), "missing training witness: "+feature)
    taught_axes = {f["axis"] for f in train if f["source"] and f["reversal"]}
    _require({f["axis"] for f in features["test"]} <= taught_axes, "untaught active spine sort")
    _require(all(f["source"] and f["global_sort"] for f in features["test"]),
             "every generated test must require rank and color partition")
    _require(len({f["count"] for f in train}) >= 2, "training count variation")


def self_test() -> dict:
    """Bounded 42-seed sweep, exact pixels, mutations, counts and variation."""
    # A hand-specified asymmetric L, tall bar and singleton, in all directions.
    fixture = np.zeros((11, 17), dtype=np.int8)
    fixture[5] = 6
    for left, depth, route, rail in ((1, 4, 1, 3), (6, 2, 2, 4), (11, 3, 1, 5)):
        fixture[5, left:left+4] = rail
        fixture[6:6+depth, left:left+4] = route
        fixture[6:6+depth, [left, left+3]] = rail
    fixture[0:3, 2] = 1
    fixture[2, 6] = fixture[3, 6] = fixture[3, 7] = 1
    fixture[0, 10] = 2
    exact = fixture.copy()
    exact[:5] = 0
    exact[8, 2] = exact[9, 2] = exact[9, 3] = 3
    exact[7, 7] = 4
    exact[6:9, 12] = 5
    for vertical in (False, True):
        for upward in (False, True):
            pair = {"input": _restore(fixture, vertical, upward),
                    "output": _restore(exact, vertical, upward)}
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "exact translation fixture")
            _check_pair(pair)
    bad = fixture.copy()
    bad[2, 2] = 0  # same height as the L, but a different mask: forbidden tie.
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous height tie accepted")
    bad = fixture.copy()
    bad[0, 10] = 0  # the route-2 dock no longer has a component.
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("address multiplicity corruption accepted")
    puzzles, shapes, masks, directions, palettes = set(), set(), set(), set(), set()
    metrics, examples = Counter(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(_key(pair[role]) for split in ("train", "test")
                          for pair in p[split] for role in ("input", "output"))
        puzzles.add(signature)
        if seed == 0:
            q = generate_puzzle(seed)
            _require(all(np.array_equal(x[role], y[role]) for s in p for x, y in zip(p[s], q[s])
                         for role in ("input", "output")), "reproducibility")
        local = {s: [] for s in p}
        for split in ("train", "test"):
            for pair in p[split]:
                examples.append(pair)
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "solve disagreement")
                _require(np.array_equal(before, pair["input"]), "solver mutates input")
                a = _analyze(before)
                f = _features(a)
                local[split].append(f)
                shapes.add(before.shape)
                masks.update(c["mask"] for c in a["components"])
                directions.add((a["vertical"], a["upward"]))
                palettes.add((a["background"], a["carrier"]))
                metrics.update({"pairs": 1, "frames": sum(_is_frame(c["mask"]) for c in a["components"]),
                                "tied_pairs": int(f["ties"]), "contact_pairs": int(f["contact"]),
                                "mixed_width_pairs": int(len({p["right"]-p["left"] for p in a["ports"]}) > 1),
                                "shared_rail_pairs": int(len({p["rail"] for p in a["ports"]}) < len(a["ports"])),
                                "cross_role_pairs": int(any(p["rail"] in a["by_route"] for p in a["ports"]))})
        metrics["more_chambers"] += max(f["count"] for f in local["test"]) > max(f["count"] for f in local["train"])
        metrics["taller_masks"] += max(f["height"] for f in local["test"]) > max(f["height"] for f in local["train"])
        metrics["more_addresses"] += max(f["addresses"] for f in local["test"]) > max(f["addresses"] for f in local["train"])
    _require(len(puzzles) == 42 and len(shapes) >= 35 and len(masks) >= 50, "structural diversity")
    _require(len(directions) == 4 and len(palettes) >= 25, "direction/palette diversity")
    _require(all(metrics[k] for k in ("frames", "tied_pairs", "contact_pairs", "mixed_width_pairs",
                                      "shared_rail_pairs", "cross_role_pairs", "more_chambers", "taller_masks", "more_addresses")),
             "missing measured variation")
    # Deliberately wrong complete outputs, inspected by the relational checker.
    rejected = set()
    modes = {"area", "source_order", "reverse_source", "global_sort", "bounding_box", "center"}
    for pair in examples:
        if rejected == modes:
            break
        a = _analyze(pair["input"])
        for mode in sorted(modes-rejected):
            assignments = []
            if mode == "global_sort":
                assignments = list(zip(a["ports"], sorted(a["components"], key=lambda c: _height(c["mask"]))))
            else:
                for route, ports in a["by_route"].items():
                    comps = a["by_color"][route]
                    if mode == "area":
                        comps = sorted(comps, key=lambda c: len(c["mask"]))
                    elif mode in ("source_order", "reverse_source"):
                        comps = sorted(comps, key=lambda c: min(col for r, col in c["cells"]),
                                       reverse=mode == "reverse_source")
                    else:
                        comps = sorted(comps, key=lambda c: _height(c["mask"]))
                    assignments.extend(zip(ports, comps))
            wrong = a["canonical"].copy()
            wrong[:a["row"]] = a["background"]
            fits = True
            for p, comp in assignments:
                mask = comp["mask"]
                h, w = _height(mask), _width(mask)
                if h > p["bottom"]-p["top"]+1 or w > p["right"]-p["left"]-1:
                    fits = False
                    break
                if mode == "bounding_box":
                    mask = tuple((r, c) for r in range(h) for c in range(w))
                left = p["left"]+1
                if mode == "center":
                    left += (p["right"]-p["left"]-1-w)//2
                for r, c in mask:
                    wrong[p["bottom"]-h+1+r, left+c] = p["rail"]
            wrong = _restore(wrong, a["vertical"], a["upward"])
            if not fits or np.array_equal(wrong, pair["output"]):
                continue
            try:
                _check_pair({"input": pair["input"], "output": wrong})
            except AssertionError:
                rejected.add(mode)
            else:
                raise AssertionError("accepted shortcut mutation: "+mode)
    _require(rejected == modes, "unexercised shortcut rejection")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Reject a wrong label and an address count corruption.
    p = generate_puzzle(501, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, distinct_puzzles=len(puzzles), shapes=len(shapes),
                normalized_masks=len(masks), directions=len(directions),
                rejected_shortcuts=sorted(rejected), metrics=dict(metrics))
