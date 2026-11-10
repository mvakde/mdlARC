"""Self-stencil gated endpoints, followed by a union of shortest tile routes.

Only NumPy and the standard library are used. Generation and inference have no
file access, global random state, saved examples, or persistent grid cache.
"""

from collections import deque
from functools import lru_cache
from hashlib import sha256

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))
HOST_SEEDS = (0, 118294947, 68275604, 623708411, 219393322, 500469792,
              447987819, 916579426, 380650275, 692619567, 606298164,
              945538987, 252569720, 283765731, 500401515, 100212174,
              335575886, 452568746, 978486370, 184052040, 726586740,
              62042683, 947054342, 427914908, 395173395, 939828856,
              413068714, 458678261, 483566388, 694267755, 513273285,
              814113281, 850617999, 579964995, 437202385, 63625719,
              799580966, 241164348, 42309890, 216207737, 453665686,
              870684521)


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    array = np.asarray(value)
    _need(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
          "grid dimensions must be 1..30")
    _need(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
    _need(np.all((array >= 0) & (array <= 9)), "invalid grid color")
    return array.astype(np.int8, copy=False)


def _fingerprint(grid):
    return grid.shape, sha256(grid.astype(np.int8).tobytes()).digest()


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and groups[-1][1] + 1 == i:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _lattice(grid):
    divider = int(grid[0, 0])
    rg = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cg = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    _need(len(rg) >= 4 and len(cg) >= 4, "missing periodic dividers")
    _need(rg[0][0] == cg[0][0] == 0 and rg[-1][1] == grid.shape[0]-1
          and cg[-1][1] == grid.shape[1]-1, "missing outer divider")
    ts = {b-a+1 for a, b in rg+cg}
    hs = {rg[i+1][0]-rg[i][1]-1 for i in range(len(rg)-1)}
    ws = {cg[i+1][0]-cg[i][1]-1 for i in range(len(cg)-1)}
    _need(len(ts) == len(hs) == len(ws) == 1, "irregular lattice")
    h, w, t = hs.pop(), ws.pop(), ts.pop()
    _need((h, w) in SHAPES and t in (1, 2), "unsupported tile geometry")
    return (h, w, t, len(rg)-1, len(cg)-1), divider


def _slice(geometry, r, c):
    h, w, t, _, _ = geometry
    y, x = t+r*(h+t), t+c*(w+t)
    return np.s_[y:y+h, x:x+w]


def _points(mask):
    return frozenset(tuple(map(int, p)) for p in np.argwhere(mask))


def _endpoints(points):
    degrees = {p: sum((p[0]+dr, p[1]+dc) in points for dr, dc in DIRS)
               for p in points}
    ends = tuple(sorted(p for p in points if degrees[p] == 1))
    _need(len(ends) == 2 and all(n in (1, 2) for n in degrees.values()),
          "ink must be an unbranched path with two endpoints")
    visited, todo = set(), [ends[0]]
    while todo:
        p = todo.pop()
        if p not in visited:
            visited.add(p)
            todo.extend((p[0]+dr, p[1]+dc) for dr, dc in DIRS
                        if (p[0]+dr, p[1]+dc) in points)
    _need(visited == set(points), "ink path must be connected")
    return ends


def _chamber(glyph, ordinary_colors, ink):
    h, w = glyph.shape
    choices = []
    for marker in sorted(ordinary_colors):
        centers = np.argwhere(glyph == marker)
        if len(centers) != 1:
            continue
        y, x = map(int, centers[0])
        if not (1 <= y < h-1 and 1 <= x < w-1):
            continue
        background = next(iter(ordinary_colors-{marker}))
        for dr, dc in DIRS:
            shell = {(y+a, x+b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
            walls = shell - {(y, x), (y+dr, x+dc)}
            if not all(glyph[p] == ink for p in walls):
                continue
            ray = []
            r, c = y+dr, x+dc
            while 0 <= r < h and 0 <= c < w:
                ray.append((r, c))
                r, c = r+dr, c+dc
            if all(glyph[p] == background for p in ray):
                nonink = glyph[glyph != ink]
                if np.count_nonzero(nonink == marker) == 1 and np.all(
                        np.isin(nonink, (background, marker))):
                    choices.append((background, marker, (y, x), (dr, dc), tuple(ray)))
    _need(len(choices) == 1, "glyph needs one marked, uniquely gated 3x3 chamber")
    return choices[0]


def _origins(mask, rows, cols, supplied):
    h, w = mask.shape
    return [(r, c) for r in range(rows-h+1) for c in range(cols-w+1)
            if all(0 <= y-r < h and 0 <= x-c < w and mask[y-r, x-c]
                   for y, x in supplied)]


def _route(rows, cols, active, ends, direction, rocks=(), gated=True):
    """All route lengths are measured in coarse tile-to-tile edges."""
    ends = tuple(ends)
    dr, dc = direction
    gates = {p: (p[0]+dr, p[1]+dc) for p in ends}
    blocked = (set(active)-set(ends)) | set(rocks)
    for p in gates.values():
        _need(0 <= p[0] < rows and 0 <= p[1] < cols and p not in active
              and p not in rocks, "terminal gate must enter an ordinary tile")

    def distances(start):
        dist = np.full((rows, cols), -1, dtype=np.int16)
        dist[start] = 0
        todo = deque([start])
        while todo:
            p = todo.popleft()
            for a, b in DIRS:
                q = (p[0]+a, p[1]+b)
                if not (0 <= q[0] < rows and 0 <= q[1] < cols):
                    continue
                if q in blocked or dist[q] >= 0:
                    continue
                if gated and ((p in gates and gates[p] != q) or
                              (q in gates and gates[q] != p)):
                    continue
                dist[q] = dist[p]+1
                todo.append(q)
        return dist

    d1, d2 = (distances(p) for p in ends)
    length = int(d1[ends[1]])
    _need(length >= 0, "gated terminals are disconnected")
    union = _points((d1 >= 0) & (d2 >= 0) & (d1+d2 == length))-set(ends)
    return length, frozenset(union)


def _infer(input_grid):
    grid = _grid(input_grid)
    geometry, divider = _lattice(grid)
    h, w, t, rows, cols = geometry
    _need(rows >= h and cols >= w, "stencil window does not fit")
    records = {}
    for r in range(rows):
        for c in range(cols):
            tile = grid[_slice(geometry, r, c)]
            _need(not np.any(tile == divider), "divider inside a tile")
            records.setdefault(tile.tobytes(), [tile.copy(), set()])[1].add((r, c))
    ordered = sorted(records.values(), key=lambda item: len(item[1]), reverse=True)
    _need(2 <= len(ordered) <= 3, "expected ordinary, glyph, and optional solid tiles")
    ordinary, ordinary_positions = ordered[0]
    _need(len(ordinary_positions) > rows*cols/2, "ordinary tile needs a strict majority")
    oc = set(map(int, np.unique(ordinary)))
    _need(len(oc) == 2, "ordinary tiles need two colors")
    glyphs = [(tile, positions) for tile, positions in ordered[1:]
              if len(np.unique(tile)) == 3 and oc <= set(map(int, np.unique(tile)))]
    _need(len(glyphs) == 1, "cannot identify exceptional glyph")
    glyph, supplied = glyphs[0]
    _need(len(supplied) >= 2, "supply at least two glyph copies")
    ink = (set(map(int, np.unique(glyph)))-oc).pop()
    rocks = set()
    for tile, positions in ordered[1:]:
        if np.array_equal(tile, glyph):
            continue
        _need(np.all(tile == ink), "unrecognized obstacle tile")
        rocks.update(positions)
    background, marker, center, direction, ray = _chamber(glyph, oc, ink)
    mask = glyph == ink
    local = _points(mask)
    local_ends = _endpoints(local)
    origins = _origins(mask, rows, cols, supplied)
    _need(len(origins) == 1, "supplied glyphs must determine exactly one translation")
    origin = origins[0]
    active = frozenset((origin[0]+r, origin[1]+c) for r, c in local)
    _need(supplied < active and not (active & rocks), "invalid missing stencil sites")
    ends = tuple((origin[0]+r, origin[1]+c) for r, c in local_ends)
    length, union = _route(rows, cols, active, ends, direction, rocks)
    completed = grid.copy()
    for r, c in active-supplied:
        completed[_slice(geometry, r, c)] = glyph
    expected = completed.copy()
    for r, c in union:
        tile = expected[_slice(geometry, r, c)]
        tile[tile == background] = marker
    for r, c in ends:
        tile = expected[_slice(geometry, r, c)]
        for p in ray:
            tile[p] = marker
    return expected, {
        "geometry": geometry, "divider": divider, "ordinary": ordinary,
        "glyph": glyph, "ink": ink, "background": background, "marker": marker,
        "mask": mask, "origin": origin, "supplied": frozenset(supplied),
        "active": active, "ends": ends, "direction": direction, "ray": ray,
        "center": center, "rocks": frozenset(rocks), "length": length,
        "union": union, "completed": completed,
        "fragment": frozenset((r-origin[0], c-origin[1]) for r, c in supplied),
    }


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the full result solely from one input; never mutate that input."""
    return _infer(input_grid)[0]


@lru_cache(maxsize=4)
def _catalog(h, w):
    """All induced path extensions of a gated shell in these small rectangles.

    There are 4, 18, 18, and 96 configurations for the four supported shapes.
    This cache contains only tiny coordinate descriptions, never sampled grids.
    Inference does not consult it.
    """
    found = set()
    for y in range(1, h-1):
        for x in range(1, w-1):
            shell = {(y+a, x+b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
            for dr, dc in DIRS:
                ray, r, c = set(), y+dr, x+dc
                while 0 <= r < h and 0 <= c < w:
                    ray.add((r, c))
                    r, c = r+dr, c+dc

                def grow(points):
                    code = (tuple(sorted(points)), (y, x), (dr, dc))
                    if code in found:
                        return
                    found.add(code)
                    for a, b in _endpoints(points):
                        for dy, dx in DIRS:
                            p = (a+dy, b+dx)
                            if (0 <= p[0] < h and 0 <= p[1] < w and
                                    p not in shell | points | ray and
                                    sum((p[0]+u, p[1]+v) in points for u, v in DIRS) == 1):
                                grow(points | {p})

                grow(shell-{(y, x), (y+dr, x+dc)})
    return tuple(sorted(found))


def _mask(shape, points):
    mask = np.zeros(shape, dtype=bool)
    for p in points:
        mask[p] = True
    return mask


def _orientation_symmetries(mask):
    def cropped(a):
        ys, xs = np.nonzero(a)
        a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
        return a.shape, a.tobytes()
    target = cropped(mask)
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(a) for a in transforms]
    return {i for i, a in enumerate(transforms) if cropped(a) == target}


def _fragment(rng, mask, rows, cols, origin, ends, rocks, missing_ends):
    active = sorted(_points(mask))
    # Every example hides an endpoint and at least one nonterminal. The second
    # demonstration hides both endpoints, so adding just one cannot suffice.
    hidden_ends = set(ends if missing_ends == 2 else [ends[int(rng.integers(2))]])
    internal = sorted(set(active)-set(ends))
    hidden = hidden_ends | {internal[int(rng.integers(len(internal)))]}
    available = [p for p in active if p not in hidden]
    limit = min(len(available), (rows*cols-1)//2-len(rocks))
    if limit < 2:
        return None
    for _ in range(48):
        count = int(rng.integers(2, limit+1))
        chosen = rng.choice(len(available), size=count, replace=False)
        supplied = frozenset((origin[0]+available[int(i)][0],
                              origin[1]+available[int(i)][1]) for i in chosen)
        if _origins(mask, rows, cols, supplied) == [origin]:
            return supplied
    return None


def _paint(geometry, palette, ordinary, glyph, supplied, rocks):
    h, w, t, rows, cols = geometry
    divider, background, marker, ink = palette
    grid = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            tile = ink if (r, c) in rocks else glyph if (r, c) in supplied else ordinary
            grid[_slice(geometry, r, c)] = tile
    return grid


def _sample_pair(rng, palette, kind, both_missing=False, wall_witness=False,
                 familiar=None, forbidden_fragments=(), forbidden_layouts=(), used=()):
    for attempt in range(384):
        if familiar is None:
            choices = SHAPES if kind != "multi" else SHAPES[1:]
            shape = choices[int(rng.integers(len(choices)))]
            configs = _catalog(*shape)
            points, center, direction = configs[int(rng.integers(len(configs)))]
        else:
            shape, points, center, direction = familiar
        h, w = shape
        mask = _mask(shape, points)
        if wall_witness and _orientation_symmetries(mask) != {0}:
            continue
        local_ends = _endpoints(frozenset(points))
        # Late attempts use the most spacious thin-divider canvas. This is a
        # finite retry policy, not a change of rule or a hidden solving hint.
        t = 1 if attempt >= 192 else int(rng.choice((1, 2), p=(.8, .2)))
        maxr, maxc = (30-t)//(h+t), (30-t)//(w+t)
        rows = maxr if attempt >= 192 or rng.random() < .6 else int(rng.integers(h, maxr+1))
        cols = maxc if attempt >= 192 or rng.random() < .6 else int(rng.integers(w, maxc+1))
        geometry = (h, w, t, rows, cols)
        origin = (int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1)))
        active = frozenset((r+origin[0], c+origin[1]) for r, c in points)
        ends = tuple((r+origin[0], c+origin[1]) for r, c in local_ends)
        try:
            base_length, base_union = _route(rows, cols, active, ends, direction)
        except ValueError:
            continue
        gates = {(r+direction[0], c+direction[1]) for r, c in ends}
        candidates = sorted({(r, c) for r in range(rows) for c in range(cols)}-active-gates)
        max_rocks = min(6, len(candidates), (rows*cols-1)//2-2)
        if max_rocks < int(wall_witness):
            continue
        nrocks = int(rng.integers(int(wall_witness), max_rocks+1))
        if nrocks:
            rocks = set(candidates[int(i)] for i in rng.choice(len(candidates), nrocks, replace=False))
        else:
            rocks = set()
        if wall_witness:
            witnesses = sorted(base_union-gates)
            if not witnesses:
                continue
            # Replace one random wall by a wall on a previously shortest route.
            rocks.discard(sorted(rocks)[int(rng.integers(len(rocks)))])
            rocks.add(witnesses[int(rng.integers(len(witnesses)))])
        try:
            length, union = _route(rows, cols, active, ends, direction, rocks)
        except ValueError:
            continue
        multi = len(union) > length-1
        if kind in ("multi", "single") and multi != (kind == "multi"):
            continue
        if wall_witness and union == base_union:
            continue
        if (rows, cols, origin, frozenset(rocks)) in forbidden_layouts:
            continue
        supplied = _fragment(rng, mask, rows, cols, origin, local_ends, rocks,
                             2 if both_missing else 1)
        if supplied is None:
            continue
        fragment = frozenset((r-origin[0], c-origin[1]) for r, c in supplied)
        if fragment in forbidden_fragments:
            continue
        divider, background, marker, ink = palette
        labels = np.zeros(h*w, dtype=bool)
        labels[rng.choice(h*w, int(rng.integers(2, h*w-1)), replace=False)] = True
        ordinary = np.where(labels.reshape(shape), marker, background).astype(np.int8)
        # A copied non-ink background pixel must visibly change somewhere.
        nonink = sorted(set((r, c) for r in range(h) for c in range(w))
                        -set(points)-{center})
        ordinary[nonink[int(rng.integers(len(nonink)))]] = marker
        glyph = np.where(mask, ink, background).astype(np.int8)
        glyph[center] = marker
        grid = _paint(geometry, palette, ordinary, glyph, supplied, rocks)
        if _fingerprint(grid) in used:
            continue
        output, info = _infer(grid)
        return {"input": grid, "output": output}, info
    # Never silently switch to an unrelated template or a different convention.
    raise RuntimeError("bounded layout sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, train_info = {"train": [], "test": []}, set(), []
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.random() < .35)
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if not stable_palette and (split == "test" or index > 0):
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            first = split == "train" and index == 0
            second = split == "train" and index == 1
            kind = "single" if first else "multi" if second else "any"
            familiar, excluded, layouts = None, (), ()
            if split == "test":
                # A real shift with the same visible blueprint: a training
                # glyph is reused with a never-demonstrated supplied fragment.
                # Translation, rocks, geometry and route are sampled afresh.
                reference = train_info[1 if index % 2 == 0 else 0]
                h, w = reference["mask"].shape
                familiar = ((h, w), tuple(sorted(_points(reference["mask"]))),
                            reference["center"], reference["direction"])
                excluded = {f["fragment"] for f in train_info
                            if f["mask"].shape == (h, w) and
                            np.array_equal(f["mask"], reference["mask"])}
                layouts = {(f["geometry"][3], f["geometry"][4], f["origin"], f["rocks"])
                           for f in train_info if f["mask"].shape == (h, w) and
                           np.array_equal(f["mask"], reference["mask"])}
            pair, info = _sample_pair(rng, palette, kind,
                                      both_missing=second or split == "test",
                                      wall_witness=second,
                                      familiar=familiar, forbidden_fragments=excluded,
                                      forbidden_layouts=layouts, used=seen)
            puzzle[split].append(pair)
            seen.add(_fingerprint(pair["input"]))
            if split == "train":
                train_info.append(info)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    before, after = (_grid(pair[k]) for k in ("input", "output"))
    expected, info = _infer(before)
    _need(before.shape == after.shape and np.array_equal(expected, after),
          "output does not equal stencil completion plus gated route union")
    active, supplied = info["active"], info["supplied"]
    _need(not set(info["ends"]) <= supplied, "completion must supply a terminal")
    _need(bool((active-supplied)-set(info["ends"])), "need a missing nonterminal copy")
    ordinary, glyph, mask = info["ordinary"], info["glyph"], info["mask"]
    outside = ~mask
    outside[info["center"]] = False
    _need(np.any(ordinary[outside] != glyph[outside]), "whole-tile background copying must matter")
    # These invariant checks concern actual rendered pixels, not latent choices.
    _need(np.array_equal(before[before == info["divider"]], after[before == info["divider"]]),
          "divider changed")
    for r, c in info["rocks"]:
        _need(np.all(after[_slice(info["geometry"], r, c)] == info["ink"]), "solid wall changed")
    for r, c in active-set(info["ends"]):
        _need(np.array_equal(after[_slice(info["geometry"], r, c)], glyph),
              "passive completed glyph must be an exact full copy")
    return info


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _fingerprint(_grid(pair["input"]))
            _need(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    _need(any(len(f["union"]) == f["length"]-1 for f in train), "missing single-route demonstration")
    _need(any(len(f["union"]) > f["length"]-1 for f in train), "missing route-union demonstration")
    _need(any(not (set(f["ends"]) & f["supplied"]) for f in train), "hide both endpoints in training")
    _need(any(set(f["ends"]) != {min(f["active"]), max(f["active"])} for f in train),
          "training must distinguish topology from reading-order endpoints")
    orientation_rules = set(range(8))
    for f in train:
        orientation_rules &= _orientation_symmetries(f["mask"])
    _need(orientation_rules == {0}, "training must distinguish preserved orientation")
    _need(any(_route(*f["geometry"][3:], f["active"], f["ends"], f["direction"])[1]
              != f["union"] for f in train), "training must show consequential walls")
    _need(any(_route(*f["geometry"][3:], f["active"], f["ends"], f["direction"],
                     f["rocks"], gated=False)[1] != f["union"] for f in train),
          "training must distinguish ungated endpoints")
    for f in infos["test"]:
        matches = [g for g in train if g["mask"].shape == f["mask"].shape
                   and np.array_equal(g["mask"], f["mask"])]
        _need(bool(matches) and all(g["fragment"] != f["fragment"] for g in matches),
              "test needs a familiar stencil with a new supplied fragment")
        layout = (f["geometry"][3:], f["origin"], f["rocks"])
        _need(all((g["geometry"][3:], g["origin"], g["rocks"]) != layout for g in matches),
              "test must use a new coarse routing layout")


def self_test():
    """Bounded, streaming 42-seed checks; nothing is written to disk."""
    fingerprints, shapes, designs, directions, geometries, unions, counts = (set() for _ in range(7))
    pairs, multiple, wall_effects, gate_effects = 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _check_pair(pair)
                digest.update(pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                designs.add(_fingerprint(info["mask"].astype(np.int8)))
                directions.add(info["direction"])
                geometries.add(info["geometry"])
                unions.add((info["length"], len(info["union"])))
                counts.add((len(info["supplied"]), len(info["active"]-info["supplied"])))
                rows, cols = info["geometry"][3:]
                wall_effects += (_route(rows, cols, info["active"], info["ends"], info["direction"])[1]
                                 != info["union"])
                gate_effects += (_route(rows, cols, info["active"], info["ends"], info["direction"],
                                       info["rocks"], gated=False)[1] != info["union"])
                multiple += len(info["union"]) > info["length"]-1
                pairs += 1
                copied = pair["input"].copy()
                _need(np.array_equal(solve(copied), pair["output"]), "solver disagreement")
                _need(np.array_equal(copied, pair["input"]), "solver mutated input")
        fingerprints.add(digest.digest())
    _need(len(fingerprints) == 42 and len(shapes) >= 12 and len(designs) >= 24,
          "insufficient shape or stencil variation")
    _need(len(geometries) >= 16 and len(directions) == 4 and len(unions) >= 6,
          "insufficient geometry, gate, or route variation")
    _need(len(counts) >= 10 and multiple >= 42 and wall_effects >= 42 and gate_effects >= 42,
          "insufficient causal witnesses")
    first, repeated = generate_puzzle(0), generate_puzzle(0)
    _need(all(np.array_equal(x[k], y[k]) for s in ("train", "test")
              for x, y in zip(first[s], repeated[s]) for k in ("input", "output")), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            p = generate_puzzle(20261023+a*10+b, a, b)
            _need((len(p["train"]), len(p["test"])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Reject corrupt output evidence for each inherited mechanism and the scaffold.
    pair = first["train"][1]
    _, info = _infer(pair["input"])
    for mode in ("divider", "completion", "route"):
        bad = {k: v.copy() for k, v in pair.items()}
        if mode == "divider":
            bad["output"][0, 0] = info["marker"]
        elif mode == "completion":
            p = sorted((info["active"]-info["supplied"])-set(info["ends"]))[0]
            bad["output"][_slice(info["geometry"], *p)] = info["ordinary"]
        else:
            p = sorted(info["union"])[0]
            bad["output"][_slice(info["geometry"], *p)] = info["ordinary"]
        try:
            _check_pair(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"damaged {mode} was accepted")
    # A partial vertical bar matches several translated U stencils. Equal
    # complete-subset matches are an error, never a nearest/best-overlap tie.
    ordinary = np.array([[0, 0, 0], [0, 2, 0], [2, 0, 0]], dtype=np.int8)
    glyph = np.array([[3, 0, 3], [3, 2, 3], [3, 3, 3]], dtype=np.int8)
    ambiguous = _paint((3, 3, 1, 7, 7), (1, 0, 2, 3), ordinary, glyph,
                       {(2, 1), (3, 1)}, set())
    try:
        solve(ambiguous)
    except ValueError as error:
        _need("translation" in str(error), "ambiguous fixture failed for an unrelated reason")
    else:
        raise AssertionError("ambiguous stencil registration was accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "canvas_shapes": len(shapes), "stencils": len(designs),
            "geometries": len(geometries), "gate_directions": len(directions),
            "route_length_union_sizes": len(unions), "fragment_count_regimes": len(counts),
            "multiple_route_pairs": int(multiple), "wall_effect_pairs": int(wall_effects),
            "gate_effect_pairs": int(gate_effects)}
