"""Counterweight scaffolds: source component counts change cargo landing edges.

Only standard library and NumPy are used. No parent code is imported. All
output-affecting information is recovered from the current input by solve().
"""
from collections import Counter
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= g) & (g <= 9)), "colors must be 0..9")
    return g.astype(int, copy=False)


def _components(g, points, diagonal=False):
    unseen = set(points)
    result = []
    neighbors = EIGHT if diagonal else FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        color = int(g[start])
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in neighbors:
                q = (p[0] + dr, p[1] + dc)
                if q in unseen and int(g[q]) == color:
                    unseen.remove(q)
                    stack.append(q)
        result.append((color, tuple(sorted(cells))))
    return result


def _normalized(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _draw_output(g, bg, bands, components, *, ignore_count=False, dots=False,
                 collective=False):
    """Pixel transform; optional ablations are used only for evidence checks."""
    counts = Counter(color for color, cells in components)
    out = np.full_like(g, bg)
    for b in bands:
        t, z, l, r, route, rail = b
        out[t:z+1, l:r+1] = g[t:z+1, l:r+1]
        n = 0 if ignore_count else counts[rail]
        _require(2*n <= z-t+1, "scaffold exceeds band depth")
        for i in range(n):
            out[z-2*i, l+1:r] = rail
        cargo = [cells for color, cells in components if color == route]
        baseline = z-2*n
        occupied = set()
        joint_bottom = max((y for cells in cargo for y, x in cells), default=0)
        for cells in cargo:
            bottom = joint_bottom if collective else max(y for y, x in cells)
            target = {(y+baseline-bottom, x) for y, x in cells}
            if dots:
                target = {(baseline, min(x for y, x in cells))}
            _require(all(t <= y <= baseline and l < x < r for y, x in target),
                     "cargo crosses a rail, scaffold, or band boundary")
            _require(not target & occupied, "actual cargo cell collision")
            occupied.update(target)
            for p in target:
                out[p] = rail
    return out


def _canonical_candidates(g, bg):
    """A complete bank suffix, separated from a nonempty source field."""
    h, w = g.shape
    rows = []
    for row in g:
        xs = np.flatnonzero(row != bg)
        if not len(xs):
            rows.append(None)
        else:
            l, r = int(xs[0]), int(xs[-1])
            if (r-l >= 6 and 0 < l < r < w-1 and
                    row[l] == row[r] and row[l] != row[l+1] and
                    row[l+1] != bg and np.all(row[l+1:r] == row[l+1])):
                rows.append((l, r, int(row[l+1]), int(row[l])))
            else:
                rows.append(False)
    starts = [first for first in range(2, h-1)
              if rows[first] and rows[first-1] is None and
              not any(info is False for info in rows[first:])]
    # Take the entire recognizable suffix. Allowing later suffixes would
    # misclassify earlier, unused bands as unmatched source objects.
    for first in starts[:1]:
        bands, y = [], first
        while y < h:
            if rows[y] is None:
                y += 1
                continue
            t, info = y, rows[y]
            while y+1 < h and rows[y+1] == info:
                y += 1
            bands.append((t, y, *info))
            y += 1
        if len({b[4] for b in bands}) != len(bands):
            continue
        if any(b[0] <= a[1]+1 for a, b in zip(bands, bands[1:])):
            continue
        points = {tuple(map(int, p)) for p in np.argwhere(g[:first] != bg)}
        if not points:
            continue
        comps = _components(g, points)
        # The family avoids undemonstrated diagonal-connectivity decisions.
        if len(_components(g, points, True)) != len(comps):
            continue
        try:
            out = _draw_output(g, bg, bands, comps)
        except AssertionError:
            continue
        yield dict(canonical=g, background=bg, bands=bands, components=comps,
                   first=first, expected=out)


def _analyze(value):
    g = _grid(value)
    corners = {int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])}
    _require(len(corners) == 1, "four background corners required")
    bg = corners.pop()
    candidates = []
    for turn in range(4):
        for detail in _canonical_candidates(np.rot90(g, turn), bg):
            detail["rotation"] = turn
            candidates.append(detail)
    _require(len(candidates) == 1, "expected one source/bank interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover background, direction, bands, masks and counts from pixels alone."""
    d = _analyze(input_grid)
    return np.rot90(d["expected"], -d["rotation"]).copy()


def _shape(rng, max_h=4, max_w=6, area=None):
    """A connected shape grammar with no category-count argument."""
    h, w = int(rng.integers(1, max_h+1)), int(rng.integers(1, max_w+1))
    kind = int(rng.integers(6))
    if area is not None:
        h, w = max_h, max_w
        n = min(area, h*w)
        kind = 5
    elif kind == 0:
        return tuple((y, x) for y in range(h) for x in range(w))
    elif kind == 1:
        return tuple((0, x) for x in range(w))
    elif kind == 2:
        return tuple(sorted({(y, 0) for y in range(h)} | {(h-1, x) for x in range(w)}))
    elif kind == 3 and h >= 3 and w >= 3:
        return tuple((y, x) for y in range(h) for x in range(w)
                     if y in (0, h-1) or x in (0, w-1))
    elif kind == 4:
        spine = int(rng.integers(w))
        cells = {(y, spine) for y in range(h)}
        for y in range(h):
            end = int(rng.integers(w))
            if rng.random() < .65:
                cells.update((y, x) for x in range(min(end, spine), max(end, spine)+1))
        return _normalized(cells)
    else:
        n = int(rng.integers(1, h*w+1))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < n:
        frontier = sorted({(y+dy, x+dx) for y, x in cells for dy, dx in FOUR
                           if 0 <= y+dy < h and 0 <= x+dx < w} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normalized(cells)


def _band_grid(bg, width, stage, band_specs, gaps, margins):
    """band_specs: depth, left, right, route, rail."""
    height = stage+sum(gaps)+sum(b[0] for b in band_specs)+margins
    if height > 30:
        return None
    g = np.full((height, width), bg, dtype=int)
    bands, cursor = [], stage
    for (depth, l, r, route, rail), gap in zip(band_specs, gaps):
        cursor += gap
        b = (cursor, cursor+depth-1, l, r, route, rail)
        bands.append(b)
        g[cursor:cursor+depth, l:r+1] = route
        g[cursor:cursor+depth, (l, r)] = rail
        cursor += depth
    return g, bands


def _finish(g, bg, bands, components, turn):
    # Labels use construction's visible components and known band geometry.
    # solve() instead rediscovers both from pixels; no construction metadata
    # leaves this function or is used by the solver.
    output = _draw_output(g, bg, bands, components)
    return {"input": np.rot90(g, turn).copy(), "output": np.rot90(output, turn).copy()}


def _teaching_pair(rng, palette, area, split, turn, common):
    """Two related but reordered scenes teach counting, routing and role reuse."""
    a, b, c, d, rail_c, bg0, bg1 = palette[:7]
    bg = bg1 if split else bg0
    stage, width, ring_top, ring_left, ring_h, extra_mask, extra_top, extra_left, cmask, ctop, cleft, depths, gaps, margin = common
    route_c, distractor = (d, c) if split else (c, d)
    specifications = [(depths[0], 1, width-2, a, b),
                      (depths[1], 1, width-2, b, a),
                      (depths[2], 1, width-2, route_c, rail_c)]
    order = list(map(int, rng.permutation(3)))
    # Ensure band order changes, even for the minimum two demonstrations.
    if split:
        order = common.order[1:]+common.order[:1]
    else:
        common.order = order
    built = _band_grid(bg, width, stage, [specifications[i] for i in order],
                       gaps, margin)
    _require(built is not None, "teaching layout budget")
    g, bands = built
    ring_w = area+3
    ring = tuple((ring_top+y, ring_left+x) for y in range(ring_h) for x in range(ring_w)
                 if y in (0, ring_h-1) or x in (0, ring_w-1))
    components = [(a, ring), (a, tuple((extra_top+y, extra_left+x) for y, x in extra_mask)),
                  (route_c, tuple((ctop+y, cleft+x) for y, x in cmask))]
    inner_y = ring_top+int(rng.integers(1, ring_h-1))
    if split:
        k = int(rng.integers(1, area))
        components += [(b, tuple((inner_y, ring_left+1+x) for x in range(k))),
                       (b, tuple((inner_y, ring_left+2+x) for x in range(k, area)))]
    else:
        components.append((b, tuple((inner_y, ring_left+1+x) for x in range(area))))
    # A color is discarded in one lesson and routed in the other. It is not
    # a rail in either scene, so it never influences either scaffold.
    components.append((distractor, ((stage-2, width-3),)))
    if split:
        # This positive rail-only vote rules out counting only colors that also
        # have a cargo destination. The first lesson demonstrates zero here.
        components.append((rail_c, ((stage-2, width-5),)))
    for color, cells in components:
        for p in cells:
            _require(g[p] == bg, "teaching source collision")
            g[p] = color
    return _finish(g, bg, bands, components, turn)


class _TeachingLayout(list):
    """Local builder convenience; never retained by solve or the puzzle."""
    pass


def _teaching_layout(rng, area):
    ring_h = int(rng.integers(3, 5))
    ring_top = int(rng.integers(1, 3))
    ring_left = int(rng.integers(2, 4))
    extra = _shape(rng, 2, 4)
    eh, ew = 1+max(y for y, x in extra), 1+max(x for y, x in extra)
    extra_left = ring_left+area+3+int(rng.integers(1, 3))
    extra_top = 1
    if extra_top+eh-1 == ring_top+ring_h-1:
        extra_top = 0
    cmask = _shape(rng, 2, 3)
    ch, cw = 1+max(y for y, x in cmask), 1+max(x for y, x in cmask)
    ctop, cleft = ring_top+ring_h+1, ring_left
    stage = max(ctop+ch+1, 7)
    width = max(18, extra_left+ew+4)+int(rng.integers(0, 3))
    depths = [ring_h+4, 5+int(rng.integers(0, 2)), 2+ch+int(rng.integers(0, 2))]
    gaps = [1, 1, 1]
    margin = 1
    while stage+sum(depths)+sum(gaps)+margin > 30:
        if depths[1] > 5:
            depths[1] -= 1
        elif depths[2] > 2+ch:
            depths[2] -= 1
        elif ring_top > 1:
            ring_top -= 1
            ctop -= 1
            stage -= 1
        else:
            raise AssertionError("minimum teaching geometry exceeds 30")
    budget = 30-stage-sum(depths)-margin
    if budget >= 4 and rng.random() < .5:
        gaps[int(rng.integers(3))] += 1
    return _TeachingLayout([stage, width, ring_top, ring_left, ring_h, extra, extra_top,
                            extra_left, cmask, ctop, cleft, depths, gaps, margin])


def _place_sources(rng, g, bg, bands, stage, specs):
    counts = Counter(color for color, mask in specs)
    lookup = {b[4]: b for b in bands}
    components, occupied_target = [], set()
    # Largest first, with shuffled equal-size objects; origins are free pixels.
    ordered = sorted(specs, key=lambda s: (-len(s[1]), float(rng.random())))
    for color, mask in ordered:
        mh, mw = 1+max(y for y, x in mask), 1+max(x for y, x in mask)
        b = lookup.get(color)
        lo, hi = (b[2]+1, b[3]-mw) if b else (1, g.shape[1]-mw-1)
        origins = [(y, x) for y in range(1, stage-mh) for x in range(lo, hi+1)]
        for index in rng.permutation(len(origins)):
            y0, x0 = origins[int(index)]
            cells = tuple((y0+y, x0+x) for y, x in mask)
            if any(g[p] != bg for p in cells):
                continue
            if any(0 <= y+dy < stage and 0 <= x+dx < g.shape[1] and
                   g[y+dy, x+dx] == color for y, x in cells for dy, dx in EIGHT):
                continue
            targets = set()
            if b:
                baseline = b[1]-2*counts[b[5]]
                targets = {(baseline-(mh-1)+y, x0+x) for y, x in mask}
                if targets & occupied_target:
                    continue
            for p in cells:
                g[p] = color
            occupied_target.update(targets)
            components.append((color, cells))
            break
        else:
            return None
    return components


def _ordinary_pair(rng, turn, level, max_count, *, extra_count=False, unused=False):
    for attempt in range(160):
        nband = int(rng.choice((1, 2, 2, 3, 4)))
        if extra_count or unused:
            nband = max(2, nband)
        palette = list(map(int, rng.permutation(10)))
        bg = palette[0]
        routes = palette[1:nband+1]
        extras = palette[nband+1:]
        rails = []
        for route in routes:
            pool = [c for c in routes if c != route]
            rails.append(int(rng.choice(pool if pool and rng.random() < .65 else extras[:2])))
        route_counts = [int(rng.integers(1, max_count+1)) for _ in routes]
        if unused and nband > 1:
            route_counts[-1] = 0
        if extra_count:
            route_counts[0] = 3
            rails[1] = routes[0]
        specs = []
        for route, count in zip(routes, route_counts):
            for _ in range(count):
                specs.append((route, _shape(rng, 3+level, 4+level)))
        # Rail-only objects are counterweights, despite having no cargo route.
        for color in sorted(set(rails)-set(routes)):
            for _ in range(int(rng.integers(0, max_count+1))):
                specs.append((color, _shape(rng, 3+level, 4+level)))
        unmatched = [c for c in extras if c not in rails]
        for _ in range(int(rng.integers(0, 3))):
            specs.append((int(rng.choice(unmatched)), _shape(rng, 3+level, 4+level)))
        counts = Counter(c for c, m in specs)
        heights = {route: max([1]+[1+max(y for y, x in m) for c, m in specs if c == route])
                   for route in routes}
        depths = [max(1, 2*counts[s]+(heights[r] if counts[r] else 0))
                  +int(rng.integers(0, 2)) for r, s in zip(routes, rails)]
        stage = max(5, max(1+max(y for y, x in m) for c, m in specs)+3,
                    int(np.ceil(len(specs)/2))+2)+int(rng.integers(0, 3))
        gaps = [int(rng.integers(1, 3)) for _ in routes]
        margin = int(rng.integers(1, 3))
        if stage+sum(depths)+sum(gaps)+margin > 30:
            continue
        widths = {r: sum(1+max(x for y, x in m) for c, m in specs if c == r)
                  +max(0, counts[r]-1) for r in routes}
        min_width = max(9, max(widths.values())+4,
                        max(1+max(x for y, x in m) for c, m in specs)+4)
        if min_width > 30:
            continue
        width = int(rng.integers(min_width, 31))
        bandspecs = []
        for route, rail, depth in zip(routes, rails, depths):
            inset = max(0, min(3, (width-widths[route]-4)//2))
            left = 1+int(rng.integers(inset+1))
            right = width-2-int(rng.integers(inset+1))
            if right-left < 6:
                left, right = 1, width-2
            bandspecs.append((depth, left, right, route, rail))
        rng.shuffle(bandspecs)
        built = _band_grid(bg, width, stage, bandspecs, gaps, margin)
        if built is None:
            continue
        g, bands = built
        comps = _place_sources(rng, g, bg, bands, stage, specs)
        if comps is None:
            continue
        pair = _finish(g, bg, bands, comps, turn)
        try:
            recovered = _analyze(pair["input"])
        except AssertionError:
            continue
        if not np.array_equal(pair["output"], np.rot90(recovered["expected"], -recovered["rotation"])):
            continue
        return pair
    raise RuntimeError("bounded source packing exhausted")


def _key(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    orientation = int(rng.choice(3, p=(.3, .35, .35)))
    base = int(rng.integers(4))
    shift = int(rng.integers(3))
    palette = list(map(int, rng.permutation(10)))
    area = int(rng.integers(4, 8))
    layout = _teaching_layout(rng, area)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if orientation == 0:
                turn = base
            elif split == "train":
                turn = (base+2*(i % 2)) % 4 if i < 2 or orientation == 2 else int(rng.integers(4))
            else:
                turn = (base+1+2*(i % 2)) % 4 if orientation == 2 else int(rng.integers(4))
            for retry in range(24):
                if split == "train" and i < 2:
                    pair = _teaching_pair(rng, palette, area, bool(i), turn, layout)
                else:
                    level = int(rng.integers(0, 2)) if split == "train" else (2 if shift == 1 else 0 if shift == 2 else 1)
                    pair = _ordinary_pair(rng, turn, level, 2 if split == "train" else 3,
                                          extra_count=split == "test" and shift == 0,
                                          unused=nt >= 3 and (split == "train" and i == 2 or
                                              split == "test" and shift != 0 and rng.random() < .6))
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct source construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(details):
    records = []
    for d in details:
        counted_colors = {b[5] for b in d["bands"]}
        for color in counted_colors:
            pieces = [cells for c, cells in d["components"] if c == color]
            records.append((color, len(pieces), sum(map(len, pieces))))
    collision = any(c == cc and a == aa and n != nn
                    for c, n, a in records for cc, nn, aa in records)
    area_independent = any(n == nn > 0 and a != aa
                           for c, n, a in records for cc, nn, aa in records)
    return collision, area_independent


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape, "dimensions changed")
            key = _key(g)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _analyze(g)
            expected = np.rot90(d["expected"], -d["rotation"])
            _require(np.array_equal(out, expected), "wrong count, landing, recoloring, or erasure")
            _require(not np.array_equal(g, out), "identity pair")
            # Check support-height and exact-mask invariants explicitly.
            counts = Counter(c for c, cells in d["components"])
            canonical_out = np.rot90(out, d["rotation"])
            for t, z, l, r, route, rail in d["bands"]:
                n = counts[rail]
                _require(np.array_equal(canonical_out[t:z+1, l], d["canonical"][t:z+1, l])
                         and np.array_equal(canonical_out[t:z+1, r], d["canonical"][t:z+1, r]),
                         "rails changed")
                for j in range(n):
                    _require(np.all(canonical_out[z-2*j, l+1:r] == rail), "missing rung")
                    _require(np.all(canonical_out[z-2*j-1, l+1:r] == route), "missing spacer row")
            _require(np.all(canonical_out[:d["first"]] == d["background"]), "source not erased")
            details[split].append(d)
    train = details["train"]
    _require(all(_evidence(train)), "training permits area-to-count shortcut")
    _require(any(not np.array_equal(d["expected"], _draw_output(d["canonical"], d["background"],
                         d["bands"], d["components"], ignore_count=True)) for d in train),
             "training does not require B")
    _require(any(not np.array_equal(d["expected"], _draw_output(d["canonical"], d["background"],
                         d["bands"], d["components"], dots=True)) for d in train),
             "training does not require A")
    _require(any(any(c == cc and max(y for y, x in cells) != max(y for y, x in other)
                     for i, (c, cells) in enumerate(d["components"])
                     for cc, other in d["components"][i+1:] if c in {b[4] for b in d["bands"]})
                 for d in train), "no independent alignment lesson")
    _require(any(any(b[5] in {a[4] for a in d["bands"]} for b in d["bands"])
                 for d in train), "no cross-role counterweight lesson")
    _require(any(any(c in {b[5] for b in d["bands"]}-{b[4] for b in d["bands"]}
                     for c, cells in d["components"]) for d in train),
             "no positive rail-only counterweight lesson")
    _require({0, 1, 2} <= {Counter(c for c, cells in d["components"])[b[5]]
                           for d in train for b in d["bands"]}, "missing count progression")
    if any(any(not any(c == b[4] for c, cells in d["components"]) for b in d["bands"])
           for d in details["test"]):
        _require(any(any(not any(c == b[4] for c, cells in d["components"]) for b in d["bands"])
                     for d in train), "unused test destination lacks a training witness")
    train_rotations = {d["rotation"] for d in train}
    if any(d["rotation"] not in train_rotations for d in details["test"]):
        _require(len(train_rotations) >= 2, "held-out direction lacks training evidence")
    return None


def puzzle_metrics(puzzle):
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            d = _analyze(pair["input"])
            counts = Counter(c for c, cells in d["components"])
            routes = {b[4] for b in d["bands"]}
            rails = {b[5] for b in d["bands"]}
            records.append(dict(split=split, shape=tuple(np.asarray(pair["input"]).shape),
                                background=d["background"], rotation=d["rotation"],
                                bands=len(d["bands"]), depths=tuple(b[1]-b[0]+1 for b in d["bands"]),
                                counts=tuple(counts[b[5]] for b in d["bands"]),
                                masks=tuple(_normalized(cells) for c, cells in d["components"]),
                                areas=tuple(len(cells) for c, cells in d["components"]),
                                insets=tuple((b[2], d["canonical"].shape[1]-1-b[3]) for b in d["bands"]),
                                cross_roles=bool(routes & rails), shared_rails=len(rails) < len(d["bands"]),
                                rail_only=any(c in rails-routes for c, cells in d["components"]),
                                distractors=sum(c not in routes | rails for c, cells in d["components"]),
                                unused=any(not counts[c] for c in routes),
                                origins=tuple(min(cells) for c, cells in d["components"])))
    return records


def self_test():
    """Bounded 42-seed diversity, count API, equivariance and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [m for p in puzzles for m in puzzle_metrics(p)]
    signatures = {tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]) for p in puzzles}
    _require(len(signatures) == 42, "cross-seed repetition")
    _require({m["background"] for m in records} == set(range(10)), "palette coverage")
    _require({m["rotation"] for m in records} == set(range(4)), "direction coverage")
    _require({m["bands"] for m in records} >= {1, 2, 3, 4}, "bank-count variation")
    _require(len({m["shape"] for m in records}) >= 30, "canvas variation")
    _require(len({mask for m in records for mask in m["masks"]}) >= 70, "shape variation")
    _require({n for m in records for n in m["counts"]} >= {0, 1, 2, 3}, "count variation")
    for feature in ("rail_only", "shared_rails", "unused"):
        _require(any(m[feature] for m in records), "missing feature: "+feature)
    _require(any(m["distractors"] == 0 for m in records), "no clean fields")
    _require(len({tuple(m["depths"]) for m in records}) >= 30, "depth variation")
    # All legal count requests, including the smallest evidence budget.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(1000+nt*7+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in
                 zip(p[s], generate_puzzle(0)[s]) for k in ("input", "output")), "determinism")
    for pair in p["train"]+p["test"]:
        original = pair["input"].copy()
        expected = pair["output"]
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(original, turn)), np.rot90(expected, turn)),
                     "rotation covariance")
        permutation = np.array([7, 3, 9, 0, 5, 1, 8, 4, 2, 6])
        _require(np.array_equal(solve(permutation[original]), permutation[expected]), "color-role covariance")
        _require(np.array_equal(original, pair["input"]), "solve mutated input")
    # Explicit expected pixels for a rail-only weight fixture. Count is two,
    # despite three weight pixels: bottom=15, rungs=15/13, cargo baseline=11.
    fixture, bands = _band_grid(0, 18, 7, [(8, 1, 16, 1, 2)], [1], 1)
    fixture[1, 5:7] = 1
    fixture[2, 5] = 1
    fixture[2, 10:12] = 2
    fixture[4, 12] = 2
    expected = fixture.copy()
    expected[:8] = 0
    expected[15, 2:16] = 2
    expected[13, 2:16] = 2
    expected[10, 5:7] = 2
    expected[11, 5] = 2
    _require(np.array_equal(solve(fixture), expected), "explicit scaffold/mask fixture")
    removed = fixture.copy()
    removed[4, 12] = 0
    less = expected.copy()
    less[8:16, 2:16] = 1
    less[15, 2:16] = 2
    less[12, 5:7] = 2
    less[13, 5] = 2
    _require(np.array_equal(solve(removed), less), "one weight must shift cargo by two")
    enlarged = fixture.copy()
    enlarged[1, 10] = 2
    _require(np.array_equal(solve(enlarged), expected), "rail-only weight area affected count")
    noisy = fixture.copy()
    noisy[5, 15] = 3
    _require(np.array_equal(solve(noisy), expected), "unmatched color affected answer")
    # Same-color masks can interlock at their destination. Bounding boxes
    # overlap, but the actual cells do not. Their source depths differ.
    interlock, _ = _band_grid(0, 18, 7, [(4, 1, 16, 1, 2)], [1], 1)
    interlock[1, 4:6] = 1
    interlock[2, 4] = interlock[5, 5] = 1
    want = interlock.copy()
    want[:8] = 0
    want[10:12, 4:6] = 2
    _require(np.array_equal(solve(interlock), want), "valid interlocking masks rejected")
    collision = interlock.copy()
    collision[5, 5], collision[5, 4] = 0, 1
    try:
        solve(collision)
    except AssertionError:
        pass
    else:
        raise AssertionError("actual overlapping cargo accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(records), puzzles=len(signatures),
                canvas_shapes=len({m["shape"] for m in records}),
                component_masks=len({mask for m in records for mask in m["masks"]}),
                band_counts=sorted({m["bands"] for m in records}),
                counterweight_counts=sorted({n for m in records for n in m["counts"]}),
                semantic_verification="generator-owned checks only")


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
