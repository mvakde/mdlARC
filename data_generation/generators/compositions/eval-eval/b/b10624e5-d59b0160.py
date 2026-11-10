"""Anchored sieves: reflected/scaled apertures scope a three-color query.

All output decisions come from visible pixels. Generation uses tile painting;
the solver projects by inverse coordinate lookup. No parent module is imported.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json

import numpy as np


SELF_TEST_SEEDS = (
    0, 276481155, 526821302, 266412734, 691301189, 140389580,
    465915329, 43789286, 431735270, 360074052, 638737460, 689639696,
    917315765, 761619479, 635574159, 315985650, 816266562, 346722061,
    8918627, 374132564, 950172970, 479897756, 721665324, 20906968,
    3268205, 970566120, 576281762, 932924395, 24327387, 988952657,
    280984789, 579895175, 376766651, 959990134, 485692921, 529644967,
    731154267, 2652115, 891023067, 262523542, 245140304, 949564290,
)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(seed, num_train, num_test):
    # Explicit None is a convenience; the public defaults remain 4 and 1.
    if num_train is not None:
        num_train = _integer(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be in 2..10")
    if num_test is not None:
        num_test = _integer(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be in 1..4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer)
    assert np.all((array >= 0) & (array <= 9))
    return array


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bounds(h, w, row, col):
    return ((0, row, 0, col), (0, row, col + 1, w),
            (row + 1, h, 0, col), (row + 1, h, col + 1, w))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = [start]
        for r, c in cells:
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    cells.append(point)
        result.append(cells)
    return result


def _gap(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read(grid):
    """Recover the cross, source, fiducials, query, and rectangles from input."""
    grid = _grid(grid)
    h, w = grid.shape
    bg = int(grid[0, 0])
    assert all(int(grid[r, c]) == bg for r, c in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    rows = [r for r in range(1, h - 1) if np.all(grid[r] == grid[r, 0]) and grid[r, 0] != bg]
    cols = [c for c in range(1, w - 1) if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != bg]
    crosses = [(r, c) for r in rows for c in cols if grid[r, 0] == grid[0, c]]
    assert len(crosses) == 1, "one full cross must be visible"
    row, col = crosses[0]
    ink = int(grid[row, col])
    bounds = _bounds(h, w, row, col)
    panels = [grid[r0:r1, c0:c1] for r0, r1, c0, c1 in bounds]
    anchors, sources = [], []
    for p, panel in enumerate(panels):
        coords = np.argwhere(panel == ink)
        assert len(coords), "each panel needs one cross-colored square"
        top, left = map(int, coords.min(0))
        bottom, right = map(int, coords.max(0) + 1)
        side = bottom - top
        assert side == right - left and len(coords) == side * side
        assert top >= 1 and left >= 1 and bottom < panel.shape[0] and right < panel.shape[1]
        anchors.append((top, left, side))
        near = panel[top - 1:bottom + 1, left - 1:right + 1].copy()
        near[1:-1, 1:-1] = bg
        if np.any(near != bg):
            sources.append(p)
    assert len(sources) == 1, "only the source stencil touches its fiducial"
    reference = sources[0]
    source = panels[reference]
    query = frozenset(map(int, np.unique(source))) - {bg, ink}
    assert len(query) == 3, "source has exactly three query identities"
    assert len(_components(source != bg)) == 1
    top, left, unit = anchors[reference]
    tiles = {}
    for y, x in np.argwhere(source != bg):
        tile = ((int(y) - top) // unit, (int(x) - left) // unit)
        value = int(source[y, x])
        assert tile not in tiles or tiles[tile] == value
        tiles[tile] = value
    for (r, c), color in tiles.items():
        y, x = top + r * unit, left + c * unit
        assert 0 <= y and y + unit <= source.shape[0]
        assert 0 <= x and x + unit <= source.shape[1]
        assert np.all(source[y:y + unit, x:x + unit] == color), "stencil tiles align to its square"
    assert tiles[(0, 0)] == ink
    assert all(color != ink for tile, color in tiles.items() if tile != (0, 0))
    records = []
    for p, panel in enumerate(panels):
        if p == reference:
            continue
        at, al, side = anchors[p]
        anchor_box = (at, at + side, al, al + side)
        objects = _components((panel != bg) & (panel != ink))
        boxes = []
        for cells in objects:
            ys, xs = zip(*cells)
            box = (min(ys), max(ys) + 1, min(xs), max(xs) + 1)
            r0, r1, c0, c1 = box
            assert len(cells) == (r1 - r0) * (c1 - c0), "records are filled rectangles"
            assert _gap(box, anchor_box) and all(_gap(box, other) for other in boxes)
            assert 1 <= r0 < r1 < panel.shape[0] and 1 <= c0 < c1 < panel.shape[1]
            boxes.append(box)
            values, counts = np.unique(panel[r0:r1, c0:c1], return_counts=True)
            records.append({"panel": p, "box": box,
                            "counts": dict(zip(map(int, values), map(int, counts)))})
        assert objects, "each destination has at least one record"
    # Filler is only used for diagnostics. The decision itself tests literal
    # query identities and does not depend on recovering a filler role.
    common = set.intersection(*(set(obj["counts"]) for obj in records)) - set(query)
    assert common
    totals = {color: sum(obj["counts"].get(color, 0) for obj in records) for color in common}
    greatest = max(totals.values())
    fillers = [color for color, n in totals.items() if n == greatest]
    assert len(fillers) == 1, "common record filler must be visibly supported"
    return {"bg": bg, "ink": ink, "reference": reference, "bounds": bounds,
            "panels": panels, "anchors": anchors, "query": query,
            "tiles": tiles, "filler": fillers[0], "records": records}


def _sieve(info, panel, *, no_horizontal=False, no_vertical=False, no_scale=False):
    """Inverse lookup: destination pixel -> source lattice tile."""
    top, left, side = info["anchors"][panel]
    if no_scale:
        side = info["anchors"][info["reference"]][2]
    rr = (np.arange(info["panels"][panel].shape[0]) - top) // side
    cc = (np.arange(info["panels"][panel].shape[1]) - left) // side
    if panel // 2 != info["reference"] // 2 and not no_vertical:
        rr = -rr
    if panel % 2 != info["reference"] % 2 and not no_horizontal:
        cc = -cc
    support = set(info["tiles"]) - {(0, 0)}
    return np.array([[(int(r), int(c)) in support for c in cc] for r in rr], dtype=bool)


def _decisions(info, **ablation):
    masks = {p: _sieve(info, p, **{k: v for k, v in ablation.items()
                                 if k in ("no_horizontal", "no_vertical", "no_scale")})
             for p in range(4) if p != info["reference"]}
    if ablation.get("bounding_query"):
        for mask in masks.values():
            coords = np.argwhere(mask)
            if len(coords):
                lo, hi = coords.min(0), coords.max(0) + 1
                mask[lo[0]:hi[0], lo[1]:hi[1]] = True
    records = []
    for record in info["records"]:
        p = record["panel"]
        r0, r1, c0, c1 = record["box"]
        block = info["panels"][p][r0:r1, c0:c1]
        mask = masks[p][r0:r1, c0:c1]
        values = block.ravel() if ablation.get("global_query") else block[mask]
        sampled = frozenset(map(int, values)) - {info["filler"]}
        if ablation.get("pointwise_query"):
            at, al, side = info["anchors"][p]
            matched = set()
            for y, x in np.argwhere(mask):
                r, c = (r0 + int(y) - at) // side, (c0 + int(x) - al) // side
                if p // 2 != info["reference"] // 2:
                    r = -r
                if p % 2 != info["reference"] % 2:
                    c = -c
                if info["tiles"].get((r, c)) == int(block[y, x]):
                    matched.add(int(block[y, x]))
            sampled = frozenset(matched)
        records.append({**record, "sampled": sampled,
                        "positive": info["query"] <= sampled,
                        "covered": int(mask.sum()), "area": int(block.size)})
    return records


def _prediction(grid, info, **ablation):
    result = grid.copy()
    for obj in _decisions(info, **ablation):
        if obj["positive"]:
            y0, _, x0, _ = info["bounds"][obj["panel"]]
            r0, r1, c0, c1 = obj["box"]
            result[y0 + r0:y0 + r1, x0 + c0:x0 + c1] = info["bg"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _prediction(grid, _read(grid))


def _stencil(rng, ink, query):
    """Connected free polyominoes, paths, combs, and hollow loops."""
    style = int(rng.integers(4))
    if style == 0:
        # Random frontier growth; the anchor may end up on an edge or inside.
        points = {(0, 0)}
        count = int(rng.integers(13, 24))
        while len(points) < count:
            frontier = sorted({(r + dr, c + dc) for r, c in points
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
                              - points)
            frontier = [pt for pt in frontier
                        if max(r for r, c in points | {pt}) - min(r for r, c in points | {pt}) < 6
                        and max(c for r, c in points | {pt}) - min(c for r, c in points | {pt}) < 6]
            points.add(frontier[int(rng.integers(len(frontier)))])
    elif style == 1:
        length = int(rng.integers(4, 7))
        points = {(0, c) for c in range(length)}
        for c in range(1, length):
            if c % 2 or rng.random() < .4:
                direction = int(rng.choice((-1, 1)))
                points.update((direction * r, c) for r in range(1, int(rng.integers(2, 5))))
    elif style == 2:
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        points = {(0, 0), (0, 1)}
        points.update((r, c + 2) for r in range(h) for c in range(w)
                      if r in (0, h - 1) or c in (0, w - 1))
    else:
        points = {(0, 0)}
        r = c = 0
        for _ in range(int(rng.integers(16, 30))):
            options = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if -2 <= r + dr <= 3 and -2 <= c + dc <= 3]
            r, c = options[int(rng.integers(len(options)))]
            points.add((r, c))
        if len(points) < 10:
            # A bounded repair, rather than recursively retrying a short walk.
            points.update((r, c) for r in (0, 1) for c in range(5))
    # Independently orient the stencil; panel choice never selects its shape.
    for _ in range(int(rng.integers(4))):
        points = {(c, -r) for r, c in points}
    if rng.random() < .5:
        points = {(r, -c) for r, c in points}
    other = sorted(points - {(0, 0)})
    rng.shuffle(other)
    tiles = {(0, 0): ink}
    for i, point in enumerate(other):
        tiles[point] = query[i % 3] if i < 3 else int(rng.choice(query))
    return tiles


def _project_tiles(tiles, shape, anchor, reference, panel, *, ignore_axis=None, side_override=None, colored=False):
    """Construction uses forward rectangle painting, unlike _sieve."""
    top, left, side = anchor
    if side_override is not None:
        side = side_override
    mask = np.full(shape, -1, dtype=np.int8) if colored else np.zeros(shape, dtype=bool)
    for (r, c), color in tiles.items():
        if (r, c) == (0, 0):
            continue
        if panel // 2 != reference // 2 and ignore_axis != "vertical":
            r = -r
        if panel % 2 != reference % 2 and ignore_axis != "horizontal":
            c = -c
        y, x = top + r * side, left + c * side
        y0, y1, x0, x1 = max(0, y), min(shape[0], y + side), max(0, x), min(shape[1], x + side)
        if y0 < y1 and x0 < x1:
            mask[y0:y1, x0:x1] = color if colored else True
    return mask


@lru_cache(maxsize=32)
def _boxes(h, w):
    # At most 32 small in-memory coordinate tables, never written to disk.
    boxes = [(r0, r1, c0, c1) for r0 in range(1, h - 1) for r1 in range(r0 + 1, h)
             for c0 in range(1, w - 1) for c1 in range(c0 + 1, w)
             if (r1 - r0) * (c1 - c0) >= 8]
    return np.asarray(boxes, dtype=np.int16).reshape(-1, 4)


def _overlaps(mask, boxes):
    prefix = np.pad(mask.astype(np.int16).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r0, r1, c0, c1 = boxes.T
    return prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]


def _gaps(boxes, other):
    r0, r1, c0, c1 = boxes.T
    return (r1 < other[0]) | (other[1] < r0) | (c1 < other[2]) | (other[3] < c0)


def _choose_box(rng, boxes):
    areas = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    # A small/large mixture leaves room for extras without tying size to class.
    if rng.random() < .6:
        boxes = boxes[areas <= np.quantile(areas, .35)]
    return tuple(map(int, boxes[int(rng.integers(len(boxes)))]))


def _layout_panel(rng, tiles, shape, side, reference, panel, kind, axis, source_side):
    boxes = _boxes(*shape)
    positions = [(r, c, side) for r in range(1, shape[0] - side)
                 for c in range(1, shape[1] - side)]
    needed = 6 if kind == "rich" else 3
    for index in rng.permutation(len(positions))[:80]:
        anchor = positions[int(index)]
        r, c, s = anchor
        mask = _project_tiles(tiles, shape, anchor, reference, panel)
        choices = boxes[_gaps(boxes, (r, r + s, c, c + s))]
        if not len(choices):
            continue
        area = (choices[:, 1] - choices[:, 0]) * (choices[:, 3] - choices[:, 2])
        count = _overlaps(mask, choices)
        valid = (count >= needed) & (area - count >= 1)
        if kind == "spatial":
            coords = np.argwhere(mask)
            lo, hi = coords.min(0), coords.max(0) + 1
            envelope = np.zeros(shape, dtype=bool)
            envelope[lo[0]:hi[0], lo[1]:hi[1]] = True
            valid &= _overlaps(envelope & ~mask, choices) > 0
        alternatives = []
        if axis is not None:
            alternatives = [
                _project_tiles(tiles, shape, anchor, reference, panel, ignore_axis=axis),
                _project_tiles(tiles, shape, anchor, reference, panel, side_override=source_side),
            ]
            for wrong in alternatives:
                valid &= _overlaps(mask & ~wrong, choices) > 0
        choices = choices[valid]
        if len(choices):
            box = _choose_box(rng, choices)
            return anchor, mask, box, alternatives
    return None


def _paint_record(rng, panel, box, mask, kind, query, filler, extras, missing=0, alternatives=(), stencil_colors=None):
    r0, r1, c0, c1 = box
    panel[r0:r1, c0:c1] = filler
    inside = [(r, c) for r in range(r0, r1) for c in range(c0, c1) if mask[r, c]]
    outside = [(r, c) for r in range(r0, r1) for c in range(c0, c1) if not mask[r, c]]
    rng.shuffle(inside)
    rng.shuffle(outside)
    available = set(inside + outside)
    marks = {}

    def put(color, points):
        point = next(pt for pt in points if pt in available)
        marks[point] = color
        available.remove(point)
        return point

    if kind in ("exact", "rich"):
        query = list(query)
        witnesses = []
        for wrong in alternatives:
            prior = [pt for pt in witnesses if not wrong[pt]]
            if not prior:
                witnesses.append(next(pt for pt in inside if not wrong[pt] and pt not in witnesses))
        if witnesses and stencil_colors is not None:
            expected_color = int(stencil_colors[witnesses[0]])
            mismatch = next(i for i, color in enumerate(query) if color != expected_color)
            query[0], query[mismatch] = query[mismatch], query[0]
        for color, pt in zip(query, witnesses):
            put(color, [pt])
        for color in query[len(witnesses):]:
            put(color, inside)
        if kind == "rich":
            # Witness identities remain unique, preserving axis/scale contrasts.
            for color in (query[2], extras[0], extras[0]):
                put(color, inside)
    elif kind == "spatial":
        for color in [q for q in query if q != query[missing]] + [extras[0]]:
            put(color, inside)
        coords = np.argwhere(mask)
        lo, hi = coords.min(0), coords.max(0) + 1
        holes = [pt for pt in outside if lo[0] <= pt[0] < hi[0] and lo[1] <= pt[1] < hi[1]]
        put(query[missing], holes or outside)
        for color in (query[(missing + 1) % 3], extras[0]):
            if len(available) > 1:
                put(color, [pt for pt in inside + outside if pt in available])
    elif kind == "global":
        for color in [q for q in query if q != query[missing]] + [extras[0]]:
            put(color, inside)
    elif kind == "random":
        points = inside + outside
        rng.shuffle(points)
        count = int(rng.integers(0, min(len(points) - 1, 16) + 1))
        for pt in points[:count]:
            put(int(rng.choice(query + extras)), [pt])
    elif kind != "empty":
        raise AssertionError("unknown record kind")
    assert available, "retain at least one filler cell"
    for point, color in marks.items():
        panel[point] = color
    return set(query) <= {color for pt, color in marks.items() if mask[pt]}


def _axis(rng, minimum, reference_half):
    first = minimum if reference_half == 0 else 9
    second = minimum if reference_half == 1 else 9
    if first + second > 29:
        return None
    total = int(rng.integers(max(21, first + second + 1), 31))
    extra = total - first - second - 1
    allocation = int(rng.integers(extra + 1))
    return first + allocation, second + extra - allocation


def _make_pair(rng, reference, package):
    colors = list(map(int, rng.permutation(10)))
    bg, ink, filler = colors[:3]
    query = sorted(colors[3:6])
    extras = colors[6:]
    tiles = _stencil(rng, ink, query)
    source_side = 1 if package == 0 else 2 if package == 1 else int(rng.integers(1, 3))
    rs, cs = zip(*tiles)
    heights = _axis(rng, (max(rs) - min(rs) + 1) * source_side + 2, reference // 2)
    widths = _axis(rng, (max(cs) - min(cs) + 1) * source_side + 2, reference % 2)
    if heights is None or widths is None:
        return None
    bounds = _bounds(sum(heights) + 1, sum(widths) + 1, heights[0], widths[0])
    inp = np.full((sum(heights) + 1, sum(widths) + 1), bg, dtype=np.int8)
    inp[heights[0]] = ink
    inp[:, widths[0]] = ink
    truth = inp.copy()
    y0, y1, x0, x1 = bounds[reference]
    panel = inp[y0:y1, x0:x1]
    at = int(rng.integers(1 - min(rs) * source_side, panel.shape[0] - (max(rs) + 1) * source_side))
    al = int(rng.integers(1 - min(cs) * source_side, panel.shape[1] - (max(cs) + 1) * source_side))
    for (r, c), color in tiles.items():
        y, x = at + r * source_side, al + c * source_side
        panel[y:y + source_side, x:x + source_side] = color
    truth[y0:y1, x0:x1] = panel
    if package == 0:
        plan = {reference ^ 1: ("exact", "horizontal", 0, 2),
                reference ^ 2: ("spatial", None, 0, int(rng.integers(1, 3))),
                reference ^ 3: ("global", None, 1, int(rng.integers(1, 3)))}
    elif package == 1:
        plan = {reference ^ 2: ("rich", "vertical", 0, 1),
                reference ^ 1: ("spatial", None, 2, int(rng.integers(1, 3))),
                reference ^ 3: ("empty", None, 0, int(rng.integers(1, 3)))}
    else:
        positive_panel = int(rng.choice([p for p in range(4) if p != reference]))
        plan = {}
        for p in range(4):
            if p == reference:
                continue
            kind = "exact" if p == positive_panel else str(rng.choice(
                ("exact", "rich", "spatial", "global", "empty", "random"),
                p=(.22, .08, .25, .15, .10, .20)))
            side = 3 if package == "test" and p == positive_panel else int(rng.integers(1, 4 if package == "test" else 3))
            plan[p] = kind, None, int(rng.integers(3)), side
    for p, (kind, axis, missing, side) in plan.items():
        r0, r1, c0, c1 = bounds[p]
        panel = inp[r0:r1, c0:c1]
        built = _layout_panel(rng, tiles, panel.shape, side, reference, p, kind, axis, source_side)
        if built is None:
            return None
        anchor, mask, primary, alternatives = built
        stencil_colors = _project_tiles(tiles, panel.shape, anchor, reference, p, colored=True)
        at, al, s = anchor
        panel[at:at + s, al:al + s] = ink
        target = truth[r0:r1, c0:c1]
        target[at:at + s, al:al + s] = ink
        objects = [primary]
        positive = _paint_record(rng, panel, primary, mask, kind, query, filler, extras, missing, alternatives, stencil_colors)
        pr0, pr1, pc0, pc1 = primary
        target[pr0:pr1, pc0:pc1] = bg if positive else panel[pr0:pr1, pc0:pc1]
        count = int(rng.integers(0, 3 if package != "test" else 4))
        for _ in range(count):
            choices = _boxes(*panel.shape)
            valid = _gaps(choices, (at, at + s, al, al + s))
            for box in objects:
                valid &= _gaps(choices, box)
            choices = choices[valid]
            if not len(choices):
                break
            box = _choose_box(rng, choices)
            objects.append(box)
            br0, br1, bc0, bc1 = box
            covered = int(mask[br0:br1, bc0:bc1].sum())
            area = (br1 - br0) * (bc1 - bc0)
            eligible = ["random", "empty"]
            if covered >= 3:
                eligible += ["exact", "exact", "global"]
                if covered < area:
                    eligible += ["spatial"]
            if covered >= 6:
                eligible += ["rich"]
            extra_kind = str(rng.choice(eligible))
            positive = _paint_record(rng, panel, box, mask, extra_kind, query, filler, extras,
                                     missing=int(rng.integers(3)))
            target[br0:br1, bc0:bc1] = bg if positive else panel[br0:br1, bc0:bc1]
    return {"input": inp, "output": truth}


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _read(inp)
    assert np.array_equal(_prediction(inp, info), out), "incorrect sieve-scoped whole-record deletion"
    records = _decisions(info)
    assert any(obj["positive"] for obj in records), "generated scenes include a visible deletion"
    info["decisions"] = records
    return info


def _teaching_coverage(pairs, infos):
    combined = [(info, obj) for info in infos for obj in info["decisions"]]
    assert any(obj["positive"] and sum(n for c, n in obj["counts"].items() if c != info["filler"]) == 3
               for info, obj in combined), "teach exact three-mark positives"
    assert any(obj["positive"] and obj["sampled"] - info["query"] for info, obj in combined)
    assert any(not (set(obj["counts"]) - {info["filler"]}) for info, obj in combined)
    assert any(not obj["positive"] and info["query"] <= set(obj["counts"]) for info, obj in combined), "global membership is insufficient"
    signatures = {status: set() for status in (False, True)}
    for info, obj in combined:
        labels = set(obj["counts"]) - {info["filler"]}
        marks = sum(obj["counts"][c] for c in labels)
        signatures[obj["positive"]].add((marks, len(labels)))
    assert signatures[False] & signatures[True], "opposite decisions must share mark and color counts"
    for k in range(3):
        assert any(not obj["positive"] and obj["sampled"] & info["query"] == info["query"] - {sorted(info["query"])[k]}
                   for info, obj in combined), "every query identity must matter"
    assert len({info["reference"] for info in infos}) >= 2
    for ablation in ("no_horizontal", "no_vertical", "no_scale", "global_query", "bounding_query", "pointwise_query"):
        assert any(not np.array_equal(_prediction(np.asarray(pair["input"]), info, **{ablation: True}), pair["output"])
                   for pair, info in zip(pairs, infos)), f"training does not expose {ablation}"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    references = list(map(int, rng.permutation(4)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            package = index if split == "train" and index < 2 else "test" if split == "test" else "free"
            reference = references[index] if split == "train" and index < 2 else int(rng.integers(4))
            for _ in range(160):
                pair = _make_pair(rng, reference, package)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                try:
                    _analyze_pair(pair)
                except AssertionError:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            infos[split].append(_analyze_pair(pair))
    _teaching_coverage(puzzle["train"], infos["train"])
    assert all(side <= 2 for info in infos["train"] for _, _, side in info["anchors"])
    assert all(any(side == 3 for p, (_, _, side) in enumerate(info["anchors"]) if p != info["reference"])
               for info in infos["test"]), "test recombines the rule at a larger anchor size"


def self_test():
    """Stream 42 seeds; retain hashes/features, never a corpus of grids."""
    puzzles, shapes, footprints, roles, record_counts, scales = set(), set(), set(), set(), set(), set()
    pairs = records = deletions = spatial_negatives = clipped_panels = 0
    deletion_counts = Counter()
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _read(pair["input"])
                objects = _decisions(info)
                records += len(objects)
                deletions += sum(obj["positive"] for obj in objects)
                deletion_counts[sum(obj["positive"] for obj in objects)] += 1
                spatial_negatives += sum(not obj["positive"] and info["query"] <= set(obj["counts"]) for obj in objects)
                shapes.add(pair["input"].shape)
                footprints.add(tuple(sorted(info["tiles"])))
                roles.add((info["bg"], info["ink"], info["filler"]))
                record_counts.add(len(objects))
                ref_side = info["anchors"][info["reference"]][2]
                for p, (_, _, side) in enumerate(info["anchors"]):
                    if p != info["reference"]:
                        scales.add((side, ref_side))
                        clipped_panels += int(_sieve(info, p).sum() < (len(info["tiles"]) - 1) * side * side)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                for name in ("input", "output"):
                    digest.update(bytes(pair[name].shape))
                    digest.update(pair[name].tobytes())
                # Source color positions do not encode pointwise matching.
                recolored = pair["input"].copy()
                y0, y1, x0, x1 = info["bounds"][info["reference"]]
                source = recolored[y0:y1, x0:x1]
                old = source.copy()
                q = sorted(info["query"])
                for i, color in enumerate(q):
                    source[old == color] = q[(i + 1) % 3]
                expected = pair["output"].copy()
                expected[y0:y1, x0:x1] = source
                assert np.array_equal(solve(recolored), expected)
        puzzles.add(digest.hexdigest())
        if seed == 0:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], repeat[s]) for k in a)
            # Validation rejects incomplete erasure and changes to kept pixels.
            for mode in ("partial", "survivor"):
                bad = {s: [{k: a.copy() for k, a in p.items()} for p in puzzle[s]] for s in puzzle}
                pair = bad["train"][0]
                info = _read(pair["input"])
                obj = next(o for o in _decisions(info) if o["positive"] == (mode == "partial"))
                y, _, x, _ = info["bounds"][obj["panel"]]
                r, _, c, _ = obj["box"]
                pair["output"][y + r, x + c] = pair["input"][y + r, x + c] if mode == "partial" else info["bg"]
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("corrupt output accepted")
    assert len(puzzles) == 42 and len(footprints) >= 50 and len(shapes) >= 30
    assert len(roles) >= 100 and len(record_counts) >= 4
    assert len(deletion_counts) >= 3
    assert {(1, 2), (2, 1), (3, 1), (3, 2)} <= scales
    assert spatial_negatives >= 84 and clipped_panels >= 42
    for a, b in ((2, 1), (10, 4), (None, 4), (10, None), (None, None)):
        p = generate_puzzle(17, a, b)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "pairs": pairs,
            "records": records, "deletions": deletions, "spatial_negatives": spatial_negatives,
            "clipped_destination_panels": clipped_panels, "shapes": len(shapes),
            "stencil_footprints": len(footprints), "role_assignments": len(roles),
            "record_counts": sorted(record_counts), "deletion_counts": dict(sorted(deletion_counts.items())),
            "scale_side_pairs": sorted(scales)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
