"""Border-indexed courtyards receiving rigid, axis-cut corner stencils.

Only standard-library modules and NumPy are used. No parent code is imported.
The solver reads visible border roles, axes, payload masks and coordinate intervals.
"""
from __future__ import annotations

import copy
import numpy as np


SIDES = ("top", "right", "bottom", "left")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _components(cells):
    left = set(cells)
    groups = []
    while left:
        first = min(left)
        left.remove(first)
        group, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in left:
                    left.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _partition(cells, ar, ac, sides):
    groups = {}
    for r, c in cells:
        if r == ar or c == ac:
            continue
        vertical = ("top" if r < ar else "bottom") if ar is not None else next(s for s in sides if s in ("top", "bottom"))
        horizontal = ("left" if c < ac else "right") if ac is not None else next(s for s in sides if s in ("left", "right"))
        groups.setdefault((vertical, horizontal), set()).add((r, c))
    return groups


def _retained(corner, sides):
    return bool(set(corner) & set(sides))


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _destination(cells, corner, room):
    """One translation for all cells, including disconnected fragments."""
    r0, r1, c0, c1 = _box(cells)
    a, b, x, y = room
    dr = a-r0 if corner[0] == "top" else b-r1
    dc = x-c0 if corner[1] == "left" else y-c1
    return {(r+dr, c+dc) for r, c in cells}


def _guide(side, point, cells, nearest=False):
    r, c = point
    vertical = side in ("top", "bottom")
    axis = 0 if vertical else 1
    aligned = [p[axis] for p in cells if p[1-axis] == point[1-axis]]
    assert aligned, "each guide must intersect payload"
    increasing = side in ("top", "left")
    endpoint = (min(aligned) if increasing else max(aligned)) if nearest else (max(aligned) if increasing else min(aligned))
    start = point[axis]
    return {(v, c) if vertical else (r, v) for v in range(min(start, endpoint), max(start, endpoint)+1)}


def _parse(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    assert min(h, w) >= 7
    frame = int(g[0, 0])
    assert all(int(g[r, c]) == frame for r, c in ((0, w-1), (h-1, 0), (h-1, w-1)))
    interior = g[1:-1, 1:-1]
    colors, counts = np.unique(interior, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert counts.max() > interior.size/2 and bg != frame
    assert not np.any(interior == frame)
    edges = {"top": g[0, 1:-1], "bottom": g[-1, 1:-1],
             "left": g[1:-1, 0], "right": g[1:-1, -1]}
    edge_colors = set(int(c) for edge in edges.values() for c in edge) - {frame}
    assert bg not in edge_colors
    unpaired = [color for color in edge_colors
                if not np.array_equal(edges["top"] == color, edges["bottom"] == color)
                or not np.array_equal(edges["left"] == color, edges["right"] == color)]
    assert len(unpaired) == 1, "one visibly unpaired control color"
    control = unpaired[0]
    assert not np.any(interior == control)
    markers = {}
    for side, edge in edges.items():
        indices = np.flatnonzero(edge == control)
        assert len(indices) <= 1
        if len(indices):
            pos = int(indices[0])+1
            markers[side] = ((0, pos) if side == "top" else (h-1, pos) if side == "bottom"
                             else (pos, 0) if side == "left" else (pos, w-1))
    sides = tuple(side for side in SIDES if side in markers)
    assert len(sides) in (1, 2)
    if len(sides) == 2:
        assert len(set(sides) & {"top", "bottom"}) == 1
        assert len(set(sides) & {"left", "right"}) == 1
    ar = next((markers[s][0] for s in sides if s in ("left", "right")), None)
    ac = next((markers[s][1] for s in sides if s in ("top", "bottom")), None)
    templates = {}
    payload = set()
    for color in sorted(set(map(int, colors)) - {bg}):
        cells = set(map(tuple, np.argwhere(g == color)))
        cells = {(int(r), int(c)) for r, c in cells if 0 < r < h-1 and 0 < c < w-1}
        payload |= cells
        rows = [int(x)+1 for x in np.flatnonzero((edges["left"] == color) & (edges["right"] == color))]
        cols = [int(x)+1 for x in np.flatnonzero((edges["top"] == color) & (edges["bottom"] == color))]
        active = len(rows) >= 2 and len(cols) >= 2
        rooms = [(a+1, b-1, x+1, y-1) for a, b in zip(rows, rows[1:])
                 for x, y in zip(cols, cols[1:])] if active else []
        groups = _partition(cells, ar, ac, sides)
        templates[color] = dict(cells=cells, rows=rows, cols=cols, active=active, rooms=rooms, groups=groups)
    assert payload and any(t["active"] for t in templates.values())
    # The union is one payload; individual color masks and cut partitions may fragment.
    assert len(_components(payload)) == 1
    guide_cells = set().union(*(_guide(s, markers[s], payload) for s in sides))
    fixed = {p for p in payload if p[0] == ar or p[1] == ac}
    preserved = fixed | set().union(*(t["cells"] for t in templates.values() if not t["active"]))
    occupied = preserved | guide_cells
    plans = []
    for color, t in templates.items():
        if not t["active"]:
            continue
        for room in t["rooms"]:
            a, b, x, y = room
            assert 1 <= a <= b < h-1 and 1 <= x <= y < w-1
            for corner, cells in t["groups"].items():
                if not _retained(corner, sides):
                    continue
                targets = _destination(cells, corner, room)
                assert all(a <= r <= b and x <= c <= y for r, c in targets), "partition does not fit courtyard"
                assert not targets & occupied, "stencils must not collide with preserved cells or other stencils"
                occupied |= targets
                plans.append((color, corner, cells, room, targets))
    assert plans
    return dict(grid=g, bg=bg, frame=frame, control=control, markers=markers, sides=sides,
                ar=ar, ac=ac, templates=templates, payload=payload, guides=guide_cells,
                fixed=fixed, preserved=preserved, plans=plans)


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    out = p["grid"].copy()
    for t in p["templates"].values():
        if t["active"]:
            for r, c in t["cells"] - p["fixed"]:
                out[r, c] = p["bg"]
    for r, c in p["guides"]:
        if out[r, c] == p["bg"]:
            out[r, c] = p["control"]
    for color, corner, cells, room, targets in p["plans"]:
        for r, c in targets:
            out[r, c] = color
    return out


def _mutant(p, kind):
    """Deliberately incomplete rules, used only as evidence/validation diagnostics."""
    out = p["grid"].copy()
    for t in p["templates"].values():
        if t["active"]:
            for r, c in t["cells"] - p["fixed"]:
                out[r, c] = p["bg"]
    rays = (set().union(*(_guide(s, p["markers"][s], p["payload"], nearest=True) for s in p["sides"]))
            if kind == "nearest" else p["guides"])
    for r, c in rays:
        if out[r, c] == p["bg"]:
            out[r, c] = p["control"]
    h, w = out.shape
    for color, t in p["templates"].items():
        if not t["active"]:
            continue
        rooms = t["rooms"]
        if kind == "no_A":
            rooms = [(1, h-2, 1, w-2)]
        elif kind == "first_room":
            rooms = rooms[:1]
        elif kind == "zip":
            rooms = [(a+1, b-1, x+1, y-1) for (a, b), (x, y) in zip(zip(t["rows"], t["rows"][1:]), zip(t["cols"], t["cols"][1:]))]
        groups = t["groups"] if kind != "no_B" else {("top", "left"): t["cells"]}
        for room in rooms:
            for corner, cells in groups.items():
                if kind not in ("all_quadrants", "no_B") and not _retained(corner, p["sides"]):
                    continue
                parts = _components(cells) if kind == "components" else [cells]
                for part in parts:
                    for r, c in _destination(part, corner, room):
                        # Mutants need not satisfy the legitimate fit/collision constraints.
                        if 0 < r < h-1 and 0 < c < w-1:
                            out[r, c] = color
    return out


def _features(g):
    p = _parse(g)
    expected = solve(g)
    active = [t for t in p["templates"].values() if t["active"]]
    gaps, short = False, False
    for side, point in p["markers"].items():
        axis = 0 if side in ("top", "bottom") else 1
        hits = sorted(v[axis] for v in p["payload"] if v[1-axis] == point[1-axis])
        gaps |= any(b-a > 1 for a, b in zip(hits, hits[1:]))
        far = max if side in ("top", "left") else min
        short |= far(hits) != far(v[axis] for v in p["payload"])
    groups = [(corner, cells) for t in active for corner, cells in t["groups"].items() if _retained(corner, p["sides"])]
    empty_corner = False
    for corner, cells in groups:
        a, b, x, y = _box(cells)
        empty_corner |= (a if corner[0] == "top" else b, x if corner[1] == "left" else y) not in cells
    return dict(shape=g.shape, sides=p["sides"], bg=p["bg"], active=len(active),
                inactive=any(not t["active"] for t in p["templates"].values()),
                products=tuple((len(t["rows"])-1, len(t["cols"])-1) for t in active),
                gap=bool(gaps), short=bool(short),
                fragmented=any(len(_components(cells)) > 1 for corner, cells in groups),
                deleted=any(not _retained(corner, p["sides"]) for t in active for corner in t["groups"]),
                empty_corner=bool(empty_corner),
                masks=tuple(tuple(sorted((r-min(v[0] for v in t["cells"]), c-min(v[1] for v in t["cells"])) for r, c in t["cells"])) for t in active),
                differences={kind: not np.array_equal(expected, _mutant(p, kind))
                             for kind in ("no_A", "no_B", "first_room", "zip", "nearest", "components", "all_quadrants")})


def _grow_mask(rng, h, w):
    """Connected waypoint paths and growth; not a finite template bank."""
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    points = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
              (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
    rng.shuffle(points)
    for tr, tc in points:
        r, c = sorted(cells)[int(rng.integers(len(cells)))]
        while (r, c) != (tr, tc):
            if r != tr and (c == tc or rng.random() < .5):
                r += int(np.sign(tr-r))
            else:
                c += int(np.sign(tc-c))
            cells.add((r, c))
    count = int(rng.uniform(.28, .85)*h*w)
    while len(cells) < count:
        choices = sorted({q for r, c in cells for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                          if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells})
        if not choices:
            break
        cells.add(choices[int(rng.integers(len(choices)))])
    return cells


def _source(rng, kind, nactive, double):
    if kind == "single_witness":
        h, w = int(rng.integers(5, 8)), int(rng.integers(4, 7))
        ac, ar = int(rng.integers(1, w-1)), None
        cells = {(r, ac-1) for r in range(h)} | {(0, ac), (h-2, ac)}
        for r in (0, h-2):
            length = int(rng.integers(1, w-ac))
            cells |= {(r, c) for c in range(ac+1, ac+1+length)}
        # Variable left branches do not close the deliberately empty axis gap.
        for r in rng.choice(h, size=int(rng.integers(1, h)), replace=False):
            cells |= {(int(r), c) for c in range(int(rng.integers(ac)), ac)}
        return [cells], ar, ac, h, w
    if kind == "double_witness":
        h = w = 5
        ar = ac = 2
        cells = {(r, 1) for r in range(5)} | {(2, c) for c in range(5)}
        cells |= {(0, 2), (0, 3), (3, 2), (3, 3)}
        if rng.random() < .5:
            cells.add((0, 4))
        if rng.random() < .5:
            cells.add((4, 3))
            cells.add((4, 2))
        secondary = {(3, 3)}
        if rng.random() < .5:
            secondary.add((2, 3))
        inactive = {(4, 1)}
        return [cells-secondary-inactive, secondary, inactive], ar, ac, h, w
    max_size = 8 if nactive == 1 else 6 if nactive == 2 else 5
    h, w = int(rng.integers(3, max_size+1)), int(rng.integers(3, max_size+1))
    cells = _grow_mask(rng, h, w)
    ac = int(rng.integers(1, w-1))
    ar = int(rng.integers(1, h-1)) if double else None
    if not any(c == ac for r, c in cells) or (double and not any(r == ar for r, c in cells)):
        return None
    inactive = bool(rng.random() < .35)
    count = nactive + int(inactive)
    if len(cells) < count+3:
        return None
    # Local Voronoi color domains, with occasional sparse patchwork assignments.
    seeds = [sorted(cells)[int(i)] for i in rng.choice(len(cells), count, replace=False)]
    masks = [set() for _ in seeds]
    for r, c in sorted(cells):
        distances = [abs(r-y)+abs(c-x)+float(rng.uniform(0, .3)) for y, x in seeds]
        idx = int(np.argmin(distances))
        if rng.random() < .10:
            idx = int(rng.integers(count))
        masks[idx].add((r, c))
    if any(not m for m in masks):
        return None
    if any(not any(_retained(k, ("top", "right") if double else ("top",))
                   for k in _partition(m, ar, ac, ("top", "right") if double else ("top",))) for m in masks[:nactive]):
        return None
    return masks, ar, ac, h, w


def _room_size(groups, sides):
    groups = {k: v for k, v in groups.items() if _retained(k, sides)}
    heights, widths = {}, {}
    for corner, cells in groups.items():
        a, b, x, y = _box(cells)
        heights[corner[0]] = max(heights.get(corner[0], 0), b-a+1)
        widths[corner[1]] = max(widths.get(corner[1], 0), y-x+1)
    return sum(heights.values())+1, sum(widths.values())+1


def _build_pair(rng, kind, nactive, double, products, turn, palette):
    source = _source(rng, kind, nactive, double)
    if source is None:
        return None
    masks, ar, ac, sh, sw = source
    sides = ("top", "right") if double else ("top",)
    r0, c0 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    cells = [{(r+r0, c+c0) for r, c in mask} for mask in masks]
    ar, ac = ar+r0 if ar is not None else None, ac+c0
    bg, frame, control = palette[:3]
    colors = list(palette[3:3+len(cells)])
    # Separate target bands horizontally, while row lattices are independently
    # spaced. Their differing row labels remain unique on the outer border.
    next_col = c0+sw+int(rng.integers(1, 3))
    row_base = r0+sh+int(rng.integers(1, 3))
    used_rows, coordinates = set(), []
    for idx in range(nactive):
        groups = _partition(cells[idx], ar, ac, sides)
        rh, rw = _room_size(groups, sides)
        nr, nc = products[idx]
        row_options = []
        for trial in range(18):
            rows = [row_base+int(rng.integers(0, 3))]
            for _ in range(nr):
                rows.append(rows[-1]+rh+1+int(rng.integers(0, 3)))
            if rows[-1] <= 28 and not set(rows) & used_rows:
                row_options = rows
                break
        if not row_options:
            return None
        used_rows |= set(row_options)
        cols = [next_col]
        for _ in range(nc):
            cols.append(cols[-1]+rw+1+int(rng.integers(0, 3)))
        if cols[-1] > 28:
            return None
        next_col = cols[-1]+int(rng.integers(1, 3))
        coordinates.append((row_options, cols))
    h = max(max(rs) for rs, cs in coordinates)+2+int(rng.integers(0, 2))
    w = max(max(cs) for rs, cs in coordinates)+2+int(rng.integers(0, 2))
    if max(h, w) > 30:
        return None
    inp = np.full((h, w), bg, dtype=np.int8)
    inp[[0, -1], :] = frame
    inp[:, [0, -1]] = frame
    markers = {"top": (0, ac)}
    if double:
        markers["right"] = (ar, w-1)
    for point in markers.values():
        inp[point] = control
    for color, mask in zip(colors, cells):
        for point in mask:
            inp[point] = color
    for color, (rows, cols) in zip(colors, coordinates):
        for r in rows:
            inp[r, 0] = inp[r, -1] = color
        for c in cols:
            inp[0, c] = inp[-1, c] = color
    for color in colors[nactive:]:
        # Three inactivity causes: no labels, only row labels, only column labels.
        mode = int(rng.integers(3))
        if mode == 1:
            options = [r for r in range(1, h-1) if inp[r, 0] == inp[r, -1] == frame]
            r = int(rng.choice(options))
            inp[r, 0] = inp[r, -1] = color
        elif mode == 2:
            options = [c for c in range(1, w-1) if inp[0, c] == inp[-1, c] == frame]
            c = int(rng.choice(options))
            inp[0, c] = inp[-1, c] = color
    # Construction-owned expected output: crop each recorded partition into a
    # Boolean mask and paint it at each room corner. The solver instead parses
    # the returned input and translates cell sets directly.
    out = inp.copy()
    all_cells = set().union(*cells)
    for mask in cells[:nactive]:
        for r, c in mask:
            if r != ar and c != ac:
                out[r, c] = bg
    for side, point in markers.items():
        for r, c in _guide(side, point, all_cells):
            if out[r, c] == bg:
                out[r, c] = control
    for color, mask, (rows, cols) in zip(colors, cells, coordinates):
        for corner, part in _partition(mask, ar, ac, sides).items():
            if not _retained(corner, sides):
                continue
            a, b, x, y = _box(part)
            crop = np.zeros((b-a+1, y-x+1), dtype=bool)
            for r, c in part:
                crop[r-a, c-x] = True
            for lo, hi in zip(rows, rows[1:]):
                for left, right in zip(cols, cols[1:]):
                    top = lo+1 if corner[0] == "top" else hi-crop.shape[0]
                    start = left+1 if corner[1] == "left" else right-crop.shape[1]
                    view = out[top:top+crop.shape[0], start:start+crop.shape[1]]
                    if view.shape != crop.shape or np.any(view[crop] != bg):
                        return None
                    view[crop] = color
    pair = {"input": np.rot90(inp, -turn).copy(), "output": np.rot90(out, -turn).copy()}
    try:
        _parse(pair["input"])
    except AssertionError:
        return None
    assert np.array_equal(solve(pair["input"]), pair["output"])
    return pair


def _make_pair(rng, kind, nactive, double, products, turn, palette):
    for _ in range(300):
        pair = _build_pair(rng, kind, nactive, double, products, turn, palette)
        if pair is not None:
            if kind == "single_witness":
                f = _features(pair["input"])
                if not (f["gap"] and f["short"] and f["fragmented"] and f["differences"]["components"]):
                    continue
            elif kind == "double_witness":
                f = _features(pair["input"])
                if not (f["deleted"] and f["inactive"] and f["differences"]["zip"]):
                    continue
            return pair
    raise AssertionError("bounded geometry search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    single_turn, double_turn = int(rng.integers(4)), int(rng.integers(4))
    mode = int(rng.integers(4))
    palette_varies = bool(rng.random() < .7)
    palette = tuple(map(int, rng.permutation(10)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            if palette_varies:
                palette = tuple(map(int, rng.permutation(10)))
            if split == "train" and index == 0:
                kind, nactive, double, products = "single_witness", 1, False, [(1, 1)]
            elif split == "train" and index == 1:
                kind, nactive, double, products = "double_witness", 2, True, [(2, 2), (1, 1)]
            else:
                kind = "random"
                if split == "test" and mode == 2:
                    nactive, double = 3, bool(rng.integers(2))
                    products = [(1, 1)]*nactive
                elif split == "test" and mode == 3:
                    nactive, double = 1, True
                    products = [(1, 3) if rng.random() < .5 else (3, 1)]
                else:
                    nactive = int(rng.choice([1, 2], p=[.65, .35]))
                    double = bool(mode != 0) if split == "test" else bool(rng.integers(2))
                    product = (1, 2) if rng.random() < .5 else (2, 1)
                    products = [product]+[(1, 1)]*(nactive-1)
            turn = double_turn if double else single_turn
            pair = _make_pair(rng, kind, nactive, double, products, turn, palette)
            # Bijective cyclic recoloring is a bounded uniqueness fallback; 10
            # maps alone may be insufficient for 14 pairs, so also swap roles.
            originals = {k: v.copy() for k, v in pair.items()}
            for attempt in range(20):
                mapping = np.arange(10, dtype=np.int8)
                if attempt >= 10:
                    mapping[palette[0]], mapping[palette[1]] = palette[1], palette[0]
                mapping = (mapping+attempt%10)%10
                candidate = {k: mapping[v] for k, v in originals.items()}
                if _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            else:
                raise AssertionError("uniqueness fallback exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    features, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            assert np.array_equal(solve(inp), out), "incorrect courtyard transformation"
            assert not np.array_equal(inp, out)
            assert np.array_equal(inp[[0, -1], :], out[[0, -1], :])
            assert np.array_equal(inp[:, [0, -1]], out[:, [0, -1]])
            for point in p["preserved"]:
                assert out[point] == inp[point]
            for point in p["guides"] - p["payload"]:
                assert out[point] == p["control"]
            for color, t in p["templates"].items():
                original_count = len(t["cells"])
                if t["active"]:
                    fixed_count = len(t["cells"] & p["fixed"])
                    retained_count = sum(len(v) for k, v in t["groups"].items() if _retained(k, p["sides"]))
                    expected = fixed_count + len(t["rooms"])*retained_count
                else:
                    expected = original_count
                assert int(np.count_nonzero(out[1:-1, 1:-1] == color)) == expected
            features[split].append(_features(inp))
    train, test = features["train"], features["test"]
    assert {f["sides"] for f in test} <= {f["sides"] for f in train}, "test uses unseen guide configuration"
    for feature in ("gap", "short", "fragmented", "deleted", "inactive"):
        if any(f[feature] for f in test):
            assert any(f[feature] for f in train), f"missing training witness for {feature}"
    for kind in ("no_A", "no_B", "first_room", "zip", "nearest", "components", "all_quadrants"):
        assert any(f["differences"][kind] for f in train), f"missing distinguishing demonstration: {kind}"
    assert any(any(a > 1 and b > 1 for a, b in f["products"]) for f in train)


def _literal_check():
    """Small manually enumerated expected case, without coordinate-placement code."""
    inp = np.zeros((13, 17), dtype=np.int8)
    inp[[0, -1], :] = 8
    inp[:, [0, -1]] = 8
    inp[0, 3], inp[3, -1] = 9, 9
    for r, c in ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)):
        inp[r, c] = 2
    for r, c in ((1, 4), (3, 4), (3, 5), (4, 5)):
        inp[r, c] = 3
    inp[5, 2] = 4
    for color, rows, cols in ((2, (6, 11), (7, 12)), (3, (5, 10), (13, 15)), (4, (2,), ())):
        for r in rows:
            inp[r, 0] = inp[r, -1] = color
        for c in cols:
            inp[0, c] = inp[-1, c] = color
    out = inp.copy()
    for point in ((2, 2), (2, 4), (4, 2), (4, 4), (1, 4), (4, 5)):
        out[point] = 0
    out[1, 3] = 9
    out[3, 6:16] = 9
    for point in ((7, 8), (7, 11), (10, 11)):
        out[point] = 2
    for point in ((6, 14), (9, 14)):
        out[point] = 3
    assert np.array_equal(solve(inp), out)
    for turn in range(4):
        for reflect in (False, True):
            a, b = np.rot90(inp, turn), np.rot90(out, turn)
            if reflect:
                a, b = a[:, ::-1], b[:, ::-1]
            assert np.array_equal(solve(a), b)
    for shift in range(10):
        assert np.array_equal(solve((inp+shift)%10), (out+shift)%10)


def self_test() -> dict:
    """Bounded 42-seed suite, legal counts, metamorphisms and mutation rejection."""
    _literal_check()
    features, structures, shapes, masks = [], set(), set(), set()
    split_sides = {"train": set(), "test": set()}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                f = _features(g)
                features.append(f)
                shapes.add(g.shape)
                masks.update(f["masks"])
                split_sides[split].add(f["sides"])
                roles = {}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in g.flat)
                structures.add((g.shape, normalized))
        # Every bad rule is rejected by full-grid validation, with two demos.
        if seed < 7:
            kind = ("no_A", "no_B", "first_room", "zip", "nearest", "components", "all_quadrants")[seed]
            chosen = next((i for i, p in enumerate(puzzle["train"]) if _features(p["input"])["differences"][kind]), None)
            assert chosen is not None
            bad = copy.deepcopy(puzzle)
            bad["train"][chosen]["output"] = _mutant(_parse(bad["train"][chosen]["input"]), kind)
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"validator accepted {kind}")
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(structures) >= 175 and len(masks) >= 100
    assert {f["bg"] for f in features} == set(range(10))
    assert {1, 2, 3} <= {f["active"] for f in features}
    products = {p for f in features for p in f["products"]}
    assert {(1, 1), (2, 2), (1, 2), (2, 1), (1, 3), (3, 1)} <= products
    for split in split_sides:
        assert len(split_sides[split]) == 8
    for feature in ("gap", "short", "fragmented", "inactive", "deleted", "empty_corner"):
        assert sum(f[feature] for f in features) >= 5, feature
    # Every legal explicit combination, using the minimal or maximal evidence budget.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(42000+a*10+b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    return dict(seeds=42, pairs=len(features), grid_shapes=len(shapes),
                normalized_inputs=len(structures), masks=len(masks),
                products=sorted(products), guide_configurations={k: len(v) for k, v in split_sides.items()},
                independent_solver=False)
