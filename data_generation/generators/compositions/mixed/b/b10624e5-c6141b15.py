"""Exchange a reference's positional roles, then transport the result by anchors.

Only NumPy and the standard library are used. No parent generator is imported.
All construction is in memory; self_test retains only small feature sets/hashes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from numbers import Integral

import numpy as np


_GLYPHS = (
    frozenset({(0, 0)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}),
    frozenset({(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}),
    frozenset({(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(-1, -1), (-1, 0), (0, 0), (0, 1), (1, -1), (1, 0)}),
    frozenset({(-2, -1), (-2, 0), (-2, 1), (-1, 0), (0, 0),
               (1, 0), (2, -1), (2, 0), (2, 1)}),
    frozenset({(-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
               (0, 0), (1, 0)}),
)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "non-lattice edge"
    n = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + k * sr, a[1] + k * sc) for k in range(n + 1))


def _components(points):
    unseen, result = set(points), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(frozenset(component))
    return result


def _figure(centers):
    return frozenset().union(*(_line(a, b) for a, b in itertools.combinations(centers, 2)))


def _copies(centers, glyph):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in glyph)


def _roles(motif, anchor):
    """Decode the source's line and repeated glyph from its logical pixels."""
    colors = set(motif.values()) - {anchor}
    assert len(colors) == 2
    candidates = []
    for edge in sorted(colors):
        points = frozenset(p for p, color in motif.items() if color == edge)
        if len(points) < 3:
            continue
        a, b = min(points), max(points)
        try:
            if _line(a, b) != points:
                continue
        except AssertionError:
            continue
        glyph_color = next(iter(colors - {edge}))
        parts = _components(p for p, color in motif.items() if color == glyph_color)
        if len(parts) not in (2, 3):
            continue
        centers, shapes = [], []
        for part in parts:
            rows, cols = zip(*part)
            if (min(rows) + max(rows)) % 2 or (min(cols) + max(cols)) % 2:
                break
            center = ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)
            shape = frozenset((r - center[0], c - center[1]) for r, c in part)
            if (0, 0) not in shape:
                break
            centers.append(center)
            shapes.append(shape)
        else:
            if any(shape != shapes[0] for shape in shapes):
                continue
            centers = tuple(sorted(centers))
            if len(centers) == 3:
                p, q, r = centers
                if (q[0]-p[0])*(r[1]-p[1]) == (q[1]-p[1])*(r[0]-p[0]):
                    continue
            try:
                figure = _figure(centers)
            except AssertionError:
                continue
            glyph = shapes[0]
            endpoint_glyphs = _copies((a, b), glyph)
            if figure & endpoint_glyphs or (0, 0) in figure | endpoint_glyphs:
                continue
            if len(_components(endpoint_glyphs)) != 2:
                continue
            # Input roles have a background gap; the register is allowed to touch.
            glyph_cells = _copies(centers, glyph)
            if any((r+dr, c+dc) in glyph_cells for r, c in points
                   for dr, dc in itertools.product((-1, 0, 1), repeat=2)):
                continue
            exchanged = {p: edge for p in figure}
            exchanged.update({p: glyph_color for p in endpoint_glyphs})
            exchanged[0, 0] = anchor
            candidates.append({"edge": edge, "glyph_color": glyph_color,
                               "endpoints": (a, b), "centers": centers,
                               "glyph": glyph, "exchanged": exchanged})
    assert len(candidates) == 1, "ambiguous source roles"
    return candidates[0]


def _bounds(h, w, row, col):
    return ((0, row, 0, col), (0, row, col+1, w),
            (row+1, h, 0, col), (row+1, h, col+1, w))


def _signs(reference, panel):
    return (-1 if reference//2 != panel//2 else 1,
            -1 if reference % 2 != panel % 2 else 1)


def _square(panel, color):
    coords = np.argwhere(panel == color)
    assert len(coords)
    top, left = map(int, coords.min(0))
    bottom, right = map(int, coords.max(0) + 1)
    side = bottom - top
    assert side == right - left and len(coords) == side * side
    return top, left, side


def _decode(grid):
    rows = [r for r in range(1, grid.shape[0]-1) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(1, grid.shape[1]-1) if np.all(grid[:, c] == grid[0, c])]
    crosses = [(r, c) for r in rows for c in cols if grid[r, 0] == grid[0, c]]
    assert len(crosses) == 1, "expected one cross divider"
    row, col = crosses[0]
    divider = int(grid[row, col])
    bounds = _bounds(*grid.shape, row, col)
    panels = [grid[t:b, l:r] for t, b, l, r in bounds]
    values = np.concatenate([p.ravel() for p in panels])
    counts = np.bincount(values, minlength=10)
    bg = int(counts.argmax())
    assert sum(counts == counts.max()) == 1 and bg != divider
    palettes = [set(map(int, np.unique(p))) for p in panels]
    shared = set.intersection(*palettes) - {bg}
    assert len(shared) == 1
    anchor = next(iter(shared))
    assert anchor != divider
    references = [p for p in range(4) if palettes[p] - {bg, anchor}]
    assert len(references) == 1
    reference = references[0]
    assert all(p == reference or palettes[p] == {bg, anchor} for p in range(4))
    anchors = [_square(panel, anchor) for panel in panels]
    top, left, unit = anchors[reference]
    panel = panels[reference]
    motif = {}
    for y, x in np.argwhere(panel != bg):
        q = ((int(y)-top)//unit, (int(x)-left)//unit)
        motif[q] = int(panel[y, x])
    assert motif.get((0, 0)) == anchor
    assert sum(v == anchor for v in motif.values()) == 1
    # The square register defines the common unit. Every source cell is a full
    # uniform block on its register-relative lattice; no interpolation occurs.
    reconstructed = np.full(panel.shape, bg, dtype=np.int8)
    for (r, c), color in motif.items():
        y, x = top+r*unit, left+c*unit
        assert 0 <= y <= panel.shape[0]-unit and 0 <= x <= panel.shape[1]-unit
        reconstructed[y:y+unit, x:x+unit] = color
    assert np.array_equal(reconstructed, panel), "incomplete logical source cell"
    roles = _roles(motif, anchor)
    # Exchange is fully drawable in the reference, even when destinations clip.
    assert all(0 <= top+r*unit <= panel.shape[0]-unit and
               0 <= left+c*unit <= panel.shape[1]-unit for r, c in roles["exchanged"])
    return {"bg": bg, "divider": divider, "anchor": anchor, "reference": reference,
            "bounds": bounds, "panels": panels, "anchors": anchors, "motif": motif,
            **roles}


def _paint(panel, motif, placement, side, signs=(1, 1)):
    """Paint solid transformed cells, clipping rectangles at this panel only."""
    top, left = placement
    vr, hc = signs
    visible, total = 0, len(motif)*side*side
    colors = set()
    for (r, c), color in motif.items():
        y, x = top+vr*r*side, left+hc*c*side
        y0, y1 = max(0, y), min(panel.shape[0], y+side)
        x0, x1 = max(0, x), min(panel.shape[1], x+side)
        if y0 < y1 and x0 < x1:
            panel[y0:y1, x0:x1] = color
            visible += (y1-y0)*(x1-x0)
            colors.add(color)
    return total-visible, colors


def _render(grid, data, *, exchange=True, reflect=True, scale=True):
    out = grid.copy()
    motif = data["exchanged"] if exchange else data["motif"]
    for p, (t, b, l, r) in enumerate(data["bounds"]):
        panel = out[t:b, l:r]
        panel[:] = data["bg"]
        y, x, side = data["anchors"][p]
        actual_side = side if scale or p == data["reference"] else data["anchors"][data["reference"]][2]
        signs = _signs(data["reference"], p) if reflect else (1, 1)
        _paint(panel, motif, (y, x), actual_side, signs)
        # Counterfactuals preserve the visibly given register as well.
        panel[y:y+side, x:x+side] = data["anchor"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every output-affecting choice from the supplied input only."""
    grid = _grid(input_grid)
    return _render(grid, _decode(grid))


def _translate(points, glyph, shape, rng):
    all_points = _copies(points, glyph)
    rows, cols = zip(*all_points)
    limits = (-min(rows), shape[0]-1-max(rows), -min(cols), shape[1]-1-max(cols))
    if limits[0] > limits[1] or limits[2] > limits[3]:
        return None
    r = int(rng.integers(limits[0], limits[1]+1))
    c = int(rng.integers(limits[2], limits[3]+1))
    return tuple((y+r, x+c) for y, x in points)


def _layout(rng, count, glyph, orientation, compact=False, test=False):
    """Sample two interacting arrangements, then attach an empty square register."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for _ in range(160):
        low, high = (6, 10) if compact else (9, 13) if test else (10, 15)
        shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
        radius = max(max(abs(r), abs(c)) for r, c in glyph)
        distance = int(rng.integers(max(3, 2*radius+2), max(3, 2*radius+2)+4))
        if count == 2:
            dr, dc = directions[int(rng.integers(4))]
            relative = ((0, 0), (dr*distance, dc*distance))
        elif rng.random() < .35:
            relative = ((0, 0), (distance, distance), (distance, -distance))
        else:
            relative = ((0, 0), (0, distance), (distance, 0))
        # Rotate and reflect the *arrangement*, independently of glyph appearance.
        k, flip = int(rng.integers(4)), int(rng.choice((-1, 1)))
        transformed = []
        for r, c in relative:
            c *= flip
            for _rotation in range(k):
                r, c = c, -r
            transformed.append((r, c))
        centers = _translate(tuple(transformed), glyph, shape, rng)
        dr, dc = directions[orientation if orientation < 2 else int(rng.integers(2, 4))]
        length = int(rng.integers(max(3, 2*radius+2), max(3, 2*radius+2)+5))
        endpoints = _translate(((0, 0), (dr*length, dc*length)), glyph, shape, rng)
        if centers is None or endpoints is None:
            continue
        edge, marks = _line(*endpoints), _copies(centers, glyph)
        if any((r+dy, c+dx) in marks for r, c in edge
               for dy, dx in itertools.product((-1, 0, 1), repeat=2)):
            continue
        if len(_components(marks)) != count:
            continue
        figure, endpoint_marks = _figure(centers), _copies(endpoints, glyph)
        if figure & endpoint_marks or len(_components(endpoint_marks)) != 2:
            continue
        occupied, exchanged = edge | marks, figure | endpoint_marks
        # A register adjacent to the exchanged figure guarantees that even a
        # small clipped destination can show a non-register part.
        available = sorted({(r+dy, c+dx) for r, c in exchanged
                            for dy, dx in itertools.product((-1, 0, 1), repeat=2)
                            if 0 <= r+dy < shape[0] and 0 <= c+dx < shape[1]}
                           - occupied - exchanged)
        if not available:
            continue
        register = available[int(rng.integers(len(available)))]
        return centers, endpoints, register, glyph
    return None


def _fallback_layout(count, orientation, glyph):
    # Finite feasible construction; randomness normally supplies the layouts.
    # These source/target unions fit inside 8x8 (two) or 12x12 (three).
    if count == 2:
        glyph = _GLYPHS[0]
        centers = ((1, 1), (1, 5))
        endpoints = (((5, 1), (5, 5)), ((3, 5), (7, 5)), ((4, 1), (7, 4)))[orientation]
    else:
        if max(max(abs(r), abs(c)) for r, c in glyph) > 1:
            glyph = _GLYPHS[3]
        centers = ((2, 2), (2, 6), (6, 2))
        endpoints = (((10, 2), (10, 8)), ((2, 10), (8, 10)), ((7, 7), (10, 10)))[orientation]
    occupied = _line(*endpoints) | _copies(centers, glyph)
    exchanged = _figure(centers) | _copies(endpoints, glyph)
    available = sorted({(r+dy, c+dx) for r, c in exchanged
                        for dy, dx in itertools.product((-1, 0, 1), repeat=2)
                        if r+dy >= 0 and c+dx >= 0} - occupied - exchanged)
    # Minimize union size to keep the finite construction within 12x12.
    register = min(available, key=lambda p: (max([p[0]]+[r for r, c in occupied | exchanged]) *
                                           max([p[1]]+[c for r, c in occupied | exchanged]), p))
    return centers, endpoints, register, glyph


def _make_motifs(layout, colors):
    centers, endpoints, register, glyph = layout
    bg, divider, anchor, edge, glyph_color = colors
    def relative(p):
        return p[0]-register[0], p[1]-register[1]
    original = {relative(p): edge for p in _line(*endpoints)}
    original.update({relative(p): glyph_color for p in _copies(centers, glyph)})
    original[0, 0] = anchor
    exchanged = {relative(p): edge for p in _figure(centers)}
    exchanged.update({relative(p): glyph_color for p in _copies(endpoints, glyph)})
    exchanged[0, 0] = anchor
    return original, exchanged


def _extent(motif, side=1):
    rows, cols = zip(*motif)
    return ((max(rows)-min(rows)+1)*side, (max(cols)-min(cols)+1)*side)


def _placements(shape, motif, side, signs, clipped, anchor):
    # At most 900 placements x <=100 logical cells, bounded; no saved rasters.
    points = np.array(list(motif), dtype=int)
    colors = np.array(list(motif.values()), dtype=int)
    points = points * np.array(signs) * side
    candidates = []
    total = len(motif)*side*side
    for y in range(shape[0]-side+1):
        for x in range(shape[1]-side+1):
            q = points + [y, x]
            h = np.maximum(0, np.minimum(shape[0], q[:, 0]+side)-np.maximum(0, q[:, 0]))
            w = np.maximum(0, np.minimum(shape[1], q[:, 1]+side)-np.maximum(0, q[:, 1]))
            area = h*w
            visible = int(area.sum())
            if (visible < total) == clipped and int(area[colors != anchor].sum()) > 0:
                candidates.append((y, x, visible))
    return candidates


def _build(rng, layout, colors, reference, units, flags):
    original, exchanged = _make_motifs(layout, colors)
    union = original | exchanged
    min_rows, min_cols = [3, 3], [3, 3]
    for p in range(4):
        h, w = _extent(union if p == reference else exchanged, units[p])
        if flags[p]:
            h = w = units[p]+2
        min_rows[p//2] = max(min_rows[p//2], h)
        min_cols[p % 2] = max(min_cols[p % 2], w)
    sizes = []
    for minimum in (min_rows, min_cols):
        if sum(minimum)+1 > 30:
            return None
        extra = int(rng.integers(0, 31-(sum(minimum)+1)))
        first = int(rng.integers(extra+1))
        sizes.append((minimum[0]+first, minimum[1]+extra-first))
    heights, widths = sizes
    h, w = sum(heights)+1, sum(widths)+1
    bounds = _bounds(h, w, heights[0], widths[0])
    inp = np.full((h, w), colors[0], dtype=np.int8)
    inp[heights[0], :] = colors[1]
    inp[:, widths[0]] = colors[1]
    out = inp.copy()
    for p, (t, b, l, r) in enumerate(bounds):
        side = units[p]
        signs = _signs(reference, p)
        candidates = _placements((b-t, r-l), union if p == reference else exchanged,
                                 side, signs, flags[p], colors[2])
        if not candidates:
            return None
        # Keep variation in clipping severity while favoring useful visible area.
        candidates.sort(key=lambda q: q[2], reverse=True)
        pool = candidates[:max(1, (len(candidates)+1)//2)] if flags[p] else candidates
        y, x, _visible = pool[int(rng.integers(len(pool)))]
        inp[t+y:t+y+side, l+x:l+x+side] = colors[2]
        if p == reference:
            _paint(inp[t:b, l:r], original, (y, x), side)
        _paint(out[t:b, l:r], exchanged, (y, x), side, signs)
    return {"input": inp, "output": out}


def _asymmetric(motif):
    return all({(vr*r, hc*c): color for (r, c), color in motif.items()} != motif
               for vr, hc in ((1, -1), (-1, 1), (-1, -1)))


def _pair_key(pair):
    return pair["input"].shape, pair["input"].tobytes()


def _palette(rng):
    colors = [int(c) for c in rng.permutation(10)[:5]]
    if rng.random() < .30:
        colors[int(rng.integers(3, 5))] = colors[1]
    return tuple(colors)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    references = list(map(int, rng.permutation(4)))
    orientations = list(map(int, rng.permutation(3)))
    featured = _GLYPHS[int(rng.integers(3, 6))]
    for split, count_pairs in (("train", num_train), ("test", num_test)):
        for index in range(count_pairs):
            first = split == "train" and index == 0
            second = split == "train" and index == 1
            test = split == "test"
            count = 2 if first else 3 if second else int(rng.integers(2, 4))
            glyph = _GLYPHS[0] if first else featured if second else _GLYPHS[int(rng.integers(len(_GLYPHS)))]
            if test and count == 3 and max(max(abs(r), abs(c)) for r, c in glyph) > 1:
                glyph = featured
            orientation = orientations[index] if first or second else int(rng.integers(3))
            reference = references[index] if first or second else int(rng.integers(4))
            dests = [p for p in range(4) if p != reference]
            units = [1]*4
            flags = [False]*4
            if first:
                units[reference] = 2  # Fully visible reduction witness.
            elif second:
                large = dests[int(rng.integers(3))]
                units[large], flags[large] = 2, True
            elif test:
                units[reference] = 2
                for p, unit in zip(dests, rng.permutation((1, 2, 3))):
                    units[p] = int(unit)
                    flags[p] = True
            else:
                units[reference] = int(rng.choice((1, 1, 2)))
                for p in dests:
                    units[p] = int(rng.integers(1, 3))
                    flags[p] = bool(rng.random() < .60)
            accepted = None
            # Bound retries, then use a compact, feasible layout. Subsequent
            # attempts vary only canvas slack/palette; no retained retry files.
            for attempt in range(72):
                colors = _palette(rng)
                layout = (_layout(rng, count, glyph, orientation, compact=first, test=test)
                          if attempt < 48 else _fallback_layout(count, orientation, glyph))
                if layout is None:
                    continue
                pair = _build(rng, layout, colors, reference, units, flags)
                if pair is None or _pair_key(pair) in seen:
                    continue
                try:
                    data = _check_pair(pair)
                    if first:
                        assert _asymmetric(data["exchanged"])
                    if second:
                        assert len(data["glyph"]) > 1
                    if index == 1 and split == "train":
                        old = _decode(puzzle["train"][0]["input"])
                        assert tuple(data[k] for k in ("bg", "divider", "anchor", "edge", "glyph_color")) != tuple(old[k] for k in ("bg", "divider", "anchor", "edge", "glyph_color"))
                except AssertionError:
                    continue
                accepted = pair
                break
            if accepted is None:
                # Extra random pairs may ask for impossible full enlargements.
                # Reuse the same rule with universally feasible all-clipped
                # destinations, a unit-one source, and a finite small layout.
                if not (first or second or test):
                    units = [1]*4
                    for p in dests:
                        flags[p] = True
                layout = _fallback_layout(count, orientation, glyph)
                for _ in range(90):
                    pair = _build(rng, layout, _palette(rng), reference, units, flags)
                    if pair is not None and _pair_key(pair) not in seen:
                        data = _check_pair(pair)
                        if first and not _asymmetric(data["exchanged"]):
                            continue
                        if second:
                            old = _decode(puzzle["train"][0]["input"])
                            keys = ("bg", "divider", "anchor", "edge", "glyph_color")
                            if tuple(data[k] for k in keys) == tuple(old[k] for k in keys):
                                continue
                        accepted = pair
                        break
            assert accepted is not None, "bounded construction exhausted"
            seen.add(_pair_key(accepted))
            puzzle[split].append(accepted)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    data = _decode(inp)
    expected = _render(inp, data)
    assert np.array_equal(expected, out), "incorrect exchange/transport"
    assert not np.array_equal(inp, out)
    # Reference performs both halves of the exchange, erasing obsolete pixels.
    for color in (data["edge"], data["glyph_color"]):
        before = {p for p, c in data["motif"].items() if c == color}
        after = {p for p, c in data["exchanged"].items() if c == color}
        assert before - after and after - before
    clipped = []
    destination_exchange_witness = False
    for p, (t, b, l, r) in enumerate(data["bounds"]):
        y, x, side = data["anchors"][p]
        assert np.all(out[t+y:t+y+side, l+x:l+x+side] == data["anchor"])
        if p == data["reference"]:
            continue
        target = np.full((b-t, r-l), data["bg"], dtype=np.int8)
        lost, colors = _paint(target, data["exchanged"], (y, x), side, _signs(data["reference"], p))
        assert colors - {data["anchor"]}, "empty destination completion"
        clipped.append(lost > 0)
        ordinary = np.full(target.shape, data["bg"], dtype=np.int8)
        _paint(ordinary, data["motif"], (y, x), side, _signs(data["reference"], p))
        destination_exchange_witness |= not np.array_equal(ordinary, target)
    assert destination_exchange_witness, "exchange invisible in every destination"
    data["clipped_count"] = sum(clipped)
    return data


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            data = _check_pair(pair)
            key = _pair_key({"input": _grid(pair["input"])})
            assert key not in seen
            seen.add(key)
            analyses[split].append(data)
    train = analyses["train"]
    assert {len(a["centers"]) for a in train} == {2, 3}
    assert len({a["reference"] for a in train}) >= 2
    def direction(a):
        p, q = a["endpoints"]
        return 0 if p[0] == q[0] else 1 if p[1] == q[1] else 2
    assert len({direction(a) for a in train}) >= 2
    assert len({tuple(a[k] for k in ("bg", "divider", "anchor", "edge", "glyph_color"))
                for a in train}) >= 2
    assert any(len(a["glyph"]) > 1 for a in train)
    witnesses = [a for a in train if a["clipped_count"] == 0 and _asymmetric(a["exchanged"])
                 and a["anchors"][a["reference"]][2] == 2
                 and all(s == 1 for p, (y, x, s) in enumerate(a["anchors"]) if p != a["reference"])]
    assert witnesses, "missing complete asymmetric scaling demonstration"
    assert any(a["anchors"][a["reference"]][2] == 1 and len(a["glyph"]) > 1
               and any(s == 2 for p, (y, x, s) in enumerate(a["anchors"]) if p != a["reference"])
               for a in train), "missing multi-cell enlargement demonstration"
    assert all(s <= 2 for a in train for y, x, s in a["anchors"])
    for a in analyses["test"]:
        assert a["anchors"][a["reference"]][2] == 2
        assert sorted(s for p, (y, x, s) in enumerate(a["anchors"]) if p != a["reference"]) == [1, 2, 3]
        assert a["clipped_count"] == 3, "missing held-out scale/clipping combination"
    for option in ("exchange", "reflect", "scale"):
        assert any(not np.array_equal(_render(_grid(pair["input"]), data, **{option: False}),
                                      _grid(pair["output"]))
                   for pair, data in zip(puzzle["train"], train)), f"unexposed {option}"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed checks, input-only semantic fixtures, and count coverage."""
    signatures, shapes, glyphs, refs, palettes, geometries, clip_counts, ratios = (set() for _ in range(8))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                a = _check_pair(pair)
                shapes.add(pair["input"].shape)
                glyphs.add(tuple(sorted(a["glyph"])))
                refs.add(a["reference"])
                palettes.add(tuple(a[k] for k in ("bg", "divider", "anchor", "edge", "glyph_color")))
                geometries.add((tuple(a["centers"]), tuple(a["endpoints"]), tuple(sorted(a["glyph"]))))
                clip_counts.add(a["clipped_count"])
                source = a["anchors"][a["reference"]][2]
                for p, (y, x, side) in enumerate(a["anchors"]):
                    if p != a["reference"]:
                        g = int(np.gcd(source, side))
                        ratios.add((side//g, source//g))
    assert len(signatures) == 42 and len(shapes) >= 25 and len(geometries) >= 100
    assert len(glyphs) >= 7 and refs == {0, 1, 2, 3} and len(palettes) >= 80
    assert clip_counts >= {0, 1, 2, 3} and ratios >= {(1, 2), (1, 1), (2, 1), (3, 2)}
    # This raster fixture is hand-specified and does not call generation.
    layout = _fallback_layout(3, 1, _GLYPHS[3])
    original, exchanged = _make_motifs(layout, (0, 1, 2, 3, 4))
    roles = _roles(original, 2)
    assert roles["exchanged"] == exchanged
    assert len(roles["centers"]) == 3 and len(roles["endpoints"]) == 2
    assert (0, 0) in exchanged and exchanged[0, 0] == 2
    # Check reflection of internal asymmetric glyph pixels and partial clipping.
    raw, reflected = np.zeros((30, 30), dtype=np.int8), np.zeros((30, 30), dtype=np.int8)
    _paint(raw, exchanged, (15, 15), 1)
    _paint(reflected, exchanged, (14, 14), 1, (-1, -1))
    assert np.array_equal(raw[::-1, ::-1], reflected)
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(4200+n*10+m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(92, 2, 1)
    broken["test"][0]["output"] = broken["test"][0]["output"].copy()
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "puzzles": len(signatures), "shapes": len(shapes),
            "glyphs": len(glyphs), "geometries": len(geometries),
            "palettes": len(palettes), "clip_counts": sorted(clip_counts),
            "ratios": [list(r) for r in sorted(ratios)]}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
