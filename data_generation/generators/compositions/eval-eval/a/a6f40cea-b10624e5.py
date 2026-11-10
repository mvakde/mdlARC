"""Mutual repair of scaled, reflected views of occluded rectangular loops.

No panel is a complete reference. The square peg is both a coordinate origin
and the unit of length. Screens hide complementary evidence, with at least one
segment hidden everywhere. Only screen interiors are revealed in the output.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _area(box, strict=False):
    t, b, l, r = box
    k = int(strict)
    return {(y, x) for y in range(t + k, b + 1 - k)
            for x in range(l + k, r + 1 - k)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _components(points):
    todo = set(points)
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _infer(observed, masks):
    """Unique loop cover of the registered evidence; no scene metadata."""
    common = set.intersection(*[_area(m) for m in masks])
    components = _components(observed)
    candidates = []
    for i, first in enumerate(components):
        for j in range(i, len(components)):
            visible = first | components[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b-t < 2 or r-l < 2:
                continue
            full = _perimeter(box)
            if full - common != visible:
                continue
            predictions = set()
            periods = []
            for period in range(1, 4):
                word, counts = {}, [0]*period
                for y, x in sorted(visible):
                    phase = (y+x) % period
                    color = observed[y, x]
                    if phase in word and word[phase] != color:
                        break
                    word[phase] = color
                    counts[phase] += 1
                else:
                    if min(counts) >= 2:
                        predictions.add(tuple(word[(y+x) % period] for y, x in sorted(full)))
                        periods.append(period)
            if len(predictions) != 1:
                continue
            paint = dict(zip(sorted(full), next(iter(predictions))))
            candidates.append((frozenset((i, j)), box, paint, min(periods)))
    covers = []

    def search(remaining, chosen, used):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        halo = {q for y, x in used for q in
                ((y, x), (y-1, x), (y+1, x), (y, x-1), (y, x+1))}
        for parts, box, paint, period in candidates:
            if first in parts and parts <= remaining and not (set(paint) & halo):
                search(remaining-parts, chosen+[(box, paint, period)], used | set(paint))

    search(set(range(len(components))), [], set())
    if len(covers) != 1 or not covers[0]:
        raise ValueError("registered evidence has no unique rectangular completion")
    return covers[0]


def _bounds(h, w, row, col):
    return [(0, row, 0, col), (0, row, col+1, w),
            (row+1, h, 0, col), (row+1, h, col+1, w)]


def _signs(panel):
    return (-1 if panel//2 else 1, -1 if panel%2 else 1)


def _orient_box(box, sy, sx):
    t, b, l, r = box
    return _bbox(((sy*t, sx*l), (sy*b, sx*r)))


def _screen_candidates(panel, bg):
    """A screen has a solid ring, uniform non-background fill, and a square peg.

    The peg side equals ring thickness. Loop interiors are background, so a
    visible loop cannot be mistaken for a screen.
    """
    candidates = []
    for frame in map(int, np.unique(panel)):
        if frame == bg:
            continue
        points = np.argwhere(panel == frame)
        t, l = points.min(0).tolist()
        b, r = points.max(0).tolist()
        for s in range(1, min((b-t+1)//3, (r-l+1)//3)+1):
            inside = panel[t+s:b-s+1, l+s:r-s+1]
            if not inside.size or not np.all(inside == inside.flat[0]):
                continue
            fill = int(inside.flat[0])
            if fill in (bg, frame):
                continue
            outer = panel[t:b+1, l:r+1]
            ring = np.ones(outer.shape, dtype=bool)
            ring[s:-s, s:-s] = False
            if len(points) != int(ring.sum()) or not np.all(outer[ring] == frame):
                continue
            if (b-t+1) % s or (r-l+1) % s:
                continue
            for anchor in map(int, np.unique(panel)):
                if anchor in (bg, frame, fill):
                    continue
                peg = np.argwhere(panel == anchor)
                at, al = peg.min(0).tolist()
                ab, ar = peg.max(0).tolist()
                if (ab-at+1, ar-al+1, len(peg)) != (s, s, s*s):
                    continue
                if any(v % s for v in (t-at, l-al)):
                    continue
                candidates.append((frame, fill, anchor, s, at, al, (t, b, l, r)))
    return candidates


def _decode(input_grid):
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("input must be an integer matrix")
    if not all(1 <= d <= 30 for d in grid.shape) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("input outside ARC bounds")
    bg = int(grid[0, 0])
    rows = [i for i in range(1, grid.shape[0]-1)
            if grid[i, 0] != bg and np.all(grid[i] == grid[i, 0])]
    cols = [j for j in range(1, grid.shape[1]-1)
            if grid[0, j] != bg and np.all(grid[:, j] == grid[0, j])]
    crosses = [(i, j) for i in rows for j in cols if grid[i, 0] == grid[0, j]]
    if len(crosses) != 1:
        raise ValueError("expected one cross divider")
    row, col = crosses[0]
    divider = int(grid[row, col])
    bounds = _bounds(*grid.shape, row, col)
    panels = [grid[t:b, l:r] for t, b, l, r in bounds]
    choices = [_screen_candidates(p, bg) for p in panels]
    solutions = []
    for group in itertools.product(*choices):
        if len({c[:3] for c in group}) == 1:
            solutions.append(group)
    if len(solutions) != 1:
        raise ValueError("screen and peg roles are not unique")
    group = solutions[0]
    frame, fill, anchor = group[0][:3]
    if len({bg, divider, frame, fill, anchor}) != 5:
        raise ValueError("role colors must be distinct")
    observed, masks, views = {}, [], []
    roles = {bg, divider, frame, fill, anchor}
    for q, (panel, candidate) in enumerate(zip(panels, group)):
        _, _, _, s, at, al, pixel_box = candidate
        sy, sx = _signs(q)
        t, b, l, r = pixel_box
        mask = _orient_box(((t-at)//s, (b-at+1)//s-1,
                            (l-al)//s, (r-al+1)//s-1), sy, sx)
        if (0, 0) in _area(mask):
            raise ValueError("screen occludes the peg")
        local = {}
        remaining = set(map(tuple, np.argwhere(~np.isin(panel, list(roles)))))
        while remaining:
            y, x = min(remaining)
            uy, ux = (int(y)-at)//s, (int(x)-al)//s
            py, px = at+uy*s, al+ux*s
            tile = panel[py:py+s, px:px+s]
            color = int(panel[y, x])
            if tile.shape != (s, s) or not np.all(tile == color):
                raise ValueError("visible geometry is not on the peg lattice")
            cell = (sy*uy, sx*ux)
            if cell in observed and observed[cell] != color:
                raise ValueError("views disagree after registration")
            observed[cell] = local[cell] = color
            remaining -= {(py+i, px+j) for i in range(s) for j in range(s)}
        masks.append(mask)
        views.append({"side": s, "anchor": (at, al), "mask": mask,
                      "pixel_box": pixel_box, "visible": local})
    loops = _infer(observed, masks)
    complete = {p: color for _, paint, _ in loops for p, color in paint.items()}
    # Check also the observed negative evidence: no absent or extra visible cells.
    for q, (view, panel) in enumerate(zip(views, panels)):
        expected = {p: c for p, c in complete.items() if p not in _area(view["mask"])}
        if view["visible"] != expected:
            raise ValueError("a view contradicts the common loop template")
        sy, sx = _signs(q)
        s, (at, al) = view["side"], view["anchor"]
        for y, x in complete:
            py, px = at+sy*y*s, al+sx*x*s
            if not (0 <= py and py+s <= panel.shape[0] and 0 <= px and px+s <= panel.shape[1]):
                raise ValueError("underlying loop would leave its panel")
    return {"background": bg, "divider": divider, "frame": frame, "fill": fill,
            "anchor_color": anchor, "bounds": bounds, "views": views,
            "loops": loops, "complete": complete, "observed": observed}


def _reveal(grid, data, paint):
    out = np.asarray(grid).copy()
    for q, view in enumerate(data["views"]):
        sy, sx = _signs(q)
        s, (at, al) = view["side"], view["anchor"]
        t, _, l, _ = data["bounds"][q]
        interior = _area(view["mask"], strict=True)
        for (y, x), color in paint.items():
            if (y, x) in interior:
                py, px = t+at+sy*y*s, l+al+sx*x*s
                out[py:py+s, px:px+s] = color
    return out


def _registration_ablation(grid, data, omit):
    """Concrete alternatives: omit an axis flip, or reuse TL's pixel scale."""
    out = np.asarray(grid).copy()
    for q, view in enumerate(data["views"]):
        sy, sx = _signs(q)
        sy = 1 if omit == "vertical" else sy
        sx = 1 if omit == "horizontal" else sx
        s, (at, al) = view["side"], view["anchor"]
        unit = data["views"][0]["side"] if omit == "scale" else s
        t, _, l, _ = data["bounds"][q]
        mt, mb, ml, mr = view["pixel_box"]
        for (y, x), color in data["complete"].items():
            yy = t+at+(y*unit if sy == 1 else s-(y+1)*unit)
            xx = l+al+(x*unit if sx == 1 else s-(x+1)*unit)
            y0, y1 = max(t+mt+s, yy), min(t+mb-s+1, yy+unit)
            x0, x1 = max(l+ml+s, xx), min(l+mr-s+1, xx+unit)
            if y0 < y1 and x0 < x1:
                out[y0:y1, x0:x1] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reveal hidden loop pieces; the pixels alone determine all conventions."""
    data = _decode(input_grid)
    return _reveal(input_grid, data, data["complete"])


def _geometry(rng, count, nested=False):
    if count == 1:
        h, w = (int(rng.integers(5, 9)) for _ in range(2))
        t, l = int(rng.integers(h-3)), int(rng.integers(w-3))
        b, r = int(rng.integers(t+2, h)), int(rng.integers(l+2, w))
        boxes = [(t, b, l, r)]
    elif nested:
        h, w = int(rng.integers(7, 9)), int(rng.integers(7, 9))
        t, l = int(rng.integers(h-6)), int(rng.integers(w-6))
        boxes = [(t, t+6, l, l+6), (t+2, t+4, l+2, l+4)]
    else:
        h = int(rng.integers(5 if count == 2 else 7, 9))
        w = int(rng.integers(7, 9))
        widths = [3, int(rng.integers(3, w-3))]
        lefts = [0, w-widths[1]]
        if count == 2:
            boxes = []
            for left, width in zip(lefts, widths):
                height = int(rng.integers(3, h+1))
                top = int(rng.integers(h-height+1))
                boxes.append((top, top+height-1, left, left+width-1))
        else:
            cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
            omitted = int(rng.integers(4))
            boxes = [(0 if y == 0 else h-3, 2 if y == 0 else h-1,
                      lefts[x], lefts[x]+widths[x]-1)
                     for i, (y, x) in enumerate(cells) if i != omitted]
    # Transpose varies wide/tall arrangements without adding a hidden convention.
    if rng.random() < .5:
        h, w = w, h
        boxes = [(l, r, t, b) for t, b, l, r in boxes]
    occupied = set.union(*[_perimeter(b) for b in boxes])
    free = sorted(_area((0, h-1, 0, w-1))-occupied)
    anchor = free[int(rng.integers(len(free)))]
    pivots = sorted(p for p in occupied if 0 < p[0] < h-1 and 0 < p[1] < w-1)
    if not pivots:
        return None
    pivot = pivots[int(rng.integers(len(pivots)))]
    options = []
    for t in range(h-2):
        for b in range(t+2, h):
            for l in range(w-2):
                for r in range(l+2, w):
                    box = (t, b, l, r)
                    if pivot in _area(box, True) and anchor not in _area(box):
                        options.append(box)
    full_options = [m for m in options if any(_perimeter(b) <= _area(m)
                                             and _perimeter(b) & _area(m, True) for b in boxes)]
    if not full_options:
        return None
    masks = [full_options[int(rng.integers(len(full_options)))]]
    masks += [options[int(rng.integers(len(options)))] for _ in range(3)]
    rng.shuffle(masks)
    if len(set(masks)) < 2:
        return None
    return boxes, masks, anchor


def _make_pair(rng, count, kind):
    """Bounded sampling; render truth from loops before applying screens."""
    for _ in range(1200):
        scene = _geometry(rng, count, nested=count == 2 and rng.random() < .35)
        if scene is None:
            continue
        boxes, masks, (ay, ax) = scene
        # All unit coordinates are translated to the peg, then optionally turned.
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))

        def orient(p):
            y, x = p[0]-ay, p[1]-ax
            for _ in range(turns):
                y, x = x, -y
            return y, -x if flip else x

        boxes = [_bbox(orient(p) for p in _perimeter(b)) for b in boxes]
        masks = [_bbox(orient(p) for p in _perimeter(b)) for b in masks]
        palette = list(map(int, rng.permutation(10)))
        bg, divider, frame, fill, anchor = palette[:5]
        detail = palette[5:]
        motif = {}
        for i, box in enumerate(boxes):
            period = 2 if kind == 0 else (3 if kind == 1 and i == 0 else
                                         1 if kind == 1 else int(rng.integers(1, 4)))
            word = list(map(int, rng.choice(detail, size=period, replace=False)))
            if period == 3 and rng.random() < .25 and kind != 1:
                word[2] = word[0]
            phase = int(rng.integers(period))
            motif.update({(y, x): word[(y+x+phase) % period] for y, x in _perimeter(box)})
        common = set.intersection(*[_area(m) for m in masks])
        observed = {p: c for p, c in motif.items() if p not in common}
        try:
            loops = _infer(observed, masks)
        except ValueError:
            continue
        if len(loops) != count or {p: c for _, paint, _ in loops for p, c in paint.items()} != motif:
            continue
        # Registration always matters, closure always matters, and no view is complete.
        if not any(set(paint) <= _area(mask) and set(paint) & _area(mask, True)
                   for _, paint, _ in loops for mask in masks):
            continue
        footprints = []
        for q in range(4):
            sy, sx = _signs(q)
            points = set(motif) | _area(masks[q]) | {(0, 0)}
            footprints.append(_bbox((sy*y, sx*x) for y, x in points))
        max_span = max(max(b-t+1, r-l+1) for t, b, l, r in footprints)
        scales = [1, 1, 1, 2 if max_span > 6 or rng.random() < .75 else 3]
        if kind > 1 and rng.random() < .15:
            scales = [1]*4
        rng.shuffle(scales)
        minimum_rows, minimum_cols = [0, 0], [0, 0]
        for q, ((t, b, l, r), s) in enumerate(zip(footprints, scales)):
            minimum_rows[q//2] = max(minimum_rows[q//2], (b-t+1)*s+2)
            minimum_cols[q%2] = max(minimum_cols[q%2], (r-l+1)*s+2)
        if max(sum(minimum_rows), sum(minimum_cols)) + 1 > 30:
            continue
        heights, widths = list(minimum_rows), list(minimum_cols)
        for lengths in (heights, widths):
            extra = int(rng.integers(30-sum(lengths)))
            first = int(rng.integers(extra+1))
            lengths[0] += first
            lengths[1] += extra-first
        h, w = sum(heights)+1, sum(widths)+1
        bounds = _bounds(h, w, heights[0], widths[0])
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[heights[0], :] = divider
        inp[:, widths[0]] = divider
        out = inp.copy()
        for q, (t, b, l, r) in enumerate(bounds):
            sy, sx = _signs(q)
            s = scales[q]
            ft, fb, fl, fr = footprints[q]
            py = t+int(rng.integers(1, b-t-(fb-ft+1)*s))-ft*s
            px = l+int(rng.integers(1, r-l-(fr-fl+1)*s))-fl*s
            for grid in (inp, out):
                grid[py:py+s, px:px+s] = anchor
                for (y, x), color in motif.items():
                    yy, xx = py+sy*y*s, px+sx*x*s
                    grid[yy:yy+s, xx:xx+s] = color
                for y, x in _area(masks[q]):
                    yy, xx = py+sy*y*s, px+sx*x*s
                    grid[yy:yy+s, xx:xx+s] = frame if (y, x) in _perimeter(masks[q]) else fill
            for (y, x), color in motif.items():
                if (y, x) in _area(masks[q], True):
                    yy, xx = py+sy*y*s, px+sx*x*s
                    out[yy:yy+s, xx:xx+s] = color
        try:
            data = _decode(inp)
        except ValueError:
            continue
        if np.array_equal(_reveal(inp, data, data["complete"]), out):
            return {"input": inp, "output": out}
    raise RuntimeError("bounded scene search exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("counts require 2..10 train, 1..4 test, train > test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            nloops = 3 if split == "test" else 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
            kind = i if split == "train" and i < 2 else 2
            for _ in range(8):
                pair = _make_pair(rng, nloops, kind)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    data = _decode(inp)
    expected = _reveal(inp, data, data["complete"])
    assert inp.shape == out.shape and np.array_equal(expected, out), "incorrect repair"
    assert not np.array_equal(inp, out), "empty repair"
    # A ablation: even pooling every visible fragment cannot supply these colors.
    pooled_only = _reveal(inp, data, data["observed"])
    assert not np.array_equal(pooled_only, out), "rectangle closure is unnecessary"
    transfer = 0
    changed = inp != out
    allowed = np.zeros(inp.shape, dtype=bool)
    for q, view in enumerate(data["views"]):
        t, _, l, _ = data["bounds"][q]
        mt, mb, ml, mr = view["pixel_box"]
        s = view["side"]
        allowed[t+mt+s:t+mb-s+1, l+ml+s:l+mr-s+1] = True
        interior = _area(view["mask"], True)
        assert set(data["complete"]) & interior, "a screen hides no repairable content"
        for _, paint, _ in data["loops"]:
            if set(paint) <= _area(view["mask"]) and set(paint) & interior:
                transfer += 1  # This panel has no evidence that this loop exists.
    assert not np.any(changed & ~allowed), "visible evidence or screen border changed"
    assert transfer, "no witness requiring another view"
    data["transfer_witnesses"] = transfer
    return data


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= d <= 30 for d in a.shape) and np.all((0 <= a) & (a <= 9))
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(_analyze(pair))
    train = analyses["train"]
    assert len(train[0]["loops"]) == 1 and len(train[1]["loops"]) == 2
    assert any(p == 2 for _, _, p in train[0]["loops"])
    assert any(p == 3 for _, _, p in train[1]["loops"])
    assert all(len({v["side"] for v in a["views"]}) > 1 for a in train[:2])
    assert all(len(a["loops"]) <= 2 for a in train)
    assert all(len(a["loops"]) == 3 for a in analyses["test"])
    for omission in ("horizontal", "vertical", "scale"):
        assert any(not np.array_equal(_registration_ablation(p["input"], a, omission), p["output"])
                   for p, a in zip(puzzle["train"], train)), f"unexposed {omission} convention"


def self_test():
    """42 seeds, no stored puzzles, bounded summaries and explicit ablations."""
    puzzles, shapes, structures, layouts, palettes, periods, scales = (set() for _ in range(7))
    nested = transfer = shared_hidden = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _analyze(pair)
                inp = np.asarray(pair["input"])
                signature.update(repr(_key(inp)).encode())
                shapes.add(inp.shape)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add(hashlib.sha256(normalized).digest())
                layouts.add(tuple(b for b, _, _ in data["loops"]))
                palettes.add((data["background"], data["frame"], data["fill"], data["anchor_color"]))
                periods.update(p for _, _, p in data["loops"])
                scales.update(v["side"] for v in data["views"])
                transfer += data["transfer_witnesses"]
                shared_hidden += len(set(data["complete"])-set(data["observed"]))
                boxes = [b for b, _, _ in data["loops"]]
                nested += any(t < a < b < z and l < c < d < r
                              for t, z, l, r in boxes for a, b, c, d in boxes)
        puzzles.add(signature.digest())
        if seed < 3:
            repeat = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], repeat[s]) for k in a)
        if seed == 0:
            # These changed inputs have no generator-side metadata. Global
            # symmetries and color bijections must commute with reconstruction.
            for pair in (puzzle["train"][0], puzzle["train"][1], puzzle["test"][0]):
                for turns in range(4):
                    for mirror in (False, True):
                        inp, out = (np.rot90(pair[k], turns) for k in ("input", "output"))
                        if mirror:
                            inp, out = inp[:, ::-1], out[:, ::-1]
                        assert np.array_equal(solve(inp), out), "global symmetry failure"
            recolor = np.array([7, 3, 9, 0, 6, 2, 8, 1, 5, 4], dtype=np.int8)
            pair = puzzle["test"][0]
            assert np.array_equal(solve(recolor[pair["input"]].tolist()), recolor[pair["output"]])
    assert len(puzzles) == 42 and len(structures) >= 200
    assert len(shapes) >= 35 and len(layouts) >= 60 and len(palettes) >= 100
    assert periods == {1, 2, 3} and scales >= {1, 2, 3}
    assert nested > 0 and transfer >= 210 and shared_hidden >= 210
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(100+nt*5+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(77, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "loop_layouts": len(layouts), "role_palettes": len(palettes),
            "periods": sorted(periods), "peg_sides": sorted(scales),
            "nested_pairs": nested, "transfer_witnesses": transfer,
            "unseen_loop_cells": shared_hidden}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
