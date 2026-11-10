"""Registered double exposures: translation, coincidence, and diagonal filling.

Only Python's standard library and NumPy are used.  The solver reads the two
plates from pixels; it has no access to the sampler or a puzzle's other pairs.
"""
from __future__ import annotations

import json
import numbers

import numpy as np


HIGHLIGHT = 2
_ALTERNATIVES = (
    "no_fill", "all_ink", "union_support", "pixel_centers", "consecutive",
    "infinite", "separate_fill", "frame_align", "seam_plus", "seam_minus",
    "no_registration", "xor",
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer rectangles, 1..30, colors 0..9")
    return a


def _key(a):
    return (a.shape, a.tobytes())


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in left:
                    left.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc):
    return {(r+dr, c+dc) for r, c in cells}


def _perimeter(box):
    top, bottom, left, right = box
    return ({(r, c) for r in (top, bottom) for c in range(left, right+1)}
            | {(r, c) for r in range(top, bottom+1) for c in (left, right)})


def _inside(p, box):
    r, c = p
    top, bottom, left, right = box
    return top < r < bottom and left < c < right


def _runs(cells, slope):
    """Maximal actual diagonal runs, in the B parent's q/t coordinates."""
    lines = {}
    for r, c in cells:
        lines.setdefault(c-slope*r, []).append(r)
    records = []
    for q in sorted(lines):
        rows = sorted(lines[q])
        lo = previous = rows[0]
        for r in rows[1:] + [None]:
            if r is not None and r == previous+1:
                previous = r
                continue
            records.append((q, 2*lo+slope*q, 2*previous+slope*q))
            if r is not None:
                lo = previous = r
    return records


def _parse(input_grid):
    """Recover all roles, ownership, slope and registration from the input."""
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    if len(colors) != 4 or HIGHLIGHT in colors or counts.max() <= g.size/2:
        raise ValueError("four input roles, no red, and a strict majority field required")
    bg = int(colors[np.argmax(counts)])
    by_color = {int(c): {tuple(map(int, p)) for p in np.argwhere(g == c)}
                for c in colors if c != bg}
    choices = []
    for frame_color, frame_cells in by_color.items():
        frames = _components(frame_cells)
        if len(frames) != 2:
            continue
        boxes = [_bbox(f) for f in frames]
        if any(b[1]-b[0] < 4 or b[3]-b[2] < 4 or f != _perimeter(b)
               for f, b in zip(frames, boxes)):
            continue
        # Disjoint bounding rectangles, with at least one blank cell gap.
        a, b = boxes
        if not (a[1]+1 < b[0] or b[1]+1 < a[0]
                or a[3]+1 < b[2] or b[3]+1 < a[2]):
            continue
        for marker_color in by_color:
            if marker_color == frame_color:
                continue
            ink_color = next(c for c in by_color if c not in (frame_color, marker_color))
            bars = _components(by_color[marker_color])
            if len(bars) != 2 or len(bars[0]) != len(bars[1]) or len(bars[0]) < 2:
                continue
            axis = None
            if all(len({r for r, c in bar}) == 1 for bar in bars):
                axis = 0
            elif all(len({c for r, c in bar}) == 1 for bar in bars):
                axis = 1
            if axis is None:
                continue
            # A four-connected straight component has no holes in its bar.
            owned_bars, inks, valid = [], [], True
            for box in boxes:
                own = [bar for bar in bars if all(_inside(p, box) for p in bar)]
                if len(own) != 1:
                    valid = False
                    break
                owned_bars.append(own[0])
                inks.append({p for p in by_color[ink_color] if _inside(p, box)})
            if not valid or not all(inks) or set.union(*inks) != by_color[ink_color]:
                continue
            for slope in (-1, 1):
                tracks = [_runs(ink, slope) for ink in inks]
                if not all(len(t) >= 2 and all(hi-lo >= 2 for q, lo, hi in t)
                           for t in tracks):
                    continue
                origins = [min(bar) for bar in owned_bars]
                relative = [_shift(ink, -p[0], -p[1]) for ink, p in zip(inks, origins)]
                bound = _bbox(set.union(*relative))
                if bound[1]-bound[0]+1 > 30 or bound[3]-bound[2]+1 > 30:
                    continue
                choices.append(dict(background=bg, frame=frame_color, marker=marker_color,
                                    ink=ink_color, slope=slope, axis=axis,
                                    length=len(bars[0]), boxes=boxes, origins=origins,
                                    raw_inks=inks, layers=relative))
    if len(choices) != 1:
        raise ValueError("plates must identify exactly one role assignment and diagonal slope")
    return choices[0]


def _fill_sweep(strokes, slope, bounds, variant=""):
    """Solver implementation: active extreme boundaries on each scanline."""
    top, bottom, left, right = bounds
    fill = set()
    for r in range(top, bottom+1):
        for c in range(left, right+1):
            q, t = c-slope*r, r+slope*c
            active = [x for x, lo, hi in strokes
                      if (variant == "infinite" or lo <= t <= hi)
                      and (variant != "pixel_centers" or (t-x) % 2 == 0)]
            if active and min(active) < q < max(active):
                fill.add((r, c))
    return fill


def _fill_pairwise(strokes, slope, bounds, consecutive=False):
    """Label/check implementation: union pairwise interval parallelograms."""
    top, bottom, left, right = bounds
    rows, cols = np.mgrid[top:bottom+1, left:right+1]
    q_grid, t_grid = cols-slope*rows, rows+slope*cols
    filled = np.zeros(rows.shape, dtype=bool)
    ranks = sorted({q for q, lo, hi in strokes})
    for i, (q1, lo1, hi1) in enumerate(strokes):
        for q2, lo2, hi2 in strokes[i+1:]:
            if q1 == q2 or (consecutive and abs(ranks.index(q1)-ranks.index(q2)) != 1):
                continue
            filled |= ((t_grid >= max(lo1, lo2)) & (t_grid <= min(hi1, hi2))
                       & (q_grid > min(q1, q2)) & (q_grid < max(q1, q2)))
    return {(int(rows[r, c]), int(cols[r, c])) for r, c in np.argwhere(filled)}


def _render(ink, fill, bg, color):
    # Fill cannot enlarge the ink bbox: every bracket lies in the convex hull
    # of two eligible diagonal segments.  Ink takes precedence over highlight.
    top, bottom, left, right = _bbox(ink)
    out = np.full((bottom-top+1, right-left+1), bg, dtype=np.int8)
    for r, c in fill-ink:
        if top <= r <= bottom and left <= c <= right:
            out[r-top, c-left] = HIGHLIGHT
    for r, c in ink:
        out[r-top, c-left] = color
    return out


def _prediction(info, variant="", pairwise=False):
    a, b = info["layers"]
    slope = info["slope"]
    if variant == "frame_align":
        a, b = [_shift(ink, -box[0], -box[2])
                for ink, box in zip(info["raw_inks"], info["boxes"])]
    elif variant in ("seam_plus", "seam_minus"):
        step = 1 if variant == "seam_plus" else -1
        b = _shift(b, step if info["axis"] == 0 else 0,
                   step if info["axis"] == 1 else 0)
    elif variant == "no_registration":
        a, b = info["raw_inks"]
    union, common = a | b, a & b
    bounds = _bbox(union)
    strokes = _runs(common, slope)
    if variant == "all_ink":
        strokes = _runs(union, slope)
    elif variant == "union_support":
        # A tempting weaker rule: a track seen on both plates is eligible
        # for its entire combined endpoint interval rather than its overlap.
        a_runs, b_runs = _runs(a, slope), _runs(b, slope)
        shared_q = {q for q, lo, hi in a_runs} & {q for q, lo, hi in b_runs}
        strokes = [(q, min(lo for x, lo, hi in a_runs+b_runs if x == q),
                    max(hi for x, lo, hi in a_runs+b_runs if x == q)) for q in shared_q]
    if variant == "no_fill":
        fill = set()
    elif variant == "separate_fill":
        fill = (_fill_pairwise(_runs(a, slope), slope, bounds)
                | _fill_pairwise(_runs(b, slope), slope, bounds))
    elif variant == "consecutive":
        fill = _fill_pairwise(strokes, slope, bounds, consecutive=True)
    elif pairwise:
        fill = _fill_pairwise(strokes, slope, bounds)
    else:
        fill = _fill_sweep(strokes, slope, bounds, variant)
    if variant == "xor":
        # Preserve the output coordinate window while cancelling coincidences.
        out = _render(union, fill, info["background"], info["ink"])
        top, _, left, _ = bounds
        for r, c in common:
            out[r-top, c-left] = info["background"]
        return out
    return _render(union, fill, info["background"], info["ink"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Align visible guides; use the shared ink to bracket continuous scanlines."""
    return _prediction(_parse(input_grid))


def _different(info, expected, alternative):
    return not np.array_equal(_prediction(info, alternative), expected)


def _cells(q, lo, hi, slope):
    assert lo <= hi and (lo-q) % 2 == (hi-q) % 2 == 0
    return {((t-slope*q)//2, (q+slope*t)//2) for t in range(lo, hi+1, 2)}


def _geometry(rng, count, slope, witness, support):
    """Two related stroke masks, never the output of a parent generator."""
    gaps = rng.integers(2, 5, size=count-1)
    qs = [0] + list(map(int, np.cumsum(gaps)))
    intervals = []
    mode = int(rng.integers(4))
    for q in qs:
        length = int(rng.integers(2, 6 if count <= 3 else 5))
        if mode == 0:
            lo = q % 2
        elif mode == 1:
            lo = -2*(length-1) + q % 2
        else:
            lo = 2*int(rng.integers(-2, 3)) + q % 2
        intervals.append((lo, lo+2*(length-1)))
    if witness:
        assert count == 3
        intervals = [(-4+q % 2, 6+q % 2) for q in qs]
        intervals[1] = (qs[1] % 2, 2+qs[1] % 2)
    a, b = set(), set()
    for index, (q, (lo, hi)) in enumerate(zip(qs, intervals)):
        start_owner, end_owner = int(rng.integers(2)), int(rng.integers(2))
        pre = int(rng.integers(support+1))
        post = int(rng.integers(support+1))
        if support == 3 and index == 0:
            # A visible test shift: a matched track has three additional ink
            # cells beyond each end on one of the two plates.
            pre = post = 3
        if witness:
            start_owner, end_owner, pre, post = 1, 0, 1, 1
        starts, ends = [lo, lo], [hi, hi]
        starts[start_owner] -= 2*pre
        ends[end_owner] += 2*post
        a |= _cells(q, starts[0], ends[0], slope)
        b |= _cells(q, starts[1], ends[1], slope)
    # At least one outside stroke belongs to only one plate. More can fall
    # inside the transverse range; they still cannot become boundaries.
    n_extra = 1 if witness else int(rng.integers(1, 4))
    occupied_q = set(qs)
    for k in range(n_extra):
        if k == 0:
            q = qs[-1]+int(rng.integers(2, 4)) if rng.integers(2) else -int(rng.integers(2, 4))
        else:
            available = [q for q in range(-3, qs[-1]+4) if q not in occupied_q]
            if not available:
                break
            q = int(rng.choice(available))
        occupied_q.add(q)
        lo = 2*int(rng.integers(-1, 2)) + q % 2
        hi = lo+2*int(rng.integers(1, 4))
        (a if rng.integers(2) else b).update(_cells(q, lo, hi, slope))
    # Occasionally put a second separated run on an existing track. The input
    # parser and sweep support repeated q coordinates and singleton overlaps.
    if not witness and rng.random() < .20:
        q = int(rng.choice(qs))
        endpoint = max(r+slope*c for r, c in a | b if c-slope*r == q)
        extra = _cells(q, endpoint+4, endpoint+6, slope)
        choice = int(rng.integers(3))
        if choice == 0:
            a |= extra
            b |= extra
        elif choice == 1:
            a |= extra
        else:
            a |= extra
            b |= _cells(q, endpoint+6, endpoint+8, slope)
    return a, b


def _plate(rng, ink, guide, colors):
    bg, carrier, marker, color = colors
    top, bottom, left, right = _bbox(ink | guide)
    pads = [int(x) for x in rng.integers(0, 2, size=4)]
    top -= 1+pads[0]
    bottom += 1+pads[1]
    left -= 1+pads[2]
    right += 1+pads[3]
    if bottom-top < 4:
        bottom = top+4
    if right-left < 4:
        right = left+4
    shape = (bottom-top+1, right-left+1)
    g = np.full(shape, bg, dtype=np.int8)
    for r, c in _perimeter((0, shape[0]-1, 0, shape[1]-1)):
        g[r, c] = carrier
    for cells, role in ((ink, color), (guide, marker)):
        for r, c in cells:
            g[r-top, c-left] = role
    return g


def _stage(rng, plates, bg):
    # Either plate can come first; staging conveys no assembly order.
    if rng.integers(2):
        plates = plates[::-1]
    a, b = plates
    possible = []
    for axis in (0, 1):
        if a.shape[axis]+b.shape[axis]+1 <= 30 and max(a.shape[1-axis], b.shape[1-axis]) <= 30:
            possible.append(axis)
    if not possible:
        return None
    axis = int(rng.choice(possible))
    if axis == 1:
        a, b = a.T, b.T
    gap = int(rng.integers(1, min(4, 30-a.shape[0]-b.shape[0])+1))
    width = max(a.shape[1], b.shape[1])
    jitter = min(3, 30-width)
    width += int(rng.integers(jitter+1))
    base = np.full((a.shape[0]+gap+b.shape[0], width), bg, dtype=np.int8)
    x1, x2 = int(rng.integers(width-a.shape[1]+1)), int(rng.integers(width-b.shape[1]+1))
    base[:a.shape[0], x1:x1+a.shape[1]] = a
    start = a.shape[0]+gap
    base[start:start+b.shape[0], x2:x2+b.shape[1]] = b
    if axis == 1:
        base = base.T
    pads = []
    for span in base.shape:
        extra = min(30-span, int(rng.integers(0, 5)))
        before = int(rng.integers(extra+1))
        pads.append((before, extra-before))
    return np.pad(base, pads, constant_values=bg)


def _make_pair(rng, colors, count, length, rotation, support, witness, seen):
    bg, carrier, marker, color = colors
    for attempt in range(320):
        slope = int(rng.choice([-1, 1]))
        a, b = _geometry(rng, count, slope, witness, support)
        top, bottom, left, right = _bbox(a | b)
        # Registration may be above/below, beside, or within the payload's bbox.
        options = [(r, c) for r in range(top-1, bottom+2)
                   for c in range(left-1, right-length+3)
                   if all((r, c+k) not in a | b for k in range(length))]
        if not options:
            continue
        r, c = options[int(rng.integers(len(options)))]
        guide = {(r, c+k) for k in range(length)}
        plates = [_plate(rng, ink, guide, colors) for ink in (a, b)]
        g = _stage(rng, plates, bg)
        if g is None:
            continue
        g = np.rot90(g, rotation).copy()
        if _key(g) in seen:
            continue
        try:
            info = _parse(g)
        except ValueError:
            continue
        out = _prediction(info, pairwise=True)
        if np.count_nonzero(out == HIGHLIGHT) == 0:
            continue
        if witness and not all(_different(info, out, v) for v in _ALTERNATIVES):
            continue
        # Labels also check the assembly made directly from the pre-staging
        # masks; matching the parser cannot silently change plate ownership.
        direct = _render(a | b, _fill_pairwise(_runs(a & b, slope), slope, _bbox(a | b)), bg, color)
        if not np.array_equal(np.rot90(direct, rotation), out):
            raise AssertionError("staging changed registration geometry")
        seen.add(_key(g))
        return {"input": g, "output": out}
    raise RuntimeError("bounded plate construction exhausted")


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral)
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    ntrain = _integer("num_train", num_train, 2, 10)
    ntest = _integer("num_test", num_test, 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = [c for c in range(10) if c != HIGHLIGHT]
    stable = tuple(map(int, rng.choice(palette, 4, replace=False)))
    changing = bool(rng.integers(2))
    mixed = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    shift = str(rng.choice(["count", "support", "key"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", ntrain), ("test", ntest)):
        for i in range(number):
            witness = split == "train" and i == 1
            count = int(rng.integers(2, 5))
            if shift == "count":
                count = int(rng.integers(2, 4) if split == "train" else rng.integers(4, 6))
            if split == "train" and i < 2:
                count = 2+i
            length = int(rng.integers(2, 5))
            if shift == "key":
                length = (2 if i % 2 == 0 else 4) if split == "train" else 3
            support = (3 if split == "test" else 1) if shift == "support" else 2
            colors = tuple(map(int, rng.choice(palette, 4, replace=False))) if changing else stable
            angle = (rotation+i % 2) % 4 if mixed and split == "train" else rotation
            if mixed and split == "test":
                angle = (rotation+int(rng.integers(2))) % 4
            puzzle[split].append(_make_pair(rng, colors, count, length, angle, support, witness, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate actual pixels and demonstrated alternatives, without metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    differences = {s: set() for s in puzzle}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            # Normalize dtype for duplicate detection, including list callers.
            signature = (g.shape, tuple(map(int, g.flat)))
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            info = _parse(g)
            expected = _prediction(info)
            assert np.array_equal(out, expected), "incorrect registration or developed region"
            assert np.array_equal(out, _prediction(info, pairwise=True)), "sweep/polygon disagreement"
            a, b = info["layers"]
            assert np.sum(out == info["ink"]) == len(a)+len(b)-len(a & b), "ink union not preserved"
            assert np.any(out == HIGHLIGHT), "must visibly demonstrate development"
            assert set(map(int, np.unique(out))) <= {info["background"], info["ink"], HIGHLIGHT}
            assert all(np.any(edge == info["ink"]) for edge in (out[0], out[-1], out[:, 0], out[:, -1])), "crop not tight"
            assert len({q for q, lo, hi in _runs(a & b, info["slope"])}) >= 2
            infos[split].append(info)
            differences[split].update(v for v in _ALTERNATIVES if _different(info, out, v))
    assert differences["test"] <= differences["train"], "test uses an untaught alternative-rule distinction"
    assert {"no_fill", "all_ink", "no_registration", "consecutive"} <= differences["train"], "both parents and inactive middle boundaries must be taught"


def _features(pair):
    info = _parse(pair["input"])
    a, b = info["layers"]
    common = _runs(a & b, info["slope"])
    q0 = min(q for q, lo, hi in common)
    t0 = min(lo for q, lo, hi in common)
    layout = tuple((q-q0, lo-t0, hi-t0) for q, lo, hi in common)
    return dict(shape=pair["input"].shape, output_shape=pair["output"].shape,
                colors=(info["background"], info["frame"], info["marker"], info["ink"]),
                slope=info["slope"], axis=info["axis"], bar=info["length"],
                shared_runs=len(common), source_runs=(len(_runs(a, info["slope"])), len(_runs(b, info["slope"]))),
                layout=layout, shared_cells=len(a & b), solitary_cells=len(a ^ b),
                filled=int(np.sum(pair["output"] == HIGHLIGHT)))


def self_test():
    """Bounded 42-seed survey, analytical fixture, negative semantic checks."""
    # Hand-worked canonical case: common q=0,4, t=0..4, slope +1.
    # At t=1/3 the continuous boundary intersects between ink pixel centers.
    common = {(0, 0), (1, 1), (2, 2), (-2, 2), (-1, 3), (0, 4)}
    expected_fill = {(-1, 2), (0, 1), (0, 2), (0, 3), (1, 2)}
    # The bbox also includes two cells on t=0 and t=4 at q=2.
    expected_fill |= {(-1, 1), (1, 3)}
    runs = _runs(common, 1)
    assert _fill_sweep(runs, 1, _bbox(common)) == expected_fill
    assert _fill_pairwise(runs, 1, _bbox(common)) == expected_fill
    assert _fill_sweep(runs, 1, _bbox(common), "pixel_centers") != expected_fill
    shapes, outputs, layouts, palettes, fingerprints = set(), set(), set(), set(), set()
    counts, axes, slopes, bars, negative_checks = set(), set(), set(), set(), set()
    changing_puzzles = mixed_puzzles = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint = tuple((_key(x["input"]), _key(x["output"]))
                            for s in ("train", "test") for x in p[s])
        fingerprints.add(fingerprint)
        local_palettes, local_axes = set(), set()
        for split in p:
            for pair in p[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                outputs.add(f["output_shape"])
                layouts.add(f["layout"])
                palettes.add(f["colors"])
                counts.add(f["shared_runs"])
                axes.add(f["axis"])
                slopes.add(f["slope"])
                bars.add(f["bar"])
                local_palettes.add(f["colors"])
                local_axes.add(f["axis"])
                g_copy = pair["input"].copy()
                assert np.array_equal(solve(g_copy), pair["output"])
                assert np.array_equal(g_copy, pair["input"]), "solver mutated input"
        changing_puzzles += len(local_palettes) > 1
        mixed_puzzles += len(local_axes) > 1
        # Every targeted wrong answer is rejected by the semantic validator.
        witness = p["train"][1]
        info = _parse(witness["input"])
        for v in _ALTERNATIVES:
            if v in negative_checks:
                continue
            wrong = _prediction(info, v)
            if max(wrong.shape) > 30 or np.array_equal(wrong, witness["output"]):
                continue
            bad = {"train": list(p["train"]), "test": list(p["test"])}
            bad["train"][1] = {"input": witness["input"], "output": wrong}
            try:
                validate_generated_puzzle(bad)
            except (AssertionError, ValueError):
                negative_checks.add(v)
            else:
                raise AssertionError(f"accepted {v}")
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(outputs) >= 30 and len(layouts) >= 120
    assert len(palettes) >= 35 and counts >= {2, 3, 4, 5}
    assert axes == {0, 1} and slopes == {-1, 1} and bars == {2, 3, 4}
    assert changing_puzzles >= 8 and mixed_puzzles >= 8
    assert negative_checks == set(_ALTERNATIVES)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(127+a*10+b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, again = generate_puzzle(927, 2, 1), generate_puzzle(927, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], again[s])
               for k in ("input", "output"))
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "output_shapes": len(outputs), "consensus_layouts": len(layouts), "palettes": len(palettes),
            "shared_run_counts": sorted(counts), "alternative_checks": len(negative_checks)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
