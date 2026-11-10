"""Register framed picture fragments, then bisect the assembled pad gap.

Frames are packaging. Matching multicolor components are translation keys.
All conventions needed by solve are visible in its one input argument.
No files, parent modules, seed tables, or generated examples are read at runtime.
"""
from __future__ import annotations

import itertools
import numbers

import numpy as np


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _norm(cells):
    r0, _, c0, _ = _bounds(cells)
    return {(r - r0, c - c0): int(v) for (r, c), v in cells.items()}


def _shape(cells):
    r0, r1, c0, c1 = _bounds(cells)
    return r1 - r0 + 1, c1 - c0 + 1


def _pattern(cells):
    return tuple((r, c, v) for (r, c), v in sorted(_norm(cells).items()))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _decode(input_grid):
    """Find the unique frame color and unpack its nonnested rectangles."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[counts.argmax()])
    if counts.max() * 2 <= a.size:
        raise ValueError("input background must occupy more than half the canvas")
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("an exterior background margin is required")
    choices = []
    for color in (int(x) for x in colors if x != bg):
        boxes = []
        for group in _groups(map(tuple, np.argwhere(a == color))):
            r0, r1, c0, c1 = _bounds(group)
            if r1 - r0 < 2 or c1 - c0 < 2:
                break
            perimeter = {(r, c) for r in range(r0, r1 + 1)
                         for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1)}
            if group != perimeter:
                break
            boxes.append((r0, r1, c0, c1))
        else:
            if not 3 <= len(boxes) <= 6:
                continue
            covered = np.zeros(a.shape, dtype=bool)
            panels = []
            valid = True
            for r0, r1, c0, c1 in boxes:
                if covered[r0:r1 + 1, c0:c1 + 1].any():
                    valid = False
                    break
                covered[r0:r1 + 1, c0:c1 + 1] = True
                inside = a[r0 + 1:r1, c0 + 1:c1]
                if np.any(inside == color):
                    valid = False
                    break
                cells = {(int(r), int(c)): int(inside[r, c])
                         for r, c in np.argwhere(inside != bg)}
                if not cells:
                    valid = False
                    break
                panels.append(_norm(cells))
            if valid and np.all(a[~covered] == bg):
                choices.append((color, panels))
    if len(choices) != 1:
        raise ValueError("expected uniquely recognizable rectangular panel frames")
    frame, panels = choices[0]
    return bg, frame, panels


def _register(panels):
    """Exact entire multicolor signatures impose a connected offset system."""
    occurrences = {}
    for i, cells in enumerate(panels):
        local = set()
        for group in _groups(cells):
            part = {p: cells[p] for p in group}
            if len(set(part.values())) < 2:
                continue
            if len(part) < 3:
                raise ValueError("a signature needs at least three colored cells")
            pat = _pattern(part)
            if pat in local:
                raise ValueError("a panel contains two indistinguishable keys")
            local.add(pat)
            r0, _, c0, _ = _bounds(part)
            occurrences.setdefault(pat, []).append((i, (r0, c0)))
        if not local:
            raise ValueError("every panel must contain a registration signature")
    adjacency = [[] for _ in panels]
    for items in occurrences.values():
        if len(items) < 2:
            raise ValueError("every multicolor signature must have another copy")
        for (i, p), (j, q) in itertools.combinations(items, 2):
            delta = p[0] - q[0], p[1] - q[1]
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in adjacency[i]:
            proposed = offsets[i][0] + dr, offsets[i][1] + dc
            if j in offsets:
                if offsets[j] != proposed:
                    raise ValueError("inconsistent signature registration")
            else:
                offsets[j] = proposed
                stack.append(j)
    if len(offsets) != len(panels):
        raise ValueError("signature graph is disconnected")
    union = {}
    for i, cells in enumerate(panels):
        dr, dc = offsets[i]
        for (r, c), v in cells.items():
            p = r + dr, c + dc
            if p in union and union[p] != v:
                raise ValueError("registered foreground colors conflict")
            union[p] = v
    union = _norm(union)
    if max(_shape(union)) > 30:
        raise ValueError("assembled picture exceeds ARC dimensions")
    return union, offsets, occurrences


def _geometry(union):
    pads, markers, keys = [], [], []
    for group in _groups(union):
        colors = {union[p] for p in group}
        if len(colors) > 1:
            keys.append(group)
        elif len(group) == 1:
            markers.append(next(iter(group)))
        else:
            r0, r1, c0, c1 = _bounds(group)
            if len(group) != (r1 - r0 + 1) * (c1 - c0 + 1) or len(group) < 4:
                raise ValueError("pads must be solid rectangles of at least four cells")
            pads.append((_bounds(group), next(iter(colors))))
    if len(pads) != 2 or len(markers) != 1 or len(keys) < 2:
        raise ValueError("need two pads, a singleton swatch, and at least two keys")
    marker = markers[0]
    pad_colors = {p[1] for p in pads}
    key_colors = {union[p] for group in keys for p in group}
    swatch = union[marker]
    if len(pad_colors) != 2 or swatch in pad_colors | key_colors or pad_colors & key_colors:
        raise ValueError("pad, signature, and swatch color roles must be disjoint")
    a, b = (p[0] for p in pads)
    candidates = []
    if min(a[1], b[1]) >= max(a[0], b[0]) and (a[3] + 2 < b[2] or b[3] + 2 < a[2]):
        ordered = sorted(pads, key=lambda p: p[0][2])
        start, stop = ordered[0][0][3] + 1, ordered[1][0][2]
        candidates.append((1, ordered, start, stop))
    if min(a[3], b[3]) >= max(a[2], b[2]) and (a[1] + 2 < b[0] or b[1] + 2 < a[0]):
        ordered = sorted(pads, key=lambda p: p[0][0])
        start, stop = ordered[0][0][1] + 1, ordered[1][0][0]
        candidates.append((0, ordered, start, stop))
    if len(candidates) != 1:
        raise ValueError("pads must oppose each other along one unique axis")
    axis, ordered, start, stop = candidates[0]
    if any(start <= p[axis] < stop for p in union):
        raise ValueError("the entire axial gap must be background before filling")
    return {"axis": axis, "pads": ordered, "start": start, "stop": stop,
            "gap": stop - start, "marker": marker, "swatch": swatch,
            "key_count": len(keys), "shape": _shape(union)}


def _render(union, bg):
    a = np.full(_shape(union), bg, dtype=np.int8)
    for p, v in union.items():
        a[p] = v
    return a


def _fill(union, bg, info):
    out = _render(union, bg)
    start, stop = info["start"], info["stop"]
    half = info["gap"] // 2
    slices = [(start, start + half, info["pads"][0][1]),
              (stop - half, stop, info["pads"][1][1])]
    if info["gap"] % 2:
        slices.append((start + half, start + half + 1, info["swatch"]))
    for lo, hi, color in slices:
        if info["axis"] == 1:
            out[:, lo:hi] = color
        else:
            out[lo:hi, :] = color
    return out


def _analyze(input_grid):
    bg, frame, panels = _decode(input_grid)
    union, offsets, occurrences = _register(panels)
    info = _geometry(union)
    info.update(background=bg, frame=frame, panels=panels, union=union,
                offsets=offsets, occurrences=occurrences)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    return _fill(info["union"], info["background"], info)


_FOOTPRINTS = (
    ("111",), ("1", "1", "1"), ("1111",), ("1", "1", "1", "1"),
    ("11", "11"), ("111", "111"), ("11", "11", "11"),
    ("111", "101", "111"), ("100", "100", "111"),
    ("110", "011"), ("010", "111", "010"),
    ("111", "011", "001"),
)


def _signature(rng, palette):
    rows = _FOOTPRINTS[int(rng.integers(len(_FOOTPRINTS)))]
    points = [(r, c) for r, row in enumerate(rows) for c, x in enumerate(row) if x == "1"]
    colors = [int(rng.choice(palette)) for _ in points]
    colors[0], colors[-1] = palette[0], palette[1]
    return dict(zip(points, colors))


def _translated(cells, r0, c0):
    return {(r + r0, c + c0): v for (r, c), v in cells.items()}


def _frame(cells, bg, frame, padding):
    cells = _norm(cells)
    h, w = _shape(cells)
    out = np.full((h + 2 + 2 * padding, w + 2 + 2 * padding), bg, dtype=np.int8)
    out[[0, -1], :] = frame
    out[:, [0, -1]] = frame
    for (r, c), v in cells.items():
        out[r + 1 + padding, c + 1 + padding] = v
    return out


def _pack(rectangles, bg, rng):
    """Enumerate bounded shelf plans; no piece is rotated by the packer."""
    n = len(rectangles)
    feasible = []
    # At most 2 * 120 * 16 plans and six rectangles are supported by the solver.
    # The constructor uses three to five rectangles, hence at most 3840 plans.
    for column_shelves in (False, True):
        shapes = [a.T.shape if column_shelves else a.shape for a in rectangles]
        for order in itertools.permutations(range(n)):
            for mask in range(1 << (n - 1)):
                rows, row = [], []
                for k, i in enumerate(order):
                    row.append(i)
                    if k == n - 1 or mask & (1 << k):
                        rows.append(row)
                        row = []
                heights = [max(shapes[i][0] for i in row) for row in rows]
                widths = [sum(shapes[i][1] for i in row) + len(row) - 1 for row in rows]
                h, w = sum(heights) + len(rows) + 1, max(widths) + 2
                if h <= 30 and w <= 30:
                    feasible.append((column_shelves, rows, heights, widths, h, w))
    if not feasible:
        raise ValueError("panels do not fit a 30 by 30 canvas")
    transpose, rows, heights, widths, low_h, low_w = feasible[int(rng.integers(len(feasible)))]
    area = sum(int(np.count_nonzero(a != bg)) for a in rectangles)
    sizes = [(h, w) for h in range(low_h, 31) for w in range(low_w, 31) if h * w > 2 * area]
    if not sizes:
        raise ValueError("not enough exterior background")
    h, w = sizes[int(rng.integers(len(sizes)))]
    out = np.full((h, w), bg, dtype=np.int8)
    extra = h - low_h
    gaps = rng.multinomial(extra, [1 / (len(rows) + 1)] * (len(rows) + 1))
    top = 1 + int(gaps[0])
    for row_index, row in enumerate(rows):
        left = 1 + int(rng.integers(w - widths[row_index] - 1))
        for i in row:
            rect = rectangles[i].T if transpose else rectangles[i]
            rh, rw = rect.shape
            r = top + int(rng.integers(heights[row_index] - rh + 1))
            out[r:r + rh, left:left + rw] = rect
            left += rw + 1
        top += heights[row_index] + 1 + int(gaps[row_index + 1])
    return out.T.copy() if transpose else out


def _make_pair(rng, axis, parity, split_count, test=False, unequal=False):
    for attempt in range(96):
        palette = [int(x) for x in rng.permutation(10)]
        bg, frame, ca, cb, swatch = palette[:5]
        key_palette = palette[5:7 + int(rng.integers(2))]
        k0, k1 = _signature(rng, key_palette), _signature(rng, key_palette)
        if _pattern(k0) == _pattern(k1):
            continue
        kh0, kw0 = _shape(k0)
        kh1, kw1 = _shape(k1)
        d0, d1 = (1, 3) if unequal else (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        if unequal and rng.integers(2):
            d0, d1 = d1, d0
        lengths = [g for g in (range(8, 12) if test else range(4, 8)) if g % 2 == parity]
        gap = int(rng.choice(lengths))
        x0, x1 = kw0 + 1, kw0 + 1 + d0 + gap
        k1x = x1 + d1 + 1
        if k1x + kw1 > 26:
            continue
        kr0, kr1 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        pr0, pr1 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        span0, span1 = int(rng.integers(4, 7)), int(rng.integers(4, 7))
        keys = [_translated(k0, kr0, 0), _translated(k1, kr1, k1x)]
        pads = [{(r, c): ca for r in range(pr0, pr0 + span0) for c in range(x0, x0 + d0)},
                {(r, c): cb for r in range(pr1, pr1 + span1) for c in range(x1, x1 + d1)}]
        marker_side = int(rng.integers(2))
        marker = {(-1, 0 if marker_side == 0 else k1x + kw1 - 1): swatch}
        # One panel carries the two reference signatures; no panel contains both pads.
        panels = [keys[0] | keys[1]]
        split_sides = set(int(x) for x in rng.choice(2, split_count, replace=False))
        for side in range(2):
            pad = pads[side]
            bands = [pad]
            if side in split_sides:
                lo, hi, _, _ = _bounds(pad)
                cut = int(rng.integers(lo + 1, hi))
                bands = [{p: v for p, v in pad.items() if p[0] <= cut},
                         {p: v for p, v in pad.items() if p[0] >= cut}]
            marker_band = int(rng.integers(len(bands)))
            for j, band in enumerate(bands):
                panel = keys[side] | band
                if side == marker_side and j == marker_band:
                    panel = panel | marker
                panels.append(panel)
        union = _norm(keys[0] | keys[1] | pads[0] | pads[1] | marker)
        if axis == 0:
            panels = [{(c, r): v for (r, c), v in p.items()} for p in panels]
            union = _norm({(c, r): v for (r, c), v in union.items()})
        try:
            info = _geometry(union)
            rectangles = [_frame(p, bg, frame, int(rng.random() < .12)) for p in panels]
            inp = _pack(rectangles, bg, rng)
            # Cross-check intended coordinates against reconstruction from the packed input.
            analysis = _analyze(inp)
            if analysis["union"] != union:
                continue
            out = _fill(union, bg, info)
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded panel construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    axis, parity = int(rng.integers(2)), int(rng.integers(2))
    plan = [(axis, parity), (1 - axis, 1 - parity)]
    missing = [(axis, 1 - parity), (1 - axis, parity)]
    rng.shuffle(missing)
    plan += missing[:min(nt, 4) - 2]
    plan += [(int(rng.integers(2)), int(rng.integers(2))) for _ in range(max(0, nt - 4))]
    tests = ([plan[int(rng.integers(nt))] for _ in range(nq)] if nt == 2 else
             [(int(rng.integers(2)), int(rng.integers(2))) for _ in range(nq)])
    puzzle, seen = {"train": [], "test": []}, set()
    for split, schedule in (("train", plan), ("test", tests)):
        for i, (axis, parity) in enumerate(schedule):
            # Both axes, parities, whole and partial pads are taught in the first two examples.
            pieces = (0 if i == 0 else 1 if i == 1 else int(rng.integers(3))) if split == "train" else int(rng.integers(1, 3))
            for _ in range(8):
                pair = _make_pair(rng, axis, parity, pieces, split == "test", split == "train" and i == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze(inp)
    union, bg = info["union"], info["background"]
    before = _render(union, bg)
    assert out.shape == before.shape, "crop must equal the registered union bounds"
    assert np.array_equal(out, _fill(union, bg, info)), "incorrect fill"
    assert info["frame"] not in set(out.flat), "packaging leaked into output"
    assert set(out.flat) == set(before.flat), "unexpected color creation or loss"
    # Each new position is checked using distances to the inward faces.
    for p in np.ndindex(out.shape):
        x = p[info["axis"]]
        if info["start"] <= x < info["stop"]:
            a, b = x - info["start"], info["stop"] - 1 - x
            color = info["pads"][0][1] if a < b else info["pads"][1][1] if b < a else info["swatch"]
            assert before[p] == bg and out[p] == color
        else:
            assert out[p] == before[p], "exterior content overwritten"
    assert not np.array_equal(out, before), "registration alone must be insufficient"
    assert sum(start <= p[info["axis"]] < stop for p in union
               for start, stop in [(info["start"], info["stop"] )]) == 0
    # No unregistered panel exposes both endpoints; their relative location needs A.
    ca, cb = (p[1] for p in info["pads"])
    assert all(not {ca, cb} <= set(p.values()) for p in info["panels"])
    assert any(not ({ca, cb} & set(p.values())) for p in info["panels"])
    cross = out.shape[1 - info["axis"]]
    for box, color in info["pads"]:
        span = box[1] - box[0] + 1 if info["axis"] == 1 else box[3] - box[2] + 1
        assert cross > span, "need visible full-slice extension beyond pad span"
        r0, r1, c0, c1 = box
        assert np.all(out[r0:r1 + 1, c0:c1 + 1] == color)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(_check_pair(pair))
    train = infos["train"]
    assert {x["axis"] for x in train} == {0, 1}
    assert {x["gap"] % 2 for x in train} == {0, 1}
    assert any(len(x["panels"]) >= 4 for x in train), "partial-pad assembly not demonstrated"
    for info in infos["test"]:
        assert info["gap"] > max(x["gap"] for x in train), "missing gap-length shift"
        if nt == 2:
            assert (info["axis"], info["gap"] % 2) in {(x["axis"], x["gap"] % 2) for x in train}


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _dependency_checks(pair):
    """Counterfactual checks, with no retained fixtures or execution traces."""
    info = _analyze(pair["input"])
    panels, axis = info["panels"], info["axis"]
    pad_colors = {color for box, color in info["pads"]}
    ruler = next(i for i, p in enumerate(panels) if not pad_colors & set(p.values()))
    # Only the two-key reference panel relates the endpoints. Removing it
    # leaves two internally registered but mutually disconnected pad groups.
    try:
        _register([p for i, p in enumerate(panels) if i != ruler])
    except ValueError as exc:
        assert str(exc) in {"signature graph is disconnected",
                            "every multicolor signature must have another copy"}
    else:
        raise AssertionError("pad groups were locatable without their reference signatures")

    # Change only reference-key spacing. Every pad-bearing panel, including
    # its marker and exact local key, is unchanged. Both gap and answer must
    # respond to this registration evidence, independently of source placement.
    changed = [dict(p) for p in panels]
    far_key = max(_groups(changed[ruler]), key=lambda group: min(p[axis] for p in group))
    moved = {}
    for p in far_key:
        q = list(p)
        q[axis] += 2
        moved[tuple(q)] = changed[ruler].pop(p)
    changed[ruler].update(moved)
    rectangles = [_frame(p, info["background"], info["frame"], 0) for p in changed]
    counter_input = _pack(rectangles, info["background"], np.random.default_rng(287))
    counter = _analyze(counter_input)
    assert counter["gap"] == info["gap"] + 2
    assert counter["swatch"] == info["swatch"]
    assert [c for box, c in counter["pads"]] == [c for box, c in info["pads"]]
    counter_output = solve(counter_input)
    assert counter_output.shape[axis] == pair["output"].shape[axis] + 2
    assert counter_output.shape[1 - axis] == pair["output"].shape[1 - axis]
    _check_pair({"input": counter_input, "output": counter_output})

    before = _render(info["union"], info["background"])
    out = _grid(pair["output"])
    cross = out.shape[1 - axis]
    assert np.count_nonzero(before != out) == info["gap"] * cross
    assert not np.array_equal(before, out), "union-only ablation must fail"
    # A two-dimensional nearest-pad fill or pad-span-only extension leaves
    # some transverse margin blank; our full cross-section rule changes it.
    margin = next(t for t in range(cross)
                  if all(not (box[0] <= t <= box[1] if axis == 1
                              else box[2] <= t <= box[3]) for box, _ in info["pads"]))
    p = (margin, info["start"]) if axis == 1 else (info["start"], margin)
    assert before[p] == info["background"] and out[p] == info["pads"][0][1]
    if info["gap"] % 2:
        center = info["start"] + info["gap"] // 2
        wrong = out.copy()
        if axis == 1:
            wrong[:, center] = info["background"]
        else:
            wrong[center, :] = info["background"]
        try:
            _check_pair({"input": pair["input"], "output": wrong})
        except AssertionError:
            pass
        else:
            raise AssertionError("background-midpoint ablation was accepted")
        # The singleton is a live color instruction, not a literal fixed color.
        replacement = next(c for c in range(10) if c not in set(pair["input"].flat))
        recolored_input = pair["input"].copy()
        recolored_input[recolored_input == info["swatch"]] = replacement
        expected = out.copy()
        expected[expected == info["swatch"]] = replacement
        assert np.array_equal(solve(recolored_input), expected)
        assert np.count_nonzero(expected != out) == cross + 1


def self_test():
    """42 seeds streamed in memory; no logs, payload files, or trace dumps."""
    signatures, input_shapes, output_shapes, patterns = set(), set(), set(), set()
    backgrounds, swatches, panel_counts, depths, gaps, spans, cases = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                backgrounds.add(info["background"])
                swatches.add(info["swatch"])
                panel_counts.add(len(info["panels"]))
                gaps.add(info["gap"])
                cases.add((info["axis"], info["gap"] % 2))
                for box, color in info["pads"]:
                    depths.add(box[3] - box[2] + 1 if info["axis"] else box[1] - box[0] + 1)
                    spans.add(box[1] - box[0] + 1 if info["axis"] else box[3] - box[2] + 1)
                for pat in info["occurrences"]:
                    roles = {}
                    patterns.add(tuple((r, c, roles.setdefault(v, len(roles))) for r, c, v in pat))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        if seed < 3:
            # Input-only symmetries and arbitrary color reassignment preserve semantics.
            pair = puzzle["test"][0]
            for k in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
            mapping = np.array([7, 4, 0, 9, 1, 6, 3, 8, 5, 2], dtype=np.int8)
            assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        if seed == 0:
            # The first two demonstrations cover both parities and axes;
            # the three-panel first example keeps the repacking check small.
            _dependency_checks(puzzle["train"][0])
            odd = next(p for p in puzzle["train"] if _analyze(p["input"])["gap"] % 2)
            if odd is not puzzle["train"][0]:
                _dependency_checks(odd)
    assert len(signatures) == 42
    assert len(input_shapes) >= 20 and len(output_shapes) >= 20
    assert len(patterns) >= 25
    assert backgrounds == swatches == set(range(10))
    assert panel_counts == {3, 4, 5} and depths == {1, 2, 3}
    assert spans == {4, 5, 6} and len(gaps) >= 6
    assert cases == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert _puzzle_key(generate_puzzle(19)) == _puzzle_key(generate_puzzle(19))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1234, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
