"""Sheared lantern: a reflected figure opens and colors its frame's sightlines.

Only NumPy and the standard library are used. All solving conventions are fixed;
no generated metadata is required by solve().
"""
from __future__ import annotations

import json
import numpy as np

_CORNERS = ("TL", "TR", "BL", "BR")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
                    cells.add(p)
        result.append(cells)
    return result


def _l_cells(corner, t, b, l, r):
    y = t if corner[0] == "T" else b
    x = l if corner[1] == "L" else r
    dy = 1 if corner[0] == "T" else -1
    dx = 1 if corner[1] == "L" else -1
    return {(y, x), (y + dy, x), (y, x + dx)}


def _perimeter(t, b, l, r):
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _analyze(input_grid):
    """Recognize the four cue components and the single reflected interior figure."""
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if int(counts[bg]) <= a.size // 2:
        raise ValueError("input background must occupy more than half the grid")
    components = _components(a != bg)
    big = [c for c in components if len(c) > 3]
    cues = [c for c in components if len(c) == 3]
    dots = [c for c in components if len(c) == 1]
    if len(big) != 1 or len(cues) != 4 or len(components) != 5 + len(dots):
        raise ValueError("expected one figure, four three-cell cues, and isolated dots")
    cue_cells = set().union(*cues)
    t, l = map(int, np.min(list(cue_cells), axis=0))
    b, r = map(int, np.max(list(cue_cells), axis=0))
    if b - t < 4 or r - l < 4:
        raise ValueError("corner cues must have separated sides")
    corner_colors = []
    for corner in _CORNERS:
        cells = _l_cells(corner, t, b, l, r)
        if cells not in cues or len({int(a[p]) for p in cells}) != 1:
            raise ValueError("cues must be aligned monochrome inward-facing Ls")
        corner_colors.append(int(a[next(iter(cells))]))
    structures = [c for c in set(corner_colors) if corner_colors.count(c) == 3]
    if len(structures) != 1:
        raise ValueError("three cues must share a color and the fourth must differ")
    structure = structures[0]
    corner = next(i for i, c in enumerate(corner_colors) if c != structure)
    fill = corner_colors[corner]
    figure = big[0]
    ft, fl = map(int, np.min(list(figure), axis=0))
    fb, fr = map(int, np.max(list(figure), axis=0))
    if not (t + 2 <= ft <= fb <= b - 2 and l + 2 <= fl <= fr <= r - 2):
        raise ValueError("figure requires an empty cell between it and each frame side")
    height = fb - ft + 1
    if height < 4 or height % 2:
        raise ValueError("figure must have even height of at least four")
    colors = tuple(sorted({int(a[p]) for p in figure}))
    if len(colors) != 2 or set(colors) & {bg, structure, fill}:
        raise ValueError("figure needs two colors distinct from all three frame roles")
    rows = []
    for y in range(ft, fb + 1):
        xs = sorted(c for yy, c in figure if yy == y)
        if not xs or xs != list(range(xs[0], xs[-1] + 1)):
            raise ValueError("figure rows must be nonempty contiguous spans")
        rows.append((xs[0], len(xs)))
    for i in range(height // 2):
        if rows[i] != rows[-1 - i]:
            raise ValueError("the complete row geometry must reflect exactly")
        x, width = rows[i]
        if not np.array_equal(a[ft + i, x:x + width], a[fb - i, x:x + width]):
            raise ValueError("the complete row colors must reflect exactly")
    middle = rows[height // 2 - 1][1]
    if middle < 2 or fr + middle > r - 2:
        raise ValueError("middle width must be at least two and the slide must fit")
    inert = set().union(*dots) if dots else set()
    if any(t <= y <= b or l <= x <= r for y, x in inert):
        raise ValueError("dots must be outside both frame spans")
    if any(int(a[p]) not in colors for p in inert):
        raise ValueError("inert dots reuse the two figure colors")
    return dict(bg=bg, structure=structure, fill=fill, corner=_CORNERS[corner],
                t=t, b=b, l=l, r=r, figure=figure, colors=colors, ft=ft, fb=fb,
                rows=rows, distance=middle, width=fr - fl + 1, inert=inert)


def _transformed(a, info, *, move=True, swap=True, distance=None):
    distance = info["distance"] if distance is None else distance
    cut = (info["ft"] + info["fb"] + 1) // 2
    c0, c1 = info["colors"]
    cells = {}
    for y, x in sorted(info["figure"]):
        xx = x + (distance if move and y < cut else 0)
        if not (info["l"] < xx < info["r"]):
            return None
        color = int(a[y, x])
        cells[y, xx] = (c1 if color == c0 else c0) if swap else color
    return cells


def _render(a, info, *, move=True, swap=True, distance=None, project=True,
            nearest=True):
    """Render the distributed source, then its four visible faces."""
    cells = _transformed(a, info, move=move, swap=swap, distance=distance)
    if cells is None:
        return None
    t, b, l, r = (info[k] for k in ("t", "b", "l", "r"))
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    cross = ((t <= yy) & (yy <= b)) | ((l <= xx) & (xx <= r))
    out = a.copy()
    out[cross] = info["fill"]
    for p in _perimeter(t, b, l, r):
        out[p] = info["structure"]
    for p, color in cells.items():
        out[p] = color
    if project:
        for y in range(t, b + 1):
            visible = sorted((x, color) for (yy, x), color in cells.items() if yy == y)
            if visible:
                out[y, :l] = visible[0 if nearest else -1][1]
                out[y, r + 1:w] = visible[-1 if nearest else 0][1]
        for x in range(l, r + 1):
            visible = sorted((y, color) for (y, xx), color in cells.items() if xx == x)
            if visible:
                out[:t, x] = visible[0 if nearest else -1][1]
                out[b + 1:h, x] = visible[-1 if nearest else 0][1]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    result = _render(a, _analyze(a))
    if result is None:
        raise ValueError("slide leaves the lantern")
    return result


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _row_pattern(rng, width, c0, c1):
    mode = int(rng.integers(5))
    if mode == 0:
        return [_pick(rng, (c0, c1))] * width
    if mode == 1:
        return [c0 if i % 2 == 0 else c1 for i in range(width)]
    if mode == 2:
        return [c0 if i in (0, width - 1) else c1 for i in range(width)]
    if mode == 3:
        cut = int(rng.integers(1, width + 1))
        return [c0] * cut + [c1] * (width - cut)
    return [c0 if v == 0 else c1 for v in rng.integers(0, 2, width)]


def _profile(rng, distance, force_overwide):
    k = int(rng.integers(2, 6))
    centered = rng.random() < .28
    options = [x for x in range(1, 10) if not centered or x % 2 == distance % 2]
    widths = [_pick(rng, options) for _ in range(k - 1)] + [distance]
    if force_overwide:
        widths[int(rng.integers(k - 1))] = _pick(rng, [x for x in options if x > distance])
    elif len(set(widths)) == 1:
        widths[0] = _pick(rng, [x for x in options if x != distance])
    if centered:
        lefts = [(max(widths) - width) // 2 for width in widths]
    else:
        lefts = [0] * k
        for i in range(k - 2, -1, -1):
            lo = max(lefts[i + 1] - 3, lefts[i + 1] - widths[i] + 1)
            hi = min(lefts[i + 1] + 3, lefts[i + 1] + widths[i + 1] - 1)
            lefts[i] = int(rng.integers(lo, hi + 1))
        base = min(lefts)
        lefts = [x - base for x in lefts]
    bounding = max(x + width for x, width in zip(lefts, widths))
    return widths, lefts, bounding


def _margins(rng, inner_size, showcase):
    available = min(12, 30 - inner_size)
    minimum = 4 if showcase else 2
    if available < minimum:
        return None
    total = int(rng.integers(minimum, available + 1))
    low = 2 if showcase else 0
    first = int(rng.integers(low, total - low + 1))
    return first, total - first


def _witnesses(a, info):
    cells = _transformed(a, info)
    cut = (info["ft"] + info["fb"] + 1) // 2
    upper = {x for y, x in cells if y < cut}
    lower = {x for y, x in cells if y >= cut}
    depth = False
    for x in upper & lower:
        line = sorted((y, color) for (y, xx), color in cells.items() if x == xx)
        depth |= line[0][1] != line[-1][1]
    return dict(depth=depth, revealed_lower=bool(lower - upper),
                revealed_upper=bool(upper - lower))


def _make_pair(rng, distance, showcase, overwide):
    for _ in range(250):
        widths, lefts, bounding = _profile(rng, distance, overwide)
        k = len(widths)
        pt, pb, pl, pr = map(int, rng.integers(0, 3, 4))
        fh, fw = 2 * k + 4 + pt + pb, bounding + distance + 4 + pl + pr
        vertical, horizontal = _margins(rng, fh, showcase), _margins(rng, fw, showcase)
        if vertical is None or horizontal is None:
            continue
        t, below = vertical
        l, right = horizontal
        h, w = fh + t + below, fw + l + right
        b, r = t + fh - 1, l + fw - 1
        bg, structure, fill, c0, c1 = map(int, rng.choice(10, 5, replace=False))
        a = np.full((h, w), bg, dtype=np.int8)
        key_corner = _pick(rng, _CORNERS)
        for corner in _CORNERS:
            for p in _l_cells(corner, t, b, l, r):
                a[p] = fill if corner == key_corner else structure
        for i, (width, offset) in enumerate(zip(widths, lefts)):
            values = _row_pattern(rng, width, c0, c1)
            if i == k - 1:
                values[0], values[-1] = c0, c1
            x = l + 2 + pl + offset
            for y in (t + 2 + pt + i, t + 2 + pt + 2 * k - 1 - i):
                a[y, x:x + width] = values
        candidates = [(y, x) for y in range(h) for x in range(w)
                      if not t <= y <= b and not l <= x <= r]
        rng.shuffle(candidates)
        target = int(rng.integers(0, 8))
        placed = 0
        for y, x in candidates:
            if placed >= target:
                break
            if any(0 <= yy < h and 0 <= xx < w and a[yy, xx] != bg
                   for yy, xx in ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))):
                continue
            a[y, x] = _pick(rng, (c0, c1))
            placed += 1
        try:
            info = _analyze(a)
        except ValueError:
            continue
        witness = _witnesses(a, info)
        if showcase and not all(witness.values()):
            continue
        return {"input": a, "output": _render(a, info)}
    raise RuntimeError("failed to sample a fitting lantern with the requested witnesses")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Hold out a distance, not a palette or a fixed figure size. The solver never
    # needs this schedule: the distance is always measured from the input.
    heldout = int(rng.integers(3, 7))
    available = [d for d in range(2, 7) if d != heldout]
    first_two = list(map(int, rng.choice(available, 2, replace=False)))
    train_distances = first_two + [_pick(rng, available) for _ in range(num_train - 2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, distances in (("train", train_distances), ("test", [heldout] * num_test)):
        for i, distance in enumerate(distances):
            for _ in range(20):
                showcase = split == "train" and i == 0
                overwide = showcase or split == "test" or rng.random() < .5
                pair = _make_pair(rng, distance, showcase, overwide)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("failed to construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    info = _analyze(a)
    cells = _transformed(a, info)
    t, b, l, r = (info[k] for k in ("t", "b", "l", "r"))
    yy, xx = np.indices(a.shape)
    cross = ((t <= yy) & (yy <= b)) | ((l <= xx) & (xx <= r))
    assert np.array_equal(out[~cross], a[~cross]), "off-cross cells must be preserved"
    assert all(int(out[p]) == info["structure"] for p in _perimeter(t, b, l, r))
    for y in range(t + 1, b):
        for x in range(l + 1, r):
            assert int(out[y, x]) == cells.get((y, x), info["fill"])
    # Direct geometric checks of each beam, including empty sightlines and
    # rays seen through the gap left by the shifted upper half.
    for y in range(t, b + 1):
        line = sorted((x, color) for (yy, x), color in cells.items() if yy == y)
        lc = line[0][1] if line else info["fill"]
        rc = line[-1][1] if line else info["fill"]
        assert np.all(out[y, :l] == lc) and np.all(out[y, r + 1:] == rc)
    for x in range(l, r + 1):
        line = sorted((y, color) for (y, xx), color in cells.items() if xx == x)
        tc = line[0][1] if line else info["fill"]
        bc = line[-1][1] if line else info["fill"]
        assert np.all(out[:t, x] == tc) and np.all(out[b + 1:, x] == bc)
    assert np.array_equal(solve(a), out)
    assert not np.array_equal(a, out)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            infos[split].append(info)
    train_distances = {i["distance"] for i in infos["train"]}
    assert len(train_distances) >= 2, "training must teach width-dependent displacement"
    assert all(i["distance"] not in train_distances for i in infos["test"])
    assert any(i["width"] > i["distance"] for i in infos["train"])
    assert any(all(_witnesses(_grid(p["input"]), i).values())
               and i["t"] > 0 and i["l"] > 0
               and i["b"] < np.asarray(p["input"]).shape[0] - 1
               and i["r"] < np.asarray(p["input"]).shape[1] - 1
               for p, i in zip(puzzle["train"], infos["train"]))
    # Every listed shortcut must be falsified by a visible training output.
    alternatives = ({"move": False}, {"swap": False},
                    {"move": False, "swap": False}, {"project": False},
                    {"nearest": False}, {"distance": 1})
    for alt in alternatives:
        assert any(not np.array_equal(_render(_grid(p["input"]), i, **alt), p["output"])
                   for p, i in zip(puzzle["train"], infos["train"])), alt
    assert any(not np.array_equal(_render(_grid(p["input"]), i, distance=i["width"]), p["output"])
               for p, i in zip(puzzle["train"], infos["train"]))
    # Test the interaction on the arms alone: moving the interior while casting
    # its old shadow must fail even if the interior itself is drawn correctly.
    beam_motion_witness = False
    for p, i in zip(puzzle["train"], infos["train"]):
        a, out = _grid(p["input"]), _grid(p["output"])
        yy, xx = np.indices(a.shape)
        arms = ((((yy < i["t"]) | (yy > i["b"])) & (i["l"] <= xx) & (xx <= i["r"]))
                | (((xx < i["l"]) | (xx > i["r"])) & (i["t"] <= yy) & (yy <= i["b"])))
        old_shadow = _render(a, i, move=False)
        beam_motion_witness |= not np.array_equal(out[arms], old_shadow[arms])
    assert beam_motion_witness, "B's slide must alter A's exterior beams"


def _fingerprint(puzzle):
    return tuple((s, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]))
                 for s in ("train", "test"))


def self_test():
    """Bounded diversity/invariant audit over 42 seeds, plus the count contract."""
    signatures, layouts, shapes, frame_shapes, profiles, palettes = (set() for _ in range(6))
    corners, distances, heights, backgrounds = (set() for _ in range(4))
    counts = dict(centered=0, staggered=0, depth=0, boundary=0, no_dots=0,
                  dots=0, monochrome=0, overwide=0, revealed=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                i = _analyze(a)
                k = len(i["rows"]) // 2
                profile = tuple(i["rows"][:k])
                centered = len({2 * x + width for x, width in profile}) == 1
                witnesses = _witnesses(a, i)
                counts["centered" if centered else "staggered"] += 1
                counts["depth"] += witnesses["depth"]
                counts["revealed"] += witnesses["revealed_lower"] and witnesses["revealed_upper"]
                counts["boundary"] += (i["t"] == 0 or i["l"] == 0 or i["b"] == a.shape[0] - 1
                                        or i["r"] == a.shape[1] - 1)
                counts["dots" if i["inert"] else "no_dots"] += 1
                counts["monochrome"] += any(len(set(a[i["ft"] + row, x:x + width])) == 1
                                             for row, (x, width) in enumerate(profile))
                counts["overwide"] += i["width"] > i["distance"]
                layouts.add((a.shape, i["t"], i["b"], i["l"], i["r"], profile))
                shapes.add(a.shape)
                frame_shapes.add((i["b"] - i["t"] + 1, i["r"] - i["l"] + 1))
                profiles.add(tuple((x - min(z[0] for z in profile), width) for x, width in profile))
                palettes.add((i["bg"], i["structure"], i["fill"], i["colors"]))
                corners.add(i["corner"])
                distances.add(i["distance"])
                heights.add(len(i["rows"]))
                backgrounds.add(i["bg"])
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 45 and len(frame_shapes) >= 35 and len(profiles) >= 90
    assert len(palettes) >= 150 and len(backgrounds) == 10
    assert corners == set(_CORNERS) and distances == set(range(2, 7))
    assert heights == {4, 6, 8, 10}
    assert all(counts.values()), counts
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42000 + nt * 10 + ns, nt, ns))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    return dict(seeds=42, distinct_puzzles=len(signatures), layouts=len(layouts),
                canvas_shapes=len(shapes), frame_shapes=len(frame_shapes),
                profiles=len(profiles), palettes=len(palettes), categories=counts)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
