"""Exact-span sparse rows decode into endpoint-painted rectangular loops.

Only standard-library modules and NumPy are used. No files are read at runtime.
The fixed 7-cell lattice and all marker conventions are input-observable.
"""
from collections import Counter
import hashlib
import json
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0, 494874332, 205623422, 657906063, 696946128, 785674729,
              339604850, 446350746, 593285230, 766883800, 215926310, 97872379,
              902908987, 134914409, 818409047, 873363066, 95140578, 23080403,
              177329778, 110403409, 634341285, 590136253, 610809737, 141435194,
              636198423, 876164063, 250111713, 957775881, 479905132, 105518348,
              971423210, 107294287, 864125766, 588358794, 451717030, 334680022,
              372364063, 522247021, 612489575, 640214466, 22508111, 523659144)


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, items):
    return items[_ri(rng, 0, len(items) - 1)]


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("expected a 2D grid with dimensions in 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("expected integer colors in 0..9")
    return a.astype(np.int8, copy=True)


def _near(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1


def _neighbors(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _components(cells, steps=D8):
    left, groups = set(cells), []
    while left:
        p = min(left)
        left.remove(p)
        group, queue = {p}, [p]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _runs(row, bg):
    runs, c = [], 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c + 1
        while end < len(row) and int(row[end]) == color:
            end += 1
        runs.append((color, c, end))
        c = end
    return runs


def _rectangles(coarse, bg):
    """Return (color, first row, exclusive last row, left, exclusive right)."""
    rows = [_runs(row, bg) for row in coarse]
    rectangles = []
    for r, runs in enumerate(rows):
        previous = set(rows[r - 1]) if r else set()
        for color, left, right in runs:
            if (color, left, right) in previous:
                continue
            end = r + 1
            while end < len(rows) and (color, left, right) in rows[end]:
                end += 1
            rectangles.append((color, r, end, left, right))
    return rectangles


def _bounds(rect):
    _, first, end, left, right = rect
    return 2 + 7 * first, 7 * end - 1, 2 + 7 * left, 7 * right - 1


def _shell(rect):
    top, bottom, left, right = _bounds(rect)
    return ({(top, c) for c in range(left, right + 1)} |
            {(bottom, c) for c in range(left, right + 1)} |
            {(r, left) for r in range(top, bottom + 1)} |
            {(r, right) for r in range(top, bottom + 1)})


def _geometry(rectangles, shape):
    footprint = np.full(shape, -1, dtype=np.int16)
    shells = []
    for i, rect in enumerate(rectangles):
        top, bottom, left, right = _bounds(rect)
        assert np.all(footprint[top:bottom + 1, left:right + 1] == -1)
        footprint[top:bottom + 1, left:right + 1] = i
        shells.append(_shell(rect))
    return footprint, shells


def _render(coarse, bg, rectangles, markers, fill=True):
    out = np.full((7 * coarse.shape[0] + 2, 7 * coarse.shape[1] + 2), bg, dtype=np.int8)
    for i, rect in enumerate(rectangles):
        top, bottom, left, right = _bounds(rect)
        color = rect[0]
        out[top, left:right + 1] = color
        out[bottom, left:right + 1] = color
        out[top:bottom + 1, left] = color
        out[top:bottom + 1, right] = color
        if fill:
            out[top + 1:bottom, left + 1:right] = markers[i]["paint"]
    for marker in markers:
        for p in marker["wire"]:
            out[p] = marker["color"]
        out[marker["source"]] = marker["paint"]
    return out


def _cavity_mask(a, bg):
    """Four-neighbor background regions disconnected from every canvas edge."""
    h, w = a.shape
    exterior = np.zeros(a.shape, dtype=bool)
    queue = []
    border = ({(r, c) for r in range(h) for c in (0, w - 1)} |
              {(r, c) for r in (0, h - 1) for c in range(w)})
    for p in sorted(border):
        if a[p] == bg:
            exterior[p] = True
            queue.append(p)
    for r, c in queue:
        for dr, dc in D4:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and a[y, x] == bg and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    return (a == bg) & ~exterior


def _intended_cavities(shape, rectangles):
    mask = np.zeros(shape, dtype=bool)
    for rect in rectangles:
        top, bottom, left, right = _bounds(rect)
        mask[top + 1:bottom, left + 1:right] = True
    return mask


def _analyze(input_grid):
    """Recover run stacks and unique source ownership using input cells only."""
    a = _grid(input_grid)
    h, w = a.shape
    if h % 7 != 2 or w % 7 != 2 or h < 9 or w < 9:
        raise ValueError("lattice dimensions must be 7R+2 by 7C+2, R,C in 1..4")
    bg = int(a[0, 0])
    coarse = a[4:h:7, 4:w:7].copy()
    if not np.any(coarse != bg):
        raise ValueError("at least one lattice dot is required")
    rectangles = _rectangles(coarse, bg)
    footprint, shells = _geometry(rectangles, a.shape)
    anchor = np.zeros(a.shape, dtype=bool)
    anchor[4:h:7, 4:w:7] = True
    extras = set(map(tuple, np.argwhere((a != bg) & ~anchor)))
    if any(footprint[p] != -1 for p in extras):
        raise ValueError("tails must stay outside every decoded rectangle")
    payload = {int(c) for c in coarse.flat if c != bg}
    markers = [None] * len(rectangles)
    for component in _components(extras):
        sources = [p for p in component if int(a[p]) not in payload]
        if len(sources) != 1:
            raise ValueError("each tail needs one endpoint of a non-boundary color")
        source = sources[0]
        wire = component - {source}
        if not 1 <= len(wire) <= 4 or len({int(a[p]) for p in wire}) != 1:
            raise ValueError("a tail must have 1..4 same-colored wire cells")
        degrees = {p: sum(_near(p, q) for q in component) for p in component}
        if degrees[source] != 1 or sorted(degrees.values()) != [1, 1] + [2] * (len(component) - 2):
            raise ValueError("a tail must be a simple eight-neighbor path ending in its paint")
        owners = {i for i, shell in enumerate(shells)
                  if _neighbors(component, a.shape) & shell}
        wire_owners = {i for i, shell in enumerate(shells)
                       if _neighbors(wire, a.shape) & shell}
        if len(owners) != 1 or owners != wire_owners:
            raise ValueError("tail ownership must be unique through wire attachment")
        owner = next(iter(owners))
        color = int(a[next(iter(wire))])
        if color != rectangles[owner][0] or markers[owner] is not None:
            raise ValueError("one matching-color tail is required per loop")
        markers[owner] = {"source": source, "paint": int(a[source]), "wire": wire, "color": color}
    if any(marker is None for marker in markers):
        raise ValueError("every decoded loop must have one productive endpoint")
    hollow = _render(coarse, bg, rectangles, markers, fill=False)
    if not np.array_equal(_cavity_mask(hollow, bg), _intended_cavities(a.shape, rectangles)):
        raise ValueError("tails may not create additional enclosed background pockets")
    return a, coarse, bg, rectangles, markers


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode, establish eight-neighbor ownership, and paint the cavities."""
    _, coarse, bg, rectangles, markers = _analyze(input_grid)
    return _render(coarse, bg, rectangles, markers)


def _features(coarse, bg):
    rows = [_runs(row, bg) for row in coarse]
    horizontal = any(right - left > 1 for row in rows for _, left, right in row)
    exact = any(set(upper) & set(lower) for upper, lower in zip(rows, rows[1:]))
    partial = any(a == b and (l, r) != (ll, rr) and max(l, ll) < min(r, rr)
                  for upper, lower in zip(rows, rows[1:])
                  for a, l, r in upper for b, ll, rr in lower)
    depth = max(rect[2] - rect[1] for rect in _rectangles(coarse, bg))
    return horizontal, bool(exact), partial, depth


def _partial_pairs(rectangles):
    return [(i, j) for i, a in enumerate(rectangles) for j, b in enumerate(rectangles)
            if i < j and a[0] == b[0] and (a[3], a[4]) != (b[3], b[4])
            and (a[2] == b[1] or b[2] == a[1]) and max(a[3], b[3]) < min(a[4], b[4])]


def _paint_run(coarse, row, left, right, color, bg):
    coarse[row, left:right] = color
    if left and coarse[row, left - 1] == color:
        coarse[row, left - 1] = bg
    if right < coarse.shape[1] and coarse[row, right] == color:
        coarse[row, right] = bg


def _sample_coarse(rng, bg, payload, role, shift, is_test):
    if is_test:
        rows = _ri(rng, 3, 4)
        cols = _ri(rng, 2 if shift == "combine" else 1, 4)
    else:
        rows = _ri(rng, 1 if role == "random" else 2, 4)
        cols = _ri(rng, 2 if role in ("exact", "partial") else 1, 4)
    for _ in range(160):
        coarse = np.full((rows, cols), bg, dtype=np.int8)
        density = _pick(rng, (0.25, 0.45, 0.65, 0.85, 1.0))
        for r in range(rows):
            if r and rng.random() < 0.3:
                coarse[r] = coarse[r - 1]
            else:
                mask = rng.random(cols) < density
                coarse[r, mask] = rng.choice(payload, size=int(mask.sum()))
        color = _pick(rng, payload)
        if role == "exact" or (is_test and shift == "deeper"):
            depth = 3 if is_test else 2
            top = _ri(rng, 0, rows - depth)
            left = _ri(rng, 0, cols - (2 if role == "exact" else 1))
            right = _ri(rng, left + (2 if role == "exact" else 1), cols)
            for r in range(top, top + depth):
                _paint_run(coarse, r, left, right, color, bg)
        elif role == "partial" or (is_test and shift == "combine"):
            spans = [(l, r) for l in range(cols) for r in range(l + 1, cols + 1)]
            choices = [(a, b) for a in spans for b in spans if a != b and max(a[0], b[0]) < min(a[1], b[1])]
            first, second = _pick(rng, choices)
            depth = _ri(rng, 2, rows - 1) if is_test else 1
            top = _ri(rng, 0, rows - depth - 1)
            for r in range(top, top + depth):
                _paint_run(coarse, r, *first, color, bg)
            _paint_run(coarse, top + depth, *second, color, bg)
        if not np.any(coarse != bg):
            continue
        horizontal, exact, partial, depth = _features(coarse, bg)
        if role == "exact" and not (horizontal and exact):
            continue
        if role == "partial" and not partial:
            continue
        if not is_test and shift == "deeper" and depth > 2:
            continue
        if not is_test and shift == "combine" and exact and partial:
            continue
        if is_test and shift == "deeper" and depth < 3:
            continue
        if is_test and shift == "combine" and not (exact and partial):
            continue
        if len(_rectangles(coarse, bg)) <= 12:
            if rng.random() < 0.5:
                coarse = coarse[::-1]
            if rng.random() < 0.5:
                coarse = coarse[:, ::-1]
            return coarse.copy()
    return None


def _place_markers(rng, coarse, bg, rectangles, paint_pool):
    shape = (7 * coarse.shape[0] + 2, 7 * coarse.shape[1] + 2)
    footprint, shells = _geometry(rectangles, shape)
    anchors = {(4 + 7 * r, 4 + 7 * c) for r in range(coarse.shape[0])
               for c in range(coarse.shape[1])}
    canvas = set(map(tuple, np.argwhere(footprint == -1))) - anchors - {(0, 0)}
    shell_near = [_neighbors(shell, shape) for shell in shells]
    allowed = []
    for i in range(len(rectangles)):
        other = set().union(*(ns for j, ns in enumerate(shell_near) if i != j))
        allowed.append(canvas - other)
    paints = [_pick(rng, paint_pool) for _ in rectangles]
    partial_pairs = _partial_pairs(rectangles)
    if partial_pairs:
        first, second = _pick(rng, partial_pairs)
        paints[second] = _pick(rng, [c for c in paint_pool if c != paints[first]])
    for _ in range(20):
        markers, occupied = [None] * len(rectangles), set()
        # Internal loops have less exterior space; ties are freshly randomized.
        order = sorted(rng.permutation(len(rectangles)), key=lambda i: len(allowed[int(i)] & shell_near[int(i)]))
        for value in order:
            i = int(value)
            free = allowed[i] - occupied - _neighbors(occupied, shape)
            starts = sorted(free & shell_near[i])
            if not starts:
                break
            found = False
            for _ in range(90):
                path = [_pick(rng, starts)]
                target = _ri(rng, 2, 5)  # 1..4 wire cells followed by paint.
                for _ in range(target - 1):
                    choices = sorted(q for q in _neighbors({path[-1]}, shape) & free
                                     if q not in path and not any(_near(q, p) for p in path[:-1]))
                    if not choices:
                        break
                    path.append(_pick(rng, choices))
                if len(path) != target:
                    continue
                markers[i] = {"wire": set(path[:-1]), "source": path[-1],
                              "paint": paints[i], "color": rectangles[i][0]}
                occupied.update(path)
                found = True
                break
            if not found:
                break
        else:
            a = np.full(shape, bg, dtype=np.int8)
            a[4:shape[0]:7, 4:shape[1]:7] = coarse
            for marker in markers:
                for p in marker["wire"]:
                    a[p] = marker["color"]
                a[marker["source"]] = marker["paint"]
            hollow = _render(coarse, bg, rectangles, markers, fill=False)
            if not np.array_equal(_cavity_mask(hollow, bg), _intended_cavities(shape, rectangles)):
                continue
            return a, _render(coarse, bg, rectangles, markers)
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = _pick(rng, ("deeper", "combine"))
    preferred_bg = _ri(rng, 0, 9)
    roles = ["exact", "partial"] + ["random"] * (num_train - 2)
    rng.shuffle(roles)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            role = roles[index] if split == "train" else "test"
            for _ in range(80):
                bg = preferred_bg if rng.random() < 0.5 else _ri(rng, 0, 9)
                palette = [int(c) for c in rng.permutation(10) if c != bg]
                payload = palette[:_ri(rng, 1, 3)]
                paint_pool = [c for c in palette if c not in payload]
                coarse = _sample_coarse(rng, bg, payload, role, shift, split == "test")
                if coarse is None:
                    continue
                rectangles = _rectangles(coarse, bg)
                result = _place_markers(rng, coarse, bg, rectangles, paint_pool)
                if result is None:
                    continue
                a, b = result
                if role == "exact":
                    # Teach eight-neighbor attachment even with only two demos.
                    _, _, _, _, markers = _analyze(a)
                    if not any(any(_near(m["source"], p) and
                                   abs(m["source"][0] - p[0]) + abs(m["source"][1] - p[1]) == 2
                                   for p in m["wire"]) for m in markers):
                        continue
                key = (a.shape, a.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append({"input": a, "output": b})
                break
            else:
                raise RuntimeError("bounded construction attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _bridge_boxes(coarse, bg):
    """Geometry cross-check: paint cores and bridges, then take components."""
    shape = (7 * coarse.shape[0] + 2, 7 * coarse.shape[1] + 2)
    body = np.full(shape, bg, dtype=np.int8)
    rows = [_runs(row, bg) for row in coarse]
    for r, row in enumerate(coarse):
        for c, color in enumerate(row):
            if color == bg:
                continue
            y, x = 2 + 7 * r, 2 + 7 * c
            body[y:y + 5, x:x + 5] = color
            if c + 1 < len(row) and row[c + 1] == color:
                body[y:y + 5, x + 5:x + 7] = color
        if r + 1 < len(rows):
            for color, left, right in set(rows[r]) & set(rows[r + 1]):
                body[7 * r + 7:7 * r + 9, 2 + 7 * left:7 * right] = color
    boxes = []
    for group in _components(map(tuple, np.argwhere(body != bg)), D4):
        ys, xs = zip(*group)
        boxes.append((int(body[next(iter(group))]), min(ys), max(ys), min(xs), max(xs)))
        assert len(group) == (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)
    return sorted(boxes)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, training_diagonal = set(), [], False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            _, coarse, bg, rectangles, markers = _analyze(a)
            assert np.array_equal(b, _render(coarse, bg, rectangles, markers))
            hollow = _render(coarse, bg, rectangles, markers, fill=False)
            assert np.all(hollow[hollow != b] == bg)
            assert np.count_nonzero(hollow != b) == sum(
                (bottom - top - 1) * (right - left - 1)
                for top, bottom, left, right in map(_bounds, rectangles))
            # Preserve every visible tail, but replace the sparse samples.
            for marker in markers:
                assert all(b[p] == a[p] for p in marker["wire"] | {marker["source"]})
            assert sorted((rect[0], *_bounds(rect)) for rect in rectangles) == _bridge_boxes(coarse, bg)
            f = _features(coarse, bg)
            if split == "train":
                training.append((f, rectangles, markers))
                training_diagonal |= any(any(_near(m["source"], p) and
                                             abs(m["source"][0] - p[0]) + abs(m["source"][1] - p[1]) == 2
                                             for p in m["wire"]) for m in markers)
    assert any(f[0] and f[1] for f, _, _ in training), "demonstrate horizontal runs and exact merging"
    assert any(f[2] and any(markers[i]["paint"] != markers[j]["paint"]
                           for i, j in _partial_pairs(rectangles))
               for f, rectangles, markers in training), "demonstrate unequal spans with different owned fills"
    assert training_diagonal, "demonstrate diagonal endpoint connectivity"
    # Every test shifts either stack depth or combination, with both facts visible.
    deeper_train = max(f[3] for f, _, _ in training)
    separate_training = all(not (f[1] and f[2]) for f, _, _ in training)
    for pair in puzzle["test"]:
        _, coarse, bg, _, _ = _analyze(pair["input"])
        f = _features(coarse, bg)
        assert f[3] > deeper_train or (separate_training and f[1] and f[2])


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain small counts/hashes, never whole sampled corpora."""
    # Literal regressions for exact spans, unequal spans, interrupted returns,
    # color changes, and a single coarse row; no parent examples are embedded.
    regressions = (
        ([[2, 2, 0], [2, 2, 0], [0, 2, 2], [2, 2, 0]],
         [(2, 0, 2, 0, 2), (2, 2, 3, 1, 3), (2, 3, 4, 0, 2)]),
        ([[2, 2], [3, 3], [3, 3]], [(2, 0, 1, 0, 2), (3, 1, 3, 0, 2)]),
        ([[2, 0, 2]], [(2, 0, 1, 0, 1), (2, 0, 1, 2, 3)]),
    )
    for index, (rows, expected) in enumerate(regressions):
        coarse = np.array(rows, dtype=np.int8)
        assert _rectangles(coarse, 0) == expected
        pair = _place_markers(np.random.default_rng(100 + index), coarse, 0, expected, [4, 5, 6])
        assert pair is not None and np.array_equal(solve(pair[0]), pair[1])
    # This otherwise valid bent tail traps (1,3) against the loop's top side.
    # It must be excluded, rather than silently leaving an extra cavity empty.
    pocket = np.zeros((9, 9), dtype=np.int8)
    pocket[4, 4] = 2
    for p in ((1, 2), (0, 3), (1, 4)):
        pocket[p] = 2
    pocket[0, 5] = 4
    try:
        solve(pocket)
    except ValueError as error:
        assert "additional enclosed" in str(error)
    else:
        raise AssertionError("tail-created exterior pocket accepted")
    stats = Counter()
    shapes, coarse_shapes, backgrounds, layouts, fingerprints = set(), set(), set(), set(), set()
    widths, depths, counts, tail_lengths, directions, paints_per_pair = set(), set(), set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_hash(puzzle))
        assert _puzzle_hash(puzzle) == _puzzle_hash(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, coarse, bg, rectangles, markers = _analyze(pair["input"])
                assert np.array_equal(solve(a), pair["output"])
                shapes.add(a.shape)
                coarse_shapes.add(coarse.shape)
                backgrounds.add(bg)
                labels = {bg: 0}
                norm = tuple(labels.setdefault(int(c), len(labels)) for c in coarse.flat)
                layouts.add((coarse.shape, norm))
                counts.add(len(rectangles))
                widths.update(rect[4] - rect[3] for rect in rectangles)
                depths.update(rect[2] - rect[1] for rect in rectangles)
                paints_per_pair.add(len({m["paint"] for m in markers}))
                stats["pairs"] += 1
                f = _features(coarse, bg)
                stats["exact_pairs"] += f[1]
                stats["partial_pairs"] += f[2]
                stats["combined_pairs"] += f[1] and f[2]
                stats["zero_payload_pairs"] += bg != 0 and bool(np.any(coarse == 0))
                stats["blank_slot_pairs"] += bool(np.any(coarse == bg))
                stats["repeated_paint_pairs"] += len(markers) > len({m["paint"] for m in markers})
                for i, marker in enumerate(markers):
                    wire = marker["wire"]
                    tail_lengths.add(len(wire))
                    source = marker["source"]
                    neighbor = next(p for p in wire if _near(source, p))
                    dr, dc = int(neighbor[0] - source[0]), int(neighbor[1] - source[1])
                    directions.add((dr, dc))
                    # A four-neighbor-only route loses this endpoint.
                    stats["diagonal_endpoint_links"] += abs(dr) + abs(dc) == 2
                    shell = _shell(rectangles[i])
                    if _neighbors(wire, a.shape) & shell and not _neighbors(wire, a.shape, D4) & shell:
                        stats["diagonal_only_loop_attachments"] += 1
                # Omitting B gives the explicitly hollow intermediate image.
                assert not np.array_equal(_render(coarse, bg, rectangles, markers, False), pair["output"])
                # Omitting A's vertical merge restores a false horizontal wall.
                for rect in rectangles:
                    if rect[2] - rect[1] > 1:
                        y = 7 * (rect[1] + 1) - 1
                        x = 3 + 7 * rect[3]
                        assert pair["output"][y, x] != rect[0]
                        stats["merge_counterfactuals"] += 1
    assert len(fingerprints) == 42 and len(layouts) >= 100
    assert len(shapes) >= 9 and len(coarse_shapes) >= 9 and len(backgrounds) == 10
    assert widths >= {1, 2, 3, 4} and depths >= {1, 2, 3}
    assert len(counts) >= 5 and tail_lengths == {1, 2, 3, 4}
    assert len(directions) == 8 and max(paints_per_pair) >= 3
    for key in ("partial_pairs", "exact_pairs", "combined_pairs", "zero_payload_pairs",
                "blank_slot_pairs", "diagonal_endpoint_links", "diagonal_only_loop_attachments"):
        assert stats[key] >= 5, (key, stats[key])
    # All 30 legal explicit count combinations; the largest holds 14 pairs.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A partially unpainted cavity must not pass merely because its color set fits.
    bad = generate_puzzle(0, 2, 1)
    pair = bad["test"][0]
    _, _, bg, rectangles, _ = _analyze(pair["input"])
    top, _, left, _ = _bounds(rectangles[0])
    pair["output"][top + 1, left + 1] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial cavity fill accepted")
    return dict(stats, seeds=42, distinct_puzzles=len(fingerprints),
                grid_shapes=len(shapes), coarse_shapes=len(coarse_shapes),
                normalized_coarse_layouts=len(layouts), backgrounds=sorted(backgrounds),
                run_widths=sorted(widths), stack_depths=sorted(depths), loop_counts=sorted(counts),
                tail_lengths=sorted(tail_lengths), endpoint_directions=sorted(directions),
                paint_counts=sorted(paints_per_pair))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
