"""Sealed color catchments: plus centers compete by chamber path distance.

No files, global RNG, generation metadata, or demonstration labels enter solve.
All random loops and diagnostics are bounded; self_test retains summaries only.
"""
from collections import deque
import hashlib
import json

import numpy as np


BACKGROUND = 0
WALL = 5
PIGMENTS = (1, 2, 3, 4, 6, 7, 8, 9)
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _regions(mask):
    """Deterministic four-connected regions, including boundary components."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            cells, queue = [], deque([(r, c)])
            exterior = False
            while queue:
                x, y = queue.popleft()
                cells.append((x, y))
                exterior |= x in (0, h - 1) or y in (0, w - 1)
                for dx, dy in D4:
                    u, v = x + dx, y + dy
                    if 0 <= u < h and 0 <= v < w and mask[u, v] and not seen[u, v]:
                        seen[u, v] = True
                        queue.append((u, v))
            result.append((cells, bool(exterior)))
    return result


def _centers(a):
    m = np.zeros(a.shape, dtype=bool)
    if min(a.shape) >= 3:
        x = a[1:-1, 1:-1]
        m[1:-1, 1:-1] = ((x != BACKGROUND) & (x != WALL)
            & (x == a[:-2, 1:-1]) & (x == a[2:, 1:-1])
            & (x == a[1:-1, :-2]) & (x == a[1:-1, 2:]))
    return m


def _seed_map(a, kind):
    if kind == "pigment":
        return (a != BACKGROUND) & (a != WALL)
    if kind == "destructive":
        b = a.copy()
        seeds = np.zeros_like(a, dtype=bool)
        for r in range(1, len(a) - 1):
            for c in range(1, a.shape[1] - 1):
                points = [(r, c)] + [(r + dr, c + dc) for dr, dc in D4]
                if b[r, c] not in (BACKGROUND, WALL) and all(b[p] == b[r, c] for p in points):
                    seeds[r, c] = True
                    for p in points:
                        b[p] = BACKGROUND
        return seeds
    return _centers(a)


def _transform(a, *, seed_kind="centers", sealed_only=True, metric="path"):
    """Distance implementation; optional alternatives are validation ablations."""
    out = a.copy()
    seeds = _seed_map(a, seed_kind)
    for cells, exterior in _regions(a != WALL):
        if sealed_only and exterior:
            continue
        points = [p for p in cells if seeds[p]]
        if not points:
            continue
        colors = sorted({int(a[p]) for p in points})
        distances = []
        allowed = np.zeros_like(a, dtype=bool)
        rr, cc = np.array(cells).T
        allowed[rr, cc] = True
        for color in colors:
            starts = [p for p in points if a[p] == color]
            d = np.full(a.shape, 32767, dtype=np.int16)
            if metric == "manhattan":
                for r, c in starts:
                    d[rr, cc] = np.minimum(d[rr, cc], np.abs(rr - r) + np.abs(cc - c))
            else:
                queue = deque(starts)
                for p in starts:
                    d[p] = 0
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in D4:
                        x, y = r + dr, c + dc
                        if (0 <= x < len(a) and 0 <= y < a.shape[1]
                                and allowed[x, y] and d[x, y] == 32767):
                            d[x, y] = d[r, c] + 1
                            queue.append((x, y))
            distances.append(d[rr, cc])
        distances = np.array(distances)
        best = distances.min(axis=0)
        winners = distances == best
        assigned = np.array(colors, dtype=a.dtype)[distances.argmin(axis=0)]
        assigned[winners.sum(axis=0) != 1] = BACKGROUND
        out[rr, cc] = assigned
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill only sealed non-wall regions from their original monochrome plus centers.

    Pigment is traversable, including near misses and source arms. Distances
    start at centers, and equal nearest distances between COLORS leave zero.
    """
    return _transform(_grid(input_grid))


def _wave_audit(a):
    """Separate synchronous set-wave check, with direct plus tests.

    This is author-written corroboration, not an independent reference solver.
    Waves carry sets of colors, so equal arrivals of the same color do not tie.
    """
    out = a.copy()
    h, w = a.shape
    features = dict(active=0, inactive=0, open_sources=0, ties=0,
                    centers=0, max_colors=0, overlap=False, irregular=False,
                    unchanged_pigment=0)
    for cells, exterior in _regions(a != WALL):
        region = set(cells)
        seeds = {}
        for r, c in cells:
            color = int(a[r, c])
            if color in (BACKGROUND, WALL):
                continue
            if all((r + dr, c + dc) in region and a[r + dr, c + dc] == color for dr, dc in D4):
                seeds[(r, c)] = {color}
        features["centers"] += len(seeds)
        if exterior:
            features["open_sources"] += len(seeds)
            continue
        if not seeds:
            features["inactive"] += 1
            features["unchanged_pigment"] += sum(a[p] not in (BACKGROUND, WALL) for p in cells)
            continue
        features["active"] += 1
        features["max_colors"] = max(features["max_colors"], len({int(a[p]) for p in seeds}))
        rr, cc = np.array(cells).T
        features["irregular"] |= len(cells) < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
        arms = set()
        for r, c in seeds:
            cross = {(r, c)} | {(r + dr, c + dc) for dr, dc in D4}
            features["overlap"] |= bool(cross & arms)
            arms |= cross
        reached = {p: set(v) for p, v in seeds.items()}
        front = reached.copy()
        while front:
            upcoming = {}
            for (r, c), labels in front.items():
                for dr, dc in D4:
                    q = (r + dr, c + dc)
                    if q in region and q not in reached:
                        upcoming.setdefault(q, set()).update(labels)
            reached.update(upcoming)
            front = upcoming
        assert len(reached) == len(region)
        for p, labels in reached.items():
            out[p] = next(iter(labels)) if len(labels) == 1 else BACKGROUND
            features["ties"] += len(labels) > 1
    return out, features


def _cross(r, c):
    return [(r, c)] + [(r + dr, c + dc) for dr, dc in D4]


def _paint(a, points, color):
    for p in points:
        a[p] = color


def _orient(rng, a):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = np.fliplr(a)
    return a.copy()


def _patch(interior, marks, opened=False):
    """A one-cell eight-neighbor wall around a traversable room shape."""
    inside = np.pad(interior, 1)
    expanded = np.zeros_like(inside)
    padded = np.pad(inside, 1)
    h, w = inside.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            expanded |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    a = np.zeros(inside.shape, dtype=np.int8)
    a[expanded & ~inside] = WALL
    a[1:-1, 1:-1][interior] = marks[interior]
    if opened:
        # At least one interior cell lies on its top bounding row.
        cols = np.flatnonzero(interior[0])
        a[0, int(cols[len(cols) // 2]) + 1] = BACKGROUND
    return a


def _small_room(rng, colors, kind):
    h, w = (5, 5) if kind != "unseeded" else (3, int(rng.choice((3, 5))))
    interior = np.ones((h, w), dtype=bool)
    marks = np.zeros((h, w), dtype=np.int8)
    color = int(rng.choice(colors))
    r, c = h // 2, w // 2
    if kind == "unseeded":
        _paint(marks, [(r, c - 1), (r, c), (r, c + 1), (r + 1, c)], color)
    else:
        _paint(marks, _cross(r, c), color)
        if rng.random() < .5:
            marks[r, -1] = color
    return _orient(rng, _patch(interior, marks, opened=kind == "open"))


def _teaching_room(rng, colors, kind):
    if kind == "contrast":
        h = int(rng.choice((7, 9, 11)))
        w = int(rng.choice((9, 11, 13)))
        inside = np.ones((h, w), dtype=bool)
        marks = np.zeros((h, w), dtype=np.int8)
        _paint(marks, _cross(h // 2, 2), colors[0])
        # Both shared-arm centers influence the competition boundary. Losing
        # either one in a destructive scan changes a different part of it.
        _paint(marks, _cross(h // 2 + 2, 2), colors[0])
        _paint(marks, _cross(h // 2, w - 3), colors[1])
        # A third pigment makes a T, not a source. Its central column ties.
        c = w // 2
        _paint(marks, [(1, c - 1), (1, c), (1, c + 1), (2, c)], colors[2])
        return _orient(rng, _patch(inside, marks))
    h = int(rng.choice((9, 11, 13)))
    w = int(rng.choice((11, 13, 15)))
    inside = np.ones((h, w), dtype=bool)
    mid = w // 2
    inside[:h - 3, mid:mid + int(rng.integers(1, 3))] = False
    marks = np.zeros((h, w), dtype=np.int8)
    # Shared-arm centers establish simultaneous recognition and same-color union.
    for c in (1, 3):
        _paint(marks, _cross(h - 3, c), colors[1])
    _paint(marks, _cross(2, w - 3), colors[2])
    # A long arm is pigment but is not a row of source centers.
    marks[3:h - 1, w - 3] = colors[2]
    _paint(marks, [(1, 1), (1, 2), (1, 3), (2, 2)], colors[0])
    return _orient(rng, _patch(inside, marks))


def _room_shape(rng, h, w):
    m = np.ones((h, w), dtype=bool)
    style = int(rng.integers(5))
    if style == 1:
        # A baffle forces a long detour but leaves a wide connected bridge.
        c = w // 2
        m[:h - int(rng.integers(3, 5)), c:c + int(rng.integers(1, 3))] = False
    elif style == 2:
        # An elbow, with independently sized legs.
        m[:h // 2, w // 2:] = False
    elif style == 3:
        # A waist between two lobes, producing bottleneck geodesics.
        mid = h // 2
        left = int(rng.integers(1, max(2, w // 2 - 1)))
        m[mid:mid + 1, :left] = False
        m[mid:mid + 1, left + 3:] = False
    elif style == 4:
        # Stepped corners vary the cavity silhouette without adding islands.
        m[:int(rng.integers(1, 4)), :int(rng.integers(1, 4))] = False
        m[-int(rng.integers(1, 4)):, -int(rng.integers(1, 4)):] = False
    return _orient(rng, m)


def _ordinary_room(rng, colors, color_count, larger=False):
    for _ in range(40):
        h = int(rng.integers(9 if larger else 7, 18 if larger else 14))
        w = int(rng.integers(11 if larger else 9, 20 if larger else 16))
        interior = _room_shape(rng, h, w)
        # At least five free cells are necessary, but the geometry may be irregular.
        valid = np.zeros_like(interior)
        valid[1:-1, 1:-1] = (interior[1:-1, 1:-1] & interior[:-2, 1:-1]
            & interior[2:, 1:-1] & interior[1:-1, :-2] & interior[1:-1, 2:])
        candidates = [tuple(map(int, p)) for p in np.argwhere(valid)]
        rng.shuffle(candidates)
        centers = []
        for p in candidates:
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 5 for q in centers):
                centers.append(p)
                if len(centers) == color_count:
                    break
        if len(centers) != color_count:
            continue
        marks = np.zeros_like(interior, dtype=np.int8)
        active_colors = list(map(int, rng.choice(colors, size=color_count, replace=False)))
        for (r, c), color in zip(centers, active_colors):
            _paint(marks, _cross(r, c), color)
        # Independent extensions and same-color overlapping motifs.
        for (r, c), color in zip(centers, active_colors):
            if rng.random() < .45:
                dr, dc = D4[int(rng.integers(4))]
                for distance in range(2, int(rng.integers(3, 7))):
                    x, y = r + dr * distance, c + dc * distance
                    if not (0 <= x < len(marks) and 0 <= y < marks.shape[1]
                            and interior[x, y] and marks[x, y] in (BACKGROUND, color)):
                        break
                    marks[x, y] = color
            if rng.random() < .35:
                dr, dc = D4[int(rng.integers(4))]
                p = (r + 2 * dr, c + 2 * dc)
                cross = _cross(*p)
                if all(0 <= x < len(marks) and 0 <= y < marks.shape[1]
                       and interior[x, y] and marks[x, y] in (BACKGROUND, color) for x, y in cross):
                    _paint(marks, cross, color)
        # Non-source camouflage: choose a blank 3x3 patch for an independent T.
        blanks = []
        for r in range(1, len(marks) - 1):
            for c in range(1, marks.shape[1] - 1):
                if interior[r - 1:r + 2, c - 1:c + 2].all() and not marks[r - 1:r + 2, c - 1:c + 2].any():
                    blanks.append((r, c))
        if blanks and rng.random() < .8:
            r, c = blanks[int(rng.integers(len(blanks)))]
            _paint(marks, [(r, c - 1), (r, c), (r, c + 1), (r + 1, c)], int(rng.choice(colors)))
        result = _patch(interior, marks)
        actual = {int(result[p]) for p in zip(*np.nonzero(_centers(result)))}
        if len(actual) == color_count:
            return _orient(rng, result)
    # Guaranteed capacity fallback; still a genuine member of the same rule.
    interior = np.ones((11, 17), dtype=bool)
    marks = np.zeros((11, 17), dtype=np.int8)
    for (r, c), color in zip(((3, 3), (7, 8), (3, 13)), colors[:color_count]):
        _paint(marks, _cross(r, c), color)
    return _orient(rng, _patch(interior, marks))


def _shelf(patches):
    """Capacity fallback, leaving at least three free rows on a 30-row canvas."""
    main, *extra = sorted(patches, key=lambda p: -p.size)
    if main.shape[0] > main.shape[1]:
        main = np.rot90(main)
    # Main shorter axis <=19; at most two auxiliary patches have axes <=7.
    h = main.shape[0] + (1 + max(p.shape[0] for p in extra) if extra else 0)
    w = max(main.shape[1], sum(p.shape[1] + 1 for p in extra) - 1)
    assert h <= 27 and w <= 30
    grid = np.zeros((h, w), dtype=np.int8)
    grid[:main.shape[0], :main.shape[1]] = main
    c = 0
    for p in extra:
        r = main.shape[0] + 1
        grid[r:r + p.shape[0], c:c + p.shape[1]] = p
        c += p.shape[1] + 1
    return grid


def _pack(rng, patches):
    """Random non-slot placement with a finite, capacity-proven shelf fallback."""
    largest = max(p.size for p in patches)
    area = sum(p.size for p in patches)
    for attempt in range(24):
        low = max(9, int(np.sqrt(area)) + (attempt // 8))
        h = int(rng.integers(min(low, 28), 31))
        w = int(rng.integers(min(low, 28), 31))
        if h * w < area + largest // 4:
            continue
        grid = np.zeros((h, w), dtype=np.int8)
        occupied = np.zeros((h, w), dtype=bool)
        for p in sorted(patches, key=lambda p: -p.size):
            ph, pw = p.shape
            options = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)
                       if not occupied[r:r + ph, c:c + pw].any()]
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            grid[r:r + ph, c:c + pw] = p
            occupied[max(0, r - 1):min(h, r + ph + 1), max(0, c - 1):min(w, c + pw + 1)] = True
        else:
            return _orient(rng, grid)
    return _orient(rng, _shelf(patches))


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.choice(PIGMENTS, size=3, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    # Test shifts to three competing colors, all individually active in training.
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for attempt in range(24):
                if split == "train" and index == 0:
                    patches = [_teaching_room(rng, colors, "contrast"), _small_room(rng, colors, "open")]
                elif split == "train" and index == 1:
                    patches = [_teaching_room(rng, colors, "detour"), _small_room(rng, colors, "unseeded")]
                else:
                    patches = [_ordinary_room(rng, colors, 3 if split == "test" else 2,
                                               larger=split == "test" and rng.random() < .5)]
                    for _ in range(int(rng.integers(3))):
                        patches.append(_small_room(rng, colors, str(rng.choice(("open", "unseeded", "single")))))
                inp = _pack(rng, patches)
                if _key(inp) not in seen:
                    break
            else:
                # Fifteen distinct inert wall bars exceed thirteen prior inputs.
                # Two blank rows separate them from all teaching geometry, so
                # they cannot seal an open door or change any path in a cavity.
                base = _shelf(patches)
                canvas = np.zeros((30, 30), dtype=np.int8)
                canvas[:len(base), :base.shape[1]] = base
                for length in range(1, 16):
                    inp = canvas.copy()
                    inp[29, :length] = WALL
                    if _key(inp) not in seen:
                        break
                else:
                    raise AssertionError("fifteen candidates exceed prior input count")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": solve(inp)})
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    ablations = dict(plus=False, topology=False, distance=False, simultaneous=False)
    train_colors, test_colors = set(), set()
    pigment_palette = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, f = _wave_audit(a)
            assert np.array_equal(expected, b), "incorrect catchment assignment"
            assert np.array_equal(solve(a), b)
            assert np.all(b[a == WALL] == WALL), "walls changed"
            pigment_palette.update(int(c) for c in np.unique(a) if c not in (BACKGROUND, WALL))
            assert f["active"] >= 1 and np.any(a != b)
            records[split].append(f)
            centers = _centers(a)
            source_union = centers.copy()
            source_union[1:] |= centers[:-1]
            source_union[:-1] |= centers[1:]
            source_union[:, 1:] |= centers[:, :-1]
            source_union[:, :-1] |= centers[:, 1:]
            assert np.array_equal(b[source_union], a[source_union]), "valid plus pigment changed"
            active_colors = set()
            for cells, exterior in _regions(a != WALL):
                source_colors = {int(a[p]) for p in cells if centers[p]}
                if exterior or not source_colors:
                    assert all(a[p] == b[p] for p in cells), "inactive/exterior region changed"
                else:
                    active_colors.update(source_colors)
                    assert all(int(b[p]) in source_colors | {BACKGROUND} for p in cells)
                    if len(source_colors) == 1:
                        assert all(b[p] == next(iter(source_colors)) for p in cells)
            (train_colors if split == "train" else test_colors).update(active_colors)
            if split == "train":
                ablations["plus"] |= not np.array_equal(b, _transform(a, seed_kind="pigment"))
                ablations["topology"] |= not np.array_equal(b, _transform(a, sealed_only=False))
                ablations["distance"] |= not np.array_equal(b, _transform(a, metric="manhattan"))
                ablations["simultaneous"] |= not np.array_equal(b, _transform(a, seed_kind="destructive"))
    assert all(ablations.values()), "missing training counterexample: " + str(ablations)
    assert len(pigment_palette) == 3 and train_colors == pigment_palette
    assert test_colors <= train_colors, "test pigment must be an active training source"
    tr = records["train"]
    assert any(f["ties"] for f in tr), "ties must be demonstrated"
    assert any(f["inactive"] and f["unchanged_pigment"] for f in tr), "show unseeded sealed chamber"
    assert any(f["overlap"] for f in tr), "overlapping source motifs must be demonstrated"
    assert max(f["max_colors"] for f in tr) == 2
    assert all(f["max_colors"] == 3 for f in records["test"]), "missing color competition shift"
    return True


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed diversity, count API, semantic and mutation regressions."""
    shapes, geometry, fingerprints, palettes = set(), set(), set(), set()
    cavity_shapes, source_counts, chamber_combinations = set(), set(), set()
    totals = dict(pairs=0, irregular=0, overlap=0, ties=0, open_sources=0, inactive=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        palette = set()
        for split in p:
            for pair in p[split]:
                a = pair["input"]
                _, f = _wave_audit(a)
                shapes.add(a.shape)
                geometry.add(hashlib.sha256((a == WALL).tobytes() + bytes(a.shape)).hexdigest())
                palette.update(int(c) for c in np.unique(a) if c not in (BACKGROUND, WALL))
                centers = _centers(a)
                sealed = active = exterior_sources = 0
                for cells, exterior in _regions(a != WALL):
                    n = int(sum(centers[q] for q in cells))
                    if exterior:
                        exterior_sources += n
                        continue
                    sealed += 1
                    active += bool(n)
                    source_counts.add(n)
                    rr, cc = np.array(cells).T
                    mask = np.zeros((rr.max() - rr.min() + 1, cc.max() - cc.min() + 1), np.uint8)
                    mask[rr - rr.min(), cc - cc.min()] = 1
                    cavity_shapes.add(_key(mask))
                chamber_combinations.add((sealed, active, exterior_sources))
                totals["pairs"] += 1
                for k in totals:
                    if k != "pairs":
                        totals[k] += int(f[k])
        palettes.add(tuple(sorted(palette)))
        if seed in (0, 17, 41):
            assert _fingerprint(p) == _fingerprint(generate_puzzle(seed))
    assert len(fingerprints) == 42 and len(shapes) >= 45
    assert len(geometry) >= 180 and len(palettes) >= 20
    assert len(cavity_shapes) >= 100 and {0, 1, 2, 3, 4} <= source_counts
    assert len(chamber_combinations) >= 8
    assert totals["irregular"] >= 42 and totals["overlap"] >= 42
    assert totals["ties"] > 100 and totals["open_sources"] >= 42 and totals["inactive"] >= 42
    # Every legal explicit request. No stored grids or seed-batch payloads.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(14000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Fully explicit pixel regression: a sealed 5x9 room with symmetric seeds.
    marks = np.zeros((5, 9), dtype=np.int8)
    _paint(marks, _cross(2, 2), 2)
    _paint(marks, _cross(2, 6), 7)
    a = _patch(np.ones((5, 9), dtype=bool), marks)
    expected = np.full(a.shape, WALL, dtype=np.int8)
    expected[1:-1, 1:5] = 2
    expected[1:-1, 5] = BACKGROUND
    expected[1:-1, 6:-1] = 7
    assert np.array_equal(solve(a), expected)
    # Opening one wall cell makes the entire region exterior; source arms
    # remain traversable, so they cannot plug the door.
    opened = a.copy()
    opened[0, 5] = BACKGROUND
    assert np.array_equal(solve(opened), opened)
    # A T and a boundary-clipped plus are both insufficient.
    near = np.array([[0, 2, 0], [2, 2, 2], [0, 0, 0]], dtype=np.int8)
    assert not _centers(near).any()
    # Same-color ties fill continuously rather than drawing a zero seam.
    a[a == 7] = 2
    out = solve(a)
    assert np.all(out[1:-1, 1:-1] == 2)
    # Validator must reject a changed wall and a wrong catchment pixel.
    for wall in (False, True):
        p = generate_puzzle(811, 2, 1)
        pair = p["test"][0]
        coords = np.argwhere(pair["input"] == WALL if wall else pair["input"] != pair["output"])
        r, c = coords[0]
        pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted pixel corruption")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shapes": len(shapes),
            "wall_geometries": len(geometry), "palettes": len(palettes),
            "cropped_cavity_shapes": len(cavity_shapes), "source_counts": sorted(source_counts),
            "chamber_combinations": len(chamber_combinations), **totals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
