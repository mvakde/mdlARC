"""Convex/concave contour labels transported by a bottom-anchored row shear.

Only NumPy and the Python standard library are used. No source files, random
state, or training examples are consulted by solve(). All generation diagnostics
are streamed in memory; running this module writes no files.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json

import numpy as np

CONVEX = 4
CONCAVE = 2
INKS = (0, 1, 3, 5, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 815993232, 274096848, 877928276, 338473241, 636222256,
              311789737, 240180066, 854988421, 54104934, 147950274,
              399287428, 61726683, 167185467, 277217167, 508038907,
              248867724, 666361403, 277059692, 947820799, 135772358,
              57791976, 56411583, 937436425, 405333990, 105591008,
              169439512, 16693939, 580594729, 667820691, 395415848,
              843223358, 318938120, 959055325, 155546846, 712820782,
              760040417, 311878065, 57579675, 422988304, 244405949, 342932257)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid of colors 0..9")
    return np.asarray(a, dtype=np.int8)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _neighbors(p, cells):
    return sorted((p[0] + dr, p[1] + dc) for dr, dc in DIRECTIONS
                  if (p[0] + dr, p[1] + dc) in cells)


def _inside_cells(cells):
    """Padded flood fill, including for contours touching an input edge."""
    lo_r = min(r for r, _ in cells) - 1
    lo_c = min(c for _, c in cells) - 1
    hi_r = max(r for r, _ in cells) + 1
    hi_c = max(c for _, c in cells) + 1
    exterior, queue = {(lo_r, lo_c)}, deque([(lo_r, lo_c)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            p = (r + dr, c + dc)
            if (lo_r <= p[0] <= hi_r and lo_c <= p[1] <= hi_c
                    and p not in cells and p not in exterior):
                exterior.add(p)
                queue.append(p)
    return {(r, c) for r in range(lo_r + 1, hi_r)
            for c in range(lo_c + 1, hi_c)
            if (r, c) not in cells and (r, c) not in exterior}


def _contour(cells):
    """Trace a simple cycle and classify signed turns relative to its area."""
    if len(cells) < 8 or any(len(_neighbors(p, cells)) != 2 for p in cells):
        raise ValueError("Contours must be simple one-cell-wide cycles")
    start = min(cells)
    path, seen = [start], {start}
    previous, current = start, _neighbors(start, cells)[0]
    while current != start:
        if current in seen:
            raise ValueError("Repeated contour cell")
        path.append(current)
        seen.add(current)
        following = [p for p in _neighbors(current, cells) if p != previous]
        previous, current = current, following[0]
    if seen != cells or not _inside_cells(cells):
        raise ValueError("A contour must enclose an interior and be one cycle")
    area = sum(c * path[(i + 1) % len(path)][0]
               - r * path[(i + 1) % len(path)][1]
               for i, (r, c) in enumerate(path))
    if area == 0:
        raise ValueError("Degenerate contour")
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % len(path)]
        cross = (c - pc) * (nr - r) - (r - pr) * (nc - c)
        classes[(r, c)] = (0 if cross == 0 else
                           1 if cross * area > 0 else 2)
    if sum(v == 1 for v in classes.values()) - sum(v == 2 for v in classes.values()) != 4:
        raise ValueError("Invalid orthogonal polygon turn balance")
    return {"cells": cells, "path": path, "classes": classes,
            "concaves": sum(v == 2 for v in classes.values())}


def _analyze(value):
    a = _grid(value)
    h, w = a.shape
    if h + w - 1 > 30:
        raise ValueError("The widened canvas would exceed 30 columns")
    colors, counts = np.unique(a, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    if max(counts) * 2 <= a.size or background in (CONVEX, CONCAVE):
        raise ValueError("Input needs a strict-majority non-label background")
    if np.any((a == CONVEX) | (a == CONCAVE)):
        raise ValueError("The two labels are reserved for output bends")
    cells = {tuple(map(int, p)) for p in np.argwhere(a != background)}
    contours = [_contour(component) for component in _components(cells)]
    if not contours:
        raise ValueError("Input must contain a contour")
    for info in contours:
        if len({int(a[p]) for p in info["cells"]}) != 1:
            raise ValueError("Each contour must initially be monochrome")
    for i, info in enumerate(contours):
        others = set().union(*(c["cells"] for c in contours[i + 1:]))
        if any((r + dr, c + dc) in others for r, c in info["cells"]
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            raise ValueError("Distinct contours must have a blank-cell gap")
    return a, background, contours


def _label(a, contours):
    b = a.copy()
    for info in contours:
        for p, kind in info["classes"].items():
            if kind:
                b[p] = CONVEX if kind == 1 else CONCAVE
    return b


def _shear(a, background):
    h, w = a.shape
    b = np.full((h, w + h - 1), background, dtype=np.int8)
    for r in range(h):
        b[r, h - 1 - r:h - 1 - r + w] = a[r]
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Classify in input geometry, then copy whole labeled row vectors."""
    a, background, contours = _analyze(input_grid)
    return _shear(_label(a, contours), background)


def _tile_boundary(tiles):
    """Boundary of a union of unit tiles; reject holes and pinched vertices."""
    edges = {}
    for r, c in sorted(tiles):
        faces = (((r - 1, c), (r, c), (r, c + 1)),
                 ((r, c + 1), (r, c + 1), (r + 1, c + 1)),
                 ((r + 1, c), (r + 1, c + 1), (r + 1, c)),
                 ((r, c - 1), (r + 1, c), (r, c)))
        for neighbor, start, end in faces:
            if neighbor not in tiles:
                if start in edges:
                    return None
                edges[start] = end
    if set(edges) != set(edges.values()):
        return None
    start = min(edges)
    path, seen, current = [], set(), start
    while current not in seen:
        seen.add(current)
        path.append(current)
        current = edges[current]
    if current != start or len(path) != len(edges):
        return None
    return [p for i, p in enumerate(path)
            if (p[0] - path[i - 1][0], p[1] - path[i - 1][1]) !=
            (path[(i + 1) % len(path)][0] - p[0],
             path[(i + 1) % len(path)][1] - p[1])]


def _rasterize(vertices, rng):
    lo_r, hi_r = min(r for r, _ in vertices), max(r for r, _ in vertices)
    lo_c, hi_c = min(c for _, c in vertices), max(c for _, c in vertices)
    # Unequal spacings produce compact shoulders, long runs and narrow necks.
    rows = np.r_[0, np.cumsum(rng.choice((1, 2, 3), hi_r - lo_r,
                                       p=(.16, .63, .21)))]
    cols = np.r_[0, np.cumsum(rng.choice((1, 2, 3), hi_c - lo_c,
                                       p=(.16, .63, .21)))]
    corners = [(int(rows[r - lo_r]), int(cols[c - lo_c])) for r, c in vertices]
    cells = set()
    for p, q in zip(corners, corners[1:] + corners[:1]):
        dr, dc = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
        length = abs(q[0] - p[0]) + abs(q[1] - p[1])
        cells.update((p[0] + k * dr, p[1] + k * dc) for k in range(length))
    try:
        info = _contour(cells)
    except ValueError:
        return None
    mask = np.zeros((max(r for r, _ in cells) + 1,
                     max(c for _, c in cells) + 1), dtype=bool)
    for p in cells:
        mask[p] = True
    k = int(rng.integers(4))
    mask = np.rot90(mask, k)
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return np.ascontiguousarray(mask), info["concaves"]


def _sample_shape(rng, max_h, max_w, low, high):
    """Grow small polyominoes, then use their contour rather than their fill."""
    for _ in range(180):
        if high == 0:
            h = int(rng.integers(3, min(max_h, 12) + 1))
            w = int(rng.integers(3, min(max_w, 12) + 1))
            mask = np.ones((h, w), dtype=bool)
            mask[1:-1, 1:-1] = False
            return mask
        tiles = {(0, 0)}
        target = int(rng.integers(3, 8) if high <= 3 else rng.integers(6, 16))
        for _ in range(target - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in tiles
                               for dr, dc in DIRECTIONS} - tiles)
            # Each growth step can extend any exposed side.
            tiles.add(frontier[int(rng.integers(len(frontier)))])
        vertices = _tile_boundary(tiles)
        if vertices is None:
            continue
        result = _rasterize(vertices, rng)
        if result is not None:
            mask, concaves = result
            if mask.shape[0] <= max_h and mask.shape[1] <= max_w and low <= concaves <= high:
                return mask
    return None


def _place(rng, shapes, h, w, mode):
    """Place by actual occupied cells; bounding boxes may overlap or nest."""
    blocked = np.zeros((h, w), dtype=bool)
    positions = []
    for i, mask in enumerate(shapes):
        sh, sw = mask.shape
        options = []
        for r in range(h - sh + 1):
            for c in range(w - sw + 1):
                if mode == "margins" and i == 0 and not (r >= 1 and c >= 1 and r + sh < h and c + sw < w):
                    continue
                if mode == "bottom" and i == 0 and r + sh != h:
                    continue
                if not np.any(blocked[r:r + sh, c:c + sw] & mask):
                    options.append((r, c))
        if not options:
            return None
        r, c = options[int(rng.integers(len(options)))]
        positions.append((r, c))
        for sr, sc in np.argwhere(mask):
            rr, cc = r + int(sr), c + int(sc)
            blocked[max(0, rr - 1):min(h, rr + 2),
                    max(0, cc - 1):min(w, cc + 2)] = True
    return positions


def _pair(rng, background, foregrounds, mode, test, forbidden_shape=None):
    for _ in range(140):
        if mode == "margins":
            h = int(rng.integers(9, 16))
            w = int(rng.integers(9, min(19, 31 - h) + 1))
            count = 1
        elif mode == "bottom":
            h = int(rng.integers(10, 18))
            w = int(rng.integers(10, min(19, 31 - h) + 1))
            count = int(rng.integers(2, 4))
        else:
            h = int(rng.integers(6, 26))
            w = int(rng.integers(max(6, 15 - h), min(25, 31 - h) + 1))
            count = int(rng.choice((1, 2, 3, 4), p=(.30, .38, .24, .08)))
        if (h, w) == forbidden_shape:
            continue
        max_h = h - (2 if mode == "margins" else 0)
        max_w = w - (2 if mode == "margins" else 0)
        shapes = []
        first_low = 4 if test else 1 if mode in ("margins", "bottom") else 0
        first_high = 10 if test else 3
        for i in range(count):
            if i == 0:
                low, high = first_low, first_high
            elif rng.random() < .68:
                low = high = 0
            else:
                low, high = 1, 3
            if high == 0:
                # Multiple compact loops remain possible on a modest canvas.
                mh, mw = min(max_h, 7), min(max_w, 7)
            else:
                mh, mw = max_h, max_w
            mask = _sample_shape(rng, mh, mw, low, high)
            if mask is None:
                break
            shapes.append(mask)
        if len(shapes) != count:
            continue
        positions = _place(rng, shapes, h, w, mode)
        if positions is None:
            continue
        a = np.full((h, w), background, dtype=np.int8)
        for mask, (r, c) in zip(shapes, positions):
            color = int(rng.choice(foregrounds))
            a[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = color
        if np.count_nonzero(a == background) * 2 <= a.size:
            continue
        return {"input": a, "output": solve(a)}
    raise RuntimeError("Bounded contour construction exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(INKS)]
    background = palette[0]
    train_colors, test_colors = palette[1:5], palette[1:]
    puzzle, seen = {"train": [], "test": []}, set()
    first_shape = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = ("margins" if split == "train" and i == 0 else
                    "bottom" if split == "train" and i == 1 else "free")
            for _ in range(16):
                pair = _pair(rng, background,
                             train_colors if split == "train" else test_colors,
                             mode, split == "test",
                             first_shape if mode == "bottom" else None)
                a = pair["input"]
                key = (a.shape, a.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and i == 0:
                        first_shape = a.shape
                    break
            else:
                raise RuntimeError("Unable to sample distinct inputs")
    return puzzle


def _point_inside(path, row, col):
    # Ray casting is used only by validation, not by the solver's turn rule.
    crossings = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:] + path[:1]):
        if (r1 > row) != (r2 > row):
            at_col = c1 + (row - r1) * (c2 - c1) / (r2 - r1)
            crossings += at_col > col
    return crossings % 2 == 1


def _local_code(p, cells):
    return tuple(i for i, (dr, dc) in enumerate(DIRECTIONS)
                 if (p[0] + dr, p[1] + dc) in cells)


def validate_generated_puzzle(puzzle):
    """Check geometry, independent bend angles, exact transport and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, backgrounds, local_evidence = set(), set(), {}
    train_classes, shifted_labels, unchanged_straights = set(), 0, 0
    train_shapes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, bg, contours = _analyze(pair["input"])
            b = _grid(pair["output"])
            h, w = a.shape
            assert b.shape == (h, h + w - 1)
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            backgrounds.add(bg)
            expected_labels = {}
            for info in contours:
                convex_count = concave_count = 0
                for p, kind in info["classes"].items():
                    if kind:
                        # Interior quadrant count: 1 at convex, 3 at concave.
                        q = sum(_point_inside(info["path"], p[0] + dr, p[1] + dc)
                                for dr in (-.25, .25) for dc in (-.25, .25))
                        assert q in (1, 3)
                        independent_kind = 1 if q == 1 else 2
                        assert kind == independent_kind
                        convex_count += q == 1
                        concave_count += q == 3
                        expected_labels[p] = CONVEX if q == 1 else CONCAVE
                        if split == "train":
                            train_classes.add(kind)
                            code = _local_code(p, info["cells"])
                            local_evidence.setdefault(code, set()).add(kind)
                            shifted_labels += p[0] < h - 1
                    elif split == "train":
                        unchanged_straights += 1
                assert convex_count - concave_count == 4
            # Cellwise equation checks payload AND newly added blank cells.
            for r in range(h):
                start = h - 1 - r
                assert np.all(b[r, :start] == bg)
                assert np.all(b[r, start + w:] == bg)
                for c in range(w):
                    assert b[r, start + c] == expected_labels.get((r, c), int(a[r, c]))
            # Both ablations differ from the actual answer on every irregular pair.
            assert not np.array_equal(b, _shear(a, bg))
            unshifted = np.full_like(b, bg)
            unshifted[:, :w] = _label(a, contours)
            assert not np.array_equal(b, unshifted)
            assert np.count_nonzero(b == CONVEX) == sum(v == CONVEX for v in expected_labels.values())
            assert np.count_nonzero(b == CONCAVE) == sum(v == CONCAVE for v in expected_labels.values())
            if split == "train":
                train_shapes.add(a.shape)
    assert len(backgrounds) == 1
    assert train_classes == {1, 2} and shifted_labels and unchanged_straights
    assert any(kinds == {1, 2} for kinds in local_evidence.values())
    assert len(train_shapes) >= 2
    first = np.asarray(puzzle["train"][0]["input"])
    second = np.asarray(puzzle["train"][1]["input"])
    bg = next(iter(backgrounds))
    assert np.all(first[-1] == bg) and np.any(second[-1] != bg)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 exact host seeds, streamed; no logs or payload files are retained."""
    # Hand-constructed domain probes are not drawn from the tile sampler.
    # Their expected bend locations are given explicitly, then transported by
    # list concatenation rather than the solver's allocation/slice operation.
    probes = []
    for h, w, r, c, sh, sw, bg, ink in (
            (3, 6, 0, 0, 3, 3, 0, 8),
            (28, 3, 25, 0, 3, 3, 7, 0)):
        a = np.full((h, w), bg, dtype=np.int8)
        a[r:r + sh, c] = a[r:r + sh, c + sw - 1] = ink
        a[r, c:c + sw] = a[r + sh - 1, c:c + sw] = ink
        labeled = a.copy()
        for rr in (r, r + sh - 1):
            for cc in (c, c + sw - 1):
                labeled[rr, cc] = CONVEX
        probes.append((a, labeled, bg))
    # Nested rectangles: both loops' corners are convex, including the inner
    # loop; total-scene inside/outside parity must not reverse its meaning.
    a = np.full((13, 13), 0, dtype=np.int8)
    labeled = a.copy()
    for lo, hi, ink in ((0, 12, 8), (4, 8, 3)):
        a[lo:hi + 1, lo] = a[lo:hi + 1, hi] = ink
        a[lo, lo:hi + 1] = a[hi, lo:hi + 1] = ink
    labeled[:] = a
    for lo, hi in ((0, 12), (4, 8)):
        for rr in (lo, hi):
            for cc in (lo, hi):
                labeled[rr, cc] = CONVEX
    probes.append((a, labeled, 0))
    # L contour with a length-one edge between adjacent unlike bends.
    a = np.full((8, 9), 0, dtype=np.int8)
    corners = [(1, 1), (1, 6), (3, 6), (3, 5), (6, 5), (6, 1)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        dr, dc = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
        for k in range(abs(q[0] - p[0]) + abs(q[1] - p[1])):
            a[p[0] + dr * k, p[1] + dc * k] = 8
    labeled = a.copy()
    for p in corners:
        labeled[p] = CONCAVE if p == (3, 5) else CONVEX
    for k in range(4):
        for reflected in (False, True):
            aa, bb = np.rot90(a, k), np.rot90(labeled, k)
            if reflected:
                aa, bb = np.fliplr(aa), np.fliplr(bb)
            probes.append((aa.copy(), bb.copy(), 0))
    for a, labeled, bg in probes:
        h, w = a.shape
        expected = np.array([[bg] * (h - 1 - r) + list(map(int, labeled[r]))
                             + [bg] * r for r in range(h)], dtype=np.int8)
        before = a.copy()
        assert np.array_equal(solve(a), expected)
        assert np.array_equal(a, before), "solve mutated its input"
    signatures, shapes, geometries, counts, concaves, backgrounds = (set() for _ in range(6))
    turn_words, foreground_palettes = set(), set()
    adjacent = edge = blank_rows = boundary = multicolor = 0
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        training_max = 0
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, bg, contours = _analyze(pair["input"])
                assert np.array_equal(solve(a.copy()), pair["output"])
                shapes.add(a.shape)
                backgrounds.add(bg)
                counts.add(len(contours))
                foreground_palettes.add(tuple(sorted(set(map(int, np.unique(a))) - {bg})))
                blank_rows += any(np.all(row == bg) for row in a)
                edge += any(np.any(side != bg) for side in (a[0], a[-1], a[:, 0], a[:, -1]))
                boundary += sum(a.shape) - 1 == 30
                multicolor += len(np.unique(a)) >= 3
                for info in contours:
                    n = info["concaves"]
                    concaves.add(n)
                    if split == "train":
                        training_max = max(training_max, n)
                    path = info["path"]
                    kinds = [info["classes"][p] for p in path]
                    turns = [k for k in kinds if k]
                    rotations = [tuple(turns[i:] + turns[:i]) for i in range(len(turns))]
                    rev = turns[::-1]
                    rotations += [tuple(rev[i:] + rev[:i]) for i in range(len(rev))]
                    turn_words.add(min(rotations))
                    adjacent += any(kinds[i] and kinds[(i + 1) % len(kinds)] for i in range(len(kinds)))
                    lo_r, lo_c = min(r for r, _ in path), min(c for _, c in path)
                    geometries.add(tuple(sorted((r - lo_r, c - lo_c) for r, c in path)))
                if split == "test":
                    assert max(info["concaves"] for info in contours) > training_max
                pairs += 1
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(geometries) >= 100
    assert {1, 2, 3}.issubset(counts) and {0, 1, 2, 3, 4}.issubset(concaves)
    assert len(backgrounds) >= 6 and len(foreground_palettes) >= 12 and len(turn_words) >= 12
    assert adjacent and edge and blank_rows and boundary and multicolor
    # Every explicit count combination, plus type/range rejection.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(12000 + 10 * train + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
            validate_generated_puzzle(p)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                        (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # Mutation checks establish that validation rejects wrong labels and phase.
    p = generate_puzzle(0, 2, 1)
    q = {split: [{k: v.copy() for k, v in pair.items()} for pair in p[split]]
         for split in ("train", "test")}
    out = q["train"][0]["output"]
    out[out == CONCAVE] = CONVEX
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("Collapsed bend labels accepted")
    q = {split: [{k: v.copy() for k, v in pair.items()} for pair in p[split]]
         for split in ("train", "test")}
    q["train"][0]["output"] = np.roll(q["train"][0]["output"], 1, axis=1)
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong shear phase accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes),
            "contour_geometries": len(geometries), "turn_patterns": len(turn_words),
            "object_counts": sorted(counts), "concave_counts": sorted(concaves),
            "backgrounds": sorted(backgrounds), "foreground_palettes": len(foreground_palettes),
            "adjacent_bend_contours": int(adjacent), "edge_pairs": int(edge),
            "blank_row_pairs": int(blank_rows), "width_30_outputs": int(boundary),
            "multiple_ink_pairs": int(multicolor)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
