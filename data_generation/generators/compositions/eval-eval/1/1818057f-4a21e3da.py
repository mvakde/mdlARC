"""Relay constellations: embedded crosses carry a border signal across gaps.

No parent code is imported. Generation metadata never enters solve(). Tests and
diagnostics stream puzzles, do not save traces, and use standard library/NumPy.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json

import numpy as np


SIDES = ("top", "right", "bottom", "left")
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 118104102, 500687657, 67066972, 13892660, 198543297,
              940345305, 833165801, 730811562, 434815916, 329561589,
              671666141, 681295239, 817860870, 780399459, 18011438,
              888554114, 983556920, 88939464, 561373201, 246871534,
              643035432, 895643037, 793208370, 762932352, 430032611,
              889467826, 34471857, 449098429, 289643067, 782235703,
              497492314, 664152476, 970597983, 944454356, 718842416,
              117522260, 841923827, 177377313, 93027543, 795989250,
              986042109)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grids must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a


def _roles(grid):
    h, w = grid.shape
    rim = np.zeros((h, w), dtype=bool)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    colors, counts = np.unique(grid[rim], return_counts=True)
    bg = int(colors[np.argmax(counts)])
    points = [tuple(map(int, p)) for p in np.argwhere(rim & (grid != bg))]
    _require(1 <= len(points) <= 2, "one or two border markers required")
    marker_colors = {int(grid[p]) for p in points}
    _require(len(marker_colors) == 1, "markers share a signal color")
    signal = marker_colors.pop()
    _require(int((grid == signal).sum()) == len(points), "signal initially occurs only on border")
    fg_colors = set(map(int, np.unique(grid))) - {bg, signal}
    _require(len(fg_colors) == 1, "exactly three input roles required")
    fg = fg_colors.pop()
    _require(not np.any(rim & (grid == fg)), "foreground must be interior")
    markers = {}
    for r, c in points:
        sides = (["top"] if r == 0 else []) + (["bottom"] if r == h - 1 else [])
        sides += (["left"] if c == 0 else []) + (["right"] if c == w - 1 else [])
        _require(len(sides) == 1 and sides[0] not in markers, "markers are noncorner and on distinct sides")
        markers[sides[0]] = (r, c)
    if len(markers) == 2:
        _require(len(set(markers) & {"top", "bottom"}) == 1,
                 "two markers must occupy adjacent sides")
    return bg, fg, signal, markers


def _centers(mask):
    found = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        found[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                  & mask[2:, 1:-1] & mask[1:-1, :-2]
                                  & mask[1:-1, 2:])
    return {tuple(map(int, p)) for p in np.argwhere(found)}


def _cross(p):
    r, c = p
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _aligned(p, side, marker):
    return p[1] == marker[1] if side in ("top", "bottom") else p[0] == marker[0]


def _activation(centers, markers):
    rows, cols = defaultdict(set), defaultdict(set)
    for p in centers:
        rows[p[0]].add(p)
        cols[p[1]].add(p)
    direct = {p for p in centers if any(_aligned(p, s, m) for s, m in markers.items())}
    depths = {p: 0 for p in direct}
    queue = deque(sorted(direct))
    while queue:
        p = queue.popleft()
        for q in sorted(rows[p[0]] | cols[p[1]]):
            if q not in depths:
                depths[q] = depths[p] + 1
                queue.append(q)
    return set(depths), depths


def _segment(a, b):
    if a[0] == b[0]:
        return {(a[0], c) for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1)}
    _require(a[1] == b[1], "segments must be axial")
    return {(r, a[1]) for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1)}


def _guide_cells(active, markers):
    lines = set()
    for side, marker in markers.items():
        aligned = [p for p in active if _aligned(p, side, marker)]
        if aligned:
            farthest = max(aligned, key=lambda p: abs(p[0] - marker[0]) + abs(p[1] - marker[1]))
            lines |= _segment(marker, farthest)
    for axis in (0, 1):
        groups = defaultdict(list)
        for p in active:
            groups[p[axis]].append(p)
        for group in groups.values():
            if len(group) >= 2:
                lines |= _segment(min(group), max(group))
    return lines


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recognize relays once, close the alignment graph, then paint its spans."""
    grid = _grid(input_grid)
    bg, fg, signal, markers = _roles(grid)
    centers = _centers(grid == fg)
    active, _ = _activation(centers, markers)
    output = grid.copy()
    for p in _guide_cells(active, markers):
        if int(grid[p]) == bg:
            output[p] = signal
    for p in active:
        for q in _cross(p):
            output[q] = signal
    return output


def _topology(rng, nr, nc, sides, low_depth, high_depth, direct_multiple=False):
    """Sample a bipartite incidence graph, not a pre-rendered layout template."""
    for attempt in range(180):
        edges = {(0, 0)}
        used_r, used_c = [0], [0]
        last_r = last_c = 0
        while len(used_r) < nr or len(used_c) < nc:
            add_row = len(used_c) == nc or (len(used_r) < nr and rng.random() < .5)
            if add_row:
                r = len(used_r)
                c = last_c if rng.random() < .78 else int(rng.choice(used_c))
                edges.add((r, c))
                used_r.append(r)
                last_r, last_c = r, c
            else:
                c = len(used_c)
                r = last_r if rng.random() < .78 else int(rng.choice(used_r))
                edges.add((r, c))
                used_c.append(c)
                last_r, last_c = r, c
        absent = sorted({(r, c) for r in range(nr) for c in range(nc)} - edges)
        rng.shuffle(absent)
        for p in absent[:int(rng.integers(0, 4 if high_depth <= 3 else 2))]:
            edges.add(p)
        candidates = sorted(edges)
        rng.shuffle(candidates)
        for root in candidates:
            markers = {s: (-10, root[1]) if s in ("top", "bottom") else (root[0], -10) for s in sides}
            _, depths = _activation(edges, markers)
            depth = max(depths.values())
            multiple = any(sum(_aligned(p, s, m) for p in edges) >= 2 for s, m in markers.items())
            if low_depth <= depth <= high_depth and (multiple or not direct_multiple):
                return edges, root
    # Finite fallback: a permutable staircase graph; two-axis roots seed only
    # its first two nodes. Geometry, padding, palette and clutter remain random.
    edges = {(0, 0)}
    for i in range(1, max(nr, nc)):
        if i < nr:
            edges.add((i, min(i - 1, nc - 1)))
        if i < nc:
            edges.add((min(i, nr - 1), i))
    root = (0, 0)
    markers = {s: (-10, 0) if s in ("top", "bottom") else (0, -10) for s in sides}
    depth = max(_activation(edges, markers)[1].values())
    _require(low_depth <= depth <= high_depth, "topology fallback does not fit depth request")
    return edges, root


def _coordinates(rng, count):
    gaps = [3] * (count - 1)
    # Independently sized nonuniform gaps, bounded before allocation.
    for i in rng.permutation(count - 1):
        budget = 25 - sum(gaps)
        gaps[int(i)] += int(rng.integers(0, min(4, budget) + 1))
    span = sum(gaps)
    size = int(rng.integers(span + 5, min(30, span + 13) + 1))
    start = int(rng.integers(2, size - span - 2))
    return size, [start] + [start + int(x) for x in np.cumsum(gaps)]


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _make_pair(rng, sides, roles, test, overlap, seen, min_train_depth=1):
    for attempt in range(40):
        nr, nc = (tuple(map(int, rng.integers(4, 6, size=2))) if test
                  else tuple(map(int, rng.integers(3 if min_train_depth >= 2 else 2, 5, size=2))))
        # Four-by-four trees can have depth five; five-by-five gives more slack.
        edges, root = _topology(rng, nr, nc, sides, 4 if test else min_train_depth,
                                8 if test else 3, direct_multiple=True)
        spare_r = 1 + int(nr < 5 and rng.random() < .3)
        spare_c = 1 + int(nc < 5 and rng.random() < .3)
        h, rr = _coordinates(rng, nr + spare_r)
        w, cc = _coordinates(rng, nc + spare_c)
        # The inactive coordinates occupy one extreme, but its location and the
        # active graph's spatial embedding are randomized independently.
        if rng.random() < .5:
            rr = rr[::-1]
        if rng.random() < .5:
            cc = cc[::-1]
        ar, ac = rr[:nr], cc[:nc]
        rng.shuffle(ar)
        rng.shuffle(ac)
        active = {(ar[r], ac[c]) for r, c in edges}
        decoys = {(rr[nr], cc[nc])}
        if spare_r == 2:
            decoys.add((rr[nr + 1], cc[nc]))
        if spare_c == 2 and rng.random() < .7:
            decoys.add((rr[nr], cc[nc + 1]))
        if overlap:
            p = sorted(active)[int(rng.integers(len(active)))]
            dc = int(rng.choice((-1, 1))) * int(rng.integers(1, 3))
            q = (p[0], p[1] + dc)
            if not (2 <= q[1] < w - 2) or any(q[1] == c for _, c in decoys):
                continue
            active.add(q)
        expected_centers = active | decoys
        mask = np.zeros((h, w), dtype=bool)
        for p in expected_centers:
            for q in _cross(p):
                mask[q] = True
        # A plain L stroke physically joins a live relay to a dead relay.
        # This is deliberately NOT an edge in the row/column relay graph.
        dead = min(decoys)
        live = sorted(active)[int(rng.integers(len(active)))]
        bend = (live[0], dead[1])
        for p in _segment(live, bend) | _segment(bend, dead):
            mask[p] = True
        if _centers(mask) != expected_centers:
            continue
        root_point = (ar[root[0]], ac[root[1]])
        markers = {s: ((0, root_point[1]) if s == "top" else
                       (h - 1, root_point[1]) if s == "bottom" else
                       (root_point[0], 0) if s == "left" else
                       (root_point[0], w - 1)) for s in sides}
        reached, depths = _activation(expected_centers, markers)
        if reached != active or not (4 <= max(depths.values()) if test else min_train_depth <= max(depths.values()) <= 3):
            continue
        plus_cells = set().union(*(_cross(p) for p in active))
        lines = _guide_cells(active, markers)
        # Foreground on a guide must survive. Adding it cannot invalidate any
        # relay; accept only if it creates no unintended new relay.
        blockers = sorted(p for p in lines - plus_cells if 0 < p[0] < h - 1 and 0 < p[1] < w - 1 and not mask[p])
        rng.shuffle(blockers)
        for p in blockers[:3]:
            candidate = mask.copy()
            candidate[p] = True
            if _centers(candidate) == expected_centers:
                mask = candidate
                break
        # T, X, hollow ring, bent stroke, and random short-stroke camouflage.
        shapes = (((0, 0), (-1, 0), (0, -1), (0, 1)),
                  ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)),
                  tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)),
                  ((0, 0), (0, 1), (1, 1), (2, 1)))
        for _ in range(int(rng.integers(2, 9))):
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            offsets = shapes[int(rng.integers(len(shapes)))]
            turn = int(rng.integers(4))
            points = []
            for a, b in offsets:
                for _ in range(turn):
                    a, b = -b, a
                points.append((r + a, c + b))
            if not all(0 < a < h - 1 and 0 < b < w - 1 for a, b in points):
                continue
            candidate = mask.copy()
            for p in points:
                candidate[p] = True
            if _centers(candidate) == expected_centers and candidate.mean() < .44:
                mask = candidate
        bg, fg, signal = roles
        grid = np.full((h, w), bg, dtype=np.int8)
        grid[mask] = fg
        for p in markers.values():
            grid[p] = signal
        if _key(grid) in seen:
            continue
        pair = {"input": grid, "output": solve(grid)}
        # Witnesses are measured from final visible grids, not sampler flags.
        f = _audit_pair(pair)
        if not f["component_trap"] or not f["preserved_on_guide"] or not f["new_guide_cells"]:
            continue
        if overlap and not f["overlap"]:
            continue
        return pair
    raise RuntimeError("bounded scene proposal budget exhausted")


def _integer(value, name, low, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(value >= low and (high is None or value <= high), name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    stable_palette = rng.random() < .3
    roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
    overlap_family = rng.random() < .6
    orientation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    # Two demonstrations suffice even at the minimum legal count: one single
    # marker and one double marker, each with actual indirect activation.
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            double = i == 1 if split == "train" and i < 2 else rng.random() < .5
            k = (orientation + (i % 2 if split == "train" and i < 2 else int(rng.integers(4)))) % 4
            sides = (SIDES[k], SIDES[(k + 1) % 4]) if double else (SIDES[k],)
            if not stable_palette:
                roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
            overlap = overlap_family and ((split == "train" and i == 1) or (split == "test" and rng.random() < .6))
            pair = _make_pair(rng, sides, roles, split == "test", overlap, seen,
                              min_train_depth=2 if split == "train" and i == 1 else 1)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    """Direct coordinate audit, separate from production detector/queue/render."""
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output dimensions must be preserved")
    bg, fg, signal, markers = _roles(inp)
    h, w = inp.shape
    centers = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
               if int(inp[r, c]) == fg and all(int(inp[r + dr, c + dc]) == fg for dr, dc in CARDINALS)}
    active = {p for p in centers if any(_aligned(p, s, m) for s, m in markers.items())}
    direct = set(active)
    depth = 0
    while True:
        more = {p for p in centers if any(p[0] == q[0] or p[1] == q[1] for q in active)}
        if more <= active:
            break
        depth += 1
        active |= more
    strokes = set(markers.values())
    for side, m in markers.items():
        points = [p for p in centers if _aligned(p, side, m)]
        if points:
            end = max(points, key=lambda p: abs(p[0] - m[0]) + abs(p[1] - m[1]))
            dr = (end[0] > m[0]) - (end[0] < m[0])
            dc = (end[1] > m[1]) - (end[1] < m[1])
            p = m
            while p != end:
                strokes.add(p)
                p = (p[0] + dr, p[1] + dc)
            strokes.add(end)
    for a in active:
        for b in active:
            if a[0] == b[0]:
                strokes.update((a[0], c) for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1))
            if a[1] == b[1]:
                strokes.update((r, a[1]) for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1))
    lit = set().union(*(_cross(p) for p in active)) if active else set()
    changed = lit | {p for p in strokes if int(inp[p]) == bg}
    expected = inp.copy()
    for p in changed:
        expected[p] = signal
    _require(np.array_equal(out, expected), "relay closure or selective painting is incorrect")
    _require(np.array_equal(inp == signal, (out == signal) & (inp == signal)), "markers must survive")
    _require(bool(centers) and bool(active) and bool(centers - active), "generation requires live and dead relays")
    # Reachability through monochrome foreground is deliberately insufficient.
    connected = set(lit)
    todo = list(lit)
    while todo:
        r, c = todo.pop()
        for dr, dc in CARDINALS:
            q = (r + dr, c + dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and int(inp[q]) == fg and q not in connected:
                connected.add(q)
                todo.append(q)
    return {"shape": (h, w), "roles": (bg, fg, signal), "sides": tuple(sorted(markers)),
            "centers": len(centers), "active": len(active), "inactive": len(centers - active),
            "depth": depth, "indirect": len(active - direct),
            "overlap": len(lit) < 5 * len(active),
            "component_trap": bool((centers - active) & connected),
            "preserved_on_guide": sum(int(inp[p]) == fg and p not in lit for p in strokes),
            "new_guide_cells": sum(int(inp[p]) == bg for p in strokes),
            "cycle": len(active) >= len({p[0] for p in active}) + len({p[1] for p in active}),
            "structure": tuple(sorted(centers)), "foreground": int((inp == fg).sum())}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    features, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact pair keys required")
            inp = _grid(pair["input"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            f = _audit_pair(pair)
            _require(f["indirect"] > 0 and f["new_guide_cells"] > 0, "both mechanisms must affect every example")
            _require(f["component_trap"] and f["preserved_on_guide"] > 0, "missing shortcut counterexample")
            features[split].append(f)
    train, test = features["train"], features["test"]
    _require({len(f["sides"]) for f in test} <= {len(f["sides"]) for f in train}, "unwitnessed marker arity")
    _require(not any(f["overlap"] for f in test) or any(f["overlap"] for f in train), "unwitnessed overlapping relays")
    _require(2 <= max(f["depth"] for f in train) <= 3 and min(f["depth"] for f in test) >= 4,
             "tests must require deeper signal propagation")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _manual_regression():
    """Declared coordinates supply expected labels without a detected graph."""
    grid = np.zeros((17, 20), dtype=np.int8)
    live = ((4, 4), (4, 11), (10, 11), (10, 6))
    dead = (13, 16)
    for r, c in (*live, dead):
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            grid[r + dr, c + dc] = 2
    grid[4, 11:17] = 2
    grid[4:14, 16] = 2  # Physical bridge does not relay a turn at (4,16).
    grid[2, 4] = 2  # Ordinary foreground inside the initial guide.
    for p in ((7, 4), (6, 4), (7, 3), (7, 5)):
        grid[p] = 2  # Incomplete plus beyond the first guide's endpoint.
    grid[0, 4] = 8
    expected = grid.copy()
    spans = ([(r, 4) for r in range(5)]
             + [(4, c) for c in range(4, 12)]
             + [(r, 11) for r in range(4, 11)]
             + [(10, c) for c in range(6, 12)])
    for p in spans:
        if grid[p] == 0:
            expected[p] = 8
    for r, c in live:
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            expected[r + dr, c + dc] = 8
    _require(np.array_equal(solve(grid), expected), "declared relay/bridge regression")
    _require(expected[dead] == 2 and expected[2, 4] == 2 and expected[1, 4] == 8,
             "dead relay, foreground preservation, or gap fill fixture failed")
    # Moving the only marker to a null axis must preserve the entire input.
    null = grid.copy()
    null[0, 4], null[0, 1] = 0, 8
    _require(np.array_equal(solve(null), null), "null marker ray regression")
    # Two adjacent markers can seed distinct alignment components.
    two = grid.copy()
    two[13, -1] = 8
    expected_two = expected.copy()
    expected_two[13, -1] = 8
    for c in range(16, 20):
        if two[13, c] == 0:
            expected_two[13, c] = 8
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        expected_two[13 + dr, 16 + dc] = 8
    _require(np.array_equal(solve(two), expected_two), "multiple seeded components regression")


def self_test():
    """Bounded streaming suite over the 42 advertised host seeds; no files."""
    _manual_regression()
    fingerprints, shapes, roles, structures = set(), set(), set(), set()
    sides = {"train": set(), "test": set()}
    depths = {"train": set(), "test": set()}
    total = overlaps = cycles = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _audit_pair(pair)
                total += 1
                overlaps += f["overlap"]
                cycles += f["cycle"]
                shapes.add(f["shape"])
                roles.add(f["roles"])
                structures.add((f["shape"], f["structure"]))
                sides[split].add(f["sides"])
                depths[split].add(f["depth"])
        if seed == 0:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "seed is not deterministic")
            # Palette and dihedral equivariance test input-only interpretation.
            pair = puzzle["test"][0]
            before = pair["input"].copy()
            for k in range(4):
                for flip in (False, True):
                    x, y = np.rot90(before, k), np.rot90(pair["output"], k)
                    if flip:
                        x, y = np.fliplr(x), np.fliplr(y)
                    _require(np.array_equal(solve(x), y), "orientation equivariance failed")
            palette = np.array([7, 4, 0, 9, 2, 8, 6, 1, 5, 3])
            _require(np.array_equal(solve(palette[before]), palette[pair["output"]]), "palette equivariance failed")
            _require(np.array_equal(before, pair["input"]), "solver modified its input")
            bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("validator accepted a damaged output")
    _require(len(fingerprints) == 42 and len(structures) == total, "insufficient structural diversity")
    _require(len(shapes) >= 45 and len(roles) >= 35, "insufficient shape or palette variation")
    _require(all(len(sides[s]) == 8 for s in sides), "missing side configuration")
    _require(min(depths["train"]) >= 1 and max(depths["train"]) == 3 and min(depths["test"]) >= 4,
             "propagation shift is missing")
    _require(overlaps >= 20 and cycles >= 10, "missing overlapping or cyclic networks")
    # Exhaust the fixed local motif, including boundaries and missing arms.
    for bits in range(512):
        mask = np.array([(bits >> j) & 1 for j in range(9)], dtype=bool).reshape(3, 3)
        truth = all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
        _require(_centers(mask) == ({(1, 1)} if truth else set()), "local detector regression")
    block = np.pad(np.ones((3, 4), dtype=bool), 1)
    _require(_centers(block) == {(2, 2), (2, 3)}, "overlap regression")
    # Host also exercises these; include type errors and count boundaries here.
    for a, b in ((2, 1), (10, 4)):
        p = generate_puzzle(123, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": total, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "role_triples": len(roles), "structures": len(structures),
            "overlap_pairs": overlaps, "cyclic_pairs": cycles,
            "configurations": {s: len(v) for s, v in sides.items()},
            "depths": {s: sorted(v) for s, v in depths.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
