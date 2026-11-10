"""Contour registration jigsaws: joint rigid assembly and sparse cycle inference.

Only the standard library and NumPy are used. No parent generator is imported.
The solver searches visible piece placements and contour edges jointly.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


_EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid ARC color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    length = max(abs(dr), abs(dc))
    if not length or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return []
    sr, sc = int(dr > 0) - int(dr < 0), int(dc > 0) - int(dc < 0)
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(length + 1)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _edge_data(points):
    points = sorted(points)
    landmarks = set(points)
    edges, rasters = [], []
    adjacency = [[] for _ in points]
    for i, a in enumerate(points):
        for j, b in enumerate(points[:i]):
            raster = _line(a, b)
            if raster and not landmarks.intersection(raster[1:-1]):
                k = len(edges)
                edges.append((i, j))
                rasters.append(set(raster))
                adjacency[i].append((j, k))
                adjacency[j].append((i, k))
    return points, edges, rasters, adjacency


def _cycles(points, limit=2):
    """All simple full cycles up to a second distinct edge set.

    Edges may not skip a landmark or cross, including half-cell diagonal
    crossings. Starting at vertex zero removes cyclic duplicates; reversal is
    deduplicated by the edge set. Adjacent landmark links are legal.
    """
    if not 4 <= len(points) <= 12:
        return []
    points, edges, rasters, adjacency = _edge_data(points)
    if any(len(a) < 2 for a in adjacency):
        return []
    conflicts = [0] * len(edges)
    for k, (i, j) in enumerate(edges):
        for l, (u, v) in enumerate(edges[:k]):
            if not {i, j} & {u, v} and _intersects(points[i], points[j], points[u], points[v]):
                conflicts[k] |= 1 << l
                conflicts[l] |= 1 << k
    results, seen = [], set()

    def visit(order, used, selected):
        if len(results) >= limit:
            return
        if len(order) == len(points):
            for vertex, edge in adjacency[order[-1]]:
                if vertex == 0 and not conflicts[edge] & selected:
                    signature = selected | (1 << edge)
                    if signature not in seen:
                        seen.add(signature)
                        chosen = [k for k in range(len(edges)) if signature >> k & 1]
                        pixels = set().union(*(rasters[k] for k in chosen))
                        if not pixels - set(points):
                            continue
                        results.append({"pixels": pixels,
                                        "edges": [edges[k] for k in chosen], "points": points,
                                        "order": order.copy(), "chords": len(edges) - len(chosen)})
            return
        for vertex, edge in adjacency[order[-1]]:
            if not used >> vertex & 1 and not conflicts[edge] & selected:
                visit(order + [vertex], used | (1 << vertex), selected | (1 << edge))

    visit([0], 1, 0)
    return results


def _tight(array):
    cells = np.argwhere(array >= 0)
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return array[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _connected(mask, neighbors=_EDGE):
    remaining = set(tuple(map(int, p)) for p in np.argwhere(mask))
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _components(grid, background):
    remaining = set(tuple(map(int, p)) for p in np.argwhere(grid != background))
    pieces = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
        piece = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=np.int8)
        for r, c in cells:
            piece[r - r0, c - c0] = grid[r, c]
        pieces.append(piece)
    return pieces


def _read_input(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > grid.size, "Background must occupy a strict majority")
    pieces = _components(grid, background)
    _require(3 <= len(pieces) <= 5, "Expected three to five separated pieces")
    appearances = {int(c): sum(bool(np.any(p == c)) for p in pieces)
                   for c in colors if c != background}
    shared = [c for c, count in appearances.items() if count > 1]
    _require(len(shared) == 1, "Exactly one landmark color must occur on multiple pieces")
    source = shared[0]
    singleton = [int(c) for c, n in zip(colors, counts) if n == 1 and c != background]
    _require(len(singleton) == 1, "Exactly one singleton companion/anchor marker is required")
    marker = singleton[0]
    _require(marker != source, "Source and companion must differ")
    carriers = []
    for p in pieces:
        carrier = set(map(int, p[p >= 0])) - {source, marker}
        _require(len(carrier) == 1, "Each piece must have one distinct carrier color")
        color = next(iter(carrier))
        _require(np.count_nonzero(p == color) >= 2, "A carrier needs at least two visible cells")
        _require(np.any(p == source), "Every piece must carry a landmark")
        _require(_connected(p >= 0, _NEIGHBORS), "Pieces must be corner-connected")
        carriers.append(color)
    _require(len(set(carriers)) == len(pieces), "Piece identities must be distinct")
    area = int(np.count_nonzero(grid != background))
    side = math.isqrt(area)
    _require(side * side == area, "Piece area must be a square")
    _require(4 <= np.count_nonzero(grid == source) <= 12, "Expected four to twelve landmarks")
    marked = next(p for p in pieces if np.any(p == marker))
    m = tuple(map(int, np.argwhere(marked == marker)[0]))
    adjacent = [tuple(map(int, p)) for p in np.argwhere(marked == source)
                if max(abs(int(p[0]) - m[0]), abs(int(p[1]) - m[1])) == 1]
    _require(len(adjacent) == 1, "Marker must neighbor exactly one landmark on its own piece")
    return {"grid": grid, "background": background, "source": source, "marker": marker,
            "pieces": pieces, "carriers": carriers, "side": side}


def _paint(base, pixels, delta, source, marker):
    """Carrier material is the drawing canvas; source strokes occlude copies."""
    out = base.copy()
    for r, c in pixels:
        p = r + delta[0], c + delta[1]
        if base[p] != source:
            out[p] = marker
    for p in pixels:
        out[p] = source
    return out


def _assemblies(facts, use_contour=True, limit=2):
    """Exact cover with colored orientations, followed by global cycle constraints.

    A second solution is sufficient to reject ambiguity. For the geometry-only
    ablation, deduplication ignores annotations and counts carrier arrangements.
    No search budget truncates the solver's legal family members.
    """
    pieces, side = facts["pieces"], facts["side"]
    source, marker = facts["source"], facts["marker"]
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    full = (1 << (side * side)) - 1
    options = []
    for i, piece in enumerate(pieces):
        orientations, placements = set(), []
        for rotation in range(4):
            p = np.rot90(piece, rotation)
            key = _key(p)
            if key in orientations:
                continue
            orientations.add(key)
            h, w = p.shape
            if h > side or w > side or (i == anchor and p[0, 0] != marker):
                continue
            cells = [(int(r), int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0)]
            mask = sum(1 << (r * side + c) for r, c, color in cells)
            offsets = [(0, 0)] if i == anchor else [(r, c) for r in range(side - h + 1)
                                                   for c in range(side - w + 1)]
            for dr, dc in offsets:
                shifted = [(r + dr, c + dc, color) for r, c, color in cells]
                dots = [(r, c) for r, c, color in shifted if color == source]
                placements.append((mask << (dr * side + dc), shifted, dots))
        options.append(placements)
    answers, base_seen, owner_seen = [], set(), set()
    placed = [None] * len(pieces)

    def search(occupied, remaining, dots):
        if len(answers) >= limit:
            return
        if not remaining:
            _require(occupied == full, "Internal exact-cover error")
            if use_contour:
                near = [p for p in dots if max(p[0], p[1]) == 1]
                if len(near) != 1:
                    return
                delta = -near[0][0], -near[0][1]
                # Every straight link stays in its endpoints' bounding box.
                if any(not (0 <= r + delta[0] < side and 0 <= c + delta[1] < side) for r, c in dots):
                    return
                valid_cycles = _cycles(dots, limit=limit)
                if not valid_cycles:
                    return
            base = np.empty((side, side), dtype=np.int8)
            owners = np.empty_like(base)
            for i, placement in enumerate(placed):
                for r, c, color in placement[1]:
                    base[r, c] = color
                    owners[r, c] = facts["carriers"][i]
            if not use_contour:
                signature = owners.tobytes()
                if signature not in owner_seen:
                    owner_seen.add(signature)
                    answers.append({"base": base, "owners": owners})
                return
            signature = base.tobytes()
            if signature in base_seen:
                return
            base_seen.add(signature)
            for cycle in valid_cycles:
                if (0, 0) in cycle["pixels"]:
                    continue
                answer = {"base": base, "owners": owners, "cycle": cycle, "delta": delta,
                          "output": _paint(base, cycle["pixels"], delta, source, marker)}
                answers.append(answer)
                if len(answers) >= limit:
                    return
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in options[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    search(occupied | candidate[0], rest, dots + candidate[2])
                    if len(answers) >= limit:
                        return
            placed[i] = None

    for candidate in options[anchor]:
        placed[anchor] = candidate
        search(candidate[0], tuple(i for i in range(len(pieces)) if i != anchor), candidate[2])
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    answers = _assemblies(facts)
    if len(answers) != 1:
        raise ValueError("Input must have exactly one anchored assembly and simple contour")
    return answers[0]["output"].copy()


def _contour(rng, side, mode, root_kind, minimum, maximum):
    root = [(1, 1), (0, 1), (1, 0)][root_kind]
    r, c = root
    h, w = side - 1 - r, side - 1 - c
    if mode == 0:
        vertices = [root, (r, c + w), (r + h, c + w), (r + h, c)]
    elif mode == 1:
        bound = max(1, min(h, w) // 3)
        a, b, d = map(int, rng.integers(bound + 1, size=3))
        b = max(1, b)
        vertices = [root, (r, c + w - a), (r + a, c + w), (r + h - b, c + w),
                    (r + h, c + w - b), (r + h, c + d), (r + h - d, c)]
        vertices = [p for i, p in enumerate(vertices) if p != vertices[i - 1]]
    else:
        start = int(rng.integers(2, w - 2))
        end = int(rng.integers(start + 2, w))
        depth = int(rng.integers(1, h - 1))
        top = [root, (r, c + w)]
        if mode == 3:
            cut = int(rng.integers(1, min(h, w) // 3 + 1))
            top = [root, (r, c + w - cut), (r + cut, c + w)]
        vertices = top + [(r + h, c + w), (r + h, c + end), (r + h - depth, c + end),
                          (r + h - depth, c + start), (r + h, c + start), (r + h, c)]
    if rng.random() < .5:
        vertices = [(c, r) for r, c in vertices]
        root = root[1], root[0]
    pixels = set().union(*(set(_line(a, b)) for a, b in zip(vertices, vertices[1:] + vertices[:1])))
    points = set(vertices)
    available = sorted(pixels - points)
    rng.shuffle(available)
    target = int(rng.integers(max(minimum, len(vertices)), maximum + 1))
    for p in available:
        if len(points) >= target:
            break
        if max(abs(p[0] - root[0]), abs(p[1] - root[1])) >= 2:
            points.add(p)
    if len(points) != target:
        return None
    return points, pixels, root


def _partition(rng, side, count, style, points, root):
    if style == "cuts":
        rectangles = [(0, 0, side, side)]
        while len(rectangles) < count:
            i = int(rng.integers(len(rectangles)))
            r, c, h, w = rectangles.pop(i)
            choices = [(0, k) for k in range(1, h) if min(k, h - k) * w >= 4]
            choices += [(1, k) for k in range(1, w) if min(k, w - k) * h >= 4]
            if not choices:
                return None
            axis, k = choices[int(rng.integers(len(choices)))]
            rectangles.extend([(r, c, k, w), (r + k, c, h - k, w)] if axis == 0 else
                              [(r, c, h, k), (r, c + k, h, w - k)])
        labels = np.empty((side, side), dtype=np.int8)
        for i, (r, c, h, w) in enumerate(rectangles):
            labels[r:r + h, c:c + w] = i
        # Steps alter real rigid silhouettes, while one protected rectangle
        # helps retain alternative geometric assemblies for route registration.
        if rng.random() < .6:
            protected = int(rng.integers(count))
            for _ in range(int(rng.integers(1, side + 1))):
                r, c = map(int, rng.integers(side, size=2))
                dr, dc = _EDGE[int(rng.integers(4))]
                rr, cc = r + dr, c + dc
                if not (0 <= rr < side and 0 <= cc < side) or (r, c) == (0, 0):
                    continue
                a, b = int(labels[r, c]), int(labels[rr, cc])
                if a == b or protected in (a, b) or np.count_nonzero(labels == a) <= 4:
                    continue
                labels[r, c] = b
                if not _connected(labels == a):
                    labels[r, c] = a
        return labels
    # Repeated inserts make geometry deliberately insufficient. Their landmark
    # arrangements identify both the correct cavity and the correct rotation.
    labels = np.zeros((side, side), dtype=np.int8)
    h, w = int(rng.integers(1, 4)), int(rng.integers(2, 5))
    mask = np.ones((h, w), dtype=bool)
    if h >= 2 and w >= 3 and rng.random() < .35:
        mask[0, 0] = False
    for i in range(1, count):
        for _ in range(60):
            candidates = [p for p in sorted(points) if labels[p] == 0 and p != root]
            if not candidates:
                return None
            p = candidates[int(rng.integers(len(candidates)))]
            ar, ac = map(int, np.argwhere(mask)[int(rng.integers(np.count_nonzero(mask)))])
            r, c = p[0] - ar, p[1] - ac
            if not (0 <= r <= side - h and 0 <= c <= side - w):
                continue
            coords = [(r + int(a), c + int(b)) for a, b in np.argwhere(mask)]
            if root in coords or (0, 0) in coords or any(labels[q] for q in coords):
                continue
            proposal = labels.copy()
            for q in coords:
                proposal[q] = i
            if _connected(proposal == 0):
                labels = proposal
                break
        else:
            return None
    return labels


def _scatter(rng, pieces, background, anchor_rotation):
    marked_index = next(i for i, p in enumerate(pieces) if np.unique(p[p >= 0]).size == 3)
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    rotations[marked_index] = anchor_rotation
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    area = sum(int(np.count_nonzero(p >= 0)) for p in pieces)
    for attempt in range(16):
        canvas = np.full((30, 30), -1, dtype=np.int8)
        blocked = np.zeros((30, 30), dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: -np.count_nonzero(oriented[i] >= 0))
        if attempt:
            rng.shuffle(order)
        maxr = maxc = 0
        aspect = float(rng.uniform(.65, 1.55))
        for i in order:
            p = oriented[i]
            h, w = p.shape
            coords = np.argwhere(p >= 0)
            rr, cc = coords[:, 0], coords[:, 1]
            choices = []
            for r in range(31 - h):
                for c in range(31 - w):
                    if np.any(blocked[rr + r, cc + c]):
                        continue
                    hr, wc = max(maxr, r + h), max(maxc, c + w)
                    score = hr * wc + abs(hr - aspect * wc) * 2 + float(rng.random()) * 30
                    choices.append((score, r, c))
            if not choices:
                break
            _, r, c = min(choices)
            for ar, ac in coords:
                nr, nc = r + int(ar), c + int(ac)
                canvas[nr, nc] = p[ar, ac]
                blocked[max(0, nr - 1):min(30, nr + 2), max(0, nc - 1):min(30, nc + 2)] = True
            maxr, maxc = max(maxr, r + h), max(maxc, c + w)
        else:
            cropped = _tight(canvas)
            h, w = cropped.shape
            H = h + int(rng.integers(min(5, 30 - h) + 1))
            W = w + int(rng.integers(min(5, 30 - w) + 1))
            while H * W <= 2 * area:
                if H < 30 and (H <= W or W == 30):
                    H += 1
                elif W < 30:
                    W += 1
                else:
                    break
            if H * W <= 2 * area:
                continue
            top, left = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
            grid = np.full((H, W), background, dtype=np.int8)
            grid[top:top + h, left:left + w] = np.where(cropped >= 0, cropped, background)
            return grid
    return None


def _unit_copy_difference(cycle, delta):
    long_pixels = set().union(*(set(_line(cycle["points"][i], cycle["points"][j]))
                               for i, j in cycle["edges"]
                               if len(_line(cycle["points"][i], cycle["points"][j])) > 2))
    extra = {(r + delta[0], c + delta[1]) for r, c in cycle["pixels"] - long_pixels}
    return len(extra - cycle["pixels"] - {(0, 0)})


def _propose(rng, side, count, style, mode, root_kind, npoints, palette, rotation, requirements=()):
    contour = _contour(rng, side, mode, root_kind, *npoints)
    if contour is None:
        return None
    points, pixels, root = contour
    cycles = _cycles(points)
    if len(cycles) != 1 or cycles[0]["pixels"] != pixels:
        return None
    delta = -root[0], -root[1]
    if "unit_copy" in requirements and not _unit_copy_difference(cycles[0], delta):
        return None
    labels = _partition(rng, side, count, style, points, root)
    if labels is None or labels[root] != labels[0, 0]:
        return None
    if any(not any(labels[p] == i for p in points) for i in range(count)):
        return None
    background, source, marker = palette[:3]
    carriers = palette[3:3 + count]
    base = np.empty((side, side), dtype=np.int8)
    for i, color in enumerate(carriers):
        base[labels == i] = color
    for p in points:
        base[p] = source
    base[0, 0] = marker
    pieces = [_tight(np.where(labels == i, base, -1)) for i in range(count)]
    if any(np.count_nonzero(p == color) < 2 for p, color in zip(pieces, carriers)):
        return None
    if "one_dot" in requirements and not any(np.count_nonzero(p == source) == 1 for p in pieces):
        return None
    # Congruent unanchored masks with congruent ink annotations can always swap
    # carrier identities without changing the global contour. That is a proof
    # of ambiguity, so avoid the more expensive complete search in these cases.
    annotation_keys = set()
    for p in pieces:
        if np.any(p == marker):
            continue
        annotation = np.where(p < 0, -1, np.where(p == source, 2, 1)).astype(np.int8)
        key = min(_key(np.rot90(annotation, k)) for k in range(4))
        if key in annotation_keys:
            return None
        annotation_keys.add(key)
    facts = {"pieces": pieces, "side": side, "source": source, "marker": marker, "carriers": carriers}
    if len(_assemblies(facts, use_contour=False)) < 2:
        return None
    answers = _assemblies(facts)
    if len(answers) != 1 or not np.array_equal(answers[0]["base"], base):
        return None
    output = _paint(base, pixels, delta, source, marker)
    grid = _scatter(rng, pieces, background, rotation)
    if grid is None:
        return None
    return {"input": grid, "output": output}


def _make_pair(rng, side, count, style, mode, root_kind, npoints, palette, rotation, requirements=()):
    # Preserve requested structures through rejection rather than letting easy
    # rectangles immediately replace all harder candidates.
    for _ in range(1200):
        pair = _propose(rng, side, count, style, mode, root_kind, npoints, palette, rotation, requirements)
        if pair is not None:
            return pair
    # Fixed construction search is a deterministic safety net, not solver state.
    # It retains the requested contour/offset/point count and relaxes only the
    # partition style. There is no private convention in these constructions.
    safety = np.random.default_rng(1907 + side * 100 + count * 10 + mode * 3 + root_kind)
    for _ in range(6000):
        pair = _propose(safety, side, count, "cuts", mode, root_kind, npoints, palette, rotation, requirements)
        if pair is not None:
            return pair
    raise RuntimeError("No unique joint assembly found within the construction bound")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(2))
    train_sides = list(map(int, rng.integers(7, 11, size=nt)))
    # Two informative demonstrations persist even for the minimum legal count.
    demo_counts = list(map(int, rng.permutation([3, 4])))
    first_palette = list(map(int, rng.permutation(10)))
    second_palette = [first_palette[3], first_palette[2], first_palette[1]]
    second_palette += [c for c in rng.permutation(10) if c not in second_palette]
    rotations = list(map(int, rng.permutation([1, 2, 3])))
    specs = []
    for i in range(nt):
        if i == 0:
            specs.append((train_sides[i], demo_counts[i], "insets", 2,
                          int(rng.integers(1, 3)), (8, 10), first_palette, rotations[0], ("one_dot",)))
        elif i == 1:
            specs.append((train_sides[i], demo_counts[i], "cuts", 1, 0,
                          (8, 10), list(map(int, second_palette)), rotations[1], ("unit_copy",)))
        else:
            specs.append((train_sides[i], int(rng.integers(3, 5)), str(rng.choice(["cuts", "insets"])),
                          int(rng.integers(4)), int(rng.integers(3)), (9, 10),
                          list(map(int, rng.permutation(10))), int(rng.integers(4))))
    for i in range(nq):
        side = int(rng.integers(max(train_sides) + 1, 13)) if shift == 0 else int(rng.choice(train_sides))
        count = int(rng.integers(3, 6)) if shift == 0 else 5
        specs.append((side, count, str(rng.choice(["cuts", "insets"])), int(rng.integers(4)),
                      int(rng.integers(3)), (11, 12), list(map(int, rng.permutation(10))),
                      int(rng.integers(4))))
    puzzle, seen = {"train": [], "test": []}, set()
    for i, spec in enumerate(specs):
        for _ in range(10):
            pair = _make_pair(rng, *spec)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle["train" if i < nt else "test"].append(pair)
                break
        else:
            raise RuntimeError("Could not scatter distinct inputs")
    # Demonstration order conveys no semantics.
    rng.shuffle(puzzle["train"])
    return puzzle


def _features(pair, facts=None, answer=None):
    facts = _read_input(pair["input"]) if facts is None else facts
    answer = _assemblies(facts)[0] if answer is None else answer
    cycle, base = answer["cycle"], answer["base"]
    pixels, points = cycle["pixels"], set(cycle["points"])
    dr, dc = answer["delta"]
    copy = {(r + dr, c + dc) for r, c in pixels}
    all_lines = set().union(*_edge_data(points)[2])
    turns = [_cross(cycle["points"][cycle["order"][i - 1]], cycle["points"][v],
                    cycle["points"][cycle["order"][(i + 1) % len(points)]])
             for i, v in enumerate(cycle["order"])]
    diagonal = sum(cycle["points"][i][0] != cycle["points"][j][0]
                   and cycle["points"][i][1] != cycle["points"][j][1] for i, j in cycle["edges"])
    rotations = 0
    for p, color in zip(facts["pieces"], facts["carriers"]):
        mask = answer["owners"] == color
        assembled = _tight(np.where(mask, base, -1))
        _require(any(np.array_equal(np.rot90(p, k), assembled) for k in range(4)), "Rigid pattern changed")
        rotations += not np.array_equal(p, assembled)
    geometry = tuple(tuple(map(int, row)) for row in answer["owners"])
    role_map = {}
    normalized = tuple(tuple(role_map.setdefault(c, len(role_map)) for c in row) for row in geometry)
    return {"side": facts["side"], "count": len(facts["pieces"]), "points": len(points),
            "delta": (dr, dc), "diagonal": diagonal,
            "concave": min(turns) < 0 < max(turns), "chord_cells": len(all_lines - pixels),
            "unit_edges": sum(len(_line(cycle["points"][i], cycle["points"][j])) == 2
                              for i, j in cycle["edges"]),
            "unit_copy_difference": _unit_copy_difference(cycle, (dr, dc)),
            "source_additions": len(pixels - points), "copy_additions": len(copy - pixels - {(0, 0)}),
            "overlap_additions": len((copy & pixels) - points), "rotations": rotations,
            "single_dot_pieces": sum(np.count_nonzero(p == facts["source"]) == 1 for p in facts["pieces"]),
            "irregular_pieces": sum(np.any(p < 0) for p in facts["pieces"]),
            "background": facts["background"], "source": facts["source"], "marker": facts["marker"],
            "shape": facts["grid"].shape, "geometry": (facts["side"], normalized)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Invalid pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            facts = _read_input(inp)
            _require(out.shape == (facts["side"], facts["side"]), "Output side must follow piece area")
            _require(out[0, 0] == facts["marker"], "Marker must anchor the upper-left")
            _require(not np.any(out == facts["background"]), "Square must be fully filled")
            _require(len(_assemblies(facts, use_contour=False)) == 2, "Geometry alone must be ambiguous")
            answers = _assemblies(facts)
            _require(len(answers) == 1, "Joint assembly and contour must be unique")
            answer = answers[0]
            _require(np.array_equal(out, answer["output"]), "Incorrect contour registration or drawing")
            # Only the two rail colors may replace carrier material.
            changed = out != answer["base"]
            _require(np.all(np.isin(out[changed], [facts["source"], facts["marker"]])), "Unrelated recoloring")
            _require(np.all(out[answer["base"] == facts["source"]] == facts["source"]), "Lost landmark")
            f = _features(pair, facts, answer)
            _require(f["source_additions"] > 0 and f["copy_additions"] > 0, "Both rails must grow visibly")
            features[split].append(f)
    tr, te = features["train"], features["test"]
    _require(any(f["concave"] and f["chord_cells"] for f in tr), "Demonstrate contour/chord selection")
    _require(any(f["diagonal"] for f in tr), "Demonstrate diagonal links")
    _require(any(f["delta"] == (-1, -1) for f in tr), "Demonstrate diagonal companion offset")
    _require(any(0 in f["delta"] and f["overlap_additions"] for f in tr), "Demonstrate overlapping companion")
    _require(any(f["single_dot_pieces"] for f in tr), "Demonstrate completion across pieces")
    _require(any(f["unit_copy_difference"] for f in tr), "Demonstrate copying the entire closed contour")
    _require(all(f["rotations"] for f in tr[:2]) or sum(bool(f["rotations"]) for f in tr) >= 2,
             "Demonstrate rigid rotation in two examples")
    _require(len({f["count"] for f in tr}) >= 2, "Demonstrate variable piece count")
    _require(len({f["background"] for f in tr}) >= 2, "Demonstrate background variation")
    _require(any(a["source"] == b["marker"] for a in tr for b in tr), "Demonstrate local palette roles")
    _require(min(f["points"] for f in te) > max(f["points"] for f in tr), "Missing landmark-count shift")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded diversity, semantic ablation, determinism, and negative checks."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                prediction = solve(pair["input"].copy())
                _require(np.array_equal(prediction, pair["output"]), "Input-only solve disagrees")
                records.append((split, _features(pair)))
    fs = [f for s, f in records]
    _require(len(signatures) == 42, "Seeds collapsed")
    _require({f["count"] for f in fs} == {3, 4, 5}, "Missing piece counts")
    _require({f["delta"] for f in fs} == {(-1, -1), (-1, 0), (0, -1)}, "Missing offsets")
    _require(len({f["side"] for f in fs}) >= 5, "Missing scales")
    _require(len({f["shape"] for f in fs}) >= 35, "Missing canvas shapes")
    _require(len({f["geometry"] for f in fs}) >= 100, "Missing structural variation")
    _require({f["background"] for f in fs} == set(range(10)), "Missing background colors")
    for split in ("train", "test"):
        rows = [f for s, f in records if s == split]
        for feature in ("concave", "diagonal", "chord_cells", "unit_edges", "single_dot_pieces", "irregular_pieces"):
            _require(sum(bool(f[feature]) for f in rows) >= 3, f"Missing {split} variation: {feature}")
    a = generate_puzzle(24601, 5, 2)
    _require(_equal(a, generate_puzzle(24601, 5, 2)), "Determinism failed")
    _require(not _equal(a, generate_puzzle(24602, 5, 2)), "Seeds must differ")
    # Counts are independently checked by verify.py; retain inexpensive edge cases.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(800 + nt * 10 + nq, nt, nq)
        _require(len(p["train"]) == nt and len(p["test"]) == nq, "Counts ignored")
        validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")
    # Mutating a newly interpolated source cell must fail the semantic validator.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    pair = bad["test"][0]
    facts = _read_input(pair["input"])
    answer = _assemblies(facts)[0]
    point = next(p for p in sorted(answer["cycle"]["pixels"]) if answer["base"][p] != facts["source"])
    pair["output"][point] = answer["base"][point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted interpolation accepted")
    return {"seeds": 42, "pairs": len(fs), "piece_counts": sorted({f["count"] for f in fs}),
            "square_sides": sorted({f["side"] for f in fs}),
            "canvas_shapes": len({f["shape"] for f in fs}),
            "carrier_structures": len({f["geometry"] for f in fs}),
            "features": {s: dict(Counter({key: sum(bool(f[key]) for part, f in records if part == s)
                          for key in ("concave", "diagonal", "chord_cells", "unit_edges", "irregular_pieces")}))
                         for s in ("train", "test")}}
