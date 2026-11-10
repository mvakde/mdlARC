"""Ribbon itineraries: ordered two-row tapes decode and color a sparse route.

The solver reads only its argument. Generation paints labels from the proposed
route; input parsing independently reconstructs that route from tape lengths.
"""

from __future__ import annotations

import hashlib
import numbers

import numpy as np


_OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_HOST_SEEDS = (0, 184193158, 910506576, 362414566, 522579030, 687124600,
               243233042, 427619747, 840590126, 333668508, 168378202,
               447062946, 452691017, 828239612, 130124163, 204234689,
               140940898, 332681466, 940882065, 843291567, 238928965,
               80859082, 721853267, 234074280, 648513433, 728639095,
               382397338, 311030868, 28029639, 998074606, 624496718,
               881342266, 238580391, 730625333, 753345862, 161120020,
               860754201, 426262444, 992569643, 839624624, 408509876,
               788049802)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _transform(a, code):
    # Codes 0..3 are rotations; 4..7 first reflect horizontally, then rotate.
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _undo(a, code):
    a = np.rot90(a, -(code % 4))
    return (np.fliplr(a) if code >= 4 else a).copy()


def _read_key(a):
    candidates = []
    for code in range(8):
        b = _undo(a, code)
        if min(b.shape) < 8:
            continue
        origin, fast, slow, bg = map(int, (b[0, 0], b[0, 1], b[1, 0], b[-1, -1]))
        if len({origin, fast, slow, bg}) != 4:
            continue
        slow_arm = np.array([slow if r % 2 else bg for r in range(1, b.shape[0])])
        if (np.all(b[0, 1:] == fast) and np.array_equal(b[1:, 0], slow_arm)
                and np.all(b[-1, 1:] == bg) and np.all(b[1:, -1] == bg)):
            candidates.append((b, code, (bg, origin, fast, slow)))
    if len(candidates) != 1:
        raise ValueError("The solid/dashed L-key must have one interpretation")
    return candidates[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, todo = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    component.append(q)
                    todo.append(q)
        result.append(component)
    return result


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _segment(a, b):
    if not _aligned(a, b):
        raise ValueError("A route edge must be orthogonal or a 45-degree diagonal")
    dr, dc = b[0] - a[0], b[1] - a[1]
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(_distance(a, b) + 1)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0])
            or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1])
            or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return (_cross(a, b, c) * _cross(a, b, d) <= 0
            and _cross(c, d, a) * _cross(c, d, b) <= 0)


def _compatible(points, edge, selected):
    i, j = edge
    for k, l in selected:
        shared = {i, j} & {k, l}
        if shared:
            # Non-collinear edges meet only at their common endpoint. Collinear
            # backtracking must not share a run of pixels.
            if (_cross(points[i], points[j], points[k]) == 0
                    and _cross(points[i], points[j], points[l]) == 0):
                common = set(_segment(points[i], points[j])) & set(_segment(points[k], points[l]))
                if common != {points[t] for t in shared}:
                    return False
        elif _intersects(points[i], points[j], points[k], points[l]):
            return False
    return True


def _candidate_edges(points):
    landmarks = set(points)
    return [(i, j) for i, a in enumerate(points) for j in range(i + 1, len(points))
            if _aligned(a, points[j])
            and not set(_segment(a, points[j])[1:-1]) & landmarks]


def _routes(points, start, closed, lengths=None, limit=2):
    """Enumerate visible geometric routes, optionally constrained by tape lengths."""
    edges = _candidate_edges(points)
    adjacency = [[] for _ in points]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    solutions, visits = [], 0

    def visit(order, used, selected):
        nonlocal visits
        visits += 1
        # At most nine vertices are admitted by the public parser. Even a
        # complete graph has only sum(P(8,k), k=0..8)=109601 prefixes from a
        # fixed start, so this bound cannot reject a valid family member.
        if visits > 120000:
            raise ValueError("Route search exceeded its bounded family budget")
        if len(solutions) >= limit:
            return
        if len(order) == len(points):
            if closed:
                edge = (order[-1], start)
                if (start not in adjacency[order[-1]]
                        or (lengths is not None and _distance(points[edge[0]], points[start]) - 1 != lengths[-1])
                        or not _compatible(points, edge, selected)):
                    return
            solutions.append(order.copy())
            return
        step = len(order) - 1
        for j in adjacency[order[-1]]:
            edge = (order[-1], j)
            if (j not in used
                    and (lengths is None or _distance(points[edge[0]], points[j]) - 1 == lengths[step])
                    and _compatible(points, edge, selected)):
                visit(order + [j], used | {j}, selected + [edge])

    visit([start], {start}, [])
    return solutions


def _ordered_edges(order, closed):
    return list(zip(order[:-1], order[1:])) + ([(order[-1], order[0])] if closed else [])


def _read_input(input_grid):
    physical = _grid(input_grid)
    a, code, roles = _read_key(physical)
    bg = roles[0]
    mask = a != bg
    mask[0, :] = False
    mask[:, 0] = False
    boxes, tape_pixels = [], set()
    for component in _components(mask):
        r0, r1 = min(r for r, c in component), max(r for r, c in component) + 1
        c0, c1 = min(c for r, c in component), max(c for r, c in component) + 1
        if r1 - r0 == 2 and c1 - c0 >= 2 and len(component) == 2 * (c1 - c0):
            boxes.append((r0, r1, c0, c1))
            tape_pixels.update(component)
    if not 2 <= len(boxes) <= 8:
        raise ValueError("Expected 2..8 separated two-row strips")
    boxes.sort(key=lambda b: (b[0], b[2]))
    lanes = sorted({b[0] for b in boxes})
    if any(y - x < 3 for x, y in zip(lanes, lanes[1:])):
        raise ValueError("Strip lanes must have separated exact two-row bands")
    for r in lanes:
        lane = [b for b in boxes if b[0] == r]
        if any(b[2] - p[3] < 1 for p, b in zip(lane, lane[1:])):
            raise ValueError("Strips in a lane must be separated")
    if any(not (1 <= b[0] < b[1] <= a.shape[0] - 1
                and 1 <= b[2] < b[3] <= a.shape[1] - 1) for b in boxes):
        raise ValueError("Payloads cannot occupy key or far borders")
    # Undoing a reflected whole canvas reflects strip thickness. Correct this
    # so the net normalization of each physical strip is a proper rotation.
    blocks = [(a[r0:r1, c0:c1][::-1] if code >= 4 else a[r0:r1, c0:c1]).copy()
              for r0, r1, c0, c1 in boxes]
    residual = [p for p in map(tuple, np.argwhere(mask).tolist()) if p not in tape_pixels]
    groups = {}
    for p in residual:
        groups.setdefault(int(a[p]), []).append(p)
    source_groups = [(c, ps) for c, ps in groups.items() if len(ps) >= 3]
    marker_groups = [(c, ps) for c, ps in groups.items() if len(ps) == 1]
    if len(groups) != 2 or len(source_groups) != 1 or len(marker_groups) != 1:
        raise ValueError("The remaining pixels must be one landmark color and one companion marker")
    source_color, points = source_groups[0]
    marker_color, marker_points = marker_groups[0]
    points = sorted(points)
    marker = marker_points[0]
    anchors = [i for i, p in enumerate(points) if _distance(p, marker) == 1]
    if len(anchors) != 1:
        raise ValueError("The marker must be adjacent to exactly one source vertex")
    start = anchors[0]
    offset = (marker[0] - points[start][0], marker[1] - points[start][1])
    n = len(blocks)
    if len(points) not in (n, n + 1):
        raise ValueError("Vertex and strip counts must encode a full open or closed route")
    closed = len(points) == n
    lengths = [b.shape[1] for b in blocks]
    routes = _routes(points, start, closed, lengths)
    if len(routes) != 1:
        raise ValueError("The ordered strip lengths must select exactly one simple route")
    route = routes[0]
    edges = _ordered_edges(route, closed)
    source_pixels = set().union(*(set(_segment(points[i], points[j])) for i, j in edges))
    copy_pixels = {(r + offset[0], c + offset[1]) for r, c in source_pixels}
    if marker in source_pixels:
        raise ValueError("The companion anchor cannot lie on a source stroke")
    if not all(1 <= r < a.shape[0] - 1 and 1 <= c < a.shape[1] - 1
               for r, c in source_pixels | copy_pixels):
        raise ValueError("Both complete strands must fit inside the far borders")
    if (source_pixels | copy_pixels) & tape_pixels:
        raise ValueError("Instruction strips and destination strokes must be disjoint")
    return {"canonical": a, "code": code, "roles": roles, "background": bg,
            "boxes": boxes, "blocks": blocks, "lengths": lengths, "lanes": lanes,
            "points": points, "start": start, "route": route, "closed": closed,
            "offset": offset, "marker": marker, "source_color": source_color,
            "marker_color": marker_color, "source_pixels": source_pixels,
            "copy_pixels": copy_pixels}


def _maps(points, order, closed, blocks, source_color, marker_color, offset):
    source = {p: source_color for p in points}
    dr, dc = offset
    companion = {(r + dr, c + dc): marker_color for r, c in points}
    for block, (i, j) in zip(blocks, _ordered_edges(order, closed)):
        gap = _segment(points[i], points[j])[1:-1]
        if len(gap) != block.shape[1]:
            raise ValueError("A strip must contain exactly the interior cells of its edge")
        for k, p in enumerate(gap):
            source[p] = int(block[0, k])
            companion[(p[0] + dr, p[1] + dc)] = int(block[1, k])
    return source, companion


def _paint_output(shape, background, source, companion):
    out = np.full(shape, background, dtype=np.int8)
    for p, color in companion.items():
        out[p] = color
    for p, color in source.items():
        out[p] = color
    return out


def _output(d):
    source, companion = _maps(d["points"], d["route"], d["closed"], d["blocks"],
                              d["source_color"], d["marker_color"], d["offset"])
    return _transform(_paint_output(d["canonical"].shape, d["background"], source, companion), d["code"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the itinerary, fill both patterned strands, and remove the controls."""
    return _output(_read_input(input_grid))


def _normalize(points):
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    return [(r - r0, c - c0) for r, c in points]


def _orient_points(points, code):
    result = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = -c, r
        result.append((r, c))
    return _normalize(result)


def _propose_points(rng, n, closed, witness=False):
    if witness:
        w, h = map(int, rng.choice(np.arange(3, 8), size=2, replace=False))
        # The open witness has one diagonal side; together with the closed
        # rectangle this teaches all three supported link directions even at 2/1.
        points = ([(0, 0), (0, w), (h, w), (h, 0)] if closed else
                  [(0, 0), (0, w), (h, w + h), (h, h)])
    elif closed:
        h, w = map(int, rng.integers(9, 17, size=2))
        if n in (7, 8) and rng.random() < .35:
            w = int(rng.integers(9, 14))
            h = int(rng.integers(6, 13))
            if n == 7:
                gap = int(rng.integers(3, min(h - 3, w - 6) + 1))
                a = int(rng.integers(3, w - gap - 2))
                b, depth = a + gap, h - gap
                points = [(0, 0), (0, w), (h, w), (h, b), (depth, b), (h, a), (h, 0)]
            else:
                a = int(rng.integers(3, w - 5))
                b = int(rng.integers(a + 3, w - 2))
                depth = int(rng.integers(3, h - 2))
                points = [(0, 0), (0, w), (h, w), (h, b), (depth, b),
                          (depth, a), (h, a), (h, 0)]
        else:
            cuts = [0] * 4
            for k in rng.choice(4, size=n - 4, replace=False):
                cuts[int(k)] = int(rng.integers(3, 6))
            a, b, c, d = cuts
            points = [(0, a), (0, w - b), (b, w), (h - c, w),
                      (h, w - c), (h, d), (h - d, 0), (a, 0)]
            points = [p for i, p in enumerate(points) if p != points[i - 1]]
    else:
        points = [(0, 0)]
        for _ in range(n):
            dr, dc = _OFFSETS[int(rng.integers(8))]
            length = int(rng.integers(3, 9))
            points.append((points[-1][0] + dr * length, points[-1][1] + dc * length))
    points = _orient_points(points, int(rng.integers(8)))
    if closed:
        first = int(rng.integers(len(points)))
        points = points[first:] + points[:first]
    if rng.random() < .5:
        points = points[::-1]
    return points


def _make_geometry(rng, n, closed, witness=False):
    for _ in range(240):
        points = _propose_points(rng, n, closed, witness)
        if len(points) != n + int(not closed) or len(set(points)) != len(points):
            continue
        if max(max(r for r, c in points), max(c for r, c in points)) > 23:
            continue
        edges = _ordered_edges(list(range(len(points))), closed)
        if not all(_aligned(points[i], points[j]) and 3 <= _distance(points[i], points[j]) <= 15
                   for i, j in edges):
            continue
        lengths = [_distance(points[i], points[j]) - 1 for i, j in edges]
        try:
            routes = _routes(points, 0, closed, lengths)
            if routes != [list(range(len(points)))]:
                continue
        except ValueError:
            continue
        pixels = set().union(*(set(_segment(points[i], points[j])) for i, j in edges))
        options = []
        for offset in _OFFSETS:
            marker = (points[0][0] + offset[0], points[0][1] + offset[1])
            if marker in pixels or sum(_distance(p, marker) == 1 for p in points) != 1:
                continue
            copy = {(r + offset[0], c + offset[1]) for r, c in pixels}
            if witness and (len(copy - pixels) < 3 or not (copy & (pixels - set(points)))):
                continue
            options.append(offset)
        if options:
            return points, lengths, pixels, options[int(rng.integers(len(options)))]
    raise ValueError("Could not construct a uniquely decodable route")


def _blocks(rng, lengths, bg, witness):
    pool = [c for c in range(10) if c != bg]
    palette = rng.choice(pool, size=int(rng.integers(3 if witness else 1, 10)), replace=False)
    result = []
    for length in lengths:
        previous = [b for b in result if b.shape[1] == length]
        if previous and rng.random() < .22:
            block = previous[int(rng.integers(len(previous)))].copy()
        else:
            mode = int(rng.integers(4))
            if mode == 0:
                block = np.full((2, length), rng.choice(palette), dtype=np.int8)
            elif mode == 1:
                block = np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1)
            elif mode == 2:
                period = int(rng.integers(1, min(4, length) + 1))
                block = np.tile(rng.choice(palette, size=(2, period)), (1, length))[:, :length]
            else:
                block = rng.choice(palette, size=(2, length))
        result.append(block.astype(np.int8))
    if witness:
        index = int(rng.integers(len(result)))
        for _ in range(80):
            block = rng.choice(palette, size=result[index].shape).astype(np.int8)
            if (not np.array_equal(block, block[::-1])
                    and not np.array_equal(block, block[:, ::-1])
                    and not np.array_equal(block, block[::-1, ::-1])):
                result[index] = block
                break
        else:
            raise AssertionError("Could not make an asymmetric strip")
        # The four-link rectangle witness has two equal-width pairs. Repeat
        # one pair without overwriting the asymmetric block, teaching that
        # identical strips remain separate occurrences in the itinerary.
        if len(result) == 4:
            pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)
                     if i != index and j != index and lengths[i] == lengths[j]]
            i, j = pairs[int(rng.integers(len(pairs)))]
            result[j] = result[i].copy()
    return result


def _halo(pixels):
    return {(r + dr, c + dc) for r, c in pixels for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _place_ledger(rng, shape, lengths, forbidden, witness):
    h, w = shape
    n = len(lengths)
    for _ in range(90):
        lane_count = int(rng.integers(2 if witness else 1, min(n - int(witness), (h - 2) // 3, 6) + 1))
        rows = []
        options = list(range(1, h - 2))
        rng.shuffle(options)
        for r in options:
            if all(abs(r - p) >= 3 for p in rows):
                rows.append(r)
                if len(rows) == lane_count:
                    break
        if len(rows) != lane_count:
            continue
        rows.sort()
        counts = [1] * lane_count
        for _ in range(n - lane_count):
            counts[int(rng.integers(lane_count))] += 1
        placements, cursor = [], 0
        success = True
        for r, count in zip(rows, counts):
            group = lengths[cursor:cursor + count]
            if sum(group) + count - 1 > w - 2:
                success = False
                break

            def place(k, lo):
                if k == count:
                    return []
                length = group[k]
                room = sum(group[k:]) + count - k - 1
                starts = list(range(lo, w - room))
                rng.shuffle(starts)
                for c in starts:
                    cells = {(rr, cc) for rr in (r, r + 1) for cc in range(c, c + length)}
                    if cells & forbidden:
                        continue
                    rest = place(k + 1, c + length + 1)
                    if rest is not None:
                        return [(r, c)] + rest
                return None

            chosen = place(0, 1)
            if chosen is None:
                success = False
                break
            placements.extend(chosen)
            cursor += count
        if success:
            # A witness also distinguishes lane-major reading from sorting all
            # strips globally by the fast coordinate.
            fast_order = sorted(range(n), key=lambda i: (placements[i][1], placements[i][0]))
            if not witness or fast_order != list(range(n)):
                return placements
    return None


def _build_pair(rng, code, n, closed, witness=False):
    for geometry_attempt in range(35):
        points, lengths, pixels, offset = _make_geometry(rng, n, closed, witness)
        bg, origin, fast, slow = map(int, rng.choice(10, size=4, replace=False))
        source_color, marker_color = map(int, rng.choice([c for c in range(10) if c != bg], size=2, replace=False))
        blocks = _blocks(rng, lengths, bg, witness)
        copy = {(r + offset[0], c + offset[1]) for r, c in pixels}
        footprint = pixels | copy
        r0, c0 = min(r for r, c in footprint), min(c for r, c in footprint)
        fh, fw = max(r for r, c in footprint) - r0 + 1, max(c for r, c in footprint) - c0 + 1
        minimum_h, minimum_w = max(8, fh + 2), max(8, fw + 2, max(lengths) + 2)
        if max(minimum_h, minimum_w) > 30:
            continue
        for packing_attempt in range(12):
            h = int(rng.integers(max(minimum_h, 12), 31)) if packing_attempt < 8 else 30
            w = int(rng.integers(max(minimum_w, 12), 31)) if packing_attempt < 8 else 30
            dr = int(rng.integers(1, h - fh)) - r0
            dc = int(rng.integers(1, w - fw)) - c0
            absolute = [(r + dr, c + dc) for r, c in points]
            destination = {(r + dr, c + dc) for r, c in footprint}
            placements = _place_ledger(rng, (h, w), lengths, _halo(destination), witness)
            if placements is None:
                continue
            inp = np.full((h, w), bg, dtype=np.int8)
            inp[0, 0], inp[0, 1:] = origin, fast
            inp[1::2, 0] = slow
            for block, (r, c) in zip(blocks, placements):
                physical = block[::-1] if code >= 4 else block
                inp[r:r + 2, c:c + block.shape[1]] = physical
            for p in absolute:
                inp[p] = source_color
            marker = (absolute[0][0] + offset[0], absolute[0][1] + offset[1])
            inp[marker] = marker_color
            source, companion = _maps(absolute, list(range(len(points))), closed,
                                      blocks, source_color, marker_color, offset)
            out = _paint_output((h, w), bg, source, companion)
            pair = {"input": _transform(inp, code), "output": _transform(out, code)}
            try:
                d = _read_input(pair["input"])
            except ValueError:
                # Rare accidental landmark components can resemble a strip.
                # They are not valid visible encodings; propose new geometry.
                break
            assert np.array_equal(_output(d), pair["output"]), "Proposed route and visible decoding differ"
            if witness:
                if not any(p not in set(absolute) and p in companion and source[p] != companion[p] for p in source):
                    # Show source-over-companion priority on a newly filled cell.
                    blocks = _blocks(rng, lengths, bg, True)
                    continue
                sequence = [_key(b) for b in blocks]
                if any([sequence[i] for i in order] == sequence for order in _reading_alternatives(d).values()):
                    blocks = _blocks(rng, lengths, bg, True)
                    continue
            return pair
    raise ValueError("Bounded ribbon construction failed")


def _legal_integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _legal_integer("num_train", num_train, 2, 10)
    nq = _legal_integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    held = int(rng.integers(8))
    first = int(rng.choice([c for c in range(8) if c != held]))
    # Opposite fast axes and opposite handedness, with neither being held out.
    second_pool = [c for c in range(8) if c != held and c % 2 != first % 2 and (c >= 4) != (first >= 4)]
    second = int(rng.choice(second_pool))
    train_codes = [first, second] + [int(rng.choice([c for c in range(8) if c != held])) for _ in range(nt - 2)]
    test_codes = [held] * nq
    puzzle, seen = {"train": [], "test": []}, set()
    for split, codes in (("train", train_codes), ("test", test_codes)):
        for i, code in enumerate(codes):
            witness = split == "train" and i < 2
            closed = i == 1 if witness else bool(rng.integers(2))
            n = (4 if closed else 3) if witness else int(rng.integers(4 if closed else 2, 9))
            for _ in range(12):
                pair = _build_pair(rng, code, n, closed, witness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise ValueError("Distinct-input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _edge_signature(order, closed):
    return frozenset(tuple(sorted(e)) for e in _ordered_edges(order, closed))


def _reading_alternatives(d):
    groups = [[i for i, b in enumerate(d["boxes"]) if b[0] == r] for r in d["lanes"]]
    return {"fast_major": sorted(range(len(d["boxes"])), key=lambda i: (d["boxes"][i][2], d["boxes"][i][0])),
            "fast_reverse": [i for group in groups for i in group[::-1]],
            "slow_reverse": [i for group in groups[::-1] for i in group]}


def _features(d):
    source, companion = _maps(d["points"], d["route"], d["closed"], d["blocks"],
                              d["source_color"], d["marker_color"], d["offset"])
    source_vertices = set(d["points"])
    alternatives = _routes(d["points"], d["start"], d["closed"], limit=16)
    topologies = {_edge_signature(r, d["closed"]) for r in alternatives}
    sequence = [_key(b) for b in d["blocks"]]
    reading_witnesses = {name: [sequence[i] for i in order] != sequence
                         for name, order in _reading_alternatives(d).items()}
    directions = set()
    for i, j in _ordered_edges(d["route"], d["closed"]):
        a, b = d["points"][i], d["points"][j]
        directions.add("horizontal" if a[0] == b[0] else "vertical" if a[1] == b[1] else "diagonal")
    selected = _edge_signature(d["route"], d["closed"])
    candidates = _candidate_edges(d["points"])
    all_aligned = set().union(*(set(_segment(d["points"][i], d["points"][j])) for i, j in candidates))
    values = list(d["roles"]) + [d["source_color"], d["marker_color"]]
    ordered_points = [d["points"][i] for i in d["route"]]
    turns = [_cross(ordered_points[i - 1], ordered_points[i], ordered_points[(i + 1) % len(ordered_points)])
             for i in range(len(ordered_points))] if d["closed"] else []
    return {"code": d["code"], "shape": d["canonical"].shape,
            "closed": d["closed"], "strips": len(d["blocks"]),
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns),
            "lanes": tuple(sum(b[0] == r for b in d["boxes"]) for r in d["lanes"]),
            "lengths": tuple(d["lengths"]), "offset": d["offset"],
            "background": d["background"], "roles": tuple(values),
            "geometry": (d["closed"], tuple(_normalize(d["points"])), tuple(sorted(selected))),
            "directions": directions, "topologies_without_lengths": len(topologies),
            "rejected_chord_cells": len(all_aligned - set(source)),
            "order_witness": reading_witnesses["fast_major"],
            "reverse_fast_witness": reading_witnesses["fast_reverse"],
            "reverse_slow_witness": reading_witnesses["slow_reverse"],
            "asymmetric": any(not np.array_equal(b, b[::-1]) and not np.array_equal(b, b[:, ::-1])
                              and not np.array_equal(b, b[::-1, ::-1]) for b in d["blocks"]),
            "priority_witness": sum(p not in source_vertices and p in companion and source[p] != companion[p] for p in source),
            "visible_copy": len(set(companion) - set(source)),
            "duplicates": len(d["blocks"]) - len({_key(b) for b in d["blocks"]}),
            "palette": len(set(int(x) for b in d["blocks"] for x in b.flat)),
            "all_solid": all(np.all(b == b.flat[0]) for b in d["blocks"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "Repeated input"
            seen.add(_key(inp))
            d = _read_input(inp)
            assert np.array_equal(_output(d), out), "Incorrect ribbon or control removal"
            canonical_out = _undo(out, d["code"])
            assert all(canonical_out[p] == d["source_color"] for p in d["points"])
            assert canonical_out[d["marker"]] == d["marker_color"]
            assert np.all(canonical_out[0, :] == d["background"])
            assert np.all(canonical_out[:, 0] == d["background"])
            assert np.any(inp != out)
            features[split].append(_features(d))
    train = features["train"]
    assert {f["closed"] for f in train} == {False, True}, "Demonstrate count-encoded closure"
    assert {f["code"] % 2 for f in train} == {0, 1}, "Demonstrate both fast axes"
    assert {f["code"] >= 4 for f in train} == {False, True}, "Demonstrate proper rotation under both key handednesses"
    assert any(f["topologies_without_lengths"] >= 2 for f in train), "Lengths must resolve geometric topology"
    assert any(f["rejected_chord_cells"] > 0 for f in train), "Do not connect every alignment"
    assert any(f["order_witness"] for f in train), "Distinguish lane-major from fast-major reading"
    assert any(f["reverse_fast_witness"] for f in train), "Demonstrate ordering multiple strips within a lane"
    assert any(f["reverse_slow_witness"] for f in train), "Demonstrate directed lane order"
    assert any(f["duplicates"] for f in train), "Teach that repeated strips are separate occurrences"
    assert any(f["priority_witness"] > 0 for f in train), "Demonstrate source-over-copy occlusion"
    assert any(f["visible_copy"] >= 3 and f["asymmetric"] for f in train), "Both tape rows must matter visibly"
    assert set().union(*(f["directions"] for f in train)) == {"horizontal", "vertical", "diagonal"}, "Demonstrate every supported link direction"
    assert any(len(set(f["lengths"])) >= 2 for f in train)
    familiar = {f["code"] for f in train}
    assert all(f["code"] not in familiar for f in features["test"]), "Queries use a held-out key combination"
    return None


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for name in ("input", "output"):
                a = _grid(p[name])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed suite with geometric, ablation, and variation checks."""
    digests, features, structures = set(), [], set()
    row_swap_witnesses = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read_input(pair["input"])
                f = _features(d)
                f["split"] = split
                features.append(f)
                a = pair["input"]
                roles = {}
                structure = tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, structure))
                assert np.array_equal(solve(a.tolist()), pair["output"])
                # Color permutation changes no semantic convention.
                permutation = np.arange(9, -1, -1, dtype=np.int8)
                assert np.array_equal(solve(permutation[a]), permutation[pair["output"]])
                # Removing the B companion mechanism changes the actual answer.
                source, companion = _maps(d["points"], d["route"], d["closed"], d["blocks"],
                                          d["source_color"], d["marker_color"], d["offset"])
                source_only = _paint_output(d["canonical"].shape, d["background"], source,
                                            {d["marker"]: d["marker_color"]})
                if f["visible_copy"] > 1:  # A collinear translation can be hidden except for its marker.
                    assert not np.array_equal(source_only, _undo(pair["output"], d["code"]))
                if f["asymmetric"]:
                    swapped = [b[::-1].copy() for b in d["blocks"]]
                    wrong_source, wrong_copy = _maps(d["points"], d["route"], d["closed"], swapped,
                                                    d["source_color"], d["marker_color"], d["offset"])
                    wrong = _paint_output(d["canonical"].shape, d["background"], wrong_source, wrong_copy)
                    assert not np.array_equal(wrong, _undo(pair["output"], d["code"]))
                    row_swap_witnesses += 1
        # Minimum counts still retain the two informative demonstrations.
        assert any(f["topologies_without_lengths"] >= 2 for f in features[-5:-1])
    assert len(digests) == 42 and len(structures) >= 180
    assert row_swap_witnesses >= 84
    assert {f["code"] for f in features} == set(range(8))
    assert {f["offset"] for f in features} == set(_OFFSETS)
    assert {f["background"] for f in features} == set(range(10))
    assert len({f["shape"] for f in features}) >= 45
    assert len({f["geometry"] for f in features}) >= 100
    assert len({f["lanes"] for f in features}) >= 12
    assert {f["strips"] for f in features} >= set(range(2, 9))
    assert set().union(*(f["directions"] for f in features)) == {"horizontal", "vertical", "diagonal"}
    assert any(f["duplicates"] for f in features)
    assert any(f["all_solid"] for f in features)
    assert all(any(f["concave"] for f in features if f["split"] == split) for split in ("train", "test"))
    assert any(any(x % 2 for x in f["lengths"]) and any(x % 2 == 0 for x in f["lengths"]) for f in features)
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
    # Exercise all eight visible key configurations on an existing input.
    # Adjust physical thickness when changing handedness, preserving the two
    # properly normalized data channels and the intended drawing.
    sample = generate_puzzle(0)
    for p in sample["train"] + sample["test"]:
        d = _read_input(p["input"])
        canonical_out = _undo(p["output"], d["code"])
        for code in range(8):
            canonical = d["canonical"].copy()
            for b, (r0, r1, c0, c1) in zip(d["blocks"], d["boxes"]):
                canonical[r0:r1, c0:c1] = b[::-1] if code >= 4 else b
            assert np.array_equal(solve(_transform(canonical, code)), _transform(canonical_out, code))
    assert _intersects((0, 0), (1, 1), (0, 1), (1, 0)), "Reject half-cell diagonal crossings"
    assert len(_routes([(0, 0), (0, 3), (3, 3), (3, 0)], 0, True, [2] * 4)) == 2, "Do not break ambiguous length ties"
    damaged = generate_puzzle(0)
    pair = damaged["test"][0]
    d = _read_input(pair["input"])
    wrong = _undo(pair["output"], d["code"])
    wrong[d["marker"]] = d["background"]
    pair["output"] = _transform(wrong, d["code"])
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("A damaged ribbon was accepted")
    return {"seeds": 42, "pairs": len(features), "structures": len(structures),
            "shapes": len({f["shape"] for f in features}),
            "geometries": len({f["geometry"] for f in features}),
            "lane_populations": len({f["lanes"] for f in features})}
