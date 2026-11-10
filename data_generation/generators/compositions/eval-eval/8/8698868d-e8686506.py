"""Cavity-locked patchwork: exact, oriented fragments assemble a topology budget.

The solver reads only the supplied grid. Generation and validation use the same
public rule, but construction also compares with its independently drawn board.
No parent code, file, seed, cached answer or learned palette is used by solve.
"""
from collections import Counter
from functools import lru_cache

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0, 348641458, 899229908, 152240425, 291360667, 611008340,
               729614239, 107620666, 620391477, 472639069, 587724198,
               167462827, 416945760, 14616076, 845505862, 954308838,
               764549944, 210273375, 709889244, 395403397, 150080721,
               290312103, 668826380, 576778741, 518073016, 623261014,
               698046948, 533540848, 369970863, 993032230, 40309227,
               585386413, 152572693, 883847106, 592965141, 44383932,
               17001590, 83247061, 904163503, 26814433, 102866849,
               814765932)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _N4
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return int(min(rr)), int(min(cc)), int(max(rr) - min(rr) + 1), int(max(cc) - min(cc) + 1)


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Bad grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _piece(grid, cells):
    r, c, _, _ = _box(cells)
    return tuple(sorted((int(rr - r), int(cc - c), int(grid[rr, cc])) for rr, cc in cells))


class _SearchLimit(Exception):
    """A uniqueness search did not finish; never accept this as uniqueness."""


def _room_options(h, w, budget, types, available, wall, void, enforce):
    """Enumerate local colored tilings, retaining exact type usage vectors."""
    by_cell = [[] for _ in range(h * w)]
    for t, shape in enumerate(types):
        sh = max(r for r, c, color in shape) + 1
        sw = max(c for r, c, color in shape) + 1
        for r in range(h - sh + 1):
            for c in range(w - sw + 1):
                cells = [(r + dr, c + dc, color) for dr, dc, color in shape]
                # All cavities must be internal to an ink perimeter.
                if any(color != wall and (rr in (0, h - 1) or cc in (0, w - 1))
                       for rr, cc, color in cells):
                    continue
                mask = sum(1 << (rr * w + cc) for rr, cc, color in cells)
                paint = sum((color + 1) << (4 * (rr * w + cc)) for rr, cc, color in cells)
                move = (t, mask, paint)
                for rr, cc, color in cells:
                    by_cell[rr * w + cc].append(move)
    nodes, results = 0, set()

    @lru_cache(maxsize=None)
    def search(remaining, counts, paint):
        nonlocal nodes
        nodes += 1
        if nodes > 20000 or len(results) > 512:
            raise _SearchLimit("local tiling bound exceeded")
        if not remaining:
            if enforce:
                mask = np.array([((paint >> (4 * i)) & 15) == void + 1
                                 for i in range(h * w)], dtype=bool).reshape(h, w)
                if len(_components(mask)) != budget:
                    return
            used = tuple(a - b for a, b in zip(available, counts))
            results.add((used, paint))
            if len(results) > 512:
                raise _SearchLimit("local result bound exceeded")
            return
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            options = [m for m in by_cell[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            bits ^= bit
        for t, mask, color_code in best:
            rest = list(counts)
            rest[t] -= 1
            search(remaining ^ mask, tuple(rest), paint | color_code)

    search((1 << (h * w)) - 1, available, 0)
    return sorted(results), nodes


def _assembly(board, rooms, pieces, wall, void, enforce=True):
    """Return at most two colored answers under translation and exact inventory."""
    inventory = Counter(pieces)
    types = sorted(inventory)
    available = tuple(inventory[t] for t in types)
    if sum(len(p) for p in pieces) != sum(h * w for r, c, h, w, k in rooms):
        return [], 0
    options, nodes = [], 0
    for r, c, h, w, k in rooms:
        opts, visited = _room_options(h, w, k, types, available, wall, void, enforce)
        nodes += visited
        if not opts:
            return [], nodes
        options.append(opts)
    order = sorted(range(len(rooms)), key=lambda i: len(options[i]))
    answers = {}
    painted = [0] * len(rooms)

    def search(depth, counts):
        nonlocal nodes
        nodes += 1
        if nodes > 100000:
            raise _SearchLimit("global inventory bound exceeded")
        if depth == len(order):
            if any(counts):
                return
            out = board.copy()
            for (r, c, h, w, k), paint in zip(rooms, painted):
                values = [((paint >> (4 * j)) & 15) - 1 for j in range(h * w)]
                out[r:r + h, c:c + w] = np.array(values, dtype=np.int8).reshape(h, w)
            answers[_key(out)] = out
            return
        i = order[depth]
        for used, paint in options[i]:
            if any(u > a for u, a in zip(used, counts)):
                continue
            painted[i] = paint
            search(depth + 1, tuple(a - u for a, u in zip(counts, used)))
            if len(answers) >= 2:
                return

    search(0, available)
    return list(answers.values()), nodes


def _scene(input_grid):
    """Recover roles from dominance, framed chambers and the exterior inventory."""
    inp = _grid(input_grid)
    colors, amounts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(amounts)])
    _require(int(amounts.max()) > inp.size // 2, "Background must occupy a strict majority")
    components = _components(inp != bg)
    candidates = []
    for index, comp in enumerate(components):
        br, bc, bh, bw = _box(comp)
        board = inp[br:br + bh, bc:bc + bw]
        frame = int(board[0, 0])
        if frame == bg or not np.all(_perimeter(board) == frame):
            continue
        other = set(map(int, np.unique(board))) - {bg, frame}
        if len(other) != 1:
            continue
        pip = other.pop()
        gaps = _components(board == bg)
        if not 2 <= len(gaps) <= 4:
            continue
        rooms, labels = [], []
        valid = True
        for gap in gaps:
            r, c, h, w = _box(gap)
            if min(h, w) < 4 or len(gap) != h * w or min(r, c) < 2 or r + h > bh - 2 or c + w > bw - 2:
                valid = False
                break
            neighbors = {(rr + dr, cc + dc) for rr, cc in gap for dr, dc in _N4} - set(gap)
            marker_cells = {cell for cell in neighbors if int(board[cell]) == pip}
            labels.append(marker_cells)
            rooms.append((r, c, h, w, len(marker_cells)))
        if not valid or set().union(*labels) != set(map(tuple, np.argwhere(board == pip).tolist())):
            continue
        if any(labels[i] & labels[j] for i in range(len(labels)) for j in range(i)):
            continue
        # Background strips between rooms are never allowed: every outside cell
        # of the board is fixed frame or a pip, and every pip has one owner.
        outside = [cells for i, cells in enumerate(components) if i != index]
        payload_colors = {int(inp[cell]) for cells in outside for cell in cells}
        if len(payload_colors) != 2 or payload_colors & {bg, frame, pip}:
            continue
        pieces = [_piece(inp, cells) for cells in outside]
        if not 4 <= len(pieces) <= 12:
            continue
        if sum(map(len, pieces)) != sum(h * w for r, c, h, w, k in rooms):
            continue
        candidates.append((bg, frame, pip, (br, bc, bh, bw), board.copy(), rooms,
                           pieces, outside, tuple(sorted(payload_colors))))
    _require(len(candidates) == 1, "Expected one framed board and exact exterior inventory")
    return candidates[0]


def _decode(input_grid):
    scene = _scene(input_grid)
    bg, frame, pip, bbox, board, rooms, pieces, outside, colors = scene
    feasible = []
    for wall, void in (colors, colors[::-1]):
        answers, nodes = _assembly(board, rooms, pieces, wall, void)
        _require(len(answers) < 2, "Cavity budgets admit distinct colored answers")
        if answers:
            feasible.append((answers[0], wall, void, nodes))
    _require(len(feasible) == 1, "Expected exactly one ink/void interpretation and completion")
    answer, wall, void, nodes = feasible[0]
    return answer, scene, wall, void, nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble all visible pieces once, using the visible chamber cavity counts."""
    return _decode(input_grid)[0].copy()


def _enclosed_in_piece(piece, void):
    occupied = {(r, c) for r, c, color in piece}
    h = max(r for r, c in occupied) + 1
    w = max(c for r, c in occupied) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c, color in piece:
        mask[r, c] = color == void
    return sum(all((r + dr, c + dc) in occupied for r, c in comp for dr, dc in _N4)
               for comp in _components(mask))


def _analyze(input_grid, output_grid, ablation=False):
    predicted, scene, wall, void, nodes = _decode(input_grid)
    out = _grid(output_grid)
    _require(np.array_equal(out, predicted), "Output differs from the unique cavity-locked assembly")
    bg, frame, pip, bbox, board, rooms, pieces, outside, colors = scene
    fixed = board != bg
    _require(np.array_equal(out[fixed], board[fixed]), "Fixed frame or pip changed")
    _require(not np.any(out == bg), "Unfilled chamber")
    _require(int(np.count_nonzero(np.asarray(input_grid) != bg)) == out.size,
             "Total painted inventory must equal the crop area")
    for color in colors:
        _require(int(np.count_nonzero(out[~fixed] == color)) ==
                 sum(sum(v == color for r, c, v in piece) for piece in pieces), "Inventory pixels changed")
    patterns = [out[r:r + h, c:c + w] == void for r, c, h, w, k in rooms]
    budgets = [k for r, c, h, w, k in rooms]
    areas = [int(p.sum()) for p in patterns]
    _require(all(len(_components(p)) == k for p, k in zip(patterns, budgets)), "Wrong cavity count")
    geometry_ambiguous = False
    if ablation:
        geometry_ambiguous = len(_assembly(board, rooms, pieces, wall, void, False)[0]) == 2
    return {
        "background": bg, "roles": (bg, frame, pip, wall, void), "board_shape": board.shape,
        "canvas_shape": np.asarray(input_grid).shape, "rooms": tuple(rooms), "budgets": tuple(budgets),
        "piece_count": len(pieces), "piece_shapes": tuple(tuple((r, c) for r, c, v in p) for p in pieces),
        "repeated_pieces": len(set(pieces)) < len(pieces),
        "jagged": any(len(p) < (max(r for r, c, v in p) + 1) * (max(c for r, c, v in p) + 1)
                      for p in pieces),
        "area_inversion": any(a < b and x > y for a, x in zip(budgets, areas)
                              for b, y in zip(budgets, areas)),
        "four_eight_witness": any(len(_components(p, True)) < k for p, k in zip(patterns, budgets)),
        "seam_closure": sum(_enclosed_in_piece(p, void) for p in pieces) != sum(budgets),
        "geometry_ambiguous": geometry_ambiguous,
        "patterns": tuple((p.shape, p.tobytes()) for p in patterns),
        "nodes": nodes,
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "Illegal pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            analyses[split].append(_analyze(inp, out, ablation=split == "train"))
    for flag in ("area_inversion", "four_eight_witness", "seam_closure", "geometry_ambiguous"):
        _require(any(a[flag] for a in analyses["train"]), "Training does not expose " + flag)
    return analyses


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _witness_masks(rng, h, w, cut):
    ring = np.zeros((h, w), dtype=bool)
    ring[1:-1, 1:-1] = True
    if rng.random() < 0.5 and h >= 5:
        ring[_choose(rng, (1, h - 2)), _choose(rng, (1, w - 2))] = False
    two = np.zeros_like(ring)
    row = int(rng.integers(1, h - 2))
    two[row, cut - 1] = two[row + 1, cut] = True
    # Optional growth retains the diagonal meeting and four/eight distinction.
    for _ in range(int(rng.integers(3))):
        options = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if two[r, c] or not any(two[r + dr, c + dc] for dr, dc in _N4):
                    continue
                trial = two.copy()
                trial[r, c] = True
                if len(_components(trial)) == 2 and len(_components(trial, True)) == 1:
                    options.append((r, c))
        if options:
            two[_choose(rng, options)] = True
    return ring, two


def _mask(rng, h, w, count):
    mask = np.zeros((h, w), dtype=bool)
    if count == 0:
        return mask
    if count == 1 and rng.random() < 0.35:
        mask[1:-1, 1:-1] = True
        return mask
    # Independent seeds of arbitrary parity, then connected growth without
    # merging the requested four-connected cavities.
    for _ in range(40):
        mask[:] = False
        for i in rng.permutation((h - 2) * (w - 2)):
            r, c = divmod(int(i), w - 2)
            r, c = r + 1, c + 1
            if not any(mask[r + dr, c + dc] for dr, dc in _N4):
                mask[r, c] = True
                if int(mask.sum()) == count:
                    break
        if int(mask.sum()) == count:
            break
    else:
        raise ValueError("cavity count exceeds room capacity")
    target = int(rng.integers(count, (h - 2) * (w - 2) + 1))
    for i in rng.permutation((h - 2) * (w - 2)):
        if mask.sum() >= target:
            break
        r, c = divmod(int(i), w - 2)
        r, c = r + 1, c + 1
        if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _N4):
            continue
        mask[r, c] = True
        if len(_components(mask)) != count:
            mask[r, c] = False
    return mask


def _fragment(rng, pattern, count, jagged, bg, axis=0):
    if axis:
        pieces = _fragment(rng, np.rot90(pattern), count, jagged, bg)
        return None if pieces is None else [np.rot90(p, 3).copy() for p in pieces]
    h, w = pattern.shape
    cuts = sorted(map(int, rng.choice(range(1, w), count - 1, replace=False)))
    profiles = np.tile([0] + cuts + [w], (h, 1))
    if jagged:
        for r in range(h):
            for j in range(1, count):
                profiles[r, j] = int(np.clip(cuts[j - 1] + rng.integers(-1, 2),
                                             profiles[r, j - 1] + 1, w - count + j))
    pieces = []
    for j in range(count):
        cells = [(r, c) for r in range(h) for c in range(profiles[r, j], profiles[r, j + 1])]
        # Narrow strips can become disconnected across a large jagged jump.
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell] = True
        if len(_components(mask)) != 1:
            return None
        r, c, sh, sw = _box(cells)
        obj = np.full((sh, sw), bg, dtype=np.int8)
        for rr, cc in cells:
            obj[rr - r, cc - c] = pattern[rr, cc]
        pieces.append(obj)
    return pieces


def _core_fragments(patterns, cut, bg, jagged, rng):
    h, w = patterns[0].shape
    profile = np.full(h, cut, dtype=int)
    if jagged:
        profile += rng.integers(-1, 2, h)
        profile = np.clip(profile, 1, w - 1)
    pieces = []
    for pattern in patterns:
        for side in (0, 1):
            cells = [(r, c) for r in range(h)
                     for c in (range(profile[r]) if side == 0 else range(profile[r], w))]
            r, c, sh, sw = _box(cells)
            obj = np.full((sh, sw), bg, dtype=np.int8)
            for rr, cc in cells:
                obj[rr - r, cc - c] = pattern[rr, cc]
            pieces.append(obj)
    return pieces


def _layout(rng, dimensions):
    n = len(dimensions)
    # Free-size rooms, uneven offsets and corridors, rather than equal square
    # tiles. Two rooms may be strips or staggered; three may make an L.
    strip = n < 4 and rng.random() < 0.4
    positions = []
    gap = int(rng.integers(3, 5))
    if strip:
        c = 2
        for h, w in dimensions:
            positions.append((2 + int(rng.integers(2)), c, h, w))
            c += w + gap
    else:
        top_h = max(h for h, w in dimensions[:2])
        left_w = max(dimensions[i][1] for i in range(0, n, 2))
        for i, (h, w) in enumerate(dimensions):
            r = 2 if i < 2 else 2 + top_h + gap
            c = 2 if i % 2 == 0 else 2 + left_w + gap
            positions.append((r + int(rng.integers(2)), c + int(rng.integers(2)), h, w))
    bh = max(r + h for r, c, h, w in positions) + 2
    bw = max(c + w for r, c, h, w in positions) + 2
    if max(bh, bw) > 27 or bh * bw + sum(h * w for h, w in dimensions) > 440:
        return None
    return (bh, bw), positions


def _scatter(rng, board, objects, bg):
    bh, bw = board.shape
    for attempt in range(24):
        blocked = np.zeros((30, 30), dtype=bool)
        canvas = np.full((30, 30), bg, dtype=np.int8)
        br = _choose(rng, (0, 30 - bh, int(rng.integers(31 - bh))))
        bc = _choose(rng, (0, 30 - bw, int(rng.integers(31 - bw))))
        canvas[br:br + bh, bc:bc + bw] = board
        blocked[max(0, br - 1):min(30, br + bh + 1), max(0, bc - 1):min(30, bc + bw + 1)] = True
        boxes = [(br, bc, bh, bw)]
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            obj = objects[i]
            h, w = obj.shape
            free = [(r, c) for r in range(31 - h) for c in range(31 - w)
                    if not np.any(blocked[r:r + h, c:c + w])]
            if not free:
                break
            edges = [(r, c) for r, c in free if r == 0 or c == 0 or r + h == 30 or c + w == 30]
            r, c = _choose(rng, edges if edges and rng.random() < 0.3 else free)
            canvas[r:r + h, c:c + w] = obj
            blocked[max(0, r - 1):min(30, r + h + 1), max(0, c - 1):min(30, c + w + 1)] = True
            boxes.append((r, c, h, w))
        else:
            if np.count_nonzero(canvas == bg) <= canvas.size // 2:
                continue
            r0 = min(r for r, c, h, w in boxes)
            c0 = min(c for r, c, h, w in boxes)
            r1 = max(r + h for r, c, h, w in boxes)
            c1 = max(c + w for r, c, h, w in boxes)
            if rng.random() < 0.85:
                r0 = int(rng.integers(r0 + 1))
                c0 = int(rng.integers(c0 + 1))
                r1 = int(rng.integers(r1, 31))
                c1 = int(rng.integers(c1, 31))
                cropped = canvas[r0:r1, c0:c1]
                if np.count_nonzero(cropped == bg) > cropped.size // 2:
                    canvas = cropped.copy()
            return canvas
    return None


def _draw_pair(rng, nrooms, rotation, palette, jagged, extra_count=None,
               fallback=False, witness=False, allowed_counts=(0, 1, 2, 3, 4)):
    bg, frame, pip, wall, void = map(int, palette[:5])
    h = 5 if fallback else int(rng.integers(5, 8 if nrooms == 2 else 6))
    w = 6 if fallback else int(rng.integers(5, 10 if nrooms == 2 else 7))
    cut = w // 2 if fallback else int(rng.integers(2, w - 1))
    if witness or fallback:
        masks = list(_witness_masks(rng, h, w, cut))
        budgets = [1, 2]
    else:
        cap = ((h - 2) * (w - 2) + 1) // 2
        budgets = list(map(int, rng.choice([k for k in allowed_counts if k <= cap], 2, replace=False)))
        masks = [_mask(rng, h, w, k) for k in budgets]
    dimensions = [(h, w), (h, w)]
    patterns = [np.where(mask, void, wall).astype(np.int8) for mask in masks]
    objects = _core_fragments(patterns, cut, bg, jagged and not fallback, rng)
    for i in range(2, nrooms):
        eh = 4 if fallback else int(rng.integers(4, 7))
        ew = 5 if fallback else int(rng.integers(4, 8))
        if extra_count is not None and not fallback:
            ew = max(ew, 2 + (2 * extra_count - 1 + eh - 3) // (eh - 2))
        cap = min(4, ((eh - 2) * (ew - 2) + 1) // 2)
        k = int(rng.integers(cap + 1)) if extra_count is None else min(extra_count, cap)
        if fallback and extra_count is None:
            k = 0 if i == 2 else 1
        mask = _mask(rng, eh, ew, k)
        pattern = np.where(mask, void, wall).astype(np.int8)
        pieces = _fragment(rng, pattern, 2 if fallback else int(rng.integers(2, 4)),
                           jagged and not fallback, bg, 0 if fallback else int(rng.integers(2)))
        if pieces is None:
            return None
        masks.append(mask)
        patterns.append(pattern)
        objects.extend(pieces)
        budgets.append(k)
        dimensions.append((eh, ew))
    layout = _layout(rng, dimensions)
    if layout is None:
        return None
    shape, positions = layout
    board = np.full(shape, frame, dtype=np.int8)
    output = board.copy()
    rooms = []
    # Only equal-sized core rooms can exchange their positions directly. All
    # other geometry is visible; random room order is set before packing.
    if rng.random() < 0.5:
        patterns[0], patterns[1] = patterns[1], patterns[0]
        budgets[0], budgets[1] = budgets[1], budgets[0]
    for (r, c, rh, rw), pattern, k in zip(positions, patterns, budgets):
        board[r:r + rh, c:c + rw] = bg
        output[r:r + rh, c:c + rw] = pattern
        border = ([(r - 1, cc) for cc in range(c, c + rw)] +
                  [(r + rh, cc) for cc in range(c, c + rw)] +
                  [(rr, c - 1) for rr in range(r, r + rh)] +
                  [(rr, c + rw) for rr in range(r, r + rh)])
        for j in rng.permutation(len(border))[:k]:
            board[border[int(j)]] = output[border[int(j)]] = pip
        rooms.append((r, c, rh, rw, k))
    pieces = [_piece(obj, [tuple(x) for x in np.argwhere(obj != bg)]) for obj in objects]
    try:
        answers, nodes = _assembly(board, rooms, pieces, wall, void)
    except _SearchLimit:
        return None
    if len(answers) != 1 or not np.array_equal(answers[0], output):
        return None
    canvas = _scatter(rng, board, objects, bg)
    if canvas is None:
        return None
    return {"input": np.rot90(canvas, rotation).copy(), "output": np.rot90(output, rotation).copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    vary_palette = bool(rng.random() < 0.75)
    base_rotation = int(rng.integers(4))
    # A visible-rule-preserving shift: either extra rooms, or a zero budget in
    # a new assembly. Neither uses a latent convention for interpreting inputs.
    shift = _choose(rng, ("growth", "zero", "interlocking"))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            nrooms = 2 if split == "train" else (4 if shift == "growth" else 3)
            if split == "train" and i >= 2:
                nrooms = int(rng.integers(2, 4))
            rotation = (base_rotation + i % 2) % 4 if split == "train" else int(rng.integers(4))
            jagged = (i == 1 or (i > 1 and rng.random() < 0.6)) if split == "train" else shift == "interlocking" or rng.random() < 0.5
            pair_palette = rng.permutation(10) if vary_palette else palette
            # Extra training rooms have positive budgets; zero is always a
            # meaningful held-out case under the zero curriculum.
            if split == "train":
                allowed_counts = (1, 2) if shift == "interlocking" else (1, 2, 3, 4)
                extra_count = _choose(rng, allowed_counts)
            else:
                allowed_counts = (0, 1, 2, 3, 4)
                extra_count = 0 if shift == "zero" else (3 if shift == "interlocking" else None)
            for attempt in range(160):
                pair = _draw_pair(rng, nrooms, rotation, pair_palette, jagged, extra_count,
                                  fallback=attempt >= 120, witness=split == "train" and i < 2,
                                  allowed_counts=allowed_counts)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                try:
                    a = _analyze(pair["input"], pair["output"], split == "train")
                except (AssertionError, _SearchLimit):
                    continue
                if split == "train" and i < 2 and not all(a[k] for k in
                        ("area_inversion", "four_eight_witness", "seam_closure", "geometry_ambiguous")):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded construction could not produce a unique scene")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    return puzzle


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed semantic checks, diversity, ablations and corruptions."""
    boards, canvases, outlines, patterns, palettes, room_counts, fragment_counts = (set() for _ in range(7))
    flags = Counter()
    fingerprints = set()
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        analyses = validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for a in analyses[split]:
                boards.add(a["board_shape"])
                canvases.add(a["canvas_shape"])
                outlines.update(a["piece_shapes"])
                patterns.update(a["patterns"])
                palettes.add(a["roles"])
                room_counts.add(len(a["rooms"]))
                fragment_counts.add(a["piece_count"])
                for flag in ("jagged", "repeated_pieces", "seam_closure", "four_eight_witness"):
                    flags[flag] += bool(a[flag])
                flags["zero"] += 0 in a["budgets"]
        _require(any(len(a["rooms"]) > 2 for a in analyses["test"]), "No combinatorial test shift")
    _require(len(fingerprints) == 42, "Seed diversity failed")
    _require(len(boards) >= 20 and len(canvases) >= 15, "Geometry collapsed")
    _require(len(outlines) >= 35 and len(patterns) >= 70 and len(palettes) >= 25, "Structural variety collapsed")
    _require(room_counts == {2, 3, 4} and len(fragment_counts) >= 4, "Inventory sizes collapsed")
    _require(flags["jagged"] >= 20 and flags["zero"] >= 5, "Missing compatible modes")
    original = generate_puzzle(0)
    generate_puzzle(1)
    _require(_same(original, generate_puzzle(0)), "Nondeterminism after an intervening seed")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123 + nt * 5 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "Ignored requested count")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    sample = generate_puzzle(0, 2, 1)
    # An invented pattern can obey every pip while violating B's exact inventory.
    # Remove one void pixel from a large connected cavity without changing its
    # topology; validator must still reject the resulting answer.
    witness = sample["train"][0]
    answer, scene, wall, void, nodes = _decode(witness["input"])
    invented = answer.copy()
    changed = False
    for r, c, h, w, k in scene[5]:
        if k != 1:
            continue
        region = invented[r:r + h, c:c + w]
        for rr, cc in np.argwhere(region == void):
            region[rr, cc] = wall
            if len(_components(region == void)) == 1:
                changed = True
                break
            region[rr, cc] = void
        if changed:
            break
    _require(changed, "No topology-only counterfactual available")
    _require(all(len(_components(invented[r:r + h, c:c + w] == void)) == k
                 for r, c, h, w, k in scene[5]), "Counterfactual changed topology")
    try:
        _analyze(witness["input"], invented)
    except AssertionError:
        pass
    else:
        raise AssertionError("Topology-only invented inventory accepted")
    # Altering one answer pixel must not silently alter the specified rule.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted fixed output accepted")
    inp = sample["train"][0]["input"].copy()
    scene = _scene(inp)
    inp[scene[7][0][0]] = scene[0]
    try:
        solve(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing inventory cell accepted")
    return {"seeds": 42, "board_shapes": len(boards), "canvas_shapes": len(canvases),
            "fragment_outlines": len(outlines), "cavity_patterns": len(patterns),
            "role_palettes": len(palettes), "modes": dict(flags), "independent_solver": False}


if __name__ == "__main__":
    print(self_test())
