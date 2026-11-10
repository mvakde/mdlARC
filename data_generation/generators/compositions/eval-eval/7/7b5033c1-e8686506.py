"""Ribbon-fed mosaics: ordered, translation-only packing of path color runs.

Only NumPy and the standard library are used. No parent code is imported.
The solver reads the scaffold, ribbon topology, and negative space from pixels.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.ravel()))


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        comp, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _shape(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _block_colors(a):
    if min(a.shape) < 2:
        return set()
    same = ((a[:-1, :-1] == a[1:, :-1]) & (a[:-1, :-1] == a[:-1, 1:])
            & (a[:-1, :-1] == a[1:, 1:]))
    return set(map(int, a[:-1, :-1][same]))


def _trace(cells):
    cells = set(cells)
    _require(len(cells) >= 2, "Ribbon is too short")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = [p for p in sorted(cells) if len(neighbors[p]) == 1]
    _require(len(ends) == 2 and all(len(v) in (1, 2) for v in neighbors.values()),
             "Ribbon must be an induced simple orthogonal chain")
    _require(ends[0][0] != ends[1][0], "Endpoint rows must differ")
    current, finish = sorted(ends)[0], sorted(ends)[1]
    path, previous = [], None
    while True:
        _require(current not in path, "Ribbon has a cycle")
        path.append(current)
        choices = [p for p in neighbors[current] if p != previous]
        if not choices:
            break
        previous, current = current, choices[0]
    _require(current == finish and len(path) == len(cells), "Ribbon must be connected")
    return path


def _runs(path, colors):
    cuts = [0] + [i for i in range(1, len(path)) if colors[i] != colors[i - 1]] + [len(path)]
    return [(int(colors[a]), _shape(path[a:b]), tuple(path[a:b]))
            for a, b in zip(cuts, cuts[1:])]


def _read(input_grid):
    inp = _grid(input_grid)
    hist = np.bincount(inp.ravel(), minlength=10)
    bg = int(hist.argmax())
    _require(hist[bg] * 2 > inp.size, "Background must be a strict majority")
    fixed_candidates = _block_colors(inp) - {bg}
    _require(len(fixed_candidates) == 1, "Exactly one scaffold color must contain a 2x2 block")
    fixed = next(iter(fixed_candidates))
    rr, cc = np.where(inp == fixed)
    top, left, bottom, right = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
    board = inp[top:bottom, left:right].copy()
    _require(min(board.shape) >= 3, "Board is too small")
    _require(all(board[p] == fixed for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))),
             "All four board corners must be fixed")
    _require(set(map(int, np.unique(board))) == {bg, fixed}, "Board contains foreign colors or has no gaps")
    cells = {tuple(map(int, p)) for p in np.argwhere((inp != bg) & (inp != fixed))}
    _require(all(not (top <= r < bottom and left <= c < right) for r, c in cells),
             "Ribbon must be outside the board")
    board_cells = {(r + top, c + left) for r, c in np.argwhere(board == fixed)}
    _require(all((r + dr, c + dc) not in board_cells for r, c in cells for dr, dc in _DIRS),
             "Ribbon must not touch the scaffold")
    path = _trace(cells)
    colors = [int(inp[p]) for p in path]
    _require(len(set(colors)) >= 2, "Ribbon must contain at least two colors")
    pieces = _runs(path, colors)
    holes = {tuple(map(int, p)) for p in np.argwhere(board == bg)}
    _require(len(holes) == len(path), "Exact inventory area must equal the gap area")
    return {"input": inp, "background": bg, "fixed": fixed, "bbox": (top, left, bottom, right),
            "board": board, "path": path, "colors": colors, "pieces": pieces, "holes": holes}


def _fill(board, background, pieces):
    """Align each piece's lexicographically first cell with the first vacancy.

    The first cell can differ from its bounding-box corner. Keeping that
    distinction is necessary for a bent piece extending left in lower rows.
    """
    out = board.copy()
    holes = {tuple(map(int, p)) for p in np.argwhere(board == background)}
    placements = []
    for color, shape, *_ in pieces:
        _require(bool(holes), "Surplus inventory")
        first = min(holes)
        anchor = shape[0]
        delta = first[0] - anchor[0], first[1] - anchor[1]
        target = {(r + delta[0], c + delta[1]) for r, c in shape}
        _require(target <= holes, "Next ribbon run cannot cover the first gap by translation")
        for cell in target:
            out[cell] = color
        holes.difference_update(target)
        placements.append(target)
    _require(not holes, "Missing inventory")
    return out, placements


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed board using only the supplied input pixels."""
    info = _read(input_grid)
    return _fill(info["board"], info["background"], info["pieces"])[0]


def _steps(path):
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]


def _sweep_matches(path, colors):
    values = dict(zip(path, colors))
    columns = sorted({c for _, c in path})
    for order in (columns, columns[::-1]):
        for reverse_last in (False, True):
            sweep = [p for c in order for p in sorted((p for p in path if p[1] == c),
                      reverse=reverse_last and c == order[-1])]
            if [values[p] for p in sweep] == colors:
                return True
    return False


def _bent(shape):
    return len({r for r, _ in shape}) > 1 and len({c for _, c in shape}) > 1


def _route_ok(path, colors, witness, above, teach_above):
    if path[0][0] == path[-1][0]:
        return False
    actual_above = min(r for r, _ in path) < path[0][0]
    if actual_above and not above:
        return False
    if teach_above and not actual_above:
        return False
    if not any(_bent(s) and len(s) >= 3 for _, s, _ in _runs(path, colors)):
        return False
    if witness:
        color_at = dict(zip(path, colors))
        if [color_at[p] for p in sorted(path)] == colors or _sweep_matches(path, colors):
            return False
    return True


def _random_walk(rng, length, persistence):
    path, occupied, previous = [(0, 0)], {(0, 0)}, None
    for _ in range(length - 1):
        candidates, weights = [], []
        for d in _DIRS:
            q = path[-1][0] + d[0], path[-1][1] + d[1]
            if q in occupied or sum((q[0] + a, q[1] + b) in occupied for a, b in _DIRS) != 1:
                continue
            rows = [r for r, _ in path] + [q[0]]
            cols = [c for _, c in path] + [q[1]]
            if max(rows) - min(rows) >= 12 or max(cols) - min(cols) >= 12:
                continue
            candidates.append((q, d))
            weights.append(persistence if d == previous else 1.0)
        if not candidates:
            return None
        weights = np.asarray(weights)
        q, previous = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        path.append(q)
        occupied.add(q)
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _skeletons():
    snake = [(0, 0)]
    for lane in range(6):
        direction = 1 if lane % 2 == 0 else -1
        for _ in range(10):
            snake.append((snake[-1][0], snake[-1][1] + direction))
        if lane < 5:
            snake.extend([(snake[-1][0] + 1, snake[-1][1]), (snake[-1][0] + 2, snake[-1][1])])
    winding = [(2, 2)]
    for (dr, dc), n in (((-1, 0), 2), ((0, 1), 4), ((1, 0), 4), ((0, -1), 4),
                       ((1, 0), 2), ((0, 1), 6), ((-1, 0), 4), ((0, 1), 2),
                       ((1, 0), 6), ((0, -1), 8), ((1, 0), 2), ((0, 1), 8)):
        for _ in range(n):
            winding.append((winding[-1][0] + dr, winding[-1][1] + dc))
    return snake, winding


def _make_route(rng, colors, witness, above, teach_above):
    persistence = float(rng.uniform(0.6, 4.5))
    for _ in range(180):
        path = _random_walk(rng, len(colors), persistence)
        if path is not None and _route_ok(path, colors, witness, above, teach_above):
            return path
    # Enumerated slices of two long induced chains are a bounded safety net.
    for skeleton in _skeletons():
        for offset in range(len(skeleton) - len(colors) + 1):
            base = skeleton[offset:offset + len(colors)]
            for reflect in (1, -1):
                path = [(r, reflect * c) for r, c in base]
                for _ in range(4):
                    if path[0][0] != path[-1][0]:
                        oriented = path if path[0][0] < path[-1][0] else path[::-1]
                        if _route_ok(oriented, colors, witness, above, teach_above):
                            return oriented
                    path = [(c, -r) for r, c in path]
    raise RuntimeError("No compatible bounded route construction")


def _inventory(rng, length, palette, recurrence, reserve_four=False):
    minimum = max(5, 2 + (length - 2 + 6) // 7)
    count = int(rng.integers(minimum, min(10, length - 2) + 1))
    singles = int(rng.integers(count - (3 if reserve_four else 2)))
    sizes = np.ones(count, dtype=int)
    for _ in range(length - count):
        candidates = [i for i in range(count) if i not in (singles, singles + 1) and sizes[i] < 7]
        sizes[int(rng.choice(candidates))] += 1
    if int(sizes.max()) < 3:
        target = next(i for i in range(count) if i not in (singles, singles + 1))
        for donor in range(count):
            if donor != target and sizes[donor] > 1 and sizes[target] < 3:
                sizes[donor] -= 1
                sizes[target] += 1
    colors = []
    for i in range(count):
        choices = [c for c in palette if not colors or c != colors[-1]]
        colors.append(int(rng.choice(choices)))
    if recurrence:
        colors[0], colors[1], colors[2] = palette[0], palette[1], palette[0]
        for i in range(3, count):
            if colors[i] == colors[i - 1]:
                colors[i] = next(c for c in palette if c != colors[i - 1])
    sequence = [c for c, n in zip(colors, sizes) for _ in range(int(n))]
    return sequence, singles


def _sharing(placements):
    union = set().union(*placements)
    groups = _components(union)
    return max(sum(bool(comp & part) for part in placements) for comp in groups)


def _shelf_layout(pieces, singles, min_share=1):
    """Guaranteed packing, with the two singleton runs touching the next run.

    Prefix/suffix shelves pack bounding boxes; the three-piece group explicitly
    shares a gap. This is used only after procedural packing exhausts its bound.
    """
    best = None
    for usable in range(8, 15):
        placements = []
        row = 2

        def shelves(items, start):
            result, r, c, height = [], start, 1, 0
            for _, shape, _ in items:
                sh, sw = max(x for x, _ in shape) + 1, max(y for _, y in shape) + 1
                if c + sw > usable + 1:
                    r, c, height = r + height + 1, 1, 0
                result.append({(r + x, c + y) for x, y in shape})
                c += sw + 1
                height = max(height, sh)
            return result, r + height + 1 if result else start

        prefix, row = shelves(pieces[:singles], row)
        placements.extend(prefix)
        shape = pieces[singles + 2][1]
        first_c = shape[0][1]
        start = max(1, first_c - 1)
        left = start + 2 - first_c
        placements += [{(row, start)}, {(row, start + 1)},
                       {(row + r, left + c) for r, c in shape}]
        row += max(r for r, _ in shape) + 2
        next_index = singles + 3
        if min_share >= 4:
            # Attach another piece beneath the group's lowest cell. It cannot
            # overlap an earlier piece, and its anchor is later in row order.
            if next_index >= len(pieces):
                # The singleton pair was chosen so there are two later runs
                # in packing-focused tests; other calls need only three.
                continue
            next_shape = pieces[next_index][1]
            last = placements[-1]
            bottom = max(r for r, _ in last)
            bottom_cols = sorted(c for r, c in last if r == bottom)
            options = []
            for c0 in bottom_cols:
                delta = bottom + 1, c0 - next_shape[0][1]
                part = {(r + delta[0], c + delta[1]) for r, c in next_shape}
                options.append(part)
            base = set().union(*placements[-3:])
            part = min(options, key=lambda p: max(c for _, c in p | base) - min(c for _, c in p | base))
            group = placements[-3:] + [part]
            shift = 1 - min(c for p in group for _, c in p)
            placements[-3:] = [{(r, c + shift) for r, c in p} for p in group]
            row = max(r for p in placements[-4:] for r, _ in p) + 2
            next_index += 1
        suffix, row = shelves(pieces[next_index:], row)
        placements.extend(suffix)
        h, w = row + 1, usable + 2
        if any(not (0 < r < h - 1 and 0 < c < w - 1) for part in placements for r, c in part):
            continue
        if best is None or h * w < best[0] * best[1]:
            best = h, w, placements
    _require(best is not None and best[0] <= 28, "Constructive shelf exceeded its bounds")
    return best


def _pack_board(rng, pieces, singles, style, min_share, max_share=None):
    largest_h = max(max(r for r, _ in s) + 1 for _, s, _ in pieces)
    largest_w = max(max(c for _, c in s) + 1 for _, s, _ in pieces)
    for _ in range(55):
        h = int(rng.integers(max(6, largest_h + 1), 15))
        w = int(rng.integers(max(6, largest_w + 1), 15))
        protected = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (0, 1), (1, 0), (1, 1)}
        used, placements, previous = set(), [], (-1, -1)
        for _, shape, _ in pieces:
            group_at = {}
            if max_share is not None:
                for comp in _components(used):
                    owners = {i for i, part in enumerate(placements) if comp & part}
                    for p in comp:
                        group_at[p] = owners
            sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
            candidates = []
            for r in range(h - sh + 1):
                for c in range(w - sw + 1):
                    cells = {(r + dr, c + dc) for dr, dc in shape}
                    first = (r + shape[0][0], c + shape[0][1])
                    if first <= previous or cells & (used | protected):
                        continue
                    touch = sum((ar + dr, ac + dc) in used for ar, ac in cells for dr, dc in _DIRS)
                    if style == "isolated" and touch:
                        continue
                    if max_share is not None:
                        joined = set()
                        for ar, ac in cells:
                            for dr, dc in _DIRS:
                                joined.update(group_at.get((ar + dr, ac + dc), ()))
                        if len(joined) + 1 > max_share:
                            continue
                    candidates.append((first, cells, touch))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            near = [x for x in candidates if x[0][0] <= candidates[0][0][0] + 1]
            if style != "isolated" and used and any(x[2] for x in near):
                near = [x for x in near if x[2]]
            choice = near[int(rng.integers(min(len(near), 14)))]
            previous = choice[0]
            placements.append(choice[1])
            used.update(choice[1])
        else:
            if _sharing(placements) >= min_share:
                return h, w, placements
    return _shelf_layout(pieces, singles, min_share)


def _embed(rng, board, path, colors, background):
    shape = _shape(path)
    ph, pw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
    bh, bw = board.shape
    min_r, min_c = min(r for r, _ in path), min(c for _, c in path)
    path = [(r - min_r, c - min_c) for r, c in path]
    layouts = []
    if bw + pw + 2 <= 30:
        layouts.append((max(bh, ph), bw + pw + 2, "horizontal"))
    if bh + ph + 2 <= 30:
        layouts.append((bh + ph + 2, max(bw, pw), "vertical"))
    _require(bool(layouts), "Board and ribbon cannot fit together")
    h0, w0, kind = layouts[int(rng.integers(len(layouts)))]
    h, w = min(30, h0 + int(rng.integers(1, 7))), min(30, w0 + int(rng.integers(1, 7)))
    while h * w <= 2 * board.size:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    _require(h <= 30 and w <= 30, "No majority-background canvas")
    if kind == "horizontal":
        gap = int(rng.integers(2, w - bw - pw + 1))
        offset = int(rng.integers(w - bw - pw - gap + 1))
        if rng.random() < 0.5:
            bc, pc = offset, offset + bw + gap
        else:
            pc, bc = offset, offset + pw + gap
        br, pr = int(rng.integers(h - bh + 1)), int(rng.integers(h - ph + 1))
    else:
        gap = int(rng.integers(2, h - bh - ph + 1))
        offset = int(rng.integers(h - bh - ph - gap + 1))
        if rng.random() < 0.5:
            br, pr = offset, offset + bh + gap
        else:
            pr, br = offset, offset + ph + gap
        bc, pc = int(rng.integers(w - bw + 1)), int(rng.integers(w - pw + 1))
    canvas = np.full((h, w), background, dtype=np.int8)
    canvas[br:br + bh, bc:bc + bw] = board
    for (r, c), color in zip(path, colors):
        canvas[pr + r, pc + c] = color
    return canvas


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    exchange = bool(rng.random() < 0.65)
    above = bool(rng.random() < 0.45)
    mode = str(rng.choice(("length", "packing")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            for attempt in range(16):
                pair_palette = list(map(int, rng.permutation(10))) if exchange else palette
                bg, fixed = pair_palette[:2]
                color_count = int(rng.integers(2, 6))
                payload = pair_palette[2:2 + color_count]
                if mode == "length":
                    length = int(rng.integers(30, 39) if split == "test" else rng.integers(16, 27))
                else:
                    length = int(rng.integers(18, 31))
                colors, singles = _inventory(rng, length, payload, split == "train" and index < 2,
                                              split == "test" and mode == "packing")
                witness = index == 0
                path = _make_route(rng, colors, witness, above, above and split == "train" and index == 0)
                pieces = _runs(path, colors)
                style = "isolated" if split == "train" and index % 3 == 1 else "mixed"
                min_share = 4 if split == "test" and mode == "packing" else (2 if witness else 1)
                max_share = 3 if split == "train" and mode == "packing" else None
                h, w, placements = _pack_board(rng, pieces, singles, style, min_share, max_share)
                board = np.full((h, w), fixed, dtype=np.int8)
                out = board.copy()
                for (color, _, _), cells in zip(pieces, placements):
                    for p in cells:
                        board[p], out[p] = bg, color
                inp = _embed(rng, board, path, colors, bg)
                pair = {"input": inp, "output": out}
                if witness:
                    evidence = _analyze_pair(pair)
                    if not (evidence["a_needed"] and evidence["b_needed"]
                            and evidence["row_sort_differs"] and evidence["sweep_failure"]
                            and evidence["reverse_differs"] and evidence["multicolor_gap"]):
                        continue
                if _key(inp) in seen:
                    continue
                seen.add(_key(inp))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Failed to construct a distinct input after bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_differs(info, pieces, answer):
    try:
        alternative = _fill(info["board"], info["background"], pieces)[0]
    except ValueError:
        return True
    return not np.array_equal(alternative, answer)


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
    info, answer = _read(pair["input"]), _grid(pair["output"])
    predicted, places = _fill(info["board"], info["background"], info["pieces"])
    _require(np.array_equal(predicted, answer), "Output differs from the input-only transformation")
    fixed_mask = info["board"] != info["background"]
    _require(np.array_equal(answer[fixed_mask], info["board"][fixed_mask]), "Fixed cells changed")
    _require(not np.any(answer == info["background"]), "Unfilled gap")
    _require(Counter(map(int, info["input"][info["input"] != info["background"]]))
             == Counter(map(int, answer.ravel())), "Foreground color inventory was not conserved")
    for (color, shape, _), cells in zip(info["pieces"], places):
        _require(_shape(cells) == shape and all(answer[p] == color for p in cells), "Piece was altered")
    pieces = info["pieces"]
    interchangeable = [(i, j) for i in range(len(pieces)) for j in range(i + 1, len(pieces))
                       if pieces[i][1] == pieces[j][1] and pieces[i][0] != pieces[j][0]]
    # Omitting A leaves a genuinely ambiguous geometric packing: these two
    # same-shaped pieces can exchange destinations, producing different colors.
    a_needed = bool(interchangeable)
    if interchangeable:
        i, j = interchangeable[0]
        alternative = list(pieces)
        alternative[i], alternative[j] = alternative[j], alternative[i]
        _require(_alternative_differs(info, alternative, answer), "Order witness did not affect answer")
    # Omitting B: consume the right number of first vacancies, ignoring shape.
    flat = info["board"].copy()
    gaps = sorted(info["holes"])
    cursor = 0
    for color, shape, _ in pieces:
        for p in gaps[cursor:cursor + len(shape)]:
            flat[p] = color
        cursor += len(shape)
    b_needed = not np.array_equal(flat, answer)
    geometry = _steps(info["path"])
    multicolor_gap = any(len({int(answer[p]) for p in comp}) >= 2 for comp in _components(info["holes"]))
    info.update({"placements": places, "a_needed": a_needed, "b_needed": b_needed,
                 "multicolor_gap": multicolor_gap,
                 "share": _sharing(places), "length": len(info["path"]), "run_count": len(pieces),
                 "turns": sum(a != b for a, b in zip(geometry, geometry[1:])),
                 "above": min(r for r, _ in info["path"]) < info["path"][0][0],
                 "recurrence": len({p[0] for p in pieces}) < len(pieces),
                 "sweep_failure": not _sweep_matches(info["path"], info["colors"]),
                 "reverse_differs": _alternative_differs(info, pieces[::-1], answer),
                 "row_sort_differs": _alternative_differs(info, sorted(pieces, key=lambda p: min(p[2])), answer),
                 "flat": flat})
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            k = _key(pair["input"])
            _require(k not in seen, "Duplicate inputs")
            seen.add(k)
            infos[split].append(info)
    tr, te = infos["train"], infos["test"]
    for split, rows in infos.items():
        _require(any(x["a_needed"] and x["b_needed"] for x in rows), f"{split} does not require both parents")
        _require(any(x["row_sort_differs"] and x["sweep_failure"] and x["reverse_differs"] for x in rows),
                 f"{split} lacks traversal witnesses")
        _require(any(x["share"] >= 2 and x["multicolor_gap"] for x in rows),
                 f"{split} lacks a shared multicolor gap")
    _require(any(x["recurrence"] for x in tr), "Training must teach separated repeated colors")
    if any(x["above"] for x in te):
        _require(any(x["above"] and x["reverse_differs"] for x in tr), "Higher-endpoint extension was not taught")
    # Either length extrapolation or a more jointly filled gap is the shift.
    _require(min(x["length"] for x in te) > max(x["length"] for x in tr)
             or min(x["share"] for x in te) > max(x["share"] for x in tr),
             "No meaningful structural test shift")


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s]) for s in ("train", "test"))


def self_test():
    """Bounded semantic, mutation, count, determinism, and 42-seed checks."""
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze_pair(pair)
                _require(np.array_equal(solve(np.asarray(pair["input"]).tolist()), pair["output"]), "List solve failed")
                records.append((split, a))
    _require(len(signatures) == 42, "Seeds did not produce distinct puzzles")
    _require(len({x["board"].shape for _, x in records}) >= 25, "Board dimensions lack variation")
    _require(len({x["input"].shape for _, x in records}) >= 60, "Canvas dimensions lack variation")
    _require(len({_shape(x["path"]) for _, x in records}) >= 130, "Ribbon geometries lack variation")
    _require(len({s for _, x in records for _, s, _ in x["pieces"]}) >= 40, "Piece shapes lack variation")
    _require(len({x["length"] for _, x in records}) >= 18, "Lengths lack variation")
    _require({x["background"] for _, x in records} == set(range(10)), "Palette roles lack variation")
    _require(len({x["run_count"] for _, x in records}) >= 5, "Inventory counts lack variation")
    _require(any(x["above"] for _, x in records), "No above-endpoint examples")
    _require(any(len(_components({tuple(p) for p in np.argwhere(x["board"] == x["fixed"])})) > 1
                 for _, x in records), "No disconnected scaffold")
    _require(any(s[0][1] > 0 for _, x in records for _, s, _ in x["pieces"]),
             "No piece whose first cell differs from its bounding-box corner")
    _require(any(len(set(s for c, s, _ in x["pieces"] if c == color)) > 1
                 for _, x in records for color in set(x["colors"])), "No recurring color with different shapes")
    # A six-cell, three-run family member lies outside the sampling ranges.
    # Its vertical piece reaches the bottom edge; its final singleton fills
    # an earlier row between the cells of the already placed vertical piece.
    board = np.full((4, 6), 8, dtype=np.int8)
    small_places = ({(1, 2), (1, 3)}, {(1, 4), (2, 4), (3, 4)}, {(2, 2)})
    expected = board.copy()
    for color, cells in zip((1, 2, 1), small_places):
        for p0 in cells:
            board[p0], expected[p0] = 0, color
    inp = np.zeros((7, 12), dtype=np.int8)
    inp[1:5, 1:7] = board
    for point, color in zip(((1, 8), (1, 9), (1, 10), (2, 10), (3, 10), (4, 10)), (1, 1, 2, 2, 2, 1)):
        inp[point] = color
    _require(np.array_equal(solve(inp), expected), "Solver relies on sampled lengths or run counts")
    # A hooked piece starts to the right of its bbox corner. Reflection is
    # illegal even when the piece has the same area and the same color.
    hooked = np.full((4, 6), 8, dtype=np.int8)
    hook_cells = {(1, 3), (2, 2), (2, 3)}
    for p0 in hook_cells:
        hooked[p0] = 0
    correct = ((0, 1), (1, 0), (1, 1))
    answer, positions = _fill(hooked, 0, [(2, correct)])
    _require(positions == [hook_cells] and all(answer[p0] == 2 for p0 in hook_cells), "Wrong piece anchor")
    try:
        _fill(hooked, 0, [(2, ((0, 0), (0, 1), (1, 1)))])
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal piece reflection was accepted")
    for cells in ({(0, 0), (0, 1), (0, 2)}, {(0, 1), (1, 1), (1, 0), (1, 2)},
                  {(0, 0), (0, 1), (1, 0), (1, 1)}, {(0, 0), (1, 0), (4, 4)}):
        try:
            _trace(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed ribbon or endpoint tie was accepted")
    # Corrupted outputs and deleted inventory cannot silently pass validation.
    p = generate_puzzle(123, 2, 1)
    info = _read(p["test"][0]["input"])
    bad_input = info["input"].copy()
    bad_input[info["path"][0]] = info["background"]
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError("Deleted inventory was accepted")
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted answer was accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 5 + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "Explicit counts ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    first = _puzzle_key(generate_puzzle(99, 5, 2))
    generate_puzzle(100, 3, 1)
    _require(first == _puzzle_key(generate_puzzle(99, 5, 2)), "Nondeterministic generation")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
