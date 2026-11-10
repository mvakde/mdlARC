"""Translate marked wall gaps, then route through the completed bilateral board.

Only NumPy and the standard library are used. No parent generator is imported.
The input-only decoder has no generated-example cache or seed-dependent state.
"""
from collections import Counter, deque
import hashlib
import json

import numpy as np


BACKGROUND, TERMINAL, ROUTE, CORNER = 0, 2, 4, 8
PAYLOAD_POOL = (1, 3, 5, 6, 7, 9)
HOST_SEEDS = (0, 471100161, 70364829, 452165869, 211568582, 157112728,
              638665662, 378706442, 144433606, 958296978, 46080067,
              793210777, 25964932, 488353472, 957383935, 960305137,
              246480992, 221622672, 42330640, 96414646, 690891854,
              95708416, 374189184, 560714717, 93338905, 344045672,
              332737975, 432305430, 613898901, 430831746, 843746081,
              188623192, 816335685, 209744180, 313269277, 9290073,
              781631262, 442522435, 995125996, 566614831, 397558624,
              442538557)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "Grid must contain integer ARC colors")
    return a.astype(np.int8, copy=False)


def _norm(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(comp)))
    return result


def _path(board):
    """BFS returns a representative path and shortest-path count capped at two."""
    ends = [tuple(p) for p in np.argwhere(board == TERMINAL).tolist()]
    _require(len(ends) == 2, "Exactly two terminals required")
    start, goal = ends
    h, w = board.shape
    dist, ways, parent = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= q[0] < h and 0 <= q[1] < w):
                continue
            if board[q] not in (BACKGROUND, TERMINAL):
                continue
            d = dist[(r, c)] + 1
            if q not in dist:
                dist[q], ways[q], parent[q] = d, ways[(r, c)], (r, c)
                queue.append(q)
            elif dist[q] == d:
                ways[q] = min(2, ways[q] + ways[(r, c)])
    if goal not in dist:
        return [], 0
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1], ways[goal]


def _turns(path):
    moves = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(moves, moves[1:]))


def _decode(inp):
    """Recover frame, exterior oriented templates, and four-corner wall pockets."""
    inp = _grid(inp)
    candidates = []
    for color in PAYLOAD_POOL:
        rr, cc = np.where(inp == color)
        if not len(rr):
            continue
        t, l, b, r = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
        crop = inp[t:b, l:r]
        if min(crop.shape) < 7:
            continue
        if all(np.all(edge == color) for edge in (crop[0], crop[-1], crop[:, 0], crop[:, -1])):
            candidates.append((color, (t, l, b, r), crop.copy()))
    _require(len(candidates) == 1, "A unique solid rectangular frame is required")
    wall, bbox, board = candidates[0]
    _require(set(map(int, np.unique(board))) <= {BACKGROUND, wall, TERMINAL, CORNER},
             "Unexpected fixed board color")
    outside = inp != BACKGROUND
    t, l, b, r = bbox
    outside[t:b, l:r] = False
    templates = {}
    for comp in _components(outside):
        colors = {int(inp[p]) for p in comp}
        _require(len(colors) == 1, "Exterior templates must be separated and monochrome")
        color = next(iter(colors))
        _require(color in PAYLOAD_POOL and color != wall, "Invalid exterior payload role")
        shape = _norm(comp)
        _require(shape not in templates, "Only one exterior clue per oriented shape")
        templates[shape] = color
    _require(bool(templates), "Missing exterior templates")
    markers = set(map(tuple, np.argwhere(board == CORNER).tolist()))
    pockets, used_markers, used_cells = [], set(), set()
    rows = sorted({p[0] for p in markers})
    for i, top in enumerate(rows):
        for bottom in rows[i + 1:]:
            columns = sorted(c for rr, c in markers if rr == top and (bottom, c) in markers)
            for j, left in enumerate(columns):
                for right in columns[j + 1:]:
                    if bottom - top < 2 or right - left < 2:
                        continue
                    corners = {(top, left), (top, right), (bottom, left), (bottom, right)}
                    inside_markers = {p for p in markers if top <= p[0] <= bottom and left <= p[1] <= right}
                    if inside_markers != corners:
                        continue
                    box = board[top:bottom + 1, left:right + 1]
                    gap = box == BACKGROUND
                    cells = [(top + rr, left + cc) for rr, cc in np.argwhere(gap).tolist()]
                    if not cells or len(_components(gap)) != 1:
                        continue
                    # A pocket cuts through one wall: it meets exactly two opposite sides.
                    touch = (bool(gap[0].any()), bool(gap[-1].any()),
                             bool(gap[:, 0].any()), bool(gap[:, -1].any()))
                    if touch not in ((True, True, False, False), (False, False, True, True)):
                        continue
                    if np.any((box != BACKGROUND) & (box != wall) & (box != CORNER)):
                        continue
                    signature = _norm(cells)
                    if signature not in templates:
                        continue
                    _require(not used_markers & corners and not used_cells.intersection(cells),
                             "Overlapping or ambiguous marked pockets")
                    used_markers.update(corners)
                    used_cells.update(cells)
                    pockets.append((tuple(cells), signature, templates[signature]))
    _require(used_markers == markers and bool(pockets), "Every corner must identify one matched pocket")
    _require({s for _, s, _ in pockets} == set(templates), "Every exterior template must have a slot")
    completed = board.copy()
    for cells, shape, color in pockets:
        for p in cells:
            completed[p] = color
    return board, completed, pockets, templates, wall, bbox


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete all marked negative-space pockets; draw only the unique shortest route."""
    _, completed, _, _, _, _ = _decode(input_grid)
    path, count = _path(completed)
    _require(count == 1 and len(path) > 2, "Completed board must have a unique nontrivial shortest route")
    result = completed.copy()
    for p in path[1:-1]:
        result[p] = ROUTE
    return result


def _shape(rng, height, symmetric):
    """A connected gate spanning the wall thickness, with optional lateral branches."""
    kind = int(rng.integers(0, 3 if symmetric else 5))
    if kind == 0:
        cells = {(r, 0) for r in range(height)}
    elif kind == 1:
        thickness = int(rng.integers(2, 4))
        cells = {(r, c) for r in range(height) for c in range(thickness)}
    elif kind == 2:
        cells = {(r, 1) for r in range(height)}
        for r in rng.choice(np.arange(1, height - 1), size=int(rng.integers(1, height - 1)), replace=False):
            cells.update(((int(r), 0), (int(r), 2)))
    elif kind == 3:
        bend = int(rng.integers(1, height - 1))
        cells = {(r, 0) for r in range(bend + 1)} | {(r, 2) for r in range(bend, height)}
        cells.add((bend, 1))
    else:
        cells = {(r, 1) for r in range(height)}
        for r in range(1, height - 1):
            if rng.random() < 0.8:
                cells.add((r, int(rng.choice((0, 2)))))
        if len(cells) == height:
            cells.add((1, 0))
    return _norm(cells)


def _scatter(rng, board, templates):
    items = list(templates.items())
    rng.shuffle(items)
    strip_h = max(max(r for r, c in shape) + 1 for shape, _ in items)
    widths = [max(c for r, c in shape) + 1 for shape, _ in items]
    required_w = sum(widths) + len(items) - 1
    h, w = board.shape
    ch = h + strip_h + int(rng.integers(2, 5))
    cw = max(w + int(rng.integers(0, 5)), required_w + int(rng.integers(0, 5)))
    _require(ch <= 30 and cw <= 30, "Canvas packing exceeds ARC bounds")
    canvas = np.zeros((ch, cw), dtype=np.int8)
    br = int(rng.integers(0, ch - strip_h - h))
    bc = int(rng.integers(0, cw - w + 1))
    canvas[br:br + h, bc:bc + w] = board
    spare = cw - required_w
    gaps = rng.multinomial(spare, np.full(len(items) + 1, 1 / (len(items) + 1)))
    col = int(gaps[0])
    for i, ((shape, color), width) in enumerate(zip(items, widths)):
        sh = max(r for r, c in shape) + 1
        row = int(rng.integers(br + h + 1, ch - sh + 1))
        for dr, dc in shape:
            canvas[row + dr, col + dc] = color
        col += width + 1 + int(gaps[i + 1])
    return canvas


def _build_pair(rng, wall, levels, intro=False):
    for _ in range(240):
        width = int(rng.integers(17, 26))
        thicknesses = [int(rng.integers(3, 6)) for _ in range(levels)]
        # Leave room for all oriented exterior clues in one reserved strip.
        if sum(thicknesses) + levels + 3 + max(thicknesses) + 4 > 30:
            continue
        height = sum(thicknesses) + levels + 3
        board = np.full((height, width), wall, dtype=np.int8)
        hall_rows, templates, pending, orbits = [], {}, [], {}
        payload_colors = [c for c in PAYLOAD_POOL if c != wall]
        row = 1
        board[row, 1:-1] = BACKGROUND
        hall_rows.append(row)
        possible = True
        for index, thick in enumerate(thicknesses):
            top, bottom = row + 1, row + thick
            # Two real passages persist after filling; their distances vary by level.
            passage = int(rng.integers(1, 4))
            board[top:bottom + 1, passage] = BACKGROUND
            board[top:bottom + 1, width - 1 - passage] = BACKGROUND
            shape = _shape(rng, thick, symmetric=(index == 0 or intro))
            shw = max(c for r, c in shape) + 1
            box_w = shw + 2
            low, high = passage + 2, (width - 1) // 2 - box_w
            if high < low:
                possible = False
                break
            left = int(rng.integers(low, high + 1))
            reflected = _norm([(r, shw - 1 - c) for r, c in shape])
            orbit = min(shape, reflected)
            if orbit not in orbits:
                unused = [c for c in payload_colors if c not in orbits.values()]
                orbits[orbit] = int(rng.choice(unused))
            color = orbits[orbit]
            for slot_left, mask in ((left, shape), (width - left - box_w, reflected)):
                for rr, cc in ((top, slot_left), (top, slot_left + box_w - 1),
                               (bottom, slot_left), (bottom, slot_left + box_w - 1)):
                    board[rr, cc] = CORNER
                cells = [(top + dr, slot_left + 1 + dc) for dr, dc in mask]
                for p in cells:
                    board[p] = BACKGROUND
                pending.append((cells, color))
                templates[mask] = color
            # Dead-end recesses are unmarked floor and must survive completion.
            if rng.random() < 0.65:
                c = int(rng.integers(1, passage + 1))
                for cc in {c, width - 1 - c}:
                    if board[top, cc] == wall:
                        board[top, cc] = BACKGROUND
            row = bottom + 1
            board[row, 1:-1] = BACKGROUND
            hall_rows.append(row)
        if not possible or (levels >= 2 and len(orbits) < 2):
            continue
        completed = board.copy()
        for cells, color in pending:
            for p in cells:
                completed[p] = color
        # A central spine splits the inner halls, creating an above/below choice
        # for internal terminals. It preserves bilateral obstacle symmetry.
        internal = levels == 2 or (levels == 3 and rng.random() < 0.6)
        if internal:
            center = {(width - 1) // 2, width // 2}
            for rr in hall_rows[1:-1]:
                for cc in center:
                    board[rr, cc] = completed[rr, cc] = wall
            endpoint_row = int(rng.choice(hall_rows[1:-1]))
            left_columns = list(range(1, min(center)))
            right_columns = list(range(max(center) + 1, width - 1))
            endpoint_options = [(endpoint_row, x, endpoint_row, y)
                                for x in left_columns for y in right_columns]
        else:
            endpoint_options = [(hall_rows[0], x, hall_rows[-1], y)
                                for x in range(1, width - 1) for y in range(1, width - 1)]
        # Select endpoints only by bounded route checks; no expected path template.
        choices = rng.permutation(len(endpoint_options))
        for code in choices[:96]:
            r1, c1, r2, c2 = endpoint_options[int(code)]
            start, goal = (r1, c1), (r2, c2)
            board[start] = completed[start] = TERMINAL
            board[goal] = completed[goal] = TERMINAL
            path, count = _path(completed)
            raw, _ = _path(board)
            if count == 1 and len(raw) < len(path) and _turns(path) >= 2:
                break
            board[start] = completed[start] = BACKGROUND
            board[goal] = completed[goal] = BACKGROUND
        else:
            continue
        answer = completed.copy()
        for p in path[1:-1]:
            answer[p] = ROUTE
        canvas = _scatter(rng, board, templates)
        rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
        canvas, answer = np.rot90(canvas, rotation), np.rot90(answer, rotation)
        if reflect:
            canvas, answer = canvas[:, ::-1], answer[:, ::-1]
        canvas, answer = canvas.copy(), answer.copy()
        _require(np.array_equal(solve(canvas), answer), "Construction and input decoder disagree")
        return {"input": canvas, "output": answer}
    raise RuntimeError("Bounded construction exhausted")


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, "Illegal example count")
    _require(num_train > num_test, "num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "Seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    wall = int(rng.choice(PAYLOAD_POOL))
    plan = [1, 2] + [int(rng.integers(1, 3)) for _ in range(int(num_train) - 2)]
    rng.shuffle(plan)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, levels_list in (("train", plan), ("test", [3] * int(num_test))):
        for levels in levels_list:
            for _ in range(12):
                pair = _build_pair(rng, wall, levels, intro=(levels == 1))
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input retry bound exhausted")
    return puzzle


def _analyze(inp, out):
    board, completed, pockets, templates, wall, bbox = _decode(inp)
    path, count = _path(completed)
    raw, raw_count = _path(board)
    _require(count == 1 and len(path) > 2, "Shortest route is not unique")
    _require(bool(raw) and len(raw) < len(path), "Omitting template completion must shorten the route")
    _require(_turns(path) >= 2, "Route must turn around completed obstacles")
    expected = completed.copy()
    for p in path[1:-1]:
        expected[p] = ROUTE
    _require(np.array_equal(out, expected), "Output must be exactly completion plus shortest route")
    fixed = board != BACKGROUND
    _require(np.array_equal(out[fixed], board[fixed]), "Fixed walls, corners, or terminals changed")
    off_route = (completed == BACKGROUND) & (expected == BACKGROUND)
    _require(bool(off_route.any()), "Some unpainted floor must remain")
    symmetry = board.copy()
    symmetry[symmetry == TERMINAL] = BACKGROUND
    _require(np.array_equal(symmetry, symmetry[:, ::-1]) or
             np.array_equal(symmetry, symmetry[::-1]), "Scaffold and pockets must be bilateral")
    completed_sym = completed.copy()
    completed_sym[completed_sym == TERMINAL] = BACKGROUND
    _require(np.array_equal(completed_sym, completed_sym[:, ::-1]) or
             np.array_equal(completed_sym, completed_sym[::-1]), "Payloads must preserve bilateral layout")
    counts = Counter(shape for _, shape, _ in pockets)
    _require(any(n > 1 for n in counts.values()), "A clue must fill repeated slots")
    _require(len(pockets) > len(templates), "Exact-inventory assembly must be insufficient")
    _require(len(_components(completed == TERMINAL)) == 2, "Terminals must be separate")
    # A longer real route must exist, not merely a single forced corridor.
    alternative = False
    for p in path[1:-1]:
        blocked = completed.copy()
        blocked[p] = wall
        other, _ = _path(blocked)
        if other:
            _require(len(other) > len(path), "A second shortest route escaped counting")
            alternative = True
            break
    _require(alternative, "Shortest-path selection needs a longer alternative")
    ends = np.argwhere(completed == TERMINAL)
    a, b = map(tuple, ends.tolist())
    line = None
    if a[0] == b[0]:
        line = completed[a[0], min(a[1], b[1]) + 1:max(a[1], b[1])]
    elif a[1] == b[1]:
        line = completed[min(a[0], b[0]) + 1:max(a[0], b[0]), a[1]]
    baffle = line is not None and 1 <= int(np.count_nonzero(line != BACKGROUND)) <= 2
    return {"board_shape": board.shape, "canvas_shape": np.asarray(inp).shape,
            "wall": wall, "slots": len(pockets), "templates": len(templates),
            "payload_colors": len(set(templates.values())),
            "shapes": tuple(templates), "route_length": len(path) - 1,
            "raw_distance": len(raw) - 1, "raw_unique": raw_count == 1,
            "turns": _turns(path), "off_route": int(off_route.sum()),
            "baffled_endpoints": bool(baffle),
            "filled_cells": sum(len(cells) for cells, _, _ in pockets),
            "asymmetric_template": any(_norm([(r, -c) for r, c in shape]) != shape for shape in templates)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, walls, slots = set(), set(), {"train": set(), "test": set()}
    multicolor_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "Repeated input")
            seen.add(key)
            a = _analyze(inp, out)
            walls.add(a["wall"])
            slots[split].add(a["slots"])
            multicolor_demo |= split == "train" and a["payload_colors"] >= 2
    _require(len(walls) == 1, "Frame role must be consistent within a puzzle")
    _require({2, 4} <= slots["train"] and slots["train"] <= {2, 4}, "Training must show one and two wall levels")
    _require(slots["test"] == {6}, "Tests recombine three completed wall levels")
    _require(multicolor_demo, "Training must distinguish shape-to-color matching from one-color filling")


def _digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
              for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 seeds, bounded grids/retries; retain only compact signatures and metrics."""
    digests, boards, canvases, shapes, walls, lengths = (set() for _ in range(6))
    asymmetric = raw_unique = total = baffled = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"], pair["output"])
                boards.add(a["board_shape"])
                canvases.add(a["canvas_shape"])
                shapes.update(a["shapes"])
                walls.add(a["wall"])
                lengths.add(a["route_length"])
                asymmetric += a["asymmetric_template"]
                raw_unique += a["raw_unique"]
                baffled += a["baffled_endpoints"]
                total += 1
    _require(len(digests) == 42, "Seed diversity failed")
    _require(len(boards) >= 35 and len(canvases) >= 35 and len(shapes) >= 30,
             "Insufficient structural variation")
    _require(len(walls) >= 5 and len(lengths) >= 20 and asymmetric >= 10 and raw_unique >= 10,
             "Insufficient role, shape, or causal route variation")
    _require(20 <= baffled <= total - 20, "Internal and outer endpoint regimes must both occur")
    first = generate_puzzle(0)
    _require(_digest(first) == _digest(generate_puzzle(0)), "Seed reproduction failed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + a * 5 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Deliberately incorrect outputs must be rejected by the family checks.
    for kind in ("no_fill", "no_route", "flood", "terminal"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = bad["train"][0]
        board, complete, _, _, _, _ = _decode(pair["input"])
        if kind == "no_fill":
            pair["output"] = board
        elif kind == "no_route":
            pair["output"] = complete
        elif kind == "flood":
            pair["output"][complete == BACKGROUND] = ROUTE
        else:
            pair["output"][complete == TERMINAL] = ROUTE
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Validator accepted " + kind)
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(digests),
            "board_shapes": len(boards), "canvas_shapes": len(canvases),
            "oriented_shapes": len(shapes), "wall_colors": len(walls),
            "route_lengths": len(lengths), "asymmetric_pairs": asymmetric,
            "unique_prefill_routes": raw_unique, "baffled_endpoint_pairs": baffled}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
