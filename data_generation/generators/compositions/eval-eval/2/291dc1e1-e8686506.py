"""Priority mosaic repair: directed traversal is the tie breaker for exact assembly.

Only standard-library modules and NumPy are used. No files, seed, demonstrations,
or generation state are consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return (a.shape, a.tobytes())


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _orient_input(a):
    """Find the visibly distinguished fast arm; return logical scan coordinates.

    The reflection here is a coordinate change only. It is undone on the output,
    so the output undergoes a proper rotation relative to the physical input.
    """
    candidates = []
    for rotation in range(4):
        rotated = np.rot90(a, rotation)
        for reverse_slow in (False, True):
            b = rotated[::-1] if reverse_slow else rotated
            if min(b.shape) < 8:
                continue
            origin, fast, slow, bg = map(int, (b[0, 0], b[0, 1], b[1, 0], b[-1, -1]))
            if len({origin, fast, slow, bg}) != 4:
                continue
            expected = np.where(np.arange(1, b.shape[1]) % 2, fast, origin)
            if (np.array_equal(b[0, 1:], expected) and np.all(b[1:, 0] == slow)
                    and np.all(b[-1, 1:] == bg) and np.all(b[1:, -1] == bg)):
                candidates.append((b.copy(), bg, reverse_slow, rotation))
    _require(len(candidates) == 1, "Expected one alternating/solid coordinate key")
    return candidates[0]


def _parse(input_grid):
    a, bg, reverse_slow, rotation = _orient_input(_grid(input_grid))
    mask = a != bg
    mask[0, :] = False
    mask[:, 0] = False
    components = _components(mask)
    boards = []
    for i, cells in enumerate(components):
        r0, c0, r1, c1 = _bounds(cells)
        if min(r1 - r0, c1 - c0) < 5:
            continue
        box = a[r0:r1, c0:c1]
        frame = np.concatenate((box[0], box[-1], box[:, 0], box[:, -1]))
        if np.all(frame == frame[0]) and int(frame[0]) != bg and np.any(box == bg):
            boards.append((i, (r0, c0, r1, c1), box.copy()))
    _require(len(boards) == 1, "Expected one closed framed board")
    board_id, bbox, board = boards[0]
    r0, c0, r1, c1 = bbox
    # Enclosed disconnected fixed islands are also part of the board.
    exterior = []
    for i, cells in enumerate(components):
        if i == board_id or all(r0 <= r < r1 and c0 <= c < c1 for r, c in cells):
            continue
        _require(all(not (r0 <= r < r1 and c0 <= c < c1) for r, c in cells), "Piece crosses frame")
        colors = {int(a[cell]) for cell in cells}
        _require(len(colors) == 1, "Loose pieces must be monochrome")
        first_r, first_c = min(cells)
        exterior.append((first_r, first_c,
                         next(iter(colors)), _normalize(cells), cells))
    exterior.sort(key=lambda p: (p[0], p[1]))
    _require(2 <= len(exterior) <= 12, "Invalid inventory size")
    pieces = [(p[2], p[3]) for p in exterior]
    fixed_colors = set(map(int, board[board != bg]))
    key_colors = {bg, int(a[0, 0]), int(a[0, 1]), int(a[1, 0])}
    _require(not fixed_colors.intersection(key_colors), "Fixed and key palettes overlap")
    _require(not key_colors.intersection(c for c, s in pieces), "Payload and key palettes overlap")
    _require(not fixed_colors.intersection(c for c, s in pieces), "Fixed and payload palettes overlap")
    return {"logical": a, "background": bg, "reverse_slow": reverse_slow,
            "rotation": rotation, "board": board, "bbox": bbox, "pieces": pieces,
            "exterior": exterior}


def _assemble(board, bg, pieces):
    """Stable first-fit at the earliest hole; the anchor translation is unique."""
    result = board.copy()
    holes = set(map(tuple, np.argwhere(board == bg).tolist()))
    _require(sum(len(s) for c, s in pieces) == len(holes), "Inventory area mismatch")
    remaining = list(range(len(pieces)))
    placements, skips = [], 0
    while holes:
        target = min(holes)
        found = False
        for rank, index in enumerate(remaining):
            color, shape = pieces[index]
            # Earliest occupied cell, not bounding-box corner (e.g. upright T).
            ar, ac = min(shape)
            dr, dc = target[0] - ar, target[1] - ac
            cells = tuple((r + dr, c + dc) for r, c in shape)
            if all(cell in holes for cell in cells):
                for cell in cells:
                    result[cell] = color
                holes.difference_update(cells)
                remaining.pop(rank)
                placements.append((index, cells))
                skips += rank
                found = True
                break
        _require(found, "Ordered inventory cannot finish this board")
    _require(not remaining, "Unused inventory")
    return result, placements, skips


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed board, fast direction right, without reflection."""
    d = _parse(input_grid)
    answer, _, _ = _assemble(d["board"], d["background"], d["pieces"])
    return (answer[::-1] if d["reverse_slow"] else answer).copy()


def _pour(board, bg, pieces):
    """Counterfactual that preserves the cell inventory but forgets piece shapes."""
    out = board.copy()
    stream = [color for color, shape in pieces for _ in shape]
    for cell, color in zip(map(tuple, np.argwhere(board == bg)), stream):
        out[cell] = color
    return out


def _analyze(pair):
    d = _parse(pair["input"])
    board, bg, pieces = d["board"], d["background"], d["pieces"]
    answer, placements, skips = _assemble(board, bg, pieces)
    physical_answer = answer[::-1] if d["reverse_slow"] else answer
    _require(np.array_equal(physical_answer, pair["output"]), "Wrong output")
    fixed = board != bg
    _require(np.array_equal(board[fixed], answer[fixed]), "Fixed cells changed")
    _require(not np.any(answer == bg), "Unfilled cells")
    _require(sorted(index for index, cells in placements) == list(range(len(pieces))), "Piece multiplicity changed")
    counts = Counter()
    for index, cells in placements:
        color, shape = pieces[index]
        _require(_normalize(cells) == shape, "Piece rotated or reflected")
        counts[color] += len(cells)
    _require(counts == Counter(map(int, answer[~fixed])), "Per-color area changed")
    _require(np.array_equal(fixed, fixed[:, ::-1]) or np.array_equal(fixed, fixed[::-1]), "Stencil lacks bilateral symmetry")
    twins = [(i, j) for i, (c, s) in enumerate(pieces) for j in range(i)
             if s == pieces[j][1] and c != pieces[j][0]]
    _require(twins, "Need differently colored interchangeable pieces")
    # A counterfactual changes only priority, not the board or inventory multiset.
    i, j = twins[0]
    swapped = list(pieces)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    other, _, _ = _assemble(board, bg, swapped)
    _require(not np.array_equal(other, answer), "Priority has no effect")
    shape_effect = not np.array_equal(_pour(board, bg, pieces), answer)
    _require(shape_effect, "Shape-blind cell pouring accidentally suffices")
    gaps = _components(~fixed)
    shared = any(len({int(answer[cell]) for cell in cells}) > 1 for cells in gaps)
    signature = tuple(sorted((s for c, s in pieces)))
    return {"board_shape": board.shape, "canvas_shape": np.asarray(pair["input"]).shape,
            "pieces": len(pieces), "skips": skips, "shared_gap": shared,
            "lanes": len({p[0] for p in d["exterior"]}),
            "configuration": (d["rotation"], d["reverse_slow"]),
            "background": bg, "shape_signature": signature,
            "layout": (board.shape, fixed.tobytes()),
            "piece_shapes": tuple(s for c, s in pieces)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            features[split].append(_analyze({"input": a, "output": b}))
    train = features["train"]
    _require(any(f["shared_gap"] for f in train), "Training omits joint gap assembly")
    _require(any(f["skips"] > 0 for f in train), "Training omits skipping incompatible pieces")
    _require({f["configuration"][1] for f in train} == {False, True}, "Both slow directions need evidence")
    _require({f["configuration"][0] % 2 for f in train} == {0, 1}, "Both fast axes need evidence")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")


def _random_shape(rng, primary=False):
    if primary:
        h, w = 2, int(rng.integers(1, 4))
        return tuple((r, c) for r in range(h) for c in range(w))
    mode = int(rng.integers(5))
    if mode == 0:
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif mode == 1:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif mode == 2:
        h = int(rng.integers(2, 5))
        cells = {(0, c) for c in range(3)} | {(r, 1) for r in range(h)}
    elif mode == 3:
        cells = {(0, c) for c in range(int(rng.integers(2, 5)))}
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(2, 8))):
            choices = sorted({(r + dr, c + dc) for r, c in cells
                              for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                              if 0 <= r + dr < 4 and 0 <= c + dc < 4} - cells)
            cells.add(choices[int(rng.integers(len(choices)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    return _normalize(cells)


def _geometry(rng, groups, shared):
    h, w = int(rng.integers(7, 14)), int(rng.integers(8, 15))
    used, tiles = set(), []
    for group in range(groups):
        shape = _random_shape(rng, primary=(group == 0))
        sh, sw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        choices = []
        for r in range(1, h - sh):
            for c in range(1, (w - 2 * sw) // 2 + 1):
                base = {(r + dr, c + dc) for dr, dc in shape}
                mate = {(rr, w - 1 - cc) for rr, cc in base}
                both = base | mate
                if base & mate or both & used or len(both | used) > .65 * (h - 2) * (w - 2):
                    continue
                touches = any((rr + dr, cc + dc) in used for rr, cc in both
                              for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                if not shared and (touches or any((rr, cc + 1) in mate for rr, cc in base)):
                    continue
                choices.append((base, mate, touches))
        if not choices:
            return None
        touching = [p for p in choices if p[2]]
        pool = touching if shared and touching and group > 0 and rng.random() < .8 else choices
        base, mate, _ = pool[int(rng.integers(len(pool)))]
        used.update(base | mate)
        tiles.extend([tuple(sorted(base)), tuple(sorted(mate))])
    if rng.random() < .5:
        tiles = [tuple(sorted((c, r) for r, c in cells)) for cells in tiles]
        h, w = w, h
    return h, w, tiles


def _scaffold(rng, h, w, tiles, palette):
    bg, frame = map(int, (palette[0], palette[4]))
    secondary = int(palette[5])
    board = np.full((h, w), frame, dtype=np.int8)
    if rng.random() < .5:
        # A nested symmetric region, preserved exactly in the answer.
        inset = int(rng.integers(1, min(h, w) // 2))
        board[inset:h - inset, inset:w - inset] = secondary
    for cells in tiles:
        for cell in cells:
            board[cell] = bg
    return board


def _slots(rng, board_shape, count):
    """Variable shelves distributed around a movable board; no fixed panels."""
    bh, bw = board_shape
    for _ in range(32):
        h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
        r0, c0 = int(rng.integers(2, h - bh - 1)), int(rng.integers(2, w - bw - 1))
        rows, cols = [], []
        cursor = int(rng.integers(2, 4))
        while cursor + 4 < h:
            rows.append(cursor)
            cursor += 4 + int(rng.integers(1, 3))
        cursor = int(rng.integers(2, 4))
        while cursor + 4 < w:
            cols.append(cursor)
            cursor += 4 + int(rng.integers(1, 3))
        slots = [(r, c) for r in rows for c in cols
                 if r + 4 < r0 or r >= r0 + bh + 1 or c + 4 < c0 or c >= c0 + bw + 1]
        if len(slots) >= count:
            chosen = sorted(slots[i] for i in rng.choice(len(slots), count, replace=False))
            return (h, w), (r0, c0), chosen
    # Constructive packing fallback, not a fallback transformation.
    h = w = 30
    slots = [(r, c) for r in (2, 7, 12, 17, 22) for c in (2, 7, 12, 17, 22)
             if r >= bh + 3 or c >= bw + 3]
    _require(len(slots) >= count, "Packing fallback capacity exceeded")
    chosen = sorted(slots[i] for i in rng.choice(len(slots), count, replace=False))
    return (h, w), (2, 2), chosen


def _make_pair(rng, config, groups, shared, need_skip):
    # No files, unbounded search, or persistent caches. At most 120 small layouts.
    for _ in range(120):
        geometry = _geometry(rng, groups, shared)
        if geometry is None:
            continue
        h, w, tiles = geometry
        palette = rng.permutation(10)
        bg, origin, fast, slow = map(int, palette[:4])
        board = _scaffold(rng, h, w, tiles, palette)
        payload = list(map(int, palette[6:6 + int(rng.integers(2, 5))]))
        colors = [int(rng.choice(payload)) for _ in tiles]
        # The two primary rectangles have identical orientations and different
        # colors; their possible exchange makes reading priority indispensable.
        colors[0], colors[1] = payload[:2]
        base_pieces = [(color, _normalize(cells)) for color, cells in zip(colors, tiles)]
        successful = None
        for attempt in range(32):
            if attempt == 31:
                order = sorted(range(len(tiles)), key=lambda i: min(tiles[i]))
            else:
                order = list(map(int, rng.permutation(len(tiles))))
            pieces = [base_pieces[i] for i in order]
            try:
                answer, placements, skips = _assemble(board, bg, pieces)
            except ValueError:
                continue
            if need_skip and not skips:
                continue
            if np.array_equal(answer, _pour(board, bg, pieces)):
                continue
            gap_components = _components(board == bg)
            has_shared = any(len({int(answer[cell]) for cell in cells}) > 1 for cells in gap_components)
            if shared and not has_shared:
                continue
            successful = (pieces, answer)
            break
        if successful is None:
            continue
        pieces, answer = successful
        shape, (br, bc), slots = _slots(rng, board.shape, len(pieces))
        canvas = np.full(shape, bg, dtype=np.int8)
        canvas[br:br + h, bc:bc + w] = board
        for (color, piece), (r, c) in zip(pieces, slots):
            sw = max(cc for rr, cc in piece) + 1
            offset = int(rng.integers(5 - sw))
            for dr, dc in piece:
                canvas[r + dr, c + offset + dc] = color
        canvas[0, 0] = origin
        canvas[0, 1:] = np.where(np.arange(1, shape[1]) % 2, fast, origin)
        canvas[1:, 0] = slow
        rotation, reverse_slow = config
        inp = np.rot90(canvas[::-1] if reverse_slow else canvas, rotation).copy()
        out = (answer[::-1] if reverse_slow else answer).copy()
        pair = {"input": inp, "output": out}
        _analyze(pair)
        return pair
    raise RuntimeError("Could not construct a priority-completable mosaic within 120 attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(4))
    configs = [(start, False), ((start + 1) % 4, True)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                config = configs[i % 2]
                shared = i % 2 == 1
                groups = int(rng.integers(2, 5))
            else:
                # Recombine demonstrated axis and slow-direction factors.
                config = (start, True) if i % 2 == 0 else ((start + 1) % 4, False)
                shared = True
                groups = int(rng.integers(2, 6))
            for _ in range(8):
                pair = _make_pair(rng, config, groups, shared, need_skip=shared)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not produce a distinct input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """42 puzzles in memory, plus bounded metamorphic and rejection checks."""
    hashes, boards, canvases, layouts, shape_sets = set(), set(), set(), set(), set()
    shapes, configurations, backgrounds, counts, lanes = set(), set(), set(), set(), set()
    shared_count = skip_count = pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze(pair)
                digest.update(repr(_key(pair["input"])).encode())
                boards.add(f["board_shape"])
                canvases.add(f["canvas_shape"])
                layouts.add(f["layout"])
                shape_sets.add(f["shape_signature"])
                shapes.update(f["piece_shapes"])
                configurations.add(f["configuration"])
                backgrounds.add(f["background"])
                counts.add(f["pieces"])
                lanes.add(f["lanes"])
                shared_count += f["shared_gap"]
                skip_count += f["skips"] > 0
                pair_count += 1
        hashes.add(digest.hexdigest())
    _require(len(hashes) == 42 and len(layouts) >= 100, "Insufficient structural diversity")
    _require(len(boards) >= 30 and len(canvases) >= 30, "Insufficient size variation")
    _require(len(shapes) >= 35 and len(shape_sets) >= 100, "Insufficient shape variation")
    _require(len(configurations) == 8 and backgrounds == set(range(10)), "Key/color diversity collapsed")
    _require({4, 6, 8, 10} <= counts and len(lanes) >= 3, "Inventory/placement diversity collapsed")
    _require(shared_count >= 80 and skip_count >= 80, "Too little joint assembly")
    p = generate_puzzle(123, 2, 1)
    q = generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "Nondeterminism")
    for pair in p["train"] + p["test"]:
        original = pair["input"].copy()
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(original, k)), pair["output"]), "Proper rotation failure")
        recolor = np.array([8, 0, 6, 2, 9, 1, 5, 4, 7, 3], dtype=np.int8)
        _require(np.array_equal(solve(recolor[original]), recolor[pair["output"]]), "Color-role failure")
        _require(np.array_equal(original, pair["input"]), "Solver mutated input")
        d = _parse(original)
        pieces = d["pieces"]
        i, j = next((i, j) for i, (c, s) in enumerate(pieces) for j in range(i)
                    if s == pieces[j][1] and c != pieces[j][0])
        priority_edit = d["logical"].copy()
        for cell in d["exterior"][i][4]:
            priority_edit[cell] = pieces[j][0]
        for cell in d["exterior"][j][4]:
            priority_edit[cell] = pieces[i][0]
        edited_answer = solve(priority_edit)
        baseline = solve(d["logical"])
        _require(not np.array_equal(edited_answer, baseline), "Visible priority edit had no effect")
        _require(Counter(map(int, edited_answer.flat)) == Counter(map(int, baseline.flat)),
                 "Priority edit changed the inventory")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt answer accepted")
    # Removing one exterior cell must violate conservation, even if it disconnects a piece.
    d = _parse(p["test"][0]["input"])
    bad_input = d["logical"].copy()
    bad_input[d["exterior"][0][4][0]] = d["background"]
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing inventory cell accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(hashes),
            "board_shapes": len(boards), "canvas_shapes": len(canvases),
            "stencil_layouts": len(layouts), "oriented_piece_shapes": len(shapes),
            "inventory_shape_sets": len(shape_sets), "configurations": len(configurations),
            "piece_counts": sorted(counts), "lane_counts": sorted(lanes),
            "shared_gap_pairs": shared_count, "skip_pairs": skip_count}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
