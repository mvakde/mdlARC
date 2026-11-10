"""Rank-locked jigsaw bars. Standalone; standard library and NumPy only."""

import itertools
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _decode(input_grid):
    """Read the frame, carrier, sockets, and keyed inventory from input alone."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > inp.size // 2, "Background must occupy over half the canvas")
    frames = []
    for color in range(10):
        if color == bg or not counts[color]:
            continue
        rr, cc = np.where(inp == color)
        r, c, end_r, end_c = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
        b = inp[r:end_r, c:end_c]
        if min(b.shape) >= 5 and all(np.all(edge == color)
                                    for edge in (b[0], b[-1], b[:, 0], b[:, -1])):
            frames.append((r, c, end_r, end_c, color))
    _require(len(frames) == 1, "Expected one monochrome rectangular frame")
    r, c, end_r, end_c, frame = frames[0]
    original_board = inp[r:end_r, c:end_c].copy()
    axes = []
    for vertical in (False, True):
        b = original_board.T if vertical else original_board
        for row in range(1, b.shape[0] - 1):
            line = b[row, 1:-1]
            if np.all((line != bg) & (line != frame)) and line[0] == line[-1]:
                axes.append((vertical, row, int(line[0])))
    _require(len(axes) == 1, "Carrier must be unique")
    vertical, row, carrier = axes[0]
    board = original_board.T.copy() if vertical else original_board.copy()
    slots = []
    col = 1
    while col < board.shape[1] - 1:
        color = int(board[row, col])
        last = col + 1
        while last < board.shape[1] - 1 and board[row, last] == color:
            last += 1
        if color != carrier:
            slots.append((col, last - col, color))
        col = last
    _require(3 <= len(slots) <= 6, "Expected three to six sockets")
    slot_colors = {s[2] for s in slots}
    _require(len(slot_colors) == len(slots), "Socket colors are distinct within an example")
    _require(not slot_colors & {bg, frame, carrier}, "Role colors overlap")
    fixed = board != bg
    expected_fixed = np.full(board.shape, frame, dtype=np.int8)
    expected_fixed[row, 1:-1] = carrier
    for x, width, color in slots:
        expected_fixed[row, x:x + width] = color
    _require(np.array_equal(board[fixed], expected_fixed[fixed]), "Unexpected fixed cells")
    outside = inp != bg
    outside[r:end_r, c:end_c] = False
    exterior_colors = set(map(int, inp[outside]))
    material_colors = exterior_colors - {carrier}
    _require(len(material_colors) == 1, "Exterior inventory must use one neutral material")
    material = next(iter(material_colors))
    _require(material not in {bg, frame, carrier} | slot_colors, "Material color overlaps a role")
    pieces = []
    for component in _components(outside):
        cells = [(cc, rr) if vertical else (rr, cc) for rr, cc in component]
        marks = [(cc, rr) if vertical else (rr, cc)
                 for rr, cc in component if inp[rr, cc] == carrier]
        _require(bool(marks), "Every piece needs a registration stripe")
        mr = marks[0][0]
        mx, mw = min(p[1] for p in marks), len(marks)
        _require(set(marks) == {(mr, x) for x in range(mx, mx + mw)}, "Stripe must be a run")
        top, left = min(p[0] for p in cells), min(p[1] for p in cells)
        shape = tuple(sorted((rr - top, cc - left) for rr, cc in cells))
        marker = (mr - top, mx - left, mw)
        _require({(rr, cc) for rr, cc in shape if rr == marker[0]} ==
                 {(marker[0], x) for x in range(marker[1], marker[1] + mw)},
                 "Piece crosses the carrier only in its stripe")
        pieces.append({"shape": shape, "marker": marker,
                       "height": max(q[0] for q in shape) + 1,
                       "signature": (shape, marker)})
    _require(len(pieces) == len(slots), "Use exactly one piece per socket")
    _require(sum(len(p["shape"]) - p["marker"][2] for p in pieces) == int(np.sum(~fixed)),
             "Inventory area does not match missing board cells")
    return {"input": inp, "background": bg, "frame": frame, "carrier": carrier,
            "material": material, "bbox": (r, c, end_r, end_c), "vertical": vertical,
            "row": row, "board": board, "slots": slots, "pieces": pieces}


def _placements(data):
    board, bg, row = data["board"], data["background"], data["row"]
    hole_positions = {tuple(cell): i for i, cell in enumerate(np.argwhere(board == bg))}
    moves = []
    for x, width, _ in data["slots"]:
        options = []
        for j, piece in enumerate(data["pieces"]):
            mr, mx, mw = piece["marker"]
            if width != mw:
                continue
            cells = [(row + rr - mr, x + cc - mx)
                     for rr, cc in piece["shape"] if rr != mr]
            if all(q in hole_positions for q in cells):
                mask = sum(1 << hole_positions[q] for q in cells)
                options.append((j, mask, tuple(cells)))
        moves.append(options)
    return moves, (1 << len(hole_positions)) - 1


def _assemblies(data, ordered=True, limit=2, area_order=False):
    """Enumerate distinct colored answers, including exact use and full coverage.

    Removing ordered=True is also a family-specific counterfactual check.
    At most six pieces implies a small finite permutation search, without a
    timeout or a dependence on construction metadata.
    """
    moves, all_holes = _placements(data)
    pieces, answers, answer_keys = data["pieces"], [], set()

    def visit(slot, used, occupied, previous, chosen):
        if slot == len(moves):
            if occupied != all_holes:
                return False
            out = data["board"].copy()
            for i, (_, _, cells) in enumerate(chosen):
                color = data["slots"][i][2]
                for q in cells:
                    out[q] = color
            key = _key(out)
            if key not in answer_keys:
                answer_keys.add(key)
                answers.append((out, tuple(m[0] for m in chosen)))
            return len(answers) >= limit
        equivalent = set()
        for move in moves[slot]:
            j, mask, _ = move
            p = pieces[j]
            score = len(p["shape"]) if area_order else p["height"]
            if (used & (1 << j) or occupied & mask or
                    (ordered and score < previous) or p["signature"] in equivalent):
                continue
            equivalent.add(p["signature"])
            if visit(slot + 1, used | (1 << j), occupied | mask, score, chosen + [move]):
                return True
        return False

    visit(0, 0, 0, 0, [])
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _decode(input_grid)
    answers = _assemblies(data)
    _require(len(answers) == 1, "Input must have one rank-constrained colored completion")
    out = answers[0][0]
    return out.T.copy() if data["vertical"] else out.copy()


def _piece(cells, width):
    return {"cells": set(cells), "width": width}


def _profile_piece(rng, height, width, above_fraction):
    above = int(np.clip(round((height - 1) * above_fraction) + rng.integers(-1, 2), 0, height - 1))
    cells = set()
    style = int(rng.integers(4))
    for row in range(-above, height - above):
        left = right = 0
        if row != 0 and style:
            left = int(rng.integers(0, 3)) if style in (1, 3) else 0
            right = int(rng.integers(0, 3)) if style in (2, 3) else 0
        cells.update((row, c) for c in range(-left, width + right))
    return _piece(cells, width)


def _layout(rng, bridge_count, short_count, total_count, cap_depth, cap_above):
    """Construct an exact packing with a deliberately ambiguous unordered cap."""
    extra = total_count - bridge_count
    prefix = int(extra > 0 and rng.random() < 0.45)
    base = int(rng.integers(2 if prefix else 1, 6))
    above = int(rng.integers(base))
    lo, hi = -above, base - above - 1
    if cap_above:
        extension = range(lo - cap_depth, lo)
    else:
        extension = range(hi + 1, hi + cap_depth + 1)
    common_width = int(rng.choice([1, 1, 2, 3] if total_count <= 4 else [1, 1, 2]))
    core = {(r, c) for r in range(lo, hi + 1) for c in range(common_width)}
    straight = core | {(r, c) for r in extension for c in range(common_width)}
    cap = core | {(r, c) for r in extension
                  for c in range(-short_count * common_width, common_width)}
    group = ([_piece(core, common_width) for _ in range(short_count)] +
             [_piece(cap, common_width)] +
             [_piece(straight, common_width) for _ in range(bridge_count - short_count - 1)])
    fraction = above / max(1, base - 1)
    before = [_profile_piece(rng, int(rng.integers(1, base)), int(rng.integers(1, 4)), fraction)] if prefix else []
    after = [_profile_piece(rng, base + cap_depth + j + 1,
                            int(rng.integers(1, 4)), fraction)
             for j in range(extra - prefix)]
    pieces, starts, position = [], [], 2
    for p in before:
        left, right = min(c for _, c in p["cells"]), max(c for _, c in p["cells"])
        starts.append(position - left)
        pieces.append(p)
        position += right - left + 1 + int(rng.integers(1, 3))
    group_start = position
    for j, p in enumerate(group):
        starts.append(group_start + j * common_width)
        pieces.append(p)
    position += bridge_count * common_width + int(rng.integers(1, 3))
    for p in after:
        left, right = min(c for _, c in p["cells"]), max(c for _, c in p["cells"])
        starts.append(position - left)
        pieces.append(p)
        position += right - left + 1 + int(rng.integers(1, 3))
    min_r = min(r for p in pieces for r, _ in p["cells"])
    max_r = max(r for p in pieces for r, _ in p["cells"])
    return pieces, starts, 2 - min_r, (max_r - min_r + 5, position + 1)


def _free_layout(rng, count):
    """Other compatible keyed shapes; no required cap or shared cavity."""
    heights = sorted(map(int, rng.integers(1, 9, size=count)))
    pieces, starts, position = [], [], 2
    for height in heights:
        p = _profile_piece(rng, height, int(rng.integers(1, 4)), float(rng.random()))
        left, right = min(c for _, c in p["cells"]), max(c for _, c in p["cells"])
        starts.append(position - left)
        pieces.append(p)
        position += right - left + 1 + int(rng.integers(0, 3))
    min_r = min(r for p in pieces for r, _ in p["cells"])
    max_r = max(r for p in pieces for r, _ in p["cells"])
    return pieces, starts, 2 - min_r, (max_r - min_r + 5, position + 2)


def _scatter(rng, board, pieces, palette):
    bg, _, carrier, material = palette[:4]
    h, w = board.shape
    arrays = []
    for p in pieces:
        cells = p["cells"]
        lo, left = min(r for r, _ in cells), min(c for _, c in cells)
        ph, pw = max(r for r, _ in cells) - lo + 1, max(c for _, c in cells) - left + 1
        a = np.full((ph, pw), bg, dtype=np.int8)
        for rr, cc in cells:
            a[rr - lo, cc - left] = carrier if rr == 0 else material
        arrays.append(a)
    for attempt in range(36):
        ch = int(rng.integers(max(17, h + 3), 31)) if attempt < 12 else 30
        cw = int(rng.integers(max(17, w + 3), 31)) if attempt < 12 else 30
        foreground_area = h * w + sum(p["width"] for p in pieces)
        if ch * cw <= 2 * foreground_area:
            continue
        r = int(rng.integers(1, ch - h))
        c = int(rng.integers(1, cw - w))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        canvas[r:r + h, c:c + w] = board
        blocked = np.zeros(canvas.shape, dtype=bool)
        blocked[r - 1:r + h + 1, c - 1:c + w + 1] = True
        order = sorted(range(len(arrays)), key=lambda j: (-arrays[j].size, float(rng.random())))
        for j in order:
            a = arrays[j]
            ph, pw = a.shape
            candidates = [(rr, cc) for rr in range(1, ch - ph)
                          for cc in range(1, cw - pw)
                          if not np.any(blocked[rr:rr + ph, cc:cc + pw])]
            if not candidates:
                break
            rr, cc = candidates[int(rng.integers(len(candidates)))]
            canvas[rr:rr + ph, cc:cc + pw] = a
            blocked[rr - 1:rr + ph + 1, cc - 1:cc + pw + 1] = True
        else:
            if np.count_nonzero(canvas == bg) > canvas.size // 2:
                return canvas
    return None


def _make_pair(rng, palette, vertical, bridge_count, count, cap_depth, cap_above, free=False):
    for _ in range(80):
        short_count = int(rng.integers(1, bridge_count - 1))
        pieces, starts, row, shape = (_free_layout(rng, count) if free else
                                     _layout(rng, bridge_count, short_count, count, cap_depth, cap_above))
        if shape[0] > 25 or shape[1] > 25:
            continue
        if 2 * (shape[0] * shape[1] + sum(p["width"] for p in pieces)) >= 900:
            continue
        bg, frame, carrier, _ = palette[:4]
        board = np.full(shape, frame, dtype=np.int8)
        board[row, 1:-1] = carrier
        output = board.copy()
        for p, x, color in zip(pieces, starts, palette[4:]):
            for dr, dc in p["cells"]:
                board[row + dr, x + dc] = color if dr == 0 else bg
                output[row + dr, x + dc] = color
        inp = _scatter(rng, board, pieces, palette)
        if inp is None:
            continue
        if vertical:
            inp, output = inp.T.copy(), output.T.copy()
        pair = {"input": inp, "output": output}
        try:
            data = _decode(inp)
        except ValueError:
            continue  # A proposal can lack material cells or have unreadable roles.
        answers = _assemblies(data)
        if len(answers) == 1 and np.array_equal(answers[0][0], output.T if vertical else output):
            return pair
    raise RuntimeError("Bounded packing construction failed")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is outside its legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    fixed_palette = bool(rng.integers(2))
    axes = int(rng.integers(3))  # horizontal, vertical, or demonstrated mixture
    shift = int(rng.integers(2))  # longer caps, or more participating sockets
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            test = split == "test"
            bridge_count = int(rng.integers(5, 7) if test and shift == 1 else rng.integers(3, 5))
            if not test and index < 2:
                bridge_count = 3 + index
            count = 6 if test and shift == 1 else int(rng.integers(bridge_count, 6 if shift == 1 else 7))
            depth = int(rng.integers(3, 5) if test and shift == 0 else rng.integers(1, 3))
            cap_above = bool(rng.integers(2)) if test or index > 1 else index == 0
            vertical = axes == 1 if axes < 2 else bool(index % 2 if not test and index < 2 else rng.integers(2))
            colors = palette if fixed_palette else list(map(int, rng.permutation(10)))
            free = bool((index >= 2 if not test else shift == 1) and rng.random() < 0.35)
            for _ in range(12):
                pair = _make_pair(rng, colors, vertical, bridge_count, count, depth, cap_above, free)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not make distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _fit_is_necessary(data, assignment):
    """A sorted assignment with two nonidentical tied pieces exchanged fails fit."""
    options, _ = _placements(data)
    by_slot = [{j: (mask, cells) for j, mask, cells in o} for o in options]
    for a, b in itertools.combinations(range(len(assignment)), 2):
        pa, pb = (data["pieces"][assignment[i]] for i in (a, b))
        if pa["height"] != pb["height"] or pa["signature"] == pb["signature"]:
            continue
        candidate = list(assignment)
        candidate[a], candidate[b] = candidate[b], candidate[a]
        occupied, valid = 0, True
        for i, j in enumerate(candidate):
            if j not in by_slot[i] or occupied & by_slot[i][j][0]:
                valid = False
                break
            occupied |= by_slot[i][j][0]
        if not valid:
            return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _count(len(puzzle["train"]), "train count", 2, 10)
    _count(len(puzzle["test"]), "test count", 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "Invalid counts")
    seen, train_evidence, train_axes = set(), 0, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            canonical_out = out.T if d["vertical"] else out
            _require(canonical_out.shape == d["board"].shape, "Wrong board crop")
            fixed = d["board"] != d["background"]
            _require(np.array_equal(canonical_out[fixed], d["board"][fixed]), "Fixed cells changed")
            _require(not np.any(canonical_out == d["background"]), "Board has unfilled cells")
            answers = _assemblies(d)
            _require(len(answers) == 1 and np.array_equal(canonical_out, answers[0][0]), "Wrong assembly")
            if split == "train":
                train_axes.add(d["vertical"])
                train_evidence += (len(_assemblies(d, ordered=False)) > 1 and
                                   _fit_is_necessary(d, answers[0][1]))
            else:
                _require(d["vertical"] in train_axes, "Test orientation was not demonstrated")
    _require(train_evidence >= 2, "Two demonstrations must expose both order and shape fitting")


def self_test():
    # A hand-specified regression, independent of the constructor's painter.
    board = np.full((5, 7), 8, dtype=np.int8)
    board[1, 2:5] = 0
    board[2, 1:6] = [6, 1, 2, 3, 6]
    inp = np.zeros((12, 15), dtype=np.int8)
    inp[1:6, 1:8] = board
    inp[8, 2] = 6  # S: stripe only
    inp[8, 6], inp[9, 6] = 5, 6  # C: straight domino
    inp[8, 10:12], inp[9, 11] = 5, 6  # L: left-overhanging triomino
    expected = board.copy()
    expected[1, 2:5] = [2, 2, 3]
    for a, b in ((inp, expected), (inp.T, expected.T)):
        d = _decode(a)
        _require(np.array_equal(solve(a), b), "Hand-specified cap failed")
        _require(len(_assemblies(d, ordered=False)) == 2, "Lost unordered cap ambiguity")
        _require(not _assemblies(d, area_order=True), "Area order incorrectly solves the cap")
        _require(_fit_is_necessary(d, _assemblies(d)[0][1]), "Tie no longer requires fitting")
    shapes, canvases, inventory, layouts, axes, widths, counts, backgrounds = (set() for _ in range(8))
    puzzle_keys = set()
    shared_gaps = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(_key(pair[k]) for s in ("train", "test")
                          for pair in p[s] for k in ("input", "output"))
        puzzle_keys.add(signature)
        for s in ("train", "test"):
            for pair in p[s]:
                inp = pair["input"].copy()
                d = _decode(inp)
                predicted = solve(inp)
                _require(np.array_equal(predicted, pair["output"]), "Solver mismatch")
                _require(np.array_equal(inp, pair["input"]), "Solver mutated input")
                shapes.add(pair["output"].shape)
                canvases.add(inp.shape)
                axes.add(d["vertical"])
                backgrounds.add(d["background"])
                counts.add(len(d["pieces"]))
                widths.update(s[1] for s in d["slots"])
                inventory.update(p["signature"] for p in d["pieces"])
                layouts.add((d["board"].shape, (d["board"] == d["background"]).tobytes(),
                             tuple((s[0], s[1]) for s in d["slots"])))
                shared_gaps += any(len({int(predicted.T[q] if d["vertical"] else predicted[q]) for q in comp}) > 1
                                   for comp in _components(d["board"] == d["background"]))
    _require(len(puzzle_keys) == 42, "Seed diversity failed")
    _require(len(shapes) >= 50 and len(canvases) >= 30 and len(layouts) >= 150, "Geometry variation collapsed")
    _require(len(inventory) >= 100 and counts == {3, 4, 5, 6}, "Inventory variation collapsed")
    _require(axes == {False, True} and widths >= {1, 2, 3} and len(backgrounds) == 10, "Role variation collapsed")
    _require(shared_gaps >= 120, "Missing shared multicolor cavities")
    first = generate_puzzle(0, 5, 2)
    generate_puzzle(112, 2, 1)
    repeat = generate_puzzle(0, 5, 2)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], repeat[s])
                 for k in a), "Determinism failed")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(42 + a + b, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    for mode in ("fixed", "inventory", "answer"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
        pair = bad["test"][0]
        d = _decode(pair["input"])
        if mode == "inventory":
            q = tuple(np.argwhere(pair["input"] == d["material"])[0])
            pair["input"][q] = d["background"]
        else:
            canonical = pair["output"].T if d["vertical"] else pair["output"]
            q = (0, 0) if mode == "fixed" else tuple(np.argwhere(d["board"] == d["background"])[0])
            canonical[q] = d["background"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("Validator accepted a corrupted " + mode)
    return {"seeds": 42, "pairs": 210, "output_shapes": len(shapes),
            "canvas_shapes": len(canvases), "keyed_shapes": len(inventory),
            "layouts": len(layouts), "shared_multicolor_gaps": shared_gaps,
            "socket_counts": sorted(counts), "stripe_widths": sorted(widths),
            "background_colors": len(backgrounds), "orientations": len(axes),
            "manual_cap_regressions": 2}


if __name__ == "__main__":
    print(self_test())
