"""Walk the broken inlay: periodic path colors plus a finite rigid inventory.

No parent code is imported. All inference uses the query grid alone. Search
limits reject a proposal; they never turn an unfinished search into uniqueness.
"""
from collections import Counter
from functools import lru_cache

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class _SearchLimit(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Expected a 1..30 by 1..30 grid")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "Expected integer ARC colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr, dc in _DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(part)))
    return result


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _walk(cells):
    """Read an induced, unbranched open path from either endpoint."""
    cells = set(cells)
    neighbors = {q: sorted((q[0] + dr, q[1] + dc) for dr, dc in _DIRECTIONS
                           if (q[0] + dr, q[1] + dc) in cells) for q in cells}
    endpoints = sorted(q for q in cells if len(neighbors[q]) == 1)
    _require(len(cells) >= 12 and len(endpoints) == 2
             and all(1 <= len(n) <= 2 for n in neighbors.values()),
             "The groove must be a simple open path")
    path, previous, current = [], None, endpoints[0]
    while current is not None:
        path.append(current)
        options = [q for q in neighbors[current] if q != previous]
        previous, current = current, options[0] if options else None
        _require(len(path) <= len(cells), "Closed loop in groove")
    _require(len(path) == len(cells), "Disconnected groove")
    return tuple(path)


def _parse(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > inp.size // 2,
             "Canvas background must occupy more than half the grid")
    candidates = []
    for structural in map(int, colors):
        if structural == background:
            continue
        rr, cc = np.where(inp == structural)
        top, left = int(rr.min()), int(cc.min())
        bottom, right = int(rr.max()) + 1, int(cc.max()) + 1
        board = inp[top:bottom, left:right].copy()
        if min(board.shape) < 3:
            continue
        if not (np.all(board[0] == structural) and np.all(board[-1] == structural)
                and np.all(board[:, 0] == structural)
                and np.all(board[:, -1] == structural)):
            continue
        cells = [tuple(map(int, q)) for q in np.argwhere(board != structural)]
        try:
            path = _walk(cells)
        except ValueError:
            continue
        exterior_mask = inp != background
        exterior_mask[top:bottom, left:right] = False
        components = _components(map(tuple, np.argwhere(exterior_mask)))
        pieces = []
        for component in components:
            values = {int(inp[q]) for q in component}
            if len(values) != 1 or structural in values:
                break
            pieces.append((values.pop(), _normalize(component)))
        else:
            holes = tuple(sorted(q for q in path if board[q] == background))
            if pieces and sum(len(s) for _, s in pieces) == len(holes):
                candidates.append({"input": inp, "background": background,
                                   "structural": structural,
                                   "bbox": (top, left, bottom, right),
                                   "board": board, "path": path,
                                   "holes": holes, "pieces": tuple(pieces),
                                   "exterior": tuple(components)})
    _require(len(candidates) == 1, "Expected one identifiable inlay board")
    return candidates[0]


def _phase_clues(parsed, period):
    clues = [-1] * period
    board, bg = parsed["board"], parsed["background"]
    for i, q in enumerate(parsed["path"]):
        color = int(board[q])
        if color == bg:
            continue
        phase = i % period
        if clues[phase] not in (-1, color):
            return None
        clues[phase] = color
    return tuple(clues)


def _moves(parsed):
    """All translation placements, retaining shape/color multiplicities."""
    holes = parsed["holes"]
    indices = {q: i for i, q in enumerate(holes)}
    inventory = Counter(parsed["pieces"])
    types = tuple(sorted(inventory))
    by_cell = [[] for _ in holes]
    h, w = parsed["board"].shape
    path_index = {q: i for i, q in enumerate(parsed["path"])}
    for t, (color, shape) in enumerate(types):
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        for r in range(h - sh + 1):
            for c in range(w - sw + 1):
                cells = tuple((r + dr, c + dc) for dr, dc in shape)
                if not all(q in indices for q in cells):
                    continue
                ids = tuple(indices[q] for q in cells)
                mask = sum(1 << i for i in ids)
                paint = sum((color + 1) << (4 * i) for i in ids)
                move = t, mask, paint, color, tuple(path_index[q] for q in cells)
                for i in ids:
                    by_cell[i].append(move)
    return types, tuple(inventory[t] for t in types), by_cell


def _assemblies(parsed, period=None, prepared=None, node_limit=None):
    """Return up to two distinct colored fills, with optional path periodicity."""
    clues = () if period is None else _phase_clues(parsed, period)
    if clues is None:
        return (), 0
    _, initial_counts, by_cell = _moves(parsed) if prepared is None else prepared
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts, phases):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _SearchLimit("Assembly search exceeded its proof budget")
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            choices = []
            for move in by_cell[bit.bit_length() - 1]:
                t, mask, _, color, steps = move
                if not counts[t] or mask & remaining != mask:
                    continue
                if period is not None and any(phases[i % period] not in (-1, color)
                                              for i in steps):
                    continue
                choices.append(move)
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint, color, steps in best:
            rest = list(counts)
            rest[t] -= 1
            next_phases = list(phases)
            if period is not None:
                for i in steps:
                    next_phases[i % period] = color
            for suffix in search(remaining ^ mask, tuple(rest), tuple(next_phases)):
                answers.add(paint | suffix)
                if len(answers) >= 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    answers = search((1 << len(parsed["holes"])) - 1, initial_counts, clues)
    return answers, nodes


def _decode(parsed, code):
    answer = parsed["board"].copy()
    for i, q in enumerate(parsed["holes"]):
        answer[q] = ((code >> (4 * i)) & 15) - 1
    return answer


def _infer(parsed, node_limit=None):
    prepared = _moves(parsed)
    answers, supported = set(), []
    # At least three full cycles; incomplete endpoints are included by modulo.
    for period in range(1, len(parsed["path"]) // 3 + 1):
        codes, _ = _assemblies(parsed, period, prepared, node_limit)
        if codes:
            supported.append(period)
            answers.update(codes)
            _require(len(answers) == 1,
                     "Periodicity and inventory allow different colored answers")
    _require(len(answers) == 1, "No periodic exact-inventory completion")
    return _decode(parsed, next(iter(answers))), tuple(supported)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore and crop the unique periodic inlay using only the input."""
    return _infer(_parse(input_grid))[0]


def _least_period(values):
    for p in range(1, len(values) // 3 + 1):
        if all(values[i] == values[i % p] for i in range(len(values))):
            return p
    raise ValueError("Answer has no supported period")


def _inventory_supports_colors(parsed, target, node_limit=None):
    """Can this exact oriented inventory realize a prescribed colored fill?"""
    types, counts, by_cell = _moves(parsed)
    restricted = [[move for move in moves
                   if all(int(target[parsed["path"][i]]) == move[3] for i in move[4])]
                  for moves in by_cell]
    codes, _ = _assemblies(parsed, prepared=(types, counts, restricted), node_limit=node_limit)
    return bool(codes)


def _analyze(input_grid, output_grid, check_ablations=True, node_limit=None):
    parsed = _parse(input_grid)
    output = _grid(output_grid)
    expected, periods = _infer(parsed, node_limit)
    _require(np.array_equal(output, expected), "Incorrect inlay completion")
    board, bg = parsed["board"], parsed["background"]
    fixed = board != bg
    _require(output.shape == board.shape and np.array_equal(output[fixed], board[fixed]),
             "Fixed substrate or surviving inlay changed")
    _require(not np.any(output == bg), "Unfilled groove")
    payload_counts = Counter(color for color, shape in parsed["pieces"] for _ in shape)
    _require(Counter(int(output[q]) for q in parsed["holes"]) == payload_counts,
             "Inventory colors are not conserved")
    values = tuple(int(output[q]) for q in parsed["path"])
    period = _least_period(values)
    clues = _phase_clues(parsed, period)
    unseen = tuple(i for i, c in enumerate(clues) if c == -1)
    _require(unseen, "Repetition alone already determines every missing color")
    if check_ablations:
        unconstrained, _ = _assemblies(parsed, node_limit=node_limit)
        _require(len(unconstrained) == 2,
                 "Geometry alone determines the answer; repetition has no role")
        # Explicit A-only alternative: recolor every occurrence of an unseen
        # phase. It still repeats and preserves all observed cells.
        alternative = output.copy()
        phase = unseen[0]
        new_color = next(c for c in values if c != values[phase])
        for i, q in enumerate(parsed["path"]):
            if i % period == phase:
                alternative[q] = new_color
        _require(not np.array_equal(alternative, output)
                 and np.array_equal(alternative[fixed], board[fixed]),
                 "Missing-phase ablation witness failed")
        _require(Counter(int(alternative[q]) for q in parsed["holes"]) != payload_counts,
                 "A-only alternative accidentally conserves inventory")
    turns = sum((path_next[0] - q[0], path_next[1] - q[1])
                != (q[0] - path_prev[0], q[1] - path_prev[1])
                for path_prev, q, path_next in
                zip(parsed["path"], parsed["path"][1:], parsed["path"][2:]))
    deltas = [(b[0] - a[0], b[1] - a[1])
              for a, b in zip(parsed["path"], parsed["path"][1:])]
    segment_lengths, previous = [], None
    for delta in deltas:
        if delta != previous:
            segment_lengths.append(1)
            previous = delta
        else:
            segment_lengths[-1] += 1
    connectors = tuple(sorted({n for n in segment_lengths[1:-1] if n <= 4}))
    shape_sets = {}
    for color, shape in parsed["pieces"]:
        shape_sets.setdefault(color, set()).add(shape)
    # Equal-count entirely absent colors defeat a counts-only reconstruction.
    absent = set(values) - {int(board[q]) for q in parsed["path"] if board[q] != bg}
    full_counts = Counter(values)
    equal_absent_counts = any(full_counts[a] == full_counts[b]
                              for a in absent for b in absent if a < b)
    counts_shortcut_fails = False
    for a in sorted(absent):
        for b in sorted(absent):
            if a >= b or full_counts[a] != full_counts[b]:
                continue
            swapped = output.copy()
            swapped[output == a] = b
            swapped[output == b] = a
            _require(np.array_equal(swapped[fixed], board[fixed])
                     and Counter(int(swapped[q]) for q in parsed["holes"]) == payload_counts,
                     "Equal-count alternative witness failed")
            if not _inventory_supports_colors(parsed, swapped, node_limit):
                counts_shortcut_fails = True
    return {"period": period, "supported_periods": periods, "length": len(values),
            "turns": turns, "shape": output.shape, "canvas": parsed["input"].shape,
            "connectors": connectors,
            "background": bg, "structural": parsed["structural"],
            "missing_phases": len(unseen), "absent_colors": len(absent),
            "equal_absent_counts": equal_absent_counts,
            "counts_shortcut_fails": counts_shortcut_fails,
            "partial": len(values) % period != 0,
            "piece_count": len(parsed["pieces"]),
            "piece_shapes": tuple(sorted({s for _, s in parsed["pieces"]})),
            "several_shapes_one_color": any(len(s) > 1 for s in shape_sets.values()),
            "bent_piece": any(max(r for r, _ in s) > 0 and max(c for _, c in s) > 0
                              for _, s in parsed["pieces"])}


def _unit(rng, hidden_count, long_chip):
    # Symbols 0/1/(4) survive; symbols 2/(3) disappear completely.
    u = int(rng.integers(2, 5))
    v = int(rng.integers(3, 5)) if long_chip else int(rng.integers(2, 5))
    blocks = [[0] * u, [2] * v, [1] * u]
    if hidden_count == 2:
        for i in range(v):
            # Keep the seam distinct from the leading 0-run; it must not
            # accidentally merge away the equal-shape witness run.
            spacer = 1 if i == v - 1 else int(i % 2)
            blocks.append([3, spacer])
    else:
        blocks.extend(([0] * int(rng.integers(1, 3)), [1] * int(rng.integers(1, 3))))
    if rng.random() < 0.45:
        blocks.append([4] * int(rng.integers(1, 4)))
    # Shuffle actual run blocks, retaining two intact equal-size known runs.
    # The final proposal uses the constructive order if every shuffle merges
    # away a witness run. No finite motif-template bank is used.
    for attempt in range(25):
        order = rng.permutation(len(blocks)) if attempt < 24 else range(len(blocks))
        word = [c for i in order for c in blocks[int(i)]]
        runs = []
        for i in range(len(word)):
            if word[i] == word[i - 1]:
                continue
            n = 1
            while n < len(word) and word[(i + n) % len(word)] == word[i]:
                n += 1
            runs.append((word[i], n))
        if (0, u) in runs and (1, u) in runs:
            return np.array(word, dtype=np.int8), u
    raise AssertionError("Constructive word order lost its known runs")


def _folded_path(rng, minimum, irregular, stretched):
    # Parallel legs can have independently varying ends and row spacings.
    # Connecting consecutive legs vertically produces no accidental contacts.
    width = int(rng.integers(15, 23))
    legs = int(rng.integers(4, 7))
    gap = [3 if stretched else 2 for _ in range(legs - 1)]
    height = 3 + sum(gap)
    row, col = 1, int(rng.integers(1, 4))
    path = [(row, col)]
    for leg in range(legs):
        end = (int(rng.integers(width - 4, width - 1)) if irregular else width - 2
               ) if leg % 2 == 0 else (int(rng.integers(1, 4)) if irregular else 1)
        step = 1 if end > col else -1
        while col != end:
            col += step
            path.append((row, col))
        if leg + 1 < legs:
            for _ in range(gap[leg]):
                row += 1
                path.append((row, col))
    if len(path) < minimum:
        # Constructive wide six-leg fallback, still parameterized by the unit.
        distance = 3 if stretched else 2
        width, height = 23, 3 + 5 * distance
        path = []
        for leg in range(6):
            r = 1 + distance * leg
            columns = range(1, 22) if leg % 2 == 0 else range(21, 0, -1)
            path.extend((r, c) for c in columns)
            if leg < 5:
                path.extend((r + j, path[-1][1]) for j in range(1, distance))
    _walk(path)
    return (height, width), path


def _runs(path, colors):
    result, first = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or colors[i] != colors[first]:
            result.append((int(colors[first]), tuple(path[first:i]), first, i))
            first = i
    return result


def _remove_chips(rng, board, path, colors, hidden_count, equal_run):
    hidden = {2, 3} if hidden_count == 2 else {2}
    runs = _runs(path, colors)
    pieces, removed = [], set()
    for color, cells, _, _ in runs:
        if color not in hidden:
            continue
        # Some bars are divided into real pieces; at least one long chip remains.
        if len(cells) >= 3 and rng.random() < 0.2:
            cut = int(rng.integers(1, len(cells)))
            chunks = (cells[:cut], cells[cut:])
        else:
            chunks = (cells,)
        pieces.extend((color, _normalize(chunk)) for chunk in chunks)
        removed.update(cells)
    # Deliberate A/B necessity witness: two rigid chips with exactly the same
    # orientation and shape, but different colors/period phases.
    candidates = {color: {} for color in (0, 1)}
    for color, cells, _, _ in runs:
        if color in candidates and len(cells) == equal_run:
            shape = _normalize(cells)
            candidates[color].setdefault(shape, []).append(cells)
    common = sorted(set(candidates[0]) & set(candidates[1]))
    _require(common, "No same-shape two-color witness")
    shape = common[int(rng.integers(len(common)))]
    for color in (0, 1):
        cells = candidates[color][shape][int(rng.integers(len(candidates[color][shape])))]
        pieces.append((color, shape))
        removed.update(cells)
    # Additional removals preserve at least two observations of every initially
    # observed phase. They vary fragmentation and adjacency without hidden rules.
    p = _least_period(tuple(map(int, colors)))
    visible = Counter(i % p for i, q in enumerate(path) if q not in removed)
    choices = [run for run in runs if run[0] not in hidden
               and not any(q in removed for q in run[1])]
    rng.shuffle(choices)
    for color, cells, first, last in choices[:int(rng.integers(0, 4))]:
        if all(visible[i % p] >= 3 for i in range(first, last)):
            pieces.append((color, _normalize(cells)))
            removed.update(cells)
            visible.subtract(i % p for i in range(first, last))
    for q in removed:
        board[q] = -1
    return pieces


def _scatter(rng, board, pieces, bg):
    h, w = board.shape
    for attempt in range(10):
        ch = 30 if attempt >= 7 else int(rng.integers(max(h + 6, 22), 31))
        cw = 30 if attempt >= 7 else int(rng.integers(max(w + 3, 25), 31))
        top, left = int(rng.integers(ch - h + 1)), int(rng.integers(cw - w + 1))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        canvas[top:top + h, left:left + w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0, top - 1):min(ch, top + h + 1),
                max(0, left - 1):min(cw, left + w + 1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), float(rng.random())))
        for i in order:
            color, shape = pieces[i]
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            choices = [(r, c) for r in range(ch - sh + 1) for c in range(cw - sw + 1)
                       if all(not blocked[r + dr, c + dc] for dr, dc in shape)]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            for dr, dc in shape:
                rr, cc = r + dr, c + dc
                canvas[rr, cc] = color
                for ar, ac in ((rr, cc),) + tuple((rr + a, cc + b) for a, b in _DIRECTIONS):
                    if 0 <= ar < ch and 0 <= ac < cw:
                        blocked[ar, ac] = True
        else:
            if np.count_nonzero(canvas == bg) > canvas.size // 2:
                return canvas
    raise ValueError("Could not scatter inventory")


def _generate_pair(rng, palette, hidden_count, partial, irregular, rotation, long_chip,
                   stretched):
    for attempt in range(90):
        unit, equal_run = _unit(rng, hidden_count, long_chip)
        p = len(unit)
        shape, path = _folded_path(rng, 4 * p + 1, irregular, stretched)
        # Length and period vary independently; complete or partial endpoints
        # keep four to six full cycles rather than a memorized rectangular width.
        cycles = int(rng.integers(4, min(6, len(path) // p) + 1))
        length = cycles * p
        if partial and length < len(path):
            length += int(rng.integers(1, min(p - 1, len(path) - length) + 1))
        path = path[:length]
        bends = [i for i in range(1, length - 1)
                 if (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                 != (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])]
        if long_chip and bends:
            # Make a genuinely bent three-cell missing chip, not a rotated bar.
            run_center = next(i for i in range(1, p - 1)
                              if np.all(unit[i - 1:i + 2] == 2))
            offset = (run_center - bends[int(rng.integers(len(bends)))]) % p
        else:
            offset = int(rng.integers(p))
        symbols = unit[(np.arange(length) + offset) % p]
        bg, structural = map(int, palette[:2])
        color_map = list(map(int, palette[2:7]))
        output = np.full(shape, structural, dtype=np.int8)
        for q, symbol in zip(path, symbols):
            output[q] = color_map[int(symbol)]
        board = output.copy()
        try:
            symbolic_pieces = _remove_chips(rng, board, path, symbols, hidden_count, equal_run)
            pieces = [(color_map[color], cells) for color, cells in symbolic_pieces]
            board[board == -1] = bg
            inp = _scatter(rng, board, pieces, bg)
            pair = {"input": np.rot90(inp, rotation).copy(),
                    "output": np.rot90(output, rotation).copy()}
            d = _analyze(pair["input"], pair["output"], node_limit=18000)
            if (hidden_count == 2 and not partial and not d["equal_absent_counts"]
                    or long_chip and not d["bent_piece"]
                    or partial and not d["partial"]):
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("Bounded inlay construction exhausted")


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    exchange = bool(rng.random() < 0.65)
    rotation = int(rng.integers(4))
    mix_orientation = bool(rng.random() < 0.8)
    irregular_style = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    # The first two demonstrations expose different hidden-color counts and
    # complete/partial cycles. Their order is sampled, not an answer convention.
    witness_order = list(map(int, rng.permutation(2)))
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                witness = witness_order[index]
                hidden_count, partial, bent = (2, False, True) if witness == 0 else (1, True, True)
            else:
                hidden_count = int(rng.integers(1, 3))
                partial = bool(rng.random() < (0.85 if split == "test" else 0.4))
                bent = bool(rng.random() < 0.75)
            irregular = True if split == "test" else (irregular_style if index >= 2 else index == 1)
            pair_rotation = int(rng.integers(4)) if mix_orientation else rotation
            prng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair_palette = prng.permutation(10) if exchange else palette
            for _ in range(4):
                pair = _generate_pair(prng, pair_palette, hidden_count, partial,
                                      irregular, pair_rotation, bent, split == "test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, descriptions = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            descriptions[split].append(_analyze(inp, out))
    train = descriptions["train"]
    _require(any(d["equal_absent_counts"] and d["counts_shortcut_fails"] for d in train),
             "Training lacks the shape-versus-count witness")
    _require(any(d["partial"] for d in train) and any(not d["partial"] for d in train),
             "Training must expose complete and partial cycles")
    _require(any(d["bent_piece"] for d in train), "Training must expose corner-spanning chips")
    _require(all(d["connectors"] == (2,) for d in train)
             and all(d["connectors"] == (3,) for d in descriptions["test"]),
             "Expected the longer-connector test shift")
    return descriptions


def self_test():
    """Bounded 42-seed semantic, diversity, corruption and API checks."""
    metrics = {k: set() for k in ("puzzles", "shapes", "canvases", "periods", "lengths",
                                "turns", "piece_counts", "piece_shapes", "roles")}
    flags = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        descriptions = validate_generated_puzzle(puzzle)
        metrics["puzzles"].add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for d in descriptions[split]:
                for k, dk in (("shapes", "shape"), ("canvases", "canvas"), ("periods", "period"),
                              ("lengths", "length"), ("turns", "turns"), ("piece_counts", "piece_count")):
                    metrics[k].add(d[dk])
                metrics["piece_shapes"].update(d["piece_shapes"])
                metrics["roles"].add((d["background"], d["structural"]))
                for flag in ("partial", "bent_piece", "several_shapes_one_color",
                             "equal_absent_counts", "counts_shortcut_fails"):
                    flags[flag] += bool(d[flag])
    _require(len(metrics["puzzles"]) == 42 and len(metrics["shapes"]) >= 25
             and len(metrics["canvases"]) >= 25, "Insufficient scene variation")
    _require(len(metrics["periods"]) >= 6 and len(metrics["lengths"]) >= 30
             and len(metrics["turns"]) >= 4, "Insufficient periodic/track variation")
    _require(len(metrics["piece_counts"]) >= 10 and len(metrics["piece_shapes"]) >= 12
             and len(metrics["roles"]) >= 15, "Insufficient inventory/palette variation")
    _require(all(flags[k] >= 30 for k in flags), "Missing structural combinations")
    first, second = generate_puzzle(9182, 2, 1), generate_puzzle(9182, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s])
                 for k in ("input", "output")), "Seed is not reproducible")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            _check_counts(a, b)
    # The host generates every legal combination. Keep this test bounded by
    # generating boundary combinations rather than repeating the entire suite.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(300 + a * 4 + b, a, b)
        validate_generated_puzzle(p)
        _require(len(p["train"]) == a and len(p["test"]) == b, "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Label corruption and inventory loss must both be rejected.
    pair = first["train"][0]
    parsed = _parse(pair["input"])
    bad_output = pair["output"].copy()
    bad_output[parsed["holes"][0]] = parsed["background"]
    try:
        _analyze(pair["input"], bad_output)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an unfilled groove")
    missing = pair["input"].copy()
    missing[parsed["exterior"][0][0]] = parsed["background"]
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted a lost inventory cell")
    # Reversal of the chosen endpoint, whole-scene rotations, and literal
    # palette changes must not change inference conventions.
    palette_map = np.random.default_rng(443).permutation(10)
    for rotation in range(4):
        query = palette_map[np.rot90(pair["input"], rotation)].astype(np.int8)
        expected = palette_map[np.rot90(pair["output"], rotation)].astype(np.int8)
        _require(np.array_equal(solve(query), expected), "Geometry/palette covariance failed")
    # Identical-piece exchanges are harmless; different-color swaps are not.
    fixture = {"board": np.array([[0, 8, 0]], dtype=np.int8),
               "holes": ((0, 0), (0, 2)), "path": ((0, 0), (0, 1), (0, 2)),
               "pieces": ((1, ((0, 0),)), (1, ((0, 0),)))}
    _require(len(_assemblies(fixture)[0]) == 1, "Identical swaps counted as two answers")
    fixture["pieces"] = ((1, ((0, 0),)), (2, ((0, 0),)))
    _require(len(_assemblies(fixture)[0]) == 2, "Colored swap ambiguity was missed")
    fixture = {"board": np.array([[0, 0], [0, 8]], dtype=np.int8),
               "holes": ((0, 0), (0, 1), (1, 0)),
               "path": ((0, 1), (0, 0), (1, 0)),
               "pieces": ((1, ((0, 1), (1, 0), (1, 1))),)}
    _require(not _assemblies(fixture)[0], "A chip was illegally rotated/reflected")
    return {"seeds": 42, "pairs": 210,
            "distinct": {k: len(v) for k, v in metrics.items()}, "flags": dict(flags)}
