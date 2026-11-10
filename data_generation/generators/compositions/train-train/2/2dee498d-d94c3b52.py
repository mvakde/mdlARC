"""ARC family: stable glyph filtering on query-bounded crossing threads.

Only the input determines the answer. Generation constructs panel-level labels
by a prefix predicate; solve collects first-occurrence nominations separately.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return array.astype(np.int8, copy=False)


def _mask(code, size):
    return np.array([(int(code) >> i) & 1 for i in range(size * size)],
                    dtype=bool).reshape(size, size)


def _code(mask):
    return sum(int(bit) << i for i, bit in enumerate(mask.flat))


def _panel(row, col, size):
    step = size + 1
    return (slice(1 + row * step, 1 + row * step + size),
            slice(1 + col * step, 1 + col * step + size))


def _analyze(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if (not np.all(grid[[0, -1], :] == background)
            or not np.all(grid[:, [0, -1]] == background)
            or len(np.unique(grid)) != 3):
        raise ValueError("input needs a background border and exactly three colors")
    candidates = []
    for size in (2, 3, 4):
        step = size + 1
        if (grid.shape[0] - 1) % step or (grid.shape[1] - 1) % step:
            continue
        rows, cols = (grid.shape[0] - 1) // step, (grid.shape[1] - 1) // step
        if min(rows, cols) < 2:
            continue
        if not (np.all(grid[::step, :] == background)
                and np.all(grid[:, ::step] == background)):
            continue
        codes = np.empty((rows, cols), dtype=np.int64)
        colors = np.empty((rows, cols), dtype=np.int8)
        counts = {}
        valid = True
        for row in range(rows):
            for col in range(cols):
                patch = grid[_panel(row, col, size)]
                occupied = patch != background
                ink = np.unique(patch[occupied])
                if len(ink) != 1 or not np.any(~occupied):
                    valid = False
                    break
                color = int(ink[0])
                colors[row, col] = color
                codes[row, col] = _code(occupied)
                counts[color] = counts.get(color, 0) + 1
            if not valid:
                break
        if not valid or len(counts) != 2:
            continue
        rare = [color for color, count in counts.items() if count == 1]
        if len(rare) != 1:
            continue
        query = rare[0]
        query_position = tuple(map(int, np.argwhere(colors == query)[0]))
        ordinary = next(color for color in counts if color != query)
        candidates.append(dict(grid=grid, background=background, query=query,
                               ordinary=ordinary, query_position=query_position,
                               target=int(codes[query_position]), codes=codes,
                               colors=colors, rows=rows, cols=cols, size=size))
    if len(candidates) != 1:
        raise ValueError("panel geometry and query must have one unambiguous interpretation")
    return candidates[0]


def _thread_info(codes, target):
    """Read the original bitmaps; collect coverage and independent nominations."""
    matches = codes == target
    horizontal = np.zeros_like(matches)
    vertical = np.zeros_like(matches)
    first_h = np.zeros_like(matches)
    first_v = np.zeros_like(matches)
    for row in range(codes.shape[0]):
        hits = np.flatnonzero(matches[row])
        if len(hits) >= 2:
            seen = set()
            for col in range(int(hits[0]) + 1, int(hits[-1])):
                if matches[row, col]:
                    continue
                horizontal[row, col] = True
                key = int(codes[row, col])
                if key not in seen:
                    first_h[row, col] = True
                    seen.add(key)
    for col in range(codes.shape[1]):
        hits = np.flatnonzero(matches[:, col])
        if len(hits) >= 2:
            seen = set()
            for row in range(int(hits[0]) + 1, int(hits[-1])):
                if matches[row, col]:
                    continue
                vertical[row, col] = True
                key = int(codes[row, col])
                if key not in seen:
                    first_v[row, col] = True
                    seen.add(key)
    covered = horizontal | vertical
    first = first_h | first_v
    return dict(matches=matches, horizontal=horizontal, vertical=vertical,
                first_h=first_h, first_v=first_v, covered=covered,
                deleted=covered & ~first, kept=~covered | first)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family input without generation data or puzzle context."""
    a = _analyze(input_grid)
    info = _thread_info(a["codes"], a["target"])
    result = a["grid"].copy()
    for row in range(a["rows"]):
        for col in range(a["cols"]):
            if info["matches"][row, col]:
                patch = result[_panel(row, col, a["size"])]
                patch[patch != a["background"]] = a["query"]
            elif info["deleted"][row, col]:
                result[_panel(row, col, a["size"])] = a["background"]
    return result


def _prefix_keep(codes, target):
    """Direct per-position definition, separate from the nomination algorithm."""
    rows, cols = codes.shape
    matches = codes == target
    keep = np.ones_like(matches)
    for row in range(rows):
        row_hits = [c for c in range(cols) if matches[row, c]]
        for col in range(cols):
            if matches[row, col]:
                continue
            col_hits = [r for r in range(rows) if matches[r, col]]
            on_h = len(row_hits) >= 2 and row_hits[0] < col < row_hits[-1]
            on_v = len(col_hits) >= 2 and col_hits[0] < row < col_hits[-1]
            first_h = on_h and not any(
                codes[row, c] == codes[row, col]
                for c in range(row_hits[0] + 1, col) if not matches[row, c])
            first_v = on_v and not any(
                codes[r, col] == codes[row, col]
                for r in range(col_hits[0] + 1, row) if not matches[r, col])
            keep[row, col] = not (on_h or on_v) or first_h or first_v
    return keep


def _render(codes, size, palette, query_position, keep=None, output=False):
    background, ordinary, query = palette
    rows, cols = codes.shape
    result = np.full((rows * (size + 1) + 1, cols * (size + 1) + 1),
                     background, dtype=np.int8)
    target = int(codes[query_position])
    for row in range(rows):
        for col in range(cols):
            if keep is not None and not keep[row, col]:
                continue
            color = query if ((output and codes[row, col] == target)
                              or (row, col) == query_position) else ordinary
            patch = result[_panel(row, col, size)]
            patch[_mask(codes[row, col], size)] = color
    return result


def _global_keep(codes, target):
    """Ablation: forget the terminal bounds but keep the same union semantics."""
    matches = codes == target
    first = np.zeros_like(matches)
    for row in range(codes.shape[0]):
        seen = set()
        for col in range(codes.shape[1]):
            key = int(codes[row, col])
            if key == target:
                continue
            if key not in seen:
                first[row, col] = True
                seen.add(key)
    for col in range(codes.shape[1]):
        seen = set()
        for row in range(codes.shape[0]):
            key = int(codes[row, col])
            if key == target:
                continue
            if key not in seen:
                first[row, col] = True
                seen.add(key)
    return first | matches


def _across_terminal_evidence(codes, target, deleted):
    for board, gone in ((codes, deleted), (codes.T, deleted.T)):
        for row in range(board.shape[0]):
            hits = np.flatnonzero(board[row] == target)
            if len(hits) < 3:
                continue
            for col in range(int(hits[0]) + 1, int(hits[-1])):
                if not gone[row, col]:
                    continue
                for earlier in range(int(hits[0]) + 1, col):
                    if (board[row, earlier] == board[row, col]
                            and np.any((hits > earlier) & (hits < col))):
                        return True
    return False


def _metrics(a):
    info = _thread_info(a["codes"], a["target"])
    cross = info["horizontal"] & info["vertical"]
    one_first = cross & (info["first_h"] ^ info["first_v"])
    double_repeat = cross & ~info["first_h"] & ~info["first_v"]
    outside = ~info["covered"] & ~info["matches"]
    protected = outside & ~_global_keep(a["codes"], a["target"])
    values, counts = np.unique(a["codes"], return_counts=True)
    frequencies = dict(zip(map(int, values), map(int, counts)))
    return dict(info=info, target=a["target"], cross_one=int(np.count_nonzero(one_first)),
                cross_double=int(np.count_nonzero(double_repeat)),
                outside_protected=int(np.count_nonzero(protected)),
                across_terminal=_across_terminal_evidence(a["codes"], a["target"],
                                                          info["deleted"]),
                frequencies=frequencies)


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        row, col = stack.pop()
        for neighbor in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == cells


def _template(rng, size):
    kind = int(rng.integers(6))
    if kind == 0 and size >= 3:
        mask = np.ones((size, size), dtype=bool)
        mask[1:-1, 1:-1] = False
        return _code(mask)
    for _ in range(80):
        weight = int(rng.integers(1, size * size))
        if kind == 1:
            weight = int(rng.integers(1, min(4, size * size)))
        flat = np.zeros(size * size, dtype=bool)
        flat[rng.choice(size * size, size=weight, replace=False)] = True
        mask = flat.reshape(size, size)
        if kind == 2:
            mask |= np.fliplr(mask)
        if np.all(mask):
            continue
        if kind == 3 and not _connected(mask):
            continue
        if kind == 4 and _connected(mask):
            continue
        return _code(mask)
    # Nonempty, nonfull, disconnected at every supported panel size.
    return 1 | (1 << (size * size - 1))


def _vocabulary(rng, target, size, count):
    maximum = (1 << (size * size)) - 1
    template = _mask(target, size)
    transforms = [_code(x) for x in (np.rot90(template), np.rot90(template, 2),
                                    np.rot90(template, 3), np.fliplr(template),
                                    np.flipud(template), template.T)]
    near = [target ^ (1 << i) for i in range(size * size)]
    occupied = [i for i in range(size * size) if (target >> i) & 1]
    empty = [i for i in range(size * size) if not (target >> i) & 1]
    same_weight = [target ^ (1 << i) ^ (1 << j) for i in occupied for j in empty]
    result = []
    for group in (transforms, near, same_weight):
        choices = sorted(set(group) - {0, maximum, target} - set(result))
        if choices and len(result) < count:
            result.append(int(rng.choice(choices)))
    while len(result) < count:
        code = int(rng.integers(1, maximum))
        if code != target and code not in result:
            result.append(code)
    rng.shuffle(result)
    return result


def _witness(rng, kind, rows, cols):
    """Return query positions and forced *distractor indices*, not pixel masks."""
    if kind == "one":
        matches = {(2, 0), (2, 4), (0, 2), (4, 2)}
        forced = {(2, 1): 0, (2, 2): 0, (2, 3): 1,
                  (1, 2): 2, (3, 2): 2}
        base_rows, base_cols = 5, 5
    else:
        matches = {(2, 0), (2, 2), (2, 5), (0, 3), (4, 3)}
        forced = {(2, 1): 0, (2, 3): 0, (2, 4): 1,
                  (1, 3): 0, (3, 3): 2}
        base_rows, base_cols = 5, 6
    transpose = bool(rng.integers(2))
    if transpose:
        matches = {(c, r) for r, c in matches}
        forced = {(c, r): value for (r, c), value in forced.items()}
        base_rows, base_cols = base_cols, base_rows
    if base_rows > rows or base_cols > cols:
        rows, cols = cols, rows
    dr = int(rng.integers(rows - base_rows + 1))
    dc = int(rng.integers(cols - base_cols + 1))
    return (rows, cols, {(r + dr, c + dc) for r, c in matches},
            {(r + dr, c + dc): value for (r, c), value in forced.items()})


def _anchors(rng, rows, cols, mode):
    def ends(length):
        choices = [(a, b) for a in range(length) for b in range(a + 3, length)]
        return choices[int(rng.integers(len(choices)))]

    if mode == "row":
        row = int(rng.integers(rows))
        left, right = ends(cols)
        result = {(row, left), (row, right)}
    elif mode == "column":
        col = int(rng.integers(cols))
        top, bottom = ends(rows)
        result = {(top, col), (bottom, col)}
    elif mode == "cross":
        row = int(rng.integers(1, rows - 1))
        col = int(rng.integers(1, cols - 1))
        result = {(row, 0), (row, cols - 1), (0, col), (rows - 1, col)}
    elif mode == "rectangle":
        top, bottom = ends(rows)
        left, right = ends(cols)
        result = {(top, left), (top, right), (bottom, left), (bottom, right)}
    elif mode == "ell":
        top, bottom = ends(rows)
        left, right = ends(cols)
        result = {(top, left), (top, right), (bottom, left)}
    elif mode == "two_runs":
        row1, row2 = map(int, rng.choice(rows, size=2, replace=False))
        left, right = ends(cols)
        other_left, other_right = ends(cols)
        result = {(row1, left), (row1, right), (row2, other_left), (row2, other_right)}
    elif mode == "ladder":
        left, right = ends(cols)
        active = rng.choice(rows, size=min(3, rows), replace=False)
        result = {(int(row), col) for row in active for col in (left, right)}
    elif mode == "weave":
        # Several row memories and column memories share ordinary glyphs.
        # Endpoints themselves also obey the outermost-span rule.
        horizontal_rows = rng.choice(np.arange(1, rows - 1),
                                     size=int(rng.integers(2, min(4, rows - 2) + 1)),
                                     replace=False)
        vertical_cols = rng.choice(np.arange(1, cols - 1),
                                   size=int(rng.integers(2, min(4, cols - 2) + 1)),
                                   replace=False)
        result = {(int(row), col) for row in horizontal_rows for col in (0, cols - 1)}
        result.update((row, int(col)) for col in vertical_cols for row in (0, rows - 1))
    else:
        count = int(rng.integers(3, min(10, rows * cols - 6) + 1))
        indices = rng.choice(rows * cols, size=count, replace=False)
        result = {(int(i) // cols, int(i) % cols) for i in indices}
    # Isolated copies still recolor, without forcing every network to have one.
    if mode in ("row", "column", "ell") and rng.random() < 0.4:
        candidates = [(r, c) for r in range(rows) for c in range(cols)
                      if all(r != x and c != y for x, y in result)]
        if candidates:
            result.add(candidates[int(rng.integers(len(candidates)))])
    return result


_MODES = ("row", "column", "cross", "rectangle", "ell", "two_runs", "ladder", "weave", "random")


def _assign_codes(rng, rows, cols, target, vocabulary, matches, forced):
    codes = np.full((rows, cols), target, dtype=np.int64)
    for position, index in forced.items():
        codes[position] = vocabulary[index]
    free = [(r, c) for r in range(rows) for c in range(cols)
            if (r, c) not in matches and (r, c) not in forced]
    rng.shuffle(free)
    needed = []
    for code in vocabulary:
        used = sum(codes[pos] == code for pos in forced)
        needed.extend([code] * max(0, 2 - int(used)))
    if len(needed) > len(free):
        return None
    weights = rng.dirichlet(np.full(len(vocabulary), 0.7))
    extra = rng.choice(vocabulary, size=len(free) - len(needed), p=weights).tolist()
    assignments = needed + extra
    rng.shuffle(assignments)
    for position, code in zip(free, assignments):
        codes[position] = code
    return codes


def _make_pair(rng, size, rows, cols, kind, seen):
    for _ in range(240):
        local_rows, local_cols = rows, cols
        if kind in ("one", "both"):
            local_rows, local_cols, matches, forced = _witness(
                rng, kind, local_rows, local_cols)
        else:
            mode = str(rng.choice(_MODES))
            matches = _anchors(rng, local_rows, local_cols, mode)
            forced = {}
        target = _template(rng, size)
        maximum = min(5 if kind in ("one", "both") else 7,
                      (local_rows * local_cols - len(matches)) // 2)
        if maximum < 3:
            continue
        vocabulary = _vocabulary(rng, target, size, int(rng.integers(3, maximum + 1)))
        codes = _assign_codes(rng, local_rows, local_cols, target, vocabulary, matches, forced)
        if codes is None:
            continue
        palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
        query_position = sorted(matches)[int(rng.integers(len(matches)))]
        inp = _render(codes, size, palette, query_position)
        key = _grid_key(inp)
        if key in seen:
            continue
        try:
            a = _analyze(inp)
        except ValueError:
            continue
        m = _metrics(a)
        info = m["info"]
        if not np.any(info["deleted"]) or not np.any(info["covered"] & info["kept"]):
            continue
        if kind == "one":
            if not (m["cross_one"] and m["outside_protected"]):
                continue
            if max(v for k, v in m["frequencies"].items() if k != target) <= len(matches):
                continue
        if kind == "both" and not (m["cross_double"] and m["across_terminal"]):
            continue
        # Labels use a direct earlier-equal predicate, not solve's seen sets.
        keep = _prefix_keep(codes, target)
        out = _render(codes, size, palette, query_position, keep=keep, output=True)
        seen.add(key)
        return {"input": inp, "output": out}
    raise RuntimeError(f"bounded sampling exhausted for {kind}, size {size}, {rows}x{cols}")


def _grid_key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4
            or num_train <= num_test):
        raise ValueError("need 2..10 training and 1..4 test examples, with train > test")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = int(num_train), int(num_test)
    seen = set()
    first_size = int(rng.integers(2, 5))
    capacity = min(6, 29 // (first_size + 1))
    first_rows = int(rng.integers(5, capacity + 1))
    first_cols = int(rng.integers(5, capacity + 1))
    train = [_make_pair(rng, first_size, first_rows, first_cols, "one", seen)]
    second_size = int(rng.integers(2, 4))
    train.append(_make_pair(rng, second_size, int(rng.integers(5, 7)), 6, "both", seen))
    while len(train) < num_train:
        size = int(rng.integers(2, 5))
        cap = min(6, 29 // (size + 1))
        rows = int(rng.integers(4, cap + 1))
        cols = int(rng.integers(4, cap + 1))
        train.append(_make_pair(rng, size, rows, cols, "random", seen))
    demonstrated = sorted({_analyze(p["input"])["size"] for p in train} & {2, 3})
    test = []
    for _ in range(num_test):
        size = int(rng.choice(demonstrated))
        cap = 29 // (size + 1)
        rows = int(rng.integers(7, cap + 1))
        cols = int(rng.integers(4, cap + 1))
        test.append(_make_pair(rng, size, rows, cols, "random", seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, visible semantics, family witnesses, and test extrapolation."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal split counts")
    seen, measurements, trained_sizes = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs need exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            key = _grid_key(inp)
            _require(key not in seen, "repeated input")
            seen.add(key)
            a = _analyze(inp)
            m = _metrics(a)
            measurements[split].append(m)
            info = m["info"]
            frequencies = m["frequencies"]
            _require(4 <= len(frequencies) <= 8, "expected a repeated glyph vocabulary")
            _require(all(v >= 2 for k, v in frequencies.items() if k != a["target"]),
                     "every sampled distractor type should repeat")
            _require(frequencies[a["target"]] >= 2, "need more than one terminal")
            _require(np.any(info["deleted"]), "pair does not expose duplicate suppression")
            _require(np.any(info["covered"] & info["kept"]), "need a surviving thread glyph")
            _require(np.all(out[inp == a["background"]] == a["background"]),
                     "an empty input cell was filled")
            keep = _prefix_keep(a["codes"], a["target"])
            expected = _render(a["codes"], a["size"],
                               (a["background"], a["ordinary"], a["query"]),
                               a["query_position"], keep=keep, output=True)
            _require(np.array_equal(out, expected), "prefix-based labels disagree")
            _require(np.array_equal(out, solve(inp)), "input-only solver disagrees")
            for row in range(a["rows"]):
                for col in range(a["cols"]):
                    patch = out[_panel(row, col, a["size"])]
                    if info["matches"][row, col]:
                        _require(np.all(patch[_mask(a["target"], a["size"])] == a["query"]),
                                 "terminal precedence failed")
                    elif info["deleted"][row, col]:
                        _require(np.all(patch == a["background"]), "redundant glyph survived")
                    else:
                        _require(np.array_equal(patch, inp[_panel(row, col, a["size"])]),
                                 "a preserved glyph changed")
            if split == "train":
                _require(a["rows"] <= 6 and a["cols"] <= 6, "training lattice too large")
                trained_sizes.add(a["size"])
            else:
                _require(a["rows"] >= 7, "test lacks spatial extrapolation")
                _require(a["size"] in trained_sizes, "test uses an undemonstrated panel size")
    first, second = measurements["train"][:2]
    _require(first["cross_one"] > 0, "need a crossing new on exactly one thread")
    _require(second["cross_double"] > 0, "need a crossing repeated on both threads")
    _require(second["across_terminal"], "need memory across an interior terminal")
    _require(any(m["outside_protected"] for m in measurements["train"]),
             "training does not defeat unbounded filtering")
    _require(any(max(v for k, v in m["frequencies"].items() if k != m["target"])
                 > m["frequencies"][m["target"]]
                 for m in measurements["train"]),
             "training does not defeat the most-frequent-key shortcut")


def _normalized(grid):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in np.asarray(grid).flat)


def _small_cases():
    """Valid low-complexity members outside the generator's informative schedule."""
    palette, size, target = (0, 1, 2), 2, 1
    boards = [np.array([[1, 2, 4], [2, 1, 2], [4, 2, 1]]),
              np.array([[1, 1, 2], [2, 2, 4]]),
              np.array([[1, 2, 4, 2, 1], [4, 2, 4, 2, 4]]),
              np.array([[1, 2, 1, 2, 4, 1], [4, 2, 4, 2, 4, 2]])]
    for codes in boards:
        query_position = tuple(map(int, np.argwhere(codes == target)[0]))
        inp = _render(codes, size, palette, query_position)
        keep = _prefix_keep(codes, target)
        expected = _render(codes, size, palette, query_position, keep=keep, output=True)
        _require(np.array_equal(solve(inp), expected), "small-network solve failed")
    # No threads and empty intervals: only template copies change ink.
    for codes in boards[:2]:
        query_position = tuple(map(int, np.argwhere(codes == target)[0]))
        inp = _render(codes, size, palette, query_position)
        expected = _render(codes, size, palette, query_position, output=True)
        _require(np.array_equal(solve(inp), expected), "empty-thread behavior failed")
    # Keep first A; the later separated A disappears, with no compaction.
    predicted = solve(_render(boards[2], size, palette, (0, 0)))
    _require(np.any(predicted[_panel(0, 1, size)] != 0)
             and np.all(predicted[_panel(0, 3, size)] == 0), "stable order failed")
    predicted = solve(_render(boards[3], size, palette, (0, 0)))
    _require(np.all(predicted[_panel(0, 3, size)] == 0), "interior terminal reset memory")
    # Two equal-weight orientations remain distinct; the later exact copy dies.
    lookalikes = np.array([[1, 7, 11, 7, 1], [4, 2, 4, 2, 4]])
    original = _render(lookalikes, size, palette, (0, 0))
    snapshot = original.copy()
    predicted = solve(original)
    _require(np.array_equal(original, snapshot), "solve mutated its input")
    _require(np.any(predicted[_panel(0, 2, size)] != 0)
             and np.all(predicted[_panel(0, 3, size)] == 0),
             "equal-weight orientations were conflated")


def self_test():
    """Bounded 42-seed study plus every legal count and targeted edge cases."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "duplicate seeded puzzles")
    _require(_puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17)),
             "default generation is not deterministic")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(314159 + train * 10 + test, train, test)
            _require((len(p["train"]), len(p["test"])) == (train, test), "counts ignored")
    explicit = generate_puzzle(12345, 10, 4)
    _require(_puzzle_key(explicit) == _puzzle_key(generate_puzzle(12345, 10, 4)),
             "explicit generation is not deterministic")
    illegal = [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
               (True, 1), (4, np.bool_(True)), (3.0, 1), (4, 1.5), (None, 1), (4, None)]
    for train, test in illegal:
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {train}/{test}")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an illegal seed")
    shapes, sizes, structures, vocab_sizes, networks, roles = set(), set(), set(), set(), set(), set()
    templates, deletion_counts, matches_counts = set(), set(), set()
    connected, disconnected, isolated, ordinary_at_cross = 0, 0, 0, 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                m = _metrics(a)
                info = m["info"]
                shapes.add(pair["input"].shape)
                sizes.add(a["size"])
                structures.add((pair["input"].shape, _normalized(pair["input"])))
                vocab_sizes.add(len(m["frequencies"]))
                templates.add((a["size"], a["target"]))
                roles.add((a["background"], a["ordinary"], a["query"]))
                deletion_counts.add(int(np.count_nonzero(info["deleted"])))
                matches_counts.add(int(np.count_nonzero(info["matches"])))
                network = (bool(np.any(info["horizontal"])), bool(np.any(info["vertical"])),
                           bool(m["cross_one"]), bool(m["cross_double"]))
                networks.add(network)
                is_connected = _connected(_mask(a["target"], a["size"]))
                connected += is_connected
                disconnected += not is_connected
                ordinary_at_cross += m["cross_one"]
                positions = list(map(tuple, np.argwhere(info["matches"])))
                isolated += any(not any(other != pos and (pos[0] == other[0] or pos[1] == other[1])
                                        for other in positions) for pos in positions)
                # Both parent ablations are evaluated on actual visible grids.
                only_b = _render(a["codes"], a["size"],
                                 (a["background"], a["ordinary"], a["query"]),
                                 a["query_position"], output=True)
                _require(not np.array_equal(only_b, pair["output"]), "A ablation made no difference")
    _require(sizes == {2, 3, 4}, "panel-size variation missing")
    _require(len(shapes) >= 15 and len(structures) >= 170, "insufficient geometry variation")
    _require(len(templates) >= 45 and len(roles) >= 100, "insufficient masks or role variation")
    _require({4, 5, 6, 7} <= vocab_sizes, "vocabulary variation missing")
    _require(len(networks) >= 6 and len(deletion_counts) >= 7 and len(matches_counts) >= 5,
             "network or suppression variation missing")
    _require(connected > 20 and disconnected > 20 and isolated > 0 and ordinary_at_cross >= 42,
             "morphology or crossing coverage missing")
    _small_cases()
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in ("train", "test")),
                input_shapes=len(shapes), normalized_structures=len(structures),
                templates=len(templates), color_role_assignments=len(roles),
                vocabulary_sizes=sorted(vocab_sizes), network_signatures=len(networks),
                deletion_counts=sorted(deletion_counts), match_counts=sorted(matches_counts),
                connected_queries=connected, disconnected_queries=disconnected,
                isolated_terminal_pairs=isolated)
