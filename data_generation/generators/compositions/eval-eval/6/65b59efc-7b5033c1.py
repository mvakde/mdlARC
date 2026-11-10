"""Stamp procession: directed, keyed, transparent printing along a simple path.

Only Python's standard library and NumPy are used. Nothing is read from disk.
The solver recovers every convention and command from the input itself.
"""
from __future__ import annotations

import copy
import json

import numpy as np


# A downward-facing library motif becomes right-facing after np.rot90(mask, 1).
_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_HEADING_NAMES = ("down", "right", "up", "left")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    result = np.asarray(value)
    _require(result.ndim == 2 and all(1 <= x <= 30 for x in result.shape),
             "grids must be two-dimensional with axes 1..30")
    _require(np.issubdtype(result.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((result >= 0) & (result <= 9))), "colors must be 0..9")
    return result.astype(np.int8, copy=False)


def _trace(cells):
    """Trace the induced pixel graph, rejecting ties and disconnected cycles."""
    cells = set(cells)
    _require(2 <= len(cells) <= 36, "route length must be 2..36")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRECTIONS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    endpoints = sorted(p for p in cells if len(neighbors[p]) == 1)
    _require(len(endpoints) == 2 and all(len(v) in (1, 2) for v in neighbors.values()),
             "route must be a single induced chain")
    _require(endpoints[0][0] != endpoints[1][0], "endpoint rows must differ")
    current, previous = endpoints[0], None
    path, visited = [], set()
    for _ in range(len(cells)):
        _require(current not in visited, "route has a cycle")
        visited.add(current)
        path.append(current)
        choices = [p for p in neighbors[current] if p != previous]
        if not choices:
            break
        _require(len(choices) == 1, "route is branched")
        previous, current = current, choices[0]
    _require(visited == cells and path[-1] == endpoints[1], "route is disconnected")
    return path


def _parse(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    s = int(grid[0, 0])
    _require(bool(np.all(grid[0] == s) and np.all(grid[-1] == s)
                  and np.all(grid[:, 0] == s) and np.all(grid[:, -1] == s)),
             "missing outer frame")
    full_rows = [r for r in range(1, h) if np.all(grid[r] == s)]
    _require(bool(full_rows), "missing shelf divider")
    n = full_rows[0] - 1
    _require(n in (3, 5) and h >= n + 6, "invalid card size or route area")
    _require(np.all(grid[n + 1] == s) and np.all(grid[n + 3] == s),
             "missing key dividers")
    b = int(grid[n + 2, 1])
    _require(b != s, "background and frame must differ")
    labels, masks, targets = [], {}, {}
    for j in range(5):
        c = 1 + j * (n + 1)
        if c + n >= w:
            break
        panel = grid[1:n + 1, c:c + n]
        colors = [int(x) for x in np.unique(panel) if x != b]
        if not colors:
            break
        _require(len(colors) == 1 and colors[0] != s, "card must have one source label")
        label = colors[0]
        _require(label not in labels, "duplicate source label")
        _require(np.all(grid[1:n + 1, c - 1] == s)
                 and np.all(grid[1:n + 1, c + n] == s), "missing card boundary")
        target = int(grid[n + 2, c + n // 2])
        _require(target != b, "invisible key swatch")
        labels.append(label)
        masks[label] = panel == label
        targets[label] = target
    k = len(labels)
    _require(2 <= k <= 5 and k * (n + 1) + 1 <= w, "invalid library count")
    end = k * (n + 1)
    _require(np.all(grid[1:n + 1, end + 1:w - 1] == b), "nonblank shelf padding")
    expected_key = np.full(w, b, dtype=np.int8)
    expected_key[[0, w - 1]] = s
    for j, label in enumerate(labels):
        expected_key[(j + 1) * (n + 1)] = s
        expected_key[1 + j * (n + 1) + n // 2] = targets[label]
    _require(np.array_equal(grid[n + 2], expected_key), "malformed recoloring key")
    board = grid[n + 4:-1, 1:-1]
    _require(set(map(int, np.unique(board))) <= {b, *labels}, "unknown route color")
    cells = [tuple(map(int, p)) for p in np.argwhere(board != b)]
    path = _trace(cells)
    sequence = [int(board[p]) for p in path]
    _require(set(sequence) == set(labels), "every card must be used in the route")
    rmin, cmin = min(p[0] for p in path), min(p[1] for p in path)
    bh = max(p[0] for p in path) - rmin + 1
    bw = max(p[1] for p in path) - cmin + 1
    _require(bh + n - 1 <= 30 and bw + n - 1 <= 30, "expanded envelope is too large")
    return dict(n=n, k=k, background=b, separator=s, labels=tuple(labels),
                masks=masks, targets=targets, path=path, sequence=sequence,
                box=(rmin, cmin, bh, bw), board_shape=board.shape)


def _headings(path, outgoing=False):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    directions = steps + [steps[-1]] if outgoing else [steps[0]] + steps
    return [_DIRECTIONS.index(d) for d in directions]


def _paint(path, sequence, masks, targets, background, *, headings=None,
           order=None, opaque=False):
    """Forward ink application, used in construction and the input-only solver."""
    n = next(iter(masks.values())).shape[0]
    rmin, cmin = min(p[0] for p in path), min(p[1] for p in path)
    bh = max(p[0] for p in path) - rmin + 1
    bw = max(p[1] for p in path) - cmin + 1
    result = np.full((bh + n - 1, bw + n - 1), background, dtype=np.int8)
    if headings is None:
        headings = _headings(path)
    if order is None:
        order = range(len(path))
    tiles = {(label, q): np.rot90(mask, q) for label, mask in masks.items()
             for q in range(4)}
    for i in order:
        label = sequence[i]
        r, c = path[i][0] - rmin, path[i][1] - cmin
        mask = tiles[label, headings[i]]
        window = result[r:r + n, c:c + n]
        if opaque:
            window[:] = np.where(mask, targets[label], background)
        else:
            window[mask] = targets[label]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _paint(info["path"], info["sequence"], info["masks"], info["targets"],
                  info["background"])


def _ownership_render(info):
    """Check each pixel's last ink owner using backward visits and coordinate rotation.

    This is a generator-owned consistency check, not an independent solver.
    It does not call the forward painter or use np.rot90.
    """
    n = info["n"]
    rmin, cmin, bh, bw = info["box"]
    result = np.full((bh + n - 1, bw + n - 1), info["background"], dtype=np.int8)
    claimed = np.zeros(result.shape, dtype=bool)
    path, sequence = info["path"], info["sequence"]
    for i in range(len(path) - 1, -1, -1):
        a, b = (path[0], path[1]) if i == 0 else (path[i - 1], path[i])
        dr, dc = b[0] - a[0], b[1] - a[1]
        label = sequence[i]
        for u, v in np.argwhere(info["masks"][label]):
            u, v = int(u), int(v)
            if (dr, dc) == (1, 0):
                x, y = u, v
            elif (dr, dc) == (0, 1):
                x, y = n - 1 - v, u
            elif (dr, dc) == (-1, 0):
                x, y = n - 1 - u, n - 1 - v
            else:
                x, y = v, n - 1 - u
            r, c = path[i][0] - rmin + x, path[i][1] - cmin + y
            if not claimed[r, c]:
                result[r, c] = info["targets"][label]
                claimed[r, c] = True
    return result


def _alternative(info, name):
    path, seq = info["path"], info["sequence"]
    masks, targets = info["masks"], info["targets"]
    headings, order, opaque = _headings(path), None, False
    if name == "no_templates":
        dot = np.zeros((info["n"], info["n"]), dtype=bool)
        dot[info["n"] // 2, info["n"] // 2] = True
        masks = dict.fromkeys(info["labels"], dot)
    elif name == "no_key":
        targets = {label: label for label in info["labels"]}
    elif name == "library_order":
        first_seen = list(dict.fromkeys(seq))
        masks = {a: info["masks"][b] for a, b in zip(first_seen, info["labels"])}
        targets = {a: info["targets"][b] for a, b in zip(first_seen, info["labels"])}
    elif name == "wrong_endpoint":
        path, seq = path[::-1], seq[::-1]
        headings = _headings(path)
    elif name == "reverse_order":
        order = range(len(path) - 1, -1, -1)
    elif name == "row_sweep":
        order = sorted(range(len(path)), key=lambda i: path[i])
    elif name.startswith("column_"):
        _, side, last = name.split("_")
        columns = sorted({p[1] for p in path}, reverse=side == "rl")
        order = []
        for c in columns:
            order.extend(sorted((i for i in range(len(path)) if path[i][1] == c),
                                key=lambda i: path[i][0],
                                reverse=last == "up" and c == columns[-1]))
    elif name == "group_color":
        order = [i for label in info["labels"] for i, x in enumerate(seq) if x == label]
    elif name == "opaque_holes":
        opaque = True
    elif name == "no_rotation":
        headings = [0] * len(path)
    elif name == "outgoing_heading":
        headings = _headings(path, outgoing=True)
    elif name.startswith("heading_"):
        _, heading, turn = name.split("_")
        q = _HEADING_NAMES.index(heading)
        headings = [int(turn) if x == q else x for x in headings]
    else:
        raise ValueError("unknown counterfactual")
    return _paint(path, seq, masks, targets, info["background"], headings=headings,
                  order=order, opaque=opaque)


_CORE_ALTERNATIVES = ("no_templates", "no_key", "library_order", "wrong_endpoint",
                      "reverse_order", "row_sweep", "column_lr_down", "column_lr_up",
                      "column_rl_down", "column_rl_up", "group_color", "opaque_holes",
                      "no_rotation", "outgoing_heading")
_HEADING_ALTERNATIVES = tuple(f"heading_{name}_{q}" for h, name in enumerate(_HEADING_NAMES)
                              for q in range(4) if h != q)


def _differences(info, names=None, expected=None):
    if names is None:
        names = _CORE_ALTERNATIVES + _HEADING_ALTERNATIVES
    if expected is None:
        expected = _ownership_render(info)
    return {name: int(np.count_nonzero(expected != _alternative(info, name))) for name in names}


def counterfactual_differences(input_grid):
    """Changed-pixel counts for explicit A/B and connecting-concept ablations."""
    return _differences(_parse(input_grid))


def _features(info):
    path, seq = info["path"], info["sequence"]
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    runs = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    return dict(length=len(path), turns=sum(a != b for a, b in zip(steps, steps[1:])),
                above=min(p[0] for p in path) < path[0][0],
                recurrence=len(runs) > len(set(runs)), runs=len(runs),
                headings=set(_headings(path)))


def _encode(n, labels, masks, targets, path, sequence, background, separator,
            margins=(0, 0, 0, 0), extra_width=0):
    """Framed shelf and route; path coordinates may be translated or negative."""
    rmin, cmin = min(p[0] for p in path), min(p[1] for p in path)
    path = [(r - rmin, c - cmin) for r, c in path]
    bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
    top, bottom, left, right = margins
    board_h = bh + top + bottom
    width = max(len(labels) * (n + 1) + 1, bw + left + right + 2) + extra_width
    grid = np.full((n + 5 + board_h, width), background, dtype=np.int8)
    grid[[0, n + 1, n + 3, grid.shape[0] - 1], :] = separator
    grid[:, [0, width - 1]] = separator
    for j, label in enumerate(labels):
        c = 1 + j * (n + 1)
        grid[1:n + 1, c:c + n] = np.where(masks[label], label, background)
        grid[1:n + 1, c + n] = separator
        grid[n + 2, c + n] = separator
        grid[n + 2, c + n // 2] = targets[label]
    for (r, c), label in zip(path, sequence):
        grid[n + 4 + top + r, 1 + left + c] = label
    return grid


def _walk(rng, length, persistence, monotone=False):
    path, occupied = [(0, 0)], {(0, 0)}
    previous = None
    directions = ((1, 0), (0, 1), (0, -1)) if monotone else _DIRECTIONS
    for _ in range(length - 1):
        candidates, weights = [], []
        for dr, dc in directions:
            p = path[-1][0] + dr, path[-1][1] + dc
            if p in occupied:
                continue
            if sum((p[0] + a, p[1] + b) in occupied for a, b in _DIRECTIONS) != 1:
                continue
            candidates.append((p, (dr, dc)))
            weights.append(persistence if (dr, dc) == previous else 1.0)
        if not candidates:
            return None
        weights = np.asarray(weights, dtype=float)
        p, previous = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        path.append(p)
        occupied.add(p)
    return path


def _orient(rng, path):
    if rng.integers(2):
        path = [(r, -c) for r, c in path]
    for _ in range(int(rng.integers(4))):
        path = [(-c, r) for r, c in path]
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _fallback_path(rng, n, length):
    """Separated four-segment routes, with variable lengths and horizontal reflection."""
    if length < 12:
        segments = [((1, 0), 1), ((0, 1), length - 3), ((1, 0), 1)]
    else:
        choices = []
        for up in range(1, 4):
            for right in range(3, min(27 - n, length)):
                for left in range(1, right - 1):
                    down = length - 1 - up - right - left
                    if up + 2 <= down <= 23 - n:
                        choices.append((up, right, down, left))
        _require(bool(choices), "no legal constructive route")
        up, right, down, left = choices[int(rng.integers(len(choices)))]
        segments = [((-1, 0), up), ((0, 1), right), ((1, 0), down), ((0, -1), left)]
    path = [(0, 0)]
    for (dr, dc), count in segments:
        for _ in range(count):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    if rng.integers(2):
        path = [(r, -c) for r, c in path]
    return path


def _make_path(rng, n, length, strong, readable=False):
    persistence = float(rng.uniform(5, 20) if readable else rng.uniform(.5, 3.5))
    for _ in range(160):
        path = _walk(rng, length, persistence, monotone=readable)
        if path is None:
            continue
        path = _orient(rng, path)
        if path is None:
            continue
        bh = max(r for r, _ in path) - min(r for r, _ in path) + 1
        bw = max(c for _, c in path) - min(c for _, c in path) + 1
        if bh > 25 - n or bw > 31 - n:
            continue
        if strong and (set(_headings(path)) != {0, 1, 2, 3}
                       or min(r for r, _ in path) >= path[0][0]):
            continue
        return path
    return _fallback_path(rng, n, length)


def _mask(rng, n, force_irregular=False):
    style = 7 if force_irregular else int(rng.integers(9))
    r, c = np.indices((n, n))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (r == 0) | (r == n - 1) | (c == 0) | (c == n - 1)
    elif style == 2:
        mask = (r == int(rng.integers(n))) | (c == int(rng.integers(n)))
    elif style == 3:
        mask = (r + c) % 2 == int(rng.integers(2))
    elif style == 4:
        mask = (r == c) | ((r + c == n - 1) & bool(rng.integers(2)))
    elif style == 5:
        mask = np.zeros((n, n), dtype=bool)
        positions = rng.choice(n * n, int(rng.integers(1, n + 1)), replace=False)
        mask.flat[positions] = True
    elif style == 6:
        mask = (r % 2 == int(rng.integers(2))) if rng.integers(2) else (c % 2 == int(rng.integers(2)))
    else:
        density = rng.uniform(.25, .72) if style == 7 else rng.uniform(.7, .95)
        mask = rng.random((n, n)) < density
    if not mask.any():
        mask.flat[int(rng.integers(n * n))] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _sequence(rng, labels, length, strong=False, readable=False):
    k = len(labels)
    low = k + 2 if strong else k
    high = min(length, 3 * k + 6)
    runs = int(rng.integers(low, high + 1)) if not readable else k
    colors = [int(x) for x in rng.permutation(labels)]
    for _ in range(runs - k):
        colors.append(int(rng.choice([x for x in labels if x != colors[-1]])))
    sizes = np.ones(runs, dtype=int)
    weights = rng.dirichlet(np.full(runs, float(rng.choice((.2, .7, 2.0)))))
    sizes += rng.multinomial(length - runs, weights)
    return [color for color, count in zip(colors, sizes) for _ in range(int(count))]


def _sample_targets(rng, labels, background, strong):
    available = [c for c in range(10) if c != background]
    if strong:
        for _ in range(40):
            values = list(map(int, rng.choice(available, len(labels), replace=False)))
            if all(a != b for a, b in zip(labels, values)):
                return dict(zip(labels, values))
        return dict(zip(labels, labels[1:] + labels[:1]))
    values = list(map(int, rng.choice(available, len(labels))))
    return dict(zip(labels, values))


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _make_pair(rng, n, k, length, strong, readable, roles, used):
    path = _make_path(rng, n, length, strong, readable)
    required = (_CORE_ALTERNATIVES + _HEADING_ALTERNATIVES if strong == "train" else
                ("no_templates", "reverse_order", "row_sweep", "wrong_endpoint")
                if strong == "test" else ())
    for attempt in range(192):
        if attempt == 96:
            # Change visibility geometry if a random route hides a heading.
            path = _fallback_path(rng, n, length)
        b, s = roles if roles is not None else tuple(map(int, rng.choice(10, 2, replace=False)))
        labels = list(map(int, rng.choice([x for x in range(10) if x not in (b, s)], k,
                                         replace=False)))
        rng.shuffle(labels)
        masks = {label: _mask(rng, n, force_irregular=bool(strong) and rng.random() < .7)
                 for label in labels}
        if not strong and not readable and rng.random() < .2:
            masks[labels[-1]] = masks[labels[0]].copy()
        targets = _sample_targets(rng, labels, b, bool(strong) or readable)
        sequence = _sequence(rng, labels, length, bool(strong), readable)
        if required:
            # Reconstruct lightweight visible command data before presentation.
            rmin, cmin = min(r for r, _ in path), min(c for _, c in path)
            bh = max(r for r, _ in path) - rmin + 1
            bw = max(c for _, c in path) - cmin + 1
            info = dict(n=n, k=k, background=b, separator=s, labels=tuple(labels),
                        masks=masks, targets=targets, path=path, sequence=sequence,
                        box=(rmin, cmin, bh, bw))
            if strong == "train" and not _features(info)["recurrence"]:
                continue
            expected = _paint(path, sequence, masks, targets, b)
            if strong == "train" and len(set(targets.values()) & set(map(int, np.unique(expected)))) < 2:
                continue
            if any(value == 0 for value in _differences(info, required, expected).values()):
                continue
        else:
            expected = _paint(path, sequence, masks, targets, b)
        bh = max(r for r, _ in path) - min(r for r, _ in path) + 1
        bw = max(c for _, c in path) - min(c for _, c in path) + 1
        vertical_spare = 25 - n - bh
        horizontal_spare = 28 - bw
        top = int(rng.integers(0, min(3, vertical_spare) + 1))
        bottom = int(rng.integers(0, min(3, vertical_spare - top) + 1))
        left = int(rng.integers(0, min(3, horizontal_spare) + 1))
        right = int(rng.integers(0, min(3, horizontal_spare - left) + 1))
        base_w = max(k * (n + 1) + 1, bw + left + right + 2)
        extra_width = int(rng.integers(0, min(3, 30 - base_w) + 1))
        grid = _encode(n, labels, masks, targets, path, sequence, b, s,
                       (top, bottom, left, right), extra_width)
        if _key(grid) not in used:
            return {"input": grid, "output": expected}
    raise RuntimeError("bounded interaction-witness construction exhausted")


def _checked_int(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_int(num_train, "num_train", 2, 10)
    num_test = _checked_int(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    n = int(rng.choice((3, 5))) if regime != 1 else 3
    max_k = min(5, 29 // (n + 1))
    k = int(rng.integers(2, max_k + 1)) if regime == 0 else int(rng.integers(2, 4))
    shared_roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .55 else None
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            readable = split == "train" and i == 0
            strong = "train" if split == "train" and i == 1 else "test" if split == "test" and i == 0 else ""
            size = 5 if split == "test" and regime == 1 else n
            count_k = int(rng.integers(4, max_k + 1)) if split == "test" and regime == 2 else k
            if readable:
                length = int(rng.integers(max(7, count_k), 14))
            elif regime == 0:
                length = int(rng.integers(24, 37) if split == "test" else rng.integers(14, 23))
            else:
                length = int(rng.integers(14, 25) if strong else rng.integers(7, 25))
            pair = _make_pair(rng, size, count_k, length, strong, readable, shared_roles, used)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, parsed = set(), {"train": [], "test": []}
    differences = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(grid)
            rmin, cmin, bh, bw = info["box"]
            _require(output.shape == (bh + info["n"] - 1, bw + info["n"] - 1), "wrong envelope")
            _require(set(map(int, np.unique(output))) <= {info["background"], *info["targets"].values()},
                     "unexpected output color")
            expected = _ownership_render(info)
            _require(np.array_equal(output, expected), "incorrect last-ink ownership")
            parsed[split].append(info)
            differences[split].append(_differences(info, expected=output))
    # Teaching constraints, distinct from the semantic validity of one family member.
    names = _CORE_ALTERNATIVES + _HEADING_ALTERNATIVES
    _require(all(any(d[name] > 0 for d in differences["train"]) for name in names),
             "training leaves a documented shortcut unrefuted")
    _require(any(_features(p)["above"] and _features(p)["headings"] == {0, 1, 2, 3}
                 and _features(p)["recurrence"] for p in parsed["train"]),
             "missing folded endpoint/heading/recurrence witness")
    _require(any(len({target for label, target in p["targets"].items() if label != target}
                     & set(map(int, np.unique(pair["output"])))) >= 2
                 for p, pair in zip(parsed["train"], puzzle["train"])),
             "missing visible recoloring convention")
    test_names = ("no_templates", "reverse_order", "row_sweep", "wrong_endpoint")
    _require(any(all(d[name] > 0 for name in test_names) for d in differences["test"]),
             "missing order-sensitive test")


def _canonical_geometry(path):
    signatures = []
    for reflect in (1, -1):
        points = [(r, reflect * c) for r, c in path]
        for _ in range(4):
            rmin, cmin = min(r for r, _ in points), min(c for _, c in points)
            signatures.append(tuple(sorted((r - rmin, c - cmin) for r, c in points)))
            points = [(-c, r) for r, c in points]
    return min(signatures)


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"])) for split in ("train", "test")
                 for pair in puzzle[split])


def self_test():
    """Bounded checks and actual structural measurements over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, geometries, motifs, lengths, turns, sizes, counts, roles = (set() for _ in range(8))
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                feature = _features(info)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                shapes.add((pair["input"].shape, pair["output"].shape))
                geometries.add(_canonical_geometry(info["path"]))
                motifs.update((info["n"], mask.tobytes()) for mask in info["masks"].values())
                lengths.add(feature["length"])
                turns.add(feature["turns"])
                sizes.add(info["n"])
                counts.add(info["k"])
                roles.add((info["background"], info["separator"]))
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity failed")
    _require(sizes == {3, 5} and counts == {2, 3, 4, 5}, "scale/library coverage failed")
    _require(len(shapes) >= 50 and len(geometries) >= 70 and len(motifs) >= 150,
             "insufficient geometric/motif diversity")
    _require(len(lengths) >= 15 and len(turns) >= 8 and len(roles) >= 20,
             "insufficient structural/role diversity")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic seed")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1701 + a + b, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b), "wrong explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Hand-calculated transparent-hole fixture: later ink at row 1, earlier
    # ink survives at row 2 through the second stamp's empty center.
    m1, m2 = np.zeros((3, 3), bool), np.zeros((3, 3), bool)
    m1[2, 1], m2[0, 1] = True, True
    masks, targets = {1: m1, 2: m2}, {1: 3, 2: 4}
    path = [(0, 0), (1, 0)]
    grid = _encode(3, [2, 1], masks, targets, path, [1, 2], 0, 9)
    expected = np.array([[0, 0, 0], [0, 4, 0], [0, 3, 0], [0, 0, 0]], dtype=int)
    _require(np.array_equal(solve(grid), expected), "transparent boundary fixture failed")
    # Right travel rotates the bottom-center pixel to the right; last arrival
    # is down, and that stamp overwrites just one of the two right-facing marks.
    path = [(0, 0), (0, 1), (1, 1)]
    grid = _encode(3, [2, 1], masks, targets, path, [1, 1, 2], 0, 9)
    expected = np.zeros((4, 4), dtype=int)
    expected[1, 2:4] = [4, 3]
    _require(np.array_equal(solve(grid), expected), "rotation/occlusion fixture failed")
    # Retain complete empty mask borders rather than cropping to actual ink.
    center = np.zeros((5, 5), bool)
    center[2, 2] = True
    grid = _encode(5, [1, 2], {1: center, 2: center}, {1: 3, 2: 3},
                   [(0, 0), (1, 0)], [1, 2], 0, 9)
    answer = solve(grid)
    _require(answer.shape == (6, 5) and np.count_nonzero(answer) == 2,
             "empty-border/shared-target fixture failed")
    for cells in ([(0, 0), (0, 1)], [(0, 1), (1, 0), (1, 1), (1, 2)],
                  [(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (1, 0), (4, 4)]):
        try:
            _trace(cells)
        except ValueError:
            continue
        raise AssertionError("malformed route accepted")
    bad = copy.deepcopy(puzzles[0])
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "input_output_shapes": len(shapes), "dihedral_normalized_path_geometries": len(geometries),
            "binary_motifs": len(motifs), "path_lengths": sorted(lengths),
            "turn_counts": sorted(turns), "stamp_sizes": sorted(sizes), "library_counts": sorted(counts),
            "frame_background_assignments": len(roles), "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
