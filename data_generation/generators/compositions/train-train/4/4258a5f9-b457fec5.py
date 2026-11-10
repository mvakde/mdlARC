"""Marker spines expand into stacks of 3x3 cards dealt from a cyclic key.

Composition of 4258a5f9 (every isolated point becomes the preserved centre
of a solid 3x3 square) and b457fec5 (a diagonal chain of overlapping squares
is coloured layer by layer from an ordered key, cyclically, later layers on
top, restarting for every object).

Every input has a dominant background, one isolated horizontal key of 2..5
distinct colours read left to right, and one or more marker objects.  A marker
object is an 8-connected path that holds exactly one cell per row over a run
of consecutive rows (a lone point is a path of length one).  Each marker cell
is the centre of a 3x3 card.  Cards of one object are dealt top to bottom:
the k-th card from the top takes key[k mod len(key)], each lower card is laid
over the cards above it, every object restarts at key[0], and finally every
original non-background cell (markers and key) is restored on top.
"""
from __future__ import annotations

import numpy as np

MAX_SIDE = 30
KEY_LENGTHS = (2, 3, 4, 5)
PATH_MODES = ("diagonal", "zigzag", "free")
SHIFTS = ("longer", "denser", "new_key", "overlap")
ALT_RULES = ("plain_stamp", "no_preserve", "upper_on_top", "key_reversed",
             "count_from_bottom", "bottom_up", "continuous_cycle", "clamped",
             "row_phase", "key_row_phase", "nearest_end_first", "leftmost_first",
             "markers_only", "marker_colour_cards")
TEST_ALT_RULES = ("plain_stamp", "no_preserve", "upper_on_top", "bottom_up",
                  "markers_only", "marker_colour_cards")


class _Retry(Exception):
    """Internal signal that a sampled scene should be redrawn."""


# --------------------------------------------------------------------------
# Input-only reading and solving
# --------------------------------------------------------------------------

def _components(mask):
    """8-connected components of a boolean mask, each a sorted cell list."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack, cells = [(row, col)], []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width
                                and mask[nr, nc] and not seen[nr, nc]):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            found.append(sorted(cells))
    return found


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _read_scene(grid):
    """Background, key colours, key cells and marker objects (top-down)."""
    background = _background(grid)
    components = _components(grid != background)
    key_index = None
    for index, cells in enumerate(components):
        if len(cells) >= 2 and len({r for r, _ in cells}) == 1:
            if key_index is None or len(cells) > len(components[key_index]):
                key_index = index
    key_cells = sorted(components[key_index], key=lambda p: p[1]) if key_index is not None else []
    key = [int(grid[r, c]) for r, c in key_cells]
    objects = [cells for index, cells in enumerate(components) if index != key_index]
    return background, key, key_cells, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    background, key, _, objects = _read_scene(grid)
    output = grid.copy()
    if not key:
        return output
    height, width = grid.shape
    for cells in objects:
        # Cells are sorted by row: the top card is dealt first, lower cards
        # are laid over it.
        for layer, (r, c) in enumerate(sorted(cells)):
            output[max(0, r - 1):min(height, r + 2),
                   max(0, c - 1):min(width, c + 2)] = key[layer % len(key)]
    keep = grid != background
    output[keep] = grid[keep]
    return output


# --------------------------------------------------------------------------
# Strict family parsing, reference rendering and alternative rules
# --------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _strict_scene(grid):
    """Parse an input and assert every family constraint visible in it."""
    height, width = grid.shape
    background = _background(grid)
    _require(np.count_nonzero(grid == background) * 2 > grid.size,
             "background must cover most of the input")
    components = _components(grid != background)
    keys = [cells for cells in components if len({r for r, _ in cells}) == 1 and len(cells) >= 2]
    _require(len(keys) == 1, "exactly one horizontal key strip is required")
    key_cells = sorted(keys[0], key=lambda p: p[1])
    key_row = key_cells[0][0]
    key_cols = [c for _, c in key_cells]
    _require(key_cols == list(range(key_cols[0], key_cols[0] + len(key_cols))),
             "key must be one contiguous strip")
    key = [int(grid[key_row, c]) for c in key_cols]
    _require(len(key) in KEY_LENGTHS, "key length out of family range")
    _require(len(set(key)) == len(key), "key colours must be distinct")
    objects, colours = [], set()
    for cells in components:
        if cells is keys[0]:
            continue
        rows = [r for r, _ in cells]
        _require(rows == list(range(rows[0], rows[0] + len(rows))),
                 "marker object must hold one cell per consecutive row")
        cols = [c for _, c in cells]
        _require(all(abs(a - b) <= 1 for a, b in zip(cols, cols[1:])),
                 "marker path must be 8-connected")
        object_colours = {int(grid[r, c]) for r, c in cells}
        _require(len(object_colours) == 1, "each marker object has one colour")
        _require(not object_colours & set(key), "marker colour must not appear in the key")
        _require(all(1 <= r <= height - 2 and 1 <= c <= width - 2 for r, c in cells),
                 "every 3x3 card must lie inside the grid")
        colours |= object_colours
        objects.append(cells)
    _require(objects, "at least one marker object is required")
    for i, first in enumerate(objects):
        a = np.array(first)
        for second in objects[i + 1:]:
            b = np.array(second)
            gap = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2).min()
            _require(gap >= 3, "cards of different objects must not overlap")
        gap = np.abs(a[:, None, :] - np.array(key_cells)[None, :, :]).max(axis=2).min()
        _require(gap >= 3, "cards must leave a background gap around the key")
    return {"background": background, "key": key, "key_cells": key_cells,
            "key_row": key_row, "objects": objects, "marker_colours": colours}


def _reference_output(grid, scene):
    """Cell-wise rule: a background cell takes the colour of the lowest
    (latest-dealt) marker of its object at Chebyshev distance one."""
    height, width = grid.shape
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    output = grid.copy()
    key = scene["key"]
    claimed = np.zeros(grid.shape, dtype=bool)
    for cells in scene["objects"]:
        marker_rows = np.array([r for r, _ in cells])[:, None, None]
        marker_cols = np.array([c for _, c in cells])[:, None, None]
        distance = np.maximum(np.abs(rows[None] - marker_rows), np.abs(cols[None] - marker_cols))
        layers = np.arange(len(cells))[:, None, None]
        latest = np.where(distance <= 1, layers, -1).max(axis=0)
        covered = (latest >= 0) & (grid == scene["background"])
        _require(not np.any(covered & claimed), "cards of two objects overlap")
        claimed |= covered
        output[covered] = np.array(key)[latest[covered] % len(key)]
    return output


def _alternative(grid, scene, rule):
    """Outputs of plausible shortcut rules, used to prove the demonstrations
    distinguish the intended rule."""
    key = list(scene["key"])
    if rule == "key_reversed":
        key = key[::-1]
    k = len(key)
    key_cells = np.array(scene["key_cells"])

    def key_distance(cell):
        return int(np.abs(key_cells - np.array(cell)).max(axis=1).min())

    output = grid.copy()
    offset = 0
    for cells in sorted(scene["objects"], key=lambda cells: cells[0]):
        n = len(cells)
        # Dealing order: the j-th dealt card takes colour index j and is laid
        # over every card dealt before it.
        deal = list(range(n))
        if rule == "bottom_up":
            deal = deal[::-1]
        elif rule == "leftmost_first":
            deal = sorted(deal, key=lambda i: (cells[i][1], cells[i][0]))
        elif rule == "nearest_end_first" and key_distance(cells[-1]) < key_distance(cells[0]):
            deal = deal[::-1]
        colours = {}
        for j, layer in enumerate(deal):
            r, c = cells[layer]
            if rule == "plain_stamp":
                index = 0
            elif rule == "count_from_bottom":
                index = n - 1 - layer
            elif rule == "continuous_cycle":
                index = offset + j
            elif rule == "clamped":
                index = min(j, k - 1)
            elif rule == "row_phase":
                index = r
            elif rule == "key_row_phase":
                index = r - scene["key_row"]
            else:
                index = j
            colours[layer] = int(grid[r, c]) if rule == "marker_colour_cards" else key[index % k]
        paint = deal[::-1] if rule == "upper_on_top" else deal
        for layer in paint:
            r, c = cells[layer]
            if rule == "markers_only":
                output[r, c] = colours[layer]
            else:
                output[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = colours[layer]
        offset += n
    if rule not in ("no_preserve", "markers_only"):
        keep = grid != scene["background"]
        output[keep] = grid[keep]
    return output


def _features(scene):
    """Visible structural features of the marker objects in one input."""
    found = set()
    for cells in scene["objects"]:
        if len(cells) == 1:
            found.add("single")
            continue
        steps = [b[1] - a[1] for a, b in zip(cells, cells[1:])]
        found |= {{-1: "left", 0: "down", 1: "right"}[s] for s in steps}
        if any(a != b for a, b in zip(steps, steps[1:])):
            found.add("bend")
    if len(scene["marker_colours"]) > 1:
        found.add("multicolour")
    return found


def _as_array(value):
    array = np.asarray(value)
    _require(array.ndim == 2, "grids must be two-dimensional")
    _require(all(1 <= side <= MAX_SIDE for side in array.shape), "grid side out of range")
    _require(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
    _require(array.size and 0 <= array.min() and array.max() <= 9, "colours must be 0..9")
    return array.astype(int)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    parsed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs need exactly input and output")
            grid, answer = _as_array(pair["input"]), _as_array(pair["output"])
            _require(grid.shape == answer.shape, "input and output shapes differ")
            signature = (grid.shape, grid.tobytes())
            _require(signature not in seen, "duplicate input in puzzle")
            seen.add(signature)
            scene = _strict_scene(grid)
            reference = _reference_output(grid, scene)
            _require(np.array_equal(reference, answer), "output disagrees with the rule")
            _require(np.array_equal(solve(grid), answer), "solve disagrees with the output")
            # Every original cell survives; exactly the card cells change.
            keep = grid != scene["background"]
            _require(np.array_equal(answer[keep], grid[keep]), "markers or key changed")
            # The lowest card of each object is complete: all of its cells
            # except preserved markers carry its key colour.
            for cells in scene["objects"]:
                r, c = cells[-1]
                block = answer[r - 1:r + 2, c - 1:c + 2]
                open_cells = grid[r - 1:r + 2, c - 1:c + 2] == scene["background"]
                colour = scene["key"][(len(cells) - 1) % len(scene["key"])]
                _require(bool(np.all(block[open_cells] == colour)), "lowest card must stay complete")
            parsed[split].append((grid, answer, scene))
    # The demonstrations must refute each shortcut rule at least once.
    for rule in ALT_RULES:
        _require(any(not np.array_equal(_alternative(g, s, rule), a) for g, a, s in parsed["train"]),
                 f"training does not refute shortcut {rule}")
    for grid, answer, scene in parsed["test"]:
        for rule in TEST_ALT_RULES:
            _require(not np.array_equal(_alternative(grid, scene, rule), answer),
                     f"test pair does not exercise both parents ({rule})")
    # Training covers each structure a test relies on.
    train_features = set().union(*(_features(s) for _, _, s in parsed["train"]))
    for _, _, scene in parsed["test"]:
        missing = _features(scene) - train_features
        _require(not missing, f"test uses untrained features {sorted(missing)}")
    _require(any(len(cells) > len(s["key"]) for _, _, s in parsed["train"] for cells in s["objects"]),
             "training must show the key wrapping")
    # Restart is visible whatever order a solver might give the objects: one
    # training pair has several objects and none of them ends a full cycle.
    _require(any(len(s["objects"]) >= 2 and all(len(cells) % len(s["key"]) for cells in s["objects"])
                 for _, _, s in parsed["train"]), "training must expose the per-object restart")
    _require(len({"left", "right", "down"} & train_features) >= 2,
             "training must show more than one card offset")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _puzzle_params(rng):
    mode = PATH_MODES[int(rng.choice(3, p=[0.42, 0.30, 0.28]))]
    shift = SHIFTS[int(rng.choice(4, p=[0.30, 0.22, 0.22, 0.26]))]
    background = 0 if rng.random() < 0.4 else int(rng.integers(1, 10))
    marker = int(rng.choice([c for c in range(10) if c != background]))
    weights = np.array([0.18, 0.32, 0.32, 0.18])
    if shift == "new_key":
        test_k = int(rng.choice(KEY_LENGTHS, p=weights))
        others = [k for k in KEY_LENGTHS if k != test_k]
        size = int(rng.integers(1, len(others) + 1))
        train_keys = sorted(int(k) for k in rng.choice(others, size=size, replace=False))
        test_keys = [test_k]
    else:
        size = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]))
        train_keys = sorted(int(k) for k in rng.choice(KEY_LENGTHS, size=size, replace=False,
                                                       p=weights))
        if rng.random() < 0.7:
            test_keys = list(train_keys)
        else:
            test_keys = list(KEY_LENGTHS)
    return {
        "mode": mode,
        "shift": shift,
        "singletons": bool(rng.random() < 0.8),
        "multicolour": bool(rng.random() < 0.22),
        "vary_background": bool(rng.random() < 0.45),
        "vary_marker": bool(rng.random() < 0.45),
        "background": background,
        "marker": marker,
        "separation": 3 if rng.random() < 0.25 else 4,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "train_cap": int(rng.integers(6, 10)) if shift == "longer" else 13,
        "tight": float(rng.choice([0.15, 0.45, 0.8])),
    }


def _chain_length(rng, k, kind, cap):
    if kind == "single":
        return 1
    if kind == "short":
        low, high = 2, k
    elif kind == "wrap":
        low, high = k + 1, 2 * k
    else:
        low, high = 2 * k + 1, 3 * k + 2
    high = min(high, cap)
    if low > high:
        low, high = 2, max(2, cap)
    return int(rng.integers(low, high + 1))


def _random_kind(rng, singletons):
    kinds = ["single", "short", "wrap", "multi"]
    p = np.array([0.28, 0.3, 0.27, 0.15]) if singletons else np.array([0.0, 0.42, 0.37, 0.21])
    return kinds[int(rng.choice(4, p=p / p.sum()))]


def _lengths(rng, P, role, k, context):
    cap = P["train_cap"] if role != "test" else 14
    single = P["singletons"]
    if role == "wrap":
        lengths = [_chain_length(rng, k, "wrap" if rng.random() < 0.55 else "multi", cap)]
        for _ in range(int(rng.integers(0, 3))):
            lengths.append(1 if single and rng.random() < 0.65 else _chain_length(rng, k, "short", cap))
        return lengths
    if role == "restart":
        lengths = []
        count = int(rng.integers(2, 4))
        for index in range(count):
            if index < 2:
                kind = ["short", "wrap", "multi"][int(rng.choice(3, p=[0.4, 0.4, 0.2]))]
                n = _chain_length(rng, k, kind, min(cap, 10))
            else:
                n = 1 if single else _chain_length(rng, k, "short", cap)
            if n % k == 0:
                n = n + 1 if n + 1 <= cap else n - 1
            if n % k == 0 or n < (2 if index < 2 else 1):
                n = 3 if k != 3 else 2
            lengths.append(n)
        return lengths
    if role == "free":
        count = int(rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.3, 0.3, 0.17, 0.08]))
        lengths = [_chain_length(rng, k, _random_kind(rng, single), cap if count < 3 else min(cap, 9))
                   for _ in range(count)]
        if max(lengths) < 2:
            lengths[0] = _chain_length(rng, k, "short", cap)
        return lengths
    # Test pairs.
    shift = P["shift"]
    if shift == "denser":
        count = min(7, context["max_train_objects"] + int(rng.integers(1, 3)))
    else:
        count = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.3, 0.15]))
    sub_cap = 14 if count <= 2 else (10 if count <= 4 else 7)
    lengths = [_chain_length(rng, k, _random_kind(rng, single), sub_cap) for _ in range(count)]
    if shift == "longer":
        floor = context["max_train_length"] + 1
        lengths[0] = int(rng.integers(min(floor, 14), min(14, floor + 4) + 1))
    if max(lengths) < 3:
        lengths[0] = int(rng.integers(3, max(4, min(sub_cap, 2 * k + 2)) + 1))
    return lengths


def _path(rng, n, mode, first_direction=None, force_bend=False):
    """Column offsets of a path with one marker per row."""
    if n == 1:
        return [0]
    direction = first_direction if first_direction is not None else (1 if rng.random() < 0.5 else -1)
    steps = []
    if mode == "diagonal":
        steps = [direction] * (n - 1)
    elif mode == "zigzag":
        while len(steps) < n - 1:
            steps += [direction] * int(rng.integers(1, 5))
            direction = -direction
        steps = steps[:n - 1]
    else:
        current = direction
        while len(steps) < n - 1:
            steps += [current] * int(rng.integers(1, 4))
            current = int(rng.choice([s for s in (-1, 0, 1) if s != current]))
        steps = steps[:n - 1]
    if force_bend and n >= 3 and all(s == steps[0] for s in steps):
        cut = int(rng.integers(1, n - 1))
        replacement = -steps[0] if steps[0] != 0 or mode == "zigzag" else (1 if rng.random() < 0.5 else -1)
        steps = steps[:cut] + [replacement] * (n - 1 - cut)
    cols = np.concatenate([[0], np.cumsum(steps)])
    return [int(c) for c in cols - cols.min()]


def _chebyshev_map(height, width, markers):
    if not len(markers):
        return np.full((height, width), 99)
    m = np.array(markers)
    rows = np.arange(height)[None, :, None]
    cols = np.arange(width)[None, None, :]
    distance = np.maximum(np.abs(rows - m[:, 0, None, None]), np.abs(cols - m[:, 1, None, None]))
    return distance.min(axis=0)


def _place(rng, P, paths, k):
    heights = [len(p) for p in paths]
    widths = [max(p) + 1 for p in paths]
    need_h = max(heights) + 2
    need_w = max(max(widths) + 2, k)
    sep = P["separation"]
    area = sum((h + sep) * (w + sep) for h, w in zip(heights, widths)) + 3 * (k + 4)
    if need_h > MAX_SIDE or need_w > MAX_SIDE or area > MAX_SIDE * MAX_SIDE:
        return None
    order = sorted(range(len(paths)), key=lambda i: -heights[i] * widths[i])
    for attempt in range(24):
        grow = attempt // 4
        if rng.random() < P["tight"]:
            height = need_h + int(rng.integers(0, 4)) + 3 * grow
            width = need_w + int(rng.integers(0, 4)) + 3 * grow
        else:
            height = int(rng.integers(need_h, MAX_SIDE + 1))
            width = int(rng.integers(need_w, MAX_SIDE + 1))
        height, width = min(height, MAX_SIDE), min(width, MAX_SIDE)
        if height * width < area * 1.15:
            scale = (area * 1.3 / (height * width)) ** 0.5
            height = min(MAX_SIDE, int(np.ceil(height * scale)))
            width = min(MAX_SIDE, int(np.ceil(width * scale)))
        markers, positions = [], [None] * len(paths)
        for index in order:
            path = paths[index]
            h, w = heights[index], widths[index]
            if height - 1 - h < 1 or width - 1 - w < 1:
                break
            cells = np.array([(r, c) for r, c in enumerate(path)])
            chosen = None
            for _ in range(60):
                top = int(rng.integers(1, height - h))
                left = int(rng.integers(1, width - w))
                shifted = cells + (top, left)
                if not markers:
                    chosen = shifted
                    break
                gap = np.abs(shifted[:, None, :] - np.array(markers)[None, :, :]).max(axis=2).min()
                if gap >= sep:
                    chosen = shifted
                    break
            if chosen is None:
                break
            positions[index] = [(int(r), int(c)) for r, c in chosen]
            markers.extend(positions[index])
        if any(p is None for p in positions):
            continue
        distance = _chebyshev_map(height, width, markers)
        slots = [(r, c) for r in range(height) for c in range(width - k + 1)
                 if distance[r, c:c + k].min() >= 3]
        if not slots:
            continue
        roll = rng.random()
        tops = {p[0][0] for p in positions}
        if roll < 0.3:
            preferred = [s for s in slots if s[0] <= 2]
        elif roll < 0.5:
            preferred = [s for s in slots if s[0] in tops]
        else:
            preferred = []
        pool = preferred or slots
        key_row, key_col = pool[int(rng.integers(len(pool)))]
        return height, width, positions, (key_row, key_col)
    return None


def _colours(rng, P, count, k):
    background = P["background"]
    if P["vary_background"] and rng.random() < 0.6:
        background = int(rng.integers(0, 10))
    free = [c for c in range(10) if c != background]
    if P["multicolour"] and count >= 2:
        n_colours = int(rng.integers(2, min(3, count) + 1))
        palette = [int(c) for c in rng.choice(free, size=n_colours, replace=False)]
        marker_colours = [palette[i % n_colours] for i in range(count)]
        marker_colours = [int(c) for c in rng.permutation(marker_colours)]
    else:
        marker = P["marker"]
        if marker == background or (P["vary_marker"] and rng.random() < 0.6):
            marker = int(rng.choice(free))
        marker_colours = [marker] * count
    remaining = [c for c in free if c not in marker_colours]
    key = [int(c) for c in rng.choice(remaining, size=k, replace=False)]
    return background, marker_colours, key


def _make_pair(rng, P, role, k, context):
    for _ in range(40):
        lengths = _lengths(rng, P, role, k, context)
        first = 1 if rng.random() < 0.5 else -1
        paths = []
        for index, n in enumerate(lengths):
            direction = None
            if role == "restart" and index < 2:
                direction = first if index == 0 else -first
            force = role in ("wrap", "restart") and P["mode"] != "diagonal" and index == 0
            paths.append(_path(rng, n, P["mode"], direction, force))
        placement = _place(rng, P, paths, k)
        if placement is None:
            continue
        height, width, positions, (key_row, key_col) = placement
        background, marker_colours, key = _colours(rng, P, len(paths), k)
        grid = np.full((height, width), background, dtype=int)
        for cells, colour in zip(positions, marker_colours):
            for r, c in cells:
                grid[r, c] = colour
        grid[key_row, key_col:key_col + k] = key
        output = grid.copy()
        for cells in positions:
            for layer, (r, c) in enumerate(cells):
                output[r - 1:r + 2, c - 1:c + 2] = key[layer % k]
        keep = grid != background
        output[keep] = grid[keep]
        try:
            scene = _strict_scene(grid)
        except AssertionError:
            continue
        if role == "test":
            if _features(scene) - context["train_features"]:
                continue
            if any(np.array_equal(_alternative(grid, scene, rule), output) for rule in TEST_ALT_RULES):
                continue
        return {"input": grid, "output": output}, scene
    raise _Retry("could not build pair")


def _build(rng, P, num_train, num_test):
    roles = ["wrap", "restart"] + ["free"] * (num_train - 2)
    train, scenes, seen = [], [], set()
    for role in roles:
        for _ in range(8):
            k = int(rng.choice(P["train_keys"]))
            pair, scene = _make_pair(rng, P, role, k, None)
            signature = pair["input"].tobytes() + bytes(pair["input"].shape)
            if signature not in seen:
                break
        else:
            raise _Retry("duplicate training input")
        seen.add(signature)
        train.append(pair)
        scenes.append(scene)
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    scenes = [scenes[int(i)] for i in order]
    context = {
        "train_features": set().union(*(_features(s) for s in scenes)),
        "max_train_length": max(len(cells) for s in scenes for cells in s["objects"]),
        "max_train_objects": max(len(s["objects"]) for s in scenes),
    }
    test = []
    for _ in range(num_test):
        for _ in range(8):
            k = int(rng.choice(P["test_keys"]))
            pair, _ = _make_pair(rng, P, "test", k, context)
            signature = pair["input"].tobytes() + bytes(pair["input"].shape)
            if signature not in seen:
                break
        else:
            raise _Retry("duplicate test input")
        seen.add(signature)
        test.append(pair)
    return {"train": train, "test": test}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        P = _puzzle_params(rng)
        try:
            puzzle = _build(rng, P, num_train, num_test)
            validate_generated_puzzle(puzzle)
        except (_Retry, AssertionError):
            continue
        return puzzle
    raise RuntimeError("could not generate a valid puzzle")


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def _same(first, second):
    return all(np.array_equal(a[name], b[name])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for name in ("input", "output"))


_HAND_INPUT = ("000000350", "000000000", "004000000", "000400000",
               "000400000", "000000040", "000000000")
_HAND_OUTPUT = ("000000350", "033300000", "034550000", "033430000",
                "003430333", "003330343", "000000333")


def _digits(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def self_test() -> None:
    # Hand-checked rule example: key 3,5; a spine (2,2),(3,3),(4,3) with a
    # rightward step then a straight drop; a lone marker at (5,7).
    hand = _digits(_HAND_INPUT)
    assert np.array_equal(solve(hand), _digits(_HAND_OUTPUT))
    hand_scene = _strict_scene(hand)
    assert np.array_equal(_reference_output(hand, hand_scene), _digits(_HAND_OUTPUT))
    assert _features(hand_scene) == {"right", "down", "bend", "single"}
    for rule in ALT_RULES:
        if rule not in ("row_phase", "key_row_phase", "nearest_end_first", "leftmost_first",
                        "count_from_bottom", "continuous_cycle", "clamped"):
            assert not np.array_equal(_alternative(hand, hand_scene, rule), _digits(_HAND_OUTPUT)), rule
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(5, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    for counts in ((2, 1), (10, 4), (5, 4), (3, 2)):
        puzzle = generate_puzzle(11, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    assert _same(generate_puzzle(123, 5, 2), generate_puzzle(123, 5, 2))
    shapes, keys, counts, lengths = set(), set(), set(), set()
    features, backgrounds = set(), set()
    varied_shape = varied_background = longer_test = denser_test = new_key_test = 0
    straight_puzzles = bent_puzzles = 0
    all_features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        pair_shapes, pair_backgrounds, puzzle_features = set(), set(), set()
        train_max = train_objects = 0
        train_keys = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = np.asarray(pair["input"])
                assert np.array_equal(solve(grid), pair["output"])
                scene = _strict_scene(grid)
                shapes.add(grid.shape)
                pair_shapes.add(grid.shape)
                keys.add(len(scene["key"]))
                counts.add(len(scene["objects"]))
                pair_backgrounds.add(scene["background"])
                backgrounds.add(scene["background"])
                longest = max(len(cells) for cells in scene["objects"])
                lengths.add(longest)
                if split == "train":
                    train_max = max(train_max, longest)
                    train_objects = max(train_objects, len(scene["objects"]))
                    train_keys.add(len(scene["key"]))
                else:
                    longer_test += longest > train_max
                    denser_test += len(scene["objects"]) > train_objects
                    new_key_test += len(scene["key"]) not in train_keys
                puzzle_features |= _features(scene)
        features |= puzzle_features
        all_features.append(frozenset(puzzle_features))
        varied_shape += len(pair_shapes) > 1
        varied_background += len(pair_backgrounds) > 1
        bent_puzzles += "bend" in puzzle_features
        straight_puzzles += not puzzle_features & {"bend", "down"}
    assert keys == set(KEY_LENGTHS), keys
    assert {"single", "left", "right", "down", "bend", "multicolour"} <= features, features
    assert len(shapes) >= 100 and varied_shape >= 38, (len(shapes), varied_shape)
    assert len(backgrounds) >= 8 and varied_background >= 10
    assert max(counts) >= 5 and min(counts) == 1
    assert max(lengths) >= 12
    assert min(longer_test, denser_test, new_key_test) >= 5, (longer_test, denser_test, new_key_test)
    assert straight_puzzles >= 8 and bent_puzzles >= 12, (straight_puzzles, bent_puzzles)
    assert len(set(all_features)) >= 8
