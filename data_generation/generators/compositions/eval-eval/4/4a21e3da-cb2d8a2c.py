"""Obstacle-encoded bent guides cut payloads into collectively moved banks.

Only Python's standard library and NumPy are used. The decoder reads every
convention from one grid; construction plans never enter solve().
"""
from collections import deque
from itertools import permutations
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _flood(blocked, start):
    h, w = blocked.shape
    reached = np.zeros((h, w), dtype=bool)
    if blocked[start]:
        return reached
    reached[start] = True
    queue = deque([start])
    while queue:
        for r, c in _neighbors(queue.popleft()):
            if 0 <= r < h and 0 <= c < w and not blocked[r, c] and not reached[r, c]:
                reached[r, c] = True
                queue.append((r, c))
    return reached


def _trace(shape, initial, bars):
    """Return a mask and ordered points in coordinates facing inward/down."""
    h, w = shape
    path, points = np.zeros(shape, dtype=bool), []
    current, forward, previous = initial, 0, -1

    def add(r, c):
        if not path[r, c]:
            path[r, c] = True
            points.append((r, c))

    for f, lo, hi, n, sign in bars:
        d = n + 1
        target = hi + d if sign == 1 else lo - d
        turn = f - d
        _require(1 <= target < w - 1, "route must stay inside the lateral edges")
        _require(turn >= 1 and turn > previous and turn >= forward, "overlapping turns")
        _require(lo <= current <= hi, "bar does not intercept incoming route")
        for r in range(forward, turn + 1):
            add(r, current)
        for c in range(current, target + sign, sign):
            add(turn, c)
        current, forward, previous = target, turn, f
    for r in range(forward, h):
        add(r, current)
    return path, points


def _read_roles(a, bg, route, roles):
    bar, mark, payload = roles
    h, w = a.shape
    cells = set(map(tuple, np.argwhere(a == payload).tolist()))
    _require(len(cells) >= 3 and len(_components(cells)) == 1, "payload must be connected")
    rs, cs = zip(*cells)
    _require(min(rs) > 0 and max(rs) < h - 1 and min(cs) > 0 and max(cs) < w - 1,
             "payload must be interior")
    _require(min(rs) != max(rs) and min(cs) != max(cs), "payload must span both dimensions")
    structure = (a == bar) | (a == mark)
    rows = np.flatnonzero(structure.any(axis=1))
    _require(1 <= len(rows) <= 4, "expected one to four bars")
    bars = []
    for raw in rows:
        f = int(raw)
        occupied = np.flatnonzero(structure[f])
        lo, hi = int(occupied[0]), int(occupied[-1])
        _require(0 < f < h - 1 and 0 < lo <= hi < w - 1, "bars must be interior")
        _require(len(occupied) == hi - lo + 1, "bar must be contiguous")
        markers = tuple(map(int, np.flatnonzero(a[f] == mark)))
        n = len(markers)
        _require(1 <= n <= 4 and hi - lo + 1 >= 2 * n + 1, "invalid unary bar")
        low = markers == tuple(lo + 2 * j for j in range(n))
        high = markers == tuple(hi - 2 * j for j in reversed(range(n)))
        _require(low != high, "marker end must be unique")
        bars.append((f, lo, hi, n, 1 if low else -1))
    initial = int(np.argwhere(a == route)[0, 1])
    path, points = _trace(a.shape, initial, bars)
    _require(not np.any(path & structure), "route collides with a bar")
    left = _flood(path, (0, 0))
    right = _flood(path, (0, w - 1))
    _require(not np.any(left & right) and np.array_equal(left | right, ~path),
             "guide must form exactly two banks")
    fixed = {p for p in cells if path[p]}
    groups = [{p for p in cells if bank[p]} for bank in (left, right)]
    _require(bool(fixed) and all(groups), "payload must meet the guide and both banks")
    return {"a": a, "bg": bg, "route": route, "bar": bar, "mark": mark,
            "payload": payload, "cells": cells, "bars": bars, "initial": initial,
            "path": path, "points": points, "banks": (left, right),
            "fixed": fixed, "groups": groups}


def _targets(cells, shape, right):
    rs, cs = zip(*cells)
    dr = -min(rs)
    dc = shape[1] - 1 - max(cs) if right else -min(cs)
    return {(r + dr, c + dc) for r, c in cells}


def _render(info, separate=False, keep_in_place=False):
    a = info["a"]
    out = np.full_like(a, info["bg"])
    structure = (a == info["bar"]) | (a == info["mark"])
    out[structure] = info["bar"]
    out[info["path"]] = info["route"]
    for p in info["fixed"]:
        out[p] = info["payload"]
    if keep_in_place:
        for p in info["cells"]:
            out[p] = info["payload"]
        return out
    for right, cells in enumerate(info["groups"]):
        pieces = _components(cells) if separate else [cells]
        for piece in pieces:
            targets = _targets(piece, out.shape, right)
            if not separate:
                _require(all(out[p] == info["bg"] for p in targets), "corner placement collides")
            for p in targets:
                out[p] = info["payload"]
    return out


def _interpret(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    _require(h >= 3 and w >= 3, "canvas too small for the family")
    rim = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors, counts = np.unique(rim, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    border = np.zeros(grid.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    seeds = np.argwhere(border & (grid != bg))
    _require(seeds.shape == (1, 2), "expected a single non-background border seed")
    r, c = map(int, seeds[0])
    flags = (r == 0, c == w - 1, r == h - 1, c == 0)
    _require(sum(flags) == 1, "seed cannot be at a corner")
    rotation = flags.index(True)
    route = int(grid[r, c])
    _require(np.count_nonzero(grid == route) == 1, "seed color must be unique in input")
    palette = set(map(int, np.unique(grid)))
    _require(len(palette) == 5, "expected five distinct visible roles")
    a = np.rot90(grid, rotation).copy()
    choices = []
    for roles in permutations(sorted(palette - {bg, route})):
        try:
            info = _read_roles(a, bg, route, roles)
            _render(info)
        except ValueError:
            continue
        choices.append(info)
    _require(len(choices) == 1, "role assignment is not unique")
    info = choices[0]
    info["rotation"] = rotation
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from the visible input; never consult a seed or examples."""
    info = _interpret(input_grid)
    return np.rot90(_render(info), -info["rotation"]).copy()


def _span_options(width, current, n, sign):
    options = []
    d = n + 1
    for lo in range(1, current + 1):
        for hi in range(max(current, lo + 2 * n), width - 1):
            target = hi + d if sign == 1 else lo - d
            if 2 <= target <= width - 3:
                options.append((lo, hi, target, sign))
    return options


def _sample_route(rng, width, counts, repeat_at=None, repeat_sign=1):
    failed, cache = set(), {}

    def search(index, current):
        if index == len(counts):
            return []
        state = index, current
        if state in failed:
            return None
        signs = (repeat_sign,) if repeat_at is not None and index in (repeat_at, repeat_at + 1) else (-1, 1)
        key = current, counts[index], signs
        if key not in cache:
            cache[key] = [v for sign in signs for v in _span_options(width, current, counts[index], sign)]
        candidates = cache[key]
        for j in rng.permutation(len(candidates)):
            candidate = candidates[int(j)]
            rest = search(index + 1, candidate[2])
            if rest is not None:
                return [candidate] + rest
        failed.add(state)
        return None

    for initial in rng.permutation(np.arange(3, width - 3)):
        initial = int(initial)
        spans = search(0, initial)
        if spans is not None:
            return initial, spans
    return None


def _positions(rng, counts, height):
    # The last bend has space for the guaranteed bent-payload witness.
    gaps = [1] * max(0, len(counts) - 1)
    if gaps:
        gaps[-1] = 3
    minimum = 5 + sum(n + 1 for n in counts) + sum(gaps) + 3 + 1
    slack = height - minimum
    _require(slack >= 0, "infeasible forward size")
    extra = rng.multinomial(slack, np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    turn = 5 + int(extra[0])
    positions = []
    for i, n in enumerate(counts):
        if i:
            turn = positions[-1] + gaps[i - 1] + int(extra[i])
        positions.append(turn + n + 1)
    return positions


def _allowed_payload(a, roles):
    bg, _, bar, mark, _ = roles
    allowed = a == bg
    allowed[[0, -1], :] = False
    allowed[:, [0, -1]] = False
    # A blank cell separates payload from every bar, for readable color roles.
    for p in map(tuple, np.argwhere((a == bar) | (a == mark)).tolist()):
        for r, c in _neighbors(p):
            if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
                allowed[r, c] = False
    return allowed


def _witness_payload(rng, a, roles, initial, bars, allowed):
    f, lo, hi, n, sign = bars[-1]
    t = f - n - 1
    x = hi + n + 1 if sign == 1 else lo - n - 1
    first_turn = bars[0][0] - bars[0][3] - 1
    dock_width = min(initial, a.shape[1] - 1 - initial)
    kmax = min(8, first_turn - 2, a.shape[0] - 2 - t)
    _require(kmax >= 3, "insufficient witness space")
    k = int(rng.integers(3, kmax + 1))
    gap = int(rng.integers(2, k))
    cells = {(t, x - sign), (t, x)}
    cells |= {(t + j, x) for j in range(1, k + 1) if j != gap}
    # A bridge on the outer bank keeps the input connected around a guide gap.
    cells |= {(t + j, x + sign) for j in range(gap - 1, gap + 2)}
    max_out = min(3, dock_width - 2,
                  a.shape[1] - 2 - x if sign == 1 else x - 1)
    cap = int(rng.integers(1, max_out + 1))
    cells |= {(t - 1, x + sign * j) for j in range(-1, cap + 1)}
    teeth = sorted({1, k} | {j for j in range(1, k + 1, 2) if j != gap})
    for j in teeth:
        for side in (-1, 1):
            length = int(rng.integers(1, min(3, dock_width - 1) + 1))
            for distance in range(1, length + 1):
                p = t + j, x + sign * side * distance
                if not (0 <= p[1] < a.shape[1] and allowed[p]):
                    break
                cells.add(p)
    _require(all(allowed[p] for p in cells), "witness overlaps another object")
    return cells


def _grow_payload(rng, a, initial, bars, allowed):
    path, points = _trace(a.shape, initial, bars)
    first_turn = bars[0][0] - bars[0][3] - 1
    dock_width = min(initial, a.shape[1] - 1 - initial)
    anchors = [p for p in points if allowed[p]]
    r, c = anchors[int(rng.integers(len(anchors)))]
    ph = int(rng.integers(3, min(9, first_turn) + 1))
    pw = int(rng.integers(3, min(9, dock_width) + 1))
    r0 = max(1, min(a.shape[0] - 1 - ph, r - int(rng.integers(ph))))
    c0 = max(1, min(a.shape[1] - 1 - pw, c - int(rng.integers(pw))))
    local = allowed.copy()
    mask = np.zeros(a.shape, dtype=bool)
    mask[r0:r0 + ph, c0:c0 + pw] = True
    local &= mask
    if not local[r, c]:
        return set()
    cells = {(r, c)}
    count = int(rng.integers(7, max(8, int(np.count_nonzero(local)) + 1)))
    style = int(rng.integers(3))
    for _ in range(count - 1):
        frontier = sorted({q for p in cells for q in _neighbors(p)
                           if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]
                           and local[q] and q not in cells})
        if not frontier:
            break
        if style != 0:
            degrees = [sum(q in cells for q in _neighbors(p)) for p in frontier]
            best = min(degrees) if style == 1 else max(degrees)
            frontier = [p for p, degree in zip(frontier, degrees) if degree == best]
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _payload_features(info):
    indices = [i for i, p in enumerate(info["points"]) if p in info["fixed"]]
    gaps = [b - a - 1 for a, b in zip(indices, indices[1:]) if b > a + 1]
    fragmented = any(len(_components(group)) > 1 for group in info["groups"])
    initial = info["initial"]
    # Compare actual guide ownership with the seed's undeviated straight axis.
    curved = any((0 if p in info["fixed"] else 1 if p in info["groups"][0] else 2)
                 != (0 if p[1] == initial else 1 if p[1] < initial else 2)
                 for p in info["cells"])
    return gaps, fragmented, curved


def _make_pair(rng, roles, counts, rotation, band, witness=False,
               repeat_at=None, repeat_sign=1):
    minimum = 5 + sum(n + 1 for n in counts) + max(0, len(counts) - 1) + (2 if len(counts) > 1 else 0) + 4
    low, high = max(minimum, band[0]), band[1]
    _require(low <= high, "size band cannot hold route")
    height = high if rng.random() < .18 else int(rng.integers(low, high + 1))
    width = int(rng.integers(max(12, 3 * max(counts) + 7), 31))
    if rng.random() < .18:
        width = height
    widths = list(dict.fromkeys((width, min(30, width + 4), 30)))
    for width in widths:
        plan = _sample_route(rng, width, counts, repeat_at, repeat_sign)
        if plan is not None:
            break
    else:
        raise ValueError("no feasible bar sequence")
    initial, spans = plan
    positions = _positions(rng, counts, height)
    bg, route, bar, mark, payload = roles
    a = np.full((height, width), bg, dtype=np.int8)
    a[0, initial] = route
    bars = []
    for f, n, (lo, hi, target, sign) in zip(positions, counts, spans):
        a[f, lo:hi + 1] = bar
        for j in range(n):
            a[f, lo + 2 * j if sign == 1 else hi - 2 * j] = mark
        bars.append((f, lo, hi, n, sign))
    allowed = _allowed_payload(a, roles)
    for attempt in range(100):
        cells = (_witness_payload(rng, a, roles, initial, bars, allowed)
                 if witness or attempt >= 70 or rng.random() < .23
                 else _grow_payload(rng, a, initial, bars, allowed))
        candidate = a.copy()
        for p in cells:
            candidate[p] = payload
        try:
            info = _read_roles(candidate, bg, route, (bar, mark, payload))
            out = _render(info)
            gaps, fragmented, curved = _payload_features(info)
            if witness and not (gaps and fragmented and curved):
                continue
        except ValueError:
            continue
        return {"input": np.rot90(candidate, -rotation).copy(),
                "output": np.rot90(out, -rotation).copy()}
    raise ValueError("failed to construct a valid payload")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    _require(low <= value <= high, f"{name} out of range")
    return value


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.integers(2, 5))
    shift = int(rng.integers(3))  # held-out orientation, longer sequence, scale
    train_max = int(rng.integers(2, 4)) if shift == 1 else 2 if shift == 2 else 4
    bar_counts = list(map(int, rng.integers(1, train_max + 1, size=num_train)))
    while sum(bar_counts) < maximum or max(bar_counts) < 2:
        i = int(rng.choice([i for i, n in enumerate(bar_counts) if n < train_max]))
        bar_counts[i] += 1
    slots = [(i, j) for i, k in enumerate(bar_counts) for j in range(k)]
    requirements = [{} for _ in bar_counts]
    for n, slot in zip(rng.permutation(np.arange(1, maximum + 1)), rng.permutation(len(slots))):
        i, j = slots[int(slot)]
        requirements[i][j] = int(n)
    sequences = []
    for k, required in zip(bar_counts, requirements):
        ns = list(map(int, rng.integers(1, maximum + 1, size=k)))
        for j, n in required.items():
            ns[j] = n
        limit = 24 if shift == 2 else 30
        while sum(ns) + 2 * k + (10 if k > 1 else 8) > limit:
            candidates = [j for j, n in enumerate(ns) if n > 1 and j not in required]
            _require(bool(candidates), "count obligations exceed size budget")
            ns[int(rng.choice(candidates))] -= 1
        sequences.append(ns)
    repeat_pair = int(rng.choice([i for i, k in enumerate(bar_counts) if k >= 2]))
    repeat_at = int(rng.integers(bar_counts[repeat_pair] - 1))
    repeat_sign = int(rng.choice((-1, 1)))
    witness_pair = int(rng.integers(num_train))
    directions = list(map(int, rng.permutation(4)))
    train_dirs = directions[:3] if shift == 0 else directions
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    recolor = bool(rng.random() < .65)
    puzzle, seen = {"train": [], "test": []}, set()

    def add(split, ns, rotation, witness=False, at=None):
        band = (11, 24) if shift == 2 and split == "train" else (26, 30) if shift == 2 else (11, 30)
        for _ in range(32):
            pair_roles = tuple(map(int, rng.choice(10, 5, replace=False))) if recolor else roles
            try:
                pair = _make_pair(rng, pair_roles, ns, rotation, band, witness, at, repeat_sign)
            except ValueError:
                continue
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("bounded construction exhausted")

    for i, ns in enumerate(sequences):
        add("train", ns, int(rng.choice(train_dirs)), i == witness_pair,
            repeat_at if i == repeat_pair else None)
    infos = [_interpret(p["input"]) for p in puzzle["train"]]
    if len({b[4] for info in infos for b in info["bars"]}) == 1:
        # Reflect a different demonstration, preserving the repeated-side witness.
        i = next(i for i in range(num_train) if i != repeat_pair)
        p, info = puzzle["train"][i], infos[i]
        for name in p:
            canonical = np.rot90(p[name], info["rotation"])
            p[name] = np.rot90(canonical[:, ::-1], -info["rotation"]).copy()
        seen = {_key(p["input"]) for p in puzzle["train"]}
    demonstrated_max = max(len(info["bars"]) for info in infos)
    for _ in range(num_test):
        k = int(rng.integers(demonstrated_max + 1, 5)) if shift == 1 else int(rng.integers(1, 5))
        ns = list(map(int, rng.integers(1, maximum + 1, size=k)))
        while sum(ns) + 2 * k + (10 if k > 1 else 8) > 30:
            j = int(rng.choice([j for j, n in enumerate(ns) if n > 1]))
            ns[j] -= 1
        rotation = directions[3] if shift == 0 else int(rng.choice(directions))
        add("test", ns, rotation)
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    """Family measurements reconstructed solely from a visible input."""
    info = _interpret(input_grid)
    gaps, fragmented, curved = _payload_features(info)
    groups = info["groups"]
    sizes = [len(g) for g in groups]
    boxes = []
    for group in groups:
        rs, cs = zip(*group)
        boxes.append((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1))
    return {"shape": tuple(np.asarray(input_grid).shape), "rotation": info["rotation"],
            "roles": tuple(info[k] for k in ("bg", "route", "bar", "mark", "payload")),
            "bar_count": len(info["bars"]), "marker_counts": tuple(b[3] for b in info["bars"]),
            "signs": tuple(b[4] for b in info["bars"]), "payload_size": len(info["cells"]),
            "fixed_size": len(info["fixed"]), "bank_sizes": sizes, "bank_boxes": boxes,
            "components": [len(_components(g)) for g in groups], "guide_gaps": gaps,
            "fragmented": fragmented, "curved_ownership": curved,
            "approach": info["bars"][0][0] - info["bars"][0][3] - 1}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "invalid split counts")
    features, seen = {"train": [], "test": []}, set()
    relocation_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output changed dimensions")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _interpret(inp)
            expected = np.rot90(_render(info), -info["rotation"])
            _require(np.array_equal(out, expected), "incorrect transformed grid")
            canonical = np.rot90(out, info["rotation"])
            if split == "train":
                relocation_witness |= not np.array_equal(_render(info, keep_in_place=True), canonical)
            _require(np.count_nonzero(out == info["payload"]) == len(info["cells"]), "payload not conserved")
            _require(not np.any(out == info["mark"]), "instruction markers not normalized")
            _require(all(canonical[p] == info["payload"] for p in info["fixed"]), "fixed payload moved")
            for right, group in enumerate(info["groups"]):
                targets = _targets(group, canonical.shape, right)
                _require(all(canonical[p] == info["payload"] for p in targets), "collective bank missing")
                _require(min(r for r, c in targets) == 0, "bank not aligned with entry edge")
                _require((max(c for r, c in targets) == canonical.shape[1] - 1) if right
                         else (min(c for r, c in targets) == 0), "bank not aligned with lateral edge")
            _require(not np.array_equal(inp, out), "identity pair")
            features[split].append(pair_features(inp))
    train, test = features["train"], features["test"]
    counts = {n for f in train for n in f["marker_counts"]}
    _require(len(counts) >= 2 and {n for f in test for n in f["marker_counts"]} <= counts,
             "missing unary counting demonstrations")
    _require({s for f in train for s in f["signs"]} == {-1, 1}, "missing bypass direction")
    _require(any(any(a == b for a, b in zip(f["signs"], f["signs"][1:])) for f in train),
             "missing repeated-side demonstration")
    _require(any(f["guide_gaps"] and f["fragmented"] and f["curved_ownership"] for f in train),
             "missing combined guide/collective-bank demonstration")
    tr_rot = {f["rotation"] for f in train}
    shifted = ({f["rotation"] for f in test} - tr_rot
               or min(f["bar_count"] for f in test) > max(f["bar_count"] for f in train)
               or min(f["shape"][f["rotation"] % 2] for f in test)
                  > max(f["shape"][f["rotation"] % 2] for f in train))
    _require(bool(shifted), "missing visible train/test shift")
    # Mutation checks ensure that both inherited operations affect answers.
    _require(relocation_witness, "payload relocation has no effect")


def self_test():
    """Bounded structural, semantic-witness, mutation, and 42-seed checks."""
    # Manually specified elbow/gap fixture. Its targets and route are literal,
    # rather than obtained from construction or the grid renderer.
    fixture = np.zeros((14, 15), dtype=np.int8)
    fixture[0, 6] = 1
    fixture[8, 3:8] = (3, 2, 2, 2, 2)
    payload_cells = {(5, 8), (5, 9), (5, 10), (6, 8), (6, 9),
                     (7, 8), (7, 9), (7, 10), (8, 10), (9, 8), (9, 9), (9, 10)}
    for p in payload_cells:
        fixture[p] = 4
    expected = np.zeros_like(fixture)
    expected[:7, 6] = expected[6, 6:10] = expected[6:, 9] = 1
    expected[8, 3:8] = 2
    for p in ((6, 8), (6, 9), (7, 9), (9, 9), (0, 0), (2, 0),
              (0, 12), (0, 13), (0, 14), (2, 14), (3, 14), (4, 14)):
        expected[p] = 4
    palette = np.array([7, 4, 0, 9, 2, 1, 3, 8, 5, 6], dtype=np.int8)
    for rotation in range(4):
        for mapping in (np.arange(10, dtype=np.int8), palette):
            assert np.array_equal(solve(mapping[np.rot90(fixture, rotation)]),
                                  mapping[np.rot90(expected, rotation)])
    fixture_info = _interpret(fixture)
    assert fixture_info["groups"][0] == {(7, 8), (9, 8)}
    assert fixture_info["groups"][1] == {(5, 8), (5, 9), (5, 10), (7, 10), (8, 10), (9, 10)}
    assert expected[8, 9] == 1 and fixture[8, 9] == 0
    assert not np.array_equal(_render(fixture_info, separate=True), expected)
    assert not np.array_equal(_render(fixture_info, keep_in_place=True), expected)
    assert _payload_features(fixture_info) == ([1], True, True)
    # Legal geometry beyond the constructor's compact approach/box preferences.
    short = np.zeros((12, 15), dtype=np.int8)
    short[0, 6] = 1
    short[6, 3:8] = (3, 2, 2, 2, 2)
    for r, c in payload_cells:
        short[r - 2, c] = 4
    short_expected = np.zeros_like(short)
    short_expected[:5, 6] = short_expected[4, 6:10] = short_expected[4:, 9] = 1
    short_expected[6, 3:8] = 2
    for p in ((4, 8), (4, 9), (5, 9), (7, 9), (0, 0), (2, 0),
              (0, 12), (0, 13), (0, 14), (2, 14), (3, 14), (4, 14)):
        short_expected[p] = 4
    wide = np.zeros((14, 30), dtype=np.int8)
    wide[0, 14] = 1
    wide[8, 11:16] = (3, 2, 2, 2, 2)
    for r, c in payload_cells:
        wide[r, c + 8] = 4
    wide[5, 19:29] = 4
    wide_expected = np.zeros_like(wide)
    wide_expected[:7, 14] = wide_expected[6, 14:18] = wide_expected[6:, 17] = 1
    wide_expected[8, 11:16] = 2
    wide_expected[0, 17:30] = wide_expected[2:5, 19] = 4
    for p in ((6, 16), (6, 17), (7, 17), (9, 17), (0, 0), (2, 0)):
        wide_expected[p] = 4
    for a, b in ((short, short_expected), (wide, wide_expected)):
        for rotation in range(4):
            assert np.array_equal(solve(np.rot90(a, rotation)), np.rot90(b, rotation))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                        for k in ("input", "output")) for puzzle in puzzles}
    assert len(signatures) == 42
    again = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in again
               for a, b in zip(again[s], puzzles[0][s]) for k in a)
    fs, shapes, normalized = [], set(), set()
    rotations = {"train": set(), "test": set()}
    signs, marker_counts, transitions = set(), set(), set()
    fragmented_mutations = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                inp = pair["input"]
                snapshot = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, snapshot), "solver mutated input"
                info = _interpret(inp)
                f = pair_features(inp)
                fs.append(f)
                shapes.add(f["shape"])
                rotations[split].add(f["rotation"])
                signs.update(f["signs"])
                marker_counts.update(f["marker_counts"])
                transitions.update(zip(f["signs"], f["signs"][1:]))
                norm = np.empty_like(inp)
                for role, color in enumerate(f["roles"]):
                    norm[inp == color] = role
                normalized.add(_key(norm))
                fragmented_mutations += not np.array_equal(_render(info, separate=True), _render(info))
    assert rotations == {"train": {0, 1, 2, 3}, "test": {0, 1, 2, 3}}
    assert marker_counts == {1, 2, 3, 4} and signs == {-1, 1}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert len(shapes) >= 50 and len(normalized) >= 180
    assert {f["bar_count"] for f in fs} == {1, 2, 3, 4}
    assert {f["roles"][0] for f in fs} == set(range(10))
    assert sum(f["curved_ownership"] for f in fs) >= 80
    assert sum(bool(f["guide_gaps"]) for f in fs) >= 42
    assert fragmented_mutations >= 42
    assert max(max(f["shape"]) for f in fs) == 30
    assert len({f["payload_size"] for f in fs}) >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(991, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    p = damaged["test"][0]
    info = _interpret(p["input"])
    canonical = np.rot90(p["output"], info["rotation"]).copy()
    point = next(q for q in info["points"] if canonical[q] == info["route"] and q != info["points"][0])
    canonical[point] = info["bg"]
    p["output"] = np.rot90(canonical, -info["rotation"]).copy()
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("missing guide cell accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
