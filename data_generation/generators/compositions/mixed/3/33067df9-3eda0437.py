"""Sparse rectangle code -> rendered blockers -> inset of the largest empty box.

No parent module, file, seed or puzzle context is consulted by solve().
Coordinates in rectangle tuples are (top, left, height, width).
"""
from __future__ import annotations

from collections import deque
import hashlib
import json

import numpy as np

CANVAS = 26
GAP = 2
HIGHLIGHT = 6
AXES = (1, 2, 3, 4, 6, 8)
SHAPES = tuple((r, c) for r in AXES for c in AXES if r * c >= 4)
HOST_SEEDS = (0, 114364523, 208951527, 265544452, 637138478, 227594590,
              795255779, 634158957, 270869586, 834292663, 691271049,
              324150833, 245992232, 115098730, 511880398, 621350450,
              708210518, 622729156, 595325915, 399829683, 518821843,
              512165571, 277017090, 603645608, 624845072, 990396325,
              359030663, 743537818, 781454121, 351413070, 507716189,
              153416808, 613430553, 222291731, 162935243, 713342494,
              966909064, 656674626, 73125064, 34606898, 775934084, 472620369)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and _integer(num_test), "counts must be integers")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             "require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    _require(h >= 3 and w >= 3 and h % 2 == 1 and w % 2 == 1, "odd sparse lattice required")
    bg = int(a[0, 0])
    _require(bg != HIGHLIGHT and not np.any(a == HIGHLIGHT), "6 is reserved for output")
    _require(bool(np.all(a[::2] == bg) and np.all(a[:, ::2] == bg)), "broken scaffold")
    coarse = a[1::2, 1::2]
    _require(all(n in AXES for n in coarse.shape), "coarse axes must divide 24 with pitch > 2")
    _require(bool(np.any(coarse != bg)), "at least one payload is required")
    return coarse, bg


def _encode(coarse, bg):
    r, c = coarse.shape
    a = np.full((2 * r + 1, 2 * c + 1), bg, dtype=np.int8)
    a[1::2, 1::2] = coarse
    return a


def _runs(row, bg):
    result = []
    x = 0
    while x < len(row):
        color = int(row[x])
        end = x + 1
        while end < len(row) and row[end] == color:
            end += 1
        if color != bg:
            result.append((color, x, end))
        x = end
    return result


def _render(coarse, bg):
    """A: maximal horizontal runs, stacked only with exact span/color equality."""
    rows, cols = coarse.shape
    py, px = 24 // rows, 24 // cols
    out = np.full((CANVAS, CANVAS), bg, dtype=np.int8)
    runs = [_runs(row, bg) for row in coarse]
    for r, row in enumerate(runs):
        previous = set(runs[r - 1]) if r else set()
        for color, left, right in row:
            if (color, left, right) in previous:
                continue
            end = r + 1
            while end < rows and (color, left, right) in runs[end]:
                end += 1
            out[2 + r * py:end * py, 2 + left * px:right * px] = color
    return out


def _bridge_render(coarse, bg, mode="exact"):
    """Cell/bridge check and explicit ablations; never used to generate answers."""
    rows, cols = coarse.shape
    py, px = 24 // rows, 24 // cols
    out = np.full((26, 26), bg, dtype=np.int8)
    runs = [_runs(row, bg) for row in coarse]
    for r, row in enumerate(runs):
        for color, left, right in row:
            out[2 + r * py:(r + 1) * py, 2 + left * px:right * px] = color
        if r == rows - 1 or mode == "none":
            continue
        for color, left, right in row:
            for other, a, b in runs[r + 1]:
                if color != other:
                    continue
                if mode == "exact" and (left, right) != (a, b):
                    continue
                lo, hi = max(left, a), min(right, b)
                if lo < hi:
                    out[(r + 1) * py:2 + (r + 1) * py, 2 + lo * px:hi * px] = color
    return out


def _overmerge_render(coarse, bg):
    """Ablation: incorrectly replace each same-color component by its box."""
    rows, cols = coarse.shape
    py, px = 24 // rows, 24 // cols
    out = np.full((26, 26), bg, dtype=np.int8)
    visited = set()
    for r in range(rows):
        for c in range(cols):
            color = int(coarse[r, c])
            if color == bg or (r, c) in visited:
                continue
            pending, cells = [(r, c)], []
            visited.add((r, c))
            while pending:
                y, x = pending.pop()
                cells.append((y, x))
                for a, b in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= a < rows and 0 <= b < cols and (a, b) not in visited
                            and coarse[a, b] == color):
                        visited.add((a, b))
                        pending.append((a, b))
            top, left = min(y for y, _ in cells), min(x for _, x in cells)
            bottom, right = max(y for y, _ in cells) + 1, max(x for _, x in cells) + 1
            out[2 + top * py:bottom * py, 2 + left * px:right * px] = color
    return out


def _candidates(rendered, bg):
    """Enumerate maximal horizontal empty runs for every eligible row band.

    Any area maximum must use a maximal run, so this includes every winner.
    The two-cell outer margin is outside the search domain [2,24)^2.
    Integer row masks keep the bounded 22x22 search inexpensive.
    """
    masks = [sum(1 << x for x, v in enumerate(row) if v == bg)
             for row in rendered[2:24, 2:24].tolist()]
    found = []
    for top in range(21):
        common = (1 << 22) - 1
        for bottom in range(top, 22):
            common &= masks[bottom]
            if not common:
                break
            height = bottom - top + 1
            if height < 2:
                continue
            remaining = common
            while remaining:
                left = (remaining & -remaining).bit_length() - 1
                shifted = remaining >> left
                width = (shifted ^ (shifted + 1)).bit_length() - 1
                if width >= 2:
                    found.append((height * width, top + 2, left + 2, height, width))
                remaining &= ~(((1 << width) - 1) << left)
    return found


def _winners(rendered, bg):
    candidates = _candidates(rendered, bg)
    area = max((item[0] for item in candidates), default=0)
    return area, tuple(item[1:] for item in candidates if item[0] == area)


def _target(rendered, bg):
    _, winners = _winners(rendered, bg)
    _require(len(winners) == 1, "empty-rectangle maximum must exist and be unique")
    return winners[0]


def _highlight(rendered, target):
    top, left, height, width = target
    out = rendered.copy()
    out[top + 1:top + height - 1, left + 1:left + width - 1] = HIGHLIGHT
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    coarse, bg = _decode(input_grid)
    rendered = _render(coarse, bg)
    return _highlight(rendered, _target(rendered, bg))


def _features(coarse, bg):
    runs = [_runs(row, bg) for row in coarse]
    horizontal = any(b - a >= 2 for row in runs for _, a, b in row)
    exact = any(set(a) & set(b) for a, b in zip(runs, runs[1:]))
    unequal = any(c == d and (a, b) != (x, y) and max(a, x) < min(b, y)
                  for upper, lower in zip(runs, runs[1:])
                  for c, a, b in upper for d, x, y in lower)
    return horizontal, exact, unequal


def _contains(outer, inner):
    y, x, h, w = outer
    a, b, c, d = inner
    return y <= a and x <= b and a + c <= y + h and b + d <= x + w


def _component_size(rendered, bg, target):
    start = target[:2]
    pending, seen = deque([start]), {start}
    while pending:
        y, x = pending.popleft()
        for a, b in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 2 <= a < 24 and 2 <= b < 24 and rendered[a, b] == bg and (a, b) not in seen:
                seen.add((a, b))
                pending.append((a, b))
    return len(seen)


def _evidence(coarse, bg, rendered=None, target=None):
    if rendered is None:
        rendered = _render(coarse, bg)
    if target is None:
        target = _target(rendered, bg)
    horizontal, exact, unequal = _features(coarse, bg)
    area = target[2] * target[3]
    rivals = [item for item in _candidates(rendered, bg) if not _contains(target, item[1:])]
    return {
        "horizontal": horizontal, "exact": exact, "unequal": unequal,
        "exact_causal": exact and not np.array_equal(rendered, _bridge_render(coarse, bg, "none")),
        "unequal_causal": unequal and _winners(_overmerge_render(coarse, bg), bg)[1] != (target,),
        "component_excess": _component_size(rendered, bg, target) > area,
        "area_comparison": any(h > target[2] or w > target[3] for _, _, _, h, w in rivals),
        "rival_area": max((item[0] for item in rivals), default=0),
    }


def _paint(coarse, row, left, right, color, bg):
    coarse[row, left:right] = color
    if left and coarse[row, left - 1] == color:
        coarse[row, left - 1] = bg
    if right < coarse.shape[1] and coarse[row, right] == color:
        coarse[row, right] = bg


def _inject(coarse, bg, colors, kind, rng):
    rows, cols = coarse.shape
    color = int(rng.choice(colors))
    left = int(rng.integers(cols))
    right = int(rng.integers(left + 1, cols + 1))
    if kind == "hole":
        top = int(rng.integers(rows))
        bottom = int(rng.integers(top + 1, rows + 1))
        coarse[top:bottom, left:right] = bg
    elif kind == "exact" and rows >= 2:
        # Ensure a horizontal join as well as a vertical equality witness.
        if cols >= 2:
            left = int(rng.integers(cols - 1))
            right = int(rng.integers(left + 2, cols + 1))
        depth = int(rng.integers(2, rows + 1))
        top = int(rng.integers(rows - depth + 1))
        for row in range(top, top + depth):
            _paint(coarse, row, left, right, color, bg)
    elif kind == "unequal" and rows >= 2 and cols >= 2:
        spans = [(a, b) for a in range(cols) for b in range(a + 1, cols + 1)]
        first = spans[int(rng.integers(len(spans)))]
        others = [s for s in spans if s != first and max(s[0], first[0]) < min(s[1], first[1])]
        second = others[int(rng.integers(len(others)))]
        row = int(rng.integers(rows - 1))
        _paint(coarse, row, *first, color, bg)
        _paint(coarse, row + 1, *second, color, bg)
    elif kind == "return" and rows >= 3:
        top = int(rng.integers(rows - 2))
        bottom = int(rng.integers(top + 2, rows))
        for row in (top, bottom):
            _paint(coarse, row, left, right, color, bg)
        coarse[top + 1, left:right] = bg
    else:
        _paint(coarse, int(rng.integers(rows)), left, right, color, bg)


def _sample_coarse(rng, shape, bg, palette, role):
    active_count = int(rng.integers(1, len(palette) + 1))
    colors = rng.choice(palette, size=active_count, replace=False)
    density = float(rng.choice((0.3, 0.45, 0.6, 0.8, 1.0)))
    coarse = rng.choice(colors, size=shape).astype(np.int8)
    coarse[rng.random(shape) > density] = bg
    kinds = ("horizontal", "exact", "unequal", "hole", "return")
    for _ in range(int(rng.integers(5))):
        _inject(coarse, bg, colors, kinds[int(rng.integers(len(kinds)))], rng)
    if role != "free":
        _inject(coarse, bg, colors, role, rng)
    if rng.integers(2):
        coarse = coarse[::-1].copy()
    if rng.integers(2):
        coarse = coarse[:, ::-1].copy()
    return coarse


def _key(grid):
    return (grid.shape, np.asarray(grid, dtype=np.int8).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bg = int(rng.choice([c for c in range(10) if c != HIGHLIGHT]))
    palette = rng.permutation([c for c in range(10) if c not in (bg, HIGHLIGHT)])[:int(rng.integers(1, 7))]
    # Reserve unseen lattice geometries for tests, independent of their order.
    order = rng.permutation(len(SHAPES))
    held = [SHAPES[int(i)] for i in order[:ns]]
    train_shapes = [s for s in SHAPES if s not in held]
    mandatory = [s for s in train_shapes if s[0] >= 2 and s[1] >= 3]
    plans = []
    for role in ("exact", "unequal"):
        pool = [s for s in mandatory if not plans or s != plans[0][2]]
        plans.append(("train", role, pool[int(rng.integers(len(pool)))]))
    while len(plans) < nt:
        plans.append(("train", "free", train_shapes[int(rng.integers(len(train_shapes)))]))
    plans = [plans[int(i)] for i in rng.permutation(nt)]
    plans += [("test", "free", s) for s in held]
    puzzle, used = {"train": [], "test": []}, set()
    for split, role, shape in plans:
        for _ in range(1600):
            coarse = _sample_coarse(rng, shape, bg, palette, role)
            if not np.any(coarse != bg):
                continue
            inp = _encode(coarse, bg)
            if _key(inp) in used:
                continue
            rendered = _render(coarse, bg)
            _, winners = _winners(rendered, bg)
            if len(winners) != 1 or min(winners[0][2:]) < 3:
                continue
            target = winners[0]
            if role != "free":
                ev = _evidence(coarse, bg, rendered, target)
                if not (ev[role + "_causal"] and ev["horizontal"]):
                    continue
                if role == "unequal" and not (ev["component_excess"] and ev["area_comparison"]):
                    continue
            used.add(_key(inp))
            puzzle[split].append({"input": inp, "output": _highlight(rendered, target)})
            break
        else:
            raise RuntimeError(f"bounded sampling exhausted: {split}/{role}/{shape}")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, backgrounds, training_shapes = set(), set(), set()
    training_evidence = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            coarse, bg = _decode(inp)
            backgrounds.add(bg)
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(out.shape == (26, 26), "output must be 26x26")
            rendered = _render(coarse, bg)
            _require(np.array_equal(rendered, _bridge_render(coarse, bg)), "run/bridge render disagreement")
            target = _target(rendered, bg)
            _require(min(target[2:]) >= 3, "generated winner must have nonempty strict interior")
            _require(np.array_equal(out, _highlight(rendered, target)), "incorrect output")
            mask = out == HIGHLIGHT
            _require(int(mask.sum()) == (target[2] - 2) * (target[3] - 2), "wrong inset area")
            _require(bool(np.all(rendered[mask] == bg)), "highlight overwrites a blocker")
            _require(bool(np.all(out[:2] == bg) and np.all(out[-2:] == bg)
                          and np.all(out[:, :2] == bg) and np.all(out[:, -2:] == bg)), "outer margin changed")
            if split == "train":
                training_shapes.add(coarse.shape)
                training_evidence.append(_evidence(coarse, bg, rendered, target))
            else:
                _require(coarse.shape not in training_shapes, "test must shift lattice geometry")
    _require(len(backgrounds) == 1, "background must be stable within a puzzle")
    _require(len(training_shapes) >= 2, "training must show more than one geometry")
    _require(any(e["exact_causal"] and e["horizontal"] for e in training_evidence), "missing exact-merge evidence")
    _require(any(e["unequal_causal"] and e["horizontal"] and e["component_excess"]
                 and e["area_comparison"] for e in training_evidence), "missing unequal-span/area evidence")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _canonical(coarse, bg):
    roles, values = {bg: 0}, []
    for cell in coarse.flat:
        color = int(cell)
        values.append(roles.setdefault(color, len(roles)))
    return coarse.shape, tuple(values)


def self_test():
    """Bounded checks: 42 host seeds, all count combinations, no saved runs."""
    shapes, layouts, targets, backgrounds, digests, palettes, counts = (set() for _ in range(7))
    exact_cases = unequal_cases = zero_payload = deep_stacks = empty_axes = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _digest(puzzle)
        _require(digest == _digest(generate_puzzle(seed)), "nondeterminism")
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                coarse, bg = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
                shapes.add(coarse.shape)
                layouts.add(_canonical(coarse, bg))
                backgrounds.add(bg)
                palettes.add(len(set(map(int, coarse.flat)) - {bg}))
                rendered = _render(coarse, bg)
                target = _target(rendered, bg)
                targets.add(target)
                counts.add(len({(r, run) for r, row in enumerate(coarse) for run in _runs(row, bg)}))
                horizontal, exact, unequal = _features(coarse, bg)
                exact_cases += int(exact)
                unequal_cases += int(unequal)
                zero_payload += int(bg != 0 and np.any(coarse == 0))
                runs = [set(_runs(row, bg)) for row in coarse]
                deep_stacks += int(any(a & b & c for a, b, c in zip(runs, runs[1:], runs[2:])))
                empty_axes += int(any(np.all(row == bg) for row in coarse)
                                  or any(np.all(col == bg) for col in coarse.T))
                _require(not np.array_equal(rendered, pair["output"]), "B omission is invisible")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(0, nt, ns)
            validate_generated_puzzle(puzzle)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "wrong counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(len(digests) == 42 and len(shapes) >= 20 and len(layouts) >= 140, "insufficient structural variation")
    _require(len(targets) >= 50 and len(backgrounds) >= 8 and len(palettes) >= 4, "insufficient role/target variation")
    _require(exact_cases >= 42 and unequal_cases >= 42 and deep_stacks >= 8
             and zero_payload >= 8 and empty_axes >= 20 and len(counts) >= 12, "insufficient family features")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "coarse_shapes": len(shapes), "normalized_layouts": len(layouts),
            "target_geometries": len(targets), "background_colors": len(backgrounds),
            "payload_palette_sizes": sorted(palettes), "exact_cases": exact_cases,
            "unequal_cases": unequal_cases, "deep_stacks": deep_stacks,
            "zero_payload_cases": zero_payload, "empty_axis_cases": empty_axes}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
