"""Coupon-fed pinwheels: finite, independently funded phases of one labyrinth.

Only NumPy and the standard library are used. No parent code, fixture, seed,
or demonstration is consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import numpy as np


HOST_SEEDS = (0, 376623093, 652594835, 423583775, 978400241, 397052352,
              355548758, 383287452, 539490635, 355290664, 481543982,
              404760395, 647884357, 784400759, 349951220, 644670031,
              439063415, 717408471, 662555486, 606796030, 351730358,
              485671297, 924700711, 162792459, 567352258, 485151678,
              305237550, 423917291, 566334644, 932854935, 752180064,
              757355225, 39346615, 794660899, 365776821, 303409901,
              836554552, 43793856, 501842508, 683263336, 580026209,
              966191145)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, diagonal=False):
    pending = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
             if (y or x) and (diagonal or abs(y) + abs(x) == 1)]
    while pending:
        start = min(pending)
        pending.remove(start)
        group, stack = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                p = y + dy, x + dx
                if p in pending:
                    pending.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _decode(value):
    """Find the hub and color-keyed coupon populations, from input only."""
    a = _grid(value)
    h, w = a.shape
    values, frequencies = np.unique(a, return_counts=True)
    _require(len(values) == 5, "need background and four spoke colors")
    bg = int(values[int(np.argmax(frequencies))])
    _require(int(np.max(frequencies)) > a.size // 2, "background must be the majority")
    foreground = set(map(int, values)) - {bg}
    candidates = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if a[y, x] != bg:
                continue
            colors = tuple(int(a[y + dy, x + dx]) for dy, dx in DIRS)
            if set(colors) != foreground:
                continue
            lengths = []
            seed = np.zeros(a.shape, dtype=bool)
            for color, (dy, dx) in zip(colors, DIRS):
                t = 1
                while 0 <= y+t*dy < h and 0 <= x+t*dx < w and a[y+t*dy, x+t*dx] == color:
                    seed[y+t*dy, x+t*dx] = True
                    t += 1
                lengths.append(t - 1)
            if len(set(lengths)) != 1 or not 2 <= lengths[0] <= 4:
                continue
            length = lengths[0]
            yy, xx = np.indices(a.shape)
            moat = (np.abs(yy-y) <= length+1) & (np.abs(xx-x) <= length+1)
            if np.any((a != bg) & moat & ~seed):
                continue
            candidates.append((y, x, length, colors, seed, moat))
    _require(len(candidates) == 1, "need one unambiguous hollow hub")
    y, x, length, colors, seed, moat = candidates[0]
    coupons = (a != bg) & ~seed
    # Coupons are mutually separated even diagonally, including across colors.
    groups = _components(coupons, diagonal=True)
    _require(all(len({int(a[p]) for p in g}) == 1 for g in groups), "coupons of different colors touch")
    _require(len(groups) == len(_components(coupons)), "diagonal-only coupon connection")
    by_color = [[g for g in groups if int(a[g[0]]) == c] for c in colors]
    counts = tuple(len(gs) for gs in by_color)
    _require(max(counts) <= 5 and max(counts) > 0, "coupon budget outside 0..5")
    return dict(grid=a, bg=bg, center=(y, x), length=length, colors=colors,
                seed=seed, moat=moat, coupons=coupons, counts=counts,
                areas=tuple(sum(map(len, gs)) for gs in by_color),
                sizes=tuple(tuple(sorted(map(len, gs))) for gs in by_color))


def _arm_cells(length, budget, arm, hand=1, jog=True):
    """Procedural infinite-plane path. Stopping at the canvas is never used."""
    dy, dx = DIRS[arm]
    cells = {(t*dy, t*dx) for t in range(1, length+1)}
    y, x = length*dy, length*dx
    dy, dx = hand*dx, -hand*dy
    for k in range(budget):
        for vy, vx, distance in ((dy, dx, 2+4*k),
                                 (hand*dx, -hand*dy, length-2 if jog else 0),
                                 (dy, dx, length)):
            for _ in range(distance):
                y, x = y+vy, x+vx
                cells.add((y, x))
        dy, dx = hand*dx, -hand*dy
    return cells


def _render(d, counts=None, hand=1, jog=True):
    counts = d["counts"] if counts is None else counts
    a = d["grid"]
    out = np.full(a.shape, d["bg"], dtype=np.int8)
    cy, cx = d["center"]
    for arm, (color, n) in enumerate(zip(d["colors"], counts)):
        for y, x in _arm_cells(d["length"], n, arm, hand, jog):
            y, x = cy+y, cx+x
            if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
                _require(out[y, x] in (d["bg"], color), "winding arms collided")
                out[y, x] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_decode(input_grid))


def _analytic_output(d):
    """Validation cross-check: closed-form intervals, no accumulated path walk.

    Stage starts follow four affine formulas. Coordinates are transformed to
    the canonical north phase separately for each color.
    """
    a = d["grid"]
    yy, xx = np.indices(a.shape)
    yy, xx = yy-d["center"][0], xx-d["center"][1]
    r = d["length"]
    out = np.full(a.shape, d["bg"], dtype=np.int8)
    for color, budget in zip(d["colors"], d["counts"]):
        mask = (xx == 0) & (yy >= -r) & (yy <= -1)
        for k in range(budget):
            q, s = divmod(k, 4)
            y, x = ((-r-8*q, -8*q), (-2-8*q, r+8*q+2),
                    (r+8*q+4, 8*q+4), (8*q+6, -r-8*q-6))[s]
            dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[s]
            along = (yy-y)*dy + (xx-x)*dx
            across = (yy-y)*dx - (xx-x)*dy
            run = 2+4*k
            mask |= ((across == 0) & (along >= 0) & (along <= run))
            mask |= ((along == run) & (across >= 0) & (across <= r-2))
            mask |= ((across == r-2) & (along >= run) & (along <= run+r))
        _require(bool(np.all(out[mask] == d["bg"])), "analytic arm collision")
        out[mask] = color
        yy, xx = -xx, yy
    return out


def _seed(shape, center, length, palette):
    a = np.full(shape, palette[0], dtype=np.int8)
    for color, (dy, dx) in zip(palette[1:], DIRS):
        for t in range(1, length+1):
            a[center[0]+t*dy, center[1]+t*dx] = color
    return a


def _grow(rng, shape, area):
    m = np.zeros(shape, dtype=bool)
    frontier = {(int(rng.integers(shape[0])), int(rng.integers(shape[1])))}
    for _ in range(area):
        options = sorted(frontier)
        y, x = options[int(rng.integers(len(options)))]
        frontier.remove((y, x))
        m[y, x] = True
        for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1] and not m[p]:
                frontier.add(p)
    yy, xx = np.where(m)
    return m[yy.min():yy.max()+1, xx.min():xx.max()+1]


def _coupon_shape(rng, area):
    if area == 8 and rng.random() < .3:
        m = np.ones((3, 3), dtype=bool)
        m[1, 1] = False
    elif area <= 6 and rng.random() < .25:
        m = np.ones((1, area), dtype=bool)
    else:
        boxes = [(h, w) for h in range(1, 5) for w in range(1, 5) if h*w >= area]
        m = _grow(rng, boxes[int(rng.integers(len(boxes)))], area)
    return np.rot90(m, int(rng.integers(4))).copy()


def _block_neighbors(blocked, points):
    h, w = blocked.shape
    for y, x in points:
        blocked[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = True


def _place_coupons(rng, a, center, length, colors, counts, equal_area=False):
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    moat = (np.abs(yy-center[0]) <= length+1) & (np.abs(xx-center[1]) <= length+1)
    objects = []
    for color, count in zip(colors, counts):
        if not count:
            continue
        if equal_area:
            # Every positive population has six ink cells, but 1, 2 or 3 objects.
            sizes = [1]*count
            for _ in range(6-count):
                sizes[int(rng.integers(count))] += 1
        else:
            cap = 4 if min(a.shape) < 20 else 9
            sizes = [int(rng.integers(1, cap+1)) if rng.random() < .8 else 1
                     for _ in range(count)]
        objects.extend((color, _coupon_shape(rng, size)) for size in sizes)
    objects.sort(key=lambda item: item[1].size, reverse=True)
    # All random searches are bounded. The fallback below preserves the rule.
    for _ in range(8):
        result, blocked = a.copy(), moat.copy()
        for color, mask in objects:
            mh, mw = mask.shape
            points = np.argwhere(mask)
            found = False
            for flat in rng.permutation((h-mh+1)*(w-mw+1)):
                y, x = divmod(int(flat), w-mw+1)
                coords = points + (y, x)
                if not np.any(blocked[coords[:, 0], coords[:, 1]]):
                    result[coords[:, 0], coords[:, 1]] = color
                    _block_neighbors(blocked, coords)
                    found = True
                    break
            if not found:
                break
        else:
            return result
    result = a.copy()
    if equal_area:
        slots = [(y, x) for y in range(0, h-2, 4) for x in range(0, w-2, 4)
                 if not np.any(moat[y:y+3, x:x+3])]
        _require(len(slots) >= len(objects), "witness packing capacity")
        rng.shuffle(slots)
        for (color, mask), (y, x) in zip(objects, slots):
            m = _grow(rng, (3, 3), int(mask.sum()))
            result[y:y+m.shape[0], x:x+m.shape[1]][m] = color
    else:
        # Isolated singleton coupons remain legitimate family members.
        slots = [(y, x) for y in range(0, h, 2) for x in range(0, w, 2) if not moat[y, x]]
        _require(len(slots) >= len(objects), "coupon packing capacity")
        rng.shuffle(slots)
        for (color, _), point in zip(objects, slots):
            result[point] = color
    return result


def _visible_budgets(shape, center, length, cap):
    cy, cx = center
    def visible(cells):
        return any(0 <= cy+y < shape[0] and 0 <= cx+x < shape[1] for y, x in cells)
    choices = []
    for arm in range(4):
        feasible = [0]
        for n in range(1, cap+1):
            delta = _arm_cells(length, n, arm) - _arm_cells(length, n-1, arm)
            if visible(delta):
                feasible.append(n)
        choices.append(feasible)
    return choices


def _choose_center(rng, shape, length):
    if rng.random() < .25:
        return tuple((n-1)//2 for n in shape)
    if rng.random() < .55:
        return tuple(int(rng.choice([length, n-1-length])) for n in shape)
    return tuple(int(rng.integers(length, n-length)) for n in shape)


def _counts_arg(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _counts_arg(num_train, "num_train", 2, 10)
    nq = _counts_arg(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    common_palette = bool(rng.random() < .25)
    witness_shape = tuple(map(int, rng.integers(25, 31, size=2)))
    edge = int(rng.integers(4))
    witness_center = [(n-1)//2 for n in witness_shape]
    axis = 0 if edge in (0, 2) else 1
    witness_center[axis] = 9 if edge in (0, 3) else witness_shape[axis]-10
    witness_center = tuple(witness_center)
    puzzle, used = {"train": [], "test": []}, set()
    for index in range(nt+nq):
        is_train = index < nt
        equal_area = index == 0
        palette = shared_palette if common_palette or index == 0 else tuple(map(int, rng.choice(10, 5, replace=False)))
        if index < 2:
            length = 2+index
            shape, center = witness_shape, witness_center
            if index == 0:
                counts = tuple(map(int, rng.permutation([0, 1, 2, 3])))
            else:
                # The four-stage arm points toward the nearest edge. The other
                # arms expose full earlier stages; this one teaches clipping.
                counts_list = list(map(int, rng.permutation([1, 2, 3])))
                counts_list.insert(edge, 4)
                counts = tuple(counts_list)
        else:
            length = int(rng.choice([2, 3]))
            if is_train:
                mode = "familiar"
                shape = tuple(map(int, rng.integers(20, 31, size=2)))
                cap = 4
            else:
                mode = str(rng.choice(["scale", "budget", "crop"]))
                cap = 5 if mode == "budget" else 4
                if mode == "scale":
                    length = 4
                    shape = tuple(map(int, rng.integers(21, 31, size=2)))
                elif mode == "budget":
                    shape = tuple(map(int, rng.integers(25, 31, size=2)))
                else:
                    shape = int(rng.integers(13, 20)), int(rng.integers(25, 31))
                    if rng.random() < .5:
                        shape = shape[::-1]
            center = (_choose_center(rng, shape, length) if mode != "budget" else
                      tuple((n-1)//2 for n in shape))
            choices = _visible_budgets(shape, center, length, cap)
            counts = [int(rng.choice(options)) for options in choices]
            if mode == "budget":
                arm = int(rng.choice([i for i, options in enumerate(choices) if 5 in options]))
                counts[arm] = 5
            if len(set(counts)) == 1 or max(counts) < 2:
                arm = int(rng.choice([i for i, options in enumerate(choices) if max(options) >= 2]))
                counts[arm] = max(choices[arm])
                counts[(arm+1) % 4] = 0
            counts = tuple(counts)
        a = _seed(shape, center, length, palette)
        a = _place_coupons(rng, a, center, length, palette[1:], counts, equal_area)
        # Color relabeling is a finite uniqueness fallback; no answer state is
        # encoded in color identities. There are more variants than pairs.
        if _key(a) in used:
            original = a.copy()
            for shift in range(10):
                for swap in range(10):
                    mapping = (np.arange(10)+shift) % 10
                    mapping[0], mapping[swap] = mapping[swap], mapping[0]
                    candidate = mapping[original].astype(np.int8)
                    if _key(candidate) not in used:
                        a = candidate
                        break
                else:
                    continue
                break
            else:
                raise AssertionError("uniqueness fallback exhausted")
        used.add(_key(a))
        puzzle["train" if is_train else "test"].append({"input": a, "output": solve(a)})
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad pair counts")
    used, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "canvas shape changed")
            _require(_key(a) not in used, "duplicate input")
            used.add(_key(a))
            d = _decode(a)
            _require(np.array_equal(b, _analytic_output(d)), "incorrect winding or coupon removal")
            _require(bool(np.all(b[d["seed"]] == a[d["seed"]])), "seed changed")
            _require(b[d["center"]] == d["bg"], "hub hole filled")
            _require(not np.array_equal(a, b), "identity pair")
            feasible = _visible_budgets(a.shape, d["center"], d["length"], 5)
            _require(all(n in options for n, options in zip(d["counts"], feasible)), "invisible final growth stage")
            (training if split == "train" else tests).append(d)
    _require({d["length"] for d in training} == {2, 3}, "training must teach lengths two and three")
    _require(set(n for d in training for n in d["counts"]) == set(range(5)), "need budgets zero through four in training")
    _require(any(any(a > 0 and a == b and n != m for a, b, n, m in
                         zip(d["areas"], d["areas"][1:]+d["areas"][:1],
                             d["counts"], d["counts"][1:]+d["counts"][:1])) for d in training),
             "missing equal-area/different-count witness")
    for alternative in ("equal_budgets", "counterclockwise", "no_jog", "ink_area"):
        differs = False
        for d in training:
            if alternative == "equal_budgets":
                pred = _render(d, (max(d["counts"]),)*4)
            elif alternative == "counterclockwise":
                pred = _render(d, hand=-1)
            elif alternative == "no_jog":
                # This alternative can collide; a color mismatch is sufficient
                # evidence, so compare each phase's cell set without painting.
                cy, cx = d["center"]
                for i, n in enumerate(d["counts"]):
                    delta = (_arm_cells(d["length"], n, i) ^
                             _arm_cells(d["length"], n, i, jog=False))
                    differs |= any(0 <= cy+y < d["grid"].shape[0] and
                                   0 <= cx+x < d["grid"].shape[1] for y, x in delta)
                continue
            else:
                pred = _render(d, tuple(min(5, x) for x in d["areas"]))
            differs |= not np.array_equal(pred, _analytic_output(d))
        _require(differs, "training does not distinguish " + alternative)
    _require(any(d["length"] == 4 or max(d["counts"]) == 5 or min(d["grid"].shape) < 20
                 for d in tests), "missing structural train/test shift")


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def variation_metrics(puzzles):
    """Small descriptive report reconstructed from grids, not sampling modes."""
    shapes, centers, populations, palettes, coupon_shapes = set(), set(), set(), set(), set()
    counts, lengths, shifts = Counter(), Counter(), Counter()
    clipping = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                shapes.add(d["grid"].shape)
                centers.add((d["grid"].shape, d["center"], d["length"]))
                populations.add(d["counts"])
                palettes.add((d["bg"],)+d["colors"])
                counts.update(d["counts"])
                lengths[(split, d["length"])] += 1
                cy, cx = d["center"]
                for i, n in enumerate(d["counts"]):
                    cells = _arm_cells(d["length"], n, i)
                    clipping += any(not (0 <= cy+y < d["grid"].shape[0] and 0 <= cx+x < d["grid"].shape[1]) for y, x in cells)
                for g in _components(d["coupons"]):
                    points = np.array(g)
                    lo, hi = points.min(axis=0), points.max(axis=0)+1
                    m = np.zeros(tuple(hi-lo), dtype=np.int8)
                    for point in points-lo:
                        m[tuple(point)] = 1
                    coupon_shapes.add(_key(m))
                if split == "test":
                    shifts["new_length" if d["length"] == 4 else "new_budget" if max(d["counts"]) == 5 else "narrow_canvas"] += 1
    return {"puzzles": len(puzzles), "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "distinct_shapes": len(shapes), "distinct_seed_geometries": len(centers),
            "distinct_budget_vectors": len(populations), "distinct_palettes": len(palettes),
            "distinct_coupon_masks": len(coupon_shapes), "clipped_arms": int(clipping),
            "budgets": dict(sorted(counts.items())),
            "lengths_by_split": {f"{s}/{n}": v for (s, n), v in sorted(lengths.items())},
            "test_shifts": dict(shifts)}


def self_test():
    # All six budget values, every scale/phase and severe boundary placement.
    # Analytic intervals are checked against the procedural path at each cell.
    for length in (2, 3, 4):
        for n in range(6):
            for shape, center in (((30, 30), (14, 15)), ((13, 30), (length, 29-length)),
                                   ((30, 17), (29-length, length))):
                d = dict(grid=np.zeros(shape, dtype=np.int8), bg=0, center=center,
                         length=length, colors=(1, 2, 3, 4), counts=(n,)*4)
                b = _render(d)
                _require(np.array_equal(b, _analytic_output(d)), "analytic/path mismatch")
                rotated = dict(d, grid=np.zeros(shape[::-1], dtype=np.int8),
                               center=(shape[1]-1-center[1], center[0]),
                               colors=(2, 3, 4, 1))
                _require(np.array_equal(np.rot90(b), _render(rotated)), "rotation mismatch")
                big = dict(d, grid=np.zeros((80, 80), dtype=np.int8), center=(40, 40))
                top, left = 40-center[0], 40-center[1]
                _require(np.array_equal(b, _analytic_output(big)[top:top+shape[0], left:left+shape[1]]), "clipping/re-entry mismatch")
    # Hand-computed north-arm stage endpoints, protecting the recurrence itself.
    _require(_arm_cells(2, 1, 0) == {(-1, 0)} | {(-2, x) for x in range(5)}, "first stage fixture")
    _require((1, 5) in _arm_cells(3, 2, 0) and (7, 4) in _arm_cells(3, 2, 0), "jog fixture")
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    _require(len({_signature(p) for p in puzzles}) == 42, "duplicate puzzles across seeds")
    m = variation_metrics(puzzles)
    _require(m["distinct_shapes"] >= 50 and m["distinct_seed_geometries"] >= 100, "geometry collapsed")
    _require(m["distinct_budget_vectors"] >= 80 and m["distinct_coupon_masks"] >= 40, "object/growth structure collapsed")
    _require(m["distinct_palettes"] >= 80 and set(m["budgets"]) == set(range(6)), "palette/budget collapsed")
    _require(set(m["test_shifts"]) == {"new_length", "new_budget", "narrow_canvas"}, "shift modes missing")
    for seed in (0, 376623093, 966191145):
        _require(_signature(generate_puzzle(seed)) == _signature(generate_puzzle(seed)), "nondeterminism")
    # Input counterfactuals: quantities and geometry, rather than generator
    # metadata, determine labels. Repositioning/reshaping coupons is irrelevant;
    # consuming one fewer coupon changes only its matching phase.
    for pair in puzzles[0]["train"][:2]:
        d = _decode(pair["input"])
        coupon_rng = np.random.default_rng(91827)
        base = _seed(d["grid"].shape, d["center"], d["length"], (d["bg"],)+d["colors"])
        moved = _place_coupons(coupon_rng, base, d["center"], d["length"], d["colors"], d["counts"])
        _require(np.array_equal(solve(moved), pair["output"]), "coupon layout/shape leaked into answer")
        mapping = np.array([9, 3, 7, 0, 5, 8, 2, 4, 1, 6], dtype=np.int8)
        _require(np.array_equal(solve(mapping[d["grid"]]), mapping[pair["output"]]), "literal colors leaked into rule")
        for arm, n in enumerate(d["counts"]):
            if n == 0:
                continue
            color = d["colors"][arm]
            component = _components(d["coupons"] & (d["grid"] == color))[0]
            fewer = d["grid"].copy()
            for point in component:
                fewer[point] = d["bg"]
            changed = solve(fewer)
            lost = pair["output"] != changed
            _require(bool(np.any(lost)) and bool(np.all(pair["output"][lost] == color))
                     and bool(np.all(changed[lost] == d["bg"])), "one-coupon ablation failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1729, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # Labels are checked using the interval implementation, not solve().
    bad = deepcopy(puzzles[0])
    bad["test"][0]["output"].flat[0] = (int(bad["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupted output")
    return m


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
