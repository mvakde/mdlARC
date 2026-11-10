"""Eye-guided locks: geometric fitting and color-dependent registration interact.

Only NumPy and the standard library are used. The solver reads only the grid.
Construction paints a known pose; decoding and validation recover it from cells.
"""
from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("counts must be integers with 2..10 train, 1..4 test, and train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if not (a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
            and np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be an integer 2D array of colors 0..9, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(cells):
    left, result = set(cells), []
    while left:
        start = min(left)
        left.remove(start)
        part, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    part.add(p)
                    todo.append(p)
        result.append(frozenset(part))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, delta):
    return frozenset((r + delta[0], c + delta[1]) for r, c in cells)


def _op(p, op):
    r, c = p
    if op >= 4:
        c = -c
    for _ in range(op % 4):
        r, c = c, -r
    return r, c


def _transform(cells, op):
    return frozenset(_op(p, op) for p in cells)


def _normal(cells):
    r, _, c, _ = _bbox(cells)
    return tuple(sorted((y - r, x - c) for y, x in cells))


def _canonical(cells):
    return min(_normal(_transform(cells, op)) for op in range(8))


def _holes(cells):
    """Enclosed cells of the union mask, using four-neighbor background flooding."""
    r0, r1, c0, c1 = _bbox(cells)
    r0, r1, c0, c1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1
    blank = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - set(cells)
    reached, todo = {(r0, c0)}, [(r0, c0)]
    while todo:
        r, c = todo.pop()
        for dr, dc in DIR4:
            p = r + dr, c + dc
            if p in blank and p not in reached:
                reached.add(p)
                todo.append(p)
    return blank - reached


def _decode(inp):
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(counts.max() > inp.size / 2, "input background must occupy more than half the grid")
    h, w = inp.shape
    anchors, mono, keys = {}, [], []
    cells = {tuple(map(int, p)) for p in np.argwhere(inp != bg)}
    for part in _components(cells):
        colors = sorted({int(inp[p]) for p in part})
        p = min(part)
        edge = p[0] in (0, h - 1) or p[1] in (0, w - 1)
        if len(part) == 1 and edge:
            anchors.setdefault(colors[0], set()).add(p)
        elif len(colors) == 1:
            _require(len(part) >= 2, "interior singleton is not a fragment")
            mono.append({"cells": part, "color": colors[0]})
        else:
            _require(len(colors) == 2, "a key must have two colors")
            regions = tuple(frozenset(p for p in part if int(inp[p]) == c) for c in colors)
            _require(all(len(_components(s)) == 1 for s in regions), "key regions must each be connected")
            holes = _holes(part)
            _require(len(holes) == 1, "a key must have exactly one enclosed blank eye")
            eye = next(iter(holes))
            surround = {(eye[0] + dr, eye[1] + dc) for dr, dc in DIR4}
            _require(surround <= part, "eye must be surrounded on all four sides")
            surrounds = {int(inp[p]) for p in surround}
            _require(len(surrounds) == 1, "the four eye neighbors must have one color")
            hi = colors.index(next(iter(surrounds)))
            keys.append({"cells": part, "regions": regions, "colors": tuple(colors),
                         "eye": eye, "handle": hi, "plug": 1 - hi})
    _require(keys and len(mono) == 2 * len(keys), "need two fixed fragments per key")
    guides = {}
    for color, ps in anchors.items():
        horizontal = len({r for r, c in ps}) == 1 and all(c in (0, w - 1) for r, c in ps)
        vertical = len({c for r, c in ps}) == 1 and all(r in (0, h - 1) for r, c in ps)
        _require(horizontal != vertical, "corner or inconsistent guide markers")
        axis = 0 if horizontal else 1
        coord = next(iter(ps))[axis]
        _require(0 < coord < inp.shape[axis] - 1, "guides cannot run along the outer edge")
        guides[color] = (axis, coord)
    _require(set(guides) == {c for key in keys for c in key["colors"]},
             "each key color needs a guide, with no unrelated guide identities")
    targets = []
    for i, a in enumerate(mono):
        for j in range(i + 1, len(mono)):
            b = mono[j]
            if a["color"] != b["color"]:
                continue
            union = a["cells"] | b["cells"]
            r0, r1, c0, c1 = _bbox(union)
            rectangle = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
            if (rectangle & cells) != union:
                continue
            slot = rectangle - union
            if not slot or len(_components(slot)) != 1:
                continue
            targets.append({"parts": frozenset((i, j)), "rect": rectangle, "slot": slot,
                            "bbox": (r0, r1, c0, c1)})
    _require(targets, "no rectangular locks")
    fixed = {p: item["color"] for item in mono for p in item["cells"]}
    fixed.update({p: c for c, ps in anchors.items() for p in ps})
    return {"bg": bg, "keys": keys, "mono": mono, "guides": guides,
            "anchors": anchors, "targets": targets, "fixed": fixed}


def _placements(inp, data, use_guides=True, policy=(0, 1), expected=None):
    result = []
    h, w = inp.shape
    for ki, key in enumerate(data["keys"]):
        choices, seen = [], set()
        for target in data["targets"]:
            sr, _, sc, _ = _bbox(target["slot"])
            for op in range(8):
                regions = tuple(_transform(s, op) for s in key["regions"])
                pr, _, pc, _ = _bbox(regions[key["plug"]])
                delta = sr - pr, sc - pc
                regions = tuple(_shift(s, delta) for s in regions)
                if regions[key["plug"]] != target["slot"]:
                    continue
                handle = regions[key["handle"]]
                if handle & target["rect"]:
                    continue
                eye0 = _op(key["eye"], op)
                eye = eye0[0] + delta[0], eye0[1] + delta[1]
                footprint = regions[0] | regions[1]
                if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint | {eye}):
                    continue
                if (footprint | {eye}) & data["fixed"].keys():
                    continue
                swap = policy[int(op >= 4)]
                colors = key["colors"][::-1] if swap else key["colors"]
                eye_color = colors[key["handle"]]
                if use_guides:
                    axis, coord = data["guides"][eye_color]
                    if eye[axis] != coord:
                        continue
                paint = {p: colors[i] for i, region in enumerate(regions) for p in region}
                if expected is not None and (int(expected[eye]) != data["bg"] or
                        any(int(expected[p]) != c for p, c in paint.items())):
                    continue
                signature = (target["parts"], eye, tuple(sorted(paint.items())))
                if signature in seen:
                    continue
                seen.add(signature)
                choices.append({"key": ki, "target": target, "paint": paint, "eye": eye,
                                "eye_color": eye_color, "op": op, "regions": regions})
        result.append(choices)
    return result


def _solutions(inp, use_guides=True, policy=(0, 1), expected=None, limit=2):
    """Deduplicate by complete colored answer, never choose a reflection by order."""
    data = _decode(inp)
    choices = _placements(inp, data, use_guides, policy, expected)
    order = sorted(range(len(choices)), key=lambda k: len(choices[k]))
    results, seen = [], set()

    def search(depth, parts, rectangles, reserved, moves):
        if len(results) >= limit:
            return
        if depth == len(order):
            if len(parts) != len(data["mono"]):
                return
            out = np.full_like(inp, data["bg"])
            for p, color in data["fixed"].items():
                out[p] = color
            for move in moves:
                for p, color in move["paint"].items():
                    out[p] = color
            sig = out.tobytes()
            if sig not in seen and (expected is None or np.array_equal(out, expected)):
                seen.add(sig)
                results.append((out, tuple(moves)))
            return
        for move in choices[order[depth]]:
            target = move["target"]
            footprint = set(move["paint"]) | {move["eye"]}
            if parts & target["parts"] or footprint & reserved:
                continue
            if any(target["rect"] & rect for rect in rectangles):
                continue
            search(depth + 1, parts | target["parts"], rectangles + (target["rect"],),
                   reserved | footprint, moves + (move,))

    search(0, frozenset(), (), set(data["fixed"]), ())
    return results, data


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    solutions, _ = _solutions(inp)
    if len(solutions) != 1:
        raise ValueError("input must have exactly one complete guided colored assembly")
    return solutions[0][0]


def _shape(rng, small=False):
    """Symmetric cuts and asymmetric punctured handles, composed procedurally."""
    for _ in range(120):
        kind = int(rng.integers(3))
        w = int(rng.integers(3, 6 if small else 8))
        if kind == 0:
            thickness = int(rng.integers(1, 3))
            top, bottom = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            h = top + thickness + bottom
            slot = {(r, c) for r in range(top, top + thickness) for c in range(w)}
        elif kind == 1:
            h = int(rng.integers(4, 7 if small else 8))
            levels = list(map(int, rng.integers(1, h - 1, size=(w + 1) // 2)))
            levels = levels + levels[:w // 2][::-1]
            slot = {(levels[c], c) for c in range(w)}
            # Mirror the vertical connectors as well as the horizontal trace.
            for c in range(w - 1):
                for r in range(min(levels[c:c + 2]), max(levels[c:c + 2]) + 1):
                    slot.add((r, c))
                    slot.add((r, w - 1 - c))
        else:
            h = int(rng.choice((5, 7)))
            mid = h // 2
            radii = list(map(int, rng.integers(0, 2, size=w)))
            slot = {(r, c) for c in range(w) for r in range(mid - radii[c], mid + radii[c] + 1)}
        rect = {(r, c) for r in range(h) for c in range(w)}
        target = rect - slot
        parts = _components(target)
        if len(parts) != 2 or min(map(len, parts)) < 2 or _bbox(target) != (0, h - 1, 0, w - 1):
            continue
        endpoint = sorted(p for p in slot if p[1] == w - 1)
        er, ec = endpoint[int(rng.integers(len(endpoint)))]
        neck = int(rng.integers(0, 3))
        eye = er + int(rng.integers(-1, 2)), w + neck + 1
        handle = {(eye[0] + dr, eye[1] + dc) for dr in range(-1, 2) for dc in range(-1, 2)
                  if dr or dc}
        handle.update((er, w + k) for k in range(neck))
        desired = int(rng.integers(max(9, len(handle) + 1), min(25, max(15, len(slot) + 6)) + 1))
        for _ in range(desired - len(handle)):
            frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIR4
                               if -3 <= r + dr < h + 3 and w <= c + dc < w + neck + 6}
                              - handle - {eye})
            legal = [p for p in frontier if _holes(handle | slot | {p}) == {eye}]
            if not legal:
                break
            handle.add(legal[int(rng.integers(len(legal)))])
        if len(_components(handle)) != 1 or _holes(handle | slot) != {eye}:
            continue
        if len({_normal(_transform(handle | slot, op)) for op in range(8)}) < 8:
            continue
        # Random base orientation changes the inherited slot/handle vocabulary.
        op = int(rng.integers(8))
        return tuple(_transform(s, op) for s in (target, slot, handle)), _op(eye, op)
    raise RuntimeError("bounded shape construction exhausted")


def _pack(rng, footprints, h, w, margin):
    blocked, origins = set(), [None] * len(footprints)
    order = sorted(range(len(footprints)), key=lambda i: -len(footprints[i]) + rng.random())
    for i in order:
        cells = footprints[i]
        r0, r1, c0, c1 = _bbox(cells)
        options = [(r, c) for r in range(margin - r0, h - margin - r1)
                   for c in range(margin - c0, w - margin - c1)]
        rng.shuffle(options)
        for delta in options:
            placed = _shift(cells, delta)
            if placed & blocked:
                continue
            origins[i] = delta
            blocked.update((r + dr, c + dc) for r, c in placed for dr, dc in ((0, 0),) + DIR4)
            break
        else:
            return None
    return origins


def _anchor_options(inp, bg, axis, coord):
    h, w = inp.shape
    ps = [(coord, 0), (coord, w - 1)] if axis == 0 else [(0, coord), (h - 1, coord)]
    return [p for p in ps if inp[p] == bg and all(
        not (0 <= p[0] + dr < h and 0 <= p[1] + dc < w) or inp[p[0] + dr, p[1] + dc] == bg
        for dr, dc in DIR4)]


def _scene(rng, n, parities, axes, palette):
    # Pairwise palette permutation; the same fixed interpretation applies to all.
    bg = palette[0]
    target_colors = palette[1:1 + n]
    key_colors = [palette[1 + n + 2 * i:3 + n + 2 * i] for i in range(n)]
    pieces, systems = [], []
    shared_targets = bool(rng.random() < .45)
    for i in range(n):
        shape, eye = _shape(rng, small=n >= 3)
        target, plug, handle = shape
        # Source and destination poses are independent. Parity is their difference.
        incoming = int(rng.integers(8))
        outgoing = int(rng.integers(4)) + 4 * (int(incoming >= 4) ^ parities[i])
        key = tuple(_transform(s, incoming) for s in (plug, handle))
        final = tuple(_transform(s, outgoing) for s in (target, plug, handle))
        final_eye = _op(eye, outgoing)
        pieces.extend((final[0] | final[1] | final[2] | {final_eye}, key[0] | key[1]))
        systems.append((final, final_eye, key, key_colors[i], target_colors[0 if shared_targets else i]))
    origins = None
    for attempt in range(14):
        if attempt >= 10:
            h = w = 30
        else:
            low = 12 if n == 1 else 20 if n == 2 else 26
            h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        # One-cell margin allows boundary-adjacent carriers; anchor placement checks isolation.
        origins = _pack(rng, pieces, h, w, int(rng.choice((1, 2))))
        if origins is not None:
            break
    if origins is None:
        return None
    inp = np.full((h, w), bg, np.int8)
    expected = inp.copy()
    eyes, true_colors = [], []
    for i, (final, eye, key, colors, tc) in enumerate(systems):
        for p in _shift(final[0], origins[2 * i]):
            inp[p] = expected[p] = tc
        for region, color in zip(key, colors):
            for p in _shift(region, origins[2 * i + 1]):
                inp[p] = color
        placed_colors = colors[::-1] if parities[i] else colors
        for region, color in zip(final[1:], placed_colors):
            for p in _shift(region, origins[2 * i]):
                expected[p] = color
        eyes.append(tuple(eye[k] + origins[2 * i][k] for k in range(2)))
        true_colors.append(placed_colors[1])
    for i, colors in enumerate(key_colors):
        true = true_colors[i]
        for color in colors:
            axis = axes[i] if color == true else int(rng.integers(2))
            coord = eyes[i][axis] if color == true else int(rng.integers(1, inp.shape[axis] - 1))
            available = _anchor_options(inp, bg, axis, coord)
            if not available:
                return None
            rng.shuffle(available)
            used = available if len(available) == 2 and rng.random() < .3 else available[:1]
            for p in used:
                inp[p] = expected[p] = color
    if np.mean(inp == bg) <= .5:
        return None
    try:
        solutions, _ = _solutions(inp)
        blind, _ = _solutions(inp, use_guides=False)
        if len(solutions) != 1 or not np.array_equal(solutions[0][0], expected) or len(blind) < 2:
            return None
        actual = sorted(int(m["op"] >= 4) for m in solutions[0][1])
        if actual != sorted(parities):
            return None
    except AssertionError:
        return None
    return {"input": inp, "output": expected}


def _signature(grid):
    return grid.shape, grid.tobytes()


def _plain_projection(inp, data):
    """Diagnostic omission of slot fitting/parity; never used to solve a member."""
    out = np.full_like(inp, data["bg"])
    for p, color in data["fixed"].items():
        out[p] = color
    for key in data["keys"]:
        axis, coord = data["guides"][key["colors"][key["handle"]]]
        delta = [0, 0]
        delta[axis] = coord - key["eye"][axis]
        for r, c in key["cells"]:
            tr, tc = r + delta[0], c + delta[1]
            if 0 <= tr < inp.shape[0] and 0 <= tc < inp.shape[1]:
                out[tr, tc] = inp[r, c]
    return out


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .3)
    palette = tuple(map(int, rng.permutation(10)))
    cap = 1 if a == 2 or rng.random() < .6 else 2
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(a + b):
        test = index >= a
        n = cap + 1 if test else 1 if index < 2 else cap if index == 2 else int(rng.integers(1, cap + 1))
        parities = list(map(int, rng.integers(2, size=n)))
        axes = list(map(int, rng.integers(2, size=n)))
        if index < 2:
            parities[0], axes[0] = index, index
        for _ in range(180):
            colors = palette if stable else tuple(map(int, rng.permutation(10)))
            pair = _scene(rng, n, parities, axes, colors)
            if pair is not None and _signature(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("bounded scene construction exhausted")
        seen.add(_signature(pair["input"]))
        puzzle["test" if test else "train"].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_parities, train_axes = set(), set(), set()
    train_counts, test_counts = [], []
    projection_disproved = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must stay the same")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            solutions, data = _solutions(inp)
            _require(len(solutions) == 1 and np.array_equal(solutions[0][0], out), "wrong or ambiguous answer")
            blind, _ = _solutions(inp, use_guides=False)
            _require(len(blind) >= 2, "guides must decide between distinct geometric/color fits")
            _require(not np.array_equal(inp, out), "unchanged scene")
            _require(all(int(out[p]) == c for p, c in data["fixed"].items()), "fixed cells moved")
            _require(all(int(out[m["eye"]]) == data["bg"] for m in solutions[0][1]), "eye must stay blank")
            _require(set(np.unique(inp)) == set(np.unique(out)), "palette must be preserved")
            (train_counts if split == "train" else test_counts).append(len(data["keys"]))
            if split == "train" and not np.array_equal(_plain_projection(inp, data), out):
                projection_disproved = True
            for move in solutions[0][1]:
                key = data["keys"][move["key"]]
                _require(len(move["paint"]) == len(key["cells"]), "rigid key lost cells")
                axis, coord = data["guides"][move["eye_color"]]
                _require(move["eye"][axis] == coord, "eye misses its resulting-color guide")
                if split == "train":
                    train_parities.add(int(move["op"] >= 4))
                    train_axes.add(axis)
    _require(train_parities == {0, 1} and train_axes == {0, 1}, "teach both parities and guide axes")
    _require(projection_disproved, "training must disprove projection without slot fitting")
    _require(min(test_counts) > max(train_counts), "tests must combine more simultaneous locks")
    for policy in ((0, 0), (1, 0), (1, 1)):
        compatible = all(bool(_solutions(_grid(p["input"]), policy=policy,
                                         expected=_grid(p["output"]), limit=1)[0])
                         for p in puzzle["train"])
        _require(not compatible, "training must distinguish alternative parity-to-swap rules")


def self_test():
    dimensions, fingerprints, plugs, handles, relations, operations = set(), set(), set(), set(), set(), set()
    axes, ends, object_counts, backgrounds = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_signature(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                inp = pair["input"]
                solutions, data = _solutions(inp)
                dimensions.add(inp.shape)
                backgrounds.add(data["bg"])
                object_counts.add(len(data["keys"]))
                for color, (axis, coord) in data["guides"].items():
                    axes.add(axis)
                    ends.add((axis, len(data["anchors"][color])))
                for move in solutions[0][1]:
                    key = data["keys"][move["key"]]
                    plug, handle = key["regions"][key["plug"]], key["regions"][key["handle"]]
                    plugs.add(_canonical(plug))
                    handles.add(_canonical(handle))
                    relations.add(int(np.sign(len(plug) - len(handle))))
                    operations.add(move["op"])
                _require(np.array_equal(solve(inp), pair["output"]), "input-only solving")
    _require(len(fingerprints) == 42 and len(dimensions) >= 45, "scene variation across 42 seeds")
    _require(len(plugs) >= 25 and len(handles) >= 70, "procedural shape variation")
    _require(relations == {-1, 0, 1}, "vary relative region areas")
    _require(operations == set(range(8)), "all rigid symmetries")
    _require(axes == {0, 1} and ends == {(0, 1), (0, 2), (1, 1), (1, 2)}, "guide variations")
    _require(object_counts == {1, 2, 3} and backgrounds == set(range(10)), "count/palette variations")
    # Change guides alone, retaining every key and lock pixel, and require a
    # different key placement. This checks a causal role, beyond marker retention.
    fixture0 = generate_puzzle(0)
    pair = next(p for p in fixture0["train"] if len(_decode(p["input"])["keys"]) == 1
                and _solutions(p["input"])[0][0][1][0]["op"] < 4)
    inp = pair["input"]
    original, data = _solutions(inp)
    original_paint = original[0][1][0]["paint"]
    changed_pose = False
    for alternative in _placements(inp, data, use_guides=False)[0]:
        if alternative["paint"] == original_paint:
            continue
        true_color = alternative["eye_color"]
        for off in range(1, max(inp.shape) - 1):
            modified = inp.copy()
            for ps in data["anchors"].values():
                for p in ps:
                    modified[p] = data["bg"]
            legal = True
            for color, ps in data["anchors"].items():
                axis, _ = data["guides"][color]
                coord = alternative["eye"][axis] if color == true_color else off
                if not 0 < coord < inp.shape[axis] - 1:
                    legal = False
                    break
                available = _anchor_options(modified, data["bg"], axis, coord)
                for old in ps:
                    point = (coord, old[1]) if axis == 0 else (old[0], coord)
                    if point not in available:
                        legal = False
                        break
                    modified[point] = color
                if not legal:
                    break
            if not legal:
                continue
            answers, _ = _solutions(modified)
            if len(answers) == 1 and answers[0][1][0]["paint"] != original_paint:
                original_body, modified_body = inp.copy(), modified.copy()
                for grid in (original_body, modified_body):
                    grid[0, :] = grid[-1, :] = data["bg"]
                    grid[:, 0] = grid[:, -1] = data["bg"]
                _require(np.array_equal(original_body, modified_body), "counterfactual changed a key or lock")
                changed_pose = True
                break
        if changed_pose:
            break
    _require(changed_pose, "moving guides must change an assembled key, not just border pixels")
    # Palette and scene rotation must not supply hidden solver information.
    fixture = generate_puzzle(314, 2, 1)
    rng = np.random.default_rng(23)
    for turn in (1, 2, 3):
        colors = rng.permutation(10)
        for s in fixture:
            for pair in fixture[s]:
                inp = colors[np.rot90(pair["input"], turn)]
                out = colors[np.rot90(pair["output"], turn)]
                _require(np.array_equal(solve(inp), out), "rotation/recolor covariance")
    # Output damage, missing guides, and a removed eye must be detected.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in fixture[s]] for s in fixture}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    inp = fixture["train"][0]["input"].copy()
    data = _decode(inp)
    for ps in data["anchors"].values():
        for p in ps:
            inp[p] = data["bg"]
    try:
        solve(inp)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("missing guides accepted")
    inp = fixture["train"][0]["input"].copy()
    key = _decode(inp)["keys"][0]
    inp[key["eye"]] = key["colors"][key["handle"]]
    try:
        solve(inp)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("filled eye accepted")
