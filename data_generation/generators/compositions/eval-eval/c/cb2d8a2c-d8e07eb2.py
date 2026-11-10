"""Keyed shutters: exact colored locks control obstacle existence and clearance.

Only Python/NumPy; no parent imports, files, cached examples, or seed in solve.
The construction renderer and the pixel decoder are separate paths.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 743869070, 333547956, 772558134, 741184521, 381194806,
              838905507, 275895755, 570598083, 567316131, 469543575, 10086141,
              131124093, 964396857, 653940964, 591271371, 808232547, 839120331,
              811280559, 465594135, 227897975, 950030984, 544824251, 831750843,
              474733024, 810841658, 282127790, 910852349, 558261354, 422206229,
              183374303, 901843326, 843004406, 867455001, 242788371, 548500587,
              628606769, 738235396, 613722557, 649839900, 559618194, 959727769)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _digest(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _puzzle_digest(puzzle):
    return hashlib.sha256("".join(_digest(p[k]) for s in ("train", "test")
                                 for p in puzzle[s] for k in ("input", "output")).encode()).hexdigest()


def _orient(a, transform):
    return np.rot90(np.fliplr(a) if transform >= 4 else a, transform % 4).copy()


def _runs(values):
    values = list(map(int, values))
    groups = []
    for value in values:
        if not groups or value != groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _glyphs(two_rows, bg, offset=0):
    """Tight 2x2 bounding boxes, separated by at least one empty column."""
    result = []
    for columns in _runs(np.flatnonzero(np.any(two_rows != bg, axis=0))):
        assert len(columns) == 2
        col = columns[0]
        token = two_rows[:, col:col + 2]
        mask = token != bg
        assert np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
        inks = np.unique(token[mask])
        assert len(inks) == 1
        result.append({"col": col + offset, "token": token.copy(),
                       "key": tuple(map(int, token.flat)), "ink": int(inks[0]),
                       "shape": tuple(map(bool, mask.flat))})
    return result


def _parse_canonical(a):
    """The divider puts the four-row key rack above a downward routing field."""
    h, w = a.shape
    assert h >= 11 and w >= 15
    bg, rail = int(a[0, 0]), int(a[4, 0])
    assert bg != rail and np.all(a[4] == rail)
    assert np.all(a[[0, 3]] == bg)
    entry = np.flatnonzero(a[5] != bg)
    assert len(entry) == 1 and 0 < entry[0] < w - 1
    initial = int(entry[0])
    route = int(a[5, initial])
    assert route not in (bg, rail) and np.count_nonzero(a == route) == 1
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[4] = rail
    expected[5, initial] = route
    query = _glyphs(a[1:3], bg)
    for token in query:
        c = token["col"]
        expected[1:3, c:c + 2] = token["token"]
    rail_rows = list(map(int, np.flatnonzero(np.any(a[6:] == rail, axis=1)) + 6))
    assert 1 <= len(rail_rows) <= 3
    gates, lookup, cap_colors, previous = [], {}, set(), 5
    for row in rail_rows:
        top = row - 2
        assert top > previous and row < h - 1
        columns = np.flatnonzero(a[row] != bg)
        assert len(columns) >= 7
        lo, hi = int(columns[0]), int(columns[-1])
        assert len(columns) == hi - lo + 1
        other = np.flatnonzero((a[row] != bg) & (a[row] != rail))
        assert len(other) == 1 and int(other[0]) in (lo, hi)
        cap = int(a[row, other[0]])
        assert cap not in (bg, rail, route)
        cap_colors.add(cap)
        side = 1 if int(other[0]) == lo else -1
        glyphs = _glyphs(a[top:row], bg)
        assert 2 <= len(glyphs) <= 3
        assert all(lo < g["col"] and g["col"] + 1 < hi for g in glyphs)
        for glyph in glyphs:
            assert glyph["ink"] not in (bg, rail, route, cap)
            assert glyph["key"] not in lookup, "locks must have unique colored identities"
            lookup[glyph["key"]] = (len(gates), len(lookup))
            c = glyph["col"]
            expected[top:row, c:c + 2] = glyph["token"]
        expected[row, lo:hi + 1] = a[row, lo:hi + 1]
        gates.append({"top": top, "row": row, "lo": lo, "hi": hi,
                      "side": side, "glyphs": glyphs, "cap": cap})
        previous = row
    assert len(cap_colors) == 1
    cap = next(iter(cap_colors))
    assert all(g["ink"] != cap for gate in gates for g in gate["glyphs"])
    query_keys = [q["key"] for q in query]
    assert len(query_keys) == len(set(query_keys))
    assert all(key in lookup for key in query_keys), "every key needs one exact lock copy"
    selected = set(query_keys)
    for gate in gates:
        gate["matched"] = tuple(g["key"] in selected for g in gate["glyphs"])
        gate["remaining"] = len(gate["glyphs"]) - sum(gate["matched"])
    assert np.array_equal(expected, a), "unexpected ink outside rack, locks, rails, or entry"
    return {"grid": a, "bg": bg, "rail": rail, "cap": cap, "route": route,
            "initial": initial, "query": query, "gates": gates}


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(a, rotation))
        except AssertionError:
            continue
        info["rotation"] = rotation
        candidates.append(info)
    assert len(candidates) == 1, "key rack and entry must give a unique orientation"
    return candidates[0]


def _decode(info):
    a = info["grid"]
    out = a.copy()
    bg, rail, route = (info[k] for k in ("bg", "rail", "route"))
    path = np.zeros(a.shape, dtype=bool)
    blocked = np.zeros(a.shape, dtype=bool)
    current, start, previous = info["initial"], 5, 5
    bends = []
    for gate in info["gates"]:
        top, row, lo, hi, side, n = (gate[k] for k in
                                    ("top", "row", "lo", "hi", "side", "remaining"))
        assert lo <= current <= hi, "each shutter crosses the arriving ray"
        for glyph, matched in zip(gate["glyphs"], gate["matched"]):
            if matched:
                c = glyph["col"]
                block = out[top:row, c:c + 2]
                block[glyph["token"] != bg] = bg
        if n == 0:
            out[row, lo:hi + 1] = bg
            continue
        out[row, lo:hi + 1] = rail
        blocked[top:row + 1, lo:hi + 1] = True
        distance = n + 1
        turn = top - distance
        target = hi + distance if side == 1 else lo - distance
        assert turn > previous and turn >= start and 0 <= target < a.shape[1]
        # Deliberately walk unit cells; construction paints complete slices.
        for r in range(start, turn + 1):
            path[r, current] = True
        step = 1 if target > current else -1
        for c in range(current, target + step, step):
            path[turn, c] = True
        bends.append((turn, current, target, n, side))
        current, start, previous = target, turn, row
    for r in range(start, len(a)):
        path[r, current] = True
    assert not np.any(path & blocked), "route cuts through a closed shutter"
    assert np.all((out[path] == bg) | (out[path] == route))
    out[path] = route
    return out, path, bends


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer orientation, exact matches, remaining counts, and the entire route."""
    info = _analyze(input_grid)
    out, _, _ = _decode(info)
    return np.rot90(out, -info["rotation"]).copy()


def _masks():
    bank = []
    for code in range(1, 16):
        mask = np.array([(code >> j) & 1 for j in range(4)], dtype=bool).reshape(2, 2)
        if np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            bank.append(mask)
    return bank


_MASKS = _masks()


def _vocabulary(rng, counts, selected, colors):
    """Fresh exact tokens, with selected/unselected shape and color confounders."""
    total = sum(counts)
    inks = colors[4:]
    bank = [(i, ink) for i in range(len(_MASKS)) for ink in inks]
    chosen = [None] * total
    yes = [i for i in range(total) if i in selected]
    no = [i for i in range(total) if i not in selected]
    if yes and no:
        # The first lesson puts the first selected key in the open shutter and
        # the first unselected twin in a different, only partly unlocked one.
        first, second = map(int, rng.choice(len(_MASKS), 2, replace=False))
        i0, i1 = map(int, rng.choice(inks, 2, replace=False))
        chosen[yes[0]], chosen[no[0]] = (first, i0), (first, i1)
        if len(no) >= 2:
            chosen[no[1]] = (second, i0)
        elif len(yes) >= 2:
            chosen[yes[1]] = (second, i1)
    used = {token for token in chosen if token is not None}
    available = [token for token in bank if token not in used]
    rng.shuffle(available)
    for i in range(total):
        if chosen[i] is None:
            chosen[i] = available.pop()
    bg = colors[0]
    arrays = []
    for index, ink in chosen:
        token = np.full((2, 2), bg, dtype=np.int8)
        token[_MASKS[index]] = ink
        arrays.append(token)
    return arrays


def _geometry(rng, width, counts, remaining, sides):
    """At most three gates; bounded finite search, no unbounded rejection loop."""
    failed = set()

    def search(index, current):
        if index == len(counts):
            return []
        state = index, current
        if state in failed:
            return None
        k, n = counts[index], remaining[index]
        options = []
        for padding in range(4):
            length = 3 * k + 1 + padding
            for lo in range(width - length + 1):
                hi = lo + length - 1
                if not lo <= current <= hi:
                    continue
                for side in ((sides[index],) if sides[index] is not None else (-1, 1)):
                    target = current if n == 0 else hi + n + 1 if side == 1 else lo - n - 1
                    if 0 <= target < width:
                        options.append((lo, hi, side, target, padding))
        for raw in rng.permutation(len(options)):
            lo, hi, side, target, padding = options[int(raw)]
            rest = search(index + 1, target)
            if rest is not None:
                left_padding = int(rng.integers(padding + 1))
                return [(lo, hi, side, target, left_padding)] + rest
        failed.add(state)
        return None

    for raw in rng.permutation(np.arange(1, width - 1)):
        initial = int(raw)
        gates = search(0, initial)
        if gates is not None:
            return initial, gates
    return None


def _make_pair(rng, colors, remaining, counts, sides, transform):
    selected, offset = set(), 0
    for k, n in zip(counts, remaining):
        for j in rng.choice(k, k - n, replace=False):
            selected.add(offset + int(j))
        offset += k
    key_count = len(selected)
    width_min = max(22, 3 * key_count + 1)
    assert width_min <= 30
    width = int(rng.integers(width_min, 31))
    plan = _geometry(rng, width, counts, remaining, sides)
    if plan is None:
        width = 30
        plan = _geometry(rng, width, counts, remaining, sides)
    assert plan is not None
    initial, spans = plan
    delays = [n + 1 if n else 1 for n in remaining]
    minimum = sum(delays) + 3 * len(counts) + 2
    assert minimum <= 25
    field_height = int(rng.integers(minimum, 26))
    slack = rng.multinomial(field_height - minimum,
                            np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    tops, previous = [], 5
    for i, delay in enumerate(delays):
        top = previous + 1 + int(slack[i]) + delay
        tops.append(top)
        previous = top + 2
    assert previous + 2 + int(slack[-1]) == field_height + 5
    bg, rail, cap, route = colors[:4]
    inp = np.full((field_height + 5, width), bg, dtype=np.int8)
    inp[4] = rail
    inp[5, initial] = route
    out = inp.copy()
    tokens = _vocabulary(rng, counts, selected, colors)
    # Rack keys are an unordered set; blank positions carry no instruction.
    capacity = (width - 1) // 3
    slots = sorted(map(int, rng.choice(capacity, key_count, replace=False)))
    order = sorted(selected)
    rng.shuffle(order)
    for slot, index in zip(slots, order):
        c = 1 + 3 * slot
        inp[1:3, c:c + 2] = tokens[index]
        out[1:3, c:c + 2] = tokens[index]
    offset, current, start = 0, initial, 5
    for top, k, n, span in zip(tops, counts, remaining, spans):
        lo, hi, side, target, left_padding = span
        row = top + 2
        inp[row, lo:hi + 1] = rail
        inp[row, lo if side == 1 else hi] = cap
        if n:
            out[row, lo:hi + 1] = rail
        for j in range(k):
            c = lo + 1 + left_padding + 3 * j
            inp[top:row, c:c + 2] = tokens[offset + j]
            if offset + j not in selected:
                out[top:row, c:c + 2] = tokens[offset + j]
        if n:
            turn = top - n - 1
            out[start:turn + 1, current] = route
            out[turn, min(current, target):max(current, target) + 1] = route
            current, start = target, turn
        offset += k
    out[start:, current] = route
    return {"input": _orient(inp, transform), "output": _orient(out, transform)}


def _check_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside allowed range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _check_count(num_train, "num_train", 2, 10)
    nq = _check_count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    inks = int(rng.integers(2, 4))
    colors = tuple(map(int, rng.choice(10, 4 + inks, replace=False)))
    orientation = int(rng.integers(8))
    mixed = nt >= 3 and bool(rng.integers(2))
    recolor = bool(rng.integers(2))
    sign = int(rng.choice((-1, 1)))
    lessons = [([0, 1], [int(rng.integers(2, 4)), 3], [None, sign]),
               ([2, 3], [3, 3], [-sign, -sign])]
    if nt >= 3:
        lessons.append(([2], [2], [None]))  # empty key rack is explicit evidence
    while len(lessons) < nt:
        count = int(rng.integers(1, 3))
        ks = list(map(int, rng.integers(2, 4, size=count)))
        rs = [int(rng.integers(k + 1)) for k in ks]
        lessons.append((rs, ks, [None] * count))
    for _ in range(nq):
        # A longer interaction sequence is the shift, independent of palette,
        # heading, canvas size, query order, and the specific lock inventory.
        ks = list(map(int, rng.integers(2, 4, size=3)))
        rs = [int(rng.integers(k + 1)) for k in ks]
        if not any(rs):
            rs[int(rng.integers(3))] = 1
        if nt == 2 and sum(k - n for k, n in zip(ks, rs)) == 0:
            rs[int(rng.integers(3))] -= 1
        lessons.append((rs, ks, [None] * 3))
    puzzle, seen = {"train": [], "test": []}, set()
    for index, (rs, ks, sides) in enumerate(lessons):
        split = "train" if index < nt else "test"
        for _ in range(16):
            local = colors
            if recolor and index >= 2:
                free = [c for c in range(10) if c not in colors[1:3] + colors[4:]]
                bg, route = map(int, rng.choice(free, 2, replace=False))
                local = (bg, colors[1], colors[2], route, *colors[4:])
            transform = int(rng.integers(8)) if mixed and index >= 2 else orientation
            pair = _make_pair(rng, local, rs, ks, sides, transform)
            key = _digest(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("bounded uniqueness attempts exhausted")
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, path, bends):
    gates = info["gates"]
    selected = {q["key"] for q in info["query"]}
    glyphs = [g for gate in gates for g in gate["glyphs"]]
    yes = [g for g in glyphs if g["key"] in selected]
    no = [g for g in glyphs if g["key"] not in selected]
    return {"shape": tuple(info["grid"].shape), "rotation": info["rotation"],
            "count": len(gates), "remaining": tuple(g["remaining"] for g in gates),
            "locks": tuple(len(g["glyphs"]) for g in gates),
            "sides": tuple(g["side"] for g in gates if g["remaining"]),
            "open": sum(g["remaining"] == 0 for g in gates), "keys": len(yes),
            "color_matters": any(g["shape"] == q["shape"] for g in no for q in yes),
            "shape_matters": any(g["ink"] == q["ink"] for g in no for q in yes),
            "bends": tuple(bends), "path_cells": int(path.sum()),
            "palette": (info["bg"], info["rail"], info["cap"], info["route"]),
            "spans": tuple((g["lo"], g["hi"]) for g in gates)}


def measure_pair(pair):
    info = _analyze(pair["input"])
    _, path, bends = _decode(info)
    return _features(info, path, bends)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            digest = _digest(a)
            assert digest not in seen, "duplicate input"
            seen.add(digest)
            info = _analyze(a)
            expected, path, bends = _decode(info)
            canonical_out = np.rot90(b, info["rotation"])
            assert np.array_equal(canonical_out, expected), "incorrect lock consumption or route"
            assert np.array_equal(canonical_out[:5], info["grid"][:5]), "rack/divider changed"
            assert canonical_out[5, info["initial"]] == info["route"]
            assert np.count_nonzero(canonical_out == info["cap"]) == 0
            # The route has one connected, unbranched entry-to-exit walk. Count
            # cells by reconstruction rather than merely checking color usage.
            assert int(np.count_nonzero(canonical_out == info["route"])) == int(path.sum())
            assert np.count_nonzero(path[-1]) == 1
            for gate in info["gates"]:
                for glyph, matched in zip(gate["glyphs"], gate["matched"]):
                    c, r = glyph["col"], gate["top"]
                    ink = glyph["token"] != info["bg"]
                    if not matched:
                        assert np.array_equal(canonical_out[r:r + 2, c:c + 2], glyph["token"])
                    else:
                        assert np.all(canonical_out[r:r + 2, c:c + 2][ink] != glyph["ink"])
            records[split].append(_features(info, path, bends))
    train, test = records["train"], records["test"]
    assert {n for f in train for n in f["remaining"]} >= {0, 1, 2, 3}
    assert {s for f in train for s in f["sides"]} == {-1, 1}
    assert any(f["color_matters"] and f["shape_matters"] for f in train)
    assert any(f["open"] and f["bends"] for f in train)
    assert all(f["count"] <= 2 for f in train) and all(f["count"] == 3 for f in test)
    assert all(f["bends"] for f in test)
    if any(f["keys"] == 0 for f in test):
        assert any(f["keys"] == 0 for f in train)
    assert {n for f in test for n in f["remaining"]} <= {n for f in train for n in f["remaining"]}
    return None


def _fixture():
    """One roomy partial shutter; literal path expectations are given in self_test."""
    a = np.zeros((23, 28), dtype=np.int8)
    a[4] = 1
    a[5, 10] = 3
    # Same shape/different color in different locations; two queried keys.
    tokens = [np.array([[4, 0], [0, 4]], np.int8),
              np.array([[5, 5], [5, 5]], np.int8),
              np.array([[5, 0], [0, 5]], np.int8)]
    for i, token in enumerate(tokens):
        a[12:14, 7 + 3 * i:9 + 3 * i] = token
    a[14, 6:17] = 1
    a[14, 6] = 2
    a[1:3, 1:3] = tokens[0]
    a[1:3, 7:9] = tokens[1]
    return a


def self_test():
    # Streaming suite: at most two puzzles retained, no generated payload files.
    signatures, shapes, directions, palettes = set(), set(), set(), set()
    test_states, spans, lock_counts, structures = set(), set(), set(), set()
    open_tests = repeats = varied_layouts = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = _puzzle_digest(puzzle)
        assert signature == _puzzle_digest(generate_puzzle(seed))
        signatures.add(signature)
        local_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                result, path, bends = _decode(info)
                f = _features(info, path, bends)
                assert np.array_equal(np.rot90(result, -info["rotation"]), pair["output"])
                shapes.add(tuple(pair["input"].shape))
                local_shapes.add(tuple(pair["input"].shape))
                palettes.add(f["palette"])
                lock_counts.update(f["locks"])
                spans.update(hi - lo + 1 for lo, hi in f["spans"])
                roles = {}
                normalized = np.array([roles.setdefault(int(c), len(roles))
                                       for c in pair["input"].flat], np.int8)
                structures.add(hashlib.sha256(normalized.tobytes()).hexdigest())
                if split == "test":
                    directions.add(info["rotation"])
                    test_states.update(f["remaining"])
                    open_tests += f["open"] > 0
                    repeats += any(x == y for x, y in zip(f["sides"], f["sides"][1:]))
        varied_layouts += len(local_shapes) > 1
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(palettes) >= 35
    assert directions == {0, 1, 2, 3} and test_states == {0, 1, 2, 3}
    assert lock_counts == {2, 3} and len(spans) >= 6
    assert open_tests >= 12 and repeats >= 8 and varied_layouts == 42
    # Independent literal fixture checks the +1 distance and opposite cap.
    a = _fixture()
    expected = a.copy()
    expected[12:14, 7:9] = 0
    expected[12:14, 10:12] = 0
    expected[14, 6] = 1
    expected[5:11, 10] = 3
    expected[10, 10:19] = 3
    expected[10:, 18] = 3
    assert np.array_equal(solve(a), expected)
    # Omitting B: no exact keys -> three remaining locks, distance four.
    empty = a.copy()
    empty[1:3] = 0
    empty_out = solve(empty)
    assert empty_out[8, 20] == 3 and empty_out[10, 18] == 0
    # Omitting A: a straight ray after lock removal collides with the rail.
    assert expected[14, 10] == 1 and expected[14, 18] == 3
    # Completion: all three keys remove the entire gate; now route is straight.
    opened = a.copy()
    opened[1:3, 13:15] = a[12:14, 13:15]
    opened_out = solve(opened)
    assert np.all(opened_out[5:, 10] == 3)
    assert np.all(opened_out[14, 6:10] == 0) and opened_out[14, 6] == 0
    # Whole-scene dihedral transformations and arbitrary color permutations.
    remap = np.array([8, 6, 9, 1, 0, 4, 7, 2, 5, 3], np.int8)
    for transform in range(8):
        assert np.array_equal(solve(_orient(remap[a], transform)),
                              _orient(remap[expected], transform))
    # Reordering keys preserves all field pixels and merely preserves the rack.
    swapped = a.copy()
    swapped[1:3, 1:3], swapped[1:3, 7:9] = a[1:3, 7:9], a[1:3, 1:3]
    assert np.array_equal(solve(swapped)[5:], expected[5:])
    # All legal explicit counts; exceptions include boolean and noninteger values.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(901 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject semantic damage and duplicates, not just malformed array schemas.
    baseline = generate_puzzle(17)
    for defect in ("route", "lock", "duplicate"):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
        pair = damaged["test"][0]
        info = _analyze(pair["input"])
        out = np.rot90(pair["output"], info["rotation"])
        if defect == "route":
            r, c = np.argwhere((out == info["route"]) & (info["grid"] == info["bg"]))[0]
            out[r, c] = info["bg"]
        elif defect == "lock":
            gate = next(g for g in info["gates"] if g["remaining"])
            glyph = next(g for g, matched in zip(gate["glyphs"], gate["matched"]) if not matched)
            dr, dc = np.argwhere(glyph["token"] != info["bg"])[0]
            out[gate["top"] + dr, glyph["col"] + dc] = info["bg"]
        else:
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "normalized_inputs": len(structures),
            "palettes": len(palettes), "open_test_examples": open_tests,
            "repeated_side_tests": repeats, "all_legal_counts": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
