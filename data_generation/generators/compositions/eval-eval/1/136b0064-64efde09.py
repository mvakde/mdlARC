"""Distributed activation of glyph tapes, followed by local-frame path execution.

The input contains the complete template, active slots, colors, and active side.
No puzzle-specific dictionary or generation state is needed by solve().
"""
from __future__ import annotations

import numpy as np


MASKS = (
    ((1, 1, 0), (1, 0, 1), (0, 1, 0)),  # right three
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),  # left two
    ((1, 0, 1), (0, 1, 0), (0, 1, 0)),  # outward two
    ((1, 1, 1), (0, 1, 0), (1, 0, 1)),  # left four
)
STROKES = ((0, 1, 3), (0, -1, 2), (1, 0, 2), (0, -1, 4))


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _components(a, bg):
    """Eight-connectivity accommodates the original diagonal-contact glyphs."""
    unseen = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in unseen:
                        unseen.remove(p)
                        cells.add(p)
                        stack.append(p)
        result.append(cells)
    return result


def _canonical(a, bg):
    """Normalize a tape using its asymmetric rail, independent of its glyphs."""
    matches = []
    for t in range(8):
        b = _orient(a, t)
        h, w = b.shape
        if h != 5 or w % 4 != 3 or not 3 <= (w + 1) // 4 <= 6:
            continue
        rail, home, anchor = int(b[4, 1]), int(b[4, 0]), int(b[3, w // 2])
        ink = set(map(int, b[:3].flat)) - {bg}
        if len(ink) != 1:
            continue
        glyph_ink = ink.pop()
        if len({bg, rail, home, anchor, glyph_ink}) != 5:
            continue
        if not np.all(b[4, 1:] == rail):
            continue
        expected_rail = np.full(w, rail)
        expected_rail[w // 2] = anchor
        if not np.array_equal(b[3], expected_rail):
            continue
        if not np.all(b[:3, 3::4] == bg):
            continue
        commands = []
        for j in range((w + 1) // 4):
            mask = tuple(map(tuple, (b[:3, 4*j:4*j+3] != bg).astype(int).tolist()))
            if mask not in MASKS:
                break
            commands.append(MASKS.index(mask))
        if len(commands) == (w + 1) // 4:
            matches.append((b, tuple(commands), (bg, rail, home, anchor, glyph_ink)))
    assert len(matches) == 1, "missing or ambiguous tape fiducials"
    return matches[0]


def _point(e, row, col):
    p = np.argwhere(e["labels"] == row * e["length"] + col)
    assert p.shape == (1, 2)
    return e["r"] + int(p[0, 0]), e["c"] + int(p[0, 1])


def _frame(e, side, col):
    row, inside = (0, 1) if side == 0 else (4, 3)
    edge = _point(e, row, col)
    inward = _point(e, inside, col)
    p0, p1 = _point(e, row, 0), _point(e, row, 1)
    normal = edge[0] - inward[0], edge[1] - inward[1]
    tangent = p1[0] - p0[0], p1[1] - p0[1]
    return edge, normal, tangent


def _corridor(e, side, slot, shape):
    edge, normal, _ = _frame(e, side, 4 * slot + 1)
    r, c = edge[0] + normal[0], edge[1] + normal[1]
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r, c = r + normal[0], c + normal[1]
    return tuple(cells)


def _path(e, side, code, commands, *, execute_all=False, restart=False, reverse=False,
          global_frame=False, leave_gaps=False):
    """A's endpoint chaining in B's exact template-relative coordinate frame."""
    origin, normal, tangent = _frame(e, side, e["length"] // 2)
    if global_frame:
        normal, tangent = (1, 0), (0, 1)
    depth = offset = 0
    painted = {}
    slots = list(range(len(commands))) if execute_all or leave_gaps else sorted(code)
    if reverse:
        slots.reverse()
    fallback_color = code[min(code)]
    for slot in slots:
        if leave_gaps and slot not in code:
            depth += 1
            continue
        if restart:
            depth = offset = 0
        dr, dc, size = STROKES[commands[slot]]
        color = code.get(slot, fallback_color)
        for k in range(size):
            d, x = depth + 1 + dr*k, offset + dc*k
            p = (origin[0] + d*normal[0] + x*tangent[0],
                 origin[1] + d*normal[1] + x*tangent[1])
            painted[p] = color
        depth += 1 + dr*(size-1)
        offset += dc*(size-1)
    return painted


def _decode(inp):
    inp = _grid(inp)
    values, counts = np.unique(inp, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    tapes, clues = [], []
    canonical = commands = roles = None
    for cells in _components(inp, bg):
        if len(cells) == 1:
            p = next(iter(cells))
            assert p[0] in (0, inp.shape[0]-1) or p[1] in (0, inp.shape[1]-1)
            clues.append((p, int(inp[p])))
            continue
        rs, cs = zip(*cells)
        r, c, r1, c1 = min(rs), min(cs), max(rs)+1, max(cs)+1
        a = inp[r:r1, c:c1]
        b, cmds, colors = _canonical(a, bg)
        if canonical is None:
            canonical, commands, roles = b, cmds, colors
        assert np.array_equal(b, canonical), "copies disagree in local coordinates"
        ts = [t for t in range(8) if np.array_equal(_orient(canonical, t), a)]
        assert len(ts) == 1
        labels = _orient(np.arange(canonical.size).reshape(canonical.shape), ts[0])
        box = {(y, x) for y in range(r, r1) for x in range(c, c1)}
        tapes.append({"r": r, "c": c, "arr": a, "labels": labels,
                      "length": canonical.shape[1], "cells": cells,
                      "box": box, "transform": ts[0]})
    assert 2 <= len(tapes) <= 4 and clues
    structural = set(roles[1:])
    assert all(color not in structural and color != bg for _, color in clues)
    occupied = set().union(*(e["box"] for e in tapes))
    corridors = {}
    for i, e in enumerate(tapes):
        for side in (0, 1):
            for slot in range(len(commands)):
                ray = _corridor(e, side, slot, inp.shape)
                if len(ray) < 2 or occupied.intersection(ray):
                    continue
                if all(int(inp[p]) == bg for p in ray[:-1]):
                    corridors.setdefault(ray[-1], []).append((i, side, slot))
    code, sides, allocations, observations = {}, set(), [set() for _ in tapes], []
    for p, color in clues:
        matches = corridors.get(p, [])
        assert len(matches) == 1, "boundary clue has no unique tape/side/slot"
        i, side, slot = matches[0]
        assert slot not in code or code[slot] == color, "inconsistent repeated clue"
        code[slot] = color
        sides.add(side)
        allocations[i].add(slot)
        observations.append((i, slot, color))
    assert len(sides) == 1 and 2 <= len(code) < len(commands)
    side = sides.pop()
    output, painted = inp.copy(), {}
    for e in tapes:
        path = _path(e, side, code, commands)
        assert len(path) == sum(STROKES[commands[j]][2] for j in code)
        for p, color in path.items():
            assert 0 <= p[0] < inp.shape[0] and 0 <= p[1] < inp.shape[1], "path out of bounds"
            assert p not in occupied and p not in painted, "path hits a tape or another path"
            assert int(inp[p]) == bg, "path hits a boundary control clue"
            output[p] = color
            painted[p] = color
    info = {"background": bg, "roles": roles, "commands": commands,
            "code": code, "side": side, "tapes": tapes, "clues": clues,
            "allocations": tuple(map(len, allocations)), "observations": observations,
            "pooled": all(len(s) < len(code) for s in allocations),
            "shape": inp.shape, "painted": painted}
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels alone, accepting every valid family configuration."""
    return _decode(np.asarray(input_grid))[0]


def _make_template(commands, roles):
    bg, rail, home, anchor, ink = roles
    width = 4*len(commands)-1
    a = np.full((5, width), bg, dtype=np.int8)
    for slot, command in enumerate(commands):
        block = a[:3, 4*slot:4*slot+3]
        block[np.asarray(MASKS[command], dtype=bool)] = ink
    a[3:] = rail
    a[4, 0], a[3, width//2] = home, anchor
    return a


def _emitter(template, t, r, c):
    a = _orient(template, t)
    labels = _orient(np.arange(template.size).reshape(template.shape), t)
    return {"arr": a, "labels": labels, "r": r, "c": c,
            "length": template.shape[1], "transform": t,
            "box": {(r+y, c+x) for y in range(a.shape[0]) for x in range(a.shape[1])}}


def _compatible(e, path, placed, paths):
    for old, old_path in zip(placed, paths):
        if (e["box"] & old["box"] or e["box"] & old_path.keys()
                or path.keys() & old["box"] or path.keys() & old_path.keys()):
            return False
        # Empty moat makes both complete tapes and isolated clues segmentable.
        if any((r+dy, c+dx) in old["box"] for r, c in e["box"]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
            return False
    return True


def _clues(rng, placed, paths, side, code, shape, pooled):
    occupied = set().union(*(e["box"] for e in placed))
    path_cells = set().union(*(p.keys() for p in paths))
    ends = {}
    for i, e in enumerate(placed):
        for s in (0, 1):
            for slot in range((e["length"]+1)//4):
                ray = _corridor(e, s, slot, shape)
                if len(ray) >= 2 and not occupied.intersection(ray):
                    ends.setdefault(ray[-1], []).append((i, s, slot))
    choices = {}
    for slot in code:
        options = []
        for i, e in enumerate(placed):
            ray = _corridor(e, side, slot, shape)
            if len(ray) < 2:
                continue
            p = ray[-1]
            if ends.get(p) != [(i, side, slot)] or p in path_cells:
                continue
            if any((p[0]+dr, p[1]+dc) in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                continue
            options.append((i, slot, p))
        rng.shuffle(options)
        if not options:
            return None
        choices[slot] = options
    order = list(code)
    answer = []

    def search(k):
        if k == len(order):
            return not pooled or len({x[0] for x in answer}) >= 2
        for x in choices[order[k]]:
            if any(max(abs(x[2][0]-q[2][0]), abs(x[2][1]-q[2][1])) <= 1 for q in answer):
                continue
            answer.append(x)
            if search(k+1):
                return True
            answer.pop()
        return False

    if not search(0):
        return None
    if not pooled and rng.random() < .3:
        extras = [x for xs in choices.values() for x in xs]
        rng.shuffle(extras)
        for x in extras:
            if all(max(abs(x[2][0]-q[2][0]), abs(x[2][1]-q[2][1])) > 1 for q in answer):
                answer.append(x)
                break
    return answer


def _place(rng, template, transforms, side, code, commands, shape, pooled):
    for _ in range(18):
        placed, paths = [], []
        for t in transforms:
            relative = _emitter(template, t, 0, 0)
            path0 = _path(relative, side, code, commands)
            footprint = relative["box"] | path0.keys()
            rs, cs = zip(*footprint)
            lo_r, hi_r = -min(rs), shape[0]-max(rs)
            lo_c, hi_c = -min(cs), shape[1]-max(cs)
            if hi_r <= lo_r or hi_c <= lo_c:
                break
            for _ in range(70):
                r, c = int(rng.integers(lo_r, hi_r)), int(rng.integers(lo_c, hi_c))
                e = _emitter(template, t, r, c)
                path = {(y+r, x+c): color for (y, x), color in path0.items()}
                if _compatible(e, path, placed, paths):
                    placed.append(e)
                    paths.append(path)
                    break
            else:
                break
        if len(placed) == len(transforms):
            clues = _clues(rng, placed, paths, side, code, shape, pooled)
            if clues is not None:
                return placed, paths, clues
    return None


def _sample_pair(rng, split, index, previous_background):
    demonstration = split == "train" and index < 2
    for attempt in range(45):
        # The first two examples jointly teach all four fixed glyph meanings.
        # The selected slots are nonconsecutive sometimes, so skipping is visible.
        if demonstration:
            n, active = int(rng.integers(4, 6)), 3
            sequence = [0, 1, 2] if index == 0 else [3, 2, 0]
            if rng.integers(2):
                sequence.reverse()
        elif split == "train":
            active = int(rng.integers(2, 4))
            n = int(rng.integers(max(3, active+1), 6))
            sequence = list(map(int, rng.integers(4, size=active)))
        else:
            active = int(rng.integers(4, 6))
            n = int(rng.integers(active+1, 7))
            sequence = list(map(int, rng.integers(4, size=active)))
            # Every test combines all four demonstrated command types.
            sequence[:4] = list(map(int, rng.permutation(4)))
        slots = sorted(map(int, rng.choice(n, active, replace=False)))
        if demonstration and index == 0 and slots == list(range(active)):
            # Guarantee an internal skipped address, teaching compaction even
            # when only the minimum two demonstrations are requested.
            slots[-1] = n-1
        commands = list(map(int, rng.integers(4, size=n)))
        for j, command in zip(slots, sequence):
            commands[j] = command
        palette = list(map(int, rng.permutation(10)))
        if palette[0] == previous_background:
            palette[0], palette[-1] = palette[-1], palette[0]
        roles, signals = tuple(palette[:5]), palette[5:]
        if demonstration:
            colors = list(map(int, rng.choice(signals, active, replace=False)))
        else:
            colors = list(map(int, rng.choice(signals, active, replace=rng.random() < .4)))
        code = dict(zip(slots, colors))
        template = _make_template(commands, roles)
        side = int(rng.integers(2))
        base = int(rng.integers(8))
        if demonstration and index == 0:
            transforms = [base, base ^ 4]  # local longitude reversal
        elif demonstration:
            transforms = [base, (base//4)*4+(base+1)%4]
        else:
            count = int(rng.choice([2, 3, 4], p=[.65, .27, .08])) if n <= 4 else 2
            mode = int(rng.integers(3))
            transforms = ([base]*count if mode == 0 else
                          [base, (base//4)*4+(base+2)%4] + [base]*(count-2)
                          if mode == 1 else list(map(int, rng.integers(8, size=count))))
        rng.shuffle(transforms)
        shape = tuple(map(int, rng.integers(24 if demonstration else 18, 31, size=2)))
        if rng.random() < .25:
            shape = shape[0], shape[0]
        result = _place(rng, template, transforms, side, code, commands, shape, demonstration)
        if result is None:
            continue
        placed, paths, clues = result
        inp = np.full(shape, roles[0], dtype=np.int8)
        for e in placed:
            h, w = e["arr"].shape
            inp[e["r"]:e["r"]+h, e["c"]:e["c"]+w] = e["arr"]
        for _, j, p in clues:
            inp[p] = code[j]
        out = inp.copy()
        for path in paths:
            for p, color in path.items():
                out[p] = color
        # Reconstruction checks the actual returned pixels, not this metadata.
        try:
            predicted, info = _decode(inp)
        except AssertionError:
            continue
        if np.array_equal(predicted, out):
            return {"input": inp, "output": out}
    return _fallback_pair(rng, split, index, previous_background)


def _fallback_pair(rng, split, index, previous_background):
    """Analytic spacious placement; keeps the same teaching obligations."""
    if split == "test":
        commands, slots = (0, 1, 2, 0, 3), (0, 1, 2, 4)
        transforms, positions = (0, 2), ((8, 1), (17, 10))
    elif index == 1:
        commands, slots = (3, 1, 2, 0), (0, 2, 3)
        transforms, positions = (0, 1), ((6, 1), (13, 23))
    else:
        commands, slots = (0, 1, 3, 2), (0, 1, 3)
        transforms, positions = (0, 4), ((6, 1), (20, 14))
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_background:
        palette[0], palette[-1] = palette[-1], palette[0]
    roles = tuple(palette[:5])
    code = dict(zip(slots, palette[5:]))
    template, shape, side = _make_template(commands, roles), (30, 30), 0
    placed = [_emitter(template, t, *p) for t, p in zip(transforms, positions)]
    paths = [_path(e, side, code, commands) for e in placed]
    clues = _clues(rng, placed, paths, side, code, shape, True)
    assert clues is not None
    inp = np.full(shape, roles[0], dtype=np.int8)
    for e in placed:
        h, w = e["arr"].shape
        inp[e["r"]:e["r"]+h, e["c"]:e["c"]+w] = e["arr"]
    for _, j, p in clues:
        inp[p] = code[j]
    out = inp.copy()
    for path in paths:
        for p, color in path.items():
            out[p] = color
    assert np.array_equal(_decode(inp)[0], out)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, previous_background = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(12):
                pair = _sample_pair(rng, split, index, previous_background)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input search exhausted")
            seen.add(key)
            previous_background = _decode(pair["input"])[1]["background"]
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation_output(inp, info, mode):
    """Concrete shortcuts used only to audit teaching examples."""
    out = inp.copy()
    for i, e in enumerate(info["tapes"]):
        code = info["code"]
        if mode == "local_only":
            code = {j: color for owner, j, color in info["observations"] if owner == i}
            if not code:
                continue
        if mode == "straight_rays":
            path = {p: color for j, color in code.items()
                    for p in _corridor(e, info["side"], j, inp.shape)}
        else:
            path = _path(e, info["side"], code, info["commands"],
                         execute_all=mode == "all_slots", restart=mode == "restart",
                         reverse=mode == "reverse_order", global_frame=mode == "global_frame",
                         leave_gaps=mode == "leave_gaps")
        for p, color in path.items():
            if 0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1]:
                out[p] = color
    return out


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in analyses:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _decode(inp)
            assert np.array_equal(out, expected), "incorrect compiled path"
            assert not np.array_equal(inp, out)
            # Control data, complete tapes, and all other cells are preserved.
            for e in info["tapes"]:
                assert all(inp[p] == out[p] for p in e["box"])
            assert all(inp[p] == out[p] for p, _ in info["clues"])
            analyses[split].append(info)
    train = analyses["train"]
    assert {f["commands"][j] for f in train for j in f["code"]} == set(range(4))
    assert any(f["pooled"] for f in train), "no demonstration of distributed activation"
    assert any(len({e["transform"] % 2 for e in f["tapes"]}) == 2 for f in train)
    assert any(any(e["transform"] >= 4 for e in f["tapes"]) for f in train)
    assert min(len(f["code"]) for f in analyses["test"]) > max(len(f["code"]) for f in train)
    for mode in ("local_only", "straight_rays", "all_slots", "restart", "reverse_order",
                 "global_frame", "leave_gaps"):
        assert any(not np.array_equal(_ablation_output(np.asarray(pair["input"]), f, mode),
                                     np.asarray(pair["output"]))
                   for pair, f in zip(puzzle["train"], train)), f"shortcut survives: {mode}"


def self_test() -> None:
    # A valid smaller, border-touching four-copy member is not guaranteed to
    # occur in the random 42-seed suite. Check the wider decoder domain directly.
    roles, commands, code = (0, 1, 2, 3, 4), (0, 1, 3), {0: 5, 1: 6}
    template = _make_template(commands, roles)
    four = [_emitter(template, t, r, c) for t, r, c in
            ((0, 3, 0), (4, 3, 19), (0, 14, 0), (4, 14, 19))]
    inp = np.zeros((20, 30), dtype=np.int8)
    for e in four:
        h, w = e["arr"].shape
        inp[e["r"]:e["r"]+h, e["c"]:e["c"]+w] = e["arr"]
    inp[0, 1], inp[0, 24] = 5, 6
    expected = inp.copy()
    for e in four:
        for p, color in _path(e, 0, code, commands).items():
            expected[p] = color
    assert np.array_equal(solve(inp), expected)
    assert len(_decode(inp)[1]["tapes"]) == 4
    for seed in range(3):
        rng = np.random.default_rng(seed)
        fallback = {"train": [_fallback_pair(rng, "train", i, None) for i in range(2)],
                    "test": [_fallback_pair(rng, "test", 0, None)]}
        validate_generated_puzzle(fallback)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_decode(pair["input"])[1] for p in puzzles for split in ("train", "test")
                for pair in p[split]]
    assert len({tuple(_key(x["input"]) for x in p["train"]+p["test"]) for p in puzzles}) == 42
    assert len({f["shape"] for f in features}) >= 30
    assert {len(f["commands"]) for f in features} == {3, 4, 5, 6}
    assert {len(f["code"]) for f in features} == {2, 3, 4, 5}
    assert {e["transform"] for f in features for e in f["tapes"]} == set(range(8))
    assert len({len(f["tapes"]) for f in features}) >= 2
    assert {f["side"] for f in features} == {0, 1}
    assert len({f["commands"] for f in features}) >= 70
    assert len({tuple(f["code"]) for f in features}) >= 15
    assert len({f["background"] for f in features}) == 10
    assert any(len(set(f["code"].values())) < len(f["code"]) for f in features)
    assert any(max(f["allocations"]) == len(f["code"]) for f in features)
    assert any(len(f["clues"]) > len(f["code"]) for f in features)
    a, b = generate_puzzle(7001, 5, 2), generate_puzzle(7001, 5, 2)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Input-only inference remains equivariant under every whole-grid isometry.
    for t in range(8):
        for pair in a["train"][:2]:
            assert np.array_equal(solve(_orient(pair["input"], t)), _orient(pair["output"], t))
            recolor = np.roll(np.arange(10), t+1)
            assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    # Inactive glyphs are executable-looking data, but genuinely disabled.
    pair = a["train"][0]
    info = _decode(pair["input"])[1]
    slot = next(j for j in range(len(info["commands"])) if j not in info["code"])
    commands = list(info["commands"])
    commands[slot] = (commands[slot]+1) % 4
    template = _make_template(commands, info["roles"])
    changed = pair["input"].copy()
    expected = pair["output"].copy()
    for e in info["tapes"]:
        block = _orient(template, e["transform"])
        h, w = block.shape
        changed[e["r"]:e["r"]+h, e["c"]:e["c"]+w] = block
        expected[e["r"]:e["r"]+h, e["c"]:e["c"]+w] = block
    assert np.array_equal(solve(changed), expected)
    assert _decode(changed)[1]["painted"] == info["painted"]
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(1700+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    f = _decode(bad["test"][0]["input"])[1]
    p = next(iter(f["painted"]))
    bad["test"][0]["output"][p] = f["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("output corruption accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed family self-test passed")
