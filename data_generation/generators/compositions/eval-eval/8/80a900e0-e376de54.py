"""Staggered hollow beams synchronize to the middle emitter's wavefront.

No parent module or saved puzzle is used.  All parsing is input-only.
"""

from __future__ import annotations

import copy
import json

import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _field(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _outline(center, radius):
    r, c = center
    return [(r + x, c + y) for x in range(-radius, radius + 1)
            for y in range(-radius, radius + 1) if abs(x) + abs(y) == radius]


def _cap(center, radius, direction):
    r, c = center
    dr, dc = direction
    return [(r + dr * (k + 1), c + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _tail(endpoint, direction, length):
    return [(endpoint[0] + k * direction[0], endpoint[1] + k * direction[1])
            for k in range(1, length + 1)]


def _background(a):
    parity = np.indices(a.shape).sum(axis=0) % 2
    colors = []
    for p in (0, 1):
        values = a[parity == p]
        counts = np.bincount(values, minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > len(values), "checkerboard is not recoverable"
        colors.append(color)
    assert colors[0] != colors[1]
    return tuple(colors)


def _analyze(input_grid):
    """Recover diamonds, caps, existing rails, order, and a common contour."""
    a = _grid(input_grid)
    h, w = a.shape
    colors = _background(a)
    base = _field(a.shape, colors)
    foreground = set(map(tuple, np.argwhere(a != base)))
    emitters = []
    # Complete monochrome outlines, recognized independently of cap colors.
    for radius in (2, 3):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        offsets = _outline((0, 0), radius)
        x, y = offsets[0]
        sample = a[radius + x:radius + x + shape[0],
                   radius + y:radius + y + shape[1]]
        mask = (sample != colors[0]) & (sample != colors[1])
        for x, y in offsets[1:]:
            mask &= a[radius + x:radius + x + shape[0],
                      radius + y:radius + y + shape[1]] == sample
        for rr, cc in np.argwhere(mask):
            emitters.append({"center": (int(rr) + radius, int(cc) + radius),
                             "radius": radius, "struct": int(sample[rr, cc])})
    assert len(emitters) in (3, 5), "expected an odd array of 3 or 5 emitters"
    covered = set()
    for e in emitters:
        outline = set(_outline(e["center"], e["radius"]))
        assert not outline & covered
        covered.update(outline)
    for e in emitters:
        candidates = []
        for d in _DIRECTIONS:
            cells = _cap(e["center"], e["radius"], d)
            if not all(_inside(p, a.shape) for p in cells):
                continue
            values = {int(a[p]) for p in cells}
            if len(values) == 1 and next(iter(values)) not in (*colors, e["struct"]):
                candidates.append((d, cells, next(iter(values))))
        assert len(candidates) == 1, "each emitter has one complete active cap"
        d, cap, payload = candidates[0]
        assert not set(cap) & covered
        covered.update(cap)
        lengths, tails = [], []
        for endpoint in (cap[0], cap[-1]):
            cells = []
            p = (endpoint[0] + d[0], endpoint[1] + d[1])
            while _inside(p, a.shape) and int(a[p]) == payload:
                assert p not in covered
                cells.append(p)
                covered.add(p)
                p = (p[0] + d[0], p[1] + d[1])
            lengths.append(len(cells))
            tails.extend(cells)
        e.update(direction=d, cap=cap, color=payload, lengths=tuple(lengths), tails=tails)
    assert foreground == covered, "unowned foreground or a broken rail"
    directions = {e["direction"] for e in emitters}
    assert len(directions) == 1, "beams within a scene must be parallel"
    dr, dc = next(iter(directions))
    for e in emitters:
        r, c = e["center"]
        e["q"] = dc * r - dr * c
        e["cap_u"] = dr * r + dc * c + e["radius"] + 2
    emitters.sort(key=lambda e: e["q"])
    for left, right in zip(emitters, emitters[1:]):
        assert right["q"] - right["radius"] - left["q"] - left["radius"] >= 2
    middle = emitters[len(emitters) // 2]
    assert middle["lengths"][0] == middle["lengths"][1] >= 2
    front = middle["cap_u"] + 2 * middle["lengths"][0]
    for e in emitters:
        distance = front - e["cap_u"]
        assert distance >= 2 and distance % 2 == 0, "unreachable common front"
        e["target"] = distance // 2
        for endpoint in (e["cap"][0], e["cap"][-1]):
            assert all(_inside(p, a.shape) for p in _tail(endpoint, (dr, dc), e["target"]))
    return {"background": colors, "base": base, "emitters": emitters,
            "direction": (dr, dc), "front": front}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair only the two endpoint rails of each beam to one spatial front."""
    a = _grid(input_grid)
    analysis = _analyze(a)
    out = a.copy()
    for e in analysis["emitters"]:
        for p in e["tails"]:
            out[p] = analysis["base"][p]
    for e in analysis["emitters"]:
        for endpoint in (e["cap"][0], e["cap"][-1]):
            for p in _tail(endpoint, e["direction"], e["target"]):
                out[p] = e["color"]
    return out


def _contour_prediction(a, analysis):
    """Coordinate-mask check: no tail walking or length-based drawing."""
    rows, cols = np.indices(a.shape)
    dr, dc = analysis["direction"]
    u, q = dr * rows + dc * cols, dc * rows - dr * cols
    out = a.copy()
    for e in analysis["emitters"]:
        endpoints = np.abs(q - e["q"]) == e["radius"]
        beyond = endpoints & (u > e["cap_u"])
        out[beyond] = analysis["base"][beyond]
        out[beyond & (u <= analysis["front"])] = e["color"]
    return out


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _geometric_witness(analysis):
    es = analysis["emitters"]
    distances = [abs(2 * e["q"] - es[0]["q"] - es[-1]["q"]) for e in es]
    choices = [i for i, v in enumerate(distances) if v == min(distances)]
    return (len(choices) == 1 and choices[0] != len(es) // 2
            and es[choices[0]]["cap_u"] + 2 * max(es[choices[0]]["lengths"]) != analysis["front"])


def _features(a, out=None):
    analysis = _analyze(a)
    es = analysis["emitters"]
    middle = es[len(es) // 2]
    fronts = [e["cap_u"] + 2 * max(e["lengths"]) for e in es]
    deltas = [e["target"] - n for e in es for n in e["lengths"]]
    evidence = False
    if out is not None:
        for e in es:
            for inner in e["cap"][1:-1]:
                p = (inner[0] + e["direction"][0], inner[1] + e["direction"][1])
                if _inside(p, a.shape) and a[p] == out[p] == analysis["base"][p]:
                    evidence |= any(n < e["target"] for n in e["lengths"])
    return {"count": len(es), "radii": tuple(e["radius"] for e in es),
            "direction": analysis["direction"], "reference": middle["lengths"][0],
            "depth_span": max(e["cap_u"] for e in es) - min(e["cap_u"] for e in es),
            "targets": tuple(e["target"] for e in es),
            "lengths": tuple(e["lengths"] for e in es),
            "q_gaps": tuple(b["q"] - a["q"] for a, b in zip(es, es[1:])),
            "extend": max(deltas), "trim": -min(deltas),
            "unbalanced": any(x != y for x, y in (e["lengths"] for e in es)),
            "nonmedian": sorted(fronts)[len(es) // 2] != analysis["front"],
            "geometric_witness": _geometric_witness(analysis),
            "hollow_witness": bool(evidence), "shape": a.shape,
            "background": analysis["background"],
            "layout": (a.shape, analysis["direction"],
                       tuple((e["center"], e["radius"], e["lengths"], e["target"]) for e in es))}


def _lengths(rng, targets, reference, witness, mode):
    n = len(targets)
    result = [[t, t] for t in targets]
    others = [int(i) for i in rng.permutation(n) if i != n // 2]
    if witness == 0:
        # Three sub-front pairs and one over-front pair defeat numerical median.
        for i in others:
            t = targets[i]
            v = int(rng.integers(max(0, t - 3), t))
            result[i] = [v, v]
        i = others[0]
        v = targets[i] + int(rng.integers(1, 3))
        result[i] = [v, v]
    elif witness == 1:
        i, j = others[:2]
        t = targets[i]
        v = max(0, t - int(rng.integers(1, 3)))
        result[i] = [v, v]
        t = targets[j]
        result[j] = [t + int(rng.integers(1, 4)), max(0, t - 1)]
        if rng.integers(2):
            result[j].reverse()
    else:
        for i in others:
            t = targets[i]
            for j in (0, 1):
                if rng.random() < .25:
                    continue
                sign = -1 if mode == "extend" else 1 if mode == "trim" else int(rng.choice((-1, 1)))
                result[i][j] = max(0, t + sign * int(rng.integers(1, 5)))
            if rng.random() < .40:
                result[i][1] = result[i][0]
        i = others[0]
        if mode == "single":
            result = [[t, t] for t in targets]
            result[i][int(rng.integers(2))] += int(rng.choice((-1, 1)))
        elif mode == "mixed":
            result[i][0] = max(0, targets[i] - 1)
            result[others[-1]][1] = targets[others[-1]] + 1
        elif all(pair == [t, t] for pair, t in zip(result, targets)):
            result[i][0] += -1 if mode == "extend" else 1
    result[n // 2] = [reference, reference]
    return result


def _sample_pair(rng, witness, test, avoid_reference=None, turn=0):
    """Construct truth directly from an explicit spatial contour, then parse it."""
    for attempt in range(800):
        n = 5 if witness == 0 else 3 if witness == 1 else int(rng.choice((3, 5)))
        radii = list(map(int, rng.choice((2, 3), n, p=(.7, .3))))
        if witness == 1:
            radii[0], radii[-1] = 2, 3
        if witness == 0 and attempt > 100:
            radii = [2] * n
        ref_choices = range(2, 5) if witness == 0 else range(3, 7) if n == 3 else range(3, 6)
        reference = int(rng.choice([x for x in ref_choices if x != avoid_reference]))
        stagger = (int(rng.choice((2, 3))) if n == 3 and reference >= 4 else 2) if test else 1
        depth = list(map(int, rng.integers(-stagger, stagger + 1, n)))
        depth[n // 2] = 0
        if witness == 0 and rng.random() < .5:
            depth = [0] * n
        if witness == 1 or test:
            others = [int(i) for i in rng.permutation(n) if i != n // 2]
            depth[others[0]], depth[others[-1]] = -stagger, stagger
        targets = [reference - v for v in depth]
        if min(targets) < 1:
            continue
        gaps = [radii[i] + radii[i + 1] + 2 for i in range(n - 1)]
        if witness == 0:
            gaps[0 if rng.integers(2) else -1] += 8
        else:
            gaps = [g + 2 * int(rng.integers(3)) for g in gaps]
        qs = [radii[0] % 2]
        for gap in gaps:
            qs.append(qs[-1] + gap)
        centers = []
        for radius, q, dep in zip(radii, qs, depth):
            s = 2 * dep - radius - 2
            assert (s - q) % 2 == 0
            centers.append(((s + q) // 2, (s - q) // 2))
        mode = str(rng.choice(("mixed", "extend", "trim", "single")))
        lengths = _lengths(rng, targets, reference, witness, mode)
        static, input_tails, output_tails, probes = [], [], [], []
        for center, radius, spans, target in zip(centers, radii, lengths, targets):
            cap = _cap(center, radius, (1, 1))
            static.append((_outline(center, radius), cap))
            its_input, its_output = [], []
            for endpoint, length in zip((cap[0], cap[-1]), spans):
                its_input.extend(_tail(endpoint, (1, 1), length))
                its_output.extend(_tail(endpoint, (1, 1), target))
                probes.append((endpoint[0] + target + 1, endpoint[1] + target + 1))
            input_tails.append(its_input)
            output_tails.append(its_output)
        cells = [p for outline, cap in static for p in outline + cap]
        cells += [p for paths in (input_tails, output_tails) for path in paths for p in path]
        cells += probes  # A visible empty step beyond the shared front.
        low = tuple(min(p[k] for p in cells) for k in (0, 1))
        box = tuple(max(p[k] for p in cells) - low[k] + 1 for k in (0, 1))
        if max(box) > 30:
            continue
        shape = tuple(int(rng.integers(max(12, b), 31)) for b in box)
        offset = tuple(int(rng.integers(shape[k] - box[k] + 1)) - low[k] for k in (0, 1))
        palette = list(map(int, rng.permutation(10)))
        background = tuple(palette[:2])
        pool = palette[2:]
        shared = bool(rng.integers(2))
        structs = [pool[0] if shared else int(rng.choice(pool)) for _ in range(n)]
        payloads = [int(rng.choice([c for c in pool if c != s])) for s in structs]
        if witness == 0:
            structs, payloads = [pool[0]] * n, [pool[1]] * n
        elif witness == 1:
            structs = [pool[0]] * n
            payloads = [pool[1], pool[2], pool[1]]
        inp = _field(shape, background)
        out = inp.copy()
        occupied_in, occupied_out = set(), set()

        def draw(grid, points, color, occupied):
            for p in points:
                cell = (p[0] + offset[0], p[1] + offset[1])
                assert cell not in occupied
                occupied.add(cell)
                grid[cell] = color

        for i, (outline, cap) in enumerate(static):
            for grid, tails, occupied in ((inp, input_tails[i], occupied_in),
                                          (out, output_tails[i], occupied_out)):
                draw(grid, outline, structs[i], occupied)
                draw(grid, cap + tails, payloads[i], occupied)
        if rng.integers(2):
            inp, out = np.fliplr(inp), np.fliplr(out)
        inp, out = np.rot90(inp, turn).copy(), np.rot90(out, turn).copy()
        try:
            analysis = _analyze(inp)
            _background(out)
            assert np.array_equal(out, _contour_prediction(inp, analysis))
            f = _features(inp, out)
            if witness == 0:
                assert f["nonmedian"] and f["geometric_witness"]
            if witness == 1:
                assert f["unbalanced"] and len(set(f["targets"])) > 1
            assert f["extend"] > 0 or f["trim"] > 0
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded geometric construction exhausted")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    base_turn = int(rng.integers(4))
    first_reference = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = i if split == "train" and i < 2 else None
            turn = (base_turn + i % 2) % 4 if split == "train" else int(rng.integers(4))
            for _ in range(40):
                pair = _sample_pair(rng, witness, split == "test",
                                    first_reference if witness == 1 else None, turn)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if witness == 0:
                        first_reference = _features(pair["input"])["reference"]
                    break
            else:
                raise RuntimeError("unique-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            analysis = _analyze(inp)
            assert np.array_equal(out, solve(inp)), "wrong beam repair"
            assert np.array_equal(out, _contour_prediction(inp, analysis)), "wrong frontier"
            change_allowed = np.zeros(inp.shape, dtype=bool)
            for e in analysis["emitters"]:
                for p in _outline(e["center"], e["radius"]) + e["cap"]:
                    assert out[p] == inp[p], "changed an emitter or cap"
                for endpoint, old_length in zip((e["cap"][0], e["cap"][-1]), e["lengths"]):
                    for k, p in enumerate(_tail(endpoint, e["direction"], max(old_length, e["target"])), 1):
                        change_allowed[p] = True
                        assert out[p] == (e["color"] if k <= e["target"] else analysis["base"][p])
                    tip = _tail(endpoint, e["direction"], e["target"])[-1]
                    assert tip[0] * e["direction"][0] + tip[1] * e["direction"][1] == analysis["front"]
                    next_cell = (tip[0] + e["direction"][0], tip[1] + e["direction"][1])
                    assert _inside(next_cell, inp.shape) and out[next_cell] == analysis["base"][next_cell]
            assert np.array_equal(inp[~change_allowed], out[~change_allowed])
            features[split].append(_features(inp, out))
    train = features["train"]
    assert any(f["nonmedian"] and f["geometric_witness"] for f in train)
    assert any(f["unbalanced"] and len(set(f["targets"])) > 1 for f in train)
    assert any(f["hollow_witness"] for f in train)
    assert any(f["extend"] > 0 for f in train) and any(f["trim"] > 0 for f in train)
    assert {f["count"] for f in train} == {3, 5}
    assert len({f["reference"] for f in train}) > 1
    assert {r for f in train for r in f["radii"]} == {2, 3}
    assert all(f["depth_span"] <= 4 for f in train)
    assert all(f["depth_span"] >= 8 for f in features["test"])


def self_test():
    """Bounded 42-seed audit, counterfactuals, symmetry, counts, and damage."""
    rng = np.random.default_rng(19177)
    records = {"train": [], "test": []}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p[r]) for s in ("train", "test") for p in puzzle[s] for r in ("input", "output"))
        signatures.add(signature)
        if seed < 3:
            again = generate_puzzle(seed)
            assert signature == tuple(_key(p[r]) for s in ("train", "test") for p in again[s] for r in ("input", "output"))
        for split in records:
            for p in puzzle[split]:
                records[split].append(_features(p["input"], p["output"]))
        pair = puzzle["test"][0]
        inp, out = pair["input"], pair["output"]
        # All color identities and all diagonal orientations are semantic symmetries.
        perm = rng.permutation(10)
        assert np.array_equal(solve(perm[inp]), perm[out])
        for k in range(4):
            assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
            assert np.array_equal(solve(np.fliplr(np.rot90(inp, k))), np.fliplr(np.rot90(out, k)))
        analysis = _analyze(inp)
        # Omitting A: propagating every cap lane fills visibly empty interiors.
        e = analysis["emitters"][0]
        inner = e["cap"][1]
        probe = (inner[0] + e["direction"][0], inner[1] + e["direction"][1])
        assert out[probe] == analysis["base"][probe] != e["color"]
        # Omitting B: continuing to the boundary passes the empty wavefront step.
        middle = analysis["emitters"][len(analysis["emitters"]) // 2]
        end = middle["cap"][0]
        probe = _tail(end, middle["direction"], middle["target"] + 1)[-1]
        assert out[probe] == analysis["base"][probe] != middle["color"]
        # Copying the middle beam's LENGTH also fails under visible staggering.
        assert any(e["target"] != middle["target"] for e in analysis["emitters"])
        # Perturb only the reference's two tips. Other emitters must respond.
        altered = inp.copy()
        for endpoint in (middle["cap"][0], middle["cap"][-1]):
            cell = _tail(endpoint, middle["direction"], middle["target"] + 1)[-1]
            altered[cell] = middle["color"]
        changed = solve(altered)
        assert any(changed[p] != out[p] for e in analysis["emitters"] if e is not middle
                   for endpoint in (e["cap"][0], e["cap"][-1])
                   for p in _tail(endpoint, e["direction"], e["target"] + 1)[-1:])
    assert len(signatures) == 42
    for split, fs in records.items():
        assert {f["count"] for f in fs} == {3, 5}
        assert {f["direction"] for f in fs} == set(_DIRECTIONS)
        assert len({f["reference"] for f in fs}) >= 3
        assert len({f["shape"] for f in fs}) >= 15
        assert len({f["layout"] for f in fs}) >= (100 if split == "train" else 35)
        assert any(f["unbalanced"] for f in fs)
        assert any(len(set(f["radii"])) > 1 for f in fs)
        assert any(0 in pair for f in fs for pair in f["lengths"])
        assert any(f["extend"] > 0 and f["trim"] > 0 for f in fs)
    # Every legal explicit count combination, plus type/range errors.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(8100 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    bad = copy.deepcopy(generate_puzzle(0))
    pair = bad["test"][0]
    cell = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][cell] = pair["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged rail")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "shapes": {s: len({f["shape"] for f in fs}) for s, fs in records.items()},
            "layouts": {s: len({f["layout"] for f in fs}) for s, fs in records.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
