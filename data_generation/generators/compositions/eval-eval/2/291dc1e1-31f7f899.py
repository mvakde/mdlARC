"""Sort signed bar spans along a directed, winding wire.

The wire remains visible through every bar.  The start (1), finish (2), and
background (0) are fixed; the wire and bar colors are geometric, local roles.
No parent generator, files, training labels, or generation state are used by solve.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    r, c = p
    return [(y, x) for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= y < shape[0] and 0 <= x < shape[1]]


def _normal(tangent):
    # Left of the direction of travel, in image coordinates.
    dr, dc = tangent
    return -dc, dr


def _trace(a):
    starts, ends = np.argwhere(a == 1), np.argwhere(a == 2)
    assert starts.shape == ends.shape == (1, 2), "One start and one finish required"
    start, end = tuple(map(int, starts[0])), tuple(map(int, ends[0]))
    adjacent = [p for p in _neighbors(start, a.shape) if a[p] != 0]
    assert len(adjacent) == 1, "Start must have one wire neighbor"
    wire = int(a[adjacent[0]])
    assert wire >= 3
    cells = set(zip(*np.nonzero((a == wire) | (a == 1) | (a == 2))))
    assert all(len([q for q in _neighbors(p, a.shape) if q in cells]) ==
               (1 if p in (start, end) else 2) for p in cells), "Wire branches or touches itself"
    path, seen = [start], {start}
    while path[-1] != end:
        choices = [p for p in _neighbors(path[-1], a.shape) if p in cells and p not in seen]
        assert len(choices) == 1, "Wire is cyclic or disconnected"
        path.append(choices[0])
        seen.add(choices[0])
    assert seen == cells, "Unconnected wire pixels"
    return path, wire


def _parse(input_grid):
    """Recover a directed local coordinate frame at each visible slot."""
    a = _grid(input_grid)
    path, wire = _trace(a)
    path_set = set(path)
    signatures, frames = {}, {}
    for i in range(1, len(path)-1):
        before = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        after = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        if before != after:
            continue  # A bend cannot be a slot.
        n = _normal(after)
        frames[i] = (after, n)
        sides = []
        for sign in (-1, 1):
            cells = []
            k = 1
            while True:
                p = (path[i][0]+sign*k*n[0], path[i][1]+sign*k*n[1])
                if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]):
                    break
                if p in path_set or a[p] in (0, 1, 2, wire):
                    break
                cells.append(int(a[p]))
                k += 1
            sides.append(cells)
        colors = {c for side in sides for c in side}
        assert len(colors) <= 1, "A slot must be monochrome on both sides"
        if colors:
            signatures[i] = (colors.pop(), -len(sides[0]), len(sides[1]))
    records = []
    for i in sorted(signatures):
        color, lo, hi = signatures[i]
        if (records and records[-1]["indices"][-1] == i-1
                and records[-1]["signature"] == (color, lo, hi)
                and records[-1]["tangent"] == frames[i][0]):
            records[-1]["indices"].append(i)
        else:
            records.append({"indices": [i], "color": color, "lo": lo, "hi": hi,
                            "signature": (color, lo, hi), "tangent": frames[i][0],
                            "normal": frames[i][1]})
    assert 2 <= len(records) <= 12, "Expected 2..12 readable bar slots"
    rebuilt = np.zeros_like(a)
    for p in path:
        rebuilt[p] = a[p]
    templates = {}
    for record in records:
        lo, hi = record["lo"], record["hi"]
        length = hi-lo+1
        assert length >= 2
        assert templates.setdefault(length, (lo, hi)) == (lo, hi), "Different-offset length tie"
        for i in record["indices"]:
            r, c = path[i]
            dr, dc = record["normal"]
            for k in range(lo, hi+1):
                if k:
                    p = (r+k*dr, c+k*dc)
                    assert p not in path_set and rebuilt[p] == 0, "Overlapping bars"
                    rebuilt[p] = record["color"]
    assert np.array_equal(rebuilt, a), "Unexplained pixels, damaged rectangle, or bar at a bend"
    # Every source template must fit at every destination, without contacting
    # a different slot or any nonlocal piece of wire. This is a visible-domain
    # constraint, not a hidden packing convention.
    footprints = []
    for record in records:
        envelope = set()
        for lo, hi in templates.values():
            for i in record["indices"]:
                r, c = path[i]
                dr, dc = record["normal"]
                for k in range(lo, hi+1):
                    p = (r+k*dr, c+k*dc)
                    assert 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1], "Span would clip"
                    assert not k or p not in path_set, "Span would hit another wire segment"
                    envelope.add(p)
        assert not any(envelope & other for other in footprints), "Transfer envelopes overlap"
        footprints.append(envelope)
    return {"grid": a, "path": path, "wire": wire, "records": records}


def _render(path, shape, wire, records, spans):
    a = np.zeros(shape, dtype=np.int8)
    for p in path:
        a[p] = wire
    a[path[0]], a[path[-1]] = 1, 2
    for record, (lo, hi) in zip(records, spans):
        dr, dc = record["normal"]
        for i in record["indices"]:
            r, c = path[i]
            for k in range(lo, hi+1):
                if k:
                    a[r+k*dr, c+k*dc] = record["color"]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _parse(input_grid)
    spans = [(r["lo"], r["hi"]) for r in d["records"]]
    # Rank selection rather than the generation path's sorted-list assignment.
    output = d["grid"].copy()
    output[~np.isin(output, [0, 1, 2, d["wire"]])] = 0
    for destination in d["records"]:
        chosen = min(spans, key=lambda s: s[1]-s[0])
        spans.remove(chosen)
        dr, dc = destination["normal"]
        for i in destination["indices"]:
            r, c = d["path"][i]
            for k in range(chosen[0], chosen[1]+1):
                if k:
                    output[r+k*dr, c+k*dc] = destination["color"]
    return output


def _key(a):
    return a.shape, a.tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"], dtype=np.int8)),
                        _key(np.asarray(p["output"], dtype=np.int8))) for p in puzzle[s])
                 for s in ("train", "test"))


def _check_pair(pair):
    """Relational invariants; no call to solve, sorting, or painting helpers."""
    a, b = _parse(pair["input"]), _parse(pair["output"])
    assert a["grid"].shape == b["grid"].shape
    assert a["path"] == b["path"] and a["wire"] == b["wire"]
    assert [(r["indices"], r["color"], r["normal"]) for r in a["records"]] == [
        (r["indices"], r["color"], r["normal"]) for r in b["records"]], "Slots must stay fixed"
    remaining = [(r["lo"], r["hi"]) for r in a["records"]]
    previous = 0
    for record in b["records"]:
        span = record["lo"], record["hi"]
        assert span in remaining, "Signed local interval must transfer exactly"
        remaining.remove(span)
        length = span[1]-span[0]+1
        assert length >= previous, "Lengths must increase in directed path order"
        previous = length
    assert not remaining
    return a, b


def _features(pair):
    a, b = _check_pair(pair)
    rs = a["records"]
    spans = [(r["lo"], r["hi"]) for r in rs]
    target = [(r["lo"], r["hi"]) for r in b["records"]]
    lengths = [hi-lo+1 for lo, hi in spans]
    physical = sorted(range(len(rs)), key=lambda i: a["path"][rs[i]["indices"][0]])
    physical_assignment = [None]*len(rs)
    for i, span in zip(physical, sorted(spans, key=lambda s: s[1]-s[0])):
        physical_assignment[i] = span
    area = [spans[i] for i in sorted(range(len(rs)), key=lambda i: lengths[i]*len(rs[i]["indices"]))]
    colored_area = [spans[i] for i in sorted(range(len(rs)), key=lambda i: (lengths[i]-1)*len(rs[i]["indices"]))]
    wing = [spans[i] for i in sorted(range(len(rs)), key=lambda i: max(-spans[i][0], spans[i][1]))]
    color_transfer = any(any(source["color"] != destination["color"] for source in rs
                            if (source["lo"], source["hi"]) == assigned)
                         for destination, assigned in zip(rs, target))
    # Sort within maximal straight segments as a competing local-only rule.
    local = spans.copy()
    start = 0
    while start < len(rs):
        stop = start+1
        while stop < len(rs) and rs[stop]["tangent"] == rs[start]["tangent"]:
            segment = a["path"][rs[stop-1]["indices"][-1]:rs[stop]["indices"][0]+1]
            if any((q[0]-p[0], q[1]-p[1]) != rs[start]["tangent"] for p, q in zip(segment, segment[1:])):
                break
            stop += 1
        local[start:stop] = sorted(spans[start:stop], key=lambda s: s[1]-s[0])
        start = stop
    cross_turn = False
    for destination, assigned in zip(rs, target):
        sources = [r for r in rs if (r["lo"], r["hi"]) == assigned]
        if assigned[0]+assigned[1] and all(abs(s["tangent"][0]) != abs(destination["tangent"][0]) for s in sources):
            cross_turn = True
    bends = sum((q[0]-p[0], q[1]-p[1]) != (s[0]-q[0], s[1]-q[1])
                for p, q, s in zip(a["path"], a["path"][1:], a["path"][2:]))
    return {"count": len(rs), "bends": bends, "shape": a["grid"].shape,
            "lengths": tuple(lengths), "widths": tuple(len(r["indices"]) for r in rs),
            "normals": tuple(r["normal"] for r in rs), "wire": a["wire"],
            "colors": tuple(r["color"] for r in rs), "spans": tuple(spans),
            "changed": spans != target, "area_disagrees": area != target,
            "colored_area_disagrees": colored_area != target,
            "wing_disagrees": wing != target, "color_transfer_disagrees": color_transfer,
            "reversal_disagrees": spans[::-1] != target,
            "physical_order_disagrees": physical_assignment != target,
            "local_sort_disagrees": local != target, "cross_turn_offset": cross_turn,
            "recenter_disagrees": any(lo+hi for lo, hi in spans),
            "ties": len(set(lengths)) < len(lengths),
            "layout": (tuple(a["path"]), tuple(tuple(r["indices"]) for r in rs), tuple(spans))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            f = _features(pair)
            if split == "train":
                train.append(f)
    assert len({f["count"] for f in train}) >= 2, "Demonstrate different populations"
    assert sum(f["changed"] for f in train) >= 2
    for name in ("area_disagrees", "colored_area_disagrees", "wing_disagrees", "color_transfer_disagrees",
                 "reversal_disagrees", "physical_order_disagrees",
                 "local_sort_disagrees", "cross_turn_offset", "recenter_disagrees"):
        assert any(f[name] for f in train), f"Missing training evidence: {name}"
    assert {n for f in train for n in f["normals"]} == {(-1, 0), (1, 0), (0, -1), (0, 1)}, "Show all local frames"
    observed = {}
    for f in train:
        for length, span in zip(f["lengths"], f["spans"]):
            observed.setdefault(length, set()).add(span)
    assert any(len(s) > 1 for s in observed.values()), "Show that a length does not determine its offset"


def _walk(rng, rows, cols, minimum):
    """Random simple coarse-grid walk; its fine wire is a path, never a loop."""
    for _ in range(100):
        current = (int(rng.integers(rows)), int(rng.integers(cols)))
        nodes, used = [current], {current}
        desired = int(rng.integers(minimum, rows*cols+1))
        while len(nodes) < desired:
            choices = [p for p in _neighbors(current, (rows, cols)) if p not in used]
            if not choices:
                break
            current = choices[int(rng.integers(len(choices)))]
            used.add(current)
            nodes.append(current)
        if len(nodes) >= minimum:
            return nodes
    raise AssertionError("Could not sample a coarse walk")


def _geometry(rng, n, radius, witness=False):
    # Coarse spacing varies independently of the path's topology. For coverage
    # witnesses a 3x3 lattice gives room for slots with all four local normals.
    for _ in range(200):
        step = 10 if witness else int(rng.integers(7, 11))
        margin = radius+1
        maximum = (29-2*margin)//step+1
        if maximum < 2:
            continue
        rows = cols = 3 if witness else 0
        if not witness:
            rows, cols = map(int, rng.integers(2, maximum+1, size=2))
        if max(rows, cols) > maximum or rows*cols < n+1:
            continue
        nodes = _walk(rng, rows, cols, n+1)
        shape0 = ((rows-1)*step+2*margin+1, (cols-1)*step+2*margin+1)
        extra = [int(rng.integers(31-s)) for s in shape0]
        offsets = [margin+int(rng.integers(e+1)) for e in extra]
        nodes = [(r*step+offsets[0], c*step+offsets[1]) for r, c in nodes]
        path, candidates = [nodes[0]], []
        for p, q in zip(nodes, nodes[1:]):
            t = ((q[0]-p[0])//step, (q[1]-p[1])//step)
            initial = len(path)-1
            path.extend((p[0]+k*t[0], p[1]+k*t[1]) for k in range(1, step+1))
            width = int(rng.integers(1, 4))
            begin = (step-width+1)//2
            if not witness:
                begin += int(rng.integers(-1, 2))
            indices = list(range(initial+begin, initial+begin+width))
            normal = _normal(t)
            # Reserve width three so later locking a witness width is safe.
            reserve_indices = list(range(initial+(step-2)//2, initial+(step-2)//2+3)) if witness else indices
            envelope = {(path[i][0]+k*normal[0], path[i][1]+k*normal[1])
                        for i in reserve_indices for k in range(-radius, radius+1)}
            candidates.append({"indices": indices, "normal": normal, "tangent": t,
                               "envelope": envelope, "initial": initial, "step": step})
        path_set = set(path)
        occupied, chosen = set(), []
        for j in rng.permutation(len(candidates)):
            candidate = candidates[int(j)]
            e = candidate["envelope"]
            if e & occupied or any(q in occupied for p in e for q in _neighbors(p, (30, 30))):
                continue
            centerline = e & path_set
            if any(p in path_set and p not in {path[i] for i in candidate["indices"]}
                   for p in e if not witness):
                continue
            if any(q in path_set and q not in centerline for p in e-path_set
                   for q in _neighbors(p, (30, 30))):
                continue
            chosen.append(candidate)
            occupied.update(e)
            if len(chosen) == n:
                break
        if len(chosen) != n:
            continue
        chosen.sort(key=lambda r: r["indices"][0])
        if witness and {r["normal"] for r in chosen} != {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            continue
        return path, tuple(s+e for s, e in zip(shape0, extra)), chosen
    raise AssertionError("Could not pack safe wire slots")


def _sample_pair(rng, n, radius, witness=False, extension=False):
    for _ in range(150):
        path, shape, records = _geometry(rng, n, radius, witness)
        available = list(range(2, 2*radius+2))
        if witness:
            lengths = [3, 6] + list(map(int, rng.choice([x for x in available if x not in (3, 6)], n-2, replace=False)))
        else:
            lengths = list(map(int, rng.choice(available, n)))
            if extension and max(lengths) < 8:
                lengths[int(rng.integers(n))] = int(rng.integers(8, 10))
        spans_by_length = {}
        for length in set(lengths):
            low = -min(radius, length-1)
            high = min(0, radius-length+1)
            lo = int(rng.integers(low, high+1))
            spans_by_length[length] = (lo, lo+length-1)
        if witness:
            spans_by_length[3] = (-2, 0) if witness == 1 else (0, 2)
            spans_by_length[6] = (-2, 3) if witness == 1 else (-3, 2)
        rng.shuffle(lengths)
        wire = int(rng.integers(3, 10))
        palette = [c for c in range(3, 10) if c != wire]
        size = int(rng.integers(2 if witness else 1, min(len(palette), n)+1))
        accents = rng.choice(palette, size=size, replace=False)
        for r, length in zip(records, lengths):
            r["color"] = int(rng.choice(accents))
            if witness:
                width = 3 if length == 3 else 1 if length == 6 else int(rng.integers(1, 4))
                begin = r["initial"]+(r["step"]-width+1)//2
                r["indices"] = list(range(begin, begin+width))
        spans = [spans_by_length[length] for length in lengths]
        inp = _render(path, shape, wire, records, spans)
        out = _render(path, shape, wire, records, sorted(spans, key=lambda s: s[1]-s[0]))
        pair = {"input": inp, "output": out}
        if not witness:
            _check_pair(pair)
            return pair
        f = _features(pair)
        if all(f[name] for name in ("changed", "area_disagrees", "colored_area_disagrees", "wing_disagrees",
                                    "color_transfer_disagrees", "reversal_disagrees",
                                    "physical_order_disagrees", "local_sort_disagrees", "cross_turn_offset")):
            return pair
    raise AssertionError("Could not expose competing-rule witnesses")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    witness_counts = [4, int(rng.integers(5, 7))]
    # Query shift is a new combination: smaller/larger population, stronger
    # offset envelope, or a different topology. Labels always follow one rule.
    regime = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i+1 if split == "train" and i < 2 else 0
            if witness:
                n, radius = witness_counts[i], 3
            elif split == "test" and regime == 0:
                n, radius = int(rng.integers(2, 4)), 3
            elif split == "test" and regime == 1:
                n, radius = int(rng.integers(2, 5)), 4
            elif split == "test":
                n, radius = int(rng.integers(7, 10)), 2
            else:
                radius = int(rng.integers(1, 4))
                step_capacity = ((29-2*(radius+1))//10+1)**2-1
                n = int(rng.integers(2, min(6, step_capacity)+1))
            for _ in range(50):
                pair = _sample_pair(rng, n, radius, witness,
                                    extension=split == "test" and regime == 1)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Distinct-input budget exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed geometry, evidence, determinism, and mutation checks."""
    signatures, features = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                features.append(_features(pair))
    assert len(signatures) == 42
    assert len({f["layout"] for f in features}) >= 200
    assert len({f["shape"] for f in features}) >= 35
    assert {f["wire"] for f in features} == set(range(3, 10))
    assert {w for f in features for w in f["widths"]} == {1, 2, 3}
    assert {n for f in features for n in f["normals"]} == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    assert len({f["bends"] for f in features}) >= 5
    assert min(f["count"] for f in features) == 2 and max(f["count"] for f in features) >= 7
    assert any(f["ties"] for f in features) and any(not f["ties"] for f in features)
    assert any(len(set(f["colors"])) < f["count"] for f in features)
    assert max(max(f["lengths"]) for f in features) == 9
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    # Hand-specified nonlattice/straight examples cover rule members beyond
    # the random constructor's width, length, topology, and population ranges.
    # The first fixture's expected pixels are written directly, not rendered
    # by the constructor or computed by the solver.
    path = [(10, c) for c in range(2, 12)] + [(r, 11) for r in range(9, 1, -1)] + [(2, c) for c in range(10, 1, -1)]
    inp = np.zeros((13, 14), dtype=np.int8)
    for p0 in path:
        inp[p0] = 9
    inp[path[0]], inp[path[-1]] = 1, 2
    inp[[9, 11, 12], 5] = 3  # length four, [-2,1], travel right
    inp[6:8, 10] = 4          # length two, [0,1], travel up
    inp[1, 6:8] = inp[3, 6:8] = 8  # length three, [-1,1], travel left
    expected = np.zeros_like(inp)
    for p0 in path:
        expected[p0] = inp[p0]
    expected[9, 5] = 3
    expected[6:8, 10] = expected[6:8, 12] = 4
    expected[0:2, 6:8] = expected[3, 6:8] = 8
    assert np.array_equal(solve(inp), expected)
    _check_pair({"input": inp, "output": expected})
    # 12 slots on a straight wire, with ties and no sampled-lattice dependence.
    path = [(2, c) for c in range(1, 29)]
    records = [{"indices": [c-1], "normal": (-1, 0), "color": 3+c%3}
               for c in range(2, 26, 2)]
    spans = [(-1, 1), (0, 1)]*6
    inp = _render(path, (5, 30), 7, records, spans)
    expected = _render(path, (5, 30), 7, records, [(0, 1)]*6+[(-1, 1)]*6)
    assert np.array_equal(solve(inp), expected)
    _check_pair({"input": inp, "output": expected})
    # A 29-cell span may touch the outer boundary; it must transfer intact.
    path = [(14, c) for c in range(7)]
    records = [{"indices": [c], "normal": (-1, 0), "color": color}
               for c, color in ((2, 3), (4, 8))]
    inp = _render(path, (30, 7), 7, records, [(-14, 14), (0, 1)])
    expected = _render(path, (30, 7), 7, records, [(0, 1), (-14, 14)])
    assert np.array_equal(solve(inp), expected)
    _check_pair({"input": inp, "output": expected})
    # Wider slots are readable on the same straight axis.
    path = [(3, c) for c in range(1, 24)]
    records = [{"indices": list(range(3, 8)), "normal": (-1, 0), "color": 3},
               {"indices": [15], "normal": (-1, 0), "color": 8}]
    inp = _render(path, (7, 25), 7, records, [(-2, 1), (0, 1)])
    expected = _render(path, (7, 25), 7, records, [(0, 1), (-2, 1)])
    assert np.array_equal(solve(inp), expected)
    _check_pair({"input": inp, "output": expected})
    # Boundary counts; the host exercises the full legal Cartesian domain.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(123+nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    witness = p["train"][0]
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(witness["input"], turns)),
                              np.rot90(witness["output"], turns))
    d = _parse(witness["input"])
    rs = d["records"]
    spans = [(r["lo"], r["hi"]) for r in rs]
    mutations = [spans, spans[::-1],
                 [spans[i] for i in sorted(range(len(rs)), key=lambda i: (spans[i][1]-spans[i][0]+1)*len(rs[i]["indices"]))],
                 sorted(spans, key=lambda s: max(-s[0], s[1])),
                 [(-hi, -lo) for lo, hi in sorted(spans, key=lambda s: s[1]-s[0])]]
    for intervals in mutations:
        bad = _render(d["path"], d["grid"].shape, d["wire"], rs, intervals)
        assert not np.array_equal(bad, witness["output"])
        try:
            _check_pair({"input": witness["input"], "output": bad})
        except AssertionError:
            pass
        else:
            raise AssertionError("Accepted wrong rank or reflected interval")
    assigned = sorted(spans, key=lambda s: s[1]-s[0])
    moved_colors = copy.deepcopy(rs)
    for destination, span in zip(moved_colors, assigned):
        destination["color"] = next(r["color"] for r in rs if (r["lo"], r["hi"]) == span)
    bad = _render(d["path"], d["grid"].shape, d["wire"], moved_colors, assigned)
    assert not np.array_equal(bad, witness["output"])
    try:
        _check_pair({"input": witness["input"], "output": bad})
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted source colors moving with templates")
    damaged = copy.deepcopy(p)
    damaged["test"][0]["output"][0, 0] = 9
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Accepted unexplained output cell")
    return {"seeds": 42, "pairs": len(features), "shapes": len({f["shape"] for f in features}),
            "layouts": len({f["layout"] for f in features}),
            "slot_counts": sorted({f["count"] for f in features}),
            "bend_counts": sorted({f["bends"] for f in features})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
