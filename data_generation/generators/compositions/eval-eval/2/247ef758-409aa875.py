"""Rendezvous stencils: materialize coordinate copies with shared shot endpoints.

The catalog, paired border labels, and unary distance ruler are all visible.
No examples, generator state, or seed are used by solve().
"""
from itertools import combinations, product
from collections import Counter
import hashlib
import numpy as np


VECTORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
PAIR_DIRECTIONS = tuple(v for v in combinations(VECTORS, 2)
                        if len({x[0] for x in v}) == len({x[1] for x in v}) == 2)
TRIPLE_DIRECTIONS = tuple(v for v in combinations(VECTORS, 3)
                          if len({x[0] for x in v}) == len({x[1] for x in v}) == 3)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    return a, b


def _offsets(vector, scale):
    dr, dc = vector
    s = scale
    if dr == 0:
        return ((0, 0), (-s, -s * dc), (s, -s * dc))
    if dc == 0:
        return ((0, 0), (-s * dr, -s), (-s * dr, s))
    return ((0, 0), (-s * dr, 0), (0, -s * dc))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_, "integer grid"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a


def _decode_arrow(cells):
    matches = []
    for tip in sorted(cells):
        for vector in VECTORS:
            for scale in (1, 2):
                mask = {(tip[0] + r, tip[1] + c) for r, c in _offsets(vector, scale)}
                if mask == cells:
                    matches.append((tip, vector, scale))
    assert len(matches) == 1, "each color must describe one scaled arrowhead"
    return matches[0]


def _instances(templates, distance, policy="cartesian"):
    records = []
    for t in templates:
        rows, cols = t["rows"], t["cols"]
        coordinates = zip(rows, cols) if policy == "zip" else product(rows, cols)
        for r, c in coordinates:
            dr, dc = t["vector"]
            cells = tuple((r + y, c + x) for y, x in _offsets(t["vector"], t["scale"]))
            records.append({"color": t["color"], "tip": (r, c), "cells": cells,
                            "endpoint": (r + distance * dr, c + distance * dc)})
    return records


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    layouts = []
    # Empty field, four matching corners, constant divider, and its ruler make
    # the layout visible without relying on the most frequent color.
    for d in range(4, w - 5):
        divider = int(a[0, d])
        if not np.all(a[:, d] == divider):
            continue
        frame = int(a[0, d + 1])
        if any(int(a[r, c]) != frame for r, c in ((0, w-1), (h-1, d+1), (h-1, w-1))):
            continue
        field = a[1:-1, d+2:-1]
        if not field.size or not np.all(field == field.flat[0]):
            continue
        bg = int(field.flat[0])
        if len({bg, frame, divider}) != 3 or h < 8:
            continue
        ruler = a[1, :d]
        length = int(np.count_nonzero(ruler == divider))
        if not (3 <= length <= 5 and length < d):
            continue
        if not (np.all(ruler[:length] == divider) and np.all(ruler[length:] == bg)):
            continue
        if not np.all(a[[0, 2, 3], :d] == bg):
            continue
        if np.any(a[4:, :d] == divider) or np.any(a[:, :d] == frame):
            continue
        layouts.append((d, bg, divider, frame, length))
    assert len(layouts) == 1, "unique catalog / framed field required"
    d, bg, divider, frame, distance = layouts[0]
    assert np.array_equal(a[0, d+2:-1], a[-1, d+2:-1]), "paired column labels"
    assert np.array_equal(a[1:-1, d+1], a[1:-1, -1]), "paired row labels"
    templates = []
    for color in sorted(set(map(int, a[4:, :d].flat)) - {bg}):
        positions = {(int(r), int(c)) for r, c in np.argwhere(a[:, :d] == color)}
        assert len(positions) == 3
        tip, vector, scale = _decode_arrow(positions)
        rows = [r for r in range(1, h-1) if int(a[r, d+1]) == color]
        cols = [c for c in range(d+2, w-1) if int(a[0, c]) == color]
        templates.append(dict(color=color, source=positions, source_tip=tip,
                              vector=vector, scale=scale, rows=rows, cols=cols))
    assert 2 <= len(templates) <= 4, "two to four distinct catalog keys"
    assert set(map(int, a[0, d+2:-1])) | set(map(int, a[1:-1, d+1])) <= {frame} | {t["color"] for t in templates}
    records = _instances(templates, distance)
    occupied = set()
    for rec in records:
        cells = set(rec["cells"])
        assert not occupied & cells, "virtual arrow masks must not overlap"
        occupied |= cells
        assert all(1 <= r < h-1 and d+2 <= c < w-1 for r, c in cells | {rec["endpoint"]}), "unclipped virtual geometry"
    return dict(grid=a, d=d, bg=bg, divider=divider, frame=frame,
                distance=distance, templates=templates, records=records)


def _prediction(parsed, distance=None, policy="cartesian", materialize_all=False):
    a = parsed["grid"]
    records = parsed["records"] if distance is None and policy == "cartesian" else _instances(
        parsed["templates"], parsed["distance"] if distance is None else distance, policy)
    counts = Counter(rec["endpoint"] for rec in records)
    out = a.copy()
    for rec in records:
        if materialize_all or counts[rec["endpoint"]] >= 2:
            for cell in rec["cells"]:
                out[cell] = rec["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy only virtual arrowheads whose forward endpoints coincide."""
    return _prediction(_parse(input_grid))


def _valid(ts, distance, h, iw):
    rows, cols, occupied = {}, {}, set()
    for i, t in enumerate(ts):
        for value in t["rows"]:
            if value in rows and rows[value] != i:
                return False
            rows[value] = i
        for value in t["cols"]:
            if value in cols and cols[value] != i:
                return False
            cols[value] = i
    for rec in _instances(ts, distance):
        cells = set(rec["cells"])
        if cells & occupied or not all(1 <= r < h-1 and 0 <= c < iw for r, c in cells | {rec["endpoint"]}):
            return False
        occupied |= cells
    return True


def _add_extra(rng, ts, distance, h, iw, index, axis, limit=3):
    t = ts[index]
    if len(t[axis]) >= limit:
        return False
    choices = list(range(1, h-1) if axis == "rows" else range(iw))
    rng.shuffle(choices)
    for value in choices:
        if any(value in other[axis] for other in ts):
            continue
        t[axis].append(value)
        if _valid(ts, distance, h, iw):
            t[axis].sort()
            return True
        t[axis].remove(value)
    return False


def _off_diagonal_witness(rng, ts, distance, h, iw):
    """Put an actual selected copy outside the zip of its sorted axes."""
    for index in rng.permutation(len(ts)):
        t = ts[int(index)]
        r0, c0 = t["rows"][0], t["cols"][0]
        options = [(r, c) for r in range(1, h-1) for c in range(iw)
                   if (r-r0)*(c-c0) < 0
                   and not any(r in x["rows"] or c in x["cols"] for x in ts)]
        rng.shuffle(options)
        for r, c in options:
            t["rows"], t["cols"] = sorted([r0, r]), sorted([c0, c])
            if _valid(ts, distance, h, iw):
                return True
            t["rows"], t["cols"] = [r0], [c0]
    return False


def _second_site(rng, ts, distance, h, iw):
    """Translate one rendezvous to another; incidental Cartesian sites are real."""
    options = [(r, c) for r in range(-8, 9) for c in range(-8, 9) if r or c]
    rng.shuffle(options)
    old = [(list(t["rows"]), list(t["cols"])) for t in ts]
    for dr, dc in options:
        for t, (rs, cs) in zip(ts, old):
            t["rows"] = sorted(set(rs + [rs[0] + dr]))
            t["cols"] = sorted(set(cs + [cs[0] + dc]))
        if _valid(ts, distance, h, iw):
            return True
        for t, (rs, cs) in zip(ts, old):
            t["rows"], t["cols"] = list(rs), list(cs)
    return False


def _render(rng, ts, distance, h, iw, sw):
    """Construction labels: pairwise equal endpoints, independently of solve."""
    palette = list(map(int, rng.permutation(10)))
    bg, divider, frame = palette[:3]
    a = np.full((h, sw + iw + 3), bg, dtype=np.int8)
    a[:, sw] = divider
    a[[0, -1], sw+1:] = frame
    a[:, [sw+1, -1]] = frame
    a[1, :distance] = divider
    # Random slack between catalog bounding boxes; source order is irrelevant.
    heights = []
    for t in ts:
        offsets = _offsets(t["vector"], t["scale"])
        heights.append(max(r for r, _ in offsets) - min(r for r, _ in offsets) + 1)
    slack = h - 4 - sum(heights) - (len(ts)-1)
    gaps = rng.multinomial(slack, np.full(len(ts)+1, 1/(len(ts)+1)))
    top = 4 + int(gaps[0])
    for i, (t, bh) in enumerate(zip(ts, heights)):
        color = palette[i+3]
        t["color"] = color
        mask = _offsets(t["vector"], t["scale"])
        min_r, min_c = min(r for r, _ in mask), min(c for _, c in mask)
        bw = max(c for _, c in mask) - min_c + 1
        left = int(rng.integers(sw-bw+1))
        for r, c in mask:
            a[top+r-min_r, left+c-min_c] = color
        for r in t["rows"]:
            a[r, sw+1] = a[r, -1] = color
        for c in t["cols"]:
            a[0, sw+2+c] = a[-1, sw+2+c] = color
        top += bh + 1 + int(gaps[i+1])
    # Descriptor-derived output: compare every virtual tip's destination to all
    # other tips' destinations, then copy the supplied mask at the launch site.
    population = []
    for t in ts:
        vr, vc = t["vector"]
        for r in t["rows"]:
            for c in t["cols"]:
                population.append((t, r, c, r+distance*vr, c+distance*vc))
    out = a.copy()
    for i, (t, r, c, er, ec) in enumerate(population):
        if any(j != i and (er, ec) == (br, bc)
               for j, (_, _, _, br, bc) in enumerate(population)):
            for y, x in _offsets(t["vector"], t["scale"]):
                out[r+y, sw+2+c+x] = t["color"]
    return {"input": a, "output": out}


def _make_pair(rng, distance, kind):
    n = 3 if kind == "teach_three" else int(rng.choice([2, 3])) if kind == "query" else 2
    pool = TRIPLE_DIRECTIONS if n == 3 else PAIR_DIRECTIONS
    for attempt in range(80):
        vectors = list(pool[int(rng.integers(len(pool)))])
        rng.shuffle(vectors)
        scales = [int(rng.choice([1, 2], p=[.7, .3])) for _ in vectors]
        if kind == "teach_three":
            scales[int(rng.integers(n))] = 2
        ts = [dict(color=i+3, vector=v, scale=s, rows=[-distance*v[0]], cols=[-distance*v[1]])
              for i, (v, s) in enumerate(zip(vectors, scales))]
        inactive = bool(kind != "teach_cartesian" and rng.random() < .35)
        extra = dict(color=n+3, vector=VECTORS[int(rng.integers(8))], scale=1, rows=[], cols=[])
        all_ts = ts + ([extra] if inactive else [])
        sw = max(distance+1, max(2*t["scale"]+1 for t in all_ts)) + int(rng.integers(3))
        base = _instances(ts, distance)
        points = [p for rec in base for p in rec["cells"]] + [(0, 0)]
        lo = [min(p[k] for p in points) for k in (0, 1)]
        hi = [max(p[k] for p in points) for k in (0, 1)]
        stage_height = 4 + len(all_ts)-1 + sum(
            max(r for r, _ in _offsets(t["vector"], t["scale"])) -
            min(r for r, _ in _offsets(t["vector"], t["scale"])) + 1 for t in all_ts)
        min_h = max(stage_height, hi[0]-lo[0]+5, 13)
        min_iw = max(hi[1]-lo[1]+5, 10)
        if min_h > 30 or min_iw > 27-sw:
            continue
        h = int(rng.integers(min_h, min(30, min_h+7)+1))
        iw = int(rng.integers(min_iw, 28-sw))
        qr = int(rng.integers(1-lo[0], h-1-hi[0]))
        qc = int(rng.integers(-lo[1], iw-hi[1]))
        for t in ts:
            t["rows"] = [t["rows"][0] + qr]
            t["cols"] = [t["cols"][0] + qc]
        if not _valid(ts, distance, h, iw):
            continue
        if kind == "query_empty":
            # A simpler held-out case: two valid launch sites, no rendezvous.
            # The rule stays identical and the unchanged field is informative.
            changed = False
            for delta in (1, -1, 2, -2):
                old_row = ts[1]["rows"][0]
                ts[1]["rows"] = [old_row + delta]
                if _valid(ts, distance, h, iw):
                    changed = True
                    break
                ts[1]["rows"] = [old_row]
            if not changed:
                continue
        if kind == "teach_cartesian" and not _off_diagonal_witness(rng, ts, distance, h, iw):
            continue
        if kind not in ("teach_cartesian", "query_empty") and rng.random() < .6:
            _second_site(rng, ts, distance, h, iw)
        # Additional axes are independent by key and axis, subject only to
        # fitting and readability. They may create new rendezvous as well.
        for _ in range(0 if kind == "query_empty" else int(rng.integers(2, 9))):
            i = int(rng.integers(n))
            axis = "rows" if rng.integers(2) else "cols"
            _add_extra(rng, ts, distance, h, iw, i, axis,
                       limit=2 if kind == "teach_cartesian" else 3)
        if inactive:
            state = int(rng.integers(3))
            if state:
                axis = "rows" if state == 1 else "cols"
                free = [v for v in (range(1, h-1) if state == 1 else range(iw))
                        if all(v not in t[axis] for t in ts)]
                if free:
                    extra[axis] = [free[int(rng.integers(len(free)))]]
        pair = _render(rng, all_ts, distance, h, iw, sw)
        parsed = _parse(pair["input"])
        correct = _prediction(parsed)
        assert np.array_equal(correct, pair["output"]), "construction / parser mismatch"
        f = _features(parsed)
        if kind == "teach_cartesian" and not (f["zip_difference"] and f["partial_keys"] and f["rejected"]):
            continue
        if kind == "teach_three" and not (f["max_quorum"] == 3 and f["reverse_difference"]):
            continue
        return pair
    raise RuntimeError(f"bounded geometry search exhausted for {kind}")


def _features(p):
    records = p["records"]
    counts = Counter(x["endpoint"] for x in records)
    kept = [x for x in records if counts[x["endpoint"]] >= 2]
    per_key = Counter(x["color"] for x in records)
    kept_key = Counter(x["color"] for x in kept)
    out = _prediction(p)
    return dict(shape=p["grid"].shape, distance=p["distance"], strip=p["d"],
                roles=(p["bg"], p["divider"], p["frame"]), keys=len(p["templates"]),
                directions=tuple(t["vector"] for t in p["templates"]),
                scales=tuple(t["scale"] for t in p["templates"]),
                products=tuple((len(t["rows"]), len(t["cols"])) for t in p["templates"]),
                virtual=len(records), selected=len(kept), rejected=len(records)-len(kept),
                sites=sum(n >= 2 for n in counts.values()), max_quorum=max(counts.values(), default=0),
                partial_keys=sum(0 < kept_key[c] < n for c, n in per_key.items()),
                zip_difference=int(np.count_nonzero(out != _prediction(p, policy="zip"))),
                all_difference=int(np.count_nonzero(out != _prediction(p, materialize_all=True))),
                reverse_difference=int(np.count_nonzero(out != _prediction(p, distance=-p["distance"]))),
                inactive=sum(not t["rows"] or not t["cols"] for t in p["templates"]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed % (1 << 64))
    distances = list(map(int, rng.permutation([3, 4, 5])))
    specs = [(distances[0], "teach_cartesian"), (distances[1], "teach_three")]
    specs += [(distances[int(rng.integers(2))], "ordinary") for _ in range(nt-2)]
    rng.shuffle(specs)
    puzzle = {"train": [], "test": []}
    seen = set()
    queries = [(distances[2], "query_empty" if rng.random() < .15 else "query") for _ in range(ne)]
    for split, choices in (("train", specs), ("test", queries)):
        for distance, kind in choices:
            pair = _make_pair(rng, distance, kind)
            if _key(pair["input"]) in seen:
                # Twenty distinct bijections exceed the thirteen prior inputs.
                p = _parse(pair["input"])
                original = {k: v.copy() for k, v in pair.items()}
                for i in range(20):
                    mapping = np.arange(10, dtype=np.int8)
                    if i >= 10:
                        mapping[p["bg"]], mapping[p["divider"]] = p["divider"], p["bg"]
                    mapping = (mapping + i % 10) % 10
                    pair = {k: mapping[v] for k, v in original.items()}
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise AssertionError("distinct-palette fallback exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, fs = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            p = _parse(a)
            assert np.array_equal(solve(a.copy()), b), "wrong selected mask or preservation"
            assert np.array_equal(a[:, :p["d"]+2], b[:, :p["d"]+2]), "catalog and divider preserved"
            assert np.array_equal(a[[0, -1]], b[[0, -1]]) and np.array_equal(a[:, -1], b[:, -1]), "border preserved"
            f = _features(p)
            assert (split == "test" or f["selected"] > 0) and f["max_quorum"] <= 3
            assert np.count_nonzero(a != b) == 3*f["selected"], "entire disjoint masks only"
            fs[split].append(f)
    train = fs["train"]
    assert any(f["zip_difference"] and f["partial_keys"] for f in train), "teach Cartesian selection within a key"
    assert any(f["all_difference"] for f in train), "teach rejection of unique endpoints"
    assert any(f["max_quorum"] == 3 for f in train), "teach at least two rather than exactly two"
    assert any(f["reverse_difference"] for f in train), "teach forward tip projection"
    assert len({f["distance"] for f in train}) == 2, "teach the visible ruler at two lengths"
    assert not {f["distance"] for f in train} & {f["distance"] for f in fs["test"]}, "held-out ruler length"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming 42-seed diversity, counterfactual, and contract checks."""
    # Literal pixel regression: off-diagonal Cartesian survivor, then a triple.
    # Expected cells are handwritten, not made by _offsets or _render.
    a = np.zeros((16, 22), dtype=np.int8)
    a[:, 4] = 1
    a[[0, -1], 5:] = 9
    a[:, [5, -1]] = 9
    a[1, :3] = 1
    for cell in ((4, 1), (5, 0), (5, 2)):
        a[cell] = 2
    for cell in ((8, 2), (7, 1), (9, 1)):
        a[cell] = 3
    a[[8, 12], 5] = a[[8, 12], -1] = 2
    a[[0, -1], 10] = a[[0, -1], 16] = 2
    a[5, 5] = a[5, -1] = 3
    a[[0, -1], 13] = 3
    expected = a.copy()
    for cell in ((8, 16), (9, 15), (9, 17)):
        expected[cell] = 2
    for cell in ((5, 13), (4, 12), (6, 12)):
        expected[cell] = 3
    assert np.array_equal(solve(a), expected) and expected[5, 16] == 0
    for cell in ((13, 1), (12, 1), (13, 2)):
        a[cell] = expected[cell] = 4
    a[2, 5] = a[2, -1] = expected[2, 5] = expected[2, -1] = 4
    a[[0, -1], 19] = expected[[0, -1], 19] = 4
    for cell in ((2, 19), (1, 19), (2, 20)):
        expected[cell] = 4
    assert np.array_equal(solve(a), expected), "all three participants survive"
    shapes, palettes, signatures, vectors, sizes, products_seen = set(), set(), set(), set(), set(), set()
    virtual, sites, selected, key_counts = set(), set(), set(), set()
    partial = inactive = mult = empty = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            p = _parse(pair["input"])
            f = _features(p)
            shapes.add(f["shape"]); palettes.add(f["roles"])
            vectors.update(f["directions"]); sizes.update(f["scales"])
            products_seen.update(f["products"])
            virtual.add(f["virtual"]); sites.add(f["sites"])
            selected.add(f["selected"]); key_counts.add(f["keys"])
            partial += bool(f["partial_keys"])
            inactive += bool(f["inactive"])
            mult += f["sites"] > 1
            empty += f["selected"] == 0
            # Arbitrary palette names do not affect the rule; input is untouched.
            a = pair["input"].copy()
            mapping = (np.arange(10, dtype=np.int8) + 3) % 10
            assert np.array_equal(solve(mapping[a]), mapping[pair["output"]])
            assert np.array_equal(a, pair["input"])
        # Corrupting a label to either omitted-parent shortcut must be rejected.
        if seed == 0:
            for policy in ("all", "zip"):
                bad = {s: [{k: v.copy() for k, v in x.items()} for x in puzzle[s]] for s in puzzle}
                index = next(i for i, x in enumerate(bad["train"])
                             if _features(_parse(x["input"]))["all_difference" if policy == "all" else "zip_difference"])
                p = _parse(bad["train"][index]["input"])
                bad["train"][index]["output"] = _prediction(p, materialize_all=policy == "all",
                                                           policy="zip" if policy == "zip" else "cartesian")
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("omitted-parent mutation accepted")
    assert len(signatures) == 42 and len(shapes) >= 35 and len(palettes) >= 80
    assert vectors == set(VECTORS) and sizes == {1, 2} and key_counts == {2, 3, 4}
    assert {(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)} <= products_seen
    assert len(virtual) >= 10 and len(selected) >= 5 and len(sites) >= 3
    assert partial >= 42 and inactive >= 10 and mult >= 20 and empty > 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(733, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _fingerprint(generate_puzzle(902, 2, 1)) == _fingerprint(generate_puzzle(902, 2, 1))
    assert _fingerprint(generate_puzzle(902)) != _fingerprint(generate_puzzle(903))


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 streaming seeds and all legal explicit counts)")
