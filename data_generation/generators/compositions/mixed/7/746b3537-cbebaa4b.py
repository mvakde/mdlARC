"""Dock a translation-only chain, then collapse equal adjacent full vectors.

Background is 0; the unique color-1 payload is the stationary anchor. The
connector hue is the common hue of all separated two-color pieces. No state
from generation or demonstrations is consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        found, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    found.add(q)
                    todo.append(q)
        result.append(found)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _normal(cells):
    t, l, _, _ = _bbox(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _key(a):
    return a.shape, a.tobytes()


def compress(grid):
    """One representative of each adjacent run; compare complete vectors."""
    a = _grid(grid)
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return a[np.ix_(rows, cols)].copy()


def _analyze(a):
    groups = _components(map(tuple, np.argwhere(a != 0).tolist()))
    _require(3 <= len(groups) <= 6, "expected 3..6 separated pieces")
    palettes = [{int(a[q]) for q in group} for group in groups]
    common = set.intersection(*palettes)
    _require(len(common) == 1 and all(len(p) == 2 for p in palettes), "common connector hue is not unique")
    marker = common.pop()
    _require(marker != 1, "anchor and connector must differ")
    pieces, occurrences = [], {}
    for i, (group, colors) in enumerate(zip(groups, palettes)):
        color = (colors - {marker}).pop()
        body = {q for q in group if int(a[q]) == color}
        _require(len(_components(body)) == 1, "payload must be connected")
        t, l, b, r = _bbox(body)
        sides = {}
        for q in sorted(group - body):
            y, x = q
            if y < t and l <= x <= r:
                side = "N"
            elif y > b and l <= x <= r:
                side = "S"
            elif x < l and t <= y <= b:
                side = "W"
            elif x > r and t <= y <= b:
                side = "E"
            else:
                raise ValueError("connector must lie outside one payload side")
            sides.setdefault(side, set()).add(q)
        _require(1 <= len(sides) <= 2, "a chain piece has one or two docking sides")
        for side, cells in sorted(sides.items()):
            occurrences.setdefault(_normal(cells), []).append((i, side, cells))
        pieces.append(dict(body=body, pins=group - body, color=color, cells=group, sides=sides))
    anchors = [i for i, p in enumerate(pieces) if p["color"] == 1]
    _require(len(anchors) == 1, "exactly one color-1 anchor payload required")
    links = [[] for _ in pieces]
    edges = []
    for signature, ends in sorted(occurrences.items()):
        _require(len(ends) == 2, "every complete signature must occur twice")
        (i, si, pi), (j, sj, pj) = ends
        _require(i != j and OPPOSITE[si] == sj, "matching keys must face opposite sides")
        ti, li, _, _ = _bbox(pi)
        tj, lj, _, _ = _bbox(pj)
        delta = ti - tj, li - lj
        _require(_shift(pj, delta) == pi, "complete geometry must coincide")
        links[i].append((j, delta))
        links[j].append((i, (-delta[0], -delta[1])))
        edges.append((i, j, signature))
    _require(sorted(map(len, links)) == [1, 1] + [2] * (len(pieces) - 2), "docking graph must be a chain")
    return marker, pieces, links, anchors[0], edges


def _assemble(input_grid):
    a = _grid(input_grid)
    marker, pieces, links, anchor, edges = _analyze(a)
    positions, todo = {anchor: (0, 0)}, [anchor]
    while todo:
        i = todo.pop()
        for j, d in links[i]:
            proposed = positions[i][0] + d[0], positions[i][1] + d[1]
            if j in positions:
                _require(positions[j] == proposed, "inconsistent translation constraints")
            else:
                positions[j] = proposed
                todo.append(j)
    _require(len(positions) == len(pieces), "chain must use all pieces")
    assembled = np.zeros_like(a)
    payload_owner, pin_owners = {}, {}
    for i, p in enumerate(pieces):
        body, pins = _shift(p["body"], positions[i]), _shift(p["pins"], positions[i])
        for y, x in body | pins:
            _require(0 <= y < a.shape[0] and 0 <= x < a.shape[1], "assembly leaves canvas")
        for q in body:
            _require(q not in payload_owner, "payload overlap")
            payload_owner[q] = i
            assembled[q] = p["color"]
        for q in pins:
            pin_owners.setdefault(q, []).append(i)
    _require(not set(payload_owner) & set(pin_owners), "connector overlaps a payload")
    _require(all(len(owners) == 2 for owners in pin_owners.values()), "every connector must have two owners")
    actual_edges = Counter(tuple(sorted(v)) for v in pin_owners.values())
    _require(set(actual_edges) == {tuple(sorted((i, j))) for i, j, _ in edges}, "unexpected overlap")
    for q in pin_owners:
        assembled[q] = marker
    _require(len(_components(payload_owner.keys() | pin_owners.keys())) == 1, "assembled chain must be connected")
    return assembled, dict(marker=marker, pieces=pieces, positions=positions,
                          anchor=anchor, edges=edges, payload_owner=payload_owner)


def solve(input_grid: np.ndarray) -> np.ndarray:
    assembled, _ = _assemble(input_grid)
    return compress(assembled)


def _rectangle(t, l, h, w):
    return {(y, x) for y in range(t, t + h) for x in range(l, l + w)}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _layout(rng, count, straight=False):
    """Grow one chain of independently sized rectangles and sparse side keys."""
    for restart in range(48):
        h, w = map(int, rng.integers(3, 6, size=2))
        nodes = [dict(rect=_rectangle(0, 0, h, w), pins=set(), sides=set(), required=set())]
        edges, directions = [], []
        straight_side = _pick(rng, list(OPPOSITE))
        for i in range(1, count):
            parent = nodes[-1]
            t, l, b, r = _bbox(parent["rect"])
            added = False
            for attempt in range(100):
                side = straight_side if straight else _pick(rng, list(OPPOSITE))
                if side in parent["sides"]:
                    continue
                h, w = map(int, rng.integers(3, 6, size=2))
                if side in ("E", "W"):
                    top = int(rng.integers(t - h + 3, b - 1))
                    left = r + 2 if side == "E" else l - w - 1
                    overlap = list(range(max(t, top), min(b, top + h - 1) + 1))
                    if len(overlap) < 3:
                        continue
                    size = 2 if len(overlap) == 3 or rng.random() < 0.72 else 3
                    teeth = sorted(int(v) for v in rng.choice(overlap, size=size, replace=False))
                    if teeth[-1] - teeth[0] < 2:
                        continue
                    x = r + 1 if side == "E" else l - 1
                    pins = {(y, x) for y in teeth}
                else:
                    left = int(rng.integers(l - w + 3, r - 1))
                    top = b + 2 if side == "S" else t - h - 1
                    overlap = list(range(max(l, left), min(r, left + w - 1) + 1))
                    if len(overlap) < 3:
                        continue
                    size = 2 if len(overlap) == 3 or rng.random() < 0.72 else 3
                    teeth = sorted(int(v) for v in rng.choice(overlap, size=size, replace=False))
                    if teeth[-1] - teeth[0] < 2:
                        continue
                    y = b + 1 if side == "S" else t - 1
                    pins = {(y, x) for x in teeth}
                rect = _rectangle(top, left, h, w)
                occupied = set().union(*(p["rect"] | p["pins"] for p in nodes))
                if (rect | pins) & occupied:
                    continue
                # No contact with unrelated bodies/keys; only the intended side
                # may touch the preceding body. Empty holes are added later.
                unrelated = set().union(*(p["rect"] | p["pins"] for p in nodes[:-1]))
                if any((y + dy, x + dx) in unrelated for y, x in rect | pins for dy, dx in N4):
                    continue
                # Distinct complete keys, rather than distinct areas, identify
                # edges. Expansion can change keys; uniqueness is checked again.
                if any(_normal(pins) == _normal(old) for _, _, old in edges):
                    continue
                bt, bl, bb, br = _bbox(occupied | rect | pins)
                if bb - bt + 1 > 23 or br - bl + 1 > 23:
                    continue
                parent["pins"] |= pins
                parent["sides"].add(side)
                nodes.append(dict(rect=rect, pins=set(pins), sides={OPPOSITE[side]}, required=set()))
                edges.append((i - 1, i, pins))
                directions.append(side)
                added = True
                break
            if not added:
                break
        if len(nodes) == count:
            return nodes, edges, directions
    return None


def _body(rng, rect):
    t, l, b, r = _bbox(rect)
    # Full boundary ensures sparse keys remain attached, even with a hole.
    boundary = {q for q in rect if q[0] in (t, b) or q[1] in (l, r)}
    mode = int(rng.integers(3))
    if mode == 0:
        return set(rect)
    if mode == 1:
        return boundary
    return boundary | {q for q in sorted(rect - boundary) if rng.random() < 0.5}


def _inflate(rng, nodes, edges):
    occupied = set().union(*(p["rect"] | p["pins"] for p in nodes))
    t, l, b, r = _bbox(occupied)
    row_count = Counter(y for p in nodes for y in {q[0] for q in p["rect"]})
    col_count = Counter(x for p in nodes for x in {q[1] for q in p["rect"]})
    row_pool = sorted(y for y in row_count if row_count[y] >= 2) or sorted(row_count)
    col_pool = sorted(x for x in col_count if col_count[x] >= 2) or sorted(col_count)
    heights = {y: 1 for y in range(t, b + 1)}
    widths = {x: 1 for x in range(l, r + 1)}
    heights[_pick(rng, row_pool)] = int(rng.integers(2, 4))
    widths[_pick(rng, col_pool)] = int(rng.integers(2, 4))
    for mapping in (heights, widths):
        for _ in range(int(rng.integers(3))):
            if sum(mapping.values()) < 26:
                k = _pick(rng, sorted(mapping))
                mapping[k] = min(3, mapping[k] + 1)
    if sum(heights.values()) > 27 or sum(widths.values()) > 27:
        return None
    row_ranges, col_ranges = {}, {}
    cursor = 0
    for y, height in sorted(heights.items()):
        row_ranges[y] = range(cursor, cursor + height)
        cursor += height
    cursor = 0
    for x, width in sorted(widths.items()):
        col_ranges[x] = range(cursor, cursor + width)
        cursor += width

    def expand(cells):
        return {(a, b) for y, x in cells for a in row_ranges[y] for b in col_ranges[x]}

    pieces = [dict(body=expand(_body(rng, p["rect"])), pins=expand(p["pins"])) for p in nodes]
    keys = [_normal(expand(pins)) for _, _, pins in edges]
    if len(set(keys)) != len(keys):
        return None
    return pieces, (sum(heights.values()), sum(widths.values()))


def _scatter(rng, pieces, anchor, shape, marker):
    h, w = shape
    for restart in range(18):
        occupied = pieces[anchor]["body"] | pieces[anchor]["pins"]
        forbidden = {(y + dy, x + dx) for y, x in occupied for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        positions = {anchor: (0, 0)}
        order = [i for i in range(len(pieces)) if i != anchor]
        rng.shuffle(order)
        order.sort(key=lambda i: -len(pieces[i]["body"] | pieces[i]["pins"]))
        for i in order:
            cells = pieces[i]["body"] | pieces[i]["pins"]
            t, l, b, r = _bbox(cells)
            for attempt in range(200):
                d = int(rng.integers(h - b + t)) - t, int(rng.integers(w - r + l)) - l
                moved = _shift(cells, d)
                if d == (0, 0) or moved & forbidden:
                    continue
                positions[i] = d
                forbidden |= {(y + dy, x + dx) for y, x in moved for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                break
            else:
                break
        if len(positions) == len(pieces):
            a = np.zeros(shape, dtype=np.int8)
            for i, p in enumerate(pieces):
                for q in _shift(p["body"], positions[i]):
                    a[q] = p["color"]
                for q in _shift(p["pins"], positions[i]):
                    a[q] = marker
            return a
    return None


def _repeat_indices(a, axis, nonblank=False):
    vectors = a if axis == 0 else a.T
    return {i for i in range(1, len(vectors)) if np.array_equal(vectors[i - 1], vectors[i])
            and (not nonblank or bool(np.any(vectors[i] != 0)))}


def _recurs(a, axis):
    vectors = a if axis == 0 else a.T
    return any(np.any(vectors[i] != 0) and np.array_equal(vectors[i], vectors[j])
               for i in range(len(vectors)) for j in range(i + 2, len(vectors)))


def _premature_differs(a, expected):
    try:
        premature = solve(compress(a))
    except ValueError:
        return True
    return not np.array_equal(premature, expected)


def _features(a, assembled, info):
    out = compress(assembled)
    keys = [key for i, j, key in info["edges"]]
    coupled_axes = []
    for axis in (0, 1):
        shared = False
        newly_equal = _repeat_indices(assembled, axis, nonblank=True) - _repeat_indices(a, axis)
        for i in newly_equal:
            owners = {owner for q, owner in info["payload_owner"].items() if q[axis] == i}
            shared |= len(owners) >= 2
        coupled_axes.append(shared)
    return dict(piece_count=len(info["pieces"]), marker=info["marker"],
                canvas=tuple(a.shape), output_shape=tuple(out.shape),
                new_nonblank_equalities=[len(_repeat_indices(assembled, axis, True)
                                             - _repeat_indices(a, axis)) for axis in (0, 1)],
                nonblank_equalities=[len(_repeat_indices(assembled, axis, True)) for axis in (0, 1)],
                coupled_axes=coupled_axes,
                recurrence=[_recurs(out, axis) for axis in (0, 1)],
                equal_area_different_keys=any(len(x) == len(y) and x != y
                                             for i, x in enumerate(keys) for y in keys[:i]),
                key_shapes=[(_bbox(key)[2] + 1, _bbox(key)[3] + 1) for key in keys],
                key_areas=[len(key) for key in keys],
                body_shapes=[_normal(p["body"]) for p in info["pieces"]],
                payload_reuse=len(info["pieces"]) - len({p["color"] for p in info["pieces"]}),
                premature_differs=_premature_differs(a, out))


def _candidate(rng, count, marker, force_collision=False, force_bent=False):
    for attempt in range(180):
        layout = _layout(rng, count, straight=(not force_bent and count <= 4 and rng.random() < 0.18))
        if layout is None:
            continue
        nodes, edges, directions = layout
        if force_bent and len(set(directions)) == 1:
            continue
        inflated = _inflate(rng, nodes, edges)
        if inflated is None:
            continue
        pieces, (bh, bw) = inflated
        h = int(rng.integers(max(bh, 19 + count), 31))
        w = int(rng.integers(max(bw, 19 + count), 31))
        offset = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        anchor = int(rng.integers(count))
        palette = [c for c in range(2, 10) if c != marker]
        distinct = int(rng.integers(1, min(count - 1, len(palette)) + 1))
        colors = [int(v) for v in rng.choice(palette, size=distinct, replace=False)]
        movable = colors + [_pick(rng, colors) for _ in range(count - 1 - distinct)]
        rng.shuffle(movable)
        for i, p in enumerate(pieces):
            p["body"] = _shift(p["body"], offset)
            p["pins"] = _shift(p["pins"], offset)
            p["color"] = 1 if i == anchor else movable.pop()
        intended = np.zeros((h, w), dtype=np.int8)
        for p in pieces:
            for q in p["body"]:
                intended[q] = p["color"]
        for p in pieces:
            for q in p["pins"]:
                intended[q] = marker
        a = _scatter(rng, pieces, anchor, (h, w), marker)
        if a is None:
            continue
        try:
            assembled, info = _assemble(a)
        except ValueError:
            continue
        if not np.array_equal(assembled, intended):
            continue
        f = _features(a, assembled, info)
        if not all(f["new_nonblank_equalities"]) or not any(f["coupled_axes"]):
            continue
        if force_bent and not all(f["coupled_axes"]):
            continue
        if force_collision and not f["equal_area_different_keys"]:
            continue
        if not any(f["recurrence"]) or not f["premature_differs"]:
            continue
        out = compress(intended)
        if np.array_equal(compress(a), out):
            continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded chain construction exhausted its attempts")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(2, 10))
    puzzle, seen = {"train": [], "test": []}, set()
    # Two essential demonstrations survive even the smallest legal request.
    # Shuffle their positions so input order carries no assembly convention.
    counts = [3, 4] + [int(rng.integers(3, 6)) for _ in range(num_train - 2)]
    rng.shuffle(counts)
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            n = counts[index] if split == "train" else (6 if index == 0 else int(rng.integers(4, 7)))
            for retry in range(12):
                pair = _candidate(rng, n, marker, force_collision=(split == "train" and n == 4),
                                  force_bent=(split == "train" and n == 3))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test schema required")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact pair schema required")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            assembled, info = _assemble(a)
            _require(np.array_equal(compress(assembled), out), "incorrect compressed assembly")
            _require(np.array_equal(compress(out), out), "output is not run-minimal")
            f = _features(a, assembled, info)
            _require(all(f["new_nonblank_equalities"]), "docking must create nonblank equalities on both axes")
            _require(any(f["coupled_axes"]), "at least one repeated vector must span multiple payloads")
            _require(any(f["recurrence"]), "a separated nonblank recurrence must be retained")
            _require(f["premature_differs"], "premature compression shortcut is not distinguished")
            _require(not np.array_equal(compress(a), out), "omitting docking must change answer")
            _require(out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1], "both compression axes must be exercised")
            _require(all(d != (0, 0) for i, d in info["positions"].items() if i != info["anchor"]), "every movable piece must move")
            records[split].append(f)
    all_records = records["train"] + records["test"]
    _require(len({f["marker"] for f in all_records}) == 1, "connector hue is fixed within a puzzle")
    _require(any(all(f["coupled_axes"]) for f in records["train"]), "training must demonstrate two-axis coordinated alignment")
    _require(any(f["equal_area_different_keys"] for f in records["train"]), "training must distinguish geometry from key area")
    _require(len({f["piece_count"] for f in records["train"]}) >= 2, "training must vary chain length")
    _require(records["test"][0]["piece_count"] > max(f["piece_count"] for f in records["train"]), "first test extends chain length")
    return True


def _digest_puzzle(puzzle):
    data = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in puzzle[s]]
            for s in ("train", "test")}
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Streaming 42-seed coverage; no files, cached puzzles, or execution traces."""
    shapes, out_shapes, bodies, key_shapes, markers, signatures = set(), set(), set(), set(), set(), set()
    counts, couplings, reuse, pairs = set(), Counter(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest_puzzle(puzzle))
        if seed == 0:
            _require(_digest_puzzle(puzzle) == _digest_puzzle(generate_puzzle(seed)), "seed reproducibility failed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                assembled, info = _assemble(a)
                f = _features(a, assembled, info)
                shapes.add(tuple(a.shape))
                out_shapes.add(tuple(pair["output"].shape))
                bodies.update(f["body_shapes"])
                key_shapes.update(f["key_shapes"])
                markers.add(f["marker"])
                counts.add(f["piece_count"])
                couplings[sum(f["coupled_axes"])] += 1
                reuse += f["payload_reuse"] > 0
                pairs += 1
                _require(np.array_equal(solve(a.copy()), pair["output"]), "input-only solve mismatch")
    _require(len(signatures) == 42, "42 seeds must yield distinct puzzles")
    _require(len(shapes) >= 20 and len(out_shapes) >= 35, "insufficient geometric variation")
    _require(len(bodies) >= 100 and len(key_shapes) >= 12, "insufficient body/key variation")
    _require(counts == {3, 4, 5, 6} and len(markers) >= 7, "count or palette coverage missing")
    _require(reuse >= 30 and couplings[2] >= 42, "structural coverage missing")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4200 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Adjacent equality is not global uniqueness, cropping, or cell-wise runs.
    fixture = np.array([[0, 0, 0, 0, 0], [0, 3, 3, 0, 0], [0, 3, 3, 0, 0],
                        [0, 4, 4, 0, 0], [0, 3, 3, 0, 0], [0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 0], [0, 3, 0], [0, 4, 0], [0, 3, 0], [0, 0, 0]])
    _require(np.array_equal(compress(fixture), expected), "run semantics failed")
    p = generate_puzzle(91, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                output_shapes=len(out_shapes), body_masks=len(bodies), key_boxes=len(key_shapes),
                connector_hues=len(markers), payload_reuse_pairs=reuse, two_axis_coupling_pairs=couplings[2])


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
