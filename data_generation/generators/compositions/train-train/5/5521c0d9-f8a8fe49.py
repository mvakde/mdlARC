"""Jumping mirrors: body-length motion changes a nearest-mirror partition.

Only the sampler uses randomness.  solve() reads geometric roles from the input.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import json

import numpy as np


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer color grid of size 1..30")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        pending = deque([first])
        cells = {first}
        while pending:
            r, c = pending.popleft()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    cells.add(point)
                    pending.append(point)
        result.append(cells)
    return result


def _rectangle(cells):
    rs, cs = zip(*cells)
    r0, r1, c0, c1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
    if len(cells) != (r1 - r0) * (c1 - c0):
        return None
    return tuple(map(int, (r0, r1, c0, c1)))


def _analyze(input_grid):
    """Recognize the pair of edge-supported mirrors, then use canonical rows.

    In canonical coordinates the normal axis is rows, with mirrors at the
    top/bottom edges.  No generator parameters or output cells are consulted.
    """
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if sum(counts == counts[bg]) != 1:
        raise ValueError("background must be uniquely most frequent")
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        groups = _components(grid == color)
        if len(groups) != 2:
            continue
        boxes = [_rectangle(cells) for cells in groups]
        if any(box is None for box in boxes):
            continue
        for transpose in (False, True):
            h, w = grid.T.shape if transpose else grid.shape
            bs = [(c0, c1, r0, r1) if transpose else (r0, r1, c0, c1)
                  for r0, r1, c0, c1 in boxes]
            bs.sort()
            first, second = bs
            if (first[0] == 0 and second[1] == h
                    and first[2:] == second[2:]):
                d0, d1 = first[1], h - second[0]
                t0, t1 = first[2:]
                if t1 - t0 >= max(5, d0 + 2, d1 + 2):
                    candidates.append((color, transpose, d0, d1, t0, t1))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous pair of rectangular mirrors")
    color, transpose, d0, d1, t0, t1 = candidates[0]
    canonical = grid.T.copy() if transpose else grid.copy()
    n = canonical.shape[0]
    a, b = 2 * d0 - 1, n - 2 * d1
    if a >= b:
        raise ValueError("moved mirrors must leave a chamber")
    payload = np.argwhere((canonical != bg) & (canonical != color))
    if len(payload) == 0:
        raise ValueError("missing payload")
    destinations = []
    sides, switched = [], []
    old_sum = d0 - 1 + n - d1
    for q, t in payload:
        q, t = int(q), int(t)
        if not (a < q < b and t0 < t < t1 - 1) or 2 * q == a + b:
            raise ValueError("payload must have a unique nearer moved face")
        side = 0 if 2 * q < a + b else 1
        target = 2 * (a if side == 0 else b) - q
        # The reflected image lands in space vacated by its owner.  This
        # excludes mirror collisions and makes the two motions exchange space.
        if not ((side == 0 and 0 <= target < d0)
                or (side == 1 and n - d1 <= target < n)):
            raise ValueError("reflection must fit in the vacated mirror footprint")
        destinations.append((target, t, int(canonical[q, t])))
        sides.append(side)
        switched.append(2 * q != old_sum and (2 * q < old_sum) != (side == 0))
    if set(sides) != {0, 1}:
        raise ValueError("both mirrors must receive payload")
    targets = {(q, t) for q, t, _ in destinations}
    if len(targets) != len(destinations):
        raise ValueError("reflected cells must be distinct")
    return dict(grid=grid, canonical=canonical, transpose=transpose,
                background=bg, mirror=color, d0=d0, d1=d1, t0=t0, t1=t1,
                a=a, b=b, payload=payload, destinations=destinations,
                sides=sides, switched=switched)


def _render_analysis(item, operation="full"):
    src = item["canonical"]
    bg, mirror = item["background"], item["mirror"]
    n = src.shape[0]
    d0, d1, t0, t1 = (item[k] for k in ("d0", "d1", "t0", "t1"))
    result = np.full_like(src, bg)
    if operation == "old_faces":
        result[:d0, t0:t1] = mirror
        result[n-d1:, t0:t1] = mirror
        for q, t in item["payload"]:
            face = d0 - 1 if 2 * q < d0 - 1 + n - d1 else n - d1
            dest = 2 * face - int(q)
            if 0 <= dest < n:
                result[dest, t] = src[q, t]
    else:
        result[d0:2*d0, t0:t1] = mirror
        result[n-2*d1:n-d1, t0:t1] = mirror
        if operation == "move_only":
            for q, t in item["payload"]:
                result[q, t] = src[q, t]
        elif operation == "old_ownership":
            old_sum = d0 - 1 + n - d1
            for q, t in item["payload"]:
                axis = item["a"] if 2*q < old_sum else item["b"]
                target = 2*axis-int(q)
                if 0 <= target < n:
                    result[target, t] = src[q, t]
        else:
            for q, t, color in item["destinations"]:
                result[q, t] = color
    return result.T.copy() if item["transpose"] else result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move the mirrors, partition payload by their new faces, then reflect."""
    return _render_analysis(_analyze(input_grid))


@lru_cache(maxsize=1)
def _geometries():
    """Geometric sampler vocabulary; solve() does not use this table."""
    result = []
    for n in range(10, 31):
        for d0 in range(1, 6):
            for d1 in range(1, 6):
                a, b = 2 * d0 - 1, n - 2 * d1
                if a >= b:
                    continue
                left = tuple(q for q in range(a + 1, b)
                             if 2*q < a+b and 0 <= 2*a-q < d0)
                right = tuple(q for q in range(a + 1, b)
                              if 2*q > a+b and n-d1 <= 2*b-q < n)
                if not left or not right:
                    continue
                old_sum = d0 - 1 + n - d1
                switches = tuple(q for q in left + right
                                 if 2*q != old_sum
                                 and (2*q < old_sum) != (2*q < a+b))
                bridge = left[-1] + 1 == right[0]
                result.append((n, d0, d1, left, right, switches, bridge))
    return tuple(result)


def _shape(rng, qs, span, style):
    """An eight-connected, asymmetric mask with the requested exact extents."""
    qs = list(qs)
    if len(qs) == 1:
        return {(qs[0], t) for t in range(span)}
    if style == "elbow":
        return {(q, 0) for q in qs} | {(qs[0], t) for t in range(span)}
    if style == "zigzag":
        cells = {(qs[t % len(qs)], t) for t in range(span)}
        # Connect the normal extents without imposing a straight tangential spine.
        cells |= {(q, 0) for q in qs}
        for t in range(1, span):
            old, new = qs[(t-1) % len(qs)], qs[t % len(qs)]
            cells |= {(q, t) for q in range(min(old, new), max(old, new)+1)}
        return cells
    if style == "hollow" and len(qs) >= 3 and span >= 3:
        return {(q, t) for q in qs for t in range(span)
                if q in (qs[0], qs[-1]) or t in (0, span-1)}
    # A full first normal column and a wandering path preserve connectivity;
    # randomized extra cells vary density without replacing the silhouette.
    cells = {(q, 0) for q in qs}
    index = int(rng.integers(len(qs)))
    for t in range(1, span):
        index = max(0, min(len(qs)-1, index + int(rng.integers(-1, 2))))
        cells.add((qs[index], t))
    for q in qs:
        for t in range(1, span):
            if rng.random() < .22 and any(abs(q-r) <= 1 and abs(t-u) <= 1
                                        for r, u in cells):
                cells.add((q, t))
    return cells


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_pair(rng, kind, rotation, palette):
    choices = list(_geometries())
    if kind == "switch_lesson":
        choices = [g for g in choices if max(g[1:3]) <= 3 and g[5]
                   and max(len(g[3]), len(g[4])) >= 2]
    elif kind == "split_lesson":
        choices = [g for g in choices if max(g[1:3]) <= 3 and g[6]]
    elif kind == "large":
        choices = [g for g in choices if max(g[1:3]) >= 4]
    elif kind == "recombine":
        choices = [g for g in choices if g[5] and g[6]]
    elif kind == "ordinary":
        choices = [g for g in choices if max(g[1:3]) <= 3]
    geometry = _pick(rng, choices)
    n, d0, d1, left, right, switches, can_bridge = geometry
    bg, mirror, *payload_colors = palette
    bridge = kind in ("split_lesson", "recombine") or (can_bridge and rng.random() < .4)
    descriptors = ["bridge"] if bridge else ["left", "right"]
    extra_count = 5-len(descriptors) if kind == "more_motifs" else int(rng.integers(0, 3))
    # A switch lesson has an explicit asymmetric motif at the switched cell.
    # A split lesson has an asymmetric connected bridge, not two adjacent bars.
    descriptors += [_pick(rng, ("left", "right")) for _ in range(extra_count)]
    rng.shuffle(descriptors)
    spans = [int(rng.integers(2, 6 if kind in ("large", "recombine", "more_motifs") else 5))
             for _ in descriptors]
    # Budget all tangential gaps and margins before rendering.
    while sum(spans) + len(spans) + 1 > 24:
        index = int(np.argmax(spans))
        spans[index] -= 1
    length = max(max(d0, d1) + 2, 5, sum(spans) + len(spans) + 1)
    length += int(rng.integers(0, min(3, 26-length) + 1))
    tangent0 = int(rng.integers(1, 3))
    tangent1 = int(rng.integers(1, min(3, 30-length-tangent0) + 1))
    width = tangent0 + length + tangent1
    inp = np.full((n, width), bg, dtype=np.int8)
    out = inp.copy()
    inp[:d0, tangent0:tangent0+length] = mirror
    inp[n-d1:, tangent0:tangent0+length] = mirror
    out[d0:2*d0, tangent0:tangent0+length] = mirror
    out[n-2*d1:n-d1, tangent0:tangent0+length] = mirror
    cursor = tangent0 + 1
    available = length - (sum(spans) + len(spans) + 1)
    spacing = rng.multinomial(available, [1/(len(spans)+1)] * (len(spans)+1))
    cursor += int(spacing[0])
    forced_switch_done = False
    for index, (descriptor, span) in enumerate(zip(descriptors, spans)):
        if descriptor == "bridge":
            qs = tuple(range(left[-1], right[0]+1))
            # Broaden either side independently, always retaining the bridge.
            if rng.random() < .7:
                qs = tuple(range(_pick(rng, left), _pick(rng, right)+1))
            style = "elbow"
        else:
            allowed = left if descriptor == "left" else right
            if kind == "switch_lesson" and not forced_switch_done and any(q in switches for q in allowed):
                qs = allowed
                forced_switch_done = True
                style = "elbow"
            else:
                first = int(rng.integers(len(allowed)))
                last = int(rng.integers(first+1, len(allowed)+1))
                qs = allowed[first:last]
                style = _pick(rng, ("elbow", "zigzag", "sparse", "hollow"))
        cells = _shape(rng, qs, span, style)
        # Recombination geometry has a switched boundary cell; the bridge
        # includes it.  Same color on both sides rules out color routing.
        color = payload_colors[0] if kind in ("switch_lesson", "split_lesson") else int(_pick(rng, payload_colors))
        a, b = 2*d0-1, n-2*d1
        for q, t in sorted(cells):
            t += cursor
            target = 2 * (a if 2*q < a+b else b) - q
            assert inp[q, t] == bg and out[target, t] == bg
            inp[q, t] = color
            out[target, t] = color
        cursor += span + 1 + int(spacing[index+1])
    if rotation:
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle, including two informative demonstrations."""
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(10, size=5, replace=False))
    varying_palette = bool(rng.random() < .5)
    rotation = int(rng.integers(4))
    mixed_orientation = bool(rng.random() < .6)
    train_rotations = [rotation, (rotation+1) % 4 if mixed_orientation else rotation]
    test_kind = _pick(rng, ("large", "recombine", "more_motifs"))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                kind = ("switch_lesson" if index == 0 else "split_lesson" if index == 1
                        else "ordinary")
                turn = train_rotations[index % 2]
            else:
                kind = test_kind
                turn = int(_pick(rng, train_rotations))
            for _ in range(80):
                colors = palette
                if varying_palette and index != 0:
                    colors = tuple(int(c) for c in rng.choice(10, size=5, replace=False))
                pair = _make_pair(rng, kind, turn, colors)
                key = _key(pair["input"])
                if key in seen:
                    continue
                item = _analyze(pair["input"])
                if kind == "switch_lesson" and not any(item["switched"]):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct distinct examples")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(item):
    src = item["canonical"]
    mask = (src != item["background"]) & (src != item["mirror"])
    components = _components(mask, diagonal=True)
    split = any(any(2*q < item["a"]+item["b"] for q, _ in cells)
                and any(2*q > item["a"]+item["b"] for q, _ in cells)
                for cells in components)
    asymmetric = False
    for cells in components:
        for side in (0, 1):
            group = {(q, t) for q, t in cells
                     if (2*q < item["a"]+item["b"]) == (side == 0)}
            if not group:
                continue
            low, high = min(q for q, _ in group), max(q for q, _ in group)
            if {(low+high-q, t) for q, t in group} != group:
                asymmetric = True
    return dict(shape=tuple(src.shape), orientation=int(item["transpose"]),
                depths=(item["d0"], item["d1"]), length=item["t1"]-item["t0"],
                split=split, switched=any(item["switched"]),
                asymmetric=asymmetric, motifs=len(components),
                roles=(item["background"], item["mirror"]),
                payload_colors=tuple(sorted(set(int(src[q, t]) for q, t in item["payload"]))))


def validate_generated_puzzle(puzzle):
    """Check geometry, conservation, exchanged footprints and lesson witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _integer("num_train", len(puzzle["train"]), 2, 10)
    _integer("num_test", len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            item = _analyze(inp)
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10))
            normal_out = out.T if item["transpose"] else out
            # Erasure, preserved reflector area, and exact image locations.
            assert all(normal_out[q, t] == item["background"] for q, t in item["payload"])
            assert all(normal_out[q, t] == color for q, t, color in item["destinations"])
            assert not np.array_equal(inp, out)
            assert not np.array_equal(_render_analysis(item, "old_faces"), out)
            assert not np.array_equal(_render_analysis(item, "move_only"), out)
            features[split].append(_features(item))
    train = features["train"]
    assert any(f["switched"] and f["asymmetric"] for f in train)
    assert any(f["split"] for f in train)
    assert {f["orientation"] for f in features["test"]} <= {f["orientation"] for f in train}
    # A test either extrapolates mirror thickness or adds a fifth independent
    # motif.  No training scene contains either of those test conditions.
    assert all(max(f["depths"]) <= 3 and f["motifs"] <= 4 for f in train)
    assert all(max(f["depths"]) >= 4 or f["motifs"] == 5 for f in features["test"])
    return features


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded audit of 42 seeds, all legal counts, ablations and covariance."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17))
    records = [f for p in puzzles for fs in validate_generated_puzzle(p).values() for f in fs]
    shapes = {p["input"].shape for puzzle in puzzles for s in puzzle for p in puzzle[s]}
    assert len(shapes) >= 45
    assert {f["orientation"] for f in records} == {0, 1}
    assert len({f["depths"] for f in records}) >= 15
    assert {d for f in records for d in f["depths"]} == {1, 2, 3, 4, 5}
    assert len({f["motifs"] for f in records}) >= 3
    assert len({f["roles"] for f in records}) >= 25
    assert sum(f["split"] and f["switched"] for f in records) >= 5
    assert any(0 in f["payload_colors"] and f["roles"][0] != 0 for f in records)
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(2026 + 10*train + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {train}/{test}")
    # Rotation and color permutations must commute with the input-only rule.
    permutation = np.array([7, 2, 9, 0, 6, 1, 5, 8, 4, 3], dtype=np.int8)
    for seed in (0, 7, 17, 41):
        for pair in puzzles[seed]["train"][:2]:
            for turn in range(4):
                inp, out = np.rot90(pair["input"], turn), np.rot90(pair["output"], turn)
                assert np.array_equal(solve(permutation[inp]), permutation[out])
    # Move and reflection ablations are exercised by every generated pair.
    # Explicitly prove that old-face ownership would route a witnessed cell
    # to the OTHER mirror, even if those mirrors were still moved correctly.
    for puzzle in puzzles:
        item = _analyze(puzzle["train"][0]["input"])
        assert any(item["switched"])
        assert not np.array_equal(_render_analysis(item, "old_ownership"),
                                  puzzle["train"][0]["output"])
        split_item = _analyze(puzzle["train"][1]["input"])
        assert _features(split_item)["split"]
    normalized = set()
    for puzzle in puzzles:
        for s in puzzle:
            for pair in puzzle[s]:
                roles = {}
                data = [roles.setdefault(int(c), len(roles)) for c in pair["input"].flat]
                normalized.add((pair["input"].shape, bytes(data)))
    assert len(normalized) >= 195
    # A valid family member outside the sampler's 1..5 thickness vocabulary.
    inp = np.zeros((30, 14), dtype=np.int8)
    expected = np.zeros_like(inp)
    inp[:6, 2:12] = inp[28:, 2:12] = 2
    expected[6:12, 2:12] = expected[26:28, 2:12] = 2
    for q, t in ((17, 4), (18, 4), (17, 5), (23, 8), (24, 9)):
        inp[q, t] = 5
        expected[22-q if q < 19 else 52-q, t] = 5
    assert np.array_equal(solve(inp), expected)
    return {"seeds": 42, "pairs": len(records), "grid_shapes": len(shapes),
            "depth_pairs": len({f["depths"] for f in records}),
            "normalized_inputs": len(normalized),
            "split_and_switch_pairs": sum(f["split"] and f["switched"] for f in records),
            "semantic_verification": "generator-owned invariants; no independent solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
