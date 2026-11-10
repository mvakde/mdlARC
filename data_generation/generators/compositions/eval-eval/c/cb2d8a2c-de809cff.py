"""Cancelling repair signals steer a boundary-seeded courier.

All inference is input-only. Construction plans are used only to draw fresh
inputs and labels; neither solve nor validation reads those plans.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _runs(indices):
    groups = []
    for x in map(int, indices):
        if not groups or x != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(x)
    return groups


def _paint(shape, background, boxes):
    a = np.full(shape, background, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        a[r0:r1, c0:c1] = color
    return a


def _fit_gate(observed, background, fills):
    """Dense axis projections locate intact rectangle extents, not speckles.

    The family retains >=3 observed cells on each material's occupied row and
    >=2 on each occupied column. False-color outliers cannot meet those
    thresholds outside its true box. Occlusion leaves both boxes exposed.
    """
    boxes = []
    for color in fills:
        mask = observed == color
        rows = np.flatnonzero(mask.sum(axis=1) >= 3)
        cols = np.flatnonzero(mask.sum(axis=0) >= 2)
        assert len(rows) >= 4 and len(cols) >= 5, "Insufficient material evidence"
        assert len(rows) == rows[-1] - rows[0] + 1
        assert len(cols) == cols[-1] - cols[0] + 1
        boxes.append((int(rows[0]), int(cols[0]), int(rows[-1]) + 1,
                      int(cols[-1]) + 1, int(color)))
    left, right = sorted(boxes, key=lambda b: b[1])
    assert left[1] < right[1] < left[3] < right[3]
    assert right[1] - left[1] >= 3 and right[3] - left[3] >= 3
    early, late = sorted(boxes, key=lambda b: b[0])
    assert early[0] + 2 <= late[0] < early[2] <= late[2] - 2
    assert early[2] - late[0] >= 2
    assert min(b[0] for b in boxes) == 0
    assert max(b[2] for b in boxes) == observed.shape[0]
    options = []
    for ordered in (boxes, boxes[::-1]):
        scene = _paint(observed.shape, background, ordered)
        loss = int(np.count_nonzero(scene != observed))
        options.append((loss, scene, ordered))
    options.sort(key=lambda item: item[0])
    assert options[1][0] > options[0][0] + 1, "Ambiguous covering order"
    loss, clean, ordered = options[0]
    assert loss <= .30 * np.count_nonzero(clean != background) + 3
    for r0, c0, r1, c1, color in boxes:
        visible = clean[r0:r1, c0:c1] == color
        correct = observed[r0:r1, c0:c1] == color
        assert np.count_nonzero(visible & correct) >= .70 * visible.sum()
    return clean, (left, right), tuple(ordered)


def _ring_demands(shape, clean, centers, fills):
    centers = set(centers)
    demands = {}
    for r, c in sorted(centers):
        opposite = fills[1] if clean[r, c] == fills[0] else fills[0]
        for rr in range(max(0, r - 1), min(shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(shape[1], c + 2)):
                q = rr, cc
                if q in centers:
                    continue
                assert q not in demands or demands[q] == opposite, "Conflicting rings"
                demands[q] = opposite
    return demands


def _render(clean, centers, fills, route_color, path):
    result = clean.copy()
    demands = _ring_demands(clean.shape, clean, centers, fills)
    assert not any(path[q] for q in demands), "Route touches a repair symbol"
    assert not any(path[q] for q in centers)
    for q, color in demands.items():
        result[q] = color
    for q in centers:
        result[q] = route_color
    result[path] = route_color
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    corners = (int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1]))
    assert len(set(corners)) == 1
    background = corners[0]
    edge = np.zeros(a.shape, dtype=bool)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    seeds = np.argwhere(edge & (a != background))
    assert seeds.shape == (1, 2), "Exactly one boundary seed is required"
    r, c = map(int, seeds[0])
    assert sum((r == 0, c == 0, r == h - 1, c == w - 1)) == 1
    route = int(a[r, c])
    assert np.count_nonzero(a == route) == 1
    rotation = 0 if r == 0 else 1 if c == 0 else 2 if r == h - 1 else 3
    a = np.rot90(a, -rotation).copy()
    fills = tuple(sorted(set(map(int, np.unique(a))) - {background, route}))
    assert len(fills) == 2
    foreground = np.isin(a, fills)
    groups = _runs(np.flatnonzero(foreground.sum(axis=1) >= 4))
    assert groups
    clean = np.full(a.shape, background, dtype=np.int8)
    initial = int(np.flatnonzero(a[0] == route)[0])
    clean[0, initial] = route
    gates, centers = [], []
    for group in groups:
        top, stop = group[0], group[-1] + 1
        assert top > 0 and stop < a.shape[0]
        scene, sides, ordered = _fit_gate(a[top:stop], background, fills)
        clean[top:stop] = scene
        holes = [tuple(map(int, p)) for p in np.argwhere(
            (scene != background) & (a[top:stop] == background))]
        counts = tuple(int(sum(scene[p] == color for p in holes)) for color in fills)
        assert counts[0] != counts[1], "Balanced gates have no command"
        surplus = abs(counts[0] - counts[1])
        # An erasure requests the OTHER material. Cancel opposite requests.
        demand_color = fills[1] if counts[0] > counts[1] else fills[0]
        side = -1 if sides[0][4] == demand_color else 1
        lo, hi = sides[0][1], sides[1][3] - 1
        d = surplus + 1
        local_centers = [(r + top, c) for r, c in holes]
        centers.extend(local_centers)
        gates.append({"top": top, "stop": stop, "lo": lo, "hi": hi,
                      "counts": counts, "surplus": surplus, "side": side,
                      "clearance": d, "centers": local_centers,
                      "boxes": tuple((r0 + top, c0, r1 + top, c1, color)
                                     for r0, c0, r1, c1, color in ordered)})
    path = np.zeros(a.shape, dtype=bool)
    current, start, previous_end = initial, 0, -1
    for gate in gates:
        turn = gate["top"] - gate["clearance"]
        target = (gate["lo"] - gate["clearance"] if gate["side"] == -1
                  else gate["hi"] + gate["clearance"])
        assert 1 <= turn and turn > previous_end and turn >= start
        assert 0 <= target < a.shape[1]
        assert gate["lo"] <= current <= gate["hi"], "Gate misses incoming ray"
        assert np.any(np.isin(clean[gate["top"]:gate["stop"], current], fills))
        for rr in range(start, turn + 1):
            path[rr, current] = True
        step = 1 if target > current else -1
        for cc in range(current, target + step, step):
            path[turn, cc] = True
        current, start, previous_end = target, turn, gate["stop"] - 1
        gate["turn"], gate["target"] = turn, target
    path[start:, current] = True
    assert not np.any(path & np.isin(clean, fills))
    result = _render(clean, centers, fills, route, path)
    errors = a != clean
    return {"expected": np.rot90(result, rotation).copy(), "a": a,
            "clean": clean, "background": background, "route": route,
            "fills": fills, "rotation": rotation, "gates": gates,
            "centers": centers, "path": path,
            "wrong": int(np.count_nonzero(errors & np.isin(clean, fills) & (a != background))),
            "speckles": int(np.count_nonzero(errors & (clean == background)))}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all colors, rectangles, holes, commands and route from this input."""
    return _analyze(input_grid)["expected"]


def _span_plan(rng, width, deltas, forced):
    failed = set()

    def search(index, current):
        if index == len(deltas):
            return []
        if (index, current) in failed:
            return None
        d = deltas[index] + 1
        options = []
        for lo in range(1, current + 1):
            for hi in range(max(current, lo + 7), min(width - 1, lo + 20)):
                for side in (-1, 1):
                    if index in forced and side != forced[index]:
                        continue
                    target = lo - d if side == -1 else hi + d
                    if 0 <= target < width:
                        options.append((lo, hi, side, target))
        for j in rng.permutation(len(options)):
            lo, hi, side, target = options[int(j)]
            rest = search(index + 1, target)
            if rest is not None:
                return [(lo, hi, side, target)] + rest
        failed.add((index, current))
        return None

    for initial in rng.permutation(np.arange(1, width - 1)):
        result = search(0, int(initial))
        if result is not None:
            return int(initial), result
    return None


def _gate_boxes(rng, height, lo, hi, fills):
    length = hi - lo + 1
    options = [(lw, overlap) for lw in range(5, 12) for overlap in range(2, 6)
               if 5 <= length - lw + overlap <= 11
               and lw - overlap >= 3 and length - lw >= 3]
    assert options
    lw, overlap = options[int(rng.integers(len(options)))]
    offset = int(rng.integers(2, min(3, height - 4) + 1))
    end = int(rng.integers(offset + 2, height - 1))
    left = (0, lo, end, lo + lw, fills[0])
    right = (offset, lo + lw - overlap, height, hi + 1, fills[1])
    boxes = [left, right]
    if rng.integers(2):
        boxes = [(height - r1, c0, height - r0, c1, color)
                 for r0, c0, r1, c1, color in boxes]
    if rng.integers(2):
        boxes.reverse()
    return boxes


def _sample_holes(rng, clean, background, fills, counts, adjacent):
    chosen = []
    damaged = clean.copy()

    def available(p, previous):
        color = int(clean[p])
        if damaged[p] == background:
            return False
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2 for q in previous):
            return False
        return (np.count_nonzero(damaged[p[0]] == color) > 3
                and np.count_nonzero(damaged[:, p[1]] == color) > 2
                and np.count_nonzero(damaged[p[0]] != background) > 4)

    # Placing the larger load first leaves room for a compatible cluster.
    order = sorted(range(2), key=lambda i: -counts[i])
    for i in order:
        color = fills[i]
        pool = [tuple(map(int, p)) for p in np.argwhere(clean == color)]
        rng.shuffle(pool)
        selected = []
        if adjacent and i == order[0]:
            pool_set = set(pool)
            pairs = [(p, q) for p in pool for q in ((p[0], p[1] + 1), (p[0] + 1, p[1]))
                     if q in pool_set]
            rng.shuffle(pairs)
            for p, q in pairs:
                if not available(p, chosen):
                    continue
                damaged[p] = background
                if available(q, chosen):
                    damaged[q] = background
                    selected.extend((p, q))
                    break
                damaged[p] = color
            if not selected:
                return None
        for p in pool:
            if len(selected) >= counts[i]:
                break
            if p in selected:
                continue
            if not available(p, chosen):
                continue
            selected.append(p)
            damaged[p] = background
        if len(selected) != counts[i]:
            return None
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2
               for p in selected for q in chosen):
            return None
        chosen.extend(selected)
    return damaged, chosen


def _make_pair(rng, palette, rotation, deltas, forced, bases, teach_cluster,
               require_noise=False):
    background, route, *fills = palette
    width = int(rng.integers(16, 31))
    plan = _span_plan(rng, width, deltas, forced)
    if plan is None:
        width = 30
        plan = _span_plan(rng, width, deltas, forced)
    assert plan is not None
    initial, spans = plan
    heights = list(map(int, rng.integers(6, 10, size=len(deltas))))
    while sum(h + d + 1 for h, d in zip(heights, deltas)) + len(deltas) + 1 > 30:
        choices = [i for i, h in enumerate(heights) if h > 6]
        assert choices
        heights[int(rng.choice(choices))] -= 1
    minimum = sum(h + d + 1 for h, d in zip(heights, deltas)) + len(deltas) + 1
    height = int(rng.integers(minimum, 31))
    allocation = rng.multinomial(height - minimum, np.full(len(deltas) + 1, 1 / (len(deltas) + 1)))
    top, previous_end = 0, -1
    geometry = []
    clean = np.full((height, width), background, dtype=np.int8)
    clean[0, initial] = route
    path = np.zeros(clean.shape, dtype=bool)
    start, current = 0, initial
    for i, (delta, gh, span) in enumerate(zip(deltas, heights, spans)):
        lo, hi, side, target = span
        turn = (1 if i == 0 else previous_end + 2) + int(allocation[i])
        top = turn + delta + 1
        material_order = list(fills)
        if rng.integers(2):
            material_order.reverse()
        local_boxes = _gate_boxes(rng, gh, lo, hi, material_order)
        local = _paint((gh, width), background, local_boxes)
        clean[top:top + gh] = local
        counts_by_side = [bases[i], bases[i]]
        counts_by_side[1 if side == -1 else 0] += delta
        counts = [counts_by_side[material_order.index(color)] for color in fills]
        geometry.append((top, local, local_boxes, counts))
        # Constructor uses slices; decoder walks a separately inferred polyline.
        path[start:turn + 1, current] = True
        path[turn, min(current, target):max(current, target) + 1] = True
        current, start, previous_end = target, turn, top + gh - 1
    path[start:, current] = True
    speckle_target = int(rng.integers(1, 5))
    for _ in range(64):
        observed = clean.copy()
        centers = []
        good = True
        for i, (top, local, local_boxes, counts) in enumerate(geometry):
            selection = _sample_holes(rng, local, background, fills, counts,
                                      teach_cluster and i == 0)
            if selection is None:
                good = False
                break
            damaged, holes = selection
            wrong_count = int(rng.integers(0, 3))
            if require_noise and i == 0:
                wrong_count = 1
            candidates = [tuple(map(int, p)) for p in np.argwhere(np.isin(local, fills))
                          if tuple(map(int, p)) not in set(holes)]
            rng.shuffle(candidates)
            wrong = []
            for p in candidates:
                if len(wrong) == wrong_count:
                    break
                if any(p[0] == q[0] or p[1] == q[1] for q in wrong):
                    continue
                color = int(local[p])
                if (np.count_nonzero(damaged[p[0]] == color) <= 3
                        or np.count_nonzero(damaged[:, p[1]] == color) <= 2):
                    continue
                damaged[p] = fills[1] if local[p] == fills[0] else fills[0]
                wrong.append(p)
            if require_noise and i == 0 and len(wrong) != 1:
                good = False
                break
            try:
                fitted, _, _ = _fit_gate(damaged, background, fills)
                assert np.array_equal(fitted, local)
                assert np.all(np.isin(damaged, fills).sum(axis=1) >= 4)
            except AssertionError:
                good = False
                break
            observed[top:top + len(local)] = damaged
            centers.extend((r + top, c) for r, c in holes)
        if not good:
            continue
        # Exterior outliers have no repeated row or column, so they cannot
        # masquerade as the dense support of a rectangle.
        exterior = [tuple(map(int, p)) for p in np.argwhere(clean == background)
                    if 0 < p[0] < height - 1 and 0 < p[1] < width - 1]
        rng.shuffle(exterior)
        speckles = []
        for p in exterior:
            if len(speckles) >= speckle_target:
                break
            if any(p[0] == q[0] or p[1] == q[1] for q in speckles):
                continue
            observed[p] = fills[int(rng.integers(2))]
            speckles.append(p)
        try:
            out = _render(clean, centers, fills, route, path)
            inp = np.rot90(observed, rotation).copy()
            answer = np.rot90(out, rotation).copy()
            analysis = _analyze(inp)
            assert np.array_equal(analysis["clean"], clean)
            assert np.array_equal(analysis["expected"], answer)
            if require_noise:
                assert analysis["wrong"] and analysis["speckles"]
            return {"input": inp, "output": answer}
        except AssertionError:
            continue
    raise AssertionError("Could not retain enough reconstruction evidence")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    recolor = bool(rng.integers(2))
    maximum = int(rng.integers(2, 4))
    shift = int(rng.integers(2))  # held-out heading or a longer gate sequence
    directions = list(map(int, rng.permutation(4)))
    counts = list(map(int, rng.choice((1, 2, 3) if shift == 0 else (1, 2), size=num_train)))
    pair_order = list(map(int, rng.permutation(num_train)))
    repeated_pair, other_pair = pair_order[:2]
    counts[repeated_pair] = max(2, counts[repeated_pair])
    repeat_at = int(rng.integers(counts[repeated_pair] - 1))
    first = (repeated_pair, repeat_at)
    second = (repeated_pair, repeat_at + 1)
    third = (other_pair, int(rng.integers(counts[other_pair])))
    sign = int(rng.choice((-1, 1)))
    # Equal surplus, unequal total: cancellation cannot be replaced by a
    # fixed total-hole offset. Repeated signs defeat compulsory zigzagging.
    requirements = {first: (1, 1, sign), second: (1, 2, sign),
                    third: (2, 2, -sign)}
    if maximum == 3:
        slots = [(i, j) for i, count in enumerate(counts) for j in range(count)
                 if (i, j) not in requirements
                 and sum(v[0] for (ii, _), v in requirements.items() if ii == i)
                 + 3 + count - sum(ii == i for ii, _ in requirements) - 1
                 <= 29 - 8 * count]
        if not slots:
            if counts[other_pair] == 1:
                counts[other_pair] = 2
                slots = [(other_pair, 1)]
            else:
                assert counts[repeated_pair] == 2
                counts[repeated_pair] = 3
                slots = [(repeated_pair, 2)]
        fourth = slots[int(rng.integers(len(slots)))]
        fourth_sign = -sign if fourth[0] == repeated_pair else int(rng.choice((-1, 1)))
        requirements[fourth] = (3, int(rng.integers(1, 3)), fourth_sign)
    # A same-fill pair is placed at the first gate of a random demonstration.
    cluster_pair = int(rng.integers(num_train))
    result, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            k = counts[i] if split == "train" else (3 if shift == 1 else int(rng.integers(1, 4)))
            deltas = list(map(int, rng.integers(1, maximum + 1, size=k)))
            bases = list(map(int, rng.integers(0, 3, size=k)))
            forced = {}
            for j in range(k):
                if split == "train" and (i, j) in requirements:
                    deltas[j], bases[j], forced[j] = requirements[i, j]
            while sum(deltas) + 8 * k + 1 > 30:
                adjustable = [j for j in range(k) if deltas[j] > 1
                              and not (split == "train" and (i, j) in requirements)]
                assert adjustable
                deltas[int(rng.choice(adjustable))] -= 1
            cluster = split == "train" and i == cluster_pair
            if cluster:
                bases[0] = max(1, bases[0])
            rotation = directions[3] if split == "test" and shift == 0 else int(rng.choice(directions[:3] if shift == 0 else directions))
            for _ in range(32):
                colors = tuple(map(int, rng.choice(10, 4, replace=False))) if recolor else palette
                try:
                    pair = _make_pair(rng, colors, rotation, deltas, forced, bases, cluster,
                                      require_noise=split == "train" and i == cluster_pair)
                except AssertionError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise AssertionError("Could not construct a distinct input")
    for split in result:
        result[split] = [result[split][int(i)] for i in rng.permutation(len(result[split]))]
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert np.array_equal(out, a["expected"]), "Incorrect input-derived label"
            # Every hole is a highlighted center; all other errors are repaired
            # before symbols and the collision-free route are painted.
            forward_out = np.rot90(out, -a["rotation"])
            assert all(forward_out[q] == a["route"] for q in a["centers"])
            analyses[split].append(a)
    train = analyses["train"]
    train_gates = [g for a in train for g in a["gates"]]
    learned = {g["surplus"] for g in train_gates}
    assert {1, 2} <= learned
    assert {g["side"] for g in train_gates} == {-1, 1}
    assert all(g["surplus"] in learned for a in analyses["test"] for g in a["gates"])
    assert any(min(g["counts"]) >= 2 for g in train_gates)
    assert any(a["wrong"] and a["speckles"] for a in train)
    corruption_changes_command = False
    for a in train:
        for g in a["gates"]:
            region = np.s_[g["top"]:g["stop"], :]
            clean, observed = a["clean"][region], a["a"][region]
            all_repairs = tuple(int(np.count_nonzero((clean == color) & (observed != clean)))
                                for color in a["fills"])
            true_counts = g["counts"]
            # (signed difference, magnitude) encodes both direction and distance;
            # a tie would also fail to supply the actual command.
            corruption_changes_command |= (all_repairs[0] - all_repairs[1]
                                           != true_counts[0] - true_counts[1])
    assert corruption_changes_command, "Ordinary repairs must not be valid signals"
    assert any(any(a["clean"][p] == a["clean"][q]
                   and max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                   for i, p in enumerate(a["centers"]) for q in a["centers"][:i])
               for a in train)
    # Ignoring cancellation changes clearance in a demonstration.
    assert any(sum(g["counts"]) != g["surplus"] for g in train_gates)
    assert any(x["surplus"] == y["surplus"] and sum(x["counts"]) != sum(y["counts"])
               for i, x in enumerate(train_gates) for y in train_gates[:i])
    assert any(any(x["side"] == y["side"] for x, y in zip(a["gates"], a["gates"][1:]))
               for a in train)
    # Ignoring complementarity chooses the opposite end at every unequal gate.
    assert all(g["counts"][0] != g["counts"][1] for g in train_gates)


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, layouts, directions, gate_counts = set(), set(), set(), set(), set()
    deltas, covering, repetitions, adjacency, noise = set(), set(), set(), 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                directions.add(a["rotation"])
                gate_counts.add(len(a["gates"]))
                canonical = np.zeros(a["a"].shape, dtype=np.int8)
                for i, color in enumerate((a["background"], a["route"], *a["fills"])):
                    canonical[a["a"] == color] = i
                layouts.add(_key(canonical))
                signs = [g["side"] for g in a["gates"]]
                repetitions.update(zip(signs, signs[1:]))
                for g in a["gates"]:
                    deltas.add(g["surplus"])
                    covering.add(g["boxes"][-1][1] > g["boxes"][0][1])
                adjacency += any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                                 for i, p in enumerate(a["centers"]) for q in a["centers"][:i])
                noise += bool(a["wrong"] and a["speckles"])
    assert len(signatures) == 42 and len(layouts) >= 200 and len(shapes) >= 50
    assert directions == {0, 1, 2, 3} and gate_counts == {1, 2, 3}
    assert deltas == {1, 2, 3} and covering == {False, True}
    assert repetitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert adjacency >= 42 and noise >= 42
    # A literal geometric regression with an independently specified polyline.
    # Holes in material 2 outnumber those in material 1 by two: the surplus
    # request is material 1, on the left. Turn three rows before row 6 and
    # pass three columns left of column 4.
    clean = _paint((18, 20), 0, [(6, 4, 10, 11, 1), (8, 8, 14, 16, 2)])
    clean[0, 8] = 8
    centers = [(7, 5), (7, 6), (10, 13), (11, 13), (11, 14), (12, 13)]
    inp = clean.copy()
    for p in centers:
        inp[p] = 0
    inp[8, 6] = 2
    inp[15, 10] = 1
    expected = clean.copy()
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expected[r + dr, c + dc] = 3 - int(clean[r, c])
    for p in centers:
        expected[p] = 8
    expected[0:4, 8] = 8
    expected[3, 1:9] = 8
    expected[3:, 1] = 8
    permutation = np.array([4, 9, 0, 6, 3, 5, 2, 8, 1, 7], dtype=np.int8)
    for rotation in range(4):
        for mirror in (False, True):
            x = inp[:, ::-1] if mirror else inp
            y = expected[:, ::-1] if mirror else expected
            assert np.array_equal(solve(np.rot90(permutation[x], rotation)),
                                  np.rot90(permutation[y], rotation))
    tied = inp.copy()
    tied[6, 4] = tied[6, 5] = 0
    try:
        solve(tied)
    except AssertionError as error:
        assert str(error) == "Balanced gates have no command"
    else:
        raise AssertionError("Balanced gate accepted")
    ambiguous = clean[6:14].copy()
    ambiguous[2, 8:11] = 1
    ambiguous[3, 8:11] = 2
    try:
        _fit_gate(ambiguous, 0, (1, 2))
    except AssertionError as error:
        assert str(error) == "Ambiguous covering order"
    else:
        raise AssertionError("Ambiguous covering order accepted")
    assert all(np.array_equal(p[role], q[role])
               for s in ("train", "test") for p, q in zip(puzzles[7][s], generate_puzzle(7)[s])
               for role in ("input", "output"))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(123 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Label mutation must not supply missing information to the input fitter.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    # Compatible symbols are independent of traversal order; centers win.
    clean = np.ones((9, 12), dtype=np.int8)
    clean[:, 7:] = 2
    centers = [(2, 2), (2, 3), (3, 3)]
    assert _ring_demands(clean.shape, clean, centers, (1, 2)) == _ring_demands(
        clean.shape, clean, centers[::-1], (1, 2))
    try:
        _ring_demands(clean.shape, clean, [(4, 6), (4, 7)], (1, 2))
    except AssertionError:
        pass
    else:
        raise AssertionError("Conflicting repair signals accepted")


if __name__ == "__main__":
    self_test()
