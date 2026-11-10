"""Compositional keyed-L relocation puzzles."""

import numpy as np


_ORIENTATION = {"TL": (1, 1), "TR": (1, -1), "BL": (-1, 1), "BR": (-1, -1)}
_SOURCE_PAIRS = (
    (("0777000000000", "0017002000080", "0007000000000", "0000000000000", "0444000000000", "0054000060000", "0004000009900", "0000000009600", "0000000009000", "3333000000000", "0083000000000", "0003000000000"),
     ("0777000000000", "0017002033330", "0007000000830", "0000000000030", "0444000000000", "0054000099000", "0004000096000", "0000000090000", "0000000000000", "0000000000000", "0000000000000", "0000000000000")),
    (("000000000000", "007000000000", "000000000000", "000000000000", "000000000000", "000000666000", "000000670000", "000000600000", "088800000000", "009800000000", "000800000000", "000000000000"),
     ("000000000000", "006660000000", "006700000000", "006000000000", "000000000000", "000000000000", "000000000000", "000000000000", "088800000000", "009800000000", "000800000000", "000000000000")),
    (("0000300000000", "0000350000000", "0000333000000", "0000000000000", "0000007000000", "0000007300000", "5000007777000", "0000000000000", "0000900000860", "0000000000888", "0000000000000", "0000000060000"),
     ("0000000000000", "0000000000000", "0000000000000", "0000000000000", "3000007000000", "3500007300000", "3330007777000", "0000000000000", "0000900000000", "0000000000000", "0000000086000", "0000000088800")),
)


def _check_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _resolve_counts(rng, num_train, num_test):
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _carrier_cells(elbow, vs, hs, hl, vl):
    r, c = elbow
    return {(r, c + hs * x) for x in range(hl)} | {(r + vs * y, c) for y in range(1, vl)}


def _inside(cells, height, width):
    return all(0 <= r < height and 0 <= c < width for r, c in cells)


def _four_close(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))


def _on_ray(position, obj):
    dr = position[0] - obj["elbow"][0]
    dc = position[1] - obj["elbow"][1]
    return dr * -obj["vs"] > 0 and dc * -obj["hs"] > 0 and dr * -obj["vs"] == dc * -obj["hs"]


def _colors(rng, n, background):
    palette = [x for x in range(10) if x != background]
    carriers = []
    for _ in range(n):
        unused = [x for x in palette if x not in carriers]
        pool = unused if unused and rng.random() < 0.68 else palette
        carriers.append(int(rng.choice(pool)))
    keys = []
    for i in range(n):
        choices = [x for x in palette if x != carriers[i]]
        overlap = [x for x in carriers if x in choices]
        repeated = [x for x in keys if x in choices]
        u = rng.random()
        pool = overlap if overlap and u < 0.37 else repeated if repeated and u < 0.65 else choices
        keys.append(int(rng.choice(pool)))
    return carriers, keys


def _distance(rng, height, width):
    cap = min(13, height - 3, width - 3)
    u = rng.random()
    if u < 0.38:
        lo, hi = 1, min(3, cap)
    elif u < 0.69 and cap >= 4:
        lo, hi = 4, min(7, cap)
    elif cap >= 8:
        lo, hi = 8, cap
    else:
        lo, hi = 1, cap
    return int(rng.integers(lo, hi + 1))


def _layout(rng, height, width, orientations, active, carriers, keys, distractor_count, background):
    objects = []
    sources, destinations, beacons = set(), set(), set()
    for i, orientation in enumerate(orientations):
        vs, hs = _ORIENTATION[orientation]
        max_h = min(8, width - 3)
        max_v = min(8, height - 3)
        hl = int(rng.integers(2, max_h + 1))
        vl = int(rng.integers(2, max_v + 1))
        if rng.random() < 0.45:
            hl = min(hl, 4)
        if rng.random() < 0.45:
            vl = min(vl, 4)
        chosen = None
        # Randomized positions and distances jointly avoid a fixed edge or distance pattern.
        for flat in rng.permutation(height * width):
            elbow = (int(flat) // width, int(flat) % width)
            carrier = _carrier_cells(elbow, vs, hs, hl, vl)
            key = (elbow[0] + vs, elbow[1] + hs)
            source = carrier | {key}
            if not _inside(source, height, width) or _four_close(source, sources | beacons) or source & destinations:
                continue
            beacon = None
            destination = set()
            if i in active:
                d = _distance(rng, height, width)
                beacon = (elbow[0] - d * vs, elbow[1] - d * hs)
                destination = {(r - d * vs, c - d * hs) for r, c in source}
                if not _inside(destination | {beacon}, height, width):
                    continue
                if _four_close({beacon}, sources | beacons):
                    continue
                if destination & (sources | destinations | beacons):
                    continue
            chosen = {"elbow": elbow, "vs": vs, "hs": hs, "hl": hl, "vl": vl,
                      "carrier": carrier, "key": key, "source": source,
                      "carrier_color": carriers[i], "key_color": keys[i],
                      "beacon": beacon, "destination": destination}
            break
        if chosen is None:
            return None
        objects.append(chosen)
        sources.update(source)
        destinations.update(destination)
        if beacon is not None:
            beacons.add(beacon)
    # Every isolated matching cell on an outward ray must be the intended beacon.
    if any(sum(pos is not None and color == obj["key_color"] and _on_ray(pos, obj)
               for pos, color in ((other["beacon"], other["key_color"]) for other in objects)) != (obj["beacon"] is not None)
           for obj in objects):
        return None
    distractors = []
    occupied = sources | beacons
    for _ in range(distractor_count):
        used_roles = set(carriers) | set(keys)
        novel = [x for x in range(10) if x != background and x not in used_roles]
        choice = rng.random()
        if choice < 0.43:
            color = int(rng.choice(keys))
        elif choice < 0.72:
            color = int(rng.choice(carriers))
        else:
            color = int(rng.choice(novel)) if novel else int(rng.choice(keys))
        placed = None
        for flat in rng.permutation(height * width):
            pos = (int(flat) // width, int(flat) % width)
            if pos in destinations or _four_close({pos}, occupied):
                continue
            if any(color == obj["key_color"] and _on_ray(pos, obj) for obj in objects):
                continue
            placed = pos
            break
        if placed is None:
            return None
        distractors.append((placed, color))
        occupied.add(placed)
    return objects, distractors


def _make_pair(rng, regime, index, train_count, test_count, coverage):
    if regime == "source_train" and index < 2:
        n = int(rng.integers(2 if index == 0 else 3, 5))
        named = list(coverage[2 * index:2 * index + 2])
        orientations = named + [str(rng.choice(tuple(_ORIENTATION))) for _ in range(n - 2)]
        active = {0, 1}
    else:
        n = int(rng.choice((1, 2, 3, 4, 5, 6), p=(0.12, 0.18, 0.25, 0.24, 0.14, 0.07)))
        orientations = [str(rng.choice(tuple(_ORIENTATION))) for _ in range(n)]
        k = int(rng.integers(0, n + 1))
        if regime == "target_test" and test_count == 1:
            k = max(1, k)
        active = set(int(x) for x in rng.choice(n, size=k, replace=False))
    # The two guaranteed demonstrations include both active and inactive objects.
    if regime == "source_train" and index == 1:
        n = max(n, 3)
    floor = 11 if n == 1 else 13 if n <= 3 else 16 if n == 4 else 22
    if n <= 3 and rng.random() < 0.25:
        height = int(rng.integers(floor, 17))
        width = int(rng.integers(floor, 17))
    else:
        height = int(rng.integers(floor, 31))
        width = int(rng.integers(floor, 31))
    if regime == "target_test" and rng.random() < 0.38:
        height = int(rng.integers(max(floor, 19), 31))
        width = int(rng.integers(max(floor, 19), 31))
    background = int(rng.integers(0, 10))
    carriers, keys = _colors(rng, n, background)
    distractor_count = int(rng.choice((0, 1, 2, 3, 4), p=(0.26, 0.28, 0.23, 0.16, 0.07)))
    for _ in range(90):
        layout = _layout(rng, height, width, orientations, active, carriers, keys, distractor_count, background)
        if layout is None:
            continue
        objects, distractors = layout
        inp = np.full((height, width), background, dtype=np.int8)
        for obj in objects:
            for r, c in obj["carrier"]:
                inp[r, c] = obj["carrier_color"]
            inp[obj["key"]] = obj["key_color"]
            if obj["beacon"] is not None:
                inp[obj["beacon"]] = obj["key_color"]
        for pos, color in distractors:
            inp[pos] = color
        out = inp.copy()
        for obj in objects:
            if obj["beacon"] is not None:
                for r, c in obj["source"]:
                    out[r, c] = background
        for obj in objects:
            if obj["beacon"] is None:
                continue
            dr = obj["beacon"][0] - obj["elbow"][0]
            dc = obj["beacon"][1] - obj["elbow"][1]
            for r, c in obj["carrier"]:
                out[r + dr, c + dc] = obj["carrier_color"]
            out[obj["key"][0] + dr, obj["key"][1] + dc] = obj["key_color"]
        return {"input": inp, "output": out}
    raise ValueError("could not place pair")


def _components(grid, bg):
    height, width = grid.shape
    seen, components = set(), []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            todo, comp = [(r, c)], set()
            seen.add((r, c))
            while todo:
                p = todo.pop()
                comp.add(p)
                for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                    if 0 <= q[0] < height and 0 <= q[1] < width and q not in seen and int(grid[q]) == color:
                        seen.add(q)
                        todo.append(q)
            components.append((color, comp))
    return components


def _find_l(comp):
    solutions = []
    for elbow in comp:
        rows = {p for p in comp if p[0] == elbow[0]}
        cols = {p for p in comp if p[1] == elbow[1]}
        if len(rows) < 2 or len(cols) < 2 or len(rows) + len(cols) - 1 != len(comp):
            continue
        for vs, hs in _ORIENTATION.values():
            if _carrier_cells(elbow, vs, hs, len(rows), len(cols)) == comp:
                solutions.append((elbow, vs, hs, len(rows), len(cols)))
    return solutions[0] if len(solutions) == 1 else None


def _decode(inp):
    height, width = inp.shape
    counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(np.argmax(counts))
    comps = _components(inp, bg)
    sizes = {p: len(comp) for _, comp in comps for p in comp}
    objects, carrier_cells, key_cells = [], set(), set()
    for color, comp in comps:
        if len(comp) == 1:
            continue
        found = _find_l(comp)
        assert found is not None
        elbow, vs, hs, hl, vl = found
        key = (elbow[0] + vs, elbow[1] + hs)
        assert 0 <= key[0] < height and 0 <= key[1] < width
        key_color = int(inp[key])
        assert key_color not in (bg, color)
        assert key not in key_cells and key not in carrier_cells
        key_cells.add(key)
        assert not comp & carrier_cells
        carrier_cells.update(comp)
        objects.append({"carrier": comp, "carrier_color": color, "elbow": elbow, "vs": vs, "hs": hs,
                        "hl": hl, "vl": vl, "key": key, "key_color": key_color})
    assert objects and not carrier_cells & key_cells
    for obj in objects:
        assert sizes[obj["key"]] == 1
        candidates = [p for color, comp in comps if color == obj["key_color"] and len(comp) == 1
                      for p in comp if p not in key_cells and _on_ray(p, obj)]
        assert len(candidates) <= 1
        obj["beacon"] = candidates[0] if candidates else None
        obj["source"] = obj["carrier"] | {obj["key"]}
    result = inp.copy()
    active = [obj for obj in objects if obj["beacon"] is not None]
    destinations = set()
    for obj in active:
        dr = obj["beacon"][0] - obj["elbow"][0]
        dc = obj["beacon"][1] - obj["elbow"][1]
        dest = {(r + dr, c + dc) for r, c in obj["source"]}
        assert _inside(dest, height, width) and not dest & destinations
        assert all(int(inp[p]) == bg or p == obj["beacon"] or p in obj["source"] for p in dest)
        destinations.update(dest)
        obj["distance"] = abs(dr)
        for p in obj["source"]:
            result[p] = bg
    for obj in active:
        dr = obj["beacon"][0] - obj["elbow"][0]
        dc = obj["beacon"][1] - obj["elbow"][1]
        for r, c in obj["carrier"]:
            result[r + dr, c + dc] = obj["carrier_color"]
        result[obj["key"][0] + dr, obj["key"][1] + dc] = obj["key_color"]
    singleton_positions = {p for _, comp in comps if len(comp) == 1 for p in comp}
    distractors = singleton_positions - key_cells - {obj["beacon"] for obj in active}
    return result, {"background": bg, "objects": objects, "active": active, "distractors": distractors,
                    "components": comps, "shape": inp.shape}


def _grid_signature(grid):
    return grid.shape, grid.tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            sig = _grid_signature(inp)
            assert sig not in seen
            seen.add(sig)
            rows_in = tuple("".join(str(int(x)) for x in row) for row in inp)
            rows_out = tuple("".join(str(int(x)) for x in row) for row in out)
            assert all(rows_in != s_in or rows_out != s_out for s_in, s_out in _SOURCE_PAIRS)
            predicted, analysis = _decode(inp)
            assert np.array_equal(predicted, out)
            analyses[split].append(analysis)
    train = analyses["train"]
    assert {(_ORIENTATION[o]) for o in _ORIENTATION} <= {(obj["vs"], obj["hs"]) for a in train for obj in a["active"]}
    assert any(len(a["active"]) < len(a["objects"]) for a in train)
    assert any(a["active"] for a in train)
    if len(puzzle["test"]) == 1:
        assert analyses["test"][0]["active"]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The invariant diagonal/key meaning is the sole puzzle-wide convention.
    coverage = tuple(str(x) for x in rng.permutation(tuple(_ORIENTATION)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, regime in (("train", train_count, "source_train"), ("test", test_count, "target_test")):
        for index in range(count):
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                try:
                    pair = _make_pair(pair_rng, regime, index, train_count, test_count, coverage)
                except ValueError:
                    continue
                signature = _grid_signature(pair["input"])
                if signature in seen:
                    continue
                try:
                    predicted, _ = _decode(pair["input"])
                    if not np.array_equal(predicted, pair["output"]):
                        continue
                except AssertionError:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("could not create an unambiguous unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    def signature(puzzle):
        return tuple((_grid_signature(p["input"]), _grid_signature(p["output"]))
                     for split in ("train", "test") for p in puzzle[split])
    assert signature(generate_puzzle(812345, 6, 3)) == signature(generate_puzzle(812345, 6, 3))
    assert signature(generate_puzzle(812345, 6, 3)) != signature(generate_puzzle(812346, 6, 3))
    for seed, t, e in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, None, 1),
                       (105, None, 4), (106, 2, None), (107, 10, None)):
        p = generate_puzzle(seed, t, e)
        assert t is None or len(p["train"]) == t
        assert e is None or len(p["test"]) == e
    for t, e in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(999, t, e)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    train_counts, test_counts, shapes, arm_profiles, orientations, moves, object_counts = set(), set(), set(), set(), set(), set(), set()
    repeated_roles = zero_decoys = novel_decoys = no_moves = test_larger = one_object = many_objects = 0
    mixed = bottom_right_test = offaxis = multi_offaxis = near_edge = all_active = compact = large = 0
    repeated_carrier = repeated_key = reverse_overlap = forward_overlap = 0
    within_backgrounds = same_orientation_train_test = 0
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(signature(p))
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        train_analyses = [_decode(q["input"])[1] for q in p["train"]]
        test_analyses = [_decode(q["input"])[1] for q in p["test"]]
        test_larger += max(a["shape"][0] for a in test_analyses) > max(a["shape"][0] for a in train_analyses)
        test_larger += max(a["shape"][1] for a in test_analyses) > max(a["shape"][1] for a in train_analyses)
        within_backgrounds += len({a["background"] for a in train_analyses + test_analyses}) > 1
        train_orientations = {(o["vs"], o["hs"]) for a in train_analyses for o in a["objects"]}
        test_orientations = {(o["vs"], o["hs"]) for a in test_analyses for o in a["objects"]}
        same_orientation_train_test += bool(train_orientations & test_orientations)
        tagged = [(pair, a, "train") for pair, a in zip(p["train"], train_analyses)]
        tagged += [(pair, a, "test") for pair, a in zip(p["test"], test_analyses)]
        for pair, a, split in tagged:
            objects = a["objects"]
            n = len(objects)
            one_object += n == 1
            many_objects += n > 4
            object_counts.add(n)
            shapes.add(a["shape"])
            no_moves += len(a["active"]) == 0
            all_active += len(a["active"]) == n
            zero_decoys += len(a["distractors"]) == 0
            role_colors = {x["carrier_color"] for x in objects} | {x["key_color"] for x in objects}
            novel_decoys += any(int(pair["input"][pos]) not in role_colors for pos in a["distractors"])
            orientations_here = {(x["vs"], x["hs"]) for x in objects}
            mixed += len(orientations_here) > 1
            bottom_right_test += split == "test" and (-1, -1) in orientations_here
            near_edge += any(min(x["beacon"][0], x["beacon"][1], a["shape"][0]-1-x["beacon"][0], a["shape"][1]-1-x["beacon"][1]) <= 1 for x in a["active"])
            compact += max(a["shape"]) <= 15
            large += min(a["shape"]) >= 25
            repeated_carrier += len({x["carrier_color"] for x in objects}) < n
            repeated_key += len({x["key_color"] for x in objects}) < n
            reverse_overlap += any(x["carrier_color"] == y["key_color"] for x in a["active"] for y in objects if y not in a["active"] and x is not y)
            forward_overlap += any(x["key_color"] == y["carrier_color"] for x in a["active"] for y in objects if y not in a["active"] and x is not y)
            for x in objects:
                if x["beacon"] is not None:
                    continue
                matches = sum(int(pair["input"][pos]) == x["key_color"] for pos in a["distractors"])
                offaxis += matches > 0
                multi_offaxis += matches >= 2
            for obj in objects:
                arm_profiles.add((obj["hl"], obj["vl"]))
                orientations.add((obj["vs"], obj["hs"]))
            moves.update(obj["distance"] for obj in a["active"])
            repeated_roles += len({x["carrier_color"] for x in objects}) < n or len({x["key_color"] for x in objects}) < n or bool({x["carrier_color"] for x in objects} & {x["key_color"] for x in objects})
    assert len(signatures) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(arm_profiles) >= 20 and len(orientations) == 4
    assert max(moves) >= 8 and min(moves) == 1 and object_counts >= {1, 2, 3, 4, 5}
    assert repeated_roles > 20 and zero_decoys > 10 and novel_decoys > 0
    assert no_moves > 0 and test_larger > 0 and one_object > 0 and many_objects > 0
    assert mixed > 0 and bottom_right_test > 0 and offaxis > 0 and multi_offaxis > 0
    assert near_edge > 0 and all_active > 0 and compact > 0 and large > 0
    assert repeated_carrier > 0 and repeated_key > 0 and reverse_overlap > 0 and forward_overlap > 0
    assert within_backgrounds > 0 and same_orientation_train_test > 0
