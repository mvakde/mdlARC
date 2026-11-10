"""Strict motif union selects three pointed vessels for unary export.

Only the standard library and NumPy are used. Generation keeps no files or
execution traces. solve reads frame geometry and payload masks from its input.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


OPERAND_LABEL = 8
RESULT_LABEL = 3
DIRECTIONS = ("U", "R", "D", "L")
STEPS = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
HOST_SEEDS = (0, 271173194, 565051753, 112277855, 502944069, 442609125,
              631901345, 30501829, 958013464, 985883248, 670395636,
              757052402, 399684941, 81615435, 752573775, 157933034,
              179133099, 969485903, 697107490, 936664976, 159517423,
              443549973, 652965309, 709445792, 669692009, 881458741,
              575172806, 945707500, 328102150, 199183559, 415166960,
              226928745, 823269877, 185000867, 580350921, 662885131,
              209154033, 880880091, 83507079, 358048067, 246988387,
              884454483)


def _local(point, direction):
    r, c = point
    return {"U": (r, c), "R": (-c, r), "D": (-r, -c), "L": (c, -r)}[direction]


def _world(point, direction):
    y, x = point
    return {"U": (y, x), "R": (x, -y), "D": (-y, -x), "L": (-x, y)}[direction]


def _components(cells):
    todo = set(cells)
    answer = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, component = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    component.add(p)
                    stack.append(p)
        answer.append(component)
    return answer


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    if counts[index] <= grid.size // 2:
        raise ValueError("background must occupy more than half the input")
    return int(colors[index])


def _rays(shape, tip, direction, quantity):
    dr, dc = STEPS[direction]
    r, c = tip
    room = {"U": r, "R": shape[1] - 1 - c,
            "D": shape[0] - 1 - r, "L": c}[direction]
    return tuple((r + k * dr, c + k * dc) for k in range(1, min(room, quantity) + 1)), room


def _geometry(frame, direction):
    """Recognize monotone thin roofs and the maximal parallel-sided bay."""
    local = {_local(p, direction) for p in frame}
    y0 = min(y for y, x in local)
    x0 = min(x for y, x in local)
    height = max(y for y, x in local) - y0 + 1
    width = max(x for y, x in local) - x0 + 1
    if not (4 <= width <= 6 and 4 <= height <= 11):
        return None
    rows = [sorted(x - x0 for y, x in local if y == y0 + j) for j in range(height)]
    if len(rows[0]) != 1 or not 0 < rows[0][0] < width - 1:
        return None
    left = right = rows[0][0]
    bay_start = None
    for j, row in enumerate(rows):
        if not row or len(row) > 2:
            return None
        lo, hi = row[0], row[-1]
        if j and not (left - 1 <= lo <= left and right <= hi <= right + 1):
            return None
        if row != sorted({lo, hi}):
            return None
        left, right = lo, hi
        if lo == 0 and hi == width - 1 and bay_start is None:
            bay_start = j
        if bay_start is not None and (lo != 0 or hi != width - 1):
            return None
    if bay_start is None or not 2 <= height - bay_start <= 4:
        return None
    bay = [[_world((y0 + y, x0 + x), direction)
            for x in range(1, width - 1)] for y in range(bay_start, height)]
    tip = _world((y0, x0 + rows[0][0]), direction)
    return tip, bay, height, width


def _read(input_grid):
    grid = _grid(input_grid)
    background = _background(grid)
    if any(label in grid for label in (OPERAND_LABEL, RESULT_LABEL)):
        raise ValueError("label colors 8 and 3 are reserved for output")
    objects = []
    for color in sorted(set(int(x) for x in np.unique(grid)) - {background}):
        for frame in _components(map(tuple, np.argwhere(grid == color))):
            if len(frame) < 7:
                continue
            candidates = []
            for direction in DIRECTIONS:
                geometry = _geometry(frame, direction)
                if geometry is None:
                    continue
                tip, bay, height, width = geometry
                values = np.array([[grid[p] for p in row] for row in bay])
                colors = set(int(v) for v in np.unique(values)) - {background}
                if len(colors) != 1 or color in colors:
                    continue
                payload_color = next(iter(colors))
                mask = values == payload_color
                if not 0 < int(mask.sum()) < mask.size:
                    continue
                ys, xs = zip(*frame)
                top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
                box = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}
                bay_cells = {p for row in bay for p in row}
                if any(grid[p] != background for p in box - frame - bay_cells):
                    continue
                payload = {p for row in bay for p in row if grid[p] == payload_color}
                ray, room = _rays(grid.shape, tip, direction, len(payload))
                candidates.append(dict(frame=frame, bay=bay, mask=mask, tip=tip,
                    direction=direction, frame_color=color, payload_color=payload_color,
                    payload=payload, quantity=len(payload), ray=ray, room=room,
                    box=box, profile=(height, width, tuple(sorted(_local(p, direction) for p in frame)))))
            if len(candidates) > 1:
                raise ValueError("ambiguous pointing direction")
            objects.extend(candidates)
    objects.sort(key=lambda obj: min(obj["frame"]))
    if not 5 <= len(objects) <= 8:
        raise ValueError("expected 5..8 vessels")
    if len({obj["mask"].shape for obj in objects}) != 1:
        raise ValueError("all canonical bays in a scene must have equal dimensions")
    covered = set()
    boxes = set()
    for obj in objects:
        if not boxes.isdisjoint(obj["box"]):
            raise ValueError("overlapping vessels")
        boxes.update(obj["box"])
        covered.update(obj["frame"] | obj["payload"])
    if covered != set(map(tuple, np.argwhere(grid != background))):
        raise ValueError("nonbackground cells outside vessels")
    return background, objects


def _bits(mask):
    return sum(1 << i for i, value in enumerate(mask.flat) if value)


def _relations(masks):
    bits = [_bits(mask) for mask in masks]
    triples = []
    for a in range(len(bits)):
        for b in range(a + 1, len(bits)):
            if not bits[a] & ~bits[b] or not bits[b] & ~bits[a]:
                continue
            for k in range(len(bits)):
                if k not in (a, b) and bits[a] | bits[b] == bits[k]:
                    triples.append((a, b, k))
    return triples


def _relation(masks):
    triples = _relations(masks)
    if not triples:
        raise ValueError("no strict union equation")
    bits = [_bits(mask) for mask in masks]
    signatures = {(tuple(sorted((bits[a], bits[b]))), bits[k]) for a, b, k in triples}
    if len(signatures) != 1:
        raise ValueError("different mask-level union equations are outside this family")
    return min(triples)


def _paint(grid, background, objects, relation, mode="full"):
    output = grid.copy()
    a, b, k = relation
    chosen = range(len(objects)) if mode == "all" else relation
    for i in chosen:
        obj = objects[i]
        if mode != "tips_only":
            for p in obj["payload"]:
                output[p] = background
            for p in obj["ray"]:
                output[p] = obj["payload_color"]
        output[obj["tip"]] = RESULT_LABEL if i == k else OPERAND_LABEL
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    background, objects = _read(grid)
    relation = _relation([obj["mask"] for obj in objects])
    return _paint(grid, background, objects, relation)


def _mask_set(rng, height, width, duplicate, extra, allow_disjoint=False,
              duplicate_result=False):
    """Construct one relation plus a same-count near-result and a subset."""
    size = height * width
    for _ in range(240):
        style = int(rng.integers(4))
        common = int(rng.integers(0 if allow_disjoint else 1, min(5, size - 4) + 1))
        if size == 6:
            common, only_a, only_b = 1, 1, 2
        else:
            only_a = int(rng.integers(1, min(3, size - common - 2) + 1))
            only_b = int(rng.integers(1, min(4, size - common - only_a - 1) + 1))
        cells = list(range(size))
        if style == 0:
            rng.shuffle(cells)
        else:
            r0, c0 = int(rng.integers(height)), int(rng.integers(width))
            jitter = rng.random(size)
            if style == 1:
                cells.sort(key=lambda i: (abs(i // width - r0) + abs(i % width - c0), jitter[i]))
            elif style == 2:
                cells.sort(key=lambda i: (i // width != r0, jitter[i]))
            else:
                cells.sort(key=lambda i: (min(i // width, height - 1 - i // width,
                                               i % width, width - 1 - i % width), jitter[i]))
        shared = set(cells[:common])
        available = cells[common:]
        rng.shuffle(available)
        aa = shared | set(available[:only_a])
        bb = shared | set(available[only_a:only_a + only_b])
        union = aa | bb
        # A two-pixel edit preserves the count but falsifies aligned equality.
        decoy = set(union)
        decoy.remove(int(rng.choice(sorted(union))))
        decoy.add(int(rng.choice(sorted(set(cells) - union))))
        subset = {int(rng.choice(sorted(aa)))}
        sets = [aa, bb, union, decoy, subset]
        if duplicate:
            sets.append(set(aa))
        if duplicate_result:
            sets.append(set(union))
        for _ in range(extra):
            if rng.random() < .3:
                # Duplicate results also obey the same row-major tie convention.
                sets.append(set(union))
            else:
                count = int(rng.integers(1, size))
                sets.append(set(int(v) for v in rng.choice(size, count, replace=False)))
        masks = []
        for points in sets:
            mask = np.zeros((height, width), dtype=bool)
            mask.flat[sorted(points)] = True
            masks.append(mask)
        try:
            _relation(masks)
        except ValueError:
            continue
        return masks
    raise RuntimeError("bounded motif construction exhausted")


def _profile(rng, height, width):
    apex = int(rng.integers(1, width + 1))
    stem = int(rng.integers(2))
    frame = {(y, apex) for y in range(stem + 1)}
    left = right = apex
    y = stem
    delays = 0
    while left > 0 or right < width + 1:
        y += 1
        move_left, move_right = left > 0, right < width + 1
        if move_left and move_right and delays < 2 and rng.random() < .35:
            if rng.random() < .5:
                move_left = False
            else:
                move_right = False
            delays += 1
        left -= int(move_left)
        right += int(move_right)
        frame.update(((y, left), (y, right)))
    start = y
    for y in range(start, start + height):
        frame.update(((y, 0), (y, width + 1)))
    bay = [[(y, x) for x in range(1, width + 1)] for y in range(start, start + height)]
    return frame, bay, (0, apex)


def _oriented(profile, direction):
    frame, bay, tip = profile
    points = [_world(p, direction) for p in frame]
    low_r, low_c = min(p[0] for p in points), min(p[1] for p in points)
    def transform(p):
        r, c = _world(p, direction)
        return r - low_r, c - low_c
    frame = {transform(p) for p in frame}
    bay = [[transform(p) for p in row] for row in bay]
    tip = transform(tip)
    return dict(frame=frame, bay=bay, tip=tip, direction=direction,
                h=max(r for r, c in frame) + 1, w=max(c for r, c in frame) + 1)


def _pack(rng, specs, height, width):
    boxes = np.zeros((height, width), dtype=bool)
    guards = boxes.copy()
    rays = boxes.copy()
    placed = []
    # Required free exports go first; ties are shuffled, unrelated to scan order.
    priorities = {"free": 0, "clip": 1, "any": 2}
    order = sorted(rng.permutation(len(specs)), key=lambda i: priorities[specs[i]["mode"]])
    for index in order:
        spec = specs[index]
        h, w, q = spec["h"], spec["w"], int(spec["mask"].sum())
        candidates = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        accepted = None
        for raw in rng.permutation(len(candidates)):
            r, c = candidates[int(raw)]
            tip = (r + spec["tip"][0], c + spec["tip"][1])
            ray, room = _rays((height, width), tip, spec["direction"], q)
            if spec["mode"] == "free" and room <= q:
                continue
            if spec["mode"] == "clip" and not 0 <= room < q:
                continue
            if np.any(guards[r:r+h, c:c+w]) or np.any(rays[r:r+h, c:c+w]):
                continue
            # Keep actual rays disjoint even for frames that remain unchanged.
            reserve, _ = _rays((height, width), tip, spec["direction"], q + 1)
            if any(boxes[p] or rays[p] for p in reserve):
                continue
            accepted = r, c, tip, ray, room, reserve
            break
        if accepted is None:
            return None
        r, c, tip, ray, room, reserve = accepted
        obj = dict(spec)
        obj.update(frame={(r+y, c+x) for y, x in spec["frame"]},
                   bay=[[(r+y, c+x) for y, x in row] for row in spec["bay"]],
                   tip=tip, ray=ray, room=room, quantity=q)
        obj["payload"] = {p for row, values in zip(obj["bay"], obj["mask"])
                          for p, value in zip(row, values) if value}
        placed.append(obj)
        boxes[r:r+h, c:c+w] = True
        guards[max(0, r-1):min(height, r+h+1), max(0, c-1):min(width, c+w+1)] = True
        for p in reserve:
            rays[p] = True
    return sorted(placed, key=lambda obj: min(obj["frame"]))


def _scene(rng, *, anchor, direction, test):
    if anchor:
        mh, mw = 2, 3
    else:
        shapes = ((3, 4), (4, 3)) if test else ((2, 3), (2, 4), (3, 2), (3, 3))
        mh, mw = shapes[int(rng.integers(len(shapes)))]
    duplicate = anchor == 2 or bool(not anchor and rng.random() < .5)
    extra = int(rng.integers(1, 3)) if test else int(rng.integers(2)) if not anchor else 0
    masks = _mask_set(rng, mh, mw, duplicate, extra, allow_disjoint=not anchor,
                      duplicate_result=anchor == 2)
    colors = [c for c in range(10) if c not in (OPERAND_LABEL, RESULT_LABEL)]
    background, frame_color, payload_color = (int(c) for c in rng.choice(colors, 3, replace=False))
    local_colors = [c for c in colors if c != background]
    common_profile = _profile(rng, mh, mw)
    specs = []
    for i, mask in enumerate(masks):
        facing = direction if anchor == 1 else DIRECTIONS[(DIRECTIONS.index(direction) + i) % 4]
        if not anchor and rng.random() < .5:
            facing = str(rng.choice(DIRECTIONS))
        profile = common_profile if anchor == 1 else _profile(rng, mh, mw)
        spec = _oriented(profile, facing)
        fc, pc = (frame_color, payload_color) if anchor == 1 else tuple(int(c) for c in rng.choice(local_colors, 2, replace=False))
        # Every interchangeable candidate receives compatible room, so choosing
        # the first occurrence cannot remove the counting/clipping witnesses.
        operand = _bits(mask) in (_bits(masks[0]), _bits(masks[1]))
        result = _bits(mask) == _bits(masks[2])
        mode = ("free" if operand or anchor == 1 and i in (2, 3) else
                "clip" if anchor == 2 and result else "any")
        spec.update(mask=mask, frame_color=fc, payload_color=pc, mode=mode)
        specs.append(spec)
    for attempt in range(36):
        if attempt < 8:
            height, width = int(rng.integers(23, 31)), int(rng.integers(23, 31))
        else:
            height = width = 30
        placed = _pack(rng, specs, height, width)
        if placed is None:
            continue
        # In the first demonstration, the equal-count impostor precedes U.
        if anchor == 1:
            ui = next(i for i, obj in enumerate(placed) if obj["mask"] is masks[2])
            di = next(i for i, obj in enumerate(placed) if obj["mask"] is masks[3])
            if di > ui:
                placed[ui]["mask"], placed[di]["mask"] = placed[di]["mask"], placed[ui]["mask"]
                for obj in (placed[ui], placed[di]):
                    obj["payload"] = {p for row, values in zip(obj["bay"], obj["mask"])
                                      for p, value in zip(row, values) if value}
        relation = _relation([obj["mask"] for obj in placed])
        # The test must retain a count-dependent endpoint, rather than hide all
        # quantities through clipping. Duplicates may change the chosen frames.
        if not any(placed[i]["room"] > placed[i]["quantity"] for i in relation):
            continue
        inp = np.full((height, width), background, dtype=np.int8)
        for obj in placed:
            for p in obj["frame"]:
                inp[p] = obj["frame_color"]
            for p in obj["payload"]:
                inp[p] = obj["payload_color"]
        out = _paint(inp, background, placed, relation)
        # Construction metadata and input recognition must agree on edits.
        if not np.array_equal(solve(inp), out):
            raise AssertionError("constructed scene disagrees with input reader")
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene packing exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    direction = str(rng.choice(DIRECTIONS))
    anchors = [1, 2]
    rng.shuffle(anchors)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_relation = None
    for split, count in (("train", n), ("test", m)):
        for i in range(count):
            anchor = anchors[i] if split == "train" and i < 2 else 0
            for _ in range(12):
                pair = _scene(rng, anchor=anchor, direction=direction, test=split == "test")
                key = _key(pair["input"])
                _, objects = _read(pair["input"])
                relation = _relation([obj["mask"] for obj in objects])
                if key in seen or split == "train" and i == 1 and relation == first_relation:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and i == 0:
                    first_relation = relation
                break
            else:
                raise RuntimeError("bounded distinct-example generation exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, full_counts, role_triples = set(), set(), set()
    train_tie = train_clip = train_overlap = train_redundancy = False
    train_operand_tie = train_result_tie = train_count_decoy = False
    train_components_differ = train_double_count_differ = False
    matched_counts = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            background, objects = _read(inp)
            masks = [obj["mask"] for obj in objects]
            relation = _relation(masks)
            a, b, k = relation
            assert np.any(masks[a] & ~masks[b]) and np.any(masks[b] & ~masks[a])
            assert np.array_equal(masks[a] | masks[b], masks[k])
            assert np.array_equal(_paint(inp, background, objects, relation), out)
            assert np.array_equal(solve(inp.copy()), out)
            assert not np.array_equal(_paint(inp, background, objects, relation, "all"), out)
            assert not np.array_equal(_paint(inp, background, objects, relation, "tips_only"), out)
            assert any(i not in relation and int(masks[i].sum()) == int(masks[k].sum())
                       and np.count_nonzero(masks[i] != masks[k]) == 2 for i in range(len(objects)))
            reserved, boxes = set(), set().union(*(obj["box"] for obj in objects))
            for i, obj in enumerate(objects):
                assert all(inp[p] == background for p in obj["ray"])
                assert reserved.isdisjoint(obj["ray"]) and boxes.isdisjoint(obj["ray"])
                reserved.update(obj["ray"])
                if i in relation:
                    assert out[obj["tip"]] == (RESULT_LABEL if i == k else OPERAND_LABEL)
                    assert all(out[p] == background for p in obj["payload"])
                    assert all(out[p] == obj["payload_color"] for p in obj["ray"])
                else:
                    assert all(out[p] == inp[p] for p in obj["box"])
            assert any(objects[i]["room"] > objects[i]["quantity"] for i in relation)
            if split == "train":
                role_triples.add(relation)
                train_tie |= len(_relations(masks)) > 1
                train_overlap |= bool(np.any(masks[a] & masks[b]))
                bits = [_bits(mask) for mask in masks]
                train_operand_tie |= any(bits.count(bits[j]) > 1 for j in (a, b))
                train_result_tie |= bits.count(bits[k]) > 1
                train_count_decoy |= any(j < k and j not in relation
                    and int(masks[j].sum()) == objects[k]["quantity"]
                    and bits[j] != bits[k] for j in range(len(objects)))
                train_components_differ |= any(objects[j]["room"] > objects[j]["quantity"]
                    and len(_components(objects[j]["payload"])) < objects[j]["quantity"]
                    for j in relation)
                train_double_count_differ |= (objects[k]["room"] > objects[k]["quantity"]
                    and objects[a]["quantity"] + objects[b]["quantity"] > objects[k]["quantity"])
                train_redundancy |= any(i != j and k2 not in (i, j) and
                    bits[i] | bits[j] == bits[k2] and not (bits[i] & ~bits[j] and bits[j] & ~bits[i])
                    for i in range(len(bits)) for j in range(i+1, len(bits)) for k2 in range(len(bits)))
                for i in relation:
                    obj = objects[i]
                    train_clip |= obj["room"] < obj["quantity"]
                    if obj["room"] > obj["quantity"]:
                        full_counts.add(obj["quantity"])
                        ys = [_local(p, obj["direction"])[0] for p in obj["frame"]]
                        xs = [_local(p, obj["direction"])[1] for p in obj["frame"]]
                        signature = (tuple(sorted((y-min(ys), x-min(xs)) for y, x in
                            (_local(p, obj["direction"]) for p in obj["frame"]))),
                            obj["direction"], obj["frame_color"], obj["payload_color"])
                        matched_counts.setdefault(signature, set()).add(obj["quantity"])
    assert len(full_counts) >= 2 and any(len(q) >= 2 for q in matched_counts.values())
    assert len(role_triples) >= 2
    assert train_tie and train_clip and train_overlap and train_redundancy
    assert train_operand_tie and train_result_tie and train_count_decoy
    assert train_components_differ and train_double_count_differ
    assert max(_read(pair["input"])[1][0]["mask"].size for pair in puzzle["train"]) < 12
    assert all(_read(pair["input"])[1][0]["mask"].size == 12 for pair in puzzle["test"])


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, count boundaries, mutation checks; no disk writes."""
    fingerprints, shapes, bays, counts, directions, palettes = (set() for _ in range(6))
    quantities, visible, profiles, selected_positions = (set() for _ in range(4))
    ties = result_ties = overlap = clipped = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, objects = _read(inp)
                relation = _relation([obj["mask"] for obj in objects])
                shapes.add(inp.shape)
                bays.add(objects[0]["mask"].shape)
                counts.add(len(objects))
                palettes.add((bg, tuple(sorted(set(int(x) for x in np.unique(inp))))))
                ties += len(_relations([obj["mask"] for obj in objects])) > 1
                result_ties += sum(np.array_equal(obj["mask"], objects[relation[2]]["mask"])
                                   for obj in objects) > 1
                overlap += np.any(objects[relation[0]]["mask"] & objects[relation[1]]["mask"])
                selected_positions.add(relation)
                for obj in objects:
                    directions.add(obj["direction"])
                    quantities.add(obj["quantity"])
                    visible.add(len(obj["ray"]))
                    clipped += obj["room"] < obj["quantity"]
                    local = [_local(p, obj["direction"]) for p in obj["frame"]]
                    y0, x0 = min(p[0] for p in local), min(p[1] for p in local)
                    profile = tuple(sorted((y-y0, x-x0) for y, x in local))
                    profiles.add(profile)
                # Label roles are fixed; all eight other colors may permute.
                permutation = np.arange(10)
                movable = [c for c in range(10) if c not in (8, 3)]
                permutation[movable] = movable[1:] + movable[:1]
                if seed == 0:
                    assert np.array_equal(solve(permutation[inp]), permutation[pair["output"]])
                    # Scan-order ties are deliberately not rotation invariant.
                    # Unique equations should commute with whole-scene rotation.
                    if len(_relations([obj["mask"] for obj in objects])) == 1:
                        for turns in range(1, 4):
                            assert np.array_equal(solve(np.rot90(inp, turns)),
                                                  np.rot90(pair["output"], turns))
                before = _key(inp)
                solve(inp)
                assert _key(inp) == before, "solve mutated its input"
    assert len(fingerprints) == 42 and len(shapes) >= 20 and len(palettes) >= 30
    assert len(bays) >= 6 and {5, 6, 7, 8} <= counts
    assert directions == set(DIRECTIONS) and len(profiles) >= 40
    assert max(quantities) >= 9 and max(visible) >= 7 and 0 in visible
    assert ties >= 42 and result_ties >= 42 and clipped >= 42 and overlap >= 42 and len(selected_positions) >= 30
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            generate_puzzle(1900 + 10*n + m, n, m)
    assert _fingerprint(generate_puzzle(123, 2, 1)) == _fingerprint(generate_puzzle(123, 2, 1))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "grid_shapes": len(shapes),
            "bay_shapes": sorted(bays), "frame_counts": sorted(counts),
            "directions": sorted(directions), "frame_profiles": len(profiles),
            "quantities": sorted(quantities), "visible_ray_lengths": sorted(visible),
            "tie_pairs": int(ties), "clipped_vessels": int(clipped),
            "result_tie_pairs": int(result_ties),
            "semantic_verification": "generator-owned checks only"}
