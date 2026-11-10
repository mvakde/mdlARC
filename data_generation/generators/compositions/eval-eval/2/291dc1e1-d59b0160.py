"""Color relay: ordered, resettable set completion with whole-strip erasure.

Only NumPy and the standard library are used. No files are read or written by
the generator, solver, validator, or self-test. See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


HOST_SEEDS = (0,333748360,70153309,213923542,936540063,812931292,877667119,
              264262911,485043579,48784884,989764763,428487382,56631625,
              294091846,561989279,891196227,255378521,739550966,332301539,
              326977279,305521796,624625166,921613301,544974820,48442570,
              703886431,232557596,447148161,502765210,665138335,40980221,
              596739755,493832130,608884024,652628055,698133413,588771379,
              723593718,470182478,423518784,455942009,938844010)
POSES = tuple(itertools.product((False, True), repeat=3))


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _counts(a, b):
    a = _integer(a, "num_train", 2, 10)
    b = _integer(b, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _pose(grid, pose):
    transpose, bottom, right = pose
    if transpose:
        grid = grid.T
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    return grid


def _unpose(grid, pose):
    transpose, bottom, right = pose
    if right:
        grid = grid[:, ::-1]
    if bottom:
        grid = grid[::-1]
    if transpose:
        grid = grid.T
    return grid


def _boxes(mask):
    """Solid components of the original input, independent of output labels."""
    todo = set(zip(*np.nonzero(mask)))
    boxes = []
    while todo:
        first = min(todo)
        todo.remove(first)
        queue = [first]
        for r, c in queue:
            for point in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if point in todo:
                    todo.remove(point)
                    queue.append(point)
        rows, cols = zip(*queue)
        box = (int(min(rows)), int(max(rows)+1), int(min(cols)), int(max(cols)+1))
        r0, r1, c0, c1 = box
        assert len(queue) == (r1-r0)*(c1-c0), "A strip has a hole or an irregular boundary"
        boxes.append(box)
    return boxes


def _read(input_grid):
    grid = _grid(input_grid)
    assert min(grid.shape) >= 8, "No room for the control key and payload"
    candidates = []
    for pose in POSES:
        view = _unpose(grid, pose)
        origin, frame = int(view[0, 0]), int(view[0, 1])
        if origin == frame or int(view[0, -1]) != origin:
            continue
        if not (np.all(view[0, 1:-1] == frame) and np.all(view[1:, 0] == frame)
                and np.all(view[4, 1:5] == frame) and np.all(view[1:5, 4] == frame)):
            continue
        values, counts = np.unique(view[1:4, 1:4], return_counts=True)
        padding = values[counts == 6]
        if len(padding) != 1:
            continue
        background = int(padding[0])
        query = tuple(int(c) for c in values if c != background)
        if (len(query) != 3 or len({origin, frame, background}) != 3 or
                set(query) & {origin, frame}):
            continue
        if (np.count_nonzero(view == origin) != 2 or
                np.count_nonzero(view == frame) != view.shape[0]+view.shape[1]+4):
            continue
        if not (np.all(view[-1, 1:] == background) and
                np.all(view[1:, -1] == background)):
            continue
        candidates.append((pose, view, origin, frame, background, query))
    assert len(candidates) == 1, "Exactly one directed corner control is required"
    pose, view, origin, frame, background, query = candidates[0]
    mask = view != background
    mask[0, :] = False
    mask[:, 0] = False
    mask[1:5, 1:5] = False
    boxes = _boxes(mask)
    assert 1 <= len(boxes) <= 15
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        assert r0 >= 6 and r1-r0 == 2 and c0 >= 1 and c1-c0 >= 2
        assert r1 < view.shape[0] and c1 < view.shape[1]
        lanes.setdefault(r0, []).append(box)
    starts = sorted(lanes)
    assert all(b-a >= 3 for a, b in zip(starts, starts[1:])), "Overlapping lane bands"
    ordered, lane_counts = [], []
    for row in starts:
        group = sorted(lanes[row], key=lambda box: box[2])
        assert all(b[2]-a[3] >= 1 for a, b in zip(group, group[1:]))
        ordered.extend(group)
        lane_counts.append(len(group))
    blocks, bits = [], []
    for r0, r1, c0, c1 in ordered:
        block = view[r0:r1, c0:c1]
        blocks.append(block)
        colors = set(map(int, block.flat))
        assert not colors & {origin, frame, background}
        bits.append(sum(1 << i for i, color in enumerate(query) if color in colors))
    return {"pose": pose, "view": view, "background": background,
            "origin": origin, "frame": frame, "query": query,
            "boxes": ordered, "blocks": blocks, "bits": bits,
            "lane_counts": lane_counts}


def _discharges(bits, order=None, reset=True):
    state = 0
    deleted = set()
    if order is None:
        order = range(len(bits))
    for i in order:
        state |= bits[i]
        if state == 7:
            deleted.add(i)
            if reset:
                state = 0
    return deleted


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the input alone; erase the record completing each fresh query set."""
    info = _read(input_grid)
    output = info["view"].copy()
    for i in _discharges(info["bits"]):
        r0, r1, c0, c1 = info["boxes"][i]
        output[r0:r1, c0:c1] = info["background"]
    return _pose(output, info["pose"]).copy()


def _groups(info):
    groups, start = [], 0
    for count in info["lane_counts"]:
        groups.append(list(range(start, start+count)))
        start += count
    return groups


def _alternative(info, name):
    bits = info["bits"]
    groups = _groups(info)
    if name == "local_only":
        return {i for i, b in enumerate(bits) if b == 7}
    if name == "no_reset":
        return _discharges(bits, reset=False)
    if name == "fast_reverse":
        order = [i for group in groups for i in group[::-1]]
    elif name == "slow_reverse":
        order = [i for group in groups[::-1] for i in group]
    elif name == "fast_major":
        order = sorted(range(len(bits)), key=lambda i: (info["boxes"][i][2], info["boxes"][i][0]))
    elif name == "lane_reset":
        return set().union(*(_discharges(bits, group) for group in groups))
    else:
        raise ValueError(name)
    return _discharges(bits, order)


def _coverage(infos):
    """Witnesses in returned cells, including when there are only two demos."""
    assert len({len(_discharges(d["bits"])) for d in infos}) >= 2, "Fixed deletion quota"
    for name in ("local_only", "no_reset", "fast_reverse", "slow_reverse", "fast_major", "lane_reset"):
        assert any(_discharges(d["bits"]) != _alternative(d, name) for d in infos), name
    records = [(d, i, i in _discharges(d["bits"])) for d in infos for i in range(len(d["bits"]))]
    assert any(d["bits"][i] == 7 and deleted for d, i, deleted in records)
    assert any(d["bits"][i] not in (0, 7) and deleted for d, i, deleted in records)
    assert {d["bits"][i] for d, i, deleted in records if not deleted} >= {1,2,3,4,5,6}
    assert any(not d["bits"][i] and len(np.unique(d["blocks"][i])) == 1
               for d, i, deleted in records if not deleted)
    # A literally identical record must be both kept and erased within one input.
    assert any(any(d["blocks"][i].shape == d["blocks"][j].shape and
                       np.array_equal(d["blocks"][i], d["blocks"][j]) and
                       ((i in _discharges(d["bits"])) != (j in _discharges(d["bits"])))
                   for i in range(len(d["bits"])) for j in range(i)) for d in infos)
    # Three exact query marks can trigger, while a richer incomplete strip stays.
    exact = rich = repeated_positive = repeated_negative = False
    for d, i, deleted in records:
        block = d["blocks"][i]
        q = set(d["query"])
        values, counts = np.unique(block, return_counts=True)
        count_map = dict(zip(map(int, values), map(int, counts)))
        # Filler is immaterial to solving; teaching exacts have all q once and
        # just one other identity, the common filler.
        if deleted and d["bits"][i] == 7:
            exact |= all(count_map[c] == 1 for c in q) and len(values) == 4
        if not deleted and d["bits"][i] != 7:
            rich |= len(values) >= 5 and sum(count_map.get(c, 0) for c in q) >= 3
        repeated = any(count_map.get(c, 0) >= 2 for c in q)
        repeated_positive |= deleted and repeated
        repeated_negative |= not deleted and repeated
    assert exact and rich and repeated_positive and repeated_negative


def _partition_counts(rng, n, witness):
    minimum = max(2 if witness else 1, (n+4)//5)
    if witness == 1:
        minimum = max(minimum, 1+(n-2)//5+1)
    maximum = min(8, n-1 if witness else n)
    lanes = int(rng.integers(minimum, maximum+1))
    counts = [1]*lanes
    for _ in range(n-lanes):
        choices = [i for i in range(lanes) if counts[i] < 5 and not (witness == 1 and i == 0)]
        counts[int(rng.choice(choices))] += 1
    return counts


def _spread(rng, total, count):
    return rng.multinomial(total, rng.dirichlet(np.full(count, .8))).tolist()


def _layout(rng, items, witness):
    counts = _partition_counts(rng, len(items), witness)
    caps = [min(16, (29-count)//count) for count in counts for _ in range(count)]
    lengths = [int(rng.integers(2, cap+1)) for cap in caps]
    tagged = {item[1]: i for i, item in enumerate(items) if item[1]}
    if witness == 1:
        i = tagged["rich"]
        lengths[i] = max(4, lengths[i])
        a, b = tagged["copy_a"], tagged["copy_b"]
        lengths[a] = lengths[b] = int(rng.integers(3, min(caps[a], caps[b])+1))
    # Per-lane ceilings ensure all lengths fit with mandatory one-cell gaps.
    groups, start = [], 0
    for count in counts:
        groups.append(lengths[start:start+count])
        start += count
    min_width = max(8, max(sum(group)+len(group)+1 for group in groups))
    width = int(rng.integers(min_width, 31))
    min_height = 3*len(counts)+6
    height = int(rng.integers(min_height, 31))
    extra_slow = _spread(rng, height-min_height, len(counts)+1)
    row = 6+extra_slow[0]
    boxes = []
    for j, group in enumerate(groups):
        extra_fast = _spread(rng, width-(sum(group)+len(group)+1), len(group)+1)
        col = 1+extra_fast[0]
        for k, length in enumerate(group):
            boxes.append((row, row+2, col, col+length))
            col += length+1+extra_fast[k+1]
        row += 3+extra_slow[j+1]
    return (height, width), boxes


def _items(rng, package):
    if package == 1:
        items = [(3,"rich"),(4,"copy_a"),(4,"copy_b"),(3,""),
                 (7,"exact"),(0,"empty"),(1,""),(2,"")]
    elif package == 2:
        items = [(6,""),(1,""),(1,""),(6,""),(5,""),(2,""),(3,"")]
        # Add a standalone completion at a point where memory is empty. This
        # preserves every witness while teaching a different deletion count.
        items.insert(int(rng.choice([0,2,4,6])), (7, ""))
    elif package == 3:
        # Delay the first discharge with redundant identities and an empty
        # supplier, then require a fresh collection after it.
        delay = int(rng.integers(4, 8))
        bits = [3, 0, 1, 2]+[int(rng.choice([0,1,2,3])) for _ in range(delay-4)]
        bits += [4, 4, 3]
        bits += [int(rng.integers(8)) for _ in range(int(rng.integers(15-len(bits))))]
        return [(bit, "") for bit in bits]
    else:
        n = int(rng.integers(1, 16))
        mode = float(rng.random())
        if mode < .14:
            bits = [7]*n
        elif mode < .28:
            allowed = int(rng.choice([3,5,6]))
            bits = [int(rng.integers(8)) & allowed for _ in range(n)]
        else:
            bits = [int(rng.integers(8)) for _ in range(n)]
        return [(bit, "") for bit in bits]
    # Neutral records alter length, gaps, population and delay without changing
    # the teaching subsequence. Never insert before its first rich supplier.
    for _ in range(int(rng.integers(5))):
        at = int(rng.integers(1, len(items)+1))
        items.insert(at, (0, ""))
    return items


def _payload(rng, shape, bit, tag, query, extras, filler):
    area = shape[0]*shape[1]
    labels = [query[i] for i in range(3) if bit & (1 << i)]
    if tag == "exact":
        marks = labels
    elif tag == "empty":
        marks = []
    elif tag == "rich":
        marks = labels+[extras[0],extras[1],labels[0],extras[0]]
    elif tag in ("copy_a", "copy_b"):
        marks = labels*2+[extras[0]]*2
    else:
        optional = list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
        labels += optional[:max(0, area-1-len(labels))]
        marks = list(labels)
        if labels:
            target = int(rng.integers(len(marks), area))
            marks += list(rng.choice(labels, size=target-len(marks)))
    assert len(marks) < area
    block = np.full(shape, filler, dtype=np.int8)
    points = rng.choice(area, size=len(marks), replace=False)
    block.flat[points] = rng.permutation(marks)
    return block


def _make_pair(rng, roles, pose, package, query=None):
    background, frame, origin, filler = map(int, roles[:4])
    labels = list(map(int, roles[4:]))
    if query is None:
        query = list(map(int, rng.choice(labels, size=3, replace=False)))
    else:
        query = list(query)
        rng.shuffle(query)
    extras = [c for c in labels if c not in query]
    rng.shuffle(extras)
    items = _items(rng, package)
    shape, boxes = _layout(rng, items, package if package in (1,2) else 0)
    grid = np.full(shape, background, dtype=np.int8)
    grid[0, :] = frame
    grid[:, 0] = frame
    grid[0, 0] = grid[0, -1] = origin
    grid[4, 1:5] = frame
    grid[1:5, 4] = frame
    for point, color in zip(rng.choice(9, 3, replace=False), query):
        grid[1+int(point)//3, 1+int(point)%3] = color
    seen, duplicate = set(), None
    truth = grid.copy()
    # Construct labels by a set-based relay on sampled payloads. The solver
    # separately segments returned cells, restores the guide basis, and uses bits.
    for (bit, tag), box in zip(items, boxes):
        r0, r1, c0, c1 = box
        block = _payload(rng, (2, c1-c0), bit, tag, query, extras, filler)
        if tag == "copy_a":
            duplicate = block.copy()
        if tag == "copy_b":
            block = duplicate.copy()
        grid[r0:r1, c0:c1] = block
        truth[r0:r1, c0:c1] = block
        seen.update(set(map(int, block.flat)) & set(query))
        if set(query) <= seen:
            truth[r0:r1, c0:c1] = background
            seen.clear()
    return {"input": _pose(grid, pose).copy(), "output": _pose(truth, pose).copy()}


def _input_key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_roles = rng.permutation(10)
    vary_roles = bool(rng.integers(2))
    first = POSES[int(rng.integers(8))]
    second = tuple(not b for b in first)
    training_poses = (first, second)
    testing_poses = [p for p in POSES if p not in training_poses]
    puzzle = {"train": [], "test": []}
    # Retry only accidental alternative-order ambiguities. Memory and records
    # are tiny; there are no files, trace journals, subprocesses, or workers.
    for attempt in range(64):
        teaching = []
        for i in range(2):
            roles = rng.permutation(10) if vary_roles else base_roles
            teaching.append(_make_pair(rng, roles, training_poses[i], i+1))
        try:
            _coverage([_read(p["input"]) for p in teaching])
        except AssertionError:
            continue
        break
    else:
        raise RuntimeError("Could not distinguish the competing traversal rules")
    puzzle["train"].extend(teaching)
    used = {_input_key(p["input"]) for p in teaching}
    for split, count in (("train", num_train-2), ("test", num_test)):
        for _ in range(count):
            for attempt in range(32):
                roles = rng.permutation(10) if vary_roles else base_roles
                if split == "train":
                    pose = training_poses[int(rng.integers(2))]
                    query = None
                else:
                    pose = testing_poses[int(rng.integers(len(testing_poses)))]
                    # Stable-palette tests select a triple not used in training;
                    # all identities are visible locally in the selector.
                    used_queries = {frozenset(_read(p["input"])["query"]) for p in puzzle["train"]}
                    choices = [q for q in itertools.combinations(map(int, roles[4:]), 3)
                               if frozenset(q) not in used_queries]
                    query = choices[int(rng.integers(len(choices)))] if choices else None
                pair = _make_pair(rng, roles, pose, 3 if split == "test" else 0, query)
                key = _input_key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unique input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in used
            used.add(key)
            info = _read(inp)
            infos[split].append(info)
            expected = solve(inp)
            assert np.array_equal(expected, out), "Incorrect relay erasure"
            # In physical coordinates, only complete record footprints change.
            aligned_out = _unpose(out, info["pose"])
            changed = np.zeros(info["view"].shape, dtype=bool)
            for i in _discharges(info["bits"]):
                r0, r1, c0, c1 = info["boxes"][i]
                changed[r0:r1, c0:c1] = True
                assert np.all(aligned_out[r0:r1, c0:c1] == info["background"])
            assert np.array_equal(info["view"] != aligned_out, changed)
            # A non-query carrier color occurs in every strip. Identifying it
            # is unnecessary: membership can use literal query identities.
            common = set.intersection(*(set(map(int, b.flat)) for b in info["blocks"]))
            assert common - set(info["query"])
    _coverage(infos["train"])
    train_poses = {d["pose"] for d in infos["train"]}
    assert len(train_poses) == 2
    assert all(d["pose"] not in train_poses for d in infos["test"])
    for d in infos["test"]:
        deleted = _discharges(d["bits"])
        assert deleted and min(deleted) >= 4, "Test must delay its first completion"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds; stream metrics and mutations, retaining no grid collection."""
    # Direct geometric boundaries beyond the sampler's 16-cell / five-per-lane
    # proposals. Labels and expected deletions here are specified explicitly.
    for wide in (True, False):
        direct = np.zeros((9 if wide else 12, 30), dtype=np.int8)
        direct[0, :] = 8
        direct[:, 0] = 8
        direct[0, 0] = direct[0, -1] = 9
        direct[4, 1:5] = direct[1:5, 4] = 8
        direct[1, 1:4] = [1,2,3]
        if wide:
            direct[6:8, 1:29] = 4
            direct[6, 1], direct[7, 14], direct[7, 28] = 1,2,3
            expected = direct.copy()
            expected[6:8, 1:29] = 0
        else:
            for i in range(15):
                r, c = (6, 1+3*i) if i < 9 else (9, 1+3*(i-9))
                direct[r:r+2, c:c+2] = 4
                direct[r, c] = 1+i % 3
            expected = direct.copy()
            for i in (2,5,8,11,14):
                r, c = (6, 1+3*i) if i < 9 else (9, 1+3*(i-9))
                expected[r:r+2, c:c+2] = 0
        for pose in POSES:
            assert np.array_equal(solve(_pose(direct, pose)), _pose(expected, pose))
    signatures, poses, shapes, layouts, counts, lengths, roles = (set() for _ in range(7))
    erased_counts, populations = set(), set()
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        signatures.add(_digest(puzzle))
        teaching = [_read(p["input"]) for p in puzzle["train"]]
        _coverage(teaching)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                pairs += 1
                poses.add(d["pose"])
                shapes.add(pair["input"].shape)
                layouts.add((d["pose"], d["view"].shape, tuple(d["boxes"])))
                counts.add(len(d["boxes"]))
                lengths.update(b.shape[1] for b in d["blocks"])
                roles.add((d["background"], d["frame"], d["origin"]))
                erased_counts.add(len(_discharges(d["bits"])))
                populations.add(tuple(d["lane_counts"]))
                # Selector positions and ordering are irrelevant to the relay.
                changed = pair["input"].copy()
                view = _unpose(changed, d["pose"])
                view[1:4, 1:4] = np.roll(view[1:4, 1:4].copy(), 1, axis=1)
                expected = pair["output"].copy()
                _unpose(expected, d["pose"])[1:4, 1:4] = view[1:4, 1:4]
                assert np.array_equal(solve(changed), expected)
                # Changing the input pose carries the output with it. This
                # exercises all physical guide directions without re-generation.
                if split == "test":
                    assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        # Same scene, changed selector identities: at least one deletion changes.
        found = False
        for pair in puzzle["train"]:
            d = _read(pair["input"])
            color_sets = [set(map(int, b.flat)) for b in d["blocks"]]
            common = set.intersection(*color_sets)
            palette = sorted(set.union(*color_sets) - set(d["query"]) - common)
            for extra in palette:
                for old in d["query"]:
                    changed = pair["input"].copy()
                    view = _unpose(changed, d["pose"])
                    selector = view[1:4, 1:4]
                    selector[selector == old] = extra
                    changed_info = _read(changed)
                    if _discharges(changed_info["bits"]) != _discharges(d["bits"]):
                        found = True
                        assert not np.array_equal(solve(changed), pair["output"])
                        break
                if found:
                    break
            if found:
                break
        assert found, "The query identities must affect which records discharge"
    assert len(signatures) == 42 and len(poses) == 8
    assert len(shapes) >= 50 and len(layouts) >= 180 and len(roles) >= 40
    assert len(counts) >= 10 and len(lengths) >= 10 and len(populations) >= 80
    assert 0 in erased_counts and len(erased_counts) >= 7
    # Every legal explicit count, plus strict rejection of non-integer counts.
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(271828, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            legal += 1
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = generate_puzzle(123)
    p = damaged["train"][0]
    d = _read(p["input"])
    i = next(iter(_discharges(d["bits"])))
    r0, _, c0, _ = d["boxes"][i]
    _unpose(p["output"], d["pose"])[r0, c0] = d["view"][r0, c0]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Partial erasure accepted")
    return {"seeds":42,"pairs":pairs,"distinct_puzzles":len(signatures),
            "poses":len(poses),"shapes":len(shapes),"layouts":len(layouts),
            "role_assignments":len(roles),"strip_counts":sorted(counts),
            "lengths":sorted(lengths),"deletion_counts":sorted(erased_counts),
            "lane_populations":len(populations),"legal_count_combinations":legal,
            "independent_solver":False,"blind_prediction":False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
