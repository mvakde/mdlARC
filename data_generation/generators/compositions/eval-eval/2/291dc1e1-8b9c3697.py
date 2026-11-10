"""Finite-capacity raster docks. Standard library and NumPy only.

Dashed/solid border arms visibly name fast/slow axes. The first encountered
cell orders both strips and docks. Geometry supplies matching and capacity.
Ground truth is built from sampled objects; solve reparses only its input.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("An ARC grid must be a 1..30 integer rectangle in colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    """Find a proper rotation with the dashed fast arm pointing right.

    The slow arm can point down (top origin) or up (bottom origin).
    Colors are read here, never learned or obtained from generation.
    """
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        if min(a.shape) < 7:
            continue
        for bottom in (False, True):
            row, opposite, neighbor = (-1, 0, -2) if bottom else (0, -1, 1)
            origin, fast, slow, bg = map(int, (a[row, 0], a[row, 1],
                                               a[neighbor, 0], a[opposite, -1]))
            if len({origin, fast, slow, bg}) != 4:
                continue
            arm = np.where(np.arange(a.shape[1] - 1) % 2 == 0, fast, bg)
            side = a[:-1, 0] if bottom else a[1:, 0]
            far = a[:-1, -1] if bottom else a[1:, -1]
            if (np.array_equal(a[row, 1:], arm) and np.all(side == slow)
                    and np.all(far == bg) and np.all(a[opposite, 1:] == bg)):
                candidates.append((a.copy(), rotation, bottom, bg, origin, fast, slow))
    if len(candidates) != 1:
        raise ValueError("The input must have one dashed-fast/solid-slow L key")
    return candidates[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        points, pending = {first}, [first]
        while pending:
            r, c = pending.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    points.add(q)
                    pending.append(q)
        result.append(points)
    return result


def _encounter(cells, bottom=False, fast_reverse=False, fast_major=False):
    def index(p):
        r, c = p
        v, u = (-r if bottom else r), (-c if fast_reverse else c)
        return (u, v) if fast_major else (v, u)
    return min(map(index, cells))


def _parse_dock(mask):
    """Rotate a monochrome component to base-up/open-down; inspect profiles."""
    candidates = []
    for rotation in range(4):
        m = np.rot90(mask, rotation)
        if not np.all(m[0]) or m.shape[0] < 2:
            continue
        lengths = np.sum(m[1:], axis=0).astype(int)
        if any(not np.array_equal(m[:, q], np.arange(m.shape[0]) <= length)
               for q, length in enumerate(lengths)):
            continue
        runs, q = [], 0
        while q < len(lengths):
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, int(lengths[q])
            while q < len(lengths) and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3 or any(runs[i][1] >= runs[i+1][0] for i in (0, 1)):
            continue
        left, center, right = runs
        width = m.shape[1]
        if (left[0] != 0 or right[1] != width or left[1] != width-right[0]
                or center[0]+center[1] != width or center[1]-center[0] < 2):
            continue
        capacity = max(0, (min(left[2], right[2]) - center[2]) // 2)
        candidates.append({"rotation": rotation, "local_shape": m.shape,
                           "runs": tuple(runs), "span": center[1]-center[0],
                           "center": center[2], "q0": center[0],
                           "capacity": capacity})
    if len(candidates) != 1:
        raise ValueError("A dock needs one base and three positive separated prongs")
    return candidates[0]


def _read(input_grid):
    a, rotation, bottom, bg, origin, fast, slow = _key(_grid(input_grid))
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, 0] = False
    strips, docks, occupied = [], [], set()
    for cells in _components(mask):
        r0, r1 = min(p[0] for p in cells), max(p[0] for p in cells)+1
        c0, c1 = min(p[1] for p in cells), max(p[1] for p in cells)+1
        if not (1 <= r0 < r1 <= a.shape[0]-1 and 1 <= c0 < c1 <= a.shape[1]-1):
            raise ValueError("Scene objects must be inside the non-key borders")
        if c0 < 2 or (bottom and r1 > a.shape[0]-2) or (not bottom and r0 < 2):
            raise ValueError("Scene objects need a background margin from both key arms")
        expanded = {(r+dr, c+dc) for r, c in cells
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if not occupied.isdisjoint(expanded):
            raise ValueError("Input objects must have a background cell between them")
        occupied.update(cells)
        crop = a[r0:r1, c0:c1]
        obj = {"cells": cells, "box": (r0, r1, c0, c1)}
        if len(cells) == crop.size:
            if crop.shape[0] != 2 or crop.shape[1] < 2:
                raise ValueError("Cartridges have thickness two along the fast axis")
            obj.update(block=crop.copy(), span=crop.shape[1])
            strips.append(obj)
        else:
            if len({int(a[p]) for p in cells}) != 1:
                raise ValueError("A receiver must be monochrome")
            local_mask = crop != bg
            obj.update(_parse_dock(local_mask))
            obj["color"] = int(a[min(cells)])
            docks.append(obj)
    if not strips or not docks:
        raise ValueError("A scene must contain supply strips and docks")
    strips.sort(key=lambda o: _encounter(o["cells"], bottom))
    docks.sort(key=lambda o: _encounter(o["cells"], bottom))
    all_bays = set()
    for dock in docks:
        r0, r1, c0, c1 = dock["box"]
        m = np.rot90(a[r0:r1, c0:c1], dock["rotation"])
        k = dock["center"] + 1
        q = dock["q0"]
        if not np.all(m[k:k+2*dock["capacity"], q:q+dock["span"]] == bg):
            raise ValueError("Every receiver bay must initially be empty")
        bay = np.zeros(m.shape, dtype=bool)
        bay[k:k+2*dock["capacity"], q:q+dock["span"]] = True
        physical = np.rot90(bay, -dock["rotation"])
        cells = {(r0+int(rr), c0+int(cc)) for rr, cc in np.argwhere(physical)}
        if not all_bays.isdisjoint(cells):
            raise ValueError("Receiver bays must be disjoint even when partly unused")
        all_bays.update(cells)
    return {"grid": a, "rotation": rotation, "bottom": bottom, "background": bg,
            "origin": origin, "fast": fast, "slow": slow,
            "strips": strips, "docks": docks}


def _allocate(strips, docks, capacities=None, match_width=True):
    capacities = ([d["capacity"] for d in docks] if capacities is None else capacities)
    stacks = [[] for _ in docks]
    rejected = []
    for strip in strips:
        for i, dock in enumerate(docks):
            if ((not match_width or strip["span"] == dock["span"])
                    and len(stacks[i]) < capacities[i]):
                stacks[i].append(strip)
                break
        else:
            rejected.append(strip)
    return stacks, rejected


def _render_reading(d, strips=None, docks=None, capacities=None):
    strips = d["strips"] if strips is None else strips
    docks = d["docks"] if docks is None else docks
    stacks, rejected = _allocate(strips, docks, capacities)
    out = d["grid"].copy()
    for strip in d["strips"]:
        r0, r1, c0, c1 = strip["box"]
        out[r0:r1, c0:c1] = d["background"]
    for dock, stack in zip(docks, stacks):
        r0, r1, c0, c1 = dock["box"]
        local = np.rot90(out[r0:r1, c0:c1], dock["rotation"]).copy()
        q, start = dock["q0"], dock["center"]+1
        for index, strip in enumerate(stack):
            local[start+2*index:start+2*index+2, q:q+dock["span"]] = strip["block"]
        out[r0:r1, c0:c1] = np.rot90(local, -dock["rotation"])
    return np.rot90(out, -d["rotation"]).copy(), stacks, rejected


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the same-size assembled scene using the visible input alone."""
    return _render_reading(_read(input_grid))[0]


def _asymmetric(block):
    return all(not np.array_equal(block, other) for other in
               (block[::-1], block[:, ::-1], block[::-1, ::-1]))


def _positions(blocked, height, width):
    """Available rectangle positions, using summed-area occupancy."""
    if height > blocked.shape[0] or width > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (integral[height:, width:] - integral[:-height, width:]
            - integral[height:, :-width] + integral[:-height, :-width])
    return np.argwhere(sums == 0)


def _reserve(blocked, r, c, h, w):
    blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _dock_object(rng, span, capacity, small=False):
    side = 1 if small else int(rng.integers(1, 3))
    gap = 1 if small else int(rng.integers(1, 3))
    center = int(rng.integers(1, 3 if small else 4))
    # The shorter guard has zero or one spare cells; the other is independent.
    if capacity == 0:
        # The stop can be longer than both guards, or leave a single spare cell.
        lengths = rng.integers(1, center+2, size=2).astype(int).tolist()
    else:
        shortest = center + 2*capacity + int(rng.integers(2))
        lengths = [shortest, shortest + int(rng.integers(0, 3))]
    rng.shuffle(lengths)
    width = span + 2*(side+gap)
    q0 = side+gap
    mask = np.zeros((max(center, *lengths)+1, width), dtype=bool)
    mask[0] = True
    mask[1:lengths[0]+1, :side] = True
    mask[1:center+1, q0:q0+span] = True
    mask[1:lengths[1]+1, -side:] = True
    turn = int(rng.integers(4))
    return {"mask": np.rot90(mask, turn).copy(), "local_shape": mask.shape,
            "turn": turn, "span": span, "capacity": capacity,
            "center": center, "q0": q0, "lengths": tuple(lengths),
            "side": side, "gap": gap}


def _payload(rng, span, palette, mode, witness=False):
    if witness:
        for _ in range(40):
            block = rng.choice(palette, size=(2, span)).astype(np.int8)
            if _asymmetric(block):
                return block
        block = np.full((2, span), palette[0], dtype=np.int8)
        block[0, 0], block[1, -1] = palette[1], palette[2]
        return block
    if mode == 0:
        return np.full((2, span), rng.choice(palette), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(palette, size=(1, span)), 2, axis=0).astype(np.int8)
    if mode == 2:
        return np.repeat(rng.choice(palette, size=(2, 1)), span, axis=1).astype(np.int8)
    return rng.choice(palette, size=(2, span)).astype(np.int8)


def _construct(rng, split, kind, configuration, roles, training_spans):
    """Pack new objects and build labels from their recorded geometry."""
    bottom, whole_turn = configuration
    bg, origin, fast, slow = roles
    for attempt in range(120):
        count = 2 if kind in (0, 1) else (int(rng.integers(1, 3)) if split == "train"
                                         else int(rng.choice([3, 4], p=[.8, .2])))
        small = count >= 3
        if kind == 0:
            spans, caps = [training_spans[0]]*2, [1, 2]
            source_lengths = [spans[0]]*4 + [training_spans[1], training_spans[2]]
        elif kind == 1:
            spans, caps = [training_spans[1], training_spans[2]], [2, 0]
            # Matching supply cannot activate a receiver without a whole slot.
            alternatives = [x for x in range(2, 8) if x not in spans]
            source_lengths = [spans[0]]*3 + [spans[1], int(rng.choice(alternatives))]
        else:
            high = 6 if count == 4 else (9 if split == "test" else 8)
            spans = rng.integers(2, high, size=count).astype(int).tolist()
            if count >= 3:
                spans[1] = spans[0]  # repeated demand, with geometric spillover
            caps = rng.integers(1, 3, size=count).astype(int).tolist()
            if count > 1 and rng.random() < .30:
                caps[int(rng.integers(count))] = 0
            if split == "test" and count == 3:
                caps[int(rng.integers(count))] = 3
            source_lengths = []
            for length in sorted(set(spans)):
                total = sum(cap for span, cap in zip(spans, caps) if span == length)
                supply = int(rng.integers(max(1, total-1), min(total+2, 7)+1))
                source_lengths.extend([length]*supply)
            unmatched = [x for x in range(2, 9) if x not in spans]
            source_lengths += [int(rng.choice(unmatched)) for _ in range(int(rng.integers(1, 3)))]
        rng.shuffle(source_lengths)
        docks = [_dock_object(rng, s, cap, small) for s, cap in zip(spans, caps)]
        # Larger crowded scenes are chosen by real packing requirements.
        low = 27 if split == "test" else (24 if kind in (0, 1) else 20)
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        if attempt > 60:
            h = w = 30
        blocked = np.zeros((h, w), dtype=bool)
        blocked[:2] = blocked[-2:] = True
        blocked[:, :2] = blocked[:, -2:] = True
        grid = np.full((h, w), bg, dtype=np.int8)
        success = True
        for index in sorted(range(count), key=lambda i: -docks[i]["mask"].size):
            dock = docks[index]
            dh, dw = dock["mask"].shape
            positions = _positions(blocked, dh, dw)
            if not len(positions):
                success = False
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            dock["box"] = (r, r+dh, c, c+dw)
            dock["cells"] = {(r+int(rr), c+int(cc)) for rr, cc in np.argwhere(dock["mask"])}
            color = int(rng.choice([x for x in range(10) if x != bg]))
            grid[r:r+dh, c:c+dw][dock["mask"]] = color
            _reserve(blocked, r, c, dh, dw)
        if not success:
            continue
        palette = rng.choice([x for x in range(10) if x != bg],
                             size=int(rng.integers(3, 10)), replace=False)
        strips, identities, rows_used = [], set(), []
        simple = kind == 2 and rng.random() < .30
        for length in sorted(source_lengths, reverse=True):
            positions = _positions(blocked, 2, length)
            if not len(positions):
                success = False
                break
            # Encourage some multi-strip rows without imposing a lane template.
            on_row = positions[np.isin(positions[:, 0], rows_used)]
            if len(on_row) and rng.random() < .6:
                positions = on_row
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            for _ in range(50):
                block = _payload(rng, length, palette, 0 if simple else int(rng.integers(4)),
                                 witness=kind in (0, 1))
                identity = block.shape, block.tobytes()
                if kind == 2 or identity not in identities:
                    break
            else:
                success = False
                break
            identities.add(identity)
            grid[r:r+2, c:c+length] = block
            cells = {(rr, cc) for rr in range(r, r+2) for cc in range(c, c+length)}
            strips.append({"box": (r, r+2, c, c+length), "cells": cells,
                           "span": length, "block": block})
            rows_used.append(r)
            _reserve(blocked, r, c, 2, length)
        if not success:
            continue
        keyrow = -1 if bottom else 0
        grid[:, 0] = slow
        grid[keyrow, 1:] = np.where(np.arange(w-1) % 2 == 0, fast, bg)
        grid[keyrow, 0] = origin
        strips.sort(key=lambda o: _encounter(o["cells"], bottom))
        docks.sort(key=lambda o: _encounter(o["cells"], bottom))
        stacks, rejected = _allocate(strips, docks)
        output = grid.copy()
        for strip in strips:
            r0, r1, c0, c1 = strip["box"]
            output[r0:r1, c0:c1] = bg
        # Construct the destination layers in the sampled dock's unrotated frame.
        for dock, stack in zip(docks, stacks):
            layer = np.full(dock["local_shape"], bg, dtype=np.int8)
            changed = np.zeros(dock["local_shape"], dtype=bool)
            k, q, span = dock["center"]+1, dock["q0"], dock["span"]
            for i, strip in enumerate(stack):
                layer[k+2*i:k+2*i+2, q:q+span] = strip["block"]
                changed[k+2*i:k+2*i+2, q:q+span] = True
            values, change = np.rot90(layer, dock["turn"]), np.rot90(changed, dock["turn"])
            r0, r1, c0, c1 = dock["box"]
            output[r0:r1, c0:c1][change] = values[change]
        pair = {"input": np.rot90(grid, whole_turn).copy(),
                "output": np.rot90(output, whole_turn).copy()}
        d = _read(pair["input"])
        if not np.array_equal(solve(pair["input"]), pair["output"]):
            raise AssertionError("Constructed label disagrees with the input-only reader")
        if kind in (0, 1) and not _asymmetric_evidence(d):
            continue
        if kind == 0 and not all(_different(d, name) for name in
                                ("fast_reverse", "slow_reverse", "fast_major", "dock_reverse",
                                 "longest_guard")):
            continue
        if kind == 1 and not _width_matters(d):
            continue
        return pair
    raise RuntimeError("Bounded packing failed to construct a scene")


def _asymmetric_evidence(d):
    stacks, _ = _allocate(d["strips"], d["docks"])
    return any(_asymmetric(s["block"]) for stack in stacks for s in stack)


def _width_matters(d):
    correct, _ = _allocate(d["strips"], d["docks"])
    unconstrained, _ = _allocate(d["strips"], d["docks"], match_width=False)
    identities = lambda groups: [[s["box"] for s in group] for group in groups]
    return identities(correct) != identities(unconstrained)


def _different(d, name):
    expected = _render_reading(d)[0]
    strips, docks = d["strips"], d["docks"]
    if name == "dock_reverse":
        alternative = _render_reading(d, docks=docks[::-1])[0]
    elif name == "one_per_dock":
        alternative = _render_reading(d, capacities=[min(1, x["capacity"]) for x in docks])[0]
    elif name == "longest_guard":
        capacities = [max(0, (max(x["runs"][0][2], x["runs"][2][2])-x["center"])//2)
                      for x in docks]
        alternative = _render_reading(d, capacities=capacities)[0]
    elif name in ("reflect_length", "reflect_thickness"):
        reflected = [dict(s, block=s["block"][:, ::-1] if name == "reflect_length"
                          else s["block"][::-1]) for s in strips]
        alternative = _render_reading(d, strips=reflected)[0]
    else:
        bottom = not d["bottom"] if name == "slow_reverse" else d["bottom"]
        ordered = sorted(strips, key=lambda o: _encounter(
            o["cells"], bottom, name == "fast_reverse", name == "fast_major"))
        alternative = _render_reading(d, strips=ordered)[0]
    return not np.array_equal(expected, alternative)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    fast, slow = map(int, rng.choice(10, size=2, replace=False))
    training_spans = rng.choice(np.arange(2, 7), size=3, replace=False).astype(int).tolist()
    base_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            kind = i if split == "train" and i < 2 else 2
            if kind == 0:
                config = False, base_turn
            elif kind == 1:
                config = True, (base_turn+1) % 4
            elif split == "test":
                config = bool(rng.integers(2)), (base_turn+2+i) % 4
            else:
                config = bool(rng.integers(2)), int(rng.integers(4))
            for _ in range(20):
                bg, origin = map(int, rng.choice([c for c in range(10) if c not in (fast, slow)],
                                                 size=2, replace=False))
                pair = _construct(rng, split, kind, config, (bg, origin, fast, slow), training_spans)
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unable to construct distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, guides, training, test = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            d = _read(inp)
            guides.add((d["fast"], d["slow"]))
            expected, stacks, rejected = _render_reading(d)
            assert np.array_equal(expected, out), "Incorrect matching, priority, orientation or capacity"
            assert not np.array_equal(inp, out)
            canonical_out = np.rot90(out, d["rotation"])
            # Only old supply and accepted slot cells may change.
            changed = np.zeros(d["grid"].shape, dtype=bool)
            for strip in d["strips"]:
                for p in strip["cells"]:
                    changed[p] = True
                    assert canonical_out[p] == d["background"]
            slots_seen = set()
            for dock, stack in zip(d["docks"], stacks):
                assert len(stack) <= dock["capacity"]
                for p in dock["cells"]:
                    assert canonical_out[p] == dock["color"]
                local = np.zeros(dock["local_shape"], dtype=bool)
                k, q = dock["center"]+1, dock["q0"]
                local[k:k+2*len(stack), q:q+dock["span"]] = True
                local = np.rot90(local, -dock["rotation"])
                r0, r1, c0, c1 = dock["box"]
                for rr, cc in np.argwhere(local):
                    p = r0+int(rr), c0+int(cc)
                    assert p not in slots_seen and d["grid"][p] == d["background"]
                    slots_seen.add(p)
                    changed[p] = True
            assert np.array_equal(canonical_out[~changed], d["grid"][~changed])
            assert sum(len(s["cells"]) for stack in stacks for s in stack) == len(slots_seen)
            assert any(stacks), "Every generated scene must perform at least one admission"
            (training if split == "train" else test).append(d)
    assert len(guides) == 1
    assert {d["bottom"] for d in training} == {False, True}
    assert len({d["rotation"] % 2 for d in training}) == 2
    for name in ("fast_reverse", "slow_reverse", "fast_major", "dock_reverse", "one_per_dock",
                 "longest_guard", "reflect_length", "reflect_thickness"):
        assert any(_different(d, name) for d in training), f"Training lacks evidence against {name}"
    assert sum(_asymmetric_evidence(d) for d in training) >= 2
    width_witness = inactive = overflow = zero_witness = False
    for d in training:
        stacks, rejected = _allocate(d["strips"], d["docks"])
        width_witness |= _width_matters(d)
        inactive |= any(not stack for stack in stacks)
        zero_witness |= any(dock["capacity"] == 0 and
                            dock["span"] in {s["span"] for s in d["strips"]}
                            for dock in d["docks"])
        overflow |= any(s["span"] in {dock["span"] for dock in d["docks"]} for s in rejected)
    assert width_witness and inactive and overflow and zero_witness
    assert all(len(d["docks"]) >= 3 for d in test)
    assert all(len(d["docks"]) <= 2 for d in training)


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _boundary_regression():
    """Hand-built rule member: odd spare cell, unequal guards, width ten,
    repeated identical cartridges, excess supply and an unmatched length.
    Expected placements are explicit slices, not obtained from solve.
    """
    grid = np.zeros((28, 30), dtype=np.int8)
    grid[:, 0] = 2
    grid[0, 1:] = np.where(np.arange(29) % 2 == 0, 1, 0)
    grid[0, 0] = 9

    def dock(r, c, span, lengths, center, turn, color):
        m = np.zeros((max(lengths)+1, span+4), dtype=np.int8)
        m[0] = color
        m[1:lengths[0]+1, 0] = color
        m[1:center+1, 2:2+span] = color
        m[1:lengths[1]+1, -1] = color
        physical = np.rot90(m, turn)
        grid[r:r+physical.shape[0], c:c+physical.shape[1]] = physical
        return m != 0

    dock(2, 2, 3, (4, 7), 1, 0, 3)
    dock(2, 11, 3, (5, 5), 1, 0, 4)
    dock(13, 2, 10, (3, 3), 1, 1, 5)
    a = np.array([[6, 7, 8], [9, 6, 7]], dtype=np.int8)
    b = np.array([[7, 6, 9], [8, 9, 6]], dtype=np.int8)
    extra = np.full((2, 3), 8, dtype=np.int8)
    wide = np.array([[6, 7, 8, 9, 1, 6, 7, 8, 9, 1],
                     [9, 6, 7, 8, 2, 9, 6, 7, 8, 2]], dtype=np.int8)
    source = [(10, 11, a), (14, 12, b), (18, 11, b.copy()),
              (22, 12, extra), (10, 18, wide), (24, 18, np.full((2, 4), 7, dtype=np.int8))]
    for r, c, block in source:
        grid[r:r+2, c:c+block.shape[1]] = block
    expected = grid.copy()
    for r, c, block in source:
        expected[r:r+2, c:c+block.shape[1]] = 0
    expected[4:6, 4:7] = a
    expected[4:6, 13:16] = b
    expected[6:8, 13:16] = b
    expected[15:25, 4:6] = np.rot90(wide, 1)
    d = _read(grid)
    assert [o["capacity"] for o in d["docks"]] == [1, 2, 1]
    assert np.all(expected[6, 4:7] == 0), "The one-cell remainder stays empty"
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(grid, turn)), np.rot90(expected, turn))
    # Broken key, a holed strip, a missing central prong and an occupied bay
    # are invalid inputs, rather than additional puzzle conventions.
    damaged = grid.copy()
    damaged[0, 2] = 1
    holed = grid.copy()
    holed[10, 12] = 0
    missing = grid.copy()
    missing[3, 4:7] = 0
    occupied = grid.copy()
    occupied[4:6, 4:7] = 6
    for malformed in (damaged, holed, missing, occupied):
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("A malformed input was accepted")


def self_test():
    """Bounded 42-seed suite, structural diversity, ablations and corruptions."""
    signatures, shapes, structures, spans, capacities, directions = set(), set(), set(), set(), set(), set()
    geometries, roles, populations, dock_counts, modes = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                roles.add((d["background"], d["origin"], d["fast"], d["slow"]))
                directions.add((d["bottom"], d["rotation"]))
                palette = {}
                structures.add(tuple(palette.setdefault(int(c), len(palette)) for c in pair["input"].flat))
                dock_counts.add(len(d["docks"]))
                populations.add(len(d["strips"]))
                for dock in d["docks"]:
                    spans.add(dock["span"])
                    capacities.add(dock["capacity"])
                    geometries.add((dock["span"], dock["runs"], dock["rotation"]))
                for strip in d["strips"]:
                    b = strip["block"]
                    modes.add("solid" if np.all(b == b.flat[0]) else
                              "stripe" if np.array_equal(b[0], b[1]) or np.all(b == b[:, :1])
                              else "pattern")
    assert len(signatures) == 42 and len(shapes) >= 30 and len(structures) >= 180
    assert len(geometries) >= 100 and len(roles) >= 100
    assert directions == {(b, k) for b in (False, True) for k in range(4)}
    assert {0, 1, 2, 3} <= capacities and {1, 2, 3, 4} <= dock_counts
    assert {2, 3, 4, 5, 6, 7, 8} <= spans and len(populations) >= 6
    assert modes == {"solid", "stripe", "pattern"}
    _boundary_regression()
    # Whole-input proper rotations commute with assembly, including square strips.
    pair = generate_puzzle(901, 2, 1)["train"][0]
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
    assert _puzzle_signature(generate_puzzle(902, 5, 4)) == _puzzle_signature(generate_puzzle(902, 5, 4))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal example counts accepted")
    bad = generate_puzzle(903, 2, 1)
    # Every corner, whether key or background, is protected by the rule.
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("A corrupted output was accepted")
    bad = generate_puzzle(904, 2, 1)
    bad["test"][0] = bad["train"][0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate inputs were accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "color_normalized_inputs": len(structures), "receiver_geometries": len(geometries),
            "key_configurations": len(directions), "source_counts": sorted(populations),
            "receiver_counts": sorted(dock_counts), "spans": sorted(spans),
            "capacities": sorted(capacities), "semantic_checks": "author-owned"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
