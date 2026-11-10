"""Pool a comb payload in chassis coordinates, then slide its full footprint.

Only 0 (wall), 1 (track), and 2 (dock) have fixed color meanings. Colors
3..9 are reassigned to chassis structure and teeth in each input.
"""
from collections import deque
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    answer = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = deque([first])
        cells = {first}
        while queue:
            r, c = queue.popleft()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    queue.append(p)
        answer.append(cells)
    return answer


def _bar_candidates(a, region):
    """Find the two-row chassis with distinct diagonal corner caps."""
    points = np.array(sorted(region))
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    box = a[lo[0]:hi[0], lo[1]:hi[1]]
    candidates = {}
    for length in range(4, 9):
        for vertical in (False, True):
            shape = (length, 2) if vertical else (2, length)
            if any(s > d for s, d in zip(shape, box.shape)):
                continue
            windows = np.lib.stride_tricks.sliding_window_view(box, shape)
            horizontal = windows.swapaxes(-1, -2) if vertical else windows
            base = horizontal[..., 0, 1]
            common = ((horizontal >= 3).all(axis=(-2, -1)) &
                      ((horizontal == base[..., None, None]).sum(axis=(-2, -1)) == 2*length-2))
            for x, y in (((0, 0), (1, length-1)), ((1, 0), (0, length-1))):
                cap_a = horizontal[..., x[0], x[1]]
                cap_b = horizontal[..., y[0], y[1]]
                valid = common & (cap_a != base) & (cap_b != base) & (cap_a != cap_b)
                for r, c in np.argwhere(valid):
                    r, c = int(r+lo[0]), int(c+lo[1])
                    arr = a[r:r+shape[0], c:c+shape[1]].copy()
                    cells = {(r+i, c+j) for i in range(shape[0]) for j in range(shape[1])}
                    if cells <= region:
                        candidates[(r, c, shape)] = {"r": r, "c": c, "arr": arr, "cells": cells}
    return list(candidates.values())


def _label_position(labels, label):
    return tuple(map(int, np.argwhere(labels == label)[0]))


def _edge(copy, side, slot, length):
    p = _label_position(copy["labels"], side*length+slot)
    q = _label_position(copy["labels"], (1-side)*length+slot)
    direction = p[0]-q[0], p[1]-q[1]
    return (copy["r"]+p[0], copy["c"]+p[1]), direction


def _observations(a, copy, length, structural):
    used = set(copy["cells"])
    observations = []
    for side in (0, 1):
        for slot in range(length):
            edge, direction = _edge(copy, side, slot, length)
            r, c = edge[0]+direction[0], edge[1]+direction[1]
            cells = []
            while 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and a[r, c] >= 3:
                cells.append((r, c))
                r, c = r+direction[0], c+direction[1]
            if cells:
                colors = {int(a[p]) for p in cells}
                _require(1 <= len(cells) <= 3 and len(colors) == 1, "invalid tooth")
                color = next(iter(colors))
                _require(color not in structural, "tooth reuses a structural color")
                observations.append((side, slot, len(cells), color))
                used.update(cells)
    colored = {p for p in copy["region"] if a[p] >= 3}
    _require(used == colored, "unexplained colored cells")
    _require(len({x[0] for x in observations}) <= 1, "teeth occupy both long sides")
    return observations


def _footprint(copy, canonical, side, code):
    length = canonical.shape[1]
    cells = {p: int(copy["arr"][p[0]-copy["r"], p[1]-copy["c"]]) for p in copy["cells"]}
    direction = None
    for slot, (depth, color) in code.items():
        p, direction = _edge(copy, side, slot, length)
        for k in range(1, depth+1):
            cells[(p[0]+k*direction[0], p[1]+k*direction[1])] = color
    # Each row of the canonical chassis has exactly one distinctive cap.
    values, counts = np.unique(canonical, return_counts=True)
    base = int(values[np.argmax(counts)])
    cap_slot = int(np.flatnonzero(canonical[side] != base)[0])
    anchor, direction = _edge(copy, side, cap_slot, length)
    return cells, direction, anchor


def _travel(a, cells, direction, anchor, goal):
    delta = goal[0]-anchor[0], goal[1]-anchor[1]
    target = delta[0]*direction[0]+delta[1]*direction[1]
    _require(target > 0 and delta == (target*direction[0], target*direction[1]),
             "dock is not forward of the active-side cap")

    def fits(step):
        for r, c in cells:
            p = r+step*direction[0], c+step*direction[1]
            if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and a[p] != 0):
                return False
        return True

    _require(fits(0), "completed vehicle does not fit its starting pocket")
    _require(fits(target), "completed vehicle does not fit the docking pocket")
    step = 0
    while step < target and fits(step+1):
        step += 1
    return step, target


def _decode(value):
    a = _grid(value)
    regions = _components(a != 0)
    _require(2 <= len(regions) <= 4, "expected two to four separate tracks")
    copies = []
    for region in regions:
        goals = [p for p in region if a[p] == 2]
        _require(len(goals) == 1, "each track needs exactly one dock")
        candidates = _bar_candidates(a, region)
        _require(len(candidates) == 1, "track must have one uniquely recognizable chassis")
        copy = candidates[0]
        copy.update(region=region, goal=goals[0])
        copies.append(copy)
    canonical = copies[0]["arr"]
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical)
    length = canonical.shape[1]
    structural = set(map(int, np.unique(canonical)))
    forms = [_orient(canonical, t) for t in range(8)]
    _require(len({_key(form) for form in forms}) == 8, "symmetric chassis")
    labels = np.arange(2*length).reshape(2, length)
    code, sides = {}, set()
    for copy in copies:
        matches = [t for t, form in enumerate(forms) if np.array_equal(form, copy["arr"])]
        _require(len(matches) == 1, "chassis copies disagree")
        copy["transform"] = matches[0]
        copy["labels"] = _orient(labels, matches[0])
        observations = _observations(a, copy, length, structural)
        copy["local_code"] = {}
        for side, slot, depth, color in observations:
            sides.add(side)
            _require(slot not in code or code[slot] == (depth, color), "conflicting tooth observations")
            code[slot] = (depth, color)
            copy["local_code"][slot] = (depth, color)
    _require(len(sides) == 1 and 1 <= len(code) <= 4, "missing or inconsistent active-side code")
    side = sides.pop()
    expected = a.copy()
    # Restore every observed vehicle before drawing final, completed shapes.
    expected[a >= 3] = 1
    for copy in copies:
        cells, direction, anchor = _footprint(copy, canonical, side, code)
        _require(set(cells) <= copy["region"], "payload crosses track boundary")
        _require(copy["goal"] not in cells, "dock overlaps the completed start")
        step, target = _travel(a, cells, direction, anchor, copy["goal"])
        terminal_cells = {(r+step*direction[0], c+step*direction[1]) for r, c in cells}
        _require(step == target or copy["goal"] not in terminal_cells,
                 "blocked footprint obscures the dock")
        for (r, c), color in cells.items():
            expected[r+step*direction[0], c+step*direction[1]] = color
        # Counterfactuals are measured from the returned pixels.
        body = {p: cells[p] for p in copy["cells"]}
        local = dict(body)
        for slot, (depth, color) in copy["local_code"].items():
            p, _ = _edge(copy, side, slot, length)
            for k in range(1, depth+1):
                local[(p[0]+k*direction[0], p[1]+k*direction[1])] = color
        local_step, _ = _travel(a, local, direction, anchor, copy["goal"])
        body_step, _ = _travel(a, body, direction, anchor, copy["goal"])
        coordinates = np.array(list(cells))
        lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
        rectangle = {(r, c): 3 for r in range(lo[0], hi[0]+1) for c in range(lo[1], hi[1]+1)}
        # The rectangle may be illegal at the dock in general family members.
        try:
            rectangle_step, _ = _travel(a, rectangle, direction, anchor, copy["goal"])
        except ValueError:
            rectangle_step = -1
        copy.update(step=step, target=target, direction=direction, full_cells=cells,
                    local_step=local_step, body_step=body_step, rectangle_step=rectangle_step)
    features = {
        "length": length, "count": len(copies), "code": code, "side": side,
        "depth": max(d for d, _ in code.values()), "shape": a.shape,
        "structural": structural, "copies": copies,
        "pooled": all(len(c["local_code"]) < len(code) for c in copies),
        "unhinted": any(not c["local_code"] for c in copies),
        "pool_changes_motion": any(c["local_step"] != c["step"] for c in copies),
        "teeth_change_motion": any(c["body_step"] != c["step"] for c in copies),
        "rectangle_changes_motion": any(c["rectangle_step"] != c["step"] for c in copies),
    }
    return expected, features


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions and the answer solely from this input."""
    return _decode(input_grid)[0]


def _template(length, palette):
    a = np.full((2, length), palette[0], dtype=np.int8)
    a[0, 0], a[1, -1] = palette[1], palette[2]
    return a


def _draw(a, template, code, top, left, observed=None):
    a[top:top+2, left:left+template.shape[1]] = template
    for slot in code if observed is None else observed:
        depth, color = code[slot]
        a[top+2:top+2+depth, left+slot] = color


def _road(rng, template, code, observed, blocked, rich=False, short=False):
    length, depth = template.shape[1], max(d for d, _ in code.values())
    low = 8+2*depth
    height = int(rng.integers(low, 15 if short else 25))
    width = length+int(rng.integers(2, 7))
    start = int(rng.integers(1, min(3, height-7-2*depth)+1))
    left = int(rng.integers(1, width-length))
    target = height-3-depth
    a = np.zeros((height, width), dtype=np.int8)
    a[1:-1, 1:-1] = 1
    slots = sorted(code)
    # The earliest finger reaches only the extreme tooth. In demonstrations
    # that tooth is short and missing locally, so both counterfactuals matter.
    from_left = not rich or bool(rng.integers(2))
    slot = slots[0] if from_left else slots[-1]
    first_row = None
    if blocked:
        first_row = int(rng.integers(start+3+depth, target))
        if from_left:
            a[first_row, 1:left+slot+1] = 0
        else:
            a[first_row, left+slot:width-1] = 0
    # Passable reductions outside the swept chassis, plus downstream fingers.
    if left > 1:
        r = 1
        while r < target:
            end = min(target, r+int(rng.integers(2, 6)))
            a[r:end, 1:int(rng.integers(1, left+1))] = 0
            r = end
    right = left+length
    if right < width-1:
        r = 1
        while r < target:
            end = min(target, r+int(rng.integers(2, 6)))
            a[r:end, int(rng.integers(right, width)):width-1] = 0
            r = end
    if blocked and rich:
        for r in range(first_row+2, target, 2):
            if rng.random() < .5:
                if rng.integers(2):
                    a[r, 1:left+int(rng.integers(length-1))+1] = 0
                else:
                    a[r, left+int(rng.integers(1, length)):width-1] = 0
    # Removing outside-lane pixels cannot destroy the central clear lane.
    # Clear the two endpoint footprints, including the pooled teeth.
    for top in (start, target):
        a[top:top+2, left:left+length] = 1
        for s, (d, _) in code.items():
            a[top+2:top+2+d, left+s] = 1
    a[target+1, left+length-1] = 2
    _draw(a, template, code, start, left, observed)
    # Construction-time answer uses direct canonical translations.
    footprint = [(i, j, int(template[i, j])) for i in range(2) for j in range(length)]
    footprint += [(2+k, s, color) for s, (d, color) in code.items() for k in range(d)]
    top = start
    while top < target and all(a[top+1+i, left+j] != 0 for i, j, _ in footprint):
        top += 1
    out = a.copy()
    out[a >= 3] = 1
    _draw(out, template, code, top, left)
    return a, out


def _pack(rng, roads):
    """Scatter disjoint track boxes; fall back to bounded shelves if necessary."""
    dims = [a.shape for a, _ in roads]
    if len(roads) == 2:
        horizontal = sum(w for _, w in dims) <= 30
        vertical = sum(h for h, _ in dims) <= 30
        if not horizontal and not vertical:
            raise ValueError("roads do not fit canvas")
        horizontal = horizontal and (not vertical or bool(rng.integers(2)))
        if horizontal:
            height, width = max(h for h, _ in dims), sum(w for _, w in dims)
            positions = [(int(rng.integers(height-h+1)), sum(dims[j][1] for j in range(i)))
                         for i, (h, w) in enumerate(dims)]
        else:
            height, width = sum(h for h, _ in dims), max(w for _, w in dims)
            positions = [(sum(dims[j][0] for j in range(i)), int(rng.integers(width-w+1)))
                         for i, (h, w) in enumerate(dims)]
    else:
        # Every short road dimension is <=14; four fit without a layout retry.
        rows = [max(dims[j][0] for j in range(i, min(i+2, len(dims)))) for i in (0, 2)]
        widths = [sum(dims[j][1] for j in range(i, min(i+2, len(dims)))) for i in (0, 2)]
        height, width = sum(rows), max(widths)
        _require(height <= 30 and width <= 30, "short roads do not fit")
        positions = []
        for i, (h, w) in enumerate(dims):
            shelf = i//2
            r = (rows[0] if shelf else 0)+int(rng.integers(rows[shelf]-h+1))
            c = (dims[i-1][1] if i % 2 else 0)
            c += int(rng.integers(width-widths[shelf]+1)) if i % 2 == 0 else 0
            # Offset the entire shelf consistently rather than overlapping boxes.
            if i % 2:
                c += positions[i-1][1]
            positions.append((r, c))
    # Spare canvas space supplies independently varying asymmetric margins.
    extra_h = int(rng.integers(31-height))
    extra_w = int(rng.integers(31-width))
    offset_r, offset_c = int(rng.integers(extra_h+1)), int(rng.integers(extra_w+1))
    inp = np.zeros((height+extra_h, width+extra_w), dtype=np.int8)
    out = inp.copy()
    fallback = [(r+offset_r, c+offset_c) for r, c in positions]
    # Try free placement of the actual irregular floor regions. Wall margins
    # can overlap, but different tracks must retain a separating wall cell.
    order = sorted(range(len(roads)), key=lambda i: -int((roads[i][0] != 0).sum()))
    scattered = None
    for _ in range(8):
        forbidden = np.zeros(inp.shape, dtype=bool)
        candidate_positions = [None]*len(roads)
        for i in order:
            a = roads[i][0]
            mask = a != 0
            h, w = a.shape
            for _ in range(70):
                r = int(rng.integers(inp.shape[0]-h+1))
                c = int(rng.integers(inp.shape[1]-w+1))
                if forbidden[r:r+h, c:c+w][mask].any():
                    continue
                candidate_positions[i] = (r, c)
                occupied = np.zeros(inp.shape, dtype=bool)
                occupied[r:r+h, c:c+w] = mask
                forbidden |= occupied
                forbidden[1:] |= occupied[:-1]
                forbidden[:-1] |= occupied[1:]
                forbidden[:, 1:] |= occupied[:, :-1]
                forbidden[:, :-1] |= occupied[:, 1:]
                break
            else:
                break
        else:
            scattered = candidate_positions
            break
    positions = scattered if scattered is not None else fallback
    for (a, b), (r, c) in zip(roads, positions):
        mask = a != 0
        inp[r:r+a.shape[0], c:c+a.shape[1]][mask] = a[mask]
        out[r:r+b.shape[0], c:c+b.shape[1]][mask] = b[mask]
    return {"input": inp, "output": out}


def _make_pair(rng, shift, split, demonstration, used):
    for attempt in range(80):
        try:
            if shift == 2:
                count = 2 if split == "train" else int(rng.integers(3, 5))
            else:
                count = int(rng.integers(2, 5))
            length = int(rng.integers(4, 6) if shift == 1 and split == "train" else
                         rng.integers(6, 9) if shift == 1 else rng.integers(4, 9))
            depth = (2 if split == "train" else 3) if shift == 0 else int(rng.integers(2, 4))
            slot_count = int(rng.integers(2, min(4, length)+1))
            if shift == 2 and split == "test":
                slot_count = int(rng.integers(3, min(4, length)+1))
            slots = sorted(map(int, rng.choice(length, slot_count, replace=False)))
            palette = list(map(int, rng.permutation(np.arange(3, 10))))
            template = _template(length, palette[:3])
            depths = list(map(int, rng.integers(1, depth+1, size=slot_count)))
            depths[0], depths[-1] = depth-1, depth
            colors = palette[3:3+slot_count]
            if demonstration is None and slot_count > 2 and rng.random() < .25:
                colors[-1] = colors[0]
            code = dict(zip(slots, zip(depths, colors)))
            observations = [set() for _ in range(count)]
            if demonstration == "rotate":
                observations[1] = set(slots)  # A complete reference, plus unhinted copies.
            else:
                # The blocking extreme tooth is known only on another vehicle.
                observations[1].add(slots[0])
                observations[0].add(slots[-1])
                for s in slots[1:-1]:
                    observations[int(rng.integers(min(2, count)))].add(s)
                if demonstration is None and rng.random() < .3:
                    observations[int(rng.integers(count))].add(int(rng.choice(slots)))
            base = int(rng.integers(8))
            transforms = [base, base ^ 4] if demonstration == "pool" else (
                [base, (base//4)*4+(base+1) % 4] if demonstration == "rotate" else
                list(map(int, rng.integers(8, size=2))))
            transforms += list(map(int, rng.integers(8, size=count-2)))
            roads = []
            for i in range(count):
                blocked = i == 0 or (i >= 2 and bool(rng.integers(2)))
                a, b = _road(rng, template, code, observations[i], blocked,
                             rich=demonstration is None, short=count >= 3 or attempt >= 40)
                roads.append((_orient(a, transforms[i]), _orient(b, transforms[i])))
            rng.shuffle(roads)
            pair = _pack(rng, roads)
            signature = _key(pair["input"])
            _require(signature not in used, "duplicate input")
            predicted, f = _decode(pair["input"])
            _require(np.array_equal(predicted, pair["output"]), "construction/decoding disagreement")
            _require(any(c["step"] < c["target"] for c in f["copies"]), "collision was erased")
            _require(any(c["step"] == c["target"] for c in f["copies"]), "docking missing")
            _require(all(c["step"] > 0 for c in f["copies"]), "movement missing")
            if demonstration == "pool":
                _require(f["pooled"] and f["pool_changes_motion"], "pooling does not determine motion")
                _require(f["rectangle_changes_motion"], "bounding rectangle shortcut survives")
            if shift == 2 and split == "test":
                _require(f["pooled"] and f["unhinted"], "test needs pooled code and an unhinted copy")
            used.add(signature)
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            demonstration = ("pool", "rotate")[i] if split == "train" and i < 2 else None
            puzzle[split].append(_make_pair(rng, shift, split, demonstration, used))
    # Vary demonstration order without weakening either evidence requirement.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        n, m = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
        seen, training = set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                _require(inp.shape == out.shape, "shape changed")
                signature = _key(inp)
                _require(signature not in seen, "duplicate input")
                seen.add(signature)
                expected, f = _decode(inp)
                _require(np.array_equal(expected, out), "incorrect completed footprint motion")
                _require(np.array_equal(inp == 0, out == 0), "walls changed")
                _require(all(c["step"] > 0 for c in f["copies"]), "generated vehicle never moved")
                _require(any(c["step"] < c["target"] for c in f["copies"]), "no collision example")
                _require(any(c["step"] == c["target"] for c in f["copies"]), "no docking example")
                # Count exactly one completed copy of each body and its teeth.
                tooth_area = sum(d for d, _ in f["code"].values())
                _require(int((out >= 3).sum()) == f["count"]*(2*f["length"]+tooth_area), "object mass wrong")
                for c in f["copies"]:
                    _require((int(out[c["goal"]]) == 2) == (c["step"] < c["target"]), "dock preservation wrong")
                if split == "train":
                    training.append(f)
        _require(any(f["pooled"] and f["pool_changes_motion"] for f in training), "missing pooling evidence")
        _require(any(f["rectangle_changes_motion"] for f in training), "missing nonrectangular-footprint evidence")
        _require(any(len({c["direction"] for c in f["copies"]}) > 1 for f in training), "missing orientation transfer")
        _require(any(any(c["transform"] >= 4 for c in f["copies"]) for f in training), "missing reflection evidence")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise AssertionError(str(exc)) from exc


def _fingerprint(p):
    return tuple(_key(np.asarray(pair[role])) for s in ("train", "test")
                 for pair in p[s] for role in ("input", "output"))


def _fixture():
    """Specified coordinates: actual stop 4, body-only 5, rectangle stop 2."""
    template = _template(5, [3, 4, 5])
    code = {0: (1, 6), 4: (3, 7)}
    roads = []
    for i in range(2):
        a = np.zeros((15, 9), dtype=np.int8)
        a[1:-1, 1:-1] = 1
        if i == 0:
            a[7, 1:3] = 0
        a[10, 6] = 2
        _draw(a, template, code, 1, 2, {4} if i == 0 else {0})
        out = a.copy()
        out[a >= 3] = 1
        _draw(out, template, code, 4 if i == 0 else 9, 2)
        roads.append((_orient(a, 0 if i == 0 else 4), _orient(out, 0 if i == 0 else 4)))
    return {"input": np.concatenate([x[0] for x in roads], axis=1),
            "output": np.concatenate([x[1] for x in roads], axis=1)}


def self_test() -> None:
    fixture = _fixture()
    predicted, f = _decode(fixture["input"])
    assert np.array_equal(predicted, fixture["output"])
    assert f["pooled"] and f["pool_changes_motion"] and f["rectangle_changes_motion"]
    blocked = next(c for c in f["copies"] if c["step"] < c["target"])
    assert (blocked["step"], blocked["local_step"], blocked["body_step"], blocked["rectangle_step"]) == (3, 4, 4, 1)
    # Changing only the remote clue's depth changes the first vehicle's stop.
    changed = fixture["input"].copy()
    changed[4, 15] = 6
    _, cf = _decode(changed)
    cb = next(c for c in cf["copies"] if c["step"] < c["target"])
    assert cb["step"] == 2
    # A farther finger can hit the long tooth before a nearer finger hits
    # the short tooth: collision order is about the translated footprint.
    farther = fixture["input"].copy()
    farther[8, 6:8] = 0
    expected_farther = fixture["output"].copy()
    expected_farther[8, 6:8] = 0
    expected_farther[:, :9][expected_farther[:, :9] >= 3] = 1
    _draw(expected_farther, _template(5, [3, 4, 5]), {0: (1, 6), 4: (3, 7)}, 3, 2)
    assert np.array_equal(solve(farther), expected_farther)
    # Compatible restrained member: one observed tooth, depth one.
    single = fixture["input"].copy()
    single[single == 7] = 1
    expected_single = fixture["output"].copy()
    expected_single[expected_single == 7] = 1
    assert np.array_equal(solve(single), expected_single)
    for t in range(8):
        assert np.array_equal(solve(_orient(fixture["input"], t)), _orient(fixture["output"], t))
    palette = np.arange(10, dtype=np.int8)
    palette[3:] = palette[3:][::-1]
    assert np.array_equal(solve(palette[fixture["input"]]), palette[fixture["output"]])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    structures = set()
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                _, ff = _decode(pair["input"])
                features.append(ff)
                mapping = {}
                normalized = np.array([[mapping.setdefault(int(x), len(mapping)) for x in row]
                                       for row in pair["input"]], dtype=np.int8)
                structures.add(_key(normalized))
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len(structures) >= 180
    assert len({f["shape"] for f in features}) >= 30
    assert {f["length"] for f in features} == set(range(4, 9))
    assert {f["count"] for f in features} == {2, 3, 4}
    assert {f["depth"] for f in features} == {2, 3}
    assert {len(f["code"]) for f in features} == {2, 3, 4}
    assert {c["direction"] for f in features for c in f["copies"]} == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    assert all(any(_decode(x["input"])[1]["pool_changes_motion"] for x in p["train"]) for p in puzzles)
    assert any(len({color for _, color in f["code"].values()}) < len(f["code"]) for f in features)
    assert any(any(b-a == 1 for a, b in zip(sorted(f["code"]), sorted(f["code"])[1:])) for f in features)
    assert any(0 in f["code"] or f["length"]-1 in f["code"] for f in features)
    assert {color for f in features for color in f["structural"]} == set(range(3, 10))
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(puzzles[0])
    assert _fingerprint(generate_puzzle(1)) != _fingerprint(puzzles[0])
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(100+n*4+m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = 9
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
