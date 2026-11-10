"""Comb-operated shutter maze: move ordered apertures, then collect through them.

Only NumPy and the standard library are used. No parent code or saved examples
are loaded. All color meanings and mechanical geometry are visible in one input.
"""
from collections import deque
import hashlib
import numpy as np


CAPACITY = 9
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _collector(a, background):
    found = []
    for color in np.unique(a):
        rr, cc = np.where(a == color)
        if len(rr) != 8 or rr.max() - rr.min() != 2 or cc.max() - cc.min() != 2:
            continue
        r, c = int(rr.min()), int(cc.min())
        block = a[r:r + 3, c:c + 3]
        center = int(block[1, 1])
        if center in (int(color), background):
            continue
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        if np.all(block[mask] == color):
            found.append((r, c, int(color), center))
    _require(len(found) == 1, "one unique marked collector is required")
    return found[0]


def _counter_mask(shape, pos):
    m = np.zeros(shape, dtype=bool)
    r, c = pos
    m[r:r + 3, c:c + 3] = True
    return m


def _canonical_read(a, bg, structure, counter):
    """Read one view with the key below a downward-open housing."""
    r, c, slot, window = counter
    colors = set(map(int, np.unique(a)))
    remaining = colors - {bg, structure, slot, window}
    _require(len(remaining) == 1, "five distinct visible roles are required")
    payload = next(iter(remaining))
    rr, cc = np.where(a == structure)
    top, key_base, left, right = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    _require(right - left >= 10 and top >= 1 and key_base < a.shape[0] - 1, "bad housing extent")
    _require(np.all(a[top, left:right + 1] == structure), "roof is not solid")
    ends = []
    for col in (left, right):
        row = top
        while row <= key_base and a[row, col] == structure:
            row += 1
        ends.append(row - 1)
    _require(ends[0] == ends[1] and ends[0] - top >= 6, "housing sides must end at one docking line")
    dock = ends[0]
    _require(dock < key_base and np.all(a[dock, left + 1:right] == bg), "docking line must be empty")
    _require(np.all(a[key_base, left + 1:right] == structure), "key base must exactly fit housing")
    lanes = np.flatnonzero(a[key_base - 1, left + 1:right] == structure) + left + 1
    _require(2 <= len(lanes) <= 4, "expected two through four shutters")
    edges = [left, *map(int, lanes), right]
    _require(all(b - x >= 4 for x, b in zip(edges, edges[1:])), "chambers need at least three columns")
    external = np.zeros(a.shape, dtype=bool)
    external[key_base, left + 1:right] = True
    heights, stacks = [], []
    pinmask = np.zeros(a.shape, dtype=bool)
    for col in lanes:
        row = key_base - 1
        while row >= 0 and a[row, col] == structure:
            row -= 1
        h = key_base - row - 1
        _require(1 <= h <= dock - top - 2 and row >= dock + 1, "bad external tooth")
        external[key_base - h:key_base, col] = True
        seg = a[top + 1:dock, col]
        occupied = np.flatnonzero(seg != bg)
        _require(len(occupied) > 0 and np.array_equal(occupied, np.arange(len(occupied))), "gapped shutter")
        seq = tuple(map(int, seg[:len(occupied)]))
        _require(set(seq) <= {structure, window} and window in seq, "shutter needs marked aperture cells")
        capacity = dock - top - h - 1
        _require(capacity <= len(seq) <= dock - top - 1, "pin must contact or overlap its tooth")
        heights.append(h)
        stacks.append(seq)
        pinmask[top + 1:top + 1 + len(seq), col] = True
    main = np.zeros(a.shape, dtype=bool)
    main[top:dock + 1, left:right + 1] = True
    _require(np.all(a[~main & ~external] == bg), "unexpected exterior object")
    _require(np.all(a[external] == structure), "malformed key")
    _require(top < r and r + 2 < dock and left < c and c + 2 < right, "collector outside housing")
    _require(not any(c <= col <= c + 2 for col in lanes), "collector must fit within one chamber")
    for col, height in zip(lanes, heights):
        _require(not np.any(a[dock - height:dock, col] == payload), "payload overlaps an incoming tooth")
    cm = _counter_mask(a.shape, (r, c))
    _require(not np.any(cm & pinmask), "collector overlaps a shutter")
    allowed_windows = pinmask.copy()
    allowed_windows[r + 1, c + 1] = True
    _require(not np.any((a == window) & ~allowed_windows), "aperture color outside shutters/marker")
    return dict(bg=bg, structure=structure, window=window, slot=slot, payload=payload,
                top=top, dock=dock, key_base=key_base, left=left, right=right,
                lanes=tuple(map(int, lanes)), heights=tuple(heights), stacks=tuple(stacks),
                pos=(r, c))


def _read(value):
    original = _grid(value)
    bg = int(original[0, 0])
    _require(all(np.all(edge == bg) for edge in (original[0], original[-1], original[:, 0], original[:, -1])),
             "a clean background border is required")
    _require(len(np.unique(original)) == 5, "expected five role colors")
    readings = []
    for rotation in range(4):
        a = np.rot90(original, rotation).copy()
        try:
            counter = _collector(a, bg)
        except ValueError:
            continue
        for s in sorted(set(map(int, np.unique(a))) - {bg, counter[2], counter[3]}):
            try:
                f = _canonical_read(a, bg, s, counter)
            except ValueError:
                continue
            f["rotation"] = rotation
            readings.append((a, f))
    _require(len(readings) == 1, "input must have one mechanical interpretation")
    return readings[0]


def _move_pixels(a, f, frozen=False):
    """Copy the key; push actual ordered pin cells one step at a time."""
    b = a.copy()
    t, d, k, l, r = (f[x] for x in ("top", "dock", "key_base", "left", "right"))
    bg, s = f["bg"], f["structure"]
    key = [(int(y), int(x)) for y, x in np.argwhere((a == s) & (np.indices(a.shape)[0] > d))]
    for y, x in key:
        b[y, x] = bg
    for col in f["lanes"]:
        b[t + 1:d, col] = bg
    for y, x in key:
        b[y + d - k, x] = s
    for col, height, seq in zip(f["lanes"], f["heights"], f["stacks"]):
        cells = [(t + 1 + i, color) for i, color in enumerate(seq)]
        if not frozen:
            while cells[-1][0] >= d - height:
                cells = [(row - 1, color) for row, color in cells if row - 1 > t]
        for row, color in cells:
            if row < d - height:
                b[row, col] = color
    return b


def _distances(a, f, center=False):
    distance = np.full(a.shape, -1, dtype=np.int16)
    r, c = f["pos"]
    queue = deque([(r + 1, c + 1)] if center else [(r + i, c + j) for i in range(3) for j in range(3)])
    for cell in queue:
        distance[cell] = 0
    h, w = a.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in STEPS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and a[yy, xx] != f["structure"] and distance[yy, xx] < 0:
                distance[yy, xx] = distance[y, x] + 1
                queue.append((yy, xx))
    return distance


def _selection(a, f, distance, strict=True):
    supply = (a == f["payload"]) & (distance >= 0)
    levels = np.sort(distance[supply])
    if len(levels) > CAPACITY and strict:
        _require(levels[8] < levels[9], "ambiguous capacity frontier")
    # Non-strict mode is diagnostic only and includes a complete tied frontier.
    threshold = int(levels[min(8, len(levels) - 1)]) if len(levels) else -1
    return supply & (distance <= threshold)


def _display_output(mechanical, f, rotation, selected):
    b = mechanical.copy()
    b[selected] = f["bg"]
    b = np.rot90(b, rotation).copy()
    cm = np.rot90(_counter_mask(mechanical.shape, f["pos"]), rotation)
    rr, cc = np.where(cm)
    r, c = int(rr.min()), int(cc.min())
    for i in range(int(selected.sum())):
        b[r + i // 3, c + i % 3] = f["payload"]
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, shifts, routes, selection and slot order from this input."""
    a, f = _read(input_grid)
    mechanical = _move_pixels(a, f)
    chosen = _selection(mechanical, f, _distances(mechanical, f))
    return _display_output(mechanical, f, -f["rotation"], chosen)


def _direct_metric(shape, pos):
    rr, cc = np.indices(shape)
    r, c = pos
    return np.maximum(r - rr, 0) + np.maximum(rr - r - 2, 0) + np.maximum(c - cc, 0) + np.maximum(cc - c - 2, 0)


def _constructed_mechanics(a, f):
    """Generation's direct placement formula; solver uses iterative pushing."""
    b = a.copy()
    t, d = f["top"], f["dock"]
    b[d + 1:] = f["bg"]
    for col in f["lanes"]:
        b[t + 1:d, col] = f["bg"]
    b[d, f["left"] + 1:f["right"]] = f["structure"]
    for col, height, seq in zip(f["lanes"], f["heights"], f["stacks"]):
        b[d - height:d, col] = f["structure"]
        start = d - height - len(seq)
        for i, value in enumerate(seq):
            if start + i > t:
                b[start + i, col] = value
    return b


def _layout(rng, settings, split, fallback=False):
    mode = settings["shift"]
    if mode == "lanes":
        n = int(rng.integers(2, 4)) if split == "train" else 4
    else:
        n = int(rng.integers(2, 5))
    lo, hi = (8, 10) if mode == "depth" and split == "train" else (12, 15) if mode == "depth" else (8, 13)
    span = int(rng.integers(lo, hi + 1))
    t, left = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    widths = [5] + [3] * n if fallback else rng.integers(3, 7 if n < 4 else 5, n + 1).tolist()
    # Width reduction retains at least three free columns in every chamber.
    while sum(widths) + n + 2 + left + 1 > 30:
        choices = [i for i, width in enumerate(widths) if width > 3]
        widths[int(rng.choice(choices))] -= 1
    lanes, edges, col = [], [left], left
    for width in widths[:-1]:
        col += width + 1
        lanes.append(col)
        edges.append(col)
    right = col + widths[-1] + 1
    edges.append(right)
    d = t + span
    cap = min(7, span - 4)
    profile = str(rng.choice(("random", "flat", "rising", "falling", "ends")))
    if mode == "profile":
        profile = "flat" if split == "test" else str(rng.choice(("rising", "falling", "ends", "random")))
    heights = rng.integers(1, cap + 1, n).tolist()
    if profile == "flat":
        heights = [int(rng.integers(1, cap + 1))] * n
    elif profile in ("rising", "falling"):
        heights[0], heights[-1] = 1, cap
        heights.sort(reverse=profile == "falling")
    elif profile == "ends":
        heights[0] = heights[-1] = cap
    if mode == "profile" and split == "train" and len(set(heights)) == 1:
        heights[0] = 1 if heights[0] != 1 else cap
    if fallback:
        heights = [2] * n if mode == "profile" and split == "test" else [2 + i % 2 for i in range(n)]
    k = d + max(heights) + int(rng.integers(2, 4))
    bottom_margin = int(rng.integers(1, min(2, 29 - k) + 1))
    right_margin = int(rng.integers(1, min(2, 29 - right) + 1))
    colors = settings["colors"] if not settings["vary_colors"] else tuple(map(int, rng.choice(10, 5, replace=False)))
    bg, s, window, slot, payload = colors
    a = np.full((k + bottom_margin + 1, right + right_margin + 1), bg, dtype=np.int8)
    a[t, left:right + 1] = s
    a[t:d + 1, left] = a[t:d + 1, right] = s
    a[k, left + 1:right] = s
    stacks = []
    opening_lane = 0 if fallback else int(rng.integers(n))
    for i, (col, height) in enumerate(zip(lanes, heights)):
        a[k - height:k, col] = s
        shift = int(rng.integers(height + 1))
        if i == opening_lane or fallback:
            shift = max(1, shift)
        if fallback:
            shift = 1 if i == 0 else height
        capacity = span - height - 1
        seq = [s] * (capacity + shift)
        if fallback and i > 0:
            seq[int(rng.integers(shift))] = window  # aperture is lost through roof
        elif i == opening_lane:
            seq[-1] = window  # old aperture is buried by the incoming tooth
        else:
            if rng.random() < .16 and shift:
                seq[int(rng.integers(shift))] = window
            else:
                count = int(rng.integers(1, min(3, capacity) + 1))
                for index in rng.choice(capacity, count, replace=False):
                    seq[int(index) + shift] = window
                if shift and rng.random() < .4:
                    seq[int(rng.integers(shift))] = window
        stacks.append(tuple(seq))
        a[t + 1:t + 1 + len(seq), col] = seq
    chamber = 0 if fallback else int(rng.integers(n + 1))
    cr = t + 1 if fallback else int(rng.integers(t + 1, d - 2))
    cc = edges[chamber] + 1 if fallback else int(rng.integers(edges[chamber] + 1, edges[chamber + 1] - 2))
    cm = _counter_mask(a.shape, (cr, cc))
    if not fallback:
        # Horizontal shelves may have one/two-cell doors or be fully closed.
        for x0, x1 in zip(edges, edges[1:]):
            count = int(rng.choice((0, 1, 2), p=(.38, .43, .19)))
            rows = rng.choice(np.arange(t + 2, d - 1), count, replace=False)
            for row in rows:
                if np.any(cm[row, x0 + 1:x1]):
                    continue
                a[row, x0 + 1:x1] = s
                if rng.random() < .88:
                    door = int(rng.integers(x0 + 1, x1))
                    a[row, door:min(x1, door + int(rng.integers(1, 3)))] = bg
    a[cr:cr + 3, cc:cc + 3] = slot
    a[cr + 1, cc + 1] = window
    f = dict(bg=bg, structure=s, window=window, slot=slot, payload=payload,
             top=t, dock=d, key_base=k, left=left, right=right,
             lanes=tuple(lanes), heights=tuple(heights), stacks=tuple(stacks), pos=(cr, cc))
    return a, f


def _pick(rng, mask, count, style, forced=()):
    available = set(map(int, np.flatnonzero(mask)))
    chosen = list(dict.fromkeys(map(int, forced)))
    _require(set(chosen) <= available and len(chosen) <= count <= len(available), "invalid supply request")
    available.difference_update(chosen)
    w = mask.shape[1]
    anchor = chosen[0] if chosen else int(rng.choice(sorted(available))) if available else 0
    ar, ac = divmod(anchor, w)
    horizontal = bool(rng.integers(2))
    while len(chosen) < count:
        cells = np.array(sorted(available), dtype=int)
        if style == "dispersed":
            cell = int(rng.choice(cells))
        else:
            rr, cc = cells // w, cells % w
            score = (rr - ar) ** 2 + (cc - ac) ** 2
            if style == "line":
                score = np.abs(rr - ar) * w + np.abs(cc - ac) if horizontal else np.abs(cc - ac) * mask.shape[0] + np.abs(rr - ar)
            cell = int(rng.choice(cells[score == score.min()]))
        chosen.append(cell)
        available.remove(cell)
    return chosen


def _place_supply(rng, a, f, mechanical, spec):
    distance = _distances(mechanical, f)
    old_distance = _distances(_move_pixels(a, f, frozen=True), f)
    cm = _counter_mask(a.shape, f["pos"])
    pool = (a == f["bg"]) & (mechanical == f["bg"]) & ~cm
    interior = np.zeros(a.shape, dtype=bool)
    interior[f["top"] + 1:f["dock"], f["left"] + 1:f["right"]] = True
    pool &= interior
    active = pool & (distance >= 0)
    inactive = pool & (distance < 0)
    opened = active & (old_distance < 0)
    load = spec["load"]
    if int(active.sum()) < load or (spec["unreachable"] and not inactive.any()) or (spec["opens"] and not opened.any()):
        return None
    style = str(rng.choice(("dispersed", "compact", "line")))
    if load <= 9:
        forced = [int(rng.choice(np.flatnonzero(opened)))] if spec["opens"] else []
        supply = _pick(rng, active, load, style, forced)
    else:
        metric = _direct_metric(a.shape, f["pos"])
        center_distance = _distances(mechanical, f, center=True)
        supply = None
        for anchor in rng.permutation(np.flatnonzero(active)):
            anchor = int(anchor)
            near = active & (distance <= distance.flat[anchor]) & (center_distance <= center_distance.flat[anchor])
            far = active & (distance > distance.flat[anchor]) & (center_distance > center_distance.flat[anchor])
            if int(near.sum()) < 9 or int(far.sum()) < load - 9:
                continue
            choices = far & (metric < metric.flat[anchor]) if spec["detour"] else far
            opening = opened & near
            if not choices.any() or (spec["opens"] and not opening.any()):
                continue
            forced = [anchor]
            if spec["opens"]:
                forced.append(int(rng.choice(np.flatnonzero(opening))))
            supply = _pick(rng, near, 9, style, forced)
            supply += _pick(rng, far, load - 9, style, [int(rng.choice(np.flatnonzero(choices)))])
            break
        if supply is None:
            return None
    minimum = 1 if spec["unreachable"] or load == 0 else 0
    maximum = min(8, int(inactive.sum()))
    if minimum > maximum:
        return None
    count = int(rng.integers(minimum, maximum + 1))
    distraction = _pick(rng, inactive, count, str(rng.choice(("compact", "dispersed", "line"))))
    a = a.copy()
    a.flat[supply + distraction] = f["payload"]
    # Every example visibly identifies the payload role, even for zero collection.
    if not supply and not distraction:
        return None
    return a


def _make_pair(rng, settings, split, spec, seen):
    for attempt in range(80):
        a, f = _layout(rng, settings, split, fallback=attempt >= 64)
        losses = [len(seq) - (f["dock"] - f["top"] - height - 1)
                  for seq, height in zip(f["stacks"], f["heights"])]
        if spec.get("contact", False):
            if len(set(losses)) < 2 or not any(loss < height and f["window"] in seq[loss:]
                                             for seq, height, loss in zip(f["stacks"], f["heights"], losses)):
                continue
        if spec.get("clip", False):
            if not any(f["window"] in seq[:loss] for seq, loss in zip(f["stacks"], losses)):
                continue
        mechanical = _constructed_mechanics(a, f)
        a = _place_supply(rng, a, f, mechanical, spec)
        if a is None:
            continue
        mechanical = _constructed_mechanics(a, f)
        selected = _selection(mechanical, f, _distances(mechanical, f))
        rotation = settings["orientation"]
        inp = np.rot90(a, rotation).copy()
        if _signature(inp) in seen:
            continue
        # Protect unique motif recognition without assuming construction metadata.
        try:
            _read(inp)
        except ValueError:
            continue
        return {"input": inp, "output": _display_output(mechanical, f, rotation, selected)}
    raise RuntimeError("bounded constructive sampling failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    settings = dict(colors=tuple(map(int, rng.choice(10, 5, replace=False))),
                    vary_colors=bool(rng.random() < .7), orientation=int(rng.integers(4)),
                    shift=str(rng.choice(("lanes", "depth", "profile"))))
    def random_spec():
        u = rng.random()
        load = 0 if u < .05 else int(rng.integers(10, 25)) if u < .4 else int(rng.integers(1, 10))
        return dict(load=load, unreachable=bool(load == 0 or rng.random() < .5), detour=False, opens=False)
    # Two shuffled demonstrations suffice even at the minimum requested count.
    specs = [dict(load=int(rng.choice((2, 4))), unreachable=True, detour=False, opens=True),
             dict(load=int(rng.integers(10, 20)), unreachable=True, detour=True, opens=True)]
    specs[int(rng.integers(2))]["clip"] = True
    specs[int(rng.integers(2))]["contact"] = True
    specs.extend(random_spec() for _ in range(num_train - 2))
    rng.shuffle(specs)
    result, seen = {"train": [], "test": []}, set()
    for split, requests in (("train", specs), ("test", [random_spec() for _ in range(num_test)])):
        for spec in requests:
            pair = _make_pair(rng, settings, split, spec, seen)
            seen.add(_signature(pair["input"]))
            result[split].append(pair)
    return result


def _features(value):
    a, f = _read(value)
    b = _move_pixels(a, f)
    distance = _distances(b, f)
    selected = _selection(b, f, distance)
    old = _distances(_move_pixels(a, f, frozen=True), f)
    supply = a == f["payload"]
    retained = supply & (distance >= 0) & ~selected
    metric = _direct_metric(a.shape, f["pos"])
    shifts = tuple(len(seq) - (f["dock"] - f["top"] - height - 1)
                   for seq, height in zip(f["stacks"], f["heights"]))
    clipped = sum(seq[i] == f["window"] for seq, shift in zip(f["stacks"], shifts) for i in range(shift))
    return dict(lanes=len(f["lanes"]), span=f["dock"] - f["top"], heights=f["heights"],
                shifts=shifts, clipped_windows=clipped, selected=int(selected.sum()),
                contact_witness=bool(len(set(shifts)) > 1 and any(shift < height and f["window"] in seq[shift:]
                                     for seq, height, shift in zip(f["stacks"], f["heights"], shifts))),
                reachable=int((supply & (distance >= 0)).sum()), unreachable=int((supply & (distance < 0)).sum()),
                opened_selected=int((selected & (old < 0)).sum()),
                path_reversal=bool(selected.any() and retained.any() and metric[selected].max() > metric[retained].min()),
                max_distance=int(distance.max()), rotation=f["rotation"],
                palette=tuple(f[k] for k in ("bg", "structure", "window", "slot", "payload")),
                shelf_cells=int(np.count_nonzero(a[f["top"] + 1:f["dock"], f["left"] + 1:f["right"]] == f["structure"]))
                - sum(seq.count(f["structure"]) for seq in f["stacks"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, orientations = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a, f = _read(inp)
            b = _move_pixels(a, f)
            _require(np.array_equal(b, _constructed_mechanics(a, f)), "pin sequence/contact mismatch")
            dist = _distances(b, f)
            selected = _selection(b, f, dist)
            _require(np.array_equal(selected, _selection(b, f, _distances(b, f, center=True))), "distance-origin ambiguity")
            prediction = _display_output(b, f, -f["rotation"], selected)
            _require(np.array_equal(prediction, out), "wrong mechanical/collection output")
            _require(not np.array_equal(inp, out), "the key must move")
            _require(np.count_nonzero(inp == f["payload"]) == np.count_nonzero(out == f["payload"]), "payload conservation failed")
            # Every barrier is roof-to-base solid except its translated windows.
            for col, height, seq in zip(f["lanes"], f["heights"], f["stacks"]):
                capacity = f["dock"] - f["top"] - height - 1
                _require(tuple(map(int, b[f["top"] + 1:f["dock"] - height, col])) == seq[-capacity:], "ordered suffix lost")
                _require(np.all(b[f["dock"] - height:f["dock"] + 1, col] == f["structure"]), "tooth contact failed")
            feature = _features(inp)
            orientations.add(feature["rotation"])
            if split == "train":
                training.append(feature)
    _require(len(orientations) == 1, "generated puzzles keep one physical orientation")
    _require(any(x["selected"] in (2, 4) and x["unreachable"] and x["opened_selected"] for x in training), "missing partial/selective/moving-gate evidence")
    _require(any(x["reachable"] > 9 and x["path_reversal"] and x["unreachable"] and x["opened_selected"] for x in training), "missing capacity/path/moving-gate evidence")
    _require(any(x["clipped_windows"] for x in training), "training must expose aperture clipping")
    _require(any(x["contact_witness"] for x in training), "training must distinguish contact from tooth-height-only displacement")


def self_test():
    """Bounded, streaming tests: no files, retained grids, logs or trace payloads."""
    signatures, shapes, profiles, palettes, rotations, spans = set(), set(), set(), set(), set(), set()
    selected_counts, shifts, lane_counts = set(), set(), set()
    clipped = reversals = stationary = shelves = opened = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for split in puzzle.values() for p in split))
        for split in puzzle.values():
            for pair in split:
                a = pair["input"].copy()
                _require(np.array_equal(solve(a), pair["output"]), "solver mismatch")
                _require(np.array_equal(a, pair["input"]), "solver mutated its input")
                f = _features(a)
                shapes.add(a.shape)
                profiles.add(f["heights"])
                palettes.add(f["palette"])
                rotations.add(f["rotation"])
                spans.add(f["span"])
                lane_counts.add(f["lanes"])
                shifts.update(f["shifts"])
                selected_counts.add(f["selected"])
                clipped += f["clipped_windows"] > 0
                reversals += f["path_reversal"]
                stationary += 0 in f["shifts"]
                shelves += f["shelf_cells"] > 0
                opened += f["opened_selected"] > 0
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(profiles) >= 40 and len(palettes) >= 40, "variation collapsed")
    _require(rotations == set(range(4)) and lane_counts == {2, 3, 4} and len(spans) >= 7, "geometry range collapsed")
    _require(selected_counts == set(range(10)) and len(shifts) >= 5, "load/motion range collapsed")
    _require(clipped >= 42 and reversals >= 42 and stationary >= 10 and shelves >= 30 and opened >= 84, "interaction coverage collapsed")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(6021 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (0, 193):
        a, b = generate_puzzle(seed, 2, 1), generate_puzzle(seed, 2, 1)
        _require(all(np.array_equal(x[k], y[k]) for split in a for x, y in zip(a[split], b[split]) for k in x), "nondeterminism")
    # Rotating the machine rotates its mechanics, while prefix order stays on
    # the displayed page. Recoloring changes no geometric decision.
    pair = next(p for p in generate_puzzle(31)["train"] if _features(p["input"])["selected"] in (2, 4))
    count = _features(pair["input"])["selected"]
    _, original_f = _read(pair["input"])
    for rotation in range(4):
        inp = np.rot90(pair["input"], rotation).copy()
        expected = np.rot90(pair["output"], rotation).copy()
        cr, cc, _, _ = _collector(inp, original_f["bg"])
        expected[cr:cr + 3, cc:cc + 3] = inp[cr:cr + 3, cc:cc + 3]
        for i in range(count):
            expected[cr + i // 3, cc + i % 3] = original_f["payload"]
        for offset in range(3):
            palette = np.array([(3 * i + offset) % 10 for i in range(10)], dtype=np.int8)
            _require(np.array_equal(solve(palette[inp]), palette[expected]), "rotation/recoloring regression")
    malformed = pair["input"].copy()
    canonical, f = _read(malformed)
    canonical[f["top"] + 1, f["lanes"][0]] = f["bg"]
    malformed = np.rot90(canonical, -f["rotation"])
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("gapped shutter accepted")
    fallback_rng = np.random.default_rng(716)
    for split in ("train", "test"):
        for load in range(10, 25):
            a, f = _layout(fallback_rng, dict(shift="profile", colors=(0, 1, 2, 3, 4), vary_colors=False), split, True)
            spec = dict(load=load, unreachable=True, opens=True, detour=True)
            _require(_place_supply(fallback_rng, a, f, _constructed_mechanics(a, f), spec) is not None, "fallback frontier failure")
    # An ambiguous tenth cell must be rejected rather than broken by queue order.
    a, f = _layout(np.random.default_rng(7), dict(shift="depth", colors=(0, 1, 2, 3, 4), vary_colors=False), "train", True)
    distances = np.full(a.shape, -1, dtype=np.int16)
    empty = np.flatnonzero(a == f["bg"])[:10]
    a.flat[empty] = f["payload"]
    distances.flat[empty] = 1
    try:
        _selection(a, f, distances)
    except ValueError:
        pass
    else:
        raise AssertionError("capacity tie accepted")
    broken = generate_puzzle(31, 2, 1)
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
