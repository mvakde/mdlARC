"""Global parity chooses a hollow's 2x2 corner; its cell count exits the tip.

Only NumPy and the standard library are used. Generation paints labels from
sampled geometry; solve recovers geometry by finding enclosed rectangular holes.
Nothing is read from disk, and tests retain only compact aggregate measurements.
"""
from collections import Counter
import hashlib
import numbers

import numpy as np


DIRECTIONS = ("U", "R", "D", "L")
STEP = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid colors must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _rectangle(t, l, b, r):
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _ring(t, l, b, r):
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t, b + 1)} |
            {(y, r) for y in range(t, b + 1)})


def _window(rect, left):
    t, l, b, r = rect
    start = l if left else r - 1
    return {(t + y, start + x) for y in range(2) for x in range(2)}


def _ray(side, tip, direction, count):
    dr, dc = STEP[direction]
    return tuple((tip[0] + i * dr, tip[1] + i * dc)
                 for i in range(1, count + 1)
                 if 0 <= tip[0] + i * dr < side and 0 <= tip[1] + i * dc < side)


def _room(side, tip, direction):
    return {"U": tip[0], "R": side - 1 - tip[1],
            "D": side - 1 - tip[0], "L": tip[1]}[direction]


def _holes(component):
    """Bounded four-neighbor flood around an eight-connected frame component."""
    ys, xs = zip(*component)
    t, b, l, r = min(ys) - 1, max(ys) + 1, min(xs) - 1, max(xs) + 1
    free = _rectangle(t, l, b, r) - component
    return [part for part in _components(free) if (t, l) not in part]


def _read_frame(component):
    holes = _holes(component)
    if len(holes) != 1:
        return None
    hole = holes[0]
    ys, xs = zip(*hole)
    t, b, l, r = min(ys), max(ys), min(xs), max(xs)
    if b - t < 1 or r - l < 3 or hole != _rectangle(t, l, b, r):
        return None
    ring = _ring(t - 1, l - 1, b + 1, r + 1)
    if not ring <= component:
        return None
    tab = component - ring
    if not tab:
        return None
    # A tab has consecutive outward layers, narrowing by two to one tip.
    # Recognition uses layer cross-sections, not generation's shape records.
    candidates = []
    for direction in DIRECTIONS:
        layers = {}
        for y, x in tab:
            depth, cross = {"U": (t - 1 - y, x), "R": (x - r - 1, y),
                            "D": (y - b - 1, x), "L": (l - 1 - x, y)}[direction]
            layers.setdefault(depth, []).append(cross)
        depth = max(layers)
        if depth < 1 or set(layers) != set(range(1, depth + 1)) or len(layers[depth]) != 1:
            continue
        center = layers[depth][0]
        if any(sorted(v) != list(range(center - depth + k, center + depth - k + 1))
               for k, v in layers.items()):
            continue
        lo, hi = (l - 1, r + 1) if direction in ("U", "D") else (t - 1, b + 1)
        if not lo < center < hi or center - depth + 1 < lo or center + depth - 1 > hi:
            continue
        tip = {"U": (t - 1 - depth, center), "R": (center, r + 1 + depth),
               "D": (b + 1 + depth, center), "L": (center, l - 1 - depth)}[direction]
        candidates.append((direction, tip))
    if len(candidates) != 1:
        return None
    direction, tip = candidates[0]
    return dict(frame=component, cavity=hole, rect=(t, l, b, r),
                direction=direction, tip=tip)


def _read(input_grid):
    a = _grid(input_grid)
    side = a.shape[0]
    _require(a.shape == (side, side), "canvas must be square")
    colors, counts = np.unique(a, return_counts=True)
    peak = int(np.argmax(counts))
    _require(counts[peak] > a.size // 2, "background must be a strict majority")
    bg = int(colors[peak])
    found = []
    for color in colors:
        if color == bg:
            continue
        for component in _components(map(tuple, np.argwhere(a == color)), diagonal=True):
            if len(component) < 17:
                continue
            obj = _read_frame(component)
            if obj is None:
                continue
            values = {int(a[p]) for p in obj["cavity"]} - {bg}
            if len(values) != 1 or int(color) in values:
                continue
            payload_color = next(iter(values))
            payload = {p for p in obj["cavity"] if a[p] == payload_color}
            selected = payload & _window(obj["rect"], bool(side % 2))
            obj.update(frame_color=int(color), payload_color=payload_color,
                       payload=payload, selected=selected, quantity=len(selected),
                       ray=_ray(side, obj["tip"], obj["direction"], len(selected)),
                       room=_room(side, obj["tip"], obj["direction"]))
            found.append(obj)
    # A hollow may contain a coincidentally frame-shaped payload. It is not
    # another object: the maximal enclosing frame owns every interior marker.
    found = [o for o in found if not any(o is not q and o["frame"] <= q["cavity"] for q in found)]
    found.sort(key=lambda o: min(o["frame"]))
    _require(bool(found), "no pointed rectangular frame")
    occupied, rays = set(), set()
    for obj in found:
        region = obj["frame"] | obj["cavity"]
        _require(occupied.isdisjoint(region), "overlapping frames or hollows")
        _require(rays.isdisjoint(obj["ray"]), "overlapping output rays")
        _require(all(a[p] == bg for p in obj["ray"]), "ray corridor must be empty")
        occupied.update(region)
        rays.update(obj["ray"])
    _require(occupied.isdisjoint(rays), "ray enters a frame or hollow")
    # All remaining marks must be isolated exterior distractors.
    outside = {tuple(p) for p in np.argwhere(a != bg)} - occupied
    _require(all(len(part) == 1 for part in _components(outside, diagonal=True)),
             "unowned foreground is not an isolated distractor")
    return bg, found


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; corners stay in canvas coordinates."""
    a = _grid(input_grid)
    bg, objects = _read(a)
    out = a.copy()
    for obj in objects:
        for p in obj["selected"]:
            out[p] = bg
        for p in obj["ray"]:
            out[p] = obj["payload_color"]
    return out


def _shape(rng, direction, small=False):
    h, w = (3, 4) if small else (int(rng.integers(3, 7)), int(rng.integers(4, 8)))
    depth = 1 if small else int(rng.integers(1, 4))
    t, l, b, r = 0, 0, h + 1, w + 1
    limit = r if direction in ("U", "D") else b
    margin = max(1, depth - 1)
    center = int(rng.integers(margin, limit - margin + 1))
    frame = _ring(t, l, b, r)
    for k in range(1, depth + 1):
        for cross in range(center - depth + k, center + depth - k + 1):
            frame.add({"U": (-k, cross), "R": (cross, r + k),
                       "D": (b + k, cross), "L": (cross, -k)}[direction])
    tip = {"U": (-depth, center), "R": (center, r + depth),
           "D": (b + depth, center), "L": (center, -depth)}[direction]
    ys, xs = zip(*frame)
    dy, dx = -min(ys), -min(xs)
    return dict(frame={(y + dy, x + dx) for y, x in frame},
                rect=(1 + dy, 1 + dx, h + dy, w + dx),
                tip=(tip[0] + dy, tip[1] + dx), direction=direction,
                height=max(ys) + dy + 1, width=max(xs) + dx + 1)


def _spec(rng, direction, colors, left_q=None, right_q=None, mode="any", small=False):
    obj = _shape(rng, direction, small)
    left = sorted(_window(obj["rect"], True))
    right = sorted(_window(obj["rect"], False))
    left_q = int(rng.integers(0, 5)) if left_q is None else left_q
    right_q = int(rng.integers(0, 5)) if right_q is None else right_q
    payload = {left[int(i)] for i in rng.choice(4, left_q, replace=False)}
    payload |= {right[int(i)] for i in rng.choice(4, right_q, replace=False)}
    t, l, b, r = obj["rect"]
    other = sorted(_rectangle(t, l, b, r) - set(left) - set(right))
    # At least one marker outside BOTH candidate windows makes count-all
    # and erase-all wrong, even if the selected window happens to be empty.
    amount = int(rng.integers(1, len(other) + 1))
    payload |= {other[int(i)] for i in rng.choice(len(other), amount, replace=False)}
    obj.update(payload=payload, frame_color=colors[0], payload_color=colors[1], mode=mode)
    return obj


def _quantity(obj, side):
    return len(obj["payload"] & _window(obj["rect"], bool(side % 2)))


def _pack(rng, side, specs):
    occupied, guards, ray_guards, placed = set(), set(), set(), []
    for obj in specs:
        q = obj.get("reserve_quantity", _quantity(obj, side))
        positions = [(r, c) for r in range(side - obj["height"] + 1)
                     for c in range(side - obj["width"] + 1)]
        accepted = False
        for index in rng.permutation(len(positions)):
            r, c = positions[int(index)]
            tip = r + obj["tip"][0], c + obj["tip"][1]
            room, mode = _room(side, tip, obj["direction"]), obj["mode"]
            if (mode == "free" and room <= q or mode == "exact" and room != q
                    or mode == "clip" and not 0 < room < q or mode == "zero" and room != 0):
                continue
            box = _rectangle(r, c, r + obj["height"] - 1, c + obj["width"] - 1)
            path = set(_ray(side, tip, obj["direction"], q + 1))
            if not box.isdisjoint(guards | ray_guards) or not path.isdisjoint(occupied | ray_guards):
                continue
            placed.append((obj, r, c))
            occupied.update(box)
            guards.update((y + dy, x + dx) for y, x in box for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            ray_guards.update(path)
            accepted = True
            break
        if not accepted:
            return None
    return placed


def _render(side, bg, placed, marks=()):
    """Labels painted from construction records, without calling solve/read."""
    inp = np.full((side, side), bg, dtype=np.int8)
    out = inp.copy()
    for obj, r, c in placed:
        for y, x in obj["frame"]:
            inp[r + y, c + x] = out[r + y, c + x] = obj["frame_color"]
        selected = _window(obj["rect"], bool(side % 2))
        for y, x in obj["payload"]:
            inp[r + y, c + x] = obj["payload_color"]
            if (y, x) not in selected:
                out[r + y, c + x] = obj["payload_color"]
        tip = r + obj["tip"][0], c + obj["tip"][1]
        for p in _ray(side, tip, obj["direction"], _quantity(obj, side)):
            out[p] = obj["payload_color"]
    for p, color in marks:
        inp[p] = out[p] = color
    _require(np.array_equal(out, solve(inp)), "constructed label and recovered geometry disagree")
    return {"input": inp, "output": out}


def _scene(rng, side, directions, count, anchor=False):
    bg = int(rng.integers(10))
    palette = [c for c in range(10) if c != bg]
    first_colors = tuple(int(c) for c in rng.choice(palette, 2, replace=False))
    color_mode = int(rng.integers(4))
    low, high = int(rng.integers(1, 3)), int(rng.integers(3, 5))
    # Finite geometry/layout retries, then a guaranteed small four-tile layout.
    for attempt in range(20):
        fallback = attempt == 19
        if attempt in (6, 12) or fallback:
            side = min(28 if anchor else 30 - side % 2, side + 4)
        if fallback:
            side = 28 if anchor else 30 - side % 2
        specs = []
        for i in range(count):
            colors = tuple(int(c) for c in rng.choice(palette, 2, replace=False))
            if color_mode == 0:
                colors = first_colors
            elif color_mode == 1:
                colors = (first_colors[0], int(rng.choice([c for c in palette if c != first_colors[0]])))
            elif color_mode == 2:
                colors = first_colors if i % 2 == 0 else first_colors[::-1]
            if anchor:
                lq, rq = (low, high) if i == 0 else (high, low)
                mode = "free"
            else:
                lq, rq = int(rng.integers(5)), int(rng.integers(5))
                if i == 0:
                    # Every test/extra scene contains a count-visible ray.
                    if side % 2:
                        lq = int(rng.integers(2, 5))
                    else:
                        rq = int(rng.integers(2, 5))
                    mode = "free"
                else:
                    q = lq if side % 2 else rq
                    mode = str(rng.choice(["any", "any", "free", "exact", "clip", "zero"]))
                    if q < 2 and mode == "clip":
                        mode = "any"
            if fallback:
                mode = "free"
            spec = _spec(rng, directions[i % len(directions)], colors, lq, rq, mode, fallback)
            if anchor:
                spec["reserve_quantity"] = max(lq, rq)
            specs.append(spec)
        if sum(len(o["frame"]) + len(o["payload"]) for o in specs) * 2 >= side * side:
            continue
        placed = _pack(rng, side, specs) if not fallback else None
        if fallback:
            placed = []
            tile = side // 2
            for i, obj in enumerate(specs):
                tr, tc = (i // 2) * tile, (i % 2) * tile
                r, c = {"U": (tr + 5, tc + 2), "R": (tr + 2, tc + 1),
                        "D": (tr + 1, tc + 2), "L": (tr + 2, tc + 5)}[obj["direction"]]
                placed.append((obj, r, c))
        if placed is not None:
            break
    marks = []
    if not anchor and rng.random() < .4:
        blocked = set()
        for obj, r, c in placed:
            blocked.update(_rectangle(r - 1, c - 1, r + obj["height"], c + obj["width"]))
            tip = r + obj["tip"][0], c + obj["tip"][1]
            blocked.update(_ray(side, tip, obj["direction"], side))
        choices = [(r, c) for r in range(side) for c in range(side) if (r, c) not in blocked]
        if choices:
            marks = [(choices[int(rng.integers(len(choices)))], int(rng.choice(palette)))]
    return side, bg, placed, marks


def _legal_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))
                 and lo <= value <= hi, f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    _require(isinstance(seed, numbers.Integral) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Two adjacent directions establish that crop coordinates do not rotate.
    start = int(rng.integers(4))
    train_dirs = [DIRECTIONS[start], DIRECTIONS[(start + 1) % 4]]
    test_dirs = [DIRECTIONS[(start + 2) % 4], DIRECTIONS[(start + 3) % 4]]
    base = int(rng.choice([18, 20, 22, 24]))
    side, bg, placed, marks = _scene(rng, base, train_dirs, 2, anchor=True)
    train = [_render(side, bg, placed), _render(side + 1, bg, placed)]
    rng.shuffle(train)
    puzzle = {"train": train, "test": []}
    seen = {_key(p["input"]) for p in train}
    for split, amount in (("train", num_train - 2), ("test", num_test)):
        for index in range(amount):
            for attempt in range(12):
                if split == "train":
                    parity = index % 2
                    choices = [n for n in range(16, 29) if n % 2 == parity and n not in (side, side + 1)]
                    n = int(rng.choice(choices))
                    dirs = [str(d) for d in rng.permutation(train_dirs)]
                    count = int(rng.integers(1, 4))
                else:
                    n = int(rng.integers(16, 31))
                    dirs = [str(d) for d in rng.permutation(test_dirs)]
                    # The first test combines the two new rotations; later
                    # tests may also be simple single-object applications.
                    count = int(rng.integers(2, 5) if index == 0 else rng.integers(1, 5))
                n, local_bg, objects, exterior = _scene(rng, n, dirs, count)
                pair = _render(n, local_bg, objects, exterior)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input retry budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual(a, objects, bg, rule):
    out = a.copy()
    for obj in objects:
        if rule == "all":
            selected = obj["payload"]
        elif rule == "no_ray":
            selected = obj["selected"]
        elif rule == "rotate_corner":
            # The competing rule rotates B's corners with A's direction.
            odd = bool(a.shape[0] % 2)
            corner = {"U": ("TL" if odd else "TR"), "R": ("TR" if odd else "BR"),
                      "D": ("BR" if odd else "BL"), "L": ("BL" if odd else "TL")}[obj["direction"]]
            t, l, b, r = obj["rect"]
            y, x = (t if corner[0] == "T" else b - 1), (l if corner[1] == "L" else r - 1)
            selected = obj["payload"] & _rectangle(y, x, y + 1, x + 1)
        else:
            selected = obj["payload"] & _window(obj["rect"], rule == "left")
        for p in selected:
            out[p] = bg
        if rule != "no_ray":
            for p in _ray(a.shape[0], obj["tip"], obj["direction"], len(selected)):
                out[p] = obj["payload_color"]
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _legal_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnesses, full_counts, failures = set(), set(), set(), set()
    scene_keys, connected_witness = {}, False
    train_dirs, test_dirs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "output must preserve dimensions")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            bg, objects = _read(a)
            _require(1 <= len(objects) <= 4, "object count outside generated family")
            _require(np.array_equal(out, solve(a)), "incorrect generated output")
            for obj in objects:
                _require(0 <= obj["quantity"] <= 4, "selected count outside 0..4")
                _require(all(out[p] == obj["frame_color"] for p in obj["frame"]), "frame changed")
                _require(all(out[p] == obj["payload_color"] for p in obj["payload"] - obj["selected"]),
                         "unselected payload changed")
                _require(all(out[p] == bg for p in obj["selected"]), "selected payload not erased")
                _require(len(obj["ray"]) == min(obj["room"], obj["quantity"]), "clipping incorrect")
                (train_dirs if split == "train" else test_dirs).add(obj["direction"])
                if split == "train" and obj["quantity"] > 0 and obj["room"] > obj["quantity"]:
                    full_counts.add(obj["quantity"])
                    connected_witness |= len(_components(obj["selected"])) < obj["quantity"]
                    left = obj["payload"] & _window(obj["rect"], True)
                    right = obj["payload"] & _window(obj["rect"], False)
                    if len(left) != len(right):
                        witnesses.add(a.shape[0] % 2)
            if split == "train":
                for rule in ("left", "right", "all", "no_ray", "rotate_corner"):
                    if not np.array_equal(out, _counterfactual(a, objects, bg, rule)):
                        failures.add(rule)
                # Same foreground scene under different canvas parity is an
                # explicit counterexample to frame-local or content-based parity.
                foreground = tuple((int(r), int(c), int(a[r, c])) for r, c in np.argwhere(a != bg))
                scene_keys.setdefault((bg, foreground), set()).add(a.shape[0] % 2)
    _require(witnesses == {0, 1}, "training must expose both parity choices with unequal counts")
    _require(len(full_counts) >= 2, "training permits a fixed-length ray shortcut")
    _require(failures == {"left", "right", "all", "no_ray", "rotate_corner"}, "a simpler shortcut fits training")
    _require(connected_witness, "training fails to distinguish cells from connected components")
    _require(any(parities == {0, 1} for parities in scene_keys.values()), "missing global-parity matched scene")
    _require(bool(test_dirs - train_dirs), "missing visible new-rotation test shift")
    _require(any(o["quantity"] > 0 and o["room"] > o["quantity"]
                 for p in puzzle["test"] for o in _read(p["input"])[1]), "tests conceal counting behind clipping")


def self_test():
    """Stream 42 seeds; retain only hashes, counts and small measurement sets."""
    hashes, shapes, backgrounds, geometries, counts, directions = set(), set(), set(), set(), set(), set()
    quantities, clearances, palettes, normalized = set(), Counter(), set(), set()
    combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                fingerprint.update(bytes(a.shape))
                fingerprint.update(a.tobytes())
                fingerprint.update(pair["output"].tobytes())
                bg, objects = _read(a)
                shapes.add(a.shape[0])
                backgrounds.add(bg)
                counts.add(len(objects))
                palettes.add(tuple(int(c) for c in np.unique(a)))
                roles = {}
                canonical = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                normalized.add(hashlib.sha256(bytes(a.shape) + canonical).hexdigest())
                combined += len(objects) >= 3 and len({o["direction"] for o in objects}) > 1
                for o in objects:
                    t, l, b, r = o["rect"]
                    geometries.add((b - t + 1, r - l + 1, len(o["frame"]) - len(_ring(t-1, l-1, b+1, r+1))))
                    directions.add(o["direction"])
                    quantities.add(o["quantity"])
                    q, room = o["quantity"], o["room"]
                    clearances["zero_count" if q == 0 else "zero_room" if room == 0 else
                               "clipped" if room < q else "exact" if room == q else "free"] += 1
        hashes.add(fingerprint.hexdigest())
    _require(len(hashes) == 42, "cross-seed duplication")
    _require(len(shapes) >= 10 and max(shapes) == 30, "insufficient canvas scale variation")
    _require(len(backgrounds) == 10 and len(palettes) >= 30, "insufficient palette variation")
    _require(len(geometries) >= 20 and counts == {1, 2, 3, 4}, "insufficient structural variation")
    _require(directions == set(DIRECTIONS) and quantities == set(range(5)), "direction/count variation missing")
    _require(all(clearances[m] > 0 for m in ("zero_count", "zero_room", "clipped", "exact", "free")),
             "boundary coverage missing")
    _require(len(normalized) >= 170 and combined >= 10, "compatible structural variations do not combine")
    # Exhaust all legal explicit counts with one deterministic seed per request.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            generate_puzzle(1000 + n * 10 + m, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for n, m in ((2, 1), (4, 1), (10, 4)):
        first, second = generate_puzzle(643711068, n, m), generate_puzzle(643711068, n, m)
        _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s])
                     for k in ("input", "output")), "seed is not deterministic")
    # Input-only metamorphic checks: recoloring preserves roles. One extra
    # bottom/right background border flips selection while leaving frames fixed.
    example = generate_puzzle(42, 2, 1)
    a = example["train"][0]["input"]
    _require(np.array_equal(solve((a + 3) % 10), (solve(a) + 3) % 10), "color-role regression")
    bg, objects = _read(a)
    bigger = np.full((a.shape[0] + 1, a.shape[1] + 1), bg, dtype=np.int8)
    bigger[:-1, :-1] = a
    target = bigger.copy()
    for o in objects:
        selected = o["payload"] & _window(o["rect"], not bool(a.shape[0] % 2))
        for p in selected:
            target[p] = bg
        for p in _ray(bigger.shape[0], o["tip"], o["direction"], len(selected)):
            target[p] = o["payload_color"]
    _require(np.array_equal(solve(bigger), target), "global canvas parity regression")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "sides": sorted(shapes),
            "backgrounds": len(backgrounds), "palettes": len(palettes),
            "geometries": len(geometries), "object_counts": sorted(counts),
            "quantities": sorted(quantities), "clearances": dict(clearances),
            "normalized_inputs": len(normalized), "mixed_multi_object_pairs": combined}
