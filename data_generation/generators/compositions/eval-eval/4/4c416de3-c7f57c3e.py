"""Clockwise relay of two incomplete corner stamps.

Frames supply anchors, orientations, scale and a cyclic connection. The input
supplies both complete stamp shapes. A receiver keeps its tint and takes the
preceding marked corner's *complete shape*, skipping unmarked corners.
"""

from collections import Counter
import json
import numpy as np


# Clockwise: top-left, top-right, bottom-right, bottom-left. Axis reflections,
# not quarter-turn rotations, map inward row/column offsets between corners.
_DIRS = ((1, 1), (1, -1), (-1, -1), (-1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _connected(points):
    seen, todo = {min(points)}, [min(points)]
    while todo:
        y, x = todo.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = y + dy, x + dx
                if p in points and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return seen == set(points)


def _separate(a, b):
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    return (ay + ah < by or by + bh < ay
            or ax + aw < bx or bx + bw < ax)


def _ring_frames(grid, color):
    frames = []
    for group in _components(grid == color):
        y0, y1 = min(y for y, x in group), max(y for y, x in group)
        x0, x1 = min(x for y, x in group), max(x for y, x in group)
        h, w = y1 - y0 + 1, x1 - x0 + 1
        mask = grid[y0:y1 + 1, x0:x1 + 1] == color
        s = 0
        while s < h and np.all(mask[s]):
            s += 1
        assert s and h % s == 0 and w % s == 0
        assert h // s >= 7 and w // s >= 7, "not a recognizable rectangular ring"
        ring = np.ones((h, w), dtype=bool)
        ring[s:-s, s:-s] = False
        assert np.array_equal(mask, ring), "frame is broken or not uniformly thick"
        assert len(group) == int(ring.sum())
        rect = (y0, x0, h, w)
        assert all(_separate(rect, f["rect"]) for f in frames)
        frames.append({"rect": rect, "scale": s, "logical": (h // s, w // s)})
    assert frames
    return sorted(frames, key=lambda f: f["rect"])


def _transpose(pattern):
    return frozenset((v, u) for u, v in pattern)


def _corner_indices(frame):
    h, w = frame["logical"]
    return ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0))


def _decode_roles(grid, bg, carrier, frames, rotation=False):
    occupied = np.zeros(grid.shape, dtype=bool)
    occurrences = []
    for fid, frame in enumerate(frames):
        y, x, h, w = frame["rect"]
        s = frame["scale"]
        crop = grid[y:y + h, x:x + w]
        small = crop[::s, ::s]
        assert np.array_equal(crop, np.repeat(np.repeat(small, s, 0), s, 1)), "uneven pixel blocks"
        occupied[y:y + h, x:x + w] = True
        corners = _corner_indices(frame)
        local = [[] for _ in range(4)]
        for iy, ix in np.argwhere((small != bg) & (small != carrier)):
            choices = []
            for q, ((cy, cx), (dy, dx)) in enumerate(zip(corners, _DIRS)):
                u, v = (int(iy) - cy) * dy, (int(ix) - cx) * dx
                if 1 <= u <= 3 and 1 <= v <= 3:
                    choices.append((q, u, v))
            assert len(choices) == 1, "stamp cell has no unique corner"
            q, u, v = choices[0]
            local[q].append((u, v, int(small[iy, ix])))
        for q, cells in enumerate(local):
            if not cells:
                continue
            colors = {c for u, v, c in cells}
            assert len(colors) == 1, "a corner stamp must have one local tint"
            pattern = frozenset((u, v) for u, v, c in cells)
            if rotation and q % 2:
                pattern = _transpose(pattern)
            occurrences.append({"frame": fid, "corner": q, "color": colors.pop(), "clue": pattern})
    assert np.all(grid[~occupied] == bg), "foreground outside frames"
    assert len(occurrences) >= 2
    observed = {o["clue"] for o in occurrences}
    maximal = sorted((p for p in observed if not any(p < other for other in observed)),
                     key=lambda p: tuple(sorted(p)))
    assert len(maximal) == 2, "need two incomparable intact templates"
    assert all(2 <= len(p) <= 8 and _connected(p) for p in maximal)
    for o in occurrences:
        matches = [i for i, p in enumerate(maximal) if o["clue"] <= p]
        assert len(matches) == 1, "clue does not identify a unique state"
        o["state"] = matches[0]
    # The actual in-bounds templates must have unique associations too.
    for fid, frame in enumerate(frames):
        corners = _corner_indices(frame)
        for o in (o for o in occurrences if o["frame"] == fid):
            cy, cx = corners[o["corner"]]
            dy, dx = _DIRS[o["corner"]]
            for pattern in maximal:
                if rotation and o["corner"] % 2:
                    pattern = _transpose(pattern)
                for u, v in pattern:
                    iy, ix = cy + dy * u, cx + dx * v
                    assert 0 < iy < frame["logical"][0] - 1
                    assert 0 < ix < frame["logical"][1] - 1
                    owners = [q for q, ((a, b), (sy, sx)) in enumerate(zip(corners, _DIRS))
                              if 1 <= (iy - a) * sy <= 3 and 1 <= (ix - b) * sx <= 3]
                    assert owners == [o["corner"]], "replacement has ambiguous corner ownership"
    return {"grid": grid, "background": bg, "carrier": carrier, "frames": frames,
            "templates": maximal, "occurrences": occurrences, "rotation": rotation}


def _read_input(value, rotation=False):
    grid = _grid(value)
    colors = list(map(int, np.unique(grid)))
    solutions = []
    for carrier in colors:
        try:
            frames = _ring_frames(grid, carrier)
        except AssertionError:
            continue
        for bg in colors:
            if bg == carrier:
                continue
            try:
                solutions.append(_decode_roles(grid, bg, carrier, frames, rotation))
            except AssertionError:
                continue
    if len(solutions) != 1:
        raise ValueError("input has missing or ambiguous frame/stamp interpretation")
    return solutions[0]


def _paint(grid, frame, q, pattern, color):
    t, l, h, w = frame["rect"]
    s = frame["scale"]
    cy, cx = _corner_indices(frame)[q]
    dy, dx = _DIRS[q]
    for u, v in pattern:
        y, x = t + (cy + dy * u) * s, l + (cx + dx * v) * s
        grid[y:y + s, x:x + s] = color


def _predict(scene, mode="relay"):
    out = scene["grid"].copy()
    templates = scene["templates"]
    for fid, frame in enumerate(scene["frames"]):
        active = sorted((o for o in scene["occurrences"] if o["frame"] == fid),
                        key=lambda o: o["corner"])
        for o in active:
            pattern = o["clue"]
            if scene["rotation"] and o["corner"] % 2:
                pattern = _transpose(pattern)
            _paint(out, frame, o["corner"], pattern, scene["background"])
        for i, o in enumerate(active):
            source = active[(i + 1 if mode == "reverse" else i - 1) % len(active)]
            if mode == "own":
                pattern = templates[o["state"]]
            elif mode == "toggle":
                pattern = templates[1 - o["state"]]
            elif mode == "fragments":
                pattern = source["clue"]
            else:
                pattern = templates[source["state"]]
            if scene["rotation"] and o["corner"] % 2:
                pattern = _transpose(pattern)
            _paint(out, frame, o["corner"], pattern, o["color"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, scale, states and clockwise routing from this input only."""
    return _predict(_read_input(input_grid))


def _sample_stamp(rng, radius):
    domain = {(u, v) for u in range(1, radius + 1) for v in range(1, radius + 1)}
    size = int(rng.integers(2 if radius == 2 else 3, 4 if radius == 2 else 9))
    choices = sorted(domain)
    points = {choices[int(rng.integers(len(choices)))]}
    while len(points) < size:
        frontier = sorted(p for p in domain - points
                          if any(max(abs(p[0] - u), abs(p[1] - v)) <= 1 for u, v in points))
        points.add(frontier[int(rng.integers(len(frontier)))])
    return frozenset(points)


def _sample_templates(rng, radius, witness):
    for _ in range(200):
        a, b = _sample_stamp(rng, radius), _sample_stamp(rng, radius)
        if a <= b or b <= a or (witness and a == _transpose(a)):
            continue
        if radius == 3 and (max(max(u, v) for u, v in a | b) < 3 or max(len(a), len(b)) < 5):
            continue
        return (a, b)
    raise RuntimeError("bounded stamp sampling exhausted")


def _shelf(sizes, width, order):
    y, x, row_h = 0, 0, 0
    placed = [None] * len(sizes)
    for i in order:
        h, w = sizes[i]
        if w > width:
            return None
        if x and x + w > width:
            y, x, row_h = y + row_h + 1, 0, 0
        if y + h > 30:
            return None
        placed[i] = (y, x, h, w)
        x += w + 1
        row_h = max(row_h, h)
    return placed


def _arrange_frames(rng, scales, radius):
    minimum = 2 * radius + 3
    for attempt in range(40):
        logical = [(minimum + (int(rng.integers(4)) if attempt < 39 else 0),
                    minimum + (int(rng.integers(4)) if attempt < 39 else 0)) for s in scales]
        sizes = [(h * s, w * s) for (h, w), s in zip(logical, scales)]
        options = []
        orders = [sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True),
                  list(map(int, rng.permutation(len(sizes))))]
        for width in range(max(w for h, w in sizes), 31):
            for order in orders:
                placed = _shelf(sizes, width, order)
                if placed:
                    options.append(placed)
        if options:
            rects = options[int(rng.integers(len(options)))]
            break
    else:
        raise RuntimeError("frame packing exhausted")
    h0 = max(y + h for y, x, h, w in rects)
    w0 = max(x + w for y, x, h, w in rects)
    height = int(rng.integers(h0, min(30, h0 + 7) + 1))
    width = int(rng.integers(w0, min(30, w0 + 7) + 1))
    oy, ox = int(rng.integers(height - h0 + 1)), int(rng.integers(width - w0 + 1))
    rects = [(y + oy, x + ox, h, w) for y, x, h, w in rects]
    for _ in range(35):
        i = int(rng.integers(len(rects)))
        _, _, h, w = rects[i]
        new = (int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1)), h, w)
        if all(_separate(new, rect) for j, rect in enumerate(rects) if j != i):
            rects[i] = new
    frames = [{"rect": rect, "scale": s, "logical": hw}
              for rect, s, hw in zip(rects, scales, logical)]
    return (height, width), frames


def _plan(rng, split, index, shift):
    if split == "train" and index == 0:
        return 2, [1], True, False
    if split == "train" and index == 1:
        return 2, [2, 1] if rng.random() < .55 else [2], False, True
    if split == "test" and shift == "scale":
        return 2, [3, 1] if rng.random() < .35 else [3], False, False
    if split == "test" and shift == "count":
        return 2, [1] * int(rng.integers(3, 6)), False, False
    if split == "test" and shift == "footprint":
        scales = [int(rng.choice([1, 1, 2]))]
        if rng.random() < .5:
            scales.append(1)
        return 3, scales, False, False
    radius = 2 if shift == "footprint" else int(rng.choice([2, 2, 3]))
    count = int(rng.integers(1, (3 if shift == "count" or radius == 3 else 4)))
    scales = [1] * count
    if count <= 2 and rng.random() < .4:
        scales[0] = 2
    return radius, scales, False, False


def _make_pair(rng, radius, scales, witness, four, stable_roles, crowded):
    templates = _sample_templates(rng, radius, witness)
    shape, frames = _arrange_frames(rng, scales, radius)
    if stable_roles is None:
        bg, carrier = map(int, rng.choice(10, 2, replace=False))
    else:
        bg, carrier = stable_roles
    palette = [c for c in range(10) if c not in (bg, carrier)]
    rng.shuffle(palette)
    palette = palette[:1 if witness else int(rng.integers(1, 5))]
    records = []
    for fid, frame in enumerate(frames):
        if witness:
            start = int(rng.integers(4))
            corners, states = [(start + i) % 4 for i in range(3)], [0, 0, 1]
        else:
            lo = (3 if len(frames) == 1 else 2) if fid == 0 else (2 if crowded else 0)
            count = 4 if fid == 0 and four else int(rng.integers(lo, 5))
            corners = sorted(map(int, rng.permutation(4)[:count]))
            states = list(map(int, rng.integers(0, 2, count)))
            if fid == 0:
                states[0], states[1] = 0, 1
        for q, state in zip(corners, states):
            records.append({"frame": fid, "corner": q, "state": state,
                            "color": int(rng.choice(palette))})
    if len(records) < 3:
        # First frame has two states; add a third active corner to expose loss.
        available = sorted(set(range(4)) - {o["corner"] for o in records if o["frame"] == 0})
        records.append({"frame": 0, "corner": int(rng.choice(available)),
                        "state": int(rng.integers(2)), "color": int(rng.choice(palette))})
    if witness:
        refs = {0, 2}
    else:
        refs = {int(rng.choice([i for i, o in enumerate(records) if o["state"] == s])) for s in (0, 1)}
    candidates = [i for i in range(len(records)) if i not in refs]
    forced_partial = int(rng.choice(candidates))
    base = np.full(shape, bg, dtype=np.int8)
    for frame in frames:
        t, l, h, w = frame["rect"]
        s = frame["scale"]
        base[t:t + s, l:l + w] = carrier
        base[t + h - s:t + h, l:l + w] = carrier
        base[t:t + h, l:l + s] = carrier
        base[t:t + h, l + w - s:l + w] = carrier
    inp, out = base.copy(), base.copy()
    for i, o in enumerate(records):
        pattern = templates[o["state"]]
        if i in refs or (i != forced_partial and not witness and rng.random() < .25):
            clue = pattern
        else:
            exclusive = sorted(pattern - templates[1 - o["state"]])
            diagnostic = exclusive[int(rng.integers(len(exclusive)))]
            remaining = sorted(pattern - {diagnostic})
            rng.shuffle(remaining)
            keep = int(rng.integers(1, len(pattern)))
            clue = frozenset([diagnostic] + remaining[:keep - 1])
        o["clue"] = clue
        _paint(inp, frames[o["frame"]], o["corner"], clue, o["color"])
    # Labels are rendered from construction data, before the input decoder is used.
    for fid, frame in enumerate(frames):
        active = sorted((o for o in records if o["frame"] == fid), key=lambda o: o["corner"])
        for i, receiver in enumerate(active):
            source = active[i - 1]
            _paint(out, frame, receiver["corner"], templates[source["state"]], receiver["color"])
    return {"input": inp, "output": out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _vocabulary(scene):
    return tuple(sorted(tuple(sorted(p)) for p in scene["templates"]))


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["scale", "count", "footprint"]))
    roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .25 else None
    puzzle, seen, vocabularies = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            radius, scales, witness, four = _plan(rng, split, index, shift)
            for _ in range(100):
                pair = _make_pair(rng, radius, scales, witness, four, roles,
                                  split == "test" and shift == "count")
                scene = _read_input(pair["input"])
                vocabulary = _vocabulary(scene)
                if _key(pair["input"]) in seen:
                    continue
                if (split == "test" or (split == "train" and index == 1)) and vocabulary in vocabularies:
                    continue
                if witness:
                    try:
                        alternative = _predict(_read_input(pair["input"], rotation=True))
                    except ValueError:
                        alternative = None
                    if alternative is not None and np.array_equal(alternative, pair["output"]):
                        continue
                assert np.array_equal(_predict(scene), pair["output"])
                seen.add(_key(pair["input"]))
                vocabularies.add(vocabulary)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("unique informative pair sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _read_input(inp)
            assert np.array_equal(_predict(scene), out), "incorrect complete clockwise relay"
            assert np.any(inp != out)
            assert np.array_equal(inp == scene["carrier"], out == scene["carrier"]), "frame changed"
            assert set(np.unique(inp)) == set(np.unique(out)), "local tint roles changed"
            os, templates = scene["occurrences"], scene["templates"]
            assert any(o["clue"] != templates[o["state"]] for o in os), "no incomplete occurrence"
            assert not np.array_equal(_predict(scene, "fragments"), out), "completion is dispensable"
            assert not np.array_equal(_predict(scene, "own"), out), "state transfer is dispensable"
            decoded_out = _read_input(out)
            assert _vocabulary(scene) == _vocabulary(decoded_out)
            assert [f["rect"] for f in scene["frames"]] == [f["rect"] for f in decoded_out["frames"]]
            before = {(o["frame"], o["corner"]): o for o in os}
            after = {(o["frame"], o["corner"]): o for o in decoded_out["occurrences"]}
            assert before.keys() == after.keys(), "marked corner set changed"
            assert all(o["color"] == after[k]["color"] for k, o in before.items())
            for fid in range(len(scene["frames"])):
                assert Counter(o["state"] for o in os if o["frame"] == fid) == Counter(
                    o["state"] for o in decoded_out["occurrences"] if o["frame"] == fid), "states crossed frames"
                assert all(o["clue"] == templates[o["state"]] for o in decoded_out["occurrences"] if o["frame"] == fid)
            scenes[split].append(scene)
    first, label = scenes["train"][0], _grid(puzzle["train"][0]["output"])
    assert not np.array_equal(_predict(first, "reverse"), label), "clockwise direction was not taught"
    assert not np.array_equal(_predict(first, "toggle"), label), "unconditional toggling fits the witness"
    assert any(len({o["state"] for o in first["occurrences"] if o["color"] == c}) == 2
               for c in set(o["color"] for o in first["occurrences"])), "color alone identifies state"
    try:
        rotated = _predict(_read_input(first["grid"], rotation=True))
    except ValueError:
        rotated = None
    assert rotated is None or not np.array_equal(rotated, label), "untaught reflection convention"
    assert any(f["scale"] > 1 for s in scenes["train"] for f in s["frames"]), "no scale demonstration"
    assert len({_vocabulary(s) for s in scenes["train"]}) >= 2
    # At least one substantive geometric shift, determined from input pixels.
    train_frames = max(len(s["frames"]) for s in scenes["train"])
    train_scale = max(f["scale"] for s in scenes["train"] for f in s["frames"])
    train_radius = max(max(max(p) for p in t) for s in scenes["train"] for t in s["templates"])
    for s in scenes["test"]:
        assert _vocabulary(s) not in {_vocabulary(t) for t in scenes["train"]}
        assert (len(s["frames"]) > train_frames
                or max(f["scale"] for f in s["frames"]) > train_scale
                or max(max(max(p) for p in t) for t in s["templates"]) > train_radius)


def _signature(puzzle):
    return tuple(_key(p[r]) for s in ("train", "test") for p in puzzle[s] for r in ("input", "output"))


def self_test():
    """Bounded semantic regressions and measurements from 42 actual puzzles."""
    fingerprints, shapes, vocabularies, frame_shapes = set(), set(), set(), set()
    scales, counts, marks, tint_counts, stamp_sizes, footprints = set(), set(), set(), set(), set(), set()
    backgrounds, carriers, partial_sizes, shifts = set(), set(), set(), Counter()
    edges = empty_frames = singleton_frames = unequal_states = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        fingerprints.add(_signature(puzzle))
        train_scenes = [_read_input(p["input"]) for p in puzzle["train"]]
        max_count = max(len(s["frames"]) for s in train_scenes)
        max_scale = max(f["scale"] for s in train_scenes for f in s["frames"])
        max_radius = max(max(max(p) for p in t) for s in train_scenes for t in s["templates"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                untouched = inp.copy()
                prediction = solve(inp)
                assert np.array_equal(inp, untouched), "solve mutated its input"
                assert np.array_equal(prediction, out)
                s = _read_input(inp)
                shapes.add(inp.shape)
                vocabularies.add(_vocabulary(s))
                backgrounds.add(s["background"])
                carriers.add(s["carrier"])
                counts.add(len(s["frames"]))
                tint_counts.add(len({o["color"] for o in s["occurrences"]}))
                unequal_states += len(s["templates"][0]) != len(s["templates"][1])
                edges += any(np.any(a != s["background"]) for a in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
                for template in s["templates"]:
                    stamp_sizes.add(len(template))
                    footprints.add((max(u for u, v in template) - min(u for u, v in template) + 1,
                                    max(v for u, v in template) - min(v for u, v in template) + 1))
                for fid, f in enumerate(s["frames"]):
                    active = [o for o in s["occurrences"] if o["frame"] == fid]
                    marks.add(len(active))
                    empty_frames += not active
                    singleton_frames += len(active) == 1
                    frame_shapes.add(f["logical"])
                    scales.add(f["scale"])
                partial_sizes.update(len(o["clue"]) for o in s["occurrences"] if o["clue"] != s["templates"][o["state"]])
                if split == "test":
                    shifts["count"] += len(s["frames"]) > max_count
                    shifts["scale"] += max(f["scale"] for f in s["frames"]) > max_scale
                    shifts["footprint"] += max(max(max(p) for p in t) for t in s["templates"]) > max_radius
        # A twelve-step relay restores the completed input: 12 is divisible
        # by every possible marked-corner cycle length (1, 2, 3, 4).
        first = _read_input(puzzle["train"][0]["input"])
        result = first["grid"]
        for _ in range(12):
            result = solve(result)
        assert np.array_equal(result, _predict(first, "own"))
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(vocabularies) >= 70 and len(frame_shapes) >= 15
    assert counts >= {1, 2, 3, 4, 5} and scales == {1, 2, 3}
    assert marks == {0, 1, 2, 3, 4} and tint_counts >= {1, 2, 3, 4}
    assert stamp_sizes >= {2, 3, 4, 5, 6, 7, 8} and len(footprints) >= 5
    assert backgrounds == set(range(10)) and carriers == set(range(10))
    assert partial_sizes >= {1, 2, 3, 4} and all(shifts[k] > 0 for k in ("count", "scale", "footprint"))
    assert edges and empty_frames and singleton_frames and unequal_states
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9917, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject an unrelated output change and a clue shared by both templates.
    p = generate_puzzle(91)
    damaged = {s: [{r: a[r].copy() for r in ("input", "output")} for a in p[s]] for s in p}
    g = damaged["test"][0]["output"]
    g[0, 0] = (int(g[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    ambiguous = np.zeros((11, 11), dtype=np.int8)
    frame = {"rect": (1, 1, 9, 9), "scale": 1, "logical": (9, 9)}
    ambiguous[1:10, 1:10] = 1
    ambiguous[2:9, 2:9] = 0
    a, b = frozenset({(1, 1), (1, 2)}), frozenset({(1, 1), (2, 1)})
    _paint(ambiguous, frame, 0, a, 2)
    _paint(ambiguous, frame, 1, b, 3)
    _paint(ambiguous, frame, 2, {(1, 1)}, 4)
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous partial-state clue accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "canvas_shapes": len(shapes),
            "stamp_vocabularies": len(vocabularies), "logical_frame_shapes": len(frame_shapes),
            "frame_counts": sorted(counts), "scales": sorted(scales), "mark_counts": sorted(marks),
            "stamp_sizes": sorted(stamp_sizes), "shift_counts": dict(shifts),
            "edge_pairs": edges, "empty_frames": empty_frames, "singleton_frames": singleton_frames}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
