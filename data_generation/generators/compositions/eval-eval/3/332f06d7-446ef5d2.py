"""Sticky jigsaw courier: edge adhesion changes the collision footprint.

Only standard-library modules and NumPy are used. solve reads the current input;
it never reads seeds, parent evidence, outputs, or generation-time metadata.
"""
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        queue = deque([first])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    queue.append(q)
        result.append(cells)
    return result


def _decode(grid):
    """Read the visible roles, L corner, channel axis, and payload rectangles."""
    g = _grid(grid)
    wall = int(g[0, 0])
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    _require(bool(np.all(border == wall)), "perimeter must be uniform wall")
    colors, counts = np.unique(g[g != wall], return_counts=True)
    _require(len(colors) >= 4, "missing structural or payload roles")
    maxima = colors[counts == counts.max()]
    _require(len(maxima) == 1, "channel color is ambiguous")
    floor = int(maxima[0])
    candidates = []
    for color, count in zip(colors, counts):
        if int(count) != 3 or int(color) == floor:
            continue
        cells = np.argwhere(g == color)
        low, high = cells.min(0), cells.max(0)
        if tuple(high - low) != (1, 1):
            continue
        square = {(r, c) for r in range(int(low[0]), int(high[0]) + 1)
                  for c in range(int(low[1]), int(high[1]) + 1)}
        missing = square - set(map(tuple, cells.tolist()))
        point = missing.pop()
        if int(g[point]) in (wall, floor, int(color)):
            continue
        candidates.append((int(color), point, tuple(map(int, low)),
                           tuple(map(tuple, cells.tolist()))))
    _require(len(candidates) == 1, "need one unambiguous external L")
    marker, point, marker_low, marker_cells = candidates[0]
    payload = (g != wall) & (g != floor) & (g != marker)
    pieces = []
    seed = None
    for cells in _components(payload):
        q = np.array(cells)
        low, high = q.min(0), q.max(0)
        shape = tuple(map(int, high - low + 1))
        _require(min(shape) >= 2 and len(cells) == int(np.prod(shape)),
                 "input fragments must be filled rectangles of sides >=2")
        if point in cells:
            seed = len(pieces)
            _require(point in {(int(r), int(c)) for r in (low[0], high[0])
                               for c in (low[1], high[1])}, "L must mark a corner")
        pieces.append({"cells": {p: int(g[p]) for p in cells},
                       "low": tuple(map(int, low)), "high": tuple(map(int, high)),
                       "shape": shape})
    _require(seed is not None and 3 <= len(pieces) <= 5, "fragment/seed count")
    payload_colors = tuple(map(int, np.unique(g[payload])))
    _require(2 <= len(payload_colors) <= 4, "payload palette size")
    _require(all(np.count_nonzero(g[payload] == c) >= 4 for c in payload_colors),
             "payload colors need >=4 cells")
    allowed = (g != wall) & (g != marker)
    _require(len(_components(allowed)) == 1, "channel must be connected")
    points = np.argwhere(allowed)
    low, high = points.min(0), points.max(0)
    spans = high - low + 1
    _require(spans[0] != spans[1], "channel has no unique long axis")
    axis = int(np.argmax(spans))
    sign = 1 if point[axis] > marker_low[axis] else -1
    direction = (sign, 0) if axis == 0 else (0, sign)
    transverse = 1 - axis
    root = pieces[seed]
    ordered = sorted(range(len(pieces)),
                     key=lambda i: sign * (pieces[i]["low"][axis]
                                           if sign > 0 else pieces[i]["high"][axis]))
    _require(ordered[0] == seed, "seed must be at rear of fragment order")
    for i in ordered:
        p = pieces[i]
        _require(p["low"][transverse] <= root["low"][transverse]
                 and p["high"][transverse] >= root["high"][transverse],
                 "fragments must share the seed's transverse core")
    for i, j in zip(ordered, ordered[1:]):
        if sign > 0:
            gap = pieces[j]["low"][axis] - pieces[i]["high"][axis] - 1
        else:
            gap = pieces[i]["low"][axis] - pieces[j]["high"][axis] - 1
        _require(gap >= 2, "input pieces need two empty cells along travel")
    lane = [tuple(root["low"][k] if k == axis else v for k in range(2))
            for v in range(root["low"][transverse], root["high"][transverse] + 1)]
    # The core lane after the starting fragment is visibly passable throughout.
    for p in lane:
        for coordinate in range(int(low[axis]), int(high[axis]) + 1):
            if sign * (coordinate - root["low"][axis]) < 0:
                continue
            q = list(p)
            q[axis] = coordinate
            _require(allowed[tuple(q)], "seed lane must reach channel end")
    return dict(grid=g, wall=wall, floor=floor, marker=marker,
                marker_cells=marker_cells, point=point, pieces=pieces, seed=seed,
                ordered=ordered, direction=direction, axis=axis, allowed=allowed,
                payload_colors=payload_colors, channel_shape=tuple(map(int, spans)))


def _contacts(cargo, piece, direction):
    dr, dc = direction
    return [(value, piece[(r + dr, c + dc)]) for (r, c), value in cargo.items()
            if (r + dr, c + dc) in piece]


def _simulate(info, *, adhesion=True, equality=True, walls=True):
    """Monotone motion with explicit contact-before-translation event order."""
    g, pieces = info["grid"], info["pieces"]
    seed, direction = info["seed"], info["direction"]
    dr, dc = direction
    cargo = dict(pieces[seed]["cells"])
    remaining = {i for i in range(len(pieces)) if i != seed}
    joined_at = {seed: 0}
    events = []
    steps = 0
    bbox_witness = False
    cause = None
    mismatch_equal = mismatch_unequal = 0
    for _ in range(64):
        touching = [(i, _contacts(cargo, pieces[i]["cells"], direction))
                    for i in sorted(remaining)]
        touching = [(i, seam) for i, seam in touching if seam]
        _require(len(touching) <= 1, "multiple simultaneous encounters outside family")
        if touching:
            i, seam = touching[0]
            if not adhesion:
                cause = "uncollected"
                break
            unequal = sum(a != b for a, b in seam)
            if equality and unequal:
                cause = "mismatch"
                mismatch_equal = len(seam) - unequal
                mismatch_unequal = unequal
                break
            _require(not (set(cargo) & set(pieces[i]["cells"])), "pickup overlap")
            cargo.update(pieces[i]["cells"])
            remaining.remove(i)
            joined_at[i] = steps
            events.append((i, steps, len(seam)))
        coords = np.array(list(cargo))
        low, high = coords.min(0), coords.max(0)
        box = info["allowed"][low[0]:high[0] + 1, low[1]:high[1] + 1]
        bbox_witness |= not bool(box.all())
        moved = {(r + dr, c + dc): v for (r, c), v in cargo.items()}
        if any(not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]) for r, c in moved):
            cause = "boundary"
            break
        if walls and any(not info["allowed"][p] for p in moved):
            cause = "wall"
            break
        occupied_loose = {p for i in remaining for p in pieces[i]["cells"]}
        _require(not (set(moved) & occupied_loose), "unhandled forward collision")
        cargo = moved
        steps += 1
    _require(cause is not None, "monotone movement exceeded canvas bound")
    output = g.copy()
    for p in info["marker_cells"]:
        output[p] = info["wall"]
    for i in joined_at:
        for p in pieces[i]["cells"]:
            output[p] = info["floor"]
    for p, value in cargo.items():
        output[p] = value
    final_coords = np.array(list(cargo))
    silhouette = tuple(map(int, final_coords.max(0) - final_coords.min(0) + 1))
    return output, dict(cause=cause, steps=steps, joined_at=joined_at,
                        remaining=remaining, events=events, cargo=cargo,
                        collected=len(joined_at) - 1, bbox_witness=bbox_witness,
                        mismatch_equal=mismatch_equal, mismatch_unequal=mismatch_unequal,
                        silhouette=silhouette,
                        terminal=len(remaining) == 0 and cause in ("wall", "boundary"))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied grid, returning a fresh integer array."""
    return _simulate(_decode(input_grid))[0]


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                 (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _pattern(rng, row, height, width, palette):
    # Connected horizontal bands rather than independent pixel speckles.
    values = rng.choice(palette, height)
    matrix = np.repeat(values[:, None], width, axis=1).astype(np.int8)
    if width >= 3 and rng.random() < 0.65:
        a = int(rng.integers(height))
        b = int(rng.integers(a + 1, height + 1))
        matrix[a:b, 1:-1] = int(rng.choice(palette))
    return matrix


def _make_pair(rng, kind, loose_count, rotation, used):
    """Construct new strips and channel, then recover all behavior from pixels."""
    for _ in range(96):
        palette_count = int(rng.integers(2, 5))
        colors = list(map(int, rng.permutation(10)[:3 + palette_count]))
        wall, floor, marker = colors[:3]
        palette = colors[3:]
        top, left = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        channel_height = int(rng.integers(6, 10))
        root_h, root_w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        root_r = top + int(rng.integers(0, 2))
        widths = [root_w] + list(map(int, rng.integers(2, 4, loose_count)))
        gaps = [2] * loose_count
        tail = int(rng.integers(2, 5))
        right = int(rng.integers(1, 4))
        budget = 30 - (left + 1 + sum(widths) + sum(gaps) + tail + right)
        if budget < 0:
            continue
        for i in rng.permutation(loose_count):
            extra = int(rng.integers(0, min(2, budget) + 1))
            gaps[int(i)] += extra
            budget -= extra
        positions = [(root_r, left + 1)]
        heights = [root_h]
        growth_index = int(rng.integers(1, loose_count)) if kind == "growth" else None
        for i in range(loose_count):
            up = int(rng.integers(0, root_r - top + 1))
            available_down = top + channel_height - (root_r + root_h)
            down = int(rng.integers(0, min(3, available_down) + 1))
            if i == 0 or i + 1 == growth_index:
                down = int(rng.integers(1, min(3, available_down) + 1))
            positions.append((root_r - up, positions[-1][1] + widths[i] + gaps[i]))
            heights.append(root_h + up + down)
        end = positions[-1][1] + widths[-1] + tail
        bottom = top + channel_height
        shape = (bottom + int(rng.integers(1, 4)), end + right)
        g = np.full(shape, wall, dtype=np.int8)
        g[top:bottom, left:end] = floor
        # Rear narrowing is passed by the small seed, before acquiring extra rows.
        first_x = positions[1][1]
        notch_len = int(rng.integers(1, gaps[0] + 1))
        g[root_r + root_h:bottom, first_x - notch_len:first_x] = wall
        if kind == "growth":
            throat = (positions[growth_index][1] + widths[growth_index]
                      + int(rng.integers(1, gaps[growth_index])))
            throat_len = int(rng.integers(1, 4))
            # The selected pickup has extra rows; earlier pickups can precede
            # this obstruction, so growth-stop timing is not tied to pickup 1.
            g[root_r + root_h:bottom, throat:min(end, throat + throat_len)] = wall
            # Further restrictions stay after the first obstruction.
            if rng.random() < 0.65:
                x = min(end - 1, positions[-1][1] + widths[-1])
                cut = int(rng.integers(root_r + root_h, bottom))
                g[cut:bottom, x:min(end, x + int(rng.integers(1, 4)))] = wall
        matrices = [_pattern(rng, r, h, w, palette)
                    for (r, _), h, w in zip(positions, heights, widths)]
        for i in range(1, len(matrices)):
            prev_r, r = positions[i - 1][0], positions[i][0]
            lo = max(prev_r, r)
            hi = min(prev_r + heights[i - 1], r + heights[i])
            matrices[i][lo - r:hi - r, 0] = matrices[i - 1][lo - prev_r:hi - prev_r, -1]
        if kind == "seam":
            bad_index = int(rng.integers(2, loose_count + 1))
            row = root_r
            old = int(matrices[bad_index][row - positions[bad_index][0], 0])
            choices = [c for c in palette if c != old]
            matrices[bad_index][row - positions[bad_index][0], 0] = int(rng.choice(choices))
        for (r, c), m in zip(positions, matrices):
            # Restrictions never erase a parked fragment. Downstream pixels are
            # part of the channel too, even if they cannot be reached after growth.
            g[r:r + m.shape[0], c:c + m.shape[1]] = m
        root_c = positions[0][1]
        marker_cells = ((root_r - 1, root_c - 1), (root_r - 1, root_c),
                        (root_r, root_c - 1))
        for p in marker_cells:
            g[p] = marker
        if rng.integers(2):
            g = np.flipud(g)
        g = np.rot90(g, rotation).copy()
        signature = (g.shape, g.tobytes())
        if signature in used:
            continue
        try:
            info = _decode(g)
            out, behavior = _simulate(info)
            _require(behavior["steps"] > 0 and behavior["collected"] >= 1,
                     "every example must visibly move and assemble")
            _require(behavior["bbox_witness"], "rear notch must distinguish occupied mask")
            if kind == "growth":
                _require(behavior["cause"] == "wall" and not behavior["terminal"]
                         and behavior["collected"] == growth_index, "wrong growth stop")
            elif kind == "seam":
                _require(behavior["cause"] == "mismatch"
                         and behavior["mismatch_equal"] >= 1, "wrong seam stop")
            else:
                _require(behavior["terminal"], "wrong terminal outcome")
            _require(any(behavior["steps"] > time for _, time, _ in behavior["events"]),
                     "must translate an acquired fragment")
        except AssertionError:
            continue
        used.add(signature)
        return {"input": g, "output": out}
    raise RuntimeError("bounded construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    origin = int(rng.integers(4))
    train_rotations = (origin, (origin + 1) % 4)
    test_rotations = ((origin + 2) % 4, (origin + 3) % 4)
    train_plan = [("growth", int(rng.integers(2, 4)), train_rotations[0]),
                  ("seam", int(rng.integers(2, 4)), train_rotations[1])]
    for _ in range(num_train - 2):
        train_plan.append((str(rng.choice(("growth", "seam", "terminal"))),
                           int(rng.integers(2, 4)), int(rng.choice(train_rotations))))
    rng.shuffle(train_plan)
    test_plan = [(str(rng.choice(("growth", "seam", "terminal"))),
                  4, int(rng.choice(test_rotations))) for _ in range(num_test)]
    puzzle, used = {"train": [], "test": []}, set()
    for split, plan in (("train", train_plan), ("test", test_plan)):
        for kind, count, rotation in plan:
            puzzle[split].append(_make_pair(rng, kind, count, rotation, used))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas dimensions must be preserved")
    info = _decode(inp)
    expected, behavior = _simulate(info)
    _require(np.array_equal(out, expected), "wrong input-derived movement/adhesion")
    _require(not np.array_equal(inp, out), "identity example")
    _require(behavior["steps"] > 0 and behavior["collected"] >= 1,
             "missing visible movement/assembly")
    _require(any(behavior["steps"] > time for _, time, _ in behavior["events"]),
             "an acquired fragment must visibly translate")
    _require(all(out[p] == info["wall"] for p in info["marker_cells"]), "marker erasure")
    _require(not np.any(out == info["marker"]), "marker remains")
    _require(np.array_equal(inp[inp == info["wall"]], out[inp == info["wall"]]), "wall changed")
    for color in info["payload_colors"]:
        _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color),
                 "payload color conservation")
    _require(np.count_nonzero(out == info["floor"]) == np.count_nonzero(inp == info["floor"]),
             "vacated floor conservation")
    dr, dc = info["direction"]
    for i, piece in enumerate(info["pieces"]):
        delta = behavior["steps"] - behavior["joined_at"][i] if i in behavior["joined_at"] else 0
        for (r, c), value in piece["cells"].items():
            _require(out[r + delta * dr, c + delta * dc] == value,
                     "fragment must translate intact or stay fixed")
    _require(behavior["bbox_witness"], "no occupied-mask witness")
    return info, behavior


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    training = []
    testing = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            info, behavior = _analyze_pair(pair)
            signature = (info["grid"].shape, info["grid"].tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            (training if split == "train" else testing).append((info, behavior))
    growth = [(i, b) for i, b in training if b["cause"] == "wall" and not b["terminal"]]
    seams = [(i, b) for i, b in training if b["cause"] == "mismatch"]
    _require(bool(growth) and bool(seams), "training needs growth and mismatch contrasts")
    _require(all(b["mismatch_equal"] >= 1 and b["mismatch_unequal"] >= 1 for _, b in seams),
             "rejecting seams must retain a matching contact too")
    for info, behavior in growth:
        actual = _simulate(info)[0]
        _require(not np.array_equal(actual, _simulate(info, adhesion=False)[0]),
                 "omitting B adhesion did not change the growth answer")
        _require(not np.array_equal(actual, _simulate(info, walls=False)[0]),
                 "omitting A collision did not change the growth answer")
    for info, behavior in seams:
        _require(not np.array_equal(_simulate(info)[0], _simulate(info, equality=False)[0]),
                 "omitting B edge equality did not change the seam answer")
    train_directions = {i["direction"] for i, _ in training}
    _require(len(train_directions) == 2, "show two training directions")
    _require(not (train_directions & {i["direction"] for i, _ in testing}), "direction holdout")
    _require(max(len(i["pieces"]) for i, _ in training) <= 4
             and all(len(i["pieces"]) == 5 for i, _ in testing), "fragment count shift")


def _fixture(kind):
    """Small coordinate-specified cases; expected destinations are not solved."""
    g = np.zeros((10, 22), dtype=np.int8)
    g[2:8, 2:20] = 1
    g[5:8, 7:9] = 0
    root = np.array([[3, 4], [4, 3]], dtype=np.int8)
    first = np.array([[4, 4], [3, 3], [4, 4]], dtype=np.int8)
    last = np.array([[4, 3], [3, 4], [4, 3]], dtype=np.int8)
    if kind == "seam":
        last[0, 0] = 3
    if kind == "growth":
        g[5:8, 14:16] = 0
    g[3:5, 3:5] = root
    g[3:6, 9:11] = first
    g[3:6, 17:19] = last
    for p in ((2, 2), (2, 3), (3, 2)):
        g[p] = 2
    expected = g.copy()
    expected[2, 2] = expected[2, 3] = expected[3, 2] = 0
    expected[3:5, 3:5] = 1
    expected[3:6, 9:11] = 1
    if kind == "growth":
        expected[3:5, 10:12] = root
        expected[3:6, 12:14] = first
    elif kind == "seam":
        expected[3:5, 13:15] = root
        expected[3:6, 15:17] = first
    else:
        expected[3:6, 17:19] = 1
        expected[3:5, 14:16] = root
        expected[3:6, 16:18] = first
        expected[3:6, 18:20] = last
    return g, expected


def _fingerprint(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k]).tobytes())
                 for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))


def self_test():
    # Direct fixtures test wall stopping, ordered equality, event order, and
    # complete delivery. They are rotated/reflected as scenes, not as pieces.
    for kind in ("growth", "seam", "terminal"):
        g, expected = _fixture(kind)
        for turns in range(4):
            for reflect in (False, True):
                a = np.flipud(g) if reflect else g
                b = np.flipud(expected) if reflect else expected
                a, b = np.rot90(a, turns).copy(), np.rot90(b, turns).copy()
                _require(np.array_equal(solve(a), b), "direct fixture failed: " + kind)
                info = _decode(a)
                _, behavior = _simulate(info)
                _require(behavior["bbox_witness"], "fixture needs stepped-mask witness")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "cross-seed repetition")
    shapes, structures, roles, dimensions, profiles, outcomes, directions = (set() for _ in range(7))
    palette_sizes, counts, silhouettes, growth_counts = set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, behavior = _analyze_pair(pair)
                g = info["grid"]
                shapes.add(g.shape)
                rename = {}
                structures.add((g.shape, tuple(rename.setdefault(int(v), len(rename)) for v in g.flat)))
                roles.add((info["wall"], info["floor"], info["marker"]))
                dimensions.update(p["shape"] for p in info["pieces"])
                profiles.add(tuple(np.sum(info["allowed"], axis=1 - info["axis"]).tolist()))
                outcomes.add((split, behavior["cause"], behavior["terminal"]))
                directions.add(info["direction"])
                palette_sizes.add(len(info["payload_colors"]))
                counts.add(len(info["pieces"]))
                silhouettes.add(behavior["silhouette"])
                if behavior["cause"] == "wall" and not behavior["terminal"]:
                    growth_counts.add(behavior["collected"])
                _require(np.array_equal(solve(g.copy()), pair["output"]), "solver mismatch")
    _require(len(shapes) >= 35 and len(structures) >= 160, "geometry diversity collapsed")
    _require(len(roles) >= 100 and len(dimensions) >= 20, "role/fragment diversity collapsed")
    _require(len(profiles) >= 90 and len(silhouettes) >= 20, "profile/silhouette diversity collapsed")
    _require(palette_sizes == {2, 3, 4} and counts == {3, 4, 5}, "palette/count diversity")
    _require(directions == {(0, 1), (0, -1), (1, 0), (-1, 0)}, "direction diversity")
    _require(len(growth_counts) >= 2, "growth stops must occur after different pickup counts")
    _require({(s, c, t) for s, c, t in outcomes if s == "test"}
             >= {("test", "wall", False), ("test", "wall", True), ("test", "mismatch", False)},
             "test outcomes collapsed")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(914 + n * 10 + m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    _require(_fingerprint(generate_puzzle(122, 7, 3)) == _fingerprint(generate_puzzle(122, 7, 3)),
             "repeat determinism")
    for mutation in ("pixel", "marker", "duplicate", "translation"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if mutation == "pixel":
            point = next(iter(_simulate(info)[1]["cargo"]))
            pair["output"][point] = info["floor"]
        elif mutation == "marker":
            pair["output"][info["marker_cells"][0]] = info["marker"]
        elif mutation == "translation":
            pair["output"] = np.roll(pair["output"], 1, axis=0)
        else:
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted mutation: " + mutation)
    return dict(seeds=42, pairs=210, shapes=len(shapes), structures=len(structures),
                role_assignments=len(roles), fragment_dimensions=len(dimensions),
                width_profiles=len(profiles), silhouettes=len(silhouettes),
                palette_sizes=sorted(palette_sizes), piece_counts=sorted(counts),
                growth_stop_pickups=sorted(growth_counts),
                independent_solver=False)


if __name__ == "__main__":
    print(self_test())
