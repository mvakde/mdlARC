"""Mirror-jump stamp trails: corner propagation with event-triggered bridges.

Only the input is needed to solve.  The three colors are background, the full
mirror line, and a solid non-square seed.  See task_spec.md for the complete rule.
"""

from __future__ import annotations

import json

import numpy as np


HOST_SEEDS = (
    0, 695282751, 547791542, 339767913, 444006435, 180425503,
    276461611, 177448956, 675627331, 438926775, 556759213,
    342221146, 499487379, 23455085, 633709708, 28642458,
    957433115, 301069417, 750864670, 472143188, 229768988,
    457945042, 305334808, 937176311, 236208837, 961253492,
    557308718, 323097034, 758498386, 738424384, 854621072,
    275062487, 575959118, 79796216, 145575457, 452041092,
    381324892, 213686183, 783043585, 730123502, 375287637,
    888060470,
)
_SHAPES = tuple((h, w) for h in range(1, 5) for w in range(1, 5) if h != w)
_RICH_SHAPES = tuple((h, w) for h, w in _SHAPES if h + w <= 6)
_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= d <= 30 for d in array.shape),
             "grids must have dimensions 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= array) & (array <= 9))), "colors must be 0..9")
    return array


def _read_input(value):
    """Normalize a horizontal mirror by transposing it to a vertical mirror."""
    original = _grid(value)
    colors, counts = np.unique(original, return_counts=True)
    _require(len(colors) == 3, "input must have exactly three role colors")
    _require(int(np.sum(counts == counts.max())) == 1, "background must be uniquely modal")
    background = int(colors[int(np.argmax(counts))])
    rows = [r for r in range(original.shape[0])
            if original[r, 0] != background and np.all(original[r] == original[r, 0])]
    cols = [c for c in range(original.shape[1])
            if original[0, c] != background and np.all(original[:, c] == original[0, c])]
    _require(len(rows) + len(cols) == 1, "input needs one unique complete control line")
    transposed = bool(rows)
    grid = original.T if transposed else original
    axis = rows[0] if transposed else cols[0]
    line = int(grid[0, axis])
    seed_color = next(int(c) for c in colors if int(c) not in (background, line))
    points = np.argwhere(grid == seed_color)
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0) + 1)
    h, w = r1 - r0, c1 - c0
    _require(h != w, "seed must be a non-square solid rectangle")
    _require(c1 <= axis or c0 > axis, "seed must lie wholly on one side of the line")
    expected = np.full(grid.shape, background, dtype=int)
    expected[:, axis] = line
    expected[r0:r1, c0:c1] = seed_color
    _require(np.array_equal(grid, expected), "only the line and one solid seed are allowed")
    return {"grid": grid, "transposed": transposed, "axis": axis,
            "background": background, "line": line, "seed_color": seed_color,
            "box": (r0, r1, c0, c1), "shape": (h, w),
            "side": -1 if c1 <= axis else 1}


def _next_box(box, dr, dc, swap=True):
    r0, r1, c0, c1 = box
    h, w = r1 - r0, c1 - c0
    nh, nw = (w, h) if swap else (h, w)
    nr0 = r0 - nh if dr < 0 else r1
    nc0 = c0 - nw if dc < 0 else c1
    return nr0, nr0 + nh, nc0, nc0 + nw


def _visible(box, shape):
    r0, r1, c0, c1 = box
    return r1 > 0 and r0 < shape[0] and c1 > 0 and c0 < shape[1]


def _full(box, shape):
    r0, r1, c0, c1 = box
    return 0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]


def _reflect(box, axis):
    r0, r1, c0, c1 = box
    return r0, r1, 2 * axis - c1 + 1, 2 * axis - c0 + 1


def _bridge(box, reflected):
    left, right = sorted((box, reflected), key=lambda b: b[2])
    return box[0], box[1], left[3], right[2]


def _paint_box(grid, box, color):
    if not _visible(box, grid.shape):
        return
    r0, r1, c0, c1 = box
    grid[max(0, r0):min(grid.shape[0], r1),
         max(0, c0):min(grid.shape[1], c1)] = color


def _construct(info, *, portals=True, swap=True, reset_phase=False):
    """Input-driven state machine, also exposing geometric event records.

    Boxes use exclusive upper bounds, even when clipped. A jump duplicates the
    CURRENT box, so it does not consume a propagation step or reset the phase.
    """
    grid = info["grid"]
    output = grid.copy()
    axis, line, seed_color = info["axis"], info["line"], info["seed_color"]
    records, events = [], []
    for direction, (dr, dc) in enumerate(_DIRECTIONS):
        current, step, jumped = info["box"], 0, False
        phase_offset = 0
        inward = dc == -info["side"]
        for _ in range(100):
            candidate = _next_box(current, dr, dc, swap)
            # A contact wholly outside the finite canvas causes no jump.
            if not _visible(candidate, grid.shape):
                break
            if portals and inward and not jumped and candidate[2] <= axis < candidate[3]:
                reflected = _reflect(current, axis)
                _require(0 <= reflected[2] < reflected[3] <= grid.shape[1],
                         "the docking stamp's reflection must fit perpendicular to the line")
                bridge = _bridge(current, reflected)
                color = seed_color if (step + phase_offset) % 2 == 0 else line
                _paint_box(output, bridge, line)
                _paint_box(output, reflected, color)
                events.append({"direction": direction, "step": step, "box": current,
                               "reflected": reflected, "bridge": bridge,
                               "blocked": candidate, "color": color})
                current, jumped = reflected, True
                if reset_phase:
                    phase_offset = -step
                candidate = _next_box(current, dr, dc, swap)
                if not _visible(candidate, grid.shape):
                    break
            step += 1
            color = seed_color if (step + phase_offset) % 2 == 0 else line
            _paint_box(output, candidate, color)
            records.append({"direction": direction, "step": step, "box": candidate,
                            "jumped": jumped, "color": color,
                            "full": _full(candidate, grid.shape)})
            current = candidate
        else:
            raise AssertionError("propagation failed to terminate")
    return output, records, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the mirror-jump construction using only the supplied input."""
    info = _read_input(input_grid)
    output, _, _ = _construct(info)
    return output.T.copy() if info["transposed"] else output


def _key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def _fingerprint(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _sum_steps(h, w, k):
    # Total parallel and perpendicular advance after k quarter-turn stamps.
    return ((k + 1) // 2) * w + (k // 2) * h, ((k + 1) // 2) * h + (k // 2) * w


def _make_input(rng, palette, shape, depth, rich, horizontal, right_side):
    """Choose geometry by the number of safe inward stamps before contact.

    depth is solely a sampling device. It is neither encoded as metadata nor
    consulted by solve: the input distance determines it exactly.
    """
    h, w = shape
    row_advance, col_advance = _sum_steps(h, w, depth)
    next_width = h if depth % 2 == 0 else w
    # Rich demonstrations must visibly add bridge material on both sides of
    # the old line; a one-cell bridge would hide this part of parent A.
    residual_gap = int(rng.integers(1 if rich else 0, next_width))
    initial_gap = col_advance + residual_gap
    current_width = w if depth % 2 == 0 else h
    if rich:
        margin = row_advance + h + w
        min_height = max(18, h + 2 * margin)
        min_axis = w + initial_gap + h + w
        right_room = current_width + residual_gap + h + w
    else:
        margin = 0
        min_height = max(16, h + row_advance + 1)
        min_axis = w + initial_gap
        right_room = current_width + residual_gap
    min_width = max(18 if rich else 16, min_axis + 1 + right_room)
    _require(min_height <= 30 and min_width <= 30, "chosen geometry exceeds ARC canvas")
    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    axis_choices = list(range(min_axis, width - right_room))
    center = (width - 1) // 2
    axis = center if center in axis_choices and rng.random() < 0.3 else int(rng.choice(axis_choices))
    col = axis - initial_gap - w
    if rich:
        row = int(rng.integers(margin, height - h - margin + 1))
    else:
        valid = [r for r in range(height - h + 1)
                 if r >= row_advance + 1 or r + h + row_advance < height]
        edges = [r for r in (0, 1, height - h - 1, height - h) if r in valid]
        row = int(rng.choice(edges if edges and rng.random() < 0.5 else valid))
    background, line, seed_color = palette
    grid = np.full((height, width), background, dtype=np.int8)
    grid[:, axis] = line
    grid[row:row + h, col:col + w] = seed_color
    if right_side:
        grid = grid[:, ::-1].copy()
    if horizontal:
        grid = grid.T.copy()
    return grid


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Create a deterministic puzzle, with new jump depths in the test split."""
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    stable_palette = bool(rng.integers(2))
    first_horizontal = bool(rng.integers(2))
    first_right = bool(rng.integers(2))
    first_depth = int(rng.choice((1, 2)))
    test_depth = int(rng.choice((0, 3)))
    puzzle, used = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        training = index < num_train
        rich = training and (index < 2 or rng.random() < 0.25)
        depth = (first_depth if index == 0 else 3 - first_depth if index == 1
                 else int(rng.choice((1, 2)))) if training else test_depth
        horizontal = first_horizontal if index == 0 else not first_horizontal if index == 1 else bool(rng.integers(2))
        right_side = first_right if index == 0 else not first_right if index == 1 else bool(rng.integers(2))
        pool = tuple((h, w) for h, w in _RICH_SHAPES
                     if (h if depth % 2 == 0 else w) >= 2) if rich else _SHAPES
        shape = pool[int(rng.integers(len(pool)))]
        roles = palette if stable_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
        for _ in range(100):
            inp = _make_input(rng, roles, shape, depth, rich, horizontal, right_side)
            if index == 0:
                _, sampled_stamps, _ = _construct(_read_input(inp))
                if all(s["full"] for s in sampled_stamps):
                    continue
            if _key(inp) not in used:
                break
        else:
            raise AssertionError("could not make distinct inputs")
        used.add(_key(inp))
        puzzle["train" if training else "test"].append({"input": inp, "output": solve(inp)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _mirror_seed_only(info):
    out = info["grid"].copy()
    reflected = _reflect(info["box"], info["axis"])
    _paint_box(out, _bridge(info["box"], reflected), info["line"])
    # This alternative is clipped if the ORIGINAL seed's reflection is outside.
    _paint_box(out, reflected, info["seed_color"])
    return out


def validate_generated_puzzle(puzzle) -> None:
    """Check actual grids, phase/transport invariants, and demonstration coverage."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "invalid pair counts")
    used, analyzed = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas changed")
            _require(_key(inp) not in used, "inputs must be distinct within a puzzle")
            used.add(_key(inp))
            info = _read_input(inp)
            _require(info["shape"] in _SHAPES, "generated seed geometry outside sampling range")
            predicted, stamps, events = _construct(info)
            actual = out.T if info["transposed"] else out
            _require(np.array_equal(predicted, actual), "incorrect output")
            _require(events, "every pair must exhibit a mirror jump")
            _require(stamps and np.any(inp != out), "every pair must propagate")
            _require(np.all(actual[:, info["axis"]] == info["line"]), "axis was damaged")
            r0, r1, c0, c1 = info["box"]
            _require(np.all(actual[r0:r1, c0:c1] == info["seed_color"]), "original seed was damaged")
            _require(set(map(int, np.unique(out))) == set(map(int, np.unique(inp))), "palette changed")
            for event in events:
                box, mirror, blocked = event["box"], event["reflected"], event["blocked"]
                _require(_full(box, info["grid"].shape) and _full(mirror, info["grid"].shape),
                         "a docking stamp and its reflection must be complete")
                _require(box[2] + mirror[3] - 1 == 2 * info["axis"], "not cell-centered reflection")
                _require(box[3] + mirror[2] - 1 == 2 * info["axis"], "reflection thickness changed")
                _require(blocked[2] <= info["axis"] < blocked[3], "jump without contact")
                _require(event["color"] == (info["seed_color"] if event["step"] % 2 == 0 else info["line"]),
                         "jump reset the phase")
                # An event changes neither the dimensions nor the parallel span.
                _require(box[:2] == mirror[:2] and box[3] - box[2] == mirror[3] - mirror[2],
                         "transport changed stamp geometry")
                _require(event["direction"] in (1, 3) if info["side"] < 0 else event["direction"] in (0, 2),
                         "an outward branch jumped")
            for stamp in stamps:
                h, w = info["shape"]
                expected_shape = (w, h) if stamp["step"] % 2 else (h, w)
                b = stamp["box"]
                _require((b[1] - b[0], b[3] - b[2]) == expected_shape, "stamp did not quarter-turn")
                _require(not (b[2] <= info["axis"] < b[3]), "a trail crossed the axis without jumping")
            no_a, _, _ = _construct(info, portals=False)
            no_a[:, info["axis"]] = info["line"]
            _require(not np.array_equal(actual, no_a), "ignoring the mirror does not change this answer")
            _require(not np.array_equal(actual, _mirror_seed_only(info)),
                     "ignoring propagation does not change this answer")
            analyzed[split].append((info, stamps, events))
    # The first two demonstrations suffice even for the smallest legal puzzle.
    first = analyzed["train"][:2]
    _require({item[0]["transposed"] for item in first} == {False, True}, "demonstrate both orientations")
    _require({item[0]["side"] for item in first} == {-1, 1}, "demonstrate both seeded sides")
    _require({events[0]["step"] for _, _, events in first} == {1, 2}, "demonstrate both docking phases")
    for info, stamps, events in first:
        _require(len(events) == 2, "rich training must expose both inward branches")
        _require({s["direction"] for s in stamps if s["step"] <= 2 and s["full"]} == set(range(4)),
                 "rich training must expose all four directions")
        for event in events:
            _require(event["bridge"][3] - event["bridge"][2] >= 3,
                     "rich training must visibly extend the bridge beyond the old line")
            following = [s for s in stamps if s["direction"] == event["direction"]
                         and s["jumped"] and s["full"]]
            _require(len(following) >= 2, "show two complete stamps after each jump")
        no_swap, _, _ = _construct(info, swap=False)
        _require(not np.array_equal(no_swap, _construct(info)[0]), "training fails to expose dimension swaps")
    _require(any(not np.array_equal(_construct(i, reset_phase=True)[0], _construct(i)[0])
                 for i, _, _ in first), "training does not distinguish phase reset")
    _require(any(not s["full"] for s in first[0][1]), "demonstrate clipping in training")
    train_depths = {e["step"] for _, _, ev in analyzed["train"] for e in ev}
    test_depths = {e["step"] for _, _, ev in analyzed["test"] for e in ev}
    _require(train_depths == {1, 2} and test_depths <= {0, 3} and test_depths,
             "missing intended distance/depth shift")


def self_test():
    """Bounded 42-seed audit, explicit-count coverage, and local metamorphic checks."""
    puzzles, signatures = [], set()
    shapes, seed_shapes, backgrounds, palettes = set(), set(), set(), set()
    geometries, depths, event_counts, clipped, aspects = set(), set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "seed is not reproducible")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _read_input(inp)
                _, stamps, events = _construct(info)
                shapes.add(inp.shape)
                seed_shapes.add(info["shape"])
                backgrounds.add(info["background"])
                palettes.add((info["background"], info["line"], info["seed_color"]))
                geometries.add((info["grid"].shape, info["axis"], info["box"], info["transposed"]))
                depths.update(e["step"] for e in events)
                event_counts.add(len({(e["box"], e["reflected"]) for e in events}))
                clipped.add(any(not s["full"] for s in stamps))
                aspects.add("square" if inp.shape[0] == inp.shape[1] else "wide" if inp.shape[1] > inp.shape[0] else "tall")
        # Rotating the whole scene, reflecting it, or renaming colors commutes
        # with solving; none of those transformations carries hidden metadata.
        pair = puzzle["test"][0]
        inp, out = pair["input"], pair["output"]
        _require(np.array_equal(solve(np.rot90(inp)), np.rot90(out)), "rotation equivariance failed")
        _require(np.array_equal(solve(inp[:, ::-1]), out[:, ::-1]), "reflection equivariance failed")
        mapping = np.array([7, 4, 9, 2, 0, 8, 1, 6, 3, 5])
        _require(np.array_equal(solve(mapping[inp]), mapping[out]), "palette equivariance failed")
    _require(len(signatures) == 42, "seeds produced duplicate puzzles")
    _require(len(shapes) >= 45 and len(geometries) >= 200, "insufficient geometric variation")
    _require(seed_shapes == set(_SHAPES), "not all sampled rectangle shapes appeared")
    _require(len(backgrounds) == 10 and len(palettes) >= 35, "palette roles lack variation")
    _require(depths == {0, 1, 2, 3}, "jump depth variation is incomplete")
    _require({1, 2} <= event_counts and clipped == {False, True}, "boundary regimes lack variation")
    _require(aspects == {"square", "wide", "tall"}, "canvas aspects lack variation")
    # Hand-calculated anchors for the event row, reflection offset, skipped box,
    # preserved phase, and the finite-line boundary. These do not use _next_box
    # or _reflect to obtain their expected coordinates.
    for col, depth in ((5, 1), (3, 2)):
        anchor = np.zeros((20, 25), dtype=int)
        anchor[:, 12] = 5
        anchor[8:10, col:col + 3] = 2
        result = solve(anchor)
        _, _, events = _construct(_read_input(anchor))
        _require({e["step"] for e in events} == {depth}, "anchor docked at the wrong step")
        _require(np.all(result[8:10, 17:20] == 0), "original seed was mirrored prematurely")
        if depth == 1:
            _require(np.all(result[5:8, 8:17] == 5), "odd docking stripe is incorrect")
            _require(np.all(result[3:5, 17:20] == 2), "odd docking reset the next stamp phase")
            _require(np.all(result[3:5, 10:12] == 0), "blocked stamp was painted")
        else:
            _require(np.all(result[3:5, 14:17] == 2), "even docking reflection is incorrect")
            _require(np.all(result[3:5, 11:14] == 5), "bridge does not span the full current stamp")
            _require(np.all(result[0:3, 17:19] == 5), "even docking reset the next stamp phase")
    boundary = np.zeros((20, 25), dtype=int)
    boundary[:, 12] = 5
    boundary[1:3, 5:8] = 2
    _, _, events = _construct(_read_input(boundary))
    _require([e["direction"] for e in events] == [3], "an off-canvas contact created a bridge")
    _require(np.all(solve(boundary)[0:1, 11:17] == np.array([0, 5, 0, 0, 0, 0])),
             "the clipped top stamp was incorrectly mirrored")
    # The rule and solver are geometric beyond the sampler's depth 0..3 range.
    deeper = np.zeros((30, 30), dtype=int)
    deeper[:, 18] = 5
    deeper[14:16, 2:5] = 2
    deep_result = solve(deeper)
    _, _, deep_events = _construct(_read_input(deeper))
    _require(len(deep_events) == 2 and {e["step"] for e in deep_events} == {5},
             "solver cannot handle a valid unsampled jump depth")
    _require(np.all(deep_result[1:4, 15:22] == 5)
             and np.all(deep_result[0, 22:25] == 2),
             "deep contact lost its bridge, phase, or clipped departure")
    # Every legal explicit pair-count combination, with fresh count-dependent seeds.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(4000 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    # An altered output must fail the validator, rather than merely shape-check.
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged output")
    return {"seeds": len(HOST_SEEDS), "pairs": sum(len(p[s]) for p in puzzles for s in ("train", "test")),
            "canvas_shapes": len(shapes), "seed_shapes": len(seed_shapes),
            "geometries": len(geometries), "role_palettes": len(palettes),
            "jump_depths": sorted(depths), "distinct_bridge_counts": sorted(event_counts),
            "clipped_and_unclipped": sorted(clipped)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
