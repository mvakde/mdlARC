"""Repair relays: marked triangles broadcast through damaged rectangular plates.

Only standard-library modules and NumPy are used. The solver recognizes visible
geometry and computes a directed closure; it never calls the sampler.
"""

from collections import deque
import json

import numpy as np


DIRECTIONS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with sides in 1..30")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _triangle_cells(apex, direction, depth):
    dr, dc = direction
    pr, pc = -dc, dr
    return {(apex[0] - k * dr + j * pr, apex[1] - k * dc + j * pc)
            for k in range(depth) for j in range(-k, k + 1)}


def _triangle(grid, cells):
    colors, counts = np.unique([grid[p] for p in cells], return_counts=True)
    if len(colors) != 2 or list(counts).count(1) != 1:
        return None
    depth = int(round(len(cells) ** 0.5))
    if depth < 2 or depth * depth != len(cells):
        return None
    marker = int(colors[np.argmin(counts)])
    body = int(colors[np.argmax(counts)])
    lo_r, hi_r, lo_c, hi_c = _bbox(cells)
    answers = []
    for direction in DIRECTIONS:
        dr, dc = direction
        apex = ((lo_r + hi_r) // 2, hi_c) if dc == 1 else (
            ((lo_r + hi_r) // 2, lo_c) if dc == -1 else (
                (lo_r, (lo_c + hi_c) // 2) if dr == -1
                else (hi_r, (lo_c + hi_c) // 2)))
        if _triangle_cells(apex, direction, depth) != cells:
            continue
        base = apex[0] - (depth - 1) * dr, apex[1] - (depth - 1) * dc
        if grid[base] != marker or any(grid[p] != body for p in cells - {base}):
            continue
        answers.append(dict(cells=cells, apex=apex, base=base, direction=direction,
                            depth=depth, marker=marker, body=body))
    return answers[0] if len(answers) == 1 else None


def _repair(cells):
    """Find the unique rectangle plus a straight, attached two-cell stem.

    Enumeration uses only the component's area and bounding box, so it is
    independent of the generator's remembered rectangle or defect direction.
    """
    area = len(cells) - 2
    if area < 9:
        return None
    r0, r1, c0, c1 = _bbox(cells)
    candidates = []
    for h in range(3, r1 - r0 + 2):
        if area % h:
            continue
        w = area // h
        if not 3 <= w <= c1 - c0 + 1:
            continue
        for r in range(r0, r1 - h + 2):
            for c in range(c0, c1 - w + 2):
                core = {(y, x) for y in range(r, r + h) for x in range(c, c + w)}
                if not core <= cells:
                    continue
                extra = cells - core
                a, b = sorted(extra)
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                    continue
                # Exactly one extra is the end of a straight outward stem.
                valid = False
                for end, tip in ((a, b), (b, a)):
                    d = tip[0] - end[0], tip[1] - end[1]
                    attachment = end[0] - d[0], end[1] - d[1]
                    if attachment in core:
                        valid = True
                if valid:
                    candidates.append(dict(cells=cells, core=core, extra=extra,
                                           bbox=(r, r + h - 1, c, c + w - 1)))
    return candidates[0] if len(candidates) == 1 else None


def _ports(plate, direction):
    """Return corresponding entering and exiting endpoints of all core runs."""
    r0, r1, c0, c1 = plate["bbox"]
    dr, dc = direction
    if dc:
        near_c, far_c = (c0, c1) if dc == 1 else (c1, c0)
        return ([(r, near_c) for r in range(r0, r1 + 1)],
                [(r, far_c) for r in range(r0, r1 + 1)])
    near_r, far_r = (r0, r1) if dr == 1 else (r1, r0)
    return ([(near_r, c) for c in range(c0, c1 + 1)],
            [(far_r, c) for c in range(c0, c1 + 1)])


def _propagate(shape, triangle, plates):
    """Finite directed reachability, with rays stopped by true rectangular cores."""
    owner = {p: i for i, plate in enumerate(plates) for p in plate["core"]}
    assert len(owner) == sum(len(p["core"]) for p in plates)
    dr, dc = triangle["direction"]
    pending = deque([(triangle["apex"], -1)])
    active, rays, faces, hits, edges = set(), set(), set(), [], set()
    while pending:
        (r, c), source = pending.popleft()
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                break
            q = r, c
            assert q not in triangle["cells"], "a relay points back into the emitter"
            if q in owner:
                i = owner[q]
                near, far = _ports(plates[i], triangle["direction"])
                assert q in near, "a ray must enter a plate through its near face"
                hits.append((i, q))
                edges.add((source, i))
                if i not in active:
                    active.add(i)
                    faces.update(near)
                    pending.extend((p, i) for p in far)
                break
            rays.add(q)
    return dict(active=active, rays=rays, faces=faces, hits=hits, edges=edges)


def _analyze(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert counts.max() > grid.size / 2, "background must have a strict majority"
    background = int(colors[np.argmax(counts)])
    parts = _components({tuple(map(int, p)) for p in np.argwhere(grid != background)})
    triangles, plates = [], []
    for part in parts:
        values = {int(grid[p]) for p in part}
        if len(values) > 1:
            triangle = _triangle(grid, part)
            assert triangle is not None, "unrecognized multicolor component"
            triangles.append(triangle)
        else:
            plate = _repair(part)
            assert plate is not None, "carrier is not uniquely repairable"
            plate["color"] = next(iter(values))
            plates.append(plate)
    assert len(triangles) == 1 and len(plates) >= 2
    triangle = triangles[0]
    carrier_colors = {p["color"] for p in plates}
    assert len(carrier_colors) == 1
    carrier = next(iter(carrier_colors))
    assert len({background, triangle["body"], triangle["marker"], carrier}) == 4
    trace = _propagate(grid.shape, triangle, plates)
    inactive = set(range(len(plates))) - trace["active"]
    assert trace["active"] and inactive, "need both activated and preserved plates"
    assert trace["rays"], "the outgoing axis must have a visible gap"
    assert not any(trace["rays"] & plates[i]["cells"] for i in inactive), (
        "an unreached stem may not occlude a ray")
    return dict(background=background, carrier=carrier, triangle=triangle,
                plates=plates, inactive=inactive, **trace)


def _render(grid, info):
    output = grid.copy()
    for i in info["active"]:
        for p in info["plates"][i]["extra"]:
            output[p] = info["background"]
    for p in info["rays"] | info["faces"]:
        output[p] = info["triangle"]["marker"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family scene from its input alone, without learned roles."""
    grid = _grid(input_grid)
    return _render(grid, _analyze(grid))


def _plate(top, left, h, w, side, port):
    core = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
    if side == 0:  # top
        extra = {(top - 1, left + port), (top - 2, left + port)}
    elif side == 1:  # right
        extra = {(top + port, left + w), (top + port, left + w + 1)}
    elif side == 2:  # bottom
        extra = {(top + h, left + port), (top + h + 1, left + port)}
    else:  # left
        extra = {(top + port, left - 1), (top + port, left - 2)}
    return dict(core=core, extra=extra, cells=core | extra,
                bbox=(top, top + h - 1, left, left + w - 1))


def _in_bounds(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _separated(a, b):
    return not (a & b) and not any((r + dr, c + dc) in b
                                 for r, c in a for dr, dc in DIRECTIONS)


def _move(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _choose_span(rng, h, height, previous, axis, overlap, older=None):
    a, b = previous
    spans = []
    for top in range(2, height - h - 1):
        rows = set(range(top, top + h))
        intersection = len(rows & set(range(a, b + 1)))
        if axis in rows or not intersection:
            continue
        if overlap == 1 and intersection != 1:
            continue
        if overlap == 2 and intersection < 2:
            continue
        if not (min(rows) < a or max(rows) > b):
            continue
        if older is not None and rows & set(range(older[0], older[1] + 1)):
            continue
        spans.append(top)
    if not spans:
        raise ValueError("no fitting off-axis receiving span")
    return spans[int(rng.integers(len(spans)))]


def _sample_scene(rng, active_count, overlap, rotation, marker_not=None,
                  first_defect=None, depth=None, identical_decoy=False):
    """Build rightward geometry, then rotate the complete interaction.

    The output below is drawn from the construction's intended active plates.
    A separate grid parse must recover the same activation set and pixels.
    """
    for attempt in range(350):
        try:
            height = int(rng.integers(22 if active_count == 3 else 18, 31))
            d = int(rng.integers(2, 5)) if depth is None else depth
            axis = int(rng.integers(6, height - 6))
            apex = axis, int(rng.integers(0, 4)) + d - 1
            triangle = dict(apex=apex, direction=(0, 1), depth=d,
                            cells=_triangle_cells(apex, (0, 1), d))
            active = []
            fork = active_count == 3 and rng.random() < 0.5
            previous_right = apex[1]
            for i in range(active_count):
                h = int(rng.integers(4 if i == 0 else 3, 8))
                w = int(rng.integers(3, 7))
                if i == 0:
                    top = axis - int(rng.integers(1, h - 1))
                    top = max(2, min(top, height - h - 2))
                    assert top <= axis < top + h
                else:
                    prev = active[0 if fork and i == 2 else -1]["bbox"][:2]
                    old = active[1 if fork else 0]["bbox"][:2] if i == 2 else None
                    top = _choose_span(rng, h, height, prev, axis,
                                       overlap if i == 1 else 0, old)
                # Defects must fit without touching the preceding object.
                side = (first_defect if i == 0 and first_defect is not None
                        else int(rng.integers(4)))
                if fork and i == 2:
                    # The two children occupy distinct transverse bands, so
                    # their axial extents may overlap instead of forming lanes.
                    left = max(max(c for _, c in active[0]["cells"]) + 2,
                               active[1]["bbox"][2] + int(rng.integers(-1, 4)))
                else:
                    left = previous_right + int(rng.integers(2, 6))
                if side == 3:
                    left += 2
                port = int(rng.integers(w if side in (0, 2) else h))
                if i == 0 and first_defect == 3:
                    port = axis - top  # A training witness: erase a spur in the beam.
                plate = _plate(top, left, h, w, side, port)
                occupied = triangle["cells"] | set().union(*(p["cells"] for p in active))
                assert _separated(plate["cells"], occupied)
                assert _in_bounds(plate["cells"], (height, 30))
                active.append(plate)
                previous_right = max(previous_right, max(c for _, c in plate["cells"]))
            min_width = max(16, previous_right + 1)
            assert min_width <= 30
            width = int(rng.integers(min_width, min(30, min_width + 6) + 1))
            shape = height, width
            trace = _propagate(shape, triangle, active)
            assert trace["active"] == set(range(active_count))
            plates = list(active)
            occupied = triangle["cells"] | set().union(*(p["cells"] for p in plates))
            decoy_count = int(rng.integers(1, 4))
            for index in range(decoy_count):
                if index == 0 and (identical_decoy or rng.random() < 0.7):
                    template = active[int(rng.integers(active_count))]
                else:
                    h, w = int(rng.integers(3, 8)), int(rng.integers(3, 7))
                    side = int(rng.integers(4))
                    template = _plate(0, 0, h, w, side,
                                      int(rng.integers(w if side in (0, 2) else h)))
                r0, r1, c0, c1 = _bbox(template["cells"])
                anchors = [(r, c) for r in range(-r0, height - r1)
                           for c in range(-c0, width - c1)]
                rng.shuffle(anchors)
                for dr, dc in anchors:
                    cells = _move(template["cells"], dr, dc)
                    if cells & trace["rays"] or not _separated(cells, occupied):
                        continue
                    plate = dict(cells=cells, core=_move(template["core"], dr, dc),
                                 extra=_move(template["extra"], dr, dc))
                    plate["bbox"] = _bbox(plate["core"])
                    plates.append(plate)
                    occupied.update(cells)
                    break
                else:
                    raise ValueError("no fitting unilluminated competitor")
            assert len(plates) == active_count + decoy_count
            colors = [int(c) for c in rng.permutation(10)[:4]]
            bg, body, carrier, marker = colors
            if marker == marker_not:
                continue
            grid = np.full(shape, bg, dtype=np.int8)
            for p in triangle["cells"]:
                grid[p] = body
            grid[axis, apex[1] - d + 1] = marker
            for plate in plates:
                for p in plate["cells"]:
                    grid[p] = carrier
            assert np.count_nonzero(grid == bg) > grid.size / 2
            output = grid.copy()
            for plate in active:
                for p in plate["extra"]:
                    output[p] = bg
            for p in trace["rays"] | trace["faces"]:
                output[p] = marker
            grid = np.rot90(grid, rotation).copy()
            output = np.rot90(output, rotation).copy()
            info = _analyze(grid)
            assert len(info["active"]) == active_count
            assert np.array_equal(output, _render(grid, info))
            return dict(input=grid, output=output)
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("could not construct a relay scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return a deterministic puzzle with locally visible, pair-level colors."""
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(2))  # 0: deeper closure; 1: one-row relay overlap.
    first_rotation = int(rng.integers(4))
    second_rotation = (first_rotation + int(rng.choice((1, 3)))) % 4
    first_depth = int(rng.integers(2, 5))
    second_depth = int(rng.choice([d for d in range(2, 5) if d != first_depth]))
    puzzle = {"train": [], "test": []}
    seen, first_marker = set(), None
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            if split == "train":
                n = 1 if index == 0 else (2 if index == 1 else int(rng.integers(1, 3)))
                overlap = 2
                rot = (first_rotation if index == 0 else
                       second_rotation if index == 1 else int(rng.integers(4)))
                defect = int(rng.choice((0, 2))) if index == 0 else (3 if index == 1 else None)
                depth = first_depth if index == 0 else second_depth if index == 1 else None
            else:
                n, overlap = (3, 0) if shift == 0 else (2, 1)
                rot, defect, depth = int(rng.integers(4)), None, None
            for retry in range(20):
                pair = _sample_scene(rng, n, overlap, rot,
                                     first_marker if split == "train" and index == 1 else None,
                                     defect, depth, split == "train" and index == 0)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("duplicate inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_marker = _analyze(pair["input"])["triangle"]["marker"]
    validate_generated_puzzle(puzzle)
    return puzzle


def _axis_intersects(triangle, core):
    ar, ac = triangle["apex"]
    dr, dc = triangle["direction"]
    return any((r - ar) * dc == (c - ac) * dr
               and (r - ar) * dr + (c - ac) * dc > 0 for r, c in core)


def _normalized(cells):
    return _move(cells, -min(r for r, _ in cells), -min(c for _, c in cells))


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _analyze(inp)
    assert np.array_equal(out, _render(inp, info)), "wrong relay output"
    bg, marker = info["background"], info["triangle"]["marker"]
    expected_changed = set(info["rays"] | info["faces"])
    for i, plate in enumerate(info["plates"]):
        if i in info["active"]:
            expected_changed.update(plate["extra"])
            assert all(out[p] != info["carrier"] for p in plate["extra"])
            near, _ = _ports(plate, info["triangle"]["direction"])
            assert all(out[p] == marker for p in near)
            assert all(out[p] == info["carrier"] for p in plate["core"] - set(near))
        else:
            assert all(out[p] == inp[p] for p in plate["cells"])
    actual_changed = {tuple(map(int, p)) for p in np.argwhere(inp != out)}
    assert actual_changed == expected_changed
    assert all(out[p] == inp[p] for p in info["triangle"]["cells"])
    assert all(out[p] == marker for p in info["rays"] | info["faces"])
    assert set(map(int, np.unique(out))) <= {bg, marker, info["carrier"], info["triangle"]["body"]}
    info["shape"] = inp.shape
    return info


def validate_generated_puzzle(puzzle):
    """Check pixels, unique repairs, propagation, competitors, and demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    assert {1, 2} <= {len(i["active"]) for i in train}
    assert len({i["triangle"]["direction"] for i in train}) >= 2
    assert len({i["triangle"]["marker"] for i in train}) >= 2
    assert len({i["triangle"]["depth"] for i in train}) >= 2
    assert any(any(not _axis_intersects(i["triangle"], i["plates"][j]["core"])
                   for j in i["active"]) for i in train), "need an off-axis cascade witness"
    assert any(any(_normalized(i["plates"][j]["cells"]) ==
                       _normalized(i["plates"][k]["cells"])
                   for j in i["active"] for k in i["inactive"]) for i in train), (
        "need an identical unactivated competitor in training")
    # Demonstrate both clean erasure and painting through a removed obstruction.
    extra_values = {int(np.asarray(pair["output"])[p]) == i["background"]
                    for pair, i in zip(puzzle["train"], train)
                    for j in i["active"] for p in i["plates"][j]["extra"]}
    assert extra_values == {False, True}


def _defect_side(plate):
    r0, r1, c0, c1 = plate["bbox"]
    if max(r for r, _ in plate["extra"]) < r0:
        return "up"
    if min(r for r, _ in plate["extra"]) > r1:
        return "down"
    return "left" if max(c for _, c in plate["extra"]) < c0 else "right"


def _network_probe():
    """A hand-specified fork and convergence, beyond the sampler's three relays.

    Expected rays are explicit rectangular slices, not a call to propagation.
    The middle two relays receive different rows of the first broadcast; both
    send rays to the last relay, which also receives bypass rays directly.
    """
    inp = np.zeros((30, 30), dtype=np.int8)
    for p in _triangle_cells((15, 1), (0, 1), 2):
        inp[p] = 4
    inp[15, 0] = 2
    plates = [_plate(12, 5, 6, 3, 0, 0), _plate(8, 12, 6, 3, 3, 2),
              _plate(16, 12, 6, 3, 1, 3), _plate(11, 20, 8, 3, 0, 0),
              _plate(1, 0, 4, 4, 2, 0)]
    for plate in plates:
        for p in plate["cells"]:
            inp[p] = 8
    expected = inp.copy()
    for plate in plates[:4]:
        for p in plate["extra"]:
            expected[p] = 0
    expected[15, 2:5] = 2
    expected[12:18, 5] = 2
    expected[12:18, 8:12] = 2
    expected[14:16, 12:20] = 2
    expected[8:14, 12] = 2
    expected[8:11, 15:30] = 2
    expected[11:14, 15:20] = 2
    expected[16:22, 12] = 2
    expected[16:19, 15:20] = 2
    expected[19:22, 15:30] = 2
    expected[11:19, 20] = 2
    expected[11:19, 23:30] = 2
    assert np.array_equal(solve(inp), expected)
    assert len(_analyze(inp)["active"]) == 4
    assert np.array_equal(solve(np.rot90(inp)), np.rot90(expected))


def self_test():
    """Bounded semantic and diversity checks on 42 fixed seeds; returns metrics."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, palettes, directions, depths, active_counts, inactive_counts = (set() for _ in range(6))
    body_sizes, defects, backgrounds, structures = (set() for _ in range(4))
    off_axis = erased_then_lit = identical = edge_contacts = forks = chains = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _check_pair(pair)
                tri = info["triangle"]
                shapes.add(info["shape"])
                palettes.add((info["background"], tri["body"], info["carrier"], tri["marker"]))
                directions.add((split, tri["direction"]))
                depths.add(tri["depth"])
                active_counts.add(len(info["active"]))
                inactive_counts.add(len(info["inactive"]))
                backgrounds.add(info["background"])
                if len(info["active"]) == 3:
                    is_fork = any(len({b for a, b in info["edges"] if a == j}) >= 2
                                  for j in info["active"])
                    forks += is_fork
                    chains += not is_fork
                roles = {}
                structures.add((info["shape"], tuple(roles.setdefault(int(x), len(roles))
                                                    for x in pair["input"].flat)))
                for j, plate in enumerate(info["plates"]):
                    r0, r1, c0, c1 = plate["bbox"]
                    body_sizes.add((r1 - r0 + 1, c1 - c0 + 1))
                    defects.add(_defect_side(plate))
                    if j in info["active"]:
                        off_axis += not _axis_intersects(tri, plate["core"])
                        erased_then_lit += len(plate["extra"] & info["rays"])
                    else:
                        normalized = _normalized(plate["cells"])
                        identical += any(normalized == _normalized(info["plates"][k]["cells"])
                                         for k in info["active"])
                edge_contacts += any(r in (0, pair["input"].shape[0] - 1)
                                     or c in (0, pair["input"].shape[1] - 1)
                                     for p in info["plates"] for r, c in p["cells"])
    assert len(shapes) >= 45 and len(palettes) >= 100 and len(structures) >= 190
    assert len(directions) == 8 and depths == {2, 3, 4}
    assert active_counts == {1, 2, 3} and inactive_counts == {1, 2, 3}
    assert len(body_sizes) >= 15 and len(defects) == 4 and len(backgrounds) == 10
    assert off_axis >= 100 and erased_then_lit >= 80 and identical >= 70 and edge_contacts >= 10
    assert forks >= 2 and chains >= 2
    _network_probe()
    assert all(_key(a[k]) == _key(b[k]) for split in ("train", "test")
               for a, b in zip(puzzles[17][split], generate_puzzle(17)[split])
               for k in ("input", "output"))
    # Rotation/reflection/color equivariance tests exercise the input-only solver.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        original = pair["input"].copy()
        for k in range(4):
            assert np.array_equal(solve(np.rot90(original, k)), np.rot90(pair["output"], k))
        assert np.array_equal(solve(np.fliplr(original)), np.fliplr(pair["output"]))
        perm = np.array([8, 3, 0, 6, 9, 2, 5, 1, 7, 4], dtype=np.int8)
        assert np.array_equal(solve(perm[original]), perm[pair["output"]])
        assert np.array_equal(original, pair["input"])
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        generate_puzzle(901 + a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    wrong = {s: [dict(p) for p in puzzles[0][s]] for s in ("train", "test")}
    wrong["test"][0]["output"] = wrong["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted omitted relay edits")
    return dict(seeds=42, pairs=210, shapes=len(shapes), palettes=len(palettes),
                color_normalized_structures=len(structures), body_sizes=len(body_sizes),
                directions=len(directions), off_axis_activations=off_axis,
                removed_stem_cells_on_rays=erased_then_lit,
                identical_unreached_competitors=identical, object_edge_contacts=edge_contacts,
                three_plate_forks=forks, three_plate_chains=chains)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
