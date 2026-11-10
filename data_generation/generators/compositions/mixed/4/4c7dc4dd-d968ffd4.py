"""Framed ink-role analogy followed by a full-cross-section gap partition.

No parent code is imported. All decoding, including the active ink role, uses
the supplied input. Axis 0 means top/bottom pads; axis 1 means left/right pads.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"invalid {name}: {value}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _components(mask):
    """Four-neighbor components of a Boolean mask, using bounded grid memory."""
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        point = min(unseen)
        unseen.remove(point)
        stack, points = [point], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for other in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        result.append(points)
    return result


def _bbox(points):
    rows, cols = zip(*points)
    return int(min(rows)), int(min(cols)), int(max(rows))+1, int(max(cols))+1


def _contains(outer, inner):
    return (outer[0] < inner[0] and outer[1] < inner[1]
            and inner[2] < outer[2] and inner[3] < outer[3])


def _disjoint(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def _outline(grid, box, color):
    r0, c0, r1, c1 = box
    grid[r0, c0:c1] = grid[r1-1, c0:c1] = color
    grid[r0:r1, c0] = grid[r0:r1, c1-1] = color


def _interior(grid, box):
    return grid[box[0]+1:box[2]-1, box[1]+1:box[3]-1]


def _scene(input_grid):
    """Recognize outlines by connected geometry, not by literal color labels."""
    grid = _grid(input_grid)
    colored_components = [(int(color), points, _bbox(points))
                          for color in np.unique(grid)
                          for points in _components(grid == color)]
    # Large solid components identify neutral ink: the missing interior and,
    # occasionally, rectangular neutral padding between the two panels.
    blank_components = [(color, box) for color, points, box in colored_components
                        if box[2]-box[0] >= 5 and box[3]-box[1] >= 5
                        and len(points) == (box[2]-box[0])*(box[3]-box[1])]
    neutrals = {color for color, _ in blank_components}
    if len(neutrals) != 1:
        raise ValueError("large blank regions must identify one neutral ink")
    neutral = next(iter(neutrals))
    rectangles = []
    for color, _, box in colored_components:
        if color == neutral or box[2]-box[0] < 7 or box[3]-box[1] < 7:
            continue
        r0, c0, r1, c1 = box
        if (np.all(grid[r0, c0:c1] == color) and np.all(grid[r1-1, c0:c1] == color)
                and np.all(grid[r0:r1, c0] == color) and np.all(grid[r0:r1, c1-1] == color)):
            rectangles.append((*box, color))
    panels = [box for box in rectangles
              if not any(_contains(box, other) for other in rectangles)]
    groups = [(box, [p for p in panels if _contains(box, p)]) for box in rectangles]
    groups = [(box, members) for box, members in groups if len(members) == 2]
    if len(panels) != 4 or len(groups) != 2 or not _disjoint(groups[0][0], groups[1][0]):
        raise ValueError("expected two separate enclosures of two panels")
    if len({(p[2]-p[0], p[3]-p[1]) for p in panels}) != 1:
        raise ValueError("the four interiors must have the same dimensions")
    blanks = [p for p in panels if np.all(_interior(grid, p) == neutral)]
    if len(blanks) != 1 or (neutral, tuple(x+delta for x, delta in zip(
            blanks[0][:4], (1, 1, -1, -1)))) not in blank_components:
        raise ValueError("blank must be the entire missing panel interior")
    missing = blanks[0]
    target_group = next(i for i, (_, ps) in enumerate(groups) if missing in ps)
    target_panels = groups[target_group][1]
    occupied = next(p for p in target_panels if p != missing)
    reference = groups[1-target_group][1]
    if reference[0][4] != reference[1][4] or occupied[4] != missing[4]:
        raise ValueError("each pair must share its panel-frame ink")
    return dict(grid=grid, neutral=neutral, panels=panels, groups=groups,
                reference=reference, occupied=occupied, missing=missing)


def _pads(raw, neutral):
    """Read two solid two-tone footprints and their common facing axis."""
    raw = _grid(raw)
    if not (np.all(raw[0] == neutral) and np.all(raw[-1] == neutral)
            and np.all(raw[:, 0] == neutral) and np.all(raw[:, -1] == neutral)):
        raise ValueError("raw pads need neutral margins on all four sides")
    components = _components(raw != neutral)
    if len(components) != 2:
        raise ValueError("expected exactly two pad footprints")
    boxes = [_bbox(points) for points in components]
    for points, box in zip(components, boxes):
        if len(points) != (box[2]-box[0])*(box[3]-box[1]):
            raise ValueError("pad footprint must be a solid rectangle")
    a, b = boxes
    horizontal = (max(a[0], b[0]) < min(a[2], b[2])
                  and (a[3]+2 <= b[1] or b[3]+2 <= a[1]))
    vertical = (max(a[1], b[1]) < min(a[3], b[3])
                and (a[2]+2 <= b[0] or b[2]+2 <= a[0]))
    if horizontal == vertical:
        raise ValueError("pads must have one unambiguous opposing axis")
    axis = 1 if horizontal else 0
    boxes.sort(key=lambda box: box[axis])
    inks = []
    for r0, c0, r1, c1 in boxes:
        pad = raw[r0:r1, c0:c1]
        oriented = pad if axis == 0 else pad.T
        strip = oriented[0]
        if not np.all(oriented == strip):
            raise ValueError("pad stripes must repeat through its axial depth")
        runs = [int(strip[0])]
        runs.extend(int(strip[i]) for i in range(1, len(strip)) if strip[i] != strip[i-1])
        if len(runs) != 3 or runs[0] != runs[2] or runs[0] == runs[1]:
            raise ValueError("pad must have outer-ink / inner-ink / outer-ink stripes")
        inks.append((runs[0], runs[1]))
    if len(set(inks[0]+inks[1])) != 4:
        raise ValueError("the four pad ink roles must be distinct within a panel")
    start, stop = boxes[0][axis+2], boxes[1][axis]
    region = raw[start:stop, :] if axis == 0 else raw[:, start:stop]
    if stop-start < 2 or not np.all(region == neutral):
        raise ValueError("the entire intervening gap must be neutral")
    return dict(axis=axis, boxes=boxes, inks=inks, start=start, stop=stop,
                gap=stop-start, shape=raw.shape)


def _partition(raw, info, role, separator):
    out = raw.copy()
    oriented = out if info["axis"] == 0 else out.T
    start, stop = info["start"], info["stop"]
    half = (stop-start)//2
    oriented[start:start+half, :] = info["inks"][0][role]
    oriented[stop-half:stop, :] = info["inks"][1][role]
    if (stop-start) % 2:
        oriented[start+half, :] = separator
    return out


def _decode(input_grid):
    scene = _scene(input_grid)
    grid, neutral = scene["grid"], scene["neutral"]
    candidates = []
    for before, after in (scene["reference"], scene["reference"][::-1]):
        raw = _interior(grid, before)
        try:
            info = _pads(raw, neutral)
        except ValueError:
            continue
        if info["gap"] % 2 != 1:
            continue  # Every complete pair explicitly shows its separator.
        for role in (0, 1):
            if np.array_equal(_partition(raw, info, role, before[4]), _interior(grid, after)):
                candidates.append((before, after, info, role))
    if len(candidates) != 1:
        raise ValueError("complete pair must demonstrate one unique ink role and partition")
    before, after, ref_info, role = candidates[0]
    target = _interior(grid, scene["occupied"])
    target_info = _pads(target, neutral)
    scene.update(before=before, after=after, ref_info=ref_info, role=role,
                 target=target, target_info=target_info)
    return scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the missing interior from this input alone; do not mutate it."""
    scene = _decode(input_grid)
    return _partition(scene["target"], scene["target_info"], scene["role"], scene["missing"][4])


def _group_minimum(shape, orientation):
    h, w = shape
    return (h+4, 2*w+7) if orientation == 1 else (2*h+7, w+4)


def _layout_options(shape):
    options = []
    for a in (0, 1):
        for b in (0, 1):
            minima = [_group_minimum(shape, a), _group_minimum(shape, b)]
            for pack in (0, 1):
                dims = [max(x[d] for x in minima) for d in (0, 1)]
                dims[pack] = sum(x[pack] for x in minima)+1
                if max(dims) <= 30:
                    options.append(((a, b), pack, minima, dims))
    return options


def _layout(rng, shape):
    options = _layout_options(shape)
    orientations, pack, minima, dims = options[int(rng.integers(len(options)))]
    canvas = [int(rng.integers(n, 31)) for n in dims]
    extra = rng.multinomial(canvas[pack]-dims[pack], [0.2]*5)
    boxes, positions = [], []
    offset = int(extra[2])
    for i, orientation in enumerate(orientations):
        size = list(minima[i])
        size[pack] += int(extra[i])
        cross = 1-pack
        size[cross] = int(rng.integers(size[cross], canvas[cross]+1))
        origin = [0, 0]
        origin[pack] = offset
        origin[cross] = int(rng.integers(canvas[cross]-size[cross]+1))
        r, c = origin
        boxes.append((r, c, r+size[0], c+size[1]))
        panel_size = [shape[0]+2, shape[1]+2]
        span, depth = size[orientation], panel_size[orientation]
        gap = int(rng.integers(1, span-2*depth-1))
        begin = int(rng.integers(1, span-2*depth-gap))
        group_positions = []
        for along in (begin, begin+depth+gap):
            pos = origin.copy()
            pos[orientation] += along
            pos[1-orientation] += int(rng.integers(1, size[1-orientation]-panel_size[1-orientation]))
            group_positions.append((pos[0], pos[1], pos[0]+panel_size[0], pos[1]+panel_size[1]))
        positions.append(group_positions)
        offset += size[pack]+1+int(extra[3])
    return tuple(canvas), boxes, positions


def _raw_panel(rng, shape, axis, parity, neutral, colors, asymmetric, min_gap=2):
    length, cross = shape[axis], shape[1-axis]
    gaps = [g for g in range(min_gap, length-3) if g % 2 == parity]
    if not gaps:
        raise ValueError("shape cannot contain the requested gap")
    gap = int(rng.choice(gaps))
    spare = length-gap-2
    d0 = int(rng.integers(1, min(2, spare-1)+1))
    d1 = int(rng.integers(1, min(2, spare-d0)+1))
    end_space = length-gap-d0-d1
    e0 = int(rng.integers(1, end_space))
    span0 = int(rng.integers(3, cross-1))
    s0 = int(rng.integers(1, cross-span0))
    if asymmetric:
        possible = [(s, span) for span in range(3, cross-1) for s in range(1, cross-span)
                    if max(s, s0) < min(s+span, s0+span0)]
        s1, span1 = possible[int(rng.integers(len(possible)))]
    else:
        s1, span1 = s0, span0
    panel = np.full(shape, neutral, dtype=np.int8)
    oriented = panel if axis == 0 else panel.T
    for index, (start, depth, side, span) in enumerate(((e0, d0, s0, span0),
                                                      (e0+d0+gap, d1, s1, span1))):
        body, core = colors[2*index:2*index+2]
        strip = np.full(span, body, dtype=np.int8)
        width = int(rng.integers(1, span-1))
        edge = int(rng.integers(1, span-width))
        strip[edge:edge+width] = core
        oriented[start:start+depth, side:side+span] = strip
    return panel


def _make_pair(rng, target_axis, target_parity, role, cross_axis, anchor=False):
    ref_axis = 1-target_axis if cross_axis else target_axis
    minimum_length = (9 if target_parity else 8) if anchor else (7 if target_parity else 6)
    shapes = [(h, w) for h in range(5, 12) for w in range(5, 12)
              if (h, w)[ref_axis] >= 7 and (h, w)[target_axis] >= minimum_length
              and _layout_options((h, w))]
    shape = shapes[int(rng.integers(len(shapes)))]
    palette = list(map(int, rng.permutation(10)))
    neutral, group_ink, ref_frame, target_frame, wallpaper = palette[:5]
    ink_pool = palette[5:]
    ref_colors = list(map(int, rng.choice(ink_pool, 4, replace=False)))
    target_colors = list(map(int, rng.choice(ink_pool, 4, replace=False)))
    # Within each pad the two roles differ. Across groups, literal identity
    # does not establish correspondence: outer/inner stripe geometry does.
    ref = _raw_panel(rng, shape, ref_axis, 1, neutral, ref_colors, bool(rng.integers(2)))
    target = _raw_panel(rng, shape, target_axis, target_parity, neutral, target_colors,
                        cross_axis or bool(rng.integers(2)), 4 if anchor else 2)
    ref_info, target_info = _pads(ref, neutral), _pads(target, neutral)
    complete = _partition(ref, ref_info, role, ref_frame)
    output = _partition(target, target_info, role, target_frame)
    canvas_shape, boxes, panels = _layout(rng, shape)
    rr, cc = np.indices(canvas_shape)
    # Two-color bounded-run camouflage cannot imitate a long frame.
    mode = int(rng.integers(3))
    field = (rr+cc) % 2 if mode == 0 else (rr//2+cc//2) % 2
    grid = np.where(field, wallpaper, neutral).astype(np.int8)
    if mode == 2:
        grid.fill(wallpaper)
        grid[::2, ::2] = neutral
        # Uniform wallpaper reaches the canvas edges, but has no solid large
        # component or rectangular outline enclosing panels.
    ref_group = int(rng.integers(2))
    contents = [None, None]
    ref_contents = [ref, complete]
    target_contents = [target, np.full(shape, neutral, dtype=np.int8)]
    rng.shuffle(ref_contents)
    rng.shuffle(target_contents)
    contents[ref_group], contents[1-ref_group] = ref_contents, target_contents
    for index, (box, local_boxes) in enumerate(zip(boxes, panels)):
        grid[box[0]:box[2], box[1]:box[3]] = neutral
        _outline(grid, box, group_ink)
        frame = ref_frame if index == ref_group else target_frame
        for panel_box, interior in zip(local_boxes, contents[index]):
            _outline(grid, panel_box, frame)
            _interior(grid, panel_box)[:] = interior
    return {"input": grid, "output": output}


def _reference_twin(pair):
    """Keep the target fixed; change only the demonstrated propagation role."""
    scene = _decode(pair["input"])
    grid = pair["input"].copy()
    other_role = 1-scene["role"]
    _interior(grid, scene["after"])[:] = _partition(
        _interior(grid, scene["before"]), scene["ref_info"], other_role, scene["before"][4])
    output = _partition(scene["target"], scene["target_info"], other_role, scene["missing"][4])
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    axis, parity, role = map(int, rng.integers(2, size=3))
    plan = [(axis, parity, role), (1-axis, 1-parity, 1-role)]
    if nt >= 3:
        plan.append(None)  # A matched demonstration refutes target-only choice.
    plan += [tuple(map(int, rng.integers(2, size=3))) for _ in range(max(0, nt-3))]
    test_plan = [tuple(map(int, rng.integers(2, size=3))) for _ in range(ns)]
    # A small-count test reuses a taught axis/parity combination. Its novel
    # transfer is across the perpendicular reference/target pad axes.
    if nt == 2:
        test_plan = [plan[int(rng.integers(2))] for _ in range(ns)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, items in (("train", plan), ("test", test_plan)):
        for index, item in enumerate(items):
            if item is None:
                source = puzzle["train"][1-parity]
                pair = _reference_twin(source)
                assert _key(pair["input"]) not in seen
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                continue
            target_axis, target_parity, active_role = item
            for _ in range(32):
                pair = _make_pair(rng, target_axis, target_parity, active_role,
                                  split == "test", split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    return puzzle


def _check_partition(raw, result, info, role, separator, neutral):
    """Cellwise distance check, including untouched pad and margin cells."""
    assert raw.shape == result.shape
    start, stop, axis = info["start"], info["stop"], info["axis"]
    changed = 0
    for r, c in np.ndindex(raw.shape):
        position = (r, c)[axis]
        if start <= position < stop:
            assert raw[r, c] == neutral
            distances = (position-start+1, stop-position)
            expected = (separator if distances[0] == distances[1]
                        else info["inks"][int(distances[1] < distances[0])][role])
            changed += 1
        else:
            expected = raw[r, c]
        assert result[r, c] == expected, "incorrect full-slice partition or altered pad"
    assert np.count_nonzero(raw != result) == changed == info["gap"]*raw.shape[1-axis]
    assert all(np.all(result[box[0]:box[2], box[1]:box[3]]
                         == raw[box[0]:box[2], box[1]:box[3]]) for box in info["boxes"])
    # Painting only the shared transverse pad span leaves these margin cells
    # untouched and is therefore observably different from the intended rule.
    oriented = result if axis == 0 else result.T
    assert oriented[start, 0] != neutral and oriented[start, -1] != neutral


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _decode(inp)
            assert out.shape == scene["target"].shape
            _check_partition(scene["target"], out, scene["target_info"], scene["role"],
                             scene["missing"][4], scene["neutral"])
            _check_partition(_interior(inp, scene["before"]), _interior(inp, scene["after"]),
                             scene["ref_info"], scene["role"], scene["before"][4], scene["neutral"])
            assert np.array_equal(solve(inp), out)
            # Removing A's choice changes both halves, for every generated pair.
            other = _partition(scene["target"], scene["target_info"], 1-scene["role"], scene["missing"][4])
            assert not np.array_equal(other, out)
            analyses[split].append(scene)
    train = analyses["train"]
    assert {s["target_info"]["axis"] for s in train} == {0, 1}
    assert {s["target_info"]["gap"] % 2 for s in train} == {0, 1}
    assert {s["role"] for s in train} == {0, 1}
    assert max(s["target_info"]["gap"] for s in train) >= 4
    assert all(s["ref_info"]["axis"] == s["target_info"]["axis"] for s in train)
    assert all(s["ref_info"]["axis"] != s["target_info"]["axis"] for s in analyses["test"])
    if nt >= 3:
        source_index = next(i for i in (0, 1) if train[i]["target_info"]["gap"] % 2)
        expected_twin = _reference_twin(puzzle["train"][source_index])
        assert all(np.array_equal(puzzle["train"][2][k], expected_twin[k])
                   for k in ("input", "output"))
        assert np.array_equal(train[2]["target"], train[source_index]["target"])
        assert not np.array_equal(puzzle["train"][2]["output"],
                                  puzzle["train"][source_index]["output"])
    if nt == 2:
        taught = {(s["target_info"]["axis"], s["target_info"]["gap"] % 2) for s in train}
        assert all((s["target_info"]["axis"], s["target_info"]["gap"] % 2) in taught
                   for s in analyses["test"])


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _diagnostics():
    puzzle = generate_puzzle(0)
    pair = next(p for p in puzzle["train"] if _decode(p["input"])["target_info"]["gap"] % 2)
    scene = _decode(pair["input"])
    original = pair["input"].copy()
    # Hold every target pixel and all layout fixed. Change ONLY the complete
    # reference partner to demonstrate the other legitimate ink role.
    twin = original.copy()
    _interior(twin, scene["after"])[:] = _partition(
        _interior(original, scene["before"]), scene["ref_info"], 1-scene["role"], scene["before"][4])
    other = solve(twin)
    assert not np.array_equal(other, pair["output"])
    assert np.array_equal(_decode(twin)["target"], scene["target"])
    # The odd center is the target's frame ink, not neutral or reference ink.
    info = scene["target_info"]
    result = solve(original)
    oriented = result if info["axis"] == 0 else result.T
    center = info["start"]+info["gap"]//2
    assert np.all(oriented[center] == scene["missing"][4])
    assert scene["missing"][4] not in (scene["neutral"], scene["before"][4])
    for transform in (lambda x: x.T.copy(), lambda x: x[::-1].copy(), lambda x: x[:, ::-1].copy()):
        assert np.array_equal(solve(transform(original)), transform(result))
    palette = np.random.default_rng(111).permutation(10).astype(np.int8)
    assert np.array_equal(solve(palette[original]), palette[result])
    assert np.array_equal(original, pair["input"])
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["train"][0]["output"][0, 0] = (int(damaged["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output was accepted")


def self_test():
    """42 sequential seeds, hashes/statistics only; no files or retained runs."""
    signatures, shapes, normalized, palettes, gaps, geometries, slots = (set() for _ in range(7))
    role_cases, test_cases, layouts, stripe_widths, pad_depths = (set() for _ in range(5))
    asymmetric = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _decode(pair["input"])
                info = scene["target_info"]
                shapes.add(tuple(info["shape"]))
                role_cases.add((info["axis"], info["gap"] % 2, scene["role"]))
                if split == "test":
                    test_cases.add((info["axis"], info["gap"] % 2))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                gaps.add(info["gap"])
                geometries.add((info["axis"], info["shape"], tuple(info["boxes"])))
                slots.add((tuple(scene["before"][:2]) < tuple(scene["after"][:2]),
                           tuple(scene["occupied"][:2]) < tuple(scene["missing"][:2]),
                           tuple(scene["reference"][0][:2]) < tuple(scene["occupied"][:2])))
                layouts.add(tuple((b[2]-b[0], b[3]-b[1]) for b, _ in scene["groups"]))
                mapping = {}
                norm = np.array([mapping.setdefault(int(v), len(mapping)) for v in pair["output"].flat], dtype=np.uint8)
                normalized.add((info["shape"], norm.tobytes()))
                a, b = info["boxes"]
                transverse = 1-info["axis"]
                asymmetric += (a[transverse], a[transverse+2]) != (b[transverse], b[transverse+2])
                for box, inks in zip(info["boxes"], info["inks"]):
                    pad = scene["target"][box[0]:box[2], box[1]:box[3]]
                    strip = pad[0] if info["axis"] == 0 else pad[:, 0]
                    stripe_widths.add(int(np.count_nonzero(strip == inks[1])))
                    pad_depths.add(box[info["axis"]+2]-box[info["axis"]])
        if seed in (0, 7, 41):
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 15 and len(normalized) >= 150 and len(geometries) >= 150
    assert len(palettes) >= 8 and len(layouts) >= 30 and len(slots) == 8
    assert role_cases == {(a, p, r) for a in (0, 1) for p in (0, 1) for r in (0, 1)}
    assert test_cases == {(a, p) for a in (0, 1) for p in (0, 1)}
    assert gaps >= {2, 3, 4, 5, 6} and pad_depths == {1, 2}
    assert stripe_widths >= {1, 2, 3} and asymmetric >= 20
    for nt in range(2, 11):
        for ns in range(1, min(nt-1, 4)+1):
            validate_generated_puzzle(generate_puzzle(124, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _diagnostics()
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), output_shapes=len(shapes),
                normalized_outputs=len(normalized), pad_geometries=len(geometries),
                input_palettes=len(palettes), enclosure_size_pairs=len(layouts),
                asymmetric_targets=asymmetric, gaps=sorted(gaps), stripe_widths=sorted(stripe_widths),
                all_legal_counts_checked=True, reference_counterfactual=True,
                independent_semantic_verification=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
