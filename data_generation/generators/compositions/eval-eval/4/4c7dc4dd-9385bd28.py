"""Transfer a locally demonstrated rectangle stack to a new arrangement.

The finished reference encodes its own palette and precedence graph. No literal
color identifies a role, and solve reads no examples, files, or random state.
Only standard-library modules and NumPy are used. Diagnostics are streamed;
self_test writes no files and retains only bounded feature/signature sets.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()).hexdigest()


def _markers(box, diagonal):
    r, c, b, d = box
    if diagonal == 0:
        return {(r, c), (r, c+1), (r+1, c), (b, d), (b, d-1), (b-1, d)}
    return {(r, d), (r, d-1), (r+1, d), (b, c), (b, c+1), (b-1, c)}


def _recognize(points):
    points = set(points)
    if len(points) != 6:
        raise ValueError("each cue must have exactly two intact L triplets")
    rows, cols = zip(*points)
    box = (min(rows), min(cols), max(rows), max(cols))
    h, w = box[2]-box[0]+1, box[3]-box[1]+1
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        raise ValueError("ambiguous or degenerate corner fragments")
    for diagonal in (0, 1):
        if points == _markers(box, diagonal):
            return dict(box=box, markers=points, diagonal=diagonal)
    raise ValueError("cue cells do not form opposite L corners")


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _inside(point, box):
    r, c = point
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _cue_objects(panel, bg):
    colors = sorted(set(map(int, np.unique(panel))) - {bg})
    if not 2 <= len(colors) <= 4:
        raise ValueError("a schematic contains 2..4 cue colors")
    result = {}
    for color in colors:
        obj = _recognize(map(tuple, np.argwhere(panel == color)))
        result[color] = dict(obj, cue=color)
    return result


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = remaining.pop()
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    stack.append(point)
        result.append(component)
    return result


def _frame_boxes(grid, bg):
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        components = _components(grid == color)
        if len(components) != 2:
            continue
        boxes = []
        for points in components:
            rows, cols = zip(*points)
            r, c, b, d = min(rows), min(cols), max(rows), max(cols)
            if (b-r < 4 or d-c < 4 or len(points) != 2*(b-r+d-c)
                    or any(y not in (r, b) and x not in (c, d) for y, x in points)):
                break
            boxes.append((r, c, b, d))
        if len(boxes) == 2 and not _overlap(*boxes):
            outside = np.ones(grid.shape, dtype=bool)
            for r, c, b, d in boxes:
                outside[r:b+1, c:d+1] = False
            if np.all(grid[outside] == bg):
                candidates.append((color, sorted(boxes)))
    if len(candidates) != 1:
        raise ValueError("expected two uniquely identifiable panel frames")
    return candidates[0]


def _unique_order(cues, edges):
    remaining, order = set(cues), []
    while remaining:
        ready = [x for x in remaining if not any((y, x) in edges for y in remaining)]
        if len(ready) != 1:
            raise ValueError("reference must determine a unique complete layer order")
        order.append(ready[0])
        remaining.remove(ready[0])
    return order


def _reference_data(panel, cues, bg):
    objects = {}
    for cue in cues:
        obj = _recognize(map(tuple, np.argwhere(panel == cue)))
        objects[cue] = dict(obj, cue=cue)
    covers = {}
    for point in np.ndindex(panel.shape):
        covers[point] = [cue for cue, obj in objects.items() if _inside(point, obj["box"])]
    for cue, obj in objects.items():
        exclusive = [int(panel[p]) for p, stack in covers.items()
                     if stack == [cue] and p not in obj["markers"]]
        if not exclusive or len(set(exclusive)) != 1 or exclusive[0] in objects:
            raise ValueError("every layer needs an unambiguous exposed fill sample")
        obj["fill"] = exclusive[0]
    if len({o["fill"] for o in objects.values()}) != len(objects):
        raise ValueError("reference fills must be distinct")
    edges = set()
    for point, stack in covers.items():
        value = int(panel[point])
        if not stack:
            if value != bg:
                raise ValueError("unexplained reference pixels")
            continue
        owners = [cue for cue in stack if value == (
            cue if point in objects[cue]["markers"] else objects[cue]["fill"])]
        if len(owners) != 1:
            raise ValueError("unreadable reference overlap")
        edges.update((owners[0], other) for other in stack if other != owners[0])
    order = _unique_order(objects, edges)
    return dict(objects=objects, order=order, edges=edges, panel=panel, background=bg)


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])  # The family requires an exterior one-cell margin.
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("missing uniform exterior margin")
    frame, boxes = _frame_boxes(grid, bg)
    panels = [grid[r+1:b, c+1:d].copy() for r, c, b, d in boxes]
    interpretations = []
    for target_index in (0, 1):
        try:
            target = _cue_objects(panels[target_index], bg)
            reference = _reference_data(panels[1-target_index], target, bg)
            if frame in set(np.unique(panels[0])) | set(np.unique(panels[1])):
                continue
            interpretations.append(dict(reference=reference, target=target,
                                        schematic=panels[target_index], boxes=boxes,
                                        target_index=target_index, frame=frame, background=bg))
        except ValueError:
            continue
    if len(interpretations) != 1:
        raise ValueError("reference and schematic are not uniquely readable")
    return interpretations[0]


def _predict(target, shape, reference, order=None, transparent=False):
    """Read each pixel's first covering layer; separate from the compositor."""
    bg = reference["background"]
    order = reference["order"] if order is None else order
    out = np.full(shape, bg, dtype=np.int8)
    for point in np.ndindex(shape):
        for cue in order:
            obj = target[cue]
            if _inside(point, obj["box"]):
                value = cue if point in obj["markers"] else reference["objects"][cue]["fill"]
                if transparent and value == bg:
                    continue
                out[point] = value
                break
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer palette and precedence exclusively from the finished input sample."""
    scene = _read(input_grid)
    return _predict(scene["target"], scene["schematic"].shape, scene["reference"])


def _compose(shape, objects, order, fills, bg):
    """Construction path: overwrite complete boxes from back to front."""
    out = np.full(shape, bg, dtype=np.int8)
    for cue in reversed(order):
        obj = objects[cue]
        r, c, b, d = obj["box"]
        out[r:b+1, c:d+1] = fills[cue]
        for point in obj["markers"]:
            out[point] = cue
    return out


def _witnesses(reference, target, shape, out):
    order, bg = reference["order"], reference["background"]
    result = set()
    for point in np.ndindex(shape):
        stack = [x for x in order if _inside(point, target[x]["box"])]
        if not stack:
            continue
        top = stack[0]
        if point not in target[top]["markers"] and int(out[point]) != bg:
            result.add("fill")
        if len(stack) >= 2:
            other = stack[1]
            below = other if point in target[other]["markers"] else reference["objects"][other]["fill"]
            if int(out[point]) != below:
                result.add("priority")
                if int(out[point]) == bg and point not in target[top]["markers"]:
                    result.add("clearing")
            if (len(stack) == 2 and point not in target[top]["markers"]
                    and point not in target[other]["markers"]
                    and not _overlap(reference["objects"][top]["box"],
                                     reference["objects"][other]["box"])):
                result.add("transitive")
        if any(point in target[x]["markers"] and int(out[point]) != x for x in stack):
            result.add("occlusion")
    if not np.array_equal(out, _predict(target, shape, reference, list(reversed(order)))):
        result.add("reverse_differs")
    if not np.array_equal(out, _predict(target, shape, reference, transparent=True)):
        result.add("transparent_differs")
    return result


def _shift_objects(objects, dr, dc):
    return {cue: dict(o, box=(o["box"][0]+dr, o["box"][1]+dc,
                             o["box"][2]+dr, o["box"][3]+dc),
                       markers={(r+dr, c+dc) for r, c in o["markers"]})
            for cue, o in objects.items()}


def _reference_panel(rng, cues, fills, bg, chain_only):
    n = len(cues)
    for attempt in range(160):
        chain = chain_only or attempt >= 120 or rng.random() < .60
        objects = {}
        if chain:
            sizes = [(int(rng.integers(3, 6)), int(rng.integers(3, 6))) for _ in cues]
            r = c = 0
            for i, (cue, (h, w)) in enumerate(zip(cues, sizes)):
                if i:
                    ph, pw = sizes[i-1]
                    r += ph-int(rng.integers(1, min(2, ph-2, h-2)+1))
                    c += pw-int(rng.integers(1, min(2, pw-2, w-2)+1))
                box = (r, c, r+h-1, c+w-1)
                objects[cue] = dict(box=box, markers=_markers(box, 1), cue=cue)
            order = list(cues) if rng.integers(2) else list(reversed(cues))
            shape = (r+sizes[-1][0], c+sizes[-1][1])
            if max(shape) > 11 or any(_overlap(objects[cues[i]]["box"], objects[cues[j]]["box"])
                                      for i in range(n) for j in range(i+2, n)):
                continue
        else:
            shape = (int(rng.integers(7, 12)), int(rng.integers(7, 12)))
            for cue in cues:
                h, w = int(rng.integers(3, shape[0]+1)), int(rng.integers(3, shape[1]+1))
                r, c = int(rng.integers(shape[0]-h+1)), int(rng.integers(shape[1]-w+1))
                box = (r, c, r+h-1, c+w-1)
                objects[cue] = dict(box=box, markers=_markers(box, int(rng.integers(2))), cue=cue)
            order = [cues[int(i)] for i in rng.permutation(n)]
        cells = [p for obj in objects.values() for p in obj["markers"]]
        if len(set(cells)) != 6*n:
            continue
        panel = _compose(shape, objects, order, fills, bg)
        if any(panel[p] != cue for cue, obj in objects.items() for p in obj["markers"]):
            continue
        try:
            recovered = _reference_data(panel, cues, bg)
        except ValueError:
            continue
        if recovered["order"] != order:
            continue
        break
    else:
        # Guaranteed path with pairwise 2x2 overlaps and exposed anti-diagonal Ls.
        objects = {}
        for i, cue in enumerate(cues):
            box = (2*i, 2*i, 2*i+3, 2*i+3)
            objects[cue] = dict(box=box, markers=_markers(box, 1), cue=cue)
        order = list(cues) if rng.integers(2) else list(reversed(cues))
        panel = _compose((2*n+2, 2*n+2), objects, order, fills, bg)
    # Orientation and padding have no relation to which layer is in front.
    if rng.integers(2):
        panel = panel[::-1].copy()
    if rng.integers(2):
        panel = panel[:, ::-1].copy()
    if rng.integers(2):
        panel = panel.T.copy()
    padding = [(int(rng.integers(2)), int(rng.integers(2))) for _ in range(2)]
    padding = [(a, min(b, 12-panel.shape[i]-a)) if panel.shape[i]+a <= 12 else (0, 0)
               for i, (a, b) in enumerate(padding)]
    panel = np.pad(panel, padding, constant_values=bg)
    return panel, _reference_data(panel, cues, bg)


def _random_target(rng, cues, shape):
    objects, used = {}, set()
    for cue in cues:
        for _ in range(50):
            h, w = int(rng.integers(3, shape[0]+1)), int(rng.integers(3, shape[1]+1))
            if rng.random() < .22:
                if rng.integers(2):
                    h, w = 2, max(4, w)
                else:
                    h, w = max(4, h), 2
            r, c = int(rng.integers(shape[0]-h+1)), int(rng.integers(shape[1]-w+1))
            box = (r, c, r+h-1, c+w-1)
            markers = _markers(box, int(rng.integers(2)))
            if markers & used or any(o["box"] == box for o in objects.values()):
                continue
            objects[cue] = dict(box=box, markers=markers, cue=cue)
            used.update(markers)
            break
        else:
            return None
    return objects


def _fallback_target(reference):
    """A readable overlap with a transitive comparison whenever n >= 3."""
    order, bg = reference["order"], reference["background"]
    boxes = [(0, 0, 5, 5), (0, 7, 3, 10), (3, 3, 8, 8)]
    if len(order) == 2:
        boxes = [boxes[0], boxes[-1]]
    elif len(order) == 4:
        boxes.insert(2, (7, 0, 10, 3))
    objects = {cue: dict(cue=cue, box=box, markers=_markers(box, 0 if i == len(order)-1 else 1))
               for i, (cue, box) in enumerate(zip(order, boxes))}
    shape = (9 if len(order) < 4 else 11, 11)
    fills = {c: o["fill"] for c, o in reference["objects"].items()}
    return objects, shape, _compose(shape, objects, order, fills, bg)


def _target_panel(rng, reference, required):
    cues, bg = list(reference["objects"]), reference["background"]
    fills = {c: o["fill"] for c, o in reference["objects"].items()}
    for _ in range(240):
        shape = (int(rng.integers(7, 12)), int(rng.integers(7, 14)))
        objects = _random_target(rng, cues, shape)
        if objects is None:
            continue
        out = _compose(shape, objects, reference["order"], fills, bg)
        if required <= _witnesses(reference, objects, shape, out):
            break
    else:
        objects, shape, out = _fallback_target(reference)
        if not required <= _witnesses(reference, objects, shape, out):
            raise RuntimeError("bounded target construction failed")
    panel = np.full(shape, bg, dtype=np.int8)
    for cue, obj in objects.items():
        for point in obj["markers"]:
            panel[point] = cue
    if rng.integers(2):
        panel, out = panel.T.copy(), out.T.copy()
    return panel, out


def _pack(rng, reference, target, bg, frame):
    panels = [reference, target]
    # Pick either axis, retaining a guaranteed one-cell exterior margin and gap.
    axes = [axis for axis in (0, 1) if sum(p.shape[axis] for p in panels) <= 23]
    if not axes:
        raise ValueError("panels do not fit")
    axis = axes[int(rng.integers(len(axes)))]
    cross = 1-axis
    gap = int(rng.integers(1, min(3, 24-sum(p.shape[axis] for p in panels))+1))
    minimum = [0, 0]
    minimum[axis] = sum(p.shape[axis]+2 for p in panels)+gap+2
    minimum[cross] = max(p.shape[cross] for p in panels)+4
    shape = tuple(int(rng.integers(v, 31)) for v in minimum)
    grid = np.full(shape, bg, dtype=np.int8)
    ordering = [0, 1] if rng.integers(2) else [1, 0]
    start = int(rng.integers(1, shape[axis]-minimum[axis]+2))
    for index in ordering:
        panel = panels[index]
        pos = [0, 0]
        pos[axis] = start
        pos[cross] = int(rng.integers(1, shape[cross]-panel.shape[cross]-2))
        r, c = pos
        h, w = panel.shape
        grid[r:r+h+2, c:c+w+2] = frame
        grid[r+1:r+h+1, c+1:c+w+1] = panel
        start += panel.shape[axis]+2+gap
    return grid


def _make_pair(rng, n, teaching=False, transitive=False):
    palette = list(map(int, rng.permutation(10)))
    bg, frame = palette[:2]
    cues = palette[2:2+n]
    fills = dict(zip(cues, palette[2+n:2+2*n]))
    erase = teaching or rng.random() < .40
    # First construct the geometry/order, then choose an eraser's rank freely.
    reference, data = _reference_panel(rng, cues, fills, bg, teaching or transitive)
    if erase:
        cue = data["order"][0] if teaching else data["order"][int(rng.integers(n))]
        old = fills[cue]
        reference[reference == old] = bg
        fills[cue] = bg
        data = _reference_data(reference, cues, bg)
    required = {"fill", "priority", "reverse_differs"}
    if teaching:
        required.update({"transitive", "occlusion", "clearing", "transparent_differs"})
    elif transitive:
        required.add("transitive")
    target, out = _target_panel(rng, data, required)
    grid = _pack(rng, reference, target, bg, frame)
    return {"input": grid, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples with a longer precedence chain in the test."""
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            n = (3 if index == 0 else int(rng.integers(2, 4))) if split == "train" else (
                4 if index == 0 else int(rng.integers(3, 5)))
            for _ in range(20):
                pair = _make_pair(rng, n, teaching=split == "train" and index == 0,
                                  transitive=split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            scene = _read(inp)
            reference, target = scene["reference"], scene["target"]
            shape = scene["schematic"].shape
            assert out.shape == shape and np.array_equal(out, _predict(target, shape, reference))
            assert not np.array_equal(out, scene["schematic"]), "uncompleted target"
            # Re-render the reference using the pixel-derived order and fills.
            assert np.array_equal(reference["panel"], _predict(reference["objects"],
                                  reference["panel"].shape, reference)), "inconsistent sample"
            witnesses = _witnesses(reference, target, shape, out)
            assert {"fill", "priority", "reverse_differs"} <= witnesses
            if split == "train":
                training.update(witnesses)
            else:
                assert "transitive" in witnesses, "test lacks the intended relational shift"
    assert {"transitive", "occlusion", "clearing", "transparent_differs", "reverse_differs"} <= training


def _diagnostics():
    # Exact target twins: only the worked sample's overlap order changes.
    cues, bg = [1, 2, 3], 0
    fills = {1: 4, 2: 5, 3: 6}
    ref_objects = {}
    for i, cue in enumerate(cues):
        box = (2*i, 2*i, 2*i+3, 2*i+3)
        ref_objects[cue] = dict(cue=cue, box=box, markers=_markers(box, 1))
    boxes = [(0, 0, 5, 5), (0, 7, 3, 10), (3, 3, 8, 8)]
    target_objects = {cue: dict(cue=cue, box=box, markers=_markers(box, 0 if i == 2 else 1))
                      for i, (cue, box) in enumerate(zip(cues, boxes))}
    target = np.zeros((9, 11), dtype=np.int8)
    for cue, obj in target_objects.items():
        for point in obj["markers"]:
            target[point] = cue
    answers, grids = [], []
    for order in (cues, list(reversed(cues))):
        ref = _compose((8, 8), ref_objects, order, fills, bg)
        grid = _pack(np.random.default_rng(81), ref, target, bg, 9)
        grids.append(grid)
        answers.append(solve(grid))
    assert grids[0].shape == grids[1].shape
    assert not np.array_equal(*answers), "reference can be bypassed"
    assert answers[0][4, 4] == 4 and answers[1][4, 4] == 6
    assert not _overlap(ref_objects[1]["box"], ref_objects[3]["box"])
    assert answers[0][3, 3] != target[3, 3], "lower marker must be occludable"
    # A changed fill, with target and order fixed, also changes the answer.
    modified = grids[0].copy()
    modified[modified == 4] = 7
    assert not np.array_equal(solve(modified), answers[0])
    # Opaque background fill clears the lower layer, rather than exposing it.
    erased = grids[0].copy()
    erased[erased == 4] = 0
    assert solve(erased)[4, 4] == 0
    # Output geometry follows the target, never a copy of the reference.
    assert answers[0].shape == target.shape != (8, 8)
    thin = _markers((0, 0, 1, 5), 0)
    assert _recognize(thin)["box"] == (0, 0, 1, 5)
    # Exercise last-resort geometry explicitly, including the two-layer case.
    for n in (2, 3, 4):
        local_cues = list(range(1, n+1))
        local_fills = {cue: 0 if cue == 1 else cue+4 for cue in local_cues}
        local_objects = {}
        for i, cue in enumerate(local_cues):
            box = (2*i, 2*i, 2*i+3, 2*i+3)
            local_objects[cue] = dict(cue=cue, box=box, markers=_markers(box, 1))
        ref = _compose((2*n+2, 2*n+2), local_objects, local_cues, local_fills, 0)
        data = _reference_data(ref, local_cues, 0)
        objects, shape, out = _fallback_target(data)
        witnesses = _witnesses(data, objects, shape, out)
        required = {"fill", "priority", "reverse_differs", "occlusion", "clearing"}
        if n >= 3:
            required.add("transitive")
        assert required <= witnesses
    return 5


def self_test():
    """Bounded 42-seed semantic/variation sweep, without saved puzzle payloads."""
    signatures, shapes, ref_shapes, layouts, palettes, geometries = (set() for _ in range(6))
    layers, diagonals, topologies, fills_at, frame_positions = (set() for _ in range(5))
    witness_counts = {}
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_digest(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                scene = _read(pair["input"])
                ref, target = scene["reference"], scene["target"]
                shapes.add(pair["input"].shape)
                ref_shapes.add(ref["panel"].shape)
                layouts.add((scene["target_index"], tuple(scene["boxes"])))
                palettes.add((scene["background"], scene["frame"], tuple(ref["order"])))
                layers.add(len(target))
                geometries.add(tuple(target[c]["box"] for c in ref["order"]))
                diagonals.update(o["diagonal"] for o in target.values())
                edges = sum(_overlap(x["box"], y["box"]) for i, x in enumerate(target.values())
                            for y in list(target.values())[i+1:])
                topologies.add((len(target), edges))
                fills_at.update(i for i, c in enumerate(ref["order"])
                                if ref["objects"][c]["fill"] == scene["background"])
                frame_positions.update((box[0], box[1]) for box in scene["boxes"])
                for name in _witnesses(ref, target, pair["output"].shape, pair["output"]):
                    witness_counts[name] = witness_counts.get(name, 0)+1
    assert len(signatures) == 42 and layers == {2, 3, 4} and diagonals == {0, 1}
    assert len(shapes) >= 25 and len(ref_shapes) >= 15 and len(geometries) >= 150
    assert len(layouts) >= 150 and len(palettes) >= 100 and len(topologies) >= 5
    assert fills_at == {0, 1, 2, 3} and len(frame_positions) >= 30
    for name in ("transitive", "occlusion", "clearing", "transparent_differs", "reverse_differs"):
        assert witness_counts.get(name, 0) >= 42, name
    first, again = generate_puzzle(518398225, 7, 3), generate_puzzle(518398225, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], again[s])
               for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1200+a*10+b, a, b)
            validate_generated_puzzle(p)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(15, 2, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    _diagnostics()
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "reference_shapes": len(ref_shapes),
            "target_geometries": len(geometries), "layouts": len(layouts),
            "palettes": len(palettes), "topologies": len(topologies),
            "witness_pairs": witness_counts, "reference_dependence_twins": True}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
