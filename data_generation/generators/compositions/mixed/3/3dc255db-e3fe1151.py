"""Four panel inventories repair missing frame payloads before unary export.

Only standard library and NumPy. No saved examples, RNG, or generation records
are consulted by solve. Generation paints its labels from sampled objects;
recognition reads monochromatic components and their directional cross-sections.
No traces or per-seed payloads are written by this module.
"""
from itertools import product
import hashlib
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Left/right reach, extra tip stem, and parallel tail rows.
PROFILES = ((2, 2, 0, 0), (2, 2, 1, 0), (2, 2, 0, 1),
            (1, 2, 0, 1), (2, 1, 0, 1), (1, 1, 0, 3))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "bad grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)),
             "grid must contain integer ARC colors")
    return g.astype(np.int8, copy=True)


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in product((-1, 0, 1), repeat=2):
                if dr == dc == 0:
                    continue
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _canonical(p, direction):
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    r, c = p
    return ((r, c), (c, -r), (-r, -c), (-c, r))[direction]


def _read_frame(cells):
    """Unique terminal tip, optional stem, and thin monotone open arms."""
    candidates = []
    for direction in range(4):
        local = {_canonical(p, direction) for p in cells}
        r0, r1 = min(r for r, _ in local), max(r for r, _ in local)
        rows = [sorted(c for r, c in local if r == level) for level in range(r0, r1 + 1)]
        if len(rows) < 3 or len(rows[0]) != 1:
            continue
        center = rows[0][0]
        lo = hi = center
        split, cavity, valid = False, set(), True
        for k, row in enumerate(rows[1:], 1):
            if row == [center] and not split:
                continue
            if len(row) != 2 or row[0] >= row[1]:
                valid = False
                break
            left, right = row
            if (left > lo or right < hi or lo - left > 1 or right - hi > 1
                    or not left < center < right):
                valid = False
                break
            lo, hi, split = left, right, True
            cavity.update(_world((r0 + k, c), direction) for c in range(left + 1, right))
        if valid and split and cavity:
            candidates.append(dict(frame=set(cells), cavity=cavity,
                                   tip=_world((r0, center), direction), direction=direction))
    _require(len(candidates) == 1, "frame has no unique pointed direction")
    return candidates[0]


def _ray(shape, tip, direction, quantity=None):
    dr, dc = DIRECTIONS[direction]
    r, c = tip
    cells = []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return cells
        cells.append((r, c))
        if quantity is not None and len(cells) >= quantity:
            return cells


def _panel_slices(shape):
    h, w = shape[0] // 2, shape[1] // 2
    return ((slice(0, h), slice(0, w)), (slice(0, h), slice(w + 1, None)),
            (slice(h + 1, None), slice(0, w)), (slice(h + 1, None), slice(w + 1, None)))


def _permute_panels(grid, order):
    """Move whole chambers without rotating their local geometry."""
    result = grid.copy()
    slices = _panel_slices(grid.shape)
    for destination, source in enumerate(order):
        result[slices[destination]] = grid[slices[int(source)]]
    return result


def _identify_frames(panel, structural):
    """Select the frame color by geometry and complete ownership of contents."""
    candidates = []
    for color in sorted(set(int(c) for c in panel.flat) - {0, structural}):
        components = _components(map(tuple, np.argwhere(panel == color)))
        if not 2 <= len(components) <= 4:
            continue
        try:
            objects = [_read_frame(cells) for cells in components]
            occupied, contents = set(), set()
            for obj in objects:
                rs, cs = zip(*obj["frame"])
                box = {(r, c) for r in range(min(rs), max(rs) + 1)
                       for c in range(min(cs), max(cs) + 1)}
                _require(not (box & occupied), "frame boxes overlap")
                occupied.update(box)
                _require(not (obj["cavity"] & contents), "cavities overlap")
                contents.update(obj["cavity"])
                vals = {int(panel[p]) for p in obj["cavity"]} - {0, structural}
                _require(color not in vals and len(vals) <= 1, "nonmonochrome frame payload")
                _require(any(panel[p] != 0 for p in obj["cavity"]), "empty input frame")
                obj["frame_color"] = color
            payload_cells = set(map(tuple, np.argwhere((panel != 0) & (panel != color))))
            _require(payload_cells <= contents, "nonframe marks outside cavities")
            candidates.append(objects)
        except ValueError:
            continue
    _require(len(candidates) == 1, "no unique frame-color interpretation")
    return candidates[0]


def _analyze(value):
    inp = _array(value)
    h, w = inp.shape
    _require(h % 2 == 1 and w % 2 == 1, "equal panels require odd dimensions")
    structural = int(inp[h // 2, w // 2])
    _require(structural != 0 and np.all(inp[h // 2] == structural)
             and np.all(inp[:, w // 2] == structural), "missing nonzero central cross")
    panels, observations, objects, holes = [], [], [], []
    frame_color, frame_size, count = None, None, None
    for rs, cs in _panel_slices(inp.shape):
        panel = inp[rs, cs].copy()
        objs = _identify_frames(panel, structural)
        fc = objs[0]["frame_color"]
        size = sum(len(o["frame"]) for o in objs)
        if frame_color is None:
            frame_color, frame_size, count = fc, size, len(objs)
        _require((fc, size, len(objs)) == (frame_color, frame_size, count),
                 "panel frame color/count/area must agree")
        positions = list(map(tuple, np.argwhere(panel == structural)))
        _require(len(positions) == 1, "each panel needs one placeholder")
        hole = positions[0]
        owners = [o for o in objs if hole in o["cavity"]]
        _require(len(owners) == 1, "placeholder needs a unique frame owner")
        observed = np.bincount(panel.ravel().astype(int), minlength=10)
        observed[[0, structural, fc]] = 0
        panels.append(panel)
        observations.append(observed)
        objects.append(objs)
        holes.append(hole)
    # Since each panel has one missing payload, the uncorrupted inventory
    # size is visible: observed total + 1. No hidden target size is needed.
    totals = [int(hist.sum()) + 1 for hist in observations]
    _require(len(set(totals)) == 1, "different inventory sizes")
    target = np.maximum.reduce(observations)
    _require(int(target.sum()) == totals[0], "common multiset not uniquely determined")
    repaired, missing = [], []
    for panel, observed, objs, hole in zip(panels, observations, objects, holes):
        deficit = target - observed
        _require(np.all(deficit >= 0) and int(deficit.sum()) == 1
                 and np.count_nonzero(deficit) == 1, "ambiguous categorical deficit")
        color = int(np.flatnonzero(deficit)[0])
        complete = panel.copy()
        complete[hole] = color
        used, paths = set(), set()
        for obj in objs:
            values = {int(complete[p]) for p in obj["cavity"]} - {0}
            _require(len(values) == 1 and frame_color not in values
                     and structural not in values, "repaired frame must have one payload color")
            payload_color = next(iter(values))
            payload = {p for p in obj["cavity"] if complete[p] == payload_color}
            ray = _ray(panel.shape, obj["tip"], obj["direction"], len(payload))
            _require(all(panel[p] == 0 for p in ray), "ray corridor is not empty")
            _require(not (set(ray) & paths), "rays overlap")
            paths.update(ray)
            used.update(obj["frame"] | obj["cavity"])
            obj.update(payload_color=payload_color, payload=payload, quantity=len(payload),
                       ray=ray, available=len(_ray(panel.shape, obj["tip"], obj["direction"])),
                       corrupted=hole in obj["cavity"], hole=hole)
        _require(not (used & paths), "ray enters a frame region")
        repaired.append(complete)
        missing.append(color)
    # This follows from uniqueness; it also records the important B witness.
    _require(len(set(missing)) >= 2, "all panels lose the same color")
    return dict(input=inp, structural=structural, frame_color=frame_color,
                panels=panels, repaired=repaired, objects=objects, holes=holes,
                target=target, missing=missing)


def _render_analysis(info, use_a=True, use_b=True):
    out = info["input"].copy()
    for sl, panel, complete, objects, hole in zip(_panel_slices(out.shape), info["panels"],
                                               info["repaired"], info["objects"], info["holes"]):
        result = complete.copy() if use_b else panel.copy()
        if not use_b:
            result[hole] = 0  # strongest simple omission: discard missing cell as noise
        if use_a:
            for obj in objects:
                for p in obj["cavity"]:
                    result[p] = 0
            for obj in objects:
                q = obj["quantity"] - (int(obj["corrupted"]) if not use_b else 0)
                if q:
                    for p in _ray(panel.shape, obj["tip"], obj["direction"], q):
                        result[p] = obj["payload_color"]
        out[sl] = result
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, unique common inventory, and all local frame exports."""
    return _render_analysis(_analyze(input_grid))


def _profile(params):
    left, right, stem, tail = params
    rows = [(0, 0)] * (stem + 1)
    rows += [(-min(i, left), min(i, right)) for i in range(1, max(left, right) + 1)]
    rows += [(-left, right)] * tail
    frame, cavity = set(), set()
    for r, (lo, hi) in enumerate(rows):
        frame.update({(r, lo + left), (r, hi + left)})
        cavity.update((r, c + left) for c in range(lo + 1, hi))
    return frame, cavity


def _oriented(params, direction):
    frame, cavity = _profile(params)
    transformed = {p: _world(p, direction) for p in frame | cavity}
    r0 = min(r for r, _ in transformed.values())
    c0 = min(c for _, c in transformed.values())
    def move(p):
        r, c = transformed[p]
        return r - r0, c - c0
    return {move(p) for p in frame}, {move(p) for p in cavity}, move((0, params[0]))


def _partitions(total, slots, minimum=1):
    if slots == 1:
        if minimum <= total:
            yield (total,)
        return
    for value in range(minimum, total // slots + 1):
        for tail in _partitions(total - value, slots - 1, value):
            yield (value,) + tail


def _allocations(targets, number):
    results = []
    for lengths in product(range(1, number + 1), repeat=len(targets)):
        if sum(lengths) != number:
            continue
        for parts in product(*[tuple(_partitions(t, n)) for t, n in zip(targets, lengths)]):
            entries = tuple((color, q) for color, group in enumerate(parts) for q in group)
            results.append(entries)
    return results


def _choose_contents(rng, allocations, panel_index, number):
    choices = []
    for entries in allocations:
        for index, (color, q) in enumerate(entries):
            if panel_index == 0 and (color != 0 or q != 1):
                continue
            if panel_index == 1 and (color != 1 or (number > 2 and q != 1)):
                continue
            if panel_index == 2 and q < 2:
                continue
            choices.append((entries, index))
    _require(bool(choices), "no compatible contents")
    entries, hole_index = choices[int(rng.integers(len(choices)))]
    order = [int(i) for i in rng.permutation(number)]
    return [entries[i] for i in order], order.index(hole_index)


def _place(rng, shape, profiles, contents, hole_index, panel_index, anchor, common_direction):
    """Random placements over the whole panel; no fixed internal slots."""
    h, w = shape
    for _ in range(28):
        # Geometry is permuted independently of categorical inventory allocation.
        profile_order = [profiles[int(i)] for i in rng.permutation(len(profiles))]
        specs = []
        for i, ((color, q), params) in enumerate(zip(contents, profile_order)):
            mode = str(rng.choice(["free", "free", "exact", "clip", "any", "zero"]))
            if i == hole_index and panel_index in (0, 1):
                mode = "free"
            if anchor and panel_index in (0, 1):
                mode = "free"
            if anchor and panel_index == 2 and i == hole_index:
                mode = "clip"
            dirs = [common_direction] if anchor and panel_index in (0, 1) else [int(d) for d in rng.permutation(4)]
            specs.append((i, color, q, params, mode, dirs))
        specs.sort(key=lambda s: (s[4] != "free", -s[2], s[0]))
        placed, guards, boxes, ray_guards = [], set(), set(), set()
        for i, color, q, params, mode, dirs in specs:
            candidates = []
            for direction in dirs:
                frame, cavity, local_tip = _oriented(params, direction)
                fh = max(r for r, _ in frame) + 1
                fw = max(c for _, c in frame) + 1
                if len(cavity) < q:
                    continue
                for r in range(h - fh + 1):
                    for c in range(w - fw + 1):
                        tip = (r + local_tip[0], c + local_tip[1])
                        room = (tip[0], w - 1 - tip[1], h - 1 - tip[0], tip[1])[direction]
                        if (mode == "free" and room <= q or mode == "exact" and room != q
                                or mode == "clip" and not 0 < room < q
                                or mode == "zero" and room != 0):
                            continue
                        candidates.append((direction, frame, cavity, r, c, tip, fh, fw))
            accepted = None
            for raw in rng.permutation(len(candidates)):
                direction, frame, cavity, r, c, tip, fh, fw = candidates[int(raw)]
                box = {(r + y, c + x) for y in range(fh) for x in range(fw)}
                ray_guard = set(_ray(shape, tip, direction, q + 1))
                if box & (guards | ray_guards) or ray_guard & (boxes | ray_guards):
                    continue
                accepted = dict(index=i, color=color, quantity=q, direction=direction, tip=tip,
                                frame={(r + y, c + x) for y, x in frame},
                                cavity={(r + y, c + x) for y, x in cavity})
                boxes.update(box)
                guards.update((y + dy, x + dx) for y, x in box
                              for dy, dx in product((-1, 0, 1), repeat=2))
                ray_guards.update(ray_guard)
                placed.append(accepted)
                break
            if accepted is None:
                break
        if len(placed) == len(contents):
            return placed
    return None


def _make_pair(rng, roles, number, targets, anchor=False):
    structural, frame_color, *palette = roles
    allocations = _allocations(targets, number)
    for _ in range(40):
        shape = (14, 14) if anchor else (int(rng.integers(12, 15)), int(rng.integers(12, 15)))
        # All panels use the same geometry multiset, hence their frame and
        # background histograms also agree before corruption, not just payloads.
        max_q = max(targets)
        compatible = [p for p in PROFILES if len(_profile(p)[1]) >= max_q]
        profiles = [(2, 2, 0, 0)] * number if anchor else [compatible[int(rng.integers(len(compatible)))] for _ in range(number)]
        inp = np.full((2 * shape[0] + 1, 2 * shape[1] + 1), structural, dtype=np.int8)
        out = inp.copy()
        common_direction = int(rng.integers(4))
        success = True
        for panel_index, sl in enumerate(_panel_slices(inp.shape)):
            contents, hole_index = _choose_contents(rng, allocations, panel_index, number)
            objects = _place(rng, shape, profiles, contents, hole_index, panel_index, anchor, common_direction)
            if objects is None:
                success = False
                break
            panel = np.zeros(shape, dtype=np.int8)
            result = panel.copy()
            for obj in objects:
                for p in obj["frame"]:
                    panel[p] = result[p] = frame_color
                cells = sorted(obj["cavity"])
                # Mixed compact and scattered arrangements; geometry is discarded.
                if rng.random() < .5:
                    start = cells[int(rng.integers(len(cells)))]
                    cells.sort(key=lambda p: (abs(p[0] - start[0]) + abs(p[1] - start[1]), p))
                    chosen = cells[:obj["quantity"]]
                else:
                    chosen = [cells[int(i)] for i in rng.choice(len(cells), obj["quantity"], replace=False)]
                color = palette[obj["color"]]
                for p in chosen:
                    panel[p] = color
                if obj["index"] == hole_index:
                    panel[chosen[int(rng.integers(len(chosen)))]] = structural
                for p in _ray(shape, obj["tip"], obj["direction"], obj["quantity"]):
                    result[p] = color
            inp[sl], out[sl] = panel, result
        if success:
            # Corruption roles are construction witnesses, never corner codes.
            order = rng.permutation(4)
            inp = _permute_panels(inp, order)
            out = _permute_panels(out, order)
            info = _analyze(inp)
            _require(np.array_equal(out, _render_analysis(info)), "painted label disagrees with reader")
            return {"input": inp, "output": out}
    raise RuntimeError("bounded panel packing failed")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    _require(low <= value <= high, f"{name} out of range")
    return value


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count(seed, "seed", 0, 2**128 - 1)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = [int(c) for c in rng.choice(np.arange(1, 10), 5, replace=False)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                number, targets, anchor = 3, [3, 3], True
            elif split == "train" and index == 1:
                number, targets, anchor = 3, [2, 4], False
                if rng.random() < .5:
                    targets.reverse()
            elif split == "train":
                number, anchor = int(rng.integers(2, 4)), False
                targets = [1, int(rng.integers(2, 5))] if number == 2 else [int(rng.integers(2, 5)) for _ in range(2)]
            else:
                number, anchor = 4, False
                k = int(rng.integers(2, 4))
                targets = [int(rng.integers(2, 5)) for _ in range(k)]
                if sum(targets) <= 4:
                    targets[0] = 3
            for _ in range(8):
                pair = _make_pair(rng, roles, number, targets, anchor)
                if split == "train" and index == 1:
                    first_missing = _analyze(puzzle["train"][0]["input"])["missing"]
                    # Deliberately contradict a fixed color for each corner,
                    # including for the smallest legal two-example training set.
                    if _analyze(pair["input"])["missing"] == first_missing:
                        pair = {role: _permute_panels(grid, (1, 2, 3, 0))
                                for role, grid in pair.items()}
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("unable to sample a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "bad pair counts")
    seen, roles, evidence = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _array(pair["input"]), _array(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            roles.add((info["structural"], info["frame_color"]))
            _require(np.array_equal(_render_analysis(info), out), "wrong output or changed inert cells")
            _require(not np.array_equal(_render_analysis(info, use_a=False), out), "A contributes nothing")
            # Require a changed RAY cell, not merely erasure of a placeholder.
            no_b = _render_analysis(info, use_b=False)
            ray_effect = False
            full_histograms = []
            for sl, complete, objs in zip(_panel_slices(inp.shape), info["repaired"], info["objects"]):
                full_histograms.append(tuple(np.bincount(complete.ravel().astype(int), minlength=10)))
                for obj in objs:
                    ray_effect |= any(no_b[sl][p] != out[sl][p] for p in obj["ray"])
            _require(ray_effect, "repair has no visible causal effect on rays")
            _require(len(set(full_histograms)) == 1, "completed full panel histograms differ")
            _require(any(o["corrupted"] and o["quantity"] == 1 and o["available"] > 1
                         for objs in info["objects"] for o in objs), "no locally unidentifiable missing payload")
            evidence[split].append(info)
    _require(len(roles) == 1, "color role convention changes within puzzle")
    _require(len({tuple(info["missing"]) for info in evidence["train"]}) >= 2,
             "training admits a fixed missing color per corner")
    full_counts, by_profile = set(), {}
    for info in evidence["train"]:
        for objs in info["objects"]:
            for obj in objs:
                if obj["available"] > obj["quantity"]:
                    full_counts.add(obj["quantity"])
                    local = [_canonical(p, obj["direction"]) for p in obj["frame"]]
                    r0, c0 = min(r for r, _ in local), min(c for _, c in local)
                    signature = (tuple(sorted((r-r0, c-c0) for r, c in local)), obj["direction"], obj["payload_color"])
                    by_profile.setdefault(signature, set()).add(obj["quantity"])
    _require({1, 2, 3} <= full_counts, "training does not demonstrate complete varying counts")
    _require(any(len(v) >= 2 for v in by_profile.values()), "count confounded with geometry/direction/color")
    _require(any(sorted(int(v) for v in info["target"] if v) == [2, 4]
                 for info in evidence["train"]), "training lacks unequal inventory evidence")
    _require(max(len(info["objects"][0]) for info in evidence["train"]) <
             min(len(info["objects"][0]) for info in evidence["test"]), "missing frame-count test shift")


def self_test():
    """42 seeds, one puzzle at a time; no disk writes and bounded summaries."""
    shapes, profiles, directions, quantities, modes = set(), set(), set(), set(), set()
    histograms, frame_counts, replacements, colors, signatures = set(), set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for first, second in zip(puzzle[split], repeat[split]):
                for role in ("input", "output"):
                    assert _key(first[role]) == _key(second[role]), "nondeterminism"
                    signature.update(first[role].tobytes())
                pairs += 1
                info = _analyze(first["input"])
                shapes.add(info["input"].shape)
                colors.update((info["structural"], info["frame_color"]))
                histograms.add(tuple(sorted(int(v) for v in info["target"] if v)))
                replacements.add(len(set(info["missing"])))
                frame_counts.add(len(info["objects"][0]))
                for objs in info["objects"]:
                    for obj in objs:
                        q, room = obj["quantity"], obj["available"]
                        directions.add(obj["direction"])
                        quantities.add(q)
                        modes.add("free" if room > q else "exact" if room == q else "zero" if room == 0 else "clip")
                        pts = [_canonical(p, obj["direction"]) for p in obj["frame"]]
                        r0, c0 = min(r for r, _ in pts), min(c for _, c in pts)
                        profiles.add(tuple(sorted((r-r0, c-c0) for r, c in pts)))
                # Rotation, reflection and arbitrary nonzero recoloring preserve semantics.
                if seed < 4 and split == "test":
                    inp, out = first["input"], first["output"]
                    for rotation in range(4):
                        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(out, rotation))
                    assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
                    permutation = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8)
                    assert np.array_equal(solve(permutation[inp]), permutation[out])
        signatures.add(signature.digest())
    assert len(signatures) == 42 and len(shapes) >= 6
    assert len(profiles) == len(PROFILES) and directions == {0, 1, 2, 3}
    assert quantities == {1, 2, 3, 4} and modes == {"free", "exact", "clip", "zero"}
    assert frame_counts == {2, 3, 4} and len(histograms) >= 8
    assert {2, 3} <= replacements and len(colors) == 9
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Whole-panel rearrangements are legal even though generation uses different
    # panels as construction witnesses. Prediction must not read corner identity.
    sample = generate_puzzle(0)
    for split in ("train", "test"):
        pair = sample[split][0]
        for order in ((1, 0, 3, 2), (3, 2, 1, 0), (2, 0, 3, 1)):
            moved_input = _permute_panels(pair["input"], order)
            before = moved_input.copy()
            expected = _permute_panels(pair["output"], order)
            assert np.array_equal(solve(moved_input), expected), "corner-dependent rule"
            assert np.array_equal(moved_input, before), "solve modified its input"
            assert np.array_equal(solve(moved_input.tolist()), expected), "list input failed"
    # Negative semantics checks: malformed cross and ambiguous missing inventories.
    bad = sample["train"][0]["input"].copy()
    bad[bad.shape[0] // 2, 0] = 0
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed cross accepted")
    info = _analyze(sample["train"][0]["input"])
    bad = info["input"].copy()
    color = int(np.flatnonzero(info["target"])[0])
    for sl, complete in zip(_panel_slices(bad.shape), info["repaired"]):
        panel = complete.copy()
        position = tuple(np.argwhere(panel == color)[0])
        panel[position] = info["structural"]
        bad[sl] = panel
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous common inventory accepted")
    return {"seeds": 42, "pairs": pairs, "shapes": len(shapes), "profiles": len(profiles),
            "counts_per_panel": sorted(frame_counts), "quantities": sorted(quantities),
            "ray_modes": sorted(modes), "histograms": len(histograms)}
