"""Telescoping L rails become a palette-ordered stack of rectangular shutters.

Only standard library and NumPy are used. No parent files, seed, training labels,
or generation state enter solve(). The sampler's compositor and the input-only
per-pixel predictor use different traversal strategies, but both are authored
checks, not an independent semantic certification.
"""

from __future__ import annotations

import copy
import json

import numpy as np


_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=False)


def _key(grid):
    return (tuple(grid.shape), grid.tobytes())


def _l_cells(obj, length=None):
    length = obj["length"] if length is None else int(length)
    r, c = obj["anchor"]
    dr, dc = obj["direction"]
    pr, pc = obj["perpendicular"]
    tip = (r + (length - 1) * dr, c + (length - 1) * dc)
    return ({(r + t * dr, c + t * dc) for t in range(length)}
            | {(tip[0] + s * pr, tip[1] + s * pc) for s in range(obj["width"])})


def _bounds(obj, length):
    r, c = obj["anchor"]
    dr, dc = obj["direction"]
    pr, pc = obj["perpendicular"]
    end = (r + (length - 1) * dr + (obj["width"] - 1) * pr,
           c + (length - 1) * dc + (obj["width"] - 1) * pc)
    return min(r, end[0]), min(c, end[1]), max(r, end[0]), max(c, end[1])


def _inside(cell, bounds):
    r, c = cell
    return bounds[0] <= r <= bounds[2] and bounds[1] <= c <= bounds[3]


def _recognize_l(coords, direction):
    """Recover stem and cap by projection, without any sampled dimensions."""
    dr, dc = direction
    transverse = (0, 1) if dr else (1, 0)
    entries = {(r * dr + c * dc, r * transverse[0] + c * transverse[1])
               for r, c in coords}
    us, vs = [u for u, v in entries], [v for u, v in entries]
    lo, hi, left, right = min(us), max(us), min(vs), max(vs)
    length, width = hi - lo + 1, right - left + 1
    if length < 3 or width < 2:
        return None
    for anchor_v, sign in ((left, 1), (right, -1)):
        expected = ({(u, anchor_v) for u in range(lo, hi + 1)}
                    | {(hi, v) for v in range(left, right + 1)})
        if entries == expected:
            anchor = (lo * dr, anchor_v) if dr else (anchor_v, lo * dc)
            return dict(anchor=anchor, direction=direction,
                        perpendicular=(transverse[0] * sign, transverse[1] * sign),
                        length=length, width=width, cells=set(coords),
                        position=anchor_v, baseline=lo)
    return None


def _parse(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    frequency = np.bincount(grid.ravel(), minlength=10)
    bg = int(frequency.argmax())
    if frequency[bg] * 2 <= grid.size:
        raise ValueError("input background must occupy more than half the cells")
    candidates = []
    for n in (3, 5, 7):
        th = n + 2
        for r in range(h - th + 1):
            for c in range(w - 3):
                border = int(grid[r, c])
                if border == bg:
                    continue
                tile = grid[r:r + th, c:c + 4]
                if not (np.all(tile[0] == border) and np.all(tile[-1] == border)
                        and np.all(tile[:, 0] == border) and np.all(tile[:, -1] == border)):
                    continue
                cues = tuple(int(x) for x in tile[1:-1, 1])
                if len(set(cues)) != n or bg in cues or border in cues:
                    continue
                table = (r, c, r + th - 1, c + 3)
                if any(grid[rr, cc] != bg
                       for rr in range(max(0, r - 1), min(h, r + th + 1))
                       for cc in range(max(0, c - 1), min(w, c + 5))
                       if not _inside((rr, cc), table)):
                    continue
                scene = grid.copy()
                scene[r:r + th, c:c + 4] = bg
                if set(int(x) for x in np.unique(scene)) != {bg, *cues}:
                    continue
                coords = {cue: {tuple(map(int, x)) for x in np.argwhere(scene == cue)}
                          for cue in cues}
                interpretations = []
                for direction in _DIRECTIONS:
                    objects = [_recognize_l(coords[cue], direction) for cue in cues]
                    if any(o is None for o in objects):
                        continue
                    if len({o["baseline"] for o in objects}) != 1:
                        continue
                    ordered = sorted(objects, key=lambda o: o["position"])
                    gaps = [b["position"] - a["position"] for a, b in zip(ordered, ordered[1:])]
                    if min(gaps) < 2 or len({o["length"] for o in objects}) == 1:
                        continue
                    for rank, (obj, cue, paint) in enumerate(zip(objects, cues, tile[1:-1, 2])):
                        obj.update(cue=cue, paint=int(paint), rank=rank)
                    reference = ordered[n // 2]["length"]
                    bounds = [_bounds(o, max(o["length"], reference)) for o in objects]
                    union = (min(b[0] for b in bounds), min(b[1] for b in bounds),
                             max(b[2] for b in bounds), max(b[3] for b in bounds))
                    if union[0] < 0 or union[1] < 0 or union[2] >= h or union[3] >= w:
                        continue
                    # Table lies outside the whole source/target scene box and its moat.
                    expanded = (table[0] - 1, table[1] - 1, table[2] + 1, table[3] + 1)
                    if not (union[2] < expanded[0] or union[0] > expanded[2]
                            or union[3] < expanded[1] or union[1] > expanded[3]):
                        continue
                    interpretations.append(dict(background=bg, border=border, table=table,
                                                cues=cues, objects=objects, ordered=ordered,
                                                direction=direction, reference=reference,
                                                gaps=tuple(gaps), scene_bounds=union))
                if len(interpretations) == 1:
                    candidates.append(interpretations[0])
    if len(candidates) != 1:
        raise ValueError("expected a unique framed table and anchored L-rail interpretation")
    return candidates[0]


def _predict(grid, parsed, *, reference="middle", normalize=True, filled=True,
             lookup=True, priority="table", transparent=False, common_arm=False):
    """Per-pixel first-covering-layer predictor; switches are review ablations."""
    objects, ordered = parsed["objects"], parsed["ordered"]
    lengths = [o["length"] for o in ordered]
    n = len(objects)
    if reference == "middle":
        target = parsed["reference"]
    elif reference == "median":
        target = sorted(lengths)[n // 2]
    elif reference == "geometric":
        center = ordered[0]["position"] + ordered[-1]["position"]
        target = min(ordered, key=lambda o: abs(2 * o["position"] - center))["length"]
    elif reference == "first":
        target = lengths[0]
    elif reference == "last":
        target = lengths[-1]
    elif reference == "shortest":
        target = min(lengths)
    elif reference == "longest":
        target = max(lengths)
    elif _integer(reference):
        target = int(reference)
    else:
        raise ValueError("unknown reference selector")
    order = (objects if priority == "table" else list(reversed(objects))
             if priority == "reverse" else ordered)
    layers = []
    for original in order:
        obj = dict(original)
        if common_arm:
            exemplar = ordered[n // 2]
            obj["width"], obj["perpendicular"] = exemplar["width"], exemplar["perpendicular"]
        length = target if normalize else obj["length"]
        layers.append((obj, _bounds(obj, length), _l_cells(obj, length)))
    output = grid.copy()
    bg = parsed["background"]
    for obj in objects:
        for cell in obj["cells"]:
            output[cell] = bg
    # The table and cells outside target footprints are preserved. All legal
    # predictions fit; counterfactual width/length rules are clipped for comparison.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if _inside((r, c), parsed["table"]):
                continue
            for obj, bounds, edge in layers:
                if not _inside((r, c), bounds):
                    continue
                if (r, c) in edge:
                    output[r, c] = obj["cue"]
                    break
                if not filled:
                    continue
                paint = obj["paint"] if lookup else obj["cue"]
                if transparent and paint == bg:
                    continue
                output[r, c] = paint
                break
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire output from the input; do not mutate the input."""
    grid = _grid(input_grid)
    return _predict(grid, _parse(grid))


def _compose(grid, objects, bg, target):
    """Generation labels: clear rails, then draw complete layers back to front."""
    out = grid.copy()
    for obj in objects:
        for cell in _l_cells(obj):
            out[cell] = bg
    for obj in reversed(objects):
        r0, c0, r1, c1 = _bounds(obj, target)
        out[r0:r1 + 1, c0:c1 + 1] = obj["paint"]
        for cell in _l_cells(obj, target):
            out[cell] = obj["cue"]
    return out


def _sample_lengths(rng, n, reference, capacity, witness):
    lengths = [reference] * n
    others = [int(x) for x in rng.permutation(n) if x != n // 2]
    def short():
        return int(rng.integers(3, reference))
    def long():
        return int(rng.integers(reference + 1, capacity + 1))
    if witness:
        majority_short = bool(rng.integers(2))
        for i in others:
            lengths[i] = short() if majority_short else long()
        lengths[others[0]] = long() if majority_short else short()
    else:
        mode = str(rng.choice(("mixed", "extend", "trim", "one_extend", "one_trim")))
        if reference == 3:
            mode = "trim" if "one" not in mode else "one_trim"
        for i in others:
            if rng.random() < .25 or (mode.startswith("one") and i != others[0]):
                continue
            lengths[i] = short() if mode in ("extend", "one_extend") or (mode == "mixed" and rng.random() < .5) else long()
        if mode == "mixed":
            lengths[others[0]], lengths[others[1]] = short(), long()
        elif all(x == reference for x in lengths):
            lengths[others[0]] = short() if mode in ("extend", "one_extend") else long()
    return lengths


def _pack(rng, objects, direction, bg, border, target):
    """Rotate the scene only; the visible table keeps its fixed column semantics."""
    rotation = _DIRECTIONS.index(direction)
    def turn(r, c):
        for _ in range(rotation):
            r, c = -c, r
        return r, c
    scene = []
    for obj in objects:
        o = dict(obj)
        o["anchor"] = turn(*obj["anchor"])
        o["direction"] = turn(*obj["direction"])
        o["perpendicular"] = turn(*obj["perpendicular"])
        scene.append(o)
    boxes = [_bounds(o, max(o["length"], target)) for o in scene]
    lo = min(b[0] for b in boxes), min(b[1] for b in boxes)
    sh = max(b[2] for b in boxes) - lo[0] + 1
    sw = max(b[3] for b in boxes) - lo[1] + 1
    th, tw = len(objects) + 2, 4
    gap = int(rng.integers(1, 4))
    options = []
    for side in ("left", "right"):
        if max(sh, th) <= 30 and sw + gap + tw <= 30:
            options.append((side, max(sh, th), sw + gap + tw))
    for side in ("top", "bottom"):
        if sh + gap + th <= 30 and max(sw, tw) <= 30:
            options.append((side, sh + gap + th, max(sw, tw)))
    if not options:
        return None
    side, ph, pw = options[int(rng.integers(len(options)))]
    if side in ("left", "right"):
        sr, tr = int(rng.integers(ph - sh + 1)), int(rng.integers(ph - th + 1))
        sc, tc = (tw + gap, 0) if side == "left" else (0, sw + gap)
    else:
        sc, tc = int(rng.integers(pw - sw + 1)), int(rng.integers(pw - tw + 1))
        sr, tr = (th + gap, 0) if side == "top" else (0, sh + gap)
    nonbg = sum(o["length"] + o["width"] - 1 for o in objects) + 2 * th + 4
    nonbg += sum(1 + (o["paint"] != bg) for o in objects)
    shapes = [(h, w) for h in range(ph, 31) for w in range(pw, 31) if h * w > 2 * nonbg]
    if not shapes:
        return None
    mode = int(rng.integers(4))
    if mode == 0:
        preferred = [s for s in shapes if s[0] <= ph + 3 and s[1] <= pw + 3]
    elif mode == 1:
        preferred = [s for s in shapes if s[0] == s[1]]
    elif mode == 2:
        preferred = [s for s in shapes if max(s[0] - ph, s[1] - pw) >= 6]
    else:
        preferred = shapes
    choices = preferred or shapes
    h, w = choices[int(rng.integers(len(choices)))]
    off_r, off_c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
    sr, sc, tr, tc = sr + off_r, sc + off_c, tr + off_r, tc + off_c
    for o in scene:
        o["anchor"] = (o["anchor"][0] - lo[0] + sr, o["anchor"][1] - lo[1] + sc)
    grid = np.full((h, w), bg, dtype=np.int64)
    grid[tr:tr + th, tc:tc + tw] = border
    for rank, o in enumerate(scene):
        grid[tr + rank + 1, tc + 1:tc + 3] = o["cue"], o["paint"]
        for cell in _l_cells(o):
            if grid[cell] != bg:
                return None
            grid[cell] = o["cue"]
    return dict(input=grid, output=_compose(grid, scene, bg, target))


def _candidate(rng, descriptor):
    n, witness = descriptor["count"], descriptor["witness"]
    references = [r for r in descriptor["references"] if r not in descriptor["avoid_reference"]]
    reference = int(rng.choice(references))
    capacity = int(rng.integers(max(reference + 1, 8), 21))
    lengths = _sample_lengths(rng, n, reference, capacity, witness)
    gaps = [2] * (n - 1)
    if witness:
        gaps[0 if rng.random() < .5 else -1] += int(rng.choice((5, 7)))
    elif rng.random() < .35:
        gaps = [int(rng.integers(2, min(4, 20 // (n - 1)) + 1))] * (n - 1)
    else:
        for _ in range(int(rng.integers(1, 21 - sum(gaps)))):
            gaps[int(rng.integers(n - 1))] += 1
    positions = [0] + [int(x) for x in np.cumsum(gaps)]
    objects = [dict(anchor=(0, p), direction=(1, 0),
                    perpendicular=(0, 1), width=2, length=length)
               for p, length in zip(positions, lengths)]
    # Cross a shorter neighbor with the longer rail's free arm. Since its
    # elbow is beyond that neighbor's original tip, the input can stay legible.
    topology = "crossing" if witness else str(rng.choice(("crossing", "free", "disjoint"), p=(.55, .25, .20)))
    i = j = None
    occupied = {(t, p) for p, length in zip(positions, lengths) for t in range(length)}
    if topology == "crossing":
        neighbor_pairs = [(a, b) for a in range(n) for b in (a - 1, a + 1)
                          if 0 <= b < n and lengths[a] > lengths[b]
                          and abs(positions[a] - positions[b]) + 2 <= 8]
        if not neighbor_pairs:
            return None
        i, j = neighbor_pairs[int(rng.integers(len(neighbor_pairs)))]
        objects[i]["perpendicular"] = (0, 1 if j > i else -1)
        objects[i]["width"] = abs(positions[i] - positions[j]) + 2
        occupied.update(_l_cells(objects[i]))
    if topology == "disjoint":
        # One common arm side permits independent widths that never reach the
        # next anchor. This deliberately includes genuinely separated scenes.
        sign = int(rng.choice((-1, 1)))
        for k, obj in enumerate(objects):
            gap_index = k if sign == 1 else k - 1
            ceiling = min(8, gaps[gap_index]) if 0 <= gap_index < n - 1 else 8
            obj["width"] = int(rng.integers(2, ceiling + 1))
            obj["perpendicular"] = (0, sign)
    else:
        # Choose arms using the actual free cells, rather than relying on rare
        # collision-free draws of many unrestricted widths. This is especially
        # useful when most stems already share the reference length.
        for k in rng.permutation(n):
            k = int(k)
            if k == i:
                continue
            choices = []
            for sign in (-1, 1):
                for width in range(2, 9):
                    extra = {(lengths[k] - 1, positions[k] + sign * s) for s in range(1, width)}
                    if not extra & occupied:
                        choices.append((sign, width, extra))
            if not choices:
                return None
            sign, width, extra = choices[int(rng.integers(len(choices)))]
            objects[k]["perpendicular"], objects[k]["width"] = (0, sign), width
            occupied.update(extra)
    box = [_bounds(o, max(o["length"], reference)) for o in objects]
    if max(b[3] for b in box) - min(b[1] for b in box) + 1 > 24:
        return None
    bg = int(rng.integers(10))
    if bg in descriptor["avoid_background"]:
        bg = int(rng.choice([c for c in range(10) if c not in descriptor["avoid_background"]]))
    colors = [int(x) for x in rng.permutation([c for c in range(10) if c != bg])]
    border, cues = colors[0], colors[1:n + 1]
    for obj, cue in zip(objects, cues):
        obj.update(cue=cue, paint=int(rng.integers(10)))
    if witness:
        objects[i]["paint"] = bg
        objects[j]["paint"] = int(rng.choice([c for c in range(10) if c != bg]))
    order = [int(x) for x in rng.permutation(n)]
    if witness and order.index(i) > order.index(j):
        ii, jj = order.index(i), order.index(j)
        order[ii], order[jj] = order[jj], order[ii]
    return _pack(rng, [objects[k] for k in order], descriptor["direction"], bg, border, reference)


def _features(grid, parsed=None):
    p = _parse(grid) if parsed is None else parsed
    objects, ordered = p["objects"], p["ordered"]
    target, bg = p["reference"], p["background"]
    expected = _predict(grid, p)
    found = set()
    if any(o["length"] < target for o in objects):
        found.add("extend")
    if any(o["length"] > target for o in objects):
        found.add("trim")
    if sorted(o["length"] for o in objects)[len(objects) // 2] != target:
        found.add("nonmedian")
    center = ordered[0]["position"] + ordered[-1]["position"]
    distances = [abs(2 * o["position"] - center) for o in ordered]
    if distances.count(min(distances)) == 1:
        nearest = distances.index(min(distances))
        if nearest != len(objects) // 2 and ordered[nearest]["length"] != target:
            found.add("geometric")
    for obj in objects:
        if obj["length"] > target:
            old_arm = {cell for cell in obj["cells"] if cell not in _l_cells(obj, target)}
            if any(expected[cell] == bg for cell in old_arm):
                found.add("transport_erase")
    if len({o["width"] for o in objects}) > 1:
        found.add("widths")
    if len({o["perpendicular"] for o in objects}) > 1:
        found.add("sides")
    comparisons = {
        "B": dict(normalize=False), "A": dict(filled=False),
        "lookup": dict(lookup=False), "priority": dict(priority="reverse"),
        "spatial_priority": dict(priority="spatial"),
        "clearing": dict(transparent=True), "arm": dict(common_arm=True),
    }
    for name, arguments in comparisons.items():
        if not np.array_equal(expected, _predict(grid, p, **arguments)):
            found.add(name)
    bounds = [_bounds(o, target) for o in objects]
    occluded = sum(expected[cell] != o["cue"] for o in objects for cell in _l_cells(o, target))
    if occluded:
        found.add("occlusion")
    overlaps = sum(not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
                   for i, a in enumerate(bounds) for b in bounds[i + 1:])
    return dict(witnesses=found, count=len(objects), reference=target,
                lengths=tuple(o["length"] for o in ordered), widths=tuple(o["width"] for o in ordered),
                sides=tuple(o["perpendicular"] for o in ordered), gaps=p["gaps"],
                direction=p["direction"], shape=grid.shape, background=bg,
                table=p["table"], order=tuple(o["cue"] for o in objects),
                spatial_order=tuple(o["cue"] for o in ordered), overlaps=int(overlaps),
                occluded=int(occluded))


_CORE_WITNESSES = {"extend", "trim", "nonmedian", "geometric", "transport_erase",
                   "widths", "sides", "A", "B", "lookup", "priority",
                   "spatial_priority", "clearing", "arm", "occlusion"}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("count", "direction", "reach")))
    base = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    training_reference = None
    training_background = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index == 0
            if split == "train":
                n = 5 if witness else 3 if index == 1 else int(rng.choice((3, 5) if regime == "count" else (3, 5, 7)))
                direction = _DIRECTIONS[base if regime == "direction" else (base + index) % 4]
                references = range(4, 9) if regime == "reach" else range(4, 13)
            else:
                n = 7 if regime == "count" and index == 0 else int(rng.choice((3, 5, 7)))
                direction = _DIRECTIONS[(base + 1 + index % 3) % 4] if regime == "direction" else _DIRECTIONS[int(rng.integers(4))]
                references = range(9, 15) if regime == "reach" else range(3, 15)
            descriptor = dict(count=n, direction=direction, references=tuple(references),
                              witness=witness,
                              avoid_reference={training_reference} if split == "train" and index == 1 else set(),
                              avoid_background={training_background} if split == "train" and index == 1 else set())
            for _ in range(1600):
                pair = _candidate(rng, descriptor)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                try:
                    parsed = _parse(pair["input"])
                    features = _features(pair["input"], parsed)
                except ValueError:
                    continue
                if not {"A", "B"} <= features["witnesses"]:
                    continue
                if witness and not _CORE_WITNESSES <= features["witnesses"]:
                    continue
                if not np.array_equal(_predict(pair["input"], parsed), pair["output"]):
                    raise AssertionError("sampled compositor and input-derived solver disagree")
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if witness:
                    training_reference, training_background = parsed["reference"], parsed["background"]
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {seed}/{split}/{index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    witnesses = set()
    alternatives = {"median", "geometric", "first", "last", "shortest", "longest", 4}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            p = _parse(inp)
            expected = _predict(inp, p)
            assert np.array_equal(out, expected), "incorrect shutter output"
            assert not np.array_equal(inp, out)
            r0, c0, r1, c1 = p["table"]
            assert np.array_equal(inp[r0:r1 + 1, c0:c1 + 1], out[r0:r1 + 1, c0:c1 + 1])
            active = set().union(*(o["cells"] for o in p["objects"]))
            for o in p["objects"]:
                b = _bounds(o, p["reference"])
                active.update((r, c) for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1))
            changed = {tuple(map(int, x)) for x in np.argwhere(inp != out)}
            assert changed <= active, "changed a cell outside consumed rails and target rectangles"
            # Direct cell ownership invariant, independent of the sampled label traversal.
            for cell in active:
                owners = [o for o in p["objects"] if _inside(cell, _bounds(o, p["reference"]))]
                if owners:
                    owner = owners[0]
                    value = owner["cue"] if cell in _l_cells(owner, p["reference"]) else owner["paint"]
                else:
                    value = p["background"]
                assert out[cell] == value, "incorrect front layer, cue edge, or cleared tail"
            f = _features(inp, p)
            assert {"A", "B"} <= f["witnesses"], "an inherited concept is inert"
            if split == "train":
                training.append(f)
                witnesses.update(f["witnesses"])
                for rule in tuple(alternatives):
                    if not np.array_equal(out, _predict(inp, p, reference=rule)):
                        alternatives.remove(rule)
    assert not alternatives, f"training leaves reference alternatives: {alternatives}"
    assert _CORE_WITNESSES <= witnesses, f"missing training evidence: {_CORE_WITNESSES - witnesses}"
    assert len({f["reference"] for f in training}) >= 2
    assert len({f["count"] for f in training}) >= 2
    assert len({f["background"] for f in training}) >= 2


def self_test():
    """Bounded semantic checks and actual-array diversity across 42 seeds."""
    rng = np.random.default_rng(21973)
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple((_key(pair["input"]), _key(pair["output"]))
                          for s in ("train", "test") for pair in p[s])
        signatures.add(signature)
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
        for s in ("train", "test"):
            for pair in p[s]:
                inp, out = pair["input"], pair["output"]
                parsed = _parse(inp)
                f = _features(inp, parsed)
                f["split"] = s
                records.append(f)
            # Palette permutation commutes with every semantic role.
            pair = p[s][0]
            permutation = rng.permutation(10)
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        # Input intervention: changing the middle rail's visible endpoint and
        # attached cap changes the reach; a constant-length solver cannot pass.
        first = p["train"][0]
        parsed = _parse(first["input"])
        middle = parsed["ordered"][len(parsed["ordered"]) // 2]
        shortened = first["input"].copy()
        for cell in middle["cells"]:
            shortened[cell] = parsed["background"]
        new_cells = _l_cells(middle, middle["length"] - 1)
        if all(shortened[cell] == parsed["background"] for cell in new_cells):
            for cell in new_cells:
                shortened[cell] = middle["cue"]
            try:
                changed = _parse(shortened)
            except ValueError:
                pass  # An intervention can invalidate the unique shared baseline.
            else:
                assert changed["reference"] == parsed["reference"] - 1
                assert not np.array_equal(solve(shortened), first["output"])
        # The table is causal independently of anchor order.
        altered = first["input"].copy()
        r0, c0, r1, c1 = parsed["table"]
        altered[r0 + 1:r1, c0 + 1:c1] = altered[r0 + 1:r1, c0 + 1:c1][::-1].copy()
        assert not np.array_equal(solve(altered), first["output"])
    assert len(signatures) == 42
    assert {f["count"] for f in records} == {3, 5, 7}
    assert {f["direction"] for f in records} == set(_DIRECTIONS)
    assert {f["background"] for f in records} == set(range(10))
    assert len({f["shape"] for f in records}) >= 30
    assert len({f["reference"] for f in records}) >= 8
    assert len({(f["count"], f["lengths"], f["widths"], f["gaps"]) for f in records}) >= 180
    assert any(f["reference"] >= 12 for f in records if f["split"] == "test")
    assert any(len(set(f["gaps"])) == 1 for f in records)
    assert any(len(set(f["gaps"])) > 1 for f in records)
    assert len({f["overlaps"] for f in records}) >= 4
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(90210, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    broken = copy.deepcopy(generate_puzzle(0, 2, 1))
    out = broken["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                grid_shapes=len({f["shape"] for f in records}),
                references=sorted({f["reference"] for f in records}),
                structural_signatures=len({(f["count"], f["lengths"], f["widths"], f["gaps"]) for f in records}),
                overlap_counts=sorted({f["overlaps"] for f in records}),
                note="Generator-owned checks; finite diversity measurements, not independent certification.")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
