"""Register keyed fragments, then transport a payload through opposed gateways.

The required proposal specifies transverse reversal. Parent B's final generator
instead preserves transverse offsets; this deliberate composition convention is
documented in composition.md. No parent code, data, or generation state is used
by the input-only solver. All experiments are bounded and run in memory.
"""
from __future__ import annotations

import hashlib
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=True)


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(int(counts.max()) * 2 > a.size, "background must occupy a majority")
    return int(colors[counts.argmax()])


def _components(cells):
    todo, groups = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append({p: int(cells[p]) for p in sorted(group)})
    return groups


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r - r0, c - c0): int(v) for (r, c), v in cells.items()}


def _shape(cells):
    return max(r for r, c in cells) + 1, max(c for r, c in cells) + 1


def _render(cells, background):
    cells = _normalize(cells)
    shape = _shape(cells)
    _require(max(shape) <= 30, "assembled scene is too large")
    a = np.full(shape, background, dtype=np.int8)
    for p, v in cells.items():
        a[p] = v
    return a


def _windows(cells):
    """A registration atom is an oriented 2x2 with four different colors."""
    result = {}
    for r, c in sorted(cells):
        points = ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
        if all(p in cells for p in points):
            pattern = tuple(cells[p] for p in points)
            if len(set(pattern)) == 4:
                _require(pattern not in result, "a key atom repeats within a fragment")
                result[pattern] = (r, c)
    return result


def _register(a):
    bg = _background(a)
    cells = {(int(r), int(c)): int(a[r, c]) for r, c in np.argwhere(a != bg)}
    parts = [_normalize(g) for g in _components(cells)]
    _require(3 <= len(parts) <= 6, "expected three to six keyed fragments")
    windows = [_windows(p) for p in parts]
    _require(all(windows), "each fragment needs a complete exact key")
    occurrences = {}
    for i, ws in enumerate(windows):
        for pattern, anchor in ws.items():
            occurrences.setdefault(pattern, []).append((i, anchor))
    _require(all(len(v) >= 2 for v in occurrences.values()), "unmatched key atom")
    key_colors = set().union(*(set(p) for p in occurrences))
    _require(len(key_colors) == 4, "one four-color key alphabet is required")
    edges = [[] for _ in parts]
    for copies in occurrences.values():
        i, p = copies[0]
        for j, q in copies[1:]:
            delta = p[0] - q[0], p[1] - q[1]
            edges[i].append((j, delta))
            edges[j].append((i, (-delta[0], -delta[1])))
    offsets, stack = {0: (0, 0)}, [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in edges[i]:
            candidate = offsets[i][0] + dr, offsets[i][1] + dc
            if j in offsets:
                _require(offsets[j] == candidate, "inconsistent signature translations")
            else:
                offsets[j] = candidate
                stack.append(j)
    _require(len(offsets) == len(parts), "disconnected registration graph")
    union = {}
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        for (r, c), color in part.items():
            p = r + dr, c + dc
            _require(p not in union or union[p] == color, "foreground collision")
            union[p] = color
    scene = _render(union, bg)
    # A whole key is merged once, and every foreground input cell is accounted for.
    merged_keys = _windows(_normalize(union))
    _require(set(merged_keys) == set(occurrences), "registration lost or duplicated a key")
    info = {"background": bg, "parts": parts, "offsets": offsets,
            "key_colors": key_colors, "atoms": len(occurrences),
            "input_foreground": len(cells), "scene_foreground": len(union)}
    _require(len(cells) > len(union), "no exact-overlap merging occurred")
    return scene, info


def _gate_geometry(scene, bg):
    detections = []
    for axis in (0, 1):
        view = scene if axis == 0 else scene.T
        length, width = view.shape
        if length < 7:
            continue
        for color in np.unique(view):
            color = int(color)
            if color == bg:
                continue
            mask = view == color
            top, bottom = np.flatnonzero(mask[0]), np.flatnonzero(mask[-1])
            if len(top) < 3 or len(top) != len(bottom):
                continue
            if not (np.all(np.diff(top) == 1) and np.all(np.diff(bottom) == 1)):
                continue
            aperture = len(top) - 2
            if not 1 <= aperture <= 4:
                continue
            lo, hi = int(top[0]), int(bottom[0])
            if not (lo + aperture + 2 < hi or hi + aperture + 2 < lo):
                continue
            expected = np.zeros_like(mask)
            expected[0, top] = True
            expected[1, [lo, lo + aperture + 1]] = True
            expected[-1, bottom] = True
            expected[-2, [hi, hi + aperture + 1]] = True
            if np.array_equal(mask, expected):
                detections.append((axis, color, aperture, lo, hi))
    _require(len(detections) == 1, "opposed gateway pair is not unique")
    return detections[0]


def _decode_scene(scene, bg, key_colors):
    axis, gate, aperture, top, bottom = _gate_geometry(scene, bg)
    view = scene if axis == 0 else scene.T
    length, width = view.shape
    rail_candidates = [int(v) for v in np.unique(view)
                       if v not in {bg, gate} | set(key_colors)
                       and bool(np.any(np.all(view == v, axis=0)))]
    _require(len(rail_candidates) == 1, "one continuous longitudinal support rail required")
    rail = rail_candidates[0]
    remaining = set(map(int, np.unique(view))) - {bg, gate, rail} - set(key_colors)
    _require(len(remaining) == 1, "one payload color required")
    payload = remaining.pop()
    points = {(int(r), int(c)): payload for r, c in np.argwhere(view == payload)}
    _require(len(_components(points)) == 1, "payload must be connected")
    t0, t1 = min(r for r, c in points), max(r for r, c in points)
    x0, x1 = min(c for r, c in points), max(c for r, c in points)
    _require(x1 - x0 + 1 == aperture, "payload must span its aperture")
    _require(1 <= t0 <= t1 <= length - 2, "source payload must be interior")
    _require((x0 == top + 1) != (x0 == bottom + 1), "ambiguous entry lane")
    direction = -1 if x0 == top + 1 else 1
    entry, exit_start = (top, bottom) if direction == -1 else (bottom, top)
    local = frozenset((r - t0, c - x0) for r, c in points)
    height = t1 - t0 + 1
    # All downstream cells in the exit aperture must be clear, not just the end.
    lane = view[:, exit_start + 1:exit_start + aperture + 1]
    # Only the exit gateway's back bar may occupy this lane.
    clear = lane.copy()
    clear[-1 if direction == -1 else 0] = bg
    _require(bool(np.all(clear == bg)), "obstructed downstream exit lane")
    return {"axis": axis, "background": bg, "gate": gate, "rail": rail,
            "payload": payload, "aperture": aperture, "entry": entry,
            "exit": exit_start, "direction": direction, "length": length,
            "width": width, "t0": t0, "t1": t1, "height": height,
            "local": local, "transverse_asymmetric":
                local != frozenset((t, aperture - 1 - x) for t, x in local),
            "axial_asymmetric":
                local != frozenset((height - 1 - t, x) for t, x in local),
            "shifted_lanes": top + bottom + aperture + 2 != width}


def _transport(scene, info, *, reverse=True, continue_to_boundary=True, reverse_axis=False):
    view = (scene if info["axis"] == 0 else scene.T).copy()
    view[view == info["payload"]] = info["background"]
    h, n, direction = info["height"], info["length"], info["direction"]
    if continue_to_boundary:
        t_start = 0 if direction == -1 else n - h
    else:
        # Stop just inside the paired exit, before the continued traversal.
        t_start = n - h - 2 if direction == -1 else 2
    for t, x in sorted(info["local"]):
        rr = t_start + (h - 1 - t if reverse_axis else t)
        cc = info["exit"] + 1 + (info["aperture"] - 1 - x if reverse else x)
        _require(view[rr, cc] == info["background"], "destination collision")
        view[rr, cc] = info["payload"]
    return view if info["axis"] == 0 else view.T.copy()


def _analyze(input_grid):
    a = _grid(input_grid)
    scene, registration = _register(a)
    transit = _decode_scene(scene, registration["background"], registration["key_colors"])
    return scene, registration, transit


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only from the given pixels: exact registration precedes transit."""
    scene, registration, transit = _analyze(input_grid)
    output = _transport(scene, transit)
    cells = {(int(r), int(c)): int(output[r, c])
             for r, c in np.argwhere(output != registration["background"])}
    return _render(cells, registration["background"])


def _put(cells, p, color):
    _require(p not in cells or cells[p] == color, "construction collision")
    cells[p] = int(color)


def _merge(*parts):
    union = {}
    for part in parts:
        for p, v in part.items():
            _put(union, p, v)
    return union


def _key_blocks(rng, colors, count):
    """Rectangles vary; different keys never reuse any oriented 2x2 atom."""
    used, blocks = set(), []
    choices = list(permutations(colors))
    for i in range(count):
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        rng.shuffle(choices)
        for perm in choices:
            block = {(r, c): perm[2 * (r % 2) + c % 2]
                     for r in range(h) for c in range(w)}
            atoms = set(_windows(block))
            if not atoms & used:
                used.update(atoms)
                blocks.append(block)
                break
        else:
            raise RuntimeError("no fresh key available")
    return blocks


def _payload(rng, aperture, force_asym):
    if aperture == 1:
        return np.ones((int(rng.integers(1, 6)), 1), dtype=bool)
    for attempt in range(60):
        h = int(rng.integers(2, 6))
        shape = np.zeros((h, aperture), dtype=bool)
        # A random connected backbone spans every row and column; grow side arms.
        x = int(rng.integers(aperture))
        shape[:, x] = True
        row = int(rng.integers(h))
        shape[row, :] = True
        for r in range(h):
            if rng.random() < .7:
                end = int(rng.integers(aperture))
                shape[r, min(x, end):max(x, end) + 1] = True
        if force_asym and (np.array_equal(shape, shape[:, ::-1]) or
                           np.array_equal(shape, shape[::-1]) or
                           np.array_equal(shape, shape[::-1, ::-1])):
            continue
        return shape
    shape = np.ones((3, aperture), dtype=bool)
    shape[0, -1] = False
    return shape


def _pack(parts, bg, rng):
    parts = [_normalize(p) for p in parts]
    shapes = [_shape(p) for p in parts]
    maxh, maxw = max(h for h, w in shapes), max(w for h, w in shapes)
    # At most ten candidate canvases and four placements per candidate. No files.
    canvases = [(int(rng.integers(max(maxh, 17), 31)),
                 int(rng.integers(max(maxw, 17), 31))) for _ in range(6)]
    canvases += [(30, 30)] * 4
    for H, W in canvases:
        for restart in range(4):
            blocked = np.zeros((H, W), dtype=bool)
            a = np.full((H, W), bg, dtype=np.int8)
            order = sorted(range(len(parts)), key=lambda i:
                           -(shapes[i][0] * shapes[i][1] + float(rng.random()) * 20))
            for i in order:
                ph, pw = shapes[i]
                nh, nw = H - ph + 1, W - pw + 1
                fits = np.ones((nh, nw), dtype=bool)
                for r, c in parts[i]:
                    fits &= ~blocked[r:r + nh, c:c + nw]
                choices = np.argwhere(fits)
                if not len(choices):
                    break
                top, left = map(int, choices[int(rng.integers(len(choices)))])
                for (r, c), color in parts[i].items():
                    rr, cc = r + top, c + left
                    a[rr, cc] = color
                    blocked[max(0, rr - 1):min(H, rr + 2),
                            max(0, cc - 1):min(W, cc + 2)] = True
            else:
                if int(np.count_nonzero(a == bg)) * 2 > a.size:
                    return a
    raise ValueError("bounded fragment packing failed")


def _make_pair(rng, colors, aperture, key_count, axis, direction, redundant, force_asym):
    bg, gate, payload_color, rail, *key_colors = colors
    blocks = _key_blocks(rng, key_colors, key_count)
    left_margin, right_margin = int(rng.integers(1, 4)), int(rng.integers(0, 4))
    if right_margin == left_margin:
        right_margin = (right_margin + 1) % 4
    top_start = left_margin
    spine = top_start + aperture + 6
    bottom_start = spine + int(rng.integers(3, 6))
    width = bottom_start + aperture + 2 + right_margin
    cursor, keys = int(rng.integers(3, 6)), []
    for block in blocks:
        kh, kw = _shape(block)
        keys.append({(r + cursor, c + spine - kw): v for (r, c), v in block.items()})
        cursor += kh + int(rng.integers(2, 4))
    length = cursor + int(rng.integers(1, 3))
    _require(length <= 30 and width <= 30, "construction dimensions")
    spans = [(min(r for r, c in key), max(r for r, c in key)) for key in keys]

    def rail_segment(lo, hi):
        return {(r, spine): rail for r in range(lo, hi + 1)}

    top_gate = {(0, c): gate for c in range(top_start, top_start + aperture + 2)}
    top_gate.update({(1, top_start): gate, (1, top_start + aperture + 1): gate})
    bottom_gate = {(length - 1, c): gate
                   for c in range(bottom_start, bottom_start + aperture + 2)}
    bottom_gate.update({(length - 2, bottom_start): gate,
                        (length - 2, bottom_start + aperture + 1): gate})
    top_links = {(0, c): rail for c in range(top_start)}
    top_links.update({(0, c): rail for c in range(top_start + aperture + 2, spine + 1)})
    bottom_links = {(length - 1, c): rail for c in range(spine, bottom_start)}
    bottom_links.update({(length - 1, c): rail
                         for c in range(bottom_start + aperture + 2, width)})
    parts = [_merge(top_gate, top_links, rail_segment(0, spans[0][1]), keys[0]),
             _merge(bottom_gate, bottom_links,
                    rail_segment(spans[-1][0], length - 1), keys[-1])]
    for i in range(key_count - 1):
        parts.append(_merge(keys[i], keys[i + 1],
                            rail_segment(spans[i][0], spans[i + 1][1])))

    shape = _payload(rng, aperture, force_asym)
    height = shape.shape[0]
    entry = top_start if direction == -1 else bottom_start
    facing_edge = aperture - 1 if direction == -1 else 0
    edge_rows = list(map(int, np.flatnonzero(shape[:, facing_edge])))
    key_index = int(rng.integers(key_count))
    occupied_rows = {r for key in keys for r, c in key}
    options = [(start, local_row) for start in range(2, length - height - 1)
               for local_row in edge_rows
               if start + local_row not in occupied_rows
               and min(abs(start + local_row - t) for t in spans[key_index]) <= 4]
    _require(bool(options), "no clear payload support attachment")
    start, local_row = options[int(rng.integers(len(options)))]
    attach = start + local_row
    payload = {(start + int(r), entry + 1 + int(c)): payload_color
               for r, c in np.argwhere(shape)}
    edge = entry + 1 + facing_edge
    support = {(attach, c): rail for c in range(min(edge, spine) + 1, max(edge, spine))}
    support[attach, spine] = rail
    payload_part = _merge(payload, support, keys[key_index],
                          rail_segment(min(attach, spans[key_index][0]),
                                       max(attach, spans[key_index][1])))
    parts.append(payload_part)
    if redundant:
        parts.append(dict(keys[int(rng.integers(key_count))]))
    scene = _render(_merge(*parts), bg)
    _require(scene.shape == (length, width), "boundary witnesses lost")
    # Construction renders the answer directly from its sampled shape, rather
    # than using solve or the registration offsets as an output oracle.
    expected = scene.copy()
    expected[expected == payload_color] = bg
    exit_start = bottom_start if direction == -1 else top_start
    dest_start = 0 if direction == -1 else length - height
    for r, c in np.argwhere(shape):
        p = dest_start + int(r), exit_start + aperture - int(c)
        _require(expected[p] == bg, "blocked constructed target")
        expected[p] = payload_color
    # Reflect the complete scene, including every copy of every key. Neither
    # diagonal ordering of the two transverse lanes has a special meaning.
    if bool(rng.integers(2)):
        parts = [{(r, width - 1 - c): v for (r, c), v in part.items()}
                 for part in parts]
        expected = expected[:, ::-1].copy()
    if axis == 1:
        parts = [{(c, r): v for (r, c), v in part.items()} for part in parts]
        expected = expected.T.copy()
    inp = _pack(parts, bg, rng)
    _require(np.array_equal(solve(inp), expected), "construction/solver disagreement")
    return {"input": inp, "output": expected}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if not low <= int(value) <= high:
            raise ValueError("counts out of range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)[:8]))
    vary_palette, vary_axis = bool(rng.integers(2)), bool(rng.integers(2))
    base_axis = int(rng.integers(2))
    base_direction = int(rng.choice([-1, 1]))
    wide = int(rng.integers(3, 5))
    train_settings = []
    for i in range(num_train):
        aperture = 2 if i == 0 else wide if i == 1 else 1 if i == 2 else int(rng.integers(1, 5))
        axis = (base_axis + i % 2) % 2 if vary_axis else base_axis
        direction = base_direction if i % 2 == 0 else -base_direction
        train_settings.append((aperture, axis, direction))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                aperture, axis, direction = train_settings[i]
                key_count = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
                redundant = i == 1 or (i > 1 and bool(rng.integers(2)))
            else:
                aperture, axis, direction = train_settings[int(rng.integers(num_train))]
                key_count = 3 if i == 0 else int(rng.integers(2, 4))
                redundant = bool(rng.integers(2))
            colors = tuple(map(int, rng.permutation(10)[:8])) if vary_palette else palette
            for attempt in range(24):
                try:
                    pair = _make_pair(rng, colors, aperture, key_count, axis, direction,
                                      redundant, force_asym=aperture > 1)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded generation failed")
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            scene, reg, info = _analyze(a)
            expected = _transport(scene, info)
            _require(np.array_equal(out, expected), "wrong portal output or crop")
            bg, payload = info["background"], info["payload"]
            _require(scene.shape == out.shape, "crop removed boundary evidence")
            fixed = scene != payload
            _require(np.array_equal(scene[fixed & (scene != bg)], out[fixed & (scene != bg)]),
                     "fixed gateways/scaffold changed")
            _require(np.count_nonzero(scene == payload) == np.count_nonzero(out == payload),
                     "payload cell count changed")
            view = out if info["axis"] == 0 else out.T
            t = np.argwhere(view == payload)[:, 0]
            _require(int(t.min()) == 0 if info["direction"] == -1
                     else int(t.max()) == info["length"] - 1, "payload is not flush")
            _require(not np.array_equal(scene, out), "parent B had no effect")
            # Without registration the scattered supports do not define the
            # opposed boundary gates and continuous scene rail.
            try:
                _decode_scene(a, bg, reg["key_colors"])
            except ValueError:
                pass
            else:
                raise ValueError("unregistered input already forms a valid portal scene")
            info["fragments"] = len(reg["parts"])
            info["atoms"] = reg["atoms"]
            information[split].append(info)
    train, test = information["train"], information["test"]
    _require(any(i["transverse_asymmetric"] and i["axial_asymmetric"] for i in train),
             "training must distinguish transverse-only reversal")
    _require({i["aperture"] for i in test} <= {i["aperture"] for i in train},
             "test aperture not demonstrated")
    _require({(i["axis"], i["direction"]) for i in test} <=
             {(i["axis"], i["direction"]) for i in train}, "test direction not demonstrated")
    _require(any(i["shifted_lanes"] for i in train), "local aperture evidence absent")
    return None


def self_test():
    """Stream 42 seeds, retain only small sets of signatures and statistics."""
    shapes, outputs, backgrounds, layouts, fingerprints, sizes, atoms = (set() for _ in range(7))
    axes, directions, apertures = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                scene, reg, info = _analyze(a)
                digest.update(a.tobytes())
                digest.update(out.tobytes())
                shapes.add(a.shape)
                outputs.add(out.shape)
                backgrounds.add(info["background"])
                sizes.add(len(reg["parts"]))
                atoms.add(reg["atoms"])
                layouts.add((info["aperture"], info["height"], tuple(sorted(info["local"]))))
                axes.add(info["axis"])
                directions.add(info["direction"])
                apertures.add(info["aperture"])
                _require(np.array_equal(solve(a), out), "input-only prediction failed")
                if info["transverse_asymmetric"]:
                    _require(not np.array_equal(_transport(scene, info, reverse=False), out),
                             "omitting transverse reversal gives the same answer")
                if info["axial_asymmetric"]:
                    _require(not np.array_equal(_transport(scene, info, reverse_axis=True), out),
                             "axial reversal was not distinguished")
                _require(not np.array_equal(_transport(scene, info, continue_to_boundary=False), out),
                         "continuation was not necessary")
        fingerprints.add(digest.digest())
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 30 and len(outputs) >= 30, "geometric diversity")
    _require(backgrounds == set(range(10)), "background variation")
    _require(sizes >= {3, 4, 5, 6} and len(atoms) >= 6, "registration diversity")
    _require(len(layouts) >= 35 and apertures == {1, 2, 3, 4}, "payload diversity")
    _require(axes == {0, 1} and directions == {-1, 1}, "direction diversity")
    p = generate_puzzle(123, 2, 1)
    q = generate_puzzle(123, 2, 1)
    _require(all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "determinism")
    pair = p["train"][0]
    a, out = pair["input"], pair["output"]
    scene, reg, info = _analyze(a)
    # Repacking and recoloring do not change the coordinate rule.
    repacked = _pack(reg["parts"], reg["background"], np.random.default_rng(87))
    _require(np.array_equal(solve(repacked), out), "fragment placement affected the answer")
    _require(np.array_equal(solve(a.T), out.T), "whole-example transposition")
    _require(np.array_equal(solve(a[:, ::-1]), out[:, ::-1]),
             "whole-example horizontal reflection")
    _require(np.array_equal(solve(a[::-1]), out[::-1]),
             "whole-example vertical reflection")
    mapping = np.array([9, 6, 2, 5, 7, 4, 0, 8, 1, 3], dtype=np.int8)
    _require(np.array_equal(solve(mapping[a]), mapping[out]), "semantic palette roles")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["train"][0]["output"] = _transport(scene, info, reverse=False)
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong unreflected label was accepted")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Returned data can be ordinary nested lists without losing rule validation.
    lists = {s: [{k: v.tolist() for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(lists)
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "output_shapes": len(outputs), "payload_patterns": len(layouts),
            "fragment_counts": sorted(sizes), "atom_counts": sorted(atoms)}
