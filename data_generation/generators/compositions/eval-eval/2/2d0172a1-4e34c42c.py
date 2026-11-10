"""Merge stamped partial containment views, then draw their canonical diagram.

Only NumPy and the standard library are used. Nothing is read or written by this
module. A stamp identifies an object within ONE input; it has no learned meaning.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            y, x = stack.pop()
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(np.asarray(group, dtype=int))
    return groups


def _holes(mask):
    padded = np.pad(mask, 1)
    reached = np.zeros(padded.shape, bool)
    reached[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~reached)[1:-1, 1:-1]


def _layout(ids, objects):
    """Infer one cardinal list from actual separation and shared projections."""
    if len(ids) <= 1:
        return None, list(ids)
    choices = []
    for axis in (0, 1):
        ordered = sorted(ids, key=lambda i: int(objects[i]["lo"][axis]))
        cross = 1-axis
        separated = all(objects[b]["lo"][axis] >= objects[a]["hi"][axis]+2
                        for a, b in zip(ordered, ordered[1:]))
        aligned = max(objects[i]["lo"][cross] for i in ids) <= min(objects[i]["hi"][cross] for i in ids)
        if separated and aligned:
            choices.append((axis, ordered))
    _require(len(choices) == 1, "ambiguous or non-cardinal local child layout")
    return choices[0]


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "input needs a uniform background border")
    counts = Counter(int(v) for v in grid.flat if v != bg)
    _require(len(counts) >= 3, "missing drawing or stamp colors")
    drawing, frequency = counts.most_common(1)[0]
    _require(sum(n == frequency for n in counts.values()) == 1, "ambiguous drawing color")
    objects = []
    for coords in _components(grid != bg):
        lo, hi = coords.min(0), coords.max(0)
        _require(np.all(hi-lo >= 1), "object smaller than its stamp")
        y, x = lo
        stamp = grid[y:y+2, x:x+2]
        key = tuple(int(v) for v in stamp.flat)
        _require(key[0] == drawing and all(v not in (bg, drawing) for v in key[1:]), "invalid upper-left stamp")
        _require(len(set(key[1:])) >= 2, "stamp needs two accent colors")
        mask = np.zeros(grid.shape, bool)
        mask[coords[:, 0], coords[:, 1]] = True
        _require(np.all(mask[y:y+2, x:x+2]), "stamp belongs to one component")
        rest = mask.copy()
        rest[y:y+2, x:x+2] = False
        _require(np.all(grid[rest] == drawing), "accent outside a stamp")
        hole = _holes(mask)
        _require(len(_components(hole)) <= 1, "multiple holes in one contour")
        kind = "F" if hole.any() else "L"
        if kind == "L":
            _require(len(coords) >= .5*np.prod(hi-lo+1), "noncompact terminal blob")
        objects.append(dict(lo=lo, hi=hi, mask=mask, hole=hole, key=key, kind=kind, area=len(coords)))
    _require(objects, "empty input")
    labels = np.zeros(grid.shape, int)
    for i, obj in enumerate(objects):
        labels[obj["mask"]] = i+1
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a > 0) & (b > 0) & (a != b)), "touching objects")
    parents = []
    for i, obj in enumerate(objects):
        containers = []
        for j, candidate in enumerate(objects):
            if j == i or candidate["kind"] != "F":
                continue
            inside = candidate["hole"][obj["mask"]]
            _require(inside.all() or not inside.any(), "crossing objects")
            if inside.all():
                containers.append(j)
        parents.append(min(containers, key=lambda j: int(objects[j]["hole"].sum())) if containers else None)
    children = {i: [] for i in range(len(objects))}
    roots = []
    for i, p in enumerate(parents):
        (roots if p is None else children[p]).append(i)
    layouts = {}
    for i, obj in enumerate(objects):
        _require(obj["kind"] == "F" or not children[i], "blob contains another object")
        layouts[i] = _layout(children[i], objects)
    return dict(background=bg, drawing=drawing, objects=objects, parents=parents,
                children=children, roots=roots, layouts=layouts)


def _unique_order(nodes, edges):
    remaining, result = set(nodes), []
    while remaining:
        ready = [v for v in remaining if not any(b == v and a in remaining for a, b in edges)]
        _require(len(ready) == 1, "child order is cyclic or underdetermined")
        result.append(ready[0])
        remaining.remove(ready[0])
    return result


def _stitch(info):
    objects = info["objects"]
    kinds, parents, axes, order_edges = {}, {}, {}, {}
    for i, obj in enumerate(objects):
        key, kind = obj["key"], obj["kind"]
        _require(key not in kinds or kinds[key] == kind, "one stamp has two object types")
        kinds[key] = kind
        axis, ordered = info["layouts"][i]
        keys = [objects[j]["key"] for j in ordered]
        _require(len(set(keys)) == len(keys), "duplicate identity in one view")
        if axis is not None:
            _require(key not in axes or axes[key] == axis, "inconsistent orientation")
            axes[key] = axis
        order_edges.setdefault(key, set()).update(zip(keys, keys[1:]))
        for child in keys:
            _require(child != key and (child not in parents or parents[child] == key), "inconsistent parent")
            parents[child] = key
    roots = set(kinds)-set(parents)
    _require(len(roots) == 1, "views do not determine a single root")
    root = next(iter(roots))
    _require(kinds[root] == "F", "root must be a contour")
    children = {}
    for key, kind in kinds.items():
        ids = [v for v in kinds if parents.get(v) == key]
        if kind == "L":
            _require(not ids, "terminal identity has children")
            children[key] = None
        else:
            _require(ids, "globally empty contour")
            children[key] = _unique_order(ids, order_edges[key])
            _require(len(ids) == 1 or key in axes, "missing child axis")
            axes.setdefault(key, 1)  # A singleton has exactly the same tile on either axis.
    visited = set()

    def visit(key):
        _require(key not in visited, "containment cycle")
        visited.add(key)
        for child in children[key] or []:
            visit(child)

    visit(root)
    _require(visited == set(kinds), "disconnected containment cycle")
    return root, children, axes


def _compose(parts, axis, gap, background):
    cross = 1-axis
    shape = [0, 0]
    shape[axis] = sum(p.shape[axis] for p in parts)+gap*(len(parts)-1)
    shape[cross] = max(p.shape[cross] for p in parts)
    result = np.full(shape, background, np.int8)
    position = 0
    for part in parts:
        start = [0, 0]
        start[axis] = position
        start[cross] = (shape[cross]-part.shape[cross])//2
        y, x = start
        h, w = part.shape
        result[y:y+h, x:x+w] = part
        position += part.shape[axis]+gap
    return result


def _canonical(root, children, axes, bg, drawing):
    def draw(key):
        if children[key] is None:
            return np.array([[drawing]], np.int8)
        content = _compose([draw(c) for c in children[key]], axes[key], 1, bg)
        result = np.pad(content, 2, constant_values=bg)
        result[0] = result[-1] = drawing
        result[:, 0] = result[:, -1] = drawing
        return result

    return draw(root)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover identities, partial child orders, and containment from pixels only."""
    info = _read(input_grid)
    root, children, axes = _stitch(info)
    answer = _canonical(root, children, axes, info["background"], info["drawing"])
    _require(max(answer.shape) <= 30, "canonical output exceeds ARC bounds")
    return answer


def _scene(rng, mode):
    """An ordered tree, independent of its later observations and metric shapes."""
    if mode in ("witness", "order"):
        children = {0: [1, 2], 1: [3, 4], 2: None, 3: None, 4: None}
        if mode == "order":
            children = {0: [1, 2, 3], 1: [4, 5], 2: None,
                        3: [6], 4: None, 5: None, 6: None}
            if rng.random() < .5:
                children[1].append(7)
                children[7] = None
        elif rng.random() < .5:
            children[1].append(5)
            children[5] = None
        axis = int(rng.integers(2))
        axes = {0: axis, 1: 1-axis}
        if mode == "order":
            axes[3] = int(rng.integers(2))
    else:
        nf = int(rng.integers(3, 5) if mode == "test" else rng.integers(2, 4))
        children = {i: [] for i in range(nf)}
        for i in range(1, nf):
            # Tests contain at least three nested contours after registration.
            parent = (i-1 if i < 3 else int(rng.integers(i))) if mode == "test" else 0
            children[parent].append(i)
        next_id = nf
        for i in range(nf):
            if not children[i] or (i == 0 and len(children[i]) == 1):
                children[i].append(next_id)
                children[next_id] = None
                next_id += 1
        for _ in range(int(rng.integers(1, 4))):
            choices = [i for i in range(nf) if len(children[i]) < 3]
            if not choices:
                break
            p = int(rng.choice(choices))
            children[p].append(next_id)
            children[next_id] = None
            next_id += 1
        axes = {i: int(rng.integers(2)) for i in range(nf)}
    for ids in children.values():
        if ids is not None:
            rng.shuffle(ids)
    return 0, children, axes


def _views(rng, root, children, force_order=False, force_absorbed=False):
    def stub(key):
        return key, None if children[key] is None else ()

    frames = [k for k in children if children[k] is not None]
    views = [(k, tuple(stub(c) for c in children[k])) for k in frames]
    candidates = [i for i, (k, _) in enumerate(views) if len(children[k]) == 3]
    if candidates and (force_order or rng.random() < .5):
        i = candidates[0] if force_order else int(rng.choice(candidates))
        k, parts = views[i]
        views[i:i+1] = [(k, parts[:2]), (k, parts[1:])]
    # A detail already visible inside one view may be repeated in its own view.
    # Only a proper subset is expanded, so no single view reveals the full tree.
    if rng.random() < .45:
        options = [(i, j, c) for i, (_, parts) in enumerate(views)
                   for j, (c, _) in enumerate(parts)
                   if children[c] is not None and len(children[c]) >= 2]
        if options:
            i, j, c = options[int(rng.integers(len(options)))]
            k, parts = views[i]
            subset = children[c][:int(rng.integers(1, len(children[c])))]
            updated = list(parts)
            updated[j] = (c, tuple(stub(v) for v in subset))
            views[i] = k, tuple(updated)
    if force_absorbed or (len(views) < 6 and rng.random() < .3):
        leaves = [k for k in children if children[k] is None]
        views.append(stub(int(rng.choice(leaves))))
    rng.shuffle(views)
    return views


def _outline(h, w, cuts, drawing, bg):
    a, b, c = cuts
    path = [(0, 0), (0, w-1-a), (a, w-1-a), (a, w-1),
            (h-1-b, w-1), (h-1-b, w-1-b), (h-1, w-1-b),
            (h-1, c), (h-1-c, c), (h-1-c, 0), (0, 0)]
    result = np.full((h, w), bg, np.int8)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        result[min(y, yy):max(y, yy)+1, x] = drawing
        result[yy, min(x, xx):max(x, xx)+1] = drawing
    return result


def _draw_view(view, stamps, axes, bg, drawing, rng, compact=False):
    key, parts = view
    if parts is None:
        h, w = (2, 2) if compact else tuple(int(v) for v in rng.integers(2, 5, 2))
        result = np.full((h, w), drawing, np.int8)
        if min(h, w) >= 3 and rng.random() < .5:
            result[-1, -1] = bg
    else:
        if parts:
            content = _compose([_draw_view(p, stamps, axes, bg, drawing, rng, compact) for p in parts],
                               axes[key], 1 if compact else int(rng.integers(1, 4)), bg)
            top, bottom, left, right = (3, 2, 3, 2) if compact else (
                int(rng.integers(3, 5)), int(rng.integers(2, 4)),
                int(rng.integers(3, 5)), int(rng.integers(2, 4)))
            h, w = content.shape[0]+top+bottom, content.shape[1]+left+right
        else:
            h, w = (5, 5) if compact else tuple(int(v) for v in rng.integers(5, 8, 2))
        # A corner step must leave a whole blank cell before the content.
        caps = (1, 1, 1) if not parts else (
            min(top, right)-2, min(bottom, right)-2, min(bottom, left)-2)
        cuts = tuple(int(rng.integers(0, cap+1)) for cap in caps)
        result = _outline(h, w, cuts, drawing, bg)
        if parts:
            result[top:top+content.shape[0], left:left+content.shape[1]] = content
    result[:2, :2] = stamps[key]
    return result


def _pack(parts, bg, rng):
    """Bounded shelf search; whole view rectangles remain separate and unnested."""
    n = len(parts)
    candidates = []
    # At most 64 * 2**5 small shelf layouts, never a pixel-level packing search.
    for _ in range(64):
        order = [int(v) for v in rng.permutation(n)]
        gaps = int(rng.integers(1, 3))
        for cuts in range(1 << (n-1)):
            rows = [[order[0]]]
            for j, i in enumerate(order[1:]):
                if cuts & (1 << j):
                    rows.append([])
                rows[-1].append(i)
            widths = [sum(parts[i].shape[1] for i in row)+gaps*(len(row)-1) for row in rows]
            heights = [max(parts[i].shape[0] for i in row) for row in rows]
            h, w = sum(heights)+gaps*(len(rows)-1), max(widths)
            if h <= 28 and w <= 28:
                candidates.append((rows, widths, heights, h, w, gaps))
        if len(candidates) >= 24:
            break
    if not candidates:
        raise ValueError("views do not fit a 30 by 30 canvas")
    rows, widths, heights, h, w, gaps = candidates[int(rng.integers(len(candidates)))]
    height = int(rng.integers(h+2, 31))
    width = int(rng.integers(w+2, 31))
    result = np.full((height, width), bg, np.int8)
    top = int(rng.integers(1, height-h))
    for row, rh, rw in zip(rows, heights, widths):
        left = int(rng.integers(1, width-rw))
        for i in row:
            ph, pw = parts[i].shape
            y = top+int(rng.integers(rh-ph+1))
            result[y:y+ph, left:left+pw] = parts[i]
            left += pw+gaps
        top += rh+gaps
    return result


def _make_pair(rng, mode, palette):
    for attempt in range(100):
        root, children, axes = _scene(rng, mode)
        bg, drawing = palette[:2]
        output = _canonical(root, children, axes, bg, drawing)
        if max(output.shape) > 29:
            continue
        accent_count = 2 if len(children) <= 6 and rng.random() < .4 else int(rng.integers(3, 5))
        accents = palette[2:2+accent_count]
        codes = [t for t in itertools.product(accents, repeat=3) if len(set(t)) >= 2]
        selected = rng.choice(len(codes), len(children), replace=False)
        stamps = {k: np.array([drawing, *codes[int(i)]], np.int8).reshape(2, 2)
                  for k, i in zip(children, selected)}
        views = _views(rng, root, children, force_order=mode == "order", force_absorbed=mode == "witness")
        compact = attempt >= 5 or rng.random() < .25
        try:
            parts = [_draw_view(v, stamps, axes, bg, drawing, rng, compact) for v in views]
            inp = _pack(parts, bg, rng)
            if rng.random() < .5:
                inp, output = inp.T.copy(), output.T.copy()
            _require(np.array_equal(solve(inp), output), "construction versus pixel inference mismatch")
        except ValueError:
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name+" outside allowed integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_palette = [int(v) for v in rng.permutation(10)]
    stable = bool(rng.integers(2))
    result = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = "test" if split == "test" else ("witness" if index == 0 else "order" if index == 1 else "train")
            for _ in range(16):
                palette = base_palette if stable else [int(v) for v in rng.permutation(10)]
                pair = _make_pair(rng, mode, palette)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
        rng.shuffle(result[split])
    return result


def _signature(root, children, axes):
    if children[root] is None:
        return "L"
    return (axes[root] if len(children[root]) > 1 else -1,
            tuple(_signature(c, children, axes) for c in children[root]))


def _depth(root, children):
    if children[root] is None:
        return 0
    return 1+max(_depth(c, children) for c in children[root])


def _check_output(out, info, model):
    bg, drawing = info["background"], info["drawing"]
    root, children, axes = model
    _require(np.array_equal(out, _canonical(root, children, axes, bg, drawing)), "incorrect answer")
    _require(set(np.unique(out)) == {bg, drawing}, "stamps should disappear")
    groups = _components(out != bg)
    expected_frames = sum(v is not None for v in children.values())
    expected_leaves = len(children)-expected_frames
    frames = leaves = 0
    for coords in groups:
        lo, hi = coords.min(0), coords.max(0)
        mask = np.zeros(out.shape, bool)
        mask[coords[:, 0], coords[:, 1]] = True
        hole = _holes(mask)
        if hole.any():
            frames += 1
            y, x = lo
            yy, xx = hi
            _require(len(coords) == 2*(yy-y+xx-x), "nonrectangular output frame")
            _require(np.all(mask[y, x:xx+1]) and np.all(mask[yy, x:xx+1])
                     and np.all(mask[y:yy+1, x]) and np.all(mask[y:yy+1, xx]), "open output frame")
            # The first interior row/column on every side must be a blank moat.
            _require(np.all(out[y+1, x+1:xx] == bg) and np.all(out[yy-1, x+1:xx] == bg)
                     and np.all(out[y+1:yy, x+1] == bg) and np.all(out[y+1:yy, xx-1] == bg), "frame moat")
        else:
            leaves += 1
            _require(len(coords) == 1, "blob did not collapse to a point")
    _require((frames, leaves) == (expected_frames, expected_leaves), "lost or duplicated identities")
    _require(np.all(out[0] == drawing) and np.all(out[-1] == drawing)
             and np.all(out[:, 0] == drawing) and np.all(out[:, -1] == drawing), "root crop")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read(inp)
            model = _stitch(info)
            _check_output(out, info, model)
            root, children, axes = model
            nf = sum(v is not None for v in children.values())
            _require(2 <= nf <= 4, "generated frame count")
            _require(len(info["roots"]) >= 2, "need multiple observations")
            _require(len(info["objects"]) > len(children), "identity merging is inactive")
            _require(sum(o["kind"] == "F" for o in info["objects"]) > nf, "contour identity must matter")
            _require(any(o["kind"] == "F" and
                         {info["objects"][j]["key"] for j in info["children"][i]} < set(children[o["key"]])
                         for i, o in enumerate(info["objects"])), "need an incomplete frame view")
            root_keys = [info["objects"][i]["key"] for i in info["roots"]]
            evidence[split].append(dict(depth=_depth(root, children), axes=set(axes.values()),
                                        order=len(root_keys) > len(set(root_keys))))
    _require(any(e["depth"] >= 2 and len(e["axes"]) == 2 for e in evidence["train"]), "mixed-axis training evidence")
    _require(any(e["order"] for e in evidence["train"]), "overlapping order views not demonstrated")
    _require(all(e["depth"] == 2 for e in evidence["train"]), "training depth contract")
    _require(all(e["depth"] >= 3 for e in evidence["test"]), "missing structural test shift")


def self_test():
    """42 streaming seeds, count boundaries, metamorphic and adversarial checks.

    No files, logs, retained puzzle corpus, workers, or recursive retries. Peak
    data is one puzzle plus small sets of shape/tree signatures (well under 1 MB).
    """
    rng = np.random.default_rng(72109)
    input_shapes, output_shapes, structures, depths, frame_counts, view_counts = (set() for _ in range(6))
    backgrounds, leaf_areas = set(), set()
    repeated_order = absorbed = irregular = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            first = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _read(inp)
                root, children, axes = _stitch(info)
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                structures.add(_signature(root, children, axes))
                depths.add(_depth(root, children))
                frame_counts.add(sum(v is not None for v in children.values()))
                view_counts.add(len(info["roots"]))
                backgrounds.add(info["background"])
                for obj in info["objects"]:
                    if obj["kind"] == "L":
                        leaf_areas.add(obj["area"])
                    else:
                        y, x = obj["lo"]
                        yy, xx = obj["hi"]
                        irregular += not (np.all(obj["mask"][y, x:xx+1]) and np.all(obj["mask"][yy, x:xx+1])
                                          and np.all(obj["mask"][y:yy+1, x]) and np.all(obj["mask"][y:yy+1, xx]))
                root_keys = [info["objects"][i]["key"] for i in info["roots"]]
                repeated_order += len(root_keys) > len(set(root_keys))
                absorbed += any(info["objects"][i]["kind"] == "L" for i in info["roots"])
                # Ignoring containment gives a single frame around every leaf.
                flat = {0: list(range(1, 1+sum(v is None for v in children.values())))}
                flat.update({k: None for k in flat[0]})
                _require(not np.array_equal(out, _canonical(0, flat, {0: 1}, info["background"], info["drawing"])), "A ablation ineffective")
        pair = puzzle["test"][0]
        permutation = rng.permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "recoloring")
        _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "transposition")
        if seed < 12:
            info = _read(pair["input"])
            _, _, axes = _stitch(info)

            def view(i):
                obj = info["objects"][i]
                return (obj["key"], None if obj["kind"] == "L" else
                        tuple(view(c) for c in info["layouts"][i][1]))

            stamps = {o["key"]: np.asarray(o["key"], np.int8).reshape(2, 2) for o in info["objects"]}
            views = [view(i) for i in info["roots"]]
            parts = [_draw_view(v, stamps, axes, info["background"], info["drawing"], rng, True) for v in views]
            redrawn = _pack(parts, info["background"], rng)
            _require(np.array_equal(solve(redrawn), pair["output"]), "independent metric redraw")
    _require(len(input_shapes) >= 25 and len(output_shapes) >= 20 and len(structures) >= 45, "structural variation")
    _require(depths >= {2, 3, 4} and frame_counts == {2, 3, 4} and len(view_counts) >= 3, "topology variation")
    _require(len(backgrounds) == 10 and len(leaf_areas) >= 4 and irregular >= 100, "geometry and role variation")
    _require(repeated_order >= 42 and absorbed >= 42, "overlap and absorption coverage")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            validate_generated_puzzle(generate_puzzle(123+train*10+test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (None, 1), (4, 1.5)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    again = generate_puzzle(0)
    _require(all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], again[s])
                 for k in ("input", "output")), "determinism")
    try:
        _unique_order([0, 1, 2], {(0, 2), (1, 2)})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous child ordering accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = int(damaged["test"][0]["input"][0, 0])
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged answer accepted")
    # A manually specified diagram, with a literal expected digit-grid answer.
    drawing, bg = 1, 0
    stamps = {i: np.asarray(t, np.int8).reshape(2, 2) for i, t in enumerate((
        (1, 2, 2, 3), (1, 2, 3, 2), (1, 3, 2, 2),
        (1, 2, 3, 3), (1, 3, 3, 2), (1, 3, 2, 3)))}
    views = [(0, ((1, ()), (2, None))), (1, ((3, None), (4, None))), (3, None)]
    axes = {0: 1, 1: 0}
    fixed_rng = np.random.default_rng(707)
    inp = _pack([_draw_view(v, stamps, axes, bg, drawing, fixed_rng, True) for v in views], bg, fixed_rng)
    golden = np.asarray([[int(c) for c in row] for row in (
        "11111111111", "10000000001", "10111110001", "10100010001",
        "10101010001", "10100010101", "10101010001", "10100010001",
        "10111110001", "10000000001", "11111111111")], np.int8)
    _require(np.array_equal(solve(inp), golden), "literal semantic fixture")
    # Keep the exact stamps, but move z from the outer frame into the detail.
    moved = [(0, ((1, ()),)), (1, ((3, None), (2, None), (4, None))), (3, None)]
    fixed_rng = np.random.default_rng(708)
    changed = _pack([_draw_view(v, stamps, axes, bg, drawing, fixed_rng, True) for v in moved], bg, fixed_rng)
    _require(not np.array_equal(solve(changed), golden), "containment intervention ineffective")
    # Keep every outline and every local containment relation fixed. Swap only
    # the stamps of two empty child frames; their unequal contents must swap.
    views = [(0, ((1, ()), (2, ()))), (1, ((3, None), (4, None))), (2, ((5, None),))]
    axes = {0: 1, 1: 0, 2: 1}
    fixed_rng = np.random.default_rng(709)
    inp = _pack([_draw_view(v, stamps, axes, bg, drawing, fixed_rng, True) for v in views], bg, fixed_rng)
    swapped = inp.copy()
    info = _read(inp)
    targets = [o for i, o in enumerate(info["objects"]) if info["parents"][i] is not None
               and o["key"] in (tuple(stamps[1].flat), tuple(stamps[2].flat))]
    _require(len(targets) == 2, "stamp intervention setup")
    for obj in targets:
        y, x = obj["lo"]
        replacement = stamps[2] if obj["key"] == tuple(stamps[1].flat) else stamps[1]
        swapped[y:y+2, x:x+2] = replacement
    _require(not np.array_equal(solve(inp), solve(swapped)), "identity intervention ineffective")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus all legal counts)")
