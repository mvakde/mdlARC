"""Generate ARC scenes that lose their outermost foreground objects."""

import hashlib
from collections import Counter, deque

import numpy as np


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (value is not None and (isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or not low <= value <= high)):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _size(box):
    return box[2] - box[0] + 1, box[3] - box[1] + 1


def _partition(box, wanted, rng, minimum=3, gap=1):
    slots = [box]
    while len(slots) < wanted:
        choices = []
        for index, slot in enumerate(slots):
            h, w = _size(slot)
            if h >= 2 * minimum + gap:
                choices.append((h * w, index, 0))
            if w >= 2 * minimum + gap:
                choices.append((h * w, index, 1))
        if not choices:
            break
        max_area = max(c[0] for c in choices)
        candidates = [c for c in choices if c[0] == max_area]
        _, index, axis = candidates[int(rng.integers(len(candidates)))]
        y0, x0, y1, x1 = slots.pop(index)
        length = (y1 - y0 + 1) if axis == 0 else (x1 - x0 + 1)
        first = int(rng.integers(minimum, length - gap - minimum + 1))
        if axis == 0:
            pieces = [(y0, x0, y0 + first - 1, x1),
                      (y0 + first + gap, x0, y1, x1)]
        else:
            pieces = [(y0, x0, y1, x0 + first - 1),
                      (y0, x0 + first + gap, y1, x1)]
        slots.extend(pieces)
    return slots


def _rectangle(box):
    y0, x0, y1, x1 = box
    return ({(y0, x) for x in range(x0, x1 + 1)}
            | {(y1, x) for x in range(x0, x1 + 1)}
            | {(y, x0) for y in range(y0, y1 + 1)}
            | {(y, x1) for y in range(y0, y1 + 1)})


def _diamond(box):
    y0, x0, y1, x1 = box
    r = (y1 - y0) // 2
    cy, cx = y0 + r, x0 + r
    return {(y, x) for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
            if abs(y - cy) + abs(x - cx) == r}


def _fitted_box(slot, kind, minimum, rng):
    y0, x0, y1, x1 = slot
    h, w = _size(slot)
    if kind == "diamond":
        max_r = (min(h, w) - 1) // 2
        if max_r < 3:
            kind = "rect"
        else:
            r = int(rng.integers(3, max_r + 1))
            side = 2 * r + 1
            top = int(rng.integers(y0, y1 - side + 2))
            left = int(rng.integers(x0, x1 - side + 2))
            return (top, left, top + side - 1, left + side - 1), kind
    if min(h, w) < minimum:
        raise ValueError("slot too small for enclosure")
    bh = int(rng.integers(minimum, h + 1))
    bw = int(rng.integers(minimum, w + 1))
    top = int(rng.integers(y0, y1 - bh + 2))
    left = int(rng.integers(x0, x1 - bw + 2))
    return (top, left, top + bh - 1, left + bw - 1), kind


def _inner_box(box, kind):
    y0, x0, y1, x1 = box
    if kind == "rect":
        return y0 + 2, x0 + 2, y1 - 2, x1 - 2
    r = (y1 - y0) // 2
    d = (r - 2) // 2
    cy, cx = y0 + r, x0 + r
    return cy - d, cx - d, cy + d, cx + d


def _leaf(slot, rng, requested=None):
    h, w = _size(slot)
    kinds = ["block", "point"]
    if max(h, w) >= 2:
        kinds.append("line")
    if h * w >= 3:
        kinds.extend(("poly", "poly"))
    if min(h, w) >= 3:
        kinds.append("hollow")
    kind = requested if requested in kinds else kinds[int(rng.integers(len(kinds)))]
    if kind == "point":
        shape = {(0, 0)}
    elif kind == "block":
        bh = int(rng.integers(1, min(h, 4) + 1))
        bw = int(rng.integers(1, min(w, 5) + 1))
        shape = {(y, x) for y in range(bh) for x in range(bw)}
    elif kind == "line":
        if (h >= w and h >= 2) or w == 1:
            shape = {(y, 0) for y in range(int(rng.integers(2, min(h, 5) + 1)))}
        else:
            shape = {(0, x) for x in range(int(rng.integers(2, min(w, 6) + 1)))}
    elif kind == "hollow":
        bh = int(rng.integers(3, min(h, 6) + 1))
        bw = int(rng.integers(3, min(w, 7) + 1))
        shape = _rectangle((0, 0, bh - 1, bw - 1))
    else:
        ph, pw = min(h, 5), min(w, 5)
        shape = {(int(rng.integers(ph)), int(rng.integers(pw)))}
        target = int(rng.integers(3, min(9, ph * pw) + 1))
        while len(shape) < target:
            frontier = {(y + dy, x + dx) for y, x in shape
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= y + dy < ph and 0 <= x + dx < pw
                        and (y + dy, x + dx) not in shape}
            if not frontier:
                break
            shape.add(sorted(frontier)[int(rng.integers(len(frontier)))])
        min_y = min(y for y, _ in shape)
        min_x = min(x for _, x in shape)
        shape = {(y - min_y, x - min_x) for y, x in shape}
    sh = 1 + max(y for y, _ in shape)
    sw = 1 + max(x for _, x in shape)
    top = int(rng.integers(slot[0], slot[2] - sh + 2))
    left = int(rng.integers(slot[1], slot[3] - sw + 2))
    return {(top + y, left + x) for y, x in shape}, kind


def _geometry(rng, mode):
    if mode == "rect":
        return "rect"
    if mode == "angle":
        return "diamond"
    return "diamond" if rng.random() < 0.45 else "rect"


def _add_leaf(objects, slot, parent, rng, requested=None):
    points, kind = _leaf(slot, rng, requested)
    objects.append({"points": points, "parent": parent, "kind": kind})


def _add_enclosure(objects, slot, parent, rng, mode, levels=0,
                   force_deep=False, frame=False):
    kind = "rect" if frame or force_deep else _geometry(rng, mode)
    if frame:
        box = slot
    else:
        minimum = 11 if force_deep else (7 if min(_size(slot)) >= 7 else 5)
        box, kind = _fitted_box(slot, kind, minimum, rng)
    points = _rectangle(box) if kind == "rect" else _diamond(box)
    index = len(objects)
    objects.append({"points": points, "parent": parent, "kind": kind})
    inner = _inner_box(box, kind)
    if force_deep:
        _add_enclosure(objects, inner, index, rng, mode, levels=0)
        return index
    ih, iw = _size(inner)
    if ih < 1 or iw < 1:
        raise ValueError("empty enclosure interior")
    if frame:
        wanted = int(rng.integers(2, 4))
    elif ih * iw >= 30:
        wanted = int(rng.integers(1, 4))
    elif ih * iw >= 12:
        wanted = int(rng.integers(1, 3))
    else:
        wanted = 1
    slots = _partition(inner, wanted, rng, minimum=2, gap=1)
    for child in slots:
        ch, cw = _size(child)
        nested = (min(ch, cw) >= 5 and
                  (frame or (levels > 0 and rng.random() < 0.55)))
        if nested:
            _add_enclosure(objects, child, index, rng, mode, levels - 1)
        else:
            _add_leaf(objects, child, index, rng)
    return index


def _special_slots(h, w, rng):
    gap = int(rng.integers(0, 3))
    axes = []
    if w >= 11 + gap + 4:
        axes.append(1)
    if h >= 11 + gap + 4:
        axes.append(0)
    axis = axes[int(rng.integers(len(axes)))]
    if axis == 1:
        width = int(rng.integers(11, w - gap - 4 + 1))
        large = (0, 0, h - 1, width - 1)
        small = (0, width + gap, h - 1, w - 1)
    else:
        height = int(rng.integers(11, h - gap - 4 + 1))
        large = (0, 0, height - 1, w - 1)
        small = (height + gap, 0, h - 1, w - 1)
    if rng.random() < 0.5:
        if axis == 1:
            large, small = ((0, w - width, h - 1, w - 1),
                            (0, 0, h - 1, w - width - gap - 1))
        else:
            large, small = ((h - height, 0, h - 1, w - 1),
                            (0, 0, h - height - gap - 1, w - 1))
    return large, small


def _adjacent(a, b):
    return any(abs(y - yy) <= 1 and abs(x - xx) <= 1
               for y, x in a for yy, xx in b)


def _color_scene(objects, rng, background, conflict):
    choices = [c for c in range(1, 10) if c != background]
    pool_size = int(rng.integers(3, min(8, len(choices)) + 1))
    pool = [int(v) for v in rng.choice(choices, size=pool_size, replace=False)]
    colors = [None] * len(objects)
    if conflict:
        root = next(i for i, o in enumerate(objects) if o["parent"] is None
                    and o["kind"] in ("rect", "diamond"))
        descendants = [i for i, o in enumerate(objects) if o["parent"] is not None]
        target = descendants[-1]
        colors[root] = colors[target] = pool[0]
    order = list(rng.permutation(len(objects)))
    for i in order:
        if colors[i] is not None:
            continue
        forbidden = {colors[j] for j in range(len(objects))
                     if colors[j] is not None
                     and _adjacent(objects[i]["points"], objects[j]["points"])}
        available = [c for c in pool if c not in forbidden]
        if not available:
            available = [c for c in choices if c not in forbidden]
        if not available:
            raise ValueError("no distinguishable color")
        colors[i] = available[int(rng.integers(len(available)))]
    return colors


def _components(g, background):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sy in range(h):
        for sx in range(w):
            if int(g[sy, sx]) == background or seen[sy, sx]:
                continue
            color = int(g[sy, sx])
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            pixels = []
            while q:
                y, x = q.popleft()
                pixels.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if (0 <= yy < h and 0 <= xx < w and
                                not seen[yy, xx] and int(g[yy, xx]) == color):
                            seen[yy, xx] = True
                            q.append((yy, xx))
            wall = np.zeros((h, w), dtype=bool)
            for y, x in pixels:
                wall[y, x] = True
            outside = np.zeros((h, w), dtype=bool)
            q = deque()
            for y in range(h):
                for x in (0, w - 1):
                    if not wall[y, x] and not outside[y, x]:
                        outside[y, x] = True
                        q.append((y, x))
            for x in range(w):
                for y in (0, h - 1):
                    if not wall[y, x] and not outside[y, x]:
                        outside[y, x] = True
                        q.append((y, x))
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and
                            not wall[yy, xx] and not outside[yy, xx]):
                        outside[yy, xx] = True
                        q.append((yy, xx))
            comps.append({"pixels": pixels, "color": color,
                          "inside": ~(wall | outside)})
    for j, comp in enumerate(comps):
        parents = []
        for i, outer in enumerate(comps):
            if i == j:
                continue
            flags = [bool(outer["inside"][y, x]) for y, x in comp["pixels"]]
            if any(flags) and not all(flags):
                raise ValueError("partial containment")
            if all(flags):
                parents.append(i)
        comp["parents"] = parents
    return comps


def _background(g):
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _apply_rule(g):
    background = _background(g)
    result = np.full_like(g, background)
    for comp in _components(g, background):
        if comp["parents"]:
            for y, x in comp["pixels"]:
                result[y, x] = comp["color"]
    return result


def _pair(rng, regime, teaching=False):
    if teaching:
        h = int(rng.integers(17, 25))
        w = int(rng.integers(23, 31))
    elif regime == "train":
        h = int(rng.integers(11, 23))
        w = int(rng.integers(12, 31))
    else:
        h = int(rng.integers(12, 29))
        w = int(rng.integers(12, 31))
    background = 0 if rng.random() < 0.62 else int(rng.integers(1, 10))
    mode = ("rect", "angle", "mixed")[int(rng.integers(3))]
    objects = []
    if teaching:
        large, small = _special_slots(h, w, rng)
        _add_enclosure(objects, large, None, rng, "rect", force_deep=True)
        if min(_size(small)) >= 3 and rng.random() < 0.45:
            box, _ = _fitted_box(small, "rect", 3, rng)
            objects.append({"points": _rectangle(box), "parent": None,
                            "kind": "empty_outline"})
        else:
            _add_leaf(objects, small, None, rng)
    elif rng.random() < (0.16 if regime == "train" else 0.28):
        _add_enclosure(objects, (0, 0, h - 1, w - 1), None, rng,
                       mode, levels=int(rng.integers(1, 3)), frame=True)
    else:
        wanted = int(rng.integers(1, 5))
        slots = _partition((0, 0, h - 1, w - 1), wanted, rng,
                           minimum=5, gap=int(rng.integers(0, 3)))
        group_count = int(rng.integers(1, min(3, len(slots)) + 1))
        indices = set(int(i) for i in rng.choice(len(slots), size=group_count,
                                                  replace=False))
        for i, slot in enumerate(slots):
            if i in indices:
                _add_enclosure(objects, slot, None, rng, mode,
                               levels=int(rng.integers(0, 3)))
            elif min(_size(slot)) >= 3 and rng.random() < 0.4:
                kind = "diamond" if mode == "angle" and min(_size(slot)) >= 7 else "rect"
                box, kind = _fitted_box(slot, kind, 3, rng)
                pts = _diamond(box) if kind == "diamond" else _rectangle(box)
                objects.append({"points": pts, "parent": None,
                                "kind": "empty_outline"})
            else:
                _add_leaf(objects, slot, None, rng)
    for i, obj in enumerate(objects):
        for j in range(i):
            if obj["points"] & objects[j]["points"]:
                raise ValueError("overlapping objects")
    colors = _color_scene(objects, rng, background, teaching)
    g = np.full((h, w), background, dtype=np.int8)
    for obj, color in zip(objects, colors):
        for y, x in obj["points"]:
            g[y, x] = color
    if np.count_nonzero(g == background) <= g.size // 2:
        raise ValueError("ambiguous background")
    comps = _components(g, background)
    if len(comps) != len(objects):
        raise ValueError("merged or split object")
    expected_roots = sum(o["parent"] is None for o in objects)
    if sum(not c["parents"] for c in comps) != expected_roots:
        raise ValueError("wrong containment")
    if not any(c["parents"] for c in comps):
        raise ValueError("no contained object")
    out = _apply_rule(g)
    return {"input": g, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    teaching_index = int(rng.integers(num_train))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(120):
                try:
                    pair = _pair(rng, regime,
                                 teaching=(regime == "train" and index == teaching_index))
                except ValueError:
                    continue
                inp = pair["input"]
                key = (inp.shape, inp.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(pair):
    h = hashlib.sha256()
    for role in ("input", "output"):
        a = np.asarray(pair[role], dtype=np.int8)
        h.update(bytes(a.shape))
        h.update(a.tobytes())
    return h.hexdigest()


_SOURCE_SIGNATURES = frozenset((
    "7e26f003380c89800ec9ab410ba02d8c54f6802c792145edee43671756e69f34",
    "11ca666ab71cbc4aaea7ee97b6da20c44e3152368c0a84ea5bccdb6c16d8dea2",
    "93d162389e46cb8c3cf8d9653759a564d2111aaf844634c1246b7569eaca39f4",
    "a0054edfeba7b46de1508f7b71bc45ca6d5575884a6e45ca335db86127c980cb",
    "02453e6b851033c96322ec6af91214d13978dd11e577c1cba3f92f7ebbf94878",
    "b26933d100f8b0081947b6cf5c8cf4cbe92f55a0c0ad744e74b532d4d4cd95d8",
))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_conflict = False
    train_depth_two = False
    for regime, series in (("train", train), ("test", test)):
        for pair in series:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            full_key = (key, out.tobytes())
            assert key not in seen_inputs and full_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(full_key)
            assert _signature(pair) not in _SOURCE_SIGNATURES
            bg = _background(inp)
            assert np.count_nonzero(inp == bg) > inp.size // 2
            assert np.array_equal(out, _apply_rule(inp))
            assert np.any(inp != out)
            comps = _components(inp, bg)
            assert any(c["parents"] for c in comps)
            assert any(not c["parents"] for c in comps)
            if regime == "train":
                root_colors = {c["color"] for c in comps if not c["parents"]}
                kept_colors = {c["color"] for c in comps if c["parents"]}
                train_conflict |= bool(root_colors & kept_colors)
                train_depth_two |= any(len(c["parents"]) >= 2 for c in comps)
    assert train_conflict and train_depth_two


def _same_puzzle(a, b):
    return (all(len(a[s]) == len(b[s]) for s in ("train", "test")) and
            all(np.array_equal(pa[k], pb[k]) for s in ("train", "test")
                for pa, pb in zip(a[s], b[s]) for k in ("input", "output")))


def _shape_flags(points):
    p = set(points)
    y0, y1 = min(y for y, _ in p), max(y for y, _ in p)
    x0, x1 = min(x for _, x in p), max(x for _, x in p)
    box = y0, x0, y1, x1
    rect = p == _rectangle(box)
    angle = (y1 - y0 == x1 - x0 and (y1 - y0) % 2 == 0
             and p == _diamond(box))
    filled = len(p) == (y1 - y0 + 1) * (x1 - x0 + 1)
    hollow = min(y1 - y0 + 1, x1 - x0 + 1) >= 3 and (rect or angle)
    return rect, angle, filled, hollow


def self_test() -> None:
    train_counts, test_counts, dimensions, palettes, structures = (set() for _ in range(5))
    root_counts, depths, train_conflicts = set(), set(), 0
    review = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        conflict = False
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp = pair["input"]
                dimensions.add(inp.shape)
                palettes.add(tuple(int(v) for v in np.unique(inp)))
                structures.add((inp.shape, (inp != _background(inp)).tobytes()))
                comps = _components(inp, _background(inp))
                root_counts.add(sum(not c["parents"] for c in comps))
                depths.add(max(len(c["parents"]) for c in comps))
                assert np.array_equal(pair["output"], _apply_rule(inp))
                bg = _background(inp)
                assert np.all(pair["output"][inp == bg] == bg)
                for comp in comps:
                    for y, x in comp["pixels"]:
                        assert int(pair["output"][y, x]) == (
                            int(inp[y, x]) if comp["parents"] else bg)
                roots = [i for i, c in enumerate(comps) if not c["parents"]]
                children = {i: [] for i in range(len(comps))}
                for j, c in enumerate(comps):
                    if c["parents"]:
                        parent = max(c["parents"],
                                     key=lambda i: len(comps[i]["parents"]))
                        children[parent].append(j)
                review["siblings"] += any(len(c) >= 2 for c in children.values())
                enclosing = [i for i in children if children[i]]
                kinds = set()
                for i in enclosing:
                    rect, angle, _, _ = _shape_flags(comps[i]["pixels"])
                    kinds.add("angle" if angle else "rect" if rect else "other")
                review["all_rect"] += kinds == {"rect"}
                review["all_angle"] += kinds == {"angle"}
                review["mixed"] += "rect" in kinds and "angle" in kinds
                review["no_standalone"] += all(children[i] for i in roots)
                for i in roots:
                    if children[i]:
                        continue
                    rect, angle, filled, hollow = _shape_flags(comps[i]["pixels"])
                    review["empty_outline"] += hollow
                    review["filled_standalone"] += filled
                    review["irregular_standalone"] += not (hollow or filled)
                for i, c in enumerate(comps):
                    if c["parents"] and not children[i]:
                        rect, angle, filled, _ = _shape_flags(c["pixels"])
                        review["irregular_kept"] += not (rect or angle or filled)
                review["compact"] += inp.shape[0] <= 15 and inp.shape[1] <= 18
                review["nonzero_background"] += bg != 0
                review["small_palette"] += len(np.unique(inp)) <= 5
                review["edge"] += any(y in (0, inp.shape[0] - 1)
                                      or x in (0, inp.shape[1] - 1)
                                      for c in comps for y, x in c["pixels"])
                review["frame"] += any(
                    not c["parents"] and _shape_flags(c["pixels"])[0]
                    and min(y for y, _ in c["pixels"]) == 0
                    and max(y for y, _ in c["pixels"]) == inp.shape[0] - 1
                    and min(x for _, x in c["pixels"]) == 0
                    and max(x for _, x in c["pixels"]) == inp.shape[1] - 1
                    for c in comps)
                touching = ((inp[:, :-1] != bg) & (inp[:, 1:] != bg)
                            & (inp[:, :-1] != inp[:, 1:]))
                touching_v = ((inp[:-1, :] != bg) & (inp[1:, :] != bg)
                              & (inp[:-1, :] != inp[1:, :]))
                review["adjacent_different_colors"] += bool(
                    np.any(touching) or np.any(touching_v))
                if regime == "train":
                    root = {c["color"] for c in comps if not c["parents"]}
                    kept = {c["color"] for c in comps if c["parents"]}
                    conflict |= bool(root & kept)
        train_conflicts += conflict
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 20 and len(palettes) >= 20
    assert len(structures) >= 100 and len(root_counts) >= 3
    assert {1, 2} <= depths and train_conflicts == 42
    assert {1, 2, 3, 4} <= root_counts
    assert all(review[name] > 0 for name in (
        "siblings", "all_rect", "all_angle", "mixed", "no_standalone",
        "empty_outline", "filled_standalone", "irregular_standalone",
        "irregular_kept", "compact", "nonzero_background",
        "small_palette", "edge", "frame", "adjacent_different_colors"))
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        a = generate_puzzle(54321, nt, ns)
        validate_generated_puzzle(a)
        assert _same_puzzle(a, generate_puzzle(54321, nt, ns))
        assert nt is None or len(a["train"]) == nt
        assert ns is None or len(a["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
