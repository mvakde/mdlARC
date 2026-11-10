"""Depth-dye shutters: an occluded laminar rectangle forest and a depth code.

Only standard-library modules and NumPy are used. Generation renders complete
scenes; solve reads pixels and reconstructs geometry without scene descriptors.
No files, execution traces, caches, or random state are retained by this module.
"""

import hashlib
import numbers
import numpy as np


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=True)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _in_box(p, box):
    t, b, l, r = box
    return t <= p[0] <= b and l <= p[1] <= r


def _contains(a, b):
    return a[0] < b[0] and b[1] < a[1] and a[2] < b[2] and b[3] < a[3]


def _laminar(a, b):
    disjoint = a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]
    return disjoint or _contains(a, b) or _contains(b, a)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], {start}
        while todo:
            y, x = todo.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    todo.append(q)
        result.append(part)
    return result


def _recover_boxes(points, mask):
    """Exact cover of visible components by uniquely completed rectangular loops.

    Each loop retains its bounding extrema. A rectangular occluder can cut a
    loop into at most two visible connected pieces. Laminarity and a one-cell
    orthogonal gap apply to complete perimeters, including their hidden parts.
    """
    parts = _components(points)
    if not 1 <= len(parts) <= 18:
        raise ValueError("missing or excessive loop fragments")
    candidates = []
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            visible = first | parts[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            edge = _perimeter(box)
            if {p for p in edge if not _in_box(p, mask)} == visible:
                candidates.append((frozenset((i, j)), box, edge))
    by_part = {i: [c for c in candidates if i in c[0]] for i in range(len(parts))}
    answers = []

    def search(left, boxes, used):
        if len(answers) > 1:
            return
        if not left:
            answers.append(tuple(sorted(boxes)))
            return
        i = min(left, key=lambda n: (sum(c[0] <= left for c in by_part[n]), n))
        for indices, box, edge in by_part[i]:
            if (indices <= left and not edge & _halo(used)
                    and all(_laminar(box, other) for other in boxes)):
                search(left - indices, boxes + [box], used | edge)

    search(set(range(len(parts))), [], set())
    if len(answers) != 1:
        raise ValueError("rectangle completion is not unique")
    return list(answers[0]), len(parts)


def _depth_map(shape, boxes):
    depth = np.zeros(shape, dtype=np.int8)
    for t, b, l, r in boxes:
        depth[t:b + 1, l:r + 1] += 1
    return depth


def _read_canonical(g):
    """Find a shutter with its coded jamb on the left, beginning at the top."""
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        return []
    result = []
    for frame in map(int, np.unique(g)):
        if frame == bg:
            continue
        points = [tuple(map(int, p)) for p in np.argwhere(g == frame)]
        t, b, l, r = _bbox(points)
        if b - t < 6 or r - l < 5:
            continue
        if not (np.all(g[t, l:r + 1] == frame)
                and np.all(g[b, l:r + 1] == frame)
                and np.all(g[t:b + 1, r] == frame)):
            continue
        column = g[t + 1:b, l]
        stops = np.flatnonzero(column == frame)
        if not len(stops):
            continue
        length = int(stops[0])
        if length < 3 or not np.all(column[length:] == frame):
            continue
        code = column[:length].copy()
        if code[-1] != code[-2] or bg in code or frame in code:
            continue
        first_row = g[t + 1, l:r]
        other = np.flatnonzero(first_row != code[0])
        if not len(other):
            continue
        jamb = int(other[0])
        if not 1 <= jamb <= 3 or r - l - jamb < 3:
            continue
        shutter = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
        shutter[0] = shutter[-1] = frame
        shutter[:, -1] = frame
        shutter[:, :jamb] = frame
        shutter[1:1 + length, :jamb] = code[:, None]
        if not np.array_equal(shutter, g[t:b + 1, l:r + 1]):
            continue
        outside = g.copy()
        outside[t:b + 1, l:r + 1] = bg
        inks = [int(c) for c in np.unique(outside) if c != bg]
        if len(inks) != 1 or inks[0] in code or inks[0] == frame:
            continue
        guide = inks[0]
        points = {tuple(map(int, p)) for p in np.argwhere(outside == guide)}
        mask = (t, b, l, r)
        try:
            boxes, pieces = _recover_boxes(points, mask)
        except ValueError:
            continue
        complete_depth = _depth_map(g.shape, boxes)
        depth = complete_depth[t + 1:b, l + jamb:r]
        if len(boxes) > 9 or int(complete_depth.max()) > len(code) - 2:
            continue
        result.append({"mask": mask, "jamb": jamb, "code": code,
                       "boxes": boxes, "pieces": pieces, "depth": depth,
                       "background": bg, "frame": frame, "guide": guide})
    return result


def _read_input(input_grid):
    g = _grid(input_grid)
    readings = []
    for turns in range(4):
        for scene in _read_canonical(np.rot90(g, turns)):
            scene["turns"] = turns
            readings.append(scene)
    if len(readings) != 1:
        raise ValueError("expected one geometrically identified depth-code shutter")
    return readings[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _read_input(input_grid)
    answer = scene["code"][scene["depth"]]
    return np.rot90(answer, -scene["turns"]).copy()


def _rand(rng, low, high):
    return int(rng.integers(low, high + 1))


def _chain(rng, box, depth):
    boxes = [box]
    if depth > 1:
        t, b, l, r = box
        minimum = 4 * (depth - 2) + 3
        hh, ww = _rand(rng, minimum, b - t - 3), _rand(rng, minimum, r - l - 3)
        yy, xx = _rand(rng, t + 2, b - hh - 1), _rand(rng, l + 2, r - ww - 1)
        boxes += _chain(rng, (yy, yy + hh - 1, xx, xx + ww - 1), depth - 1)
    return boxes


def _random_boxes(rng, h, w, depth, extras):
    minimum = 4 * (depth - 1) + 5
    hh, ww = _rand(rng, minimum, h - 2), _rand(rng, minimum, w - 2)
    yy, xx = _rand(rng, 1, h - hh - 1), _rand(rng, 1, w - ww - 1)
    boxes = _chain(rng, (yy, yy + hh - 1, xx, xx + ww - 1), depth)
    edges = set().union(*map(_perimeter, boxes))
    # New siblings or roots are accepted by geometry, not by a layout template.
    for _ in range(extras):
        for _ in range(45):
            hh, ww = _rand(rng, 3, min(10, h - 2)), _rand(rng, 3, min(10, w - 2))
            yy, xx = _rand(rng, 1, h - hh - 1), _rand(rng, 1, w - ww - 1)
            box = (yy, yy + hh - 1, xx, xx + ww - 1)
            edge = _perimeter(box)
            if not all(_laminar(box, a) for a in boxes) or edge & _halo(edges):
                continue
            proposed = boxes + [box]
            if int(_depth_map((h, w), proposed).max()) != depth:
                continue
            boxes, edges = proposed, edges | edge
            break
    return boxes


def _geometry_ok(shape, boxes, mask, jamb, depth):
    t, b, l, r = mask
    for box in boxes:
        visible = {p for p in _perimeter(box) if not _in_box(p, mask)}
        if not visible or _bbox(visible) != box:
            return False
    inside = _depth_map(shape, boxes)[t + 1:b, l + jamb:r]
    # Every code entry is exercised; the innermost region is not a lone pixel.
    if set(map(int, np.unique(inside))) != set(range(depth + 1)):
        return False
    if any(np.count_nonzero(inside == d) < 2 for d in range(depth + 1)):
        return False
    return True


def _sample_geometry(rng, depth, require_split=False, extras=1, require_roots=False):
    """Bounded random laminar forests, with a constructive corner fallback."""
    if require_roots:
        # Construct the pedagogical witness directly instead of repeatedly
        # sampling a large root that leaves no room for a second chamber.
        assert depth == 2
        h, w = _rand(rng, 24, 30), _rand(rng, 24, 30)
        jamb = _rand(rng, 1, 3)
        t, l, b, r = _rand(rng, 5, 8), _rand(rng, 5, 8), h - 3, w - 3
        outer = (_rand(rng, 1, 2), _rand(rng, t + 4, b - 4),
                 _rand(rng, 1, 2), _rand(rng, l + jamb + 4, r - 4))
        inner = (_rand(rng, outer[0] + 2, t - 1), _rand(rng, t + 2, outer[1] - 2),
                 _rand(rng, outer[2] + 2, l - 1), _rand(rng, l + jamb + 1, outer[3] - 2))
        other = (_rand(rng, outer[1] + 2, b - 2), _rand(rng, b + 1, h - 2),
                 _rand(rng, outer[3] + 2, r - 2), _rand(rng, r + 1, w - 2))
        boxes, mask = [outer, inner, other], (t, b, l, r)
        assert _geometry_ok((h, w), boxes, mask, jamb, depth)
        return (h, w), boxes, mask, jamb
    for _ in range(65):
        low = max(20, 4 * depth + 10)
        h, w = _rand(rng, low, 30), _rand(rng, low, 30)
        boxes = _random_boxes(rng, h, w, depth, extras)
        for _ in range(22):
            jamb = _rand(rng, 1, 3)
            oh = _rand(rng, depth + 4, min(21, h - 4))
            ow = _rand(rng, 6, min(21, w - jamb - 3))
            t, l = _rand(rng, 1, h - oh - 3), _rand(rng, 1, w - ow - jamb - 2)
            mask = (t, t + oh + 1, l, l + jamb + ow)
            if not _geometry_ok((h, w), boxes, mask, jamb, depth):
                continue
            visible = set().union(*map(_perimeter, boxes))
            visible = {p for p in visible if not _in_box(p, mask)}
            if require_split and len(_components(visible)) <= len(boxes):
                continue
            try:
                recovered, _ = _recover_boxes(visible, mask)
            except ValueError:
                continue
            if sorted(recovered) == sorted(boxes):
                return (h, w), boxes, mask, jamb
    # The fallback also has two fragment components per loop when requested.
    # Coordinate room for depth <= 4 is guaranteed on a 30x30 canvas.
    if require_split:
        boxes = [(1 + 2 * i, 28 - 2 * i, 3 + 2 * i, 26 - 2 * i)
                 for i in range(depth)]
        jamb = 1
        mask = (12, 21, 1, 28)
    else:
        t = 2 * depth + 1
        boxes = [(1 + 2 * i, 4 * depth + 3 - 2 * i,
                  1 + 2 * i, 4 * depth + 3 - 2 * i) for i in range(depth)]
        if extras:
            start = 4 * depth + 6
            boxes.append((start, 28, start, 28))
        jamb, mask = 1, (t, 27, t, 27)
    assert _geometry_ok((30, 30), boxes, mask, jamb, depth)
    return (30, 30), boxes, mask, jamb


def _code(rng, palette, depth, demo):
    if demo == 0:
        layers = list(palette[:3])  # three different depth labels
    elif demo == 1:
        layers = [palette[0], palette[0], palette[1]]  # preserve repeated depth
    else:
        count = _rand(rng, 2, min(len(palette), depth + 1))
        layers = list(palette[:count])
        layers += [palette[_rand(rng, 0, count - 1)] for _ in range(depth + 1 - count)]
        rng.shuffle(layers)
    return np.asarray(layers + [layers[-1]], dtype=np.int8)


def _render_pair(rng, background, depth, demo, require_split, extras, turns):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    frame, guide = palette[:2]
    code = _code(rng, palette[2:], depth, demo)
    shape, boxes, mask, jamb = _sample_geometry(rng, depth, require_split, extras,
                                              require_roots=demo == 0)
    g = np.full(shape, background, dtype=np.int8)
    for box in boxes:
        for p in _perimeter(box):
            g[p] = guide
    # Truth is a complete painted scene rendered BEFORE the shutter is added.
    # Paint outer regions first, then overwrite with deeper nested regions.
    painted = np.full(shape, code[0], dtype=np.int8)
    levels = [(1 + sum(_contains(a, box) for a in boxes), box) for box in boxes]
    for level, (y0, y1, x0, x1) in sorted(levels):
        painted[y0:y1 + 1, x0:x1 + 1] = code[level]
    t, b, l, r = mask
    answer = painted[t + 1:b, l + jamb:r].copy()
    g[t:b + 1, l:r + 1] = background
    g[t, l:r + 1] = g[b, l:r + 1] = frame
    g[t:b + 1, r] = frame
    g[t:b + 1, l:l + jamb] = frame
    g[t + 1:t + len(code) + 1, l:l + jamb] = code[:, None]
    return {"input": np.rot90(g, turns).copy(), "output": np.rot90(answer, turns).copy()}


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen = {"train": [], "test": []}, set()
    initial_turn = _rand(rng, 0, 3)
    # The first two demonstrations suffice even for the minimum count request.
    # Their code semantics and orientation are visibly encoded, never latent.
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            ordinal = len(seen)
            demo = index if split == "train" and index < 2 else 2
            depth = 2 if demo < 2 else (_rand(rng, 1, 2) if split == "train"
                                       else _rand(rng, 3, 4))
            require_split = demo == 1
            extras = 1 if demo == 0 else _rand(rng, 1, 2)
            turns = initial_turn if demo == 0 else ((initial_turn + 1) % 4 if demo == 1
                                                   else _rand(rng, 0, 3))
            for _ in range(12):
                pair = _render_pair(rng, backgrounds[ordinal % 10], depth, demo,
                                    require_split, extras, turns)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    scene = _read_input(pair["input"])
    depth, boxes, code = scene["depth"], scene["boxes"], scene["code"]
    roots = sum(not any(_contains(a, b) for a in boxes) for b in boxes)
    parents = [sum(_contains(a, b) and not any(_contains(a, c) and _contains(c, b)
                                            for c in boxes) for b in boxes) for a in boxes]
    nonconcentric = any(_contains(a, b) and (a[0] + a[1], a[2] + a[3])
                        != (b[0] + b[1], b[2] + b[3]) for a in boxes for b in boxes)
    norm = {}
    structural = np.asarray(pair["input"])
    signature = bytes(norm.setdefault(int(c), len(norm)) for c in structural.flat)
    return {"depth": len(code) - 2, "loops": len(boxes), "roots": roots,
            "branching": max(parents, default=0) > 1, "split": scene["pieces"] > len(boxes),
            "nonconcentric": nonconcentric, "jamb": scene["jamb"],
            "orientation": scene["turns"], "background": scene["background"],
            "code_palette": len(set(map(int, code))),
            "repeated_depth": bool(np.any(code[1:-1] == code[:-2])),
            "return": len(set(map(int, code[:-1]))) < len(code) - 1,
            "levels_seen": len(np.unique(depth)),
            "structure": (structural.shape, hashlib.sha256(signature).hexdigest())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _read_input(inp)
            depth, code = scene["depth"], scene["code"]
            assert set(map(int, np.unique(depth))) == set(range(len(code) - 1))
            assert all(np.count_nonzero(depth == d) >= 2 for d in range(len(code) - 1))
            assert np.array_equal(out, np.rot90(code[depth], -scene["turns"]))
            assert scene["guide"] not in out and scene["frame"] not in out
            assert set(map(int, np.unique(out))) == set(map(int, code))
            assert not np.array_equal(inp, out)
            features[split].append(_features(pair))
    first, second = [_read_input(p["input"]) for p in puzzle["train"][:2]]
    assert len(set(map(int, first["code"][:-1]))) == 3
    t, b, l, r = first["mask"]
    roots = [box for box in first["boxes"]
             if not any(_contains(a, box) for a in first["boxes"])]
    assert sum(max(0, min(b - 1, box[1]) - max(t + 1, box[0]) + 1)
               * max(0, min(r - 1, box[3]) - max(l + first["jamb"], box[2]) + 1) >= 2
               for box in roots) >= 2
    assert second["code"][0] == second["code"][1] != second["code"][2]
    assert second["pieces"] > len(second["boxes"])
    assert first["turns"] != second["turns"]
    assert all(f["depth"] > max(t["depth"] for t in features["train"])
               for f in features["test"])
    # Explicit concept ablations: constant fill ignores B; flattening every
    # positive depth to depth one ignores A's ordered nested-layer indexing.
    for scene in (first, second):
        real = scene["code"][scene["depth"]]
        without_b = np.full(real.shape, scene["code"][0])
        without_a = scene["code"][np.minimum(scene["depth"], 1)]
        assert np.any(real != without_b) and np.any(real != without_a)
    # Additional plausible shortcuts must be contradicted by demonstrations.
    assert np.any(first["code"][:-1][::-1][first["depth"]]
                  != first["code"][first["depth"]])
    collapsed = np.array([second["code"][0], second["code"][2], second["code"][2]])
    assert np.any(collapsed[second["depth"]] != second["code"][second["depth"]])
    return None


def self_test():
    """42 seeds, all legal counts, corruption tests; no retained grid corpus."""
    shapes, out_shapes, structures, colors, orientations, jambs = set(), set(), set(), set(), set(), set()
    depths, loops, roots, puzzles = set(), set(), set(), set()
    split_count = nonconcentric = branching = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                f = _features(pair)
                structures.add(f["structure"])
                shapes.add(pair["input"].shape)
                out_shapes.add(pair["output"].shape)
                colors.update(map(int, np.unique(pair["input"])))
                orientations.add(f["orientation"])
                jambs.add(f["jamb"])
                depths.add(f["depth"])
                loops.add(f["loops"])
                roots.add(f["roots"])
                split_count += f["split"]
                nonconcentric += f["nonconcentric"]
                branching += f["branching"]
                for grid in pair.values():
                    digest.update(repr(grid.shape).encode())
                    digest.update(grid.tobytes())
        puzzles.add(digest.hexdigest())
    assert len(puzzles) == 42 and len(structures) >= 160
    assert len(shapes) >= 20 and len(out_shapes) >= 35
    assert colors == set(range(10)) and orientations == set(range(4)) and jambs == {1, 2, 3}
    assert depths == {1, 2, 3, 4} and len(loops) >= 4 and max(roots) >= 2
    assert split_count >= 42 and nonconcentric >= 80 and branching >= 1
    for a in range(2, 11):
        for b in range(1, min(a, 5)):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(987654321, 6, 3), generate_puzzle(987654321, 6, 3)
    assert all(_key(x[k]) == _key(y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    bad = generate_puzzle(314159, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "structures": len(structures),
            "input_shapes": len(shapes), "output_shapes": len(out_shapes),
            "depths": sorted(depths), "loop_counts": sorted(loops),
            "root_counts": sorted(roots), "split_pairs": split_count,
            "nonconcentric_pairs": nonconcentric, "branching_pairs": branching}
