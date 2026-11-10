"""Room genealogy: turn a nested partition map into a compact spatial tree.

No parent generator is imported. Generation composes clean maps and expected
diagrams separately; solve recovers walls, rooms and containment from pixels.
Only standard library and NumPy are required. Nothing is written by this module.
"""

from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo, cells = [(y, x)], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _rooms(mask):
    """Actual bounded four-connected faces of this color alone."""
    free = np.pad(~mask, 1, constant_values=True)
    result = []
    for cells in _components(free):
        lo, hi = cells.min(0), cells.max(0)
        if np.any(lo == 0) or np.any(hi == np.array(free.shape)-1):
            continue
        assert len(cells) == int(np.prod(hi-lo+1)), "nonrectangular chamber"
        result.append(tuple(map(int, (* (lo-1), * (hi-1)))))
    return sorted(result)


def _perimeter(mask, box):
    t, l, b, r = box
    mask[t, l:r+1] = mask[b, l:r+1] = True
    mask[t:b+1, l] = mask[t:b+1, r] = True


def _recover(grid, color, background):
    """Bridge only occupied gaps of length one or two, with same-color ends.

    Room interiors are at least three cells wide/high, so a background noise
    patch cannot complete a wall across an intentionally open room.
    """
    mask = grid == color
    if mask.sum() < 12:
        return None
    for axis in (0, 1):
        m = mask if axis == 1 else mask.T
        g = grid if axis == 1 else grid.T
        for row in range(m.shape[0]):
            positions = np.flatnonzero(m[row])
            for a, b in zip(positions, positions[1:]):
                if 2 <= b-a <= 3 and np.all(g[row, a+1:b] != background):
                    m[row, a:b+1] = True
    cells = np.argwhere(mask)
    lo, hi = cells.min(0), cells.max(0)
    t, l, b, r = map(int, (*lo, *hi))
    if b-t < 4 or r-l < 4:
        return None
    if not (mask[t, l:r+1].all() and mask[b, l:r+1].all()
            and mask[t:b+1, l].all() and mask[t:b+1, r].all()):
        return None
    try:
        rooms = _rooms(mask[t:b+1, l:r+1])
        rooms = [(a+t, c+l, d+t, e+l) for a, c, d, e in rooms]
        assert rooms and all(d-a >= 2 and e-c >= 2 for a, c, d, e in rooms)
        rebuilt = np.zeros_like(mask)
        for a, c, d, e in rooms:
            _perimeter(rebuilt, (a-1, c-1, d+1, e+1))
        assert np.array_equal(mask, rebuilt), "unexplained wall pixels"
        assert len(_components(mask)) == 1
        layout = _slice_layout(mask, (t, l, b, r), rooms)
    except AssertionError:
        return None
    return dict(color=int(color), mask=mask, box=(t, l, b, r),
                rooms=rooms, layout=layout, repaired=int(np.sum(mask & (grid != color))))


def _slice_layout(mask, box, rooms):
    """Recover maximal full-span cuts; their axis must be unambiguous."""
    t, l, b, r = box
    hs = [y for y in range(t+1, b) if mask[y, l:r+1].all()]
    vs = [x for x in range(l+1, r) if mask[t:b+1, x].all()]
    assert not (hs and vs), "crossed full-span cuts have no unique grouping"
    if not hs and not vs:
        room = (t+1, l+1, b-1, r-1)
        assert room in rooms
        return ("R", rooms.index(room))
    axis = 0 if hs else 1
    cuts = [t, *hs, b] if hs else [l, *vs, r]
    children = []
    for a, z in zip(cuts, cuts[1:]):
        subbox = (a, l, z, r) if axis == 0 else (t, a, b, z)
        children.append(_slice_layout(mask, subbox, rooms))
    return ("G", axis, tuple(children))


def _inside(box, room):
    t, l, b, r = box
    a, c, d, e = room
    return a < t and c < l and b < d and r < e


def _decode(grid):
    values, frequencies = np.unique(grid, return_counts=True)
    assert len(values) >= 3
    order = np.argsort(frequencies)
    assert frequencies[order[-1]] > frequencies[order[-2]], "ambiguous background"
    bg = int(values[order[-1]])
    nets, noises = [], []
    for c in values:
        if c == bg:
            continue
        net = _recover(grid, int(c), bg)
        if net is None:
            noises.append(int(c))
        else:
            nets.append(net)
    assert len(noises) == 1 and nets, "one visible noise role is required"
    noise = noises[0]
    parents = [None] * len(nets)
    occupancy = {}
    for i, child in enumerate(nets):
        containers = []
        for j, parent in enumerate(nets):
            if i == j:
                continue
            assert not np.any(child["mask"] & parent["mask"]), "walls cross"
            for k, room in enumerate(parent["rooms"]):
                if _inside(child["box"], room):
                    a, c, d, e = room
                    containers.append(((d-a+1)*(e-c+1), j, k))
        if containers:
            _, j, k = min(containers)
            assert (j, k) not in occupancy, "multiple immediate children in one room"
            occupancy[j, k] = i
            parents[i] = j
    roots = [i for i, p in enumerate(parents) if p is None]
    assert len(roots) == 1, "one rooted building"
    root = roots[0]
    # All other structures belong to a room of this one root, never its wall.
    for i, n in enumerate(nets):
        depth, p = 1, parents[i]
        while p is not None:
            depth += 1
            assert depth <= len(nets), "containment cycle"
            p = parents[p]
        n["depth"] = depth
    return dict(background=bg, noise=noise, networks=nets, parents=parents,
                occupancy=occupancy, root=root)


def _join(parts, axis, background):
    shape = [max(p.shape[k] for p in parts) for k in (0, 1)]
    shape[axis] = sum(p.shape[axis] for p in parts)+len(parts)-1
    out = np.full(shape, background, dtype=np.int8)
    position = 0
    for part in parts:
        y, x = (position, 0) if axis == 0 else (0, position)
        h, w = part.shape
        out[y:y+h, x:x+w] = part
        position += part.shape[axis]+1
    return out


def _frame(content, color, background):
    out = np.pad(content, 2, constant_values=background)
    out[0, :] = out[-1, :] = color
    out[:, 0] = out[:, -1] = color
    return out.astype(np.int8)


def _normal_form(info):
    bg = info["background"]

    def network(i):
        net = info["networks"][i]

        def layout(node):
            if node[0] == "R":
                child = info["occupancy"].get((i, node[1]))
                return (np.array([[net["color"]]], dtype=np.int8)
                        if child is None else network(child))
            return _join([layout(c) for c in node[2]], node[1], bg)

        return _frame(layout(net["layout"]), net["color"], bg)

    return network(info["root"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution. No seed, examples, cache or generation metadata."""
    grid = _grid(input_grid)
    answer = _normal_form(_decode(grid))
    assert max(answer.shape) <= 30
    return answer


# Generation grammar: a network owns a slicing layout, whose rooms can contain
# another network. Spatial groups do not create containment frames.
def _leaf(child=None):
    return ("R", child)


def _group(rng, leaves, axis=None):
    if len(leaves) == 1:
        return leaves[0]
    axis = int(rng.integers(2)) if axis is None else axis
    if len(leaves) == 2 or rng.random() < .38:
        return ("G", axis, tuple(leaves))
    cut = int(rng.integers(1, len(leaves)))
    return ("G", axis, (_group(rng, leaves[:cut], 1-axis),
                         _group(rng, leaves[cut:], 1-axis)))


def _net(rng, count, children=(), mixed=False):
    leaves = [_leaf(c) for c in children]+[_leaf()]*(count-len(children))
    rng.shuffle(leaves)
    if mixed and count >= 3:
        axis = int(rng.integers(2))
        cut = int(rng.integers(1, count))
        return ("G", axis, (_group(rng, leaves[:cut], 1-axis),
                             _group(rng, leaves[cut:], 1-axis)))
    return _group(rng, leaves)


def _scene(rng, kind):
    if kind == "partition":
        child = _net(rng, int(rng.integers(2, 4)))
        return _net(rng, int(rng.integers(3, 5)), [child], mixed=True)
    if kind == "substitution":
        small = _net(rng, 1)
        large = _net(rng, int(rng.integers(2, 4)))
        return _net(rng, int(rng.integers(2, 4)), [small, large])
    if kind == "test":
        inner = _net(rng, int(rng.integers(1, 4)))
        middle = _net(rng, int(rng.integers(1, 4)), [inner], mixed=rng.random() < .6)
        children = [middle]
        if rng.random() < .4:
            children.append(_net(rng, int(rng.integers(1, 3))))
        return _net(rng, int(rng.integers(max(2, len(children)), 6)), children,
                    mixed=rng.random() < .8)
    count = int(rng.integers(1, 6))
    children = []
    if rng.random() < .7:
        for _ in range(int(rng.integers(1, min(3, count)+1))):
            children.append(_net(rng, int(rng.integers(1, 5)), mixed=rng.random() < .5))
    return _net(rng, count, children, mixed=rng.random() < .65)


def _network_count(node):
    def visit(layout):
        if layout[0] == "R":
            return 0 if layout[1] is None else _network_count(layout[1])
        return sum(visit(c) for c in layout[2])
    return 1+visit(node)


def _construct(node, rng, colors, background, compact):
    """Render a clean metric map and a separate abstract expected diagram."""
    color = next(colors)

    def plan(layout):
        if layout[0] == "R":
            if layout[1] is None:
                h, w = map(int, rng.integers(5, 7 if compact else 10, size=2))
                return dict(kind="R", minimum=(h, w), child=None,
                            expected=np.array([[color]], dtype=np.int8))
            child, expected = _construct(layout[1], rng, colors, background, compact)
            gap = [int(rng.integers(1, 2 if compact else 3)) for _ in range(2)]
            minimum = tuple(child.shape[k]+2*gap[k]+2 for k in (0, 1))
            return dict(kind="R", minimum=minimum, child=child, expected=expected)
        parts = [plan(c) for c in layout[2]]
        axis = layout[1]
        minimum = [max(p["minimum"][k] for p in parts) for k in (0, 1)]
        minimum[axis] = sum(p["minimum"][axis]-1 for p in parts)+1
        return dict(kind="G", axis=axis, parts=parts, minimum=tuple(minimum),
                    expected=_join([p["expected"] for p in parts], axis, background))

    planned = plan(node)
    h, w = planned["minimum"]
    if max(h, w) > 30:
        raise ValueError("metric map does not fit")
    result = np.full((h, w), background, dtype=np.int8)

    def paint(p, box):
        t, l, b, r = box
        if p["kind"] == "R":
            result[t, l:r+1] = result[b, l:r+1] = color
            result[t:b+1, l] = result[t:b+1, r] = color
            if p["child"] is not None:
                tile = p["child"]
                yy = int(rng.integers(t+2, b-tile.shape[0]))
                xx = int(rng.integers(l+2, r-tile.shape[1]))
                result[yy:yy+tile.shape[0], xx:xx+tile.shape[1]] = tile
            return
        axis, parts = p["axis"], p["parts"]
        span = (b-t if axis == 0 else r-l)
        sizes = np.array([q["minimum"][axis]-1 for q in parts])
        for _ in range(span-int(sizes.sum())):
            sizes[int(rng.integers(len(parts)))] += 1
        pos = t if axis == 0 else l
        for q, size in zip(parts, sizes):
            end = pos+int(size)
            paint(q, (pos, l, end, r) if axis == 0 else (t, pos, b, end))
            pos = end

    paint(planned, (0, 0, h-1, w-1))
    return result, _frame(planned["expected"], color, background)


def _damage(clean, rng, background, noise, required):
    grid = clean.copy()
    for color in np.unique(clean):
        if color == background:
            continue
        mask = clean == color
        horizontal = np.zeros_like(mask)
        vertical = np.zeros_like(mask)
        horizontal[:, 1:-1] = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
        vertical[1:-1] = mask[:-2] & mask[1:-1] & mask[2:]
        # Only straight, nonjunction sites; corner and T endpoints remain clear.
        safe = (horizontal ^ vertical)
        safe[1:] &= ~((horizontal & vertical)[:-1])
        safe[:-1] &= ~((horizontal & vertical)[1:])
        safe[:, 1:] &= ~((horizontal & vertical)[:, :-1])
        safe[:, :-1] &= ~((horizontal & vertical)[:, 1:])
        sites = np.argwhere(safe)
        if not len(sites):
            continue
        amount = int(rng.integers(1 if required else 0, min(5, len(sites))+1))
        for index in rng.permutation(len(sites))[:amount]:
            y, x = map(int, sites[index])
            # Keep each damaged run of length at most two and bracketed.
            candidate = grid.copy()
            candidate[y, x] = noise
            recovered = _recover(candidate, int(color), background)
            if recovered is not None and np.array_equal(recovered["mask"], mask):
                grid = candidate
    # Scatter only a few irrelevant noise marks. No reserved border cue.
    sites = np.argwhere(clean == background)
    amount = min(len(sites), int(rng.integers(1, 9)))
    for i in rng.choice(len(sites), amount, replace=False):
        y, x = sites[i]
        grid[y, x] = noise
    return grid


def _make_pair(rng, kind, palette):
    background, noise = palette[:2]
    for attempt in range(120):
        scene = _scene(rng, kind)
        number = _network_count(scene)
        if number > 7:
            continue
        structural = list(palette[2:2+number])
        try:
            clean, expected = _construct(scene, rng, iter(structural), background, compact=attempt >= 5)
        except ValueError:
            continue
        if max(expected.shape) > 30:
            continue
        # Transpose/reflection on the metric map changes the visible layout;
        # keep expected construction independent by transforming the grammar
        # only through sampled split axes and room order above.
        h, w = clean.shape
        height = int(rng.integers(h, min(30, h+5)+1))
        width = int(rng.integers(w, min(30, w+5)+1))
        yy, xx = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
        canvas = np.full((height, width), background, dtype=np.int8)
        canvas[yy:yy+h, xx:xx+w] = clean
        inp = _damage(canvas, rng, background, noise, required=kind == "partition")
        values, frequencies = np.unique(inp, return_counts=True)
        if int(values[np.argmax(frequencies)]) != background:
            continue
        try:
            info = _decode(inp)
            assert len(info["networks"]) == number
            for n in info["networks"]:
                assert np.array_equal(n["mask"], canvas == n["color"])
            assert np.array_equal(_normal_form(info), expected)
            features = _features(info)
            if kind == "partition":
                assert features["partial"] and features["repaired"] > 0
            if kind == "substitution":
                assert features["max_children"] >= 2
        except AssertionError:
            continue
        return {"input": inp, "output": expected}
    raise RuntimeError("bounded room-genealogy construction exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.permutation(10)))
    stable_palette = bool(rng.random() < .25)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = ("partition" if i == 0 else "substitution" if i == 1 else "ordinary") if split == "train" else "test"
            for _ in range(8):
                palette = base_palette if stable_palette else tuple(map(int, rng.permutation(10)))
                pair = _make_pair(rng, kind, palette)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate inputs after bounded retries")
        rng.shuffle(puzzle[split])
    return puzzle


def _features(info):
    nets = info["networks"]
    partial = False
    signatures = []
    for n in nets:
        t, l, b, r = n["box"]
        m = n["mask"]
        # Partial walls create degree-three junctions strictly inside a frame.
        degree = np.zeros(m.shape, dtype=int)
        degree[1:] += m[:-1]
        degree[:-1] += m[1:]
        degree[:, 1:] += m[:, :-1]
        degree[:, :-1] += m[:, 1:]
        partial |= bool(np.any(m[t+1:b, l+1:r] & (degree[t+1:b, l+1:r] == 3)))
        signatures.append(_layout_signature(n["layout"]))
    child_counts = [sum(p == i for p in info["parents"]) for i in range(len(nets))]
    empty = sum(len(n["rooms"]) for n in nets)-(len(nets)-1)
    return dict(networks=len(nets), depth=max(n["depth"] for n in nets),
                rooms=tuple(sorted(len(n["rooms"]) for n in nets)),
                empty=empty, partial=partial, max_children=max(child_counts),
                repaired=sum(n["repaired"] for n in nets),
                layouts=tuple(signatures))


def _layout_signature(layout):
    return "R" if layout[0] == "R" else (layout[1], tuple(_layout_signature(c) for c in layout[2]))


def _tree_signature(info):
    """Geometry- and palette-free structure, retaining occupied room addresses."""
    def network(i):
        def layout(node):
            if node[0] == "R":
                child = info["occupancy"].get((i, node[1]))
                return ("E",) if child is None else network(child)
            return (node[1], tuple(layout(c) for c in node[2]))
        return ("N", layout(info["networks"][i]["layout"]))
    return network(info["root"])


def _color_normalized_output(out, background):
    roles = {background: 0}
    return np.array([roles.setdefault(int(c), len(roles)) for c in out.ravel()],
                    dtype=np.int8).reshape(out.shape)


def _check_output(info, output):
    """Check the room-substitution accounting directly on output components."""
    bg = info["background"]
    assert info["noise"] not in output
    frame_total = point_total = 0
    for i, n in enumerate(info["networks"]):
        components = _components(output == n["color"])
        frames, points = [], []
        for cells in components:
            if len(cells) == 1:
                points.append(tuple(cells[0]))
            else:
                lo, hi = cells.min(0), cells.max(0)
                mask = np.zeros_like(output, dtype=bool)
                _perimeter(mask, tuple(map(int, (*lo, *hi))))
                assert len(cells) == mask.sum()
                assert np.all(output[mask] == n["color"])
                frames.append((lo, hi))
        assert len(frames) == 1, "one frame per input network"
        direct = sum(p == i for p in info["parents"])
        assert len(points) == len(n["rooms"])-direct, "one point per vacant room"
        lo, hi = frames[0]
        # Every direct child frame and own point has a full one-cell moat.
        for p in points:
            assert np.all(np.asarray(p) >= lo+2) and np.all(np.asarray(p) <= hi-2)
        for j, parent in enumerate(info["parents"]):
            if parent == i:
                child_cells = np.argwhere(output == info["networks"][j]["color"])
                assert np.all(child_cells.min(0) >= lo+2)
                assert np.all(child_cells.max(0) <= hi-2)
        frame_total += len(frames)
        point_total += len(points)
    assert frame_total == len(info["networks"])
    assert point_total == _features(info)["empty"]
    root_color = info["networks"][info["root"]]["color"]
    assert np.all(output[0] == root_color) and np.all(output[-1] == root_color)
    assert np.all(output[:, 0] == root_color) and np.all(output[:, -1] == root_color)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _normal_form(info)), "incorrect normalization"
            _check_output(info, out)
            f = _features(info)
            features[split].append(f)
    train = features["train"]
    assert any(f["partial"] and f["repaired"] and f["depth"] >= 2 for f in train)
    assert any(f["max_children"] >= 2 for f in train)
    assert max(f["depth"] for f in train) == 2
    assert all(f["depth"] == 3 for f in features["test"])


def self_test():
    """Bounded streaming audit of 42 seeds; no files or retained grid corpus."""
    _regressions()
    shapes, answers, layouts, counts, palettes, depths = set(), set(), set(), set(), set(), set()
    point_counts, network_counts, structures = set(), set(), set()
    partial = repaired = total = 0
    rng = np.random.default_rng(71007)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _decode(inp)
                f = _features(info)
                assert np.array_equal(solve(inp), out)
                shapes.add(inp.shape)
                answers.add(_key(_color_normalized_output(out, info["background"])))
                layouts.update(repr(s) for s in f["layouts"])
                counts.add(f["rooms"])
                palettes.add(tuple(map(int, np.unique(inp))))
                depths.add(f["depth"])
                point_counts.add(f["empty"])
                network_counts.add(f["networks"])
                structures.add(_tree_signature(info))
                partial += f["partial"]
                repaired += f["repaired"] > 0
                total += 1
        # Palette permutation must preserve every geometric decision.
        target = puzzle["test"][0]
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[target["input"]]), permutation[target["output"]])
        assert np.array_equal(solve(target["input"].T), target["output"].T)
        # Erasure-free reconstruction has exactly the same normalized answer.
        info = _decode(target["input"])
        restored = target["input"].copy()
        for n in info["networks"]:
            restored[n["mask"]] = n["color"]
        # The visible noise role is still present in background scatter.
        assert np.array_equal(solve(restored), target["output"])
        if seed in (0, 19):
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], again[s]) for k in a)
    assert total == 210 and len(shapes) >= 45
    assert len(answers) >= 90 and len(layouts) >= 12 and len(structures) >= 70
    assert len(counts) >= 35 and len(palettes) >= 20
    assert depths == {1, 2, 3} and len(network_counts) >= 4 and len(point_counts) >= 7
    assert partial >= 60 and repaired >= 150
    # Minimum/maximum legal counts and all illegal type/boundary classes.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(802+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupting a vacancy point, nested frame, or output crop must be detected.
    p = generate_puzzle(0, 2, 1)
    for kind in ("point", "frame", "crop"):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = q["test"][0]
        info = _decode(pair["input"])
        bg = info["background"]
        if kind == "crop":
            pair["output"] = pair["output"][1:]
        else:
            chosen = None
            for c in np.unique(pair["output"]):
                if c == bg:
                    continue
                for cells in _components(pair["output"] == c):
                    if (len(cells) == 1) == (kind == "point"):
                        chosen = tuple(cells[0])
                        break
                if chosen is not None:
                    break
            assert chosen is not None
            pair["output"][chosen] = bg
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted answer accepted")


def _regressions():
    """Hand-specified partial map: counts, spatial substitution and invariance."""
    raw = np.zeros((17, 21), dtype=np.int8)
    wall = np.zeros_like(raw, dtype=bool)
    _perimeter(wall, (0, 0, 16, 20))
    wall[:, 10] = True
    wall[8, :11] = True
    raw[wall] = 1
    child = np.zeros_like(wall)
    _perimeter(child, (5, 13, 9, 17))
    raw[child] = 2
    raw[0, 2] = raw[2, 3] = 9
    expected = np.array([[int(c) for c in row] for row in (
        "11111111111", "10000000001", "10102222201",
        "10002000201", "10102020201", "10002000201",
        "10002222201", "10000000001", "11111111111")], dtype=np.int8)
    assert np.array_equal(solve(raw), expected)
    info = _decode(raw)
    assert [len(n["rooms"]) for n in info["networks"]] == [3, 1]
    # Filling the T's absent continuation would predict four root rooms.
    root = info["networks"][info["root"]]
    t, l, b, r = root["box"]
    hs = np.any(root["mask"][:, :-1] & root["mask"][:, 1:], axis=1).sum()
    vs = np.any(root["mask"][:-1] & root["mask"][1:], axis=0).sum()
    assert (hs-1)*(vs-1) == 4 != len(root["rooms"])
    # Same dimensions, colors, nested network, and outer shape; removing just
    # one actual divider removes one point without changing output dimensions.
    fewer_rooms = raw.copy()
    fewer_rooms[8, 1:10] = 0
    changed = solve(fewer_rooms)
    assert changed.shape == expected.shape and not np.array_equal(changed, expected)
    # The same counts and nesting depth, but a different occupied chamber:
    # normalize the substituted subtree at its actual spatial address.
    moved = raw.copy()
    moved[child] = 0
    new_child = np.zeros_like(wall)
    _perimeter(new_child, (2, 3, 6, 7))
    moved[new_child] = 2
    moved[2, 3] = 2
    moved_info = _decode(moved)
    assert sorted(len(n["rooms"]) for n in moved_info["networks"]) == [1, 3]
    assert solve(moved).shape == (11, 11)
    # Metric size and position inside a chamber carry no information.
    shifted = raw.copy()
    shifted[child] = 0
    new_child[:] = False
    _perimeter(new_child, (3, 13, 11, 17))
    shifted[new_child] = 2
    assert np.array_equal(solve(shifted), expected)
    no_damage = raw.copy()
    no_damage[0, 2] = 1
    assert np.array_equal(solve(no_damage), expected)

