"""Nested shape councils: topology determines which votes are delegated.

No files, parent generators, seed-dependent solving, or stored execution traces.
All stochastic work uses the generate_puzzle call's NumPy Generator.
"""
from collections import Counter
from functools import lru_cache

import numpy as np


def _normal(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


def _rotations(cells):
    result, current = set(), _normal(cells)
    for _ in range(4):
        result.add(current)
        current = _normal((c, -r) for r, c in current)
    return tuple(sorted(result))


def _mirror(cells):
    return _normal((r, -c) for r, c in cells)


def _variants(cells):
    return tuple(sorted(set(_rotations(cells)) | set(_rotations(_mirror(cells)))))


def _signature(cells, mode="free"):
    raw = _normal(cells)
    if mode == "area":
        return len(raw)
    if mode == "box":
        return tuple(sorted(_extent(raw)))
    if mode == "translation":
        return raw
    if mode == "rotation":
        return min(_rotations(raw))
    return min(_variants(raw))


def _extent(cells):
    return max(r for r, _ in cells)+1, max(c for _, c in cells)+1


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, cells = [(int(r), int(c))], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(tuple(cells))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _check_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=False)


def _parse(grid):
    grid = _check_grid(grid)
    bg = int(grid[0, 0])
    if not all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("the input needs a uniform background border")
    objects, labels = [], np.zeros(grid.shape, int)
    for index, cells in enumerate(_components(grid != bg)):
        rr, cc = np.array(cells).T
        colors = np.unique(grid[rr, cc])
        if len(colors) != 1:
            raise ValueError("each component must be monochromatic")
        lo, hi = (int(rr.min()), int(cc.min())), (int(rr.max()), int(cc.max()))
        local = np.zeros((hi[0]-lo[0]+1, hi[1]-lo[1]+1), bool)
        local[rr-lo[0], cc-lo[1]] = True
        hole = _holes(local)
        kind = "region" if hole.any() else "token"
        if kind == "region" and len(_components(hole)) != 1:
            raise ValueError("a contour must have one enclosed region")
        hole_cells = frozenset((int(r)+lo[0], int(c)+lo[1]) for r, c in zip(*np.nonzero(hole)))
        objects.append(dict(cells=cells, lo=lo, hi=hi, color=int(colors[0]),
                            kind=kind, hole=hole_cells, raw=_normal(cells), mask=local))
        labels[rr, cc] = index+1
    if not objects:
        raise ValueError("empty drawing")
    for a, b in ((labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        if np.any((a != 0) & (b != 0) & (a != b)):
            raise ValueError("distinct objects need background separation")
    regions = [i for i, o in enumerate(objects) if o["kind"] == "region"]
    lines = {objects[i]["color"] for i in regions}
    if len(lines) != 1:
        raise ValueError("all contours share one drawing color")
    line = lines.pop()
    if any(o["kind"] == "token" and o["color"] == line for o in objects):
        raise ValueError("tokens cannot use the contour color")
    parents, children = [], {i: [] for i in regions}
    for i, obj in enumerate(objects):
        containing = []
        for j in regions:
            if i == j:
                continue
            inside = [cell in objects[j]["hole"] for cell in obj["cells"]]
            if any(inside) and not all(inside):
                raise ValueError("crossing objects")
            if all(inside):
                containing.append(j)
        parents.append(min(containing, key=lambda j: len(objects[j]["hole"])) if containing else None)
    roots = [i for i, p in enumerate(parents) if p is None]
    if len(roots) != 1 or roots[0] not in regions:
        raise ValueError("one outer contour must contain the whole drawing")
    for i, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(i)
    if any(not ids for ids in children.values()):
        raise ValueError("empty district")
    return dict(grid=grid, background=bg, line=line, objects=objects,
                parents=parents, children=children, root=roots[0], regions=regions)


def _plurality(counts):
    best = max(counts.values())
    winners = [c for c, n in counts.items() if n == best]
    if len(winners) != 1:
        raise ValueError("tied effective shape ballot")
    return winners[0]


def _elections(info, mode="free", policy="delegates"):
    """Return winners and effective ballots; alternatives are for validation."""
    winners, ballots, raw_ballots = {}, {}, {}
    objects = info["objects"]

    def visit(region):
        local, raw = {}, {}
        for i in info["children"][region]:
            obj = objects[i]
            if obj["kind"] == "token":
                key = _signature(obj["raw"], mode)
                for target in (local, raw):
                    target.setdefault(key, Counter())[obj["color"]] += 1
            else:
                visit(i)
                for key, counts in raw_ballots[i].items():
                    raw.setdefault(key, Counter()).update(counts)
                if policy == "delegates":
                    for key, color in winners[i].items():
                        local.setdefault(key, Counter())[color] += 1
        if policy == "pooled":
            local = raw
        # Local-only is deliberately allowed to ignore classes without targets.
        ballots[region], raw_ballots[region] = local, raw
        winners[region] = {key: _plurality(counts) for key, counts in local.items()}

    visit(info["root"])
    return winners, ballots


def _join(parts, axis, bg, gap=1):
    """Centered cardinal composition. Canonical tiles always have odd sizes."""
    cross = 1-axis
    before = max((p.shape[cross]-1)//2 for p in parts)
    after = max(p.shape[cross]-1-(p.shape[cross]-1)//2 for p in parts)
    shape = [0, 0]
    shape[axis] = sum(p.shape[axis] for p in parts)+gap*(len(parts)-1)
    shape[cross] = before+after+1
    result = np.full(shape, bg, np.int8)
    position = 0
    for part in parts:
        start = [0, 0]
        start[axis] = position
        start[cross] = before-(part.shape[cross]-1)//2
        y, x = start
        # Parts' bounding rectangles do not overlap on the joining axis.
        result[y:y+part.shape[0], x:x+part.shape[1]] = part
        position += part.shape[axis]+gap
    return result


def _frame(content, bg, line):
    result = np.pad(content, 2, constant_values=bg)
    result[0, :] = result[-1, :] = line
    result[:, 0] = result[:, -1] = line
    return result


def _redraw(info, winners, mode="free", original_colors=False):
    objects, bg = info["objects"], info["background"]
    cache = {}

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            obj = objects[i]
            if obj["kind"] == "token":
                color = obj["color"] if original_colors else winners[info["parents"][i]][_signature(obj["raw"], mode)]
                result = np.array([[color]], np.int8)
            else:
                result = _frame(group(info["children"][i]), bg, info["line"])
        else:
            candidates = []
            for axis in (0, 1):
                partitions, end = [], -100
                for i in sorted(ids, key=lambda i: objects[i]["lo"][axis]):
                    if objects[i]["lo"][axis] > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, objects[i]["hi"][axis])
                if len(partitions) < 2:
                    continue
                cross = 1-axis
                centers = [min(objects[i]["lo"][cross] for i in part)
                           + max(objects[i]["hi"][cross] for i in part) for part in partitions]
                # At most a half-cell center discrepancy due to even input sizes.
                if max(centers)-min(centers) > 1:
                    continue
                try:
                    candidates.append(_join([group(part) for part in partitions], axis, bg))
                except ValueError:
                    continue
            if not candidates or any(not np.array_equal(candidates[0], c) for c in candidates[1:]):
                raise ValueError("ambiguous or uncentered cardinal layout")
            result = candidates[0]
        cache[ids] = result
        return result

    return group([info["root"]])


def _decode(input_grid, mode="free", policy="delegates"):
    info = _parse(input_grid)
    winners, ballots = _elections(info, mode, policy)
    output = _redraw(info, winners, mode)
    return output, info, winners, ballots


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve exclusively from visible contours, shapes, colors, and positions."""
    return _decode(input_grid)[0]


@lru_cache(maxsize=1)
def _pool():
    # Enumerate free polyominoes through area six; no examples or answers cached.
    current = {((0, 0),)}
    result = set(current)
    for _ in range(5):
        grown = set()
        for shape in current:
            occupied = set(shape)
            frontier = {(r+dr, c+dc) for r, c in shape
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}-occupied
            for point in sorted(frontier):
                candidate = _signature(occupied | {point})
                h, w = _extent(candidate)
                if max(h, w) <= 4:
                    mask = np.zeros((h, w), bool)
                    for r, c in candidate:
                        mask[r, c] = True
                    if not _holes(mask).any():
                        grown.add(candidate)
        current = grown
        result.update(grown)
    return tuple(sorted(result))


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _token(rng, shape, color, allowed=None):
    return dict(kind="token", shape=shape, color=int(color),
                raw=_choice(rng, _variants(shape) if allowed is None else allowed))


def _group(axis, items):
    return items[0] if len(items) == 1 else ("group", int(axis), tuple(items))


def _layout(rng, items):
    items = [items[int(i)] for i in rng.permutation(len(items))]
    if len(items) == 1:
        return items[0]
    axis = int(rng.integers(2))
    if len(items) <= 3 and rng.random() < .55:
        return _group(axis, items)
    bands = int(rng.integers(2, min(3, len(items))+1))
    lengths = [len(items)//bands+(i < len(items) % bands) for i in range(bands)]
    parts, start = [], 0
    for n in lengths:
        parts.append(_group(1-axis, items[start:start+n]))
        start += n
    return _group(axis, parts)


def _region(rng, items, layout=None, padding=None):
    return dict(kind="region", items=items, layout=_layout(rng, items) if layout is None else layout,
                padding=padding)


def _scene_elections(node, winners):
    counts = {}
    for child in node["items"]:
        if child["kind"] == "token":
            counts.setdefault(child["shape"], Counter())[child["color"]] += 1
        else:
            for key, color in _scene_elections(child, winners).items():
                counts.setdefault(key, Counter())[color] += 1
    elected = {key: _plurality(ballot) for key, ballot in counts.items()}
    winners[id(node)] = elected
    return elected


def _outline(h, w, padding, rng):
    """Closed stepped corner bays, with a protected rectangular content core."""
    ptop, pbottom, pleft, pright = padding
    result = np.zeros((h, w), bool)
    result[0, :] = result[-1, :] = True
    result[:, 0] = result[:, -1] = True
    for top, left, cap in ((True, True, min(ptop, pleft)-2),
                           (True, False, min(ptop, pright)-2),
                           (False, True, min(pbottom, pleft)-2),
                           (False, False, min(pbottom, pright)-2)):
        k = int(rng.integers(cap+1))
        if k == 0:
            continue
        # Replace the outer corner by an orthogonal inset corner.
        for y in range(k+1):
            for x in range(k+1):
                yy, xx = (y if top else h-1-y), (x if left else w-1-x)
                result[yy, xx] = (y == k or x == k)
    return result


def _draw_scene(node, rng, bg, line, compact=False):
    winners = {}
    _scene_elections(node, winners)

    def layout(tree, owner):
        if isinstance(tree, tuple):
            pairs = [layout(child, owner) for child in tree[2]]
            gap = 1 if compact else int(rng.integers(1, 4))
            return (_join([p[0] for p in pairs], tree[1], bg, gap),
                    _join([p[1] for p in pairs], tree[1], bg))
        if tree["kind"] == "token":
            raw = tree["raw"]
            inp = np.full(_extent(raw), bg, np.int8)
            for r, c in raw:
                inp[r, c] = tree["color"]
            return inp, np.array([[winners[id(owner)][tree["shape"]]]], np.int8)
        return region(tree)

    def region(obj):
        inp, out = layout(obj["layout"], obj)
        if obj["padding"] is not None:
            pads = [obj["padding"]]*4
        elif compact:
            pads = [2]*4
        else:
            # Symmetric padding also makes whole-object centers easy to read.
            a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            pads = [a, a, b, b]
        t, b, l, r = pads
        drawing = np.full((inp.shape[0]+t+b, inp.shape[1]+l+r), bg, np.int8)
        drawing[t:t+inp.shape[0], l:l+inp.shape[1]] = inp
        drawing[_outline(*drawing.shape, pads, rng)] = line
        return drawing, _frame(out, bg, line)

    return region(node)


def _finish(scene, rng, palette, compact=False):
    bg, line = palette[:2]
    try:
        inp, out = _draw_scene(scene, rng, bg, line, compact)
        if max(*inp.shape, *out.shape) > 28:
            return None
        extra_y = int(rng.integers(0, min(4, 29-inp.shape[0])))
        extra_x = int(rng.integers(0, min(4, 29-inp.shape[1])))
        top, left = int(rng.integers(extra_y+1)), int(rng.integers(extra_x+1))
        inp = np.pad(inp, ((1+top, 1+extra_y-top), (1+left, 1+extra_x-left)), constant_values=bg)
        # Ground truth was drawn from the abstract elections; reconstruction
        # uses only the visible input and rejects ambiguous metric realizations.
        prediction = solve(inp)
        if not np.array_equal(prediction, out):
            return None
        return dict(input=inp, output=out)
    except ValueError:
        return None


def _witness_scope(rng, anchor, companion, palette):
    red, blue, green = palette[2:5]
    narrow = lambda shape: tuple(v for v in _variants(shape) if _extent(v)[1] == 2)
    leaf = lambda s, c: _token(rng, s, c, narrow(s))
    a = [leaf(anchor, red) for _ in range(3)]+[leaf(companion, green)]
    b = [leaf(anchor, red) for _ in range(2)]+[leaf(companion, green)]
    # Two-by-two or ragged two-band children; the order is not a semantic code.
    rng.shuffle(a)
    rng.shuffle(b)
    ca = _region(rng, a, _group(0, [_group(1, a[:2]), _group(1, a[2:])]), padding=3)
    cb = _region(rng, b, _group(0, [_group(1, b[:2]), b[2]]), padding=3)
    direct = [leaf(anchor, blue) for _ in range(3)]+[leaf(companion, blue)]
    rng.shuffle(direct)
    kids = [ca, cb]
    rng.shuffle(kids)
    bands = [_group(1, kids), _group(1, direct)]
    rng.shuffle(bands)
    scene = _region(rng, kids+direct, _group(0, bands), padding=2)
    # Compact dimensions are guaranteed even for 4-by-2 witnesses.
    for _ in range(12):
        pair = _finish(scene, rng, palette, compact=True)
        if pair is not None:
            # Transpose is safe for the centered cardinal convention.
            if rng.random() < .5:
                pair = {k: v.T.copy() for k, v in pair.items()}
            return pair
    raise RuntimeError("scope teaching construction failed")


def _witness_plurality(rng, anchor, palette):
    red, blue, green = palette[2:5]
    reflected, native = _rotations(_mirror(anchor)), _rotations(anchor)
    items = [_token(rng, anchor, red, reflected) for _ in range(3)]
    items += [_token(rng, anchor, blue, native) for _ in range(2)]
    items += [_token(rng, anchor, green, native), _token(rng, anchor, green, reflected)]
    for _ in range(40):
        scene = _region(rng, items)
        pair = _finish(scene, rng, palette, compact=True)
        if pair is not None:
            return pair
    raise RuntimeError("plurality teaching construction failed")


def _random_scene(rng, shapes, colors, depth, required_shape=None):
    def build(level):
        children = [build(level+1)] if level < depth else []
        if depth == 2 and level == 1 and rng.random() < .28:
            children.append(build(2))
        n = int(rng.integers(1, 5 if level == depth else 4))
        preferred = int(_choice(rng, colors))
        tokens = []
        for i in range(n):
            shape = required_shape if level == depth and i == 0 and required_shape is not None else _choice(rng, shapes)
            color = preferred if rng.random() < .78 else int(_choice(rng, colors))
            tokens.append(_token(rng, shape, color))
        return _region(rng, children+tokens)
    return build(1)


def _general_pair(rng, palette, shapes, depth, novel=None):
    for attempt in range(96):
        scene = _random_scene(rng, shapes, palette[2:6], depth, novel)
        pair = _finish(scene, rng, palette, compact=attempt >= 20)
        if pair is not None:
            return pair
    # A bounded, easily fitting chain, still with visible ballots and no
    # hidden convention. The test keeps its required novel class and depth.
    shape = novel if novel is not None else _choice(rng, shapes)
    color = palette[2]
    node = _region(rng, [_token(rng, shape, color)], padding=2)
    for _ in range(depth-1):
        leaf = _token(rng, shape, color)
        node = _region(rng, [leaf, node], _group(0, [leaf, node]), padding=2)
    pair = _finish(node, rng, palette, compact=True)
    if pair is None:
        raise RuntimeError("bounded chain construction failed")
    return pair


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(name+" is outside the allowed integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _pool()
    eligible = [s for s in pool if 4 <= len(s) <= 6 and sorted(_extent(s)) in ([2, 3], [2, 4])
                and _mirror(s) not in _rotations(s)
                and any(t != s and len(t) == len(s) and sorted(_extent(t)) == sorted(_extent(s)) for t in pool)]
    anchor = _choice(rng, eligible)
    companion = _choice(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                            and sorted(_extent(s)) == sorted(_extent(anchor))])
    base_palette = tuple(map(int, rng.permutation(10)))
    stable_palette = rng.random() < .4
    puzzle, seen, used = dict(train=[], test=[]), set(), set()
    first_winner = base_palette[3]
    for split, n in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(n):
            for retry in range(16):
                palette = base_palette if stable_palette or (split == "train" and index == 0) else tuple(map(int, rng.permutation(10)))
                if split == "train" and index == 0:
                    pair = _witness_scope(rng, anchor, companion, palette)
                elif split == "train" and index == 1:
                    # The recurring anchor changes winner, refuting a dictionary.
                    palette = list(palette)
                    if palette[2] == first_winner:
                        palette[2], palette[3] = palette[3], palette[2]
                    pair = _witness_plurality(rng, anchor, palette)
                else:
                    depth = 3 if split == "test" else int(rng.integers(1, 3))
                    novel = _choice(rng, [s for s in pool if s not in used]) if split == "test" else None
                    repertoire = [pool[int(i)] for i in rng.choice(len(pool), int(rng.integers(1, 4)), replace=False)]
                    if novel is not None:
                        repertoire[0] = novel
                    pair = _general_pair(rng, palette, repertoire, depth, novel)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                used.update(_signature(o["raw"]) for o in _parse(pair["input"])["objects"] if o["kind"] == "token")
        rng.shuffle(puzzle[split])
    return puzzle


def _alternative_differs(pair, mode="free", policy="delegates"):
    try:
        return not np.array_equal(_decode(pair["input"], mode, policy)[0], pair["output"])
    except (ValueError, KeyError):
        return True


def _depth(info, index):
    depth = 0
    while index is not None:
        depth += info["objects"][index]["kind"] == "region"
        index = info["parents"][index]
    return depth


def _topology(info, index):
    obj = info["objects"][index]
    return "P" if obj["kind"] == "token" else tuple(sorted((_topology(info, i) for i in info["children"][index]), key=repr))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records, vocab = set(), {"train": [], "test": []}, {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            prediction, info, winners, ballots = _decode(inp)
            assert np.array_equal(prediction, out), "incorrect election, compression, or crop"
            assert np.all(out[0] == info["line"]) and np.all(out[-1] == info["line"])
            assert np.all(out[:, 0] == info["line"]) and np.all(out[:, -1] == info["line"])
            # Parse the output with a temporary border, checking true enclosure,
            # contour count, dot count, containment, frame geometry, and moat.
            assert max(out.shape) <= 28
            normalized = _parse(np.pad(out, 1, constant_values=info["background"]))
            assert _topology(info, info["root"]) == _topology(normalized, normalized["root"])
            for i, obj in enumerate(normalized["objects"]):
                if obj["kind"] == "token":
                    assert len(obj["cells"]) == 1
                else:
                    h, w = obj["mask"].shape
                    assert len(obj["cells"]) == 2*(h+w)-4
                    ids = normalized["children"][i]
                    low = np.min([normalized["objects"][j]["lo"] for j in ids], axis=0)
                    high = np.max([normalized["objects"][j]["hi"] for j in ids], axis=0)
                    assert np.array_equal(low, np.array(obj["lo"])+2)
                    assert np.array_equal(high, np.array(obj["hi"])-2)
            # Every child sends exactly one vote per class, even if no direct
            # parent token has that shape. Rebuild the counters from pixels.
            for region, ids in info["children"].items():
                expected = {}
                for i in ids:
                    obj = info["objects"][i]
                    support = winners[i].items() if obj["kind"] == "region" else [(_signature(obj["raw"]), obj["color"])]
                    for shape, color in support:
                        expected.setdefault(shape, Counter())[color] += 1
                assert expected == ballots[region]
            shapes = {_signature(o["raw"]) for o in info["objects"] if o["kind"] == "token"}
            vocab[split].update(shapes)
            records[split].append((info, winners, ballots, shapes))
    train = puzzle["train"]
    for mode, policy in (("area", "delegates"), ("box", "delegates"),
                         ("rotation", "delegates"), ("translation", "delegates"),
                         ("free", "pooled"), ("free", "local")):
        assert any(_alternative_differs(p, mode, policy) for p in train), (mode, policy)
    assert any(not np.array_equal(_redraw(i, w, original_colors=True), p["output"])
               for p, (i, w, _, _) in zip(train, records["train"]))
    assert any(any(len(counts) >= 3 and 2*max(counts.values()) <= sum(counts.values())
                   for counts in ballot.values()) for _, _, ballots, _ in records["train"] for ballot in ballots.values())
    history = {}
    for info, winners, _, _ in records["train"]:
        for shape, color in winners[info["root"]].items():
            history.setdefault(shape, set()).add(color)
    assert any(len(colors) > 1 for colors in history.values())
    assert all(max(_depth(info, i) for i in info["regions"]) <= 2 for info, _, _, _ in records["train"])
    for info, _, _, shapes in records["test"]:
        assert max(_depth(info, i) for i in info["regions"]) == 3
        assert shapes-vocab["train"], "test must exercise a visibly supported new shape"


def self_test():
    """42 seeds, streaming measurements, metamorphic and corruption checks.

    No grids, logs, traces, or reports are stored by this function.
    """
    signatures, dimensions, topologies, vocab, normalized_outputs = set(), set(), set(), set(), set()
    ballots, raw_sizes, line_colors, nonrect = set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, info, winners, votes = _decode(pair["input"])
                dimensions.add(pair["input"].shape)
                topologies.add(_topology(info, info["root"]))
                vocab.update(_signature(o["raw"]) for o in info["objects"] if o["kind"] == "token")
                raw_sizes.update(len(o["cells"]) for o in info["objects"] if o["kind"] == "token")
                line_colors.add(info["line"])
                normalized_outputs.add(_key(_redraw(info, winners, original_colors=True) != info["background"]))
                ballots.update(tuple(sorted(v.values(), reverse=True)) for b in votes.values() for v in b.values())
                nonrect += sum(not (o["mask"][0].all() and o["mask"][-1].all()
                                    and o["mask"][:, 0].all() and o["mask"][:, -1].all())
                               for o in info["objects"] if o["kind"] == "region")
        # Color permutations, rotations/reflections, and translation must not
        # alter elections or normalized relations.
        pair = puzzle["test"][0]
        rng = np.random.default_rng(90000+seed)
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
        if max(pair["input"].shape) <= 28:
            bg = int(pair["input"][0, 0])
            assert np.array_equal(solve(np.pad(pair["input"], 1, constant_values=bg)), pair["output"])
    assert len(signatures) == 42
    assert len(dimensions) >= 30 and len(topologies) >= 25
    assert len(vocab) >= 25 and len(normalized_outputs) >= 50
    assert len(raw_sizes) >= 5 and len(line_colors) == 10
    assert nonrect >= 42 and {(3, 2, 2), (3, 2), (2, 1), (1,)} <= ballots
    # A shape must pass through a district with no direct token of that shape.
    # The resulting delegate resolves a root tie; omitting it is undefined.
    rng = np.random.default_rng(711)
    xshape, yshape = _signature(((0, 0), (0, 1))), ((0, 0),)
    inner_tokens = [_token(rng, xshape, 2, (xshape,)) for _ in range(3)]
    inner = _region(rng, inner_tokens, _group(1, inner_tokens), padding=2)
    middle_token = _token(rng, yshape, 4)
    middle = _region(rng, [middle_token, inner], _group(0, [middle_token, inner]), padding=2)
    root_tokens = [_token(rng, xshape, c, (xshape,)) for c in (2, 3)]
    scene = _region(rng, root_tokens+[middle],
                    _group(0, [_group(1, root_tokens), middle]), padding=2)
    metric_drawings, canonical_answers = set(), set()
    for _ in range(12):
        raw, answer = _draw_scene(scene, rng, 0, 1)
        if max(raw.shape) > 28:
            continue
        inp = np.pad(raw, 1)
        actual, info, winners, votes = _decode(inp)
        assert np.array_equal(actual, answer)
        metric_drawings.add(_key(inp))
        canonical_answers.add(_key(answer))
        mid = next(i for i in info["regions"] if _depth(info, i) == 2)
        assert xshape in winners[mid]
        assert all(_signature(info["objects"][i]["raw"]) != xshape
                   for i in info["children"][mid] if info["objects"][i]["kind"] == "token")
        assert votes[info["root"]][xshape] == Counter({2: 2, 3: 1})
    assert len(metric_drawings) >= 3 and len(canonical_answers) == 1
    a, b = generate_puzzle(123), generate_puzzle(123)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, ne in ((2, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(42+nt, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0)
    pair = broken["test"][0]
    info = _parse(pair["input"])
    dots = np.argwhere((pair["output"] != info["background"]) & (pair["output"] != info["line"]))
    y, x = dots[0]
    pair["output"][y, x] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("deleted elected point accepted")
