"""Reveal nested rectangular cages and discard records matching a color query.

Only visible pixels are used by solve.  Generation paints the answer from a
complete scene; the reader independently recovers geometry and dot ownership.
"""

from collections import Counter
import json
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _inside(p, box):
    y, x = p
    t, b, l, r = box
    return t < y < b and l < x < r


def _size(box):
    t, b, l, r = box
    return (b - t + 1) * (r - l + 1)


def _contains(a, b):
    return a[0] < b[0] and b[1] < a[1] and a[2] < b[2] and b[3] < a[3]


def _separate(a, b):
    # At least one complete background row or column separates root boxes.
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _halo(points, radius=1):
    return {(y + dy, x + dx) for y, x in points
            for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)}


def _components(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, queue = {p}, [p]
        for y, x in queue:
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _corner_view(grid, corner):
    if corner & 2:
        grid = grid[::-1, :]
    if corner & 1:
        grid = grid[:, ::-1]
    return grid


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _read(input_grid):
    """Recover the selector, shutter, full cages, patterns and scoped dots."""
    grid = _grid(input_grid)
    assert min(grid.shape) >= 5
    selectors = []
    for corner in range(4):
        view = _corner_view(grid, corner)
        frame = int(view[3, 3])
        if not (np.all(view[3, :4] == frame) and np.all(view[:4, 3] == frame)):
            continue
        colors, counts = np.unique(view[:3, :3], return_counts=True)
        padding = colors[counts == 6]
        if len(padding) != 1:
            continue
        bg = int(padding[0])
        query = frozenset(int(c) for c in colors if c != bg)
        if len(query) == 3 and frame not in query and frame != bg:
            selectors.append((corner, bg, frame, query))
    assert len(selectors) == 1, "unique L selector required"
    corner, bg, frame, query = selectors[0]
    selector_mask = np.zeros(grid.shape, dtype=bool)
    _corner_view(selector_mask, corner)[:4, :4] = True
    selector = set(zip(*np.nonzero(selector_mask)))
    shutter_points = set(zip(*np.nonzero((grid == frame) & ~selector_mask)))
    assert shutter_points
    shutter = _bbox(shutter_points)
    assert shutter_points == _perimeter(shutter), "one complete shutter border required"
    t, b, l, r = shutter
    assert b-t >= 3 and r-l >= 3
    mask = _area(shutter)
    assert not mask & _halo(selector)
    fill = int(grid[t+1, l+1])
    assert np.all(grid[t+1:b, l+1:r] == fill) and fill != frame and fill not in query

    points = set(zip(*np.nonzero(grid != bg))) - mask - selector
    groups = _components(points)
    dots, cages = [], []
    for group in groups:
        if len(group) == 1:
            dots.append(next(iter(group)))
            continue
        box = _bbox(group)
        assert box[1]-box[0] >= 4 and box[3]-box[2] >= 4
        perimeter = _perimeter(box)
        assert group == perimeter-mask, "fragment must expose all four cage extrema"
        values = {int(grid[p]) for p in group}
        assert not values & ({bg, frame, fill} | set(query))
        assert 1 <= len(values) <= 2
        if len(values) == 1:
            word = (next(iter(values)),) * 2
        else:
            parity = {}
            for y, x in sorted(group):
                k, value = (y+x) % 2, int(grid[y, x])
                assert k not in parity or parity[k] == value
                parity[k] = value
            assert set(parity) == {0, 1}
            word = (parity[0], parity[1])
        cages.append({"box": box, "visible": group, "full": perimeter,
                      "word": word, "dots": [], "counts": Counter()})
    assert cages
    cages.sort(key=lambda c: c["box"])
    boundaries = set().union(*(c["full"] for c in cages))
    boundary_colors = set().union(*(set(c["word"]) for c in cages))
    assert not boundary_colors & ({bg, frame, fill} | set(query))
    for i, a in enumerate(cages):
        for other in cages[i+1:]:
            ab, bb = a["box"], other["box"]
            assert _contains(ab, bb) or _contains(bb, ab) or _separate(ab, bb)
            assert not a["full"] & _halo(other["full"])
    for p in dots:
        value = int(grid[p])
        assert value not in boundary_colors | {bg, frame, fill}
        assert p not in _halo(boundaries | mask | selector)
        containing = [c for c in cages if _inside(p, c["box"])]
        assert containing, "every record dot must have a cage"
        owner = min(containing, key=lambda c: _size(c["box"]))
        owner["dots"].append(p)
        owner["counts"][value] += 1
    assert all(p not in _halo(set(dots)-{p}) for p in dots), "dots must be isolated"
    for cage in cages:
        cage["labels"] = frozenset(cage["counts"])
        cage["delete"] = query <= cage["labels"]
        cage["children"] = [j for j, c in enumerate(cages) if _contains(cage["box"], c["box"])]
    return {"background": bg, "frame": frame, "fill": fill, "query": query,
            "corner": corner, "selector": selector, "shutter": shutter, "cages": cages}


def _render_solution(grid, info, mode="rule"):
    grid = np.asarray(grid)
    out = grid.copy()
    t, b, l, r = info["shutter"]
    out[t:b+1, l:r+1] = info["background"]
    for cage in info["cages"]:
        delete = cage["delete"]
        if mode == "no_filter":
            delete = False
        elif mode == "borrow_children":
            labels = {int(grid[p]) for c in info["cages"] for p in c["dots"]
                      if _inside(p, cage["box"])}
            delete = info["query"] <= labels
        if delete:
            for p in cage["visible"] | set(cage["dots"]):
                out[p] = info["background"]
        else:
            cells = cage["visible"] if mode == "no_completion" else cage["full"]
            for y, x in cells:
                out[y, x] = cage["word"][(y+x) % 2]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _render_solution(grid, _read(grid))


def _ri(rng, lo, hi):
    if lo > hi:
        raise ValueError("empty construction interval")
    return int(rng.integers(lo, hi+1))


def _child(rng, box, corner):
    t, b, l, r = box
    top, bottom = ((4, 5), (2, 3)) if corner in ("NW", "NE") else ((2, 3), (4, 5))
    left, right = ((4, 5), (2, 3)) if corner == "NW" else ((2, 3), (4, 5))
    child = (t+_ri(rng, *top), b-_ri(rng, *bottom),
             l+_ri(rng, *left), r-_ri(rng, *right))
    if child[1]-child[0] < 6 or child[3]-child[2] < 6:
        raise ValueError("child too small")
    return child


def _slots(boxes, shutter, selector, h, w):
    """Four parity sublattices yield mutually separated visible dot positions."""
    boundaries = set().union(*(_perimeter(b) for b in boxes))
    blocked = _halo(boundaries | _area(shutter) | selector)
    result = [[[] for _ in range(4)] for _ in boxes]
    for y in range(h):
        for x in range(w):
            p = (y, x)
            if p in blocked:
                continue
            containing = [i for i, box in enumerate(boxes) if _inside(p, box)]
            if containing:
                owner = min(containing, key=lambda i: _size(boxes[i]))
                result[owner][2*(y % 2)+x % 2].append(p)
    return result


def _geometry(rng, two_nests, requirements, extra):
    # A random separating column partitions two independently sized root cages.
    # Roots may approach from adjacent or opposite sides of the shutter.
    h, w = _ri(rng, 25, 30), _ri(rng, 27, 30)
    selector = _area((h-4, h-1, 0, 3))
    cut = _ri(rng, 13, w-14)
    a_width = _ri(rng, 13, cut)
    a_left = _ri(rng, 0, cut-a_width)
    a_top = _ri(rng, 0, 2)
    a_height = _ri(rng, 13, min(19, h-7-a_top))
    a = (a_top, a_top+a_height-1, a_left, a_left+a_width-1)
    c = _child(rng, a, "NW")
    b_width = _ri(rng, 13, w-cut-1)
    b_left = _ri(rng, cut+1, w-b_width)
    same_side = bool(rng.integers(2))
    b_height = _ri(rng, 13, min(19, h-5))
    if same_side:
        b_top = _ri(rng, 0, 2)
        b = (b_top, b_top+b_height-1, b_left, b_left+b_width-1)
        d = _child(rng, b, "NE")
        top = _ri(rng, max(c[0], d[0])+3, min(c[1], d[1])-1)
        bottom = _ri(rng, max(a[1], b[1])+1, h-3)
    else:
        b_bottom = h-1-_ri(rng, 0, 2)
        b = (b_bottom-b_height+1, b_bottom, b_left, b_left+b_width-1)
        d = _child(rng, b, "SE")
        top = _ri(rng, c[0]+3, min(c[1]-1, b[0]-1))
        bottom = _ri(rng, max(a[1]+1, d[0]+1), d[1]-3)
    left = _ri(rng, c[2]+3, c[3]-1)
    right = _ri(rng, d[2]+1, d[3]-3)
    shutter = (top, bottom, left, right)
    if bottom-top < 4 or right-left < 4 or _area(shutter) & _halo(selector):
        raise ValueError("shutter packing")
    boxes = [a, c, b] + ([d] if two_nests else [])
    if extra:
        # Exterior records use free positions, not a reserved object slot.
        for _ in range(160):
            hh, ww = _ri(rng, 5, 10), _ri(rng, 5, 12)
            y, x = _ri(rng, 0, h-hh), _ri(rng, 0, w-ww)
            e = (y, y+hh-1, x, x+ww-1)
            if (all(_separate(e, box) for box in boxes) and
                    not _area(e) & _halo(_area(shutter) | selector)):
                boxes.append(e)
                break
        else:
            raise ValueError("no free exterior cage")
    mask = _area(shutter)
    for i, box in enumerate(boxes):
        visible = _perimeter(box)-mask
        if not visible or _bbox(visible) != box or len(_components(visible)) != 1:
            raise ValueError("ambiguous fragment")
        if i < 3 + int(two_nests) and len(_perimeter(box) & mask) < 2:
            raise ValueError("missing occlusion")
    slots = _slots(boxes, shutter, selector, h, w)
    if any(max(map(len, choices)) < n for choices, n in zip(slots, requirements)):
        raise ValueError("insufficient visible owner space")
    return h, w, shutter, boxes, slots


def _marks(rng, query, extras, package, two_nests, extra):
    q, x = list(query), list(extras)
    if package == 0:
        # Outer missing q0 must not borrow q0 from its exact-match child.
        marks = [[q[1], q[2], x[0], x[1], q[1], x[0]], q,
                 [q[0], q[2], x[0], q[0]]]
    elif package == 1:
        # Deleting a parent must preserve its nonmatching child.
        marks = [q+[x[0], q[0], x[0]], [q[0], q[1], x[0]], q+[x[1]]]
    else:
        marks = []
        for _ in range(3+int(two_nests)):
            k = _ri(rng, 0, 3)
            selected = [int(v) for v in rng.permutation(q)[:k]]
            selected += [int(v) for v in rng.permutation(x)[:_ri(rng, 0, 2)]]
            if selected and rng.random() < .45 and len(selected) < 5:
                selected.append(int(rng.choice(selected)))
            marks.append(selected)
        # Tests always exercise both inheritance mechanisms, with a changed
        # nesting forest rather than a hidden new decision rule.
        if two_nests:
            marks[0] = [q[1], q[2], x[0], x[1]]
            marks[1] = q
            if rng.random() < .5:
                # Swap outcomes within the first parent/child pair when capacities
                # permit, so nesting depth does not encode membership.
                marks[0], marks[1] = marks[1], marks[0]
        else:
            if not any(set(q) <= set(m) for m in marks):
                marks[int(rng.integers(3))] = q
            if all(set(q) <= set(m) for m in marks):
                marks[int(rng.integers(3))] = [q[0], q[1]]
    if extra:
        marks.append([] if rng.random() < .5 else [int(rng.choice(q+x))])
    return marks


def _fallback_geometry(two_nests, requirements, extra):
    """Guaranteed readable packing for even the largest requested dot sets."""
    h = w = 30
    shutter = (10, 18, 8, 20)
    boxes = [(0, 16, 0, 12), (4, 14, 4, 10), (13, 29, 16, 28)]
    if two_nests:
        boxes.append((15, 25, 18, 24))
    if extra:
        boxes.append((0, 6, 16, 28))
    slots = _slots(boxes, shutter, _area((26, 29, 0, 3)), h, w)
    assert len(slots) == len(requirements)
    assert all(max(map(len, choices)) >= n for choices, n in zip(slots, requirements))
    return h, w, shutter, boxes, slots


def _make_pair(rng, package, test, background, query_override=None, force_fallback=False):
    reserved = set(query_override or [])
    colors = [int(c) for c in rng.permutation([c for c in range(10)
                                             if c != background and c not in reserved])]
    frame, fill = colors[:2]
    edge_colors = colors[2:4]
    label_colors = list(query_override or []) + colors[4:]
    if package == 0 or (package > 1 and rng.random() < .35):
        fill = background
    q = (list(query_override) if query_override is not None else
         [int(c) for c in rng.choice(label_colors, 3, replace=False)])
    extras = [c for c in label_colors if c not in q]
    two_nests = test
    extra = package > 1 and rng.random() < .42
    marks = _marks(rng, q, extras, package, two_nests, extra)
    requirements = [len(m) for m in marks]
    geometry = None
    for _ in range(0 if force_fallback else 500):
        try:
            geometry = _geometry(rng, two_nests, requirements, extra)
            break
        except ValueError:
            continue
    if geometry is None:
        geometry = _fallback_geometry(two_nests, requirements, extra)
    h, w, shutter, boxes, slots = geometry
    selector = np.full((h, w), background, dtype=np.int8)
    view = _corner_view(selector, 2)  # bottom-left; whole-scene orientation follows
    view[3, :4] = frame
    view[:4, 3] = frame
    for pos, color in zip(rng.choice(9, 3, replace=False), rng.permutation(q)):
        view[int(pos)//3, int(pos)%3] = color
    grid, answer = selector.copy(), selector.copy()
    for i, (box, choices, labels) in enumerate(zip(boxes, slots, marks)):
        # Both solid and alternating outlines occur in each teaching pair.
        alternate = i % 2 == 0 if package < 2 else bool(rng.integers(2))
        word = tuple(int(c) for c in rng.permutation(edge_colors))
        if not alternate:
            word = (word[0], word[0])
        valid = [pts for pts in choices if len(pts) >= len(labels)]
        available = valid[int(rng.integers(len(valid)))]
        indices = rng.choice(len(available), len(labels), replace=False)
        dot_values = [(available[int(index)], int(color))
                      for index, color in zip(indices, rng.permutation(labels))]
        delete = set(q) <= set(labels)
        for y, x in _perimeter(box):
            grid[y, x] = word[(y+x) % 2]
            if not delete:
                answer[y, x] = word[(y+x) % 2]
        for p, value in dot_values:
            grid[p] = value
            if not delete:
                answer[p] = value
    t, b, l, r = shutter
    grid[t:b+1, l:r+1] = fill
    for p in _perimeter(shutter):
        grid[p] = frame
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    grid, answer = np.rot90(grid, turns), np.rot90(answer, turns)
    if flip:
        grid, answer = grid[:, ::-1], answer[:, ::-1]
    return {"input": grid.copy(), "output": answer.copy()}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    teaching_query = [int(c) for c in rng.choice(10, 3, replace=False)]
    backgrounds = ([int(c) for c in rng.permutation([c for c in range(10) if c not in teaching_query])]
                   + [int(c) for c in rng.permutation(teaching_query)])
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            package = i if split == "train" and i < 2 else 2
            for _ in range(8):
                pair = _make_pair(rng, package, split == "test", backgrounds[len(seen) % 10],
                                  teaching_query if package < 2 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input retries exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    records = [(info, cage) for info in infos for cage in info["cages"]]
    pos = [(i, c) for i, c in records if c["delete"]]
    neg = [(i, c) for i, c in records if not c["delete"]]
    assert pos and neg
    assert any(sum(c["counts"].values()) == 3 and c["labels"] == i["query"] for i, c in pos)
    assert any(len(c["labels"]) > 3 for _, c in neg)
    assert any(c["labels"]-i["query"] for i, c in pos)
    assert any(max(c["counts"].values()) > 1 for _, c in pos)
    for solid in (False, True):
        assert any((c["word"][0] == c["word"][1]) == solid for _, c in pos)
        assert any((c["word"][0] == c["word"][1]) == solid for _, c in neg)
    assert {len([c for c in i["cages"] if c["delete"]]) for i in infos} >= {1, 2}
    # Each missing identity has a visible hard negative in the teaching pairs.
    assert any({next(iter(q-c["labels"])) for i in infos if i["query"] == q
                for c in i["cages"] if not c["delete"] and len(q-c["labels"]) == 1} == set(q)
               for q in {i["query"] for i in infos})
    assert any(not c["delete"] and any(i["cages"][j]["delete"] for j in c["children"])
               for i, c in records)
    assert any(c["delete"] and any(not i["cages"][j]["delete"] for j in c["children"])
               for i, c in records)
    assert {i["fill"] == i["background"] for i in infos} == {False, True}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            info = _read(grid)
            assert np.array_equal(_render_solution(grid, info), output), "wrong reveal/deletion"
            assert np.array_equal(grid, pair["input"]), "reader mutated input"
            assert all(int(output[p]) == int(grid[p]) for p in info["selector"])
            # Every generated scene really needs A and B, including tests.
            assert not np.array_equal(_render_solution(grid, info, "no_filter"), output)
            assert not np.array_equal(_render_solution(grid, info, "no_completion"), output)
            infos[split].append(info)
    _training_coverage(infos["train"])
    assert any(not np.array_equal(_render_solution(p["input"], i, "borrow_children"), p["output"])
               for p, i in zip(puzzle["train"], infos["train"]))
    assert all(sum(bool(c["children"]) for c in i["cages"]) == 1 for i in infos["train"])
    assert all(sum(bool(c["children"]) for c in i["cages"]) == 2 for i in infos["test"])


def _deep_fixture():
    """An explicit depth-three case, outside the sampler's geometry schedule."""
    grid = np.zeros((30, 30), dtype=np.int8)
    # Top-right selector: frame 9, query {1,2,3}.
    view = _corner_view(grid, 1)
    view[3, :4] = 9
    view[:4, 3] = 9
    view[0, 0], view[1, 1], view[2, 2] = 1, 2, 3
    answer = grid.copy()
    boxes = [(0, 24, 0, 24), (4, 20, 4, 20), (8, 16, 8, 16)]
    dots = [[((2, 2), 1), ((2, 4), 2), ((2, 6), 4)],
            [((6, 6), 1), ((6, 8), 2), ((6, 10), 3), ((6, 12), 4)],
            []]
    # Only the middle cage matches; its empty child and incomplete parent stay.
    for i, (box, labels) in enumerate(zip(boxes, dots)):
        for y, x in _perimeter(box):
            value = 5 if i == 0 else (5, 6)[(y+x) % 2]
            grid[y, x] = value
            if i != 1:
                answer[y, x] = value
        for p, value in labels:
            grid[p] = value
            if i != 1:
                answer[p] = value
    shutter = (12, 20, 12, 20)
    t, b, l, r = shutter
    grid[t:b+1, l:r+1] = 8
    for p in _perimeter(shutter):
        grid[p] = 9
    return grid, answer


def self_test():
    """Bounded 42-seed invariants, transformations, counterfactuals and mutations."""
    shapes, layouts, puzzles, palettes, corners = set(), set(), set(), set(), set()
    counts, styles, fills = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                before = grid.copy()
                info = _read(grid)
                assert np.array_equal(solve(grid), output) and np.array_equal(grid, before)
                shapes.add(grid.shape)
                layouts.add((grid.shape, info["shutter"], tuple(c["box"] for c in info["cages"])))
                palettes.add((info["background"], info["frame"], info["fill"]))
                corners.add(info["corner"])
                fills.add(info["fill"] == info["background"])
                counts.add(len(info["cages"]))
                styles.update(c["word"][0] == c["word"][1] for c in info["cages"])
        # Spatial rearrangement of the three query cells leaves record decisions
        # unchanged; only the copied selector pixels move in the answer.
        pair = puzzle["train"][0]
        info = _read(pair["input"])
        changed, expected = pair["input"].copy(), pair["output"].copy()
        cv, ev = _corner_view(changed, info["corner"]), _corner_view(expected, info["corner"])
        cv[:3, :3] = np.roll(cv[:3, :3].copy(), 1, axis=0)
        ev[:3, :3] = cv[:3, :3]
        assert np.array_equal(solve(changed), expected)
        # Query identities do matter: substitute the three identities of a hard
        # negative, preserving every cage and dot position.
        witness = None
        for pair in puzzle["train"]:
            info = _read(pair["input"])
            witness = next((c for c in info["cages"] if not c["delete"] and len(c["labels"]) == 3), None)
            if witness:
                break
        assert witness is not None
        changed = pair["input"].copy()
        cv = _corner_view(changed, info["corner"])
        positions = list(zip(*np.nonzero(cv[:3, :3] != info["background"])))
        for p, value in zip(positions, sorted(witness["labels"])):
            cv[p] = value
        changed_info = _read(changed)
        record = next(c for c in changed_info["cages"] if c["box"] == witness["box"])
        assert record["delete"]
        changed_output = solve(changed)
        assert all(int(changed_output[p]) == info["background"] for p in record["full"] | set(record["dots"]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][j][k], again[s][j][k])
                       for s in puzzle for j in range(len(puzzle[s])) for k in ("input", "output"))
    assert len(puzzles) == 42 and len(shapes) >= 20 and len(layouts) >= 180
    assert len(palettes) >= 80 and corners == {0, 1, 2, 3}
    assert counts >= {3, 4, 5} and styles == {False, True} and fills == {False, True}
    # Completion, scope and pattern phase also work with a third nesting level,
    # an empty record, an edge-touching cage, and unseen geometry. Check every
    # orientation and a global permutation with background 0 mapped to 7.
    deep_input, deep_output = _deep_fixture()
    recolor = np.array([7, 9, 8, 0, 6, 3, 1, 2, 5, 4], dtype=np.int8)
    for turns in range(4):
        for flip in (False, True):
            inp, out = np.rot90(deep_input, turns), np.rot90(deep_output, turns)
            if flip:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(solve(recolor[inp]), recolor[out])
    for two_nests in (False, True):
        for extra in (False, True):
            requirements = [6, 5, 6] + ([5] if two_nests else []) + ([3] if extra else [])
            _fallback_geometry(two_nests, requirements, extra)
        for seed in range(3):
            pair = _make_pair(np.random.default_rng(seed), 2, two_nests, seed, force_fallback=True)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    for a, b in ((2, 1), (10, 1), (10, 4), (6, 3)):
        generate_puzzle(430+a+b, a, b)
    plain = generate_puzzle(140, 2, 1)
    plain = {s: [{k: v.tolist() for k, v in pair.items()} for pair in plain[s]] for s in plain}
    validate_generated_puzzle(plain)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(42, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
