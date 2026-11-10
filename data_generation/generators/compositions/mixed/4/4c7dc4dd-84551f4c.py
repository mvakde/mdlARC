"""A framed analogy calibrates the trigger color of an exact-contact cascade.

Only Python's standard library and NumPy are used. Generation, validation and
self-test do no file I/O, create no traces, and retain no corpus or global state.
"""
from collections import Counter
from itertools import product

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r + h < y + bh and c + w < x + bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r + h <= y or y + bh <= r or c + w <= x or x + bw <= c


def _frames(a):
    """Read isolated monochrome outline components, without color constants."""
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    boxes = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(a[r, c])
            seen[r, c] = True
            points, todo = [], [(r, c)]
            while todo:
                y, x = todo.pop()
                points.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and a[yy, xx] == color):
                        seen[yy, xx] = True
                        todo.append((yy, xx))
            y0 = min(p[0] for p in points)
            y1 = max(p[0] for p in points)
            x0 = min(p[1] for p in points)
            x1 = max(p[1] for p in points)
            bh, bw = y1-y0+1, x1-x0+1
            if (bh >= 4 and bw >= 4 and len(points) == 2*(bh+bw)-4
                    and all(y in (y0, y1) or x in (x0, x1) for y, x in points)):
                boxes.append((y0, x0, bh, bw, color))
    return boxes


def _extract(a):
    frames = _frames(a)
    leaves = [b for b in frames if not any(_inside(x, b) for x in frames)]
    if len(leaves) != 4 or len({b[2:4] for b in leaves}) != 1:
        raise ValueError("Expected four equal-sized panel interiors")
    groups = []
    for b in frames:
        members = [p for p in leaves if _inside(p, b)]
        if len(members) == 2 and _disjoint(*members):
            groups.append((b, sorted(members)))
    if (len(groups) != 2 or not _disjoint(groups[0][0], groups[1][0])
            or set(groups[0][1]) & set(groups[1][1])):
        raise ValueError("Expected two separate enclosures with two panels each")
    groups.sort()
    interiors = [[a[r+1:r+h-1, c+1:c+w-1].copy()
                  for r, c, h, w, _ in panels] for _, panels in groups]
    blank = [(i, j, int(p[0, 0])) for i in range(2)
             for j, p in enumerate(interiors[i]) if np.all(p == p[0, 0])]
    if len(blank) != 1:
        raise ValueError("Expected one uniformly blank interior")
    target_group, blank_slot, bg = blank[0]
    return {"bg": bg, "reference": interiors[1-target_group],
            "target": interiors[target_group][1-blank_slot], "groups": groups,
            "reference_group": 1-target_group, "blank_slot": blank_slot,
            "shape": interiors[0][0].shape}


def _rods(a, bg):
    """A rod is (column, height, color); the baseline is the last interior row."""
    h, w = a.shape
    rods = []
    for x in range(w):
        rows = np.flatnonzero(a[:, x] != bg)
        if not len(rows):
            continue
        n = len(rows)
        color = int(a[rows[0], x])
        if (n < 2 or not np.array_equal(rows, np.arange(h-n, h))
                or not np.all(a[rows, x] == color) or x+n > w):
            raise ValueError("Not separated, bottom-aligned upright rods")
        rods.append((x, n, color))
    if not rods or any(b[0] < a[0]+a[1] for a, b in zip(rods, rods[1:])):
        raise ValueError("Rod landing intervals must not overlap")
    return rods


def _closure(rods, trigger, recursive=True):
    by_x = {x: i for i, (x, _, _) in enumerate(rods)}
    active = {i for i, (_, _, c) in enumerate(rods) if c == trigger}
    todo = list(active)
    while todo:
        i = todo.pop()
        x, h, _ = rods[i]
        j = by_x.get(x+h)
        if j is not None and j not in active:
            active.add(j)
            if recursive:
                todo.append(j)
    return active


def _upright(shape, rods, bg):
    a = np.full(shape, bg, dtype=np.int8)
    for x, h, c in rods:
        a[-h:, x] = c
    return a


def _land(shape, rods, bg, trigger, recursive=True):
    active = _closure(rods, trigger, recursive)
    out = np.full(shape, bg, dtype=np.int8)
    for i, (x, h, color) in enumerate(rods):
        if i in active:
            out[-1, x:x+h] = color
        else:
            out[-h:, x] = color
    return out


def _topple(a, bg, trigger, recursive=True):
    return _land(a.shape, _rods(a, bg), bg, trigger, recursive)


def _model(scene):
    """The worked pair selects exactly one upright state and trigger color."""
    bg = scene["bg"]
    models = []
    for index in range(2):
        before, after = scene["reference"][index], scene["reference"][1-index]
        try:
            rods = _rods(before, bg)
        except ValueError:
            continue
        colors = {c for _, _, c in rods}
        if len(colors) != 2:
            continue
        for trigger in sorted(colors):
            if np.array_equal(_land(before.shape, rods, bg, trigger), after):
                models.append((trigger, index, rods))
    if len(models) != 1:
        raise ValueError("The complete pair must uniquely identify the trigger")
    return models[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied pixels; no training set or hidden state."""
    scene = _extract(_grid(input_grid))
    trigger, _, ref_rods = _model(scene)
    target_rods = _rods(scene["target"], scene["bg"])
    if {c for _, _, c in target_rods} != {c for _, _, c in ref_rods}:
        raise ValueError("Target and reference must share both rod colors")
    return _land(scene["shape"], target_rods, scene["bg"], trigger)


def _stats(rods, trigger):
    active = _closure(rods, trigger)
    by_x = {x: i for i, (x, _, _) in enumerate(rods)}
    longest = 0
    for i, (_, _, c) in enumerate(rods):
        if c != trigger:
            continue
        length, current = 0, i
        while current is not None:
            length += 1
            x, h, _ = rods[current]
            current = by_x.get(x+h)
        longest = max(longest, length)
    miss = any(i in active and j not in active and rods[j][2] != trigger
               and rods[j][0] == x+h+1
               for i, (x, h, _) in enumerate(rods)
               for j in (i+1,) if j < len(rods))
    return {"active": active, "longest": longest, "miss": miss,
            "inactive": any(i not in active for i in range(len(rods)))}


def _heights(rng, count, ceiling, budget, mixed=False):
    heights = [2]*count
    extra = int(rng.integers(int(mixed), min(budget-2*count, count*(ceiling-2))+1))
    for _ in range(extra):
        eligible = [i for i, h in enumerate(heights) if h < ceiling]
        heights[int(rng.choice(eligible))] += 1
    if mixed and len(set(heights)) == 1:
        heights[int(rng.integers(count))] -= 1
    return heights


def _reference_rods(rng, shape, trigger, passive):
    """Always show T->P->P, then a one-cell miss leaving P upright."""
    h, w = shape
    heights = _heights(rng, 4, min(h, 5), min(w-1, 13), mixed=h >= 3 and w >= 10)
    start = int(rng.integers(0, min(3, w-sum(heights))))
    rods, x = [], start
    for i, height in enumerate(heights):
        rods.append((x, height, trigger if i == 0 else passive))
        x += height + int(i == 2)
    # Optional leading passive and independent tail keep the mandatory witness.
    if start >= 2 and rng.random() < .6:
        rods.insert(0, (0, 2, passive))
    while x+2 <= w and len(rods) < 8 and rng.random() < .6:
        gap = int(rng.integers(0, min(3, w-x-1)))
        x += gap
        height = int(rng.integers(2, min(h, 5, w-x)+1))
        rods.append((x, height, trigger if rng.random() < .5 else passive))
        x += height
    return rods


def _anchor_target(rng, shape, trigger, passive):
    h, w = shape
    heights = _heights(rng, 4, min(h, 5), min(w-1, 11), mixed=True)
    x = int(rng.integers(0, w-sum(heights)))
    rods = []
    for i, height in enumerate(heights):
        rods.append((x, height, trigger if i == 1 else passive))
        x += height + int(i == 0)
    return rods


def _calibration_rods(rng, shape, first, second):
    """The same upright scene supports two informative, distinct cascades.

    Each trigger assignment demonstrates two passive transmissions followed
    by a one-cell halt. Only the worked result identifies which color starts.
    Mixed heights 2/3 are mandatory, and the seven rods fit without clipping.
    """
    h, w = shape
    heights = _heights(rng, 7, min(h, 3), min(w-2, 20), mixed=True)
    x = int(rng.integers(w-sum(heights)-2+1))
    colors = (first, second, second, second, first, first, first)
    rods = []
    for i, (height, color) in enumerate(zip(heights, colors)):
        rods.append((x, height, color))
        x += height + int(i in (2, 5))
    return rods


def _random_target(rng, shape, colors, split, long_chain=False):
    h, w = shape
    trigger, passive = colors
    if long_chain:
        count = int(rng.integers(5, min(8, (w-1)//2)+1))
        heights = _heights(rng, count, min(h, 3), w, mixed=True)
        slack = w-sum(heights)
        # A leading passive is permitted; at least four later rods then fall.
        leading = bool(slack >= 1 and count >= 6 and rng.random() < .5)
        x = int(rng.integers(0, max(1, slack-int(leading)+1)))
        rods = []
        for i, height in enumerate(heights):
            role = trigger if i == int(leading) else passive
            if i >= int(leading)+3 and rng.random() < .25:
                role = trigger
            rods.append((x, height, role))
            x += height + int(leading and i == 0)
        return rods
    for _ in range(160):
        count = int(rng.integers(2, min(4 if split == "train" else 9, w//2)+1))
        x, rods = int(rng.integers(0, 3)), []
        for i in range(count):
            if w-x < 2:
                break
            height = int(rng.integers(2, min(h, 5, w-x)+1))
            rods.append((x, height, colors[int(rng.integers(2))]))
            x += height + int(rng.choice((0, 0, 0, 1, 1, 2, 3)))
        if (len(rods) == count and {r[2] for r in rods} == set(colors)
                and (split != "train" or _stats(rods, trigger)["longest"] <= 3)):
            return rods
    # A bounded construction of the same rule, never a different fallback rule.
    return [(0, 2, passive), (3, 2, trigger)]


def _outline(a, box, color):
    r, c, h, w = box[:4]
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _layout(rng, shape):
    ph, pw = shape[0]+2, shape[1]+2
    options = []
    for axes, arrangement, gaps in product(product((0, 1), repeat=2), (0, 1),
                                             product((1, 2, 3), repeat=3)):
        sizes = [[ph+2, 2*pw+gaps[i]+2] if axis == 0
                 else [2*ph+gaps[i]+2, pw+2] for i, axis in enumerate(axes)]
        height = sum(s[0] for s in sizes)+gaps[2] if arrangement == 0 else max(s[0] for s in sizes)
        width = max(s[1] for s in sizes) if arrangement == 0 else sum(s[1] for s in sizes)+gaps[2]
        if max(height, width) <= 30:
            options.append((axes, arrangement, gaps, sizes))
    if not options:
        raise ValueError("Panel shape cannot fit four framed panels")
    axes, arrangement, gaps, sizes = options[int(rng.integers(len(options)))]
    sizes = [s.copy() for s in sizes]
    def extent():
        if arrangement == 0:
            return sum(s[0] for s in sizes)+gaps[2], max(s[1] for s in sizes)
        return max(s[0] for s in sizes), sum(s[1] for s in sizes)+gaps[2]
    for _ in range(int(rng.integers(0, 9))):
        i, axis = map(int, rng.integers(0, 2, size=2))
        sizes[i][axis] += 1
        if max(extent()) > 30:
            sizes[i][axis] -= 1
    eh, ew = extent()
    height = int(rng.integers(eh, min(30, eh+4)+1))
    width = int(rng.integers(ew, min(30, ew+4)+1))
    along_slack = height-eh if arrangement == 0 else width-ew
    before, between, _ = map(int, rng.multinomial(along_slack, [1/3]*3))
    boxes, panels = [], []
    cursor = before
    for i, ((gh, gw), axis) in enumerate(zip(sizes, axes)):
        if arrangement == 0:
            top, left = cursor, int(rng.integers(width-gw+1))
            cursor += gh+gaps[2]+between
        else:
            top, left = int(rng.integers(height-gh+1)), cursor
            cursor += gw+gaps[2]+between
        boxes.append((top, left, gh, gw))
        if axis == 0:
            start = int(rng.integers(1, gw-2*pw-gaps[i]))
            positions = [(top+int(rng.integers(1, gh-ph)), left+start+j*(pw+gaps[i])) for j in range(2)]
        else:
            start = int(rng.integers(1, gh-2*ph-gaps[i]))
            positions = [(top+start+j*(ph+gaps[i]), left+int(rng.integers(1, gw-pw))) for j in range(2)]
        panels.append(positions)
    return (height, width), boxes, panels


def _carrier(rng, shape, palette):
    canvas_shape, boxes, positions = _layout(rng, shape)
    bg, a, b, frame, g0, g1, w0, w1, w2, f1 = map(int, palette)
    candidates = [bg, a, b, w0, w1, w2]
    wallpaper = rng.choice(candidates, int(rng.integers(2, 7)), replace=False)
    rr, cc = np.indices(canvas_shape)
    rb, cb = map(int, rng.integers(1, 3, size=2))
    canvas = wallpaper[(rr//rb+cc//cb+int(rng.integers(6))) % len(wallpaper)].astype(np.int8)
    frame_colors = [frame, f1 if rng.random() < .5 else frame]
    for i, box in enumerate(boxes):
        _outline(canvas, box, (g0, g1)[i])
        for r, c in positions[i]:
            _outline(canvas, (r, c, shape[0]+2, shape[1]+2), frame_colors[i])
    return canvas, positions


def _scene_pair(canvas, positions, shape, ref_rods, target_rods, bg, trigger,
                reference_group, before_slot, target_slot):
    ref_before = _upright(shape, ref_rods, bg)
    ref_after = _land(shape, ref_rods, bg, trigger)
    target = _upright(shape, target_rods, bg)
    contents = [[None, None], [None, None]]
    contents[reference_group][before_slot] = ref_before
    contents[reference_group][1-before_slot] = ref_after
    contents[1-reference_group][target_slot] = target
    contents[1-reference_group][1-target_slot] = np.full(shape, bg, dtype=np.int8)
    grid = canvas.copy()
    for i in range(2):
        for j, (r, c) in enumerate(positions[i]):
            grid[r+1:r+1+shape[0], c+1:c+1+shape[1]] = contents[i][j]
    return {"input": grid, "output": _land(shape, target_rods, bg, trigger)}


def _anchor_twins(rng, palette):
    shape = 3, int(rng.integers(20, 25))
    bg, trigger, passive = map(int, palette[:3])
    if rng.integers(2):
        trigger, passive = passive, trigger
    target = _anchor_target(rng, shape, trigger, passive)
    reference = _calibration_rods(rng, shape, trigger, passive)
    canvas, positions = _carrier(rng, shape, palette)
    slots = tuple(map(int, rng.integers(0, 2, size=3)))
    first = _scene_pair(canvas, positions, shape, reference, target, bg, trigger, *slots)
    second = _scene_pair(canvas, positions, shape, reference, target, bg, passive, *slots)
    twins = [first, second]
    rng.shuffle(twins)
    return twins


def _new_pair(rng, base_palette, split, first_test=False):
    palette = base_palette if rng.random() < .4 else rng.permutation(10)
    bg, trigger, passive = map(int, palette[:3])
    if rng.integers(2):
        trigger, passive = passive, trigger
    if first_test:
        shape = 3, int(rng.integers(16, 25))
    elif rng.random() < (.3 if split == "train" else .5):
        shape = int(rng.integers(2, 4)), int(rng.integers(12, 25))
    else:
        shape = int(rng.integers(3, 8)), int(rng.integers(9, 12))
    target = _random_target(rng, shape, (trigger, passive), split, long_chain=first_test)
    reference = _reference_rods(rng, shape, trigger, passive)
    canvas, positions = _carrier(rng, shape, palette)
    slots = tuple(map(int, rng.integers(0, 2, size=3)))
    return _scene_pair(canvas, positions, shape, reference, target, bg, trigger, *slots)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    puzzle = {"train": _anchor_twins(rng, palette), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", a), ("test", b)):
        while len(puzzle[split]) < count:
            for _ in range(32):
                pair = _new_pair(rng, palette, split, first_test=split == "test" and not puzzle[split])
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct input sampling exhausted its bounded retries")
    return puzzle


def _check_physics(before, after, bg, trigger):
    rods = _rods(before, bg)
    active = _closure(rods, trigger)
    assert Counter(map(int, before.flat)) == Counter(map(int, after.flat)), "Rod colors/area must be conserved"
    allowed = np.zeros(before.shape, dtype=bool)
    for i, (x, h, color) in enumerate(rods):
        if i in active:
            assert np.all(after[-h:-1, x] == bg), "Fallen upper cells must be vacated"
            assert np.all(after[-1, x:x+h] == color), "Height must equal reach"
            allowed[-h:, x] = True
            allowed[-1, x:x+h] = True
        else:
            assert np.all(after[-h:, x] == color), "Unreached rods must remain upright"
    assert np.array_equal(before[~allowed], after[~allowed]), "Unrelated cells changed"
    return rods


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate inputs"
            seen.add(_key(inp))
            scene = _extract(inp)
            trigger, index, ref = _model(scene)
            bg = scene["bg"]
            assert out.shape == scene["shape"]
            assert np.array_equal(out, solve(inp))
            _check_physics(scene["reference"][index], scene["reference"][1-index], bg, trigger)
            rs = _stats(ref, trigger)
            assert rs["longest"] >= 3 and rs["miss"] and rs["inactive"], "Reference needs recursion and a one-cell halt"
            rods = _check_physics(scene["target"], out, bg, trigger)
            colors = {c for _, _, c in rods}
            assert len(colors) == 2 and colors == {c for _, _, c in ref}
            other = next(c for c in colors if c != trigger)
            assert not np.array_equal(out, _topple(scene["target"], bg, other)), "Analogy must affect the answer"
            stats = _stats(rods, trigger)
            record = (scene, trigger, rods, stats)
            (training if split == "train" else testing).append(record)
    # The first two demonstrations are reference-dependent counterfactual twins.
    s0, t0, rods0, _ = training[0]
    s1, t1, rods1, _ = training[1]
    assert t0 != t1 and rods0 == rods1 and s0["bg"] == s1["bg"]
    assert np.array_equal(s0["target"], s1["target"])
    _, before0, _ = _model(s0)
    _, before1, _ = _model(s1)
    assert before0 == before1 and np.array_equal(s0["reference"][before0], s1["reference"][before1])
    r, c, h, w, _ = s0["groups"][s0["reference_group"]][1][1-before0]
    unchanged = np.ones(np.asarray(puzzle["train"][0]["input"]).shape, dtype=bool)
    unchanged[r+1:r+h-1, c+1:c+w-1] = False
    assert np.array_equal(np.asarray(puzzle["train"][0]["input"])[unchanged],
                          np.asarray(puzzle["train"][1]["input"])[unchanged])
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    assert any(not np.array_equal(_topple(s["target"], s["bg"], t, False),
                                 _topple(s["target"], s["bg"], t)) for s, t, _, _ in training)
    assert any(st["miss"] for _, _, _, st in training)
    assert max(len(r) for _, _, r, _ in training) <= 4
    assert len(testing[0][2]) >= 5 and testing[0][3]["longest"] >= 4
    assert not np.array_equal(_topple(testing[0][0]["target"], testing[0][0]["bg"], testing[0][1], False),
                              _topple(testing[0][0]["target"], testing[0][0]["bg"], testing[0][1]))


def self_test():
    """Bounded 42-seed sweep plus explicit causal and interface diagnostics."""
    shapes, geometries, palettes, layouts, rod_counts, heights = set(), set(), set(), set(), set(), set()
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for p in puzzle["train"]+puzzle["test"]))
        for pair in puzzle["train"]+puzzle["test"]:
            a = pair["input"]
            scene = _extract(a)
            trigger, _, _ = _model(scene)
            rods = _rods(scene["target"], scene["bg"])
            shapes.add((a.shape, scene["shape"]))
            geometries.add((scene["shape"], tuple((x, h, int(c == trigger)) for x, h, c in rods)))
            palettes.add((scene["bg"], trigger, tuple(sorted({c for _, _, c in rods}))))
            layouts.add(tuple(tuple(p[:4]) for _, ps in scene["groups"] for p in ps))
            rod_counts.add(len(rods))
            heights.update(h for _, h, _ in rods)
            original = a.copy()
            assert np.array_equal(solve(a), pair["output"])
            assert np.array_equal(a, original), "solve mutated its input"
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(geometries) >= 90 and len(layouts) >= 90
    assert len(palettes) >= 35 and len(rod_counts) >= 5 and {2, 3, 4, 5} <= heights
    # Height is causal: the same next column misses at height 2, hits at 3.
    shape = (4, 11)
    short = [(0, 2, 1), (3, 3, 2), (6, 2, 2)]
    tall = [(0, 3, 1), (3, 3, 2), (6, 2, 2)]
    assert _closure(short, 1) == {0}
    assert _closure(tall, 1) == {0, 1, 2}
    expected = np.zeros(shape, dtype=np.int8)
    expected[-1, :8] = [1, 1, 1, 2, 2, 2, 2, 2]
    assert np.array_equal(_land(shape, tall, 0, 1), expected)
    assert _closure(tall, 1, False) == {0, 1}
    # A passive can hit another initiator; activation is idempotent.
    joined = [(0, 2, 1), (2, 3, 2), (5, 2, 1), (8, 2, 2)]
    assert _closure(joined, 1) == {0, 1, 2}
    # Panel order is presentation, not direction: all three independent swaps
    # (reference order, target order, enclosing groups) preserve the answer.
    example = generate_puzzle(0)["train"][0]
    for ref_swap, target_swap, group_swap in product((False, True), repeat=3):
        inp = example["input"].copy()
        scene = _extract(inp)
        contents = [[inp[r+1:r+h-1, c+1:c+w-1].copy()
                     for r, c, h, w, _ in panels] for _, panels in scene["groups"]]
        if ref_swap:
            contents[scene["reference_group"]].reverse()
        if target_swap:
            contents[1-scene["reference_group"]].reverse()
        if group_swap:
            contents.reverse()
        for panels, interiors in zip((g[1] for g in scene["groups"]), contents):
            for (r, c, h, w, _), interior in zip(panels, interiors):
                inp[r+1:r+h-1, c+1:c+w-1] = interior
        assert np.array_equal(solve(inp), example["output"])
    # Color identities are read locally, including nonzero backgrounds.
    remap = np.array([9, 3, 7, 0, 5, 1, 8, 2, 4, 6], dtype=np.int8)
    assert np.array_equal(solve(remap[example["input"]]), remap[example["output"]])
    # Determinism and every legal count combination, without retaining runs.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(991, a, b)
            validate_generated_puzzle(p)
            q = generate_puzzle(991, a, b)
            assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                       for x, y in zip(p[s], q[s]) for k in ("input", "output"))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0)
    listed = {s: [{k: v.tolist() for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(listed)
    assert np.array_equal(solve(listed["test"][0]["input"]), p["test"][0]["output"])
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shape_combinations": len(shapes),
            "rod_geometries": len(geometries), "layouts": len(layouts), "palettes": len(palettes),
            "target_rod_counts": sorted(rod_counts), "rod_heights": sorted(heights)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
