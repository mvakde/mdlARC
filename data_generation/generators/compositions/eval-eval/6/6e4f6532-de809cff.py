"""Material-keyed repair stamps: negative registration, color compass, and repair.

Only standard-library modules and NumPy are used.  All inference is input-only;
the constructor certifies a separately constructed answer with that inference.
"""
from itertools import product
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_EIGHT = tuple(p for p in product((-1, 0, 1), repeat=2) if p != (0, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _EIGHT:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.add(q)
        result.append(cells)
    return result


def _transform(cells, t):
    result = []
    for (r, c), v in sorted(cells.items()):
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result.append((r, c, v))
    r0 = min(r for r, c, v in result)
    c0 = min(c for r, c, v in result)
    return {(r-r0, c-c0): v for r, c, v in result}


def _shift(cells, r, c):
    return {(rr+r, cc+c): v for (rr, cc), v in cells.items()}


def _shape(points):
    return tuple(sorted(_transform({p: 1 for p in points}, 0)))


def _canonical(points):
    return min(tuple(sorted(_transform({p: 1 for p in points}, t)))
               for t in range(8))


def _halo(points):
    return {(r+dr, c+dc) for r, c in points
            for dr, dc in product((-1, 0, 1), repeat=2)}


def _frame(shape, bg, sides):
    a = np.full(shape, bg, dtype=np.int8)
    a[0, :] = sides[0]
    a[-1, :] = sides[2]
    a[1:-1, -1] = sides[1]
    a[1:-1, 0] = sides[3]
    return a


def _paint(base, boxes):
    a = base.copy()
    for r0, c0, r1, c1, color in boxes:
        a[r0:r1, c0:c1] = color
    return a


def _label_directions(cells, bands):
    core = [p for p, v in cells.items() if v not in bands]
    if not core:
        return None
    r0, r1 = min(p[0] for p in core), max(p[0] for p in core)
    c0, c1 = min(p[1] for p in core), max(p[1] for p in core)
    labels = {}
    for p, v in cells.items():
        if v not in bands:
            continue
        r, c = p
        ds = [d for d, yes in enumerate((r < r0, c > c1, r > r1, c < c0)) if yes]
        if len(ds) != 1:
            return None
        labels[p] = ds[0]
    return labels


def _recover(observed, bg, fills, sides):
    """Dense row/column support fixes both boxes; compare both covering orders.

    A line outside the true extent has at most two noise pixels of that fill.
    Each true extreme line has at least three surviving pixels of that fill.
    These visible support conditions give an exact, bounded reconstruction.
    """
    boxes = []
    for fill in fills:
        mask = observed[1:-1, 1:-1] == fill
        rows = np.flatnonzero(mask.sum(axis=1) >= 3) + 1
        cols = np.flatnonzero(mask.sum(axis=0) >= 3) + 1
        if not len(rows) or not len(cols):
            raise ValueError("Missing dense rectangle evidence")
        box = (int(rows[0]), int(cols[0]), int(rows[-1])+1, int(cols[-1])+1, fill)
        if min(box[2]-box[0], box[3]-box[1]) < 6:
            raise ValueError("Rectangles must have sides at least six")
        boxes.append(box)
    a, b = boxes
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    if overlap < 6:
        raise ValueError("Two materially distinct rectangles must overlap")
    base = _frame(observed.shape, bg, sides)
    candidates = []
    for order in (boxes, boxes[::-1]):
        scene = _paint(base, order)
        cost = int(np.count_nonzero(scene[1:-1, 1:-1] != observed[1:-1, 1:-1]))
        candidates.append((cost, scene, order))
    candidates.sort(key=lambda item: item[0])
    loss, scene, order = candidates[0]
    if candidates[1][0] <= loss + 1:
        raise ValueError("Covering order is ambiguous")
    for r0, c0, r1, c1, fill in boxes:
        visible = scene[r0:r1, c0:c1] == fill
        correct = visible & (observed[r0:r1, c0:c1] == fill)
        if (correct.sum() < .78 * visible.sum()
                or min(correct[0].sum(), correct[-1].sum(),
                       correct[:, 0].sum(), correct[:, -1].sum()) < 3):
            raise ValueError("Insufficient surviving boundary or interior evidence")
        # Noise beyond a rectangle cannot masquerade as a supported line.
        outside = (observed == fill).copy()
        outside[r0:r1, c0:c1] = False
        if max(outside[1:-1, 1:-1].sum(axis=0).max(),
               outside[1:-1, 1:-1].sum(axis=1).max()) > 2:
            raise ValueError("Exterior fill noise is too dense")
    return scene, tuple(order)


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    if min(h, w) < 12:
        raise ValueError("The framed scene is too small")
    sides = (int(a[0, 0]), int(a[1, -1]), int(a[-1, 0]), int(a[1, 0]))
    bands = set(sides)
    if (len(bands) != 4 or not np.all(a[0] == sides[0])
            or not np.all(a[-1] == sides[2])
            or not np.all(a[1:-1, -1] == sides[1])
            or not np.all(a[1:-1, 0] == sides[3])):
        raise ValueError("Expected four distinct one-cell compass sides")
    counts = np.bincount(a[1:-1, 1:-1].ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if bg in bands or np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("Background must be uniquely most frequent in the interior")
    mask = a != bg
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    sources = []
    observed = a.copy()
    for comp in _components(mask):
        if any(int(a[p]) in bands for p in comp):
            cells = {p: int(a[p]) for p in comp}
            if not _label_directions(cells, bands):
                raise ValueError("Malformed loose compass stamp")
            sources.append(cells)
            for p in comp:
                observed[p] = bg
    if not 1 <= len(sources) <= 4:
        raise ValueError("Expected one to four isolated stamps")
    fills = tuple(sorted(set(map(int, np.unique(observed))) - bands - {bg}))
    if len(fills) != 2:
        raise ValueError("Expected exactly two rectangle fill roles")
    registrations = set()
    for source in sources:
        regs = set(source.values()) - bands - set(fills)
        if len(regs) != 1 or len(set(source.values()) & set(fills)) != 1:
            raise ValueError("Stamp needs one ink and one registration color")
        registrations.update(regs)
    if len(registrations) != 1:
        raise ValueError("Registration color must agree across all loose stamps")
    reg = registrations.pop()
    if len(bands | set(fills) | {bg, reg}) != 8:
        raise ValueError("All eight semantic roles must be distinct")
    scene, boxes = _recover(observed, bg, fills, sides)
    holes = set(map(tuple, np.argwhere((observed == bg) & np.isin(scene, fills))))
    if not holes:
        raise ValueError("Expected erased material cells")
    for source in sources:
        ink = next(iter(set(source.values()) & set(fills)))
        marks = {p for p, v in source.items() if v == reg}
        if not 1 <= len(marks) <= 4:
            raise ValueError("A stamp has one to four registration centers")
        # Ring evidence must be complete in a loose stamp, even at adjacent centers.
        for r, c in marks:
            for dr, dc in _EIGHT:
                q = (r+dr, c+dc)
                if source.get(q) != (reg if q in marks else ink):
                    raise ValueError("Loose stamp lacks a complementary 3x3 brush")
    if sum(sum(v == reg for v in s.values()) for s in sources) != len(holes):
        raise ValueError("Registration centers must exactly cover the recovered holes")
    return {"input": a, "background": bg, "sides": sides, "bands": bands,
            "fills": fills, "registration": reg, "sources": sources,
            "observed": observed, "scene": scene, "boxes": boxes, "holes": holes}


def _placements(source, info, use_compass=True, use_material=True):
    reg, holes = info["registration"], info["holes"]
    h, w = info["input"].shape
    ink = next(iter(set(source.values()) & set(info["fills"])))
    intended = next(f for f in info["fills"] if f != ink)
    result, seen = [], set()
    for t in range(8):
        cells = _transform(source, t)
        labels = _label_directions(cells, info["bands"])
        if labels is None or (use_compass and any(cells[p] != info["sides"][d]
                                                for p, d in labels.items())):
            continue
        marks = {p for p, v in cells.items() if v == reg}
        first = min(marks)
        for target in sorted(holes):
            dr, dc = target[0]-first[0], target[1]-first[1]
            moved = _shift(cells, dr, dc)
            centers = {(r+dr, c+dc) for r, c in marks}
            if not centers <= holes:
                continue
            if use_material and any(info["scene"][p] != intended for p in centers):
                continue
            if any(not (1 <= r < h-1 and 1 <= c < w-1) for r, c in moved):
                continue
            # A tab may never cover another stamp's registration center.
            if any(p in holes and v in info["bands"] for p, v in moved.items()):
                continue
            signature = tuple(sorted(moved.items()))
            if signature not in seen:
                seen.add(signature)
                result.append((centers, moved))
    return result


def _answers(info, use_compass=True, use_material=True, limit=2):
    """Exact cover, compatible simultaneous ink demands, deduplicated outputs."""
    choices = [_placements(s, info, use_compass, use_material) for s in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    reg, holes = info["registration"], info["holes"]
    answers = {}

    def visit(depth, covered, demands):
        if len(answers) >= limit:
            return
        if depth == len(order):
            if covered != holes:
                return
            out = info["scene"].copy()
            for p, v in demands.items():
                out[p] = v
            for p in holes:
                out[p] = reg
            answers[_key(out)] = out
            return
        for centers, cells in choices[order[depth]]:
            if covered & centers:
                continue
            new = demands.copy()
            compatible = True
            for p, v in cells.items():
                if p in holes:
                    continue  # All registration centers have final precedence.
                if p in new and new[p] != v:
                    compatible = False
                    break
                new[p] = v
            if compatible:
                visit(depth+1, covered | centers, new)
    visit(0, set(), {})
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    answers = _answers(info)
    if len(answers) != 1:
        raise ValueError("Input does not determine one repair imprint")
    return answers[0]


def _pattern(rng, count):
    points = {(0, 0)}
    while len(points) < count:
        r, c = sorted(points)[int(rng.integers(len(points)))]
        dr, dc = _EIGHT[int(rng.integers(8))]
        trial = points | {(r+dr, c+dc)}
        if (max(r for r, c in trial)-min(r for r, c in trial) <= 2
                and max(c for r, c in trial)-min(c for r, c in trial) <= 2):
            points = trial
    return set(_shape(points))


def _stamp(rng, marks, ink, reg, sides, simple=False, selected=None):
    core = _halo(marks)
    # Additional ink is an asymmetric tool footprint, not a filled bounding box.
    if not simple:
        tip = None
        for _ in range(int(rng.integers(2, 10))):
            frontier = sorted({(r+dr, c+dc) for r, c in core for dr, dc in _DIRS} - core)
            rng.shuffle(frontier)
            if tip is not None and rng.random() < .65:
                frontier.sort(key=lambda p: abs(p[0]-tip[0])+abs(p[1]-tip[1]))
            for q in frontier:
                trial = core | {q}
                if (max(r for r, c in trial)-min(r for r, c in trial) <= 6
                        and max(c for r, c in trial)-min(c for r, c in trial) <= 6):
                    core = trial
                    tip = q
                    break
    cells = {p: ink for p in core}
    cells.update({p: reg for p in marks})
    r0, r1 = min(r for r, c in core), max(r for r, c in core)
    c0, c1 = min(c for r, c in core), max(c for r, c in core)
    if selected is None:
        corner = int(rng.integers(4))
        selected = {corner, (corner+1) % 4}
        for d in range(4):
            if rng.random() < .25:
                selected.add(d)
    for d in sorted(selected):
        edge = sorted(p for p in core if (p[0] == r0, p[1] == c1,
                                         p[0] == r1, p[1] == c0)[d])
        rng.shuffle(edge)
        for r, c in edge[:1 if simple else int(rng.integers(1, min(2, len(edge))+1))]:
            dr, dc = _DIRS[d]
            cells[(r+dr, c+dc)] = sides[d]
    return _transform(cells, 0)


def _scene(rng, shape, bg, fills, sides):
    h, w = shape
    base = _frame(shape, bg, sides)
    for _ in range(160):
        boxes = []
        for fill in fills:
            rh = int(rng.integers(6, min(15, h-5)+1))
            rw = int(rng.integers(6, min(15, w-5)+1))
            r0 = int(rng.integers(1, h-rh))
            c0 = int(rng.integers(1, w-rw))
            boxes.append((r0, c0, r0+rh, c0+rw, fill))
        if rng.integers(2):
            boxes.reverse()
        clean = _paint(base, boxes)
        if np.count_nonzero(clean[1:-1, 1:-1] != bg) > .50*(h-2)*(w-2):
            continue
        try:
            recovered, _ = _recover(clean, bg, fills, sides)
            if np.array_equal(clean, recovered):
                return clean
        except ValueError:
            pass
    return None


def _origins(cells, shape):
    hh = 1+max(r for r, c in cells)
    ww = 1+max(c for r, c in cells)
    return [(r, c) for r in range(1, shape[0]-hh) for c in range(1, shape[1]-ww)]


def _compose(clean, maps, reg):
    centers = {p for cells in maps for p, v in cells.items() if v == reg}
    demands = {}
    for cells in maps:
        for p, v in cells.items():
            if p in centers:
                # The caller separately prohibits compass tabs at centers.
                if v not in (reg, *set(np.unique(clean[1:-1, 1:-1]))):
                    return None
                continue
            if p in demands and demands[p] != v:
                return None
            demands[p] = v
    out = clean.copy()
    for p, v in demands.items():
        out[p] = v
    for p in centers:
        out[p] = reg
    return out


def _construct(rng, palette, count, teaching, test, fallback=False):
    bg, f0, f1, reg, *sides = palette
    fills = (f0, f1)
    shape = (30, 30) if fallback else tuple(map(int, rng.integers(22 if count < 3 else 26, 31, size=2)))
    if fallback:
        clean = _paint(_frame(shape, bg, sides), [(2, 2, 11, 15, f0), (8, 8, 18, 18, f1)])
        count = 2
    else:
        clean = _scene(rng, shape, bg, fills, sides)
        if clean is None:
            return None
    shared = {(0, 0), (0, 1)}
    finals = []
    hole_union = set()
    inks = [f1, f0] + [fills[int(rng.integers(2))] for _ in range(count-2)] if count >= 2 else [fills[int(rng.integers(2))]]
    for i, ink in enumerate(inks):
        if teaching == 0:
            pattern = shared
        elif teaching == 1:
            pattern = {(0, 0), (0, 1)} if i == 0 else {(0, 0), (1, 1)}
        else:
            number = 4 if test and i == 0 else int(rng.integers(1, 4))
            pattern = _pattern(rng, number)
        if fallback and test:
            pattern = {(0, 0), (0, 1), (1, 0), (1, 1)} if i == 0 else shared
        tool = _stamp(rng, pattern, ink, reg, sides, simple=fallback,
                      selected={0, 3} if teaching == 0 or fallback else None)
        origins = _origins(tool, shape)
        rng.shuffle(origins)
        if fallback:
            # Anchor by registration, avoiding dependence on normalized tool size.
            mark = min(p for p, v in tool.items() if v == reg)
            anchor = (5, 5) if i == 0 else (13, 13)
            origins = [(anchor[0]-mark[0], anchor[1]-mark[1])]
        intended = next(f for f in fills if f != ink)
        found = False
        for r, c in origins:
            placed = _shift(tool, r, c)
            holes = {p for p, v in placed.items() if v == reg}
            if holes & hole_union or any(clean[p] != intended for p in holes):
                continue
            # Labels are not permitted on another registration center.
            proposed_holes = hole_union | holes
            if any(p in proposed_holes and v in sides for m in finals+[placed] for p, v in m.items()):
                continue
            if _compose(clean, finals+[placed], reg) is None:
                continue
            finals.append(placed)
            hole_union.update(holes)
            found = True
            break
        if not found:
            return None
    out = _compose(clean, finals, reg)
    inp = clean.copy()
    for p in hole_union:
        inp[p] = bg
    # Ordinary color errors, including seam bites, never become docking targets.
    eligible = [tuple(p) for p in np.argwhere(np.isin(clean, fills)) if tuple(p) not in hole_union]
    rng.shuffle(eligible)
    wrong_count = int(rng.integers(1 if teaching is not None else 0, 5))
    for p in eligible[:wrong_count]:
        inp[p] = f1 if clean[p] == f0 else f0
    sources = []
    blocked = _halo(set(map(tuple, np.argwhere(np.isin(clean, fills)))))
    for i, final in enumerate(finals):
        t = int(rng.integers(1, 4) if teaching == 0 and i == 0
                else rng.integers(4, 8) if teaching == 1 and i == 0
                else rng.integers(8))
        source = _transform(final, t)
        origins = _origins(source, shape)
        rng.shuffle(origins)
        if fallback:
            origins = [(2, 21) if i == 0 else (21, 3)]
        for r, c in origins:
            placed = _shift(source, r, c)
            if any(not (1 <= rr < shape[0]-1 and 1 <= cc < shape[1]-1) for rr, cc in placed):
                continue
            if placed.keys() & blocked:
                continue
            sources.append(placed)
            blocked.update(_halo(placed))
            break
        else:
            return None
    # Sparse exterior speckles may meet scene edges, but cannot merge with a tool.
    tool_halo = _halo({p for s in sources for p in s})
    exterior = [tuple(p) for p in np.argwhere(clean == bg)
                if 1 <= p[0] < shape[0]-1 and 1 <= p[1] < shape[1]-1 and tuple(p) not in tool_halo]
    rng.shuffle(exterior)
    for p in exterior[:int(rng.integers(1, 5))]:
        inp[p] = fills[int(rng.integers(2))]
    for source in sources:
        for p, v in source.items():
            inp[p] = v
    try:
        info = _parse(inp)
        answers = _answers(info)
        if (not np.array_equal(info["scene"], clean) or len(answers) != 1
                or not np.array_equal(answers[0], out)):
            return None
        if teaching == 0 and (len(_answers(info, use_compass=False)) < 2
                              or len(_answers(info, use_material=False)) < 2):
            return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                  (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    change_roles = bool(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            colors = list(map(int, rng.permutation(10)))
            palette = colors[:8] if change_roles else list(roles)+[c for c in colors if c not in roles][:4]
            teaching = index if split == "train" and index < 2 else None
            count = 2 if teaching is not None else int(rng.choice([1, 2, 3], p=[.25, .45, .30]))
            if split == "test":
                count = int(rng.choice([2, 3], p=[.40, .60]))
            for attempt in range(240):
                pair = _construct(rng, palette, count, teaching, split == "test", fallback=attempt >= 220)
                if pair is not None and _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("Bounded construction failed")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def _motions(info):
    result = []
    for source in info["sources"]:
        # Compass tabs fix a pose even before a target translation is chosen.
        ts = []
        for t in range(8):
            cells = _transform(source, t)
            labels = _label_directions(cells, info["bands"])
            if labels and all(cells[p] == info["sides"][d] for p, d in labels.items()):
                ts.append(t)
        result.append("translation" if 0 in ts else "rotation" if any(t < 4 for t in ts) else "reflection")
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, trainings, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            answers = _answers(info)
            assert len(answers) == 1 and np.array_equal(answers[0], out)
            assert np.array_equal(inp[0], out[0]) and np.array_equal(inp[-1], out[-1])
            assert np.array_equal(inp[:, 0], out[:, 0]) and np.array_equal(inp[:, -1], out[:, -1])
            reg = info["registration"]
            assert set(map(tuple, np.argwhere(out == reg))) == info["holes"]
            # A local check of B's complementary rings, separate from placement
            # enumeration.  Another center in the neighborhood stays highlighted.
            for r, c in info["holes"]:
                complement = next(f for f in info["fills"] if f != info["scene"][r, c])
                for dr, dc in _EIGHT:
                    q = (r+dr, c+dc)
                    assert out[q] == (reg if q in info["holes"] else complement)
            # Sources are erased before every destination is pasted.  A final
            # imprint may occupy part of a source's old exterior footprint.
            (trainings if split == "train" else tests).append(info)
    assert any(len(_answers(i, use_compass=False)) >= 2 for i in trainings)
    assert any(len(_answers(i, use_material=False)) >= 2 for i in trainings)
    assert any("rotation" in _motions(i) for i in trainings)
    assert any("reflection" in _motions(i) for i in trainings)
    # Demonstrations teach both complementary material directions and merging.
    assert any({int(i["scene"][p]) for p in i["holes"]} == set(i["fills"]) for i in trainings)
    assert any(any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                   for p in i["holes"] for q in i["holes"] if p != q) for i in trainings)
    assert all(max(sum(v == i["registration"] for v in s.values()) for s in i["sources"]) <= 3 for i in trainings)
    assert all(any(sum(v == i["registration"] for v in s.values()) == 4 for s in i["sources"]) for i in tests)


def self_test():
    shapes, geometry, patterns, counts, palettes = set(), set(), set(), set(), set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(tuple(_key(_grid(p[k])) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for pair in puzzle["train"]+puzzle["test"]:
            i = _parse(pair["input"])
            shapes.add(i["input"].shape)
            geometry.add(tuple(b[:4] for b in i["boxes"]))
            counts.add(len(i["sources"]))
            palettes.add((i["background"], i["fills"], i["registration"], i["sides"]))
            for s in i["sources"]:
                patterns.add(_canonical({p for p, v in s.items() if v == i["registration"]}))
            assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
            assert len(_answers(i, limit=3)) == 1
            # Exact cover and simultaneous painting cannot depend on object order.
            reversed_info = dict(i, sources=i["sources"][::-1])
            assert np.array_equal(_answers(reversed_info)[0], pair["output"])
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 25 and len(geometry) >= 100
    assert len(patterns) >= 10 and counts >= {1, 2, 3} and len(palettes) >= 40
    again = generate_puzzle(7)
    assert puzzles[7] == tuple(_key(_grid(p[k])) for s in ("train", "test") for p in again[s] for k in ("input", "output"))
    # Visible roles, not literal color values, control all inference.
    permutation = np.array([7, 3, 9, 0, 8, 1, 5, 2, 6, 4], dtype=np.int8)
    example = again["test"][0]
    assert np.array_equal(solve(permutation[example["input"]]), permutation[example["output"]])
    # Turn and reflect an entire scene, restoring the family's corner ownership.
    for t in range(8):
        a, b = example["input"].copy(), example["output"].copy()
        if t >= 4:
            a, b = np.fliplr(a), np.fliplr(b)
        a, b = np.rot90(a, t % 4).copy(), np.rot90(b, t % 4).copy()
        for grid in (a, b):
            grid[0, :] = grid[0, 1]
            grid[-1, :] = grid[-1, 1]
        assert np.array_equal(solve(a), b)
    # The broader rule permits four sources even though the public curriculum
    # samples at most three.  Exercise that input-only decoding path explicitly.
    fixture_rng = np.random.default_rng(54321)
    for _ in range(100):
        four = _construct(fixture_rng, list(range(8)), 4, None, False)
        if four is not None:
            break
    else:
        raise AssertionError("Could not construct the four-stamp fixture")
    assert len(_parse(four["input"])["sources"]) == 4
    assert np.array_equal(solve(four["input"]), four["output"])
    # Adjacent centers in opposite materials demand contradictory ring ink.
    # No object-list or center-list ordering is allowed to resolve that conflict.
    conflict = _paint(_frame((25, 25), 0, (4, 5, 6, 7)),
                      [(2, 2, 14, 14, 1), (8, 8, 20, 20, 2)])
    conflict[10, 7] = conflict[10, 8] = 0
    for ink, origin in ((2, (2, 17)), (1, (17, 2))):
        stamp = _stamp(fixture_rng, {(0, 0)}, ink, 3, (4, 5, 6, 7),
                       simple=True, selected={0, 3})
        for p, v in _shift(stamp, *origin).items():
            conflict[p] = v
    assert not _answers(_parse(conflict))
    try:
        solve(conflict)
    except ValueError:
        pass
    else:
        raise AssertionError("Contradictory complementary rings accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(3000+nt*10+ne, nt, ne)
            validate_generated_puzzle(puzzle)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")


if __name__ == "__main__":
    self_test()
