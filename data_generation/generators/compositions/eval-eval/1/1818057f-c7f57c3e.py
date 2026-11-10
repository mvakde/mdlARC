"""Exchange the plus-shaped part of the disagreement of two visible states.

Only NumPy and the standard library are used. Inference starts from the pixels;
the constructive renderer and sampled designs are not used by solve().
"""
from math import gcd
import numpy as np

CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _canonical_geometry(points):
    variants = []
    for reflected in (False, True):
        for turns in range(4):
            moved = []
            for r, c in points:
                c = -c if reflected else c
                for _ in range(turns):
                    r, c = c, -r
                moved.append((r, c))
            r0, c0, _, _ = _bounds(moved)
            variants.append(tuple(sorted((r - r0, c - c0) for r, c in moved)))
    return min(variants)


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _components(points):
    left = set(points)
    groups = []
    while left:
        start = min(left)
        left.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in left:
                        left.remove(p)
                        group.add(p)
                        stack.append(p)
        groups.append(group)
    return groups


def _plus_union(points):
    centers = {p for p in points
               if all((p[0] + dr, p[1] + dc) in points for dr, dc in CARDINALS)}
    union = set(centers)
    for r, c in centers:
        union.update((r + dr, c + dc) for dr, dc in CARDINALS)
    return centers, union


def _sequential_union(points):
    # A deliberately wrong, destructive alternative for demonstration checks.
    left, changed = set(points), set()
    for r, c in sorted(points):
        cross = {(r, c)} | {(r + dr, c + dc) for dr, dc in CARDINALS}
        if cross <= left:
            changed |= cross
            left -= cross
    return changed


def _primitive(crop):
    scale = 0
    for a in (crop, crop.T):
        for row in a:
            breaks = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            for lo, hi in zip(breaks, breaks[1:]):
                scale = gcd(scale, hi - lo)
    small = crop[::scale, ::scale]
    if not np.array_equal(crop, np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)):
        raise ValueError("nonuniform logical pixel blocks")
    return scale, small


def _separated(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h < yy or yy + hh < y or x + w < xx or xx + ww < x


def _read_input(grid):
    """Recover the two states, their registration, scales and comparison mask."""
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= grid.size:
        raise ValueError("a strict majority background is required")
    groups = _components(tuple(map(int, p)) for p in np.argwhere(grid != bg))
    objects = []
    for group in groups:
        r0, c0, r1, c1 = _bounds(group)
        crop = grid[r0:r1 + 1, c0:c1 + 1]
        if int(np.count_nonzero(crop != bg)) != len(group):
            raise ValueError("interleaved occurrences")
        scale, small = _primitive(crop)
        raw = tuple((int(r), int(c), int(small[r, c]))
                    for r, c in np.argwhere(small != bg))
        objects.append(((r0, c0), scale, raw))
    signatures = sorted({o[2] for o in objects})
    if len(signatures) != 2:
        raise ValueError("exactly two normalized input states are required")
    minimum = min(o[1] for o in objects)
    if minimum != 1 or {o[2] for o in objects if o[1] == 1} != set(signatures):
        raise ValueError("each state needs a scale-one reference")
    maps = [{(r, c): color for r, c, color in sig} for sig in signatures]
    common_colors = set(maps[0].values()) & set(maps[1].values())
    if not common_colors:
        raise ValueError("missing invariant colored scaffold")
    shifts = set()
    for color in sorted(common_colors):
        a = {p for p, v in maps[0].items() if v == color}
        b = {p for p, v in maps[1].items() if v == color}
        ar, ac, _, _ = _bounds(a)
        br, bc, _, _ = _bounds(b)
        shift = ar - br, ac - bc
        if a != {(r + shift[0], c + shift[1]) for r, c in b}:
            raise ValueError("a shared color must be an invariant whole-color pattern")
        shifts.add(shift)
    if len(shifts) != 1:
        raise ValueError("shared colored patterns disagree on registration")
    dr, dc = next(iter(shifts))
    maps[1] = {(r + dr, c + dc): v for (r, c), v in maps[1].items()}
    support = set(maps[0]) | set(maps[1])
    difference = {p for p in support if maps[0].get(p, bg) != maps[1].get(p, bg)}
    centers, active = _plus_union(difference)
    r0, c0, r1, c1 = _bounds(support)
    occurrences, rectangles = [], []
    for box, scale, raw in objects:
        state = signatures.index(raw)
        anchor = (box[0] - dr * scale * state, box[1] - dc * scale * state)
        rect = (anchor[0] + r0 * scale, anchor[1] + c0 * scale,
                (r1 - r0 + 1) * scale, (c1 - c0 + 1) * scale)
        y, x, h, w = rect
        if not (0 <= y and 0 <= x and y + h <= grid.shape[0] and x + w <= grid.shape[1]):
            raise ValueError("the complete two-state region must fit")
        if not all(_separated(rect, old) for old in rectangles):
            raise ValueError("two-state replacement rectangles need a blank separating line")
        rectangles.append(rect)
        occurrences.append((anchor, scale, state))
    return dict(background=bg, templates=maps, support=support, difference=difference,
                centers=centers, active=active, occurrences=occurrences,
                common_colors=common_colors, rectangles=rectangles)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    info = _read_input(grid)
    out = grid.copy()
    bg, maps = info["background"], info["templates"]
    for (y, x), scale, state in info["occurrences"]:
        for r, c in info["active"]:
            yy, xx = y + r * scale, x + c * scale
            out[yy:yy + scale, xx:xx + scale] = maps[1 - state].get((r, c), bg)
    return out


def _draw(grid, mapping, anchor, scale):
    for (r, c), color in mapping.items():
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        grid[y:y + scale, x:x + scale] = color


def _crop_map(mapping, bg):
    r0, c0, r1, c1 = _bounds(mapping)
    a = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), v in mapping.items():
        a[r - r0, c - c0] = v
    return a


def _make_design(rng, kind, change, registered=False):
    for _ in range(250):
        h, w = map(int, rng.integers(4, 8, size=2))
        cells = {(r, c) for r in range(h) for c in range(w)}
        d = set()
        if kind == "overlap":
            r, c = int(rng.integers(h - 2)), int(rng.integers(w - 3))
            d = {(r + a, c + b) for a in range(3) for b in range(4)}
        else:
            count = 1 if kind == "single" else int(rng.integers(2, 5))
            for _ in range(count):
                r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
                d.add((r, c))
                d.update((r + dr, c + dc) for dr, dc in CARDINALS)
        # Differences may be diagonal clutter, tails, additional crosses or near misses.
        available = sorted(cells - d)
        extra = int(rng.integers(1, min(len(available), 1 + h * w // 5) + 1))
        d.update(available[int(i)] for i in rng.permutation(len(available))[:extra])
        tip = None
        if registered:
            # One active endpoint extends beyond all scaffold material. Its
            # absence in state zero shifts the object's cropped bounding box.
            column = int(rng.integers(1, w - 1))
            d = {p for p in d if p[0] > 0}
            d |= {(1, column), (0, column), (2, column), (1, column - 1), (1, column + 1)}
            tip = (0, column)
        centers, active = _plus_union(d)
        if not active or not d - active:
            continue
        if kind == "overlap" and active == _sequential_union(d):
            continue
        # An irregular scaffold, with independently missing common sites.
        stable = {p for p in sorted(cells - d) if rng.random() < .83}
        if registered:
            stable = {p for p in stable if p[0] > 0}
        if len(stable) < 5:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg = palette[0]
        nstable = int(rng.integers(1, min(3, len(stable) - 3) + 1))
        n0, n1 = map(int, rng.integers(1, 3, size=2))
        stable_colors = palette[1:1 + nstable]
        accents = (palette[1 + nstable:1 + nstable + n0],
                   palette[1 + nstable + n0:1 + nstable + n0 + n1])
        ordered = sorted(stable)
        rng.shuffle(ordered)
        scaffold = {p: stable_colors[0] for p in ordered}
        for i, color in enumerate(stable_colors[1:]):
            scaffold[ordered[i]] = color
        maps = [dict(scaffold), dict(scaffold)]
        ordered = sorted(d)
        rng.shuffle(ordered)
        forced = sorted(active)
        rng.shuffle(forced)
        # 0=recolor, 1=appears in state 1, 2=appears in state 0.
        force_types = ([0, 1, 2] if change == "combined" else
                       [1, 2] if change == "relocate" else [0])
        fixed = dict(zip(forced, force_types))
        if registered:
            fixed[tip] = 1
            fixed[(1, tip[1])] = 0 if change == "combined" else 1
            other = next(p for p in forced if p not in (tip, (1, tip[1])))
            fixed[other] = 2
        for p in ordered:
            if change == "recolor":
                t = 0
            elif change == "relocate":
                t = int(rng.choice([1, 2]))
            else:
                t = int(rng.choice([0, 1, 2], p=[.6, .2, .2]))
            t = fixed.get(p, t)
            for state in (0, 1):
                if (t == 1 and state == 0) or (t == 2 and state == 1):
                    continue
                maps[state][p] = int(rng.choice(accents[state]))
        if any(len(_components(m)) != 1 for m in maps):
            continue
        if any(_primitive(_crop_map(m, bg))[0] != 1 for m in maps):
            continue
        hybrid = []
        for state in (0, 1):
            m = dict(maps[state])
            for p in active:
                m.pop(p, None)
                if p in maps[1 - state]:
                    m[p] = maps[1 - state][p]
            hybrid.append(m)
        if max(map(len, maps + hybrid)) > 40:
            continue
        return bg, maps, hybrid, active
    raise AssertionError("bounded design construction exhausted")


def _shelf(sizes, width, gap, order):
    rects = [None] * len(sizes)
    y = x = shelf_height = 0
    for i in order:
        h, w = sizes[i]
        if x and x + w > width:
            y, x, shelf_height = y + shelf_height + gap, 0, 0
        if y + h > 30:
            return None
        rects[i] = (y, x, h, w)
        x += w + gap
        shelf_height = max(shelf_height, h)
    return rects


def _layout(rng, maps, hybrids, scale_limit, count_range, force_scale, enlarged_state):
    support = set(maps[0]) | set(maps[1])
    r0, c0, r1, c1 = _bounds(support)
    logical = r1 - r0 + 1, c1 - c0 + 1
    max_area = max(map(len, maps + hybrids))
    options = []
    for attempt in range(80):
        n = int(rng.integers(count_range[0], count_range[1] + 1))
        scales = [1, 1] + [int(rng.choice([1, 1, 1] + list(range(2, scale_limit + 1))))
                            for _ in range(n - 2)]
        if force_scale:
            scales[2] = force_scale
            if enlarged_state is not None:
                scales[3] = force_scale
        if attempt == 79:
            scales = ([1, 1, force_scale, force_scale] if enlarged_state is not None
                      else [1, 1, force_scale] if force_scale else [1] * count_range[0])
        area = max_area * sum(s * s for s in scales)
        if 2 * area >= 900:
            continue
        sizes = [(logical[0] * s, logical[1] * s) for s in scales]
        orders = [sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True),
                  list(map(int, rng.permutation(len(sizes))))]
        for gap in (1, int(rng.integers(1, 4))):
            for width in range(max(w for h, w in sizes), 31):
                for order in orders:
                    rects = _shelf(sizes, width, gap, order)
                    if rects is not None:
                        options.append(rects)
        if options:
            break
    if not options:
        raise AssertionError("replacement regions cannot fit")
    rects = options[int(rng.integers(len(options)))]
    h0, w0 = max(y + h for y, x, h, w in rects), max(x + w for y, x, h, w in rects)
    h, w = int(rng.integers(h0, min(30, h0 + 9) + 1)), int(rng.integers(w0, min(30, w0 + 9) + 1))
    while h * w <= 2 * area:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    dy, dx = int(rng.integers(h - h0 + 1)), int(rng.integers(w - w0 + 1))
    rects = [(y + dy, x + dx, rh, rw) for y, x, rh, rw in rects]
    for _ in range(45):
        i = int(rng.integers(len(rects)))
        _, _, rh, rw = rects[i]
        candidate = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1)), rh, rw
        if all(_separated(candidate, other) for j, other in enumerate(rects) if j != i):
            rects[i] = candidate
    states = [0, 1] + [int(rng.integers(2)) for _ in scales[2:]]
    if enlarged_state is not None:
        states[2], states[3] = 0, 1
    places = [((y - r0 * s, x - c0 * s), s, state)
              for (y, x, _, _), s, state in zip(rects, scales, states)]
    return (h, w), places


def _pair(rng, kind, change, scale_limit, count_range, force_scale=0, enlarged_state=None):
    registered = change != "recolor" and (enlarged_state == 0 or rng.random() < .4)
    bg, maps, hybrids, active = _make_design(rng, kind, change, registered)
    shape, places = _layout(rng, maps, hybrids, scale_limit, count_range, force_scale, enlarged_state)
    inp, out = np.full(shape, bg, dtype=int), np.full(shape, bg, dtype=int)
    for anchor, scale, state in places:
        _draw(inp, maps[state], anchor, scale)
        _draw(out, hybrids[state], anchor, scale)
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, turns), np.rot90(out, turns)
    if flip:
        inp, out = np.fliplr(inp), np.fliplr(out)
    if enlarged_state == 0:
        templates = _read_input(inp)["templates"]
        if _bounds(templates[0])[:2] == _bounds(templates[1])[:2]:
            # A below/right endpoint changes extents but need not invalidate
            # top-left alignment. Put this required witness above/left.
            inp, out = np.rot90(inp, 2), np.rot90(out, 2)
    return {"input": inp.copy(), "output": out.copy()}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(["scale", "count", "fresh_geometry"], p=[.6, .2, .2]))
    puzzle, seen = {"train": [], "test": []}, set()
    training_scaffolds = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            mandatory = split == "train" and index < 2
            kind = ("single" if index == 0 else "overlap") if mandatory else str(rng.choice(["single", "overlap", "multiple"]))
            change = "combined" if mandatory else str(rng.choice(["recolor", "relocate", "combined"]))
            limit = 3 if split == "test" and shift == "scale" else 2
            forced = 2 if mandatory else 3 if split == "test" and shift == "scale" and index == 0 else 0
            count_range = (4, 4) if mandatory else (3, 4) if shift == "count" and split == "train" else (5, 7) if shift == "count" else (2, 7)
            if forced:
                count_range = max(3, count_range[0]), count_range[1]
            for _ in range(80):
                pair = _pair(rng, kind, change, limit, count_range, forced, index if mandatory else None)
                key = _key(pair["input"])
                info = _read_input(pair["input"])
                common = info["support"] - info["difference"]
                geometry = _canonical_geometry(common)
                if key not in seen and (split == "train" or geometry not in training_scaffolds):
                    seen.add(key)
                    if split == "train":
                        training_scaffolds.add(geometry)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("unique-input construction exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_scan(difference):
    # Validation uses a direct neighborhood enumeration, separate from union construction.
    selected = set()
    for r, c in difference:
        points = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        if all(p in difference for p in points):
            selected.update(points)
    return selected


def _edit_types(info):
    a, b = info["templates"]
    bg = info["background"]
    types = set()
    for p in info["active"]:
        x, y = a.get(p, bg), b.get(p, bg)
        types.add("add_erase" if bg in (x, y) else "recolor")
    return types


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(inp)
            infos[split].append(info)
            assert info["active"] and info["difference"] - info["active"], "need changed and retained differences"
            assert info["active"] == _cell_scan(info["difference"])
            expected = inp.copy()
            changed_mask = np.zeros_like(inp, dtype=bool)
            bg, maps = info["background"], info["templates"]
            for (y, x), s, state in info["occurrences"]:
                for r, c in info["active"]:
                    yy, xx = y + r * s, x + c * s
                    expected[yy:yy + s, xx:xx + s] = maps[1 - state].get((r, c), bg)
                    changed_mask[yy:yy + s, xx:xx + s] = True
            assert np.array_equal(out, expected), "incorrect partial state exchange"
            assert np.array_equal(inp != out, changed_mask), "edits must be exactly the expanded plus union"
            assert np.array_equal(out[~changed_mask], inp[~changed_mask])
            assert np.array_equal(solve(inp), out)
    train = infos["train"]
    required = set().union(*(_edit_types(i) for i in infos["test"]))
    assert required <= set().union(*(_edit_types(i) for i in train))
    assert {state for i in train for anchor, scale, state in i["occurrences"] if scale > 1} == {0, 1}, "show both enlarged exchange directions"
    assert any(i["active"] != _sequential_union(i["difference"]) for i in train), "show simultaneous overlapping masks"
    assert any(_bounds(i["templates"][0])[:2] != _bounds(i["templates"][1])[:2] for i in train), "show scaffold rather than cropped top-left registration"


def self_test():
    """Bounded regression and observable diversity audit over 42 seeds."""
    signatures, shapes, palettes, designs, differences = set(), set(), set(), set(), set()
    counts, scales, edits = set(), set(), set()
    edge = shifted_boxes = within_geometry = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        local = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                i = _read_input(inp)
                a, b = i["templates"]
                common = set(a) & set(b) - i["difference"]
                geometry = _canonical_geometry(common)
                designs.add(geometry)
                local.add(geometry)
                differences.add(_canonical_geometry(i["difference"]))
                shapes.add(inp.shape)
                palettes.add(tuple(map(int, np.unique(inp))))
                counts.add(len(i["occurrences"]))
                scales.update(s for _, s, _ in i["occurrences"])
                edits.update(_edit_types(i))
                shifted_boxes += _bounds(a) != _bounds(b)
                edge += any(np.any(g[0] != i["background"]) or np.any(g[-1] != i["background"])
                            or np.any(g[:, 0] != i["background"]) or np.any(g[:, -1] != i["background"])
                            for g in (inp, out))
        within_geometry += len(local) > 1
    assert len(signatures) == 42 and len(shapes) >= 40
    assert len(designs) >= 70 and len(differences) >= 70 and len(palettes) >= 30
    assert within_geometry == 42 and scales == {1, 2, 3}
    assert counts >= {2, 3, 4, 5, 6} and edits == {"recolor", "add_erase"}
    assert edge >= 5 and shifted_boxes >= 5
    p = generate_puzzle(17)
    assert tuple(_key(x[k]) for s in p for x in p[s] for k in x) == tuple(_key(x[k]) for s in p for x in generate_puzzle(17)[s] for k in x)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            generate_puzzle(100 + nt * 10 + ns, nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Exhaustively verify the exact local shape, including clipped neighborhoods.
    for bits in range(512):
        d = {(r, c) for r in range(3) for c in range(3) if bits & (1 << (r * 3 + c))}
        assert _plus_union(d)[1] == _cell_scan(d)
    block = {(r, c) for r in range(3) for c in range(4)}
    assert len(_plus_union(block)[1]) == 8 and len(_sequential_union(block)) == 5
    quiet = np.zeros((4, 9), dtype=int)
    for anchor, accent in (((1, 1), 2), ((1, 6), 3)):
        _draw(quiet, {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): accent}, anchor, 1)
    assert np.array_equal(solve(quiet), quiet), "no comparison plus means no edit"
    # Wrong output, whole-state exchange, and missing state must be rejected.
    bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("damaged output accepted")
    inp = p["test"][0]["input"]
    info = _read_input(inp)
    wrong = inp.copy()
    for (y, x), scale, state in info["occurrences"]:
        for r, c in info["difference"]:
            yy, xx = y + r * scale, x + c * scale
            wrong[yy:yy + scale, xx:xx + scale] = info["templates"][1 - state].get((r, c), info["background"])
    assert not np.array_equal(wrong, solve(inp)), "omitting A must change the answer"
    witness = next(x for x in p["train"] if _edit_types(_read_input(x["input"])) == {"recolor", "add_erase"})
    assert len(np.unique(witness["output"][witness["input"] != witness["output"]])) >= 3, "one highlight color cannot replace B's actual state values"
    bad["test"][0]["output"] = wrong
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("whole-state exchange accepted")
    missing = inp.copy()
    for (y, x), scale, state in info["occurrences"]:
        for r, c in info["support"]:
            yy, xx = y + r * scale, x + c * scale
            missing[yy:yy + scale, xx:xx + scale] = info["background"]
        _draw(missing, info["templates"][0], (y, x), scale)
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("absent counterpart state accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "scaffolds": len(designs), "difference_masks": len(differences),
            "palettes": len(palettes), "object_counts": sorted(counts),
            "scales": sorted(scales), "edge_pairs": edge, "shifted_state_boxes": shifted_boxes}
