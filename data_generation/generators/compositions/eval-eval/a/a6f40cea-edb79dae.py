"""Rectangle-registered stencil interference.

The input reader has no access to sampling state. Legend cards explicitly bound
the stencil, including disconnected islands and empty outer rows or columns.
"""
from __future__ import annotations

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _bbox(points):
    a = np.asarray(list(points), dtype=int)
    assert len(a)
    return int(a[:, 0].min()), int(a[:, 0].max()), int(a[:, 1].min()), int(a[:, 1].max())


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _inside(p, box, strict=False):
    y, x = p
    t, b, l, r = box
    return t < y < b and l < x < r if strict else t <= y <= b and l <= x <= r


def _draw_frame(grid, box, color):
    t, b, l, r = box
    grid[t, l:r+1] = grid[b, l:r+1] = color
    grid[t:b+1, l] = grid[t:b+1, r] = color


def _read_input(input_grid):
    """Parse the window, explicit legend cards, and color-grouped rectangles."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(int(grid[p]) == bg for p in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))), "background corner convention"
    candidates = []
    for color in np.unique(grid):
        if color == bg:
            continue
        points = set(map(tuple, np.argwhere(grid == color)))
        box = _bbox(points)
        t, b, l, r = box
        if b-t < 7 or r-l < 7 or points != _perimeter(box):
            continue
        inner = grid[t+1:b, l+1:r]
        if len(np.unique(inner)) == 1 and int(inner[0, 0]) != int(color):
            candidates.append((box, int(color), int(inner[0, 0])))
    assert len(candidates) == 1, "unique exclusive-color opaque window required"
    window, frame, fill = candidates[0]
    t, b, l, r = window
    exterior = grid != bg
    exterior[t:b+1, l:r+1] = False
    cards, card_cells = {}, np.zeros(grid.shape, dtype=bool)
    # A card's connected border bounds the whole tile; stencil islands need not
    # connect to that border, or to each other.
    for group in _components(exterior):
        a, z, c, d = _bbox(group)
        if z-a != d-c or not 3 <= z-a <= 5:
            continue
        box = (a, z, c, d)
        source = int(grid[a, c])
        if source in (bg, frame, fill):
            continue
        if not all(int(grid[p]) == source for p in _perimeter(box)):
            continue
        tile = grid[a+1:z, c+1:d]
        targets = set(map(int, np.unique(tile))) - {bg}
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        if target in (source, frame, fill):
            continue
        assert not np.all(tile == target), "legend stencils must include inactive cells"
        assert source not in cards, "one legend card per source"
        cards[source] = dict(box=box, mask=(tile == target).copy(), target=target)
        card_cells[a:z+1, c:d+1] = True
    assert 1 <= len(cards) <= 3, "need 1..3 explicit stencil cards"
    sizes = {c["mask"].shape for c in cards.values()}
    assert len(sizes) == 1
    n = next(iter(sizes))[0]
    assert 2 <= n <= 4
    boxes = [c["box"] for c in cards.values()]
    for i, (a, z, c, d) in enumerate(boxes):
        assert not any(_inside(p, window) for p in _perimeter((a, z, c, d)))
        for aa, zz, cc, dd in boxes[i+1:]:
            assert z+1 < aa or zz+1 < a or d+1 < cc or dd+1 < c, "cards need a blank gap"
    remaining = exterior & ~card_cells
    assert set(map(int, np.unique(grid[remaining]))) == set(cards)
    plates = []
    for source in sorted(cards):
        observed = set(map(tuple, np.argwhere(remaining & (grid == source))))
        box = _bbox(observed)
        a, z, c, d = box
        assert z-a >= 7 and d-c >= 7, "plate sides must be at least eight cells"
        full = _perimeter(box)
        assert observed <= full, "source pixels must lie on one rectangle"
        visible = {p for p in full if not _inside(p, window)}
        assert _bbox(visible) == box, "all four extrema need exterior witnesses"
        assert len({p for p in full if _inside(p, window, True)}) >= 2
        plates.append(dict(source=source, box=box, observed=observed,
                           mask=cards[source]["mask"], target=cards[source]["target"]))
    # At a visible crossing, one outline may conceal another. Missing pixels
    # are permitted only where another recovered perimeter explains them.
    for plate in plates:
        for p in _perimeter(plate["box"]):
            if _inside(p, window):
                continue
            if int(grid[p]) != plate["source"]:
                assert any(other["source"] == int(grid[p]) and
                           p in _perimeter(other["box"]) for other in plates)
    return dict(grid=grid, background=bg, frame=frame, fill=fill, window=window,
                cards=cards, plates=plates, n=n)


def _layers(scene, window_anchor=False, solid=False, anchor="plate"):
    t, b, l, r = scene["window"]
    yy, xx = np.indices((b-t-1, r-l-1))
    yy, xx = yy+t+1, xx+l+1
    layers = []
    for p in scene["plates"]:
        a, z, c, d = p["box"]
        inside = (yy > a) & (yy < z) & (xx > c) & (xx < d)
        if solid:
            active = inside
        else:
            oy, ox = (t+1, l+1) if window_anchor else (a+1, c+1)
            if anchor == "border":
                oy, ox = a, c
            elif anchor == "canvas":
                oy, ox = 0, 0
            n = p["mask"].shape[0]
            active = inside & p["mask"][(yy-oy) % n, (xx-ox) % n]
        layers.append(active)
    return layers


def _render(scene, *, window_anchor=False, solid=False, overlay=False, raw=False, omit=None, anchor="plate"):
    layers = _layers(scene, window_anchor, solid, anchor)
    count = np.zeros(layers[0].shape, dtype=int)
    for i, active in enumerate(layers):
        if i != omit:
            count += active
    out = np.full(count.shape, scene["fill"], dtype=np.int8)
    for i, (plate, active) in enumerate(zip(scene["plates"], layers)):
        if i != omit:
            out[active if overlay else active & (count == 1)] = plate["source"] if raw else plate["target"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; family conventions are explicit and fixed."""
    return _render(_read_input(input_grid))


def _orient_box(box, height, width, turns, flip):
    points = []
    for y, x in ((box[0], box[2]), (box[1], box[3])):
        h, w = height, width
        for _ in range(turns):
            y, x, h, w = w-1-x, y, w, h
        if flip:
            x = w-1-x
        points.append((y, x))
    return _bbox(points)


def _geometry(rng, n, k, fallback=False):
    h = 30 if fallback else int(rng.integers(max(25, n+24), 31))
    w = 30 if fallback else int(rng.integers(23, 31))
    lo, hi = n+6, h-2
    if fallback:
        window = (14, 25, 8, 23)
        plates = [(10, 24, 2, 21), (18, 28, 13, 27), (12, 22, 4, 19)][:k]
    else:
        oh = int(rng.integers(6, min(14, hi-lo-5)+1))
        ow = int(rng.integers(6, min(16, w-8)+1))
        t = int(rng.integers(lo+2, hi-oh-2))
        l = int(rng.integers(3, w-ow-4))
        window = (t, t+oh+1, l, l+ow+1)
        plates = []
        for _ in range(500):
            a, z = sorted(map(int, rng.choice(np.arange(lo, hi+1), 2, replace=False)))
            c, d = sorted(map(int, rng.choice(np.arange(1, w-1), 2, replace=False)))
            box = (a, z, c, d)
            if z-a < 7 or d-c < 7 or box in plates:
                continue
            full = _perimeter(box)
            visible = {p for p in full if not _inside(p, window)}
            hidden = {p for p in full if _inside(p, window, True)}
            if visible and _bbox(visible) == box and len(hidden) >= 2:
                plates.append(box)
                if len(plates) == k:
                    break
        if len(plates) != k:
            return None
    # Shuffle and disperse card positions independently of source/map ordering.
    side = n+2
    slack = w-2 - (k*side + (k-1))
    gaps = rng.multinomial(slack, np.full(k+1, 1/(k+1)))
    x, cards = 1+int(gaps[0]), []
    for i in range(k):
        y = int(rng.integers(1, 3))
        cards.append((y, y+side-1, x, x+side-1))
        x += side+1+int(gaps[i+1])
    rng.shuffle(cards)
    q, flip = int(rng.integers(4)), bool(rng.integers(2))
    orient = lambda box: _orient_box(box, h, w, q, flip)
    return ((w, h) if q % 2 else (h, w)), orient(window), list(map(orient, plates)), list(map(orient, cards))


def _mask(rng, n, informative=False):
    y, x = np.indices((n, n))
    kind = int(rng.integers(7))
    if kind == 0:
        m = (x+y) % 2 == int(rng.integers(2))
    elif kind == 1:
        m = (x == y) | ((x+y == n-1) & bool(rng.integers(2)))
    elif kind == 2:
        m = y % 2 == int(rng.integers(2))
    elif kind == 3:
        m = (x == n//2) | (y == n//2)
    elif kind == 4:
        m = (x == 0) | (x == n-1) | (y == 0) | (y == n-1)
    else:
        m = rng.random((n, n)) < float(rng.uniform(.25, .8))
    if not m.any():
        m[int(rng.integers(n)), int(rng.integers(n))] = True
    if m.all():
        m[int(rng.integers(n)), int(rng.integers(n))] = False
    if informative:
        for _ in range(40):
            m = rng.random((n, n)) < float(rng.uniform(.35, .75))
            transforms = [np.rot90(m, q) for q in (1, 2, 3)]
            if n >= 3:
                transforms += [np.rot90(np.fliplr(m), q) for q in range(4)]
            if m.any() and not m.all() and all(not np.array_equal(m, other) for other in transforms):
                break
        else:
            m = np.zeros((n, n), bool)
            m[0, :] = m[:, 0] = True
            m[-1, 0] = False
    return np.rot90(m, int(rng.integers(4))).copy()


def _mapping(rng, sources, spare, demo):
    if demo == 0:
        return {s: spare[0] for s in sources}
    if demo == 1:
        return {sources[0]: sources[1], sources[1]: spare[0]}
    kind = int(rng.integers(5))
    if kind == 0:
        return {s: int(rng.choice(spare)) for s in sources}
    if kind == 1 and len(sources) > 1:
        order = list(map(int, rng.permutation(sources)))
        return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}
    if kind == 2:
        return {s: spare[0] for s in sources}
    if kind == 3:
        order = list(map(int, rng.permutation(sources)))
        return {s: order[i+1] if i+1 < len(order) else spare[0] for i, s in enumerate(order)}
    return {s: int(rng.choice([c for c in sources+spare if c != s])) for s in sources}


def _evidence(scene, output, demo=False):
    layers = _layers(scene)
    count = sum(layers)
    assert count.max() <= 2, "three-way active interference is outside this family"
    assert all(active.any() for active in layers), "each plate reaches the window"
    assert all(not np.array_equal(output, _render(scene, omit=i)) for i in range(len(layers))), "redundant plate"
    assert np.any(output != scene["fill"]), "empty answer"
    if len(layers) > 1:
        assert np.any(count > 1), "need stencil interference"
        assert all(np.any(active & (count == 1)) for active in layers), "each source needs exclusive ink"
    if demo:
        assert not np.array_equal(output, _render(scene, window_anchor=True)), "anchor shortcut"
        assert not np.array_equal(output, _render(scene, anchor="border")), "border-origin shortcut"
        assert not np.array_equal(output, _render(scene, anchor="canvas")), "global-origin shortcut"
        assert not np.array_equal(output, _render(scene, solid=True)), "solid-rectangle shortcut"
        assert not np.array_equal(output, _render(scene, overlay=True)), "ordinary overpainting shortcut"
        assert not np.array_equal(output, _render(scene, raw=True)), "ignore color map shortcut"
        # At least one correctly colored motif cell visibly repeats one tile
        # period away, so finite single stamping cannot explain the demos.
        n = scene["n"]
        assert any(np.any((a[:-n] & a[n:] & (count[:-n] == 1) & (count[n:] == 1))) or
                   np.any((a[:, :-n] & a[:, n:] & (count[:, :-n] == 1) & (count[:, n:] == 1)))
                   for a in layers), "no exposed repetition"
        for q in (1, 2, 3):
            changed = dict(scene, plates=[dict(p, mask=np.rot90(p["mask"], q)) for p in scene["plates"]])
            assert not np.array_equal(output, _render(changed)), "rotated motif shortcut"


def _scatter_cards(grid, window, card_boxes, rng):
    """Move cards into arbitrary clear exterior pockets, retaining explicit gaps."""
    t, b, l, r = window
    bg = int(grid[0, 0])
    for i in rng.permutation(len(card_boxes)):
        if rng.random() > .6:
            continue
        a, z, c, d = card_boxes[int(i)]
        tile = grid[a:z+1, c:d+1].copy()
        side = tile.shape[0]
        grid[a:z+1, c:d+1] = bg
        positions = []
        for y in range(1, grid.shape[0]-side):
            for x in range(1, grid.shape[1]-side):
                if not (y+side < t or y > b+1 or x+side < l or x > r+1):
                    continue
                if np.all(grid[y-1:y+side+1, x-1:x+side+1] == bg):
                    positions.append((y, x))
        if positions:
            y, x = positions[int(rng.integers(len(positions)))]
        else:
            y, x = a, c
        grid[y:y+side, x:x+side] = tile
        card_boxes[int(i)] = (y, y+side-1, x, x+side-1)


def _make_pair(rng, n, k, demo):
    for attempt in range(480):
        geometry = _geometry(rng, n, k, fallback=attempt >= 360)
        if geometry is None:
            continue
        shape, window, boxes, card_boxes = geometry
        colors = list(map(int, rng.permutation(10)))
        bg, frame = colors[:2]
        fill = bg if demo == 0 or (demo < 0 and rng.random() < .4) else colors[2]
        sources = colors[3:3+k]
        spare = colors[3+k:]
        mapping = _mapping(rng, sources, spare, demo)
        masks = [_mask(rng, n, informative=demo >= 0 and i == 0) for i in range(k)]
        if demo == 0:
            for _ in range(40):
                masks[1] = _mask(rng, n)
                if len(_components(masks[1])) > 1:
                    break
            else:
                masks[1] = np.eye(n, dtype=bool)
        grid = np.full(shape, bg, dtype=np.int8)
        # Rectangle order is random and has no role in printing/cancellation.
        for i in rng.permutation(k):
            _draw_frame(grid, boxes[int(i)], sources[int(i)])
        t, b, l, r = window
        grid[t:b+1, l:r+1] = fill
        _draw_frame(grid, window, frame)
        for source, mask, box in zip(sources, masks, card_boxes):
            a, z, c, d = box
            _draw_frame(grid, box, source)
            grid[a+1:z, c+1:d] = np.where(mask, mapping[source], bg)
        _scatter_cards(grid, window, card_boxes, rng)
        # The expected output is rendered from complete sampled rectangles,
        # before parsing. Reader agreement checks recovery of those rectangles.
        scene = dict(window=window, fill=fill, n=n,
                     plates=[dict(source=s, box=box, mask=m, target=mapping[s])
                             for s, box, m in zip(sources, boxes, masks)])
        output = _render(scene)
        try:
            parsed = _read_input(grid)
            assert {p["source"]: p["box"] for p in parsed["plates"]} == dict(zip(sources, boxes))
            assert np.array_equal(_render(parsed), output)
            _evidence(parsed, output, demo >= 0)
            if demo == 0:
                _island_evidence(parsed)
        except AssertionError:
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded construction could not produce an evidenced scene")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _island_evidence(scene):
    """Two distinct islands of a legend mask must have exclusive output witnesses."""
    layers = _layers(scene)
    total = sum(layers)
    witnessed = False
    for i, plate in enumerate(scene["plates"]):
        islands = _components(plate["mask"])
        if len(islands) < 2:
            continue
        count = 0
        for island in islands:
            m = np.zeros_like(plate["mask"])
            for p in island:
                m[p] = True
            altered = dict(scene, plates=[dict(q, mask=m) if j == i else q
                                         for j, q in enumerate(scene["plates"])])
            count += bool(np.any(_layers(altered)[i] & (total == 1)))
        witnessed |= count >= 2
    assert witnessed, "disconnected legend islands must be visibly transferred"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Sampling shifts never change the rule or key interpretation.
    shift = int(rng.integers(2))  # 0: third plate; 1: a larger stencil tile
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            demo = index if split == "train" and index < 2 else -1
            if demo >= 0:
                k, n = 2, 3 if index == 0 else int(rng.integers(2, 4))
            elif shift == 0:
                k = int(rng.integers(1, 3)) if split == "train" else 3
                n = int(rng.integers(2, 5))
            else:
                k = int(rng.integers(1, 4))
                n = int(rng.integers(2, 4)) if split == "train" else 4
            for _ in range(12):
                pair = _make_pair(rng, n, k, demo)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate rejection failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            scene = _read_input(grid)
            expected = _render(scene)
            assert np.array_equal(expected, output), "incorrect registered stencil print"
            assert not np.array_equal(output, grid)
            _evidence(scene, output, split == "train" and index < 2)
            scenes[split].append(scene)
    # Both minimal-count demonstrations carry the diagnostic evidence.
    first, second = scenes["train"][:2]
    assert len(first["plates"]) == len(second["plates"]) == 2
    assert first["fill"] == first["background"]
    assert second["fill"] != second["background"]
    assert len({p["target"] for p in first["plates"]}) == 1
    _island_evidence(first)
    mapping = {p["source"]: p["target"] for p in second["plates"]}
    assert sum(t in mapping for t in mapping.values()) == 1, "single-pass chain demonstration"
    for p in second["plates"]:
        if p["target"] in mapping:
            recursive = second.copy()
            recursive["plates"] = [dict(q, target=mapping.get(q["target"], q["target"]))
                                   for q in second["plates"]]
            assert not np.array_equal(_render(second), _render(recursive))
    train_k = {len(s["plates"]) for s in scenes["train"]}
    train_n = {s["n"] for s in scenes["train"]}
    assert (max(train_k) <= 2 and all(len(s["plates"]) == 3 for s in scenes["test"])) or \
           (max(train_n) <= 3 and all(s["n"] == 4 for s in scenes["test"])), "missing documented test shift"


def self_test() -> dict:
    """Bounded 42-seed checks, diagnostic mutations, and structural diversity."""
    puzzles, input_shapes, output_shapes, structures = set(), set(), set(), set()
    sizes, counts, card_sides, backgrounds, fills, maps = set(), set(), set(), set(), set(), set()
    masks, nested, crossing, disconnected, triple_regions = set(), 0, 0, 0, 0
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        tr, te = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _read_input(pair["input"])
                (tr if split == "train" else te).append(a)
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                sizes.add(a["n"])
                counts.add(len(a["plates"]))
                backgrounds.add(a["background"])
                fills.add(a["fill"] == a["background"])
                structure = np.asarray(pair["input"])
                roles = {}
                structures.add((structure.shape, tuple(roles.setdefault(int(c), len(roles)) for c in structure.flat)))
                t, b, l, r = a["window"]
                for card in a["cards"].values():
                    y, z, x, d = card["box"]
                    card_sides.add("N" if z < t else "S" if y > b else "W" if d < l else "E")
                    masks.add(_key(card["mask"]))
                    disconnected += len(_components(card["mask"])) > 1
                boxes = [p["box"] for p in a["plates"]]
                for i, one in enumerate(boxes):
                    for two in boxes[i+1:]:
                        aa, zz, cc, dd = one
                        ee, ff, gg, hh = two
                        nested += (aa < ee < ff < zz and cc < gg < hh < dd) or \
                                  (ee < aa < zz < ff and gg < cc < dd < hh)
                        crossing += bool(_perimeter(one) & _perimeter(two))
                triple_regions += bool(np.any(sum(_layers(a, solid=True)) == 3))
                mapping = {p["source"]: p["target"] for p in a["plates"]}
                maps.add((len(mapping), sum(v in mapping for v in mapping.values()), len(set(mapping.values()))))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        shifts.add("count" if max(len(a["plates"]) for a in tr) < min(len(a["plates"]) for a in te) else "scale")
        if seed < 4:
            # Whole-puzzle color permutations preserve every visible role.
            permutation = np.random.default_rng(9900+seed).permutation(10)
            recolored = {s: [{k: permutation[np.asarray(v)] for k, v in p.items()} for p in puzzle[s]]
                         for s in ("train", "test")}
            validate_generated_puzzle(recolored)
            for p in recolored["test"]:
                assert np.array_equal(solve(p["input"]), p["output"])
            # A changed inactive/active stencil pixel must change the prediction
            # when its residue is used exclusively inside the window.
            p = puzzle["train"][1]
            a = _read_input(p["input"])
            layers, modified = _layers(a), False
            total = sum(layers)
            for plate, active in zip(a["plates"], layers):
                yy, xx = np.argwhere(active & (total == 1))[0]
                t, _, l, _ = a["window"]
                y, _, x, _ = plate["box"]
                dy, dx = (int(yy)+t-y) % a["n"], (int(xx)+l-x) % a["n"]
                card = a["cards"][plate["source"]]
                cy, _, cx, _ = card["box"]
                changed = p["input"].copy()
                changed[cy+1+dy, cx+1+dx] = a["background"]
                if card["mask"].sum() > 1:
                    assert not np.array_equal(solve(changed), p["output"])
                    modified = True
                    break
            assert modified
    assert len(puzzles) == 42 and len(structures) >= 190
    assert len(input_shapes) >= 20 and len(output_shapes) >= 25
    assert sizes == {2, 3, 4} and counts == {1, 2, 3}
    assert card_sides == {"N", "S", "E", "W"}
    assert backgrounds == set(range(10)) and fills == {False, True}
    assert len(masks) >= 80 and len(maps) >= 10
    assert nested > 0 and crossing > 50 and disconnected > 50 and triple_regions > 0
    assert shifts == {"count", "scale"}
    # Every legal explicit count combination is part of the bounded self-test.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(8100+10*nt+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    one, two = generate_puzzle(103, 2, 1), generate_puzzle(103, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
               for a, b in zip(one[s], two[s]) for k in ("input", "output"))
    one["test"][0]["output"][0, 0] = (int(one["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(one)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzles),
                normalized_structures=len(structures), input_shapes=len(input_shapes),
                output_shapes=len(output_shapes), masks=len(masks), nested_pairs=nested,
                crossing_pairs=crossing, disconnected_masks=disconnected,
                three_region_overlaps=triple_regions, shifts=sorted(shifts))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
